"""Offline source-rig projection and reproducible landmark embedding for the probe."""
from pathlib import Path
import json
import cv2
import numpy as np
from scipy.spatial import cKDTree

LIP_PAIRS = np.array([[9013, 10613], [941, 10433], [8859, 10434],
                      [2963, 3726], [1439, 1475], [5980, 7505],
                      [6093, 18810], [263, 18809], [17362, 13358]])
INNER_UP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
INNER_DOWN = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
OUTER_UP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
OUTER_DOWN = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]


def source_contact_mask(reg,weights):
    """Keep source contact labels independent of target landmark registration."""
    upper,lower=LIP_PAIRS.T
    n=reg['source_neutral'];b=reg['source_basis']
    baseline=n[upper,1]-n[lower,1]
    basis=b[upper,1,:]-b[lower,1,:]
    return np.median(baseline+np.asarray(weights)@basis.T,axis=-1)<=.1


def load_rig(assets, sandbox_assets):
    skin = np.load(assets / 'bs_skin_James.npz')
    names = [x.decode() for x in skin['poseNames'][1:]]
    neutral = skin['neutral'].astype(float)
    basis = np.stack([skin[n] for n in names], axis=-1).astype(float)
    meta = json.loads((sandbox_assets / 'rig.json').read_text())
    binary = (sandbox_assets / 'rig.bin').read_bytes()
    skin_meta = meta['meshes']['skin']
    triangles = np.frombuffer(binary, '<u4', skin_meta['indexCount'],
                              skin_meta['indexOffset']).reshape(-1, 3).copy()
    return names, neutral, basis, triangles, meta, binary


def projection(points, size=900):
    # Orthographic front view; source y is up and z increases toward viewer.
    xy = points[..., :2].copy()
    xy[..., 0] = size / 2 + xy[..., 0] * size / 23
    xy[..., 1] = size / 2 - (xy[..., 1] - 172) * size / 23
    return xy


def render_source(points, triangles, path, size=900):
    p = projection(points, size)
    faces = points[triangles]
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
    # Smooth vertex normals keep the synthetic face detectable without textures.
    vn = np.zeros_like(points)
    for j in range(3):
        np.add.at(vn, triangles[:, j], normals)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-10)
    face_normals = vn[triangles].mean(1)
    light = np.array([-.3, .45, 1.]); light /= np.linalg.norm(light)
    shade = .32 + .68 * np.maximum(0, face_normals @ light)
    image = np.full((size, size, 3), 35, np.uint8)
    order = np.argsort(faces[:, :, 2].mean(1))
    for i in order:
        poly = np.rint(p[triangles[i]]).astype(np.int32)
        if poly[:, 0].max() < 0 or poly[:, 0].min() >= size or poly[:, 1].max() < 0 or poly[:, 1].min() >= size:
            continue
        color = tuple(int(c * shade[i]) for c in (170, 188, 210))
        cv2.fillConvexPoly(image, poly, color, lineType=cv2.LINE_AA)
    cv2.imwrite(str(path), image)
    return image


def detect(image, task):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    with FaceLandmarker.create_from_options(FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(task)), num_faces=1,
            min_face_detection_confidence=.25, min_face_presence_confidence=.25)) as detector:
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    if len(result.face_landmarks) != 1:
        raise ValueError('Source projection did not yield one detectable face')
    return np.array([[p.x * image.shape[1], p.y * image.shape[0]]
                     for p in result.face_landmarks[0]])


def embed(projected, points, triangles, landmarks):
    """Choose the frontmost containing triangle, not an unconstrained closest vertex."""
    xy = projected[triangles]
    a, b = xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]
    det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    safe = np.abs(det) > 1e-7
    triangle_ids, bary, errors = [], [], []
    near = cKDTree(projected)
    for p in landmarks:
        c = p - xy[:, 0]
        u = np.divide(c[:, 0] * b[:, 1] - c[:, 1] * b[:, 0], det,
                      out=np.full(len(det), np.nan), where=safe)
        v = np.divide(a[:, 0] * c[:, 1] - a[:, 1] * c[:, 0], det,
                      out=np.full(len(det), np.nan), where=safe)
        candidates = np.flatnonzero(safe & (u >= -1e-5) & (v >= -1e-5) & (u + v <= 1.00001))
        if not len(candidates):
            distance, vertex = near.query(p)
            t = np.flatnonzero(np.any(triangles == vertex, axis=1))[0]
            weights = np.zeros(3); weights[np.where(triangles[t] == vertex)[0][0]] = 1
            errors.append(float(distance))
        else:
            w = np.column_stack([1-u[candidates]-v[candidates], u[candidates], v[candidates]])
            z = np.sum(points[triangles[candidates], 2] * w, axis=1)
            k = np.argmax(z); t = candidates[k]; weights = w[k]
            errors.append(0.)
        triangle_ids.append(t); bary.append(weights)
    return np.array(triangle_ids), np.array(bary), errors


def build_registration(assets, sandbox_assets, task, out, curve_registration=False):
    out.mkdir(parents=True, exist_ok=True)
    names, neutral, basis, triangles, _, _ = load_rig(assets, sandbox_assets)
    pose = neutral + .35 * basis[..., names.index('jawOpen')]
    image = render_source(pose, triangles, out / 'source-registration.png')
    lm = detect(image, task)
    ids, bary, errors = embed(projection(pose), pose, triangles, lm)
    base = np.einsum('ij,ijk->ik', bary, neutral[triangles[ids]])
    delta = np.einsum('ij,ijkl->ikl', bary, basis[triangles[ids]])
    # The source rig provides nine verified lip pairs. Assign them in image order
    # to distinct upper/lower inner contours, avoiding cross-lip surface matches.
    order = np.argsort(neutral[LIP_PAIRS[:, 0], 0])
    pairs = LIP_PAIRS[order]
    if curve_registration:
        # Nine source gap anchors are central samples, not homologous MediaPipe
        # vertices. Resample their source curves at the detected anatomical x
        # locations rather than assigning those samples one-to-one by index.
        # All weights are fixed after registration and applied to every target.
        for contour,vertices in [(INNER_UP,pairs[:,0]),(INNER_DOWN,pairs[:,1])]:
            bn=np.vstack([base[78],neutral[vertices],base[308]])
            bd=np.concatenate([delta[78][None],basis[vertices],delta[308][None]])
            opened=bn+.35*bd[:,:,names.index('jawOpen')]
            px=projection(opened)[:,0]
            if np.any(np.diff(px)<=0):raise ValueError('Source lip guide x order is not monotone')
            for index in contour[1:-1]:
                x=float(np.clip(lm[index,0],px[0],px[-1]))
                left=min(max(int(np.searchsorted(px,x))-1,0),len(px)-2)
                t=(x-px[left])/(px[left+1]-px[left])
                base[index]=bn[left]*(1-t)+bn[left+1]*t
                delta[index]=bd[left]*(1-t)+bd[left+1]*t
    else:
        for contour, vertices in [(INNER_UP[1:-1], pairs[:, 0]), (INNER_DOWN[1:-1], pairs[:, 1])]:
            base[contour] = neutral[vertices]
            delta[contour] = basis[vertices]
    # Keep the detected true inner corners. The nine verified pairs lie inside
    # those corners and must not shorten the mouth by replacing its endpoints.
    np.savez_compressed(out / 'source-registration.npz', neutral=base, basis=delta,
                        source_neutral=neutral, source_basis=basis,
                        source_triangles=triangles, triangle_ids=ids, barycentric=bary,
                        channels=np.array(names), detected_xy=lm)
    overlay = image.copy()
    loc = projection(base + .35 * delta[..., names.index('jawOpen')])
    for contour, color in [(OUTER_UP, (0,255,255)), (OUTER_DOWN,(0,255,255)),
                            (INNER_UP,(0,255,0)), (INNER_DOWN,(255,100,0))]:
        cv2.polylines(overlay, [np.rint(loc[contour]).astype(np.int32)], False, color, 2)
    cv2.imwrite(str(out / 'source-registration-overlay.png'), overlay)
    return dict(max_projected_embedding_error_px=max(errors),
                approximate_registration=True,
                inner_lips='source lip guide curves resampled at anatomical image positions' if curve_registration
                    else 'nine source rig lip pairs; fixed vertex correspondence',
                other_landmarks='automatic front projection barycentric embedding; visual review required')
