"""Pure, fit-specific image-plane calibration; no player code or image mutations."""
import numpy as np
from source_registration import INNER_UP, INNER_DOWN, OUTER_UP, OUTER_DOWN

LIPS = np.array(sorted(set(INNER_UP + INNER_DOWN + OUTER_UP + OUTER_DOWN)))
UP = np.array(INNER_UP[1:-1])
DOWN = np.array(INNER_DOWN[1:-1])
ANCHORS = np.array([33, 133, 362, 263, 168, 6, 197, 195, 5, 151, 10])


def align_to_reference(points, reference):
    """Least-squares similarity on stable upper face anchors only."""
    x, y = points[ANCHORS, :2], reference[ANCHORS, :2]
    xc, yc = x.mean(0), y.mean(0)
    x0, y0 = x - xc, y - yc
    u, sv, vt = np.linalg.svd(x0.T @ y0)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    scale = np.sum((x0 @ r) * y0) / np.sum(x0 * x0)
    out = points[:, :2] @ r * scale + yc - xc @ r * scale
    return out


def local_frame(reference):
    axis = reference[291, :2] - reference[61, :2]
    width = np.linalg.norm(axis)
    if width < 1e-6:
        raise ValueError('Degenerate mouth width')
    x = axis / width
    rot = np.array([x, [-x[1], x[0]]]).T
    center = (reference[291, :2] + reference[61, :2]) / 2
    return center, width, rot


def normalize(points, frame):
    center, width, rot = frame
    return (points[..., :2] - center) @ rot / width


def denormalize(points, frame):
    center, width, rot = frame
    return points @ rot.T * width + center


def gaps(points):
    return points[..., DOWN, 1] - points[..., UP, 1]


def mouth_metrics(points):
    g = gaps(points)
    return dict(median_gap=float(np.median(g)), max_gap=float(np.max(g)),
                min_gap=float(np.min(g)), width=float(np.linalg.norm(points[291]-points[61])))


def fit_mapping(reference, target_poses, coefficients, source_reference, prior, ridge=.03):
    """Fit residuals only in the measured coefficient subspace, via small SVD."""
    q = np.asarray(coefficients) - source_reference
    y = np.asarray(target_poses) - reference
    prediction = np.einsum('nck,vk->vnc', prior, q)
    residual = y - prediction
    u, singular, vt = np.linalg.svd(q, full_matrices=False)
    gain = singular / (singular * singular + ridge)
    correction = np.einsum('kv,vtc->tck', (vt.T * gain) @ u.T, residual)
    mapping = prior + correction
    supported = singular > 1e-3
    return mapping, dict(rank=int(supported.sum()), singular_values=singular.tolist(),
                         condition=float(singular[0]/singular[supported][-1]) if supported.any() else None,
                         ridge=ridge)


def evaluate(reference, mapping, coefficients, source_reference):
    return reference + np.einsum('nck,...k->...nc', mapping,
                                 np.asarray(coefficients)-source_reference)


def contact_correct(points, target_contact, strength):
    """Bounded vertical inner-lip correction with smooth outer-lip follow.

    This is a candidate constraint, not proof of full renderer mouth safety.
    It preserves x and lip rim thickness by translating each side's outer rim.
    """
    out = points.copy()
    before = gaps(points)
    desired = np.clip(gaps(target_contact), .001, .008)
    shift = (desired - before) * np.clip(strength, 0, 1) / 2
    shift = np.clip(shift, -.15, .15)
    out[UP, 1] -= shift
    out[DOWN, 1] += shift
    # Each of the two contours has nine corresponding non-corner samples.
    for j, idx in enumerate(OUTER_UP[1:-1]):
        out[idx, 1] -= shift[j]
    for j, idx in enumerate(OUTER_DOWN[1:-1]):
        out[idx, 1] += shift[j]
    return out, float(np.max(np.abs(shift)))


def lip_error(predicted, expected):
    return float(np.sqrt(np.mean(np.sum((predicted[LIPS]-expected[LIPS])**2, axis=1))))
