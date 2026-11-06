"""Project source-image detail onto the visible surface of a LiTo splat."""

import argparse
import math
from pathlib import Path

import cv2
import gsplat_mlx.rendering as renderer
import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps
from plyfile import PlyData, PlyElement
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation

SIZE = 1024
FOCAL = SIZE / (2 * math.tan(math.radians(42) / 2))
FRONT = np.array([1, 0, 0], dtype=np.float32)
# LiTo places the source-facing view on +X with +Z up.
CAMERA_ROTATION = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], dtype=np.float32)


class UnsuitableImage(Exception):
    """The input does not provide a reliable source detail projection."""


composite = mx.fast.metal_kernel(
    name="lito_surface_projection",
    input_names=["means", "conics", "colors", "opacities", "offsets", "ids", "info"],
    output_names=["rgb", "alpha_out"],
    source="""
    uint pixel = thread_position_in_grid.x;
    uint width = info[0], height = info[1], tile = info[2];
    if (pixel >= width * height) return;
    uint x = pixel % width, y = pixel / width;
    uint tiles_x = (width + tile - 1) / tile;
    uint tiles_y = (height + tile - 1) / tile;
    uint tile_index = (y / tile) * tiles_x + x / tile;
    int start = offsets[tile_index];
    int end = tile_index + 1 < tiles_x * tiles_y ? offsets[tile_index + 1] : info[3];
    float remaining = 1.0f;
    float3 color = float3(0.0f);
    float surface = 0.0f;
    for (int i = start; i < end; ++i) {
        uint g = ids[i];
        float dx = float(x) + 0.5f - means[g * 2];
        float dy = float(y) + 0.5f - means[g * 2 + 1];
        float power = 0.5f * (conics[g * 3] * dx * dx + conics[g * 3 + 2] * dy * dy)
            + conics[g * 3 + 1] * dx * dy;
        if (power < 0.0f) continue;
        float a = min(opacities[g] * exp(-power), 0.999f);
        if (a < 1.0f / 255.0f) continue;
        float3 c = float3(colors[g * 4], colors[g * 4 + 1], colors[g * 4 + 2]);
        color += (remaining * a) * c;
        remaining *= 1.0f - a;
        // Take the first depth to reach 30% opacity so faint floaters are skipped.
        if (surface == 0.0f && remaining < 0.70f) surface = colors[g * 4 + 3];
        if (remaining < 0.0001f) break;
    }
    color += remaining * 0.5f;
    rgb[pixel * 4] = color.x;
    rgb[pixel * 4 + 1] = color.y;
    rgb[pixel * 4 + 2] = color.z;
    rgb[pixel * 4 + 3] = surface;
    alpha_out[pixel] = 1.0f - remaining;
    """,
)


def depth_raster(means, conics, colors, opacities, width, height, tile, offsets, ids, **_):
    return composite(
        inputs=[means, conics, colors, opacities, offsets, ids, mx.array([width, height, tile, ids.size])],
        output_shapes=[(1, height, width, 4), (1, height, width, 1)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(width * height, 1, 1),
        threadgroup=(128, 1, 1),
    )


def render_front(vertices):
    def fields(names):
        return np.column_stack([vertices[name] for name in names]).astype(np.float32)

    xyz = fields(("x", "y", "z"))
    dc = fields([f"f_dc_{i}" for i in range(3)])[:, None]
    rest_names = sorted(
        (name for name in vertices.dtype.names if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    rest = fields(rest_names).reshape(len(vertices), 3, -1).transpose(0, 2, 1) if rest_names else dc[:, :0]
    sh = np.concatenate((dc, rest), axis=1)
    degree = math.isqrt(sh.shape[1]) - 1
    if (degree + 1) ** 2 != sh.shape[1]:
        raise ValueError("Invalid spherical harmonic coefficients in LiTo output.")

    visible = xyz[vertices["opacity"] > -2]
    lo, hi = np.quantile(visible if len(visible) else xyz, [0.01, 0.99], axis=0)
    radius = 1.5 * float(max(hi - lo))
    if radius <= 0:
        raise UnsuitableImage("The generated splat has no measurable surface.")
    eye = (lo + hi) / 2 + FRONT * radius
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = CAMERA_ROTATION
    view[:3, 3] = -CAMERA_ROTATION @ eye
    intrinsic = mx.array([[[FOCAL, 0, SIZE / 2], [0, FOCAL, SIZE / 2], [0, 0, 1]]], dtype=mx.float32)

    # This standalone process substitutes only the final pixel compositor.
    renderer.rasterize_to_pixels = depth_raster
    rgbd, opacity, _ = renderer.rasterization(
        mx.array(xyz), mx.array(fields([f"rot_{i}" for i in range(4)])),
        mx.exp(mx.array(fields([f"scale_{i}" for i in range(3)]))),
        mx.sigmoid(mx.array(vertices["opacity"].astype(np.float32))), mx.array(sh),
        mx.array(view[None]), intrinsic, width=SIZE, height=SIZE, tile_size=16,
        sh_degree=degree, render_mode="RGB+D", differentiable=False,
    )
    rgbd = np.array(rgbd[0])
    return rgbd[:, :, :3].clip(0, 1), rgbd[:, :, 3], np.array(opacity[0, :, :, 0]), view


def bbox(mask):
    rows, cols = np.where(mask)
    if not rows.size or np.ptp(cols) < 2 or np.ptp(rows) < 2:
        raise UnsuitableImage("The source or splat has no usable foreground.")
    return np.array([cols.min(), rows.min()]), np.array([cols.max(), rows.max()])


def load_source(image):
    with Image.open(image) as opened:
        source = ImageOps.exif_transpose(opened).convert("RGBA")
    source.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    return source


def prepare_source(image, output):
    source = load_source(image)
    rgba = np.asarray(source).copy()
    if np.any(rgba[:, :, 3] < 26):
        return 4  # The original already has usable transparency.

    # Only key a consistent, saturated green or blue screen that reaches the border.
    rgb = rgba[:, :, :3]
    height, width = rgb.shape[:2]
    patch = max(1, min(height, width) // 32)
    corners = np.stack((rgb[:patch, :patch], rgb[:patch, -patch:],
                        rgb[-patch:, :patch], rgb[-patch:, -patch:]))
    corner_colors = np.median(corners, axis=(1, 2))
    for channel in (1, 2):
        others = [i for i in range(3) if i != channel]
        screen = corner_colors[:, channel] - corner_colors[:, others].max(axis=1) > 48
        if screen.sum() >= 3:
            key = np.median(corner_colors[screen], axis=0)
            break
    else:
        return 3

    delta = np.max(np.abs(rgb.astype(np.int16) - key.astype(np.int16)), axis=2)
    candidates = delta <= 42
    border = np.concatenate((candidates[0], candidates[-1], candidates[:, 0], candidates[:, -1]))
    if border.mean() < 0.55:
        return 3
    _, labels = cv2.connectedComponents(candidates.astype(np.uint8), connectivity=4)
    border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    background = np.isin(labels, border_labels[border_labels != 0])
    if not 0.05 <= background.mean() <= 0.95:
        return 3

    # Despill the narrow edge that mixes subject pixels with the screen color.
    near_background = cv2.dilate(background.astype(np.uint8), np.ones((7, 7), dtype=np.uint8)).astype(bool)
    edge = near_background & ~background
    rgb[edge, channel] = np.minimum(rgb[edge, channel], rgb[:, :, others].max(axis=2)[edge])
    rgba[:, :, 3] = cv2.GaussianBlur((~background).astype(np.uint8) * 255, (3, 3), 0.6)
    Image.fromarray(rgba).save(output)
    print(f"Keyed {'green' if channel == 1 else 'blue'} screen from {background.mean():.1%} of the image.")
    return 0


def project_detail(vertices, image):
    rgba = np.asarray(load_source(image), dtype=np.float32) / 255
    src_alpha = rgba[:, :, 3]
    if not np.any(src_alpha < 0.1):
        raise UnsuitableImage("Source detail requires a transparent background.")
    src_lo, src_hi = bbox(src_alpha > 0.7)
    src_rgb = rgba[:, :, :3] * src_alpha[:, :, None] + 0.5 * (1 - src_alpha[:, :, None])
    base, depth, alpha, view = render_front(vertices)
    lo, hi = bbox(alpha > 0.6)
    scale = (hi - lo) / (src_hi - src_lo)
    offset = lo - src_lo * scale
    affine = np.column_stack((np.diag(scale), offset)).astype(np.float32)
    aligned = cv2.warpAffine(src_rgb, affine, (SIZE, SIZE), flags=cv2.INTER_CUBIC, borderValue=(0.5,) * 3).clip(0, 1)
    aligned_alpha = cv2.warpAffine(src_alpha, affine, (SIZE, SIZE))
    base_gray = cv2.cvtColor((base * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    source_gray = cv2.cvtColor((aligned * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow.setFinestScale(1)
    forward = flow.calc(base_gray, source_gray, None)
    reverse = flow.calc(source_gray, base_gray, None)
    y, x = np.indices((SIZE, SIZE), dtype=np.float32)
    u, v = x + forward[:, :, 0], y + forward[:, :, 1]

    def sample(values, interpolation=cv2.INTER_LINEAR):
        return cv2.remap(values, u, v, interpolation)

    warped = sample(aligned, cv2.INTER_CUBIC).clip(0, 1)
    warped_alpha = sample(aligned_alpha)
    cycle = np.linalg.norm(forward + sample(reverse), axis=-1)
    valid = (alpha > 0.98) & (warped_alpha > 0.98) & (depth > 0)
    if not valid.any():
        raise UnsuitableImage("The source image and splat do not overlap reliably.")
    edge_distance = distance_transform_edt(valid)
    confidence = np.clip((edge_distance - 5) / 12, 0, 1) * np.exp(-np.square(cycle / 5))
    confidence *= np.clip((70 - np.linalg.norm(forward, axis=-1)) / 40, 0, 1)

    filled = depth.copy()
    indices = distance_transform_edt(depth <= 0, return_distances=False, return_indices=True)
    filled[depth <= 0] = depth[tuple(indices[:, depth <= 0])]
    smooth = cv2.bilateralFilter(filled, 9, 0.014, 4)
    z = smooth - 0.008
    points_cam = np.stack(((x + 0.5 - SIZE / 2) * z / FOCAL, (y + 0.5 - SIZE / 2) * z / FOCAL, z), axis=-1)
    world = (points_cam - view[:3, 3]) @ view[:3, :3]
    du = np.gradient(world, axis=1)
    dv = np.gradient(world, axis=0)
    normals = np.cross(du, dv)
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-7)
    frontness = np.abs(normals @ FRONT)
    confidence *= np.clip((frontness - 0.35) / 0.4, 0, 1)
    confidence *= np.exp(-np.square((smooth - filled) / 0.018))

    # Keep base low frequencies and transfer source color/detail in well-aligned regions.
    blur = cv2.GaussianBlur(warped, (0, 0), 4)
    detail = warped - blur
    delta = warped - base
    strength = np.clip(np.max(np.abs(delta), axis=-1) * 8, 0, 1)
    weights = 0.8 * confidence * strength
    mask = weights > 0.10
    if not mask.any():
        raise UnsuitableImage("No visible surface matched the source image reliably.")
    target = (base + 0.60 * np.clip(delta, -0.30, 0.30) + 0.55 * detail).clip(0, 1)
    uu = du[mask]
    vv = dv[mask]
    tangent = uu / np.maximum(np.linalg.norm(uu, axis=-1, keepdims=True), 1e-7)
    normal = normals[mask]
    bitangent = np.cross(normal, tangent)
    rotation = Rotation.from_matrix(np.stack((tangent, bitangent, normal), axis=-1)).as_quat(scalar_first=True)
    footprint = z[mask] / FOCAL
    scales = np.column_stack((
        np.minimum(np.linalg.norm(uu, axis=-1), 2.2 * footprint) * 0.63,
        np.minimum(np.linalg.norm(vv, axis=-1), 2.2 * footprint) * 0.63,
        footprint * 0.28,
    ))
    extra = np.zeros(np.count_nonzero(mask), dtype=vertices.dtype)
    for i, key in enumerate(("x", "y", "z")):
        extra[key] = world[mask, i]
    for i in range(3):
        extra[f"scale_{i}"] = np.log(np.maximum(scales[:, i], 1e-7))
        extra[f"f_dc_{i}"] = (target[mask, i] - 0.5) / 0.28209479177387814
    for i in range(4):
        extra[f"rot_{i}"] = rotation[:, i]
    extra["opacity"] = np.log(weights[mask] / (1 - weights[mask]))
    return extra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("image", type=Path)
    prepare.add_argument("output", type=Path)
    project = commands.add_parser("project")
    project.add_argument("ply", type=Path)
    project.add_argument("image", type=Path)
    project.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        return prepare_source(args.image, args.output)

    vertices = PlyData.read(args.ply)["vertex"].data
    if not len(vertices) or not all(np.isfinite(vertices[name]).all() for name in vertices.dtype.names):
        raise ValueError("LiTo generated an empty or invalid splat.")
    try:
        extra = project_detail(vertices, args.image)
    except UnsuitableImage as error:
        print(error)
        return 3
    PlyData([PlyElement.describe(np.concatenate((vertices, extra)), "vertex")], text=False).write(args.output)
    print(f"Added {len(extra):,} detail Gaussians to {len(vertices):,} originals.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
