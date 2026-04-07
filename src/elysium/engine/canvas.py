"""Deterministic canvas executor.

Accepts an ActionChunk (or raw list of action dicts) and applies each action
sequentially to a NumPy image array using OpenCV. Sparse control-point trajectories
are interpolated into smooth Bezier curves before rendering.

Input:  np.ndarray (H, W, 3) float32 in [0, 1], RGB
Output: np.ndarray (H, W, 3) float32 in [0, 1], RGB (new array, input never mutated)
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from elysium.log import logger
from elysium.schemas.actions import (
    Action,
    ActionChunk,
    BrushAction,
    CloneStampAction,
    ColorAdjustAction,
    EraserAction,
    FillAction,
    GaussianBlurAction,
    NoopAction,
    PatternBrushAction,
    PencilAction,
    ScatterBrushAction,
    TextOverlayAction,
    parse_action,
)

__all__ = [
    "execute_chunk",
    "execute_action",
    "apply_color_adjust_rgb01",
    "brush_segment_mask",
    "brush_dab_mask",
    "draw_shape_mask",
]

_BEZIER_STEPS = 40


def apply_color_adjust_rgb01(
    canvas: np.ndarray,
    brightness: int,
    contrast: float,
    saturation: float,
    *,
    exposure: int = 0,
    highlights: int = 0,
    shadows: int = 0,
    hue_shift: int = 0,
    temperature: int = 0,
) -> np.ndarray:
    assert canvas.ndim == 3 and canvas.shape[2] == 3
    img = canvas.astype(np.float32).copy()

    if exposure != 0:
        img = img * (2.0 ** (exposure / 50.0))

    img = img + brightness / 255.0

    mean = float(img.mean())
    img = (img - mean) * contrast + mean

    if highlights != 0 or shadows != 0:
        lum = img.mean(axis=2, keepdims=True)
        if highlights != 0:
            mask_h = np.clip((lum - 0.5) * 2.0, 0.0, 1.0)
            img = img + (highlights / 200.0) * mask_h
        if shadows != 0:
            mask_s = np.clip(1.0 - lum / 0.5, 0.0, 1.0)
            img = img + (shadows / 200.0) * mask_s

    if temperature != 0:
        delta = temperature / 200.0
        img[:, :, 0] = img[:, :, 0] + delta
        img[:, :, 2] = img[:, :, 2] - delta

    img = np.clip(img, 0.0, 1.0)

    hsv = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

    if hue_shift != 0:
        hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + hue_shift // 2) % 180

    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0.0, 255.0)
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return rgb.astype(np.float32) / 255.0


def _bezier_points(control_pts: list[tuple[int, int]], steps: int = _BEZIER_STEPS) -> list[tuple[int, int]]:
    """Interpolate control points into a smooth Bezier curve.

    Uses de Casteljau's algorithm for arbitrary-degree Bezier curves.

    Args:
        control_pts: Sparse (x, y) control points.
        steps: Number of interpolated points to generate.

    Returns:
        Dense list of (x, y) integer positions along the curve.
    """
    if len(control_pts) == 1:
        return control_pts
    if len(control_pts) == 2:
        p0, p1 = np.array(control_pts[0], float), np.array(control_pts[1], float)
        ts = np.linspace(0, 1, steps)
        pts = [(int(round((1 - t) * p0[0] + t * p1[0])), int(round((1 - t) * p0[1] + t * p1[1]))) for t in ts]
        return pts

    pts = np.array(control_pts, dtype=float)
    ts = np.linspace(0, 1, steps)
    result: list[tuple[int, int]] = []
    for t in ts:
        temp = pts.copy()
        n = len(temp)
        for _ in range(n - 1):
            temp = (1 - t) * temp[:-1] + t * temp[1:]
        result.append((int(round(temp[0, 0])), int(round(temp[0, 1]))))
    return result


def _to_bgr_uint8(canvas: np.ndarray) -> np.ndarray:
    """Convert float32 RGB [0,1] -> uint8 BGR for OpenCV drawing."""
    return cv2.cvtColor((canvas * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _from_bgr_uint8(img: np.ndarray) -> np.ndarray:
    """Convert uint8 BGR -> float32 RGB [0,1]."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _stroke_coverage_mask(
    h: int,
    w: int,
    trajectory: list[list[int]],
    thickness: int,
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = [(p[0], p[1]) for p in trajectory]
    smooth = _bezier_points(pts)
    for i in range(len(smooth) - 1):
        cv2.line(mask, smooth[i], smooth[i + 1], 255, thickness, lineType=cv2.LINE_AA)
    if len(smooth) == 1:
        cv2.circle(mask, smooth[0], max(1, thickness // 2), 255, -1)
    return mask.astype(np.float32) / 255.0


def _composite_brush_coverage(base: np.ndarray, add: np.ndarray) -> None:
    base[:] = 1.0 - (1.0 - base) * (1.0 - add)


def _mesh_dist_sq_to_segment(
    px: np.ndarray,
    py: np.ndarray,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> np.ndarray:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab2 = abx * abx + aby * aby
    t = np.where(
        ab2 < 1e-12,
        0.0,
        np.clip((apx * abx + apy * aby) / ab2, 0.0, 1.0),
    )
    qx = ax + t * abx
    qy = ay + t * aby
    dx = px - qx
    dy = py - qy
    return dx * dx + dy * dy


def _brush_radial_weight(d: np.ndarray, radius: float, hardness: int) -> np.ndarray:
    if hardness <= 0:
        return np.zeros_like(d, dtype=np.float32)
    h_frac = hardness / 100.0
    r = radius + 1e-6
    t = d / r
    t = np.clip(t, 0.0, 1.0)
    if h_frac >= 0.9999:
        return np.where(d <= radius + 1e-6, np.float32(1.0), np.float32(0.0))
    core = r * h_frac
    inner = d <= core + 1e-6
    outer = (d > core) & (d <= radius + 1e-6)
    band = radius - core + 1e-8
    fall = np.clip((radius - d) / band, 0.0, 1.0)
    return np.where(inner, np.float32(1.0), np.where(outer, fall.astype(np.float32), np.float32(0.0)))


def brush_segment_mask(
    h: int,
    w: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    stroke_size: int,
    hardness: int,
) -> np.ndarray:
    if hardness <= 0:
        return np.zeros((h, w), dtype=np.float32)
    thickness = max(1, 2 * stroke_size)
    if hardness >= 100:
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.line(m, (x1, y1), (x2, y2), 255, thickness, lineType=cv2.LINE_AA)
        return m.astype(np.float32) / 255.0
    radius = float(max(1, stroke_size))
    ax, ay = float(x1), float(y1)
    bx, by = float(x2), float(y2)
    pad = int(np.ceil(radius)) + 2
    xmin = int(np.floor(min(ax, bx) - pad))
    xmax = int(np.ceil(max(ax, bx) + pad))
    ymin = int(np.floor(min(ay, by) - pad))
    ymax = int(np.ceil(max(ay, by) + pad))
    xmin = max(0, xmin)
    xmax = min(w, xmax + 1)
    ymin = max(0, ymin)
    ymax = min(h, ymax + 1)
    out = np.zeros((h, w), dtype=np.float32)
    if xmin >= xmax or ymin >= ymax:
        return out
    xs = np.arange(xmin, xmax, dtype=np.float64)
    ys = np.arange(ymin, ymax, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    px = xx + 0.5
    py = yy + 0.5
    dsq = _mesh_dist_sq_to_segment(px, py, ax, ay, bx, by)
    d = np.sqrt(dsq)
    wts = _brush_radial_weight(d, radius, hardness)
    out[ymin:ymax, xmin:xmax] = wts.astype(np.float32)
    return out


def brush_dab_mask(
    h: int,
    w: int,
    cx: int,
    cy: int,
    stroke_size: int,
    hardness: int,
) -> np.ndarray:
    if hardness <= 0:
        return np.zeros((h, w), dtype=np.float32)
    r_int = max(1, stroke_size)
    if hardness >= 100:
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(m, (cx, cy), r_int, 255, -1)
        return m.astype(np.float32) / 255.0
    radius = float(r_int)
    pad = int(np.ceil(radius)) + 2
    xmin = max(0, cx - pad)
    xmax = min(w, cx + pad + 1)
    ymin = max(0, cy - pad)
    ymax = min(h, cy + pad + 1)
    out = np.zeros((h, w), dtype=np.float32)
    if xmin >= xmax or ymin >= ymax:
        return out
    xs = np.arange(xmin, xmax, dtype=np.float64)
    ys = np.arange(ymin, ymax, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    px = xx + 0.5
    py = yy + 0.5
    d = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    wts = _brush_radial_weight(d, radius, hardness)
    out[ymin:ymax, xmin:xmax] = wts.astype(np.float32)
    return out


def _brush_stroke_coverage_mask(
    h: int,
    w: int,
    trajectory: list[list[int]],
    stroke_size: int,
    hardness: int,
) -> np.ndarray:
    if hardness <= 0:
        return np.zeros((h, w), dtype=np.float32)
    if hardness >= 100:
        thick = max(1, 2 * stroke_size)
        return _stroke_coverage_mask(h, w, trajectory, thick)
    pts = [(p[0], p[1]) for p in trajectory]
    smooth = _bezier_points(pts)
    mask = np.zeros((h, w), dtype=np.float32)
    if len(smooth) == 1:
        x0, y0 = int(smooth[0][0]), int(smooth[0][1])
        return brush_dab_mask(h, w, x0, y0, stroke_size, hardness)
    for i in range(len(smooth) - 1):
        x1, y1 = int(smooth[i][0]), int(smooth[i][1])
        x2, y2 = int(smooth[i + 1][0]), int(smooth[i + 1][1])
        seg = brush_segment_mask(h, w, x1, y1, x2, y2, stroke_size, hardness)
        _composite_brush_coverage(mask, seg)
    return mask


_HERSHEY_FONTS: dict[str, int] = {
    "simplex": cv2.FONT_HERSHEY_SIMPLEX,
    "duplex": cv2.FONT_HERSHEY_DUPLEX,
    "complex": cv2.FONT_HERSHEY_COMPLEX,
    "triplex": cv2.FONT_HERSHEY_TRIPLEX,
    "script": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
}


def _text_overlay_mask(
    h: int,
    w: int,
    text: str,
    position: tuple[int, int],
    font: int,
    font_size: float,
    thickness: int,
) -> np.ndarray:
    blank = np.zeros((h, w), dtype=np.uint8)
    cv2.putText(blank, text, position, font, font_size, 255, thickness, cv2.LINE_AA)
    return blank.astype(np.float32) / 255.0


def _polygon_points(cx: int, cy: int, r: int, n: int, angle_deg: float) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.radians(angle_deg)
    return np.array(
        [[int(cx + r * np.cos(a)), int(cy + r * np.sin(a))] for a in angles],
        dtype=np.int32,
    )


def _star_points(cx: int, cy: int, r_outer: int, r_inner: int, n: int, angle_deg: float) -> np.ndarray:
    pts = []
    for i in range(2 * n):
        r = r_outer if i % 2 == 0 else r_inner
        a = i * np.pi / n + np.radians(angle_deg)
        pts.append([int(cx + r * np.cos(a)), int(cy + r * np.sin(a))])
    return np.array(pts, dtype=np.int32)


def draw_shape_mask(
    h: int,
    w: int,
    cx: int,
    cy: int,
    size: int,
    angle_deg: float,
    shape: str,
    *,
    thickness: int = 1,
    length: int = 0,
) -> np.ndarray:
    """Return a float32 [0,1] mask with the given shape stamped at (cx, cy).

    Args:
        h, w: Canvas dimensions.
        cx, cy: Center of the stamp in pixel coords.
        size: Radius / half-size of the shape in pixels.
        angle_deg: Rotation of the shape in degrees.
        shape: One of "circle", "leaf", "star", "triangle", "dash".

    Returns:
        float32 mask of shape (h, w), values in [0, 1].
    """
    tmp = np.zeros((h, w), dtype=np.uint8)
    s = max(1, size)
    if shape == "circle":
        cv2.circle(tmp, (cx, cy), s, 255, -1, cv2.LINE_AA)
    elif shape == "leaf":
        cv2.ellipse(tmp, (cx, cy), (s, max(1, s // 3)), angle_deg, 0, 360, 255, -1, cv2.LINE_AA)
    elif shape == "triangle":
        cv2.fillPoly(tmp, [_polygon_points(cx, cy, s, 3, angle_deg)], 255)
    elif shape == "star":
        cv2.fillPoly(tmp, [_star_points(cx, cy, s, max(1, s // 2), 5, angle_deg)], 255)
    elif shape == "dash":
        half = max(1, length) if length > 0 else s
        rad = np.radians(angle_deg)
        x1, y1 = int(cx + np.cos(rad) * half), int(cy + np.sin(rad) * half)
        x2, y2 = int(cx - np.cos(rad) * half), int(cy - np.sin(rad) * half)
        cv2.line(tmp, (x1, y1), (x2, y2), 255, max(1, thickness), cv2.LINE_AA)
    return tmp.astype(np.float32) / 255.0


def _scatter_tangent_deg_along_path(path_pts: list[tuple[int, int]], i: int) -> float:
    if len(path_pts) < 2:
        return 0.0
    if i < len(path_pts) - 1:
        x, y = path_pts[i]
        x2, y2 = path_pts[i + 1]
        return float(np.degrees(np.arctan2(y2 - y, x2 - x)))
    x0, y0 = path_pts[i - 1]
    x, y = path_pts[i]
    return float(np.degrees(np.arctan2(y - y0, x - x0)))


def _execute_scatter_brush(canvas: np.ndarray, action: ScatterBrushAction) -> np.ndarray:
    h, w = canvas.shape[:2]
    coverage = np.zeros((h, w), dtype=np.float32)
    rng = np.random.default_rng(action.seed)
    pts = _bezier_points([(p[0], p[1]) for p in action.trajectory])
    sample_step = max(1, _BEZIER_STEPS // 10)
    sampled = pts[::sample_step]
    for si, (x, y) in enumerate(sampled):
        base_tangent = _scatter_tangent_deg_along_path(sampled, si)
        for _ in range(max(1, action.density)):
            if action.shape == "dash":
                base = float(action.base_angle) if action.base_angle >= 0 else base_tangent
                if action.angle_jitter <= 0:
                    angle = base
                else:
                    angle = base + float(rng.uniform(-float(action.angle_jitter), float(action.angle_jitter)))
            else:
                cap = max(1.0, min(360.0, float(action.angle_jitter)))
                if action.angle_jitter <= 0:
                    angle = float(rng.uniform(0.0, 360.0))
                else:
                    angle = float(rng.uniform(0.0, cap))
            scatter_r = action.scatter / 100.0 * action.size * 2.5
            dx = float(rng.uniform(-scatter_r, scatter_r))
            dy = float(rng.uniform(-scatter_r, scatter_r))
            scale = 1.0 + float(rng.uniform(-action.size_jitter / 100.0, action.size_jitter / 100.0))
            stamp_size = max(1, int(action.size * scale))
            stamp_cx = int(max(0, min(w - 1, x + dx)))
            stamp_cy = int(max(0, min(h - 1, y + dy)))
            stamp = draw_shape_mask(
                h,
                w,
                stamp_cx,
                stamp_cy,
                stamp_size,
                angle,
                action.shape,
                thickness=action.thickness,
                length=action.length,
            )
            _composite_brush_coverage(coverage, stamp)
    return _blend_rgba_on_rgb01(canvas.astype(np.float32).copy(), coverage, action.color_rgba)


def _execute_pattern_brush(canvas: np.ndarray, action: PatternBrushAction) -> np.ndarray:
    h, w = canvas.shape[:2]
    coverage = np.zeros((h, w), dtype=np.float32)
    rng = np.random.default_rng(0)
    pts = _bezier_points([(p[0], p[1]) for p in action.trajectory])
    stamp = draw_shape_mask(
        h,
        w,
        pts[0][0],
        pts[0][1],
        action.size,
        0.0,
        action.shape,
        thickness=action.thickness,
        length=action.length,
    )
    _composite_brush_coverage(coverage, stamp)
    dist_acc = 0.0
    dir_angle = 0.0
    for i in range(1, len(pts)):
        x1, y1 = pts[i - 1]
        x2, y2 = pts[i]
        seg_len = float(np.hypot(x2 - x1, y2 - y1))
        if seg_len > 0:
            dir_angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        dist_acc += seg_len
        while dist_acc >= action.spacing:
            t = max(0.0, min(1.0, 1.0 - (dist_acc - action.spacing) / seg_len)) if seg_len > 0 else 0.0
            sx = int(x1 + (x2 - x1) * t)
            sy = int(y1 + (y2 - y1) * t)
            jitter = float(rng.uniform(-action.angle_jitter, action.angle_jitter))
            stamp = draw_shape_mask(
                h,
                w,
                sx,
                sy,
                action.size,
                dir_angle + jitter,
                action.shape,
                thickness=action.thickness,
                length=action.length,
            )
            _composite_brush_coverage(coverage, stamp)
            dist_acc -= action.spacing
    return _blend_rgba_on_rgb01(canvas.astype(np.float32).copy(), coverage, action.color_rgba)


def _flood_fill_region_mask(img_bgr: np.ndarray, position: tuple[int, int]) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flags = cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    cv2.floodFill(
        img_bgr.copy(),
        flood_mask,
        position,
        (0, 0, 0),
        loDiff=(10, 10, 10),
        upDiff=(10, 10, 10),
        flags=flags,
    )
    return flood_mask[1 : h + 1, 1 : w + 1].astype(np.float32) / 255.0


def _blend_rgba_on_rgb01(rgb: np.ndarray, coverage: np.ndarray, rgba: tuple[int, int, int, int]) -> np.ndarray:
    r, g, b, a = rgba
    fr = np.float32(r) / 255.0
    fg = np.float32(g) / 255.0
    fb = np.float32(b) / 255.0
    alpha = coverage * (np.float32(a) / 255.0)
    a3 = alpha[..., np.newaxis]
    fg_rgb = np.array([fr, fg, fb], dtype=np.float32)
    return np.clip(rgb * (1.0 - a3) + fg_rgb * a3, 0.0, 1.0)


def execute_action(canvas: np.ndarray, action: Action, original: np.ndarray | None = None) -> np.ndarray:
    """Execute a single action on the canvas.

    Args:
        canvas: Current canvas as float32 RGB [0, 1], shape (H, W, 3).
        action: Typed Action to execute.
        original: Original unedited image for eraser restoration. Same shape as canvas.

    Returns:
        New canvas with action applied. Input canvas is never mutated.
    """
    assert canvas.ndim == 3 and canvas.shape[2] == 3, f"Expected (H,W,3) canvas, got {canvas.shape}"

    if isinstance(action, NoopAction):
        return canvas.copy()

    if isinstance(action, ColorAdjustAction):
        return apply_color_adjust_rgb01(canvas, action.brightness, action.contrast, action.saturation)

    h, w = canvas.shape[:2]
    base = canvas.astype(np.float32).copy()

    if isinstance(action, BrushAction):
        cov = _brush_stroke_coverage_mask(h, w, action.trajectory, action.stroke_size, action.hardness)
        return _blend_rgba_on_rgb01(base, cov, action.color_rgba)

    if isinstance(action, PencilAction):
        cov = _stroke_coverage_mask(h, w, action.trajectory, thickness=1)
        return _blend_rgba_on_rgb01(base, cov, action.color_rgba)

    if isinstance(action, EraserAction):
        img_bgr = _to_bgr_uint8(canvas)
        if original is None:
            logger.warning("Eraser called without original image; skipping")
            return canvas.copy()
        orig_bgr = _to_bgr_uint8(original)
        pts = [(p[0], p[1]) for p in action.trajectory]
        smooth = _bezier_points(pts)
        mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        eraser_thickness = max(1, 2 * action.stroke_size)
        for i in range(len(smooth) - 1):
            cv2.line(mask, smooth[i], smooth[i + 1], 255, eraser_thickness)
        if len(smooth) == 1:
            cv2.circle(mask, smooth[0], max(1, action.stroke_size), 255, -1)
        img_bgr[mask > 0] = orig_bgr[mask > 0]
        return _from_bgr_uint8(img_bgr)

    if isinstance(action, FillAction):
        img_bgr = _to_bgr_uint8(canvas)
        x, y = int(action.position[0]), int(action.position[1])
        hh, ww = img_bgr.shape[:2]
        x = max(0, min(x, ww - 1))
        y = max(0, min(y, hh - 1))
        region = _flood_fill_region_mask(img_bgr, (x, y))
        rgb_f = canvas.astype(np.float32).copy()
        rgb_f = _blend_rgba_on_rgb01(rgb_f, region, action.color_rgba)
        return rgb_f

    if isinstance(action, TextOverlayAction):
        font = _HERSHEY_FONTS[action.font_name]
        mask = _text_overlay_mask(
            h, w,
            action.text,
            (int(action.position[0]), int(action.position[1])),
            font,
            action.font_size,
            action.thickness,
        )
        return _blend_rgba_on_rgb01(base, mask, action.color_rgba)

    if isinstance(action, GaussianBlurAction):
        k = 2 * action.radius + 1
        img_uint8 = (canvas * 255.0).clip(0, 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(img_uint8, (k, k), 0)
        return blurred.astype(np.float32) / 255.0

    if isinstance(action, CloneStampAction):
        sx, sy = int(action.source[0]), int(action.source[1])
        dx, dy = int(action.destination[0]), int(action.destination[1])
        r = action.size
        src_patch = base.copy()
        offset_x = dx - sx
        offset_y = dy - sy
        shifted = np.roll(np.roll(src_patch, offset_y, axis=0), offset_x, axis=1)
        mask = brush_dab_mask(h, w, dx, dy, r, hardness=80)
        a3 = mask[..., np.newaxis]
        return np.clip(base * (1.0 - a3) + shifted * a3, 0.0, 1.0)

    if isinstance(action, ScatterBrushAction):
        return _execute_scatter_brush(canvas, action)

    if isinstance(action, PatternBrushAction):
        return _execute_pattern_brush(canvas, action)

    assert False, f"Unhandled action type {type(action)}"


def execute_chunk(
    canvas: np.ndarray,
    chunk: ActionChunk | list[dict[str, Any]],
    original: np.ndarray | None = None,
) -> np.ndarray:
    """Execute all actions in an ActionChunk sequentially.

    Args:
        canvas: Current canvas as float32 RGB [0, 1], shape (H, W, 3).
        chunk: ActionChunk or raw list of action dicts.
        original: Original unedited image for eraser restoration.

    Returns:
        Canvas after all actions in the chunk have been applied.
    """
    if isinstance(chunk, list):
        actions = [parse_action(a) for a in chunk]
    else:
        actions = chunk.actions

    current = canvas
    for action in actions:
        current = execute_action(current, action, original=original)
    return current
