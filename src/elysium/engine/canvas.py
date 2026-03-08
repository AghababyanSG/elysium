"""Deterministic canvas executor.

Accepts an ActionChunk (or raw list of action dicts) and applies each action
sequentially to a NumPy image array using OpenCV. Sparse control-point trajectories
are interpolated into smooth Bezier curves before rendering.

Input:  np.ndarray (H, W, 3) float32 in [0, 1], RGB
Output: np.ndarray (H, W, 3) float32 in [0, 1], RGB (new array, input never mutated)
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from elysium.schemas.actions import (
    Action,
    ActionChunk,
    BrushAction,
    EraserAction,
    FillAction,
    NoopAction,
    PencilAction,
    parse_action,
)

__all__ = ["execute_chunk", "execute_action"]

logger = logging.getLogger(__name__)

_BEZIER_STEPS = 40


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


def _color_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return (rgb[2], rgb[1], rgb[0])


def _draw_stroke(
    img_bgr: np.ndarray,
    trajectory: list[list[int]],
    color_bgr: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw a Bezier-interpolated stroke onto img_bgr in-place."""
    pts = [(p[0], p[1]) for p in trajectory]
    smooth = _bezier_points(pts)
    for i in range(len(smooth) - 1):
        cv2.line(img_bgr, smooth[i], smooth[i + 1], color_bgr, thickness, lineType=cv2.LINE_AA)
    if len(smooth) == 1:
        cv2.circle(img_bgr, smooth[0], max(1, thickness // 2), color_bgr, -1)


def _flood_fill(
    img_bgr: np.ndarray,
    position: tuple[int, int],
    color_bgr: tuple[int, int, int],
) -> np.ndarray:
    """Flood-fill from position with color. Returns modified copy."""
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    img_copy = img_bgr.copy()
    cv2.floodFill(img_copy, mask, position, color_bgr, loDiff=(10,) * 3, upDiff=(10,) * 3)
    return img_copy


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

    img_bgr = _to_bgr_uint8(canvas)

    if isinstance(action, BrushAction):
        _draw_stroke(img_bgr, action.trajectory, _color_bgr(action.color_rgb), action.stroke_size)

    elif isinstance(action, PencilAction):
        _draw_stroke(img_bgr, action.trajectory, _color_bgr(action.color_rgb), thickness=1)

    elif isinstance(action, EraserAction):
        if original is None:
            logger.warning("Eraser called without original image; skipping")
            return canvas.copy()
        orig_bgr = _to_bgr_uint8(original)
        pts = [(p[0], p[1]) for p in action.trajectory]
        smooth = _bezier_points(pts)
        mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        for i in range(len(smooth) - 1):
            cv2.line(mask, smooth[i], smooth[i + 1], 255, action.stroke_size)
        if len(smooth) == 1:
            cv2.circle(mask, smooth[0], action.stroke_size // 2, 255, -1)
        img_bgr[mask > 0] = orig_bgr[mask > 0]

    elif isinstance(action, FillAction):
        x, y = int(action.position[0]), int(action.position[1])
        h, w = img_bgr.shape[:2]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        img_bgr = _flood_fill(img_bgr, (x, y), _color_bgr(action.color_rgb))

    return _from_bgr_uint8(img_bgr)


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
