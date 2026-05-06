"""Trajectory compression for raw Pygame annotation logs.

Raw logs record every mouse event at ~30fps, producing hundreds of near-duplicate
(start_pos, end_pos) pairs per stroke. This module:
  1. Groups consecutive same-tool operations into strokes.
  2. Applies Ramer-Douglas-Peucker (RDP) to compress each stroke's coordinate
     sequence into sparse control points.

Input:  data/raw/sessions/{session_name}.json
Output: data/interim/compressed/{session_name}.json
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from rdp import rdp

from elysium.schemas.actions import CANVAS_SIZE

__all__ = ["compress_session", "compress_all"]

logger = logging.getLogger(__name__)

# Tools that are grouped into strokes (multi-point trajectories).
_STROKE_TOOLS = {"brush", "pencil", "eraser", "fill"}

# Tools logged as a single complete operation (already have all data in one entry).
_ATOMIC_TOOLS = {"scatter_brush", "pattern_brush", "forward_warp", "text_overlay"}

# Tools logged per-dab (one entry per mouse position; treated atomically here).
_DAB_TOOLS = {"gaussian_blur", "clone_stamp"}


def _finalize_stroke(
    current: dict[str, Any],
    last_draw_op: dict[str, Any],
) -> dict[str, Any]:
    current["trajectory"].append(
        last_draw_op.get("end_pos", last_draw_op["start_pos"]),
    )
    return {k: v for k, v in current.items() if k != "last_ts"}


def _operations_to_strokes(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    strokes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    last_draw_op: dict[str, Any] | None = None

    for op in operations:
        tool = op.get("tool")

        # --- atomic / single-entry tools ---
        if tool == "color_adjust":
            if current is not None and last_draw_op is not None:
                strokes.append(_finalize_stroke(current, last_draw_op))
                current = None
                last_draw_op = None
            strokes.append({
                "tool": "color_adjust",
                "brightness": op.get("brightness", 0),
                "exposure": op.get("exposure", 0),
                "contrast": op.get("contrast", 1.0),
                "highlights": op.get("highlights", 0),
                "shadows": op.get("shadows", 0),
                "saturation": op.get("saturation", 1.0),
                "hue_shift": op.get("hue_shift", 0),
                "temperature": op.get("temperature", 0),
                "frame_id": op["frame_id"],
            })
            continue

        if tool in _ATOMIC_TOOLS:
            if current is not None and last_draw_op is not None:
                strokes.append(_finalize_stroke(current, last_draw_op))
                current = None
                last_draw_op = None
            strokes.append(dict(op))
            continue

        if tool in _DAB_TOOLS:
            if current is not None and last_draw_op is not None:
                strokes.append(_finalize_stroke(current, last_draw_op))
                current = None
                last_draw_op = None
            strokes.append(dict(op))
            continue

        if tool in ("undo", "redo", "picker", "clear"):
            continue

        if tool not in _STROKE_TOOLS:
            continue

        # --- stroke-grouping for brush / pencil / eraser / fill ---
        size = op["size"]
        color = op.get("color")
        hardness = op.get("hardness", 100)
        point = op["start_pos"]
        ts = op["timestamp"]

        same_tool = current is not None and current["tool"] == tool
        same_color = current is not None and current.get("color") == color
        same_size = current is not None and current["size"] == size
        same_hardness = current is not None and current.get("hardness", 100) == hardness
        no_gap = current is not None and (ts - current["last_ts"]) < 1.0

        if same_tool and same_color and same_size and same_hardness and no_gap:
            current["trajectory"].append(point)
            current["last_ts"] = ts
            last_draw_op = op
        else:
            if current is not None and last_draw_op is not None:
                strokes.append(_finalize_stroke(current, last_draw_op))
            current = {
                "tool": tool,
                "size": size,
                "color": color,
                "hardness": hardness,
                "frame_id": op["frame_id"],
                "trajectory": [point],
                "last_ts": ts,
            }
            last_draw_op = op

    if current is not None and last_draw_op is not None:
        strokes.append(_finalize_stroke(current, last_draw_op))

    return strokes


def _rescale_operation(op: dict[str, Any], scale: float) -> dict[str, Any]:
    """Return a copy of op with spatial fields scaled by `scale`.

    Touches start_pos / end_pos / size / trajectory — these are pixel-unit
    fields that come from the raw annotator log. Other fields (timestamp,
    color, percent-valued knobs) are left alone.
    """
    out = dict(op)
    for key in ("start_pos", "end_pos"):
        v = out.get(key)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            out[key] = [int(round(v[0] * scale)), int(round(v[1] * scale))]
    sz = out.get("size")
    if isinstance(sz, (int, float)):
        out["size"] = max(1, int(round(sz * scale)))
    traj = out.get("trajectory")
    if isinstance(traj, list):
        out["trajectory"] = [
            [int(round(p[0] * scale)), int(round(p[1] * scale))] for p in traj
        ]
    return out


def _to_rgba(color: Sequence[int]) -> list[int]:
    if len(color) == 3:
        return [int(color[0]), int(color[1]), int(color[2]), 255]
    assert len(color) == 4, f"Unexpected color shape: {color!r}"
    return [int(c) for c in color]


_MAX_SEGMENTS_PER_STROKE = 64


def _compress_trajectory(trajectory: list[list[int]], epsilon: float) -> list[list[int]]:
    """Apply RDP to a trajectory, preserving at least 2 points.

    Args:
        trajectory: List of [x, y] coordinate pairs.
        epsilon: RDP tolerance in pixels. Higher = more compression.

    Returns:
        Compressed list of [x, y] control points, capped at _MAX_SEGMENTS_PER_STROKE+1.
    """
    cap = _MAX_SEGMENTS_PER_STROKE + 1
    if len(trajectory) <= 2:
        return trajectory[:cap]
    pts = np.array(trajectory, dtype=float)
    mask = rdp(pts, epsilon=epsilon, return_mask=True)
    compressed = pts[mask].astype(int).tolist()
    if len(compressed) < 2:
        compressed = [trajectory[0], trajectory[-1]]
    if len(compressed) <= cap:
        return compressed
    indices = np.linspace(0, len(compressed) - 1, cap, dtype=int)
    subsampled = [compressed[i] for i in indices]
    subsampled[0] = compressed[0]
    subsampled[-1] = compressed[-1]
    return subsampled


def _segment_actions_from_trajectory(
    base: dict[str, Any],
    trajectory: list[list[int]],
) -> list[dict[str, Any]]:
    """Turn a (compressed) polyline into one action per consecutive segment.

    Each emitted action is `base` plus start_point/end_point keys. Strokes with a
    single point produce a degenerate self-segment so the dab still renders.
    """
    if not trajectory:
        return []
    if len(trajectory) == 1:
        p = trajectory[0]
        return [{**base, "start_point": [int(p[0]), int(p[1])], "end_point": [int(p[0]), int(p[1])]}]
    out: list[dict[str, Any]] = []
    for i in range(len(trajectory) - 1):
        a, b = trajectory[i], trajectory[i + 1]
        out.append(
            {**base, "start_point": [int(a[0]), int(a[1])], "end_point": [int(b[0]), int(b[1])]}
        )
    return out


def compress_session(session_path: Path, output_path: Path, epsilon: float = 2.0) -> list[dict[str, Any]]:
    """Compress a single session log into a list of compressed strokes.

    Args:
        session_path: Path to raw session JSON.
        output_path: Path to write compressed JSON.
        epsilon: RDP tolerance in pixels.

    Returns:
        List of compressed stroke dicts.
    """
    with session_path.open() as f:
        session = json.load(f)

    operations: list[dict[str, Any]] = session.get("operations", [])
    image_name: str = session.get("image_name", session_path.stem)
    canvas_size: int = session.get("canvas_size", 512)

    scale = CANVAS_SIZE / canvas_size
    if scale != 1.0:
        operations = [_rescale_operation(op, scale) for op in operations]
        epsilon = epsilon * scale
        logger.info(
            "Rescaling %s: canvas_size %d -> %d (scale=%.3f, epsilon=%.3f)",
            session_path.stem,
            canvas_size,
            CANVAS_SIZE,
            scale,
            epsilon,
        )

    strokes = _operations_to_strokes(operations)

    compressed_strokes: list[dict[str, Any]] = []
    for stroke in strokes:
        tool = stroke["tool"]

        if tool == "color_adjust":
            compressed_strokes.append({
                "action_type": "color_adjust",
                "brightness": stroke.get("brightness", 0),
                "exposure": stroke.get("exposure", 0),
                "contrast": stroke.get("contrast", 1.0),
                "highlights": stroke.get("highlights", 0),
                "shadows": stroke.get("shadows", 0),
                "saturation": stroke.get("saturation", 1.0),
                "hue_shift": stroke.get("hue_shift", 0),
                "temperature": stroke.get("temperature", 0),
                "frame_id": stroke["frame_id"],
            })
            continue

        if tool == "scatter_brush":
            traj = stroke.get("trajectory", [stroke.get("start_pos", [0, 0])])
            color = stroke.get("color")
            start = traj[0] if traj else [0, 0]
            end = traj[-1] if traj else start
            compressed_strokes.append({
                "action_type": "scatter_brush",
                "color_rgba": _to_rgba(color) if color else [0, 0, 0, 255],
                "start_point": [int(start[0]), int(start[1])],
                "end_point": [int(end[0]), int(end[1])],
                "size": max(1, int(stroke.get("size", 8))),
                "shape": stroke.get("shape", "circle"),
                "density": stroke.get("density", 5),
                "scatter": stroke.get("scatter", 30),
                "size_jitter": stroke.get("size_jitter", 50),
                "angle_jitter": stroke.get("angle_jitter", 0),
                "seed": stroke.get("seed", 0),
                "thickness": stroke.get("thickness", 1),
                "length": stroke.get("length", 0),
                "base_angle": stroke.get("base_angle", -1),
                "frame_id": stroke["frame_id"],
            })
            continue

        if tool == "pattern_brush":
            traj = stroke.get("trajectory", [stroke.get("start_pos", [0, 0])])
            color = stroke.get("color")
            start = traj[0] if traj else [0, 0]
            end = traj[-1] if traj else start
            compressed_strokes.append({
                "action_type": "pattern_brush",
                "color_rgba": _to_rgba(color) if color else [0, 0, 0, 255],
                "start_point": [int(start[0]), int(start[1])],
                "end_point": [int(end[0]), int(end[1])],
                "size": max(1, int(stroke.get("size", 10))),
                "shape": stroke.get("shape", "leaf"),
                "spacing": stroke.get("spacing", 20),
                "angle_jitter": stroke.get("angle_jitter", 15),
                "thickness": stroke.get("thickness", 1),
                "length": stroke.get("length", 0),
                "frame_id": stroke["frame_id"],
            })
            continue

        if tool == "forward_warp":
            traj = stroke.get("trajectory", [stroke.get("start_pos", [0, 0]), stroke.get("end_pos", [0, 0])])
            start = traj[0] if traj else [0, 0]
            end = traj[-1] if len(traj) >= 2 else start
            compressed_strokes.append({
                "action_type": "forward_warp",
                "start_point": [int(start[0]), int(start[1])],
                "end_point": [int(end[0]), int(end[1])],
                "size": max(1, int(stroke.get("size", 20))),
                "strength": stroke.get("strength", 50),
                "frame_id": stroke["frame_id"],
            })
            continue

        if tool == "text_overlay":
            color = stroke.get("color")
            compressed_strokes.append({
                "action_type": "text_overlay",
                "text": stroke.get("text", ""),
                "position": stroke.get("start_pos", [0, 0]),
                "font_name": stroke.get("font_name", "simplex"),
                "font_size": float(stroke.get("font_size", 1.0)),
                "color_rgba": _to_rgba(color) if color else [0, 0, 0, 255],
                "thickness": stroke.get("thickness", 1),
                "frame_id": stroke["frame_id"],
            })
            continue

        if tool == "gaussian_blur":
            compressed_strokes.append({
                "action_type": "gaussian_blur",
                "radius": max(1, int(stroke.get("size", 5))),
                "frame_id": stroke["frame_id"],
            })
            continue

        if tool == "clone_stamp":
            compressed_strokes.append({
                "action_type": "clone_stamp",
                "source": stroke.get("start_pos", [0, 0]),
                "destination": stroke.get("end_pos", [0, 0]),
                "size": max(1, int(stroke.get("size", 10))),
                "frame_id": stroke["frame_id"],
            })
            continue

        traj = stroke["trajectory"]

        if tool == "fill":
            compressed_strokes.append({
                "action_type": "fill",
                "color_rgba": _to_rgba(stroke["color"]),
                "position": traj[0] if traj else [0, 0],
                "frame_id": stroke["frame_id"],
            })
        elif tool == "eraser":
            base = {
                "action_type": "eraser",
                "stroke_size": max(1, stroke["size"]),
                "frame_id": stroke["frame_id"],
            }
            compressed_strokes.extend(
                _segment_actions_from_trajectory(base, _compress_trajectory(traj, epsilon))
            )
        elif tool == "pencil":
            base = {
                "action_type": "pencil",
                "color_rgba": _to_rgba(stroke["color"]),
                "frame_id": stroke["frame_id"],
            }
            compressed_strokes.extend(
                _segment_actions_from_trajectory(base, _compress_trajectory(traj, epsilon))
            )
        else:  # brush
            base = {
                "action_type": "brush",
                "color_rgba": _to_rgba(stroke["color"]),
                "stroke_size": max(1, stroke["size"]),
                "hardness": stroke.get("hardness", 100),
                "frame_id": stroke["frame_id"],
            }
            compressed_strokes.extend(
                _segment_actions_from_trajectory(base, _compress_trajectory(traj, epsilon))
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "image_name": image_name,
        "canvas_size": CANVAS_SIZE,
        "session": session_path.stem,
        "strokes": compressed_strokes,
    }
    with output_path.open("w") as f:
        json.dump(result, f, indent=2)

    original_ops = len(operations)
    compressed_ops = len(compressed_strokes)
    logger.info(
        "Compressed %s: %d ops -> %d action segments",
        session_path.stem,
        original_ops,
        compressed_ops,
    )
    return compressed_strokes


def compress_all(
    sessions_dir: Path,
    output_dir: Path,
    epsilon: float = 2.0,
) -> dict[str, list[dict[str, Any]]]:
    """Compress all session logs in a directory.

    Args:
        sessions_dir: Directory containing raw session JSONs.
        output_dir: Directory to write compressed JSONs.
        epsilon: RDP tolerance in pixels.

    Returns:
        Mapping of session name -> compressed strokes.
    """
    session_files = sorted(sessions_dir.glob("*.json"))
    if not session_files:
        logger.warning("No session files found in %s", sessions_dir)
        return {}

    results: dict[str, list[dict[str, Any]]] = {}
    for session_path in session_files:
        out_path = output_dir / session_path.name
        results[session_path.stem] = compress_session(session_path, out_path, epsilon)

    logger.info("Compressed %d sessions", len(results))
    return results
