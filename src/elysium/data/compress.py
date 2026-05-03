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

__all__ = ["compress_session", "compress_all"]

logger = logging.getLogger(__name__)


_STROKE_TOOLS = {"brush", "pencil", "eraser", "fill"}


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
        if tool == "color_adjust":
            if current is not None and last_draw_op is not None:
                strokes.append(_finalize_stroke(current, last_draw_op))
                current = None
                last_draw_op = None
            strokes.append({
                "tool": "color_adjust",
                "brightness": op["brightness"],
                "contrast": op["contrast"],
                "saturation": op["saturation"],
                "frame_id": op["frame_id"],
            })
            continue
        if tool not in _STROKE_TOOLS:
            continue

        size = op["size"]
        color = op["color"]
        point = op["start_pos"]
        ts = op["timestamp"]

        same_tool = current is not None and current["tool"] == tool
        same_color = current is not None and current["color"] == color
        same_size = current is not None and current["size"] == size
        no_gap = current is not None and (ts - current["last_ts"]) < 1.0

        if same_tool and same_color and same_size and no_gap:
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
                "frame_id": op["frame_id"],
                "trajectory": [point],
                "last_ts": ts,
            }
            last_draw_op = op

    if current is not None and last_draw_op is not None:
        strokes.append(_finalize_stroke(current, last_draw_op))

    return strokes


def _to_rgba(color: Sequence[int]) -> list[int]:
    if len(color) == 3:
        return [int(color[0]), int(color[1]), int(color[2]), 255]
    assert len(color) == 4, f"Unexpected color shape: {color!r}"
    return [int(c) for c in color]


def _compress_trajectory(trajectory: list[list[int]], epsilon: float) -> list[list[int]]:
    """Apply RDP to a trajectory, preserving at least 2 points.

    Args:
        trajectory: List of [x, y] coordinate pairs.
        epsilon: RDP tolerance in pixels. Higher = more compression.

    Returns:
        Compressed list of [x, y] control points, capped at MAX_TRAJECTORY_POINTS.
    """
    from elysium.schemas.actions import MAX_TRAJECTORY_POINTS

    if len(trajectory) <= 2:
        return trajectory[:MAX_TRAJECTORY_POINTS]
    pts = np.array(trajectory, dtype=float)
    mask = rdp(pts, epsilon=epsilon, return_mask=True)
    compressed = pts[mask].astype(int).tolist()
    if len(compressed) < 2:
        compressed = [trajectory[0], trajectory[-1]]
    if len(compressed) <= MAX_TRAJECTORY_POINTS:
        return compressed
    indices = np.linspace(0, len(compressed) - 1, MAX_TRAJECTORY_POINTS, dtype=int)
    subsampled = [compressed[i] for i in indices]
    subsampled[0] = compressed[0]
    subsampled[-1] = compressed[-1]
    return subsampled


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
    canvas_size: int = session.get("canvas_size", 256)

    strokes = _operations_to_strokes(operations)

    compressed_strokes: list[dict[str, Any]] = []
    for stroke in strokes:
        tool = stroke["tool"]

        if tool == "color_adjust":
            compressed_strokes.append({
                "action_type": "color_adjust",
                "brightness": stroke["brightness"],
                "contrast": stroke["contrast"],
                "saturation": stroke["saturation"],
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
            compressed_strokes.append({
                "action_type": "eraser",
                "stroke_size": max(1, stroke["size"]),
                "trajectory": _compress_trajectory(traj, epsilon),
                "frame_id": stroke["frame_id"],
            })
        elif tool == "pencil":
            compressed_strokes.append({
                "action_type": "pencil",
                "color_rgba": _to_rgba(stroke["color"]),
                "trajectory": _compress_trajectory(traj, epsilon),
                "frame_id": stroke["frame_id"],
            })
        else:
            compressed_strokes.append({
                "action_type": "brush",
                "color_rgba": _to_rgba(stroke["color"]),
                "stroke_size": max(1, stroke["size"]),
                "trajectory": _compress_trajectory(traj, epsilon),
                "frame_id": stroke["frame_id"],
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "image_name": image_name,
        "canvas_size": canvas_size,
        "session": session_path.stem,
        "strokes": compressed_strokes,
    }
    with output_path.open("w") as f:
        json.dump(result, f, indent=2)

    original_ops = len(operations)
    compressed_ops = sum(len(s.get("trajectory", [s.get("position", [])])) for s in compressed_strokes)
    logger.info(
        "Compressed %s: %d ops -> %d strokes (%d control points)",
        session_path.stem,
        original_ops,
        len(compressed_strokes),
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
