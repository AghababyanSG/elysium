"""Action chunking via sliding window over compressed strokes.

Transforms a list of compressed strokes into overlapping (observation_frame, actions)
pairs for training. Each pair contains:
  - observation_frame: path to the canvas JPEG captured *before* the first action
  - actions: list of `horizon` consecutive strokes (padded with noop if needed)

Sliding window with stride=1 maximises samples: N strokes -> N - horizon + 1 chunks
(or N chunks if padding is applied for sessions shorter than horizon).

Input:  data/interim/compressed/{session_name}.json
Output: data/interim/chunks/{session_name}.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

__all__ = ["chunk_session", "chunk_all"]

logger = logging.getLogger(__name__)

_NOOP = {"action_type": "noop"}


def _make_action_payload(stroke: dict[str, Any]) -> dict[str, Any]:
    """Strip internal fields (frame_id) from a compressed stroke before storing."""
    return {k: v for k, v in stroke.items() if k != "frame_id"}


def chunk_session(
    compressed_path: Path,
    frames_dir: Path,
    output_path: Path,
    horizon: int = 5,
) -> list[dict[str, Any]]:
    """Produce sliding-window chunks from a compressed session.

    Args:
        compressed_path: Path to compressed session JSON.
        frames_dir: Directory holding canvas frame JPEGs (data/raw/frames/).
        output_path: Path to write chunks JSON.
        horizon: Number of actions per chunk.

    Returns:
        List of chunk dicts, each with `observation_frame` and `actions`.
    """
    with compressed_path.open() as f:
        data = json.load(f)

    image_name: str = data["image_name"]
    session: str = data["session"]
    strokes: list[dict[str, Any]] = data.get("strokes", [])

    if not strokes:
        logger.warning("Session %s has no strokes, skipping", session)
        return []

    actions = [_make_action_payload(s) for s in strokes]

    chunks: list[dict[str, Any]] = []
    n = len(actions)

    for i in range(n):
        window = actions[i : i + horizon]
        if len(window) < horizon:
            window = window + [_NOOP] * (horizon - len(window))

        frame_id = strokes[i]["frame_id"]
        frame_path = frames_dir / session / f"{frame_id}.jpg"

        if not frame_path.exists():
            logger.debug("Frame %s not found, skipping chunk %d", frame_path, i)
            continue

        chunks.append({
            "session": session,
            "chunk_index": i,
            "observation_frame": str(frame_path.resolve()),
            "actions": window,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump({"session": session, "horizon": horizon, "chunks": chunks}, f, indent=2)

    logger.info(
        "Chunked %s: %d strokes -> %d chunks (horizon=%d)",
        session,
        n,
        len(chunks),
        horizon,
    )
    return chunks


def chunk_all(
    compressed_dir: Path,
    frames_dir: Path,
    output_dir: Path,
    horizon: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Chunk all compressed sessions in a directory.

    Args:
        compressed_dir: Directory containing compressed session JSONs.
        frames_dir: Directory holding canvas frame JPEGs.
        output_dir: Directory to write chunk JSONs.
        horizon: Number of actions per chunk.

    Returns:
        Mapping of session name -> list of chunk dicts.
    """
    compressed_files = sorted(compressed_dir.glob("*.json"))
    if not compressed_files:
        logger.warning("No compressed files found in %s", compressed_dir)
        return {}

    results: dict[str, list[dict[str, Any]]] = {}
    for cp in compressed_files:
        out_path = output_dir / cp.name
        chunks = chunk_session(cp, frames_dir, out_path, horizon=horizon)
        results[cp.stem] = chunks

    total = sum(len(v) for v in results.values())
    logger.info("Chunked %d sessions, %d total chunks", len(results), total)
    return results
