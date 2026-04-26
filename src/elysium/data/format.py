"""Dataset formatter: converts action chunks + manual instructions into a HuggingFace Dataset.

Reads:
  - data/interim/chunks/{session}.json  -- action chunks from chunk.py
  - configs/instructions.yaml           -- session -> instruction mapping

Produces a HuggingFace Dataset saved to data/processed/ with records shaped as
Qwen2.5-VL chat-template conversations:

  user:      [image token] + instruction text
  assistant: JSON string {"actions": [...]}  (the 5-action chunk)

Additional fields for RL training:
  gt_actions   -- raw JSON string of ground-truth actions (same as assistant text)
  next_image   -- path to the canvas frame before chunk i+1 (next-state target for
                  visual reward). Empty string for the last chunk in a session.

The dataset is split into train/validation subsets.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import yaml

from elysium.schemas.actions import SYSTEM_PROMPT

__all__ = ["build_dataset"]

logger = logging.getLogger(__name__)


def _load_instructions(instructions_path: Path) -> dict[str, str]:
    """Return mapping of session_name -> instruction string.

    Args:
        instructions_path: Path to configs/instructions.yaml.

    Returns:
        Dict of {session_name: instruction}.
    """
    with instructions_path.open() as f:
        config = yaml.safe_load(f)

    session_to_instruction: dict[str, str] = {}
    for task_info in config.get("tasks", {}).values():
        instruction = task_info["instruction"]
        for session in task_info.get("sessions", []):
            session_to_instruction[session] = instruction

    return session_to_instruction


def _load_chunks(chunks_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load all chunk records grouped by session, sorted by chunk_index.

    Args:
        chunks_dir: Directory containing {session}.json chunk files.

    Returns:
        Dict mapping session name -> sorted list of chunk dicts.
    """
    sessions: dict[str, list[dict[str, Any]]] = {}
    for chunk_file in sorted(chunks_dir.glob("*.json")):
        with chunk_file.open() as f:
            data = json.load(f)
        session_name = data.get("session", chunk_file.stem)
        chunks = sorted(data.get("chunks", []), key=lambda c: c.get("chunk_index", 0))
        sessions[session_name] = chunks
    return sessions


def _to_conversation(
    chunk: dict[str, Any],
    instruction: str,
    next_observation_frame: str = "",
) -> dict[str, Any]:
    """Convert a single chunk into a chat-template conversation record.

    Args:
        chunk: Chunk dict with observation_frame and actions.
        instruction: Natural language instruction for this chunk.
        next_observation_frame: Path to the next chunk's observation frame,
            used as the visual reward target during RL training. Empty string
            if this is the last chunk in the session.

    Returns:
        Dict with "messages", "gt_actions", and "next_image" fields.
    """
    action_json = json.dumps({"actions": chunk["actions"]}, separators=(",", ":"))

    image_path = str(Path(chunk["observation_frame"]).resolve())
    return {
        "image": image_path,
        "gt_actions": action_json,
        "next_image": next_observation_frame,
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": instruction},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": action_json}],
            },
        ],
    }


def build_dataset(
    chunks_dir: Path,
    instructions_path: Path,
    output_dir: Path,
    train_split: float = 0.8,
    seed: int = 42,
) -> None:
    """Build and save a HuggingFace Dataset from chunks and instructions.

    Args:
        chunks_dir: Directory containing chunk JSONs.
        instructions_path: Path to configs/instructions.yaml.
        output_dir: Directory to save the HuggingFace dataset.
        train_split: Fraction of data to use for training.
        seed: Random seed for reproducible splits.
    """
    try:
        from datasets import Dataset, DatasetDict
    except ImportError as e:
        raise ImportError("Install `datasets` to use this module: pip install datasets") from e

    session_to_instruction = _load_instructions(instructions_path)
    sessions = _load_chunks(chunks_dir)

    if not sessions:
        raise ValueError(f"No chunks found in {chunks_dir}. Run prepare_data.py first.")

    records: list[dict[str, Any]] = []
    skipped = 0
    for session_name, chunks in sessions.items():
        instruction = session_to_instruction.get(session_name)
        if instruction is None:
            skipped += len(chunks)
            continue
        for i, chunk in enumerate(chunks):
            next_frame = (
                str(Path(chunks[i + 1]["observation_frame"]).resolve())
                if i + 1 < len(chunks)
                else ""
            )
            records.append(_to_conversation(chunk, instruction, next_frame))

    if not records:
        raise ValueError(
            "No records produced. Ensure sessions in configs/instructions.yaml match "
            "session names in data/interim/chunks/."
        )

    if skipped:
        logger.warning("Skipped %d chunks from sessions with no instruction mapping", skipped)

    random.seed(seed)
    random.shuffle(records)

    split_idx = int(len(records) * train_split)
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    dataset = DatasetDict({
        "train": Dataset.from_list(train_records),
        "validation": Dataset.from_list(val_records),
    })

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))

    logger.info(
        "Dataset saved to %s: %d train, %d validation records",
        output_dir,
        len(train_records),
        len(val_records),
    )
