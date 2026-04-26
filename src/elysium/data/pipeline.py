"""Data preparation pipeline orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from elysium.data.chunk import chunk_all
from elysium.data.compress import compress_all
from elysium.data.format import build_dataset

__all__ = ["run_pipeline", "DataPaths"]

logger = logging.getLogger(__name__)


def _resolve_path(root: Path, value: str | Path) -> Path:
    return root / value if isinstance(value, str) and not Path(value).is_absolute() else Path(value)


class DataPaths:
    def __init__(self, root: Path, cfg: dict[str, Any]) -> None:
        data = cfg.get("data", {})
        raw = data.get("raw", {})
        interim = data.get("interim", {})

        self.root = root
        self.sessions = _resolve_path(root, raw.get("sessions", "data/raw/sessions"))
        self.frames = _resolve_path(root, raw.get("frames", "data/raw/frames"))
        self.compressed = _resolve_path(root, interim.get("compressed", "data/interim/compressed"))
        self.chunks = _resolve_path(root, interim.get("chunks", "data/interim/chunks"))
        self.processed = _resolve_path(root, data.get("dataset_path", "data/processed"))
        self.instructions = _resolve_path(root, data.get("instructions", "configs/instructions.yaml"))


def run_pipeline(
    config_path: Path,
    root: Path | None = None,
    rdp_epsilon: float | None = None,
    instructions_path: Path | None = None,
) -> None:
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    paths = DataPaths(root or config_path.parent.parent, cfg)
    epsilon = rdp_epsilon if rdp_epsilon is not None else data_cfg.get("rdp_epsilon", 2.0)
    horizon = data_cfg["action_horizon"]
    train_split = data_cfg["train_split"]
    instr_path = instructions_path or paths.instructions

    logger.info("Step 1/3: Compressing sessions (epsilon=%.1f)", epsilon)
    compress_all(paths.sessions, paths.compressed, epsilon=epsilon)

    logger.info("Step 2/3: Chunking (horizon=%d)", horizon)
    chunk_all(paths.compressed, paths.frames, paths.chunks, horizon=horizon)

    logger.info("Step 3/3: Building dataset (train_split=%.2f)", train_split)
    build_dataset(paths.chunks, instr_path, paths.processed, train_split=train_split)

    logger.info("Data pipeline complete.")
