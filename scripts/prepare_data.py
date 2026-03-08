"""Data preparation pipeline.

  1. Compress raw session logs (RDP trajectory compression)
  2. Chunk compressed strokes into horizon-sized training examples
  3. Build HuggingFace dataset with manual instruction labels

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --config configs/train.yaml --epsilon 2.0
    python scripts/prepare_data.py --instructions configs/instructions.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from elysium.data.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full data preparation pipeline")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train.yaml")
    parser.add_argument("--epsilon", type=float, default=None, help="RDP epsilon override")
    parser.add_argument(
        "--instructions",
        type=Path,
        default=None,
        help="Override instructions.yaml path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        args.config,
        root=ROOT,
        rdp_epsilon=args.epsilon,
        instructions_path=args.instructions,
    )


if __name__ == "__main__":
    main()
