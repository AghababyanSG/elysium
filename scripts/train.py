"""Launch QLoRA fine-tuning (SFT) or REINFORCE RL training.

Usage:
    python scripts/train.py                          # SFT only
    python scripts/train.py --epochs 5 --batch-size 2
    python scripts/train.py --rl                     # RL only (requires SFT checkpoint)
    python scripts/train.py --sft --rl               # SFT warmup then RL
    python scripts/train.py --config configs/train.yaml --rl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from elysium.model.train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3.5 with QLoRA or REINFORCE RL")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override SFT epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override SFT batch size")
    parser.add_argument(
        "--rl",
        action="store_true",
        help="Run REINFORCE RL training (after SFT warmup if --sft is also given)",
    )
    parser.add_argument(
        "--sft",
        action="store_true",
        help="Run SFT before RL (ignored when --rl is not given; SFT always runs without --rl)",
    )
    parser.add_argument(
        "--sft-checkpoint",
        type=Path,
        default=None,
        help="Path to SFT checkpoint to initialise RL from (default: models/checkpoints/final)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_sft = not args.rl or args.sft
    run_rl = args.rl

    if run_sft:
        overrides: dict[str, int] = {}
        if args.epochs is not None:
            overrides["epochs"] = args.epochs
        if args.batch_size is not None:
            overrides["batch_size"] = args.batch_size
        run_training(config_path=args.config, **overrides)

    if run_rl:
        from elysium.model.rl_train import run_rl_training
        run_rl_training(config_path=args.config, checkpoint_dir=args.sft_checkpoint)


if __name__ == "__main__":
    main()
