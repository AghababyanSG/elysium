"""Phase 7.3 entry point: train the SigLIP DrawingCritic.

Usage:
    python scripts/train_critic.py
    python scripts/train_critic.py --epochs 10 --batch-size 32 --lr 5e-5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from elysium.model.critic import DEFAULT_SIGLIP_MODEL
from elysium.model.critic_train import run_training


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path,
                   default=Path("data/processed_critic"))
    p.add_argument("--output-dir", type=Path, default=Path("models/critic"))
    p.add_argument("--siglip-model", type=str, default=DEFAULT_SIGLIP_MODEL)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--pos-ratio", type=float, default=0.5,
                   help="Fraction of each batch that is positive class.")
    p.add_argument("--pos-loss-weight", type=float, default=1.5,
                   help="Multiplier on per-element BCE loss for positive samples.")
    p.add_argument("--hidden-dim", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_training(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        siglip_model=args.siglip_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pos_ratio=args.pos_ratio,
        pos_loss_weight=args.pos_loss_weight,
        hidden_dim=args.hidden_dim,
    )


if __name__ == "__main__":
    main()
