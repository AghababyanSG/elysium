"""Export a TensorBoard scalar to CSV.

Default invocation (no args) dumps the RL training loss for the latest
RL run to ``outputs/figures/rl_training_loss.csv``.

Usage:
    python scripts/tb_to_csv.py
    python scripts/tb_to_csv.py --tag train/reward
    python scripts/tb_to_csv.py --run-dir models/checkpoints/rl_final/runs/<id> \
        --tag train/kl --out outputs/figures/rl_kl.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_RUN_DIR = Path("models/checkpoints/rl_final/runs/May17_13-21-41_zeus")
DEFAULT_TAG = "train/loss"
DEFAULT_OUT = Path("outputs/figures/rl_training_loss.csv")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR,
                   help="TensorBoard event-file directory.")
    p.add_argument("--tag", type=str, default=DEFAULT_TAG,
                   help="Scalar tag to export (e.g. train/loss, train/reward).")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="CSV output path.")
    args = p.parse_args()

    if not args.run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {args.run_dir}")

    ea = EventAccumulator(str(args.run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    if args.tag not in ea.Tags()["scalars"]:
        raise KeyError(
            f"Tag {args.tag!r} not found. Available: {ea.Tags()['scalars']}"
        )

    events = ea.Scalars(args.tag)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "wall_time", args.tag.replace("/", "_")])
        for e in events:
            w.writerow([e.step, e.wall_time, e.value])

    print(f"Wrote {len(events)} rows from {args.tag} -> {args.out}")


if __name__ == "__main__":
    main()
