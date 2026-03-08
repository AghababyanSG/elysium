"""Run inference on an image with a text instruction.

Usage:
    python scripts/infer.py <image_path> "<instruction>"
    python scripts/infer.py data/raw/images/pirlo.jpg "Draw a mustache on the face"
    python scripts/infer.py data/raw/images/pirlo.jpg "Draw glasses on the face"

Options:
    --checkpoint   Path to model checkpoint dir (default: models/checkpoints/final)
    --output       Path to save result image    (default: outputs/<stem>_result.jpg)
    --config       Path to train.yaml           (default: configs/train.yaml)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from elysium.model.predict import run_inference

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLA inference on an image")
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument("instruction", type=str, help="Editing instruction")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models/checkpoints/final",
        help="Path to fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save result image (default: outputs/<stem>_result.jpg)",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train.yaml")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a live preview window; close it to stop inference early",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image.exists():
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    if not args.checkpoint.exists():
        print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        print("Run scripts/train.py first.", file=sys.stderr)
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        output_path = ROOT / "outputs" / f"{args.image.stem}_result.jpg"

    run_inference(
        image_path=args.image,
        instruction=args.instruction,
        checkpoint_dir=args.checkpoint,
        output_path=output_path,
        config_path=args.config,
        show_preview=args.preview,
    )


if __name__ == "__main__":
    main()
