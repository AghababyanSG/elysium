"""Rescale images to CANVAS_SIZE x CANVAS_SIZE (see elysium.schemas.actions).

Usage:
    python scripts/rescale_images.py <input_path> [--output <output_path>]
    python scripts/rescale_images.py data/raw/images/pirlo.jpg
    python scripts/rescale_images.py data/raw/images/ --output-dir data/rescaled/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from elysium.schemas.actions import CANVAS_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Rescale images to {CANVAS_SIZE}x{CANVAS_SIZE}")
    parser.add_argument("input", type=Path, help="Input image or directory of images")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for single image (default: in-place)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for batch (default: same as input)",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=None,
        help="Filter by extension when batch (e.g. .jpg)",
    )
    return parser.parse_args()


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def rescale_image(src: Path, dst: Path, size: tuple[int, int] | None = None) -> None:
    if size is None:
        size = (CANVAS_SIZE, CANVAS_SIZE)
    img = Image.open(src).convert("RGB")
    img = img.resize(size, Image.Resampling.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"Error: not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    size = (CANVAS_SIZE, CANVAS_SIZE)

    if args.input.is_file():
        out = args.output
        if out is None:
            out = args.input
        rescale_image(args.input, out, size)
        print(f"Rescaled: {args.input} -> {out}")
        return

    out_dir = args.output_dir if args.output_dir is not None else args.input
    ext_filter = args.ext.lower() if args.ext else None

    count = 0
    for path in args.input.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        if ext_filter and path.suffix.lower() != ext_filter:
            continue
        dst = out_dir / path.relative_to(args.input)
        rescale_image(path, dst, size)
        count += 1
        print(f"Rescaled: {path} -> {dst}")

    print(f"Done: {count} images rescaled to {size[0]}x{size[1]}")


if __name__ == "__main__":
    main()
