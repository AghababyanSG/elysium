from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from elysium.schemas.actions import CANVAS_SIZE


def convert_to_jpg(input_path: str, output_path: str) -> None:
    img = Image.open(input_path)
    img = img.convert("RGB")
    img = img.resize((CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.LANCZOS)
    img.save(output_path, "JPEG", quality=95)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python to_jpg.py <input_image> <output_image.jpg>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    convert_to_jpg(input_path, output_path)
    print(f"Converted {input_path} -> {output_path}")
