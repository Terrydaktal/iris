#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert near-white pixels to transparency in a PNG."
    )
    parser.add_argument("src", nargs="?", default="icon_source.png")
    parser.add_argument("dst", nargs="?", default="icon.png")
    return parser.parse_args()


def make_transparent(img_path: Path, output_path: Path) -> None:
    img = Image.open(img_path).convert("RGBA")
    data = img.getdata()

    new_data = []
    for item in data:
        r, g, b, a = item
        if r > 240 and g > 240 and b > 240:
            min_val = min(r, g, b)
            alpha = int((255 - min_val) * (255.0 / 15.0))
            alpha = max(0, min(255, alpha))
            new_data.append((r, g, b, alpha))
        else:
            new_data.append((r, g, b, a))

    img.putdata(new_data)
    img.save(output_path, "PNG")


if __name__ == "__main__":
    args = parse_args()
    src = Path(args.src).expanduser()
    dst = Path(args.dst).expanduser()
    print(f"Loading image from {src}...")
    make_transparent(src, dst)
    print(f"Saved transparent PNG to {dst}")
