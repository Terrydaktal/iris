#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

from PIL import Image, ImageFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove a solid or noisy background connected to the image edges and save "
            "the result as a transparent PNG."
        )
    )
    parser.add_argument("src", nargs="?", default="icon_source.png")
    parser.add_argument("dst", nargs="?", default="icon.png")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=55.0,
        help="Maximum RGB distance from the detected border background (default: 55).",
    )
    parser.add_argument(
        "--feather",
        type=float,
        default=0.8,
        help="Edge feather radius in pixels; use 0 for a hard edge (default: 0.8).",
    )
    return parser.parse_args()


def border_background_color(image: Image.Image) -> tuple[int, int, int]:
    width, height = image.size
    pixels = image.load()
    border = [
        *(pixels[x, 0][:3] for x in range(width)),
        *(pixels[x, height - 1][:3] for x in range(width)),
        *(pixels[0, y][:3] for y in range(height)),
        *(pixels[width - 1, y][:3] for y in range(height)),
    ]
    return tuple(sorted(pixel[channel] for pixel in border)[len(border) // 2] for channel in range(3))


def connected_background_mask(
    image: Image.Image,
    background: tuple[int, int, int],
    tolerance: float,
) -> Image.Image:
    width, height = image.size
    pixels = image.load()
    background_pixels = bytearray(width * height)
    queued = bytearray(width * height)
    pending: deque[tuple[int, int]] = deque()
    tolerance_squared = tolerance * tolerance

    def enqueue(x: int, y: int) -> None:
        index = y * width + x
        if not queued[index]:
            queued[index] = 1
            pending.append((x, y))

    for x in range(width):
        enqueue(x, 0)
        enqueue(x, height - 1)
    for y in range(height):
        enqueue(0, y)
        enqueue(width - 1, y)

    while pending:
        x, y = pending.popleft()
        r, g, b, _ = pixels[x, y]
        distance_squared = (
            (r - background[0]) ** 2
            + (g - background[1]) ** 2
            + (b - background[2]) ** 2
        )
        if distance_squared > tolerance_squared:
            continue

        background_pixels[y * width + x] = 1
        if x > 0:
            enqueue(x - 1, y)
        if x + 1 < width:
            enqueue(x + 1, y)
        if y > 0:
            enqueue(x, y - 1)
        if y + 1 < height:
            enqueue(x, y + 1)

    alpha = Image.new("L", image.size, 255)
    alpha_pixels = alpha.load()
    for index, is_background in enumerate(background_pixels):
        if is_background:
            alpha_pixels[index % width, index // width] = 0
    return alpha


def remove_background(image: Image.Image, tolerance: float, feather: float) -> Image.Image:
    rgba = image.convert("RGBA")
    background = border_background_color(rgba)
    alpha = connected_background_mask(rgba, background, tolerance)
    if feather > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather))

    source_pixels = rgba.load()
    alpha_pixels = alpha.load()
    output = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    output_pixels = output.load()

    for y in range(rgba.height):
        for x in range(rgba.width):
            alpha_value = alpha_pixels[x, y]
            if alpha_value <= 8:
                continue

            source_r, source_g, source_b, source_alpha = source_pixels[x, y]
            combined_alpha = round(alpha_value * source_alpha / 255)
            if combined_alpha <= 8:
                continue

            # Remove the detected background color from partially transparent edge pixels.
            opacity = alpha_value / 255.0
            clean_channels = []
            for source, backdrop in zip(
                (source_r, source_g, source_b),
                background,
                strict=True,
            ):
                clean = (source - (1.0 - opacity) * backdrop) / opacity
                clean_channels.append(max(0, min(255, round(clean))))
            output_pixels[x, y] = (*clean_channels, combined_alpha)

    return output


def make_transparent(
    img_path: Path,
    output_path: Path,
    tolerance: float = 55.0,
    feather: float = 0.8,
) -> None:
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if feather < 0:
        raise ValueError("feather must be non-negative")
    with Image.open(img_path) as image:
        output = remove_background(image, tolerance, feather)
    output.save(output_path, "PNG")


if __name__ == "__main__":
    args = parse_args()
    src = Path(args.src).expanduser()
    dst = Path(args.dst).expanduser()
    print(f"Loading image from {src}...")
    make_transparent(src, dst, tolerance=args.tolerance, feather=args.feather)
    print(f"Saved transparent PNG to {dst}")
