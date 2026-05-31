#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute CLIP and/or face embeddings for a single image path."
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--faces", action="store_true")
    return parser.parse_args()


def emit(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=True))
    return 0


def resolve_imagesearch_root() -> Path:
    from_env = os.environ.get("IRIS_IMAGESEARCH_DIR", "").strip()
    if from_env:
        return Path(from_env).expanduser()

    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "imagesearch",
        Path.cwd() / "../imagesearch",
        Path.cwd() / "imagesearch",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return Path("imagesearch")


def main() -> int:
    args = parse_args()
    if not args.clip and not args.faces:
        args.clip = True
        args.faces = True

    image_path = args.image.expanduser()
    if not image_path.is_file():
        return emit({"ok": False, "error": f"image does not exist: {image_path}"})

    # Reuse the existing imagesearch implementation and model configuration.
    imagesearch_root = resolve_imagesearch_root()
    sys.path.insert(0, str(imagesearch_root))
    try:
        from main import (
            ClipEmbedder,
            DEFAULT_CLIP_MODEL,
            FaceEmbedder,
            default_insightface_root,
            read_image_bgr,
        )
    except Exception as exc:
        return emit({"ok": False, "error": f"failed to import imagesearch main.py: {exc}"})

    try:
        frame = read_image_bgr(image_path)
        clip_embedding = None
        face_embeddings: list[list[float]] = []

        if args.clip:
            clip = ClipEmbedder(DEFAULT_CLIP_MODEL, 1)
            vectors = clip.embed_frames([frame])
            if vectors:
                clip_embedding = vectors[0]

        if args.faces:
            face = FaceEmbedder(default_insightface_root())
            face_embeddings = face.detect_and_embed_frame(frame)

        return emit(
            {
                "ok": True,
                "clip_embedding": clip_embedding,
                "face_embeddings": face_embeddings,
            }
        )
    except Exception as exc:
        return emit({"ok": False, "error": str(exc)})


if __name__ == "__main__":
    raise SystemExit(main())
