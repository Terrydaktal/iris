#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
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
    parser.add_argument(
        "--face-details",
        action="store_true",
        help="Return normalized bounding boxes with face embeddings.",
    )
    return parser.parse_args()


def emit(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=True))
    return 0


def resolve_media_indexer_root() -> Path:
    from_env = os.environ.get("IRIS_MEDIA_INDEXER_DIR", "").strip()
    if from_env:
        return Path(from_env).expanduser()

    here = Path(__file__).resolve()
    candidates = [
        here.parent / "media_indexer",
        here.parent.parent / "tools" / "media_indexer",
        Path.cwd() / "tools" / "media_indexer",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return Path("tools/media_indexer")


def clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def compute_clip_embedding(frame, clip_embedder_cls, model_name: str, oom_predicate) -> list[float] | None:
    clip = None
    try:
        clip = clip_embedder_cls(model_name, 1)
        vectors = clip.embed_frames([frame])
        return vectors[0] if vectors else None
    except Exception as exc:
        if not oom_predicate(exc):
            raise
        clip = None
        clear_cuda_cache()
        clip = clip_embedder_cls(model_name, 1, device="cpu")
        vectors = clip.embed_frames([frame])
        return vectors[0] if vectors else None


def detect_face_details(frame, face_embedder) -> list[dict]:
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    variants = [
        (None, "none"),
        (cv2.ROTATE_90_CLOCKWISE, "cw"),
        (cv2.ROTATE_180, "180"),
        (cv2.ROTATE_90_COUNTERCLOCKWISE, "ccw"),
    ]

    def original_bbox(bbox_raw, rotation: str) -> list[float]:
        x1, y1, x2, y2 = (float(value) for value in bbox_raw[:4])
        if rotation == "cw":
            return [y1, height - x2, y2, height - x1]
        if rotation == "180":
            return [width - x2, height - y2, width - x1, height - y1]
        if rotation == "ccw":
            return [width - y2, x1, width - y1, x2]
        return [x1, y1, x2, y2]

    for det_size in (face_embedder.det_size, face_embedder.fallback_det_size):
        if det_size <= 0:
            continue
        face_embedder._prepare_det_size(det_size)
        for rotation_code, rotation in variants:
            variant = frame if rotation_code is None else cv2.rotate(frame, rotation_code)
            details = []
            for detected in face_embedder.app.get(variant):
                vec_raw = getattr(detected, "normed_embedding", None)
                if vec_raw is None:
                    vec_raw = getattr(detected, "embedding", None)
                bbox_raw = getattr(detected, "bbox", None)
                if vec_raw is None or bbox_raw is None:
                    continue

                vector = np.asarray(vec_raw, dtype=np.float32).flatten()
                norm = float(np.linalg.norm(vector))
                if norm <= 0:
                    continue
                vector = vector / norm
                x1, y1, x2, y2 = original_bbox(bbox_raw, rotation)
                details.append(
                    {
                        "embedding": vector.tolist(),
                        "bbox": [
                            max(0.0, min(1.0, x1 / width)),
                            max(0.0, min(1.0, y1 / height)),
                            max(0.0, min(1.0, x2 / width)),
                            max(0.0, min(1.0, y2 / height)),
                        ],
                    }
                )
            if details:
                return details
    return []


def main() -> int:
    args = parse_args()
    if args.face_details:
        args.faces = True
    if not args.clip and not args.faces:
        args.clip = True
        args.faces = True

    image_path = args.image.expanduser()
    if not image_path.is_file():
        return emit({"ok": False, "error": f"image does not exist: {image_path}"})

    # Reuse the integrated media indexer implementation and model configuration.
    media_indexer_root = resolve_media_indexer_root()
    sys.path.insert(0, str(media_indexer_root))
    try:
        from main import (
            ClipEmbedder,
            DEFAULT_CLIP_MODEL,
            FaceEmbedder,
            default_insightface_root,
            is_cuda_oom_error,
            read_image_bgr,
        )
    except Exception as exc:
        return emit({"ok": False, "error": f"failed to import media_indexer main.py: {exc}"})

    try:
        frame = read_image_bgr(image_path)
        clip_embedding = None
        face_embeddings: list[list[float]] = []
        face_details: list[dict] = []

        if args.clip:
            clip_embedding = compute_clip_embedding(
                frame,
                ClipEmbedder,
                DEFAULT_CLIP_MODEL,
                is_cuda_oom_error,
            )

        if args.faces:
            face = FaceEmbedder(default_insightface_root())
            if args.face_details:
                face_details = detect_face_details(frame, face)
                face_embeddings = [detail["embedding"] for detail in face_details]
            else:
                face_embeddings = face.detect_and_embed_frame(frame)

        return emit(
            {
                "ok": True,
                "clip_embedding": clip_embedding,
                "face_embeddings": face_embeddings,
                "face_details": face_details,
            }
        )
    except Exception as exc:
        return emit({"ok": False, "error": str(exc)})


if __name__ == "__main__":
    raise SystemExit(main())
