from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2

if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--canvas-size", type=int, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    return parser.parse_args()


def read_image_bgr(path: Path) -> Any:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Failed to decode image: {path}")
    return frame


def resize_frame_max_side(frame: Any, max_side: int) -> Any:
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return frame
    scale = max_side / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def extract_text(reader: Any, image_path: Path, max_side: int, batch_size: int, canvas_size: int) -> str:
    frame = resize_frame_max_side(read_image_bgr(image_path), max_side)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    texts = reader.readtext(
        rgb,
        detail=0,
        paragraph=False,
        batch_size=batch_size,
        canvas_size=canvas_size,
    )

    lines: list[str] = []
    seen: set[str] = set()
    for raw in texts:
        text = str(raw).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(text)
    return "\n".join(lines)


def write_response(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=True), flush=True)


def main() -> None:
    args = parse_args()
    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    if not langs:
        raise RuntimeError("At least one EasyOCR language is required.")

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("EasyOCR worker was configured for CUDA but torch cannot see a CUDA GPU.")
        # This environment's torch/cuDNN stack aborts in cuDNN with:
        # "Invalid handle. Cannot load symbol cublasLtCreate". Keep EasyOCR on
        # CUDA, but bypass cuDNN so PyTorch uses non-cuDNN CUDA kernels instead.
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.empty(1, device="cuda")
        torch.cuda.synchronize()

    import easyocr

    reader = easyocr.Reader(langs, gpu=args.device == "cuda")
    write_response({"status": "ready", "device": args.device})

    for line in sys.stdin:
        if not line.strip():
            continue
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            job_id = request["id"]
            text = extract_text(
                reader,
                Path(request["image_path"]),
                int(request["max_side"]),
                int(args.batch_size),
                int(args.canvas_size),
            )
            write_response({"id": job_id, "ok": True, "text": text})
        except Exception as exc:
            write_response(
                {
                    "id": request.get("id") if isinstance(request, dict) else None,
                    "ok": False,
                    "error": str(exc).strip() or exc.__class__.__name__,
                }
            )


if __name__ == "__main__":
    main()
