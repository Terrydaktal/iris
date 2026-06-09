from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
# oneDNN can trip PIR conversion errors on CPU fallback paths in some builds.
os.environ.setdefault("FLAGS_use_mkldnn", "0")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--device", choices=("gpu:0", "cpu"), required=True)
    return parser.parse_args()


def read_image_bgr(cv2_module: Any | None, path: Path) -> Any:
    if cv2_module is not None:
        frame = cv2_module.imread(str(path), cv2_module.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to decode image: {path}")
        return frame
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            arr = np.array(rgb, dtype=np.uint8)
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"Failed to decode image: {path}") from exc
    # PaddleOCR expects OpenCV-style BGR frames.
    return arr[:, :, ::-1].copy()


def resize_frame_max_side(cv2_module: Any | None, frame: Any, max_side: int) -> Any:
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return frame
    scale = max_side / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    if cv2_module is not None:
        return cv2_module.resize(frame, (new_width, new_height), interpolation=cv2_module.INTER_AREA)
    from PIL import Image
    import numpy as np

    rgb = frame[:, :, ::-1]
    resized = Image.fromarray(rgb, mode="RGB").resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
    arr = np.array(resized, dtype=np.uint8)
    return arr[:, :, ::-1].copy()


def polys_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return len(value) > 0
    if hasattr(value, "shape"):
        return int(value.shape[0]) > 0
    try:
        return len(value) > 0
    except TypeError:
        return False


def text_detection_result_has_polys(result: Any) -> bool:
    if isinstance(result, dict) and polys_present(result.get("dt_polys")):
        return True
    data = getattr(result, "json", None)
    if isinstance(data, dict):
        if polys_present(data.get("dt_polys")):
            return True
        nested = data.get("res")
        if isinstance(nested, dict) and polys_present(nested.get("dt_polys")):
            return True
    return False


def write_response(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=True), flush=True)


def main() -> None:
    try:
        args = parse_args()
        cv2 = None
        cv2_import_error: str | None = None
        try:
            import cv2 as _cv2

            cv2 = _cv2
            if hasattr(cv2, "setLogLevel"):
                cv2.setLogLevel(0)
        except Exception as exc:  # pragma: no cover - fallback is runtime dependent
            cv2_import_error = str(exc).strip() or exc.__class__.__name__
        if args.device.startswith("gpu"):
            import paddle

            if not paddle.is_compiled_with_cuda():
                raise RuntimeError(
                    "paddle worker configured for GPU but installed Paddle is CPU-only. "
                    "Install paddlepaddle-gpu into this interpreter."
                )
            if paddle.device.cuda.device_count() < 1:
                raise RuntimeError("paddle worker configured for GPU but no CUDA GPU is visible.")

        from paddleocr import TextDetection

        detector = TextDetection(model_name=args.model_name, device=args.device)
        write_response(
            {
                "status": "ready",
                "device": args.device,
                "decoder": "cv2" if cv2 is not None else "pillow",
                "cv2_error": cv2_import_error,
            }
        )
    except Exception as exc:
        write_response(
            {
                "status": "error",
                "error": str(exc).strip() or exc.__class__.__name__,
                "python_exe": sys.executable,
                "python_version": sys.version.split()[0],
                "sys_path_head": sys.path[:6],
                "venv": os.environ.get("VIRTUAL_ENV"),
                "cwd": str(Path.cwd()),
            }
        )
        raise

    for line in sys.stdin:
        if not line.strip():
            continue
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            job_id = request["id"]
            image_path = Path(request["image_path"])
            max_side = int(request["max_side"])
            frame = resize_frame_max_side(cv2, read_image_bgr(cv2, image_path), max_side)
            results = detector.predict(frame)
            has_text = bool(results) and any(text_detection_result_has_polys(result) for result in results)
            write_response({"id": job_id, "ok": True, "text_detected": bool(has_text)})
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
