from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import site
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2
import numpy as np

if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--insightface-root", required=True)
    parser.add_argument("--det-threshold", type=float, required=True)
    parser.add_argument("--det-size", type=int, required=True)
    parser.add_argument("--fallback-det-size", type=int, required=True)
    parser.add_argument("--dedupe-cosine", type=float, required=True)
    return parser.parse_args()


def read_image_bgr(path: Path) -> Any:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Failed to decode image: {path}")
    return frame


def apply_nvidia_library_path() -> None:
    lib_dirs: list[str] = []
    for root in site.getsitepackages():
        nvidia_root = Path(root) / "nvidia"
        if nvidia_root.is_dir():
            lib_dirs.extend(str(path) for path in sorted(nvidia_root.glob("*/lib")) if path.is_dir())
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    existing_parts = [part for part in existing.split(os.pathsep) if part]
    ordered = list(dict.fromkeys(lib_dirs + existing_parts))
    if ordered:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(ordered)


def nvidia_library_paths(names: list[str]) -> list[Path]:
    paths: list[Path] = []
    for root in site.getsitepackages():
        nvidia_root = Path(root) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for name in names:
            paths.extend(sorted(nvidia_root.glob(f"*/lib/{name}")))
    return list(dict.fromkeys(paths))


def preload_cublas_global() -> None:
    for path in nvidia_library_paths(["libcublasLt.so.12", "libcublas.so.12"]):
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def preload_onnxruntime_cuda() -> None:
    # Keep stdout clean: the parent process reads this worker's stdout as JSON.
    with contextlib.redirect_stdout(sys.stderr):
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")


class FaceEmbedder:
    def __init__(
        self,
        insightface_root: Path,
        det_thresh: float,
        det_size: int,
        fallback_det_size: int,
        dedupe_cosine: float,
    ) -> None:
        apply_nvidia_library_path()
        preload_cublas_global()
        preload_onnxruntime_cuda()
        from insightface.app import FaceAnalysis

        insightface_root.mkdir(parents=True, exist_ok=True)
        self.det_thresh = float(det_thresh)
        self.det_size = int(det_size)
        self.fallback_det_size = int(fallback_det_size)
        self.dedupe_cosine = float(dedupe_cosine)
        self.current_det_size = 0
        self.app = FaceAnalysis(
            name="buffalo_l",
            root=str(insightface_root),
            allowed_modules=["detection", "recognition"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            provider_options=[
                {
                    "device_id": "0",
                    "cudnn_conv_algo_search": "DEFAULT",
                    "cudnn_conv_use_max_workspace": "0",
                    "do_copy_in_default_stream": "1",
                    "enable_cuda_graph": "0",
                    "arena_extend_strategy": "kSameAsRequested",
                },
                {},
            ],
        )
        self._prepare_det_size(self.det_size)
        self._assert_cuda()

    def _assert_cuda(self) -> None:
        for name, model in self.app.models.items():
            session = getattr(model, "session", None)
            if session is None:
                continue
            providers = session.get_providers()
            if "CUDAExecutionProvider" not in providers:
                raise RuntimeError(f"InsightFace {name} is not running on CUDAExecutionProvider.")

    def _prepare_det_size(self, det_size: int) -> None:
        if self.current_det_size == det_size and hasattr(self, "app"):
            return
        with contextlib.redirect_stdout(sys.stderr):
            self.app.prepare(ctx_id=0, det_size=(det_size, det_size), det_thresh=self.det_thresh)
        self.current_det_size = det_size
        self._assert_cuda()

    def _detect(self, frame: Any) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for face in self.app.get(frame):
            vec_raw = getattr(face, "normed_embedding", None)
            if vec_raw is None:
                vec_raw = getattr(face, "embedding", None)
            if vec_raw is None:
                continue
            vec = np.asarray(vec_raw, dtype=np.float32).flatten()
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec.tolist())
        return embeddings

    def _dedupe_embeddings(self, existing: list[list[float]], incoming: list[list[float]]) -> list[list[float]]:
        if not incoming:
            return existing
        merged = [list(vec) for vec in existing]
        existing_np = [np.asarray(vec, dtype=np.float32) for vec in merged]
        for vec in incoming:
            vec_np = np.asarray(vec, dtype=np.float32)
            if any(float(np.dot(vec_np, prior)) >= self.dedupe_cosine for prior in existing_np):
                continue
            merged.append(vec)
            existing_np.append(vec_np)
        return merged

    @staticmethod
    def _rotation_variants(frame: Any) -> list[Any]:
        return [
            cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(frame, cv2.ROTATE_180),
            cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]

    def detect_and_embed_frame(self, frame: Any) -> list[list[float]]:
        self._prepare_det_size(self.det_size)
        embeddings = self._detect(frame)
        if embeddings:
            return embeddings

        for rotated in self._rotation_variants(frame):
            embeddings = self._dedupe_embeddings(embeddings, self._detect(rotated))
        if embeddings or self.fallback_det_size <= self.det_size:
            return embeddings

        self._prepare_det_size(self.fallback_det_size)
        embeddings = self._dedupe_embeddings(embeddings, self._detect(frame))
        if embeddings:
            return embeddings
        for rotated in self._rotation_variants(frame):
            embeddings = self._dedupe_embeddings(embeddings, self._detect(rotated))
        return embeddings


def write_response(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=True), flush=True)


def main() -> None:
    args = parse_args()
    apply_nvidia_library_path()
    with contextlib.redirect_stdout(sys.stderr):
        embedder = FaceEmbedder(
            Path(args.insightface_root),
            det_thresh=args.det_threshold,
            det_size=args.det_size,
            fallback_det_size=args.fallback_det_size,
            dedupe_cosine=args.dedupe_cosine,
        )
    write_response({"status": "ready"})

    for line in sys.stdin:
        if not line.strip():
            continue
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            job_id = request["id"]
            frame = read_image_bgr(Path(request["image_path"]))
            embeddings = embedder.detect_and_embed_frame(frame)
            write_response({"id": job_id, "ok": True, "embeddings": embeddings})
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
