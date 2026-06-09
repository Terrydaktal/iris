#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import os
import site
import sys
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from face_worker import FaceEmbedder, apply_nvidia_library_path, preload_onnxruntime_cuda, read_image_bgr


def default_insightface_root() -> Path:
    if Path("/data/.cache").exists():
        return Path("/data/.cache/insightface")
    return Path("~/.insightface").expanduser()


def _probe_cublaslt() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("libcublasLt.so.12", "libcublasLt.so.11"):
        try:
            lib = ctypes.CDLL(name)
            has_symbol = hasattr(lib, "cublasLtCreate")
            result[name] = {"load": True, "cublasLtCreate": bool(has_symbol)}
        except Exception as exc:  # pragma: no cover - diagnostics only
            result[name] = {"load": False, "error": repr(exc)}
    return result


def _child(conn, insightface_root: str, image_path: str | None) -> None:
    out: dict[str, Any] = {}
    try:
        env = os.environ
        apply_nvidia_library_path()
        out["ld_library_path"] = env.get("LD_LIBRARY_PATH", "")
        out["cublaslt"] = _probe_cublaslt()

        import onnxruntime as ort

        out["ort_version"] = ort.__version__
        out["ort_available_providers"] = list(ort.get_available_providers())
        try:
            preload_onnxruntime_cuda()
            out["ort_preload_dlls"] = True
        except Exception as exc:
            out["ort_preload_dlls"] = False
            out["ort_preload_error"] = repr(exc)
        try:
            ort_path = Path(ort.__file__).resolve().parent / "capi" / "libonnxruntime_providers_cuda.so"
            out["ort_cuda_provider_so"] = str(ort_path)
        except Exception:
            pass
        torch_libs: list[str] = []
        for root in site.getsitepackages():
            lib_dir = Path(root) / "torch" / "lib"
            if lib_dir.is_dir():
                torch_libs.append(str(lib_dir))
        out["torch_lib_dirs"] = torch_libs

        embedder = FaceEmbedder(
            Path(insightface_root),
            det_thresh=0.25,
            det_size=640,
            fallback_det_size=960,
            dedupe_cosine=0.99,
        )
        model_providers: dict[str, list[str]] = {}
        for model_name, model in embedder.app.models.items():
            session = getattr(model, "session", None)
            if session is not None:
                model_providers[model_name] = list(session.get_providers())
        out["ok"] = True
        out["model_providers"] = model_providers
        if image_path:
            frame = read_image_bgr(Path(image_path))
            out["image_shape"] = list(frame.shape)
            out["fallback_face_count"] = len(embedder.detect_and_embed_frame(frame))
    except Exception as exc:
        out["ok"] = False
        out["error"] = repr(exc)
        out["traceback"] = traceback.format_exc()
    conn.send(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe InsightFace CUDA worker startup and optional fallback detection.")
    parser.add_argument("--image", type=Path, help="Optional image to run through the same fallback detector path.")
    parser.add_argument("--timeout-seconds", type=float, default=60.0, help="Maximum seconds to wait for the probe child.")
    args = parser.parse_args()
    insightface_root = str(default_insightface_root())
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    image_path = str(args.image.resolve()) if args.image else None
    proc = ctx.Process(target=_child, args=(child, insightface_root, image_path), daemon=False)
    proc.start()
    if parent.poll(max(1.0, float(args.timeout_seconds))):
        payload = parent.recv()
    else:
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
        payload = {"ok": False, "error": f"probe timed out after {args.timeout_seconds}s"}
    proc.join(timeout=5)
    payload["exit_code"] = proc.exitcode
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
