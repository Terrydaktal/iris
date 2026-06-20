from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing as mp
import os
import re
import select
import site
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, TypeVar

# Keep noisy FFmpeg/OpenCV decoder warnings from breaking tqdm output.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("MPLCONFIGDIR", "/data/.cache/matplotlib")


def _cuda_library_dirs() -> list[str]:
    lib_dirs: list[str] = []
    for root in site.getsitepackages():
        site_root = Path(root)
        nvidia_root = site_root / "nvidia"
        if nvidia_root.is_dir():
            lib_dirs.extend(str(path) for path in sorted(nvidia_root.glob("*/lib")) if path.is_dir())
    return list(dict.fromkeys(lib_dirs))


def _bootstrap_cuda_library_path() -> None:
    # LD_LIBRARY_PATH is read by the dynamic linker at process startup. Re-exec
    # once so ORT/Paddle/PyTorch native libraries resolve against this venv.
    if os.environ.get("EMBEDIMAGES_CUDA_ENV_BOOTSTRAPPED") == "1":
        return
    lib_dirs = _cuda_library_dirs()
    if not lib_dirs:
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    existing_parts = [part for part in existing.split(os.pathsep) if part]
    ordered = list(dict.fromkeys(lib_dirs + existing_parts))
    new_ld = os.pathsep.join(ordered)
    if sys.argv and sys.argv[0] in {"-c", "-"}:
        os.environ["LD_LIBRARY_PATH"] = new_ld
        os.environ["EMBEDIMAGES_CUDA_ENV_BOOTSTRAPPED"] = "1"
        return
    if existing == new_ld:
        os.environ["EMBEDIMAGES_CUDA_ENV_BOOTSTRAPPED"] = "1"
        return
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = new_ld
    env["EMBEDIMAGES_CUDA_ENV_BOOTSTRAPPED"] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


_bootstrap_cuda_library_path()

import cv2
import lancedb
import numpy as np
import pyarrow as pa
import torch
from PIL import Image
from tqdm import tqdm

if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)


IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".heic",
    ".heif",
}
VIDEO_EXTS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
    ".wmv",
    ".mpg",
    ".mpeg",
}

DEFAULT_CLIP_MODEL = "hf-hub:timm/ViT-L-16-SigLIP2-384"
DEFAULT_PADDLE_DET_MODEL = "PP-OCRv5_mobile_det"
MAX_STILLS_PER_VIDEO = 100
TIMESTAMP_ROUND_DIGITS = 3
FACE_VECTOR_DIM = 512
ANN_INDEX_TYPE = "IVF_HNSW_SQ"
ANN_DISTANCE_METRIC = "cosine"
UPSERT_BATCH_SIZE = 1000
CLIP_UPSERT_BATCH_SIZE = 50
PADDLE_OCR_MAX_SIDE = 2048
EASYOCR_MAX_SIDE = 1600
EASYOCR_CANVAS_SIZE = 1600
EASYOCR_BATCH_SIZE = 8
EASYOCR_WORKER_READY_TIMEOUT_SECONDS = 120.0
EASYOCR_WORKER_JOB_TIMEOUT_SECONDS = 300.0
PADDLE_WORKER_READY_TIMEOUT_SECONDS = 180.0
PADDLE_WORKER_JOB_TIMEOUT_SECONDS = 180.0
FACE_DET_THRESHOLD_DEFAULT = 0.5
FACE_DET_SIZE_DEFAULT = 640
FACE_FALLBACK_DET_SIZE_DEFAULT = 1280
FACE_DEDUP_COSINE_DEFAULT = 0.995
FACE_MIN_SIDE_DEFAULT = 160
FACE_SKIP_ASPECT_RATIO_DEFAULT = 3.5
FACE_SKIP_ASPECT_MIN_SIDE_DEFAULT = 256
SIFT_MIN_RATIO_DEFAULT = 0.75
SIFT_MIN_INLIERS_DEFAULT = 10
SIFT_MIN_INLIER_RATIO_DEFAULT = 0.75
SIFT_CONTRAST_THRESHOLD_DEFAULT = 0.03
SIFT_CANDIDATE_TOPK_DEFAULT = 64
SIFT_MAX_SIDE_DEFAULT = 1920
SIFT_FEATURE_CACHE_SIZE = 256
SIFT_MAX_FEATURES_DEFAULT = 2000
SIFT_MAX_RANSAC_MATCHES_DEFAULT = 500
PROGRESS_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
    "[elapsed {elapsed}, left {remaining}, {rate_fmt}]"
)
PROGRESS_DELAY_SECONDS = 2.0
CLIP_PREPROCESS_MAX_SIDE = 2048
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_CYAN = "\033[36m"
ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"
ANSI_YELLOW = "\033[33m"
ANSI_ORANGE = "\033[38;5;208m"
T = TypeVar("T")
SIFT_THREAD_LOCAL = threading.local()

FACE_GROUP_STRUCT = pa.struct(
    [
        pa.field("timestamp_sec", pa.float32()),
        pa.field("face_embeddings", pa.list_(pa.list_(pa.float32()))),
    ]
)
VIDEO_FRAME_PHASH_STRUCT = pa.struct(
    [
        pa.field("timestamp_sec", pa.float32()),
        pa.field("phash_hex", pa.string()),
    ]
)
CLIP_GROUP_STRUCT = pa.struct(
    [
        pa.field("timestamp_sec", pa.float32()),
        pa.field("clip_embedding", pa.list_(pa.float32())),
    ]
)
OCR_GROUP_STRUCT = pa.struct(
    [
        pa.field("timestamp_sec", pa.float32()),
        pa.field("text_detected", pa.bool_()),
        pa.field("text", pa.string()),
    ]
)

SCHEMA = pa.schema(
    [
        pa.field("file_name", pa.string()),
        pa.field("collection_id", pa.string()),
        pa.field("is_video", pa.bool_()),
        pa.field("source_size", pa.int64()),
        pa.field("source_mtime_ns", pa.int64()),
        pa.field("image_width", pa.int32()),
        pa.field("image_height", pa.int32()),
        pa.field("faces", pa.bool_()),
        pa.field("phash_hex", pa.string()),
        pa.field("video_frame_phashes", pa.list_(VIDEO_FRAME_PHASH_STRUCT)),
        pa.field("skip_processing", pa.bool_()),
        pa.field("dedupe_match_file", pa.string()),
        pa.field("dedupe_similarity_pct", pa.float32()),
        pa.field(
            "cross_media_matches",
            pa.list_(
                pa.struct(
                    [
                        pa.field("file_name", pa.string()),
                        pa.field("is_video", pa.bool_()),
                        pa.field("similarity_pct", pa.float32()),
                    ]
                )
            ),
        ),
        pa.field("sift_match_file", pa.string()),
        pa.field("sift_match_score", pa.float32()),
        pa.field("sift_match_inliers", pa.int32()),
        pa.field("sift_match_good_matches", pa.int32()),
        pa.field("sift_match_inlier_ratio", pa.float32()),
        pa.field("sift_match_checked", pa.bool_()),
        pa.field("processing_error_stage", pa.string()),
        pa.field("processing_error", pa.string()),
        pa.field("face_groups", pa.list_(FACE_GROUP_STRUCT)),
        pa.field("clip_groups", pa.list_(CLIP_GROUP_STRUCT)),
        pa.field("ocr_groups", pa.list_(OCR_GROUP_STRUCT)),
    ]
)


@dataclass
class AppConfig:
    input_dir: Path
    collection_id: str
    db_dir: Path
    table_name: str
    insightface_root: Path
    clip_model: str
    max_face_embeddings_per_file: int
    clip_batch_size: int
    easyocr_langs: list[str]
    easyocr_max_side: int
    easyocr_canvas_size: int
    easyocr_batch_size: int
    easyocr_gpu: bool
    ocr_text_model: str
    ann_text_batch_size: int
    ocr_text_device: str
    paddle_det_model: str
    paddle_device: str
    paddle_python: Path
    paddle_ocr_max_side: int
    skip_paddle_ocr: bool
    rerun_paddle_ocr: bool
    wipe_paddle_failures_before_run: bool
    hash_workers: int
    face_timeout_seconds: int
    max_consecutive_face_timeouts: int
    face_det_threshold: float
    face_det_size: int
    face_fallback_det_size: int
    face_dedupe_cosine: float
    face_min_side: int
    face_skip_aspect_ratio: float
    face_skip_aspect_min_side: int
    rerun_face_failures: bool
    rerun_zero_face_detections: bool
    phash_skip_similarity_pct: float
    cross_media_similarity_pct: float
    video_hash_skip_similarity_pct: float
    run_sift_master_match: bool
    rerun_sift_master_match: bool
    sift_min_ratio: float
    sift_min_inliers: int
    sift_min_inlier_ratio: float
    sift_contrast_threshold: float
    sift_candidate_topk: int
    sift_max_side: int
    sift_max_features: int
    sift_max_ransac_matches: int
    scene_threshold: float
    scene_min_scene_len: int
    repair_image_masters: bool
    repair_only: bool
    rerun_stages: set[str]


@dataclass(frozen=True)
class FrameRef:
    timestamp_sec: float
    image_path: Path


@dataclass
class MediaItem:
    source_path: Path
    file_name: str
    collection_id: str
    is_video: bool
    frame_refs: list[FrameRef]


@dataclass(frozen=True)
class VideoFrameHashRef:
    file_name: str
    timestamp_sec: float
    phash_hex: str


class HumanTqdm(tqdm):
    @staticmethod
    def format_interval(seconds: float) -> str:
        total = max(0, int(round(seconds)))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours}h {minutes:02d}m {secs:02d}s"
        if minutes:
            return f"{minutes}m {secs:02d}s"
        return f"{secs}s"


def progress(
    iterable: Iterable[T],
    *,
    desc: str,
    unit: str,
    total: int | None = None,
) -> HumanTqdm:
    return HumanTqdm(
        iterable,
        desc=format_stage_label(desc, stream=sys.stderr),
        unit=unit,
        total=total,
        delay=PROGRESS_DELAY_SECONDS,
        smoothing=0.05,
        dynamic_ncols=True,
        bar_format=PROGRESS_BAR_FORMAT,
    )


def use_color(stream: Any = sys.stdout) -> bool:
    return os.environ.get("NO_COLOR") is None and bool(getattr(stream, "isatty", lambda: False)())


def color_text(text: str, color: str, *, stream: Any = sys.stdout, bold: bool = False) -> str:
    if not use_color(stream):
        return text
    prefix = f"{ANSI_BOLD if bold else ''}{color}"
    return f"{prefix}{text}{ANSI_RESET}"


def format_stage_label(stage_label: str, *, stream: Any = sys.stdout) -> str:
    match = re.match(r"^(Stages?\s+)(\S+)(.*)$", stage_label)
    if not match or not use_color(stream):
        return stage_label
    prefix, number, name = match.groups()
    return (
        color_text(prefix, ANSI_CYAN, stream=stream, bold=True)
        + color_text(number, ANSI_YELLOW, stream=stream, bold=True)
        + color_text(name, ANSI_CYAN, stream=stream, bold=True)
    )


def report_stage_complete(stage_label: str, count: int, unit: str) -> None:
    label = format_stage_label(stage_label)
    status = color_text("complete", ANSI_GREEN, bold=True)
    print(f"{label}: {status} ({count}/{count} {unit})")


def report_stage_incomplete(stage_label: str, completed: int, total: int, unit: str, reason: str) -> None:
    label = format_stage_label(stage_label)
    status = color_text("incomplete", ANSI_ORANGE, bold=True)
    detail = color_text(reason, ANSI_ORANGE)
    print(f"{label}: {status} ({completed}/{total} {unit}; {detail})")


def report_stage_skipped(stage_label: str, reason: str) -> None:
    label = format_stage_label(stage_label)
    status = color_text("skipped", ANSI_ORANGE, bold=True)
    detail = color_text(reason, ANSI_ORANGE)
    print(f"{label}: {status} ({detail})")


def print_warning(message: str) -> None:
    print(color_text(message, ANSI_ORANGE, stream=sys.stderr), file=sys.stderr)


def print_error(message: str) -> None:
    print(color_text(message, ANSI_RED, stream=sys.stderr, bold=True), file=sys.stderr)


def print_info(message: str) -> None:
    print(color_text(message, ANSI_CYAN, bold=True))


class TimedStep:
    def __init__(self, label: str) -> None:
        self.label = label
        self.started_at = 0.0

    def __enter__(self) -> "TimedStep":
        self.started_at = time.monotonic()
        print_info(f"{format_stage_label(self.label)}: starting")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is not None:
            print_error(f"{self.label}: failed after {time.monotonic() - self.started_at:.1f}s")
            return
        print_info(f"{format_stage_label(self.label)}: complete in {time.monotonic() - self.started_at:.1f}s")


def start_worker_stderr_forwarder(process: subprocess.Popen[str]) -> None:
    if process.stderr is None:
        return

    suppressed_worker_prefixes = (
        "Applied providers:",
        "model ignore:",
        "find model:",
        "set det-size:",
        "warning: det_size is already set",
    )
    suppressed_worker_fragments = (
        "RequestsDependencyWarning:",
        "urllib3",
        "chardet",
        "charset_normalizer",
        "FutureWarning: `estimate` is deprecated",
    )

    def forward() -> None:
        for raw_line in process.stderr:
            line = raw_line.rstrip()
            if not line:
                continue
            if line.startswith(suppressed_worker_prefixes) or any(
                fragment in line for fragment in suppressed_worker_fragments
            ):
                continue
            lowered = line.lower()
            if any(token in lowered for token in ("error", "failed", "traceback", "exception")):
                print_error(line)
            else:
                print_warning(line)

    threading.Thread(target=forward, daemon=True).start()


def shorten_for_status(value: str, max_len: int = 90) -> str:
    if len(value) <= max_len:
        return value
    return "..." + value[-(max_len - 3) :]


def write_status(cfg: AppConfig, **data: Any) -> None:
    cfg.db_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "updated_at_unix": time.time(),
        **data,
    }
    status_path = cfg.db_dir / "embedimages-status.json"
    temp_path = status_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    temp_path.replace(status_path)


def read_status(cfg: AppConfig) -> dict[str, Any] | None:
    status_path = cfg.db_dir / "embedimages-status.json"
    if not status_path.exists():
        return None
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def cross_media_state_path(cfg: AppConfig) -> Path:
    return cfg.db_dir / "cross-media-state.json"


def cross_media_work_path(cfg: AppConfig) -> Path:
    collection_hash = hashlib.sha1(cfg.collection_id.encode("utf-8")).hexdigest()[:12]
    return cfg.db_dir / "cross-media-work" / f"{collection_hash}.json"


def read_cross_media_state(cfg: AppConfig) -> dict[str, Any]:
    path = cross_media_state_path(cfg)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_cross_media_state(cfg: AppConfig, state: dict[str, Any]) -> None:
    path = cross_media_state_path(cfg)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def read_cross_media_work(cfg: AppConfig, fingerprint: str) -> dict[str, Any] | None:
    path = cross_media_work_path(cfg)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("collection_id") != cfg.collection_id or data.get("fingerprint") != fingerprint:
        return None
    desired = data.get("desired_by_file")
    if not isinstance(desired, dict):
        return None
    normalized: dict[str, list[dict[str, Any]]] = {}
    for file_name, matches in desired.items():
        normalized[str(file_name)] = normalize_cross_media_matches(matches) or []
    data["desired_by_file"] = normalized
    return data


def write_cross_media_work(
    cfg: AppConfig,
    *,
    fingerprint: str,
    state: str,
    desired_by_file: dict[str, list[dict[str, Any]]],
    image_total: int,
    video_total: int,
) -> None:
    path = cross_media_work_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "collection_id": cfg.collection_id,
        "fingerprint": fingerprint,
        "state": state,
        "image_total": image_total,
        "video_total": video_total,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "desired_by_file": desired_by_file,
    }
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def clear_cross_media_work(cfg: AppConfig) -> None:
    try:
        cross_media_work_path(cfg).unlink()
    except FileNotFoundError:
        pass


def cross_media_input_fingerprint(
    cfg: AppConfig,
    records: dict[str, dict[str, Any]],
    current_image_files: set[str],
    current_video_files: set[str],
) -> str:
    digest = hashlib.sha256()

    def add(value: str) -> None:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    add("cross-media-v2")
    add(f"threshold={cfg.cross_media_similarity_pct:.9f}")
    for file_name in sorted(current_image_files):
        add(f"scope-image:{file_name}")
    for file_name in sorted(current_video_files):
        add(f"scope-video:{file_name}")
    for file_name in sorted(records):
        record = records[file_name]
        if bool(record.get("is_video")):
            groups = normalize_video_frame_phashes(record.get("video_frame_phashes")) or []
            for group in groups:
                add(
                    f"video-frame:{file_name}:{float(group['timestamp_sec']):.6f}:"
                    f"{group['phash_hex']}"
                )
            continue
        phash_hex = normalize_phash_hex(record.get("phash_hex"))
        if phash_hex is not None:
            add(f"image:{file_name}:{phash_hex}")
    return digest.hexdigest()


class HammingBkNode:
    __slots__ = ("value", "file_name", "children")

    def __init__(self, value: int, file_name: str) -> None:
        self.value = value
        self.file_name = file_name
        self.children: dict[int, HammingBkNode] = {}


class HammingBkTree:
    def __init__(self) -> None:
        self.root: HammingBkNode | None = None

    def add(self, value: int, file_name: str) -> None:
        if self.root is None:
            self.root = HammingBkNode(value, file_name)
            return
        node = self.root
        while True:
            dist = hamming_distance_u64(value, node.value)
            child = node.children.get(dist)
            if child is None:
                node.children[dist] = HammingBkNode(value, file_name)
                return
            node = child

    def find_best(self, value: int, max_distance: int) -> tuple[str, int] | None:
        if self.root is None:
            return None
        best: tuple[str, int] | None = None
        stack = [self.root]
        while stack:
            node = stack.pop()
            dist = hamming_distance_u64(value, node.value)
            if dist <= max_distance:
                if best is None or dist < best[1] or (
                    dist == best[1] and node.file_name < best[0]
                ):
                    best = (node.file_name, dist)
            low = max(0, dist - max_distance)
            high = dist + max_distance
            for edge_dist, child in node.children.items():
                if low <= edge_dist <= high:
                    stack.append(child)
        return best

    def find_all(self, value: int, max_distance: int) -> list[tuple[str, int]]:
        if self.root is None:
            return []
        results: list[tuple[str, int]] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            dist = hamming_distance_u64(value, node.value)
            if dist <= max_distance:
                results.append((node.file_name, dist))
            low = max(0, dist - max_distance)
            high = dist + max_distance
            for edge_dist, child in node.children.items():
                if low <= edge_dist <= high:
                    stack.append(child)
        results.sort(key=lambda entry: (entry[1], entry[0]))
        return results


def default_collection_id(input_dir: Path) -> str:
    root = input_dir.resolve().as_posix()
    digest = hashlib.sha1(root.encode("utf-8")).hexdigest()[:12]
    name = input_dir.name.strip() or "collection"
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    return f"{safe_name}@{digest}"


def default_hash_workers() -> int:
    cpu_count = os.cpu_count() or 4
    return max(1, min(16, cpu_count))


def default_insightface_root() -> Path:
    if Path("/data/.cache").exists():
        return Path("/data/.cache/insightface")
    return Path("~/.insightface").expanduser()


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally process photos/videos with PySceneDetect, InsightFace+ArcFace, "
            "CLIP ViT-L-16-SigLIP2-384, PaddleOCR text detection, and EasyOCR."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Directory to scan for media.")
    parser.add_argument(
        "--collection-id",
        default=None,
        help=(
            "Namespace for this scan inside the shared database. "
            "Defaults to '<folder-name>@<path-hash>'. Use a stable value "
            "to add multiple folders into one DB without key collisions."
        ),
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=Path("./lancedb"),
        help="LanceDB directory.",
    )
    parser.add_argument(
        "--table-name",
        default="media_index",
        help="LanceDB table name.",
    )
    parser.add_argument(
        "--insightface-root",
        type=Path,
        default=default_insightface_root(),
        help="Directory used to cache/download InsightFace face models.",
    )
    parser.add_argument(
        "--clip-model",
        default=DEFAULT_CLIP_MODEL,
        help="OpenCLIP model id for CLIP/SigLIP image embeddings.",
    )
    parser.add_argument(
        "--max-face-embeddings-per-file",
        type=int,
        default=256,
        help="Cap number of ArcFace vectors per file.",
    )
    parser.add_argument(
        "--clip-batch-size",
        type=int,
        default=8,
        help="Batch size for CLIP image encoding.",
    )
    parser.add_argument(
        "--easyocr-langs",
        default="en",
        help="Comma separated EasyOCR languages, e.g. en,fr,de.",
    )
    parser.add_argument(
        "--easyocr-max-side",
        type=int,
        default=EASYOCR_MAX_SIDE,
        help="Resize images so the longest side is at most this many pixels before EasyOCR.",
    )
    parser.add_argument(
        "--easyocr-canvas-size",
        type=int,
        default=EASYOCR_CANVAS_SIZE,
        help="EasyOCR canvas_size. Lower is faster, higher preserves small text.",
    )
    parser.add_argument(
        "--easyocr-batch-size",
        type=int,
        default=EASYOCR_BATCH_SIZE,
        help="EasyOCR recognition batch size inside each image.",
    )
    parser.add_argument(
        "--easyocr-device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Device for EasyOCR text extraction.",
    )
    parser.add_argument(
        "--ocr-text-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers model for OCR text ANN embeddings.",
    )
    parser.add_argument(
        "--ocr-text-device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Device for OCR text ANN embeddings.",
    )
    parser.add_argument(
        "--ann-text-batch-size",
        type=int,
        default=128,
        help="Batch size for OCR text embedding generation.",
    )
    parser.add_argument(
        "--paddle-ocr-max-side",
        type=int,
        default=PADDLE_OCR_MAX_SIDE,
        help="Resize images so the longest side is at most this many pixels before PaddleOCR.",
    )
    parser.add_argument(
        "--paddle-det-model",
        default=DEFAULT_PADDLE_DET_MODEL,
        help="Paddle text-detection-only model for Stage 7 text y/n detection.",
    )
    parser.add_argument(
        "--paddle-device",
        choices=("gpu:0", "cpu"),
        default="gpu:0",
        help="Device for Paddle text y/n detection.",
    )
    parser.add_argument(
        "--paddle-python",
        default=sys.executable,
        help=(
            "Python interpreter used for the isolated Paddle worker. "
            "Point this at a separate env with paddlepaddle-gpu to avoid dependency conflicts."
        ),
    )
    parser.add_argument(
        "--rerun-paddle-ocr",
        action="store_true",
        help="Ignore existing Stage 7 OCR detection groups and recompute Paddle text y/n detection.",
    )
    parser.add_argument(
        "--wipe-paddle-failures-before-run",
        action="store_true",
        help=(
            "Before Stage 7, clear only rows that previously failed in paddle_ocr "
            "(processing_error_stage='paddle_ocr') for files in this run."
        ),
    )
    parser.add_argument(
        "--skip-paddle-ocr",
        action="store_true",
        help="Skip Stage 7 Paddle text y/n detection and continue with later stages.",
    )
    parser.add_argument(
        "--hash-workers",
        type=int,
        default=default_hash_workers(),
        help="Parallel CPU workers for image pHash and VideoHash stages.",
    )
    parser.add_argument(
        "--face-timeout-seconds",
        type=int,
        default=90,
        help=(
            "Maximum seconds to allow one frame in the isolated face worker before "
            "marking that file as face-failed and restarting the worker."
        ),
    )
    parser.add_argument(
        "--max-consecutive-face-timeouts",
        type=int,
        default=3,
        help=(
            "Abort the face stage after this many consecutive per-file timeouts. "
            "Raise this for dirty archives only after CUDA initialization is stable."
        ),
    )
    parser.add_argument(
        "--face-det-threshold",
        type=float,
        default=FACE_DET_THRESHOLD_DEFAULT,
        help="InsightFace detection threshold for stage 1 face detection.",
    )
    parser.add_argument(
        "--face-det-size",
        type=int,
        default=FACE_DET_SIZE_DEFAULT,
        help="Primary InsightFace detector size (square).",
    )
    parser.add_argument(
        "--face-fallback-det-size",
        type=int,
        default=FACE_FALLBACK_DET_SIZE_DEFAULT,
        help=(
            "Fallback detector size used only when no faces are found on first pass. "
            "Set <= --face-det-size to disable larger-size fallback."
        ),
    )
    parser.add_argument(
        "--face-dedupe-cosine",
        type=float,
        default=FACE_DEDUP_COSINE_DEFAULT,
        help="Cosine threshold used to dedupe duplicate embeddings produced by fallback rotations.",
    )
    parser.add_argument(
        "--face-min-side",
        type=int,
        default=FACE_MIN_SIDE_DEFAULT,
        help="Skip face detection for images whose shortest side is below this many pixels.",
    )
    parser.add_argument(
        "--face-skip-aspect-ratio",
        type=float,
        default=FACE_SKIP_ASPECT_RATIO_DEFAULT,
        help="Skip face detection for very wide/tall UI-like images at or above this aspect ratio.",
    )
    parser.add_argument(
        "--face-skip-aspect-min-side",
        type=int,
        default=FACE_SKIP_ASPECT_MIN_SIDE_DEFAULT,
        help="Only apply aspect-ratio face skip when the shortest side is below this many pixels.",
    )
    parser.add_argument(
        "--rerun-face-failures",
        action="store_true",
        help="Rerun Stage 6 face detection for rows that previously failed with processing_error_stage=faces.",
    )
    parser.add_argument(
        "--rerun-zero-face-detections",
        action="store_true",
        help="Rerun Stage 6 face detection for rows that previously completed with zero detected faces.",
    )
    parser.add_argument(
        "--phash-skip-similarity-pct",
        type=float,
        default=95.0,
        help="Skip full processing for images when pHash similarity is >= this percent.",
    )
    parser.add_argument(
        "--cross-media-similarity-pct",
        type=float,
        default=95.0,
        help=(
            "Relate an image to a video when its pHash similarity to an extracted video frame "
            "is >= this percent."
        ),
    )
    parser.add_argument(
        "--video-hash-skip-similarity-pct",
        type=float,
        default=80.0,
        help="Skip full processing for videos when VideoHash similarity is >= this percent.",
    )
    parser.add_argument(
        "--run-sift-master-match",
        action="store_true",
        help=(
            "After CLIP stage, annotate image-to-master matches using "
            "CLIP ANN candidates plus SIFT/RANSAC verification."
        ),
    )
    parser.add_argument(
        "--rerun-sift-master-match",
        action="store_true",
        help="Recompute SIFT master matches even for rows previously marked as checked.",
    )
    parser.add_argument(
        "--sift-min-ratio",
        type=float,
        default=SIFT_MIN_RATIO_DEFAULT,
        help="SIFT Lowe ratio-test threshold (default 0.75).",
    )
    parser.add_argument(
        "--sift-min-inliers",
        type=int,
        default=SIFT_MIN_INLIERS_DEFAULT,
        help="Minimum SIFT homography inlier matches required to accept a master match.",
    )
    parser.add_argument(
        "--sift-min-inlier-ratio",
        type=float,
        default=SIFT_MIN_INLIER_RATIO_DEFAULT,
        help=(
            "Minimum fraction of Lowe-filtered SIFT matches that must survive RANSAC "
            "to accept a master match (default 0.90)."
        ),
    )
    parser.add_argument(
        "--sift-contrast-threshold",
        type=float,
        default=SIFT_CONTRAST_THRESHOLD_DEFAULT,
        help="SIFT contrastThreshold parameter.",
    )
    parser.add_argument(
        "--sift-candidate-topk",
        type=int,
        default=SIFT_CANDIDATE_TOPK_DEFAULT,
        help="Top-K CLIP ANN image-master candidates to verify by SIFT/RANSAC per image.",
    )
    parser.add_argument(
        "--sift-max-side",
        type=int,
        default=SIFT_MAX_SIDE_DEFAULT,
        help="Resize longest image side to this before SIFT for speed.",
    )
    parser.add_argument(
        "--sift-max-features",
        type=int,
        default=SIFT_MAX_FEATURES_DEFAULT,
        help="Maximum SIFT keypoints/descriptors per image. Caps pathological high-texture images.",
    )
    parser.add_argument(
        "--sift-max-ransac-matches",
        type=int,
        default=SIFT_MAX_RANSAC_MATCHES_DEFAULT,
        help="Maximum Lowe-filtered matches passed to homography RANSAC per pair.",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=27.0,
        help="PySceneDetect content threshold.",
    )
    parser.add_argument(
        "--scene-min-scene-len",
        type=int,
        default=15,
        help="PySceneDetect minimum scene length in frames.",
    )
    parser.add_argument(
        "--rerun-stage",
        action="append",
        default=[],
        choices=(
            "all",
            "1",
            "1a",
            "1b",
            "2",
            "3",
            "3a",
            "3b",
            "3c",
            "3d",
            "3e",
            "3f",
            "4",
            "4a",
            "4b",
            "5",
            "5a",
            "5b",
            "6",
            "6a",
            "6b",
            "7",
            "8",
            "8a",
            "8b",
            "9",
        ),
        help=(
            "Force a completed stage to rerun. Repeat for multiple stages; parent stages "
            "such as 3, 4, 6, and 8 include their lettered substages, and 'all' reruns everything."
        ),
    )
    parser.add_argument(
        "--repair-image-masters",
        action="store_true",
        help=(
            "Repair image duplicate groups so likely originals/larger files become pHash masters. "
            "Processing fields are cleared only on files demoted to non-master duplicates."
        ),
    )
    parser.add_argument(
        "--repair-only",
        action="store_true",
        help="Run requested repair steps and exit without face/CLIP/OCR processing.",
    )
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    paddle_python = Path(args.paddle_python).expanduser()
    if not paddle_python.is_absolute():
        paddle_python = Path.cwd() / paddle_python
    return AppConfig(
        input_dir=input_dir,
        collection_id=(
            args.collection_id.strip()
            if isinstance(args.collection_id, str) and args.collection_id.strip()
            else default_collection_id(input_dir)
        ),
        db_dir=args.db_dir.resolve(),
        table_name=args.table_name,
        insightface_root=args.insightface_root.expanduser().resolve(),
        clip_model=args.clip_model,
        max_face_embeddings_per_file=args.max_face_embeddings_per_file,
        clip_batch_size=args.clip_batch_size,
        easyocr_langs=[x.strip() for x in args.easyocr_langs.split(",") if x.strip()],
        easyocr_max_side=max(256, int(args.easyocr_max_side)),
        easyocr_canvas_size=max(256, int(args.easyocr_canvas_size)),
        easyocr_batch_size=max(1, int(args.easyocr_batch_size)),
        easyocr_gpu=args.easyocr_device == "cuda",
        ocr_text_model=args.ocr_text_model,
        ann_text_batch_size=args.ann_text_batch_size,
        ocr_text_device=args.ocr_text_device,
        paddle_det_model=args.paddle_det_model,
        paddle_device=args.paddle_device,
        paddle_python=paddle_python,
        paddle_ocr_max_side=max(256, int(args.paddle_ocr_max_side)),
        skip_paddle_ocr=bool(args.skip_paddle_ocr),
        rerun_paddle_ocr=bool(args.rerun_paddle_ocr),
        wipe_paddle_failures_before_run=bool(args.wipe_paddle_failures_before_run),
        hash_workers=max(1, int(args.hash_workers)),
        face_timeout_seconds=max(1, int(args.face_timeout_seconds)),
        max_consecutive_face_timeouts=max(1, int(args.max_consecutive_face_timeouts)),
        face_det_threshold=min(1.0, max(0.0, float(args.face_det_threshold))),
        face_det_size=max(160, int(args.face_det_size)),
        face_fallback_det_size=max(160, int(args.face_fallback_det_size)),
        face_dedupe_cosine=min(0.9999, max(0.5, float(args.face_dedupe_cosine))),
        face_min_side=max(1, int(args.face_min_side)),
        face_skip_aspect_ratio=max(1.0, float(args.face_skip_aspect_ratio)),
        face_skip_aspect_min_side=max(1, int(args.face_skip_aspect_min_side)),
        rerun_face_failures=bool(args.rerun_face_failures),
        rerun_zero_face_detections=bool(args.rerun_zero_face_detections),
        phash_skip_similarity_pct=min(100.0, max(0.0, float(args.phash_skip_similarity_pct))),
        cross_media_similarity_pct=min(100.0, max(0.0, float(args.cross_media_similarity_pct))),
        video_hash_skip_similarity_pct=min(100.0, max(0.0, float(args.video_hash_skip_similarity_pct))),
        run_sift_master_match=bool(args.run_sift_master_match),
        rerun_sift_master_match=bool(args.rerun_sift_master_match),
        sift_min_ratio=float(args.sift_min_ratio),
        sift_min_inliers=max(1, int(args.sift_min_inliers)),
        sift_min_inlier_ratio=min(1.0, max(0.0, float(args.sift_min_inlier_ratio))),
        sift_contrast_threshold=max(0.0, float(args.sift_contrast_threshold)),
        sift_candidate_topk=max(1, int(args.sift_candidate_topk)),
        sift_max_side=max(256, int(args.sift_max_side)),
        sift_max_features=max(128, int(args.sift_max_features)),
        sift_max_ransac_matches=max(16, int(args.sift_max_ransac_matches)),
        scene_threshold=args.scene_threshold,
        scene_min_scene_len=args.scene_min_scene_len,
        repair_image_masters=bool(args.repair_image_masters),
        repair_only=bool(args.repair_only),
        rerun_stages=set(args.rerun_stage),
    )


def should_rerun_stage(cfg: AppConfig, stage: str) -> bool:
    parent = "".join(char for char in stage if char.isdigit())
    return "all" in cfg.rerun_stages or stage in cfg.rerun_stages or parent in cfg.rerun_stages


def ensure_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required but unavailable. This tool enforces GPU processing. "
            f"torch={torch.__version__}, torch.cuda={torch.version.cuda}"
        )

    try:
        probe = torch.zeros(1, device="cuda")
        probe.add_(1)
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(
            "CUDA is visible to PyTorch, but a trivial CUDA allocation failed. "
            "This usually means the NVIDIA driver, CUDA runtime, or torch wheel are incompatible. "
            f"torch={torch.__version__}, torch.cuda={torch.version.cuda}"
        ) from exc

    if "CUDAExecutionProvider" not in _ort_providers():
        raise RuntimeError(
            "onnxruntime-gpu CUDAExecutionProvider is unavailable. "
            "InsightFace/ArcFace require GPU execution."
        )


def _ort_providers() -> list[str]:
    import onnxruntime as ort

    return list(ort.get_available_providers())


def discover_media_files(root: Path) -> list[Path]:
    files: list[Path] = []
    scanned = 0
    for path in progress(root.rglob("*"), desc="Stage 0a/8 Startup scan media files", unit="path"):
        scanned += 1
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTS or suffix in VIDEO_EXTS:
            files.append(path)
    files.sort()
    print_info(
        f"{format_stage_label('Stage 0a/8 Startup scan media files')}: "
        f"discovered {len(files)} supported media files from {scanned} paths"
    )
    return files


def split_media(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    images: list[Path] = []
    videos: list[Path] = []
    for path in paths:
        if is_video(path):
            videos.append(path)
        else:
            images.append(path)
    return images, videos


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def relative_file_name(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def scoped_file_name(cfg: AppConfig, path: Path) -> str:
    rel = relative_file_name(cfg.input_dir, path)
    return f"{cfg.collection_id}/{rel}"


def migrate_legacy_records_for_scan(
    cfg: AppConfig,
    table,
    records: dict[str, dict[str, Any]],
    media_paths: list[Path],
) -> None:
    rename_pairs: list[tuple[str, str]] = []
    for path in progress(media_paths, desc="Stage 0g/8 Startup check legacy DB keys", unit="file"):
        legacy_key = relative_file_name(cfg.input_dir, path)
        new_key = scoped_file_name(cfg, path)
        if new_key in records:
            continue
        if legacy_key in records:
            rename_pairs.append((legacy_key, new_key))
    if not rename_pairs:
        print_info(f"{format_stage_label('Stage 0h/8 Startup migrate legacy DB keys')}: not needed")
        return

    print_info(f"{format_stage_label('Stage 0h/8 Startup migrate legacy DB keys')}: migrating {len(rename_pairs)} rows")
    remap = {old: new for old, new in rename_pairs}
    for old_key, new_key in progress(rename_pairs, desc="Stage 0h/8 Startup migrate legacy DB keys", unit="row"):
        row = dict(records[old_key])
        row["file_name"] = new_key
        row["collection_id"] = cfg.collection_id
        match_key = row.get("dedupe_match_file")
        if isinstance(match_key, str) and match_key in remap:
            row["dedupe_match_file"] = remap[match_key]
        sift_match_key = row.get("sift_match_file")
        if isinstance(sift_match_key, str) and sift_match_key in remap:
            row["sift_match_file"] = remap[sift_match_key]
        cross_media_matches = row.get("cross_media_matches")
        if isinstance(cross_media_matches, list):
            for entry in cross_media_matches:
                if not isinstance(entry, dict):
                    continue
                match_file = entry.get("file_name")
                if isinstance(match_file, str) and match_file in remap:
                    entry["file_name"] = remap[match_file]
        table.delete(f"file_name = '{escape_sql(old_key)}'")
        records.pop(old_key, None)
        upsert_record(table, records, row)


def default_record(
    file_name: str,
    is_video_file: bool,
    collection_id: str | None = None,
) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "collection_id": collection_id,
        "is_video": is_video_file,
        "source_size": None,
        "source_mtime_ns": None,
        "image_width": None,
        "image_height": None,
        "faces": None,
        "phash_hex": None,
        "video_frame_phashes": None,
        "skip_processing": None,
        "dedupe_match_file": None,
        "dedupe_similarity_pct": None,
        "cross_media_matches": None,
        "sift_match_file": None,
        "sift_match_score": None,
        "sift_match_inliers": None,
        "sift_match_good_matches": None,
        "sift_match_inlier_ratio": None,
        "sift_match_checked": None,
        "processing_error_stage": None,
        "processing_error": None,
        "face_groups": None,
        "clip_groups": None,
        "ocr_groups": None,
    }


def db_table_names(db) -> set[str]:
    if hasattr(db, "list_tables"):
        tables = db.list_tables()
        if hasattr(tables, "tables"):
            tables = tables.tables
        return {str(table) for table in tables}
    return set(db.table_names())


def connect_table(db_dir: Path, table_name: str):
    with TimedStep(f"Stage 0c/8 Startup open LanceDB table {table_name}"):
        db_dir.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(db_dir))
        if table_name not in db_table_names(db):
            table = db.create_table(table_name, data=[], schema=SCHEMA)
            ensure_file_name_index(table)
            return table

        table = db.open_table(table_name)
        existing_fields = {field.name: field.type for field in table.schema}
        missing_fields = [field for field in SCHEMA if field.name not in existing_fields]
        incompatible_fields = [
            field.name
            for field in SCHEMA
            if field.name in existing_fields and existing_fields[field.name] != field.type
        ]
        if not incompatible_fields and missing_fields:
            print_info(
                f"[db] adding nullable columns to {table_name}: "
                + ", ".join(field.name for field in missing_fields)
            )
            table.add_columns(missing_fields)
            existing_fields.update({field.name: field.type for field in missing_fields})
        if not incompatible_fields and all(field.name in existing_fields for field in SCHEMA):
            ensure_file_name_index(table)
            return table

        print_info(f"[db] migrating {table_name} to the current schema; existing rows will be preserved")
        old_rows = table.to_arrow().to_pylist()
        db.drop_table(table_name)
        table = db.create_table(table_name, data=[], schema=SCHEMA)
        migrated = [migrate_row_to_current_schema(row) for row in old_rows]
        if migrated:
            table.add(migrated)
        print_info(f"[db] migrated {len(migrated)} rows")
        ensure_file_name_index(table)
        return table


def ensure_file_name_index(table) -> None:
    try:
        if table.count_rows() == 0:
            return
    except Exception:
        return
    index_name = "file_name_idx"
    try:
        index_names = {index.name for index in table.list_indices()}
    except Exception:
        return
    if index_name in index_names:
        return
    try:
        print_info(f"{format_stage_label('Stage 0d/8 Startup ensure file_name index')}: creating")
        table.create_scalar_index("file_name", replace=False, name=index_name)
        print_info(f"{format_stage_label('Stage 0d/8 Startup ensure file_name index')}: ready")
    except Exception as exc:
        print_warning(f"[db] warning: failed to create file_name scalar index: {exc}")


def migrate_row_to_current_schema(old_row: dict[str, Any]) -> dict[str, Any]:
    file_name = str(old_row.get("file_name", ""))
    rec = default_record(
        file_name=file_name,
        is_video_file=bool(old_row.get("is_video")) if old_row.get("is_video") is not None else is_video(Path(file_name)),
        collection_id=old_row.get("collection_id"),
    )

    old_face_groups = old_row.get("face_groups")
    if isinstance(old_face_groups, list):
        rec["face_groups"] = old_face_groups
    else:
        old_face_embeddings = old_row.get("face_embeddings")
        if old_face_embeddings is not None:
            rec["face_groups"] = [
                {
                    "timestamp_sec": 0.0,
                    "face_embeddings": old_face_embeddings,
                }
            ]

    old_clip_groups = old_row.get("clip_groups")
    if isinstance(old_clip_groups, list):
        rec["clip_groups"] = old_clip_groups
    else:
        old_clip_embedding = old_row.get("clip_embedding")
        if old_clip_embedding is not None:
            rec["clip_groups"] = [
                {
                    "timestamp_sec": 0.0,
                    "clip_embedding": old_clip_embedding,
                }
            ]

    old_ocr_groups = old_row.get("ocr_groups")
    if isinstance(old_ocr_groups, list):
        rec["ocr_groups"] = old_ocr_groups
    else:
        old_text_detected = old_row.get("text_detected")
        old_text = old_row.get("text")
        if old_text_detected is not None:
            rec["ocr_groups"] = [
                {
                    "timestamp_sec": 0.0,
                    "text_detected": bool(old_text_detected),
                    "text": old_text if old_text_detected else "",
                }
            ]

    rec["faces"] = old_row.get("faces")
    rec["source_size"] = old_row.get("source_size")
    rec["source_mtime_ns"] = old_row.get("source_mtime_ns")
    rec["image_width"] = old_row.get("image_width")
    rec["image_height"] = old_row.get("image_height")
    rec["phash_hex"] = old_row.get("phash_hex")
    rec["video_frame_phashes"] = normalize_video_frame_phashes(old_row.get("video_frame_phashes"))
    rec["skip_processing"] = old_row.get("skip_processing")
    rec["dedupe_match_file"] = old_row.get("dedupe_match_file")
    rec["dedupe_similarity_pct"] = old_row.get("dedupe_similarity_pct")
    rec["cross_media_matches"] = normalize_cross_media_matches(old_row.get("cross_media_matches"))
    rec["sift_match_file"] = old_row.get("sift_match_file")
    rec["sift_match_score"] = old_row.get("sift_match_score")
    rec["sift_match_inliers"] = old_row.get("sift_match_inliers")
    rec["sift_match_good_matches"] = old_row.get("sift_match_good_matches")
    rec["sift_match_inlier_ratio"] = old_row.get("sift_match_inlier_ratio")
    rec["sift_match_checked"] = old_row.get("sift_match_checked")
    rec["processing_error_stage"] = old_row.get("processing_error_stage")
    rec["processing_error"] = old_row.get("processing_error")
    recompute_aggregate_fields(rec)
    return rec


def load_records(table) -> dict[str, dict[str, Any]]:
    with TimedStep("Stage 0e/8 Startup load DB rows into memory"):
        try:
            rows = table.to_arrow().to_pylist()
        except Exception:
            return {}

    records: dict[str, dict[str, Any]] = {}
    for row in progress(rows, desc="Stage 0f/8 Startup normalize DB rows", unit="row"):
        file_name = row["file_name"]
        records[file_name] = {
            "file_name": file_name,
            "collection_id": row.get("collection_id"),
            "is_video": row.get("is_video"),
            "source_size": row.get("source_size"),
            "source_mtime_ns": row.get("source_mtime_ns"),
            "image_width": row.get("image_width"),
            "image_height": row.get("image_height"),
            "faces": row.get("faces"),
            "phash_hex": row.get("phash_hex"),
            "video_frame_phashes": normalize_video_frame_phashes(row.get("video_frame_phashes")),
            "skip_processing": row.get("skip_processing"),
            "dedupe_match_file": row.get("dedupe_match_file"),
            "dedupe_similarity_pct": row.get("dedupe_similarity_pct"),
            "cross_media_matches": normalize_cross_media_matches(row.get("cross_media_matches")),
            "sift_match_file": row.get("sift_match_file"),
            "sift_match_score": row.get("sift_match_score"),
            "sift_match_inliers": row.get("sift_match_inliers"),
            "sift_match_good_matches": row.get("sift_match_good_matches"),
            "sift_match_inlier_ratio": row.get("sift_match_inlier_ratio"),
            "sift_match_checked": row.get("sift_match_checked"),
            "processing_error_stage": row.get("processing_error_stage"),
            "processing_error": row.get("processing_error"),
            "face_groups": row.get("face_groups"),
            "clip_groups": row.get("clip_groups"),
            "ocr_groups": row.get("ocr_groups"),
        }
    print_info(f"{format_stage_label('Stage 0f/8 Startup normalize DB rows')}: loaded {len(records)} DB rows")
    return records


def escape_sql(value: str) -> str:
    return value.replace("'", "''")


def sql_string_literal(value: str) -> str:
    return f"'{escape_sql(value)}'"


def upsert_record(table, records: dict[str, dict[str, Any]], record: dict[str, Any]) -> None:
    file_name = record["file_name"]
    upsert_records_to_table(table, [record])
    records[file_name] = record


def upsert_records_to_table(table, batch: list[dict[str, Any]]) -> None:
    if not batch:
        return
    batch_by_file: dict[str, dict[str, Any]] = {}
    for record in batch:
        # LanceDB merge_insert rejects multiple source rows for the same key.
        # Keep the newest queued version of each row when a stage touches it twice.
        batch_by_file[record["file_name"]] = record
    batch = list(batch_by_file.values())
    arrow_batch = pa.Table.from_pylist(batch, schema=SCHEMA)
    if hasattr(table, "merge_insert"):
        (
            table.merge_insert("file_name")
            .use_index(True)
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(arrow_batch)
        )
        return

    for record in batch:
        table.delete(f"file_name = '{escape_sql(record['file_name'])}'")
    table.add(arrow_batch)


def upsert_records_batch(
    table,
    records: dict[str, dict[str, Any]],
    batch: list[dict[str, Any]],
) -> None:
    if not batch:
        return
    upsert_records_to_table(table, batch)
    for record in {record["file_name"]: record for record in batch}.values():
        records[record["file_name"]] = record


def bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def clamp_pil_max_side(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.BICUBIC)


def normalize_phash_hex(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    if len(candidate) != 16:
        return None
    try:
        int(candidate, 16)
    except ValueError:
        return None
    return candidate


def normalize_video_frame_phashes(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    normalized: list[dict[str, Any]] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        phash_hex = normalize_phash_hex(entry.get("phash_hex"))
        if phash_hex is None:
            continue
        try:
            timestamp_sec = round_timestamp(float(entry.get("timestamp_sec", 0.0)))
        except (TypeError, ValueError):
            timestamp_sec = 0.0
        normalized.append(
            {
                "timestamp_sec": float(timestamp_sec),
                "phash_hex": phash_hex,
            }
        )
    normalized.sort(key=lambda entry: (float(entry["timestamp_sec"]), str(entry["phash_hex"])))
    return normalized


def normalize_cross_media_matches(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    best_by_file: dict[str, dict[str, Any]] = {}
    for entry in value:
        if not isinstance(entry, dict):
            continue
        file_name = entry.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            continue
        similarity_raw = entry.get("similarity_pct")
        similarity_pct = float(similarity_raw) if isinstance(similarity_raw, (float, int)) else None
        normalized = {
            "file_name": file_name,
            "is_video": bool(entry.get("is_video")),
            "similarity_pct": round(float(similarity_pct), 3) if similarity_pct is not None else None,
        }
        existing = best_by_file.get(file_name)
        if existing is None:
            best_by_file[file_name] = normalized
            continue
        current_similarity = normalized["similarity_pct"]
        existing_similarity = existing["similarity_pct"]
        if current_similarity is None:
            continue
        if existing_similarity is None or float(current_similarity) > float(existing_similarity):
            best_by_file[file_name] = normalized
    normalized_list = list(best_by_file.values())
    normalized_list.sort(
        key=lambda entry: (
            -(float(entry["similarity_pct"]) if entry["similarity_pct"] is not None else -1.0),
            str(entry["file_name"]),
        )
    )
    return normalized_list


def frame_to_phash_bits(frame: np.ndarray) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    low = dct[:8, :8].flatten()
    median = float(np.median(low[1:]))
    bits = 0
    for coeff in low:
        bits = (bits << 1) | int(float(coeff) > median)
    return bits


def compute_phash_hex(image_path: Path) -> str:
    frame = read_image_bgr(image_path)
    bits = frame_to_phash_bits(frame)
    return f"{bits:016x}"


def compute_video_hash_hex(video_path: Path, max_samples: int = 16) -> str:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for hashing: {video_path}")

    frame_hashes: list[int] = []
    try:
        frame_count_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = max(0, frame_count_raw)
        sample_count = max(1, min(int(max_samples), frame_count if frame_count > 0 else int(max_samples)))

        if frame_count > 0:
            indices = sorted(set(np.linspace(0, frame_count - 1, num=sample_count, dtype=int).tolist()))
            for frame_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                frame_hashes.append(frame_to_phash_bits(frame))
        else:
            while len(frame_hashes) < sample_count:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame_hashes.append(frame_to_phash_bits(frame))
    finally:
        cap.release()

    if not frame_hashes:
        raise RuntimeError(f"Video hash failed: no decodable frames in {video_path}")

    bit_votes = np.zeros(64, dtype=np.int32)
    for value in frame_hashes:
        for bit_index in range(64):
            shift = 63 - bit_index
            bit_set = (value >> shift) & 1
            bit_votes[bit_index] += 1 if bit_set else -1

    merged = 0
    for vote in bit_votes:
        merged = (merged << 1) | int(vote >= 0)
    return f"{merged:016x}"


def hamming_distance_u64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def phash_similarity_pct(a_hex: str, b_hex: str) -> float:
    a = int(a_hex, 16)
    b = int(b_hex, 16)
    distance = hamming_distance_u64(a, b)
    return float((64 - distance) * 100.0 / 64.0)


def max_hamming_for_similarity(similarity_pct: float) -> int:
    threshold = min(max(float(similarity_pct), 0.0), 100.0)
    return int(np.floor((100.0 - threshold) * 64.0 / 100.0))


def hash_error_stage(is_video_items: bool) -> str:
    return "video_hash" if is_video_items else "image_phash"


def compact_error_message(exc: Exception, limit: int = 500) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


def mark_hash_failure_record(
    records: dict[str, dict[str, Any]],
    item: MediaItem,
    is_video_items: bool,
    exc: Exception,
) -> dict[str, Any]:
    base = records.get(
        item.file_name,
        default_record(item.file_name, item.is_video, item.collection_id),
    )
    base["collection_id"] = item.collection_id
    base["is_video"] = is_video_items
    base["phash_hex"] = None
    base["skip_processing"] = True
    base["dedupe_match_file"] = None
    base["dedupe_similarity_pct"] = None
    clear_sift_match_fields(base)
    base["sift_match_checked"] = None
    base["processing_error_stage"] = hash_error_stage(is_video_items)
    base["processing_error"] = compact_error_message(exc)
    base["faces"] = None
    base["face_groups"] = None
    base["clip_groups"] = None
    base["ocr_groups"] = None
    return base


def empty_face_groups(item: MediaItem) -> list[dict[str, Any]]:
    return [
        {
            "timestamp_sec": float(frame_ref.timestamp_sec),
            "face_embeddings": [],
        }
        for frame_ref in item.frame_refs
    ]


def should_skip_face_detection_by_shape(cfg: AppConfig, image_path: Path) -> bool:
    width, height = safe_image_dimensions(image_path)
    if width <= 0 or height <= 0:
        return False
    min_side = min(width, height)
    if min_side < cfg.face_min_side:
        return True
    aspect_ratio = max(width, height) / float(min_side)
    return min_side < cfg.face_skip_aspect_min_side and aspect_ratio >= cfg.face_skip_aspect_ratio


def mark_face_failure_record(
    records: dict[str, dict[str, Any]],
    item: MediaItem,
    message: str,
) -> dict[str, Any]:
    base = records.get(
        item.file_name,
        default_record(item.file_name, item.is_video, item.collection_id),
    )
    base["collection_id"] = item.collection_id
    base["is_video"] = item.is_video
    base["skip_processing"] = False
    base["face_groups"] = empty_face_groups(item)
    base["processing_error_stage"] = "faces"
    base["processing_error"] = message[:500]
    recompute_aggregate_fields(base)
    return base


def empty_ocr_groups(item: MediaItem) -> list[dict[str, Any]]:
    return [
        {
            "timestamp_sec": float(frame_ref.timestamp_sec),
            "text_detected": False,
            "text": "",
        }
        for frame_ref in item.frame_refs
    ]


def mark_paddle_failure_record(
    records: dict[str, dict[str, Any]],
    item: MediaItem,
    message: str,
) -> dict[str, Any]:
    base = records.get(
        item.file_name,
        default_record(item.file_name, item.is_video, item.collection_id),
    )
    base["collection_id"] = item.collection_id
    base["is_video"] = item.is_video
    base["skip_processing"] = False
    base["ocr_groups"] = empty_ocr_groups(item)
    base["processing_error_stage"] = "paddle_ocr"
    base["processing_error"] = message[:500]
    recompute_aggregate_fields(base)
    return base


def wipe_failed_paddle_rows_for_run(
    table,
    records: dict[str, dict[str, Any]],
    media_items: list[MediaItem],
) -> int:
    target_files = {item.file_name for item in media_items}
    if not target_files:
        return 0
    upsert_batch: list[dict[str, Any]] = []
    changed = 0
    for file_name in sorted(target_files):
        rec = records.get(file_name)
        if not rec:
            continue
        if rec.get("processing_error_stage") != "paddle_ocr":
            continue
        rec["ocr_groups"] = None
        rec["processing_error_stage"] = None
        rec["processing_error"] = None
        recompute_aggregate_fields(rec)
        append_stage_upsert(table, records, upsert_batch, rec)
        changed += 1
    upsert_records_batch(table, records, upsert_batch)
    return changed


def flush_hash_upserts(
    table,
    records: dict[str, dict[str, Any]],
    batch: list[dict[str, Any]],
) -> None:
    if not batch:
        return
    upsert_records_batch(table, records, batch)
    batch.clear()


def append_stage_upsert(
    table,
    records: dict[str, dict[str, Any]],
    batch: list[dict[str, Any]],
    record: dict[str, Any],
) -> None:
    batch.append(record)
    if len(batch) >= UPSERT_BATCH_SIZE:
        upsert_records_batch(table, records, batch)
        batch.clear()


def compute_item_hash(item: MediaItem, hash_func: Any) -> str:
    hash_hex = normalize_phash_hex(str(hash_func(item)))
    if hash_hex is None:
        raise RuntimeError("produced non-hex 64-bit hash")
    return hash_hex


def hash_gate_record_complete(
    record: dict[str, Any],
    item: MediaItem,
    is_video_items: bool,
) -> bool:
    if record.get("processing_error"):
        return record.get("skip_processing") is True
    if normalize_phash_hex(record.get("phash_hex")) is None:
        return False
    if record.get("is_video") is not is_video_items:
        return False
    if record.get("collection_id") != item.collection_id:
        return False
    if record.get("skip_processing") is False:
        return True
    if record.get("skip_processing") is True:
        return bool(record.get("dedupe_match_file"))
    return False


def should_add_record_to_hash_tree(record: dict[str, Any]) -> bool:
    if normalize_phash_hex(record.get("phash_hex")) is None:
        return False
    return not (record.get("skip_processing") is True and record.get("dedupe_match_file"))


def is_likely_thumbnail(path: Path) -> bool:
    lower_name = path.name.lower()
    if "thumb" in lower_name:
        return True
    return any(part.lower() in {"thumb", "thumbs", "thumbnails"} for part in path.parts)


def safe_file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def safe_image_dimensions(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as img:
            width, height = img.size
            return int(width), int(height)
    except Exception:
        return 0, 0


def cached_image_master_sort_key(
    item: MediaItem,
    records: dict[str, dict[str, Any]],
) -> tuple[int, int, int, int, str]:
    record = records.get(item.file_name) or {}
    width = int(record.get("image_width") or 0)
    height = int(record.get("image_height") or 0)
    max_side = max(width, height)
    area = width * height
    file_size = int(record.get("source_size") or 0)
    is_thumb = 1 if is_likely_thumbnail(item.source_path) else 0
    return (-max_side, -area, -file_size, is_thumb, item.file_name)


def cache_image_metadata(
    table,
    records: dict[str, dict[str, Any]],
    image_items: list[MediaItem],
    force: bool = False,
) -> set[str]:
    pending: list[tuple[MediaItem, os.stat_result]] = []
    for item in progress(image_items, desc="Stage 3a/8 check cached image metadata", unit="image"):
        try:
            stat = item.source_path.stat()
        except OSError:
            continue
        record = records.get(item.file_name)
        if not force and (
            record is not None
            and record.get("source_size") == stat.st_size
            and record.get("source_mtime_ns") == stat.st_mtime_ns
            and record.get("image_width") is not None
            and record.get("image_height") is not None
        ):
            continue
        pending.append((item, stat))

    if not pending:
        report_stage_complete("Stage 3a/8 cached image metadata", len(image_items), "images")
        return set()

    changed: set[str] = set()
    upsert_batch: list[dict[str, Any]] = []
    for item, stat in progress(pending, desc="Stage 3a/8 cache image metadata", unit="image"):
        width, height = safe_image_dimensions(item.source_path)
        record = records.get(
            item.file_name,
            default_record(item.file_name, False, item.collection_id),
        )
        record["collection_id"] = item.collection_id
        record["is_video"] = False
        record["source_size"] = int(stat.st_size)
        record["source_mtime_ns"] = int(stat.st_mtime_ns)
        record["image_width"] = int(width)
        record["image_height"] = int(height)
        append_stage_upsert(table, records, upsert_batch, record)
        changed.add(item.file_name)
    upsert_records_batch(table, records, upsert_batch)
    report_stage_complete("Stage 3a/8 cached image metadata", len(image_items), "images")
    return changed


def clear_processing_fields(record: dict[str, Any]) -> None:
    record["faces"] = None
    record["face_groups"] = None
    record["clip_groups"] = None
    record["ocr_groups"] = None
    clear_sift_match_fields(record)
    record["sift_match_checked"] = None
    if record.get("processing_error_stage") not in {"image_phash", "video_hash"}:
        record["processing_error_stage"] = None
        record["processing_error"] = None


def run_hash_gate_for_items(
    *,
    table,
    records: dict[str, dict[str, Any]],
    items: list[MediaItem],
    is_video_items: bool,
    similarity_pct: float,
    stage_label: str,
    hash_func: Any,
    hash_workers: int,
    apply_stage_label: str | None = None,
    before_apply: Any | None = None,
    force_hash: bool = False,
    force_apply: bool = False,
) -> set[str]:
    if not items:
        report_stage_complete(stage_label, 0, "files")
        report_stage_complete(apply_stage_label or f"{stage_label} apply", 0, "files")
        return set()

    max_distance = max_hamming_for_similarity(similarity_pct)
    current_names = {item.file_name for item in items}
    tree = HammingBkTree()
    for file_name, rec in records.items():
        if bool(rec.get("is_video")) != is_video_items:
            continue
        if file_name in current_names:
            continue
        hex_hash = normalize_phash_hex(rec.get("phash_hex"))
        if hex_hash is None:
            continue
        if not should_add_record_to_hash_tree(rec):
            continue
        tree.add(int(hex_hash, 16), file_name)

    changed: set[str] = set()
    active_items: list[MediaItem] = []
    hash_results: dict[str, tuple[str | None, Exception | None]] = {}
    futures: dict[Future[str], MediaItem] = {}

    with ThreadPoolExecutor(max_workers=max(1, hash_workers)) as executor:
        for item in items:
            base = records.get(
                item.file_name,
                default_record(item.file_name, item.is_video, item.collection_id),
            )
            if not force_hash and base.get("skip_processing") is True and base.get("processing_error"):
                continue
            existing_hash = normalize_phash_hex(base.get("phash_hex"))
            if (
                not force_hash
                and not force_apply
                and existing_hash is not None
                and hash_gate_record_complete(base, item, is_video_items)
            ):
                if should_add_record_to_hash_tree(base):
                    tree.add(int(existing_hash, 16), item.file_name)
                continue
            active_items.append(item)
            if existing_hash is not None and not force_hash:
                hash_results[item.file_name] = (existing_hash, None)
                continue
            futures[executor.submit(compute_item_hash, item, hash_func)] = item

        if futures:
            for future in progress(
                as_completed(futures),
                desc=stage_label,
                unit="file",
                total=len(futures),
            ):
                item = futures[future]
                try:
                    hash_results[item.file_name] = (future.result(), None)
                except Exception as exc:
                    hash_results[item.file_name] = (None, exc)

    if not futures:
        report_stage_complete(stage_label, len(items), "files")
    if before_apply is not None:
        changed |= before_apply()

    upsert_batch: list[dict[str, Any]] = []
    if not active_items:
        report_stage_complete(apply_stage_label or f"{stage_label} apply", len(items), "files")
    for item in progress(active_items, desc=apply_stage_label or f"{stage_label} apply", unit="file"):
        base = records.get(
            item.file_name,
            default_record(item.file_name, item.is_video, item.collection_id),
        )
        existing_hash, hash_error = hash_results.get(item.file_name, (None, None))
        if hash_error is not None:
            print_error(f"[hash] failed: {item.file_name}: {hash_error}")
            failed = mark_hash_failure_record(records, item, is_video_items, hash_error)
            upsert_batch.append(failed)
            changed.add(item.file_name)
            if len(upsert_batch) >= UPSERT_BATCH_SIZE:
                flush_hash_upserts(table, records, upsert_batch)
            continue
        if existing_hash is None:
            exc = RuntimeError("hash result missing")
            print_error(f"[hash] failed: {item.file_name}: {exc}")
            failed = mark_hash_failure_record(records, item, is_video_items, exc)
            upsert_batch.append(failed)
            changed.add(item.file_name)
            if len(upsert_batch) >= UPSERT_BATCH_SIZE:
                flush_hash_upserts(table, records, upsert_batch)
            continue

        existing_hash_int = int(existing_hash, 16)
        best = tree.find_best(existing_hash_int, max_distance=max_distance)
        matched_file = best[0] if best is not None else None
        similarity = float((64 - best[1]) * 100.0 / 64.0) if best is not None else None
        has_stage_data = any(
            base.get(key) is not None for key in ("face_groups", "clip_groups", "ocr_groups")
        )
        should_skip = matched_file is not None and not has_stage_data

        mutated = False
        if base.get("is_video") is not is_video_items:
            base["is_video"] = is_video_items
            mutated = True
        if base.get("collection_id") != item.collection_id:
            base["collection_id"] = item.collection_id
            mutated = True
        if base.get("phash_hex") != existing_hash:
            base["phash_hex"] = existing_hash
            mutated = True
        if base.get("processing_error_stage") is not None:
            base["processing_error_stage"] = None
            mutated = True
        if base.get("processing_error") is not None:
            base["processing_error"] = None
            mutated = True

        if should_skip:
            if base.get("skip_processing") is not True:
                base["skip_processing"] = True
                mutated = True
            if base.get("dedupe_match_file") != matched_file:
                base["dedupe_match_file"] = matched_file
                mutated = True
            rounded_similarity = round(float(similarity), 3) if similarity is not None else None
            if base.get("dedupe_similarity_pct") != rounded_similarity:
                base["dedupe_similarity_pct"] = rounded_similarity
                mutated = True
            if base.get("face_groups") is not None:
                base["face_groups"] = None
                mutated = True
            if base.get("clip_groups") is not None:
                base["clip_groups"] = None
                mutated = True
            if base.get("ocr_groups") is not None:
                base["ocr_groups"] = None
                mutated = True
            if base.get("faces") is not None:
                base["faces"] = None
                mutated = True
            if any(
                base.get(key) is not None
                for key in (
                    "sift_match_file",
                    "sift_match_score",
                    "sift_match_inliers",
                    "sift_match_good_matches",
                    "sift_match_inlier_ratio",
                )
            ):
                clear_sift_match_fields(base)
                mutated = True
            if base.get("sift_match_checked") is not None:
                base["sift_match_checked"] = None
                mutated = True
        else:
            if base.get("skip_processing") is not False:
                base["skip_processing"] = False
                mutated = True
            if base.get("dedupe_match_file") is not None:
                base["dedupe_match_file"] = None
                mutated = True
            if base.get("dedupe_similarity_pct") is not None:
                base["dedupe_similarity_pct"] = None
                mutated = True
            if base.get("sift_match_checked") is not None:
                base["sift_match_checked"] = None
                mutated = True

        if mutated:
            recompute_aggregate_fields(base)
            upsert_batch.append(base)
            changed.add(item.file_name)
            if len(upsert_batch) >= UPSERT_BATCH_SIZE:
                flush_hash_upserts(table, records, upsert_batch)

        if not should_skip:
            tree.add(existing_hash_int, item.file_name)

    flush_hash_upserts(table, records, upsert_batch)
    return changed


def run_phash_gate_stage(
    cfg: AppConfig,
    table,
    records: dict[str, dict[str, Any]],
    media_items: list[MediaItem],
    video_paths: list[Path],
    extracted_video_frame_map: dict[Path, list[FrameRef]],
) -> set[str]:
    image_items = [item for item in media_items if not item.is_video]
    metadata_changed = cache_image_metadata(
        table,
        records,
        image_items,
        force=should_rerun_stage(cfg, "3a"),
    )
    image_items.sort(key=lambda item: cached_image_master_sort_key(item, records))
    if not image_items:
        report_stage_complete("Stage 3b/8 pHash images", 0, "files")
        report_stage_complete("Stage 3d/8 apply image pHash groups", 0, "files")
        return metadata_changed | run_video_frame_phash_stage(
            cfg,
            table,
            records,
            video_paths,
            extracted_video_frame_map,
        )
    return metadata_changed | run_hash_gate_for_items(
        table=table,
        records=records,
        items=image_items,
        is_video_items=False,
        similarity_pct=cfg.phash_skip_similarity_pct,
        stage_label="Stage 3b/8 pHash images",
        hash_func=lambda item: compute_phash_hex(item.source_path),
        hash_workers=cfg.hash_workers,
        apply_stage_label="Stage 3d/8 apply image pHash groups",
        before_apply=lambda: run_video_frame_phash_stage(
            cfg,
            table,
            records,
            video_paths,
            extracted_video_frame_map,
        ),
        force_hash=should_rerun_stage(cfg, "3b"),
        force_apply=should_rerun_stage(cfg, "3d"),
    )


def repair_image_masters(
    cfg: AppConfig,
    table,
    records: dict[str, dict[str, Any]],
    media_items: list[MediaItem],
) -> tuple[set[str], set[str]]:
    image_items = [item for item in media_items if not item.is_video]
    metadata_changed = cache_image_metadata(table, records, image_items)
    image_items.sort(key=lambda item: cached_image_master_sort_key(item, records))
    if not image_items:
        return metadata_changed, set()

    max_distance = max_hamming_for_similarity(cfg.phash_skip_similarity_pct)
    tree = HammingBkTree()
    changed: set[str] = set(metadata_changed)
    cleared_processing: set[str] = set()
    masters = 0
    demoted = 0
    retargeted = 0
    upsert_batch: list[dict[str, Any]] = []

    for item in progress(image_items, desc="Repair image pHash masters", unit="file"):
        base = records.get(item.file_name)
        if not base:
            continue
        if bool(base.get("is_video")):
            continue
        if base.get("skip_processing") is True and base.get("processing_error"):
            continue
        existing_hash = normalize_phash_hex(base.get("phash_hex"))
        if existing_hash is None:
            continue

        existing_hash_int = int(existing_hash, 16)
        best = tree.find_best(existing_hash_int, max_distance=max_distance)
        mutated = False
        if best is None:
            masters += 1
            if base.get("skip_processing") is not False:
                base["skip_processing"] = False
                mutated = True
            if base.get("dedupe_match_file") is not None:
                base["dedupe_match_file"] = None
                mutated = True
            if base.get("dedupe_similarity_pct") is not None:
                base["dedupe_similarity_pct"] = None
                mutated = True
            if base.get("collection_id") != item.collection_id:
                base["collection_id"] = item.collection_id
                mutated = True
            if base.get("is_video") is not False:
                base["is_video"] = False
                mutated = True
            if any(
                base.get(key) is not None
                for key in (
                    "sift_match_file",
                    "sift_match_score",
                    "sift_match_inliers",
                    "sift_match_good_matches",
                    "sift_match_inlier_ratio",
                )
            ):
                clear_sift_match_fields(base)
                mutated = True
            if base.get("sift_match_checked") is not None:
                base["sift_match_checked"] = None
                mutated = True
            tree.add(existing_hash_int, item.file_name)
        else:
            matched_file, distance = best
            rounded_similarity = round(float((64 - distance) * 100.0 / 64.0), 3)
            was_master = base.get("skip_processing") is not True
            old_match = base.get("dedupe_match_file")
            if was_master:
                demoted += 1
            elif old_match != matched_file:
                retargeted += 1

            if base.get("skip_processing") is not True:
                base["skip_processing"] = True
                mutated = True
            if base.get("dedupe_match_file") != matched_file:
                base["dedupe_match_file"] = matched_file
                mutated = True
            if base.get("dedupe_similarity_pct") != rounded_similarity:
                base["dedupe_similarity_pct"] = rounded_similarity
                mutated = True
            if base.get("collection_id") != item.collection_id:
                base["collection_id"] = item.collection_id
                mutated = True
            if base.get("is_video") is not False:
                base["is_video"] = False
                mutated = True
            if any(
                base.get(key) is not None
                for key in (
                    "sift_match_file",
                    "sift_match_score",
                    "sift_match_inliers",
                    "sift_match_good_matches",
                    "sift_match_inlier_ratio",
                )
            ):
                clear_sift_match_fields(base)
                mutated = True
            if base.get("sift_match_checked") is not None:
                base["sift_match_checked"] = None
                mutated = True

            had_processing = any(
                base.get(key) is not None for key in ("faces", "face_groups", "clip_groups", "ocr_groups")
            )
            clear_processing_fields(base)
            if had_processing:
                cleared_processing.add(item.file_name)
                mutated = True

        if mutated:
            recompute_aggregate_fields(base)
            upsert_batch.append(base)
            changed.add(item.file_name)
            if len(upsert_batch) >= UPSERT_BATCH_SIZE:
                upsert_records_batch(table, records, upsert_batch)
                upsert_batch.clear()

    upsert_records_batch(table, records, upsert_batch)
    print(
        "[repair] image masters: "
        f"{masters} masters, {demoted} demoted old masters, "
        f"{retargeted} retargeted duplicates, {len(cleared_processing)} rows had processing cleared"
    )
    return changed, cleared_processing


def run_video_hash_gate_stage(
    cfg: AppConfig,
    table,
    records: dict[str, dict[str, Any]],
    video_paths: list[Path],
) -> set[str]:
    video_items = sorted(
        [
            MediaItem(
                source_path=path,
                file_name=scoped_file_name(cfg, path),
                collection_id=cfg.collection_id,
                is_video=True,
                frame_refs=[],
            )
            for path in video_paths
        ],
        key=lambda item: item.file_name,
    )
    return run_hash_gate_for_items(
        table=table,
        records=records,
        items=video_items,
        is_video_items=True,
        similarity_pct=cfg.video_hash_skip_similarity_pct,
        stage_label="Stage 1a/8 VideoHash (videos)",
        hash_func=lambda item: compute_video_hash_hex(item.source_path),
        hash_workers=cfg.hash_workers,
        apply_stage_label="Stage 1b/8 VideoHash apply",
        force_hash=should_rerun_stage(cfg, "1a"),
        force_apply=should_rerun_stage(cfg, "1b"),
    )


class FaceEmbedder:
    def __init__(
        self,
        insightface_root: Path,
        det_thresh: float = FACE_DET_THRESHOLD_DEFAULT,
        det_size: int = FACE_DET_SIZE_DEFAULT,
        fallback_det_size: int = FACE_FALLBACK_DET_SIZE_DEFAULT,
        dedupe_cosine: float = FACE_DEDUP_COSINE_DEFAULT,
    ) -> None:
        from insightface.app import FaceAnalysis

        insightface_root.mkdir(parents=True, exist_ok=True)
        self.det_thresh = float(det_thresh)
        self.det_size = int(det_size)
        self.fallback_det_size = int(fallback_det_size)
        self.dedupe_cosine = float(dedupe_cosine)
        self.current_det_size = self.det_size
        self.app = FaceAnalysis(
            name="buffalo_l",
            root=str(insightface_root),
            allowed_modules=["detection", "recognition"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(self.det_size, self.det_size), det_thresh=self.det_thresh)
        self._assert_cuda(self.app)

    def _assert_cuda(self, app: Any) -> None:
        for name, model in app.models.items():
            session = getattr(model, "session", None)
            if session is None:
                continue
            providers = session.get_providers()
            if "CUDAExecutionProvider" not in providers:
                raise RuntimeError(f"InsightFace {name} is not running on CUDAExecutionProvider.")

    def _detect_with_app(self, app: Any, frame: np.ndarray) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for face in app.get(frame):
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

    def _prepare_det_size(self, det_size: int) -> None:
        if self.current_det_size == det_size:
            return
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size), det_thresh=self.det_thresh)
        self.current_det_size = det_size
        self._assert_cuda(self.app)

    def _dedupe_embeddings(
        self, existing: list[list[float]], incoming: list[list[float]], cosine_thresh: float
    ) -> list[list[float]]:
        if not incoming:
            return existing
        merged: list[list[float]] = [list(vec) for vec in existing]
        existing_np = [np.asarray(vec, dtype=np.float32) for vec in merged]
        for vec in incoming:
            vec_np = np.asarray(vec, dtype=np.float32)
            if any(float(np.dot(vec_np, prior)) >= cosine_thresh for prior in existing_np):
                continue
            merged.append(vec)
            existing_np.append(vec_np)
        return merged

    def _rotation_variants(self, frame: np.ndarray) -> list[np.ndarray]:
        return [
            cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(frame, cv2.ROTATE_180),
            cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]

    def detect_and_embed_frame(self, frame: np.ndarray) -> list[list[float]]:
        self._prepare_det_size(self.det_size)
        embeddings = self._detect_with_app(self.app, frame)
        if embeddings:
            return embeddings

        for rotated in self._rotation_variants(frame):
            rotated_embeddings = self._detect_with_app(self.app, rotated)
            embeddings = self._dedupe_embeddings(embeddings, rotated_embeddings, self.dedupe_cosine)
        if embeddings or self.fallback_det_size <= self.det_size:
            return embeddings

        self._prepare_det_size(self.fallback_det_size)
        fallback_embeddings = self._detect_with_app(self.app, frame)
        embeddings = self._dedupe_embeddings(embeddings, fallback_embeddings, self.dedupe_cosine)
        if embeddings:
            return embeddings
        for rotated in self._rotation_variants(frame):
            rotated_embeddings = self._detect_with_app(self.app, rotated)
            embeddings = self._dedupe_embeddings(embeddings, rotated_embeddings, self.dedupe_cosine)
        return embeddings


def face_worker_main(
    insightface_root: str,
    det_thresh: float,
    det_size: int,
    fallback_det_size: int,
    dedupe_cosine: float,
    conn: Any,
) -> None:
    # Ensure child process resolves CUDA libs from this venv first.
    apply_cuda_library_path(os.environ)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                embedder = FaceEmbedder(
                    Path(insightface_root),
                    det_thresh=det_thresh,
                    det_size=det_size,
                    fallback_det_size=fallback_det_size,
                    dedupe_cosine=dedupe_cosine,
                )
    except Exception as exc:
        conn.send(("init_error", compact_error_message(exc)))
        return

    conn.send(("ready",))
    while True:
        job = conn.recv()
        if job is None:
            return

        job_id, image_path = job
        try:
            frame = read_image_bgr(Path(image_path))
            embeddings = embedder.detect_and_embed_frame(frame)
        except Exception as exc:
            conn.send(("error", job_id, compact_error_message(exc)))
            continue
        conn.send(("ok", job_id, embeddings))


class FaceWorkerPipeError(RuntimeError):
    pass


class FaceWorker:
    def __init__(
        self,
        insightface_root: Path,
        det_thresh: float,
        det_size: int,
        fallback_det_size: int,
        dedupe_cosine: float,
        init_timeout_seconds: int = 300,
    ) -> None:
        self.insightface_root = insightface_root
        self.det_thresh = float(det_thresh)
        self.det_size = int(det_size)
        self.fallback_det_size = int(fallback_det_size)
        self.dedupe_cosine = float(dedupe_cosine)
        self.init_timeout_seconds = init_timeout_seconds
        self.process: subprocess.Popen[str] | None = None
        self.job_counter = 0
        self.start()

    def start(self) -> None:
        worker_path = Path(__file__).with_name("face_worker.py")
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env.setdefault("CUDA_MODULE_LOADING", "LAZY")
        lib_dirs: list[str] = []
        preload_libs: list[str] = []
        for root in site.getsitepackages():
            nvidia_root = Path(root) / "nvidia"
            if nvidia_root.is_dir():
                lib_dirs.extend(str(path) for path in sorted(nvidia_root.glob("*/lib")) if path.is_dir())
                preload_libs.extend(str(path) for path in sorted(nvidia_root.glob("*/lib/libcublasLt.so.12")))
                preload_libs.extend(str(path) for path in sorted(nvidia_root.glob("*/lib/libcublas.so.12")))
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        existing_parts = [part for part in existing_ld.split(os.pathsep) if part]
        ordered_ld = list(dict.fromkeys(lib_dirs + existing_parts))
        if ordered_ld:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(ordered_ld)
        existing_preload = env.get("LD_PRELOAD", "")
        existing_preload_parts = [part for part in existing_preload.split() if part]
        ordered_preload = list(dict.fromkeys(preload_libs + existing_preload_parts))
        if ordered_preload:
            env["LD_PRELOAD"] = " ".join(ordered_preload)
        # Keep this worker clean: do not import main.py, torch, paddle, lancedb, or pyarrow.
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(worker_path),
                "--insightface-root",
                str(self.insightface_root),
                "--det-threshold",
                str(self.det_thresh),
                "--det-size",
                str(self.det_size),
                "--fallback-det-size",
                str(self.fallback_det_size),
                "--dedupe-cosine",
                str(self.dedupe_cosine),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        start_worker_stderr_forwarder(self.process)
        ready = self._read_json_line(self.init_timeout_seconds)
        if ready is None:
            self.terminate()
            raise RuntimeError("Timed out starting InsightFace worker.")
        if ready.get("status") == "ready":
            return
        self.terminate()
        raise RuntimeError(f"Unexpected InsightFace worker status: {ready}")

    def terminate(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
        if process is not None:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
        self.process = None

    def restart(self) -> None:
        self.terminate()
        self.start()

    def _ensure_alive(self) -> None:
        if self.process is None or self.process.poll() is not None:
            self.restart()

    def close(self) -> None:
        if self.process is not None and self.process.stdin is not None and self.process.poll() is None:
            try:
                self.process.stdin.close()
            except Exception:
                pass
        self.terminate()

    def _read_json_line(self, timeout_seconds: float) -> dict[str, Any] | None:
        process = self.process
        if process is None or process.stdout is None:
            return None
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            if process.poll() is not None:
                return None
            ready, _, _ = select.select([process.stdout], [], [], remaining)
            if not ready:
                continue
            line = process.stdout.readline()
            if not line:
                return None
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                print(f"[face-worker] {line.strip()}")
                continue
            if isinstance(value, dict):
                return value

    def detect(self, image_path: Path, timeout_seconds: int) -> list[list[float]]:
        last_exc: Exception | None = None
        for attempt in range(2):
            self._ensure_alive()
            assert self.process is not None
            assert self.process.stdin is not None

            self.job_counter += 1
            job_id = self.job_counter
            try:
                self.process.stdin.write(json.dumps({"id": job_id, "image_path": str(image_path)}) + "\n")
                self.process.stdin.flush()
                message = self._read_json_line(timeout_seconds)
                if message is None:
                    self.terminate()
                    raise TimeoutError(f"Face detection exceeded {timeout_seconds}s for {image_path}")
            except TimeoutError:
                raise
            except (BrokenPipeError, EOFError, OSError) as exc:
                last_exc = exc
                exit_code = self.process.poll() if self.process is not None else None
                if attempt == 0:
                    self.restart()
                    continue
                raise FaceWorkerPipeError(
                    f"InsightFace worker pipe failure for {image_path}: {exc}; worker_exit_code={exit_code}"
                ) from exc

            if message.get("ok") is True and message.get("id") == job_id:
                return message.get("embeddings") or []
            if message.get("ok") is False and message.get("id") == job_id:
                raise RuntimeError(str(message.get("error") or "Face worker error"))
            raise RuntimeError(f"Unexpected InsightFace worker response: {message}")
        if last_exc is not None:
            exit_code = self.process.poll() if self.process is not None else None
            raise FaceWorkerPipeError(
                f"InsightFace worker pipe failure for {image_path}: {last_exc}; worker_exit_code={exit_code}"
            ) from last_exc
        raise FaceWorkerPipeError(f"InsightFace worker failed unexpectedly for {image_path}")


def normalize_clip_model_name(model_name: str) -> str:
    value = model_name.strip()
    if value in {"clip-ViT-L-16-SigLIP2-384", "ViT-L-16-SigLIP2-384"}:
        return DEFAULT_CLIP_MODEL
    if value.startswith("timm/") or value.startswith("laion/"):
        return f"hf-hub:{value}"
    return value


class ClipEmbedder:
    def __init__(self, model_name: str, batch_size: int, device: str = "cuda") -> None:
        import open_clip

        self.batch_size = batch_size
        requested_device = device.lower()
        if requested_device not in {"cuda", "cpu"}:
            raise ValueError(f"unsupported CLIP device: {device}")
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CLIP/SigLIP image embeddings require CUDA but CUDA is unavailable.")
        self.device = torch.device(requested_device)
        resolved_model_name = normalize_clip_model_name(model_name)
        self.model, _, self.transform = open_clip.create_model_and_transforms(
            resolved_model_name,
            device=self.device,
        )
        self.model.eval()

    def embed_frames(self, frames_bgr: list[np.ndarray]) -> list[list[float]]:
        images = [
            clamp_pil_max_side(bgr_to_pil(frame), CLIP_PREPROCESS_MAX_SIDE)
            for frame in frames_bgr
        ]
        if not images:
            return []
        all_vectors: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch_images = images[start : start + self.batch_size]
                batch = torch.stack([self.transform(image) for image in batch_images]).to(self.device)
                features = self.model.encode_image(batch)
                if features.ndim > 2:
                    features = features.flatten(2).mean(dim=-1)
                features = torch.nn.functional.normalize(features.float(), dim=-1)
                all_vectors.append(features.detach().cpu().numpy().astype(np.float32))
                del batch, features
        vectors_np = np.concatenate(all_vectors, axis=0)
        return [vec.tolist() for vec in vectors_np]


class PaddleTextDetector:
    def __init__(self, python_exe: Path, model_name: str, device: str) -> None:
        self._next_job_id = 1
        worker_path = Path(__file__).resolve().with_name("paddle_worker.py")
        cmd = [
            str(python_exe),
            str(worker_path),
            "--model-name",
            model_name,
            "--device",
            device,
        ]
        env = build_paddle_worker_env(python_exe)
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
        start_worker_stderr_forwarder(self.process)
        ready = self._read_json_line(PADDLE_WORKER_READY_TIMEOUT_SECONDS)
        if ready.get("status") != "ready":
            diag = ""
            err_text = str(ready.get("error") or "")
            if "No module named" in err_text:
                probe = self._probe_worker_imports(python_exe, env)
                if probe:
                    diag = f"; import_probe={probe}"
            raise RuntimeError(f"Paddle worker did not report ready: {ready}{diag}")

    @staticmethod
    def _probe_worker_imports(python_exe: Path, env: dict[str, str]) -> str:
        script = """
import importlib
import json
import sys

out = {"python_exe": sys.executable, "version": sys.version.split()[0], "mods": {}}
for module_name in ("paddle", "paddleocr", "cv2"):
    try:
        importlib.import_module(module_name)
        out["mods"][module_name] = "ok"
    except Exception as exc:  # pragma: no cover - diagnostic only
        out["mods"][module_name] = f"err:{type(exc).__name__}:{exc}"
print(json.dumps(out, ensure_ascii=True))
""".strip()
        try:
            proc = subprocess.run(
                [str(python_exe), "-c", script],
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except Exception as exc:
            return f"probe-failed:{exc}"
        summary = {
            "rc": proc.returncode,
            "stdout": (proc.stdout or "").strip()[:500],
            "stderr": (proc.stderr or "").strip()[:500],
        }
        return json.dumps(summary, ensure_ascii=True)

    def _read_json_line(self, timeout_seconds: float) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("Paddle worker stdout is unavailable.")
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Paddle worker response timed out.")
            if self.process.poll() is not None:
                raise RuntimeError(f"Paddle worker exited with code {self.process.returncode}")
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not ready:
                continue
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Paddle worker closed stdout.")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                print(f"[paddle-worker] {line.strip()}")
                continue
            if isinstance(payload, dict):
                return payload

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
        except Exception:
            pass
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)

    def has_text(self, image_path: Path, max_side: int) -> bool:
        if self.process.poll() is not None:
            raise RuntimeError(f"Paddle worker exited with code {self.process.returncode}")
        if self.process.stdin is None:
            raise RuntimeError("Paddle worker stdin is unavailable.")
        job_id = self._next_job_id
        self._next_job_id += 1
        self.process.stdin.write(
            json.dumps(
                {
                    "id": job_id,
                    "image_path": str(image_path),
                    "max_side": int(max_side),
                },
                ensure_ascii=True,
            )
            + "\n"
        )
        self.process.stdin.flush()
        response = self._read_json_line(PADDLE_WORKER_JOB_TIMEOUT_SECONDS)
        if response.get("id") != job_id:
            raise RuntimeError(f"Paddle worker response id mismatch: expected {job_id}, got {response.get('id')}")
        if response.get("ok") is not True:
            raise RuntimeError(str(response.get("error") or "Paddle worker failed"))
        return bool(response.get("text_detected"))


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


def polys_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, np.ndarray):
        return value.shape[0] > 0
    try:
        return len(value) > 0
    except TypeError:
        return False


class EasyOcrWorker:
    def __init__(
        self,
        langs: Sequence[str],
        batch_size: int,
        canvas_size: int,
        gpu: bool,
    ) -> None:
        self._next_job_id = 1
        worker_path = Path(__file__).resolve().with_name("easyocr_worker.py")
        cmd = [
            sys.executable,
            str(worker_path),
            "--langs",
            ",".join(langs),
            "--batch-size",
            str(batch_size),
            "--canvas-size",
            str(canvas_size),
            "--device",
            "cuda" if gpu else "cpu",
        ]
        env = build_easyocr_worker_env()
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
        start_worker_stderr_forwarder(self.process)
        ready = self._read_json_line(EASYOCR_WORKER_READY_TIMEOUT_SECONDS)
        if ready.get("status") != "ready":
            raise RuntimeError(f"EasyOCR worker did not report ready: {ready}")

    def is_alive(self) -> bool:
        return self.process.poll() is None

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
        except Exception:
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)

    def extract_text(self, image_path: Path, max_side: int) -> str:
        job_id = self._next_job_id
        self._next_job_id += 1
        self._write_json_line(
            {
                "id": job_id,
                "image_path": str(image_path),
                "max_side": int(max_side),
            }
        )
        while True:
            response = self._read_json_line(EASYOCR_WORKER_JOB_TIMEOUT_SECONDS)
            if response.get("id") != job_id:
                continue
            if response.get("ok") is True:
                return str(response.get("text") or "")
            raise RuntimeError(str(response.get("error") or "EasyOCR worker failed"))

    def _write_json_line(self, payload: dict[str, Any]) -> None:
        if self.process.stdin is None:
            raise RuntimeError("EasyOCR worker stdin is closed")
        try:
            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError("EasyOCR worker exited while receiving a request") from exc

    def _read_json_line(self, timeout_seconds: float) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("EasyOCR worker stdout is closed")
        deadline = time.monotonic() + timeout_seconds
        while True:
            return_code = self.process.poll()
            if return_code is not None:
                raise RuntimeError(f"EasyOCR worker exited with code {return_code}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for EasyOCR worker after {timeout_seconds:.0f}s")
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not ready:
                continue
            line = self.process.stdout.readline()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                print(f"[easyocr-worker] {line.strip()}")
                continue
            if isinstance(value, dict):
                return value


def build_easyocr_worker_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    apply_cuda_library_path(env)
    return env


def is_cuda_oom_error(exc: Exception) -> bool:
    message = compact_error_message(exc).lower()
    return "cuda out of memory" in message or "out of memory" in message


def is_gpu_runtime_error(exc: Exception) -> bool:
    message = compact_error_message(exc).lower()
    markers = (
        "cuda",
        "cublas",
        "cudnn",
        "nvidia",
        "gpu",
        "device-side assert",
        "illegal memory access",
        "driver shutting down",
        "driver initialization failed",
        "unspecified launch failure",
        "out of memory",
    )
    return any(marker in message for marker in markers)


def reduced_easyocr_retry_settings(cfg: AppConfig) -> tuple[int, int, int]:
    reduced_batch_size = 1
    reduced_max_side = min(cfg.easyocr_max_side, 1280)
    reduced_canvas_size = min(cfg.easyocr_canvas_size, 1280)
    if reduced_max_side >= cfg.easyocr_max_side and cfg.easyocr_max_side > 512:
        reduced_max_side = max(512, int(round(cfg.easyocr_max_side * 0.75)))
    if reduced_canvas_size >= cfg.easyocr_canvas_size and cfg.easyocr_canvas_size > 512:
        reduced_canvas_size = max(512, int(round(cfg.easyocr_canvas_size * 0.75)))
    return reduced_batch_size, reduced_canvas_size, reduced_max_side


def _discover_cuda_library_dirs_for_python(python_exe: Path) -> list[str]:
    script = (
        "import json, site; from pathlib import Path; "
        "dirs=[]; "
        "[(dirs.extend(str(p) for p in sorted((Path(root)/'nvidia').glob('*/lib')) if p.is_dir())) "
        "for root in site.getsitepackages()]; "
        "print(json.dumps(list(dict.fromkeys(dirs))))"
    )
    try:
        result = subprocess.run(
            [str(python_exe), "-c", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return []
    try:
        payload = json.loads(result.stdout.strip() or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(part) for part in payload if isinstance(part, str) and part]


def build_paddle_worker_env(python_exe: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    env.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    paddle_python_bin = str(python_exe.parent)
    paddle_env_root = str(python_exe.parent.parent)
    env["VIRTUAL_ENV"] = paddle_env_root
    existing_path = env.get("PATH", "")
    path_parts = [part for part in existing_path.split(os.pathsep) if part]
    env["PATH"] = os.pathsep.join([paddle_python_bin, *[p for p in path_parts if p != paddle_python_bin]])
    # Keep Paddle isolated from this process' CUDA path bootstrap so it can
    # resolve against the libraries installed in --paddle-python's environment.
    main_cuda_dirs = set(discover_cuda_library_dirs())
    main_prefix = str(Path(sys.prefix).resolve())
    worker_cuda_dirs = _discover_cuda_library_dirs_for_python(python_exe)
    existing = env.get("LD_LIBRARY_PATH", "")
    existing_parts = [part for part in existing.split(os.pathsep) if part]
    filtered_existing = [
        part
        for part in existing_parts
        if part not in main_cuda_dirs and not str(Path(part).resolve()).startswith(main_prefix)
    ]
    ordered = list(dict.fromkeys(worker_cuda_dirs + filtered_existing))
    if ordered:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(ordered)
    else:
        env.pop("LD_LIBRARY_PATH", None)
    return env


def discover_cuda_library_dirs() -> list[str]:
    return _cuda_library_dirs()


def apply_cuda_library_path(env: dict[str, str]) -> None:
    lib_dirs = discover_cuda_library_dirs()
    existing = env.get("LD_LIBRARY_PATH", "")
    existing_parts = [part for part in existing.split(os.pathsep) if part]
    ordered = list(dict.fromkeys(lib_dirs + existing_parts))
    if ordered:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(ordered)


class TextEmbedder:
    def __init__(self, model_name: str, batch_size: int, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.batch_size = batch_size
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("OCR text embedding model was configured for CUDA but CUDA is unavailable.")
        with TimedStep(f"load OCR text embedding model {model_name} on {device}"):
            self.model = SentenceTransformer(model_name, device=device)
        if getattr(self.model, "device", None) is None:
            raise RuntimeError("Could not verify OCR text embedding model device.")
        if self.model.device.type != device:
            raise RuntimeError(f"OCR text embedding model did not initialize on {device}.")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        chunks: list[np.ndarray] = []
        ranges = range(0, len(texts), self.batch_size)
        iterator = progress(ranges, desc="Stage 8b/8 OCR text search index embed text", unit="batch") if len(texts) > self.batch_size else ranges
        for start in iterator:
            batch_texts = texts[start : start + self.batch_size]
            vectors = self.model.encode(
                batch_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            chunks.append(np.asarray(vectors, dtype=np.float32))
        vectors_np = np.concatenate(chunks, axis=0)
        if vectors_np.ndim == 1:
            vectors_np = np.expand_dims(vectors_np, axis=0)
        return [vec.tolist() for vec in vectors_np]


def mark_stale_face_status_if_needed(
    cfg: AppConfig,
    table,
    records: dict[str, dict[str, Any]],
    media_items: list[MediaItem],
) -> None:
    status = read_status(cfg)
    if not status:
        return
    if status.get("stage") != "faces":
        return
    if status.get("state") not in {"starting_file", "reading_frame", "detecting_faces"}:
        return

    file_name = status.get("file_name")
    if not isinstance(file_name, str) or not file_name:
        return

    item_by_name = {item.file_name: item for item in media_items}
    item = item_by_name.get(file_name)
    if item is None:
        return

    rec = records.get(item.file_name)
    if face_stage_complete(rec, item):
        return
    if skip_processing_applies(rec):
        return

    message = (
        f"previous run stopped while face stage was {status.get('state')}; "
        "marked face stage failed/no faces to avoid repeat hang"
    )
    failed = mark_face_failure_record(records, item, message)
    upsert_record(table, records, failed)
    print(f"[faces] skipping previously stuck file: {item.file_name}")


def run_face_stage(
    cfg: AppConfig,
    table,
    records: dict[str, dict[str, Any]],
    media_items: list[MediaItem],
) -> set[str]:
    mark_stale_face_status_if_needed(cfg, table, records, media_items)
    pending: list[MediaItem] = []
    force = should_rerun_stage(cfg, "6a")
    for item in media_items:
        rec = records.get(item.file_name)
        if force and not skip_processing_applies(rec):
            pending.append(item)
            continue
        is_complete = face_stage_complete(rec, item)
        if not is_complete:
            pending.append(item)
            continue
        if should_rerun_face_row(cfg, rec, item):
            pending.append(item)
    if not pending:
        report_stage_complete("Stage 6a/8 Faces", len(media_items), "files")
        return set()

    worker = FaceWorker(
        cfg.insightface_root,
        det_thresh=cfg.face_det_threshold,
        det_size=cfg.face_det_size,
        fallback_det_size=cfg.face_fallback_det_size,
        dedupe_cosine=cfg.face_dedupe_cosine,
    )
    upsert_batch: list[dict[str, Any]] = []
    changed: set[str] = set()
    consecutive_face_timeouts = 0
    pbar = progress(pending, desc="Stage 6a/8 Faces", unit="file")
    try:
        for file_index, item in enumerate(pbar, start=1):
            pbar.set_postfix_str(shorten_for_status(item.file_name), refresh=True)
            write_status(
                cfg,
                stage="faces",
                state="starting_file",
                file_name=item.file_name,
                file_index=file_index,
                file_total=len(pending),
                is_video=item.is_video,
                frame_total=len(item.frame_refs),
            )
            base = records.get(
                item.file_name,
                default_record(item.file_name, item.is_video, item.collection_id),
            )
            try:
                groups: list[dict[str, Any]] = []
                total_embeddings = 0
                for idx, frame_ref in enumerate(item.frame_refs):
                    frame_position = idx + 1
                    pbar.set_postfix_str(
                        shorten_for_status(f"{item.file_name} frame {frame_position}/{len(item.frame_refs)}"),
                        refresh=True,
                    )
                    write_status(
                        cfg,
                        stage="faces",
                        state="detecting_faces",
                        file_name=item.file_name,
                        file_index=file_index,
                        file_total=len(pending),
                        is_video=item.is_video,
                        frame_index=frame_position,
                        frame_total=len(item.frame_refs),
                        timestamp_sec=float(frame_ref.timestamp_sec),
                        frame_path=str(frame_ref.image_path),
                        timeout_seconds=cfg.face_timeout_seconds,
                    )
                    if should_skip_face_detection_by_shape(cfg, frame_ref.image_path):
                        embeddings = []
                    else:
                        embeddings = worker.detect(frame_ref.image_path, cfg.face_timeout_seconds)
                    remaining = cfg.max_face_embeddings_per_file - total_embeddings
                    if remaining <= 0:
                        embeddings = []
                    elif len(embeddings) > remaining:
                        embeddings = embeddings[:remaining]
                    total_embeddings += len(embeddings)
                    groups.append(
                        {
                            "timestamp_sec": float(frame_ref.timestamp_sec),
                            "face_embeddings": embeddings,
                        }
                    )
                    if total_embeddings >= cfg.max_face_embeddings_per_file:
                        for rest in item.frame_refs[idx + 1 :]:
                            groups.append(
                                {
                                    "timestamp_sec": float(rest.timestamp_sec),
                                    "face_embeddings": [],
                                }
                            )
                        break
            except TimeoutError as exc:
                consecutive_face_timeouts += 1
                if consecutive_face_timeouts >= cfg.max_consecutive_face_timeouts:
                    raise RuntimeError(
                        "InsightFace worker timed out repeatedly; stopping face stage without marking more rows. "
                        f"Timeouts: {consecutive_face_timeouts}/{cfg.max_consecutive_face_timeouts}. "
                        f"Last file: {item.file_name}. Error: {exc}"
                    ) from exc
                print_warning(f"[faces] timeout: {item.file_name}: {exc}")
                failed = mark_face_failure_record(records, item, compact_error_message(exc))
                append_stage_upsert(table, records, upsert_batch, failed)
                changed.add(item.file_name)
                continue
            except FaceWorkerPipeError as exc:
                raise RuntimeError(
                    "InsightFace worker crashed. Stopping face stage without marking more rows failed. "
                    f"Last file: {item.file_name}. Error: {exc}"
                ) from exc
            except Exception as exc:
                consecutive_face_timeouts = 0
                print_error(f"[faces] failed: {item.file_name}: {exc}")
                failed = mark_face_failure_record(records, item, compact_error_message(exc))
                if len(upsert_batch) + 1 >= UPSERT_BATCH_SIZE:
                    write_status(
                        cfg,
                        stage="faces",
                        state="flushing_db_batch",
                        file_name=item.file_name,
                        file_index=file_index,
                        file_total=len(pending),
                        pending_records=len(upsert_batch) + 1,
                    )
                append_stage_upsert(table, records, upsert_batch, failed)
                changed.add(item.file_name)
                continue

            consecutive_face_timeouts = 0
            base["is_video"] = item.is_video
            base["collection_id"] = item.collection_id
            base["face_groups"] = groups
            if base.get("processing_error_stage") == "faces":
                base["processing_error_stage"] = None
                base["processing_error"] = None
            recompute_aggregate_fields(base)
            if len(upsert_batch) + 1 >= UPSERT_BATCH_SIZE:
                write_status(
                    cfg,
                    stage="faces",
                    state="flushing_db_batch",
                    file_name=item.file_name,
                    file_index=file_index,
                    file_total=len(pending),
                    pending_records=len(upsert_batch) + 1,
                )
            append_stage_upsert(table, records, upsert_batch, base)
            changed.add(item.file_name)
    finally:
        worker.close()
    write_status(cfg, stage="faces", state="flushing_db_batch", pending_records=len(upsert_batch))
    upsert_records_batch(table, records, upsert_batch)
    write_status(cfg, stage="faces", state="complete", file_total=len(pending))
    return changed


def run_clip_stage(
    cfg: AppConfig,
    table,
    records: dict[str, dict[str, Any]],
    media_items: list[MediaItem],
) -> set[str]:
    pending: list[MediaItem] = []
    force = should_rerun_stage(cfg, "4a")
    for item in media_items:
        rec = records.get(item.file_name)
        if (force and not skip_processing_applies(rec)) or not clip_stage_complete(rec, item):
            pending.append(item)
    if not pending:
        report_stage_complete("Stage 4a/8 CLIP embeddings", len(media_items), "files")
        return set()

    try:
        embedder = ClipEmbedder(cfg.clip_model, cfg.clip_batch_size)
    except Exception as exc:
        raise RuntimeError(
            f"CLIP stage failed to initialize on CUDA/OpenCLIP: {compact_error_message(exc)}"
        ) from exc
    upsert_batch: list[dict[str, Any]] = []
    changed: set[str] = set()
    pbar = progress(pending, desc="Stage 4a/8 CLIP embeddings", unit="file")
    for file_index, item in enumerate(pbar, start=1):
        pbar.set_postfix_str(shorten_for_status(item.file_name), refresh=True)
        base = records.get(
            item.file_name,
            default_record(item.file_name, item.is_video, item.collection_id),
        )
        try:
            write_status(
                cfg,
                stage="clip",
                state="starting_file",
                file_name=item.file_name,
                file_index=file_index,
                file_total=len(pending),
                frame_count=len(item.frame_refs),
            )
            frames = [read_image_bgr(frame.image_path) for frame in item.frame_refs]
            write_status(
                cfg,
                stage="clip",
                state="encoding_file",
                file_name=item.file_name,
                file_index=file_index,
                file_total=len(pending),
                frame_count=len(item.frame_refs),
                batch_size=cfg.clip_batch_size,
            )
            vectors = embedder.embed_frames(frames)
        except Exception as exc:
            message = compact_error_message(exc)
            write_status(
                cfg,
                stage="clip",
                state="failed_file",
                file_name=item.file_name,
                file_index=file_index,
                file_total=len(pending),
                frame_count=len(item.frame_refs),
                error=message,
            )
            if is_gpu_runtime_error(exc):
                write_status(
                    cfg,
                    stage="clip",
                    state="fatal_gpu_error",
                    file_name=item.file_name,
                    file_index=file_index,
                    file_total=len(pending),
                    frame_count=len(item.frame_refs),
                    error=message,
                )
                raise RuntimeError(
                    f"CLIP GPU runtime failed while embedding {item.file_name}: {message}"
                ) from exc
            print_error(f"[clip] failed: {item.file_name}: {message}")
            continue

        base["is_video"] = item.is_video
        base["collection_id"] = item.collection_id
        base["clip_groups"] = [
            {
                "timestamp_sec": float(frame_ref.timestamp_sec),
                "clip_embedding": vectors[i],
            }
            for i, frame_ref in enumerate(item.frame_refs)
        ]
        # Keep existing SIFT linkage when refreshing the shortlist embeddings.
        recompute_aggregate_fields(base)
        append_stage_upsert(table, records, upsert_batch, base)
        if len(upsert_batch) >= CLIP_UPSERT_BATCH_SIZE:
            write_status(
                cfg,
                stage="clip",
                state="flushing_db_batch",
                file_name=item.file_name,
                file_index=file_index,
                file_total=len(pending),
                pending_records=len(upsert_batch),
            )
            upsert_records_batch(table, records, upsert_batch)
            upsert_batch.clear()
        changed.add(item.file_name)
        write_status(
            cfg,
            stage="clip",
            state="finished_file",
            file_name=item.file_name,
            file_index=file_index,
            file_total=len(pending),
            frame_count=len(item.frame_refs),
        )
    write_status(cfg, stage="clip", state="flushing_db_batch", pending_records=len(upsert_batch))
    upsert_records_batch(table, records, upsert_batch)
    write_status(cfg, stage="clip", state="complete", file_total=len(pending))
    return changed


def run_paddle_detection_stage(
    cfg: AppConfig,
    table,
    records: dict[str, dict[str, Any]],
    media_items: list[MediaItem],
) -> set[str]:
    pending: list[MediaItem] = []
    force = cfg.rerun_paddle_ocr or should_rerun_stage(cfg, "7")
    for item in media_items:
        rec = records.get(item.file_name)
        if skip_processing_applies(rec):
            continue
        if force or not paddle_stage_complete(rec, item):
            pending.append(item)
    if not pending:
        report_stage_complete("Stage 7/8 PaddleOCR", len(media_items), "files")
        return set()

    print(f"[paddle] worker interpreter: {cfg.paddle_python}")
    detector = PaddleTextDetector(cfg.paddle_python, cfg.paddle_det_model, cfg.paddle_device)
    upsert_batch: list[dict[str, Any]] = []
    changed: set[str] = set()
    try:
        for item in progress(pending, desc="Stage 7/8 PaddleOCR", unit="file"):
            base = records.get(
                item.file_name,
                default_record(item.file_name, item.is_video, item.collection_id),
            )
            try:
                groups: list[dict[str, Any]] = []
                for frame_ref in item.frame_refs:
                    has_text = detector.has_text(frame_ref.image_path, cfg.paddle_ocr_max_side)
                    groups.append(
                        {
                            "timestamp_sec": float(frame_ref.timestamp_sec),
                            "text_detected": bool(has_text),
                            "text": None if has_text else "",
                        }
                    )
            except Exception as exc:
                message = compact_error_message(exc)
                print_error(f"[paddle] failed: {item.file_name}: {message}")
                base = mark_paddle_failure_record(records, item, message)
                append_stage_upsert(table, records, upsert_batch, base)
                changed.add(item.file_name)
                continue

            base["is_video"] = item.is_video
            base["collection_id"] = item.collection_id
            base["ocr_groups"] = groups
            if base.get("processing_error_stage") == "paddle_ocr":
                base["processing_error_stage"] = None
                base["processing_error"] = None
            recompute_aggregate_fields(base)
            append_stage_upsert(table, records, upsert_batch, base)
            changed.add(item.file_name)
    finally:
        detector.close()
    write_status(
        cfg,
        stage="paddle_ocr",
        state="flushing_db_batch",
        pending_records=len(upsert_batch),
    )
    upsert_records_batch(table, records, upsert_batch)
    write_status(cfg, stage="paddle_ocr", state="complete", file_total=len(pending))
    return changed


def run_easyocr_stage(
    cfg: AppConfig,
    table,
    records: dict[str, dict[str, Any]],
    media_items: list[MediaItem],
) -> set[str]:
    pending: list[MediaItem] = []
    force = should_rerun_stage(cfg, "8a")
    for item in media_items:
        rec = records.get(item.file_name)
        groups = rec.get("ocr_groups") if rec else None
        can_rerun = (
            force
            and not skip_processing_applies(rec)
            and groups_match_item(groups, item)
            and any(group.get("text_detected") is True for group in groups)
        )
        if can_rerun or easyocr_stage_needed(rec, item):
            pending.append(item)
    if not pending:
        report_stage_complete("Stage 8a/8 EasyOCR text extraction", len(media_items), "files")
        return set()

    reduced_batch_size, reduced_canvas_size, reduced_max_side = reduced_easyocr_retry_settings(cfg)
    use_reduced_gpu_retry = cfg.easyocr_gpu and (
        reduced_batch_size != cfg.easyocr_batch_size
        or reduced_canvas_size != cfg.easyocr_canvas_size
        or reduced_max_side != cfg.easyocr_max_side
    )
    primary_extractor = start_easyocr_worker(cfg, cfg.easyocr_gpu)
    reduced_gpu_extractor: EasyOcrWorker | None = None
    cpu_extractor: EasyOcrWorker | None = None
    upsert_batch: list[dict[str, Any]] = []
    changed: set[str] = set()
    primary_gpu_retry_count = 0
    reduced_gpu_retry_count = 0

    def should_log_easyocr_retry(count: int) -> bool:
        return count <= 3 or count % 100 == 0

    def ensure_easyocr_worker(kind: str) -> EasyOcrWorker:
        nonlocal primary_extractor, reduced_gpu_extractor, cpu_extractor
        if kind == "primary":
            if not primary_extractor.is_alive():
                primary_extractor.close()
                primary_extractor = start_easyocr_worker(cfg, cfg.easyocr_gpu)
            return primary_extractor
        if kind == "reduced_gpu":
            if reduced_gpu_extractor is None or not reduced_gpu_extractor.is_alive():
                if reduced_gpu_extractor is not None:
                    reduced_gpu_extractor.close()
                reduced_gpu_extractor = start_easyocr_worker(
                    cfg,
                    True,
                    batch_size_override=reduced_batch_size,
                    canvas_size_override=reduced_canvas_size,
                )
            return reduced_gpu_extractor
        if cpu_extractor is None or not cpu_extractor.is_alive():
            if cpu_extractor is not None:
                cpu_extractor.close()
            cpu_extractor = start_easyocr_worker(
                cfg,
                False,
                batch_size_override=reduced_batch_size,
                canvas_size_override=reduced_canvas_size,
            )
        return cpu_extractor

    try:
        for item in progress(pending, desc="Stage 8a/8 EasyOCR text extraction", unit="file"):
            base = records[item.file_name]
            try:
                groups = [dict(group) for group in base["ocr_groups"]]
                for i, group in enumerate(groups):
                    if group.get("text_detected") is True and (
                        force or group.get("text") is None
                    ):
                        frame_path = item.frame_refs[i].image_path
                        attempts: list[tuple[str, int]] = [("primary", cfg.easyocr_max_side)]
                        if use_reduced_gpu_retry:
                            attempts.append(("reduced_gpu", reduced_max_side))
                        if cfg.easyocr_gpu:
                            attempts.append(("cpu", reduced_max_side))
                        last_exc: Exception | None = None
                        for attempt_kind, attempt_max_side in attempts:
                            worker = ensure_easyocr_worker(attempt_kind)
                            try:
                                text = worker.extract_text(frame_path, attempt_max_side)
                                break
                            except Exception as exc:
                                last_exc = exc
                                worker_alive = worker.is_alive()
                                if attempt_kind == "primary" and cfg.easyocr_gpu and (
                                    is_cuda_oom_error(exc) or not worker_alive
                                ):
                                    if use_reduced_gpu_retry:
                                        primary_gpu_retry_count += 1
                                        if should_log_easyocr_retry(primary_gpu_retry_count):
                                            print_warning(
                                                f"[easyocr] primary CUDA OCR fallback #{primary_gpu_retry_count}; "
                                                f"using reduced GPU profile "
                                                f"(max_side={reduced_max_side}, canvas_size={reduced_canvas_size}, "
                                                f"batch_size={reduced_batch_size}); latest={frame_path}"
                                            )
                                        continue
                                    primary_gpu_retry_count += 1
                                    if should_log_easyocr_retry(primary_gpu_retry_count):
                                        print_warning(
                                            f"[easyocr] primary CUDA OCR fallback #{primary_gpu_retry_count}; "
                                            f"retrying on CPU; latest={frame_path}"
                                        )
                                    continue
                                if attempt_kind == "reduced_gpu" and (is_cuda_oom_error(exc) or not worker_alive):
                                    reduced_gpu_retry_count += 1
                                    if should_log_easyocr_retry(reduced_gpu_retry_count):
                                        print_warning(
                                            f"[easyocr] reduced GPU OCR fallback #{reduced_gpu_retry_count}; "
                                            f"retrying on CPU; latest={frame_path}"
                                        )
                                    continue
                                raise
                        else:
                            assert last_exc is not None
                            raise last_exc
                        group["text"] = text
                    elif group.get("text_detected") is False and group.get("text") is None:
                        group["text"] = ""
            except Exception as exc:
                print_error(f"[easyocr] failed: {item.file_name}: {exc}")
                if not primary_extractor.is_alive():
                    primary_extractor.close()
                    primary_extractor = start_easyocr_worker(cfg, cfg.easyocr_gpu)
                continue

            base["collection_id"] = item.collection_id
            base["ocr_groups"] = groups
            recompute_aggregate_fields(base)
            append_stage_upsert(table, records, upsert_batch, base)
            changed.add(item.file_name)
    finally:
        primary_extractor.close()
        if reduced_gpu_extractor is not None:
            reduced_gpu_extractor.close()
        if cpu_extractor is not None:
            cpu_extractor.close()
    upsert_records_batch(table, records, upsert_batch)
    return changed


def start_easyocr_worker(
    cfg: AppConfig,
    gpu: bool,
    *,
    batch_size_override: int | None = None,
    canvas_size_override: int | None = None,
) -> EasyOcrWorker:
    try:
        return EasyOcrWorker(
            cfg.easyocr_langs,
            batch_size_override if batch_size_override is not None else cfg.easyocr_batch_size,
            canvas_size_override if canvas_size_override is not None else cfg.easyocr_canvas_size,
            gpu,
        )
    except Exception as exc:
        if gpu:
            raise RuntimeError(
                "EasyOCR CUDA worker failed to start. Refusing to continue on CPU because "
                "--easyocr-device cuda was requested."
            ) from exc
        raise


def print_summary(records: dict[str, dict[str, Any]], media_items: list[MediaItem]) -> None:
    total = len(media_items)
    relevant_records = [records.get(item.file_name) for item in media_items]
    faces_done = sum(
        1 for i, rec in enumerate(relevant_records) if face_stage_complete(rec, media_items[i])
    )
    clip_done = sum(
        1 for i, rec in enumerate(relevant_records) if clip_stage_complete(rec, media_items[i])
    )
    paddle_done = sum(
        1 for i, rec in enumerate(relevant_records) if paddle_stage_complete(rec, media_items[i])
    )
    easy_done = sum(
        1 for i, rec in enumerate(relevant_records) if easyocr_stage_complete(rec, media_items[i])
    )
    video_count = sum(1 for item in media_items if item.is_video)
    image_count = total - video_count
    print(f"Records in LanceDB for this scan: {total} ({image_count} images, {video_count} videos)")
    print(f"Faces stage complete: {faces_done}/{total}")
    print(f"CLIP stage complete: {clip_done}/{total}")
    print(f"PaddleOCR stage complete: {paddle_done}/{total}")
    print(f"EasyOCR stage complete: {easy_done}/{total}")


def read_image_bgr(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Failed to decode image: {path}")
    return frame


def resize_frame_max_side(frame: np.ndarray, max_side: int) -> np.ndarray:
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return frame
    scale = max_side / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def read_image_bgr_resized(path: Path, max_side: int) -> np.ndarray:
    return resize_frame_max_side(read_image_bgr(path), max_side)


def round_timestamp(value: float) -> float:
    return round(float(value), TIMESTAMP_ROUND_DIGITS)


def scene_stills_root(input_dir: Path, db_dir: Path) -> Path:
    return db_dir / f"{input_dir.name}-video"


def build_video_output_dir(input_dir: Path, video_path: Path, root: Path) -> Path:
    rel = video_path.resolve().relative_to(input_dir)
    suffix_tag = rel.suffix.lower().replace(".", "_")
    leaf = f"{rel.stem}{suffix_tag}"
    return root / rel.parent / leaf


def select_pruned_timestamps(timestamps: list[float], max_count: int) -> list[float]:
    if len(timestamps) <= max_count:
        return timestamps
    indices = np.linspace(0, len(timestamps) - 1, num=max_count, dtype=int).tolist()
    deduped_indices = sorted(set(indices))
    return [timestamps[i] for i in deduped_indices]


def detect_scene_timestamps(video_path: Path, cfg: AppConfig) -> list[float]:
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector

    video = open_video(str(video_path), backend="opencv")
    manager = SceneManager()
    manager.add_detector(
        ContentDetector(
            threshold=cfg.scene_threshold,
            min_scene_len=cfg.scene_min_scene_len,
        )
    )
    manager.detect_scenes(video=video, show_progress=False)
    scene_list = manager.get_scene_list(start_in_scene=True)
    timestamps: list[float] = []
    for start, end in scene_list:
        start_sec = float(start.get_seconds())
        end_sec = float(end.get_seconds())
        mid_sec = start_sec if end_sec <= start_sec else (start_sec + end_sec) / 2.0
        timestamps.append(round_timestamp(mid_sec))
    if not timestamps:
        timestamps = [0.0]
    timestamps = select_pruned_timestamps(timestamps, MAX_STILLS_PER_VIDEO)
    return timestamps


def load_manifest(manifest_path: Path) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_manifest(manifest_path: Path, data: dict[str, Any]) -> None:
    temp = manifest_path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(manifest_path)


def existing_stills_valid(
    manifest: dict[str, Any] | None,
    video_path: Path,
    cfg: AppConfig,
    video_dir: Path,
) -> bool:
    if not manifest:
        return False
    if manifest.get("source_size") != video_path.stat().st_size:
        return False
    if manifest.get("source_mtime_ns") != video_path.stat().st_mtime_ns:
        return False
    if manifest.get("scene_threshold") != cfg.scene_threshold:
        return False
    if manifest.get("scene_min_scene_len") != cfg.scene_min_scene_len:
        return False
    if manifest.get("max_stills_per_video") != MAX_STILLS_PER_VIDEO:
        return False
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        return False
    for frame in frames:
        file_name = frame.get("image_file")
        if not file_name:
            return False
        if not (video_dir / file_name).exists():
            return False
    return True


def extract_stills_for_video(
    video_path: Path,
    cfg: AppConfig,
    video_dir: Path,
    force: bool = False,
) -> list[FrameRef]:
    video_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = video_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    if not force and existing_stills_valid(manifest, video_path, cfg, video_dir):
        return [
            FrameRef(
                timestamp_sec=round_timestamp(float(frame["timestamp_sec"])),
                image_path=(video_dir / str(frame["image_file"])).resolve(),
            )
            for frame in manifest["frames"]
        ]

    for old_jpg in video_dir.glob("*.jpg"):
        old_jpg.unlink()

    timestamps = detect_scene_timestamps(video_path, cfg)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for still extraction: {video_path}")

    frames: list[FrameRef] = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        for idx, timestamp_sec in enumerate(timestamps):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_sec) * 1000.0)
            ok, frame = cap.read()
            if not ok and fps and fps > 0:
                frame_idx = int(round(float(timestamp_sec) * float(fps)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
                ok, frame = cap.read()
            if not ok or frame is None:
                continue
            ms = int(round(float(timestamp_sec) * 1000.0))
            image_file = f"t_{idx:04d}_{ms:010d}.jpg"
            image_path = video_dir / image_file
            cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            frames.append(
                FrameRef(
                    timestamp_sec=round_timestamp(timestamp_sec),
                    image_path=image_path.resolve(),
                )
            )
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"PySceneDetect produced no usable stills for: {video_path}")

    manifest_data = {
        "source_size": video_path.stat().st_size,
        "source_mtime_ns": video_path.stat().st_mtime_ns,
        "scene_threshold": cfg.scene_threshold,
        "scene_min_scene_len": cfg.scene_min_scene_len,
        "max_stills_per_video": MAX_STILLS_PER_VIDEO,
        "frames": [
            {
                "timestamp_sec": float(frame.timestamp_sec),
                "image_file": frame.image_path.name,
            }
            for frame in frames
        ],
    }
    write_manifest(manifest_path, manifest_data)
    return frames


def extract_video_stills(
    cfg: AppConfig,
    video_paths: list[Path],
    records: dict[str, dict[str, Any]],
) -> tuple[Path, dict[Path, list[FrameRef]]]:
    root = scene_stills_root(cfg.input_dir, cfg.db_dir)
    root.mkdir(parents=True, exist_ok=True)
    frame_map: dict[Path, list[FrameRef]] = {}

    videos_to_extract: list[Path] = []
    failed_count = 0
    force = should_rerun_stage(cfg, "2")
    for video_path in video_paths:
        file_name = scoped_file_name(cfg, video_path)
        existing = records.get(file_name)
        if not force and existing is not None and skip_processing_applies(existing):
            continue
        if not force and load_existing_video_frame_refs(cfg, video_path):
            continue
        videos_to_extract.append(video_path)

    for video_path in progress(videos_to_extract, desc="Stage 2/8 PySceneDetect", unit="video"):
        video_dir = build_video_output_dir(cfg.input_dir, video_path, root)
        try:
            frame_map[video_path] = extract_stills_for_video(video_path, cfg, video_dir, force=force)
        except Exception as exc:
            print_error(f"[scenedetect] failed: {relative_file_name(cfg.input_dir, video_path)}: {exc}")
            frame_map[video_path] = []
            failed_count += 1
    if failed_count:
        report_stage_incomplete(
            "Stage 2/8 PySceneDetect",
            len(video_paths) - failed_count,
            len(video_paths),
            "videos",
            f"{failed_count} failed",
        )
    else:
        report_stage_complete("Stage 2/8 PySceneDetect", len(video_paths), "videos")
    return root, frame_map


def build_media_items(
    cfg: AppConfig,
    image_paths: list[Path],
    video_paths: list[Path],
    video_frames: dict[Path, list[FrameRef]],
) -> list[MediaItem]:
    items: list[MediaItem] = []
    for path in image_paths:
        items.append(
            MediaItem(
                source_path=path,
                file_name=scoped_file_name(cfg, path),
                collection_id=cfg.collection_id,
                is_video=False,
                frame_refs=[
                    FrameRef(
                        timestamp_sec=0.0,
                        image_path=path.resolve(),
                    )
                ],
            )
        )
    for path in video_paths:
        frames = video_frames.get(path)
        if frames is None:
            frames = load_existing_video_frame_refs(cfg, path)
        if not frames:
            continue
        items.append(
            MediaItem(
                source_path=path,
                file_name=scoped_file_name(cfg, path),
                collection_id=cfg.collection_id,
                is_video=True,
                frame_refs=frames,
            )
        )
    items.sort(key=lambda item: item.file_name)
    return items


def item_timestamps(item: MediaItem) -> list[float]:
    return [round_timestamp(frame.timestamp_sec) for frame in item.frame_refs]


def group_timestamps(groups: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for group in groups:
        ts = group.get("timestamp_sec")
        values.append(round_timestamp(float(ts)))
    return values


def groups_match_item(groups: Any, item: MediaItem) -> bool:
    if not isinstance(groups, list):
        return False
    if len(groups) != len(item.frame_refs):
        return False
    return group_timestamps(groups) == item_timestamps(item)


def skip_processing_applies(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    if record.get("skip_processing") is not True:
        return False
    if normalize_phash_hex(record.get("phash_hex")) is not None:
        return True
    return bool(record.get("processing_error"))


def load_existing_video_frame_refs(
    cfg: AppConfig,
    video_path: Path,
) -> list[FrameRef]:
    root = scene_stills_root(cfg.input_dir, cfg.db_dir)
    video_dir = build_video_output_dir(cfg.input_dir, video_path, root)
    manifest_path = video_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    if not existing_stills_valid(manifest, video_path, cfg, video_dir):
        return []
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        return []
    result: list[FrameRef] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        image_file = frame.get("image_file")
        if not isinstance(image_file, str) or not image_file:
            continue
        try:
            timestamp_sec = round_timestamp(float(frame.get("timestamp_sec", 0.0)))
        except (TypeError, ValueError):
            timestamp_sec = 0.0
        result.append(
            FrameRef(
                timestamp_sec=timestamp_sec,
                image_path=(video_dir / image_file).resolve(),
            )
        )
    return result


def video_frame_phashes_by_timestamp(value: Any) -> dict[float, str]:
    groups = normalize_video_frame_phashes(value) or []
    return {
        round_timestamp(float(group["timestamp_sec"])): str(group["phash_hex"])
        for group in groups
    }


def filtered_cross_media_matches(
    matches: list[dict[str, Any]] | None,
    excluded_files: set[str],
) -> list[dict[str, Any]]:
    if not matches:
        return []
    filtered = [dict(entry) for entry in matches if str(entry.get("file_name")) not in excluded_files]
    return normalize_cross_media_matches(filtered) or []


def merge_cross_media_match_entries(
    base_matches: list[dict[str, Any]] | None,
    extra_matches: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    if base_matches:
        merged.extend(dict(entry) for entry in base_matches)
    if extra_matches:
        merged.extend(dict(entry) for entry in extra_matches)
    return normalize_cross_media_matches(merged) or []


def record_cross_media_entry(
    target: dict[str, list[dict[str, Any]]],
    *,
    file_name: str,
    related_file_name: str,
    related_is_video: bool,
    similarity_pct: float,
) -> None:
    target.setdefault(file_name, []).append(
        {
            "file_name": related_file_name,
            "is_video": related_is_video,
            "similarity_pct": round(float(similarity_pct), 3),
        }
    )


def build_video_frame_hash_index(
    records: dict[str, dict[str, Any]],
) -> tuple[HammingBkTree, dict[str, VideoFrameHashRef]]:
    tree = HammingBkTree()
    refs_by_key: dict[str, VideoFrameHashRef] = {}
    counter = 0
    for file_name, record in records.items():
        if not bool(record.get("is_video")):
            continue
        frame_groups = normalize_video_frame_phashes(record.get("video_frame_phashes")) or []
        for group in frame_groups:
            phash_hex = normalize_phash_hex(group.get("phash_hex"))
            if phash_hex is None:
                continue
            key = f"{file_name}\0{counter}"
            counter += 1
            ref = VideoFrameHashRef(
                file_name=file_name,
                timestamp_sec=float(group.get("timestamp_sec", 0.0)),
                phash_hex=phash_hex,
            )
            refs_by_key[key] = ref
            tree.add(int(phash_hex, 16), key)
    return tree, refs_by_key


def run_video_frame_phash_stage(
    cfg: AppConfig,
    table,
    records: dict[str, dict[str, Any]],
    video_paths: list[Path],
    extracted_video_frame_map: dict[Path, list[FrameRef]],
) -> set[str]:
    changed: set[str] = set()
    upsert_batch: list[dict[str, Any]] = []
    current_video_frame_refs: dict[str, list[FrameRef]] = {}
    current_video_hashes: dict[str, dict[float, str]] = {}
    pending_frame_hashes: list[tuple[str, FrameRef]] = []
    reused_frame_count = 0
    force = should_rerun_stage(cfg, "3c")

    for video_path in video_paths:
        file_name = scoped_file_name(cfg, video_path)
        record = records.get(file_name, default_record(file_name, True, cfg.collection_id))
        frame_refs = extracted_video_frame_map.get(video_path)
        allow_existing_hashes = frame_refs is None and not force
        if frame_refs is None:
            frame_refs = load_existing_video_frame_refs(cfg, video_path)
        current_video_frame_refs[file_name] = frame_refs
        existing_hashes = (
            video_frame_phashes_by_timestamp(record.get("video_frame_phashes"))
            if allow_existing_hashes
            else {}
        )
        current_hashes: dict[float, str] = {}
        if not frame_refs:
            current_hashes.update(existing_hashes)
        for frame_ref in frame_refs:
            timestamp_sec = round_timestamp(frame_ref.timestamp_sec)
            existing_hash = existing_hashes.get(timestamp_sec)
            if existing_hash is not None:
                current_hashes[timestamp_sec] = existing_hash
                reused_frame_count += 1
            else:
                pending_frame_hashes.append((file_name, frame_ref))
        current_video_hashes[file_name] = current_hashes

    print(
        f"[video-frame-phash] reusing {reused_frame_count} existing video-still pHashes; "
        f"hashing {len(pending_frame_hashes)} new or changed frames"
    )
    if not pending_frame_hashes:
        report_stage_complete("Stage 3c/8 pHash video stills", reused_frame_count, "frames")
    write_status(
        cfg,
        stage="video_frame_phash",
        state="running",
        reused_frames=reused_frame_count,
        frame_total=len(pending_frame_hashes),
    )
    failed_frame_count = 0
    with ThreadPoolExecutor(max_workers=max(1, cfg.hash_workers)) as executor:
        futures = {
            executor.submit(compute_phash_hex, frame_ref.image_path): (file_name, frame_ref)
            for file_name, frame_ref in pending_frame_hashes
        }
        for future in progress(
            as_completed(futures),
            desc="Stage 3c/8 pHash video stills",
            unit="frame",
            total=len(futures),
        ):
            file_name, frame_ref = futures[future]
            try:
                phash_hex = future.result()
            except Exception as exc:
                failed_frame_count += 1
                print_error(f"[video-frame-phash] failed: {file_name}: {frame_ref.image_path}: {exc}")
                continue
            timestamp_sec = round_timestamp(frame_ref.timestamp_sec)
            current_video_hashes[file_name][timestamp_sec] = phash_hex
    write_status(
        cfg,
        stage="video_frame_phash",
        state="complete",
        reused_frames=reused_frame_count,
        frame_total=len(pending_frame_hashes),
        failed_frames=failed_frame_count,
    )

    for video_path in video_paths:
        file_name = scoped_file_name(cfg, video_path)
        record = records.get(file_name, default_record(file_name, True, cfg.collection_id))
        frame_hash_groups = normalize_video_frame_phashes(
            [
                {
                    "timestamp_sec": float(round_timestamp(frame_ref.timestamp_sec)),
                    "phash_hex": current_video_hashes[file_name][round_timestamp(frame_ref.timestamp_sec)],
                }
                for frame_ref in current_video_frame_refs[file_name]
                if round_timestamp(frame_ref.timestamp_sec) in current_video_hashes[file_name]
            ]
        ) or []
        if not current_video_frame_refs[file_name]:
            frame_hash_groups = normalize_video_frame_phashes(record.get("video_frame_phashes")) or []
        if normalize_video_frame_phashes(record.get("video_frame_phashes")) != frame_hash_groups:
            record["video_frame_phashes"] = frame_hash_groups
            record["collection_id"] = cfg.collection_id
            record["is_video"] = True
            append_stage_upsert(table, records, upsert_batch, record)
            changed.add(file_name)

    flush_hash_upserts(table, records, upsert_batch)
    return changed


def run_cross_media_match_stage(
    cfg: AppConfig,
    table,
    records: dict[str, dict[str, Any]],
    image_paths: list[Path],
    video_paths: list[Path],
) -> set[str]:
    current_image_files = {scoped_file_name(cfg, path) for path in image_paths}
    current_video_files = {scoped_file_name(cfg, path) for path in video_paths}
    current_files = current_image_files | current_video_files
    if not current_files:
        report_stage_complete("Stage 3e/8 image↔video frame match", 0, "images")
        report_stage_complete("Stage 3f/8 video frame↔image match", 0, "videos")
        return set()

    print("[cross-media] checking completion fingerprint")
    input_fingerprint = cross_media_input_fingerprint(
        cfg,
        records,
        current_image_files,
        current_video_files,
    )
    state = read_cross_media_state(cfg)
    collections = state.get("collections")
    if not isinstance(collections, dict):
        collections = {}
    collection_state = collections.get(cfg.collection_id)
    force = should_rerun_stage(cfg, "3e") or should_rerun_stage(cfg, "3f")
    if (
        not force
        and isinstance(collection_state, dict)
        and collection_state.get("fingerprint") == input_fingerprint
    ):
        clear_cross_media_work(cfg)
        report_stage_complete("Stage 3e/8 image↔video frame match", len(image_paths), "images")
        report_stage_complete("Stage 3f/8 video frame↔image match", len(video_paths), "videos")
        return set()

    changed: set[str] = set()
    upsert_batch: list[dict[str, Any]] = []
    work = None if force else read_cross_media_work(cfg, input_fingerprint)
    work_state = str(work.get("state")) if work else ""
    if work and work_state == "image_to_video_complete":
        print_info("[cross-media] resuming from saved Stage 3e image-to-video results")
    elif work and work_state == "computed":
        print_info("[cross-media] reusing saved Stage 3e/3f results and resuming DB apply")

    for file_name, record in records.items():
        existing_matches = normalize_cross_media_matches(record.get("cross_media_matches")) or []
        filtered = filtered_cross_media_matches(
            existing_matches,
            current_files,
        )
        if existing_matches and existing_matches != filtered:
            record["cross_media_matches"] = filtered
            append_stage_upsert(table, records, upsert_batch, record)
            changed.add(file_name)

    current_video_hash_groups = {
        scoped_file_name(cfg, video_path): normalize_video_frame_phashes(
            records.get(scoped_file_name(cfg, video_path), {}).get("video_frame_phashes")
        )
        or []
        for video_path in video_paths
    }

    image_tree = HammingBkTree()
    for file_name, record in records.items():
        if bool(record.get("is_video")):
            continue
        phash_hex = normalize_phash_hex(record.get("phash_hex"))
        if phash_hex is None:
            continue
        image_tree.add(int(phash_hex, 16), file_name)

    video_tree, video_refs_by_key = build_video_frame_hash_index(records)
    max_distance = max_hamming_for_similarity(cfg.cross_media_similarity_pct)
    desired_by_file: dict[str, list[dict[str, Any]]]

    if work and work_state in {"image_to_video_complete", "computed"}:
        desired_by_file = {
            str(file_name): list(matches)
            for file_name, matches in work["desired_by_file"].items()
        }
        report_stage_complete("Stage 3e/8 image↔video frame match", len(image_paths), "images")
    else:
        desired_by_file = {file_name: [] for file_name in current_files}
        write_status(
            cfg,
            stage="cross_media_image_to_video",
            state="running",
            image_total=len(image_paths),
        )
        for image_path in progress(image_paths, desc="Stage 3e/8 image↔video frame match", unit="image"):
            file_name = scoped_file_name(cfg, image_path)
            record = records.get(file_name)
            if not record:
                continue
            phash_hex = normalize_phash_hex(record.get("phash_hex"))
            if phash_hex is None:
                continue
            per_video_best: dict[str, dict[str, Any]] = {}
            for ref_key, distance in video_tree.find_all(int(phash_hex, 16), max_distance=max_distance):
                ref = video_refs_by_key.get(ref_key)
                if ref is None:
                    continue
                if ref.file_name == file_name:
                    continue
                similarity_pct = round(float((64 - distance) * 100.0 / 64.0), 3)
                existing = per_video_best.get(ref.file_name)
                if existing is None or float(existing["similarity_pct"]) < similarity_pct:
                    per_video_best[ref.file_name] = {
                        "file_name": ref.file_name,
                        "is_video": True,
                        "similarity_pct": similarity_pct,
                    }
            matches = normalize_cross_media_matches(list(per_video_best.values())) or []
            desired_by_file[file_name] = matches
            for entry in matches:
                record_cross_media_entry(
                    desired_by_file,
                    file_name=str(entry["file_name"]),
                    related_file_name=file_name,
                    related_is_video=False,
                    similarity_pct=float(entry["similarity_pct"]),
                )

        write_status(
            cfg,
            stage="cross_media_image_to_video",
            state="complete",
            image_total=len(image_paths),
        )
        write_cross_media_work(
            cfg,
            fingerprint=input_fingerprint,
            state="image_to_video_complete",
            desired_by_file=desired_by_file,
            image_total=len(image_paths),
            video_total=len(video_paths),
        )
    if work and work_state == "computed":
        report_stage_complete("Stage 3f/8 video frame↔image match", len(video_paths), "videos")
    else:
        write_status(
            cfg,
            stage="cross_media_video_to_image",
            state="running",
            video_total=len(video_paths),
        )
        for video_path in progress(video_paths, desc="Stage 3f/8 video frame↔image match", unit="video"):
            file_name = scoped_file_name(cfg, video_path)
            frame_hash_groups = current_video_hash_groups.get(file_name) or []
            per_image_best: dict[str, dict[str, Any]] = {}
            for group in frame_hash_groups:
                phash_hex = normalize_phash_hex(group.get("phash_hex"))
                if phash_hex is None:
                    continue
                for image_file_name, distance in image_tree.find_all(int(phash_hex, 16), max_distance=max_distance):
                    if image_file_name == file_name:
                        continue
                    similarity_pct = round(float((64 - distance) * 100.0 / 64.0), 3)
                    existing = per_image_best.get(image_file_name)
                    if existing is None or float(existing["similarity_pct"]) < similarity_pct:
                        per_image_best[image_file_name] = {
                            "file_name": image_file_name,
                            "is_video": False,
                            "similarity_pct": similarity_pct,
                        }
            matches = normalize_cross_media_matches(list(per_image_best.values())) or []
            desired_by_file[file_name] = merge_cross_media_match_entries(desired_by_file.get(file_name), matches)
            for entry in matches:
                record_cross_media_entry(
                    desired_by_file,
                    file_name=str(entry["file_name"]),
                    related_file_name=file_name,
                    related_is_video=True,
                    similarity_pct=float(entry["similarity_pct"]),
                )

        write_status(
            cfg,
            stage="cross_media_video_to_image",
            state="complete",
            video_total=len(video_paths),
        )
        write_cross_media_work(
            cfg,
            fingerprint=input_fingerprint,
            state="computed",
            desired_by_file=desired_by_file,
            image_total=len(image_paths),
            video_total=len(video_paths),
        )
    for file_name, desired_matches in desired_by_file.items():
        record = records.get(file_name)
        if record is None:
            continue
        filtered_existing = filtered_cross_media_matches(
            normalize_cross_media_matches(record.get("cross_media_matches")),
            current_files,
        )
        final_matches = desired_matches if file_name in current_files else merge_cross_media_match_entries(
            filtered_existing,
            desired_matches,
        )
        if normalize_cross_media_matches(record.get("cross_media_matches")) != final_matches:
            record["cross_media_matches"] = final_matches
            append_stage_upsert(table, records, upsert_batch, record)
            changed.add(file_name)

    upsert_records_batch(table, records, upsert_batch)
    collections[cfg.collection_id] = {
        "fingerprint": input_fingerprint,
        "image_total": len(image_paths),
        "video_total": len(video_paths),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    state["version"] = 1
    state["collections"] = collections
    write_cross_media_state(cfg, state)
    clear_cross_media_work(cfg)
    return changed


def face_stage_complete(record: dict[str, Any] | None, item: MediaItem) -> bool:
    if skip_processing_applies(record):
        return True
    if not record:
        return False
    groups = record.get("face_groups")
    if not groups_match_item(groups, item):
        return False
    return all(group.get("face_embeddings") is not None for group in groups)


def face_groups_have_embeddings(groups: Any) -> bool:
    if not isinstance(groups, list):
        return False
    return any(len(group.get("face_embeddings") or []) > 0 for group in groups)


def should_rerun_face_row(cfg: AppConfig, record: dict[str, Any] | None, item: MediaItem) -> bool:
    if not record:
        return False
    if not groups_match_item(record.get("face_groups"), item):
        return False
    if cfg.rerun_face_failures and record.get("processing_error_stage") == "faces":
        return True
    if cfg.rerun_zero_face_detections and not face_groups_have_embeddings(record.get("face_groups")):
        return True
    return False


def clip_stage_complete(record: dict[str, Any] | None, item: MediaItem) -> bool:
    if skip_processing_applies(record):
        return True
    if not record:
        return False
    groups = record.get("clip_groups")
    if not groups_match_item(groups, item):
        return False
    return all(group.get("clip_embedding") is not None for group in groups)


def paddle_stage_complete(record: dict[str, Any] | None, item: MediaItem) -> bool:
    if skip_processing_applies(record):
        return True
    if not record:
        return False
    groups = record.get("ocr_groups")
    if not groups_match_item(groups, item):
        return False
    return all(group.get("text_detected") is not None for group in groups)


def easyocr_stage_complete(record: dict[str, Any] | None, item: MediaItem) -> bool:
    if skip_processing_applies(record):
        return True
    if not record:
        return False
    groups = record.get("ocr_groups")
    if not groups_match_item(groups, item):
        return False
    for group in groups:
        if group.get("text_detected") is True and group.get("text") is None:
            return False
    return True


def easyocr_stage_needed(record: dict[str, Any] | None, item: MediaItem) -> bool:
    if skip_processing_applies(record):
        return False
    if not record:
        return False
    groups = record.get("ocr_groups")
    if not groups_match_item(groups, item):
        return False
    for group in groups:
        if group.get("text_detected") is True and group.get("text") is None:
            return True
    return False


def recompute_aggregate_fields(record: dict[str, Any]) -> None:
    face_groups = record.get("face_groups")
    if isinstance(face_groups, list):
        any_faces = any(len(group.get("face_embeddings") or []) > 0 for group in face_groups)
        record["faces"] = any_faces
    else:
        record["faces"] = None


def clear_sift_match_fields(record: dict[str, Any]) -> None:
    record["sift_match_file"] = None
    record["sift_match_score"] = None
    record["sift_match_inliers"] = None
    record["sift_match_good_matches"] = None
    record["sift_match_inlier_ratio"] = None


def build_image_path_map(media_items: list[MediaItem]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in media_items:
        if item.is_video:
            continue
        result[item.file_name] = item.source_path
    return result


def sift_match_pair(
    sift: Any,
    image_path: Path,
    master_path: Path,
    cfg: AppConfig,
    query_features: tuple[Any, Any, int] | None = None,
    master_features: tuple[Any, Any, int] | None = None,
) -> dict[str, Any]:
    if query_features is None:
        img_a = read_image_bgr_resized(image_path, cfg.sift_max_side)
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        keypoints_a, descriptors_a = sift.detectAndCompute(gray_a, None)
        kp_count_a = int(len(keypoints_a) if keypoints_a is not None else 0)
    else:
        keypoints_a, descriptors_a, kp_count_a = query_features

    if master_features is None:
        img_b = read_image_bgr_resized(master_path, cfg.sift_max_side)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        keypoints_b, descriptors_b = sift.detectAndCompute(gray_b, None)
        kp_count_b = int(len(keypoints_b) if keypoints_b is not None else 0)
    else:
        keypoints_b, descriptors_b, kp_count_b = master_features
    if descriptors_a is None or descriptors_b is None or kp_count_a == 0 or kp_count_b == 0:
        return {
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "score": 0.0,
            "accepted": False,
        }

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good_matches: list[Any] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < cfg.sift_min_ratio * n.distance:
            good_matches.append(m)
    if len(good_matches) > cfg.sift_max_ransac_matches:
        good_matches.sort(key=lambda match: match.distance)
        good_matches = good_matches[: cfg.sift_max_ransac_matches]

    if len(good_matches) < 4:
        return {
            "good_matches": int(len(good_matches)),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "score": 0.0,
            "accepted": False,
        }

    src_pts = np.float32([keypoints_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        4.0,
        maxIters=1000,
        confidence=0.995,
    )
    inliers = int(mask.ravel().sum()) if mask is not None else 0
    good_count = int(len(good_matches))
    inlier_ratio = float(inliers / good_count) if good_count > 0 else 0.0
    denom = float(max(1, min(kp_count_a, kp_count_b)))
    score = float(inliers / denom)
    accepted = inliers >= cfg.sift_min_inliers and inlier_ratio >= cfg.sift_min_inlier_ratio
    return {
        "good_matches": good_count,
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "score": score,
        "accepted": accepted,
    }


def clip_vector_for_record(record: dict[str, Any] | None) -> np.ndarray | None:
    if not record:
        return None
    groups = record.get("clip_groups")
    if not isinstance(groups, list):
        return None
    for group in groups:
        vector = group.get("clip_embedding")
        if isinstance(vector, (list, tuple)) and vector:
            return np.asarray(vector, dtype=np.float32)
    return None


def sift_clip_ann_candidates(
    table,
    file_name: str,
    vector: np.ndarray,
    topk: int,
    eligible_files: set[str],
) -> list[str]:
    # Over-fetch because the shared CLIP ANN table also contains videos,
    # extracted video frames, and pHash duplicates that SIFT does not compare.
    rows = table.search(vector).limit(max(topk * 16, topk + 1)).to_list()
    candidates: list[str] = []
    seen: set[str] = set()
    for row in rows:
        candidate = row.get("file_name")
        if (
            not isinstance(candidate, str)
            or candidate == file_name
            or candidate not in eligible_files
            or candidate in seen
        ):
            continue
        seen.add(candidate)
        candidates.append(candidate)
        if len(candidates) >= topk:
            break
    return candidates


def get_thread_sift(contrast_threshold: float, max_features: int) -> Any:
    sift = getattr(SIFT_THREAD_LOCAL, "sift", None)
    current_threshold = getattr(SIFT_THREAD_LOCAL, "contrast_threshold", None)
    current_max_features = getattr(SIFT_THREAD_LOCAL, "max_features", None)
    if (
        sift is None
        or current_threshold != contrast_threshold
        or current_max_features != max_features
    ):
        sift = cv2.SIFT_create(
            nfeatures=max_features,
            contrastThreshold=contrast_threshold,
        )
        SIFT_THREAD_LOCAL.sift = sift
        SIFT_THREAD_LOCAL.contrast_threshold = contrast_threshold
        SIFT_THREAD_LOCAL.max_features = max_features
    return sift


def get_thread_sift_feature_cache() -> OrderedDict[str, tuple[Any, Any, int]]:
    cache = getattr(SIFT_THREAD_LOCAL, "sift_feature_cache", None)
    if cache is None:
        cache = OrderedDict()
        SIFT_THREAD_LOCAL.sift_feature_cache = cache
    return cache


def evaluate_sift_candidates_for_item(
    cfg: AppConfig,
    image_paths: dict[str, Path],
    file_name: str,
    candidates: list[str],
) -> tuple[str | None, dict[str, Any] | None]:
    if not candidates:
        return None, None
    image_path = image_paths.get(file_name)
    if image_path is None:
        return None, None
    sift = get_thread_sift(cfg.sift_contrast_threshold, cfg.sift_max_features)
    # Reuse query SIFT features for every candidate instead of recomputing N times.
    img_query = read_image_bgr_resized(image_path, cfg.sift_max_side)
    gray_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2GRAY)
    kp_query, des_query = sift.detectAndCompute(gray_query, None)
    query_features = (
        kp_query,
        des_query,
        int(len(kp_query) if kp_query is not None else 0),
    )
    if query_features[1] is None or query_features[2] == 0:
        return None, None

    best_match: str | None = None
    best_metrics: dict[str, Any] | None = None
    feature_cache = get_thread_sift_feature_cache()
    for candidate in candidates:
        master_path = image_paths.get(candidate)
        if master_path is None:
            continue
        cache_key = str(master_path)
        master_features = feature_cache.get(cache_key)
        if master_features is None:
            img_master = read_image_bgr_resized(master_path, cfg.sift_max_side)
            gray_master = cv2.cvtColor(img_master, cv2.COLOR_BGR2GRAY)
            kp_master, des_master = sift.detectAndCompute(gray_master, None)
            master_features = (
                kp_master,
                des_master,
                int(len(kp_master) if kp_master is not None else 0),
            )
            feature_cache[cache_key] = master_features
            if len(feature_cache) > SIFT_FEATURE_CACHE_SIZE:
                feature_cache.popitem(last=False)
        else:
            feature_cache.move_to_end(cache_key)
        metrics = sift_match_pair(
            sift,
            image_path,
            master_path,
            cfg,
            query_features=query_features,
            master_features=master_features,
        )
        if not metrics["accepted"]:
            continue
        if best_metrics is None:
            best_match = candidate
            best_metrics = metrics
            continue
        # Prefer the strongest geometric match for stable DB grouping.
        if int(metrics["inliers"]) > int(best_metrics["inliers"]) or (
            int(metrics["inliers"]) == int(best_metrics["inliers"])
            and float(metrics["score"]) > float(best_metrics["score"])
        ):
            best_match = candidate
            best_metrics = metrics
    return best_match, best_metrics


def run_sift_master_match_stage(
    cfg: AppConfig,
    table,
    records: dict[str, dict[str, Any]],
    media_items: list[MediaItem],
) -> set[str]:
    image_paths = build_image_path_map(media_items)
    item_by_name = {item.file_name: item for item in media_items if not item.is_video}
    master_items: list[MediaItem] = []
    for file_name, record in records.items():
        if record.get("is_video") is True:
            continue
        if record.get("dedupe_match_file"):
            continue
        if file_name not in image_paths:
            continue
        item = item_by_name.get(file_name)
        if item is None:
            continue
        master_items.append(item)

    if not master_items:
        report_stage_complete("Stage 5a/8 SIFT CLIP ANN shortlist", 0, "image masters")
        report_stage_complete("Stage 5b/8 SIFT master match", 0, "image masters")
        return set()

    master_items.sort(key=lambda item: cached_image_master_sort_key(item, records))
    force = (
        cfg.rerun_sift_master_match
        or should_rerun_stage(cfg, "5a")
        or should_rerun_stage(cfg, "5b")
    )
    checked_count = sum(
        1
        for item in master_items
        if records.get(item.file_name, {}).get("sift_match_checked") is True
    )
    if not cfg.run_sift_master_match and not force:
        if checked_count == len(master_items):
            report_stage_complete("Stage 5a/8 SIFT CLIP ANN shortlist", checked_count, "image masters")
            report_stage_complete("Stage 5b/8 SIFT master match", checked_count, "image masters")
        else:
            report_stage_skipped(
                "Stage 5a/8 SIFT CLIP ANN shortlist",
                f"{checked_count}/{len(master_items)} image masters complete; "
                "pass --run-sift-master-match to process unchecked rows",
            )
            report_stage_skipped(
                "Stage 5b/8 SIFT master match",
                f"{checked_count}/{len(master_items)} image masters complete; "
                "pass --run-sift-master-match to process unchecked rows",
            )
        return set()
    if not hasattr(cv2, "SIFT_create"):
        report_stage_skipped("Stage 5a/8 SIFT CLIP ANN shortlist", "OpenCV SIFT is unavailable")
        report_stage_skipped("Stage 5b/8 SIFT master match", "OpenCV SIFT is unavailable")
        return set()

    db = lancedb.connect(str(cfg.db_dir))
    clip_table_name = ann_table_name(cfg.table_name, "clip")
    if clip_table_name not in db_table_names(db):
        report_stage_skipped("Stage 5a/8 SIFT CLIP ANN shortlist", "CLIP ANN table is unavailable")
        report_stage_skipped("Stage 5b/8 SIFT master match", "CLIP ANN table is unavailable")
        return set()
    clip_table = db.open_table(clip_table_name)
    eligible_files = {item.file_name for item in master_items}

    changed: set[str] = set()
    upsert_batch: list[dict[str, Any]] = []
    pending = master_items
    print(
        f"[sift-clip] master-only mode: evaluating {len(pending)} pHash-surviving masters "
        f"with top{cfg.sift_candidate_topk} CLIP candidates"
    )
    jobs: list[tuple[str, str, list[str]]] = []
    total_candidates = 0
    zero_candidate_jobs = 0
    for item in progress(
        pending,
        desc="Stage 5a/8 SIFT CLIP ANN shortlist",
        unit="file",
    ):
        base = records.get(item.file_name)
        if not base:
            continue
        if not force and base.get("sift_match_checked") is True:
            continue
        query_vector = clip_vector_for_record(base)
        if query_vector is None:
            zero_candidate_jobs += 1
            jobs.append((item.file_name, item.collection_id, []))
            continue

        candidates = sift_clip_ann_candidates(
            clip_table,
            item.file_name,
            query_vector,
            cfg.sift_candidate_topk,
            eligible_files,
        )
        total_candidates += len(candidates)
        if not candidates:
            zero_candidate_jobs += 1
        jobs.append((item.file_name, item.collection_id, candidates))

    if not jobs:
        report_stage_complete("Stage 5a/8 SIFT CLIP ANN shortlist", len(master_items), "image masters")
        report_stage_complete("Stage 5b/8 SIFT master match", len(master_items), "image masters")
        return changed
    report_stage_complete("Stage 5a/8 SIFT CLIP ANN shortlist", len(master_items), "image masters")
    avg_candidates = total_candidates / max(1, len(jobs))
    print(
        f"[sift-clip] candidate shortlist ready for {len(jobs)} masters; "
        f"{total_candidates} SIFT candidate pairs, avg {avg_candidates:.2f}/master, "
        f"{zero_candidate_jobs} with no candidates"
    )

    results: dict[str, tuple[str | None, dict[str, Any] | None]] = {}
    max_workers = max(1, int(cfg.hash_workers))
    max_in_flight = max_workers * 8
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        job_iter = iter(jobs)
        in_flight: set[Future] = set()
        future_to_job: dict[Future, tuple[str, str]] = {}

        def submit_next_job() -> bool:
            try:
                file_name, collection_id, candidates = next(job_iter)
            except StopIteration:
                return False
            future = executor.submit(
                evaluate_sift_candidates_for_item,
                cfg,
                image_paths,
                file_name,
                candidates,
            )
            in_flight.add(future)
            future_to_job[future] = (file_name, collection_id)
            return True

        for _ in range(min(max_in_flight, len(jobs))):
            if not submit_next_job():
                break

        pbar = HumanTqdm(
            total=len(jobs),
            desc=format_stage_label("Stage 5b/8 SIFT master match", stream=sys.stderr),
            unit="file",
            delay=PROGRESS_DELAY_SECONDS,
            smoothing=0.05,
            dynamic_ncols=True,
            bar_format=PROGRESS_BAR_FORMAT,
        )
        try:
            while in_flight:
                done_futures = next(as_completed(in_flight))
                completed: list[Future] = [done_futures]
                for future in completed:
                    in_flight.discard(future)
                    file_name, _ = future_to_job.pop(future)
                    try:
                        best_match, best_metrics = future.result()
                    except Exception as exc:
                        print_error(f"[sift] failed: {file_name}: {exc}")
                        best_match = None
                        best_metrics = None
                    results[file_name] = (best_match, best_metrics)
                    pbar.update(1)

                while len(in_flight) < max_in_flight and submit_next_job():
                    pass
        finally:
            pbar.close()

    direct_match: dict[str, tuple[str | None, dict[str, Any] | None]] = {}
    for file_name, _, _ in jobs:
        direct_match[file_name] = results.get(file_name, (None, None))

    # Canonicalize every connected SIFT component to the largest-dimension image.
    scope_names = {file_name for file_name, _, _ in jobs}
    for file_name in scope_names:
        match_name = direct_match.get(file_name, (None, None))[0]
        if not isinstance(match_name, str):
            continue
        if match_name not in scope_names:
            continue
        if match_name == file_name:
            continue
        # Keep only the link for group-building; master target resolution happens below.
        direct_match[file_name] = (match_name, direct_match[file_name][1])

    parent: dict[str, str] = {name: name for name in scope_names}

    def uf_find(name: str) -> str:
        root = parent[name]
        while root != parent[root]:
            root = parent[root]
        cur = name
        while cur != root:
            nxt = parent[cur]
            parent[cur] = root
            cur = nxt
        return root

    def uf_union(a: str, b: str) -> None:
        ra = uf_find(a)
        rb = uf_find(b)
        if ra == rb:
            return
        parent[rb] = ra

    for file_name, (match_name, _) in direct_match.items():
        if isinstance(match_name, str) and match_name in scope_names and match_name != file_name:
            uf_union(file_name, match_name)

    groups: dict[str, list[str]] = {}
    for name in scope_names:
        groups.setdefault(uf_find(name), []).append(name)

    def image_rank(name: str) -> tuple[int, int, int, int, str]:
        path = image_paths.get(name)
        if path is None:
            return (0, 0, 0, 0, name)
        record = records.get(name) or {}
        width = int(record.get("image_width") or 0)
        height = int(record.get("image_height") or 0)
        max_side = max(width, height)
        area = width * height
        file_size = int(record.get("source_size") or 0)
        non_thumb = 1 if not is_likely_thumbnail(path) else 0
        return (max_side, area, file_size, non_thumb, name)

    canonical_master: dict[str, str] = {}
    for members in groups.values():
        if len(members) == 1:
            canonical_master[members[0]] = members[0]
            continue
        best = max(members, key=image_rank)
        for member in members:
            canonical_master[member] = best

    for file_name, collection_id, _ in jobs:
        base = records.get(file_name)
        if not base:
            continue
        best_match, best_metrics = direct_match.get(file_name, (None, None))
        canonical = canonical_master.get(file_name, file_name)
        if canonical == file_name:
            best_match = None
            best_metrics = None
        else:
            best_match = canonical
            # If canonicalization changed the direct target, clear pair metrics to avoid
            # storing misleading inlier/score values for a link we didn't directly verify.
            direct_target = direct_match.get(file_name, (None, None))[0]
            if direct_target != canonical:
                best_metrics = None
        mutated = False
        if best_match is None:
            if base.get("sift_match_file") is not None:
                clear_sift_match_fields(base)
                mutated = True
            if base.get("sift_match_checked") is not True:
                base["sift_match_checked"] = True
                mutated = True
        else:
            if base.get("sift_match_file") != best_match:
                base["sift_match_file"] = best_match
                mutated = True
            if best_metrics is None:
                if (
                    base.get("sift_match_score") is not None
                    or base.get("sift_match_inliers") is not None
                    or base.get("sift_match_good_matches") is not None
                    or base.get("sift_match_inlier_ratio") is not None
                ):
                    base["sift_match_score"] = None
                    base["sift_match_inliers"] = None
                    base["sift_match_good_matches"] = None
                    base["sift_match_inlier_ratio"] = None
                    mutated = True
            else:
                score = round(float(best_metrics["score"]), 6)
                inliers = int(best_metrics["inliers"])
                good_matches = int(best_metrics["good_matches"])
                inlier_ratio = round(float(best_metrics["inlier_ratio"]), 6)
                if base.get("sift_match_score") != score:
                    base["sift_match_score"] = score
                    mutated = True
                if base.get("sift_match_inliers") != inliers:
                    base["sift_match_inliers"] = inliers
                    mutated = True
                if base.get("sift_match_good_matches") != good_matches:
                    base["sift_match_good_matches"] = good_matches
                    mutated = True
                if base.get("sift_match_inlier_ratio") != inlier_ratio:
                    base["sift_match_inlier_ratio"] = inlier_ratio
                    mutated = True
            if base.get("sift_match_checked") is not True:
                base["sift_match_checked"] = True
                mutated = True

        if mutated:
            base["collection_id"] = collection_id
            base["is_video"] = False
            recompute_aggregate_fields(base)
            append_stage_upsert(table, records, upsert_batch, base)
            changed.add(file_name)
    upsert_records_batch(table, records, upsert_batch)
    report_stage_complete("Stage 5b/8 SIFT master match", len(master_items), "image masters")
    return changed


def ann_table_name(base_table_name: str, modality: str) -> str:
    return f"{base_table_name}_{modality}_ann"


def make_face_ann_schema(vector_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("file_name", pa.string()),
            pa.field("timestamp_sec", pa.float32()),
            pa.field("face_index", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
        ]
    )


def make_clip_ann_schema(vector_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("file_name", pa.string()),
            pa.field("timestamp_sec", pa.float32()),
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
        ]
    )


def make_ocr_ann_schema(vector_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("file_name", pa.string()),
            pa.field("timestamp_sec", pa.float32()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
        ]
    )


def get_fixed_vector_dim(schema: pa.Schema, vector_field_name: str = "vector") -> int | None:
    try:
        field = schema.field(vector_field_name)
    except KeyError:
        return None
    field_type = field.type
    if pa.types.is_fixed_size_list(field_type):
        return int(field_type.list_size)
    return None


def schema_equivalent(a: pa.Schema, b: pa.Schema) -> bool:
    if list(a.names) != list(b.names):
        return False
    for name in a.names:
        if a.field(name).type != b.field(name).type:
            return False
    return True


def ensure_ann_table(db, table_name: str, schema: pa.Schema):
    if table_name not in db_table_names(db):
        return db.create_table(table_name, data=[], schema=schema), True
    table = db.open_table(table_name)
    if schema_equivalent(table.schema, schema):
        return table, False
    db.drop_table(table_name)
    return db.create_table(table_name, data=[], schema=schema), True


def safe_vector(vec: Any, expected_dim: int) -> list[float] | None:
    if not isinstance(vec, (list, tuple)):
        return None
    if len(vec) != expected_dim:
        return None
    try:
        return np.asarray(vec, dtype=np.float32).tolist()
    except Exception:
        return None


def iter_file_names_for_ann(file_names: set[str], desc: str) -> Iterable[str]:
    sorted_names = sorted(file_names)
    if len(sorted_names) >= 100:
        return progress(sorted_names, desc=desc, unit="file")
    return sorted_names


def build_face_ann_rows(records: dict[str, dict[str, Any]], file_names: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_name in iter_file_names_for_ann(file_names, "Stage 6b/8 Face ANN row build/delete build rows"):
        rec = records.get(file_name)
        if not rec:
            continue
        groups = rec.get("face_groups")
        if not isinstance(groups, list):
            continue
        for group in groups:
            ts = round_timestamp(float(group.get("timestamp_sec", 0.0)))
            embeddings = group.get("face_embeddings") or []
            for face_index, vec in enumerate(embeddings):
                clean = safe_vector(vec, FACE_VECTOR_DIM)
                if clean is None:
                    continue
                rows.append(
                    {
                        "id": f"{file_name}|{ts:.3f}|{face_index}",
                        "file_name": file_name,
                        "timestamp_sec": float(ts),
                        "face_index": int(face_index),
                        "vector": clean,
                    }
                )
    return rows


def build_clip_ann_rows(
    records: dict[str, dict[str, Any]],
    file_names: set[str],
    expected_dim: int | None = None,
) -> tuple[list[dict[str, Any]], int | None]:
    rows: list[dict[str, Any]] = []
    dim = expected_dim
    for file_name in iter_file_names_for_ann(file_names, "Stage 4b/8 CLIP ANN row build/delete build rows"):
        rec = records.get(file_name)
        if not rec:
            continue
        groups = rec.get("clip_groups")
        if not isinstance(groups, list):
            continue
        for group in groups:
            vec = group.get("clip_embedding")
            if not isinstance(vec, (list, tuple)) or len(vec) == 0:
                continue
            if dim is None:
                dim = len(vec)
            clean = safe_vector(vec, dim)
            if clean is None:
                continue
            ts = round_timestamp(float(group.get("timestamp_sec", 0.0)))
            rows.append(
                {
                    "id": f"{file_name}|{ts:.3f}",
                    "file_name": file_name,
                    "timestamp_sec": float(ts),
                    "vector": clean,
                }
            )
    return rows, dim


def infer_clip_vector_dim(records: dict[str, dict[str, Any]], file_names: set[str]) -> int | None:
    for file_name in sorted(file_names):
        rec = records.get(file_name)
        if not rec:
            continue
        groups = rec.get("clip_groups")
        if not isinstance(groups, list):
            continue
        for group in groups:
            vec = group.get("clip_embedding")
            if isinstance(vec, (list, tuple)) and len(vec) > 0:
                return len(vec)
    return None


def collect_ocr_entries(
    records: dict[str, dict[str, Any]],
    file_names: set[str],
) -> list[tuple[str, float, str]]:
    entries: list[tuple[str, float, str]] = []
    for file_name in iter_file_names_for_ann(file_names, "Stage 8b/8 OCR ANN row build/delete collect text"):
        rec = records.get(file_name)
        if not rec:
            continue
        groups = rec.get("ocr_groups")
        if not isinstance(groups, list):
            continue
        for group in groups:
            if group.get("text_detected") is not True:
                continue
            text = (group.get("text") or "").strip()
            if not text:
                continue
            ts = round_timestamp(float(group.get("timestamp_sec", 0.0)))
            entries.append((file_name, ts, text))
    return entries


def build_ocr_ann_rows_from_entries(
    entries: list[tuple[str, float, str]],
    embedder: TextEmbedder,
    expected_dim: int | None = None,
) -> tuple[list[dict[str, Any]], int | None]:
    if not entries:
        return [], expected_dim

    print_info(f"{format_stage_label('Stage 8b/8 OCR ANN row build/delete')}: embedding {len(entries)} OCR text entries")
    texts = [entry[2] for entry in entries]
    vectors = embedder.embed(texts)
    if not vectors:
        return [], expected_dim

    dim = expected_dim if expected_dim is not None else len(vectors[0])
    rows: list[dict[str, Any]] = []
    row_iter = enumerate(entries)
    if len(entries) >= 100:
        row_iter = progress(row_iter, desc="Stage 8b/8 OCR ANN row build/delete build rows", unit="row", total=len(entries))
    for i, (file_name, ts, text) in row_iter:
        vec = vectors[i]
        clean = safe_vector(vec, dim)
        if clean is None:
            continue
        rows.append(
            {
                "id": f"{file_name}|{ts:.3f}",
                "file_name": file_name,
                "timestamp_sec": float(ts),
                "text": text,
                "vector": clean,
            }
        )
    return rows, dim


def chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def delete_ann_rows_for_files(
    table,
    file_names: set[str],
    label: str = "ANN search index delete rows",
) -> None:
    names = sorted(file_names)
    if not names:
        return
    chunk_size = 500
    batches = list(chunks(names, chunk_size))
    iterator: Iterable[Sequence[str]]
    if len(names) >= 100:
        iterator = progress(batches, desc=label, unit="batch")
    else:
        iterator = batches
    for batch in iterator:
        if len(batch) == 1:
            table.delete(f"file_name = {sql_string_literal(batch[0])}")
            continue
        literals = ", ".join(sql_string_literal(file_name) for file_name in batch)
        table.delete(f"file_name IN ({literals})")


def delete_ann_rows_for_files_from_all_ann(cfg: AppConfig, file_names: set[str]) -> None:
    if not file_names:
        return
    db = lancedb.connect(str(cfg.db_dir))
    for table_name in (
        ann_table_name(cfg.table_name, "face"),
        ann_table_name(cfg.table_name, "clip"),
        ann_table_name(cfg.table_name, "ocr"),
    ):
        if table_name not in db_table_names(db):
            continue
        table = db.open_table(table_name)
        delete_ann_rows_for_files(table, file_names, label=f"ANN search index cleanup {table_name} delete rows")


def ensure_ann_index(table, index_name: str, stage_label: str) -> None:
    if table.to_arrow().num_rows == 0:
        return
    index_names = {index.name for index in table.list_indices()}
    if index_name in index_names:
        return
    print_info(f"{format_stage_label(stage_label)}: building vector index")
    table.create_index(
        vector_column_name="vector",
        metric=ANN_DISTANCE_METRIC,
        index_type=ANN_INDEX_TYPE,
        replace=False,
        name=index_name,
    )
    table.wait_for_index([index_name])
    print_info(f"{format_stage_label(stage_label)}: vector index ready")


def sync_face_ann_table(
    db,
    base_table_name: str,
    records: dict[str, dict[str, Any]],
    target_file_names: set[str],
) -> None:
    table_name = ann_table_name(base_table_name, "face")
    schema = make_face_ann_schema(FACE_VECTOR_DIM)
    table, recreated = ensure_ann_table(db, table_name, schema)
    print_info(f"{format_stage_label('Stage 6b/8 Face ANN row build/delete')}: syncing {len(target_file_names)} files")
    if recreated:
        print_info(f"{format_stage_label('Stage 6b/8 Face ANN row build/delete')}: {table_name} was created/recreated; skipping delete pass")
    else:
        delete_ann_rows_for_files(table, target_file_names, label="Stage 6b/8 Face ANN row build/delete delete stale rows")
    rows = build_face_ann_rows(records, target_file_names)
    if rows:
        print_info(f"{format_stage_label('Stage 6b/8 Face ANN row build/delete')}: adding {len(rows)} face vectors")
        table.add(rows)
    ensure_ann_index(table, f"{table_name}_vec_idx", "Stage 6c/8 Face ANN index build/finalize")


def sync_clip_ann_table(
    db,
    base_table_name: str,
    records: dict[str, dict[str, Any]],
    target_file_names: set[str],
) -> None:
    table_name = ann_table_name(base_table_name, "clip")
    table = db.open_table(table_name) if table_name in db_table_names(db) else None
    existing_dim = get_fixed_vector_dim(table.schema) if table is not None else None
    target_dim = infer_clip_vector_dim(records, target_file_names)
    if existing_dim is not None and target_dim is not None and target_dim != existing_dim:
        print_info(f"{format_stage_label('Stage 4b/8 CLIP ANN row build/delete')}: recreating {table_name}; vector dimension changed {existing_dim} -> {target_dim}")
        db.drop_table(table_name)
        table = None
        existing_dim = None
    rows, inferred_dim = build_clip_ann_rows(records, target_file_names, expected_dim=existing_dim)
    dim = existing_dim if existing_dim is not None else (target_dim if target_dim is not None else inferred_dim)
    if dim is None:
        return

    schema = make_clip_ann_schema(dim)
    table, recreated = ensure_ann_table(db, table_name, schema)
    print_info(f"{format_stage_label('Stage 4b/8 CLIP ANN row build/delete')}: syncing {len(target_file_names)} files")
    if recreated:
        print_info(f"{format_stage_label('Stage 4b/8 CLIP ANN row build/delete')}: {table_name} was created/recreated; skipping delete pass")
    else:
        delete_ann_rows_for_files(table, target_file_names, label="Stage 4b/8 CLIP ANN row build/delete delete stale rows")
    if rows:
        print_info(f"{format_stage_label('Stage 4b/8 CLIP ANN row build/delete')}: adding {len(rows)} CLIP vectors")
        table.add(rows)
    ensure_ann_index(table, f"{table_name}_vec_idx", "Stage 4c/8 CLIP ANN index build/finalize")


def sync_ocr_ann_table(
    db,
    cfg: AppConfig,
    base_table_name: str,
    records: dict[str, dict[str, Any]],
    target_file_names: set[str],
) -> None:
    table_name = ann_table_name(base_table_name, "ocr")
    table = db.open_table(table_name) if table_name in db_table_names(db) else None
    existing_dim = get_fixed_vector_dim(table.schema) if table is not None else None
    entries = collect_ocr_entries(records, target_file_names)
    if not entries and existing_dim is None:
        return
    if not entries and existing_dim is not None:
        schema = make_ocr_ann_schema(existing_dim)
        table, recreated = ensure_ann_table(db, table_name, schema)
        print_info(f"{format_stage_label('Stage 8b/8 OCR ANN row build/delete')}: clearing rows for {len(target_file_names)} files")
        if recreated:
            print_info(f"{format_stage_label('Stage 8b/8 OCR ANN row build/delete')}: {table_name} was created/recreated; skipping delete pass")
        else:
            delete_ann_rows_for_files(table, target_file_names, label="Stage 8b/8 OCR ANN row build/delete delete stale rows")
        ensure_ann_index(table, f"{table_name}_vec_idx", "Stage 8c/8 OCR ANN index build/finalize")
        return

    embedder = TextEmbedder(cfg.ocr_text_model, cfg.ann_text_batch_size, cfg.ocr_text_device)
    if entries:
        probe_vec = embedder.embed([entries[0][2]])
        target_dim = len(probe_vec[0]) if probe_vec else None
    else:
        target_dim = None
    if existing_dim is not None and target_dim is not None and target_dim != existing_dim:
        print_info(f"{format_stage_label('Stage 8b/8 OCR ANN row build/delete')}: recreating {table_name}; vector dimension changed {existing_dim} -> {target_dim}")
        db.drop_table(table_name)
        table = None
        existing_dim = None
    rows, inferred_dim = build_ocr_ann_rows_from_entries(entries, embedder, expected_dim=existing_dim)
    dim = existing_dim if existing_dim is not None else (target_dim if target_dim is not None else inferred_dim)
    if dim is None:
        return

    schema = make_ocr_ann_schema(dim)
    table, recreated = ensure_ann_table(db, table_name, schema)
    print_info(f"{format_stage_label('Stage 8b/8 OCR ANN row build/delete')}: syncing {len(target_file_names)} files")
    if recreated:
        print_info(f"{format_stage_label('Stage 8b/8 OCR ANN row build/delete')}: {table_name} was created/recreated; skipping delete pass")
    else:
        delete_ann_rows_for_files(table, target_file_names, label="Stage 8b/8 OCR ANN row build/delete delete stale rows")
    if rows:
        print_info(f"{format_stage_label('Stage 8b/8 OCR ANN row build/delete')}: adding {len(rows)} OCR text vectors")
        table.add(rows)
    ensure_ann_index(table, f"{table_name}_vec_idx", "Stage 8c/8 OCR ANN index build/finalize")


def has_searchable_ocr_text(records: dict[str, dict[str, Any]]) -> bool:
    for rec in records.values():
        groups = rec.get("ocr_groups")
        if not isinstance(groups, list):
            continue
        for group in groups:
            if group.get("text_detected") is True and (group.get("text") or "").strip():
                return True
    return False


def missing_ann_tables(db, base_table_name: str, records: dict[str, dict[str, Any]]) -> set[str]:
    required = {
        ann_table_name(base_table_name, "face"),
        ann_table_name(base_table_name, "clip"),
    }
    if has_searchable_ocr_text(records):
        required.add(ann_table_name(base_table_name, "ocr"))
    existing = db_table_names(db)
    return required - existing


def main() -> None:
    cfg = parse_args()
    if not cfg.input_dir.exists() or not cfg.input_dir.is_dir():
        raise RuntimeError(f"Input directory does not exist or is not a directory: {cfg.input_dir}")

    with TimedStep(f"scan media under {cfg.input_dir}"):
        media_paths = discover_media_files(cfg.input_dir)
    if not media_paths:
        print("No supported photos/videos found.")
        return

    image_paths, video_paths = split_media(media_paths)
    print_info(f"{format_stage_label('Stage 0b/8 Startup split media files')}: {len(image_paths)} images, {len(video_paths)} videos")
    table = connect_table(cfg.db_dir, cfg.table_name)
    records = load_records(table)
    migrate_legacy_records_for_scan(cfg, table, records, media_paths)

    if cfg.repair_only:
        media_items = build_media_items(cfg, image_paths, [], {})
        changed_file_names = run_phash_gate_stage(cfg, table, records, media_items, [], {})
        cleared_processing: set[str] = set()
        if cfg.repair_image_masters:
            repair_changed, cleared_processing = repair_image_masters(cfg, table, records, media_items)
            changed_file_names |= repair_changed
            delete_ann_rows_for_files_from_all_ann(cfg, cleared_processing)
        print(
            f"Repair complete. Changed rows: {len(changed_file_names)}. "
            f"Rows with processing cleared: {len(cleared_processing)}."
        )
        return

    changed_file_names = run_video_hash_gate_stage(cfg, table, records, video_paths)
    _, video_frame_map = extract_video_stills(cfg, video_paths, records)
    media_items = build_media_items(cfg, image_paths, video_paths, video_frame_map)
    if not media_items:
        report_stage_complete("Stage 3a/8 cached image metadata", 0, "images")
        report_stage_complete("Stage 3b/8 pHash images", 0, "files")
        report_stage_complete("Stage 3d/8 apply image pHash groups", 0, "files")
        changed_file_names |= run_video_frame_phash_stage(
            cfg,
            table,
            records,
            video_paths,
            video_frame_map,
        )
        changed_file_names |= run_cross_media_match_stage(
            cfg,
            table,
            records,
            image_paths,
            video_paths,
        )
        db = lancedb.connect(str(cfg.db_dir))
        missing_tables = missing_ann_tables(db, cfg.table_name, records)
        face_table_name = ann_table_name(cfg.table_name, "face")
        clip_table_name = ann_table_name(cfg.table_name, "clip")
        ocr_table_name = ann_table_name(cfg.table_name, "ocr")
        force_clip_sync = should_rerun_stage(cfg, "4b")
        force_face_sync = should_rerun_stage(cfg, "6b")
        force_ocr_sync = should_rerun_stage(cfg, "8b")
        if missing_tables:
            if face_table_name in missing_tables:
                sync_face_ann_table(db, cfg.table_name, records, all_file_names)
            if clip_table_name in missing_tables:
                sync_clip_ann_table(db, cfg.table_name, records, all_file_names)
            if ocr_table_name in missing_tables:
                sync_ocr_ann_table(db, cfg, cfg.table_name, records, all_file_names)
            print("No new media required processing. Missing ANN side tables were rebuilt from existing records.")
        else:
            print("No new or incomplete media items require processing.")
        all_file_names = set(records.keys())
        if force_clip_sync and clip_table_name not in missing_tables:
            sync_clip_ann_table(db, cfg.table_name, records, all_file_names)
        if force_face_sync and face_table_name not in missing_tables:
            sync_face_ann_table(db, cfg.table_name, records, all_file_names)
        if force_ocr_sync and ocr_table_name not in missing_tables:
            sync_ocr_ann_table(db, cfg, cfg.table_name, records, all_file_names)
        report_stage_complete("Stage 4a/8 CLIP embeddings", 0, "files")
        if not force_clip_sync and clip_table_name not in missing_tables:
            report_stage_complete("Stage 4b/8 CLIP ANN row build/delete", 0, "files")
            report_stage_complete("Stage 4c/8 CLIP ANN index build/finalize", 0, "files")
        report_stage_complete("Stage 5a/8 SIFT CLIP ANN shortlist", 0, "image masters")
        report_stage_complete("Stage 5b/8 SIFT master match", 0, "image masters")
        report_stage_complete("Stage 6a/8 Faces", 0, "files")
        if not force_face_sync and face_table_name not in missing_tables:
            report_stage_complete("Stage 6b/8 Face ANN row build/delete", 0, "files")
            report_stage_complete("Stage 6c/8 Face ANN index build/finalize", 0, "files")
        if cfg.skip_paddle_ocr and not should_rerun_stage(cfg, "7"):
            report_stage_skipped("Stage 7/8 PaddleOCR", "--skip-paddle-ocr was requested")
        else:
            report_stage_complete("Stage 7/8 PaddleOCR", 0, "files")
        report_stage_complete("Stage 8a/8 EasyOCR text extraction", 0, "files")
        if not force_ocr_sync and ocr_table_name not in missing_tables:
            report_stage_complete("Stage 8b/8 OCR ANN row build/delete", 0, "files")
            report_stage_complete("Stage 8c/8 OCR ANN index build/finalize", 0, "files")
        return

    changed_file_names |= run_phash_gate_stage(
        cfg,
        table,
        records,
        media_items,
        video_paths,
        video_frame_map,
    )
    changed_file_names |= run_cross_media_match_stage(
        cfg,
        table,
        records,
        image_paths,
        video_paths,
    )
    if cfg.repair_image_masters:
        repair_changed, cleared_processing = repair_image_masters(cfg, table, records, media_items)
        changed_file_names |= repair_changed
        delete_ann_rows_for_files_from_all_ann(cfg, cleared_processing)
    clip_changed = run_clip_stage(cfg, table, records, media_items)
    changed_file_names |= clip_changed
    db = lancedb.connect(str(cfg.db_dir))
    clip_table_name = ann_table_name(cfg.table_name, "clip")
    clip_sync_files = (
        set(records.keys())
        if should_rerun_stage(cfg, "4b") or clip_table_name not in db_table_names(db)
        else clip_changed
    )
    if clip_sync_files:
        sync_clip_ann_table(db, cfg.table_name, records, clip_sync_files)
    else:
        report_stage_complete("Stage 4b/8 CLIP ANN row build/delete", 0, "files")
        report_stage_complete("Stage 4c/8 CLIP ANN index build/finalize", 0, "files")
    changed_file_names |= run_sift_master_match_stage(cfg, table, records, media_items)
    face_changed = run_face_stage(cfg, table, records, media_items)
    changed_file_names |= face_changed
    face_table_name = ann_table_name(cfg.table_name, "face")
    face_sync_files = (
        set(records.keys())
        if should_rerun_stage(cfg, "6b") or face_table_name not in db_table_names(db)
        else face_changed
    )
    if face_sync_files:
        sync_face_ann_table(db, cfg.table_name, records, face_sync_files)
    else:
        report_stage_complete("Stage 6b/8 Face ANN row build/delete", 0, "files")
        report_stage_complete("Stage 6c/8 Face ANN index build/finalize", 0, "files")
    if cfg.wipe_paddle_failures_before_run:
        reset_count = wipe_failed_paddle_rows_for_run(table, records, media_items)
        if reset_count > 0:
            print(f"[paddle] cleared {reset_count} previously failed paddle_ocr rows for this run")
    paddle_changed: set[str] = set()
    if not cfg.skip_paddle_ocr or should_rerun_stage(cfg, "7"):
        paddle_changed = run_paddle_detection_stage(cfg, table, records, media_items)
        changed_file_names |= paddle_changed
    else:
        report_stage_skipped("Stage 7/8 PaddleOCR", "--skip-paddle-ocr was requested")
    easyocr_changed = run_easyocr_stage(cfg, table, records, media_items)
    changed_file_names |= easyocr_changed
    ocr_table_name = ann_table_name(cfg.table_name, "ocr")
    ocr_sync_files = (
        set(records.keys())
        if should_rerun_stage(cfg, "8b") or ocr_table_name not in db_table_names(db)
        else (paddle_changed | easyocr_changed)
    )
    if ocr_sync_files:
        sync_ocr_ann_table(db, cfg, cfg.table_name, records, ocr_sync_files)
    else:
        report_stage_complete("Stage 8b/8 OCR ANN row build/delete", 0, "files")
        report_stage_complete("Stage 8c/8 OCR ANN index build/finalize", 0, "files")
    print_summary(records, media_items)


if __name__ == "__main__":
    main()
