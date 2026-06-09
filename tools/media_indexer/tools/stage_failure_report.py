#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import lancedb


@dataclass
class StageStats:
    total_rows: int = 0
    masters_total: int = 0
    image_rows: int = 0
    video_rows: int = 0

    vhash_failures: int = 0
    vhash_successes: int = 0
    vhash_not_run: int = 0
    vhash_masters: int = 0
    vhash_similars: int = 0

    phash_failures: int = 0
    phash_successes: int = 0
    phash_not_run: int = 0
    phash_masters: int = 0
    phash_similars: int = 0

    face_detect_failures: int = 0
    face_embed_failures: int = 0
    face_successes: int = 0
    face_not_run: int = 0
    face_success_with_faces: int = 0
    face_success_without_faces: int = 0

    clip_failures: int = 0
    clip_successes: int = 0
    clip_not_run: int = 0

    sift_failures: int = 0
    sift_successes: int = 0
    sift_not_run: int = 0
    sift_success_with_similar: int = 0
    sift_success_without_similar: int = 0

    paddle_failures: int = 0
    paddle_successes: int = 0
    paddle_not_run: int = 0
    paddle_success_with_text: int = 0
    paddle_success_without_text: int = 0

    easyocr_failures: int = 0
    easyocr_successes: int = 0
    easyocr_not_run: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report stage failures/success/not-run counts per collection_id for "
            "vHash, pHash, Faces, SIFT, PaddleOCR, and EasyOCR."
        )
    )
    parser.add_argument("--db-dir", type=Path, required=True)
    parser.add_argument("--table", default="media_index")
    parser.add_argument(
        "--collection-id",
        action="append",
        default=[],
        help="Optional filter. Repeat to include multiple collection IDs.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text table.")
    return parser.parse_args()


def easyocr_pending(ocr_groups: Any) -> bool:
    if not isinstance(ocr_groups, list):
        return False
    for group in ocr_groups:
        if group.get("text_detected") is True and group.get("text") is None:
            return True
    return False


def clip_complete(clip_groups: Any) -> bool:
    if not isinstance(clip_groups, list) or len(clip_groups) == 0:
        return False
    return all(group.get("clip_embedding") is not None for group in clip_groups)


def paddle_complete(ocr_groups: Any) -> bool:
    if not isinstance(ocr_groups, list) or len(ocr_groups) == 0:
        return False
    return all(group.get("text_detected") in (True, False) for group in ocr_groups)


def paddle_has_text(ocr_groups: Any) -> bool:
    if not isinstance(ocr_groups, list):
        return False
    return any(group.get("text_detected") is True for group in ocr_groups)


def classify_face_failure(processing_error: str) -> tuple[int, int]:
    text = processing_error.lower()
    if "embed" in text or "embedding" in text:
        return 0, 1
    return 1, 0


def format_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No rows."
    columns = list(rows[0].keys())
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(str(row[col])))
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    sep = "-+-".join("-" * widths[col] for col in columns)
    lines = [header, sep]
    for row in rows:
        lines.append(" | ".join(str(row[col]).rjust(widths[col]) if isinstance(row[col], int) else str(row[col]).ljust(widths[col]) for col in columns))
    return "\n".join(lines)


def stage_rows(
    stats_by_collection: dict[str, StageStats],
    stats_total: StageStats,
    columns: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for collection_id in sorted(stats_by_collection.keys()):
        stats = stats_by_collection[collection_id]
        row: dict[str, Any] = {"collection_id": collection_id}
        for heading, attr in columns:
            row[heading] = getattr(stats, attr)
        rows.append(row)
    total_row: dict[str, Any] = {"collection_id": "__TOTAL__"}
    for heading, attr in columns:
        total_row[heading] = getattr(stats_total, attr)
    rows.append(total_row)
    return rows


def main() -> int:
    args = parse_args()
    db = lancedb.connect(str(args.db_dir))
    table = db.open_table(args.table)
    selected_columns = [
        "collection_id",
        "is_video",
        "phash_hex",
        "dedupe_match_file",
        "faces",
        "clip_groups",
        "ocr_groups",
        "sift_match_file",
        "sift_match_checked",
        "processing_error_stage",
        "processing_error",
    ]
    rows = table.search(None).select(selected_columns).to_arrow().to_pylist()

    filter_ids = set(args.collection_id or [])
    stats_by_collection: dict[str, StageStats] = defaultdict(StageStats)
    stats_total = StageStats()

    for row in rows:
        collection_id = row.get("collection_id") or "<missing>"
        if filter_ids and collection_id not in filter_ids:
            continue
        per_collection = stats_by_collection[collection_id]
        for stats in (per_collection, stats_total):
            stats.total_rows += 1

        is_video = bool(row.get("is_video"))
        phash_hex = row.get("phash_hex")
        dedupe_match_file = row.get("dedupe_match_file")
        processing_error_stage = str(row.get("processing_error_stage") or "")
        processing_error = str(row.get("processing_error") or "")
        faces_value = row.get("faces")
        clip_groups = row.get("clip_groups")
        ocr_groups = row.get("ocr_groups")
        sift_checked = row.get("sift_match_checked") is True
        sift_match_file = row.get("sift_match_file")

        for stats in (per_collection, stats_total):
            if is_video:
                stats.video_rows += 1
            else:
                stats.image_rows += 1

            is_similar = isinstance(dedupe_match_file, str) and dedupe_match_file.strip()
            is_master_scope = not is_similar
            if is_master_scope:
                stats.masters_total += 1

            if is_video:
                if processing_error_stage == "video_hash":
                    stats.vhash_failures += 1
                elif phash_hex:
                    stats.vhash_successes += 1
                else:
                    stats.vhash_not_run += 1
                if is_similar:
                    stats.vhash_similars += 1
                else:
                    stats.vhash_masters += 1
            else:
                if processing_error_stage == "image_phash":
                    stats.phash_failures += 1
                elif phash_hex:
                    stats.phash_successes += 1
                else:
                    stats.phash_not_run += 1
                if is_similar:
                    stats.phash_similars += 1
                else:
                    stats.phash_masters += 1

            if is_master_scope:
                if processing_error_stage == "faces":
                    detect_fail, embed_fail = classify_face_failure(processing_error)
                    stats.face_detect_failures += detect_fail
                    stats.face_embed_failures += embed_fail
                elif faces_value is None:
                    stats.face_not_run += 1
                else:
                    stats.face_successes += 1
                    if faces_value is True:
                        stats.face_success_with_faces += 1
                    else:
                        stats.face_success_without_faces += 1

                if processing_error_stage == "clip":
                    stats.clip_failures += 1
                elif clip_complete(clip_groups):
                    stats.clip_successes += 1
                else:
                    stats.clip_not_run += 1

            # SIFT runs on pHash-surviving image masters; exclude pHash similars.
            is_sift_master_scope = (
                (not is_video)
                and is_master_scope
            )
            if is_sift_master_scope:
                if processing_error_stage.startswith("sift"):
                    stats.sift_failures += 1
                if sift_checked:
                    stats.sift_successes += 1
                    if isinstance(sift_match_file, str) and sift_match_file.strip():
                        stats.sift_success_with_similar += 1
                    else:
                        stats.sift_success_without_similar += 1
                else:
                    stats.sift_not_run += 1

            if is_master_scope:
                if processing_error_stage == "paddle_ocr":
                    stats.paddle_failures += 1
                elif paddle_complete(ocr_groups):
                    stats.paddle_successes += 1
                    if paddle_has_text(ocr_groups):
                        stats.paddle_success_with_text += 1
                    else:
                        stats.paddle_success_without_text += 1
                else:
                    stats.paddle_not_run += 1

                pending_easyocr = easyocr_pending(ocr_groups)
                if processing_error_stage == "easyocr" or pending_easyocr:
                    stats.easyocr_failures += 1
                elif isinstance(ocr_groups, list):
                    stats.easyocr_successes += 1
                else:
                    stats.easyocr_not_run += 1

    ordered_collection_ids = sorted(stats_by_collection.keys())
    output_rows: list[dict[str, Any]] = []
    for collection_id in ordered_collection_ids:
        output_rows.append({"collection_id": collection_id, **asdict(stats_by_collection[collection_id])})
    output_rows.append({"collection_id": "__TOTAL__", **asdict(stats_total)})

    if args.json:
        print(json.dumps(output_rows, indent=2, ensure_ascii=True))
    else:
        stage_defs: list[tuple[str, list[tuple[str, str]]]] = [
            (
                "Stage 1: vHash (Videos)",
                [
                    ("videos", "video_rows"),
                    ("successes", "vhash_successes"),
                    ("failures", "vhash_failures"),
                    ("not_run", "vhash_not_run"),
                    ("masters", "vhash_masters"),
                    ("vhash_similars", "vhash_similars"),
                ],
            ),
            (
                "Stage 2: pHash (Images)",
                [
                    ("images", "image_rows"),
                    ("successes", "phash_successes"),
                    ("failures", "phash_failures"),
                    ("not_run", "phash_not_run"),
                    ("masters", "phash_masters"),
                    ("phash_similars", "phash_similars"),
                ],
            ),
            (
                "Stage 3: SIFT",
                [
                    ("masters", "phash_masters"),
                    ("successes", "sift_successes"),
                    ("failures", "sift_failures"),
                    ("not_run", "sift_not_run"),
                    ("with_similar", "sift_success_with_similar"),
                    ("without_similar", "sift_success_without_similar"),
                ],
            ),
            (
                "Stage 4: Face Detection/Embedding",
                [
                    ("masters", "masters_total"),
                    ("successes", "face_successes"),
                    ("detect_fail", "face_detect_failures"),
                    ("embed_fail", "face_embed_failures"),
                    ("not_run", "face_not_run"),
                    ("with_faces", "face_success_with_faces"),
                    ("without_faces", "face_success_without_faces"),
                ],
            ),
            (
                "Stage 5: CLIP",
                [
                    ("masters", "masters_total"),
                    ("successes", "clip_successes"),
                    ("failures", "clip_failures"),
                    ("not_run", "clip_not_run"),
                ],
            ),
            (
                "Stage 6: PaddleOCR Detection",
                [
                    ("masters", "masters_total"),
                    ("successes", "paddle_successes"),
                    ("failures", "paddle_failures"),
                    ("not_run", "paddle_not_run"),
                    ("with_text", "paddle_success_with_text"),
                    ("without_text", "paddle_success_without_text"),
                ],
            ),
            (
                "Stage 7: EasyOCR Extraction",
                [
                    ("masters", "masters_total"),
                    ("successes", "easyocr_successes"),
                    ("failures", "easyocr_failures"),
                    ("not_run", "easyocr_not_run"),
                ],
            ),
            (
                "Stage 8: Overall Rows",
                [
                    ("total_rows", "total_rows"),
                    ("image_rows", "image_rows"),
                    ("video_rows", "video_rows"),
                ],
            ),
        ]
        for stage_name, columns in stage_defs:
            print(stage_name)
            rows_for_stage = stage_rows(stats_by_collection, stats_total, columns)
            print(format_table(rows_for_stage))
            print()
        print()
        print("Notes:")
        print("- `face_detect_failures` / `face_embed_failures` are split heuristically from `processing_error` text when stage=`faces`.")
        print("- `sift_failures` only counts explicit stage markers (none are usually persisted); see `sift_not_run` / `sift_successes`.")
        print("- `easyocr_failures` includes explicit `easyocr` errors and inferred pending rows (`text_detected=true` with `text=null`).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
