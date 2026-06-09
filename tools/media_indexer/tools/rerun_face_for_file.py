#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

from main import (
    FaceEmbedder,
    SCHEMA,
    compact_error_message,
    default_insightface_root,
    read_image_bgr,
    recompute_aggregate_fields,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun face detection for one DB file row.")
    parser.add_argument("--db-dir", required=True, type=Path)
    parser.add_argument("--table", default="media_index")
    parser.add_argument("--file-name", required=True)
    parser.add_argument("--collection-root", action="append", default=[], metavar="COLLECTION_ID=/ABS/PATH")
    parser.add_argument("--insightface-root", type=Path, default=default_insightface_root())
    parser.add_argument("--det-threshold", type=float, default=0.25)
    return parser.parse_args()


def parse_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        collection, sep, root = value.partition("=")
        if not sep or not collection or not root:
            raise SystemExit(f"invalid --collection-root: {value}")
        roots[collection] = Path(root)
    return roots


def resolve_path(roots: dict[str, Path], file_name: str) -> Path:
    collection, sep, rel = file_name.partition("/")
    if not sep:
        raise SystemExit(f"file_name has no collection id: {file_name}")
    root = roots.get(collection)
    if root is None:
        raise SystemExit(f"no --collection-root supplied for collection {collection}")
    return root / rel


def escape_sql(value: str) -> str:
    return value.replace("'", "''")


def upsert_row(table: Any, row: dict[str, Any]) -> None:
    arrow_table = pa.Table.from_pylist([row], schema=SCHEMA)
    if hasattr(table, "merge_insert"):
        (
            table.merge_insert("file_name")
            .use_index(True)
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(arrow_table)
        )
        return
    table.delete(f"file_name = '{escape_sql(row['file_name'])}'")
    table.add(arrow_table)


def main() -> int:
    args = parse_args()
    roots = parse_roots(args.collection_root)
    path = resolve_path(roots, args.file_name)
    if not path.exists():
        raise SystemExit(f"source file does not exist: {path}")

    db = lancedb.connect(str(args.db_dir))
    table = db.open_table(args.table)
    rows_by_name = {row["file_name"]: row for row in table.to_arrow().to_pylist()}
    row = rows_by_name.get(args.file_name)
    if row is None:
        raise SystemExit(f"file not found in DB: {args.file_name}")
    if row.get("is_video") is True:
        raise SystemExit("single-file face rerun currently expects an image row")

    updated = dict(row)
    try:
        frame = read_image_bgr(path)
        embedder = FaceEmbedder(args.insightface_root.expanduser(), det_thresh=args.det_threshold)
        embeddings = embedder.detect_and_embed_frame(frame)
        updated["face_groups"] = [
            {
                "timestamp_sec": 0.0,
                "face_embeddings": embeddings,
            }
        ]
        if updated.get("processing_error_stage") == "faces":
            updated["processing_error_stage"] = None
            updated["processing_error"] = None
    except Exception as exc:
        updated["face_groups"] = [
            {
                "timestamp_sec": 0.0,
                "face_embeddings": [],
            }
        ]
        updated["processing_error_stage"] = "faces"
        updated["processing_error"] = compact_error_message(exc)

    recompute_aggregate_fields(updated)
    upsert_row(table, updated)
    print(
        json.dumps(
            {
                "file_name": args.file_name,
                "face_count": len(updated["face_groups"][0]["face_embeddings"]),
                "faces": bool(updated.get("faces")),
                "det_threshold": float(args.det_threshold),
                "processing_error_stage": updated.get("processing_error_stage"),
                "processing_error": updated.get("processing_error"),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
