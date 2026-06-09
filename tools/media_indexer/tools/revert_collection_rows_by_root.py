#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lancedb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Revert accidental rows for one collection by deleting DB rows whose "
            "file_name does not exist under the intended collection root."
        )
    )
    parser.add_argument("--db-dir", type=Path, required=True)
    parser.add_argument("--table", default="media_index")
    parser.add_argument("--collection-id", required=True)
    parser.add_argument(
        "--collection-root",
        type=Path,
        required=True,
        help="Expected on-disk root for this collection id (e.g. /path/to/media-root).",
    )
    parser.add_argument(
        "--include-ann",
        action="store_true",
        help="Also delete matching file_name rows from *_face_ann, *_clip_ann, *_ocr_ann, and *_sift_bovw_ann tables.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Delete batch size for SQL IN(...) predicates.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply deletion. Without this flag, only prints a dry-run summary.",
    )
    parser.add_argument(
        "--show-sample",
        type=int,
        default=20,
        help="Number of suspect file_names to print in dry-run output.",
    )
    return parser.parse_args()


def escape_sql(value: str) -> str:
    return value.replace("'", "''")


def chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def delete_file_names(table: Any, file_names: list[str], batch_size: int) -> None:
    for batch in chunked(file_names, max(1, int(batch_size))):
        sql_values = ", ".join(f"'{escape_sql(v)}'" for v in batch)
        table.delete(f"file_name IN ({sql_values})")


def ann_table_names(base_table_name: str) -> list[str]:
    return [
        f"{base_table_name}_face_ann",
        f"{base_table_name}_clip_ann",
        f"{base_table_name}_ocr_ann",
        f"{base_table_name}_sift_bovw_ann",
    ]


def relative_part(file_name: str, collection_id: str) -> str | None:
    prefix = f"{collection_id}/"
    if file_name.startswith(prefix):
        return file_name[len(prefix) :]
    # Fallback for odd legacy rows.
    _, sep, rest = file_name.partition("/")
    return rest if sep else None


def main() -> int:
    args = parse_args()
    db = lancedb.connect(str(args.db_dir))
    base_table = db.open_table(args.table)
    rows = (
        base_table.search(None)
        .select(["file_name", "collection_id"])
        .where(f"collection_id = '{escape_sql(args.collection_id)}'")
        .to_arrow()
        .to_pylist()
    )

    root = args.collection_root.expanduser().resolve()
    suspects: list[str] = []
    checked = 0
    for row in rows:
        file_name = row.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            continue
        rel = relative_part(file_name, args.collection_id)
        if not rel:
            continue
        checked += 1
        expected_path = root / rel
        if not expected_path.exists():
            suspects.append(file_name)

    summary = {
        "collection_id": args.collection_id,
        "collection_root": str(root),
        "rows_in_collection": len(rows),
        "rows_checked": checked,
        "suspect_rows": len(suspects),
        "apply": bool(args.apply),
        "include_ann": bool(args.include_ann),
    }

    if not args.apply:
        sample_n = max(0, int(args.show_sample))
        sample = suspects[:sample_n]
        print(json.dumps({"summary": summary, "sample_file_names": sample}, ensure_ascii=True, indent=2))
        return 0

    if not suspects:
        print(json.dumps({"summary": summary, "deleted_base_rows": 0, "deleted_ann_rows": {}}, ensure_ascii=True, indent=2))
        return 0

    delete_file_names(base_table, suspects, args.batch_size)
    deleted_ann_rows: dict[str, int] = {}
    if args.include_ann:
        existing = set(db.table_names()) if hasattr(db, "table_names") else set(db.list_tables())
        for table_name in ann_table_names(args.table):
            if table_name not in existing:
                continue
            ann_table = db.open_table(table_name)
            delete_file_names(ann_table, suspects, args.batch_size)
            deleted_ann_rows[table_name] = len(suspects)

    print(
        json.dumps(
            {
                "summary": summary,
                "deleted_base_rows": len(suspects),
                "deleted_ann_rows": deleted_ann_rows,
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
