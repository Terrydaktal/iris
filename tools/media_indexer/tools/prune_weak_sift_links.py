#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clear existing weak SIFT links from LanceDB.")
    parser.add_argument("--db-dir", required=True, type=Path)
    parser.add_argument("--table", default="media_index")
    parser.add_argument("--min-inliers", type=int, default=80)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.90)
    parser.add_argument("--min-score", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def weak_sift_link(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if not row.get("sift_match_file"):
        return False
    inliers = row.get("sift_match_inliers")
    ratio = row.get("sift_match_inlier_ratio")
    score = row.get("sift_match_score")
    if inliers is None or ratio is None or score is None:
        return True
    return (
        int(inliers) < args.min_inliers
        or float(ratio) < args.min_inlier_ratio
        or float(score) < args.min_score
    )


def clear_sift(row: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    updated["sift_match_file"] = None
    updated["sift_match_score"] = None
    updated["sift_match_inliers"] = None
    updated["sift_match_good_matches"] = None
    updated["sift_match_inlier_ratio"] = None
    updated["sift_match_checked"] = False
    return updated


def write_updates(table, schema: pa.Schema, updates: list[dict[str, Any]], batch_size: int) -> None:
    batch_size = max(1, int(batch_size))
    for start in range(0, len(updates), batch_size):
        batch = updates[start : start + batch_size]
        arrow_table = pa.Table.from_pylist(batch, schema=schema)
        if hasattr(table, "merge_insert"):
            (
                table.merge_insert("file_name")
                .use_index(True)
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(arrow_table)
            )
        else:
            for row in batch:
                table.delete(f"file_name = '{row['file_name'].replace(chr(39), chr(39) + chr(39))}'")
            table.add(arrow_table)


def main() -> int:
    args = parse_args()
    db = lancedb.connect(str(args.db_dir))
    table = db.open_table(args.table)
    rows = table.to_arrow().to_pylist()
    weak_rows = [row for row in rows if weak_sift_link(row, args)]
    updates = [clear_sift(row) for row in weak_rows]
    if updates and not args.dry_run:
        write_updates(table, table.schema, updates, args.batch_size)

    print(
        json.dumps(
            {
                "weak_links": len(weak_rows),
                "updated": 0 if args.dry_run else len(updates),
                "dry_run": bool(args.dry_run),
                "batch_size": args.batch_size,
                "thresholds": {
                    "min_inliers": args.min_inliers,
                    "min_inlier_ratio": args.min_inlier_ratio,
                    "min_score": args.min_score,
                },
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
