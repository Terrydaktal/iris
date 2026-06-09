#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa


SIFT_FIELDS = (
    "sift_match_file",
    "sift_match_score",
    "sift_match_inliers",
    "sift_match_good_matches",
    "sift_match_inlier_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clear SIFT links for one connected component.")
    parser.add_argument("--db-dir", required=True, type=Path)
    parser.add_argument("--table", default="media_index")
    parser.add_argument("--file", required=True, help="Any file inside the broken SIFT component.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def escape_sql(value: str) -> str:
    return value.replace("'", "''")


def clear_sift_fields(row: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    for field in SIFT_FIELDS:
        updated[field] = None
    updated["sift_match_checked"] = False
    return updated


def main() -> int:
    args = parse_args()
    db = lancedb.connect(str(args.db_dir))
    table = db.open_table(args.table)
    rows = table.to_arrow().to_pylist()
    rows_by_name = {row["file_name"]: row for row in rows}
    if args.file not in rows_by_name:
        raise SystemExit(f"file not found in DB: {args.file}")

    graph: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        file_name = row.get("file_name")
        match_file = row.get("sift_match_file")
        if not isinstance(file_name, str) or not isinstance(match_file, str):
            continue
        if match_file not in rows_by_name:
            continue
        graph[file_name].add(match_file)
        graph[match_file].add(file_name)

    component: set[str] = set()
    queue = deque([args.file])
    while queue:
        current = queue.popleft()
        if current in component:
            continue
        component.add(current)
        queue.extend(graph.get(current, ()))

    updates = [clear_sift_fields(rows_by_name[name]) for name in sorted(component)]
    if updates and not args.dry_run:
        arrow_table = pa.Table.from_pylist(updates, schema=table.schema)
        if hasattr(table, "merge_insert"):
            (
                table.merge_insert("file_name")
                .use_index(True)
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(arrow_table)
            )
        else:
            for row in updates:
                table.delete(f"file_name = '{escape_sql(row['file_name'])}'")
            table.add(arrow_table)

    print(
        json.dumps(
            {
                "component_size": len(component),
                "updated": 0 if args.dry_run else len(updates),
                "dry_run": bool(args.dry_run),
                "files": sorted(component),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
