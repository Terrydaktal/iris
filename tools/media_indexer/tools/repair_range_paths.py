"""Repair collection keys after inserting date-range folders, without moving media.

Only path metadata is read or updated. Lance versions are retained for rollback;
OCR, vectors, perceptual hashes, and all other payload columns are untouched.
Close Iris and stop indexers before applying. Dry-run is the default.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import re
import stat
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import lancedb
import pyarrow as pa

TEMP_KEY = "__iris_range_repair_key"
PATH_COLUMNS = ("file_name", "id", "dedupe_match_file", "sift_match_file", "cross_media_matches")
RANGE = re.compile(r"\d{4}-\d{2}-\d{2}T[^/]+__\d{4}-\d{2}-\d{2}T[^/]+")


class RepairError(RuntimeError):
    pass


def _has_symlink(path: Path) -> bool:
    absolute = path.absolute()
    return any(parent.is_symlink() for parent in (absolute, *absolute.parents))


def _plain_file(path: Path, root: Path) -> bool:
    try:
        if _has_symlink(root):
            return False
        relative = path.relative_to(root)
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                return False
        return path.is_file()
    except (OSError, ValueError):
        return False


def _candidates(
    root: Path, key: str, collection_id: str, size: int | None, ranges: dict[str, list[Path]] | None = None
) -> list[str]:
    prefix = collection_id + "/"
    if not key.startswith(prefix) or _has_symlink(root):
        return []
    relative = key[len(prefix) :]
    parts = PurePosixPath(relative).parts
    if (
        len(parts) < 3
        or relative.startswith("/")
        or ".." in parts
        or "\\" in relative
        or RANGE.fullmatch(parts[1])
    ):
        return []
    # Do not inspect the deliberately excluded private chat, even as a recovery source.
    if parts[0].casefold() == "p":
        return []
    if _plain_file(root.joinpath(*parts), root):
        return []
    chat = root / parts[0]
    if chat.is_symlink() or not chat.is_dir():
        return []
    cache = ranges if ranges is not None else {}
    if parts[0] not in cache:
        cache[parts[0]] = [
            child
            for child in chat.iterdir()
            if RANGE.fullmatch(child.name) and not child.is_symlink() and child.is_dir()
        ]
    result = []
    for folder in cache[parts[0]]:
        candidate = folder.joinpath(*parts[1:])
        if _plain_file(candidate, root) and (size is None or candidate.stat().st_size == size):
            result.append(prefix + candidate.relative_to(root).as_posix())
    return sorted(result)


def resolve_relocated_key(root: Path, key: str, collection_id: str, size: int | None = None) -> str | None:
    candidates = _candidates(root, key, collection_id, size)
    return candidates[0] if len(candidates) == 1 else None


def rewrite_metadata(row: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    rewritten = copy.deepcopy(row)
    old_key = row.get("file_name")
    new_key = mapping.get(old_key, old_key)
    if new_key != old_key:
        rewritten["file_name"] = new_key
        identifier = row.get("id")
        if isinstance(identifier, str):
            if identifier == old_key:
                rewritten["id"] = new_key
            elif identifier.startswith(old_key + "|"):
                rewritten["id"] = new_key + identifier[len(old_key) :]
    for column in ("dedupe_match_file", "sift_match_file"):
        if isinstance(row.get(column), str):
            rewritten[column] = mapping.get(row[column], row[column])
    for item in rewritten.get("cross_media_matches") or []:
        if isinstance(item, dict) and isinstance(item.get("file_name"), str):
            item["file_name"] = mapping.get(item["file_name"], item["file_name"])
    return rewritten


def _names(db: Any) -> list[str]:
    return sorted(db.list_tables().tables)


def _indices(table: Any) -> list[tuple[str, tuple[str, ...], str]]:
    return sorted((index.name, tuple(index.columns), str(index.index_type)) for index in table.list_indices())


def build_plan(db: Any, collection_id: str) -> dict[str, Any]:
    roots = db.open_table("collection_roots").search().select(["collection_id", "root_path"]).to_list()
    matching = [row["root_path"] for row in roots if row["collection_id"] == collection_id]
    if len(matching) != 1:
        raise RepairError("collection must have exactly one root")
    original_root = Path(matching[0]).expanduser()
    if _has_symlink(original_root):
        raise RepairError("collection root contains a symbolic link")
    root = original_root.resolve(strict=True)
    if not root.is_dir():
        raise RepairError("collection root is not an accessible directory")
    versions, counts, schemas, indices, snapshots = {}, {}, {}, {}, {}
    known_keys: set[str] = set()
    sizes: dict[str, int | None] = {}
    for name in _names(db):
        table = db.open_table(name)
        versions[name], counts[name] = table.version, table.count_rows()
        schemas[name], indices[name] = str(table.schema), _indices(table)
        if TEMP_KEY in table.schema.names:
            raise RepairError("unfinished repair column exists; inspect the previous rollback journal")
        if "file_name" not in table.schema.names:
            continue
        # Immutable relocation/observation ledgers must never be rewritten.
        if name.startswith("relocation_"):
            continue
        columns = [column for column in PATH_COLUMNS if column in table.schema.names]
        if "source_size" in table.schema.names:
            columns.append("source_size")
        rows = table.search().select(columns).to_arrow().to_pylist()
        snapshots[name] = rows
        for row in rows:
            key = row.get("file_name")
            if isinstance(key, str):
                known_keys.add(key)
                if "source_size" in row:
                    sizes[key] = row["source_size"]
    ranges: dict[str, list[Path]] = {}
    mapping: dict[str, str] = {}
    unresolved = ambiguous = 0
    for key in sorted(known_keys):
        prefix = collection_id + "/"
        if not key.startswith(prefix):
            continue
        parts = PurePosixPath(key[len(prefix) :]).parts
        if len(parts) < 3 or RANGE.fullmatch(parts[1]) or parts[0].casefold() == "p":
            continue
        candidates = _candidates(root, key, collection_id, sizes.get(key), ranges)
        if len(candidates) == 1:
            mapping[key] = candidates[0]
        elif len(candidates) > 1:
            ambiguous += 1
        elif not _plain_file(root.joinpath(*parts), root) and ranges.get(parts[0]):
            unresolved += 1
    destination_counts = Counter(mapping.values())
    collisions = {
        key for key, target in mapping.items() if target in known_keys or destination_counts[target] != 1
    }
    for key in collisions:
        del mapping[key]
    tables = {}
    for name, rows in snapshots.items():
        columns = [column for column in PATH_COLUMNS if column in rows[0]] if rows else []
        join = "id" if "id" in columns else "file_name"
        identifiers = [row.get(join) for row in rows]
        if len(set(identifiers)) != len(identifiers) or any(not isinstance(key, str) for key in identifiers):
            raise RepairError(f"{name} lacks a unique non-null repair join key")
        changed = []
        target_ids = []
        for row in rows:
            paths = {column: row[column] for column in columns}
            updated = rewrite_metadata(paths, mapping)
            target_ids.append(updated[join])
            if updated != paths:
                updated[TEMP_KEY] = row[join]
                changed.append(updated)
        if len(set(target_ids)) != len(target_ids):
            raise RepairError(f"{name} repair would create duplicate row identities")
        if changed:
            tables[name] = {"columns": columns, "join": join, "rows": changed}
    signatures = {}
    for new_key in mapping.values():
        path = root / new_key[len(collection_id) + 1 :]
        info = path.stat()
        signatures[new_key] = [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]
    info = root.stat()
    return {
        "collection_id": collection_id,
        "root": str(root),
        "root_identity": [info.st_dev, info.st_ino],
        "mapping": mapping,
        "versions": versions,
        "counts": counts,
        "schemas": schemas,
        "indices": indices,
        "tables": tables,
        "signatures": signatures,
        "unresolved": unresolved,
        "ambiguous": ambiguous,
        "collisions": len(collisions),
    }


def _write_report(path: Path, report: dict[str, Any], *, create: bool = False) -> None:
    if _has_symlink(path.parent) or path.is_symlink():
        raise RepairError("rollback journal path contains a symbolic link")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".range-repair-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if create:
            os.link(temporary, path, follow_symlinks=False)
            os.unlink(temporary)
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _patch_table(table: Any, data: dict[str, Any], progress: Callable[[int], None]) -> None:
    result = table.add_columns({TEMP_KEY: data["join"]})
    progress(result.version)
    schema = pa.schema([table.schema.field(column) for column in [*data["columns"], TEMP_KEY]])
    batch = pa.Table.from_pylist(data["rows"], schema=schema)
    result = table.merge_insert(TEMP_KEY).when_matched_update_all().execute(batch)
    progress(result.version)
    if result.num_updated_rows != len(data["rows"]) or result.num_inserted_rows or result.num_deleted_rows:
        raise RepairError("path patch did not update exactly the planned rows")
    result = table.drop_columns([TEMP_KEY])
    progress(result.version)


def apply_plan(db: Any, plan: dict[str, Any], report_path: Path) -> dict[str, Any]:
    root = Path(plan["root"])
    if [root.stat().st_dev, root.stat().st_ino] != plan["root_identity"]:
        raise RepairError("collection root changed or was unmounted")
    for key, signature in plan["signatures"].items():
        path = root / key[len(plan["collection_id"]) + 1 :]
        if not _plain_file(path, root):
            raise RepairError("a verified repair destination is no longer available")
        info = path.stat()
        if [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns] != signature:
            raise RepairError("a verified repair destination changed")
    report = {key: value for key, value in plan.items() if key not in ("tables", "signatures")}
    report.update(status="applying", touched=[], heads=dict(plan["versions"]))
    if os.path.lexists(report_path):
        raise RepairError("choose a new report path; existing rollback records are never overwritten")
    _write_report(report_path, report, create=True)
    tables = {name: db.open_table(name) for name in plan["versions"]}

    def assert_heads() -> None:
        for name, table in tables.items():
            table.checkout_latest()
            if table.version != report["heads"][name]:
                raise RepairError(f"concurrent database write detected in {name}")

    assert_heads()
    try:
        for name, data in plan["tables"].items():
            assert_heads()
            table = tables[name]
            report["touched"].append(name)
            _write_report(report_path, report)

            def checkpoint(version: int, table_name: str = name, current_table: Any = table) -> None:
                if version != report["heads"][table_name] + 1:
                    raise RepairError("foreign commit occurred during repair; manual recovery required")
                report["heads"][table_name] = version
                _write_report(report_path, report)
                current_table.checkout_latest()
                if current_table.version != version:
                    raise RepairError("concurrent writer advanced the repair table")
                assert_heads()

            print(f"{name}: repairing {len(data['rows']):,} metadata row(s)", flush=True)
            _patch_table(table, data, checkpoint)
            if table.count_rows() != plan["counts"][name] or str(table.schema) != plan["schemas"][name]:
                raise RepairError("row count or schema changed during repair")
            if _indices(table) != plan["indices"][name]:
                raise RepairError("repair changed an index definition")
        assert_heads()
        for name, data in plan["tables"].items():
            columns = data["columns"]
            remaining = tables[name].search().select(columns).to_arrow().to_pylist()
            if any(rewrite_metadata(row, plan["mapping"]) != row for row in remaining):
                raise RepairError("stale path references remain after repair")
        assert_heads()
        report["status"] = "complete"
        _write_report(report_path, report)
        return report
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        failures = []
        for name in reversed(report["touched"]):
            try:
                table = tables[name]
                table.checkout_latest()
                if table.version != report["heads"][name]:
                    raise RepairError(
                        "unexpected head; manual recovery required to avoid discarding another writer"
                    )
                table.restore(plan["versions"][name])
                report["heads"][name] = table.version
                if table.count_rows() != plan["counts"][name] or str(table.schema) != plan["schemas"][name]:
                    raise RepairError("rollback verification failed")
            except BaseException as restore_error:  # noqa: BLE001 -- finish the remaining rollback attempts even on interruption.
                failures.append(f"{name}: {restore_error}")
        report["status"] = "rollback-required" if failures else "rolled-back"
        report["rollback_errors"] = failures
        _write_report(report_path, report)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-dir", type=Path, required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--memory-pool-mib",
        type=int,
        default=1024,
        help="Lance execution-pool budget when LANCE_MEM_POOL_SIZE is unset (default: 1024 MiB)",
    )
    args = parser.parse_args()
    if args.memory_pool_mib < 1:
        parser.error("--memory-pool-mib must be positive")
    os.environ.setdefault("LANCE_MEM_POOL_SIZE", str(args.memory_pool_mib * 1024 * 1024))
    if "LANCE_BYPASS_SPILLING" in os.environ:
        raise RepairError("unset LANCE_BYPASS_SPILLING to retain bounded-memory disk spilling")
    db_dir = args.db_dir.expanduser().resolve(strict=True)
    if not (db_dir / "collection_roots.lance").is_dir():
        raise RepairError("expected an existing Iris LanceDB directory")
    descriptor = os.open(db_dir / ".range-path-repair.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
        os.close(descriptor)
        raise RepairError("repair lock must be a regular file owned by the current user")
    with os.fdopen(descriptor, "a") as lock:
        os.fchmod(lock.fileno(), 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = lancedb.connect(str(db_dir))
        plan = build_plan(db, args.collection_id)
        print(
            json.dumps(
                {
                    "mapped": len(plan["mapping"]),
                    "unresolved": plan["unresolved"],
                    "ambiguous": plan["ambiguous"],
                    "collisions": plan["collisions"],
                    "table_rows": {name: len(data["rows"]) for name, data in plan["tables"].items()},
                },
                indent=2,
            ),
            flush=True,
        )
        if args.apply:
            if args.report is None:
                raise RepairError("--apply requires --report for the rollback journal")
            apply_plan(db, plan, args.report.expanduser().absolute())
            print("Repair complete; media files, embeddings and OCR are unchanged.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
