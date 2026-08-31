#!/usr/bin/env python3
"""Relocate an Iris collection using a committed tgbackman media manifest.

The source LanceDB is always opened read-only by convention and never mutated.
An apply builds a complete sibling database under a unique partial directory,
validates it, and only then renames it to the requested output path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import time
import uuid
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import lancedb
import pyarrow as pa

MANIFEST_VERSION = 1
ROOTS_TABLE = "collection_roots"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
VECTOR_INDEX_TYPES = {
    "IvfFlat": "IVF_FLAT",
    "IvfSq": "IVF_SQ",
    "IvfPq": "IVF_PQ",
    "IvfRq": "IVF_RQ",
    "IvfHnswSq": "IVF_HNSW_SQ",
    "IvfHnswPq": "IVF_HNSW_PQ",
}
SCALAR_INDEX_TYPES = {
    "BTree": "BTREE",
    "Bitmap": "BITMAP",
    "LabelList": "LABEL_LIST",
}


class RelocationError(RuntimeError):
    """The manifest or database cannot be relocated safely."""


@dataclass(frozen=True)
class Destination:
    file_name: str
    path: Path
    chat_id: str
    size: int
    sha256: str
    mtime_ns: int


@dataclass
class RelocationPlan:
    manifest_path: Path
    manifest_sha256: str
    source_root: Path
    destination_root: Path
    collection_id: str
    mapping: dict[str, tuple[Destination, ...]]
    target_chat_by_key: dict[str, str]

    @property
    def target_keys(self) -> set[str]:
        return {
            destination.file_name
            for destinations in self.mapping.values()
            for destination in destinations
        }


@dataclass
class Inventory:
    source_table_versions: dict[str, int]
    source_table_rows: dict[str, int]
    collection_rows: int
    mapped_rows: int
    mapped_source_keys: list[str]
    mapped_sources_without_rows: list[str]
    unmapped_rows: list[str]
    collisions: list[str]
    duplicate_file_names: list[str]
    current_collection_root: str | None
    root_matches_manifest: bool


@dataclass
class TableResult:
    source_rows: int
    output_rows: int
    relocated_rows: int = 0
    cloned_rows: int = 0
    dropped_rows: int = 0
    rewritten_reference_rows: int = 0


@dataclass
class RelocationReport:
    schema_version: int
    status: str
    manifest_path: str
    manifest_sha256: str
    source_db: str
    output_db: str | None
    collection_id: str
    source_root: str
    destination_root: str
    verified_manifest_files: int
    collection_rows: int
    mapped_rows: int
    unmapped_rows: int
    drop_unmapped: bool
    tables: dict[str, TableResult] = field(default_factory=dict)
    invalidated_caches: list[str] = field(default_factory=list)
    copied_assets: list[str] = field(default_factory=list)
    completed_unix: int | None = None


@dataclass(frozen=True)
class IndexSpec:
    name: str
    columns: tuple[str, ...]
    index_type: str
    distance_type: str | None


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _has_symlink_component(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current = current / part
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                return True
        except OSError:
            return True
    return False


def _safe_relative(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise RelocationError(f"{label} must be a non-empty relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(
        part in ("", ".", "..") for part in relative.parts
    ):
        raise RelocationError(f"unsafe {label}: {value!r}")
    return relative


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise RelocationError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RelocationError("manifest root must be a JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def _verify_destination(destination: Destination) -> Destination:
    try:
        metadata = destination.path.lstat()
    except OSError as exc:
        raise RelocationError(
            f"missing manifest destination {destination.path}: {exc}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise RelocationError(
            f"manifest destination is not a regular file: {destination.path}"
        )
    if metadata.st_size != destination.size:
        raise RelocationError(
            f"manifest destination size mismatch for {destination.path}: "
            f"expected {destination.size}, found {metadata.st_size}"
        )
    digest = _sha256_file(destination.path)
    if digest != destination.sha256:
        raise RelocationError(
            f"manifest destination SHA-256 mismatch for {destination.path}: "
            f"expected {destination.sha256}, found {digest}"
        )
    return Destination(
        file_name=destination.file_name,
        path=destination.path,
        chat_id=destination.chat_id,
        size=destination.size,
        sha256=destination.sha256,
        mtime_ns=metadata.st_mtime_ns,
    )


def _verify_destinations(
    destinations: Sequence[Destination], workers: int
) -> dict[str, Destination]:
    verified: dict[str, Destination] = {}
    total = len(destinations)
    if total == 0:
        return verified
    print(f"Verifying {total:,} manifest destination file(s) by size and SHA-256...")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_verify_destination, item): item for item in destinations
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            item = future.result()
            verified[item.file_name] = item
            if completed % 500 == 0 or completed == total:
                print(f"  verified {completed:,}/{total:,}")
    return verified


def load_relocation_plan(
    manifest_path: Path,
    collection_id: str,
    *,
    hash_workers: int = 4,
) -> RelocationPlan:
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    payload, manifest_digest = _read_json_object(manifest_path)
    if payload.get("version") != MANIFEST_VERSION:
        raise RelocationError(
            f"unsupported tgbackman manifest version: {payload.get('version')!r}"
        )
    if payload.get("db_committed") is not True:
        raise RelocationError(
            "tgbackman manifest is not committed; finish the media migration before relocating Iris"
        )
    for field_name in ("missing", "unsafe", "mismatched"):
        problems = payload.get(field_name, [])
        if not isinstance(problems, list):
            raise RelocationError(f"manifest field {field_name!r} must be a list")
        if problems:
            raise RelocationError(
                f"manifest contains {len(problems):,} unresolved {field_name} item(s)"
            )

    try:
        source_root = _absolute(Path(payload["source_root"])).resolve(strict=True)
        destination_root = _absolute(Path(payload["destination_root"])).resolve(
            strict=True
        )
    except (KeyError, OSError, TypeError) as exc:
        raise RelocationError(
            f"manifest source/destination root is invalid: {exc}"
        ) from exc
    if not source_root.is_dir() or not destination_root.is_dir():
        raise RelocationError(
            "manifest source_root and destination_root must be mounted directories"
        )
    if source_root == destination_root:
        raise RelocationError("manifest source and destination roots are identical")

    chats = payload.get("chats")
    media = payload.get("media")
    if not isinstance(chats, list) or not isinstance(media, list):
        raise RelocationError("manifest chats and media must be lists")

    chat_destinations: dict[str, Path] = {}
    for row in chats:
        if not isinstance(row, dict):
            raise RelocationError("manifest chat entry must be an object")
        chat_id = str(row.get("chat_id", "")).strip()
        if not chat_id or chat_id in chat_destinations:
            raise RelocationError(f"missing or duplicate manifest chat_id: {chat_id!r}")
        raw_destination = row.get("destination")
        if not isinstance(raw_destination, str):
            raise RelocationError(f"chat {chat_id} has no destination")
        try:
            destination = _absolute(Path(raw_destination)).resolve(strict=True)
        except OSError as exc:
            raise RelocationError(
                f"chat destination is missing for {chat_id}: {exc}"
            ) from exc
        if not destination.is_dir() or not _within(destination, destination_root):
            raise RelocationError(
                f"chat destination is outside destination_root for {chat_id}: {destination}"
            )
        if _has_symlink_component(destination, destination_root):
            raise RelocationError(
                f"chat destination contains a symbolic link: {destination}"
            )
        chat_destinations[chat_id] = destination

    by_old_key: dict[str, dict[str, Destination]] = {}
    target_owners: dict[str, str] = {}
    unverified_by_key: dict[str, Destination] = {}
    source_root_lexical = _absolute(source_root)
    for index, row in enumerate(media, start=1):
        if not isinstance(row, dict):
            raise RelocationError(f"manifest media entry {index} must be an object")
        chat_id = str(row.get("chat_id", "")).strip()
        chat_destination = chat_destinations.get(chat_id)
        if chat_destination is None:
            raise RelocationError(
                f"media entry {index} refers to unknown chat {chat_id!r}"
            )
        if row.get("status") not in ("copied", "reused"):
            raise RelocationError(
                f"media entry {index} is not complete: status={row.get('status')!r}"
            )
        source_value = row.get("source")
        if not isinstance(source_value, str):
            raise RelocationError(f"media entry {index} has no source path")
        source_path = _absolute(Path(source_value))
        try:
            source_relative = source_path.relative_to(source_root_lexical)
        except ValueError as exc:
            raise RelocationError(
                f"media source is outside source_root: {source_path}"
            ) from exc
        relative = _safe_relative(row.get("relative"), f"media[{index}].relative")
        destination_lexical = chat_destination.joinpath(*relative.parts)
        if not _within(destination_lexical, destination_root):
            raise RelocationError(
                f"media destination escapes destination_root: {destination_lexical}"
            )
        if _has_symlink_component(destination_lexical, destination_root):
            raise RelocationError(
                f"media destination contains a symbolic link: {destination_lexical}"
            )
        try:
            size = int(row["size"])
            digest = str(row["sha256"]).lower()
        except (KeyError, TypeError, ValueError) as exc:
            raise RelocationError(f"media entry {index} has invalid size/hash") from exc
        if size < 0 or not SHA256_RE.fullmatch(digest):
            raise RelocationError(f"media entry {index} has invalid size/hash")

        old_key = f"{collection_id}/{source_relative.as_posix()}"
        destination_relative = destination_lexical.relative_to(destination_root)
        new_key = f"{collection_id}/{destination_relative.as_posix()}"
        destination = Destination(
            file_name=new_key,
            path=destination_lexical,
            chat_id=chat_id,
            size=size,
            sha256=digest,
            mtime_ns=0,
        )
        previous_owner = target_owners.get(new_key)
        if previous_owner is not None and previous_owner != old_key:
            raise RelocationError(
                f"manifest destination key has multiple source owners: {new_key}"
            )
        target_owners[new_key] = old_key
        existing = unverified_by_key.get(new_key)
        if existing is not None and (existing.size, existing.sha256) != (size, digest):
            raise RelocationError(
                f"manifest destination has conflicting content: {new_key}"
            )
        unverified_by_key[new_key] = destination
        by_old_key.setdefault(old_key, {})[new_key] = destination

    verified = _verify_destinations(
        list(unverified_by_key.values()), max(1, int(hash_workers))
    )
    mapping = {
        old_key: tuple(verified[key] for key in sorted(destinations))
        for old_key, destinations in sorted(by_old_key.items())
    }
    target_chat_by_key = {
        destination.file_name: destination.chat_id
        for destinations in mapping.values()
        for destination in destinations
    }
    return RelocationPlan(
        manifest_path=manifest_path,
        manifest_sha256=manifest_digest,
        source_root=source_root,
        destination_root=destination_root,
        collection_id=collection_id,
        mapping=mapping,
        target_chat_by_key=target_chat_by_key,
    )


def _table_names(db: Any) -> list[str]:
    if hasattr(db, "list_tables"):
        result = db.list_tables()
        names = result.tables if hasattr(result, "tables") else result
        return sorted(str(name) for name in names)
    return sorted(str(name) for name in db.table_names())


def _iter_rows(
    table: Any, columns: Sequence[str] | None = None, batch_size: int = 2048
) -> Iterator[dict[str, Any]]:
    query = table.search(None)
    if columns is not None:
        query = query.select(list(columns))
    for batch in query.to_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def _load_collection_root(db: Any, collection_id: str) -> tuple[str | None, bool]:
    if ROOTS_TABLE not in _table_names(db):
        return None, False
    table = db.open_table(ROOTS_TABLE)
    found: list[str] = []
    for row in _iter_rows(table, ("collection_id", "root_path")):
        if row.get("collection_id") == collection_id:
            found.append(str(row.get("root_path", "")))
    if len(found) > 1:
        raise RelocationError(
            f"collection_roots contains duplicate rows for {collection_id!r}"
        )
    return (found[0] if found else None), bool(found)


def inspect_database(db_dir: Path, table_name: str, plan: RelocationPlan) -> Inventory:
    db_dir = db_dir.expanduser().resolve(strict=True)
    db = lancedb.connect(str(db_dir))
    names = _table_names(db)
    if table_name not in names:
        raise RelocationError(f"source database has no {table_name!r} table")
    versions = {name: int(db.open_table(name).version) for name in names}
    row_counts = {name: int(db.open_table(name).count_rows()) for name in names}

    table = db.open_table(table_name)
    if (
        "file_name" not in table.schema.names
        or "collection_id" not in table.schema.names
    ):
        raise RelocationError(
            f"{table_name!r} lacks required file_name/collection_id columns"
        )
    all_names: set[str] = set()
    duplicate_names: set[str] = set()
    collection_names: set[str] = set()
    prefix = f"{plan.collection_id}/"
    for row in _iter_rows(table, ("file_name", "collection_id")):
        file_name = row.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            raise RelocationError(f"{table_name!r} contains an empty file_name")
        if file_name in all_names:
            duplicate_names.add(file_name)
        all_names.add(file_name)
        row_collection = row.get("collection_id")
        if row_collection == plan.collection_id:
            if not file_name.startswith(prefix):
                raise RelocationError(
                    f"row in collection {plan.collection_id!r} has unscoped key {file_name!r}"
                )
            collection_names.add(file_name)
        elif file_name.startswith(prefix):
            raise RelocationError(
                f"key {file_name!r} is scoped to {plan.collection_id!r} but has collection_id={row_collection!r}"
            )

    mapped_sources = set(plan.mapping) & collection_names
    target_owners: dict[str, str] = {}
    collisions: set[str] = set()
    for old_key in mapped_sources:
        for destination in plan.mapping[old_key]:
            owner = target_owners.get(destination.file_name)
            if owner is not None and owner != old_key:
                collisions.add(destination.file_name)
            target_owners[destination.file_name] = old_key
            if (
                destination.file_name in all_names
                and destination.file_name not in mapped_sources
            ):
                collisions.add(destination.file_name)

    current_root, has_root = _load_collection_root(db, plan.collection_id)
    root_matches = False
    if has_root and current_root:
        try:
            root_matches = (
                _absolute(Path(current_root)).resolve(strict=True) == plan.source_root
            )
        except OSError:
            root_matches = False
    elif not has_root:
        root_matches = True

    return Inventory(
        source_table_versions=versions,
        source_table_rows=row_counts,
        collection_rows=len(collection_names),
        mapped_rows=len(mapped_sources),
        mapped_source_keys=sorted(mapped_sources),
        mapped_sources_without_rows=sorted(set(plan.mapping) - collection_names),
        unmapped_rows=sorted(collection_names - set(plan.mapping)),
        collisions=sorted(collisions),
        duplicate_file_names=sorted(duplicate_names),
        current_collection_root=current_root,
        root_matches_manifest=root_matches,
    )


def _validate_inventory(inventory: Inventory, *, drop_unmapped: bool) -> None:
    if inventory.duplicate_file_names:
        raise RelocationError(
            f"source media table has {len(inventory.duplicate_file_names):,} duplicate file_name key(s)"
        )
    if inventory.collisions:
        raise RelocationError(
            f"relocation would collide with {len(inventory.collisions):,} existing key(s)"
        )
    if not inventory.root_matches_manifest:
        raise RelocationError(
            "collection_roots does not point at the manifest source_root: "
            f"found {inventory.current_collection_root!r}"
        )
    if inventory.unmapped_rows and not drop_unmapped:
        raise RelocationError(
            f"{len(inventory.unmapped_rows):,} collection row(s) are absent from the manifest; "
            "review the dry-run sample and pass --drop-unmapped only if those legacy rows are redundant"
        )


def _choose_target(plan: RelocationPlan, old_key: str, context_key: str | None) -> str:
    destinations = plan.mapping.get(old_key)
    if not destinations:
        return old_key
    if context_key is not None:
        context_chat = plan.target_chat_by_key.get(context_key)
        if context_chat is not None:
            for destination in destinations:
                if destination.chat_id == context_chat:
                    return destination.file_name
    return destinations[0].file_name


def _rewrite_cross_media(value: Any, plan: RelocationPlan) -> tuple[Any, bool]:
    if not isinstance(value, list):
        return value, False
    rewritten: list[Any] = []
    changed = False
    seen: set[tuple[Any, ...]] = set()
    for entry in value:
        if not isinstance(entry, dict):
            rewritten.append(entry)
            continue
        old_key = entry.get("file_name")
        targets = plan.mapping.get(old_key) if isinstance(old_key, str) else None
        entries = targets or (None,)
        for target in entries:
            clone = dict(entry)
            if target is not None:
                clone["file_name"] = target.file_name
                changed = changed or target.file_name != old_key or len(entries) > 1
            identity = (
                clone.get("file_name"),
                clone.get("is_video"),
                clone.get("similarity_pct"),
            )
            if identity in seen:
                changed = True
                continue
            seen.add(identity)
            rewritten.append(clone)
    return rewritten, changed


def _drop_cross_media(value: Any, dropped_keys: set[str]) -> tuple[Any, bool]:
    if not isinstance(value, list):
        return value, False
    rewritten = [
        entry
        for entry in value
        if not (
            isinstance(entry, dict)
            and isinstance(entry.get("file_name"), str)
            and entry["file_name"] in dropped_keys
        )
    ]
    return rewritten, len(rewritten) != len(value)


def _transform_main_row(
    row: dict[str, Any],
    plan: RelocationPlan,
    unmapped: set[str],
    drop_unmapped: bool,
) -> tuple[list[dict[str, Any]], bool, bool]:
    old_key = row.get("file_name")
    if not isinstance(old_key, str):
        raise RelocationError("media table row has a non-string file_name")
    if old_key in unmapped and drop_unmapped:
        return [], False, True
    destinations = plan.mapping.get(old_key)
    targets: tuple[Destination | None, ...] = destinations or (None,)
    output: list[dict[str, Any]] = []
    reference_changed = False
    for destination in targets:
        new_key = destination.file_name if destination is not None else old_key
        clone = copy.deepcopy(row)
        clone["file_name"] = new_key
        if destination is not None:
            if "source_size" in clone:
                clone["source_size"] = destination.size
            if "source_mtime_ns" in clone:
                clone["source_mtime_ns"] = destination.mtime_ns
        for field_name in ("dedupe_match_file", "sift_match_file"):
            value = clone.get(field_name)
            if drop_unmapped and isinstance(value, str) and value in unmapped:
                clone[field_name] = None
                reference_changed = True
                continue
            if isinstance(value, str) and value in plan.mapping:
                replacement = _choose_target(plan, value, new_key)
                reference_changed = reference_changed or replacement != value
                clone[field_name] = replacement
        if "cross_media_matches" in clone:
            rewritten, dropped = _drop_cross_media(
                clone.get("cross_media_matches"), unmapped if drop_unmapped else set()
            )
            rewritten, changed = _rewrite_cross_media(rewritten, plan)
            clone["cross_media_matches"] = rewritten
            reference_changed = reference_changed or dropped or changed
        output.append(clone)
    return output, reference_changed, False


def _rewrite_path_id(
    identifier: Any, old_key: str, new_key: str, clone_count: int
) -> Any:
    if not isinstance(identifier, str):
        return identifier
    if identifier == old_key:
        return new_key
    prefix = f"{old_key}|"
    if identifier.startswith(prefix):
        return f"{new_key}{identifier[len(old_key) :]}"
    if clone_count > 1:
        raise RelocationError(
            f"cannot safely clone path row with opaque id {identifier!r} for {old_key!r}"
        )
    return identifier


def _transform_path_row(
    row: dict[str, Any],
    plan: RelocationPlan,
    unmapped: set[str],
    drop_unmapped: bool,
) -> tuple[list[dict[str, Any]], bool]:
    old_key = row.get("file_name")
    if not isinstance(old_key, str):
        raise RelocationError("path side-table row has a non-string file_name")
    if old_key in unmapped and drop_unmapped:
        return [], True
    destinations = plan.mapping.get(old_key)
    if not destinations:
        return [row], False
    output: list[dict[str, Any]] = []
    for destination in destinations:
        clone = copy.deepcopy(row)
        clone["file_name"] = destination.file_name
        if "id" in clone:
            clone["id"] = _rewrite_path_id(
                clone.get("id"), old_key, destination.file_name, len(destinations)
            )
        output.append(clone)
    return output, False


def _capture_indices(table: Any) -> list[IndexSpec]:
    captured: list[IndexSpec] = []
    for index in table.list_indices():
        index_type = str(index.index_type)
        distance_type: str | None = None
        if index_type in VECTOR_INDEX_TYPES:
            statistics = table.index_stats(index.name)
            if statistics is None or not statistics.distance_type:
                raise RelocationError(f"cannot inspect vector index {index.name!r}")
            distance_type = str(statistics.distance_type)
        elif index_type not in SCALAR_INDEX_TYPES:
            raise RelocationError(
                f"cannot reproduce unsupported index type {index_type!r} ({index.name!r})"
            )
        captured.append(
            IndexSpec(
                name=str(index.name),
                columns=tuple(str(column) for column in index.columns),
                index_type=index_type,
                distance_type=distance_type,
            )
        )
    return captured


def _rebuild_indices(table: Any, specs: Sequence[IndexSpec]) -> None:
    if table.count_rows() == 0:
        return
    for spec in specs:
        if len(spec.columns) != 1:
            raise RelocationError(f"cannot reproduce multi-column index {spec.name!r}")
        print(f"  rebuilding index {spec.name} ({spec.index_type})")
        if spec.index_type in VECTOR_INDEX_TYPES:
            table.create_index(
                vector_column_name=spec.columns[0],
                metric=spec.distance_type or "cosine",
                index_type=VECTOR_INDEX_TYPES[spec.index_type],
                replace=False,
                name=spec.name,
            )
            table.wait_for_index([spec.name])
        else:
            table.create_scalar_index(
                spec.columns[0],
                replace=False,
                index_type=SCALAR_INDEX_TYPES[spec.index_type],
                name=spec.name,
            )


def _validate_indices(db: Any, expected: dict[str, list[IndexSpec]]) -> None:
    for table_name, specs in expected.items():
        actual = {
            (str(index.name), tuple(str(column) for column in index.columns)): str(
                index.index_type
            )
            for index in db.open_table(table_name).list_indices()
        }
        for spec in specs:
            key = (spec.name, spec.columns)
            if actual.get(key) != spec.index_type:
                raise RelocationError(
                    f"output index validation failed for {table_name}.{spec.name}: "
                    f"expected {spec.index_type}, found {actual.get(key)!r}"
                )


def _write_rows(table: Any, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    if rows:
        table.add(pa.Table.from_pylist(rows, schema=schema))


def _copy_table(
    source_table: Any,
    destination_db: Any,
    table_name: str,
    plan: RelocationPlan,
    main_table_name: str,
    unmapped: set[str],
    drop_unmapped: bool,
    batch_size: int,
) -> tuple[TableResult, list[IndexSpec]]:
    schema = source_table.schema
    source_count = int(source_table.count_rows())
    indices = _capture_indices(source_table)
    destination_table = destination_db.create_table(table_name, data=[], schema=schema)
    result = TableResult(source_rows=source_count, output_rows=0)
    processed = 0
    pending: list[dict[str, Any]] = []
    for batch in source_table.search(None).to_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            old_key = row.get("file_name") if "file_name" in schema.names else None
            if table_name == main_table_name:
                transformed, reference_changed, dropped = _transform_main_row(
                    row, plan, unmapped, drop_unmapped
                )
                if dropped:
                    result.dropped_rows += 1
                elif isinstance(old_key, str) and old_key in plan.mapping:
                    result.relocated_rows += 1
                    result.cloned_rows += max(0, len(transformed) - 1)
                if reference_changed:
                    result.rewritten_reference_rows += 1
            elif "file_name" in schema.names:
                transformed, dropped = _transform_path_row(
                    row, plan, unmapped, drop_unmapped
                )
                if dropped:
                    result.dropped_rows += 1
                elif isinstance(old_key, str) and old_key in plan.mapping:
                    result.relocated_rows += 1
                    result.cloned_rows += max(0, len(transformed) - 1)
            else:
                transformed = [row]
            pending.extend(transformed)
            result.output_rows += len(transformed)
            if len(pending) >= batch_size:
                _write_rows(destination_table, pending, schema)
                pending.clear()
        processed += batch.num_rows
        if processed % 50_000 < batch.num_rows or processed == source_count:
            print(f"  {table_name}: {processed:,}/{source_count:,} source row(s)")
    _write_rows(destination_table, pending, schema)
    if destination_table.count_rows() != result.output_rows:
        raise RelocationError(
            f"row-count mismatch while writing {table_name}: "
            f"expected {result.output_rows}, found {destination_table.count_rows()}"
        )
    return result, indices


def _copy_roots_table(
    source_db: Any,
    destination_db: Any,
    plan: RelocationPlan,
    batch_size: int,
) -> TableResult:
    names = _table_names(source_db)
    rows: list[dict[str, Any]] = []
    source_count = 0
    schema = pa.schema(
        [
            pa.field("collection_id", pa.string(), nullable=False),
            pa.field("root_path", pa.string(), nullable=False),
        ]
    )
    found = False
    if ROOTS_TABLE in names:
        source_table = source_db.open_table(ROOTS_TABLE)
        schema = source_table.schema
        source_count = int(source_table.count_rows())
        for row in _iter_rows(source_table, batch_size=batch_size):
            clone = dict(row)
            if clone.get("collection_id") == plan.collection_id:
                if found:
                    raise RelocationError(
                        f"duplicate collection_roots row for {plan.collection_id!r}"
                    )
                clone["root_path"] = str(plan.destination_root)
                found = True
            rows.append(clone)
    if not found:
        rows.append(
            {
                "collection_id": plan.collection_id,
                "root_path": str(plan.destination_root),
            }
        )
    table = destination_db.create_table(ROOTS_TABLE, data=[], schema=schema)
    _write_rows(table, rows, schema)
    return TableResult(
        source_rows=source_count, output_rows=len(rows), relocated_rows=1
    )


def _copy_sift_assets(
    source_db_dir: Path, partial_db_dir: Path, table_name: str
) -> list[str]:
    copied: list[str] = []
    asset_name = f"{table_name}_sift_bovw"
    source = source_db_dir / asset_name
    if source.is_dir():
        shutil.copytree(source, partial_db_dir / asset_name, copy_function=shutil.copy2)
        copied.append(asset_name)
    return copied


def _check_source_versions(
    source_db_dir: Path, expected_versions: dict[str, int]
) -> None:
    db = lancedb.connect(str(source_db_dir))
    names = _table_names(db)
    if sorted(expected_versions) != names:
        raise RelocationError("source database table set changed during relocation")
    changed = [
        name
        for name, version in expected_versions.items()
        if int(db.open_table(name).version) != version
    ]
    if changed:
        raise RelocationError(
            "source database changed during relocation; discard the partial output and retry: "
            + ", ".join(changed)
        )


def _validate_output(
    output_db_dir: Path,
    table_name: str,
    plan: RelocationPlan,
    report: RelocationReport,
    unmapped: set[str],
    mapped_source_keys: set[str],
    drop_unmapped: bool,
) -> None:
    db = lancedb.connect(str(output_db_dir))
    for name, result in report.tables.items():
        found = int(db.open_table(name).count_rows())
        if found != result.output_rows:
            raise RelocationError(
                f"output validation failed for {name}: expected {result.output_rows}, found {found}"
            )
    root, found_root = _load_collection_root(db, plan.collection_id)
    if not found_root or _absolute(Path(root or "")) != _absolute(
        plan.destination_root
    ):
        raise RelocationError("output collection_roots was not updated")

    main = db.open_table(table_name)
    file_names: set[str] = set()
    duplicate_names: set[str] = set()
    stale_keys = {
        old_key
        for old_key, destinations in plan.mapping.items()
        if old_key not in {destination.file_name for destination in destinations}
    }
    expected_targets = {
        destination.file_name
        for old_key, destinations in plan.mapping.items()
        if old_key in mapped_source_keys
        for destination in destinations
    }
    stale_references: list[str] = []
    for row in _iter_rows(main):
        file_name = row.get("file_name")
        if not isinstance(file_name, str):
            raise RelocationError("output media table contains a non-string file_name")
        if file_name in file_names:
            duplicate_names.add(file_name)
        file_names.add(file_name)
        for field_name in ("dedupe_match_file", "sift_match_file"):
            value = row.get(field_name)
            if value in stale_keys:
                stale_references.append(str(value))
        for entry in row.get("cross_media_matches") or []:
            if isinstance(entry, dict) and entry.get("file_name") in stale_keys:
                stale_references.append(str(entry.get("file_name")))
    if duplicate_names:
        raise RelocationError(
            f"output media table has {len(duplicate_names):,} duplicate file_name key(s)"
        )
    missing_targets = expected_targets - file_names
    if missing_targets:
        raise RelocationError(
            f"output media table is missing {len(missing_targets):,} manifest target key(s)"
        )
    if stale_references:
        raise RelocationError(
            f"output media table retains {len(stale_references):,} stale path reference(s)"
        )
    if drop_unmapped and file_names & unmapped:
        raise RelocationError("output media table retains rows requested for removal")

    for name in _table_names(db):
        if name in (table_name, ROOTS_TABLE):
            continue
        table = db.open_table(name)
        if "file_name" not in table.schema.names:
            continue
        for row in _iter_rows(table, ("file_name",)):
            key = row.get("file_name")
            if key in stale_keys or (drop_unmapped and key in unmapped):
                raise RelocationError(
                    f"output path table {name} retains stale key {key!r}"
                )


def apply_relocation(
    source_db_dir: Path,
    output_db_dir: Path,
    table_name: str,
    plan: RelocationPlan,
    inventory: Inventory,
    *,
    drop_unmapped: bool,
    batch_size: int = 1024,
) -> RelocationReport:
    _validate_inventory(inventory, drop_unmapped=drop_unmapped)
    source_db_dir = source_db_dir.expanduser().resolve(strict=True)
    output_db_dir = _absolute(output_db_dir)
    try:
        output_parent = output_db_dir.parent.resolve(strict=True)
    except OSError as exc:
        raise RelocationError(
            f"output database parent must already exist: {output_db_dir.parent}"
        ) from exc
    output_db_dir = output_parent / output_db_dir.name
    if output_db_dir == source_db_dir or _within(output_db_dir, source_db_dir):
        raise RelocationError(
            "output database must be separate from the source database"
        )
    if output_db_dir.exists():
        raise RelocationError(f"output database path already exists: {output_db_dir}")
    partial = output_db_dir.with_name(
        f".{output_db_dir.name}.partial-{uuid.uuid4().hex[:12]}"
    )
    if partial.exists():
        raise RelocationError(f"partial output path already exists: {partial}")
    partial.mkdir(mode=0o700)

    report = RelocationReport(
        schema_version=1,
        status="building",
        manifest_path=str(plan.manifest_path),
        manifest_sha256=plan.manifest_sha256,
        source_db=str(source_db_dir),
        output_db=str(output_db_dir),
        collection_id=plan.collection_id,
        source_root=str(plan.source_root),
        destination_root=str(plan.destination_root),
        verified_manifest_files=len(plan.target_keys),
        collection_rows=inventory.collection_rows,
        mapped_rows=inventory.mapped_rows,
        unmapped_rows=len(inventory.unmapped_rows),
        drop_unmapped=drop_unmapped,
        invalidated_caches=[
            "cross-media-state.json",
            "cross-media-work/",
            "embedimages-status*.json",
            "*-video/ derived scene-still caches",
        ],
    )
    source_db = lancedb.connect(str(source_db_dir))
    destination_db = lancedb.connect(str(partial))
    index_specs: dict[str, list[IndexSpec]] = {}
    unmapped = set(inventory.unmapped_rows)
    try:
        for name in _table_names(source_db):
            if name == ROOTS_TABLE:
                continue
            print(f"Copying and relocating table {name}...")
            result, specs = _copy_table(
                source_db.open_table(name),
                destination_db,
                name,
                plan,
                table_name,
                unmapped,
                drop_unmapped,
                max(1, int(batch_size)),
            )
            report.tables[name] = result
            index_specs[name] = specs
        report.tables[ROOTS_TABLE] = _copy_roots_table(
            source_db, destination_db, plan, max(1, int(batch_size))
        )
        report.copied_assets = _copy_sift_assets(source_db_dir, partial, table_name)

        for name, specs in index_specs.items():
            if not specs:
                continue
            print(f"Restoring {len(specs)} index(es) on {name}...")
            _rebuild_indices(destination_db.open_table(name), specs)

        _validate_indices(destination_db, index_specs)

        _check_source_versions(source_db_dir, inventory.source_table_versions)
        _validate_output(
            partial,
            table_name,
            plan,
            report,
            unmapped,
            set(inventory.mapped_source_keys),
            drop_unmapped,
        )
        report.status = "complete"
        report.completed_unix = int(time.time())
        (partial / "relocation-report.json").write_text(
            json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _check_source_versions(source_db_dir, inventory.source_table_versions)
        os.replace(partial, output_db_dir)
        return report
    except Exception as exc:
        failure = asdict(report)
        failure["status"] = "failed"
        failure["error"] = f"{type(exc).__name__}: {exc}"
        try:
            (partial / "relocation-report.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        print(f"Partial output retained for diagnosis: {partial}", file=sys.stderr)
        raise


def _inventory_json(inventory: Inventory, sample_size: int) -> dict[str, Any]:
    limit = max(0, sample_size)
    return {
        "collection_rows": inventory.collection_rows,
        "mapped_rows": inventory.mapped_rows,
        "unmapped_rows": len(inventory.unmapped_rows),
        "mapped_sources_without_rows": len(inventory.mapped_sources_without_rows),
        "collisions": len(inventory.collisions),
        "duplicate_file_names": len(inventory.duplicate_file_names),
        "current_collection_root": inventory.current_collection_root,
        "root_matches_manifest": inventory.root_matches_manifest,
        "samples": {
            "unmapped_rows": inventory.unmapped_rows[:limit],
            "mapped_sources_without_rows": inventory.mapped_sources_without_rows[
                :limit
            ],
            "collisions": inventory.collisions[:limit],
            "duplicate_file_names": inventory.duplicate_file_names[:limit],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Relocate Iris LanceDB paths using a committed tgbackman media-reorganisation "
            "manifest. The default is a read-only dry run; --apply always creates a new database."
        )
    )
    parser.add_argument(
        "--db-dir", type=Path, required=True, help="source Iris LanceDB directory"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="committed tgbackman reorganisation manifest",
    )
    parser.add_argument(
        "--collection-id", required=True, help="Iris collection id to relocate"
    )
    parser.add_argument(
        "--table", default="media_index", help="primary Iris media table"
    )
    parser.add_argument(
        "--output-db-dir",
        type=Path,
        help="new database directory; required with --apply",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="build, validate, and publish a new relocated database",
    )
    parser.add_argument(
        "--drop-unmapped",
        action="store_true",
        help="omit collection rows absent from the manifest and their ANN rows in the new database",
    )
    parser.add_argument(
        "--hash-workers",
        type=int,
        default=4,
        help="parallel manifest destination hash workers (default: 4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="LanceDB copy batch size (default: 1024)",
    )
    parser.add_argument(
        "--show-sample",
        type=int,
        default=20,
        help="dry-run sample size for blockers (default: 20)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.collection_id.strip():
        parser.error("--collection-id must not be empty")
    if args.hash_workers < 1 or args.batch_size < 1:
        parser.error("--hash-workers and --batch-size must be positive")
    if args.apply and args.output_db_dir is None:
        parser.error("--apply requires --output-db-dir")
    try:
        plan = load_relocation_plan(
            args.manifest,
            args.collection_id.strip(),
            hash_workers=args.hash_workers,
        )
        inventory = inspect_database(args.db_dir, args.table, plan)
        dry_run = {
            "mode": "apply" if args.apply else "dry-run",
            "manifest": str(plan.manifest_path),
            "manifest_sha256": plan.manifest_sha256,
            "source_root": str(plan.source_root),
            "destination_root": str(plan.destination_root),
            "manifest_source_keys": len(plan.mapping),
            "manifest_destination_keys": len(plan.target_keys),
            "drop_unmapped": bool(args.drop_unmapped),
            "inventory": _inventory_json(inventory, args.show_sample),
        }
        print(json.dumps(dry_run, indent=2, sort_keys=True))
        if not args.apply:
            try:
                _validate_inventory(inventory, drop_unmapped=bool(args.drop_unmapped))
            except RelocationError as exc:
                print(f"Dry run found an apply blocker: {exc}", file=sys.stderr)
                return 2
            print("Dry run complete: Iris and the archive were not changed.")
            return 0
        _validate_inventory(inventory, drop_unmapped=args.drop_unmapped)
        report = apply_relocation(
            args.db_dir,
            args.output_db_dir,
            args.table,
            plan,
            inventory,
            drop_unmapped=args.drop_unmapped,
            batch_size=args.batch_size,
        )
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
        print(f"Relocated Iris database created at: {args.output_db_dir}")
        print(
            "The source Iris database and both Telegram media trees were not modified."
        )
        return 0
    except (OSError, RelocationError, pa.ArrowException) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "Interrupted; source database and archive remain unchanged.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
