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
import sqlite3
import stat
import sys
import tempfile
import time
import uuid
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from itertools import groupby
from pathlib import Path, PurePosixPath
from typing import Any

import lancedb
import pyarrow as pa

MANIFEST_VERSION = 1
ROOTS_TABLE = "collection_roots"
RELOCATION_TABLE = "relocation_paths"
OBSERVATION_PREFIX = "relocation_observations_"
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
    missing: bool = False
    reason: str | None = None
    verified_signature: tuple[int, int, int, int, int] | None = None


@dataclass
class RelocationPlan:
    manifest_path: Path
    manifest_sha256: str
    source_root: Path
    destination_root: Path
    collection_id: str
    mapping: dict[str, tuple[Destination, ...]]
    target_chat_by_key: dict[str, str]
    preserve_missing: bool = False
    excluded_sources: dict[str, str] = field(default_factory=dict)
    preferred_sources: dict[str, tuple[str, ...]] = field(default_factory=dict)
    verified_source_signatures: dict[str, tuple[int, int, int, int, int]] = field(default_factory=dict)
    coalesce_equivalent_sources: bool = False
    media_root_identities: dict[str, tuple[int, int, str, int, int]] = field(default_factory=dict)

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
    excluded_rows: list[str] = field(default_factory=list)
    coalescible_targets: list[str] = field(default_factory=list)
    main_source_keys: list[str] = field(default_factory=list)
    side_only_source_keys: list[str] = field(default_factory=list)


@dataclass
class TableResult:
    source_rows: int
    output_rows: int
    relocated_rows: int = 0
    cloned_rows: int = 0
    dropped_rows: int = 0
    rewritten_reference_rows: int = 0
    coalesced_rows: int = 0
    conflict_rows: int = 0
    variant_rows: int = 0
    added_rows: int = 0
    observation_rows: int = 0


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
    preserved_missing_rows: int = 0
    excluded_rows: int = 0
    source_table_versions: dict[str, int] = field(default_factory=dict)
    coalesce_equivalent_sources: bool = False
    observation_tables: dict[str, str] = field(default_factory=dict)


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


def _safe_absent_destination(path: Path, root: Path) -> None:
    """A retained missing entry must be absent, with no symlink ancestors."""
    if not _within(path, root):
        raise RelocationError("missing destination escapes destination_root")
    current = root
    for component in path.relative_to(root).parts:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise RelocationError("missing destination contains a symbolic link")
        if current == path:
            raise RelocationError(
                "retained missing destination unexpectedly exists; refresh the manifest"
            )
        if not stat.S_ISDIR(mode):
            raise RelocationError("missing destination has a non-directory ancestor")


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
    before_signature = (metadata.st_dev, metadata.st_ino, metadata.st_size,
                        metadata.st_mtime_ns, metadata.st_ctime_ns)
    digest = _sha256_file(destination.path)
    if _source_signature(destination.path) != before_signature:
        raise RelocationError("manifest destination changed during SHA-256 verification")
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
        verified_signature=before_signature,
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
    coalesce_equivalent_sources: bool = False,
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
        if problems and not (
            field_name == "missing" and payload.get("preserve_missing") is True
        ):
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
    if _within(source_root, destination_root) or _within(destination_root, source_root):
        raise RelocationError("manifest source and destination roots must be disjoint")
    root_identities = {str(root): _media_root_identity(root) for root in (source_root, destination_root)}

    chats = payload.get("chats")
    media = payload.get("media")
    if not isinstance(chats, list) or not isinstance(media, list):
        raise RelocationError("manifest chats and media must be lists")
    missing_media = payload.get("missing_media", [])
    if not isinstance(missing_media, list):
        raise RelocationError("manifest missing_media must be a list")
    if missing_media and payload.get("preserve_missing") is not True:
        raise RelocationError(
            "missing_media requires the explicit preserve_missing policy"
        )

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
    unverified_by_key: dict[str, Destination] = {}
    source_root_lexical = _absolute(source_root)
    missing_by_key: dict[str, Destination] = {}
    physical_source_keys: set[str] = set()
    preferred_sources: dict[str, set[str]] = {}
    for index, row in enumerate([*media, *missing_media], start=1):
        if not isinstance(row, dict):
            raise RelocationError(f"manifest media entry {index} must be an object")
        chat_id = str(row.get("chat_id", "")).strip()
        chat_destination = chat_destinations.get(chat_id)
        if chat_destination is None:
            raise RelocationError(
                f"media entry {index} refers to unknown chat {chat_id!r}"
            )
        is_missing = index > len(media)
        allowed_status = ("missing",) if is_missing else ("copied", "reused")
        if row.get("status") not in allowed_status:
            raise RelocationError(
                f"media entry {index} is not complete: status={row.get('status')!r}"
            )
        source_value = row.get("source")
        if not isinstance(source_value, str) or not Path(source_value).is_absolute():
            raise RelocationError(f"media entry {index} has no source path")
        physical_source_path = _absolute(Path(source_value))
        if not _within(physical_source_path, source_root_lexical):
            raise RelocationError("physical media source escapes source_root")
        physical_source_keys.add(
            f"{collection_id}/{physical_source_path.relative_to(source_root_lexical).as_posix()}"
        )
        iris_source = row.get("iris_source") or source_value
        if not isinstance(iris_source, str) or not Path(iris_source).is_absolute():
            raise RelocationError("iris_source must be an absolute original path")
        source_path = _absolute(Path(iris_source))
        if is_missing:
            try:
                source_path.lstat()
            except FileNotFoundError:
                pass
            else:
                raise RelocationError(
                    "retained missing source unexpectedly exists; refresh the manifest"
                )
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
        if is_missing:
            _safe_absent_destination(destination_lexical, destination_root)
        elif _has_symlink_component(destination_lexical, destination_root):
            raise RelocationError(
                f"media destination contains a symbolic link: {destination_lexical}"
            )
        try:
            size = int(row.get("size") or 0) if is_missing else int(row["size"])
            digest = (
                str(row.get("sha256") or "").lower()
                if is_missing
                else str(row["sha256"]).lower()
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RelocationError(f"media entry {index} has invalid size/hash") from exc
        if size < 0 or (not is_missing and not SHA256_RE.fullmatch(digest)):
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
            missing=is_missing,
            reason=str(row.get("reason") or "source file absent")
            if is_missing
            else None,
        )
        existing = unverified_by_key.get(new_key)
        if existing is not None and (existing.size, existing.sha256) != (size, digest):
            raise RelocationError(
                f"manifest destination has conflicting content: {new_key}"
            )
        if is_missing:
            if new_key in unverified_by_key:
                raise RelocationError(
                    "destination is simultaneously present and missing"
                )
            missing_by_key[new_key] = destination
        else:
            if new_key in missing_by_key:
                raise RelocationError(
                    "destination is simultaneously present and missing"
                )
            unverified_by_key[new_key] = destination
        by_old_key.setdefault(old_key, {})[new_key] = destination
        if row.get("primary") is True:
            preferred_sources.setdefault(new_key, set()).add(old_key)
        aliases = row.get("source_aliases", [])
        if not isinstance(aliases, list):
            raise RelocationError("source_aliases must be a list")
        for alias in aliases:
            if not isinstance(alias, str) or not Path(alias).is_absolute():
                raise RelocationError("source alias must be an absolute path")
            alias_path = _absolute(Path(alias))
            try:
                alias_relative = alias_path.relative_to(source_root_lexical)
            except ValueError as exc:
                raise RelocationError("source alias escapes source_root") from exc
            current = source_root_lexical
            for part in alias_relative.parts:
                current /= part
                try:
                    if stat.S_ISLNK(current.lstat().st_mode):
                        raise RelocationError("source alias contains a symbolic link")
                except FileNotFoundError:
                    pass
            alias_key = f"{collection_id}/{alias_relative.as_posix()}"
            by_old_key.setdefault(alias_key, {})[new_key] = destination

    verified = _verify_destinations(
        list(unverified_by_key.values()), max(1, int(hash_workers))
    )
    verified.update(missing_by_key)
    mapping = {
        old_key: tuple(verified[key] for key in sorted(destinations))
        for old_key, destinations in sorted(by_old_key.items())
    }
    target_chat_by_key = {
        destination.file_name: destination.chat_id
        for destinations in mapping.values()
        for destination in destinations
    }
    exclusions = payload.get("excluded_sources", [])
    if not isinstance(exclusions, list):
        raise RelocationError("excluded_sources must be a list")
    excluded_sources: dict[str, str] = {}
    for entry in exclusions:
        if not isinstance(entry, dict) or not isinstance(entry.get("source"), str):
            raise RelocationError(
                "excluded source must identify an absolute original path"
            )
        source = Path(entry["source"])
        if not source.is_absolute():
            raise RelocationError("excluded source must be absolute")
        try:
            relative = _absolute(source).relative_to(source_root)
        except ValueError as exc:
            raise RelocationError("excluded source escapes source_root") from exc
        key = f"{collection_id}/{relative.as_posix()}"
        if key in mapping or key in physical_source_keys:
            raise RelocationError(
                "source is simultaneously preserved and explicitly excluded"
            )
        reason = entry.get("reason")
        if reason not in {"previously purged", "trash/quarantine"}:
            raise RelocationError(
                "excluded source requires an explicit purge or quarantine reason"
            )
        excluded_sources[key] = str(reason)
    plan = RelocationPlan(
        manifest_path=manifest_path,
        manifest_sha256=manifest_digest,
        source_root=source_root,
        destination_root=destination_root,
        collection_id=collection_id,
        mapping=mapping,
        target_chat_by_key=target_chat_by_key,
        preserve_missing=payload.get("preserve_missing") is True,
        excluded_sources=excluded_sources,
        preferred_sources={key: tuple(sorted(values)) for key, values in preferred_sources.items()},
        coalesce_equivalent_sources=coalesce_equivalent_sources,
        media_root_identities=root_identities,
    )
    if coalesce_equivalent_sources:
        _verify_fanin_sources(plan)
    _check_media_bindings(plan)
    return plan


def _source_signature(path: Path) -> tuple[int, int, int, int, int]:
    item = path.lstat()
    return item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns


def _media_root_identity(path: Path) -> tuple[int, int, str, int, int]:
    if _has_symlink_component(path, Path(path.anchor)):
        raise RelocationError("media root contains a symbolic link or is inaccessible")
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise RelocationError("media root is not a directory")
    mount = next((candidate for candidate in (path, *path.parents) if os.path.ismount(candidate)), None)
    if mount is None:
        raise RelocationError("media root has no mounted ancestor")
    mount_metadata = mount.lstat()
    return metadata.st_dev, metadata.st_ino, str(mount), mount_metadata.st_dev, mount_metadata.st_ino


def _check_media_bindings(plan: RelocationPlan) -> None:
    """Rebind verified hashes to unchanged files and mounted roots before publish."""
    try:
        if set(plan.media_root_identities) != {str(plan.source_root), str(plan.destination_root)}:
            raise RelocationError("media root identities were not captured during verification")
        for value, expected in plan.media_root_identities.items():
            if _media_root_identity(Path(value)) != expected:
                raise RelocationError("verified media root identity or mount changed during relocation")
        destinations = {target.file_name: target for values in plan.mapping.values() for target in values}
        for target in destinations.values():
            if target.missing:
                _safe_absent_destination(target.path, plan.destination_root)
                continue
            if target.verified_signature is None:
                raise RelocationError("manifest destination lacks a bound SHA-256 identity")
            if _has_symlink_component(target.path, plan.destination_root) or _source_signature(target.path) != target.verified_signature:
                raise RelocationError("verified manifest destination changed during relocation")
    except OSError as exc:
        raise RelocationError("verified media root or destination became inaccessible during relocation") from exc


def _verify_fanin_sources(plan: RelocationPlan) -> None:
    """An alias is accepted only after actual old-file bytes prove equivalence."""
    owners: dict[str, list[tuple[str, Destination]]] = {}
    for old_key, targets in plan.mapping.items():
        for target in targets:
            owners.setdefault(target.file_name, []).append((old_key, target))
    expected: dict[str, tuple[int, str]] = {}
    for group in owners.values():
        if len(group) < 2:
            continue
        for old_key, target in group:
            if target.missing or not SHA256_RE.fullmatch(target.sha256):
                raise RelocationError("fan-in requires verified physical media and SHA-256 evidence")
            evidence = (target.size, target.sha256)
            if old_key in expected and expected[old_key] != evidence:
                raise RelocationError("old source has contradictory fan-in content evidence")
            expected[old_key] = evidence
    print(f"Verifying {len(expected):,} physical fan-in source file(s) by size and SHA-256...")
    for old_key, (size, digest) in sorted(expected.items()):
        path = plan.source_root / _safe_relative(old_key[len(plan.collection_id) + 1 :], "old source")
        try:
            before = path.lstat()
            if _has_symlink_component(path, plan.source_root):
                raise RelocationError("fan-in source contains a symbolic link")
            if not stat.S_ISREG(before.st_mode) or before.st_size != size:
                raise RelocationError("fan-in old source size/type does not prove equivalence")
            before_signature = _source_signature(path)
            actual = _sha256_file(path)
            if actual != digest or _source_signature(path) != before_signature:
                raise RelocationError("fan-in old source SHA-256 equivalence failed or source changed")
        except OSError as exc:
            raise RelocationError("fan-in source is absent or inaccessible; preserve separate observations") from exc
        plan.verified_source_signatures[old_key] = before_signature


def _check_fanin_sources(plan: RelocationPlan) -> None:
    for old_key, expected in plan.verified_source_signatures.items():
        path = plan.source_root / old_key[len(plan.collection_id) + 1 :]
        if _has_symlink_component(path, plan.source_root) or _source_signature(path) != expected:
            raise RelocationError("verified fan-in source changed during relocation")


def _table_names(db: Any) -> list[str]:
    if hasattr(db, "list_tables"):
        result = db.list_tables()
        names = result.tables if hasattr(result, "tables") else result
        return sorted(str(name) for name in names)
    return sorted(str(name) for name in db.table_names())


def _iter_rows(
    table: Any, columns: Sequence[str] | None = None, batch_size: int = 2048,
    *, where: str | None = None,
) -> Iterator[dict[str, Any]]:
    query = table.search(None)
    if where is not None:
        query = query.where(where)
    if columns is not None:
        query = query.select(list(columns))
    for batch in query.to_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def _active_payload_query(table: Any, excluded_keys: set[str]) -> tuple[Any, int]:
    """Filter exact dropped paths using metadata before decoding any payloads.

    Null/malformed paths stay in the payload query so normal validation still
    rejects them. Exclusions never apply to immutable historical evidence.
    """
    query = table.search(None)
    if not excluded_keys:
        return query, 0
    excluded_count = sum(
        row.get("file_name") in excluded_keys
        for row in _iter_rows(table, ("file_name",))
    )
    literals = ",".join("'" + key.replace("'", "''") + "'" for key in sorted(excluded_keys))
    return query.where(f"file_name IS NULL OR file_name NOT IN ({literals})"), excluded_count


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


def _collection_path_keys(db: Any, main_name: str, collection_id: str) -> tuple[set[str], set[str], set[str], set[str]]:
    """Metadata-only union of active paths; historical evidence is not live media."""
    prefix = f"{collection_id}/"
    main_keys: set[str] = set()
    active_keys: set[str] = set()
    all_keys: set[str] = set()
    main_seen: set[str] = set()
    duplicate_main: set[str] = set()
    for name in _table_names(db):
        if name in (ROOTS_TABLE, RELOCATION_TABLE) or name.startswith(OBSERVATION_PREFIX):
            continue
        table = db.open_table(name)
        if "file_name" not in table.schema.names:
            continue
        has_collection = "collection_id" in table.schema.names
        if name == main_name and not has_collection:
            raise RelocationError("main media table lacks a collection_id column")
        columns = ("file_name", "collection_id") if has_collection else ("file_name",)
        for row in _iter_rows(table, columns):
            key = row.get("file_name")
            row_collection = row.get("collection_id")
            if not isinstance(key, str) or not key:
                if name == main_name or row_collection == collection_id:
                    raise RelocationError("active collection path table contains an empty file_name")
                continue
            all_keys.add(key)
            if name == main_name:
                if key in main_seen:
                    duplicate_main.add(key)
                main_seen.add(key)
            scoped = key.startswith(prefix)
            if has_collection and ((row_collection == collection_id) != scoped):
                raise RelocationError("active collection path has contradictory file_name/collection_id scope")
            if not scoped:
                continue
            _safe_relative(key[len(prefix):], "active collection file_name")
            active_keys.add(key)
            if name == main_name:
                main_keys.add(key)
    return main_keys, active_keys, all_keys, duplicate_main


def export_collection_inventory(
    db_dir: Path, table_name: str, collection_id: str, output: Path
) -> dict[str, Any]:
    """Export paths and stat metadata only, never OCR, vectors, or media contents.

    This is the input for tgbackman's preservation-first media planner. The
    source table versions are checked before publishing the private inventory.
    """
    db_dir = db_dir.expanduser().resolve(strict=True)
    db = lancedb.connect(str(db_dir))
    names = _table_names(db)
    if table_name not in names:
        raise RelocationError(f"source database has no {table_name!r} table")
    versions = {name: int(db.open_table(name).version) for name in names}
    root_value, found = _load_collection_root(db, collection_id)
    if not found or not root_value:
        raise RelocationError("inventory requires a registered collection root")
    source_root = _absolute(Path(root_value)).resolve(strict=True)
    if not source_root.is_dir():
        raise RelocationError("registered collection root is not a mounted directory")
    root_identity = _media_root_identity(source_root)
    rows: list[dict[str, Any]] = []
    prefix = f"{collection_id}/"
    main_keys, active_keys, _, duplicate_main = _collection_path_keys(db, table_name, collection_id)
    if duplicate_main:
        raise RelocationError("collection contains duplicate file_name keys")
    for key in sorted(active_keys):
        relative = _safe_relative(key[len(prefix) :], "inventory file_name")
        source = source_root.joinpath(*relative.parts)
        current = source_root
        for part in relative.parts:
            current /= part
            try:
                if stat.S_ISLNK(current.lstat().st_mode):
                    raise RelocationError("inventory source contains a symbolic link")
            except FileNotFoundError:
                pass
        try:
            metadata = source.lstat()
        except FileNotFoundError:
            metadata = None
        if metadata is not None and not stat.S_ISREG(metadata.st_mode):
            raise RelocationError("inventory source is not a regular file")
        rows.append(
            {
                "file_name": key,
                "source": str(source),
                "exists": metadata is not None,
                "size": metadata.st_size if metadata else None,
                "mtime_ns": metadata.st_mtime_ns if metadata else None,
            }
        )
    _check_source_versions(db_dir, versions)
    if _media_root_identity(source_root) != root_identity:
        raise RelocationError("collection root identity or mount changed during inventory")
    payload = {
        "version": 1,
        "source_db": str(db_dir),
        "table": table_name,
        "collection_id": collection_id,
        "source_root": str(source_root),
        "source_table_versions": versions,
        "active_path_scope": "main-and-active-path-tables",
        "main_keys": len(main_keys),
        "side_only_keys": len(active_keys-main_keys),
        "files": rows,
    }
    output = _absolute(output)
    output = output.parent.resolve(strict=True) / output.name
    if _within(output, db_dir):
        raise RelocationError("inventory output must not be inside the source database")
    descriptor = os.open(
        output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "files": len(rows),
        "existing": sum(row["exists"] for row in rows),
        "missing": sum(not row["exists"] for row in rows),
        "side_only_keys": len(active_keys-main_keys),
        "output": str(output),
    }


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
    main_names, collection_names, all_names, duplicate_names = _collection_path_keys(db, table_name, plan.collection_id)

    mapped_sources = set(plan.mapping) & collection_names
    target_owners: dict[str, str] = {}
    collisions: set[str] = set()
    coalescible: set[str] = set()
    for old_key in mapped_sources:
        for destination in plan.mapping[old_key]:
            owner = target_owners.get(destination.file_name)
            if owner is not None and owner != old_key:
                if plan.coalesce_equivalent_sources and owner in plan.verified_source_signatures and old_key in plan.verified_source_signatures:
                    coalescible.add(destination.file_name)
                else:
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
        unmapped_rows=sorted(
            collection_names - set(plan.mapping) - set(plan.excluded_sources)
        ),
        collisions=sorted(collisions),
        duplicate_file_names=sorted(duplicate_names),
        current_collection_root=current_root,
        root_matches_manifest=root_matches,
        excluded_rows=sorted(collection_names & set(plan.excluded_sources)),
        coalescible_targets=sorted(coalescible),
        main_source_keys=sorted(main_names),
        side_only_source_keys=sorted(collection_names-main_names),
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


def _drop_cross_media(
    value: Any, dropped_keys: set[str] | dict[str, str]
) -> tuple[Any, bool]:
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
    if old_key in plan.excluded_sources or (old_key in unmapped and drop_unmapped):
        return [], False, True
    destinations = plan.mapping.get(old_key)
    targets: tuple[Destination | None, ...] = destinations or (None,)
    output: list[dict[str, Any]] = []
    reference_changed = False
    for destination in targets:
        new_key = destination.file_name if destination is not None else old_key
        clone = copy.deepcopy(row)
        clone["file_name"] = new_key
        if destination is not None and not destination.missing:
            if "source_size" in clone:
                clone["source_size"] = destination.size
            if "source_mtime_ns" in clone:
                clone["source_mtime_ns"] = destination.mtime_ns
        for field_name in ("dedupe_match_file", "sift_match_file"):
            value = clone.get(field_name)
            if isinstance(value, str) and (
                value in plan.excluded_sources or (drop_unmapped and value in unmapped)
            ):
                clone[field_name] = None
                reference_changed = True
                continue
            if isinstance(value, str) and value in plan.mapping:
                replacement = _choose_target(plan, value, new_key)
                reference_changed = reference_changed or replacement != value
                clone[field_name] = replacement
        if "cross_media_matches" in clone:
            rewritten, dropped = _drop_cross_media(
                clone.get("cross_media_matches"),
                unmapped if drop_unmapped else plan.excluded_sources,
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
    if old_key in plan.excluded_sources or (old_key in unmapped and drop_unmapped):
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
    historical = table_name == RELOCATION_TABLE or table_name.startswith(OBSERVATION_PREFIX)
    active = "file_name" in schema.names and not historical
    dropped_keys = set(plan.excluded_sources) | (unmapped if drop_unmapped else set())
    query, prefiltered_count = (
        _active_payload_query(source_table, dropped_keys)
        if active else (source_table.search(None), 0)
    )
    result.dropped_rows = prefiltered_count
    processed += prefiltered_count
    for batch in query.to_batches(batch_size=batch_size):
        if historical:
            # Old ledgers/typed observations are immutable evidence. Transfer
            # their Arrow buffers without inspecting or decoding old payloads.
            destination_table.add(pa.Table.from_batches([batch], schema=schema))
            result.output_rows += batch.num_rows
            processed += batch.num_rows
            continue
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


def _row_bytes(row: dict[str, Any], schema: pa.Schema) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, schema) as writer:
        writer.write_table(pa.Table.from_pylist([row], schema=schema))
    return sink.getvalue().to_pybytes()


def _row_from_bytes(payload: bytes) -> dict[str, Any]:
    return pa.ipc.open_stream(payload).read_all().to_pylist()[0]


def _merge_canonical_main(
    records: list[tuple[str, bytes, bytes]], schema: pa.Schema, plan: RelocationPlan
) -> tuple[dict[str, Any], bool]:
    if all(normalized == records[0][2] for _, _, normalized in records):
        return _row_from_bytes(records[0][2]), False
    rows = [(old, _row_from_bytes(normalized)) for old, _, normalized in records]
    target = rows[0][1]["file_name"]
    preferred = set(plan.preferred_sources.get(target, ()))
    explicit = [row for old, row in rows if old in preferred]
    conflicts: set[str] = set()
    values: dict[str, dict[bytes, Any]] = {}
    for field_spec in schema:
        unique = {}
        for _, row in rows:
            value = row.get(field_spec.name)
            if value is not None:
                unique[_row_bytes({field_spec.name: value}, pa.schema([field_spec]))] = value
        values[field_spec.name] = unique
        if len(unique) > 1:
            conflicts.add(field_spec.name)
    if conflicts:
        if not explicit or len({_row_bytes(row, schema) for row in explicit}) != 1:
            raise RelocationError("conflicting canonical metadata requires one explicit manifest-primary source")
        canonical = copy.deepcopy(explicit[0])
    else:
        canonical = copy.deepcopy(explicit[0] if explicit else rows[0][1])
    # Related scores/state are observations, not interchangeable independent fields.
    families = (
        ("dedupe_match_file", "dedupe_similarity_pct"),
        ("sift_match_file", "sift_match_score", "sift_match_inliers", "sift_match_good_matches", "sift_match_inlier_ratio", "sift_match_checked"),
        ("skip_processing", "processing_error_stage", "processing_error"),
        ("image_width", "image_height"),
    )
    family_fields = {name for family in families for name in family}
    for name, unique in values.items():
        if name not in family_fields and canonical.get(name) is None and len(unique) == 1:
            canonical[name] = next(iter(unique.values()))
    for family in families:
        present = [name for name in family if name in values]
        if present and all(canonical.get(name) is None for name in present) and not conflicts.intersection(present):
            donor = next((row for _, row in rows if any(row.get(name) is not None for name in present)), None)
            if donor is not None:
                for name in present:
                    canonical[name] = copy.deepcopy(donor.get(name))
    return canonical, bool(conflicts)


def _copy_coalesced_table(
    source_table: Any, destination_db: Any, table_name: str, plan: RelocationPlan,
    main_table_name: str, unmapped: set[str], drop_unmapped: bool,
    fanin_targets: set[str], spool_directory: Path, batch_size: int,
) -> tuple[TableResult, list[IndexSpec], str, TableResult]:
    """Stream ordinary rows; externally group fan-in rows, preserving typed originals."""
    schema = source_table.schema
    specs = _capture_indices(source_table)
    source_version = int(source_table.version)
    output = destination_db.create_table(table_name, data=[], schema=schema)
    result = TableResult(source_rows=int(source_table.count_rows()), output_rows=0)
    dropped_keys = set(plan.excluded_sources) | (unmapped if drop_unmapped else set())
    payload_query, prefiltered_count = _active_payload_query(source_table, dropped_keys)
    result.dropped_rows = prefiltered_count
    observation_name = OBSERVATION_PREFIX + table_name
    observation_schema = pa.schema([
        pa.field("generation_id", pa.string()), pa.field("source_table_version", pa.int64()),
        pa.field("source_file_name", pa.string()), pa.field("source_id", pa.string()),
        pa.field("canonical_file_name", pa.string()), pa.field("canonical_id", pa.string()),
        pa.field("role", pa.string()), pa.field("original_row_sha256", pa.string()),
        pa.field("original_row", pa.struct(list(schema))),
    ])
    if observation_name in _table_names(destination_db):
        observations = destination_db.open_table(observation_name)
        if observations.schema != observation_schema:
            raise RelocationError("historical observation schema is incompatible; no output will be published")
    else:
        observations = destination_db.create_table(observation_name, data=[], schema=observation_schema)
    observation_result = TableResult(source_rows=int(observations.count_rows()), output_rows=int(observations.count_rows()))
    pending: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []

    def emit(row: dict[str, Any]) -> None:
        pending.append(row)
        result.output_rows += 1
        if len(pending) >= batch_size:
            _write_rows(output, pending, schema)
            pending.clear()

    def record(original: dict[str, Any], normalized: dict[str, Any], role: str) -> None:
        raw = _row_bytes(original, schema)
        evidence.append({
            "generation_id": plan.manifest_sha256,
            "source_table_version": source_version,
            "source_file_name": original.get("file_name"),
            "source_id": str(original["id"]) if original.get("id") is not None else None,
            "canonical_file_name": normalized.get("file_name"),
            "canonical_id": str(normalized["id"]) if normalized.get("id") is not None else None,
            "role": role, "original_row_sha256": hashlib.sha256(raw).hexdigest(),
            "original_row": original,
        })
        result.observation_rows += 1
        observation_result.added_rows += 1
        observation_result.output_rows += 1
        if len(evidence) >= batch_size:
            _write_rows(observations, evidence, observation_schema)
            evidence.clear()

    with tempfile.TemporaryDirectory(prefix=".coalesce-", dir=spool_directory) as temporary:
        spool_path = Path(temporary) / "rows.sqlite"
        spool = sqlite3.connect(spool_path)
        try:
            spool.execute("PRAGMA journal_mode=OFF")
            spool.execute("CREATE TABLE rows (target TEXT, identity TEXT, old_key TEXT, ordinal INTEGER, original BLOB, normalized BLOB)")
            originals = (
                original
                for batch in payload_query.to_batches(batch_size=batch_size)
                for original in batch.to_pylist()
            )
            for ordinal, original in enumerate(originals):
                old_key = original["file_name"]
                if table_name == main_table_name:
                    transformed, changed, dropped = _transform_main_row(original, plan, unmapped, drop_unmapped)
                    result.rewritten_reference_rows += int(changed)
                else:
                    transformed, dropped = _transform_path_row(original, plan, unmapped, drop_unmapped)
                if dropped:
                    result.dropped_rows += 1
                    continue
                if old_key in plan.mapping:
                    result.relocated_rows += 1
                    result.cloned_rows += max(0, len(transformed) - 1)
                for normalized in transformed:
                    target = normalized["file_name"]
                    if table_name == main_table_name and old_key in plan.mapping:
                        # Canonicalizing duplicate paths can turn former relationships
                        # into self-edges. Preserve their originals in the observations,
                        # not as misleading current matches against the same image.
                        for match_field, score_fields in (
                            ("dedupe_match_file", ("dedupe_similarity_pct",)),
                            ("sift_match_file", ("sift_match_score", "sift_match_inliers",
                                                 "sift_match_good_matches", "sift_match_inlier_ratio")),
                        ):
                            if normalized.get(match_field) == target:
                                normalized[match_field] = None
                                for score_field in score_fields:
                                    if score_field in normalized:
                                        normalized[score_field] = None
                        matches = normalized.get("cross_media_matches")
                        if isinstance(matches, list):
                            normalized["cross_media_matches"] = [
                                entry for entry in matches
                                if not isinstance(entry, dict) or entry.get("file_name") != target
                            ]
                    if target not in fanin_targets:
                        emit(normalized)
                        if old_key in plan.mapping:
                            record(original, normalized, "relocated")
                        continue
                    if "id" in schema.names:
                        identifier = original.get("id")
                        if not isinstance(identifier, str) or not (identifier == old_key or identifier.startswith(old_key + "|")):
                            raise RelocationError("fan-in cannot safely rewrite an opaque ANN row identity")
                    identity = str(normalized.get("id", target))
                    spool.execute("INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?)", (
                        target, identity, old_key, ordinal,
                        _row_bytes(original, schema), _row_bytes(normalized, schema),
                    ))
                if ordinal and ordinal % 50_000 == 0:
                    print(f"  {table_name}: {ordinal:,}/{result.source_rows:,} source row(s)")
            spool.commit()
            spool.execute("CREATE INDEX grouped_rows ON rows(target, identity, old_key, ordinal)")
            cursor = spool.execute("SELECT target, identity, old_key, original, normalized FROM rows ORDER BY target, identity, old_key, ordinal")
            for _, group in groupby(cursor, key=lambda row: (row[0], row[1])):
                records = [(row[2], bytes(row[3]), bytes(row[4])) for row in group]
                if table_name == main_table_name:
                    canonical, conflict = _merge_canonical_main(records, schema, plan)
                    emit(canonical)
                    result.coalesced_rows += len(records) - 1
                    result.conflict_rows += len(records) - 1 if conflict else 0
                    for _, raw, normalized in records:
                        record(_row_from_bytes(raw), canonical, "conflict-preserved" if conflict else "coalesced")
                else:
                    variants: dict[bytes, list[bytes]] = {}
                    for _, raw, normalized in records:
                        variants.setdefault(normalized, []).append(raw)
                    result.coalesced_rows += len(records) - len(variants)
                    result.variant_rows += len(variants) if len(variants) > 1 else 0
                    result.conflict_rows += len(records) if len(variants) > 1 else 0
                    if len(variants) > 1 and "id" not in schema.names:
                        raise RelocationError("conflicting path-table metadata lacks a stable variant identity")
                    identifiers: set[str] = set()
                    for normalized, originals in sorted(variants.items()):
                        canonical = _row_from_bytes(normalized)
                        if len(variants) > 1:
                            canonical["id"] += "|variant:" + hashlib.sha256(normalized).hexdigest()
                        if canonical.get("id") in identifiers:
                            raise RelocationError("ANN variant identity collision")
                        identifiers.add(canonical.get("id"))
                        emit(canonical)
                        for raw in originals:
                            record(_row_from_bytes(raw), canonical, "variant" if len(variants) > 1 else "coalesced")
        finally:
            spool.close()
    _write_rows(output, pending, schema)
    _write_rows(observations, evidence, observation_schema)
    if output.count_rows() != result.output_rows or observations.count_rows() != observation_result.output_rows:
        raise RelocationError("canonical or observation row-count conservation failed")
    if result.output_rows != result.source_rows - result.dropped_rows + result.cloned_rows - result.coalesced_rows:
        raise RelocationError("canonical fan-in conservation failed")
    return result, specs, observation_name, observation_result


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


def _write_relocation_paths(
    destination_db: Any, plan: RelocationPlan, batch_size: int
) -> TableResult:
    """Retain auditable original paths, including intentionally absent files."""
    schema = pa.schema(
        [
            pa.field("collection_id", pa.string()),
            pa.field("file_name", pa.string()),
            pa.field("old_file_name", pa.string()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("source_root", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("destination_root", pa.string()),
            pa.field("manifest_sha256", pa.string()),
            pa.field("size", pa.int64()),
            pa.field("sha256", pa.string()),
        ]
    )
    names = _table_names(destination_db)
    if RELOCATION_TABLE in names:
        table = destination_db.open_table(RELOCATION_TABLE)
        if table.schema != schema:
            raise RelocationError(
                "existing relocation_paths has an incompatible schema"
            )
        source_count = int(table.count_rows())
    else:
        table = destination_db.create_table(RELOCATION_TABLE, data=[], schema=schema)
        source_count = 0
    pending: list[dict[str, Any]] = []
    added = 0
    for old_key, destinations in plan.mapping.items():
        source_relative = old_key[len(plan.collection_id) + 1 :]
        for target in destinations:
            pending.append(
                {
                    "collection_id": plan.collection_id,
                    "file_name": target.file_name,
                    "old_file_name": old_key,
                    "status": "missing" if target.missing else "present",
                    "reason": target.reason,
                    "source_root": str(plan.source_root),
                    "source_path": str(plan.source_root / source_relative),
                    "destination_root": str(plan.destination_root),
                    "manifest_sha256": plan.manifest_sha256,
                    "size": None if target.missing else target.size,
                    "sha256": None if target.missing else target.sha256,
                }
            )
            added += 1
            if len(pending) >= batch_size:
                _write_rows(table, pending, schema)
                pending.clear()
    for old_key, reason in plan.excluded_sources.items():
        pending.append(
            {
                "collection_id": plan.collection_id,
                "file_name": None,
                "old_file_name": old_key,
                "status": "excluded",
                "reason": reason,
                "source_root": str(plan.source_root),
                "source_path": str(
                    plan.source_root / old_key[len(plan.collection_id) + 1 :]
                ),
                "destination_root": str(plan.destination_root),
                "manifest_sha256": plan.manifest_sha256,
                "size": None,
                "sha256": None,
            }
        )
        added += 1
        if len(pending) >= batch_size:
            _write_rows(table, pending, schema)
            pending.clear()
    _write_rows(table, pending, schema)
    return TableResult(
        source_rows=source_count, output_rows=source_count + added, relocated_rows=added, added_rows=added
    )


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
    main_source_keys: set[str] | None = None,
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
    expected_main_targets = {
        target.file_name for key in (main_source_keys if main_source_keys is not None else mapped_source_keys)
        for target in plan.mapping.get(key, ())
    }
    scoped_prefix = plan.collection_id + "/"
    all_target_keys = plan.target_keys
    stale_references: list[str] = []
    for row in _iter_rows(main):
        file_name = row.get("file_name")
        if not isinstance(file_name, str):
            raise RelocationError("output media table contains a non-string file_name")
        if file_name in file_names:
            duplicate_names.add(file_name)
        file_names.add(file_name)
        if file_name.startswith(scoped_prefix) and file_name not in all_target_keys:
            raise RelocationError("output main table retains an uncovered scoped path")
        for field_name in ("dedupe_match_file", "sift_match_file"):
            value = row.get(field_name)
            if value in stale_keys or value in plan.excluded_sources:
                stale_references.append(str(value))
        for entry in row.get("cross_media_matches") or []:
            if isinstance(entry, dict) and (
                entry.get("file_name") in stale_keys
                or entry.get("file_name") in plan.excluded_sources
            ):
                stale_references.append(str(entry.get("file_name")))
    if duplicate_names:
        raise RelocationError(
            f"output media table has {len(duplicate_names):,} duplicate file_name key(s)"
        )
    missing_targets = expected_main_targets - file_names
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
    if file_names & set(plan.excluded_sources):
        raise RelocationError(
            "output media table retains explicitly purged or quarantined rows"
        )

    for name in _table_names(db):
        if name in (table_name, ROOTS_TABLE):
            continue
        table = db.open_table(name)
        if "file_name" not in table.schema.names:
            continue
        columns = ("file_name", "manifest_sha256") if name == RELOCATION_TABLE else ("file_name",)
        for row in _iter_rows(table, columns):
            if name == RELOCATION_TABLE and row.get("manifest_sha256") != plan.manifest_sha256:
                continue
            key = row.get("file_name")
            if isinstance(key, str) and key.startswith(scoped_prefix) and key not in all_target_keys:
                raise RelocationError("output active path table retains an uncovered scoped path")
            if (
                key in stale_keys
                or key in plan.excluded_sources
                or (drop_unmapped and key in unmapped)
            ):
                raise RelocationError(
                    f"output path table {name} retains stale key {key!r}"
                )
    for source_name, observation_name in report.observation_tables.items():
        source_result = report.tables[source_name]
        observations = db.open_table(observation_name)
        original_schema = db.open_table(source_name).schema
        active = db.open_table(source_name)
        binding_columns = ("file_name", "id") if "id" in active.schema.names else ("file_name",)
        valid_bindings = set()
        seen_ids = set()
        for active_row in _iter_rows(active, binding_columns):
            valid_bindings.add((active_row["file_name"], active_row.get("id")))
            if "id" in active.schema.names and active_row["file_name"] in expected_targets:
                if active_row["id"] in seen_ids:
                    raise RelocationError("canonical ANN contains duplicate row identities")
                seen_ids.add(active_row["id"])
        added = 0
        for row in _iter_rows(
            observations,
            where=f"generation_id = '{plan.manifest_sha256}'",
        ):
            added += 1
            original = row["original_row"]
            raw = _row_bytes(original, original_schema)
            if hashlib.sha256(raw).hexdigest() != row["original_row_sha256"]:
                raise RelocationError("typed original observation payload was not conserved")
            if original.get("file_name") != row["source_file_name"]:
                raise RelocationError("original observation source identity mismatch")
            if row["source_table_version"] != report.source_table_versions[source_name]:
                raise RelocationError("original observation source version mismatch")
            allowed = {target.file_name for target in plan.mapping.get(row["source_file_name"], ())}
            if row["canonical_file_name"] not in allowed:
                raise RelocationError("original observation canonical binding mismatch")
            if (row["canonical_file_name"], row["canonical_id"]) not in valid_bindings:
                raise RelocationError("original observation has no canonical row or ANN variant")
        if added != source_result.observation_rows:
            raise RelocationError("original observation cardinality was not conserved")
        if source_result.output_rows != source_result.source_rows - source_result.dropped_rows + source_result.cloned_rows - source_result.coalesced_rows:
            raise RelocationError("canonical source/coalescing conservation mismatch")


def apply_relocation(
    source_db_dir: Path,
    output_db_dir: Path,
    table_name: str,
    plan: RelocationPlan,
    inventory: Inventory,
    *,
    drop_unmapped: bool,
    batch_size: int = 1024,
    coalesce_equivalent_sources: bool = False,
) -> RelocationReport:
    if plan.preserve_missing and drop_unmapped:
        raise RelocationError("preservation-first relocation cannot drop unmapped rows")
    if coalesce_equivalent_sources != plan.coalesce_equivalent_sources:
        raise RelocationError("fan-in policy must match the explicitly verified manifest plan")
    _check_fanin_sources(plan)
    _check_media_bindings(plan)
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
    output_parent_identity = _media_root_identity(output_parent)
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
        verified_manifest_files=len(
            {
                target.file_name
                for values in plan.mapping.values()
                for target in values
                if not target.missing
            }
        ),
        collection_rows=inventory.collection_rows,
        mapped_rows=inventory.mapped_rows,
        unmapped_rows=len(inventory.unmapped_rows),
        drop_unmapped=drop_unmapped,
        preserved_missing_rows=sum(
            any(target.missing for target in plan.mapping[key])
            for key in inventory.mapped_source_keys
        ),
        excluded_rows=len(inventory.excluded_rows),
        source_table_versions=dict(inventory.source_table_versions),
        coalesce_equivalent_sources=coalesce_equivalent_sources,
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
    unmapped = set(inventory.unmapped_rows) | set(plan.excluded_sources)
    try:
        names = _table_names(source_db)
        names.sort(key=lambda name: (not (name == RELOCATION_TABLE or name.startswith(OBSERVATION_PREFIX)), name))
        for name in names:
            if name == ROOTS_TABLE:
                continue
            print(f"Copying and relocating table {name}...")
            source_table = source_db.open_table(name)
            active_path_table = "file_name" in source_table.schema.names and name != RELOCATION_TABLE and not name.startswith(OBSERVATION_PREFIX)
            if coalesce_equivalent_sources and active_path_table:
                result, specs, observation_name, observation_result = _copy_coalesced_table(
                    source_table, destination_db, name, plan, table_name, unmapped,
                    drop_unmapped, set(inventory.coalescible_targets), partial, max(1, int(batch_size)),
                )
                report.observation_tables[name] = observation_name
                report.tables[observation_name] = observation_result
            else:
                result, specs = _copy_table(
                    source_table, destination_db, name, plan, table_name, unmapped,
                    drop_unmapped, max(1, int(batch_size)),
                )
            report.tables[name] = result
            index_specs[name] = specs
        report.tables[ROOTS_TABLE] = _copy_roots_table(
            source_db, destination_db, plan, max(1, int(batch_size))
        )
        report.tables[RELOCATION_TABLE] = _write_relocation_paths(
            destination_db, plan, max(1, int(batch_size))
        )
        report.copied_assets = _copy_sift_assets(source_db_dir, partial, table_name)

        for name, specs in index_specs.items():
            if not specs:
                continue
            print(f"Restoring {len(specs)} index(es) on {name}...")
            _rebuild_indices(destination_db.open_table(name), specs)

        _validate_indices(destination_db, index_specs)

        _check_source_versions(source_db_dir, inventory.source_table_versions)
        _check_fanin_sources(plan)
        _validate_output(
            partial,
            table_name,
            plan,
            report,
            unmapped,
            set(inventory.mapped_source_keys),
            drop_unmapped,
            set(inventory.main_source_keys),
        )
        report.status = "complete"
        report.completed_unix = int(time.time())
        (partial / "relocation-report.json").write_text(
            json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _check_source_versions(source_db_dir, inventory.source_table_versions)
        _check_fanin_sources(plan)
        _check_media_bindings(plan)
        if _media_root_identity(output_parent) != output_parent_identity:
            raise RelocationError("output database parent identity or mount changed during relocation")
        if _read_json_object(plan.manifest_path)[1] != plan.manifest_sha256:
            raise RelocationError("migration manifest changed during relocation")
        if output_db_dir.exists():
            raise RelocationError(
                "output database appeared during relocation; partial retained"
            )
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
        "main_collection_keys": len(inventory.main_source_keys),
        "side_only_collection_keys": len(inventory.side_only_source_keys),
        "collection_rows": inventory.collection_rows,
        "mapped_rows": inventory.mapped_rows,
        "unmapped_rows": len(inventory.unmapped_rows),
        "excluded_rows": len(inventory.excluded_rows),
        "mapped_sources_without_rows": len(inventory.mapped_sources_without_rows),
        "collisions": len(inventory.collisions),
        "coalescible_targets": len(inventory.coalescible_targets),
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
        help="committed tgbackman reorganisation manifest",
    )
    parser.add_argument(
        "--export-inventory",
        type=Path,
        help="export private path/stat metadata for tgbackman planning; never reads OCR or embeddings",
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
        "--coalesce-equivalent-sources",
        action="store_true",
        help="opt in to SHA-verified many-to-one paths, retaining original metadata and ANN variants in typed observation evidence",
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
    if args.export_inventory is not None:
        if (
            args.manifest is not None
            or args.apply
            or args.drop_unmapped
            or args.output_db_dir is not None
            or args.coalesce_equivalent_sources
        ):
            parser.error(
                "--export-inventory cannot be combined with relocation options"
            )
    elif args.manifest is None:
        parser.error("--manifest is required unless exporting an inventory")
    try:
        if args.export_inventory is not None:
            summary = export_collection_inventory(
                args.db_dir,
                args.table,
                args.collection_id.strip(),
                args.export_inventory,
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        plan = load_relocation_plan(
            args.manifest,
            args.collection_id.strip(),
            hash_workers=args.hash_workers,
            coalesce_equivalent_sources=args.coalesce_equivalent_sources,
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
            "coalesce_equivalent_sources": args.coalesce_equivalent_sources,
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
            coalesce_equivalent_sources=args.coalesce_equivalent_sources,
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
