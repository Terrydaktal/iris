"""Metadata-only regressions for repairing one inserted export-range directory."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import lancedb
import pyarrow as pa
import pytest

SCRIPT = Path(__file__).parents[1] / "tools" / "repair_range_paths.py"
spec = importlib.util.spec_from_file_location("repair_range_paths_under_test", SCRIPT)
assert spec is not None and spec.loader is not None
repair = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = repair
spec.loader.exec_module(repair)

COLLECTION = "fixture_collection"
RANGE = "2020-01-01T00-00-00Z__2020-02-01T00-00-00Z"
OTHER_RANGE = "2020-03-01T00-00-00Z__2020-04-01T00-00-00Z"
OLD_A = f"{COLLECTION}/Chat One/media/payload-a.bin"
OLD_B = f"{COLLECTION}/Chat One/media/payload-b.bin"
NEW_A = f"{COLLECTION}/Chat One/{RANGE}/media/payload-a.bin"
NEW_B = f"{COLLECTION}/Chat One/{RANGE}/media/payload-b.bin"


def placeholder(root: Path, relative: str, size: int = 7) -> Path:
    """Synthetic bytes, never an image, embedding model input or user media."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def test_resolver_inserts_exactly_one_range_and_preserves_tail(tmp_path: Path) -> None:
    root = tmp_path / "media"
    placeholder(root, f"Chat One/{RANGE}/media/payload-a.bin")
    assert repair.resolve_relocated_key(root, OLD_A, COLLECTION, size=7) == NEW_A


def test_resolver_never_guesses_between_two_ranges(tmp_path: Path) -> None:
    root = tmp_path / "media"
    for date_range in (RANGE, OTHER_RANGE):
        placeholder(root, f"Chat One/{date_range}/media/payload-a.bin")
    assert repair.resolve_relocated_key(root, OLD_A, COLLECTION, size=7) is None


@pytest.mark.parametrize(
    "layout",
    [
        f"Different Chat/{RANGE}/media/payload-a.bin",
        "Chat One/not-a-date-range/media/payload-a.bin",
        f"Chat One/{RANGE}/different/payload-a.bin",
        f"Chat One/{RANGE}/{OTHER_RANGE}/media/payload-a.bin",
    ],
)
def test_resolver_rejects_nonexact_or_deeper_relocations(tmp_path: Path, layout: str) -> None:
    root = tmp_path / "media"
    placeholder(root, layout)
    assert repair.resolve_relocated_key(root, OLD_A, COLLECTION, size=7) is None


def test_resolver_checks_expected_size_and_collection_namespace(tmp_path: Path) -> None:
    root = tmp_path / "media"
    placeholder(root, f"Chat One/{RANGE}/media/payload-a.bin", size=8)
    assert repair.resolve_relocated_key(root, OLD_A, COLLECTION, size=7) is None
    assert repair.resolve_relocated_key(root, OLD_A, "different_collection", size=8) is None


@pytest.mark.parametrize(
    "key",
    [
        f"{COLLECTION}/../outside/payload-a.bin",
        f"{COLLECTION}/Chat One/../../outside/payload-a.bin",
        f"{COLLECTION}/Chat One/./media/payload-a.bin",
        "/absolute/payload-a.bin",
        f"{COLLECTION}//Chat One/media/payload-a.bin",
    ],
)
def test_resolver_rejects_traversal_and_noncanonical_keys(tmp_path: Path, key: str) -> None:
    root = tmp_path / "media"
    root.mkdir()
    assert repair.resolve_relocated_key(root, key, COLLECTION, size=7) is None


@pytest.mark.parametrize("link_kind", ["root", "parent", "range", "file"])
def test_resolver_rejects_symlink_components(tmp_path: Path, link_kind: str) -> None:
    root = tmp_path / "media"
    outside = tmp_path / "outside"
    outside.mkdir()
    if link_kind == "root":
        placeholder(outside, f"Chat One/{RANGE}/media/payload-a.bin")
        root.symlink_to(outside, target_is_directory=True)
    elif link_kind == "parent":
        placeholder(outside / "media", f"Chat One/{RANGE}/media/payload-a.bin")
        linked_parent = tmp_path / "linked-parent"
        linked_parent.symlink_to(outside, target_is_directory=True)
        root = linked_parent / "media"
    elif link_kind == "range":
        placeholder(outside, "media/payload-a.bin")
        (root / "Chat One").mkdir(parents=True)
        (root / "Chat One" / RANGE).symlink_to(outside, target_is_directory=True)
    else:
        external = placeholder(outside, "payload-a.bin")
        destination = root / "Chat One" / RANGE / "media"
        destination.mkdir(parents=True)
        (destination / "payload-a.bin").symlink_to(external)
    assert repair.resolve_relocated_key(root, OLD_A, COLLECTION, size=7) is None


def test_metadata_rewrite_changes_only_explicit_path_fields_without_mutation() -> None:
    row = {
        "file_name": OLD_A,
        "id": OLD_A + "|0.250|face:0",
        "dedupe_match_file": OLD_B,
        "sift_match_file": OLD_A,
        "cross_media_matches": [
            {"file_name": OLD_B, "score": 0.9, "kind": "clip"},
            {"file_name": OLD_B, "score": 0.8, "kind": "ocr"},
            {"file_name": "other/untouched.bin", "score": 0.7, "kind": "sift"},
        ],
        "vector": [0.125, -0.75, 0.0],
        "ocr_text": f"Synthetic OCR mentions {OLD_A}, which is not a path field",
        "opaque_metadata": {"file_name": OLD_A, "note": "retain exactly"},
        "phash": "0123456789abcdef",
        "source_size": 7,
        "source_mtime_ns": 987654321,
    }
    before = copy.deepcopy(row)
    result = repair.rewrite_metadata(row, {OLD_A: NEW_A, OLD_B: NEW_B})
    expected = copy.deepcopy(before)
    expected.update(
        file_name=NEW_A,
        id=NEW_A + "|0.250|face:0",
        dedupe_match_file=NEW_B,
        sift_match_file=NEW_A,
    )
    expected["cross_media_matches"][0]["file_name"] = NEW_B
    expected["cross_media_matches"][1]["file_name"] = NEW_B
    assert result == expected
    assert row == before
    assert len(result["cross_media_matches"]) == 3


@pytest.mark.parametrize("identifier", ["opaque-uuid", OLD_A + "suffix", "prefix:" + OLD_A + "|0.0"])
def test_metadata_rewrite_preserves_opaque_identifiers(identifier: str) -> None:
    row = {
        "file_name": OLD_A,
        "id": identifier,
        "vector": [1.0, 2.0],
        "ocr_text": "fixture",
    }
    result = repair.rewrite_metadata(row, {OLD_A: NEW_A})
    assert result["file_name"] == NEW_A
    assert result["id"] == identifier
    assert result["vector"] == row["vector"]
    assert result["ocr_text"] == row["ocr_text"]


def test_metadata_rewrite_is_idempotent_and_keeps_null_references() -> None:
    row = {
        "file_name": OLD_A,
        "dedupe_match_file": None,
        "sift_match_file": "",
        "cross_media_matches": None,
    }
    first = repair.rewrite_metadata(row, {OLD_A: NEW_A})
    assert first == {**row, "file_name": NEW_A}
    assert repair.rewrite_metadata(first, {OLD_A: NEW_A}) == first


def test_metadata_rewrite_updates_an_identifier_that_is_exactly_a_file_key() -> None:
    assert repair.rewrite_metadata({"file_name": OLD_A, "id": OLD_A}, {OLD_A: NEW_A}) == {
        "file_name": NEW_A,
        "id": NEW_A,
    }


CROSS_MEDIA = pa.list_(
    pa.struct(
        [
            pa.field("file_name", pa.string()),
            pa.field("is_video", pa.bool_()),
            pa.field("similarity_pct", pa.float32()),
        ]
    )
)


@pytest.fixture
def archive(tmp_path: Path) -> tuple[Path, Path]:
    """Real disposable Lance tables; all text and vectors are artificial metadata."""
    media = tmp_path / "media"
    placeholder(media, f"Chat One/{RANGE}/media/payload-a.bin")
    placeholder(media, f"Chat One/{RANGE}/media/payload-b.bin")
    database_path = tmp_path / "lance"
    database = lancedb.connect(str(database_path))
    main_schema = pa.schema(
        [
            pa.field("file_name", pa.string()),
            pa.field("collection_id", pa.string()),
            pa.field("source_size", pa.int64()),
            pa.field("source_mtime_ns", pa.int64()),
            pa.field("dedupe_match_file", pa.string()),
            pa.field("sift_match_file", pa.string()),
            pa.field("cross_media_matches", CROSS_MEDIA),
            pa.field("vector", pa.list_(pa.float32(), 3)),
            pa.field("ocr_text", pa.string()),
            pa.field("phash", pa.string()),
            pa.field("opaque_json", pa.string()),
        ]
    )
    main = database.create_table(
        "media_index",
        [
            {
                "file_name": OLD_A,
                "collection_id": COLLECTION,
                "source_size": 7,
                "source_mtime_ns": 1234,
                "dedupe_match_file": OLD_B,
                "sift_match_file": OLD_B,
                "cross_media_matches": [
                    {"file_name": OLD_B, "is_video": False, "similarity_pct": 93.75},
                    {"file_name": OLD_B, "is_video": False, "similarity_pct": 87.5},
                ],
                "vector": [0.125, -0.75, 0.5],
                "ocr_text": "Artificial OCR alpha",
                "phash": "0102030405060708",
                "opaque_json": '{"fixture":"A"}',
            },
            {
                "file_name": OLD_B,
                "collection_id": COLLECTION,
                "source_size": 7,
                "source_mtime_ns": 5678,
                "dedupe_match_file": OLD_A,
                "sift_match_file": None,
                "cross_media_matches": [],
                "vector": [0.25, -0.5, 0.75],
                "ocr_text": "Artificial OCR beta",
                "phash": "0807060504030201",
                "opaque_json": '{"fixture":"B"}',
            },
            {
                "file_name": "other/reference.bin",
                "collection_id": "other",
                "source_size": 3,
                "source_mtime_ns": 9999,
                "dedupe_match_file": OLD_A,
                "sift_match_file": OLD_B,
                "cross_media_matches": [{"file_name": OLD_A, "is_video": False, "similarity_pct": 75.0}],
                "vector": [0.5, 0.25, -0.125],
                "ocr_text": "Artificial OCR other collection",
                "phash": "0000000000000000",
                "opaque_json": '{"fixture":"other"}',
            },
        ],
        schema=main_schema,
    )
    main.create_scalar_index("file_name", name="fixture_file_name_index")
    ann_schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("file_name", pa.string()),
            pa.field("timestamp_sec", pa.float32()),
            pa.field("vector", pa.list_(pa.float32(), 3)),
            pa.field("ocr_text", pa.string()),
        ]
    )
    for name in ("media_index_clip_ann", "media_index_ocr_ann"):
        ann = database.create_table(
            name,
            [
                {
                    "id": OLD_A + "|0.250",
                    "file_name": OLD_A,
                    "timestamp_sec": 0.25,
                    "vector": [0.125, 0.25, 0.5],
                    "ocr_text": "Artificial ANN OCR alpha",
                },
                {
                    "id": "opaque-embedding-uuid",
                    "file_name": OLD_B,
                    "timestamp_sec": 1.5,
                    "vector": [-0.125, -0.25, -0.5],
                    "ocr_text": "Artificial ANN OCR beta",
                },
            ],
            schema=ann_schema,
        )
        if name == "media_index_clip_ann":
            ann.create_index(
                vector_column_name="vector",
                metric="cosine",
                index_type="IVF_FLAT",
                num_partitions=1,
                name="fixture_clip_vector_index",
            )
    database.create_table(
        "collection_roots",
        [
            {"collection_id": COLLECTION, "root_path": str(media)},
            {"collection_id": "other", "root_path": str(tmp_path / "other")},
        ],
        schema=pa.schema(
            [
                pa.field("collection_id", pa.string()),
                pa.field("root_path", pa.string()),
            ]
        ),
    )
    return database_path, media


def snapshot(database_path: Path) -> dict[str, list[dict]]:
    database = lancedb.connect(str(database_path))
    return {
        name: sorted(
            database.open_table(name).to_arrow().to_pylist(),
            key=lambda row: json.dumps(row, sort_keys=True),
        )
        for name in database.list_tables().tables
    }


def expected_repaired(before: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Independent fixture oracle: expected literal paths, no implementation helper."""
    expected = copy.deepcopy(before)
    for name, rows in expected.items():
        if name == "collection_roots":
            continue
        for row in rows:
            if row["file_name"] == OLD_A:
                row["file_name"] = NEW_A
            elif row["file_name"] == OLD_B:
                row["file_name"] = NEW_B
            if row.get("id") == OLD_A + "|0.250":
                row["id"] = NEW_A + "|0.250"
            for field in ("dedupe_match_file", "sift_match_file"):
                if row.get(field) == OLD_A:
                    row[field] = NEW_A
                elif row.get(field) == OLD_B:
                    row[field] = NEW_B
            for match in row.get("cross_media_matches") or []:
                if match["file_name"] == OLD_A:
                    match["file_name"] = NEW_A
                elif match["file_name"] == OLD_B:
                    match["file_name"] = NEW_B
        rows.sort(key=lambda row: json.dumps(row, sort_keys=True))
    return expected


def test_plan_is_read_only_and_maps_each_key_once(archive: tuple[Path, Path]) -> None:
    database_path, media = archive
    before = snapshot(database_path)
    database = lancedb.connect(str(database_path))
    versions = {name: database.open_table(name).version for name in before}
    plan = repair.build_plan(database, COLLECTION)
    assert plan["mapping"] == {OLD_A: NEW_A, OLD_B: NEW_B}
    assert plan["root"] == str(media)
    assert plan["collection_id"] == COLLECTION
    assert plan["unresolved"] == plan["ambiguous"] == plan["collisions"] == 0
    for name in ("media_index", "media_index_clip_ann", "media_index_ocr_ann"):
        assert plan["versions"][name] == versions[name]
        assert plan["counts"][name] == len(before[name])
    assert snapshot(database_path) == before
    assert {name: database.open_table(name).version for name in before} == versions


def test_apply_preserves_vectors_ocr_all_nonpath_metadata_and_duplicate_references(
    archive: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    database_path, _media = archive
    database = lancedb.connect(str(database_path))
    before = snapshot(database_path)
    index_names = {
        name: [index.name for index in database.open_table(name).list_indices()] for name in before
    }
    plan = repair.build_plan(database, COLLECTION)
    report_path = tmp_path / "private-report.json"
    report = repair.apply_plan(database, plan, report_path)
    assert report["status"] == "complete"
    assert report_path.is_file()
    assert snapshot(database_path) == expected_repaired(before)
    assert {
        name: [index.name for index in database.open_table(name).list_indices()] for name in before
    } == index_names
    for name in before:
        assert database.open_table(name).count_rows() == len(before[name])
        assert "__iris_range_repair_key" not in database.open_table(name).schema.names
    matched = (
        database.open_table("media_index")
        .search()
        .where(
            f"file_name = '{NEW_A}'",
            prefilter=True,
        )
        .limit(10)
        .to_list()
    )
    assert len(matched) == 1
    assert matched[0]["ocr_text"] == "Artificial OCR alpha"
    assert matched[0]["vector"] == [0.125, -0.75, 0.5]
    nearest = database.open_table("media_index_clip_ann").search([0.125, 0.25, 0.5]).limit(1).to_list()
    assert nearest[0]["file_name"] == NEW_A
    assert nearest[0]["id"] == NEW_A + "|0.250"
    assert nearest[0]["ocr_text"] == "Artificial ANN OCR alpha"


def test_applied_repair_is_idempotent_on_replanning(archive: tuple[Path, Path], tmp_path: Path) -> None:
    database_path, _media = archive
    database = lancedb.connect(str(database_path))
    repair.apply_plan(database, repair.build_plan(database, COLLECTION), tmp_path / "first.json")
    first = snapshot(database_path)
    second_plan = repair.build_plan(database, COLLECTION)
    assert second_plan["mapping"] == {}
    assert second_plan["tables"] == {}
    repair.apply_plan(database, second_plan, tmp_path / "second.json")
    assert snapshot(database_path) == first


def test_existing_original_file_is_never_redirected(archive: tuple[Path, Path]) -> None:
    database_path, media = archive
    placeholder(media, "Chat One/media/payload-a.bin")
    assert repair.resolve_relocated_key(media, OLD_A, COLLECTION, size=7) is None
    assert repair.build_plan(lancedb.connect(str(database_path)), COLLECTION)["mapping"] == {OLD_B: NEW_B}


def test_ambiguous_destination_is_skipped_without_losing_other_repairs(
    archive: tuple[Path, Path],
) -> None:
    database_path, media = archive
    placeholder(media, f"Chat One/{OTHER_RANGE}/media/payload-a.bin")
    plan = repair.build_plan(lancedb.connect(str(database_path)), COLLECTION)
    assert plan["mapping"] == {OLD_B: NEW_B}
    assert plan["ambiguous"] == 1


def test_collision_is_not_merged_or_duplicated(archive: tuple[Path, Path]) -> None:
    database_path, _media = archive
    database = lancedb.connect(str(database_path))
    table = database.open_table("media_index")
    row = copy.deepcopy(table.to_arrow().to_pylist()[0])
    row["file_name"] = NEW_A
    row["ocr_text"] = "Distinct existing destination OCR must survive"
    table.add(pa.Table.from_pylist([row], schema=table.schema))
    before = snapshot(database_path)
    plan = repair.build_plan(database, COLLECTION)
    assert plan["mapping"] == {OLD_B: NEW_B}
    assert plan["collisions"] == 1
    assert snapshot(database_path) == before


def test_apply_rejects_stale_table_versions_before_writing(
    archive: tuple[Path, Path], tmp_path: Path
) -> None:
    database_path, _media = archive
    database = lancedb.connect(str(database_path))
    plan = repair.build_plan(database, COLLECTION)
    database.open_table("media_index_clip_ann").update(
        where="id = 'opaque-embedding-uuid'",
        values={"ocr_text": "Concurrent fixture edit"},
    )
    after_concurrent_edit = snapshot(database_path)
    with pytest.raises((RuntimeError, ValueError), match="(?i)version|changed|stale|concurrent"):
        repair.apply_plan(database, plan, tmp_path / "stale.json")
    assert snapshot(database_path) == after_concurrent_edit


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("injected second-table failure"),
        KeyboardInterrupt("injected cancellation"),
    ],
)
def test_partial_apply_failure_restores_every_touched_table(
    archive: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    database_path, _media = archive
    database = lancedb.connect(str(database_path))
    before = snapshot(database_path)
    plan = repair.build_plan(database, COLLECTION)
    original_patch = repair._patch_table
    calls = 0

    def fail_after_first_table(table, rows, progress_callback):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise error
        return original_patch(table, rows, progress_callback)

    monkeypatch.setattr(repair, "_patch_table", fail_after_first_table)
    with pytest.raises(type(error), match="injected"):
        repair.apply_plan(database, plan, tmp_path / "failure.json")
    assert calls == 2
    assert snapshot(database_path) == before
    for name in before:
        assert "__iris_range_repair_key" not in database.open_table(name).schema.names
    journal = json.loads((tmp_path / "failure.json").read_text())
    assert journal["status"] == "rolled-back"
    assert journal["rollback_errors"] == []
    assert type(error).__name__ in journal["error"]


def test_preexisting_private_join_column_is_rejected_without_writes(
    archive: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    database_path, _media = archive
    database = lancedb.connect(str(database_path))
    table = database.open_table("media_index_clip_ann")
    table.add_columns({"__iris_range_repair_key": "file_name"})
    before = snapshot(database_path)
    with pytest.raises((RuntimeError, ValueError), match="(?i)column|temporary|private|repair"):
        plan = repair.build_plan(database, COLLECTION)
        repair.apply_plan(database, plan, tmp_path / "private-column.json")
    assert snapshot(database_path) == before


def test_ann_path_derived_identity_collision_refuses_without_writes(
    archive: tuple[Path, Path],
) -> None:
    database_path, _media = archive
    database = lancedb.connect(str(database_path))
    table = database.open_table("media_index_clip_ann")
    row = copy.deepcopy(table.to_arrow().to_pylist()[0])
    row["id"] = NEW_A + "|0.250"
    row["file_name"] = "other/different.bin"
    row["ocr_text"] = "Distinct colliding ANN record"
    table.add(pa.Table.from_pylist([row], schema=table.schema))
    before = snapshot(database_path)
    with pytest.raises((RuntimeError, ValueError), match="(?i)duplicate|identit|collision"):
        repair.build_plan(database, COLLECTION)
    assert snapshot(database_path) == before


def test_missing_destination_retains_its_ocr_vectors_and_references(
    archive: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    database_path, media = archive
    missing = media / "Chat One" / RANGE / "media" / "payload-a.bin"
    assert missing.is_file() and not missing.is_symlink()
    missing.unlink()
    database = lancedb.connect(str(database_path))
    before = snapshot(database_path)
    plan = repair.build_plan(database, COLLECTION)
    assert plan["mapping"] == {OLD_B: NEW_B}
    assert plan["unresolved"] == 1
    repair.apply_plan(database, plan, tmp_path / "partial.json")
    after = snapshot(database_path)
    original_main = next(row for row in before["media_index"] if row["file_name"] == OLD_A)
    missing_main = next(row for row in after["media_index"] if row["file_name"] == OLD_A)
    for field in (
        "ocr_text",
        "vector",
        "phash",
        "source_size",
        "source_mtime_ns",
        "opaque_json",
    ):
        assert missing_main[field] == original_main[field]
    for name in ("media_index_clip_ann", "media_index_ocr_ann"):
        before_missing = next(row for row in before[name] if row["file_name"] == OLD_A)
        assert next(row for row in after[name] if row["file_name"] == OLD_A) == before_missing
    for name in before:
        assert len(after[name]) == len(before[name])
    assert (
        next(row for row in after["media_index"] if row["file_name"] == NEW_B)["dedupe_match_file"] == OLD_A
    )


def test_existing_recovery_report_is_never_overwritten(
    archive: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    database_path, _media = archive
    database = lancedb.connect(str(database_path))
    plan = repair.build_plan(database, COLLECTION)
    report_path = tmp_path / "existing-report.json"
    original_report = b'{"synthetic_previous_recovery":"preserve"}\n'
    report_path.write_bytes(original_report)
    before = snapshot(database_path)
    with pytest.raises((RuntimeError, ValueError), match="(?i)existing|new report|overwritten"):
        repair.apply_plan(database, plan, report_path)
    assert report_path.read_bytes() == original_report
    assert snapshot(database_path) == before
