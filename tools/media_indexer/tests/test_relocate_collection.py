from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import lancedb
import pyarrow as pa
import relocate_collection
from relocate_collection import (
    RelocationError,
    _table_names,
    apply_relocation,
    export_collection_inventory,
    inspect_database,
    load_relocation_plan,
    main,
)

CROSS_MEDIA_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("file_name", pa.string()),
            pa.field("is_video", pa.bool_()),
            pa.field("similarity_pct", pa.float32()),
        ]
    )
)


class RelocateCollectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source_root = self.root / "old-media"
        self.destination_root = self.root / "new-media"
        self.source_root.mkdir()
        self.destination_root.mkdir()
        (self.source_root / "Legacy").mkdir()
        self.chat_one = self.destination_root / "Chat One__user_1"
        self.chat_two = self.destination_root / "Chat Two__user_2"
        (self.chat_one / "media/photo").mkdir(parents=True)
        (self.chat_two / "media/photo").mkdir(parents=True)

        self.old_a = self.source_root / "Legacy/a.jpg"
        self.old_b = self.source_root / "Legacy/b.jpg"
        self.old_a.write_bytes(b"image-a")
        self.old_b.write_bytes(b"image-b")
        self.new_a_one = self.chat_one / "media/photo/a.jpg"
        self.new_a_two = self.chat_two / "media/photo/a.jpg"
        self.new_b = self.chat_one / "media/photo/b.jpg"
        self.new_a_one.write_bytes(b"image-a")
        self.new_a_two.write_bytes(b"image-a")
        self.new_b.write_bytes(b"image-b")

        self.manifest = self.root / "reorganize.json"
        self._write_manifest()
        self.db_dir = self.root / "lancedb"
        self._create_database(include_unmapped=False)

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _write_manifest(self, *, committed: bool = True) -> None:
        payload = {
            "version": 1,
            "db": str(self.root / "telegram.db"),
            "source_root": str(self.source_root),
            "destination_root": str(self.destination_root),
            "db_committed": committed,
            "missing": [],
            "unsafe": [],
            "mismatched": [],
            "chats": [
                {"chat_id": "one", "destination": str(self.chat_one)},
                {"chat_id": "two", "destination": str(self.chat_two)},
            ],
            "media": [
                {
                    "chat_id": "one",
                    "message_id": 1,
                    "source": str(self.old_a),
                    "relative": "media/photo/a.jpg",
                    "size": self.new_a_one.stat().st_size,
                    "sha256": self._digest(self.new_a_one),
                    "media_type": "photo",
                    "status": "copied",
                },
                {
                    "chat_id": "two",
                    "message_id": 2,
                    "source": str(self.old_a),
                    "relative": "media/photo/a.jpg",
                    "size": self.new_a_two.stat().st_size,
                    "sha256": self._digest(self.new_a_two),
                    "media_type": "photo",
                    "status": "reused",
                },
                {
                    "chat_id": "one",
                    "message_id": 3,
                    "source": str(self.old_b),
                    "relative": "media/photo/b.jpg",
                    "size": self.new_b.stat().st_size,
                    "sha256": self._digest(self.new_b),
                    "media_type": "photo",
                    "status": "copied",
                },
            ],
        }
        self.manifest.write_text(json.dumps(payload), encoding="utf-8")

    def _create_database(self, *, include_unmapped: bool) -> None:
        db = lancedb.connect(str(self.db_dir))
        main_schema = pa.schema(
            [
                pa.field("file_name", pa.string()),
                pa.field("collection_id", pa.string()),
                pa.field("source_size", pa.int64()),
                pa.field("source_mtime_ns", pa.int64()),
                pa.field("dedupe_match_file", pa.string()),
                pa.field("sift_match_file", pa.string()),
                pa.field("cross_media_matches", CROSS_MEDIA_TYPE),
            ]
        )
        old_a_key = "telegram_backup/Legacy/a.jpg"
        old_b_key = "telegram_backup/Legacy/b.jpg"
        rows = [
            {
                "file_name": old_a_key,
                "collection_id": "telegram_backup",
                "source_size": 1,
                "source_mtime_ns": 1,
                "dedupe_match_file": old_b_key,
                "sift_match_file": old_b_key,
                "cross_media_matches": [
                    {
                        "file_name": old_b_key,
                        "is_video": False,
                        "similarity_pct": 99.0,
                    }
                ],
            },
            {
                "file_name": old_b_key,
                "collection_id": "telegram_backup",
                "source_size": 1,
                "source_mtime_ns": 1,
                "dedupe_match_file": old_a_key,
                "sift_match_file": None,
                "cross_media_matches": [],
            },
            {
                "file_name": "other/reference.jpg",
                "collection_id": "other",
                "source_size": 4,
                "source_mtime_ns": 4,
                "dedupe_match_file": old_a_key,
                "sift_match_file": None,
                "cross_media_matches": [],
            },
        ]
        if include_unmapped:
            rows[2]["dedupe_match_file"] = "telegram_backup/Legacy/unmapped.jpg"
            rows[2]["cross_media_matches"] = [
                {
                    "file_name": "telegram_backup/Legacy/unmapped.jpg",
                    "is_video": False,
                    "similarity_pct": 98.0,
                }
            ]
            rows.append(
                {
                    "file_name": "telegram_backup/Legacy/unmapped.jpg",
                    "collection_id": "telegram_backup",
                    "source_size": 1,
                    "source_mtime_ns": 1,
                    "dedupe_match_file": None,
                    "sift_match_file": None,
                    "cross_media_matches": [],
                }
            )
        main = db.create_table("media_index", rows, schema=main_schema)
        main.create_scalar_index("file_name", name="file_name_idx")
        ann_schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("file_name", pa.string()),
                pa.field("timestamp_sec", pa.float32()),
                pa.field("vector", pa.list_(pa.float32(), 2)),
            ]
        )
        ann = db.create_table(
            "media_index_clip_ann",
            [
                {
                    "id": f"{old_a_key}|0.000",
                    "file_name": old_a_key,
                    "timestamp_sec": 0.0,
                    "vector": [0.25, 0.75],
                }
            ],
            schema=ann_schema,
        )
        ann.create_index(
            vector_column_name="vector",
            metric="cosine",
            index_type="IVF_FLAT",
            num_partitions=1,
            name="media_index_clip_ann_vec_idx",
        )
        roots_schema = pa.schema(
            [
                pa.field("collection_id", pa.string(), nullable=False),
                pa.field("root_path", pa.string(), nullable=False),
            ]
        )
        db.create_table(
            "collection_roots",
            [
                {
                    "collection_id": "telegram_backup",
                    "root_path": str(self.source_root),
                },
                {"collection_id": "other", "root_path": str(self.root / "other")},
            ],
            schema=roots_schema,
        )

    def test_apply_clones_rows_and_rewrites_all_path_references(self) -> None:
        plan = load_relocation_plan(self.manifest, "telegram_backup", hash_workers=2)
        inventory = inspect_database(self.db_dir, "media_index", plan)
        self.assertEqual(inventory.collection_rows, 2)
        self.assertEqual(inventory.mapped_rows, 2)
        self.assertEqual(inventory.unmapped_rows, [])

        output = self.root / "lancedb-relocated"
        report = apply_relocation(
            self.db_dir,
            output,
            "media_index",
            plan,
            inventory,
            drop_unmapped=False,
            batch_size=2,
        )
        self.assertEqual(report.status, "complete")
        self.assertTrue((output / "relocation-report.json").is_file())

        source_db = lancedb.connect(str(self.db_dir))
        self.assertEqual(source_db.open_table("media_index").count_rows(), 3)
        output_db = lancedb.connect(str(output))
        rows = output_db.open_table("media_index").to_arrow().to_pylist()
        by_name = {row["file_name"]: row for row in rows}
        a_one = "telegram_backup/Chat One__user_1/media/photo/a.jpg"
        a_two = "telegram_backup/Chat Two__user_2/media/photo/a.jpg"
        b_one = "telegram_backup/Chat One__user_1/media/photo/b.jpg"
        self.assertEqual(set(by_name), {a_one, a_two, b_one, "other/reference.jpg"})
        self.assertEqual(by_name[a_one]["dedupe_match_file"], b_one)
        self.assertEqual(by_name[a_two]["dedupe_match_file"], b_one)
        self.assertEqual(by_name[b_one]["dedupe_match_file"], a_one)
        self.assertEqual(by_name["other/reference.jpg"]["dedupe_match_file"], a_one)
        self.assertEqual(by_name[a_one]["source_size"], len(b"image-a"))
        self.assertEqual(
            by_name[a_one]["source_mtime_ns"], self.new_a_one.stat().st_mtime_ns
        )

        ann_rows = output_db.open_table("media_index_clip_ann").to_arrow().to_pylist()
        self.assertEqual({row["file_name"] for row in ann_rows}, {a_one, a_two})
        self.assertEqual(
            {row["id"] for row in ann_rows},
            {f"{a_one}|0.000", f"{a_two}|0.000"},
        )
        self.assertEqual(
            output_db.open_table("media_index_clip_ann").list_indices()[0].name,
            "media_index_clip_ann_vec_idx",
        )
        roots = {
            row["collection_id"]: row["root_path"]
            for row in output_db.open_table("collection_roots").to_arrow().to_pylist()
        }
        self.assertEqual(roots["telegram_backup"], str(self.destination_root))

    def test_published_report_retains_original_captured_source_versions(self) -> None:
        source_db = lancedb.connect(str(self.db_dir))
        expected_versions = {
            name: int(source_db.open_table(name).version)
            for name in _table_names(source_db)
        }
        plan = load_relocation_plan(self.manifest, "telegram_backup")
        inventory = inspect_database(self.db_dir, "media_index", plan)
        self.assertEqual(inventory.source_table_versions, expected_versions)
        output = self.root / "lancedb-report-versions"
        report = apply_relocation(
            self.db_dir,
            output,
            "media_index",
            plan,
            inventory,
            drop_unmapped=False,
        )
        persisted = json.loads((output / "relocation-report.json").read_text())
        self.assertEqual(report.source_table_versions, expected_versions)
        self.assertEqual(persisted["source_table_versions"], expected_versions)
        self.assertIsNot(report.source_table_versions, inventory.source_table_versions)
        self.assertEqual(
            {
                name: int(source_db.open_table(name).version)
                for name in _table_names(source_db)
            },
            expected_versions,
        )

    def test_uncommitted_or_tampered_manifest_is_rejected(self) -> None:
        self._write_manifest(committed=False)
        with self.assertRaisesRegex(RelocationError, "not committed"):
            load_relocation_plan(self.manifest, "telegram_backup")

        self._write_manifest(committed=True)
        self.new_b.write_bytes(b"tampered")
        with self.assertRaisesRegex(RelocationError, "size mismatch|SHA-256 mismatch"):
            load_relocation_plan(self.manifest, "telegram_backup")

    def test_unmapped_rows_require_explicit_shadow_database_drop(self) -> None:
        second_db = self.root / "lancedb-with-unmapped"
        self.db_dir = second_db
        self._create_database(include_unmapped=True)
        plan = load_relocation_plan(self.manifest, "telegram_backup")
        inventory = inspect_database(second_db, "media_index", plan)
        self.assertEqual(
            inventory.unmapped_rows, ["telegram_backup/Legacy/unmapped.jpg"]
        )
        with self.assertRaisesRegex(RelocationError, "absent from the manifest"):
            apply_relocation(
                second_db,
                self.root / "blocked-output",
                "media_index",
                plan,
                inventory,
                drop_unmapped=False,
            )

        output = self.root / "drop-output"
        apply_relocation(
            second_db,
            output,
            "media_index",
            plan,
            inventory,
            drop_unmapped=True,
            batch_size=2,
        )
        output_rows = (
            lancedb.connect(str(output))
            .open_table("media_index")
            .to_arrow()
            .to_pylist()
        )
        names = {row["file_name"] for row in output_rows}
        self.assertNotIn("telegram_backup/Legacy/unmapped.jpg", names)
        other = next(
            row for row in output_rows if row["file_name"] == "other/reference.jpg"
        )
        self.assertIsNone(other["dedupe_match_file"])
        self.assertEqual(other["cross_media_matches"], [])

    def test_dry_run_exit_status_matches_unmapped_policy(self) -> None:
        second_db = self.root / "lancedb-dry-run-unmapped"
        self.db_dir = second_db
        self._create_database(include_unmapped=True)
        arguments = [
            "--db-dir",
            str(second_db),
            "--manifest",
            str(self.manifest),
            "--collection-id",
            "telegram_backup",
        ]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main(arguments), 2)
            self.assertEqual(main([*arguments, "--drop-unmapped"]), 0)

    def test_unused_manifest_source_does_not_require_an_iris_row(self) -> None:
        old_unused = self.source_root / "Legacy/unused.jpg"
        old_unused.write_bytes(b"unused")
        new_unused = self.chat_one / "media/photo/unused.jpg"
        new_unused.write_bytes(b"unused")
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        payload["media"].append(
            {
                "chat_id": "one",
                "message_id": 4,
                "source": str(old_unused),
                "relative": "media/photo/unused.jpg",
                "size": new_unused.stat().st_size,
                "sha256": self._digest(new_unused),
                "media_type": "photo",
                "status": "copied",
            }
        )
        self.manifest.write_text(json.dumps(payload), encoding="utf-8")
        plan = load_relocation_plan(self.manifest, "telegram_backup")
        inventory = inspect_database(self.db_dir, "media_index", plan)
        self.assertEqual(
            inventory.mapped_sources_without_rows,
            ["telegram_backup/Legacy/unused.jpg"],
        )
        output = self.root / "unused-manifest-output"
        apply_relocation(
            self.db_dir,
            output,
            "media_index",
            plan,
            inventory,
            drop_unmapped=False,
            batch_size=2,
        )
        names = {
            row["file_name"]
            for row in lancedb.connect(str(output))
            .open_table("media_index")
            .to_arrow()
            .to_pylist()
        }
        self.assertNotIn(
            "telegram_backup/Chat One__user_1/media/photo/unused.jpg", names
        )

    def _preserve_unmapped(self, *, missing: bool) -> str:
        self.db_dir = self.root / "lancedb-preservation"
        self._create_database(include_unmapped=True)
        old_key = "telegram_backup/Legacy/unmapped.jpg"
        destination = self.destination_root / "legacy-unreferenced"
        destination.mkdir()
        new_file = destination / "Legacy/unmapped.jpg"
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        payload["preserve_missing"] = True
        payload["chats"].append(
            {"chat_id": "__iris_unreferenced__", "destination": str(destination)}
        )
        row = {
            "chat_id": "__iris_unreferenced__",
            "source": str(self.source_root / "Legacy/unmapped.jpg"),
            "relative": "Legacy/unmapped.jpg",
            "status": "missing" if missing else "copied",
            "reason": "not present in original archive",
            "size": None,
            "sha256": None,
        }
        if missing:
            payload["missing"] = ["original media absent"]
            payload["missing_media"] = [row]
        else:
            (self.source_root / "Legacy/unmapped.jpg").write_bytes(b"iris-only")
            new_file.parent.mkdir()
            new_file.write_bytes(b"iris-only")
            row.update(size=len(b"iris-only"), sha256=self._digest(new_file))
            payload["media"].append(row)
        self.manifest.write_text(json.dumps(payload), encoding="utf-8")
        db = lancedb.connect(str(self.db_dir))
        db.open_table("media_index_clip_ann").add(
            [
                {
                    "id": old_key + "|0.000",
                    "file_name": old_key,
                    "timestamp_sec": 0.0,
                    "vector": [0.8, 0.2],
                }
            ]
        )
        return "telegram_backup/legacy-unreferenced/Legacy/unmapped.jpg"

    def test_missing_rows_and_vectors_are_preserved_with_auditable_paths(self) -> None:
        new_key = self._preserve_unmapped(missing=True)
        plan = load_relocation_plan(self.manifest, "telegram_backup")
        inventory = inspect_database(self.db_dir, "media_index", plan)
        output = self.root / "preserved-missing-output"
        report = apply_relocation(
            self.db_dir, output, "media_index", plan, inventory, drop_unmapped=False
        )
        self.assertEqual(report.preserved_missing_rows, 1)
        db = lancedb.connect(str(output))
        rows = {
            row["file_name"]: row
            for row in db.open_table("media_index").to_arrow().to_pylist()
        }
        self.assertEqual(rows[new_key]["source_size"], 1)
        self.assertEqual(rows[new_key]["source_mtime_ns"], 1)
        self.assertEqual(rows["other/reference.jpg"]["dedupe_match_file"], new_key)
        ann = [
            row
            for row in db.open_table("media_index_clip_ann").to_arrow().to_pylist()
            if row["file_name"] == new_key
        ]
        self.assertEqual(len(ann), 1)
        self.assertAlmostEqual(ann[0]["vector"][0], 0.8, places=6)
        paths = [
            row
            for row in db.open_table("relocation_paths").to_arrow().to_pylist()
            if row["file_name"] == new_key
        ]
        self.assertEqual(paths[0]["status"], "missing")
        self.assertEqual(
            paths[0]["old_file_name"], "telegram_backup/Legacy/unmapped.jpg"
        )
        self.assertFalse(
            (self.destination_root / "legacy-unreferenced/Legacy/unmapped.jpg").exists()
        )
        self.assertEqual(
            lancedb.connect(str(self.db_dir)).open_table("media_index").count_rows(), 4
        )

    def test_existing_iris_only_files_preserved_without_drop(self) -> None:
        new_key = self._preserve_unmapped(missing=False)
        plan = load_relocation_plan(self.manifest, "telegram_backup")
        inventory = inspect_database(self.db_dir, "media_index", plan)
        output = self.root / "preserved-existing-output"
        report = apply_relocation(
            self.db_dir, output, "media_index", plan, inventory, drop_unmapped=False
        )
        self.assertEqual(report.preserved_missing_rows, 0)
        db = lancedb.connect(str(output))
        self.assertIn(
            new_key,
            {
                row["file_name"]
                for row in db.open_table("media_index").to_arrow().to_pylist()
            },
        )
        self.assertEqual(db.open_table("media_index_clip_ann").count_rows(), 3)

    def test_missing_policy_and_false_missing_destination_rejected(self) -> None:
        self._preserve_unmapped(missing=True)
        payload = json.loads(self.manifest.read_text())
        payload["preserve_missing"] = False
        self.manifest.write_text(json.dumps(payload))
        with self.assertRaises(RelocationError):
            load_relocation_plan(self.manifest, "telegram_backup")
        payload["preserve_missing"] = True
        self.manifest.write_text(json.dumps(payload))
        destination = self.destination_root / "legacy-unreferenced/Legacy/unmapped.jpg"
        destination.parent.mkdir()
        destination.write_bytes(b"unexpected")
        with self.assertRaisesRegex(RelocationError, "unexpectedly exists"):
            load_relocation_plan(self.manifest, "telegram_backup")

    def test_metadata_inventory_is_private_and_source_versions_unchanged(self) -> None:
        self._preserve_unmapped(missing=True)
        source = lancedb.connect(str(self.db_dir))
        versions = {
            name: source.open_table(name).version for name in _table_names(source)
        }
        inventory = self.root / "inventory.json"
        summary = export_collection_inventory(
            self.db_dir, "media_index", "telegram_backup", inventory
        )
        self.assertEqual(summary["files"], 3)
        self.assertEqual(summary["missing"], 1)
        self.assertEqual(os.stat(inventory).st_mode & 0o777, 0o600)
        payload = json.loads(inventory.read_text())
        self.assertEqual(
            set(payload["files"][0]),
            {"file_name", "source", "size", "mtime_ns", "exists"},
        )
        self.assertEqual(versions, payload["source_table_versions"])
        self.assertEqual(
            versions,
            {name: source.open_table(name).version for name in _table_names(source)},
        )
        with self.assertRaises(FileExistsError):
            export_collection_inventory(
                self.db_dir, "media_index", "telegram_backup", inventory
            )

    def test_source_version_change_retains_partial_without_publishing(self) -> None:
        plan = load_relocation_plan(self.manifest, "telegram_backup")
        inventory = inspect_database(self.db_dir, "media_index", plan)
        output = self.root / "changed-source-output"
        with (
            patch(
                "relocate_collection._check_source_versions",
                side_effect=RelocationError("source database changed"),
            ),
            self.assertRaisesRegex(RelocationError, "source database changed"),
        ):
            apply_relocation(
                self.db_dir, output, "media_index", plan, inventory, drop_unmapped=False
            )
        self.assertFalse(output.exists())
        self.assertEqual(
            len(list(self.root.glob(".changed-source-output.partial-*"))), 1
        )

    def test_real_hnsw_sq_index_and_fanout_vectors_survive(self) -> None:
        source = lancedb.connect(str(self.db_dir)).open_table("media_index_clip_ann")
        source.create_index(
            vector_column_name="vector",
            metric="cosine",
            index_type="IVF_HNSW_SQ",
            num_partitions=1,
            replace=True,
            name="media_index_clip_ann_vec_idx",
        )
        plan = load_relocation_plan(self.manifest, "telegram_backup")
        inventory = inspect_database(self.db_dir, "media_index", plan)
        output = self.root / "hnsw-output"
        apply_relocation(
            self.db_dir, output, "media_index", plan, inventory, drop_unmapped=False
        )
        table = lancedb.connect(str(output)).open_table("media_index_clip_ann")
        self.assertEqual(table.list_indices()[0].index_type, "IvfHnswSq")
        rows = table.search([0.25, 0.75]).limit(2).to_list()
        self.assertEqual(len(rows), 2)
        self.assertEqual({tuple(row["vector"]) for row in rows}, {(0.25, 0.75)})

    def test_optimized_cli_inventory_then_preservation_apply_outside_repo(self) -> None:
        new_key = self._preserve_unmapped(missing=True)
        script = Path(__file__).resolve().parents[1] / "relocate_collection.py"
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(self.root / "isolated-home"),
            "XDG_CONFIG_HOME": str(self.root / "isolated-config"),
            "XDG_DATA_HOME": str(self.root / "isolated-data"),
        }
        inventory = self.root / "private inventory.json"
        result = subprocess.run(
            [
                sys.executable,
                "-O",
                str(script),
                "--db-dir",
                str(self.db_dir),
                "--collection-id",
                "telegram_backup",
                "--export-inventory",
                str(inventory),
            ],
            cwd=self.root,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["missing"], 1)
        output = self.root / "new lance with spaces"
        result = subprocess.run(
            [
                sys.executable,
                "-O",
                str(script),
                "--db-dir",
                str(self.db_dir),
                "--collection-id",
                "telegram_backup",
                "--manifest",
                str(self.manifest),
                "--apply",
                "--output-db-dir",
                str(output),
            ],
            cwd=self.root,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((output / "relocation-report.json").read_text())
        self.assertEqual(report["preserved_missing_rows"], 1)
        self.assertEqual(
            report["source_table_versions"],
            json.loads(inventory.read_text())["source_table_versions"],
        )
        rows = (
            lancedb.connect(str(output))
            .open_table("media_index_clip_ann")
            .to_arrow()
            .to_pylist()
        )
        self.assertIn(new_key, {row["file_name"] for row in rows})

    def test_recovered_alias_keeps_distinct_iris_row_and_vectors(self) -> None:
        new_key = self._preserve_unmapped(missing=True)
        destination = self.destination_root / "legacy-unreferenced/Legacy/unmapped.jpg"
        destination.parent.mkdir()
        destination.write_bytes(b"image-a")
        payload = json.loads(self.manifest.read_text())
        row = payload["missing_media"].pop()
        row.update(
            source=str(self.old_a),
            iris_source=str(self.source_root / "Legacy/unmapped.jpg"),
            status="copied",
            size=len(b"image-a"),
            sha256=self._digest(destination),
        )
        payload["missing"] = []
        payload["media"].append(row)
        self.manifest.write_text(json.dumps(payload))
        plan = load_relocation_plan(self.manifest, "telegram_backup")
        inventory = inspect_database(self.db_dir, "media_index", plan)
        self.assertEqual(inventory.collisions, [])
        output = self.root / "recovered-alias-output"
        apply_relocation(
            self.db_dir, output, "media_index", plan, inventory, drop_unmapped=False
        )
        db = lancedb.connect(str(output))
        rows = {
            row["file_name"]: row
            for row in db.open_table("media_index_clip_ann").to_arrow().to_pylist()
        }
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(rows[new_key]["vector"][0], 0.8, places=6)
        self.assertEqual(
            rows["telegram_backup/Chat One__user_1/media/photo/a.jpg"]["vector"],
            [0.25, 0.75],
        )

    def test_many_to_one_iris_keys_block_instead_of_losing_metadata(self) -> None:
        payload = json.loads(self.manifest.read_text())
        payload["media"][0]["source_aliases"] = [str(self.old_b)]
        self.manifest.write_text(json.dumps(payload))
        plan = load_relocation_plan(self.manifest, "telegram_backup")
        inventory = inspect_database(self.db_dir, "media_index", plan)
        self.assertTrue(inventory.collisions)
        output = self.root / "collision-output"
        with self.assertRaisesRegex(RelocationError, "collide"):
            apply_relocation(
                self.db_dir, output, "media_index", plan, inventory, drop_unmapped=False
            )
        self.assertFalse(output.exists())

    def _prepare_fanin(self, *, annotation_conflict=False, ann_conflict=False, preferred=True):
        self.old_b.write_bytes(b"image-a")
        payload = json.loads(self.manifest.read_text())
        payload["media"] = payload["media"][:2]
        payload["media"][0]["source_aliases"] = [str(self.old_b)]
        payload["media"][0]["primary"] = preferred
        self.manifest.write_text(json.dumps(payload))
        db = lancedb.connect(str(self.db_dir))
        table = db.open_table("media_index")
        schema = table.schema.append(pa.field("annotation", pa.string()))
        rows = table.to_arrow().to_pylist()
        first = dict(rows[0], annotation="primary annotation")
        second = dict(first, file_name=rows[1]["file_name"])
        if annotation_conflict:
            second["annotation"] = "different historical annotation"
        rows = [first, second, dict(rows[2], annotation=None)]
        main = db.create_table("media_index", rows, schema=schema, mode="overwrite")
        main.create_scalar_index("file_name", name="file_name_idx")
        ann = db.open_table("media_index_clip_ann")
        ann.add([{
            "id": "telegram_backup/Legacy/b.jpg|0.000",
            "file_name": "telegram_backup/Legacy/b.jpg",
            "timestamp_sec": 0.0,
            "vector": [0.75, 0.25] if ann_conflict else [0.25, 0.75],
        }])
        return rows, ann.to_arrow().to_pylist()

    def _apply_fanin(self, name="fanin-output"):
        plan = load_relocation_plan(self.manifest, "telegram_backup", coalesce_equivalent_sources=True)
        inventory = inspect_database(self.db_dir, "media_index", plan)
        self.assertEqual(len(inventory.coalescible_targets), 1)
        self.assertEqual(inventory.collisions, [])
        output = self.root / name
        report = apply_relocation(self.db_dir, output, "media_index", plan, inventory,
                                  drop_unmapped=False, coalesce_equivalent_sources=True, batch_size=2)
        return output, report, plan, inventory

    def test_opt_in_fanin_and_fanout_preserve_typed_originals(self):
        originals, original_ann = self._prepare_fanin()
        output, report, _, _ = self._apply_fanin()
        db = lancedb.connect(str(output))
        main = db.open_table("media_index").to_arrow().to_pylist()
        self.assertEqual(len(main), 3)
        self.assertEqual(len({row["file_name"] for row in main}), 3)
        self.assertEqual(report.tables["media_index"].coalesced_rows, 1)
        self.assertEqual(report.tables["media_index"].cloned_rows, 1)
        canonical = next(row for row in main if row["file_name"] ==
                         "telegram_backup/Chat One__user_1/media/photo/a.jpg")
        self.assertIsNone(canonical["dedupe_match_file"])
        self.assertIsNone(canonical["sift_match_file"])
        self.assertFalse(any(entry["file_name"] == canonical["file_name"]
                             for entry in canonical["cross_media_matches"]))
        observations = db.open_table("relocation_observations_media_index").to_arrow().to_pylist()
        self.assertEqual(len(observations), 3)
        self.assertEqual({row["source_file_name"] for row in observations}, {row["file_name"] for row in originals[:2]})
        original_by_key = {row["file_name"]: row for row in originals}
        for evidence in observations:
            self.assertEqual(evidence["original_row"], original_by_key[evidence["source_file_name"]])
        ann = db.open_table("media_index_clip_ann").to_arrow().to_pylist()
        self.assertEqual(len(ann), 2)
        self.assertEqual({tuple(row["vector"]) for row in ann}, {(0.25, 0.75)})
        evidence = db.open_table("relocation_observations_media_index_clip_ann").to_arrow().to_pylist()
        self.assertEqual(len(evidence), 3)
        self.assertEqual({row["source_id"] for row in evidence}, {row["id"] for row in original_ann})
        source = lancedb.connect(str(self.db_dir))
        self.assertEqual(source.open_table("media_index").to_arrow().to_pylist(), originals)
        self.assertEqual(source.open_table("media_index_clip_ann").to_arrow().to_pylist(), original_ann)

    def test_fanin_conflicting_annotations_require_explicit_primary_and_retain_variants(self):
        self._prepare_fanin(annotation_conflict=True)
        output, report, _, _ = self._apply_fanin()
        db = lancedb.connect(str(output))
        rows = db.open_table("media_index").to_arrow().to_pylist()
        self.assertEqual(next(row for row in rows if "Chat One" in row["file_name"])["annotation"], "primary annotation")
        evidence = db.open_table("relocation_observations_media_index").to_arrow().to_pylist()
        self.assertEqual({row["original_row"]["annotation"] for row in evidence}, {"primary annotation", "different historical annotation"})
        self.assertEqual(report.tables["media_index"].conflict_rows, 1)

    def test_repeated_primary_media_associations_share_one_explicit_iris_owner(self):
        self._prepare_fanin(annotation_conflict=True, ann_conflict=True)
        payload = json.loads(self.manifest.read_text())
        association = dict(payload["media"][0], source=str(self.old_b),
                           iris_source=str(self.old_a), source_aliases=[], primary=True)
        payload["media"].append(association)
        self.manifest.write_text(json.dumps(payload))
        output, report, _, _ = self._apply_fanin()
        db = lancedb.connect(str(output))
        key = "telegram_backup/Chat One__user_1/media/photo/a.jpg"
        current = next(row for row in db.open_table("media_index").to_arrow().to_pylist()
                       if row["file_name"] == key)
        self.assertEqual(current["annotation"], "primary annotation")
        observed = db.open_table("relocation_observations_media_index").to_arrow().to_pylist()
        self.assertEqual({row["original_row"]["annotation"] for row in observed},
                         {"primary annotation", "different historical annotation"})
        self.assertEqual(report.tables["media_index_clip_ann"].variant_rows, 2)

    def test_fanin_conflicting_metadata_without_primary_never_publishes(self):
        self._prepare_fanin(annotation_conflict=True, preferred=False)
        with self.assertRaisesRegex(RelocationError, "explicit manifest-primary"):
            self._apply_fanin()
        self.assertFalse((self.root / "fanin-output").exists())

    def test_fanin_ann_conflicts_remain_searchable_with_unique_variant_ids(self):
        _, originals = self._prepare_fanin(ann_conflict=True)
        output, report, _, _ = self._apply_fanin()
        table = lancedb.connect(str(output)).open_table("media_index_clip_ann")
        rows = table.to_arrow().to_pylist()
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({row["id"] for row in rows}), 3)
        shared = [row for row in rows if "Chat One" in row["file_name"]]
        self.assertEqual(len(shared), 2)
        self.assertTrue(all("|variant:" in row["id"] for row in shared))
        self.assertEqual({tuple(row["vector"]) for row in shared}, {(0.25, 0.75), (0.75, 0.25)})
        self.assertEqual(len(table.search([0.75, 0.25]).limit(10).to_list()), 3)
        self.assertEqual(report.tables["media_index_clip_ann"].variant_rows, 2)
        evidence = lancedb.connect(str(output)).open_table("relocation_observations_media_index_clip_ann").to_arrow().to_pylist()
        original_by_id = {row["id"]: row for row in originals}
        for row in evidence:
            self.assertEqual(row["original_row"], original_by_id[row["source_id"]])

    def test_fanin_alias_requires_actual_strong_hash_not_manifest_claim(self):
        self._prepare_fanin()
        self.old_b.write_bytes(b"image-b")
        with self.assertRaisesRegex(RelocationError, "SHA-256"):
            load_relocation_plan(self.manifest, "telegram_backup", coalesce_equivalent_sources=True)

    def test_fanin_source_change_after_proof_refuses_apply(self):
        self._prepare_fanin()
        plan = load_relocation_plan(self.manifest, "telegram_backup", coalesce_equivalent_sources=True)
        inventory = inspect_database(self.db_dir, "media_index", plan)
        self.old_b.write_bytes(b"image-b")
        with self.assertRaisesRegex(RelocationError, "source changed"):
            apply_relocation(self.db_dir, self.root / "changed-fanin", "media_index", plan, inventory,
                             drop_unmapped=False, coalesce_equivalent_sources=True)
        self.assertFalse((self.root / "changed-fanin").exists())

    def test_fanin_replay_refuses_existing_output(self):
        self._prepare_fanin()
        output, _, plan, inventory = self._apply_fanin()
        with self.assertRaisesRegex(RelocationError, "already exists"):
            apply_relocation(self.db_dir, output, "media_index", plan, inventory,
                             drop_unmapped=False, coalesce_equivalent_sources=True)

    def test_fanin_failure_keeps_source_and_never_publishes(self):
        originals, _ = self._prepare_fanin()
        with patch("relocate_collection._check_source_versions", side_effect=RelocationError("injected version drift")), self.assertRaisesRegex(RelocationError, "injected version drift"):
            self._apply_fanin()
        self.assertFalse((self.root / "fanin-output").exists())
        self.assertEqual(lancedb.connect(str(self.db_dir)).open_table("media_index").to_arrow().to_pylist(), originals)

    def test_chained_relocation_preserves_immutable_old_generation_events(self):
        self._prepare_fanin()
        first, _, _, _ = self._apply_fanin()
        first_db = lancedb.connect(str(first))
        original_events = first_db.open_table("relocation_paths").to_arrow().to_pylist()
        inventory_file = self.root / "chained-active-inventory.json"
        export_collection_inventory(first,"media_index","telegram_backup",inventory_file)
        inventoried = {row['file_name'] for row in json.loads(inventory_file.read_text())['files']}
        self.assertEqual(inventoried, {row['file_name'] for row in first_db.open_table('media_index').to_arrow().to_pylist()
                                      if row['collection_id']=='telegram_backup'})
        second_root = self.root / "third-media"
        second_chat = second_root / "Merged/media"
        second_chat.mkdir(parents=True)
        image = second_chat / "a.jpg"
        image.write_bytes(b"image-a")
        second_manifest = self.root / "second-generation.json"
        payload = {"version": 1, "db_committed": True, "source_root": str(self.destination_root),
                   "destination_root": str(second_root), "chats": [{"chat_id": "merged", "destination": str(second_root / "Merged")}],
                   "media": [{"chat_id": "merged", "source": str(self.new_a_one), "source_aliases": [str(self.new_a_two)],
                              "relative": "media/a.jpg", "size": 7, "sha256": self._digest(image), "status": "copied", "primary": True}]}
        second_manifest.write_text(json.dumps(payload))
        plan = load_relocation_plan(second_manifest, "telegram_backup", coalesce_equivalent_sources=True)
        inventory = inspect_database(first, "media_index", plan)
        output = self.root / "second-generation-lance"
        report = apply_relocation(first, output, "media_index", plan, inventory,
                                  drop_unmapped=False, coalesce_equivalent_sources=True)
        events = lancedb.connect(str(output)).open_table("relocation_paths").to_arrow().to_pylist()
        historical = [row for row in events if row["manifest_sha256"] != plan.manifest_sha256]
        self.assertEqual(historical, original_events)
        self.assertEqual(report.tables["relocation_paths"].source_rows, len(original_events))
        self.assertEqual(report.tables["relocation_paths"].added_rows, 2)
        evidence = lancedb.connect(str(output)).open_table("relocation_observations_media_index").to_arrow().to_pylist()
        self.assertEqual(len(evidence), 5)

    def test_fanin_compatible_null_fields_fill_without_discarding_original(self):
        self._prepare_fanin()
        db = lancedb.connect(str(self.db_dir))
        table = db.open_table("media_index")
        rows = table.to_arrow().to_pylist()
        rows[0]["annotation"] = None
        rows[1]["annotation"] = "recovered missing annotation"
        db.create_table("media_index", rows, schema=table.schema, mode="overwrite")
        output, _, _, _ = self._apply_fanin()
        main = lancedb.connect(str(output)).open_table("media_index").to_arrow().to_pylist()
        self.assertEqual(next(row for row in main if "Chat One" in row["file_name"])["annotation"], "recovered missing annotation")
        evidence = lancedb.connect(str(output)).open_table("relocation_observations_media_index").to_arrow().to_pylist()
        self.assertIn(None, {row["original_row"]["annotation"] for row in evidence})

    def test_fanin_absent_physical_alias_never_coalesces(self):
        self._prepare_fanin()
        self.old_b.unlink()
        with self.assertRaisesRegex(RelocationError, "absent or inaccessible"):
            load_relocation_plan(self.manifest, "telegram_backup", coalesce_equivalent_sources=True)

    def test_optimized_cli_fanin_requires_explicit_policy_and_keeps_observations(self):
        self._prepare_fanin(ann_conflict=True)
        script = Path(__file__).resolve().parents[1] / "relocate_collection.py"
        output = self.root / "optimized fanin output"
        base = [sys.executable, "-O", str(script), "--db-dir", str(self.db_dir),
                "--collection-id", "telegram_backup", "--manifest", str(self.manifest),
                "--apply", "--output-db-dir", str(output), "--show-sample", "0"]
        blocked = subprocess.run(base, cwd=self.root, capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(blocked.returncode, 2)
        self.assertFalse(output.exists())
        accepted = subprocess.run([*base, "--coalesce-equivalent-sources"], cwd=self.root,
                                  capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        report = json.loads((output / "relocation-report.json").read_text())
        self.assertTrue(report["coalesce_equivalent_sources"])
        self.assertEqual(report["tables"]["media_index_clip_ann"]["variant_rows"], 2)
        self.assertEqual(report["tables"]["relocation_observations_media_index"]["added_rows"], 3)

    def test_fanin_same_inode_rewrite_with_restored_mtime_invalidates_proof(self):
        self._prepare_fanin()
        plan = load_relocation_plan(self.manifest, "telegram_backup", coalesce_equivalent_sources=True)
        inventory = inspect_database(self.db_dir, "media_index", plan)
        before = self.old_b.stat()
        self.old_b.write_bytes(b"image-z")
        os.utime(self.old_b, ns=(before.st_atime_ns,before.st_mtime_ns))
        self.assertEqual(self.old_b.stat().st_ino, before.st_ino)
        self.assertEqual(self.old_b.stat().st_mtime_ns, before.st_mtime_ns)
        output = self.root / "same-inode-changed-source"
        with self.assertRaisesRegex(RelocationError, "source changed"):
            apply_relocation(self.db_dir,output,"media_index",plan,inventory,
                             drop_unmapped=False,coalesce_equivalent_sources=True)
        self.assertFalse(output.exists())

    def test_destination_change_after_plan_load_refuses_before_output(self):
        plan = load_relocation_plan(self.manifest, "telegram_backup")
        inventory = inspect_database(self.db_dir, "media_index", plan)
        before = self.new_a_one.stat()
        self.new_a_one.write_bytes(b"image-z")
        os.utime(self.new_a_one, ns=(before.st_atime_ns,before.st_mtime_ns))
        output = self.root / "changed-destination-before-apply"
        with self.assertRaisesRegex(RelocationError, "destination.*changed"):
            apply_relocation(self.db_dir,output,"media_index",plan,inventory,drop_unmapped=False)
        self.assertFalse(output.exists())

    def test_destination_change_during_build_never_publishes(self):
        plan = load_relocation_plan(self.manifest, "telegram_backup")
        inventory = inspect_database(self.db_dir, "media_index", plan)
        output = self.root / "changed-destination-during-build"
        import relocate_collection as module
        original_validation = module._validate_output

        def mutate_after_validation(*args, **kwargs):
            original_validation(*args, **kwargs)
            before = self.new_a_one.stat()
            self.new_a_one.write_bytes(b"image-z")
            os.utime(self.new_a_one, ns=(before.st_atime_ns,before.st_mtime_ns))

        with patch("relocate_collection._validate_output", side_effect=mutate_after_validation), self.assertRaisesRegex(
            RelocationError, "destination.*changed"
        ):
            apply_relocation(self.db_dir,output,"media_index",plan,inventory,drop_unmapped=False)
        self.assertFalse(output.exists())
        self.assertTrue(list(self.root.glob(".changed-destination-during-build.partial-*")))

    def test_destination_root_identity_replaced_during_build_never_publishes(self):
        plan = load_relocation_plan(self.manifest, "telegram_backup")
        inventory = inspect_database(self.db_dir, "media_index", plan)
        output = self.root / "replaced-media-root"
        import relocate_collection as module
        original_validation = module._validate_output

        def replace_after_validation(*args, **kwargs):
            original_validation(*args, **kwargs)
            self.destination_root.rename(self.root / "detached-original-media")
            self.destination_root.mkdir()

        with patch("relocate_collection._validate_output", side_effect=replace_after_validation), self.assertRaisesRegex(
            RelocationError, "root.*changed"
        ):
            apply_relocation(self.db_dir,output,"media_index",plan,inventory,drop_unmapped=False)
        self.assertFalse(output.exists())

    def test_ann_only_keys_are_inventoried_and_require_explicit_preservation(self):
        key = "telegram_backup/Legacy/ann-only.jpg"
        db = lancedb.connect(str(self.db_dir))
        table = db.open_table("media_index_clip_ann")
        table.add([{"id":key+"|0.000","file_name":key,"timestamp_sec":0.0,"vector":[0.3,0.7]}])
        inventory_path = self.root / "all-active-inventory.json"
        export_collection_inventory(self.db_dir,"media_index","telegram_backup",inventory_path)
        inventory_payload = json.loads(inventory_path.read_text())
        self.assertIn(key, {row['file_name'] for row in inventory_payload['files']})
        plan = load_relocation_plan(self.manifest, "telegram_backup")
        inventory = inspect_database(self.db_dir, "media_index", plan)
        self.assertIn(key, inventory.unmapped_rows)
        output = self.root / "ann-only-unmapped"
        with self.assertRaisesRegex(RelocationError, "unmapped"):
            apply_relocation(self.db_dir,output,"media_index",plan,inventory,drop_unmapped=False)
        self.assertFalse(output.exists())

        payload = json.loads(self.manifest.read_text())
        payload['preserve_missing'] = True
        payload['missing'] = ['synthetic missing ANN-only source']
        payload['missing_media'] = [{"chat_id":"one","message_id":0,"source":str(self.source_root/'Legacy/ann-only.jpg'),
                                     "relative":"media/photo/ann-only.jpg","status":"missing","reason":"unresolved", "size":None,"sha256":None}]
        self.manifest.write_text(json.dumps(payload))
        preserved_plan = load_relocation_plan(self.manifest, "telegram_backup")
        preserved_inventory = inspect_database(self.db_dir, "media_index", preserved_plan)
        preserved_output = self.root / "ann-only-preserved"
        apply_relocation(self.db_dir,preserved_output,"media_index",preserved_plan,preserved_inventory,drop_unmapped=False)
        rows = lancedb.connect(str(preserved_output)).open_table('media_index_clip_ann').to_arrow().to_pylist()
        new_key = 'telegram_backup/Chat One__user_1/media/photo/ann-only.jpg'
        retained = next(row for row in rows if row['file_name']==new_key)
        self.assertEqual(retained['id'],new_key+'|0.000')
        self.assertAlmostEqual(retained['vector'][0],0.3,places=6)
        self.assertFalse((self.chat_one/'media/photo/ann-only.jpg').exists())

    def test_ann_only_existing_source_and_arbitrary_side_table_are_preserved(self):
        key = 'telegram_backup/Legacy/ann-only-existing.jpg'
        source = self.source_root / 'Legacy/ann-only-existing.jpg'
        source.write_bytes(b'image-c')
        destination = self.chat_one / 'media/photo/ann-only-existing.jpg'
        destination.write_bytes(b'image-c')
        db = lancedb.connect(str(self.db_dir))
        db.open_table('media_index_clip_ann').add([{'id':key+'|0.000','file_name':key,'timestamp_sec':0.0,'vector':[0.3,0.7]}])
        db.create_table('custom_side', [{'file_name':key, 'metadata':'synthetic side evidence'}])
        output_inventory = self.root / 'existing-side-inventory.json'
        export_collection_inventory(self.db_dir,'media_index','telegram_backup',output_inventory)
        payload = json.loads(output_inventory.read_text())
        self.assertEqual(payload['side_only_keys'],1)
        self.assertTrue(next(row for row in payload['files'] if row['file_name']==key)['exists'])
        manifest = json.loads(self.manifest.read_text())
        manifest['media'].append({'chat_id':'one','source':str(source),'relative':'media/photo/ann-only-existing.jpg',
                                  'size':7,'sha256':self._digest(destination),'status':'copied'})
        self.manifest.write_text(json.dumps(manifest))
        plan = load_relocation_plan(self.manifest,'telegram_backup')
        inventory = inspect_database(self.db_dir,'media_index',plan)
        self.assertEqual(inventory.side_only_source_keys,[key])
        output = self.root / 'existing-side-preserved'
        apply_relocation(self.db_dir,output,'media_index',plan,inventory,drop_unmapped=False)
        copied = lancedb.connect(str(output))
        new_key = 'telegram_backup/Chat One__user_1/media/photo/ann-only-existing.jpg'
        self.assertEqual(copied.open_table('custom_side').to_arrow().to_pylist(),
                         [{'file_name':new_key,'metadata':'synthetic side evidence'}])
        found = next(row for row in copied.open_table('media_index_clip_ann').to_arrow().to_pylist() if row['file_name']==new_key)
        self.assertAlmostEqual(found['vector'][0],0.3,places=6)
        self.assertEqual(copied.open_table('media_index').count_rows(),4)

    def test_output_validation_refuses_unknown_scoped_side_path(self):
        plan = load_relocation_plan(self.manifest,'telegram_backup')
        inventory = inspect_database(self.db_dir,'media_index',plan)
        output = self.root / 'uncovered-scoped-side-path'
        import relocate_collection as module
        original_transform = module._transform_path_row

        def invalid_scoped_path(*args, **kwargs):
            rows, dropped = original_transform(*args, **kwargs)
            if rows:
                rows[0]['file_name'] = 'telegram_backup/not-covered-by-manifest.jpg'
            return rows, dropped

        with patch('relocate_collection._transform_path_row',side_effect=invalid_scoped_path), self.assertRaisesRegex(
            RelocationError,'uncovered scoped path'
        ):
            apply_relocation(self.db_dir,output,'media_index',plan,inventory,drop_unmapped=False)
        self.assertFalse(output.exists())

    def test_destination_change_inside_hash_verification_invalidates_digest(self):
        import relocate_collection as module
        original_hash = module._sha256_file

        def change_after_read(path):
            value = original_hash(path)
            if path==self.new_a_one:
                before = path.stat()
                path.write_bytes(b'image-z')
                os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
            return value

        with patch('relocate_collection._sha256_file',side_effect=change_after_read), self.assertRaisesRegex(
            RelocationError,'destination changed during SHA-256'
        ):
            load_relocation_plan(self.manifest,'telegram_backup')

    def test_changed_manifest_never_publishes_output(self) -> None:
        plan = load_relocation_plan(self.manifest, "telegram_backup")
        inventory = inspect_database(self.db_dir, "media_index", plan)
        output = self.root / "changed-manifest-output"
        with (
            patch(
                "relocate_collection._read_json_object", return_value=({}, "changed")
            ),
            self.assertRaisesRegex(RelocationError, "manifest changed"),
        ):
            apply_relocation(
                self.db_dir, output, "media_index", plan, inventory, drop_unmapped=False
            )
        self.assertFalse(output.exists())

    def test_explicit_null_optional_iris_source_uses_physical_source_key(self) -> None:
        payload = json.loads(self.manifest.read_text())
        for row in payload["media"]:
            row["iris_source"] = None
        self.manifest.write_text(json.dumps(payload))
        plan = load_relocation_plan(self.manifest, "telegram_backup")
        inventory = inspect_database(self.db_dir, "media_index", plan)
        self.assertEqual(inventory.mapped_rows, 2)
        self.assertEqual(inventory.unmapped_rows, [])

    def test_explicit_purge_excludes_only_exact_rows_and_ann_references(self) -> None:
        payload = json.loads(self.manifest.read_text())
        payload["preserve_missing"] = True
        payload["media"] = [
            row for row in payload["media"] if row["source"] != str(self.old_b)
        ]
        payload["excluded_sources"] = [
            {
                "source": str(self.old_b),
                "reason": "previously purged",
                "purge_key": "synthetic",
            }
        ]
        self.manifest.write_text(json.dumps(payload))
        # Identical bytes do not authorize removing the retained path's row.
        self.old_b.write_bytes(b"image-a")
        source_db = lancedb.connect(str(self.db_dir))
        old_b_key = "telegram_backup/Legacy/b.jpg"
        source_db.open_table("media_index_clip_ann").add(
            [
                {
                    "id": old_b_key + "|0.000",
                    "file_name": old_b_key,
                    "timestamp_sec": 0.0,
                    "vector": [0.5, 0.5],
                }
            ]
        )
        plan = load_relocation_plan(self.manifest, "telegram_backup")
        inventory = inspect_database(self.db_dir, "media_index", plan)
        self.assertEqual(inventory.excluded_rows, [old_b_key])
        self.assertEqual(inventory.unmapped_rows, [])
        output = self.root / "explicit-purge-output"
        report = apply_relocation(
            self.db_dir, output, "media_index", plan, inventory, drop_unmapped=False
        )
        self.assertEqual(report.excluded_rows, 1)
        self.assertEqual(report.tables["media_index"].dropped_rows, 1)
        db = lancedb.connect(str(output))
        rows = db.open_table("media_index").to_arrow().to_pylist()
        a_one = "telegram_backup/Chat One__user_1/media/photo/a.jpg"
        a_two = "telegram_backup/Chat Two__user_2/media/photo/a.jpg"
        self.assertEqual(
            {row["file_name"] for row in rows}, {a_one, a_two, "other/reference.jpg"}
        )
        self.assertTrue(all(row["dedupe_match_file"] != old_b_key for row in rows))
        self.assertEqual(
            next(row for row in rows if row["file_name"] == a_one)[
                "cross_media_matches"
            ],
            [],
        )
        self.assertEqual(
            {
                row["file_name"]
                for row in db.open_table("media_index_clip_ann").to_arrow().to_pylist()
            },
            {a_one, a_two},
        )
        audit = [
            row
            for row in db.open_table("relocation_paths").to_arrow().to_pylist()
            if row["status"] == "excluded"
        ]
        self.assertEqual(audit[0]["old_file_name"], old_b_key)
        self.assertIsNone(audit[0]["file_name"])
        self.assertEqual(source_db.open_table("media_index").count_rows(), 3)
        self.assertEqual(source_db.open_table("media_index_clip_ann").count_rows(), 2)

    def _assert_payload_prefilter(self, *, coalesced: bool, explicit: bool,
                                 bypass_filter: bool = False) -> None:
        """A decoder trap proves forbidden rows are removed before Python decoding."""
        if coalesced:
            self._prepare_fanin()
        source = lancedb.connect(str(self.db_dir))
        forbidden_key = "telegram_backup/Legacy/p's-private.jpg"
        main = source.open_table("media_index")
        original = main.to_arrow().to_pylist()[0]
        original.update(file_name=forbidden_key, dedupe_match_file=None, sift_match_file=None,
                        cross_media_matches=[])
        main.add(pa.Table.from_pylist([original], schema=main.schema))
        ann = source.open_table("media_index_clip_ann")
        ann.add(pa.Table.from_pylist([{"id": forbidden_key+"|0.000", "file_name": forbidden_key,
                                     "timestamp_sec": 0.0, "vector": [0.5, 0.5]}], schema=ann.schema))
        if explicit:
            payload = json.loads(self.manifest.read_text())
            payload["excluded_sources"] = [{"source": str(self.source_root / "Legacy/p's-private.jpg"),
                                            "reason": "previously purged"}]
            self.manifest.write_text(json.dumps(payload))
        plan = load_relocation_plan(self.manifest, "telegram_backup", coalesce_equivalent_sources=coalesced)
        inventory = inspect_database(self.db_dir, "media_index", plan)
        before = {name: source.open_table(name).version for name in _table_names(source)}
        output = self.root / ("prefilter-coalesced" if coalesced else "prefilter-direct")
        real_query = relocate_collection._active_payload_query
        guarded_tables = set()

        class GuardedBatch:
            def __init__(inner, batch):
                inner.batch = batch

            def __getattr__(inner, name):
                return getattr(inner.batch, name)

            def to_pylist(inner):
                keys = inner.batch.select(["file_name"]).to_pylist()
                if any(row["file_name"] == forbidden_key for row in keys):
                    raise AssertionError("forbidden payload was decoded before metadata-only filtering")
                return inner.batch.to_pylist()

        class GuardedQuery:
            def __init__(inner, query):
                inner.query = query

            def to_batches(inner, **kwargs):
                for batch in inner.query.to_batches(**kwargs):
                    yield GuardedBatch(batch)

        def guarded_query(table, excluded):
            query, count = real_query(table, excluded)
            if bypass_filter:
                query = table.search(None)
            guarded_tables.add(table.name)
            return GuardedQuery(query), count

        with patch("relocate_collection._active_payload_query", side_effect=guarded_query):
            report = apply_relocation(self.db_dir, output, "media_index", plan, inventory,
                                      drop_unmapped=not explicit, coalesce_equivalent_sources=coalesced,
                                      batch_size=2)
        self.assertIn("media_index", guarded_tables)
        self.assertIn("media_index_clip_ann", guarded_tables)
        self.assertEqual(report.tables["media_index"].dropped_rows, 1)
        self.assertEqual(report.tables["media_index_clip_ann"].dropped_rows, 1)
        for name in ("media_index", "media_index_clip_ann"):
            keys = {row["file_name"] for row in lancedb.connect(str(output)).open_table(name)
                    .search(None).select(["file_name"]).to_list()}
            self.assertNotIn(forbidden_key, keys)
        self.assertIn("other/reference.jpg", {row["file_name"] for row in
                      lancedb.connect(str(output)).open_table("media_index")
                      .search(None).select(["file_name"]).to_list()})
        self.assertEqual(before, {name: source.open_table(name).version for name in _table_names(source)})

    def test_direct_exclusions_are_filtered_before_payload_decode(self) -> None:
        self._assert_payload_prefilter(coalesced=False, explicit=True)

    def test_coalesced_exclusions_are_filtered_before_payload_decode(self) -> None:
        self._assert_payload_prefilter(coalesced=True, explicit=True)

    def test_direct_unmapped_drops_are_filtered_before_payload_decode(self) -> None:
        self._assert_payload_prefilter(coalesced=False, explicit=False)

    def test_coalesced_unmapped_drops_are_filtered_before_payload_decode(self) -> None:
        self._assert_payload_prefilter(coalesced=True, explicit=False)

    def test_payload_privacy_decoder_trap_rejects_disabled_filter(self) -> None:
        with self.assertRaisesRegex(AssertionError, "forbidden payload was decoded"):
            self._assert_payload_prefilter(coalesced=True, explicit=True, bypass_filter=True)

    def test_observation_validation_filters_current_generation_before_decode(self) -> None:
        self._prepare_fanin()
        prior, _, _, _ = self._apply_fanin("prior-observation-generation")
        prior_table = lancedb.connect(str(prior)).open_table("relocation_observations_media_index")
        older = prior_table.to_arrow().to_pylist()[:1]
        older[0]["generation_id"] = "synthetic-older-generation"
        source = lancedb.connect(str(self.db_dir))
        source.create_table(prior_table.name, data=pa.Table.from_pylist(older, schema=prior_table.schema))
        original_iter = relocate_collection._iter_rows

        def only_current_observations(table, columns=None, batch_size=2048, *, where=None):
            if table.name.startswith("relocation_observations_") and columns is None:
                self.assertIsNotNone(where, "historical observations reached full Python decoding")
                self.assertTrue(where.startswith("generation_id = '"))
            yield from original_iter(table, columns, batch_size, where=where)

        with patch("relocate_collection._iter_rows", side_effect=only_current_observations):
            output, _, _, _ = self._apply_fanin("current-generation-only")
        preserved = lancedb.connect(str(output)).open_table(prior_table.name)
        self.assertEqual(preserved.search(None).where("generation_id = 'synthetic-older-generation'").to_list(), older)

    def test_immutable_history_is_copied_without_python_payload_decode(self) -> None:
        source = lancedb.connect(str(self.db_dir))
        schema = pa.schema([pa.field("generation_id", pa.string()),
                            pa.field("original_row", pa.struct([pa.field("annotation", pa.string())]))])
        historical = source.create_table("relocation_observations_synthetic", data=pa.Table.from_pylist(
            [{"generation_id": "old", "original_row": {"annotation": "synthetic private observation"}}],
            schema=schema))
        plan = load_relocation_plan(self.manifest, "telegram_backup")
        destination = lancedb.connect(str(self.root / "opaque-history"))
        real_arrow = relocate_collection.pa

        class GuardedBatch:
            def __init__(inner, batch):
                inner.batch = batch

            def __getattr__(inner, name):
                return getattr(inner.batch, name)

            def to_pylist(inner):
                raise AssertionError("immutable historical payload was decoded in Python")

        class GuardedSource:
            def __getattr__(inner, name):
                return getattr(historical, name)

            def search(inner, _value):
                class Query:
                    def to_batches(query, **kwargs):
                        for batch in historical.search(None).to_batches(**kwargs):
                            yield GuardedBatch(batch)
                return Query()

        class ArrowTable:
            @staticmethod
            def from_batches(batches, schema):
                return real_arrow.Table.from_batches([batch.batch for batch in batches], schema=schema)

        class ArrowProxy:
            Table = ArrowTable

            def __getattr__(inner, name):
                return getattr(real_arrow, name)

        with patch("relocate_collection.pa", ArrowProxy()):
            result, _ = relocate_collection._copy_table(GuardedSource(), destination, historical.name,
                plan, "media_index", set(), False, 2)
        self.assertEqual(result.source_rows, 1)
        self.assertEqual(result.output_rows, 1)
        self.assertEqual(result.dropped_rows, 0)
        self.assertEqual(historical.to_arrow(), destination.open_table(historical.name).to_arrow())
        self.assertTrue(self.old_b.is_file())

    def test_purge_cannot_conflict_with_preservation_or_use_blacklist_reason(
        self,
    ) -> None:
        payload = json.loads(self.manifest.read_text())
        payload["excluded_sources"] = [
            {"source": str(self.old_a), "reason": "previously purged"}
        ]
        self.manifest.write_text(json.dumps(payload))
        with self.assertRaisesRegex(RelocationError, "simultaneously preserved"):
            load_relocation_plan(self.manifest, "telegram_backup")
        payload["excluded_sources"] = [
            {
                "source": str(self.source_root / "Legacy/unmapped.jpg"),
                "reason": "blacklisted",
            }
        ]
        self.manifest.write_text(json.dumps(payload))
        with self.assertRaisesRegex(RelocationError, "explicit purge"):
            load_relocation_plan(self.manifest, "telegram_backup")


if __name__ == "__main__":
    unittest.main()
