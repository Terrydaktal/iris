from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import lancedb
import pyarrow as pa

from relocate_collection import (
    RelocationError,
    apply_relocation,
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


if __name__ == "__main__":
    unittest.main()
