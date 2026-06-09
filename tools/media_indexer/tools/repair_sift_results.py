#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import lancedb
import numpy as np
import pyarrow as pa
from PIL import Image
from tqdm import tqdm


# Lowe descriptor-ratio filter, not the final "percent similar" threshold.
MIN_RATIO = 0.75
MIN_INLIERS = 10
# Final geometric acceptance: RANSAC inliers / Lowe-filtered matches.
MIN_INLIER_RATIO = 0.75
MIN_SCORE = 0.0
SIFT_CONTRAST_THRESHOLD = 0.03
SIFT_MAX_SIDE = 1920


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair SIFT grouping for a bounded set of viewer results."
    )
    parser.add_argument("--db-dir", required=True, type=Path)
    parser.add_argument("--table", default="media_index")
    parser.add_argument("--files-json", required=True, type=Path)
    parser.add_argument(
        "--collection-root",
        action="append",
        default=[],
        metavar="COLLECTION_ID=/ABS/PATH",
    )
    parser.add_argument("--min-ratio", type=float, default=MIN_RATIO)
    parser.add_argument("--min-inliers", type=int, default=MIN_INLIERS)
    parser.add_argument("--min-inlier-ratio", type=float, default=MIN_INLIER_RATIO)
    parser.add_argument("--min-score", type=float, default=MIN_SCORE)
    parser.add_argument("--contrast-threshold", type=float, default=SIFT_CONTRAST_THRESHOLD)
    parser.add_argument("--max-side", type=int, default=SIFT_MAX_SIDE)
    parser.add_argument(
        "--fast-pair",
        action="store_true",
        help="For exactly two images, avoid full-table component repair and update only the lower-ranked image.",
    )
    return parser.parse_args()


def parse_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        collection, sep, root = value.partition("=")
        if not sep or not collection or not root:
            raise SystemExit(f"invalid --collection-root: {value}")
        roots[collection] = Path(root)
    return roots


def resolve_path(roots: dict[str, Path], file_name: str) -> Path | None:
    collection, sep, rel = file_name.partition("/")
    if not sep:
        return None
    root = roots.get(collection)
    if root is None:
        return None
    return root / rel


def is_image_row(row: dict[str, Any]) -> bool:
    if row.get("is_video") is True:
        return False
    if row.get("skip_processing") is True:
        return False
    return True


def read_gray_resized(path: Path, max_side: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"failed to decode image: {path}")
    height, width = img.shape[:2]
    longest = max(width, height)
    if longest > max_side > 0:
        scale = max_side / float(longest)
        img = cv2.resize(
            img,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return img


def extract_features(sift: Any, path: Path, max_side: int) -> tuple[Any, Any, int]:
    gray = read_gray_resized(path, max_side)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors, int(len(keypoints) if keypoints is not None else 0)


def sift_metrics(
    features_a: tuple[Any, Any, int],
    features_b: tuple[Any, Any, int],
    min_ratio: float,
    min_inliers: int,
    min_inlier_ratio: float,
    min_score: float,
) -> dict[str, Any]:
    keypoints_a, descriptors_a, kp_count_a = features_a
    keypoints_b, descriptors_b, kp_count_b = features_b
    if descriptors_a is None or descriptors_b is None or kp_count_a == 0 or kp_count_b == 0:
        return {"accepted": False, "good_matches": 0, "inliers": 0, "inlier_ratio": 0.0, "score": 0.0}

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good_matches = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < min_ratio * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return {
            "accepted": False,
            "good_matches": int(len(good_matches)),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "score": 0.0,
        }

    src = np.float32([keypoints_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst = np.float32([keypoints_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    inliers = int(mask.ravel().sum()) if mask is not None else 0
    good_count = int(len(good_matches))
    inlier_ratio = float(inliers / good_count) if good_count > 0 else 0.0
    score = float(inliers / max(1, min(kp_count_a, kp_count_b)))
    if math.isnan(score) or math.isinf(score):
        score = 0.0
    accepted = (
        inliers >= min_inliers
        and inlier_ratio >= min_inlier_ratio
        and score >= min_score
    )
    return {
        "accepted": accepted,
        "good_matches": good_count,
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "score": score,
    }


def best_sift_metrics(
    features_a: tuple[Any, Any, int],
    features_b: tuple[Any, Any, int],
    min_ratio: float,
    min_inliers: int,
    min_inlier_ratio: float,
    min_score: float,
) -> dict[str, Any]:
    forward = sift_metrics(
        features_a,
        features_b,
        min_ratio,
        min_inliers,
        min_inlier_ratio,
        min_score,
    )
    reverse = sift_metrics(
        features_b,
        features_a,
        min_ratio,
        min_inliers,
        min_inlier_ratio,
        min_score,
    )
    return max(
        (forward, reverse),
        key=lambda metrics: (
            bool(metrics["accepted"]),
            int(metrics["inliers"]),
            float(metrics["inlier_ratio"]),
            float(metrics["score"]),
        ),
    )


def image_dimensions(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as img:
            width, height = img.size
            return int(width), int(height)
    except Exception:
        return 0, 0


def file_rank(file_name: str, path: Path) -> tuple[int, int, int, int, str]:
    width, height = image_dimensions(path)
    max_side = max(width, height)
    area = width * height
    try:
        file_size = path.stat().st_size
    except OSError:
        file_size = 0
    lower = path.as_posix().lower()
    non_thumb = 0 if "thumb" in lower else 1
    return max_side, area, file_size, non_thumb, file_name


class Dsu:
    def __init__(self, names: list[str]) -> None:
        self.parent = {name: name for name in names}

    def find(self, name: str) -> str:
        root = name
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[name] != name:
            nxt = self.parent[name]
            self.parent[name] = root
            name = nxt
        return root

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def valid_existing_sift_link(
    row: dict[str, Any],
    rows_by_name: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> str | None:
    file_name = row.get("file_name")
    target = row.get("sift_match_file")
    if not isinstance(file_name, str) or not isinstance(target, str):
        return None
    if target == file_name or target not in rows_by_name:
        return None
    if not is_image_row(row) or not is_image_row(rows_by_name[target]):
        return None
    if row.get("sift_match_checked") is not True:
        return None
    if int(row.get("sift_match_inliers") or 0) < int(args.min_inliers):
        return None
    if float(row.get("sift_match_inlier_ratio") or 0.0) < float(args.min_inlier_ratio):
        return None
    if float(row.get("sift_match_score") or 0.0) < float(args.min_score):
        return None
    return target


def escape_sql(value: str) -> str:
    return value.replace("'", "''")


def fetch_rows_by_name(table: Any, file_names: list[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for file_name in file_names:
        found = (
            table.search()
            .where(f"file_name = '{escape_sql(file_name)}'")
            .limit(1)
            .to_list()
        )
        if found:
            rows[file_name] = found[0]
    return rows


def write_updates(table: Any, schema: pa.Schema, updates: list[dict[str, Any]]) -> None:
    if not updates:
        return
    arrow_table = pa.Table.from_pylist(updates, schema=schema)
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


def run_fast_pair_repair(
    args: argparse.Namespace,
    roots: dict[str, Path],
    table: Any,
    requested_files: list[str],
) -> int:
    file_names = []
    seen = set()
    for file_name in requested_files:
        if isinstance(file_name, str) and file_name not in seen:
            seen.add(file_name)
            file_names.append(file_name)
    if len(file_names) != 2:
        raise SystemExit("--fast-pair requires exactly two distinct file names")

    rows_by_name = fetch_rows_by_name(table, file_names)
    items: list[tuple[str, Path, dict[str, Any]]] = []
    for file_name in file_names:
        row = rows_by_name.get(file_name)
        if row is None or not is_image_row(row):
            continue
        path = resolve_path(roots, file_name)
        if path is None or not path.exists():
            continue
        items.append((file_name, path, row))

    if len(items) != 2:
        print(json.dumps({"images": len(items), "pairs": 0, "accepted_pairs": 0, "updated": 0}))
        return 0
    if not hasattr(cv2, "SIFT_create"):
        raise SystemExit("OpenCV SIFT is unavailable")

    sift = cv2.SIFT_create(contrastThreshold=args.contrast_threshold)
    features_a = extract_features(sift, items[0][1], args.max_side)
    features_b = extract_features(sift, items[1][1], args.max_side)
    metrics = best_sift_metrics(
        features_a,
        features_b,
        args.min_ratio,
        args.min_inliers,
        args.min_inlier_ratio,
        args.min_score,
    )
    if not metrics["accepted"]:
        print(
            json.dumps(
                {
                    "images": 2,
                    "pairs": 1,
                    "accepted_pairs": 0,
                    "linked_images": 0,
                    "updated": 0,
                    "inliers": int(metrics["inliers"]),
                    "good_matches": int(metrics["good_matches"]),
                    "inlier_ratio": float(metrics["inlier_ratio"]),
                    "score": float(metrics["score"]),
                },
                ensure_ascii=True,
            )
        )
        return 0

    rank_a = file_rank(items[0][0], items[0][1])
    rank_b = file_rank(items[1][0], items[1][1])
    if rank_a >= rank_b:
        master_name, child = items[0][0], items[1]
    else:
        master_name, child = items[1][0], items[0]

    updated = dict(child[2])
    updated["sift_match_file"] = master_name
    updated["sift_match_score"] = round(float(metrics["score"]), 6)
    updated["sift_match_inliers"] = int(metrics["inliers"])
    updated["sift_match_good_matches"] = int(metrics["good_matches"])
    updated["sift_match_inlier_ratio"] = round(float(metrics["inlier_ratio"]), 6)
    updated["sift_match_checked"] = True
    updates = [updated] if updated != child[2] else []
    write_updates(table, table.schema, updates)
    print(
        json.dumps(
            {
                "images": 2,
                "pairs": 1,
                "accepted_pairs": 1,
                "linked_images": 1,
                "updated": len(updates),
                "master": master_name,
                "child": child[0],
                "inliers": int(metrics["inliers"]),
                "good_matches": int(metrics["good_matches"]),
                "inlier_ratio": float(metrics["inlier_ratio"]),
                "score": float(metrics["score"]),
            },
            ensure_ascii=True,
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    roots = parse_roots(args.collection_root)
    requested_files = json.loads(args.files_json.read_text(encoding="utf-8"))
    if not isinstance(requested_files, list):
        raise SystemExit("--files-json must contain a JSON list")

    db = lancedb.connect(str(args.db_dir))
    table = db.open_table(args.table)
    if args.fast_pair:
        return run_fast_pair_repair(args, roots, table, requested_files)

    requested_unique: list[str] = []
    seen: set[str] = set()
    for file_name in requested_files:
        if not isinstance(file_name, str) or file_name in seen:
            continue
        seen.add(file_name)
        requested_unique.append(file_name)
    rows_by_name = fetch_rows_by_name(table, requested_unique)

    items: list[tuple[str, Path, dict[str, Any]]] = []
    for file_name in requested_unique:
        row = rows_by_name.get(file_name)
        if row is None or not is_image_row(row):
            continue
        path = resolve_path(roots, file_name)
        if path is None or not path.exists():
            continue
        items.append((file_name, path, row))

    if len(items) < 2:
        print(json.dumps({"images": len(items), "pairs": 0, "accepted_pairs": 0, "updated": 0}))
        return 0

    if not hasattr(cv2, "SIFT_create"):
        raise SystemExit("OpenCV SIFT is unavailable")

    sift = cv2.SIFT_create(contrastThreshold=args.contrast_threshold)
    features: dict[str, tuple[Any, Any, int]] = {}
    for file_name, path, _ in tqdm(items, desc="extract SIFT", unit="image"):
        try:
            features[file_name] = extract_features(sift, path, args.max_side)
        except Exception as exc:
            print(f"[repair-sift] feature failed: {file_name}: {exc}")

    pair_metrics: dict[tuple[str, str], dict[str, Any]] = {}
    accepted_edges: list[tuple[str, str, dict[str, Any]]] = []
    pairs = 0
    accepted_pairs = 0
    for i in tqdm(range(len(items)), desc="pairwise SIFT", unit="image"):
        file_a = items[i][0]
        feat_a = features.get(file_a)
        if feat_a is None:
            continue
        for j in range(i + 1, len(items)):
            file_b = items[j][0]
            feat_b = features.get(file_b)
            if feat_b is None:
                continue
            pairs += 1
            metrics = best_sift_metrics(
                feat_a,
                feat_b,
                args.min_ratio,
                args.min_inliers,
                args.min_inlier_ratio,
                args.min_score,
            )
            if not metrics["accepted"]:
                continue
            accepted_pairs += 1
            pair_metrics[(file_a, file_b)] = metrics
            pair_metrics[(file_b, file_a)] = metrics
            accepted_edges.append((file_a, file_b, metrics))

    image_names = [
        name
        for name, row in rows_by_name.items()
        if isinstance(name, str) and is_image_row(row)
    ]
    existing_dsu = Dsu(image_names)
    valid_direct_target: dict[str, str] = {}
    for name in image_names:
        target = valid_existing_sift_link(rows_by_name[name], rows_by_name, args)
        if target is None:
            continue
        valid_direct_target[name] = target
        existing_dsu.union(name, target)

    dedupe_children_by_master: dict[str, int] = {}
    for row in rows_by_name.values():
        master = row.get("dedupe_match_file")
        if isinstance(master, str):
            dedupe_children_by_master[master] = dedupe_children_by_master.get(master, 0) + 1

    rank_cache: dict[str, tuple[int, int, int, int, str]] = {}

    def rank_for_name(file_name: str) -> tuple[int, int, int, int, str]:
        cached = rank_cache.get(file_name)
        if cached is not None:
            return cached
        path = resolve_path(roots, file_name)
        if path is None:
            rank = (0, 0, 0, 0, file_name)
        else:
            rank = file_rank(file_name, path)
        rank_cache[file_name] = rank
        return rank

    existing_component_members: dict[str, list[str]] = {}
    for name in image_names:
        existing_component_members.setdefault(existing_dsu.find(name), []).append(name)

    def existing_component_weight(root: str) -> int:
        members = existing_component_members.get(root, [root])
        return len(members) + sum(dedupe_children_by_master.get(member, 0) for member in members)

    def existing_component_best_rank(root: str) -> tuple[int, int, int, int, str]:
        return max((rank_for_name(member) for member in existing_component_members.get(root, [root])), default=(0, 0, 0, 0, root))

    def existing_component_sort_key(root: str) -> tuple[int, int, int, int, int, str]:
        rank = existing_component_best_rank(root)
        return (existing_component_weight(root), *rank)

    def existing_component_anchor(root: str) -> str:
        members = existing_component_members.get(root, [root])
        root_like = [member for member in members if member not in valid_direct_target]
        candidates = root_like or members
        return max(candidates, key=rank_for_name)

    merge_dsu = Dsu(image_names)
    for child, target in valid_direct_target.items():
        merge_dsu.union(child, target)
    for file_a, file_b, _ in accepted_edges:
        merge_dsu.union(file_a, file_b)

    merged_component_members: dict[str, list[str]] = {}
    for name in image_names:
        merged_component_members.setdefault(merge_dsu.find(name), []).append(name)

    direct_master_by_file: dict[str, str] = {}
    direct_metrics_by_file: dict[str, dict[str, Any]] = {}
    for merged_members in merged_component_members.values():
        existing_roots = sorted({existing_dsu.find(member) for member in merged_members})
        if len(existing_roots) <= 1:
            continue

        target_root = max(existing_roots, key=existing_component_sort_key)
        connected_roots = {target_root}
        while len(connected_roots) < len(existing_roots):
            best_bridge: tuple[tuple[int, float, float, str, str, str], str, str, str, dict[str, Any]] | None = None
            for file_a, file_b, metrics in accepted_edges:
                root_a = existing_dsu.find(file_a)
                root_b = existing_dsu.find(file_b)
                if root_a == root_b:
                    continue
                a_connected = root_a in connected_roots
                b_connected = root_b in connected_roots
                if a_connected == b_connected:
                    continue
                connected_root = root_a if a_connected else root_b
                child_root = root_b if a_connected else root_a
                if child_root not in existing_roots or connected_root not in existing_roots:
                    continue
                child = existing_component_anchor(child_root)
                master = existing_component_anchor(connected_root)
                bridge_key = (
                    int(metrics["inliers"]),
                    float(metrics["inlier_ratio"]),
                    float(metrics["score"]),
                    child_root,
                    child,
                    master,
                )
                candidate = (bridge_key, child_root, child, master, metrics)
                if best_bridge is None or candidate > best_bridge:
                    best_bridge = candidate
            if best_bridge is None:
                break
            _, child_root, child, master, metrics = best_bridge
            if child != master:
                direct_master_by_file[child] = master
                direct_metrics_by_file[child] = metrics
            connected_roots.add(child_root)

    schema = table.schema
    updates: list[dict[str, Any]] = []
    for file_name in sorted(direct_master_by_file):
        row = rows_by_name.get(file_name)
        if row is None:
            continue
        updated = dict(row)
        master = direct_master_by_file.get(file_name)
        metrics = direct_metrics_by_file[file_name]
        updated["sift_match_file"] = master
        updated["sift_match_score"] = round(float(metrics["score"]), 6)
        updated["sift_match_inliers"] = int(metrics["inliers"])
        updated["sift_match_good_matches"] = int(metrics["good_matches"])
        updated["sift_match_inlier_ratio"] = round(float(metrics["inlier_ratio"]), 6)
        updated["sift_match_checked"] = True
        if updated != row:
            updates.append(updated)

    write_updates(table, schema, updates)

    print(
        json.dumps(
            {
                "images": len(items),
                "pairs": pairs,
                "accepted_pairs": accepted_pairs,
                "linked_images": len(direct_master_by_file),
                "updated": len(updates),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
