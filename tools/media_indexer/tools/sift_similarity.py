#!/usr/bin/env python3
import json
import hashlib
import math
import sys
from pathlib import Path

import cv2
import numpy as np


def out(payload: dict, code: int = 0) -> None:
    print(json.dumps(payload, ensure_ascii=True))
    raise SystemExit(code)


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"failed to read image: {path}")
    return img


def load_color_resized(path: Path, max_side: int = 1920) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to read image: {path}")
    height, width = img.shape[:2]
    longest = max(height, width)
    if longest > max_side:
        scale = max_side / float(longest)
        img = cv2.resize(
            img,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return img


def match_sift_images(
    img_a: np.ndarray,
    img_b: np.ndarray,
    *,
    return_homography: bool = False,
) -> tuple[dict, np.ndarray | None]:
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("OpenCV SIFT is unavailable in this environment")

    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=4000)
    kp_a, des_a = sift.detectAndCompute(gray_a, None)
    kp_b, des_b = sift.detectAndCompute(gray_b, None)

    keypoints_a = int(len(kp_a) if kp_a is not None else 0)
    keypoints_b = int(len(kp_b) if kp_b is not None else 0)
    if des_a is None or des_b is None or keypoints_a == 0 or keypoints_b == 0:
        return (
            {
                "keypoints_a": keypoints_a,
                "keypoints_b": keypoints_b,
                "good_matches": 0,
                "inlier_matches": 0,
                "inlier_ratio": 0.0,
                "score": 0.0,
            },
            None,
        )

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(des_a, des_b, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    inlier_matches = 0
    homography = None
    if len(good) >= 4:
        # The normal SIFT report maps A -> B. For display alignment we also
        # return the inverse-direction matrix that maps B onto A.
        src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        homography_a_to_b, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if mask is not None:
            inlier_matches = int(mask.ravel().sum())
        if return_homography and homography_a_to_b is not None:
            try:
                homography = np.linalg.inv(homography_a_to_b)
            except np.linalg.LinAlgError:
                homography = None

    good_matches = int(len(good))
    inlier_ratio = float(inlier_matches / good_matches) if good_matches > 0 else 0.0
    denom = float(min(keypoints_a, keypoints_b))
    score = float(inlier_matches / denom) if denom > 0 else 0.0
    if math.isnan(score) or math.isinf(score):
        score = 0.0
    if math.isnan(inlier_ratio) or math.isinf(inlier_ratio):
        inlier_ratio = 0.0

    return (
        {
            "keypoints_a": keypoints_a,
            "keypoints_b": keypoints_b,
            "good_matches": good_matches,
            "inlier_matches": inlier_matches,
            "inlier_ratio": inlier_ratio,
            "score": score,
        },
        homography,
    )


def run_sift(path_a: Path, path_b: Path) -> dict:
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("OpenCV SIFT is unavailable in this environment")

    img_a = load_gray(path_a)
    img_b = load_gray(path_b)

    sift = cv2.SIFT_create(nfeatures=4000)
    kp_a, des_a = sift.detectAndCompute(img_a, None)
    kp_b, des_b = sift.detectAndCompute(img_b, None)

    if des_a is None or des_b is None or len(kp_a) == 0 or len(kp_b) == 0:
        return {
            "keypoints_a": int(len(kp_a) if kp_a is not None else 0),
            "keypoints_b": int(len(kp_b) if kp_b is not None else 0),
            "good_matches": 0,
            "inlier_matches": 0,
            "inlier_ratio": 0.0,
            "score": 0.0,
        }

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(des_a, des_b, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    inlier_matches = 0
    if len(good) >= 4:
        src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if mask is not None:
            inlier_matches = int(mask.ravel().sum())

    good_matches = int(len(good))
    inlier_ratio = float(inlier_matches / good_matches) if good_matches > 0 else 0.0
    denom = float(min(len(kp_a), len(kp_b)))
    score = float(inlier_matches / denom) if denom > 0 else 0.0

    if math.isnan(score) or math.isinf(score):
        score = 0.0
    if math.isnan(inlier_ratio) or math.isinf(inlier_ratio):
        inlier_ratio = 0.0

    return {
        "keypoints_a": int(len(kp_a)),
        "keypoints_b": int(len(kp_b)),
        "good_matches": good_matches,
        "inlier_matches": inlier_matches,
        "inlier_ratio": inlier_ratio,
        "score": score,
    }


def align_images(reference: Path, candidates: list[Path], output_dir: Path) -> dict:
    reference_img = load_color_resized(reference)
    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = reference_img.shape[:2]
    results = []

    for candidate in candidates:
        item = {"path": str(candidate.resolve())}
        try:
            candidate_img = load_color_resized(candidate)
            metrics, homography = match_sift_images(
                reference_img,
                candidate_img,
                return_homography=True,
            )
            item.update(metrics)
        except Exception as exc:  # noqa: BLE001
            item["error"] = str(exc)
            results.append(item)
            continue

        if homography is None or metrics["inlier_matches"] < 4:
            item["error"] = "not enough geometrically consistent SIFT matches"
            results.append(item)
            continue

        digest = hashlib.sha256(str(candidate.resolve()).encode("utf-8")).hexdigest()[:16]
        aligned_path = output_dir / f"aligned_{digest}.png"
        try:
            aligned = cv2.warpPerspective(
                candidate_img,
                homography,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(24, 24, 24),
            )
            if not cv2.imwrite(str(aligned_path), aligned):
                item["error"] = f"failed to write aligned image: {aligned_path}"
            else:
                item["aligned_path"] = str(aligned_path.resolve())
        except Exception as exc:  # noqa: BLE001
            item["error"] = str(exc)
        results.append(item)

    return {
        "reference": str(reference.resolve()),
        "results": results,
    }


def main() -> int:
    if len(sys.argv) >= 5 and sys.argv[1] == "--align-all":
        reference = Path(sys.argv[2])
        output_dir = Path(sys.argv[3])
        candidates = [Path(value) for value in sys.argv[4:]]
        try:
            result = align_images(reference, candidates, output_dir)
        except Exception as exc:  # noqa: BLE001
            out({"error": str(exc)}, 1)
        out(result, 0)

    if len(sys.argv) != 3:
        out({"error": "usage: sift_similarity.py <image_a> <image_b>"}, 2)

    path_a = Path(sys.argv[1])
    path_b = Path(sys.argv[2])

    try:
        result = run_sift(path_a, path_b)
    except Exception as exc:  # noqa: BLE001
        out({"error": str(exc)}, 1)

    out(result, 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
