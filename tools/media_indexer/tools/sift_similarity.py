#!/usr/bin/env python3
import json
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


def main() -> int:
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
