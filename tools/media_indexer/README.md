# Media Indexer

Incremental GPU pipeline for media files:
1. VideoHash is computed for videos and compared only against other videos
2. If video hash similarity is `>= 80%`, the video is marked `skip_processing=true` and later stages are skipped
3. PySceneDetect extracts still images for remaining videos
4. Video stills are pruned to a maximum of 100 images per video
5. Stills are stored under `<input-folder-name>-video` in the parent of the scanned folder
6. Image pHash pre-gate runs on photos only and compares against other photos (not videos)
7. If image pHash similarity is `>= 85%`, the image is marked `skip_processing=true` and later stages are skipped
8. InsightFace face detection on GPU
9. ArcFace face embeddings on GPU
10. CLIP `ViT-L-16-SigLIP2-384` embeddings on GPU
11. PaddleOCR text detection on GPU
12. EasyOCR text extraction on GPU for frames where text is detected
13. ANN indexes are maintained in LanceDB side tables for fast similarity search:
   - `<table>_face_ann`
   - `<table>_clip_ann`
   - `<table>_ocr_ann`

Results are stored in LanceDB with one record per file:
- `file_name`
- `collection_id`
- `is_video`
- `faces` (y/n)
- `phash_hex`
- `skip_processing`
- `dedupe_match_file`
- `dedupe_similarity_pct`
- `sift_match_file`
- `sift_match_score`
- `sift_match_inliers`
- `sift_match_good_matches`
- `sift_match_inlier_ratio`
- `sift_match_checked`
- `processing_error_stage`
- `processing_error`
- `face_groups` (timestamped face vectors)
- `clip_groups` (timestamped clip vectors)
- cached source metadata (`source_size`, `source_mtime_ns`, `image_width`, and `image_height`)
- `ocr_groups` (timestamped OCR detect/text)

## Run

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --db-dir ./lancedb
```

## Text Model Export

Iris uses the exported CLIP text tower in `models/clip-text` for in-app CLIP text search. Rebuild it with:

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run python tools/export_clip_text_onnx.py --out-dir models/clip-text
```

## GPU Paddle Runtime

`paddleocr` does not install the PaddlePaddle runtime itself. This project pins `paddlepaddle-gpu` to Paddle's CUDA 13 index, so `uv run`/`uv sync` installs it automatically. To force installation before a long run:

```bash
UV_CACHE_DIR="/data/.cache/uv" uv sync
```

Verify it:

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run python -c "import paddle; print(paddle.__version__); print(paddle.device.cuda.device_count())"
```

Add multiple folders into the same DB by using different collection ids:

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages "/path/to/photo-archive" --db-dir ./lancedb --collection-id photo_archive
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages "/path/to/screenshots" --db-dir ./lancedb --collection-id screenshots
```

Notes:
- If `--collection-id` is omitted, it defaults to `<folder-name>@<path-hash>`.
- `file_name` is stored as `<collection_id>/<relative/path/to/file>`.
- Dedupe matching still works across the whole DB, including across collections.

Optional pHash skip threshold:

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --phash-skip-similarity-pct 85
```

Optional hash worker count for CPU pHash/VideoHash:

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --hash-workers 16
```

Optional InsightFace model cache:

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --insightface-root /data/.cache/insightface
```

Optional video hash skip threshold:

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --video-hash-skip-similarity-pct 80
```

Optional SIFT master-match annotation stage (uses CLIP shortlist + SIFT verify):

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run python main.py /path/to/media \
  --run-sift-master-match \
  --sift-candidate-topk 64 \
  --sift-min-ratio 0.75 \
  --sift-min-inliers 14 \
  --sift-contrast-threshold 0.03
```

Existing SIFT checks, including checks produced by the earlier DINO shortlist pipeline, count as
complete. Process only unchecked masters with `--run-sift-master-match`, or explicitly recompute
all eligible masters with the current CLIP shortlist:

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --rerun-stage 5
```

Optional OCR ANN text model:

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --ocr-text-model sentence-transformers/all-MiniLM-L6-v2
```

Optional PaddleOCR input cap for very large screenshots:

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --paddle-ocr-max-side 2048
```

Stage 7 uses Paddle's detection-only model for text y/n detection. It defaults to the faster mobile detector:

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --paddle-det-model PP-OCRv5_mobile_det
```

Force Stage 7 text y/n detection to rerun without redoing faces or CLIP:

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --rerun-paddle-ocr
```

Any completed stage can be rerun with the repeatable `--rerun-stage` option. A numbered parent
includes all of its lettered substages:

```bash
# Rerun image metadata, image/still pHashes, grouping, and cross-media matching.
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --rerun-stage 3

# Rerun only image pHash grouping and EasyOCR extraction.
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --rerun-stage 3d --rerun-stage 8

# Rerun every stage.
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --rerun-stage all
```

Speed/quality knobs for Stage 8 EasyOCR extraction:

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --easyocr-max-side 1600 --easyocr-canvas-size 1600 --easyocr-batch-size 8
```

If Paddle's CUDA runtime conflicts with PyTorch CUDA, keep Paddle text detection on CPU while EasyOCR still uses CUDA:

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --paddle-device cpu --easyocr-device cuda --ocr-text-device cuda
```

## Resume Behavior

The tool only processes unfinished stages:
- Stages with no pending work print a completed count instead of silently disappearing from the run output
- Interactive terminal output colors stage numbers yellow, stage names cyan, completed states green, warnings/skips/incomplete states orange, and errors red; colors are disabled when output is redirected or `NO_COLOR` is set
- Stage 0 is startup: filesystem scan, LanceDB open/schema checks, DB row loading, DB row normalization, and legacy key migration checks
- Stage 3a caches image dimensions and file sizes in the database, refreshing them only when source size or modification time changes
- Stages 1a/1b compute and apply VideoHash results, marking near-duplicates as `skip_processing=true`
- Stage 2 skips duplicate videos and reuses valid still manifests when the source video and scene settings are unchanged
- Stage 3b computes missing image pHashes
- Stage 3c computes missing or changed extracted-video-still pHashes
- Stage 3d applies image pHash grouping results
- Stages 3e/3f maintain bidirectional image-to-video-frame relationships
- Stages 3e/3f are skipped when their saved completion fingerprint still matches the file scope, image pHashes, video-still pHashes, and similarity threshold
- Stage 4 computes missing CLIP embeddings for images and video stills, then updates the shared CLIP ANN table
- Stages 5a/5b shortlist the top 64 CLIP image-master candidates and verify them with SIFT/RANSAC
- Existing `sift_match_checked=true` rows are treated as complete regardless of whether DINO or CLIP originally produced them
- `--run-sift-master-match` processes unchecked SIFT masters; `--rerun-stage 5` explicitly recomputes completed SIFT checks with CLIP
- Stage 6 runs face detection where `face_groups` are missing or outdated
- Stage 7 runs PaddleOCR where `ocr_groups` are missing or outdated
- Stage 8 runs EasyOCR where an `ocr_groups` entry has `text_detected=true` and `text=null`
- Stage 9 is search-index sync: it maintains Face, CLIP, and OCR text ANN side tables used by Iris search
- Stage 9 prints which side tables are missing; missing/recreated search-index tables are filled directly instead of deleting one row per file first
- Existing search-index side tables are updated only for rows changed by the processing stages
- Search-index side tables are incrementally updated for processed files and indexed with IVF-HNSW-SQ
- Database writes are batched during hash, face, CLIP, PaddleOCR, and EasyOCR stages to reduce LanceDB write overhead
