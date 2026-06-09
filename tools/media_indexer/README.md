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
  --sift-min-ratio 0.75 \
  --sift-min-inliers 14 \
  --sift-contrast-threshold 0.03
```

Optional OCR ANN text model:

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --ocr-text-model sentence-transformers/all-MiniLM-L6-v2
```

Optional PaddleOCR input cap for very large screenshots:

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --paddle-ocr-max-side 2048
```

Stage 3 uses Paddle's detection-only model for text y/n detection. It defaults to the faster mobile detector:

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --paddle-det-model PP-OCRv5_mobile_det
```

Force Stage 3 text y/n detection to rerun without redoing faces or CLIP:

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --rerun-paddle-ocr
```

Speed/quality knobs for Stage 4 EasyOCR extraction:

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --easyocr-max-side 1600 --easyocr-canvas-size 1600 --easyocr-batch-size 8
```

If Paddle's CUDA runtime conflicts with PyTorch CUDA, keep Paddle text detection on CPU while EasyOCR still uses CUDA:

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --paddle-device cpu --easyocr-device cuda --ocr-text-device cuda
```

## Resume Behavior

The tool only processes unfinished stages:
- Stage 0.0 computes VideoHash for videos and marks near-duplicates as `skip_processing=true`
- Stage 0.0 stores `processing_error_stage` and `processing_error` for videos that cannot be opened/decoded for hashing, then skips them on later runs
- Stage 0.1 skips PySceneDetect for videos that are already skipped or already have embeddings/OCR groups in DB
- Stage 0.1 reuses existing extracted stills when source video + scene settings are unchanged
- Stage 0.5 computes pHash for images and marks near-duplicates as `skip_processing=true`
- Faces stage runs where `face_groups` are missing/outdated
- CLIP stage runs where `clip_groups` are missing/outdated
- PaddleOCR stage runs where `ocr_groups` are missing/outdated
- EasyOCR stage runs where an `ocr_groups` entry has `text_detected=true` and `text=null`
- ANN side tables are incrementally updated for processed files and indexed with IVF-HNSW-SQ
- Database writes are batched during hash, face, CLIP, PaddleOCR, and EasyOCR stages to reduce LanceDB write overhead
