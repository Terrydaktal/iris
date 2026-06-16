# Media Indexer

Incremental media indexing pipeline. CPU stages scan files, calculate hashes, and verify
SIFT matches. GPU stages create CLIP/SigLIP2, face, and OCR embeddings when their
configured runtimes support CUDA.

## Stage Implementation Reference

| Stage | Purpose | Model, algorithm, or tool |
| --- | --- | --- |
| `0a` | Scan media files | Python `Path.rglob` filesystem traversal and supported-extension filtering |
| `0b` | Split images and videos | Extension-based classification |
| `0c` | Open/check the database | LanceDB with a PyArrow schema; creates or migrates `media_index` |
| `0d` | Ensure the file-name index | LanceDB scalar index on `file_name` |
| `0e` | Load database rows | LanceDB/Apache Arrow table read |
| `0f` | Normalize rows | Python schema normalization into the in-memory record map |
| `0g` | Check legacy keys | Filesystem-to-database key comparison |
| `0h` | Migrate legacy keys | LanceDB row delete/upsert migration |
| `1a` | Hash videos | Iris custom VideoHash: OpenCV `VideoCapture`, up to 16 evenly spaced frames, a custom 64-bit DCT pHash per frame, then per-bit majority vote |
| `1b` | Apply video duplicate groups | Hamming-distance BK-tree; default duplicate threshold `80%` |
| `2` | Extract video stills | PySceneDetect `ContentDetector` using the OpenCV backend; defaults: threshold `27`, minimum scene length `15`, scene-midpoint JPEGs, maximum `100` stills per video |
| `3a` | Cache image metadata | Filesystem stat plus Pillow dimensions |
| `3b` | Hash images | Iris custom OpenCV 64-bit DCT pHash: grayscale, `32x32`, DCT, median threshold over the low-frequency `8x8` block |
| `3c` | Hash video stills | The same custom 64-bit DCT pHash used by Stage `3b` |
| `3d` | Apply image pHash groups | Hamming-distance BK-tree; default threshold `95%`; prefers the highest-resolution image as group master |
| `3e` | Match images to video frames | BK-tree comparison of image pHashes against extracted-video-still pHashes; default threshold `95%` |
| `3f` | Match video frames to images | Reverse BK-tree comparison using the same pHashes and cross-media threshold |
| `4a` | Embed images and video stills | OpenCLIP `model.encode_image` on CUDA; default model `hf-hub:timm/ViT-L-16-SigLIP2-384`; output vectors are mean-pooled when needed and L2-normalized; default batch size `8` |
| `4b` | Build/delete CLIP/SigLIP2 ANN rows | LanceDB side-table sync for changed rows, or a full rebuild when the ANN table is missing/recreated |
| `4c` | Build/finalize the CLIP/SigLIP2 ANN index | LanceDB `IVF_HNSW_SQ` vector index with cosine distance |
| `5a` | Shortlist SIFT candidates | LanceDB cosine search over the Stage `4b` SigLIP2 index; default top `64` eligible image masters |
| `5b` | Verify and group SIFT matches | OpenCV SIFT, L2 brute-force KNN matcher, Lowe ratio test, and homography RANSAC; defaults: ratio `0.75`, minimum `10` inliers, minimum inlier ratio `0.75` |
| `6a` | Detect and embed faces | InsightFace `buffalo_l` detection/recognition through ONNX Runtime CUDA; normalized 512-dimensional ArcFace-compatible vectors, with rotation and larger-detector fallback |
| `6b` | Build/delete face ANN rows | LanceDB side-table sync for changed rows, or a full rebuild when the ANN table is missing/recreated |
| `6c` | Build/finalize the face ANN index | LanceDB `IVF_HNSW_SQ` vector index with cosine distance |
| `7` | Detect whether text exists | PaddleOCR `TextDetection`; default model `PP-OCRv5_mobile_det`, default device `gpu:0` |
| `8a` | Extract detected text | EasyOCR; default language `en`, default device CUDA, with reduced-GPU then per-frame CPU fallback after CUDA failure |
| `8b` | Build/delete OCR ANN rows | SentenceTransformers `sentence-transformers/all-MiniLM-L6-v2` on CUDA by default; LanceDB side-table sync for changed text rows, or a full rebuild when the ANN table is missing/recreated |
| `8c` | Build/finalize the OCR ANN index | LanceDB `IVF_HNSW_SQ` vector index with cosine distance |

The application calls Stage `4a` and its search mode "CLIP," but the default image
embedder is specifically SigLIP2 loaded through `open_clip.create_model_and_transforms`.
Iris text-to-image search uses an ONNX export of the same model's text tower from
`models/clip-text`. On-demand image and face embedding imports the same OpenCLIP and
InsightFace implementations from this directory.

The searchable vector indexes are maintained in LanceDB side tables:

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

Iris uses an exported ONNX text tower from the default
`hf-hub:timm/ViT-L-16-SigLIP2-384` model in `models/clip-text` for in-app CLIP text
search. Rebuild it with:

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
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --phash-skip-similarity-pct 95
```

Image-to-video-frame relationships use a separate threshold, which also defaults to `95%`:

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --cross-media-similarity-pct 95
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
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --rerun-stage 3d --rerun-stage 8a

# Rebuild only the OCR text search side table from current DB rows.
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --rerun-stage 8b

# Rerun every stage.
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --rerun-stage all
```

Speed/quality knobs for Stage 8a EasyOCR extraction:

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
- Stage 4a computes missing CLIP embeddings for images and video stills
- Stage 4b syncs the CLIP ANN side table rows from changed records, or from all records when the table is missing/recreated or `--rerun-stage 4b` is used
- Stage 4c builds or refreshes the CLIP ANN vector index after Stage 4b changes the side table
- Stages 5a/5b shortlist the top 64 CLIP image-master candidates and verify them with SIFT/RANSAC
- Existing `sift_match_checked=true` rows are treated as complete regardless of whether DINO or CLIP originally produced them
- `--run-sift-master-match` processes unchecked SIFT masters; `--rerun-stage 5` explicitly recomputes completed SIFT checks with CLIP
- Stage 6a runs face detection where `face_groups` are missing or outdated
- Stage 6b syncs the face ANN side table rows from changed records, or from all records when the table is missing/recreated or `--rerun-stage 6b` is used
- Stage 6c builds or refreshes the face ANN vector index after Stage 6b changes the side table
- Stage 7 runs PaddleOCR where `ocr_groups` are missing or outdated
- Stage 8a runs EasyOCR where an `ocr_groups` entry has `text_detected=true` and `text=null`
- Stage 8b syncs the OCR text ANN side table rows from PaddleOCR/EasyOCR changes, or from all rows when the table is missing/recreated or `--rerun-stage 8b` is used
- Stage 8c builds or refreshes the OCR ANN vector index after Stage 8b changes the side table
- Missing CLIP, face, and OCR ANN side tables are rebuilt inside their own `4b`/`4c`, `6b`/`6c`, and `8b`/`8c` stages instead of a separate final verification stage
- Existing search-index side tables are updated only for rows changed by the processing stages
- Search-index side tables are incrementally updated for processed files and indexed with IVF-HNSW-SQ
- Database writes are batched during hash, face, CLIP, PaddleOCR, and EasyOCR stages to reduce LanceDB write overhead
