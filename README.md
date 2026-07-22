# Iris

Iris is a local media browser for large photo and video archives. It can browse folders directly, inspect metadata, open videos in `mpv`, show duplicate/similar media relationships, and search an indexed LanceDB database by filename, CLIP similarity, or OCR text.

The app is intentionally local-first. Media files, embeddings, OCR output, duplicate links, and collection roots live in a local LanceDB directory. No personal collection paths should be hardcoded in the source tree.

## Project Structure

```text
.
├── Cargo.toml
├── Cargo.lock
├── README.md
├── iris.desktop
├── icon.png
├── make_transparent.py
├── src
│   ├── main.rs
│   └── bin
│       └── make_transparent.rs
└── tools
    ├── on_demand_embeddings.py
    └── media_indexer
        ├── README.md
        ├── main.py
        ├── easyocr_worker.py
        ├── face_worker.py
        ├── paddle_worker.py
        ├── pyproject.toml
        ├── uv.lock
        ├── models/clip-text   # generated locally, ignored by Git
        └── tools
```

Generated local data is ignored by Git, including `target/`, `lancedb/`, Python virtualenv/cache folders, downloaded/exported indexer models, reports, and extracted `*-video/` still folders.

## Main Components

### Iris Viewer: `src/main.rs`

The Rust desktop application. It provides:

- Folder and file gallery browsing for images and videos.
- Filename, CLIP, and OCR search modes from one search UI.
- Optional folder-scoped CLIP/OCR search across indexed collections.
- Image viewing, video handoff to `mpv`, and folder opening.
- Comparison mode for two to six selected files, with independent zoom and pan state for each file, plus on-demand SIFT alignment to the first image.
- EXIF/raw metadata via `exiftool` and video stream/format metadata via `ffprobe`.
- Duplicate sidebar for SIFT groups, pHash/VideoHash similar files, and image-video cross-media matches.
- Right-click actions for showing visually similar files and more of the same person.
- Ctrl-click multi-selection for SIFT compare/repair tooling.
- Embedded crop/rotate editor for image edits.
- On-demand CLIP/face embedding for files that have not already been indexed.

### Media Indexer: `tools/media_indexer/main.py`

The Python indexing pipeline that builds and updates the LanceDB database. It handles:

- Collection-aware media indexing using `<collection_id>/<relative/path>` file keys.
- Collection root mapping through the `collection_roots` table.
- A custom OpenCV video hash built from 16 sampled 64-bit DCT pHashes.
- PySceneDetect `ContentDetector` still extraction through the OpenCV backend.
- Custom OpenCV 64-bit DCT pHash generation for images and video stills.
- Cross-media image-to-video and video-to-image relationships when an image matches a video still.
- OpenCLIP image embeddings using `hf-hub:timm/ViT-L-16-SigLIP2-384` by default.
- OpenCV SIFT duplicate grouping using SigLIP2 ANN candidates.
- InsightFace `buffalo_l` face detection and ArcFace-compatible 512-dimensional embeddings.
- PaddleOCR `PP-OCRv5_mobile_det` text detection and EasyOCR text extraction.
- LanceDB `IVF_HNSW_SQ` cosine indexes for face, CLIP/SigLIP2, and OCR text search.

See [tools/media_indexer/README.md](tools/media_indexer/README.md) for the full indexing pipeline and stage details.

### On-Demand Embeddings: `tools/on_demand_embeddings.py`

Used by Iris when a file needs CLIP or face vectors but does not already have them in the database.

Example output:

```json
{"ok": true, "clip_embedding": [0.1, 0.2], "face_embeddings": [[0.1, 0.2]]}
```

### Icon Helpers

- `make_transparent.py`: Python icon cleanup tool. It removes edge-connected backgrounds and writes a transparent PNG.
- `src/bin/make_transparent.rs`: Rust icon utility kept for the Cargo workspace.
- `icon.png`: app icon used by the desktop entry.
- `iris.desktop`: Linux desktop entry. It uses `Icon=iris` and `StartupWMClass=iris`; the app sets the matching native app id.

## Database Location

Iris chooses the database directory in this order:

1. `IRIS_DB_DIR`, if set.
2. The repo-local `lancedb` directory, normally `~/Dev/iris/lancedb` when running from this checkout.
3. Existing discovered `lancedb` directories on common mounted locations.
4. `${XDG_DATA_HOME}/iris/lancedb`, `${HOME}/.local/share/iris/lancedb`, or `./lancedb` as fallbacks.

For this repo, the expected default is the repo-local `lancedb/`. That directory is ignored by Git because it contains generated database tables and extracted still metadata.

## Runtime Configuration

Environment variables:

- `IRIS_DB_DIR`: override LanceDB directory.
- `IRIS_MEDIA_INDEXER_DIR`: override the integrated media indexer/runtime helper directory. Defaults to `tools/media_indexer`.
- `IRIS_ON_DEMAND_EMBED_SCRIPT`: override the on-demand embedding helper path.
- `IRIS_EXIFTOOL`: explicit `exiftool` path.
- `IRIS_FFPROBE`: explicit `ffprobe` path.

External commands used by the viewer:

- `exiftool` for image metadata.
- `ffprobe` for video metadata.
- `mpv` for video playback.
- `dolphin` if available for opening folders, with fallback openers otherwise.

## Build And Run Iris

```bash
cargo run --release -- --no-daemon
```

Open a file or folder:

```bash
cargo run --release -- --no-daemon /path/to/file-or-folder
```

Open two to six files in comparison mode. Use the left and right arrow keys to switch between them; Iris restores each file's zoom and pan position independently:

```bash
cargo run --release -- --no-daemon /path/to/first.jpg /path/to/second.jpg /path/to/third.jpg
```

The normal `Ctrl+O` picker opens one file. To enter paths inside Iris, use `Compare Paths` or `Ctrl+Shift+O` and enter one path per line. Select one file with `Ctrl+O` for normal single-file viewing. In comparison mode, `SIFT Align All` uses the first image as the reference, computes missing pairwise SIFT homographies in a background worker, and displays aligned temporary copies without changing the source files. It is intended for still images; files that cannot produce a reliable homography remain unwarped and are reported in the status.

Reuse an existing window:

```bash
cargo run --release -- --same-window /path/to/file-or-folder
```

Install the desktop file/icon in the normal Linux locations if you want launcher/taskbar integration. The repo contains the desktop file template, but the actual icon lookup depends on `iris` being installed into the local icon theme or system icon theme.

## Build Or Update The Media Database

Use `uv` from the media indexer directory:

```bash
cd tools/media_indexer
UV_CACHE_DIR="/data/.cache/uv" uv sync
```

Export the CLIP text model used by Iris text search if it is not already present:

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run python tools/export_clip_text_onnx.py --out-dir models/clip-text
```

Index or incrementally update a collection:

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media \
  --db-dir ../../lancedb \
  --collection-id my_collection
```

Use stable collection ids. Iris resolves database paths such as `my_collection/sub/folder/file.jpg` through the `collection_roots` table, not through hardcoded source paths.

## Indexing Stages

The media indexer prints numbered stages and completion status. The concrete implementation used by every stage is:

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

Most stages are incremental. Completed rows are skipped unless a rerun flag is used. Cross-media work writes a resumable work file before final DB application, so a crash during DB upsert can resume from saved `3e`/`3f` results. Search-index sync is also incremental: CLIP, face, and OCR ANN side tables update immediately after the stage that produces their source data, and missing ANN side tables are rebuilt in the relevant `4b`/`4c`, `6b`/`6c`, or `8b`/`8c` stage rather than by a separate final stage.

The application calls Stage `4a` and its search mode "CLIP," but its default image
embedder is specifically SigLIP2, loaded with
`open_clip.create_model_and_transforms`. Iris text-to-image search uses an ONNX export
of the text tower from the same default SigLIP2 model, stored under
`tools/media_indexer/models/clip-text`. On-demand image and face embedding also imports
the same OpenCLIP and InsightFace implementations from the media indexer.

Force a stage to rerun:

```bash
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media \
  --db-dir ../../lancedb \
  --collection-id my_collection \
  --rerun-stage 3e \
  --rerun-stage 3f
```

## Search Behavior

Filename, CLIP, and OCR share one search UI in Iris.

- Filename search scans the visible filesystem tree.
- CLIP search uses the LanceDB CLIP ANN side table and can search by text, pasted image, or image path.
- OCR search supports normal multi-term matching and quoted exact phrases.
- Folder scope can be blank for all indexed folders, an absolute path, a collection-relative path, a partial path section, or a single folder segment.

## Notes For Public Repo Hygiene

Do not commit:

- `lancedb/` or `.lance` tables.
- downloaded/exported model files under `tools/media_indexer/models/`.
- extracted video still folders.
- personal media paths, collection roots, or generated reports.
- ad-hoc images or screenshots unless they are deliberately part of the app assets.
