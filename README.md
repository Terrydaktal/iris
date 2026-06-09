# Iris

Iris is a Rust desktop app for browsing local images/videos, inspecting metadata, and searching indexed media by filename, CLIP similarity, or OCR text.

## Project Structure

```text
.
├── Cargo.toml
├── Cargo.lock
├── README.md
├── iris.desktop
├── icon.png
├── make_transparent.py
├── clip_viewer_ref.rs
├── src
│   ├── main.rs
│   └── bin
│       └── make_transparent.rs
└── tools
    ├── on_demand_embeddings.py
    └── media_indexer
        ├── main.py
        ├── easyocr_worker.py
        ├── face_worker.py
        ├── paddle_worker.py
        ├── pyproject.toml
        ├── uv.lock
        ├── models/clip-text   # generated locally, ignored by Git
        └── tools
```

## What Each File Does

### `src/main.rs`

- Main GUI application.
- Inputs:
  - CLI args: `iris [--same-window|-s|--reuse-window|-r] [--new-window|-n] [--no-daemon] [PATH]`
  - Filesystem media under the opened path.
  - LanceDB tables (`media_index`, optional `collection_roots`).
  - External tools: `exiftool`, `ffprobe`, `mpv`, and optional `dolphin`.
  - Python tooling in `tools/media_indexer` for database indexing, SIFT diagnostics, and on-demand CLIP/face embeddings.
- Outputs:
  - Interactive gallery/viewer UI.
  - Similarity and OCR/description search results.
  - EXIF/raw metadata and ffprobe output for videos.

### `tools/on_demand_embeddings.py`

- Helper used by Iris when a file does not already have stored vectors.
- Inputs:
  - `--image <path>`
  - `--clip` and/or `--faces`
- Output:
  - JSON payload on stdout:
    - `{"ok": true, "clip_embedding": [...], "face_embeddings": [[...], ...]}`
    - or `{"ok": false, "error": "..."}`
- Runtime dependency:
  - Imports model/runtime code from `tools/media_indexer` by default, or from `IRIS_IMAGESEARCH_DIR` when set.

### `tools/media_indexer/main.py`

- Database builder for media collections.
- Inputs:
  - `uv run embedimages <media-folder> --db-dir <lancedb-dir>`
  - Optional collection id, OCR, face, CLIP, pHash, VideoHash, and SIFT tuning flags.
- Outputs:
  - LanceDB `media_index` rows.
  - ANN side tables for face, CLIP, and OCR search.
  - Optional extracted video stills next to scanned video folders.

### `tools/media_indexer/*_worker.py`

- Worker processes used by the database builder:
  - `face_worker.py`: InsightFace detection and ArcFace embeddings.
  - `paddle_worker.py`: PaddleOCR text detection.
  - `easyocr_worker.py`: EasyOCR text extraction.

### `tools/media_indexer/tools`

- Maintenance and diagnostic scripts used by Iris and the indexer.
- Includes SIFT comparison/repair, weak-link pruning, face reruns, collection rollback, CUDA probing, failure reports, and CLIP text model export.

### `tools/media_indexer/models/clip-text`

- ONNX CLIP text encoder and tokenizer files used by Iris for in-app CLIP text search.
- Generated locally with `tools/media_indexer/tools/export_clip_text_onnx.py`.
- Ignored by Git because the exported ONNX data file is multi-GB.

### `src/bin/make_transparent.rs`

- Small icon utility.
- Inputs:
  - `cargo run --bin make_transparent -- [src] [dst]`
  - Defaults: `icon_source.png` -> `icon.png`
- Output:
  - Transparent PNG icon.

### `make_transparent.py`

- Python version of the icon utility.
- Inputs:
  - `python3 make_transparent.py [src] [dst]`
  - Defaults: `icon_source.png` -> `icon.png`
- Output:
  - Transparent PNG icon.

### `clip_viewer_ref.rs`

- Reference/experimental viewer implementation kept outside the main build path.

### `iris.desktop`

- Desktop entry template for Linux launchers.

## Configuration

Iris no longer requires hardcoded personal paths. Use environment variables:

- `IRIS_DB_DIR`: LanceDB directory. If unset, Iris looks for an existing `lancedb` directory on mounted volumes, then falls back to `${XDG_DATA_HOME}/iris/lancedb` or `${HOME}/.local/share/iris/lancedb`.
- `IRIS_IMAGESEARCH_DIR`: optional override path for the image indexing/runtime helpers. If unset, Iris uses `tools/media_indexer`.
- `IRIS_ON_DEMAND_EMBED_SCRIPT`: optional override for the on-demand embedding script path.
- `IRIS_EXIFTOOL`: optional explicit path to `exiftool`.
- `IRIS_FFPROBE`: optional explicit path to `ffprobe`.

Collection root mapping is resolved from the `collection_roots` table in LanceDB (and can be discovered from indexed file samples).

## External Dependencies

- Rust + Cargo
- `uv` (for Python helper execution)
- Python dependencies from `tools/media_indexer/pyproject.toml` when building or repairing the media index
- `exiftool`
- `ffprobe` (from ffmpeg)
- `mpv` (video open action)
- `dolphin` (preferred folder opener; app has a fallback opener if unavailable)

## Runtime Pipeline

1. Start app and open a file/folder.
2. Build gallery list from filesystem scan.
3. If AI-backed search is used:
   - Lazy-load DB indices and text encoder.
   - Search mode:
     - CLIP -> CLIP text encoder + vector index.
     - OCR search -> OCR text index with phrase/term ranking.
4. For files missing embeddings:
   - Trigger `tools/on_demand_embeddings.py`.
   - Compute CLIP/face vectors on the fly.
   - Run “Show most similar” or “Show more of this person” using returned vectors.
5. Side panels:
   - EXIF/raw metadata via `exiftool`
   - video metadata via `ffprobe`
   - duplicates/similarity diagnostics from DB + SIFT helper

## Build Or Update The Media Database

```bash
cd tools/media_indexer
UV_CACHE_DIR="/data/.cache/uv" uv sync
UV_CACHE_DIR="/data/.cache/uv" uv run python tools/export_clip_text_onnx.py --out-dir models/clip-text
UV_CACHE_DIR="/data/.cache/uv" uv run embedimages /path/to/media --db-dir /path/to/lancedb
```

Use `--collection-id <id>` when adding multiple folders to one database. Stored DB file names use `<collection_id>/<relative/path>`, and Iris resolves those through the `collection_roots` table.

## Run

```bash
# from repo root
cargo run --release -- --no-daemon
```

```bash
# open a specific path
cargo run --release -- --no-daemon /path/to/file-or-folder
```

```bash
# reuse an existing window
cargo run --release -- --same-window /path/to/file-or-folder
```
