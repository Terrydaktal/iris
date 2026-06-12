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
- VideoHash for videos.
- PySceneDetect still extraction for videos.
- Image pHash and video-still pHash generation.
- Cross-media image-to-video and video-to-image relationships when an image matches a video still.
- CLIP embeddings for images and video stills.
- SIFT duplicate grouping using CLIP ANN candidates.
- Face detection and embeddings.
- PaddleOCR text detection and EasyOCR text extraction.
- Face, CLIP, and OCR text ANN side tables for Iris search.

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
- `IRIS_IMAGESEARCH_DIR`: override the media indexer/runtime helper directory. Defaults to `tools/media_indexer`.
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

The media indexer prints numbered stages and completion status. Current stages are:

- `Stage 0a/9`: startup scan media files.
- `Stage 0b/9`: split images and videos.
- `Stage 0c/9`: open LanceDB table and check schema.
- `Stage 0d/9`: ensure `file_name` scalar index.
- `Stage 0e/9`: load DB rows into memory.
- `Stage 0f/9`: normalize DB rows.
- `Stage 0g/9`: check legacy DB keys.
- `Stage 0h/9`: migrate legacy DB keys if needed.
- `Stage 1a/9`: VideoHash videos.
- `Stage 1b/9`: apply VideoHash groups.
- `Stage 2/9`: PySceneDetect video still extraction.
- `Stage 3a/9`: cached image metadata.
- `Stage 3b/9`: image pHash.
- `Stage 3c/9`: video-still pHash.
- `Stage 3d/9`: apply image pHash groups.
- `Stage 3e/9`: image-to-video-frame matching.
- `Stage 3f/9`: video-frame-to-image matching.
- `Stage 4/9`: CLIP embeddings for images and video stills.
- `Stage 5a/9`: SIFT CLIP ANN shortlist.
- `Stage 5b/9`: SIFT master match.
- `Stage 6/9`: faces.
- `Stage 7/9`: PaddleOCR text detection.
- `Stage 8/9`: EasyOCR text extraction.
- `Stage 9/9`: search-index sync for Face, CLIP, and OCR ANN tables.

Most stages are incremental. Completed rows are skipped unless a rerun flag is used. Cross-media work writes a resumable work file before final DB application, so a crash during DB upsert can resume from saved `3e`/`3f` results.

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
