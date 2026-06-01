# Iris

Iris is a Rust desktop app for browsing local images/videos, inspecting metadata, and searching indexed media by filename, description, or OCR text.

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
    └── on_demand_embeddings.py
```

## What Each File Does

### `src/main.rs`

- Main GUI application.
- Inputs:
  - CLI args: `iris [--same-window|-s|--reuse-window|-r] [--new-window|-n] [--no-daemon] [PATH]`
  - Filesystem media under the opened path.
  - LanceDB tables (`media_index`, optional `collection_roots`).
  - External tools: `exiftool`, `ffprobe`, `mpv`, and optional `dolphin`.
  - Python tooling in `imagesearch` for SIFT diagnostics and on-demand CLIP/face embeddings.
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
  - Imports model/runtime code from the `imagesearch` project directory.

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
- `IRIS_IMAGESEARCH_DIR`: path to the `imagesearch` project used for model/runtime helpers.
- `IRIS_ON_DEMAND_EMBED_SCRIPT`: optional override for the on-demand embedding script path.
- `IRIS_EXIFTOOL`: optional explicit path to `exiftool`.
- `IRIS_FFPROBE`: optional explicit path to `ffprobe`.

Collection root mapping is resolved from the `collection_roots` table in LanceDB (and can be discovered from indexed file samples).

## External Dependencies

- Rust + Cargo
- `uv` (for Python helper execution)
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
