# Iris

Iris is a native Rust desktop media viewer built with `eframe/egui`.

Current build includes:
- interactive image/video viewer
- folder gallery grid
- filename filtering
- CLIP text search and OCR text search backed by LanceDB
- EXIF panel and binary layout inspector
- duplicate/similarity tools (SIFT + pHash/VideoHash metadata)
- home-page folder browser start when no path is provided

## Project Structure

```text
.
├── Cargo.lock
├── Cargo.toml
├── README.md
├── clip_viewer_ref.rs
├── icon.png
├── iris.desktop
├── make_transparent.py
├── src
│   ├── bin
│   │   └── make_transparent.rs
│   └── main.rs
└── target                      # build artifacts (generated)
```

## Files, Inputs, Outputs

### `src/main.rs`
- Purpose: main GUI application (`iris`).
- Inputs:
  - CLI args: `iris [--same-window|-s|--reuse-window|-r] [--new-window|-n] [--no-daemon] [PATH]`
  - optional UNIX socket messages from other iris processes
  - media files from selected folders
  - EXIF output from `exiftool`
  - optional AI index/model files (see AI requirements)
- Outputs:
  - desktop GUI window
  - gallery grid and detail viewport
  - EXIF/layout/duplicate side panels
  - optional commands to open folders (`dolphin`) and videos (`mpv`)

### `src/bin/make_transparent.rs`
- Purpose: Rust utility that converts near-white pixels to transparency in an icon image.
- Input: hardcoded source image path in the file.
- Output: writes transparent PNG to `icon.png`.

### `make_transparent.py`
- Purpose: Python utility equivalent of the Rust icon transparency tool.
- Input: hardcoded source image path in the file.
- Output: writes transparent PNG to `icon.png`.

### `clip_viewer_ref.rs`
- Purpose: reference code snapshot used for comparison/porting ideas.
- Input/Output: not part of Cargo build by default.

### `iris.desktop`
- Purpose: desktop launcher entry.
- Input: none.
- Output: launcher metadata consumed by desktop environments.

## Runtime Requirements

Base viewer:
- Rust toolchain + Cargo
- graphics stack supported by `eframe`
- `exiftool` available in `PATH` (for metadata panel)

Optional integrations:
- `dolphin` (preferred folder opener; falls back to `xdg-open`)
- `mpv` (video open actions)

AI search path assumptions in current code:
- LanceDB directory: `/media/lewis/1b/lancedb`
- collection roots:
  - `/media/lewis/1b/Phone`
  - `/media/lewis/1b/Telegram Backup`
- CLIP text model: `/home/lewis/Dev/imagesearch/models/clip-text/clip_text.onnx`
- tokenizer: `/home/lewis/Dev/imagesearch/models/clip-text/tokenizer.json`

If your machine differs, update these constants in `src/main.rs`.

## Operation and Execution Order

1. Optional: regenerate app icon transparency.
   - Python:
     ```bash
     python3 make_transparent.py
     ```
   - Rust:
     ```bash
     cargo run --bin make_transparent
     ```

2. Build and run Iris.
   - start at home-page browser (no path):
     ```bash
     cargo run --release -- --no-daemon
     ```
   - open a specific path immediately:
     ```bash
     cargo run --release -- --no-daemon /path/to/media/or/folder
     ```

3. Reuse an existing window from another shell:
   ```bash
   cargo run --release -- --same-window /path/to/file
   ```

4. In-app workflow:
- press `G` to open/close gallery grid
- `Filename` tab filters current folder contents in-grid
- `AI Description` / `OCR Text` tabs run semantic search and show results in the same grid
- open side panels for binary layout, raw EXIF, and duplicates tools

## Notes

- Launching without `--no-daemon` daemonizes by default.
- `--new-window` currently behaves as a no-op because new-window behavior is already default.
- The previous nested `iris/` subcrate was removed; this repo is now a single crate.
