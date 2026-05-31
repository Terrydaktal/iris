# Iris

Iris is a Rust desktop app for browsing local image and video folders, checking metadata, and searching a collection by filename, description, or OCR text.

## What it does

- folder gallery and single-image viewer
- filename filtering in the gallery
- description search and OCR text search backed by local LanceDB indices
- EXIF inspection and a binary layout view
- duplicate and similarity tools based on SIFT, pHash, and VideoHash metadata
- home-page folder browser when the app starts without a path

## Repository Layout

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
└── target                      # generated build output
```

## Files and Outputs

### `src/main.rs`

- Purpose: main GUI application.
- Inputs:
  - command line arguments: `iris [--same-window|-s|--reuse-window|-r] [--new-window|-n] [--no-daemon] [PATH]`
  - optional UNIX socket messages from other Iris processes
  - media files from the selected folder or folder tree
  - EXIF output from `exiftool`
  - local description/OCR index and model files
- Outputs:
  - the desktop GUI window
  - gallery and detail views
  - layout, EXIF, and duplicate side panels
  - optional folder opens through `dolphin`
  - optional video opens through `mpv`

### `src/bin/make_transparent.rs`

- Purpose: small Rust utility for turning a light icon source into a transparent PNG.
- Input: hardcoded source image path in the file.
- Output: `icon.png`.

### `make_transparent.py`

- Purpose: Python version of the same icon helper.
- Input: hardcoded source image path in the file.
- Output: `icon.png`.

### `clip_viewer_ref.rs`

- Purpose: reference code kept around for comparison and porting.
- Input/output: not part of the Cargo build.

### `iris.desktop`

- Purpose: desktop launcher entry for Linux desktop environments.
- Input: none.
- Output: launcher metadata.

## Requirements

- Rust toolchain with Cargo
- `exiftool` in `PATH`
- `dolphin` if you want the preferred folder opener
- `mpv` if you want the video open action

The current code also expects the local search assets to live here:

- LanceDB directory: `/media/lewis/1b/lancedb`
- collection roots:
  - `/media/lewis/1b/Phone`
  - `/media/lewis/1b/Telegram Backup`
- description model: `/home/lewis/Dev/imagesearch/models/clip-text/clip_text.onnx`
- tokenizer: `/home/lewis/Dev/imagesearch/models/clip-text/tokenizer.json`

If your setup is different, update those paths in `src/main.rs`.

## How To Run It

1. Optional: regenerate the icon transparency.
   ```bash
   python3 make_transparent.py
   ```

   or:

   ```bash
   cargo run --bin make_transparent
   ```

2. Start the app.
   ```bash
   cargo run --release -- --no-daemon
   ```

3. Open a folder or file directly.
   ```bash
   cargo run --release -- --no-daemon /path/to/media/or/folder
   ```

4. Reuse an existing window from another shell.
   ```bash
   cargo run --release -- --same-window /path/to/file
   ```

## In-App Flow

- `G` toggles the gallery grid.
- The `Filename`, `Description Search`, and `OCR Text` tabs all use the same grid area.
- The right sidebar switches between binary layout, raw EXIF, and duplicates.
- The app can also open folders and videos with external tools when those tools are installed.

## Notes

- The app daemonizes unless you pass `--no-daemon`.
- `--new-window` is currently a no-op because the default behavior already opens a new window.
- The old nested `iris/` subcrate is no longer part of this repository.
