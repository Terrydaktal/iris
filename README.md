# Iris

Iris is a highly optimized, high-fidelity, native graphical image viewer and binary file layout inspector built using **Rust** and the **egui** framework. It is designed to explore large directory trees instantly, stream images in real-time, and parse low-level headers for various image formats (PNG, JPEG, WebP, BMP) with zero UI lag, minimal memory footprint (< 200MB), and near-0% idle CPU usage.

---

## Project Structure & Directory Tree

```text
.
├── Cargo.lock                     # Workspace dependency lock file
├── Cargo.toml                     # Root package definition (build manifest)
├── icon.png                       # Application icon (transparency processed)
├── iris.desktop                   # Linux/Wayland Desktop Application launcher entry
├── make_transparent.py            # Python utility script for transparency processing
├── src/                           # Root crate source code
│   ├── main.rs                    # Application entrypoint & core GUI logic
│   └── bin/
│       └── make_transparent.rs    # Rust utility binary for transparency processing
├── iris/                          # Nested duplicate/subcrate crate (for absolute parity)
│   ├── Cargo.lock
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs                # Exactly identical duplicate of root main.rs
│   │   └── bin/
│   │       └── make_transparent.rs
└── README.md                      # Detailed project documentation
```

---

## Script & Binary Details

### 1. `iris` Application (Core Viewer)
* **Script / Code Location**: [src/main.rs](file:///home/lewis/Dev/iris/src/main.rs) & [iris/src/main.rs](file:///home/lewis/Dev/iris/iris/src/main.rs)
* **Purpose**: Serves as the primary graphical interface for image viewing, EXIF tag querying, recursive directory scanning, and low-level chunk parsing.
* **Key Features**:
  - **Asynchronous Neighbour Scanning**: Flat neighbours are loaded in a background thread to prevent GUI blockage.
  - **Progressive MPSC Streaming**: Thumbnails and recursive image paths are streamed dynamically via `mpsc` channels, popping up in the gallery in real-time.
  - **Thread-Throttled Decoding**: Decodes thumbnails using at most 4 background worker threads, keeping memory consumption < 200MB.
  - **Dual Side Panel**: Toggles between a collapsible byte-level **Binary Layout** inspector (supporting PNG, JPEG, WebP, BMP) and a searchable **Raw EXIF Metadata** dump.
  - **Winit Graphics Fallback**: Seamlessly falls back to XWayland if Wayland graphics context initialization fails under NVIDIA drivers.
* **Input**:
  - Command-line arguments: `iris [--same-window | -s] [--no-daemon] <image_path_or_directory>`
* **Output**:
  - Detaches itself as a desktop background daemon by default, launching a fluid, GPU-accelerated window.

### 2. `make_transparent` Utility (Python / Rust equivalents)
* **Python Script**: [make_transparent.py](file:///home/lewis/Dev/iris/make_transparent.py)
  - **Input**: Absolute path to an image file (defaults to cached white-background app icon).
  - **Output**: Saves a transparent PNG to [icon.png](file:///home/lewis/Dev/iris/icon.png) by smoothly alpha-scaling light/white pixels.
* **Rust Binary**: [src/bin/make_transparent.rs](file:///home/lewis/Dev/iris/src/bin/make_transparent.rs) (nested at [iris/src/bin/make_transparent.rs](file:///home/lewis/Dev/iris/iris/src/bin/make_transparent.rs))
  - **Input**: Reads image file bytes from a white-background source path.
  - **Output**: Writes the processed transparent PNG back to [icon.png](file:///home/lewis/Dev/iris/icon.png).
  - **Execution**: Can be run via Cargo:
    ```bash
    cargo run --bin make_transparent
    ```

---

## Execution & Pipeline Order

Follow these steps to compile, run, and configure the application:

### Step 1: Process the Application Icon (Optional)
If you need to regenerate the transparent icon from a white-background source:
* **Option A (Python)**:
  ```bash
  python make_transparent.py
  ```
* **Option B (Rust)**:
  ```bash
  cargo run --bin make_transparent
  ```

### Step 2: Compile & Run the Primary GUI App
Launch Iris by compiling and executing the binary using Cargo:
* **Foreground Execution (with logging)**:
  ```bash
  cargo run --release -- --no-daemon .
  ```
* **Background Daemon Mode (releases terminal instantly)**:
  ```bash
  cargo run --release -- .
  ```

### Step 3: Integrate with Desktop Environment (Optional)
To register Iris as a system-wide desktop application on Linux/KDE/GNOME, copy the `.desktop` launcher to your local applications folder:
```bash
cp iris.desktop ~/.local/share/applications/
```
