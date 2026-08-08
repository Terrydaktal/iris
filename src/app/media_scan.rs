use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Sender;

pub(crate) fn is_supported_media_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|x| x.to_str())
            .map(|x| x.to_ascii_lowercase())
            .as_deref(),
        Some(
            "jpg"
                | "jpeg"
                | "png"
                | "bmp"
                | "gif"
                | "webp"
                | "tiff"
                | "avif"
                | "heif"
                | "heic"
                | "ico"
                | "icns"
                | "svg"
                | "mp4"
                | "mov"
                | "avi"
                | "mkv"
                | "webm"
                | "m4v"
                | "wmv"
                | "mpg"
                | "mpeg"
        )
    )
}

pub(crate) fn collect_images_recursive_cancelable(
    dir: &Path,
    tx: &Sender<PathBuf>,
    visited: &mut HashSet<PathBuf>,
    token: &AtomicU64,
    generation: u64,
) {
    if token.load(Ordering::Relaxed) != generation {
        return;
    }
    let canon_dir = match dir.canonicalize() {
        Ok(c) => c,
        Err(_) => dir.to_path_buf(),
    };
    if !visited.insert(canon_dir) {
        return;
    }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            if token.load(Ordering::Relaxed) != generation {
                return;
            }
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_symlink() {
                continue;
            }
            let path = entry.path();
            if file_type.is_dir() {
                collect_images_recursive_cancelable(&path, tx, visited, token, generation);
            } else if file_type.is_file() {
                if is_supported_media_path(&path) {
                    if let Ok(canon) = path.canonicalize() {
                        let _ = tx.send(canon);
                    } else {
                        let _ = tx.send(path);
                    }
                }
            }
        }
    }
}

pub(crate) fn collect_flat_images(dir: &Path) -> Vec<PathBuf> {
    let mut collected = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.filter_map(|entry| entry.ok()) {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            let path = entry.path();
            if file_type.is_file() && is_supported_media_path(&path) {
                collected.push(path);
            }
        }
    }
    collected.sort();
    collected
}
