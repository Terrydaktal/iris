use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};

pub(crate) fn drain_scan_batch(rx: &Receiver<PathBuf>, limit: usize) -> (Vec<PathBuf>, bool) {
    let mut paths = Vec::new();
    while paths.len() < limit {
        match rx.try_recv() {
            Ok(path) => paths.push(path),
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => return (paths, true),
        }
    }
    (paths, false)
}

pub(crate) fn apply_scan_batch(
    paths: &mut Arc<Vec<PathBuf>>,
    video_indices: &mut Vec<usize>,
    batch: Vec<PathBuf>,
    complete: bool,
) {
    // A filename search or viewer may retain an immutable snapshot. Otherwise
    // append in place instead of cloning the growing path list every repaint.
    let paths = Arc::make_mut(paths);
    let first_index = paths.len();
    video_indices.extend(batch.iter().enumerate().filter_map(|(index, path)| {
        super::paths::is_video_path(path).then_some(first_index + index)
    }));
    paths.extend(batch);
    if complete {
        paths.sort_by(|a, b| b.cmp(a));
        *video_indices = paths
            .iter()
            .enumerate()
            .filter_map(|(index, path)| super::paths::is_video_path(path).then_some(index))
            .collect();
    }
}

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
    if !visited.insert(canon_dir.clone()) {
        return;
    }
    if let Ok(entries) = std::fs::read_dir(&canon_dir) {
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
                // The directory is already canonical and symlinks are skipped.
                // Do not perform a separate filesystem lookup for every file.
                if is_supported_media_path(&path) && tx.send(path).is_err() {
                    return;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn discovery_is_published_before_the_scan_finishes() {
        let (tx, rx) = mpsc::channel();
        tx.send(PathBuf::from("first.jpg")).unwrap();
        tx.send(PathBuf::from("second.mp4")).unwrap();
        let (batch, complete) = drain_scan_batch(&rx, 4096);
        assert!(!complete);

        let mut paths = Arc::new(Vec::new());
        let mut videos = Vec::new();
        apply_scan_batch(&mut paths, &mut videos, batch, complete);
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0], PathBuf::from("first.jpg"));
        assert_eq!(videos, vec![1]);

        drop(tx);
        let (batch, complete) = drain_scan_batch(&rx, 4096);
        assert!(complete);
        apply_scan_batch(&mut paths, &mut videos, batch, complete);
        assert_eq!(paths[0], PathBuf::from("second.mp4"));
        assert_eq!(videos, vec![0]);
    }

    #[test]
    fn discovery_batches_bound_work_without_losing_paths() {
        let (tx, rx) = mpsc::channel();
        for name in ["a.jpg", "b.jpg", "c.jpg"] {
            tx.send(PathBuf::from(name)).unwrap();
        }
        drop(tx);
        let (first, complete) = drain_scan_batch(&rx, 2);
        assert_eq!(first, vec![PathBuf::from("a.jpg"), PathBuf::from("b.jpg")]);
        assert!(!complete);
        let (second, complete) = drain_scan_batch(&rx, 2);
        assert_eq!(second, vec![PathBuf::from("c.jpg")]);
        assert!(complete);
    }

    #[test]
    fn discovery_appends_in_place_and_keeps_search_and_navigation_snapshots_stable() {
        let mut paths = Arc::new(vec![PathBuf::from("a.jpg"), PathBuf::from("z.mp4")]);
        let mut videos = vec![1];
        let original = Arc::as_ptr(&paths);
        apply_scan_batch(&mut paths, &mut videos, vec![PathBuf::from("m.png")], false);
        assert_eq!(original, Arc::as_ptr(&paths));

        let snapshot = Arc::clone(&paths);
        apply_scan_batch(&mut paths, &mut videos, vec![PathBuf::from("b.mov")], true);
        assert_eq!(snapshot.len(), 3);
        assert_eq!(snapshot[1], PathBuf::from("z.mp4"));
        assert_eq!(
            paths.as_ref(),
            &vec![
                PathBuf::from("z.mp4"),
                PathBuf::from("m.png"),
                PathBuf::from("b.mov"),
                PathBuf::from("a.jpg"),
            ]
        );
        assert_eq!(videos, vec![0, 2]);
    }

    #[test]
    fn recursive_discovery_only_reads_names_and_skips_symlinks() {
        struct TestDirectory(PathBuf);
        impl Drop for TestDirectory {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let root = std::env::temp_dir().join(format!(
            "iris-scan-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        let _cleanup = TestDirectory(root.clone());
        std::fs::create_dir(root.join("nested")).unwrap();
        // Empty files cannot be decoded: discovery must not inspect media data.
        std::fs::write(root.join("still.jpg"), []).unwrap();
        std::fs::write(root.join("nested/video.mp4"), []).unwrap();
        std::fs::write(root.join("ignored.txt"), []).unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(root.join("still.jpg"), root.join("linked.jpg")).unwrap();
            std::os::unix::fs::symlink(&root, root.join("loop")).unwrap();
        }
        let (tx, rx) = mpsc::channel();
        let token = AtomicU64::new(1);
        collect_images_recursive_cancelable(&root, &tx, &mut HashSet::new(), &token, 1);
        drop(tx);
        let mut found: Vec<_> = rx.try_iter().collect();
        found.sort();
        let root = root.canonicalize().unwrap();
        assert_eq!(
            found,
            vec![root.join("nested/video.mp4"), root.join("still.jpg")]
        );
    }

    #[test]
    fn cancelled_discovery_publishes_no_files() {
        let (tx, rx) = mpsc::channel();
        collect_images_recursive_cancelable(
            Path::new("."),
            &tx,
            &mut HashSet::new(),
            &AtomicU64::new(2),
            1,
        );
        assert_eq!(rx.try_iter().count(), 0);
    }
}
