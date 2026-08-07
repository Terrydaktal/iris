use super::*;

pub(crate) fn get_system_disks() -> Vec<PathBuf> {
    let mut disks = Vec::new();
    let mut seen = HashSet::new();

    // 1. Root directory is always a disk
    let root = PathBuf::from("/");
    seen.insert(root.clone());
    disks.push(root);

    // 2. Check for username to scan user-specific media directories
    let usernames = vec![
        std::env::var("USER").unwrap_or_default(),
        std::env::var("USERNAME").unwrap_or_default(),
    ];

    let mut scan_paths = vec![PathBuf::from("/media"), PathBuf::from("/mnt")];

    for user in usernames {
        if !user.is_empty() {
            scan_paths.push(PathBuf::from(format!("/media/{}", user)));
            scan_paths.push(PathBuf::from(format!("/run/media/{}", user)));
        }
    }

    // Add user home directory as a top-level shortcut
    if let Ok(home) = std::env::var("HOME") {
        let home_path = PathBuf::from(home);
        if home_path.is_dir() {
            let canon = home_path.canonicalize().unwrap_or(home_path);
            if seen.insert(canon.clone()) {
                disks.push(canon);
            }
        }
    }

    for path in scan_paths {
        if path.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&path) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let entry_path = entry.path();
                    if entry_path.is_dir() {
                        let canon = entry_path.canonicalize().unwrap_or(entry_path);
                        if seen.insert(canon.clone()) {
                            disks.push(canon);
                        }
                    }
                }
            }
        }
    }

    // Sort so it is deterministic
    disks.sort();
    disks
}

pub(crate) fn normalized_path_for_match(path: &Path) -> String {
    path.to_string_lossy()
        .replace('\\', "/")
        .trim_end_matches('/')
        .to_lowercase()
}

pub(crate) fn path_matches_db_root(path: &Path, root: &Path) -> bool {
    if path.starts_with(root) || root.starts_with(path) {
        return true;
    }

    let path_norm = normalized_path_for_match(path);
    let root_norm = normalized_path_for_match(root);
    if path_norm == root_norm {
        return true;
    }
    if path_norm.starts_with(&(root_norm.clone() + "/"))
        || root_norm.starts_with(&(path_norm.clone() + "/"))
    {
        return true;
    }

    let root_name = root
        .file_name()
        .and_then(|n| n.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();
    if !root_name.is_empty() {
        let needle = format!("/{}/", root_name);
        if path_norm.contains(&needle) || path_norm.ends_with(&format!("/{root_name}")) {
            return true;
        }
    }

    false
}

pub(crate) fn is_path_ai_backed(path: &Path) -> bool {
    let db_roots = get_db_roots();
    is_path_ai_backed_with_roots(path, &db_roots)
}

pub(crate) fn is_path_ai_backed_with_roots(
    path: &Path,
    db_roots: &HashMap<String, PathBuf>,
) -> bool {
    db_roots
        .values()
        .any(|root| path_matches_db_root(path, root))
}
