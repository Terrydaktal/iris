use super::*;

pub(crate) const MEDIA_INDEX_TABLE: &str = "media_index";
pub(crate) const COLLECTION_ROOTS_TABLE: &str = "collection_roots";

pub(crate) fn looks_like_lancedb_dir(path: &Path) -> bool {
    path.join(format!("{MEDIA_INDEX_TABLE}.lance")).is_dir()
        || path
            .join(format!("{COLLECTION_ROOTS_TABLE}.lance"))
            .is_dir()
}

pub(crate) fn discover_existing_db_dir() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("lancedb"));
        if let Some(parent) = cwd.parent() {
            candidates.push(parent.join("lancedb"));
        }
    }

    let users = [
        std::env::var("USER").unwrap_or_default(),
        std::env::var("USERNAME").unwrap_or_default(),
    ];
    for user in users.iter().filter(|user| !user.is_empty()) {
        for mount_base in [
            PathBuf::from("/media").join(user),
            PathBuf::from("/run/media").join(user),
        ] {
            if let Ok(entries) = std::fs::read_dir(&mount_base) {
                for entry in entries.filter_map(|entry| entry.ok()) {
                    let path = entry.path();
                    if path.is_dir() {
                        candidates.push(path.join("lancedb"));
                    }
                }
            }
        }
    }
    if let Ok(entries) = std::fs::read_dir("/mnt") {
        for entry in entries.filter_map(|entry| entry.ok()) {
            let path = entry.path();
            if path.is_dir() {
                candidates.push(path.join("lancedb"));
            }
        }
    }

    candidates
        .into_iter()
        .filter(|path| looks_like_lancedb_dir(path))
        .find_map(|path| path.canonicalize().ok().or(Some(path)))
}

pub(crate) fn default_db_dir() -> PathBuf {
    let repo_db_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("lancedb");
    if repo_db_dir.exists() || cfg!(debug_assertions) {
        return repo_db_dir;
    }

    if let Some(discovered) = discover_existing_db_dir() {
        return discovered;
    }

    if let Ok(raw) = std::env::var("XDG_DATA_HOME") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed).join("iris").join("lancedb");
        }
    }
    if let Ok(raw) = std::env::var("HOME") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed)
                .join(".local")
                .join("share")
                .join("iris")
                .join("lancedb");
        }
    }
    PathBuf::from("./lancedb")
}

pub(crate) fn get_db_dir() -> PathBuf {
    static DB_DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    DB_DIR
        .get_or_init(|| {
            if let Ok(raw) = std::env::var("IRIS_DB_DIR") {
                let trimmed = raw.trim();
                if !trimmed.is_empty() {
                    return PathBuf::from(trimmed);
                }
            }
            default_db_dir()
        })
        .clone()
}

pub(crate) fn resolve_media_indexer_dir() -> PathBuf {
    if let Ok(raw) = std::env::var("IRIS_MEDIA_INDEXER_DIR") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }

    let mut candidates = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("tools/media_indexer"));
    }
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tools/media_indexer"));

    for candidate in candidates {
        if candidate.is_dir() {
            return candidate.canonicalize().unwrap_or(candidate);
        }
    }

    PathBuf::from("tools/media_indexer")
}

pub(crate) fn resolve_on_demand_embeddings_script_path() -> PathBuf {
    if let Ok(raw) = std::env::var("IRIS_ON_DEMAND_EMBED_SCRIPT") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tools")
        .join("on_demand_embeddings.py")
}

pub(crate) fn dedupe_dirs(candidates: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for candidate in candidates {
        if !candidate.is_dir() {
            continue;
        }
        let canonical = candidate.canonicalize().unwrap_or(candidate.clone());
        let key = canonical.to_string_lossy().to_string();
        if seen.insert(key) {
            out.push(canonical);
        }
    }
    out
}

pub(crate) fn add_dir_and_children(base: &Path, out: &mut Vec<PathBuf>) {
    if !base.is_dir() {
        return;
    }
    out.push(base.to_path_buf());
    if let Ok(entries) = std::fs::read_dir(base) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_dir() {
                out.push(path);
            }
        }
    }
}

pub(crate) fn add_dir_children_depth2(base: &Path, out: &mut Vec<PathBuf>) {
    if !base.is_dir() {
        return;
    }
    if let Ok(entries) = std::fs::read_dir(base) {
        for entry in entries.filter_map(|e| e.ok()) {
            let level1 = entry.path();
            if !level1.is_dir() {
                continue;
            }
            out.push(level1.clone());
            if let Ok(level2_entries) = std::fs::read_dir(&level1) {
                for entry2 in level2_entries.filter_map(|e| e.ok()) {
                    let level2 = entry2.path();
                    if level2.is_dir() {
                        out.push(level2);
                    }
                }
            }
        }
    }
}

pub(crate) fn candidate_root_dirs(db_dir: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(parent) = db_dir.parent() {
        add_dir_and_children(parent, &mut candidates);
        if let Some(grand) = parent.parent() {
            add_dir_and_children(grand, &mut candidates);
        }
    }
    add_dir_children_depth2(Path::new("/media"), &mut candidates);
    add_dir_children_depth2(Path::new("/run/media"), &mut candidates);
    add_dir_children_depth2(Path::new("/mnt"), &mut candidates);
    dedupe_dirs(candidates)
}

async fn load_collection_roots_from_table(db_dir: &Path) -> Result<HashMap<String, PathBuf>> {
    let db = lancedb::connect(db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let table_names = db.table_names().execute().await?;
    if !table_names
        .iter()
        .any(|name| name == COLLECTION_ROOTS_TABLE)
    {
        return Ok(HashMap::new());
    }

    let table = db.open_table(COLLECTION_ROOTS_TABLE).execute().await?;
    let stream = table
        .query()
        .select(Select::columns(&["collection_id", "root_path"]))
        .execute()
        .await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;

    let mut roots = HashMap::new();
    for batch in &batches {
        let ids = string_col(batch, "collection_id")?;
        let paths = string_col(batch, "root_path")?;
        for row in 0..batch.num_rows() {
            if ids.is_null(row) || paths.is_null(row) {
                continue;
            }
            let collection = ids.value(row).trim();
            let root_path = paths.value(row).trim();
            if collection.is_empty() || root_path.is_empty() {
                continue;
            }
            let root = PathBuf::from(root_path)
                .canonicalize()
                .unwrap_or_else(|_| PathBuf::from(root_path));
            roots.insert(collection.to_string(), root);
        }
    }
    Ok(roots)
}

async fn collect_collection_samples_from_media_index(
    db_dir: &Path,
) -> Result<HashMap<String, Vec<String>>> {
    let db = lancedb::connect(db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let table = db.open_table(MEDIA_INDEX_TABLE).execute().await?;
    let stream = table
        .query()
        .select(Select::columns(&["file_name"]))
        .execute()
        .await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;

    let mut samples: HashMap<String, Vec<String>> = HashMap::new();
    for batch in &batches {
        let file_names = string_col(batch, "file_name")?;
        for row in 0..batch.num_rows() {
            if file_names.is_null(row) {
                continue;
            }
            let file_name = file_names.value(row);
            let Some((collection, rel)) = file_name.split_once('/') else {
                continue;
            };
            let rel = rel.trim_start_matches('/').to_string();
            if rel.is_empty() {
                continue;
            }
            let bucket = samples.entry(collection.to_string()).or_default();
            if bucket.len() < 16 && !bucket.contains(&rel) {
                bucket.push(rel);
            }
        }
    }
    Ok(samples)
}

pub(crate) fn discover_collection_roots_from_samples(
    db_dir: &Path,
    samples: &HashMap<String, Vec<String>>,
) -> HashMap<String, PathBuf> {
    let candidates = candidate_root_dirs(db_dir);
    let mut roots = HashMap::new();

    for (collection, rel_samples) in samples {
        let mut best_path: Option<PathBuf> = None;
        let mut best_hits = 0usize;
        for candidate in &candidates {
            let hits = rel_samples
                .iter()
                .filter(|rel| candidate.join(rel.as_str()).exists())
                .count();
            if hits > best_hits {
                best_hits = hits;
                best_path = Some(candidate.clone());
            }
        }
        if best_hits > 0 {
            if let Some(path) = best_path {
                roots.insert(collection.clone(), path);
            }
        }
    }
    roots
}

async fn write_collection_roots_table(
    db_dir: &Path,
    roots: &HashMap<String, PathBuf>,
) -> Result<()> {
    if roots.is_empty() {
        return Ok(());
    }

    let db = lancedb::connect(db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let table_names = db.table_names().execute().await?;
    let table_exists = table_names
        .iter()
        .any(|name| name == COLLECTION_ROOTS_TABLE);

    let mut rows: Vec<(String, String)> = roots
        .iter()
        .map(|(collection, root)| (collection.clone(), root.to_string_lossy().to_string()))
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));

    let collection_ids: Vec<String> = rows.iter().map(|(id, _)| id.clone()).collect();
    let root_paths: Vec<String> = rows.iter().map(|(_, path)| path.clone()).collect();
    let batch = RecordBatch::try_from_iter(vec![
        (
            "collection_id",
            Arc::new(StringArray::from(collection_ids)) as ArrayRef,
        ),
        (
            "root_path",
            Arc::new(StringArray::from(root_paths)) as ArrayRef,
        ),
    ])?;
    let schema = batch.schema();
    let batches = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);

    if table_exists {
        let table = db.open_table(COLLECTION_ROOTS_TABLE).execute().await?;
        table
            .add(Box::new(batches))
            .mode(AddDataMode::Overwrite)
            .execute()
            .await?;
    } else {
        db.create_table(COLLECTION_ROOTS_TABLE, Box::new(batches))
            .execute()
            .await?;
    }
    Ok(())
}

pub(crate) fn load_or_discover_db_roots(db_dir: &Path) -> HashMap<String, PathBuf> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build();
    let Ok(runtime) = runtime else {
        return HashMap::new();
    };

    runtime.block_on(async {
        let mut roots = load_collection_roots_from_table(db_dir)
            .await
            .unwrap_or_default();

        let samples = collect_collection_samples_from_media_index(db_dir)
            .await
            .unwrap_or_default();
        let discovered = discover_collection_roots_from_samples(db_dir, &samples);

        let mut changed = false;
        for (collection, discovered_root) in discovered {
            match roots.get(&collection) {
                Some(existing_root) => {
                    if !existing_root.exists() && discovered_root.exists() {
                        roots.insert(collection, discovered_root);
                        changed = true;
                    }
                }
                None => {
                    roots.insert(collection, discovered_root);
                    changed = true;
                }
            }
        }

        if changed {
            let _ = write_collection_roots_table(db_dir, &roots).await;
        }
        roots
    })
}

pub(crate) fn get_db_roots() -> HashMap<String, PathBuf> {
    static ROOTS_CACHE: std::sync::OnceLock<
        std::sync::Mutex<(Instant, HashMap<String, PathBuf>, bool)>,
    > = std::sync::OnceLock::new();

    let cache = ROOTS_CACHE.get_or_init(|| {
        std::sync::Mutex::new((
            Instant::now() - Duration::from_secs(3600),
            HashMap::new(),
            false,
        ))
    });

    let mut should_refresh = false;
    let roots = {
        let mut guard = match cache.lock() {
            Ok(guard) => guard,
            Err(_) => return HashMap::new(),
        };

        let refresh_after = if guard.1.is_empty() {
            Duration::from_secs(2)
        } else {
            Duration::from_secs(60)
        };
        if guard.0.elapsed() >= refresh_after && !guard.2 {
            guard.2 = true;
            should_refresh = true;
        }

        guard.1.clone()
    };

    if should_refresh {
        let cache_ref = cache;
        std::thread::spawn(move || {
            let db_dir = get_db_dir();
            let fresh = load_or_discover_db_roots(&db_dir);
            if let Ok(mut guard) = cache_ref.lock() {
                if !fresh.is_empty() || guard.1.is_empty() {
                    guard.1 = fresh;
                }
                guard.0 = Instant::now();
                guard.2 = false;
            }
        });
    }

    roots
}

pub(crate) fn file_matches_folder(
    file_name: &str,
    folder: &str,
    db_roots: &HashMap<String, PathBuf>,
) -> bool {
    let folder = folder.trim();
    if folder.is_empty() {
        return true;
    }
    let normalized_file = file_name.replace('\\', "/").to_lowercase();
    let normalized_folder = folder.replace('\\', "/").to_lowercase();
    let folder_path = Path::new(folder);
    let is_path_like = normalized_folder.contains('/')
        || normalized_folder.contains('\\')
        || folder_path.is_absolute();

    if !is_path_like {
        let rel_segments: Vec<&str> = normalized_file.split('/').collect();
        if rel_segments.len() > 2 {
            for segment in &rel_segments[1..rel_segments.len() - 1] {
                if segment.contains(&normalized_folder) {
                    return true;
                }
            }
        }
        if let Ok(source_path) = resolve_source_path(db_roots, file_name) {
            let source_segments: Vec<String> = source_path
                .components()
                .map(|component| component.as_os_str().to_string_lossy().to_lowercase())
                .collect();
            if source_segments.len() > 1 {
                for segment in &source_segments[..source_segments.len() - 1] {
                    if segment.contains(&normalized_folder) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    let folder_prefix = normalized_folder.trim_end_matches('/');
    if normalized_file == folder_prefix
        || normalized_file.starts_with(&format!("{folder_prefix}/"))
        || normalized_file.contains(&format!("/{folder_prefix}/"))
    {
        return true;
    }
    if let Ok(source_path) = resolve_source_path(db_roots, file_name) {
        let source_str = source_path
            .to_string_lossy()
            .replace('\\', "/")
            .to_lowercase();
        if source_str == folder_prefix || source_str.starts_with(&format!("{folder_prefix}/")) {
            return true;
        }
    }
    false
}

pub(crate) fn text_edit_enter_pressed(response: &egui::Response) -> bool {
    let owns_enter = response.has_focus() || response.lost_focus();
    owns_enter
        && response.ctx.input(|input| {
            input.events.iter().any(|event| {
                matches!(
                    event,
                    egui::Event::Key {
                        key: egui::Key::Enter,
                        pressed: true,
                        repeat: false,
                        ..
                    }
                )
            })
        })
}

pub(crate) fn bounded_edit_distance(a: &str, b: &str, max_distance: usize) -> Option<usize> {
    if a.len().abs_diff(b.len()) > max_distance {
        return None;
    }
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    let mut current = vec![0; b.len() + 1];
    for (row, a_byte) in a.bytes().enumerate() {
        current[0] = row + 1;
        let mut row_min = current[0];
        for (col, b_byte) in b.bytes().enumerate() {
            current[col + 1] = (previous[col + 1] + 1)
                .min(current[col] + 1)
                .min(previous[col] + usize::from(a_byte != b_byte));
            row_min = row_min.min(current[col + 1]);
        }
        if row_min > max_distance {
            return None;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    (previous[b.len()] <= max_distance).then_some(previous[b.len()])
}

pub(crate) fn fuzzy_path_component_matches(query: &str, candidate: &str) -> bool {
    if candidate.contains(query) || query.contains(candidate) {
        return true;
    }
    let max_distance = if query.len() >= 12 {
        2
    } else if query.len() >= 5 {
        1
    } else {
        0
    };
    max_distance > 0 && bounded_edit_distance(query, candidate, max_distance).is_some()
}

pub(crate) fn partial_path_matches(query: &str, candidate: &str) -> bool {
    if candidate.contains(query) {
        return true;
    }
    let query_parts: Vec<&str> = query.split('/').filter(|part| !part.is_empty()).collect();
    let candidate_parts: Vec<&str> = candidate
        .split('/')
        .filter(|part| !part.is_empty())
        .collect();
    if query_parts.is_empty() || query_parts.len() > candidate_parts.len() {
        return false;
    }
    candidate_parts.windows(query_parts.len()).any(|window| {
        query_parts
            .iter()
            .zip(window)
            .all(|(query_part, candidate_part)| {
                fuzzy_path_component_matches(query_part, candidate_part)
            })
    })
}

pub(crate) fn resolve_media_path(
    roots: &HashMap<String, PathBuf>,
    db_dir: &Path,
    file_name: &str,
    timestamp_sec: f32,
) -> Result<PathBuf> {
    let source = resolve_source_path(roots, file_name)?;
    let (_collection, rel) = file_name
        .split_once('/')
        .ok_or_else(|| anyhow!("file_name does not contain collection id"))?;
    let rel_path = Path::new(rel);
    if is_video_path(&source) {
        let (collection, _) = file_name
            .split_once('/')
            .ok_or_else(|| anyhow!("file_name does not contain collection id"))?;
        let root = roots
            .get(collection)
            .cloned()
            .ok_or_else(|| anyhow!("no collection-root for {collection}"))?;
        if let Some(still) = resolve_video_still(&root, db_dir, rel_path, timestamp_sec)? {
            return Ok(still);
        }
    }
    Ok(source)
}

pub(crate) fn resolve_video_still(
    root: &Path,
    db_dir: &Path,
    rel_path: &Path,
    timestamp_sec: f32,
) -> Result<Option<PathBuf>> {
    let scene_dir_name = format!(
        "{}-video",
        root.file_name().and_then(|x| x.to_str()).unwrap_or("video")
    );
    let mut scene_roots = Vec::with_capacity(2);
    scene_roots.push(db_dir.join(&scene_dir_name));
    if let Some(parent) = root.parent() {
        scene_roots.push(parent.join(&scene_dir_name));
    }
    let parent = rel_path.parent().unwrap_or_else(|| Path::new(""));
    let stem = rel_path
        .file_stem()
        .and_then(|x| x.to_str())
        .unwrap_or("video");
    let suffix = rel_path
        .extension()
        .and_then(|x| x.to_str())
        .map(|x| format!("_{}", x.to_ascii_lowercase()))
        .unwrap_or_default();
    for scene_root in scene_roots {
        let video_dir = scene_root.join(parent).join(format!("{stem}{suffix}"));
        let manifest_path = video_dir.join("manifest.json");
        let data = match std::fs::read_to_string(&manifest_path) {
            Ok(data) => data,
            Err(_) => continue,
        };
        let manifest: Value = serde_json::from_str(&data)?;
        let frames = manifest
            .get("frames")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("manifest has no frames"))?;
        let mut best: Option<(f32, PathBuf)> = None;
        for frame in frames {
            let ts = frame
                .get("timestamp_sec")
                .and_then(Value::as_f64)
                .unwrap_or(0.0) as f32;
            let image_file = match frame.get("image_file").and_then(Value::as_str) {
                Some(value) => value,
                None => continue,
            };
            let distance = (ts - timestamp_sec).abs();
            let path = video_dir.join(image_file);
            if best
                .as_ref()
                .map(|(best_dist, _)| distance < *best_dist)
                .unwrap_or(true)
            {
                best = Some((distance, path));
            }
        }
        if let Some((_, path)) = best {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

pub(crate) fn resolve_source_path(
    roots: &HashMap<String, PathBuf>,
    file_name: &str,
) -> Result<PathBuf> {
    let (collection, rel) = file_name
        .split_once('/')
        .ok_or_else(|| anyhow!("file_name does not contain collection id"))?;
    let root = roots
        .get(collection)
        .cloned()
        .ok_or_else(|| anyhow!("no collection-root for {collection}"))?;
    Ok(root.join(Path::new(rel)))
}

pub(crate) fn db_filename_from_video_still_path(path: &Path) -> Option<String> {
    let path_str = path.to_string_lossy();
    if !path_str.contains("-video") && !path_str.contains("lancedb") {
        return None;
    }

    let db_dir_buf = get_db_dir();
    let db_dir = db_dir_buf.as_path();

    let rel = path
        .strip_prefix(db_dir)
        .ok()
        .map(|p| p.to_path_buf())
        .or_else(|| {
            let canon_path = path.canonicalize().ok()?;
            let canon_db = db_dir.canonicalize().ok()?;
            canon_path
                .strip_prefix(canon_db)
                .ok()
                .map(|p| p.to_path_buf())
        })?;

    let components: Vec<_> = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();

    if components.len() < 3 {
        return None;
    }

    let first = &components[0];
    if !first.ends_with("-video") {
        return None;
    }
    let collection_id = &first[..first.len() - "-video".len()];

    let video_dir_name = &components[components.len() - 2];
    let (stem, ext) = video_dir_name.rsplit_once('_')?;

    let mut rel_parts = Vec::new();
    for part in &components[1..components.len() - 2] {
        rel_parts.push(part.as_str());
    }
    let reconstructed_video_filename = format!("{}.{}", stem, ext);
    rel_parts.push(&reconstructed_video_filename);

    Some(format!("{}/{}", collection_id, rel_parts.join("/")))
}

pub(crate) fn open_in_dolphin_or_fallback(file_path: &Path) {
    let path = file_path.to_path_buf();
    std::thread::spawn(move || {
        let success = if let Ok(mut child) = std::process::Command::new("dolphin")
            .arg("--select")
            .arg(&path)
            .spawn()
        {
            child.wait().map(|s| s.success()).unwrap_or(false)
        } else {
            false
        };

        if !success {
            if let Some(parent) = path.parent() {
                let _ = std::process::Command::new("xdg-open").arg(parent).spawn();
            }
        }
    });
}

pub(crate) fn is_video_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|x| x.to_str())
            .map(|x| x.to_ascii_lowercase())
            .as_deref(),
        Some("mp4" | "mov" | "avi" | "mkv" | "webm" | "m4v" | "wmv" | "mpg" | "mpeg")
    )
}
