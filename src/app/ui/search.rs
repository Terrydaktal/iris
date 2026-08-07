use super::*;

impl ImageViewer {
    pub(crate) fn run_semantic_search_mode(&mut self, mode: SearchMode, ctx: &egui::Context) {
        match mode {
            SearchMode::Filename => {}
            SearchMode::Clip => self.search_clip_now(ctx),
            SearchMode::Ocr => self.search_ocr_now(),
        }
    }

    pub(crate) fn current_search_snapshot(&self) -> SearchSnapshot {
        SearchSnapshot {
            semantic_query: self.semantic_query.clone(),
            applied_filename_query: self.applied_filename_query.clone(),
            filename_search_results: self.filename_search_results.clone(),
            semantic_folder: self.semantic_folder.clone(),
            semantic_limit: self.semantic_limit,
            semantic_video_only: self.semantic_video_only,
            semantic_mode: self.semantic_mode,
            semantic_results: self.semantic_results.clone(),
            semantic_results_mode: self.semantic_results_mode,
            semantic_status: self.semantic_status.clone(),
        }
    }

    pub(crate) fn same_search_snapshot(left: &SearchSnapshot, right: &SearchSnapshot) -> bool {
        left.semantic_query == right.semantic_query
            && left.applied_filename_query == right.applied_filename_query
            && left.filename_search_results == right.filename_search_results
            && left.semantic_folder == right.semantic_folder
            && left.semantic_limit == right.semantic_limit
            && left.semantic_video_only == right.semantic_video_only
            && left.semantic_mode == right.semantic_mode
            && left.semantic_results_mode == right.semantic_results_mode
            && left.semantic_results.len() == right.semantic_results.len()
            && left
                .semantic_results
                .iter()
                .zip(&right.semantic_results)
                .all(|(left, right)| {
                    left.file_name == right.file_name
                        && left.is_video == right.is_video
                        && left.timestamp_sec.to_bits() == right.timestamp_sec.to_bits()
                        && left.media_path == right.media_path
                })
    }

    pub(crate) fn push_search_history(&mut self) {
        let snapshot = self.current_search_snapshot();
        self.search_forward_history.clear();
        // A new search starts a new navigation branch, so an image from the old
        // gallery branch must not be restored by a later forward click.
        self.gallery_image_forward = None;
        if self
            .search_history
            .last()
            .is_some_and(|last| Self::same_search_snapshot(last, &snapshot))
        {
            return;
        }

        self.search_history.push(snapshot);
        const MAX_SEARCH_HISTORY: usize = 20;
        if self.search_history.len() > MAX_SEARCH_HISTORY {
            self.search_history.remove(0);
        }
    }

    pub(crate) fn remember_gallery_image(&mut self) {
        if self.images.is_empty() || self.current_index >= self.images.len() {
            self.gallery_image_forward = None;
            return;
        }

        self.gallery_image_forward = Some(GalleryImageSnapshot {
            images: self.images.clone(),
            current_index: self.current_index,
        });
    }

    pub(crate) fn restore_gallery_image(&mut self, ctx: &egui::Context) -> bool {
        let Some(snapshot) = self.gallery_image_forward.clone() else {
            return false;
        };

        self.images = snapshot.images;
        self.current_index = snapshot
            .current_index
            .min(self.images.len().saturating_sub(1));
        self.show_grid = false;
        self.back_target_is_gallery = true;
        self.zoom = 1.0;
        self.offset = egui::Vec2::ZERO;
        self.update_current_file_info();
        self.update_side_panel_metadata_if_needed();
        ctx.request_repaint();
        true
    }

    pub(crate) fn apply_search_snapshot(&mut self, snapshot: SearchSnapshot, ctx: &egui::Context) {
        self.semantic_query = snapshot.semantic_query;
        self.applied_filename_query = snapshot.applied_filename_query;
        self.filename_search_results = snapshot.filename_search_results;
        self.semantic_folder = snapshot.semantic_folder;
        self.semantic_limit = snapshot.semantic_limit;
        self.semantic_video_only = snapshot.semantic_video_only;
        self.semantic_mode = snapshot.semantic_mode;
        self.semantic_results = snapshot.semantic_results;
        self.semantic_results_mode = snapshot.semantic_results_mode;
        self.semantic_status = snapshot.semantic_status;
        self.pending_search_request = None;
        self.pending_semantic_search_mode = None;
        self.on_demand_embed_rx = None;
        self.selected_grid_items.clear();
        self.show_grid = true;
        self.back_target_is_gallery = false;
        ctx.request_repaint();
    }

    pub(crate) fn restore_previous_search(&mut self, ctx: &egui::Context) -> bool {
        let Some(snapshot) = self.search_history.pop() else {
            return false;
        };

        let current = self.current_search_snapshot();
        if !Self::same_search_snapshot(&current, &snapshot) {
            self.search_forward_history.push(current);
        }
        self.apply_search_snapshot(snapshot, ctx);
        true
    }

    pub(crate) fn restore_next_search(&mut self, ctx: &egui::Context) -> bool {
        let Some(snapshot) = self.search_forward_history.pop() else {
            return false;
        };

        let current = self.current_search_snapshot();
        if !Self::same_search_snapshot(&current, &snapshot) {
            self.search_history.push(current);
        }
        self.apply_search_snapshot(snapshot, ctx);
        true
    }

    pub(crate) fn apply_filename_search(&mut self) {
        self.applied_filename_query = self.semantic_query.trim().to_string();
        if self.applied_filename_query.is_empty() {
            self.filename_search_results = None;
            self.semantic_status = "Filename filter cleared.".to_string();
            return;
        }

        let query = self
            .applied_filename_query
            .to_lowercase()
            .replace('\\', "/");
        let query_is_path = query.contains('/');
        let query_basename = query.rsplit('/').next().filter(|name| name.contains('.'));
        let roots = get_db_roots();
        let mut matches = Vec::new();
        for (index, path) in self.recursive_images.iter().enumerate() {
            let matched = if query_is_path {
                if query_basename.is_some_and(|query_name| {
                    !path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.eq_ignore_ascii_case(query_name))
                }) {
                    false
                } else {
                    let physical_path = path.to_string_lossy().replace('\\', "/").to_lowercase();
                    if partial_path_matches(&query, &physical_path) {
                        true
                    } else if let Some(db_name) = self.resolve_db_filename(path) {
                        let db_name = db_name.to_lowercase();
                        let relative_name = db_name
                            .split_once('/')
                            .map(|(_, rel)| rel)
                            .unwrap_or(&db_name);
                        partial_path_matches(&query, &db_name)
                            || partial_path_matches(&query, relative_name)
                            || db_name.split_once('/').is_some_and(|(collection, rel)| {
                                roots.get(collection).is_some_and(|root| {
                                    let full_path = root
                                        .join(rel)
                                        .to_string_lossy()
                                        .replace('\\', "/")
                                        .to_lowercase();
                                    partial_path_matches(&query, &full_path)
                                })
                            })
                    } else {
                        false
                    }
                }
            } else {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.to_lowercase().contains(&query))
            };
            if matched {
                matches.push(index);
            }
        }

        matches.reverse();

        self.semantic_status = format!(
            "Filename search found {} item(s) for {}.",
            matches.len(),
            self.applied_filename_query
        );
        self.filename_search_results = Some(matches);
    }

    pub(crate) fn submit_semantic_search(&mut self, ctx: &egui::Context) {
        self.push_search_history();
        if self.semantic_mode == SearchMode::Filename {
            self.apply_filename_search();
            return;
        }

        self.pending_semantic_search_mode = Some(self.semantic_mode);
        self.semantic_results.clear();
        self.semantic_results_mode = None;
        let mode_label = match self.semantic_mode {
            SearchMode::Clip => "CLIP",
            SearchMode::Ocr => "OCR",
            SearchMode::Filename => "Filename",
        };
        self.semantic_status = format!(
            "Starting {mode_label} search for \"{}\"...",
            self.semantic_query.trim()
        );
        ctx.request_repaint();

        if self.semantic_mode == SearchMode::Ocr
            && self.db_loaded
            && !self.db_supplemental_loaded
            && !self.db_supplemental_loading
        {
            self.pending_semantic_search_mode = None;
            self.semantic_status =
                "OCR search is unavailable because supplemental database loading failed."
                    .to_string();
            return;
        }
        if !self.db_loaded
            || (self.semantic_mode == SearchMode::Ocr && !self.db_supplemental_loaded)
        {
            self.semantic_status = format!(
                "Loading AI DB for {mode_label} search of \"{}\"...",
                self.semantic_query.trim()
            );
            if self.db_failed {
                self.db_failed = false;
            }
            if !self.db_loading && !self.db_loaded {
                self.start_lazy_db_load(ctx);
            }
            return;
        }

        let mode = self
            .pending_semantic_search_mode
            .take()
            .unwrap_or(self.semantic_mode);
        self.run_semantic_search_mode(mode, ctx);
    }

    pub(crate) fn default_semantic_folder(&self) -> PathBuf {
        if self.open_target_is_dir {
            self.open_target.clone()
        } else {
            self.open_target
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf()
        }
    }

    pub(crate) fn effective_semantic_folder(&self) -> String {
        let trimmed = self.semantic_folder.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
        String::new()
    }

    pub(crate) fn folder_has_db_mappings(&self, folder: &str) -> bool {
        let roots = get_db_roots();
        if roots.is_empty() {
            return false;
        }
        let trimmed = folder.trim();
        if trimmed.is_empty() {
            return roots
                .values()
                .any(|col_path| path_matches_db_root(&self.default_semantic_folder(), col_path));
        }
        let normalized = trimmed.replace('\\', "/");
        if let Some((collection_id, rel)) = normalized.split_once('/') {
            if roots.contains_key(collection_id) && !rel.trim_matches('/').is_empty() {
                return true;
            }
        }
        let folder_path = Path::new(trimmed);
        if !normalized.contains('/') && !normalized.contains('\\') && !folder_path.is_absolute() {
            return true;
        }
        let folder_path = Path::new(trimmed);
        roots.values().any(|col_path| {
            path_matches_db_root(folder_path, col_path)
                || path_matches_db_root(col_path, folder_path)
        })
    }

    pub(crate) fn clip_query_to_pending_request(
        &self,
        query: &str,
    ) -> Option<PendingSearchRequest> {
        let mut raw = query.trim();
        if let Some(rest) = raw.strip_prefix("Current:") {
            raw = rest.trim();
        } else if let Some(rest) = raw.strip_prefix("current:") {
            raw = rest.trim();
        }
        if raw.is_empty() {
            return None;
        }

        let query_path = PathBuf::from(raw);
        let db_roots = get_db_roots();

        let request_from_db_name = |db_name: String| {
            let media_path =
                resolve_source_path(&db_roots, &db_name).unwrap_or_else(|_| query_path.clone());
            let is_video = is_video_path(&media_path) || is_video_path(Path::new(&db_name));
            PendingSearchRequest::Similar {
                db_file_name: Some(db_name),
                media_path,
                is_video,
                timestamp_sec: 0.0,
            }
        };

        if let Some(db_name) = self.resolve_db_filename(&query_path) {
            let resolved_exists = resolve_source_path(&db_roots, &db_name)
                .map(|path| path.exists())
                .unwrap_or(false);
            if resolved_exists {
                return Some(request_from_db_name(db_name));
            }
        }

        let normalized = raw
            .replace('\\', "/")
            .trim_start_matches("./")
            .trim_start_matches('/')
            .to_string();
        if let Some((collection, rel)) = normalized.split_once('/') {
            if !rel.is_empty()
                && db_roots.contains_key(collection)
                && is_supported_media_path(Path::new(rel))
            {
                let db_name = format!("{}/{}", collection, rel.trim_start_matches('/'));
                let resolved_exists = resolve_source_path(&db_roots, &db_name)
                    .map(|path| path.exists())
                    .unwrap_or(false);
                if resolved_exists {
                    return Some(request_from_db_name(db_name));
                }
            }
        }

        if query_path.exists() && is_supported_media_path(&query_path) {
            let is_video = is_video_path(&query_path);
            return Some(PendingSearchRequest::Similar {
                db_file_name: None,
                media_path: query_path,
                is_video,
                timestamp_sec: 0.0,
            });
        }

        if let Some(indices) = &self.db_indices {
            if let Some(base) = query_path.file_name() {
                let base = base.to_string_lossy().to_lowercase();
                if let Some(db_name) = indices.basename_to_db_filename.get(&base) {
                    return Some(request_from_db_name(db_name.clone()));
                }
            }
        }

        None
    }

    pub(crate) fn search_clip_now(&mut self, ctx: &egui::Context) {
        let q = self.semantic_query.trim().to_string();
        let folder_scope = self.effective_semantic_folder();
        if !self.db_supplemental_loaded {
            self.pending_semantic_search_mode = Some(SearchMode::Ocr);
            self.semantic_status = if self.db_supplemental_loading {
                "Loading OCR index in the background...".to_string()
            } else {
                "OCR index is unavailable because supplemental database loading failed.".to_string()
            };
            return;
        }
        if q.is_empty() {
            self.semantic_status = "Please enter a search phrase first.".to_string();
            self.semantic_results.clear();
            self.semantic_results_mode = None;
            return;
        }

        if let Some(request) = self.clip_query_to_pending_request(&q) {
            self.semantic_results.clear();
            self.run_search_request_now(request, ctx);
            return;
        }

        let Some(indices) = &mut self.db_indices else {
            self.semantic_status = "AI Database index is not loaded yet.".to_string();
            return;
        };

        let started = Instant::now();
        let query_vector = match indices.encoder.embed(&q) {
            Ok(vec) => vec,
            Err(err) => {
                self.semantic_status = format!("❌ Text Embedding failed: {err}");
                return;
            }
        };

        if query_vector.len() != indices.clip_index.dim {
            self.semantic_status = format!(
                "❌ Error: Query dim {} does not match index dim {}",
                query_vector.len(),
                indices.clip_index.dim
            );
            return;
        }

        let pre_limit = (self.semantic_limit.saturating_mul(6)).max(self.semantic_limit);
        let mut results = search_index(
            &indices.clip_index,
            &query_vector,
            pre_limit,
            self.semantic_video_only,
            &folder_scope,
        );
        if !self.semantic_video_only {
            results = collapse_sift_grouped_results(
                results,
                &indices.sift_root_by_file,
                self.semantic_limit,
            );
        } else {
            results.truncate(self.semantic_limit);
        }

        let db_roots = get_db_roots();
        let db_dir_buf = get_db_dir();
        let db_dir = db_dir_buf.as_path();
        for row in &mut results {
            row.media_path =
                resolve_media_path(&db_roots, db_dir, &row.file_name, row.timestamp_sec).ok();
            if let Some(path) = &row.media_path {
                self.db_filename_by_path
                    .insert(path.clone(), row.file_name.clone());
            }
        }

        let took = started.elapsed().as_millis();
        self.semantic_status = format!(
            "✓ Found {} CLIP results in {} ms across {} index vectors within {}",
            results.len(),
            took,
            indices.clip_index.entries.len(),
            folder_scope
        );
        self.semantic_results = results;
        self.semantic_results_mode = Some(SearchMode::Clip);
    }

    pub(crate) fn search_clip_from_clipboard_image(
        &mut self,
        ctx: &egui::Context,
        pasted_text: Option<&str>,
        report_no_image: bool,
    ) {
        let path = match save_clipboard_image_to_temp(pasted_text) {
            Ok(Some(path)) => path,
            Ok(None) => {
                if report_no_image {
                    self.semantic_status =
                        "Clipboard does not contain an image or image file path.".to_string();
                }
                return;
            }
            Err(err) => {
                self.semantic_status = format!("Clipboard image paste failed: {err}");
                return;
            }
        };

        let request = PendingSearchRequest::Similar {
            db_file_name: None,
            media_path: path.clone(),
            is_video: false,
            timestamp_sec: 0.0,
        };
        self.request_search_action(request, ctx);
    }

    pub(crate) fn search_ocr_now(&mut self) {
        let q = self.semantic_query.trim().to_string();
        let folder_scope = self.effective_semantic_folder();
        if q.is_empty() {
            self.semantic_status = "Please enter an OCR word or phrase first.".to_string();
            self.semantic_results.clear();
            self.semantic_results_mode = None;
            return;
        }
        let Some(indices) = &self.db_indices else {
            self.semantic_status = "AI Database index is not loaded yet.".to_string();
            return;
        };

        let started = Instant::now();
        let pre_limit = (self.semantic_limit.saturating_mul(6)).max(self.semantic_limit);
        let mut results = search_ocr_index(
            &indices.ocr_index,
            &q,
            pre_limit,
            self.semantic_video_only,
            &folder_scope,
        );
        if !self.semantic_video_only {
            results = collapse_sift_grouped_results(
                results,
                &indices.sift_root_by_file,
                self.semantic_limit,
            );
        } else {
            results.truncate(self.semantic_limit);
        }

        let db_roots = get_db_roots();
        let db_dir_buf = get_db_dir();
        let db_dir = db_dir_buf.as_path();
        for row in &mut results {
            row.media_path =
                resolve_media_path(&db_roots, db_dir, &row.file_name, row.timestamp_sec).ok();
            if let Some(path) = &row.media_path {
                self.db_filename_by_path
                    .insert(path.clone(), row.file_name.clone());
            }
        }

        let took = started.elapsed().as_millis();
        self.semantic_status = format!(
            "✓ Found {} OCR results in {} ms across {} index entries within {}",
            results.len(),
            took,
            indices.ocr_index.entries.len(),
            folder_scope
        );
        self.semantic_results = results;
        self.semantic_results_mode = Some(SearchMode::Ocr);
    }
}

impl ImageViewer {
    pub(crate) fn clip_vector_for_result(&self, row: &SearchResult) -> Option<Vec<f32>> {
        let Some(indices) = &self.db_indices else {
            return None;
        };
        let mut best: Option<(&ClipEntry, f32)> = None;
        for entry in &indices.clip_index.entries {
            if entry.file_name.as_ref() != row.file_name.as_str() {
                continue;
            }
            let dt = (entry.timestamp_sec - row.timestamp_sec).abs();
            match best {
                Some((_current, best_dt)) if dt >= best_dt => {}
                _ => best = Some((entry, dt)),
            }
        }
        best.map(|(entry, _)| entry.vector.clone())
    }

    pub(crate) fn show_most_similar_from_vector(
        &mut self,
        query_vector: Vec<f32>,
        source: Option<SearchResult>,
        label: &str,
    ) {
        let Some(indices) = &self.db_indices else {
            return;
        };
        if query_vector.len() != indices.clip_index.dim {
            self.semantic_status = format!(
                "source vector dim {} does not match index dim {}",
                query_vector.len(),
                indices.clip_index.dim
            );
            return;
        }
        let started = Instant::now();
        let pre_limit = (self.semantic_limit.saturating_mul(12)).max(self.semantic_limit + 32);
        let mut results = search_index(&indices.clip_index, &query_vector, pre_limit, false, "");
        if let Some(source) = &source {
            results.retain(|candidate| candidate.file_name != source.file_name);
        }
        results =
            collapse_sift_grouped_results(results, &indices.sift_root_by_file, self.semantic_limit);

        let db_roots = get_db_roots();
        let db_dir_buf = get_db_dir();
        let db_dir = db_dir_buf.as_path();
        for candidate in &mut results {
            candidate.media_path = resolve_media_path(
                &db_roots,
                db_dir,
                &candidate.file_name,
                candidate.timestamp_sec,
            )
            .ok();
            if let Some(path) = &candidate.media_path {
                self.db_filename_by_path
                    .insert(path.clone(), candidate.file_name.clone());
            }
        }
        if let Some(mut source) = source {
            if source.media_path.is_none() {
                source.media_path =
                    resolve_media_path(&db_roots, db_dir, &source.file_name, source.timestamp_sec)
                        .ok();
            }
            if let Some(source_path) = &source.media_path {
                self.db_filename_by_path
                    .insert(source_path.clone(), source.file_name.clone());
                results.retain(|candidate| candidate.media_path.as_ref() != Some(source_path));
            }
            source.score = 1.0;
            results.insert(0, source);
            results.truncate(self.semantic_limit);
        }
        for (idx, row) in results.iter_mut().enumerate() {
            row.rank = idx + 1;
        }
        let took = started.elapsed().as_millis();
        self.semantic_status = format!(
            "✓ Found {} CLIP-similar results in {} ms for {}",
            results.len(),
            took,
            label
        );
        self.semantic_results = results;
        self.semantic_results_mode = Some(SearchMode::Clip);
    }

    pub(crate) fn show_most_similar_clip(&mut self, row: &SearchResult) {
        let Some(query_vector) = self.clip_vector_for_result(row) else {
            self.semantic_status = format!("no CLIP vector found for {}", row.file_name);
            return;
        };
        self.show_most_similar_from_vector(query_vector, Some(row.clone()), &row.file_name);
    }

    pub(crate) fn face_vectors_for_file(
        indices: &DatabaseIndices,
        file_name: &str,
    ) -> Vec<Vec<f32>> {
        indices
            .face_index
            .entries
            .iter()
            .filter(|entry| entry.file_name.as_ref() == file_name)
            .map(|entry| entry.vector.clone())
            .collect()
    }

    pub(crate) fn related_files_for_face_seed(
        indices: &DatabaseIndices,
        file_name: &str,
    ) -> Vec<String> {
        let mut related = Vec::new();
        let mut seen = HashSet::new();
        if seen.insert(file_name.to_string()) {
            related.push(file_name.to_string());
        }

        let root = if let Some(canonical) = indices.sift_root_by_file.get(file_name) {
            canonical.clone()
        } else {
            file_name.to_string()
        };

        if let Some(members) = indices.sift_members_by_root.get(root.as_str()) {
            for member in members {
                if seen.insert(member.clone()) {
                    related.push(member.clone());
                }
                if let Some(children) = indices.similar_by_master.get(member.as_str()) {
                    for child in children {
                        if !child.is_video && seen.insert(child.file_name.clone()) {
                            related.push(child.file_name.clone());
                        }
                    }
                }
            }
        } else if let Some(children) = indices.similar_by_master.get(file_name) {
            for child in children {
                if !child.is_video && seen.insert(child.file_name.clone()) {
                    related.push(child.file_name.clone());
                }
            }
        }

        related
    }

    pub(crate) fn query_face_vectors_for_seed(&self, file_name: &str) -> Vec<Vec<f32>> {
        let Some(indices) = &self.db_indices else {
            return Vec::new();
        };
        let related_files = Self::related_files_for_face_seed(indices, file_name);
        let mut query_faces = Vec::new();
        for related in &related_files {
            query_faces.extend(Self::face_vectors_for_file(indices, related));
        }
        query_faces
    }

    pub(crate) fn show_more_of_this_person_with_vectors(
        &mut self,
        query_faces: Vec<Vec<f32>>,
        label: &str,
    ) {
        let Some(indices) = &self.db_indices else {
            return;
        };
        if query_faces.is_empty() {
            self.semantic_status = format!("No face embeddings available for {label}");
            self.semantic_results = Vec::new();
            self.semantic_results_mode = Some(SearchMode::Clip);
            return;
        }
        let started = Instant::now();
        let mut results =
            search_face_index(&indices.face_index, &query_faces, 500, FACE_MATCH_MIN_SCORE);
        results = collapse_sift_grouped_results(results, &indices.sift_root_by_file, 500);

        let db_roots = get_db_roots();
        let db_dir_buf = get_db_dir();
        let db_dir = db_dir_buf.as_path();
        for row in &mut results {
            row.media_path =
                resolve_media_path(&db_roots, db_dir, &row.file_name, row.timestamp_sec).ok();
            if let Some(path) = &row.media_path {
                self.db_filename_by_path
                    .insert(path.clone(), row.file_name.clone());
            }
        }
        let took = started.elapsed().as_millis();
        self.semantic_status = format!(
            "✓ Found {} person results in {} ms using {} query face vector(s) for {}",
            results.len(),
            took,
            query_faces.len(),
            label
        );
        self.semantic_results = results;
        self.semantic_results_mode = Some(SearchMode::Clip);
    }

    pub(crate) fn label_for_request(request: &PendingSearchRequest) -> String {
        match request {
            PendingSearchRequest::Similar {
                db_file_name,
                media_path,
                ..
            }
            | PendingSearchRequest::Person {
                db_file_name,
                media_path,
                ..
            } => db_file_name.clone().unwrap_or_else(|| {
                if is_clipboard_image_path(media_path) {
                    "clipboard image".to_string()
                } else {
                    media_path.to_string_lossy().to_string()
                }
            }),
        }
    }

    pub(crate) fn inspect_path_for_request(&self, request: &PendingSearchRequest) -> PathBuf {
        match request {
            PendingSearchRequest::Similar {
                media_path,
                is_video,
                ..
            }
            | PendingSearchRequest::Person {
                media_path,
                is_video,
                ..
            } => {
                let resolved = self.resolve_actual_path(media_path);
                if *is_video {
                    self.get_thumbnail_path(&resolved)
                } else {
                    resolved
                }
            }
        }
    }

    pub(crate) fn start_on_demand_embedding_request(
        &mut self,
        request: PendingSearchRequest,
        need_clip: bool,
        need_faces: bool,
        ctx: &egui::Context,
    ) {
        if self.on_demand_embed_rx.is_some() {
            return;
        }
        let image_path = self.inspect_path_for_request(&request);
        let (tx, rx) = std::sync::mpsc::channel::<Result<OnDemandEmbedResult, String>>();
        self.on_demand_embed_rx = Some(rx);
        let request_clone = request.clone();
        let ctx_clone = ctx.clone();
        std::thread::spawn(move || {
            let result = compute_on_demand_embeddings(&image_path, need_clip, need_faces)
                .map(|(clip_vector, face_vectors)| OnDemandEmbedResult {
                    request: request_clone,
                    clip_vector,
                    face_vectors,
                })
                .map_err(|e| e.to_string());
            let _ = tx.send(result);
            ctx_clone.request_repaint();
        });
    }

    pub(crate) fn run_search_request_now(
        &mut self,
        request: PendingSearchRequest,
        ctx: &egui::Context,
    ) {
        match &request {
            PendingSearchRequest::Similar {
                db_file_name,
                media_path,
                is_video,
                timestamp_sec,
            } => {
                if let Some(db_name) = db_file_name {
                    let row = SearchResult {
                        rank: 0,
                        score: 1.0,
                        file_name: db_name.clone(),
                        is_video: *is_video,
                        timestamp_sec: *timestamp_sec,
                        media_path: Some(media_path.clone()),
                        ocr_term_hits: 0,
                        ocr_query_terms: 0,
                        ocr_phrase_query: false,
                    };
                    if self.clip_vector_for_result(&row).is_some() {
                        self.show_most_similar_clip(&row);
                        return;
                    }
                }
                self.semantic_status = format!(
                    "Computing CLIP embedding on the fly for {}...",
                    Self::label_for_request(&request)
                );
                self.start_on_demand_embedding_request(request, true, false, ctx);
            }
            PendingSearchRequest::Person { db_file_name, .. } => {
                if let Some(db_name) = db_file_name {
                    let query_faces = self.query_face_vectors_for_seed(db_name);
                    if !query_faces.is_empty() {
                        self.show_more_of_this_person_with_vectors(query_faces, db_name);
                        return;
                    }
                }
                self.semantic_status = format!(
                    "Computing face embeddings on the fly for {}...",
                    Self::label_for_request(&request)
                );
                self.start_on_demand_embedding_request(request, false, true, ctx);
            }
        }
    }

    pub(crate) fn request_search_action(
        &mut self,
        request: PendingSearchRequest,
        ctx: &egui::Context,
    ) {
        self.push_search_history();
        // Related-image searches are gallery searches. Leave the current image
        // viewer immediately so the pending and completed results are visible.
        self.show_grid = true;
        self.back_target_is_gallery = false;
        let label = Self::label_for_request(&request);
        self.semantic_mode = SearchMode::Clip;
        self.semantic_query = label.clone();
        self.semantic_results.clear();
        self.semantic_results_mode = None;
        self.pending_search_request = Some(request.clone());

        let needs_supplemental = matches!(&request, PendingSearchRequest::Person { .. });
        if needs_supplemental
            && self.db_loaded
            && !self.db_supplemental_loaded
            && !self.db_supplemental_loading
        {
            self.pending_search_request = None;
            self.semantic_status =
                "Person search is unavailable because supplemental database loading failed."
                    .to_string();
            return;
        }
        if !self.db_loaded || (needs_supplemental && !self.db_supplemental_loaded) {
            self.semantic_status =
                format!("Loading AI DB to search for matches related to {label}...");
            if !self.db_failed && !self.db_loading {
                self.start_lazy_db_load(ctx);
            }
            return;
        }

        self.pending_search_request = None;
        self.run_search_request_now(request, ctx);
    }
}
