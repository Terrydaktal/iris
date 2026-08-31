use super::*;

impl ImageViewer {
    pub(crate) fn thumbnail_worker_limit(path: &Path) -> usize {
        if is_video_path(path) {
            // Video thumbnails each launch a decoder. Keep HDD seeks and process
            // pressure bounded when a directory contains many unindexed videos.
            4
        } else {
            num_cpus::get().max(4)
        }
    }

    pub(crate) fn load_thumbnail_color_image(
        path: &Path,
        width: u32,
        height: u32,
        fill: bool,
    ) -> Result<egui::ColorImage, String> {
        let image = if is_video_path(path) {
            load_video_thumbnail(path, width, height, fill)?
        } else {
            image::open(path).map_err(|error| format!("{}: {error}", path.display()))?
        };
        let thumbnail = if fill {
            image.resize_to_fill(width, height, image::imageops::FilterType::Triangle)
        } else {
            image.thumbnail(width, height)
        };
        let size = [thumbnail.width() as usize, thumbnail.height() as usize];
        let pixels = thumbnail.to_rgba8().into_raw();
        Ok(egui::ColorImage::from_rgba_unmultiplied(size, &pixels))
    }

    pub(crate) fn get_db_filename_from_path(&self, path: &Path) -> Option<String> {
        let roots = get_db_roots();
        let path_norm = path.to_string_lossy().replace('\\', "/");

        // Fast path 0: already a DB-style relative path such as:
        //   <collection_id>/...
        // This must resolve directly to collection roots.
        let trimmed = path_norm.trim_start_matches("./").trim_start_matches('/');
        if let Some((collection, rel)) = trimmed.split_once('/') {
            if !rel.is_empty() && roots.contains_key(collection) {
                return Some(format!("{}/{}", collection, rel.trim_start_matches('/')));
            }
        }

        // Match only an actual collection-root prefix. A shared leaf folder name is
        // not enough to identify a collection and can silently map the wrong file.
        for (col_id, root_path) in &roots {
            if let Ok(rel) = path.strip_prefix(root_path) {
                let rel_str = rel.to_string_lossy().replace('\\', "/");
                return Some(format!("{}/{}", col_id, rel_str.trim_start_matches('/')));
            }
        }

        // Handle equivalent textual prefixes with different separator styles.
        let path_str = path.to_string_lossy().replace('\\', "/");
        for (col_id, root_path) in &roots {
            let root_str = root_path.to_string_lossy().replace('\\', "/");
            if path_str == root_str || path_str.starts_with(&format!("{root_str}/")) {
                let rel = &path_str[root_str.len()..];
                return Some(format!("{}/{}", col_id, rel.trim_start_matches('/')));
            }
        }

        // Tertiary: basename lookup in loaded index.
        // Only use this for non-existing/virtual paths (e.g. synthesized preview references).
        // For real existing full paths outside mapped roots, basename matching is too ambiguous.
        if !path.exists() {
            if let Some(indices) = &self.db_indices {
                if let Some(fname) = path.file_name() {
                    let base = fname.to_string_lossy().to_lowercase();
                    if let Some(resolved) = indices.basename_to_db_filename.get(&base) {
                        if resolved.len() == 1 {
                            return resolved.first().cloned();
                        }
                    }
                }
            }
        }
        None
    }

    /// Look up the database filename for a given filesystem path.
    /// Checks the cached `db_filename_by_path` map first (populated from AI search results),
    /// then falls back to the path-prefix heuristic in `get_db_filename_from_path`.
    pub(crate) fn resolve_db_filename(&self, path: &Path) -> Option<String> {
        // Fast path 1: exact match from AI search results (handles video stills, etc.)
        if let Some(name) = self.db_filename_by_path.get(path) {
            return Some(name.clone());
        }
        if let Some(name) = db_filename_from_video_still_path(path) {
            return Some(name);
        }
        // Fast path 2: Check the cache to avoid synchronous disk canonicalization and linear scans
        if let Some(cached) = self.db_filename_cache.borrow().get(path) {
            return Some(cached.clone());
        }
        // Fallback: derive from filesystem path vs collection roots
        let resolved = self.get_db_filename_from_path(path);
        // Do not cache misses: collection roots can be discovered asynchronously.
        if let Some(name) = &resolved {
            self.db_filename_cache
                .borrow_mut()
                .insert(path.to_path_buf(), name.clone());
        }
        resolved
    }

    pub(crate) fn resolve_actual_path(&self, path: &Path) -> PathBuf {
        if let Some(db_name) = self.resolve_db_filename(path) {
            let roots = get_db_roots();
            if let Ok(src_path) = resolve_source_path(&roots, &db_name) {
                return src_path;
            }
        }
        path.to_path_buf()
    }

    pub(crate) fn get_thumbnail_path(&self, path: &Path) -> PathBuf {
        if is_video_path(path) {
            // Check cache first to avoid synchronous disk I/O on UI thread
            if let Some(cached) = self.video_still_cache.borrow().get(path) {
                return cached.clone();
            }

            if let Some(file_name) = self.resolve_db_filename(path) {
                let db_dir_buf = get_db_dir();
                let db_dir = db_dir_buf.as_path();
                let db_roots = get_db_roots();
                if let Some((collection, rel)) = file_name.split_once('/') {
                    if let Some(root) = db_roots.get(collection) {
                        let rel_path = Path::new(rel);
                        if let Ok(Some(still)) = resolve_video_still(root, db_dir, rel_path, 0.0) {
                            self.video_still_cache
                                .borrow_mut()
                                .insert(path.to_path_buf(), still.clone());
                            return still;
                        }
                    }
                }
            }
        }
        path.to_path_buf()
    }

    pub(crate) fn comparison_display_path(&self, path: &Path) -> PathBuf {
        self.comparison_aligned_paths
            .get(path)
            .filter(|aligned| aligned.is_file())
            .cloned()
            .unwrap_or_else(|| self.get_thumbnail_path(path))
    }

    pub(crate) fn video_source_path_for_tile(
        &self,
        path: &Path,
        db_filename: Option<&str>,
    ) -> PathBuf {
        if let Some(db_name) = db_filename {
            let roots = get_db_roots();
            if let Ok(source) = resolve_source_path(&roots, db_name) {
                return source;
            }
        }
        self.resolve_actual_path(path)
    }

    pub(crate) fn cached_video_metadata(
        &self,
        path: &Path,
        ctx: &egui::Context,
    ) -> Option<VideoMetadata> {
        if let Some(metadata) = self.video_duration_cache.borrow().get(path) {
            return *metadata;
        }
        if self
            .video_duration_loading
            .borrow_mut()
            .insert(path.to_path_buf())
        {
            let path_clone = path.to_path_buf();
            let tx = self.video_duration_tx.clone();
            let ctx_clone = ctx.clone();
            let diagnostics = self.diagnostics.clone();
            rayon::spawn(move || {
                let task = diagnostics.task_guard("video_metadata_probe");
                let metadata = load_video_metadata(&path_clone);
                let _ = tx.send((path_clone, metadata));
                ctx_clone.request_repaint();
                task.complete();
            });
        }
        None
    }

    pub(crate) fn get_file_resolution_and_size(&self, path: &Path) -> String {
        if let Some(cached) = self.resolution_size_cache.borrow().get(path) {
            return cached.clone();
        }
        let result = file_resolution_and_size(path);
        self.resolution_size_cache
            .borrow_mut()
            .insert(path.to_path_buf(), result.clone());
        result
    }

    pub(crate) fn get_duplicate_media_info(
        &self,
        path: &Path,
        is_video: bool,
        ctx: &egui::Context,
    ) -> String {
        if !is_video {
            return self.get_file_resolution_and_size(path);
        }
        let size_modified = file_size_and_modified(path);
        let metadata = self.cached_video_metadata(path, ctx);
        let duration = metadata
            .and_then(|metadata| metadata.duration_sec)
            .map(format_video_duration)
            .unwrap_or_else(|| "n/a".to_string());
        let resolution = metadata
            .and_then(|metadata| match (metadata.width, metadata.height) {
                (Some(width), Some(height)) => Some(format!("{}x{}", width, height)),
                _ => None,
            })
            .unwrap_or_else(|| "n/a".to_string());
        format!("{} | {} | {}", duration, resolution, size_modified)
    }

    pub(crate) fn draw_thumbnail_async(&mut self, ui: &mut egui::Ui, path: &Path, side_thumb: f32) {
        let resolved_path = self.get_thumbnail_path(path);
        if let Some(texture) = self.thumbnail_textures.get(&resolved_path) {
            ui.add(
                egui::Image::from_texture(texture)
                    .max_size(egui::vec2(side_thumb, side_thumb))
                    .maintain_aspect_ratio(true),
            );
        } else if self.thumbnail_failed.contains(&resolved_path) {
            let (rect, _) =
                ui.allocate_exact_size(egui::vec2(side_thumb, side_thumb), egui::Sense::hover());
            ui.painter()
                .rect_filled(rect, 4.0, egui::Color32::from_gray(30));
            let text = if is_video_path(path) {
                "📹 Video"
            } else {
                "⚠️ Error"
            };
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                text,
                egui::FontId::proportional(10.0),
                egui::Color32::GRAY,
            );
        } else {
            let (rect, _) =
                ui.allocate_exact_size(egui::vec2(side_thumb, side_thumb), egui::Sense::hover());
            ui.painter()
                .rect_filled(rect, 4.0, egui::Color32::from_gray(40));
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "⏳ Loading",
                egui::FontId::proportional(10.0),
                egui::Color32::GRAY,
            );

            let max_threads = Self::thumbnail_worker_limit(&resolved_path);
            if !self.thumbnail_loading.contains(&resolved_path)
                && self.thumbnail_active_threads < max_threads
            {
                self.thumbnail_loading.insert(resolved_path.to_path_buf());
                self.thumbnail_active_threads += 1;
                let path_clone = resolved_path.to_path_buf();
                let thumbnail_generation = self.gallery_thumbnail_generation;
                let tx_clone = self.thumbnail_tx.clone();
                let ctx_clone = ui.ctx().clone();
                let diagnostics = self.diagnostics.clone();
                rayon::spawn(move || {
                    let task = diagnostics.task_guard("sidebar_thumbnail_decode");
                    if let Ok(color_img) =
                        Self::load_thumbnail_color_image(&path_clone, 128, 128, false)
                    {
                        let _ = tx_clone.send((thumbnail_generation, path_clone, color_img));
                        ctx_clone.request_repaint();
                    } else {
                        let empty_img = egui::ColorImage::new([0, 0], Vec::new());
                        let _ = tx_clone.send((thumbnail_generation, path_clone, empty_img));
                        ctx_clone.request_repaint();
                    }
                    task.complete();
                });
            }
        }
    }

    pub(crate) fn grouped_master_for(&self, file_name: &str, is_video: bool) -> String {
        if is_video {
            return file_name.to_string();
        }
        if let Some(indices) = &self.db_indices {
            indices
                .sift_root_by_file
                .get(file_name)
                .cloned()
                .unwrap_or_else(|| file_name.to_string())
        } else {
            file_name.to_string()
        }
    }
}
