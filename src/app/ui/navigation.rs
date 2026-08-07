use super::*;

impl ImageViewer {
    pub(crate) fn start_recursive_scan(&mut self) {
        self.grid_loading = true;
        self.recursive_images.clear();
        self.recursive_video_indices.clear();
        self.applied_filename_query.clear();
        self.filename_search_results = None;
        self.thumbnail_textures.clear();
        self.thumbnail_loading.clear();
        self.thumbnail_failed.clear();
        self.thumbnail_active_threads = 0;
        self.video_duration_cache.borrow_mut().clear();
        self.video_duration_loading.borrow_mut().clear();

        let start_dir = if self.open_target_is_dir {
            self.open_target.clone()
        } else {
            self.open_target
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf()
        };

        let (tx, rx) = std::sync::mpsc::channel();
        self.recursive_rx = Some(rx);

        std::thread::spawn(move || {
            let start_dir_canon = start_dir.canonicalize().unwrap_or(start_dir);
            let mut visited = std::collections::HashSet::new();
            collect_images_recursive(&start_dir_canon, &tx, &mut visited);
        });
    }

    pub(crate) fn is_comparison_mode(&self) -> bool {
        self.comparison_paths
            .as_ref()
            .is_some_and(|paths| paths.len() >= 2)
    }

    pub(crate) fn clear_comparison_alignment(&mut self) {
        self.comparison_aligned_paths.clear();
        self.comparison_alignment_status.clear();
        self.sift_align_all_rx = None;
        self.sift_align_all_running = false;
        if let Some(output_dir) = self.comparison_alignment_temp_dir.take() {
            let _ = std::fs::remove_dir_all(output_dir);
        }
    }

    pub(crate) fn clear_comparison_mode(&mut self) {
        self.clear_comparison_alignment();
        self.comparison_paths = None;
        self.comparison_view_states.clear();
        self.comparison_sync_view = false;
    }

    pub(crate) fn apply_comparison_view_state_to_all(&mut self) {
        if !self.is_comparison_mode() || !self.comparison_sync_view {
            return;
        }
        let state = ImageViewState {
            zoom: self.zoom,
            offset: self.offset,
        };
        for path in self.images.clone() {
            self.comparison_view_states.insert(path, state);
        }
    }

    pub(crate) fn save_comparison_view_state(&mut self) {
        if !self.is_comparison_mode() {
            return;
        }
        let Some(path) = self.images.get(self.current_index).cloned() else {
            return;
        };
        let state = ImageViewState {
            zoom: self.zoom,
            offset: self.offset,
        };
        if self.comparison_sync_view {
            for path in self.images.clone() {
                self.comparison_view_states.insert(path, state);
            }
        } else {
            self.comparison_view_states.insert(path, state);
        }
    }

    pub(crate) fn switch_comparison_image(&mut self, index: usize) {
        if !self.is_comparison_mode() || self.images.is_empty() {
            return;
        }
        self.save_comparison_view_state();
        self.current_index = index.min(self.images.len().saturating_sub(1));
        if self.comparison_sync_view {
            self.update_current_file_info();
            self.update_side_panel_metadata_if_needed();
            return;
        }
        let path = self.images[self.current_index].clone();
        let state = self
            .comparison_view_states
            .get(&path)
            .copied()
            .unwrap_or(ImageViewState {
                zoom: 1.0,
                offset: egui::Vec2::ZERO,
            });
        self.zoom = state.zoom;
        self.offset = state.offset;
        self.update_current_file_info();
        self.update_side_panel_metadata_if_needed();
    }

    pub(crate) fn open_comparison_paths(&mut self, paths: Vec<PathBuf>, ctx: &egui::Context) {
        let mut unique_paths = Vec::new();
        for path in paths.into_iter().take(6) {
            let path = path.canonicalize().unwrap_or(path);
            if !path.is_file() || !is_supported_media_path(&path) || unique_paths.contains(&path) {
                continue;
            }
            unique_paths.push(path);
        }
        if unique_paths.len() < 2 {
            if let Some(path) = unique_paths.into_iter().next() {
                self.open_image_path(path);
            }
            return;
        }

        self.image_editor = None;
        self.clear_comparison_alignment();
        self.compare_target = None;
        self.sift_pair_overlay = None;
        self.selected_grid_items.clear();
        self.close_side_panel(ctx);
        self.show_home_page = false;
        self.show_grid = false;
        self.back_target_is_gallery = false;
        self.gallery_image_forward = None;
        self.open_target = unique_paths[0].clone();
        self.open_target_is_dir = false;
        self.flat_loading = false;
        self.flat_refresh_in_flight = false;
        self.flat_directory_mtime = None;
        if let Ok(mut lock) = self.flat_images_shared.lock() {
            *lock = None;
        }
        self.images = unique_paths.clone();
        self.current_index = 0;
        self.comparison_paths = Some(unique_paths.clone());
        self.comparison_sync_view = false;
        self.comparison_view_states = unique_paths
            .into_iter()
            .map(|path| {
                (
                    path,
                    ImageViewState {
                        zoom: 1.0,
                        offset: egui::Vec2::ZERO,
                    },
                )
            })
            .collect();
        self.zoom = 1.0;
        self.offset = egui::Vec2::ZERO;
        self.viewer_rotation_path = None;
        self.update_current_file_info();
        self.update_side_panel_metadata_if_needed();
        ctx.request_repaint();
    }

    pub(crate) fn open_image_path(&mut self, path: PathBuf) {
        self.gallery_image_forward = None;
        self.open_path(path, None);
    }

    pub(crate) fn open_comparison_path_dialog(&mut self) {
        self.comparison_path_input.clear();
        self.comparison_path_dialog_open = true;
    }

    pub(crate) fn show_comparison_path_dialog(&mut self, ctx: &egui::Context) {
        if !self.comparison_path_dialog_open {
            return;
        }

        let mut dialog_open = true;
        let mut submit = false;
        let mut cancel = false;
        egui::Window::new("Compare Paths")
            .collapsible(false)
            .resizable(true)
            .open(&mut dialog_open)
            .show(ctx, |ui| {
                ui.label("Enter one image or video path per line (2-6 paths):");
                ui.add(
                    egui::TextEdit::multiline(&mut self.comparison_path_input)
                        .desired_width(620.0)
                        .desired_rows(6)
                        .hint_text("/path/to/first.jpg\n/path/to/second.jpg"),
                );
                ui.horizontal(|ui| {
                    if ui.button("Compare").clicked() {
                        submit = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });
            });

        if cancel {
            dialog_open = false;
        }
        self.comparison_path_dialog_open = dialog_open;
        if !submit || !dialog_open {
            return;
        }

        let paths: Vec<PathBuf> = self
            .comparison_path_input
            .lines()
            .map(str::trim)
            .filter(|path| !path.is_empty())
            .map(PathBuf::from)
            .collect();
        if !(2..=6).contains(&paths.len()) {
            self.semantic_status = "Enter between two and six paths.".to_string();
            return;
        }
        let valid_paths = paths
            .iter()
            .filter(|path| path.is_file() && is_supported_media_path(path))
            .count();
        if valid_paths < 2 {
            self.semantic_status =
                "At least two valid image or video paths are required.".to_string();
            return;
        }

        self.comparison_path_dialog_open = false;
        self.open_comparison_paths(paths, ctx);
    }

    pub(crate) fn open_file_dialog(&mut self, ctx: &egui::Context) {
        let Some(path) = rfd::FileDialog::new()
            .set_title("Open image or video")
            .add_filter(
                "Images and videos",
                &[
                    "jpg", "jpeg", "png", "bmp", "gif", "webp", "tiff", "avif", "heif", "heic",
                    "ico", "icns", "svg", "mp4", "mov", "avi", "mkv", "webm", "m4v", "wmv", "mpg",
                    "mpeg",
                ],
            )
            .pick_file()
        else {
            return;
        };

        self.image_editor = None;
        self.compare_target = None;
        self.sift_pair_overlay = None;
        self.selected_grid_items.clear();
        self.close_side_panel(ctx);
        self.show_home_page = false;
        self.show_grid = false;
        self.back_target_is_gallery = false;
        self.gallery_image_forward = None;
        self.open_image_path(path);
        ctx.request_repaint();
    }

    pub(crate) fn open_folder_path(&mut self, path: PathBuf) {
        self.gallery_image_forward = None;
        self.open_path(path, Some(true));
    }

    pub(crate) fn open_path(&mut self, path: PathBuf, known_is_dir: Option<bool>) {
        self.clear_comparison_mode();
        self.face_overlay_boxes.clear();
        let old_start_dir = if self.open_target_is_dir {
            self.open_target.clone()
        } else {
            self.open_target
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf()
        };
        let old_start_dir_norm = normalized_path_for_match(&old_start_dir);

        let path_is_dir = known_is_dir.unwrap_or_else(|| path.is_dir());
        self.open_target = path.clone();
        self.open_target_is_dir = path_is_dir;
        self.zoom = 1.0;
        self.offset = egui::Vec2::ZERO;

        let new_start_dir = if self.open_target_is_dir {
            self.open_target.clone()
        } else {
            self.open_target
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf()
        };
        let new_start_dir_norm = normalized_path_for_match(&new_start_dir);

        if old_start_dir_norm != new_start_dir_norm {
            self.recursive_images.clear();
            self.recursive_video_indices.clear();
            self.back_target_is_gallery = false;
            self.flat_directory_mtime = None;
        }

        if path_is_dir {
            self.images.clear();
            self.current_index = 0;
            self.update_current_file_info();
            self.flat_loading = false;
            self.flat_refresh_in_flight = false;
            self.flat_directory_mtime = None;
            if let Ok(mut lock) = self.flat_images_shared.lock() {
                *lock = None;
            }
        } else {
            self.images = vec![path.clone()];
            self.current_index = 0;
            self.update_current_file_info();
            self.flat_loading = true;
            self.flat_refresh_in_flight = false;
            self.flat_directory_mtime = None;
            if let Ok(mut lock) = self.flat_images_shared.lock() {
                *lock = None;
            }

            let shared = self.flat_images_shared.clone();
            let parent = path.parent().unwrap_or(Path::new(".")).to_path_buf();
            std::thread::spawn(move || {
                let parent_absolute = parent.canonicalize().unwrap_or(parent);
                let collected = collect_flat_images(&parent_absolute);
                if let Ok(mut lock) = shared.lock() {
                    *lock = Some(collected);
                }
            });
        }
    }
}
