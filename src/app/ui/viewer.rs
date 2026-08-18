use super::*;

impl ImageViewer {
    fn request_metadata_load(&mut self, load_exif: bool, load_layout: bool) {
        let Some(path) = self.images.get(self.current_index).cloned() else {
            self.metadata_generation = self.metadata_generation.wrapping_add(1);
            self.exif_data.clear();
            self.chunks.clear();
            self.side_panel_metadata_path = None;
            self.side_panel_layout_path = None;
            self.metadata_loading = false;
            self.metadata_loading_path = None;
            self.metadata_loading_exif = false;
            self.metadata_loading_layout = false;
            return;
        };
        let same_path = self.metadata_loading_path.as_ref() == Some(&path);
        let request_exif = load_exif || (same_path && self.metadata_loading_exif);
        let request_layout = load_layout || (same_path && self.metadata_loading_layout);
        if same_path
            && (!load_exif || self.metadata_loading_exif)
            && (!load_layout || self.metadata_loading_layout)
        {
            return;
        }
        self.metadata_generation = self.metadata_generation.wrapping_add(1);
        let generation = self.metadata_generation;
        self.metadata_loading = true;
        self.metadata_loading_path = Some(path.clone());
        self.metadata_loading_exif = request_exif;
        self.metadata_loading_layout = request_layout;
        if request_exif {
            self.exif_data = "Loading metadata...".to_string();
            self.side_panel_metadata_path = None;
        }
        if request_layout {
            self.chunks.clear();
            self.side_panel_layout_path = None;
        }
        let resolved_path = self.resolve_actual_path(&path);
        let inspect_path = if resolved_path.exists() {
            resolved_path
        } else {
            path.clone()
        };
        let mut state = match self.metadata_worker_queue.state.lock() {
            Ok(state) => state,
            Err(_) => return,
        };
        if state.shutdown {
            self.metadata_loading = false;
            return;
        }
        state.pending = Some(MetadataLoadRequest {
            logical_path: path,
            inspect_path,
            generation,
            load_exif: request_exif,
            load_layout: request_layout,
        });
        self.metadata_worker_queue.wake.notify_one();
    }

    pub(crate) fn update_exif(&mut self) {
        self.update_current_file_info();
        self.request_metadata_load(true, false);
    }

    pub(crate) fn update_current_file_info(&mut self) {
        if let Some(path) = self.images.get(self.current_index) {
            let resolved_path = self.resolve_actual_path(path);
            let inspect_path: &Path = if resolved_path.exists() {
                resolved_path.as_path()
            } else {
                path.as_path()
            };

            self.current_dimensions = if is_video_path(inspect_path) {
                "Video".to_string()
            } else {
                match image::image_dimensions(inspect_path) {
                    Ok((w, h)) => format!("{}x{}", w, h),
                    Err(_) => "Unknown px".to_string(),
                }
            };

            self.current_file_size = std::fs::metadata(inspect_path)
                .map(|m| {
                    let bytes = m.len();
                    if bytes >= 1_048_576 {
                        format!("{:.2} MB", bytes as f64 / 1_048_576.0)
                    } else if bytes >= 1024 {
                        format!("{:.1} KB", bytes as f64 / 1024.0)
                    } else {
                        format!("{} B", bytes)
                    }
                })
                .unwrap_or_else(|_| "Unknown size".to_string());
        } else {
            self.current_dimensions = String::new();
            self.current_file_size = String::new();
        }
    }

    pub(crate) fn editor_texture(
        ctx: &egui::Context,
        image: &image::DynamicImage,
    ) -> egui::TextureHandle {
        ctx.load_texture(
            "image_editor_preview",
            viewer_color_image_ref(image),
            egui::TextureOptions::LINEAR,
        )
    }

    pub(crate) fn trim_viewer_texture_cache(&mut self) {
        let retained_images = if self.is_comparison_mode() {
            self.images.clone()
        } else {
            self.images
                .get(self.current_index)
                .cloned()
                .into_iter()
                .collect()
        };
        let mut keep_paths = HashSet::new();
        for path in retained_images {
            keep_paths.insert(path.clone());
            keep_paths.insert(if self.is_comparison_mode() {
                self.comparison_display_path(&path)
            } else {
                self.get_thumbnail_path(&path)
            });
        }
        if let Some(compare_path) = self.compare_target.clone() {
            keep_paths.insert(compare_path.clone());
            keep_paths.insert(self.get_thumbnail_path(&compare_path));
        }
        for path in self.comparison_aligned_paths.values() {
            keep_paths.insert(path.clone());
        }
        self.viewer_textures
            .retain(|path, _| keep_paths.contains(path));
        self.viewer_texture_failed
            .retain(|path| keep_paths.contains(path));
        self.viewer_texture_revisions
            .retain(|path, _| keep_paths.contains(path));
        self.viewer_texture_retry_at
            .retain(|path, _| keep_paths.contains(path));
    }

    pub(crate) fn request_viewer_texture(
        &mut self,
        path: &Path,
        ctx: &egui::Context,
    ) -> Option<egui::TextureHandle> {
        if let Some(texture) = self.viewer_textures.get(path) {
            return Some(texture.clone());
        }
        if self.viewer_texture_failed.contains(path) {
            let retry_ready = self
                .viewer_texture_retry_at
                .get(path)
                .is_none_or(|retry_at| Instant::now() >= *retry_at);
            if !retry_ready {
                return None;
            }
            self.viewer_texture_failed.remove(path);
            self.viewer_texture_retry_at.remove(path);
        }
        if self.viewer_texture_loading.contains(path) {
            return None;
        }

        // Decode one full-size source at a time. The resulting GPU texture is capped
        // by viewer_color_image, preventing several huge files being uploaded at once.
        if self.viewer_texture_loading.is_empty() {
            let path = path.to_path_buf();
            let revision = self
                .viewer_texture_revisions
                .get(&path)
                .copied()
                .unwrap_or(0);
            self.viewer_texture_loading.insert(path.clone());
            let tx = self.viewer_texture_tx.clone();
            let ctx = ctx.clone();
            let diagnostics = self.diagnostics.clone();
            rayon::spawn(move || {
                let task = diagnostics.task_guard("viewer_texture_decode");
                let result = image::open(&path)
                    .map(viewer_color_image)
                    .map_err(|err| format!("{}: {err}", path.display()));
                let _ = tx.send((path, revision, result));
                ctx.request_repaint();
                task.complete();
            });
        }
        None
    }

    pub(crate) fn start_image_editor(&mut self, path: &Path, ctx: &egui::Context) {
        // Prefer the database-resolved path, but keep working when the item already
        // contains a valid filesystem path and the collection mapping is stale.
        let resolved_path = self.resolve_actual_path(path);
        let source_path = if resolved_path.is_file() {
            resolved_path
        } else if path.is_file() {
            path.to_path_buf()
        } else {
            resolved_path
        };
        let viewer_rotation = if self.viewer_rotation_path.as_deref() == Some(path) {
            self.viewer_rotation_quarter_turns
        } else {
            0
        };
        match image::open(&source_path) {
            Ok(mut image) => {
                image = match viewer_rotation {
                    1 => image.rotate90(),
                    2 => image.rotate180(),
                    3 => image.rotate270(),
                    _ => image,
                };
                self.image_editor = Some(ImageEditor {
                    texture: Self::editor_texture(ctx, &image),
                    source_path,
                    image,
                    crop_min: egui::pos2(0.0, 0.0),
                    crop_max: egui::pos2(1.0, 1.0),
                    crop_drag_mode: None,
                    crop_drag_origin: egui::Pos2::ZERO,
                    crop_drag_initial_min: egui::Pos2::ZERO,
                    crop_drag_initial_max: egui::pos2(1.0, 1.0),
                    status: String::new(),
                });
            }
            Err(err) => {
                self.semantic_status = format!("Unable to open image for editing: {err}");
            }
        }
        ctx.request_repaint();
    }

    pub(crate) fn rotate_editor_image(
        editor: &mut ImageEditor,
        quarter_turns: i32,
        ctx: &egui::Context,
    ) {
        let old_min = editor.crop_min;
        let old_max = editor.crop_max;
        let (new_min, new_max, rotated) = match quarter_turns.rem_euclid(4) {
            1 => (
                egui::pos2(1.0 - old_max.y, old_min.x),
                egui::pos2(1.0 - old_min.y, old_max.x),
                editor.image.rotate90(),
            ),
            2 => (
                egui::pos2(1.0 - old_max.x, 1.0 - old_max.y),
                egui::pos2(1.0 - old_min.x, 1.0 - old_min.y),
                editor.image.rotate180(),
            ),
            3 => (
                egui::pos2(old_min.y, 1.0 - old_max.x),
                egui::pos2(old_max.y, 1.0 - old_min.x),
                editor.image.rotate270(),
            ),
            _ => return,
        };
        editor.image = rotated;
        editor.crop_min = new_min;
        editor.crop_max = new_max;
        editor.crop_drag_mode = None;
        editor.crop_drag_initial_min = new_min;
        editor.crop_drag_initial_max = new_max;
        editor.texture = Self::editor_texture(ctx, &editor.image);
    }

    pub(crate) fn edited_copy_path(source: &Path) -> PathBuf {
        let stem = source
            .file_stem()
            .and_then(|part| part.to_str())
            .unwrap_or("image");
        let extension = source
            .extension()
            .and_then(|part| part.to_str())
            .unwrap_or("png");
        let parent = source.parent().unwrap_or_else(|| Path::new("."));
        let mut candidate = parent.join(format!("{stem}_edited.{extension}"));
        let mut suffix = 2;
        while candidate.exists() {
            candidate = parent.join(format!("{stem}_edited_{suffix}.{extension}"));
            suffix += 1;
        }
        candidate
    }

    pub(crate) fn save_editor_image(
        editor: &ImageEditor,
        destination: &Path,
        overwrite: bool,
    ) -> Result<image::DynamicImage> {
        let image_width = editor.image.width();
        let image_height = editor.image.height();
        let left = (editor.crop_min.x * image_width as f32).round() as u32;
        let top = (editor.crop_min.y * image_height as f32).round() as u32;
        let right = (editor.crop_max.x * image_width as f32).round() as u32;
        let bottom = (editor.crop_max.y * image_height as f32).round() as u32;
        let width = right.saturating_sub(left);
        let height = bottom.saturating_sub(top);
        if width == 0 || height == 0 {
            bail!("Crop area cannot be empty");
        }
        let cropped = editor.image.crop_imm(left, top, width, height);
        let format = image::ImageFormat::from_path(destination)
            .with_context(|| format!("Unsupported output format: {}", destination.display()))?;
        if overwrite {
            let extension = destination
                .extension()
                .and_then(|part| part.to_str())
                .unwrap_or("png");
            let temp_path = destination.with_file_name(format!(
                ".{}.iris-edit-tmp-{}-{}.{}",
                destination
                    .file_stem()
                    .and_then(|part| part.to_str())
                    .unwrap_or("image"),
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|duration| duration.as_nanos())
                    .unwrap_or_default(),
                extension,
            ));
            cropped.save_with_format(&temp_path, format)?;
            std::fs::rename(&temp_path, destination)?;
        } else {
            cropped.save_with_format(destination, format)?;
        }
        Ok(cropped)
    }

    pub(crate) fn refresh_after_image_edit(
        &mut self,
        path: &Path,
        edited_image: image::DynamicImage,
        ctx: &egui::Context,
    ) {
        let mut cache_paths = HashSet::from([path.to_path_buf()]);
        for candidate in &self.images {
            if candidate == path || self.resolve_actual_path(candidate) == path {
                cache_paths.insert(candidate.clone());
                cache_paths.insert(self.get_thumbnail_path(candidate));
            }
        }
        let mut latest_revision = 0;
        for cache_path in &cache_paths {
            self.thumbnail_textures.remove(cache_path.as_path());
            self.thumbnail_failed.remove(cache_path.as_path());
            self.thumbnail_retry_at.remove(cache_path.as_path());
            self.thumbnail_loading.remove(cache_path.as_path());
            self.viewer_textures.remove(cache_path.as_path());
            self.viewer_texture_failed.remove(cache_path.as_path());
            self.viewer_texture_retry_at.remove(cache_path.as_path());
            self.viewer_texture_loading.remove(cache_path.as_path());
            let revision = self
                .viewer_texture_revisions
                .entry(cache_path.clone())
                .or_default();
            *revision += 1;
            latest_revision = latest_revision.max(*revision);
            self.resolution_size_cache
                .borrow_mut()
                .remove(cache_path.as_path());
            ctx.forget_image(&format!("file://{}", cache_path.to_string_lossy()));
        }
        let texture = ctx.load_texture(
            format!("viewer_image edited: {}:{latest_revision}", path.display()),
            viewer_color_image(edited_image),
            egui::TextureOptions::LINEAR,
        );
        for cache_path in cache_paths {
            self.viewer_textures.insert(cache_path, texture.clone());
        }
        self.viewer_rotation_quarter_turns = 0;
        self.viewer_rotation_path = self.images.get(self.current_index).cloned();
        self.gallery_filter_cache_key = None;
        self.side_panel_metadata_path = None;
        self.side_panel_layout_path = None;
        self.update_current_file_info();
        self.update_side_panel_metadata_if_needed();
        ctx.request_repaint();
    }

    pub(crate) fn show_image_editor(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let Some(mut editor) = self.image_editor.take() else {
            return;
        };
        let mut close_editor = false;
        let editor_for_ui = &mut editor;
        egui::Frame::NONE
            .fill(egui::Color32::from_black_alpha(210))
            .inner_margin(8.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.strong("Crop image");
                    ui.separator();
                    if ui.button("Rotate left").clicked() {
                        Self::rotate_editor_image(editor_for_ui, 3, ctx);
                    }
                    if ui.button("Rotate right").clicked() {
                        Self::rotate_editor_image(editor_for_ui, 1, ctx);
                    }
                    if ui.button("Rotate 180").clicked() {
                        Self::rotate_editor_image(editor_for_ui, 2, ctx);
                    }
                    if ui.button("Reset crop").clicked() {
                        editor_for_ui.crop_min = egui::pos2(0.0, 0.0);
                        editor_for_ui.crop_max = egui::pos2(1.0, 1.0);
                    }
                    if ui.button("Fit width").clicked() {
                        editor_for_ui.crop_min.x = 0.0;
                        editor_for_ui.crop_max.x = 1.0;
                    }
                    if ui.button("Fit height").clicked() {
                        editor_for_ui.crop_min.y = 0.0;
                        editor_for_ui.crop_max.y = 1.0;
                    }
                    ui.separator();
                    if ui.button("Save in place").clicked() {
                        match Self::save_editor_image(
                            editor_for_ui,
                            &editor_for_ui.source_path,
                            true,
                        ) {
                            Ok(edited_image) => {
                                let path = editor_for_ui.source_path.clone();
                                self.refresh_after_image_edit(&path, edited_image, ctx);
                                close_editor = true;
                            }
                            Err(err) => editor_for_ui.status = format!("Save failed: {err}"),
                        }
                    }
                    if ui.button("Save edited copy").clicked() {
                        let destination = Self::edited_copy_path(&editor_for_ui.source_path);
                        match Self::save_editor_image(editor_for_ui, &destination, false) {
                            Ok(_) => {
                                editor_for_ui.status = format!("Saved {}", destination.display());
                                let index = self.recursive_images.len();
                                if is_video_path(&destination) {
                                    self.recursive_video_indices.push(index);
                                }
                                let mut recursive_images = self.recursive_images.to_vec();
                                recursive_images.push(destination);
                                self.recursive_images = Arc::from(recursive_images);
                                self.recursive_images_snapshot = Arc::clone(&self.recursive_images);
                            }
                            Err(err) => editor_for_ui.status = format!("Save failed: {err}"),
                        }
                    }
                    if ui.button("Cancel").clicked() {
                        close_editor = true;
                    }
                });

                ui.label("Drag outside the selection to create a crop, inside it to move, or its edges and corners to resize.");
                if !editor_for_ui.status.is_empty() {
                    ui.label(&editor_for_ui.status);
                }

                let available = ui.available_size();
                let image_aspect = editor_for_ui.image.width() as f32 / editor_for_ui.image.height() as f32;
                let available_aspect = available.x / available.y.max(1.0);
                let draw_size = if available_aspect > image_aspect {
                    egui::vec2(available.y * image_aspect, available.y)
                } else {
                    egui::vec2(available.x, available.x / image_aspect)
                };
                let (viewport_rect, _) = ui.allocate_exact_size(available, egui::Sense::hover());
                let image_rect = egui::Rect::from_center_size(viewport_rect.center(), draw_size);
                let crop_response = ui.interact(
                    image_rect.expand(12.0),
                    ui.make_persistent_id("embedded_image_crop"),
                    egui::Sense::click_and_drag(),
                );
                ui.painter().image(
                    editor_for_ui.texture.id(),
                    image_rect,
                    egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );

                let pointer_to_normalized = |pointer: egui::Pos2| {
                    egui::pos2(
                        ((pointer.x - image_rect.left()) / image_rect.width()).clamp(0.0, 1.0),
                        ((pointer.y - image_rect.top()) / image_rect.height()).clamp(0.0, 1.0),
                    )
                };
                let crop_rect = egui::Rect::from_min_max(
                    egui::pos2(
                        image_rect.left() + editor_for_ui.crop_min.x * image_rect.width(),
                        image_rect.top() + editor_for_ui.crop_min.y * image_rect.height(),
                    ),
                    egui::pos2(
                        image_rect.left() + editor_for_ui.crop_max.x * image_rect.width(),
                        image_rect.top() + editor_for_ui.crop_max.y * image_rect.height(),
                    ),
                );
                let handle_distance = 14.0;
                if crop_response.hovered() {
                    if let Some(pointer) = ctx.pointer_hover_pos() {
                        let near_left = (pointer.x - crop_rect.left()).abs() <= handle_distance;
                        let near_right = (pointer.x - crop_rect.right()).abs() <= handle_distance;
                        let near_top = (pointer.y - crop_rect.top()).abs() <= handle_distance;
                        let near_bottom = (pointer.y - crop_rect.bottom()).abs() <= handle_distance;
                        let within_x = pointer.x >= crop_rect.left() - handle_distance
                            && pointer.x <= crop_rect.right() + handle_distance;
                        let within_y = pointer.y >= crop_rect.top() - handle_distance
                            && pointer.y <= crop_rect.bottom() + handle_distance;
                        let cursor = if (near_left && near_top) || (near_right && near_bottom) {
                            egui::CursorIcon::ResizeNwSe
                        } else if (near_right && near_top) || (near_left && near_bottom) {
                            egui::CursorIcon::ResizeNeSw
                        } else if (near_left || near_right) && within_y {
                            egui::CursorIcon::ResizeHorizontal
                        } else if (near_top || near_bottom) && within_x {
                            egui::CursorIcon::ResizeVertical
                        } else if crop_rect.contains(pointer) {
                            egui::CursorIcon::Move
                        } else {
                            egui::CursorIcon::Crosshair
                        };
                        ctx.set_cursor_icon(cursor);
                    }
                }
                if crop_response.drag_started() {
                    if let Some(pointer) = crop_response.interact_pointer_pos() {
                        let near_left = (pointer.x - crop_rect.left()).abs() <= handle_distance;
                        let near_right = (pointer.x - crop_rect.right()).abs() <= handle_distance;
                        let near_top = (pointer.y - crop_rect.top()).abs() <= handle_distance;
                        let near_bottom = (pointer.y - crop_rect.bottom()).abs() <= handle_distance;
                        let within_x = pointer.x >= crop_rect.left() - handle_distance
                            && pointer.x <= crop_rect.right() + handle_distance;
                        let within_y = pointer.y >= crop_rect.top() - handle_distance
                            && pointer.y <= crop_rect.bottom() + handle_distance;
                        editor_for_ui.crop_drag_mode = Some(if near_left && near_top {
                            CropDragMode::TopLeft
                        } else if near_right && near_top {
                            CropDragMode::TopRight
                        } else if near_left && near_bottom {
                            CropDragMode::BottomLeft
                        } else if near_right && near_bottom {
                            CropDragMode::BottomRight
                        } else if near_left && within_y {
                            CropDragMode::Left
                        } else if near_right && within_y {
                            CropDragMode::Right
                        } else if near_top && within_x {
                            CropDragMode::Top
                        } else if near_bottom && within_x {
                            CropDragMode::Bottom
                        } else if crop_rect.contains(pointer) {
                            CropDragMode::Move
                        } else {
                            CropDragMode::New
                        });
                        editor_for_ui.crop_drag_origin = pointer_to_normalized(pointer);
                        editor_for_ui.crop_drag_initial_min = editor_for_ui.crop_min;
                        editor_for_ui.crop_drag_initial_max = editor_for_ui.crop_max;
                    }
                }
                if crop_response.dragged() {
                    if let (Some(mode), Some(pointer)) =
                        (editor_for_ui.crop_drag_mode, crop_response.interact_pointer_pos())
                    {
                        let current = pointer_to_normalized(pointer);
                        let origin = editor_for_ui.crop_drag_origin;
                        let initial_min = editor_for_ui.crop_drag_initial_min;
                        let initial_max = editor_for_ui.crop_drag_initial_max;
                        let minimum = egui::vec2(
                            2.0 / editor_for_ui.image.width().max(1) as f32,
                            2.0 / editor_for_ui.image.height().max(1) as f32,
                        );
                        match mode {
                            CropDragMode::New => {
                                editor_for_ui.crop_min =
                                    egui::pos2(origin.x.min(current.x), origin.y.min(current.y));
                                editor_for_ui.crop_max =
                                    egui::pos2(origin.x.max(current.x), origin.y.max(current.y));
                            }
                            CropDragMode::Move => {
                                let size = initial_max - initial_min;
                                let delta = current - origin;
                                let mut min = initial_min + delta;
                                min.x = min.x.clamp(0.0, 1.0 - size.x);
                                min.y = min.y.clamp(0.0, 1.0 - size.y);
                                editor_for_ui.crop_min = min;
                                editor_for_ui.crop_max = min + size;
                            }
                            CropDragMode::Left | CropDragMode::TopLeft | CropDragMode::BottomLeft => {
                                editor_for_ui.crop_min.x =
                                    current.x.clamp(0.0, editor_for_ui.crop_max.x - minimum.x);
                            }
                            CropDragMode::Right | CropDragMode::TopRight | CropDragMode::BottomRight => {
                                editor_for_ui.crop_max.x =
                                    current.x.clamp(editor_for_ui.crop_min.x + minimum.x, 1.0);
                            }
                            _ => {}
                        }
                        match mode {
                            CropDragMode::Top | CropDragMode::TopLeft | CropDragMode::TopRight => {
                                editor_for_ui.crop_min.y =
                                    current.y.clamp(0.0, editor_for_ui.crop_max.y - minimum.y);
                            }
                            CropDragMode::Bottom
                            | CropDragMode::BottomLeft
                            | CropDragMode::BottomRight => {
                                editor_for_ui.crop_max.y =
                                    current.y.clamp(editor_for_ui.crop_min.y + minimum.y, 1.0);
                            }
                            _ => {}
                        }
                    }
                }
                if crop_response.drag_stopped() {
                    editor_for_ui.crop_drag_mode = None;
                }
                let shade = egui::Color32::from_black_alpha(150);
                ui.painter().rect_filled(
                    egui::Rect::from_min_max(image_rect.min, egui::pos2(image_rect.right(), crop_rect.top())),
                    0.0,
                    shade,
                );
                ui.painter().rect_filled(
                    egui::Rect::from_min_max(egui::pos2(image_rect.left(), crop_rect.bottom()), image_rect.max),
                    0.0,
                    shade,
                );
                ui.painter().rect_filled(
                    egui::Rect::from_min_max(egui::pos2(image_rect.left(), crop_rect.top()), egui::pos2(crop_rect.left(), crop_rect.bottom())),
                    0.0,
                    shade,
                );
                ui.painter().rect_filled(
                    egui::Rect::from_min_max(egui::pos2(crop_rect.right(), crop_rect.top()), egui::pos2(image_rect.right(), crop_rect.bottom())),
                    0.0,
                    shade,
                );
                ui.painter().rect_stroke(
                    crop_rect,
                    0.0,
                    egui::Stroke::new(2.0, egui::Color32::WHITE),
                    egui::StrokeKind::Inside,
                );
                for handle in [
                    crop_rect.left_top(),
                    crop_rect.right_top(),
                    crop_rect.left_bottom(),
                    crop_rect.right_bottom(),
                    egui::pos2(crop_rect.center().x, crop_rect.top()),
                    egui::pos2(crop_rect.center().x, crop_rect.bottom()),
                    egui::pos2(crop_rect.left(), crop_rect.center().y),
                    egui::pos2(crop_rect.right(), crop_rect.center().y),
                ] {
                    ui.painter().circle_filled(handle, 5.0, egui::Color32::WHITE);
                    ui.painter().circle_stroke(
                        handle,
                        5.0,
                        egui::Stroke::new(1.0, egui::Color32::BLACK),
                    );
                }
            });
        if !close_editor {
            self.image_editor = Some(editor);
        }
    }

    pub(crate) fn update_side_panel_metadata_if_needed(&mut self) {
        if self.side_panel_mode != SidePanelMode::Exif {
            return;
        }
        let current_path = self.images.get(self.current_index).cloned();
        if self.side_panel_metadata_path != current_path {
            self.update_exif();
        }
    }

    pub(crate) fn update_layout_if_needed(&mut self) {
        if self.side_panel_mode != SidePanelMode::Layout {
            return;
        }
        let Some(path) = self.images.get(self.current_index).cloned() else {
            self.chunks = Vec::new();
            self.side_panel_layout_path = None;
            return;
        };
        if self.side_panel_layout_path.as_ref() == Some(&path) {
            return;
        }

        self.update_current_file_info();
        self.request_metadata_load(false, true);
    }

    pub(crate) fn current_flat_directory(&self) -> Option<PathBuf> {
        if self.open_target_is_dir {
            return None;
        }
        let path = self
            .images
            .get(self.current_index)
            .unwrap_or(&self.open_target);
        path.parent().map(Path::to_path_buf)
    }

    pub(crate) fn current_flat_directory_mtime(&self) -> Option<SystemTime> {
        self.current_flat_directory()
            .and_then(|directory| std::fs::metadata(directory).ok())
            .and_then(|metadata| metadata.modified().ok())
    }

    pub(crate) fn start_flat_refresh_if_changed(&mut self) {
        if self.is_comparison_mode()
            || self.open_target_is_dir
            || self.flat_loading
            || self.flat_refresh_in_flight
        {
            return;
        }
        let Some(directory) = self.current_flat_directory() else {
            return;
        };
        let current_mtime = std::fs::metadata(&directory)
            .ok()
            .and_then(|metadata| metadata.modified().ok());
        if current_mtime == self.flat_directory_mtime {
            return;
        }

        self.flat_refresh_in_flight = true;
        self.flat_loading = true;
        self.flat_refresh_generation = self.flat_refresh_generation.wrapping_add(1);
        let generation = self.flat_refresh_generation;
        if let Ok(mut lock) = self.flat_images_shared.lock() {
            *lock = None;
        }

        let shared = self.flat_images_shared.clone();
        let diagnostics = self.diagnostics.clone();
        std::thread::spawn(move || {
            let task = diagnostics.task_guard("viewer_flat_refresh");
            let directory = directory.canonicalize().unwrap_or(directory);
            let collected = collect_flat_images(&directory);
            if let Ok(mut lock) = shared.lock() {
                let replace = lock
                    .as_ref()
                    .is_none_or(|existing| existing.generation <= generation);
                if replace {
                    *lock = Some(FlatRefreshResult {
                        generation,
                        directory,
                        images: collected,
                    });
                }
            }
            task.complete();
        });
    }

    pub(crate) fn poll_flat_directory_refresh(&mut self, ctx: &egui::Context) {
        if self.is_comparison_mode() || self.open_target_is_dir || self.images.is_empty() {
            return;
        }
        if self.flat_last_refresh_check.elapsed() >= Duration::from_secs(1) {
            self.flat_last_refresh_check = Instant::now();
            self.start_flat_refresh_if_changed();
        }
        ctx.request_repaint_after(Duration::from_secs(1));
    }

    pub(crate) fn next_image(&mut self) {
        if let Some(indices) = self.gallery_navigation_indices.clone() {
            if !indices.is_empty() {
                self.gallery_navigation_position =
                    (self.gallery_navigation_position + 1) % indices.len();
                let path = self.recursive_images[indices[self.gallery_navigation_position]].clone();
                self.images = vec![path];
                self.current_index = 0;
                self.update_current_file_info();
                self.update_side_panel_metadata_if_needed();
            }
            return;
        }
        if !self.images.is_empty() {
            if self.is_comparison_mode() {
                self.switch_comparison_image((self.current_index + 1) % self.images.len());
            } else {
                self.current_index = (self.current_index + 1) % self.images.len();
                self.update_current_file_info();
                self.update_side_panel_metadata_if_needed();
            }
        }
    }

    pub(crate) fn prev_image(&mut self) {
        if let Some(indices) = self.gallery_navigation_indices.clone() {
            if !indices.is_empty() {
                self.gallery_navigation_position = if self.gallery_navigation_position == 0 {
                    indices.len() - 1
                } else {
                    self.gallery_navigation_position - 1
                };
                let path = self.recursive_images[indices[self.gallery_navigation_position]].clone();
                self.images = vec![path];
                self.current_index = 0;
                self.update_current_file_info();
                self.update_side_panel_metadata_if_needed();
            }
            return;
        }
        if !self.images.is_empty() {
            if self.is_comparison_mode() {
                let index = if self.current_index == 0 {
                    self.images.len() - 1
                } else {
                    self.current_index - 1
                };
                self.switch_comparison_image(index);
            } else {
                if self.current_index == 0 {
                    self.current_index = self.images.len() - 1;
                } else {
                    self.current_index -= 1;
                }
                self.update_current_file_info();
                self.update_side_panel_metadata_if_needed();
            }
        }
    }

    pub(crate) fn current_folder_has_db_mappings(&self) -> bool {
        self.folder_has_db_mappings(&self.default_semantic_folder().to_string_lossy())
    }
}
