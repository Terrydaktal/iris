use super::*;

impl ImageViewState {
    pub(crate) const MIN_ZOOM: f32 = 0.05;
    pub(crate) const MAX_ZOOM: f32 = 256.0;

    pub(crate) fn zoom_at(&mut self, zoom_factor: f32, anchor: egui::Vec2) -> bool {
        let zoom = (self.zoom * zoom_factor).clamp(Self::MIN_ZOOM, Self::MAX_ZOOM);
        if zoom == self.zoom {
            return false;
        }
        // Keep the cursor over the same image point using the actual scale
        // change, not the requested one, which can exceed the zoom limits.
        let actual_factor = zoom / self.zoom;
        self.offset = anchor - (anchor - self.offset) * actual_factor;
        self.zoom = zoom;
        true
    }
}

impl ImageViewer {
    pub(crate) fn show_viewer_sampling_controls(&mut self, ui: &mut egui::Ui) {
        let mut sampling = self.viewer_sampling;
        ui.selectable_value(&mut sampling, ViewerSampling::Smooth, "Smooth")
            .on_hover_text("Smooth enlarged text and edges. This is the default.");
        ui.selectable_value(&mut sampling, ViewerSampling::PixelSharp, "Pixel Sharp")
            .on_hover_text(
                "Show distinct source pixels when enlarged, without smoothing them together.",
            );
        self.set_viewer_sampling(sampling, ui.ctx());
    }

    pub(crate) fn set_viewer_sampling(&mut self, sampling: ViewerSampling, ctx: &egui::Context) {
        if sampling != self.viewer_sampling {
            self.viewer_sampling = sampling;
            for texture in self.viewer_textures.values_mut() {
                texture.set_sampling(sampling);
            }
            self.viewer_detail.set_sampling(sampling);
            ctx.request_repaint();
        }
    }

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
        self.viewer_source_sizes
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
            return Some(texture.handle.clone());
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
                    .map(ViewerPreview::from_image)
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
            self.viewer_rotation_degrees
        } else {
            0.0
        };
        match image::open(&source_path) {
            Ok(image) => {
                self.image_editor = Some(ImageEditor {
                    texture: Self::editor_texture(ctx, &image),
                    source_path,
                    image,
                    rotation_degrees: viewer_rotation,
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

    pub(crate) fn normalized_rotation_degrees(degrees: f32) -> f32 {
        let normalized = degrees.rem_euclid(360.0);
        if normalized.abs() < 0.001 || (360.0 - normalized).abs() < 0.001 {
            0.0
        } else {
            normalized
        }
    }

    pub(crate) fn set_viewer_rotation(&mut self, path: &Path, degrees: f32, ctx: &egui::Context) {
        self.viewer_rotation_degrees = Self::normalized_rotation_degrees(degrees);
        self.viewer_rotation_path = Some(path.to_path_buf());
        self.zoom = 1.0;
        self.offset = egui::Vec2::ZERO;
        self.apply_comparison_view_state_to_all();
        ctx.request_repaint();
    }

    pub(crate) fn rotated_bounding_size(size: egui::Vec2, angle: f32) -> egui::Vec2 {
        let (sin, cos) = angle.sin_cos();
        egui::vec2(
            size.x * cos.abs() + size.y * sin.abs(),
            size.x * sin.abs() + size.y * cos.abs(),
        )
    }

    fn rotate_point(point: egui::Pos2, center: egui::Pos2, angle: f32) -> egui::Pos2 {
        let relative = point - center;
        let (sin, cos) = angle.sin_cos();
        center
            + egui::vec2(
                relative.x * cos - relative.y * sin,
                relative.x * sin + relative.y * cos,
            )
    }

    fn rotate_image_degrees(
        image: &image::DynamicImage,
        degrees: f32,
    ) -> Result<image::DynamicImage> {
        let degrees = Self::normalized_rotation_degrees(degrees);
        let rounded = degrees.round();
        if (degrees - rounded).abs() < 0.001 {
            match rounded as i32 {
                0 | 360 => return Ok(image.clone()),
                90 => return Ok(image.rotate90()),
                180 => return Ok(image.rotate180()),
                270 => return Ok(image.rotate270()),
                _ => {}
            }
        }

        let source = image.to_rgba8();
        let source_width = source.width();
        let source_height = source.height();
        let angle = degrees.to_radians();
        let bounds = Self::rotated_bounding_size(
            egui::vec2(source_width as f32, source_height as f32),
            angle,
        );
        let output_width = bounds.x.ceil().max(1.0) as u32;
        let output_height = bounds.y.ceil().max(1.0) as u32;
        let source_center_x = (source_width.saturating_sub(1)) as f32 / 2.0;
        let source_center_y = (source_height.saturating_sub(1)) as f32 / 2.0;
        let output_center_x = (output_width.saturating_sub(1)) as f32 / 2.0;
        let output_center_y = (output_height.saturating_sub(1)) as f32 / 2.0;
        let (sin, cos) = angle.sin_cos();
        let source_bytes = source.as_raw();
        let source_stride = source_width as usize * 4;
        let output_stride = output_width as usize * 4;
        let output_len = output_stride
            .checked_mul(output_height as usize)
            .ok_or_else(|| anyhow!("Rotated image dimensions are too large"))?;
        let mut output = vec![0_u8; output_len];

        output
            .par_chunks_mut(output_stride)
            .enumerate()
            .for_each(|(output_y, row)| {
                let dy = output_y as f32 - output_center_y;
                for output_x in 0..output_width as usize {
                    let dx = output_x as f32 - output_center_x;
                    let source_x = dx * cos + dy * sin + source_center_x;
                    let source_y = -dx * sin + dy * cos + source_center_y;
                    if source_x < 0.0
                        || source_y < 0.0
                        || source_x > source_width.saturating_sub(1) as f32
                        || source_y > source_height.saturating_sub(1) as f32
                    {
                        continue;
                    }

                    let x0 = source_x.floor() as usize;
                    let y0 = source_y.floor() as usize;
                    let x1 = (x0 + 1).min(source_width.saturating_sub(1) as usize);
                    let y1 = (y0 + 1).min(source_height.saturating_sub(1) as usize);
                    let x_fraction = source_x - x0 as f32;
                    let y_fraction = source_y - y0 as f32;
                    let samples = [
                        (x0, y0, (1.0 - x_fraction) * (1.0 - y_fraction)),
                        (x1, y0, x_fraction * (1.0 - y_fraction)),
                        (x0, y1, (1.0 - x_fraction) * y_fraction),
                        (x1, y1, x_fraction * y_fraction),
                    ];
                    let mut weighted_alpha = 0.0_f32;
                    let mut premultiplied = [0.0_f32; 3];
                    for (sample_x, sample_y, weight) in samples {
                        let offset = sample_y * source_stride + sample_x * 4;
                        let alpha = source_bytes[offset + 3] as f32;
                        weighted_alpha += alpha * weight;
                        for channel in 0..3 {
                            premultiplied[channel] +=
                                source_bytes[offset + channel] as f32 * alpha * weight;
                        }
                    }
                    let destination = &mut row[output_x * 4..output_x * 4 + 4];
                    if weighted_alpha > 0.0 {
                        for channel in 0..3 {
                            destination[channel] =
                                (premultiplied[channel] / weighted_alpha).round() as u8;
                        }
                    }
                    destination[3] = weighted_alpha.round() as u8;
                }
            });

        image::RgbaImage::from_raw(output_width, output_height, output)
            .map(image::DynamicImage::ImageRgba8)
            .ok_or_else(|| anyhow!("Unable to construct rotated image buffer"))
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
        let edited = Self::rotate_image_degrees(&cropped, editor.rotation_degrees)?;
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
            edited.save_with_format(&temp_path, format)?;
            std::fs::rename(&temp_path, destination)?;
        } else {
            edited.save_with_format(destination, format)?;
        }
        Ok(edited)
    }

    pub(crate) fn refresh_after_image_edit(
        &mut self,
        path: &Path,
        edited_image: image::DynamicImage,
        ctx: &egui::Context,
    ) {
        self.viewer_detail.clear();
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
        let preview = ViewerPreview::from_image(edited_image);
        let native_resolution = preview.is_native_resolution();
        let texture = ViewerTexture::new(
            ctx,
            format!("viewer_image edited: {}:{latest_revision}", path.display()),
            preview.image,
            native_resolution,
            self.viewer_sampling,
        );
        for cache_path in cache_paths {
            self.viewer_source_sizes
                .insert(cache_path.clone(), preview.source_size);
            self.viewer_textures.insert(cache_path, texture.clone());
        }
        self.viewer_rotation_degrees = 0.0;
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
                        editor_for_ui.rotation_degrees = Self::normalized_rotation_degrees(
                            editor_for_ui.rotation_degrees - 90.0,
                        );
                    }
                    if ui.button("Rotate right").clicked() {
                        editor_for_ui.rotation_degrees = Self::normalized_rotation_degrees(
                            editor_for_ui.rotation_degrees + 90.0,
                        );
                    }
                    ui.add(
                        egui::Slider::new(&mut editor_for_ui.rotation_degrees, 0.0..=360.0)
                            .text("Rotation")
                            .suffix("°")
                            .step_by(1.0),
                    );
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
                                Arc::make_mut(&mut self.recursive_images).push(destination);
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
                let source_size = egui::vec2(
                    editor_for_ui.image.width() as f32,
                    editor_for_ui.image.height() as f32,
                );
                let angle = editor_for_ui.rotation_degrees.to_radians();
                let rotated_source_size = Self::rotated_bounding_size(source_size, angle);
                let fit_scale = (available.x / rotated_source_size.x.max(1.0))
                    .min(available.y / rotated_source_size.y.max(1.0));
                let draw_size = source_size * fit_scale;
                let rotated_draw_size = rotated_source_size * fit_scale;
                let (viewport_rect, _) = ui.allocate_exact_size(available, egui::Sense::hover());
                let image_rect = egui::Rect::from_center_size(viewport_rect.center(), draw_size);
                let rotated_bounds =
                    egui::Rect::from_center_size(viewport_rect.center(), rotated_draw_size);
                let crop_response = ui.interact(
                    rotated_bounds.expand(12.0),
                    ui.make_persistent_id("embedded_image_crop"),
                    egui::Sense::click_and_drag(),
                );
                let rotate_point = |point| Self::rotate_point(point, image_rect.center(), angle);
                let pointer_to_normalized = |pointer: egui::Pos2| {
                    let pointer = Self::rotate_point(pointer, image_rect.center(), -angle);
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
                let paint_image_region =
                    |rect: egui::Rect, uv: egui::Rect, tint: egui::Color32| {
                        let mut mesh = egui::Mesh::with_texture(editor_for_ui.texture.id());
                        mesh.add_rect_with_uv(rect, uv, tint);
                        for vertex in &mut mesh.vertices {
                            vertex.pos = rotate_point(vertex.pos);
                        }
                        ui.painter().add(egui::Shape::mesh(mesh));
                    };
                paint_image_region(
                    image_rect,
                    egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                    egui::Color32::from_gray(90),
                );
                paint_image_region(
                    crop_rect,
                    egui::Rect::from_min_max(editor_for_ui.crop_min, editor_for_ui.crop_max),
                    egui::Color32::WHITE,
                );

                let handle_distance = 14.0;
                if crop_response.hovered() {
                    if let Some(pointer) = ctx.pointer_hover_pos() {
                        let pointer = Self::rotate_point(pointer, image_rect.center(), -angle);
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
                        let unrotated_pointer =
                            Self::rotate_point(pointer, image_rect.center(), -angle);
                        let near_left =
                            (unrotated_pointer.x - crop_rect.left()).abs() <= handle_distance;
                        let near_right =
                            (unrotated_pointer.x - crop_rect.right()).abs() <= handle_distance;
                        let near_top =
                            (unrotated_pointer.y - crop_rect.top()).abs() <= handle_distance;
                        let near_bottom =
                            (unrotated_pointer.y - crop_rect.bottom()).abs() <= handle_distance;
                        let within_x = unrotated_pointer.x >= crop_rect.left() - handle_distance
                            && unrotated_pointer.x <= crop_rect.right() + handle_distance;
                        let within_y = unrotated_pointer.y >= crop_rect.top() - handle_distance
                            && unrotated_pointer.y <= crop_rect.bottom() + handle_distance;
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
                        } else if crop_rect.contains(unrotated_pointer) {
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
                let crop_points = [
                    crop_rect.left_top(),
                    crop_rect.right_top(),
                    crop_rect.right_bottom(),
                    crop_rect.left_bottom(),
                ]
                .map(rotate_point);
                for edge in 0..4 {
                    ui.painter().line_segment(
                        [crop_points[edge], crop_points[(edge + 1) % 4]],
                        egui::Stroke::new(2.0, egui::Color32::WHITE),
                    );
                }
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
                    let handle = rotate_point(handle);
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
                let paths = self
                    .gallery_navigation_paths
                    .as_ref()
                    .unwrap_or(&self.recursive_images);
                let path = paths[indices[self.gallery_navigation_position]].clone();
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
                let paths = self
                    .gallery_navigation_paths
                    .as_ref()
                    .unwrap_or(&self.recursive_images);
                let path = paths[indices[self.gallery_navigation_position]].clone();
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

#[cfg(test)]
mod rotation_tests {
    use super::*;
    use image::GenericImageView as _;

    #[test]
    fn arbitrary_rotation_expands_the_output_without_clipping() {
        let source = image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            4,
            2,
            image::Rgba([255, 0, 0, 255]),
        ));

        let quarter_turn = ImageViewer::rotate_image_degrees(&source, 90.0).unwrap();
        let diagonal = ImageViewer::rotate_image_degrees(&source, 45.0).unwrap();

        assert_eq!(quarter_turn.dimensions(), (2, 4));
        assert_eq!(diagonal.dimensions(), (5, 5));

        let mut encoded = std::io::Cursor::new(Vec::new());
        diagonal
            .write_to(&mut encoded, image::ImageFormat::Jpeg)
            .unwrap();
        assert!(!encoded.into_inner().is_empty());
    }

    #[test]
    fn rotation_degrees_wrap_cleanly() {
        assert_eq!(ImageViewer::normalized_rotation_degrees(-90.0), 270.0);
        assert_eq!(ImageViewer::normalized_rotation_degrees(360.0), 0.0);
        assert_eq!(ImageViewer::normalized_rotation_degrees(450.0), 90.0);
    }
}
