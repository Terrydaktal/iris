use super::*;

impl ImageViewer {
    pub(crate) fn show_grid_view(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        #[derive(Clone)]
        struct GalleryItem {
            path: PathBuf,
            is_video: bool,
            score_label: Option<String>,
            timestamp_sec: f32,
            db_filename: Option<String>,
        }

        ui.vertical(|ui| {
            // Dynamic Database Mapping Check & Auto Lazy Load
            let default_scope = self.default_semantic_folder().to_string_lossy().to_string();
            let effective_scope = self.effective_semantic_folder();
            let scope_has_db = self.folder_has_db_mappings(&effective_scope);
            let has_db = self.current_folder_has_db_mappings();
            if (self.semantic_mode == SearchMode::Clip || self.semantic_mode == SearchMode::Ocr) && !self.db_loaded && !self.db_loading {
                self.start_lazy_db_load(ctx);
            }
            let (clip_paste_shortcut_pressed, clip_pasted_text) = if self.semantic_mode == SearchMode::Clip {
                clipboard_paste_signal(ui)
            } else {
                (false, None)
            };
            if self.semantic_mode == SearchMode::Clip {
                if let Some(text) = clip_pasted_text.as_deref() {
                    if image_path_from_pasted_text(text).is_some() {
                        self.search_clip_from_clipboard_image(ui.ctx(), Some(text), false);
                    } else if clip_paste_shortcut_pressed {
                        self.search_clip_from_clipboard_image(ui.ctx(), Some(text), true);
                    } else {
                        self.search_clip_from_clipboard_image(ui.ctx(), Some(text), false);
                    }
                } else if clip_paste_shortcut_pressed {
                    self.search_clip_from_clipboard_image(ui.ctx(), None, true);
                }
            }

            // Toolbar Controls
            let old_mode = self.semantic_mode;
            ui.horizontal(|ui| {
                ui.label("Mode:");
                ui.selectable_value(&mut self.semantic_mode, SearchMode::Filename, "Filename");
                ui.selectable_value(&mut self.semantic_mode, SearchMode::Clip, "CLIP");
                ui.selectable_value(&mut self.semantic_mode, SearchMode::Ocr, "OCR");

                ui.add_space(12.0);
                ui.checkbox(&mut self.semantic_video_only, "Videos only");
                ui.weak(format!(
                    "Thumbs {:.0}%",
                    self.gallery_thumbnail_scale * 100.0
                ))
                .on_hover_text("Hold Ctrl and scroll over the gallery to resize thumbnails.");

                ui.separator();
                let selected_count = self.selected_grid_items.len();
                let selected_sift_count = self.selected_sift_file_names().len();
                let repair_label = if self.sift_repair_running {
                    "Repairing SIFT..."
                } else {
                    "Repair selected SIFT"
                };
                if ui
                    .add_enabled(
                        has_db && !self.sift_repair_running && selected_sift_count >= 2,
                        egui::Button::new(repair_label),
                    )
                    .clicked()
                {
                    self.start_selected_sift_repair(ctx);
                }
                if ui
                    .add_enabled(
                        has_db && !self.sift_repair_running && selected_sift_count >= 2,
                        egui::Button::new("Compare selected SIFT"),
                    )
                    .clicked()
                {
                    self.start_selected_sift_compare(ctx);
                }
                let face_label = if self.face_compare_running {
                    "Comparing faces..."
                } else {
                    "Compare selected faces"
                };
                if ui
                    .add_enabled(
                        !self.face_compare_running && selected_count == 2,
                        egui::Button::new(face_label),
                    )
                    .on_hover_text(
                        "Compare two photos, or find the video frame whose face best matches a selected photo.",
                    )
                    .clicked()
                {
                    self.start_selected_face_compare(ctx);
                }
                if ui
                    .add_enabled(
                        !self.sift_repair_running && selected_count > 0,
                        egui::Button::new("Clear selected"),
                    )
                    .clicked()
                {
                    self.selected_grid_items.clear();
                    self.semantic_status = "Selection cleared.".to_string();
                }
                if selected_count > 0 || self.sift_repair_running {
                    ui.weak(format!("{selected_count} selected (Ctrl-click tiles to toggle)"));
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Refresh").clicked() {
                        self.start_recursive_scan();
                    }
                });
            });
            if self.semantic_mode != old_mode {
                self.semantic_results_mode = None;
                self.pending_semantic_search_mode = None;
            }
            ui.add_space(6.0);

            ui.horizontal(|ui| {
                let hint = match self.semantic_mode {
                    SearchMode::Filename => "Filter by filename...",
                    SearchMode::Clip => "Describe the photo or paste an image with Ctrl+V",
                    SearchMode::Ocr => "Type word/text found inside the image",
                };

                ui.label("Search:");
                let search_resp = ui.add(egui::TextEdit::singleline(&mut self.semantic_query)
                    .hint_text(hint)
                    .desired_width(320.0));
                let enter_pressed = text_edit_enter_pressed(&search_resp);

                ui.add_space(8.0);
                if self.semantic_mode == SearchMode::Clip && ui.button("Paste Image").clicked() {
                    self.search_clip_from_clipboard_image(ui.ctx(), None, true);
                }
                if self.semantic_mode == SearchMode::Clip {
                    ui.add_space(8.0);
                }
                ui.add(egui::Slider::new(&mut self.semantic_limit, 1..=500).text("Limit"));

                ui.add_space(8.0);
                if ui.button("Search").clicked() || enter_pressed {
                    self.submit_semantic_search(ctx);
                }
            });
            if matches!(self.semantic_mode, SearchMode::Clip | SearchMode::Ocr) {
                ui.add_space(6.0);
                let mut folder_enter_pressed = false;
                ui.horizontal(|ui| {
                    ui.label("Folder:");
                    let scope_resp = ui.add(
                        egui::TextEdit::singleline(&mut self.semantic_folder)
                            .hint_text("Blank = all indexed folders, or enter indexed folder path")
                            .desired_width(520.0),
                    );
                    folder_enter_pressed = text_edit_enter_pressed(&scope_resp);
                    if scope_resp.hovered() {
                        scope_resp.on_hover_text(
                            "Use an absolute filesystem path or a database-style path like collection_id/sub/folder. Leave blank to search across all indexed folders."
                        );
                    }
                    if ui.button("Use current").clicked() {
                        self.semantic_folder = default_scope.clone();
                    }
                    if ui.button("Clear").clicked() {
                        self.semantic_folder.clear();
                    }
                });
                if folder_enter_pressed {
                    self.submit_semantic_search(ctx);
                }
                let scope_label = if self.semantic_folder.trim().is_empty() {
                    "Active scope: all indexed folders".to_string()
                } else {
                    format!("Active scope: {effective_scope}")
                };
                ui.weak(scope_label);
                if !scope_has_db {
                    ui.weak("The selected scope is not inside a mapped database collection root.");
                }
            }
            ui.add_space(8.0);

            let is_active_semantic_search = match self.semantic_mode {
                SearchMode::Filename => false,
                SearchMode::Clip | SearchMode::Ocr => {
                    self.semantic_results_mode == Some(self.semantic_mode)
                }
            };

            // Keep the folder grid empty until the recursive scan has completed so the
            // user does not see a transient partial ordering before the final sorted set.
            let filename_candidates: Vec<usize> = if !is_active_semantic_search && self.grid_loading {
                Vec::new()
            } else if self.semantic_mode == SearchMode::Filename {
                self.filename_search_results
                    .clone()
                    .unwrap_or_else(|| (0..self.recursive_images.len()).collect())
            } else {
                (0..self.recursive_images.len()).collect()
            };
            let filtered_images: Vec<usize> = if self.semantic_video_only {
                if self.filename_search_results.is_some() {
                    filename_candidates
                        .into_iter()
                        .filter(|index| self.recursive_video_indices.binary_search(index).is_ok())
                        .collect()
                } else {
                    // The scan builds this once, avoiding an extension check for every
                    // indexed path on every UI frame while scrolling.
                    self.recursive_video_indices.clone()
                }
            } else {
                filename_candidates
            };

            // Status message label
            ui.horizontal(|ui| {
                let show_sift_status = self.sift_repair_running
                    || self.semantic_status.starts_with("SIFT repair")
                    || self.semantic_status.starts_with("Running SIFT")
                    || self.semantic_status.starts_with("Loading database index before SIFT")
                    || self.semantic_status.starts_with("Select at least")
                    || self.semantic_status.contains("selected")
                    || self.semantic_status.starts_with("Only indexed");
                if show_sift_status {
                    if self.sift_repair_running {
                        ui.add(egui::Spinner::new().size(14.0));
                    }
                    ui.weak(&self.semantic_status);
                } else if self.db_loading
                    && matches!(self.semantic_mode, SearchMode::Clip | SearchMode::Ocr)
                {
                    ui.add(egui::Spinner::new().size(14.0));
                    ui.weak(&self.semantic_status);
                } else if is_active_semantic_search {
                    ui.weak(&self.semantic_status);
                } else if matches!(self.semantic_mode, SearchMode::Clip | SearchMode::Ocr) {
                    ui.weak(&self.semantic_status);
                } else if self.grid_loading {
                    ui.add(egui::Spinner::new().size(14.0));
                    ui.weak("Scanning subdirectories...");
                } else {
                    let label = if self.semantic_video_only { "videos" } else { "items" };
                    ui.weak(format!("{} {} found in this folder", filtered_images.len(), label));
                }
            });
            ui.add_space(6.0);
            ui.separator();
            ui.add_space(8.0);

            if self.db_loading
                && (is_active_semantic_search || self.pending_semantic_search_mode.is_some())
            {
                ui.centered_and_justified(|ui| {
                    ui.vertical_centered(|ui| {
                        ui.add(egui::Spinner::new().size(36.0));
                        ui.add_space(16.0);
                        ui.heading("Lazy-loading AI Database Models & ONNX session...");
                        ui.weak("Initializing standard text encoders and reading index maps. This happens fully in the background.");
                    });
                });
            } else {
                // Populate unified Gallery Items (only if active semantic search is active)
                let mut gallery_items: Vec<GalleryItem> = Vec::new();
                if is_active_semantic_search {
                    let mut seen_semantic_paths: HashSet<PathBuf> = HashSet::new();
                    for item in &self.semantic_results {
                        // Some semantic result producers deliberately search broadly (for
                        // example, "show most similar" and person searches). Apply the
                        // gallery toggle here as the final display filter so it works for
                        // every semantic result source, not only text searches.
                        if self.semantic_video_only && !item.is_video {
                            continue;
                        }
                        if let Some(path) = &item.media_path {
                            if !seen_semantic_paths.insert(path.clone()) {
                                continue;
                            }
                            gallery_items.push(GalleryItem {
                                path: path.clone(),
                                is_video: item.is_video,
                                score_label: Some(match self.semantic_mode {
                                    SearchMode::Clip => format!("{:.0}% Match", (item.score * 100.0).clamp(0.0, 100.0)),
                                    SearchMode::Ocr => {
                                        if item.ocr_phrase_query {
                                            format!(
                                                "{} / {} words (exact phrase)",
                                                item.ocr_term_hits,
                                                item.ocr_query_terms
                                            )
                                        } else if item.ocr_query_terms > 0 {
                                            format!(
                                                "{} / {} words",
                                                item.ocr_term_hits,
                                                item.ocr_query_terms
                                            )
                                        } else {
                                            "OCR Match".to_string()
                                        }
                                    }
                                    _ => String::new(),
                                }),
                                timestamp_sec: item.timestamp_sec,
                                db_filename: Some(item.file_name.clone()),
                            });
                        }
                    }
                }

                let num_items = if is_active_semantic_search {
                    gallery_items.len()
                } else {
                    filtered_images.len()
                };

                if num_items == 0 {
                    ui.centered_and_justified(|ui| {
                        if self.grid_loading {
                            ui.weak("Scanning files, please wait...");
                        } else {
                            ui.weak("No files found matching filter or query.");
                        }
                    });
                } else {
                    let gallery_rect = ui.available_rect_before_wrap();
            let ctrl_zoom_delta = if ui.rect_contains_pointer(gallery_rect) {
                ui.input(|input| {
                    if input.modifiers.ctrl {
                        input.zoom_delta()
                    } else {
                        1.0
                    }
                })
            } else {
                1.0
            };
            if ctrl_zoom_delta != 1.0 {
                self.gallery_thumbnail_scale =
                    (self.gallery_thumbnail_scale * ctrl_zoom_delta).clamp(0.5, 3.0);
                ui.ctx().request_repaint();
            }

                    let tile_width = 130.0 * self.gallery_thumbnail_scale;
                    let tile_height = 160.0 * self.gallery_thumbnail_scale;
                    let available_width = (ui.available_width() - 16.0).max(tile_width);
                    let col_width = tile_width + 12.0;
                    let cols = (available_width / col_width).floor().max(1.0) as usize;
                    let num_rows = (num_items + cols - 1) / cols;
                    let row_height = tile_height + 12.0;
                    let visible_rows = (gallery_rect.height() / row_height).max(1.0);
                    let scroll_speed_multiplier = (10.0 / visible_rows).clamp(0.75, 4.0);

                    let mut double_clicked_item: Option<GalleryItem> = None;
                    let mut single_clicked_item: Option<GalleryItem> = None;
                    let mut clicked_similar: Option<PendingSearchRequest> = None;
                    let mut clicked_person: Option<PendingSearchRequest> = None;
                    let navigation_button_down = ui.input(|input| {
                        input.pointer.button_down(egui::PointerButton::Extra1)
                            || input.pointer.button_down(egui::PointerButton::Extra2)
                    });
                    let gallery_scroll_source = if navigation_button_down {
                        egui::scroll_area::ScrollSource::NONE
                    } else {
                        egui::scroll_area::ScrollSource::ALL
                    };

                    egui::ScrollArea::vertical()
                        .id_salt("gallery_scroll_area")
                        .scroll_source(gallery_scroll_source)
                        .wheel_scroll_multiplier(egui::vec2(1.0, scroll_speed_multiplier))
                        .show_rows(ui, row_height, num_rows, |ui, row_range| {
                        for row_idx in row_range {
                            let start_idx = row_idx * cols;
                            let end_idx = (start_idx + cols).min(num_items);
                            let row_width = (cols as f32 * tile_width)
                                + (cols.saturating_sub(1) as f32 * 12.0);
                            let (row_rect, _) = ui.allocate_exact_size(
                                egui::vec2(row_width, tile_height),
                                egui::Sense::hover(),
                            );

                            for (col_idx, item_idx) in (start_idx..end_idx).enumerate() {
                                    let temp_item;
                                    let item = if is_active_semantic_search {
                                        &gallery_items[item_idx]
                                    } else {
                                        let global_idx = filtered_images[item_idx];
                                        let p = &self.recursive_images[global_idx];
                                        let is_vid = is_video_path(p);
                                        let db_name = self.resolve_db_filename(p);
                                        temp_item = GalleryItem {
                                            path: p.clone(),
                                            is_video: is_vid,
                                            score_label: None,
                                            timestamp_sec: 0.0,
                                            db_filename: db_name,
                                        };
                                        &temp_item
                                    };
                                    let path = &item.path;
                                    let is_selected = self.selected_grid_items.iter().any(|selected| {
                                        selected.matches(path, item.db_filename.as_deref())
                                    });
                                    let is_current = if let Some(curr_p) = self.images.get(self.current_index) {
                                        curr_p == path
                                    } else {
                                        false
                                    };

                                    let rect = egui::Rect::from_min_size(
                                        egui::pos2(row_rect.min.x + col_idx as f32 * col_width, row_rect.min.y),
                                        egui::vec2(tile_width, tile_height),
                                    );
                                    let response = ui.interact(
                                        rect,
                                        ui.make_persistent_id(("gallery_card", item_idx)),
                                        egui::Sense::click(),
                                    );

                                    response.context_menu(|ui| {
                                        if ui.button("📂 Show in parent folder").clicked() {
                                            let actual_path = self.resolve_actual_path(path);
                                            open_in_dolphin_or_fallback(&actual_path);
                                            ui.close();
                                        }
                                        if ui.button("📋 Copy image").clicked() {
                                            let resolved_path = self.get_thumbnail_path(path);
                                            if let Err(err) = copy_image_file_to_clipboard(&resolved_path) {
                                                self.semantic_status = format!("Copy image failed: {err}");
                                            }
                                            ui.close();
                                        }
                                        if ui.button("📋 Copy full path").clicked() {
                                            ui.ctx().copy_text(path.to_string_lossy().to_string());
                                            ui.close();
                                        }
                                        if item.is_video {
                                            if ui.button("🎬 Open in mpv").clicked() {
                                                let playback_path = if let Some(db_name) = &item.db_filename {
                                                    let roots = get_db_roots();
                                                    resolve_source_path(&roots, db_name)
                                                        .ok()
                                                        .unwrap_or_else(|| self.resolve_actual_path(path))
                                                } else {
                                                    self.resolve_actual_path(path)
                                                };
                                                let _ = std::process::Command::new("mpv")
                                                    .arg(format!("--start={:.3}", item.timestamp_sec.max(0.0)))
                                                    .arg(playback_path)
                                                    .spawn();
                                                ui.close();
                                            }
                                        } else if ui.button("✏ Edit image").clicked() {
                                            self.start_image_editor(path, ui.ctx());
                                            ui.close();
                                        }
                                        ui.separator();
                                        if ui.button("Show most similar").clicked() {
                                            clicked_similar = Some(PendingSearchRequest::Similar {
                                                db_file_name: item.db_filename.clone(),
                                                media_path: path.clone(),
                                                is_video: item.is_video,
                                                timestamp_sec: item.timestamp_sec,
                                            });
                                            ui.close();
                                        }
                                        if ui.button("Show more of this person").clicked() {
                                            clicked_person = Some(PendingSearchRequest::Person {
                                                db_file_name: item.db_filename.clone(),
                                                media_path: path.clone(),
                                                is_video: item.is_video,
                                            });
                                            ui.close();
                                        }
                                    });

                                    let is_hovered = response.hovered();
                                    let is_clicked = response.clicked();

                                    let card_bg = if is_selected {
                                        ui.visuals().selection.bg_fill.gamma_multiply(0.45)
                                    } else if is_clicked {
                                        ui.visuals().selection.bg_fill.gamma_multiply(0.3)
                                    } else if is_hovered {
                                        ui.visuals().code_bg_color.gamma_multiply(1.5)
                                    } else if is_current {
                                        ui.visuals().selection.bg_fill.gamma_multiply(0.15)
                                    } else {
                                        ui.visuals().code_bg_color
                                    };

                                    let card_stroke = if is_selected {
                                        egui::Stroke::new(3.0, egui::Color32::from_rgb(100, 200, 120))
                                    } else if is_current {
                                        egui::Stroke::new(2.0, ui.visuals().selection.bg_fill)
                                    } else if is_hovered {
                                        egui::Stroke::new(1.0, ui.visuals().selection.bg_fill.gamma_multiply(0.5))
                                    } else {
                                        egui::Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color.gamma_multiply(0.3))
                                    };

                                    let builder = egui::UiBuilder::new()
                                        .max_rect(rect)
                                        .id_salt((path, item_idx));
                                    let mut child_ui = ui.new_child(builder);
                                    egui::Frame::NONE
                                        .fill(card_bg)
                                        .stroke(card_stroke)
                                        .inner_margin(0.0)
                                        .corner_radius(6.0)
                                        .show(&mut child_ui, |ui| {
                                            let resolved_path = self.get_thumbnail_path(path);
                                            if let Some(texture) = self.thumbnail_textures.get(&resolved_path) {
                                                ui.centered_and_justified(|ui| {
                                                    ui.add(
                                                        egui::Image::from_texture(texture)
                                                            .max_size(egui::vec2(tile_width, tile_height))
                                                            .maintain_aspect_ratio(false)
                                                    );
                                                });
                                            } else if self.thumbnail_failed.contains(&resolved_path) {
                                                ui.vertical_centered(|ui| {
                                                    ui.add_space(55.0 * self.gallery_thumbnail_scale);
                                                    if is_video_path(path) {
                                                        ui.weak("📹 Video");
                                                    } else {
                                                        ui.weak("⚠️ Failed");
                                                    }
                                                });
                                            } else {
                                                ui.vertical_centered(|ui| {
                                                    ui.add_space(60.0 * self.gallery_thumbnail_scale);
                                                    ui.add(egui::Spinner::new().size(20.0 * self.gallery_thumbnail_scale));
                                                });

                                                let max_threads = num_cpus::get().max(4);
                                                if !self.thumbnail_loading.contains(&resolved_path) && self.thumbnail_active_threads < max_threads {
                                                    self.thumbnail_loading.insert(resolved_path.clone());
                                                    self.thumbnail_active_threads += 1;
                                                    let path_clone = resolved_path.clone();
                                                    let tx_clone = self.thumbnail_tx.clone();
                                                    let ctx_clone = ui.ctx().clone();
                                                    rayon::spawn(move || {
                                                        if let Ok(img) = image::open(&path_clone) {
                                                            let thumb = img.resize_to_fill(260, 320, image::imageops::FilterType::Triangle);
                                                            let size = [thumb.width() as usize, thumb.height() as usize];
                                                            let pixels = thumb.to_rgba8().into_raw();
                                                            let color_img = egui::ColorImage::from_rgba_unmultiplied(size, &pixels);
                                                            let _ = tx_clone.send((path_clone, color_img));
                                                            ctx_clone.request_repaint();
                                                        } else {
                                                            let empty_img = egui::ColorImage::new([0, 0], Vec::new());
                                                            let _ = tx_clone.send((path_clone, empty_img));
                                                            ctx_clone.request_repaint();
                                                        }
                                                    });
                                                }
                                            }
                                        });

                    // Overlay 1: Filename banner at the bottom (semi-transparent black with rounded bottom corners)
                                    let overlay_scale = self.gallery_thumbnail_scale;
                                    let banner_height = 24.0 * overlay_scale;
                                    let banner_rect = egui::Rect::from_min_max(
                                        egui::pos2(rect.min.x, rect.max.y - banner_height),
                                        rect.max
                                    );
                                    let banner_rounding = egui::CornerRadius {
                                        nw: 0,
                                        ne: 0,
                                        sw: (6.0 * overlay_scale).round().clamp(0.0, 255.0) as u8,
                                        se: (6.0 * overlay_scale).round().clamp(0.0, 255.0) as u8,
                                    };
                                    ui.painter().rect_filled(banner_rect, banner_rounding, egui::Color32::from_black_alpha(180));

                                    let filename_owned = if let Some(db_name) = &item.db_filename {
                                        db_name
                                            .split_once('/')
                                            .map(|(_, rel)| rel)
                                            .and_then(|rel| Path::new(rel).file_name())
                                            .and_then(|s| s.to_str())
                                            .map(|s| s.to_string())
                                            .unwrap_or_else(|| {
                                                path.file_name()
                                                    .and_then(|s| s.to_str())
                                                    .unwrap_or("")
                                                    .to_string()
                                            })
                                    } else {
                                        path.file_name()
                                            .and_then(|s| s.to_str())
                                            .unwrap_or("")
                                            .to_string()
                                    };
                                    let filename = filename_owned.as_str();
                                    let filename_label = if filename.chars().count() > 22 {
                                        format!("{}...", filename.chars().take(19).collect::<String>())
                                    } else {
                                        filename.to_string()
                                    };
                                    ui.painter().text(
                                        banner_rect.center(),
                                        egui::Align2::CENTER_CENTER,
                                        filename_label,
                                        egui::FontId::proportional(9.0 * overlay_scale),
                                        egui::Color32::WHITE,
                                    );

                                    // Overlay 2: Score / Match Badge pill overlay in the top-left
                                    if let Some(lbl) = &item.score_label {
                                        let badge_rect = egui::Rect::from_min_max(
                                            egui::pos2(
                                                rect.min.x + 6.0 * overlay_scale,
                                                rect.min.y + 6.0 * overlay_scale,
                                            ),
                                            egui::pos2(
                                                rect.min.x + 66.0 * overlay_scale,
                                                rect.min.y + 22.0 * overlay_scale,
                                            )
                                        );
                                        let badge_bg = if lbl.contains("Match") && !lbl.contains("OCR") {
                                            egui::Color32::from_rgb(16, 124, 65).gamma_multiply(0.85)
                                        } else {
                                            egui::Color32::from_rgb(0, 90, 158).gamma_multiply(0.85)
                                        };
                                        ui.painter().rect_filled(
                                            badge_rect,
                                            4.0 * overlay_scale,
                                            badge_bg,
                                        );
                                        ui.painter().text(
                                            badge_rect.center(),
                                            egui::Align2::CENTER_CENTER,
                                            lbl,
                                            egui::FontId::proportional(8.0 * overlay_scale),
                                            egui::Color32::WHITE,
                                        );
                                    }

                                    // Overlay 3: Video indicator badge with full duration in the top-right.
                                    if item.is_video {
                                        let video_source_path = self.video_source_path_for_tile(
                                            path,
                                            item.db_filename.as_deref(),
                                        );
                                        let badge_text = self
                                            .cached_video_metadata(&video_source_path, ctx)
                                            .and_then(|metadata| metadata.duration_sec)
                                            .map(|duration| format!("📹 {}", format_video_duration(duration)))
                                            .unwrap_or_else(|| "📹 Video".to_string());

                                        let badge_rect = egui::Rect::from_min_max(
                                            egui::pos2(
                                                rect.max.x - 78.0 * overlay_scale,
                                                rect.min.y + 6.0 * overlay_scale,
                                            ),
                                            egui::pos2(
                                                rect.max.x - 6.0 * overlay_scale,
                                                rect.min.y + 22.0 * overlay_scale,
                                            )
                                        );
                                        ui.painter().rect_filled(
                                            badge_rect,
                                            4.0 * overlay_scale,
                                            egui::Color32::from_black_alpha(160),
                                        );
                                        ui.painter().text(
                                            badge_rect.center(),
                                            egui::Align2::CENTER_CENTER,
                                            badge_text,
                                            egui::FontId::proportional(8.0 * overlay_scale),
                                            egui::Color32::WHITE,
                                        );
                                    }

                                    let ctrl_clicked = response.clicked()
                                        && ui.input(|i| i.modifiers.matches_logically(egui::Modifiers::CTRL));
                                    if ctrl_clicked {
                                        let source_path = if item.is_video {
                                            self.video_source_path_for_tile(
                                                path,
                                                item.db_filename.as_deref(),
                                            )
                                        } else {
                                            self.resolve_actual_path(path)
                                        };
                                        if let Some(pos) = self.selected_grid_items.iter().position(
                                            |selected| {
                                                selected.matches(
                                                    &source_path,
                                                    item.db_filename.as_deref(),
                                                )
                                            },
                                        ) {
                                            self.selected_grid_items.remove(pos);
                                        } else {
                                            self.selected_grid_items.push(GallerySelection {
                                                path: source_path,
                                                db_filename: item.db_filename.clone(),
                                                is_video: item.is_video,
                                            });
                                        }
                                        if !self.db_loaded && !self.db_loading && !self.db_failed {
                                            self.start_lazy_db_load(ui.ctx());
                                        }
                                        let selected_count = self.selected_grid_items.len();
                                        self.semantic_status =
                                            format!("{selected_count} media item(s) selected.");
                                    } else if response.double_clicked() {
                                        double_clicked_item = Some(item.clone());
                                    } else if response.clicked() {
                                        single_clicked_item = Some(item.clone());
                                    }
                            }
                        }
                    });

                    if let Some(item) = double_clicked_item {
                        let path = item.path.clone();
                        if let Some(db_name) = &item.db_filename {
                            self.db_filename_by_path.insert(path.clone(), db_name.clone());
                        }

                        if item.is_video {
                            let playback_path = if let Some(db_name) = &item.db_filename {
                                let roots = get_db_roots();
                                resolve_source_path(&roots, db_name)
                                    .ok()
                                    .unwrap_or_else(|| self.resolve_actual_path(&path))
                            } else {
                                self.resolve_actual_path(&path)
                            };
                            let _ = std::process::Command::new("mpv")
                                .arg(format!("--start={:.3}", item.timestamp_sec.max(0.0)))
                                .arg(playback_path)
                                .spawn();
                        } else {
                        let active_paths: Vec<PathBuf> = if is_active_semantic_search {
                            for item in &gallery_items {
                                if let Some(db_name) = &item.db_filename {
                                    self.db_filename_by_path.insert(item.path.clone(), db_name.clone());
                                }
                            }
                            gallery_items.iter().map(|item| item.path.clone()).collect()
                        } else {
                            if let Some(db_name) = self.resolve_db_filename(&path) {
                                self.db_filename_by_path.insert(path.clone(), db_name);
                            }
                            filtered_images.iter().map(|&idx| self.recursive_images[idx].clone()).collect()
                        };
                        self.images = active_paths;
                        self.current_index = self.images.iter().position(|p| p == &path).unwrap_or(0);
                        self.remember_gallery_image();
                        self.show_grid = false;
                        self.back_target_is_gallery = true;
                        self.zoom = 1.0;
                        self.offset = egui::Vec2::ZERO;
                        self.update_current_file_info();
                        self.update_side_panel_metadata_if_needed();
                        ui.ctx().request_repaint();
                        }
                    }

                    if let Some(item) = single_clicked_item {
                        let path = item.path.clone();
                        let active_paths: Vec<PathBuf> = if is_active_semantic_search {
                            for item in &gallery_items {
                                if let Some(db_name) = &item.db_filename {
                                    self.db_filename_by_path.insert(item.path.clone(), db_name.clone());
                                }
                            }
                            gallery_items.iter().map(|item| item.path.clone()).collect()
                        } else {
                            if let Some(db_name) = self.resolve_db_filename(&path) {
                                self.db_filename_by_path.insert(path.clone(), db_name);
                            }
                            filtered_images.iter().map(|&idx| self.recursive_images[idx].clone()).collect()
                        };
                        if let Some(pos) = active_paths.iter().position(|p| p == &path) {
                            self.images = active_paths;
                            self.current_index = pos;
                            self.update_current_file_info();
                            if self.show_exif || self.side_panel_open_pending {
                                self.update_side_panel_metadata_if_needed();
                            } else {
                                self.open_side_panel(ui.ctx(), SidePanelMode::Duplicates);
                            }
                            ui.ctx().request_repaint();
                        }
                    }

                    if let Some(request) = clicked_similar {
                        self.request_search_action(request, ui.ctx());
                        ui.ctx().request_repaint();
                    }

                    if let Some(request) = clicked_person {
                        self.request_search_action(request, ui.ctx());
                        ui.ctx().request_repaint();
                    }
                }
            }
        });
    }
}
