use super::*;

impl eframe::App for ImageViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let frame_scroll_delta = ctx.input(|i| {
            if i.smooth_scroll_delta.y.abs() > f32::EPSILON {
                i.smooth_scroll_delta.y
            } else {
                i.raw_scroll_delta.y
            }
        });

        if let Some(size) = self.pending_initial_window_size.take() {
            ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(size));
        }

        if let Ok(mut lock) = self.ctx_shared.lock() {
            if lock.is_none() {
                *lock = Some(ctx.clone());
            }
        }

        self.poll_db_load();
        self.poll_sift_alignment();
        self.poll_sift_align_all();
        self.poll_sift_repair(ctx);
        self.poll_on_demand_embeddings(ctx);
        self.poll_face_compare(ctx);

        if !self.db_loaded && !self.db_loading {
            let is_ai = if let Some(p) = self.images.get(self.current_index) {
                is_path_ai_backed(p)
            } else if is_path_ai_backed(&self.open_target) {
                true
            } else {
                false
            };
            if is_ai {
                self.start_lazy_db_load(ctx);
            }
        }

        while let Ok((path, color_image)) = self.thumbnail_rx.try_recv() {
            if color_image.size[0] == 0 {
                self.thumbnail_failed.insert(path.clone());
            } else {
                let texture = ctx.load_texture(
                    path.to_string_lossy(),
                    color_image,
                    egui::TextureOptions::default(),
                );
                self.thumbnail_textures.insert(path.clone(), texture);
            }
            self.thumbnail_loading.remove(&path);
            self.thumbnail_active_threads = self.thumbnail_active_threads.saturating_sub(1);
            ctx.request_repaint();
        }

        while let Ok((path, revision, result)) = self.viewer_texture_rx.try_recv() {
            let current_revision = self
                .viewer_texture_revisions
                .get(&path)
                .copied()
                .unwrap_or(0);
            if revision != current_revision {
                continue;
            }
            self.viewer_texture_loading.remove(&path);
            match result {
                Ok(color_image) => {
                    let texture = ctx.load_texture(
                        format!("viewer_image: {}", path.display()),
                        color_image,
                        egui::TextureOptions::LINEAR,
                    );
                    self.viewer_textures.insert(path, texture);
                }
                Err(_) => {
                    self.viewer_texture_failed.insert(path);
                }
            }
            ctx.request_repaint();
        }
        self.trim_viewer_texture_cache();

        while let Ok((path, metadata)) = self.video_duration_rx.try_recv() {
            self.video_duration_loading.borrow_mut().remove(&path);
            self.video_duration_cache
                .borrow_mut()
                .insert(path, metadata);
            ctx.request_repaint();
        }

        if let Ok(request) = self.rx.try_recv() {
            match request {
                OpenRequest::Single(path) => self.open_image_path(path),
                OpenRequest::Comparison(paths) => self.open_comparison_paths(paths, ctx),
            }
            self.show_home_page = false;
            ctx.request_repaint();
        }

        if let Some(rx) = &self.recursive_rx {
            let mut new_images = Vec::new();
            let mut disconnected = false;
            loop {
                match rx.try_recv() {
                    Ok(path) => new_images.push(path),
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
            if !new_images.is_empty() {
                self.recursive_images.extend(new_images);
                ctx.request_repaint();
            }
            if disconnected {
                self.recursive_images.sort_by(|a, b| b.cmp(a));
                self.recursive_video_indices = self
                    .recursive_images
                    .iter()
                    .enumerate()
                    .filter_map(|(index, path)| is_video_path(path).then_some(index))
                    .collect();
                self.grid_loading = false;
                self.recursive_rx = None;
                ctx.request_repaint();
            }
        }

        if self.flat_loading {
            let mut collected = None;
            if let Ok(mut lock) = self.flat_images_shared.try_lock() {
                if let Some(imgs) = lock.take() {
                    collected = Some(imgs);
                }
            }
            if let Some(imgs) = collected {
                let current_path = self
                    .images
                    .get(self.current_index)
                    .cloned()
                    .or_else(|| (!self.open_target_is_dir).then(|| self.open_target.clone()));
                self.images = imgs;
                self.current_index = current_path
                    .as_ref()
                    .and_then(|path| self.images.iter().position(|candidate| candidate == path))
                    .or_else(|| {
                        self.images
                            .iter()
                            .position(|path| path == &self.open_target)
                    })
                    .unwrap_or(0);
                self.flat_loading = false;
                self.flat_refresh_in_flight = false;
                self.flat_directory_mtime = self.current_flat_directory_mtime();
                self.update_current_file_info();
                self.update_side_panel_metadata_if_needed();
                ctx.request_repaint();
            }
        }
        if !self.show_home_page && !self.show_grid && !self.open_target_is_dir {
            self.poll_flat_directory_refresh(ctx);
        }
        // Mouse Back click handling:
        // allow returning to gallery if explicitly marked, or if a gallery list is available.
        let can_back_to_gallery = self.back_target_is_gallery
            || (!self.show_home_page && !self.recursive_images.is_empty());
        let back_clicked = ctx.input(|i| i.pointer.button_clicked(egui::PointerButton::Extra1));
        let forward_clicked = ctx.input(|i| i.pointer.button_clicked(egui::PointerButton::Extra2));

        // Some window/input configurations report browser-button clicks alongside a
        // scroll delta. Navigation buttons must never move the active scroll area.
        if back_clicked || forward_clicked {
            ctx.input_mut(|i| {
                i.raw_scroll_delta = egui::Vec2::ZERO;
                i.smooth_scroll_delta = egui::Vec2::ZERO;
            });
        }

        if back_clicked {
            if !self.show_grid && can_back_to_gallery {
                self.show_grid = true;
                self.back_target_is_gallery = true;
                ctx.request_repaint();
            } else if self.show_grid {
                self.restore_previous_search(ctx);
            }
        }
        if forward_clicked && self.show_grid {
            if self.back_target_is_gallery {
                self.restore_gallery_image(ctx);
            } else if !self.restore_next_search(ctx) {
                // A search-history forward step may lead back to the gallery
                // state that preceded the viewed image. Restore that image only
                // after the search-history stack has been exhausted.
                self.restore_gallery_image(ctx);
            }
        }

        // Keyboard handling
        if !ctx.wants_keyboard_input() {
            let mut open_file_requested = false;
            let mut compare_paths_requested = false;
            ctx.input(|i| {
                if (i.modifiers.matches_logically(egui::Modifiers::COMMAND)
                    || i.modifiers.matches_logically(egui::Modifiers::CTRL))
                    && i.key_pressed(egui::Key::O)
                {
                    if i.modifiers.shift {
                        compare_paths_requested = true;
                    } else {
                        open_file_requested = true;
                    }
                }
                if !self.show_home_page {
                    if !self.show_grid && self.image_editor.is_none() {
                        if i.key_pressed(egui::Key::ArrowRight) {
                            self.next_image();
                        }
                        if i.key_pressed(egui::Key::ArrowLeft) {
                            self.prev_image();
                        }
                        if i.key_pressed(egui::Key::F) {
                            self.zoom = 1.0;
                            self.offset = egui::Vec2::ZERO;
                        }
                        if i.key_pressed(egui::Key::Num0) {
                            self.zoom = 1.0;
                            self.offset = egui::Vec2::ZERO;
                        }
                    }
                    if i.key_pressed(egui::Key::G) {
                        if !self.show_grid {
                            self.clear_comparison_mode();
                        }
                        self.show_grid = !self.show_grid;
                        if self.show_grid && self.recursive_images.is_empty() {
                            self.start_recursive_scan();
                        }
                    }
                    if i.key_pressed(egui::Key::E) {
                        self.toggle_layout_side_panel(ctx);
                    }
                    if i.key_pressed(egui::Key::Backspace) {
                        self.show_home_page = true;
                    }
                }
                if i.key_pressed(egui::Key::Q) {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
                if i.key_pressed(egui::Key::Escape) {
                    if self.image_editor.is_some() {
                        self.image_editor = None;
                    } else if self.show_home_page {
                        if self.home_current_dir.is_some() {
                            self.home_current_dir = None;
                        } else {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    } else if self.show_grid {
                        self.show_home_page = true;
                    } else {
                        self.show_grid = true;
                    }
                }
            });
            if compare_paths_requested {
                self.open_comparison_path_dialog();
            } else if open_file_requested {
                self.open_file_dialog(ctx);
            }
        }

        self.show_comparison_path_dialog(ctx);

        if self.show_home_page {
            self.show_home_page_view(ctx);
            return;
        }

        egui::TopBottomPanel::top("top_bar")
            .exact_height(IMAGE_VIEWER_TOP_BAR_HEIGHT)
            .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Iris");
                ui.separator();

                if ui.button("🏠 Filesystem").clicked() {
                    self.clear_comparison_mode();
                    self.show_home_page = true;
                }
                if ui.button("Open File [Ctrl+O]").clicked() {
                    self.open_file_dialog(ctx);
                }
                if ui.button("Compare Paths [Ctrl+Shift+O]").clicked() {
                    self.open_comparison_path_dialog();
                }

                if self.is_comparison_mode() {
                    ui.separator();
                    ui.colored_label(
                        egui::Color32::from_rgb(255, 190, 80),
                        format!(
                            "Compare {}/{} (←/→)",
                            self.current_index + 1,
                            self.images.len()
                        ),
                    );
                    if ui
                        .checkbox(&mut self.comparison_sync_view, "Sync View")
                        .on_hover_text(
                            "Share the current zoom and pan position across every comparison image.",
                        )
                        .changed()
                        && self.comparison_sync_view
                    {
                        self.apply_comparison_view_state_to_all();
                    }
                    let can_align_all = !self.sift_align_all_running
                        && self.images.iter().all(|path| !is_video_path(path));
                    let align_label = if self.sift_align_all_running {
                        "SIFT Aligning..."
                    } else if !self.comparison_aligned_paths.is_empty() {
                        "SIFT Re-align All"
                    } else {
                        "SIFT Align All"
                    };
                    if ui
                        .add_enabled(can_align_all, egui::Button::new(align_label))
                        .on_hover_text(
                            "Use the first comparison image as the reference and recompute every SIFT alignment.",
                        )
                        .clicked()
                    {
                        self.start_sift_align_all(ctx);
                    }
                    if !self.comparison_alignment_status.is_empty() {
                        ui.weak(&self.comparison_alignment_status);
                    }
                    if ui.button("Exit Compare").clicked() {
                        if let Some(path) = self.images.get(self.current_index).cloned() {
                            self.open_image_path(path);
                        }
                    }
                }

                ui.separator();
                if let Some(path) = self.images.get(self.current_index) {
                    let filename = self
                        .resolve_db_filename(path)
                        .and_then(|db_name| {
                            db_name
                                .split_once('/')
                                .map(|(_, rel)| rel.to_string())
                        })
                        .and_then(|rel| {
                            Path::new(&rel)
                                .file_name()
                                .and_then(|f| f.to_str())
                                .map(|s| s.to_string())
                        })
                        .unwrap_or_else(|| {
                            path.file_name()
                                .and_then(|f| f.to_str())
                                .unwrap_or("")
                                .to_string()
                        });
                    ui.label(format!("{} ({}/{}) - {} - {}", filename, self.current_index + 1, self.images.len(), self.current_dimensions, self.current_file_size));
                } else {
                    ui.label("No image loaded");
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Layout Button
                    let show_layout_active = (self.show_exif || self.side_panel_open_pending)
                        && self.side_panel_mode == SidePanelMode::Layout;
                    let layout_button_text = if show_layout_active { "📂 Hide Layout [E]" } else { "📂 Show Layout [E]" };
                    if ui.button(layout_button_text).clicked() {
                        self.toggle_layout_side_panel(ctx);
                    }

                    ui.add_space(8.0);

                    let gallery_text = if self.show_grid { "🖼 Hide Gallery [G]" } else { "🖼 Show Gallery [G]" };
                    if ui.button(gallery_text).clicked() {
                        if !self.show_grid {
                            self.clear_comparison_mode();
                        }
                        self.show_grid = !self.show_grid;
                        if self.show_grid && self.recursive_images.is_empty() {
                            self.start_recursive_scan();
                        }
                    }
                });
            });
        });

        self.apply_pending_side_panel_open(ctx);

        // Side panel opens only after the native window has expanded, avoiding gallery reflow flicker.
        if self.show_exif {
            egui::SidePanel::right("exif_panel")
                .resizable(false)
                .exact_width(Self::SIDE_PANEL_WIDTH)
                .show(ctx, |ui| {
                // Header Tabs
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.side_panel_mode, SidePanelMode::Layout, "📂 Binary Layout");
                    ui.selectable_value(&mut self.side_panel_mode, SidePanelMode::Exif, "🏷 Raw EXIF");
                    ui.selectable_value(&mut self.side_panel_mode, SidePanelMode::Duplicates, "👥 Duplicates");

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("❌").clicked() {
                            self.close_side_panel(ui.ctx());
                        }
                    });
                });
                ui.separator();
                ui.add_space(4.0);

                match self.side_panel_mode {
                    SidePanelMode::Layout => {
                        self.update_layout_if_needed();
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            if self.chunks.is_empty() {
                                ui.label("No layout data available.");
                            } else {
                                for chunk in &self.chunks {
                                    let size_str = if chunk.length >= 1048576 {
                                        format!("{:.2} MB", chunk.length as f64 / 1048576.0)
                                    } else if chunk.length >= 1024 {
                                        format!("{:.1} KB", chunk.length as f64 / 1024.0)
                                    } else {
                                        format!("{} B", chunk.length)
                                    };

                                    egui::Frame::NONE
                                        .fill(ui.visuals().code_bg_color)
                                        .stroke(egui::Stroke::new(1.0, chunk.color.gamma_multiply(0.3)))
                                        .inner_margin(8.0)
                                        .corner_radius(6.0)
                                        .show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                let (rect, _response) = ui.allocate_at_least(egui::vec2(6.0, 32.0), egui::Sense::hover());
                                                ui.painter().rect_filled(rect, 3.0, chunk.color);

                                                ui.vertical(|ui| {
                                                    let is_system = chunk.name == "System Metadata";
                                                    let default_open = is_system;
                                                    let id = ui.make_persistent_id(chunk.offset + if default_open { 99999 } else { 0 });
                                                    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, default_open)
                                                        .show_header(ui, |ui| {
                                                            ui.horizontal(|ui| {
                                                                ui.colored_label(ui.visuals().strong_text_color(), &chunk.name);
                                                                if !is_system {
                                                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                                                        ui.weak(&size_str);
                                                                    });
                                                                }
                                                            });
                                                        })
                                                        .body(|ui| {
                                                            if !is_system {
                                                                ui.add_space(4.0);
                                                                ui.horizontal(|ui| {
                                                                    ui.weak(format!("Offset: 0x{:08X}", chunk.offset));
                                                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                                                        ui.weak(format!("Len: {}", chunk.length));
                                                                    });
                                                                });
                                                            }
                                                            ui.add_space(4.0);
                                                             egui::ScrollArea::horizontal().show(ui, |ui| {
                                                                 ui.add(egui::Label::new(egui::RichText::new(&chunk.parsed_data).monospace()).selectable(true));
                                                             });
                                                        });
                                                });
                                            });
                                        }).response.on_hover_text(&chunk.description);

                                    ui.add_space(6.0);
                                }
                            }
                        });
                    }
                    SidePanelMode::Exif => {
                        self.update_side_panel_metadata_if_needed();
                        ui.horizontal(|ui| {
                            ui.label("Search:");
                            ui.add(egui::TextEdit::singleline(&mut self.exif_search)
                                .hint_text("🔍 Search EXIF tags...")
                                .desired_width(180.0));
                            if ui.button("❌").clicked() {
                                self.exif_search.clear();
                            }
                        });
                        ui.add_space(6.0);
                        ui.separator();
                        ui.add_space(6.0);

                        egui::ScrollArea::vertical().show(ui, |ui| {
                            if self.exif_data.is_empty() {
                                ui.label("No EXIF data available.");
                            } else {
                                let filter = self.exif_search.to_lowercase();
                                let filtered_lines: Vec<String> = self.exif_data
                                    .lines()
                                    .filter(|line| {
                                        if filter.is_empty() {
                                            true
                                        } else {
                                            line.to_lowercase().contains(&filter)
                                        }
                                    })
                                    .map(|s| s.to_string())
                                    .collect();

                                if filtered_lines.is_empty() {
                                    ui.weak("No matching tags found.");
                                } else {
                                    let content = filtered_lines.join("\n");
                                    egui::ScrollArea::horizontal().show(ui, |ui| {
                                        ui.add(
                                            egui::Label::new(
                                                egui::RichText::new(content)
                                                    .monospace()
                                                    .size(11.0)
                                            )
                                            .selectable(true)
                                        );
                                    });
                                }
                            }
                        });
                    }
                    SidePanelMode::Duplicates => {
                        if !self.db_loaded || !self.db_supplemental_loaded {
                            if !self.db_failed && !self.db_loading {
                                self.start_lazy_db_load(ui.ctx());
                            }
                            ui.vertical_centered(|ui| {
                                ui.add_space(20.0);
                                if self.db_failed || (self.db_loaded && !self.db_supplemental_loading) {
                                    ui.colored_label(egui::Color32::from_rgb(255, 100, 100), &self.semantic_status);
                                } else {
                                    ui.add(egui::Spinner::new().size(24.0));
                                    ui.add_space(12.0);
                                    ui.weak("Loading duplicate and SIFT indexes...");
                                }
                            });
                        } else if let Some(path) = self.images.get(self.current_index).cloned() {
                            let filename_opt = self.resolve_db_filename(&path);
                            if let Some(filename) = filename_opt {
                                let indices = self.db_indices.as_ref().unwrap();

                                // Check if the current file is a video in the DB
                                let current_is_video = is_video_path(Path::new(&filename));

                                // Resolve grouped master: prefer the current image's SIFT
                                // component, then fall back through pHash/VideoHash.
                                let master_file_name = if current_is_video {
                                    indices.phash_master_by_file
                                        .get(&filename)
                                        .cloned()
                                        .unwrap_or_else(|| filename.clone())
                                } else {
                                    indices.sift_root_by_file
                                        .get(&filename)
                                        .cloned()
                                        .unwrap_or_else(|| {
                                            let phash_master = indices.phash_master_by_file
                                                .get(&filename)
                                                .cloned()
                                                .unwrap_or_else(|| filename.clone());
                                            indices.sift_root_by_file
                                                .get(&phash_master)
                                                .cloned()
                                                .unwrap_or(phash_master)
                                        })
                                };

                                // Fetch SIFT members in this group
                                let sift_members = indices.sift_members_by_root
                                    .get(&master_file_name)
                                    .cloned()
                                    .unwrap_or_default();

                                let mut displayed_sift_members = Vec::new();
                                let mut displayed_seen = HashSet::new();
                                if displayed_seen.insert(filename.clone()) {
                                    displayed_sift_members.push(filename.clone());
                                }
                                for member in &sift_members {
                                    if displayed_seen.insert(member.clone()) {
                                        displayed_sift_members.push(member.clone());
                                    }
                                }

                                let phash_group_seeds: Vec<String> = if sift_members.is_empty() {
                                    vec![master_file_name.clone()]
                                } else {
                                    let mut seeds = Vec::new();
                                    let mut seen = HashSet::new();
                                    if seen.insert(master_file_name.clone()) {
                                        seeds.push(master_file_name.clone());
                                    }
                                    for member in &sift_members {
                                        if seen.insert(member.clone()) {
                                            seeds.push(member.clone());
                                        }
                                    }
                                    seeds
                                };
                                let clip_embedded_files = Arc::clone(&indices.clip_embedded_files);
                                let ocr_embedded_files = Arc::clone(&indices.ocr_embedded_files);
                                let skipped_processing_files = Arc::clone(&indices.skipped_processing_files);
                                let use_sift_seed_similarity = sift_members.len() > 1;

                                let mut phash_similar_groups: Vec<(String, String, Vec<SimilarFile>)> = Vec::new();
                                let mut video_hash_similar_groups: Vec<(String, String, Vec<SimilarFile>)> = Vec::new();
                                for (seed_index, seed) in phash_group_seeds.iter().enumerate() {
                                    if let Some(items) = indices.similar_by_master.get(seed.as_str()) {
                                        let mut group_items = items.clone();
                                        let similarity_reference = if use_sift_seed_similarity {
                                            seed
                                        } else {
                                            &filename
                                        };
                                        for item in &mut group_items {
                                            item.similarity_pct = similarity_to_active(
                                                similarity_reference,
                                                &item.file_name,
                                                &indices.phash_by_file,
                                                &indices.video_frame_phashes_by_file,
                                            );
                                        }
                                        if !group_items.iter().any(|item| item.file_name == *seed) {
                                            group_items.push(SimilarFile {
                                                file_name: seed.clone(),
                                                is_video: is_video_path(Path::new(seed)),
                                                similarity_pct: similarity_to_active(
                                                    similarity_reference,
                                                    seed,
                                                    &indices.phash_by_file,
                                                    &indices.video_frame_phashes_by_file,
                                                ),
                                            });
                                        }
                                        group_items.retain(|item| item.file_name != filename);
                                        group_items.sort_by(|a, b| {
                                            b.similarity_pct
                                                .unwrap_or(f32::NEG_INFINITY)
                                                .partial_cmp(&a.similarity_pct.unwrap_or(f32::NEG_INFINITY))
                                                .unwrap_or(Ordering::Equal)
                                                .then_with(|| a.file_name.cmp(&b.file_name))
                                        });
                                        let mut phash_items = Vec::new();
                                        let mut video_hash_items = Vec::new();
                                        for item in group_items {
                                            let item_is_video = item.is_video
                                                || is_video_path(Path::new(&item.file_name));
                                            if current_is_video && item_is_video {
                                                video_hash_items.push(item);
                                            } else {
                                                phash_items.push(item);
                                            }
                                        }
                                        if !phash_items.is_empty() {
                                            let section_label = if use_sift_seed_similarity {
                                                format!("SIFT master {} pHash similars", seed_index + 1)
                                            } else if current_is_video {
                                                "Active video pHash similars".to_string()
                                            } else {
                                                "Active image pHash similars".to_string()
                                            };
                                            phash_similar_groups.push((seed.clone(), section_label, phash_items));
                                        }
                                        if !video_hash_items.is_empty() {
                                            let section_label = if use_sift_seed_similarity {
                                                format!("SIFT master {} VideoHash similars", seed_index + 1)
                                            } else {
                                                "Active video VideoHash similars".to_string()
                                            };
                                            video_hash_similar_groups.push((
                                                seed.clone(),
                                                section_label,
                                                video_hash_items,
                                            ));
                                        }
                                    }
                                }
                                let mut phash_unique_files = HashSet::new();
                                for (seed, _, items) in &phash_similar_groups {
                                    if seed != &filename {
                                        phash_unique_files.insert(seed.clone());
                                    }
                                    for item in items {
                                        phash_unique_files.insert(item.file_name.clone());
                                    }
                                }
                                let phash_unique_count = phash_unique_files.len();
                                let mut video_hash_unique_files = HashSet::new();
                                for (seed, _, items) in &video_hash_similar_groups {
                                    if seed != &filename {
                                        video_hash_unique_files.insert(seed.clone());
                                    }
                                    for item in items {
                                        video_hash_unique_files.insert(item.file_name.clone());
                                    }
                                }
                                let video_hash_unique_count = video_hash_unique_files.len();

                                // Precompute SIFT members metadata
                                // source_path = original file (video/image), preview_path = video still or image
                                let mut displayed_sift_metadata = Vec::new();
                                let roots = get_db_roots();
                                for member in &displayed_sift_members {
                                    let source_path_opt = resolve_source_path(&roots, member).ok();
                                    let preview_path_opt = source_path_opt.as_ref().map(|p| self.get_thumbnail_path(p));
                                    let member_is_video = is_video_path(Path::new(member))
                                        || source_path_opt.as_ref().is_some_and(|p| is_video_path(p));
                                    let res_size_str = source_path_opt.as_ref()
                                        .map(|p| self.get_duplicate_media_info(p, member_is_video, ctx))
                                        .unwrap_or_else(|| "n/a".to_string());
                                    let sift_str = if sift_members.len() <= 1 && member == &filename {
                                        "SIFT: no grouped match".to_string()
                                    } else if member == &master_file_name {
                                        "SIFT: group root".to_string()
                                    } else {
                                        sift_info_line(&indices.sift_info_by_file, member)
                                    };
                                    let has_clip = clip_embedded_files.contains(member);
                                    let has_ocr = ocr_embedded_files.contains(member);
                                    let skipped = skipped_processing_files.contains(member);
                                    displayed_sift_metadata.push((member.clone(), source_path_opt, preview_path_opt, member_is_video, res_size_str, sift_str, has_clip, has_ocr, skipped));
                                }
                                let mut database_details: HashMap<(String, String), Vec<String>> = HashMap::new();
                                for (member, _, _, member_is_video, _, _, _, _, _) in &displayed_sift_metadata {
                                    if !self.expanded_duplicate_rows.contains(member) {
                                        continue;
                                    }
                                    database_details.insert(
                                        (master_file_name.clone(), member.clone()),
                                        duplicate_database_detail_lines(
                                            member,
                                            &master_file_name,
                                            *member_is_video,
                                            &indices.phash_by_file,
                                            &indices.video_frame_phashes_by_file,
                                        ),
                                    );
                                }
                                for similar_groups in [&phash_similar_groups, &video_hash_similar_groups] {
                                    for (group_seed, _, group_items) in similar_groups {
                                        for item in group_items {
                                            if !self.expanded_duplicate_rows.contains(&item.file_name) {
                                                continue;
                                            }
                                            database_details.insert(
                                                (group_seed.clone(), item.file_name.clone()),
                                                duplicate_database_detail_lines(
                                                    &item.file_name,
                                                    group_seed,
                                                    item.is_video,
                                                    &indices.phash_by_file,
                                                    &indices.video_frame_phashes_by_file,
                                                ),
                                            );
                                        }
                                    }
                                }

                                ui.heading("👥 Duplicate Matches");
                                ui.add_space(4.0);
                                ui.weak(format!("Current: {}", filename));
                                ui.add_space(2.0);
                                ui.weak(format!(
                                    "pHash similar count: {} across {} group(s)",
                                    phash_unique_count,
                                    phash_similar_groups.len()
                                ));
                                if current_is_video {
                                    ui.weak(format!(
                                        "VideoHash similar count: {} across {} group(s)",
                                        video_hash_unique_count,
                                        video_hash_similar_groups.len()
                                    ));
                                }
                                ui.add_space(8.0);

                                egui::ScrollArea::vertical().show(ui, |ui| {
                                    let side_thumb = 90.0_f32;

                                    // 1. SIFT Cluster Members (Duplicates)
                                    if !displayed_sift_metadata.is_empty() {
                                        ui.horizontal(|ui| {
                                            ui.colored_label(egui::Color32::from_rgb(100, 200, 100), "✓ SIFT Group");
                                            ui.weak(format!("({} files)", displayed_sift_metadata.len()));
                                        });
                                        ui.add_space(6.0);

                                        for (member, source_path_opt, preview_path_opt, member_is_video, res_size_str, sift_str, has_clip, has_ocr, skipped) in &displayed_sift_metadata {
                                            let detail_lines = database_details
                                                .get(&(master_file_name.clone(), member.clone()))
                                                .cloned()
                                                .unwrap_or_default();
                                            let expanded = self.expanded_duplicate_rows.contains(member);
                                            ui.horizontal(|ui| {
                                                // Left: Thumbnail preview (use preview_path for video stills)
                                                let thumb_path = preview_path_opt.as_ref().or(source_path_opt.as_ref());
                                                if let Some(t_path) = thumb_path {
                                                    self.draw_thumbnail_async(ui, t_path, side_thumb);
                                                } else {
                                                    let (rect, _) = ui.allocate_exact_size(
                                                        egui::vec2(side_thumb, side_thumb),
                                                        egui::Sense::hover(),
                                                    );
                                                    ui.painter().rect_filled(rect, 4.0, egui::Color32::from_gray(30));
                                                }

                                                // Right: Info and buttons
                                                ui.vertical(|ui| {
                                                    ui.horizontal(|ui| {
                                                        ui.colored_label(
                                                            if *member_is_video { egui::Color32::LIGHT_BLUE } else { egui::Color32::LIGHT_GREEN },
                                                            if *member_is_video { "📹 Video" } else { "🖼 Image" }
                                                        );
                                                        draw_embedding_markers(ui, *has_clip, *has_ocr, *skipped);
                                                        if member == &filename {
                                                            ui.colored_label(egui::Color32::from_rgb(255, 180, 50), "• Active");
                                                        }
                                                    });

                                                    ui.weak(res_size_str);
                                                    ui.weak(sift_str);

                                                    wrapping_monospace_path(ui, member);

                                                    ui.horizontal_wrapped(|ui| {
                                                        if let Some(s_path) = source_path_opt.as_ref() {
                                                            if ui.button("📂 Open Folder").clicked() {
                                                                open_in_dolphin_or_fallback(s_path);
                                                            }
                                                            if *member_is_video {
                                                                if ui.button("▶ Open in mpv").clicked() {
                                                                    let _ = std::process::Command::new("mpv")
                                                                        .arg(s_path)
                                                                        .spawn();
                                                                }
                                                            } else {
                                                                if ui.button("👁 View").clicked() {
                                                                    if let Some(pos) = self.images.iter().position(|p| p == s_path) {
                                                                        self.current_index = pos;
                                                                    } else {
                                                                        self.images.insert(self.current_index + 1, s_path.clone());
                                                                        self.db_filename_by_path.insert(s_path.clone(), member.clone());
                                                                        self.current_index += 1;
                                                                    }
                                                                    self.remember_gallery_image();
                                                                    self.show_grid = false;
                                                                    self.back_target_is_gallery = true;
                                                                    self.update_current_file_info();
                                                                    self.update_side_panel_metadata_if_needed();
                                                                }
                                                            }
                                                        }
                                                        if ui.button(if expanded { "Collapse" } else { "Expand" }).clicked() {
                                                            if expanded {
                                                                self.expanded_duplicate_rows.remove(member);
                                                            } else {
                                                                self.expanded_duplicate_rows.insert(member.clone());
                                                            }
                                                        }
                                                    });
                                                    if expanded {
                                                        for line in &detail_lines {
                                                            ui.monospace(line);
                                                        }
                                                    }
                                                });
                                            });
                                            ui.add_space(8.0);
                                            ui.separator();
                                            ui.add_space(8.0);
                                        }
                                        ui.add_space(8.0);
                                    }

                                    // 2. pHash and VideoHash similars grouped by SIFT master/member seed
                                    let mut rendered_similar_section = false;
                                    for (section_title, similar_groups, unique_count) in [
                                        ("pHash similars", &phash_similar_groups, phash_unique_count),
                                        ("VideoHash similars", &video_hash_similar_groups, video_hash_unique_count),
                                    ] {
                                        if similar_groups.is_empty() {
                                            continue;
                                        }
                                        rendered_similar_section = true;
                                        ui.horizontal(|ui| {
                                            ui.colored_label(
                                                egui::Color32::from_rgb(100, 180, 255),
                                                format!("🔗 Similar Files ({section_title})"),
                                            );
                                            ui.weak(format!("({unique_count} unique files)"));
                                        });
                                        ui.add_space(6.0);

                                        for (group_seed, section_label, group_items) in similar_groups {
                                            egui::Frame::NONE
                                                .fill(ui.visuals().extreme_bg_color.gamma_multiply(0.35))
                                                .stroke(egui::Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color.gamma_multiply(0.35)))
                                                .inner_margin(8.0)
                                                .corner_radius(6.0)
                                                .show(ui, |ui| {
                                                    ui.horizontal(|ui| {
                                                        ui.colored_label(egui::Color32::from_rgb(120, 190, 255), section_label);
                                                    });
                                                    let linked_count = group_items
                                                        .iter()
                                                        .filter(|item| item.file_name != *group_seed)
                                                        .count();
                                                    ui.weak(format!("{linked_count} similar file(s) linked to this reference"));
                                                    ui.add_space(6.0);

                                                    let row_height = side_thumb + 24.0;
                                                    for item in group_items.iter().cloned() {
                                                        let source_path_opt = resolve_source_path(&roots, &item.file_name).ok();
                                                        let preview_path_opt = source_path_opt.as_ref().map(|p| self.get_thumbnail_path(p));
                                                        let item_is_video = item.is_video
                                                            || is_video_path(Path::new(&item.file_name))
                                                            || source_path_opt.as_ref().is_some_and(|p| is_video_path(p));
                                                        let res_size_str = source_path_opt.as_ref()
                                                            .map(|p| self.get_duplicate_media_info(p, item_is_video, ctx))
                                                            .unwrap_or_else(|| "n/a".to_string());
                                                        let item_has_clip = clip_embedded_files.contains(&item.file_name);
                                                        let item_has_ocr = ocr_embedded_files.contains(&item.file_name);
                                                        let item_skipped = skipped_processing_files.contains(&item.file_name);
                                                        let detail_lines = database_details
                                                            .get(&(group_seed.clone(), item.file_name.clone()))
                                                            .cloned()
                                                            .unwrap_or_default();
                                                        let expanded = self.expanded_duplicate_rows.contains(&item.file_name);

                                                        ui.allocate_ui(egui::vec2(ui.available_width(), row_height), |ui| {
                                                            ui.horizontal(|ui| {
                                                                // Left: Thumbnail preview (use preview_path for video stills)
                                                                let thumb_path = preview_path_opt.as_ref().or(source_path_opt.as_ref());
                                                                if let Some(t_path) = thumb_path {
                                                                    self.draw_thumbnail_async(ui, t_path, side_thumb);
                                                                } else {
                                                                    let (rect, _) = ui.allocate_exact_size(
                                                                        egui::vec2(side_thumb, side_thumb),
                                                                        egui::Sense::hover(),
                                                                    );
                                                                    ui.painter().rect_filled(rect, 4.0, egui::Color32::from_gray(30));
                                                                }

                                                                // Right: Info and buttons
                                                                ui.vertical(|ui| {
                                                                    ui.horizontal(|ui| {
                                                                        ui.colored_label(
                                                                            if item_is_video { egui::Color32::LIGHT_BLUE } else { egui::Color32::LIGHT_GREEN },
                                                                            if item_is_video { "📹 Video" } else { "🖼 Image" }
                                                                        );
                                                                        draw_embedding_markers(ui, item_has_clip, item_has_ocr, item_skipped);
                                                                        if item.file_name == filename {
                                                                            ui.colored_label(egui::Color32::from_rgb(255, 180, 50), "• Active");
                                                                        }
                                                                        if item.file_name == *group_seed {
                                                                            ui.colored_label(egui::Color32::from_rgb(120, 190, 255), "• Seed");
                                                                        }
                                                                    });

                                                                    let similarity_label = item.similarity_pct
                                                                        .map(|v| {
                                                                            if use_sift_seed_similarity {
                                                                                format!("similarity to this SIFT master {:.2}%", v)
                                                                            } else {
                                                                                format!("similarity to active {:.2}%", v)
                                                                            }
                                                                        })
                                                                        .unwrap_or_else(|| {
                                                                            if use_sift_seed_similarity {
                                                                                "similarity to this SIFT master n/a".to_string()
                                                                            } else {
                                                                                "similarity to active n/a".to_string()
                                                                            }
                                                                        });
                                                                    ui.colored_label(egui::Color32::from_rgb(100, 180, 255), similarity_label);

                                                                    ui.weak(&res_size_str);

                                                                    wrapping_monospace_path(ui, &item.file_name);

                                                                    ui.horizontal_wrapped(|ui| {
                                                                        if let Some(s_path) = source_path_opt.as_ref() {
                                                                            if ui.button("📂 Open Folder").clicked() {
                                                                                open_in_dolphin_or_fallback(s_path);
                                                                            }
                                                                            if item_is_video {
                                                                                if ui.button("▶ Open in mpv").clicked() {
                                                                                    let _ = std::process::Command::new("mpv")
                                                                                        .arg(s_path)
                                                                                        .spawn();
                                                                                }
                                                                            } else {
                                                                                if ui.button("👁 View").clicked() {
                                                                                    if let Some(pos) = self.images.iter().position(|p| p == s_path) {
                                                                                        self.current_index = pos;
                                                                                    } else {
                                                                                        self.images.insert(self.current_index + 1, s_path.clone());
                                                                                        self.db_filename_by_path.insert(s_path.clone(), item.file_name.clone());
                                                                                        self.current_index += 1;
                                                                                    }
                                                                                    self.remember_gallery_image();
                                                                                    self.show_grid = false;
                                                                                    self.back_target_is_gallery = true;
                                                                                    self.update_current_file_info();
                                                                                    self.update_side_panel_metadata_if_needed();
                                                                                }
                                                                            }
                                                                        }
                                                                        if ui.button(if expanded { "Collapse" } else { "Expand" }).clicked() {
                                                                            if expanded {
                                                                                self.expanded_duplicate_rows.remove(&item.file_name);
                                                                            } else {
                                                                                self.expanded_duplicate_rows.insert(item.file_name.clone());
                                                                            }
                                                                        }
                                                                    });
                                                                    if expanded {
                                                                        for line in &detail_lines {
                                                                            ui.monospace(line);
                                                                        }
                                                                    }
                                                                });
                                                            });
                                                        });
                                                    }
                                                });
                                            ui.add_space(8.0);
                                        }
                                    }
                                    if !rendered_similar_section {
                                        ui.weak("No duplicates or similar files found in database.");
                                    }
                                });
                            } else {
                                ui.weak("Current file is not indexed in the database.");
                            }
                        } else {
                            ui.weak("No image loaded.");
                        }
                    }
                }
                });
        }

        let mut panel = egui::CentralPanel::default();
        if let Some(bg) = self.viewport_bg {
            panel = panel.frame(egui::Frame::NONE.fill(bg));
        }
        panel.show(ctx, |ui| {
            if self.image_editor.is_some() {
                self.show_image_editor(ui, ctx);
            } else if self.show_grid {
                self.show_grid_view(ui, ctx);
            } else {
                if let Some(path) = self.images.get(self.current_index).cloned() {
                    if self.viewer_rotation_path.as_ref() != Some(&path) {
                        self.viewer_rotation_path = Some(path.clone());
                        self.viewer_rotation_quarter_turns = 0;
                    }
                    let resolved_path = if self.is_comparison_mode() {
                        self.comparison_display_path(&path)
                    } else {
                        self.get_thumbnail_path(&path)
                    };
                    // Click and Drag to pan (allocated first to allow zoom-to-mouse math using rect)
                    let (rect, response) =
                        ui.allocate_at_least(ui.available_size(), egui::Sense::click_and_drag());
                    let mut view_changed = false;
                    let primary_button_down =
                        ui.input(|i| i.pointer.button_down(egui::PointerButton::Primary));
                    if response.dragged() && primary_button_down {
                        self.offset += response.drag_delta();
                        view_changed = true;
                    }

                    // Middle click to recentre and fit
                    if response.middle_clicked() {
                        self.zoom = 1.0;
                        self.offset = egui::Vec2::ZERO;
                        view_changed = true;
                    }

                    // Interaction / Zoom
                    let pointer_over_viewport = ui.input(|i| {
                        i.pointer
                            .hover_pos()
                            .is_some_and(|position| rect.contains(position))
                    });
                    if pointer_over_viewport && frame_scroll_delta != 0.0 {
                        let zoom_factor = (frame_scroll_delta / 200.0).exp();

                        if let Some(mouse_pos) = ui.input(|i| i.pointer.hover_pos()) {
                            let screen_center = rect.center();
                            let v_m = mouse_pos - screen_center;
                            self.offset = v_m - (v_m - self.offset) * zoom_factor;
                        }

                        self.zoom = (self.zoom * zoom_factor).clamp(0.05, 32.0);
                        view_changed = true;
                    }
                    if view_changed {
                        self.apply_comparison_view_state_to_all();
                    }

                    // Right click context menu to copy path, image, or recenter
                    response.context_menu(|ui| {
                        let db_name = self.resolve_db_filename(&path);
                        let is_video_item = db_name
                            .as_ref()
                            .map(|name| is_video_path(Path::new(name)))
                            .unwrap_or_else(|| is_video_path(&path));
                        if ui.button("📂 Show in parent folder").clicked() {
                            let actual_path = self.resolve_actual_path(&path);
                            open_in_dolphin_or_fallback(&actual_path);
                            ui.close();
                        }
                        if ui.button("📋 Copy Image Path").clicked() {
                            ui.ctx().copy_text(path.to_string_lossy().to_string());
                            ui.close();
                        }
                        if ui.button("🖼 Copy Image").clicked() {
                            let resolved_path = self.get_thumbnail_path(&path);
                            if let Err(err) = copy_image_file_to_clipboard(&resolved_path) {
                                self.semantic_status = format!("Copy image failed: {err}");
                            }
                            ui.close();
                        }
                        if !is_video_item && ui.button("✏ Edit image").clicked() {
                            self.start_image_editor(&path, ui.ctx());
                            ui.close();
                        }
                        if !is_video_item && ui.button("🔄 Rotate image").clicked() {
                            self.viewer_rotation_quarter_turns =
                                (self.viewer_rotation_quarter_turns + 1) % 4;
                            self.viewer_rotation_path = Some(path.clone());
                            self.zoom = 1.0;
                            self.offset = egui::Vec2::ZERO;
                            self.apply_comparison_view_state_to_all();
                            ui.ctx().request_repaint();
                            ui.close();
                        }
                        if ui.button("🔍 Fit Image / Recenter").clicked() {
                            self.zoom = 1.0;
                            self.offset = egui::Vec2::ZERO;
                            self.apply_comparison_view_state_to_all();
                            ui.close();
                        }
                        ui.separator();
                        if ui.button("Show most similar").clicked() {
                            let request = PendingSearchRequest::Similar {
                                db_file_name: db_name.clone(),
                                media_path: path.clone(),
                                is_video: is_video_item,
                                timestamp_sec: 0.0,
                            };
                            self.request_search_action(request, ui.ctx());
                            ui.close();
                        }
                        if ui.button("Show more of this person").clicked() {
                            let request = PendingSearchRequest::Person {
                                db_file_name: db_name,
                                media_path: path.clone(),
                                is_video: is_video_item,
                            };
                            self.request_search_action(request, ui.ctx());
                            ui.close();
                        }
                        ui.separator();
                        ui.menu_button("🎨 Viewport Background", |ui| {
                            if ui
                                .radio_value(&mut self.viewport_bg, None, "Default Theme")
                                .clicked()
                            {
                                ui.close();
                            }
                            if ui
                                .radio_value(
                                    &mut self.viewport_bg,
                                    Some(egui::Color32::BLACK),
                                    "Pure Black",
                                )
                                .clicked()
                            {
                                ui.close();
                            }
                            if ui
                                .radio_value(
                                    &mut self.viewport_bg,
                                    Some(egui::Color32::WHITE),
                                    "Pure White",
                                )
                                .clicked()
                            {
                                ui.close();
                            }
                            if ui
                                .radio_value(
                                    &mut self.viewport_bg,
                                    Some(egui::Color32::from_rgb(30, 30, 30)),
                                    "Dark Charcoal",
                                )
                                .clicked()
                            {
                                ui.close();
                            }
                            if ui
                                .radio_value(
                                    &mut self.viewport_bg,
                                    Some(egui::Color32::from_rgb(128, 128, 128)),
                                    "Slate Gray",
                                )
                                .clicked()
                            {
                                ui.close();
                            }
                        });
                    });

                    if let Some(compare_path) = self.compare_target.clone() {
                        let left_resolved = self.get_thumbnail_path(&path);
                        let right_resolved = self.get_thumbnail_path(&compare_path);
                        let left_texture = self.request_viewer_texture(&left_resolved, ctx);
                        let right_texture = self.request_viewer_texture(&right_resolved, ctx);
                        let left_failed = self.viewer_texture_failed.contains(&left_resolved);
                        let right_failed = self.viewer_texture_failed.contains(&right_resolved);

                        let builder = egui::UiBuilder::new()
                            .max_rect(rect)
                            .id_salt("sift_compare_viewport");
                        let mut compare_ui = ui.new_child(builder);
                        let avail_size = rect.size();
                        let half_w = (avail_size.x / 2.0 - 12.0).max(10.0);
                        let h = (avail_size.y - 120.0).max(10.0);

                        compare_ui.horizontal(|ui| {
                            if let Some(texture) = left_texture {
                                ui.add_sized(
                                    egui::vec2(half_w, h),
                                    egui::Image::from_texture(&texture).maintain_aspect_ratio(true),
                                );
                            } else {
                                let (placeholder, _) = ui.allocate_exact_size(
                                    egui::vec2(half_w, h),
                                    egui::Sense::hover(),
                                );
                                ui.painter().text(
                                    placeholder.center(),
                                    egui::Align2::CENTER_CENTER,
                                    if left_failed {
                                        "Unable to load image"
                                    } else {
                                        "Loading image..."
                                    },
                                    egui::FontId::proportional(14.0),
                                    egui::Color32::GRAY,
                                );
                            }

                            ui.add_space(12.0);

                            if let Some(texture) = right_texture {
                                ui.add_sized(
                                    egui::vec2(half_w, h),
                                    egui::Image::from_texture(&texture).maintain_aspect_ratio(true),
                                );
                            } else {
                                let (placeholder, _) = ui.allocate_exact_size(
                                    egui::vec2(half_w, h),
                                    egui::Sense::hover(),
                                );
                                ui.painter().text(
                                    placeholder.center(),
                                    egui::Align2::CENTER_CENTER,
                                    if right_failed {
                                        "Unable to load image"
                                    } else {
                                        "Loading image..."
                                    },
                                    egui::FontId::proportional(14.0),
                                    egui::Color32::GRAY,
                                );
                            }
                        });

                        // Draw SIFT alignment info overlay at the bottom of the central viewport
                        let summary_text = if self.sift_running {
                            "⌛ Calculating SIFT correspondence alignment...".to_string()
                        } else if let Some(summary) = &self.sift_pair_overlay {
                            summary.clone()
                        } else {
                            "SIFT alignment not calculated.".to_string()
                        };

                        compare_ui.add_space(16.0);
                        compare_ui.vertical_centered(|ui| {
                            egui::Frame::NONE
                                .fill(egui::Color32::from_black_alpha(190))
                                .stroke(egui::Stroke::new(
                                    1.0,
                                    egui::Color32::from_rgb(100, 180, 255).gamma_multiply(0.4),
                                ))
                                .inner_margin(12.0)
                                .corner_radius(8.0)
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.colored_label(
                                            egui::Color32::from_rgb(100, 180, 255),
                                            "👥 SIFT Matcher Status",
                                        );
                                        ui.separator();
                                        ui.add(egui::Label::new(
                                            egui::RichText::new(summary_text)
                                                .monospace()
                                                .size(12.0)
                                                .color(egui::Color32::WHITE),
                                        ));

                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                if ui.button("❌ Close comparison").clicked() {
                                                    self.compare_target = None;
                                                    self.sift_pair_overlay = None;
                                                }
                                            },
                                        );
                                    });
                                });
                        });
                    } else {
                        // Calculate image rect
                        let base_size = rect.size();
                        let draw_size = base_size * self.zoom;
                        let draw_pos = rect.center() + self.offset - draw_size / 2.0;
                        let draw_rect = egui::Rect::from_min_size(draw_pos, draw_size);

                        let viewer_texture = self.request_viewer_texture(&resolved_path, ctx);
                        if let Some(texture) = viewer_texture {
                            let texture_size = texture.size_vec2();
                            let fit_scale = (draw_rect.width() / texture_size.x)
                                .min(draw_rect.height() / texture_size.y);
                            let image_rect = egui::Rect::from_center_size(
                                draw_rect.center(),
                                texture_size * fit_scale,
                            );
                            let mut mesh = egui::Mesh::with_texture(texture.id());
                            mesh.add_rect_with_uv(
                                image_rect,
                                egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                                egui::Color32::WHITE,
                            );
                            let angle = self.viewer_rotation_quarter_turns as f32
                                * std::f32::consts::FRAC_PI_2;
                            if angle != 0.0 {
                                let (sin, cos) = angle.sin_cos();
                                for vertex in &mut mesh.vertices {
                                    let relative = vertex.pos - image_rect.center();
                                    vertex.pos = image_rect.center()
                                        + egui::vec2(
                                            relative.x * cos - relative.y * sin,
                                            relative.x * sin + relative.y * cos,
                                        );
                                }
                            }
                            ui.painter().add(egui::Shape::mesh(mesh));
                            if let Some(boxes) = self.face_overlay_boxes.get(&path) {
                                let angle = self.viewer_rotation_quarter_turns as f32
                                    * std::f32::consts::FRAC_PI_2;
                                let (sin, cos) = angle.sin_cos();
                                for bbox in boxes {
                                    let points = [
                                        egui::pos2(bbox[0], bbox[1]),
                                        egui::pos2(bbox[2], bbox[1]),
                                        egui::pos2(bbox[2], bbox[3]),
                                        egui::pos2(bbox[0], bbox[3]),
                                    ]
                                    .map(|point| {
                                        let unrotated = egui::pos2(
                                            image_rect.min.x + point.x * image_rect.width(),
                                            image_rect.min.y + point.y * image_rect.height(),
                                        );
                                        let relative = unrotated - image_rect.center();
                                        image_rect.center()
                                            + egui::vec2(
                                                relative.x * cos - relative.y * sin,
                                                relative.x * sin + relative.y * cos,
                                            )
                                    });
                                    let stroke = egui::Stroke::new(3.0, egui::Color32::RED);
                                    for edge in 0..4 {
                                        ui.painter().line_segment(
                                            [points[edge], points[(edge + 1) % 4]],
                                            stroke,
                                        );
                                    }
                                }
                            }
                        } else {
                            ui.painter().text(
                                draw_rect.center(),
                                egui::Align2::CENTER_CENTER,
                                if self.viewer_texture_failed.contains(&resolved_path) {
                                    "Unable to load image"
                                } else {
                                    "Loading image..."
                                },
                                egui::FontId::proportional(14.0),
                                egui::Color32::GRAY,
                            );
                        }
                    }
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.label("No image loaded");
                    });
                }
            }
        });

        if self.flat_loading || self.grid_loading || self.sift_align_all_running {
            ctx.request_repaint();
        }
    }
}
