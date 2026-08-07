use super::*;

impl ImageViewer {
    pub(crate) fn get_subdirectories(&self, path: &Path) -> Vec<PathBuf> {
        let mut dirs = Vec::new();
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.filter_map(|e| e.ok()) {
                let p = entry.path();
                if p.is_dir() {
                    if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                        if !name.starts_with('.') {
                            dirs.push(p);
                        }
                    }
                }
            }
        }
        dirs.sort_by(|a, b| {
            let a_name = a
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_lowercase();
            let b_name = b
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_lowercase();
            a_name.cmp(&b_name)
        });
        dirs
    }

    pub(crate) fn show_home_page_view(&mut self, ctx: &egui::Context) {
        let current_dir_opt = self.home_current_dir.clone();

        egui::CentralPanel::default().show(ctx, |ui| {
            let mut opened_folder = false;
            let mut opened_file = false;
            ui.add_space(8.0);

            // Dolphin-like location toolbar at the top
            ui.horizontal(|ui| {
                // 1. Up Button
                let has_parent = current_dir_opt.is_some();
                let up_btn = ui.add_enabled_ui(has_parent, |ui| {
                    ui.add(egui::Button::new("⬆ Up").min_size(egui::vec2(50.0, 26.0)))
                });

                if has_parent && up_btn.inner.clicked() {
                    if let Some(ref current_dir) = current_dir_opt {
                        self.home_current_dir = current_dir.parent().map(|p| p.to_path_buf());
                        self.home_selected_dir = None;
                    }
                }

                ui.add_space(4.0);

                // 2. Path Display Location Bar (matches standard Dolphin address bar)
                let path_str = match &current_dir_opt {
                    Some(p) => p.to_string_lossy().to_string(),
                    None => "/".to_string(), // Root disks
                };

                egui::Frame::NONE
                    .fill(ui.visuals().extreme_bg_color)
                    .stroke(egui::Stroke::new(
                        1.0,
                        ui.visuals().widgets.noninteractive.bg_stroke.color,
                    ))
                    .inner_margin(egui::vec2(8.0, 4.0))
                    .corner_radius(4.0)
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width() - 170.0);
                        ui.label(egui::RichText::new(&path_str).monospace().size(13.0));
                    });

                // 3. Open button on the right
                let target_dir = self.home_selected_dir.clone().or(current_dir_opt.clone());
                let has_target = target_dir.is_some();

                let open_btn = ui.add_enabled_ui(has_target, |ui| {
                    ui.add(
                        egui::Button::new(
                            egui::RichText::new("Open Folder")
                                .strong()
                                .color(egui::Color32::WHITE),
                        )
                        .fill(ui.visuals().selection.bg_fill)
                        .min_size(egui::vec2(100.0, 26.0)),
                    )
                });

                if has_target && open_btn.inner.clicked() {
                    if let Some(dir) = target_dir {
                        self.open_folder_path(dir.clone());
                        self.show_home_page = false;
                        self.show_grid = true;
                        self.start_recursive_scan();
                        opened_folder = true;
                    }
                }

                if ui.button("Open File").clicked() {
                    self.open_file_dialog(ctx);
                    opened_file = true;
                }
                if ui.button("Compare Paths").clicked() {
                    self.open_comparison_path_dialog();
                    opened_file = true;
                }
            });

            if opened_folder || opened_file {
                ctx.request_repaint();
                return;
            }

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(8.0);

            // Pane container frame (high density desktop file pane)
            egui::Frame::NONE
                .fill(ui.visuals().extreme_bg_color)
                .stroke(egui::Stroke::new(
                    1.0,
                    ui.visuals().widgets.noninteractive.bg_stroke.color,
                ))
                .inner_margin(2.0)
                .corner_radius(4.0)
                .show(ui, |ui| {
                    ui.set_min_height(ui.available_height() - 6.0);

                    egui::ScrollArea::vertical().show(ui, |ui| {
                        let mut items = Vec::new();
                        let is_disk_level = current_dir_opt.is_none();

                        let db_roots = get_db_roots();

                        if is_disk_level {
                            for disk in get_system_disks() {
                                items.push((disk, true));
                            }
                        } else {
                            if let Some(ref current_dir) = current_dir_opt {
                                for sub in self.get_subdirectories(current_dir) {
                                    items.push((sub, false));
                                }
                            }
                        }

                        if items.is_empty() {
                            ui.vertical_centered(|ui| {
                                ui.add_space(40.0);
                                ui.weak("This folder contains no subfolders.");
                                ui.add_space(40.0);
                            });
                        } else {
                            ui.vertical(|ui| {
                                ui.spacing_mut().item_spacing = egui::vec2(0.0, 1.0); // Dense desktop list spacing

                                for (item_path, is_disk) in items {
                                    let name = if is_disk {
                                        if item_path == PathBuf::from("/") {
                                            "System Root (/)".to_string()
                                        } else if item_path.to_string_lossy().contains("/home/") {
                                            format!(
                                                "Home ({})",
                                                item_path
                                                    .file_name()
                                                    .and_then(|n| n.to_str())
                                                    .unwrap_or("User")
                                            )
                                        } else {
                                            item_path
                                                .file_name()
                                                .and_then(|n| n.to_str())
                                                .map(|s| s.to_string())
                                                .unwrap_or_else(|| {
                                                    item_path.to_string_lossy().to_string()
                                                })
                                        }
                                    } else {
                                        item_path
                                            .file_name()
                                            .and_then(|n| n.to_str())
                                            .unwrap_or("")
                                            .to_string()
                                    };

                                    let is_ai = is_path_ai_backed_with_roots(&item_path, &db_roots);
                                    let is_selected =
                                        self.home_selected_dir.as_ref() == Some(&item_path);

                                    // Allocate dense row rect
                                    let (rect, response) = ui.allocate_exact_size(
                                        egui::vec2(ui.available_width(), 26.0),
                                        egui::Sense::click(),
                                    );

                                    if response.double_clicked() {
                                        self.home_current_dir = Some(item_path.clone());
                                        self.home_selected_dir = None;
                                    } else if response.clicked() {
                                        self.home_selected_dir = Some(item_path.clone());
                                    }

                                    // Draw row background selection/hover highlight
                                    let row_bg = if is_selected {
                                        ui.visuals().selection.bg_fill
                                    } else if response.hovered() {
                                        ui.visuals().widgets.hovered.bg_fill.gamma_multiply(0.2)
                                    } else {
                                        egui::Color32::TRANSPARENT
                                    };

                                    if row_bg != egui::Color32::TRANSPARENT {
                                        ui.painter().rect_filled(rect, 2.0, row_bg);
                                    }

                                    // Render columns inside the row
                                    ui.allocate_ui_at_rect(
                                        rect.shrink2(egui::vec2(8.0, 2.0)),
                                        |ui| {
                                            ui.horizontal(|ui| {
                                                let icon = if is_disk { "💾" } else { "📁" };

                                                let text_color = if is_selected {
                                                    egui::Color32::WHITE
                                                } else {
                                                    ui.visuals().widgets.noninteractive.text_color()
                                                };

                                                ui.label(egui::RichText::new(icon).size(14.0));
                                                ui.add_space(4.0);
                                                ui.label(
                                                    egui::RichText::new(&name).color(text_color),
                                                );

                                                // Push column 2 to the right
                                                ui.with_layout(
                                                    egui::Layout::right_to_left(
                                                        egui::Align::Center,
                                                    ),
                                                    |ui| {
                                                        let type_str = if is_disk {
                                                            if item_path == PathBuf::from("/") {
                                                                "System Disk"
                                                            } else if item_path
                                                                .to_string_lossy()
                                                                .contains("/home/")
                                                            {
                                                                "User Directory"
                                                            } else {
                                                                "Disk Partition"
                                                            }
                                                        } else if is_ai {
                                                            "Indexed Folder"
                                                        } else {
                                                            "Folder"
                                                        };

                                                        let type_color = if is_selected {
                                                            egui::Color32::WHITE
                                                        } else if is_ai {
                                                            egui::Color32::from_rgb(140, 160, 255)
                                                        } else {
                                                            ui.visuals().weak_text_color()
                                                        };

                                                        ui.label(
                                                            egui::RichText::new(type_str)
                                                                .color(type_color)
                                                                .size(11.0),
                                                        );
                                                    },
                                                );
                                            });
                                        },
                                    );
                                }
                            });
                        }
                    });
                });
        });
    }
}
