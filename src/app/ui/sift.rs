use super::*;

impl ImageViewer {
    pub(crate) fn selected_sift_file_names(&self) -> Vec<String> {
        self.selected_grid_items
            .iter()
            .filter(|item| !item.is_video)
            .filter_map(|item| item.db_filename.clone())
            .collect()
    }

    pub(crate) fn start_sift_alignment(
        &mut self,
        path_a: PathBuf,
        path_b: PathBuf,
        ctx: egui::Context,
    ) {
        if self.sift_running {
            return;
        }
        self.sift_running = true;
        self.sift_pair_overlay = None;
        let (tx, rx) = std::sync::mpsc::channel();
        self.sift_rx = Some(rx);
        let diagnostics = self.diagnostics.clone();

        std::thread::spawn(move || {
            let task = diagnostics.task_guard("sift_pair_compare");
            let result = compute_sift_summary(&path_a, &path_b).map_err(|e| e.to_string());
            let _ = tx.send(result);
            ctx.request_repaint();
            task.complete();
        });
    }

    pub(crate) fn start_sift_align_all(&mut self, ctx: &egui::Context) {
        if !self.is_comparison_mode() || self.sift_align_all_running {
            return;
        }
        if self.images.len() < 2 {
            self.comparison_alignment_status =
                "SIFT alignment needs at least two comparison images.".to_string();
            return;
        }
        if self.images.iter().any(|path| is_video_path(path)) {
            self.comparison_alignment_status =
                "SIFT alignment currently supports still images only.".to_string();
            return;
        }

        self.clear_comparison_alignment();
        let reference = self.images[0].clone();
        let candidates: Vec<PathBuf> = self.images.iter().skip(1).cloned().collect();

        let output_dir = self
            .comparison_alignment_temp_dir
            .clone()
            .unwrap_or_else(|| {
                let stamp = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis();
                std::env::temp_dir().join(format!(
                    "iris_sift_align_{}_{}",
                    std::process::id(),
                    stamp
                ))
            });
        if let Err(err) = std::fs::create_dir_all(&output_dir) {
            self.comparison_alignment_status =
                format!("Could not create temporary SIFT alignment directory: {err}");
            return;
        }

        self.comparison_alignment_temp_dir = Some(output_dir.clone());
        self.comparison_alignment_status = format!(
            "Aligning {} comparison images to the first image with SIFT...",
            candidates.len()
        );
        self.sift_align_all_running = true;
        let (tx, rx) = std::sync::mpsc::channel();
        self.sift_align_all_rx = Some(rx);
        let ctx = ctx.clone();
        let diagnostics = self.diagnostics.clone();
        std::thread::spawn(move || {
            let task = diagnostics.task_guard("sift_align_all");
            let result = run_sift_alignment_batch(&reference, &candidates, &output_dir)
                .map_err(|err| err.to_string());
            let _ = tx.send(result);
            ctx.request_repaint();
            task.complete();
        });
    }

    pub(crate) fn poll_sift_align_all(&mut self) {
        let Some(rx) = self.sift_align_all_rx.take() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(result)) => {
                self.sift_align_all_running = false;
                let current_reference = self
                    .comparison_paths
                    .as_ref()
                    .and_then(|paths| paths.first())
                    .cloned();
                if current_reference.as_ref() == Some(&result.reference) {
                    self.comparison_aligned_paths.extend(result.aligned_paths);
                    self.comparison_alignment_status = result.summary;
                    self.semantic_status = result.details.join(" | ");
                } else {
                    let _ = std::fs::remove_dir_all(&result.output_dir);
                }
            }
            Ok(Err(err)) => {
                self.sift_align_all_running = false;
                if let Some(output_dir) = self.comparison_alignment_temp_dir.take() {
                    let _ = std::fs::remove_dir_all(output_dir);
                }
                self.comparison_alignment_status = format!("SIFT alignment failed: {err}");
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                self.sift_align_all_rx = Some(rx);
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.sift_align_all_running = false;
                self.comparison_alignment_status =
                    "SIFT alignment worker disconnected unexpectedly.".to_string();
            }
        }
    }

    pub(crate) fn poll_sift_alignment(&mut self) {
        let Some(rx) = self.sift_rx.take() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(summary)) => {
                self.sift_running = false;
                self.sift_pair_overlay = Some(summary);
            }
            Ok(Err(err)) => {
                self.sift_running = false;
                self.sift_pair_overlay = Some(format!("❌ SIFT Alignment failed: {err}"));
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                self.sift_rx = Some(rx);
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.sift_running = false;
                self.sift_pair_overlay =
                    Some("❌ SIFT calculation worker disconnected.".to_string());
            }
        }
    }

    pub(crate) fn start_selected_sift_repair(&mut self, ctx: &egui::Context) {
        if self.sift_repair_running {
            return;
        }
        let selected_files = self.selected_sift_file_names();
        if selected_files.len() < 2 {
            self.semantic_status =
                "Select at least two indexed images before running SIFT repair.".to_string();
            return;
        }
        if !self.db_loaded || !self.db_supplemental_loaded {
            if !self.db_loading && !self.db_failed {
                self.start_lazy_db_load(ctx);
            }
            self.semantic_status =
                "Loading duplicate and SIFT indexes before SIFT repair...".to_string();
            return;
        }

        let file_names = self.expanded_sift_repair_selection();
        let selected_count = selected_files.len();
        let repair_count = file_names.len();
        let (tx, rx) = std::sync::mpsc::channel();
        self.sift_repair_rx = Some(rx);
        self.sift_repair_running = true;
        self.semantic_status = format!(
            "Running SIFT repair on {selected_count} selected images ({repair_count} including current SIFT groups)..."
        );
        let ctx = ctx.clone();
        let diagnostics = self.diagnostics.clone();
        std::thread::spawn(move || {
            let task = diagnostics.task_guard("sift_repair");
            let result = run_sift_repair_for_files(&file_names).map_err(|err| err.to_string());
            let _ = tx.send(result);
            ctx.request_repaint();
            task.complete();
        });
    }

    pub(crate) fn expanded_sift_repair_selection(&self) -> Vec<String> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        let Some(indices) = &self.db_indices else {
            return self.selected_sift_file_names();
        };

        for file_name in self.selected_sift_file_names() {
            if seen.insert(file_name.clone()) {
                out.push(file_name.clone());
            }
            let root = indices
                .sift_root_by_file
                .get(&file_name)
                .cloned()
                .unwrap_or_else(|| file_name.clone());
            if let Some(members) = indices.sift_members_by_root.get(root.as_str()) {
                for member in members {
                    if seen.insert(member.clone()) {
                        out.push(member.clone());
                    }
                }
            }
        }

        out
    }

    pub(crate) fn start_selected_sift_compare(&mut self, ctx: &egui::Context) {
        let selected_files = self.selected_sift_file_names();
        if selected_files.len() < 2 {
            self.semantic_status =
                "Select two indexed images before running SIFT compare.".to_string();
            return;
        }

        let file_a = selected_files[0].clone();
        let file_b = selected_files[1].clone();
        let roots = get_db_roots();
        let path_a = match resolve_source_path(&roots, &file_a) {
            Ok(path) => path,
            Err(err) => {
                self.semantic_status = format!("SIFT compare failed to resolve first image: {err}");
                return;
            }
        };
        let path_b = match resolve_source_path(&roots, &file_b) {
            Ok(path) => path,
            Err(err) => {
                self.semantic_status =
                    format!("SIFT compare failed to resolve second image: {err}");
                return;
            }
        };

        self.images = vec![path_a.clone(), path_b.clone()];
        self.current_index = 0;
        self.compare_target = Some(path_b.clone());
        self.show_grid = false;
        self.back_target_is_gallery = true;
        self.zoom = 1.0;
        self.offset = egui::Vec2::ZERO;
        self.start_sift_alignment(path_a, path_b, ctx.clone());
    }

    pub(crate) fn poll_sift_repair(&mut self, ctx: &egui::Context) {
        let Some(rx) = self.sift_repair_rx.take() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(result)) => {
                self.sift_repair_running = false;
                self.selected_grid_items.clear();
                self.compare_target = None;
                self.sift_pair_overlay = None;
                self.db_loaded = false;
                self.db_loading = false;
                self.db_supplemental_loaded = false;
                self.db_supplemental_loading = false;
                self.db_failed = false;
                self.db_indices = None;
                self.db_rx = None;
                self.start_lazy_db_load(ctx);
                self.semantic_status = format!(
                    "{} Reloading database maps after repairing {} selected files.",
                    result.summary, result.files
                );
            }
            Ok(Err(err)) => {
                self.sift_repair_running = false;
                self.semantic_status = format!("SIFT repair failed: {err}");
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                self.sift_repair_rx = Some(rx);
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.sift_repair_running = false;
                self.semantic_status = "SIFT repair worker disconnected unexpectedly.".to_string();
            }
        }
    }
}
