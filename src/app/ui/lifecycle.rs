use super::*;

impl ImageViewer {
    pub(crate) fn new(
        path: PathBuf,
        rx: Receiver<OpenRequest>,
        ctx_shared: Arc<Mutex<Option<egui::Context>>>,
        start_on_home_page: bool,
        comparison_paths: Option<Vec<PathBuf>>,
    ) -> Self {
        let path = path.canonicalize().unwrap_or(path);
        let open_target = path.clone();
        let comparison_paths = comparison_paths.map(|paths| {
            paths
                .into_iter()
                .map(|path| path.canonicalize().unwrap_or(path))
                .take(6)
                .collect::<Vec<_>>()
        });
        let comparison_mode = comparison_paths
            .as_ref()
            .is_some_and(|paths| paths.len() >= 2);
        let initial_images = if comparison_mode {
            comparison_paths.clone().unwrap_or_default()
        } else {
            Vec::new()
        };
        let open_target_is_dir = if comparison_mode {
            false
        } else {
            start_on_home_page || path.is_dir()
        };
        let pending_initial_window_size =
            (!start_on_home_page && path.is_file() && !is_video_path(&path)).then(|| {
                let [width, height] = initial_window_size(&path, false);
                egui::vec2(width, height)
            });

        let mut images = initial_images;
        let flat_loading = !start_on_home_page && !comparison_mode;
        let flat_images_shared = Arc::new(Mutex::new(None));

        if !start_on_home_page && !comparison_mode {
            if path.is_dir() {
                let shared = flat_images_shared.clone();
                let parent_absolute = path.clone();
                std::thread::spawn(move || {
                    let mut collected = Vec::new();
                    if let Ok(entries) = std::fs::read_dir(&parent_absolute) {
                        for entry in entries.filter_map(|e| e.ok()) {
                            let p = entry.path();
                            if p.is_file() {
                                let ext = p
                                    .extension()
                                    .and_then(|s| s.to_str())
                                    .unwrap_or("")
                                    .to_lowercase();
                                if matches!(
                                    ext.as_str(),
                                    "jpg"
                                        | "jpeg"
                                        | "png"
                                        | "bmp"
                                        | "gif"
                                        | "webp"
                                        | "tiff"
                                        | "avif"
                                        | "heif"
                                        | "heic"
                                        | "ico"
                                        | "icns"
                                        | "svg"
                                        | "mp4"
                                        | "mov"
                                        | "avi"
                                        | "mkv"
                                        | "webm"
                                        | "m4v"
                                        | "wmv"
                                        | "mpg"
                                        | "mpeg"
                                ) {
                                    collected.push(p);
                                }
                            }
                        }
                    }
                    collected.sort();
                    if let Ok(mut lock) = shared.lock() {
                        *lock = Some(collected);
                    }
                });
            } else {
                images.push(path.clone());
                let shared = flat_images_shared.clone();
                let parent = path.parent().unwrap_or(Path::new(".")).to_path_buf();
                let parent_absolute = parent.canonicalize().unwrap_or(parent);
                std::thread::spawn(move || {
                    let mut collected = Vec::new();
                    if let Ok(entries) = std::fs::read_dir(&parent_absolute) {
                        for entry in entries.filter_map(|e| e.ok()) {
                            let p = entry.path();
                            if p.is_file() {
                                let ext = p
                                    .extension()
                                    .and_then(|s| s.to_str())
                                    .unwrap_or("")
                                    .to_lowercase();
                                if matches!(
                                    ext.as_str(),
                                    "jpg"
                                        | "jpeg"
                                        | "png"
                                        | "bmp"
                                        | "gif"
                                        | "webp"
                                        | "tiff"
                                        | "avif"
                                        | "heif"
                                        | "heic"
                                        | "ico"
                                        | "icns"
                                        | "svg"
                                        | "mp4"
                                        | "mov"
                                        | "avi"
                                        | "mkv"
                                        | "webm"
                                        | "m4v"
                                        | "wmv"
                                        | "mpg"
                                        | "mpeg"
                                ) {
                                    collected.push(p);
                                }
                            }
                        }
                    }
                    collected.sort();
                    if let Ok(mut lock) = shared.lock() {
                        *lock = Some(collected);
                    }
                });
            }
        }

        let (thumbnail_tx, thumbnail_rx) =
            std::sync::mpsc::channel::<(PathBuf, egui::ColorImage)>();
        let (viewer_texture_tx, viewer_texture_rx) =
            std::sync::mpsc::channel::<(PathBuf, u64, Result<egui::ColorImage, String>)>();
        let (video_duration_tx, video_duration_rx) =
            std::sync::mpsc::channel::<(PathBuf, Option<VideoMetadata>)>();

        let mut viewer = Self {
            images,
            current_index: 0,
            comparison_paths,
            comparison_view_states: HashMap::new(),
            comparison_sync_view: false,
            comparison_aligned_paths: HashMap::new(),
            comparison_alignment_temp_dir: None,
            comparison_alignment_status: String::new(),
            comparison_path_dialog_open: false,
            comparison_path_input: String::new(),
            zoom: 1.0,
            offset: egui::Vec2::ZERO,
            exif_data: String::new(),
            side_panel_metadata_path: None,
            side_panel_layout_path: None,
            show_exif: false,
            side_panel_window_expanded: false,
            side_panel_open_pending: false,
            side_panel_expand_target_width: None,
            side_panel_open_pending_frames: 0,
            chunks: Vec::new(),
            viewport_bg: None,
            pending_initial_window_size,
            rx,
            show_grid: false,
            recursive_images: Vec::new(),
            recursive_video_indices: Vec::new(),
            gallery_thumbnail_scale: 1.0,
            grid_loading: false,
            recursive_rx: None,
            back_target_is_gallery: false,
            side_panel_mode: SidePanelMode::Layout,
            exif_search: String::new(),
            open_target,
            open_target_is_dir,
            flat_loading,
            flat_refresh_in_flight: false,
            flat_last_refresh_check: Instant::now(),
            flat_directory_mtime: None,
            flat_images_shared,
            current_dimensions: String::new(),
            current_file_size: String::new(),
            ctx_shared,
            thumbnail_textures: std::collections::HashMap::new(),
            thumbnail_loading: std::collections::HashSet::new(),
            thumbnail_failed: std::collections::HashSet::new(),
            thumbnail_rx,
            thumbnail_tx,
            thumbnail_active_threads: 0,
            viewer_textures: HashMap::new(),
            viewer_texture_loading: HashSet::new(),
            viewer_texture_failed: HashSet::new(),
            viewer_texture_revisions: HashMap::new(),
            viewer_texture_rx,
            viewer_texture_tx,
            video_duration_cache: std::cell::RefCell::new(HashMap::new()),
            video_duration_loading: std::cell::RefCell::new(HashSet::new()),
            video_duration_rx,
            video_duration_tx,

            // AI Explorer defaults
            db_loaded: false,
            db_loading: false,
            db_supplemental_loaded: false,
            db_supplemental_loading: false,
            db_failed: false,
            db_rx: None,
            db_indices: None,
            semantic_query: String::new(),
            search_history: Vec::new(),
            search_forward_history: Vec::new(),
            gallery_image_forward: None,
            applied_filename_query: String::new(),
            filename_search_results: None,
            semantic_folder: String::new(),
            semantic_limit: 300,
            semantic_video_only: false,
            semantic_mode: SearchMode::Filename,
            semantic_results: Vec::new(),
            semantic_results_mode: None,
            semantic_status: "Ready. Enter a phrase and press Search.".to_string(),
            pending_search_request: None,
            pending_semantic_search_mode: None,
            on_demand_embed_rx: None,

            // SIFT defaults
            compare_target: None,
            sift_pair_overlay: None,
            expanded_duplicate_rows: HashSet::new(),
            sift_running: false,
            sift_rx: None,
            sift_align_all_running: false,
            sift_align_all_rx: None,
            selected_grid_items: Vec::new(),
            sift_repair_running: false,
            sift_repair_rx: None,
            face_compare_running: false,
            face_compare_rx: None,
            face_overlay_boxes: HashMap::new(),
            image_editor: None,

            db_filename_by_path: HashMap::new(),
            video_still_cache: std::cell::RefCell::new(HashMap::new()),
            resolution_size_cache: std::cell::RefCell::new(HashMap::new()),
            db_filename_cache: std::cell::RefCell::new(HashMap::new()),
            show_home_page: start_on_home_page,
            home_current_dir: None,
            home_selected_dir: None,
            viewer_rotation_quarter_turns: 0,
            viewer_rotation_path: None,
        };

        if viewer.is_comparison_mode() {
            for path in &viewer.images {
                viewer.comparison_view_states.insert(
                    path.clone(),
                    ImageViewState {
                        zoom: 1.0,
                        offset: egui::Vec2::ZERO,
                    },
                );
            }
        }

        if !start_on_home_page {
            viewer.update_current_file_info();
        }
        viewer
    }

    pub(crate) fn start_lazy_db_load(&mut self, ctx: &egui::Context) {
        if self.db_loaded || self.db_loading || self.db_failed {
            return;
        }
        self.db_loading = true;
        self.db_supplemental_loaded = false;
        self.db_supplemental_loading = false;
        self.semantic_status = "Loading CLIP index and text encoder...".to_string();
        let (tx, rx) = std::sync::mpsc::channel();
        self.db_rx = Some(rx);
        let ctx_clone = ctx.clone();

        std::thread::spawn(move || {
            let rt = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    let _ = tx.send(DatabaseLoadMessage::ClipReady(Err(format!(
                        "Failed to create tokio runtime: {e}"
                    ))));
                    ctx_clone.request_repaint();
                    return;
                }
            };

            let clip_result: Result<(ClipIndex, ClipTextEncoder), anyhow::Error> =
                rt.block_on(async {
                    let db_dir_buf = get_db_dir();
                    let db_dir = db_dir_buf.as_path();
                    let table_name = MEDIA_INDEX_TABLE;
                    let media_indexer_dir = resolve_media_indexer_dir();
                    let onnx_path_buf = media_indexer_dir.join("models/clip-text/clip_text.onnx");
                    let tokenizer_path_buf =
                        media_indexer_dir.join("models/clip-text/tokenizer.json");
                    let onnx_path = onnx_path_buf.as_path();
                    let tokenizer_path = tokenizer_path_buf.as_path();

                    let db_fut = load_clip_database_index(db_dir, table_name);
                    let encoder_fut = async { ClipTextEncoder::new(onnx_path, tokenizer_path, 64) };
                    tokio::try_join!(db_fut, encoder_fut)
                });

            let clip_loaded = clip_result.is_ok();
            let _ = tx.send(DatabaseLoadMessage::ClipReady(
                clip_result.map_err(|e| e.to_string()),
            ));
            ctx_clone.request_repaint();
            if !clip_loaded {
                return;
            }

            let supplemental_result = rt
                .block_on(load_supplemental_database_indices(
                    get_db_dir().as_path(),
                    MEDIA_INDEX_TABLE,
                ))
                .map_err(|e| e.to_string());
            let _ = tx.send(DatabaseLoadMessage::SupplementalReady(supplemental_result));
            ctx_clone.request_repaint();
        });
    }

    pub(crate) fn poll_db_load(&mut self) {
        let Some(rx) = self.db_rx.take() else {
            return;
        };
        match rx.try_recv() {
            Ok(DatabaseLoadMessage::ClipReady(Ok((clip_index, encoder)))) => {
                let clip_embedded_files: HashSet<String> = clip_index
                    .entries
                    .iter()
                    .map(|entry| entry.file_name.to_string())
                    .collect();
                let mut basename_to_db_filename = HashMap::new();
                for entry in &clip_index.entries {
                    if let Some(fname) = Path::new(entry.file_name.as_ref()).file_name() {
                        basename_to_db_filename
                            .entry(fname.to_string_lossy().to_lowercase())
                            .or_insert_with(|| entry.file_name.to_string());
                    }
                }
                self.db_indices = Some(DatabaseIndices {
                    clip_index: Arc::new(clip_index),
                    face_index: Arc::new(FaceIndex {
                        entries: Vec::new(),
                        file_count: 0,
                    }),
                    ocr_index: Arc::new(OcrIndex {
                        entries: Vec::new(),
                        file_count: 0,
                    }),
                    clip_embedded_files: Arc::new(clip_embedded_files),
                    ocr_embedded_files: Arc::new(HashSet::new()),
                    similar_by_master: HashMap::new(),
                    phash_master_by_file: HashMap::new(),
                    phash_by_file: HashMap::new(),
                    video_frame_phashes_by_file: HashMap::new(),
                    sift_info_by_file: HashMap::new(),
                    sift_root_by_file: HashMap::new(),
                    sift_members_by_root: HashMap::new(),
                    skipped_processing_files: Arc::new(HashSet::new()),
                    basename_to_db_filename,
                    encoder,
                });
                self.db_loaded = true;
                self.db_loading = false;
                self.db_supplemental_loaded = false;
                self.db_supplemental_loading = true;
                self.semantic_status =
                    "CLIP ready. Loading OCR, face, duplicate, and SIFT indexes in the background."
                        .to_string();
                self.run_pending_db_request(false);
                self.db_rx = Some(rx);
            }
            Ok(DatabaseLoadMessage::ClipReady(Err(err))) => {
                self.fail_db_load(err);
            }
            Ok(DatabaseLoadMessage::SupplementalReady(Ok(data))) => {
                if let Some(indices) = self.db_indices.as_mut() {
                    indices.face_index = Arc::new(data.face_index);
                    indices.ocr_index = Arc::new(data.ocr_index);
                    indices.ocr_embedded_files = Arc::new(data.ocr_embedded_files);
                    indices.similar_by_master = data.similar_by_master;
                    indices.phash_master_by_file = data.phash_master_by_file;
                    indices.phash_by_file = data.phash_by_file;
                    indices.video_frame_phashes_by_file = data.video_frame_phashes_by_file;
                    indices.sift_info_by_file = data.sift_info_by_file;
                    indices.sift_root_by_file = data.sift_root_by_file;
                    indices.sift_members_by_root = data.sift_members_by_root;
                    indices.skipped_processing_files = Arc::new(data.skipped_processing_files);
                    for key in indices
                        .phash_master_by_file
                        .keys()
                        .chain(indices.similar_by_master.keys())
                        .chain(indices.sift_info_by_file.keys())
                    {
                        if let Some(fname) = Path::new(key).file_name() {
                            indices
                                .basename_to_db_filename
                                .entry(fname.to_string_lossy().to_lowercase())
                                .or_insert_with(|| key.clone());
                        }
                    }
                }
                self.db_supplemental_loaded = true;
                self.db_supplemental_loading = false;
                if self.semantic_status.starts_with("CLIP ready.") {
                    self.semantic_status =
                        "CLIP, OCR, face, duplicate, and SIFT indexes ready.".to_string();
                }
                self.run_pending_db_request(true);
            }
            Ok(DatabaseLoadMessage::SupplementalReady(Err(err))) => {
                self.db_supplemental_loading = false;
                self.semantic_status = format!(
                    "CLIP is ready, but supplemental database indexes failed to load: {err}"
                );
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                self.db_rx = Some(rx);
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                if !self.db_loaded {
                    self.fail_db_load("AI DB loader thread disconnected unexpectedly.".to_string());
                } else {
                    self.db_supplemental_loading = false;
                }
            }
        }
    }

    pub(crate) fn fail_db_load(&mut self, err: String) {
        self.db_loading = false;
        self.db_supplemental_loaded = false;
        self.db_supplemental_loading = false;
        self.db_failed = true;
        self.pending_search_request = None;
        self.pending_semantic_search_mode = None;
        self.semantic_status = format!("AI DB initialization failed: {err}");
    }

    pub(crate) fn run_pending_db_request(&mut self, supplemental_ready: bool) {
        let maybe_ctx = self
            .ctx_shared
            .lock()
            .ok()
            .and_then(|lock| lock.as_ref().cloned());
        let Some(ctx) = maybe_ctx else {
            return;
        };
        if let Some(request) = self.pending_search_request.take() {
            if supplemental_ready || matches!(&request, PendingSearchRequest::Similar { .. }) {
                self.run_search_request_now(request, &ctx);
                return;
            }
            self.pending_search_request = Some(request);
        }
        if let Some(mode) = self.pending_semantic_search_mode.take() {
            if supplemental_ready || mode == SearchMode::Clip {
                self.run_semantic_search_mode(mode, &ctx);
            } else {
                self.pending_semantic_search_mode = Some(mode);
            }
        }
    }

    pub(crate) fn poll_on_demand_embeddings(&mut self, ctx: &egui::Context) {
        let Some(rx) = self.on_demand_embed_rx.take() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(payload)) => {
                let label = Self::label_for_request(&payload.request);
                match payload.request {
                    PendingSearchRequest::Similar {
                        db_file_name,
                        media_path,
                        is_video,
                        timestamp_sec,
                    } => {
                        if let Some(query_vector) = payload.clip_vector {
                            let source = SearchResult {
                                rank: 0,
                                score: 1.0,
                                file_name: db_file_name
                                    .clone()
                                    .unwrap_or_else(|| media_path.to_string_lossy().to_string()),
                                is_video,
                                timestamp_sec,
                                media_path: Some(media_path),
                                ocr_term_hits: 0,
                                ocr_query_terms: 0,
                                ocr_phrase_query: false,
                            };
                            self.show_most_similar_from_vector(query_vector, Some(source), &label);
                        } else {
                            self.semantic_status =
                                format!("No CLIP embedding produced for {label}");
                        }
                    }
                    PendingSearchRequest::Person { .. } => {
                        self.show_more_of_this_person_with_vectors(payload.face_vectors, &label);
                    }
                }
                ctx.request_repaint();
            }
            Ok(Err(err)) => {
                self.semantic_status = format!("On-demand embedding failed: {err}");
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                self.on_demand_embed_rx = Some(rx);
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.semantic_status =
                    "On-demand embedding worker disconnected unexpectedly.".to_string();
            }
        }
    }
}
