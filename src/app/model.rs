use super::{DiagnosticState, binary::FileChunk, metadata::VideoMetadata};
use anyhow::Result;
use eframe::egui;
use ort::session::Session;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Instant, SystemTime};
use tokenizers::Tokenizer;

#[derive(PartialEq, Clone, Copy)]
pub(crate) enum SidePanelMode {
    Layout,
    Exif,
    Duplicates,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SearchMode {
    Filename,
    Clip,
    Ocr,
}

pub(crate) enum OpenRequest {
    Single(PathBuf),
    Comparison(Vec<PathBuf>),
}

#[derive(Clone)]
pub(crate) struct SearchSnapshot {
    pub(crate) semantic_query: String,
    pub(crate) applied_filename_query: String,
    pub(crate) filename_search_results: Option<Vec<usize>>,
    pub(crate) semantic_folder: String,
    pub(crate) semantic_limit: usize,
    pub(crate) semantic_video_only: bool,
    pub(crate) semantic_mode: SearchMode,
    pub(crate) semantic_results: Vec<SearchResult>,
    pub(crate) semantic_results_mode: Option<SearchMode>,
    pub(crate) semantic_status: String,
}

#[derive(Clone)]
pub(crate) struct GalleryImageSnapshot {
    pub(crate) images: Vec<PathBuf>,
    pub(crate) current_index: usize,
    pub(crate) navigation_indices: Option<Arc<[usize]>>,
    pub(crate) navigation_position: usize,
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct GalleryFilterKey {
    pub(crate) scan_generation: u64,
    pub(crate) grid_loading: bool,
    pub(crate) applied_filename_query: String,
    pub(crate) has_filename_results: bool,
    pub(crate) semantic_mode: SearchMode,
    pub(crate) semantic_results_mode: Option<SearchMode>,
    pub(crate) video_only: bool,
}

pub(crate) struct FlatRefreshResult {
    pub(crate) generation: u64,
    pub(crate) directory: PathBuf,
    pub(crate) images: Vec<PathBuf>,
}

#[derive(Clone, Copy)]
pub(crate) struct ImageViewState {
    pub(crate) zoom: f32,
    pub(crate) offset: egui::Vec2,
}

pub(crate) struct ClipIndex {
    pub(crate) entries: Vec<ClipEntry>,
    pub(crate) dim: usize,
    pub(crate) file_count: usize,
}

pub(crate) struct FaceIndex {
    pub(crate) entries: Vec<FaceEntry>,
    pub(crate) file_count: usize,
}

pub(crate) struct OcrIndex {
    pub(crate) entries: Vec<OcrEntry>,
    pub(crate) file_count: usize,
}

#[derive(Clone)]
pub(crate) struct ClipEntry {
    pub(crate) file_name: Arc<str>,
    pub(crate) is_video: bool,
    pub(crate) timestamp_sec: f32,
    pub(crate) vector: Vec<f32>,
}

#[derive(Clone)]
pub(crate) struct FaceEntry {
    pub(crate) file_name: Arc<str>,
    pub(crate) is_video: bool,
    pub(crate) timestamp_sec: f32,
    pub(crate) vector: Vec<f32>,
}

#[derive(Clone)]
pub(crate) struct GallerySelection {
    pub(crate) path: PathBuf,
    pub(crate) db_filename: Option<String>,
    pub(crate) is_video: bool,
}

impl GallerySelection {
    pub(crate) fn matches(&self, path: &std::path::Path, db_filename: Option<&str>) -> bool {
        match (self.db_filename.as_deref(), db_filename) {
            (Some(selected), Some(candidate)) => selected == candidate,
            _ => self.path == path,
        }
    }
}

#[derive(Clone)]
pub(crate) struct FaceDetail {
    pub(crate) vector: Vec<f32>,
    pub(crate) bbox: [f32; 4],
}

pub(crate) struct FaceComparisonResult {
    pub(crate) paths: Vec<PathBuf>,
    pub(crate) active_index: usize,
    pub(crate) overlay_boxes: Vec<(PathBuf, [f32; 4])>,
    pub(crate) summary: String,
}

pub(crate) struct MetadataLoadResult {
    pub(crate) generation: u64,
    pub(crate) path: PathBuf,
    pub(crate) exif_data: String,
    pub(crate) chunks: Vec<FileChunk>,
    pub(crate) load_exif: bool,
    pub(crate) load_layout: bool,
}

pub(crate) struct MetadataLoadRequest {
    pub(crate) logical_path: PathBuf,
    pub(crate) inspect_path: PathBuf,
    pub(crate) generation: u64,
    pub(crate) load_exif: bool,
    pub(crate) load_layout: bool,
}

pub(crate) struct MetadataJobQueueState {
    pub(crate) pending: Option<MetadataLoadRequest>,
    pub(crate) shutdown: bool,
}

pub(crate) struct MetadataJobQueue {
    pub(crate) state: Mutex<MetadataJobQueueState>,
    pub(crate) wake: Condvar,
}

impl Default for MetadataJobQueue {
    fn default() -> Self {
        Self {
            state: Mutex::new(MetadataJobQueueState {
                pending: None,
                shutdown: false,
            }),
            wake: Condvar::new(),
        }
    }
}

pub(crate) struct SemanticSearchWorkerResult {
    pub(crate) generation: u64,
    pub(crate) mode: SearchMode,
    pub(crate) rows: Vec<SearchResult>,
    pub(crate) took_ms: u128,
    pub(crate) indexed_items: usize,
    pub(crate) folder_scope: String,
    pub(crate) display_label: String,
    pub(crate) limit: usize,
    pub(crate) video_only: bool,
}

pub(crate) struct FilenameSearchWorkerResult {
    pub(crate) generation: u64,
    pub(crate) gallery_generation: u64,
    pub(crate) matches: Vec<usize>,
}

#[derive(Clone)]
pub(crate) struct OcrEntry {
    pub(crate) file_name: Arc<str>,
    pub(crate) is_video: bool,
    pub(crate) timestamp_sec: f32,
    pub(crate) text_lower: String,
}

#[derive(Clone)]
pub(crate) struct SearchResult {
    pub(crate) rank: usize,
    pub(crate) score: f32,
    pub(crate) file_name: String,
    pub(crate) is_video: bool,
    pub(crate) timestamp_sec: f32,
    pub(crate) media_path: Option<PathBuf>,
    pub(crate) ocr_term_hits: usize,
    pub(crate) ocr_query_terms: usize,
    pub(crate) ocr_phrase_query: bool,
}

#[derive(Clone)]
pub(crate) enum PendingSearchRequest {
    Similar {
        db_file_name: Option<String>,
        media_path: PathBuf,
        is_video: bool,
        timestamp_sec: f32,
    },
    Person {
        db_file_name: Option<String>,
        media_path: PathBuf,
        is_video: bool,
    },
}

pub(crate) struct OnDemandEmbedResult {
    pub(crate) request: PendingSearchRequest,
    pub(crate) clip_vector: Option<Vec<f32>>,
    pub(crate) face_vectors: Vec<Vec<f32>>,
}

pub(crate) struct SiftRepairResult {
    pub(crate) summary: String,
    pub(crate) files: usize,
}

pub(crate) struct SiftAlignAllResult {
    pub(crate) reference: PathBuf,
    pub(crate) aligned_paths: HashMap<PathBuf, PathBuf>,
    pub(crate) summary: String,
    pub(crate) details: Vec<String>,
    pub(crate) output_dir: PathBuf,
}

#[derive(Clone)]
pub(crate) struct SimilarFile {
    pub(crate) file_name: String,
    pub(crate) is_video: bool,
    pub(crate) similarity_pct: Option<f32>,
}

#[derive(Clone)]
pub(crate) struct VideoFramePhash {
    pub(crate) timestamp_sec: f32,
    pub(crate) phash: u64,
}

#[derive(Clone, Default)]
pub(crate) struct SiftInfo {
    pub(crate) match_file: Option<String>,
    pub(crate) score: Option<f32>,
    pub(crate) inliers: Option<i32>,
    pub(crate) good_matches: Option<i32>,
    pub(crate) inlier_ratio: Option<f32>,
    pub(crate) checked: Option<bool>,
}

pub(crate) fn valid_sift_link(info: &SiftInfo) -> bool {
    info.checked == Some(true)
        && info.match_file.is_some()
        && info.inliers.unwrap_or(0) >= 10
        && info.inlier_ratio.unwrap_or(0.0) >= 0.40
        && info.score.unwrap_or(0.0) >= 0.0
}

pub(crate) struct ClipTextEncoder {
    pub(crate) tokenizer: Tokenizer,
    pub(crate) session: Session,
    pub(crate) context_len: usize,
}

pub(crate) struct DatabaseIndices {
    pub(crate) clip_index: Arc<ClipIndex>,
    pub(crate) face_index: Arc<FaceIndex>,
    pub(crate) ocr_index: Arc<OcrIndex>,
    pub(crate) clip_embedded_files: Arc<HashSet<String>>,
    pub(crate) ocr_embedded_files: Arc<HashSet<String>>,
    pub(crate) similar_by_master: HashMap<String, Vec<SimilarFile>>,
    pub(crate) phash_master_by_file: HashMap<String, String>,
    pub(crate) phash_by_file: HashMap<String, u64>,
    pub(crate) video_frame_phashes_by_file: HashMap<String, Vec<VideoFramePhash>>,
    pub(crate) sift_info_by_file: HashMap<String, SiftInfo>,
    pub(crate) sift_root_by_file: HashMap<String, String>,
    pub(crate) sift_members_by_root: HashMap<String, Vec<String>>,
    pub(crate) skipped_processing_files: Arc<HashSet<String>>,
    pub(crate) basename_to_db_filename: Arc<HashMap<String, Vec<String>>>,
    pub(crate) encoder: ClipTextEncoder,
}

pub(crate) struct SupplementalDbData {
    pub(crate) face_index: FaceIndex,
    pub(crate) ocr_index: OcrIndex,
    pub(crate) ocr_embedded_files: HashSet<String>,
    pub(crate) similar_by_master: HashMap<String, Vec<SimilarFile>>,
    pub(crate) phash_master_by_file: HashMap<String, String>,
    pub(crate) phash_by_file: HashMap<String, u64>,
    pub(crate) video_frame_phashes_by_file: HashMap<String, Vec<VideoFramePhash>>,
    pub(crate) sift_info_by_file: HashMap<String, SiftInfo>,
    pub(crate) sift_root_by_file: HashMap<String, String>,
    pub(crate) sift_members_by_root: HashMap<String, Vec<String>>,
    pub(crate) skipped_processing_files: HashSet<String>,
}

pub(crate) enum DatabaseLoadMessage {
    ClipReady(Result<(ClipIndex, ClipTextEncoder), String>),
    SupplementalReady(Result<SupplementalDbData, String>),
}

#[derive(Clone, Copy)]
pub(crate) enum CropDragMode {
    New,
    Move,
    Left,
    Right,
    Top,
    Bottom,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

pub(crate) struct ImageEditor {
    pub(crate) source_path: PathBuf,
    pub(crate) image: image::DynamicImage,
    pub(crate) texture: egui::TextureHandle,
    pub(crate) crop_min: egui::Pos2,
    pub(crate) crop_max: egui::Pos2,
    pub(crate) crop_drag_mode: Option<CropDragMode>,
    pub(crate) crop_drag_origin: egui::Pos2,
    pub(crate) crop_drag_initial_min: egui::Pos2,
    pub(crate) crop_drag_initial_max: egui::Pos2,
    pub(crate) status: String,
}

pub(crate) struct ImageViewer {
    pub(crate) diagnostics: DiagnosticState,
    pub(crate) images: Vec<PathBuf>,
    pub(crate) current_index: usize,
    pub(crate) comparison_paths: Option<Vec<PathBuf>>,
    pub(crate) comparison_view_states: HashMap<PathBuf, ImageViewState>,
    pub(crate) comparison_sync_view: bool,
    pub(crate) comparison_aligned_paths: HashMap<PathBuf, PathBuf>,
    pub(crate) comparison_alignment_temp_dir: Option<PathBuf>,
    pub(crate) comparison_alignment_status: String,
    pub(crate) comparison_path_dialog_open: bool,
    pub(crate) comparison_path_input: String,
    pub(crate) zoom: f32,
    pub(crate) offset: egui::Vec2,
    pub(crate) exif_data: String,
    pub(crate) side_panel_metadata_path: Option<PathBuf>,
    pub(crate) side_panel_layout_path: Option<PathBuf>,
    pub(crate) show_exif: bool,
    pub(crate) side_panel_window_expanded: bool,
    pub(crate) side_panel_open_pending: bool,
    pub(crate) side_panel_expand_target_width: Option<f32>,
    pub(crate) side_panel_open_pending_frames: u8,
    pub(crate) chunks: Vec<FileChunk>,
    pub(crate) viewport_bg: Option<egui::Color32>,
    pub(crate) pending_initial_window_size: Option<egui::Vec2>,
    pub(crate) rx: Receiver<OpenRequest>,
    pub(crate) show_grid: bool,
    pub(crate) recursive_images: Arc<[PathBuf]>,
    pub(crate) recursive_scan_paths: Vec<PathBuf>,
    pub(crate) recursive_images_snapshot: Arc<[PathBuf]>,
    pub(crate) recursive_video_indices: Vec<usize>,
    pub(crate) gallery_thumbnail_scale: f32,
    pub(crate) grid_loading: bool,
    pub(crate) recursive_rx: Option<Receiver<PathBuf>>,
    pub(crate) recursive_scan_token: Arc<AtomicU64>,
    pub(crate) recursive_scan_generation: u64,
    pub(crate) gallery_thumbnail_generation: u64,
    pub(crate) gallery_visible_thumbnail_paths: HashSet<PathBuf>,
    pub(crate) back_target_is_gallery: bool,
    pub(crate) side_panel_mode: SidePanelMode,
    pub(crate) exif_search: String,
    pub(crate) open_target: PathBuf,
    pub(crate) open_target_is_dir: bool,
    pub(crate) flat_loading: bool,
    pub(crate) flat_refresh_in_flight: bool,
    pub(crate) flat_refresh_generation: u64,
    pub(crate) flat_last_refresh_check: Instant,
    pub(crate) flat_directory_mtime: Option<SystemTime>,
    pub(crate) flat_images_shared: Arc<Mutex<Option<FlatRefreshResult>>>,
    pub(crate) current_dimensions: String,
    pub(crate) current_file_size: String,
    pub(crate) ctx_shared: Arc<Mutex<Option<egui::Context>>>,
    pub(crate) thumbnail_textures: HashMap<PathBuf, egui::TextureHandle>,
    pub(crate) thumbnail_loading: HashSet<PathBuf>,
    pub(crate) thumbnail_failed: HashSet<PathBuf>,
    pub(crate) thumbnail_retry_at: HashMap<PathBuf, Instant>,
    pub(crate) thumbnail_rx: std::sync::mpsc::Receiver<(u64, PathBuf, egui::ColorImage)>,
    pub(crate) thumbnail_tx: std::sync::mpsc::Sender<(u64, PathBuf, egui::ColorImage)>,
    pub(crate) thumbnail_active_threads: usize,
    pub(crate) viewer_textures: HashMap<PathBuf, egui::TextureHandle>,
    pub(crate) viewer_texture_loading: HashSet<PathBuf>,
    pub(crate) viewer_texture_failed: HashSet<PathBuf>,
    pub(crate) viewer_texture_retry_at: HashMap<PathBuf, Instant>,
    pub(crate) viewer_texture_revisions: HashMap<PathBuf, u64>,
    pub(crate) viewer_texture_rx:
        std::sync::mpsc::Receiver<(PathBuf, u64, Result<egui::ColorImage, String>)>,
    pub(crate) viewer_texture_tx:
        std::sync::mpsc::Sender<(PathBuf, u64, Result<egui::ColorImage, String>)>,
    pub(crate) video_duration_cache: std::cell::RefCell<HashMap<PathBuf, Option<VideoMetadata>>>,
    pub(crate) video_duration_loading: std::cell::RefCell<HashSet<PathBuf>>,
    pub(crate) video_duration_rx: std::sync::mpsc::Receiver<(PathBuf, Option<VideoMetadata>)>,
    pub(crate) video_duration_tx: std::sync::mpsc::Sender<(PathBuf, Option<VideoMetadata>)>,
    pub(crate) db_loaded: bool,
    pub(crate) db_loading: bool,
    pub(crate) db_supplemental_loaded: bool,
    pub(crate) db_supplemental_loading: bool,
    pub(crate) db_failed: bool,
    pub(crate) db_rx: Option<Receiver<DatabaseLoadMessage>>,
    pub(crate) db_indices: Option<DatabaseIndices>,
    pub(crate) semantic_query: String,
    pub(crate) search_history: Vec<SearchSnapshot>,
    pub(crate) search_forward_history: Vec<SearchSnapshot>,
    pub(crate) gallery_image_forward: Option<GalleryImageSnapshot>,
    pub(crate) gallery_scan_generation: u64,
    pub(crate) gallery_filter_cache_key: Option<GalleryFilterKey>,
    pub(crate) gallery_filtered_indices: Arc<[usize]>,
    pub(crate) gallery_navigation_indices: Option<Arc<[usize]>>,
    pub(crate) gallery_navigation_position: usize,
    pub(crate) applied_filename_query: String,
    pub(crate) filename_search_results: Option<Vec<usize>>,
    pub(crate) semantic_folder: String,
    pub(crate) semantic_limit: usize,
    pub(crate) semantic_video_only: bool,
    pub(crate) semantic_mode: SearchMode,
    pub(crate) semantic_results: Vec<SearchResult>,
    pub(crate) semantic_results_mode: Option<SearchMode>,
    pub(crate) semantic_status: String,
    pub(crate) pending_search_request: Option<PendingSearchRequest>,
    pub(crate) pending_semantic_search_mode: Option<SearchMode>,
    pub(crate) on_demand_embed_rx: Option<Receiver<Result<OnDemandEmbedResult, String>>>,
    pub(crate) compare_target: Option<PathBuf>,
    pub(crate) sift_pair_overlay: Option<String>,
    pub(crate) expanded_duplicate_rows: HashSet<String>,
    pub(crate) sift_running: bool,
    pub(crate) sift_rx: Option<Receiver<Result<String, String>>>,
    pub(crate) sift_align_all_running: bool,
    pub(crate) sift_align_all_rx: Option<Receiver<Result<SiftAlignAllResult, String>>>,
    pub(crate) selected_grid_items: Vec<GallerySelection>,
    pub(crate) sift_repair_running: bool,
    pub(crate) sift_repair_rx: Option<Receiver<Result<SiftRepairResult, String>>>,
    pub(crate) face_compare_running: bool,
    pub(crate) face_compare_rx: Option<Receiver<Result<FaceComparisonResult, String>>>,
    pub(crate) face_overlay_boxes: HashMap<PathBuf, Vec<[f32; 4]>>,
    pub(crate) image_editor: Option<ImageEditor>,
    pub(crate) db_filename_by_path: HashMap<PathBuf, String>,
    pub(crate) video_still_cache: std::cell::RefCell<HashMap<PathBuf, PathBuf>>,
    pub(crate) resolution_size_cache: std::cell::RefCell<HashMap<PathBuf, String>>,
    pub(crate) db_filename_cache: std::cell::RefCell<HashMap<PathBuf, String>>,
    pub(crate) metadata_rx: std::sync::mpsc::Receiver<MetadataLoadResult>,
    pub(crate) metadata_worker_queue: Arc<MetadataJobQueue>,
    pub(crate) metadata_loading: bool,
    pub(crate) metadata_loading_path: Option<PathBuf>,
    pub(crate) metadata_loading_exif: bool,
    pub(crate) metadata_loading_layout: bool,
    pub(crate) metadata_generation: u64,
    pub(crate) semantic_search_rx: Option<Receiver<Result<SemanticSearchWorkerResult, String>>>,
    pub(crate) semantic_search_generation: u64,
    pub(crate) filename_search_rx: Option<Receiver<FilenameSearchWorkerResult>>,
    pub(crate) filename_search_generation: u64,
    pub(crate) pending_similarity_source: Option<SearchResult>,
    pub(crate) pending_similarity_label: Option<String>,
    pub(crate) show_home_page: bool,
    pub(crate) home_current_dir: Option<PathBuf>,
    pub(crate) home_selected_dir: Option<PathBuf>,
    pub(crate) viewer_rotation_quarter_turns: u8,
    pub(crate) viewer_rotation_path: Option<PathBuf>,
}
