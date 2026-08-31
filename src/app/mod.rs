use eframe::egui;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, ListArray, RecordBatch,
    RecordBatchIterator, StringArray, StructArray,
};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::table::AddDataMode;
use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;
use serde_json::Value;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

mod binary;
mod bootstrap;
mod clipboard;
mod diagnostics;
mod formatting;
mod hashing;
mod media_scan;
mod metadata;
mod model;
mod paths;
mod platform;
mod runtime;
mod search;
#[cfg(test)]
mod tests;
mod ui;
mod viewer_assets;

// This is the small internal facade used by the feature modules. Keep the
// exports explicit so a module's cross-cutting dependencies remain visible.
pub(crate) use bootstrap::{initial_window_size, run};
pub(crate) use clipboard::{
    clipboard_paste_signal, copy_image_file_to_clipboard, image_path_from_pasted_text,
    is_clipboard_image_path, save_clipboard_image_to_temp,
};
#[cfg(test)]
pub(crate) use diagnostics::build_info_value;
pub(crate) use diagnostics::{DiagnosticState, handle_cli};
pub(crate) use formatting::{
    file_resolution_and_size, file_size_and_modified, raw_exif_layout_job, sift_info_line,
    wrapping_monospace_path,
};
pub(crate) use hashing::{
    compute_face_details, compute_on_demand_embeddings, compute_sift_summary,
    run_sift_alignment_batch, run_sift_repair_for_files,
};
pub(crate) use media_scan::{
    collect_flat_images, collect_images_recursive_cancelable, is_supported_media_path,
};
pub(crate) use metadata::{
    VideoMetadata, format_video_duration, load_video_metadata, load_video_thumbnail,
    run_metadata_worker,
};
pub(crate) use model::{
    ClipEntry, ClipIndex, ClipTextEncoder, CropDragMode, DatabaseIndices, DatabaseLoadMessage,
    FaceComparisonResult, FaceDetail, FaceEntry, FaceIndex, FilenameSearchWorkerResult,
    FlatRefreshResult, GalleryFilterKey, GalleryImageSnapshot, GallerySelection, ImageEditor,
    ImageViewState, ImageViewer, MetadataJobQueue, MetadataLoadRequest, MetadataLoadResult,
    OcrEntry, OcrIndex, OnDemandEmbedResult, OpenRequest, PendingSearchRequest, SearchMode,
    SearchResult, SearchSnapshot, SemanticSearchWorkerResult, SidePanelMode, SiftAlignAllResult,
    SiftInfo, SiftRepairResult, SimilarFile, SupplementalDbData, VideoFramePhash,
};
pub(crate) use paths::{
    MEDIA_INDEX_TABLE, db_filename_from_video_still_path, file_matches_folder, get_db_dir,
    get_db_roots, is_video_path, load_or_discover_db_roots, open_in_dolphin_or_fallback,
    partial_path_matches, resolve_media_indexer_dir, resolve_media_path,
    resolve_on_demand_embeddings_script_path, resolve_source_path, resolve_video_still,
    text_edit_enter_pressed,
};
pub(crate) use platform::{
    get_system_disks, is_path_ai_backed, is_path_ai_backed_with_roots, normalized_path_for_match,
    path_matches_db_root,
};
pub(crate) use search::search_clip_ann;
pub(crate) use search::search_face_ann;
pub(crate) use search::{
    FACE_MATCH_MIN_SCORE, collapse_sift_grouped_results, dot, draw_embedding_markers,
    duplicate_database_detail_lines, load_clip_database_index, load_supplemental_database_indices,
    search_face_index, search_index, search_ocr_index, similarity_to_active, string_col,
};
pub(crate) use viewer_assets::{
    IMAGE_VIEWER_TOP_BAR_HEIGHT, INITIAL_IMAGE_DISPLAY_HEIGHT, viewer_color_image,
    viewer_color_image_ref,
};
