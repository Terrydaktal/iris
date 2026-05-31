use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use arrow_array::{
    Array, BooleanArray, Float32Array, Float64Array, ListArray, RecordBatch,
    StringArray, StructArray,
};
use clap::Parser;
use eframe::egui;
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;
use serde_json::Value;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

const SIFT_MIN_INLIERS: i32 = 10;
const SIFT_MIN_INLIER_RATIO: f32 = 0.75;
const SIFT_MIN_SCORE: f32 = 0.0;
const FACE_MATCH_MIN_SCORE: f32 = 0.35;

#[derive(Parser, Debug)]
#[command(name = "clip-viewer")]
#[command(about = "Fast native Rust CLIP viewer for embedimages LanceDB")]
struct Args {
    #[arg(long, default_value = "./lancedb")]
    db_dir: PathBuf,

    #[arg(long, default_value = "media_index")]
    table: String,

    #[arg(long, default_value = "models/clip-text/clip_text.onnx")]
    text_onnx: PathBuf,

    #[arg(long, default_value = "models/clip-text/tokenizer.json")]
    tokenizer_json: PathBuf,

    #[arg(long, default_value_t = 80)]
    default_limit: usize,

    #[arg(long = "collection-root", value_name = "COLLECTION_ID=/ABS/PATH")]
    collection_roots: Vec<String>,
}

struct ClipIndex {
    entries: Vec<ClipEntry>,
    dim: usize,
    file_count: usize,
}

struct FaceIndex {
    entries: Vec<FaceEntry>,
    file_count: usize,
}

struct OcrIndex {
    entries: Vec<OcrEntry>,
    file_count: usize,
}

#[derive(Clone)]
struct ClipEntry {
    file_name: Arc<str>,
    is_video: bool,
    timestamp_sec: f32,
    vector: Vec<f32>,
}

#[derive(Clone)]
struct FaceEntry {
    file_name: Arc<str>,
    is_video: bool,
    timestamp_sec: f32,
    vector: Vec<f32>,
}

#[derive(Clone)]
struct OcrEntry {
    file_name: Arc<str>,
    is_video: bool,
    timestamp_sec: f32,
    text_lower: String,
}

#[derive(Clone)]
struct SearchResult {
    rank: usize,
    score: f32,
    file_name: String,
    is_video: bool,
    timestamp_sec: f32,
    media_path: Option<PathBuf>,
}

#[derive(Clone)]
struct ResultTab {
    title: String,
    results: Vec<SearchResult>,
}

#[derive(Clone)]
struct SimilarFile {
    file_name: String,
    is_video: bool,
    similarity_pct: Option<f32>,
}

#[derive(Clone, Default)]
struct SiftInfo {
    match_file: Option<String>,
    score: Option<f32>,
    inliers: Option<i32>,
    good_matches: Option<i32>,
    inlier_ratio: Option<f32>,
    checked: Option<bool>,
}

#[derive(Clone, Copy)]
struct SiftRepairThresholds {
    min_inliers: i32,
    min_inlier_ratio: f32,
    min_score: f32,
}

struct ClipTextEncoder {
    tokenizer: Tokenizer,
    session: Session,
    context_len: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SearchMode {
    Clip,
    Ocr,
}

struct ViewerApp {
    index: Arc<ClipIndex>,
    face_index: Arc<FaceIndex>,
    ocr_index: Arc<OcrIndex>,
    db_dir: PathBuf,
    table_name: String,
    roots: Arc<HashMap<String, PathBuf>>,
    similar_by_master: HashMap<String, Vec<SimilarFile>>,
    sift_info_by_file: HashMap<String, SiftInfo>,
    sift_root_by_file: HashMap<String, String>,
    sift_members_by_root: HashMap<String, Vec<String>>,
    encoder: ClipTextEncoder,
    query: String,
    limit: usize,
    video_only: bool,
    search_mode: SearchMode,
    tabs: Vec<ResultTab>,
    active_tab: usize,
    status: String,
    cache: HashMap<PathBuf, egui::TextureHandle>,
    loaded: usize,
    searched_vectors: usize,
    master_files: usize,
    selected_file: Option<String>,
    selected_master: Option<String>,
    repair_rx: Option<Receiver<Result<Value, String>>>,
    repair_running: bool,
    face_rerun_rx: Option<Receiver<Result<Value, String>>>,
    face_rerun_running: bool,
    selection_mode: bool,
    selected_images: HashSet<String>,
    selected_pair_sift_diag: Option<String>,
    selected_pair_key: Option<(String, String)>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let roots = Arc::new(parse_collection_roots(&args.collection_roots)?);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to create Tokio runtime")?;

    let started = Instant::now();
    let index = Arc::new(runtime.block_on(load_clip_index(&args.db_dir, &args.table))?);
    let face_index = Arc::new(runtime.block_on(load_face_index(&args.db_dir, &args.table))?);
    let ocr_index = Arc::new(runtime.block_on(load_ocr_index(&args.db_dir, &args.table))?);
    eprintln!(
        "loaded {} CLIP vectors for {} master files, {} face vectors for {} files, and {} OCR entries for {} files in {:.2}s",
        index.entries.len(),
        index.file_count,
        face_index.entries.len(),
        face_index.file_count,
        ocr_index.entries.len(),
        ocr_index.file_count,
        started.elapsed().as_secs_f32()
    );

    let encoder = ClipTextEncoder::new(&args.text_onnx, &args.tokenizer_json, 64)
        .context("failed to load native Rust CLIP text encoder")?;
    let similar_by_master = runtime
        .block_on(load_similar_map(&args.db_dir, &args.table))
        .context("failed to load pHash/VideoHash similarity map")?;
    let sift_info_by_file = runtime
        .block_on(load_sift_info_map(&args.db_dir, &args.table))
        .context("failed to load SIFT info map")?;
    let (sift_root_by_file, sift_members_by_root) = runtime
        .block_on(load_sift_groups(&args.db_dir, &args.table))
        .context("failed to load SIFT master grouping map")?;

    let app = ViewerApp {
        searched_vectors: index.entries.len(),
        master_files: index.file_count,
        index,
        face_index,
        ocr_index,
        db_dir: args.db_dir.clone(),
        table_name: args.table.clone(),
        roots,
        similar_by_master,
        sift_info_by_file,
        sift_root_by_file,
        sift_members_by_root,
        encoder,
        query: String::new(),
        limit: args.default_limit.clamp(1, 500),
        video_only: false,
        search_mode: SearchMode::Clip,
        tabs: vec![ResultTab {
            title: "Search".to_string(),
            results: Vec::new(),
        }],
        active_tab: 0,
        status: "Ready. Enter a phrase and press Search.".to_string(),
        cache: HashMap::new(),
        loaded: 0,
        selected_file: None,
        selected_master: None,
        repair_rx: None,
        repair_running: false,
        face_rerun_rx: None,
        face_rerun_running: false,
        selection_mode: false,
        selected_images: HashSet::new(),
        selected_pair_sift_diag: None,
        selected_pair_key: None,
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1480.0, 920.0]),
        ..Default::default()
    };

    eframe::run_native(
        "CLIP Viewer (Native Rust)",
        options,
        Box::new(move |_| Ok(Box::new(app))),
    )
    .map_err(|err| anyhow!("failed to launch viewer: {err}"))?;

    Ok(())
}

fn parse_collection_roots(values: &[String]) -> Result<HashMap<String, PathBuf>> {
    let mut roots = HashMap::new();
    for value in values {
        let (collection, root) = value
            .split_once('=')
            .ok_or_else(|| anyhow!("--collection-root must be COLLECTION_ID=/abs/path: {value}"))?;
        if collection.trim().is_empty() {
            bail!("collection id cannot be empty in --collection-root {value}");
        }
        let path = PathBuf::from(root)
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(root));
        roots.insert(collection.to_string(), path);
    }
    Ok(roots)
}

async fn load_clip_index(db_dir: &Path, table_name: &str) -> Result<ClipIndex> {
    let db = lancedb::connect(db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let table = db.open_table(table_name).execute().await?;
    let stream = table
        .query()
        .select(Select::columns(&[
            "file_name",
            "is_video",
            "skip_processing",
            "clip_groups",
        ]))
        .execute()
        .await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;

    let mut entries = Vec::new();
    let mut dim = None;
    let mut seen_files = std::collections::HashSet::new();

    for batch in batches {
        parse_batch(&batch, &mut entries, &mut dim, &mut seen_files)?;
    }

    let dim = dim.ok_or_else(|| anyhow!("no clip vectors found in table {table_name}"))?;
    Ok(ClipIndex {
        entries,
        dim,
        file_count: seen_files.len(),
    })
}

async fn load_face_index(db_dir: &Path, table_name: &str) -> Result<FaceIndex> {
    let db = lancedb::connect(db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let table = db.open_table(table_name).execute().await?;
    let stream = table
        .query()
        .select(Select::columns(&[
            "file_name",
            "is_video",
            "skip_processing",
            "face_groups",
        ]))
        .execute()
        .await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;

    let mut entries = Vec::new();
    let mut seen_files = HashSet::new();
    for batch in batches {
        parse_face_batch(&batch, &mut entries, &mut seen_files)?;
    }
    Ok(FaceIndex {
        entries,
        file_count: seen_files.len(),
    })
}

async fn load_ocr_index(db_dir: &Path, table_name: &str) -> Result<OcrIndex> {
    let db = lancedb::connect(db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let table = db.open_table(table_name).execute().await?;
    let stream = table
        .query()
        .select(Select::columns(&[
            "file_name",
            "is_video",
            "skip_processing",
            "ocr_groups",
        ]))
        .execute()
        .await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;

    let mut entries = Vec::new();
    let mut seen_files = HashSet::new();
    for batch in batches {
        parse_ocr_batch(&batch, &mut entries, &mut seen_files)?;
    }
    Ok(OcrIndex {
        entries,
        file_count: seen_files.len(),
    })
}

async fn load_similar_map(
    db_dir: &Path,
    table_name: &str,
) -> Result<HashMap<String, Vec<SimilarFile>>> {
    let db = lancedb::connect(db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let table = db.open_table(table_name).execute().await?;
    let stream = table
        .query()
        .select(Select::columns(&[
            "file_name",
            "is_video",
            "dedupe_match_file",
            "dedupe_similarity_pct",
        ]))
        .execute()
        .await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    let mut map: HashMap<String, Vec<SimilarFile>> = HashMap::new();

    for batch in batches {
        let file_names = string_col(&batch, "file_name")?;
        let is_video = bool_col(&batch, "is_video")?;
        let dedupe_match = string_col(&batch, "dedupe_match_file")?;
        let similarity_col = batch.column_by_name("dedupe_similarity_pct");

        for row in 0..batch.num_rows() {
            if dedupe_match.is_null(row) || file_names.is_null(row) {
                continue;
            }
            let master = dedupe_match.value(row).to_string();
            let similar_file = file_names.value(row).to_string();
            if master == similar_file {
                continue;
            }
            let similarity_pct = similarity_col.and_then(|col| float_value(col.as_ref(), row));
            map.entry(master).or_default().push(SimilarFile {
                file_name: similar_file,
                is_video: bool_value(is_video, row).unwrap_or(false),
                similarity_pct,
            });
        }
    }

    for values in map.values_mut() {
        values.sort_by(|a, b| {
            b.similarity_pct
                .unwrap_or(f32::NEG_INFINITY)
                .partial_cmp(&a.similarity_pct.unwrap_or(f32::NEG_INFINITY))
                .unwrap_or(Ordering::Equal)
        });
    }
    Ok(map)
}

async fn load_sift_info_map(db_dir: &Path, table_name: &str) -> Result<HashMap<String, SiftInfo>> {
    let db = lancedb::connect(db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let table = db.open_table(table_name).execute().await?;
    let stream = table
        .query()
        .select(Select::columns(&[
            "file_name",
            "sift_match_file",
            "sift_match_score",
            "sift_match_inliers",
            "sift_match_good_matches",
            "sift_match_inlier_ratio",
            "sift_match_checked",
        ]))
        .execute()
        .await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    let mut map: HashMap<String, SiftInfo> = HashMap::new();

    for batch in batches {
        let file_names = string_col(&batch, "file_name")?;
        let sift_match_file = string_col(&batch, "sift_match_file")?;
        let sift_score = batch.column_by_name("sift_match_score");
        let sift_inliers = batch.column_by_name("sift_match_inliers");
        let sift_good = batch.column_by_name("sift_match_good_matches");
        let sift_ratio = batch.column_by_name("sift_match_inlier_ratio");
        let sift_checked = bool_col(&batch, "sift_match_checked")?;

        for row in 0..batch.num_rows() {
            if file_names.is_null(row) {
                continue;
            }
            let file_name = file_names.value(row).to_string();
            let match_file = if sift_match_file.is_null(row) {
                None
            } else {
                Some(sift_match_file.value(row).to_string())
            };
            let inliers = sift_inliers.and_then(|col| {
                if let Some(arr) = col.as_any().downcast_ref::<arrow_array::Int32Array>() {
                    if arr.is_null(row) {
                        None
                    } else {
                        Some(arr.value(row))
                    }
                } else {
                    None
                }
            });
            let good_matches = sift_good.and_then(|col| {
                if let Some(arr) = col.as_any().downcast_ref::<arrow_array::Int32Array>() {
                    if arr.is_null(row) {
                        None
                    } else {
                        Some(arr.value(row))
                    }
                } else {
                    None
                }
            });
            map.insert(
                file_name,
                SiftInfo {
                    match_file,
                    score: sift_score.and_then(|col| float_value(col.as_ref(), row)),
                    inliers,
                    good_matches,
                    inlier_ratio: sift_ratio.and_then(|col| float_value(col.as_ref(), row)),
                    checked: bool_value(sift_checked, row),
                },
            );
        }
    }
    Ok(map)
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut idx = 0usize;
    while value >= 1024.0 && idx + 1 < UNITS.len() {
        value /= 1024.0;
        idx += 1;
    }
    if idx == 0 {
        format!("{bytes} {}", UNITS[idx])
    } else {
        format!("{value:.1} {}", UNITS[idx])
    }
}

fn file_resolution_and_size(path: &Path) -> String {
    let size_label = match fs::metadata(path) {
        Ok(meta) => format_bytes(meta.len()),
        Err(_) => "n/a".to_string(),
    };
    match image::image_dimensions(path) {
        Ok((w, h)) => format!("{w}x{h} | {size_label}"),
        Err(_) => size_label,
    }
}

fn short_tab_title(prefix: &str, file_name: &str) -> String {
    let leaf = Path::new(file_name)
        .file_name()
        .and_then(|x| x.to_str())
        .unwrap_or(file_name);
    let mut value = leaf.to_string();
    if value.chars().count() > 28 {
        value = value.chars().take(25).collect::<String>() + "...";
    }
    format!("{prefix}: {value}")
}

#[derive(Default)]
struct Dsu {
    parent: HashMap<String, String>,
}

impl Dsu {
    fn add(&mut self, value: &str) {
        self.parent
            .entry(value.to_string())
            .or_insert_with(|| value.to_string());
    }

    fn find(&mut self, value: &str) -> Option<String> {
        let parent = self.parent.get(value)?.clone();
        if parent == value {
            return Some(parent);
        }
        let root = self.find(&parent)?;
        self.parent.insert(value.to_string(), root.clone());
        Some(root)
    }

    fn union(&mut self, a: &str, b: &str) {
        self.add(a);
        self.add(b);
        let root_a = match self.find(a) {
            Some(v) => v,
            None => return,
        };
        let root_b = match self.find(b) {
            Some(v) => v,
            None => return,
        };
        if root_a == root_b {
            return;
        }
        if root_a < root_b {
            self.parent.insert(root_b, root_a);
        } else {
            self.parent.insert(root_a, root_b);
        }
    }
}

async fn load_sift_groups(
    db_dir: &Path,
    table_name: &str,
) -> Result<(HashMap<String, String>, HashMap<String, Vec<String>>)> {
    let db = lancedb::connect(db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let table = db.open_table(table_name).execute().await?;
    let stream = table
        .query()
        .select(Select::columns(&[
            "file_name",
            "is_video",
            "skip_processing",
            "sift_match_file",
            "sift_match_score",
            "sift_match_inliers",
            "sift_match_inlier_ratio",
            "sift_match_checked",
        ]))
        .execute()
        .await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;

    let mut master_images = HashSet::new();
    let mut direct_root_by_file: HashMap<String, String> = HashMap::new();

    for batch in batches {
        let file_names = string_col(&batch, "file_name")?;
        let is_video = bool_col(&batch, "is_video")?;
        let skip_processing = bool_col(&batch, "skip_processing")?;
        let sift_match_file = string_col(&batch, "sift_match_file")?;
        let sift_checked = bool_col(&batch, "sift_match_checked")?;

        for row in 0..batch.num_rows() {
            if file_names.is_null(row) {
                continue;
            }
            if bool_value(is_video, row).unwrap_or(false) {
                continue;
            }
            if bool_value(skip_processing, row) == Some(true) {
                continue;
            }
            let file_name = file_names.value(row).to_string();
            master_images.insert(file_name.clone());

            if sift_match_file.is_null(row) {
                continue;
            }
            if bool_value(sift_checked, row) != Some(true) {
                continue;
            }
            let target = sift_match_file.value(row).to_string();
            if target == file_name {
                continue;
            }
            direct_root_by_file.insert(file_name, target);
        }
    }

    let mut dsu = Dsu::default();
    for file_name in &master_images {
        dsu.add(file_name.as_str());
    }
    for (file_name, target) in &direct_root_by_file {
        if !master_images.contains(file_name.as_str()) || !master_images.contains(target.as_str()) {
            continue;
        }
        dsu.union(file_name.as_str(), target.as_str());
    }

    fn resolve_root(
        name: &str,
        direct_root_by_file: &HashMap<String, String>,
        master_images: &HashSet<String>,
    ) -> String {
        let mut seen: HashSet<String> = HashSet::new();
        let mut current = name.to_string();
        seen.insert(current.clone());
        loop {
            let next = match direct_root_by_file.get(current.as_str()) {
                Some(v) => v.clone(),
                None => return current,
            };
            if !master_images.contains(next.as_str()) {
                return current;
            }
            if !seen.insert(next.clone()) {
                // Cycle fallback: deterministic but stable.
                let mut values: Vec<String> = seen.into_iter().collect();
                values.sort_unstable();
                return values[0].clone();
            }
            current = next;
        }
    }

    let mut sift_root_by_file: HashMap<String, String> = HashMap::new();
    let mut sift_members_by_root: HashMap<String, Vec<String>> = HashMap::new();
    let mut raw_groups: HashMap<String, Vec<String>> = HashMap::new();
    for file_name in &master_images {
        let root = resolve_root(file_name.as_str(), &direct_root_by_file, &master_images);
        raw_groups.entry(root).or_default().push(file_name.clone());
    }
    for members in raw_groups.into_values() {
        if members.len() <= 1 {
            continue;
        }
        let mut sorted_members = members;
        sorted_members.sort_unstable();
        let canonical = resolve_root(
            sorted_members[0].as_str(),
            &direct_root_by_file,
            &master_images,
        );
        for member in &sorted_members {
            sift_root_by_file.insert(member.clone(), canonical.clone());
        }
        sift_members_by_root.insert(canonical, sorted_members);
    }

    Ok((sift_root_by_file, sift_members_by_root))
}

fn parse_batch(
    batch: &RecordBatch,
    entries: &mut Vec<ClipEntry>,
    dim: &mut Option<usize>,
    seen_files: &mut std::collections::HashSet<String>,
) -> Result<()> {
    let file_names = string_col(batch, "file_name")?;
    let is_video = bool_col(batch, "is_video")?;
    let skip_processing = bool_col(batch, "skip_processing")?;
    let clip_groups = list_col(batch, "clip_groups")?;

    for row in 0..batch.num_rows() {
        if bool_value(skip_processing, row) == Some(true)
            || clip_groups.is_null(row)
            || file_names.is_null(row)
        {
            continue;
        }
        let file_name = Arc::<str>::from(file_names.value(row));
        let video = bool_value(is_video, row).unwrap_or(false);
        let groups_any = clip_groups.value(row);
        let groups = groups_any
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| anyhow!("clip_groups value is not a struct array"))?;
        let timestamps = groups
            .column_by_name("timestamp_sec")
            .ok_or_else(|| anyhow!("clip_groups missing timestamp_sec"))?
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| anyhow!("timestamp_sec is not float32"))?;
        let vectors = groups
            .column_by_name("clip_embedding")
            .ok_or_else(|| anyhow!("clip_groups missing clip_embedding"))?
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| anyhow!("clip_embedding is not list<float>"))?;

        let mut added = false;
        for group_idx in 0..groups.len() {
            if vectors.is_null(group_idx) {
                continue;
            }
            let vector_any = vectors.value(group_idx);
            let vector_arr = vector_any
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| anyhow!("clip vector is not float32"))?;
            if vector_arr.is_empty() {
                continue;
            }
            let mut vector = Vec::with_capacity(vector_arr.len());
            for i in 0..vector_arr.len() {
                vector.push(vector_arr.value(i));
            }
            if let Some(expected) = *dim {
                if vector.len() != expected {
                    continue;
                }
            } else {
                *dim = Some(vector.len());
            }
            let ts = if timestamps.is_null(group_idx) {
                0.0
            } else {
                timestamps.value(group_idx)
            };
            entries.push(ClipEntry {
                file_name: file_name.clone(),
                is_video: video,
                timestamp_sec: ts,
                vector,
            });
            added = true;
        }
        if added {
            seen_files.insert(file_name.to_string());
        }
    }
    Ok(())
}

fn parse_face_batch(
    batch: &RecordBatch,
    entries: &mut Vec<FaceEntry>,
    seen_files: &mut HashSet<String>,
) -> Result<()> {
    let file_names = string_col(batch, "file_name")?;
    let is_video = bool_col(batch, "is_video")?;
    let skip_processing = bool_col(batch, "skip_processing")?;
    let face_groups = list_col(batch, "face_groups")?;

    for row in 0..batch.num_rows() {
        if bool_value(skip_processing, row) == Some(true)
            || face_groups.is_null(row)
            || file_names.is_null(row)
        {
            continue;
        }
        let file_name = Arc::<str>::from(file_names.value(row));
        let video = bool_value(is_video, row).unwrap_or(false);
        let groups_any = face_groups.value(row);
        let groups = groups_any
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| anyhow!("face_groups value is not a struct array"))?;
        let timestamps = groups
            .column_by_name("timestamp_sec")
            .ok_or_else(|| anyhow!("face_groups missing timestamp_sec"))?
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| anyhow!("face timestamp_sec is not float32"))?;
        let embeddings = groups
            .column_by_name("face_embeddings")
            .ok_or_else(|| anyhow!("face_groups missing face_embeddings"))?
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| anyhow!("face_embeddings is not list<list<float>>"))?;

        let mut added = false;
        for group_idx in 0..groups.len() {
            if embeddings.is_null(group_idx) {
                continue;
            }
            let faces_any = embeddings.value(group_idx);
            let faces = faces_any
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| anyhow!("face embedding group is not list<float>"))?;
            let ts = if timestamps.is_null(group_idx) {
                0.0
            } else {
                timestamps.value(group_idx)
            };
            for face_idx in 0..faces.len() {
                if faces.is_null(face_idx) {
                    continue;
                }
                let vector_any = faces.value(face_idx);
                let vector_arr = vector_any
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| anyhow!("face vector is not float32"))?;
                if vector_arr.is_empty() {
                    continue;
                }
                let mut vector = Vec::with_capacity(vector_arr.len());
                for i in 0..vector_arr.len() {
                    vector.push(vector_arr.value(i));
                }
                normalize_in_place(&mut vector);
                entries.push(FaceEntry {
                    file_name: file_name.clone(),
                    is_video: video,
                    timestamp_sec: ts,
                    vector,
                });
                added = true;
            }
        }
        if added {
            seen_files.insert(file_name.to_string());
        }
    }
    Ok(())
}

fn parse_ocr_batch(
    batch: &RecordBatch,
    entries: &mut Vec<OcrEntry>,
    seen_files: &mut HashSet<String>,
) -> Result<()> {
    let file_names = string_col(batch, "file_name")?;
    let is_video = bool_col(batch, "is_video")?;
    let skip_processing = bool_col(batch, "skip_processing")?;
    let ocr_groups = list_col(batch, "ocr_groups")?;

    for row in 0..batch.num_rows() {
        if bool_value(skip_processing, row) == Some(true) || ocr_groups.is_null(row) || file_names.is_null(row) {
            continue;
        }
        let file_name = Arc::<str>::from(file_names.value(row));
        let video = bool_value(is_video, row).unwrap_or(false);
        let groups_any = ocr_groups.value(row);
        let groups = groups_any
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| anyhow!("ocr_groups value is not a struct array"))?;
        let timestamps = groups
            .column_by_name("timestamp_sec")
            .ok_or_else(|| anyhow!("ocr_groups missing timestamp_sec"))?
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| anyhow!("ocr timestamp_sec is not float32"))?;
        let text_detected = groups
            .column_by_name("text_detected")
            .ok_or_else(|| anyhow!("ocr_groups missing text_detected"))?
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| anyhow!("text_detected is not bool"))?;
        let texts = groups
            .column_by_name("text")
            .ok_or_else(|| anyhow!("ocr_groups missing text"))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow!("ocr text is not string"))?;

        let mut added = false;
        for group_idx in 0..groups.len() {
            if text_detected.is_null(group_idx) || !text_detected.value(group_idx) || texts.is_null(group_idx) {
                continue;
            }
            let text = texts.value(group_idx).trim();
            if text.is_empty() {
                continue;
            }
            let ts = if timestamps.is_null(group_idx) {
                0.0
            } else {
                timestamps.value(group_idx)
            };
            entries.push(OcrEntry {
                file_name: file_name.clone(),
                is_video: video,
                timestamp_sec: ts,
                text_lower: text.to_lowercase(),
            });
            added = true;
        }
        if added {
            seen_files.insert(file_name.to_string());
        }
    }
    Ok(())
}

fn string_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    batch
        .column_by_name(name)
        .ok_or_else(|| anyhow!("missing column {name}"))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow!("column {name} is not string"))
}

fn bool_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a BooleanArray> {
    batch
        .column_by_name(name)
        .ok_or_else(|| anyhow!("missing column {name}"))?
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| anyhow!("column {name} is not bool"))
}

fn list_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ListArray> {
    batch
        .column_by_name(name)
        .ok_or_else(|| anyhow!("missing column {name}"))?
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| anyhow!("column {name} is not list"))
}

fn bool_value(array: &BooleanArray, row: usize) -> Option<bool> {
    if array.is_null(row) {
        None
    } else {
        Some(array.value(row))
    }
}

fn float_value(array: &dyn Array, row: usize) -> Option<f32> {
    if let Some(arr) = array.as_any().downcast_ref::<Float32Array>() {
        if arr.is_null(row) {
            None
        } else {
            Some(arr.value(row))
        }
    } else if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
        if arr.is_null(row) {
            None
        } else {
            Some(arr.value(row) as f32)
        }
    } else {
        None
    }
}

fn valid_sift_link(info: &SiftInfo) -> bool {
    info.checked == Some(true)
        && info.match_file.is_some()
        && info.inliers.unwrap_or(0) >= SIFT_MIN_INLIERS
        && info.inlier_ratio.unwrap_or(0.0) >= SIFT_MIN_INLIER_RATIO
        && info.score.unwrap_or(0.0) >= SIFT_MIN_SCORE
}

impl ClipTextEncoder {
    fn new(onnx_path: &Path, tokenizer_path: &Path, context_len: usize) -> Result<Self> {
        let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|err| {
            anyhow!(
                "failed to load tokenizer {}: {err}",
                tokenizer_path.display()
            )
        })?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: context_len,
                ..Default::default()
            }))
            .map_err(|err| anyhow!("failed to configure tokenizer truncation: {err}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(context_len),
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "<pad>".to_string(),
            ..Default::default()
        }));
        let session = Session::builder()
            .map_err(|err| anyhow!("failed to create ONNX Runtime session builder: {err}"))?
            .with_intra_threads(num_cpus::get().min(16))
            .map_err(|err| anyhow!("failed to configure ONNX Runtime threads: {err}"))?
            .commit_from_file(onnx_path)
            .map_err(|err| anyhow!("failed to load ONNX model {}: {err}", onnx_path.display()))?;
        Ok(Self {
            tokenizer,
            session,
            context_len,
        })
    }

    fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|err| anyhow!("tokenization failed: {err}"))?;
        let mut ids: Vec<i64> = encoding.get_ids().iter().map(|&id| i64::from(id)).collect();
        ids.truncate(self.context_len);
        ids.resize(self.context_len, 0);

        let input = Tensor::from_array(([1usize, self.context_len], ids.into_boxed_slice()))
            .map_err(|err| anyhow!("failed to create ONNX input tensor: {err}"))?;
        let outputs = self
            .session
            .run(ort::inputs! { "input_ids" => input })
            .map_err(|err| anyhow!("ONNX text encoder inference failed: {err}"))?;
        let (_, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|err| anyhow!("failed to extract ONNX text embedding: {err}"))?;
        let mut vector = data.to_vec();
        normalize_in_place(&mut vector);
        Ok(vector)
    }
}

fn normalize_in_place(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector {
            *value /= norm;
        }
    }
}

fn search_index(
    index: &ClipIndex,
    query: &[f32],
    limit: usize,
    video_only: bool,
) -> Vec<SearchResult> {
    let merged = index
        .entries
        .par_chunks(4096)
        .map(|chunk| {
            let mut local: HashMap<String, (f32, bool, f32)> = HashMap::new();
            for entry in chunk {
                if video_only && !entry.is_video {
                    continue;
                }
                let score = dot(query, &entry.vector);
                local
                    .entry(entry.file_name.to_string())
                    .and_modify(|best| {
                        if score > best.0 {
                            *best = (score, entry.is_video, entry.timestamp_sec);
                        }
                    })
                    .or_insert((score, entry.is_video, entry.timestamp_sec));
            }
            local
        })
        .reduce(HashMap::new, |mut acc, local| {
            for (file_name, candidate) in local {
                acc.entry(file_name)
                    .and_modify(|best| {
                        if candidate.0 > best.0 {
                            *best = candidate;
                        }
                    })
                    .or_insert(candidate);
            }
            acc
        });

    let mut rows: Vec<_> = merged
        .into_iter()
        .map(
            |(file_name, (score, is_video, timestamp_sec))| SearchResult {
                rank: 0,
                score,
                file_name,
                is_video,
                timestamp_sec,
                media_path: None,
            },
        )
        .collect();

    rows.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    rows.truncate(limit);
    for (idx, row) in rows.iter_mut().enumerate() {
        row.rank = idx + 1;
    }
    rows
}

fn search_ocr_index(
    index: &OcrIndex,
    query: &str,
    limit: usize,
    video_only: bool,
) -> Vec<SearchResult> {
    let query_lower = query.trim().to_lowercase();
    if query_lower.is_empty() {
        return Vec::new();
    }
    let terms: Vec<&str> = query_lower
        .split_whitespace()
        .filter(|term| !term.is_empty())
        .collect();
    let term_den = terms.len().max(1) as f32;

    let merged = index
        .entries
        .par_chunks(4096)
        .map(|chunk| {
            let mut local: HashMap<String, (f32, bool, f32)> = HashMap::new();
            for entry in chunk {
                if video_only && !entry.is_video {
                    continue;
                }
                let phrase_hit = entry.text_lower.contains(query_lower.as_str());
                let term_hits = terms
                    .iter()
                    .filter(|term| entry.text_lower.contains(**term))
                    .count() as f32;
                if !phrase_hit && term_hits <= 0.0 {
                    continue;
                }
                let term_score = term_hits / term_den;
                let score = if phrase_hit { 2.0 + term_score } else { term_score };
                local
                    .entry(entry.file_name.to_string())
                    .and_modify(|best| {
                        if score > best.0 {
                            *best = (score, entry.is_video, entry.timestamp_sec);
                        }
                    })
                    .or_insert((score, entry.is_video, entry.timestamp_sec));
            }
            local
        })
        .reduce(HashMap::new, |mut acc, local| {
            for (file_name, candidate) in local {
                acc.entry(file_name)
                    .and_modify(|best| {
                        if candidate.0 > best.0 {
                            *best = candidate;
                        }
                    })
                    .or_insert(candidate);
            }
            acc
        });

    let mut rows: Vec<_> = merged
        .into_iter()
        .map(
            |(file_name, (score, is_video, timestamp_sec))| SearchResult {
                rank: 0,
                score,
                file_name,
                is_video,
                timestamp_sec,
                media_path: None,
            },
        )
        .collect();

    rows.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    rows.truncate(limit);
    for (idx, row) in rows.iter_mut().enumerate() {
        row.rank = idx + 1;
    }
    rows
}

fn search_face_index(
    index: &FaceIndex,
    query_vectors: &[Vec<f32>],
    limit: usize,
    min_score: f32,
) -> Vec<SearchResult> {
    if query_vectors.is_empty() {
        return Vec::new();
    }
    let merged = index
        .entries
        .par_chunks(4096)
        .map(|chunk| {
            let mut local: HashMap<String, (f32, bool, f32)> = HashMap::new();
            for entry in chunk {
                let score = query_vectors
                    .iter()
                    .map(|query| dot(query, &entry.vector))
                    .fold(f32::NEG_INFINITY, f32::max);
                if score < min_score {
                    continue;
                }
                local
                    .entry(entry.file_name.to_string())
                    .and_modify(|best| {
                        if score > best.0 {
                            *best = (score, entry.is_video, entry.timestamp_sec);
                        }
                    })
                    .or_insert((score, entry.is_video, entry.timestamp_sec));
            }
            local
        })
        .reduce(HashMap::new, |mut acc, local| {
            for (file_name, candidate) in local {
                acc.entry(file_name)
                    .and_modify(|best| {
                        if candidate.0 > best.0 {
                            *best = candidate;
                        }
                    })
                    .or_insert(candidate);
            }
            acc
        });

    let mut rows: Vec<_> = merged
        .into_iter()
        .map(
            |(file_name, (score, is_video, timestamp_sec))| SearchResult {
                rank: 0,
                score,
                file_name,
                is_video,
                timestamp_sec,
                media_path: None,
            },
        )
        .collect();
    rows.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    rows.truncate(limit);
    for (idx, row) in rows.iter_mut().enumerate() {
        row.rank = idx + 1;
    }
    rows
}

fn collapse_sift_grouped_results(
    mut rows: Vec<SearchResult>,
    sift_root_by_file: &HashMap<String, String>,
    limit: usize,
) -> Vec<SearchResult> {
    let mut rows_by_group: HashMap<String, Vec<SearchResult>> = HashMap::new();
    for row in rows.drain(..) {
        let group_key = if row.is_video {
            row.file_name.clone()
        } else {
            sift_root_by_file
                .get(row.file_name.as_str())
                .cloned()
                .unwrap_or_else(|| row.file_name.clone())
        };
        rows_by_group.entry(group_key).or_default().push(row);
    }
    let mut collapsed: Vec<SearchResult> = Vec::with_capacity(rows_by_group.len());
    for (_group_key, group_rows) in rows_by_group {
        let mut best = group_rows[0].clone();
        for row in &group_rows[1..] {
            if row.score > best.score {
                best = row.clone();
            }
        }
        // Keep the highest-scoring member as the displayed tile for this group.
        collapsed.push(best);
    }
    collapsed.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    collapsed.truncate(limit);
    for (idx, row) in collapsed.iter_mut().enumerate() {
        row.rank = idx + 1;
    }
    collapsed
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn resolve_media_path(
    roots: &HashMap<String, PathBuf>,
    db_dir: &Path,
    file_name: &str,
    timestamp_sec: f32,
) -> Result<PathBuf> {
    let (collection, rel) = file_name
        .split_once('/')
        .ok_or_else(|| anyhow!("file_name does not contain collection id"))?;
    let root = roots
        .get(collection)
        .ok_or_else(|| anyhow!("no --collection-root supplied for collection {collection}"))?;
    let rel_path = Path::new(rel);
    let source = root.join(rel_path);
    if is_video_path(&source) {
        if let Some(still) = resolve_video_still(root, db_dir, rel_path, timestamp_sec)? {
            return Ok(still);
        }
    }
    Ok(source)
}

fn resolve_source_path(roots: &HashMap<String, PathBuf>, file_name: &str) -> Result<PathBuf> {
    let (collection, rel) = file_name
        .split_once('/')
        .ok_or_else(|| anyhow!("file_name does not contain collection id"))?;
    let root = roots
        .get(collection)
        .ok_or_else(|| anyhow!("no --collection-root supplied for collection {collection}"))?;
    Ok(root.join(Path::new(rel)))
}

fn display_path_label(roots: &HashMap<String, PathBuf>, file_name: &str) -> String {
    resolve_source_path(roots, file_name)
        .map(|path| path.display().to_string())
        .unwrap_or_else(|_| file_name.to_string())
}

fn is_video_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|x| x.to_str())
            .map(|x| x.to_ascii_lowercase())
            .as_deref(),
        Some("mp4" | "mov" | "avi" | "mkv" | "webm" | "m4v" | "wmv" | "mpg" | "mpeg")
    )
}

fn resolve_video_still(
    root: &Path,
    db_dir: &Path,
    rel_path: &Path,
    timestamp_sec: f32,
) -> Result<Option<PathBuf>> {
    let scene_dir_name = format!(
        "{}-video",
        root.file_name().and_then(|x| x.to_str()).unwrap_or("video")
    );
    let mut scene_roots = Vec::with_capacity(2);
    scene_roots.push(db_dir.join(&scene_dir_name));
    if let Some(parent) = root.parent() {
        scene_roots.push(parent.join(&scene_dir_name));
    }
    let parent = rel_path.parent().unwrap_or_else(|| Path::new(""));
    let stem = rel_path
        .file_stem()
        .and_then(|x| x.to_str())
        .unwrap_or("video");
    let suffix = rel_path
        .extension()
        .and_then(|x| x.to_str())
        .map(|x| format!("_{}", x.to_ascii_lowercase()))
        .unwrap_or_default();
    for scene_root in scene_roots {
        let video_dir = scene_root.join(parent).join(format!("{stem}{suffix}"));
        let manifest_path = video_dir.join("manifest.json");
        let data = match std::fs::read_to_string(&manifest_path) {
            Ok(data) => data,
            Err(_) => continue,
        };
        let manifest: Value = serde_json::from_str(&data)?;
        let frames = manifest
            .get("frames")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("manifest has no frames"))?;
        let mut best: Option<(f32, PathBuf)> = None;
        for frame in frames {
            let ts = frame
                .get("timestamp_sec")
                .and_then(Value::as_f64)
                .unwrap_or(0.0) as f32;
            let image_file = match frame.get("image_file").and_then(Value::as_str) {
                Some(value) => value,
                None => continue,
            };
            let distance = (ts - timestamp_sec).abs();
            let path = video_dir.join(image_file);
            if best
                .as_ref()
                .map(|(best_dist, _)| distance < *best_dist)
                .unwrap_or(true)
            {
                best = Some((distance, path));
            }
        }
        if let Some((_, path)) = best {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

fn open_in_iris(path: &Path) -> Result<()> {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    Command::new("iris")
        .arg(&canonical)
        .spawn()
        .with_context(|| format!("failed to launch iris for {}", canonical.display()))?;
    Ok(())
}

fn copy_image_to_clipboard(ctx: &egui::Context, path: &Path) -> Result<()> {
    let dynamic = image::open(path)
        .with_context(|| format!("failed to decode image for clipboard: {}", path.display()))?;
    let rgba = dynamic.to_rgba8();
    let (width, height) = rgba.dimensions();
    let color = egui::ColorImage::from_rgba_unmultiplied(
        [width as usize, height as usize],
        rgba.as_raw(),
    );
    ctx.copy_image(color);
    Ok(())
}

fn open_in_mpv(path: &Path, timestamp_sec: f32) -> Result<()> {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    Command::new("mpv")
        .arg(format!("--start={:.3}", timestamp_sec.max(0.0)))
        .arg(&canonical)
        .spawn()
        .with_context(|| format!("failed to launch mpv for {}", canonical.display()))?;
    Ok(())
}

fn open_parent_folder(path: &Path) -> Result<()> {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    Command::new("dolphin")
        .arg("--select")
        .arg(&canonical)
        .spawn()
        .with_context(|| format!("failed to launch dolphin for {}", canonical.display()))?;
    Ok(())
}

fn compute_sift_summary(path_a: &Path, path_b: &Path) -> Result<String> {
    let output = Command::new("uv")
        .args([
            "run",
            "python",
            "tools/sift_similarity.py",
            path_a.to_string_lossy().as_ref(),
            path_b.to_string_lossy().as_ref(),
        ])
        .output()
        .context("failed to run sift_similarity.py")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("sift script failed: {}", stderr.trim());
    }
    let payload: Value = serde_json::from_slice(&output.stdout).context("invalid sift json")?;
    if let Some(err) = payload.get("error").and_then(Value::as_str) {
        bail!("{err}");
    }
    let keypoints_a = payload
        .get("keypoints_a")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let keypoints_b = payload
        .get("keypoints_b")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let good = payload
        .get("good_matches")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let inliers = payload
        .get("inlier_matches")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let inlier_ratio = payload
        .get("inlier_ratio")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let score = payload.get("score").and_then(Value::as_f64).unwrap_or(0.0);
    Ok(format!(
        "selected pair SIFT: score {:.4} | inliers {} / good {} ({:.2}%) | kpA {} kpB {}",
        score,
        inliers,
        good,
        inlier_ratio * 100.0,
        keypoints_a,
        keypoints_b
    ))
}

fn run_sift_result_repair(
    db_dir: &Path,
    table_name: &str,
    roots: &HashMap<String, PathBuf>,
    file_names: &[String],
    thresholds: SiftRepairThresholds,
    fast_pair: bool,
) -> Result<Value> {
    let repo_root = locate_repo_root()?;
    let script_path = repo_root.join("tools/repair_sift_results.py");
    let payload = serde_json::to_string(file_names).context("failed to serialize result files")?;
    let temp_path = std::env::temp_dir().join(format!(
        "clip_viewer_sift_repair_{}_{}.json",
        std::process::id(),
        chrono_like_millis()
    ));
    fs::write(&temp_path, payload)
        .with_context(|| format!("failed to write {}", temp_path.display()))?;

    let mut command = Command::new("uv");
    command
        .current_dir(&repo_root)
        .arg("run")
        .arg("python")
        .arg(&script_path)
        .arg("--db-dir")
        .arg(db_dir)
        .arg("--table")
        .arg(table_name)
        .arg("--files-json")
        .arg(&temp_path)
        .arg("--min-inliers")
        .arg(thresholds.min_inliers.to_string())
        .arg("--min-inlier-ratio")
        .arg(format!("{:.4}", thresholds.min_inlier_ratio))
        .arg("--min-score")
        .arg(format!("{:.4}", thresholds.min_score));
    if fast_pair {
        command.arg("--fast-pair");
    }
    let mut root_keys: Vec<_> = roots.keys().cloned().collect();
    root_keys.sort_unstable();
    for key in root_keys {
        if let Some(path) = roots.get(key.as_str()) {
            command
                .arg("--collection-root")
                .arg(format!("{}={}", key, path.display()));
        }
    }

    let output = command
        .output()
        .context("failed to run repair_sift_results.py")?;
    let _ = fs::remove_file(&temp_path);
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("repair failed: {}", stderr.trim());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let last_json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .ok_or_else(|| anyhow!("repair produced no JSON summary"))?;
    serde_json::from_str(last_json).context("invalid repair JSON summary")
}

fn run_face_rerun(
    db_dir: &Path,
    table_name: &str,
    roots: &HashMap<String, PathBuf>,
    file_name: &str,
) -> Result<Value> {
    let repo_root = locate_repo_root()?;
    let script_path = repo_root.join("tools/rerun_face_for_file.py");
    let mut command = Command::new("uv");
    command
        .current_dir(&repo_root)
        .arg("run")
        .arg("python")
        .arg(&script_path)
        .arg("--db-dir")
        .arg(db_dir)
        .arg("--table")
        .arg(table_name)
        .arg("--file-name")
        .arg(file_name)
        .arg("--det-threshold")
        .arg("0.25");
    let mut root_keys: Vec<_> = roots.keys().cloned().collect();
    root_keys.sort_unstable();
    for key in root_keys {
        if let Some(path) = roots.get(key.as_str()) {
            command
                .arg("--collection-root")
                .arg(format!("{}={}", key, path.display()));
        }
    }

    let output = command
        .output()
        .context("failed to run rerun_face_for_file.py")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("face rerun failed: {}", stderr.trim());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let last_json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .ok_or_else(|| anyhow!("face rerun produced no JSON summary"))?;
    serde_json::from_str(last_json).context("invalid face rerun JSON summary")
}

fn locate_repo_root() -> Result<PathBuf> {
    let current = std::env::current_dir().context("failed to read current directory")?;
    if current.join("tools/repair_sift_results.py").exists() {
        return Ok(current);
    }
    let exe = std::env::current_exe().context("failed to locate current executable")?;
    for ancestor in exe.ancestors() {
        if ancestor.join("tools/repair_sift_results.py").exists() {
            return Ok(ancestor.to_path_buf());
        }
    }
    bail!("could not locate tools/repair_sift_results.py")
}

fn chrono_like_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn draw_similarity_badge(ui: &egui::Ui, rect: egui::Rect, count: usize) {
    let label = format!("{count}");
    let badge_w = 12.0 + (label.len() as f32 * 8.5);
    let badge_h = 20.0;
    let badge_rect = egui::Rect::from_min_size(
        egui::pos2(rect.right() - badge_w - 6.0, rect.top() + 6.0),
        egui::vec2(badge_w, badge_h),
    );
    ui.painter().rect_filled(
        badge_rect,
        6.0,
        egui::Color32::from_rgba_premultiplied(10, 10, 10, 210),
    );
    ui.painter().text(
        badge_rect.center(),
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::proportional(13.0),
        egui::Color32::WHITE,
    );
}

impl ViewerApp {
    fn active_results(&self) -> &[SearchResult] {
        self.tabs
            .get(self.active_tab)
            .map(|tab| tab.results.as_slice())
            .unwrap_or(&[])
    }

    fn set_active_results(&mut self, title: String, results: Vec<SearchResult>) {
        if self.tabs.is_empty() {
            self.tabs.push(ResultTab { title, results });
            self.active_tab = 0;
            return;
        }
        let idx = self.active_tab.min(self.tabs.len() - 1);
        self.tabs[idx] = ResultTab { title, results };
        self.active_tab = idx;
    }

    fn push_results_tab(&mut self, title: String, results: Vec<SearchResult>) {
        self.tabs.push(ResultTab { title, results });
        self.active_tab = self.tabs.len() - 1;
    }

    fn reload_sift_groups(&mut self) -> Result<()> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("failed to create reload runtime")?;
        let sift_info = runtime
            .block_on(load_sift_info_map(&self.db_dir, &self.table_name))
            .context("failed to reload SIFT info map")?;
        let (roots, members) = runtime
            .block_on(load_sift_groups(&self.db_dir, &self.table_name))
            .context("failed to reload SIFT groups")?;
        self.sift_info_by_file = sift_info;
        self.sift_root_by_file = roots;
        self.sift_members_by_root = members;
        Ok(())
    }

    fn reload_face_index(&mut self) -> Result<()> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("failed to create face reload runtime")?;
        self.face_index = Arc::new(
            runtime
                .block_on(load_face_index(&self.db_dir, &self.table_name))
                .context("failed to reload face index")?,
        );
        Ok(())
    }

    fn start_face_rerun(&mut self, file_name: String) {
        if self.face_rerun_running {
            self.status = "Face rerun is already running.".to_string();
            return;
        }
        self.status = format!("Rerunning face detection at low threshold for {file_name}...");
        let db_dir = self.db_dir.clone();
        let table_name = self.table_name.clone();
        let roots = (*self.roots).clone();
        let (tx, rx) = mpsc::channel();
        self.face_rerun_rx = Some(rx);
        self.face_rerun_running = true;
        std::thread::spawn(move || {
            let result = run_face_rerun(&db_dir, &table_name, &roots, &file_name)
                .map_err(|err| err.to_string());
            let _ = tx.send(result);
        });
    }

    fn repair_current_results_sift(&mut self) {
        if self.repair_running {
            self.status = "SIFT repair is already running.".to_string();
            return;
        }
        let mut seen = HashSet::new();
        let mut file_names: Vec<String> = Vec::new();
        for row in self.active_results().iter().filter(|row| !row.is_video) {
            if seen.insert(row.file_name.clone()) {
                file_names.push(row.file_name.clone());
            }
        }
        if let Some(selected) = self.selected_master.as_ref() {
            if seen.insert(selected.clone()) {
                file_names.push(selected.clone());
            }
        }
        self.start_sift_repair(
            file_names,
            "visible image results",
            SiftRepairThresholds {
                min_inliers: SIFT_MIN_INLIERS,
                min_inlier_ratio: SIFT_MIN_INLIER_RATIO,
                min_score: SIFT_MIN_SCORE,
            },
        );
    }

    fn repair_selected_images_sift(&mut self) {
        if self.repair_running {
            self.status = "SIFT repair is already running.".to_string();
            return;
        }
        let file_names: Vec<String> = self.selected_images.iter().cloned().collect();
        let fast_pair = file_names.len() == 2;
        self.start_sift_repair_with_mode(
            file_names,
            "selected images",
            SiftRepairThresholds {
                min_inliers: SIFT_MIN_INLIERS,
                min_inlier_ratio: SIFT_MIN_INLIER_RATIO,
                min_score: SIFT_MIN_SCORE,
            },
            fast_pair,
        );
    }

    fn force_repair_selected_images_sift(&mut self) {
        if self.repair_running {
            self.status = "SIFT repair is already running.".to_string();
            return;
        }
        let file_names: Vec<String> = self.selected_images.iter().cloned().collect();
        let fast_pair = file_names.len() == 2;
        self.start_sift_repair_with_mode(
            file_names,
            "selected images (force)",
            SiftRepairThresholds {
                min_inliers: 0,
                min_inlier_ratio: 0.0,
                min_score: 0.0,
            },
            fast_pair,
        );
    }

    fn toggle_selected_image(&mut self, file_name: &str, is_video: bool) {
        if is_video {
            self.status = "Selection mode only supports images.".to_string();
            return;
        }
        if self.selected_images.contains(file_name) {
            self.selected_images.remove(file_name);
        } else {
            self.selected_images.insert(file_name.to_string());
        }
        self.recompute_selected_pair_sift_diag();
    }

    fn recompute_selected_pair_sift_diag(&mut self) {
        if self.selected_images.len() != 2 {
            self.selected_pair_sift_diag = None;
            self.selected_pair_key = None;
            return;
        }
        let mut pair: Vec<String> = self.selected_images.iter().cloned().collect();
        pair.sort_unstable();
        let key = (pair[0].clone(), pair[1].clone());
        if self.selected_pair_key.as_ref() == Some(&key) {
            return;
        }
        let path_a = match resolve_source_path(&self.roots, &pair[0]) {
            Ok(path) => path,
            Err(err) => {
                self.selected_pair_key = Some(key);
                self.selected_pair_sift_diag = Some(format!("selected pair SIFT error: {err}"));
                return;
            }
        };
        let path_b = match resolve_source_path(&self.roots, &pair[1]) {
            Ok(path) => path,
            Err(err) => {
                self.selected_pair_key = Some(key);
                self.selected_pair_sift_diag = Some(format!("selected pair SIFT error: {err}"));
                return;
            }
        };
        self.selected_pair_sift_diag = Some(match compute_sift_summary(&path_a, &path_b) {
            Ok(summary) => summary,
            Err(err) => format!("selected pair SIFT error: {err}"),
        });
        self.selected_pair_key = Some(key);
    }

    fn start_sift_repair(
        &mut self,
        file_names: Vec<String>,
        label: &str,
        thresholds: SiftRepairThresholds,
    ) {
        self.start_sift_repair_with_mode(file_names, label, thresholds, false);
    }

    fn start_sift_repair_with_mode(
        &mut self,
        file_names: Vec<String>,
        label: &str,
        thresholds: SiftRepairThresholds,
        fast_pair: bool,
    ) {
        if file_names.len() < 2 {
            self.status = "Need at least two image results to repair SIFT grouping.".to_string();
            return;
        }
        self.status = format!(
            "Repairing SIFT grouping for {} {} (inliers >= {}, ratio >= {:.0}%, score >= {:.2})...",
            file_names.len(),
            label,
            thresholds.min_inliers,
            thresholds.min_inlier_ratio * 100.0,
            thresholds.min_score
        );
        let db_dir = self.db_dir.clone();
        let table_name = self.table_name.clone();
        let roots = (*self.roots).clone();
        let (tx, rx) = mpsc::channel();
        self.repair_rx = Some(rx);
        self.repair_running = true;
        std::thread::spawn(move || {
            let result = run_sift_result_repair(
                &db_dir,
                &table_name,
                &roots,
                &file_names,
                thresholds,
                fast_pair,
            )
            .map_err(|err| err.to_string());
            let _ = tx.send(result);
        });
    }

    fn poll_sift_repair(&mut self) {
        let Some(rx) = self.repair_rx.take() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(summary)) => {
                self.repair_running = false;
                let reload_result = self.reload_sift_groups();
                if let Err(err) = reload_result {
                    self.status = format!("repair done but reload failed: {err}");
                    self.repair_rx = None;
                    return;
                }
                let updated = summary.get("updated").and_then(Value::as_u64).unwrap_or(0);
                let linked = summary
                    .get("linked_images")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                let pairs = summary.get("pairs").and_then(Value::as_u64).unwrap_or(0);
                let accepted = summary
                    .get("accepted_pairs")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                self.status = format!(
                    "SIFT repair updated {updated} rows, linked {linked} images ({accepted}/{pairs} accepted pairs)"
                );
            }
            Ok(Err(err)) => {
                self.repair_running = false;
                self.status = format!("SIFT repair failed: {err}");
            }
            Err(err) => match err {
                mpsc::TryRecvError::Empty => {
                    self.repair_rx = Some(rx);
                }
                mpsc::TryRecvError::Disconnected => {
                    self.repair_running = false;
                    self.status = "SIFT repair worker disconnected.".to_string();
                }
            },
        }
    }

    fn poll_face_rerun(&mut self) {
        let Some(rx) = self.face_rerun_rx.take() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(summary)) => {
                self.face_rerun_running = false;
                let reload_result = self.reload_face_index();
                if let Err(err) = reload_result {
                    self.status = format!("face rerun done but reload failed: {err}");
                    self.face_rerun_rx = None;
                    return;
                }
                let face_count = summary
                    .get("face_count")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                let threshold = summary
                    .get("det_threshold")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.25);
                self.status = format!(
                    "Face rerun complete: {face_count} face vector(s) at threshold {threshold:.2}"
                );
            }
            Ok(Err(err)) => {
                self.face_rerun_running = false;
                self.status = format!("Face rerun failed: {err}");
            }
            Err(err) => match err {
                mpsc::TryRecvError::Empty => {
                    self.face_rerun_rx = Some(rx);
                }
                mpsc::TryRecvError::Disconnected => {
                    self.face_rerun_running = false;
                    self.status = "Face rerun worker disconnected.".to_string();
                }
            },
        }
    }

    fn grouped_master_for(&self, file_name: &str, is_video: bool) -> String {
        if is_video {
            return file_name.to_string();
        }
        self.sift_root_by_file
            .get(file_name)
            .cloned()
            .unwrap_or_else(|| file_name.to_string())
    }

    fn similar_count_for(&self, file_name: &str, is_video: bool) -> usize {
        if is_video {
            let dedupe_count = self
                .similar_by_master
                .get(file_name)
                .map(|items| items.len())
                .unwrap_or(0);
            return 1 + dedupe_count;
        }
        let root = self.grouped_master_for(file_name, false);
        let members: Vec<String> = self
            .sift_members_by_root
            .get(root.as_str())
            .cloned()
            .unwrap_or_else(|| vec![file_name.to_string()]);
        let master_count = members.len();
        let phash_children_count = members
            .iter()
            .map(|member| {
                self.similar_by_master
                    .get(member.as_str())
                    .map(|items| items.len())
                    .unwrap_or(0)
            })
            .sum::<usize>();
        master_count + phash_children_count
    }

    fn sift_info_line(&self, file_name: &str) -> String {
        let Some(info) = self.sift_info_by_file.get(file_name) else {
            return "SIFT: n/a".to_string();
        };
        let target = info
            .match_file
            .as_deref()
            .and_then(|value| Path::new(value).file_name().and_then(|x| x.to_str()))
            .unwrap_or("-");
        let inliers = info
            .inliers
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".to_string());
        let good = info
            .good_matches
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".to_string());
        let score = info
            .score
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "-".to_string());
        let ratio = info
            .inlier_ratio
            .map(|v| format!("{:.2}%", v * 100.0))
            .unwrap_or_else(|| "-".to_string());
        let checked = info.checked.unwrap_or(false);
        format!(
            "SIFT checked={} -> {} | inliers {} / {} | ratio {} | score {}",
            checked, target, inliers, good, ratio, score
        )
    }

    fn search_now(&mut self) {
        match self.search_mode {
            SearchMode::Clip => self.search_clip_now(),
            SearchMode::Ocr => self.search_ocr_now(),
        }
    }

    fn search_clip_now(&mut self) {
        let q = self.query.trim();
        if q.is_empty() {
            self.status = "Enter a phrase first.".to_string();
            return;
        }

        let started = Instant::now();
        let query_vector = match self.encoder.embed(q) {
            Ok(vec) => vec,
            Err(err) => {
                self.status = format!("embed failed: {err}");
                return;
            }
        };

        if query_vector.len() != self.index.dim {
            self.status = format!(
                "query dim {} does not match index dim {}",
                query_vector.len(),
                self.index.dim
            );
            return;
        }

        let pre_limit = (self.limit.saturating_mul(6)).max(self.limit);
        let mut results = search_index(&self.index, &query_vector, pre_limit, self.video_only);
        if !self.video_only {
            results = collapse_sift_grouped_results(results, &self.sift_root_by_file, self.limit);
        } else {
            results.truncate(self.limit);
        }
        for row in &mut results {
            row.media_path =
                resolve_media_path(&self.roots, &self.db_dir, &row.file_name, row.timestamp_sec).ok();
        }

        let took = started.elapsed().as_millis();
        self.loaded = 0;
        self.cache.clear();
        self.status = format!(
            "{} results in {} ms across {} vectors / {} master files",
            results.len(),
            took,
            self.searched_vectors,
            self.master_files
        );
        let title = if self.video_only {
            format!("Videos: {q}")
        } else {
            format!("CLIP: {q}")
        };
        self.set_active_results(title, results);
    }

    fn search_ocr_now(&mut self) {
        let q = self.query.trim();
        if q.is_empty() {
            self.status = "Enter a phrase first.".to_string();
            return;
        }
        let started = Instant::now();
        let pre_limit = (self.limit.saturating_mul(6)).max(self.limit);
        let mut results = search_ocr_index(&self.ocr_index, q, pre_limit, self.video_only);
        if !self.video_only {
            results = collapse_sift_grouped_results(results, &self.sift_root_by_file, self.limit);
        } else {
            results.truncate(self.limit);
        }
        for row in &mut results {
            row.media_path =
                resolve_media_path(&self.roots, &self.db_dir, &row.file_name, row.timestamp_sec).ok();
        }

        let took = started.elapsed().as_millis();
        self.loaded = 0;
        self.cache.clear();
        self.status = format!(
            "{} OCR results in {} ms across {} OCR entries / {} files",
            results.len(),
            took,
            self.ocr_index.entries.len(),
            self.ocr_index.file_count
        );
        let title = if self.video_only {
            format!("OCR Videos: {q}")
        } else {
            format!("OCR: {q}")
        };
        self.set_active_results(title, results);
    }

    fn clip_vector_for_result(&self, row: &SearchResult) -> Option<Vec<f32>> {
        let mut best: Option<(&ClipEntry, f32)> = None;
        for entry in &self.index.entries {
            if entry.file_name.as_ref() != row.file_name.as_str() {
                continue;
            }
            let dt = (entry.timestamp_sec - row.timestamp_sec).abs();
            match best {
                Some((_current, best_dt)) if dt >= best_dt => {}
                _ => best = Some((entry, dt)),
            }
        }
        best.map(|(entry, _)| entry.vector.clone())
    }

    fn show_most_similar_clip(&mut self, row: &SearchResult) {
        let Some(query_vector) = self.clip_vector_for_result(row) else {
            self.status = format!("no CLIP vector found for {}", row.file_name);
            return;
        };
        if query_vector.len() != self.index.dim {
            self.status = format!(
                "source vector dim {} does not match index dim {}",
                query_vector.len(),
                self.index.dim
            );
            return;
        }
        let started = Instant::now();
        let pre_limit = (self.limit.saturating_mul(12)).max(self.limit + 32);
        // Always search across both image and video vectors so an image query can surface
        // matching video stills (and vice versa).
        let mut results = search_index(&self.index, &query_vector, pre_limit, false);
        results.retain(|candidate| candidate.file_name != row.file_name);
        results = collapse_sift_grouped_results(results, &self.sift_root_by_file, self.limit);
        for candidate in &mut results {
            candidate.media_path =
                resolve_media_path(&self.roots, &self.db_dir, &candidate.file_name, candidate.timestamp_sec).ok();
        }
        let took = started.elapsed().as_millis();
        self.loaded = 0;
        self.cache.clear();
        self.status = format!(
            "{} CLIP-similar results in {} ms for {}",
            results.len(),
            took,
            row.file_name
        );
        let title = short_tab_title("Similar", &row.file_name);
        self.push_results_tab(title, results);
    }

    fn face_vectors_for_file(&self, file_name: &str) -> Vec<Vec<f32>> {
        self.face_index
            .entries
            .iter()
            .filter(|entry| entry.file_name.as_ref() == file_name)
            .map(|entry| entry.vector.clone())
            .collect()
    }

    fn related_files_for_face_seed(&self, file_name: &str) -> Vec<String> {
        let mut related = Vec::new();
        let mut seen = HashSet::new();
        if seen.insert(file_name.to_string()) {
            related.push(file_name.to_string());
        }

        let root = self.grouped_master_for(file_name, false);
        if let Some(members) = self.sift_members_by_root.get(root.as_str()) {
            for member in members {
                if seen.insert(member.clone()) {
                    related.push(member.clone());
                }
                if let Some(children) = self.similar_by_master.get(member.as_str()) {
                    for child in children {
                        if !child.is_video && seen.insert(child.file_name.clone()) {
                            related.push(child.file_name.clone());
                        }
                    }
                }
            }
        } else if let Some(children) = self.similar_by_master.get(file_name) {
            for child in children {
                if !child.is_video && seen.insert(child.file_name.clone()) {
                    related.push(child.file_name.clone());
                }
            }
        }

        related
    }

    fn show_more_of_this_person(&mut self, file_name: &str) {
        let related_files = self.related_files_for_face_seed(file_name);
        let mut query_faces = Vec::new();
        for related in &related_files {
            query_faces.extend(self.face_vectors_for_file(related));
        }
        let title = short_tab_title("Person", file_name);
        if query_faces.is_empty() {
            self.status = format!(
                "No stored face vectors for {file_name} or {} related file(s)",
                related_files.len().saturating_sub(1)
            );
            self.push_results_tab(title, Vec::new());
            return;
        }
        let started = Instant::now();
        let mut results =
            search_face_index(&self.face_index, &query_faces, 500, FACE_MATCH_MIN_SCORE);
        results = collapse_sift_grouped_results(results, &self.sift_root_by_file, 500);
        for row in &mut results {
            row.media_path =
                resolve_media_path(&self.roots, &self.db_dir, &row.file_name, row.timestamp_sec).ok();
        }
        let took = started.elapsed().as_millis();
        self.loaded = 0;
        self.cache.clear();
        self.status = format!(
            "{} person results in {} ms using {} query face vector(s) from {} related file(s), threshold {:.2}",
            results.len(),
            took,
            query_faces.len(),
            related_files.len(),
            FACE_MATCH_MIN_SCORE
        );
        self.push_results_tab(title, results);
    }

    fn load_texture_if_needed(
        &mut self,
        ctx: &egui::Context,
        path: &Path,
    ) -> Option<egui::TextureHandle> {
        if let Some(existing) = self.cache.get(path) {
            return Some(existing.clone());
        }
        let image = image::open(path).ok()?;
        let image = if image.width() > 1024 || image.height() > 1024 {
            image.thumbnail(1024, 1024)
        } else {
            image
        };
        let rgba = image.to_rgba8();
        let size = [rgba.width() as usize, rgba.height() as usize];
        let color_image = egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());
        let texture = ctx.load_texture(
            path.to_string_lossy(),
            color_image,
            egui::TextureOptions::LINEAR,
        );
        self.loaded += 1;
        self.cache.insert(path.to_path_buf(), texture.clone());
        Some(texture)
    }

}

impl eframe::App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_sift_repair();
        self.poll_face_rerun();
        if let Some(master_file_name) = self.selected_master.clone() {
            let selected_file_name = self
                .selected_file
                .clone()
                .unwrap_or_else(|| master_file_name.clone());
            egui::SidePanel::right("similar_files_panel")
                .default_width(460.0)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading("Similar Files");
                        if ui.button("Close").clicked() {
                            self.selected_file = None;
                            self.selected_master = None;
                        }
                    });
                    ui.separator();
                    ui.label("Master:");
                    ui.monospace(display_path_label(&self.roots, master_file_name.as_str()));
                    let sift_members = self
                        .sift_members_by_root
                        .get(master_file_name.as_str())
                        .cloned()
                        .unwrap_or_default();
                    let sift_evidence_members: Vec<String> = sift_members
                        .iter()
                        .filter(|member| {
                            self.sift_info_by_file
                                .get(member.as_str())
                                .is_some_and(valid_sift_link)
                        })
                        .cloned()
                        .collect();
                    let mut displayed_sift_members = Vec::new();
                    let mut displayed_seen = HashSet::new();
                    if displayed_seen.insert(selected_file_name.clone()) {
                        displayed_sift_members.push(selected_file_name.clone());
                    }
                    for member in &sift_evidence_members {
                        if displayed_seen.insert(member.clone()) {
                            displayed_sift_members.push(member.clone());
                        }
                    }
                    if !sift_members.is_empty() {
                        ui.label(format!(
                            "SIFT matched files: {}",
                            sift_evidence_members.len()
                        ));
                    }
                    let member_sources: Vec<String> = if sift_members.is_empty() {
                        vec![master_file_name.clone()]
                    } else {
                        sift_members.clone()
                    };
                    let mut combined_similars: Vec<SimilarFile> = Vec::new();
                    let mut combined_seen = HashSet::new();
                    let mut grouped_similars: Vec<(String, Vec<SimilarFile>)> = Vec::new();
                    for member in &member_sources {
                        let mut items = self
                            .similar_by_master
                            .get(member.as_str())
                            .cloned()
                            .unwrap_or_default();
                        items.sort_by(|a, b| {
                            b.similarity_pct
                                .unwrap_or(f32::NEG_INFINITY)
                                .partial_cmp(&a.similarity_pct.unwrap_or(f32::NEG_INFINITY))
                                .unwrap_or(Ordering::Equal)
                        });
                        for item in &items {
                            if combined_seen.insert(item.file_name.clone()) {
                                combined_similars.push(item.clone());
                            }
                        }
                        grouped_similars.push((member.clone(), items));
                    }
                    combined_similars.sort_by(|a, b| {
                        b.similarity_pct
                            .unwrap_or(f32::NEG_INFINITY)
                            .partial_cmp(&a.similarity_pct.unwrap_or(f32::NEG_INFINITY))
                            .unwrap_or(Ordering::Equal)
                    });
                    ui.label(format!(
                        "pHash/VideoHash similar count: {}",
                        combined_similars.len()
                    ));
                    ui.separator();
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        let side_thumb = 110.0_f32;
                        if !displayed_sift_members.is_empty() {
                            ui.label("SIFT-grouped masters:");
                            for member in &displayed_sift_members {
                                let source_path = resolve_source_path(&self.roots, member).ok();
                                let member_is_video =
                                    source_path.as_ref().is_some_and(|path| is_video_path(path));
                                let preview_path =
                                    resolve_media_path(&self.roots, &self.db_dir, member, 0.0).ok();
                                ui.horizontal(|ui| {
                                    if let Some(path) = preview_path.as_ref() {
                                        if let Some(texture) =
                                            self.load_texture_if_needed(ctx, path)
                                        {
                                            ui.add(egui::Image::new(&texture).fit_to_exact_size(
                                                egui::vec2(side_thumb, side_thumb),
                                            ));
                                        } else {
                                            let (rect, _) = ui.allocate_exact_size(
                                                egui::vec2(side_thumb, side_thumb),
                                                egui::Sense::hover(),
                                            );
                                            ui.painter().rect_filled(
                                                rect,
                                                4.0,
                                                egui::Color32::from_gray(40),
                                            );
                                        }
                                    }
                                    ui.vertical(|ui| {
                                        ui.label(if member_is_video { "video" } else { "image" });
                                        if member == &selected_file_name {
                                            ui.label("clicked image");
                                        }
                                        if ui.button("Open").clicked() {
                                            if let Some(path) = source_path.as_ref() {
                                                let open_result = if member_is_video {
                                                    open_in_mpv(path, 0.0)
                                                } else {
                                                    open_in_iris(path)
                                                };
                                                match open_result {
                                                    Ok(()) => {
                                                        let app_name = if member_is_video {
                                                            "mpv"
                                                        } else {
                                                            "iris"
                                                        };
                                                        self.status = format!(
                                                            "opened in {app_name}: {}",
                                                            path.display()
                                                        );
                                                    }
                                                    Err(err) => {
                                                        self.status = format!("open failed: {err}");
                                                    }
                                                }
                                            } else {
                                                self.status =
                                                    "resolve failed for SIFT member".to_string();
                                            }
                                        }
                                        if let Some(path) = source_path.as_ref() {
                                            ui.label(file_resolution_and_size(path));
                                        } else {
                                            ui.label("n/a");
                                        }
                                        ui.label(self.sift_info_line(member));
                                        ui.monospace(display_path_label(&self.roots, member.as_str()));
                                    });
                                });
                                ui.separator();
                            }
                            ui.separator();
                        }
                        if !combined_similars.is_empty() {
                            ui.label("All pHash/VideoHash similars in this SIFT group:");
                            for item in &combined_similars {
                                let source_path =
                                    resolve_source_path(&self.roots, &item.file_name).ok();
                                let preview_path =
                                    resolve_media_path(&self.roots, &self.db_dir, &item.file_name, 0.0).ok();
                                ui.horizontal(|ui| {
                                    if let Some(path) = preview_path.as_ref() {
                                        if let Some(texture) =
                                            self.load_texture_if_needed(ctx, path)
                                        {
                                            ui.add(egui::Image::new(&texture).fit_to_exact_size(
                                                egui::vec2(side_thumb, side_thumb),
                                            ));
                                        } else {
                                            let (rect, _) = ui.allocate_exact_size(
                                                egui::vec2(side_thumb, side_thumb),
                                                egui::Sense::hover(),
                                            );
                                            ui.painter().rect_filled(
                                                rect,
                                                4.0,
                                                egui::Color32::from_gray(40),
                                            );
                                        }
                                    }
                                    ui.vertical(|ui| {
                                        ui.label(if item.is_video { "video" } else { "image" });
                                        let similarity_label = item
                                            .similarity_pct
                                            .map(|v| format!("pHash similarity {v:.2}%"))
                                            .unwrap_or_else(|| "pHash similarity n/a".to_string());
                                        ui.label(similarity_label);
                                        if ui.button("Open").clicked() {
                                            if let Some(path) = source_path.as_ref() {
                                                let open_result = if item.is_video {
                                                    open_in_mpv(path, 0.0)
                                                } else {
                                                    open_in_iris(path)
                                                };
                                                match open_result {
                                                    Ok(()) => {
                                                        self.status =
                                                            format!("opened: {}", path.display());
                                                    }
                                                    Err(err) => {
                                                        self.status = format!("open failed: {err}");
                                                    }
                                                }
                                            } else {
                                                self.status =
                                                    "resolve failed for similar file".to_string();
                                            }
                                        }
                                        if let Some(path) = source_path.as_ref() {
                                            ui.label(file_resolution_and_size(path));
                                        } else {
                                            ui.label("n/a");
                                        }
                                        ui.monospace(display_path_label(&self.roots, item.file_name.as_str()));
                                    });
                                });
                                ui.separator();
                            }
                            ui.separator();
                        }
                    });
                });
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.heading("Media search (native Rust)");
            ui.horizontal(|ui| {
                ui.label("Mode:");
                ui.selectable_value(&mut self.search_mode, SearchMode::Clip, "CLIP");
                ui.selectable_value(&mut self.search_mode, SearchMode::Ocr, "OCR");
                let response = ui.add_sized(
                    [760.0, 28.0],
                    egui::TextEdit::singleline(&mut self.query)
                        .hint_text(match self.search_mode {
                            SearchMode::Clip => "e.g. passport photo, receipt, beach sunset",
                            SearchMode::Ocr => "e.g. invoice number, call me, account name",
                        }),
                );
                let enter = response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                ui.add(egui::Slider::new(&mut self.limit, 1..=500).text("limit"));
                ui.checkbox(&mut self.video_only, "Videos only");
                if ui.button("Search").clicked() || enter {
                    self.search_now();
                }
                let repair_label = if self.repair_running {
                    "Repairing SIFT..."
                } else {
                    "Repair SIFT"
                };
                if ui
                    .add_enabled(!self.repair_running, egui::Button::new(repair_label))
                    .clicked()
                {
                    self.repair_current_results_sift();
                }
                let selection_label = if self.selection_mode {
                    "Stop selecting"
                } else {
                    "Select images"
                };
                if ui.button(selection_label).clicked() {
                    self.selection_mode = !self.selection_mode;
                }
                if ui
                    .add_enabled(
                        !self.selected_images.is_empty(),
                        egui::Button::new("Clear selected"),
                    )
                    .clicked()
                {
                    self.selected_images.clear();
                    self.recompute_selected_pair_sift_diag();
                }
                let repair_selected_label = if self.repair_running {
                    "Repairing selected..."
                } else {
                    "Repair selected SIFT"
                };
                if ui
                    .add_enabled(
                        !self.repair_running && self.selected_images.len() >= 2,
                        egui::Button::new(repair_selected_label),
                    )
                    .clicked()
                {
                    self.repair_selected_images_sift();
                }
                let force_repair_selected_label = if self.repair_running {
                    "Force repairing..."
                } else {
                    "Force repair selected"
                };
                if ui
                    .add_enabled(
                        !self.repair_running && self.selected_images.len() >= 2,
                        egui::Button::new(force_repair_selected_label),
                    )
                    .clicked()
                {
                    self.force_repair_selected_images_sift();
                }
            });
            ui.horizontal_wrapped(|ui| {
                ui.label("Tabs:");
                let mut close_tab = None;
                for idx in 0..self.tabs.len() {
                    let selected = idx == self.active_tab;
                    let title = self.tabs[idx].title.clone();
                    if ui.selectable_label(selected, title).clicked() {
                        self.active_tab = idx;
                    }
                    if self.tabs.len() > 1 && ui.small_button("x").clicked() {
                        close_tab = Some(idx);
                    }
                }
                if let Some(idx) = close_tab {
                    self.tabs.remove(idx);
                    if self.active_tab >= self.tabs.len() {
                        self.active_tab = self.tabs.len().saturating_sub(1);
                    } else if idx < self.active_tab {
                        self.active_tab = self.active_tab.saturating_sub(1);
                    }
                }
            });
            ui.label(&self.status);
            ui.label(format!("cached previews: {}", self.loaded));
            ui.label(format!(
                "selection: {} image(s){}",
                self.selected_images.len(),
                if self.selection_mode {
                    " | click tiles to select"
                } else {
                    ""
                }
            ));
            if let Some(summary) = self.selected_pair_sift_diag.as_ref() {
                ui.label(summary);
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                let thumb_size = 220.0_f32;
                let cell_width = thumb_size + 12.0;
                let width = ui.available_width();
                let cols = (width / cell_width).floor().max(1.0) as usize;
                let rows = self.active_results().to_vec();
                if rows.is_empty() {
                    ui.label("No results in this tab.");
                    return;
                }
                egui::Grid::new("results_grid")
                    .num_columns(cols)
                    .spacing([12.0, 12.0])
                    .show(ui, |ui| {
                        for (idx, row) in rows.iter().enumerate() {
                            let similar_count =
                                self.similar_count_for(&row.file_name, row.is_video);
                            let is_video = row.is_video;
                            let file_name = row.file_name.clone();
                            let timestamp_sec = row.timestamp_sec;
                            let result_row = row.clone();
                            let source_path = resolve_source_path(&self.roots, &file_name).ok();
                            let open_image_path = source_path.clone().or_else(|| row.media_path.clone());

                            let response = if let Some(path) = &row.media_path {
                                if let Some(texture) = self.load_texture_if_needed(ctx, path) {
                                    ui.add(
                                        egui::Image::new(&texture)
                                            .fit_to_exact_size(egui::vec2(thumb_size, thumb_size))
                                            .sense(egui::Sense::click()),
                                    )
                                } else {
                                    let (rect, response) = ui.allocate_exact_size(
                                        egui::vec2(thumb_size, thumb_size),
                                        egui::Sense::click(),
                                    );
                                    ui.painter()
                                        .rect_filled(rect, 4.0, egui::Color32::from_gray(40));
                                    response
                                }
                            } else {
                                let (rect, response) = ui.allocate_exact_size(
                                    egui::vec2(thumb_size, thumb_size),
                                    egui::Sense::click(),
                                );
                                ui.painter()
                                    .rect_filled(rect, 4.0, egui::Color32::from_gray(40));
                                response
                            };

                            if self.selection_mode && response.clicked() && !response.double_clicked() {
                                self.toggle_selected_image(&file_name, is_video);
                            }
                            let image_selected = self.selected_images.contains(&file_name);
                            if image_selected {
                                ui.painter().rect_stroke(
                                    response.rect.expand(1.0),
                                    4.0,
                                    egui::Stroke::new(3.0, egui::Color32::LIGHT_GREEN),
                                    egui::StrokeKind::Outside,
                                );
                            }
                            if response.double_clicked() {
                                self.selected_file = Some(file_name.clone());
                                self.selected_master = Some(self.grouped_master_for(&file_name, is_video));
                            }
                            response.context_menu(|ui| {
                                if !is_video {
                                    let selection_label = if image_selected {
                                        "Deselect image"
                                    } else {
                                        "Select image"
                                    };
                                    if ui.button(selection_label).clicked() {
                                        self.toggle_selected_image(&file_name, is_video);
                                        ui.close();
                                    }
                                    ui.separator();
                                }
                                if is_video {
                                    if ui.button("Open in mpv").clicked() {
                                        match resolve_source_path(&self.roots, &file_name)
                                            .and_then(|video_path| {
                                                open_in_mpv(&video_path, timestamp_sec)?;
                                                Ok(video_path)
                                            }) {
                                            Ok(video_path) => {
                                                self.status = format!(
                                                    "opened in mpv @ {:.3}s: {}",
                                                    timestamp_sec,
                                                    video_path.display()
                                                );
                                            }
                                            Err(err) => {
                                                self.status = format!("open in mpv failed: {err}");
                                            }
                                        }
                                        ui.close();
                                    }
                                } else {
                                    if ui.button("Show most similar").clicked() {
                                        self.show_most_similar_clip(&result_row);
                                        ui.close();
                                    }
                                    if ui.button("Open in Iris").clicked() {
                                        if let Some(path) = open_image_path.as_ref() {
                                            if let Err(err) = open_in_iris(path) {
                                                self.status = format!("open in iris failed: {err}");
                                            } else {
                                                self.status =
                                                    format!("opened in iris: {}", path.display());
                                            }
                                        } else {
                                            self.status =
                                                format!("resolve failed for image: {}", file_name);
                                        }
                                        ui.close();
                                    }
                                    if ui.button("Copy image").clicked() {
                                        if let Some(path) = open_image_path.as_ref() {
                                            match copy_image_to_clipboard(ui.ctx(), path) {
                                                Ok(()) => {
                                                    self.status =
                                                        format!("copied image: {}", path.display());
                                                }
                                                Err(err) => {
                                                    self.status =
                                                        format!("copy image failed: {err}");
                                                }
                                            }
                                        } else {
                                            self.status =
                                                format!("resolve failed for image: {}", file_name);
                                        }
                                        ui.close();
                                    }
                                    if ui.button("Show more of this person").clicked() {
                                        self.show_more_of_this_person(&file_name);
                                        ui.close();
                                    }
                                    if ui
                                        .add_enabled(
                                            !self.face_rerun_running,
                                            egui::Button::new("Rerun face detection (low threshold)"),
                                        )
                                        .clicked()
                                    {
                                        self.start_face_rerun(file_name.clone());
                                        ui.close();
                                    }
                                }
                                ui.separator();
                                if ui.button("Copy full path").clicked() {
                                    if let Some(path) = source_path.as_ref() {
                                        let full_path = path.display().to_string();
                                        ui.ctx().copy_text(full_path.clone());
                                        self.status = format!("copied full path: {full_path}");
                                    } else {
                                        self.status = format!("resolve failed for file: {}", file_name);
                                    }
                                    ui.close();
                                }
                                if ui.button("Open parent folder").clicked() {
                                    if let Some(path) = source_path.as_ref() {
                                        match open_parent_folder(path) {
                                            Ok(()) => {
                                                if let Some(parent) = path.parent() {
                                                    self.status = format!(
                                                        "opened parent folder: {}",
                                                        parent.display()
                                                    );
                                                } else {
                                                    self.status = format!(
                                                        "open parent folder failed: no parent for {}",
                                                        path.display()
                                                    );
                                                }
                                            }
                                            Err(err) => {
                                                self.status =
                                                    format!("open parent folder failed: {err}");
                                            }
                                        }
                                    } else {
                                        self.status = format!("resolve failed for file: {}", file_name);
                                    }
                                    ui.close();
                                }
                            });
                            draw_similarity_badge(ui, response.rect, similar_count);
                            if (idx + 1) % cols == 0 {
                                ui.end_row();
                            }
                        }
                    });
            });
        });
    }
}
