use super::*;
use chrono::{Local, LocalResult, NaiveDateTime, TimeZone};

#[derive(Clone, Copy)]
struct CachedSemanticDate {
    modified: Option<SystemTime>,
    file_size: u64,
    effective_timestamp: Option<i64>,
    cached_at: Instant,
}

static SEMANTIC_DATE_CACHE: std::sync::OnceLock<Mutex<HashMap<PathBuf, CachedSemanticDate>>> =
    std::sync::OnceLock::new();
static SEMANTIC_DATE_POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> =
    std::sync::OnceLock::new();
const SEMANTIC_DATE_CACHE_LIMIT: usize = 20_000;
const SEMANTIC_DATE_CACHE_TTL: Duration = Duration::from_secs(300);

pub(crate) fn parse_phash_hex(value: &str) -> Option<u64> {
    let value = value.trim();
    if value.len() != 16 {
        return None;
    }
    u64::from_str_radix(value, 16).ok()
}

pub(crate) fn phash_similarity_pct(a: u64, b: u64) -> f32 {
    (64 - (a ^ b).count_ones()) as f32 * 100.0 / 64.0
}

pub(crate) fn similarity_to_active(
    active_file: &str,
    candidate_file: &str,
    phash_by_file: &HashMap<String, u64>,
    video_frame_phashes_by_file: &HashMap<String, Vec<VideoFramePhash>>,
) -> Option<f32> {
    let active_frames = video_frame_phashes_by_file.get(active_file);
    let candidate_frames = video_frame_phashes_by_file.get(candidate_file);
    match (active_frames, candidate_frames) {
        (Some(_), Some(_)) => {
            let active_hash = phash_by_file.get(active_file)?;
            let candidate_hash = phash_by_file.get(candidate_file)?;
            Some(phash_similarity_pct(*active_hash, *candidate_hash))
        }
        (Some(frames), None) => {
            let candidate_hash = phash_by_file.get(candidate_file)?;
            frames
                .iter()
                .map(|frame| phash_similarity_pct(frame.phash, *candidate_hash))
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        }
        (None, Some(frames)) => {
            let active_hash = phash_by_file.get(active_file)?;
            frames
                .iter()
                .map(|frame| phash_similarity_pct(*active_hash, frame.phash))
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        }
        (None, None) => {
            let active_hash = phash_by_file.get(active_file)?;
            let candidate_hash = phash_by_file.get(candidate_file)?;
            Some(phash_similarity_pct(*active_hash, *candidate_hash))
        }
    }
}

pub(crate) fn duplicate_database_detail_lines(
    file_name: &str,
    reference_file: &str,
    is_video: bool,
    phash_by_file: &HashMap<String, u64>,
    video_frame_phashes_by_file: &HashMap<String, Vec<VideoFramePhash>>,
) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push(format!("Reference: {reference_file}"));
    match phash_by_file.get(file_name) {
        Some(hash) => lines.push(format!(
            "{}: {:016x}",
            if is_video { "VideoHash" } else { "pHash" },
            hash
        )),
        None => lines.push(format!(
            "{}: not stored",
            if is_video { "VideoHash" } else { "pHash" }
        )),
    }

    if is_video {
        let frames = video_frame_phashes_by_file.get(file_name);
        lines.push(format!(
            "Video still pHashes: {}",
            frames.map_or(0, Vec::len)
        ));
        if let (Some(frames), Some(reference_hash)) = (frames, phash_by_file.get(reference_file)) {
            if let Some(best) = frames.iter().max_by(|a, b| {
                phash_similarity_pct(a.phash, *reference_hash)
                    .partial_cmp(&phash_similarity_pct(b.phash, *reference_hash))
                    .unwrap_or(Ordering::Equal)
            }) {
                lines.push(format!(
                    "Best still vs reference: {:.3}s | pHash {:016x} | {:.2}%",
                    best.timestamp_sec,
                    best.phash,
                    phash_similarity_pct(best.phash, *reference_hash)
                ));
            }
        }
    }
    lines
}

pub(crate) fn draw_embedding_markers(
    ui: &mut egui::Ui,
    has_clip: bool,
    has_ocr: bool,
    skipped: bool,
) {
    let missing_color = if skipped {
        egui::Color32::GRAY
    } else {
        egui::Color32::YELLOW
    };
    ui.colored_label(
        if has_clip {
            egui::Color32::LIGHT_GREEN
        } else {
            missing_color
        },
        "C",
    )
    .on_hover_text(if has_clip {
        "CLIP embedded"
    } else if skipped {
        "CLIP not embedded: skipped as a pHash similar"
    } else {
        "CLIP not embedded: processing incomplete or failed"
    });
    ui.colored_label(
        if has_ocr {
            egui::Color32::LIGHT_GREEN
        } else {
            missing_color
        },
        "O",
    )
    .on_hover_text(if has_ocr {
        "OCR embedded"
    } else if skipped {
        "OCR not embedded: skipped as a pHash similar"
    } else {
        "OCR has no searchable text, processing is incomplete, or processing failed"
    });
}

pub(crate) async fn load_clip_database_index(db_dir: &Path, table_name: &str) -> Result<ClipIndex> {
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
    let mut seen = HashSet::new();
    for batch in &batches {
        parse_batch(batch, &mut entries, &mut dim, &mut seen)?;
    }
    Ok(ClipIndex {
        entries,
        dim: dim.unwrap_or(512),
        file_count: seen.len(),
    })
}

/// Query the media indexer's dedicated CLIP ANN table. The base table remains
/// the source of truth, while this narrow table avoids loading every embedding
/// into the UI process just to answer an interactive search.
pub(crate) async fn search_clip_ann(
    db_dir: &Path,
    table_name: &str,
    query: &[f32],
    video_only: bool,
    folder_filter: &str,
    min_score: f32,
) -> Result<Vec<SearchResult>> {
    if video_only || !folder_filter.trim().is_empty() {
        bail!("CLIP ANN cannot apply the requested media or folder filter");
    }
    let db = lancedb::connect(db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let ann_table_name = format!("{table_name}_clip_ann");
    let table_names = db.table_names().execute().await?;
    if !table_names.iter().any(|name| name == &ann_table_name) {
        bail!("ANN table {ann_table_name} is not available");
    }
    let table = db.open_table(&ann_table_name).execute().await?;
    // LanceDB requires an explicit limit for nearest-neighbor queries. Use the
    // complete table size so the threshold, rather than an arbitrary top-K,
    // decides which matches are returned.
    let row_count = table.count_rows(None).await?;
    let mut stream = table
        .query()
        .select(Select::columns(&[
            "file_name",
            "timestamp_sec",
            "_distance",
        ]))
        .nearest_to(query.to_vec())?
        .distance_type(lancedb::DistanceType::Cosine)
        .limit(row_count.max(1))
        .execute()
        .await?;
    let db_roots = get_db_roots();
    let mut best_by_file: HashMap<String, (f32, bool, f32)> = HashMap::new();
    while let Some(batch) = stream.try_next().await? {
        let file_names = batch
            .column_by_name("file_name")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| anyhow!("CLIP ANN result has no file_name column"))?;
        let timestamps = batch
            .column_by_name("timestamp_sec")
            .and_then(|column| column.as_any().downcast_ref::<Float32Array>());
        let distances = batch
            .column_by_name("_distance")
            .and_then(|column| column.as_any().downcast_ref::<Float32Array>());
        for row in 0..batch.num_rows() {
            if file_names.is_null(row) {
                continue;
            }
            let file_name = file_names.value(row).to_string();
            let is_video = is_video_path(Path::new(&file_name));
            if video_only && !is_video {
                continue;
            }
            if !folder_filter.is_empty()
                && !file_matches_folder(&file_name, folder_filter, &db_roots)
            {
                continue;
            }
            let distance = distances
                .filter(|array| !array.is_null(row))
                .map(|array| array.value(row))
                .unwrap_or(1.0);
            let score = 1.0 - distance;
            if score < min_score {
                continue;
            }
            let timestamp_sec = timestamps
                .filter(|array| !array.is_null(row))
                .map(|array| array.value(row))
                .unwrap_or(0.0);
            best_by_file
                .entry(file_name)
                .and_modify(|best| {
                    if score > best.0 {
                        *best = (score, is_video, timestamp_sec);
                    }
                })
                .or_insert((score, is_video, timestamp_sec));
        }
    }
    let mut rows: Vec<_> = best_by_file
        .into_iter()
        .map(
            |(file_name, (score, is_video, timestamp_sec))| SearchResult {
                rank: 0,
                score,
                file_name,
                is_video,
                timestamp_sec,
                media_path: None,
                ocr_term_hits: 0,
                ocr_query_terms: 0,
                ocr_phrase_query: false,
            },
        )
        .collect();
    rows.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    for (idx, row) in rows.iter_mut().enumerate() {
        row.rank = idx + 1;
    }
    Ok(rows)
}

pub(crate) async fn search_face_ann(
    db_dir: &Path,
    table_name: &str,
    query_vectors: &[Vec<f32>],
    min_score: f32,
) -> Result<Vec<SearchResult>> {
    if query_vectors.is_empty() {
        return Ok(Vec::new());
    }
    let db = lancedb::connect(db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let ann_table_name = format!("{table_name}_face_ann");
    let table_names = db.table_names().execute().await?;
    if !table_names.iter().any(|name| name == &ann_table_name) {
        bail!("ANN table {ann_table_name} is not available");
    }
    let table = db.open_table(&ann_table_name).execute().await?;
    // Query every stored face vector; the caller's score threshold is the
    // result filter, not a hidden candidate cap.
    let row_count = table.count_rows(None).await?;
    let mut best_by_file: HashMap<String, (f32, bool, f32)> = HashMap::new();
    for query_vector in query_vectors {
        let mut stream = table
            .query()
            .select(Select::columns(&[
                "file_name",
                "timestamp_sec",
                "_distance",
            ]))
            .nearest_to(query_vector.clone())?
            .distance_type(lancedb::DistanceType::Cosine)
            .limit(row_count)
            .execute()
            .await?;
        while let Some(batch) = stream.try_next().await? {
            let file_names = batch
                .column_by_name("file_name")
                .and_then(|column| column.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| anyhow!("face ANN result has no file_name column"))?;
            let timestamps = batch
                .column_by_name("timestamp_sec")
                .and_then(|column| column.as_any().downcast_ref::<Float32Array>());
            let distances = batch
                .column_by_name("_distance")
                .and_then(|column| column.as_any().downcast_ref::<Float32Array>());
            for row in 0..batch.num_rows() {
                if file_names.is_null(row) {
                    continue;
                }
                let file_name = file_names.value(row).to_string();
                let distance = distances
                    .filter(|array| !array.is_null(row))
                    .map(|array| array.value(row))
                    .unwrap_or(1.0);
                let score = 1.0 - distance;
                if score < min_score {
                    continue;
                }
                let timestamp_sec = timestamps
                    .filter(|array| !array.is_null(row))
                    .map(|array| array.value(row))
                    .unwrap_or(0.0);
                let is_video = is_video_path(Path::new(&file_name));
                best_by_file
                    .entry(file_name)
                    .and_modify(|best| {
                        if score > best.0 {
                            *best = (score, is_video, timestamp_sec);
                        }
                    })
                    .or_insert((score, is_video, timestamp_sec));
            }
        }
    }
    let mut rows: Vec<_> = best_by_file
        .into_iter()
        .map(
            |(file_name, (score, is_video, timestamp_sec))| SearchResult {
                rank: 0,
                score,
                file_name,
                is_video,
                timestamp_sec,
                media_path: None,
                ocr_term_hits: 0,
                ocr_query_terms: 0,
                ocr_phrase_query: false,
            },
        )
        .collect();
    rows.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    for (idx, row) in rows.iter_mut().enumerate() {
        row.rank = idx + 1;
    }
    Ok(rows)
}

pub(crate) async fn load_supplemental_database_indices(
    db_dir: &Path,
    table_name: &str,
) -> Result<SupplementalDbData> {
    let db = lancedb::connect(db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let table = db.open_table(table_name).execute().await?;
    let table_schema = table.schema().await?;
    let has_cross_media_matches = table_schema.field_with_name("cross_media_matches").is_ok();
    let has_source_mtime = table_schema.field_with_name("source_mtime_ns").is_ok();
    let mut selected_columns = vec![
        "file_name",
        "is_video",
        "skip_processing",
        "face_groups",
        "ocr_groups",
        "dedupe_match_file",
        "dedupe_similarity_pct",
        "phash_hex",
        "video_frame_phashes",
        "sift_match_file",
        "sift_match_score",
        "sift_match_inliers",
        "sift_match_good_matches",
        "sift_match_inlier_ratio",
        "sift_match_checked",
    ];
    if has_cross_media_matches {
        selected_columns.push("cross_media_matches");
    }
    if has_source_mtime {
        selected_columns.push("source_mtime_ns");
    }
    let stream = table
        .query()
        .select(Select::columns(&selected_columns))
        .execute()
        .await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;

    let mut face_entries = Vec::new();
    let mut face_seen = HashSet::new();

    let mut ocr_entries = Vec::new();
    let mut ocr_seen = HashSet::new();

    let mut similar_by_master: HashMap<String, Vec<SimilarFile>> = HashMap::new();
    let mut source_mtime_by_file: HashMap<String, i64> = HashMap::new();
    let mut phash_master_by_file: HashMap<String, String> = HashMap::new();
    let mut phash_members_by_master: HashMap<String, Vec<String>> = HashMap::new();
    let mut phash_by_file: HashMap<String, u64> = HashMap::new();
    let mut video_frame_phashes_by_file: HashMap<String, Vec<VideoFramePhash>> = HashMap::new();
    let mut sift_info_by_file: HashMap<String, SiftInfo> = HashMap::new();
    let mut master_images = HashSet::new();
    let mut direct_root_by_file: HashMap<String, String> = HashMap::new();
    let mut skipped_processing_files = HashSet::new();

    for batch in &batches {
        // Parse Face
        parse_face_batch(batch, &mut face_entries, &mut face_seen)?;

        // Parse OCR
        parse_ocr_batch(batch, &mut ocr_entries, &mut ocr_seen)?;

        // Parse Similar
        let file_names = string_col(batch, "file_name")?;
        let is_video = bool_col(batch, "is_video")?;
        let dedupe_match = string_col(batch, "dedupe_match_file")?;
        let similarity_col = batch.column_by_name("dedupe_similarity_pct");
        let phash_hex = string_col(batch, "phash_hex")?;
        let video_frame_phashes = list_col(batch, "video_frame_phashes")?;
        let source_mtime_ns = batch
            .column_by_name("source_mtime_ns")
            .and_then(|column| column.as_any().downcast_ref::<Int64Array>());
        let cross_media_matches = batch
            .column_by_name("cross_media_matches")
            .and_then(|column| column.as_any().downcast_ref::<ListArray>());

        for row in 0..batch.num_rows() {
            if file_names.is_null(row) {
                continue;
            }
            let file_name = file_names.value(row).to_string();
            if let Some(source_mtime_ns) = source_mtime_ns
                && !source_mtime_ns.is_null(row)
            {
                source_mtime_by_file.insert(
                    file_name.clone(),
                    source_mtime_ns.value(row).div_euclid(1_000_000_000),
                );
            }
            if !phash_hex.is_null(row) {
                if let Some(hash) = parse_phash_hex(phash_hex.value(row)) {
                    phash_by_file.insert(file_name.clone(), hash);
                }
            }
            if !video_frame_phashes.is_null(row) {
                let groups_any = video_frame_phashes.value(row);
                let groups = groups_any
                    .as_any()
                    .downcast_ref::<StructArray>()
                    .ok_or_else(|| anyhow!("video_frame_phashes value is not a struct array"))?;
                let hashes = groups
                    .column_by_name("phash_hex")
                    .ok_or_else(|| anyhow!("video_frame_phashes missing phash_hex"))?
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| anyhow!("video_frame_phashes phash_hex is not string"))?;
                let timestamps = groups
                    .column_by_name("timestamp_sec")
                    .ok_or_else(|| anyhow!("video_frame_phashes missing timestamp_sec"))?
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| anyhow!("video_frame_phashes timestamp_sec is not float32"))?;
                let parsed: Vec<VideoFramePhash> = (0..hashes.len())
                    .filter(|&idx| !hashes.is_null(idx) && !timestamps.is_null(idx))
                    .filter_map(|idx| {
                        parse_phash_hex(hashes.value(idx)).map(|phash| VideoFramePhash {
                            timestamp_sec: timestamps.value(idx),
                            phash,
                        })
                    })
                    .collect();
                if !parsed.is_empty() {
                    video_frame_phashes_by_file.insert(file_name, parsed);
                }
            }
        }

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
            similar_by_master
                .entry(master.clone())
                .or_default()
                .push(SimilarFile {
                    file_name: similar_file.clone(),
                    is_video: bool_value(is_video, row).unwrap_or(false),
                    similarity_pct,
                });
            phash_master_by_file.insert(similar_file, master);
        }

        if let Some(cross_media_matches) = cross_media_matches {
            for row in 0..batch.num_rows() {
                if file_names.is_null(row) || cross_media_matches.is_null(row) {
                    continue;
                }
                let source_file = file_names.value(row).to_string();
                let matches_any = cross_media_matches.value(row);
                let matches = matches_any
                    .as_any()
                    .downcast_ref::<StructArray>()
                    .ok_or_else(|| anyhow!("cross_media_matches value is not a struct array"))?;
                let related_files = matches
                    .column_by_name("file_name")
                    .ok_or_else(|| anyhow!("cross_media_matches missing file_name"))?
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| anyhow!("cross_media_matches file_name is not string"))?;
                let related_is_video = matches
                    .column_by_name("is_video")
                    .ok_or_else(|| anyhow!("cross_media_matches missing is_video"))?
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| anyhow!("cross_media_matches is_video is not bool"))?;
                let related_similarity = matches.column_by_name("similarity_pct");

                for match_idx in 0..matches.len() {
                    if related_files.is_null(match_idx) {
                        continue;
                    }
                    let related_file = related_files.value(match_idx).to_string();
                    if related_file == source_file {
                        continue;
                    }
                    similar_by_master
                        .entry(source_file.clone())
                        .or_default()
                        .push(SimilarFile {
                            file_name: related_file,
                            is_video: if related_is_video.is_null(match_idx) {
                                false
                            } else {
                                related_is_video.value(match_idx)
                            },
                            similarity_pct: related_similarity
                                .and_then(|col| float_value(col.as_ref(), match_idx)),
                        });
                }
            }
        }

        // Parse Sift Info & Groups
        let sift_match_file = string_col(batch, "sift_match_file")?;
        let sift_score = batch.column_by_name("sift_match_score");
        let sift_inliers = batch.column_by_name("sift_match_inliers");
        let sift_good = batch.column_by_name("sift_match_good_matches");
        let sift_ratio = batch.column_by_name("sift_match_inlier_ratio");
        let sift_checked = bool_col(batch, "sift_match_checked")?;
        let skip_processing = bool_col(batch, "skip_processing")?;

        for row in 0..batch.num_rows() {
            if file_names.is_null(row) {
                continue;
            }
            let file_name = file_names.value(row).to_string();
            if bool_value(skip_processing, row) == Some(true) {
                skipped_processing_files.insert(file_name.clone());
            }
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
            sift_info_by_file.insert(
                file_name.clone(),
                SiftInfo {
                    match_file,
                    score: sift_score.and_then(|col| float_value(col.as_ref(), row)),
                    inliers,
                    good_matches,
                    inlier_ratio: sift_ratio.and_then(|col| float_value(col.as_ref(), row)),
                    checked: bool_value(sift_checked, row),
                },
            );

            // SIFT grouping collection
            if bool_value(is_video, row).unwrap_or(false) {
                continue;
            }
            if bool_value(skip_processing, row) == Some(true) {
                continue;
            }
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

    let face_index = FaceIndex {
        entries: face_entries,
        file_count: face_seen.len(),
    };

    let ocr_index = OcrIndex {
        entries: ocr_entries,
        file_count: ocr_seen.len(),
    };

    for values in similar_by_master.values_mut() {
        let mut best_by_file: HashMap<String, SimilarFile> = HashMap::new();
        for value in values.drain(..) {
            match best_by_file.get(&value.file_name) {
                Some(existing) => {
                    let existing_similarity = existing.similarity_pct.unwrap_or(f32::NEG_INFINITY);
                    let new_similarity = value.similarity_pct.unwrap_or(f32::NEG_INFINITY);
                    if new_similarity > existing_similarity {
                        best_by_file.insert(value.file_name.clone(), value);
                    }
                }
                None => {
                    best_by_file.insert(value.file_name.clone(), value);
                }
            }
        }
        values.extend(best_by_file.into_values());
        values.sort_by(|a, b| {
            b.similarity_pct
                .unwrap_or(f32::NEG_INFINITY)
                .partial_cmp(&a.similarity_pct.unwrap_or(f32::NEG_INFINITY))
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.file_name.cmp(&b.file_name))
        });
    }

    for (member, master) in &phash_master_by_file {
        phash_members_by_master
            .entry(master.clone())
            .or_default()
            .push(member.clone());
    }
    for (master, members) in &mut phash_members_by_master {
        members.push(master.clone());
        members.sort_unstable();
        members.dedup();
    }

    // Build SIFT groups from undirected connected components. The stored
    // sift_match_file value is directional, but grouping is not.
    let mut sift_root_by_file: HashMap<String, String> = HashMap::new();
    let mut sift_members_by_root: HashMap<String, Vec<String>> = HashMap::new();
    let mut sift_neighbors: HashMap<String, Vec<String>> = HashMap::new();
    for file_name in &master_images {
        sift_neighbors.entry(file_name.clone()).or_default();
    }
    for (file_name, target) in &direct_root_by_file {
        if !master_images.contains(file_name.as_str()) || !master_images.contains(target.as_str()) {
            continue;
        }
        sift_neighbors
            .entry(file_name.clone())
            .or_default()
            .push(target.clone());
        sift_neighbors
            .entry(target.clone())
            .or_default()
            .push(file_name.clone());
    }

    let mut visited_sift = HashSet::new();
    for file_name in &master_images {
        if !visited_sift.insert(file_name.clone()) {
            continue;
        }

        let mut stack = vec![file_name.clone()];
        let mut sorted_members = Vec::new();
        while let Some(current) = stack.pop() {
            sorted_members.push(current.clone());
            if let Some(neighbors) = sift_neighbors.get(&current) {
                for neighbor in neighbors {
                    if visited_sift.insert(neighbor.clone()) {
                        stack.push(neighbor.clone());
                    }
                }
            }
        }

        if sorted_members.len() <= 1 {
            continue;
        }
        sorted_members.sort_unstable();
        let canonical = sorted_members[0].clone();
        for member in &sorted_members {
            sift_root_by_file.insert(member.clone(), canonical.clone());
        }
        sift_members_by_root.insert(canonical, sorted_members);
    }

    Ok(SupplementalDbData {
        face_index,
        ocr_index,
        ocr_embedded_files: ocr_seen,
        similar_by_master,
        source_mtime_by_file,
        phash_master_by_file,
        phash_members_by_master,
        phash_by_file,
        video_frame_phashes_by_file,
        sift_info_by_file,
        sift_root_by_file,
        sift_members_by_root,
        skipped_processing_files,
    })
}

pub(crate) fn parse_batch(
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

pub(crate) fn parse_face_batch(
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

pub(crate) fn parse_ocr_batch(
    batch: &RecordBatch,
    entries: &mut Vec<OcrEntry>,
    seen_files: &mut HashSet<String>,
) -> Result<()> {
    let file_names = string_col(batch, "file_name")?;
    let is_video = bool_col(batch, "is_video")?;
    let skip_processing = bool_col(batch, "skip_processing")?;
    let ocr_groups = list_col(batch, "ocr_groups")?;

    for row in 0..batch.num_rows() {
        if bool_value(skip_processing, row) == Some(true)
            || ocr_groups.is_null(row)
            || file_names.is_null(row)
        {
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
            if text_detected.is_null(group_idx)
                || !text_detected.value(group_idx)
                || texts.is_null(group_idx)
            {
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

pub(crate) fn string_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    batch
        .column_by_name(name)
        .ok_or_else(|| anyhow!("missing column {name}"))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow!("column {name} is not string"))
}

pub(crate) fn bool_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a BooleanArray> {
    batch
        .column_by_name(name)
        .ok_or_else(|| anyhow!("missing column {name}"))?
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| anyhow!("column {name} is not bool"))
}

pub(crate) fn list_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ListArray> {
    batch
        .column_by_name(name)
        .ok_or_else(|| anyhow!("missing column {name}"))?
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| anyhow!("column {name} is not list"))
}

pub(crate) fn bool_value(array: &BooleanArray, row: usize) -> Option<bool> {
    if array.is_null(row) {
        None
    } else {
        Some(array.value(row))
    }
}

pub(crate) fn float_value(array: &dyn Array, row: usize) -> Option<f32> {
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

impl ClipTextEncoder {
    pub(crate) fn new(onnx_path: &Path, tokenizer_path: &Path, context_len: usize) -> Result<Self> {
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

    pub(crate) fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
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

pub(crate) fn normalize_in_place(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector {
            *value /= norm;
        }
    }
}

pub(crate) fn search_index(
    index: &ClipIndex,
    query: &[f32],
    min_score: f32,
    video_only: bool,
    folder_filter: &str,
) -> Vec<SearchResult> {
    let db_roots = get_db_roots();
    let merged = index
        .entries
        .par_chunks(4096)
        .map(|chunk| {
            let mut local: HashMap<String, (f32, bool, f32)> = HashMap::new();
            for entry in chunk {
                if video_only && !entry.is_video {
                    continue;
                }
                if !folder_filter.is_empty()
                    && !file_matches_folder(&entry.file_name, folder_filter, &db_roots)
                {
                    continue;
                }
                let score = dot(query, &entry.vector);
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
                ocr_term_hits: 0,
                ocr_query_terms: 0,
                ocr_phrase_query: false,
            },
        )
        .collect();

    rows.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    for (idx, row) in rows.iter_mut().enumerate() {
        row.rank = idx + 1;
    }
    rows
}

fn system_time_timestamp(time: SystemTime) -> Option<i64> {
    match time.duration_since(UNIX_EPOCH) {
        Ok(duration) => i64::try_from(duration.as_secs()).ok(),
        Err(error) => i64::try_from(error.duration().as_secs())
            .ok()
            .map(|seconds| -seconds),
    }
}

fn parse_exif_datetime(value: &[u8]) -> Option<i64> {
    let value = std::str::from_utf8(value).ok()?.trim_matches('\0').trim();
    let datetime = NaiveDateTime::parse_from_str(value.get(..19)?, "%Y:%m:%d %H:%M:%S").ok()?;
    Some(local_timestamp(datetime))
}

fn local_timestamp(datetime: NaiveDateTime) -> i64 {
    match Local.from_local_datetime(&datetime) {
        LocalResult::Single(value) => value.timestamp(),
        LocalResult::Ambiguous(first, second) => first.timestamp().min(second.timestamp()),
        LocalResult::None => datetime.and_utc().timestamp(),
    }
}

fn filename_capture_timestamp(path: &Path) -> Option<i64> {
    let stem = path.file_stem()?.to_str()?;
    const FORMATS: [(&str, usize); 6] = [
        ("%d-%m-%Y_%H-%M-%S", 19),
        ("%Y%m%d_%H%M%S", 15),
        ("%Y%m%d-%H%M%S", 15),
        ("%Y-%m-%d_%H-%M-%S", 19),
        ("%Y-%m-%dT%H-%M-%S", 19),
        ("%Y-%m-%dT%H:%M:%S", 19),
    ];
    for (format, width) in FORMATS {
        for (start, _) in stem.char_indices() {
            let Some(candidate) = stem.get(start..start.saturating_add(width)) else {
                continue;
            };
            if let Ok(datetime) = NaiveDateTime::parse_from_str(candidate, format) {
                return Some(local_timestamp(datetime));
            }
        }
    }
    None
}

fn photo_taken_timestamp(path: &Path) -> Option<i64> {
    let file = std::fs::File::open(path).ok()?;
    let mut reader = std::io::BufReader::new(file);
    let exif = exif::Reader::new().read_from_container(&mut reader).ok()?;
    [
        exif::Tag::DateTimeOriginal,
        exif::Tag::DateTimeDigitized,
        exif::Tag::DateTime,
    ]
    .into_iter()
    .find_map(|tag| {
        let field = exif.get_field(tag, exif::In::PRIMARY)?;
        let exif::Value::Ascii(values) = &field.value else {
            return None;
        };
        parse_exif_datetime(values.first()?)
    })
}

fn earlier_available_timestamp(modified: Option<i64>, taken: Option<i64>) -> Option<i64> {
    match (modified, taken) {
        (Some(modified), Some(taken)) => Some(modified.min(taken)),
        (Some(modified), None) => Some(modified),
        (None, Some(taken)) => Some(taken),
        (None, None) => None,
    }
}

fn effective_path_timestamp(
    path: &Path,
    is_video: bool,
    indexed_modified_timestamp: Option<i64>,
) -> Option<i64> {
    let cache = SEMANTIC_DATE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(cache) = cache.lock()
        && let Some(entry) = cache.get(path)
        && entry.cached_at.elapsed() < SEMANTIC_DATE_CACHE_TTL
    {
        return entry.effective_timestamp;
    }

    let (modified, file_size, modified_timestamp) = match indexed_modified_timestamp {
        Some(timestamp) => (None, 0, Some(timestamp)),
        None => {
            let metadata = std::fs::metadata(path).ok()?;
            let modified = metadata.modified().ok();
            let file_size = metadata.len();
            if let Ok(mut cache) = cache.lock()
                && let Some(entry) = cache.get_mut(path)
                && entry.modified == modified
                && entry.file_size == file_size
            {
                entry.cached_at = Instant::now();
                return entry.effective_timestamp;
            }
            (
                modified,
                file_size,
                modified.and_then(system_time_timestamp),
            )
        }
    };
    let metadata_capture_timestamp = (!is_video).then(|| photo_taken_timestamp(path)).flatten();
    let capture_timestamp = metadata_capture_timestamp.or_else(|| filename_capture_timestamp(path));
    let effective_timestamp = earlier_available_timestamp(modified_timestamp, capture_timestamp);

    if let Ok(mut cache) = cache.lock() {
        if cache.len() >= SEMANTIC_DATE_CACHE_LIMIT && !cache.contains_key(path) {
            cache.clear();
        }
        cache.insert(
            path.to_path_buf(),
            CachedSemanticDate {
                modified,
                file_size,
                effective_timestamp,
                cached_at: Instant::now(),
            },
        );
    }
    effective_timestamp
}

fn effective_media_timestamp(row: &SearchResult) -> Option<i64> {
    effective_path_timestamp(row.media_path.as_deref()?, row.is_video, None)
}

fn duplicate_group_members(
    file_name: &str,
    phash_master_by_file: &HashMap<String, String>,
    phash_members_by_master: &HashMap<String, Vec<String>>,
    sift_root_by_file: &HashMap<String, String>,
    sift_members_by_root: &HashMap<String, Vec<String>>,
) -> (String, Vec<String>) {
    let phash_master = phash_master_by_file
        .get(file_name)
        .map(String::as_str)
        .unwrap_or(file_name);
    let sift_root = sift_root_by_file
        .get(phash_master)
        .map(String::as_str)
        .unwrap_or(phash_master);
    let sift_members: Vec<&str> = sift_members_by_root
        .get(sift_root)
        .map(|members| members.iter().map(String::as_str).collect())
        .unwrap_or_else(|| vec![sift_root]);

    let mut members = Vec::new();
    for sift_member in sift_members {
        if let Some(phash_members) = phash_members_by_master.get(sift_member) {
            members.extend(phash_members.iter().cloned());
        } else {
            members.push(sift_member.to_string());
        }
    }
    members.sort_unstable();
    members.dedup();
    (sift_root.to_string(), members)
}

pub(crate) fn oldest_duplicate_timestamps(
    rows: &[SearchResult],
    index: &DuplicateDateIndex,
    db_roots: &HashMap<String, PathBuf>,
    generation_token: &std::sync::atomic::AtomicU64,
    generation: u64,
) -> Option<HashMap<String, Option<i64>>> {
    let is_cancelled = || generation_token.load(std::sync::atomic::Ordering::Acquire) != generation;
    let mut members_by_group: HashMap<String, Vec<String>> = HashMap::new();
    let mut result_groups = Vec::with_capacity(rows.len());
    for row in rows {
        let (group, mut members) = duplicate_group_members(
            &row.file_name,
            &index.phash_master_by_file,
            &index.phash_members_by_master,
            &index.sift_root_by_file,
            &index.sift_members_by_root,
        );
        if !members.iter().any(|member| member == &row.file_name) {
            members.push(row.file_name.clone());
        }
        members_by_group.entry(group.clone()).or_insert(members);
        result_groups.push((row.file_name.clone(), group));
    }

    let work: Vec<_> = members_by_group.into_iter().collect();
    let calculate_group = |(group, members): &(String, Vec<String>)| {
        let mut oldest = None;
        let mut has_filename_capture = false;
        let mut missing_mtime = Vec::new();
        let mut undated_members = Vec::new();
        for member in members {
            if is_cancelled() {
                break;
            }
            let is_video = is_video_path(Path::new(member));
            let indexed_modified = index.source_mtime_by_file.get(member).copied();
            if let Some(timestamp) = indexed_modified {
                oldest = Some(oldest.map_or(timestamp, |current: i64| current.min(timestamp)));
            } else {
                missing_mtime.push(member);
            }
            if let Some(timestamp) = filename_capture_timestamp(Path::new(member)) {
                has_filename_capture = true;
                oldest = Some(oldest.map_or(timestamp, |current: i64| current.min(timestamp)));
            } else {
                undated_members.push((member, is_video, indexed_modified));
            }
        }

        if has_filename_capture {
            // SIFT/pHash group members represent the same source image. A dated copy
            // supplies the capture date; only missing filesystem mtimes still need I/O.
            for member in missing_mtime {
                if is_cancelled() {
                    break;
                }
                let Ok(path) = resolve_source_path(db_roots, member) else {
                    continue;
                };
                if let Some(timestamp) = std::fs::metadata(path)
                    .ok()
                    .and_then(|metadata| metadata.modified().ok())
                    .and_then(system_time_timestamp)
                {
                    oldest = Some(oldest.map_or(timestamp, |current: i64| current.min(timestamp)));
                }
            }
        } else {
            for (member, is_video, indexed_modified) in undated_members {
                if is_cancelled() {
                    break;
                }
                let Ok(path) = resolve_source_path(db_roots, member) else {
                    continue;
                };
                if let Some(timestamp) = effective_path_timestamp(&path, is_video, indexed_modified)
                {
                    oldest = Some(oldest.map_or(timestamp, |current: i64| current.min(timestamp)));
                }
            }
        }
        (group.clone(), oldest)
    };
    let pool = SEMANTIC_DATE_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .thread_name(|index| format!("semantic-date-{index}"))
            .build()
            .ok()
    });
    let oldest_by_group: HashMap<String, Option<i64>> = match pool {
        Some(pool) => pool.install(|| work.par_iter().map(calculate_group).collect()),
        None => work.iter().map(calculate_group).collect(),
    };
    if is_cancelled() {
        return None;
    }

    Some(
        result_groups
            .into_iter()
            .map(|(file_name, group)| (file_name, oldest_by_group.get(&group).copied().flatten()))
            .collect(),
    )
}

pub(crate) fn sort_semantic_results_by_order(
    rows: &mut Vec<SearchResult>,
    order: SemanticResultOrder,
    oldest_duplicate_dates: Option<&HashMap<String, Option<i64>>>,
) {
    if order == SemanticResultOrder::BestMatch {
        rows.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.file_name.cmp(&b.file_name))
        });
    } else {
        let mut dated: Vec<(Option<i64>, SearchResult)> = rows
            .drain(..)
            .map(|row| {
                let date = match oldest_duplicate_dates.and_then(|dates| dates.get(&row.file_name))
                {
                    Some(date) => *date,
                    None => effective_media_timestamp(&row),
                };
                (date, row)
            })
            .collect();
        dated.sort_by(|(left_date, left), (right_date, right)| {
            let date_order = match (left_date, right_date) {
                (Some(left), Some(right)) => match order {
                    SemanticResultOrder::DateDescending => right.cmp(left),
                    SemanticResultOrder::DateAscending => left.cmp(right),
                    SemanticResultOrder::BestMatch => Ordering::Equal,
                },
                (Some(_), None) => Ordering::Less,
                (None, Some(_)) => Ordering::Greater,
                (None, None) => Ordering::Equal,
            };
            date_order.then_with(|| left.file_name.cmp(&right.file_name))
        });
        rows.extend(dated.into_iter().map(|(_, row)| row));
    }
    for (idx, row) in rows.iter_mut().enumerate() {
        row.rank = idx + 1;
    }
}

pub(crate) fn search_ocr_index(
    index: &OcrIndex,
    query: &str,
    min_score: f32,
    video_only: bool,
    folder_filter: &str,
) -> Vec<SearchResult> {
    let query_trimmed = query.trim();
    if query_trimmed.is_empty() {
        return Vec::new();
    }
    let query_is_quoted =
        query_trimmed.starts_with('"') && query_trimmed.ends_with('"') && query_trimmed.len() >= 2;
    let normalized_query = if query_is_quoted {
        query_trimmed[1..query_trimmed.len() - 1].trim()
    } else {
        query_trimmed
    };
    let query_lower = normalized_query.to_lowercase();
    if query_lower.is_empty() {
        return Vec::new();
    }

    let terms: Vec<&str> = query_lower
        .split_whitespace()
        .filter(|term| !term.is_empty())
        .collect();
    if terms.is_empty() {
        return Vec::new();
    }
    let query_term_count = terms.len();
    let require_phrase_match = query_is_quoted;
    let db_roots = get_db_roots();

    let merged = index
        .entries
        .par_chunks(4096)
        .map(|chunk| {
            let mut local: HashMap<String, (f32, f32, bool, f32, usize, usize, bool)> =
                HashMap::new();
            for entry in chunk {
                if video_only && !entry.is_video {
                    continue;
                }
                if !folder_filter.is_empty()
                    && !file_matches_folder(&entry.file_name, folder_filter, &db_roots)
                {
                    continue;
                }
                let phrase_hit = entry.text_lower.contains(query_lower.as_str());
                let term_hits = terms
                    .iter()
                    .filter(|term| entry.text_lower.contains(**term))
                    .count();
                if require_phrase_match {
                    if !phrase_hit {
                        continue;
                    }
                } else if term_hits == 0 {
                    continue;
                }
                let match_score = if require_phrase_match {
                    1.0
                } else {
                    term_hits as f32 / query_term_count as f32
                };
                if match_score < min_score {
                    continue;
                }
                // Unquoted mode: prioritize rows that match more query terms.
                // Quoted mode: exact phrase required; term count keeps deterministic tie ordering.
                let score = if require_phrase_match {
                    10_000.0 + term_hits as f32
                } else {
                    let all_terms_bonus = if term_hits == query_term_count {
                        100.0
                    } else {
                        0.0
                    };
                    let phrase_bonus = if phrase_hit { 10.0 } else { 0.0 };
                    (term_hits as f32) * 1000.0 + all_terms_bonus + phrase_bonus
                };
                local
                    .entry(entry.file_name.to_string())
                    .and_modify(|best| {
                        if score > best.0 {
                            *best = (
                                score,
                                match_score,
                                entry.is_video,
                                entry.timestamp_sec,
                                term_hits,
                                query_term_count,
                                require_phrase_match,
                            );
                        }
                    })
                    .or_insert((
                        score,
                        match_score,
                        entry.is_video,
                        entry.timestamp_sec,
                        term_hits,
                        query_term_count,
                        require_phrase_match,
                    ));
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
            |(
                file_name,
                (
                    _score,
                    match_score,
                    is_video,
                    timestamp_sec,
                    ocr_term_hits,
                    ocr_query_terms,
                    ocr_phrase_query,
                ),
            )| SearchResult {
                rank: 0,
                score: match_score,
                file_name,
                is_video,
                timestamp_sec,
                media_path: None,
                ocr_term_hits,
                ocr_query_terms,
                ocr_phrase_query,
            },
        )
        .collect();

    rows.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    for (idx, row) in rows.iter_mut().enumerate() {
        row.rank = idx + 1;
    }
    rows
}

pub(crate) const FACE_MATCH_MIN_SCORE: f32 = 0.30;

pub(crate) fn search_face_index(
    index: &FaceIndex,
    query_vectors: &[Vec<f32>],
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
                ocr_term_hits: 0,
                ocr_query_terms: 0,
                ocr_phrase_query: false,
            },
        )
        .collect();
    rows.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    for (idx, row) in rows.iter_mut().enumerate() {
        row.rank = idx + 1;
    }
    rows
}

pub(crate) fn collapse_sift_grouped_results(
    mut rows: Vec<SearchResult>,
    sift_root_by_file: &HashMap<String, String>,
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
        collapsed.push(best);
    }
    collapsed.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    for (idx, row) in collapsed.iter_mut().enumerate() {
        row.rank = idx + 1;
    }
    collapsed
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod semantic_date_tests {
    use super::*;

    #[test]
    fn clip_search_returns_all_rows_passing_threshold() {
        let entries = (0..501)
            .map(|index| ClipEntry {
                file_name: Arc::<str>::from(format!("image-{index}.jpg")),
                is_video: false,
                timestamp_sec: 0.0,
                vector: vec![0.9, 0.1],
            })
            .chain(std::iter::once(ClipEntry {
                file_name: Arc::<str>::from("below-threshold.jpg"),
                is_video: false,
                timestamp_sec: 0.0,
                vector: vec![0.4, 0.1],
            }))
            .collect();
        let index = ClipIndex {
            entries,
            dim: 2,
            file_count: 502,
        };

        let rows = search_index(&index, &[1.0, 0.0], 0.5, false, "");

        assert_eq!(rows.len(), 501);
        assert!(rows.iter().all(|row| row.score >= 0.5));
    }

    #[test]
    fn ocr_search_threshold_filters_by_fraction_of_query_terms() {
        let index = OcrIndex {
            entries: vec![
                OcrEntry {
                    file_name: Arc::<str>::from("two-terms.jpg"),
                    is_video: false,
                    timestamp_sec: 0.0,
                    text_lower: "word1 word2".to_string(),
                },
                OcrEntry {
                    file_name: Arc::<str>::from("one-term.jpg"),
                    is_video: false,
                    timestamp_sec: 0.0,
                    text_lower: "word1".to_string(),
                },
            ],
            file_count: 2,
        };

        let rows = search_ocr_index(&index, "word1 word2", 1.0, false, "");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].file_name, "two-terms.jpg");
        assert_eq!(rows[0].score, 1.0);
    }

    #[test]
    fn exif_datetime_is_parsed_as_local_capture_time() {
        let parsed = parse_exif_datetime(b"2016:07:14 12:34:56\0").expect("parse EXIF date");
        let naive = NaiveDateTime::parse_from_str("2016:07:14 12:34:56", "%Y:%m:%d %H:%M:%S")
            .expect("parse expected date");
        let expected = match Local.from_local_datetime(&naive) {
            LocalResult::Single(value) => value.timestamp(),
            LocalResult::Ambiguous(first, second) => first.timestamp().min(second.timestamp()),
            LocalResult::None => naive.and_utc().timestamp(),
        };
        assert_eq!(parsed, expected);
    }

    #[test]
    fn photo_sort_date_uses_whichever_timestamp_is_earlier() {
        assert_eq!(
            earlier_available_timestamp(Some(2_000), Some(1_000)),
            Some(1_000)
        );
        assert_eq!(
            earlier_available_timestamp(Some(1_000), Some(2_000)),
            Some(1_000)
        );
        assert_eq!(earlier_available_timestamp(Some(1_000), None), Some(1_000));
        assert_eq!(earlier_available_timestamp(None, Some(1_000)), Some(1_000));
        assert_eq!(earlier_available_timestamp(None, None), None);
    }

    #[test]
    fn capture_dates_are_recognized_in_export_and_camera_filenames() {
        let telegram = filename_capture_timestamp(Path::new("photo_8555@30-09-2016_20-03-46.jpg"));
        let camera = filename_capture_timestamp(Path::new("IMG_20221124_165440.jpg"));
        let screenshot =
            filename_capture_timestamp(Path::new("Screenshot_20221227_230518_app.jpg"));
        let unrelated = filename_capture_timestamp(Path::new("445739505_349698.jpg"));

        assert_eq!(
            telegram,
            Some(local_timestamp(
                NaiveDateTime::parse_from_str("2016-09-30 20:03:46", "%Y-%m-%d %H:%M:%S")
                    .expect("parse expected Telegram date")
            ))
        );
        assert_eq!(
            camera,
            Some(local_timestamp(
                NaiveDateTime::parse_from_str("2022-11-24 16:54:40", "%Y-%m-%d %H:%M:%S")
                    .expect("parse expected camera date")
            ))
        );
        assert!(screenshot.is_some());
        assert_eq!(unrelated, None);
    }

    #[test]
    fn duplicate_date_groups_include_phash_copies_of_every_sift_member() {
        let phash_master_by_file = HashMap::from([
            ("copy-a.jpg".to_string(), "master-a.jpg".to_string()),
            ("copy-b.jpg".to_string(), "master-b.jpg".to_string()),
        ]);
        let phash_members_by_master = HashMap::from([
            (
                "master-a.jpg".to_string(),
                vec!["copy-a.jpg".to_string(), "master-a.jpg".to_string()],
            ),
            (
                "master-b.jpg".to_string(),
                vec!["copy-b.jpg".to_string(), "master-b.jpg".to_string()],
            ),
        ]);
        let sift_root_by_file = HashMap::from([
            ("master-a.jpg".to_string(), "master-a.jpg".to_string()),
            ("master-b.jpg".to_string(), "master-a.jpg".to_string()),
        ]);
        let sift_members_by_root = HashMap::from([(
            "master-a.jpg".to_string(),
            vec!["master-a.jpg".to_string(), "master-b.jpg".to_string()],
        )]);

        let (group, members) = duplicate_group_members(
            "copy-a.jpg",
            &phash_master_by_file,
            &phash_members_by_master,
            &sift_root_by_file,
            &sift_members_by_root,
        );

        assert_eq!(group, "master-a.jpg");
        assert_eq!(
            members,
            ["copy-a.jpg", "copy-b.jpg", "master-a.jpg", "master-b.jpg"]
        );
    }

    #[test]
    fn duplicate_date_resolution_uses_the_oldest_copy_and_honors_cancellation() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("test clock after epoch")
            .as_nanos();
        let temp_dir = std::env::temp_dir().join(format!(
            "iris-duplicate-date-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&temp_dir).expect("create duplicate date test directory");
        let newest_name = "collection/photo_20240102_120000.jpg";
        let oldest_name = "collection/photo_20200102_120000.jpg";
        let newest_path = temp_dir.join("photo_20240102_120000.jpg");
        let oldest_path = temp_dir.join("photo_20200102_120000.jpg");
        std::fs::write(&newest_path, []).expect("create newer duplicate");
        std::fs::write(&oldest_path, []).expect("create older duplicate");

        let rows = vec![SearchResult {
            rank: 1,
            score: 1.0,
            file_name: newest_name.to_string(),
            is_video: false,
            timestamp_sec: 0.0,
            media_path: Some(newest_path),
            ocr_term_hits: 0,
            ocr_query_terms: 0,
            ocr_phrase_query: false,
        }];
        let phash_master_by_file = HashMap::from([
            (newest_name.to_string(), newest_name.to_string()),
            (oldest_name.to_string(), newest_name.to_string()),
        ]);
        let phash_members_by_master = HashMap::from([(
            newest_name.to_string(),
            vec![newest_name.to_string(), oldest_name.to_string()],
        )]);
        let index = DuplicateDateIndex {
            phash_master_by_file: Arc::new(phash_master_by_file),
            phash_members_by_master: Arc::new(phash_members_by_master),
            ..DuplicateDateIndex::default()
        };
        let roots = HashMap::from([("collection".to_string(), temp_dir.clone())]);
        let token = std::sync::atomic::AtomicU64::new(7);
        let dates = oldest_duplicate_timestamps(&rows, &index, &roots, &token, 7)
            .expect("date resolution should complete");
        assert_eq!(
            dates.get(newest_name).copied().flatten(),
            filename_capture_timestamp(&oldest_path)
        );

        token.store(8, std::sync::atomic::Ordering::Release);
        assert!(oldest_duplicate_timestamps(&rows, &index, &roots, &token, 7).is_none());
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn indexed_mtime_and_dated_filename_do_not_require_media_file_access() {
        let file_name = "collection/photo_20200102_120000.jpg";
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("test clock after epoch")
            .as_nanos();
        let missing_root = std::env::temp_dir().join(format!(
            "iris-missing-date-source-{}-{unique}",
            std::process::id()
        ));
        let rows = vec![SearchResult {
            rank: 1,
            score: 1.0,
            file_name: file_name.to_string(),
            is_video: false,
            timestamp_sec: 0.0,
            media_path: None,
            ocr_term_hits: 0,
            ocr_query_terms: 0,
            ocr_phrase_query: false,
        }];
        let indexed_modified = local_timestamp(
            NaiveDateTime::parse_from_str("2024-01-02 12:00:00", "%Y-%m-%d %H:%M:%S")
                .expect("parse indexed mtime"),
        );
        let index = DuplicateDateIndex {
            source_mtime_by_file: Arc::new(HashMap::from([(
                file_name.to_string(),
                indexed_modified,
            )])),
            ..DuplicateDateIndex::default()
        };
        let roots = HashMap::from([("collection".to_string(), missing_root.clone())]);
        let token = std::sync::atomic::AtomicU64::new(2);

        let dates = oldest_duplicate_timestamps(&rows, &index, &roots, &token, 2)
            .expect("indexed date resolution should complete");

        assert_eq!(
            dates.get(file_name).copied().flatten(),
            filename_capture_timestamp(Path::new(file_name))
        );
        assert!(!missing_root.exists());
    }

    #[test]
    fn video_date_resolution_uses_the_video_instead_of_its_preview_still() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("test clock after epoch")
            .as_nanos();
        let temp_dir =
            std::env::temp_dir().join(format!("iris-video-date-{}-{unique}", std::process::id()));
        std::fs::create_dir_all(&temp_dir).expect("create video date test directory");
        let video_path = temp_dir.join("video.mp4");
        let preview_path = temp_dir.join("video-preview.jpg");
        let video_modified = SystemTime::now() - Duration::from_secs(3_600);
        std::fs::File::create(&video_path)
            .expect("create test video")
            .set_modified(video_modified)
            .expect("set test video date");
        std::fs::File::create(&preview_path).expect("create test preview");
        let file_name = "collection/video.mp4";
        let rows = vec![SearchResult {
            rank: 1,
            score: 1.0,
            file_name: file_name.to_string(),
            is_video: true,
            timestamp_sec: 5.0,
            media_path: Some(preview_path),
            ocr_term_hits: 0,
            ocr_query_terms: 0,
            ocr_phrase_query: false,
        }];
        let roots = HashMap::from([("collection".to_string(), temp_dir.clone())]);
        let token = std::sync::atomic::AtomicU64::new(4);
        let dates =
            oldest_duplicate_timestamps(&rows, &DuplicateDateIndex::default(), &roots, &token, 4)
                .expect("video date resolution should complete");
        assert_eq!(
            dates.get(file_name).copied().flatten(),
            system_time_timestamp(video_modified)
        );
        let _ = std::fs::remove_dir_all(temp_dir);
    }
}
