use super::*;

fn best_face_pair<'a, 'b>(
    left: &'a [FaceDetail],
    right: &'b [FaceDetail],
) -> Option<(f32, &'a FaceDetail, &'b FaceDetail)> {
    let mut best: Option<(f32, &'a FaceDetail, &'b FaceDetail)> = None;
    for left_face in left {
        for right_face in right {
            let score = dot(&left_face.vector, &right_face.vector);
            if best.as_ref().is_none_or(|current| score > current.0) {
                best = Some((score, left_face, right_face));
            }
        }
    }
    best
}

impl ImageViewer {
    pub(crate) fn start_selected_face_compare(&mut self, ctx: &egui::Context) {
        if self.face_compare_running || self.selected_grid_items.len() != 2 {
            return;
        }

        let selected = self.selected_grid_items.clone();
        let video_count = selected.iter().filter(|item| item.is_video).count();
        if video_count > 1 {
            self.semantic_status =
                "Face comparison needs two photos or one photo and one video.".to_string();
            return;
        }
        if video_count == 1 && (!self.db_loaded || self.db_indices.is_none()) {
            if !self.db_loading && !self.db_failed {
                self.start_lazy_db_load(ctx);
            }
            self.semantic_status =
                "Loading indexed video face timestamps before face comparison...".to_string();
            return;
        }

        let video_faces = if video_count == 1 {
            let video = selected.iter().find(|item| item.is_video).unwrap();
            let Some(db_filename) = video.db_filename.as_deref() else {
                self.semantic_status = "Video face matching requires an indexed video.".to_string();
                return;
            };
            let faces = self
                .db_indices
                .as_ref()
                .map(|indices| {
                    indices
                        .face_index
                        .entries
                        .iter()
                        .filter(|entry| entry.is_video && entry.file_name.as_ref() == db_filename)
                        .cloned()
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if faces.is_empty() {
                self.semantic_status =
                    "The selected video has no indexed face embeddings.".to_string();
                return;
            }
            faces
        } else {
            Vec::new()
        };

        self.face_compare_running = true;
        self.semantic_status = if video_count == 1 {
            "Finding the video frame with the closest matching face...".to_string()
        } else {
            "Calculating face resemblance between the selected photos...".to_string()
        };
        let (tx, rx) = std::sync::mpsc::channel();
        self.face_compare_rx = Some(rx);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = if video_count == 0 {
                compare_photo_faces(&selected[0].path, &selected[1].path)
            } else {
                compare_photo_to_video(&selected, &video_faces)
            };
            let _ = tx.send(result.map_err(|err| err.to_string()));
            ctx.request_repaint();
        });
    }

    pub(crate) fn poll_face_compare(&mut self, ctx: &egui::Context) {
        let Some(rx) = self.face_compare_rx.take() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(result)) => {
                self.face_compare_running = false;
                self.face_overlay_boxes.clear();
                self.open_comparison_paths(result.paths, ctx);
                self.face_overlay_boxes.extend(
                    result
                        .overlay_boxes
                        .into_iter()
                        .map(|(path, bbox)| (path.canonicalize().unwrap_or(path), vec![bbox])),
                );
                self.current_index = result.active_index.min(self.images.len().saturating_sub(1));
                self.comparison_alignment_status = result.summary.clone();
                self.semantic_status = result.summary;
                self.update_current_file_info();
            }
            Ok(Err(err)) => {
                self.face_compare_running = false;
                self.semantic_status = format!("Face comparison failed: {err}");
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                self.face_compare_rx = Some(rx);
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.face_compare_running = false;
                self.semantic_status = "Face comparison worker disconnected.".to_string();
            }
        }
    }
}

fn compare_photo_faces(path_a: &Path, path_b: &Path) -> Result<FaceComparisonResult> {
    let faces_a = compute_face_details(path_a)?;
    let faces_b = compute_face_details(path_b)?;
    let Some((score, face_a, face_b)) = best_face_pair(&faces_a, &faces_b) else {
        bail!("a face could not be detected in both selected photos");
    };
    let path_a = path_a.to_path_buf();
    let path_b = path_b.to_path_buf();
    Ok(FaceComparisonResult {
        paths: vec![path_a.clone(), path_b.clone()],
        active_index: 0,
        overlay_boxes: vec![(path_a, face_a.bbox), (path_b, face_b.bbox)],
        summary: format!(
            "Closest face resemblance: {:.1}% cosine similarity",
            score.clamp(0.0, 1.0) * 100.0
        ),
    })
}

fn compare_photo_to_video(
    selected: &[GallerySelection],
    video_faces: &[FaceEntry],
) -> Result<FaceComparisonResult> {
    let photo = selected
        .iter()
        .find(|item| !item.is_video)
        .ok_or_else(|| anyhow!("no photo was selected"))?;
    let video = selected
        .iter()
        .find(|item| item.is_video)
        .ok_or_else(|| anyhow!("no video was selected"))?;
    let photo_faces = compute_face_details(&photo.path)?;
    if photo_faces.is_empty() {
        bail!("no face was detected in the selected photo");
    }

    let mut best: Option<(f32, &FaceDetail, &FaceEntry)> = None;
    for photo_face in &photo_faces {
        for video_face in video_faces {
            let score = dot(&photo_face.vector, &video_face.vector);
            if best.as_ref().is_none_or(|current| score > current.0) {
                best = Some((score, photo_face, video_face));
            }
        }
    }
    let Some((indexed_score, photo_face, indexed_face)) = best else {
        bail!("the selected video has no comparable face embeddings");
    };
    if indexed_score < FACE_MATCH_MIN_SCORE {
        bail!(
            "no credible face match was found in the selected video (best cosine similarity {:.1}%)",
            indexed_score.max(0.0) * 100.0
        );
    }
    let db_filename = video
        .db_filename
        .as_deref()
        .ok_or_else(|| anyhow!("the selected video is not indexed"))?;
    let roots = get_db_roots();
    let still = resolve_media_path(
        &roots,
        &get_db_dir(),
        db_filename,
        indexed_face.timestamp_sec,
    )?;
    if still == video.path || is_video_path(&still) {
        bail!("no extracted still is available near the matching video timestamp");
    }

    let still_faces = compute_face_details(&still)?;
    let Some((_, _, still_face)) = best_face_pair(std::slice::from_ref(photo_face), &still_faces)
    else {
        bail!("the matching face could not be located on the extracted video frame");
    };
    let score = indexed_score;
    Ok(FaceComparisonResult {
        paths: vec![photo.path.clone(), still.clone()],
        active_index: 1,
        overlay_boxes: vec![(still, still_face.bbox)],
        summary: format!(
            "Best video face match: {:.1}% cosine similarity at {:.3}s",
            score.clamp(0.0, 1.0) * 100.0,
            indexed_face.timestamp_sec
        ),
    })
}
