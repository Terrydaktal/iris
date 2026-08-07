use super::*;

pub(crate) fn compute_sift_summary(path_a: &Path, path_b: &Path) -> Result<String> {
    let media_indexer_dir = resolve_media_indexer_dir();
    let output = Command::new("uv")
        .current_dir(&media_indexer_dir)
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
        "SIFT Alignment: score {:.4} | inliers {} / good {} ({:.1}%) | kpA: {}, kpB: {}",
        score,
        inliers,
        good,
        inlier_ratio * 100.0,
        keypoints_a,
        keypoints_b
    ))
}

pub(crate) fn run_sift_alignment_batch(
    reference: &Path,
    candidates: &[PathBuf],
    output_dir: &Path,
) -> Result<SiftAlignAllResult> {
    if candidates.is_empty() {
        bail!("there are no comparison images to align");
    }

    let media_indexer_dir = resolve_media_indexer_dir();
    let mut command = Command::new("uv");
    command
        .current_dir(&media_indexer_dir)
        .args(["run", "python", "tools/sift_similarity.py", "--align-all"])
        .arg(reference)
        .arg(output_dir);
    for candidate in candidates {
        command.arg(candidate);
    }

    let output = command
        .output()
        .context("failed to run the SIFT alignment helper")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("SIFT alignment helper failed: {}", stderr.trim());
    }
    let payload: Value =
        serde_json::from_slice(&output.stdout).context("invalid SIFT alignment helper JSON")?;
    if let Some(err) = payload.get("error").and_then(Value::as_str) {
        bail!("{err}");
    }

    let reference = payload
        .get("reference")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .unwrap_or_else(|| reference.to_path_buf());
    let mut aligned_paths = HashMap::new();
    let mut details = Vec::new();
    let mut aligned_count = 0usize;
    let mut failed_count = 0usize;

    for item in payload
        .get("results")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(path) = item.get("path").and_then(Value::as_str) else {
            continue;
        };
        let path = PathBuf::from(path);
        let inliers = item
            .get("inlier_matches")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let good_matches = item
            .get("good_matches")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let score = item.get("score").and_then(Value::as_f64).unwrap_or(0.0);
        let inlier_ratio = item
            .get("inlier_ratio")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("comparison image");

        if let Some(aligned_path) = item.get("aligned_path").and_then(Value::as_str) {
            aligned_count += 1;
            aligned_paths.insert(path.clone(), PathBuf::from(aligned_path));
            details.push(format!(
                "{name}: aligned, score {score:.4}, {inliers}/{good_matches} inliers ({:.1}%)",
                inlier_ratio * 100.0
            ));
        } else {
            failed_count += 1;
            let error = item
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("no aligned output");
            details.push(format!("{name}: not aligned ({error})"));
        }
    }

    let summary = format!(
        "SIFT alignment complete: {aligned_count}/{} comparison images aligned{}.",
        candidates.len(),
        if failed_count > 0 {
            format!("; {failed_count} could not be aligned")
        } else {
            String::new()
        }
    );
    Ok(SiftAlignAllResult {
        reference,
        aligned_paths,
        summary,
        details,
        output_dir: output_dir.to_path_buf(),
    })
}

pub(crate) fn run_sift_repair_for_files(file_names: &[String]) -> Result<SiftRepairResult> {
    if file_names.len() < 2 {
        bail!("select at least two indexed images");
    }

    let db_dir = get_db_dir();
    let roots = load_or_discover_db_roots(&db_dir);
    if roots.is_empty() {
        bail!("no database collection roots are configured");
    }

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let temp_path = std::env::temp_dir().join(format!(
        "iris_sift_repair_{}_{}.json",
        std::process::id(),
        stamp
    ));
    let payload =
        serde_json::to_string(file_names).context("failed to serialize selected file list")?;
    std::fs::write(&temp_path, payload).context("failed to write selected file list")?;

    let media_indexer_dir = resolve_media_indexer_dir();
    let mut command = Command::new("uv");
    command.current_dir(&media_indexer_dir).args([
        "run",
        "python",
        "tools/repair_sift_results.py",
        "--db-dir",
        db_dir.to_string_lossy().as_ref(),
        "--table",
        MEDIA_INDEX_TABLE,
        "--files-json",
        temp_path.to_string_lossy().as_ref(),
        "--min-inliers",
        "10",
        "--min-inlier-ratio",
        "0.40",
    ]);
    for (collection, root) in &roots {
        command.arg("--collection-root");
        command.arg(format!("{}={}", collection, root.to_string_lossy()));
    }
    if file_names.len() == 2 {
        command.arg("--fast-pair");
    }

    let output = command
        .output()
        .context("failed to run repair_sift_results.py")?;
    let _ = std::fs::remove_file(&temp_path);
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("repair failed: {}", stderr.trim());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json_line = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .ok_or_else(|| anyhow!("repair script did not return a JSON summary"))?;
    let payload: Value = serde_json::from_str(json_line).context("invalid repair summary JSON")?;
    let images = payload.get("images").and_then(Value::as_u64).unwrap_or(0);
    let pairs = payload.get("pairs").and_then(Value::as_u64).unwrap_or(0);
    let accepted = payload
        .get("accepted_pairs")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let linked = payload
        .get("linked_images")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let updated = payload.get("updated").and_then(Value::as_u64).unwrap_or(0);
    Ok(SiftRepairResult {
        summary: format!(
            "SIFT repair finished: {images} images, {pairs} pairs checked, {accepted} accepted, {linked} linked, {updated} database rows updated."
        ),
        files: file_names.len(),
    })
}

fn run_on_demand_embedding_helper(image_path: &Path, flags: &[&str]) -> Result<Value> {
    let media_indexer_dir = resolve_media_indexer_dir();
    let helper_script = resolve_on_demand_embeddings_script_path();
    let mut cmd = Command::new("uv");
    cmd.current_dir(&media_indexer_dir);
    cmd.env("UV_CACHE_DIR", "/data/.cache/uv");
    cmd.args(["run", "python"]);
    cmd.arg(&helper_script);
    cmd.arg("--image");
    cmd.arg(image_path);
    for flag in flags {
        cmd.arg(flag);
    }

    let output = cmd
        .output()
        .map_err(|e| anyhow!("failed to run embedding helper: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        bail!(
            "embedding helper failed (status {}) using {} and {}: {}",
            output.status,
            media_indexer_dir.display(),
            helper_script.display(),
            if stderr.is_empty() {
                "unknown error"
            } else {
                stderr.as_str()
            }
        );
    }

    let payload: Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| anyhow!("invalid embedding helper JSON: {e}"))?;
    if !payload.get("ok").and_then(Value::as_bool).unwrap_or(false) {
        let err = payload
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("embedding helper returned ok=false");
        bail!("{err}");
    }
    Ok(payload)
}

pub(crate) fn compute_on_demand_embeddings(
    image_path: &Path,
    need_clip: bool,
    need_faces: bool,
) -> Result<(Option<Vec<f32>>, Vec<Vec<f32>>)> {
    let mut flags = Vec::new();
    if need_clip {
        flags.push("--clip");
    }
    if need_faces {
        flags.push("--faces");
    }
    let payload = run_on_demand_embedding_helper(image_path, &flags)?;

    let clip_vector = payload
        .get("clip_embedding")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_f64)
                .map(|v| v as f32)
                .collect::<Vec<f32>>()
        })
        .filter(|vec| !vec.is_empty());

    let face_vectors = payload
        .get("face_embeddings")
        .and_then(Value::as_array)
        .map(|outer| {
            outer
                .iter()
                .filter_map(Value::as_array)
                .map(|inner| {
                    inner
                        .iter()
                        .filter_map(Value::as_f64)
                        .map(|v| v as f32)
                        .collect::<Vec<f32>>()
                })
                .filter(|vec| !vec.is_empty())
                .collect::<Vec<Vec<f32>>>()
        })
        .unwrap_or_default();

    Ok((clip_vector, face_vectors))
}

pub(crate) fn compute_face_details(image_path: &Path) -> Result<Vec<FaceDetail>> {
    let payload = run_on_demand_embedding_helper(image_path, &["--face-details"])?;
    let details = payload
        .get("face_details")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    let vector = item
                        .get("embedding")?
                        .as_array()?
                        .iter()
                        .filter_map(Value::as_f64)
                        .map(|value| value as f32)
                        .collect::<Vec<_>>();
                    let bbox_values = item.get("bbox")?.as_array()?;
                    if vector.is_empty() || bbox_values.len() != 4 {
                        return None;
                    }
                    let bbox = [
                        bbox_values[0].as_f64()? as f32,
                        bbox_values[1].as_f64()? as f32,
                        bbox_values[2].as_f64()? as f32,
                        bbox_values[3].as_f64()? as f32,
                    ];
                    Some(FaceDetail { vector, bbox })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Ok(details)
}
