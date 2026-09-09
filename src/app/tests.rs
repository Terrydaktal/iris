use super::*;

#[test]
fn viewer_zoom_stops_moving_at_both_limits() {
    for (zoom, factor) in [
        (ImageViewState::MAX_ZOOM, 1.2),
        (ImageViewState::MIN_ZOOM, 0.8),
    ] {
        let offset = egui::vec2(85.5, 125.25);
        let mut view = ImageViewState { zoom, offset };
        for step in 0..100 {
            let anchor = egui::vec2(-148.25 + step as f32, -339.5);
            assert!(!view.zoom_at(factor, anchor));
            assert_eq!(view.zoom, zoom);
            assert_eq!(view.offset, offset);
        }
    }
}

#[test]
fn viewer_zoom_preserves_cursor_anchor_when_crossing_limits() {
    for (zoom, factor, expected_zoom) in [
        (
            ImageViewState::MAX_ZOOM - 1.0,
            1.2,
            ImageViewState::MAX_ZOOM,
        ),
        (0.06, 0.5, 0.05),
        (1.0, 1.2, 1.2),
        (1.0, 0.8, 0.8),
    ] {
        let mut view = ImageViewState {
            zoom,
            offset: egui::vec2(85.5, 125.25),
        };
        let anchor = egui::vec2(-148.25, -339.5);
        let image_point_at_cursor = (anchor - view.offset) / view.zoom;
        assert!(view.zoom_at(factor, anchor));
        assert_eq!(view.zoom, expected_zoom);
        let projected_cursor = view.offset + image_point_at_cursor * view.zoom;
        assert!((projected_cursor - anchor).length() < 0.001);
    }
}

#[test]
fn viewer_zoom_can_reverse_after_scrolling_past_limit() {
    for (zoom, outward, inward) in [
        (ImageViewState::MAX_ZOOM, 1.2, 0.9),
        (ImageViewState::MIN_ZOOM, 0.8, 1.1),
    ] {
        let mut view = ImageViewState {
            zoom,
            offset: egui::vec2(150.0, 275.0),
        };
        let anchor = egui::vec2(-120.0, -230.0);
        let image_point_at_cursor = (anchor - view.offset) / view.zoom;
        for _ in 0..10 {
            view.zoom_at(outward, anchor);
        }
        assert!(view.zoom_at(inward, anchor));
        assert_eq!(view.zoom, zoom * inward);
        let projected_cursor = view.offset + image_point_at_cursor * view.zoom;
        assert!((projected_cursor - anchor).length() < 0.001);
    }
}

#[test]
fn viewer_zoom_continues_beyond_previous_limit() {
    let anchor = egui::vec2(-120.0, -230.0);
    let mut view = ImageViewState {
        zoom: 32.0,
        offset: egui::vec2(150.0, 275.0),
    };
    let image_point_at_cursor = (anchor - view.offset) / view.zoom;
    for expected_zoom in [64.0, 128.0, 256.0] {
        assert!(view.zoom_at(2.0, anchor));
        assert_eq!(view.zoom, expected_zoom);
        assert!((view.offset + image_point_at_cursor * view.zoom - anchor).length() < 0.001);
    }
}

fn test_roots() -> HashMap<String, PathBuf> {
    HashMap::from([("collection".to_string(), PathBuf::from("/media/library"))])
}

#[test]
fn folder_filter_matches_supported_scope_forms() {
    let roots = test_roots();
    let file_name = "collection/People/Ayman/Trips/Sensitive Information 5/photos/image.jpg";

    assert!(file_matches_folder(
        file_name,
        "/media/library/People/Ayman",
        &roots
    ));
    assert!(file_matches_folder(file_name, "People/Ayman/Trips", &roots));
    assert!(file_matches_folder(
        file_name,
        "Sensitive Information 5/photos",
        &roots
    ));
    assert!(file_matches_folder(file_name, "ayman", &roots));
}

#[test]
fn single_segment_folder_filter_does_not_match_filename() {
    let roots = test_roots();

    assert!(!file_matches_folder(
        "collection/People/Trips/ayman-photo.jpg",
        "ayman",
        &roots
    ));
}

#[test]
fn metadata_queue_keeps_only_the_latest_pending_request() {
    let queue = MetadataJobQueue::default();
    let mut state = queue.state.lock().expect("metadata queue lock");
    state.pending = Some(MetadataLoadRequest {
        logical_path: PathBuf::from("first.jpg"),
        inspect_path: PathBuf::from("first.jpg"),
        generation: 1,
        load_exif: true,
        load_layout: false,
    });
    state.pending = Some(MetadataLoadRequest {
        logical_path: PathBuf::from("second.jpg"),
        inspect_path: PathBuf::from("second.jpg"),
        generation: 2,
        load_exif: true,
        load_layout: true,
    });

    let request = state.pending.take().expect("latest metadata request");
    assert_eq!(request.generation, 2);
    assert_eq!(request.logical_path, PathBuf::from("second.jpg"));
    assert!(request.load_layout);
}

#[test]
fn build_identity_exposes_reproducible_fields() {
    let info = build_info_value();

    assert_eq!(info["application"], "iris");
    assert!(
        !info["package_version"]
            .as_str()
            .unwrap_or_default()
            .is_empty()
    );
    assert!(!info["git_revision"].as_str().unwrap_or_default().is_empty());
    assert!(!info["rustc"].as_str().unwrap_or_default().is_empty());
}

#[test]
fn ffmpeg_extracts_thumbnail_for_unindexed_video() {
    if std::process::Command::new("ffmpeg")
        .arg("-version")
        .output()
        .is_err()
    {
        return;
    }

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp_dir = std::env::temp_dir().join(format!(
        "iris-video-thumbnail-test-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&temp_dir).expect("create video thumbnail test directory");
    let video_path = temp_dir.join("unindexed.mkv");
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=64x48:d=0.1",
            "-frames:v",
            "1",
            "-c:v",
            "ffv1",
        ])
        .arg(&video_path)
        .status()
        .expect("run ffmpeg test fixture generation");
    assert!(status.success(), "ffmpeg should create the test video");

    let thumbnail =
        load_video_thumbnail(&video_path, 32, 32, true).expect("extract video thumbnail");
    assert_eq!((thumbnail.width(), thumbnail.height()), (32, 32));

    let _ = std::fs::remove_file(video_path);
    let _ = std::fs::remove_dir(temp_dir);
}

#[test]
fn raw_exif_highlights_only_resolution_and_file_dates() {
    assert!(formatting::is_important_exif_line(
        "[Composite] Image Size : 3840x2160"
    ));
    assert!(formatting::is_important_exif_line("    \"width\": 1920,"));
    assert!(formatting::is_important_exif_line("[PNG] Bit Depth : 16"));
    assert!(formatting::is_important_exif_line(
        "[EXIF] Date/Time Original : 2026:08:30 12:34:56"
    ));
    assert!(formatting::is_important_exif_line(
        "[File] File Modification Date/Time : 2026:08:30 12:34:56+01:00"
    ));
    assert!(!formatting::is_important_exif_line(
        "[EXIF] Camera Model Name : Example Camera"
    ));
    assert!(!formatting::is_important_exif_line(
        "[EXIF] Exposure Time : 1/250"
    ));
    assert!(!formatting::is_important_exif_line("[EXIF] ISO : 100"));
    assert!(!formatting::is_important_exif_line(
        "0x0012 X Resolution : 72"
    ));
    assert!(!formatting::is_important_exif_line(
        "0x0014 Y Resolution : 72"
    ));
    assert!(!formatting::is_important_exif_line(
        "[ExifTool] ExifTool Version Number : 13.00"
    ));
}

#[test]
fn semantic_results_can_be_sorted_by_file_date_in_both_directions() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp_dir = std::env::temp_dir().join(format!(
        "iris-semantic-date-sort-test-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&temp_dir).expect("create semantic date sort test directory");

    let oldest_path = temp_dir.join("oldest.jpg");
    let middle_path = temp_dir.join("middle.jpg");
    let newest_path = temp_dir.join("newest.jpg");
    std::fs::File::create(&oldest_path)
        .expect("create oldest test file")
        .set_modified(SystemTime::now() - std::time::Duration::from_secs(120))
        .expect("set oldest test file date");
    std::fs::File::create(&middle_path)
        .expect("create middle test file")
        .set_modified(SystemTime::now() - std::time::Duration::from_secs(60))
        .expect("set middle test file date");
    std::fs::File::create(&newest_path)
        .expect("create newest test file")
        .set_modified(SystemTime::now())
        .expect("set newest test file date");

    let make_result = |name: &str, path: &Path, score: f32| SearchResult {
        rank: 0,
        score,
        file_name: name.to_string(),
        is_video: false,
        timestamp_sec: 0.0,
        media_path: Some(path.to_path_buf()),
        ocr_term_hits: 0,
        ocr_query_terms: 0,
        ocr_phrase_query: false,
    };
    let mut rows = vec![
        make_result("collection/oldest.jpg", &oldest_path, 0.9),
        make_result("collection/newest.jpg", &newest_path, 0.1),
        make_result("collection/middle.jpg", &middle_path, 0.5),
    ];

    sort_semantic_results_by_order(&mut rows, SemanticResultOrder::DateDescending, None);
    assert_eq!(
        rows.iter()
            .map(|row| row.file_name.as_str())
            .collect::<Vec<_>>(),
        [
            "collection/newest.jpg",
            "collection/middle.jpg",
            "collection/oldest.jpg"
        ]
    );

    sort_semantic_results_by_order(&mut rows, SemanticResultOrder::DateAscending, None);
    assert_eq!(
        rows.iter()
            .map(|row| row.file_name.as_str())
            .collect::<Vec<_>>(),
        [
            "collection/oldest.jpg",
            "collection/middle.jpg",
            "collection/newest.jpg"
        ]
    );

    sort_semantic_results_by_order(&mut rows, SemanticResultOrder::BestMatch, None);
    assert_eq!(
        rows.iter()
            .map(|row| row.file_name.as_str())
            .collect::<Vec<_>>(),
        [
            "collection/oldest.jpg",
            "collection/middle.jpg",
            "collection/newest.jpg"
        ]
    );

    let _ = std::fs::remove_dir_all(temp_dir);
}
