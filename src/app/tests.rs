use super::*;

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
