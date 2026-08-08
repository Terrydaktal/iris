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
