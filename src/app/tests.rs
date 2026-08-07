use super::*;

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
