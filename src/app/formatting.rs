use super::model::{SiftInfo, valid_sift_link};
use chrono::{DateTime, Local};
use eframe::egui;
use std::collections::HashMap;
use std::path::Path;

pub(crate) fn file_size_and_modified(path: &Path) -> String {
    let metadata = std::fs::metadata(path).ok();
    let size_label = match metadata.as_ref() {
        Some(meta) => {
            let bytes = meta.len();
            const KB: u64 = 1024;
            const MB: u64 = KB * 1024;
            const GB: u64 = MB * 1024;
            if bytes >= GB {
                format!("{:.2} GB", bytes as f64 / GB as f64)
            } else if bytes >= MB {
                format!("{:.2} MB", bytes as f64 / MB as f64)
            } else if bytes >= KB {
                format!("{:.2} KB", bytes as f64 / KB as f64)
            } else {
                format!("{} B", bytes)
            }
        }
        None => "n/a".to_string(),
    };
    let modified_label = metadata
        .as_ref()
        .and_then(|meta| meta.modified().ok())
        .map(|modified| {
            DateTime::<Local>::from(modified)
                .format("%Y-%m-%d")
                .to_string()
        })
        .unwrap_or_else(|| "unknown".to_string());
    format!("{} | Modified {}", size_label, modified_label)
}

pub(crate) fn file_resolution_and_size(path: &Path) -> String {
    let size_modified = file_size_and_modified(path);
    match image::image_dimensions(path) {
        Ok((w, h)) => format!("{}x{} | {}", w, h, size_modified),
        Err(_) => size_modified,
    }
}

pub(crate) fn sift_info_line(
    sift_info_by_file: &HashMap<String, SiftInfo>,
    file_name: &str,
) -> String {
    let Some(info) = sift_info_by_file.get(file_name) else {
        return "SIFT: n/a".to_string();
    };
    if !valid_sift_link(info) {
        return "SIFT: no valid link".to_string();
    }
    format!(
        "SIFT: score {:.2}, inliers {}, ratio {:.2}",
        info.score.unwrap_or(0.0),
        info.inliers.unwrap_or(0),
        info.inlier_ratio.unwrap_or(0.0)
    )
}

pub(crate) fn wrapping_monospace_path(ui: &mut egui::Ui, text: &str) {
    let label = egui::Label::new(egui::RichText::new(text).monospace())
        .wrap()
        .selectable(true);
    ui.add(label);
}
