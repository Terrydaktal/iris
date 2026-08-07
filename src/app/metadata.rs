use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;

pub(crate) fn extract_system_block(exif_data: &str) -> String {
    let mut lines = Vec::new();
    let mut in_system = false;
    for line in exif_data.lines() {
        if line.contains("---- System ----") {
            in_system = true;
            continue;
        }
        if in_system {
            if line.starts_with("----") {
                break;
            }
            lines.push(line.to_string());
        }
    }

    if lines.is_empty() {
        return "No System metadata available".to_string();
    }

    let mut result = "---- System ----\n".to_string();
    for line in lines {
        let cleaned_line = if line.contains(']') {
            let parts: Vec<&str> = line.splitn(2, ']').collect();
            if parts.len() > 1 {
                parts[1].trim()
            } else {
                line.trim()
            }
        } else {
            line.trim()
        };

        if !cleaned_line.is_empty() {
            if let Some(colon_pos) = cleaned_line.find(':') {
                let key = cleaned_line[0..colon_pos].trim();
                let val = cleaned_line[colon_pos + 1..].trim();
                result.push_str(&format!("     - {:<32} : {}\n", key, val));
            } else {
                result.push_str(&format!("     - {}\n", cleaned_line));
            }
        }
    }
    result
}

pub(crate) fn resolve_exiftool_path() -> Option<PathBuf> {
    static EXIFTOOL_PATH: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    EXIFTOOL_PATH
        .get_or_init(|| {
            if let Some(path) = std::env::var_os("IRIS_EXIFTOOL").map(PathBuf::from) {
                if path.is_file() {
                    return Some(path);
                }
            }

            if let Some(paths) = std::env::var_os("PATH") {
                for dir in std::env::split_paths(&paths) {
                    let candidate = dir.join("exiftool");
                    if candidate.is_file() {
                        return Some(candidate);
                    }
                }
            }

            [
                "/usr/bin/exiftool",
                "/usr/bin/vendor_perl/exiftool",
                "/usr/local/bin/exiftool",
                "/bin/exiftool",
            ]
            .iter()
            .map(PathBuf::from)
            .find(|path| path.is_file())
        })
        .clone()
}

pub(crate) fn resolve_ffprobe_path() -> Option<PathBuf> {
    static FFPROBE_PATH: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    FFPROBE_PATH
        .get_or_init(|| {
            if let Some(path) = std::env::var_os("IRIS_FFPROBE").map(PathBuf::from) {
                if path.is_file() {
                    return Some(path);
                }
            }

            if let Some(paths) = std::env::var_os("PATH") {
                for dir in std::env::split_paths(&paths) {
                    let candidate = dir.join("ffprobe");
                    if candidate.is_file() {
                        return Some(candidate);
                    }
                }
            }

            ["/usr/bin/ffprobe", "/usr/local/bin/ffprobe", "/bin/ffprobe"]
                .iter()
                .map(PathBuf::from)
                .find(|path| path.is_file())
        })
        .clone()
}

pub(crate) fn load_ffprobe_metadata(path: &Path) -> String {
    if let Some(ffprobe_path) = resolve_ffprobe_path() {
        match Command::new(&ffprobe_path)
            .args([
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-show_chapters",
                "-show_programs",
                "-show_data",
                "-show_private_data",
                "-print_format",
                "json",
            ])
            .arg(path)
            .output()
        {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                if !stdout.trim().is_empty() {
                    stdout
                } else {
                    let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
                    if stderr.is_empty() {
                        format!("ffprobe produced no output for {}", path.display())
                    } else {
                        format!("ffprobe error: {}", stderr)
                    }
                }
            }
            Err(e) => format!("Error running ffprobe at {}: {}", ffprobe_path.display(), e),
        }
    } else {
        "Error running ffprobe: executable not found. Set IRIS_FFPROBE or install ffmpeg."
            .to_string()
    }
}

#[derive(Clone, Copy)]
pub(crate) struct VideoMetadata {
    pub(crate) duration_sec: Option<f32>,
    pub(crate) width: Option<u32>,
    pub(crate) height: Option<u32>,
}

pub(crate) fn load_video_metadata(path: &Path) -> Option<VideoMetadata> {
    let ffprobe_path = resolve_ffprobe_path()?;
    let out = Command::new(ffprobe_path)
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let json: Value = serde_json::from_slice(&out.stdout).ok()?;
    let duration_sec = json
        .get("format")
        .and_then(|format| format.get("duration"))
        .and_then(|value| {
            value
                .as_f64()
                .map(|duration| duration as f32)
                .or_else(|| value.as_str()?.parse::<f32>().ok())
        })
        .filter(|duration| duration.is_finite() && *duration > 0.0);
    let video_stream = json
        .get("streams")
        .and_then(Value::as_array)
        .and_then(|streams| {
            streams
                .iter()
                .find(|stream| stream.get("codec_type").and_then(Value::as_str) == Some("video"))
        });
    let width = video_stream
        .and_then(|stream| stream.get("width"))
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok());
    let height = video_stream
        .and_then(|stream| stream.get("height"))
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok());
    (duration_sec.is_some() || width.is_some() || height.is_some()).then_some(VideoMetadata {
        duration_sec,
        width,
        height,
    })
}

pub(crate) fn format_video_duration(duration_sec: f32) -> String {
    let total = duration_sec.max(0.0).round() as u64;
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let seconds = total % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes:02}:{seconds:02}")
    }
}
