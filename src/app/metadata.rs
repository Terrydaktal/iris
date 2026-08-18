use super::binary::{FileChunk, parse_bmp, parse_generic, parse_jpeg, parse_png, parse_webp};
use serde_json::Value;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub(crate) fn run_metadata_worker(
    queue: Arc<crate::app::MetadataJobQueue>,
    tx: std::sync::mpsc::Sender<crate::app::MetadataLoadResult>,
    ctx_shared: Arc<Mutex<Option<eframe::egui::Context>>>,
    diagnostics: crate::app::DiagnosticState,
) {
    let operation_id = diagnostics.next_operation_id();
    diagnostics.task_started("metadata_worker", operation_id);
    loop {
        let request = {
            #[cfg(feature = "diagnostics")]
            let lock_started = Instant::now();
            let mut state = match queue.state.lock() {
                Ok(state) => state,
                Err(_) => {
                    diagnostics.task_failed("metadata_worker", operation_id, "queue_lock_poisoned");
                    return;
                }
            };
            while state.pending.is_none() && !state.shutdown {
                state = match queue.wake.wait(state) {
                    Ok(state) => state,
                    Err(_) => {
                        diagnostics.task_failed(
                            "metadata_worker",
                            operation_id,
                            "queue_wait_failed",
                        );
                        return;
                    }
                };
            }
            #[cfg(feature = "diagnostics")]
            diagnostics.record_lock_wait("metadata_queue", lock_started.elapsed());
            if state.shutdown {
                diagnostics.task_completed("metadata_worker", operation_id);
                return;
            }
            state.pending.take()
        };

        let Some(request) = request else {
            continue;
        };
        let result = load_file_metadata(
            request.logical_path,
            request.inspect_path,
            request.generation,
            request.load_exif,
            request.load_layout,
        );
        if tx.send(result).is_err() {
            diagnostics.task_completed("metadata_worker", operation_id);
            return;
        }
        #[cfg(feature = "diagnostics")]
        let ctx_lock_started = Instant::now();
        if let Ok(lock) = ctx_shared.lock() {
            #[cfg(feature = "diagnostics")]
            diagnostics.record_lock_wait("metadata_context", ctx_lock_started.elapsed());
            if let Some(ctx) = lock.as_ref() {
                ctx.request_repaint();
            }
        } else {
            diagnostics.task_failed_with_code(
                "metadata_worker",
                operation_id,
                "context_lock_poisoned",
                "metadata repaint context lock was poisoned",
            );
            return;
        }
    }
}

fn command_output_with_timeout(mut command: Command, timeout: Duration) -> std::io::Result<Output> {
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command.spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| std::io::Error::other("metadata command stdout was not captured"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| std::io::Error::other("metadata command stderr was not captured"))?;
    const MAX_OUTPUT_BYTES: u64 = 16 * 1024 * 1024;
    let stdout_reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = stdout.take(MAX_OUTPUT_BYTES).read_to_end(&mut bytes);
        bytes
    });
    let stderr_reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = stderr.take(MAX_OUTPUT_BYTES).read_to_end(&mut bytes);
        bytes
    });
    let deadline = Instant::now() + timeout;
    loop {
        if child.try_wait()?.is_some() {
            let status = child.wait()?;
            return Ok(Output {
                status,
                stdout: stdout_reader.join().unwrap_or_default(),
                stderr: stderr_reader.join().unwrap_or_default(),
            });
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "metadata command timed out",
            ));
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

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
        let mut command = Command::new(&ffprobe_path);
        command
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
            .arg(path);
        match command_output_with_timeout(command, Duration::from_secs(30)) {
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
    let mut command = Command::new(ffprobe_path);
    command
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height",
            "-of",
            "json",
        ])
        .arg(path);
    let out = command_output_with_timeout(command, Duration::from_secs(15)).ok()?;
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

pub(crate) fn load_file_metadata(
    logical_path: PathBuf,
    inspect_path: PathBuf,
    generation: u64,
    load_exif: bool,
    load_layout: bool,
) -> crate::app::MetadataLoadResult {
    let exif_data = if load_exif {
        let exiftool_data = if !inspect_path.exists() {
            format!("Resolved file does not exist: {}", inspect_path.display())
        } else if let Some(exiftool_path) = resolve_exiftool_path() {
            let mut command = Command::new(&exiftool_path);
            command.args(["-a", "-u", "-g1", "-H"]).arg(&inspect_path);
            match command_output_with_timeout(command, Duration::from_secs(30)) {
                Ok(out) => {
                    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                    if !stdout.trim().is_empty() {
                        stdout
                    } else {
                        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
                        if stderr.is_empty() {
                            format!("exiftool produced no output for {}", inspect_path.display())
                        } else {
                            format!("exiftool error: {stderr}")
                        }
                    }
                }
                Err(error) => format!("Error running exiftool: {error}"),
            }
        } else {
            "Error running exiftool: executable not found. Set IRIS_EXIFTOOL or install exiftool."
                .to_string()
        };
        if inspect_path.exists() && super::is_video_path(&inspect_path) {
            format!(
                "{}\n\n---- FFprobe JSON ----\n{}",
                exiftool_data.trim_end(),
                load_ffprobe_metadata(&inspect_path)
            )
        } else {
            exiftool_data
        }
    } else {
        String::new()
    };

    let chunks = if !load_layout || !inspect_path.exists() {
        Vec::new()
    } else if super::is_video_path(&inspect_path) {
        vec![FileChunk {
            name: "Video File".to_string(),
            offset: 0,
            length: std::fs::metadata(&inspect_path)
                .map(|metadata| metadata.len().min(usize::MAX as u64) as usize)
                .unwrap_or(0),
            description: "Video files do not use the image binary layout parser.".to_string(),
            color: eframe::egui::Color32::from_rgb(140, 150, 170),
            parsed_data: "Use Raw EXIF to view exiftool and ffprobe metadata for this video."
                .to_string(),
        }]
    } else {
        let mut bytes = Vec::new();
        match std::fs::File::open(&inspect_path)
            .and_then(|mut file| file.by_ref().take(64 * 1024 * 1024).read_to_end(&mut bytes))
        {
            Ok(_) => {
                let mut chunks = if let Some(chunks) = parse_png(&bytes) {
                    chunks
                } else if let Some(chunks) = parse_jpeg(&bytes) {
                    chunks
                } else if let Some(chunks) = parse_webp(&bytes) {
                    chunks
                } else if let Some(chunks) = parse_bmp(&bytes) {
                    chunks
                } else {
                    parse_generic(&bytes)
                };
                if load_exif {
                    chunks.insert(
                        0,
                        FileChunk {
                            name: "System Metadata".to_string(),
                            offset: 0,
                            length: 0,
                            description: "Operating system-level file attributes, timestamps, and permissions."
                                .to_string(),
                            color: eframe::egui::Color32::from_rgb(140, 150, 170),
                            parsed_data: extract_system_block(&exif_data),
                        },
                    );
                }
                chunks
            }
            Err(_) => Vec::new(),
        }
    };

    crate::app::MetadataLoadResult {
        generation,
        path: logical_path,
        exif_data,
        chunks,
        load_exif,
        load_layout,
    }
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
