use eframe::egui;
use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, ListArray, RecordBatch,
    RecordBatchIterator,
    StringArray, StructArray,
};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::table::AddDataMode;
use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;
use serde_json::Value;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

fn get_socket_path() -> PathBuf {
    let username = std::env::var("USER")
        .unwrap_or_else(|_| std::env::var("USERNAME").unwrap_or_else(|_| "default".to_string()));
    std::env::temp_dir().join(format!("iris_{}.sock", username))
}

fn main() -> eframe::Result {
    let args: Vec<String> = std::env::args().collect();
    let mut reuse_window = false;
    let mut no_daemon = false;
    let mut image_arg = None;

    for arg in args.iter().skip(1) {
        if arg == "--same-window" || arg == "-s" || arg == "--reuse-window" || arg == "-r" {
            reuse_window = true;
        } else if arg == "--new-window" || arg == "-n" {
            // New window is now the default behavior, so this flag is a no-op
        } else if arg == "--no-daemon" {
            no_daemon = true;
        } else {
            image_arg = Some(arg.clone());
        }
    }

    let mut start_on_home_page = false;
    let image_path = match image_arg {
        Some(path_str) => PathBuf::from(&path_str).canonicalize().unwrap_or_else(|_| PathBuf::from(&path_str)),
        None => {
            start_on_home_page = true;
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        }
    };

    let socket_path = get_socket_path();
    let mut socket_active = false;

    // Check if another instance is already actively listening on the socket
    if let Ok(mut stream) = std::os::unix::net::UnixStream::connect(&socket_path) {
        socket_active = true;
        if reuse_window {
            use std::io::Write;
            let path_str = image_path.to_string_lossy().to_string();
            if let Err(e) = stream.write_all(path_str.as_bytes()) {
                eprintln!("Error sending path to existing instance: {}", e);
            } else {
                println!("Opened {} in the existing window.", path_str);
                std::process::exit(0);
            }
        }
    }

    // Since we need to spawn the GUI, daemonize if we are not already the background daemon child
    let is_daemon_child = std::env::var("IRIS_DAEMON").is_ok();
    if !is_daemon_child && !no_daemon {
        let mut cmd = Command::new(std::env::current_exe().unwrap());
        for arg in args.iter().skip(1) {
            cmd.arg(arg);
        }
        cmd.env("IRIS_DAEMON", "1");
        cmd.stdin(std::process::Stdio::null());
        cmd.stdout(std::process::Stdio::null());
        cmd.stderr(std::process::Stdio::null());
        match cmd.spawn() {
            Ok(_) => {
                std::process::exit(0);
            }
            Err(e) => {
                eprintln!("Failed to spawn background instance: {}", e);
            }
        }
    }

    let ctx_shared: Arc<Mutex<Option<egui::Context>>> = Arc::new(Mutex::new(None));
    let (tx, rx) = std::sync::mpsc::channel::<PathBuf>();
    let bind_socket = !socket_active;

    if bind_socket {
        if socket_path.exists() {
            let _ = std::fs::remove_file(&socket_path);
        }
        
        let tx_clone = tx.clone();
        let socket_path_clone = socket_path.clone();
        let ctx_shared_clone = ctx_shared.clone();
        std::thread::spawn(move || {
            if let Ok(listener) = std::os::unix::net::UnixListener::bind(&socket_path_clone) {
                for stream in listener.incoming() {
                    if let Ok(mut stream) = stream {
                        use std::io::Read;
                        let mut buf = Vec::new();
                        if let Ok(_) = stream.read_to_end(&mut buf) {
                            if !buf.is_empty() {
                                if let Ok(path_str) = String::from_utf8(buf) {
                                    let new_path = PathBuf::from(path_str);
                                    let _ = tx_clone.send(new_path);
                                    if let Ok(lock) = ctx_shared_clone.lock() {
                                        if let Some(ctx) = lock.as_ref() {
                                            ctx.request_repaint();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_app_id("iris")
            .with_inner_size([1200.0, 800.0])
            .with_title("Iris"),
        ..Default::default()
    };

    let rx_shared = Arc::new(Mutex::new(Some(rx)));
    let rx_shared_clone = rx_shared.clone();
    let image_path_clone = image_path.clone();
    let ctx_shared_clone = ctx_shared.clone();

    let mut result = eframe::run_native(
        "iris",
        options.clone(),
        Box::new(move |cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            let rx_taken = rx_shared_clone.lock().unwrap().take().unwrap();
            Ok(Box::new(ImageViewer::new(image_path_clone, rx_taken, ctx_shared_clone, start_on_home_page)))
        }),
    );

    if result.is_err() {
        // Fallback to X11 backend if Wayland graphics context fails (e.g., NVIDIA EGL OutOfMemory)
        unsafe { std::env::set_var("WINIT_UNIX_BACKEND", "x11"); }
        result = eframe::run_native(
            "iris",
            options,
            Box::new(move |cc| {
                egui_extras::install_image_loaders(&cc.egui_ctx);
                let rx_taken = rx_shared.lock().unwrap().take().unwrap();
                Ok(Box::new(ImageViewer::new(image_path, rx_taken, ctx_shared, start_on_home_page)))
            }),
        );
    }

    if bind_socket {
        let _ = std::fs::remove_file(&socket_path);
    }

    result
}

fn extract_system_block(exif_data: &str) -> String {
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
                let val = cleaned_line[colon_pos+1..].trim();
                result.push_str(&format!("     - {:<32} : {}\n", key, val));
            } else {
                result.push_str(&format!("     - {}\n", cleaned_line));
            }
        }
    }
    result
}

fn resolve_exiftool_path() -> Option<PathBuf> {
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

fn resolve_ffprobe_path() -> Option<PathBuf> {
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

fn load_ffprobe_metadata(path: &Path) -> String {
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
        "Error running ffprobe: executable not found. Set IRIS_FFPROBE or install ffmpeg.".to_string()
    }
}

#[derive(PartialEq, Clone, Copy)]
enum SidePanelMode {
    Layout,
    Exif,
    Duplicates,
}

#[derive(Clone)]
struct FileChunk {
    name: String,
    offset: usize,
    length: usize,
    description: String,
    color: egui::Color32,
    parsed_data: String,
}

fn generate_hex_dump(chunk_bytes: &[u8], absolute_offset: usize, max_len: usize) -> String {
    let len_to_dump = std::cmp::min(chunk_bytes.len(), max_len);
    let dumped_bytes = &chunk_bytes[0..len_to_dump];
    
    let mut dump = String::new();
    let mut line_offset = 0;
    while line_offset < dumped_bytes.len() {
        let line_end = std::cmp::min(dumped_bytes.len(), line_offset + 16);
        let line = &dumped_bytes[line_offset..line_end];
        
        let mut hex_part = String::new();
        for (i, &b) in line.iter().enumerate() {
            hex_part.push_str(&format!("{:02X} ", b));
            if i == 7 {
                hex_part.push(' ');
            }
        }
        while hex_part.len() < 50 {
            hex_part.push(' ');
        }
        
        let mut ascii_part = String::new();
        for &b in line {
            if b >= 32 && b <= 126 {
                ascii_part.push(b as char);
            } else {
                ascii_part.push('.');
            }
        }
        
        dump.push_str(&format!("0x{:04X}:   {} |{}|\n", absolute_offset + line_offset, hex_part, ascii_part));
        line_offset += 16;
    }
    if chunk_bytes.len() > max_len {
        dump.push_str("... (truncated) ...\n");
    }
    dump
}

fn parse_png(bytes: &[u8]) -> Option<Vec<FileChunk>> {
    if bytes.len() < 8 || &bytes[0..8] != [137, 80, 78, 71, 13, 10, 26, 10] {
        return None;
    }
    
    let mut chunks = vec![
        FileChunk {
            name: "PNG Signature".to_string(),
            offset: 0,
            length: 8,
            description: "8-byte magic number identifying the file as a PNG image.".to_string(),
            color: egui::Color32::from_rgb(120, 110, 255), // Indigo
            parsed_data: generate_hex_dump(&bytes[0..8], 0, 1024),
        }
    ];
    
    let mut pos = 8;
    while pos + 8 <= bytes.len() {
        let len = u32::from_be_bytes([bytes[pos], bytes[pos+1], bytes[pos+2], bytes[pos+3]]) as usize;
        let type_bytes = &bytes[pos+4..pos+8];
        let chunk_type = String::from_utf8_lossy(type_bytes).to_string();
        
        let description = match chunk_type.as_str() {
            "IHDR" => "Image Header: Contains width, height, bit depth, color type, compression, filter, and interlace methods.".to_string(),
            "PLTE" => "Palette: Contains the list of colors used in an indexed-color image.".to_string(),
            "IDAT" => "Image Data: Contains the actual compressed image pixel data.".to_string(),
            "IEND" => "Image End: Marks the end of the PNG file structure.".to_string(),
            "eXIf" => "EXIF Metadata: Contains camera settings, location, and other EXIF information.".to_string(),
            "tEXt" => "Textual Data: Uncompressed key-value metadata text.".to_string(),
            "zTXt" => "Compressed Textual Data: Key-value text metadata, compressed with zlib.".to_string(),
            "iTXt" => "International Textual Data: UTF-8 translated textual metadata.".to_string(),
            "cHRM" => "Primary Chromaticities: Chromaticity coordinates of the red, green, blue primaries and white point.".to_string(),
            "gAMA" => "Image Gamma: Specifies the relationship between the image samples and desired display output intensity.".to_string(),
            "iCCP" => "ICC Profile: Embedded International Color Consortium color profile descriptor.".to_string(),
            "sRGB" => "Standard RGB: Indicates that the image uses the standard sRGB color space.".to_string(),
            "sBIT" => "Significant Bits: Specifies the original color resolution of the source image.".to_string(),
            "pHYs" => "Physical Pixel Dimensions: Specifies the intended resolution/pixel aspect ratio of the image.".to_string(),
            "tIME" => "Last Modification Time: Stores the date and time of the last modification to the image.".to_string(),
            "tRNS" => "Transparency: Contains transparency values for indexed colors or alpha palette.".to_string(),
            "bKGD" => "Background Color: Specifies a default background color to use when displaying the image.".to_string(),
            "hIST" => "Palette Histogram: Records the usage frequency of each color in the palette.".to_string(),
            _ => "Other Chunk: Custom/ancillary chunk containing extension or auxiliary information.".to_string(),
        };
        
        let color = match chunk_type.as_str() {
            "IHDR" => egui::Color32::from_rgb(255, 100, 100), // Coral
            "PLTE" => egui::Color32::from_rgb(255, 180, 50),  // Amber
            "IDAT" => egui::Color32::from_rgb(50, 200, 120),  // Emerald
            "IEND" => egui::Color32::from_rgb(180, 180, 180), // Gray
            "eXIf" => egui::Color32::from_rgb(255, 110, 220), // Magenta/Pink
            "tEXt" | "zTXt" | "iTXt" => egui::Color32::from_rgb(50, 160, 250), // Sky Blue
            _ => egui::Color32::from_rgb(200, 140, 255), // Lavender
        };

        let parsed_data = match chunk_type.as_str() {
            "IHDR" if len >= 13 && pos + 21 <= bytes.len() => {
                 let w = u32::from_be_bytes([bytes[pos+8], bytes[pos+9], bytes[pos+10], bytes[pos+11]]);
                 let h = u32::from_be_bytes([bytes[pos+12], bytes[pos+13], bytes[pos+14], bytes[pos+15]]);
                 let depth = bytes[pos+16];
                 let color = bytes[pos+17];
                 let comp = bytes[pos+18];
                 let filter = bytes[pos+19];
                 let interlace = bytes[pos+20];
                 let color_str = match color {
                     0 => "Grayscale",
                     2 => "Truecolor (RGB)",
                     3 => "Indexed Color",
                     4 => "Grayscale + Alpha",
                     6 => "Truecolor + Alpha (RGBA)",
                     _ => "Unknown",
                 };
                 
                 let english_lines = vec![
                     format!("Dimensions: {} x {}", w, h),
                     format!("Bit Depth: {}", depth),
                     format!("Color Type: {} ({})", color, color_str),
                     format!("Compression: {}", comp),
                     format!("Filter: {}", filter),
                     format!("Interlace: {}", interlace),
                 ];
                 
                 let chunk_end = if pos + 12 > bytes.len() {
                     bytes.len()
                 } else {
                     let max_len = bytes.len() - pos - 12;
                     if len > max_len {
                         bytes.len()
                     } else {
                         pos + len + 12
                     }
                 };
                 let hex_dump = generate_hex_dump(&bytes[pos..chunk_end], pos, 1024);
                 let hex_lines: Vec<&str> = hex_dump.lines().collect();
                 
                 let mut combined = String::new();
                 let max_lines = std::cmp::max(hex_lines.len(), english_lines.len());
                 for i in 0..max_lines {
                     let hex_part = if i < hex_lines.len() { hex_lines[i] } else { "" };
                     let english_part = if i < english_lines.len() { &english_lines[i] } else { "" };
                     
                     if !hex_part.is_empty() || !english_part.is_empty() {
                         if !hex_part.is_empty() {
                             combined.push_str(&format!("{:<80}  # {}\n", hex_part, english_part));
                         } else {
                             combined.push_str(&format!("{:<80}  # {}\n", "", english_part));
                         }
                     }
                 }
                 combined
            }
            "tEXt" | "zTXt" | "iTXt" if len > 0 && pos + 8 + len <= bytes.len() => {
                let chunk_data = &bytes[pos+8..pos+8+len];
                if let Some(null_pos) = chunk_data.iter().position(|&b| b == 0) {
                    let key = String::from_utf8_lossy(&chunk_data[0..null_pos]).to_string();
                    let val = String::from_utf8_lossy(&chunk_data[null_pos+1..]).to_string();
                    format!("{}: {}", key, val)
                } else {
                    String::from_utf8_lossy(chunk_data).to_string()
                }
            }
            "sRGB" if len >= 1 && pos + 9 <= bytes.len() => {
                let intent = bytes[pos+8];
                let intent_str = match intent {
                    0 => "Perceptual",
                    1 => "Relative Colorimetric",
                    2 => "Saturation",
                    3 => "Absolute Colorimetric",
                    _ => "Unknown",
                };
                format!("Rendering Intent: {} ({})", intent, intent_str)
            }
            "pHYs" if len >= 9 && pos + 17 <= bytes.len() => {
                let x = u32::from_be_bytes([bytes[pos+8], bytes[pos+9], bytes[pos+10], bytes[pos+11]]);
                let y = u32::from_be_bytes([bytes[pos+12], bytes[pos+13], bytes[pos+14], bytes[pos+15]]);
                let unit = bytes[pos+16];
                let unit_str = if unit == 1 { "meter" } else { "unknown" };
                format!("Pixels per unit X: {}\nPixels per unit Y: {}\nUnit: {} ({})", x, y, unit, unit_str)
            }
            _ => {
                let chunk_end = if pos + 12 > bytes.len() {
                    bytes.len()
                } else {
                    let max_len = bytes.len() - pos - 12;
                    if len > max_len {
                        bytes.len()
                    } else {
                        pos + len + 12
                    }
                };
                generate_hex_dump(&bytes[pos..chunk_end], pos, 1024)
            }
        };
        
        chunks.push(FileChunk {
            name: format!("{} Chunk", chunk_type),
            offset: pos,
            length: len + 12,
            description,
            color,
            parsed_data,
        });
        
        if pos + 12 > bytes.len() {
            break;
        }
        let max_len = bytes.len() - pos - 12;
        if len > max_len {
            break;
        }
        pos += len + 12;
        if chunk_type == "IEND" {
            break;
        }
    }
    
    Some(chunks)
}

fn parse_jpeg(bytes: &[u8]) -> Option<Vec<FileChunk>> {
    if bytes.len() < 4 || &bytes[0..2] != [0xFF, 0xD8] {
        return None;
    }
    
    let mut chunks = vec![
        FileChunk {
            name: "SOI Marker".to_string(),
            offset: 0,
            length: 2,
            description: "Start of Image: Identifies the beginning of the JPEG stream.".to_string(),
            color: egui::Color32::from_rgb(120, 110, 255), // Indigo
            parsed_data: generate_hex_dump(&bytes[0..2], 0, 1024),
        }
    ];
    
    let mut pos = 2;
    while pos + 2 <= bytes.len() {
        if bytes[pos] != 0xFF {
            let mut next_ff = pos;
            while next_ff + 1 < bytes.len() {
                if bytes[next_ff] == 0xFF && bytes[next_ff + 1] != 0x00 && (bytes[next_ff + 1] < 0xD0 || bytes[next_ff + 1] > 0xD7) {
                    break;
                }
                next_ff += 1;
            }
            let length = next_ff - pos;
            if length > 0 {
                chunks.push(FileChunk {
                    name: "Entropy Coded Scan Data".to_string(),
                    offset: pos,
                    length,
                    description: "Main body of the image containing the compressed Huffman-coded image bitstream.".to_string(),
                    color: egui::Color32::from_rgb(50, 200, 120), // Emerald
                    parsed_data: {
                        let chunk_end = std::cmp::min(bytes.len(), pos + length);
                        generate_hex_dump(&bytes[pos..chunk_end], pos, 1024)
                    },
                });
                pos = next_ff;
                continue;
            }
        }
        
        let marker = bytes[pos + 1];
        if marker == 0xD9 {
            chunks.push(FileChunk {
                name: "EOI Marker".to_string(),
                offset: pos,
                length: 2,
                description: "End of Image: Identifies the termination of the JPEG stream.".to_string(),
                color: egui::Color32::from_rgb(180, 180, 180), // Gray
                parsed_data: generate_hex_dump(&bytes[pos..std::cmp::min(bytes.len(), pos + 2)], pos, 1024),
            });
            break;
        }
        
        if marker == 0xD8 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7) {
            chunks.push(FileChunk {
                name: format!("Marker FF{:02X}", marker),
                offset: pos,
                length: 2,
                description: "JPEG Marker without length payload.".to_string(),
                color: egui::Color32::from_rgb(200, 140, 255),
                parsed_data: generate_hex_dump(&bytes[pos..std::cmp::min(bytes.len(), pos + 2)], pos, 1024),
            });
            pos += 2;
            continue;
        }
        
        if pos + 4 > bytes.len() {
            break;
        }
        let seg_len = u16::from_be_bytes([bytes[pos + 2], bytes[pos + 3]]) as usize;
        
        let name = match marker {
            0xE0 => "APP0 Segment (JFIF)".to_string(),
            0xE1 => "APP1 Segment (EXIF/XMP)".to_string(),
            0xE2 => "APP2 Segment (ICC Profile)".to_string(),
            0xEE => "APP14 Segment (Adobe)".to_string(),
            0xEF => "APP15 Segment".to_string(),
            0xDB => "DQT Segment (Quantization)".to_string(),
            0xC0 => "SOF0 Segment (Baseline DCT)".to_string(),
            0xC2 => "SOF2 Segment (Progressive DCT)".to_string(),
            0xC4 => "DHT Segment (Huffman Tables)".to_string(),
            0xDA => "SOS Segment (Start of Scan)".to_string(),
            0xFE => "COM Segment (Comment)".to_string(),
            _ => format!("APP{:X} Segment", marker & 0x0F),
        };
        
        let description = match marker {
            0xE0 => "Application Segment 0: Contains JFIF/JFXX format parameters and thumbnails.".to_string(),
            0xE1 => "Application Segment 1: Contains EXIF camera metadata and/or XMP copyright/editor info.".to_string(),
            0xE2 => "Application Segment 2: Contains embedded color management ICC profiles.".to_string(),
            0xDB => "Define Quantization Tables: Specifies the compression quantization matrix tables.".to_string(),
            0xC0 => "Start of Frame 0 (Baseline): Specifies image width, height, component count, and sampling rates.".to_string(),
            0xC2 => "Start of Frame 2 (Progressive): Specifies image metadata for a progressive rendering DCT bitstream.".to_string(),
            0xC4 => "Define Huffman Tables: Specifies the entropy tables used for compression decoding.".to_string(),
            0xDA => "Start of Scan: Marks the beginning of the compressed image scan data and components mapping.".to_string(),
            0xFE => "Comment: Contains arbitrary metadata textual comments.".to_string(),
            _ => "JPEG Application or Extension Segment.".to_string(),
        };
        
        let color = match marker {
            0xE1 => egui::Color32::from_rgb(255, 110, 220), // Magenta (EXIF)
            0xDB => egui::Color32::from_rgb(255, 180, 50),  // Amber
            0xC0 | 0xC2 => egui::Color32::from_rgb(255, 100, 100), // Coral
            0xC4 => egui::Color32::from_rgb(50, 160, 250),  // Sky Blue
            0xDA => egui::Color32::from_rgb(50, 200, 120),  // Emerald
            _ => egui::Color32::from_rgb(200, 140, 255),    // Lavender
        };

        let parsed_data = match marker {
            0xC0 | 0xC2 if seg_len >= 8 && pos + 10 <= bytes.len() => {
                let precision = bytes[pos+4];
                let h = u16::from_be_bytes([bytes[pos+5], bytes[pos+6]]);
                let w = u16::from_be_bytes([bytes[pos+7], bytes[pos+8]]);
                let components = bytes[pos+9];
                
                let english_lines = vec![
                    format!("Dimensions: {} x {}", w, h),
                    format!("Sample Precision: {} bits", precision),
                    format!("Number of Components: {}", components),
                ];
                
                let chunk_end = if pos + 2 > bytes.len() {
                     bytes.len()
                 } else {
                     let max_len = bytes.len() - pos - 2;
                     if seg_len > max_len {
                         bytes.len()
                     } else {
                         pos + seg_len + 2
                     }
                 };
                let hex_dump = generate_hex_dump(&bytes[pos..chunk_end], pos, 1024);
                let hex_lines: Vec<&str> = hex_dump.lines().collect();
                
                let mut combined = String::new();
                let max_lines = std::cmp::max(hex_lines.len(), english_lines.len());
                for i in 0..max_lines {
                    let hex_part = if i < hex_lines.len() { hex_lines[i] } else { "" };
                    let english_part = if i < english_lines.len() { &english_lines[i] } else { "" };
                    
                    if !hex_part.is_empty() || !english_part.is_empty() {
                        if !hex_part.is_empty() {
                            combined.push_str(&format!("{:<80}  # {}\n", hex_part, english_part));
                        } else {
                            combined.push_str(&format!("{:<80}  # {}\n", "", english_part));
                        }
                    }
                }
                combined
            }
            0xFE if seg_len >= 2 && pos + 4 + seg_len - 2 <= bytes.len() => {
                let comment_bytes = &bytes[pos+4..pos+2+seg_len];
                String::from_utf8_lossy(comment_bytes).to_string()
            }
            _ => {
                let chunk_end = if pos + 2 > bytes.len() {
                    bytes.len()
                } else {
                    let max_len = bytes.len() - pos - 2;
                    if seg_len > max_len {
                        bytes.len()
                    } else {
                        pos + seg_len + 2
                    }
                };
                generate_hex_dump(&bytes[pos..chunk_end], pos, 1024)
            }
        };
        
        chunks.push(FileChunk {
            name,
            offset: pos,
            length: seg_len + 2,
            description,
            color,
            parsed_data,
        });
        
        if pos + 2 > bytes.len() {
            break;
        }
        let max_len = bytes.len() - pos - 2;
        if seg_len > max_len {
            break;
        }
        pos += seg_len + 2;
    }
    
    Some(chunks)
}

fn parse_webp(bytes: &[u8]) -> Option<Vec<FileChunk>> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WEBP" {
        return None;
    }
    
    let mut chunks = vec![
        FileChunk {
            name: "RIFF Header".to_string(),
            offset: 0,
            length: 12,
            description: "RIFF Container Header: Identifies the file as a WEBP resource.".to_string(),
            color: egui::Color32::from_rgb(120, 110, 255), // Indigo
            parsed_data: generate_hex_dump(&bytes[0..12], 0, 1024),
        }
    ];
    
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let type_bytes = &bytes[pos..pos+4];
        let chunk_type = String::from_utf8_lossy(type_bytes).to_string();
        
        let len = u32::from_le_bytes([bytes[pos+4], bytes[pos+5], bytes[pos+6], bytes[pos+7]]) as usize;
        
        let description = match chunk_type.as_str() {
            "VP8 " => "Lossy Image Bitstream: Contains lossy encoded image pixel data.".to_string(),
            "VP8L" => "Lossless Image Bitstream: Contains lossless encoded image pixel data.".to_string(),
            "VP8X" => "Extended File Header: Specifies whether the image has an alpha channel, animation, ICC profile, or EXIF metadata.".to_string(),
            "ANIM" => "Animation Control: Contains global animation properties (background color, loop count).".to_string(),
            "ANMF" => "Animation Frame: Contains frame size, coordinates, duration, and frame data.".to_string(),
            "EXIF" => "EXIF Metadata: Contains camera settings, location, and other EXIF information.".to_string(),
            "ICCP" => "ICC Profile: Embedded color management profile descriptor.".to_string(),
            "ALPH" => "Alpha Channel Data: Contains transparency/alpha channel bitstream.".to_string(),
            _ => "WEBP ancillary or custom data chunk.".to_string(),
        };
        
        let color = match chunk_type.as_str() {
            "VP8X" => egui::Color32::from_rgb(255, 100, 100), // Coral
            "VP8 " | "VP8L" | "ANMF" => egui::Color32::from_rgb(50, 200, 120), // Emerald
            "EXIF" => egui::Color32::from_rgb(255, 110, 220), // Magenta (EXIF)
            "ICCP" => egui::Color32::from_rgb(255, 180, 50),  // Amber
            "ALPH" => egui::Color32::from_rgb(50, 160, 250),  // Sky Blue
            _ => egui::Color32::from_rgb(200, 140, 255),    // Lavender
        };

        let parsed_data = match chunk_type.as_str() {
            "VP8X" if len >= 10 && pos + 18 <= bytes.len() => {
                let flags = bytes[pos+8];
                let has_icc = (flags & 32) != 0;
                let has_alpha = (flags & 16) != 0;
                let has_exif = (flags & 8) != 0;
                let has_xmp = (flags & 4) != 0;
                let has_anim = (flags & 2) != 0;
                format!(
                    "Extended Features:\n- ICC Profile: {}\n- Alpha channel: {}\n- EXIF metadata: {}\n- XMP metadata: {}\n- Animation: {}",
                    has_icc, has_alpha, has_exif, has_xmp, has_anim
                )
            }
             _ => {
                 let chunk_end = if pos + 8 > bytes.len() {
                     bytes.len()
                 } else {
                     let max_len = bytes.len() - pos - 8;
                     if len > max_len {
                         bytes.len()
                     } else {
                         pos + len + 8
                     }
                 };
                 generate_hex_dump(&bytes[pos..chunk_end], pos, 1024)
             }
        };
        
        chunks.push(FileChunk {
            name: format!("{} Chunk", chunk_type),
            offset: pos,
            length: len + 8,
            description,
            color,
            parsed_data,
        });
        
        if pos + 8 > bytes.len() {
            break;
        }
        let max_len = bytes.len() - pos - 8;
        if len > max_len {
            break;
        }
        pos += len + 8;
        if len % 2 == 1 {
            if pos < bytes.len() {
                pos += 1;
            } else {
                break;
            }
        }
    }
    
    Some(chunks)
}


fn parse_bmp(bytes: &[u8]) -> Option<Vec<FileChunk>> {
    if bytes.len() < 14 || &bytes[0..2] != b"BM" {
        return None;
    }
    
    let file_size = u32::from_le_bytes([bytes[2], bytes[3], bytes[4], bytes[5]]) as usize;
    let pixel_array_offset = u32::from_le_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]) as usize;
    
    let file_header_desc = format!(
        "BMP File Header: Magic signature 'BM', total file size {} bytes, and pixel array starting at offset 0x{:08X}.",
        file_size, pixel_array_offset
    );
    
    let file_header_chunk_end = 14;
    let file_header_hex = generate_hex_dump(&bytes[0..file_header_chunk_end], 0, 1024);
    let fh_hex_lines: Vec<&str> = file_header_hex.lines().collect();
    let fh_english = vec![
        "Signature: BM".to_string(),
        format!("File Size: {} bytes", file_size),
        format!("Pixel Array Offset: 0x{:08X}", pixel_array_offset),
    ];
    let mut fh_combined = String::new();
    let fh_max = std::cmp::max(fh_hex_lines.len(), fh_english.len());
    for i in 0..fh_max {
        let hex = if i < fh_hex_lines.len() { fh_hex_lines[i] } else { "" };
        let eng = if i < fh_english.len() { &fh_english[i] } else { "" };
        if !hex.is_empty() || !eng.is_empty() {
            fh_combined.push_str(&format!("{:<80}  # {}\n", hex, eng));
        }
    }
    
    let mut chunks = vec![
        FileChunk {
            name: "BMP File Header".to_string(),
            offset: 0,
            length: 14,
            description: file_header_desc,
            color: egui::Color32::from_rgb(120, 110, 255), // Indigo
            parsed_data: fh_combined,
        }
    ];
    
    if bytes.len() >= 18 {
        let dib_size = u32::from_le_bytes([bytes[14], bytes[15], bytes[16], bytes[17]]) as usize;
        if dib_size <= bytes.len() - 14 {
            let dib_end = 14 + dib_size;
            let mut dib_english = vec![
                format!("DIB Header Size: {} bytes", dib_size),
            ];
            
            let mut dib_desc = format!("DIB Header (Size: {}): Specifies the size of the DIB, image dimensions, bit depth, compression, and color details.", dib_size);
            
            if dib_size >= 40 && bytes.len() >= 54 {
                let w = i32::from_le_bytes([bytes[18], bytes[19], bytes[20], bytes[21]]);
                let h = i32::from_le_bytes([bytes[22], bytes[23], bytes[24], bytes[25]]);
                let planes = u16::from_le_bytes([bytes[26], bytes[27]]);
                let bpp = u16::from_le_bytes([bytes[28], bytes[29]]);
                let comp = u32::from_le_bytes([bytes[30], bytes[31], bytes[32], bytes[33]]);
                let img_size = u32::from_le_bytes([bytes[34], bytes[35], bytes[36], bytes[37]]);
                let colors_used = u32::from_le_bytes([bytes[46], bytes[47], bytes[48], bytes[49]]);
                
                let comp_str = match comp {
                    0 => "BI_RGB (uncompressed)",
                    1 => "BI_RLE8 (RLE 8-bit)",
                    2 => "BI_RLE4 (RLE 4-bit)",
                    3 => "BI_BITFIELDS",
                    4 => "BI_JPEG",
                    5 => "BI_PNG",
                    _ => "Unknown",
                };
                
                dib_desc = format!(
                    "DIB Header: {}x{} pixels, {} bits per pixel, compressed with {}.",
                    w, h, bpp, comp_str
                );
                
                dib_english = vec![
                    format!("Header Size: {} bytes", dib_size),
                    format!("Width: {} px", w),
                    format!("Height: {} px", h),
                    format!("Planes: {}", planes),
                    format!("Bit Depth: {} bpp", bpp),
                    format!("Compression: {} ({})", comp, comp_str),
                    format!("Image Size: {} bytes", img_size),
                    format!("Colors in Palette: {}", colors_used),
                ];
            }
            
            let dib_hex = generate_hex_dump(&bytes[14..dib_end], 14, 1024);
            let dib_hex_lines: Vec<&str> = dib_hex.lines().collect();
            let mut dib_combined = String::new();
            let dib_max = std::cmp::max(dib_hex_lines.len(), dib_english.len());
            for i in 0..dib_max {
                let hex = if i < dib_hex_lines.len() { dib_hex_lines[i] } else { "" };
                let eng = if i < dib_english.len() { &dib_english[i] } else { "" };
                if !hex.is_empty() || !eng.is_empty() {
                    dib_combined.push_str(&format!("{:<80}  # {}\n", hex, eng));
                }
            }
            
            chunks.push(FileChunk {
                name: "DIB Header".to_string(),
                offset: 14,
                length: dib_size,
                description: dib_desc,
                color: egui::Color32::from_rgb(255, 100, 100), // Coral
                parsed_data: dib_combined,
            });
            
            if pixel_array_offset > dib_end && pixel_array_offset <= bytes.len() {
                let palette_len = pixel_array_offset - dib_end;
                let palette_end = pixel_array_offset;
                
                let num_colors = palette_len / 4;
                let palette_desc = format!(
                    "Color Palette: Contains {} color lookup table entries (4 bytes each: Blue, Green, Red, Reserved).",
                    num_colors
                );
                
                let palette_hex = generate_hex_dump(&bytes[dib_end..palette_end], dib_end, 1024);
                
                chunks.push(FileChunk {
                    name: "Color Palette".to_string(),
                    offset: dib_end,
                    length: palette_len,
                    description: palette_desc,
                    color: egui::Color32::from_rgb(255, 180, 50), // Amber
                    parsed_data: palette_hex,
                });
            }
        }
    }
    
    if pixel_array_offset < bytes.len() {
        let pixel_array_len = bytes.len() - pixel_array_offset;
        let pixel_array_desc = format!(
            "Pixel Array: Main body of the image containing raw compressed or uncompressed pixel color indices/values starting at offset 0x{:08X}.",
            pixel_array_offset
        );
        
        let pixel_array_hex = generate_hex_dump(&bytes[pixel_array_offset..], pixel_array_offset, 1024);
        
        chunks.push(FileChunk {
            name: "Pixel Array".to_string(),
            offset: pixel_array_offset,
            length: pixel_array_len,
            description: pixel_array_desc,
            color: egui::Color32::from_rgb(50, 200, 120), // Emerald
            parsed_data: pixel_array_hex,
        });
    }
    
    Some(chunks)
}

fn parse_generic(bytes: &[u8]) -> Vec<FileChunk> {
    let header_len = std::cmp::min(bytes.len(), 1024);
    let payload_offset = header_len;
    let payload_len = if bytes.len() > 2048 { bytes.len() - 2048 } else { 0 };
    let trailer_offset = if bytes.len() > 1024 { bytes.len() - 1024 } else { 0 };
    let trailer_len = std::cmp::min(bytes.len(), 1024);
    
    vec![
        FileChunk {
            name: "File Header".to_string(),
            offset: 0,
            length: header_len,
            description: "Binary File Signature / Header Block.".to_string(),
            color: egui::Color32::from_rgb(120, 110, 255),
            parsed_data: generate_hex_dump(&bytes[0..header_len], 0, 1024),
        },
        FileChunk {
            name: "Primary Payload Data".to_string(),
            offset: payload_offset,
            length: payload_len,
            description: "Main body containing compressed or uncompressed binary image payload.".to_string(),
            color: egui::Color32::from_rgb(50, 200, 120),
            parsed_data: generate_hex_dump(&bytes[payload_offset..payload_offset + payload_len], payload_offset, 1024),
        },
        FileChunk {
            name: "File Termination Block".to_string(),
            offset: trailer_offset,
            length: trailer_len,
            description: "End of file structure / trailer payload.".to_string(),
            color: egui::Color32::from_rgb(180, 180, 180),
            parsed_data: generate_hex_dump(&bytes[trailer_offset..trailer_offset + trailer_len], trailer_offset, 1024),
        }
    ]
}

fn collect_images_recursive(dir: &Path, tx: &std::sync::mpsc::Sender<PathBuf>, visited: &mut std::collections::HashSet<PathBuf>) {
    let canon_dir = match dir.canonicalize() {
        Ok(c) => c,
        Err(_) => dir.to_path_buf(),
    };
    if !visited.insert(canon_dir) {
        return;
    }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_dir() {
                collect_images_recursive(&path, tx, visited);
            } else if path.is_file() {
                let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
                if matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "bmp" | "gif" | "webp" | "tiff" | "avif" | "heif" | "heic" | "ico" | "icns" | "svg" |
                            "mp4" | "mov" | "avi" | "mkv" | "webm" | "m4v" | "wmv" | "mpg" | "mpeg") {
                    if let Ok(canon) = path.canonicalize() {
                        let _ = tx.send(canon);
                    } else {
                        let _ = tx.send(path);
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SearchMode {
    Filename,
    Clip,
    Ocr,
}

struct ClipIndex {
    entries: Vec<ClipEntry>,
    dim: usize,
    file_count: usize,
}

struct FaceIndex {
    entries: Vec<FaceEntry>,
    file_count: usize,
}

struct OcrIndex {
    entries: Vec<OcrEntry>,
    file_count: usize,
}

#[derive(Clone)]
struct ClipEntry {
    file_name: Arc<str>,
    is_video: bool,
    timestamp_sec: f32,
    vector: Vec<f32>,
}

#[derive(Clone)]
struct FaceEntry {
    file_name: Arc<str>,
    is_video: bool,
    timestamp_sec: f32,
    vector: Vec<f32>,
}

#[derive(Clone)]
struct OcrEntry {
    file_name: Arc<str>,
    is_video: bool,
    timestamp_sec: f32,
    text_lower: String,
}

#[derive(Clone)]
struct SearchResult {
    rank: usize,
    score: f32,
    file_name: String,
    is_video: bool,
    timestamp_sec: f32,
    media_path: Option<PathBuf>,
    ocr_term_hits: usize,
    ocr_query_terms: usize,
    ocr_phrase_query: bool,
}

#[derive(Clone)]
enum PendingSearchRequest {
    Similar {
        db_file_name: Option<String>,
        media_path: PathBuf,
        is_video: bool,
        timestamp_sec: f32,
    },
    Person {
        db_file_name: Option<String>,
        media_path: PathBuf,
        is_video: bool,
    },
}

struct OnDemandEmbedResult {
    request: PendingSearchRequest,
    clip_vector: Option<Vec<f32>>,
    face_vectors: Vec<Vec<f32>>,
}

struct SiftRepairResult {
    summary: String,
    files: usize,
}

#[derive(Clone)]
struct SimilarFile {
    file_name: String,
    is_video: bool,
    similarity_pct: Option<f32>,
}

#[derive(Clone)]
struct VideoFramePhash {
    timestamp_sec: f32,
    phash: u64,
}

fn parse_phash_hex(value: &str) -> Option<u64> {
    let value = value.trim();
    if value.len() != 16 {
        return None;
    }
    u64::from_str_radix(value, 16).ok()
}

fn phash_similarity_pct(a: u64, b: u64) -> f32 {
    (64 - (a ^ b).count_ones()) as f32 * 100.0 / 64.0
}

fn similarity_to_active(
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

fn duplicate_database_detail_lines(
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
        lines.push(format!("Video still pHashes: {}", frames.map_or(0, Vec::len)));
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

fn draw_embedding_markers(ui: &mut egui::Ui, has_clip: bool, has_ocr: bool, skipped: bool) {
    let missing_color = if skipped { egui::Color32::GRAY } else { egui::Color32::YELLOW };
    ui.colored_label(
        if has_clip { egui::Color32::LIGHT_GREEN } else { missing_color },
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
        if has_ocr { egui::Color32::LIGHT_GREEN } else { missing_color },
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

#[derive(Clone, Default)]
struct SiftInfo {
    match_file: Option<String>,
    score: Option<f32>,
    inliers: Option<i32>,
    good_matches: Option<i32>,
    inlier_ratio: Option<f32>,
    checked: Option<bool>,
}

struct ClipTextEncoder {
    tokenizer: Tokenizer,
    session: Session,
    context_len: usize,
}

struct DatabaseIndices {
    clip_index: Arc<ClipIndex>,
    face_index: Arc<FaceIndex>,
    ocr_index: Arc<OcrIndex>,
    clip_embedded_files: Arc<HashSet<String>>,
    ocr_embedded_files: Arc<HashSet<String>>,
    similar_by_master: HashMap<String, Vec<SimilarFile>>,
    phash_master_by_file: HashMap<String, String>,
    phash_by_file: HashMap<String, u64>,
    video_frame_phashes_by_file: HashMap<String, Vec<VideoFramePhash>>,
    sift_info_by_file: HashMap<String, SiftInfo>,
    sift_root_by_file: HashMap<String, String>,
    sift_members_by_root: HashMap<String, Vec<String>>,
    skipped_processing_files: Arc<HashSet<String>>,
    basename_to_db_filename: HashMap<String, String>,
    encoder: ClipTextEncoder,
}

struct SupplementalDbData {
    face_index: FaceIndex,
    ocr_index: OcrIndex,
    ocr_embedded_files: HashSet<String>,
    similar_by_master: HashMap<String, Vec<SimilarFile>>,
    phash_master_by_file: HashMap<String, String>,
    phash_by_file: HashMap<String, u64>,
    video_frame_phashes_by_file: HashMap<String, Vec<VideoFramePhash>>,
    sift_info_by_file: HashMap<String, SiftInfo>,
    sift_root_by_file: HashMap<String, String>,
    sift_members_by_root: HashMap<String, Vec<String>>,
    skipped_processing_files: HashSet<String>,
}

enum DatabaseLoadMessage {
    ClipReady(Result<(ClipIndex, ClipTextEncoder), String>),
    SupplementalReady(Result<SupplementalDbData, String>),
}

async fn load_clip_database_index(db_dir: &Path, table_name: &str) -> Result<ClipIndex> {
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

async fn load_supplemental_database_indices(
    db_dir: &Path,
    table_name: &str,
) -> Result<SupplementalDbData> {
    let db = lancedb::connect(db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let table = db.open_table(table_name).execute().await?;
    let table_schema = table.schema().await?;
    let has_cross_media_matches = table_schema.field_with_name("cross_media_matches").is_ok();
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
    let mut phash_master_by_file: HashMap<String, String> = HashMap::new();
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
        let cross_media_matches = batch
            .column_by_name("cross_media_matches")
            .and_then(|column| column.as_any().downcast_ref::<ListArray>());

        for row in 0..batch.num_rows() {
            if file_names.is_null(row) {
                continue;
            }
            let file_name = file_names.value(row).to_string();
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
            similar_by_master.entry(master.clone()).or_default().push(SimilarFile {
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
        sift_neighbors.entry(file_name.clone()).or_default().push(target.clone());
        sift_neighbors.entry(target.clone()).or_default().push(file_name.clone());
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
        phash_master_by_file,
        phash_by_file,
        video_frame_phashes_by_file,
        sift_info_by_file,
        sift_root_by_file,
        sift_members_by_root,
        skipped_processing_files,
    })
}

fn parse_batch(
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

fn parse_face_batch(
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

fn parse_ocr_batch(
    batch: &RecordBatch,
    entries: &mut Vec<OcrEntry>,
    seen_files: &mut HashSet<String>,
) -> Result<()> {
    let file_names = string_col(batch, "file_name")?;
    let is_video = bool_col(batch, "is_video")?;
    let skip_processing = bool_col(batch, "skip_processing")?;
    let ocr_groups = list_col(batch, "ocr_groups")?;

    for row in 0..batch.num_rows() {
        if bool_value(skip_processing, row) == Some(true) || ocr_groups.is_null(row) || file_names.is_null(row) {
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
            if text_detected.is_null(group_idx) || !text_detected.value(group_idx) || texts.is_null(group_idx) {
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

fn string_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    batch
        .column_by_name(name)
        .ok_or_else(|| anyhow!("missing column {name}"))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow!("column {name} is not string"))
}

fn bool_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a BooleanArray> {
    batch
        .column_by_name(name)
        .ok_or_else(|| anyhow!("missing column {name}"))?
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| anyhow!("column {name} is not bool"))
}

fn list_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ListArray> {
    batch
        .column_by_name(name)
        .ok_or_else(|| anyhow!("missing column {name}"))?
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| anyhow!("column {name} is not list"))
}

fn bool_value(array: &BooleanArray, row: usize) -> Option<bool> {
    if array.is_null(row) {
        None
    } else {
        Some(array.value(row))
    }
}

fn float_value(array: &dyn Array, row: usize) -> Option<f32> {
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

fn valid_sift_link(info: &SiftInfo) -> bool {
    info.checked == Some(true)
        && info.match_file.is_some()
        && info.inliers.unwrap_or(0) >= 10
        && info.inlier_ratio.unwrap_or(0.0) >= 0.40
        && info.score.unwrap_or(0.0) >= 0.0
}

impl ClipTextEncoder {
    fn new(onnx_path: &Path, tokenizer_path: &Path, context_len: usize) -> Result<Self> {
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

    fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
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

fn normalize_in_place(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector {
            *value /= norm;
        }
    }
}

fn search_index(
    index: &ClipIndex,
    query: &[f32],
    limit: usize,
    video_only: bool,
    folder_filter: &str,
) -> Vec<SearchResult> {
    let merged = index
        .entries
        .par_chunks(4096)
        .map(|chunk| {
            let db_roots = get_db_roots();
            let mut local: HashMap<String, (f32, bool, f32)> = HashMap::new();
            for entry in chunk {
                if video_only && !entry.is_video {
                    continue;
                }
                if !folder_filter.is_empty() && !file_matches_folder(&entry.file_name, folder_filter, &db_roots) {
                    continue;
                }
                let score = dot(query, &entry.vector);
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
    rows.truncate(limit);
    for (idx, row) in rows.iter_mut().enumerate() {
        row.rank = idx + 1;
    }
    rows
}

fn search_ocr_index(
    index: &OcrIndex,
    query: &str,
    limit: usize,
    video_only: bool,
    folder_filter: &str,
) -> Vec<SearchResult> {
    let query_trimmed = query.trim();
    if query_trimmed.is_empty() {
        return Vec::new();
    }
    let query_is_quoted = query_trimmed.starts_with('"')
        && query_trimmed.ends_with('"')
        && query_trimmed.len() >= 2;
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

    let merged = index
        .entries
        .par_chunks(4096)
        .map(|chunk| {
            let db_roots = get_db_roots();
            let mut local: HashMap<String, (f32, bool, f32, usize, usize, bool)> = HashMap::new();
            for entry in chunk {
                if video_only && !entry.is_video {
                    continue;
                }
                if !folder_filter.is_empty() && !file_matches_folder(&entry.file_name, folder_filter, &db_roots) {
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
                // Unquoted mode: prioritize rows that match more query terms.
                // Quoted mode: exact phrase required; term count keeps deterministic tie ordering.
                let score = if require_phrase_match {
                    10_000.0 + term_hits as f32
                } else {
                    let all_terms_bonus = if term_hits == query_term_count { 100.0 } else { 0.0 };
                    let phrase_bonus = if phrase_hit { 10.0 } else { 0.0 };
                    (term_hits as f32) * 1000.0 + all_terms_bonus + phrase_bonus
                };
                local
                    .entry(entry.file_name.to_string())
                    .and_modify(|best| {
                        if score > best.0 {
                            *best = (
                                score,
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
            |(file_name, (score, is_video, timestamp_sec, ocr_term_hits, ocr_query_terms, ocr_phrase_query))| SearchResult {
                rank: 0,
                score,
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
    rows.truncate(limit);
    for (idx, row) in rows.iter_mut().enumerate() {
        row.rank = idx + 1;
    }
    rows
}

const FACE_MATCH_MIN_SCORE: f32 = 0.30;

fn search_face_index(
    index: &FaceIndex,
    query_vectors: &[Vec<f32>],
    limit: usize,
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
    rows.truncate(limit);
    for (idx, row) in rows.iter_mut().enumerate() {
        row.rank = idx + 1;
    }
    rows
}

fn collapse_sift_grouped_results(
    mut rows: Vec<SearchResult>,
    sift_root_by_file: &HashMap<String, String>,
    limit: usize,
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
    collapsed.truncate(limit);
    for (idx, row) in collapsed.iter_mut().enumerate() {
        row.rank = idx + 1;
    }
    collapsed
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

const MEDIA_INDEX_TABLE: &str = "media_index";
const COLLECTION_ROOTS_TABLE: &str = "collection_roots";

fn looks_like_lancedb_dir(path: &Path) -> bool {
    path.join(format!("{MEDIA_INDEX_TABLE}.lance")).is_dir()
        || path.join(format!("{COLLECTION_ROOTS_TABLE}.lance")).is_dir()
}

fn discover_existing_db_dir() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("lancedb"));
        if let Some(parent) = cwd.parent() {
            candidates.push(parent.join("lancedb"));
        }
    }

    let users = [
        std::env::var("USER").unwrap_or_default(),
        std::env::var("USERNAME").unwrap_or_default(),
    ];
    for user in users.iter().filter(|user| !user.is_empty()) {
        for mount_base in [
            PathBuf::from("/media").join(user),
            PathBuf::from("/run/media").join(user),
        ] {
            if let Ok(entries) = std::fs::read_dir(&mount_base) {
                for entry in entries.filter_map(|entry| entry.ok()) {
                    let path = entry.path();
                    if path.is_dir() {
                        candidates.push(path.join("lancedb"));
                    }
                }
            }
        }
    }
    if let Ok(entries) = std::fs::read_dir("/mnt") {
        for entry in entries.filter_map(|entry| entry.ok()) {
            let path = entry.path();
            if path.is_dir() {
                candidates.push(path.join("lancedb"));
            }
        }
    }

    candidates
        .into_iter()
        .filter(|path| looks_like_lancedb_dir(path))
        .find_map(|path| path.canonicalize().ok().or(Some(path)))
}

fn default_db_dir() -> PathBuf {
    let repo_db_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("lancedb");
    if repo_db_dir.exists() || cfg!(debug_assertions) {
        return repo_db_dir;
    }

    if let Some(discovered) = discover_existing_db_dir() {
        return discovered;
    }

    if let Ok(raw) = std::env::var("XDG_DATA_HOME") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed).join("iris").join("lancedb");
        }
    }
    if let Ok(raw) = std::env::var("HOME") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed)
                .join(".local")
                .join("share")
                .join("iris")
                .join("lancedb");
        }
    }
    PathBuf::from("./lancedb")
}

fn get_db_dir() -> PathBuf {
    static DB_DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    DB_DIR
        .get_or_init(|| {
            if let Ok(raw) = std::env::var("IRIS_DB_DIR") {
                let trimmed = raw.trim();
                if !trimmed.is_empty() {
                    return PathBuf::from(trimmed);
                }
            }
            default_db_dir()
        })
        .clone()
}

fn resolve_imagesearch_dir() -> PathBuf {
    if let Ok(raw) = std::env::var("IRIS_IMAGESEARCH_DIR") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }

    let mut candidates = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("tools/media_indexer"));
        candidates.push(cwd.join("../imagesearch"));
        candidates.push(cwd.join("imagesearch"));
    }
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tools/media_indexer"));
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../imagesearch"));

    for candidate in candidates {
        if candidate.is_dir() {
            return candidate
                .canonicalize()
                .unwrap_or(candidate);
        }
    }

    PathBuf::from("tools/media_indexer")
}

fn resolve_on_demand_embeddings_script_path() -> PathBuf {
    if let Ok(raw) = std::env::var("IRIS_ON_DEMAND_EMBED_SCRIPT") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tools")
        .join("on_demand_embeddings.py")
}

fn dedupe_dirs(candidates: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for candidate in candidates {
        if !candidate.is_dir() {
            continue;
        }
        let canonical = candidate.canonicalize().unwrap_or(candidate.clone());
        let key = canonical.to_string_lossy().to_string();
        if seen.insert(key) {
            out.push(canonical);
        }
    }
    out
}

fn add_dir_and_children(base: &Path, out: &mut Vec<PathBuf>) {
    if !base.is_dir() {
        return;
    }
    out.push(base.to_path_buf());
    if let Ok(entries) = std::fs::read_dir(base) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_dir() {
                out.push(path);
            }
        }
    }
}

fn add_dir_children_depth2(base: &Path, out: &mut Vec<PathBuf>) {
    if !base.is_dir() {
        return;
    }
    if let Ok(entries) = std::fs::read_dir(base) {
        for entry in entries.filter_map(|e| e.ok()) {
            let level1 = entry.path();
            if !level1.is_dir() {
                continue;
            }
            out.push(level1.clone());
            if let Ok(level2_entries) = std::fs::read_dir(&level1) {
                for entry2 in level2_entries.filter_map(|e| e.ok()) {
                    let level2 = entry2.path();
                    if level2.is_dir() {
                        out.push(level2);
                    }
                }
            }
        }
    }
}

fn candidate_root_dirs(db_dir: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(parent) = db_dir.parent() {
        add_dir_and_children(parent, &mut candidates);
        if let Some(grand) = parent.parent() {
            add_dir_and_children(grand, &mut candidates);
        }
    }
    add_dir_children_depth2(Path::new("/media"), &mut candidates);
    add_dir_children_depth2(Path::new("/run/media"), &mut candidates);
    add_dir_children_depth2(Path::new("/mnt"), &mut candidates);
    dedupe_dirs(candidates)
}

async fn load_collection_roots_from_table(db_dir: &Path) -> Result<HashMap<String, PathBuf>> {
    let db = lancedb::connect(db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let table_names = db.table_names().execute().await?;
    if !table_names.iter().any(|name| name == COLLECTION_ROOTS_TABLE) {
        return Ok(HashMap::new());
    }

    let table = db.open_table(COLLECTION_ROOTS_TABLE).execute().await?;
    let stream = table
        .query()
        .select(Select::columns(&["collection_id", "root_path"]))
        .execute()
        .await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;

    let mut roots = HashMap::new();
    for batch in &batches {
        let ids = string_col(batch, "collection_id")?;
        let paths = string_col(batch, "root_path")?;
        for row in 0..batch.num_rows() {
            if ids.is_null(row) || paths.is_null(row) {
                continue;
            }
            let collection = ids.value(row).trim();
            let root_path = paths.value(row).trim();
            if collection.is_empty() || root_path.is_empty() {
                continue;
            }
            let root = PathBuf::from(root_path)
                .canonicalize()
                .unwrap_or_else(|_| PathBuf::from(root_path));
            roots.insert(collection.to_string(), root);
        }
    }
    Ok(roots)
}

async fn collect_collection_samples_from_media_index(
    db_dir: &Path,
) -> Result<HashMap<String, Vec<String>>> {
    let db = lancedb::connect(db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let table = db.open_table(MEDIA_INDEX_TABLE).execute().await?;
    let stream = table
        .query()
        .select(Select::columns(&["file_name"]))
        .execute()
        .await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;

    let mut samples: HashMap<String, Vec<String>> = HashMap::new();
    for batch in &batches {
        let file_names = string_col(batch, "file_name")?;
        for row in 0..batch.num_rows() {
            if file_names.is_null(row) {
                continue;
            }
            let file_name = file_names.value(row);
            let Some((collection, rel)) = file_name.split_once('/') else {
                continue;
            };
            let rel = rel.trim_start_matches('/').to_string();
            if rel.is_empty() {
                continue;
            }
            let bucket = samples.entry(collection.to_string()).or_default();
            if bucket.len() < 16 && !bucket.contains(&rel) {
                bucket.push(rel);
            }
        }
    }
    Ok(samples)
}

fn discover_collection_roots_from_samples(
    db_dir: &Path,
    samples: &HashMap<String, Vec<String>>,
) -> HashMap<String, PathBuf> {
    let candidates = candidate_root_dirs(db_dir);
    let mut roots = HashMap::new();

    for (collection, rel_samples) in samples {
        let mut best_path: Option<PathBuf> = None;
        let mut best_hits = 0usize;
        for candidate in &candidates {
            let hits = rel_samples
                .iter()
                .filter(|rel| candidate.join(rel.as_str()).exists())
                .count();
            if hits > best_hits {
                best_hits = hits;
                best_path = Some(candidate.clone());
            }
        }
        if best_hits > 0 {
            if let Some(path) = best_path {
                roots.insert(collection.clone(), path);
            }
        }
    }
    roots
}

async fn write_collection_roots_table(db_dir: &Path, roots: &HashMap<String, PathBuf>) -> Result<()> {
    if roots.is_empty() {
        return Ok(());
    }

    let db = lancedb::connect(db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let table_names = db.table_names().execute().await?;
    let table_exists = table_names.iter().any(|name| name == COLLECTION_ROOTS_TABLE);

    let mut rows: Vec<(String, String)> = roots
        .iter()
        .map(|(collection, root)| (collection.clone(), root.to_string_lossy().to_string()))
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));

    let collection_ids: Vec<String> = rows.iter().map(|(id, _)| id.clone()).collect();
    let root_paths: Vec<String> = rows.iter().map(|(_, path)| path.clone()).collect();
    let batch = RecordBatch::try_from_iter(vec![
        ("collection_id", Arc::new(StringArray::from(collection_ids)) as ArrayRef),
        ("root_path", Arc::new(StringArray::from(root_paths)) as ArrayRef),
    ])?;
    let schema = batch.schema();
    let batches = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);

    if table_exists {
        let table = db.open_table(COLLECTION_ROOTS_TABLE).execute().await?;
        table
            .add(Box::new(batches))
            .mode(AddDataMode::Overwrite)
            .execute()
            .await?;
    } else {
        db.create_table(COLLECTION_ROOTS_TABLE, Box::new(batches))
            .execute()
            .await?;
    }
    Ok(())
}

fn load_or_discover_db_roots(db_dir: &Path) -> HashMap<String, PathBuf> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build();
    let Ok(runtime) = runtime else {
        return HashMap::new();
    };

    runtime.block_on(async {
        let mut roots = load_collection_roots_from_table(db_dir)
            .await
            .unwrap_or_default();
        if !roots.is_empty() {
            return roots;
        }

        let samples = collect_collection_samples_from_media_index(db_dir)
            .await
            .unwrap_or_default();
        let discovered = discover_collection_roots_from_samples(db_dir, &samples);

        let mut changed = false;
        for (collection, discovered_root) in discovered {
            match roots.get(&collection) {
                Some(existing_root) => {
                    if !existing_root.exists() && discovered_root.exists() {
                        roots.insert(collection, discovered_root);
                        changed = true;
                    }
                }
                None => {
                    roots.insert(collection, discovered_root);
                    changed = true;
                }
            }
        }

        if changed {
            let _ = write_collection_roots_table(db_dir, &roots).await;
        }
        roots
    })
}

fn get_db_roots() -> HashMap<String, PathBuf> {
    static ROOTS_CACHE: std::sync::OnceLock<std::sync::Mutex<(Instant, HashMap<String, PathBuf>, bool)>> =
        std::sync::OnceLock::new();

    let cache = ROOTS_CACHE.get_or_init(|| {
        std::sync::Mutex::new((
            Instant::now() - Duration::from_secs(3600),
            HashMap::new(),
            false,
        ))
    });

    let mut should_refresh = false;
    let roots = {
        let mut guard = match cache.lock() {
            Ok(guard) => guard,
            Err(_) => return HashMap::new(),
        };

        let refresh_after = if guard.1.is_empty() {
            Duration::from_secs(2)
        } else {
            Duration::from_secs(60)
        };
        if guard.0.elapsed() >= refresh_after && !guard.2 {
            guard.2 = true;
            should_refresh = true;
        }

        guard.1.clone()
    };

    if should_refresh {
        let cache_ref = cache;
        std::thread::spawn(move || {
            let db_dir = get_db_dir();
            let fresh = load_or_discover_db_roots(&db_dir);
            if let Ok(mut guard) = cache_ref.lock() {
                if !fresh.is_empty() || guard.1.is_empty() {
                    guard.1 = fresh;
                }
                guard.0 = Instant::now();
                guard.2 = false;
            }
        });
    }

    roots
}

fn file_matches_folder(file_name: &str, folder: &str, db_roots: &HashMap<String, PathBuf>) -> bool {
    let folder = folder.trim();
    if folder.is_empty() {
        return true;
    }
    let normalized_file = file_name.replace('\\', "/").to_lowercase();
    let normalized_folder = folder.replace('\\', "/").to_lowercase();
    let folder_path = Path::new(folder);
    let is_path_like = normalized_folder.contains('/')
        || normalized_folder.contains('\\')
        || folder_path.is_absolute();

    if !is_path_like {
        let rel_segments: Vec<&str> = normalized_file.split('/').collect();
        if rel_segments.len() > 2 {
            for segment in &rel_segments[1..rel_segments.len() - 1] {
                if segment.contains(&normalized_folder) {
                    return true;
                }
            }
        }
        if let Ok(source_path) = resolve_source_path(db_roots, file_name) {
            let source_segments: Vec<String> = source_path
                .components()
                .map(|component| component.as_os_str().to_string_lossy().to_lowercase())
                .collect();
            if source_segments.len() > 1 {
                for segment in &source_segments[..source_segments.len() - 1] {
                    if segment.contains(&normalized_folder) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    if normalized_file.contains(&normalized_folder) {
        return true;
    }
    if let Ok(canon_folder) = folder_path.canonicalize() {
        if let Ok(source_path) = resolve_source_path(db_roots, file_name) {
            if let Ok(canon_source) = source_path.canonicalize() {
                if canon_source.starts_with(&canon_folder) {
                    return true;
                }
            }
            if source_path.starts_with(folder_path) {
                return true;
            }
        }
    } else if let Ok(source_path) = resolve_source_path(db_roots, file_name) {
        let source_str = source_path.to_string_lossy().replace('\\', "/").to_lowercase();
        if source_str.contains(&normalized_folder) {
            return true;
        }
    }
    false
}

fn text_edit_enter_pressed(response: &egui::Response) -> bool {
    let owns_enter = response.has_focus() || response.lost_focus();
    owns_enter
        && response.ctx.input(|input| {
            input.events.iter().any(|event| {
                matches!(
                    event,
                    egui::Event::Key {
                        key: egui::Key::Enter,
                        pressed: true,
                        repeat: false,
                        ..
                    }
                )
            })
        })
}

fn bounded_edit_distance(a: &str, b: &str, max_distance: usize) -> Option<usize> {
    if a.len().abs_diff(b.len()) > max_distance {
        return None;
    }
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    let mut current = vec![0; b.len() + 1];
    for (row, a_byte) in a.bytes().enumerate() {
        current[0] = row + 1;
        let mut row_min = current[0];
        for (col, b_byte) in b.bytes().enumerate() {
            current[col + 1] = (previous[col + 1] + 1)
                .min(current[col] + 1)
                .min(previous[col] + usize::from(a_byte != b_byte));
            row_min = row_min.min(current[col + 1]);
        }
        if row_min > max_distance {
            return None;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    (previous[b.len()] <= max_distance).then_some(previous[b.len()])
}

fn fuzzy_path_component_matches(query: &str, candidate: &str) -> bool {
    if candidate.contains(query) || query.contains(candidate) {
        return true;
    }
    let max_distance = if query.len() >= 12 {
        2
    } else if query.len() >= 5 {
        1
    } else {
        0
    };
    max_distance > 0 && bounded_edit_distance(query, candidate, max_distance).is_some()
}

fn partial_path_matches(query: &str, candidate: &str) -> bool {
    if candidate.contains(query) {
        return true;
    }
    let query_parts: Vec<&str> = query.split('/').filter(|part| !part.is_empty()).collect();
    let candidate_parts: Vec<&str> = candidate.split('/').filter(|part| !part.is_empty()).collect();
    if query_parts.is_empty() || query_parts.len() > candidate_parts.len() {
        return false;
    }
    candidate_parts.windows(query_parts.len()).any(|window| {
        query_parts
            .iter()
            .zip(window)
            .all(|(query_part, candidate_part)| fuzzy_path_component_matches(query_part, candidate_part))
    })
}

fn resolve_media_path(
    roots: &HashMap<String, PathBuf>,
    db_dir: &Path,
    file_name: &str,
    timestamp_sec: f32,
) -> Result<PathBuf> {
    let source = resolve_source_path(roots, file_name)?;
    let (_collection, rel) = file_name
        .split_once('/')
        .ok_or_else(|| anyhow!("file_name does not contain collection id"))?;
    let rel_path = Path::new(rel);
    if is_video_path(&source) {
        let (collection, _) = file_name
            .split_once('/')
            .ok_or_else(|| anyhow!("file_name does not contain collection id"))?;
        let root = roots
            .get(collection)
            .cloned()
            .ok_or_else(|| anyhow!("no collection-root for {collection}"))?;
        if let Some(still) = resolve_video_still(&root, db_dir, rel_path, timestamp_sec)? {
            return Ok(still);
        }
    }
    Ok(source)
}

fn resolve_source_path(roots: &HashMap<String, PathBuf>, file_name: &str) -> Result<PathBuf> {
    let (collection, rel) = file_name
        .split_once('/')
        .ok_or_else(|| anyhow!("file_name does not contain collection id"))?;
    let root = roots
        .get(collection)
        .cloned()
        .ok_or_else(|| anyhow!("no collection-root for {collection}"))?;
    Ok(root.join(Path::new(rel)))
}

fn db_filename_from_video_still_path(path: &Path) -> Option<String> {
    let path_str = path.to_string_lossy();
    if !path_str.contains("-video") && !path_str.contains("lancedb") {
        return None;
    }
    
    let db_dir_buf = get_db_dir();
    let db_dir = db_dir_buf.as_path();
    
    let rel = path.strip_prefix(db_dir).ok().map(|p| p.to_path_buf()).or_else(|| {
        let canon_path = path.canonicalize().ok()?;
        let canon_db = db_dir.canonicalize().ok()?;
        canon_path.strip_prefix(canon_db).ok().map(|p| p.to_path_buf())
    })?;
    
    let components: Vec<_> = rel.components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    
    if components.len() < 3 {
        return None;
    }
    
    let first = &components[0];
    if !first.ends_with("-video") {
        return None;
    }
    let collection_id = &first[..first.len() - "-video".len()];
    
    let video_dir_name = &components[components.len() - 2];
    let (stem, ext) = video_dir_name.rsplit_once('_')?;
    
    let mut rel_parts = Vec::new();
    for part in &components[1..components.len() - 2] {
        rel_parts.push(part.as_str());
    }
    let reconstructed_video_filename = format!("{}.{}", stem, ext);
    rel_parts.push(&reconstructed_video_filename);
    
    Some(format!("{}/{}", collection_id, rel_parts.join("/")))
}

fn open_in_dolphin_or_fallback(file_path: &Path) {
    let path = file_path.to_path_buf();
    std::thread::spawn(move || {
        let success = if let Ok(mut child) = std::process::Command::new("dolphin")
            .arg("--select")
            .arg(&path)
            .spawn()
        {
            child.wait().map(|s| s.success()).unwrap_or(false)
        } else {
            false
        };
        
        if !success {
            if let Some(parent) = path.parent() {
                let _ = std::process::Command::new("xdg-open")
                    .arg(parent)
                    .spawn();
            }
        }
    });
}

fn is_video_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|x| x.to_str())
            .map(|x| x.to_ascii_lowercase())
            .as_deref(),
        Some("mp4" | "mov" | "avi" | "mkv" | "webm" | "m4v" | "wmv" | "mpg" | "mpeg")
    )
}

fn is_supported_media_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|x| x.to_str())
            .map(|x| x.to_ascii_lowercase())
            .as_deref(),
        Some(
            "jpg"
                | "jpeg"
                | "png"
                | "bmp"
                | "gif"
                | "webp"
                | "tiff"
                | "avif"
                | "heif"
                | "heic"
                | "ico"
                | "icns"
                | "svg"
                | "mp4"
                | "mov"
                | "avi"
                | "mkv"
                | "webm"
                | "m4v"
                | "wmv"
                | "mpg"
                | "mpeg"
        )
    )
}

fn resolve_video_still(
    root: &Path,
    db_dir: &Path,
    rel_path: &Path,
    timestamp_sec: f32,
) -> Result<Option<PathBuf>> {
    let scene_dir_name = format!(
        "{}-video",
        root.file_name().and_then(|x| x.to_str()).unwrap_or("video")
    );
    let mut scene_roots = Vec::with_capacity(2);
    scene_roots.push(db_dir.join(&scene_dir_name));
    if let Some(parent) = root.parent() {
        scene_roots.push(parent.join(&scene_dir_name));
    }
    let parent = rel_path.parent().unwrap_or_else(|| Path::new(""));
    let stem = rel_path
        .file_stem()
        .and_then(|x| x.to_str())
        .unwrap_or("video");
    let suffix = rel_path
        .extension()
        .and_then(|x| x.to_str())
        .map(|x| format!("_{}", x.to_ascii_lowercase()))
        .unwrap_or_default();
    for scene_root in scene_roots {
        let video_dir = scene_root.join(parent).join(format!("{stem}{suffix}"));
        let manifest_path = video_dir.join("manifest.json");
        let data = match std::fs::read_to_string(&manifest_path) {
            Ok(data) => data,
            Err(_) => continue,
        };
        let manifest: Value = serde_json::from_str(&data)?;
        let frames = manifest
            .get("frames")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("manifest has no frames"))?;
        let mut best: Option<(f32, PathBuf)> = None;
        for frame in frames {
            let ts = frame
                .get("timestamp_sec")
                .and_then(Value::as_f64)
                .unwrap_or(0.0) as f32;
            let image_file = match frame.get("image_file").and_then(Value::as_str) {
                Some(value) => value,
                None => continue,
            };
            let distance = (ts - timestamp_sec).abs();
            let path = video_dir.join(image_file);
            if best
                .as_ref()
                .map(|(best_dist, _)| distance < *best_dist)
                .unwrap_or(true)
            {
                best = Some((distance, path));
            }
        }
        if let Some((_, path)) = best {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

fn compute_sift_summary(path_a: &Path, path_b: &Path) -> Result<String> {
    let imagesearch_dir = resolve_imagesearch_dir();
    let output = Command::new("uv")
        .current_dir(&imagesearch_dir)
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

fn run_sift_repair_for_files(file_names: &[String]) -> Result<SiftRepairResult> {
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
    let payload = serde_json::to_string(file_names).context("failed to serialize selected file list")?;
    std::fs::write(&temp_path, payload).context("failed to write selected file list")?;

    let imagesearch_dir = resolve_imagesearch_dir();
    let mut command = Command::new("uv");
    command
        .current_dir(&imagesearch_dir)
        .args([
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

    let output = command.output().context("failed to run repair_sift_results.py")?;
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
    let accepted = payload.get("accepted_pairs").and_then(Value::as_u64).unwrap_or(0);
    let linked = payload.get("linked_images").and_then(Value::as_u64).unwrap_or(0);
    let updated = payload.get("updated").and_then(Value::as_u64).unwrap_or(0);
    Ok(SiftRepairResult {
        summary: format!(
            "SIFT repair finished: {images} images, {pairs} pairs checked, {accepted} accepted, {linked} linked, {updated} database rows updated."
        ),
        files: file_names.len(),
    })
}

fn compute_on_demand_embeddings(
    image_path: &Path,
    need_clip: bool,
    need_faces: bool,
) -> Result<(Option<Vec<f32>>, Vec<Vec<f32>>)> {
    let imagesearch_dir = resolve_imagesearch_dir();
    let helper_script = resolve_on_demand_embeddings_script_path();
    let mut cmd = Command::new("uv");
    cmd.current_dir(&imagesearch_dir);
    cmd.env("UV_CACHE_DIR", "/data/.cache/uv");
    cmd.args(["run", "python"]);
    cmd.arg(&helper_script);
    cmd.arg("--image");
    cmd.arg(image_path);
    if need_clip {
        cmd.arg("--clip");
    }
    if need_faces {
        cmd.arg("--faces");
    }

    let output = cmd
        .output()
        .map_err(|e| anyhow!("failed to run embedding helper: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        bail!(
        "embedding helper failed (status {}) using {} and {}: {}",
            output.status,
            imagesearch_dir.display(),
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

#[derive(Clone, Copy)]
enum CropDragMode {
    New,
    Move,
    Left,
    Right,
    Top,
    Bottom,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

struct ImageEditor {
    source_path: PathBuf,
    image: image::DynamicImage,
    texture: egui::TextureHandle,
    crop_min: egui::Pos2,
    crop_max: egui::Pos2,
    crop_drag_mode: Option<CropDragMode>,
    crop_drag_origin: egui::Pos2,
    crop_drag_initial_min: egui::Pos2,
    crop_drag_initial_max: egui::Pos2,
    status: String,
}

struct ImageViewer {
    images: Vec<PathBuf>,
    current_index: usize,
    zoom: f32,
    offset: egui::Vec2,
    exif_data: String,
    side_panel_metadata_path: Option<PathBuf>,
    side_panel_layout_path: Option<PathBuf>,
    show_exif: bool,
    side_panel_window_expanded: bool,
    side_panel_open_pending: bool,
    side_panel_expand_target_width: Option<f32>,
    side_panel_open_pending_frames: u8,
    chunks: Vec<FileChunk>,
    viewport_bg: Option<egui::Color32>,
    rx: Receiver<PathBuf>,
    show_grid: bool,
    recursive_images: Vec<PathBuf>,
    grid_loading: bool,
    recursive_rx: Option<Receiver<PathBuf>>,
    back_target_is_gallery: bool,
    side_panel_mode: SidePanelMode,
    exif_search: String,
    open_target: PathBuf,
    open_target_is_dir: bool,
    flat_loading: bool,
    flat_images_shared: Arc<Mutex<Option<Vec<PathBuf>>>>,
    current_dimensions: String,
    current_file_size: String,
    ctx_shared: Arc<Mutex<Option<egui::Context>>>,
    thumbnail_textures: std::collections::HashMap<PathBuf, egui::TextureHandle>,
    thumbnail_loading: std::collections::HashSet<PathBuf>,
    thumbnail_failed: std::collections::HashSet<PathBuf>,
    thumbnail_rx: std::sync::mpsc::Receiver<(PathBuf, egui::ColorImage)>,
    thumbnail_tx: std::sync::mpsc::Sender<(PathBuf, egui::ColorImage)>,
    thumbnail_active_threads: usize,
    
    // AI Database & Explorer States
    db_loaded: bool,
    db_loading: bool,
    db_supplemental_loaded: bool,
    db_supplemental_loading: bool,
    db_failed: bool,
    db_rx: Option<Receiver<DatabaseLoadMessage>>,
    db_indices: Option<DatabaseIndices>,
    semantic_query: String,
    applied_filename_query: String,
    filename_search_results: Option<Vec<usize>>,
    semantic_folder: String,
    semantic_limit: usize,
    semantic_video_only: bool,
    semantic_mode: SearchMode,
    semantic_results: Vec<SearchResult>,
    semantic_results_mode: Option<SearchMode>,
    semantic_status: String,
    pending_search_request: Option<PendingSearchRequest>,
    pending_semantic_search_mode: Option<SearchMode>,
    on_demand_embed_rx: Option<Receiver<Result<OnDemandEmbedResult, String>>>,
    
    // Duplicates & SIFT states
    compare_target: Option<PathBuf>,
    sift_pair_overlay: Option<String>,
    expanded_duplicate_rows: HashSet<String>,
    sift_running: bool,
    sift_rx: Option<Receiver<Result<String, String>>>,
    selected_grid_files: Vec<String>,
    sift_repair_running: bool,
    sift_repair_rx: Option<Receiver<Result<SiftRepairResult, String>>>,
    image_editor: Option<ImageEditor>,

    // Maps resolved media_path → database file_name for AI search results.
    // Avoids reverse-mapping video stills back to collection paths.
    db_filename_by_path: HashMap<PathBuf, String>,

    // Cache resolved video still paths to avoid redundant manifest.json reads on the UI thread
    video_still_cache: std::cell::RefCell<HashMap<PathBuf, PathBuf>>,

    // Cache computed resolution and size strings to avoid heavy sync disk I/O on the UI thread
    resolution_size_cache: std::cell::RefCell<HashMap<PathBuf, String>>,

    // Cache resolved database filenames to avoid synchronous path canonicalization on the UI thread
    db_filename_cache: std::cell::RefCell<HashMap<PathBuf, Option<String>>>,

    // Home Page states
    show_home_page: bool,
    home_current_dir: Option<PathBuf>,
    home_selected_dir: Option<PathBuf>,
}

fn file_resolution_and_size(path: &Path) -> String {
    let size_label = match std::fs::metadata(path) {
        Ok(meta) => {
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
        Err(_) => "n/a".to_string(),
    };
    match image::image_dimensions(path) {
        Ok((w, h)) => format!("{}x{} | {}", w, h, size_label),
        Err(_) => size_label,
    }
}

fn sift_info_line(sift_info_by_file: &HashMap<String, SiftInfo>, file_name: &str) -> String {
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

fn wrapping_monospace_path(ui: &mut egui::Ui, text: &str) {
    let label = egui::Label::new(egui::RichText::new(text).monospace())
        .wrap()
        .selectable(true);
    ui.add(label);
}

fn clipboard_image_dir() -> PathBuf {
    std::env::temp_dir().join("iris-clipboard")
}

fn is_clipboard_image_path(path: &Path) -> bool {
    path.starts_with(clipboard_image_dir())
}

fn percent_decode_file_uri_path(raw: &str) -> String {
    let bytes = raw.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut idx = 0;
    while idx < bytes.len() {
        if bytes[idx] == b'%' && idx + 2 < bytes.len() {
            if let Ok(hex) = std::str::from_utf8(&bytes[idx + 1..idx + 3]) {
                if let Ok(value) = u8::from_str_radix(hex, 16) {
                    out.push(value);
                    idx += 3;
                    continue;
                }
            }
        }
        out.push(bytes[idx]);
        idx += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn image_path_from_pasted_text(text: &str) -> Option<PathBuf> {
    for raw_line in text.lines() {
        let line = raw_line.trim().trim_matches('"').trim_matches('\'');
        if line.is_empty() || line.eq_ignore_ascii_case("copy") {
            continue;
        }
        let path_text = line
            .strip_prefix("file://")
            .map(percent_decode_file_uri_path)
            .unwrap_or_else(|| line.to_string());
        let path = PathBuf::from(path_text);
        if path.exists() && is_supported_media_path(&path) && !is_video_path(&path) {
            return Some(path);
        }
    }
    None
}

fn clipboard_paste_signal(ui: &egui::Ui) -> (bool, Option<String>) {
    ui.input(|i| {
        let pasted_text = i.events.iter().find_map(|event| {
            if let egui::Event::Paste(text) = event {
                Some(text.clone())
            } else {
                None
            }
        });
        let shortcut = i.events.iter().any(|event| {
            matches!(
                event,
                egui::Event::Key {
                    key: egui::Key::V | egui::Key::Paste,
                    pressed: true,
                    modifiers,
                    ..
                } if modifiers.matches_logically(egui::Modifiers::COMMAND)
                    || modifiers.matches_logically(egui::Modifiers::CTRL)
            )
        }) || ((i.modifiers.matches_logically(egui::Modifiers::COMMAND)
            || i.modifiers.matches_logically(egui::Modifiers::CTRL))
            && (i.key_pressed(egui::Key::V) || i.key_pressed(egui::Key::Paste)));
        (shortcut, pasted_text)
    })
}

fn command_available(name: &str) -> bool {
    let Some(paths) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&paths).any(|dir| dir.join(name).is_file())
}

fn clipboard_command_output(program: &str, args: &[&str]) -> Result<Option<Vec<u8>>> {
    if !command_available(program) {
        return Ok(None);
    }

    let output = if command_available("timeout") {
        let mut cmd = Command::new("timeout");
        cmd.arg("2s").arg(program).args(args).output()
    } else {
        Command::new(program).args(args).output()
    }
    .with_context(|| format!("failed to run {program}"))?;

    if !output.status.success() || output.stdout.is_empty() {
        return Ok(None);
    }
    Ok(Some(output.stdout))
}

fn save_clipboard_bytes_to_temp(bytes: &[u8], ext: &str) -> Result<Option<PathBuf>> {
    if bytes.is_empty() {
        return Ok(None);
    }
    image::load_from_memory(bytes).context("clipboard image bytes are not a supported image")?;

    let dir = clipboard_image_dir();
    std::fs::create_dir_all(&dir).context("failed to create clipboard image temp directory")?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let path = dir.join(format!(
        "clipboard_{stamp}_{}.{}",
        std::process::id(),
        ext.trim_start_matches('.')
    ));
    std::fs::write(&path, bytes)
        .with_context(|| format!("failed to save clipboard image to {}", path.display()))?;
    Ok(Some(path))
}

fn wl_paste_clipboard_image_to_temp() -> Result<Option<PathBuf>> {
    if !command_available("wl-paste") {
        return Ok(None);
    }

    let type_list = clipboard_command_output("wl-paste", &["--list-types"])?
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .unwrap_or_default();
    let types: Vec<&str> = type_list.lines().map(str::trim).filter(|line| !line.is_empty()).collect();

    let text_types = [
        "text/uri-list",
        "x-special/gnome-copied-files",
        "text/plain;charset=utf-8",
        "text/plain",
        "UTF8_STRING",
        "STRING",
    ];
    for mime in text_types {
        if !types.is_empty() && !types.iter().any(|item| *item == mime) {
            continue;
        }
        if let Some(bytes) = clipboard_command_output("wl-paste", &["--no-newline", "--type", mime])? {
            if let Ok(text) = String::from_utf8(bytes) {
                if let Some(path) = image_path_from_pasted_text(&text) {
                    return Ok(Some(path));
                }
            }
        }
    }

    let image_types = [
        ("image/png", "png"),
        ("image/jpeg", "jpg"),
        ("image/jpg", "jpg"),
        ("image/webp", "webp"),
        ("image/bmp", "bmp"),
        ("image/tiff", "tiff"),
        ("image/gif", "gif"),
    ];
    for (mime, ext) in image_types {
        if !types.is_empty() && !types.iter().any(|item| *item == mime) {
            continue;
        }
        if let Some(bytes) = clipboard_command_output("wl-paste", &["--type", mime])? {
            if let Some(path) = save_clipboard_bytes_to_temp(&bytes, ext)? {
                return Ok(Some(path));
            }
        }
    }

    Ok(None)
}

fn save_clipboard_image_to_temp(pasted_text: Option<&str>) -> Result<Option<PathBuf>> {
    if let Some(text) = pasted_text {
        if let Some(path) = image_path_from_pasted_text(text) {
            return Ok(Some(path));
        }
    }

    if let Some(path) = wl_paste_clipboard_image_to_temp()? {
        return Ok(Some(path));
    }

    let mut clipboard = arboard::Clipboard::new().context("failed to open clipboard")?;
    let image = match clipboard.get_image() {
        Ok(image) => image,
        Err(arboard::Error::ContentNotAvailable) => return Ok(None),
        Err(err) => bail!("failed to read clipboard image: {err}"),
    };

    let width = u32::try_from(image.width).context("clipboard image width is too large")?;
    let height = u32::try_from(image.height).context("clipboard image height is too large")?;
    let bytes = image.bytes.into_owned();
    let expected_len = image
        .width
        .checked_mul(image.height)
        .and_then(|pixels| pixels.checked_mul(4))
        .context("clipboard image dimensions overflow")?;
    if bytes.len() != expected_len {
        bail!(
            "clipboard image has {} bytes, expected {} bytes for {}x{} RGBA",
            bytes.len(),
            expected_len,
            image.width,
            image.height
        );
    }

    let rgba = image::RgbaImage::from_raw(width, height, bytes)
        .ok_or_else(|| anyhow!("failed to build RGBA image from clipboard pixels"))?;
    let dir = clipboard_image_dir();
    std::fs::create_dir_all(&dir).context("failed to create clipboard image temp directory")?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let path = dir.join(format!("clipboard_{stamp}_{}.png", std::process::id()));
    rgba.save(&path)
        .with_context(|| format!("failed to save clipboard image to {}", path.display()))?;
    Ok(Some(path))
}

fn copy_image_file_to_clipboard(path: &Path) -> Result<()> {
    let img = image::open(path)
        .with_context(|| format!("failed to open image for clipboard copy: {}", path.display()))?;
    if command_available("wl-copy") {
        let mut png_bytes = std::io::Cursor::new(Vec::new());
        img.write_to(&mut png_bytes, image::ImageFormat::Png)
            .context("failed to encode image as PNG for clipboard copy")?;
        let mut child = Command::new("wl-copy")
            .args(["--type", "image/png"])
            .stdin(std::process::Stdio::piped())
            .spawn()
            .context("failed to start wl-copy")?;
        {
            use std::io::Write;
            let stdin = child.stdin.as_mut().context("wl-copy stdin is unavailable")?;
            stdin
                .write_all(png_bytes.get_ref())
                .context("failed to send image to wl-copy")?;
        }
        let status = child.wait().context("failed waiting for wl-copy")?;
        if status.success() {
            return Ok(());
        }
        bail!("wl-copy exited with status {status}");
    }

    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();
    let mut clipboard = arboard::Clipboard::new().context("failed to open clipboard")?;
    clipboard
        .set_image(arboard::ImageData {
            width: width as usize,
            height: height as usize,
            bytes: Cow::Owned(rgba.into_raw()),
        })
        .context("failed to copy image to clipboard")?;
    Ok(())
}

fn get_system_disks() -> Vec<PathBuf> {
    let mut disks = Vec::new();
    let mut seen = HashSet::new();
    
    // 1. Root directory is always a disk
    let root = PathBuf::from("/");
    seen.insert(root.clone());
    disks.push(root);
    
    // 2. Check for username to scan user-specific media directories
    let usernames = vec![
        std::env::var("USER").unwrap_or_default(),
        std::env::var("USERNAME").unwrap_or_default(),
    ];
    
    let mut scan_paths = vec![
        PathBuf::from("/media"),
        PathBuf::from("/mnt"),
    ];
    
    for user in usernames {
        if !user.is_empty() {
            scan_paths.push(PathBuf::from(format!("/media/{}", user)));
            scan_paths.push(PathBuf::from(format!("/run/media/{}", user)));
        }
    }
    
    // Add user home directory as a top-level shortcut
    if let Ok(home) = std::env::var("HOME") {
        let home_path = PathBuf::from(home);
        if home_path.is_dir() {
            let canon = home_path.canonicalize().unwrap_or(home_path);
            if seen.insert(canon.clone()) {
                disks.push(canon);
            }
        }
    }
    
    for path in scan_paths {
        if path.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&path) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let entry_path = entry.path();
                    if entry_path.is_dir() {
                        let canon = entry_path.canonicalize().unwrap_or(entry_path);
                        if seen.insert(canon.clone()) {
                            disks.push(canon);
                        }
                    }
                }
            }
        }
    }
    
    // Sort so it is deterministic
    disks.sort();
    disks
}

fn normalized_path_for_match(path: &Path) -> String {
    path.to_string_lossy()
        .replace('\\', "/")
        .trim_end_matches('/')
        .to_lowercase()
}

fn path_matches_db_root(path: &Path, root: &Path) -> bool {
    if path.starts_with(root) || root.starts_with(path) {
        return true;
    }

    let path_norm = normalized_path_for_match(path);
    let root_norm = normalized_path_for_match(root);
    if path_norm == root_norm {
        return true;
    }
    if path_norm.starts_with(&(root_norm.clone() + "/"))
        || root_norm.starts_with(&(path_norm.clone() + "/"))
    {
        return true;
    }

    let root_name = root
        .file_name()
        .and_then(|n| n.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();
    if !root_name.is_empty() {
        let needle = format!("/{}/", root_name);
        if path_norm.contains(&needle) || path_norm.ends_with(&format!("/{root_name}")) {
            return true;
        }
    }

    false
}

fn is_path_ai_backed(path: &Path) -> bool {
    let db_roots = get_db_roots();
    is_path_ai_backed_with_roots(path, &db_roots)
}

fn is_path_ai_backed_with_roots(path: &Path, db_roots: &HashMap<String, PathBuf>) -> bool {
    db_roots
        .values()
        .any(|root| path_matches_db_root(path, root))
}

impl ImageViewer {
    const SIDE_PANEL_WIDTH: f32 = 400.0;
    const MIN_WINDOW_WIDTH: f32 = 640.0;
    const SIDE_PANEL_RESIZE_TOLERANCE: f32 = 6.0;
    const SIDE_PANEL_OPEN_FALLBACK_FRAMES: u8 = 8;

    fn viewport_inner_size(ctx: &egui::Context) -> egui::Vec2 {
        ctx.input(|input| {
            input
                .viewport()
                .inner_rect
                .map(|rect| rect.size())
                .unwrap_or_else(|| input.viewport_rect().size())
        })
    }

    fn set_window_width(ctx: &egui::Context, width: f32, height: f32) {
        ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
            width,
            height,
        )));
    }

    fn open_side_panel(&mut self, ctx: &egui::Context, mode: SidePanelMode) {
        self.side_panel_mode = mode;
        if self.show_exif || self.side_panel_open_pending {
            return;
        }

        let current_size = Self::viewport_inner_size(ctx);
        let target_width = (current_size.x + Self::SIDE_PANEL_WIDTH).max(Self::MIN_WINDOW_WIDTH);
        Self::set_window_width(ctx, target_width, current_size.y);
        self.side_panel_window_expanded = true;
        self.side_panel_open_pending = true;
        self.side_panel_expand_target_width = Some(target_width);
        self.side_panel_open_pending_frames = 0;
        ctx.request_repaint();
    }

    fn close_side_panel(&mut self, ctx: &egui::Context) {
        let should_shrink =
            self.show_exif || self.side_panel_open_pending || self.side_panel_window_expanded;
        let was_pending_only = self.side_panel_open_pending && !self.show_exif;
        let expand_target_width = self.side_panel_expand_target_width;

        self.show_exif = false;
        self.side_panel_open_pending = false;
        self.side_panel_expand_target_width = None;
        self.side_panel_open_pending_frames = 0;

        if should_shrink {
            let current_size = Self::viewport_inner_size(ctx);
            let resize_has_landed = expand_target_width
                .map(|target| current_size.x + Self::SIDE_PANEL_RESIZE_TOLERANCE >= target)
                .unwrap_or(true);
            let target_width = if was_pending_only && !resize_has_landed {
                current_size.x
            } else {
                (current_size.x - Self::SIDE_PANEL_WIDTH).max(Self::MIN_WINDOW_WIDTH)
            };
            Self::set_window_width(ctx, target_width, current_size.y);
            self.side_panel_window_expanded = false;
            ctx.request_repaint();
        }
    }

    fn toggle_layout_side_panel(&mut self, ctx: &egui::Context) {
        let layout_active = (self.show_exif || self.side_panel_open_pending)
            && self.side_panel_mode == SidePanelMode::Layout;
        if layout_active {
            self.close_side_panel(ctx);
        } else {
            self.open_side_panel(ctx, SidePanelMode::Layout);
        }
    }

    fn apply_pending_side_panel_open(&mut self, ctx: &egui::Context) {
        if !self.side_panel_open_pending {
            return;
        }

        let current_size = Self::viewport_inner_size(ctx);
        let target_width = self
            .side_panel_expand_target_width
            .unwrap_or(current_size.x);
        self.side_panel_open_pending_frames =
            self.side_panel_open_pending_frames.saturating_add(1);

        let resize_landed = current_size.x + Self::SIDE_PANEL_RESIZE_TOLERANCE >= target_width;
        let waited_too_long =
            self.side_panel_open_pending_frames >= Self::SIDE_PANEL_OPEN_FALLBACK_FRAMES;
        if resize_landed || waited_too_long {
            self.show_exif = true;
            self.side_panel_open_pending = false;
            self.side_panel_expand_target_width = None;
        }

        ctx.request_repaint();
    }

    fn new(
        path: PathBuf,
        rx: Receiver<PathBuf>,
        ctx_shared: Arc<Mutex<Option<egui::Context>>>,
        start_on_home_page: bool,
    ) -> Self {
        let path = path.canonicalize().unwrap_or(path);
        let open_target = path.clone();
        let open_target_is_dir = start_on_home_page || path.is_dir();

        let mut images = Vec::new();
        let flat_loading = !start_on_home_page;
        let flat_images_shared = Arc::new(Mutex::new(None));

        if !start_on_home_page {
            if path.is_dir() {
                let shared = flat_images_shared.clone();
                let parent_absolute = path.clone();
                std::thread::spawn(move || {
                    let mut collected = Vec::new();
                    if let Ok(entries) = std::fs::read_dir(&parent_absolute) {
                        for entry in entries.filter_map(|e| e.ok()) {
                            let p = entry.path();
                            if p.is_file() {
                                let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
                                if matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "bmp" | "gif" | "webp" | "tiff" | "avif" | "heif" | "heic" | "ico" | "icns" | "svg" |
                                            "mp4" | "mov" | "avi" | "mkv" | "webm" | "m4v" | "wmv" | "mpg" | "mpeg") {
                                    collected.push(p);
                                }
                            }
                        }
                    }
                    collected.sort();
                    if let Ok(mut lock) = shared.lock() {
                        *lock = Some(collected);
                    }
                });
            } else {
                images.push(path.clone());
                let shared = flat_images_shared.clone();
                let parent = path.parent().unwrap_or(Path::new(".")).to_path_buf();
                let parent_absolute = parent.canonicalize().unwrap_or(parent);
                std::thread::spawn(move || {
                    let mut collected = Vec::new();
                    if let Ok(entries) = std::fs::read_dir(&parent_absolute) {
                        for entry in entries.filter_map(|e| e.ok()) {
                            let p = entry.path();
                            if p.is_file() {
                                let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
                                if matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "bmp" | "gif" | "webp" | "tiff" | "avif" | "heif" | "heic" | "ico" | "icns" | "svg" |
                                            "mp4" | "mov" | "avi" | "mkv" | "webm" | "m4v" | "wmv" | "mpg" | "mpeg") {
                                    collected.push(p);
                                }
                            }
                        }
                    }
                    collected.sort();
                    if let Ok(mut lock) = shared.lock() {
                        *lock = Some(collected);
                    }
                });
            }
        }

        let (thumbnail_tx, thumbnail_rx) = std::sync::mpsc::channel::<(PathBuf, egui::ColorImage)>();

        let mut viewer = Self {
            images,
            current_index: 0,
            zoom: 1.0,
            offset: egui::Vec2::ZERO,
            exif_data: String::new(),
            side_panel_metadata_path: None,
            side_panel_layout_path: None,
            show_exif: false,
            side_panel_window_expanded: false,
            side_panel_open_pending: false,
            side_panel_expand_target_width: None,
            side_panel_open_pending_frames: 0,
            chunks: Vec::new(),
            viewport_bg: None,
            rx,
            show_grid: false,
            recursive_images: Vec::new(),
            grid_loading: false,
            recursive_rx: None,
            back_target_is_gallery: false,
            side_panel_mode: SidePanelMode::Layout,
            exif_search: String::new(),
            open_target,
            open_target_is_dir,
            flat_loading,
            flat_images_shared,
            current_dimensions: String::new(),
            current_file_size: String::new(),
            ctx_shared,
            thumbnail_textures: std::collections::HashMap::new(),
            thumbnail_loading: std::collections::HashSet::new(),
            thumbnail_failed: std::collections::HashSet::new(),
            thumbnail_rx,
            thumbnail_tx,
            thumbnail_active_threads: 0,
            
            // AI Explorer defaults
            db_loaded: false,
            db_loading: false,
            db_supplemental_loaded: false,
            db_supplemental_loading: false,
            db_failed: false,
            db_rx: None,
            db_indices: None,
            semantic_query: String::new(),
            applied_filename_query: String::new(),
            filename_search_results: None,
            semantic_folder: String::new(),
            semantic_limit: 80,
            semantic_video_only: false,
            semantic_mode: SearchMode::Filename,
            semantic_results: Vec::new(),
            semantic_results_mode: None,
            semantic_status: "Ready. Enter a phrase and press Search.".to_string(),
            pending_search_request: None,
            pending_semantic_search_mode: None,
            on_demand_embed_rx: None,
            
            // SIFT defaults
            compare_target: None,
            sift_pair_overlay: None,
            expanded_duplicate_rows: HashSet::new(),
            sift_running: false,
            sift_rx: None,
            selected_grid_files: Vec::new(),
            sift_repair_running: false,
            sift_repair_rx: None,
            image_editor: None,

            db_filename_by_path: HashMap::new(),
            video_still_cache: std::cell::RefCell::new(HashMap::new()),
            resolution_size_cache: std::cell::RefCell::new(HashMap::new()),
            db_filename_cache: std::cell::RefCell::new(HashMap::new()),
            show_home_page: start_on_home_page,
            home_current_dir: None,
            home_selected_dir: None,
        };
        
        if !start_on_home_page {
            viewer.update_current_file_info();
        }
        viewer
    }

    fn start_lazy_db_load(&mut self, ctx: &egui::Context) {
        if self.db_loaded || self.db_loading || self.db_failed {
            return;
        }
        self.db_loading = true;
        self.db_supplemental_loaded = false;
        self.db_supplemental_loading = false;
        self.semantic_status = "Loading CLIP index and text encoder...".to_string();
        let (tx, rx) = std::sync::mpsc::channel();
        self.db_rx = Some(rx);
        let ctx_clone = ctx.clone();
        
        std::thread::spawn(move || {
            let rt = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build() {
                    Ok(rt) => rt,
                    Err(e) => {
                        let _ = tx.send(DatabaseLoadMessage::ClipReady(Err(format!(
                            "Failed to create tokio runtime: {e}"
                        ))));
                        ctx_clone.request_repaint();
                        return;
                    }
                };

            let clip_result: Result<(ClipIndex, ClipTextEncoder), anyhow::Error> = rt.block_on(async {
                let db_dir_buf = get_db_dir();
                let db_dir = db_dir_buf.as_path();
                let table_name = MEDIA_INDEX_TABLE;
                let imagesearch_dir = resolve_imagesearch_dir();
                let onnx_path_buf = imagesearch_dir.join("models/clip-text/clip_text.onnx");
                let tokenizer_path_buf = imagesearch_dir.join("models/clip-text/tokenizer.json");
                let onnx_path = onnx_path_buf.as_path();
                let tokenizer_path = tokenizer_path_buf.as_path();

                let db_fut = load_clip_database_index(db_dir, table_name);
                let encoder_fut = async {
                    ClipTextEncoder::new(onnx_path, tokenizer_path, 64)
                };
                tokio::try_join!(db_fut, encoder_fut)
            });

            let clip_loaded = clip_result.is_ok();
            let _ = tx.send(DatabaseLoadMessage::ClipReady(
                clip_result.map_err(|e| e.to_string()),
            ));
            ctx_clone.request_repaint();
            if !clip_loaded {
                return;
            }

            let supplemental_result = rt
                .block_on(load_supplemental_database_indices(
                    get_db_dir().as_path(),
                    MEDIA_INDEX_TABLE,
                ))
                .map_err(|e| e.to_string());
            let _ = tx.send(DatabaseLoadMessage::SupplementalReady(supplemental_result));
            ctx_clone.request_repaint();
        });
    }

    fn poll_db_load(&mut self) {
        let Some(rx) = self.db_rx.take() else {
            return;
        };
        match rx.try_recv() {
            Ok(DatabaseLoadMessage::ClipReady(Ok((clip_index, encoder)))) => {
                let clip_embedded_files: HashSet<String> = clip_index
                    .entries
                    .iter()
                    .map(|entry| entry.file_name.to_string())
                    .collect();
                let mut basename_to_db_filename = HashMap::new();
                for entry in &clip_index.entries {
                    if let Some(fname) = Path::new(entry.file_name.as_ref()).file_name() {
                        basename_to_db_filename
                            .entry(fname.to_string_lossy().to_lowercase())
                            .or_insert_with(|| entry.file_name.to_string());
                    }
                }
                self.db_indices = Some(DatabaseIndices {
                    clip_index: Arc::new(clip_index),
                    face_index: Arc::new(FaceIndex { entries: Vec::new(), file_count: 0 }),
                    ocr_index: Arc::new(OcrIndex { entries: Vec::new(), file_count: 0 }),
                    clip_embedded_files: Arc::new(clip_embedded_files),
                    ocr_embedded_files: Arc::new(HashSet::new()),
                    similar_by_master: HashMap::new(),
                    phash_master_by_file: HashMap::new(),
                    phash_by_file: HashMap::new(),
                    video_frame_phashes_by_file: HashMap::new(),
                    sift_info_by_file: HashMap::new(),
                    sift_root_by_file: HashMap::new(),
                    sift_members_by_root: HashMap::new(),
                    skipped_processing_files: Arc::new(HashSet::new()),
                    basename_to_db_filename,
                    encoder,
                });
                self.db_loaded = true;
                self.db_loading = false;
                self.db_supplemental_loaded = false;
                self.db_supplemental_loading = true;
                self.semantic_status =
                    "CLIP ready. Loading OCR, face, duplicate, and SIFT indexes in the background."
                        .to_string();
                self.run_pending_db_request(false);
                self.db_rx = Some(rx);
            }
            Ok(DatabaseLoadMessage::ClipReady(Err(err))) => {
                self.fail_db_load(err);
            }
            Ok(DatabaseLoadMessage::SupplementalReady(Ok(data))) => {
                if let Some(indices) = self.db_indices.as_mut() {
                    indices.face_index = Arc::new(data.face_index);
                    indices.ocr_index = Arc::new(data.ocr_index);
                    indices.ocr_embedded_files = Arc::new(data.ocr_embedded_files);
                    indices.similar_by_master = data.similar_by_master;
                    indices.phash_master_by_file = data.phash_master_by_file;
                    indices.phash_by_file = data.phash_by_file;
                    indices.video_frame_phashes_by_file = data.video_frame_phashes_by_file;
                    indices.sift_info_by_file = data.sift_info_by_file;
                    indices.sift_root_by_file = data.sift_root_by_file;
                    indices.sift_members_by_root = data.sift_members_by_root;
                    indices.skipped_processing_files = Arc::new(data.skipped_processing_files);
                    for key in indices
                        .phash_master_by_file
                        .keys()
                        .chain(indices.similar_by_master.keys())
                        .chain(indices.sift_info_by_file.keys())
                    {
                        if let Some(fname) = Path::new(key).file_name() {
                            indices
                                .basename_to_db_filename
                                .entry(fname.to_string_lossy().to_lowercase())
                                .or_insert_with(|| key.clone());
                        }
                    }
                }
                self.db_supplemental_loaded = true;
                self.db_supplemental_loading = false;
                if self.semantic_status.starts_with("CLIP ready.") {
                    self.semantic_status =
                        "CLIP, OCR, face, duplicate, and SIFT indexes ready.".to_string();
                }
                self.run_pending_db_request(true);
            }
            Ok(DatabaseLoadMessage::SupplementalReady(Err(err))) => {
                self.db_supplemental_loading = false;
                self.semantic_status = format!(
                    "CLIP is ready, but supplemental database indexes failed to load: {err}"
                );
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                self.db_rx = Some(rx);
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                if !self.db_loaded {
                    self.fail_db_load("AI DB loader thread disconnected unexpectedly.".to_string());
                } else {
                    self.db_supplemental_loading = false;
                }
            }
        }
    }

    fn fail_db_load(&mut self, err: String) {
        self.db_loading = false;
        self.db_supplemental_loaded = false;
        self.db_supplemental_loading = false;
        self.db_failed = true;
        self.pending_search_request = None;
        self.pending_semantic_search_mode = None;
        self.semantic_status = format!("AI DB initialization failed: {err}");
    }

    fn run_pending_db_request(&mut self, supplemental_ready: bool) {
        let maybe_ctx = self
            .ctx_shared
            .lock()
            .ok()
            .and_then(|lock| lock.as_ref().cloned());
        let Some(ctx) = maybe_ctx else {
            return;
        };
        if let Some(request) = self.pending_search_request.take() {
            if supplemental_ready || matches!(&request, PendingSearchRequest::Similar { .. }) {
                self.run_search_request_now(request, &ctx);
                return;
            }
            self.pending_search_request = Some(request);
        }
        if let Some(mode) = self.pending_semantic_search_mode.take() {
            if supplemental_ready || mode == SearchMode::Clip {
                self.run_semantic_search_mode(mode, &ctx);
            } else {
                self.pending_semantic_search_mode = Some(mode);
            }
        }
    }

    fn poll_on_demand_embeddings(&mut self, ctx: &egui::Context) {
        let Some(rx) = self.on_demand_embed_rx.take() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(payload)) => {
                let label = Self::label_for_request(&payload.request);
                match payload.request {
                    PendingSearchRequest::Similar {
                        db_file_name,
                        media_path,
                        is_video,
                        timestamp_sec,
                    } => {
                        if let Some(query_vector) = payload.clip_vector {
                            let source = SearchResult {
                                rank: 0,
                                score: 1.0,
                                file_name: db_file_name
                                    .clone()
                                    .unwrap_or_else(|| media_path.to_string_lossy().to_string()),
                                is_video,
                                timestamp_sec,
                                media_path: Some(media_path),
                                ocr_term_hits: 0,
                                ocr_query_terms: 0,
                                ocr_phrase_query: false,
                            };
                            self.show_most_similar_from_vector(
                                query_vector,
                                Some(source),
                                &label,
                            );
                        } else {
                            self.semantic_status =
                                format!("No CLIP embedding produced for {label}");
                        }
                    }
                    PendingSearchRequest::Person { .. } => {
                        self.show_more_of_this_person_with_vectors(payload.face_vectors, &label);
                    }
                }
                ctx.request_repaint();
            }
            Ok(Err(err)) => {
                self.semantic_status = format!("On-demand embedding failed: {err}");
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                self.on_demand_embed_rx = Some(rx);
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.semantic_status =
                    "On-demand embedding worker disconnected unexpectedly.".to_string();
            }
        }
    }

    fn get_db_filename_from_path(&self, path: &Path) -> Option<String> {
        let roots = get_db_roots();
        let path_norm = path.to_string_lossy().replace('\\', "/");

        // Fast path 0: already a DB-style relative path such as:
        //   <collection_id>/...
        // This must resolve directly to collection roots.
        let trimmed = path_norm.trim_start_matches("./").trim_start_matches('/');
        if let Some((collection, rel)) = trimmed.split_once('/') {
            if !rel.is_empty() && roots.contains_key(collection) {
                return Some(format!("{}/{}", collection, rel.trim_start_matches('/')));
            }
        }
        
        // Fast path 1: folder component substring match (extremely robust against mount path/canonicalize differences like /media vs /run/media)
        for (col_id, root_path) in &roots {
            let folder_name = root_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(col_id.as_str());
            
            let path_lower = path_norm.to_lowercase();
            let match_str_lower = format!("/{}/", folder_name.to_lowercase());
            if let Some(pos) = path_lower.find(&match_str_lower) {
                let rel = &path_norm[pos + match_str_lower.len()..];
                return Some(format!("{}/{}", col_id, rel));
            }
        }
        
        // Fast path 2: prefix-strip against known roots without touching the disk.
        for (col_id, root_path) in &roots {
            if let Ok(rel) = path.strip_prefix(root_path) {
                let rel_str = rel.to_string_lossy().replace('\\', "/");
                return Some(format!("{}/{}", col_id, rel_str.trim_start_matches('/')));
            }
        }
        
        // Secondary: substring matching (handles trailing-slash edge cases)
        let path_str = path.to_string_lossy().replace('\\', "/");
        for (col_id, root_path) in &roots {
            let root_str = root_path.to_string_lossy().replace('\\', "/");
            if path_str.starts_with(&root_str) {
                let rel = &path_str[root_str.len()..];
                return Some(format!("{}/{}", col_id, rel.trim_start_matches('/')));
            }
        }

        // Tertiary: basename lookup in loaded index.
        // Only use this for non-existing/virtual paths (e.g. synthesized preview references).
        // For real existing full paths outside mapped roots, basename matching is too ambiguous.
        if !path.exists() {
            if let Some(indices) = &self.db_indices {
                if let Some(fname) = path.file_name() {
                    let base = fname.to_string_lossy().to_lowercase();
                    if let Some(resolved) = indices.basename_to_db_filename.get(&base) {
                        return Some(resolved.clone());
                    }
                }
            }
        }
        None
    }

    /// Look up the database filename for a given filesystem path.
    /// Checks the cached `db_filename_by_path` map first (populated from AI search results),
    /// then falls back to the path-prefix heuristic in `get_db_filename_from_path`.
    fn resolve_db_filename(&self, path: &Path) -> Option<String> {
        // Fast path 1: exact match from AI search results (handles video stills, etc.)
        if let Some(name) = self.db_filename_by_path.get(path) {
            return Some(name.clone());
        }
        if let Some(name) = db_filename_from_video_still_path(path) {
            return Some(name);
        }
        // Fast path 2: Check the cache to avoid synchronous disk canonicalization and linear scans
        if let Some(cached) = self.db_filename_cache.borrow().get(path) {
            return cached.clone();
        }
        // Fallback: derive from filesystem path vs collection roots
        let resolved = self.get_db_filename_from_path(path);
        // Cache the result (even if None) to avoid repeating this heavy calculation
        self.db_filename_cache.borrow_mut().insert(path.to_path_buf(), resolved.clone());
        resolved
    }

    fn resolve_actual_path(&self, path: &Path) -> PathBuf {
        if let Some(db_name) = self.resolve_db_filename(path) {
            let roots = get_db_roots();
            if let Ok(src_path) = resolve_source_path(&roots, &db_name) {
                return src_path;
            }
        }
        path.to_path_buf()
    }

    fn get_thumbnail_path(&self, path: &Path) -> PathBuf {
        if is_video_path(path) {
            // Check cache first to avoid synchronous disk I/O on UI thread
            if let Some(cached) = self.video_still_cache.borrow().get(path) {
                return cached.clone();
            }

            if let Some(file_name) = self.resolve_db_filename(path) {
                let db_dir_buf = get_db_dir();
                let db_dir = db_dir_buf.as_path();
                let db_roots = get_db_roots();
                if let Some((collection, rel)) = file_name.split_once('/') {
                    if let Some(root) = db_roots.get(collection) {
                        let rel_path = Path::new(rel);
                        if let Ok(Some(still)) = resolve_video_still(root, db_dir, rel_path, 0.0) {
                            self.video_still_cache.borrow_mut().insert(path.to_path_buf(), still.clone());
                            return still;
                        }
                    }
                }
            }
        }
        path.to_path_buf()
    }

    fn get_file_resolution_and_size(&self, path: &Path) -> String {
        if let Some(cached) = self.resolution_size_cache.borrow().get(path) {
            return cached.clone();
        }
        let result = file_resolution_and_size(path);
        self.resolution_size_cache.borrow_mut().insert(path.to_path_buf(), result.clone());
        result
    }


    fn draw_thumbnail_async(
        &mut self,
        ui: &mut egui::Ui,
        path: &Path,
        side_thumb: f32,
    ) {
        let resolved_path = self.get_thumbnail_path(path);
        if let Some(texture) = self.thumbnail_textures.get(&resolved_path) {
            ui.add(
                egui::Image::from_texture(texture)
                    .max_size(egui::vec2(side_thumb, side_thumb))
                    .maintain_aspect_ratio(true)
            );
        } else if self.thumbnail_failed.contains(&resolved_path) {
            let (rect, _) = ui.allocate_exact_size(
                egui::vec2(side_thumb, side_thumb),
                egui::Sense::hover(),
            );
            ui.painter().rect_filled(
                rect,
                4.0,
                egui::Color32::from_gray(30),
            );
            let text = if is_video_path(path) { "📹 Video" } else { "⚠️ Error" };
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                text,
                egui::FontId::proportional(10.0),
                egui::Color32::GRAY,
            );
        } else {
            let (rect, _) = ui.allocate_exact_size(
                egui::vec2(side_thumb, side_thumb),
                egui::Sense::hover(),
            );
            ui.painter().rect_filled(
                rect,
                4.0,
                egui::Color32::from_gray(40),
            );
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "⏳ Loading",
                egui::FontId::proportional(10.0),
                egui::Color32::GRAY,
            );
            
            let max_threads = num_cpus::get().max(4);
            if !self.thumbnail_loading.contains(&resolved_path) && self.thumbnail_active_threads < max_threads {
                self.thumbnail_loading.insert(resolved_path.to_path_buf());
                self.thumbnail_active_threads += 1;
                let path_clone = resolved_path.to_path_buf();
                let tx_clone = self.thumbnail_tx.clone();
                let ctx_clone = ui.ctx().clone();
                rayon::spawn(move || {
                    if let Ok(img) = image::open(&path_clone) {
                        let thumb = img.thumbnail(128, 128);
                        let size = [thumb.width() as usize, thumb.height() as usize];
                        let pixels = thumb.to_rgba8().into_raw();
                        let color_img = egui::ColorImage::from_rgba_unmultiplied(size, &pixels);
                        let _ = tx_clone.send((path_clone, color_img));
                        ctx_clone.request_repaint();
                    } else {
                        let empty_img = egui::ColorImage::new([0, 0], Vec::new());
                        let _ = tx_clone.send((path_clone, empty_img));
                        ctx_clone.request_repaint();
                    }
                });
            }
        }
    }

    fn grouped_master_for(&self, file_name: &str, is_video: bool) -> String {
        if is_video {
            return file_name.to_string();
        }
        if let Some(indices) = &self.db_indices {
            indices.sift_root_by_file
                .get(file_name)
                .cloned()
                .unwrap_or_else(|| file_name.to_string())
        } else {
            file_name.to_string()
        }
    }

    fn run_semantic_search_mode(&mut self, mode: SearchMode, ctx: &egui::Context) {
        match mode {
            SearchMode::Filename => {}
            SearchMode::Clip => self.search_clip_now(ctx),
            SearchMode::Ocr => self.search_ocr_now(),
        }
    }

    fn apply_filename_search(&mut self) {
        self.applied_filename_query = self.semantic_query.trim().to_string();
        if self.applied_filename_query.is_empty() {
            self.filename_search_results = None;
            self.semantic_status = "Filename filter cleared.".to_string();
            return;
        }

        let query = self.applied_filename_query.to_lowercase().replace('\\', "/");
        let query_is_path = query.contains('/');
        let query_basename = query
            .rsplit('/')
            .next()
            .filter(|name| name.contains('.'));
        let roots = get_db_roots();
        let mut matches = Vec::new();
        for (index, path) in self.recursive_images.iter().enumerate() {
            let matched = if query_is_path {
                if query_basename.is_some_and(|query_name| {
                    !path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.eq_ignore_ascii_case(query_name))
                }) {
                    false
                } else {
                    let physical_path = path.to_string_lossy().replace('\\', "/").to_lowercase();
                    if partial_path_matches(&query, &physical_path) {
                        true
                    } else if let Some(db_name) = self.resolve_db_filename(path) {
                        let db_name = db_name.to_lowercase();
                        let relative_name =
                            db_name.split_once('/').map(|(_, rel)| rel).unwrap_or(&db_name);
                        partial_path_matches(&query, &db_name)
                            || partial_path_matches(&query, relative_name)
                            || db_name.split_once('/').is_some_and(|(collection, rel)| {
                                roots.get(collection).is_some_and(|root| {
                                    let full_path = root
                                        .join(rel)
                                        .to_string_lossy()
                                        .replace('\\', "/")
                                        .to_lowercase();
                                    partial_path_matches(&query, &full_path)
                                })
                            })
                    } else {
                        false
                    }
                }
            } else {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.to_lowercase().contains(&query))
            };
            if matched {
                matches.push(index);
            }
        }

        self.semantic_status = format!(
            "Filename search found {} item(s) for {}.",
            matches.len(),
            self.applied_filename_query
        );
        self.filename_search_results = Some(matches);
    }

    fn submit_semantic_search(&mut self, ctx: &egui::Context) {
        if self.semantic_mode == SearchMode::Filename {
            self.apply_filename_search();
            return;
        }

        self.pending_semantic_search_mode = Some(self.semantic_mode);
        self.semantic_results.clear();
        self.semantic_results_mode = None;
        let mode_label = match self.semantic_mode {
            SearchMode::Clip => "CLIP",
            SearchMode::Ocr => "OCR",
            SearchMode::Filename => "Filename",
        };
        self.semantic_status = format!(
            "Starting {mode_label} search for \"{}\"...",
            self.semantic_query.trim()
        );
        ctx.request_repaint();

        if self.semantic_mode == SearchMode::Ocr
            && self.db_loaded
            && !self.db_supplemental_loaded
            && !self.db_supplemental_loading
        {
            self.pending_semantic_search_mode = None;
            self.semantic_status =
                "OCR search is unavailable because supplemental database loading failed."
                    .to_string();
            return;
        }
        if !self.db_loaded
            || (self.semantic_mode == SearchMode::Ocr && !self.db_supplemental_loaded)
        {
            self.semantic_status = format!(
                "Loading AI DB for {mode_label} search of \"{}\"...",
                self.semantic_query.trim()
            );
            if self.db_failed {
                self.db_failed = false;
            }
            if !self.db_loading && !self.db_loaded {
                self.start_lazy_db_load(ctx);
            }
            return;
        }

        let mode = self.pending_semantic_search_mode.take().unwrap_or(self.semantic_mode);
        self.run_semantic_search_mode(mode, ctx);
    }

    fn default_semantic_folder(&self) -> PathBuf {
        if self.open_target_is_dir {
            self.open_target.clone()
        } else {
            self.open_target
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf()
        }
    }

    fn effective_semantic_folder(&self) -> String {
        let trimmed = self.semantic_folder.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
        String::new()
    }

    fn folder_has_db_mappings(&self, folder: &str) -> bool {
        let roots = get_db_roots();
        if roots.is_empty() {
            return false;
        }
        let trimmed = folder.trim();
        if trimmed.is_empty() {
            return roots
                .values()
                .any(|col_path| path_matches_db_root(&self.default_semantic_folder(), col_path));
        }
        let normalized = trimmed.replace('\\', "/");
        if let Some((collection_id, rel)) = normalized.split_once('/') {
            if roots.contains_key(collection_id) && !rel.trim_matches('/').is_empty() {
                return true;
            }
        }
        let folder_path = Path::new(trimmed);
        if !normalized.contains('/') && !normalized.contains('\\') && !folder_path.is_absolute() {
            return true;
        }
        let folder_path = Path::new(trimmed);
        roots
            .values()
            .any(|col_path| path_matches_db_root(folder_path, col_path) || path_matches_db_root(col_path, folder_path))
    }

    fn clip_query_to_pending_request(&self, query: &str) -> Option<PendingSearchRequest> {
        let mut raw = query.trim();
        if let Some(rest) = raw.strip_prefix("Current:") {
            raw = rest.trim();
        } else if let Some(rest) = raw.strip_prefix("current:") {
            raw = rest.trim();
        }
        if raw.is_empty() {
            return None;
        }

        let query_path = PathBuf::from(raw);
        let db_roots = get_db_roots();

        let request_from_db_name = |db_name: String| {
            let media_path = resolve_source_path(&db_roots, &db_name).unwrap_or_else(|_| query_path.clone());
            let is_video = is_video_path(&media_path) || is_video_path(Path::new(&db_name));
            PendingSearchRequest::Similar {
                db_file_name: Some(db_name),
                media_path,
                is_video,
                timestamp_sec: 0.0,
            }
        };

        if let Some(db_name) = self.resolve_db_filename(&query_path) {
            let resolved_exists = resolve_source_path(&db_roots, &db_name)
                .map(|path| path.exists())
                .unwrap_or(false);
            if resolved_exists {
                return Some(request_from_db_name(db_name));
            }
        }

        let normalized = raw
            .replace('\\', "/")
            .trim_start_matches("./")
            .trim_start_matches('/')
            .to_string();
        if let Some((collection, rel)) = normalized.split_once('/') {
            if !rel.is_empty() && db_roots.contains_key(collection) && is_supported_media_path(Path::new(rel)) {
                let db_name = format!("{}/{}", collection, rel.trim_start_matches('/'));
                let resolved_exists = resolve_source_path(&db_roots, &db_name)
                    .map(|path| path.exists())
                    .unwrap_or(false);
                if resolved_exists {
                    return Some(request_from_db_name(db_name));
                }
            }
        }

        if query_path.exists() && is_supported_media_path(&query_path) {
            let is_video = is_video_path(&query_path);
            return Some(PendingSearchRequest::Similar {
                db_file_name: None,
                media_path: query_path,
                is_video,
                timestamp_sec: 0.0,
            });
        }

        if let Some(indices) = &self.db_indices {
            if let Some(base) = query_path.file_name() {
                let base = base.to_string_lossy().to_lowercase();
                if let Some(db_name) = indices.basename_to_db_filename.get(&base) {
                    return Some(request_from_db_name(db_name.clone()));
                }
            }
        }

        None
    }

    fn search_clip_now(&mut self, ctx: &egui::Context) {
        let q = self.semantic_query.trim().to_string();
        let folder_scope = self.effective_semantic_folder();
        if !self.db_supplemental_loaded {
            self.pending_semantic_search_mode = Some(SearchMode::Ocr);
            self.semantic_status = if self.db_supplemental_loading {
                "Loading OCR index in the background...".to_string()
            } else {
                "OCR index is unavailable because supplemental database loading failed.".to_string()
            };
            return;
        }
        if q.is_empty() {
            self.semantic_status = "Please enter a search phrase first.".to_string();
            self.semantic_results.clear();
            self.semantic_results_mode = None;
            return;
        }

        if let Some(request) = self.clip_query_to_pending_request(&q) {
            self.semantic_results.clear();
            self.run_search_request_now(request, ctx);
            return;
        }

        let Some(indices) = &mut self.db_indices else {
            self.semantic_status = "AI Database index is not loaded yet.".to_string();
            return;
        };

        let started = Instant::now();
        let query_vector = match indices.encoder.embed(&q) {
            Ok(vec) => vec,
            Err(err) => {
                self.semantic_status = format!("❌ Text Embedding failed: {err}");
                return;
            }
        };

        if query_vector.len() != indices.clip_index.dim {
            self.semantic_status = format!(
                "❌ Error: Query dim {} does not match index dim {}",
                query_vector.len(),
                indices.clip_index.dim
            );
            return;
        }

        let pre_limit = (self.semantic_limit.saturating_mul(6)).max(self.semantic_limit);
        let mut results = search_index(&indices.clip_index, &query_vector, pre_limit, self.semantic_video_only, &folder_scope);
        if !self.semantic_video_only {
            results = collapse_sift_grouped_results(results, &indices.sift_root_by_file, self.semantic_limit);
        } else {
            results.truncate(self.semantic_limit);
        }
        
        let db_roots = get_db_roots();
        let db_dir_buf = get_db_dir();
        let db_dir = db_dir_buf.as_path();
        for row in &mut results {
            row.media_path = resolve_media_path(&db_roots, db_dir, &row.file_name, row.timestamp_sec).ok();
            if let Some(path) = &row.media_path {
                self.db_filename_by_path.insert(path.clone(), row.file_name.clone());
            }
        }

        let took = started.elapsed().as_millis();
        self.semantic_status = format!(
            "✓ Found {} CLIP results in {} ms across {} index vectors within {}",
            results.len(),
            took,
            indices.clip_index.entries.len(),
            folder_scope
        );
        self.semantic_results = results;
        self.semantic_results_mode = Some(SearchMode::Clip);
    }

    fn search_clip_from_clipboard_image(
        &mut self,
        ctx: &egui::Context,
        pasted_text: Option<&str>,
        report_no_image: bool,
    ) {
        let path = match save_clipboard_image_to_temp(pasted_text) {
            Ok(Some(path)) => path,
            Ok(None) => {
                if report_no_image {
                    self.semantic_status =
                        "Clipboard does not contain an image or image file path.".to_string();
                }
                return;
            }
            Err(err) => {
                self.semantic_status = format!("Clipboard image paste failed: {err}");
                return;
            }
        };

        let request = PendingSearchRequest::Similar {
            db_file_name: None,
            media_path: path.clone(),
            is_video: false,
            timestamp_sec: 0.0,
        };
        self.semantic_query = "clipboard image".to_string();
        self.semantic_results.clear();
        self.semantic_results_mode = None;
        self.semantic_status = format!(
            "Computing CLIP embedding for pasted clipboard image {}...",
            path.file_name().and_then(|name| name.to_str()).unwrap_or("clipboard image")
        );
        self.request_search_action(request, ctx);
    }

    fn search_ocr_now(&mut self) {
        let q = self.semantic_query.trim().to_string();
        let folder_scope = self.effective_semantic_folder();
        if q.is_empty() {
            self.semantic_status = "Please enter an OCR word or phrase first.".to_string();
            self.semantic_results.clear();
            self.semantic_results_mode = None;
            return;
        }
        let Some(indices) = &self.db_indices else {
            self.semantic_status = "AI Database index is not loaded yet.".to_string();
            return;
        };

        let started = Instant::now();
        let pre_limit = (self.semantic_limit.saturating_mul(6)).max(self.semantic_limit);
        let mut results = search_ocr_index(&indices.ocr_index, &q, pre_limit, self.semantic_video_only, &folder_scope);
        if !self.semantic_video_only {
            results = collapse_sift_grouped_results(results, &indices.sift_root_by_file, self.semantic_limit);
        } else {
            results.truncate(self.semantic_limit);
        }
        
        let db_roots = get_db_roots();
        let db_dir_buf = get_db_dir();
        let db_dir = db_dir_buf.as_path();
        for row in &mut results {
            row.media_path = resolve_media_path(&db_roots, db_dir, &row.file_name, row.timestamp_sec).ok();
            if let Some(path) = &row.media_path {
                self.db_filename_by_path.insert(path.clone(), row.file_name.clone());
            }
        }

        let took = started.elapsed().as_millis();
        self.semantic_status = format!(
            "✓ Found {} OCR results in {} ms across {} index entries within {}",
            results.len(),
            took,
            indices.ocr_index.entries.len(),
            folder_scope
        );
        self.semantic_results = results;
        self.semantic_results_mode = Some(SearchMode::Ocr);
    }

    fn start_sift_alignment(&mut self, path_a: PathBuf, path_b: PathBuf, ctx: egui::Context) {
        if self.sift_running {
            return;
        }
        self.sift_running = true;
        self.sift_pair_overlay = None;
        let (tx, rx) = std::sync::mpsc::channel();
        self.sift_rx = Some(rx);
        
        std::thread::spawn(move || {
            let result = compute_sift_summary(&path_a, &path_b)
                .map_err(|e| e.to_string());
            let _ = tx.send(result);
            ctx.request_repaint();
        });
    }

    fn poll_sift_alignment(&mut self) {
        let Some(rx) = self.sift_rx.take() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(summary)) => {
                self.sift_running = false;
                self.sift_pair_overlay = Some(summary);
            }
            Ok(Err(err)) => {
                self.sift_running = false;
                self.sift_pair_overlay = Some(format!("❌ SIFT Alignment failed: {err}"));
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                self.sift_rx = Some(rx);
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.sift_running = false;
                self.sift_pair_overlay = Some("❌ SIFT calculation worker disconnected.".to_string());
            }
        }
    }

    fn start_selected_sift_repair(&mut self, ctx: &egui::Context) {
        if self.sift_repair_running {
            return;
        }
        if self.selected_grid_files.len() < 2 {
            self.semantic_status = "Select at least two indexed images before running SIFT repair.".to_string();
            return;
        }
        if !self.db_loaded || !self.db_supplemental_loaded {
            if !self.db_loading && !self.db_failed {
                self.start_lazy_db_load(ctx);
            }
            self.semantic_status = "Loading duplicate and SIFT indexes before SIFT repair...".to_string();
            return;
        }

        let file_names = self.expanded_sift_repair_selection();
        let selected_count = self.selected_grid_files.len();
        let repair_count = file_names.len();
        let (tx, rx) = std::sync::mpsc::channel();
        self.sift_repair_rx = Some(rx);
        self.sift_repair_running = true;
        self.semantic_status = format!(
            "Running SIFT repair on {selected_count} selected images ({repair_count} including current SIFT groups)..."
        );
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = run_sift_repair_for_files(&file_names).map_err(|err| err.to_string());
            let _ = tx.send(result);
            ctx.request_repaint();
        });
    }

    fn expanded_sift_repair_selection(&self) -> Vec<String> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        let Some(indices) = &self.db_indices else {
            return self.selected_grid_files.clone();
        };

        for file_name in &self.selected_grid_files {
            if seen.insert(file_name.clone()) {
                out.push(file_name.clone());
            }
            let root = indices
                .sift_root_by_file
                .get(file_name)
                .cloned()
                .unwrap_or_else(|| file_name.clone());
            if let Some(members) = indices.sift_members_by_root.get(root.as_str()) {
                for member in members {
                    if seen.insert(member.clone()) {
                        out.push(member.clone());
                    }
                }
            }
        }

        out
    }

    fn start_selected_sift_compare(&mut self, ctx: &egui::Context) {
        if self.selected_grid_files.len() < 2 {
            self.semantic_status = "Select two indexed images before running SIFT compare.".to_string();
            return;
        }

        let file_a = self.selected_grid_files[0].clone();
        let file_b = self.selected_grid_files[1].clone();
        let roots = get_db_roots();
        let path_a = match resolve_source_path(&roots, &file_a) {
            Ok(path) => path,
            Err(err) => {
                self.semantic_status = format!("SIFT compare failed to resolve first image: {err}");
                return;
            }
        };
        let path_b = match resolve_source_path(&roots, &file_b) {
            Ok(path) => path,
            Err(err) => {
                self.semantic_status = format!("SIFT compare failed to resolve second image: {err}");
                return;
            }
        };

        self.images = vec![path_a.clone(), path_b.clone()];
        self.current_index = 0;
        self.compare_target = Some(path_b.clone());
        self.show_grid = false;
        self.back_target_is_gallery = true;
        self.zoom = 1.0;
        self.offset = egui::Vec2::ZERO;
        self.start_sift_alignment(path_a, path_b, ctx.clone());
    }

    fn poll_sift_repair(&mut self, ctx: &egui::Context) {
        let Some(rx) = self.sift_repair_rx.take() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(result)) => {
                self.sift_repair_running = false;
                self.selected_grid_files.clear();
                self.compare_target = None;
                self.sift_pair_overlay = None;
                self.db_loaded = false;
                self.db_loading = false;
                self.db_supplemental_loaded = false;
                self.db_supplemental_loading = false;
                self.db_failed = false;
                self.db_indices = None;
                self.db_rx = None;
                self.start_lazy_db_load(ctx);
                self.semantic_status = format!(
                    "{} Reloading database maps after repairing {} selected files.",
                    result.summary, result.files
                );
            }
            Ok(Err(err)) => {
                self.sift_repair_running = false;
                self.semantic_status = format!("SIFT repair failed: {err}");
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                self.sift_repair_rx = Some(rx);
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.sift_repair_running = false;
                self.semantic_status = "SIFT repair worker disconnected unexpectedly.".to_string();
            }
        }
    }

    fn start_recursive_scan(&mut self) {
        self.grid_loading = true;
        self.recursive_images.clear();
        self.applied_filename_query.clear();
        self.filename_search_results = None;
        self.thumbnail_textures.clear();
        self.thumbnail_loading.clear();
        self.thumbnail_failed.clear();
        self.thumbnail_active_threads = 0;

        let start_dir = if self.open_target_is_dir {
            self.open_target.clone()
        } else {
            self.open_target.parent().unwrap_or_else(|| Path::new(".")).to_path_buf()
        };

        let (tx, rx) = std::sync::mpsc::channel();
        self.recursive_rx = Some(rx);

        std::thread::spawn(move || {
            let start_dir_canon = start_dir.canonicalize().unwrap_or(start_dir);
            let mut visited = std::collections::HashSet::new();
            collect_images_recursive(&start_dir_canon, &tx, &mut visited);
        });
    }

    fn open_image_path(&mut self, path: PathBuf) {
        self.open_path(path, None);
    }

    fn open_folder_path(&mut self, path: PathBuf) {
        self.open_path(path, Some(true));
    }

    fn open_path(&mut self, path: PathBuf, known_is_dir: Option<bool>) {
        let old_start_dir = if self.open_target_is_dir {
            self.open_target.clone()
        } else {
            self.open_target.parent().unwrap_or_else(|| Path::new(".")).to_path_buf()
        };
        let old_start_dir_norm = normalized_path_for_match(&old_start_dir);

        let path_is_dir = known_is_dir.unwrap_or_else(|| path.is_dir());
        self.open_target = path.clone();
        self.open_target_is_dir = path_is_dir;
        self.zoom = 1.0;
        self.offset = egui::Vec2::ZERO;

        let new_start_dir = if self.open_target_is_dir {
            self.open_target.clone()
        } else {
            self.open_target.parent().unwrap_or_else(|| Path::new(".")).to_path_buf()
        };
        let new_start_dir_norm = normalized_path_for_match(&new_start_dir);

        if old_start_dir_norm != new_start_dir_norm {
            self.recursive_images.clear();
            self.back_target_is_gallery = false;
        }

        if path_is_dir {
            self.images.clear();
            self.current_index = 0;
            self.update_current_file_info();
            self.flat_loading = false;
            if let Ok(mut lock) = self.flat_images_shared.lock() {
                *lock = None;
            }
        } else {
            self.images = vec![path.clone()];
            self.current_index = 0;
            self.update_current_file_info();
            self.flat_loading = true;
            if let Ok(mut lock) = self.flat_images_shared.lock() {
                *lock = None;
            }

            let shared = self.flat_images_shared.clone();
            let parent = path.parent().unwrap_or(Path::new(".")).to_path_buf();
            std::thread::spawn(move || {
                let parent_absolute = parent.canonicalize().unwrap_or(parent);
                let mut collected = Vec::new();
                if let Ok(entries) = std::fs::read_dir(&parent_absolute) {
                    for entry in entries.filter_map(|e| e.ok()) {
                        let p = entry.path();
                        if p.is_file() {
                            let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
                            if matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "bmp" | "gif" | "webp" | "tiff" | "avif" | "heif" | "heic" | "ico" | "icns" | "svg") {
                                collected.push(p);
                            }
                        }
                    }
                }
                collected.sort();
                if let Ok(mut lock) = shared.lock() {
                    *lock = Some(collected);
                }
            });
        }
    }

    fn update_exif(&mut self) {
        self.update_current_file_info();
        if let Some(path) = self.images.get(self.current_index).cloned() {
            let resolved_path = self.resolve_actual_path(&path);
            let inspect_path: &Path = if resolved_path.exists() {
                resolved_path.as_path()
            } else {
                path.as_path()
            };

            let exiftool_data = if !inspect_path.exists() {
                format!("Resolved file does not exist: {}", inspect_path.display())
            } else if let Some(exiftool_path) = resolve_exiftool_path() {
                match Command::new(&exiftool_path)
                    .args(["-a", "-u", "-g1", "-H"])
                    .arg(inspect_path)
                    .output()
                {
                    Ok(out) => {
                        let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                        if !stdout.trim().is_empty() {
                            stdout
                        } else {
                            let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
                            if stderr.is_empty() {
                                format!(
                                    "exiftool produced no output for {}",
                                    inspect_path.display()
                                )
                            } else {
                                format!("exiftool error: {}", stderr)
                            }
                        }
                    }
                    Err(e) => format!(
                        "Error running exiftool at {}: {}",
                        exiftool_path.display(),
                        e
                    ),
                }
            } else {
                "Error running exiftool: executable not found. Set IRIS_EXIFTOOL or install exiftool.".to_string()
            };
            self.exif_data = if inspect_path.exists() && is_video_path(inspect_path) {
                format!(
                    "{}\n\n---- FFprobe JSON ----\n{}",
                    exiftool_data.trim_end(),
                    load_ffprobe_metadata(inspect_path)
                )
            } else {
                exiftool_data
            };

            if is_video_path(inspect_path) {
                self.chunks = vec![FileChunk {
                    name: "Video File".to_string(),
                    offset: 0,
                    length: std::fs::metadata(inspect_path)
                        .map(|m| m.len().min(usize::MAX as u64) as usize)
                        .unwrap_or(0),
                    description: "Video files do not use the image binary layout parser.".to_string(),
                    color: egui::Color32::from_rgb(140, 150, 170),
                    parsed_data: "Use Raw EXIF to view exiftool and ffprobe metadata for this video.".to_string(),
                }];
            } else if let Ok(bytes) = std::fs::read(inspect_path) {
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

                let system_block = extract_system_block(&self.exif_data);
                chunks.insert(0, FileChunk {
                    name: "System Metadata".to_string(),
                    offset: 0,
                    length: 0,
                    description: "Operating system-level file attributes, timestamps, and permissions.".to_string(),
                    color: egui::Color32::from_rgb(140, 150, 170), // Slate gray
                    parsed_data: system_block,
                });

                self.chunks = chunks;
            } else {
                self.chunks = Vec::new();
            }
            self.side_panel_layout_path = Some(path.clone());
            self.side_panel_metadata_path = Some(path);
        } else {
            self.exif_data = String::new();
            self.chunks = Vec::new();
            self.current_dimensions = String::new();
            self.current_file_size = String::new();
            self.side_panel_metadata_path = None;
            self.side_panel_layout_path = None;
        }
    }

    fn update_current_file_info(&mut self) {
        if let Some(path) = self.images.get(self.current_index) {
            let resolved_path = self.resolve_actual_path(path);
            let inspect_path: &Path = if resolved_path.exists() {
                resolved_path.as_path()
            } else {
                path.as_path()
            };

            self.current_dimensions = if is_video_path(inspect_path) {
                "Video".to_string()
            } else {
                match image::image_dimensions(inspect_path) {
                    Ok((w, h)) => format!("{}x{}", w, h),
                    Err(_) => "Unknown px".to_string(),
                }
            };

            self.current_file_size = std::fs::metadata(inspect_path)
                .map(|m| {
                    let bytes = m.len();
                    if bytes >= 1_048_576 {
                        format!("{:.2} MB", bytes as f64 / 1_048_576.0)
                    } else if bytes >= 1024 {
                        format!("{:.1} KB", bytes as f64 / 1024.0)
                    } else {
                        format!("{} B", bytes)
                    }
                })
                .unwrap_or_else(|_| "Unknown size".to_string());
        } else {
            self.current_dimensions = String::new();
            self.current_file_size = String::new();
        }
    }

    fn editor_texture(ctx: &egui::Context, image: &image::DynamicImage) -> egui::TextureHandle {
        let rgba = image.to_rgba8();
        let color_image = egui::ColorImage::from_rgba_unmultiplied(
            [rgba.width() as usize, rgba.height() as usize],
            rgba.as_raw(),
        );
        ctx.load_texture("image_editor_preview", color_image, egui::TextureOptions::LINEAR)
    }

    fn start_image_editor(&mut self, path: &Path, ctx: &egui::Context) {
        let source_path = self.resolve_actual_path(path);
        match image::open(&source_path) {
            Ok(image) => {
                self.image_editor = Some(ImageEditor {
                    texture: Self::editor_texture(ctx, &image),
                    source_path,
                    image,
                    crop_min: egui::pos2(0.0, 0.0),
                    crop_max: egui::pos2(1.0, 1.0),
                    crop_drag_mode: None,
                    crop_drag_origin: egui::Pos2::ZERO,
                    crop_drag_initial_min: egui::Pos2::ZERO,
                    crop_drag_initial_max: egui::pos2(1.0, 1.0),
                    status: String::new(),
                });
            }
            Err(err) => {
                self.semantic_status = format!("Unable to open image for editing: {err}");
            }
        }
    }

    fn edited_copy_path(source: &Path) -> PathBuf {
        let stem = source.file_stem().and_then(|part| part.to_str()).unwrap_or("image");
        let extension = source.extension().and_then(|part| part.to_str()).unwrap_or("png");
        let parent = source.parent().unwrap_or_else(|| Path::new("."));
        let mut candidate = parent.join(format!("{stem}_edited.{extension}"));
        let mut suffix = 2;
        while candidate.exists() {
            candidate = parent.join(format!("{stem}_edited_{suffix}.{extension}"));
            suffix += 1;
        }
        candidate
    }

    fn save_editor_image(editor: &ImageEditor, destination: &Path, overwrite: bool) -> Result<()> {
        let image_width = editor.image.width();
        let image_height = editor.image.height();
        let left = (editor.crop_min.x * image_width as f32).round() as u32;
        let top = (editor.crop_min.y * image_height as f32).round() as u32;
        let right = (editor.crop_max.x * image_width as f32).round() as u32;
        let bottom = (editor.crop_max.y * image_height as f32).round() as u32;
        let width = right.saturating_sub(left);
        let height = bottom.saturating_sub(top);
        if width == 0 || height == 0 {
            bail!("Crop area cannot be empty");
        }
        let cropped = editor.image.crop_imm(left, top, width, height);
        let format = image::ImageFormat::from_path(destination)
            .with_context(|| format!("Unsupported output format: {}", destination.display()))?;
        if overwrite {
            let extension = destination.extension().and_then(|part| part.to_str()).unwrap_or("png");
            let temp_path = destination.with_file_name(format!(
                ".{}.iris-edit-tmp.{extension}",
                destination.file_stem().and_then(|part| part.to_str()).unwrap_or("image")
            ));
            cropped.save_with_format(&temp_path, format)?;
            std::fs::rename(&temp_path, destination)?;
        } else {
            cropped.save_with_format(destination, format)?;
        }
        Ok(())
    }

    fn refresh_after_image_edit(&mut self, path: &Path, ctx: &egui::Context) {
        self.thumbnail_textures.remove(path);
        self.thumbnail_failed.remove(path);
        self.resolution_size_cache.borrow_mut().remove(path);
        ctx.forget_image(&format!("file://{}", path.to_string_lossy()));
        self.update_current_file_info();
        self.update_side_panel_metadata_if_needed();
        ctx.request_repaint();
    }

    fn show_image_editor(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let Some(mut editor) = self.image_editor.take() else {
            return;
        };
        let mut close_editor = false;
        egui::Frame::NONE
            .fill(egui::Color32::from_black_alpha(210))
            .inner_margin(8.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.strong("Crop image");
                    ui.separator();
                    if ui.button("Rotate left").clicked() {
                        editor.image = editor.image.rotate270();
                        editor.crop_min = egui::pos2(0.0, 0.0);
                        editor.crop_max = egui::pos2(1.0, 1.0);
                        editor.texture = Self::editor_texture(ctx, &editor.image);
                    }
                    if ui.button("Rotate right").clicked() {
                        editor.image = editor.image.rotate90();
                        editor.crop_min = egui::pos2(0.0, 0.0);
                        editor.crop_max = egui::pos2(1.0, 1.0);
                        editor.texture = Self::editor_texture(ctx, &editor.image);
                    }
                    if ui.button("Rotate 180").clicked() {
                        editor.image = editor.image.rotate180();
                        editor.texture = Self::editor_texture(ctx, &editor.image);
                    }
                    if ui.button("Reset crop").clicked() {
                        editor.crop_min = egui::pos2(0.0, 0.0);
                        editor.crop_max = egui::pos2(1.0, 1.0);
                    }
                    if ui.button("Fit width").clicked() {
                        editor.crop_min.x = 0.0;
                        editor.crop_max.x = 1.0;
                    }
                    if ui.button("Fit height").clicked() {
                        editor.crop_min.y = 0.0;
                        editor.crop_max.y = 1.0;
                    }
                    ui.separator();
                    if ui.button("Save in place").clicked() {
                        match Self::save_editor_image(&editor, &editor.source_path, true) {
                            Ok(()) => {
                                let path = editor.source_path.clone();
                                self.refresh_after_image_edit(&path, ctx);
                                close_editor = true;
                            }
                            Err(err) => editor.status = format!("Save failed: {err}"),
                        }
                    }
                    if ui.button("Save edited copy").clicked() {
                        let destination = Self::edited_copy_path(&editor.source_path);
                        match Self::save_editor_image(&editor, &destination, false) {
                            Ok(()) => {
                                editor.status = format!("Saved {}", destination.display());
                                self.recursive_images.push(destination);
                            }
                            Err(err) => editor.status = format!("Save failed: {err}"),
                        }
                    }
                    if ui.button("Cancel").clicked() {
                        close_editor = true;
                    }
                });

                ui.label("Drag outside the selection to create a crop, inside it to move, or its edges and corners to resize.");
                if !editor.status.is_empty() {
                    ui.label(&editor.status);
                }

                let available = ui.available_size();
                let image_aspect = editor.image.width() as f32 / editor.image.height() as f32;
                let available_aspect = available.x / available.y.max(1.0);
                let draw_size = if available_aspect > image_aspect {
                    egui::vec2(available.y * image_aspect, available.y)
                } else {
                    egui::vec2(available.x, available.x / image_aspect)
                };
                let (viewport_rect, _) = ui.allocate_exact_size(available, egui::Sense::hover());
                let image_rect = egui::Rect::from_center_size(viewport_rect.center(), draw_size);
                let crop_response = ui.interact(
                    image_rect.expand(12.0),
                    ui.make_persistent_id("embedded_image_crop"),
                    egui::Sense::click_and_drag(),
                );
                ui.painter().image(
                    editor.texture.id(),
                    image_rect,
                    egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );

                let pointer_to_normalized = |pointer: egui::Pos2| {
                    egui::pos2(
                        ((pointer.x - image_rect.left()) / image_rect.width()).clamp(0.0, 1.0),
                        ((pointer.y - image_rect.top()) / image_rect.height()).clamp(0.0, 1.0),
                    )
                };
                let crop_rect = egui::Rect::from_min_max(
                    egui::pos2(
                        image_rect.left() + editor.crop_min.x * image_rect.width(),
                        image_rect.top() + editor.crop_min.y * image_rect.height(),
                    ),
                    egui::pos2(
                        image_rect.left() + editor.crop_max.x * image_rect.width(),
                        image_rect.top() + editor.crop_max.y * image_rect.height(),
                    ),
                );
                let handle_distance = 14.0;
                if crop_response.hovered() {
                    if let Some(pointer) = ctx.pointer_hover_pos() {
                        let near_left = (pointer.x - crop_rect.left()).abs() <= handle_distance;
                        let near_right = (pointer.x - crop_rect.right()).abs() <= handle_distance;
                        let near_top = (pointer.y - crop_rect.top()).abs() <= handle_distance;
                        let near_bottom = (pointer.y - crop_rect.bottom()).abs() <= handle_distance;
                        let within_x = pointer.x >= crop_rect.left() - handle_distance
                            && pointer.x <= crop_rect.right() + handle_distance;
                        let within_y = pointer.y >= crop_rect.top() - handle_distance
                            && pointer.y <= crop_rect.bottom() + handle_distance;
                        let cursor = if (near_left && near_top) || (near_right && near_bottom) {
                            egui::CursorIcon::ResizeNwSe
                        } else if (near_right && near_top) || (near_left && near_bottom) {
                            egui::CursorIcon::ResizeNeSw
                        } else if (near_left || near_right) && within_y {
                            egui::CursorIcon::ResizeHorizontal
                        } else if (near_top || near_bottom) && within_x {
                            egui::CursorIcon::ResizeVertical
                        } else if crop_rect.contains(pointer) {
                            egui::CursorIcon::Move
                        } else {
                            egui::CursorIcon::Crosshair
                        };
                        ctx.set_cursor_icon(cursor);
                    }
                }
                if crop_response.drag_started() {
                    if let Some(pointer) = crop_response.interact_pointer_pos() {
                        let near_left = (pointer.x - crop_rect.left()).abs() <= handle_distance;
                        let near_right = (pointer.x - crop_rect.right()).abs() <= handle_distance;
                        let near_top = (pointer.y - crop_rect.top()).abs() <= handle_distance;
                        let near_bottom = (pointer.y - crop_rect.bottom()).abs() <= handle_distance;
                        let within_x = pointer.x >= crop_rect.left() - handle_distance
                            && pointer.x <= crop_rect.right() + handle_distance;
                        let within_y = pointer.y >= crop_rect.top() - handle_distance
                            && pointer.y <= crop_rect.bottom() + handle_distance;
                        editor.crop_drag_mode = Some(if near_left && near_top {
                            CropDragMode::TopLeft
                        } else if near_right && near_top {
                            CropDragMode::TopRight
                        } else if near_left && near_bottom {
                            CropDragMode::BottomLeft
                        } else if near_right && near_bottom {
                            CropDragMode::BottomRight
                        } else if near_left && within_y {
                            CropDragMode::Left
                        } else if near_right && within_y {
                            CropDragMode::Right
                        } else if near_top && within_x {
                            CropDragMode::Top
                        } else if near_bottom && within_x {
                            CropDragMode::Bottom
                        } else if crop_rect.contains(pointer) {
                            CropDragMode::Move
                        } else {
                            CropDragMode::New
                        });
                        editor.crop_drag_origin = pointer_to_normalized(pointer);
                        editor.crop_drag_initial_min = editor.crop_min;
                        editor.crop_drag_initial_max = editor.crop_max;
                    }
                }
                if crop_response.dragged() {
                    if let (Some(mode), Some(pointer)) =
                        (editor.crop_drag_mode, crop_response.interact_pointer_pos())
                    {
                        let current = pointer_to_normalized(pointer);
                        let origin = editor.crop_drag_origin;
                        let initial_min = editor.crop_drag_initial_min;
                        let initial_max = editor.crop_drag_initial_max;
                        let minimum = egui::vec2(
                            2.0 / editor.image.width().max(1) as f32,
                            2.0 / editor.image.height().max(1) as f32,
                        );
                        match mode {
                            CropDragMode::New => {
                                editor.crop_min =
                                    egui::pos2(origin.x.min(current.x), origin.y.min(current.y));
                                editor.crop_max =
                                    egui::pos2(origin.x.max(current.x), origin.y.max(current.y));
                            }
                            CropDragMode::Move => {
                                let size = initial_max - initial_min;
                                let delta = current - origin;
                                let mut min = initial_min + delta;
                                min.x = min.x.clamp(0.0, 1.0 - size.x);
                                min.y = min.y.clamp(0.0, 1.0 - size.y);
                                editor.crop_min = min;
                                editor.crop_max = min + size;
                            }
                            CropDragMode::Left | CropDragMode::TopLeft | CropDragMode::BottomLeft => {
                                editor.crop_min.x =
                                    current.x.clamp(0.0, editor.crop_max.x - minimum.x);
                            }
                            CropDragMode::Right | CropDragMode::TopRight | CropDragMode::BottomRight => {
                                editor.crop_max.x =
                                    current.x.clamp(editor.crop_min.x + minimum.x, 1.0);
                            }
                            _ => {}
                        }
                        match mode {
                            CropDragMode::Top | CropDragMode::TopLeft | CropDragMode::TopRight => {
                                editor.crop_min.y =
                                    current.y.clamp(0.0, editor.crop_max.y - minimum.y);
                            }
                            CropDragMode::Bottom
                            | CropDragMode::BottomLeft
                            | CropDragMode::BottomRight => {
                                editor.crop_max.y =
                                    current.y.clamp(editor.crop_min.y + minimum.y, 1.0);
                            }
                            _ => {}
                        }
                    }
                }
                if crop_response.drag_stopped() {
                    editor.crop_drag_mode = None;
                }
                let shade = egui::Color32::from_black_alpha(150);
                ui.painter().rect_filled(
                    egui::Rect::from_min_max(image_rect.min, egui::pos2(image_rect.right(), crop_rect.top())),
                    0.0,
                    shade,
                );
                ui.painter().rect_filled(
                    egui::Rect::from_min_max(egui::pos2(image_rect.left(), crop_rect.bottom()), image_rect.max),
                    0.0,
                    shade,
                );
                ui.painter().rect_filled(
                    egui::Rect::from_min_max(egui::pos2(image_rect.left(), crop_rect.top()), egui::pos2(crop_rect.left(), crop_rect.bottom())),
                    0.0,
                    shade,
                );
                ui.painter().rect_filled(
                    egui::Rect::from_min_max(egui::pos2(crop_rect.right(), crop_rect.top()), egui::pos2(image_rect.right(), crop_rect.bottom())),
                    0.0,
                    shade,
                );
                ui.painter().rect_stroke(
                    crop_rect,
                    0.0,
                    egui::Stroke::new(2.0, egui::Color32::WHITE),
                    egui::StrokeKind::Inside,
                );
                for handle in [
                    crop_rect.left_top(),
                    crop_rect.right_top(),
                    crop_rect.left_bottom(),
                    crop_rect.right_bottom(),
                    egui::pos2(crop_rect.center().x, crop_rect.top()),
                    egui::pos2(crop_rect.center().x, crop_rect.bottom()),
                    egui::pos2(crop_rect.left(), crop_rect.center().y),
                    egui::pos2(crop_rect.right(), crop_rect.center().y),
                ] {
                    ui.painter().circle_filled(handle, 5.0, egui::Color32::WHITE);
                    ui.painter().circle_stroke(
                        handle,
                        5.0,
                        egui::Stroke::new(1.0, egui::Color32::BLACK),
                    );
                }
            });
        if !close_editor {
            self.image_editor = Some(editor);
        }
    }

    fn update_side_panel_metadata_if_needed(&mut self) {
        if self.side_panel_mode != SidePanelMode::Exif {
            return;
        }
        let current_path = self.images.get(self.current_index).cloned();
        if self.side_panel_metadata_path != current_path {
            self.update_exif();
        }
    }

    fn update_layout_if_needed(&mut self) {
        if self.side_panel_mode != SidePanelMode::Layout {
            return;
        }
        let Some(path) = self.images.get(self.current_index).cloned() else {
            self.chunks = Vec::new();
            self.side_panel_layout_path = None;
            return;
        };
        if self.side_panel_layout_path.as_ref() == Some(&path) {
            return;
        }

        self.update_current_file_info();
        let resolved_path = self.resolve_actual_path(&path);
        let inspect_path: &Path = if resolved_path.exists() {
            resolved_path.as_path()
        } else {
            path.as_path()
        };

        if is_video_path(inspect_path) {
            self.chunks = vec![FileChunk {
                name: "Video File".to_string(),
                offset: 0,
                length: std::fs::metadata(inspect_path)
                    .map(|m| m.len().min(usize::MAX as u64) as usize)
                    .unwrap_or(0),
                description: "Video files do not use the image binary layout parser.".to_string(),
                color: egui::Color32::from_rgb(140, 150, 170),
                parsed_data: "Use Raw EXIF to load exiftool and ffprobe metadata for this video.".to_string(),
            }];
            self.side_panel_layout_path = Some(path);
            return;
        }

        if let Ok(bytes) = std::fs::read(inspect_path) {
            let chunks = if let Some(chunks) = parse_png(&bytes) {
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
            self.chunks = chunks;
        } else {
            self.chunks = Vec::new();
        }
        self.side_panel_layout_path = Some(path);
    }

    fn next_image(&mut self) {
        if !self.images.is_empty() {
            self.current_index = (self.current_index + 1) % self.images.len();
            self.update_current_file_info();
            self.update_side_panel_metadata_if_needed();
        }
    }

    fn prev_image(&mut self) {
        if !self.images.is_empty() {
            if self.current_index == 0 {
                self.current_index = self.images.len() - 1;
            } else {
                self.current_index -= 1;
            }
            self.update_current_file_info();
            self.update_side_panel_metadata_if_needed();
        }
    }

    fn current_folder_has_db_mappings(&self) -> bool {
        self.folder_has_db_mappings(&self.default_semantic_folder().to_string_lossy())
    }

    fn show_grid_view(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        #[derive(Clone)]
        struct GalleryItem {
            path: PathBuf,
            is_video: bool,
            score_label: Option<String>,
            timestamp_sec: f32,
            db_filename: Option<String>,
        }

        ui.vertical(|ui| {
            // Dynamic Database Mapping Check & Auto Lazy Load
            let default_scope = self.default_semantic_folder().to_string_lossy().to_string();
            let effective_scope = self.effective_semantic_folder();
            let scope_has_db = self.folder_has_db_mappings(&effective_scope);
            let has_db = self.current_folder_has_db_mappings();
            if (self.semantic_mode == SearchMode::Clip || self.semantic_mode == SearchMode::Ocr) && !self.db_loaded && !self.db_loading {
                self.start_lazy_db_load(ctx);
            }
            let (clip_paste_shortcut_pressed, clip_pasted_text) = if self.semantic_mode == SearchMode::Clip {
                clipboard_paste_signal(ui)
            } else {
                (false, None)
            };
            if self.semantic_mode == SearchMode::Clip {
                if let Some(text) = clip_pasted_text.as_deref() {
                    if image_path_from_pasted_text(text).is_some() {
                        self.search_clip_from_clipboard_image(ui.ctx(), Some(text), false);
                    } else if clip_paste_shortcut_pressed {
                        self.search_clip_from_clipboard_image(ui.ctx(), Some(text), true);
                    } else {
                        self.search_clip_from_clipboard_image(ui.ctx(), Some(text), false);
                    }
                } else if clip_paste_shortcut_pressed {
                    self.search_clip_from_clipboard_image(ui.ctx(), None, true);
                }
            }

            // Toolbar Controls
            let old_mode = self.semantic_mode;
            ui.horizontal(|ui| {
                ui.label("Mode:");
                ui.selectable_value(&mut self.semantic_mode, SearchMode::Filename, "Filename");
                ui.selectable_value(&mut self.semantic_mode, SearchMode::Clip, "CLIP");
                ui.selectable_value(&mut self.semantic_mode, SearchMode::Ocr, "OCR");
                
                ui.add_space(12.0);
                ui.checkbox(&mut self.semantic_video_only, "Videos only");

                ui.separator();
                let selected_count = self.selected_grid_files.len();
                let repair_label = if self.sift_repair_running {
                    "Repairing SIFT..."
                } else {
                    "Repair selected SIFT"
                };
                if ui
                    .add_enabled(
                        has_db && !self.sift_repair_running && selected_count >= 2,
                        egui::Button::new(repair_label),
                    )
                    .clicked()
                {
                    self.start_selected_sift_repair(ctx);
                }
                if ui
                    .add_enabled(
                        has_db && !self.sift_repair_running && selected_count >= 2,
                        egui::Button::new("Compare selected SIFT"),
                    )
                    .clicked()
                {
                    self.start_selected_sift_compare(ctx);
                }
                if ui
                    .add_enabled(
                        !self.sift_repair_running && selected_count > 0,
                        egui::Button::new("Clear selected"),
                    )
                    .clicked()
                {
                    self.selected_grid_files.clear();
                    self.semantic_status = "Selection cleared.".to_string();
                }
                if selected_count > 0 || self.sift_repair_running {
                    ui.weak(format!("{selected_count} selected (Ctrl-click tiles to toggle)"));
                }
                
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Refresh").clicked() {
                        self.start_recursive_scan();
                    }
                });
            });
            if self.semantic_mode != old_mode {
                self.semantic_results_mode = None;
                self.pending_semantic_search_mode = None;
            }
            ui.add_space(6.0);

            ui.horizontal(|ui| {
                let hint = match self.semantic_mode {
                    SearchMode::Filename => "Filter by filename...",
                    SearchMode::Clip => "Describe the photo or paste an image with Ctrl+V",
                    SearchMode::Ocr => "Type word/text found inside the image",
                };

                ui.label("Search:");
                let search_resp = ui.add(egui::TextEdit::singleline(&mut self.semantic_query)
                    .hint_text(hint)
                    .desired_width(320.0));
                let enter_pressed = text_edit_enter_pressed(&search_resp);
                
                ui.add_space(8.0);
                if self.semantic_mode == SearchMode::Clip && ui.button("Paste Image").clicked() {
                    self.search_clip_from_clipboard_image(ui.ctx(), None, true);
                }
                if self.semantic_mode == SearchMode::Clip {
                    ui.add_space(8.0);
                }
                ui.add(egui::Slider::new(&mut self.semantic_limit, 1..=500).text("Limit"));
                
                ui.add_space(8.0);
                if ui.button("Search").clicked() || enter_pressed {
                    self.submit_semantic_search(ctx);
                }
            });
            if matches!(self.semantic_mode, SearchMode::Clip | SearchMode::Ocr) {
                ui.add_space(6.0);
                let mut folder_enter_pressed = false;
                ui.horizontal(|ui| {
                    ui.label("Folder:");
                    let scope_resp = ui.add(
                        egui::TextEdit::singleline(&mut self.semantic_folder)
                            .hint_text("Blank = all indexed folders, or enter indexed folder path")
                            .desired_width(520.0),
                    );
                    folder_enter_pressed = text_edit_enter_pressed(&scope_resp);
                    if scope_resp.hovered() {
                        scope_resp.on_hover_text(
                            "Use an absolute filesystem path or a database-style path like collection_id/sub/folder. Leave blank to search across all indexed folders."
                        );
                    }
                    if ui.button("Use current").clicked() {
                        self.semantic_folder = default_scope.clone();
                    }
                    if ui.button("Clear").clicked() {
                        self.semantic_folder.clear();
                    }
                });
                if folder_enter_pressed {
                    self.submit_semantic_search(ctx);
                }
                let scope_label = if self.semantic_folder.trim().is_empty() {
                    "Active scope: all indexed folders".to_string()
                } else {
                    format!("Active scope: {effective_scope}")
                };
                ui.weak(scope_label);
                if !scope_has_db {
                    ui.weak("The selected scope is not inside a mapped database collection root.");
                }
            }
            ui.add_space(8.0);

            // Submitted filename searches are cached so path resolution does not run every frame.
            let filename_candidates: Vec<usize> = if self.semantic_mode == SearchMode::Filename {
                self.filename_search_results
                    .clone()
                    .unwrap_or_else(|| (0..self.recursive_images.len()).collect())
            } else {
                (0..self.recursive_images.len()).collect()
            };
            let filtered_images: Vec<usize> = filename_candidates
                .into_iter()
                .filter(|&index| {
                    !self.semantic_video_only || is_video_path(&self.recursive_images[index])
                })
                .collect();

            let is_active_semantic_search = match self.semantic_mode {
                SearchMode::Filename => false,
                SearchMode::Clip | SearchMode::Ocr => {
                    self.semantic_results_mode == Some(self.semantic_mode)
                }
            };

            // Status message label
            ui.horizontal(|ui| {
                let show_sift_status = self.sift_repair_running
                    || self.semantic_status.starts_with("SIFT repair")
                    || self.semantic_status.starts_with("Running SIFT")
                    || self.semantic_status.starts_with("Loading database index before SIFT")
                    || self.semantic_status.starts_with("Select at least")
                    || self.semantic_status.contains("selected")
                    || self.semantic_status.starts_with("Only indexed");
                if show_sift_status {
                    if self.sift_repair_running {
                        ui.add(egui::Spinner::new().size(14.0));
                    }
                    ui.weak(&self.semantic_status);
                } else if self.db_loading
                    && matches!(self.semantic_mode, SearchMode::Clip | SearchMode::Ocr)
                {
                    ui.add(egui::Spinner::new().size(14.0));
                    ui.weak(&self.semantic_status);
                } else if is_active_semantic_search {
                    ui.weak(&self.semantic_status);
                } else if matches!(self.semantic_mode, SearchMode::Clip | SearchMode::Ocr) {
                    ui.weak(&self.semantic_status);
                } else if self.grid_loading {
                    ui.add(egui::Spinner::new().size(14.0));
                    ui.weak("Scanning subdirectories...");
                } else {
                    let label = if self.semantic_video_only { "videos" } else { "items" };
                    ui.weak(format!("{} {} found in this folder", filtered_images.len(), label));
                }
            });
            ui.add_space(6.0);
            ui.separator();
            ui.add_space(8.0);

            if self.db_loading
                && (is_active_semantic_search || self.pending_semantic_search_mode.is_some())
            {
                ui.centered_and_justified(|ui| {
                    ui.vertical_centered(|ui| {
                        ui.add(egui::Spinner::new().size(36.0));
                        ui.add_space(16.0);
                        ui.heading("Lazy-loading AI Database Models & ONNX session...");
                        ui.weak("Initializing standard text encoders and reading index maps. This happens fully in the background.");
                    });
                });
            } else {
                // Populate unified Gallery Items (only if active semantic search is active)
                let mut gallery_items: Vec<GalleryItem> = Vec::new();
                if is_active_semantic_search {
                    let mut seen_semantic_paths: HashSet<PathBuf> = HashSet::new();
                    for item in &self.semantic_results {
                        if let Some(path) = &item.media_path {
                            if !seen_semantic_paths.insert(path.clone()) {
                                continue;
                            }
                            gallery_items.push(GalleryItem {
                                path: path.clone(),
                                is_video: item.is_video,
                                score_label: Some(match self.semantic_mode {
                                    SearchMode::Clip => format!("{:.0}% Match", (item.score * 100.0).clamp(0.0, 100.0)),
                                    SearchMode::Ocr => {
                                        if item.ocr_phrase_query {
                                            format!(
                                                "{} / {} words (exact phrase)",
                                                item.ocr_term_hits,
                                                item.ocr_query_terms
                                            )
                                        } else if item.ocr_query_terms > 0 {
                                            format!(
                                                "{} / {} words",
                                                item.ocr_term_hits,
                                                item.ocr_query_terms
                                            )
                                        } else {
                                            "OCR Match".to_string()
                                        }
                                    }
                                    _ => String::new(),
                                }),
                                timestamp_sec: item.timestamp_sec,
                                db_filename: Some(item.file_name.clone()),
                            });
                        }
                    }
                }

                let num_items = if is_active_semantic_search {
                    gallery_items.len()
                } else {
                    filtered_images.len()
                };

                if num_items == 0 {
                    ui.centered_and_justified(|ui| {
                        if self.grid_loading {
                            ui.weak("Scanning files, please wait...");
                        } else {
                            ui.weak("No files found matching filter or query.");
                        }
                    });
                } else {
                    let available_width = (ui.available_width() - 16.0).max(130.0);
                    let col_width = 130.0 + 12.0;
                    let cols = (available_width / col_width).floor().max(1.0) as usize;
                    let num_rows = (num_items + cols - 1) / cols;
                    let row_height = 160.0 + 12.0;

                    let mut double_clicked_item: Option<GalleryItem> = None;
                    let mut single_clicked_item: Option<GalleryItem> = None;
                    let mut clicked_similar: Option<PendingSearchRequest> = None;
                    let mut clicked_person: Option<PendingSearchRequest> = None;

                    egui::ScrollArea::vertical().id_salt("gallery_scroll_area").show_rows(ui, row_height, num_rows, |ui, row_range| {
                        for row_idx in row_range {
                            let start_idx = row_idx * cols;
                            let end_idx = (start_idx + cols).min(num_items);
                            let row_width = (cols as f32 * 130.0) + (cols.saturating_sub(1) as f32 * 12.0);
                            let (row_rect, _) = ui.allocate_exact_size(egui::vec2(row_width, 160.0), egui::Sense::hover());

                            for (col_idx, item_idx) in (start_idx..end_idx).enumerate() {
                                    let temp_item;
                                    let item = if is_active_semantic_search {
                                        &gallery_items[item_idx]
                                    } else {
                                        let global_idx = filtered_images[item_idx];
                                        let p = &self.recursive_images[global_idx];
                                        let is_vid = is_video_path(p);
                                        let db_name = self.resolve_db_filename(p);
                                        temp_item = GalleryItem {
                                            path: p.clone(),
                                            is_video: is_vid,
                                            score_label: None,
                                            timestamp_sec: 0.0,
                                            db_filename: db_name,
                                        };
                                        &temp_item
                                    };
                                    let path = &item.path;
                                    let is_selected = item
                                        .db_filename
                                        .as_ref()
                                        .is_some_and(|name| self.selected_grid_files.contains(name));
                                    let is_current = if let Some(curr_p) = self.images.get(self.current_index) {
                                        curr_p == path
                                    } else {
                                        false
                                    };
                                    
                                    let rect = egui::Rect::from_min_size(
                                        egui::pos2(row_rect.min.x + col_idx as f32 * col_width, row_rect.min.y),
                                        egui::vec2(130.0, 160.0),
                                    );
                                    let response = ui.interact(
                                        rect,
                                        ui.make_persistent_id(("gallery_card", item_idx)),
                                        egui::Sense::click(),
                                    );
                                    
                                    response.context_menu(|ui| {
                                        if ui.button("📂 Open parent folder").clicked() {
                                            let actual_path = self.resolve_actual_path(path);
                                            open_in_dolphin_or_fallback(&actual_path);
                                            ui.close();
                                        }
                                        if ui.button("📋 Copy image").clicked() {
                                            let resolved_path = self.get_thumbnail_path(path);
                                            if let Err(err) = copy_image_file_to_clipboard(&resolved_path) {
                                                self.semantic_status = format!("Copy image failed: {err}");
                                            }
                                            ui.close();
                                        }
                                        if ui.button("📋 Copy full path").clicked() {
                                            ui.ctx().copy_text(path.to_string_lossy().to_string());
                                            ui.close();
                                        }
                                        if item.is_video {
                                            if ui.button("🎬 Open in mpv").clicked() {
                                                let playback_path = if let Some(db_name) = &item.db_filename {
                                                    let roots = get_db_roots();
                                                    resolve_source_path(&roots, db_name)
                                                        .ok()
                                                        .unwrap_or_else(|| self.resolve_actual_path(path))
                                                } else {
                                                    self.resolve_actual_path(path)
                                                };
                                                let _ = std::process::Command::new("mpv")
                                                    .arg(format!("--start={:.3}", item.timestamp_sec.max(0.0)))
                                                    .arg(playback_path)
                                                    .spawn();
                                                ui.close();
                                            }
                                        } else if ui.button("✏ Edit image").clicked() {
                                            self.start_image_editor(path, ui.ctx());
                                            ui.close();
                                        }
                                        ui.separator();
                                        if ui.button("Show most similar").clicked() {
                                            clicked_similar = Some(PendingSearchRequest::Similar {
                                                db_file_name: item.db_filename.clone(),
                                                media_path: path.clone(),
                                                is_video: item.is_video,
                                                timestamp_sec: item.timestamp_sec,
                                            });
                                            ui.close();
                                        }
                                        if ui.button("Show more of this person").clicked() {
                                            clicked_person = Some(PendingSearchRequest::Person {
                                                db_file_name: item.db_filename.clone(),
                                                media_path: path.clone(),
                                                is_video: item.is_video,
                                            });
                                            ui.close();
                                        }
                                    });

                                    let is_hovered = response.hovered();
                                    let is_clicked = response.clicked();
                                    
                                    let card_bg = if is_selected {
                                        ui.visuals().selection.bg_fill.gamma_multiply(0.45)
                                    } else if is_clicked {
                                        ui.visuals().selection.bg_fill.gamma_multiply(0.3)
                                    } else if is_hovered {
                                        ui.visuals().code_bg_color.gamma_multiply(1.5)
                                    } else if is_current {
                                        ui.visuals().selection.bg_fill.gamma_multiply(0.15)
                                    } else {
                                        ui.visuals().code_bg_color
                                    };
                                    
                                    let card_stroke = if is_selected {
                                        egui::Stroke::new(3.0, egui::Color32::from_rgb(100, 200, 120))
                                    } else if is_current {
                                        egui::Stroke::new(2.0, ui.visuals().selection.bg_fill)
                                    } else if is_hovered {
                                        egui::Stroke::new(1.0, ui.visuals().selection.bg_fill.gamma_multiply(0.5))
                                    } else {
                                        egui::Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color.gamma_multiply(0.3))
                                    };
                                    
                                    let builder = egui::UiBuilder::new()
                                        .max_rect(rect)
                                        .id_salt((path, item_idx));
                                    let mut child_ui = ui.new_child(builder);
                                    egui::Frame::NONE
                                        .fill(card_bg)
                                        .stroke(card_stroke)
                                        .inner_margin(0.0)
                                        .corner_radius(6.0)
                                        .show(&mut child_ui, |ui| {
                                            let resolved_path = self.get_thumbnail_path(path);
                                            if let Some(texture) = self.thumbnail_textures.get(&resolved_path) {
                                                ui.centered_and_justified(|ui| {
                                                    ui.add(
                                                        egui::Image::from_texture(texture)
                                                            .max_size(egui::vec2(130.0, 160.0))
                                                            .maintain_aspect_ratio(false)
                                                    );
                                                });
                                            } else if self.thumbnail_failed.contains(&resolved_path) {
                                                ui.vertical_centered(|ui| {
                                                    ui.add_space(55.0);
                                                    if is_video_path(path) {
                                                        ui.weak("📹 Video");
                                                    } else {
                                                        ui.weak("⚠️ Failed");
                                                    }
                                                });
                                            } else {
                                                ui.vertical_centered(|ui| {
                                                    ui.add_space(60.0);
                                                    ui.add(egui::Spinner::new().size(20.0));
                                                });
                                                
                                                let max_threads = num_cpus::get().max(4);
                                                if !self.thumbnail_loading.contains(&resolved_path) && self.thumbnail_active_threads < max_threads {
                                                    self.thumbnail_loading.insert(resolved_path.clone());
                                                    self.thumbnail_active_threads += 1;
                                                    let path_clone = resolved_path.clone();
                                                    let tx_clone = self.thumbnail_tx.clone();
                                                    let ctx_clone = ui.ctx().clone();
                                                    rayon::spawn(move || {
                                                        if let Ok(img) = image::open(&path_clone) {
                                                            let thumb = img.resize_to_fill(260, 320, image::imageops::FilterType::Triangle);
                                                            let size = [thumb.width() as usize, thumb.height() as usize];
                                                            let pixels = thumb.to_rgba8().into_raw();
                                                            let color_img = egui::ColorImage::from_rgba_unmultiplied(size, &pixels);
                                                            let _ = tx_clone.send((path_clone, color_img));
                                                            ctx_clone.request_repaint();
                                                        } else {
                                                            let empty_img = egui::ColorImage::new([0, 0], Vec::new());
                                                            let _ = tx_clone.send((path_clone, empty_img));
                                                            ctx_clone.request_repaint();
                                                        }
                                                    });
                                                }
                                            }
                                        });

                                    // Overlay 1: Filename banner at the bottom (semi-transparent black with rounded bottom corners)
                                    let banner_rect = egui::Rect::from_min_max(
                                        egui::pos2(rect.min.x, rect.max.y - 24.0),
                                        rect.max
                                    );
                                    let banner_rounding = egui::CornerRadius {
                                        nw: 0,
                                        ne: 0,
                                        sw: 6,
                                        se: 6,
                                    };
                                    ui.painter().rect_filled(banner_rect, banner_rounding, egui::Color32::from_black_alpha(180));

                                    let filename_owned = if let Some(db_name) = &item.db_filename {
                                        db_name
                                            .split_once('/')
                                            .map(|(_, rel)| rel)
                                            .and_then(|rel| Path::new(rel).file_name())
                                            .and_then(|s| s.to_str())
                                            .map(|s| s.to_string())
                                            .unwrap_or_else(|| {
                                                path.file_name()
                                                    .and_then(|s| s.to_str())
                                                    .unwrap_or("")
                                                    .to_string()
                                            })
                                    } else {
                                        path.file_name()
                                            .and_then(|s| s.to_str())
                                            .unwrap_or("")
                                            .to_string()
                                    };
                                    let filename = filename_owned.as_str();
                                    let filename_label = if filename.chars().count() > 22 {
                                        format!("{}...", filename.chars().take(19).collect::<String>())
                                    } else {
                                        filename.to_string()
                                    };
                                    ui.painter().text(
                                        banner_rect.center(),
                                        egui::Align2::CENTER_CENTER,
                                        filename_label,
                                        egui::FontId::proportional(9.0),
                                        egui::Color32::WHITE,
                                    );

                                    // Overlay 2: Score / Match Badge pill overlay in the top-left
                                    if let Some(lbl) = &item.score_label {
                                        let badge_rect = egui::Rect::from_min_max(
                                            egui::pos2(rect.min.x + 6.0, rect.min.y + 6.0),
                                            egui::pos2(rect.min.x + 66.0, rect.min.y + 22.0)
                                        );
                                        let badge_bg = if lbl.contains("Match") && !lbl.contains("OCR") {
                                            egui::Color32::from_rgb(16, 124, 65).gamma_multiply(0.85)
                                        } else {
                                            egui::Color32::from_rgb(0, 90, 158).gamma_multiply(0.85)
                                        };
                                        ui.painter().rect_filled(badge_rect, 4.0, badge_bg);
                                        ui.painter().text(
                                            badge_rect.center(),
                                            egui::Align2::CENTER_CENTER,
                                            lbl,
                                            egui::FontId::proportional(8.0),
                                            egui::Color32::WHITE,
                                        );
                                    }

                                    // Overlay 3: Video Indicator badge with timestamp in the top-right
                                    if item.is_video {
                                        let badge_text = if item.timestamp_sec > 0.0 {
                                            let ts_mins = (item.timestamp_sec / 60.0).floor() as i32;
                                            let ts_secs = (item.timestamp_sec % 60.0).floor() as i32;
                                            format!("📹 {:02}:{:02}", ts_mins, ts_secs)
                                        } else {
                                            "📹 Video".to_string()
                                        };
                                        
                                        let badge_rect = egui::Rect::from_min_max(
                                            egui::pos2(rect.max.x - 66.0, rect.min.y + 6.0),
                                            egui::pos2(rect.max.x - 6.0, rect.min.y + 22.0)
                                        );
                                        ui.painter().rect_filled(badge_rect, 4.0, egui::Color32::from_black_alpha(160));
                                        ui.painter().text(
                                            badge_rect.center(),
                                            egui::Align2::CENTER_CENTER,
                                            badge_text,
                                            egui::FontId::proportional(8.0),
                                            egui::Color32::WHITE,
                                        );
                                    }
                                        
                                    let ctrl_clicked = response.clicked()
                                        && ui.input(|i| i.modifiers.matches_logically(egui::Modifiers::CTRL));
                                    if ctrl_clicked {
                                        if item.is_video {
                                            self.semantic_status = "Selection only supports indexed images for SIFT actions.".to_string();
                                        } else if let Some(db_name) = &item.db_filename {
                                            if !self.db_loaded && !self.db_loading && !self.db_failed {
                                                self.start_lazy_db_load(ui.ctx());
                                            }
                                            if let Some(pos) = self
                                                .selected_grid_files
                                                .iter()
                                                .position(|name| name == db_name)
                                            {
                                                self.selected_grid_files.remove(pos);
                                            } else {
                                                self.selected_grid_files.push(db_name.clone());
                                            }
                                            let selected_count = self.selected_grid_files.len();
                                            self.semantic_status = format!(
                                                "{selected_count} indexed image(s) selected."
                                            );
                                        } else {
                                            if !self.db_loaded && !self.db_loading && !self.db_failed {
                                                self.start_lazy_db_load(ui.ctx());
                                            }
                                            self.semantic_status = "Only indexed database images can be selected.".to_string();
                                        }
                                    } else if response.double_clicked() {
                                        double_clicked_item = Some(item.clone());
                                    } else if response.clicked() {
                                        single_clicked_item = Some(item.clone());
                                    }
                            }
                        }
                    });
                    
                    if let Some(item) = double_clicked_item {
                        let path = item.path.clone();
                        if let Some(db_name) = &item.db_filename {
                            self.db_filename_by_path.insert(path.clone(), db_name.clone());
                        }

                        if item.is_video {
                            let playback_path = if let Some(db_name) = &item.db_filename {
                                let roots = get_db_roots();
                                resolve_source_path(&roots, db_name)
                                    .ok()
                                    .unwrap_or_else(|| self.resolve_actual_path(&path))
                            } else {
                                self.resolve_actual_path(&path)
                            };
                            let _ = std::process::Command::new("mpv")
                                .arg(format!("--start={:.3}", item.timestamp_sec.max(0.0)))
                                .arg(playback_path)
                                .spawn();
                        } else {
                        let active_paths: Vec<PathBuf> = if is_active_semantic_search {
                            for item in &gallery_items {
                                if let Some(db_name) = &item.db_filename {
                                    self.db_filename_by_path.insert(item.path.clone(), db_name.clone());
                                }
                            }
                            gallery_items.iter().map(|item| item.path.clone()).collect()
                        } else {
                            if let Some(db_name) = self.resolve_db_filename(&path) {
                                self.db_filename_by_path.insert(path.clone(), db_name);
                            }
                            filtered_images.iter().map(|&idx| self.recursive_images[idx].clone()).collect()
                        };
                        self.images = active_paths;
                        self.current_index = self.images.iter().position(|p| p == &path).unwrap_or(0);
                        self.show_grid = false;
                        self.back_target_is_gallery = true;
                        self.zoom = 1.0;
                        self.offset = egui::Vec2::ZERO;
                        self.update_current_file_info();
                        self.update_side_panel_metadata_if_needed();
                        ui.ctx().request_repaint();
                        }
                    }
                    
                    if let Some(item) = single_clicked_item {
                        let path = item.path.clone();
                        let active_paths: Vec<PathBuf> = if is_active_semantic_search {
                            for item in &gallery_items {
                                if let Some(db_name) = &item.db_filename {
                                    self.db_filename_by_path.insert(item.path.clone(), db_name.clone());
                                }
                            }
                            gallery_items.iter().map(|item| item.path.clone()).collect()
                        } else {
                            if let Some(db_name) = self.resolve_db_filename(&path) {
                                self.db_filename_by_path.insert(path.clone(), db_name);
                            }
                            filtered_images.iter().map(|&idx| self.recursive_images[idx].clone()).collect()
                        };
                        if let Some(pos) = active_paths.iter().position(|p| p == &path) {
                            self.images = active_paths;
                            self.current_index = pos;
                            self.update_current_file_info();
                            if self.show_exif || self.side_panel_open_pending {
                                self.update_side_panel_metadata_if_needed();
                            } else {
                                self.open_side_panel(ui.ctx(), SidePanelMode::Duplicates);
                            }
                            ui.ctx().request_repaint();
                        }
                    }

                    if let Some(request) = clicked_similar {
                        self.request_search_action(request, ui.ctx());
                        ui.ctx().request_repaint();
                    }

                    if let Some(request) = clicked_person {
                        self.request_search_action(request, ui.ctx());
                        ui.ctx().request_repaint();
                    }
                }
            }
        });
    }

    fn clip_vector_for_result(&self, row: &SearchResult) -> Option<Vec<f32>> {
        let Some(indices) = &self.db_indices else {
            return None;
        };
        let mut best: Option<(&ClipEntry, f32)> = None;
        for entry in &indices.clip_index.entries {
            if entry.file_name.as_ref() != row.file_name.as_str() {
                continue;
            }
            let dt = (entry.timestamp_sec - row.timestamp_sec).abs();
            match best {
                Some((_current, best_dt)) if dt >= best_dt => {}
                _ => best = Some((entry, dt)),
            }
        }
        best.map(|(entry, _)| entry.vector.clone())
    }

    fn show_most_similar_from_vector(
        &mut self,
        query_vector: Vec<f32>,
        source: Option<SearchResult>,
        label: &str,
    ) {
        let Some(indices) = &self.db_indices else {
            return;
        };
        if query_vector.len() != indices.clip_index.dim {
            self.semantic_status = format!(
                "source vector dim {} does not match index dim {}",
                query_vector.len(),
                indices.clip_index.dim
            );
            return;
        }
        let started = Instant::now();
        let pre_limit = (self.semantic_limit.saturating_mul(12)).max(self.semantic_limit + 32);
        let mut results = search_index(&indices.clip_index, &query_vector, pre_limit, false, "");
        if let Some(source) = &source {
            results.retain(|candidate| candidate.file_name != source.file_name);
        }
        results = collapse_sift_grouped_results(results, &indices.sift_root_by_file, self.semantic_limit);
        
        let db_roots = get_db_roots();
        let db_dir_buf = get_db_dir();
        let db_dir = db_dir_buf.as_path();
        for candidate in &mut results {
            candidate.media_path =
                resolve_media_path(&db_roots, db_dir, &candidate.file_name, candidate.timestamp_sec).ok();
            if let Some(path) = &candidate.media_path {
                self.db_filename_by_path.insert(path.clone(), candidate.file_name.clone());
            }
        }
        if let Some(mut source) = source {
            if source.media_path.is_none() {
                source.media_path =
                    resolve_media_path(&db_roots, db_dir, &source.file_name, source.timestamp_sec).ok();
            }
            if let Some(source_path) = &source.media_path {
                self.db_filename_by_path
                    .insert(source_path.clone(), source.file_name.clone());
                results.retain(|candidate| candidate.media_path.as_ref() != Some(source_path));
            }
            source.score = 1.0;
            results.insert(0, source);
            results.truncate(self.semantic_limit);
        }
        for (idx, row) in results.iter_mut().enumerate() {
            row.rank = idx + 1;
        }
        let took = started.elapsed().as_millis();
        self.semantic_status = format!(
            "✓ Found {} CLIP-similar results in {} ms for {}",
            results.len(),
            took,
            label
        );
        self.semantic_results = results;
        self.semantic_results_mode = Some(SearchMode::Clip);
    }

    fn show_most_similar_clip(&mut self, row: &SearchResult) {
        let Some(query_vector) = self.clip_vector_for_result(row) else {
            self.semantic_status = format!("no CLIP vector found for {}", row.file_name);
            return;
        };
        self.show_most_similar_from_vector(query_vector, Some(row.clone()), &row.file_name);
    }

    fn face_vectors_for_file(indices: &DatabaseIndices, file_name: &str) -> Vec<Vec<f32>> {
        indices.face_index
            .entries
            .iter()
            .filter(|entry| entry.file_name.as_ref() == file_name)
            .map(|entry| entry.vector.clone())
            .collect()
    }

    fn related_files_for_face_seed(indices: &DatabaseIndices, file_name: &str) -> Vec<String> {
        let mut related = Vec::new();
        let mut seen = HashSet::new();
        if seen.insert(file_name.to_string()) {
            related.push(file_name.to_string());
        }

        let root = if let Some(canonical) = indices.sift_root_by_file.get(file_name) {
            canonical.clone()
        } else {
            file_name.to_string()
        };
        
        if let Some(members) = indices.sift_members_by_root.get(root.as_str()) {
            for member in members {
                if seen.insert(member.clone()) {
                    related.push(member.clone());
                }
                if let Some(children) = indices.similar_by_master.get(member.as_str()) {
                    for child in children {
                        if !child.is_video && seen.insert(child.file_name.clone()) {
                            related.push(child.file_name.clone());
                        }
                    }
                }
            }
        } else if let Some(children) = indices.similar_by_master.get(file_name) {
            for child in children {
                if !child.is_video && seen.insert(child.file_name.clone()) {
                    related.push(child.file_name.clone());
                }
            }
        }

        related
    }

    fn query_face_vectors_for_seed(&self, file_name: &str) -> Vec<Vec<f32>> {
        let Some(indices) = &self.db_indices else {
            return Vec::new();
        };
        let related_files = Self::related_files_for_face_seed(indices, file_name);
        let mut query_faces = Vec::new();
        for related in &related_files {
            query_faces.extend(Self::face_vectors_for_file(indices, related));
        }
        query_faces
    }

    fn show_more_of_this_person_with_vectors(&mut self, query_faces: Vec<Vec<f32>>, label: &str) {
        let Some(indices) = &self.db_indices else {
            return;
        };
        if query_faces.is_empty() {
            self.semantic_status = format!("No face embeddings available for {label}");
            self.semantic_results = Vec::new();
            self.semantic_results_mode = Some(SearchMode::Clip);
            return;
        }
        let started = Instant::now();
        let mut results =
            search_face_index(&indices.face_index, &query_faces, 500, FACE_MATCH_MIN_SCORE);
        results = collapse_sift_grouped_results(results, &indices.sift_root_by_file, 500);

        let db_roots = get_db_roots();
        let db_dir_buf = get_db_dir();
        let db_dir = db_dir_buf.as_path();
        for row in &mut results {
            row.media_path =
                resolve_media_path(&db_roots, db_dir, &row.file_name, row.timestamp_sec).ok();
            if let Some(path) = &row.media_path {
                self.db_filename_by_path.insert(path.clone(), row.file_name.clone());
            }
        }
        let took = started.elapsed().as_millis();
        self.semantic_status = format!(
            "✓ Found {} person results in {} ms using {} query face vector(s) for {}",
            results.len(),
            took,
            query_faces.len(),
            label
        );
        self.semantic_results = results;
        self.semantic_results_mode = Some(SearchMode::Clip);
    }

    fn label_for_request(request: &PendingSearchRequest) -> String {
        match request {
            PendingSearchRequest::Similar {
                db_file_name,
                media_path,
                ..
            }
            | PendingSearchRequest::Person {
                db_file_name,
                media_path,
                ..
            } => db_file_name
                .clone()
                .unwrap_or_else(|| {
                    if is_clipboard_image_path(media_path) {
                        "clipboard image".to_string()
                    } else {
                        media_path.to_string_lossy().to_string()
                    }
                }),
        }
    }

    fn inspect_path_for_request(&self, request: &PendingSearchRequest) -> PathBuf {
        match request {
            PendingSearchRequest::Similar {
                media_path,
                is_video,
                ..
            }
            | PendingSearchRequest::Person {
                media_path,
                is_video,
                ..
            } => {
                let resolved = self.resolve_actual_path(media_path);
                if *is_video {
                    self.get_thumbnail_path(&resolved)
                } else {
                    resolved
                }
            }
        }
    }

    fn start_on_demand_embedding_request(
        &mut self,
        request: PendingSearchRequest,
        need_clip: bool,
        need_faces: bool,
        ctx: &egui::Context,
    ) {
        if self.on_demand_embed_rx.is_some() {
            return;
        }
        let image_path = self.inspect_path_for_request(&request);
        let (tx, rx) = std::sync::mpsc::channel::<Result<OnDemandEmbedResult, String>>();
        self.on_demand_embed_rx = Some(rx);
        let request_clone = request.clone();
        let ctx_clone = ctx.clone();
        std::thread::spawn(move || {
            let result = compute_on_demand_embeddings(&image_path, need_clip, need_faces)
                .map(|(clip_vector, face_vectors)| OnDemandEmbedResult {
                    request: request_clone,
                    clip_vector,
                    face_vectors,
                })
                .map_err(|e| e.to_string());
            let _ = tx.send(result);
            ctx_clone.request_repaint();
        });
    }

    fn run_search_request_now(&mut self, request: PendingSearchRequest, ctx: &egui::Context) {
        match &request {
            PendingSearchRequest::Similar {
                db_file_name,
                media_path,
                is_video,
                timestamp_sec,
            } => {
                if let Some(db_name) = db_file_name {
                    let row = SearchResult {
                        rank: 0,
                        score: 1.0,
                        file_name: db_name.clone(),
                        is_video: *is_video,
                        timestamp_sec: *timestamp_sec,
                        media_path: Some(media_path.clone()),
                        ocr_term_hits: 0,
                        ocr_query_terms: 0,
                        ocr_phrase_query: false,
                    };
                    if self.clip_vector_for_result(&row).is_some() {
                        self.show_most_similar_clip(&row);
                        return;
                    }
                }
                self.semantic_status = format!(
                    "Computing CLIP embedding on the fly for {}...",
                    Self::label_for_request(&request)
                );
                self.start_on_demand_embedding_request(request, true, false, ctx);
            }
            PendingSearchRequest::Person { db_file_name, .. } => {
                if let Some(db_name) = db_file_name {
                    let query_faces = self.query_face_vectors_for_seed(db_name);
                    if !query_faces.is_empty() {
                        self.show_more_of_this_person_with_vectors(query_faces, db_name);
                        return;
                    }
                }
                self.semantic_status = format!(
                    "Computing face embeddings on the fly for {}...",
                    Self::label_for_request(&request)
                );
                self.start_on_demand_embedding_request(request, false, true, ctx);
            }
        }
    }

    fn request_search_action(&mut self, request: PendingSearchRequest, ctx: &egui::Context) {
        let label = Self::label_for_request(&request);
        self.semantic_mode = SearchMode::Clip;
        self.semantic_query = label.clone();
        self.semantic_results.clear();
        self.semantic_results_mode = None;
        self.pending_search_request = Some(request.clone());

        let needs_supplemental = matches!(&request, PendingSearchRequest::Person { .. });
        if needs_supplemental
            && self.db_loaded
            && !self.db_supplemental_loaded
            && !self.db_supplemental_loading
        {
            self.pending_search_request = None;
            self.semantic_status =
                "Person search is unavailable because supplemental database loading failed."
                    .to_string();
            return;
        }
        if !self.db_loaded || (needs_supplemental && !self.db_supplemental_loaded) {
            self.semantic_status =
                format!("Loading AI DB to search for matches related to {label}...");
            if !self.db_failed && !self.db_loading {
                self.start_lazy_db_load(ctx);
            }
            return;
        }

        self.pending_search_request = None;
        self.run_search_request_now(request, ctx);
    }

}

impl eframe::App for ImageViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Ok(mut lock) = self.ctx_shared.lock() {
            if lock.is_none() {
                *lock = Some(ctx.clone());
            }
        }

        self.poll_db_load();
        self.poll_sift_alignment();
        self.poll_sift_repair(ctx);
        self.poll_on_demand_embeddings(ctx);

        if !self.db_loaded && !self.db_loading {
            let is_ai = if let Some(p) = self.images.get(self.current_index) {
                is_path_ai_backed(p)
            } else if is_path_ai_backed(&self.open_target) {
                true
            } else {
                false
            };
            if is_ai {
                self.start_lazy_db_load(ctx);
            }
        }

        while let Ok((path, color_image)) = self.thumbnail_rx.try_recv() {
            if color_image.size[0] == 0 {
                self.thumbnail_failed.insert(path.clone());
            } else {
                let texture = ctx.load_texture(
                    path.to_string_lossy(),
                    color_image,
                    egui::TextureOptions::default(),
                );
                self.thumbnail_textures.insert(path.clone(), texture);
            }
            self.thumbnail_loading.remove(&path);
            self.thumbnail_active_threads = self.thumbnail_active_threads.saturating_sub(1);
            ctx.request_repaint();
        }

        if let Ok(new_path) = self.rx.try_recv() {
            self.open_image_path(new_path);
            self.show_home_page = false;
            ctx.request_repaint();
        }

        if let Some(rx) = &self.recursive_rx {
            let mut new_images = Vec::new();
            let mut disconnected = false;
            loop {
                match rx.try_recv() {
                    Ok(path) => new_images.push(path),
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
            if !new_images.is_empty() {
                self.recursive_images.extend(new_images);
                ctx.request_repaint();
            }
            if disconnected {
                self.recursive_images.sort();
                self.grid_loading = false;
                self.recursive_rx = None;
                ctx.request_repaint();
            }
        }

        if self.flat_loading {
            let mut collected = None;
            if let Ok(mut lock) = self.flat_images_shared.try_lock() {
                if let Some(imgs) = lock.take() {
                    collected = Some(imgs);
                }
            }
            if let Some(imgs) = collected {
                self.images = imgs;
                self.current_index = self.images.iter().position(|p| p == &self.open_target).unwrap_or(0);
                self.flat_loading = false;
                self.update_current_file_info();
                self.update_side_panel_metadata_if_needed();
                ctx.request_repaint();
            }
        }
        // Mouse Back click handling:
        // allow returning to gallery if explicitly marked, or if a gallery list is available.
        let can_back_to_gallery =
            self.back_target_is_gallery || (!self.show_home_page && !self.recursive_images.is_empty());
        if !self.show_grid && can_back_to_gallery {
            let back_clicked = ctx.input(|i| i.pointer.button_clicked(egui::PointerButton::Extra1));
            if back_clicked {
                self.show_grid = true;
                self.back_target_is_gallery = true;
                ctx.request_repaint();
            }
        }

        // Keyboard handling
        if !ctx.wants_keyboard_input() {
            ctx.input(|i| {
                if !self.show_home_page {
                    if !self.show_grid && self.image_editor.is_none() {
                        if i.key_pressed(egui::Key::ArrowRight) {
                            self.next_image();
                        }
                        if i.key_pressed(egui::Key::ArrowLeft) {
                            self.prev_image();
                        }
                        if i.key_pressed(egui::Key::F) {
                            self.zoom = 1.0;
                            self.offset = egui::Vec2::ZERO;
                        }
                        if i.key_pressed(egui::Key::Num0) {
                            self.zoom = 1.0;
                            self.offset = egui::Vec2::ZERO;
                        }
                    }
                    if i.key_pressed(egui::Key::G) {
                        self.show_grid = !self.show_grid;
                        if self.show_grid && self.recursive_images.is_empty() {
                            self.start_recursive_scan();
                        }
                    }
                    if i.key_pressed(egui::Key::E) {
                        self.toggle_layout_side_panel(ctx);
                    }
                    if i.key_pressed(egui::Key::Backspace) {
                        self.show_home_page = true;
                    }
                }
                if i.key_pressed(egui::Key::Q) {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
                if i.key_pressed(egui::Key::Escape) {
                    if self.image_editor.is_some() {
                        self.image_editor = None;
                    } else if self.show_home_page {
                        if self.home_current_dir.is_some() {
                            self.home_current_dir = None;
                        } else {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    } else if self.show_grid {
                        self.show_home_page = true;
                    } else {
                        self.show_grid = true;
                    }
                }
            });
        }

        if self.show_home_page {
            self.show_home_page_view(ctx);
            return;
        }

        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Iris");
                ui.separator();
                
                if ui.button("🏠 Filesystem").clicked() {
                    self.show_home_page = true;
                }
                
                ui.separator();
                if let Some(path) = self.images.get(self.current_index) {
                    let filename = self
                        .resolve_db_filename(path)
                        .and_then(|db_name| {
                            db_name
                                .split_once('/')
                                .map(|(_, rel)| rel.to_string())
                        })
                        .and_then(|rel| {
                            Path::new(&rel)
                                .file_name()
                                .and_then(|f| f.to_str())
                                .map(|s| s.to_string())
                        })
                        .unwrap_or_else(|| {
                            path.file_name()
                                .and_then(|f| f.to_str())
                                .unwrap_or("")
                                .to_string()
                        });
                    ui.label(format!("{} ({}/{}) - {} - {}", filename, self.current_index + 1, self.images.len(), self.current_dimensions, self.current_file_size));
                } else {
                    ui.label("No image loaded");
                }
                
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Layout Button
                    let show_layout_active = (self.show_exif || self.side_panel_open_pending)
                        && self.side_panel_mode == SidePanelMode::Layout;
                    let layout_button_text = if show_layout_active { "📂 Hide Layout [E]" } else { "📂 Show Layout [E]" };
                    if ui.button(layout_button_text).clicked() {
                        self.toggle_layout_side_panel(ctx);
                    }
                    
                    ui.add_space(8.0);
                    
                    let gallery_text = if self.show_grid { "🖼 Hide Gallery [G]" } else { "🖼 Show Gallery [G]" };
                    if ui.button(gallery_text).clicked() {
                        self.show_grid = !self.show_grid;
                        if self.show_grid && self.recursive_images.is_empty() {
                            self.start_recursive_scan();
                        }
                    }
                });
            });
        });

        self.apply_pending_side_panel_open(ctx);

        // Side panel opens only after the native window has expanded, avoiding gallery reflow flicker.
        if self.show_exif {
            egui::SidePanel::right("exif_panel")
                .resizable(false)
                .exact_width(Self::SIDE_PANEL_WIDTH)
                .show(ctx, |ui| {
                // Header Tabs
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.side_panel_mode, SidePanelMode::Layout, "📂 Binary Layout");
                    ui.selectable_value(&mut self.side_panel_mode, SidePanelMode::Exif, "🏷 Raw EXIF");
                    ui.selectable_value(&mut self.side_panel_mode, SidePanelMode::Duplicates, "👥 Duplicates");
                    
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("❌").clicked() {
                            self.close_side_panel(ui.ctx());
                        }
                    });
                });
                ui.separator();
                ui.add_space(4.0);

                match self.side_panel_mode {
                    SidePanelMode::Layout => {
                        self.update_layout_if_needed();
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            if self.chunks.is_empty() {
                                ui.label("No layout data available.");
                            } else {
                                for chunk in &self.chunks {
                                    let size_str = if chunk.length >= 1048576 {
                                        format!("{:.2} MB", chunk.length as f64 / 1048576.0)
                                    } else if chunk.length >= 1024 {
                                        format!("{:.1} KB", chunk.length as f64 / 1024.0)
                                    } else {
                                        format!("{} B", chunk.length)
                                    };
                                    
                                    egui::Frame::NONE
                                        .fill(ui.visuals().code_bg_color)
                                        .stroke(egui::Stroke::new(1.0, chunk.color.gamma_multiply(0.3)))
                                        .inner_margin(8.0)
                                        .corner_radius(6.0)
                                        .show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                let (rect, _response) = ui.allocate_at_least(egui::vec2(6.0, 32.0), egui::Sense::hover());
                                                ui.painter().rect_filled(rect, 3.0, chunk.color);
                                                
                                                ui.vertical(|ui| {
                                                    let is_system = chunk.name == "System Metadata";
                                                    let default_open = is_system;
                                                    let id = ui.make_persistent_id(chunk.offset + if default_open { 99999 } else { 0 });
                                                    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, default_open)
                                                        .show_header(ui, |ui| {
                                                            ui.horizontal(|ui| {
                                                                ui.colored_label(ui.visuals().strong_text_color(), &chunk.name);
                                                                if !is_system {
                                                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                                                        ui.weak(&size_str);
                                                                    });
                                                                }
                                                            });
                                                        })
                                                        .body(|ui| {
                                                            if !is_system {
                                                                ui.add_space(4.0);
                                                                ui.horizontal(|ui| {
                                                                    ui.weak(format!("Offset: 0x{:08X}", chunk.offset));
                                                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                                                        ui.weak(format!("Len: {}", chunk.length));
                                                                    });
                                                                });
                                                            }
                                                            ui.add_space(4.0);
                                                             egui::ScrollArea::horizontal().show(ui, |ui| {
                                                                 ui.add(egui::Label::new(egui::RichText::new(&chunk.parsed_data).monospace()).selectable(true));
                                                             });
                                                        });
                                                });
                                            });
                                        }).response.on_hover_text(&chunk.description);
                                        
                                    ui.add_space(6.0);
                                }
                            }
                        });
                    }
                    SidePanelMode::Exif => {
                        self.update_side_panel_metadata_if_needed();
                        ui.horizontal(|ui| {
                            ui.label("Search:");
                            ui.add(egui::TextEdit::singleline(&mut self.exif_search)
                                .hint_text("🔍 Search EXIF tags...")
                                .desired_width(180.0));
                            if ui.button("❌").clicked() {
                                self.exif_search.clear();
                            }
                        });
                        ui.add_space(6.0);
                        ui.separator();
                        ui.add_space(6.0);

                        egui::ScrollArea::vertical().show(ui, |ui| {
                            if self.exif_data.is_empty() {
                                ui.label("No EXIF data available.");
                            } else {
                                let filter = self.exif_search.to_lowercase();
                                let filtered_lines: Vec<String> = self.exif_data
                                    .lines()
                                    .filter(|line| {
                                        if filter.is_empty() {
                                            true
                                        } else {
                                            line.to_lowercase().contains(&filter)
                                        }
                                    })
                                    .map(|s| s.to_string())
                                    .collect();

                                if filtered_lines.is_empty() {
                                    ui.weak("No matching tags found.");
                                } else {
                                    let content = filtered_lines.join("\n");
                                    egui::ScrollArea::horizontal().show(ui, |ui| {
                                        ui.add(
                                            egui::Label::new(
                                                egui::RichText::new(content)
                                                    .monospace()
                                                    .size(11.0)
                                            )
                                            .selectable(true)
                                        );
                                    });
                                }
                            }
                        });
                    }
                    SidePanelMode::Duplicates => {
                        if !self.db_loaded || !self.db_supplemental_loaded {
                            if !self.db_failed && !self.db_loading {
                                self.start_lazy_db_load(ui.ctx());
                            }
                            ui.vertical_centered(|ui| {
                                ui.add_space(20.0);
                                if self.db_failed || (self.db_loaded && !self.db_supplemental_loading) {
                                    ui.colored_label(egui::Color32::from_rgb(255, 100, 100), &self.semantic_status);
                                } else {
                                    ui.add(egui::Spinner::new().size(24.0));
                                    ui.add_space(12.0);
                                    ui.weak("Loading duplicate and SIFT indexes...");
                                }
                            });
                        } else if let Some(path) = self.images.get(self.current_index).cloned() {
                            let filename_opt = self.resolve_db_filename(&path);
                            if let Some(filename) = filename_opt {
                                let indices = self.db_indices.as_ref().unwrap();
                                
                                // Check if the current file is a video in the DB
                                let current_is_video = is_video_path(Path::new(&filename));
                                
                                // Resolve grouped master: prefer the current image's SIFT
                                // component, then fall back through pHash/VideoHash.
                                let master_file_name = if current_is_video {
                                    indices.phash_master_by_file
                                        .get(&filename)
                                        .cloned()
                                        .unwrap_or_else(|| filename.clone())
                                } else {
                                    indices.sift_root_by_file
                                        .get(&filename)
                                        .cloned()
                                        .unwrap_or_else(|| {
                                            let phash_master = indices.phash_master_by_file
                                                .get(&filename)
                                                .cloned()
                                                .unwrap_or_else(|| filename.clone());
                                            indices.sift_root_by_file
                                                .get(&phash_master)
                                                .cloned()
                                                .unwrap_or(phash_master)
                                        })
                                };
                                
                                // Fetch SIFT members in this group
                                let sift_members = indices.sift_members_by_root
                                    .get(&master_file_name)
                                    .cloned()
                                    .unwrap_or_default();
                                
                                let mut displayed_sift_members = Vec::new();
                                let mut displayed_seen = HashSet::new();
                                if displayed_seen.insert(filename.clone()) {
                                    displayed_sift_members.push(filename.clone());
                                }
                                for member in &sift_members {
                                    if displayed_seen.insert(member.clone()) {
                                        displayed_sift_members.push(member.clone());
                                    }
                                }
                                
                                let phash_group_seeds: Vec<String> = if sift_members.is_empty() {
                                    vec![master_file_name.clone()]
                                } else {
                                    let mut seeds = Vec::new();
                                    let mut seen = HashSet::new();
                                    if seen.insert(master_file_name.clone()) {
                                        seeds.push(master_file_name.clone());
                                    }
                                    for member in &sift_members {
                                        if seen.insert(member.clone()) {
                                            seeds.push(member.clone());
                                        }
                                    }
                                    seeds
                                };
                                let clip_embedded_files = Arc::clone(&indices.clip_embedded_files);
                                let ocr_embedded_files = Arc::clone(&indices.ocr_embedded_files);
                                let skipped_processing_files = Arc::clone(&indices.skipped_processing_files);
                                let use_sift_seed_similarity = sift_members.len() > 1;

                                let mut phash_similar_groups: Vec<(String, String, Vec<SimilarFile>)> = Vec::new();
                                for (seed_index, seed) in phash_group_seeds.iter().enumerate() {
                                    if let Some(items) = indices.similar_by_master.get(seed.as_str()) {
                                        let mut group_items = items.clone();
                                        let similarity_reference = if use_sift_seed_similarity {
                                            seed
                                        } else {
                                            &filename
                                        };
                                        for item in &mut group_items {
                                            item.similarity_pct = similarity_to_active(
                                                similarity_reference,
                                                &item.file_name,
                                                &indices.phash_by_file,
                                                &indices.video_frame_phashes_by_file,
                                            );
                                        }
                                        if !group_items.iter().any(|item| item.file_name == *seed) {
                                            group_items.push(SimilarFile {
                                                file_name: seed.clone(),
                                                is_video: is_video_path(Path::new(seed)),
                                                similarity_pct: similarity_to_active(
                                                    similarity_reference,
                                                    seed,
                                                    &indices.phash_by_file,
                                                    &indices.video_frame_phashes_by_file,
                                                ),
                                            });
                                        }
                                        group_items.retain(|item| item.file_name != filename);
                                        group_items.sort_by(|a, b| {
                                            b.similarity_pct
                                                .unwrap_or(f32::NEG_INFINITY)
                                                .partial_cmp(&a.similarity_pct.unwrap_or(f32::NEG_INFINITY))
                                                .unwrap_or(Ordering::Equal)
                                                .then_with(|| a.file_name.cmp(&b.file_name))
                                        });
                                        if !group_items.is_empty() {
                                            let section_label = if use_sift_seed_similarity {
                                                format!("SIFT master {} pHash/VideoHash similars", seed_index + 1)
                                            } else {
                                                "Active image pHash/VideoHash similars".to_string()
                                            };
                                            phash_similar_groups.push((seed.clone(), section_label, group_items));
                                        }
                                    }
                                }
                                let mut phash_unique_files = HashSet::new();
                                for (seed, _, items) in &phash_similar_groups {
                                    if seed != &filename {
                                        phash_unique_files.insert(seed.clone());
                                    }
                                    for item in items {
                                        phash_unique_files.insert(item.file_name.clone());
                                    }
                                }
                                let phash_unique_count = phash_unique_files.len();
                                
                                // Precompute SIFT members metadata
                                // source_path = original file (video/image), preview_path = video still or image
                                let mut displayed_sift_metadata = Vec::new();
                                let roots = get_db_roots();
                                for member in &displayed_sift_members {
                                    let source_path_opt = resolve_source_path(&roots, member).ok();
                                    let preview_path_opt = source_path_opt.as_ref().map(|p| self.get_thumbnail_path(p));
                                    let member_is_video = is_video_path(Path::new(member))
                                        || source_path_opt.as_ref().is_some_and(|p| is_video_path(p));
                                    let res_size_str = source_path_opt.as_ref()
                                        .map(|p| self.get_file_resolution_and_size(p))
                                        .unwrap_or_else(|| "n/a".to_string());
                                    let sift_str = if sift_members.len() <= 1 && member == &filename {
                                        "SIFT: no grouped match".to_string()
                                    } else if member == &master_file_name {
                                        "SIFT: group root".to_string()
                                    } else {
                                        sift_info_line(&indices.sift_info_by_file, member)
                                    };
                                    let has_clip = clip_embedded_files.contains(member);
                                    let has_ocr = ocr_embedded_files.contains(member);
                                    let skipped = skipped_processing_files.contains(member);
                                    displayed_sift_metadata.push((member.clone(), source_path_opt, preview_path_opt, member_is_video, res_size_str, sift_str, has_clip, has_ocr, skipped));
                                }
                                let mut database_details: HashMap<(String, String), Vec<String>> = HashMap::new();
                                for (member, _, _, member_is_video, _, _, _, _, _) in &displayed_sift_metadata {
                                    if !self.expanded_duplicate_rows.contains(member) {
                                        continue;
                                    }
                                    database_details.insert(
                                        (master_file_name.clone(), member.clone()),
                                        duplicate_database_detail_lines(
                                            member,
                                            &master_file_name,
                                            *member_is_video,
                                            &indices.phash_by_file,
                                            &indices.video_frame_phashes_by_file,
                                        ),
                                    );
                                }
                                for (group_seed, _, group_items) in &phash_similar_groups {
                                    for item in group_items {
                                        if !self.expanded_duplicate_rows.contains(&item.file_name) {
                                            continue;
                                        }
                                        database_details.insert(
                                            (group_seed.clone(), item.file_name.clone()),
                                            duplicate_database_detail_lines(
                                                &item.file_name,
                                                group_seed,
                                                item.is_video,
                                                &indices.phash_by_file,
                                                &indices.video_frame_phashes_by_file,
                                            ),
                                        );
                                    }
                                }

                                ui.heading("👥 Duplicate Matches");
                                ui.add_space(4.0);
                                ui.weak(format!("Current: {}", filename));
                                ui.add_space(2.0);
                                ui.weak(format!(
                                    "pHash/VideoHash similar count: {} across {} group(s)",
                                    phash_unique_count,
                                    phash_similar_groups.len()
                                ));
                                ui.add_space(8.0);
                                
                                egui::ScrollArea::vertical().show(ui, |ui| {
                                    let side_thumb = 90.0_f32;
                                    
                                    // 1. SIFT Cluster Members (Duplicates)
                                    if !displayed_sift_metadata.is_empty() {
                                        ui.horizontal(|ui| {
                                            ui.colored_label(egui::Color32::from_rgb(100, 200, 100), "✓ SIFT Group");
                                            ui.weak(format!("({} files)", displayed_sift_metadata.len()));
                                        });
                                        ui.add_space(6.0);
                                        
                                        for (member, source_path_opt, preview_path_opt, member_is_video, res_size_str, sift_str, has_clip, has_ocr, skipped) in &displayed_sift_metadata {
                                            let detail_lines = database_details
                                                .get(&(master_file_name.clone(), member.clone()))
                                                .cloned()
                                                .unwrap_or_default();
                                            let expanded = self.expanded_duplicate_rows.contains(member);
                                            ui.horizontal(|ui| {
                                                // Left: Thumbnail preview (use preview_path for video stills)
                                                let thumb_path = preview_path_opt.as_ref().or(source_path_opt.as_ref());
                                                if let Some(t_path) = thumb_path {
                                                    self.draw_thumbnail_async(ui, t_path, side_thumb);
                                                } else {
                                                    let (rect, _) = ui.allocate_exact_size(
                                                        egui::vec2(side_thumb, side_thumb),
                                                        egui::Sense::hover(),
                                                    );
                                                    ui.painter().rect_filled(rect, 4.0, egui::Color32::from_gray(30));
                                                }
                                                
                                                // Right: Info and buttons
                                                ui.vertical(|ui| {
                                                    ui.horizontal(|ui| {
                                                        ui.colored_label(
                                                            if *member_is_video { egui::Color32::LIGHT_BLUE } else { egui::Color32::LIGHT_GREEN },
                                                            if *member_is_video { "📹 Video" } else { "🖼 Image" }
                                                        );
                                                        draw_embedding_markers(ui, *has_clip, *has_ocr, *skipped);
                                                        if member == &filename {
                                                            ui.colored_label(egui::Color32::from_rgb(255, 180, 50), "• Active");
                                                        }
                                                    });
                                                    
                                                    ui.weak(res_size_str);
                                                    ui.weak(sift_str);
                                                    
                                                    let display_name = member.split_once('/').map(|x| x.1).unwrap_or(member);
                                                    wrapping_monospace_path(ui, display_name);
                                                    
                                                    ui.horizontal_wrapped(|ui| {
                                                        if let Some(s_path) = source_path_opt.as_ref() {
                                                            if ui.button("📂 Open Folder").clicked() {
                                                                open_in_dolphin_or_fallback(s_path);
                                                            }
                                                            if *member_is_video {
                                                                if ui.button("▶ Open in mpv").clicked() {
                                                                    let _ = std::process::Command::new("mpv")
                                                                        .arg(s_path)
                                                                        .spawn();
                                                                }
                                                            } else {
                                                                if ui.button("👁 View").clicked() {
                                                                    if let Some(pos) = self.images.iter().position(|p| p == s_path) {
                                                                        self.current_index = pos;
                                                                    } else {
                                                                        self.images.insert(self.current_index + 1, s_path.clone());
                                                                        self.db_filename_by_path.insert(s_path.clone(), member.clone());
                                                                        self.current_index += 1;
                                                                    }
                                                                    self.show_grid = false;
                                                                    self.back_target_is_gallery = true;
                                                                    self.update_current_file_info();
                                                                    self.update_side_panel_metadata_if_needed();
                                                                }
                                                            }
                                                        }
                                                        if ui.button(if expanded { "Collapse" } else { "Expand" }).clicked() {
                                                            if expanded {
                                                                self.expanded_duplicate_rows.remove(member);
                                                            } else {
                                                                self.expanded_duplicate_rows.insert(member.clone());
                                                            }
                                                        }
                                                    });
                                                    if expanded {
                                                        for line in &detail_lines {
                                                            ui.monospace(line);
                                                        }
                                                    }
                                                });
                                            });
                                            ui.add_space(8.0);
                                            ui.separator();
                                            ui.add_space(8.0);
                                        }
                                        ui.add_space(8.0);
                                    }
                                    
                                    // 2. pHash/VideoHash similars grouped by SIFT master/member seed
                                    if !phash_similar_groups.is_empty() {
                                        ui.horizontal(|ui| {
                                            ui.colored_label(egui::Color32::from_rgb(100, 180, 255), "🔗 Similar Files (pHash/VideoHash)");
                                            ui.weak(format!("({} unique files)", phash_unique_count));
                                        });
                                        ui.add_space(6.0);
                                        
                                        for (group_seed, section_label, group_items) in &phash_similar_groups {
                                            egui::Frame::NONE
                                                .fill(ui.visuals().extreme_bg_color.gamma_multiply(0.35))
                                                .stroke(egui::Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color.gamma_multiply(0.35)))
                                                .inner_margin(8.0)
                                                .corner_radius(6.0)
                                                .show(ui, |ui| {
                                                    ui.horizontal(|ui| {
                                                        ui.colored_label(egui::Color32::from_rgb(120, 190, 255), section_label);
                                                    });
                                                    let linked_count = group_items
                                                        .iter()
                                                        .filter(|item| item.file_name != *group_seed)
                                                        .count();
                                                    ui.weak(format!("{linked_count} similar file(s) linked to this reference"));
                                                    ui.add_space(6.0);

                                                    let row_height = side_thumb + 24.0;
                                                    for item in group_items.iter().cloned() {
                                                        let source_path_opt = resolve_source_path(&roots, &item.file_name).ok();
                                                        let preview_path_opt = source_path_opt.as_ref().map(|p| self.get_thumbnail_path(p));
                                                        let item_is_video = item.is_video
                                                            || is_video_path(Path::new(&item.file_name))
                                                            || source_path_opt.as_ref().is_some_and(|p| is_video_path(p));
                                                        let res_size_str = source_path_opt.as_ref()
                                                            .map(|p| self.get_file_resolution_and_size(p))
                                                            .unwrap_or_else(|| "n/a".to_string());
                                                        let item_has_clip = clip_embedded_files.contains(&item.file_name);
                                                        let item_has_ocr = ocr_embedded_files.contains(&item.file_name);
                                                        let item_skipped = skipped_processing_files.contains(&item.file_name);
                                                        let detail_lines = database_details
                                                            .get(&(group_seed.clone(), item.file_name.clone()))
                                                            .cloned()
                                                            .unwrap_or_default();
                                                        let expanded = self.expanded_duplicate_rows.contains(&item.file_name);

                                                        ui.allocate_ui(egui::vec2(ui.available_width(), row_height), |ui| {
                                                            ui.horizontal(|ui| {
                                                                // Left: Thumbnail preview (use preview_path for video stills)
                                                                let thumb_path = preview_path_opt.as_ref().or(source_path_opt.as_ref());
                                                                if let Some(t_path) = thumb_path {
                                                                    self.draw_thumbnail_async(ui, t_path, side_thumb);
                                                                } else {
                                                                    let (rect, _) = ui.allocate_exact_size(
                                                                        egui::vec2(side_thumb, side_thumb),
                                                                        egui::Sense::hover(),
                                                                    );
                                                                    ui.painter().rect_filled(rect, 4.0, egui::Color32::from_gray(30));
                                                                }
                                                                
                                                                // Right: Info and buttons
                                                                ui.vertical(|ui| {
                                                                    ui.horizontal(|ui| {
                                                                        ui.colored_label(
                                                                            if item_is_video { egui::Color32::LIGHT_BLUE } else { egui::Color32::LIGHT_GREEN },
                                                                            if item_is_video { "📹 Video" } else { "🖼 Image" }
                                                                        );
                                                                        draw_embedding_markers(ui, item_has_clip, item_has_ocr, item_skipped);
                                                                        if item.file_name == filename {
                                                                            ui.colored_label(egui::Color32::from_rgb(255, 180, 50), "• Active");
                                                                        }
                                                                        if item.file_name == *group_seed {
                                                                            ui.colored_label(egui::Color32::from_rgb(120, 190, 255), "• Seed");
                                                                        }
                                                                    });
                                                                    
                                                                    let similarity_label = item.similarity_pct
                                                                        .map(|v| {
                                                                            if use_sift_seed_similarity {
                                                                                format!("similarity to this SIFT master {:.2}%", v)
                                                                            } else {
                                                                                format!("similarity to active {:.2}%", v)
                                                                            }
                                                                        })
                                                                        .unwrap_or_else(|| {
                                                                            if use_sift_seed_similarity {
                                                                                "similarity to this SIFT master n/a".to_string()
                                                                            } else {
                                                                                "similarity to active n/a".to_string()
                                                                            }
                                                                        });
                                                                    ui.colored_label(egui::Color32::from_rgb(100, 180, 255), similarity_label);
                                                                    
                                                                    ui.weak(&res_size_str);
                                                                    
                                                                    let display_name = item.file_name.split_once('/').map(|x| x.1).unwrap_or(&item.file_name);
                                                                    wrapping_monospace_path(ui, display_name);
                                                                    
                                                                    ui.horizontal_wrapped(|ui| {
                                                                        if let Some(s_path) = source_path_opt.as_ref() {
                                                                            if ui.button("📂 Open Folder").clicked() {
                                                                                open_in_dolphin_or_fallback(s_path);
                                                                            }
                                                                            if item_is_video {
                                                                                if ui.button("▶ Open in mpv").clicked() {
                                                                                    let _ = std::process::Command::new("mpv")
                                                                                        .arg(s_path)
                                                                                        .spawn();
                                                                                }
                                                                            } else {
                                                                                if ui.button("👁 View").clicked() {
                                                                                    if let Some(pos) = self.images.iter().position(|p| p == s_path) {
                                                                                        self.current_index = pos;
                                                                                    } else {
                                                                                        self.images.insert(self.current_index + 1, s_path.clone());
                                                                                        self.db_filename_by_path.insert(s_path.clone(), item.file_name.clone());
                                                                                        self.current_index += 1;
                                                                                    }
                                                                                    self.show_grid = false;
                                                                                    self.back_target_is_gallery = true;
                                                                                    self.update_current_file_info();
                                                                                    self.update_side_panel_metadata_if_needed();
                                                                                }
                                                                            }
                                                                        }
                                                                        if ui.button(if expanded { "Collapse" } else { "Expand" }).clicked() {
                                                                            if expanded {
                                                                                self.expanded_duplicate_rows.remove(&item.file_name);
                                                                            } else {
                                                                                self.expanded_duplicate_rows.insert(item.file_name.clone());
                                                                            }
                                                                        }
                                                                    });
                                                                    if expanded {
                                                                        for line in &detail_lines {
                                                                            ui.monospace(line);
                                                                        }
                                                                    }
                                                                });
                                                            });
                                                        });
                                                    }
                                                });
                                            ui.add_space(8.0);
                                        }
                                    } else if phash_similar_groups.is_empty() {
                                        ui.weak("No duplicates or similar files found in database.");
                                    }
                                });
                            } else {
                                ui.weak("Current file is not indexed in the database.");
                            }
                        } else {
                            ui.weak("No image loaded.");
                        }
                    }
                }
                });
        }

        let mut panel = egui::CentralPanel::default();
        if let Some(bg) = self.viewport_bg {
            panel = panel.frame(egui::Frame::NONE.fill(bg));
        }
        panel.show(ctx, |ui| {
            if self.image_editor.is_some() {
                self.show_image_editor(ui, ctx);
            } else if self.show_grid {
                self.show_grid_view(ui, ctx);
            } else {
                if let Some(path) = self.images.get(self.current_index).cloned() {
                let resolved_path = self.get_thumbnail_path(&path);
                let uri = format!("file://{}", resolved_path.to_string_lossy());
                
                // Click and Drag to pan (allocated first to allow zoom-to-mouse math using rect)
                let (rect, response) = ui.allocate_at_least(ui.available_size(), egui::Sense::click_and_drag());
                if response.dragged() {
                    self.offset += response.drag_delta();
                }
 
                // Middle click to recentre and fit
                if response.middle_clicked() {
                    self.zoom = 1.0;
                    self.offset = egui::Vec2::ZERO;
                }
 
                // Interaction / Zoom
                let scroll_delta = ui.input(|i| i.smooth_scroll_delta.y);
                if response.hovered() && scroll_delta != 0.0 {
                    let zoom_factor = (scroll_delta / 200.0).exp();
                    
                    if let Some(mouse_pos) = ui.input(|i| i.pointer.hover_pos()) {
                        let screen_center = rect.center();
                        let v_m = mouse_pos - screen_center;
                        self.offset = v_m - (v_m - self.offset) * zoom_factor;
                    }
                    
                    self.zoom *= zoom_factor;
                }
 
                // Right click context menu to copy path, image, or recenter
                response.context_menu(|ui| {
                    let db_name = self.resolve_db_filename(&path);
                    let is_video_item = db_name
                        .as_ref()
                        .map(|name| is_video_path(Path::new(name)))
                        .unwrap_or_else(|| is_video_path(&path));
                    if ui.button("📂 Open parent folder").clicked() {
                        let actual_path = self.resolve_actual_path(&path);
                        open_in_dolphin_or_fallback(&actual_path);
                        ui.close();
                    }
                    if ui.button("📋 Copy Image Path").clicked() {
                        ui.ctx().copy_text(path.to_string_lossy().to_string());
                        ui.close();
                    }
                    if ui.button("🖼 Copy Image").clicked() {
                        let resolved_path = self.get_thumbnail_path(&path);
                        if let Err(err) = copy_image_file_to_clipboard(&resolved_path) {
                            self.semantic_status = format!("Copy image failed: {err}");
                        }
                        ui.close();
                    }
                    if !is_video_item && ui.button("✏ Edit image").clicked() {
                        self.start_image_editor(&path, ui.ctx());
                        ui.close();
                    }
                    if ui.button("🔍 Fit Image / Recenter").clicked() {
                        self.zoom = 1.0;
                        self.offset = egui::Vec2::ZERO;
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Show most similar").clicked() {
                        let request = PendingSearchRequest::Similar {
                            db_file_name: db_name.clone(),
                            media_path: path.clone(),
                            is_video: is_video_item,
                            timestamp_sec: 0.0,
                        };
                        self.request_search_action(request, ui.ctx());
                        ui.close();
                    }
                    if ui.button("Show more of this person").clicked() {
                        let request = PendingSearchRequest::Person {
                            db_file_name: db_name,
                            media_path: path.clone(),
                            is_video: is_video_item,
                        };
                        self.request_search_action(request, ui.ctx());
                        ui.close();
                    }
                    ui.separator();
                    ui.menu_button("🎨 Viewport Background", |ui| {
                        if ui.radio_value(&mut self.viewport_bg, None, "Default Theme").clicked() {
                            ui.close();
                        }
                        if ui.radio_value(&mut self.viewport_bg, Some(egui::Color32::BLACK), "Pure Black").clicked() {
                            ui.close();
                        }
                        if ui.radio_value(&mut self.viewport_bg, Some(egui::Color32::WHITE), "Pure White").clicked() {
                            ui.close();
                        }
                        if ui.radio_value(&mut self.viewport_bg, Some(egui::Color32::from_rgb(30, 30, 30)), "Dark Charcoal").clicked() {
                            ui.close();
                        }
                        if ui.radio_value(&mut self.viewport_bg, Some(egui::Color32::from_rgb(128, 128, 128)), "Slate Gray").clicked() {
                            ui.close();
                        }
                    });
                });

                if let Some(ref compare_path) = self.compare_target {
                    let left_resolved = self.get_thumbnail_path(&path);
                    let right_resolved = self.get_thumbnail_path(compare_path);
                    let left_uri = format!("file://{}", left_resolved.to_string_lossy());
                    let right_uri = format!("file://{}", right_resolved.to_string_lossy());
                    
                    let builder = egui::UiBuilder::new()
                        .max_rect(rect)
                        .id_salt("sift_compare_viewport");
                    let mut compare_ui = ui.new_child(builder);
                    let avail_size = rect.size();
                    let half_w = (avail_size.x / 2.0 - 12.0).max(10.0);
                    let h = (avail_size.y - 120.0).max(10.0);
                    
                    compare_ui.horizontal(|ui| {
                        ui.add_sized(
                            egui::vec2(half_w, h),
                            egui::Image::new(left_uri)
                                .maintain_aspect_ratio(true)
                                .show_loading_spinner(true)
                        );
                        
                        ui.add_space(12.0);
                        
                        ui.add_sized(
                            egui::vec2(half_w, h),
                            egui::Image::new(right_uri)
                                .maintain_aspect_ratio(true)
                                .show_loading_spinner(true)
                        );
                    });
                    
                    // Draw SIFT alignment info overlay at the bottom of the central viewport
                    let summary_text = if self.sift_running {
                        "⌛ Calculating SIFT correspondence alignment...".to_string()
                    } else if let Some(summary) = &self.sift_pair_overlay {
                        summary.clone()
                    } else {
                        "SIFT alignment not calculated.".to_string()
                    };
                    
                    compare_ui.add_space(16.0);
                    compare_ui.vertical_centered(|ui| {
                        egui::Frame::NONE
                            .fill(egui::Color32::from_black_alpha(190))
                            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(100, 180, 255).gamma_multiply(0.4)))
                            .inner_margin(12.0)
                            .corner_radius(8.0)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.colored_label(egui::Color32::from_rgb(100, 180, 255), "👥 SIFT Matcher Status");
                                    ui.separator();
                                    ui.add(egui::Label::new(
                                        egui::RichText::new(summary_text)
                                            .monospace()
                                            .size(12.0)
                                            .color(egui::Color32::WHITE)
                                    ));
                                    
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        if ui.button("❌ Close comparison").clicked() {
                                            self.compare_target = None;
                                            self.sift_pair_overlay = None;
                                        }
                                    });
                                });
                            });
                    });
                } else {
                    // Calculate image rect
                    let base_size = rect.size();
                    let draw_size = base_size * self.zoom;
                    let draw_pos = rect.center() + self.offset - draw_size / 2.0;
                    let draw_rect = egui::Rect::from_min_size(draw_pos, draw_size);
                    
                    // Use ui.put to place the image widget
                    ui.put(draw_rect, egui::Image::new(uri).maintain_aspect_ratio(true).show_loading_spinner(false));
                }
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("No image loaded");
                });
            }
            }
        });

        if self.flat_loading || self.grid_loading {
            ctx.request_repaint();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_roots() -> HashMap<String, PathBuf> {
        HashMap::from([(
            "collection".to_string(),
            PathBuf::from("/media/library"),
        )])
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
}

impl ImageViewer {
    fn get_subdirectories(&self, path: &Path) -> Vec<PathBuf> {
        let mut dirs = Vec::new();
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.filter_map(|e| e.ok()) {
                let p = entry.path();
                if p.is_dir() {
                    if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                        if !name.starts_with('.') {
                            dirs.push(p);
                        }
                    }
                }
            }
        }
        dirs.sort_by(|a, b| {
            let a_name = a.file_name().and_then(|n| n.to_str()).unwrap_or("").to_lowercase();
            let b_name = b.file_name().and_then(|n| n.to_str()).unwrap_or("").to_lowercase();
            a_name.cmp(&b_name)
        });
        dirs
    }

    fn show_home_page_view(&mut self, ctx: &egui::Context) {
        let current_dir_opt = self.home_current_dir.clone();

        egui::CentralPanel::default().show(ctx, |ui| {
            let mut opened_folder = false;
            ui.add_space(8.0);
            
            // Dolphin-like location toolbar at the top
            ui.horizontal(|ui| {
                // 1. Up Button
                let has_parent = current_dir_opt.is_some();
                let up_btn = ui.add_enabled_ui(has_parent, |ui| {
                    ui.add(egui::Button::new("⬆ Up").min_size(egui::vec2(50.0, 26.0)))
                });
                
                if has_parent && up_btn.inner.clicked() {
                    if let Some(ref current_dir) = current_dir_opt {
                        self.home_current_dir = current_dir.parent().map(|p| p.to_path_buf());
                        self.home_selected_dir = None;
                    }
                }
                
                ui.add_space(4.0);
                
                // 2. Path Display Location Bar (matches standard Dolphin address bar)
                let path_str = match &current_dir_opt {
                    Some(p) => p.to_string_lossy().to_string(),
                    None => "/".to_string(), // Root disks
                };
                
                egui::Frame::NONE
                    .fill(ui.visuals().extreme_bg_color)
                    .stroke(egui::Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color))
                    .inner_margin(egui::vec2(8.0, 4.0))
                    .corner_radius(4.0)
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width() - 170.0);
                        ui.label(egui::RichText::new(&path_str).monospace().size(13.0));
                    });
                
                // 3. Open button on the right
                let target_dir = self.home_selected_dir.clone().or(current_dir_opt.clone());
                let has_target = target_dir.is_some();
                
                let open_btn = ui.add_enabled_ui(has_target, |ui| {
                    ui.add(egui::Button::new(
                        egui::RichText::new("Open Folder")
                            .strong()
                            .color(egui::Color32::WHITE)
                    )
                    .fill(ui.visuals().selection.bg_fill)
                    .min_size(egui::vec2(100.0, 26.0)))
                });
                
                if has_target && open_btn.inner.clicked() {
                    if let Some(dir) = target_dir {
                        self.open_folder_path(dir.clone());
                        self.show_home_page = false;
                        self.show_grid = true;
                        self.start_recursive_scan();
                        opened_folder = true;
                    }
                }
            });

            if opened_folder {
                ctx.request_repaint();
                return;
            }
            
            ui.add_space(8.0);
            ui.separator();
            ui.add_space(8.0);
            
            // Pane container frame (high density desktop file pane)
            egui::Frame::NONE
                .fill(ui.visuals().extreme_bg_color)
                .stroke(egui::Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color))
                .inner_margin(2.0)
                .corner_radius(4.0)
                .show(ui, |ui| {
                    ui.set_min_height(ui.available_height() - 6.0);
                    
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        let mut items = Vec::new();
                        let is_disk_level = current_dir_opt.is_none();
                        
                        let db_roots = get_db_roots();

                        if is_disk_level {
                            for disk in get_system_disks() {
                                items.push((disk, true));
                            }
                        } else {
                            if let Some(ref current_dir) = current_dir_opt {
                                for sub in self.get_subdirectories(current_dir) {
                                    items.push((sub, false));
                                }
                            }
                        }
                        
                        if items.is_empty() {
                            ui.vertical_centered(|ui| {
                                ui.add_space(40.0);
                                ui.weak("This folder contains no subfolders.");
                                ui.add_space(40.0);
                            });
                        } else {
                            ui.vertical(|ui| {
                                ui.spacing_mut().item_spacing = egui::vec2(0.0, 1.0); // Dense desktop list spacing
                                
                                for (item_path, is_disk) in items {
                                    let name = if is_disk {
                                        if item_path == PathBuf::from("/") {
                                            "System Root (/)".to_string()
                                        } else if item_path.to_string_lossy().contains("/home/") {
                                            format!("Home ({})", item_path.file_name().and_then(|n| n.to_str()).unwrap_or("User"))
                                        } else {
                                            item_path.file_name()
                                                .and_then(|n| n.to_str())
                                                .map(|s| s.to_string())
                                                .unwrap_or_else(|| item_path.to_string_lossy().to_string())
                                        }
                                    } else {
                                        item_path.file_name()
                                            .and_then(|n| n.to_str())
                                            .unwrap_or("")
                                            .to_string()
                                    };
                                    
                                    let is_ai = is_path_ai_backed_with_roots(&item_path, &db_roots);
                                    let is_selected = self.home_selected_dir.as_ref() == Some(&item_path);
                                    
                                    // Allocate dense row rect
                                    let (rect, response) = ui.allocate_exact_size(
                                        egui::vec2(ui.available_width(), 26.0),
                                        egui::Sense::click()
                                    );
                                    
                                    if response.double_clicked() {
                                        self.home_current_dir = Some(item_path.clone());
                                        self.home_selected_dir = None;
                                    } else if response.clicked() {
                                        self.home_selected_dir = Some(item_path.clone());
                                    }
                                    
                                    // Draw row background selection/hover highlight
                                    let row_bg = if is_selected {
                                        ui.visuals().selection.bg_fill
                                    } else if response.hovered() {
                                        ui.visuals().widgets.hovered.bg_fill.gamma_multiply(0.2)
                                    } else {
                                        egui::Color32::TRANSPARENT
                                    };
                                    
                                    if row_bg != egui::Color32::TRANSPARENT {
                                        ui.painter().rect_filled(rect, 2.0, row_bg);
                                    }
                                    
                                    // Render columns inside the row
                                    ui.allocate_ui_at_rect(rect.shrink2(egui::vec2(8.0, 2.0)), |ui| {
                                        ui.horizontal(|ui| {
                                            let icon = if is_disk {
                                                "💾"
                                            } else {
                                                "📁"
                                            };
                                            
                                            let text_color = if is_selected {
                                                egui::Color32::WHITE
                                            } else {
                                                ui.visuals().widgets.noninteractive.text_color()
                                            };
                                            
                                            ui.label(egui::RichText::new(icon).size(14.0));
                                            ui.add_space(4.0);
                                            ui.label(egui::RichText::new(&name).color(text_color));
                                            
                                            // Push column 2 to the right
                                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                                let type_str = if is_disk {
                                                    if item_path == PathBuf::from("/") {
                                                        "System Disk"
                                                    } else if item_path.to_string_lossy().contains("/home/") {
                                                        "User Directory"
                                                    } else {
                                                        "Disk Partition"
                                                    }
                                                } else if is_ai {
                                                    "Indexed Folder"
                                                } else {
                                                    "Folder"
                                                };
                                                
                                                let type_color = if is_selected {
                                                    egui::Color32::WHITE
                                                } else if is_ai {
                                                    egui::Color32::from_rgb(140, 160, 255)
                                                } else {
                                                    ui.visuals().weak_text_color()
                                                };
                                                
                                                ui.label(egui::RichText::new(type_str).color(type_color).size(11.0));
                                            });
                                        });
                                    });
                                }
                            });
                        }
                    });
                });
        });
    }
}
