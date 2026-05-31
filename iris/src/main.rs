use eframe::egui;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use arrow_array::{
    Array, BooleanArray, Float32Array, Float64Array, ListArray, RecordBatch,
    StringArray, StructArray,
};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
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
    if args.len() < 2 {
        eprintln!("Usage: iris [--same-window | -s] [--no-daemon] <image_path>");
        std::process::exit(1);
    }

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

    let image_path = match image_arg {
        Some(path_str) => PathBuf::from(&path_str).canonicalize().unwrap_or_else(|_| PathBuf::from(&path_str)),
        None => {
            eprintln!("Usage: iris [--same-window | -s] [--no-daemon] <image_path>");
            std::process::exit(1);
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
            Ok(Box::new(ImageViewer::new(image_path_clone, rx_taken, ctx_shared_clone)))
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
                Ok(Box::new(ImageViewer::new(image_path, rx_taken, ctx_shared)))
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
                if matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "bmp" | "gif" | "webp" | "tiff" | "avif" | "heif" | "heic" | "ico" | "icns" | "svg") {
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

#[derive(PartialEq, Clone, Copy)]
enum ExploreMode {
    Filesystem,
    Semantic,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SearchMode {
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
}

#[derive(Clone)]
struct SimilarFile {
    file_name: String,
    is_video: bool,
    similarity_pct: Option<f32>,
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
    similar_by_master: HashMap<String, Vec<SimilarFile>>,
    phash_master_by_file: HashMap<String, String>,
    sift_info_by_file: HashMap<String, SiftInfo>,
    sift_root_by_file: HashMap<String, String>,
    sift_members_by_root: HashMap<String, Vec<String>>,
    basename_to_db_filename: HashMap<String, String>,
    encoder: ClipTextEncoder,
}

struct UnifiedDbData {
    clip_index: ClipIndex,
    face_index: FaceIndex,
    ocr_index: OcrIndex,
    similar_by_master: HashMap<String, Vec<SimilarFile>>,
    phash_master_by_file: HashMap<String, String>,
    sift_info_by_file: HashMap<String, SiftInfo>,
    sift_root_by_file: HashMap<String, String>,
    sift_members_by_root: HashMap<String, Vec<String>>,
}

fn resolve_root(
    name: &str,
    direct_root_by_file: &HashMap<String, String>,
    master_images: &HashSet<String>,
) -> String {
    let mut current = name.to_string();
    for _ in 0..16 {
        let next = match direct_root_by_file.get(current.as_str()) {
            Some(v) => v.clone(),
            None => return current,
        };
        if !master_images.contains(next.as_str()) {
            return current;
        }
        if next == current {
            return current;
        }
        current = next;
    }
    current
}

async fn load_all_database_indices(db_dir: &Path, table_name: &str) -> Result<UnifiedDbData> {
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
            "face_groups",
            "ocr_groups",
            "dedupe_match_file",
            "dedupe_similarity_pct",
            "sift_match_file",
            "sift_match_score",
            "sift_match_inliers",
            "sift_match_good_matches",
            "sift_match_inlier_ratio",
            "sift_match_checked",
        ]))
        .execute()
        .await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;

    let mut clip_entries = Vec::new();
    let mut clip_dim = None;
    let mut clip_seen = HashSet::new();
    
    let mut face_entries = Vec::new();
    let mut face_seen = HashSet::new();
    
    let mut ocr_entries = Vec::new();
    let mut ocr_seen = HashSet::new();

    let mut similar_by_master: HashMap<String, Vec<SimilarFile>> = HashMap::new();
    let mut phash_master_by_file: HashMap<String, String> = HashMap::new();
    let mut sift_info_by_file: HashMap<String, SiftInfo> = HashMap::new();
    let mut master_images = HashSet::new();
    let mut direct_root_by_file: HashMap<String, String> = HashMap::new();

    for batch in &batches {
        // Parse Clip
        parse_batch(batch, &mut clip_entries, &mut clip_dim, &mut clip_seen)?;
        
        // Parse Face
        parse_face_batch(batch, &mut face_entries, &mut face_seen)?;
        
        // Parse OCR
        parse_ocr_batch(batch, &mut ocr_entries, &mut ocr_seen)?;

        // Parse Similar
        let file_names = string_col(batch, "file_name")?;
        let is_video = bool_col(batch, "is_video")?;
        let dedupe_match = string_col(batch, "dedupe_match_file")?;
        let similarity_col = batch.column_by_name("dedupe_similarity_pct");

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

    let clip_dim = clip_dim.unwrap_or(512);
    let clip_index = ClipIndex {
        entries: clip_entries,
        dim: clip_dim,
        file_count: clip_seen.len(),
    };

    let face_index = FaceIndex {
        entries: face_entries,
        file_count: face_seen.len(),
    };

    let ocr_index = OcrIndex {
        entries: ocr_entries,
        file_count: ocr_seen.len(),
    };

    for values in similar_by_master.values_mut() {
        values.sort_by(|a, b| {
            b.similarity_pct
                .unwrap_or(f32::NEG_INFINITY)
                .partial_cmp(&a.similarity_pct.unwrap_or(f32::NEG_INFINITY))
                .unwrap_or(Ordering::Equal)
        });
    }

    // Build SIFT groups using resolve_root
    let mut sift_root_by_file: HashMap<String, String> = HashMap::new();
    let mut sift_members_by_root: HashMap<String, Vec<String>> = HashMap::new();
    let mut raw_groups: HashMap<String, Vec<String>> = HashMap::new();
    for file_name in &master_images {
        let root = resolve_root(file_name.as_str(), &direct_root_by_file, &master_images);
        raw_groups.entry(root).or_default().push(file_name.clone());
    }
    for members in raw_groups.into_values() {
        if members.len() <= 1 {
            continue;
        }
        let mut sorted_members = members;
        sorted_members.sort_unstable();
        let canonical = resolve_root(
            sorted_members[0].as_str(),
            &direct_root_by_file,
            &master_images,
        );
        for member in &sorted_members {
            sift_root_by_file.insert(member.clone(), canonical.clone());
        }
        sift_members_by_root.insert(canonical, sorted_members);
    }

    Ok(UnifiedDbData {
        clip_index,
        face_index,
        ocr_index,
        similar_by_master,
        phash_master_by_file,
        sift_info_by_file,
        sift_root_by_file,
        sift_members_by_root,
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
        && info.inlier_ratio.unwrap_or(0.0) >= 0.75
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
) -> Vec<SearchResult> {
    let merged = index
        .entries
        .par_chunks(4096)
        .map(|chunk| {
            let mut local: HashMap<String, (f32, bool, f32)> = HashMap::new();
            for entry in chunk {
                if video_only && !entry.is_video {
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
) -> Vec<SearchResult> {
    let query_lower = query.trim().to_lowercase();
    if query_lower.is_empty() {
        return Vec::new();
    }
    let terms: Vec<&str> = query_lower
        .split_whitespace()
        .filter(|term| !term.is_empty())
        .collect();
    let term_den = terms.len().max(1) as f32;

    let merged = index
        .entries
        .par_chunks(4096)
        .map(|chunk| {
            let mut local: HashMap<String, (f32, bool, f32)> = HashMap::new();
            for entry in chunk {
                if video_only && !entry.is_video {
                    continue;
                }
                let phrase_hit = entry.text_lower.contains(query_lower.as_str());
                let term_hits = terms
                    .iter()
                    .filter(|term| entry.text_lower.contains(**term))
                    .count() as f32;
                if !phrase_hit && term_hits <= 0.0 {
                    continue;
                }
                let term_score = term_hits / term_den;
                let score = if phrase_hit { 2.0 + term_score } else { term_score };
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

const FACE_MATCH_MIN_SCORE: f32 = 0.35;

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

fn get_db_roots() -> HashMap<String, PathBuf> {
    let mut roots = HashMap::new();
    roots.insert(
        "phone".to_string(),
        PathBuf::from("/media/lewis/1b/Phone")
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from("/media/lewis/1b/Phone")),
    );
    roots.insert(
        "telegram_backup".to_string(),
        PathBuf::from("/media/lewis/1b/Telegram Backup")
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from("/media/lewis/1b/Telegram Backup")),
    );
    roots
}

fn resolve_media_path(
    roots: &HashMap<String, PathBuf>,
    db_dir: &Path,
    file_name: &str,
    timestamp_sec: f32,
) -> Result<PathBuf> {
    let (collection, rel) = file_name
        .split_once('/')
        .ok_or_else(|| anyhow!("file_name does not contain collection id"))?;
    let root = roots
        .get(collection)
        .ok_or_else(|| anyhow!("no collection-root for {collection}"))?;
    let rel_path = Path::new(rel);
    let source = root.join(rel_path);
    if is_video_path(&source) {
        if let Some(still) = resolve_video_still(root, db_dir, rel_path, timestamp_sec)? {
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
        .ok_or_else(|| anyhow!("no collection-root for {collection}"))?;
    Ok(root.join(Path::new(rel)))
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
    let output = Command::new("uv")
        .current_dir("/home/lewis/Dev/imagesearch")
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

struct ImageViewer {
    images: Vec<PathBuf>,
    current_index: usize,
    zoom: f32,
    offset: egui::Vec2,
    exif_data: String,
    show_exif: bool,
    chunks: Vec<FileChunk>,
    viewport_bg: Option<egui::Color32>,
    rx: Receiver<PathBuf>,
    show_grid: bool,
    recursive_images: Vec<PathBuf>,
    grid_filter: String,
    grid_loading: bool,
    recursive_rx: Option<Receiver<PathBuf>>,
    back_target_is_gallery: bool,
    side_panel_mode: SidePanelMode,
    exif_search: String,
    open_target: PathBuf,
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
    explore_mode: ExploreMode,
    db_loaded: bool,
    db_loading: bool,
    db_rx: Option<Receiver<Result<DatabaseIndices, String>>>,
    db_indices: Option<DatabaseIndices>,
    semantic_query: String,
    semantic_limit: usize,
    semantic_video_only: bool,
    semantic_mode: SearchMode,
    semantic_results: Vec<SearchResult>,
    semantic_status: String,
    
    // Duplicates & SIFT states
    compare_target: Option<PathBuf>,
    sift_pair_overlay: Option<String>,
    sift_running: bool,
    sift_rx: Option<Receiver<Result<String, String>>>,
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

impl ImageViewer {
    fn new(path: PathBuf, rx: Receiver<PathBuf>, ctx_shared: Arc<Mutex<Option<egui::Context>>>) -> Self {
        let path = path.canonicalize().unwrap_or(path);
        let open_target = path.clone();

        let mut images = Vec::new();
        let flat_loading = true;
        let flat_images_shared = Arc::new(Mutex::new(None));

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

        let (thumbnail_tx, thumbnail_rx) = std::sync::mpsc::channel::<(PathBuf, egui::ColorImage)>();

        let mut viewer = Self {
            images,
            current_index: 0,
            zoom: 1.0,
            offset: egui::Vec2::ZERO,
            exif_data: String::new(),
            show_exif: false,
            chunks: Vec::new(),
            viewport_bg: None,
            rx,
            show_grid: false,
            recursive_images: Vec::new(),
            grid_filter: String::new(),
            grid_loading: false,
            recursive_rx: None,
            back_target_is_gallery: false,
            side_panel_mode: SidePanelMode::Layout,
            exif_search: String::new(),
            open_target,
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
            explore_mode: ExploreMode::Filesystem,
            db_loaded: false,
            db_loading: false,
            db_rx: None,
            db_indices: None,
            semantic_query: String::new(),
            semantic_limit: 80,
            semantic_video_only: false,
            semantic_mode: SearchMode::Clip,
            semantic_results: Vec::new(),
            semantic_status: "Ready. Enter a phrase and press Search.".to_string(),
            
            // SIFT defaults
            compare_target: None,
            sift_pair_overlay: None,
            sift_running: false,
            sift_rx: None,
        };
        
        viewer.update_exif();
        viewer
    }

    fn start_lazy_db_load(&mut self, ctx: &egui::Context) {
        if self.db_loaded || self.db_loading {
            return;
        }
        self.db_loading = true;
        self.semantic_status = "Initializing ONNX Runtime and loading database indices... Please wait.".to_string();
        let (tx, rx) = std::sync::mpsc::channel();
        self.db_rx = Some(rx);
        let ctx_clone = ctx.clone();
        
        std::thread::spawn(move || {
            let rt = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build() {
                    Ok(rt) => rt,
                    Err(e) => {
                        let _ = tx.send(Err(format!("Failed to create tokio runtime: {}", e)));
                        ctx_clone.request_repaint();
                        return;
                    }
                };
                
            let result: Result<DatabaseIndices, anyhow::Error> = rt.block_on(async {
                let db_dir = Path::new("/media/lewis/1b/lancedb");
                let table_name = "media_index";
                let onnx_path = Path::new("/home/lewis/Dev/imagesearch/models/clip-text/clip_text.onnx");
                let tokenizer_path = Path::new("/home/lewis/Dev/imagesearch/models/clip-text/tokenizer.json");
                
                let db_fut = load_all_database_indices(db_dir, table_name);
                let encoder_fut = async {
                    ClipTextEncoder::new(onnx_path, tokenizer_path, 64)
                };

                let (db_data, encoder) = tokio::try_join!(db_fut, encoder_fut)?;
                
                let mut basename_to_db_filename = HashMap::new();
                for entry in &db_data.clip_index.entries {
                    if let Some(fname) = Path::new(entry.file_name.as_ref()).file_name() {
                        let base = fname.to_string_lossy().to_lowercase();
                        basename_to_db_filename.entry(base).or_insert_with(|| entry.file_name.to_string());
                    }
                }
                for key in db_data.phash_master_by_file.keys() {
                    if let Some(fname) = Path::new(key).file_name() {
                        let base = fname.to_string_lossy().to_lowercase();
                        basename_to_db_filename.entry(base).or_insert_with(|| key.clone());
                    }
                }
                for key in db_data.similar_by_master.keys() {
                    if let Some(fname) = Path::new(key).file_name() {
                        let base = fname.to_string_lossy().to_lowercase();
                        basename_to_db_filename.entry(base).or_insert_with(|| key.clone());
                    }
                }
                for key in db_data.sift_info_by_file.keys() {
                    if let Some(fname) = Path::new(key).file_name() {
                        let base = fname.to_string_lossy().to_lowercase();
                        basename_to_db_filename.entry(base).or_insert_with(|| key.clone());
                    }
                }

                Ok(DatabaseIndices {
                    clip_index: Arc::new(db_data.clip_index),
                    face_index: Arc::new(db_data.face_index),
                    ocr_index: Arc::new(db_data.ocr_index),
                    similar_by_master: db_data.similar_by_master,
                    phash_master_by_file: db_data.phash_master_by_file,
                    sift_info_by_file: db_data.sift_info_by_file,
                    sift_root_by_file: db_data.sift_root_by_file,
                    sift_members_by_root: db_data.sift_members_by_root,
                    basename_to_db_filename,
                    encoder,
                })
            });
            
            match result {
                Ok(indices) => {
                    let _ = tx.send(Ok(indices));
                }
                Err(e) => {
                    let _ = tx.send(Err(e.to_string()));
                }
            }
            ctx_clone.request_repaint();
        });
    }

    fn poll_db_load(&mut self) {
        let Some(rx) = self.db_rx.take() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(indices)) => {
                self.db_indices = Some(indices);
                self.db_loaded = true;
                self.db_loading = false;
                self.semantic_status = "AI Explorer Database Loaded successfully! Ready to search.".to_string();
            }
            Ok(Err(err)) => {
                self.db_loading = false;
                self.semantic_status = format!("❌ AI DB Initialization failed: {err}");
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                self.db_rx = Some(rx);
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.db_loading = false;
                self.semantic_status = "❌ AI DB Loader thread disconnected unexpectedly.".to_string();
            }
        }
    }

    fn get_db_filename_from_path(&self, path: &Path) -> Option<String> {
        let roots = get_db_roots();
        let canon_path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        
        for (col_id, root_path) in &roots {
            let root_canon = root_path.canonicalize().unwrap_or_else(|_| root_path.clone());
            if let Ok(rel) = canon_path.strip_prefix(&root_canon) {
                let rel_str = rel.to_string_lossy().replace('\\', "/");
                return Some(format!("{}/{}", col_id, rel_str.trim_start_matches('/')));
            }
            if let Ok(rel) = path.strip_prefix(root_path) {
                let rel_str = rel.to_string_lossy().replace('\\', "/");
                return Some(format!("{}/{}", col_id, rel_str.trim_start_matches('/')));
            }
        }
        
        // Fallback to substring matching if prefix checks fail
        let path_str = path.to_string_lossy().replace('\\', "/");
        for (col_id, root_path) in &roots {
            let root_str = root_path.to_string_lossy().replace('\\', "/");
            if path_str.starts_with(&root_str) {
                let rel = &path_str[root_str.len()..];
                return Some(format!("{}/{}", col_id, rel.trim_start_matches('/')));
            }
            if let Ok(root_canon) = root_path.canonicalize() {
                let root_canon_str = root_canon.to_string_lossy().replace('\\', "/");
                if path_str.starts_with(&root_canon_str) {
                    let rel = &path_str[root_canon_str.len()..];
                    return Some(format!("{}/{}", col_id, rel.trim_start_matches('/')));
                }
            }
        }
        None
    }

    fn draw_thumbnail_async(
        &mut self,
        ui: &mut egui::Ui,
        path: &Path,
        side_thumb: f32,
    ) {
        if let Some(texture) = self.thumbnail_textures.get(path) {
            ui.add(
                egui::Image::from_texture(texture)
                    .max_size(egui::vec2(side_thumb, side_thumb))
                    .maintain_aspect_ratio(true)
            );
        } else if self.thumbnail_failed.contains(path) {
            let (rect, _) = ui.allocate_exact_size(
                egui::vec2(side_thumb, side_thumb),
                egui::Sense::hover(),
            );
            ui.painter().rect_filled(
                rect,
                4.0,
                egui::Color32::from_gray(30),
            );
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "⚠️ Error",
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
            
            if !self.thumbnail_loading.contains(path) && self.thumbnail_active_threads < 8 {
                self.thumbnail_loading.insert(path.to_path_buf());
                self.thumbnail_active_threads += 1;
                let path_clone = path.to_path_buf();
                let tx_clone = self.thumbnail_tx.clone();
                let ctx_clone = ui.ctx().clone();
                std::thread::spawn(move || {
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

    fn search_now(&mut self) {
        match self.semantic_mode {
            SearchMode::Clip => self.search_clip_now(),
            SearchMode::Ocr => self.search_ocr_now(),
        }
    }

    fn search_clip_now(&mut self) {
        let q = self.semantic_query.trim().to_string();
        if q.is_empty() {
            self.semantic_status = "Please enter a search phrase first.".to_string();
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
        let mut results = search_index(&indices.clip_index, &query_vector, pre_limit, self.semantic_video_only);
        if !self.semantic_video_only {
            results = collapse_sift_grouped_results(results, &indices.sift_root_by_file, self.semantic_limit);
        } else {
            results.truncate(self.semantic_limit);
        }
        
        let db_roots = get_db_roots();
        let db_dir = Path::new("/media/lewis/1b/lancedb");
        for row in &mut results {
            row.media_path = resolve_media_path(&db_roots, db_dir, &row.file_name, row.timestamp_sec).ok();
        }

        let took = started.elapsed().as_millis();
        self.semantic_status = format!(
            "✓ Found {} CLIP results in {} ms across {} index vectors",
            results.len(),
            took,
            indices.clip_index.entries.len()
        );
        self.semantic_results = results;
    }

    fn search_ocr_now(&mut self) {
        let q = self.semantic_query.trim().to_string();
        if q.is_empty() {
            self.semantic_status = "Please enter an OCR word or phrase first.".to_string();
            return;
        }
        let Some(indices) = &self.db_indices else {
            self.semantic_status = "AI Database index is not loaded yet.".to_string();
            return;
        };

        let started = Instant::now();
        let pre_limit = (self.semantic_limit.saturating_mul(6)).max(self.semantic_limit);
        let mut results = search_ocr_index(&indices.ocr_index, &q, pre_limit, self.semantic_video_only);
        if !self.semantic_video_only {
            results = collapse_sift_grouped_results(results, &indices.sift_root_by_file, self.semantic_limit);
        } else {
            results.truncate(self.semantic_limit);
        }
        
        let db_roots = get_db_roots();
        let db_dir = Path::new("/media/lewis/1b/lancedb");
        for row in &mut results {
            row.media_path = resolve_media_path(&db_roots, db_dir, &row.file_name, row.timestamp_sec).ok();
        }

        let took = started.elapsed().as_millis();
        self.semantic_status = format!(
            "✓ Found {} OCR results in {} ms across {} index entries",
            results.len(),
            took,
            indices.ocr_index.entries.len()
        );
        self.semantic_results = results;
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

    fn start_recursive_scan(&mut self) {
        self.grid_loading = true;
        self.recursive_images.clear();
        self.thumbnail_textures.clear();
        self.thumbnail_loading.clear();
        self.thumbnail_failed.clear();
        self.thumbnail_active_threads = 0;

        let start_dir = if self.open_target.is_dir() {
            self.open_target.clone()
        } else {
            self.open_target.parent().unwrap_or_else(|| Path::new(".")).to_path_buf()
        };

        let start_dir_canon = start_dir.canonicalize().unwrap_or(start_dir);

        let (tx, rx) = std::sync::mpsc::channel();
        self.recursive_rx = Some(rx);

        std::thread::spawn(move || {
            let mut visited = std::collections::HashSet::new();
            collect_images_recursive(&start_dir_canon, &tx, &mut visited);
        });
    }

    fn open_image_path(&mut self, path: PathBuf) {
        let old_start_dir = if self.open_target.is_dir() {
            self.open_target.clone()
        } else {
            self.open_target.parent().unwrap_or_else(|| Path::new(".")).to_path_buf()
        };
        let old_start_dir_canon = old_start_dir.canonicalize().unwrap_or(old_start_dir);

        let path = path.canonicalize().unwrap_or(path);
        self.open_target = path.clone();
        self.zoom = 1.0;
        self.offset = egui::Vec2::ZERO;

        let new_start_dir = if self.open_target.is_dir() {
            self.open_target.clone()
        } else {
            self.open_target.parent().unwrap_or_else(|| Path::new(".")).to_path_buf()
        };
        let new_start_dir_canon = new_start_dir.canonicalize().unwrap_or(new_start_dir);

        if old_start_dir_canon != new_start_dir_canon {
            self.recursive_images.clear();
            self.back_target_is_gallery = false;
        }

        if path.is_dir() {
            self.images.clear();
            self.current_index = 0;
            self.update_exif(); // Clear exif since it's a dir
            self.flat_loading = true;
            if let Ok(mut lock) = self.flat_images_shared.lock() {
                *lock = None;
            }
            
            let shared = self.flat_images_shared.clone();
            let parent_absolute = path.clone();
            std::thread::spawn(move || {
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
        } else {
            self.images = vec![path.clone()];
            self.current_index = 0;
            self.update_exif(); // Load exif immediately for the active image
            self.flat_loading = true;
            if let Ok(mut lock) = self.flat_images_shared.lock() {
                *lock = None;
            }

            let shared = self.flat_images_shared.clone();
            let parent = path.parent().unwrap_or(Path::new(".")).to_path_buf();
            let parent_absolute = parent.canonicalize().unwrap_or(parent);
            std::thread::spawn(move || {
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
        if let Some(path) = self.images.get(self.current_index) {
            self.current_dimensions = match image::image_dimensions(path) {
                Ok((w, h)) => format!("{}x{}", w, h),
                Err(_) => "Unknown px".to_string(),
            };

            self.current_file_size = std::fs::metadata(path)
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

            let output = Command::new("exiftool")
                .args(["-a", "-u", "-g1", "-H"])
                .arg(path)
                .output();

            self.exif_data = match output {
                Ok(out) => String::from_utf8_lossy(&out.stdout).to_string(),
                Err(e) => format!("Error running exiftool: {}", e),
            };

            if let Ok(bytes) = std::fs::read(path) {
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
        } else {
            self.exif_data = String::new();
            self.chunks = Vec::new();
            self.current_dimensions = String::new();
            self.current_file_size = String::new();
        }
    }

    fn next_image(&mut self) {
        if !self.images.is_empty() {
            self.current_index = (self.current_index + 1) % self.images.len();
            self.update_exif();
        }
    }

    fn prev_image(&mut self) {
        if !self.images.is_empty() {
            if self.current_index == 0 {
                self.current_index = self.images.len() - 1;
            } else {
                self.current_index -= 1;
            }
            self.update_exif();
        }
    }

    fn show_grid_view(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.vertical(|ui| {
            // Top Tab switcher
            ui.horizontal(|ui| {
                let fs_text = egui::RichText::new("🖼 Filesystem Gallery").strong();
                let ai_text = egui::RichText::new("🔍 AI Explorer").strong();
                ui.selectable_value(&mut self.explore_mode, ExploreMode::Filesystem, fs_text);
                ui.selectable_value(&mut self.explore_mode, ExploreMode::Semantic, ai_text);
                
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("❌ Close Gallery [G]").clicked() {
                        self.show_grid = false;
                    }
                });
            });
            ui.add_space(6.0);
            ui.separator();
            ui.add_space(8.0);
            
            match self.explore_mode {
                ExploreMode::Filesystem => {
                    // Original filesystem view
                    ui.horizontal(|ui| {
                        ui.heading("Local filesystem directory scan");
                        ui.add_space(12.0);
                        ui.label("Filter:");
                        ui.add(egui::TextEdit::singleline(&mut self.grid_filter)
                            .hint_text("🔍 Filter by filename...")
                            .desired_width(200.0));
                        ui.add_space(12.0);
                        if self.grid_loading {
                            ui.add(egui::Spinner::new().size(16.0));
                            ui.weak("Scanning subdirectories...");
                        } else {
                            ui.weak(format!("{} images found", self.recursive_images.len()));
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("🔄 Refresh").clicked() {
                                self.start_recursive_scan();
                            }
                        });
                    });
                    ui.add_space(8.0);
                    
                    let filter = self.grid_filter.to_lowercase();
                    let filtered_images: Vec<&PathBuf> = self.recursive_images.iter()
                        .filter(|p| {
                            if filter.is_empty() {
                                true
                            } else {
                                let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
                                name.contains(&filter)
                            }
                        })
                        .collect();
                        
                    let mut clicked_path = None;
                    let mut single_clicked_path = None;
                    
                    if filtered_images.is_empty() {
                        ui.centered_and_justified(|ui| {
                            if self.grid_loading {
                                ui.weak("Scanning files, please wait...");
                            } else {
                                ui.weak("No images found matching filter.");
                            }
                        });
                    } else {
                        let available_width = ui.available_width() - 16.0;
                        let col_width = 130.0 + 12.0;
                        let cols = (available_width / col_width).floor().max(1.0) as usize;
                        let rows: Vec<&[&PathBuf]> = filtered_images.chunks(cols).collect();

                        let row_height = 150.0 + 12.0;
                        let num_rows = rows.len();

                        egui::ScrollArea::vertical().id_salt("gallery_scroll_area").show_rows(ui, row_height, num_rows, |ui, row_range| {
                            ui.spacing_mut().item_spacing = egui::vec2(12.0, 12.0);
                            for row_idx in row_range {
                                let row_images = rows[row_idx];
                                ui.horizontal(|ui| {
                                    for path in row_images {
                                        let is_current = if let Some(curr_p) = self.images.get(self.current_index) {
                                            curr_p == *path
                                        } else {
                                            false
                                        };
                                        
                                        let (rect, response) = ui.allocate_at_least(egui::vec2(130.0, 150.0), egui::Sense::click());
                                        let is_hovered = response.hovered();
                                        let is_clicked = response.clicked();
                                        
                                        let card_bg = if is_clicked {
                                            ui.visuals().selection.bg_fill.gamma_multiply(0.3)
                                        } else if is_hovered {
                                            ui.visuals().code_bg_color.gamma_multiply(1.5)
                                        } else if is_current {
                                            ui.visuals().selection.bg_fill.gamma_multiply(0.15)
                                        } else {
                                            ui.visuals().code_bg_color
                                        };
                                        
                                        let card_stroke = if is_current {
                                            egui::Stroke::new(2.0, ui.visuals().selection.bg_fill)
                                        } else if is_hovered {
                                            egui::Stroke::new(1.0, ui.visuals().selection.bg_fill.gamma_multiply(0.5))
                                        } else {
                                            egui::Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color.gamma_multiply(0.3))
                                        };
                                        
                                        let builder = egui::UiBuilder::new()
                                            .max_rect(rect)
                                            .layout(egui::Layout::top_down(egui::Align::Center))
                                            .id_salt(*path);
                                        let mut child_ui = ui.new_child(builder);
                                        egui::Frame::NONE
                                            .fill(card_bg)
                                            .stroke(card_stroke)
                                            .inner_margin(8.0)
                                            .corner_radius(6.0)
                                            .show(&mut child_ui, |ui| {
                                                ui.vertical_centered(|ui| {
                                                    if let Some(texture) = self.thumbnail_textures.get(*path) {
                                                        ui.add(
                                                            egui::Image::from_texture(texture)
                                                                .max_size(egui::vec2(110.0, 110.0))
                                                                .maintain_aspect_ratio(true)
                                                        );
                                                    } else if self.thumbnail_failed.contains(*path) {
                                                        ui.vertical_centered(|ui| {
                                                            ui.add_space(40.0);
                                                            ui.weak("⚠️ Failed");
                                                            ui.add_space(40.0);
                                                        });
                                                    } else {
                                                        ui.vertical_centered(|ui| {
                                                            ui.add_space(45.0);
                                                            ui.add(egui::Spinner::new().size(20.0));
                                                            ui.add_space(45.0);
                                                        });
                                                        
                                                        if !self.thumbnail_loading.contains(*path) && self.thumbnail_active_threads < 4 {
                                                            self.thumbnail_loading.insert((*path).clone());
                                                            self.thumbnail_active_threads += 1;
                                                            let path_clone = (*path).clone();
                                                            let tx_clone = self.thumbnail_tx.clone();
                                                            let ctx_clone = ui.ctx().clone();
                                                            std::thread::spawn(move || {
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
                                                    
                                                    ui.add_space(6.0);
                                                    
                                                    let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
                                                    let label_text = egui::RichText::new(filename)
                                                        .size(11.0)
                                                        .line_height(Some(13.0));
                                                    ui.add(egui::Label::new(label_text).truncate());
                                                });
                                            });
                                            
                                        if response.double_clicked() {
                                            clicked_path = Some((*path).clone());
                                        } else if response.clicked() {
                                            single_clicked_path = Some((*path).clone());
                                        }
                                    }
                                });
                            }
                        });
                    }
                    
                    if let Some(path) = clicked_path {
                        self.images = self.recursive_images.clone();
                        self.current_index = self.images.iter().position(|p| p == &path).unwrap_or(0);
                        self.show_grid = false;
                        self.back_target_is_gallery = true;
                        self.zoom = 1.0;
                        self.offset = egui::Vec2::ZERO;
                        self.update_exif();
                        ui.ctx().request_repaint();
                    }
                    
                    if let Some(path) = single_clicked_path {
                        if let Some(pos) = self.recursive_images.iter().position(|p| p == &path) {
                            self.images = self.recursive_images.clone();
                            self.current_index = pos;
                            self.update_exif();
                            self.show_exif = true;
                            self.side_panel_mode = SidePanelMode::Duplicates;
                            ui.ctx().request_repaint();
                        }
                    }
                }
                
                ExploreMode::Semantic => {
                    // Semantic Explorer View
                    if !self.db_loaded {
                        self.start_lazy_db_load(ctx);
                        ui.centered_and_justified(|ui| {
                            ui.vertical_centered(|ui| {
                                ui.add(egui::Spinner::new().size(36.0));
                                ui.add_space(16.0);
                                ui.heading("Lazy-loading AI Database Models & ONNX session...");
                                ui.weak("Initializing standard text encoders and reading index maps. This happens fully in the background.");
                            });
                        });
                    } else {
                        // DB Loaded! Render AI Search Bar Controls
                        ui.horizontal(|ui| {
                            ui.heading("🔍 AI Explorer");
                            ui.add_space(16.0);
                            
                            // Mode switches
                            ui.selectable_value(&mut self.semantic_mode, SearchMode::Clip, "🔍 AI Search");
                            ui.selectable_value(&mut self.semantic_mode, SearchMode::Ocr, "📝 OCR Search");
                            ui.add_space(8.0);
                            
                            // Query input text box
                            let hint = match self.semantic_mode {
                                SearchMode::Clip => "Describe the photo (e.g., 'a cat laying on a keyboard')",
                                SearchMode::Ocr => "Type word/text found inside the image",
                            };
                            let search_resp = ui.add(egui::TextEdit::singleline(&mut self.semantic_query)
                                .hint_text(hint)
                                .desired_width(320.0));
                            let enter_pressed = search_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                            
                            ui.add_space(8.0);
                            ui.add(egui::Slider::new(&mut self.semantic_limit, 1..=500).text("Limit"));
                            ui.checkbox(&mut self.semantic_video_only, "Videos only");
                            
                            if ui.button("🔍 Search").clicked() || enter_pressed {
                                self.search_now();
                            }
                        });
                        ui.add_space(6.0);
                        
                        // Status line with a nice look
                        ui.horizontal(|ui| {
                            ui.weak(&self.semantic_status);
                        });
                        ui.add_space(6.0);
                        ui.separator();
                        ui.add_space(8.0);
                        
                        if self.semantic_results.is_empty() {
                            ui.centered_and_justified(|ui| {
                                ui.weak("No results found. Type a description and click Search!");
                            });
                        } else {
                            let available_width = ui.available_width() - 16.0;
                            let col_width = 130.0 + 12.0;
                            let cols = (available_width / col_width).floor().max(1.0) as usize;
                            
                            let mut clicked_result_idx = None;
                            let mut single_clicked_result_idx = None;
                            let mut clicked_similar = None;
                            let mut clicked_person = None;
                            
                            // Chunk row results
                            let rows: Vec<&[SearchResult]> = self.semantic_results.chunks(cols).collect();
                            let row_height = 160.0 + 12.0;
                            let num_rows = rows.len();
                            
                            egui::ScrollArea::vertical().id_salt("semantic_gallery_scroll").show_rows(ui, row_height, num_rows, |ui, row_range| {
                                ui.spacing_mut().item_spacing = egui::vec2(12.0, 12.0);
                                for row_idx in row_range {
                                    let row_items = rows[row_idx];
                                    ui.horizontal(|ui| {
                                        for item in row_items {
                                            let Some(path) = &item.media_path else {
                                                continue;
                                            };
                                            
                                            let is_current = if let Some(curr_p) = self.images.get(self.current_index) {
                                                curr_p == path
                                            } else {
                                                false
                                            };
                                            
                                            let (rect, response) = ui.allocate_at_least(egui::vec2(130.0, 160.0), egui::Sense::click());
                                            response.context_menu(|ui| {
                                                if ui.button("📂 Open parent folder").clicked() {
                                                    if let Some(parent) = path.parent() {
                                                        let _ = std::process::Command::new("xdg-open")
                                                            .arg(parent)
                                                            .spawn();
                                                    }
                                                    ui.close();
                                                }
                                                if ui.button("📋 Copy image").clicked() {
                                                    let path_clone = path.clone();
                                                    let ctx = ui.ctx().clone();
                                                    std::thread::spawn(move || {
                                                        if let Ok(img) = image::open(&path_clone) {
                                                            let rgba = img.to_rgba8();
                                                            let (width, height) = rgba.dimensions();
                                                            let color = egui::ColorImage::from_rgba_unmultiplied(
                                                                [width as usize, height as usize],
                                                                rgba.as_raw(),
                                                            );
                                                            ctx.copy_image(color);
                                                        }
                                                    });
                                                    ui.close();
                                                }
                                                if ui.button("📋 Copy full path").clicked() {
                                                    ui.ctx().copy_text(path.to_string_lossy().to_string());
                                                    ui.close();
                                                }
                                                if item.is_video {
                                                    if ui.button("🎬 Open in mpv").clicked() {
                                                        let _ = std::process::Command::new("mpv")
                                                            .arg(format!("--start={:.3}", item.timestamp_sec.max(0.0)))
                                                            .arg(path)
                                                            .spawn();
                                                        ui.close();
                                                    }
                                                }
                                                ui.separator();
                                                if ui.button("🔍 Show most similar").clicked() {
                                                    clicked_similar = Some(item.clone());
                                                    ui.close();
                                                }
                                                if ui.button("👥 Show more of this person").clicked() {
                                                    clicked_person = Some(item.file_name.clone());
                                                    ui.close();
                                                }
                                            });
                                            let is_hovered = response.hovered();
                                            let is_clicked = response.clicked();
                                            
                                            let card_bg = if is_clicked {
                                                ui.visuals().selection.bg_fill.gamma_multiply(0.3)
                                            } else if is_hovered {
                                                ui.visuals().code_bg_color.gamma_multiply(1.5)
                                            } else if is_current {
                                                ui.visuals().selection.bg_fill.gamma_multiply(0.15)
                                            } else {
                                                ui.visuals().code_bg_color
                                            };
                                            
                                            let card_stroke = if is_current {
                                                egui::Stroke::new(2.0, ui.visuals().selection.bg_fill)
                                            } else if is_hovered {
                                                egui::Stroke::new(1.0, ui.visuals().selection.bg_fill.gamma_multiply(0.5))
                                            } else {
                                                egui::Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color.gamma_multiply(0.3))
                                            };
                                            
                                            let builder = egui::UiBuilder::new()
                                                .max_rect(rect)
                                                .layout(egui::Layout::top_down(egui::Align::Center))
                                                .id_salt(path);
                                            let mut child_ui = ui.new_child(builder);
                                            egui::Frame::NONE
                                                .fill(card_bg)
                                                .stroke(card_stroke)
                                                .inner_margin(8.0)
                                                .corner_radius(6.0)
                                                .show(&mut child_ui, |ui| {
                                                    ui.vertical_centered(|ui| {
                                                        if let Some(texture) = self.thumbnail_textures.get(path) {
                                                            ui.add(
                                                                egui::Image::from_texture(texture)
                                                                    .max_size(egui::vec2(110.0, 110.0))
                                                                    .maintain_aspect_ratio(true)
                                                            );
                                                        } else if self.thumbnail_failed.contains(path) {
                                                            ui.vertical_centered(|ui| {
                                                                ui.add_space(40.0);
                                                                ui.weak("⚠️ Failed");
                                                                ui.add_space(40.0);
                                                            });
                                                        } else {
                                                            ui.vertical_centered(|ui| {
                                                                ui.add_space(45.0);
                                                                ui.add(egui::Spinner::new().size(20.0));
                                                                ui.add_space(45.0);
                                                            });
                                                            
                                                            if !self.thumbnail_loading.contains(path) && self.thumbnail_active_threads < 4 {
                                                                self.thumbnail_loading.insert(path.clone());
                                                                self.thumbnail_active_threads += 1;
                                                                let path_clone = path.clone();
                                                                let tx_clone = self.thumbnail_tx.clone();
                                                                let ctx_clone = ui.ctx().clone();
                                                                std::thread::spawn(move || {
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
                                                        
                                                        ui.add_space(4.0);
                                                        
                                                        // Draw match score or OCR badge
                                                        match self.semantic_mode {
                                                            SearchMode::Clip => {
                                                                let pct = (item.score * 100.0).clamp(0.0, 100.0);
                                                                ui.colored_label(
                                                                    egui::Color32::from_rgb(100, 200, 100),
                                                                    format!("{:.0}% Match", pct)
                                                                );
                                                            }
                                                            SearchMode::Ocr => {
                                                                ui.colored_label(
                                                                    egui::Color32::from_rgb(100, 180, 255),
                                                                    "📝 OCR Match"
                                                                );
                                                            }
                                                        }
                                                        
                                                        let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
                                                        let label_text = egui::RichText::new(filename)
                                                            .size(10.0)
                                                            .line_height(Some(12.0));
                                                        ui.add(egui::Label::new(label_text).truncate());
                                                    });
                                                });
                                                
                                            if response.double_clicked() {
                                                if let Some(pos) = self.semantic_results.iter().position(|r| r.media_path.as_ref() == Some(path)) {
                                                    clicked_result_idx = Some(pos);
                                                }
                                            } else if response.clicked() {
                                                if let Some(pos) = self.semantic_results.iter().position(|r| r.media_path.as_ref() == Some(path)) {
                                                    single_clicked_result_idx = Some(pos);
                                                }
                                            }
                                        }
                                    });
                                }
                            });
                            
                            if let Some(idx) = clicked_result_idx {
                                // Transition to single view mode and lock arrow navigations to search pool
                                let active_paths: Vec<PathBuf> = self.semantic_results.iter()
                                    .filter_map(|r| r.media_path.clone())
                                    .collect();
                                self.images = active_paths;
                                self.current_index = idx;
                                self.show_grid = false;
                                self.back_target_is_gallery = true;
                                self.zoom = 1.0;
                                self.offset = egui::Vec2::ZERO;
                                self.update_exif();
                                ui.ctx().request_repaint();
                            }
                            if let Some(idx) = single_clicked_result_idx {
                                let active_paths: Vec<PathBuf> = self.semantic_results.iter()
                                    .filter_map(|r| r.media_path.clone())
                                    .collect();
                                self.images = active_paths;
                                self.current_index = idx;
                                self.update_exif();
                                self.show_exif = true;
                                self.side_panel_mode = SidePanelMode::Duplicates;
                                ui.ctx().request_repaint();
                            }
                            if let Some(item) = clicked_similar {
                                self.show_most_similar_clip(&item);
                                ui.ctx().request_repaint();
                            }
                            if let Some(name) = clicked_person {
                                self.show_more_of_this_person(&name);
                                ui.ctx().request_repaint();
                            }
                        }
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

    fn show_most_similar_clip(&mut self, row: &SearchResult) {
        let Some(indices) = &self.db_indices else {
            return;
        };
        let Some(query_vector) = self.clip_vector_for_result(row) else {
            self.semantic_status = format!("no CLIP vector found for {}", row.file_name);
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
        let mut results = search_index(&indices.clip_index, &query_vector, pre_limit, false);
        results.retain(|candidate| candidate.file_name != row.file_name);
        results = collapse_sift_grouped_results(results, &indices.sift_root_by_file, self.semantic_limit);
        
        let db_roots = get_db_roots();
        let db_dir = Path::new("/media/lewis/1b/lancedb");
        for candidate in &mut results {
            candidate.media_path =
                resolve_media_path(&db_roots, db_dir, &candidate.file_name, candidate.timestamp_sec).ok();
        }
        let took = started.elapsed().as_millis();
        self.semantic_status = format!(
            "✓ Found {} CLIP-similar results in {} ms for {}",
            results.len(),
            took,
            row.file_name
        );
        self.semantic_results = results;
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

    fn show_more_of_this_person(&mut self, file_name: &str) {
        let Some(indices) = &self.db_indices else {
            return;
        };
        let related_files = Self::related_files_for_face_seed(indices, file_name);
        let mut query_faces = Vec::new();
        for related in &related_files {
            query_faces.extend(Self::face_vectors_for_file(indices, related));
        }
        if query_faces.is_empty() {
            self.semantic_status = format!(
                "No stored face vectors for {file_name} or {} related file(s)",
                related_files.len().saturating_sub(1)
            );
            self.semantic_results = Vec::new();
            return;
        }
        let started = Instant::now();
        let mut results =
            search_face_index(&indices.face_index, &query_faces, 500, FACE_MATCH_MIN_SCORE);
        results = collapse_sift_grouped_results(results, &indices.sift_root_by_file, 500);
        
        let db_roots = get_db_roots();
        let db_dir = Path::new("/media/lewis/1b/lancedb");
        for row in &mut results {
            row.media_path =
                resolve_media_path(&db_roots, db_dir, &row.file_name, row.timestamp_sec).ok();
        }
        let took = started.elapsed().as_millis();
        self.semantic_status = format!(
            "✓ Found {} person results in {} ms using {} query face vector(s)",
            results.len(),
            took,
            query_faces.len()
        );
        self.semantic_results = results;
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
                self.update_exif();
                ctx.request_repaint();
            }
        }
        // Mouse Back click handling
        if !self.show_grid && self.back_target_is_gallery {
            let back_clicked = ctx.input(|i| i.pointer.button_clicked(egui::PointerButton::Extra1));
            if back_clicked {
                self.show_grid = true;
                ctx.request_repaint();
            }
        }

        // Keyboard handling
        if !ctx.wants_keyboard_input() {
            ctx.input(|i| {
                if !self.show_grid {
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
                    let show_layout_active = self.show_exif && self.side_panel_mode == SidePanelMode::Layout;
                    if show_layout_active {
                        self.show_exif = false;
                    } else {
                        self.show_exif = true;
                        self.side_panel_mode = SidePanelMode::Layout;
                    }
                }
                if i.key_pressed(egui::Key::Q) {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
                if i.key_pressed(egui::Key::Escape) {
                    if self.show_grid {
                        self.show_grid = false;
                    } else {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                }
            });
        }

        // Top bar
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Iris");
                ui.separator();
                if let Some(path) = self.images.get(self.current_index) {
                    let filename = path.file_name().and_then(|f| f.to_str()).unwrap_or("");
                    ui.label(format!("{} ({}/{}) - {} - {}", filename, self.current_index + 1, self.images.len(), self.current_dimensions, self.current_file_size));
                } else {
                    ui.label("No image loaded");
                }
                
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // EXIF Button
                    let show_exif_active = self.show_exif && self.side_panel_mode == SidePanelMode::Exif;
                    let exif_button_text = if show_exif_active { "🏷 Hide EXIF" } else { "🏷 Show EXIF" };
                    if ui.button(exif_button_text).clicked() {
                        if show_exif_active {
                            self.show_exif = false;
                        } else {
                            self.show_exif = true;
                            self.side_panel_mode = SidePanelMode::Exif;
                        }
                    }

                    ui.add_space(8.0);

                    // Layout Button
                    let show_layout_active = self.show_exif && self.side_panel_mode == SidePanelMode::Layout;
                    let layout_button_text = if show_layout_active { "📂 Hide Layout [E]" } else { "📂 Show Layout [E]" };
                    if ui.button(layout_button_text).clicked() {
                        if show_layout_active {
                            self.show_exif = false;
                        } else {
                            self.show_exif = true;
                            self.side_panel_mode = SidePanelMode::Layout;
                        }
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

        // Collapsible EXIF Side Panel (Shows Binary File Layout Diagram or Raw EXIF)
        egui::SidePanel::right("exif_panel")
            .resizable(true)
            .default_width(400.0)
            .show_animated(ctx, self.show_exif, |ui| {
                // Header Tabs
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.side_panel_mode, SidePanelMode::Layout, "📂 Binary Layout");
                    ui.selectable_value(&mut self.side_panel_mode, SidePanelMode::Exif, "🏷 Raw EXIF");
                    ui.selectable_value(&mut self.side_panel_mode, SidePanelMode::Duplicates, "👥 Duplicates");
                    
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("❌").clicked() {
                            self.show_exif = false;
                        }
                    });
                });
                ui.separator();
                ui.add_space(4.0);

                match self.side_panel_mode {
                    SidePanelMode::Layout => {
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
                        if !self.db_loaded {
                            self.start_lazy_db_load(ui.ctx());
                            ui.vertical_centered(|ui| {
                                ui.add_space(20.0);
                                ui.add(egui::Spinner::new().size(24.0));
                                ui.add_space(12.0);
                                ui.weak("Loading database index to scan duplicates...");
                            });
                        } else if let Some(path) = self.images.get(self.current_index).cloned() {
                            let filename_opt = self.get_db_filename_from_path(&path);
                            if let Some(filename) = filename_opt {
                                let indices = self.db_indices.as_ref().unwrap();
                                
                                // Resolve master file if this is a pHash duplicate/similar
                                let master_file_name = indices.phash_master_by_file
                                    .get(&filename)
                                    .cloned()
                                    .unwrap_or_else(|| filename.clone());
                                
                                // Find SIFT master
                                let sift_master = indices.sift_root_by_file
                                    .get(&master_file_name)
                                    .cloned()
                                    .unwrap_or_else(|| master_file_name.clone());
                                
                                // Fetch SIFT members in this group
                                let sift_members = indices.sift_members_by_root
                                    .get(&sift_master)
                                    .cloned()
                                    .unwrap_or_default();
                                
                                let sift_evidence_members: Vec<String> = sift_members
                                    .iter()
                                    .filter(|member| {
                                        indices.sift_info_by_file
                                            .get(member.as_str())
                                            .is_some_and(valid_sift_link)
                                    })
                                    .cloned()
                                    .collect();
                                
                                let mut displayed_sift_members = Vec::new();
                                let mut displayed_seen = HashSet::new();
                                if displayed_seen.insert(filename.clone()) {
                                    displayed_sift_members.push(filename.clone());
                                }
                                for member in &sift_evidence_members {
                                    if displayed_seen.insert(member.clone()) {
                                        displayed_sift_members.push(member.clone());
                                    }
                                }
                                
                                // Combine pHash similars for all SIFT members in this group
                                let member_sources: Vec<String> = if sift_members.is_empty() {
                                    vec![master_file_name.clone()]
                                } else {
                                    sift_members.clone()
                                };
                                
                                let mut combined_similars: Vec<SimilarFile> = Vec::new();
                                let mut combined_seen = HashSet::new();
                                for member in &member_sources {
                                    let mut items = indices.similar_by_master
                                        .get(member.as_str())
                                        .cloned()
                                        .unwrap_or_default();
                                    items.sort_by(|a, b| {
                                        b.similarity_pct
                                            .unwrap_or(f32::NEG_INFINITY)
                                            .partial_cmp(&a.similarity_pct.unwrap_or(f32::NEG_INFINITY))
                                            .unwrap_or(Ordering::Equal)
                                    });
                                    for item in &items {
                                        if combined_seen.insert(item.file_name.clone()) {
                                            combined_similars.push(item.clone());
                                        }
                                    }
                                }
                                combined_similars.sort_by(|a, b| {
                                    b.similarity_pct
                                        .unwrap_or(f32::NEG_INFINITY)
                                        .partial_cmp(&a.similarity_pct.unwrap_or(f32::NEG_INFINITY))
                                        .unwrap_or(Ordering::Equal)
                                });
                                
                                // Precompute SIFT members metadata (paths and text) to avoid borrow checker errors in the ScrollArea closure
                                let mut displayed_sift_metadata = Vec::new();
                                let roots = get_db_roots();
                                for member in &displayed_sift_members {
                                    let source_path_opt = resolve_source_path(&roots, member).ok();
                                    let member_is_video = is_video_path(Path::new(member));
                                    let res_size_str = source_path_opt.as_ref()
                                        .map(|p| file_resolution_and_size(p))
                                        .unwrap_or_else(|| "n/a".to_string());
                                    let sift_str = sift_info_line(&indices.sift_info_by_file, member);
                                    displayed_sift_metadata.push((member.clone(), source_path_opt, member_is_video, res_size_str, sift_str));
                                }

                                // Precompute combined similars metadata (paths and text)
                                let mut combined_similars_metadata = Vec::new();
                                for item in &combined_similars {
                                    let source_path_opt = resolve_source_path(&roots, &item.file_name).ok();
                                    let res_size_str = source_path_opt.as_ref()
                                        .map(|p| file_resolution_and_size(p))
                                        .unwrap_or_else(|| "n/a".to_string());
                                    combined_similars_metadata.push((item.clone(), source_path_opt, res_size_str));
                                }
                                
                                ui.heading("👥 Duplicate Matches");
                                ui.add_space(4.0);
                                ui.weak(format!("Current: {}", filename));
                                ui.add_space(8.0);
                                
                                egui::ScrollArea::vertical().show(ui, |ui| {
                                    let side_thumb = 90.0_f32;
                                    
                                    // 1. SIFT Cluster Members (Duplicates)
                                    if !sift_members.is_empty() {
                                        ui.horizontal(|ui| {
                                            ui.colored_label(egui::Color32::from_rgb(100, 200, 100), "✓ SIFT Duplicate Cluster");
                                            ui.weak(format!("({} files)", sift_members.len()));
                                        });
                                        ui.add_space(6.0);
                                        
                                        for (member, source_path_opt, member_is_video, res_size_str, sift_str) in &displayed_sift_metadata {
                                            ui.horizontal(|ui| {
                                                // Left: Thumbnail preview
                                                if let Some(s_path) = source_path_opt.as_ref() {
                                                    self.draw_thumbnail_async(ui, s_path, side_thumb);
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
                                                        if member == &filename {
                                                            ui.colored_label(egui::Color32::from_rgb(255, 180, 50), "• Active");
                                                        }
                                                    });
                                                    
                                                    ui.weak(res_size_str);
                                                    ui.weak(sift_str);
                                                    
                                                    let display_name = member.split_once('/').map(|x| x.1).unwrap_or(member);
                                                    ui.monospace(display_name);
                                                    
                                                    ui.horizontal(|ui| {
                                                        if let Some(s_path) = source_path_opt.as_ref() {
                                                            if member != &filename {
                                                                if ui.button("👁 View").clicked() {
                                                                    if let Some(pos) = self.images.iter().position(|p| p == s_path) {
                                                                        self.current_index = pos;
                                                                    } else {
                                                                        self.images.insert(self.current_index + 1, s_path.clone());
                                                                        self.current_index += 1;
                                                                    }
                                                                    self.show_grid = false;
                                                                    self.update_exif();
                                                                }
                                                            }
                                                            
                                                            let is_active_compare = self.compare_target.as_ref() == Some(s_path);
                                                            let btn_label = if is_active_compare { "🎯 Comparing" } else { "⚖ Compare" };
                                                            if ui.selectable_label(is_active_compare, btn_label).clicked() {
                                                                if is_active_compare {
                                                                    self.compare_target = None;
                                                                    self.sift_pair_overlay = None;
                                                                } else {
                                                                    self.compare_target = Some(s_path.clone());
                                                                    self.start_sift_alignment(path.clone(), s_path.clone(), ui.ctx().clone());
                                                                }
                                                            }
                                                        }
                                                    });
                                                });
                                            });
                                            ui.add_space(8.0);
                                            ui.separator();
                                            ui.add_space(8.0);
                                        }
                                        ui.add_space(8.0);
                                    }
                                    
                                    // 2. pHash Similars (Similarity Map)
                                    if !combined_similars.is_empty() {
                                        ui.horizontal(|ui| {
                                            ui.colored_label(egui::Color32::from_rgb(100, 180, 255), "🔗 Similar Images (pHash)");
                                            ui.weak(format!("({} files)", combined_similars.len()));
                                        });
                                        ui.add_space(6.0);
                                        
                                        for (item, source_path_opt, res_size_str) in &combined_similars_metadata {
                                            let item_is_video = item.is_video;
                                            
                                            ui.horizontal(|ui| {
                                                // Left: Thumbnail preview
                                                if let Some(s_path) = source_path_opt.as_ref() {
                                                    self.draw_thumbnail_async(ui, s_path, side_thumb);
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
                                                        if item.file_name == filename {
                                                            ui.colored_label(egui::Color32::from_rgb(255, 180, 50), "• Active");
                                                        }
                                                    });
                                                    
                                                    let similarity_label = item.similarity_pct
                                                        .map(|v| format!("pHash similarity {:.2}%", v))
                                                        .unwrap_or_else(|| "pHash similarity n/a".to_string());
                                                    ui.colored_label(egui::Color32::from_rgb(100, 180, 255), similarity_label);
                                                    
                                                    ui.weak(res_size_str);
                                                    
                                                    let display_name = item.file_name.split_once('/').map(|x| x.1).unwrap_or(&item.file_name);
                                                    ui.monospace(display_name);
                                                    
                                                    ui.horizontal(|ui| {
                                                        if let Some(s_path) = source_path_opt.as_ref() {
                                                            if item.file_name != filename {
                                                                if ui.button("👁 View").clicked() {
                                                                    if let Some(pos) = self.images.iter().position(|p| p == s_path) {
                                                                        self.current_index = pos;
                                                                    } else {
                                                                        self.images.insert(self.current_index + 1, s_path.clone());
                                                                        self.current_index += 1;
                                                                    }
                                                                    self.show_grid = false;
                                                                    self.update_exif();
                                                                }
                                                            }
                                                            
                                                            let is_active_compare = self.compare_target.as_ref() == Some(s_path);
                                                            let btn_label = if is_active_compare { "🎯 Comparing" } else { "⚖ Compare" };
                                                            if ui.selectable_label(is_active_compare, btn_label).clicked() {
                                                                if is_active_compare {
                                                                    self.compare_target = None;
                                                                    self.sift_pair_overlay = None;
                                                                } else {
                                                                    self.compare_target = Some(s_path.clone());
                                                                    self.start_sift_alignment(path.clone(), s_path.clone(), ui.ctx().clone());
                                                                }
                                                            }
                                                        }
                                                    });
                                                });
                                            });
                                            ui.add_space(8.0);
                                            ui.separator();
                                            ui.add_space(8.0);
                                        }
                                    } else if sift_members.is_empty() {
                                        ui.weak("No duplicates or similar files found in database.");
                                    }
                                });
                            } else {
                                ui.weak("Current file is not indexed in the database (not in Phone or Telegram Backup folders).");
                            }
                        } else {
                            ui.weak("No image loaded.");
                        }
                    }
                }
            });

        let mut panel = egui::CentralPanel::default();
        if let Some(bg) = self.viewport_bg {
            panel = panel.frame(egui::Frame::NONE.fill(bg));
        }
        panel.show(ctx, |ui| {
            if self.show_grid {
                self.show_grid_view(ui, ctx);
            } else {
                if let Some(path) = self.images.get(self.current_index) {
                let uri = format!("file://{}", path.to_string_lossy());
                
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
                if scroll_delta != 0.0 {
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
                    if ui.button("📋 Copy Image Path").clicked() {
                        ui.ctx().copy_text(path.to_string_lossy().to_string());
                        ui.close();
                    }
                    if ui.button("🖼 Copy Image").clicked() {
                        if let Ok(img) = image::open(path) {
                            let size = [img.width() as usize, img.height() as usize];
                            let img_rgba = img.to_rgba8();
                            let color_image = egui::ColorImage::from_rgba_unmultiplied(
                                size,
                                img_rgba.as_raw(),
                            );
                            ui.ctx().copy_image(color_image);
                        }
                        ui.close();
                    }
                    if ui.button("🔍 Fit Image / Recenter").clicked() {
                        self.zoom = 1.0;
                        self.offset = egui::Vec2::ZERO;
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
                    let left_uri = format!("file://{}", path.to_string_lossy());
                    let right_uri = format!("file://{}", compare_path.to_string_lossy());
                    
                    let avail_size = ui.available_size();
                    let half_w = (avail_size.x / 2.0 - 12.0).max(10.0);
                    let h = (avail_size.y - 120.0).max(10.0); // leave space for SIFT overlay
                    
                    ui.horizontal(|ui| {
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
                    
                    ui.add_space(16.0);
                    ui.vertical_centered(|ui| {
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
