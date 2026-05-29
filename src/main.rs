use eframe::egui;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

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
        };
        
        viewer.update_exif();
        viewer
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

    fn show_grid_view(&mut self, ui: &mut egui::Ui, _ctx: &egui::Context) {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.heading("🖼 Native Image Gallery (Recursive)");
                ui.add_space(12.0);
                
                // Search box
                ui.label("Filter:");
                ui.add(egui::TextEdit::singleline(&mut self.grid_filter)
                    .hint_text("🔍 Filter by filename...")
                    .desired_width(200.0));
                
                ui.add_space(12.0);
                
                // Status / Spinner
                if self.grid_loading {
                    ui.add(egui::Spinner::new().size(16.0));
                    ui.weak("Scanning subdirectories...");
                } else {
                    ui.weak(format!("{} images found", self.recursive_images.len()));
                }
                
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("❌ Close Gallery [G]").clicked() {
                        self.show_grid = false;
                    }
                    
                    if ui.button("🔄 Refresh").clicked() {
                        self.start_recursive_scan();
                    }
                });
            });
            
            ui.add_space(8.0);
            ui.separator();
            ui.add_space(8.0);
            
            // Filter list
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
                                    
                                if is_clicked {
                                    clicked_path = Some((*path).clone());
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
        });
    }
}

impl eframe::App for ImageViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Ok(mut lock) = self.ctx_shared.lock() {
            if lock.is_none() {
                *lock = Some(ctx.clone());
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

                // Calculate image rect
                let base_size = rect.size();
                let draw_size = base_size * self.zoom;
                let draw_pos = rect.center() + self.offset - draw_size / 2.0;
                let draw_rect = egui::Rect::from_min_size(draw_pos, draw_size);
                
                // Use ui.put to place the image widget
                ui.put(draw_rect, egui::Image::new(uri).maintain_aspect_ratio(true).show_loading_spinner(false));
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
