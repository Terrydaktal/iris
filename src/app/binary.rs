use eframe::egui;

#[derive(Clone)]
pub(crate) struct FileChunk {
    pub(crate) name: String,
    pub(crate) offset: usize,
    pub(crate) length: usize,
    pub(crate) description: String,
    pub(crate) color: egui::Color32,
    pub(crate) parsed_data: String,
}

pub(crate) fn generate_hex_dump(
    chunk_bytes: &[u8],
    absolute_offset: usize,
    max_len: usize,
) -> String {
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

        dump.push_str(&format!(
            "0x{:04X}:   {} |{}|\n",
            absolute_offset + line_offset,
            hex_part,
            ascii_part
        ));
        line_offset += 16;
    }
    if chunk_bytes.len() > max_len {
        dump.push_str("... (truncated) ...\n");
    }
    dump
}

pub(crate) fn parse_png(bytes: &[u8]) -> Option<Vec<FileChunk>> {
    if bytes.len() < 8 || &bytes[0..8] != [137, 80, 78, 71, 13, 10, 26, 10] {
        return None;
    }

    let mut chunks = vec![FileChunk {
        name: "PNG Signature".to_string(),
        offset: 0,
        length: 8,
        description: "8-byte magic number identifying the file as a PNG image.".to_string(),
        color: egui::Color32::from_rgb(120, 110, 255), // Indigo
        parsed_data: generate_hex_dump(&bytes[0..8], 0, 1024),
    }];

    let mut pos = 8;
    while pos + 8 <= bytes.len() {
        let len = u32::from_be_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
            as usize;
        let type_bytes = &bytes[pos + 4..pos + 8];
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
            _ => egui::Color32::from_rgb(200, 140, 255),      // Lavender
        };

        let parsed_data = match chunk_type.as_str() {
            "IHDR" if len >= 13 && pos + 21 <= bytes.len() => {
                let w = u32::from_be_bytes([
                    bytes[pos + 8],
                    bytes[pos + 9],
                    bytes[pos + 10],
                    bytes[pos + 11],
                ]);
                let h = u32::from_be_bytes([
                    bytes[pos + 12],
                    bytes[pos + 13],
                    bytes[pos + 14],
                    bytes[pos + 15],
                ]);
                let depth = bytes[pos + 16];
                let color = bytes[pos + 17];
                let comp = bytes[pos + 18];
                let filter = bytes[pos + 19];
                let interlace = bytes[pos + 20];
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
                    let hex_part = if i < hex_lines.len() {
                        hex_lines[i]
                    } else {
                        ""
                    };
                    let english_part = if i < english_lines.len() {
                        &english_lines[i]
                    } else {
                        ""
                    };

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
                let chunk_data = &bytes[pos + 8..pos + 8 + len];
                if let Some(null_pos) = chunk_data.iter().position(|&b| b == 0) {
                    let key = String::from_utf8_lossy(&chunk_data[0..null_pos]).to_string();
                    let val = String::from_utf8_lossy(&chunk_data[null_pos + 1..]).to_string();
                    format!("{}: {}", key, val)
                } else {
                    String::from_utf8_lossy(chunk_data).to_string()
                }
            }
            "sRGB" if len >= 1 && pos + 9 <= bytes.len() => {
                let intent = bytes[pos + 8];
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
                let x = u32::from_be_bytes([
                    bytes[pos + 8],
                    bytes[pos + 9],
                    bytes[pos + 10],
                    bytes[pos + 11],
                ]);
                let y = u32::from_be_bytes([
                    bytes[pos + 12],
                    bytes[pos + 13],
                    bytes[pos + 14],
                    bytes[pos + 15],
                ]);
                let unit = bytes[pos + 16];
                let unit_str = if unit == 1 { "meter" } else { "unknown" };
                format!(
                    "Pixels per unit X: {}\nPixels per unit Y: {}\nUnit: {} ({})",
                    x, y, unit, unit_str
                )
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

pub(crate) fn parse_jpeg(bytes: &[u8]) -> Option<Vec<FileChunk>> {
    if bytes.len() < 4 || &bytes[0..2] != [0xFF, 0xD8] {
        return None;
    }

    let mut chunks = vec![FileChunk {
        name: "SOI Marker".to_string(),
        offset: 0,
        length: 2,
        description: "Start of Image: Identifies the beginning of the JPEG stream.".to_string(),
        color: egui::Color32::from_rgb(120, 110, 255), // Indigo
        parsed_data: generate_hex_dump(&bytes[0..2], 0, 1024),
    }];

    let mut pos = 2;
    while pos + 2 <= bytes.len() {
        if bytes[pos] != 0xFF {
            let mut next_ff = pos;
            while next_ff + 1 < bytes.len() {
                if bytes[next_ff] == 0xFF
                    && bytes[next_ff + 1] != 0x00
                    && (bytes[next_ff + 1] < 0xD0 || bytes[next_ff + 1] > 0xD7)
                {
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
                description: "End of Image: Identifies the termination of the JPEG stream."
                    .to_string(),
                color: egui::Color32::from_rgb(180, 180, 180), // Gray
                parsed_data: generate_hex_dump(
                    &bytes[pos..std::cmp::min(bytes.len(), pos + 2)],
                    pos,
                    1024,
                ),
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
                parsed_data: generate_hex_dump(
                    &bytes[pos..std::cmp::min(bytes.len(), pos + 2)],
                    pos,
                    1024,
                ),
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
                let precision = bytes[pos + 4];
                let h = u16::from_be_bytes([bytes[pos + 5], bytes[pos + 6]]);
                let w = u16::from_be_bytes([bytes[pos + 7], bytes[pos + 8]]);
                let components = bytes[pos + 9];

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
                    let hex_part = if i < hex_lines.len() {
                        hex_lines[i]
                    } else {
                        ""
                    };
                    let english_part = if i < english_lines.len() {
                        &english_lines[i]
                    } else {
                        ""
                    };

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
                let comment_bytes = &bytes[pos + 4..pos + 2 + seg_len];
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

pub(crate) fn parse_webp(bytes: &[u8]) -> Option<Vec<FileChunk>> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WEBP" {
        return None;
    }

    let mut chunks = vec![FileChunk {
        name: "RIFF Header".to_string(),
        offset: 0,
        length: 12,
        description: "RIFF Container Header: Identifies the file as a WEBP resource.".to_string(),
        color: egui::Color32::from_rgb(120, 110, 255), // Indigo
        parsed_data: generate_hex_dump(&bytes[0..12], 0, 1024),
    }];

    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let type_bytes = &bytes[pos..pos + 4];
        let chunk_type = String::from_utf8_lossy(type_bytes).to_string();

        let len = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;

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
            _ => egui::Color32::from_rgb(200, 140, 255),      // Lavender
        };

        let parsed_data = match chunk_type.as_str() {
            "VP8X" if len >= 10 && pos + 18 <= bytes.len() => {
                let flags = bytes[pos + 8];
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

pub(crate) fn parse_bmp(bytes: &[u8]) -> Option<Vec<FileChunk>> {
    if bytes.len() < 14 || &bytes[0..2] != b"BM" {
        return None;
    }

    let file_size = u32::from_le_bytes([bytes[2], bytes[3], bytes[4], bytes[5]]) as usize;
    let pixel_array_offset =
        u32::from_le_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]) as usize;

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
        let hex = if i < fh_hex_lines.len() {
            fh_hex_lines[i]
        } else {
            ""
        };
        let eng = if i < fh_english.len() {
            &fh_english[i]
        } else {
            ""
        };
        if !hex.is_empty() || !eng.is_empty() {
            fh_combined.push_str(&format!("{:<80}  # {}\n", hex, eng));
        }
    }

    let mut chunks = vec![FileChunk {
        name: "BMP File Header".to_string(),
        offset: 0,
        length: 14,
        description: file_header_desc,
        color: egui::Color32::from_rgb(120, 110, 255), // Indigo
        parsed_data: fh_combined,
    }];

    if bytes.len() >= 18 {
        let dib_size = u32::from_le_bytes([bytes[14], bytes[15], bytes[16], bytes[17]]) as usize;
        if dib_size <= bytes.len() - 14 {
            let dib_end = 14 + dib_size;
            let mut dib_english = vec![format!("DIB Header Size: {} bytes", dib_size)];

            let mut dib_desc = format!(
                "DIB Header (Size: {}): Specifies the size of the DIB, image dimensions, bit depth, compression, and color details.",
                dib_size
            );

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
                let hex = if i < dib_hex_lines.len() {
                    dib_hex_lines[i]
                } else {
                    ""
                };
                let eng = if i < dib_english.len() {
                    &dib_english[i]
                } else {
                    ""
                };
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

        let pixel_array_hex =
            generate_hex_dump(&bytes[pixel_array_offset..], pixel_array_offset, 1024);

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

pub(crate) fn parse_generic(bytes: &[u8]) -> Vec<FileChunk> {
    let header_len = std::cmp::min(bytes.len(), 1024);
    let payload_offset = header_len;
    let payload_len = if bytes.len() > 2048 {
        bytes.len() - 2048
    } else {
        0
    };
    let trailer_offset = if bytes.len() > 1024 {
        bytes.len() - 1024
    } else {
        0
    };
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
            description: "Main body containing compressed or uncompressed binary image payload."
                .to_string(),
            color: egui::Color32::from_rgb(50, 200, 120),
            parsed_data: generate_hex_dump(
                &bytes[payload_offset..payload_offset + payload_len],
                payload_offset,
                1024,
            ),
        },
        FileChunk {
            name: "File Termination Block".to_string(),
            offset: trailer_offset,
            length: trailer_len,
            description: "End of file structure / trailer payload.".to_string(),
            color: egui::Color32::from_rgb(180, 180, 180),
            parsed_data: generate_hex_dump(
                &bytes[trailer_offset..trailer_offset + trailer_len],
                trailer_offset,
                1024,
            ),
        },
    ]
}
