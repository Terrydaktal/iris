use super::*;

pub(crate) fn clipboard_image_dir() -> PathBuf {
    std::env::temp_dir().join("iris-clipboard")
}

pub(crate) fn is_clipboard_image_path(path: &Path) -> bool {
    path.starts_with(clipboard_image_dir())
}

pub(crate) fn percent_decode_file_uri_path(raw: &str) -> String {
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

pub(crate) fn image_path_from_pasted_text(text: &str) -> Option<PathBuf> {
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

pub(crate) fn clipboard_paste_signal(ui: &egui::Ui) -> (bool, Option<String>) {
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

pub(crate) fn command_available(name: &str) -> bool {
    let Some(paths) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&paths).any(|dir| dir.join(name).is_file())
}

pub(crate) fn clipboard_command_output(program: &str, args: &[&str]) -> Result<Option<Vec<u8>>> {
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

pub(crate) fn save_clipboard_bytes_to_temp(bytes: &[u8], ext: &str) -> Result<Option<PathBuf>> {
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

pub(crate) fn wl_paste_clipboard_image_to_temp() -> Result<Option<PathBuf>> {
    if !command_available("wl-paste") {
        return Ok(None);
    }

    let type_list = clipboard_command_output("wl-paste", &["--list-types"])?
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .unwrap_or_default();
    let types: Vec<&str> = type_list
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();

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
        if let Some(bytes) =
            clipboard_command_output("wl-paste", &["--no-newline", "--type", mime])?
        {
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

pub(crate) fn save_clipboard_image_to_temp(pasted_text: Option<&str>) -> Result<Option<PathBuf>> {
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

pub(crate) fn copy_image_file_to_clipboard(path: &Path) -> Result<()> {
    let img = image::open(path).with_context(|| {
        format!(
            "failed to open image for clipboard copy: {}",
            path.display()
        )
    })?;
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
            let stdin = child
                .stdin
                .as_mut()
                .context("wl-copy stdin is unavailable")?;
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
