use super::*;

fn get_socket_path() -> PathBuf {
    let username = std::env::var("USER")
        .unwrap_or_else(|_| std::env::var("USERNAME").unwrap_or_else(|_| "default".to_string()));
    std::env::temp_dir().join(format!("iris_{}.sock", username))
}

pub(crate) fn initial_window_size(path: &Path, start_on_home_page: bool) -> [f32; 2] {
    if !start_on_home_page && path.is_file() && !is_video_path(path) {
        if let Ok((width, height)) = image::image_dimensions(path) {
            if width > 0 && height > 0 {
                let scale = INITIAL_IMAGE_DISPLAY_HEIGHT / height as f32;
                return [
                    width as f32 * scale,
                    INITIAL_IMAGE_DISPLAY_HEIGHT + IMAGE_VIEWER_TOP_BAR_HEIGHT,
                ];
            }
        }
    }
    [1200.0, 800.0]
}

pub(crate) fn run() -> eframe::Result {
    let args: Vec<String> = std::env::args().collect();
    let mut reuse_window = false;
    let mut no_daemon = false;
    let mut image_args = Vec::new();

    for arg in args.iter().skip(1) {
        if arg == "--same-window" || arg == "-s" || arg == "--reuse-window" || arg == "-r" {
            reuse_window = true;
        } else if arg == "--new-window" || arg == "-n" {
            // New window is now the default behavior, so this flag is a no-op
        } else if arg == "--no-daemon" {
            no_daemon = true;
        } else {
            image_args.push(arg.clone());
        }
    }

    if image_args.len() > 6 {
        eprintln!("Iris comparison mode accepts at most six paths.");
        return Ok(());
    }
    let requested_paths: Vec<PathBuf> = image_args
        .into_iter()
        .map(|path| {
            let path = PathBuf::from(path);
            path.canonicalize().unwrap_or(path)
        })
        .collect();
    let start_on_home_page = requested_paths.is_empty();
    let comparison_paths = (requested_paths.len() >= 2).then(|| requested_paths.clone());
    let image_path = requested_paths
        .first()
        .cloned()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let request_payload = || {
        comparison_paths
            .as_ref()
            .map(|paths| {
                serde_json::to_string(
                    &paths
                        .iter()
                        .map(|path| path.to_string_lossy().to_string())
                        .collect::<Vec<_>>(),
                )
                .expect("comparison paths should be serializable")
            })
            .unwrap_or_else(|| image_path.to_string_lossy().to_string())
    };

    let socket_path = get_socket_path();
    let mut socket_active = false;

    // Check if another instance is already actively listening on the socket
    if let Ok(mut stream) = std::os::unix::net::UnixStream::connect(&socket_path) {
        socket_active = true;
        if reuse_window {
            use std::io::Write;
            let payload = request_payload();
            if let Err(e) = stream.write_all(payload.as_bytes()) {
                eprintln!("Error sending path to existing instance: {}", e);
            } else {
                println!("Opened requested path(s) in the existing window.");
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
    let (tx, rx) = std::sync::mpsc::channel::<OpenRequest>();
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
                                    let request = serde_json::from_str::<Vec<String>>(&path_str)
                                        .map(|paths| {
                                            OpenRequest::Comparison(
                                                paths.into_iter().map(PathBuf::from).collect(),
                                            )
                                        })
                                        .unwrap_or_else(|_| {
                                            OpenRequest::Single(PathBuf::from(path_str))
                                        });
                                    let _ = tx_clone.send(request);
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

    let window_size = initial_window_size(&image_path, start_on_home_page);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_app_id("iris")
            .with_inner_size(window_size)
            .with_title("Iris"),
        ..Default::default()
    };

    let rx_shared = Arc::new(Mutex::new(Some(rx)));
    let rx_shared_clone = rx_shared.clone();
    let image_path_clone = image_path.clone();
    let comparison_paths_clone = comparison_paths.clone();
    let ctx_shared_clone = ctx_shared.clone();

    let mut result = eframe::run_native(
        "iris",
        options.clone(),
        Box::new(move |cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            let rx_taken = rx_shared_clone.lock().unwrap().take().unwrap();
            Ok(Box::new(ImageViewer::new(
                image_path_clone,
                rx_taken,
                ctx_shared_clone,
                start_on_home_page,
                comparison_paths_clone,
            )))
        }),
    );

    if result.is_err() {
        // Fallback to X11 backend if Wayland graphics context fails (e.g., NVIDIA EGL OutOfMemory)
        unsafe {
            std::env::set_var("WINIT_UNIX_BACKEND", "x11");
        }
        result = eframe::run_native(
            "iris",
            options,
            Box::new(move |cc| {
                egui_extras::install_image_loaders(&cc.egui_ctx);
                let rx_taken = rx_shared.lock().unwrap().take().unwrap();
                Ok(Box::new(ImageViewer::new(
                    image_path,
                    rx_taken,
                    ctx_shared,
                    start_on_home_page,
                    comparison_paths,
                )))
            }),
        );
    }

    if bind_socket {
        let _ = std::fs::remove_file(&socket_path);
    }

    result
}
