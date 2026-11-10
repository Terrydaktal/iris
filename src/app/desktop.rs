use std::path::Path;
use std::process::Command;

fn mpv_command(path: &Path, start_seconds: Option<f32>) -> Command {
    let mut command = Command::new("mpv");
    command
        .arg("--force-window=immediate")
        .env_remove("XDG_ACTIVATION_TOKEN")
        .env_remove("DESKTOP_STARTUP_ID");
    if let Some(seconds) = start_seconds {
        command.arg(format!("--start={:.3}", seconds.max(0.0)));
    }
    command.arg("--").arg(path);
    command
}

fn spawn_mpv(command: &mut Command, activation_token: Option<&str>) -> Result<(), String> {
    if let Some(token) = activation_token {
        command.env("XDG_ACTIVATION_TOKEN", token);
    }
    command
        .spawn()
        .map(|_| ())
        .map_err(|error| format!("Unable to open video in mpv: {error}"))
}

#[cfg(test)]
mod playback_tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn mpv_launch_requests_focus_and_preserves_the_video_path_and_frame_offset() {
        let command = mpv_command(Path::new("/media/library/video one.mp4"), Some(12.345));
        let args: Vec<_> = command.get_args().collect();
        assert_eq!(command.get_program(), OsStr::new("mpv"));
        assert_eq!(
            args,
            [
                "--force-window=immediate",
                "--start=12.345",
                "--",
                "/media/library/video one.mp4",
            ]
        );
    }

    #[test]
    fn mpv_launch_discards_stale_tokens_and_only_adds_requested_seek_offsets() {
        let command = mpv_command(Path::new("video.mp4"), None);
        let env: std::collections::HashMap<_, _> = command.get_envs().collect();
        assert_eq!(env.get(OsStr::new("XDG_ACTIVATION_TOKEN")), Some(&None));
        assert_eq!(env.get(OsStr::new("DESKTOP_STARTUP_ID")), Some(&None));
        assert!(
            !command
                .get_args()
                .any(|arg| arg.to_string_lossy().starts_with("--start="))
        );
        let command = mpv_command(Path::new("video.mp4"), Some(-4.0));
        assert!(command.get_args().any(|arg| arg == "--start=0.000"));
    }
}

#[derive(Default)]
struct ViewerDragGesture {
    export_attempted: bool,
}

impl ViewerDragGesture {
    fn export_requested(&mut self, response: &eframe::egui::Response) -> bool {
        use eframe::egui::{Event, PointerButton};

        let (primary_down, shift_down, outside_window) = response.ctx.input(|input| {
            let outside_window = input.pointer.latest_pos().map_or_else(
                || {
                    input
                        .events
                        .iter()
                        .any(|event| matches!(event, Event::PointerGone))
                },
                |position| !input.viewport_rect().contains(position),
            );
            (
                input.pointer.button_down(PointerButton::Primary),
                input.modifiers.shift,
                outside_window,
            )
        });
        if !primary_down || response.drag_started_by(PointerButton::Primary) {
            self.export_attempted = false;
        }
        if self.export_attempted || !response.dragged_by(PointerButton::Primary) {
            return false;
        }

        // Keep panning inside the window; hand the held drag to the desktop on exit.
        if outside_window || (shift_down && response.drag_started()) {
            self.export_attempted = true;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod viewer_gesture_tests {
    use super::*;
    use eframe::egui;

    fn frame(
        ctx: &egui::Context,
        gesture: &mut ViewerDragGesture,
        events: Vec<egui::Event>,
        modifiers: egui::Modifiers,
    ) -> (bool, bool) {
        let mut requested = false;
        let mut dragging = false;
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(640.0, 480.0),
                )),
                events,
                modifiers,
                ..Default::default()
            },
            |ctx| {
                egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
                    ui.label("Toolbar");
                });
                egui::SidePanel::right("duplicates")
                    .exact_width(180.0)
                    .show(ctx, |ui| {
                        ui.label("Duplicates");
                    });
                egui::CentralPanel::default().show(ctx, |ui| {
                    let (_, response) =
                        ui.allocate_at_least(ui.available_size(), egui::Sense::click_and_drag());
                    dragging |= response.dragged_by(egui::PointerButton::Primary);
                    requested |= gesture.export_requested(&response);
                });
            },
        );
        (requested, dragging)
    }

    fn button(pressed: bool, button: egui::PointerButton) -> egui::Event {
        egui::Event::PointerButton {
            pos: egui::pos2(100.0, 100.0),
            button,
            pressed,
            modifiers: egui::Modifiers::NONE,
        }
    }

    #[test]
    fn main_viewer_drag_pans_inside_then_exports_once_outside_and_can_be_repeated() {
        let ctx = egui::Context::default();
        let mut gesture = ViewerDragGesture::default();
        let modifiers = egui::Modifiers::NONE;
        frame(&ctx, &mut gesture, Vec::new(), modifiers);
        for _ in 0..2 {
            frame(
                &ctx,
                &mut gesture,
                vec![button(true, egui::PointerButton::Primary)],
                modifiers,
            );
            for position in [
                egui::pos2(200.0, 130.0),
                egui::pos2(520.0, 130.0), // Sidebar is still inside the app.
                egui::pos2(200.0, 10.0),  // Toolbar is still inside the app.
            ] {
                assert_eq!(
                    frame(
                        &ctx,
                        &mut gesture,
                        vec![egui::Event::PointerMoved(position)],
                        modifiers,
                    ),
                    (false, true),
                );
            }
            assert!(
                frame(
                    &ctx,
                    &mut gesture,
                    vec![egui::Event::PointerMoved(egui::pos2(650.0, 130.0))],
                    modifiers,
                )
                .0
            );
            assert!(
                !frame(
                    &ctx,
                    &mut gesture,
                    vec![egui::Event::PointerMoved(egui::pos2(670.0, 130.0))],
                    modifiers,
                )
                .0
            );
            frame(
                &ctx,
                &mut gesture,
                vec![button(false, egui::PointerButton::Primary)],
                modifiers,
            );
        }
    }

    #[test]
    fn window_exit_without_pointer_coordinates_still_exports_a_held_viewer_drag() {
        let ctx = egui::Context::default();
        let mut gesture = ViewerDragGesture::default();
        let modifiers = egui::Modifiers::NONE;
        frame(&ctx, &mut gesture, Vec::new(), modifiers);
        assert!(
            !frame(
                &ctx,
                &mut gesture,
                vec![egui::Event::PointerGone],
                modifiers,
            )
            .0
        );
        frame(
            &ctx,
            &mut gesture,
            vec![button(true, egui::PointerButton::Primary)],
            modifiers,
        );
        frame(
            &ctx,
            &mut gesture,
            vec![egui::Event::PointerMoved(egui::pos2(220.0, 140.0))],
            modifiers,
        );
        assert!(
            frame(
                &ctx,
                &mut gesture,
                vec![egui::Event::PointerGone],
                modifiers,
            )
            .0
        );
        assert!(!frame(&ctx, &mut gesture, Vec::new(), modifiers).0);
    }

    #[test]
    fn shift_starts_export_inside_window_but_other_mouse_buttons_do_not() {
        for button_kind in [
            egui::PointerButton::Primary,
            egui::PointerButton::Middle,
            egui::PointerButton::Extra1,
            egui::PointerButton::Extra2,
        ] {
            let ctx = egui::Context::default();
            let mut gesture = ViewerDragGesture::default();
            let modifiers = egui::Modifiers::SHIFT;
            frame(&ctx, &mut gesture, Vec::new(), modifiers);
            assert!(
                !frame(
                    &ctx,
                    &mut gesture,
                    vec![button(true, button_kind)],
                    modifiers,
                )
                .0
            );
            let requested = frame(
                &ctx,
                &mut gesture,
                vec![egui::Event::PointerMoved(egui::pos2(220.0, 140.0))],
                modifiers,
            )
            .0;
            assert_eq!(requested, button_kind == egui::PointerButton::Primary);
        }
    }
}

pub(crate) struct DesktopIntegration {
    ctx: eframe::egui::Context,
    viewer_gesture: ViewerDragGesture,
    #[cfg(target_os = "linux")]
    wayland: Option<wayland::WaylandDesktop>,
}

impl DesktopIntegration {
    pub(crate) fn new(creation_context: &eframe::CreationContext<'_>) -> Self {
        #[cfg(target_os = "linux")]
        {
            Self {
                ctx: creation_context.egui_ctx.clone(),
                viewer_gesture: ViewerDragGesture::default(),
                wayland: wayland::WaylandDesktop::new(creation_context).ok(),
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            Self {
                ctx: creation_context.egui_ctx.clone(),
                viewer_gesture: ViewerDragGesture::default(),
            }
        }
    }

    pub(crate) fn poll(&mut self) -> Result<(), String> {
        #[cfg(target_os = "linux")]
        if let Some(wayland) = self.wayland.as_mut() {
            return wayland.poll();
        }

        Ok(())
    }

    pub(crate) fn start(&mut self, path: &Path) -> Result<(), String> {
        #[cfg(target_os = "linux")]
        if let Some(wayland) = self.wayland.as_mut() {
            return wayland.start(path);
        }

        let _ = path;
        Err("outbound file dragging is unavailable on this display backend".to_string())
    }

    pub(crate) fn open_video(
        &mut self,
        path: &Path,
        start_seconds: Option<f32>,
    ) -> Result<(), String> {
        let mut command = mpv_command(path, start_seconds);
        #[cfg(target_os = "linux")]
        if let Some(wayland) = self.wayland.as_mut() {
            let result = wayland.launch_mpv(command);
            self.ctx.request_repaint();
            return result;
        }
        spawn_mpv(&mut command, None)
    }

    pub(crate) fn needs_poll(&self) -> bool {
        #[cfg(target_os = "linux")]
        if let Some(wayland) = self.wayland.as_ref() {
            return wayland.is_active() || wayland.has_pending_launches();
        }
        false
    }

    pub(crate) fn viewer_export_requested(&mut self, response: &eframe::egui::Response) -> bool {
        let requested = self.viewer_gesture.export_requested(response);
        requested && !self.is_active()
    }

    pub(crate) fn is_active(&self) -> bool {
        #[cfg(target_os = "linux")]
        if let Some(wayland) = self.wayland.as_ref() {
            return wayland.is_active();
        }

        false
    }
}

#[cfg(target_os = "linux")]
mod wayland {
    use super::spawn_mpv;
    use raw_window_handle::{HasDisplayHandle as _, RawDisplayHandle};
    use std::fs::File;
    use std::io::Write as _;
    use std::os::fd::OwnedFd;
    use std::os::unix::ffi::OsStrExt as _;
    use std::path::Path;
    use std::process::Command;
    use std::time::{Duration, Instant};
    use wayland_client::backend::Backend;
    use wayland_client::protocol::wl_data_device::{self, WlDataDevice};
    use wayland_client::protocol::wl_data_device_manager::{self, DndAction, WlDataDeviceManager};
    use wayland_client::protocol::wl_data_offer::{self, WlDataOffer};
    use wayland_client::protocol::wl_data_source::{self, WlDataSource};
    use wayland_client::protocol::wl_pointer::{self, WlPointer};
    use wayland_client::protocol::wl_registry::{self, WlRegistry};
    use wayland_client::protocol::wl_seat::{self, WlSeat};
    use wayland_client::protocol::wl_surface::WlSurface;
    use wayland_client::{Connection, Dispatch, EventQueue, Proxy as _, QueueHandle, WEnum};
    use wayland_protocols::xdg::activation::v1::client::xdg_activation_token_v1::{
        self, XdgActivationTokenV1,
    };
    use wayland_protocols::xdg::activation::v1::client::xdg_activation_v1::{
        self, XdgActivationV1,
    };

    const URI_LIST_MIME: &str = "text/uri-list";
    const BTN_LEFT: u32 = 0x110;

    pub(super) struct WaylandDesktop {
        connection: Connection,
        queue: EventQueue<DragState>,
        state: DragState,
    }

    #[derive(Default)]
    struct DragState {
        activation: Option<XdgActivationV1>,
        activation_surface: Option<WlSurface>,
        activation_serial: Option<u32>,
        pending_launches: Vec<PendingLaunch>,
        launch_error: Option<String>,
        manager: Option<WlDataDeviceManager>,
        manager_version: u32,
        seat: Option<WlSeat>,
        pointer: Option<WlPointer>,
        data_device: Option<WlDataDevice>,
        pointer_surface: Option<WlSurface>,
        press_surface: Option<WlSurface>,
        press_serial: Option<u32>,
        source: Option<WlDataSource>,
        payload: Vec<u8>,
        active: bool,
    }

    struct PendingLaunch {
        token: XdgActivationTokenV1,
        command: Command,
        deadline: Instant,
    }

    impl WaylandDesktop {
        pub(super) fn new(creation_context: &eframe::CreationContext<'_>) -> Result<Self, String> {
            let display_handle = creation_context
                .display_handle()
                .map_err(|error| format!("failed to access display handle: {error}"))?;
            let RawDisplayHandle::Wayland(display) = display_handle.as_raw() else {
                return Err("Iris is not running on native Wayland".to_string());
            };

            // eframe owns the display. This guest backend has its own event queue and
            // neither disconnects the display nor independently reads its socket.
            let backend = unsafe { Backend::from_foreign_display(display.display.as_ptr().cast()) };
            let connection = Connection::from_backend(backend);
            let mut queue = connection.new_event_queue();
            let qh = queue.handle();
            let mut state = DragState::default();
            let _registry = connection.display().get_registry(&qh, ());

            // First enumerate globals, then receive the initial seat capabilities.
            queue
                .roundtrip(&mut state)
                .map_err(|error| format!("failed to enumerate Wayland globals: {error}"))?;
            queue.roundtrip(&mut state).map_err(|error| {
                format!("failed to initialize Wayland desktop integration: {error}")
            })?;

            Ok(Self {
                connection,
                queue,
                state,
            })
        }

        pub(super) fn poll(&mut self) -> Result<(), String> {
            self.queue
                .dispatch_pending(&mut self.state)
                .map_err(|error| format!("Wayland desktop dispatch failed: {error}"))?;
            // An unresponsive activation service must not prevent video playback.
            let now = Instant::now();
            for index in (0..self.state.pending_launches.len()).rev() {
                if self.state.pending_launches[index].deadline <= now {
                    let mut launch = self.state.pending_launches.swap_remove(index);
                    launch.token.destroy();
                    spawn_mpv(&mut launch.command, None)?;
                }
            }
            if let Some(error) = self.state.launch_error.take() {
                return Err(error);
            }
            Ok(())
        }

        pub(super) fn launch_mpv(&mut self, mut command: Command) -> Result<(), String> {
            self.poll()?;
            let Some(activation) = &self.state.activation else {
                return spawn_mpv(&mut command, None);
            };
            let token = activation.get_activation_token(&self.queue.handle(), ());
            if let (Some(serial), Some(seat)) = (self.state.activation_serial, &self.state.seat) {
                token.set_serial(serial, seat);
            }
            if let Some(surface) = &self.state.activation_surface {
                token.set_surface(surface);
            }
            token.set_app_id("mpv".to_string());
            token.commit();
            self.state.pending_launches.push(PendingLaunch {
                token,
                command,
                deadline: Instant::now() + Duration::from_secs(1),
            });
            if self.connection.flush().is_err() {
                let mut launch = self.state.pending_launches.pop().unwrap();
                launch.token.destroy();
                return spawn_mpv(&mut launch.command, None);
            }
            Ok(())
        }

        pub(super) fn has_pending_launches(&self) -> bool {
            !self.state.pending_launches.is_empty()
        }

        pub(super) fn start(&mut self, path: &Path) -> Result<(), String> {
            self.poll()?;
            if self.state.active {
                return Ok(());
            }

            let path = std::fs::canonicalize(path)
                .map_err(|error| format!("cannot drag {}: {error}", path.display()))?;
            let manager = self
                .state
                .manager
                .clone()
                .ok_or_else(|| "Wayland data-device manager is unavailable".to_string())?;
            let data_device = self
                .state
                .data_device
                .clone()
                .ok_or_else(|| "Wayland data device is unavailable".to_string())?;
            let origin = self
                .state
                .press_surface
                .clone()
                .ok_or_else(|| "the initiating pointer press was not available".to_string())?;
            let serial = self
                .state
                .press_serial
                .ok_or_else(|| "the initiating pointer serial was not available".to_string())?;

            let qh = self.queue.handle();
            let source = manager.create_data_source(&qh, ());
            source.offer(URI_LIST_MIME.to_string());
            if self.state.manager_version >= 3 {
                source.set_actions(DndAction::Copy);
            }

            self.state.payload = path_to_uri_list(&path).into_bytes();
            self.state.source = Some(source.clone());
            self.state.active = true;
            data_device.start_drag(Some(&source), &origin, None, serial);
            if let Err(error) = self.connection.flush() {
                self.state.finish_drag(&source);
                return Err(format!("failed to start Wayland file drag: {error}"));
            }
            Ok(())
        }

        pub(super) fn is_active(&self) -> bool {
            self.state.active
        }
    }

    impl DragState {
        fn ensure_drag_devices(&mut self, qh: &QueueHandle<Self>) {
            if self.data_device.is_none()
                && let (Some(manager), Some(seat)) = (self.manager.as_ref(), self.seat.as_ref())
            {
                self.data_device = Some(manager.get_data_device(seat, qh, ()));
            }
        }

        fn finish_drag(&mut self, source: &WlDataSource) {
            source.destroy();
            self.source = None;
            self.payload.clear();
            self.active = false;
            self.press_surface = None;
            self.press_serial = None;
        }
    }

    impl Dispatch<WlRegistry, ()> for DragState {
        fn event(
            state: &mut Self,
            registry: &WlRegistry,
            event: wl_registry::Event,
            _data: &(),
            _connection: &Connection,
            qh: &QueueHandle<Self>,
        ) {
            let wl_registry::Event::Global {
                name,
                interface,
                version,
            } = event
            else {
                return;
            };

            if interface == XdgActivationV1::interface().name && state.activation.is_none() {
                state.activation = Some(registry.bind(name, 1, qh, ()));
            } else if interface == WlDataDeviceManager::interface().name && state.manager.is_none()
            {
                let version = version.min(3);
                state.manager = Some(registry.bind(name, version, qh, ()));
                state.manager_version = version;
                state.ensure_drag_devices(qh);
            } else if interface == WlSeat::interface().name && state.seat.is_none() {
                state.seat = Some(registry.bind(name, version.min(7), qh, ()));
                state.ensure_drag_devices(qh);
            }
        }
    }

    impl Dispatch<WlSeat, ()> for DragState {
        fn event(
            state: &mut Self,
            seat: &WlSeat,
            event: wl_seat::Event,
            _data: &(),
            _connection: &Connection,
            qh: &QueueHandle<Self>,
        ) {
            if let wl_seat::Event::Capabilities { capabilities } = event
                && let WEnum::Value(capabilities) = capabilities
                && capabilities.contains(wl_seat::Capability::Pointer)
                && state.pointer.is_none()
            {
                state.pointer = Some(seat.get_pointer(qh, ()));
            }
        }
    }

    impl Dispatch<WlPointer, ()> for DragState {
        fn event(
            state: &mut Self,
            _pointer: &WlPointer,
            event: wl_pointer::Event,
            _data: &(),
            _connection: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            match event {
                wl_pointer::Event::Enter {
                    surface, serial, ..
                } => {
                    state.activation_surface = Some(surface.clone());
                    state.activation_serial = Some(serial);
                    state.pointer_surface = Some(surface);
                }
                wl_pointer::Event::Leave { surface, .. } => {
                    if state
                        .pointer_surface
                        .as_ref()
                        .is_some_and(|current| current == &surface)
                    {
                        state.pointer_surface = None;
                    }
                }
                wl_pointer::Event::Button {
                    serial,
                    button,
                    state: button_state,
                    ..
                } => {
                    state.activation_serial = Some(serial);
                    if let Some(surface) = &state.pointer_surface {
                        state.activation_surface = Some(surface.clone());
                    }
                    if button != BTN_LEFT {
                        return;
                    }
                    match button_state {
                        WEnum::Value(wl_pointer::ButtonState::Pressed) => {
                            state.press_serial = Some(serial);
                            state.press_surface = state.pointer_surface.clone();
                        }
                        WEnum::Value(wl_pointer::ButtonState::Released) if !state.active => {
                            state.press_serial = None;
                            state.press_surface = None;
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }

    impl Dispatch<WlDataDeviceManager, ()> for DragState {
        fn event(
            _state: &mut Self,
            _manager: &WlDataDeviceManager,
            _event: wl_data_device_manager::Event,
            _data: &(),
            _connection: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
        }
    }

    impl Dispatch<XdgActivationV1, ()> for DragState {
        fn event(
            _state: &mut Self,
            _activation: &XdgActivationV1,
            _event: xdg_activation_v1::Event,
            _data: &(),
            _connection: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
        }
    }

    impl Dispatch<XdgActivationTokenV1, ()> for DragState {
        fn event(
            state: &mut Self,
            token: &XdgActivationTokenV1,
            event: xdg_activation_token_v1::Event,
            _data: &(),
            _connection: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            if let xdg_activation_token_v1::Event::Done {
                token: activation_token,
            } = event
                && let Some(index) = state
                    .pending_launches
                    .iter()
                    .position(|launch| &launch.token == token)
            {
                let mut launch = state.pending_launches.swap_remove(index);
                launch.token.destroy();
                if let Err(error) = spawn_mpv(&mut launch.command, Some(&activation_token)) {
                    state.launch_error = Some(error);
                }
            }
        }
    }

    impl Dispatch<WlDataDevice, ()> for DragState {
        wayland_client::event_created_child!(DragState, WlDataDevice, [
            0 => (WlDataOffer, ())
        ]);

        fn event(
            _state: &mut Self,
            _device: &WlDataDevice,
            _event: wl_data_device::Event,
            _data: &(),
            _connection: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
        }
    }

    impl Dispatch<WlDataOffer, ()> for DragState {
        fn event(
            _state: &mut Self,
            _offer: &WlDataOffer,
            _event: wl_data_offer::Event,
            _data: &(),
            _connection: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
        }
    }

    impl Dispatch<WlDataSource, ()> for DragState {
        fn event(
            state: &mut Self,
            source: &WlDataSource,
            event: wl_data_source::Event,
            _data: &(),
            _connection: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            match event {
                wl_data_source::Event::Send { mime_type, fd } if mime_type == URI_LIST_MIME => {
                    write_payload(fd, &state.payload);
                }
                wl_data_source::Event::Cancelled | wl_data_source::Event::DndFinished => {
                    state.finish_drag(source);
                }
                _ => {}
            }
        }
    }

    fn write_payload(fd: OwnedFd, payload: &[u8]) {
        let mut file = File::from(fd);
        let _ = file.write_all(payload);
        let _ = file.flush();
    }

    fn path_to_uri_list(path: &Path) -> String {
        let mut uri = String::from("file://");
        for byte in path.as_os_str().as_bytes() {
            if byte.is_ascii_alphanumeric() || matches!(byte, b'/' | b'-' | b'.' | b'_' | b'~') {
                uri.push(*byte as char);
            } else {
                use std::fmt::Write as _;
                let _ = write!(uri, "%{byte:02X}");
            }
        }
        uri.push_str("\r\n");
        uri
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::path::PathBuf;

        #[test]
        fn file_drag_uri_list_escapes_reserved_path_bytes() {
            assert_eq!(
                path_to_uri_list(&PathBuf::from("/tmp/photo one#2.jpg")),
                "file:///tmp/photo%20one%232.jpg\r\n"
            );
        }
    }
}
