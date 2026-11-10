use std::path::Path;

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
    viewer_gesture: ViewerDragGesture,
    #[cfg(target_os = "linux")]
    wayland: Option<wayland::WaylandFileDrag>,
}

impl DesktopIntegration {
    pub(crate) fn new(creation_context: &eframe::CreationContext<'_>) -> Self {
        #[cfg(target_os = "linux")]
        {
            Self {
                viewer_gesture: ViewerDragGesture::default(),
                wayland: wayland::WaylandFileDrag::new(creation_context).ok(),
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            let _ = creation_context;
            Self {
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
    use raw_window_handle::{HasDisplayHandle as _, RawDisplayHandle};
    use std::fs::File;
    use std::io::Write as _;
    use std::os::fd::OwnedFd;
    use std::os::unix::ffi::OsStrExt as _;
    use std::path::Path;
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

    const URI_LIST_MIME: &str = "text/uri-list";
    const BTN_LEFT: u32 = 0x110;

    pub(super) struct WaylandFileDrag {
        connection: Connection,
        queue: EventQueue<DragState>,
        state: DragState,
    }

    #[derive(Default)]
    struct DragState {
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

    impl WaylandFileDrag {
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
            queue
                .roundtrip(&mut state)
                .map_err(|error| format!("failed to initialize Wayland drag source: {error}"))?;

            if state.manager.is_none() || state.data_device.is_none() || state.pointer.is_none() {
                return Err("the Wayland compositor did not provide file drag support".to_string());
            }

            Ok(Self {
                connection,
                queue,
                state,
            })
        }

        pub(super) fn poll(&mut self) -> Result<(), String> {
            self.queue
                .dispatch_pending(&mut self.state)
                .map(|_| ())
                .map_err(|error| format!("Wayland file drag dispatch failed: {error}"))
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

            if interface == WlDataDeviceManager::interface().name && state.manager.is_none() {
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
                wl_pointer::Event::Enter { surface, .. } => {
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
                } if button == BTN_LEFT => match button_state {
                    WEnum::Value(wl_pointer::ButtonState::Pressed) => {
                        state.press_serial = Some(serial);
                        state.press_surface = state.pointer_surface.clone();
                    }
                    WEnum::Value(wl_pointer::ButtonState::Released) if !state.active => {
                        state.press_serial = None;
                        state.press_surface = None;
                    }
                    _ => {}
                },
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
