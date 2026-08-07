use super::*;

impl ImageViewer {
    pub(crate) const SIDE_PANEL_WIDTH: f32 = 400.0;
    pub(crate) const MIN_WINDOW_WIDTH: f32 = 640.0;
    pub(crate) const SIDE_PANEL_RESIZE_TOLERANCE: f32 = 6.0;
    pub(crate) const SIDE_PANEL_OPEN_FALLBACK_FRAMES: u8 = 8;

    pub(crate) fn viewport_inner_size(ctx: &egui::Context) -> egui::Vec2 {
        ctx.input(|input| {
            input
                .viewport()
                .inner_rect
                .map(|rect| rect.size())
                .unwrap_or_else(|| input.viewport_rect().size())
        })
    }

    pub(crate) fn set_window_width(ctx: &egui::Context, width: f32, height: f32) {
        ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(width, height)));
    }

    pub(crate) fn open_side_panel(&mut self, ctx: &egui::Context, mode: SidePanelMode) {
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

    pub(crate) fn close_side_panel(&mut self, ctx: &egui::Context) {
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

    pub(crate) fn toggle_layout_side_panel(&mut self, ctx: &egui::Context) {
        let layout_active = (self.show_exif || self.side_panel_open_pending)
            && self.side_panel_mode == SidePanelMode::Layout;
        if layout_active {
            self.close_side_panel(ctx);
        } else {
            self.open_side_panel(ctx, SidePanelMode::Layout);
        }
    }

    pub(crate) fn apply_pending_side_panel_open(&mut self, ctx: &egui::Context) {
        if !self.side_panel_open_pending {
            return;
        }

        let current_size = Self::viewport_inner_size(ctx);
        let target_width = self
            .side_panel_expand_target_width
            .unwrap_or(current_size.x);
        self.side_panel_open_pending_frames = self.side_panel_open_pending_frames.saturating_add(1);

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
}
