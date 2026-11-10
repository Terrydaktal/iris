use super::*;

#[derive(Clone, Copy)]
enum ToolbarAction {
    Filesystem,
    OpenFile,
    ComparePaths,
    Gallery,
    Layout,
    Sampling(ViewerSampling),
    SyncView(bool),
    Align,
    ExitCompare,
}

struct ComparisonControls<'a> {
    sync: bool,
    can_align: bool,
    align_label: &'static str,
    status: &'a str,
}

struct ToolbarInfo<'a> {
    filename: &'a str,
    details: &'a str,
    position: &'a str,
    gallery: bool,
    layout: bool,
    sampling: Option<ViewerSampling>,
    comparison: Option<ComparisonControls<'a>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ComparisonPlacement {
    Inline,
    More,
    File,
}

struct ToolbarLayout {
    gallery: bool,
    layout: bool,
    sampling: bool,
    sampling_width: f32,
    comparison: ComparisonPlacement,
}

fn text_width(ui: &egui::Ui, text: &str, style: egui::TextStyle) -> f32 {
    ui.painter()
        .layout_no_wrap(
            text.to_string(),
            style.resolve(ui.style()),
            ui.visuals().text_color(),
        )
        .size()
        .x
        .ceil()
}

fn button_width(ui: &egui::Ui, text: &str) -> f32 {
    (text_width(ui, text, egui::TextStyle::Button) + 2.0 * ui.spacing().button_padding.x + 2.0)
        .max(ui.spacing().interact_size.x)
}

impl ToolbarLayout {
    fn for_ui(ui: &egui::Ui, info: &ToolbarInfo<'_>) -> Self {
        let gap = ui.spacing().item_spacing.x;
        let title_width = 72.0 + text_width(ui, info.position, egui::TextStyle::Body) + gap;
        let mut remaining =
            (ui.available_width() - button_width(ui, "File") - gap - title_width).max(0.0);
        let mut fits = |width: f32| {
            if remaining >= width + gap {
                remaining -= width + gap;
                true
            } else {
                false
            }
        };
        let sampling_width =
            button_width(ui, "Pixel Sharp") + ui.spacing().icon_width + ui.spacing().icon_spacing;
        let gallery = fits(button_width(ui, "Gallery"));
        let layout = fits(button_width(ui, "Layout"));
        let sampling = info.sampling.is_some() && fits(sampling_width);
        let comparison = if let Some(comparison) = &info.comparison {
            let inline_width = button_width(ui, "Sync View")
                + button_width(ui, comparison.align_label)
                + button_width(ui, "Exit Compare")
                + 2.0 * gap;
            if fits(inline_width) {
                ComparisonPlacement::Inline
            } else if fits(button_width(ui, "More")) {
                ComparisonPlacement::More
            } else {
                ComparisonPlacement::File
            }
        } else {
            ComparisonPlacement::File
        };
        Self {
            gallery,
            layout,
            sampling,
            sampling_width,
            comparison,
        }
    }
}

fn sampling_menu(ui: &mut egui::Ui, sampling: ViewerSampling, action: &mut Option<ToolbarAction>) {
    for (mode, label) in [
        (ViewerSampling::Smooth, "Smooth"),
        (ViewerSampling::PixelSharp, "Pixel Sharp"),
    ] {
        if ui.selectable_label(sampling == mode, label).clicked() {
            *action = Some(ToolbarAction::Sampling(mode));
            ui.close();
        }
    }
}

fn comparison_controls(
    ui: &mut egui::Ui,
    comparison: &ComparisonControls<'_>,
    action: &mut Option<ToolbarAction>,
) {
    if ui
        .add(egui::Button::new("Sync View").selected(comparison.sync))
        .on_hover_text("Share the current zoom and pan position across comparison images.")
        .clicked()
    {
        *action = Some(ToolbarAction::SyncView(!comparison.sync));
    }
    if ui
        .add_enabled(
            comparison.can_align,
            egui::Button::new(comparison.align_label),
        )
        .on_hover_text("Align every comparison image to the first image using SIFT.")
        .clicked()
    {
        *action = Some(ToolbarAction::Align);
    }
    if ui
        .button("Exit Compare")
        .on_hover_text("Return to normal viewing of the current image.")
        .clicked()
    {
        *action = Some(ToolbarAction::ExitCompare);
    }
}

fn gallery_control(ui: &mut egui::Ui, active: bool, action: &mut Option<ToolbarAction>) {
    if ui
        .add(egui::Button::new("Gallery").selected(active))
        .on_hover_text("Show or hide the gallery (G)")
        .clicked()
    {
        *action = Some(ToolbarAction::Gallery);
    }
}

fn layout_control(ui: &mut egui::Ui, active: bool, action: &mut Option<ToolbarAction>) {
    if ui
        .add(egui::Button::new("Layout").selected(active))
        .on_hover_text("Show or hide the layout sidebar (E)")
        .clicked()
    {
        *action = Some(ToolbarAction::Layout);
    }
}

fn file_menu(
    ui: &mut egui::Ui,
    info: &ToolbarInfo<'_>,
    layout: &ToolbarLayout,
    action: &mut Option<ToolbarAction>,
) {
    for (label, shortcut, next) in [
        ("Filesystem", "", ToolbarAction::Filesystem),
        ("Open File", "Ctrl+O", ToolbarAction::OpenFile),
        ("Compare Paths", "Ctrl+Shift+O", ToolbarAction::ComparePaths),
    ] {
        if ui
            .add(egui::Button::new(label).shortcut_text(shortcut))
            .clicked()
        {
            *action = Some(next);
            ui.close();
        }
    }
    if !layout.gallery || !layout.layout || (!layout.sampling && info.sampling.is_some()) {
        ui.separator();
        if !layout.gallery {
            gallery_control(ui, info.gallery, action);
        }
        if !layout.layout {
            layout_control(ui, info.layout, action);
        }
        if !layout.sampling
            && let Some(sampling) = info.sampling
        {
            ui.separator();
            ui.weak("Image Smoothing");
            sampling_menu(ui, sampling, action);
        }
    }
    if layout.comparison == ComparisonPlacement::File
        && let Some(comparison) = &info.comparison
    {
        ui.separator();
        ui.weak("Comparison");
        comparison_controls(ui, comparison, action);
        if !comparison.status.is_empty() {
            ui.separator();
            ui.add(egui::Label::new(comparison.status).wrap());
        }
    }
}

fn filename_label(ui: &mut egui::Ui, info: &ToolbarInfo<'_>) {
    let available = ui.available_size_before_wrap();
    if available.x <= 0.0 {
        return;
    }
    let (mut filename_rect, _) = ui.allocate_exact_size(
        egui::vec2(available.x, ui.spacing().interact_size.y),
        egui::Sense::hover(),
    );
    let position_width = text_width(ui, info.position, egui::TextStyle::Body);
    if !info.position.is_empty() && filename_rect.width() > position_width + 32.0 {
        let position_rect = egui::Rect::from_min_max(
            egui::pos2(filename_rect.max.x - position_width, filename_rect.min.y),
            filename_rect.max,
        );
        filename_rect.max.x = position_rect.min.x - ui.spacing().item_spacing.x;
        ui.put(
            position_rect,
            egui::Label::new(info.position)
                .truncate()
                .show_tooltip_when_elided(false),
        )
        .on_hover_text(info.details);
    }
    ui.put(
        filename_rect,
        egui::Label::new(info.filename)
            .truncate()
            .show_tooltip_when_elided(false)
            .halign(egui::Align::Min),
    )
    .on_hover_text(info.details);
}

fn toolbar_contents(ui: &mut egui::Ui, info: &ToolbarInfo<'_>) -> Option<ToolbarAction> {
    let mut action = None;
    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
    ui.horizontal(|ui| {
        // Budget controls first. Only the filename may consume the remaining width.
        let layout = ToolbarLayout::for_ui(ui, info);
        ui.menu_button("File", |ui| file_menu(ui, info, &layout, &mut action));
        if layout.gallery {
            gallery_control(ui, info.gallery, &mut action);
        }
        if layout.layout {
            layout_control(ui, info.layout, &mut action);
        }
        if layout.sampling
            && let Some(sampling) = info.sampling
        {
            egui::ComboBox::from_id_salt("viewer_smoothing")
                .width(layout.sampling_width)
                .selected_text(match sampling {
                    ViewerSampling::Smooth => "Smooth",
                    ViewerSampling::PixelSharp => "Pixel Sharp",
                })
                .show_ui(ui, |ui| sampling_menu(ui, sampling, &mut action))
                .response
                .on_hover_text("Image smoothing: smooth edges or distinct enlarged pixels.");
        }
        if let Some(comparison) = &info.comparison {
            match layout.comparison {
                ComparisonPlacement::Inline => comparison_controls(ui, comparison, &mut action),
                ComparisonPlacement::More => {
                    ui.menu_button("More", |ui| {
                        comparison_controls(ui, comparison, &mut action);
                        if !comparison.status.is_empty() {
                            ui.separator();
                            ui.add(egui::Label::new(comparison.status).wrap());
                        }
                    })
                    .response
                    .on_hover_text(if comparison.status.is_empty() {
                        "Comparison actions"
                    } else {
                        comparison.status
                    });
                }
                ComparisonPlacement::File => {}
            }
        }
        filename_label(ui, info);
    });
    action
}

impl ImageViewer {
    pub(crate) fn show_top_bar(&mut self, ctx: &egui::Context) {
        let (filename, mut details) = if let Some(path) = self.images.get(self.current_index) {
            let db_name = self.resolve_db_filename(path);
            let display_path = db_name.as_deref().map(Path::new).unwrap_or(path);
            (
                display_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
                format!(
                    "{}\n{} | {}",
                    display_path.display(),
                    self.current_dimensions,
                    self.current_file_size
                ),
            )
        } else {
            ("No image loaded".to_string(), String::new())
        };
        let position = if self.images.get(self.current_index).is_some() {
            let (index, count) = if let Some(indices) = &self.gallery_navigation_indices {
                (self.gallery_navigation_position + 1, indices.len())
            } else {
                (self.current_index + 1, self.images.len())
            };
            if self.is_comparison_mode() {
                format!("Compare {index}/{count}")
            } else {
                format!("({index}/{count})")
            }
        } else {
            String::new()
        };
        if !position.is_empty() {
            details.push('\n');
            details.push_str(&position);
        }
        let comparison = self.is_comparison_mode().then(|| ComparisonControls {
            sync: self.comparison_sync_view,
            can_align: !self.sift_align_all_running
                && self.images.iter().all(|path| !is_video_path(path)),
            align_label: if self.sift_align_all_running {
                "SIFT Aligning"
            } else if !self.comparison_aligned_paths.is_empty() {
                "SIFT Re-align"
            } else {
                "SIFT Align"
            },
            status: &self.comparison_alignment_status,
        });
        if let Some(comparison) = &comparison {
            details.push_str("\nComparison: use Left/Right to switch images.");
            if !comparison.status.is_empty() {
                details.push('\n');
                details.push_str(comparison.status);
            }
        }
        let info = ToolbarInfo {
            filename: &filename,
            details: &details,
            position: &position,
            gallery: self.show_grid,
            layout: (self.show_exif || self.side_panel_open_pending)
                && self.side_panel_mode == SidePanelMode::Layout,
            sampling: (!self.show_grid && self.image_editor.is_none())
                .then_some(self.viewer_sampling),
            comparison,
        };
        let action = egui::TopBottomPanel::top("top_bar")
            .exact_height(IMAGE_VIEWER_TOP_BAR_HEIGHT)
            .show(ctx, |ui| toolbar_contents(ui, &info))
            .inner;
        match action {
            Some(ToolbarAction::Filesystem) => {
                self.clear_comparison_mode();
                self.show_home_page = true;
            }
            Some(ToolbarAction::OpenFile) => self.open_file_dialog(ctx),
            Some(ToolbarAction::ComparePaths) => self.open_comparison_path_dialog(),
            Some(ToolbarAction::Gallery) => {
                if !self.show_grid {
                    self.clear_comparison_mode();
                }
                self.show_grid = !self.show_grid;
                if self.show_grid && self.recursive_images.is_empty() && !self.grid_loading {
                    self.start_recursive_scan();
                }
            }
            Some(ToolbarAction::Layout) => self.toggle_layout_side_panel(ctx),
            Some(ToolbarAction::Sampling(sampling)) => self.set_viewer_sampling(sampling, ctx),
            Some(ToolbarAction::SyncView(sync)) => {
                self.comparison_sync_view = sync;
                if sync {
                    self.apply_comparison_view_state_to_all();
                }
            }
            Some(ToolbarAction::Align) => self.start_sift_align_all(ctx),
            Some(ToolbarAction::ExitCompare) => {
                if let Some(path) = self.images.get(self.current_index).cloned() {
                    self.open_image_path(path);
                }
            }
            None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(comparison: bool) -> ToolbarInfo<'static> {
        ToolbarInfo {
            filename: "a_very_long_screenshot_filename_that_must_not_push_controls_outside_the_window_or_overlap_any_buttons_even_when_comparing_several_images_at_once.png",
            details: "/media/library/screenshots/example.png\n1280x10235 | 744 KiB",
            position: if comparison { "Compare 2/6" } else { "(1/500000)" },
            gallery: false,
            layout: true,
            sampling: Some(ViewerSampling::Smooth),
            comparison: comparison.then_some(ComparisonControls {
                sync: false,
                can_align: true,
                align_label: "SIFT Re-align",
                status: "Alignment complete. This deliberately long status belongs in the tooltip or menu, not in the toolbar where it would displace the file name and other controls.",
            }),
        }
    }

    fn frame(
        ctx: &egui::Context,
        width: f32,
        info: &ToolbarInfo<'_>,
        events: Vec<egui::Event>,
    ) -> (egui::FullOutput, Option<ToolbarAction>) {
        let mut action = None;
        let output = ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(width, 600.0),
                )),
                events,
                ..Default::default()
            },
            |ctx| {
                let response = egui::TopBottomPanel::top("top_bar")
                    .exact_height(IMAGE_VIEWER_TOP_BAR_HEIGHT)
                    .show(ctx, |ui| toolbar_contents(ui, info));
                assert_eq!(response.response.rect.height(), IMAGE_VIEWER_TOP_BAR_HEIGHT);
                if response.inner.is_some() {
                    action = response.inner;
                }
            },
        );
        (output, action)
    }

    fn text_shapes(output: &egui::FullOutput) -> Vec<&egui::epaint::TextShape> {
        fn collect<'a>(shape: &'a egui::Shape, texts: &mut Vec<&'a egui::epaint::TextShape>) {
            match shape {
                egui::Shape::Text(text) => texts.push(text),
                egui::Shape::Vec(shapes) => {
                    for shape in shapes {
                        collect(shape, texts);
                    }
                }
                _ => {}
            }
        }
        let mut texts = Vec::new();
        for shape in &output.shapes {
            collect(&shape.shape, &mut texts);
        }
        texts
    }

    fn click_text(
        ctx: &egui::Context,
        width: f32,
        info: &ToolbarInfo<'_>,
        output: &egui::FullOutput,
        label: &str,
    ) -> (egui::FullOutput, Option<ToolbarAction>) {
        let texts = text_shapes(output);
        let target = texts
            .iter()
            .find(|text| text.galley.job.text == label)
            .unwrap_or_else(|| {
                panic!(
                    "Missing {label}; visible labels: {:?}",
                    texts
                        .iter()
                        .map(|text| text.galley.job.text.as_str())
                        .collect::<Vec<_>>()
                )
            });
        let pos = target.visual_bounding_rect().center();
        let _ = frame(
            ctx,
            width,
            info,
            vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        frame(
            ctx,
            width,
            info,
            vec![egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            }],
        )
    }

    #[test]
    fn toolbar_keeps_one_row_and_never_overlaps_text_at_narrow_widths() {
        for width in [80.0, 150.0, 220.0, 320.0, 480.0, 600.0, 900.0, 1400.0] {
            for comparison in [false, true] {
                for sampling in [
                    None,
                    Some(ViewerSampling::Smooth),
                    Some(ViewerSampling::PixelSharp),
                ] {
                    let ctx = egui::Context::default();
                    let mut info = fixture(comparison);
                    info.sampling = sampling;
                    info.gallery = sampling.is_none();
                    let (output, action) = frame(&ctx, width, &info, Vec::new());
                    assert!(action.is_none());
                    let texts = text_shapes(&output);
                    assert!(texts.iter().any(|text| text.galley.job.text == "File"));
                    let bounds = egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(width, IMAGE_VIEWER_TOP_BAR_HEIGHT),
                    );
                    for (index, text) in texts.iter().enumerate() {
                        let rect = text.visual_bounding_rect();
                        assert!(
                            bounds.expand(1.0).contains_rect(rect),
                            "Text outside toolbar at width {width}: {} {rect:?}",
                            text.galley.job.text
                        );
                        for other in texts.iter().skip(index + 1) {
                            assert!(
                                !rect.intersect(other.visual_bounding_rect()).is_positive(),
                                "Overlapping text at width {width}: {} and {}",
                                text.galley.job.text,
                                other.galley.job.text
                            );
                        }
                    }
                    assert!(
                        texts
                            .iter()
                            .any(|text| text.galley.job.text == info.filename),
                        "Filename missing at width {width}"
                    );
                    if width <= 900.0 {
                        assert!(
                            texts
                                .iter()
                                .any(|text| text.galley.job.text == info.filename
                                    && text.galley.elided),
                            "Missing truncated filename at width {width}, comparison={comparison}: {:?}",
                            texts
                                .iter()
                                .map(|text| (&text.galley.job.text, text.galley.elided))
                                .collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn file_menu_preserves_actions_and_contains_controls_that_do_not_fit() {
        for label in [
            "Filesystem",
            "Open File",
            "Compare Paths",
            "Gallery",
            "Layout",
        ] {
            let ctx = egui::Context::default();
            let info = fixture(false);
            let (output, _) = frame(&ctx, 180.0, &info, Vec::new());
            let _ = click_text(&ctx, 180.0, &info, &output, "File");
            let (output, _) = frame(&ctx, 180.0, &info, Vec::new());
            let (_, action) = click_text(&ctx, 180.0, &info, &output, label);
            assert!(
                matches!(
                    (label, action),
                    ("Filesystem", Some(ToolbarAction::Filesystem))
                        | ("Open File", Some(ToolbarAction::OpenFile))
                        | ("Compare Paths", Some(ToolbarAction::ComparePaths))
                        | ("Gallery", Some(ToolbarAction::Gallery))
                        | ("Layout", Some(ToolbarAction::Layout))
                ),
                "Wrong action for {label}"
            );
        }
    }

    #[test]
    fn smoothing_dropdown_and_narrow_window_menu_switch_to_pixel_sharp() {
        for width in [180.0, 900.0] {
            let ctx = egui::Context::default();
            let info = fixture(false);
            let (mut output, _) = frame(&ctx, width, &info, Vec::new());
            if width == 180.0 {
                let _ = click_text(&ctx, width, &info, &output, "File");
            } else {
                let _ = click_text(&ctx, width, &info, &output, "Smooth");
            }
            (output, _) = frame(&ctx, width, &info, Vec::new());
            let (_, action) = click_text(&ctx, width, &info, &output, "Pixel Sharp");
            assert!(matches!(
                action,
                Some(ToolbarAction::Sampling(ViewerSampling::PixelSharp))
            ));
        }
    }

    #[test]
    fn comparison_actions_remain_available_inline_and_in_overflow_menus() {
        for width in [180.0, 600.0, 1400.0] {
            let ctx = egui::Context::default();
            let info = fixture(true);
            let (mut output, _) = frame(&ctx, width, &info, Vec::new());
            if width == 180.0 {
                let _ = click_text(&ctx, width, &info, &output, "File");
                (output, _) = frame(&ctx, width, &info, Vec::new());
            } else if width == 600.0 {
                let _ = click_text(&ctx, width, &info, &output, "More");
                (output, _) = frame(&ctx, width, &info, Vec::new());
            }
            let (_, action) = click_text(&ctx, width, &info, &output, "Sync View");
            assert!(matches!(action, Some(ToolbarAction::SyncView(true))));
        }
    }
}
