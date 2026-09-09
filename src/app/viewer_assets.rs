use eframe::egui;
use image::DynamicImage;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use super::DiagnosticState;

pub(crate) const IMAGE_VIEWER_TOP_BAR_HEIGHT: f32 = 32.0;
pub(crate) const INITIAL_IMAGE_DISPLAY_HEIGHT: f32 = 1200.0;
pub(crate) const MAX_VIEWER_TEXTURE_DIMENSION: u32 = 4096;

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum ViewerSampling {
    #[default]
    Smooth,
    PixelSharp,
}

impl ViewerSampling {
    fn texture_options(self, native_resolution: bool) -> egui::TextureOptions {
        if self == Self::PixelSharp && native_resolution {
            egui::TextureOptions {
                magnification: egui::TextureFilter::Nearest,
                ..egui::TextureOptions::LINEAR
            }
        } else {
            egui::TextureOptions::LINEAR
        }
    }
}

#[derive(Clone)]
pub(crate) struct ViewerTexture {
    pub(crate) handle: egui::TextureHandle,
    first_pixel: egui::Color32,
    native_resolution: bool,
    options: egui::TextureOptions,
}

impl ViewerTexture {
    pub(crate) fn new(
        ctx: &egui::Context,
        name: String,
        image: egui::ColorImage,
        native_resolution: bool,
        sampling: ViewerSampling,
    ) -> Self {
        let options = sampling.texture_options(native_resolution);
        let first_pixel = image.pixels[0];
        Self {
            handle: ctx.load_texture(name, image, options),
            first_pixel,
            native_resolution,
            options,
        }
    }

    pub(crate) fn set_sampling(&mut self, sampling: ViewerSampling) {
        let options = sampling.texture_options(self.native_resolution);
        if options == self.options {
            return;
        }
        // egui changes filtering through texture updates. Resend one unchanged
        // pixel instead of retaining or uploading a second copy of the image.
        self.handle.set_partial(
            [0, 0],
            egui::ColorImage::filled([1, 1], self.first_pixel),
            options,
        );
        self.options = options;
    }
}

pub(crate) struct ViewerPreview {
    pub(crate) image: egui::ColorImage,
    pub(crate) source_size: [u32; 2],
}

impl ViewerPreview {
    pub(crate) fn from_image(image: DynamicImage) -> Self {
        Self {
            source_size: [image.width(), image.height()],
            image: viewer_color_image(image),
        }
    }

    pub(crate) fn is_native_resolution(&self) -> bool {
        self.image.size == self.source_size.map(|side| side as usize)
    }
}

pub(crate) fn downsample_for_viewer(image: DynamicImage) -> DynamicImage {
    let largest_dimension = image.width().max(image.height());
    if largest_dimension <= MAX_VIEWER_TEXTURE_DIMENSION {
        image
    } else {
        image.resize(
            MAX_VIEWER_TEXTURE_DIMENSION,
            MAX_VIEWER_TEXTURE_DIMENSION,
            image::imageops::FilterType::Triangle,
        )
    }
}

pub(crate) fn viewer_color_image(image: DynamicImage) -> egui::ColorImage {
    let image = downsample_for_viewer(image);
    viewer_color_image_ref(&image)
}

pub(crate) fn viewer_color_image_ref(image: &DynamicImage) -> egui::ColorImage {
    let image = if image.width().max(image.height()) > MAX_VIEWER_TEXTURE_DIMENSION {
        image.resize(
            MAX_VIEWER_TEXTURE_DIMENSION,
            MAX_VIEWER_TEXTURE_DIMENSION,
            image::imageops::FilterType::Triangle,
        )
    } else {
        image.clone()
    };
    let rgba = image.to_rgba8();
    egui::ColorImage::from_rgba_unmultiplied(
        [rgba.width() as usize, rgba.height() as usize],
        rgba.as_raw(),
    )
}

const MAX_DETAIL_SOURCE_BYTES: usize = 128 * 1024 * 1024;
const MAX_DETAIL_DECODE_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Clone, PartialEq, Eq)]
struct DetailSource {
    path: PathBuf,
    revision: u64,
    size: [u32; 2],
}

#[derive(Clone, Copy, Debug)]
struct DetailRegion {
    bounds: egui::Rect,
    texture_size: [u32; 2],
}

impl DetailRegion {
    fn for_view(visible: egui::Rect, source_size: [u32; 2], scale: f32, max_side: u32) -> Self {
        let source_rect = egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(source_size[0] as f32, source_size[1] as f32),
        );
        // Overscan avoids a new crop for every small pan, but must not reduce
        // visible detail just to fit the extra margin into the texture budget.
        let scale = scale.min(1.0);
        let margin_budget = ((egui::Vec2::splat(max_side as f32 / scale) - visible.size()) * 0.5
            - egui::Vec2::splat(1.0))
        .max(egui::Vec2::ZERO);
        let padded = visible
            .expand2((visible.size() * 0.15).min(margin_budget))
            .intersect(source_rect);
        let bounds = egui::Rect::from_min_max(padded.min.floor(), padded.max.ceil());
        let scale = scale
            .min(1.0)
            .min(max_side as f32 / bounds.width().max(bounds.height()));
        Self {
            bounds,
            texture_size: [
                (bounds.width() * scale).ceil().clamp(1.0, max_side as f32) as u32,
                (bounds.height() * scale).ceil().clamp(1.0, max_side as f32) as u32,
            ],
        }
    }

    fn scale(self) -> f32 {
        (self.texture_size[0] as f32 / self.bounds.width())
            .min(self.texture_size[1] as f32 / self.bounds.height())
    }

    fn covers(self, visible: egui::Rect, requested: Self) -> bool {
        self.bounds.contains_rect(visible)
            && if requested.is_native_resolution() {
                self.is_native_resolution()
            } else {
                self.scale() >= requested.scale() * 0.98
            }
    }

    fn is_native_resolution(self) -> bool {
        self.texture_size.map(|side| side as f32) == [self.bounds.width(), self.bounds.height()]
    }

    fn uv(self, size: [u32; 2]) -> egui::Rect {
        let size = egui::vec2(size[0] as f32, size[1] as f32);
        egui::Rect::from_min_max(
            egui::Pos2::ZERO + self.bounds.min.to_vec2() / size,
            egui::Pos2::ZERO + self.bounds.max.to_vec2() / size,
        )
    }
}

fn rotate_point(point: egui::Pos2, center: egui::Pos2, angle: f32) -> egui::Pos2 {
    let relative = point - center;
    let (sin, cos) = angle.sin_cos();
    center
        + egui::vec2(
            relative.x * cos - relative.y * sin,
            relative.x * sin + relative.y * cos,
        )
}

fn visible_source_rect(
    image_rect: egui::Rect,
    viewport: egui::Rect,
    angle: f32,
    source_size: [u32; 2],
) -> Option<egui::Rect> {
    if !image_rect.is_finite() || image_rect.width() <= 0.0 || image_rect.height() <= 0.0 {
        return None;
    }
    let mut unrotated = egui::Rect::NOTHING;
    for point in [
        viewport.left_top(),
        viewport.right_top(),
        viewport.right_bottom(),
        viewport.left_bottom(),
    ] {
        unrotated.extend_with(rotate_point(point, image_rect.center(), -angle));
    }
    let visible = unrotated.intersect(image_rect);
    if !visible.is_positive() {
        return None;
    }
    let scale = egui::vec2(source_size[0] as f32, source_size[1] as f32) / image_rect.size();
    Some(egui::Rect::from_min_max(
        egui::Pos2::ZERO + (visible.min - image_rect.min) * scale,
        egui::Pos2::ZERO + (visible.max - image_rect.min) * scale,
    ))
}

fn paint_image_region(
    painter: &egui::Painter,
    texture: egui::TextureId,
    image_rect: egui::Rect,
    region_uv: egui::Rect,
    texture_uv: egui::Rect,
    angle: f32,
) {
    if !region_uv.is_positive() {
        return;
    }
    let rect = egui::Rect::from_min_max(
        image_rect.min + region_uv.min.to_vec2() * image_rect.size(),
        image_rect.min + region_uv.max.to_vec2() * image_rect.size(),
    );
    let mut mesh = egui::Mesh::with_texture(texture);
    mesh.add_rect_with_uv(rect, texture_uv, egui::Color32::WHITE);
    for vertex in &mut mesh.vertices {
        vertex.pos = rotate_point(vertex.pos, image_rect.center(), angle);
    }
    painter.add(egui::Shape::mesh(mesh));
}

fn preview_regions(detail_uv: egui::Rect) -> [egui::Rect; 4] {
    // Partition instead of overlaying: transparent pixels must be composited only once.
    [
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, detail_uv.min.y)),
        egui::Rect::from_min_max(egui::pos2(0.0, detail_uv.max.y), egui::pos2(1.0, 1.0)),
        egui::Rect::from_min_max(
            egui::pos2(0.0, detail_uv.min.y),
            egui::pos2(detail_uv.min.x, detail_uv.max.y),
        ),
        egui::Rect::from_min_max(
            egui::pos2(detail_uv.max.x, detail_uv.min.y),
            egui::pos2(1.0, detail_uv.max.y),
        ),
    ]
}

#[derive(Default)]
struct DetailDecoder {
    cached: Option<(DetailSource, DynamicImage)>,
}

impl DetailDecoder {
    fn crop(
        &mut self,
        source: &DetailSource,
        region: DetailRegion,
    ) -> Result<egui::ColorImage, String> {
        if self.cached.as_ref().is_none_or(|(key, _)| key != source) {
            self.cached = None;
            let mut reader =
                image::ImageReader::open(&source.path).map_err(|err| err.to_string())?;
            let mut limits = image::Limits::default();
            limits.max_alloc = Some(MAX_DETAIL_DECODE_BYTES);
            reader.limits(limits);
            let image = reader.decode().map_err(|err| err.to_string())?;
            if [image.width(), image.height()] != source.size {
                return Err("Image dimensions changed while loading detail".to_string());
            }
            self.cached = Some((source.clone(), image));
        }
        let image = &self.cached.as_ref().expect("source decoded").1;
        let crop = image.crop_imm(
            region.bounds.min.x as u32,
            region.bounds.min.y as u32,
            region.bounds.width() as u32,
            region.bounds.height() as u32,
        );
        let crop = if [crop.width(), crop.height()] == region.texture_size {
            crop
        } else {
            crop.resize_exact(
                region.texture_size[0],
                region.texture_size[1],
                image::imageops::FilterType::Triangle,
            )
        };
        let rgba = crop.to_rgba8();
        let result = egui::ColorImage::from_rgba_unmultiplied(
            [rgba.width() as usize, rgba.height() as usize],
            rgba.as_raw(),
        );
        if image.as_bytes().len() > MAX_DETAIL_SOURCE_BYTES {
            self.cached = None;
        }
        Ok(result)
    }
}

struct PendingDetail {
    epoch: u64,
    region: DetailRegion,
    rx: mpsc::Receiver<Result<egui::ColorImage, String>>,
}

#[derive(Default)]
pub(crate) struct ViewerDetail {
    source: Option<DetailSource>,
    epoch: u64,
    texture: Option<(DetailRegion, ViewerTexture)>,
    pending: Option<PendingDetail>,
    decoder: Arc<Mutex<DetailDecoder>>,
    retry_at: Option<Instant>,
    sampling: ViewerSampling,
}

impl ViewerDetail {
    pub(crate) fn set_sampling(&mut self, sampling: ViewerSampling) {
        self.sampling = sampling;
        if let Some((_, texture)) = &mut self.texture {
            texture.set_sampling(sampling);
        }
    }

    pub(crate) fn clear(&mut self) {
        if self.source.take().is_some() {
            self.epoch = self.epoch.wrapping_add(1);
            self.texture = None;
            self.retry_at = None;
            self.decoder = Arc::default();
        }
        // Keep the receiver until completion so changing images cannot launch
        // unbounded concurrent decodes. Its epoch prevents stale uploads.
    }

    pub(crate) fn poll(&mut self, ctx: &egui::Context) {
        let Some(pending) = self.pending.take() else {
            return;
        };
        match pending.rx.try_recv() {
            Ok(result) if pending.epoch == self.epoch => match result {
                Ok(image) => {
                    self.texture = Some((
                        pending.region,
                        ViewerTexture::new(
                            ctx,
                            "viewer_detail".to_string(),
                            image,
                            pending.region.is_native_resolution(),
                            self.sampling,
                        ),
                    ));
                }
                Err(_) => self.retry_at = Some(Instant::now() + Duration::from_secs(2)),
            },
            Err(mpsc::TryRecvError::Empty) => self.pending = Some(pending),
            _ => {}
        }
    }

    pub(crate) fn paint(
        &mut self,
        painter: &egui::Painter,
        path: &Path,
        revision: u64,
        source_size: [u32; 2],
        preview: &egui::TextureHandle,
        image_rect: egui::Rect,
        angle: f32,
        diagnostics: &DiagnosticState,
    ) {
        let ctx = painter.ctx();
        let full_uv = egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0));
        let scale = image_rect.width() * ctx.pixels_per_point() / source_size[0] as f32;
        let preview_scale = preview.size()[0] as f32 / source_size[0] as f32;
        let visible = visible_source_rect(image_rect, painter.clip_rect(), angle, source_size);
        if scale <= preview_scale * 1.05
            || preview.size() == source_size.map(|v| v as usize)
            || visible.is_none()
        {
            self.clear();
            self.poll(ctx);
            paint_image_region(painter, preview.id(), image_rect, full_uv, full_uv, angle);
            return;
        }
        if self.source.as_ref().is_none_or(|key| {
            key.path != path || key.revision != revision || key.size != source_size
        }) {
            self.clear();
            self.source = Some(DetailSource {
                path: path.to_path_buf(),
                revision,
                size: source_size,
            });
        }
        self.poll(ctx);
        let visible = visible.expect("visible image region");
        let max_side =
            MAX_VIEWER_TEXTURE_DIMENSION.min(ctx.input(|input| input.max_texture_side) as u32);
        let requested = DetailRegion::for_view(visible, source_size, scale, max_side);
        let ready = self
            .texture
            .as_ref()
            .is_some_and(|(region, _)| region.covers(visible, requested));
        if !ready && self.pending.is_none() && self.retry_at.is_none_or(|at| Instant::now() >= at) {
            let source = self.source.clone().expect("detail source selected");
            let decoder = self.decoder.clone();
            let (tx, rx) = mpsc::channel();
            self.pending = Some(PendingDetail {
                epoch: self.epoch,
                region: requested,
                rx,
            });
            let ctx = ctx.clone();
            let diagnostics = diagnostics.clone();
            rayon::spawn(move || {
                let task = diagnostics.task_guard("viewer_detail_decode");
                let result = decoder
                    .lock()
                    .map_err(|_| "Detail decoder lock poisoned".to_string())
                    .and_then(|mut decoder| decoder.crop(&source, requested));
                let _ = tx.send(result);
                ctx.request_repaint();
                task.complete();
            });
        }
        if let Some((region, texture)) = &self.texture {
            let detail_uv = region.uv(source_size);
            for uv in preview_regions(detail_uv) {
                paint_image_region(painter, preview.id(), image_rect, uv, uv, angle);
            }
            paint_image_region(
                painter,
                texture.handle.id(),
                image_rect,
                detail_uv,
                full_uv,
                angle,
            );
        } else {
            paint_image_region(painter, preview.id(), image_rect, full_uv, full_uv, angle);
        }
    }
}

#[cfg(test)]
mod detail_tests {
    use super::*;

    #[test]
    fn sampling_defaults_to_smooth_and_pixel_sharp_only_changes_native_enlargement() {
        assert!(ViewerSampling::default() == ViewerSampling::Smooth);
        assert_eq!(
            ViewerSampling::Smooth.texture_options(true),
            egui::TextureOptions::LINEAR
        );
        assert_eq!(
            ViewerSampling::PixelSharp.texture_options(false),
            egui::TextureOptions::LINEAR
        );
        let options = ViewerSampling::PixelSharp.texture_options(true);
        assert_eq!(options.magnification, egui::TextureFilter::Nearest);
        assert_eq!(options.minification, egui::TextureFilter::Linear);
    }

    #[test]
    fn changing_filter_reuses_texture_and_only_uploads_one_unchanged_pixel() {
        let ctx = egui::Context::default();
        let first_pixel = egui::Color32::from_rgba_unmultiplied(31, 57, 89, 170);
        let mut texture = ViewerTexture::new(
            &ctx,
            "filter_test".to_string(),
            egui::ColorImage::filled([64, 64], first_pixel),
            true,
            ViewerSampling::Smooth,
        );
        let id = texture.handle.id();
        let manager = ctx.tex_manager();
        let _ = manager.write().take_delta();
        let allocated = manager.read().num_allocated();
        for sampling in [ViewerSampling::PixelSharp, ViewerSampling::Smooth] {
            texture.set_sampling(sampling);
            assert_eq!(texture.handle.id(), id);
            assert_eq!(texture.handle.size(), [64, 64]);
            assert_eq!(manager.read().num_allocated(), allocated);
            let delta = manager.write().take_delta();
            assert!(delta.free.is_empty());
            assert_eq!(delta.set.len(), 1);
            let (updated_id, update) = &delta.set[0];
            assert_eq!(*updated_id, id);
            assert_eq!(update.pos, Some([0, 0]));
            assert_eq!(update.image.size(), [1, 1]);
            assert_eq!(update.options, sampling.texture_options(true));
            match &update.image {
                egui::ImageData::Color(image) => assert_eq!(image.pixels, [first_pixel]),
            }
            texture.set_sampling(sampling);
            assert!(manager.write().take_delta().set.is_empty());
        }
    }

    #[test]
    fn pending_detail_uses_latest_filter_without_restarting_decode() {
        let ctx = egui::Context::default();
        let mut detail = ViewerDetail::default();
        detail.source = Some(DetailSource {
            path: PathBuf::from("screenshot.png"),
            revision: 0,
            size: [1280, 10235],
        });
        let region = DetailRegion::for_view(
            egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(128.0, 100.0)),
            [1280, 10235],
            1.0,
            4096,
        );
        let (tx, rx) = mpsc::channel();
        detail.pending = Some(PendingDetail {
            epoch: detail.epoch,
            region,
            rx,
        });
        detail.set_sampling(ViewerSampling::PixelSharp);
        assert!(detail.pending.is_some());
        assert_eq!(detail.epoch, 0);
        tx.send(Ok(egui::ColorImage::filled(
            region.texture_size.map(|side| side as usize),
            egui::Color32::GRAY,
        )))
        .unwrap();
        detail.poll(&ctx);
        let texture = &detail.texture.as_ref().unwrap().1;
        assert_eq!(
            ctx.tex_manager()
                .read()
                .meta(texture.handle.id())
                .unwrap()
                .options,
            ViewerSampling::PixelSharp.texture_options(true)
        );
        detail.clear();
        assert!(detail.sampling == ViewerSampling::PixelSharp);
        assert!(detail.texture.is_none());
    }

    #[test]
    fn detail_textures_use_smooth_filtering_at_all_scales() {
        let ctx = egui::Context::default();
        let mut detail = ViewerDetail::default();
        let visible = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(128.0, 100.0));
        for scale in [0.5, 1.0, 4.0] {
            let region = DetailRegion::for_view(visible, [1280, 10235], scale, 4096);
            let (tx, rx) = mpsc::channel();
            detail.pending = Some(PendingDetail {
                epoch: detail.epoch,
                region,
                rx,
            });
            tx.send(Ok(egui::ColorImage::filled(
                region.texture_size.map(|side| side as usize),
                egui::Color32::GRAY,
            )))
            .unwrap();
            detail.poll(&ctx);
            let texture = &detail.texture.as_ref().unwrap().1;
            assert_eq!(
                ctx.tex_manager()
                    .read()
                    .meta(texture.handle.id())
                    .unwrap()
                    .options,
                egui::TextureOptions::LINEAR
            );
        }
    }

    #[test]
    fn native_zoom_does_not_reuse_almost_native_resampled_detail() {
        let visible = egui::Rect::from_min_max(egui::pos2(0.0, 4000.0), egui::pos2(1280.0, 5000.0));
        let reduced = DetailRegion::for_view(visible, [1280, 10235], 0.99, 4096);
        let native = DetailRegion::for_view(visible, [1280, 10235], 1.0, 4096);
        assert!(!reduced.is_native_resolution());
        assert!(native.is_native_resolution());
        assert!(!reduced.covers(visible, native));
        assert!(reduced.covers(visible, reduced));
        assert!(native.covers(visible, reduced));
    }

    #[test]
    fn deeper_zoom_keeps_native_detail_and_bounded_texture_size() {
        let size = [1280, 10235];
        let source_size = egui::vec2(size[0] as f32, size[1] as f32);
        let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 1200.0));
        let source_center = egui::vec2(640.0, 5000.0);
        let mut previous_area = u64::MAX;
        for zoom in [16.0, 32.0, 64.0, 128.0, 256.0] {
            let scale = viewport.height() / source_size.y * zoom;
            let image_rect = egui::Rect::from_min_size(
                viewport.center() - source_center * scale,
                source_size * scale,
            );
            let visible = visible_source_rect(image_rect, viewport, 0.0, size).unwrap();
            let region = DetailRegion::for_view(visible, size, scale, 4096);
            assert!(region.is_native_resolution());
            assert!(region.bounds.contains_rect(visible));
            let area = region.texture_size[0] as u64 * region.texture_size[1] as u64;
            assert!(area <= previous_area);
            assert!(area * 4 <= 64 * 1024 * 1024);
            previous_area = area;
        }
    }

    #[test]
    fn tall_screenshot_uses_native_pixels_for_zoomed_region() {
        let size = [1280, 10235];
        let image_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 10235.0));
        let viewport =
            egui::Rect::from_min_max(egui::pos2(0.0, 4000.0), egui::pos2(1280.0, 5000.0));
        let visible = visible_source_rect(image_rect, viewport, 0.0, size).unwrap();
        let region = DetailRegion::for_view(visible, size, 1.0, 4096);
        assert_eq!(region.texture_size, [1280, 1300]);
        assert_eq!(region.scale(), 1.0);
        assert!(region.bounds.contains_rect(visible));
        assert!(region.covers(viewport.translate(egui::vec2(0.0, 50.0)), region));
        assert!(!region.covers(viewport.translate(egui::vec2(0.0, 2000.0)), region));
        let rotated_4k =
            egui::Rect::from_min_max(egui::pos2(0.0, 3000.0), egui::pos2(1280.0, 6840.0));
        let region = DetailRegion::for_view(rotated_4k, size, 1.0, 4096);
        assert_eq!(region.scale(), 1.0);
        assert!(region.texture_size[1] <= 4096);
    }

    #[test]
    fn detail_respects_gpu_limits_and_refreshes_after_zoom() {
        let visible =
            egui::Rect::from_min_max(egui::pos2(1000.0, 1000.0), egui::pos2(20000.0, 10000.0));
        let region = DetailRegion::for_view(visible, [30000, 20000], 0.5, 4096);
        assert!(region.texture_size.iter().all(|&side| side <= 4096));
        assert!(
            region.texture_size[0] as u64 * region.texture_size[1] as u64 * 4 <= 64 * 1024 * 1024
        );
        let zoomed =
            egui::Rect::from_min_max(egui::pos2(5000.0, 4000.0), egui::pos2(6000.0, 5000.0));
        let requested = DetailRegion::for_view(zoomed, [30000, 20000], 2.0, 4096);
        assert_eq!(requested.scale(), 1.0);
        assert!(!region.covers(zoomed, requested));
        let limited = DetailRegion::for_view(visible, [30000, 20000], 1.0, 2048);
        assert!(limited.texture_size.iter().all(|&side| side <= 2048));
    }

    #[test]
    fn rotated_viewport_maps_back_to_correct_source_pixels() {
        let image_rect =
            egui::Rect::from_min_size(egui::pos2(50.0, -2000.0), egui::vec2(1280.0, 10235.0));
        let source_rect =
            egui::Rect::from_min_max(egui::pos2(0.0, 4000.0), egui::pos2(1280.0, 5000.0));
        let angle = std::f32::consts::FRAC_PI_2;
        let mut viewport = egui::Rect::NOTHING;
        for point in [
            source_rect.left_top(),
            source_rect.right_top(),
            source_rect.right_bottom(),
            source_rect.left_bottom(),
        ] {
            viewport.extend_with(rotate_point(
                image_rect.min + point.to_vec2(),
                image_rect.center(),
                angle,
            ));
        }
        let actual = visible_source_rect(image_rect, viewport, angle, [1280, 10235]).unwrap();
        assert!((actual.min - source_rect.min).length() < 0.01);
        assert!((actual.max - source_rect.max).length() < 0.01);
        let offscreen = viewport.translate(egui::vec2(30000.0, 0.0));
        assert!(visible_source_rect(image_rect, offscreen, angle, [1280, 10235]).is_none());
    }

    #[test]
    fn transparent_preview_and_detail_never_overlap() {
        let detail = egui::Rect::from_min_max(egui::pos2(0.2, 0.3), egui::pos2(0.8, 0.7));
        let preview = preview_regions(detail);
        assert!(
            (preview.iter().map(|r| r.area()).sum::<f32>() + detail.area() - 1.0).abs() < 0.0001
        );
        for (i, region) in preview.iter().enumerate() {
            assert!(!region.intersect(detail).is_positive());
            for other in preview.iter().skip(i + 1) {
                assert!(!region.intersect(*other).is_positive());
            }
        }
    }

    #[test]
    fn native_crop_preserves_pixels_reuses_decode_and_invalidates_edits() {
        let dir = std::env::temp_dir().join(format!(
            "iris-detail-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("screenshot.png");
        let original = image::RgbaImage::from_fn(1280, 10235, |x, y| {
            image::Rgba([(x % 251) as u8, (y % 253) as u8, ((x + y) % 255) as u8, 180])
        });
        original.save(&path).unwrap();
        let mut source = DetailSource {
            path,
            revision: 0,
            size: [1280, 10235],
        };
        let mut decoder = DetailDecoder::default();
        for top in [4000, 9000] {
            let visible = egui::Rect::from_min_max(
                egui::pos2(0.0, top as f32),
                egui::pos2(1280.0, (top + 500) as f32),
            );
            let region = DetailRegion::for_view(visible, source.size, 1.0, 4096);
            let cached_pointer = decoder
                .cached
                .as_ref()
                .map(|(_, image)| image.as_bytes().as_ptr());
            let crop = decoder.crop(&source, region).unwrap();
            for (x, y) in [(0, 0), (1279, 123), (100, 499)] {
                let pixel = original
                    .get_pixel(
                        x + region.bounds.min.x as u32,
                        y + region.bounds.min.y as u32,
                    )
                    .0;
                assert_eq!(
                    crop[(x as usize, y as usize)],
                    egui::Color32::from_rgba_unmultiplied(pixel[0], pixel[1], pixel[2], pixel[3])
                );
            }
            if let Some(pointer) = cached_pointer {
                assert_eq!(
                    decoder.cached.as_ref().unwrap().1.as_bytes().as_ptr(),
                    pointer
                );
            }
        }
        image::RgbaImage::from_pixel(1280, 10235, image::Rgba([20, 40, 80, 255]))
            .save(&source.path)
            .unwrap();
        source.revision += 1;
        let region = DetailRegion::for_view(
            egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(128.0, 128.0)),
            source.size,
            1.0,
            4096,
        );
        let crop = decoder.crop(&source, region).unwrap();
        assert_eq!(crop[(50, 50)], egui::Color32::from_rgb(20, 40, 80));

        // Exercise the real asynchronous renderer without a window or GPU driver.
        let ctx = egui::Context::default();
        ctx.begin_pass(egui::RawInput {
            max_texture_side: Some(4096),
            ..Default::default()
        });
        let _ = ctx.end_pass();
        let preview = ctx.load_texture(
            "test_preview",
            egui::ColorImage::filled([512, 4096], egui::Color32::BLUE),
            egui::TextureOptions::LINEAR,
        );
        let viewport =
            egui::Rect::from_min_max(egui::pos2(0.0, 4000.0), egui::pos2(1280.0, 5000.0));
        let image_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 10235.0));
        let mut detail = ViewerDetail::default();
        let diagnostics = DiagnosticState::new();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            ctx.begin_pass(egui::RawInput {
                screen_rect: Some(viewport),
                ..Default::default()
            });
            let painter = ctx
                .layer_painter(egui::LayerId::background())
                .with_clip_rect(viewport);
            detail.paint(
                &painter,
                &source.path,
                source.revision,
                source.size,
                &preview,
                image_rect,
                0.0,
                &diagnostics,
            );
            let _ = ctx.end_pass();
            if detail.texture.is_some() {
                break;
            }
            assert!(Instant::now() < deadline, "detail renderer did not finish");
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            detail.texture.as_ref().unwrap().1.handle.size(),
            [1280, 1300]
        );
        assert_eq!(
            ctx.tex_manager()
                .read()
                .meta(detail.texture.as_ref().unwrap().1.handle.id())
                .unwrap()
                .options,
            egui::TextureOptions::LINEAR
        );
        let detail_id = detail.texture.as_ref().unwrap().1.handle.id();
        detail.set_sampling(ViewerSampling::PixelSharp);
        for _ in 0..3 {
            ctx.begin_pass(egui::RawInput {
                screen_rect: Some(viewport),
                ..Default::default()
            });
            let painter = ctx
                .layer_painter(egui::LayerId::background())
                .with_clip_rect(viewport);
            detail.paint(
                &painter,
                &source.path,
                source.revision,
                source.size,
                &preview,
                image_rect,
                0.0,
                &diagnostics,
            );
            assert!(
                detail.pending.is_none(),
                "unchanged view should reuse detail"
            );
            let _ = ctx.end_pass();
        }
        assert_eq!(detail.texture.as_ref().unwrap().1.handle.id(), detail_id);
        detail.clear();
        assert!(detail.texture.is_none());
        assert!(detail.decoder.lock().unwrap().cached.is_none());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn leaving_image_discards_pending_detail_without_starting_another_decode() {
        let source = DetailSource {
            path: PathBuf::from("screenshot.png"),
            revision: 0,
            size: [1280, 10235],
        };
        let region = DetailRegion::for_view(
            egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(100.0, 100.0)),
            source.size,
            1.0,
            4096,
        );
        let (tx, rx) = mpsc::channel();
        let mut detail = ViewerDetail::default();
        detail.source = Some(source.clone());
        detail.pending = Some(PendingDetail {
            epoch: detail.epoch,
            region,
            rx,
        });
        detail.clear();
        assert!(detail.pending.is_some());
        detail.source = Some(source);
        tx.send(Ok(egui::ColorImage::filled([1, 1], egui::Color32::RED)))
            .unwrap();
        detail.poll(&egui::Context::default());
        assert!(detail.texture.is_none());
        assert!(detail.pending.is_none());
    }
}
