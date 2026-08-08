use eframe::egui;
use image::DynamicImage;

pub(crate) const IMAGE_VIEWER_TOP_BAR_HEIGHT: f32 = 32.0;
pub(crate) const INITIAL_IMAGE_DISPLAY_HEIGHT: f32 = 1200.0;
pub(crate) const MAX_VIEWER_TEXTURE_DIMENSION: u32 = 4096;

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
