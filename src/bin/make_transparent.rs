use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let src_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("icon_source.png"));
    let dst_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("icon.png"));

    println!("Loading image bytes from: {}", src_path.display());
    let bytes = std::fs::read(&src_path)?;
    let img = image::load_from_memory(&bytes)?;
    let mut rgba_img = img.to_rgba8();

    for pixel in rgba_img.pixels_mut() {
        let r = pixel[0];
        let g = pixel[1];
        let b = pixel[2];

        if r > 240 && g > 240 && b > 240 {
            let min_val = r.min(g).min(b);
            let alpha = ((255 - min_val) as f32 * (255.0 / 15.0)) as u8;
            pixel[3] = alpha;
        }
    }

    println!("Saving transparent PNG to: {}", dst_path.display());
    rgba_img.save(&dst_path)?;
    println!("Done.");
    Ok(())
}
