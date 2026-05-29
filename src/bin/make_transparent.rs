fn main() -> Result<(), Box<dyn std::error::Error>> {
    let src_path = "/home/lewis/.gemini/antigravity-cli/brain/be3bc4a9-53a8-4547-885a-3e66f330a8ac/iris_app_icon_white_1779567533978.png";
    let dst_path = "/home/lewis/Dev/iris/icon.png";
    
    println!("Loading image bytes from: {}", src_path);
    let bytes = std::fs::read(src_path)?;
    println!("File size: {} bytes", bytes.len());
    let sig_len = std::cmp::min(bytes.len(), 16);
    println!("First {} bytes: {:02X?}", sig_len, &bytes[0..sig_len]);
    
    println!("Decoding image content (detecting format from header)...");
    let img = image::load_from_memory(&bytes)?;
    let mut rgba_img = img.to_rgba8();
    
    println!("Processing pixels for transparency...");
    for pixel in rgba_img.pixels_mut() {
        let r = pixel[0];
        let g = pixel[1];
        let b = pixel[2];
        
        // If it's close to pure white, fade the alpha channel smoothly
        if r > 240 && g > 240 && b > 240 {
            let min_val = r.min(g).min(b);
            let alpha = ((255 - min_val) as f32 * (255.0 / 15.0)) as u8;
            pixel[3] = alpha;
        }
    }
    
    println!("Saving transparent PNG to: {}", dst_path);
    rgba_img.save(dst_path)?;
    println!("Done! Transparent icon saved successfully.");
    
    Ok(())
}
