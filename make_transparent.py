import sys
import os

try:
    from PIL import Image
except ImportError:
    print("PIL/Pillow not found. Installing pillow via pip...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])
    from PIL import Image

def make_transparent(img_path, output_path):
    print(f"Loading image from {img_path}...")
    img = Image.open(img_path).convert("RGBA")
    data = img.getdata()
    
    new_data = []
    for item in data:
        r, g, b, a = item
        # If R, G, B are all very close to white, we make them transparent
        # We smoothly scale the transparency between 240 and 255
        if r > 240 and g > 240 and b > 240:
            min_val = min(r, g, b)
            alpha = int((255 - min_val) * (255.0 / 15.0))
            if alpha < 0:
                alpha = 0
            if alpha > 255:
                alpha = 255
            new_data.append((r, g, b, alpha))
        else:
            new_data.append(item)
            
    img.putdata(new_data)
    print(f"Saving transparent PNG to {output_path}...")
    img.save(output_path, "PNG")
    print("Done!")

if __name__ == "__main__":
    src = "/home/lewis/.gemini/antigravity-cli/brain/be3bc4a9-53a8-4547-885a-3e66f330a8ac/iris_app_icon_white_1779567533978.png"
    dst = "/home/lewis/Dev/iris/icon.png"
    make_transparent(src, dst)
