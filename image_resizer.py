import os
from PIL import Image
import io

# CONFIGURATION
source_folder = "source_images"         # folder where original images are located
destination_folder = "formatted_images" # folder to save formatted images
min_size_kb = 5
max_size_kb = 20
required_width = 160
min_height = 200
max_height = 212

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Resize and convert images
for filename in os.listdir(source_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
        img_path = os.path.join(source_folder, filename)
        with Image.open(img_path) as img:
            # Convert to RGB (for JPEG)
            img = img.convert("RGB")
            
            # Resize height between 200-212 (keeping width 160)
            aspect_ratio = img.height / img.width
            for height in range(min_height, max_height + 1):
                resized_img = img.resize((required_width, height))
                
                # Save to a memory buffer to check size
                buffer = io.BytesIO()
                resized_img.save(buffer, format="JPEG", quality=85)
                size_kb = buffer.tell() / 1024

                # Check file size
                if min_size_kb <= size_kb <= max_size_kb:
                    output_path = os.path.join(destination_folder, f"{os.path.splitext(filename)[0]}.jpg")
                    with open(output_path, "wb") as f:
                        f.write(buffer.getvalue())
                    print(f"Saved: {output_path} | Size: {size_kb:.2f} KB | Height: {height}px")
                    break
            else:
                print(f"Could not resize {filename} within size constraints.")
