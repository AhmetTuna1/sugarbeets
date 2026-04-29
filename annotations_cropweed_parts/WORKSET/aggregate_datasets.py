import os
import shutil

source_base = r"..\ijrr_sugarbeets_2016_annotations"
dest_images = r"images"
dest_masks = r"masks"

os.makedirs(dest_images, exist_ok=True)
os.makedirs(dest_masks, exist_ok=True)

img_count = 0
mask_count = 0

for folder in os.listdir(source_base):
    if folder.startswith("CKA_"):
        folder_path = os.path.join(source_base, folder)
        img_src = os.path.join(folder_path, "images", "rgb")
        ann_src = os.path.join(folder_path, "annotations", "dlp", "iMapCleaned")
        
        if os.path.exists(img_src):
            for f in os.listdir(img_src):
                if f.lower().endswith(".png"):
                    shutil.copy2(os.path.join(img_src, f), os.path.join(dest_images, f))
                    img_count += 1
                    
        if os.path.exists(ann_src):
            for f in os.listdir(ann_src):
                if f.lower().endswith(".png"):
                    shutil.copy2(os.path.join(ann_src, f), os.path.join(dest_masks, f))
                    mask_count += 1

print(f"Aggregation complete. Copied {img_count} images and {mask_count} masks.")
