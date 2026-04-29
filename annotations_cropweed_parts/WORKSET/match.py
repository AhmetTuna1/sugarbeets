import os
import shutil

images_dir = r"images"
masks_dir = r"masks"

out_img = r"DATASET_READY\images"
out_mask = r"DATASET_READY\masks"

if os.path.exists("DATASET_READY"):
    shutil.rmtree("DATASET_READY")


os.makedirs(out_img, exist_ok=True)
os.makedirs(out_mask, exist_ok=True)

image_files = [f for f in os.listdir(images_dir) if f.endswith(".png")]
mask_files = [f for f in os.listdir(masks_dir) if f.endswith(".png")]

print("Images:", len(image_files))
print("Masks:", len(mask_files))

count = 0

for img in image_files:
    base = img.split(".")[0]

    # mask içinde bu base ile başlayan dosyayı bul
    candidates = [m for m in mask_files if m.startswith(base)]

    if len(candidates) > 0:
        mask = candidates[0]

        shutil.copy2(os.path.join(images_dir, img),
                     os.path.join(out_img, img))

        shutil.copy2(os.path.join(masks_dir, mask),
                     os.path.join(out_mask, img))   # mask adını image ile aynı yapıyoruz

        count += 1

print("Matched:", count)