import os
import shutil
import random

images_dir = r"DATASET_READY\images"
masks_dir = r"DATASET_READY\binary_masks"

base_split = r"DATASET_SPLIT"

if os.path.exists(base_split):
    shutil.rmtree(base_split)


train_img = os.path.join(base_split, "train", "images")
train_msk = os.path.join(base_split, "train", "masks")

val_img = os.path.join(base_split, "val", "images")
val_msk = os.path.join(base_split, "val", "masks")

os.makedirs(train_img, exist_ok=True)
os.makedirs(train_msk, exist_ok=True)
os.makedirs(val_img, exist_ok=True)
os.makedirs(val_msk, exist_ok=True)

image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(".png")]
random.shuffle(image_files)

split_idx = int(len(image_files) * 0.8)

train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

for f in train_files:
    shutil.copy2(os.path.join(images_dir, f), os.path.join(train_img, f))
    shutil.copy2(os.path.join(masks_dir, f), os.path.join(train_msk, f))

for f in val_files:
    shutil.copy2(os.path.join(images_dir, f), os.path.join(val_img, f))
    shutil.copy2(os.path.join(masks_dir, f), os.path.join(val_msk, f))

print(f"Split done: {len(train_files)} train, {len(val_files)} val.")
