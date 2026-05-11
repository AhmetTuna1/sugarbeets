import os
import shutil
import random
import zipfile

# Kaynak Klasörler
images_dir = "annotations_cropweed_parts/WORKSET/images"
masks_dir = "annotations_cropweed_parts/WORKSET/masks"  # Orijinal 16-bit maskeler!

# Hedef Klasör
base_split = "annotations_cropweed_parts/WORKSET/DATASET_SPLIT"

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

image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(".png") and os.path.exists(os.path.join(masks_dir, f))]
random.seed(42)
random.shuffle(image_files)

split_idx = int(len(image_files) * 0.8)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

print("Dosyalar kopyalanıyor... Lütfen bekleyin.")

for f in train_files:
    shutil.copy2(os.path.join(images_dir, f), os.path.join(train_img, f))
    shutil.copy2(os.path.join(masks_dir, f), os.path.join(train_msk, f))

for f in val_files:
    shutil.copy2(os.path.join(images_dir, f), os.path.join(val_img, f))
    shutil.copy2(os.path.join(masks_dir, f), os.path.join(val_msk, f))

print(f"Split tamamlandı: {len(train_files)} train, {len(val_files)} val.")

# ZIP İşlemi
zip_path = "driveSugarBeets_MultiClass.zip"
print(f"Yeni ZIP dosyası oluşturuluyor: {zip_path} ...")

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(base_split):
        for file in files:
            file_path = os.path.join(root, file)
            # base_split'in içini ZIP'in root'u olarak kaydet
            arcname = os.path.relpath(file_path, base_split)
            zipf.write(file_path, arcname)

print("Her şey hazır! Yeni ZIP dosyanız ana dizinde oluşturuldu.")
