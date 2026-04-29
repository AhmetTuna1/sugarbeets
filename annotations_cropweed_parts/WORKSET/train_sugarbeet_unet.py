import os
import zipfile
import random
import warnings

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

warnings.filterwarnings("ignore")


# =========================================================
# 1) AYARLAR
# =========================================================

# Proje kök dizini (WORKSET)
WORKSET_DIR = r"C:\Users\90534\OneDrive\Masaüstü\Sugarbeets\annotations_cropweed_parts\WORKSET"

# ZIP dosyanın yolu
ZIP_PATH = os.path.join(WORKSET_DIR, "DATASET_SPLIT_zip.zip")
# ZIP açıldıktan sonra dosyaların çıkarılacağı yer
EXTRACT_DIR = WORKSET_DIR

# Dataset kök dizini
BASE_PATH = os.path.join(EXTRACT_DIR, "DATASET_SPLIT")

# Eğitim ayarları
IMG_SIZE = 256
BATCH_SIZE = 16
NUM_WORKERS = 0
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
SEED = 42

# Model kayıt yolu
MODEL_SAVE_PATH = os.path.join(WORKSET_DIR, "best_unet_sugarbeets.pth")

# Mask uzantısı varsayımı
MASK_EXTENSION = ".png"

# Dataset zaten mevcut olduğu için zip açmaya gerek yok
UNZIP_DATASET = False


# =========================================================
# 2) SABİT TOHUMLAR
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =========================================================
# 3) CİHAZ
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =========================================================
# 4) ZIP AÇMA
# =========================================================
def unzip_dataset(zip_path, extract_dir):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP dosyası bulunamadı: {zip_path}")

    print(f"\nZIP açılıyor:\n{zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print("ZIP açıldı.")


if UNZIP_DATASET:
    unzip_dataset(ZIP_PATH, EXTRACT_DIR)


# =========================================================
# 5) KLASÖR YOLLARI
# =========================================================
train_img_dir = os.path.join(BASE_PATH, "train", "images")
train_mask_dir = os.path.join(BASE_PATH, "train", "masks")
val_img_dir = os.path.join(BASE_PATH, "val", "images")
val_mask_dir = os.path.join(BASE_PATH, "val", "masks")

required_dirs = [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]

print("\nKlasör kontrolü:")
for d in required_dirs:
    print(d, "->", os.path.exists(d))

for d in required_dirs:
    if not os.path.exists(d):
        raise FileNotFoundError(f"Gerekli klasör bulunamadı: {d}")


# =========================================================
# 6) DOSYA LİSTELEME
# =========================================================
def list_files(folder):
    return sorted([
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ])


train_images = list_files(train_img_dir)
train_masks = list_files(train_mask_dir)
val_images = list_files(val_img_dir)
val_masks = list_files(val_mask_dir)

print("\nDosya sayıları:")
print("Train images:", len(train_images))
print("Train masks :", len(train_masks))
print("Val images  :", len(val_images))
print("Val masks   :", len(val_masks))

print("\nİlk 5 train image:", train_images[:5])
print("İlk 5 train mask :", train_masks[:5])
print("İlk 5 val image  :", val_images[:5])
print("İlk 5 val mask   :", val_masks[:5])


# =========================================================
# 7) IMAGE-MASK EŞLEŞME KONTROLÜ
# =========================================================
def stem(filename):
    return os.path.splitext(filename)[0]


train_img_stems = set(stem(f) for f in train_images)
train_mask_stems = set(stem(f) for f in train_masks)
val_img_stems = set(stem(f) for f in val_images)
val_mask_stems = set(stem(f) for f in val_masks)

print("\nEşleşme kontrolü:")
print("Train eşleşmeyen image sayısı:", len(train_img_stems - train_mask_stems))
print("Train eşleşmeyen mask sayısı :", len(train_mask_stems - train_img_stems))
print("Val eşleşmeyen image sayısı  :", len(val_img_stems - val_mask_stems))
print("Val eşleşmeyen mask sayısı   :", len(val_mask_stems - val_img_stems))

if train_img_stems != train_mask_stems:
    missing_masks = sorted(list(train_img_stems - train_mask_stems))[:10]
    extra_masks = sorted(list(train_mask_stems - train_img_stems))[:10]
    raise ValueError(
        f"Train image-mask eşleşmesi bozuk!\n"
        f"Maskesi eksik ilk örnekler: {missing_masks}\n"
        f"Fazla mask ilk örnekler: {extra_masks}"
    )

if val_img_stems != val_mask_stems:
    missing_masks = sorted(list(val_img_stems - val_mask_stems))[:10]
    extra_masks = sorted(list(val_mask_stems - val_img_stems))[:10]
    raise ValueError(
        f"Val image-mask eşleşmesi bozuk!\n"
        f"Maskesi eksik ilk örnekler: {missing_masks}\n"
        f"Fazla mask ilk örnekler: {extra_masks}"
    )

print("Image-mask eşleşmesi başarılı.")


# =========================================================
# 8) MASK DEĞER KONTROLÜ
# =========================================================
sample_mask_path = os.path.join(train_mask_dir, train_masks[0])
try:
    sample_mask = np.array(Image.open(sample_mask_path).convert("L"))
except Exception as e:
    raise ValueError(f"Örnek maske okunamadı: {sample_mask_path} -> {e}")

print("\nÖrnek mask unique değerleri:", np.unique(sample_mask))


# =========================================================
# 9) AUGMENTATION
# =========================================================
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.10,
        rotate_limit=15,
        border_mode=0,  # cv2.BORDER_CONSTANT
        p=0.3
    ),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# =========================================================
# 10) DATASET
# =========================================================
class SugarBeetDataset(Dataset):
    def __init__(self, img_dir, mask_dir, image_files, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = os.path.splitext(img_name)[0] + MASK_EXTENSION

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            raise ValueError(f"Görüntü okunamadı: {img_path} -> {e}")

        try:
            mask = np.array(Image.open(mask_path).convert("L"))
        except Exception as e:
            raise ValueError(f"Maske okunamadı: {mask_path} -> {e}")

        mask = (mask > 0).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)

        mask = mask.unsqueeze(0).float()
        return image, mask


train_dataset = SugarBeetDataset(
    img_dir=train_img_dir,
    mask_dir=train_mask_dir,
    image_files=train_images,
    transform=train_transform
)

val_dataset = SugarBeetDataset(
    img_dir=val_img_dir,
    mask_dir=val_mask_dir,
    image_files=val_images,
    transform=val_transform
)


# =========================================================
# 11) DATALOADER
# =========================================================
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print("\nDataset özeti:")
print("Train örnek sayısı:", len(train_dataset))
print("Val örnek sayısı  :", len(val_dataset))
print("Train batch sayısı:", len(train_loader))
print("Val batch sayısı  :", len(val_loader))


# =========================================================
# 12) ÖRNEK BATCH GÖSTER
# =========================================================
images, masks = next(iter(train_loader))
print("\nBatch shape kontrolü:")
print("Image batch shape:", images.shape)
print("Mask batch shape :", masks.shape)

sample_img = images[0].permute(1, 2, 0).cpu().numpy()
sample_img = np.clip(
    sample_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]),
    0, 1
)
sample_mask = masks[0].squeeze().cpu().numpy()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(sample_img)
plt.title("Sample Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(sample_mask, cmap="gray")
plt.title("Sample Mask")
plt.axis("off")
plt.tight_layout()
# plt.show()  # Yorum satırı yapıldı, işlemi durdurmaması için.
plt.savefig(os.path.join(WORKSET_DIR, "sample_batch.png"))
plt.close()


# =========================================================
# 13) MODEL
# =========================================================
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).to(device)

print("\nModel hazır.")


# =========================================================
# 14) LOSS / OPTIMIZER / SCHEDULER
# =========================================================
dice_loss = smp.losses.DiceLoss(mode="binary")
bce_loss = nn.BCEWithLogitsLoss()

def loss_fn(pred, target):
    return 0.5 * bce_loss(pred, target) + 0.5 * dice_loss(pred, target)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# =========================================================
# 15) METRİK
# =========================================================
def compute_iou(pred, target, threshold=0.5, eps=1e-7):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


# =========================================================
# 16) TRAIN / VAL FONKSİYONLARI
# =========================================================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    running_iou = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        iou_val = compute_iou(outputs, masks)
        running_iou += iou_val
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "IoU": f"{iou_val:.4f}"})

    return running_loss / len(loader), running_iou / len(loader)


def validate_one_epoch(model, loader, device):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0

    pbar = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            running_loss += loss.item()
            iou_val = compute_iou(outputs, masks)
            running_iou += iou_val
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "IoU": f"{iou_val:.4f}"})

    return running_loss / len(loader), running_iou / len(loader)


# =========================================================
# 17) EĞİTİM
# =========================================================
best_val_iou = 0.0
train_losses = []
val_losses = []
train_ious = []
val_ious = []

print("\nEğitim başlıyor...\n")

for epoch in range(NUM_EPOCHS):
    train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, device)
    val_loss, val_iou = validate_one_epoch(model, val_loader, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_ious.append(train_iou)
    val_ious.append(val_iou)

    scheduler.step(val_iou)

    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   IoU: {val_iou:.4f}")
    print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    print("-" * 50)

    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"En iyi model kaydedildi. Val IoU: {best_val_iou:.4f}")

print("\nEğitim tamamlandı.")
print("Best Val IoU:", best_val_iou)
print("Model kayıt yolu:", MODEL_SAVE_PATH)


# =========================================================
# 18) EN İYİ MODELİ YÜKLE
# =========================================================
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()
print("\nEn iyi model yüklendi.")


# =========================================================
# 19) GRAFİKLER
# =========================================================
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_ious, label="Train IoU")
plt.plot(val_ious, label="Val IoU")
plt.title("IoU")
plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.legend()
plt.grid(True)

plt.tight_layout()
# plt.show() # Yorum satırı yapıldı
plt.savefig(os.path.join(WORKSET_DIR, "training_metrics.png"))
plt.close()


# =========================================================
# 20) TAHMİN GÖSTERİMİ
# =========================================================
def denormalize_image(img_tensor):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    return img


def show_predictions(model, dataset, device, num_samples=5):
    model.eval()

    num_samples = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

    plt.figure(figsize=(12, 4 * num_samples))

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = dataset[idx]
            input_tensor = image.unsqueeze(0).to(device)

            output = model(input_tensor)
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_binary = (pred > 0.5).astype(np.float32)

            img_np = denormalize_image(image)
            mask_np = mask.squeeze().cpu().numpy()

            plt.subplot(num_samples, 3, i * 3 + 1)
            plt.imshow(img_np)
            plt.title("Image")
            plt.axis("off")

            plt.subplot(num_samples, 3, i * 3 + 2)
            plt.imshow(mask_np, cmap="gray")
            plt.title("Ground Truth")
            plt.axis("off")

            plt.subplot(num_samples, 3, i * 3 + 3)
            plt.imshow(pred_binary, cmap="gray")
            plt.title("Prediction")
            plt.axis("off")

    plt.tight_layout()
    # plt.show() # Yorum satırı yapıldı
    plt.savefig(os.path.join(WORKSET_DIR, "predictions.png"))
    plt.close()


show_predictions(model, val_dataset, device, num_samples=5)