"""
YOLOv26s Multi-Class Object Detection Eğitim Scripti (Colab-Ready)
=================================================================
Bu script, orijinal (Bonn) maskelerden hem 'Yabancı Ot' (Weed) hem de
'Şeker Pancarı' (Crop) nesnelerini tespit edip 2 sınıflı YOLOv26s eğitimi başlatır.
"""

import os
import cv2
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


# ======================== AYARLAR ========================
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    DATASET_ROOT = "/content/dataset"
    RESULTS_DIR = "/content/drive/MyDrive/SugarBeetsProject/yolo_results"
    DEVICE = 0
    WORKERS = 8
    BATCH = 64
else:
    DATASET_ROOT = r"C:\Users\90534\OneDrive\Masaüstü\Sugarbeets\annotations_cropweed_parts\WORKSET\DATASET_SPLIT"
    RESULTS_DIR = "runs"
    DEVICE = 0
    WORKERS = 0
    BATCH = 16

CLASS_NAMES = {0: "weed", 1: "crop"}


# ======================== ADIM 1: MASK → MULTI-CLASS BBOX ========================
def mask_to_multiclass_bboxes(mask_path):
    try:
        mask_pil = Image.open(mask_path)
        mask = np.array(mask_pil)
    except Exception as e:
        print(f"[UYARI] Mask okunamadı: {mask_path}")
        return []
    
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
        
    H, W = mask.shape
    bboxes = []
    
    # --- Sınıf 0: Yabancı Ot (Weed) ---
    weed_binary = np.where(mask == 10000, 255, 0).astype(np.uint8)
    contours_weed, _ = cv2.findContours(weed_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours_weed:
        if cv2.contourArea(contour) < 25:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        xc, yc = (x + w / 2) / W, (y + h / 2) / H
        bboxes.append(f"0 {xc:.6f} {yc:.6f} {w/W:.6f} {h/H:.6f}")

    # --- Sınıf 1: Şeker Pancarı (Crop) ---
    crop_binary = np.where((mask > 0) & (mask < 10000), 255, 0).astype(np.uint8)
    contours_crop, _ = cv2.findContours(crop_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours_crop:
        if cv2.contourArea(contour) < 25:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        xc, yc = (x + w / 2) / W, (y + h / 2) / H
        bboxes.append(f"1 {xc:.6f} {yc:.6f} {w/W:.6f} {h/H:.6f}")
        
    return bboxes


def convert_masks_to_multiclass_labels(mask_dir, label_dir):
    # Eğer label'lar varsa atla
    if os.path.exists(label_dir) and len(os.listdir(label_dir)) > 0:
        return

    os.makedirs(label_dir, exist_ok=True)
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith('.png')])
    
    total_weeds = 0
    total_crops = 0
    empty_labels = 0
    
    for mask_file in tqdm(mask_files, desc=f"Multi-Class Bbox dönüşüm: {os.path.basename(os.path.dirname(mask_dir))}"):
        mask_path = os.path.join(mask_dir, mask_file)
        label_file = os.path.splitext(mask_file)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)
        
        bbox_lines = mask_to_multiclass_bboxes(mask_path)
        
        with open(label_path, 'w') as f:
            if bbox_lines:
                f.write("\n".join(bbox_lines) + "\n")
                total_weeds += sum(1 for line in bbox_lines if line.startswith("0"))
                total_crops += sum(1 for line in bbox_lines if line.startswith("1"))
            else:
                empty_labels += 1
                
    print(f"  → İşlem Tamam! {total_weeds} Ot (Weed), {total_crops} Pancar (Crop) kutusu üretildi. {empty_labels} boş.")


# ======================== ADIM 2: YAML OLUŞTURMA ========================
def create_multiclass_yaml(dataset_root):
    yaml_path = os.path.join(dataset_root, "sugarbeet_multiclass.yaml")
    
    config = {
        'path': dataset_root,
        'train': 'train/images',
        'val': 'val/images',
        'names': CLASS_NAMES
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return yaml_path


def fix_dataset_paths(dataset_dir):
    """
    Colab'da zip açılırken Windows (backslash \\) yollarının düz dosya 
    olarak çıkmasını (train\\masks\\resim.png) düzeltip klasörlere ayırır.
    """
    import shutil
    if not os.path.exists(dataset_dir): return
    for filename in os.listdir(dataset_dir):
        if "\\" in filename:
            parts = filename.split("\\")
            target_dir = os.path.join(dataset_dir, *parts[:-1])
            os.makedirs(target_dir, exist_ok=True)
            shutil.move(os.path.join(dataset_dir, filename), os.path.join(target_dir, parts[-1]))

# ======================== ADIM 3: EĞİTİM ========================
def train_multiclass_detection():
    print("=" * 60)
    print("  YOLOv26s MULTI-CLASS Object Detection (Weed & Crop)")
    print("=" * 60)
    
    # 0. Zip yol düzeltme (Eğer \ karakteriyle dosya çıktıysa düzeltir)
    if IN_COLAB:
        print("[BİLGİ] Veriseti klasör yapısı kontrol ediliyor...")
        fix_dataset_paths(DATASET_ROOT)
    
    train_mask_dir = os.path.join(DATASET_ROOT, "train", "masks")
    train_label_dir = os.path.join(DATASET_ROOT, "train", "labels")
    val_mask_dir = os.path.join(DATASET_ROOT, "val", "masks")
    val_label_dir = os.path.join(DATASET_ROOT, "val", "labels")
    
    convert_masks_to_multiclass_labels(train_mask_dir, train_label_dir)
    convert_masks_to_multiclass_labels(val_mask_dir, val_label_dir)
    
    yaml_path = create_multiclass_yaml(DATASET_ROOT)
    
    # AUTO-RESUME
    last_pt_path = os.path.join(RESULTS_DIR, "sugarbeet_multiclass_yolov26s", "weights", "last.pt")
    
    if os.path.exists(last_pt_path):
        print(f"\n[BİLGİ] Önceki eğitim bulundu! Kaldığı epochtan devam ediliyor...\n")
        model = YOLO(last_pt_path)
        results = model.train(resume=True)
    else:
        print(f"\n[BİLGİ] YOLO11s ile Sıfırdan yeni bir eğitim başlatılıyor...\n")
        # YOLOv26 henüz icat edilmediği için en son teknoloji olan YOLO11 kullanıyoruz.
        model = YOLO("yolo11s.pt")
        results = model.train(
            data=yaml_path,
            epochs=100,
            patience=20,
            imgsz=640,
            batch=BATCH,
            cache=True,
            device=DEVICE,
            project=RESULTS_DIR,
            name="sugarbeet_multiclass_yolov26s",
            workers=WORKERS,
            degrees=180.0,
            flipud=0.5,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            cos_lr=True,
        )
    
    print("\n" + "=" * 60)
    print("  Eğitim tamamlandı!")
    print(f"  Sonuçlar: {RESULTS_DIR}/sugarbeet_multiclass_yolov26s/")
    print("=" * 60)


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    train_multiclass_detection()
