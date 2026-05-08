"""
YOLOv8s Object Detection Eğitim Scripti (Colab-Ready)
=====================================================
Segmentation mask'larından bounding box label'ları üretir ve
YOLOv8s detection modeli eğitir.

Colab'da Kullanım:
    1) DATASET_SPLIT_zip.zip dosyasını Drive'a yükle
    2) Bu scripti Colab'a kopyala ve çalıştır

Colab Hücre Sıralaması:
    Hücre 1: Drive mount & zip aç
        from google.colab import drive
        drive.mount('/content/drive')
        !unzip -q /content/drive/MyDrive/DATASET_SPLIT_zip.zip -d /content/

    Hücre 2: Gereklilikleri kur
        !pip install ultralytics opencv-python-headless

    Hücre 3: Bu scripti çalıştır
        %run train_yolo_det.py
"""

import os
import cv2
import yaml
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


# ======================== AYARLAR ========================
# Ortam tespiti
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    DATASET_ROOT = "/content/DATASET_SPLIT"
    RESULTS_DIR = "/content/drive/MyDrive/SugarBeetsProject/yolo_results"
    DEVICE = 0
    WORKERS = 8
    BATCH = 64          # A100/L4 için ideal (hata verirse 32'ye düşür)
else:
    DATASET_ROOT = r"C:\Users\90534\OneDrive\Masaüstü\Sugarbeets\annotations_cropweed_parts\WORKSET\DATASET_SPLIT"
    RESULTS_DIR = "runs"
    DEVICE = 0
    WORKERS = 0          # Windows için
    BATCH = 16           # RTX 3050 Ti 4GB için güvenli

CLASS_ID = 0              # Tek sınıf: weed (ot)
CLASS_NAME = "weed"


# ======================== ADIM 1: MASK → BBOX DÖNÜŞÜMÜ ========================
def mask_to_bboxes(mask_path, class_id=0):
    """
    Binary mask'tan YOLO formatında bounding box'lar üretir.
    Her bağımsız ot bölgesi için ayrı bir bbox oluşturur.
    
    Returns:
        list of str: Her satır "class_id x_center y_center width height" formatında
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[UYARI] Mask okunamadı: {mask_path}")
        return []
    
    H, W = mask.shape
    
    # Binary threshold
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Her bağımsız ot bölgesini bul
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    lines = []
    for contour in contours:
        # Çok küçük konturları atla (gürültü filtresi)
        area = cv2.contourArea(contour)
        if area < 25:  # 5x5 pikselden küçük nesneleri atla
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # YOLO formatına normalize et (x_center, y_center, width, height)
        x_center = (x + w / 2) / W
        y_center = (y + h / 2) / H
        norm_w = w / W
        norm_h = h / H
        
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
    
    return lines


def convert_masks_to_bbox_labels(mask_dir, label_dir, class_id=0):
    """Bir klasördeki tüm mask'ları YOLO bbox label dosyalarına çevirir."""
    os.makedirs(label_dir, exist_ok=True)
    
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith('.png')])
    
    total_boxes = 0
    empty_labels = 0
    
    for mask_file in tqdm(mask_files, desc=f"Bbox dönüşüm: {os.path.basename(os.path.dirname(mask_dir))}"):
        mask_path = os.path.join(mask_dir, mask_file)
        label_file = os.path.splitext(mask_file)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)
        
        bbox_lines = mask_to_bboxes(mask_path, class_id)
        
        # Boş olsa bile dosya oluştur (YOLO boş label = negatif örnek)
        with open(label_path, 'w') as f:
            if bbox_lines:
                f.write("\n".join(bbox_lines) + "\n")
                total_boxes += len(bbox_lines)
            else:
                empty_labels += 1
    
    print(f"  → {len(mask_files)} mask işlendi, {total_boxes} bbox üretildi, {empty_labels} boş label (negatif örnek)")
    return total_boxes


# ======================== ADIM 2: YAML OLUŞTURMA ========================
def create_detection_yaml(dataset_root):
    """Object detection için YOLO yaml dosyası oluşturur."""
    yaml_path = os.path.join(dataset_root, "sugarbeet_det.yaml")
    
    config = {
        'path': dataset_root,
        'train': 'train/images',
        'val': 'val/images',
        'names': {
            0: CLASS_NAME
        }
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Detection YAML oluşturuldu: {yaml_path}")
    return yaml_path


# ======================== ADIM 3: EĞİTİM ========================
def train_detection():
    """YOLOv8s detection modelini eğitir."""
    
    print("=" * 60)
    print("  YOLOv8s Object Detection - Sugarbeet Weed Detection")
    print("=" * 60)
    
    # --- ADIM 1: Mask → Bbox dönüşümü ---
    print("\n[ADIM 1/3] Mask'lardan bounding box label'ları üretiliyor...\n")
    
    train_mask_dir = os.path.join(DATASET_ROOT, "train", "masks")
    train_label_dir = os.path.join(DATASET_ROOT, "train", "labels")
    val_mask_dir = os.path.join(DATASET_ROOT, "val", "masks")
    val_label_dir = os.path.join(DATASET_ROOT, "val", "labels")
    
    print("Train seti:")
    convert_masks_to_bbox_labels(train_mask_dir, train_label_dir, CLASS_ID)
    
    print("\nValidation seti:")
    convert_masks_to_bbox_labels(val_mask_dir, val_label_dir, CLASS_ID)
    
    # --- ADIM 2: YAML oluştur ---
    print(f"\n[ADIM 2/3] Detection YAML oluşturuluyor...\n")
    yaml_path = create_detection_yaml(DATASET_ROOT)
    
    # --- ADIM 3: Eğitim ---
    print(f"\n[ADIM 3/3] YOLOv8s Detection eğitimi başlatılıyor...\n")
    
    # YOLOv8s Detection modeli (segmentation DEĞİL!)
    model = YOLO("yolov8s.pt")
    
    results = model.train(
        data=yaml_path,
        epochs=100,
        patience=20,
        imgsz=640,
        batch=BATCH,
        cache=True,
        device=DEVICE,
        project=RESULTS_DIR,
        name="sugarbeet_det_s",
        workers=WORKERS,
        
        # Tarımsal veri çoğaltma (segmentation ile aynı ayarlar)
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
    print(f"  Sonuçlar: {RESULTS_DIR}/sugarbeet_det_s/")
    print("=" * 60)
    
    return results


# ======================== ÇALIŞTIR ========================
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    train_detection()
