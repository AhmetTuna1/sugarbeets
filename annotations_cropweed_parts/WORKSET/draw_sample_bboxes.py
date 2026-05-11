import os
import cv2
import random
import numpy as np
from PIL import Image

# Ayarlar
DATASET_ROOT = "annotations_cropweed_parts/WORKSET/DATASET_SPLIT"
IMAGES_DIR = os.path.join(DATASET_ROOT, "train", "images")
MASKS_DIR = os.path.join(DATASET_ROOT, "train", "masks")
OUTPUT_DIR = "annotations_cropweed_parts/WORKSET/sample_bboxes"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def mask_to_bboxes(mask_path):
    try:
        mask_pil = Image.open(mask_path).convert('L')
        mask = np.array(mask_pil)
    except Exception as e:
        print(f"Maske okunamadı: {e}")
        return []
    
    H, W = mask.shape
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 25:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        # normalize
        x_center = (x + w / 2) / W
        y_center = (y + h / 2) / H
        norm_w = w / W
        norm_h = h / H
        bboxes.append((x_center, y_center, norm_w, norm_h))
    return bboxes

# Tüm resimleri bul
all_images = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Rastgele 3 tanesini seç
random.seed(101) # Yeni 3 resim için seed'i değiştirdim
sample_images = random.sample(all_images, 3)

for idx, img_name in enumerate(sample_images):
    img_path = os.path.join(IMAGES_DIR, img_name)
    
    # Maske ismini bul
    mask_name = os.path.splitext(img_name)[0] + ".png"
    mask_path = os.path.join(MASKS_DIR, mask_name)
    
    # Resmi oku
    try:
        img_pil = Image.open(img_path).convert('RGB')
        img_original = np.array(img_pil)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Resim okunamadı: {e}")
        continue
        
    H, W = img_original.shape[:2]
    
    # 1. VERSİYON: Orijinal Resim
    path_original = os.path.join(OUTPUT_DIR, f"sample_{idx+1}_original.png")
    cv2.imwrite(path_original, img_original)
    
    # 2. VERSİYON: Sadece Maske (Weed Maskesi)
    try:
        mask_pil = Image.open(mask_path).convert('L')
        mask_array = np.array(mask_pil)
        _, binary_mask = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)
        path_mask = os.path.join(OUTPUT_DIR, f"sample_{idx+1}_mask.png")
        cv2.imwrite(path_mask, binary_mask)
    except:
        print("Maske kaydedilemedi")
        continue

    # 3. VERSİYON: Bbox'lı Orijinal Resim
    img_bbox = img_original.copy()
    bboxes = mask_to_bboxes(mask_path)
    
    for (xc, yc, nw, nh) in bboxes:
        w = int(nw * W)
        h = int(nh * H)
        x = int((xc * W) - (w / 2))
        y = int((yc * H) - (h / 2))
        
        cv2.rectangle(img_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_bbox, "weed", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    path_bbox = os.path.join(OUTPUT_DIR, f"sample_{idx+1}_bbox.png")
    cv2.imwrite(path_bbox, img_bbox)
    
    print(f"Örnek {idx+1} için 3 dosya da kaydedildi.")
