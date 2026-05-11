import os
import cv2
import random
import numpy as np
from PIL import Image

# Ayarlar
DATASET_ROOT = "annotations_cropweed_parts/WORKSET/DATASET_SPLIT"
IMAGES_DIR = os.path.join(DATASET_ROOT, "train", "images")
MASKS_DIR = r"annotations_cropweed_parts\WORKSET\masks"  # Orijinal (Bonn) maskelerin olduğu klasör
OUTPUT_DIR = "annotations_cropweed_parts/WORKSET/multiclass_bboxes"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def mask_to_multiclass_bboxes(mask_path):
    try:
        mask_pil = Image.open(mask_path)
        mask = np.array(mask_pil)
    except Exception as e:
        print(f"Maske okunamadı: {e}")
        return []
    
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
        
    H, W = mask.shape
    bboxes = []
    
    # --- Sınıf 0: Yabancı Ot (Weed) ---
    weed_binary = np.where(mask == 10000, 255, 0).astype(np.uint8)
    contours_weed, _ = cv2.findContours(weed_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_weed:
        if cv2.contourArea(contour) < 25: continue
        x, y, w, h = cv2.boundingRect(contour)
        xc, yc = (x + w / 2) / W, (y + h / 2) / H
        bboxes.append((0, xc, yc, w/W, h/H))

    # --- Sınıf 1: Şeker Pancarı (Crop) ---
    crop_binary = np.where((mask > 0) & (mask < 10000), 255, 0).astype(np.uint8)
    contours_crop, _ = cv2.findContours(crop_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_crop:
        if cv2.contourArea(contour) < 25: continue
        x, y, w, h = cv2.boundingRect(contour)
        xc, yc = (x + w / 2) / W, (y + h / 2) / H
        bboxes.append((1, xc, yc, w/W, h/H))
        
    return bboxes

all_images = [f for f in os.listdir(MASKS_DIR) if f.lower().endswith('.png')]
random.shuffle(all_images)

saved_count = 0

for mask_name in all_images:
    if saved_count >= 3:
        break
    # Orijinal resim ismini bul (mask_name ile aynı)
    img_name = mask_name
    img_path = os.path.join(IMAGES_DIR, img_name)
    mask_path = os.path.join(MASKS_DIR, mask_name)
    
    if not os.path.exists(img_path):
        # Eger train/images icinde yoksa baska bir tane dene, basite indirgemek icin gecelim
        continue

    try:
        img_pil = Image.open(img_path).convert('RGB')
        img_original = np.array(img_pil)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Resim okunamadı: {e}")
        continue
        
    H, W = img_original.shape[:2]
    
    # Bbox'ları al ve kontrol et: Hem ot hem pancar var mı?
    bboxes = mask_to_multiclass_bboxes(mask_path)
    has_weed = any(cls_id == 0 for cls_id, _, _, _, _ in bboxes)
    has_crop = any(cls_id == 1 for cls_id, _, _, _, _ in bboxes)
    
    # Sadece hem ot hem pancar olan resimleri kaydet ki kullanıcı iki kutuyu da görebilsin
    if not (has_weed and has_crop):
        continue
        
    idx = saved_count
    saved_count += 1
    
    # 1. VERSİYON: Orijinal
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"sample_{idx+1}_original.png"), img_original)
    
    # 2. VERSİYON: Maske
    try:
        mask_pil = Image.open(mask_path)
        mask_array = np.array(mask_pil)
        
        # Weed=Beyaz, Crop=Gri, Arkaplan=Siyah
        visual_mask = np.zeros((H, W), dtype=np.uint8)
        visual_mask[mask_array == 10000] = 255
        visual_mask[(mask_array > 0) & (mask_array < 10000)] = 127
        
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"sample_{idx+1}_mask.png"), visual_mask)
    except:
        print("Maske kaydedilemedi")
        continue

    # 3. VERSİYON: Bbox
    img_bbox = img_original.copy()
    bboxes = mask_to_multiclass_bboxes(mask_path)
    
    for (cls_id, xc, yc, nw, nh) in bboxes:
        w = int(nw * W)
        h = int(nh * H)
        x = int((xc * W) - (w / 2))
        y = int((yc * H) - (h / 2))
        
        if cls_id == 0:
            # Weed -> Kırmızı Kutu
            cv2.rectangle(img_bbox, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img_bbox, "weed", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            # Crop -> Yeşil Kutu
            cv2.rectangle(img_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_bbox, "crop", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"sample_{idx+1}_bbox.png"), img_bbox)
    print(f"Örnek {idx+1} için 3 dosya kaydedildi.")
