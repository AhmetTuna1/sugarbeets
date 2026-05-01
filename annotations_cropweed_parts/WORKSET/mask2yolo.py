import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

def mask_to_polygon(mask_path, class_id=0):
    try:
        mask = np.array(Image.open(mask_path).convert("L"))
    except Exception as e:
        print(f"Error reading {mask_path}: {e}")
        return []
    
    H, W = mask.shape
    # Threshold the mask
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if len(contour) > 2: # At least 3 points required for a polygon
            contour = contour.flatten().tolist()
            # Normalize
            normalized_contour = []
            for i in range(0, len(contour), 2):
                x = contour[i] / W
                y = contour[i+1] / H
                normalized_contour.append(x)
                normalized_contour.append(y)
            
            polygons.append(normalized_contour)
            
    return polygons

def convert_dataset(mask_dir, label_dir, class_id=0):
    os.makedirs(label_dir, exist_ok=True)
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    
    for mask_file in tqdm(mask_files, desc=f"Converting {os.path.basename(os.path.dirname(mask_dir))}"):
        mask_path = os.path.join(mask_dir, mask_file)
        txt_filename = os.path.splitext(mask_file)[0] + ".txt"
        txt_path = os.path.join(label_dir, txt_filename)
        
        polygons = mask_to_polygon(mask_path, class_id)
        
        with open(txt_path, 'w') as f:
            for poly in polygons:
                line = f"{class_id} " + " ".join([f"{val:.6f}" for val in poly]) + "\n"
                f.write(line)

if __name__ == "__main__":
    WORKSET_DIR = r"C:\Users\90534\OneDrive\Masaüstü\Sugarbeets\annotations_cropweed_parts\WORKSET"
    DATASET_SPLIT_DIR = os.path.join(WORKSET_DIR, "DATASET_SPLIT")
    
    train_mask_dir = os.path.join(DATASET_SPLIT_DIR, "train", "masks")
    train_label_dir = os.path.join(DATASET_SPLIT_DIR, "train", "labels")
    
    val_mask_dir = os.path.join(DATASET_SPLIT_DIR, "val", "masks")
    val_label_dir = os.path.join(DATASET_SPLIT_DIR, "val", "labels")
    
    print("Converting train masks to YOLO format...")
    convert_dataset(train_mask_dir, train_label_dir)
    
    print("Converting val masks to YOLO format...")
    convert_dataset(val_mask_dir, val_label_dir)
    
    print("Mask to YOLO conversion completed successfully!")
