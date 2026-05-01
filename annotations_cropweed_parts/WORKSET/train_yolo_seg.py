import os
from ultralytics import YOLO

def main():
    # Load YOLOv8 Nano Segmentation model
    model = YOLO("yolov8n-seg.pt")
    
    yaml_path = r"C:\Users\90534\OneDrive\Masaüstü\Sugarbeets\annotations_cropweed_parts\WORKSET\sugarbeet.yaml"
    
    print("YOLO Segmentasyon eğitimi başlatılıyor...")
    
    # Eğitimi başlat
    results = model.train(
        data=yaml_path,
        epochs=100,         # Kullanıcı talebi: 100 epoch
        patience=20,        # Kullanıcı talebi: Overfitting durumunda eğitimi durduran mekanizma (Early stopping)
        imgsz=256,          # U-Net kodundaki ile aynı çözünürlük
        batch=16,           # RTX 3050 Ti 4GB için güvenli batch_size
        device=0,           # CUDA GPU kullanımı
        project="runs",     # Çıktıların kaydedileceği ana klasör
        name="sugarbeet_seg", # Çalışmanın adı
        workers=0           # Windows sistemlerinde dataloader kilitlenmesini engellemek için 0
    )
    
    print("Eğitim tamamlandı!")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
