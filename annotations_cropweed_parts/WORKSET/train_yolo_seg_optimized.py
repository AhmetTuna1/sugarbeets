import os
from ultralytics import YOLO

def main():
    # 1. Makale seviyesi yüksek kapasiteli model (Medium)
    model = YOLO("yolov8m-seg.pt")
    
    # Ortam tespiti
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if IN_COLAB:
        # Colab'da git pull yaptıktan sonra sugarbeets ana klasöründe çalışıldığı için yol:
        yaml_path = "annotations_cropweed_parts/WORKSET/sugarbeet.yaml"
        device = 0
        workers = 8  # Colab için
    else:
        yaml_path = r"C:\Users\90534\OneDrive\Masaüstü\Sugarbeets\annotations_cropweed_parts\WORKSET\sugarbeet.yaml"
        device = 0
        workers = 0  # Windows için

    print("YOLO Segmentasyon OPZİMİZE eğitimi başlatılıyor...")
    
    # Eğitimi başlat
    results = model.train(
        data=yaml_path,
        epochs=150,             # Epoch biraz artırıldı
        patience=50,            # Sabır süresi artırıldı
        imgsz=640,              # Makale odaklı: Yüksek çözünürlük
        batch=64,               # A100 40GB RAM için çok uygun (Bozulursa 32'ye düşürülebilir)
        device=device,
        project="/content/drive/MyDrive/SugarBeetsProject/yolo_results" if IN_COLAB else "runs", # Colab'da sonuçlar kaybolmasın diye Drive'a kaydeder
        name="sugarbeet_seg_opt_m", # Yeni model kayıt adı
        workers=workers,
        
        # Tarımsal Veri Çoğaltma (Augmentation)
        degrees=180.0,          # Bitkiler yukarıdan her açıyla görünebilir
        flipud=0.5,             # Yukarı-aşağı çevirme
        fliplr=0.5,             # Sağa-sola çevirme
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        cos_lr=True,            # Cosine Learning Rate Scheduler (daha yumuşak öğrenme)
    )
    
    print("Eğitim tamamlandı!")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
