import os
from ultralytics import YOLO

def main():
    # 1. İstediğin Small Model ("yolo26s" olarak aklında kalan model)
    model = YOLO("yolov8s-seg.pt")
    
    # Ortam tespiti
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if IN_COLAB:
        # Colab'da veriyi zip'ten çıkardığın için kendi oluşturduğumuz YAML dosyasını kullanıyoruz
        yaml_path = "/content/sugarbeet.yaml"
        device = 0
        workers = 8  # Colab için
    else:
        yaml_path = r"C:\Users\90534\OneDrive\Masaüstü\Sugarbeets\annotations_cropweed_parts\WORKSET\sugarbeet.yaml"
        device = 0
        workers = 0  # Windows için

    print("YOLO Segmentasyon SMALL (S) modeli eğitimi başlatılıyor...")
    
    # Eğitimi başlat
    results = model.train(
        data=yaml_path,
        epochs=100,             # İsteğine göre 100 epoch
        patience=20,            # İsteğine göre 20 epoch sabır
        imgsz=640,              # Diğer tüm ayarlar Medium model ile aynı
        batch=128,              # A100/L4 için ideal hız (Hata verirse 64 yapabilirsin)
        cache=True,             # RAM üzerinden hızlı okuma
        device=device,
        project="/content/drive/MyDrive/SugarBeetsProject/yolo_results" if IN_COLAB else "runs", 
        name="sugarbeet_seg_opt_s", # Küçük model olduğu için kaydederken sonuna _s ekledik
        workers=workers,
        
        # Tarımsal Veri Çoğaltma (Augmentation) - Birebir aynıları kopyalandı
        degrees=180.0,          
        flipud=0.5,             
        fliplr=0.5,             
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        cos_lr=True,            
    )
    
    print("Eğitim tamamlandı!")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
