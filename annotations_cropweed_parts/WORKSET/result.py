import os
import matplotlib.pyplot as plt
from PIL import Image

def visualize_yolo_results(run_dir):
    """
    YOLOv8 eğitim sonuç klasöründeki standart görselleri 
    tek bir ekranda matplotlib ile gösterir.
    """
    if not os.path.exists(run_dir):
        print(f"Hata: Klasör bulunamadı -> {run_dir}")
        return

    # YOLOv8 tarafından otomatik üretilen grafikler
    images_to_show = [
        ("results.png", "Eğitim Grafikleri (Loss & mAP)"),
        ("confusion_matrix_normalized.png", "Karmaşıklık Matrisi (Normalize)"),
        ("confusion_matrix.png", "Karmaşıklık Matrisi (Confusion Matrix)"),
        ("BoxPR_curve.png", "Box PR Eğrisi (Kesinlik-Duyarlılık)"),
        ("MaskPR_curve.png", "Mask PR Eğrisi (Kesinlik-Duyarlılık)"),
        ("val_batch0_labels.jpg", "Gerçek Etiketler (Ground Truth)"),
        ("val_batch0_pred.jpg", "Model Tahminleri (Predictions)")
    ]

    found_images = []
    # Bazı dosyalar mevcut olmayabilir, olanları listeye al
    for img_name, title in images_to_show:
        img_path = os.path.join(run_dir, img_name)
        if os.path.exists(img_path):
            found_images.append((img_path, title))

    if not found_images:
        print("Gösterilecek görsel bulunamadı. Lütfen eğitim sonuç klasörünün yolunu doğru verdiğinizden emin olun.")
        return

    num_images = len(found_images)
    cols = 2
    rows = (num_images + 1) // cols

    plt.figure(figsize=(15, 6 * rows))
    for i, (img_path, title) in enumerate(found_images):
        plt.subplot(rows, cols, i + 1)
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("="*50)
    print("YOLOv8 Eğitim Sonuçları Görselleştirici")
    print("="*50)
    # Varsayılan olarak Colab üzerindeki son eğitim dizinini önerelim
    default_dir = "/content/drive/MyDrive/SugarBeetsProject/yolo_results/sugarbeet_seg-5"
    
    # Kullanıcıdan yol isteyelim veya varsayılanı kullanalım
    print(f"Varsayılan yol: {default_dir}")
    user_input = input("Sonuç klasör yolunu girin (Varsayılanı kullanmak için Enter'a basın): ").strip()
    
    target_dir = user_input if user_input else default_dir
    visualize_yolo_results(target_dir)
