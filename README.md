# Makine Öğrenmesi ve Derin Öğrenme Yöntemleri ile Görüntü Tabanlı Yangın Tespiti
Bu proje, görüntü işleme ve makine öğrenmesi/derin öğrenme teknikleri kullanarak görüntülerdeki yangın varlığını tespit etmeyi amaçlamaktadır. Proje kapsamında iki temel yaklaşım (El Yapımı Öznitelikler + Klasik Makine Öğrenmesi, ileride CNN) araştırılmakta ve uygulanmaktadır.

## Proje Yapısı
Proje dizini aşağıdaki gibi organize edilmiştir:
```
├── data_subsets/ # Veri setlerinin alt kümeleri veya işlenmiş halleri için
├── fire_dataset/ # Kullanılan yangın veri setleri (push'lanmadı)
├── Fire-Detection-Dataset/ # Dummy veri seti
├── lit_surv_review/ # Literatür taraması
├── models/ # Eğitilmiş model dosyaları ve transformerlar
├── notebooks/ # Notebook dosyaları, comparison'lar hariç benim denemelerim
├── utils/ # Yardımcılar
│ ├── pycache/
│ ├── prep_general.py
│ └── prep_svm.py
├── .gitignore 
├── README.md 
├── subset_data.py # Veri alt kümesi oluşturma
└── YAP_470_Dönem_Projesi_1._Ara_Rapor.pdf # Projenin 1. Ara Raporu
```

## Kullanılan Veri Setleri
1.  **Fire Dataset (Kaggle):** Görüntü tabanlı yangın tespiti için kullanılan ikili sınıflandırma veri setidir. Detaylı bilgi ve erişim için: [https://www.kaggle.com/datasets/phylake1337/fire-dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset)
2.  **D-Fire Dataset:** Yangın ve duman tespiti için etiketlenmiş görüntüler içeren bir veri setidir. Projede bu veri setinin `train` klasöründen bir alt küme kullanılmıştır. Detaylı bilgi ve erişim için: [https://github.com/gaiasd/DFireDataset?tab=readme-ov-file](https://github.com/gaiasd/DFireDataset?tab=readme-ov-file)

## Yöntem 1: El Yapımı Öznitelikler ve Klasik Makine Öğrenmesi
Bu yaklaşımda görüntülerden renk histogramları, LBP ve HOG olmak üzere iki set el yapımı öznitelikler çıkarılmıştır. Çıkarılan öznitelikler ölçeklendirilmiş ve farklı öznitelik seçim yöntemleri (Korelasyon Analizi, RFE) ile boyut indirgeme uygulanmıştır. Ardından, çeşitli klasik makine öğrenmesi modelleri (SVM, LightGBM, MLP) bu öznitelik setleri üzerinde Random Search ve Cross-Validation ile eğitilmiş ve hiperparametreleri optimize edilmiştir.

Uygulama adımları ve detayları `notebooks/test_method1.ipynb`, `notebooks/comparison_dfire.ipynb` ve `notebooks/comparison_kaggle.ipynb` dosyalarında görülebilir. Başlangıçtaki SVM denemeleri `notebooks/svm.ipynb` ve ilgili diğer `svm_*.ipynb` dosyalarında yer almaktadır.

*   **D-Fire Veri Seti:** SVM Scaled\_RFE75% kombinasyonu test setinde en yüksek F1 skorunu (0.8393) elde etmiştir. RFE'nin CV performansını artırma potansiyeli gözlemlense de test setine yansıması öznitelik sayısına bağlı olmuştur.
*   **Kaggle Veri Seti:** Genel olarak daha yüksek performans metrikleri elde edilmiştir. LightGBM Scaled\_Corr75% ve Scaled\_RFE50% kombinasyonları test setinde en yüksek F1 skoruna (0.9574) ulaşmıştır. SVM modelleri de bu veri setinde yüksek ve kararlı performans göstermiştir.

Detaylı sonuçlar ve yorumlamalar projenin 2. Ara Raporu'nda ve ilgili comparison notebook'larında bulunabilir.
