import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm 

DATASET_PATH = os.path.abspath("fire_dataset")
IMAGE_SIZE = (128, 128) 
N_SPLITS_KFOLD = 5
RANDOM_STATE = 42

def load_and_preprocess_data(dataset_path, image_size):
    print(f"load_and_preprocess_data çağrıldı. dataset_path: {dataset_path}") # Ek debug
    images = []
    labels = []
    try:
        if not os.path.isdir(dataset_path):
            print(f"HATA: Veri seti yolu '{dataset_path}' bulunamadı veya bir klasör değil.")
            return None, None, None
        
        print(f"'{dataset_path}' içindeki öğeler: {os.listdir(dataset_path)}") # Ek debug
        
        class_names_temp = os.listdir(dataset_path)
        class_names = sorted([d for d in class_names_temp if os.path.isdir(os.path.join(dataset_path, d))])

    except FileNotFoundError:
        print(f"HATA: Veri seti yolu '{dataset_path}' bulunamadı.")
        return None, None, None
    except Exception as e:
        print(f"HATA: Veri seti yolu '{dataset_path}' okunurken bir hata oluştu: {e}")
        return None, None, None

    if not class_names or len(class_names) < 2:
        print(f"HATA: {dataset_path} içinde yeterli sınıf klasörü bulunamadı (en az 2 bekleniyor). Bulunanlar: {class_names}")
        return None, None, None

    print(f"Sınıflar yükleniyor: {class_names}")
    actual_class_names_found = []

    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path): 
            print(f"Uyarı: {class_path} (döngü içinde kontrol edildi) bir klasör değil, atlanıyor.")
            continue
        
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"'{class_name}' sınıfından {len(image_files)} görüntü yükleniyor...")
        
        if not image_files:
            print(f"Uyarı: '{class_name}' klasöründe desteklenen formatta görüntü bulunamadı.")
            continue
        
        images_loaded_from_this_class = 0 
        for image_name in tqdm(image_files, desc=f"'{class_name}' yükleniyor"):
            image_path = os.path.join(class_path, image_name)
            try:
                if not os.path.exists(image_path):
                    print(f"UYARI: Dosya bulunamadı (os.path.exists ile): {image_path}, atlanıyor.")
                    continue

                file_bytes = np.fromfile(image_path, dtype=np.uint8)
                image = None 
                if file_bytes.size > 0:
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            

                if image is None:
                    print(f"Uyarı: {image_path} yüklenemedi (np.fromfile/cv2.imdecode denendi), atlanıyor.")
                    continue

                image = cv2.resize(image, image_size)
                images.append(image)
                
                is_fire_class = class_name.lower().startswith('fire') or class_name.lower() == 'fire'
                labels.append(1 if is_fire_class else 0)
                images_loaded_from_this_class += 1

            except Exception as e:
                print(f"Hata: {image_path} işlenirken sorun oluştu: {e}, atlanıyor.")
                continue
        
        if images_loaded_from_this_class > 0:
            print(f"'{class_name}' sınıfından {images_loaded_from_this_class} görüntü başarıyla işlendi.")
            if class_name not in actual_class_names_found:
                actual_class_names_found.append(class_name)
        else:
            print(f"'{class_name}' sınıfından hiç görüntü işlenemedi.")


    if not images:
        print("HATA: Hiçbir görüntü yüklenemedi. Veri seti yolunu ve içeriğini kontrol edin.")
        print(f"Kontrol edilen ana klasör: {dataset_path}")
        return None, None, None
        
    print(f"load_and_preprocess_data tamamlandı. Yüklenen görüntü sayısı: {len(images)}, Etiket sayısı: {len(labels)}") # Ek debug
    return np.array(images), np.array(labels), sorted(actual_class_names_found)

def extract_color_histogram(image, bins=(8, 8, 8), hsv=True):
    if hsv:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist) 
    return hist.flatten()

def extract_hog_features(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(image_gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
    return features

def extract_lbp_features(image, P=24, R=3, method='uniform'): 
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(image_gray, P, R, method=method)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6) 
    return hist

def get_all_features(images_data, feature_types=['color', 'hog', 'lbp']):
    all_image_features = []
    print("Özellikler çıkarılıyor...")
    for image in tqdm(images_data, desc="Özellik çıkarımı"):
        current_features = []
        if 'color' in feature_types:
            current_features.append(extract_color_histogram(image, hsv=True)) 
        if 'hog' in feature_types:
            current_features.append(extract_hog_features(image))
        if 'lbp' in feature_types:
            current_features.append(extract_lbp_features(image))
        
        combined_features = np.concatenate(current_features) if current_features else np.array([])
        all_image_features.append(combined_features)
    
    return np.array(all_image_features)

def get_classifiers():
    classifiers = {
        "KNN": KNeighborsClassifier(n_neighbors=5), 
        "SVM": SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE), 
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, max_iter=200) 
    }
    return classifiers

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Karışıklık matrisini çizer.
    """
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def evaluate_classifiers(X, y, classifiers, n_splits=N_SPLITS_KFOLD, class_names_map=None):
    """
    Sınıflandırıcıları k-fold cross-validation ile değerlendirir ve sonuçları yazdırır.
    """
    if class_names_map is None:
        class_names_map = ["Class 0", "Class 1"]

    results = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scaler = StandardScaler()

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro', zero_division=0),
        'recall': make_scorer(recall_score, average='macro', zero_division=0),
        'f1_score': make_scorer(f1_score, average='macro', zero_division=0)
    }

    for name, clf_original in classifiers.items():
        print(f"\n--- {name} değerlendiriliyor ---")        
        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1s = []
        all_y_true = []
        all_y_pred = []

        for fold_idx, (train_index, test_index) in enumerate(kf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            import copy
            clf = copy.deepcopy(clf_original)

            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)

            fold_accuracies.append(accuracy_score(y_test, y_pred))
            fold_precisions.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
            fold_recalls.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
            fold_f1s.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
            
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

        avg_accuracy = np.mean(fold_accuracies)
        avg_precision = np.mean(fold_precisions)
        avg_recall = np.mean(fold_recalls)
        avg_f1 = np.mean(fold_f1s)

        results[name] = {
            'Accuracy': avg_accuracy,
            'Precision': avg_precision,
            'Recall': avg_recall,
            'F1-score': avg_f1
        }
        print(f"  Ortalama Accuracy: {avg_accuracy:.4f}")
        print(f"  Ortalama Precision (Macro): {avg_precision:.4f}")
        print(f"  Ortalama Recall (Macro): {avg_recall:.4f}")
        print(f"  Ortalama F1-score (Macro): {avg_f1:.4f}")
        cm = confusion_matrix(all_y_true, all_y_pred)
        plot_confusion_matrix(cm, classes=class_names_map, title=f'{name} - Confusion Matrix (All Folds)')
        
    return results

if __name__ == "__main__":
    print("Proje Başlatılıyor: Görüntü Tabanlı Yangın Tespiti")
    if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
        print(f"Uyarı: '{DATASET_PATH}' bulunamadı veya boş. Test için dummy veri seti oluşturuluyor.")
        print("Lütfen Kaggle'dan veri setini indirin ve DATASET_PATH değişkenini güncelleyin.")
        images_data, labels, class_names_loaded = load_and_preprocess_data(DATASET_PATH, IMAGE_SIZE)

    if images_data is None or labels is None:
        print("Veri yükleme başarısız. Program sonlandırılıyor.")
        exit()
        
    print(f"\nToplam {len(images_data)} görüntü yüklendi.")
    print(f"Etiket dağılımı: {np.bincount(labels)}") 
    
    non_fire_label = "Non-Fire"
    fire_label = "Fire"
    for name in class_names_loaded:
        if 'fire' in name.lower():
            fire_label = name
        else:
            non_fire_label = name
    class_names_for_cm = ["Non-Fire", "Fire"] 
    
    feature_combinations = {
        "ColorHistogram": ['color'],
        "HOG": ['hog'],
        "LBP": ['lbp'],
        "Color+HOG": ['color', 'hog'],
        "HOG+LBP": ['hog', 'lbp'],
        "Color+LBP": ['color', 'lbp'],
        "All_Features": ['color', 'hog', 'lbp']
    }
    
    all_experiment_results = {}

    for set_name, feature_list in feature_combinations.items():
        print(f"\n===== ÖZELLİK SETİ: {set_name} ({', '.join(feature_list)}) =====")
        X_features = get_all_features(images_data, feature_types=feature_list)
        
        if X_features.shape[0] == 0 or X_features.shape[1] == 0:
            print(f"'{set_name}' için özellik çıkarılamadı veya boş özellik vektörü döndü. Atlanıyor.")
            continue

        print(f"Özellik vektörü boyutu: {X_features.shape}")

        classifiers = get_classifiers()

        print(f"\n{set_name} özellikleri ile sınıflandırıcılar değerlendiriliyor...")
        results_for_set = evaluate_classifiers(X_features, labels, classifiers, class_names_map=class_names_for_cm)
        all_experiment_results[set_name] = results_for_set

    print("\n\n===== TÜM DENEY SONUÇLARI ÖZETİ =====")
    for feature_set_name, model_results in all_experiment_results.items():
        print(f"\n--- Özellik Seti: {feature_set_name} ---")
        for model_name, metrics in model_results.items():
            print(f"  {model_name}:")
            for metric_name, value in metrics.items():
                print(f"    {metric_name}: {value:.4f}")

    print("\nProje Tamamlandı.")