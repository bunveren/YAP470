import cv2
import os
import numpy as np
from tqdm import tqdm
from skimage.feature import local_binary_pattern, hog
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time
from sklearn.decomposition import PCA

def load_prep_img(data_root, target_size, 
    color_spaces=['bgr', 'hsv', 'ycbcr'], normalize=1):
    processed_data = {cs: [] for cs in color_spaces + ['gray']}
    labels = []
    class_dirs = {'fire_images': 1, 'non_fire_images': 0}
    img_extensions = ['.jpg', '.jpeg', '.png']
    
    total_images_processed = 0
    total_images_skipped = 0
    for class_name, label in class_dirs.items():
        class_dir_path = os.path.join(data_root, class_name)

        if not os.path.isdir(class_dir_path): continue
        
        all_files = os.listdir(class_dir_path)
        image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in img_extensions]

        if not image_files: continue
        
        for filename in tqdm(image_files, desc=f"Processing {class_name}"):
            file_path = os.path.join(class_dir_path, filename)
            img_bgr = cv2.imread(file_path)

            if img_bgr is None:
                total_images_skipped += 1
                continue

            img_resized = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_LINEAR)
            if normalize: img_resized = img_resized.astype(np.float32) / 255.0

            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            processed_data['gray'].append(img_gray)

            if 'bgr' in color_spaces:
                 processed_data['bgr'].append(img_resized)
            if 'hsv' in color_spaces:
                img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
                processed_data['hsv'].append(img_hsv)
            if 'ycbcr' in color_spaces:
                img_ycbcr = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YCrCb)
                processed_data['ycbcr'].append(img_ycbcr)

            labels.append(label)
            total_images_processed += 1

        print(f"\ntoplam islenen: {total_images_processed} atlanan: {total_images_skipped}")
    
    output_dtype = np.float32 if normalize else np.uint8
    numpy_data = {}
    for cs, data_list in processed_data.items():
        if data_list:
             numpy_data[cs] = np.array(data_list)
        else:
             if cs == 'gray':
                  numpy_data[cs] = np.array([], dtype=output_dtype).reshape(0, target_size[1], target_size[0])
             else:
                  numpy_data[cs] = np.array([], dtype=output_dtype).reshape(0, target_size[1], target_size[0], 3)

    numpy_data['labels'] = np.array(labels)
    return numpy_data    

def load_prep_dfire(split_root_path, target_size, fire_class_ids,
    color_spaces=['bgr', 'hsv', 'ycbcr'], normalize=1):
    processed_data = {cs: [] for cs in color_spaces + ['gray']}
    labels = []
    img_extensions = ['.jpg', '.jpeg', '.png']
    annotation_extension = '.txt' 
    images_dir = os.path.join(split_root_path, 'images')
    labels_dir = os.path.join(split_root_path, 'labels')
    print("aa")
    if not os.path.isdir(images_dir):
        return None 
    if not os.path.isdir(labels_dir):
        return None 
    all_image_files = [f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in img_extensions]
    total_images_processed = 0
    total_images_skipped = 0
    images_with_fire = 0
    images_without_fire = 0

    for filename in tqdm(all_image_files, desc=f"Processing {os.path.basename(split_root_path)}"):
        image_path = os.path.join(images_dir, filename)
        image_name_without_ext = os.path.splitext(filename)[0]
        annotation_path = os.path.join(labels_dir, image_name_without_ext + annotation_extension)

        has_fire = False
        if os.path.exists(annotation_path):
            try:
                with open(annotation_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts: 
                            class_id = int(parts[0])
                            if class_id in fire_class_ids:
                                has_fire = True
                                break 
            except Exception as e:
                has_fire = False
                total_images_skipped += 1
                continue 

        label = 1 if has_fire else 0
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            total_images_skipped += 1
            continue

        img_resized = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_LINEAR)
        if normalize: img_resized = img_resized.astype(np.float32) / 255.0
       
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        processed_data['gray'].append(img_gray)

        if 'bgr' in color_spaces:
             processed_data['bgr'].append(img_resized)
        if 'hsv' in color_spaces:
            img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
            processed_data['hsv'].append(img_hsv)
        if 'ycbcr' in color_spaces:
            img_ycbcr = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YCrCb)
            processed_data['ycbcr'].append(img_ycbcr)

        labels.append(label)
        total_images_processed += 1
        if has_fire:
             images_with_fire += 1
        else:
             images_without_fire += 1

    print(f"\ntoplam islenen: {total_images_processed} atlanan: {total_images_skipped}")
    print(f"yangin var (1): {images_with_fire}, yok (0): {images_without_fire}")

    output_dtype = np.float32 if normalize else np.uint8

    numpy_data = {}
    for cs, data_list in processed_data.items():
        if data_list:
             numpy_data[cs] = np.array(data_list)
        else:
             if cs == 'gray':
                  numpy_data[cs] = np.array([], dtype=output_dtype).reshape(0, target_size[1], target_size[0])
             else:
                  numpy_data[cs] = np.array([], dtype=output_dtype).reshape(0, target_size[1], target_size[0], 3)
                  
    numpy_data['labels'] = np.array(labels) 
    return numpy_data     

def is_dfire_image_fire(annotation_path, fire_class_ids):
    if not os.path.exists(annotation_path):
        return False
    try:
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts and len(parts) > 0:
                    if parts[0].isdigit():
                        class_id = int(parts[0])
                        if class_id in fire_class_ids:
                            return True 
    except Exception as e: pass 
    return False

def load_and_extract_features_memory_safe(config, feature_params):
    dataset_choice = config.get('dataset_choice', 'dfire') 
    data_root = config.get('data_root')
    target_size = config.get('target_img_size')
    color_spaces_to_load = config.get('color_spaces_to_load', ['bgr', 'hsv', 'ycbcr'])
    normalize_pixels = config.get('normalize_pixels', 1)
    fire_class_ids = config.get('fire_class_ids', []) 

    if not data_root or not target_size:
        return np.array([]), np.array([])

    image_label_pairs = []
    img_extensions = ['.jpg', '.jpeg', '.png']
    annotation_extension = '.txt'

    images_dir = os.path.join(data_root, 'images')
    labels_dir = os.path.join(data_root, 'labels')
    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        return np.array([]), np.array([])

    all_image_files = [f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in img_extensions]
    for filename in tqdm(all_image_files, desc="Determining Labels"):
        image_path = os.path.join(images_dir, filename)
        image_name_without_ext = os.path.splitext(filename)[0]
        annotation_path = os.path.join(labels_dir, image_name_without_ext + annotation_extension)
        label = 1 if is_dfire_image_fire(annotation_path, fire_class_ids) else 0
        image_label_pairs.append((image_path, label))
  
    if not image_label_pairs:
        return np.array([]), np.array([])

    all_features_list = []
    all_labels_list = []
    total_images_processed = 0
    total_images_skipped_reading = 0
    total_images_skipped_feature = 0

    for image_path, label in tqdm(image_label_pairs, desc="dfire memsafe feat exc."):
        img_bgr = cv2.imread(image_path)

        if img_bgr is None:
            total_images_skipped_reading += 1
            continue
        img_resized = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_LINEAR)
        if normalize_pixels:
             img_resized = img_resized.astype(np.float32) / 255.0
        else:
             img_resized = img_resized.astype(np.uint8)

        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        img_dict_single = {'gray': img_gray}
        if 'bgr' in color_spaces_to_load:
             img_dict_single['bgr'] = img_resized
        if 'hsv' in color_spaces_to_load:
             if normalize_pixels:
                  img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
             else:
                  img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
             img_dict_single['hsv'] = img_hsv
        if 'ycbcr' in color_spaces_to_load:
             if normalize_pixels:
                 img_ycbcr = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YCrCb)
             else:
                 img_ycbcr = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YCrCb)
             img_dict_single['ycbcr'] = img_ycbcr
        features_single = combine_features(img_dict_single, feature_params)

        if features_single.size > 0:
            all_features_list.append(features_single)
            all_labels_list.append(label)
            total_images_processed += 1
        else:
            total_images_skipped_feature += 1
            
    if not all_features_list:
        return np.array([]), np.array([])

    features_array = np.array(all_features_list, dtype=np.float32)
    labels_array = np.array(all_labels_list, dtype=np.int32) 
    return features_array, labels_array
  
def extract_color_histograms(img_processed, color_space, bins):
    histograms = []
    if color_space == 'hsv':
        if img_processed.dtype == np.float32:
            hist_h = cv2.calcHist([img_processed], [0], None, [bins], [0, 360])
            hist_s = cv2.calcHist([img_processed], [1], None, [bins], [0, 1])
            histograms.extend([hist_h.flatten(), hist_s.flatten()]) 
        elif img_processed.dtype == np.uint8:
            hist_h = cv2.calcHist([img_processed], [0], None, [bins], [0, 180])
            hist_s = cv2.calcHist([img_processed], [1], None, [bins], [0, 256])
            histograms.extend([hist_h.flatten(), hist_s.flatten()]) 
    elif color_space == 'ycbcr':
        if img_processed.dtype == np.float32:
            hist_y = cv2.calcHist([img_processed], [0], None, [bins], [0, 1])
            hist_cb = cv2.calcHist([img_processed], [1], None, [bins], [-0.5, 0.5])
            hist_cr = cv2.calcHist([img_processed], [2], None, [bins], [-0.5, 0.5])
            histograms.extend([hist_y.flatten(), hist_cb.flatten(), hist_cr.flatten()])
        elif img_processed.dtype == np.uint8:
            hist_y = cv2.calcHist([img_processed], [0], None, [bins], [0, 256])
            hist_cb = cv2.calcHist([img_processed], [1], None, [bins], [0, 256])
            hist_cr = cv2.calcHist([img_processed], [2], None, [bins], [0, 256])
            histograms.extend([hist_y.flatten(), hist_cb.flatten(), hist_cr.flatten()])

    if histograms: return np.concatenate(histograms)
    else: return np.array([]) 

def extract_lbp_features(img_gray, radius, n_points, method):
    if n_points is None:
        n_points = 8 * radius
    lbp_image = local_binary_pattern(img_gray, n_points, radius, method=method)
    if method == 'uniform':
        n_bins = n_points + 2
    else:
         n_bins = int(lbp_image.max() + 1)
    lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=n_bins, range=(0, n_bins))
    lbp_hist = lbp_hist.astype(np.float32)
    if lbp_hist.sum() > 0:
        lbp_hist /= lbp_hist.sum()
    return lbp_hist.flatten()

def extract_hog_features(img_gray, orientations, pixels_per_cell, cells_per_block, block_norm):
    hog_features = hog(img_gray, orientations=orientations,
                       pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block,
                       block_norm=block_norm,
                       visualize=False, feature_vector=True)
    return hog_features.flatten() 

def combine_features(img_dict, feature_params):
    all_features = []
    if 'hsv' in img_dict:
        hsv_hist = extract_color_histograms(img_dict['hsv'], 'hsv', bins=feature_params.get('hist_bins', 100))
        if hsv_hist.size > 0:
            all_features.append(hsv_hist)
    if 'ycbcr' in img_dict:
         ycbcr_hist = extract_color_histograms(img_dict['ycbcr'], 'ycbcr', bins=feature_params.get('hist_bins', 100))
         if ycbcr_hist.size > 0:
              all_features.append(ycbcr_hist)

    if 'gray' in img_dict:
        img_gray_processed = img_dict['gray'] 

        lbp_features = extract_lbp_features(img_gray_processed,
                                            radius=feature_params.get('lbp_radius', 3),
                                            n_points=feature_params.get('lbp_n_points', None),
                                            method=feature_params.get('lbp_method', 'uniform'))
        if lbp_features.size > 0:
             all_features.append(lbp_features)

        hog_features = extract_hog_features(img_gray_processed,
                                           orientations=feature_params.get('hog_orientations', 9),
                                           pixels_per_cell=feature_params.get('hog_pixels_per_cell', (8, 8)),
                                           cells_per_block=feature_params.get('hog_cells_per_block', (2, 2)),
                                           block_norm=feature_params.get('hog_block_norm', 'L2-Hys'))
        if hog_features.size > 0:
             all_features.append(hog_features)

    if all_features:
        combined_vector = np.concatenate(all_features)
        return combined_vector
    else:
        return np.array([])

def get_config(dataset_choice):
    config = {}
    if dataset_choice == 'kaggle':
        config['data_root'] = "fire_dataset"
        config['target_img_size'] = (128, 128)
        config['color_spaces_to_load'] = ['bgr', 'hsv', 'ycbcr']
        config['normalize_pixels'] = 1
        config['fire_class_ids'] = None 
    elif dataset_choice == 'dfire':
        config['dfire_root'] = "D-Fire" 
        config['split_name'] = "train" 
        config['data_root'] = os.path.join(config['dfire_root'], config['split_name'])
        config['target_img_size'] = (128, 128)
        config['color_spaces_to_load'] = ['bgr', 'hsv', 'ycbcr']
        config['normalize_pixels'] = 1
        config['fire_class_ids'] = [0, 1] 
    else:
        raise ValueError()

    print(f"Using dataset: {dataset_choice}")
    print(f"Data root: {config.get('data_root')}")
    print(f"Target image size: {config['target_img_size']}")
    print(f"Color spaces loaded: {config['color_spaces_to_load']}")
    print(f"Normalize pixels: {bool(config['normalize_pixels'])}")
    if dataset_choice == 'dfire':
         print(f"D-Fire Split: {config['split_name']}")
         print(f"D-Fire Fire Class IDs: {config['fire_class_ids']}")
    return config

def get_feature_params():
    feature_params = {
        'hist_bins': 100,
        'lbp_radius': 3,
        'lbp_n_points': None, 
        'lbp_method': 'uniform', # 'uniform', 'default', 'ror', 'nri_uniform'
        'hog_orientations': 9,
        'hog_pixels_per_cell': (8, 8),
        'hog_cells_per_block': (2, 2),
        'hog_block_norm': 'L2-Hys' # 'L2-Hys', 'L2', 'L1', 'L1-sqrt'
    }
    print("\nFeature extraction parameters:", feature_params)
    return feature_params

def load_and_preprocess_data(config):
    data_root = config['data_root']
    target_img_size = config['target_img_size']
    color_spaces_to_load = config['color_spaces_to_load']
    normalize_pixels = config['normalize_pixels']
    dataset_choice = config.get('dataset_choice', 'kaggle')

    processed_data_dict = None
    if dataset_choice == 'kaggle':
         if os.path.exists(data_root):
            processed_data_dict = load_prep_img(
                data_root,
                target_img_size,
                color_spaces=color_spaces_to_load,
                normalize=normalize_pixels
            )
         else:
            print(f"Kaggle dataset root not found at {data_root}")

    elif dataset_choice == 'dfire':
         fire_class_ids = config['fire_class_ids']
         if os.path.exists(data_root):
            processed_data_dict = load_prep_dfire(
                data_root,
                target_img_size,
                fire_class_ids=fire_class_ids,
                color_spaces=color_spaces_to_load,
                normalize=normalize_pixels
            )
         else:
            print(f"D-Fire split root not found at {data_root}")

    if processed_data_dict and 'labels' in processed_data_dict and len(processed_data_dict['labels']) > 0:
        print(f"\nSuccessfully loaded and preprocessed {len(processed_data_dict['labels'])} images.")
    else:
        raise ValueError()

    return processed_data_dict

def extract_features(processed_data_dict, feature_params):
    features = []
    valid_labels = []

    if processed_data_dict and 'labels' in processed_data_dict and len(processed_data_dict['labels']) > 0:
        labels_all = processed_data_dict['labels']
        num_images_total = len(labels_all)
        available_data_keys = [key for key in processed_data_dict.keys() if key != 'labels']

        for i in tqdm(range(num_images_total), desc=f"Extracting Features"):
            img_dict_single = {}
            for key in available_data_keys:
                 if processed_data_dict[key] is not None and i < len(processed_data_dict[key]):
                      img_dict_single[key] = processed_data_dict[key][i]

            if img_dict_single:
                features_single = combine_features(img_dict_single, feature_params)
                if features_single.size > 0:
                    features.append(features_single)
                    valid_labels.append(labels_all[i])
                else:
                    pass 
            else:
                pass 
        features_array = np.array(features)
        labels_array = np.array(valid_labels)

        print(f"\ntotal featurei cikarilabilmis resim s.: {features_array.shape[0]}")
        if features_array.shape[0] > 0:
             print(f"feature arr shape: {features_array.shape}")
             print(f"label arr shape: {labels_array.shape}")
        else:
             print("feature cikarilamadi!")

        return features_array, labels_array
    else:
        print("feature cikarilamadi!")
        return np.array([]), np.array([])

def split_data(features_array, labels_array, test_size=0.2, random_state=42):
    if features_array.shape[0] == 0:
        print("feature arrayi bos!")
        return None, None, None, None

    print(f"\ntraining split: ({1-test_size:.0%}) testing split: ({test_size:.0%})")
    X_train, X_test, y_train, y_test = train_test_split(
        features_array,
        labels_array,
        test_size=test_size,
        random_state=random_state,
        stratify=labels_array 
    )

    print(f"training features shape: {X_train.shape}")
    print(f"testing features shape: {X_test.shape}")
    print(f"training labels shape: {y_train.shape}")
    print(f"testing labels shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    if X_train is None or X_test is None or X_train.shape[0] == 0:
         print("scale edilemedi! train ya da test data bos")
         return None, None, None
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def apply_pca(X_train, X_test, n_components=None):
    if X_train is None or X_test is None or X_train.shape[0] == 0:
         return None, None, None

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca
