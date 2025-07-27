import os
import sys
import time
import joblib
import numpy as np
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore') # Suppress warnings for cleaner output

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU, Input, ReLU # Ensure ReLU is imported if used
from scikeras.wrappers import KerasClassifier # For wrapping Keras models in scikit-learn API

from skimage.feature import local_binary_pattern, hog
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.base import ClassifierMixin
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt # For plotting single images


# --- Helper Functions (Copied from your initial setup) ---

def is_dfire_image_fire(annotation_path, fire_class_ids):
    if not os.path.exists(annotation_path): return False
    try:
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts and len(parts) > 0:
                    if parts[0].isdigit():
                        class_id = int(parts[0])
                        if class_id in fire_class_ids: return True
    except Exception as e: pass
    return False

def extract_color_histograms(img_processed, color_space, bins):
    histograms = []
    ranges = {
        'hsv': {'float': ([0, 1], [0, 1], [0, 1]), 'uint8': ([0, 180], [0, 256], [0, 256])},
        'ycbcr': {'float': ([0, 1], [-0.5, 0.5], [-0.5, 0.5]), 'uint8': ([0, 256], [0, 256], [0, 256])}
    }
    channel_indices = {'hsv': [0, 1], 'ycbcr': [1, 2]}
    dtype_key = 'float' if img_processed.dtype in [np.float32, np.float64] else 'uint8'

    if color_space in ranges and color_space in channel_indices:
        for i in channel_indices[color_space]:
            current_range = ranges[color_space][dtype_key][i]
            if img_processed.dtype != np.float32 and img_processed.dtype != np.uint8:
                 img_processed = img_processed.astype(np.float32)
                 dtype_key = 'float'
                 current_range = ranges[color_space][dtype_key][i]

            hist = cv2.calcHist([img_processed], [i], None, [bins], current_range)
            histograms.append(hist.flatten())

    if histograms: return np.concatenate(histograms)
    else: return np.array([])

def extract_lbp_features(img_gray, radius, n_points, method):

    if img_gray is None or img_gray.size == 0: return np.array([])
    if n_points is None: n_points = 8 * radius

    if img_gray.dtype != np.uint8 and img_gray.dtype != np.float64:
         img_gray = img_gray.astype(np.float64)

    try:
        lbp_image = local_binary_pattern(img_gray, n_points, radius, method=method)
        if method == 'uniform' or method == 'nri_uniform':
            n_bins = int(n_points + 2)
            hist_range = (0, n_bins)
        elif method == 'ror':
            n_bins = int(n_points / radius + 2)
            hist_range = (0, n_bins)
        else:
            n_bins = int(2**n_points)
            hist_range = (0, n_bins)
        lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=n_bins, range=hist_range)
        lbp_hist = lbp_hist.astype(np.float32)
        if lbp_hist.sum() > 0: lbp_hist /= lbp_hist.sum()
        return lbp_hist.flatten()

    except Exception as e:
        return np.array([])

def extract_hog_features(img_gray, orientations, pixels_per_cell, cells_per_block, block_norm):
    if img_gray is None or img_gray.size == 0: return np.array([])
    if img_gray.dtype != np.uint8 and img_gray.dtype != np.float64:
         img_gray = img_gray.astype(np.float64)
    img_h, img_w = img_gray.shape
    cell_h, cell_w = pixels_per_cell
    block_h, block_w = cells_per_block
    min_img_h = cell_h * block_h
    min_img_w = cell_w * block_w

    if img_h < min_img_h or img_w < min_img_w:
        return np.array([])
    try:
        hog_features = hog(img_gray, orientations=orientations,
                           pixels_per_cell=pixels_per_cell,
                           cells_per_block=cells_per_block,
                           block_norm=block_norm,
                           visualize=False, feature_vector=True)
        return hog_features.flatten()
    except Exception as e:
        return np.array([])

def combine_features(img_dict, feature_params):
    all_features = []
    if 'hsv' in img_dict and img_dict['hsv'] is not None:
        hsv_hist = extract_color_histograms(img_dict['hsv'], 'hsv', bins=feature_params.get('hist_bins', 100))
        if hsv_hist.size > 0: all_features.append(hsv_hist)
    if 'ycbcr' in img_dict and img_dict['ycbcr'] is not None:
        ycbcr_hist = extract_color_histograms(img_dict['ycbcr'], 'ycbcr', bins=feature_params.get('hist_bins', 100))
        if ycbcr_hist.size > 0: all_features.append(ycbcr_hist)
    if 'gray' in img_dict and img_dict['gray'] is not None:
        img_gray_processed = img_dict['gray']

        lbp_features = extract_lbp_features(img_gray_processed,
                                            radius=feature_params.get('lbp_radius', 3),
                                            n_points=feature_params.get('lbp_n_points', None),
                                            method=feature_params.get('lbp_method', 'uniform'))
        if lbp_features.size > 0: all_features.append(lbp_features)

        hog_features = extract_hog_features(img_gray_processed,
                                           orientations=feature_params.get('hog_orientations', 9),
                                           pixels_per_cell=feature_params.get('hog_pixels_per_cell', (8, 8)),
                                           cells_per_block=feature_params.get('hog_cells_per_block', (2, 2)),
                                           block_norm=feature_params.get('hog_block_norm', 'L2-Hys'))
        if hog_features.size > 0: all_features.append(hog_features)

    if all_features:
        all_features = [f.astype(np.float32) for f in all_features]
        combined_vector = np.concatenate(all_features)
        return combined_vector
    else:
        return np.array([])

def load_and_extract_features_memory_safe(config, feature_params):
    dataset_choice = config.get('dataset_choice', 'dfire')
    data_root = config.get('data_root')
    target_size = config.get('target_img_size')
    color_spaces_to_load = config.get('color_spaces_to_load', ['bgr', 'hsv', 'ycbcr'])
    normalize_pixels = config.get('normalize_pixels', 1)
    fire_class_ids = config.get('fire_class_ids', [])

    if not data_root or not target_size: return np.array([]), np.array([])

    image_label_pairs = []
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp') # Ensure it's a tuple for endswith
    annotation_extension = '.txt'
    images_dir = os.path.join(data_root, 'images')
    labels_dir = os.path.join(data_root, 'labels')

    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        print(f"Error: Images or Labels directory not found in {data_root}")
        return np.array([]), np.array([])

    all_image_files = [f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in img_extensions]

    if not all_image_files:
        print(f"No image files found in {images_dir}")
        return np.array([]), np.array([])

    print("Determining Labels...")
    for filename in tqdm(all_image_files, desc="Determining Labels", leave=False):
        image_name_without_ext = os.path.splitext(filename)[0]
        annotation_path = os.path.join(labels_dir, image_name_without_ext + annotation_extension)
        label = 1 if is_dfire_image_fire(annotation_path, fire_class_ids) else 0
        image_label_pairs.append((os.path.join(images_dir, filename), label))
    print("Label determination complete.")

    if not image_label_pairs:
        print("No images with labels found.")
        return np.array([]), np.array([])

    all_features_list = []
    all_labels_list = []
    total_images_processed = 0
    total_images_skipped_reading = 0
    total_images_skipped_feature = 0

    print("Loading images and extracting features...")
    for image_path, label in tqdm(image_label_pairs, desc="Memory-safe Feature Extraction", leave=False):
        img_bgr = cv2.imread(image_path)

        if img_bgr is None:
            total_images_skipped_reading += 1
            continue

        try:
            img_resized = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_LINEAR)
            img_dict_single = {}
            img_processed_bgr = None
            if normalize_pixels:
                img_processed_bgr = img_resized.astype(np.float32) / 255.0
                img_resized_uint8 = img_resized.astype(np.uint8)
                img_gray = cv2.cvtColor(img_resized_uint8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                if 'hsv' in color_spaces_to_load:
                    img_dict_single['hsv'] = cv2.cvtColor(img_resized_uint8, cv2.COLOR_BGR2HSV).astype(np.float32) / np.array([180, 255, 255], dtype=np.float32)
                if 'ycbcr' in color_spaces_to_load:
                    img_dict_single['ycbcr'] = cv2.cvtColor(img_resized_uint8, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
            else:
                img_processed_bgr = img_resized.astype(np.uint8)
                img_gray = cv2.cvtColor(img_processed_bgr, cv2.COLOR_BGR2GRAY)
                if 'hsv' in color_spaces_to_load:
                    img_dict_single['hsv'] = cv2.cvtColor(img_processed_bgr, cv2.COLOR_BGR2HSV)
                if 'ycbcr' in color_spaces_to_load:
                    img_dict_single['ycbcr'] = cv2.cvtColor(img_processed_bgr, cv2.COLOR_BGR2YCrCb)

            img_dict_single['gray'] = img_gray
            if 'bgr' in color_spaces_to_load:
                img_dict_single['bgr'] = img_processed_bgr

            features_single = combine_features(img_dict_single, feature_params)

            if features_single.size > 0:
                all_features_list.append(features_single)
                all_labels_list.append(label)
                total_images_processed += 1
            else:
                total_images_skipped_feature += 1

        except Exception as e:
            total_images_skipped_feature += 1


    print("Feature extraction complete.")
    print(f"Total images initially found: {len(image_label_pairs)}")
    print(f"Images skipped (read error): {total_images_skipped_reading}")
    print(f"Images skipped (feature error): {total_images_skipped_feature}")
    print(f"Images successfully processed: {total_images_processed}")

    if not all_features_list:
        print("No features extracted from any image.")
        return np.array([]), np.array([])

    features_array = np.array(all_features_list, dtype=np.float32)
    labels_array = np.array(all_labels_list, dtype=np.int32)

    print(f"Final features array shape: {features_array.shape}")
    print(f"Final labels array shape: {labels_array.shape}")

    return features_array, labels_array

def get_config(dataset_choice):
    config = {}
    if dataset_choice == 'kaggle':
        config['dataset_choice'] = 'kaggle'
        config['data_root'] = os.path.join('..', 'fire_dataset')
        config['target_img_size'] = (128, 128)
        config['color_spaces_to_load'] = ['bgr', 'hsv', 'ycbcr']
        config['normalize_pixels'] = 1
        config['fire_class_ids'] = None
    elif dataset_choice == 'dfire':
        config['dataset_choice'] = 'dfire'
        config['dfire_root'] = os.path.join('..', 'data_subsets', 'D-Fire')
        config['split_name'] = "train"
        config['data_root'] = os.path.join(config['dfire_root'], config['split_name'])
        config['target_img_size'] = (128, 128)
        config['color_spaces_to_load'] = ['bgr', 'hsv', 'ycbcr']
        config['normalize_pixels'] = 1
        config['fire_class_ids'] = [0, 1]
    else:
        raise ValueError(f"Unknown dataset choice: {dataset_choice}. Choose 'kaggle' or 'dfire'.")

    print(f"Using dataset: {config['dataset_choice']}")
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
        'lbp_method': 'uniform',
        'hog_orientations': 9,
        'hog_pixels_per_cell': (8, 8),
        'hog_cells_per_block': (2, 2),
        'hog_block_norm': 'L2-Hys'
    }
    print("\nFeature extraction parameters:", feature_params)
    return feature_params

def split_data(features_array, labels_array, test_size=0.2, random_state=42):
    if features_array.shape[0] == 0:
        print("Feature array is empty, cannot split.")
        return None, None, None, None

    print(f"\nSplitting data: training ({1-test_size:.0%}) testing ({test_size:.0%})")

    X_train, X_test, y_train, y_test = train_test_split(
        features_array,
        labels_array,
        test_size=test_size,
        random_state=random_state,
        stratify=labels_array
    )

    print(f"Training features shape: {X_train.shape}")
    print(f"Testing features shape: {X_test.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Testing labels shape: {y_test.shape}")

    train_labels, train_counts = np.unique(y_train, return_counts=True)
    test_labels, test_counts = np.unique(y_test, return_counts=True)
    print(f"Train label distribution: {dict(zip(train_labels, train_counts))}")
    print(f"Test label distribution: {dict(zip(test_labels, test_counts))}")

    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    if X_train is None or X_test is None or X_train.shape[0] == 0:
         print("Scaling skipped: train or test data is empty.")
         return None, None, None
    print("\nScaling features...")
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Scaling complete.")
    return X_train_scaled, X_test_scaled, scaler

def perform_correlation_selection(X_train, y_train, X_test, k_features):
    if X_train is None or X_test is None or X_train.shape[0] == 0:
         print("Correlation Selection skipped: train or test data is empty.")
         return None, None, None

    n_total_features = X_train.shape[1]
    k_features_int = k_features

    percentage_str = None # Initialize to None
    if isinstance(k_features, str) and k_features.endswith('%'):
        try:
            percentage = float(k_features[:-1]) / 100.0
            k_features_int = max(1, int(n_total_features * percentage))
            percentage_str = k_features # Keep original string for printing
            print(f"Selecting top {k_features_int} features based on {percentage_str} percentage using Correlation...")
        except ValueError:
            print(f"Invalid percentage string for k_features: {k_features}. Skipping selection.")
            return X_train, X_test, None
    elif k_features == 'all':
         print("Selecting all features (no correlation selection)...\n")
         return X_train, X_test, None
    elif isinstance(k_features, int) and k_features > 0:
        k_features_int = min(k_features, n_total_features)
        print(f"Selecting top {k_features_int} features by correlation...")
    else:
        print(f"Invalid k_features value: {k_features}. Must be int > 0, 'all', or percentage string (e.g., '75%'). Skipping selection.")
        return X_train, X_test, None

    if k_features_int <= 0 or k_features_int > n_total_features:
         print(f"Calculated number of features to select ({k_features_int}) is invalid. Skipping selection.")
         return X_train, X_test, None
    if k_features_int == n_total_features:
         print("Number of features to select is equal to total features. Skipping selection.\n")
         return X_train, X_test, None

    selector = SelectKBest(score_func=f_classif, k=k_features_int)
    selector.fit(X_train, y_train)

    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    print(f"Original feature shape: {X_train.shape}")
    print(f"Selected feature shape: {X_train_selected.shape}")

    return X_train_selected, X_test_selected, selector

def perform_rfe_selection(X_train, y_train, X_test, n_features_to_select, step=0.1, estimator=None):
    if X_train is None or X_test is None or X_train.shape[0] == 0:
         print("RFE Selection skipped: train or test data is empty.")
         return None, None, None

    n_total_features = X_train.shape[1]
    n_features_int = n_features_to_select

    if estimator is None:
        estimator = LogisticRegression(solver='liblinear', random_state=42, max_iter=2000)

    percentage_str = None # Initialize to None
    if isinstance(n_features_to_select, str) and n_features_to_select.endswith('%'):
        try:
            percentage = float(n_features_to_select[:-1]) / 100.0
            n_features_int = max(1, int(n_total_features * percentage))
            percentage_str = n_features_to_select # Keep original string for printing
            print(f"Selecting top {n_features_int} features based on {percentage_str} percentage using RFE...")
        except ValueError:
            print(f"Invalid percentage string for n_features_to_select: {n_features_to_select}. Skipping RFE.")
            return X_train, X_test, None
    elif isinstance(n_features_to_select, int) and n_features_to_select > 0:
        n_features_int = min(n_features_to_select, n_total_features)
        print(f"Selecting {n_features_int} features using RFE...")
    elif n_features_to_select == 'auto':
        print("RFE with 'auto' feature selection requires RFECV, which is not implemented in this helper. Skipping selection.")
        return X_train, X_test, None
    else:
        print(f"Invalid n_features_to_select value: {n_features_to_select}. Skipping RFE.")
        return X_train, X_test, None

    if n_features_int <= 0 or n_features_int > n_total_features:
        print(f"Calculated number of features for RFE ({n_features_int}) is invalid. Skipping selection.")
        return X_train, X_test, None
    if n_features_int == n_total_features:
        print("Number of features to select is equal to total features. Skipping selection.\n")
        return X_train, X_test, None

    try:
        rfe = RFE(estimator=estimator, n_features_to_select=n_features_int, step=step)
        rfe.fit(X_train, y_train)

        X_train_selected = rfe.transform(X_train)
        X_test_selected = rfe.transform(X_test)

        print(f"Original feature shape: {X_train.shape}")
        print(f"Selected feature shape: {X_train_selected.shape}")

        return X_train_selected, X_test_selected, rfe
    except Exception as e:
        print(f"Error during RFE fit/transform: {e}")
        return X_train, X_test, None

def tune_model_hyperparameters(model_estimator, X_train, y_train, param_grid, cv_strategy, scoring='f1', search_method='GridSearch'):
    if X_train is None or y_train is None or X_train.shape[0] == 0:
        print("Hyperparameter tuning skipped: training data is empty.")
        return None

    print(f"\nPerforming {search_method} tuning (scoring='{scoring}')...")
    start_time = time.time()

    if search_method == 'GridSearch':
        search_cv = GridSearchCV(
            estimator=model_estimator,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=4,
            verbose=1
        )
    elif search_method == 'RandomSearch':
         n_iter_search = 20
         search_cv = RandomizedSearchCV(
            estimator=model_estimator,
            param_distributions=param_grid,
            n_iter=n_iter_search,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=4,
            verbose=1,
            random_state=42
         )
    else:
        print(f"Unknown search_method: {search_method}. Use 'GridSearch' or 'RandomSearch'.")
        return None

    search_cv.fit(X_train, y_train)

    end_time = time.time()
    print(f"{search_method} duration: {end_time - start_time:.2f} seconds")
    print("\nBest parameters found:")
    print(search_cv.best_params_)
    print("\nBest CV score:")
    print(search_cv.best_score_)

    return search_cv

def evaluate_model(model, X_test, y_test, model_name="Model", feature_set_name="Unknown Feature Set"):
    if model is None or X_test is None or y_test is None or X_test.shape[0] == 0:
        print(f"{model_name} evaluation skipped on {feature_set_name}: model not trained or test data is empty.")
        return {}

    print(f"\nEvaluating {model_name} on the test set using {feature_set_name}...")
    start_time = time.time()
    if isinstance(model, tf.keras.Model): # For raw Keras models used directly
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int)
    else: # For scikit-learn models and KerasClassifier wrappers
        y_pred = model.predict(X_test)
    end_time = time.time()
    print(f"Prediction duration: {end_time - start_time:.4f} seconds")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix ({model_name} on {feature_set_name}):")
    print(conf_matrix)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix.tolist()
    }

def perform_pca_dimension_reduction(X_train, X_test, n_components):
    if X_train is None or X_test is None or X_train.shape[0] == 0:
        print("PCA skipped bc the data is empty..")
        return None, None, None

    try:
        n_total_features = X_train.shape[1]
        if isinstance(n_components, float) and 0 < n_components < 1:
            print(f"Applying PCA to retain {n_components:.0%} of variance...")
        elif isinstance(n_components, int) and 0 < n_components < n_total_features:
            print(f"Applying PCA to reduce to {n_components} components...")
        else:
            print(f"Invalid n_components value: {n_components}. Skipping PCA.")
            return X_train, X_test, None
        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        print(f"Original feature shape: {X_train.shape}")
        print(f"PCA transformed feature shape: {X_train_pca.shape}")
        print(f"Explained variance ratio with {pca.n_components_} components: {np.sum(pca.explained_variance_ratio_):.4f}")

        return X_train_pca, X_test_pca, pca
    except Exception as e:
        print(f"Error during PCA: {e}")
        return X_train, X_test, None

def create_custom_mlp(hidden_layer_1_neurons=128, hidden_layer_2_neurons=64,
                        dropout_rate=0.3, activation='leaky_relu', learning_rate=0.001,
                        meta=None):
    n_features_in = meta["n_features_in_"]

    model = Sequential()
    model.add(Input(shape=(n_features_in,)))
    model.add(Dense(hidden_layer_1_neurons))
    model.add(BatchNormalization())
    if activation == 'leaky_relu': model.add(LeakyReLU(alpha=0.1))
    else: model.add(tf.keras.layers.ReLU())
    model.add(Dropout(dropout_rate))
    if hidden_layer_2_neurons is not None and hidden_layer_2_neurons > 0:
        model.add(Dense(hidden_layer_2_neurons))
        model.add(BatchNormalization())
        if activation == 'leaky_relu': model.add(LeakyReLU(alpha=0.1))
        else: model.add(tf.keras.layers.ReLU())
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# --- New Helper for single image feature extraction ---
def extract_features_single_image(image_path_single, feature_params_single, config_single):
    img_bgr = cv2.imread(image_path_single)
    if img_bgr is None: return np.array([])

    img_resized = cv2.resize(img_bgr, config_single['target_img_size'], interpolation=cv2.INTER_LINEAR)
    img_dict_single = {}

    # Need to normalize based on the original training setting
    if config_single['normalize_pixels']:
        img_resized_uint8 = img_resized.astype(np.uint8) # Convert to uint8 for color space conversion if needed
        img_gray = cv2.cvtColor(img_resized_uint8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        # HSV normalization for float32: H range 0-180, S/V range 0-255 -> divide by [180, 255, 255]
        img_dict_single['hsv'] = cv2.cvtColor(img_resized_uint8, cv2.COLOR_BGR2HSV).astype(np.float32) / np.array([180, 255, 255], dtype=np.float32)
        img_dict_single['ycbcr'] = cv2.cvtColor(img_resized_uint8, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
    else:
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        img_dict_single['hsv'] = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        img_dict_single['ycbcr'] = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YCrCb)

    img_dict_single['gray'] = img_gray
    return combine_features(img_dict_single, feature_params_single)


print("Imports and helper functions loaded.")

# --- Phase 1: Data Loading, Splitting, Scaling, and Feature Engineering ---

dataset_choice = 'dfire' # Explicitly set to D-Fire

try:
    config = get_config(dataset_choice)
    feature_params = get_feature_params()
    features_array_orig, labels_array = load_and_extract_features_memory_safe(config, feature_params)

except ValueError as e:
    print(f"Configuration Error: {e}")
    features_array_orig = np.array([])
    labels_array = np.array([])
except FileNotFoundError as e:
    print(f"File Not Found Error: {e}. Please check data paths in get_config.")
    features_array_orig = np.array([])
    labels_array = np.array([])
except Exception as e:
    print(f"An unexpected error occurred during data loading and feature extraction: {e}")
    features_array_orig = np.array([])
    labels_array = np.array([])

if features_array_orig.shape[0] == 0:
    print("No features loaded. Cannot proceed with splitting or modeling.")
    X_train_orig, X_test_orig, y_train, y_test = None, None, None, None
    X_train_scaled, X_test_scaled, scaler = None, None, None
    feature_sets = {}
    feature_transformers = {}
else:
    print("\n--- Splitting Data ---")
    X_train_orig, X_test_orig, y_train, y_test = split_data(features_array_orig, labels_array, test_size=0.2, random_state=42)
    print("\n--- Scaling Features (Initial) ---")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_orig, X_test_orig)
    feature_sets = {'Scaled_All': (X_train_scaled, X_test_scaled)}
    feature_transformers = {'Scaled_All': scaler} # Store the global scaler here

    # Feature Engineering (Selection)
    if X_train_scaled is not None and y_train is not None:
        print("\n--- Performing Feature Engineering (Selection) ---")
        original_feature_count = X_train_scaled.shape[1]
        print(f"Starting with {original_feature_count} features after scaling.")
        corr_feature_percentages = ['75%', '50%']

        for percentage_str in corr_feature_percentages:
            print(f"\nAttempting Correlation Selection with {percentage_str}...")
            try:
                X_train_corr, X_test_corr, corr_selector = perform_correlation_selection(
                    X_train_scaled, y_train, X_test_scaled, k_features=percentage_str
                )
                if X_train_corr is not None and X_train_corr.shape[1] < original_feature_count:
                    feature_sets[f'Scaled_Corr{percentage_str}'] = (X_train_corr, X_test_corr)
                    feature_transformers[f'Scaled_Corr{percentage_str}'] = corr_selector
                else:
                     print(f"Correlation Selection with {percentage_str} did not reduce features or failed.")
            except Exception as e:
                print(f"Error during Correlation Selection ({percentage_str}): {e}")

        rfe_feature_percentages = ['75%', '50%']
        rfe_step_val = 0.1
        rfe_estimator = LogisticRegression(solver='liblinear', random_state=42, max_iter=2000)
        for percentage_str in rfe_feature_percentages:
             print(f"\nAttempting RFE Selection with {percentage_str} (step={rfe_step_val})...")
             try:
                X_train_rfe, X_test_rfe, rfe_selector = perform_rfe_selection(
                    X_train_scaled, y_train, X_test_scaled, n_features_to_select=percentage_str, step=rfe_step_val, estimator=rfe_estimator
                )
                if X_train_rfe is not None and X_train_rfe.shape[1] < original_feature_count:
                    feature_sets[f'Scaled_RFE{percentage_str}'] = (X_train_rfe, X_test_rfe)
                    feature_transformers[f'Scaled_RFE{percentage_str}'] = rfe_selector
                else:
                     print(f"RFE Selection with {percentage_str} did not reduce features or failed.")
             except Exception as e:
                print(f"Error during RFE Selection ({percentage_str}): {e}")

        print("\n--- Available Feature Sets for Tuning ---")
        for name, (X_train_fs, _) in feature_sets.items():
            print(f"- {name}: {X_train_fs.shape[1]} features, before PCA..")

        pca_components= [0.95, 1000] # aiming for .95 variance retention & 1000 feature for first trial
        for n_comp in pca_components:
            print(f"\nPCA with n_components={n_comp}...")
            try:
                X_train_pca, X_test_pca, pca_transformer = perform_pca_dimension_reduction(X_train_scaled, X_test_scaled, n_components=n_comp)
                if X_train_pca is not None and X_train_pca.shape[1] < original_feature_count:
                    fs_name_suffix = f"{int(n_comp*100)}%" if isinstance(n_comp, float) else str(n_comp)
                    fs_name = f'Scaled_PCA_{fs_name_suffix}'
                    feature_sets[fs_name] = (X_train_pca, X_test_pca)
                    feature_transformers[fs_name] = pca_transformer
                else:
                    print(f"n_components={n_comp} failed..")
            except Exception as e:
                print(f"error during n_components={n_comp}: {e}")

        print("\n--- AFTER PCA: ---")
        for name, (X_train_fs, _) in feature_sets.items():
            print(f"- {name}: {X_train_fs.shape[1]} features")

    else:
        print("Skipping feature engineering as scaled data is not available.")


# --- Phase 2: Model Training and Hyperparameter Tuning ---

if not feature_sets or y_train is None:
    print("Skipping model training and tuning: No feature sets available or labels are missing.")
else:
    print("\n--- Starting Model Training and Hyperparameter Tuning ---")
    models_to_tune = {
        'SVM': {
            'estimator': SVC(random_state=42),
            'param_grid': {
                'C': [0.1, 1, 10, 50],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear']
            }
        },
        'LightGBM': {
            'estimator': lgb.LGBMClassifier(random_state=42, objective='binary', metric='binary_logloss', verbosity=-1),
            'param_grid': {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [-1, 10, 20],
                'num_leaves': [31, 50, 70],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'min_split_gain': [0.1],
                'min_child_samples': [5]
            }
        },
        'Custom_MLP': {
            'estimator': KerasClassifier(
                model=create_custom_mlp,
                loss=tf.keras.losses.BinaryCrossentropy,
                epochs=100,
                batch_size=32,
                verbose=1,
                validation_split=0.2,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=0, restore_best_weights=True)]
            ),
            'param_grid': {
                'model__hidden_layer_1_neurons': [64, 128, 256],
                'model__hidden_layer_2_neurons': [0, 64, 128],
                'model__dropout_rate': [0.2, 0.4, 0.6],
                'model__activation': ['relu', 'leaky_relu'],
                'optimizer__learning_rate': [0.001, 0.005, 0.01]
            }
        }
    }
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring_metric = 'f1'
    all_results = {}
    best_overall_test_score = -np.inf
    best_overall_combination = None
    best_overall_trained_model = None
    best_overall_X_test = None
    best_overall_transformer = None

    for model_name, model_config in models_to_tune.items():
        all_results[model_name] = {}
        estimator = model_config['estimator']
        param_grid = model_config['param_grid']

        print(f"\n\n=== Training and Tuning {model_name} ===")
        for fs_name, (X_train_fs, X_test_fs) in feature_sets.items():
            print(f"\n--- Tuning {model_name} on Feature Set: {fs_name} ({X_train_fs.shape[1]} features) ---")
            if X_train_fs is None or X_train_fs.shape[0] == 0:
                print(f"Skipping tuning for {model_name} on {fs_name}: Training data is empty.")
                continue
            tuned_search = tune_model_hyperparameters(
                estimator,
                X_train_fs,
                y_train,
                param_grid=param_grid,
                cv_strategy=cv_strategy,
                scoring=scoring_metric,
                search_method='RandomSearch'
            )

            if tuned_search:
                best_model_for_combination = tuned_search.best_estimator_
                best_cv_score = tuned_search.best_score_
                best_params = tuned_search.best_params_
                print(f"Best CV {scoring_metric} for {model_name} on {fs_name}: {best_cv_score:.4f}")
                test_metrics = evaluate_model(best_model_for_combination, X_test_fs, y_test, model_name, fs_name)
                all_results[model_name][fs_name] = {
                    'best_cv_score': best_cv_score,
                    'best_params': best_params,
                    'test_metrics': test_metrics,
                    'trained_model': best_model_for_combination,
                    'transformer': feature_transformers.get(fs_name)
                }

                if test_metrics and test_metrics.get('f1_score', -np.inf) > best_overall_test_score:
                    best_overall_test_score = test_metrics['f1_score']
                    best_overall_combination = (model_name, fs_name)
                    best_overall_trained_model = best_model_for_combination
                    best_overall_X_test = X_test_fs
                    best_overall_transformer = feature_transformers.get(fs_name)

    # Determine the best feature set for each model based on test F1 score for saving
    best_f1_per_model = {}
    for model_name, fs_results in all_results.items():
        if fs_results:
            best_f1_for_this_model = -np.inf
            best_fs_name_for_this_model = None
            for fs_name, result in fs_results.items():
                current_test_f1 = result.get('test_metrics', {}).get('f1_score', -np.inf)
                if current_test_f1 > best_f1_for_this_model:
                    best_f1_for_this_model = current_test_f1
                    best_fs_name_for_this_model = fs_name
            if best_fs_name_for_this_model:
                best_f1_per_model[model_name] = (best_fs_name_for_this_model, best_f1_for_this_model)
    print("\n--- Model Training and Hyperparameter Tuning Complete ---")

# --- Results Summary (As before) ---
print("\n\n=== Results Summary Across Models and Feature Sets ===")

if 'all_results' not in locals() or not all_results:
    print("No results available to summarize.")
else:
    print("\nCross-Validation Results (Best CV F1 Score):")
    print("-------------------------------------------------")
    for model_name, fs_results in all_results.items():
        print(f"\n{model_name}:")
        if fs_results:
            for fs_name, result in fs_results.items():
                 cv_score = result.get('best_cv_score', float('nan'))
                 print(f"  - {fs_name}: {cv_score:.4f}")
        else:
            print("  No results for this model.")

    print("\nTest Set Results (F1 Score):")
    print("----------------------------")
    for model_name, fs_results in all_results.items():
        print(f"\n{model_name}:")
        if fs_results:
            for fs_name, result in fs_results.items():
                 test_f1 = result.get('test_metrics', {}).get('f1_score', float('nan'))
                 print(f"  - {fs_name}: {test_f1:.4f}")
        else:
            print("  No test results for this model.")

    print("\n=== Overall Best Combination on Test Set (Based on F1 Score) ===")
    if best_overall_combination:
        model_name, fs_name = best_overall_combination
        best_result = all_results[model_name][fs_name]
        test_metrics = best_result['test_metrics']
        print(f"Best Model: {model_name}")
        print(f"Best Feature Set: {fs_name} ({feature_sets[fs_name][0].shape[1]} features)")
        print(f"Best CV F1 Score: {best_result['best_cv_score']:.4f}")
        print(f"Test F1 Score: {test_metrics['f1_score']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        print(f"Best Parameters: {best_result['best_params']}")
        print(f"Confusion Matrix:\n{np.array(test_metrics['confusion_matrix'])}")
    else:
        print("No successful model tuning and evaluation completed to determine the best combination.")


# --- Phase 3: Model Saving ---
MODEL_SAVE_DIR = os.path.join('..', 'models')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
print("\n--- Saving Best Model Per Algorithm (Based on Test F1) ---")

# Save the initial StandardScaler
if 'scaler' in locals() and scaler is not None:
    try:
        # Saving with a generic DFire name as per your file list
        joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, 'dfires1_scaler.pkl'))
        print(f"   Saved global StandardScaler: {os.path.join(MODEL_SAVE_DIR, 'dfires1_scaler.pkl')}")
    except Exception as e:
        print(f"   Error saving global StandardScaler: {e}")
else:
    print("   Global StandardScaler not found or is None, skipping save.")

# Now, save the best model and its associated feature transformer for each algorithm
if 'best_f1_per_model' not in locals() or not best_f1_per_model:
     print("Could not determine best feature set per model. No 'best_f1_per_model' found or it is empty.")
else:
    for model_name, (best_fs_name_for_model, best_test_f1_for_model) in best_f1_per_model.items():
        print(f"\nProcessing {model_name}...")
        # Check if the model and feature set combination actually exists in all_results
        if model_name in all_results and best_fs_name_for_model in all_results[model_name]:
            best_combination_results = all_results[model_name][best_fs_name_for_model]
            model_to_save = best_combination_results.get('trained_model')
            transformer_to_save = best_combination_results.get('transformer') # This is the specific selector/PCA

            # Determine the filename based on the model and feature set, matching your screenshot
            if model_name == 'SVM' and 'PCA' in best_fs_name_for_model:
                model_filename_prefix = 'dfires1_best_svm_model_pca'
                transformer_filename_prefix = 'dfires1_scaler_pca' # Assuming this is for PCA
            elif model_name == 'SVM' and 'All' in best_fs_name_for_model:
                model_filename_prefix = 'Kaggle_svm_best_model_Scaled_All'
                transformer_filename_prefix = None # No specific transformer
            elif model_name == 'SVM': # For other SVMs (like dfires1_best_svm_model.pkl)
                 model_filename_prefix = 'dfires1_best_svm_model'
                 transformer_filename_prefix = None # No specific transformer
            elif model_name == 'LightGBM':
                model_filename_prefix = f'Kaggle_lightgbm_best_model_{best_fs_name_for_model}'
                transformer_filename_prefix = f'Kaggle_selector_{best_fs_name_for_model}'
            elif model_name == 'Custom_MLP':
                model_filename_prefix = f'Kaggle_custom_mlp_best_model_{best_fs_name_for_model}'
                transformer_filename_prefix = f'Kaggle_selector_{best_fs_name_for_model}' # Selector for MLP
            else:
                model_filename_prefix = f'{model_name.lower()}_best_model_{best_fs_name_for_model}'
                transformer_filename_prefix = f'{model_name.lower()}_transformer_{best_fs_name_for_model}'


            if model_to_save:
                is_keras_classifier = isinstance(model_to_save, KerasClassifier)
                file_extension = '.keras' if is_keras_classifier else '.pkl'
                model_filename = f'{model_filename_prefix}{file_extension}'
                MODEL_SAVE_PATH_ALG = os.path.join(MODEL_SAVE_DIR, model_filename)

                try:
                    if is_keras_classifier:
                        model_to_save.model_.save(MODEL_SAVE_PATH_ALG)
                        print(f"   Saved Keras model: {MODEL_SAVE_PATH_ALG}")
                    else:
                        joblib.dump(model_to_save, MODEL_SAVE_PATH_ALG)
                        print(f"   Saved scikit-learn model: {MODEL_SAVE_PATH_ALG}")
                except Exception as e:
                    print(f"   Error saving {model_name} model to {MODEL_SAVE_PATH_ALG}: {e}")
            else:
                print(f"   No trained model found for {model_name} on {best_fs_name_for_model}.")

            # Save the specific feature selection/reduction transformer if it exists and is not the global scaler
            if transformer_to_save is not None and transformer_to_save is not scaler:
                 # Special handling for specific file names based on screenshot
                 if model_name == 'SVM' and 'PCA' in best_fs_name_for_model:
                     transformer_filename = 'dfires1_scaler_pca.pkl'
                 elif transformer_filename_prefix:
                     transformer_filename = f'{transformer_filename_prefix}.pkl'
                 else:
                     transformer_filename = None # No specific transformer to save

                 if transformer_filename:
                    TRANSFORMER_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, transformer_filename)
                    try:
                        joblib.dump(transformer_to_save, TRANSFORMER_SAVE_PATH)
                        print(f"   Saved feature transformer: {TRANSFORMER_SAVE_PATH}")
                    except Exception as e:
                       print(f"   Error saving transformer for {best_fs_name_for_model} to {TRANSFORMER_SAVE_PATH}: {e}")
                 else:
                    print(f"   No specific feature selection transformer to save for {best_fs_name_for_model}.")
            elif transformer_to_save is scaler:
                 print(f"   Transformer for {best_fs_name_for_model} is the global scaler, already saved.")

        else:
            print(f"No valid results found in all_results for the best feature set '{best_fs_name_for_model}' for model {model_name}.")

print("\n--- Saving Process Complete ---")


# --- Phase 4: Comprehensive Evaluation for Feature-Based Models ---

# Configuration for D-Fire dataset (aligned with your notebook's overall setup)
# (Re-defining for clarity in standalone script, assumes DFIRE_CONFIG is correct)
DFIRE_CONFIG_EVAL = {
    'fire_class_ids': [0, 1],
    'target_img_size': (128, 128),
    'test_size': 0.2, # For train_test_split reproduction
    'random_state': 42,
    'img_extensions': ('.png', '.jpg', '.jpeg', '.bmp', '.gif'),
    'annotation_extension': '.txt',
    'model_dir': os.path.join('..', 'models'), # This points to the main models folder
    'color_spaces_to_load': ['bgr', 'hsv', 'ycbcr'], # Important for feature extraction consistency
    'normalize_pixels': 1 # Important for feature extraction consistency
}

MODEL_SAVE_DIR_EVAL = DFIRE_CONFIG_EVAL['model_dir']
DFIRE_ROOT_EVAL = os.path.join('..', 'data_subsets', 'D-Fire')
DFIRE_TRAIN_ROOT_EVAL = os.path.join(DFIRE_ROOT_EVAL, 'train') # For reproducing train split
DFIRE_TEST_ROOT_EVAL = os.path.join(DFIRE_ROOT_EVAL, 'test') # For dedicated test folder evaluation
DFIRE_TEST_IMAGES_DIR_EVAL = os.path.join(DFIRE_TEST_ROOT_EVAL, 'images')
DFIRE_TEST_LABELS_DIR_EVAL = os.path.join(DFIRE_TEST_ROOT_EVAL, 'labels')


def load_artifacts_for_feature_models_eval(model_config_list, model_dir):
    """Loads all specified models and their associated transformers for evaluation."""
    artifacts = {}
    print("\n--- Loading Models and Transformers for Evaluation ---")
    for item in tqdm(model_config_list, desc="Loading evaluation artifacts"):
        model_display_name = item['display_name']
        model_filename = item['model_filename']
        transformer_filename = item['transformer_filename']

        model_path = os.path.join(model_dir, model_filename)
        loaded_model = None
        specific_transformer = None

        try:
            if '.keras' in model_filename:
                loaded_model_wrapper = tf.keras.models.load_model(
                    model_path,
                    custom_objects={'create_custom_mlp': create_custom_mlp,
                                    'KerasClassifier': KerasClassifier,
                                    'LeakyReLU': tf.keras.layers.LeakyReLU,
                                    'ReLU': tf.keras.layers.ReLU}
                )
                loaded_model = loaded_model_wrapper.model_ if hasattr(loaded_model_wrapper, 'model_') else loaded_model_wrapper
            else:
                loaded_model = joblib.load(model_path)

            if transformer_filename:
                transformer_path = os.path.join(model_dir, transformer_filename)
                specific_transformer = joblib.load(transformer_path)

            artifacts[model_display_name] = {
                'model': loaded_model,
                'transformer': specific_transformer,
                'filename': model_filename
            }
        except FileNotFoundError:
            print(f"  Warning: File not found for {model_display_name} ({model_filename} or {transformer_filename}). Skipping.")
        except Exception as e:
            print(f"  Error loading {model_display_name} artifact: {e}. Skipping.")
    return artifacts

def reproduce_original_test_split_features_eval(dfire_train_root, config, feature_params, global_scaler, artifacts_dict):
    """
    Reproduces a train-test split from the D-Fire 'train' data
    and evaluates each loaded model on its 'test' portion.
    """
    print("\n--- Reproducing Original Test Set Results from D-Fire/train data (Feature Models) ---")

    print("  Loading and extracting features from D-Fire 'train' split...")
    all_train_features_raw, all_train_labels = load_and_extract_features_memory_safe(
        {'dataset_choice': 'dfire', 'data_root': dfire_train_root,
         'target_img_size': config['target_img_size'],
         'color_spaces_to_load': config['color_spaces_to_load'],
         'normalize_pixels': config['normalize_pixels'],
         'fire_class_ids': config['fire_class_ids']},
        feature_params
    )

    if all_train_features_raw.shape[0] == 0:
        print("  No features loaded from 'train' split. Cannot reproduce test results.")
        return

    all_train_features_scaled = global_scaler.transform(all_train_features_raw)

    X_train_reproduced, X_test_reproduced, y_train_reproduced, y_test_reproduced = train_test_split(
        all_train_features_scaled, all_train_labels,
        test_size=config['test_size'],
        random_state=config['random_state'],
        stratify=all_train_labels
    )
    print(f"  Successfully recreated test split with {X_test_reproduced.shape[0]} samples from 'train' data.")

    for model_name, model_data in artifacts_dict.items():
        print(f"\n  Evaluating {model_name} on recreated test data...")
        model = model_data['model']
        transformer = model_data['transformer']

        X_test_for_model = X_test_reproduced
        if transformer is not None:
            try:
                X_test_for_model = transformer.transform(X_test_reproduced)
                print(f"    Applied model-specific transformer to recreated test data.")
            except Exception as e:
                print(f"    Error applying transformer for {model_name}: {e}. Using globally scaled data.")

        if model is not None:
            try:
                if isinstance(model, tf.keras.Model):
                    y_pred_proba = model.predict(X_test_for_model, verbose=0)
                    y_pred = (y_pred_proba > 0.5).astype(int)
                else:
                    y_pred = model.predict(X_test_for_model)

                accuracy = accuracy_score(y_test_reproduced, y_pred)
                precision = precision_score(y_test_reproduced, y_pred, zero_division=0)
                recall = recall_score(y_test_reproduced, y_pred, zero_division=0)
                f1 = f1_score(y_test_reproduced, y_pred, zero_division=0)
                conf_matrix = confusion_matrix(y_test_reproduced, y_pred)

                print(f"    Accuracy: {accuracy:.4f}")
                print(f"    Precision: {precision:.4f}")
                print(f"    Recall (Sensitivity): {recall:.4f}")
                print(f"    F1 Score: {f1:.4f}")
                print(f"    Confusion Matrix:\n{conf_matrix}")
            except Exception as e:
                print(f"    Error during prediction or evaluation for {model_name}: {e}")
        else:
            print(f"  Model {model_name} was not loaded successfully.")

def evaluate_feature_folder_eval(images_folder_path, labels_folder_path, artifacts_dict, config, feature_params, global_scaler):
    """
    Loads features from a dedicated test folder, applies transformations,
    and evaluates each loaded model.
    """
    print(f"\n--- Evaluating models on dedicated test folder: {os.path.basename(images_folder_path)} (Feature Models) ---")

    test_data_config = {'dataset_choice': 'dfire', 'data_root': images_folder_path.replace('images', ''), # Adjust data_root for load_and_extract_features_memory_safe
                        'target_img_size': config['target_img_size'],
                        'color_spaces_to_load': config['color_spaces_to_load'],
                        'normalize_pixels': config['normalize_pixels'],
                        'fire_class_ids': config['fire_class_ids']}

    print("  Loading and extracting features from dedicated test folder...")
    X_test_raw_folder, y_test_folder = load_and_extract_features_memory_safe(test_data_config, feature_params)

    if X_test_raw_folder.shape[0] == 0:
        print("  No features loaded from dedicated test folder. Cannot evaluate.")
        return

    X_test_scaled_folder = global_scaler.transform(X_test_raw_folder)
    print(f"  Loaded {X_test_scaled_folder.shape[0]} samples from dedicated test folder.")

    for model_name, model_data in artifacts_dict.items():
        print(f"\n  Evaluating {model_name} on dedicated test folder data...")
        model = model_data['model']
        transformer = model_data['transformer']

        X_test_for_model = X_test_scaled_folder
        if transformer is not None:
            try:
                X_test_for_model = transformer.transform(X_test_scaled_folder)
                print(f"    Applied model-specific transformer to dedicated test data.")
            except Exception as e:
                print(f"    Error applying transformer for {model_name}: {e}. Using globally scaled data.")

        if model is not None:
            try:
                if isinstance(model, tf.keras.Model):
                    y_pred_proba = model.predict(X_test_for_model, verbose=0)
                    y_pred = (y_pred_proba > 0.5).astype(int)
                else:
                    y_pred = model.predict(X_test_for_model)

                accuracy = accuracy_score(y_test_folder, y_pred)
                precision = precision_score(y_test_folder, y_pred, zero_division=0)
                recall = recall_score(y_test_folder, y_pred, zero_division=0)
                f1 = f1_score(y_test_folder, y_pred, zero_division=0)
                conf_matrix = confusion_matrix(y_test_folder, y_pred)

                print(f"    Accuracy: {accuracy:.4f}")
                print(f"    Precision: {precision:.4f}")
                print(f"    Recall (Sensitivity): {recall:.4f}")
                print(f"    F1 Score: {f1:.4f}")
                print(f"    Confusion Matrix:\n{conf_matrix}")
            except Exception as e:
                print(f"    Error during prediction or evaluation for {model_name}: {e}")
        else:
            print(f"  Model {model_name} was not loaded successfully.")


def process_single_image_feature_model_eval(image_path, labels_root_dir, artifacts_dict, config, feature_params, global_scaler):
    """
    Processes a single image: extracts features, applies transformations,
    makes predictions with all loaded models, and displays the image.
    """
    print(f"\n--- Processing single image: {os.path.basename(image_path)} (Feature Models) ---")

    img_display = cv2.imread(image_path)
    if img_display is None:
        print(f"Error: Could not read image for display: {image_path}")
        return

    img_name_without_ext = os.path.splitext(os.path.basename(image_path))[0]
    annotation_path = os.path.join(labels_root_dir, img_name_without_ext + config['annotation_extension'])
    true_label_num = 1 if is_dfire_image_fire(annotation_path, config['fire_class_ids']) else 0
    true_label_text = 'Fire' if true_label_num == 1 else 'Non-Fire'

    raw_features_single = extract_features_single_image(image_path, feature_params, config)
    if raw_features_single.size == 0:
        print(f"  Failed to extract features for {os.path.basename(image_path)}. Skipping.")
        return

    raw_features_single = raw_features_single.reshape(1, -1) # Reshape for single sample (1, n_features)

    # Apply global scaling
    scaled_features_single = global_scaler.transform(raw_features_single)

    predictions_summary = []
    for model_name, model_data in artifacts_dict.items():
        model = model_data['model']
        transformer = model_data['transformer']

        features_for_prediction = scaled_features_single
        if transformer is not None:
            try:
                features_for_prediction = transformer.transform(scaled_features_single)
            except Exception as e:
                predictions_summary.append(f"{model_name}: Transform Error ({e})")
                continue

        if model is not None:
            try:
                if isinstance(model, tf.keras.Model):
                    prediction_proba = model.predict(features_for_prediction, verbose=0)
                    prediction = (prediction_proba > 0.5).astype(int)[0][0]
                else:
                    prediction = model.predict(features_for_prediction)[0]
                predictions_summary.append(f"{model_name}: {'Fire' if prediction == 1 else 'Non-Fire'}")
            except Exception as e:
                predictions_summary.append(f"{model_name}: Prediction Error ({e})")
        else:
            predictions_summary.append(f"{model_name}: Model Not Loaded")

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    title_text = f"Image: {os.path.basename(image_path)}\nTrue Label: {true_label_text}\n" + "\n".join(predictions_summary)
    plt.title(title_text)
    plt.axis('off')
    plt.show()


# --- Main Execution for Feature-Based Models Comprehensive Evaluation ---
print(f"\n\n=== D-Fire Feature-Based Model Comprehensive Evaluation ===")
print(f"DFIRE_ROOT set to: {DFIRE_ROOT_EVAL}")
print(f"DFIRE_TRAIN_ROOT set to: {DFIRE_TRAIN_ROOT_EVAL}")
print(f"DFIRE_TEST_ROOT set to: {DFIRE_TEST_ROOT_EVAL}")
print(f"MODEL_DIR set to: {MODEL_SAVE_DIR_EVAL}")

# Define the list of models and their associated transformers to evaluate
# Based *EXACTLY* on your provided screenshot (excluding CNN).
models_to_load_and_evaluate_final = [
    {
        'display_name': 'dfires1 SVM (PCA Model)',
        'model_filename': 'dfires1_best_svm_model_pca.pkl',
        'transformer_filename': 'dfires1_scaler_pca.pkl'
    },
    {
        'display_name': 'dfires1 SVM (Best Model)',
        'model_filename': 'dfires1_best_svm_model.pkl',
        'transformer_filename': None # Assumed no specific transformer beyond global scaler
    },
    {
        'display_name': 'dfires1 SVM (PCA Model v2)',
        'model_filename': 'dfires1_svm_pca_model.pkl',
        'transformer_filename': 'dfires1_scaler_pca.pkl' # Reusing the PCA scaler
    },
    {
        'display_name': 'Kaggle Custom MLP (Scaled PCA 500)',
        'model_filename': 'Kaggle_custom_mlp_best_model_Scaled_PCA_500.keras',
        'transformer_filename': 'Kaggle_selector_Scaled_PCA_500.pkl' # This will be the PCA object
    },
    {
        'display_name': 'Kaggle LightGBM (Scaled Corr 50%)',
        'model_filename': 'Kaggle_lightgbm_best_model_Scaled_Corr50%.pkl',
        'transformer_filename': 'Kaggle_selector_Scaled_Corr50%.pkl' # This will be the SelectKBest object
    },
    {
        'display_name': 'Kaggle SVM (Scaled All)',
        'model_filename': 'Kaggle_svm_best_model_Scaled_All.pkl',
        'transformer_filename': None # This model used all features after global scaling
    }
]

# Load the main global scaler first, as it's needed for all feature processing
global_scaler_filename = 'dfires1_scaler.pkl'
global_scaler_path = os.path.join(MODEL_SAVE_DIR_EVAL, global_scaler_filename)
global_scaler_obj_eval = None
try:
    global_scaler_obj_eval = joblib.load(global_scaler_path)
    print(f"Loaded global scaler: {global_scaler_path}")
except FileNotFoundError:
    print(f"Error: Global scaler not found at {global_scaler_path}. Feature-based evaluation cannot proceed.")
except Exception as e:
    print(f"Error loading global scaler: {e}. Feature-based evaluation cannot proceed.")


if global_scaler_obj_eval:
    # Load all models and their specific transformers
    loaded_artifacts_eval = load_artifacts_for_feature_models_eval(models_to_load_and_evaluate_final, MODEL_SAVE_DIR_EVAL)

    if loaded_artifacts_eval:
        print("\n--- Loaded feature-based models successfully. Proceeding with evaluations. ---")

        # 1. Reproduce results on a split from the D-Fire 'train' data
        reproduce_original_test_split_features_eval(DFIRE_TRAIN_ROOT_EVAL, DFIRE_CONFIG_EVAL, get_feature_params(), global_scaler_obj_eval, loaded_artifacts_eval)

        # 2. Evaluate on the dedicated D-Fire 'test' folder
        evaluate_feature_folder_eval(DFIRE_TEST_IMAGES_DIR_EVAL, DFIRE_TEST_LABELS_DIR_EVAL, loaded_artifacts_eval, DFIRE_CONFIG_EVAL, get_feature_params(), global_scaler_obj_eval)

        # 3. Prepare and process sample images from the dedicated D-Fire 'test' folder for visualization
        print("\n--- Preparing sample images for single image processing (Feature Models) ---")

        all_test_image_paths = [os.path.join(DFIRE_TEST_IMAGES_DIR_EVAL, f) for f in os.listdir(DFIRE_TEST_IMAGES_DIR_EVAL) if f.lower().endswith(DFIRE_CONFIG_EVAL['img_extensions'])]
        all_test_image_paths.sort() # Ensure consistent order

        fire_images_for_display = []
        non_fire_images_for_display = []

        for img_path in tqdm(all_test_image_paths, desc="Categorizing test images for display"):
            img_name_without_ext = os.path.splitext(os.path.basename(img_path))[0]
            annotation_path = os.path.join(DFIRE_TEST_LABELS_DIR_EVAL, img_name_without_ext + DFIRE_CONFIG_EVAL['annotation_extension'])
            if is_dfire_image_fire(annotation_path, DFIRE_CONFIG_EVAL['fire_class_ids']):
                fire_images_for_display.append(img_path)
            else:
                non_fire_images_for_display.append(img_path)

        num_samples_to_show = min(3, len(fire_images_for_display)) # Show up to 3 examples
        if num_samples_to_show > 0:
            print(f"\n--- Processing {num_samples_to_show} sample FIRE images for display ---")
            for i in range(num_samples_to_show):
                process_single_image_feature_model_eval(fire_images_for_display[i], DFIRE_TEST_LABELS_DIR_EVAL, loaded_artifacts_eval, DFIRE_CONFIG_EVAL, get_feature_params(), global_scaler_obj_eval)
        else:
            print("\nNo fire images found in the test set for single image processing.")

        num_samples_to_show = min(3, len(non_fire_images_for_display)) # Show up to 3 examples
        if num_samples_to_show > 0:
            print(f"\n--- Processing {num_samples_to_show} sample NON-FIRE images for display ---")
            for i in range(num_samples_to_show):
                process_single_image_feature_model_eval(non_fire_images_for_display[i], DFIRE_TEST_LABELS_DIR_EVAL, loaded_artifacts_eval, DFIRE_CONFIG_EVAL, get_feature_params(), global_scaler_obj_eval)
        else:
            print("\nNo non-fire images found in the test set for single image processing.")

    else:
        print("No feature-based models were loaded successfully. Skipping feature-based evaluation tasks.")
else:
    print("Cannot proceed with feature-based model evaluation due to missing global scaler.")

print("\n--- All D-Fire Feature-Based Model Evaluations Complete ---")