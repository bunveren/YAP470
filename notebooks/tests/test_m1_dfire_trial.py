import os
import sys
import time
import joblib
import numpy as np
import cv2
from tqdm import tqdm
import warnings
import inspect # For robust path detection

warnings.filterwarnings('ignore') # Suppress warnings for cleaner output

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU, Input, ReLU
from scikeras.wrappers import KerasClassifier # For wrapping Keras models in scikit-learn API

from skimage.feature import local_binary_pattern, hog
# Note: For testing saved models, train_test_split and other sklearn utilities
# are used within the helper functions for data preparation/evaluation reproduction.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC # Included for type checking in evaluation, though not directly used for loading
import lightgbm as lgb # Included for type checking in evaluation
from sklearn.feature_selection import SelectKBest, RFE # Included for transformer loading
from sklearn.linear_model import LogisticRegression # Included for RFE
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA # Included for transformer loading

import matplotlib.pyplot as plt # For plotting single images


print("Imports and helper functions loaded.")

# --- Robust Path Definitions ---
# Try to get the actual path of the current notebook/script
try:
    # This is more reliable in many Jupyter/IPython environments
    if '__file__' in locals(): # Check if __file__ is defined (for regular scripts)
        _current_file_path = os.path.abspath(__file__)
    else:
        # For Jupyter notebooks, inspect can often get the path of the current file
        _current_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
except Exception:
    # Fallback for environments where the above don't work (e.g., bare IPython shell)
    print("Warning: Could not reliably determine script path. Falling back to os.getcwd(). Ensure this script is run from a predictable location relative to YAP470.")
    _current_file_path = os.path.abspath(os.getcwd())

# Assuming your script is in YAP470/notebooks/tests/
NOTEBOOK_DIR = os.path.dirname(_current_file_path)
PROJECT_ROOT = os.path.abspath(os.path.join(NOTEBOOK_DIR, '..', '..')) # Go up two levels

MODEL_SAVE_DIR_EVAL = os.path.join(PROJECT_ROOT, 'models')
DFIRE_ROOT_EVAL = os.path.join(PROJECT_ROOT, 'data_subsets', 'D-Fire')

DFIRE_TRAIN_ROOT_EVAL = os.path.join(DFIRE_ROOT_EVAL, 'train')
DFIRE_TEST_ROOT_EVAL = os.path.join(DFIRE_ROOT_EVAL, 'test')
DFIRE_TEST_IMAGES_DIR_EVAL = os.path.join(DFIRE_TEST_ROOT_EVAL, 'images')
DFIRE_TEST_LABELS_DIR_EVAL = os.path.join(DFIRE_TEST_ROOT_EVAL, 'labels')

# --- DFIRE_CONFIG_EVAL (aligned with original training setup) ---
DFIRE_CONFIG_EVAL = {
    'fire_class_ids': [0, 1],
    'target_img_size': (128, 128),
    'test_size': 0.2, # Used for reproducing original train/test split
    'random_state': 42,
    'img_extensions': ('.png', '.jpg', '.jpeg', '.bmp', '.gif'),
    'annotation_extension': '.txt',
    'model_dir': MODEL_SAVE_DIR_EVAL, # This points to the main models folder
    'color_spaces_to_load': ['bgr', 'hsv', 'ycbcr'], # Important for feature extraction consistency
    'normalize_pixels': 1 # Important for feature extraction consistency
}

# --- Helper Functions (Updated for image reading and general use) ---

def is_dfire_image_fire(annotation_path, fire_class_ids):
    """
    Checks if an annotation file indicates the presence of fire.
    Handles potential encoding issues by specifying utf-8 and errors='ignore'.
    """
    if not os.path.exists(annotation_path): return False
    try:
        with open(annotation_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if parts and len(parts) > 0:
                    if parts[0].isdigit():
                        class_id = int(parts[0])
                        if class_id in fire_class_ids: return True
    except Exception as e:
        # print(f"Warning: Error reading annotation {annotation_path}: {e}") # Uncomment for debug
        pass
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
        # print(f"Warning: LBP feature extraction failed: {e}") # Uncomment for debug
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
        # print(f"Warning: Image size {img_h}x{img_w} too small for HOG (min {min_img_h}x{min_img_w}).") # Uncomment for debug
        return np.array([])
    try:
        hog_features = hog(img_gray, orientations=orientations,
                           pixels_per_cell=pixels_per_cell,
                           cells_per_block=cells_per_block,
                           block_norm=block_norm,
                           visualize=False, feature_vector=True)
        return hog_features.flatten()
    except Exception as e:
        # print(f"Warning: HOG feature extraction failed: {e}") # Uncomment for debug
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
    """
    Loads images and extracts features in a memory-safe way, handling Unicode paths with cv2.imdecode.
    """
    dataset_choice = config.get('dataset_choice', 'dfire')
    data_root = config.get('data_root')
    target_size = config.get('target_img_size')
    color_spaces_to_load = config.get('color_spaces_to_load', ['bgr', 'hsv', 'ycbcr'])
    normalize_pixels = config.get('normalize_pixels', 1)
    fire_class_ids = config.get('fire_class_ids', [])

    if not data_root or not target_size: return np.array([]), np.array([])

    image_label_pairs = []
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
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
        # Use np.fromfile with cv2.imdecode to handle Unicode paths
        img_bgr = None
        try:
            img_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            # print(f"Error reading image {image_path}: {e}") # Uncomment for debug
            total_images_skipped_reading += 1
            continue

        if img_bgr is None:
            # print(f"Warning: cv2.imdecode returned None for {image_path}. Image might be corrupted.") # Uncomment for debug
            total_images_skipped_reading += 1
            continue

        try:
            img_resized = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_LINEAR)
            img_dict_single = {}
            # img_processed_bgr is generally for display/debugging; not strictly needed for feature extraction paths
            # but keeping for consistency with original script logic
            img_processed_bgr = None
            if normalize_pixels:
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
            # print(f"Error processing features for {image_path}: {e}") # Uncomment for debug
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

def get_feature_params():
    """Returns the feature extraction parameters."""
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
    # print("\nFeature extraction parameters:", feature_params) # Uncomment to see params on every call
    return feature_params

# Placeholder functions for compatibility, not directly used in *this* test script
# but needed if any other function calls them.
def get_config(dataset_choice):
    """Dummy get_config, not used directly in this evaluation script as config is DFIRE_CONFIG_EVAL."""
    raise NotImplementedError("get_config is not implemented in this evaluation script. Use DFIRE_CONFIG_EVAL directly.")

def split_data(*args, **kwargs):
    """Dummy split_data, data is split manually for evaluation reproduction."""
    raise NotImplementedError("split_data is not implemented in this evaluation script. Split manually for evaluation reproduction.")

def scale_features(X_train, X_test):
    """Dummy scale_features, global scaler is loaded directly."""
    raise NotImplementedError("scale_features is not implemented in this evaluation script. Global scaler is loaded directly.")

def perform_correlation_selection(*args, **kwargs):
    """Dummy perform_correlation_selection, selectors are loaded directly."""
    raise NotImplementedError("perform_correlation_selection is not implemented in this evaluation script. Selectors are loaded directly.")

def perform_rfe_selection(*args, **kwargs):
    """Dummy perform_rfe_selection, selectors are loaded directly."""
    raise NotImplementedError("perform_rfe_selection is not implemented in this evaluation script. Selectors are loaded directly.")

def tune_model_hyperparameters(*args, **kwargs):
    """Dummy tune_model_hyperparameters, models are loaded directly."""
    raise NotImplementedError("tune_model_hyperparameters is not implemented in this evaluation script. Models are loaded directly.")

def perform_pca_dimension_reduction(*args, **kwargs):
    """Dummy perform_pca_dimension_reduction, PCA transformers are loaded directly."""
    raise NotImplementedError("perform_pca_dimension_reduction is not implemented in this evaluation script. PCA transformers are loaded directly.")

def create_custom_mlp(hidden_layer_1_neurons=128, hidden_layer_2_neurons=64,
                        dropout_rate=0.3, activation='leaky_relu', learning_rate=0.001,
                        meta=None):
    """Keras MLP model definition."""
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

def evaluate_model(model, X_test, y_test, model_name="Model", feature_set_name="Unknown Feature Set"):
    """Evaluates a given model and prints metrics."""
    if model is None or X_test is None or y_test is None or X_test.shape[0] == 0:
        print(f"{model_name} evaluation skipped on {feature_set_name}: model not trained or test data is empty.")
        return {}

    # print(f"\nEvaluating {model_name} on the test set using {feature_set_name}...") # Uncomment for verbose
    start_time = time.time()
    if isinstance(model, KerasClassifier): # Scikeras wrapper around Keras model
        y_pred = model.predict(X_test)
    elif isinstance(model, tf.keras.Model): # Raw Keras model (if directly loaded without wrapper)
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int)
    else: # For scikit-learn models
        y_pred = model.predict(X_test)
    end_time = time.time()
    # print(f"Prediction duration: {end_time - start_time:.4f} seconds") # Uncomment for verbose

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall (Sensitivity): {recall:.4f}")
    print(f"    F1 Score: {f1:.4f}")
    print(f"    Confusion Matrix:\n{conf_matrix}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix.tolist()
    }

def extract_features_single_image(image_path_single, feature_params_single, config_single):
    """
    Extracts features from a single image using the defined feature parameters and config.
    Uses np.fromfile and cv2.imdecode for robust image reading.
    """
    img_bgr = None
    try:
        img_bgr = cv2.imdecode(np.fromfile(image_path_single, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error reading image for feature extraction: {image_path_single}: {e}")
        return np.array([])

    if img_bgr is None:
        return np.array([])

    img_resized = cv2.resize(img_bgr, config_single['target_img_size'], interpolation=cv2.INTER_LINEAR)
    img_dict_single = {}

    if config_single['normalize_pixels']:
        img_resized_uint8 = img_resized.astype(np.uint8)
        img_gray = cv2.cvtColor(img_resized_uint8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        img_dict_single['hsv'] = cv2.cvtColor(img_resized_uint8, cv2.COLOR_BGR2HSV).astype(np.float32) / np.array([180, 255, 255], dtype=np.float32)
        img_dict_single['ycbcr'] = cv2.cvtColor(img_resized_uint8, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
    else:
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        img_dict_single['hsv'] = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        img_dict_single['ycbcr'] = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YCrCb)

    img_dict_single['gray'] = img_gray
    return combine_features(img_dict_single, feature_params_single)

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
            if '.keras' in model_filename.lower(): # Check for .keras extension (case-insensitive)
                # When loading a KerasClassifier (Scikeras wrapper), load_model might return the wrapper
                # or just the raw Keras model. We need to handle both cases if you used KerasClassifier
                # during training. The custom_objects ensure create_custom_mlp is known.
                loaded_model_temp = tf.keras.models.load_model(
                    model_path,
                    custom_objects={'create_custom_mlp': create_custom_mlp,
                                    'KerasClassifier': KerasClassifier,
                                    'LeakyReLU': tf.keras.layers.LeakyReLU,
                                    'ReLU': tf.keras.layers.ReLU}
                )
                # If it's a KerasClassifier instance, its actual Keras model is in .model_
                loaded_model = loaded_model_temp.model_ if isinstance(loaded_model_temp, KerasClassifier) else loaded_model_temp
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
    # Use load_and_extract_features_memory_safe which has the unicode fix
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
                # print(f"    Applied model-specific transformer to recreated test data.") # Uncomment for verbose
            except Exception as e:
                print(f"    Error applying transformer for {model_name}: {e}. Using globally scaled data.")

        if model is not None:
            evaluate_model(model, X_test_for_model, y_test_reproduced, model_name, "Recreated Test Split")
        else:
            print(f"  Model {model_name} was not loaded successfully.")

def evaluate_feature_folder_eval(images_folder_path, labels_folder_path, artifacts_dict, config, feature_params, global_scaler):
    """
    Loads features from a dedicated test folder, applies transformations,
    and evaluates each loaded model.
    """
    print(f"\n--- Evaluating models on dedicated test folder: {os.path.basename(images_folder_path)} (Feature Models) ---")

    test_data_config = {'dataset_choice': 'dfire', 'data_root': os.path.dirname(images_folder_path), # Adjust data_root for load_and_extract_features_memory_safe
                        'target_img_size': config['target_img_size'],
                        'color_spaces_to_load': config['color_spaces_to_load'],
                        'normalize_pixels': config['normalize_pixels'],
                        'fire_class_ids': config['fire_class_ids']}

    print("  Loading and extracting features from dedicated test folder...")
    # Use load_and_extract_features_memory_safe which has the unicode fix
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
                # print(f"    Applied model-specific transformer to dedicated test data.") # Uncomment for verbose
            except Exception as e:
                print(f"    Error applying transformer for {model_name}: {e}. Using globally scaled data.")

        if model is not None:
            evaluate_model(model, X_test_for_model, y_test_folder, model_name, "Dedicated Test Folder")
        else:
            print(f"  Model {model_name} was not loaded successfully.")

def process_single_image_feature_model_eval(image_path, labels_root_dir, artifacts_dict, config, feature_params, global_scaler):
    """
    Processes a single image: extracts features, applies transformations,
    makes predictions with all loaded models, and displays the image.
    Uses np.fromfile and cv2.imdecode for robust image reading for display.
    """
    print(f"\n--- Processing single image: {os.path.basename(image_path)} (Feature Models) ---")

    # Use np.fromfile with cv2.imdecode for robust image reading for display
    img_display = None
    try:
        img_display = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error: Could not read image for display: {image_path}. Error: {e}")
        return

    if img_display is None:
        print(f"Error: cv2.imdecode returned None for {image_path}. Image might be corrupted or not a valid image format.")
        return

    img_name_without_ext = os.path.splitext(os.path.basename(image_path))[0]
    annotation_path = os.path.join(labels_root_dir, img_name_without_ext + config['annotation_extension'])
    true_label_num = 1 if is_dfire_image_fire(annotation_path, config['fire_class_ids']) else 0
    true_label_text = 'Fire' if true_label_num == 1 else 'Non-Fire'

    raw_features_single = extract_features_single_image(image_path, feature_params, config)
    if raw_features_single.size == 0:
        print(f"  Failed to extract features for {os.path.basename(image_path)}. Skipping further predictions.")
        # Still show the image if it loaded, even if features couldn't be extracted
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
        title_text = f"Image: {os.path.basename(image_path)}\nTrue Label: {true_label_text}\n(Feature Extraction Failed)"
        plt.title(title_text)
        plt.axis('off')
        plt.show()
        return

    raw_features_single = raw_features_single.reshape(1, -1)
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
                if isinstance(model, KerasClassifier):
                    prediction_proba = model.predict_proba(features_for_prediction)[:, 1] # Get probability for positive class
                    prediction = (prediction_proba > 0.5).astype(int)[0]
                elif isinstance(model, tf.keras.Model):
                    prediction_proba = model.predict(features_for_prediction, verbose=0)[0][0] # Raw Keras model
                    prediction = (prediction_proba > 0.5).astype(int)
                else: # For scikit-learn models
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
print(f"Project Root: {PROJECT_ROOT}")
print(f"DFIRE_ROOT set to: {DFIRE_ROOT_EVAL}")
print(f"DFIRE_TRAIN_ROOT set to: {DFIRE_TRAIN_ROOT_EVAL}")
print(f"DFIRE_TEST_ROOT set to: {DFIRE_TEST_ROOT_EVAL}")
print(f"MODEL_DIR set to: {MODEL_SAVE_DIR_EVAL}")

# Define the list of models and their associated transformers to evaluate
models_to_load_and_evaluate_final = [
    {
        'display_name': 'Kaggle Custom MLP (Scaled RFE 75%)',
        'model_filename': 'Kaggle_custom_mlp_best_model_Scaled_RFE75%.keras',
        'transformer_filename': 'Kaggle_selector_Scaled_RFE75%.pkl'
    },
    {
        'display_name': 'Kaggle LightGBM (Scaled Corr 50%)',
        'model_filename': 'Kaggle_lightgbm_best_model_Scaled_Corr50%.pkl',
        'transformer_filename': 'Kaggle_selector_Scaled_Corr50%.pkl'
    },
    {
        'display_name': 'Kaggle SVM (Scaled PCA 1000)',
        'model_filename': 'Kaggle_svm_best_model_Scaled_PCA_1000.pkl',
        'transformer_filename': 'Kaggle_selector_Scaled_PCA_1000.pkl'
    }
]

# Load the main global scaler first, as it's needed for all feature processing
global_scaler_filename = 'dfirem1_global_scaler.pkl'
global_scaler_path = os.path.join(MODEL_SAVE_DIR_EVAL, global_scaler_filename) # Use os.path.join

global_scaler_obj_eval = None
try:
    global_scaler_obj_eval = joblib.load(global_scaler_path)
    print(f"Loaded global scaler: {global_scaler_path}")
except FileNotFoundError:
    print(f"Error: Global scaler not found at {global_scaler_path}. Feature-based evaluation cannot proceed.")
except Exception as e:
    print(f"Error loading global scaler: {e}. Feature-based evaluation cannot proceed.")


if global_scaler_obj_eval:
    feature_params_eval = get_feature_params() # Get feature params for evaluation
    loaded_artifacts_eval = load_artifacts_for_feature_models_eval(models_to_load_and_evaluate_final, MODEL_SAVE_DIR_EVAL)

    if loaded_artifacts_eval:
        print("\n--- Loaded feature-based models successfully. Proceeding with evaluations. ---")

        # 1. Reproduce results on a split from the D-Fire 'train' data
        # This will reload and extract features from the 'train' split and then split them again
        # to ensure consistency with how the models were originally validated.
        reproduce_original_test_split_features_eval(
            DFIRE_TRAIN_ROOT_EVAL,
            DFIRE_CONFIG_EVAL,
            feature_params_eval,
            global_scaler_obj_eval,
            loaded_artifacts_eval
        )

        # 2. Evaluate on the dedicated D-Fire 'test' folder
        evaluate_feature_folder_eval(
            DFIRE_TEST_IMAGES_DIR_EVAL,
            DFIRE_TEST_LABELS_DIR_EVAL,
            loaded_artifacts_eval,
            DFIRE_CONFIG_EVAL,
            feature_params_eval,
            global_scaler_obj_eval
        )

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
                process_single_image_feature_model_eval(
                    fire_images_for_display[i],
                    DFIRE_TEST_LABELS_DIR_EVAL,
                    loaded_artifacts_eval,
                    DFIRE_CONFIG_EVAL,
                    feature_params_eval,
                    global_scaler_obj_eval
                )
        else:
            print("\nNo fire images found in the test set for single image processing.")

        num_samples_to_show = min(3, len(non_fire_images_for_display)) # Show up to 3 examples
        if num_samples_to_show > 0:
            print(f"\n--- Processing {num_samples_to_show} sample NON-FIRE images for display ---")
            for i in range(num_samples_to_show):
                process_single_image_feature_model_eval(
                    non_fire_images_for_display[i],
                    DFIRE_TEST_LABELS_DIR_EVAL,
                    loaded_artifacts_eval,
                    DFIRE_CONFIG_EVAL,
                    feature_params_eval,
                    global_scaler_obj_eval
                )
        else:
            print("\nNo non-fire images found in the test set for single image processing.")

    else:
        print("No feature-based models were loaded successfully. Skipping feature-based evaluation tasks.")
else:
    print("Cannot proceed with feature-based model evaluation due to missing global scaler.")

print("\n--- All D-Fire Feature-Based Model Evaluations Complete ---")