import cv2
import os
import numpy as np
from tqdm import tqdm
from skimage.feature import local_binary_pattern, hog
from skimage.color import rgb2gray

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

               
if __name__ == "__main__":
    data_root = ("fire_dataset")
    target_img_size = (128, 128)

    if os.path.exists(data_root):
        processed_data_dict = load_prep_img(data_root, target_img_size, color_spaces=['bgr', 'hsv', 'ycbcr'], normalize=1)
    
    if processed_data_dict:
        features = []
        labels = processed_data_dict['labels'] 
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

        num_images = len(labels)
        if num_images > 0:
            for i in tqdm(range(num_images), desc=f"Extracting Features {data_root}"):
                img_dict_single = {}
                for cs in processed_data_dict.keys():
                    if cs != 'labels' and i < len(processed_data_dict[cs]): 
                        img_dict_single[cs] = processed_data_dict[cs][i]
                features_single = combine_features(img_dict_single, feature_params)
                if features_single.size > 0:
                    features.append(features_single)
                else: pass # TODO
            features_array = np.array(features)
            print(f"\n{data_root} feat arr: {features_array.shape}")
            print(f"{data_root} labels: {labels.shape}")

    data_root = ("Fire-Detection-Dataset")
    if os.path.exists(data_root):
        processed_data_dict = load_prep_img(
            data_root, 
            target_img_size, 
            color_spaces=['bgr', 'hsv', 'ycbcr'], 
            normalize=1)

"""
    dfire_root = "D-Fire" 
    target_img_size = (128, 128)
    fire_classes_ids = [1, 2]

    train_data_root = os.path.join(dfire_root, "train")
    test_data_root = os.path.join(dfire_root, "test")

    if os.path.exists(train_data_root):
        processed_data_dict = load_prep_dfire(
            train_data_root,
            target_img_size,
            fire_classes_ids, 
            color_spaces=['bgr', 'hsv', 'ycbcr'],
            normalize=1 
        )
"""