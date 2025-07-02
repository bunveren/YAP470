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
    print(f"yangin Var (1): {images_with_fire}, yok (0): {images_without_fire}")

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
               
if __name__ == "__main__":
    data_root = ("fire_dataset")
    target_img_size = (128, 128)

    if os.path.exists(data_root):
        processed_data_dict = load_prep_img(data_root, target_img_size, color_spaces=['bgr', 'hsv', 'ycbcr'], normalize=1)
    
    data_root = ("Fire-Detection-Dataset")
    if os.path.exists(data_root):
        processed_data_dict = load_prep_img(
            data_root, 
            target_img_size, 
            color_spaces=['bgr', 'hsv', 'ycbcr'], 
            normalize=1)

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