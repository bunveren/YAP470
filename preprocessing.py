import cv2
import os
import numpy as np
from tqdm import tqdm
from skimage.feature import local_binary_pattern, hog
from skimage.color import rgb2gray

def load_prep_img(data_root, target_size, 
    color_spaces=['bgr', 'hsv', 'ycbcr']):
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
        
    numpy_data = {}
    for cs, data_list in processed_data.items():
        if data_list:
             numpy_data[cs] = np.array(data_list)
        else:
             if cs == 'gray':
                  numpy_data[cs] = np.array([], dtype=np.uint8).reshape(0, target_size[1], target_size[0])
             else:
                  numpy_data[cs] = np.array([], dtype=np.uint8).reshape(0, target_size[1], target_size[0], 3)

    numpy_data['labels'] = np.array(labels)
    return numpy_data    
        
        
        
               
if __name__ == "__main__":
    data_root = ("fire_dataset")
    target_img_size = (128, 128)

    if os.path.exists(data_root):
        processed_data_dict = load_prep_img(data_root, target_img_size, color_spaces=['bgr', 'hsv', 'ycbcr'])
