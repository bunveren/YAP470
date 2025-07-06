import os, shutil, random
from tqdm import tqdm

FULL_KAGGLE_ROOT = "fire_dataset"
FULL_DFIRE_ROOT = "D-Fire" 
SUBSET_REPO_ROOT = "data_subsets"

KAGGLE_SUBSET_SIZE_PER_CLASS = 30
DFIRE_SUBSET_SIZE_PER_CLASS = 30
DFIRE_FIRE_CLASS_IDS = [0, 1] 

def is_dfire_image_fire(annotation_path, fire_class_ids):
    if not os.path.exists(annotation_path):
        return False 
    try:
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts and len(parts) > 0:
                    class_id = int(parts[0])
                    if class_id in fire_class_ids:
                        return True
    except Exception as e:
        pass
    return False 

def create_kaggle_subset(full_root, subset_root, size_per_class):
    fire_src_dir = os.path.join(full_root, 'fire_images')
    non_fire_src_dir = os.path.join(full_root, 'non_fire_images')

    fire_dest_dir = os.path.join(subset_root, 'fire_dataset', 'fire_images')
    non_fire_dest_dir = os.path.join(subset_root, 'fire_dataset', 'non_fire_images')

    os.makedirs(fire_dest_dir, exist_ok=True)
    os.makedirs(non_fire_dest_dir, exist_ok=True)

    fire_files = [f for f in os.listdir(fire_src_dir) if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']]
    non_fire_files = [f for f in os.listdir(non_fire_src_dir) if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']]

    if len(fire_files) < size_per_class:
        selected_fire = fire_files
    else:
        selected_fire = random.sample(fire_files, size_per_class)

    if len(non_fire_files) < size_per_class:
        selected_non_fire = non_fire_files
    else:
        selected_non_fire = random.sample(non_fire_files, size_per_class)

    for f in tqdm(selected_fire):
        shutil.copy(os.path.join(fire_src_dir, f), os.path.join(fire_dest_dir, f))

    for f in tqdm(selected_non_fire):
        shutil.copy(os.path.join(non_fire_src_dir, f), os.path.join(non_fire_dest_dir, f))


def create_dfire_subset(full_root, subset_root, split_name, size_per_class, fire_class_ids):
    images_src_dir = os.path.join(full_root, split_name, 'images')
    labels_src_dir = os.path.join(full_root, split_name, 'labels')

    images_dest_dir = os.path.join(subset_root, 'D-Fire', split_name, 'images')
    labels_dest_dir = os.path.join(subset_root, 'D-Fire', split_name, 'labels')

    os.makedirs(images_dest_dir, exist_ok=True)
    os.makedirs(labels_dest_dir, exist_ok=True)

    all_image_files = [f for f in os.listdir(images_src_dir) if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']]
    fire_images = []
    non_fire_images = []

    for filename in tqdm(all_image_files):
        image_name_without_ext = os.path.splitext(filename)[0]
        annotation_path = os.path.join(labels_src_dir, image_name_without_ext + '.txt')

        if is_dfire_image_fire(annotation_path, fire_class_ids):
            fire_images.append(filename)
        else:
            non_fire_images.append(filename)

    if len(fire_images) < size_per_class:
        selected_fire = fire_images
    else:
        selected_fire = random.sample(fire_images, size_per_class)

    if len(non_fire_images) < size_per_class:
        selected_non_fire = non_fire_images
    else:
        selected_non_fire = random.sample(non_fire_images, size_per_class)

    selected_files = selected_fire + selected_non_fire
    for filename in tqdm(selected_files):
        shutil.copy(os.path.join(images_src_dir, filename), os.path.join(images_dest_dir, filename))
        image_name_without_ext = os.path.splitext(filename)[0]
        src_label_path = os.path.join(labels_src_dir, image_name_without_ext + '.txt')
        dest_label_path = os.path.join(labels_dest_dir, image_name_without_ext + '.txt')
        if os.path.exists(src_label_path):
             shutil.copy(src_label_path, dest_label_path)
    full_obj_names_path = os.path.join(full_root, 'obj.names')
    subset_obj_names_path = os.path.join(subset_root, 'D-Fire', 'obj.names')
    if os.path.exists(full_obj_names_path):
         shutil.copy(full_obj_names_path, subset_obj_names_path)

if __name__ == "__main__":
    random.seed(42) 
    if os.path.exists(FULL_KAGGLE_ROOT):
        create_kaggle_subset(FULL_KAGGLE_ROOT, SUBSET_REPO_ROOT, KAGGLE_SUBSET_SIZE_PER_CLASS)
    dfire_splits_to_subset = ['train', 'test']
    if os.path.exists(FULL_DFIRE_ROOT):
         for split in dfire_splits_to_subset:
              create_dfire_subset(FULL_DFIRE_ROOT, SUBSET_REPO_ROOT, split, DFIRE_SUBSET_SIZE_PER_CLASS, DFIRE_FIRE_CLASS_IDS)