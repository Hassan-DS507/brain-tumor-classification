import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_images_to_numpy(root_path, target_size=None, max_images=None):
    """
    Reads images from class-subfolders and returns them with class names.

    Parameters:
    - root_path: str, e.g., 'dataset/train'
    - target_size: tuple, optional image size (e.g., (224, 224))
    - max_images: int, optional total limit

    Returns:
    - X: np.ndarray of images (n_samples, H, W, 3)
    - y: list of corresponding class names (strings)
    """

    X = []
    y = []
    total = 0
    class_folders = sorted([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])

    for class_name in class_folders:
        class_dir = os.path.join(root_path, class_name)
        for file in tqdm(os.listdir(class_dir), desc=f"Loading '{class_name}'"):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, file)
                try:
                    img = Image.open(img_path).convert("RGB")
                    if target_size:
                        img = img.resize(target_size)
                    img_array = np.array(img)
                    X.append(img_array)
                    y.append(class_name)  # أو ممكن تستخدم اسم الصورة نفسها: y.append(file)
                    total += 1
                    if max_images and total >= max_images:
                        break
                except Exception as e:
                    print(f"Skipped image: {img_path} due to error: {e}")
        if max_images and total >= max_images:
            break

    return np.array(X), y
