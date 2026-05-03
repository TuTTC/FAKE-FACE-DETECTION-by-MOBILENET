import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
from io import BytesIO

import torchvision.transforms.functional as TF
import torchvision.transforms as transforms


# ======================
# Config
# ======================
RESIZE_SIZE = (224, 224)
NORMALIZE = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)


# ======================
# Utils
# ======================
def to_uint8(tensor):
    img_np = (tensor.numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255
    return np.clip(img_np, 0, 255).astype(np.uint8)


def save_augmented(image_pil, base_name, aug_type, transform_func, save_dir):
    img = image_pil.resize(RESIZE_SIZE)
    img = transform_func(img)

    img_tensor = TF.to_tensor(img)
    img_tensor = NORMALIZE(img_tensor)
    img_np = to_uint8(img_tensor)

    save_path = os.path.join(save_dir, f"{base_name}_{aug_type}.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))


# ======================
# Augmentations
# ======================
def lowlight(img):
    return TF.adjust_brightness(img, 0.3)


def jpeg_compress(img, quality=25):
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return Image.open(buffer)


def add_gaussian_noise(img, std=15):
    img_np = np.array(img)
    noise = np.random.normal(0, std, img_np.shape).astype(np.int16)
    noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


# ======================
# Main processing
# ======================
def process_class(src_root, dst_root, class_name, limit=5000):
    src_folder = os.path.join(src_root, class_name)
    dst_folder = os.path.join(dst_root, class_name)
    os.makedirs(dst_folder, exist_ok=True)

    img_list = sorted(os.listdir(src_folder))[:limit]

    for i, fname in enumerate(tqdm(img_list, desc=f"Processing {class_name}")):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                path = os.path.join(src_folder, fname)
                image = Image.open(path).convert("RGB")
                base_name = f"{i:05d}"

                save_augmented(image, base_name, "original", lambda x: x, dst_folder)
                save_augmented(image, base_name, "lowlight", lowlight, dst_folder)
                save_augmented(image, base_name, "jpeg", jpeg_compress, dst_folder)
                save_augmented(image, base_name, "gaussnoise", add_gaussian_noise, dst_folder)

            except Exception as e:
                print(f"Error {fname}: {e}")


def run_augmentation(src_root, dst_root, classes=("real", "fake"), limit=5000):
    for cls in classes:
        process_class(src_root, dst_root, cls, limit)

    print("Zipping output...")
    shutil.make_archive(dst_root, 'zip', dst_root)
    print("Done!")


# ======================
# CLI entrypoint
# ======================
if __name__ == "__main__":
    SRC_ROOT = "/kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/test"
    DST_ROOT = "/kaggle/working/test_augmented_faces"

    run_augmentation(SRC_ROOT, DST_ROOT)