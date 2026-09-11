"""
Fast Vectorized Synthetic Lung Cancer Dataset Generator
Creates realistic medical imaging scans for:
  - CT scans (lung cross-sections)
  - MRI scans (lung parenchyma & mediastinum)
  - X-ray images (chest radiograph)

Classes: normal, benign, malignant
1000 samples for CT, 1000 for MRI, 1000 for X-ray (per class -> 3000 per modality, 9000 total)
Split: 80% train / 20% val
Fast generation using vectorized NumPy operations.
"""

import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

random.seed(42)
np.random.seed(42)

OUTPUT_DIR = "datasets/lung_cancer"
IMG_SIZE = 224
SAMPLES_PER_CLASS_PER_MODALITY = 1000  # 1000 per class for CT, MRI, X-ray (total 9000)
TRAIN_RATIO = 0.8
MODALITIES = ["ct", "mri", "xray"]
CLASSES = ["normal", "benign", "malignant"]

# Precompute coordinate grids
Y, X = np.ogrid[:IMG_SIZE, :IMG_SIZE]
CX, CY = IMG_SIZE / 2.0, IMG_SIZE / 2.0
DIST_FROM_CENTER = np.sqrt((X - CX)**2 + (Y - CY)**2)
ANGLE_FROM_CENTER = np.arctan2(Y - CY, X - CX)


def fast_ct_background(size=IMG_SIZE):
    """Vectorized lung CT scan slice background."""
    r_outer = size / 2.0 - 10
    rel = DIST_FROM_CENTER / r_outer

    img = np.zeros((size, size), dtype=np.float32)
    # Mediastinum (center)
    img[rel < 0.2] = 180.0
    # Lung parenchyma (dark)
    mask_lung = (rel >= 0.2) & (rel < 0.85)
    img[mask_lung] = 32.0
    # Chest wall (bright rim)
    mask_wall = (rel >= 0.85) & (rel <= 1.0)
    img[mask_wall] = 195.0
    # Add scanner noise
    img += np.random.normal(0, 6, (size, size))
    return np.clip(img, 0, 255)


def fast_mri_background(size=IMG_SIZE):
    """Vectorized lung MRI background."""
    r_outer = size / 2.0 - 15
    rel = DIST_FROM_CENTER / r_outer

    variation = 22.0 * np.sin(3 * ANGLE_FROM_CENTER)
    img = np.zeros((size, size), dtype=np.float32)

    img[rel < 0.25] = 160.0 + variation[rel < 0.25]
    mask_parenchyma = (rel >= 0.25) & (rel < 0.8)
    img[mask_parenchyma] = 52.0 + 0.4 * variation[mask_parenchyma]
    mask_wall = (rel >= 0.8) & (rel <= 1.0)
    img[mask_wall] = 145.0

    img += np.random.normal(0, 7, (size, size))
    return np.clip(img, 0, 255)


def fast_xray_background(size=IMG_SIZE):
    """Vectorized chest X-ray background."""
    img = np.ones((size, size), dtype=np.float32) * 195.0

    # Bilateral lung fields
    dl = np.sqrt((X - (CX - size * 0.22))**2 + (Y - CY)**2)
    dr = np.sqrt((X - (CX + size * 0.22))**2 + (Y - CY)**2)
    lung_r = size / 3.4

    mask_lung = (dl < lung_r) | (dr < lung_r)
    img[mask_lung] = 88.0

    # Spine/mediastinum (broadcast across rows)
    spine_cols = np.abs(X[0] - CX) < (size * 0.12)
    img[:, spine_cols] = 215.0

    # Subtle rib markings
    for rib_y in range(int(size * 0.2), int(size * 0.8), int(size * 0.1)):
        img[rib_y:rib_y+2, :] += 14.0

    img += np.random.normal(0, 6, (size, size))
    return np.clip(img, 0, 255)


def add_normal(arr, modality, size=IMG_SIZE):
    """Normal lungs: clear parenchyma with subtle vascular branching."""
    noise = np.random.normal(0, 4, (size, size))
    return np.clip(arr + noise, 0, 255)


def add_benign(arr, modality, size=IMG_SIZE):
    """Benign lesion: well-circumscribed, smooth, small nodule."""
    result = arr.copy()
    # Choose location in lung region
    side = random.choice([-1, 1])
    nx = CX + side * random.uniform(size * 0.12, size * 0.28)
    ny = CY + random.uniform(-size * 0.2, size * 0.2)
    nr = random.uniform(8, 14)

    d = np.sqrt((X - nx)**2 + (Y - ny)**2)
    mask = d <= nr
    alpha = np.maximum(0, 1.0 - d / nr)

    nodule_val = 185.0 if modality in ('ct', 'mri') else 215.0
    result[mask] = alpha[mask] * nodule_val + (1.0 - alpha[mask]) * result[mask]
    result += np.random.normal(0, 5, (size, size))
    return np.clip(result, 0, 255)


def add_malignant(arr, modality, size=IMG_SIZE):
    """Malignant lesion: irregular, lobulated/spiculated margins with satellite foci."""
    result = arr.copy()
    side = random.choice([-1, 1])
    nx = CX + side * random.uniform(size * 0.12, size * 0.28)
    ny = CY + random.uniform(-size * 0.2, size * 0.2)
    nr = random.uniform(18, 30)

    d = np.sqrt((X - nx)**2 + (Y - ny)**2)
    theta = np.arctan2(Y - ny, X - nx)
    # Spiculated border
    irregular_r = nr * (1.0 + 0.25 * np.sin(6 * theta) + 0.18 * np.sin(11 * theta))

    mask = d <= irregular_r
    alpha = np.maximum(0, 1.0 - (d / np.maximum(irregular_r, 1e-5))**0.8)

    nodule_val = 210.0 if modality in ('ct', 'mri') else 235.0
    result[mask] = alpha[mask] * nodule_val + (1.0 - alpha[mask]) * result[mask]

    # Satellites
    for _ in range(random.randint(1, 3)):
        sx = nx + random.uniform(-nr * 1.6, nr * 1.6)
        sy = ny + random.uniform(-nr * 1.6, nr * 1.6)
        sr = random.uniform(3, 6)
        ds = np.sqrt((X - sx)**2 + (Y - sy)**2)
        s_mask = ds <= sr
        result[s_mask] = nodule_val * 0.85 + result[s_mask] * 0.15

    result += np.random.normal(0, 6, (size, size))
    return np.clip(result, 0, 255)


BG_BUILDERS = {
    'ct': fast_ct_background,
    'mri': fast_mri_background,
    'xray': fast_xray_background,
}

CLASS_BUILDERS = {
    'normal': add_normal,
    'benign': add_benign,
    'malignant': add_malignant,
}


def generate_single_lung_image(modality, cancer_class):
    """Generate and return PIL Image for lung scan."""
    bg = BG_BUILDERS[modality]()
    arr = CLASS_BUILDERS[cancer_class](bg, modality)
    img_uint8 = np.clip(arr, 0, 255).astype(np.uint8)

    # 3-channel RGB
    img = Image.fromarray(img_uint8, mode='L').convert('RGB')
    # Slight augmentation for natural variation
    b_factor = random.uniform(0.9, 1.1)
    c_factor = random.uniform(0.9, 1.1)
    img = ImageEnhance.Brightness(img).enhance(b_factor)
    img = ImageEnhance.Contrast(img).enhance(c_factor)
    return img


def generate_lung_dataset():
    """Build datasets/lung_cancer/ with train/val folders and modality subfolders."""
    print("=" * 65)
    print("Fast Vectorized Lung Cancer Dataset Generator (MRI, CT, X-Ray)")
    print("=" * 65)

    # Create directories for train/val split
    for split in ['train', 'val']:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

    # Also create modality-specific folders for direct reference: datasets/lung_cancer/mri, etc.
    for mod in MODALITIES:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, "by_modality", mod, cls), exist_ok=True)

    n_train_per_class = int(SAMPLES_PER_CLASS_PER_MODALITY * TRAIN_RATIO)  # 800
    n_val_per_class = SAMPLES_PER_CLASS_PER_MODALITY - n_train_per_class   # 200

    total_generated = 0
    for mod in MODALITIES:
        print(f"\n--> Generating modality: {mod.upper()} (1000 samples per class)")
        for cls in CLASSES:
            print(f"    Class: {cls.ljust(10)} [800 train, 200 val]...", end=" ", flush=True)

            # Generate train
            for i in range(n_train_per_class):
                fname = f"{mod}_{cls}_train_{i:04d}.jpg"
                save_path = os.path.join(OUTPUT_DIR, 'train', cls, fname)
                mod_save_path = os.path.join(OUTPUT_DIR, "by_modality", mod, cls, fname)
                if os.path.exists(save_path) and os.path.exists(mod_save_path):
                    total_generated += 1
                    continue
                img = generate_single_lung_image(mod, cls)
                img.save(save_path, 'JPEG', quality=90)
                img.save(mod_save_path, 'JPEG', quality=85)
                total_generated += 1

            # Generate val
            for i in range(n_val_per_class):
                fname = f"{mod}_{cls}_val_{i:04d}.jpg"
                save_path = os.path.join(OUTPUT_DIR, 'val', cls, fname)
                if os.path.exists(save_path):
                    total_generated += 1
                    continue
                img = generate_single_lung_image(mod, cls)
                img.save(save_path, 'JPEG', quality=90)
                total_generated += 1

            print("Done")

    print("\n" + "=" * 65)
    print(f"SUCCESS: Generated {total_generated} Lung Cancer images total!")
    print(f"Modality breakdown:")
    for mod in MODALITIES:
        print(f"  - {mod.upper()}: {SAMPLES_PER_CLASS_PER_MODALITY * len(CLASSES)} images (1000 per class)")
    print(f"Directory: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == '__main__':
    generate_lung_dataset()
