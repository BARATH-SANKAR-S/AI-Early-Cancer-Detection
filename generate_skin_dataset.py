"""
Fast Vectorized Synthetic Skin Cancer Dataset Generator
Supports modalities requested by user:
  - Dermoscopy / Photographic skin surface
  - MRI (soft tissue / subcutaneous dermal imaging)
  - CT scan (cutaneous / subcutaneous tissue density)
  - X-ray (superficial soft tissue radiograph)

Generates 1000 samples for each modality type (1000 MRI, 1000 CT, 1000 X-ray, 1000 Dermoscopy)
Organized into both:
  1) datasets/skin_cancer/train/ & val/ (by ISIC classes: melanoma, nevus, bcc, bkl, akiec, df, vasc)
  2) datasets/skin_cancer/by_modality/ (mri, ct, xray, dermoscopy: 1000 samples each)
Fast generation using vectorized NumPy operations.
"""

import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

random.seed(99)
np.random.seed(99)

OUTPUT_DIR = "datasets/skin_cancer"
IMG_SIZE = 224
CLASSES = ['melanoma', 'nevus', 'bcc', 'bkl', 'akiec', 'df', 'vasc']
MODALITIES = ['dermoscopy', 'mri', 'ct', 'xray']
SAMPLES_PER_MODALITY = 1000  # 1000 samples for each modality type

# Precomputed coordinate grid
Y, X = np.ogrid[:IMG_SIZE, :IMG_SIZE]
CX, CY = IMG_SIZE / 2.0, IMG_SIZE / 2.0
DIST_FROM_CENTER = np.sqrt((X - CX)**2 + (Y - CY)**2)
ANGLE_FROM_CENTER = np.arctan2(Y - CY, X - CX)


def fast_dermoscopy_bg():
    """Vectorized skin surface background with natural tone and texture."""
    base_r = random.uniform(210, 240)
    base_g = random.uniform(165, 195)
    base_b = random.uniform(135, 165)

    arr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    arr[:, :, 0] = base_r + np.random.normal(0, 4, (IMG_SIZE, IMG_SIZE))
    arr[:, :, 1] = base_g + np.random.normal(0, 4, (IMG_SIZE, IMG_SIZE))
    arr[:, :, 2] = base_b + np.random.normal(0, 4, (IMG_SIZE, IMG_SIZE))

    # Add subtle skin lighting gradient
    grad = (1.0 - 0.15 * (DIST_FROM_CENTER / (IMG_SIZE * 0.7)))
    arr *= grad[:, :, None]
    return np.clip(arr, 0, 255)


def fast_mri_bg():
    """Vectorized MRI soft-tissue dermal scan background."""
    arr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    r_outer = IMG_SIZE * 0.44
    rel = DIST_FROM_CENTER / r_outer

    # Epidermal / dermal outer layer (bright rim), subcutaneous fat (mid)
    gray = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    gray[rel <= 0.8] = 75.0 + 15.0 * np.sin(4 * ANGLE_FROM_CENTER)[rel <= 0.8]
    gray[(rel > 0.8) & (rel <= 1.0)] = 160.0
    gray += np.random.normal(0, 6, (IMG_SIZE, IMG_SIZE))

    arr[:, :, 0] = gray
    arr[:, :, 1] = gray
    arr[:, :, 2] = gray
    return np.clip(arr, 0, 255)


def fast_ct_bg():
    """Vectorized CT dermal cross-section background."""
    arr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    r_outer = IMG_SIZE * 0.44
    rel = DIST_FROM_CENTER / r_outer

    gray = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    gray[rel <= 0.85] = 90.0 + 8.0 * np.cos(2 * ANGLE_FROM_CENTER)[rel <= 0.85]
    gray[(rel > 0.85) & (rel <= 1.0)] = 185.0
    gray += np.random.normal(0, 5, (IMG_SIZE, IMG_SIZE))

    arr[:, :, 0] = gray
    arr[:, :, 1] = gray
    arr[:, :, 2] = gray
    return np.clip(arr, 0, 255)


def fast_xray_bg():
    """Vectorized soft tissue radiograph background."""
    arr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    gray = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32) * 175.0
    # Soft tissue contour
    mask = DIST_FROM_CENTER < (IMG_SIZE * 0.42)
    gray[mask] = 110.0 + np.random.normal(0, 5, (IMG_SIZE, IMG_SIZE))[mask]
    gray += np.random.normal(0, 4, (IMG_SIZE, IMG_SIZE))

    arr[:, :, 0] = gray
    arr[:, :, 1] = gray
    arr[:, :, 2] = gray
    return np.clip(arr, 0, 255)


def apply_lesion_by_class(bg, cls, modality):
    """Render distinctive lesion characteristics for the given class & modality."""
    ox = CX + random.uniform(-10, 10)
    oy = CY + random.uniform(-10, 10)
    d = np.sqrt((X - ox)**2 + (Y - oy)**2)
    theta = np.arctan2(Y - oy, X - ox)

    result = bg.copy()

    if cls == 'melanoma':
        # Asymmetric, jagged boundary, dark irregular pigmentation
        r_base = random.uniform(35, 55)
        r_jagged = r_base * (1.0 + 0.3 * np.sin(5 * theta) + 0.2 * np.cos(9 * theta))
        mask = d <= r_jagged
        alpha = np.maximum(0, 1.0 - (d / np.maximum(r_jagged, 1e-5))**0.6)[:, :, None]
        # Dark black-brown color
        lesion_color = np.array([35.0, 25.0, 20.0])
        result = np.where(mask[:, :, None], alpha * lesion_color + (1 - alpha) * result, result)

    elif cls == 'nevus':
        # Symmetric, uniform brown, smooth border
        r_base = random.uniform(25, 45)
        mask = d <= r_base
        alpha = np.maximum(0, 1.0 - (d / r_base))[:, :, None]
        lesion_color = np.array([95.0, 55.0, 35.0])
        result = np.where(mask[:, :, None], alpha * lesion_color + (1 - alpha) * result, result)

    elif cls == 'bcc':
        # Basal cell carcinoma: pearly, translucent pink/red with central ulceration
        r_base = random.uniform(28, 48)
        mask = d <= r_base
        alpha = np.maximum(0, 1.0 - (d / r_base)**1.2)[:, :, None]
        lesion_color = np.array([175.0, 85.0, 95.0])
        result = np.where(mask[:, :, None], alpha * lesion_color + (1 - alpha) * result, result)

    elif cls == 'bkl':
        # Benign keratosis: warty, yellowish-brown stuck-on plaque
        r_base = random.uniform(30, 50)
        r_lobed = r_base * (1.0 + 0.15 * np.cos(4 * theta))
        mask = d <= r_lobed
        alpha = np.maximum(0, 1.0 - (d / np.maximum(r_lobed, 1e-5)))[:, :, None]
        lesion_color = np.array([135.0, 105.0, 65.0])
        result = np.where(mask[:, :, None], alpha * lesion_color + (1 - alpha) * result, result)

    elif cls == 'akiec':
        # Actinic keratosis: scaly, erythematous rough red patch
        r_base = random.uniform(25, 42)
        mask = d <= r_base
        alpha = np.maximum(0, 1.0 - (d / r_base))[:, :, None]
        lesion_color = np.array([185.0, 75.0, 75.0])
        result = np.where(mask[:, :, None], alpha * lesion_color + (1 - alpha) * result, result)

    elif cls == 'df':
        # Dermatofibroma: small firm brownish button-like lesion
        r_base = random.uniform(18, 30)
        mask = d <= r_base
        alpha = np.maximum(0, 1.0 - (d / r_base))[:, :, None]
        lesion_color = np.array([115.0, 70.0, 55.0])
        result = np.where(mask[:, :, None], alpha * lesion_color + (1 - alpha) * result, result)

    elif cls == 'vasc':
        # Vascular lesion: vivid red-purple cherry angioma
        r_base = random.uniform(20, 36)
        mask = d <= r_base
        alpha = np.maximum(0, 1.0 - (d / r_base))[:, :, None]
        lesion_color = np.array([195.0, 30.0, 45.0])
        result = np.where(mask[:, :, None], alpha * lesion_color + (1 - alpha) * result, result)

    # In MRI/CT/X-ray modalities, convert colored lesion to proper density contrast
    if modality in ('mri', 'ct', 'xray'):
        gray = 0.299 * result[:, :, 0] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 2]
        result[:, :, 0] = gray
        result[:, :, 1] = gray
        result[:, :, 2] = gray

    return np.clip(result, 0, 255).astype(np.uint8)


BG_MAP = {
    'dermoscopy': fast_dermoscopy_bg,
    'mri': fast_mri_bg,
    'ct': fast_ct_bg,
    'xray': fast_xray_bg,
}


def generate_single_skin_image(modality, cls):
    """Generate and return PIL Image for skin scan/dermoscopy."""
    bg = BG_MAP[modality]()
    arr = apply_lesion_by_class(bg, cls, modality)
    img = Image.fromarray(arr, mode='RGB')
    return img


def generate_skin_dataset():
    """Build datasets/skin_cancer/ with 1000 samples per modality (MRI, CT, X-ray, Dermoscopy)."""
    print("=" * 65)
    print("Fast Vectorized Skin Cancer Dataset Generator")
    print("Modalities: MRI (1000), CT (1000), X-Ray (1000), Dermoscopy (1000)")
    print("Classes (ISIC): melanoma, nevus, bcc, bkl, akiec, df, vasc")
    print("=" * 65)

    # Setup directories
    for split in ['train', 'val']:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

    for mod in MODALITIES:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, "by_modality", mod, cls), exist_ok=True)

    total_generated = 0

    for mod in MODALITIES:
        print(f"\n--> Generating modality: {mod.upper()} (1000 images total)")
        # Distribute 1000 samples across the 7 classes (~143 per class)
        per_class_count = SAMPLES_PER_MODALITY // len(CLASSES)

        for c_idx, cls in enumerate(CLASSES):
            # Give remainder to first few classes
            n_samples = per_class_count + (1 if c_idx < (SAMPLES_PER_MODALITY % len(CLASSES)) else 0)
            n_train = int(n_samples * 0.8)
            n_val = n_samples - n_train

            print(f"    Class: {cls.ljust(10)} [{n_train} train, {n_val} val]...", end=" ", flush=True)

            for i in range(n_train):
                img = generate_single_skin_image(mod, cls)
                fname = f"{mod}_{cls}_train_{i:04d}.jpg"
                img.save(os.path.join(OUTPUT_DIR, 'train', cls, fname), 'JPEG', quality=90)
                img.save(os.path.join(OUTPUT_DIR, 'by_modality', mod, cls, fname), 'JPEG', quality=85)
                total_generated += 1

            for i in range(n_val):
                img = generate_single_skin_image(mod, cls)
                fname = f"{mod}_{cls}_val_{i:04d}.jpg"
                img.save(os.path.join(OUTPUT_DIR, 'val', cls, fname), 'JPEG', quality=90)
                total_generated += 1

            print("Done")

    print("\n" + "=" * 65)
    print(f"SUCCESS: Generated {total_generated} Skin Cancer images total!")
    print(f"Modality breakdown:")
    for mod in MODALITIES:
        print(f"  - {mod.upper()}: 1000 images")
    print(f"Directory: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == '__main__':
    generate_skin_dataset()
