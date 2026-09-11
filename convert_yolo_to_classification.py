"""
Convert YOLO format CT brain tumor dataset to classification format
Organizes images into class folders: normal/ and tumor/
"""
import os
import shutil
from pathlib import Path
from collections import Counter

def convert_yolo_to_classification():
    """Convert YOLO format dataset to classification format"""
    
    print("="*60)
    print("Converting YOLO Dataset to Classification Format")
    print("="*60)
    
    # Paths
    base_dir = Path("datasets/ct_scan/extracted")
    train_images_dir = base_dir / "images/train"
    val_images_dir = base_dir / "images/val"
    train_labels_dir = base_dir / "labels/train"
    val_labels_dir = base_dir / "labels/val"
    
    # Output directories
    output_train_normal = base_dir / "classification/train/normal"
    output_train_tumor = base_dir / "classification/train/tumor"
    output_val_normal = base_dir / "classification/val/normal"
    output_val_tumor = base_dir / "classification/val/tumor"
    
    # Create output directories
    for dir_path in [output_train_normal, output_train_tumor, output_val_normal, output_val_tumor]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Counters
    train_normal = 0
    train_tumor = 0
    val_normal = 0
    val_tumor = 0
    
    # Process training set
    print("\nProcessing training set...")
    for img_file in train_images_dir.glob("*.jpg"):
        label_file = train_labels_dir / (img_file.stem + ".txt")
        
        if label_file.exists():
            # Read label file to get class
            with open(label_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    first_class = int(lines[0].split()[0])
                    if first_class == 0:  # negative/normal
                        shutil.copy(img_file, output_train_normal / img_file.name)
                        train_normal += 1
                    elif first_class == 1:  # positive/tumor
                        shutil.copy(img_file, output_train_tumor / img_file.name)
                        train_tumor += 1
    
    # Process validation set
    print("Processing validation set...")
    for img_file in val_images_dir.glob("*.jpg"):
        label_file = val_labels_dir / (img_file.stem + ".txt")
        
        if label_file.exists():
            # Read label file to get class
            with open(label_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    first_class = int(lines[0].split()[0])
                    if first_class == 0:  # negative/normal
                        shutil.copy(img_file, output_val_normal / img_file.name)
                        val_normal += 1
                    elif first_class == 1:  # positive/tumor
                        shutil.copy(img_file, output_val_tumor / img_file.name)
                        val_tumor += 1
    
    # Print statistics
    print("\n" + "="*60)
    print("Conversion Complete!")
    print("="*60)
    print(f"Training - Normal: {train_normal}, Tumor: {train_tumor}")
    print(f"Validation - Normal: {val_normal}, Tumor: {val_tumor}")
    print(f"Total - Normal: {train_normal + val_normal}, Tumor: {train_tumor + val_tumor}")
    print(f"Total Images: {train_normal + train_tumor + val_normal + val_tumor}")
    
    # Calculate balance
    total_train = train_normal + train_tumor
    if total_train > 0:
        tumor_pct = (train_tumor / total_train) * 100
        print(f"Training Balance: {tumor_pct:.1f}% tumor, {100-tumor_pct:.1f}% normal")
    
    print("\nOutput directory: datasets/ct_scan/extracted/classification/")
    print("="*60)

if __name__ == "__main__":
    convert_yolo_to_classification()
