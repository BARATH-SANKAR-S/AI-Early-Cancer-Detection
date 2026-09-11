"""
Preprocessing Configuration for Module 8
Reproducible configuration for MRI data preprocessing
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class PreprocessingConfig:
    """Configuration for MRI dataset preprocessing"""
    
    # Dataset paths
    PROJECT_ROOT = Path('.')
    DATASET_ROOT = PROJECT_ROOT
    TRAINING_DIR = DATASET_ROOT / 'Training'
    TESTING_DIR = DATASET_ROOT / 'Testing'
    
    # Output paths
    PROCESSED_DIR = DATASET_ROOT / 'processed'
    REPORTS_DIR = DATASET_ROOT / 'reports'
    
    # Image preprocessing parameters
    IMAGE_SIZE = (224, 224)  # Standard size for Vision Transformers
    NUM_CHANNELS = 3  # RGB
    BATCH_SIZE = 32
    
    # Dataset split parameters
    TRAIN_VAL_RATIO = 0.8  # 80% training, 20% validation
    RANDOM_SEED = 42
    
    # Class names (discovered from dataset analysis)
    CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
    NUM_CLASSES = len(CLASS_NAMES)
    
    # Class to index mapping
    CLASS_TO_INDEX = {class_name: idx for idx, class_name in enumerate(CLASS_NAMES)}
    INDEX_TO_CLASS = {idx: class_name for class_name, idx in CLASS_TO_INDEX.items()}
    
    # Augmentation parameters (only for training)
    AUGMENTATION_ENABLED = True
    ROTATION_RANGE = 15
    HORIZONTAL_FLIP = True
    VERTICAL_FLIP = False
    BRIGHTNESS_RANGE = (0.8, 1.2)
    CONTRAST_RANGE = (0.8, 1.2)
    
    # Hugging Face model parameters
    MODEL_NAME = "google/vit-base-patch16-224"  # Vision Transformer
    
    # Data quality parameters
    SKIP_CORRUPTED = True
    SKIP_DUPLICATES = True
    
    # File extensions
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']
    
    @classmethod
    def to_dict(cls) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'image_size': cls.IMAGE_SIZE,
            'batch_size': cls.BATCH_SIZE,
            'random_seed': cls.RANDOM_SEED,
            'train_val_ratio': cls.TRAIN_VAL_RATIO,
            'num_classes': cls.NUM_CLASSES,
            'class_names': cls.CLASS_NAMES,
            'class_to_index': cls.CLASS_TO_INDEX,
            'index_to_class': cls.INDEX_TO_CLASS,
            'augmentation_enabled': cls.AUGMENTATION_ENABLED,
            'rotation_range': cls.ROTATION_RANGE,
            'horizontal_flip': cls.HORIZONTAL_FLIP,
            'vertical_flip': cls.VERTICAL_FLIP,
            'brightness_range': cls.BRIGHTNESS_RANGE,
            'contrast_range': cls.CONTRAST_RANGE,
            'model_name': cls.MODEL_NAME,
            'skip_corrupted': cls.SKIP_CORRUPTED,
            'skip_duplicates': cls.SKIP_DUPLICATES,
            'supported_formats': cls.SUPPORTED_FORMATS
        }
    
    @classmethod
    def save_config(cls, output_path: str = 'reports/preprocessing_config.json'):
        """Save configuration to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(cls.to_dict(), f, indent=2)
        
        print(f"Configuration saved to {output_file}")
    
    @classmethod
    def load_config(cls, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)


if __name__ == '__main__':
    # Save configuration
    PreprocessingConfig.save_config()
    print("Configuration:")
    print(json.dumps(PreprocessingConfig.to_dict(), indent=2))
