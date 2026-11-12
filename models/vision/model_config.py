"""
Model Configuration for Module 9
Configuration for Vision Transformer model
"""

import json
import sys
from pathlib import Path
from typing import Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ViTModelConfig:
    """Configuration for Vision Transformer model"""
    
    # Hugging Face model selection
    # Selected: google/vit-base-patch16-224
    # Reason: Well-established, widely used, good balance of performance and efficiency
    # Suitable for medical imaging with 224x224 input size
    MODEL_NAME = "google/vit-base-patch16-224"
    
    # Dataset parameters (from preprocessing)
    NUM_CLASSES = 4
    CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
    CLASS_TO_INDEX = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
    INDEX_TO_CLASS = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}
    
    # Dataset paths (from preprocessing config)
    PROJECT_ROOT = Path('.')
    DATASET_ROOT = PROJECT_ROOT
    TRAINING_DIR = DATASET_ROOT / 'Training'
    TESTING_DIR = DATASET_ROOT / 'Testing'
    
    # Image parameters
    IMAGE_SIZE = (224, 224)
    NUM_CHANNELS = 3
    
    # Training parameters
    BATCH_SIZE = 16  # Smaller batch size for memory efficiency
    LEARNING_RATE = 3e-5  # Conservative learning rate for fine-tuning
    NUM_EPOCHS = 10
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 100
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # Device configuration
    DEVICE = 'cuda' if False else 'cpu'  # Will be set dynamically
    
    # Model saving
    MODEL_SAVE_DIR = Path('models/vision/checkpoints')
    MODEL_SAVE_PATH = MODEL_SAVE_DIR / 'vit_brain_tumor.pth'
    
    # Training settings
    GRADIENT_ACCUMULATION_STEPS = 1
    MAX_GRAD_NORM = 1.0
    LOGGING_STEPS = 10
    EVALUATION_STEPS = 100
    SAVE_STEPS = 500
    
    # Data augmentation settings
    AUGMENTATION_ENABLED = True
    ROTATION_RANGE = 15
    HORIZONTAL_FLIP = True
    VERTICAL_FLIP = False
    BRIGHTNESS_RANGE = (0.8, 1.2)
    CONTRAST_RANGE = (0.8, 1.2)
    
    # Data quality parameters
    SKIP_CORRUPTED = True
    SKIP_DUPLICATES = True
    
    # File extensions
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']
    
    @classmethod
    def to_dict(cls) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'model_name': cls.MODEL_NAME,
            'num_classes': cls.NUM_CLASSES,
            'class_names': cls.CLASS_NAMES,
            'class_to_index': cls.CLASS_TO_INDEX,
            'index_to_class': cls.INDEX_TO_CLASS,
            'image_size': cls.IMAGE_SIZE,
            'num_channels': cls.NUM_CHANNELS,
            'batch_size': cls.BATCH_SIZE,
            'learning_rate': cls.LEARNING_RATE,
            'num_epochs': cls.NUM_EPOCHS,
            'weight_decay': cls.WEIGHT_DECAY,
            'warmup_steps': cls.WARMUP_STEPS,
            'random_seed': cls.RANDOM_SEED,
            'device': cls.DEVICE,
            'gradient_accumulation_steps': cls.GRADIENT_ACCUMULATION_STEPS,
            'max_grad_norm': cls.MAX_GRAD_NORM,
            'augmentation_enabled': cls.AUGMENTATION_ENABLED
        }
    
    @classmethod
    def save_config(cls, output_path: str = 'models/vision/model_config.json'):
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
    ViTModelConfig.save_config()
    print("Model Configuration:")
    print(json.dumps(ViTModelConfig.to_dict(), indent=2))
