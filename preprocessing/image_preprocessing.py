"""
Image Preprocessing Script for Module 8
Preprocesses MRI images with augmentation and Hugging Face processor
"""

import random
import sys
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from typing import List, Tuple, Optional
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.config import PreprocessingConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocesses MRI images with augmentation and normalization"""
    
    def __init__(self, config: PreprocessingConfig = PreprocessingConfig):
        self.config = config
        self.image_size = config.IMAGE_SIZE
        self.num_channels = config.NUM_CHANNELS
        
        # Try to load Hugging Face image processor
        self.hf_processor = None
        try:
            from transformers import AutoImageProcessor
            self.hf_processor = AutoImageProcessor.from_pretrained(config.MODEL_NAME)
            logger.info(f"Loaded Hugging Face processor: {config.MODEL_NAME}")
        except ImportError:
            logger.warning("transformers not installed. Using basic preprocessing.")
        except Exception as e:
            logger.warning(f"Could not load Hugging Face processor: {e}")
    
    def resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize image to target size
        
        Args:
            image: PIL Image
            
        Returns:
            Resized PIL Image
        """
        return image.resize(self.image_size, Image.LANCZOS)
    
    def normalize_with_hf_processor(self, image: Image.Image) -> np.ndarray:
        """
        Normalize image using Hugging Face processor
        
        Args:
            image: PIL Image
            
        Returns:
            Normalized numpy array
        """
        if self.hf_processor is None:
            # Fallback to simple normalization
            return np.array(image) / 255.0
        
        # Use Hugging Face processor
        inputs = self.hf_processor(image, return_tensors="np")
        return inputs['pixel_values'][0]
    
    def normalize_basic(self, image: Image.Image) -> np.ndarray:
        """
        Basic normalization (min-max scaling)
        
        Args:
            image: PIL Image
            
        Returns:
            Normalized numpy array
        """
        img_array = np.array(image, dtype=np.float32)
        return img_array / 255.0
    
    def augment_image(self, image: Image.Image) -> Image.Image:
        """
        Apply random augmentation to image (only for training)
        
        Args:
            image: PIL Image
            
        Returns:
            Augmented PIL Image
        """
        if not self.config.AUGMENTATION_ENABLED:
            return image
        
        # Random rotation
        if self.config.ROTATION_RANGE > 0:
            angle = random.uniform(-self.config.ROTATION_RANGE, self.config.ROTATION_RANGE)
            image = image.rotate(angle, expand=False, fillcolor=0)
        
        # Random horizontal flip
        if self.config.HORIZONTAL_FLIP and random.random() > 0.5:
            image = ImageOps.mirror(image)
        
        # Random vertical flip
        if self.config.VERTICAL_FLIP and random.random() > 0.5:
            image = ImageOps.flip(image)
        
        # Random brightness
        if self.config.BRIGHTNESS_RANGE:
            factor = random.uniform(*self.config.BRIGHTNESS_RANGE)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(factor)
        
        # Random contrast
        if self.config.CONTRAST_RANGE:
            factor = random.uniform(*self.config.CONTRAST_RANGE)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(factor)
        
        return image
    
    def preprocess_image(self, image: Image.Image, augment: bool = False) -> np.ndarray:
        """
        Complete preprocessing pipeline for a single image
        
        Args:
            image: PIL Image
            augment: Whether to apply augmentation (only for training)
            
        Returns:
            Preprocessed numpy array
        """
        # Step 1: Resize
        image = self.resize_image(image)
        
        # Step 2: Augmentation (only for training)
        if augment:
            image = self.augment_image(image)
        
        # Step 3: Normalization
        if self.hf_processor is not None:
            processed = self.normalize_with_hf_processor(image)
        else:
            processed = self.normalize_basic(image)
        
        return processed
    
    def preprocess_batch(self, images: List[Image.Image], augment: bool = False) -> np.ndarray:
        """
        Preprocess a batch of images
        
        Args:
            images: List of PIL Images
            augment: Whether to apply augmentation
            
        Returns:
            Batch of preprocessed images as numpy array
        """
        processed_images = []
        
        for image in images:
            processed = self.preprocess_image(image, augment=augment)
            processed_images.append(processed)
        
        return np.array(processed_images)
    
    def preprocess_dataset(self, dataset: List[Tuple[Image.Image, int, str]], 
                          augment: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Preprocess entire dataset
        
        Args:
            dataset: List of (image, label, path) tuples
            augment: Whether to apply augmentation
            
        Returns:
            Tuple of (images, labels, paths)
        """
        images = []
        labels = []
        paths = []
        
        for image, label, path in dataset:
            processed = self.preprocess_image(image, augment=augment)
            images.append(processed)
            labels.append(label)
            paths.append(path)
        
        return np.array(images), np.array(labels), paths


def main():
    """Test the image preprocessor"""
    from preprocessing.dataset_loader import MRIDatasetLoader
    
    config = PreprocessingConfig()
    preprocessor = ImagePreprocessor(config)
    loader = MRIDatasetLoader(config)
    
    # Load a few test images
    test_images = loader.load_testing_data()[:5]
    
    if test_images:
        print(f"Testing preprocessing on {len(test_images)} images...")
        
        # Test without augmentation
        images, labels, paths = preprocessor.preprocess_dataset(test_images, augment=False)
        print(f"Preprocessed shape: {images.shape}")
        print(f"Labels: {labels}")
        
        # Test with augmentation
        images_aug, labels_aug, paths_aug = preprocessor.preprocess_dataset(test_images, augment=True)
        print(f"Augmented shape: {images_aug.shape}")
        print(f"Augmented labels: {labels_aug}")


if __name__ == '__main__':
    main()
