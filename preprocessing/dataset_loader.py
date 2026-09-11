"""
Dataset Loader for Module 8
Loads MRI images from the brain tumor dataset
"""

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.config import PreprocessingConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MRIDatasetLoader:
    """Loads MRI images from the brain tumor dataset"""
    
    def __init__(self, config: PreprocessingConfig = PreprocessingConfig):
        self.config = config
        self.class_names = config.CLASS_NAMES
        self.class_to_index = config.CLASS_TO_INDEX
        self.supported_formats = config.SUPPORTED_FORMATS
        
    def load_image(self, image_path: Path) -> Optional[Image.Image]:
        """
        Load a single image and check for corruption
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image or None if corrupted
        """
        try:
            # Verify image integrity
            with Image.open(image_path) as img:
                img.verify()
            
            # Re-open and load the image
            with Image.open(image_path) as img:
                img.load()
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                return img
                
        except Exception as e:
            logger.error(f"Corrupted image: {image_path} - {e}")
            return None
    
    def load_dataset(self, data_dir: Path) -> List[Tuple[Image.Image, int, str]]:
        """
        Load all images from a directory
        
        Args:
            data_dir: Directory containing class subdirectories
            
        Returns:
            List of (image, label, image_path) tuples
        """
        images = []
        skipped_count = 0
        
        if not data_dir.exists():
            logger.warning(f"Directory does not exist: {data_dir}")
            return images
        
        for class_name in self.class_names:
            class_dir = data_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory does not exist: {class_dir}")
                continue
            
            label = self.class_to_index[class_name]
            image_files = [f for f in class_dir.iterdir() 
                          if f.is_file() and f.suffix.lower() in self.supported_formats]
            
            logger.info(f"Loading {len(image_files)} images from {class_name}")
            
            for image_file in image_files:
                image = self.load_image(image_file)
                
                if image is not None:
                    images.append((image, label, str(image_file)))
                else:
                    skipped_count += 1
        
        logger.info(f"Loaded {len(images)} images, skipped {skipped_count} corrupted images")
        return images
    
    def load_training_data(self) -> List[Tuple[Image.Image, int, str]]:
        """Load training data"""
        return self.load_dataset(self.config.TRAINING_DIR)
    
    def load_testing_data(self) -> List[Tuple[Image.Image, int, str]]:
        """Load testing data"""
        return self.load_dataset(self.config.TESTING_DIR)
    
    def get_class_distribution(self, images: List[Tuple[Image.Image, int, str]]) -> Dict[str, int]:
        """
        Get class distribution from loaded images
        
        Args:
            images: List of (image, label, path) tuples
            
        Returns:
            Dictionary with class names and counts
        """
        distribution = {class_name: 0 for class_name in self.class_names}
        
        for _, label, _ in images:
            class_name = self.config.INDEX_TO_CLASS[label]
            distribution[class_name] += 1
        
        return distribution
    
    def get_image_info(self, images: List[Tuple[Image.Image, int, str]]) -> Dict:
        """
        Get information about loaded images
        
        Args:
            images: List of (image, label, path) tuples
            
        Returns:
            Dictionary with image statistics
        """
        if not images:
            return {}
        
        sizes = [img.size for img, _, _ in images]
        widths = [size[0] for size in sizes]
        heights = [size[1] for size in sizes]
        
        return {
            'total_images': len(images),
            'min_width': min(widths),
            'max_width': max(widths),
            'avg_width': np.mean(widths),
            'min_height': min(heights),
            'max_height': max(heights),
            'avg_height': np.mean(heights),
            'unique_sizes': len(set(sizes))
        }


def main():
    """Test the dataset loader"""
    loader = MRIDatasetLoader()
    
    # Load training data
    print("Loading training data...")
    training_images = loader.load_training_data()
    print(f"Training images: {len(training_images)}")
    
    # Load testing data
    print("\nLoading testing data...")
    testing_images = loader.load_testing_data()
    print(f"Testing images: {len(testing_images)}")
    
    # Get class distributions
    if training_images:
        print("\nTraining class distribution:")
        print(loader.get_class_distribution(training_images))
    
    if testing_images:
        print("\nTesting class distribution:")
        print(loader.get_class_distribution(testing_images))
    
    # Get image info
    if testing_images:
        print("\nTesting image info:")
        print(loader.get_image_info(testing_images))


if __name__ == '__main__':
    main()
