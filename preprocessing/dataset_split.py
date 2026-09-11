"""
Dataset Split Script for Module 8
Splits dataset into training, validation, and test sets
"""

import random
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.config import PreprocessingConfig
from preprocessing.dataset_loader import MRIDatasetLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetSplitter:
    """Splits dataset into train/val/test sets"""
    
    def __init__(self, config: PreprocessingConfig = PreprocessingConfig):
        self.config = config
        self.loader = MRIDatasetLoader(config)
        
    def split_by_class(self, images: List[Tuple], train_ratio: float = 0.8, 
                      val_ratio: float = 0.1, test_ratio: float = 0.1) -> Dict:
        """
        Split images by class to maintain class distribution
        
        Args:
            images: List of (image, label, path) tuples
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            
        Returns:
            Dictionary with train, val, test splits
        """
        # Group images by class
        class_images = defaultdict(list)
        for img, label, path in images:
            class_name = self.config.INDEX_TO_CLASS[label]
            class_images[class_name].append((img, label, path))
        
        # Split each class
        train_images = []
        val_images = []
        test_images = []
        
        random.seed(self.config.RANDOM_SEED)
        
        for class_name, class_imgs in class_images.items():
            random.shuffle(class_imgs)
            
            n_total = len(class_imgs)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            n_test = n_total - n_train - n_val
            
            train_images.extend(class_imgs[:n_train])
            val_images.extend(class_imgs[n_train:n_train + n_val])
            test_images.extend(class_imgs[n_train + n_val:])
            
            logger.info(f"{class_name}: {n_train} train, {n_val} val, {n_test} test")
        
        return {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }
    
    def split_dataset(self) -> Dict:
        """
        Split the dataset into train/val/test sets
        
        Returns:
            Dictionary with split information
        """
        logger.info("Starting dataset split...")
        
        # Load all available data
        # Since Training folder is empty, load from Testing folder
        all_images = self.loader.load_testing_data()
        
        if not all_images:
            logger.error("No images found to split!")
            return {}
        
        logger.info(f"Total images loaded: {len(all_images)}")
        
        # Split into train/val/test
        # Using 70% train, 15% val, 15% test since we need to create all splits from Testing folder
        splits = self.split_by_class(all_images, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        
        # Calculate statistics
        split_info = {
            'train': {
                'count': len(splits['train']),
                'distribution': self.loader.get_class_distribution(splits['train'])
            },
            'val': {
                'count': len(splits['val']),
                'distribution': self.loader.get_class_distribution(splits['val'])
            },
            'test': {
                'count': len(splits['test']),
                'distribution': self.loader.get_class_distribution(splits['test'])
            }
        }
        
        logger.info(f"Train: {split_info['train']['count']}")
        logger.info(f"Val: {split_info['val']['count']}")
        logger.info(f"Test: {split_info['test']['count']}")
        
        return {
            'splits': splits,
            'info': split_info
        }
    
    def save_split_info(self, split_info: Dict, output_path: str = 'reports/dataset_split_info.json'):
        """Save split information to JSON"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        serializable_info = {
            'train': {
                'count': split_info['train']['count'],
                'distribution': split_info['train']['distribution']
            },
            'val': {
                'count': split_info['val']['count'],
                'distribution': split_info['val']['distribution']
            },
            'test': {
                'count': split_info['test']['count'],
                'distribution': split_info['test']['distribution']
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_info, f, indent=2)
        
        logger.info(f"Split info saved to {output_file}")


def main():
    """Test the dataset splitter"""
    splitter = DatasetSplitter()
    
    # Split dataset
    result = splitter.split_dataset()
    
    if result:
        # Save split info
        splitter.save_split_info(result['info'])
        
        print("\nDataset Split Results:")
        print(f"Train: {result['info']['train']['count']} images")
        print(f"Validation: {result['info']['val']['count']} images")
        print(f"Test: {result['info']['test']['count']} images")
        
        print("\nClass Distribution:")
        for split_name in ['train', 'val', 'test']:
            print(f"\n{split_name.upper()}:")
            for class_name, count in result['info'][split_name]['distribution'].items():
                print(f"  {class_name}: {count}")


if __name__ == '__main__':
    main()
