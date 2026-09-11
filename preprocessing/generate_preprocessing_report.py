"""
Preprocessing Report Generator for Module 8
Generates comprehensive preprocessing report
"""

import json
import sys
from pathlib import Path
from typing import Dict
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.config import PreprocessingConfig
from preprocessing.dataset_loader import MRIDatasetLoader
from preprocessing.dataset_split import DatasetSplitter
from preprocessing.image_preprocessing import ImagePreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PreprocessingReportGenerator:
    """Generates preprocessing report"""
    
    def __init__(self, config: PreprocessingConfig = PreprocessingConfig):
        self.config = config
        self.loader = MRIDatasetLoader(config)
        self.splitter = DatasetSplitter(config)
        self.preprocessor = ImagePreprocessor(config)
        
    def generate_report(self) -> Dict:
        """Generate complete preprocessing report"""
        logger.info("Generating preprocessing report...")
        
        # Load and split dataset
        split_result = self.splitter.split_dataset()
        
        if not split_result:
            logger.error("Failed to split dataset")
            return {}
        
        splits = split_result['splits']
        split_info = split_result['info']
        
        # Preprocess each split
        logger.info("Preprocessing training data...")
        train_images, train_labels, train_paths = self.preprocessor.preprocess_dataset(
            splits['train'], augment=True
        )
        
        logger.info("Preprocessing validation data...")
        val_images, val_labels, val_paths = self.preprocessor.preprocess_dataset(
            splits['val'], augment=False
        )
        
        logger.info("Preprocessing test data...")
        test_images, test_labels, test_paths = self.preprocessor.preprocess_dataset(
            splits['test'], augment=False
        )
        
        # Generate report
        report = {
            'configuration': self.config.to_dict(),
            'dataset_statistics': {
                'training_image_count': len(train_images),
                'validation_image_count': len(val_images),
                'test_image_count': len(test_images),
                'total_images': len(train_images) + len(val_images) + len(test_images),
                'image_size': self.config.IMAGE_SIZE,
                'num_classes': self.config.NUM_CLASSES,
                'class_names': self.config.CLASS_NAMES,
                'class_to_index': self.config.CLASS_TO_INDEX,
                'index_to_class': self.config.INDEX_TO_CLASS
            },
            'class_distribution': {
                'training': split_info['train']['distribution'],
                'validation': split_info['val']['distribution'],
                'test': split_info['test']['distribution']
            },
            'preprocessing_info': {
                'augmentation_enabled': self.config.AUGMENTATION_ENABLED,
                'augmentation_parameters': {
                    'rotation_range': self.config.ROTATION_RANGE,
                    'horizontal_flip': self.config.HORIZONTAL_FLIP,
                    'vertical_flip': self.config.VERTICAL_FLIP,
                    'brightness_range': self.config.BRIGHTNESS_RANGE,
                    'contrast_range': self.config.CONTRAST_RANGE
                },
                'normalization_method': 'hugging_face_processor' if self.preprocessor.hf_processor else 'basic_minmax',
                'model_name': self.config.MODEL_NAME
            },
            'data_shapes': {
                'training_images': train_images.shape if len(train_images) > 0 else None,
                'validation_images': val_images.shape if len(val_images) > 0 else None,
                'test_images': test_images.shape if len(test_images) > 0 else None
            }
        }
        
        return report
    
    def save_report(self, report: Dict, output_path: str = 'reports/preprocessing_report.json'):
        """Save report to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Preprocessing report saved to {output_file}")


def main():
    """Generate preprocessing report"""
    generator = PreprocessingReportGenerator()
    
    # Generate report
    report = generator.generate_report()
    
    if report:
        # Save report
        generator.save_report(report)
        
        # Print summary
        print("\n" + "=" * 60)
        print("PREPROCESSING REPORT SUMMARY")
        print("=" * 60)
        print(f"Training Images: {report['dataset_statistics']['training_image_count']}")
        print(f"Validation Images: {report['dataset_statistics']['validation_image_count']}")
        print(f"Test Images: {report['dataset_statistics']['test_image_count']}")
        print(f"Total Images: {report['dataset_statistics']['total_images']}")
        print(f"Image Size: {report['dataset_statistics']['image_size']}")
        print(f"Number of Classes: {report['dataset_statistics']['num_classes']}")
        print(f"Class Names: {', '.join(report['dataset_statistics']['class_names'])}")
        print(f"Class Mapping: {report['dataset_statistics']['class_to_index']}")
        print(f"\nClass Distribution:")
        for split in ['training', 'validation', 'test']:
            print(f"\n{split.capitalize()}:")
            for class_name, count in report['class_distribution'][split].items():
                print(f"  {class_name}: {count}")
        print(f"\nPreprocessing:")
        print(f"  Augmentation: {report['preprocessing_info']['augmentation_enabled']}")
        print(f"  Normalization: {report['preprocessing_info']['normalization_method']}")
        print(f"  Model: {report['preprocessing_info']['model_name']}")
        print("=" * 60)


if __name__ == '__main__':
    main()
