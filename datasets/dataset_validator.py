"""
Dataset Validator for Module 5
Validates integrity and quality of cancer detection datasets
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import logging
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetValidator:
    """Base class for validating datasets"""
    
    def __init__(self, data_dir: str = 'datasets'):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        
    def validate_directory_structure(self) -> Dict[str, bool]:
        """
        Validate that required directories exist
        
        Returns:
            Dictionary with directory validation results
        """
        required_dirs = {
            'datasets': self.data_dir,
            'raw': self.raw_dir,
            'processed': self.processed_dir,
            'lung_cancer': self.data_dir / 'lung_cancer',
            'breast_cancer': self.data_dir / 'breast_cancer',
            'skin_cancer': self.data_dir / 'skin_cancer'
        }
        
        results = {}
        for name, path in required_dirs.items():
            results[name] = path.exists()
            if results[name]:
                logger.info(f"✓ Directory exists: {name}")
            else:
                logger.warning(f"✗ Directory missing: {name}")
        
        return results
    
    def calculate_file_hash(self, file_path: Path, algorithm: str = 'md5') -> Optional[str]:
        """
        Calculate hash of a file
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm (md5, sha256)
            
        Returns:
            Hash string or None if failed
        """
        try:
            hash_func = hashlib.new(algorithm)
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return None
    
    def validate_file_integrity(self, file_path: Path, expected_hash: Optional[str] = None) -> Dict:
        """
        Validate file integrity
        
        Args:
            file_path: Path to file
            expected_hash: Optional expected hash value
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'exists': file_path.exists(),
            'size': 0,
            'hash': None,
            'hash_match': False
        }
        
        if not result['exists']:
            return result
        
        try:
            result['size'] = file_path.stat().st_size
            result['hash'] = self.calculate_file_hash(file_path)
            
            if expected_hash and result['hash']:
                result['hash_match'] = result['hash'] == expected_hash
            
            logger.info(f"File: {file_path.name} - Size: {result['size']} bytes - Hash: {result['hash']}")
            
        except Exception as e:
            logger.error(f"Failed to validate file {file_path}: {e}")
        
        return result


class ImageDatasetValidator(DatasetValidator):
    """Validator for image-based datasets"""
    
    def validate_image(self, image_path: Path, min_size: Tuple[int, int] = (32, 32)) -> Dict:
        """
        Validate an image file
        
        Args:
            image_path: Path to image file
            min_size: Minimum allowed size (width, height)
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'exists': image_path.exists(),
            'valid': False,
            'size': (0, 0),
            'mode': None,
            'channels': 0,
            'corrupted': False
        }
        
        if not result['exists']:
            return result
        
        try:
            with Image.open(image_path) as img:
                result['size'] = img.size
                result['mode'] = img.mode
                
                # Check number of channels
                if img.mode in ('RGB', 'RGBA'):
                    result['channels'] = len(img.mode)
                elif img.mode == 'L':
                    result['channels'] = 1
                else:
                    result['channels'] = 0
                
                # Check minimum size
                result['valid'] = (img.size[0] >= min_size[0] and img.size[1] >= min_size[1])
                
                # Try to load pixels to check for corruption
                img.load()
                
        except Exception as e:
            logger.error(f"Image validation failed for {image_path}: {e}")
            result['corrupted'] = True
        
        return result
    
    def validate_image_directory(self, image_dir: Path, extensions: List[str] = ['.jpg', '.jpeg', '.png']) -> Dict:
        """
        Validate all images in a directory
        
        Args:
            image_dir: Directory containing images
            extensions: Allowed file extensions
            
        Returns:
            Dictionary with validation summary
        """
        result = {
            'total_files': 0,
            'valid_images': 0,
            'invalid_images': 0,
            'corrupted_images': 0,
            'issues': []
        }
        
        if not image_dir.exists():
            result['issues'].append(f"Directory does not exist: {image_dir}")
            return result
        
        for ext in extensions:
            image_files = list(image_dir.glob(f'*{ext}')) + list(image_dir.glob(f'*{ext.upper()}'))
            result['total_files'] += len(image_files)
            
            for img_file in image_files:
                validation = self.validate_image(img_file)
                
                if not validation['exists']:
                    continue
                
                if validation['corrupted']:
                    result['corrupted_images'] += 1
                    result['issues'].append(f"Corrupted image: {img_file.name}")
                elif validation['valid']:
                    result['valid_images'] += 1
                else:
                    result['invalid_images'] += 1
                    result['issues'].append(f"Invalid image (size too small): {img_file.name}")
        
        logger.info(f"Image validation: {result['valid_images']}/{result['total_files']} valid")
        
        return result


class MetadataValidator(DatasetValidator):
    """Validator for metadata files"""
    
    def validate_csv(self, csv_path: Path, required_columns: Optional[List[str]] = None) -> Dict:
        """
        Validate a CSV metadata file
        
        Args:
            csv_path: Path to CSV file
            required_columns: List of required column names
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'exists': csv_path.exists(),
            'valid': False,
            'rows': 0,
            'columns': [],
            'missing_columns': [],
            'null_counts': {}
        }
        
        if not result['exists']:
            return result
        
        try:
            df = pd.read_csv(csv_path)
            result['rows'] = len(df)
            result['columns'] = df.columns.tolist()
            
            # Check for required columns
            if required_columns:
                result['missing_columns'] = [col for col in required_columns if col not in df.columns]
                result['valid'] = len(result['missing_columns']) == 0
            else:
                result['valid'] = True
            
            # Check for null values
            result['null_counts'] = df.isnull().sum().to_dict()
            
            logger.info(f"CSV validation: {result['rows']} rows, {len(result['columns'])} columns")
            
            if result['missing_columns']:
                logger.warning(f"Missing columns: {result['missing_columns']}")
            
        except Exception as e:
            logger.error(f"CSV validation failed for {csv_path}: {e}")
        
        return result


class ISICDatasetValidator(ImageDatasetValidator, MetadataValidator):
    """Validator for ISIC skin cancer datasets"""
    
    def validate_isic_dataset(self, dataset_dir: Path) -> Dict:
        """
        Validate ISIC dataset structure and content
        
        Args:
            dataset_dir: Path to ISIC dataset directory
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'dataset_name': 'ISIC',
            'structure_valid': False,
            'metadata_valid': False,
            'images_valid': False,
            'total_images': 0,
            'issues': []
        }
        
        # Check directory structure
        if not dataset_dir.exists():
            result['issues'].append(f"Dataset directory does not exist: {dataset_dir}")
            return result
        
        result['structure_valid'] = True
        
        # Validate metadata
        metadata_files = list(dataset_dir.glob('*.csv'))
        if metadata_files:
            metadata_validation = self.validate_csv(metadata_files[0], required_columns=['diagnosis'])
            result['metadata_valid'] = metadata_validation['valid']
            result['issues'].extend([f"Metadata: {issue}" for issue in metadata_validation.get('missing_columns', [])])
        else:
            result['issues'].append("No metadata CSV file found")
        
        # Validate images
        image_dir = dataset_dir
        image_validation = self.validate_image_directory(image_dir)
        result['images_valid'] = image_validation['valid_images'] > 0
        result['total_images'] = image_validation['total_files']
        result['issues'].extend(image_validation['issues'])
        
        return result


class LIDCDatasetValidator(DatasetValidator):
    """Validator for LIDC-IDRI lung cancer dataset"""
    
    def validate_lidc_dataset(self, dataset_dir: Path) -> Dict:
        """
        Validate LIDC-IDRI dataset structure
        
        Args:
            dataset_dir: Path to LIDC dataset directory
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'dataset_name': 'LIDC-IDRI',
            'structure_valid': False,
            'metadata_valid': False,
            'issues': []
        }
        
        if not dataset_dir.exists():
            result['issues'].append(f"Dataset directory does not exist: {dataset_dir}")
            return result
        
        result['structure_valid'] = True
        
        # Check for metadata
        metadata_files = list(dataset_dir.glob('*.csv'))
        if metadata_files:
            result['metadata_valid'] = True
            logger.info(f"Found metadata file: {metadata_files[0].name}")
        else:
            result['issues'].append("No metadata CSV file found")
        
        # LIDC uses DICOM files - check for .dcm files
        dicom_files = list(dataset_dir.rglob('*.dcm'))
        if dicom_files:
            logger.info(f"Found {len(dicom_files)} DICOM files")
        else:
            result['issues'].append("No DICOM files found")
        
        return result


def validate_all_datasets(data_dir: str = 'datasets') -> Dict:
    """
    Validate all datasets in the data directory
    
    Args:
        data_dir: Base directory for datasets
        
    Returns:
        Dictionary with validation results for all datasets
    """
    base_validator = DatasetValidator(data_dir)
    
    # Validate directory structure
    dir_results = base_validator.validate_directory_structure()
    
    # Validate specific datasets
    results = {
        'directory_structure': dir_results,
        'datasets': {}
    }
    
    # ISIC Dataset
    isic_validator = ISICDatasetValidator(data_dir)
    isic_dir = Path(data_dir) / 'skin_cancer' / 'isic_dicm_17k'
    results['datasets']['isic'] = isic_validator.validate_isic_dataset(isic_dir)
    
    # LIDC Dataset
    lidc_validator = LIDCDatasetValidator(data_dir)
    lidc_dir = Path(data_dir) / 'lung_cancer' / 'lidc_idri'
    results['datasets']['lidc'] = lidc_validator.validate_lidc_dataset(lidc_dir)
    
    return results


def main():
    """Main function to run dataset validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate cancer detection datasets')
    parser.add_argument('--data-dir', type=str, default='datasets', help='Datasets directory')
    parser.add_argument('--dataset', type=str, choices=['all', 'isic', 'lidc'], default='all',
                       help='Dataset to validate')
    
    args = parser.parse_args()
    
    if args.dataset == 'all':
        results = validate_all_datasets(args.data_dir)
        print("\n=== Validation Results ===")
        print(f"Directory Structure: {sum(results['directory_structure'].values())}/{len(results['directory_structure'])} valid")
        for dataset_name, dataset_results in results['datasets'].items():
            print(f"\n{dataset_results['dataset_name']}:")
            print(f"  Structure: {'✓' if dataset_results['structure_valid'] else '✗'}")
            print(f"  Metadata: {'✓' if dataset_results.get('metadata_valid', False) else '✗'}")
            print(f"  Images: {'✓' if dataset_results.get('images_valid', False) else '✗'}")
            if dataset_results.get('total_images'):
                print(f"  Total Images: {dataset_results['total_images']}")
            if dataset_results['issues']:
                print(f"  Issues: {len(dataset_results['issues'])}")
    else:
        if args.dataset == 'isic':
            validator = ISICDatasetValidator(args.data_dir)
            results = validator.validate_isic_dataset(Path(args.data_dir) / 'skin_cancer' / 'isic_dicm_17k')
        elif args.dataset == 'lidc':
            validator = LIDCDatasetValidator(args.data_dir)
            results = validator.validate_lidc_dataset(Path(args.data_dir) / 'lung_cancer' / 'lidc_idri')
        
        print(f"\n=== {results['dataset_name']} Validation ===")
        print(f"Structure: {'✓' if results['structure_valid'] else '✗'}")
        print(f"Metadata: {'✓' if results.get('metadata_valid', False) else '✗'}")
        print(f"Images: {'✓' if results.get('images_valid', False) else '✗'}")
        if results['issues']:
            print("\nIssues:")
            for issue in results['issues']:
                print(f"  - {issue}")


if __name__ == '__main__':
    main()
