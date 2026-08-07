"""
Dataset Loader for Module 5
Loads and preprocesses cancer detection datasets
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetLoader:
    """Base class for loading cancer detection datasets"""
    
    def __init__(self, data_dir: str = 'datasets'):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_metadata(self, metadata_path: Path) -> pd.DataFrame:
        """
        Load metadata from CSV file
        
        Args:
            metadata_path: Path to metadata CSV file
            
        Returns:
            DataFrame with metadata
        """
        try:
            df = pd.read_csv(metadata_path)
            logger.info(f"Loaded metadata with {len(df)} records from {metadata_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return pd.DataFrame()
    
    def load_image(self, image_path: Path, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Load and optionally resize an image
        
        Args:
            image_path: Path to image file
            target_size: Optional target size (width, height)
            
        Returns:
            Image as numpy array
        """
        try:
            img = Image.open(image_path)
            
            if target_size:
                img = img.resize(target_size, Image.LANCZOS)
            
            img_array = np.array(img)
            
            # Normalize to 0-1 range
            if img_array.max() > 1:
                img_array = img_array / 255.0
            
            return img_array
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return np.array([])


class ISICDatasetLoader(DatasetLoader):
    """Loader for ISIC skin cancer datasets"""
    
    def __init__(self, data_dir: str = 'datasets'):
        super().__init__(data_dir)
        self.skin_dir = self.data_dir / 'skin_cancer'
    
    def load_isic_dicm_17k(self, metadata_file: str = 'metadata.csv') -> Tuple[pd.DataFrame, Dict]:
        """
        Load ISIC-DICM-17K dataset
        
        Args:
            metadata_file: Name of metadata file
            
        Returns:
            Tuple of (metadata DataFrame, dataset info dict)
        """
        metadata_path = self.skin_dir / 'isic_dicm_17k' / metadata_file
        
        if not metadata_path.exists():
            logger.error(f"Metadata file not found: {metadata_path}")
            return pd.DataFrame(), {}
        
        df = self.load_metadata(metadata_path)
        
        dataset_info = {
            'name': 'ISIC-DICM-17K',
            'total_images': len(df),
            'classes': df['diagnosis'].unique() if 'diagnosis' in df.columns else [],
            'image_dir': self.skin_dir / 'isic_dicm_17k',
            'target_column': 'diagnosis' if 'diagnosis' in df.columns else None
        }
        
        logger.info(f"Loaded ISIC-DICM-17K: {dataset_info['total_images']} images")
        logger.info(f"Classes: {dataset_info['classes']}")
        
        return df, dataset_info
    
    def load_images_batch(self, image_paths: List[Path], target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Load a batch of images
        
        Args:
            image_paths: List of image paths
            target_size: Target size for images
            
        Returns:
            Array of images
        """
        images = []
        for img_path in image_paths:
            img = self.load_image(img_path, target_size)
            if img.size > 0:
                images.append(img)
        
        return np.array(images) if images else np.array([])


class LIDCDatasetLoader(DatasetLoader):
    """Loader for LIDC-IDRI lung cancer dataset"""
    
    def __init__(self, data_dir: str = 'datasets'):
        super().__init__(data_dir)
        self.lung_dir = self.data_dir / 'lung_cancer'
    
    def load_lidc_metadata(self, metadata_file: str = 'LIDC-IDRI_MetaData.csv') -> pd.DataFrame:
        """
        Load LIDC-IDRI metadata
        
        Args:
            metadata_file: Name of metadata file
            
        Returns:
            DataFrame with metadata
        """
        metadata_path = self.lung_dir / 'lidc_idri' / metadata_file
        
        if not metadata_path.exists():
            logger.error(f"Metadata file not found: {metadata_path}")
            return pd.DataFrame()
        
        df = self.load_metadata(metadata_path)
        
        logger.info(f"Loaded LIDC-IDRI metadata with {len(df)} records")
        
        return df
    
    def load_dicom_series(self, series_dir: Path) -> List[np.ndarray]:
        """
        Load DICOM series (requires pydicom)
        
        Args:
            series_dir: Directory containing DICOM files
            
        Returns:
            List of image arrays
        """
        try:
            import pydicom
        except ImportError:
            logger.error("pydicom not installed. Install with: pip install pydicom")
            return []
        
        images = []
        dicom_files = sorted(series_dir.glob('*.dcm'))
        
        for dcm_file in dicom_files:
            try:
                dcm = pydicom.dcmread(dcm_file)
                img_array = dcm.pixel_array
                
                # Normalize
                img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
                images.append(img_array)
                
            except Exception as e:
                logger.error(f"Failed to load DICOM file {dcm_file}: {e}")
        
        return images


class CBISDDSMLoader(DatasetLoader):
    """Loader for CBIS-DDSM breast cancer dataset"""
    
    def __init__(self, data_dir: str = 'datasets'):
        super().__init__(data_dir)
        self.breast_dir = self.data_dir / 'breast_cancer'
    
    def load_cbis_metadata(self, metadata_file: str = 'mass_case_description_train_set.csv') -> pd.DataFrame:
        """
        Load CBIS-DDSM metadata
        
        Args:
            metadata_file: Name of metadata file
            
        Returns:
            DataFrame with metadata
        """
        metadata_path = self.breast_dir / 'cbis_ddsm' / metadata_file
        
        if not metadata_path.exists():
            logger.error(f"Metadata file not found: {metadata_path}")
            return pd.DataFrame()
        
        df = self.load_metadata(metadata_path)
        
        dataset_info = {
            'name': 'CBIS-DDSM',
            'total_cases': len(df),
            'pathology': df['pathology'].unique() if 'pathology' in df.columns else [],
            'assessment': df['assessment'].unique() if 'assessment' in df.columns else []
        }
        
        logger.info(f"Loaded CBIS-DDSM: {dataset_info['total_cases']} cases")
        logger.info(f"Pathology types: {dataset_info['pathology']}")
        
        return df, dataset_info


def get_dataset_loader(dataset_type: str, data_dir: str = 'datasets') -> DatasetLoader:
    """
    Factory function to get appropriate dataset loader
    
    Args:
        dataset_type: Type of dataset ('skin', 'lung', 'breast')
        data_dir: Base directory for datasets
        
    Returns:
        DatasetLoader instance
    """
    loaders = {
        'skin': ISICDatasetLoader,
        'lung': LIDCDatasetLoader,
        'breast': CBISDDSMLoader
    }
    
    if dataset_type not in loaders:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(loaders.keys())}")
    
    return loaders[dataset_type](data_dir)


def main():
    """Main function to test dataset loaders"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Load cancer detection datasets')
    parser.add_argument('--dataset', type=str, required=True, choices=['skin', 'lung', 'breast'],
                       help='Dataset type to load')
    parser.add_argument('--data-dir', type=str, default='datasets', help='Datasets directory')
    
    args = parser.parse_args()
    
    loader = get_dataset_loader(args.dataset, args.data_dir)
    
    if args.dataset == 'skin':
        df, info = loader.load_isic_dicm_17k()
        print(f"Loaded {info['name']}: {info['total_images']} images")
    elif args.dataset == 'lung':
        df = loader.load_lidc_metadata()
        print(f"Loaded LIDC-IDRI metadata: {len(df)} records")
    elif args.dataset == 'breast':
        df, info = loader.load_cbis_metadata()
        print(f"Loaded {info['name']}: {info['total_cases']} cases")


if __name__ == '__main__':
    main()
