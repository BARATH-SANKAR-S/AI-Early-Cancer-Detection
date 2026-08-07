"""
Dataset Download Script for Module 5
Downloads publicly available cancer detection datasets
"""

import os
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dataset configurations
DATASETS = {
    'isic_dicm_17k': {
        'name': 'ISIC-DICM-17K',
        'url': 'https://api.isic-archive.com/api/v1/download/10.34970/233480',
        'destination': 'datasets/skin_cancer/isic_dicm_17k',
        'filename': 'isic_dicm_17k.zip',
        'description': 'Skin cancer dataset with 17,060 dermoscopic images'
    },
    'lidc_idri': {
        'name': 'LIDC-IDRI',
        'url': 'https://wiki.cancerimagingarchive.net/images/1/13/LIDC-IDRI_MetaData.csv.zip',
        'destination': 'datasets/lung_cancer/lidc_idri',
        'filename': 'lidc_idri_metadata.zip',
        'description': 'Lung cancer dataset with CT scans (metadata only for demo)',
        'note': 'Full dataset requires TCIA Data Retriever tool'
    }
}


class DatasetDownloader:
    """Handles downloading and extracting datasets"""
    
    def __init__(self, datasets_dir: str = 'datasets'):
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, url: str, destination: Path, filename: str, 
                     chunk_size: int = 8192) -> Optional[Path]:
        """
        Download a file from URL to destination
        
        Args:
            url: URL to download from
            destination: Directory to save file
            filename: Name of the file to save
            chunk_size: Size of chunks for download
            
        Returns:
            Path to downloaded file or None if failed
        """
        destination.mkdir(parents=True, exist_ok=True)
        file_path = destination / filename
        
        # Check if file already exists
        if file_path.exists():
            logger.info(f"File already exists: {file_path}")
            return file_path
        
        try:
            logger.info(f"Downloading {url} to {file_path}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            logger.info(f"Downloaded: {percent:.1f}%")
            
            logger.info(f"Download completed: {file_path}")
            return file_path
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Download failed: {e}")
            return None
    
    def extract_zip(self, zip_path: Path, destination: Path) -> bool:
        """
        Extract a zip file to destination
        
        Args:
            zip_path: Path to zip file
            destination: Directory to extract to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Extracting {zip_path} to {destination}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(destination)
            logger.info(f"Extraction completed")
            return True
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return False
    
    def extract_tar(self, tar_path: Path, destination: Path) -> bool:
        """
        Extract a tar file to destination
        
        Args:
            tar_path: Path to tar file
            destination: Directory to extract to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Extracting {tar_path} to {destination}")
            with tarfile.open(tar_path, 'r:*') as tar_ref:
                tar_ref.extractall(destination)
            logger.info(f"Extraction completed")
            return True
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return False
    
    def download_dataset(self, dataset_key: str, extract: bool = True) -> bool:
        """
        Download and optionally extract a dataset
        
        Args:
            dataset_key: Key from DATASETS dictionary
            extract: Whether to extract the downloaded file
            
        Returns:
            True if successful, False otherwise
        """
        if dataset_key not in DATASETS:
            logger.error(f"Unknown dataset key: {dataset_key}")
            return False
        
        config = DATASETS[dataset_key]
        logger.info(f"Downloading dataset: {config['name']}")
        logger.info(f"Description: {config['description']}")
        
        if 'note' in config:
            logger.info(f"Note: {config['note']}")
        
        destination = self.datasets_dir / config['destination']
        downloaded_file = self.download_file(
            config['url'],
            destination,
            config['filename']
        )
        
        if downloaded_file is None:
            return False
        
        if extract:
            if downloaded_file.suffix == '.zip':
                return self.extract_zip(downloaded_file, destination)
            elif downloaded_file.suffix in ['.tar', '.gz', '.tgz']:
                return self.extract_tar(downloaded_file, destination)
        
        return True
    
    def download_all(self, extract: bool = True) -> dict:
        """
        Download all configured datasets
        
        Args:
            extract: Whether to extract downloaded files
            
        Returns:
            Dictionary with download results
        """
        results = {}
        for key in DATASETS:
            results[key] = self.download_dataset(key, extract)
        return results


def main():
    """Main function to run dataset downloads"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download cancer detection datasets')
    parser.add_argument('--dataset', type=str, help='Specific dataset to download (e.g., isic_dicm_17k)')
    parser.add_argument('--all', action='store_true', help='Download all datasets')
    parser.add_argument('--no-extract', action='store_true', help='Do not extract downloaded files')
    parser.add_argument('--datasets-dir', type=str, default='datasets', help='Datasets directory')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.datasets_dir)
    
    if args.all:
        logger.info("Downloading all datasets...")
        results = downloader.download_all(extract=not args.no_extract)
        for key, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            logger.info(f"{DATASETS[key]['name']}: {status}")
    elif args.dataset:
        logger.info(f"Downloading dataset: {args.dataset}")
        success = downloader.download_dataset(args.dataset, extract=not args.no_extract)
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"Download {status}")
    else:
        logger.info("Available datasets:")
        for key, config in DATASETS.items():
            logger.info(f"  {key}: {config['name']}")
            logger.info(f"    {config['description']}")


if __name__ == '__main__':
    main()
