"""
Dataset Analysis Script for Module 7
Analyzes the Brain Tumor MRI dataset structure and statistics
"""

import os
import json
from pathlib import Path
from PIL import Image
import hashlib
from collections import defaultdict
from typing import Dict, List, Tuple
import logging
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """Analyzes brain tumor MRI dataset structure and statistics"""
    
    def __init__(self, project_root: str = '.'):
        self.project_root = Path(project_root)
        self.dataset_paths = {
            'training': self.project_root / 'Training',
            'testing': self.project_root / 'Testing'
        }
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        
        # Analysis results
        self.classes = set()
        self.class_counts = defaultdict(lambda: {'training': 0, 'testing': 0})
        self.image_formats = set()
        self.image_dimensions = set()
        self.corrupted_images = []
        self.duplicate_files = []
        self.total_images = 0
        self.training_count = 0
        self.testing_count = 0
        
    def find_dataset_root(self) -> Path:
        """Find the dataset root directory"""
        logger.info("Searching for dataset root directory...")
        
        # Check for Training and Testing folders in project root
        if self.dataset_paths['training'].exists() and self.dataset_paths['testing'].exists():
            logger.info(f"Found dataset at project root: {self.project_root}")
            return self.project_root
        
        # Check datasets folder
        datasets_folder = self.project_root / 'datasets'
        if datasets_folder.exists():
            for subdir in datasets_folder.iterdir():
                if subdir.is_dir():
                    training = subdir / 'Training'
                    testing = subdir / 'Testing'
                    if training.exists() and testing.exists():
                        logger.info(f"Found dataset at: {subdir}")
                        return subdir
        
        logger.warning("Could not find Training/Testing folders")
        return None
    
    def get_folder_structure(self, root_path: Path) -> Dict:
        """Get complete folder structure"""
        structure = {}
        
        for split in ['Training', 'Testing']:
            split_path = root_path / split
            if split_path.exists():
                structure[split] = {}
                for class_folder in split_path.iterdir():
                    if class_folder.is_dir():
                        structure[split][class_folder.name] = {
                            'path': str(class_folder),
                            'files': [f.name for f in class_folder.iterdir() if f.is_file()]
                        }
                        self.classes.add(class_folder.name)
        
        return structure
    
    def count_images_by_class(self) -> Dict:
        """Count images in each class for training and testing"""
        logger.info("Counting images by class...")
        
        for split_name, split_path in self.dataset_paths.items():
            if not split_path.exists():
                continue
            
            split_key = 'training' if split_name == 'Training' else 'testing'
            
            for class_folder in split_path.iterdir():
                if class_folder.is_dir():
                    class_name = class_folder.name
                    image_count = 0
                    
                    for file in class_folder.iterdir():
                        if file.is_file() and file.suffix.lower() in self.supported_formats:
                            image_count += 1
                    
                    self.class_counts[class_name][split_key] = image_count
                    logger.info(f"{split_name}/{class_name}: {image_count} images")
        
        return dict(self.class_counts)
    
    def analyze_image_properties(self) -> Dict:
        """Analyze image formats, dimensions, and channels"""
        logger.info("Analyzing image properties...")
        
        formats_count = defaultdict(int)
        dimensions_count = defaultdict(int)
        channels_count = defaultdict(int)
        
        for split_path in self.dataset_paths.values():
            if not split_path.exists():
                continue
            
            for class_folder in split_path.iterdir():
                if not class_folder.is_dir():
                    continue
                
                for file in class_folder.iterdir():
                    if file.is_file() and file.suffix.lower() in self.supported_formats:
                        try:
                            with Image.open(file) as img:
                                # Format
                                fmt = file.suffix.lower()
                                formats_count[fmt] += 1
                                self.image_formats.add(fmt)
                                
                                # Dimensions
                                dims = img.size
                                dimensions_count[dims] += 1
                                self.image_dimensions.add(dims)
                                
                                # Channels
                                mode = img.mode
                                if mode == 'RGB':
                                    channels_count[3] += 1
                                elif mode == 'RGBA':
                                    channels_count[4] += 1
                                elif mode == 'L':
                                    channels_count[1] += 1
                                elif mode == 'CMYK':
                                    channels_count[4] += 1
                                
                        except Exception as e:
                            logger.error(f"Error analyzing {file}: {e}")
                            self.corrupted_images.append(str(file))
        
        return {
            'formats': dict(formats_count),
            'dimensions': {str(k): v for k, v in dimensions_count.items()},
            'channels': dict(channels_count)
        }
    
    def check_corrupted_images(self) -> List[str]:
        """Check for corrupted or unreadable images"""
        logger.info("Checking for corrupted images...")
        
        corrupted = []
        
        for split_path in self.dataset_paths.values():
            if not split_path.exists():
                continue
            
            for class_folder in split_path.iterdir():
                if not class_folder.is_dir():
                    continue
                
                for file in class_folder.iterdir():
                    if file.is_file() and file.suffix.lower() in self.supported_formats:
                        try:
                            with Image.open(file) as img:
                                img.verify()
                                # Re-open after verify
                                with Image.open(file) as img2:
                                    img2.load()
                        except Exception as e:
                            logger.error(f"Corrupted image: {file} - {e}")
                            corrupted.append(str(file))
        
        self.corrupted_images = corrupted
        logger.info(f"Found {len(corrupted)} corrupted images")
        return corrupted
    
    def check_duplicate_files(self) -> List[Tuple[str, str]]:
        """Check for duplicate files using hash comparison"""
        logger.info("Checking for duplicate files...")
        
        file_hashes = {}
        duplicates = []
        
        for split_path in self.dataset_paths.values():
            if not split_path.exists():
                continue
            
            for class_folder in split_path.iterdir():
                if not class_folder.is_dir():
                    continue
                
                for file in class_folder.iterdir():
                    if file.is_file() and file.suffix.lower() in self.supported_formats:
                        try:
                            # Calculate file hash
                            file_hash = hashlib.md5()
                            with open(file, 'rb') as f:
                                for chunk in iter(lambda: f.read(4096), b''):
                                    file_hash.update(chunk)
                            hash_value = file_hash.hexdigest()
                            
                            if hash_value in file_hashes:
                                duplicates.append((str(file_hashes[hash_value]), str(file)))
                                logger.warning(f"Duplicate found: {file_hashes[hash_value]} == {file}")
                            else:
                                file_hashes[hash_value] = str(file)
                                
                        except Exception as e:
                            logger.error(f"Error hashing {file}: {e}")
        
        self.duplicate_files = duplicates
        logger.info(f"Found {len(duplicates)} duplicate files")
        return duplicates
    
    def get_total_counts(self) -> Dict:
        """Get total image counts"""
        training_total = sum(counts['training'] for counts in self.class_counts.values())
        testing_total = sum(counts['testing'] for counts in self.class_counts.values())
        total = training_total + testing_total
        
        self.total_images = total
        self.training_count = training_total
        self.testing_count = testing_total
        
        return {
            'total': total,
            'training': training_total,
            'testing': testing_total
        }
    
    def generate_report(self) -> Dict:
        """Generate comprehensive dataset report"""
        logger.info("Generating dataset report...")
        
        report = {
            'dataset_path': str(self.project_root),
            'dataset_type': 'Brain Tumor MRI Dataset',
            'classes': sorted(list(self.classes)),
            'class_distribution': {
                class_name: dict(counts) 
                for class_name, counts in self.class_counts.items()
            },
            'total_images': self.total_images,
            'training_images': self.training_count,
            'testing_images': self.testing_count,
            'image_formats': sorted(list(self.image_formats)),
            'image_dimensions': [str(d) for d in sorted(self.image_dimensions)],
            'corrupted_images': self.corrupted_images,
            'corrupted_image_count': len(self.corrupted_images),
            'duplicate_files': self.duplicate_files,
            'duplicate_file_count': len(self.duplicate_files),
            'folder_structure': {
                'training': str(self.dataset_paths['training']),
                'testing': str(self.dataset_paths['testing'])
            }
        }
        
        return report
    
    def save_report(self, report: Dict, output_path: str = 'reports/dataset_report.json'):
        """Save report to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {output_file}")
    
    def create_visualization(self, report: Dict, output_path: str = 'reports/class_distribution.png'):
        """Create class distribution visualization"""
        logger.info("Creating class distribution visualization...")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        classes = report['classes']
        training_counts = [report['class_distribution'][cls]['training'] for cls in classes]
        testing_counts = [report['class_distribution'][cls]['testing'] for cls in classes]
        
        # Create bar chart
        x = np.arange(len(classes))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, training_counts, width, label='Training', color='skyblue')
        bars2 = ax.bar(x + width/2, testing_counts, width, label='Testing', color='lightcoral')
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Number of Images')
        ax.set_title('Brain Tumor MRI Dataset - Class Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {output_path}")
    
    def run_full_analysis(self) -> Dict:
        """Run complete dataset analysis"""
        logger.info("=" * 60)
        logger.info("Starting Brain Tumor MRI Dataset Analysis")
        logger.info("=" * 60)
        
        # Find dataset
        dataset_root = self.find_dataset_root()
        if not dataset_root:
            logger.error("Dataset not found!")
            return None
        
        # Get folder structure
        structure = self.get_folder_structure(dataset_root)
        logger.info(f"Folder structure: {list(structure.keys())}")
        
        # Count images by class
        class_counts = self.count_images_by_class()
        
        # Analyze image properties
        image_properties = self.analyze_image_properties()
        
        # Check corrupted images
        corrupted = self.check_corrupted_images()
        
        # Check duplicates
        duplicates = self.check_duplicate_files()
        
        # Get total counts
        totals = self.get_total_counts()
        
        # Generate report
        report = self.generate_report()
        
        # Save report
        self.save_report(report)
        
        # Create visualization
        self.create_visualization(report)
        
        logger.info("=" * 60)
        logger.info("Dataset Analysis Complete")
        logger.info("=" * 60)
        
        return report


def main():
    """Main function to run dataset analysis"""
    analyzer = DatasetAnalyzer()
    report = analyzer.run_full_analysis()
    
    if report:
        print("\n" + "=" * 60)
        print("DATASET ANALYSIS RESULTS")
        print("=" * 60)
        print(f"Dataset Path: {report['dataset_path']}")
        print(f"Classes: {', '.join(report['classes'])}")
        print(f"Total Images: {report['total_images']}")
        print(f"Training Images: {report['training_images']}")
        print(f"Testing Images: {report['testing_images']}")
        print(f"Image Formats: {', '.join(report['image_formats'])}")
        print(f"Image Dimensions: {', '.join(report['image_dimensions'])}")
        print(f"Corrupted Images: {report['corrupted_image_count']}")
        print(f"Duplicate Files: {report['duplicate_file_count']}")
        print("\nClass Distribution:")
        for class_name, counts in report['class_distribution'].items():
            print(f"  {class_name}:")
            print(f"    Training: {counts['training']}")
            print(f"    Testing: {counts['testing']}")
            print(f"    Total: {counts['training'] + counts['testing']}")
        print("=" * 60)


if __name__ == '__main__':
    main()
