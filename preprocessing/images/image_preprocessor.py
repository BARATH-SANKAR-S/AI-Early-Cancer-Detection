"""
Medical Image Preprocessor for Module 6
Handles resize, normalization, noise removal, contrast enhancement, and data augmentation
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from typing import Tuple, List, Optional, Union
import logging
from pathlib import Path
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Complete preprocessing pipeline for medical images
    
    Steps:
    1. Resize
    2. Normalize
    3. Noise Removal
    4. Contrast Enhancement
    5. Data Augmentation
    """
    
    def __init__(self, output_dir: str = 'datasets/processed'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def resize_image(self, image: Union[np.ndarray, Image.Image], 
                    target_size: Tuple[int, int], 
                    method: str = 'bilinear') -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image (numpy array or PIL Image)
            target_size: Target size (width, height)
            method: Resampling method ('nearest', 'bilinear', 'bicubic', 'lanczos')
            
        Returns:
            Resized image as numpy array
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            img_pil = image
        else:
            img_pil = Image.fromarray(image)
        
        # Map method to PIL resampling filter
        method_map = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'lanczos': Image.LANCZOS
        }
        
        resample_filter = method_map.get(method, Image.BILINEAR)
        
        # Resize image
        resized_img = img_pil.resize(target_size, resample_filter)
        
        logger.info(f"Resized image to {target_size} using {method}")
        
        return np.array(resized_img)
    
    def normalize_image(self, image: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize image pixel values
        
        Args:
            image: Input image as numpy array
            method: Normalization method ('minmax', 'zscore', 'percentile')
            
        Returns:
            Normalized image
        """
        logger.info(f"Normalizing image using {method} method")
        
        if method == 'minmax':
            # Min-Max normalization to [0, 1]
            normalized = (image - image.min()) / (image.max() - image.min() + 1e-8)
            
        elif method == 'zscore':
            # Z-score normalization (mean=0, std=1)
            normalized = (image - image.mean()) / (image.std() + 1e-8)
            
        elif method == 'percentile':
            # Percentile clipping (1st and 99th percentiles)
            p1, p99 = np.percentile(image, (1, 99))
            normalized = np.clip(image, p1, p99)
            normalized = (normalized - p1) / (p99 - p1 + 1e-8)
            
        else:
            logger.warning(f"Unknown normalization method: {method}. Using minmax.")
            normalized = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        return normalized
    
    def remove_noise(self, image: np.ndarray, method: str = 'gaussian') -> np.ndarray:
        """
        Remove noise from image
        
        Args:
            image: Input image as numpy array
            method: Denoising method ('gaussian', 'median', 'bilateral', 'non_local_means')
            
        Returns:
            Denoised image
        """
        logger.info(f"Removing noise using {method} method")
        
        # Convert to uint8 for OpenCV operations
        if image.dtype != np.uint8:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image
        
        if method == 'gaussian':
            # Gaussian blur
            denoised = cv2.GaussianBlur(image_uint8, (5, 5), 0)
            
        elif method == 'median':
            # Median filter (good for salt-and-pepper noise)
            denoised = cv2.medianBlur(image_uint8, 5)
            
        elif method == 'bilateral':
            # Bilateral filter (preserves edges while reducing noise)
            denoised = cv2.bilateralFilter(image_uint8, 9, 75, 75)
            
        elif method == 'non_local_means':
            # Non-local means denoising
            if len(image_uint8.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(image_uint8, None, 10, 10, 7, 21)
            else:
                denoised = cv2.fastNlMeansDenoising(image_uint8, None, 10, 7, 21)
                
        else:
            logger.warning(f"Unknown denoising method: {method}. Using gaussian.")
            denoised = cv2.GaussianBlur(image_uint8, (5, 5), 0)
        
        # Convert back to original dtype range
        if image.dtype != np.uint8:
            denoised = denoised.astype(np.float32) / 255.0
        
        return denoised
    
    def enhance_contrast(self, image: np.ndarray, method: str = 'clahe') -> np.ndarray:
        """
        Enhance image contrast
        
        Args:
            image: Input image as numpy array
            method: Enhancement method ('clahe', 'histogram_eq', 'gamma')
            
        Returns:
            Contrast-enhanced image
        """
        logger.info(f"Enhancing contrast using {method} method")
        
        # Convert to uint8 for OpenCV operations
        if image.dtype != np.uint8:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image
        
        if method == 'clahe':
            # Contrast Limited Adaptive Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            
            if len(image_uint8.shape) == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                enhanced = clahe.apply(image_uint8)
                
        elif method == 'histogram_eq':
            # Standard histogram equalization
            if len(image_uint8.shape) == 3:
                # Convert to YUV color space
                yuv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2YUV)
                yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
                enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
            else:
                enhanced = cv2.equalizeHist(image_uint8)
                
        elif method == 'gamma':
            # Gamma correction
            gamma = 1.5
            enhanced = np.power(image_uint8 / 255.0, gamma) * 255.0
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
        else:
            logger.warning(f"Unknown enhancement method: {method}. Using CLAHE.")
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image_uint8)
        
        # Convert back to original dtype range
        if image.dtype != np.uint8:
            enhanced = enhanced.astype(np.float32) / 255.0
        
        return enhanced
    
    def augment_image(self, image: np.ndarray, augmentations: List[str]) -> List[np.ndarray]:
        """
        Apply data augmentation to image
        
        Args:
            image: Input image as numpy array
            augmentations: List of augmentation techniques to apply
            
        Returns:
            List of augmented images
        """
        logger.info(f"Applying augmentations: {augmentations}")
        
        augmented_images = [image]
        
        for aug in augmentations:
            if aug == 'horizontal_flip':
                # Horizontal flip
                flipped = np.fliplr(image)
                augmented_images.append(flipped)
                
            elif aug == 'vertical_flip':
                # Vertical flip
                flipped = np.flipud(image)
                augmented_images.append(flipped)
                
            elif aug == 'rotation':
                # Random rotation (-15 to +15 degrees)
                angle = random.uniform(-15, 15)
                img_pil = Image.fromarray((image * 255).astype(np.uint8))
                rotated = img_pil.rotate(angle, expand=False, fillcolor=0)
                augmented_images.append(np.array(rotated) / 255.0)
                
            elif aug == 'brightness':
                # Random brightness adjustment
                img_pil = Image.fromarray((image * 255).astype(np.uint8))
                enhancer = ImageEnhance.Brightness(img_pil)
                factor = random.uniform(0.7, 1.3)
                brightened = enhancer.enhance(factor)
                augmented_images.append(np.array(brightened) / 255.0)
                
            elif aug == 'contrast':
                # Random contrast adjustment
                img_pil = Image.fromarray((image * 255).astype(np.uint8))
                enhancer = ImageEnhance.Contrast(img_pil)
                factor = random.uniform(0.7, 1.3)
                contrasted = enhancer.enhance(factor)
                augmented_images.append(np.array(contrasted) / 255.0)
                
            elif aug == 'zoom':
                # Random zoom (crop and resize)
                h, w = image.shape[:2]
                zoom_factor = random.uniform(0.8, 1.0)
                new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
                
                top = random.randint(0, h - new_h)
                left = random.randint(0, w - new_w)
                
                cropped = image[top:top+new_h, left:left+new_w]
                zoomed = cv2.resize(cropped, (w, h))
                augmented_images.append(zoomed)
                
            elif aug == 'noise':
                # Add Gaussian noise
                noise = np.random.normal(0, 0.02, image.shape)
                noisy = np.clip(image + noise, 0, 1)
                augmented_images.append(noisy)
        
        logger.info(f"Generated {len(augmented_images)} augmented images")
        
        return augmented_images
    
    def preprocess_pipeline(self, image: Union[np.ndarray, Image.Image, str],
                          target_size: Tuple[int, int] = (224, 224),
                          normalize_method: str = 'minmax',
                          denoise_method: str = 'gaussian',
                          enhance_method: str = 'clahe',
                          augmentations: Optional[List[str]] = None) -> List[np.ndarray]:
        """
        Complete preprocessing pipeline for medical images
        
        Args:
            image: Input image (numpy array, PIL Image, or file path)
            target_size: Target size for resizing
            normalize_method: Normalization method
            denoise_method: Denoising method
            enhance_method: Contrast enhancement method
            augmentations: List of augmentations to apply (None = no augmentation)
            
        Returns:
            List of preprocessed images
        """
        logger.info("Starting image preprocessing pipeline")
        
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image)
            image = np.array(image)
            logger.info(f"Loaded image from path: {image.shape}")
        
        # Step 1: Resize
        image = self.resize_image(image, target_size)
        
        # Step 2: Normalize
        image = self.normalize_image(image, method=normalize_method)
        
        # Step 3: Remove noise
        image = self.remove_noise(image, method=denoise_method)
        
        # Step 4: Enhance contrast
        image = self.enhance_contrast(image, method=enhance_method)
        
        # Step 5: Data augmentation (if specified)
        if augmentations:
            images = self.augment_image(image, augmentations)
        else:
            images = [image]
        
        logger.info(f"Preprocessing pipeline completed. Generated {len(images)} images")
        
        return images
    
    def preprocess_directory(self, input_dir: str, output_dir: str,
                           target_size: Tuple[int, int] = (224, 224),
                           normalize_method: str = 'minmax',
                           denoise_method: str = 'gaussian',
                           enhance_method: str = 'clahe',
                           augment: bool = False,
                           augmentations: Optional[List[str]] = None) -> int:
        """
        Preprocess all images in a directory
        
        Args:
            input_dir: Input directory containing images
            output_dir: Output directory for preprocessed images
            target_size: Target size for resizing
            normalize_method: Normalization method
            denoise_method: Denoising method
            enhance_method: Contrast enhancement method
            augment: Whether to apply augmentation
            augmentations: List of augmentations to apply
            
        Returns:
            Number of preprocessed images
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Supported image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        logger.info(f"Found {len(image_files)} images in {input_dir}")
        
        processed_count = 0
        
        for img_file in image_files:
            try:
                # Preprocess image
                images = self.preprocess_pipeline(
                    image=str(img_file),
                    target_size=target_size,
                    normalize_method=normalize_method,
                    denoise_method=denoise_method,
                    enhance_method=enhance_method,
                    augmentations=augmentations if augment else None
                )
                
                # Save preprocessed images
                base_name = img_file.stem
                for i, img in enumerate(images):
                    output_file = output_path / f"{base_name}_processed{i}.png"
                    
                    # Convert to uint8 for saving
                    img_uint8 = (img * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_uint8)
                    img_pil.save(output_file)
                    
                    processed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process {img_file}: {e}")
        
        logger.info(f"Preprocessed {processed_count} images saved to {output_dir}")
        
        return processed_count


def main():
    """Main function to test image preprocessing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess medical images')
    parser.add_argument('--input', type=str, required=True, help='Input image file or directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--target-size', type=int, nargs=2, default=[224, 224],
                       help='Target size (width height)')
    parser.add_argument('--normalize', type=str, default='minmax',
                       choices=['minmax', 'zscore', 'percentile'])
    parser.add_argument('--denoise', type=str, default='gaussian',
                       choices=['gaussian', 'median', 'bilateral', 'non_local_means'])
    parser.add_argument('--enhance', type=str, default='clahe',
                       choices=['clahe', 'histogram_eq', 'gamma'])
    parser.add_argument('--augment', action='store_true', help='Apply data augmentation')
    parser.add_argument('--augmentations', type=str, nargs='+',
                       choices=['horizontal_flip', 'vertical_flip', 'rotation', 
                               'brightness', 'contrast', 'zoom', 'noise'],
                       help='Augmentation techniques to apply')
    
    args = parser.parse_args()
    
    preprocessor = ImagePreprocessor()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image preprocessing
        images = preprocessor.preprocess_pipeline(
            image=str(input_path),
            target_size=tuple(args.target_size),
            normalize_method=args.normalize,
            denoise_method=args.denoise,
            enhance_method=args.enhance,
            augmentations=args.augmentations if args.augment else None
        )
        
        # Save preprocessed images
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        base_name = input_path.stem
        for i, img in enumerate(images):
            output_file = output_path / f"{base_name}_processed{i}.png"
            img_uint8 = (img * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_uint8)
            img_pil.save(output_file)
            logger.info(f"Saved preprocessed image to {output_file}")
            
    elif input_path.is_dir():
        # Directory preprocessing
        processed_count = preprocessor.preprocess_directory(
            input_dir=str(input_path),
            output_dir=args.output,
            target_size=tuple(args.target_size),
            normalize_method=args.normalize,
            denoise_method=args.denoise,
            enhance_method=args.enhance,
            augment=args.augment,
            augmentations=args.augmentations
        )
        logger.info(f"Preprocessed {processed_count} images")
    
    else:
        logger.error(f"Input path does not exist: {args.input}")


if __name__ == '__main__':
    main()
