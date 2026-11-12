"""
DICOM Medical Imaging Service
Handles processing of DICOM (.dcm) medical image files.
Converts DICOM to standard image formats for AI processing.
"""

import logging
import os
import tempfile
from typing import Optional, Dict, Tuple
from datetime import datetime
import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)


class DICOMService:
    """
    Service for processing DICOM medical imaging files.
    Supports reading DICOM files, extracting metadata, and converting to standard formats.
    """
    
    def __init__(self):
        """Initialize the DICOM service"""
        self.pydicom_available = self._check_pydicom()
        if not self.pydicom_available:
            logger.warning("pydicom not available. DICOM support will be limited.")
    
    def _check_pydicom(self) -> bool:
        """Check if pydicom library is available"""
        try:
            import pydicom
            return True
        except ImportError:
            return False
    
    def is_dicom_file(self, file_path: str) -> bool:
        """
        Check if a file is a valid DICOM file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file is a valid DICOM file, False otherwise
        """
        if not self.pydicom_available:
            return False
        
        try:
            import pydicom
            
            # Try to read the file as DICOM
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            return True
            
        except Exception as e:
            logger.debug(f"File {file_path} is not a valid DICOM file: {e}")
            return False
    
    def read_dicom_file(self, file_path: str) -> Optional[Dict]:
        """
        Read DICOM file and extract metadata and image data.
        
        Args:
            file_path: Path to the DICOM file
            
        Returns:
            Dictionary containing metadata and image data, or None if failed
        """
        if not self.pydicom_available:
            logger.error("pydicom not available")
            return None
        
        try:
            import pydicom
            
            # Read DICOM file
            ds = pydicom.dcmread(file_path)
            
            # Extract metadata
            metadata = self._extract_metadata(ds)
            
            # Extract image data
            image_data = self._extract_image_data(ds)
            
            return {
                'metadata': metadata,
                'image_data': image_data,
                'dataset': ds
            }
            
        except Exception as e:
            logger.error(f"Error reading DICOM file {file_path}: {e}")
            return None
    
    def _extract_metadata(self, dicom_dataset) -> Dict:
        """
        Extract relevant metadata from DICOM dataset.
        
        Args:
            dicom_dataset: pydicom dataset object
            
        Returns:
            Dictionary of extracted metadata
        """
        metadata = {}
        
        try:
            # Patient information
            metadata['patient_id'] = self._safe_get(dicom_dataset, 'PatientID', 'Unknown')
            metadata['patient_name'] = str(self._safe_get(dicom_dataset, 'PatientName', 'Unknown'))
            metadata['patient_birth_date'] = str(self._safe_get(dicom_dataset, 'PatientBirthDate', 'Unknown'))
            metadata['patient_sex'] = self._safe_get(dicom_dataset, 'PatientSex', 'Unknown')
            
            # Study information
            metadata['study_date'] = str(self._safe_get(dicom_dataset, 'StudyDate', 'Unknown'))
            metadata['study_time'] = str(self._safe_get(dicom_dataset, 'StudyTime', 'Unknown'))
            metadata['study_description'] = self._safe_get(dicom_dataset, 'StudyDescription', 'Unknown')
            metadata['modality'] = self._safe_get(dicom_dataset, 'Modality', 'Unknown')
            
            # Series information
            metadata['series_description'] = self._safe_get(dicom_dataset, 'SeriesDescription', 'Unknown')
            metadata['series_number'] = self._safe_get(dicom_dataset, 'SeriesNumber', 'Unknown')
            
            # Image information
            metadata['image_position'] = self._safe_get(dicom_dataset, 'ImagePositionPatient', 'Unknown')
            metadata['image_orientation'] = self._safe_get(dicom_dataset, 'ImageOrientationPatient', 'Unknown')
            metadata['slice_thickness'] = self._safe_get(dicom_dataset, 'SliceThickness', 'Unknown')
            metadata['pixel_spacing'] = self._safe_get(dicom_dataset, 'PixelSpacing', 'Unknown')
            
            # Acquisition information
            metadata['acquisition_date'] = str(self._safe_get(dicom_dataset, 'AcquisitionDate', 'Unknown'))
            metadata['acquisition_time'] = str(self._safe_get(dicom_dataset, 'AcquisitionTime', 'Unknown'))
            
            # Institution information
            metadata['institution_name'] = self._safe_get(dicom_dataset, 'InstitutionName', 'Unknown')
            metadata['manufacturer'] = self._safe_get(dicom_dataset, 'Manufacturer', 'Unknown')
            metadata['model'] = self._safe_get(dicom_dataset, 'ManufacturerModelName', 'Unknown')
            
        except Exception as e:
            logger.warning(f"Error extracting DICOM metadata: {e}")
        
        return metadata
    
    def _safe_get(self, dataset, tag, default):
        """Safely get a DICOM tag value"""
        try:
            value = getattr(dataset, tag, None)
            if value is not None:
                return value
            return default
        except:
            return default
    
    def _extract_image_data(self, dicom_dataset) -> Optional[np.ndarray]:
        """
        Extract pixel data from DICOM dataset.
        
        Args:
            dicom_dataset: pydicom dataset object
            
        Returns:
            NumPy array of pixel data or None if failed
        """
        try:
            # Get pixel data
            pixel_array = dicom_dataset.pixel_array
            
            # Apply rescale slope and intercept if available
            if hasattr(dicom_dataset, 'RescaleSlope') and hasattr(dicom_dataset, 'RescaleIntercept'):
                pixel_array = pixel_array * dicom_dataset.RescaleSlope + dicom_dataset.RescaleIntercept
            
            return pixel_array
            
        except Exception as e:
            logger.error(f"Error extracting DICOM image data: {e}")
            return None
    
    def convert_to_standard_format(
        self,
        dicom_data: Dict,
        output_format: str = 'png',
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Convert DICOM image data to standard image format.
        
        Args:
            dicom_data: Dictionary containing DICOM data from read_dicom_file
            output_format: Output format (png, jpg, tiff)
            output_path: Optional output file path
            
        Returns:
            Path to converted image file or None if failed
        """
        if not dicom_data or 'image_data' not in dicom_data:
            logger.error("Invalid DICOM data provided")
            return None
        
        try:
            pixel_array = dicom_data['image_data']
            
            # Normalize pixel values to 0-255 range
            pixel_array = self._normalize_pixel_array(pixel_array)
            
            # Convert to PIL Image
            image = Image.fromarray(pixel_array.astype(np.uint8))
            
            # Convert to RGB if grayscale
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save to file
            if output_path is None:
                # Create temporary file
                temp_dir = tempfile.gettempdir()
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(temp_dir, f'dicom_converted_{timestamp}.{output_format}')
            
            image.save(output_path, format=output_format.upper())
            logger.info(f"DICOM converted to {output_format}: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error converting DICOM to standard format: {e}")
            return None
    
    def _normalize_pixel_array(self, pixel_array: np.ndarray) -> np.ndarray:
        """
        Normalize pixel array to 0-255 range for image conversion.
        
        Args:
            pixel_array: Input pixel array
            
        Returns:
            Normalized pixel array
        """
        # Handle different data types
        if pixel_array.dtype == np.float32 or pixel_array.dtype == np.float64:
            # Normalize float values
            min_val = np.min(pixel_array)
            max_val = np.max(pixel_array)
            if max_val > min_val:
                pixel_array = ((pixel_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
        elif pixel_array.dtype == np.int16 or pixel_array.dtype == np.int32:
            # Normalize signed integers
            min_val = np.min(pixel_array)
            max_val = np.max(pixel_array)
            if max_val > min_val:
                pixel_array = ((pixel_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
        else:
            # Already uint8 or similar
            if pixel_array.max() > 255:
                pixel_array = (pixel_array / pixel_array.max() * 255).astype(np.uint8)
        
        return pixel_array
    
    def get_thumbnail(
        self,
        dicom_data: Dict,
        size: Tuple[int, int] = (256, 256),
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate thumbnail from DICOM image.
        
        Args:
            dicom_data: Dictionary containing DICOM data
            size: Thumbnail size (width, height)
            output_path: Optional output file path
            
        Returns:
            Path to thumbnail file or None if failed
        """
        if not dicom_data or 'image_data' not in dicom_data:
            logger.error("Invalid DICOM data provided")
            return None
        
        try:
            # Convert to standard format first
            temp_path = self.convert_to_standard_format(dicom_data, 'png')
            if not temp_path:
                return None
            
            # Create thumbnail
            image = Image.open(temp_path)
            image.thumbnail(size)
            
            # Save thumbnail
            if output_path is None:
                temp_dir = tempfile.gettempdir()
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(temp_dir, f'dicom_thumbnail_{timestamp}.png')
            
            image.save(output_path, 'PNG')
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            logger.info(f"DICOM thumbnail created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating DICOM thumbnail: {e}")
            return None
    
    def get_image_preview(self, dicom_data: Dict) -> Optional[str]:
        """
        Get base64 encoded image preview for web display.
        
        Args:
            dicom_data: Dictionary containing DICOM data
            
        Returns:
            Base64 encoded image string or None if failed
        """
        try:
            # Convert to standard format
            temp_path = self.convert_to_standard_format(dicom_data, 'png')
            if not temp_path:
                return None
            
            # Read file and encode to base64
            with open(temp_path, 'rb') as f:
                image_data = f.read()
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Encode to base64
            import base64
            base64_string = base64.b64encode(image_data).decode('utf-8')
            
            return f"data:image/png;base64,{base64_string}"
            
        except Exception as e:
            logger.error(f"Error creating DICOM image preview: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if DICOM service is available"""
        return self.pydicom_available


# Singleton instance
_dicom_service = None

def get_dicom_service() -> DICOMService:
    """Get or create the DICOM service singleton"""
    global _dicom_service
    if _dicom_service is None:
        _dicom_service = DICOMService()
    return _dicom_service