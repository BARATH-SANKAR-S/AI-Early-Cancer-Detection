"""
ViT Inference Service for Flask Application
Integrates trained Vision Transformer model for brain tumor prediction
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.vision.model_config import ViTModelConfig
from models.vision.vit_model import ViTBrainTumorClassifier, load_image_processor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ViTInferenceService:
    """Service for ViT model inference"""
    
    def __init__(self):
        self.config = ViTModelConfig()
        self.model = None
        self.processor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
    
    def _load_model(self):
        """Load the trained ViT model and processor"""
        try:
            logger.info("Loading ViT model...")
            self.model = ViTBrainTumorClassifier(self.config).to(self.device)
            
            # Load checkpoint
            checkpoint_path = Path('models/vision/checkpoints/best_model.pth')
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                logger.info("Model loaded successfully from checkpoint")
            else:
                logger.error("Checkpoint not found")
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            
            # Load processor
            self.processor = load_image_processor(self.config)
            logger.info("Image processor loaded")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, image_path):
        """
        Run prediction on an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Prediction results with class, confidence, and probabilities
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class_idx].item()
            
            # Get class names
            predicted_class = self.config.INDEX_TO_CLASS[predicted_class_idx]
            
            # Build results dictionary
            from app.services.brain_inference import get_brain_risk_level, BRAIN_CLASS_LABELS
            risk_level = get_brain_risk_level(predicted_class, confidence)
            label = BRAIN_CLASS_LABELS.get(predicted_class, predicted_class.capitalize())

            results = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'risk_level': risk_level,
                'label': label,
                'is_malignant': predicted_class in ('glioma', 'malignant'),
                'model_name': self.config.MODEL_NAME,
                'cancer_type': 'brain_tumor',
                'image_type': 'brain_mri',
                'class_probabilities': {
                    self.config.CLASS_NAMES[i]: probabilities[0][i].item() 
                    for i in range(len(self.config.CLASS_NAMES))
                }
            }
            
            logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.4f}, risk: {risk_level})")
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise


# Singleton instance
_vit_service = None

def get_vit_service():
    """Get the singleton ViT inference service instance"""
    global _vit_service
    if _vit_service is None:
        _vit_service = ViTInferenceService()
    return _vit_service


def predict_image(image_path):
    """
    Convenience function to predict on an image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict: Prediction results
    """
    service = get_vit_service()
    return service.predict(image_path)
