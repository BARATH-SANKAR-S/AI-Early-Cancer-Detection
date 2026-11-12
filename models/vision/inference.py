"""
Inference Script for Module 9
Performs inference on single images using trained ViT model
"""

import sys
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from typing import Union, Dict, List
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.vision.model_config import ViTModelConfig
from models.vision.vit_model import ViTBrainTumorClassifier, load_image_processor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ViTInference:
    """Inference class for Vision Transformer"""
    
    def __init__(self, config: ViTModelConfig = ViTModelConfig):
        self.config = config
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config.DEVICE = str(self.device)
        
        # Load model
        self.model = ViTBrainTumorClassifier(config).to(self.device)
        self.model.eval()
        
        # Load processor
        self.processor = load_image_processor(config)
    
    def load_trained_model(self, model_path: Path):
        """Load trained model checkpoint"""
        self.model.load_model(model_path)
        logger.info(f"Loaded trained model from {model_path}")
    
    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess single image for inference
        
        Args:
            image: Image path, PIL Image, or numpy array
            
        Returns:
            Preprocessed tensor
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        return pixel_values
    
    def predict(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Dict:
        """
        Make prediction on single image
        
        Args:
            image: Image path, PIL Image, or numpy array
            
        Returns:
            Prediction dictionary
        """
        # Preprocess
        pixel_values = self.preprocess_image(image)
        pixel_values = pixel_values.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get class name
        class_name = self.config.INDEX_TO_CLASS[predicted_class]
        
        # Get all class probabilities
        all_probabilities = {
            self.config.INDEX_TO_CLASS[i]: prob.item() 
            for i, prob in enumerate(probabilities[0])
        }
        
        return {
            'predicted_class': predicted_class,
            'predicted_class_name': class_name,
            'confidence': confidence,
            'all_probabilities': all_probabilities
        }
    
    def predict_batch(self, images: List[Union[str, Path, Image.Image, np.ndarray]]) -> List[Dict]:
        """
        Make predictions on batch of images
        
        Args:
            images: List of images
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        for image in images:
            pred = self.predict(image)
            predictions.append(pred)
        
        return predictions


def main():
    """Main inference function"""
    config = ViTModelConfig()
    inference = ViTInference(config)
    
    logger.info("Inference script ready")
    logger.info("To perform inference, load trained model and call inference.predict(image)")


if __name__ == '__main__':
    main()
