"""
X-ray image inference using pre-trained models
Uses torchxrayvision for chest X-ray analysis
"""

import torch
import torchxrayvision as xrv
from PIL import Image
import numpy as np
import os

class XRayInference:
    """X-ray image analysis using pre-trained models"""
    
    def __init__(self):
        self.model = None
        self.model_name = "densenet121-res224-nih"
        self.load_model()
    
    def load_model(self):
        """Load pre-trained X-ray model"""
        try:
            print(f"Loading X-ray model: {self.model_name}")
            try:
                self.model = xrv.models.get_model(self.model_name)
            except Exception:
                self.model = xrv.models.DenseNet(weights=self.model_name)
            self.model.eval()
            print("X-ray model loaded successfully")
        except Exception as e:
            print(f"Error loading X-ray model: {str(e)}")
            self.model = None
    
    def preprocess_image(self, image_path):
        """Preprocess X-ray image for model input"""
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            
            # Normalize to 0-1 range
            if img_array.max() > 1:
                img_array = img_array / 255.0
            
            # Convert to tensor
            img_tensor = torch.from_numpy(img_array).float()
            
            # Add batch dimension and channel dimension if needed
            if len(img_tensor.shape) == 3:
                img_tensor = img_tensor.permute(2, 0, 1)  # HWC to CHW
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
            
            return img_tensor
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None
    
    def predict(self, image_path):
        """
        Predict X-ray findings
        Args:
            image_path: Path to X-ray image
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            return {
                'error': 'Model not loaded',
                'predicted_class': 'error',
                'confidence': 0.0,
                'model_name': self.model_name
            }
        
        try:
            # Preprocess image
            img_tensor = self.preprocess_image(image_path)
            if img_tensor is None:
                return {
                    'error': 'Image preprocessing failed',
                    'predicted_class': 'error',
                    'confidence': 0.0,
                    'model_name': self.model_name
                }
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top prediction
            confidence, predicted = torch.max(probabilities, 1)
            
            # Get class labels (torchxrayvision provides default labels)
            pathologies = self.model.pathologies
            
            # Create results dictionary
            results = {
                'predicted_class': pathologies[predicted.item()],
                'confidence': confidence.item(),
                'model_name': self.model_name,
                'image_type': 'xray',
                'class_probabilities': {}
            }
            
            # Add all class probabilities
            for i, pathology in enumerate(pathologies):
                results['class_probabilities'][pathology] = probabilities[0][i].item()
            
            return results
            
        except Exception as e:
            return {
                'error': str(e),
                'predicted_class': 'error',
                'confidence': 0.0,
                'model_name': self.model_name,
                'image_type': 'xray'
            }

def predict_xray(image_path):
    """Convenience function for X-ray prediction"""
    detector = XRayInference()
    return detector.predict(image_path)
