"""
CT scan inference using pre-trained models
Uses ResNet50 for brain tumor detection with transfer learning
"""

import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import numpy as np
import os
import torchvision.transforms as transforms

class CTInference:
    """CT scan analysis using pre-trained ResNet50 for brain tumor detection"""
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = "ResNet50 (Transfer Learning)"
        self.load_model()
    
    def load_model(self):
        """Load trained ResNet50 model for brain tumor detection"""
        try:
            print(f"Loading CT brain tumor model: {self.model_name}")
            
            # Load pre-trained ResNet50
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            
            # Modify the final layer for binary classification (tumor vs normal)
            # This must match the training architecture exactly
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 2)  # Binary: Normal vs Tumor
            )
            
            # Load trained weights if available
            model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'ct_brain_tumor_advanced.pth')
            if os.path.exists(model_path):
                print(f"Loading trained weights from: {model_path}")
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print("Trained weights loaded successfully")
            else:
                print(f"Warning: Trained model not found at {model_path}")
                print("Using pre-trained ResNet50 without fine-tuning")
            
            self.model.to(self.device)
            self.model.eval()
            print("CT brain tumor model loaded successfully")
            
        except Exception as e:
            print(f"Error loading CT model: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Falling back to heuristic-based analysis")
            self.model = None
    
    def preprocess_image(self, image_path):
        """Preprocess CT image for model input"""
        try:
            # Load image directly as RGB (matching training preprocessing)
            img = Image.open(image_path).convert('RGB')
            
            # Apply transforms for ResNet (matching training)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            return img_tensor
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict(self, image_path):
        """
        Predict CT findings
        Args:
            image_path: Path to CT image
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            return {
                'error': 'Model not loaded',
                'predicted_class': 'error',
                'confidence': 0.0,
                'model_name': self.model_name,
                'image_type': 'ct'
            }
        
        try:
            # Preprocess image
            img_tensor = self.preprocess_image(image_path)
            if img_tensor is None:
                return {
                    'error': 'Image preprocessing failed',
                    'predicted_class': 'error',
                    'confidence': 0.0,
                    'model_name': self.model_name,
                    'image_type': 'ct'
                }
            
            # Make prediction using ResNet50
            with torch.no_grad():
                outputs = self.model(img_tensor)
                
                # Get probabilities using softmax
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
                
                # Map to classes: 0 = normal, 1 = tumor
                classes = ['normal', 'tumor']
                predicted_class = classes[predicted.item()]
                
                # Convert confidence to percentage
                confidence_score = confidence.item()
            
            # Create results dictionary for binary classification
            results = {
                'predicted_class': predicted_class,
                'confidence': confidence_score,
                'model_name': self.model_name,
                'image_type': 'ct',
                'class_probabilities': {
                    'normal': probs[0][0].item(),
                    'tumor': probs[0][1].item()
                }
            }
            
            return results
            
        except Exception as e:
            return {
                'error': str(e),
                'predicted_class': 'error',
                'confidence': 0.0,
                'model_name': self.model_name,
                'image_type': 'ct'
            }

def predict_ct(image_path):
    """Convenience function for CT prediction"""
    detector = CTInference()
    return detector.predict(image_path)
