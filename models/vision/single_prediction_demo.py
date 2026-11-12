"""
Single MRI Prediction Demo for Module 11
Demonstrates real prediction on one test image from the Testing dataset
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import torch
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.vision.model_config import ViTModelConfig
from models.vision.vit_model import ViTBrainTumorClassifier, load_image_processor


def run_single_prediction_demo():
    """Run prediction on a single test image"""
    
    print("=" * 70)
    print("SINGLE MRI PREDICTION DEMO")
    print("=" * 70)
    
    # Select test image from Testing dataset
    test_image_path = Path('Testing/notumor/Te-no_1.jpg')
    actual_class = 'notumor'
    
    print(f"\nSELECTED TEST IMAGE:")
    print(f"Image Path: {test_image_path.absolute()}")
    print(f"Actual Class: {actual_class}")
    print(f"Image Exists: {test_image_path.exists()}")
    
    if not test_image_path.exists():
        print(f"ERROR: Test image not found at {test_image_path}")
        return
    
    # Load configuration
    config = ViTModelConfig()
    
    # Load model
    print(f"\nLOADING MODEL:")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Checkpoint: models/vision/checkpoints/best_model.pth")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViTBrainTumorClassifier(config).to(device)
    
    # Load checkpoint
    checkpoint_path = Path('models/vision/checkpoints/best_model.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Model loaded successfully")
    
    # Load image processor
    processor = load_image_processor(config)
    print(f"✓ Image processor loaded")
    
    # Load and preprocess image
    print(f"\nPREPROCESSING IMAGE:")
    image = Image.open(test_image_path)
    print(f"Original size: {image.size}")
    print(f"Mode: {image.mode}")
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
        print(f"Converted to RGB")
    
    # Process image
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    print(f"Processed shape: {pixel_values.shape}")
    
    # Run inference
    print(f"\nRUNNING INFERENCE:")
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class_idx].item()
    
    # Get class names
    predicted_class = config.INDEX_TO_CLASS[predicted_class_idx]
    
    # Display results
    print(f"\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    
    print(f"\nActual Class: {actual_class}")
    print(f"AI Predicted Class: {predicted_class}")
    print(f"Model Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    
    print(f"\nClass Probabilities:")
    for idx, class_name in enumerate(config.CLASS_NAMES):
        prob = probabilities[0][idx].item()
        print(f"- {class_name}: {prob:.4f} ({prob*100:.2f}%)")
    
    print(f"\n" + "=" * 70)
    print("IMPORTANT DISCLAIMER")
    print("=" * 70)
    print(f"This is an academic AI demonstration, not a medical diagnosis.")
    print(f"Doctor Review Required")
    print("=" * 70)
    
    return {
        'actual_class': actual_class,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': {config.CLASS_NAMES[i]: probabilities[0][i].item() for i in range(len(config.CLASS_NAMES))}
    }


if __name__ == '__main__':
    run_single_prediction_demo()
