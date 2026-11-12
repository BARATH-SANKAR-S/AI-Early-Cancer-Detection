"""
Vision Transformer Model for Module 9
ViT model with adapted classification head for brain tumor classification
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.vision.model_config import ViTModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ViTBrainTumorClassifier(nn.Module):
    """Vision Transformer for brain tumor classification"""
    
    def __init__(self, config: ViTModelConfig = ViTModelConfig):
        super(ViTBrainTumorClassifier, self).__init__()
        self.config = config
        
        # Load pretrained ViT model from Hugging Face
        try:
            from transformers import ViTForImageClassification
            logger.info(f"Loading pretrained model: {config.MODEL_NAME}")
            
            # Load model with num_labels parameter to adapt classification head
            self.model = ViTForImageClassification.from_pretrained(
                config.MODEL_NAME,
                num_labels=config.NUM_CLASSES,
                ignore_mismatched_sizes=True  # Allow head replacement
            )
            
            logger.info("Model loaded successfully")
            
        except ImportError:
            logger.error("transformers library not installed. Install with: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Forward pass through the model
        
        Args:
            pixel_values: Input images [batch_size, channels, height, width]
            labels: Ground truth labels [batch_size]
            
        Returns:
            Model outputs
        """
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        return outputs
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.config.MODEL_NAME,
            'num_classes': self.config.NUM_CLASSES,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'image_size': self.config.IMAGE_SIZE,
            'device': str(next(self.parameters()).device)
        }
    
    def save_model(self, save_path: Optional[Path] = None):
        """Save model checkpoint"""
        if save_path is None:
            save_path = self.config.MODEL_SAVE_PATH
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(load_path, map_location=self.config.DEVICE)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {load_path}")


def load_image_processor(config: ViTModelConfig = ViTModelConfig):
    """
    Load the Hugging Face image processor for the model
    
    Args:
        config: Model configuration
        
    Returns:
        Image processor
    """
    try:
        from transformers import AutoImageProcessor
        logger.info(f"Loading image processor for: {config.MODEL_NAME}")
        
        processor = AutoImageProcessor.from_pretrained(config.MODEL_NAME)
        logger.info("Image processor loaded successfully")
        
        return processor
        
    except ImportError:
        logger.error("transformers library not installed")
        raise
    except Exception as e:
        logger.error(f"Error loading image processor: {e}")
        raise


def main():
    """Test the ViT model"""
    import torch
    
    # Initialize config
    config = ViTModelConfig()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.DEVICE = str(device)
    
    print("=" * 60)
    print("VISION TRANSFORMER MODEL INFORMATION")
    print("=" * 60)
    
    # Load model
    print(f"\n1. Selected ViT Model: {config.MODEL_NAME}")
    print(f"   Reason: Well-established, widely used, good balance of performance and efficiency")
    print(f"   Suitable for medical imaging with 224x224 input size")
    
    # Create model
    model = ViTBrainTumorClassifier(config).to(device)
    
    # Load image processor
    processor = load_image_processor(config)
    
    # Get model info
    info = model.get_model_info()
    
    print(f"\n2. Number of Classes: {info['num_classes']}")
    print(f"   Class Labels: {config.CLASS_NAMES}")
    print(f"   Class Mapping: {config.CLASS_TO_INDEX}")
    
    print(f"\n3. Input Image Size: {info['image_size']}")
    print(f"   Channels: {config.NUM_CHANNELS}")
    
    print(f"\n4. Trainable Parameters: {info['trainable_parameters']:,}")
    print(f"   Total Parameters: {info['total_parameters']:,}")
    print(f"   Non-Trainable: {info['non_trainable_parameters']:,}")
    
    print(f"\n5. Device: {info['device']}")
    print(f"   GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    
    # Test forward pass
    print(f"\n6. Testing Forward Pass...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    dummy_labels = torch.tensor([0]).to(device)
    
    with torch.no_grad():
        outputs = model(dummy_input, dummy_labels)
    
    print(f"   Forward pass successful!")
    print(f"   Output logits shape: {outputs.logits.shape}")
    print(f"   Loss: {outputs.loss.item() if outputs.loss is not None else 'N/A'}")
    
    print("\n" + "=" * 60)
    print("MODEL VERIFICATION COMPLETE")
    print("=" * 60)
    
    return model, processor, config


if __name__ == '__main__':
    model, processor, config = main()
