"""
Evaluation Script for Module 9
Evaluates trained Vision Transformer model
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict
import logging
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.vision.model_config import ViTModelConfig
from models.vision.vit_model import ViTBrainTumorClassifier, load_image_processor
from models.vision.train_vit import MRIDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ViTEvaluator:
    """Evaluates Vision Transformer model"""
    
    def __init__(self, config: ViTModelConfig = ViTModelConfig):
        self.config = config
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config.DEVICE = str(self.device)
        
        # Load model
        self.model = ViTBrainTumorClassifier(config).to(self.device)
        
        # Load processor
        self.processor = load_image_processor(config)
    
    def load_trained_model(self, model_path: Path):
        """Load trained model checkpoint"""
        self.model.load_model(model_path)
        logger.info(f"Loaded trained model from {model_path}")
    
    def evaluate(self, test_images: np.ndarray, test_labels: np.ndarray) -> Dict:
        """
        Evaluate model on test set
        
        Args:
            test_images: Test images
            test_labels: Test labels
            
        Returns:
            Evaluation metrics
        """
        # Create dataset and dataloader
        test_dataset = MRIDataset(test_images, test_labels, self.processor)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in test_loader:
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(test_loader)
        accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_predictions)
        
        # Classification report
        class_report = classification_report(
            all_labels, 
            all_predictions,
            target_names=self.config.CLASS_NAMES,
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def plot_confusion_matrix(self, conf_matrix: np.ndarray, save_path: str = 'reports/confusion_matrix.png'):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.config.CLASS_NAMES,
            yticklabels=self.config.CLASS_NAMES
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {save_path}")


def main():
    """Main evaluation function"""
    config = ViTModelConfig()
    evaluator = ViTEvaluator(config)
    
    logger.info("Evaluation script ready")
    logger.info("To evaluate, load trained model and test data")
    logger.info("Then call evaluator.evaluate(test_images, test_labels)")


if __name__ == '__main__':
    main()
