"""
Training Script for Module 9
Fine-tunes Vision Transformer on brain tumor MRI dataset
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from PIL import Image
import numpy as np
from typing import List, Tuple
import logging
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.vision.model_config import ViTModelConfig
from models.vision.vit_model import ViTBrainTumorClassifier, load_image_processor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MRIDataset(Dataset):
    """PyTorch Dataset for MRI images"""
    
    def __init__(self, images: np.ndarray, labels: np.ndarray, processor):
        self.images = images
        self.labels = labels
        self.processor = processor
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ViTTrainer:
    """Trains Vision Transformer model"""
    
    def __init__(self, config: ViTModelConfig = ViTModelConfig):
        self.config = config
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config.DEVICE = str(self.device)
        
        # Load model
        self.model = ViTBrainTumorClassifier(config).to(self.device)
        
        # Load processor
        self.processor = load_image_processor(config)
        
        # Setup optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        
    def setup_training(self, train_images: np.ndarray, train_labels: np.ndarray,
                      val_images: np.ndarray, val_labels: np.ndarray):
        """Setup training components"""
        # Create datasets
        train_dataset = MRIDataset(train_images, train_labels, self.processor)
        val_dataset = MRIDataset(val_images, val_labels, self.processor)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # Setup scheduler
        total_steps = len(self.train_loader) * self.config.NUM_EPOCHS
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
        
        logger.info(f"Training setup complete")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(progress_bar):
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if (batch_idx + 1) // self.config.LOGGING_STEPS == 0:
                logger.info(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_images: np.ndarray, train_labels: np.ndarray,
             val_images: np.ndarray, val_labels: np.ndarray):
        """Complete training loop"""
        logger.info("Starting training...")
        
        # Setup training
        self.setup_training(train_images, train_labels, val_images, val_labels)
        
        # Training loop
        best_val_accuracy = 0
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_accuracy = self.validate()
            
            logger.info(f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Val Accuracy: {val_accuracy:.4f}")
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.model.save_model()
                logger.info(f"New best model saved with accuracy: {best_val_accuracy:.4f}")
        
        logger.info("Training complete!")
        logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")


def main():
    """Main training function"""
    # Note: This is a placeholder for actual training
    # In a real scenario, you would load the preprocessed data
    # from the preprocessing module
    
    config = ViTModelConfig()
    trainer = ViTTrainer(config)
    
    logger.info("Training script ready")
    logger.info("To train, load preprocessed data from preprocessing module")
    logger.info("Then call trainer.train(train_images, train_labels, val_images, val_labels)")


if __name__ == '__main__':
    main()
