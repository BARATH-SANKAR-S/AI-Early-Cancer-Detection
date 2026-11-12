"""
Resume Training Script for Module 10
Resumes training from existing checkpoint
"""

import os
import sys
from pathlib import Path

# Set environment variable to fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from PIL import Image
import numpy as np
import json
import time
from typing import List, Tuple, Dict
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.vision.model_config import ViTModelConfig
from models.vision.vit_model import ViTBrainTumorClassifier, load_image_processor
from preprocessing.dataset_loader import MRIDatasetLoader
from preprocessing.dataset_split import DatasetSplitter
from preprocessing.image_preprocessing import ImagePreprocessor

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
            # Check if already in the right format for processor
            if image.ndim == 3 and image.shape[0] == 3:
                # Already in (C, H, W) format, convert to (H, W, C) for PIL
                image = np.transpose(image, (1, 2, 0))
            
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            
            # Ensure correct dtype
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            image = Image.fromarray(image)
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_accuracy: float):
        if self.best_score is None:
            self.best_score = val_accuracy
        elif val_accuracy < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_accuracy
            self.counter = 0


class ViTTrainer:
    """Trains Vision Transformer model with complete tracking"""
    
    def __init__(self, config: ViTModelConfig = ViTModelConfig):
        self.config = config
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config.DEVICE = str(self.device)
        
        # Set random seed for reproducibility
        torch.manual_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        
        # Load model
        self.model = ViTBrainTumorClassifier(config).to(self.device)
        
        # Load processor
        self.processor = load_image_processor(config)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'epoch_times': []
        }
        
        # Early stopping disabled to complete all epochs
        self.early_stopping = EarlyStopping(patience=100, min_delta=0.001)
        
        # Setup optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        
        # Starting epoch
        self.start_epoch = 0
        self.best_val_accuracy = 0
        
    def load_checkpoint(self, checkpoint_path: Path):
        """Load training checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load history from checkpoint if available
        if 'history' in checkpoint:
            self.history = checkpoint['history']
            logger.info(f"Loaded training history from checkpoint")
        
        # Also try to load from external training history file (may have more recent data)
        history_file = Path('models/vision/training_history.json')
        if history_file.exists():
            import json
            with open(history_file, 'r') as f:
                external_history = json.load(f)
            # Use the external history if it has more epochs
            if len(external_history.get('train_loss', [])) > len(self.history.get('train_loss', [])):
                self.history = external_history
                logger.info(f"Using more recent training history from external file")
        
        # Set starting epoch based on actual training history length
        if self.history and 'train_loss' in self.history:
            self.start_epoch = len(self.history['train_loss'])
            logger.info(f"Resuming from epoch {self.start_epoch} (based on training history)")
        elif 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resuming from epoch {self.start_epoch} (based on checkpoint)")
        
        # Set best validation accuracy
        if self.history and 'val_accuracy' in self.history and len(self.history['val_accuracy']) > 0:
            self.best_val_accuracy = max(self.history['val_accuracy'])
            self.early_stopping.best_score = self.best_val_accuracy
            logger.info(f"Best validation accuracy from history: {self.best_val_accuracy:.4f}")
        elif 'val_accuracy' in checkpoint:
            self.best_val_accuracy = checkpoint['val_accuracy']
            self.early_stopping.best_score = self.best_val_accuracy
            logger.info(f"Best validation accuracy from checkpoint: {self.best_val_accuracy:.4f}")
        
        return checkpoint
    
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        logger.info("Loading and preprocessing dataset...")
        
        # Load and split dataset
        loader = MRIDatasetLoader(self.config)
        splitter = DatasetSplitter(self.config)
        preprocessor = ImagePreprocessor(self.config)
        
        # Split dataset
        split_result = splitter.split_dataset()
        splits = split_result['splits']
        
        # Preprocess each split
        logger.info("Preprocessing training data...")
        train_images, train_labels, _ = preprocessor.preprocess_dataset(splits['train'], augment=True)
        
        logger.info("Preprocessing validation data...")
        val_images, val_labels, _ = preprocessor.preprocess_dataset(splits['val'], augment=False)
        
        logger.info(f"Training samples: {len(train_images)}")
        logger.info(f"Validation samples: {len(val_images)}")
        
        return train_images, train_labels, val_images, val_labels
    
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
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(progress_bar):
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
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
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
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
    
    def save_checkpoint(self, epoch: int, val_accuracy: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = self.config.MODEL_SAVE_DIR
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_accuracy': val_accuracy,
            'config': self.config.to_dict(),
            'history': self.history
        }
        
        if is_best:
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Best model saved to {checkpoint_path}")
        else:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def plot_training_history(self):
        """Plot and save training history"""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        axes[0, 0].plot(self.history['train_loss'], label='Training Loss', color='blue')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Validation loss
        axes[0, 1].plot(self.history['val_loss'], label='Validation Loss', color='red')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Training accuracy
        axes[1, 0].plot(self.history['train_accuracy'], label='Training Accuracy', color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Training Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Validation accuracy
        axes[1, 1].plot(self.history['val_accuracy'], label='Validation Accuracy', color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Validation Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save individual plots
        reports_dir = Path('reports')
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Training loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        plt.savefig(reports_dir / 'training_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['val_loss'], color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.grid(True)
        plt.savefig(reports_dir / 'validation_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Training accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_accuracy'], color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.grid(True)
        plt.savefig(reports_dir / 'training_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Validation accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['val_accuracy'], color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.grid(True)
        plt.savefig(reports_dir / 'validation_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Training graphs saved to reports/")
    
    def save_training_history(self):
        """Save training history to JSON"""
        history_path = Path('models/vision/training_history.json')
        history_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Training history saved to {history_path}")
    
    def train(self, resume_from_checkpoint: Path = None):
        """Complete training loop with resume capability"""
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"GPU Available: {torch.cuda.is_available()}")
        
        # Load checkpoint if provided
        if resume_from_checkpoint and resume_from_checkpoint.exists():
            self.load_checkpoint(resume_from_checkpoint)
        
        # Load and preprocess data
        train_images, train_labels, val_images, val_labels = self.load_and_preprocess_data()
        
        # Setup training
        self.setup_training(train_images, train_labels, val_images, val_labels)
        
        # Training loop
        best_epoch = self.start_epoch
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.config.NUM_EPOCHS):
            epoch_start = time.time()
            
            # Train
            train_loss, train_accuracy = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_accuracy = self.validate()
            
            epoch_time = time.time() - epoch_start
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_accuracy'].append(train_accuracy)
            self.history['val_accuracy'].append(val_accuracy)
            self.history['epoch_times'].append(epoch_time)
            
            logger.info(f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            logger.info(f"Epoch Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                best_epoch = epoch + 1
                self.save_checkpoint(epoch, val_accuracy, is_best=True)
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.SAVE_STEPS == 0:
                self.save_checkpoint(epoch, val_accuracy, is_best=False)
            
            # Early stopping
            self.early_stopping(val_accuracy)
            if self.early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        total_time = time.time() - start_time
        
        # Save final results
        self.save_training_history()
        self.plot_training_history()
        
        # Final report
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total Training Time: {total_time:.2f}s ({total_time/60:.2f}min)")
        logger.info(f"Best Validation Accuracy: {self.best_val_accuracy:.4f}")
        logger.info(f"Best Epoch: {best_epoch}")
        logger.info(f"Final Training Loss: {self.history['train_loss'][-1]:.4f}")
        logger.info(f"Final Validation Loss: {self.history['val_loss'][-1]:.4f}")
        logger.info(f"Model Save Location: {self.config.MODEL_SAVE_DIR / 'best_model.pth'}")
        logger.info("=" * 60)
        
        return {
            'total_time': total_time,
            'best_val_accuracy': self.best_val_accuracy,
            'best_epoch': best_epoch,
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'model_save_location': str(self.config.MODEL_SAVE_DIR / 'best_model.pth')
        }


def main():
    """Main training function"""
    config = ViTModelConfig()
    trainer = ViTTrainer(config)
    
    # Resume from existing checkpoint
    checkpoint_path = Path('models/vision/checkpoints/best_model.pth')
    
    # Train
    results = trainer.train(resume_from_checkpoint=checkpoint_path)
    
    # Print final results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Training Time: {results['total_time']:.2f}s ({results['total_time']/60:.2f}min)")
    print(f"Best Validation Accuracy: {results['best_val_accuracy']:.4f}")
    print(f"Best Epoch: {results['best_epoch']}")
    print(f"Final Training Loss: {results['final_train_loss']:.4f}")
    print(f"Final Validation Loss: {results['final_val_loss']:.4f}")
    print(f"Model Save Location: {results['model_save_location']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
