"""
Improved CT brain tumor training with better architecture and data augmentation
Target: 80%+ accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

class BrainTumorCTDataset(Dataset):
    """Dataset for brain tumor CT classification with improved loading"""
    
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        
        # Get all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_files.extend(list(self.images_dir.glob(ext)))
        
        print(f"Found {len(self.image_files)} images")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('L')  # Keep grayscale for CT
        
        # Get label from corresponding label file
        label_path = self.labels_dir / (img_path.stem + '.txt')
        
        if label_path.exists():
            # Read YOLO format label
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # If label file has content, check if any object is class 1 (positive/tumor)
            has_tumor = False
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 1:
                    class_id = int(parts[0])
                    if class_id == 1:  # positive class
                        has_tumor = True
                        break
            
            label = 1 if has_tumor else 0  # 1 = tumor, 0 = normal
        else:
            # If no label file, assume normal
            label = 0
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ImprovedCTCNN(nn.Module):
    """Custom CNN architecture optimized for CT brain tumor detection"""
    
    def __init__(self):
        super(ImprovedCTCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # Binary: Normal vs Tumor
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def create_data_loaders(data_root, batch_size=16):
    """Create train and validation data loaders with improved augmentation"""
    
    # Enhanced data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # CT-specific normalization
    ])
    
    # No augmentation for validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create datasets
    train_dataset = BrainTumorCTDataset(
        images_dir=f"{data_root}/images/train",
        labels_dir=f"{data_root}/labels/train",
        transform=train_transform
    )
    
    val_dataset = BrainTumorCTDataset(
        images_dir=f"{data_root}/images/val",
        labels_dir=f"{data_root}/labels/val",
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, train_dataset, val_dataset

def train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=0.001):
    """Train the model with improved training strategy"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss function with class weighting
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = 8  # Early stopping patience
    
    print(f"Training on {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"✓ New best model saved with validation accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Best Val Acc: {best_val_acc:.2f}%')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 50)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
        
        # Stop if we achieve 80%+ accuracy
        if val_acc >= 80.0:
            print(f"🎉 Target accuracy of 80% achieved! Stopping training.")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, train_accs, val_accs

def save_trained_model(model, save_path='models/ct/brain_tumor_improved.pth'):
    """Save the trained model"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': 'Improved CNN Brain Tumor Detection',
        'classes': ['normal', 'tumor']
    }, save_path)
    print(f"✓ Model saved to {save_path}")

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_improved.png')
    print("✓ Training history plot saved to training_history_improved.png")

def main():
    """Main training function"""
    
    # Configuration
    DATA_ROOT = "datasets/ct_scan/extracted"
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    
    print("="*60)
    print("Improved CT Brain Tumor Training")
    print("Target: 80%+ accuracy")
    print("="*60)
    
    # Create data loaders
    print("\nLoading datasets...")
    train_loader, val_loader, train_dataset, val_dataset = create_data_loaders(DATA_ROOT, BATCH_SIZE)
    
    # Analyze class distribution
    train_labels = [label for _, label in train_dataset]
    val_labels = [label for _, label in val_dataset]
    
    print(f"\nClass distribution:")
    print(f"  Train - Normal: {train_labels.count(0)}, Tumor: {train_labels.count(1)}")
    print(f"  Val - Normal: {val_labels.count(0)}, Tumor: {val_labels.count(1)}")
    
    # Create improved CNN model
    print("\nCreating improved CNN model...")
    model = ImprovedCTCNN()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train the model
    print("\nStarting training...")
    trained_model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE
    )
    
    # Save the trained model
    print("\nSaving trained model...")
    save_trained_model(trained_model)
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Final Validation Accuracy: {val_accs[-1]:.2f}%")
    print(f"Best Validation Accuracy: {max(val_accs):.2f}%")
    print("="*60)
    
    if max(val_accs) >= 80.0:
        print("🎉 SUCCESS: Achieved 80%+ accuracy target!")
    else:
        print(f"⚠ Accuracy: {max(val_accs):.2f}% - Further optimization needed")

if __name__ == "__main__":
    main()
