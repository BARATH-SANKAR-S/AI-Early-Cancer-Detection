"""
Advanced CT Brain Tumor Detection Training
Uses ResNet50 with transfer learning, advanced augmentation, and optimized training
Target: 80%+ accuracy
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from collections import Counter
import copy

# Configuration
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATIENCE = 10  # Early stopping patience

# Paths
DATA_DIR = Path("datasets/ct_scan/extracted/classification")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
MODEL_SAVE_PATH = "models/ct_brain_tumor_advanced.pth"

class CTDataset(Dataset):
    """Custom dataset for CT brain tumor images"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        # Load all images and their labels
        for class_name in ['normal', 'tumor']:
            class_dir = self.root_dir / class_name
            label = 0 if class_name == 'normal' else 1
            
            for img_path in class_dir.glob('*.jpg'):
                self.samples.append((str(img_path), label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loaders():
    """Create training and validation data loaders with advanced augmentation"""
    
    # Advanced training augmentation
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = CTDataset(TRAIN_DIR, transform=train_transform)
    val_dataset = CTDataset(VAL_DIR, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, train_dataset, val_dataset

def create_model():
    """Create ResNet50 model with transfer learning"""
    
    # Load pretrained ResNet50
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    # Freeze early layers
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False
    
    # Modify final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    
    return model.to(DEVICE)

def calculate_class_weights(train_dataset):
    """Calculate class weights for balanced training"""
    
    labels = [label for _, label in train_dataset.samples]
    label_counts = Counter(labels)
    
    total = len(labels)
    num_classes = 2
    
    class_weights = []
    for i in range(num_classes):
        weight = total / (num_classes * label_counts[i])
        class_weights.append(weight)
    
    return torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

def train_epoch(model, train_loader, criterion, optimizer, scheduler):
    """Train for one epoch"""
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    scheduler.step()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion):
    """Validate the model"""
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_labels, all_predictions

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot training history"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_advanced.png')
    plt.close()
    print("Training history saved to training_history_advanced.png")

def plot_confusion_matrix(labels, predictions):
    """Plot confusion matrix"""
    
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Tumor'],
                yticklabels=['Normal', 'Tumor'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_advanced.png')
    plt.close()
    print("Confusion matrix saved to confusion_matrix_advanced.png")

def main():
    """Main training function"""
    
    print("="*60)
    print("Advanced CT Brain Tumor Detection Training")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print("="*60)
    
    # Create model save directory
    Path("models").mkdir(exist_ok=True)
    
    # Get data loaders
    print("\nLoading datasets...")
    train_loader, val_loader, train_dataset, val_dataset = get_data_loaders()
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Calculate class distribution
    train_labels = [label for _, label in train_dataset.samples]
    print(f"Training distribution: {Counter(train_labels)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model()
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_dataset)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    
    print("\nStarting training...")
    print("="*60)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, val_labels, val_preds = validate(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✓ New best model saved! Val Acc: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    model.load_state_dict(best_model_wts)
    
    # Final evaluation
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # Plot confusion matrix
    plot_confusion_matrix(val_labels, val_preds)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=['Normal', 'Tumor']))
    
    print("\n" + "="*60)
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print("="*60)
    
    return best_val_acc

if __name__ == "__main__":
    best_accuracy = main()
    
    if best_accuracy >= 80:
        print(f"\n🎉 SUCCESS! Achieved {best_accuracy:.2f}% accuracy (target: 80%)")
    else:
        print(f"\n⚠ Accuracy: {best_accuracy:.2f}% (target: 80%)")
