"""
Improved CT Brain Tumor Detection Training with EfficientNet
Uses EfficientNet-B3, Focal Loss, and Test-Time Augmentation
Target: 75%+ accuracy
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from collections import Counter
import copy

# Configuration
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.0003
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATIENCE = 10  # Early stopping patience
NUM_TTA = 5  # Test-time augmentation

# Paths
DATA_DIR = Path("datasets/ct_scan/extracted/classification")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
MODEL_SAVE_PATH = "models/ct_brain_tumor_efficientnet.pth"

class CTDataset(Dataset):
    """Custom dataset for CT brain tumor images"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
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

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss

def get_data_loaders():
    """Create training and validation data loaders with improved augmentation"""
    
    # Enhanced training augmentation
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transform
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # TTA transforms
    tta_transforms = [
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ]
    
    train_dataset = CTDataset(TRAIN_DIR, transform=train_transform)
    val_dataset = CTDataset(VAL_DIR, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, train_dataset, val_dataset, tta_transforms

def create_model():
    """Create EfficientNet-B3 model with transfer learning"""
    
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    
    # Freeze early layers
    for param in list(model.parameters())[:-30]:
        param.requires_grad = False
    
    # Modify classifier for binary classification
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
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

def validate(model, val_loader, criterion, tta_transforms=None):
    """Validate the model with optional test-time augmentation"""
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            if tta_transforms:
                # Test-time augmentation
                predictions = []
                for transform in tta_transforms:
                    aug_images = torch.stack([transform(img.cpu()) for img in images])
                    aug_images = aug_images.to(DEVICE)
                    outputs = model(aug_images)
                    predictions.append(outputs)
                
                outputs = torch.mean(torch.stack(predictions), dim=0)
            else:
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
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_efficientnet.png')
    plt.close()
    print("Training history saved to training_history_efficientnet.png")

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
    plt.savefig('confusion_matrix_efficientnet.png')
    plt.close()
    print("Confusion matrix saved to confusion_matrix_efficientnet.png")

def main():
    """Main training function"""
    
    print("="*60)
    print("Improved CT Brain Tumor Detection Training")
    print("Using EfficientNet-B3 with Focal Loss")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Test-Time Augmentation: {NUM_TTA} transforms")
    print("="*60)
    
    Path("models").mkdir(exist_ok=True)
    
    print("\nLoading datasets...")
    train_loader, val_loader, train_dataset, val_dataset, tta_transforms = get_data_loaders()
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_labels = [label for _, label in train_dataset.samples]
    print(f"Training distribution: {Counter(train_labels)}")
    
    print("\nCreating model...")
    model = create_model()
    
    class_weights = calculate_class_weights(train_dataset)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Use Focal Loss
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )
    
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
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        val_loss, val_acc, val_labels, val_preds = validate(model, val_loader, criterion, tta_transforms)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✓ New best model saved! Val Acc: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    model.load_state_dict(best_model_wts)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(val_labels, val_preds)
    
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=['Normal', 'Tumor']))
    
    print("\n" + "="*60)
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print("="*60)
    
    return best_val_acc

if __name__ == "__main__":
    best_accuracy = main()
    
    if best_accuracy >= 75:
        print(f"\n🎉 SUCCESS! Achieved {best_accuracy:.2f}% accuracy (target: 75%)")
    else:
        print(f"\n⚠ Accuracy: {best_accuracy:.2f}% (target: 75%)")
