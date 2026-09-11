"""
Fine-tuning script for brain tumor detection models
Fine-tunes pre-trained models on your specific brain tumor datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

class BrainTumorDataset(Dataset):
    """Custom dataset for brain tumor images"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load images and labels
        # Assuming structure: data_dir/class_name/image.jpg
        classes = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        for class_name in classes:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])
        
        print(f"Loaded {len(self.images)} images from {data_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loaders(train_dir, val_dir, batch_size=16):
    """Create data loaders for training and validation"""
    
    # Data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = BrainTumorDataset(train_dir, transform=train_transform)
    val_dataset = BrainTumorDataset(val_dir, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, len(train_dataset.classes)

def fine_tune_model(model, train_loader, val_loader, num_classes, num_epochs=10, learning_rate=0.001):
    """Fine-tune a pre-trained model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Replace the final layer for your specific number of classes
    if hasattr(model, 'fc'):
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif hasattr(model, 'classifier'):
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    print(f"Starting fine-tuning for {num_epochs} epochs...")
    
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
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            train_pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                val_pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
            print(f'Saved best model with val_acc: {val_acc:.2f}%')
    
    return model, history

def main():
    """Main fine-tuning function"""
    
    # Configuration
    TRAIN_DIR = 'Training'  # Your existing brain tumor dataset
    VAL_DIR = 'Testing'     # Your existing brain tumor dataset
    NUM_EPOCHS = 10
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    
    print("Brain Tumor Model Fine-Tuning")
    print("=" * 50)
    
    # Check if directories exist
    if not os.path.exists(TRAIN_DIR):
        print(f"Training directory not found: {TRAIN_DIR}")
        print("Please update TRAIN_DIR with your actual dataset path")
        return
    
    # Create data loaders
    print("Loading datasets...")
    train_loader, val_loader, num_classes = get_data_loaders(TRAIN_DIR, VAL_DIR, BATCH_SIZE)
    print(f"Number of classes: {num_classes}")
    
    # Load pre-trained model (ResNet50 as example)
    print("Loading pre-trained model...")
    from torchvision import models
    model = models.resnet50(pretrained=True)
    
    # Fine-tune the model
    print("Starting fine-tuning...")
    model, history = fine_tune_model(
        model, 
        train_loader, 
        val_loader, 
        num_classes, 
        NUM_EPOCHS, 
        LEARNING_RATE
    )
    
    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("Fine-tuning completed!")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    print("Model saved as 'best_model.pth'")

if __name__ == '__main__':
    main()
