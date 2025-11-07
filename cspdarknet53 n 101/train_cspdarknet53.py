#!/usr/bin/env python3
"""
Lung Cancer Classification - CSPDarkNet53
Enhanced version with comprehensive metrics and model saving
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model Architecture: CSPDarkNet53
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels // 2, kernel_size=1, padding=0)
        self.conv2 = ConvBlock(channels // 2, channels, kernel_size=3)
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(CSPBlock, self).__init__()
        self.downsample = ConvBlock(in_channels, out_channels, kernel_size=3, stride=2)
        self.split_conv0 = ConvBlock(out_channels, out_channels // 2, kernel_size=1, padding=0)
        self.split_conv1 = ConvBlock(out_channels, out_channels // 2, kernel_size=1, padding=0)
        self.blocks = nn.Sequential(*[ResidualBlock(out_channels // 2) for _ in range(num_blocks)])
        self.concat_conv = ConvBlock(out_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        x = self.downsample(x)
        x1 = self.blocks(self.split_conv0(x))
        x2 = self.split_conv1(x)
        x = torch.cat((x1, x2), dim=1)
        return self.concat_conv(x)

class CSPDarkNet53(nn.Module):
    def __init__(self, num_classes=3):
        super(CSPDarkNet53, self).__init__()
        self.conv1 = ConvBlock(3, 32, 3)
        self.stage1 = CSPBlock(32, 64, num_blocks=1)
        self.stage2 = CSPBlock(64, 128, num_blocks=2)
        self.stage3 = CSPBlock(128, 256, num_blocks=8)  # Increased from 2 to 8
        self.stage4 = CSPBlock(256, 512, num_blocks=8)  # Increased from 1 to 8
        self.stage5 = CSPBlock(512, 1024, num_blocks=4) # New stage
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def load_data(data_dir, batch_size=16):
    """Load dataset with train/val split"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Load full dataset
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    
    # Split into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Update validation dataset transform
    val_dataset.dataset.transform = eval_transform
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"✅ Loaded {len(train_dataset)} training images, {len(val_dataset)} validation images")
    print(f"📸 Classes: {full_dataset.classes}")
    
    return train_loader, val_loader, full_dataset.classes, eval_transform

def calculate_metrics(y_true, y_pred, class_names):
    """Calculate comprehensive metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    for i, class_name in enumerate(class_names):
        metrics[f'precision_{class_name}'] = precision_per_class[i]
        metrics[f'recall_{class_name}'] = recall_per_class[i]
        metrics[f'f1_{class_name}'] = f1_per_class[i]
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - CSPDarkNet53')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Confusion matrix saved to {save_path}")

def plot_training_history(history, save_path):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss over Epochs')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_title('Accuracy over Epochs')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 Score
    axes[1, 0].plot(history['val_f1'], label='Val F1 Score', color='green')
    axes[1, 0].set_title('F1 Score over Epochs')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning Rate
    axes[1, 1].plot(history['lr'], label='Learning Rate', color='red')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('LR')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Training history saved to {save_path}")

def train_model(model, train_loader, val_loader, class_names, num_epochs=50):
    """Train the model with validation"""
    # Compute class weights
    class_counts = []
    for cls in class_names:
        cls_path = os.path.join('dataset', cls)
        if os.path.isdir(cls_path):
            count = len([x for x in os.listdir(cls_path) if x.endswith(('.jpg', '.png', '.jpeg'))])
            class_counts.append(count)
    
    total = sum(class_counts)
    class_weights = torch.tensor([total / c for c in class_counts], dtype=torch.float).to(device)
    print(f"⚖️ Class weights: {class_weights.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)
    
    best_val_acc = 0
    best_val_f1 = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'lr': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_acc = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        val_metrics = calculate_metrics(all_labels, all_preds, class_names)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_metrics['f1_macro'])
        history['lr'].append(scheduler.get_last_lr()[0])
        
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_metrics['f1_macro']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_metrics': val_metrics
            }, "cspdarknet53_best.pth")
        
        print(f"\n📘 Epoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"   Val F1: {val_metrics['f1_macro']:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 70)
    
    print(f"\n✅ Training Complete!")
    print(f"   Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"   Best Val F1 Score: {best_val_f1:.4f}")
    
    return model, history, all_labels, all_preds

def test_model(model, test_dir, eval_transform, class_names):
    """Test model on test images"""
    model.eval()
    test_images = []
    
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                test_images.append(os.path.join(root, file))
    
    print(f"\n🧪 Testing on {len(test_images)} images")
    results = []
    
    for img_path in test_images:
        input_image = Image.open(img_path).convert("RGB")
        input_tensor = eval_transform(input_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
        
        result = {
            'image': os.path.basename(img_path),
            'predicted_class': class_names[pred_class],
            'confidence': probs[0][pred_class].item() * 100,
            'all_probs': {class_names[i]: probs[0][i].item() * 100 for i in range(len(class_names))}
        }
        results.append(result)
        print(f"📸 {result['image']}: {result['predicted_class']} ({result['confidence']:.2f}%)")
    
    return results

if __name__ == "__main__":
    print("=" * 70)
    print(" Lung Cancer Classification - CSPDarkNet53")
    print("=" * 70)
    
    if not os.path.exists('dataset'):
        print("❌ Error: 'dataset' folder not found!")
        exit(1)
    
    # Create output directory
    os.makedirs('cspdarknet53_results', exist_ok=True)
    
    # Load data
    print("\n📂 Loading dataset...")
    train_loader, val_loader, class_names, eval_transform = load_data('dataset', batch_size=16)
    
    # Initialize model
    num_classes = len(class_names)
    model = CSPDarkNet53(num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🔧 CSPDarkNet53 initialized with {total_params:,} parameters")
    
    # Train model
    print("\n🚀 Starting training...")
    model, history, val_labels, val_preds = train_model(model, train_loader, val_loader, class_names, num_epochs=10)
    
    print("\n💾 Best model saved as 'cspdarknet53_best.pth'")
    
    # Generate plots
    plot_training_history(history, 'cspdarknet53_results/training_history.png')
    plot_confusion_matrix(val_labels, val_preds, class_names, 'cspdarknet53_results/confusion_matrix.png')
    
    # Calculate final metrics
    final_metrics = calculate_metrics(val_labels, val_preds, class_names)
    
    # Save metrics
    with open('cspdarknet53_results/metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=4)
    print("\n📊 Metrics saved to 'cspdarknet53_results/metrics.json'")
    
    # Generate classification report
    report = classification_report(val_labels, val_preds, target_names=class_names)
    with open('cspdarknet53_results/classification_report.txt', 'w') as f:
        f.write("CSPDarkNet53 Classification Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(report)
    print("\n📄 Classification report saved")
    
    # Test on test images
    if os.path.exists('test images'):
        print("\n" + "=" * 70)
        print(" Testing Phase")
        print("=" * 70)
        test_results = test_model(model, 'test images', eval_transform, class_names)
        
        with open('cspdarknet53_results/test_results.json', 'w') as f:
            json.dump(test_results, f, indent=4)
        print("\n💾 Test results saved")
    
    print("\n" + "=" * 70)
    print(" ✅ Process Complete!")
    print("=" * 70)
    print(f"\nFinal Validation Metrics:")
    print(f"  Accuracy: {final_metrics['accuracy']*100:.2f}%")
    print(f"  F1 Score (Macro): {final_metrics['f1_macro']:.4f}")
    print(f"  Precision (Macro): {final_metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro): {final_metrics['recall_macro']:.4f}")
