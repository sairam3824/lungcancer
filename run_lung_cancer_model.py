#!/usr/bin/env python3
"""
Lung Cancer Classification Model - CSPDarkNet
Adapted from Colab notebook to run locally
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
import warnings
warnings.filterwarnings("ignore")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# Model Architecture: CSPDarkNet
# ============================================================

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

class CSPDarkNetSmall(nn.Module):
    def __init__(self, num_classes=3):
        super(CSPDarkNetSmall, self).__init__()
        self.conv1 = ConvBlock(3, 32, 3)
        self.stage1 = CSPBlock(32, 64, num_blocks=1)
        self.stage2 = CSPBlock(64, 128, num_blocks=2)
        self.stage3 = CSPBlock(128, 256, num_blocks=2)
        self.stage4 = CSPBlock(256, 512, num_blocks=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CSPDarkNet53(nn.Module):
    def __init__(self, num_classes=3):
        super(CSPDarkNet53, self).__init__()
        self.conv1 = ConvBlock(3, 32, 3)
        self.stage1 = CSPBlock(32, 64, num_blocks=1)
        self.stage2 = CSPBlock(64, 128, num_blocks=2)
        self.stage3 = CSPBlock(128, 256, num_blocks=8)
        self.stage4 = CSPBlock(256, 512, num_blocks=8)
        self.stage5 = CSPBlock(512, 1024, num_blocks=4)
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

# ============================================================
# Data Loading
# ============================================================

def load_data(data_dir):
    """Load dataset from local directory"""
    
    # Define transformations
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
    
    # Load datasets
    train_data = datasets.ImageFolder(data_dir, transform=train_transform)
    
    # Create DataLoader
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    
    print(f"✅ Loaded {len(train_data)} training images")
    print(f"📸 Classes found: {train_data.classes}")
    
    return train_loader, train_data, eval_transform

# ============================================================
# Training Function
# ============================================================

def train_model(model, train_loader, num_epochs=30):
    """Train the model"""
    
    # Compute class weights
    class_counts = [len([x for x in os.listdir(os.path.join('dataset', cls)) if x.endswith(('.jpg', '.png'))]) 
                    for cls in os.listdir('dataset') if os.path.isdir(os.path.join('dataset', cls))]
    total = sum(class_counts)
    class_weights = torch.tensor([total / c for c in class_counts], dtype=torch.float).to(device)
    
    print(f"⚖️ Class weights: {class_weights.cpu().numpy()}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    best_train_acc = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]"):
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
        
        scheduler.step()
        
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save(model.state_dict(), "best_lung_cancer_model.pth")
        
        print(f"\n📘 Epoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"   🔹 Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   🔹 LR: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 55)
    
    print(f"\n🎯 Training Complete! Best Training Accuracy: {best_train_acc:.2f}%")
    return model

# ============================================================
# Testing Function
# ============================================================

def test_model(model, test_dir, eval_transform, class_names):
    """Test the model on test images"""
    
    model.eval()
    
    # Get all test images
    test_images = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                test_images.append(os.path.join(root, file))
    
    print(f"\n🧪 Testing on {len(test_images)} images from {test_dir}")
    
    results = []
    
    for img_path in test_images:
        # Load and preprocess image
        input_image = Image.open(img_path).convert("RGB")
        input_tensor = eval_transform(input_image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
        
        result = {
            'image': os.path.basename(img_path),
            'predicted_class': class_names[pred_class],
            'confidence': probs[0][pred_class].item() * 100
        }
        results.append(result)
        
        print(f"📸 {result['image']}: {result['predicted_class']} ({result['confidence']:.2f}%)")
    
    return results

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🫁 Lung Cancer Classification Model - CSPDarkNet")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists('dataset'):
        print("❌ Error: 'dataset' folder not found!")
        exit(1)
    
    # Load data
    print("\n📂 Loading dataset...")
    train_loader, train_data, eval_transform = load_data('dataset')
    
    # Initialize model
    num_classes = len(train_data.classes)
    model = CSPDarkNet53(num_classes=num_classes).to(device)
    print(f"\n✅ CSPDarkNet53 model created for {num_classes} classes")
    
    # Train model
    print("\n🚀 Starting training...")
    model = train_model(model, train_loader, num_epochs=30)
    
    # Test on test images
    if os.path.exists('test images'):
        print("\n" + "=" * 60)
        print("🧪 Testing on images from 'test images' folder")
        print("=" * 60)
        results = test_model(model, 'test images', eval_transform, train_data.classes)
        
        # Save results
        with open('test_results.txt', 'w') as f:
            f.write("Lung Cancer Classification Results\n")
            f.write("=" * 60 + "\n\n")
            for r in results:
                f.write(f"{r['image']}: {r['predicted_class']} ({r['confidence']:.2f}%)\n")
        
        print("\n✅ Results saved to 'test_results.txt'")
    else:
        print("\n⚠️ 'test images' folder not found. Skipping testing phase.")
    
    print("\n" + "=" * 60)
    print("✅ Process completed!")
    print("=" * 60)
