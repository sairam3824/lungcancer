#!/usr/bin/env python3
"""
Comprehensive Model Analysis Script
Evaluates accuracy, precision, recall, F1-score, and confusion matrix
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

# Import model architecture from run_lung_cancer_model.py
from run_lung_cancer_model import CSPDarkNetSmall

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ============================================================
# Load Model and Data
# ============================================================

def load_model(model_path, num_classes=3):
    """Load the trained model"""
    model = CSPDarkNetSmall(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def load_dataset(data_dir, batch_size=16):
    """Load the dataset for evaluation"""
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = datasets.ImageFolder(data_dir, transform=eval_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return loader, dataset

# ============================================================
# Evaluation Functions
# ============================================================

def evaluate_model(model, data_loader):
    """Evaluate model and return predictions and labels"""
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def calculate_metrics(y_true, y_pred, class_names):
    """Calculate comprehensive metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class
    }

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix saved to {save_path}")
    
    return cm

def plot_class_distribution(dataset, save_path='class_distribution.png'):
    """Plot class distribution"""
    class_counts = {}
    for _, label in dataset.samples:
        class_name = dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_counts.keys(), class_counts.values(), color=['#3498db', '#e74c3c', '#2ecc71'])
    plt.title('Class Distribution in Dataset', fontsize=16, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Class distribution saved to {save_path}")
    
    return class_counts

def plot_per_class_metrics(metrics, class_names, save_path='per_class_metrics.png'):
    """Plot per-class metrics"""
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, metrics['precision_per_class'], width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, metrics['recall_per_class'], width, label='Recall', color='#e74c3c')
    bars3 = ax.bar(x + width, metrics['f1_per_class'], width, label='F1-Score', color='#2ecc71')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Per-class metrics saved to {save_path}")

# ============================================================
# Main Analysis
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🔬 COMPREHENSIVE MODEL ANALYSIS")
    print("=" * 70)
    
    # Check if model exists
    model_path = 'best_lung_cancer_model.pth'
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found!")
        exit(1)
    
    # Check if dataset exists
    data_dir = 'dataset'
    if not os.path.exists(data_dir):
        print(f"❌ Error: Dataset directory '{data_dir}' not found!")
        exit(1)
    
    # Load model and data
    print("\n📂 Loading model and dataset...")
    data_loader, dataset = load_dataset(data_dir)
    model = load_model(model_path, num_classes=len(dataset.classes))
    
    print(f"✅ Model loaded from {model_path}")
    print(f"✅ Dataset loaded: {len(dataset)} images")
    print(f"✅ Classes: {dataset.classes}\n")
    
    # Evaluate model
    print("🔄 Evaluating model...")
    y_pred, y_true, y_probs = evaluate_model(model, data_loader)
    
    # Calculate metrics
    print("\n📊 Calculating metrics...")
    metrics = calculate_metrics(y_true, y_pred, dataset.classes)
    
    # Print overall metrics
    print("\n" + "=" * 70)
    print("📈 OVERALL PERFORMANCE METRICS")
    print("=" * 70)
    print(f"🎯 Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"🎯 Precision: {metrics['precision']*100:.2f}%")
    print(f"🎯 Recall:    {metrics['recall']*100:.2f}%")
    print(f"🎯 F1-Score:  {metrics['f1_score']*100:.2f}%")
    
    # Print per-class metrics
    print("\n" + "=" * 70)
    print("📊 PER-CLASS PERFORMANCE METRICS")
    print("=" * 70)
    for i, class_name in enumerate(dataset.classes):
        print(f"\n{class_name}:")
        print(f"  Precision: {metrics['precision_per_class'][i]*100:.2f}%")
        print(f"  Recall:    {metrics['recall_per_class'][i]*100:.2f}%")
        print(f"  F1-Score:  {metrics['f1_per_class'][i]*100:.2f}%")
    
    # Classification report
    print("\n" + "=" * 70)
    print("📋 DETAILED CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(y_true, y_pred, target_names=dataset.classes))
    
    # Plot visualizations
    print("\n📊 Generating visualizations...")
    cm = plot_confusion_matrix(y_true, y_pred, dataset.classes)
    class_counts = plot_class_distribution(dataset)
    plot_per_class_metrics(metrics, dataset.classes)
    
    # Save detailed report
    print("\n💾 Saving detailed report...")
    with open('model_analysis_report.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("LUNG CANCER CLASSIFICATION MODEL - ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("OVERALL PERFORMANCE METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Accuracy:  {metrics['accuracy']*100:.2f}%\n")
        f.write(f"Precision: {metrics['precision']*100:.2f}%\n")
        f.write(f"Recall:    {metrics['recall']*100:.2f}%\n")
        f.write(f"F1-Score:  {metrics['f1_score']*100:.2f}%\n\n")
        
        f.write("PER-CLASS PERFORMANCE METRICS\n")
        f.write("-" * 70 + "\n")
        for i, class_name in enumerate(dataset.classes):
            f.write(f"\n{class_name}:\n")
            f.write(f"  Precision: {metrics['precision_per_class'][i]*100:.2f}%\n")
            f.write(f"  Recall:    {metrics['recall_per_class'][i]*100:.2f}%\n")
            f.write(f"  F1-Score:  {metrics['f1_per_class'][i]*100:.2f}%\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(classification_report(y_true, y_pred, target_names=dataset.classes))
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("=" * 70 + "\n")
        f.write(str(cm) + "\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("CLASS DISTRIBUTION\n")
        f.write("=" * 70 + "\n")
        for class_name, count in class_counts.items():
            f.write(f"{class_name}: {count} samples\n")
    
    print("✅ Detailed report saved to 'model_analysis_report.txt'")
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  📄 model_analysis_report.txt")
    print("  📊 confusion_matrix.png")
    print("  📊 class_distribution.png")
    print("  📊 per_class_metrics.png")
