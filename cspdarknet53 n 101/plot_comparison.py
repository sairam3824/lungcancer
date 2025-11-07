#!/usr/bin/env python3
"""
Model Comparison Visualization
Generate comprehensive comparison graphs for CSPDarkNet101 vs CSPDarkNet53
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

# Load metrics
with open('cspdarknet101_results/metrics.json', 'r') as f:
    metrics_101 = json.load(f)

with open('cspdarknet53_results/metrics.json', 'r') as f:
    metrics_53 = json.load(f)

# Create figure with subplots
fig = plt.figure(figsize=(18, 12))

# 1. Overall Metrics Comparison (Bar Chart)
ax1 = plt.subplot(2, 3, 1)
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
cspdarknet101_scores = [
    metrics_101['accuracy'],
    metrics_101['precision_macro'],
    metrics_101['recall_macro'],
    metrics_101['f1_macro']
]
cspdarknet53_scores = [
    metrics_53['accuracy'],
    metrics_53['precision_macro'],
    metrics_53['recall_macro'],
    metrics_53['f1_macro']
]

x = np.arange(len(metrics_names))
width = 0.35

bars1 = ax1.bar(x - width/2, cspdarknet101_scores, width, label='CSPDarkNet101', color='#2E86AB', alpha=0.8)
bars2 = ax1.bar(x + width/2, cspdarknet53_scores, width, label='CSPDarkNet53', color='#A23B72', alpha=0.8)

ax1.set_ylabel('Score')
ax1.set_title('Overall Performance Comparison', fontweight='bold', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics_names)
ax1.legend()
ax1.set_ylim([0, 1.1])
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# 2. Per-Class Precision Comparison
ax2 = plt.subplot(2, 3, 2)
classes = ['Benign', 'Malignant', 'Normal']
precision_101 = [
    metrics_101['precision_Bengin cases'],
    metrics_101['precision_Malignant cases'],
    metrics_101['precision_Normal cases']
]
precision_53 = [
    metrics_53['precision_Bengin cases'],
    metrics_53['precision_Malignant cases'],
    metrics_53['precision_Normal cases']
]

x = np.arange(len(classes))
bars1 = ax2.bar(x - width/2, precision_101, width, label='CSPDarkNet101', color='#2E86AB', alpha=0.8)
bars2 = ax2.bar(x + width/2, precision_53, width, label='CSPDarkNet53', color='#A23B72', alpha=0.8)

ax2.set_ylabel('Precision')
ax2.set_title('Per-Class Precision Comparison', fontweight='bold', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(classes)
ax2.legend()
ax2.set_ylim([0, 1.1])
ax2.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

# 3. Per-Class Recall Comparison
ax3 = plt.subplot(2, 3, 3)
recall_101 = [
    metrics_101['recall_Bengin cases'],
    metrics_101['recall_Malignant cases'],
    metrics_101['recall_Normal cases']
]
recall_53 = [
    metrics_53['recall_Bengin cases'],
    metrics_53['recall_Malignant cases'],
    metrics_53['recall_Normal cases']
]

bars1 = ax3.bar(x - width/2, recall_101, width, label='CSPDarkNet101', color='#2E86AB', alpha=0.8)
bars2 = ax3.bar(x + width/2, recall_53, width, label='CSPDarkNet53', color='#A23B72', alpha=0.8)

ax3.set_ylabel('Recall')
ax3.set_title('Per-Class Recall Comparison', fontweight='bold', fontsize=12)
ax3.set_xticks(x)
ax3.set_xticklabels(classes)
ax3.legend()
ax3.set_ylim([0, 1.1])
ax3.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

# 4. Per-Class F1-Score Comparison
ax4 = plt.subplot(2, 3, 4)
f1_101 = [
    metrics_101['f1_Bengin cases'],
    metrics_101['f1_Malignant cases'],
    metrics_101['f1_Normal cases']
]
f1_53 = [
    metrics_53['f1_Bengin cases'],
    metrics_53['f1_Malignant cases'],
    metrics_53['f1_Normal cases']
]

bars1 = ax4.bar(x - width/2, f1_101, width, label='CSPDarkNet101', color='#2E86AB', alpha=0.8)
bars2 = ax4.bar(x + width/2, f1_53, width, label='CSPDarkNet53', color='#A23B72', alpha=0.8)

ax4.set_ylabel('F1-Score')
ax4.set_title('Per-Class F1-Score Comparison', fontweight='bold', fontsize=12)
ax4.set_xticks(x)
ax4.set_xticklabels(classes)
ax4.legend()
ax4.set_ylim([0, 1.1])
ax4.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

# 5. Radar Chart for Overall Metrics
ax5 = plt.subplot(2, 3, 5, projection='polar')
categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
N = len(categories)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
cspdarknet101_scores += cspdarknet101_scores[:1]
cspdarknet53_scores += cspdarknet53_scores[:1]
angles += angles[:1]

ax5.plot(angles, cspdarknet101_scores, 'o-', linewidth=2, label='CSPDarkNet101', color='#2E86AB')
ax5.fill(angles, cspdarknet101_scores, alpha=0.25, color='#2E86AB')
ax5.plot(angles, cspdarknet53_scores, 'o-', linewidth=2, label='CSPDarkNet53', color='#A23B72')
ax5.fill(angles, cspdarknet53_scores, alpha=0.25, color='#A23B72')

ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(categories)
ax5.set_ylim(0, 1)
ax5.set_title('Performance Radar Chart', fontweight='bold', fontsize=12, pad=20)
ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax5.grid(True)

# 6. Heatmap Comparison
ax6 = plt.subplot(2, 3, 6)
comparison_data = pd.DataFrame({
    'CSPDarkNet101': [
        metrics_101['accuracy'],
        metrics_101['precision_macro'],
        metrics_101['recall_macro'],
        metrics_101['f1_macro'],
        np.mean(precision_101),
        np.mean(recall_101),
        np.mean(f1_101)
    ],
    'CSPDarkNet53': [
        metrics_53['accuracy'],
        metrics_53['precision_macro'],
        metrics_53['recall_macro'],
        metrics_53['f1_macro'],
        np.mean(precision_53),
        np.mean(recall_53),
        np.mean(f1_53)
    ]
}, index=['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)', 
          'Avg Precision', 'Avg Recall', 'Avg F1'])

sns.heatmap(comparison_data.T, annot=True, fmt='.3f', cmap='RdYlGn', 
            vmin=0.5, vmax=1.0, ax=ax6, cbar_kws={'label': 'Score'})
ax6.set_title('Metrics Heatmap Comparison', fontweight='bold', fontsize=12)
ax6.set_xlabel('')
ax6.set_ylabel('Model')

plt.tight_layout()
plt.savefig('model_comparison_graphs.png', dpi=300, bbox_inches='tight')
print("✅ Comparison graphs saved as 'model_comparison_graphs.png'")

# Create a second figure for detailed per-class analysis
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))

# Per-class metrics grouped bar chart
ax1 = axes[0, 0]
metrics_df = pd.DataFrame({
    'Precision (101)': precision_101,
    'Recall (101)': recall_101,
    'F1 (101)': f1_101,
    'Precision (53)': precision_53,
    'Recall (53)': recall_53,
    'F1 (53)': f1_53
}, index=classes)

metrics_df.plot(kind='bar', ax=ax1, width=0.8, colormap='tab10')
ax1.set_title('Detailed Per-Class Metrics', fontweight='bold', fontsize=12)
ax1.set_ylabel('Score')
ax1.set_xlabel('Class')
ax1.legend(loc='lower right', ncol=2)
ax1.set_ylim([0, 1.1])
ax1.grid(axis='y', alpha=0.3)
ax1.set_xticklabels(classes, rotation=45, ha='right')

# Model performance difference
ax2 = axes[0, 1]
diff_data = {
    'Accuracy': metrics_101['accuracy'] - metrics_53['accuracy'],
    'Precision': metrics_101['precision_macro'] - metrics_53['precision_macro'],
    'Recall': metrics_101['recall_macro'] - metrics_53['recall_macro'],
    'F1-Score': metrics_101['f1_macro'] - metrics_53['f1_macro']
}
colors = ['green' if v > 0 else 'red' for v in diff_data.values()]
bars = ax2.bar(diff_data.keys(), diff_data.values(), color=colors, alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.set_title('CSPDarkNet101 vs CSPDarkNet53 (Difference)', fontweight='bold', fontsize=12)
ax2.set_ylabel('Score Difference')
ax2.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:+.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

# Class-wise performance comparison (line plot)
ax3 = axes[1, 0]
x_pos = np.arange(len(classes))
ax3.plot(x_pos, precision_101, 'o-', label='Precision (101)', linewidth=2, markersize=8)
ax3.plot(x_pos, recall_101, 's-', label='Recall (101)', linewidth=2, markersize=8)
ax3.plot(x_pos, f1_101, '^-', label='F1 (101)', linewidth=2, markersize=8)
ax3.plot(x_pos, precision_53, 'o--', label='Precision (53)', linewidth=2, markersize=8, alpha=0.7)
ax3.plot(x_pos, recall_53, 's--', label='Recall (53)', linewidth=2, markersize=8, alpha=0.7)
ax3.plot(x_pos, f1_53, '^--', label='F1 (53)', linewidth=2, markersize=8, alpha=0.7)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(classes)
ax3.set_ylabel('Score')
ax3.set_title('Class-wise Performance Trends', fontweight='bold', fontsize=12)
ax3.legend(loc='best', ncol=2)
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, 1.1])

# Summary statistics table
ax4 = axes[1, 1]
ax4.axis('off')
summary_data = [
    ['Metric', 'CSPDarkNet101', 'CSPDarkNet53', 'Winner'],
    ['Accuracy', f"{metrics_101['accuracy']:.3f}", f"{metrics_53['accuracy']:.3f}", 
     '101' if metrics_101['accuracy'] > metrics_53['accuracy'] else '53'],
    ['Precision', f"{metrics_101['precision_macro']:.3f}", f"{metrics_53['precision_macro']:.3f}",
     '101' if metrics_101['precision_macro'] > metrics_53['precision_macro'] else '53'],
    ['Recall', f"{metrics_101['recall_macro']:.3f}", f"{metrics_53['recall_macro']:.3f}",
     '101' if metrics_101['recall_macro'] > metrics_53['recall_macro'] else '53'],
    ['F1-Score', f"{metrics_101['f1_macro']:.3f}", f"{metrics_53['f1_macro']:.3f}",
     '101' if metrics_101['f1_macro'] > metrics_53['f1_macro'] else '53'],
]

table = ax4.table(cellText=summary_data, cellLoc='center', loc='center',
                  colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color winner cells
for i in range(1, 5):
    if summary_data[i][3] == '101':
        table[(i, 3)].set_facecolor('#90EE90')
    else:
        table[(i, 3)].set_facecolor('#FFB6C1')

ax4.set_title('Performance Summary', fontweight='bold', fontsize=12, pad=20)

plt.tight_layout()
plt.savefig('detailed_comparison_graphs.png', dpi=300, bbox_inches='tight')
print("✅ Detailed comparison graphs saved as 'detailed_comparison_graphs.png'")

print("\n" + "="*70)
print("📊 Summary:")
print("="*70)
print(f"CSPDarkNet101 - Accuracy: {metrics_101['accuracy']:.2%}, F1: {metrics_101['f1_macro']:.3f}")
print(f"CSPDarkNet53  - Accuracy: {metrics_53['accuracy']:.2%}, F1: {metrics_53['f1_macro']:.3f}")
print(f"\n🏆 Winner: CSPDarkNet{'101' if metrics_101['accuracy'] > metrics_53['accuracy'] else '53'}")
print("="*70)
