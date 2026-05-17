import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Create an output directory for the graphs
os.makedirs('graphs', exist_ok=True)

# Load stats for all models
models = [
    ('lr', 'Logistic Regression', '#7c3aed'),
    ('svm', 'Support Vector Machine', '#67e8f9'),
    ('rf', 'Random Forest', '#10b981'),
    ('gb', 'Gradient Boosting', '#fcd34d')
]

metrics = ['accuracy', 'precision', 'recall', 'f1_score']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

# Set global dark theme styling
plt.style.use('dark_background')
bg_color = '#0b0f1e'
surface_color = '#111827'
text_color = '#e2e8f0'
grid_color = '#1f2d45'

# ---------------------------------------------------------
# 1. GENERATE COMPARISON GRAPH (ALL 4 MODELS)
# ---------------------------------------------------------
print("Generating comparative graph...")
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

x = np.arange(len(metrics))
width = 0.2

for i, (key, name, color) in enumerate(models):
    with open(f'models/{key}_stats.json') as f:
        stats = json.load(f)
    
    # Get scores * 100 for percentage
    scores = [stats[m] * 100 for m in metrics]
    
    # Calculate offset for grouped bars
    offset = (i - 1.5) * width
    
    rects = ax.bar(x + offset, scores, width, label=name, color=color)
    
    # Add value labels on top of bars
    ax.bar_label(rects, padding=3, fmt='%.1f', color=text_color, fontsize=9)

ax.set_ylabel('Score (%)', fontsize=12, color=text_color, labelpad=10)
ax.set_title('Machine Learning Models Comparison', fontsize=16, color=text_color, pad=20, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metric_names, fontsize=12, color=text_color)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=4, frameon=False, fontsize=11, labelcolor=text_color)

# Clean up axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color(grid_color)
ax.spines['left'].set_color(grid_color)
ax.tick_params(colors=text_color)
ax.grid(axis='y', color=grid_color, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('graphs/model_comparison_all.png', dpi=300, bbox_inches='tight', facecolor=bg_color)
plt.close()

# ---------------------------------------------------------
# 2. GENERATE INDIVIDUAL MODEL GRAPHS (Like the screenshot)
# ---------------------------------------------------------
for key, name, color in models:
    print(f"Generating graph for {name}...")
    with open(f'models/{key}_stats.json') as f:
        stats = json.load(f)
        
    scores = [stats[m] * 100 for m in metrics]
    auc_score = stats['roc_auc']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(surface_color)
    ax.set_facecolor(surface_color)
    
    x_pos = np.arange(len(metrics))
    
    # Create bars
    rects = ax.bar(x_pos, scores, width=0.4, color=color)
    
    # Add labels
    ax.bar_label(rects, padding=5, fmt='%.1f', color=text_color, fontsize=11, fontweight='bold')
    
    # Styling
    ax.set_title(f'Performance Metrics: {name}', fontsize=14, color=text_color, pad=20, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names, fontsize=11, color=text_color)
    ax.set_ylim(0, 100)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(grid_color)
    ax.spines['left'].set_color(grid_color)
    ax.tick_params(colors=text_color)
    ax.grid(axis='y', color=grid_color, linestyle='-', alpha=0.5)
    
    # Add the AUC-ROC Score box at the bottom
    plt.subplots_adjust(bottom=0.25) # Make room
    fig.text(0.5, 0.05, f'AUC-ROC Score: {auc_score:.3f}', 
             ha='center', va='center', fontsize=14, fontweight='bold',
             color=surface_color, bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.8'))
    
    plt.savefig(f'graphs/{key}_performance.png', dpi=300, bbox_inches='tight', facecolor=surface_color)
    plt.close()

print("\n[OK] All graphs generated successfully in the 'graphs/' folder!")
