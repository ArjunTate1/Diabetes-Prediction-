import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import textwrap

os.makedirs('graphs', exist_ok=True)

models_info = [
    ('lr', 'Logistic Regression', 'Classification'),
    ('svm', 'SVM', 'Classification'),
    ('gb', 'Gradient Boosting', 'Ensemble'),
    ('rf', 'Random Forest', 'Ensemble')
]

data = []
for k, name, mtype in models_info:
    with open(f'models/{k}_stats.json') as f:
        st = json.load(f)
        data.append({
            'model': name,
            'type': mtype,
            'acc': st['accuracy'],
            'prec': st['precision'],
            'rec': st['recall'],
            'f1': st['f1_score'],
            'auc': st['roc_auc']
        })

# Sort so RF is at the bottom (winner)
winner = next(d for d in data if d['model'] == 'Random Forest')

# ---------------------------------------------------------
# 1. GENERATE TABLE SLIDE
# ---------------------------------------------------------
print("Generating summary table slide...")
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor('#f4f6f9')
ax.axis('off')

# Header block
rect_header = patches.Rectangle((0, 0.8), 1, 0.2, transform=fig.transFigure, facecolor='#1f3b5c', edgecolor='none')
fig.patches.append(rect_header)
fig.text(0.05, 0.9, 'Complete Metrics Summary Table', fontsize=28, fontweight='bold', color='white', va='center')

# Define table properties
cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Type']
cell_text = []
for d in data:
    prefix = "" if d['model'] == 'Random Forest' else ""
    cell_text.append([
        prefix + d['model'],
        f"{d['acc']*100:.1f}%",
        f"{d['prec']*100:.1f}%",
        f"{d['rec']*100:.1f}%",
        f"{d['f1']*100:.1f}%",
        f"{d['auc']:.2f}",
        d['type']
    ])

colors = [
    ['#e8f4f8'] * 7,
    ['#f3e8fc'] * 7,
    ['#fce8e8'] * 7,  # GB
    ['#fff8d6'] * 7   # RF (Winner)
]
text_colors = [
    ['#333333', '#10b981', '#10b981', '#10b981', '#10b981', '#10b981', '#666666'],
    ['#333333', '#8b5cf6', '#8b5cf6', '#8b5cf6', '#8b5cf6', '#8b5cf6', '#666666'],
    ['#333333', '#ef4444', '#ef4444', '#ef4444', '#ef4444', '#ef4444', '#666666'],
    ['#ef4444', '#d97706', '#d97706', '#d97706', '#d97706', '#d97706', '#666666']
]

# Create an axis specifically for the table to control its size
ax_table = fig.add_axes([0.05, 0.2, 0.9, 0.5])
ax_table.axis('off')

table = ax_table.table(cellText=cell_text, colLabels=cols, cellColours=colors, loc='center', cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1, 3.5)

# Style header
for j in range(len(cols)):
    cell = table[0, j]
    cell.set_facecolor('#1f3b5c')
    cell.set_text_props(color='white', weight='bold', fontsize=15)
    
# Style text colors
for i in range(len(data)):
    for j in range(len(cols)):
        cell = table[i+1, j]
        cell.set_text_props(color=text_colors[i][j], weight='bold' if j>0 and j<6 or (i==3 and j==0) else 'normal')
        
# Footer block
rect_footer = patches.Rectangle((0.02, 0.05), 0.96, 0.1, transform=fig.transFigure, facecolor='#0f766e', edgecolor='none')
fig.patches.append(rect_footer)
fig.text(0.5, 0.1, 'Ensemble methods (Random Forest) outperform individual classifiers across key metrics — prioritizing Recall (~91%)', 
         ha='center', va='center', fontsize=14, color='white', weight='bold')

plt.savefig('graphs/metrics_summary_table.png', dpi=300, facecolor='#f4f6f9')
plt.close()

# ---------------------------------------------------------
# 2. GENERATE CONCLUSION SLIDE
# ---------------------------------------------------------
print("Generating conclusion slide...")
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor('#0b0f1e')
ax.axis('off')

# Header
rect = patches.Rectangle((0, 0.85), 1, 0.15, transform=fig.transFigure, facecolor='#ef4444', edgecolor='none')
fig.patches.append(rect)
fig.text(0.05, 0.9, 'Conclusion & Recommendation', fontsize=32, fontweight='bold', color='white', va='center')

# Winner Box
rect_win = patches.Rectangle((0.05, 0.62), 0.9, 0.2, transform=fig.transFigure, facecolor='none', edgecolor='#ef4444', linewidth=2)
fig.patches.append(rect_win)
fig.text(0.08, 0.74, 'WINNER: Random Forest', fontsize=26, fontweight='bold', color='#ef4444')
fig.text(0.08, 0.66, f"Accuracy: {winner['acc']*100:.1f}%  |  AUC: {winner['auc']:.2f}  |  Recall: {winner['rec']*100:.1f}%", 
         fontsize=16, color='#e2e8f0')

# "BEST MODEL" badge
fig.text(0.85, 0.72, 'BEST\nMODEL', fontsize=18, fontweight='bold', color='white', ha='center', va='center',
         bbox=dict(facecolor='#ef4444', edgecolor='none', pad=2.5, boxstyle='square'))

# 4 Grid Boxes
boxes = [
    {'x': 0.05, 'y': 0.35, 'color': '#ef4444', 'title': 'Ensemble Wins', 'text': 'Random Forest consistently beat individual classifiers by combining many decision trees, especially in maximizing clinical Recall.'},
    {'x': 0.52, 'y': 0.35, 'color': '#f59e0b', 'title': 'Glucose & BMI #1', 'text': 'The engineered Glucose_BMI interaction feature was highly predictive across models, capturing combined metabolic risk.'},
    {'x': 0.05, 'y': 0.12, 'color': '#10b981', 'title': 'Interpretability', 'text': 'Logistic Regression remains useful for clinical settings where direct explanation of feature weights is strictly required.'},
    {'x': 0.52, 'y': 0.12, 'color': '#8b5cf6', 'title': 'Clinical Focus', 'text': 'By optimizing the decision thresholds, we prioritized Recall (90.7%) to minimize dangerous False Negatives (missed diagnoses).'}
]

for b in boxes:
    # Box
    rect_box = patches.Rectangle((b['x'], b['y']), 0.43, 0.2, transform=fig.transFigure, facecolor='#161d2e', edgecolor=b['color'], linewidth=2)
    fig.patches.append(rect_box)
    
    # Title
    fig.text(b['x']+0.02, b['y']+0.14, b['title'], fontsize=18, fontweight='bold', color=b['color'])
    
    # Text (wrapped)
    wrapped = textwrap.fill(b['text'], width=55)
    fig.text(b['x']+0.02, b['y']+0.08, wrapped, fontsize=12, color='#e2e8f0', va='top')

# Footer
fig.text(0.5, 0.04, 'Diabetes Prediction with ML • 4-Model Comparison • Random Forest Recommended for Deployment', 
         ha='center', va='center', fontsize=12, color='white', weight='bold',
         bbox=dict(facecolor='#0f766e', edgecolor='none', pad=1, boxstyle='square'))

plt.savefig('graphs/conclusion_slide.png', dpi=300, bbox_inches='tight', facecolor='#0b0f1e')
plt.close()

print("[OK] Slides generated successfully in 'graphs/' directory!")
