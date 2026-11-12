import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
features = np.load("enhanced_depression_features.npy")
labels = np.load("enhanced_depression_labels.npy")

# Select first 20 features and first 30 samples for visualization
selected_features = features[:30, :20]
selected_labels = labels[:30]

# Create DataFrame
df_heatmap = pd.DataFrame(
    selected_features,
    columns=[f'F{i+1}' for i in range(20)]
)
df_heatmap['Group'] = selected_labels
df_heatmap['Group'] = df_heatmap['Group'].map({0: 'Normal', 1: 'Stage1', 2: 'Stage2'})

# Sort by group for better visualization
df_heatmap = df_heatmap.sort_values('Group')

# Create heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(
    df_heatmap.iloc[:, :-1],  # All columns except 'Group'
    cmap='RdYlBu_r',
    center=0,
    cbar_kws={'label': 'Feature Value'},
    yticklabels=df_heatmap['Group'].values,
    xticklabels=[f'F{i+1}' for i in range(20)]
)
plt.title('Feature Patterns Across Depression Groups\n(First 20 Features, 30 Samples)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Feature Dimension', fontsize=12)
plt.ylabel('Audio Sample (sorted by group)', fontsize=12)
plt.tight_layout()
plt.savefig('feature_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Saved: feature_heatmap.png")