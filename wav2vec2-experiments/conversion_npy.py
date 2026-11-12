import numpy as np
import pandas as pd

# Load the data
features = np.load("enhanced_depression_features.npy")
labels = np.load("enhanced_depression_labels.npy")

# Create a DataFrame
# Each row is one audio sample, each column is a feature dimension
df = pd.DataFrame(features)
df['label'] = labels

# Add readable label names
label_names = {0: 'Normal', 1: 'Stage1', 2: 'Stage2'}
df['label_name'] = df['label'].map(label_names)

# Save to CSV
df.to_csv("enhanced_depression_features.csv", index=False)
print("✅ Saved to enhanced_depression_features.csv")