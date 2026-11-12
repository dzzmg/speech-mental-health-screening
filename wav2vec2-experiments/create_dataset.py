import os
import pandas as pd
import librosa
import numpy as np
from sklearn.model_selection import train_test_split

def create_labeled_dataset(dataset_path):
    """Create a labeled dataset from the folder structure"""
    
    audio_files = []
    labels = []
    label_map = {
        'Normal': 0,           # Non-depressed
        'Stage1': 1,           # Mid-depressed  
        'Stage2': 2            # High-depressed
    }
    
    # Walk through the dataset
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                full_path = os.path.join(root, file)
                audio_files.append(full_path)
                
                # Determine label from folder structure
                if 'Normal' in root:
                    labels.append(0)
                elif 'Stage1' in root:
                    labels.append(1)
                elif 'Stage2' in root:
                    labels.append(2)
                else:
                    print(f"Warning: Could not label {full_path}")
    
    # Create DataFrame
    df = pd.DataFrame({
        'audio_path': audio_files,
        'label': labels,
        'label_name': ['Normal' if l == 0 else 'Stage1' if l == 1 else 'Stage2' for l in labels]
    })
    
    print("=== Dataset Summary ===")
    print(f"Total files: {len(df)}")
    print(f"Class distribution:")
    print(df['label_name'].value_counts())
    
    return df

def validate_audio_files(df, sample_size=5):
    """Check if audio files are loadable"""
    print("\n=== Audio Validation ===")
    
    valid_files = []
    issues = []
    
    for i, row in df.iterrows():
        if i >= sample_size:  # Check only first few files
            break
            
        try:
            y, sr = librosa.load(row['audio_path'], sr=16000, duration=5)
            duration = len(y) / sr
            print(f"✓ {os.path.basename(row['audio_path'])}: {duration:.2f}s, label: {row['label_name']}")
            valid_files.append(True)
        except Exception as e:
            print(f"✗ {row['audio_path']}: {e}")
            issues.append(row['audio_path'])
            valid_files.append(False)
    
    return valid_files, issues

# Main execution
if __name__ == "__main__":
    dataset_path = "data/Original_Dataset/"
    
    # Create labeled dataset
    df = create_labeled_dataset(dataset_path)
    
    # Validate audio files
    valid_files, issues = validate_audio_files(df)
    
    # Save the dataset info
    df.to_csv('dataset_labels.csv', index=False)
    print(f"\nSaved dataset info to 'dataset_labels.csv'")
    
    print("\n🎉 Ready for next step: Baseline model!")