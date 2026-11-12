import os
import pandas as pd
import json

def explore_kaggle_dataset(dataset_path):
    print("=== Exploring Kaggle Dataset ===")
    
    # List all files and folders
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(dataset_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files only
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")
    
    # Look for specific file types
    print("\n=== Looking for Audio Files ===")
    audio_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Look for metadata (PHQ-9 scores)
    print("\n=== Looking for Metadata ===")
    metadata_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if any(keyword in file.lower() for keyword in ['score', 'phq', 'metadata', 'label', 'csv', 'json']):
                metadata_files.append(os.path.join(root, file))
                print(f"Found metadata file: {file}")

    return audio_files, metadata_files

# Run exploration
dataset_path = "data"  # Change this to your actual path
audio_files, metadata_files = explore_kaggle_dataset(dataset_path)

# Show first few audio files
print("\n=== First 10 Audio Files ===")
for audio_file in audio_files[:10]:
    print(audio_file) 