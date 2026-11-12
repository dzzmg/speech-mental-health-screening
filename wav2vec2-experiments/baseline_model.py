import pandas as pd
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load your dataset
df = pd.read_csv('dataset_labels.csv')
print("=== Loading Dataset ===")
print(f"Total samples: {len(df)}")
print(df['label_name'].value_counts())

def extract_features_robust(audio_path):
    """Extract audio features with robust error handling"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Ensure we have exactly 3 seconds of audio
        target_samples = int(3.0 * sr)
        if len(y) < target_samples:
            # Pad with zeros if too short
            y = np.pad(y, (0, target_samples - len(y)), mode='constant')
        else:
            # Take first 3 seconds if too long
            y = y[:target_samples]
        
        features = []
        
        # MFCC features - ensure they're flat arrays
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1).flatten().tolist()
        mfcc_std = np.std(mfcc, axis=1).flatten().tolist()
        
        features.extend(mfcc_mean)
        features.extend(mfcc_std)
        
        # Spectral features - ensure they're scalars
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        
        features.extend([spectral_centroid, spectral_rolloff, spectral_bandwidth])
        
        # Rhythm features
        try:
            tempo = float(librosa.beat.beat_track(y=y, sr=sr)[0])
        except:
            tempo = 0.0
        features.append(tempo)
        
        # Zero crossing rate
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        features.append(zcr)
        
        # Root Mean Square Energy
        rms = float(np.mean(librosa.feature.rms(y=y)))
        features.append(rms)
        
        # Convert all to Python floats (not numpy types)
        features = [float(x) for x in features]
        
        # Final validation
        if len(features) != 32:
            print(f"Warning: {audio_path} has {len(features)} features, padding/truncating")
            if len(features) < 32:
                features.extend([0.0] * (32 - len(features)))
            else:
                features = features[:32]
        
        return features
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def debug_detailed_features(df, sample_size=5):
    """Debug feature structure in detail"""
    print("\n=== Detailed Feature Debug ===")
    
    for idx, row in df.head(sample_size).iterrows():
        features = extract_features_robust(row['audio_path'])
        if features is not None:
            print(f"\nFile {idx}: {os.path.basename(row['audio_path'])}")
            print(f"  Number of features: {len(features)}")
            print(f"  Feature types: {[type(x) for x in features[:3]]}...")  # Show first 3 types
            print(f"  Feature shapes: All scalars? {all(np.isscalar(x) for x in features)}")
            print(f"  First 3 features: {features[:3]}")
        else:
            print(f"File {idx}: FAILED")

def create_baseline_model_robust(df, sample_size=200):
    """Create and evaluate baseline model with robust feature handling"""
    print("\n=== Creating Baseline Model ===")
    print("Extracting features... This may take a few minutes...")
    
    # Use a subset for quick testing
    if sample_size:
        df_subset = df.sample(n=min(sample_size, len(df)), random_state=42)
    else:
        df_subset = df
    
    X = []
    y = []
    failed_files = []
    
    for idx, row in df_subset.iterrows():
        features = extract_features_robust(row['audio_path'])
        if features is not None and len(features) == 32:
            # Ensure all features are simple Python floats, not numpy arrays
            features_clean = [float(x) for x in features]
            X.append(features_clean)
            y.append(int(row['label']))  # Ensure label is int
        else:
            failed_files.append(row['audio_path'])
    
    print(f"Successfully processed {len(X)}/{len(df_subset)} samples")
    if failed_files:
        print(f"Failed to process {len(failed_files)} files")
    
    if len(X) < 20:
        print("Not enough samples for training")
        return None, None
    
    # Convert to numpy arrays with explicit dtype
    try:
        X = np.array(X, dtype=np.float64)  # Force float64 type
        y = np.array(y, dtype=np.int64)    # Force int64 type
    except Exception as e:
        print(f"Error converting to numpy arrays: {e}")
        # Debug the problematic data
        for i, feat in enumerate(X[:3]):
            print(f"Sample {i} - Length: {len(feat)}, Types: {[type(x) for x in feat[:3]]}")
        return None, None
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Data type: {X.dtype}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Train Random Forest classifier
    print("Training model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n🎯 BASELINE ACCURACY: {accuracy:.3f}")
    print("\n📊 Detailed Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Normal', 'Stage1', 'Stage2']))
    
    # Simple confusion matrix
    print("\n📈 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print("Rows = True labels, Columns = Predicted labels")
    print("        Normal  Stage1  Stage2")
    for i, row in enumerate(cm):
        label_name = ['Normal', 'Stage1', 'Stage2'][i]
        print(f"{label_name:8} {row[0]:6} {row[1]:6} {row[2]:6}")
    
    # Feature importance
    feature_names = [f'MFCC_mean_{i}' for i in range(13)] + \
                   [f'MFCC_std_{i}' for i in range(13)] + \
                   ['Spectral_Centroid', 'Spectral_Rolloff', 'Spectral_Bandwidth',
                    'Tempo', 'ZCR', 'RMS']
    
    # Get top 10 most important features
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    print("\n🔍 Top 10 Most Important Features:")
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    return accuracy, clf

# Add import
import os

# Run the baseline model
if __name__ == "__main__":
    # First debug in detail
    debug_detailed_features(df, sample_size=3)
    
    # Then run the full model
    accuracy, model = create_baseline_model_robust(df, sample_size=200)
    
    if accuracy is not None:
        if accuracy > 0.55:
            print("\n🎉 SUCCESS: Your dataset shows predictable patterns!")
            print("Next step: Fine-tune wav2vec2 model!")
        elif accuracy > 0.33:  # Better than random chance (33.3% for 3 classes)
            print("\n⚠️ Some patterns detected, but accuracy is modest")
            print("wav2vec2 might still work well!")
        else:
            print("\n❌ Baseline accuracy is very low.")
            print("The dataset might not have strong acoustic patterns for depression detection.")
    else:
        print("\n❌ Model training failed due to data issues")