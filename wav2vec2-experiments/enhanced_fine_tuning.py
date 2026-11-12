# enhanced_fine_tuning_FIXED.py
import torch
import librosa
import numpy as np
from transformers import (
    Wav2Vec2Processor, Wav2Vec2ForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from datasets import Dataset
from collections import Counter
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    cm = confusion_matrix(labels, predictions)
    print("\n🔍 Confusion Matrix:")
    print("         Predicted:")
    print("        Normal  Stage1  Stage2")
    label_names = ['Normal', 'Stage1', 'Stage2']
    for i, row in enumerate(cm):
        print(f"  {label_names[i]}:  {row[0]:6d}  {row[1]:6d}  {row[2]:6d}")
    
    print("\n📊 Classification Report:")
    print(classification_report(labels, predictions, 
                                target_names=['Normal', 'Stage1', 'Stage2'],
                                digits=4))
    
    return {"accuracy": accuracy, "f1": f1}

class EnhancedDepressionTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def compute_class_weights(self, labels):
        """Calculate class weights for imbalanced dataset"""
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        print(f"📊 Class weights: {class_weights}")
        return torch.tensor(class_weights, dtype=torch.float32).to(self.device)
    
    def audio_augmentation(self, audio_array, sr=16000):
        """Apply GENTLE audio data augmentation"""
        augmented = audio_array.copy()
        
        # GENTLER time stretching (0.9x to 1.1x speed)
        if np.random.random() > 0.7:  # Less frequent
            rate = np.random.uniform(0.9, 1.1)  # Smaller range
            try:
                augmented = librosa.effects.time_stretch(augmented, rate=rate)
            except:
                pass
        
        # GENTLER pitch shifting (-1 to +1 semitones)
        if np.random.random() > 0.7:
            n_steps = np.random.randint(-1, 2)  # Smaller range
            try:
                augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=n_steps)
            except:
                pass
        
        # Very gentle noise
        if np.random.random() > 0.8:  # Less frequent
            noise = np.random.normal(0, 0.002, augmented.shape)  # Less noise
            augmented = augmented + noise
        
        return augmented
    
    def oversample_minority_classes(self, indices_by_class, target_count):
        """Oversample minority classes - FIXED VERSION"""
        print("🔄 Oversampling minority classes...")
        
        oversampled_indices = []
        for label, indices in indices_by_class.items():
            if len(indices) < target_count:
                oversampled = resample(
                    indices, 
                    replace=True, 
                    n_samples=target_count, 
                    random_state=42
                )
                oversampled_indices.extend(oversampled)
                print(f"  Class {label}: {len(indices)} → {target_count}")
            else:
                oversampled_indices.extend(indices)
                print(f"  Class {label}: {len(indices)} (no oversampling)")
        
        return oversampled_indices
    
    def extract_features_for_analysis(self, dataset, model, processor):
        """Extract features for statistical analysis"""
        model.eval()
        all_features = []
        all_labels = []
        all_audio_paths = []
        
        print("\n" + "="*60)
        print("🔬 EXTRACTING FEATURES FOR ANALYSIS")
        print("="*60)
        
        with torch.no_grad():
            for i in range(len(dataset)):
                if i % 50 == 0:
                    print(f"📊 Progress: {i}/{len(dataset)}")
                    
                audio_path = dataset[i]["audio_path"]
                label = dataset[i]["label"]
                
                try:
                    audio_array, _ = librosa.load(audio_path, sr=16000)
                    
                    target_length = 80000
                    if len(audio_array) > target_length:
                        audio_array = audio_array[:target_length]
                    elif len(audio_array) < target_length:
                        padding = target_length - len(audio_array)
                        audio_array = np.pad(audio_array, (0, padding))
                    
                    inputs = processor(
                        audio_array, 
                        sampling_rate=16000, 
                        return_tensors="pt",
                        padding=True,
                        max_length=target_length,
                        truncation=True
                    )
                    
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = model.wav2vec2(**inputs)
                    feature_vector = outputs.last_hidden_state.mean(dim=1)
                    
                    all_features.append(feature_vector.cpu().numpy())
                    all_labels.append(label)
                    all_audio_paths.append(audio_path)
                    
                except Exception as e:
                    print(f"⚠️ Error processing {audio_path}: {e}")
                    continue
        
        features_array = np.vstack(all_features)
        labels_array = np.array(all_labels)
        
        print(f"\n📐 Features shape: {features_array.shape}")
        print(f"🏷️ Labels shape: {labels_array.shape}")
        
        return features_array, labels_array, all_audio_paths

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Custom loss function with class weights"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def load_depression_dataset_flexible(base_dir):
    """Load dataset"""
    import os
    import glob
    
    audio_paths = []
    labels = []
    
    normal_path = os.path.join(base_dir, "Normal")
    stage1_path = os.path.join(base_dir, "Depression", "Stage1")
    stage2_path = os.path.join(base_dir, "Depression", "Stage2")
    
    if os.path.exists(normal_path):
        normal_files = glob.glob(os.path.join(normal_path, "*.wav"))
        audio_paths.extend(normal_files)
        labels.extend([0] * len(normal_files))
        print(f"✅ Found {len(normal_files)} Normal samples")
    
    if os.path.exists(stage1_path):
        stage1_files = glob.glob(os.path.join(stage1_path, "*.wav"))
        audio_paths.extend(stage1_files)
        labels.extend([1] * len(stage1_files))
        print(f"✅ Found {len(stage1_files)} Stage1 Depression samples")
    
    if os.path.exists(stage2_path):
        stage2_files = glob.glob(os.path.join(stage2_path, "*.wav"))
        audio_paths.extend(stage2_files)
        labels.extend([2] * len(stage2_files))
        print(f"✅ Found {len(stage2_files)} Stage2 Depression samples")
    
    if len(audio_paths) == 0:
        raise ValueError("No audio files found.")
    
    dataset_dict = {'audio_path': audio_paths, 'label': labels}
    dataset = Dataset.from_dict(dataset_dict)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Normal (0): {labels.count(0)} samples")
    print(f"   Stage1 (1): {labels.count(1)} samples") 
    print(f"   Stage2 (2): {labels.count(2)} samples")
    print(f"   Total: {len(labels)} samples")
    
    return dataset

# FIXED MAIN TRAINING FUNCTION
def enhanced_fine_tuning_FIXED(dataset, labels):
    """FIXED: Enhanced fine-tuning with train/val/test splits"""
    
    enhancer = EnhancedDepressionTrainer()
    
    # ============================================================
    # STEP 1: SPLIT INTO TRAIN/VAL/TEST (before any oversampling!)
    # ============================================================
    print("\n" + "="*60)
    print("📊 STEP 1: Splitting dataset into Train/Val/Test")
    print("="*60)
    
    indices = list(range(len(dataset)))
    
    # First: Split off test set (15%)
    train_val_indices, test_indices = train_test_split(
        indices, test_size=0.15, stratify=labels, random_state=42
    )

    # Then: Split train/val from remaining data
    train_val_labels = [labels[i] for i in train_val_indices]
    train_indices, val_indices = train_test_split(
        train_val_indices, 
        test_size=0.18,  # ~15% of total
        stratify=train_val_labels, 
        random_state=42
    )

    # Save test indices for later evaluation
    np.save("test_indices.npy", test_indices)
    
    train_labels = [labels[i] for i in train_indices]
    val_labels = [labels[i] for i in val_indices]
    test_labels = [labels[i] for i in test_indices]
    
    print(f"Training set: {len(train_indices)} samples")
    print(f"  Normal: {train_labels.count(0)}, Stage1: {train_labels.count(1)}, Stage2: {train_labels.count(2)}")
    
    print(f"Validation set: {len(val_indices)} samples")
    print(f"  Normal: {val_labels.count(0)}, Stage1: {val_labels.count(1)}, Stage2: {val_labels.count(2)}")
    
    print(f"Test set: {len(test_indices)} samples")
    print(f"  Normal: {test_labels.count(0)}, Stage1: {test_labels.count(1)}, Stage2: {test_labels.count(2)}")
    
    # ============================================================
    # STEP 2: OVERSAMPLE ONLY TRAINING SET
    # ============================================================
    print("\n" + "="*60)
    print("📊 STEP 2: Oversampling ONLY training set")
    print("="*60)
    
    # Group training indices by class
    train_indices_by_class = {}
    for idx in train_indices:
        label = labels[idx]
        if label not in train_indices_by_class:
            train_indices_by_class[label] = []
        train_indices_by_class[label].append(idx)
    
    # Find max class size in training set
    max_train_count = max(len(indices) for indices in train_indices_by_class.values())
    
    # Oversample minority classes in training set
    balanced_train_indices = enhancer.oversample_minority_classes(
        train_indices_by_class, 
        max_train_count
    )
    
    print(f"\nAfter oversampling:")
    print(f"  Training samples: {len(balanced_train_indices)}")
    
    # ============================================================
    # STEP 3: Create datasets
    # ============================================================
    print("\n" + "="*60)
    print("📊 STEP 3: Creating train/val/test datasets")
    print("="*60)
    
    # Training dataset (oversampled)
    train_dataset = dataset.select(balanced_train_indices)
    
    # Validation dataset (ORIGINAL, NO oversampling)
    val_dataset = dataset.select(val_indices)
    
    # Test dataset (ORIGINAL, NO oversampling)
    test_dataset = dataset.select(test_indices)
    
    # ============================================================
    # STEP 4: Compute class weights from ORIGINAL training distribution
    # ============================================================
    class_weights = enhancer.compute_class_weights(train_labels)
    
    # ============================================================
    # STEP 5: Load model
    # ============================================================
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base",
        num_labels=3,
        problem_type="single_label_classification"
    )
    model.to(enhancer.device)
    
    # ============================================================
    # STEP 6: Preprocessing
    # ============================================================
    def enhanced_preprocess_function(examples, augment=False):
        audio_arrays = []
        target_length = 80000
        
        for audio_path in examples["audio_path"]:
            try:
                audio_array, _ = librosa.load(audio_path, sr=16000)
                
                # Apply gentle augmentation only to training data
                if augment and np.random.random() > 0.5:
                    audio_array = enhancer.audio_augmentation(audio_array)
                
                if len(audio_array) > target_length:
                    audio_array = audio_array[:target_length]
                elif len(audio_array) < target_length:
                    padding = target_length - len(audio_array)
                    audio_array = np.pad(audio_array, (0, padding))
                    
                audio_arrays.append(audio_array)
            except Exception as e:
                print(f"⚠️ Error loading {audio_path}: {e}")
                audio_arrays.append(np.zeros(target_length))
        
        inputs = processor(
            audio_arrays,
            sampling_rate=16000,
            padding=True,
            max_length=target_length,
            truncation=True,
            return_tensors="pt",
        )
        return inputs
    
    print("🔄 Preprocessing training data WITH augmentation...")
    train_dataset = train_dataset.map(
        lambda x: enhanced_preprocess_function(x, augment=True),
        batched=True,
        batch_size=4,
        remove_columns=['audio_path']
    )
    
    print("🔄 Preprocessing validation data WITHOUT augmentation...")
    val_dataset = val_dataset.map(
        lambda x: enhanced_preprocess_function(x, augment=False),
        batched=True,
        batch_size=4,
        remove_columns=['audio_path']
    )
    
    train_dataset.set_format(type='torch', columns=['input_values', 'label'])
    val_dataset.set_format(type='torch', columns=['input_values', 'label'])
    
    print(f"\n✅ Final dataset sizes:")
    print(f"  Training: {len(train_dataset)} samples (oversampled)")
    print(f"  Validation: {len(val_dataset)} samples (original)")
    print(f"  Test: {len(test_dataset)} samples (original) - SAVED FOR LATER")
    
    # ============================================================
    # STEP 7: Training arguments (ADJUSTED)
    # ============================================================
    training_args = TrainingArguments(
        output_dir="./enhanced_depression_model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,  # Reduced from 5e-5
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=20,  # Reduced from 40
        warmup_steps=100,  # Reduced from 200
        weight_decay=0.01,  # Reduced from 0.1
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",  # Changed from accuracy
        greater_is_better=True,
        push_to_hub=False,
        no_cuda=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # ============================================================
    # STEP 8: Train
    # ============================================================
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    print("\n" + "="*60)
    print("🚀 STARTING TRAINING")
    print("="*60)
    trainer.train()
    
    # ============================================================
    # STEP 9: Extract features from ORIGINAL dataset
    # ============================================================
    print("\n" + "="*60)
    print("💾 EXTRACTING FEATURES FROM ORIGINAL DATASET")
    print("="*60)
    
    features, feature_labels, audio_paths = enhancer.extract_features_for_analysis(
        dataset, model, processor
    )
    
    np.save("enhanced_depression_features_1.npy", features)
    np.save("enhanced_depression_labels_1.npy", feature_labels)
    np.save("enhanced_audio_paths_1.npy", audio_paths)
    
    print(f"✅ Features saved: {features.shape}")
    
    return trainer, processor, features, feature_labels, test_dataset

# Usage
if __name__ == "__main__":
    dataset = load_depression_dataset_flexible(
        "data/Corrected_Dataset"
    )
    labels = [dataset[i]['label'] for i in range(len(dataset))]
    
    print(f"📁 Loaded dataset: {len(dataset)} samples")
    
    # Run FIXED training
    trainer, processor, features, feature_labels, test_dataset = enhanced_fine_tuning_FIXED(dataset, labels)
    
    trainer.save_model("./enhanced_depression_model")
    processor.save_pretrained("./enhanced_depression_model")
    
    print("\n🎉 TRAINING COMPLETE!")
    print("💾 Test dataset saved for future evaluation")