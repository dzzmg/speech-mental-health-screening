# Import all required libraries
import torch
import librosa
import numpy as np
import pandas as pd
from datasets import Dataset, ClassLabel
from transformers import (
    Wav2Vec2Processor, 
    Wav2Vec2ForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import os
from pathlib import Path
import glob

# 1. Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Dataset loading
def load_depression_dataset_flexible(base_dir):
    """
    Flexible dataset loading that adapts to your folder structure
    """
    audio_paths = []
    labels = []
    
    # Define paths based on your discovered structure
    normal_path = os.path.join(base_dir, "Normal")
    stage1_path = os.path.join(base_dir, "Depression", "Stage1")
    stage2_path = os.path.join(base_dir, "Depression", "Stage2")
    
    # Load normal samples (label 0)
    if os.path.exists(normal_path):
        normal_files = glob.glob(os.path.join(normal_path, "*.wav"))
        audio_paths.extend(normal_files)
        labels.extend([0] * len(normal_files))
        print(f"✅ Found {len(normal_files)} Normal samples")
    
    # Load stage1 depression samples (label 1)
    if os.path.exists(stage1_path):
        stage1_files = glob.glob(os.path.join(stage1_path, "*.wav"))
        audio_paths.extend(stage1_files)
        labels.extend([1] * len(stage1_files))
        print(f"✅ Found {len(stage1_files)} Stage1 Depression samples")
    
    # Load stage2 depression samples (label 2)
    if os.path.exists(stage2_path):
        stage2_files = glob.glob(os.path.join(stage2_path, "*.wav"))
        audio_paths.extend(stage2_files)
        labels.extend([2] * len(stage2_files))
        print(f"✅ Found {len(stage2_files)} Stage2 Depression samples")
    
    if len(audio_paths) == 0:
        raise ValueError("No audio files found. Please check your folder structure.")
    
    dataset_dict = {'audio_path': audio_paths, 'label': labels}
    dataset = Dataset.from_dict(dataset_dict)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Normal (0): {labels.count(0)} samples")
    print(f"   Stage1 (1): {labels.count(1)} samples") 
    print(f"   Stage2 (2): {labels.count(2)} samples")
    print(f"   Total: {len(labels)} samples")
    
    return dataset

# Load your dataset
print("Loading dataset...")
dataset = load_depression_dataset_flexible(
    base_dir="data/Corrected_Dataset"
)

print(f"\n✅ Dataset loaded with {len(dataset)} samples")

# Convert labels to ClassLabel
print("\nConverting labels to ClassLabel type...")
def ensure_int_labels(example):
    return {'label': int(example['label'])}

dataset = dataset.map(ensure_int_labels)
dataset = dataset.cast_column('label', ClassLabel(num_classes=3, names=['normal', 'stage1', 'stage2']))
print("✅ Labels converted to ClassLabel type")

# 3. Load processor and model
print("\nLoading wav2vec2 model for depression detection...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=3,
    problem_type="single_label_classification"
)

model.to(device)
print(f"✅ Model loaded and moved to {device}")

# 4. Preprocessing function with standardized audio length
def preprocess_function(examples):
    audio_arrays = []
    target_length = 80000  # 5 seconds at 16kHz
    
    for audio_path in examples["audio_path"]:
        try:
            audio_array, _ = librosa.load(audio_path, sr=16000)
            
            # Standardize length: pad or truncate to 5 seconds
            if len(audio_array) > target_length:
                audio_array = audio_array[:target_length]  # Truncate
            elif len(audio_array) < target_length:
                # Pad with zeros
                padding = target_length - len(audio_array)
                audio_array = np.pad(audio_array, (0, padding))
                
            audio_arrays.append(audio_array)
        except Exception as e:
            print(f"⚠️  Error loading {audio_path}: {e}")
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

# 5. Apply preprocessing
print("\nPreprocessing audio data...")
encoded_dataset = dataset.map(
    preprocess_function,
    batched=True,
    batch_size=4,
    remove_columns=['audio_path']
)

# Set format for PyTorch
encoded_dataset.set_format(type='torch', columns=['input_values', 'label'])

# Create data collator for dynamic padding - FIXED VERSION
data_collator = DataCollatorWithPadding(
    tokenizer=processor,  # Changed from 'processor' to 'tokenizer'
    padding=True,
    max_length=16000 * 10,
)

print("✅ Data collator created for dynamic padding")

# 6. Split dataset
print("\nSplitting dataset...")
train_test_split = encoded_dataset.train_test_split(
    test_size=0.2, 
    stratify_by_column="label",
    seed=42
)

train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

print(f"📈 Training samples: {len(train_dataset)}")
print(f"📊 Validation samples: {len(eval_dataset)}")

# Check class distribution in splits
print("\n📋 Class distribution in splits:")
train_labels = train_dataset['label']
eval_labels = eval_dataset['label']

train_labels_np = train_labels.numpy()
eval_labels_np = eval_labels.numpy()

print(f"Training - Normal: {np.sum(train_labels_np == 0)}, Stage1: {np.sum(train_labels_np == 1)}, Stage2: {np.sum(train_labels_np == 2)}")
print(f"Validation - Normal: {np.sum(eval_labels_np == 0)}, Stage1: {np.sum(eval_labels_np == 1)}, Stage2: {np.sum(eval_labels_np == 2)}")

# 7. Compute metrics function
def compute_metrics(eval_pred):
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

# 8. Training arguments
training_args = TrainingArguments(
    output_dir="./wav2vec2_depression_detector",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=30,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    push_to_hub=False,
    no_cuda=True,  # Force CPU
    dataloader_pin_memory=False,
    remove_unused_columns=False,
)

# 9. Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    data_collator=data_collator,
)

# 10. Train the model
print("\n" + "="*60)
print("🚀 Starting Training...")
print("="*60)
train_results = trainer.train()

# 11. Evaluate final model
print("\n" + "="*60)
print("📈 Evaluating Final Model...")
print("="*60)
eval_results = trainer.evaluate()
print("\n🎯 Final Evaluation Results:")
for key, value in eval_results.items():
    if isinstance(value, float):
        print(f"   {key}: {value:.4f}")
    else:
        print(f"   {key}: {value}")

# 12. Save the fine-tuned model
trainer.save_model("./final_depression_model")
processor.save_pretrained("./final_depression_model")
print("\n💾 Model saved to './final_depression_model'")

# 13. Feature extraction for your analysis
def extract_features_for_analysis(original_dataset, model, processor):
    """
    Extract features for your statistical analysis and visualization
    """
    model.eval()
    all_features = []
    all_labels = []
    
    print("\n" + "="*60)
    print("🔬 Extracting Features for Statistical Analysis...")
    print("="*60)
    
    with torch.no_grad():
        for i in range(len(original_dataset)):
            if i % 50 == 0:
                print(f"📊 Progress: {i}/{len(original_dataset)}")
                
            # Get audio path from ORIGINAL dataset
            audio_path = original_dataset[i]["audio_path"]
            label = original_dataset[i]["label"]
            
            try:
                # Load audio using librosa
                audio_array, _ = librosa.load(audio_path, sr=16000)
                
                # Standardize length
                target_length = 80000
                if len(audio_array) > target_length:
                    audio_array = audio_array[:target_length]
                elif len(audio_array) < target_length:
                    padding = target_length - len(audio_array)
                    audio_array = np.pad(audio_array, (0, padding))
                
                # Process audio
                inputs = processor(
                    audio_array, 
                    sampling_rate=16000, 
                    return_tensors="pt",
                    padding=True,
                    max_length=target_length,
                    truncation=True
                )
                
                # Move to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Extract features (before classification head)
                outputs = model.wav2vec2(**inputs)
                
                # Mean pooling over time dimension
                feature_vector = outputs.last_hidden_state.mean(dim=1)
                
                all_features.append(feature_vector.cpu().numpy())
                all_labels.append(label)
            except Exception as e:
                print(f"⚠️  Error processing {audio_path}: {e}")
                continue
    
    return np.vstack(all_features), np.array(all_labels)

print("\n🔬 Extracting features from original dataset...")
features, labels = extract_features_for_analysis(dataset, model, processor)

print(f"\n📐 Features shape: {features.shape}")
print(f"🏷️  Labels shape: {labels.shape}")

# 14. Save features for your thesis analysis
np.save("depression_features.npy", features)
np.save("depression_labels.npy", labels)

print("\n" + "="*60)
print("✅ FEATURE EXTRACTION COMPLETE!")
print("📊 Features saved for t-SNE, MANOVA, and statistical testing!")
print("="*60)

# 15. Print final summary
print("\n" + "="*60)
print("🎓 THESIS TRAINING COMPLETE - SUMMARY")
print("="*60)
print(f"💾 Model saved to: ./final_depression_model")
print(f"📊 Features saved to: depression_features.npy")
print(f"🏷️  Labels saved to: depression_labels.npy")
print(f"\n📈 Dataset info:")
print(f"   Total samples: {len(dataset)}")
print(f"   Training samples: {len(train_dataset)}")
print(f"   Validation samples: {len(eval_dataset)}")
print(f"\n🎯 Final performance:")
print(f"   Accuracy: {eval_results.get('eval_accuracy', 0):.4f}")
print(f"   F1 Score: {eval_results.get('eval_f1', 0):.4f}")
print(f"\n🔬 Feature dimensions: {features.shape}")
print("   (Ready for statistical analysis in your thesis!)")
print("\n" + "="*60)