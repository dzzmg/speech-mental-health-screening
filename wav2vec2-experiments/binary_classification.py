# binary_depression_classifier.py - 2-Class: Normal vs Depression
import torch
import librosa
import numpy as np
from transformers import (
    Wav2Vec2Processor, Wav2Vec2ForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from datasets import Dataset
from collections import Counter
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import warnings
import datetime
import os
import glob
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
warnings.filterwarnings('ignore')

# ============================================================
# TENSORBOARD CALLBACK
# ============================================================
class SimpleTensorBoardCallback(TrainerCallback):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = None
        print(f"🚀 TensorBoard will log to: {log_dir}")
        
    def on_train_begin(self, args, state, control, **kwargs):
        try:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            self.writer.add_scalar('test/initialization', 1.0, 0)
            self.writer.flush()
            print("✅ TensorBoard initialized!")
        except Exception as e:
            print(f"⚠️ TensorBoard error: {e}")
            
    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.writer is not None and logs is not None:
            try:
                for key, value in logs.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f"train/{key}", value, state.global_step)
                self.writer.flush()
            except:
                pass
            
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.writer is not None and metrics is not None:
            try:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        clean_key = key.replace("eval_", "")
                        self.writer.add_scalar(f"eval/{clean_key}", value, state.global_step)
                self.writer.flush()
            except:
                pass
            
    def on_train_end(self, args, state, control, **kwargs):
        if self.writer is not None:
            try:
                self.writer.close()
                print(f"✅ TensorBoard logs saved to: {self.log_dir}")
            except:
                pass

def compute_metrics(eval_pred):
    """Compute metrics for binary classification"""
    predictions, labels = eval_pred
    
    # Get probabilities for ROC-AUC
    probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=1)[:, 1].numpy()
    
    # Get class predictions
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary', zero_division=0)
    
    # Calculate ROC-AUC
    try:
        roc_auc = roc_auc_score(labels, probs)
    except:
        roc_auc = 0.0
    
    cm = confusion_matrix(labels, predictions)
    print("\n🔍 Confusion Matrix:")
    print("         Predicted:")
    print("        Normal  Depression")
    label_names = ['Normal', 'Depression']
    for i, row in enumerate(cm):
        print(f"  {label_names[i]:10s}:  {row[0]:6d}  {row[1]:6d}")
    
    # Calculate precision and recall
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    print(f"\n🎯 METRICS:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   ROC-AUC:   {roc_auc:.4f}")
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc
    }

class BinaryDepressionTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def gentle_audio_augmentation(self, audio_array, sr=16000):
        """Very gentle augmentation"""
        augmented = audio_array.copy()
        
        if np.random.random() > 0.8:
            rate = np.random.uniform(0.95, 1.05)
            try:
                augmented = librosa.effects.time_stretch(augmented, rate=rate)
            except:
                pass
        
        if np.random.random() > 0.8:
            n_steps = np.random.choice([-0.5, 0.5])
            try:
                augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=n_steps)
            except:
                pass
        
        return augmented
    
    def balanced_oversample(self, indices_by_class, target_ratio=0.9):
        """Oversample to target_ratio of max class"""
        max_count = max(len(indices) for indices in indices_by_class.values())
        target_count = int(max_count * target_ratio)
        
        print(f"🔄 Oversampling to {target_ratio*100:.0f}% of max class ({target_count})...")
        
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

class BinaryWeightedTrainer(Trainer):
    def __init__(self, class_weights=None, label_smoothing=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Binary loss with label smoothing"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Label smoothing for binary classification
        if self.label_smoothing > 0:
            num_classes = 2
            labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
            labels_smooth = labels_one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes
            
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(labels_smooth * log_probs).sum(dim=-1)
            
            if self.class_weights is not None:
                weights = self.class_weights[labels]
                loss = (loss * weights).mean()
            else:
                loss = loss.mean()
        else:
            if self.class_weights is not None:
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def load_binary_depression_dataset(base_dir):
    """Load dataset and convert to binary classification"""
    audio_paths = []
    labels = []
    original_labels = []  # Keep track of original 3-class labels
    
    normal_path = os.path.join(base_dir, "Normal")
    stage1_path = os.path.join(base_dir, "Depression", "Stage1")
    stage2_path = os.path.join(base_dir, "Depression", "Stage2")
    
    # Class 0: Normal
    if os.path.exists(normal_path):
        normal_files = glob.glob(os.path.join(normal_path, "*.wav"))
        audio_paths.extend(normal_files)
        labels.extend([0] * len(normal_files))  # 0 = Normal
        original_labels.extend(['Normal'] * len(normal_files))
        print(f"✅ Normal: {len(normal_files)} samples → Class 0")
    
    # Class 1: Depression (combining Stage1 + Stage2)
    depression_count = 0
    if os.path.exists(stage1_path):
        stage1_files = glob.glob(os.path.join(stage1_path, "*.wav"))
        audio_paths.extend(stage1_files)
        labels.extend([1] * len(stage1_files))  # 1 = Depression
        original_labels.extend(['Stage1'] * len(stage1_files))
        depression_count += len(stage1_files)
        print(f"✅ Stage1: {len(stage1_files)} samples → Class 1 (Depression)")
    
    if os.path.exists(stage2_path):
        stage2_files = glob.glob(os.path.join(stage2_path, "*.wav"))
        audio_paths.extend(stage2_files)
        labels.extend([1] * len(stage2_files))  # 1 = Depression
        original_labels.extend(['Stage2'] * len(stage2_files))
        depression_count += len(stage2_files)
        print(f"✅ Stage2: {len(stage2_files)} samples → Class 1 (Depression)")
    
    if len(audio_paths) == 0:
        raise ValueError("No audio files found.")
    
    dataset_dict = {
        'audio_path': audio_paths, 
        'label': labels,
        'original_label': original_labels
    }
    dataset = Dataset.from_dict(dataset_dict)
    
    print(f"\n📊 BINARY CLASSIFICATION DATASET:")
    print(f"   Class 0 (Normal):     {labels.count(0)} samples")
    print(f"   Class 1 (Depression): {labels.count(1)} samples")
    print(f"   Total:                {len(labels)} samples")
    print(f"\n   Depression breakdown:")
    print(f"     - Stage1: {original_labels.count('Stage1')} samples")
    print(f"     - Stage2: {original_labels.count('Stage2')} samples")
    
    return dataset

def binary_fine_tuning(dataset, labels):
    """Binary classification: Normal vs Depression"""
    
    trainer_helper = BinaryDepressionTrainer()
    
    # ============================================================
    # STEP 1: Setup
    # ============================================================
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./tensorboard_logs/binary_model_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n📊 TensorBoard logs: {log_dir}")
    
    # ============================================================
    # STEP 2: Split dataset
    # ============================================================
    print("\n" + "="*60)
    print("📊 SPLITTING DATASET")
    print("="*60)
    
    indices = list(range(len(dataset)))
    
    # Split: 70% train, 15% val, 15% test
    train_val_indices, test_indices = train_test_split(
        indices, test_size=0.15, stratify=labels, random_state=42
    )
    
    train_val_labels = [labels[i] for i in train_val_indices]
    train_indices, val_indices = train_test_split(
        train_val_indices, 
        test_size=0.18,
        stratify=train_val_labels, 
        random_state=42
    )
    
    train_labels = [labels[i] for i in train_indices]
    val_labels = [labels[i] for i in val_indices]
    test_labels = [labels[i] for i in test_indices]
    
    print(f"Training: {len(train_indices)} samples")
    print(f"  Class distribution: {Counter(train_labels)}")
    print(f"Validation: {len(val_indices)} samples")
    print(f"  Class distribution: {Counter(val_labels)}")
    print(f"Test: {len(test_indices)} samples")
    print(f"  Class distribution: {Counter(test_labels)}")
    
    np.save("test_indices_binary.npy", test_indices)
    
    # ============================================================
    # STEP 3: Balanced oversampling
    # ============================================================
    print("\n" + "="*60)
    print("📊 BALANCED OVERSAMPLING")
    print("="*60)
    
    train_indices_by_class = {}
    for idx in train_indices:
        label = labels[idx]
        if label not in train_indices_by_class:
            train_indices_by_class[label] = []
        train_indices_by_class[label].append(idx)
    
    balanced_train_indices = trainer_helper.balanced_oversample(train_indices_by_class, target_ratio=0.95)
    
    print(f"Final training size: {len(balanced_train_indices)}")
    balanced_train_labels = [labels[i] for i in balanced_train_indices]
    print(f"Balanced distribution: {Counter(balanced_train_labels)}")
    
    # ============================================================
    # STEP 4: Create datasets
    # ============================================================
    train_dataset = dataset.select(balanced_train_indices)
    val_dataset = dataset.select(val_indices)
    test_dataset = dataset.select(test_indices)
    
    # ============================================================
    # STEP 5: Load model for BINARY classification
    # ============================================================
    print("\n" + "="*60)
    print("📊 LOADING BINARY MODEL")
    print("="*60)
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base",
        num_labels=2,  # BINARY: Normal vs Depression
        problem_type="single_label_classification",
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.1,
        final_dropout=0.15,
        classifier_proj_size=256,
    )
    
    # Simple initialization
    print("🎯 Initializing binary classifier...")
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(model.classifier.weight)
        model.classifier.bias.data.zero_()
        
        print("✅ Initialization complete:")
        for i in range(2):
            print(f"  Class {i}: weight_norm={torch.norm(model.classifier.weight[i]):.3f}, "
                  f"bias={model.classifier.bias[i].item():.3f}")
    
    model.to(trainer_helper.device)
    
    # ============================================================
    # STEP 6: Calculate class weights
    # ============================================================
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    
    # Moderate clipping
    class_weights_array = np.clip(class_weights_array, 0.5, 2.0)
    class_weights = torch.tensor(class_weights_array, dtype=torch.float32).to(trainer_helper.device)
    
    print(f"\n📊 Class weights: {class_weights_array}")
    
    # ============================================================
    # STEP 7: Preprocessing
    # ============================================================
    def preprocess_function(examples, augment=False):
        audio_arrays = []
        target_length = 80000
        
        for audio_path in examples["audio_path"]:
            try:
                audio_array, _ = librosa.load(audio_path, sr=16000)
                
                if augment and np.random.random() > 0.6:
                    audio_array = trainer_helper.gentle_audio_augmentation(audio_array)
                
                if len(audio_array) > target_length:
                    audio_array = audio_array[:target_length]
                elif len(audio_array) < target_length:
                    padding = target_length - len(audio_array)
                    audio_array = np.pad(audio_array, (0, padding))
                    
                audio_arrays.append(audio_array)
            except Exception as e:
                print(f"⚠️ Error: {e}")
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
    
    print("\n🔄 Preprocessing datasets...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, augment=True),
        batched=True,
        batch_size=4,
        remove_columns=['audio_path', 'original_label']
    )
    
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, augment=False),
        batched=True,
        batch_size=4,
        remove_columns=['audio_path', 'original_label']
    )
    
    train_dataset.set_format(type='torch', columns=['input_values', 'label'])
    val_dataset.set_format(type='torch', columns=['input_values', 'label'])
    
    # ============================================================
    # STEP 8: Training arguments for BINARY classification
    # ============================================================
    training_args = TrainingArguments(
        output_dir="./binary_depression_model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        
        # Moderate learning rate for binary task
        learning_rate=5e-6,
        
        # Batch settings
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        
        # Epochs
        num_train_epochs=30,  # More epochs for binary (simpler task)
        
        # Regularization
        weight_decay=0.01,
        
        # Warmup
        warmup_ratio=0.15,
        
        # Logging
        logging_steps=10,
        eval_steps=50,
        save_steps=50,
        
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        
        # TensorBoard
        logging_dir=log_dir,
        report_to=["tensorboard"],
        
        # Gradient clipping
        max_grad_norm=1.0,
        
        # Other settings
        fp16=False,
        dataloader_num_workers=0,
        no_cuda=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # ============================================================
    # STEP 9: Create trainer
    # ============================================================
    tensorboard_callback = SimpleTensorBoardCallback(log_dir=log_dir)
    
    trainer = BinaryWeightedTrainer(
        class_weights=class_weights,
        label_smoothing=0.1,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            tensorboard_callback,
            EarlyStoppingCallback(early_stopping_patience=5)
        ]
    )
    
    # ============================================================
    # STEP 10: Initial check
    # ============================================================
    print("\n" + "="*60)
    print("🔍 INITIAL MODEL CHECK")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        test_preds = []
        for i in range(min(30, len(train_dataset))):
            batch = train_dataset[i]
            inputs = {k: v.unsqueeze(0).to(trainer_helper.device) 
                     for k, v in batch.items() if k != 'label'}
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            test_preds.append(pred)
        
        dist = Counter(test_preds)
        print(f"📊 Initial predictions: {dict(dist)}")
        print(f"🎯 Using {len(dist)} different classes")
        
        if len(dist) == 2:
            print("✅ Model is using BOTH classes from the start!")
        else:
            print("⚠️ Warning: Model only using 1 class initially")
    
    # ============================================================
    # STEP 11: Start training
    # ============================================================
    print("\n" + "="*60)
    print("🚀 STARTING BINARY CLASSIFICATION TRAINING")
    print("="*60)
    print(f"📊 TensorBoard: tensorboard --logdir '{os.path.abspath('./tensorboard_logs')}'")
    print("="*60)
    
    trainer.train()
    
    # ============================================================
    # STEP 12: Final evaluation
    # ============================================================
    print("\n" + "="*60)
    print("🎯 FINAL EVALUATION ON VALIDATION SET")
    print("="*60)
    
    eval_results = trainer.evaluate()
    
    print("\n📊 Final Validation Results:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
    
    return trainer, processor, test_dataset, log_dir

# Usage
if __name__ == "__main__":
    dataset = load_binary_depression_dataset(
        "data/Corrected_Dataset"
    )
    labels = [dataset[i]['label'] for i in range(len(dataset))]
    
    print(f"\n📁 Total dataset: {len(dataset)} samples")
    print(f"📊 Binary distribution: {Counter(labels)}")
    
    # Run binary training
    trainer, processor, test_dataset, log_dir = binary_fine_tuning(dataset, labels)
    
    # Save model
    trainer.save_model("./binary_depression_model")
    processor.save_pretrained("./binary_depression_model")
    
    print("\n" + "="*60)
    print("🎉 BINARY CLASSIFICATION TRAINING COMPLETE!")
    print("="*60)
    print("💾 Model saved to: ./binary_depression_model")
    print(f"📊 TensorBoard logs: {log_dir}")
    print("🧪 Test set saved: test_indices_binary.npy")
    print("\n📋 Model predicts:")
    print("   Class 0: Normal (healthy)")
    print("   Class 1: Depression (Stage1 + Stage2 combined)")
    print("="*60)