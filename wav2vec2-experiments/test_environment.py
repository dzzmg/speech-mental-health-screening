# test_environment.py
# activate the mental-health-audio environment

import torch
import torchaudio
import transformers
import librosa
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=== Environment Test ===")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Transformers: {transformers.__version__}")
print(f"Librosa: {librosa.__version__}")
print(f"Sklearn: {sklearn.__version__}")

# Test basic functionality
print("\n=== Basic Functionality Tests ===")
# Test audio processing
dummy_audio = np.random.random(16000)
mfcc = librosa.feature.mfcc(y=dummy_audio, sr=16000, n_mfcc=13)
print(f"✓ MFCC features shape: {mfcc.shape}")

# Test PyTorch
x = torch.tensor([1, 2, 3])
print(f"✓ PyTorch tensor: {x}")

print("🎉 Environment is ready!")