"""
Configuration file for ASL Fairness Benchmark
OPTIMIZED FOR M4 MAC
"""
import os
from pathlib import Path
import torch

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"
MODELS_DIR = PROJECT_ROOT / "saved_models"

# Create directories
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_URL = "https://www.kaggle.com/datasets/grassknoted/asl-alphabet"
IMG_SIZE = 224
BATCH_SIZE = 32  # M4 can handle larger batches
NUM_WORKERS = 0  # Mac doesn't need multiprocessing for data loading

# ASL Classes
ASL_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'space', 'del', 'nothing'
]
NUM_CLASSES = len(ASL_CLASSES)

# M4 DEVICE CONFIGURATION
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("Using M4 Neural Engine (MPS) - FAST AS HELL")
else:
    DEVICE = "cpu"
    print("MPS not available, falling back to CPU")

# Training configuration - OPTIMIZED FOR M4
EPOCHS = 3  # M4 can handle this easily
LEARNING_RATE = 0.001

# Experiment configuration
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2]
OCCLUSION_SIZES = [0, 20, 40, 60, 80]
BRIGHTNESS_FACTORS = [0.5, 0.75, 1.0, 1.25, 1.5]

# Model paths
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.pkl"
CLASSICAL_MODEL_PATH = MODELS_DIR / "classical_model.pkl"
DEEP_MODEL_PATH = MODELS_DIR / "deep_model.pth"

print(f"M4 Configuration loaded. Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}")