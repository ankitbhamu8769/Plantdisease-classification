"""
config.py
---------
Central configuration for PlantVillage classification pipeline.
"""

import os
import torch

# ---------------------------------------------------------------------------
# Paths (Kaggle Specific)
# ---------------------------------------------------------------------------
PROJECT_ROOT = "/kaggle/working/Archive"
RAW_DATA_DIR = "/kaggle/input/plantvillage-dataset" 

DATA_DIR = "/kaggle/working/data"           
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

CHECKPOINT_DIR = "/kaggle/working/checkpoints"
OUTPUT_DIR = "/kaggle/working/outputs"       
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
LATEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "latest_model.pth")
RESUME_TRAINING = True

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

USE_PIN_MEMORY = (DEVICE.type == "cuda")

# ---------------------------------------------------------------------------
# Data / Training Hyperparameters
# ---------------------------------------------------------------------------
IMG_SIZE = 224
BATCH_SIZE = 32

NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DROPOUT = 0.3
NUM_CLASSES = 38

# ---------------------------------------------------------------------------
# Imbalance Handling
# ---------------------------------------------------------------------------
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.1
USE_CLASS_ALPHA = True

# Fast, native PyTorch class balancing:
# Set USE_WEIGHTED_SAMPLER = True for zero-RAM, zero-latency class-balanced epochs.
USE_WEIGHTED_SAMPLER = True

# Keep False to prevent Kaggle 30GB RAM OOM crash.
# Set True only if you specifically want to run memory-capped latent SMOTE.
USE_SMOTE = False

# ---------------------------------------------------------------------------
# Model Architecture Switches
# ---------------------------------------------------------------------------
BACKBONE_NAME = "tf_efficientnetv2_b0"
FREEZE_BACKBONE_STAGES = 3

NUM_GROUPS = 4
TRANSFORMER_HEADS = 4
TRANSFORMER_DEPTH = 2
ATTENTION_TYPE = "triplet"

# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
GRADCAM_MID_LAYER_STAGE = -4
GRADCAM_LATE_LAYER_STAGE = -2
NUM_LIME_SAMPLES = 8

# ---------------------------------------------------------------------------
# Training Pipeline & Fast Crop Configuration
# ---------------------------------------------------------------------------
UNFREEZE_EPOCH = 25
FINE_TUNE_LEARNING_RATE = 1e-5

USE_CV2_CROP = True
NUM_WORKERS = 4