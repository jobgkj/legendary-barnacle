"""
=============================================================================
Configuration — XCT Defect Detection Pipeline
=============================================================================
All hyperparameters, paths, and settings in one place.
Change values here only; do not hardcode elsewhere.
=============================================================================
"""

import os

# ---------------------------------------------------------------------------
# Paths for Storag and Usage
# ---------------------------------------------------------------------------
NIST_VOL_DIR    = "data/nist/volumes"
NIST_MASK_DIR   = "data/nist/masks"
PODFAM_VOL_DIR  = "data/podfam/volumes"
PODFAM_MASK_DIR = "data/podfam/masks"

CKPT_DIR        = "checkpoints"
MLFLOW_URI      = "mlruns"

os.makedirs(NIST_MASK_DIR,   exist_ok=True)
os.makedirs(PODFAM_MASK_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,        exist_ok=True)

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
NORM_LOW_PERCENTILE  = 1
NORM_HIGH_PERCENTILE = 99
NORM_EPS             = 1e-7

BHC_POLY_DEGREE = 3

NLM_PATCH_SIZE    = 5
NLM_PATCH_DIST    = 6
NLM_H             = 0.08

RING_FILTER_RADIUS = 15

# ---------------------------------------------------------------------------
# Pseudo-label generation
# ---------------------------------------------------------------------------
MORPH_OPEN_SIZE  = 3
MORPH_CLOSE_SIZE = 3
MIN_DEFECT_SIZE  = 5

# ---------------------------------------------------------------------------
# Patch extraction (2D)
# ---------------------------------------------------------------------------
PATCH_SIZE    = 256      # 2D patch size (H=W)
PATCH_STRIDE  = 128
FG_BG_RATIO   = (1, 3)
MIN_FG_PIXELS = 10

# ---------------------------------------------------------------------------
# Patch extraction (3D)
# ---------------------------------------------------------------------------
# Must be divisible by 2^len(ENCODER_CHANNELS) = 16
PATCH_SIZE_3D = (16, 128, 128)   # (D, H, W)

# ---------------------------------------------------------------------------
# Data augmentation (slice-wise)
# ---------------------------------------------------------------------------
AUG_FLIP_PROB       = 0.5
AUG_ROTATE_PROB     = 0.5
AUG_ELASTIC_PROB    = 0.3
AUG_ELASTIC_ALPHA   = 34
AUG_ELASTIC_SIGMA   = 4

AUG_INTENSITY_PROB  = 0.5
AUG_INTENSITY_RANGE = (0.9, 1.1)

AUG_NOISE_PROB      = 0.5
AUG_NOISE_STD_RANGE = (0.01, 0.05)

AUG_GAMMA_PROB      = 0.5
AUG_GAMMA_RANGE     = (0.8, 1.2)

# ---------------------------------------------------------------------------
# Model (shared by 2D and 3D)
# ---------------------------------------------------------------------------
ENCODER_CHANNELS = [64, 128, 256, 512]
DROPOUT_RATE     = 0.2

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
DEVICE = "cuda"  # or "cpu"

# Separate batch sizes (CRITICAL)
BATCH_SIZE_2D = 8
BATCH_SIZE_3D = 1

NUM_EPOCHS    = 50
LEARNING_RATE = 4e-5
WEIGHT_DECAY  = 1e-5

VAL_SPLIT  = 0.2
TEST_SPLIT = 0.1

LOSS_FUNCTION      = "dice_focal"   # "bce", "dice", "focal", "dice_focal"
DICE_FOCAL_LAMBDA  = 0.5
FOCAL_ALPHA        = 0.25
FOCAL_GAMMA        = 2.0

EARLY_STOP_PATIENCE = 10
SCHEDULER_PATIENCE  = 5

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
DICE_THRESHOLD  = 0.5
ACCEPTANCE_DICE = 0.75
ACCEPTANCE_IOU  = 0.60
ACCEPTANCE_REC  = 0.80

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_EXPERIMENT = "XCT_Defect_Detection"
