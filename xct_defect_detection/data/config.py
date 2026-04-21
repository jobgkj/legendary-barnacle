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
# Paths
# ---------------------------------------------------------------------------
NIST_VOL_DIR    = "data/nist/volumes"       # NIST XCT TIFF stacks
NIST_MASK_DIR   = "data/nist/masks"         # Will be created by pseudo-label generator
PODFAM_VOL_DIR  = "data/podfam/volumes"     # PODFAM XCT TIFF stacks
PODFAM_MASK_DIR = "data/podfam/masks"       # Will be created by pseudo-label generator
CKPT_DIR        = "checkpoints"             # Model checkpoint save directory
MLFLOW_URI      = "mlruns"                  # MLflow tracking directory

os.makedirs(NIST_MASK_DIR,   exist_ok=True)
os.makedirs(PODFAM_MASK_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,        exist_ok=True)

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
NORM_LOW_PERCENTILE  = 1            # Lower percentile for intensity clipping
NORM_HIGH_PERCENTILE = 99           # Upper percentile for intensity clipping
NORM_EPS             = 1e-7         # Avoid division by zero

# Beam hardening correction polynomial degree
BHC_POLY_DEGREE = 3

# Non-local means denoising
NLM_PATCH_SIZE    = 5               # Patch size for similarity comparison
NLM_PATCH_DIST    = 6               # Search window radius
NLM_H             = 0.08            # Filter strength (higher = more smoothing)

# Ring artefact suppression
RING_FILTER_RADIUS = 15             # Polar median filter radius (pixels)

# ---------------------------------------------------------------------------
# Pseudo-label generation
# ---------------------------------------------------------------------------
MORPH_OPEN_SIZE  = 3                # Structuring element for morphological opening
MORPH_CLOSE_SIZE = 3                # Structuring element for morphological closing
MIN_DEFECT_SIZE  = 5                # Minimum connected component size (voxels)

# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------
PATCH_SIZE          = 256           # 2D patch size (pixels)
PATCH_STRIDE        = 128           # Stride for sliding window extraction
FG_BG_RATIO         = (1, 3)        # Foreground:background patch sampling ratio
MIN_FG_PIXELS       = 10           # Minimum defect pixels for a patch to be
                                    # counted as foreground

# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------
AUG_FLIP_PROB       = 0.5           # Probability of horizontal/vertical flip
AUG_ROTATE_PROB     = 0.5           # Probability of 90-degree rotation
AUG_ELASTIC_PROB    = 0.3           # Probability of elastic deformation
AUG_ELASTIC_ALPHA   = 34            # Elastic deformation magnitude
AUG_ELASTIC_SIGMA   = 4             # Elastic deformation smoothness
AUG_INTENSITY_PROB  = 0.5           # Probability of intensity scaling
AUG_INTENSITY_RANGE = (0.9, 1.1)    # Intensity scaling range
AUG_NOISE_PROB      = 0.5           # Probability of Gaussian noise
AUG_NOISE_STD_RANGE = (0.01, 0.05)  # Gaussian noise std range
AUG_GAMMA_PROB      = 0.5           # Probability of gamma correction
AUG_GAMMA_RANGE     = (0.8, 1.2)    # Gamma correction range

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
ENCODER_CHANNELS    = [64, 128, 256, 512]   # 2D U-Net encoder feature channels
DROPOUT_RATE        = 0.2                   # Dropout probability in bottleneck

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
DEVICE          = "cuda"            # "cuda" or "cpu"
BATCH_SIZE      = 8                 # Training batch size
NUM_EPOCHS      = 50                # Total training epochs
LEARNING_RATE   = 4e-5              # Adam learning rate (from reference thesis)
WEIGHT_DECAY    = 1e-5              # Adam weight decay
VAL_SPLIT       = 0.2              # Fraction of data reserved for validation
TEST_SPLIT      = 0.1              # Fraction of data reserved for testing
LOSS_FUNCTION   = "dice_focal"      # Options: "bce", "dice", "focal", "dice_focal"
DICE_FOCAL_LAMBDA = 0.5             # Weight of Dice in combined loss
FOCAL_ALPHA     = 0.25              # Focal loss alpha
FOCAL_GAMMA     = 2.0               # Focal loss gamma
EARLY_STOP_PATIENCE = 10            # Epochs without improvement before stopping
SCHEDULER_PATIENCE  = 5            # Epochs before LR reduction

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
DICE_THRESHOLD  = 0.5               # Probability threshold for binary prediction
ACCEPTANCE_DICE = 0.75              # Minimum acceptable Dice (thesis criterion)
ACCEPTANCE_IOU  = 0.60              # Minimum acceptable IoU
ACCEPTANCE_REC  = 0.80              # Minimum acceptable Recall

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_EXPERIMENT = "XCT_Defect_Detection"
