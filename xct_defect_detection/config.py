"""
=============================================================================
Configuration — XCT Defect Detection Pipeline
=============================================================================
All hyperparameters, paths, and settings in one place.
Change values here only; do not hardcode elsewhere.
=============================================================================
"""

import os
from pathlib import Path

# =============================================================================
# Repository root
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent

# =============================================================================
# Sample
# =============================================================================

# Default single sample — used by io.py / run_pipeline.py
SAMPLE_NAME = "sample_01"

# All samples — auto-discovered at import time
SAMPLE_NAMES = sorted([
    d.name for d in (REPO_ROOT / "data" / "raw").iterdir()
    if d.is_dir()
]) if (REPO_ROOT / "data" / "raw").exists() else []

# =============================================================================
# Training data selection — the two knobs to turn when you want to
# change what pipeline.py trains on and run it again later
# =============================================================================
#
# By default (TRAINING_SAMPLES_OVERRIDE = None), pipeline.py auto-ranks
# every discovered sample by defect (void) content and keeps the top
# MAX_TRAINING_SAMPLES most defect-rich ones — training on near-empty
# samples wastes most of every patch on background, and this also
# guards against known-bad samples (see SUSPICIOUS_DEFECT_FRACTION in
# pipeline.py). Needs at least 3 samples for a train/val/test split.
#
# To force a *specific* set of samples instead of auto-ranking — the
# fastest way to try a different data mix without touching any other
# code — set TRAINING_SAMPLES_OVERRIDE to an explicit list, most-voids-
# first (the first entries become train, the last become val/test; see
# pipeline.py::split_volumes):
#
#     TRAINING_SAMPLES_OVERRIDE = ["sample_02", "sample_04", "sample_03"]
#
# Every time you change either knob, also bump EXPERIMENT_NAME below —
# that keeps this run's checkpoint and prediction outputs from
# overwriting a previous run's, so you can compare them side by side
# later instead of only ever having "the latest" result.
TRAINING_SAMPLES_OVERRIDE = ["sample_02", "sample_04", "sample_03", "sample_01"]
MAX_TRAINING_SAMPLES      = 6

# Identifies one training configuration. Used to name the saved
# checkpoint (artifacts/best_model_<EXPERIMENT_NAME>.pt) and to
# namespace scripts/predict_and_visualize_3d.py's output folders/files
# — change this whenever you change TRAINING_SAMPLES_OVERRIDE or
# MAX_TRAINING_SAMPLES so old and new results don't collide.
#
# Loss-function ablation (RQ4): TRAINING_SAMPLES_OVERRIDE is pinned to
# the exact same 4 samples as the "top6_by_voids" (dice_focal) run so
# these experiments are a controlled comparison — only LOSS_FUNCTION
# (below) differs between them.
EXPERIMENT_NAME = os.environ.get("XCT_EXPERIMENT_NAME", "loss_focal")

# Skip the 3D U-Net training stage in pipeline.py — 2D-only for now.
SKIP_3D_TRAINING = True

# =============================================================================
# Data directories
# =============================================================================

RAW_DATA_DIR       = REPO_ROOT / "data" / "raw"
# XCT_PROCESSED_DIR / XCT_MASKS_DIR let a run point at an alternate
# preprocessed-data tree (e.g. data/processed_bhc_ring) without changing
# the defaults every other script in this repo relies on.
PROCESSED_DATA_DIR = Path(os.environ.get("XCT_PROCESSED_DIR", str(REPO_ROOT / "data" / "processed")))
MASKS_DIR          = Path(os.environ.get("XCT_MASKS_DIR", str(REPO_ROOT / "data" / "masks")))

# =============================================================================
# Results directories
# =============================================================================

CKPT_DIR    = REPO_ROOT / "artifacts"
FIGURES_DIR = REPO_ROOT / "results" / "figures"
METRICS_DIR = REPO_ROOT / "results" / "metrics"
# XCT_CACHE_DIR mirrors XCT_PROCESSED_DIR/XCT_MASKS_DIR: data/cache.py's
# build_cache() reuses an existing <sample>_volume.npy purely by sample
# name, with no check against which PROCESSED_DATA_DIR/MASKS_DIR built
# it — so a run against an alternate preprocessed-data tree MUST also
# point CACHE_DIR somewhere new, or it will silently reuse whatever was
# cached from the default data/processed/ tree instead of rebuilding.
CACHE_DIR   = Path(os.environ.get("XCT_CACHE_DIR", str(REPO_ROOT / "data" / "cache")))

# =============================================================================
# Directory creation — call explicitly at pipeline startup
# =============================================================================

def create_dirs() -> None:
    """
    Create all required output directories.
    Call once at the top of any pipeline script — not at import time.
    """
    for d in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MASKS_DIR,
        CKPT_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        CACHE_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Preprocessing
# =============================================================================

NORM_LOW_PERCENTILE  = 1
NORM_HIGH_PERCENTILE = 99
NORM_EPS             = 1e-7

# Number of evenly-spaced raw slices sampled to estimate a STACK-WIDE
# (p1, p99) normalisation range, instead of each slice computing its
# own range independently. Per-slice normalisation stretches whatever
# raw intensity spread that one slice happens to have to fill the full
# output range — fine for a well-illuminated slice, but for a thin- or
# low-density-cross-section slice (e.g. near the top/bottom of a scan,
# where material tapers off) it takes a genuinely low-contrast signal
# and force-stretches it to look like full-strength contrast, which
# Bernsen then reads as spurious defect boundaries. A shared stack-wide
# range keeps a genuinely low-signal slice looking low-signal relative
# to the rest of the stack, rather than artificially equalised to
# match the well-illuminated middle.
STACK_NORM_N_SAMPLE_SLICES = 20

MEDIAN_KERNEL_SIZE   = 3

BHC_POLY_DEGREE      = 3

NLM_PATCH_SIZE       = 5
NLM_PATCH_DIST       = 6
NLM_H_FACTOR         = 0.6    # h = NLM_H_FACTOR * estimated_sigma (adaptive)

RING_FILTER_RADIUS   = 15

# =============================================================================
# Thresholding
# =============================================================================

BERNSEN_RADIUS              = 5      # local window radius in pixels (Kim et al. 2017)
BERNSEN_LOW_CONTRAST_THRESH = 128    # fixed fallback — only used when BERNSEN_LOW_CONTRAST_ADAPTIVE = False

# ------------------------------------------------------------------
# Adaptive low-contrast fallback threshold
# ------------------------------------------------------------------
# BERNSEN_LOW_CONTRAST_THRESH (128) is a single value assumed to work
# across every sample's brightness range. In practice DCT (below) comes
# out far higher than Kim et al.'s reported ~15 across every sample
# measured so far — which routes almost every pixel through this
# fallback rather than genuine local adaptive thresholding — and 128
# only happens to be a reasonable split point for brighter-averaging
# samples. Darker-averaging samples (median well under 128) have most
# of their genuinely solid material misclassified as pore: measured
# directly, samples with median intensity ~35-73 showed 50%+ "defect"
# fraction from this alone, even after sample-mask background removal.
#
# When True, the fallback threshold is instead computed per image via
# src.thresholding.compute_low_contrast_threshold() — adapting to each
# sample's own brightness distribution instead of assuming one global
# constant fits every scan.
#
# UPDATE: originally left OFF by default because a raw Otsu split was
# measurably regressing already-fine samples (a near-zero-porosity
# sample going from ~0% to ~20% "defect" fraction) — Otsu is biased
# when one class vastly outnumbers the other (solid vastly outnumbers
# pore in-mask), so the split landed right next to the solid phase's
# own peak, misclassifying the normal partial-volume dip near every
# edge as pore (a thick false-positive halo hugging the whole
# boundary). compute_low_contrast_threshold() no longer returns the
# raw split — it anchors to the solid phase's own histogram PEAK and
# backs off by BERNSEN_LOW_CONTRAST_MARGIN, verified to reproduce
# fixed=128's known-good behaviour closely on bright-solid samples
# while still adapting correctly for a genuinely dark-averaging one.
# Turned back ON now that the underlying computation is fixed.
BERNSEN_LOW_CONTRAST_ADAPTIVE = True

BERNSEN_LOW_CONTRAST_MARGIN = 100   # intensity units to back off below the
                                     # solid phase's own histogram peak when
                                     # computing the adaptive low-contrast
                                     # threshold. Calibrated so bright-solid
                                     # samples (peak ~190-235) land close to
                                     # the old fixed=128 default; scales down
                                     # automatically for a lower solid peak.

# ------------------------------------------------------------------
# Auto-DCT computation (Kim et al. 2017 method)
# ------------------------------------------------------------------
# DCT is computed per image as:
#     DCT = BERNSEN_DCT_STD_MULTIPLIER × mean(local std of solid phase)
#
# The paper measured avg std = 0.847 for Sample 2 after NLM filtering,
# giving DCT = 18 × 0.847 ≈ 15.
# For noisier images the std will be higher → higher DCT → less over-segmentation.
#
# Set BERNSEN_DCT_AUTO = False and adjust BERNSEN_DCT to use a fixed value.
# ------------------------------------------------------------------

BERNSEN_DCT_AUTO           = True   # True = compute per image (recommended)
BERNSEN_DCT_STD_MULTIPLIER = 18     # from Kim et al. 2017 (18× avg std)
BERNSEN_DCT_N_LOCATIONS    = 5      # grid points per axis for std sampling
BERNSEN_DCT_WINDOW_RADIUS  = 5      # local window radius for std measurement
BERNSEN_DCT_MIN            = 5      # minimum DCT floor (prevents under-segmentation)

# Fallback fixed DCT — only used when BERNSEN_DCT_AUTO = False
# Raised from 15 → 30 to reduce over-segmentation on low-porosity samples
BERNSEN_DCT                = 30

# =============================================================================
# Metrics
# =============================================================================

PIXEL_SIZE_UM   = 1.0   # µm per pixel — update with your scanner's voxel size
MIN_DEFECT_SIZE = 5     # minimum pore area in pixels

# =============================================================================
# Pseudo-label generation
# =============================================================================

MORPH_OPEN_SIZE  = 3
MORPH_CLOSE_SIZE = 3

# =============================================================================
# Patch extraction (2D)
# =============================================================================

PATCH_SIZE    = 256
PATCH_STRIDE  = 128
FG_BG_RATIO   = (1, 3)
MIN_FG_PIXELS = 10

# 2.5D input: number of adjacent slices stacked as input channels around
# the centre slice, giving the 2D U-Net some volumetric context without
# the memory/compute cost of true 3D convolutions (see SKIP_3D_TRAINING
# above). Must be odd — the model still predicts a mask for the centre
# slice only; its (N//2) neighbours above and below are extra context
# channels. Neighbours past a volume's edge are clamped (edge slice
# repeated) rather than zero-padded. Set to 1 to fall back to the
# original single-slice 2D behaviour.
UNET_INPUT_SLICES = int(os.environ.get("XCT_UNET_INPUT_SLICES", "3"))

# Evenly-spaced slices sampled per volume for the 2D dataset, instead of
# scanning every slice. Full volumes run ~900 slices each; with the low
# RAM ceiling on this machine (see data/dataset.py's slice cache), an
# exhaustive scan makes each epoch take 3+ hours. Sampling keeps epoch
# time practical at the cost of not seeing every slice each epoch.
TRAIN_SLICES_PER_VOLUME = 80

# =============================================================================
# Patch extraction (3D)
# =============================================================================

PATCH_SIZE_3D = (16, 128, 128)   # (D, H, W) — must be divisible by 16

# =============================================================================
# Data augmentation
# =============================================================================

AUG_FLIP_PROB        = 0.5
AUG_ROTATE_PROB      = 0.5
AUG_ELASTIC_PROB     = 0.3
AUG_ELASTIC_ALPHA    = 34
AUG_ELASTIC_SIGMA    = 4

AUG_INTENSITY_PROB   = 0.5
AUG_INTENSITY_RANGE  = (0.9, 1.1)

AUG_NOISE_PROB       = 0.5
AUG_NOISE_STD_RANGE  = (0.01, 0.05)

AUG_GAMMA_PROB       = 0.5
AUG_GAMMA_RANGE      = (0.8, 1.2)

# =============================================================================
# Model (shared by 2D and 3D)
# =============================================================================

ENCODER_CHANNELS = [64, 128, 256, 512]
DROPOUT_RATE     = 0.2

# =============================================================================
# Training
# =============================================================================

DEVICE        = "cuda"   # or "cpu"

BATCH_SIZE_2D = 8
BATCH_SIZE_3D = 1
BATCH_SIZE    = BATCH_SIZE_2D   # logged to MLflow only — actual batch size
                                 # for each stage is BATCH_SIZE_2D / BATCH_SIZE_3D

NUM_EPOCHS    = 50
LEARNING_RATE = 4e-5
WEIGHT_DECAY  = 1e-5

VAL_SPLIT     = 0.2
TEST_SPLIT    = 0.1

LOSS_FUNCTION      = os.environ.get("XCT_LOSS_FUNCTION", "focal")   # "bce", "dice", "focal", "dice_focal"
DICE_FOCAL_LAMBDA  = 0.5
FOCAL_ALPHA        = 0.25
FOCAL_GAMMA        = 2.0

EARLY_STOP_PATIENCE = 10
SCHEDULER_PATIENCE  = 5

# =============================================================================
# Evaluation
# =============================================================================

DICE_THRESHOLD  = 0.5
ACCEPTANCE_DICE = 0.75
ACCEPTANCE_IOU  = 0.60
ACCEPTANCE_REC  = 0.80

# =============================================================================
# MLflow
# =============================================================================

MLFLOW_EXPERIMENT = "XCT_Defect_Detection"
MLFLOW_URI        = (REPO_ROOT / "mlruns").as_uri()
# config.py

# Enable or disable sample mask detection
USE_SAMPLE_MASK = False   # circular sample-boundary mask disabled entirely — thresholding
                          # runs on the full slice, air background included

# Number of pixels to erode from the detected boundary. Partial-volume
# edge pixels (a mix of air and metal intensity right at the sample
# surface) are unreliable for thresholding and read as spurious dark
# "pore-like" pixels if left in; raised from 5 -> 8 for a larger safety
# margin, after measured background leakage of 40-45 percentage points
# (e.g. 56% unmasked -> 9-13% masked) on the lower-coverage samples.
SAMPLE_MASK_EROSION_RADIUS = 8   # adjust based on your dataset

# Intensity threshold to classify air vs. sample
SAMPLE_MASK_AIR_THRESHOLD = 30   # typical range: 20–50 for uint8 XCT slices

# ------------------------------------------------------------------
# Empty-slice detection (top/bottom of a TIFF stack)
# ------------------------------------------------------------------
# The stack-level sample mask (detect_sample_mask_stack) is fit from
# slices in the CENTRE half of the stack only, then the same circle is
# reused for every slice, including ones near the top/bottom of the
# scan where the physical part hasn't started yet or has already ended
# — those slices are just air/noise inside that circle. Running
# thresholding on them anyway misclassifies the whole circular region
# as one large "pore", which shows up in a 3D reconstruction as flat,
# over-segmented discs at the top and bottom of the stack.
#
# Before thresholding each slice, the fraction of sample-mask pixels
# that are actually bright/solid (> SAMPLE_MASK_AIR_THRESHOLD) is
# checked; below this fraction, the slice is treated as containing no
# real object and its masks are written as all-zero (no defects — an
# empty slice cannot contain a defect) instead of being thresholded.
EMPTY_SLICE_MIN_OBJECT_FRACTION = 0.05
