"""
=============================================================================
train_2d_single_sample.py — plain 2D U-Net (UNET_INPUT_SLICES=1), trained
on ONE sample, evaluated against the highest-porosity NIST sample.
=============================================================================
Not a variant of pipeline.py's normal flow: split_volumes() requires at
least 3 samples for its train/val/test split, so a genuine single-sample
experiment needs its own train/val split (done here by slice index within
that one sample, 80/20) and its own held-out test sample, passed in
separately.

Default: trains on sample_02 (most defect-rich of the established
4-sample split), tests on sample_05 (highest published porosity, 72%,
of the 5 NIST samples) — never seen during training.

Run from repository root:
    python scripts/train_2d_single_sample.py [train_sample] [test_sample]
=============================================================================
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

import config
from config import create_dirs, VAL_SPLIT, EXPERIMENT_NAME, UNET_INPUT_SLICES
from data.cache import load_cache
from data.dataset import build_dataloaders
from models.unet2d import get_model as get_model_2d
from training.losses import get_loss_function
from training.trainer import train
from training.metrics import check_acceptance_criteria
from pipeline import evaluate_test_set_2d

create_dirs()

train_sample = sys.argv[1] if len(sys.argv) > 1 else "sample_02"
test_sample  = sys.argv[2] if len(sys.argv) > 2 else "sample_05"

print(f"UNET_INPUT_SLICES = {UNET_INPUT_SLICES}  "
      f"({'plain 2D' if UNET_INPUT_SLICES == 1 else '2.5D'})")
print(f"EXPERIMENT_NAME   = {EXPERIMENT_NAME}")
print(f"Train sample      = {train_sample}  (single sample, internal 80/20 split)")
print(f"Test sample       = {test_sample}  (held out — never seen in training)")
print(f"PROCESSED_DATA_DIR = {config.PROCESSED_DATA_DIR}")
print(f"MASKS_DIR           = {config.MASKS_DIR}")
print(f"CACHE_DIR            = {config.CACHE_DIR}\n")

# ---------------------------------------------------------------------------
# Load the training sample and split it 80/20 by slice index (not by
# sample, since there's only one) into train/val.
# ---------------------------------------------------------------------------
print(f"Loading training sample '{train_sample}' ...")
vol, mask = load_cache(train_sample, mask_method="bernsen")
vol, mask = np.asarray(vol), np.asarray(mask)
n = vol.shape[0]
n_val = max(1, int(n * VAL_SPLIT))
n_train = n - n_val
print(f"  {n} slices -> {n_train} train / {n_val} val (internal split)")

train_v, train_m = [vol[:n_train]], [mask[:n_train]]
val_v,   val_m   = [vol[n_train:]], [mask[n_train:]]

train_loader, val_loader = build_dataloaders(
    train_v, train_m, val_v, val_m, config.BATCH_SIZE_2D
)

loss_fn = get_loss_function()
model = get_model_2d()
best_ckpt = train(model, train_loader, val_loader, loss_fn)

# ---------------------------------------------------------------------------
# Evaluate on the held-out highest-porosity sample
# ---------------------------------------------------------------------------
print(f"\nLoading test sample '{test_sample}' (never seen in training) ...")
test_vol, test_mask = load_cache(test_sample, mask_method="bernsen")
test_v, test_m = [np.asarray(test_vol)], [np.asarray(test_mask)]

metrics = evaluate_test_set_2d(get_model_2d(), test_v, test_m, best_ckpt)

print("\n" + "=" * 65)
print(f"FINAL TEST SET RESULTS — trained on '{train_sample}' only, "
      f"tested on '{test_sample}'")
print("=" * 65)
print(f"2D Dice     : {metrics['dice']:.4f}")
print(f"2D IoU      : {metrics['iou']:.4f}")
print(f"2D Precision: {metrics['precision']:.4f}")
print(f"2D Recall   : {metrics['recall']:.4f}")
print("=" * 65)
acc = check_acceptance_criteria(metrics)
print(f"\n2D Acceptance: {'PASS' if acc['all_pass'] else 'FAIL'}")
