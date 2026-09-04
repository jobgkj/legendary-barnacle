"""
=============================================================================
eval_baseline_test_set.py — pure-inference re-evaluation of the EXISTING
artifacts/best_model_loss_bce.pt checkpoint on its held-out test sample.
=============================================================================
No training. Uses the ORIGINAL (non-BHC/ring) data/processed + data/masks
(default config paths — no XCT_* env overrides), the exact same
split_volumes()/evaluate_test_set_2d() logic as pipeline.py, so the result
is directly comparable to the BHC+ring run's "FINAL TEST SET RESULTS".

Run from repository root:
    python scripts/eval_baseline_test_set.py
=============================================================================
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import config
from config import create_dirs, VAL_SPLIT, TEST_SPLIT
from pipeline import load_all_volumes, split_volumes, evaluate_test_set_2d
from models.unet2d import get_model as get_model_2d
from training.metrics import check_acceptance_criteria

create_dirs()

print(f"PROCESSED_DATA_DIR = {config.PROCESSED_DATA_DIR}")
print(f"MASKS_DIR           = {config.MASKS_DIR}")
print(f"CACHE_DIR            = {config.CACHE_DIR}")

all_vols, all_masks = load_all_volumes()
train_v, train_m, val_v, val_m, test_v, test_m = split_volumes(
    all_vols, all_masks, VAL_SPLIT, TEST_SPLIT
)

ckpt_path = config.CKPT_DIR / "best_model_loss_bce.pt"
print(f"\nEvaluating checkpoint: {ckpt_path}")

metrics = evaluate_test_set_2d(get_model_2d(), test_v, test_m, ckpt_path)

print("\n" + "=" * 65)
print("BASELINE (original 3-stage preprocessing) FINAL TEST SET RESULTS")
print("=" * 65)
print(f"2D Dice     : {metrics['dice']:.4f}")
print(f"2D IoU      : {metrics['iou']:.4f}")
print(f"2D Precision: {metrics['precision']:.4f}")
print(f"2D Recall   : {metrics['recall']:.4f}")
print("=" * 65)
acc = check_acceptance_criteria(metrics)
print(f"\n2D Acceptance: {'PASS' if acc['all_pass'] else 'FAIL'}")
