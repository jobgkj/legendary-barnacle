"""
=============================================================================
End-to-End XCT Defect Detection Pipeline
=============================================================================
Master entry point. Runs all stages in sequence:

    Stage 1  Load NIST and PODFAM volumes
    Stage 2  Full preprocessing (normalise, BHC, ring, NLM)
    Stage 3  Generate Otsu pseudo-labels (or load cached)
    Stage 4  Split into train / val / test sets
    Stage 5  Build patch datasets and DataLoaders
    Stage 6  Instantiate model and loss function
    Stage 7  Train with MLflow tracking
    Stage 8  Evaluate on test set and report metrics

Run from project root:
    python pipeline.py
=============================================================================
"""

import os
import glob
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    NIST_VOL_DIR, NIST_MASK_DIR,
    PODFAM_VOL_DIR, PODFAM_MASK_DIR,
    VAL_SPLIT, TEST_SPLIT, BATCH_SIZE, DEVICE, CKPT_DIR
)
from data.loader          import load_tiff_stack, full_preprocess
from data.pseudo_labels   import generate_and_save_pseudo_labels
from data.dataset         import build_dataloaders, XCTPatchDataset
from models.unet2d        import get_model
from training.losses      import get_loss_function
from training.trainer     import train
from training.metrics     import compute_all_metrics, check_acceptance_criteria
from torch.utils.data     import DataLoader


def load_all_volumes(vol_dir: str) -> dict[str, np.ndarray]:
    """
    Load and preprocess all TIFF stacks found in vol_dir.

    Each subdirectory or volume file is treated as one scan.
    Supports both flat directories (all slices in one folder)
    and nested directories (one subdirectory per scan).

    Parameters
    ----------
    vol_dir : str  — root directory containing volume folders

    Returns
    -------
    dict[str, np.ndarray]  — {volume_name: preprocessed_volume}
    """
    volumes = {}

    # Check for subdirectories (one per scan)
    subdirs = sorted([
        d for d in glob.glob(os.path.join(vol_dir, "*"))
        if os.path.isdir(d)
    ])

    if subdirs:
        # Nested: each subdir is one scan
        for subdir in subdirs:
            name = os.path.basename(subdir)
            print(f"\n  Loading volume: '{name}' ...")
            raw     = load_tiff_stack(subdir)
            volumes[name] = full_preprocess(raw)
    else:
        # Flat: all slices in vol_dir form a single scan
        name = os.path.basename(vol_dir.rstrip("/"))
        print(f"\n  Loading volume: '{name}' ...")
        raw           = load_tiff_stack(vol_dir)
        volumes[name] = full_preprocess(raw)

    return volumes


def split_volumes(
    volumes: dict[str, np.ndarray],
    masks:   dict[str, np.ndarray],
    val_split:  float,
    test_split: float
) -> tuple:
    """
    Split volume/mask pairs into train, validation, and test sets.

    Parameters
    ----------
    volumes    : dict[str, np.ndarray]  — all preprocessed volumes
    masks      : dict[str, np.ndarray]  — all pseudo-label masks
    val_split  : float                  — fraction for validation
    test_split : float                  — fraction for test

    Returns
    -------
    tuple of (train_vols, train_masks, val_vols, val_masks,
              test_vols,  test_masks)
    """
    names = list(volumes.keys())
    vols  = [volumes[n] for n in names]
    msks  = [masks[n]   for n in names]

    # First split off test set
    idx = list(range(len(names)))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_split, random_state=42
    )
    # Then split trainval into train and val
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_split / (1 - test_split),
        random_state=42
    )

    def _select(indices):
        return [vols[i] for i in indices], [msks[i] for i in indices]

    train_v, train_m = _select(idx_train)
    val_v,   val_m   = _select(idx_val)
    test_v,  test_m  = _select(idx_test)

    print(f"\n  [Split] Train: {len(train_v)}  "
          f"Val: {len(val_v)}  Test: {len(test_v)} volumes")
    return train_v, train_m, val_v, val_m, test_v, test_m


def evaluate_test_set(
    model:      torch.nn.Module,
    test_vols:  list[np.ndarray],
    test_masks: list[np.ndarray],
    ckpt_path:  str
) -> dict[str, float]:
    """
    Load best checkpoint and evaluate on the held-out test set.

    Parameters
    ----------
    model      : nn.Module          — U-Net model (uninitialised weights)
    test_vols  : list[np.ndarray]   — test volumes
    test_masks : list[np.ndarray]   — test pseudo-label masks
    ckpt_path  : str                — path to best checkpoint file

    Returns
    -------
    dict[str, float]  — mean metrics over the test set
    """
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    test_ds     = XCTPatchDataset(test_vols, test_masks,
                                   augment=False, split="test")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4)

    all_metrics = {"dice": [], "iou": [], "precision": [], "recall": []}

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            preds  = model(images).cpu()
            m      = compute_all_metrics(preds, masks)
            for k in all_metrics:
                all_metrics[k].append(m[k])

    mean_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    return mean_metrics


def main():
    print()
    print("=" * 65)
    print("   XCT Defect Detection — End-to-End Pipeline")
    print("   Master's Thesis, University West, 2026")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Stage 1: Load all volumes
    # ------------------------------------------------------------------
    print("\n[STAGE 1] Loading NIST volumes ...")
    nist_volumes = load_all_volumes(NIST_VOL_DIR)

    print("\n[STAGE 1] Loading PODFAM volumes ...")
    podfam_volumes = load_all_volumes(PODFAM_VOL_DIR)

    # Merge all volumes into one collection
    all_volumes = {**nist_volumes, **podfam_volumes}
    print(f"\n  Total volumes loaded: {len(all_volumes)}")

    # ------------------------------------------------------------------
    # Stage 2: Generate pseudo-labels (or load cached)
    # ------------------------------------------------------------------
    print("\n[STAGE 2] Generating Otsu pseudo-labels ...")

    nist_masks = generate_and_save_pseudo_labels(
        NIST_VOL_DIR, NIST_MASK_DIR, nist_volumes
    )
    podfam_masks = generate_and_save_pseudo_labels(
        PODFAM_VOL_DIR, PODFAM_MASK_DIR, podfam_volumes
    )
    all_masks = {**nist_masks, **podfam_masks}

    # ------------------------------------------------------------------
    # Stage 3: Train/val/test split
    # ------------------------------------------------------------------
    print("\n[STAGE 3] Splitting into train/val/test sets ...")
    train_v, train_m, val_v, val_m, test_v, test_m = split_volumes(
        all_volumes, all_masks, VAL_SPLIT, TEST_SPLIT
    )

    # ------------------------------------------------------------------
    # Stage 4: Build DataLoaders
    # ------------------------------------------------------------------
    print("\n[STAGE 4] Building patch datasets and DataLoaders ...")
    train_loader, val_loader = build_dataloaders(
        train_v, train_m, val_v, val_m, BATCH_SIZE
    )

    # ------------------------------------------------------------------
    # Stage 5: Model and loss function
    # ------------------------------------------------------------------
    print("\n[STAGE 5] Instantiating model and loss function ...")
    model   = get_model()
    loss_fn = get_loss_function()

    # ------------------------------------------------------------------
    # Stage 6: Train
    # ------------------------------------------------------------------
    print("\n[STAGE 6] Training ...")
    best_ckpt = train(model, train_loader, val_loader, loss_fn)

    # ------------------------------------------------------------------
    # Stage 7: Test set evaluation
    # ------------------------------------------------------------------
    print("\n[STAGE 7] Evaluating on held-out test set ...")
    model_fresh  = get_model()
    test_metrics = evaluate_test_set(model_fresh, test_v, test_m, best_ckpt)
    acceptance   = check_acceptance_criteria(test_metrics)

    print("\n" + "=" * 65)
    print("  FINAL TEST SET RESULTS")
    print("=" * 65)
    print(f"  Dice Score : {test_metrics['dice']:.4f}  "
          f"{'✓ PASS' if acceptance['dice_pass']   else '✗ FAIL'}")
    print(f"  IoU Score  : {test_metrics['iou']:.4f}  "
          f"{'✓ PASS' if acceptance['iou_pass']    else '✗ FAIL'}")
    print(f"  Precision  : {test_metrics['precision']:.4f}")
    print(f"  Recall     : {test_metrics['recall']:.4f}  "
          f"{'✓ PASS' if acceptance['recall_pass'] else '✗ FAIL'}")
    print(f"\n  Overall: {'ALL CRITERIA PASSED ✓' if acceptance['all_pass'] else 'CRITERIA NOT YET MET ✗'}")
    print("=" * 65)
    print("\n[DONE] Pipeline complete.\n")


if __name__ == "__main__":
    main()
