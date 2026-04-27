"""
=============================================================================
End-to-End XCT Defect Detection Pipeline (2D + 3D)
=============================================================================
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

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
from config import (
    NIST_VOL_DIR, NIST_MASK_DIR,
    PODFAM_VOL_DIR, PODFAM_MASK_DIR,
    VAL_SPLIT, TEST_SPLIT,
    BATCH_SIZE_2D, BATCH_SIZE_3D,
    PATCH_SIZE_3D,
    DEVICE
)

# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------
from data.loader          import load_tiff_stack, full_preprocess
from data.pseudo_labels   import generate_and_save_pseudo_labels
from data.dataset         import build_dataloaders, XCTPatchDataset
from data.dataset_3d      import build_dataloaders_3d, XCTPatchDataset3D

from models.unet2d        import get_model as get_model_2d
from models.unet3d        import get_model as get_model_3d

from training.losses      import get_loss_function
from training.trainer     import train
from training.metrics     import compute_all_metrics, check_acceptance_criteria
from torch.utils.data     import DataLoader


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
def load_all_volumes(vol_dir: str) -> dict[str, np.ndarray]:
    volumes = {}
    subdirs = sorted([
        d for d in glob.glob(os.path.join(vol_dir, "*"))
        if os.path.isdir(d)
    ])

    if subdirs:
        for subdir in subdirs:
            name = os.path.basename(subdir)
            print(f"  Loading volume '{name}'")
            raw = load_tiff_stack(subdir)
            volumes[name] = full_preprocess(raw)
    else:
        name = os.path.basename(vol_dir.rstrip("/"))
        raw = load_tiff_stack(vol_dir)
        volumes[name] = full_preprocess(raw)

    return volumes


def split_volumes(volumes, masks, val_split, test_split):
    names = list(volumes.keys())
    vols  = [volumes[n] for n in names]
    msks  = [masks[n]   for n in names]

    idx = list(range(len(names)))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_split, random_state=42
    )
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_split / (1 - test_split),
        random_state=42
    )

    def sel(i):
        return [vols[j] for j in i], [msks[j] for j in i]

    train_v, train_m = sel(idx_train)
    val_v,   val_m   = sel(idx_val)
    test_v,  test_m  = sel(idx_test)

    print(f"  Split → Train: {len(train_v)}, Val: {len(val_v)}, Test: {len(test_v)}")
    return train_v, train_m, val_v, val_m, test_v, test_m


# -------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------
def evaluate_test_set_2d(model, test_vols, test_masks, ckpt_path):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    test_ds = XCTPatchDataset(test_vols, test_masks,
                              augment=False, split="test")
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE_2D, shuffle=False
    )

    scores = {"dice": [], "iou": [], "precision": [], "recall": []}
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            m = compute_all_metrics(model(x).cpu(), y)
            for k in scores:
                scores[k].append(m[k])

    return {k: float(np.mean(v)) for k, v in scores.items()}


def evaluate_test_set_3d(model, test_vols, test_masks, ckpt_path):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    test_ds = XCTPatchDataset3D(
        test_vols, test_masks,
        patch_size=PATCH_SIZE_3D,
        augment=False,
        split="test"
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False
    )

    scores = {"dice": [], "iou": [], "precision": [], "recall": []}
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            m = compute_all_metrics(model(x).cpu(), y)
            for k in scores:
                scores[k].append(m[k])

    return {k: float(np.mean(v)) for k, v in scores.items()}


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():
    print("\n" + "=" * 65)
    print("   XCT Defect Detection — 2D + 3D Pipeline")
    print("=" * 65)

    # --------------------------------------------------------------
    # Stage 1–2: Load & preprocess
    # --------------------------------------------------------------
    print("\n[STAGE 1] Loading volumes")
    nist_vols   = load_all_volumes(NIST_VOL_DIR)
    podfam_vols = load_all_volumes(PODFAM_VOL_DIR)
    all_vols    = {**nist_vols, **podfam_vols}

    print("\n[STAGE 2] Generating / loading pseudo-labels")
    nist_masks   = generate_and_save_pseudo_labels(
        NIST_VOL_DIR, NIST_MASK_DIR, nist_vols
    )
    podfam_masks = generate_and_save_pseudo_labels(
        PODFAM_VOL_DIR, PODFAM_MASK_DIR, podfam_vols
    )
    all_masks = {**nist_masks, **podfam_masks}

    # --------------------------------------------------------------
    # Stage 3: Split
    # --------------------------------------------------------------
    print("\n[STAGE 3] Train / Val / Test split")
    train_v, train_m, val_v, val_m, test_v, test_m = split_volumes(
        all_vols, all_masks, VAL_SPLIT, TEST_SPLIT
    )

    loss_fn = get_loss_function()

    # ==============================================================
    # 2D TRAINING
    # ==============================================================
    print("\n[2D] Training slice-wise U-Net")
    train_loader_2d, val_loader_2d = build_dataloaders(
        train_v, train_m, val_v, val_m, BATCH_SIZE_2D
    )

    model_2d = get_model_2d()
    best_ckpt_2d = train(
        model_2d, train_loader_2d, val_loader_2d, loss_fn
    )

    metrics_2d = evaluate_test_set_2d(
        get_model_2d(), test_v, test_m, best_ckpt_2d
    )

    # ==============================================================
    # 3D TRAINING
    # ==============================================================
    print("\n[3D] Training volumetric U-Net")
    train_loader_3d, val_loader_3d = build_dataloaders_3d(
        train_v, train_m, val_v, val_m,
        BATCH_SIZE_3D, PATCH_SIZE_3D
    )

    model_3d = get_model_3d()
    best_ckpt_3d = train(
        model_3d, train_loader_3d, val_loader_3d, loss_fn
    )

    metrics_3d = evaluate_test_set_3d(
        get_model_3d(), test_v, test_m, best_ckpt_3d
    )

    # --------------------------------------------------------------
    # Final report
    # --------------------------------------------------------------
    print("\n" + "=" * 65)
    print("FINAL TEST SET RESULTS")
    print("=" * 65)
    print(f"2D Dice: {metrics_2d['dice']:.4f}")
    print(f"2D IoU : {metrics_2d['iou']:.4f}")
    print("-" * 65)
    print(f"3D Dice: {metrics_3d['dice']:.4f}")
    print(f"3D IoU : {metrics_3d['iou']:.4f}")
    print("=" * 65)

    acc_2d = check_acceptance_criteria(metrics_2d)
    acc_3d = check_acceptance_criteria(metrics_3d)

    print(f"\n2D Acceptance: {'PASS' if acc_2d['all_pass'] else 'FAIL'}")
    print(f"3D Acceptance: {'PASS' if acc_3d['all_pass'] else 'FAIL'}")
    print("\n[DONE] Pipeline complete.\n")


if __name__ == "__main__":
    main()
