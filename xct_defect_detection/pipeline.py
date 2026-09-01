"""
=============================================================================
End-to-End XCT Defect Detection Pipeline (2D + 3D)
=============================================================================
Trains a 2D slice-wise U-Net and a 3D volumetric U-Net on the samples in
data/raw/, using Bernsen pseudo-label masks as training targets.

Prerequisites (run once, or let this script trigger them on demand):
    python scripts/run_preprocess.py     # data/raw/  -> data/processed/
    python scripts/run_all_samples.py    # data/processed/ -> data/masks/
(scripts/generate_training_data.py runs both non-interactively)

Volumes and masks are loaded as memory-mapped arrays (data/cache.py) —
full-resolution stacks (hundreds of slices) are never fully materialised
in RAM.

Run from project root:
    python pipeline.py
=============================================================================
"""

import os
import numpy as np
import torch
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Reconfigure stdout/stderr to UTF-8: this script's own and training/trainer.py's
# progress prints include Unicode characters (e.g. checkmarks). When stdout isn't
# attached to a real console — piped, redirected to a file, or run under some
# terminals' default Windows code page (cp1252) — Python falls back to a
# non-UTF-8 encoding that can't represent them, crashing training with a
# UnicodeEncodeError right after the first checkpoint save. errors="replace"
# additionally ensures no *other* future non-ASCII print can crash a run either.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
import config
from config import (
    SAMPLE_NAMES,
    TRAINING_SAMPLES_OVERRIDE,
    MAX_TRAINING_SAMPLES,
    EXPERIMENT_NAME,
    SKIP_3D_TRAINING,
    VAL_SPLIT, TEST_SPLIT,
    BATCH_SIZE_2D, BATCH_SIZE_3D,
    PATCH_SIZE_3D,
    DEVICE,
)

# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------
from data.cache            import load_cache
from data.dataset           import build_dataloaders, XCTPatchDataset
from data.dataset_3d        import build_dataloaders_3d, XCTPatchDataset3D

from models.unet2d        import get_model as get_model_2d
from models.unet3d        import get_model as get_model_3d

from training.losses      import get_loss_function
from training.trainer     import train
from training.metrics     import compute_all_metrics, check_acceptance_criteria
from torch.utils.data     import DataLoader


# -------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------
SUSPICIOUS_DEFECT_FRACTION = 0.10  # 10% — real AM porosity is a few % at most


def rank_samples_by_defect_content(sample_names: list[str]) -> list[tuple[str, float]]:
    """
    Rank samples by pseudo-label defect fraction (mean of the cached
    Bernsen mask), most voids first.

    Training on the emptiest samples wastes most of every patch on
    background — the model sees very few real defect examples per epoch.
    Prioritising defect-rich samples gives it far more positive signal for
    the same epoch budget.

    Guards against a real failure mode seen in this dataset: Bernsen's
    auto-computed DCT (src/thresholding.py::compute_dct_from_image) can
    come out far above the ~15 its own reference expects, which routes
    almost every pixel into the "low contrast" fallback — a fixed
    threshold at 128. For a sample whose preprocessed images average
    brighter than 128 this fallback still roughly works by luck; for a
    darker-averaging sample it misclassifies most of the solid material
    as void, producing a defect fraction of 50%+ that is an artifact, not
    real porosity. Anything above SUSPICIOUS_DEFECT_FRACTION is flagged
    and excluded rather than silently "winning" the ranking.

    Reads each sample's full cached mask once (~1 sequential pass over the
    mask.npy — a few seconds per sample on NVMe, longer if the cache still
    needs building from data/masks/<sample>/bernsen/).
    """
    ranked = []
    for name in sample_names:
        try:
            _, mask = load_cache(name, mask_method="bernsen")
            defect_frac = float(np.asarray(mask).mean())
        except (FileNotFoundError, ValueError) as e:
            print(f"  [Skip] '{name}' not ready yet: {e}")
            continue

        if defect_frac > SUSPICIOUS_DEFECT_FRACTION:
            print(f"    {name}: defect fraction = {defect_frac*100:.3f}%  "
                  f"[EXCLUDED — implausibly high, likely a Bernsen "
                  f"thresholding artifact, not real porosity]")
            continue

        ranked.append((name, defect_frac))
        print(f"    {name}: defect fraction = {defect_frac*100:.3f}%")

    ranked.sort(key=lambda t: t[1], reverse=True)
    return ranked


def select_training_samples() -> list[str]:
    """
    Return the ordered list of sample names this experiment trains on
    (most voids first) — the single source of truth for "what does
    EXPERIMENT_NAME train on", also imported by
    scripts/predict_and_visualize_3d.py so its "samples not used in
    training" default stays correct no matter how this is configured.

    If config.TRAINING_SAMPLES_OVERRIDE is set, that exact list is used
    as-is — the fastest way to try a different data mix (see config.py
    for the override + EXPERIMENT_NAME instructions). Otherwise every
    discovered sample is ranked by defect content and the top
    MAX_TRAINING_SAMPLES are kept.
    """
    if TRAINING_SAMPLES_OVERRIDE is not None:
        print(f"  TRAINING_SAMPLES_OVERRIDE set — using: {TRAINING_SAMPLES_OVERRIDE}")
        return list(TRAINING_SAMPLES_OVERRIDE)

    print("  Ranking samples by defect (void) content ...")
    ranked = rank_samples_by_defect_content(SAMPLE_NAMES)
    selected = (
        ranked[:MAX_TRAINING_SAMPLES]
        if MAX_TRAINING_SAMPLES is not None
        else ranked
    )
    return [name for name, _ in selected]


def load_all_volumes() -> tuple[dict, dict]:
    """
    Load this experiment's selected samples (see select_training_samples)
    as memory-mapped (volume, mask) pairs, keyed by sample name.

    A sample is skipped (with a warning) if its data/processed/<sample>/
    or data/masks/<sample>/bernsen/ isn't fully generated yet — e.g. while
    scripts/generate_training_data.py is still working through it in the
    background. Requires at least 3 ready samples for a train/val/test split.
    """
    sample_names = select_training_samples()
    print(f"  Experiment '{EXPERIMENT_NAME}' — training on "
          f"{len(sample_names)} sample(s): {sample_names}")

    volumes, masks = {}, {}
    for name in sample_names:
        vol, mask = load_cache(name, mask_method="bernsen")
        print(f"  Loaded cache for '{name}'  (shape {vol.shape})")
        volumes[name] = vol
        masks[name]   = mask

    return volumes, masks


def split_volumes(volumes: dict, masks: dict, val_split: float, test_split: float):
    """
    Split samples into train/val/test by rank, not by a random shuffle.

    `volumes`/`masks` are expected in defect-content-ranked order (most
    voids first — see load_all_volumes/rank_samples_by_defect_content).
    A random index-based split (the original approach, via sklearn's
    train_test_split with a fixed seed) fixes *which position* goes where,
    but not *which sample* ends up in that position — reordering the input
    (e.g. by rank instead of alphabetically) silently reassigns physical
    samples to different roles. That directly undermined the point of
    ranking: it once put the most defect-rich sample in test/val instead
    of train, and the model could barely learn the defect pattern found
    almost exclusively in that sample.

    Instead: keep the most defect-rich samples for training (that's the
    entire reason they were ranked to the front) and take val/test from
    the tail (the least defect-rich) — deterministic, and consistent with
    "train on the samples with the most voids".
    """
    names = list(volumes.keys())
    n = len(names)
    if n < 3:
        raise ValueError(
            f"Need at least 3 samples for a train/val/test split, "
            f"found {n}: {names}"
        )

    n_test  = max(1, round(n * test_split))
    n_val   = max(1, round(n * val_split))
    n_train = n - n_test - n_val
    if n_train < 1:
        raise ValueError(
            f"val_split ({val_split}) + test_split ({test_split}) leave no "
            f"samples for training with only {n} sample(s): {names}"
        )

    train_n = names[:n_train]
    val_n   = names[n_train:n_train + n_val]
    test_n  = names[n_train + n_val:]

    def sel(names_subset):
        return [volumes[n] for n in names_subset], [masks[n] for n in names_subset]

    train_v, train_m = sel(train_n)
    val_v,   val_m   = sel(val_n)
    test_v,  test_m  = sel(test_n)

    print(f"  Split (most voids -> train) -> "
          f"Train: {train_n}, Val: {val_n}, Test: {test_n}")
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
        test_ds, batch_size=BATCH_SIZE_2D, shuffle=False, num_workers=0
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
        test_ds, batch_size=1, shuffle=False, num_workers=0
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
    # Stage 1: Load cached volumes/masks (built by
    # scripts/generate_training_data.py if not already present)
    # --------------------------------------------------------------
    print("\n[STAGE 1] Loading volumes + pseudo-labels (memory-mapped)")
    all_vols, all_masks = load_all_volumes()

    # --------------------------------------------------------------
    # Stage 2: Split
    # --------------------------------------------------------------
    print("\n[STAGE 2] Train / Val / Test split")
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

    if SKIP_3D_TRAINING:
        print("\n" + "=" * 65)
        print("FINAL TEST SET RESULTS (2D only — SKIP_3D_TRAINING=True)")
        print("=" * 65)
        print(f"2D Dice     : {metrics_2d['dice']:.4f}")
        print(f"2D IoU      : {metrics_2d['iou']:.4f}")
        print(f"2D Precision: {metrics_2d['precision']:.4f}")
        print(f"2D Recall   : {metrics_2d['recall']:.4f}")
        print("=" * 65)
        acc_2d = check_acceptance_criteria(metrics_2d)
        print(f"\n2D Acceptance: {'PASS' if acc_2d['all_pass'] else 'FAIL'}")
        print("\n[DONE] Pipeline complete (2D only).\n")
        return

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
    print(f"2D Dice     : {metrics_2d['dice']:.4f}")
    print(f"2D IoU      : {metrics_2d['iou']:.4f}")
    print(f"2D Precision: {metrics_2d['precision']:.4f}")
    print(f"2D Recall   : {metrics_2d['recall']:.4f}")
    print("-" * 65)
    print(f"3D Dice     : {metrics_3d['dice']:.4f}")
    print(f"3D IoU      : {metrics_3d['iou']:.4f}")
    print(f"3D Precision: {metrics_3d['precision']:.4f}")
    print(f"3D Recall   : {metrics_3d['recall']:.4f}")
    print("=" * 65)

    acc_2d = check_acceptance_criteria(metrics_2d)
    acc_3d = check_acceptance_criteria(metrics_3d)

    print(f"\n2D Acceptance: {'PASS' if acc_2d['all_pass'] else 'FAIL'}")
    print(f"3D Acceptance: {'PASS' if acc_3d['all_pass'] else 'FAIL'}")
    print("\n[DONE] Pipeline complete.\n")


if __name__ == "__main__":
    main()
