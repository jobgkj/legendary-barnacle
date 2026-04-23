"""
=============================================================================
Training Loop with MLflow Experiment Tracking
=============================================================================
Implements the full training and validation loop with:
    - Adam optimiser
    - ReduceLROnPlateau scheduler
    - Early stopping
    - Best model checkpointing
    - MLflow logging of all hyperparameters, metrics, and model artifacts
=============================================================================
"""

import os
import torch
import torch.nn as nn
import numpy as np
import mlflow
import mlflow.pytorch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEVICE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS,
    EARLY_STOP_PATIENCE, SCHEDULER_PATIENCE,
    MLFLOW_EXPERIMENT, MLFLOW_URI, CKPT_DIR,
    LOSS_FUNCTION, BATCH_SIZE, PATCH_SIZE,
    ENCODER_CHANNELS, DROPOUT_RATE,
    FOCAL_ALPHA, FOCAL_GAMMA, DICE_FOCAL_LAMBDA
)
from training.metrics import compute_all_metrics, check_acceptance_criteria


def train_one_epoch(
    model:       nn.Module,
    loader:      torch.utils.data.DataLoader,
    optimizer:   torch.optim.Optimizer,
    loss_fn:     nn.Module,
    device:      str
) -> tuple[float, dict]:
    """
    Run one training epoch.

    Parameters
    ----------
    model     : nn.Module         — U-Net model
    loader    : DataLoader        — training DataLoader
    optimizer : Optimizer         — Adam optimiser
    loss_fn   : nn.Module         — configured loss function
    device    : str               — "cuda" or "cpu"

    Returns
    -------
    tuple (mean_loss, mean_metrics_dict)
    """
    model.train()
    epoch_loss    = 0.0
    epoch_metrics = {"dice": 0.0, "iou": 0.0,
                     "precision": 0.0, "recall": 0.0}

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss  = loss_fn(preds, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_metrics = compute_all_metrics(preds.detach().cpu(),
                                            masks.detach().cpu())
        for k in epoch_metrics:
            epoch_metrics[k] += batch_metrics[k]

    n = len(loader)
    return epoch_loss / n, {k: v / n for k, v in epoch_metrics.items()}


def validate_one_epoch(
    model:   nn.Module,
    loader:  torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device:  str
) -> tuple[float, dict]:
    """
    Run one validation epoch (no gradient computation).

    Parameters
    ----------
    model   : nn.Module   — U-Net model
    loader  : DataLoader  — validation DataLoader
    loss_fn : nn.Module   — configured loss function
    device  : str         — "cuda" or "cpu"

    Returns
    -------
    tuple (mean_loss, mean_metrics_dict)
    """
    model.eval()
    epoch_loss    = 0.0
    epoch_metrics = {"dice": 0.0, "iou": 0.0,
                     "precision": 0.0, "recall": 0.0}

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)

            preds = model(images)
            loss  = loss_fn(preds, masks)

            epoch_loss += loss.item()
            batch_metrics = compute_all_metrics(preds.cpu(), masks.cpu())
            for k in epoch_metrics:
                epoch_metrics[k] += batch_metrics[k]

    n = len(loader)
    return epoch_loss / n, {k: v / n for k, v in epoch_metrics.items()}


def train(
    model:        nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader:   torch.utils.data.DataLoader,
    loss_fn:      nn.Module
) -> str:
    """
    Full training loop with MLflow tracking, early stopping, and
    best-model checkpointing.

    Parameters
    ----------
    model        : nn.Module   — instantiated U-Net model
    train_loader : DataLoader  — training data
    val_loader   : DataLoader  — validation data
    loss_fn      : nn.Module   — configured loss function

    Returns
    -------
    str  — path to the best saved model checkpoint
    """
    device    = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE,
                     weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max",
                                   factor=0.5, patience=SCHEDULER_PATIENCE,
                                   verbose=True)

    best_dice       = 0.0
    patience_counter = 0
    best_ckpt_path  = os.path.join(CKPT_DIR, "best_model.pt")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():

        # Log all hyperparameters
        mlflow.log_params({
            "loss_function"    : LOSS_FUNCTION,
            "learning_rate"    : LEARNING_RATE,
            "weight_decay"     : WEIGHT_DECAY,
            "batch_size"       : BATCH_SIZE,
            "num_epochs"       : NUM_EPOCHS,
            "patch_size"       : PATCH_SIZE,
            "encoder_channels" : str(ENCODER_CHANNELS),
            "dropout_rate"     : DROPOUT_RATE,
            "focal_alpha"      : FOCAL_ALPHA,
            "focal_gamma"      : FOCAL_GAMMA,
            "dice_focal_lambda": DICE_FOCAL_LAMBDA,
            "device"           : str(device),
        })

        print(f"\n  [Trainer] Training on {device} for {NUM_EPOCHS} epochs.")
        print(f"  [Trainer] Loss: {LOSS_FUNCTION}  |  "
              f"LR: {LEARNING_RATE}  |  Batch: {BATCH_SIZE}\n")

        for epoch in range(1, NUM_EPOCHS + 1):

            train_loss, train_metrics = train_one_epoch(
                model, train_loader, optimizer, loss_fn, device
            )
            val_loss, val_metrics = validate_one_epoch(
                model, val_loader, loss_fn, device
            )

            scheduler.step(val_metrics["dice"])

            # MLflow logging
            mlflow.log_metrics({
                "train_loss"      : train_loss,
                "train_dice"      : train_metrics["dice"],
                "train_iou"       : train_metrics["iou"],
                "train_precision" : train_metrics["precision"],
                "train_recall"    : train_metrics["recall"],
                "val_loss"        : val_loss,
                "val_dice"        : val_metrics["dice"],
                "val_iou"         : val_metrics["iou"],
                "val_precision"   : val_metrics["precision"],
                "val_recall"      : val_metrics["recall"],
                "learning_rate"   : optimizer.param_groups[0]["lr"],
            }, step=epoch)

            print(
                f"  Epoch [{epoch:03d}/{NUM_EPOCHS}] "
                f"| Train Loss: {train_loss:.4f} "
                f"| Val Dice: {val_metrics['dice']:.4f} "
                f"| Val IoU: {val_metrics['iou']:.4f} "
                f"| Val Rec: {val_metrics['recall']:.4f}"
            )

            # Save best model
            if val_metrics["dice"] > best_dice:
                best_dice = val_metrics["dice"]
                patience_counter = 0
                torch.save({
                    "epoch"       : epoch,
                    "model_state" : model.state_dict(),
                    "optim_state" : optimizer.state_dict(),
                    "val_dice"    : best_dice,
                    "val_metrics" : val_metrics,
                }, best_ckpt_path)
                print(f"    ✓ Best model saved  "
                      f"(Val Dice: {best_dice:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"\n  [Trainer] Early stopping at epoch {epoch} "
                          f"(no improvement for {EARLY_STOP_PATIENCE} epochs).")
                    break

        # Check acceptance criteria on best validation metrics
        acceptance = check_acceptance_criteria(val_metrics)
        mlflow.log_params(acceptance)
        mlflow.log_artifact(best_ckpt_path)

        print("\n  ======================================")
        print("  ACCEPTANCE CRITERIA CHECK")
        print("  ======================================")
        print(f"  Dice  >= {0.75}  : "
              f"{'PASS ✓' if acceptance['dice_pass']   else 'FAIL ✗'}"
              f"  ({val_metrics['dice']:.4f})")
        print(f"  IoU   >= {0.60}  : "
              f"{'PASS ✓' if acceptance['iou_pass']    else 'FAIL ✗'}"
              f"  ({val_metrics['iou']:.4f})")
        print(f"  Recall >= {0.80} : "
              f"{'PASS ✓' if acceptance['recall_pass'] else 'FAIL ✗'}"
              f"  ({val_metrics['recall']:.4f})")
        print(f"  Overall: {'ALL PASS ✓' if acceptance['all_pass'] else 'NOT YET ✗'}")
        print("  ======================================\n")

    return best_ckpt_path
