"""
=============================================================================
Evaluation Metrics
=============================================================================
Implements all quantitative metrics defined in the thesis (Section 3.5):
    - Dice Similarity Coefficient (DSC)
    - Intersection over Union (IoU / Jaccard Index)
    - Precision
    - Recall
    - Acceptance criteria checking
=============================================================================
"""

import torch
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DICE_THRESHOLD,
    ACCEPTANCE_DICE,
    ACCEPTANCE_IOU,
    ACCEPTANCE_REC
)


def binarize(pred: torch.Tensor,
             threshold: float = DICE_THRESHOLD) -> torch.Tensor:
    """
    Convert probability map to binary prediction.

    Parameters
    ----------
    pred      : torch.Tensor  — predicted probabilities in [0, 1]
    threshold : float         — binarisation threshold (default 0.5)

    Returns
    -------
    torch.Tensor  — binary tensor of same shape as pred
    """
    return (pred >= threshold).float()


def dice_coefficient(
    pred:   torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6
) -> float:
    """
    Compute Dice Similarity Coefficient between prediction and target.

    DSC = 2|A ∩ B| / (|A| + |B|)

    Parameters
    ----------
    pred   : torch.Tensor  — binary prediction (B, 1, H, W)
    target : torch.Tensor  — binary ground truth (B, 1, H, W)
    smooth : float         — smoothing constant

    Returns
    -------
    float  — mean Dice score across the batch
    """
    pred   = binarize(pred).view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    return float(
        (2.0 * intersection + smooth) /
        (pred.sum() + target.sum() + smooth)
    )


def iou_score(
    pred:   torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6
) -> float:
    """
    Compute Intersection over Union (Jaccard Index).

    IoU = |A ∩ B| / |A ∪ B|

    Parameters
    ----------
    pred   : torch.Tensor  — binary prediction (B, 1, H, W)
    target : torch.Tensor  — binary ground truth (B, 1, H, W)
    smooth : float         — smoothing constant

    Returns
    -------
    float  — mean IoU score across the batch
    """
    pred   = binarize(pred).view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    union        = pred.sum() + target.sum() - intersection
    return float((intersection + smooth) / (union + smooth))


def precision_recall(
    pred:   torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6
) -> tuple[float, float]:
    """
    Compute voxel-level Precision and Recall.

    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)

    Parameters
    ----------
    pred   : torch.Tensor  — binary prediction (B, 1, H, W)
    target : torch.Tensor  — binary ground truth (B, 1, H, W)
    smooth : float         — smoothing constant

    Returns
    -------
    tuple[float, float]  — (precision, recall)
    """
    pred   = binarize(pred).view(-1)
    target = target.view(-1)

    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()

    precision = float((tp + smooth) / (tp + fp + smooth))
    recall    = float((tp + smooth) / (tp + fn + smooth))
    return precision, recall


def compute_all_metrics(
    pred:   torch.Tensor,
    target: torch.Tensor
) -> dict[str, float]:
    """
    Compute all evaluation metrics for a prediction/target pair.

    Parameters
    ----------
    pred   : torch.Tensor  — predicted probabilities (B, 1, H, W)
    target : torch.Tensor  — binary ground truth (B, 1, H, W)

    Returns
    -------
    dict with keys: 'dice', 'iou', 'precision', 'recall'
    """
    dice          = dice_coefficient(pred, target)
    iou           = iou_score(pred, target)
    prec, rec     = precision_recall(pred, target)
    return {
        "dice"      : dice,
        "iou"       : iou,
        "precision" : prec,
        "recall"    : rec,
    }


def check_acceptance_criteria(metrics: dict[str, float]) -> dict[str, bool]:
    """
    Check whether computed metrics meet the thesis acceptance criteria.

    Criteria (Section 3.5 of thesis):
        Dice  >= 0.75  (primary criterion)
        IoU   >= 0.60
        Recall >= 0.80

    Parameters
    ----------
    metrics : dict[str, float]  — output of compute_all_metrics()

    Returns
    -------
    dict[str, bool]  — per-criterion pass/fail status
    """
    results = {
        "dice_pass"   : metrics["dice"]   >= ACCEPTANCE_DICE,
        "iou_pass"    : metrics["iou"]    >= ACCEPTANCE_IOU,
        "recall_pass" : metrics["recall"] >= ACCEPTANCE_REC,
        "all_pass"    : (
            metrics["dice"]   >= ACCEPTANCE_DICE and
            metrics["iou"]    >= ACCEPTANCE_IOU  and
            metrics["recall"] >= ACCEPTANCE_REC
        )
    }
    return results
