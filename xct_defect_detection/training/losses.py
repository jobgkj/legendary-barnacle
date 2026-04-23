"""
=============================================================================
Loss Functions
=============================================================================
Implements all four loss function formulations evaluated in the thesis:
    1. Binary Cross-Entropy (BCE)          — standard baseline
    2. Dice Loss                           — overlap-based, imbalance-robust
    3. Focal Loss                          — hard-example focusing
    4. Combined Dice-Focal Loss            — thesis primary formulation
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FOCAL_ALPHA, FOCAL_GAMMA, DICE_FOCAL_LAMBDA, LOSS_FUNCTION
)


class BCELoss(nn.Module):
    """
    Standard binary cross-entropy loss.

    L_BCE = -[y·log(p) + (1-y)·log(1-p)]

    Used as the degenerate baseline to demonstrate class collapse
    on severely imbalanced XCT defect data.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return self.bce(pred, target)


class DiceLoss(nn.Module):
    """
    Dice loss derived from the Sørensen-Dice similarity coefficient.

    L_Dice = 1 - (2·Σ p_i·g_i + ε) / (Σ p_i + Σ g_i + ε)

    Inherently insensitive to class imbalance because it is normalised
    by the sum of predicted and true foreground sizes rather than
    total volume (Sudre et al., 2017).

    Parameters
    ----------
    smooth : float  — smoothing constant ε to avoid division by zero
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self,
                pred:   torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        pred   = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        dice_score   = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        return 1.0 - dice_score


class FocalLoss(nn.Module):
    """
    Focal loss for dense binary prediction (Lin et al., 2017).

    L_Focal = -α·(1 - p_t)^γ·log(p_t)

    The modulating factor (1 - p_t)^γ reduces the loss contribution
    of well-classified (easy) examples, focusing gradient signal on
    hard misclassified examples — particularly the rare defect voxels.

    Parameters
    ----------
    alpha : float  — class balancing weight (default 0.25)
    gamma : float  — focusing parameter (default 2.0)
    """

    def __init__(
        self,
        alpha: float = FOCAL_ALPHA,
        gamma: float = FOCAL_GAMMA
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self,
                pred:   torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy(pred, target, reduction="none")

        # p_t: probability of the true class
        p_t      = pred * target + (1 - pred) * (1 - target)
        # alpha_t: class balancing weight
        alpha_t  = self.alpha * target + (1 - self.alpha) * (1 - target)

        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        return focal_loss.mean()


class DiceFocalLoss(nn.Module):
    """
    Combined Dice-Focal loss (thesis primary loss formulation).

    L_DF = λ·L_Dice + (1-λ)·L_Focal

    Combines the global overlap sensitivity of Dice loss with the
    hard-example focusing of Focal loss, consistently outperforming
    either component alone on class-imbalanced segmentation tasks
    (Sudre et al., 2017; Lin et al., 2017).

    Parameters
    ----------
    lambda_dice : float  — weight of Dice component (default 0.5)
    alpha       : float  — Focal loss alpha
    gamma       : float  — Focal loss gamma
    """

    def __init__(
        self,
        lambda_dice : float = DICE_FOCAL_LAMBDA,
        alpha       : float = FOCAL_ALPHA,
        gamma       : float = FOCAL_GAMMA
    ):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.dice        = DiceLoss()
        self.focal       = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self,
                pred:   torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        l_dice  = self.dice(pred,  target)
        l_focal = self.focal(pred, target)
        return self.lambda_dice * l_dice + (1 - self.lambda_dice) * l_focal


def get_loss_function() -> nn.Module:
    """
    Return the configured loss function based on LOSS_FUNCTION in config.

    Returns
    -------
    nn.Module  — one of BCELoss, DiceLoss, FocalLoss, DiceFocalLoss
    """
    options = {
        "bce"        : BCELoss,
        "dice"       : DiceLoss,
        "focal"      : FocalLoss,
        "dice_focal" : DiceFocalLoss,
    }
    if LOSS_FUNCTION not in options:
        raise ValueError(
            f"Unknown loss function '{LOSS_FUNCTION}'. "
            f"Choose from: {list(options.keys())}"
        )
    loss_fn = options[LOSS_FUNCTION]()
    print(f"  [Loss] Using loss function: {LOSS_FUNCTION}")
    return loss_fn
