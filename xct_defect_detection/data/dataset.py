"""
=============================================================================
PyTorch Dataset — Patch Extraction with Stratified Sampling
=============================================================================
Extracts 2D patches from XCT volumes and their pseudo-label masks using
a sliding window. Applies stratified foreground:background sampling to
address the severe class imbalance inherent in XCT defect data (defect
voxels constitute 1.2–2.8% of total volume).
=============================================================================
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PATCH_SIZE, PATCH_STRIDE, FG_BG_RATIO, MIN_FG_PIXELS
)
from data.augmentation import apply_augmentation


class XCTPatchDataset(Dataset):
    """
    PyTorch Dataset that yields (image_patch, mask_patch) pairs extracted
    from one or more XCT volumes and their corresponding pseudo-label masks.

    Patch extraction strategy
    -------------------------
    All patches are extracted using a sliding window with stride PATCH_STRIDE.
    Patches are then split into foreground (containing >= MIN_FG_PIXELS defect
    pixels) and background (no defect pixels) pools. During training, patches
    are sampled from these pools at the ratio FG_BG_RATIO to ensure the model
    receives sufficient gradient signal from the minority defect class.

    Parameters
    ----------
    volumes    : list[np.ndarray]  — preprocessed float32 volumes
    masks      : list[np.ndarray]  — binary uint8 pseudo-label masks
    augment    : bool              — whether to apply augmentation (train only)
    split      : str               — "train", "val", or "test" (for logging)
    """

    def __init__(
        self,
        volumes : list[np.ndarray],
        masks   : list[np.ndarray],
        augment : bool = False,
        split   : str  = "train"
    ):
        self.augment = augment
        self.split   = split

        self.fg_patches : list[tuple[np.ndarray, np.ndarray]] = []
        self.bg_patches : list[tuple[np.ndarray, np.ndarray]] = []

        for volume, mask in zip(volumes, masks):
            self._extract_patches(volume, mask)

        # Build final index list with FG_BG_RATIO oversampling
        fg_count = len(self.fg_patches)
        bg_count = min(
            len(self.bg_patches),
            fg_count * FG_BG_RATIO[1] // FG_BG_RATIO[0]
        )

        # Sample background patches without replacement up to bg_count
        bg_indices       = np.random.choice(
            len(self.bg_patches), size=bg_count, replace=False
        )
        self.all_patches = (
            self.fg_patches +
            [self.bg_patches[i] for i in bg_indices]
        )

        print(f"  [Dataset:{split}] FG patches : {fg_count}")
        print(f"  [Dataset:{split}] BG patches : {bg_count}  "
              f"(ratio {FG_BG_RATIO[0]}:{FG_BG_RATIO[1]})")
        print(f"  [Dataset:{split}] Total      : {len(self.all_patches)}")

    def _extract_patches(
        self,
        volume: np.ndarray,
        mask:   np.ndarray
    ) -> None:
        """
        Extract all valid patches from a single volume/mask pair.

        For 3D volumes, patches are extracted from each 2D slice
        independently (2D U-Net approach). For 2D volumes, the
        single slice is patched directly.

        Parameters
        ----------
        volume : np.ndarray  — preprocessed float32 volume (2D or 3D)
        mask   : np.ndarray  — binary uint8 mask (2D or 3D)
        """
        slices_v = [volume] if volume.ndim == 2 else [
            volume[i] for i in range(volume.shape[0])
        ]
        slices_m = [mask]   if mask.ndim   == 2 else [
            mask[i]   for i in range(mask.shape[0])
        ]

        P = PATCH_SIZE
        S = PATCH_STRIDE

        for slc_v, slc_m in zip(slices_v, slices_m):
            H, W = slc_v.shape

            for y in range(0, H - P + 1, S):
                for x in range(0, W - P + 1, S):
                    patch_v = slc_v[y:y+P, x:x+P].copy()
                    patch_m = slc_m[y:y+P, x:x+P].copy()

                    if patch_m.sum() >= MIN_FG_PIXELS:
                        self.fg_patches.append((patch_v, patch_m))
                    else:
                        self.bg_patches.append((patch_v, patch_m))

    def __len__(self) -> int:
        return len(self.all_patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return an (image, mask) tensor pair for a given patch index.

        Augmentation is applied only during training (augment=True).
        Tensors are returned with a channel dimension added:
            image : float32 tensor of shape (1, H, W)
            mask  : float32 tensor of shape (1, H, W)

        Parameters
        ----------
        idx : int  — patch index

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
        """
        image, mask = self.all_patches[idx]

        if self.augment:
            image, mask = apply_augmentation(image, mask)

        # Add channel dimension
        image_t = torch.from_numpy(image).unsqueeze(0).float()     # (1, H, W)
        mask_t  = torch.from_numpy(mask.astype(np.float32)         # (1, H, W)
                                   ).unsqueeze(0)
        return image_t, mask_t


def build_dataloaders(
    volumes_train : list[np.ndarray],
    masks_train   : list[np.ndarray],
    volumes_val   : list[np.ndarray],
    masks_val     : list[np.ndarray],
    batch_size    : int
) -> tuple:
    """
    Build training and validation DataLoaders.

    Parameters
    ----------
    volumes_train : list[np.ndarray]  — training volumes
    masks_train   : list[np.ndarray]  — training pseudo-label masks
    volumes_val   : list[np.ndarray]  — validation volumes
    masks_val     : list[np.ndarray]  — validation pseudo-label masks
    batch_size    : int

    Returns
    -------
    tuple (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader

    train_ds = XCTPatchDataset(volumes_train, masks_train,
                                augment=True,  split="train")
    val_ds   = XCTPatchDataset(volumes_val,   masks_val,
                                augment=False, split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=4,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=4,
                              pin_memory=True)
    return train_loader, val_loader
