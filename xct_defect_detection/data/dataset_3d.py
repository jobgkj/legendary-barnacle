"""
=============================================================================
PyTorch Dataset — 3D Patch Extraction with Stratified Random Sampling
=============================================================================
Extracts volumetric (D×H×W) sub-volume patches from XCT volumes and their
pseudo-label masks using random sampling with stratified FG:BG balancing.

Input:  (B, 1, D, H, W)  float32 sub-volume
Output: (B, 1, D, H, W)  float32 binary mask
=============================================================================
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FG_BG_RATIO, MIN_FG_PIXELS
from data.augmentation import apply_augmentation   # slice-wise 2D augmentation

# 3D patch config — adjust to fit your GPU memory
PATCH_D          = 16    # depth  (number of slices per patch)
PATCH_H          = 128   # height
PATCH_W          = 128   # width
SAMPLES_PER_VOL  = 200   # random patches drawn per volume per epoch


class XCTPatchDataset3D(Dataset):
    """
    PyTorch Dataset yielding (sub_volume, mask_patch) tensor pairs.

    Uses random patch sampling rather than exhaustive sliding window
    to keep dataset size manageable for large 3D volumes. Stratified
    FG:BG sampling ensures the model sees enough defect patches despite
    severe class imbalance.

    Parameters
    ----------
    volumes         : list[np.ndarray]  — preprocessed float32 volumes (D,H,W)
    masks           : list[np.ndarray]  — binary uint8 masks (D,H,W)
    patch_size      : tuple             — (D, H, W) patch dimensions
    augment         : bool              — apply slice-wise augmentation
    split           : str               — "train", "val", or "test"
    samples_per_vol : int               — random patches per volume
    max_fg_attempts : int               — attempts to find a FG patch before
                                          falling back to random
    """

    def __init__(
        self,
        volumes         : list,
        masks           : list,
        patch_size      : tuple = (PATCH_D, PATCH_H, PATCH_W),
        augment         : bool  = False,
        split           : str   = "train",
        samples_per_vol : int   = SAMPLES_PER_VOL,
        max_fg_attempts : int   = 10,
    ):
        self.augment         = augment
        self.split           = split
        self.ps              = patch_size
        self.max_fg_attempts = max_fg_attempts

        # Pre-build a flat list of (volume, mask) pairs
        self.pairs = list(zip(volumes, masks))

        # Total dataset length
        self.length = len(self.pairs) * samples_per_vol

        # Compute and log FG patch fraction for info
        total_defect = sum(int(m.sum()) for _, m in self.pairs)
        total_vox    = sum(int(m.size)  for _, m in self.pairs)
        fg_pct       = total_defect / max(total_vox, 1) * 100

        print(f"  [Dataset3D:{split}] Volumes        : {len(self.pairs)}")
        print(f"  [Dataset3D:{split}] Patch size     : {patch_size}")
        print(f"  [Dataset3D:{split}] Samples/vol    : {samples_per_vol}")
        print(f"  [Dataset3D:{split}] Total patches  : {self.length}")
        print(f"  [Dataset3D:{split}] Defect fraction: {fg_pct:.3f}%")

    def __len__(self) -> int:
        return self.length

    def _random_patch(
        self,
        volume: np.ndarray,
        mask:   np.ndarray,
        force_fg: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample one random patch from volume/mask.

        If force_fg=True, retries up to max_fg_attempts times to find
        a patch containing at least MIN_FG_PIXELS defect voxels.
        """
        pd, ph, pw = self.ps
        D, H, W    = volume.shape

        # Pad if volume is smaller than patch in any dimension
        pad = (
            max(0, pd - D),
            max(0, ph - H),
            max(0, pw - W),
        )
        if any(p > 0 for p in pad):
            volume = np.pad(volume,
                            ((0, pad[0]), (0, pad[1]), (0, pad[2])),
                            mode="constant", constant_values=0.0)
            mask   = np.pad(mask,
                            ((0, pad[0]), (0, pad[1]), (0, pad[2])),
                            mode="constant", constant_values=0)
            D, H, W = volume.shape

        attempts = self.max_fg_attempts if force_fg else 1
        for _ in range(attempts):
            z = np.random.randint(0, max(1, D - pd + 1))
            y = np.random.randint(0, max(1, H - ph + 1))
            x = np.random.randint(0, max(1, W - pw + 1))

            vp = volume[z:z+pd, y:y+ph, x:x+pw]
            mp = mask  [z:z+pd, y:y+ph, x:x+pw]

            if not force_fg or mp.sum() >= MIN_FG_PIXELS:
                return vp.copy(), mp.copy()

        # Fallback: return whatever we got on the last attempt
        return vp.copy(), mp.copy()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return one (image, mask) tensor pair.

        Every FG_BG_RATIO[0] out of FG_BG_RATIO[1] samples are forced
        to contain at least MIN_FG_PIXELS defect voxels.

        Returns
        -------
        image : float32 tensor (1, D, H, W)
        mask  : float32 tensor (1, D, H, W)
        """
        vol_idx    = idx % len(self.pairs)
        volume, mask = self.pairs[vol_idx]

        # Stratified: force FG patch according to FG_BG_RATIO
        force_fg = (idx % FG_BG_RATIO[1]) < FG_BG_RATIO[0]
        vp, mp   = self._random_patch(volume, mask, force_fg=force_fg)

        # Slice-wise 2D augmentation (applied to each depth slice)
        if self.augment:
            for i in range(vp.shape[0]):
                vp[i], mp[i] = apply_augmentation(
                    vp[i].astype(np.float32),
                    mp[i].astype(np.uint8)
                )

        image_t = torch.from_numpy(vp).unsqueeze(0).float()          # (1,D,H,W)
        mask_t  = torch.from_numpy(mp.astype(np.float32)).unsqueeze(0)  # (1,D,H,W)
        return image_t, mask_t


def build_dataloaders_3d(
    volumes_train   : list,
    masks_train     : list,
    volumes_val     : list,
    masks_val       : list,
    batch_size      : int,
    patch_size      : tuple = (PATCH_D, PATCH_H, PATCH_W),
    samples_per_vol : int   = SAMPLES_PER_VOL,
    num_workers     : int   = 0,
) -> tuple:
    """
    Build 3D training and validation DataLoaders.

    Parameters
    ----------
    volumes_train   : list  — training volumes
    masks_train     : list  — training masks
    volumes_val     : list  — validation volumes
    masks_val       : list  — validation masks
    batch_size      : int
    patch_size      : tuple — (D, H, W)
    samples_per_vol : int   — patches per volume per epoch
    num_workers     : int   — DataLoader workers. Defaults to 0: volumes/masks
                              are memory-mapped (see data/cache.py), and
                              numpy's default pickling of a memmap array
                              materialises its full contents in memory —
                              exactly what memmapping was meant to avoid.
                              Raise this only if volumes/masks are plain
                              in-RAM arrays.

    Returns
    -------
    tuple (train_loader, val_loader)
    """
    train_ds = XCTPatchDataset3D(
        volumes_train, masks_train,
        patch_size=patch_size,
        augment=True, split="train",
        samples_per_vol=samples_per_vol,
    )
    val_ds = XCTPatchDataset3D(
        volumes_val, masks_val,
        patch_size=patch_size,
        augment=False, split="val",
        samples_per_vol=max(1, samples_per_vol // 4),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
    )
    return train_loader, val_loader
