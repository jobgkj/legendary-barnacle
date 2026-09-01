"""
=============================================================================
PyTorch Dataset — Lazy 2D Patch Extraction with Stratified Sampling
=============================================================================
Extracts 2D patches from XCT volumes and their pseudo-label masks using
a sliding window. Applies stratified foreground:background sampling to
address the severe class imbalance inherent in XCT defect data.

Memory design
-------------
`volumes` / `masks` are expected to be memory-mapped arrays (see
data/cache.py) — full-resolution stacks (hundreds of slices) do not fit
comfortably in RAM as plain numpy arrays, and pre-extracting every patch
into a Python list (the original approach) multiplies that problem further.

Instead, __init__ only *scans* each volume to classify candidate patch
locations as foreground/background (cheap — no patch data is copied or
retained), storing lightweight (volume_idx, y, x) coordinate tuples.
Actual pixel data is read from the memmap array lazily in __getitem__,
one patch at a time, keeping RAM usage roughly constant regardless of
dataset size.
=============================================================================
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PATCH_SIZE, PATCH_STRIDE, FG_BG_RATIO, MIN_FG_PIXELS,
    TRAIN_SLICES_PER_VOLUME, UNET_INPUT_SLICES,
)
from data.augmentation import apply_augmentation


class XCTPatchDataset(Dataset):
    """
    PyTorch Dataset that yields (image_patch, mask_patch) pairs extracted
    from one or more XCT volumes and their corresponding pseudo-label masks.

    Patch extraction strategy
    -------------------------
    All patch locations are found using a sliding window with stride
    PATCH_STRIDE. Locations are classified foreground (>= MIN_FG_PIXELS
    defect pixels) or background, then background locations are
    subsampled to FG_BG_RATIO so the model receives sufficient gradient
    signal from the minority defect class. Only coordinates are stored;
    pixel data is read on demand in __getitem__.

    Parameters
    ----------
    volumes          : list[np.ndarray]  — preprocessed float32 volumes (memmap or in-RAM)
    masks            : list[np.ndarray]  — binary uint8 pseudo-label masks (memmap or in-RAM)
    augment          : bool              — whether to apply augmentation (train only)
    split            : str               — "train", "val", or "test" (for logging)
    slices_per_volume : int | None       — evenly-spaced slices to sample per
                                            volume, instead of scanning every
                                            one. Defaults to
                                            config.TRAIN_SLICES_PER_VOLUME.
                                            Pass None to scan every slice.
    """

    def __init__(
        self,
        volumes : list[np.ndarray],
        masks   : list[np.ndarray],
        augment : bool = False,
        split   : str  = "train",
        slices_per_volume: "int | None" = TRAIN_SLICES_PER_VOLUME,
    ):
        self.augment = augment
        self.split   = split
        self.volumes = volumes
        self.masks   = masks
        self.slices_per_volume = slices_per_volume
        self._slice_cache: dict = {}
        self._slice_cache_max = 32

        fg_coords: list[tuple[int, int, int]] = []
        bg_coords: list[tuple[int, int, int]] = []

        for vol_idx, mask in enumerate(masks):
            self._scan_locations(vol_idx, mask, fg_coords, bg_coords)

        fg_count = len(fg_coords)
        bg_count = min(
            len(bg_coords),
            fg_count * FG_BG_RATIO[1] // FG_BG_RATIO[0]
        ) if fg_count > 0 else len(bg_coords)

        if bg_count > 0 and len(bg_coords) > 0:
            bg_indices = np.random.choice(
                len(bg_coords), size=bg_count, replace=False
            )
            bg_coords = [bg_coords[i] for i in bg_indices]
        else:
            bg_coords = []

        self.coords = fg_coords + bg_coords

        print(f"  [Dataset:{split}] FG patches : {fg_count}")
        print(f"  [Dataset:{split}] BG patches : {len(bg_coords)}  "
              f"(ratio {FG_BG_RATIO[0]}:{FG_BG_RATIO[1]})")
        print(f"  [Dataset:{split}] Total      : {len(self.coords)}")

    def _scan_locations(
        self,
        vol_idx: int,
        mask:    np.ndarray,
        fg_coords: list,
        bg_coords: list,
    ) -> None:
        """
        Classify every sliding-window location in `mask` as FG or BG.

        For 3D volumes, each 2D slice is scanned independently (the FG/BG
        label always comes from that slice's own mask — the neighbouring
        slices used for 2.5D input context in _get_slice_stack don't
        affect this classification). Only (vol_idx, slice_idx, y, x)
        coordinates are stored — no pixel data is retained here.

        If `self.slices_per_volume` is set, only that many evenly-spaced
        slices are scanned per volume instead of every one — a full
        900-slice volume otherwise makes __init__ itself (let alone
        training) take hours; see config.TRAIN_SLICES_PER_VOLUME.

        Each slice is pulled into RAM exactly once (`.copy()`) before the
        sliding window runs over it. `mask` may be a memory-mapped array
        (see data/cache.py); indexing it (`mask[i]`) returns a *view* still
        backed by the mapped file, not a materialised copy, so without this
        the 50%-overlap sliding window (PATCH_STRIDE < PATCH_SIZE) would
        fault the same regions of the file back in from disk repeatedly —
        a severe, easy-to-miss slowdown on a full-resolution volume.
        """
        P = PATCH_SIZE
        S = PATCH_STRIDE
        n_slices = 1 if mask.ndim == 2 else mask.shape[0]

        if mask.ndim == 2 or self.slices_per_volume is None:
            slice_indices = range(n_slices)
        else:
            n_sample = min(self.slices_per_volume, n_slices)
            slice_indices = np.linspace(0, n_slices - 1, n_sample, dtype=int)

        for slice_idx in slice_indices:
            slc_m = mask.copy() if mask.ndim == 2 else mask[slice_idx].copy()
            H, W = slc_m.shape

            for y in range(0, H - P + 1, S):
                for x in range(0, W - P + 1, S):
                    patch_m = slc_m[y:y+P, x:x+P]
                    coord = (vol_idx, slice_idx, y, x)
                    if int(patch_m.sum()) >= MIN_FG_PIXELS:
                        fg_coords.append(coord)
                    else:
                        bg_coords.append(coord)

    def _get_slice(self, vol_idx: int, slice_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the full (image, mask) slice for (vol_idx, slice_idx),
        materialised into RAM and cached — see class docstring on why a
        bare memmap view is unsafe to slice from repeatedly.
        """
        key = (vol_idx, slice_idx)
        cached = self._slice_cache.get(key)
        if cached is not None:
            return cached

        volume = self.volumes[vol_idx]
        mask   = self.masks[vol_idx]
        if volume.ndim == 2:
            slc_v, slc_m = volume.copy(), mask.copy()
        else:
            slc_v, slc_m = volume[slice_idx].copy(), mask[slice_idx].copy()

        if len(self._slice_cache) >= self._slice_cache_max:
            self._slice_cache.pop(next(iter(self._slice_cache)))
        self._slice_cache[key] = (slc_v, slc_m)
        return slc_v, slc_m

    def _get_slice_stack(
        self, vol_idx: int, slice_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the 2.5D input stack for (vol_idx, slice_idx): UNET_INPUT_SLICES
        adjacent images centred on slice_idx, stacked as (N, H, W), plus the
        single (H, W) mask for the centre slice only — the model predicts a
        mask for the centre slice alone; the neighbouring slices are extra
        input context.

        Neighbour indices past a volume's edge are clamped to the valid
        range (the boundary slice is effectively repeated) rather than
        zero-padded, so slices near the top/bottom of a volume still see
        real tissue context instead of blank padding. A genuinely 2D
        volume (mask.ndim == 2, no z-axis) has no neighbours to draw on,
        so its single slice is simply repeated UNET_INPUT_SLICES times.

        Reuses `_get_slice`'s per-(vol_idx, slice_idx) cache — sliding-
        window patches at the same slice_idx (the common case) request the
        same neighbour stack repeatedly, so only the first patch at a given
        slice actually re-reads each neighbour from the memmap.
        """
        volume = self.volumes[vol_idx]
        half = UNET_INPUT_SLICES // 2

        if volume.ndim == 2:
            slc_v, slc_m = self._get_slice(vol_idx, slice_idx)
            stack = np.repeat(slc_v[np.newaxis, ...], UNET_INPUT_SLICES, axis=0)
            return stack, slc_m

        n_slices = volume.shape[0]
        neighbour_idxs = [
            int(np.clip(slice_idx + offset, 0, n_slices - 1))
            for offset in range(-half, half + 1)
        ]

        fetched = [self._get_slice(vol_idx, i) for i in neighbour_idxs]
        stack   = np.stack([v for v, _m in fetched], axis=0)  # (N, H, W)
        _, slc_m = fetched[half]                               # centre slice's mask
        return stack, slc_m

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return an (image, mask) tensor pair for a given patch index.

        Augmentation is applied only during training (augment=True).
        Tensors are returned as:
            image : float32 tensor of shape (UNET_INPUT_SLICES, H, W) —
                    the 2.5D input stack (adjacent slices as channels)
            mask  : float32 tensor of shape (1, H, W) — centre-slice label
        """
        vol_idx, slice_idx, y, x = self.coords[idx]
        img_stack, slc_m = self._get_slice_stack(vol_idx, slice_idx)

        P = PATCH_SIZE
        image = np.array(img_stack[:, y:y+P, x:x+P], dtype=np.float32)  # (N, P, P)
        mask  = np.array(slc_m[y:y+P, x:x+P], dtype=np.uint8)

        if self.augment:
            image, mask = apply_augmentation(image, mask)
            # flips/rotations return negative-stride views — torch.from_numpy
            # can't wrap those directly.
            image = np.ascontiguousarray(image)
            mask  = np.ascontiguousarray(mask)

        image_t = torch.from_numpy(image).float()                         # (N, P, P)
        mask_t  = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)  # (1, P, P)
        return image_t, mask_t


class SliceGroupedShuffleSampler(Sampler):
    """
    Shuffles patches at the (volume, slice) level rather than individually.

    Plain per-index shuffling scatters the ~30 overlapping patches that
    share a slice across the entire epoch. With RAM too small to hold
    every training volume at once (see data/cache.py), that turns almost
    every batch into a fresh disk read of a slice that was *just* read for
    a different patch — the single-threaded data loader becomes the
    bottleneck and the GPU sits idle between batches.

    Instead: shuffle the order in which distinct slices are visited, and
    shuffle patches within each slice, then concatenate. Every slice is
    still read once (`XCTPatchDataset._get_slice`'s small LRU cache is
    then enough to serve every patch drawn from it), while randomisation
    across an epoch is preserved at the slice level — the granularity
    that actually matters for SGD on data this correlated already
    (adjacent patches from the same slice are highly correlated anyway).
    """

    def __init__(self, dataset: "XCTPatchDataset"):
        self.dataset = dataset
        groups: dict = {}
        for idx, (vol_idx, slice_idx, _y, _x) in enumerate(dataset.coords):
            groups.setdefault((vol_idx, slice_idx), []).append(idx)
        self.groups = list(groups.values())

    def __iter__(self):
        order = torch.randperm(len(self.groups)).tolist()
        for g in order:
            group = self.groups[g]
            local = torch.randperm(len(group)).tolist()
            for i in local:
                yield group[i]

    def __len__(self):
        return len(self.dataset)


def build_dataloaders(
    volumes_train : list[np.ndarray],
    masks_train   : list[np.ndarray],
    volumes_val   : list[np.ndarray],
    masks_val     : list[np.ndarray],
    batch_size    : int
) -> tuple:
    """
    Build training and validation DataLoaders.

    Returns
    -------
    tuple (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader

    train_ds = XCTPatchDataset(volumes_train, masks_train,
                                augment=True,  split="train")
    val_ds   = XCTPatchDataset(volumes_val,   masks_val,
                                augment=False, split="val")

    # num_workers=0: memmap arrays + fork-based worker processes on some
    # platforms don't play well together (each worker would re-open the
    # mmap, which is fine, but Windows uses spawn — keep this simple and
    # safe by default; raise if you've verified worker startup on your OS).
    #
    # sampler=SliceGroupedShuffleSampler instead of shuffle=True: see that
    # class's docstring — plain per-patch shuffling turns almost every
    # batch into a disk read against volumes too large to cache wholesale.
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=SliceGroupedShuffleSampler(train_ds),
                              num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0)
    return train_loader, val_loader
