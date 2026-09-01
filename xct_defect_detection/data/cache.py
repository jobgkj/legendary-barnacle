"""
=============================================================================
cache.py — Memory-mapped volume/mask cache for training
=============================================================================
Converts a sample's per-slice TIFF stack (data/processed/<sample>/ and
data/masks/<sample>/bernsen/) into two on-disk .npy arrays that can be
opened with mmap_mode="r" — so training never needs the full volume
resident in RAM, regardless of how many slices the sample has.

Cache files:
    data/cache/<sample>_volume.npy   float32, values in [0, 1], (N, H, W)
    data/cache/<sample>_mask.npy     uint8,   0/1,               (N, H, W)

Building the cache still touches each slice only once (O(1) RAM) — writes
go straight to the memory-mapped file rather than through a Python list.
=============================================================================
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tifffile as tiff

import config


def _volume_cache_path(sample_name: str) -> Path:
    return config.CACHE_DIR / f"{sample_name}_volume.npy"


def _mask_cache_path(sample_name: str) -> Path:
    return config.CACHE_DIR / f"{sample_name}_mask.npy"


def build_cache(
    sample_name: str,
    mask_method: str = "bernsen",
    force: bool = False,
) -> tuple[Path, Path]:
    """
    Build (or reuse) the memmap cache for one sample.

    Parameters
    ----------
    sample_name : str
        Sample folder name under data/processed/ and data/masks/.
    mask_method : str
        Which thresholding method's masks to use as training labels.
        Defaults to "bernsen" (the field-recommended primary method —
        see src/thresholding.py).
    force : bool
        Rebuild even if cache files already exist.

    Returns
    -------
    (volume_path, mask_path)
    """
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    vol_path  = _volume_cache_path(sample_name)
    mask_path = _mask_cache_path(sample_name)

    if vol_path.exists() and mask_path.exists() and not force:
        return vol_path, mask_path

    proc_dir = config.PROCESSED_DATA_DIR / sample_name
    mask_dir = config.MASKS_DIR / sample_name / mask_method

    proc_files = sorted(
        list(proc_dir.glob("*.tif")) + list(proc_dir.glob("*.tiff"))
    )
    if not proc_files:
        raise FileNotFoundError(
            f"No preprocessed slices found in {proc_dir}. "
            "Run scripts/run_preprocess.py first."
        )

    mask_files = sorted(
        list(mask_dir.glob("*.tif")) + list(mask_dir.glob("*.tiff"))
    )
    if not mask_files:
        raise FileNotFoundError(
            f"No '{mask_method}' masks found in {mask_dir}. "
            "Run scripts/run_all_samples.py (or src.io.load_and_generate_masks) first."
        )
    if len(proc_files) != len(mask_files):
        raise ValueError(
            f"Slice count mismatch for '{sample_name}': "
            f"{len(proc_files)} preprocessed vs {len(mask_files)} masks."
        )

    n = len(proc_files)
    first = tiff.imread(proc_files[0])
    h, w = first.shape

    vol_mm = np.lib.format.open_memmap(
        vol_path, mode="w+", dtype=np.float32, shape=(n, h, w)
    )
    mask_mm = np.lib.format.open_memmap(
        mask_path, mode="w+", dtype=np.uint8, shape=(n, h, w)
    )

    for i, (pf, mf) in enumerate(zip(proc_files, mask_files)):
        img = tiff.imread(pf)
        vol_mm[i] = img.astype(np.float32) / 255.0

        m = tiff.imread(mf)
        mask_mm[i] = (m > 0).astype(np.uint8)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  [Cache] {sample_name}: {i+1}/{n} slices cached", end="\r")
    print()

    vol_mm.flush()
    mask_mm.flush()
    del vol_mm, mask_mm

    return vol_path, mask_path


def load_cache(sample_name: str, mask_method: str = "bernsen") -> tuple[np.ndarray, np.ndarray]:
    """
    Build the cache if needed, then open both arrays memory-mapped
    (mmap_mode="r") — near-zero RAM cost regardless of volume size.

    Returns
    -------
    (volume, mask) — np.memmap arrays, shape (N, H, W)
    """
    vol_path, mask_path = build_cache(sample_name, mask_method=mask_method)
    volume = np.load(vol_path, mmap_mode="r")
    mask   = np.load(mask_path, mmap_mode="r")
    return volume, mask
