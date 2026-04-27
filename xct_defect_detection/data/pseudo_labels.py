"""
=============================================================================
Pseudo-Label Generator
=============================================================================
Since no ground truth masks are available for either NIST or PODFAM data,
this module generates binary defect masks automatically using Otsu
thresholding followed by morphological cleaning and connected component
filtering.

Generated masks are saved as TIFF files alongside the volume data so they
can be reused across training runs without regeneration.
=============================================================================
"""

import os
import numpy as np
import tifffile as tiff
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from scipy.ndimage import binary_opening, binary_closing
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MORPH_OPEN_SIZE, MORPH_CLOSE_SIZE, MIN_DEFECT_SIZE


def generate_otsu_mask(volume: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Generate a binary defect mask using Otsu global thresholding.

    In XCT data of metal AM components, defect voxels (gas pores, LoF
    voids) correspond to lower X-ray attenuation and appear as darker
    regions. The defect mask is defined as voxels BELOW the Otsu
    threshold.

    Parameters
    ----------
    volume : np.ndarray
        Preprocessed float32 volume in [0, 1] (2D or 3D).

    Returns
    -------
    mask : np.ndarray
        Binary uint8 mask (1 = defect, 0 = matrix).
    threshold : float
        Applied Otsu threshold value.
    """
    threshold = threshold_otsu(volume)
    mask      = (volume < threshold).astype(np.uint8)
    return mask, threshold


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Remove noise from a binary mask using morphological operations.

    - Opening: removes isolated noise voxels (spurious foreground)
    - Closing: fills small holes within genuine defect regions

    Parameters
    ----------
    mask : np.ndarray
        Raw binary mask (uint8, 2D or 3D).

    Returns
    -------
    np.ndarray
        Cleaned binary mask (uint8).
    """
    if mask.ndim == 3:
        struct_open  = np.ones(
            (MORPH_OPEN_SIZE,  MORPH_OPEN_SIZE,  MORPH_OPEN_SIZE),  dtype=bool)
        struct_close = np.ones(
            (MORPH_CLOSE_SIZE, MORPH_CLOSE_SIZE, MORPH_CLOSE_SIZE), dtype=bool)
    else:
        struct_open  = np.ones((MORPH_OPEN_SIZE,  MORPH_OPEN_SIZE),  dtype=bool)
        struct_close = np.ones((MORPH_CLOSE_SIZE, MORPH_CLOSE_SIZE), dtype=bool)

    opened = binary_opening(mask,   structure=struct_open)
    closed = binary_closing(opened, structure=struct_close)
    return closed.astype(np.uint8)


def filter_small_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Remove connected components smaller than MIN_DEFECT_SIZE voxels.

    This removes noise artefacts that survive morphological cleaning,
    such as single-voxel detections caused by beam hardening gradients.

    Parameters
    ----------
    mask : np.ndarray
        Cleaned binary mask (uint8, 2D or 3D).

    Returns
    -------
    filtered_mask : np.ndarray
        Binary mask with small components removed (uint8).
    n_defects : int
        Number of retained connected defect regions.
    """
    labelled     = label(mask, connectivity=mask.ndim)
    props        = regionprops(labelled)
    filtered     = np.zeros_like(mask)
    n_defects    = 0

    for region in props:
        if region.area >= MIN_DEFECT_SIZE:
            filtered[labelled == region.label] = 1
            n_defects += 1

    return filtered.astype(np.uint8), n_defects


def generate_pseudo_label(volume: np.ndarray) -> dict:
    """
    Run the full pseudo-label generation pipeline on a preprocessed volume.

    Pipeline:
        1. Otsu global thresholding
        2. Morphological opening + closing
        3. Small component removal

    Parameters
    ----------
    volume : np.ndarray
        Preprocessed float32 volume in [0, 1] (2D or 3D).

    Returns
    -------
    dict with keys:
        'mask'            : np.ndarray  — final binary pseudo-label mask
        'threshold'       : float       — applied Otsu threshold
        'n_defects'       : int         — number of retained defect regions
        'defect_fraction' : float       — fraction of voxels labelled as defect
    """
    mask_raw,  threshold = generate_otsu_mask(volume)
    mask_clean           = clean_mask(mask_raw)
    mask_final, n_defects = filter_small_components(mask_clean)

    defect_fraction = float(mask_final.sum()) / mask_final.size

    return {
        "mask"            : mask_final,
        "threshold"       : threshold,
        "n_defects"       : n_defects,
        "defect_fraction" : defect_fraction,
    }


def save_mask_as_tiff(mask: np.ndarray, save_path: str) -> None:
    """
    Save a binary mask as a TIFF file (or TIFF stack for 3D).

    Parameters
    ----------
    mask      : np.ndarray  — binary uint8 mask (2D or 3D)
    save_path : str         — full file path including .tif extension
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tiff.imwrite(save_path, mask)


def generate_and_save_pseudo_labels(
    vol_dir:  str,
    mask_dir: str,
    volumes:  dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """
    Generate pseudo-label masks for all volumes and save them to disk.

    If a mask file already exists for a volume, it is loaded from disk
    rather than regenerated, avoiding redundant computation across runs.

    Parameters
    ----------
    vol_dir  : str
        Directory where volume TIFF stacks were loaded from.
    mask_dir : str
        Directory where generated masks will be saved.
    volumes  : dict[str, np.ndarray]
        Dictionary mapping volume name → preprocessed float32 volume.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping volume name → binary pseudo-label mask.
    """
    masks = {}

    for name, volume in volumes.items():
        mask_path = os.path.join(mask_dir, f"{name}_mask.tif")

        if os.path.exists(mask_path):
            print(f"  [PseudoLabel] Loading cached mask for '{name}' ...")
            masks[name] = tiff.imread(mask_path).astype(np.uint8)
            continue

        print(f"  [PseudoLabel] Generating pseudo-label for '{name}' ...")
        result = generate_pseudo_label(volume)
        mask   = result["mask"]

        save_mask_as_tiff(mask, mask_path)
        masks[name] = mask

        print(f"    Otsu threshold   : {result['threshold']:.4f}")
        print(f"    Defect fraction  : {result['defect_fraction']*100:.3f}%")
        print(f"    Retained defects : {result['n_defects']}")

    return masks
