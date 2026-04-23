"""
=============================================================================
Data Loader — XCT TIFF Stack Loading and Preprocessing
=============================================================================
Handles loading, normalisation, beam hardening correction,
ring artefact suppression, and non-local means denoising.
=============================================================================
"""

import os
import glob
import numpy as np
import tifffile as tiff
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy.ndimage import median_filter
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    NORM_LOW_PERCENTILE, NORM_HIGH_PERCENTILE, NORM_EPS,
    BHC_POLY_DEGREE, NLM_PATCH_SIZE, NLM_PATCH_DIST, NLM_H,
    RING_FILTER_RADIUS
)


def load_tiff_stack(folder: str) -> np.ndarray:
    """
    Load all TIFF files from a folder into a float32 numpy array.

    Parameters
    ----------
    folder : str
        Path to directory containing .tif slice files,
        named so that alphabetical sort gives correct slice order.

    Returns
    -------
    np.ndarray
        3D array (N_slices, H, W) or 2D array (H, W) for single slice.

    Raises
    ------
    RuntimeError
        If no .tif files are found.
    ValueError
        If slices have inconsistent spatial dimensions.
    """
    files = sorted(glob.glob(os.path.join(folder, "*.tif")))
    if not files:
        raise RuntimeError(f"No .tif files found in '{folder}'.")

    print(f"  [Loader] Found {len(files)} slice(s) in '{folder}'.")

    reference_shape = None
    slices = []
    for f in files:
        arr = tiff.imread(f).astype(np.float32)
        if reference_shape is None:
            reference_shape = arr.shape
        elif arr.shape != reference_shape:
            raise ValueError(
                f"Inconsistent slice shape at '{f}': "
                f"expected {reference_shape}, got {arr.shape}."
            )
        slices.append(arr)

    if len(slices) == 1:
        print("  [Loader] Single slice — 2D mode.")
        return slices[0]

    volume = np.stack(slices, axis=0)
    print(f"  [Loader] Volume shape: {volume.shape}  (slices × H × W).")
    return volume


def normalize_percentile(volume: np.ndarray) -> np.ndarray:
    """
    Clip and normalise volume to [0, 1] using robust percentile clipping.

    Reduces influence of extreme intensity outliers such as metal streak
    artefacts or hot pixels on the normalised intensity range.

    Parameters
    ----------
    volume : np.ndarray
        Raw float32 XCT volume (2D or 3D).

    Returns
    -------
    np.ndarray
        Normalised float32 array with values in [0, 1].
    """
    p_low  = np.percentile(volume, NORM_LOW_PERCENTILE)
    p_high = np.percentile(volume, NORM_HIGH_PERCENTILE)
    clipped    = np.clip(volume, p_low, p_high)
    normalised = (clipped - p_low) / (p_high - p_low + NORM_EPS)
    print(f"  [Norm] p{NORM_LOW_PERCENTILE}={p_low:.2f}  "
          f"p{NORM_HIGH_PERCENTILE}={p_high:.2f}  "
          f"→ range [{normalised.min():.4f}, {normalised.max():.4f}]")
    return normalised.astype(np.float32)


def correct_beam_hardening(volume: np.ndarray) -> np.ndarray:
    """
    Apply polynomial beam hardening correction.

    Fits a polynomial of degree BHC_POLY_DEGREE to the mean intensity
    profile along the beam axis (axis 0 for 3D, horizontal axis for 2D)
    and subtracts the fitted trend from the volume, flattening the
    systematic intensity gradient caused by differential X-ray attenuation.

    Parameters
    ----------
    volume : np.ndarray
        Normalised float32 volume (2D or 3D).

    Returns
    -------
    np.ndarray
        Beam-hardening-corrected float32 volume, clipped to [0, 1].
    """
    if volume.ndim == 3:
        # Mean intensity profile along depth axis
        profile = volume.mean(axis=(1, 2))              # Shape: (N_slices,)
        x       = np.arange(len(profile))
        coeffs  = np.polyfit(x, profile, BHC_POLY_DEGREE)
        trend   = np.polyval(coeffs, x)                 # Fitted polynomial
        # Subtract trend from each slice
        corrected = volume - (trend - trend.mean())[:, None, None]
    else:
        profile = volume.mean(axis=1)                   # Row-wise mean
        x       = np.arange(len(profile))
        coeffs  = np.polyfit(x, profile, BHC_POLY_DEGREE)
        trend   = np.polyval(coeffs, x)
        corrected = volume - (trend - trend.mean())[:, None]

    corrected = np.clip(corrected, 0.0, 1.0)
    print(f"  [BHC] Beam hardening correction applied "
          f"(polynomial degree {BHC_POLY_DEGREE}).")
    return corrected.astype(np.float32)


def suppress_ring_artefacts(volume: np.ndarray) -> np.ndarray:
    """
    Suppress ring artefacts using a polar-coordinate median filter.

    Converts each slice to polar coordinates, applies a median filter
    along the angular axis (where ring artefacts appear as horizontal
    bands), then converts back to Cartesian coordinates.

    Parameters
    ----------
    volume : np.ndarray
        Float32 volume (2D or 3D), values in [0, 1].

    Returns
    -------
    np.ndarray
        Ring-artefact-suppressed float32 volume.
    """
    def _suppress_slice(slc: np.ndarray) -> np.ndarray:
        h, w   = slc.shape
        cy, cx = h // 2, w // 2

        # Build polar coordinate arrays
        y_idx, x_idx = np.indices((h, w))
        r     = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
        theta = np.arctan2(y_idx - cy, x_idx - cx)

        # Discretise radius
        r_int = r.astype(np.int32)
        r_max = r_int.max() + 1

        # Compute radial mean profile and smooth it
        radial_mean = np.array([
            slc[r_int == ri].mean() if (r_int == ri).any() else 0.0
            for ri in range(r_max)
        ])
        radial_smooth = median_filter(radial_mean, size=RING_FILTER_RADIUS)

        # Subtract ring component
        correction = radial_smooth[np.clip(r_int, 0, r_max - 1)] - radial_mean[
            np.clip(r_int, 0, r_max - 1)
        ]
        corrected = slc + correction
        return np.clip(corrected, 0.0, 1.0).astype(np.float32)

    if volume.ndim == 3:
        corrected = np.stack([_suppress_slice(volume[i])
                              for i in range(volume.shape[0])], axis=0)
    else:
        corrected = _suppress_slice(volume)

    print("  [Ring] Ring artefact suppression applied.")
    return corrected


def denoise_nlm(volume: np.ndarray) -> np.ndarray:
    """
    Apply non-local means (NLM) denoising to improve signal-to-noise ratio.

    NLM replaces each voxel intensity with a weighted average of voxels
    in a search window, where weights are determined by patch-level
    intensity similarity. This preserves defect boundaries while
    suppressing random noise.

    Parameters
    ----------
    volume : np.ndarray
        Float32 volume (2D or 3D), values in [0, 1].

    Returns
    -------
    np.ndarray
        Denoised float32 volume.
    """
    if volume.ndim == 3:
        denoised = np.stack([
            denoise_nl_means(
                volume[i],
                h              = NLM_H * estimate_sigma(volume[i]),
                patch_size     = NLM_PATCH_SIZE,
                patch_distance = NLM_PATCH_DIST,
                fast_mode      = True
            ).astype(np.float32)
            for i in range(volume.shape[0])
        ], axis=0)
    else:
        sigma    = estimate_sigma(volume)
        denoised = denoise_nl_means(
            volume,
            h              = NLM_H * sigma,
            patch_size     = NLM_PATCH_SIZE,
            patch_distance = NLM_PATCH_DIST,
            fast_mode      = True
        ).astype(np.float32)

    print("  [NLM] Non-local means denoising applied.")
    return denoised


def full_preprocess(volume_raw: np.ndarray) -> np.ndarray:
    """
    Apply the complete preprocessing pipeline to a raw XCT volume.

    Pipeline stages (matching thesis Section 3.3.1):
        1. Percentile normalisation
        2. Beam hardening correction
        3. Ring artefact suppression
        4. Non-local means denoising

    Parameters
    ----------
    volume_raw : np.ndarray
        Raw float32 XCT volume loaded from TIFF stack.

    Returns
    -------
    np.ndarray
        Fully preprocessed float32 volume in [0, 1].
    """
    print("  [Preprocess] Starting full preprocessing pipeline ...")
    v = normalize_percentile(volume_raw)
    v = correct_beam_hardening(v)
    v = suppress_ring_artefacts(v)
    v = denoise_nlm(v)
    print("  [Preprocess] Done.")
    return v
