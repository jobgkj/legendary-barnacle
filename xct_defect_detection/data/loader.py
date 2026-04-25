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
import time
import warnings
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
        Path to directory containing .tif or .tiff slice files,
        named so that alphabetical sort gives correct slice order.

    Returns
    -------
    np.ndarray
        3D array (N_slices, H, W) or 2D array (H, W) for single slice.

    Raises
    ------
    RuntimeError
        If no .tif / .tiff files are found.
    ValueError
        If slices have inconsistent spatial dimensions.
    """
    # Support both .tif and .tiff extensions
    files = sorted(
        glob.glob(os.path.join(folder, "*.tif")) +
        glob.glob(os.path.join(folder, "*.tiff"))
    )

    if not files:
        raise RuntimeError(
            f"No .tif / .tiff files found in '{folder}'.\n"
            f"  → Check that the path exists and contains TIFF slices."
        )

    print(f"  [Loader] Found {len(files)} slice(s) in '{folder}'.")

    reference_shape = None
    slices = []
    for i, f in enumerate(files):
        arr = tiff.imread(f).astype(np.float32)

        # Handle RGB/RGBA TIFFs — convert to grayscale
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            warnings.warn(
                f"Slice '{os.path.basename(f)}' is RGB/RGBA — converting to grayscale.",
                UserWarning
            )
            arr = arr[..., :3].mean(axis=2).astype(np.float32)

        if reference_shape is None:
            reference_shape = arr.shape
        elif arr.shape != reference_shape:
            raise ValueError(
                f"Inconsistent slice shape at '{f}':\n"
                f"  expected {reference_shape}, got {arr.shape}."
            )

        slices.append(arr)

        # Progress every 10 slices
        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            print(f"  [Loader] Loaded {i + 1}/{len(files)} slices ...", end="\r")

    print()  # newline after progress

    if len(slices) == 1:
        print("  [Loader] Single slice detected — running in 2D mode.")
        return slices[0]

    volume = np.stack(slices, axis=0)
    print(f"  [Loader] Volume shape: {volume.shape}  (slices × H × W).")
    print(f"  [Loader] Intensity range: [{volume.min():.2f}, {volume.max():.2f}]")
    return volume


def normalize_percentile(volume: np.ndarray) -> np.ndarray:
    """
    Clip and normalise volume to [0, 1] using robust percentile clipping.

    Reduces influence of extreme intensity outliers (metal streaks, hot pixels).

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

    if p_high - p_low < NORM_EPS:
        warnings.warn(
            "Normalisation range is near-zero — volume may be blank or constant.",
            UserWarning
        )

    clipped    = np.clip(volume, p_low, p_high)
    normalised = (clipped - p_low) / (p_high - p_low + NORM_EPS)

    print(f"  [Norm] p{NORM_LOW_PERCENTILE}={p_low:.4f}  "
          f"p{NORM_HIGH_PERCENTILE}={p_high:.4f}  "
          f"→ output range [{normalised.min():.4f}, {normalised.max():.4f}]")
    return normalised.astype(np.float32)


def correct_beam_hardening(volume: np.ndarray) -> np.ndarray:
    """
    Apply polynomial beam hardening correction.

    Fits a polynomial of degree BHC_POLY_DEGREE to the mean intensity
    profile and subtracts the fitted trend, flattening the systematic
    intensity gradient caused by differential X-ray attenuation.

    Parameters
    ----------
    volume : np.ndarray
        Normalised float32 volume (2D or 3D).

    Returns
    -------
    np.ndarray
        Beam-hardening-corrected float32 volume, clipped to [0, 1].
    """
    t0 = time.time()

    if volume.ndim == 3:
        profile = volume.mean(axis=(1, 2))          # (N_slices,)
        x       = np.arange(len(profile))
        coeffs  = np.polyfit(x, profile, BHC_POLY_DEGREE)
        trend   = np.polyval(coeffs, x)
        corrected = volume - (trend - trend.mean())[:, None, None]
    else:
        profile = volume.mean(axis=1)               # Row-wise mean
        x       = np.arange(len(profile))
        coeffs  = np.polyfit(x, profile, BHC_POLY_DEGREE)
        trend   = np.polyval(coeffs, x)
        corrected = volume - (trend - trend.mean())[:, None]

    corrected = np.clip(corrected, 0.0, 1.0)
    print(f"  [BHC] Beam hardening correction applied "
          f"(degree {BHC_POLY_DEGREE})  [{time.time()-t0:.1f}s]")
    return corrected.astype(np.float32)


def suppress_ring_artefacts(volume: np.ndarray) -> np.ndarray:
    """
    Suppress ring artefacts using a radial median filter.

    Computes the radial mean profile per slice, smooths it with a median
    filter, and subtracts the ring component from each pixel based on its
    radial distance from the image centre.

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

        y_idx, x_idx = np.indices((h, w))
        r     = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
        r_int = r.astype(np.int32)
        r_max = r_int.max() + 1

        # Vectorised radial mean (faster than loop)
        radial_mean = np.zeros(r_max, dtype=np.float32)
        for ri in range(r_max):
            mask = r_int == ri
            if mask.any():
                radial_mean[ri] = slc[mask].mean()

        radial_smooth = median_filter(radial_mean, size=RING_FILTER_RADIUS)
        r_clipped  = np.clip(r_int, 0, r_max - 1)
        correction = radial_smooth[r_clipped] - radial_mean[r_clipped]
        return np.clip(slc + correction, 0.0, 1.0).astype(np.float32)

    t0 = time.time()
    if volume.ndim == 3:
        corrected = np.stack([
            _suppress_slice(volume[i]) for i in range(volume.shape[0])
        ], axis=0)
    else:
        corrected = _suppress_slice(volume)

    print(f"  [Ring] Ring artefact suppression applied  [{time.time()-t0:.1f}s]")
    return corrected


def denoise_nlm(volume: np.ndarray) -> np.ndarray:
    """
    Apply non-local means (NLM) denoising slice-by-slice.

    NLM preserves defect boundaries while suppressing random noise by
    replacing each voxel with a weighted average based on patch similarity.

    Parameters
    ----------
    volume : np.ndarray
        Float32 volume (2D or 3D), values in [0, 1].

    Returns
    -------
    np.ndarray
        Denoised float32 volume.
    """
    t0 = time.time()

    def _denoise_slice(slc: np.ndarray) -> np.ndarray:
        sigma = estimate_sigma(slc)
        if sigma < 1e-6:
            warnings.warn("Near-zero sigma estimated — skipping NLM for this slice.", UserWarning)
            return slc
        return denoise_nl_means(
            slc,
            h              = NLM_H * sigma,
            patch_size     = NLM_PATCH_SIZE,
            patch_distance = NLM_PATCH_DIST,
            fast_mode      = True,
            preserve_range = True
        ).astype(np.float32)

    if volume.ndim == 3:
        n = volume.shape[0]
        denoised_slices = []
        for i in range(n):
            denoised_slices.append(_denoise_slice(volume[i]))
            print(f"  [NLM] Denoising slice {i+1}/{n} ...", end="\r")
        print()
        denoised = np.stack(denoised_slices, axis=0)
    else:
        denoised = _denoise_slice(volume)

    print(f"  [NLM] Non-local means denoising complete  [{time.time()-t0:.1f}s]")
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
    t_start = time.time()
    print("\n  [Preprocess] ── Starting full preprocessing pipeline ──")
    v = normalize_percentile(volume_raw)
    v = correct_beam_hardening(v)
    v = suppress_ring_artefacts(v)
    v = denoise_nlm(v)
    print(f"  [Preprocess] ── Done  (total: {time.time()-t_start:.1f}s) ──\n")
    return v
