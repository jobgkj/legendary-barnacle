"""
=============================================================================
preprocess.py — XCT slice-wise preprocessing
=============================================================================

Pipeline (per slice)
--------------------
1. Median filtering       — speckle noise removal
2. Non-Local Means (NLM)  — edge-preserving denoising
3. Intensity normalization — uint8 [0, 255]

Design goals
------------
- 2D only (slice-wise)
- Constant memory usage
- No file I/O side effects
- Config-driven defaults, all overridable per call
- GUI-friendly: get_default_params() for field pre-population
=============================================================================
"""

import warnings
from pathlib import Path

import numpy as np
import tifffile as tiff
from scipy.ndimage import median_filter
from skimage.restoration import denoise_nl_means, estimate_sigma

import config


# =============================================================================
# GUI helper
# =============================================================================

def get_default_params() -> dict:
    """
    Return current preprocessing defaults from config.

    GUI can call this to pre-populate its input fields,
    then pass updated values back to preprocess_slice().

    Returns
    -------
    dict with keys:
        use_nlm        : bool
        median_size    : int
        nlm_h_factor   : float  — h = nlm_h_factor * estimated_sigma
        patch_size     : int
        patch_distance : int
    """
    return {
        "use_nlm":        True,
        "median_size":    config.MEDIAN_KERNEL_SIZE,
        "nlm_h_factor":   config.NLM_H_FACTOR,
        "patch_size":     config.NLM_PATCH_SIZE,
        "patch_distance": config.NLM_PATCH_DIST,
    }


# =============================================================================
# Stack-wide normalisation range
# =============================================================================

def estimate_stack_norm_range(
    tiff_files:     list,
    n_sample_slices: int   = None,
    low_percentile:  float = None,
    high_percentile: float = None,
) -> tuple:
    """
    Estimate a single (vmin, vmax) intensity range for an entire TIFF
    stack, from raw pixel values pooled across evenly-spaced sample
    slices.

    Used so every slice in a stack is normalised against the same
    range (see preprocess_slice's vmin/vmax parameters), instead of
    each slice independently stretching its own intensity spread to
    fill the output range — see config.STACK_NORM_N_SAMPLE_SLICES for
    why that matters.

    Percentiles are computed on RAW pixel values (before median/NLM
    filtering) as a cheap proxy for the post-filter range — those
    filters smooth noise but don't materially shift the underlying
    intensity scale, and computing this from raw data avoids running
    the expensive NLM step twice per sample.

    Parameters
    ----------
    tiff_files : list of Path
        Sorted list of TIFF file paths for one sample (same list
        passed to detect_sample_mask_stack).
    n_sample_slices : int, optional
        Number of evenly-spaced slices to sample.
        Defaults to config.STACK_NORM_N_SAMPLE_SLICES.
    low_percentile, high_percentile : float, optional
        Defaults to config.NORM_LOW_PERCENTILE / NORM_HIGH_PERCENTILE.

    Returns
    -------
    tuple (vmin, vmax)
        Pooled percentile bounds across the sampled slices, as floats.

    Raises
    ------
    ValueError
        If no slice could be read.
    """
    if n_sample_slices is None: n_sample_slices = config.STACK_NORM_N_SAMPLE_SLICES
    if low_percentile  is None: low_percentile  = config.NORM_LOW_PERCENTILE
    if high_percentile is None: high_percentile = config.NORM_HIGH_PERCENTILE

    n = len(tiff_files)
    n_sample_slices = min(n_sample_slices, n)
    indices = np.linspace(0, n - 1, n_sample_slices, dtype=int)

    pooled = []
    for idx in indices:
        try:
            raw = tiff.imread(tiff_files[idx])
            if raw.ndim != 2:
                continue
            pooled.append(raw.astype(np.float32).ravel())
        except Exception as e:
            warnings.warn(f"estimate_stack_norm_range: skipping slice {idx}: {e}")
            continue

    if not pooled:
        raise ValueError(
            "estimate_stack_norm_range: could not read any sample slice "
            f"from {n} file(s)."
        )

    pooled = np.concatenate(pooled)
    vmin = float(np.percentile(pooled, low_percentile))
    vmax = float(np.percentile(pooled, high_percentile))

    return vmin, vmax


# =============================================================================
# Beam hardening correction (BHC) — volume-level pre-pass
# =============================================================================
#
# Unlike median/NLM filtering (2D, per-slice), beam hardening correction
# needs the depth-wise (slice-to-slice) intensity trend across the WHOLE
# stack to fit against — a single slice has no way to know where it sits
# on that trend. This mirrors estimate_stack_norm_range's two-pass
# design: sample N slices to fit a cheap model of the whole stack, then
# apply a per-slice correction derived from that model to every slice
# without re-reading the stack a second time.
#
# Operates on RAW pixel values (before median/NLM/normalisation), since
# beam hardening is a property of the raw detector signal, not an
# artefact of later filtering steps.

def estimate_stack_bhc_correction(
    tiff_files:      list,
    degree:          int = None,
    n_sample_slices: int = None,
) -> np.ndarray:
    """
    Fit a polynomial trend to the mean raw intensity across depth
    (slice index), sampled from N evenly-spaced slices, and evaluate
    a zero-mean correction value for every slice in the stack.

    Parameters
    ----------
    tiff_files : list of Path
        Sorted list of TIFF file paths for one sample.
    degree : int, optional
        Polynomial degree. Defaults to config.BHC_POLY_DEGREE.
    n_sample_slices : int, optional
        Number of evenly-spaced slices to sample when fitting the
        trend. Defaults to config.STACK_NORM_N_SAMPLE_SLICES.

    Returns
    -------
    np.ndarray
        Array of length len(tiff_files); correction[i] is the value
        to SUBTRACT from slice i's raw intensity to flatten the
        depth-wise trend (zero-mean across the sampled fit, so total
        volume brightness is preserved on average).
    """
    if degree is None:
        degree = config.BHC_POLY_DEGREE
    if n_sample_slices is None:
        n_sample_slices = config.STACK_NORM_N_SAMPLE_SLICES

    n = len(tiff_files)
    n_sample = min(n_sample_slices, n)
    sample_idx = np.linspace(0, n - 1, n_sample, dtype=int)

    means = []
    valid_idx = []
    for idx in sample_idx:
        try:
            raw = tiff.imread(tiff_files[idx])
            if raw.ndim != 2:
                continue
            means.append(float(raw.astype(np.float32).mean()))
            valid_idx.append(idx)
        except Exception as e:
            warnings.warn(f"estimate_stack_bhc_correction: skipping slice {idx}: {e}")
            continue

    if len(valid_idx) < degree + 1:
        warnings.warn(
            "estimate_stack_bhc_correction: not enough valid slices to fit "
            f"degree-{degree} trend — returning zero correction."
        )
        return np.zeros(n, dtype=np.float32)

    coeffs = np.polyfit(np.array(valid_idx, dtype=np.float64), np.array(means), degree)
    full_idx = np.arange(n, dtype=np.float64)
    trend_full = np.polyval(coeffs, full_idx)
    correction = (trend_full - trend_full.mean()).astype(np.float32)
    return correction


# =============================================================================
# Ring artefact suppression — per-slice, raw intensity
# =============================================================================

def apply_ring_suppression(img: np.ndarray, radius: int = None) -> np.ndarray:
    """
    Suppress ring artefacts via a polar-coordinate radial median filter.

    Ring artefacts appear as roughly circular bands centred on the
    rotation axis (image centre). Converting to polar coordinates turns
    each ring into a near-constant band along the radial axis, where a
    1D median filter can smooth it out without blurring genuine
    angularly-varying structure (e.g. pores).

    Parameters
    ----------
    img : np.ndarray
        2D raw-scale slice (any numeric dtype; converted to float32).
    radius : int, optional
        Radial median filter window size. Defaults to
        config.RING_FILTER_RADIUS.

    Returns
    -------
    np.ndarray
        float32 image with the fitted radial-mean correction applied.
    """
    if radius is None:
        radius = config.RING_FILTER_RADIUS

    img = img.astype(np.float32)
    h, w = img.shape
    cy, cx = h // 2, w // 2
    yi, xi = np.indices((h, w))
    r = np.sqrt((xi - cx) ** 2 + (yi - cy) ** 2).astype(np.int32)
    r_max = int(r.max()) + 1

    counts = np.bincount(r.ravel(), minlength=r_max)
    sums = np.bincount(r.ravel(), weights=img.ravel(), minlength=r_max)
    radial_mean = np.zeros(r_max, dtype=np.float32)
    valid = counts > 0
    radial_mean[valid] = sums[valid] / counts[valid]

    radial_smoothed = median_filter(radial_mean, size=radius)
    correction = radial_smoothed[r] - radial_mean[r]
    return img + correction


# =============================================================================
# Core preprocessing function
# =============================================================================

def preprocess_slice(
    img:            np.ndarray,
    use_nlm:        bool  = None,
    median_size:    int   = None,
    nlm_h_factor:   float = None,
    patch_size:     int   = None,
    patch_distance: int   = None,
    vmin:           float = None,
    vmax:           float = None,
) -> np.ndarray:
    """
    Preprocess a single 2D XCT slice.

    All parameters fall back to config.py values if not supplied.
    This makes the function safe to call from a GUI where the user
    may only change a subset of parameters.

    Parameters
    ----------
    img : np.ndarray
        Input 2D XCT slice. Any numeric dtype accepted.
    use_nlm : bool, optional
        Enable Non-Local Means denoising.
        Defaults to True.
    median_size : int, optional
        Kernel size for median filter.
        Defaults to config.MEDIAN_KERNEL_SIZE.
    nlm_h_factor : float, optional
        NLM filter strength as a multiple of the estimated noise sigma.
        h = nlm_h_factor * sigma  (adaptive — recommended for XCT).
        Defaults to config.NLM_H_FACTOR.
    patch_size : int, optional
        NLM patch size (pixels).
        Defaults to config.NLM_PATCH_SIZE.
    patch_distance : int, optional
        NLM patch search radius (pixels).
        Defaults to config.NLM_PATCH_DIST.
    vmin, vmax : float, optional
        Intensity bounds to clip-and-rescale to [0, 1] before the final
        uint8 conversion (equivalent to a percentile clip when computed
        via estimate_stack_norm_range). When both are given, they are
        used directly instead of computing a range from this slice
        alone — pass the same stack-wide (vmin, vmax) for every slice
        in a stack (see estimate_stack_norm_range) so a low-contrast
        slice isn't independently stretched to look like a
        full-contrast one. When not given, falls back to this slice's
        own (config.NORM_LOW_PERCENTILE, config.NORM_HIGH_PERCENTILE)
        percentiles — still more robust to outlier pixels than a bare
        min/max, but only as good as one slice's own statistics.

    Returns
    -------
    np.ndarray
        Preprocessed uint8 image in range [0, 255].
        Returns a zero-filled image with a warning if the
        intensity range is zero (blank slice).

    Raises
    ------
    ValueError
        If img is not 2D.
    TypeError
        If img is not a numeric array.
    """

    # ------------------------------------------------------------------
    # Fall back to config for any unset parameters
    # ------------------------------------------------------------------
    if median_size    is None: median_size    = config.MEDIAN_KERNEL_SIZE
    if nlm_h_factor   is None: nlm_h_factor   = config.NLM_H_FACTOR
    if patch_size     is None: patch_size     = config.NLM_PATCH_SIZE
    if patch_distance is None: patch_distance = config.NLM_PATCH_DIST
    if use_nlm        is None: use_nlm        = True

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(img).__name__}.")

    if not np.issubdtype(img.dtype, np.number):
        raise TypeError(f"Expected numeric image, got dtype {img.dtype}.")

    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}.")

    # ------------------------------------------------------------------
    # Convert to float32 for all processing
    # ------------------------------------------------------------------
    img = img.astype(np.float32)

    # ------------------------------------------------------------------
    # Step 1 — Median filter (speckle noise removal)
    # ------------------------------------------------------------------
    img = median_filter(img, size=median_size)

    # ------------------------------------------------------------------
    # Step 2 — Non-Local Means (edge-preserving denoising)
    # ------------------------------------------------------------------
    if use_nlm:
        sigma = estimate_sigma(img, channel_axis=None)

        if sigma > 0:
            img = denoise_nl_means(
                img,
                h=nlm_h_factor * sigma,
                patch_size=patch_size,
                patch_distance=patch_distance,
                fast_mode=True,
                channel_axis=None,
            )
        else:
            warnings.warn(
                "Estimated noise sigma is zero — skipping NLM denoising."
            )

    # ------------------------------------------------------------------
    # Step 3 — Percentile-clip and normalize to uint8 [0, 255]
    # ------------------------------------------------------------------
    if vmin is None or vmax is None:
        # No stack-wide range given — fall back to this slice's own
        # percentiles (still more robust than a bare min/max, but see
        # the vmin/vmax docstring above for why a shared stack-wide
        # range is preferred whenever one is available).
        vmin = float(np.percentile(img, config.NORM_LOW_PERCENTILE))
        vmax = float(np.percentile(img, config.NORM_HIGH_PERCENTILE))

    if vmax <= vmin:
        warnings.warn(
            "Slice has zero intensity range — returning blank slice."
        )
        return np.zeros(img.shape, dtype=np.uint8)

    img = (img - vmin) / (vmax - vmin + config.NORM_EPS)
    return (255.0 * img).clip(0, 255).astype(np.uint8)
