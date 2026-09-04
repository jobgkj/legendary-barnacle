"""
=============================================================================
sample_mask.py — Automatic circular sample mask detection for XCT data
=============================================================================

Detects the cylindrical sample boundary in XCT slices and generates
a binary sample mask:
    1 = inside sample  (valid region for thresholding)
    0 = outside sample (air background — forced to solid in all masks)

This prevents air background pixels (dark, outside the cylinder) from
being misclassified as pores, which was causing:
  - Over-estimated porosity
  - Inverted-looking 3D visualisations

Method
------
1. Threshold the image with a low global threshold to separate
   air (very dark) from sample (bright)
2. Find the largest connected component — this is the sample
3. Fit a circle using the component's centroid and equivalent radius
4. Return a filled circular mask

Usage
-----
    from src.sample_mask import detect_sample_mask
    from src.thresholding import bernsen

    sample_mask  = detect_sample_mask(preprocessed_slice)
    mask         = bernsen(preprocessed_slice, sample_mask=sample_mask)

Reference
---------
Kim et al. (2017), Additive Manufacturing.
doi:10.1016/j.addma.2017.06.011
=============================================================================
"""

import warnings
import numpy as np
import tifffile as tiff
from scipy.ndimage import binary_fill_holes, gaussian_filter1d
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.measure import find_contours, label, regionprops
from skimage.morphology import binary_erosion, binary_closing, disk
from skimage.transform import resize

import config


# =============================================================================
# GUI helper
# =============================================================================

def get_default_params() -> dict:
    """
    Return current sample mask defaults from config.

    Returns
    -------
    dict with keys:
        sample_mask_erosion_radius : int
        sample_mask_air_threshold  : int
    """
    return {
        "sample_mask_erosion_radius": config.SAMPLE_MASK_EROSION_RADIUS,
        "sample_mask_air_threshold":  config.SAMPLE_MASK_AIR_THRESHOLD,
    }


# =============================================================================
# Core function
# =============================================================================

def detect_sample_mask(
    img:                 np.ndarray,
    erosion_radius:      int  = None,
    air_threshold:       int  = None,
    return_circle:       bool = True,
    adaptive_threshold:  bool = False,
) -> np.ndarray:
    """
    Detect the circular sample boundary and return a binary sample mask.

    Parameters
    ----------
    img : np.ndarray
        2D uint8 preprocessed XCT slice.
        Dark background (air) + bright sample (metal).
    erosion_radius : int, optional
        Erode the detected boundary by this many pixels to avoid
        including partial-volume edge pixels in the valid region.
        Defaults to config.SAMPLE_MASK_EROSION_RADIUS.
    air_threshold : int, optional
        Pixels below this value are classified as air (background).
        Defaults to config.SAMPLE_MASK_AIR_THRESHOLD. Ignored if
        adaptive_threshold=True.
    return_circle : bool
        If True  → fit a filled circle to the detected region (recommended).
                   More robust against slice-to-slice variation.
        If False → return the raw largest-component mask (exact boundary).
    adaptive_threshold : bool
        If True, compute the air/solid split per-slice via Otsu instead
        of using a fixed air_threshold. Needed near the top/bottom of a
        scan where "background" may be structured support material
        rather than open air — its intensity can sit close to (or above)
        a fixed low threshold like config.SAMPLE_MASK_AIR_THRESHOLD=30,
        which then misclassifies the whole slice as "sample" (see
        scripts/run_masks_shape_aware.py). Defaults to False, matching
        prior behaviour.

    Returns
    -------
    np.ndarray
        uint8 mask, same shape as img:
            1 = inside sample
            0 = outside sample (air background)

    Raises
    ------
    ValueError
        If no sample region can be detected.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(img).__name__}.")
    if img.dtype != np.uint8:
        raise TypeError(f"Expected uint8 image, got {img.dtype}.")
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}.")

    if erosion_radius is None: erosion_radius = config.SAMPLE_MASK_EROSION_RADIUS

    if adaptive_threshold:
        air_threshold = int(threshold_otsu(img))
    elif air_threshold is None:
        air_threshold = config.SAMPLE_MASK_AIR_THRESHOLD

    h, w = img.shape

    # ------------------------------------------------------------------
    # Step 1 — Threshold to separate air/background from sample
    # ------------------------------------------------------------------
    binary = img > air_threshold   # True = sample, False = background

    # ------------------------------------------------------------------
    # Step 2 — Fill holes (pores inside sample should be inside mask)
    # ------------------------------------------------------------------
    filled = binary_fill_holes(binary)

    # ------------------------------------------------------------------
    # Step 3 — Find largest connected component = sample cylinder
    # ------------------------------------------------------------------
    labeled  = label(filled)
    regions  = regionprops(labeled)

    if not regions:
        raise ValueError(
            "detect_sample_mask: no regions found. "
            f"Check that air_threshold={air_threshold} is appropriate "
            "for your data (sample should be brighter than air)."
        )

    # Largest region by area = the sample
    sample_region = max(regions, key=lambda r: r.area)

    if return_circle:
        # ------------------------------------------------------------------
        # Step 4a — Fit a circle using centroid + equivalent radius
        # More robust than raw mask — handles incomplete slices at stack edges
        # ------------------------------------------------------------------
        cy, cx = sample_region.centroid
        radius = np.sqrt(sample_region.area / np.pi)

        yi, xi  = np.ogrid[:h, :w]
        circle  = ((xi - cx)**2 + (yi - cy)**2) <= radius**2
        sample_mask = circle.astype(np.uint8)

    else:
        # ------------------------------------------------------------------
        # Step 4b — Use raw component mask
        # ------------------------------------------------------------------
        sample_mask = (labeled == sample_region.label).astype(np.uint8)

    # ------------------------------------------------------------------
    # Step 5 — Erode boundary to exclude partial-volume edge pixels
    # Edge pixels mix air and metal intensity → unreliable for thresholding
    # ------------------------------------------------------------------
    if erosion_radius > 0:
        eroded      = binary_erosion(
            sample_mask.astype(bool),
            footprint=disk(erosion_radius)
        )
        sample_mask = eroded.astype(np.uint8)

    # Sanity check
    coverage = sample_mask.sum() / sample_mask.size
    if coverage < 0.05:
        warnings.warn(
            f"detect_sample_mask: sample mask covers only {coverage:.1%} "
            "of the image — mask may be incorrect. "
            f"Try lowering air_threshold (currently {air_threshold})."
        )
    elif coverage > 0.98:
        warnings.warn(
            f"detect_sample_mask: sample mask covers {coverage:.1%} "
            "of the image — background may not be detected. "
            f"Try raising air_threshold (currently {air_threshold})."
        )

    return sample_mask


# =============================================================================
# Stack-level mask (consistent circle across all slices)
# =============================================================================

def detect_sample_mask_stack(
    tiff_files:     list,
    n_sample_slices: int = 5,
    erosion_radius:  int = None,
    air_threshold:   int = None,
) -> tuple:
    """
    Compute a single consistent circular mask for an entire TIFF stack.

    Averages the detected circle parameters (centre + radius) across
    n_sample_slices evenly-spaced slices to produce one stable mask
    that can be applied to every slice in the stack.

    Parameters
    ----------
    tiff_files : list of Path
        Sorted list of TIFF file paths for one sample.
    n_sample_slices : int
        Number of slices to sample for circle detection.
    erosion_radius : int, optional
        Defaults to config.SAMPLE_MASK_EROSION_RADIUS.
    air_threshold : int, optional
        Defaults to config.SAMPLE_MASK_AIR_THRESHOLD.

    Returns
    -------
    tuple (cx, cy, radius, erosion_radius)
        Circle parameters — use build_circle_mask() to generate the mask.
    """
    import tifffile as tiff
    from src.preprocess import preprocess_slice

    if erosion_radius is None: erosion_radius = config.SAMPLE_MASK_EROSION_RADIUS
    if air_threshold  is None: air_threshold  = config.SAMPLE_MASK_AIR_THRESHOLD

    n       = len(tiff_files)
    indices = np.linspace(n // 4, 3 * n // 4, n_sample_slices, dtype=int)

    cxs, cys, radii = [], [], []

    for idx in indices:
        try:
            raw  = tiff.imread(tiff_files[idx]).astype(np.float32)
            if raw.ndim != 2:
                continue
            prep = preprocess_slice(raw)

            binary = prep > air_threshold
            filled = binary_fill_holes(binary)
            labeled = label(filled)
            regions = regionprops(labeled)

            if not regions:
                continue

            region = max(regions, key=lambda r: r.area)
            cy, cx = region.centroid
            radius = np.sqrt(region.area / np.pi)

            cxs.append(cx)
            cys.append(cy)
            radii.append(radius)

        except Exception as e:
            warnings.warn(f"Skipping slice {idx} for mask detection: {e}")
            continue

    if not cxs:
        raise ValueError(
            "detect_sample_mask_stack: could not detect sample circle "
            "in any sampled slice."
        )

    return (
        float(np.mean(cxs)),
        float(np.mean(cys)),
        float(np.mean(radii)),
        erosion_radius,
    )


def build_circle_mask(
    h:              int,
    w:              int,
    cx:             float,
    cy:             float,
    radius:         float,
    erosion_radius: int = 0,
) -> np.ndarray:
    """
    Build a filled circular uint8 mask given circle parameters.

    Parameters
    ----------
    h, w           : image height and width
    cx, cy         : circle centre (x, y)
    radius         : circle radius in pixels
    erosion_radius : shrink mask by this many pixels (edge exclusion)

    Returns
    -------
    np.ndarray
        uint8 mask — 1 = inside sample, 0 = outside.
    """
    yi, xi      = np.ogrid[:h, :w]
    circle      = ((xi - cx)**2 + (yi - cy)**2) <= radius**2
    sample_mask = circle.astype(np.uint8)

    if erosion_radius > 0:
        eroded      = binary_erosion(
            sample_mask.astype(bool),
            footprint=disk(erosion_radius)
        )
        sample_mask = eroded.astype(np.uint8)

    return sample_mask


# =============================================================================
# Robust global solid-intensity reference
# =============================================================================

def compute_robust_solid_level(
    tiff_files:     list,
    good_indices:   list,
    n_sample:       int = 15,
) -> float:
    """
    Estimate a single, stack-wide "what does solid metal actually look
    like" intensity reference, from slices already known to be reliable
    (see the correlation-based reliable-range scan in
    scripts/regenerate_bernsen_filtered.py) — NOT derived per-slice.

    Motivation: both the fixed air_threshold and the per-slice adaptive
    Otsu fallback are derived independently from each slice's own
    content, so an outlier slice (support structure, low real contrast)
    can corrupt its own threshold either way — too much background kept
    (fixed) or too little solid kept (adaptive, once porosity or content
    pulls the per-slice split point too high). A robust reference
    computed from many KNOWN-GOOD slices doesn't have this problem: one
    bad slice can't corrupt a statistic built from fifteen good ones.

    Parameters
    ----------
    tiff_files : list of Path
        Full sorted list of preprocessed slice paths for one sample.
    good_indices : list of int
        Indices (into tiff_files) already established as reliable —
        e.g. the [start, end] range from the correlation-based scan.
    n_sample : int
        How many evenly-spaced slices within good_indices to sample.

    Returns
    -------
    float
        Median solid-phase intensity across the sampled good slices.
    """
    good_indices = sorted(good_indices)
    lo, hi = good_indices[0], good_indices[-1]
    sample_idx = np.linspace(lo, hi, min(n_sample, hi - lo + 1), dtype=int)

    solid_vals = []
    for i in sample_idx:
        img = tiff.imread(tiff_files[i])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask = detect_sample_mask(img, return_circle=False)
        if mask.sum() > 0:
            solid_vals.append(img[mask.astype(bool)])

    if not solid_vals:
        raise ValueError(
            "compute_robust_solid_level: no solid pixels found in any "
            f"of the {len(sample_idx)} sampled 'good' slices."
        )

    return float(np.median(np.concatenate(solid_vals)))


def detect_sample_mask_robust(img: np.ndarray, robust_solid_level: float,
                               erosion_radius: int = None) -> np.ndarray:
    """
    Radial solid<->air boundary using a fixed, stack-wide reference
    instead of a per-slice-derived one. air_threshold is set to half
    the robust solid level (same midpoint logic as Bernsen's own
    T = (I_max + I_min) / 2, applied globally rather than per-window).

    Intended as a third fallback tier: fixed threshold -> per-slice
    adaptive Otsu -> this, for slices where both of the first two
    produce implausible coverage (see regenerate_bernsen_filtered.py).
    """
    air_threshold = robust_solid_level * 0.5
    return detect_sample_mask(img, return_circle=False,
                               air_threshold=air_threshold,
                               erosion_radius=erosion_radius)


# =============================================================================
# Sliding-window local-histogram "near-max peak" boundary detection
# =============================================================================

def _local_near_max_peak(window: np.ndarray, min_count: float = 5.0) -> float:
    """
    Histogram one local window; return the peak (mode) intensity closest
    to the maximum end -- scan down from bin 255 for the first real
    local maximum with enough pixel support to not just be noise. Falls
    back to the global argmax of the (smoothed) histogram if no such
    peak is found (e.g. a window with almost no variation at all).
    """
    hist, _ = np.histogram(window, bins=256, range=(0, 255))
    hist_s = gaussian_filter1d(hist.astype(np.float64), sigma=2)
    for v in range(255, 0, -1):
        if (hist_s[v] >= min_count and hist_s[v] >= hist_s[v - 1]
                and hist_s[v] >= hist_s[min(255, v + 1)]):
            return float(v)
    return float(np.argmax(hist_s))


def scan_near_max_peak_map(img: np.ndarray, window: int = 41, stride: int = 10,
                            min_count: float = 5.0) -> np.ndarray:
    """
    Slide a window across img in `stride`-pixel steps; at each position
    compute the local histogram and record the peak nearest the max
    intensity (see _local_near_max_peak). Produces a coarse spatial map
    -- one "locally dominant near-max intensity" value per window
    position -- rather than one global or per-slice number.

    Unlike a single global/per-slice threshold, this only requires
    there to be SOME local contrast between a bright and a dark
    population *somewhere* in the frame, which is why it held up on
    near-edge/transition slices where fixed and per-slice-adaptive
    thresholds failed (verified informally on sample_06 idx 60, 734,
    1420).
    """
    h, w = img.shape
    half = window // 2
    ys = list(range(half, max(half + 1, h - half), stride)) or [h // 2]
    xs = list(range(half, max(half + 1, w - half), stride)) or [w // 2]
    peak_map = np.zeros((len(ys), len(xs)), dtype=np.float32)
    for iy, y in enumerate(ys):
        y0, y1 = max(0, y - half), min(h, y + half)
        for ix, x in enumerate(xs):
            x0, x1 = max(0, x - half), min(w, x + half)
            peak_map[iy, ix] = _local_near_max_peak(img[y0:y1, x0:x1], min_count)
    return peak_map


def detect_sample_mask_peak_scan(img: np.ndarray, window: int = 41, stride: int = 10,
                                  erosion_radius: int = None) -> np.ndarray:
    """
    Segment solid material from background using the sliding-window
    near-max-peak scan (scan_near_max_peak_map), instead of one global
    or per-slice-derived intensity threshold.

    The coarse peak map is bimodal wherever a slice actually contains
    both material and background (material windows peak near the
    solid intensity, background windows peak near zero) -- Otsu-split
    it, upsample the split back to full resolution (nearest-neighbour,
    since this is a segmentation, not intensity data), then keep only
    the largest connected filled region, matching detect_sample_mask's
    convention.

    Returns an all-zero mask (no detection) if the peak map has no
    contrast to split at all -- e.g. a slice that is uniformly
    background.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(img).__name__}.")
    if img.dtype != np.uint8:
        raise TypeError(f"Expected uint8 image, got {img.dtype}.")
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}.")

    if erosion_radius is None:
        erosion_radius = config.SAMPLE_MASK_EROSION_RADIUS

    h, w = img.shape
    peak_map = scan_near_max_peak_map(img, window=window, stride=stride)

    pm_u8 = np.clip(peak_map, 0, 255).astype(np.uint8)
    if pm_u8.min() == pm_u8.max():
        return np.zeros((h, w), dtype=np.uint8)
    try:
        split = threshold_otsu(pm_u8)
    except Exception:
        return np.zeros((h, w), dtype=np.uint8)

    coarse_binary = peak_map > split
    full_binary = resize(coarse_binary.astype(np.float32), (h, w), order=0,
                          preserve_range=True, anti_aliasing=False) > 0.5

    filled = binary_fill_holes(full_binary)
    labeled = label(filled)
    regions = regionprops(labeled)
    if not regions:
        return np.zeros((h, w), dtype=np.uint8)

    largest = max(regions, key=lambda r: r.area)
    sample_mask = (labeled == largest.label).astype(np.uint8)

    if erosion_radius > 0:
        eroded = binary_erosion(sample_mask.astype(bool), footprint=disk(erosion_radius))
        sample_mask = eroded.astype(np.uint8)

    return sample_mask


# =============================================================================
# Canny "zone detection" — tries several (sigma, close_radius) configs per
# slice and picks whichever produces the most plausible boundary, instead
# of one fixed config for the whole volume (needed because the right
# config genuinely varies by cross-section size -- verified on sample_06:
# large hexagon slices need sigma=3/close=5, a small crescent needs
# sigma=6-8/close=2, a trapezoid needs sigma=5/close=3, etc.)
# =============================================================================

CANNY_ZONE_CONFIGS = [(3, 5), (3, 3), (3, 2), (4, 3), (5, 3), (6, 2), (6, 3), (8, 2)]

# Quantile-thresholded candidates: (sigma, low_q, high_q, close_radius).
# low/high_threshold as quantiles of THIS image's own gradient-magnitude
# distribution (use_quantiles=True) rather than sigma-driven smoothing
# alone. Added specifically because sigma-only configs completely failed
# (0% coverage on every one of the 8 CANNY_ZONE_CONFIGS) on sample_07's
# heavily-porous slices (idx 1059/1172/1286) -- the dense internal
# porosity apparently produces enough spurious gradient response that no
# fixed-sigma/absolute-threshold combination cleanly separated the true
# outer boundary from internal noise. Quantile thresholds at sigma=2
# recovered these cleanly and consistently (43-45% area across all 4
# quantile pairs tested, matching the ~41-42% reference) where sigma-only
# configs found nothing at all.
CANNY_QUANTILE_ZONE_CONFIGS = [
    (2, 0.7, 0.9, 3), (2, 0.6, 0.85, 3), (2, 0.5, 0.8, 3), (2, 0.8, 0.95, 3),
]

METAL_BRIGHT_FRAC_MIN = 0.80   # candidate region must be MOSTLY brighter than
                                # this slice's OWN whole-image Otsu split, to
                                # count as genuine solid metal rather than a
                                # background/noise/vignetting artifact that
                                # happens to pass the area+solidity checks.
                                #
                                # NOT a fixed absolute brightness (history: an
                                # earlier version used a hardcoded threshold
                                # of 150 intensity units -- calibrated only on
                                # sample_06, where solid peaks measured
                                # 190-250. Broke completely on sample_07 idx
                                # 1172, whose whole-image median is only 37 --
                                # every candidate's bright_frac read 0% against
                                # a fixed 150, even the visually-correct one,
                                # which only reached mean~102. Exactly the
                                # same "one constant assumed to fit every
                                # scan" trap already documented and fixed once
                                # this session for Bernsen's own
                                # low_contrast_thresh -- same fix applies:
                                # anchor to the image's own distribution.)
                                #
                                # threshold_otsu on the WHOLE image (not
                                # restricted to any one candidate) gives a
                                # natural background-vs-everything-brighter
                                # split specific to this slice, since
                                # sample_06/07's true background is genuinely
                                # near-zero (unlike sample_01-05's bright-halo
                                # background -- this function is not used for
                                # those samples). Calibrated separation still
                                # huge with this fix: true detections ~98-100%
                                # bright-fraction, false ones ~0-1%.


def detect_sample_mask_canny_zones(img: np.ndarray, configs=None,
                                    erosion_radius: int = None,
                                    prev_area_frac: float = None) -> np.ndarray:
    """
    Canny boundary detection with per-slice config selection ("zone
    detection"), a metal-brightness sanity check, and an optional
    depth-continuity prior.

    For each candidate (sigma, close_radius) in `configs`: run Canny,
    close the edges, fill holes, take the largest connected component.
    Keep only candidates whose area fraction is plausible (same bounds
    as has_genuine_material_boundary) AND whose interior is mostly
    bright (>= METAL_BRIGHT_FRAC_MIN of pixels above the image's own
    Otsu split) -- i.e. actually looks like solid metal, not just a
    conveniently-shaped region of background/noise.

    Selecting the survivor purely by highest solidity is NOT safe on
    its own: on sample_06 idx=1420, the only two survivors were a
    scaffold-following false positive (area=29.96%, solidity=0.55) and
    the actual thin-sliver material (area=1.84%, solidity=0.17) --
    solidity alone picks the WRONG one, confidently. A real material
    part's cross-sectional area changes gradually slice-to-slice
    (except at genuine transitions); the wrong candidate's area (29.96%)
    is wildly discontinuous from the tapering-to-near-zero trend every
    neighbouring slice shows, while the correct one (1.84%) fits it.

    prev_area_frac : float, optional
        The accepted area fraction of the previous slice in depth order
        (None if unavailable, e.g. the first slice of a range, or the
        caller isn't processing sequentially). When given, survivors are
        ranked by closeness to this value instead of raw solidity --
        among survivors within CONTINUITY_TOLERANCE of it, the closest
        wins; if none are within tolerance, falls back to the
        solidity-tie-margin rule below (handles genuine transitions,
        where a real, large area change is expected). When None, uses
        the solidity-tie-margin rule unconditionally.

    Returns an all-zero mask if no candidate survives both checks
    (caller should treat this as "undetectable", same convention as
    every other boundary method in this codebase).
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(img).__name__}.")
    if img.dtype != np.uint8:
        raise TypeError(f"Expected uint8 image, got {img.dtype}.")
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}.")

    if configs is None:
        configs = CANNY_ZONE_CONFIGS
    if erosion_radius is None:
        erosion_radius = config.SAMPLE_MASK_EROSION_RADIUS

    try:
        bright_ref = threshold_otsu(img)
    except Exception:
        bright_ref = 0   # degenerate (uniform image) -- can't split, let every
                          # candidate through on this check alone

    survivors = []   # (solidity, area_frac, mask)

    def _evaluate(edges, close_r):
        closed = binary_closing(edges, footprint=disk(close_r))
        filled = binary_fill_holes(closed)
        labeled = label(filled)
        regions = regionprops(labeled)
        if not regions:
            return
        largest = max(regions, key=lambda r: r.area)
        area_frac = largest.area / img.size
        if not (MIN_BOUNDARY_AREA_FRAC <= area_frac <= MAX_BOUNDARY_AREA_FRAC):
            return
        candidate_mask = (labeled == largest.label)
        vals = img[candidate_mask]
        bright_frac = float((vals > bright_ref).mean()) if vals.size else 0.0
        if bright_frac < METAL_BRIGHT_FRAC_MIN:
            return
        survivors.append((largest.solidity, area_frac, candidate_mask.astype(np.uint8)))

    for sigma, close_r in configs:
        edges = canny(img.astype(np.float64), sigma=sigma)
        _evaluate(edges, close_r)

    for sigma, low_q, high_q, close_r in CANNY_QUANTILE_ZONE_CONFIGS:
        edges = canny(img.astype(np.float64), sigma=sigma, low_threshold=low_q,
                       high_threshold=high_q, use_quantiles=True)
        _evaluate(edges, close_r)

    if not survivors:
        return np.zeros_like(img, dtype=np.uint8)

    # Picking the single highest-solidity survivor is NOT safe on its own:
    # a tiny, accidentally-very-compact fragment (e.g. a small solid speck
    # isolated by heavy smoothing) can score marginally higher solidity
    # than the correct, much larger whole-cross-section candidate --
    # confirmed on sample_06 idx 826 (a 0.27%-area fragment at
    # solidity=0.984 narrowly beat the correct 48.6%-area candidate at
    # solidity=0.976). Fix: only let a smaller candidate win over a larger
    # one when its solidity is MEANINGFULLY better (SOLIDITY_TIE_MARGIN),
    # not just ahead by noise -- among near-tied solidity, prefer more
    # area. Re-verified this does not regress the opposite case (sample_06
    # idx 147: the correct 12.6%-area candidate has solidity ~0.99 vs the
    # nearest larger-area alternative's ~0.89-0.90 -- a 0.09+ gap, well
    # outside the tie margin, so the correct small candidate still wins).
    SOLIDITY_TIE_MARGIN = 0.03

    # Depth-continuity prior, tried FIRST when available: prefer whichever
    # survivor's area is closest to the previous accepted slice's area,
    # among those within CONTINUITY_TOLERANCE (relative) of it. This is
    # what correctly rejects the sample_06 idx=1420 scaffold false
    # positive (29.96%, wildly discontinuous from the tapering trend)
    # in favour of the real 1.84% sliver. Only engages within tolerance
    # so a genuine transition (real, large area change between slices)
    # still falls through to the solidity rule instead of being forced
    # to match a now-stale previous value.
    CONTINUITY_TOLERANCE = 0.5   # 50% relative -- generous on purpose;
                                  # only meant to catch wildly-wrong
                                  # candidates like the scaffold case,
                                  # not to fine-tune among plausible ones
    chosen = None
    if prev_area_frac is not None and prev_area_frac > 0:
        within = [c for c in survivors
                  if abs(c[1] - prev_area_frac) <= CONTINUITY_TOLERANCE * prev_area_frac]
        if within:
            chosen = min(within, key=lambda c: abs(c[1] - prev_area_frac))

    if chosen is None:
        best_solidity = max(s for s, _, _ in survivors)
        tied = [c for c in survivors if c[0] >= best_solidity - SOLIDITY_TIE_MARGIN]
        chosen = max(tied, key=lambda c: c[1])

    _, _, sample_mask = chosen
    if erosion_radius > 0:
        eroded = binary_erosion(sample_mask.astype(bool), footprint=disk(erosion_radius))
        sample_mask = eroded.astype(np.uint8)

    return sample_mask


def detect_sample_mask_canny_zones_sequence(imgs, erosion_radius: int = None):
    """
    Run detect_sample_mask_canny_zones() over an ordered sequence of
    slices (e.g. one full depth range of a sample), carrying the
    depth-continuity reference forward from one slice to the next.

    Missing/undetectable slices (returned mask is empty) do NOT reset
    the continuity reference -- the last known-good accepted area is
    carried forward unchanged and offered to the next slice, so one or
    a few slices failing outright doesn't blind the continuity check
    for everything after them. Only a genuinely accepted (non-empty)
    result updates the reference.

    Parameters
    ----------
    imgs : iterable of np.ndarray
        2D uint8 slices, IN DEPTH ORDER. Must already be restricted to
        whatever range this method should run on (e.g. the
        correlation-verified reliable range) -- this function has no
        opinion on that, it only tracks continuity across what it's
        given.
    erosion_radius : int, optional
        Passed through to detect_sample_mask_canny_zones on every call.

    Returns
    -------
    list of np.ndarray
        One uint8 mask per input slice, same order (empty mask where
        undetectable, same convention as every other boundary method).
    """
    masks = []
    prev_area_frac = None
    for img in imgs:
        mask = detect_sample_mask_canny_zones(img, erosion_radius=erosion_radius,
                                               prev_area_frac=prev_area_frac)
        masks.append(mask)
        area_frac = float(mask.mean())
        if area_frac > 0:
            prev_area_frac = area_frac
        # else: leave prev_area_frac untouched -- skip this slice for
        # continuity-tracking purposes without losing the last-known-good
        # reference for the slices that follow.
    return masks


# =============================================================================
# Canny edge detection boundary (traces the TRUE cross-section shape,
# circular or not, unlike detect_sample_mask's idealised circle fit)
# =============================================================================

MIN_BOUNDARY_AREA_FRAC = 0.001   # same rationale as regenerate_bernsen_material_value.py:
MAX_BOUNDARY_AREA_FRAC = 0.90    # reject near-empty AND near-full-frame "boundaries" --
                                  # the latter is Canny's actual failure mode on hard
                                  # near-edge / garbage-range slices: with no real edge
                                  # to find, fill_holes + largest-component collapses to
                                  # ~99-100% coverage (verified: sample_06 idx 60/1420/
                                  # 0/10 all landed at 97.9-99.98% coverage).


def has_genuine_material_boundary(mask: np.ndarray) -> bool:
    """Shared validation logic -- see regenerate_bernsen_material_value.py's
    version of this function for the full rationale (contour existence +
    plausible area-fraction bounds, both directions)."""
    if mask.sum() == 0:
        return False
    contours = find_contours(mask.astype(float), 0.5)
    if not contours:
        return False
    labeled = label(mask)
    regions = regionprops(labeled)
    if not regions:
        return False
    largest = max(regions, key=lambda r: r.area)
    area_frac = largest.area / mask.size
    return MIN_BOUNDARY_AREA_FRAC <= area_frac <= MAX_BOUNDARY_AREA_FRAC


def detect_sample_mask_canny(img: np.ndarray, sigma: float = 3, close_radius: int = 5,
                              erosion_radius: int = None) -> np.ndarray:
    """
    Segment solid material from background via Canny edge detection,
    rather than an intensity threshold -- traces the TRUE cross-section
    shape (circular or not) since it responds to the actual edge
    gradient, not a brightness split. Verified clean (no rim, no notch)
    on both circular (sample_01-05) and non-circular/hexagonal
    (sample_06/07) cross-sections on good slices.

    Failure mode: on slices with no real material boundary at all (deep
    into a garbage/transition range, or certain thin near-edge slices),
    Canny finds no closed edge and the mask collapses to ~99-100% of
    the frame after fill_holes + largest-component. Returns an all-zero
    mask in that case (via has_genuine_material_boundary) rather than
    the degenerate near-full-frame result -- caller should treat an
    all-zero return the same as "undetectable", same convention as
    detect_sample_mask_peak_scan.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(img).__name__}.")
    if img.dtype != np.uint8:
        raise TypeError(f"Expected uint8 image, got {img.dtype}.")
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}.")

    if erosion_radius is None:
        erosion_radius = config.SAMPLE_MASK_EROSION_RADIUS

    edges = canny(img.astype(np.float64), sigma=sigma)
    closed = binary_closing(edges, footprint=disk(close_radius))
    filled = binary_fill_holes(closed)
    labeled = label(filled)
    regions = regionprops(labeled)
    if not regions:
        return np.zeros_like(img, dtype=np.uint8)

    largest = max(regions, key=lambda r: r.area)
    sample_mask = (labeled == largest.label).astype(np.uint8)

    if not has_genuine_material_boundary(sample_mask):
        return np.zeros_like(img, dtype=np.uint8)

    if erosion_radius > 0:
        eroded = binary_erosion(sample_mask.astype(bool), footprint=disk(erosion_radius))
        sample_mask = eroded.astype(np.uint8)

    return sample_mask