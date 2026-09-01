"""
=============================================================================
I/O utilities for streaming 2D XCT slice processing
and binary mask generation.

Directory layout
----------------
Input  (raw):       data/raw/<sample_name>/<slice>.tif
Input  (processed): data/processed/<sample_name>/<slice>.tif
Output (masks):     data/masks/<sample_name>/<method>/<slice>.tif

All output filenames match the input filename exactly.

Design goals
------------
- 2D slice-wise only
- Constant memory usage (no full volume loading)
- GitHub-safe (no raw data committed)
- Config-driven defaults, all overridable per call
- GUI-friendly: get_default_params() for field pre-population
- Sample mask support: removes air background from thresholding
=============================================================================
"""

from pathlib import Path
import warnings

import numpy as np
import tifffile as tiff

from src.thresholding import otsu, yen, bernsen
from src.sample_mask import detect_sample_mask_stack, build_circle_mask
import config


# =============================================================================
# GUI helper
# =============================================================================

def get_default_params() -> dict:
    """
    Return current default parameters from config.

    GUI can call this to pre-populate its input fields,
    then pass updated values back to load_and_generate_masks().

    Returns
    -------
    dict with keys:
        repo_root            : Path
        sample_name          : str
        use_processed        : bool
        bernsen_radius       : int
        bernsen_dct          : int
        use_sample_mask      : bool
    """
    return {
        "repo_root":        config.REPO_ROOT,
        "sample_name":      config.SAMPLE_NAME,
        "use_processed":    True,
        "bernsen_radius":   config.BERNSEN_RADIUS,
        "bernsen_dct":      config.BERNSEN_DCT,
        "use_sample_mask":  config.USE_SAMPLE_MASK,
    }


# =============================================================================
# Internal helpers
# =============================================================================

def _normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Normalize a 2D image to uint8 [0, 255].

    Returns a zero image with a warning if the intensity range is zero.
    """
    img  = img.astype(np.float32)
    vmin, vmax = img.min(), img.max()

    if vmax <= vmin:
        warnings.warn("Slice has zero intensity range — returning blank slice.")
        return np.zeros(img.shape, dtype=np.uint8)

    img = (img - vmin) / (vmax - vmin)
    return (255.0 * img).clip(0, 255).astype(np.uint8)


# =============================================================================
# Main entry point
# =============================================================================

def load_and_generate_masks(
    repo_root:         "Path | str"  = None,
    sample_name:       str           = None,
    use_processed:     bool          = True,
    bernsen_radius:    int           = None,
    bernsen_dct:       int           = None,
    use_sample_mask:   bool          = None,
    progress_callback: callable      = None,
) -> dict:
    """
    Stream 2D XCT slices from disk, generate segmentation
    masks, and save them to disk.

    All parameters fall back to config.py values if not supplied.

    Sample mask
    -----------
    When use_sample_mask=True (default), a circular sample boundary
    is detected automatically and applied to all three thresholding
    methods. This prevents air background pixels from being classified
    as pores, fixing over-segmentation in low-porosity samples.

    Parameters
    ----------
    repo_root : Path or str, optional
        Root directory of the git repository.
        Defaults to config.REPO_ROOT.
    sample_name : str, optional
        Name of the sample folder under data/raw/ or data/processed/.
        Defaults to config.SAMPLE_NAME.
    use_processed : bool
        If True  → read from data/processed/<sample_name>/  (default)
        If False → read from data/raw/<sample_name>/
    bernsen_radius : int, optional
        Radius for Bernsen local thresholding.
        Defaults to config.BERNSEN_RADIUS.
    bernsen_dct : int, optional
        Fixed DCT override. If None, auto-computed per slice.
        Defaults to None (auto-compute).
    use_sample_mask : bool, optional
        Apply circular sample mask to exclude air background.
        Defaults to config.USE_SAMPLE_MASK.
    progress_callback : callable, optional
        Called each slice with (current, total, filename).
        Use to update a GUI progress bar.

    Returns
    -------
    dict with keys:
        total     : int  — total slices found
        processed : int  — slices successfully processed
        skipped   : int  — slices skipped due to errors
        mask_dir  : Path — root output directory for this sample's masks
    """

    # ------------------------------------------------------------------
    # Fall back to config for any unset parameters
    # ------------------------------------------------------------------
    repo_root        = Path(repo_root)      if repo_root        is not None else config.REPO_ROOT
    sample_name      = sample_name          if sample_name       is not None else config.SAMPLE_NAME
    bernsen_radius   = bernsen_radius       if bernsen_radius    is not None else config.BERNSEN_RADIUS
    use_sample_mask  = use_sample_mask      if use_sample_mask   is not None else config.USE_SAMPLE_MASK
    # bernsen_dct left as None → auto-computed per slice in bernsen()

    # ------------------------------------------------------------------
    # Input directory
    # ------------------------------------------------------------------
    if use_processed:
        input_dir = repo_root / "data" / "processed" / sample_name
    else:
        input_dir = repo_root / "data" / "raw" / sample_name

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # ------------------------------------------------------------------
    # Output directories — data/masks/<sample_name>/<method>/
    # ------------------------------------------------------------------
    mask_root = repo_root / "data" / "masks" / sample_name

    out_dirs = {
        "otsu":    mask_root / "otsu",
        "yen":     mask_root / "yen",
        "bernsen": mask_root / "bernsen",
    }

    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Collect TIFF files (sorted for correct Z-order)
    # ------------------------------------------------------------------
    tiff_files = sorted(
        list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    )

    if not tiff_files:
        raise ValueError(f"No TIFF files found in {input_dir}")

    total   = len(tiff_files)
    skipped = 0
    empty   = 0

    print(f"  Found {total} slices in {input_dir}")

    # ------------------------------------------------------------------
    # Detect sample mask (once per stack — consistent circle)
    # ------------------------------------------------------------------
    stack_cx = stack_cy = stack_radius = None

    if use_sample_mask:
        print("  Detecting sample boundary (circular mask) ...")
        try:
            stack_cx, stack_cy, stack_radius, _ = detect_sample_mask_stack(
                tiff_files,
                n_sample_slices=5,
            )
            print(
                f"  Sample circle: centre=({stack_cx:.1f}, {stack_cy:.1f}), "
                f"radius={stack_radius:.1f}px"
            )
        except Exception as e:
            warnings.warn(
                f"  Sample mask detection failed: {e} — "
                "continuing without sample mask."
            )
            use_sample_mask = False

    # ------------------------------------------------------------------
    # Stream slices one-by-one
    # ------------------------------------------------------------------
    for idx, f in enumerate(tiff_files):

        print(f"  Processing {idx + 1}/{total}: {f.name}", end="\r")

        if progress_callback is not None:
            progress_callback(idx + 1, total, f.name)

        # Load
        try:
            img = tiff.imread(f)
        except Exception as e:
            warnings.warn(f"Skipping {f.name} — could not read: {e}")
            skipped += 1
            continue

        if img.ndim != 2:
            warnings.warn(
                f"Skipping {f.name} — not a 2D slice (shape={img.shape})"
            )
            skipped += 1
            continue

        if img.dtype != np.uint8:
            img = _normalize_to_uint8(img)

        # Build sample mask for this slice
        sample_mask = None
        if use_sample_mask and stack_cx is not None:
            h, w = img.shape
            sample_mask = build_circle_mask(
                h, w,
                cx=stack_cx,
                cy=stack_cy,
                radius=stack_radius,
                erosion_radius=config.SAMPLE_MASK_EROSION_RADIUS,
            )

        # Empty-slice check: the stack-level circle is reused for every
        # slice, including ones near the top/bottom of the scan where no
        # physical object is actually present yet (or any more) — just
        # air/noise inside that circle. Thresholding those anyway
        # misclassifies the whole circular region as one large "pore",
        # which shows up as flat, over-segmented discs at the top/bottom
        # of a 3D reconstruction. If too little of the (would-be) sample
        # region is actually bright/solid material, skip thresholding
        # and write empty (all-zero, no-defect) masks instead.
        is_empty_slice = False
        if sample_mask is not None:
            region = img[sample_mask.astype(bool)]
            if region.size > 0:
                object_fraction = float((region > config.SAMPLE_MASK_AIR_THRESHOLD).mean())
                is_empty_slice = object_fraction < config.EMPTY_SLICE_MIN_OBJECT_FRACTION

        if is_empty_slice:
            empty += 1
            zeros = np.zeros_like(img, dtype=np.uint8)
            masks = {"otsu": zeros, "yen": zeros, "bernsen": zeros.copy()}
        else:
            # Generate masks — pass sample_mask to exclude air background
            masks = {
                "otsu":    otsu(img,     sample_mask=sample_mask),
                "yen":     yen(img,      sample_mask=sample_mask),
                "bernsen": bernsen(
                    img,
                    radius=bernsen_radius,
                    DCT=bernsen_dct,          # None = auto-compute per slice
                    sample_mask=sample_mask,
                ),
            }

        # Save — preserve original filename exactly
        for method, mask in masks.items():
            tiff.imwrite(
                out_dirs[method] / f.name,
                mask.astype(np.uint8),
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    processed = total - skipped

    print(f"\n  Done — {processed}/{total} slices processed.")
    if skipped:
        print(f"  Skipped: {skipped} slice(s) — check warnings above.")
    if empty:
        print(f"  Empty (no object present): {empty} slice(s) — written as "
              f"all-zero masks instead of thresholded.")
    print(f"  Masks saved to: {mask_root}")

    return {
        "total":     total,
        "processed": processed,
        "skipped":   skipped,
        "empty":     empty,
        "mask_dir":  mask_root,
    }