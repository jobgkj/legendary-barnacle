"""
=============================================================================
regenerate_bernsen_filtered.py — Bernsen-only mask regeneration for
sample_06 / sample_07, excluding unreliable slices at the top/bottom of
each scan.

"Unreliable" is detected via neighbour-to-neighbour correlation on the
RAW stack: real XCT slices through a solid part look nearly identical to
their immediate depth-neighbours, so a genuine transition/support-
structure slice (not part of the main body) shows up as a sharp drop in
that correlation. This is a tighter, more precise signal than either a
depth-wise mean-intensity trend (§ earlier over-exposure test, found
nothing) or a fixed percentage-of-depth cutoff.

Boundaries are found by scanning inward from each end of the stack until
correlation exceeds CORR_THRESHOLD for CONSECUTIVE_STABLE consecutive
slices in a row (avoids stopping on a single noisy fluctuation).

Excluded slices get an empty (all-zero) Bernsen mask, matching the
existing "no detectable sample" convention used elsewhere in this
pipeline -- not skipped/left stale, explicitly written as "no defect
data here."

Run from repository root:
    python scripts/regenerate_bernsen_filtered.py sample_06 sample_07
=============================================================================
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import tifffile as tiff

import warnings

import config
from src.thresholding import bernsen
from src.sample_mask import (
    detect_sample_mask,
    compute_robust_solid_level,
    detect_sample_mask_robust,
)

CORR_THRESHOLD     = 0.85
CONSECUTIVE_STABLE = 5


def detect_hybrid_3tier(img, robust_solid_level):
    """
    Tier 1: fixed air_threshold=30 (correct for the great majority of
    slices).
    Tier 2: per-slice adaptive Otsu, if tier 1 implausibly finds ~no
    background (>95% coverage) -- the known transition-slice failure.
    Tier 3: robust stack-wide solid-level reference, if tier 2 ALSO
    gives implausible coverage (<5% or >95%) -- the known failure of
    adaptive Otsu on slices whose local content skews its own split
    point (e.g. very high real porosity, or transition-zone content).
    Tier 3's own coverage is validated too -- it is still a single
    fixed threshold, so on a slice whose background floor sits even
    above that reference it fails the exact same way tier 1 did
    (verified empirically: some slices hit cov_robust=1.000, and
    Bernsen on an effectively unmasked frame then reads 99%+ as
    "defect"). If tier 3 ALSO produces implausible coverage, the slice
    is treated as genuinely undetectable and given an empty mask,
    rather than trusting a result known to be wrong.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = detect_sample_mask(img, return_circle=False)
    coverage = float(mask.mean())
    if coverage <= 0.95:
        return mask, "fixed"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = detect_sample_mask(img, return_circle=False, adaptive_threshold=True)
    coverage = float(mask.mean())
    if 0.05 <= coverage <= 0.95:
        return mask, "adaptive"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = detect_sample_mask_robust(img, robust_solid_level)
    coverage = float(mask.mean())
    if 0.05 <= coverage <= 0.95:
        return mask, "robust"

    # All three tiers failed to find a plausible boundary on this
    # slice -- genuinely undetectable, not "trust the last attempt".
    return np.zeros_like(img, dtype=np.uint8), "undetectable"


def globtiff(d: Path):
    return sorted(list(d.glob("*.tif")) + list(d.glob("*.tiff")))


def find_reliable_range(raw_files):
    """Scan inward from both ends; return (start, end) inclusive indices
    of the reliable region (everything outside gets an empty mask)."""
    n = len(raw_files)
    imgs_cache = {}

    def get(i):
        if i not in imgs_cache:
            imgs_cache[i] = tiff.imread(raw_files[i]).astype(np.float32).ravel()
        return imgs_cache[i]

    def corr(i, j):
        a, b = get(i), get(j)
        return float(np.corrcoef(a, b)[0, 1])

    # scan forward from the start
    start = 0
    stable = 0
    for i in range(1, n):
        c = corr(i - 1, i)
        if c >= CORR_THRESHOLD:
            stable += 1
            if stable >= CONSECUTIVE_STABLE:
                start = i - CONSECUTIVE_STABLE + 1
                break
        else:
            stable = 0
    else:
        start = 0

    imgs_cache.clear()

    # scan backward from the end
    end = n - 1
    stable = 0
    for i in range(n - 2, -1, -1):
        c = corr(i, i + 1)
        if c >= CORR_THRESHOLD:
            stable += 1
            if stable >= CONSECUTIVE_STABLE:
                end = i + CONSECUTIVE_STABLE - 1
                break
        else:
            stable = 0
    else:
        end = n - 1

    return start, end


def main():
    sample_names = sys.argv[1:] or ["sample_06", "sample_07"]

    for sample_name in sample_names:
        print(f"\n=== {sample_name} ===")
        raw_dir = config.RAW_DATA_DIR / sample_name
        raw_files = globtiff(raw_dir)
        proc_dir = config.REPO_ROOT / "data" / "processed_bhc_ring" / sample_name
        proc_files = globtiff(proc_dir)
        n = len(proc_files)

        print(f"  Scanning for reliable slice range (neighbour correlation >= {CORR_THRESHOLD}) ...")
        start, end = find_reliable_range(raw_files)
        n_excluded = start + (n - 1 - end)
        print(f"  Reliable range: [{start}, {end}] of {n} slices "
              f"({100*start/n:.1f}% - {100*(n-1-end)/n:.1f}% excluded from each end, "
              f"{n_excluded} slices total excluded)")

        print(f"  Computing robust stack-wide solid-intensity reference from the reliable range ...")
        robust_solid_level = compute_robust_solid_level(proc_files, [start, end])
        print(f"  Robust solid level: {robust_solid_level:.1f}  "
              f"(air_threshold fallback = {robust_solid_level * 0.5:.1f})")

        out_dir = config.REPO_ROOT / "data" / "masks_bhc_ring" / sample_name / "bernsen"
        out_dir.mkdir(parents=True, exist_ok=True)

        n_written = n_empty = 0
        tier_counts = {"fixed": 0, "adaptive": 0, "robust": 0, "undetectable": 0}
        for i, f in enumerate(proc_files):
            img = tiff.imread(f)
            if i < start or i > end:
                tiff.imwrite(out_dir / f.name, np.zeros_like(img, dtype=np.uint8))
                n_empty += 1
            else:
                shape_mask, tier = detect_hybrid_3tier(img, robust_solid_level)
                tier_counts[tier] += 1
                mask = bernsen(img, sample_mask=shape_mask)
                tiff.imwrite(out_dir / f.name, (mask.astype(np.uint8)) * 255)
                n_written += 1
            if (i + 1) % 100 == 0 or (i + 1) == n:
                print(f"    {i+1}/{n}", end="\r")
        print()
        print(f"  Done: {n_written} Bernsen masks written, {n_empty} excluded slices set to empty.")
        print(f"  Mask-detection tiers used: {tier_counts}")


if __name__ == "__main__":
    main()
