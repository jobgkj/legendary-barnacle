"""
=============================================================================
regenerate_bernsen_material_value.py — Bernsen-only mask regeneration
for sample_06 / sample_07, FULL VOLUME, using the sliding-window
local-histogram "near-max peak" boundary (detect_sample_mask_peak_scan)
as the SOLE mask-detection method for every slice in the
correlation-verified reliable range.

This replaced the earlier stack-wide robust-solid-level approach
(detect_sample_mask_robust): instead of one global/per-slice intensity
threshold, a 41x41 window is slid across the slice, a local histogram
is taken at each position, and the peak nearest the max intensity end
is kept -- Otsu-split of the resulting coarse map separates material
from background. Verified (sample_06) to produce tight, accurate
boundaries on slices that broke every earlier method, including the
two known hard near-edge cases -- but it is sensitive to ANY local
contrast, so it WILL trace a false boundary around subtle non-material
gradients (e.g. cone-beam vignetting) on truly garbage slices; this is
safe here only because such slices are already excluded upstream by
the correlation-verified RELIABLE_RANGES gate below, not by this
function itself. See src/sample_mask.py::detect_sample_mask_peak_scan.

Slices outside the reliable range (garbage/transition, per the
correlation scan) get an empty mask, same convention as
regenerate_bernsen_filtered.py.

Run from repository root:
    python scripts/regenerate_bernsen_material_value.py sample_06 sample_07
=============================================================================
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import warnings
import numpy as np
import tifffile as tiff
from skimage.measure import find_contours, label, regionprops

import config
from src.thresholding import bernsen
from src.sample_mask import detect_sample_mask_peak_scan
from scripts.regenerate_bernsen_filtered import globtiff

RELIABLE_RANGES = {
    "sample_01": (1, 898),
    "sample_02": (1, 898),
    "sample_03": (1, 898),
    "sample_04": (1, 747),
    "sample_05": (1, 898),
    "sample_06": (55, 1467),
    "sample_07": (69, 1471),
}

EROSION_RADIUS_BY_SAMPLE = {
    "sample_06": 35,   # tuned on sample_06 itself -- see rationale below.
    "sample_07": 35,   # same PODFAM industrial geometry as sample_06;
                        # not independently re-validated on sample_07's
                        # own slices, carried over on that assumption.
}
DEFAULT_EROSION_RADIUS = 0   # samples 01-05: NIST reference samples, a
                              # different geometry never tested with this
                              # boundary method -- run with NO erosion per
                              # explicit instruction, rather than assume
                              # the sample_06-tuned value transfers. The
                              # false-positive "defect rim" found at low
                              # erosion on sample_06 (2.92% -> 0.03% defect
                              # fraction between erosion 8 and 35) may
                              # reappear here -- results should be visually
                              # checked for a rim pattern before being
                              # treated as clean.

def erosion_radius_for(sample_name):
    return EROSION_RADIUS_BY_SAMPLE.get(sample_name, DEFAULT_EROSION_RADIUS)

DEFECT_FRAC_PERCENTILE_CUTOFF = 80   # data-driven cutoff instead of a fixed
                                      # absolute value: exclude whichever
                                      # slices fall at/above this sample's
                                      # OWN 90th percentile of defect
                                      # fraction. Requires two passes --
                                      # the cutoff can't be known until
                                      # every slice's defect fraction has
                                      # been computed once.

MIN_BOUNDARY_AREA_FRAC = 0.001  # largest region must cover at least 0.1% of the
                                 # frame -- lowered from the old 2% floor because
                                 # the peak-scan boundary is a genuine tight fit,
                                 # not a coverage estimate: confirmed real material
                                 # footprints as small as 0.3-0.8% occur at the very
                                 # ends of the reliable range (sample_06 idx 55-64).
MAX_BOUNDARY_AREA_FRAC = 0.90   # ...and no more than 90% -- a real material
                                 # cross-section has SOME background margin;
                                 # near-full-frame coverage means no boundary
                                 # was actually found (verified: a naive
                                 # "at least one contour exists" check passes
                                 # trivially even at 99%+ coverage, since the
                                 # frame edge itself still traces a contour)


def has_genuine_material_boundary(mask: np.ndarray) -> bool:
    """
    True only if a real, closed boundary around a plausible material
    region was found -- with actual background margin around it -- not
    just "some contour exists somewhere" or "coverage falls in a vague
    numeric range". Both a scattered/tiny detection AND a near-full-frame
    non-detection must be rejected.

    Checks:
      1. At least one closed contour exists at all.
      2. The largest connected component covers a plausible fraction of
         the frame: enough to be a real region (not noise), but not so
         much that no real background boundary could have been found.
    """
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


def main():
    sample_names = sys.argv[1:] or ["sample_06", "sample_07"]

    for sample_name in sample_names:
        print(f"\n=== {sample_name} (material-value boundary, full volume) ===")
        proc_dir = config.REPO_ROOT / "data" / "processed_bhc_ring" / sample_name
        proc_files = globtiff(proc_dir)
        n = len(proc_files)

        start, end = RELIABLE_RANGES.get(sample_name, (0, n - 1))
        erosion_radius = erosion_radius_for(sample_name)
        print(f"  Reliable range: [{start}, {end}] of {n} slices")
        print(f"  Boundary method: sliding-window near-max-peak scan "
              f"(window=41, stride=10, erosion_radius={erosion_radius})")

        out_dir = config.REPO_ROOT / "data" / "masks_bhc_ring" / sample_name / "bernsen"
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- Pass 1: compute every in-range slice's Bernsen result once,
        # keep it in memory, so the percentile cutoff can be computed from
        # real data before anything is written.
        print(f"  Pass 1/2: computing Bernsen result for every slice in range ...")
        results = {}   # i -> (result_mask, defect_frac) or None (no boundary)
        for i in range(start, end + 1):
            img = tiff.imread(proc_files[i])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mask = detect_sample_mask_peak_scan(img, erosion_radius=erosion_radius)
            if not has_genuine_material_boundary(mask):
                results[i] = None
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = bernsen(img, sample_mask=mask)
                results[i] = (result, float(result.mean()))
            if (i - start + 1) % 100 == 0 or i == end:
                print(f"    {i - start + 1}/{end - start + 1}", end="\r")
        print()

        fracs = np.array([r[1] for r in results.values() if r is not None])
        cutoff = float(np.percentile(fracs, DEFECT_FRAC_PERCENTILE_CUTOFF)) if len(fracs) else 1.0
        print(f"  p{DEFECT_FRAC_PERCENTILE_CUTOFF} defect-fraction cutoff (data-driven): "
              f"{cutoff*100:.4f}%  (from {len(fracs)} slices with a real boundary)")

        # ---- Pass 2: write, applying the now-known cutoff.
        print(f"  Pass 2/2: writing masks ...")
        n_written = n_empty = n_undetected = n_implausible = 0
        for i, f in enumerate(proc_files):
            if i < start or i > end:
                img = tiff.imread(f)
                tiff.imwrite(out_dir / f.name, np.zeros_like(img, dtype=np.uint8))
                n_empty += 1
            else:
                entry = results[i]
                if entry is None:
                    img = tiff.imread(f)
                    tiff.imwrite(out_dir / f.name, np.zeros_like(img, dtype=np.uint8))
                    n_undetected += 1
                else:
                    result, defect_frac = entry
                    if defect_frac >= cutoff:
                        tiff.imwrite(out_dir / f.name, np.zeros_like(result, dtype=np.uint8))
                        n_implausible += 1
                    else:
                        tiff.imwrite(out_dir / f.name, (result.astype(np.uint8)) * 255)
                        n_written += 1
            if (i + 1) % 200 == 0 or (i + 1) == n:
                print(f"    {i+1}/{n}", end="\r")
        print()
        print(f"  Done: {n_written} written, {n_empty} excluded (garbage range), "
              f"{n_undetected} undetectable (no boundary), "
              f"{n_implausible} excluded (defect fraction >= p{DEFECT_FRAC_PERCENTILE_CUTOFF} = {cutoff*100:.4f}%)")


if __name__ == "__main__":
    main()
