"""
=============================================================================
regenerate_bernsen_circlefit.py — Bernsen mask regeneration for
sample_01-04, FULL VOLUME, using the classical circle-fit boundary
method (detect_sample_mask, return_circle=True, adaptive_threshold=True)
-- unmodified, as opposed to the sliding-window peak-scan method used
for sample_06/07.

Why: the peak-scan method assumes material is bright and background is
dark and cleanly separable. samples_01-05 (NIST reference samples) have
a bright halo/glow surrounding the part -- background strips read
mean~200 std~47, not near-zero like sample_06's mean~0.4 std~5.3. That
breaks the peak-scan's Otsu split (collapses to split=0, "material" =
everything except the 4 pure-black image corners, producing a false
notch wherever the halo meets a corner). The classical circle-fit
method (fit an idealised circle from the largest connected region's
centroid + equivalent radius, adaptive per-slice Otsu threshold) does
NOT have this failure mode -- verified visually clean on sample_01-04
at idx 100/200 (smooth boundary at the true edge, no rim, no notch).

sample_05 is EXCLUDED here: circle-fit also produces a badly wrong
result on it (6.3% coverage vs ~73% for 01-04, circle far too small) --
needs separate, dedicated investigation, not run by this script.

Run from repository root:
    python scripts/regenerate_bernsen_circlefit.py sample_01 sample_02 sample_03 sample_04
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

import config
from src.thresholding import bernsen
from src.sample_mask import detect_sample_mask
from scripts.regenerate_bernsen_filtered import globtiff

RELIABLE_RANGES = {
    "sample_01": (1, 898),
    "sample_02": (1, 898),
    "sample_03": (1, 898),
    "sample_04": (1, 747),
    "sample_05": (1, 898),
}

# sample_05 is ~72% porosity (published) -- the solid phase is a minority,
# fragmented into disconnected islands, so adaptive per-slice Otsu (which
# assumes ONE dominant bright "solid" population) places its split inside
# the material's own bimodal (solid vs. pore) distribution instead of at
# the true background boundary -- confirmed: adaptive gave 6.3% coverage
# on sample_05 vs ~73% for samples 01-04, an obviously wrong small circle.
# A FIXED low threshold (any value 10-40 gave IDENTICAL 73.5% coverage --
# there's a clean gap between true-zero background and everything
# belonging to the part) correctly captures the whole part regardless of
# internal porosity. samples 01-04 keep adaptive Otsu (already verified
# clean); only sample_05 uses this fixed-threshold path.
FIXED_THRESHOLD_SAMPLES = {"sample_05": 20}

DEFECT_FRAC_PERCENTILE_CUTOFF = 80


def main():
    sample_names = sys.argv[1:] or list(RELIABLE_RANGES.keys())

    for sample_name in sample_names:
        if sample_name not in RELIABLE_RANGES:
            print(f"[skip] {sample_name}: no reliable range known, not run by this script")
            continue

        print(f"\n=== {sample_name} (circle-fit boundary, full volume) ===")
        proc_dir = config.REPO_ROOT / "data" / "processed_bhc_ring" / sample_name
        proc_files = globtiff(proc_dir)
        n = len(proc_files)

        start, end = RELIABLE_RANGES[sample_name]
        fixed_thresh = FIXED_THRESHOLD_SAMPLES.get(sample_name)
        print(f"  Reliable range: [{start}, {end}] of {n} slices")
        if fixed_thresh is not None:
            print(f"  Boundary method: classical circle-fit "
                  f"(detect_sample_mask, return_circle=True, fixed air_threshold={fixed_thresh})")
        else:
            print(f"  Boundary method: classical circle-fit "
                  f"(detect_sample_mask, return_circle=True, adaptive_threshold=True)")

        out_dir = config.REPO_ROOT / "data" / "masks_bhc_ring" / sample_name / "bernsen"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Pass 1/2: computing Bernsen result for every slice in range ...")
        results = {}
        for i in range(start, end + 1):
            img = tiff.imread(proc_files[i])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    if fixed_thresh is not None:
                        mask = detect_sample_mask(img, return_circle=True,
                                                   adaptive_threshold=False, air_threshold=fixed_thresh)
                    else:
                        mask = detect_sample_mask(img, return_circle=True, adaptive_threshold=True)
                except ValueError:
                    mask = None
            if mask is None or mask.sum() == 0:
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
