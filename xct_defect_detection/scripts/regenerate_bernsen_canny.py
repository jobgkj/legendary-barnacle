"""
=============================================================================
regenerate_bernsen_canny.py — Bernsen mask regeneration for sample_01-05,
FULL VOLUME, using Canny edge detection + morphological closing as the
boundary method (detect_sample_mask_canny) -- traces the TRUE
cross-section shape (verified clean, no rim/notch, on both circular
sample_01-05 and hexagonal sample_06/07 test slices), unlike the
circle-fit method's idealised-circle assumption.

Guarded by has_genuine_material_boundary (min/max area-fraction bounds)
against Canny's own known failure mode: on slices with no real
boundary at all, it collapses to ~99-100% frame coverage after
fill_holes + largest-component -- rejected and treated as
"undetectable", same convention as every other method this session.

Run from repository root:
    python scripts/regenerate_bernsen_canny.py sample_01 sample_02 sample_03 sample_04 sample_05
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
from src.sample_mask import detect_sample_mask_canny
from scripts.regenerate_bernsen_filtered import globtiff

RELIABLE_RANGES = {
    "sample_01": (1, 898),
    "sample_02": (1, 898),
    "sample_03": (1, 898),
    "sample_04": (1, 747),
    "sample_05": (1, 898),
}

DEFECT_FRAC_PERCENTILE_CUTOFF = 80


def main():
    sample_names = sys.argv[1:] or list(RELIABLE_RANGES.keys())

    for sample_name in sample_names:
        if sample_name not in RELIABLE_RANGES:
            print(f"[skip] {sample_name}: no reliable range known, not run by this script")
            continue

        print(f"\n=== {sample_name} (Canny boundary, full volume) ===")
        proc_dir = config.REPO_ROOT / "data" / "processed_bhc_ring" / sample_name
        proc_files = globtiff(proc_dir)
        n = len(proc_files)

        start, end = RELIABLE_RANGES[sample_name]
        print(f"  Reliable range: [{start}, {end}] of {n} slices")
        print(f"  Boundary method: Canny edge detection + closing "
              f"(sigma=3, close_radius=5, erosion_radius=default)")

        # NOTE: writes to a SEPARATE tree from masks_bhc_ring -- the
        # circle-fit results already there are validated and clean;
        # this keeps both around for comparison rather than overwriting.
        out_dir = config.REPO_ROOT / "data" / "masks_bhc_ring_canny" / sample_name / "bernsen"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Pass 1/2: computing Bernsen result for every slice in range ...")
        results = {}
        for i in range(start, end + 1):
            img = tiff.imread(proc_files[i])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mask = detect_sample_mask_canny(img)
            if mask.sum() == 0:
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
