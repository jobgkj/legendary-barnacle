"""
=============================================================================
regenerate_masks_shape_aware_full.py — PRODUCTION full-volume mask
regeneration using the shape-agnostic hybrid mask (see
src/sample_mask.py::detect_sample_mask, run_masks_shape_aware.py).

Unlike run_masks_shape_aware.py (5 representative slices, separate
experimental output directories), this script:
  - processes every slice in every sample (full volumes)
  - OVERWRITES the existing production mask directories in place, at
    their existing filenames, instead of writing to a new tree:
        data/processed/           -> data/masks/<sample>/<method>/
        data/processed_bhc_ring/  -> data/masks_bhc_ring/<sample>/<method>/

This intentionally replaces the previous circular-mask-based masks in
data/masks/ and data/masks_bhc_ring/ with the new shape-aware ones —
no separate copy is kept, to avoid growing disk usage further. Any
thesis numbers or CSVs already exported from the old masks are
snapshots and are unaffected by this; only what gets recomputed from
disk AFTER this run reflects the new masks.

Mask logic (matches run_masks_shape_aware.py): fixed air_threshold=30
first; falls back to a per-slice adaptive Otsu threshold only when the
fixed threshold implausibly finds ~no background (>95% coverage) --
the known failure mode on sample_06/07's transition slices near the
top/bottom of the scan.

Run from repository root:
    python scripts/regenerate_masks_shape_aware_full.py [sample_name ...]
=============================================================================
"""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import tifffile as tiff

import config
from src.thresholding import otsu, yen, bernsen
from src.sample_mask import detect_sample_mask

PIPELINES = {
    "baseline": {
        "proc_dir": config.REPO_ROOT / "data" / "processed",
        "out_dir":  config.REPO_ROOT / "data" / "masks",           # existing production path
    },
    "bhc_ring": {
        "proc_dir": config.REPO_ROOT / "data" / "processed_bhc_ring",
        "out_dir":  config.REPO_ROOT / "data" / "masks_bhc_ring",  # existing production path
    },
}


def globtiff(d: Path):
    return sorted(list(d.glob("*.tif")) + list(d.glob("*.tiff")))


def detect_hybrid(img):
    """Fixed threshold first; adaptive Otsu fallback only if the fixed
    threshold implausibly finds ~no background (coverage > 0.95)."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        mask = detect_sample_mask(img, return_circle=False)
    if float(mask.mean()) > 0.95:
        mask = detect_sample_mask(img, return_circle=False, adaptive_threshold=True)
    return mask


def generate_masks(pipeline_name, sample_name, proc_root, out_root):
    proc_dir = proc_root / sample_name
    proc_files = globtiff(proc_dir)
    if not proc_files:
        print(f"  [{pipeline_name}:{sample_name}] no preprocessed files found, skipping")
        return 0, 0

    total = len(proc_files)
    written = failed = 0

    methods = [("otsu", otsu), ("yen", yen), ("bernsen", bernsen)]
    out_dirs = {}
    for method_name, _ in methods:
        out_dirs[method_name] = out_root / sample_name / method_name
        out_dirs[method_name].mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i, f in enumerate(proc_files):
        img = tiff.imread(f)
        # Shape mask computed ONCE per slice (not once per method) --
        # it's identical input to all three thresholding methods.
        try:
            shape_mask = detect_hybrid(img)
        except ValueError as e:
            warnings.warn(f"  [{pipeline_name}:{sample_name}] {f.name}: "
                           f"shape-mask detection failed ({e}), writing empty masks")
            zeros = np.zeros_like(img, dtype=np.uint8)
            for method_name, _ in methods:
                tiff.imwrite(out_dirs[method_name] / f.name, zeros)
            failed += 1
            continue

        for method_name, fn in methods:
            mask = fn(img, sample_mask=shape_mask)
            tiff.imwrite(out_dirs[method_name] / f.name, (mask.astype(np.uint8)) * 255)
        written += 1

        if (i + 1) % 100 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            eta = (elapsed / (i + 1)) * (total - (i + 1))
            print(f"    [{pipeline_name}:{sample_name}] {i+1}/{total} "
                  f"elapsed {elapsed:6.1f}s  ETA {eta:6.1f}s", end="\r")
    print()

    return written, failed


def main():
    requested = sys.argv[1:]
    sample_names = requested if requested else sorted(
        d.name for d in config.RAW_DATA_DIR.iterdir() if d.is_dir()
    )

    # Optional: XCT_MASK_PIPELINES="bhc_ring" (or "baseline", or
    # "baseline,bhc_ring") to skip re-running a pipeline that's already
    # complete for these samples. Defaults to both.
    import os
    requested_pipelines = os.environ.get("XCT_MASK_PIPELINES")
    pipelines = (
        {k: v for k, v in PIPELINES.items() if k in requested_pipelines.split(",")}
        if requested_pipelines else PIPELINES
    )

    print(f"Samples: {sample_names}")
    for pname, paths in pipelines.items():
        print(f"  {pname}: {paths['proc_dir']} -> {paths['out_dir']}  (OVERWRITE IN PLACE)")
    print()

    results = {}
    for pname, paths in pipelines.items():
        print(f"\n=== Pipeline: {pname} ===")
        for name in sample_names:
            print(f"--- {pname}:{name} ---")
            try:
                w, f = generate_masks(pname, name, paths["proc_dir"], paths["out_dir"])
                results[(pname, name)] = (w, f)
            except Exception as e:
                print(f"  [ERROR] {pname}:{name}: {e}")
                results[(pname, name)] = (0, -1)

    print("\n" + "=" * 60)
    print("SUMMARY (full-volume shape-aware mask regeneration)")
    for (pname, name), (w, f) in results.items():
        print(f"  {pname}:{name}: {w} written, {f} used-empty-fallback")
    print("=" * 60)


if __name__ == "__main__":
    main()
