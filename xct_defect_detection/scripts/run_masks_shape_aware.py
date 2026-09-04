"""
=============================================================================
run_masks_shape_aware.py — regenerate Otsu/Yen/Bernsen masks using a
SHAPE-AGNOSTIC sample-boundary mask (largest connected bright
component, holes filled, NO circle fit) instead of either the
circular mask or no mask at all.

Unlike detect_sample_mask_stack() (one circle fit for the whole
stack), this calls detect_sample_mask(img, return_circle=False)
independently per slice — so it follows each slice's actual outline,
whatever the cross-section shape (circular, hexagonal, irregular),
and tracks any change in that outline through the depth of the scan.

Output, kept separate from every other mask tree:
    data/processed/           -> data/masks_shape_aware_baseline/<sample>/<method>/
    data/processed_bhc_ring/  -> data/masks_shape_aware_bhc_ring/<sample>/<method>/

Run from repository root:
    python scripts/run_masks_shape_aware.py [sample_name ...]
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
        "out_dir":  config.REPO_ROOT / "data" / "masks_shape_aware_hybrid_baseline",
    },
    "bhc_ring": {
        "proc_dir": config.REPO_ROOT / "data" / "processed_bhc_ring",
        "out_dir":  config.REPO_ROOT / "data" / "masks_shape_aware_hybrid_bhc_ring",
    },
}


def globtiff(d: Path):
    return sorted(list(d.glob("*.tif")) + list(d.glob("*.tiff")))


def pick_representative(files, n_slices):
    n = len(files)
    if n_slices >= n:
        return files
    fracs = [(k + 0.5) / n_slices for k in range(n_slices)]
    idxs = sorted(set(min(n - 1, int(n * fr)) for fr in fracs))
    return [files[i] for i in idxs]


def generate_masks(pipeline_name, sample_name, proc_root, out_root, n_slices=5):
    proc_dir = proc_root / sample_name
    all_files = globtiff(proc_dir)
    if not all_files:
        print(f"  [{pipeline_name}:{sample_name}] no preprocessed files found, skipping")
        return
    proc_files = pick_representative(all_files, n_slices)

    for method_name, fn in [("otsu", otsu), ("yen", yen), ("bernsen", bernsen)]:
        out_dir = out_root / sample_name / method_name
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for i, f in enumerate(proc_files):
            out_path = out_dir / f.name
            if out_path.exists():
                continue
            img = tiff.imread(f)
            # Fixed air_threshold=30 first (correct for normal-porosity
            # slices; a per-slice adaptive Otsu threshold gets pulled up
            # by heavy internal porosity -- e.g. sample_05 at 72%
            # published porosity -- and starts excluding real solid
            # material). Only fall back to adaptive when the fixed
            # threshold implausibly finds ~no background at all -- the
            # actual failure mode seen on sample_06/07's transition
            # slices near the top/bottom of the scan, where "background"
            # is structured support material rather than open air.
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    shape_mask = detect_sample_mask(img, return_circle=False)
                coverage = float(shape_mask.mean())
                if coverage > 0.95:
                    shape_mask = detect_sample_mask(img, return_circle=False, adaptive_threshold=True)
            except ValueError as e:
                warnings.warn(f"  [{pipeline_name}:{sample_name}] {f.name}: "
                               f"shape-mask detection failed ({e}), skipping slice")
                continue
            mask = fn(img, sample_mask=shape_mask)
            tiff.imwrite(out_path, (mask.astype(np.uint8)) * 255)
            if (i + 1) % 5 == 0 or (i + 1) == len(proc_files):
                print(f"    [{pipeline_name}:{sample_name}:{method_name}] {i+1}/{len(proc_files)} "
                      f"({time.time()-t0:.1f}s)", end="\r")
        print()


def main():
    requested = sys.argv[1:]
    sample_names = requested if requested else sorted(
        d.name for d in config.RAW_DATA_DIR.iterdir() if d.is_dir()
    )

    for pname, paths in PIPELINES.items():
        paths["out_dir"].mkdir(parents=True, exist_ok=True)
        print(f"\n=== Pipeline: {pname}  (source: {paths['proc_dir']}) ===")
        for name in sample_names:
            print(f"--- {pname}:{name} ---")
            try:
                generate_masks(pname, name, paths["proc_dir"], paths["out_dir"], n_slices=5)
            except Exception as e:
                print(f"  [ERROR] {pname}:{name}: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY (shape-aware mask generation)")
    for pname, paths in PIPELINES.items():
        for name in sample_names:
            n = 0
            for m in ("otsu", "yen", "bernsen"):
                d = paths["out_dir"] / name / m
                n += len(globtiff(d)) if d.exists() else 0
            print(f"  {pname}:{name}: {n} mask files")
    print("=" * 60)


if __name__ == "__main__":
    main()
