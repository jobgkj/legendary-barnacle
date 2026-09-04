"""
=============================================================================
run_masks_no_circle.py — regenerate Otsu/Yen/Bernsen masks with the
circular sample-boundary mask DISABLED (sample_mask=None explicitly,
independent of config.USE_SAMPLE_MASK), for both existing preprocessed
trees, each written to its own new, separate output directory:

    data/processed/           -> data/masks_no_circle_baseline/<sample>/<method>/
    data/processed_bhc_ring/  -> data/masks_no_circle_bhc_ring/<sample>/<method>/

Pure mask (re)generation only — no preprocessing is re-run, since both
source trees already exist on disk. Does not touch data/masks/ or
data/masks_bhc_ring/ (the existing, circle-masked versions).

Run from repository root:
    python scripts/run_masks_no_circle.py [sample_name ...]
=============================================================================
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import tifffile as tiff

import config
from src.thresholding import otsu, yen, bernsen

PIPELINES = {
    "baseline": {
        "proc_dir": config.REPO_ROOT / "data" / "processed",
        "out_dir":  config.REPO_ROOT / "data" / "masks_no_circle_baseline",
    },
    "bhc_ring": {
        "proc_dir": config.REPO_ROOT / "data" / "processed_bhc_ring",
        "out_dir":  config.REPO_ROOT / "data" / "masks_no_circle_bhc_ring",
    },
}


def globtiff(d: Path):
    return sorted(list(d.glob("*.tif")) + list(d.glob("*.tiff")))


def generate_masks(pipeline_name: str, sample_name: str, proc_root: Path, out_root: Path,
                    n_slices: int = 5):
    proc_dir = proc_root / sample_name
    all_files = globtiff(proc_dir)
    if not all_files:
        print(f"  [{pipeline_name}:{sample_name}] no preprocessed files found, skipping")
        return

    # Representative subset only — evenly spaced through depth, not the
    # full volume (this is a quick look, not a full regeneration).
    n = len(all_files)
    if n_slices >= n:
        proc_files = all_files
    else:
        fracs = [ (k + 0.5) / n_slices for k in range(n_slices) ]
        idxs = sorted(set(min(n - 1, int(n * fr)) for fr in fracs))
        proc_files = [all_files[i] for i in idxs]

    for method_name, fn in [("otsu", otsu), ("yen", yen), ("bernsen", bernsen)]:
        out_dir = out_root / sample_name / method_name
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for i, f in enumerate(proc_files):
            out_path = out_dir / f.name
            if out_path.exists():
                continue
            img = tiff.imread(f)
            mask = fn(img, sample_mask=None)   # explicitly no circular mask
            tiff.imwrite(out_path, (mask.astype(np.uint8)) * 255)
            if (i + 1) % 200 == 0 or (i + 1) == len(proc_files):
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
    print("SUMMARY (no-circle mask generation)")
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
