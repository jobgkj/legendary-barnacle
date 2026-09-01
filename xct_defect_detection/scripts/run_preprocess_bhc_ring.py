"""
=============================================================================
run_preprocess_bhc_ring.py — Full 4-stage production pipeline (with BHC +
ring artefact suppression added), written to a SEPARATE output tree
=============================================================================
Applies, per sample: BHC (volume-level pre-pass) -> ring suppression
(per-slice) -> median filter -> NLM denoising -> normalise to uint8,
via preprocess_slice() for the last three stages (unchanged, same
function used by the existing production pipeline) plus the two new
functions estimate_stack_bhc_correction() / apply_ring_suppression().

Output goes to data/processed_bhc_ring/<sample>/ and
data/masks_bhc_ring/<sample>/{otsu,yen,bernsen}/ — separate from the
existing data/processed/ and data/masks/ trees, so the original
(3-stage) baseline used throughout the rest of this thesis is left
untouched and both can be compared directly.

Run from repository root:
    python scripts/run_preprocess_bhc_ring.py [sample_name ...]
=============================================================================
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import time
import warnings

import numpy as np
import tifffile as tiff

import config
from config import create_dirs
from src.preprocess import (
    preprocess_slice,
    estimate_stack_norm_range,
    estimate_stack_bhc_correction,
    apply_ring_suppression,
)
from src.thresholding import otsu, yen, bernsen
from src.sample_mask import detect_sample_mask_stack, build_circle_mask

PROCESSED_OUT = config.REPO_ROOT / "data" / "processed_bhc_ring"
MASKS_OUT = config.REPO_ROOT / "data" / "masks_bhc_ring"


def process_sample(sample_name: str) -> dict:
    raw_dir = config.RAW_DATA_DIR / sample_name
    proc_dir = PROCESSED_OUT / sample_name
    proc_dir.mkdir(parents=True, exist_ok=True)

    tiff_files = sorted(list(raw_dir.glob("*.tif")) + list(raw_dir.glob("*.tiff")))
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {raw_dir}")

    total = len(tiff_files)
    t0 = time.time()

    print(f"  [{sample_name}] estimating stack-wide normalisation range...")
    try:
        stack_vmin, stack_vmax = estimate_stack_norm_range(tiff_files)
        print(f"  [{sample_name}] normalisation range: [{stack_vmin:.1f}, {stack_vmax:.1f}]")
    except Exception as e:
        warnings.warn(f"  normalisation range estimation failed: {e}")
        stack_vmin = stack_vmax = None

    print(f"  [{sample_name}] estimating BHC depth-wise correction (degree={config.BHC_POLY_DEGREE})...")
    bhc_correction = estimate_stack_bhc_correction(tiff_files)
    print(f"  [{sample_name}] BHC correction range: [{bhc_correction.min():.1f}, {bhc_correction.max():.1f}] "
          f"(raw intensity units)")

    saved = skipped = 0
    for idx, f in enumerate(tiff_files):
        out_path = proc_dir / f.name
        if out_path.exists():
            saved += 1
        else:
            try:
                raw = tiff.imread(f).astype(np.float32)
            except Exception as e:
                warnings.warn(f"Skipping {f.name} — could not read: {e}")
                skipped += 1
                continue
            if raw.ndim != 2:
                skipped += 1
                continue

            # Stage 1 (new, production): BHC — subtract this slice's
            # depth-position correction from the raw intensity.
            raw = raw - bhc_correction[idx]

            # Stage 2 (new, production): ring artefact suppression.
            raw = apply_ring_suppression(raw)

            # Stages 3-5 (existing, unchanged): median filter, NLM,
            # normalise to uint8 — same preprocess_slice() used by the
            # original 3-stage production pipeline.
            try:
                prep = preprocess_slice(raw, vmin=stack_vmin, vmax=stack_vmax)
            except Exception as e:
                warnings.warn(f"Skipping {f.name} — preprocessing failed: {e}")
                skipped += 1
                continue

            tiff.imwrite(out_path, prep)
            saved += 1

        elapsed = time.time() - t0
        done = idx + 1
        eta = (elapsed / done) * (total - done) if done else 0
        print(f"  [{sample_name}] {done}/{total}  elapsed {elapsed:6.1f}s  ETA {eta:6.1f}s", end="\r")
    print()

    return {"sample": sample_name, "total": total, "saved": saved, "skipped": skipped,
            "proc_dir": proc_dir, "tiff_files": tiff_files}


def generate_masks(sample_name: str, tiff_files_raw: list):
    """Run Otsu/Yen/Bernsen on the newly-preprocessed (BHC+ring) volume,
    same procedure as the existing production classical pipeline, and
    save to data/masks_bhc_ring/<sample>/<method>/."""
    proc_dir = PROCESSED_OUT / sample_name
    proc_files = sorted(list(proc_dir.glob("*.tif")) + list(proc_dir.glob("*.tiff")))
    if not proc_files:
        print(f"  [{sample_name}] no preprocessed files found, skipping masks")
        return

    print(f"  [{sample_name}] detecting sample boundary mask...")
    first = tiff.imread(proc_files[0])
    h, w = first.shape
    cx, cy, radius, erosion_radius = detect_sample_mask_stack(tiff_files_raw)
    sample_mask = build_circle_mask(h, w, cx, cy, radius, erosion_radius)
    print(f"  [{sample_name}] sample mask: centre=({cx:.1f},{cy:.1f}) radius={radius:.1f}")

    for method_name, fn in [("otsu", otsu), ("yen", yen), ("bernsen", bernsen)]:
        out_dir = MASKS_OUT / sample_name / method_name
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for i, f in enumerate(proc_files):
            out_path = out_dir / f.name
            if out_path.exists():
                continue
            img = tiff.imread(f)
            mask = fn(img, sample_mask=sample_mask)
            tiff.imwrite(out_path, (mask.astype(np.uint8)) * 255)
            if (i + 1) % 100 == 0 or (i + 1) == len(proc_files):
                print(f"    [{sample_name}:{method_name}] {i+1}/{len(proc_files)} "
                      f"({time.time()-t0:.1f}s)", end="\r")
        print()


def main():
    create_dirs()
    PROCESSED_OUT.mkdir(parents=True, exist_ok=True)
    MASKS_OUT.mkdir(parents=True, exist_ok=True)

    requested = sys.argv[1:]
    if requested:
        sample_names = requested
    else:
        sample_names = sorted(d.name for d in config.RAW_DATA_DIR.iterdir() if d.is_dir())

    print(f"Samples: {sample_names}")
    print(f"Processed output: {PROCESSED_OUT}")
    print(f"Masks output: {MASKS_OUT}\n")

    results = []
    for name in sample_names:
        print(f"=== {name} (preprocess: BHC + ring + median + NLM) ===")
        try:
            r = process_sample(name)
            results.append(r)
        except Exception as e:
            print(f"  [ERROR] {name} preprocessing: {e}")
            continue

        print(f"=== {name} (classical thresholding on new preprocessed data) ===")
        try:
            generate_masks(name, r["tiff_files"])
        except Exception as e:
            print(f"  [ERROR] {name} mask generation: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    for r in results:
        print(f"  {r['sample']}: {r['saved']}/{r['total']} saved, {r['skipped']} skipped")
    print("=" * 60)


if __name__ == "__main__":
    main()
