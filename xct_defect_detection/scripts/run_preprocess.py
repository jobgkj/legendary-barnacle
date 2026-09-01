"""
=============================================================================
run_preprocess.py — Stream raw XCT slices through preprocessing to disk
=============================================================================
Reads data/raw/<sample_name>/*.tif, applies preprocess_slice() (median +
NLM denoising, normalised to uint8), and saves the result to
data/processed/<sample_name>/ with the same filenames.

O(1) memory — one slice is held in RAM at a time.

This is Step 2 of the pipeline (see README) and must be run before
src/io.py::load_and_generate_masks() / scripts/run_all_samples.py, which
read from data/processed/ by default.

Run from repository root:
    python scripts/run_preprocess.py
=============================================================================
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# See pipeline.py for why this is needed: non-ASCII prints (checkmarks, banners)
# otherwise crash under a non-UTF-8 console/redirect encoding (e.g. Windows cp1252).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import time
import warnings

import tifffile as tiff

import config
from config import create_dirs
from src.preprocess import preprocess_slice, estimate_stack_norm_range


def process_sample(sample_name: str) -> dict:
    raw_dir = config.RAW_DATA_DIR / sample_name
    out_dir = config.PROCESSED_DATA_DIR / sample_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tiff_files = sorted(
        list(raw_dir.glob("*.tif")) + list(raw_dir.glob("*.tiff"))
    )
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {raw_dir}")

    total = len(tiff_files)
    saved = 0
    skipped = 0
    t0 = time.time()

    # Estimate one stack-wide normalisation range up front and reuse it
    # for every slice, instead of each slice normalising against its own
    # intensity range independently — see estimate_stack_norm_range's
    # docstring for why that matters (it otherwise manufactures fake
    # full-strength contrast out of genuinely low-signal slices, e.g.
    # near the top/bottom of a scan).
    try:
        stack_vmin, stack_vmax = estimate_stack_norm_range(tiff_files)
        print(f"  Stack-wide normalisation range: [{stack_vmin:.1f}, {stack_vmax:.1f}]")
    except Exception as e:
        warnings.warn(
            f"  Stack-wide range estimation failed: {e} — "
            "falling back to per-slice normalisation."
        )
        stack_vmin = stack_vmax = None

    for idx, f in enumerate(tiff_files):
        out_path = out_dir / f.name

        if out_path.exists():
            saved += 1
        else:
            try:
                raw = tiff.imread(f)
            except Exception as e:
                warnings.warn(f"Skipping {f.name} — could not read: {e}")
                skipped += 1
                continue

            if raw.ndim != 2:
                warnings.warn(f"Skipping {f.name} — not 2D (shape={raw.shape})")
                skipped += 1
                continue

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
        print(f"  [{sample_name}] {done}/{total}  "
              f"elapsed {elapsed:6.1f}s  ETA {eta:6.1f}s", end="\r")

    print()
    return {"sample": sample_name, "total": total, "saved": saved, "skipped": skipped}


def main():
    create_dirs()

    sample_dirs = sorted(
        d for d in config.RAW_DATA_DIR.iterdir() if d.is_dir()
    )
    if not sample_dirs:
        print(f"No sample directories found in {config.RAW_DATA_DIR}")
        return

    print(f"Found {len(sample_dirs)} sample(s): {[d.name for d in sample_dirs]}")
    print(f"Output: {config.PROCESSED_DATA_DIR}\n")

    results = []
    for d in sample_dirs:
        print(f"=== {d.name} ===")
        try:
            results.append(process_sample(d.name))
        except Exception as e:
            print(f"  [ERROR] {d.name}: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    for r in results:
        print(f"  {r['sample']}: {r['saved']}/{r['total']} saved, "
              f"{r['skipped']} skipped")
    print("=" * 60)


if __name__ == "__main__":
    main()
