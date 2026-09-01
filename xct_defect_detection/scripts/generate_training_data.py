"""
=============================================================================
generate_training_data.py — Non-interactive Step 2 + Step 3/4 driver
=============================================================================
For every sample in data/raw/:
    1. Streams raw slices through preprocess_slice() -> data/processed/
    2. Generates Otsu / Yen / Bernsen masks (with circular sample-mask
       gating) via src.io.load_and_generate_masks() -> data/masks/

Safe to re-run: Step 1 skips slices already present in data/processed/.

Run from repository root:
    python scripts/generate_training_data.py
=============================================================================
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# See pipeline.py for why this is needed: non-ASCII prints (checkmarks, banners)
# otherwise crash under a non-UTF-8 console/redirect encoding (e.g. Windows cp1252) —
# including inside this script's own error handler, which then masks the real error.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import time
import warnings

import config
from config import create_dirs
from scripts.run_preprocess import process_sample as preprocess_sample
from src.io import load_and_generate_masks


def main():
    create_dirs()

    sample_dirs = sorted(
        d for d in config.RAW_DATA_DIR.iterdir() if d.is_dir()
    )
    if not sample_dirs:
        print(f"No sample directories found in {config.RAW_DATA_DIR}")
        return

    names = [d.name for d in sample_dirs]
    print(f"Found {len(names)} sample(s): {names}\n")

    t_start = time.time()

    for name in names:
        print("=" * 65)
        print(f"SAMPLE: {name}")
        print("=" * 65)

        print(f"\n[Step 2] Preprocessing -> data/processed/{name}/")
        t0 = time.time()
        try:
            result = preprocess_sample(name)
            print(f"  {result['saved']}/{result['total']} saved, "
                  f"{result['skipped']} skipped  [{time.time()-t0:.1f}s]")
        except Exception as e:
            print(f"  [ERROR] Preprocessing failed for {name}: {e}")
            continue

        print(f"\n[Step 3/4] Generating masks -> data/masks/{name}/")
        t0 = time.time()
        try:
            result = load_and_generate_masks(
                repo_root=config.REPO_ROOT,
                sample_name=name,
                use_processed=True,
            )
            print(f"  {result['processed']}/{result['total']} processed, "
                  f"{result['skipped']} skipped  [{time.time()-t0:.1f}s]")
        except Exception as e:
            print(f"  [ERROR] Mask generation failed for {name}: {e}")
            continue

        print()

    print("=" * 65)
    print(f"ALL SAMPLES COMPLETE — {time.time()-t_start:.1f}s total")
    print("=" * 65)


if __name__ == "__main__":
    main()
