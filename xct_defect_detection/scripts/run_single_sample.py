"""
=============================================================================
run_single_sample.py — Preprocess + generate masks for one named sample
=============================================================================
Same two steps as generate_training_data.py (preprocess -> data/processed/,
then Otsu/Yen/Bernsen masks -> data/masks/), scoped to a single sample
instead of looping over every discovered one.

Run from repository root:
    python scripts/run_single_sample.py sample_07
=============================================================================
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

import config
from config import create_dirs
from scripts.run_preprocess import process_sample as preprocess_sample
from src.io import load_and_generate_masks


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_single_sample.py <sample_name>")
        sys.exit(1)

    name = sys.argv[1]
    create_dirs()

    print(f"[Step 2] Preprocessing -> data/processed/{name}/")
    t0 = time.time()
    result = preprocess_sample(name)
    print(f"  {result['saved']}/{result['total']} saved, "
          f"{result['skipped']} skipped  [{time.time()-t0:.1f}s]")

    print(f"\n[Step 3/4] Generating masks -> data/masks/{name}/")
    t0 = time.time()
    result = load_and_generate_masks(
        repo_root=config.REPO_ROOT,
        sample_name=name,
        use_processed=True,
    )
    print(f"  {result['processed']}/{result['total']} processed, "
          f"{result['skipped']} skipped  [{time.time()-t0:.1f}s]")

    print(f"\n[DONE] {name} ready.")


if __name__ == "__main__":
    main()
