"""
=============================================================================
run_preprocess.py — XCT Preprocessing Runner
=============================================================================
Run this from your repo root:
    cd "C:\Users\Job George\Desktop\Degree Project\Code\podfam_research_project_XCT_Anlaysis"
    python run_preprocess.py
=============================================================================
"""

import os
import sys
import time
import numpy as np
import tifffile as tiff

# ── Paths ──────────────────────────────────────────────────────────────────
# Input: folder containing your raw .tif slices
INPUT_FOLDER = r"data\tiff_stack"

# Output: folder where preprocessed TIFFs will be saved
OUTPUT_FOLDER = r"data\tiff_output"

# ── Make sure project root is on the path ─────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Import your data loader ────────────────────────────────────────────────
# Adjust the import path to match where data_loader.py lives in your repo.
# Common layouts:
#   Option A — data_loader.py is directly in repo root
#       from data_loader import load_tiff_stack, full_preprocess
#   Option B — data_loader.py is inside a subfolder e.g. 'src' or 'preprocessing'
#       from src.data_loader import load_tiff_stack, full_preprocess
#
# Change the line below to match your actual folder structure:
from data_loader import load_tiff_stack, full_preprocess   # ← adjust if needed


def save_volume(volume: np.ndarray, output_folder: str) -> None:
    """
    Save a preprocessed volume as individual TIFF slices.

    Each slice is saved as a float32 TIFF named slice_0000.tif, slice_0001.tif, ...
    If the volume is 2D (single slice), it is saved as slice_0000.tif.

    Parameters
    ----------
    volume : np.ndarray
        Preprocessed float32 volume (2D or 3D), values in [0, 1].
    output_folder : str
        Directory to write output slices into (created if it does not exist).
    """
    os.makedirs(output_folder, exist_ok=True)

    if volume.ndim == 2:
        out_path = os.path.join(output_folder, "slice_0000.tif")
        tiff.imwrite(out_path, volume.astype(np.float32))
        print(f"  [Save] Saved single slice → '{out_path}'")
        return

    n = volume.shape[0]
    for i in range(n):
        out_path = os.path.join(output_folder, f"slice_{i:04d}.tif")
        tiff.imwrite(out_path, volume[i].astype(np.float32))
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  [Save] Saved {i+1}/{n} slices ...", end="\r")

    print()  # newline after progress
    print(f"  [Save] All {n} slices saved to '{output_folder}'")


def main() -> None:
    print("=" * 60)
    print("  XCT Preprocessing Pipeline")
    print("=" * 60)

    # ── Resolve absolute paths ─────────────────────────────────────────────
    input_path  = os.path.join(ROOT, INPUT_FOLDER)
    output_path = os.path.join(ROOT, OUTPUT_FOLDER)

    print(f"\n  Input  : {input_path}")
    print(f"  Output : {output_path}\n")

    # ── Validate input folder ──────────────────────────────────────────────
    if not os.path.isdir(input_path):
        print(f"[ERROR] Input folder not found: '{input_path}'")
        print("  → Create it and place your .tif slices inside, then re-run.")
        sys.exit(1)

    # ── Load ───────────────────────────────────────────────────────────────
    print("── Step 1 / 2 : Loading TIFF stack ──")
    t0 = time.time()
    volume_raw = load_tiff_stack(input_path)
    print(f"  Loaded in {time.time()-t0:.1f}s\n")

    # ── Preprocess ────────────────────────────────────────────────────────
    print("── Step 2 / 2 : Preprocessing ──")
    volume_processed = full_preprocess(volume_raw)

    # ── Save output ───────────────────────────────────────────────────────
    print("── Saving output ──")
    save_volume(volume_processed, output_path)

    print("\n" + "=" * 60)
    print("  Preprocessing complete!")
    print(f"  Output saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
