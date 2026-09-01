"""
=============================================================================
run_pipeline.py — End-to-end XCT processing pipeline (2D, slice-wise)
=============================================================================

This script:
1. Asks which sample(s) to process
2. Preprocesses each slice (median + NLM)
3. Segments defects using Otsu, Yen, and Bernsen thresholding
4. Computes slice-wise metrics
5. Saves masks to data/masks/<sample_name>/<method>/
6. Saves metrics to results/metrics/<sample_name>_metrics.csv

Run from repository root:
    python scripts/run_pipeline.py
=============================================================================
"""
import sys
from pathlib import Path

# Add project root to path so config.py can be found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import warnings
import csv
from pathlib import Path

import tifffile as tiff

import config
from config import create_dirs
from src.preprocess import preprocess_slice
from src.thresholding import bernsen, otsu, yen
from src.metrics import summarize_slice


# -----------------------------------------------------------------------------
# Sample selection (reused from run_all_samples.py)
# -----------------------------------------------------------------------------

def select_samples() -> list[str]:
    """
    Discover available samples and prompt the user to select which to process.

    Returns
    -------
    list[str]
        List of selected sample folder names.
    """
    sample_dirs = sorted(
        d for d in config.RAW_DATA_DIR.iterdir() if d.is_dir()
    )

    if not sample_dirs:
        print(f"No sample directories found in {config.RAW_DATA_DIR}")
        return []

    print("\nAvailable samples:")
    print("-" * 40)
    for idx, d in enumerate(sample_dirs, start=1):
        print(f"  [{idx}] {d.name}")
    print(f"  [0] Run ALL samples")
    print("-" * 40)

    while True:
        raw = input(
            "Enter sample numbers to process (e.g. 1 3 4), or 0 for all: "
        ).strip()

        if not raw:
            print("  No input given — please enter at least one number.")
            continue

        try:
            choices = [int(x) for x in raw.split()]
        except ValueError:
            print("  Invalid input — enter numbers only, separated by spaces.")
            continue

        if 0 in choices:
            return [d.name for d in sample_dirs]

        invalid = [c for c in choices if c < 1 or c > len(sample_dirs)]
        if invalid:
            print(
                f"  Invalid selection(s): {invalid} — "
                f"choose between 1 and {len(sample_dirs)}."
            )
            continue

        seen     = set()
        selected = []
        for c in choices:
            if c not in seen:
                seen.add(c)
                selected.append(sample_dirs[c - 1].name)

        return selected


# -----------------------------------------------------------------------------
# Single sample pipeline
# -----------------------------------------------------------------------------

def process_sample(sample_name: str) -> list[dict]:
    """
    Run the full preprocessing + thresholding + metrics pipeline
    for a single sample.

    Parameters
    ----------
    sample_name : str
        Name of the sample folder under data/raw/.

    Returns
    -------
    list[dict]
        All slice-method metric rows for this sample.
    """

    # ------------------------------------------------------------------
    # Input directory — read from processed if available, else raw
    # ------------------------------------------------------------------
    processed_dir = config.REPO_ROOT / "data" / "processed" / sample_name
    raw_dir       = config.RAW_DATA_DIR / sample_name

    if processed_dir.exists():
        input_dir    = processed_dir
        run_preprocess = False
        print(f"  Using preprocessed data from {processed_dir}")
    elif raw_dir.exists():
        input_dir    = raw_dir
        run_preprocess = True
        print(f"  Preprocessed data not found — running preprocessing on raw slices.")
    else:
        raise FileNotFoundError(
            f"Neither processed nor raw directory found for '{sample_name}'."
        )

    # ------------------------------------------------------------------
    # Output directories — data/masks/<sample_name>/<method>/
    # ------------------------------------------------------------------
    mask_root = config.REPO_ROOT / "data" / "masks" / sample_name

    out_dirs = {
        "otsu":    mask_root / "otsu",
        "yen":     mask_root / "yen",
        "bernsen": mask_root / "bernsen",
    }

    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Collect slices
    # ------------------------------------------------------------------
    tiff_files = sorted(
        list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    )

    if not tiff_files:
        raise ValueError(f"No TIFF files found in {input_dir}")

    total      = len(tiff_files)
    skipped    = 0
    all_metrics = []

    print(f"  Found {total} slice(s)")

    # ------------------------------------------------------------------
    # Slice-wise loop
    # ------------------------------------------------------------------
    for idx, f in enumerate(tiff_files):

        print(f"  Slice {idx + 1}/{total}: {f.name}", end="\r")

        # Load
        try:
            img = tiff.imread(f)
        except Exception as e:
            warnings.warn(f"Skipping {f.name} — could not read: {e}")
            skipped += 1
            continue

        if img.ndim != 2:
            warnings.warn(f"Skipping {f.name} — not 2D (shape={img.shape})")
            skipped += 1
            continue

        # Preprocess if needed
        if run_preprocess:
            try:
                img = preprocess_slice(img)
            except Exception as e:
                warnings.warn(f"Skipping {f.name} — preprocessing failed: {e}")
                skipped += 1
                continue

        # Threshold
        masks = {
            "otsu":    otsu(img),
            "yen":     yen(img),
            "bernsen": bernsen(img),   # reads radius + DCT from config
        }

        # Save masks + compute metrics
        for method, mask in masks.items():

            # Save — preserve original filename
            tiff.imwrite(
                out_dirs[method] / f.name,
                mask,
            )

            # Metrics
            stats             = summarize_slice(mask)
            stats["sample"]   = sample_name
            stats["method"]   = method
            stats["slice"]    = f.stem
            all_metrics.append(stats)

    processed = total - skipped
    print(f"\n  Done — {processed}/{total} slices, skipped {skipped}")

    return all_metrics


# -----------------------------------------------------------------------------
# Metrics export
# -----------------------------------------------------------------------------

def save_metrics(all_metrics: list[dict], sample_name: str) -> Path:
    """
    Save slice-wise metrics to a CSV file.

    Output: results/metrics/<sample_name>_metrics.csv

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    if not all_metrics:
        print("  No metrics to save.")
        return None

    out_path = config.METRICS_DIR / f"{sample_name}_metrics.csv"
    fieldnames = list(all_metrics[0].keys())

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)

    print(f"  Metrics saved to: {out_path}")
    return out_path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():

    # Ensure all output directories exist
    create_dirs()

    # Ask which samples to run
    selected = select_samples()

    if not selected:
        print("No samples selected. Exiting.")
        return

    total  = len(selected)
    passed = []
    failed = []

    print(f"\nProcessing {total} sample(s):")
    print("=" * 60)

    for idx, sample_name in enumerate(selected, start=1):
        print(f"\n[{idx}/{total}] {sample_name}")

        try:
            all_metrics = process_sample(sample_name)
            save_metrics(all_metrics, sample_name)
            passed.append(sample_name)

        except Exception as e:
            failed.append((sample_name, str(e)))
            warnings.warn(f"  Failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY — {total} sample(s) selected")
    print(f"  Passed : {len(passed)}")
    print(f"  Failed : {len(failed)}")

    if failed:
        print("\nFailed samples:")
        for name, reason in failed:
            print(f"  {name}: {reason}")

    print("=" * 60)


if __name__ == "__main__":
    main()
