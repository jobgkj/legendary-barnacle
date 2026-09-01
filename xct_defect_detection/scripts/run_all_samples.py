"""
=============================================================================
run_all_samples.py — Process selected samples in data/raw/
=============================================================================

Discovers all sample folders, lets the user choose which ones to run,
then generates segmentation masks for each selected sample.

Run from repository root:
    python scripts/run_all_samples.py
=============================================================================
"""
import sys
from pathlib import Path

# Add project root to path so config.py can be found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import warnings

from config import create_dirs
from src.io import load_and_generate_masks


# -----------------------------------------------------------------------------
# Sample selection
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

    # Display available samples
    print("\nAvailable samples:")
    print("-" * 40)
    for idx, d in enumerate(sample_dirs, start=1):
        print(f"  [{idx}] {d.name}")
    print(f"  [0] Run ALL samples")
    print("-" * 40)

    # Prompt until valid input
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

        # Run all
        if 0 in choices:
            return [d.name for d in sample_dirs]

        # Validate range
        invalid = [c for c in choices if c < 1 or c > len(sample_dirs)]
        if invalid:
            print(
                f"  Invalid selection(s): {invalid} — "
                f"choose between 1 and {len(sample_dirs)}."
            )
            continue

        # Remove duplicates, preserve order
        seen = set()
        selected = []
        for c in choices:
            if c not in seen:
                seen.add(c)
                selected.append(sample_dirs[c - 1].name)

        return selected


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():

    # Ensure all output directories exist
    create_dirs()

    # Ask user which samples to run
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
            result = load_and_generate_masks(
                repo_root=config.REPO_ROOT,
                sample_name=sample_name,
            )
            passed.append(sample_name)
            print(
                f"  Done — {result['processed']}/{result['total']} slices, "
                f"skipped {result['skipped']}"
            )

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
