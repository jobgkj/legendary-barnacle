"""
=============================================================================
compare_methods_report.py — Unified classical-vs-U-Net comparison table
=============================================================================
Produces one CSV with full-volume statistics for every thresholding method
(Otsu, Yen, Bernsen) AND the trained U-Net's predictions (where available),
per sample — meant to be dropped directly into a thesis results table.

Per-pore statistics (count, mean area, mean equivalent diameter) are pooled
across every slice in the volume using the same src/metrics.py functions
already used elsewhere in this project, rather than only the single middle
slice thesis_analysis.py originally reported — that mismatch (single-slice
classical numbers vs. full-volume U-Net numbers) would otherwise make any
side-by-side comparison table apples-to-oranges.

Output: results/comparison_table_<EXPERIMENT_NAME>.csv

Run from repository root:
    python scripts/compare_methods_report.py
    python scripts/compare_methods_report.py sample_01 sample_02
=============================================================================
"""
import sys
import csv
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# See pipeline.py for why this is needed: non-ASCII prints (banners, symbols)
# otherwise crash under a non-UTF-8 console/redirect encoding (e.g. Windows cp1252).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import tifffile as tiff

import config
from config import EXPERIMENT_NAME
from src.metrics import pore_properties

METHODS = ["otsu", "yen", "bernsen"]


def _load_mask_volume(mask_dir: Path) -> "np.ndarray | None":
    files = sorted(list(mask_dir.glob("*.tif")) + list(mask_dir.glob("*.tiff")))
    if not files:
        return None
    slices = [tiff.imread(f) for f in files]
    return np.stack(slices, axis=0)


def summarize_volume(mask_vol: np.ndarray, method_label: str, sample: str) -> dict:
    """
    Pool per-pore statistics across every slice (2D connected components,
    consistent with the rest of this project — no 3D connected-component
    labelling is implemented here), plus a true full-volume defect fraction.
    """
    n_slices = mask_vol.shape[0]
    total_pixels = mask_vol.size
    defect_fraction = float((mask_vol > 0).sum()) / total_pixels

    all_areas, all_diams = [], []
    for i in range(n_slices):
        slc = (mask_vol[i] > 0).astype(np.uint8)
        props = pore_properties(slc)
        all_areas.append(props["areas"])
        all_diams.append(props["equivalent_diameters"])
        if (i + 1) % 100 == 0 or (i + 1) == n_slices:
            print(f"      {method_label}: {i+1}/{n_slices} slices analysed", end="\r")
    print()

    areas = np.concatenate(all_areas) if all_areas else np.array([])
    diams = np.concatenate(all_diams) if all_diams else np.array([])
    px = config.PIXEL_SIZE_UM

    return {
        "sample": sample,
        "method": method_label,
        "n_slices": n_slices,
        "defect_fraction_pct": round(defect_fraction * 100, 5),
        "pore_count_total": int(areas.size),
        "mean_pore_count_per_slice": round(areas.size / n_slices, 2) if n_slices else 0.0,
        "mean_pore_area_px2": round(float(areas.mean()), 3) if areas.size else 0.0,
        "mean_pore_area_um2": round(float(areas.mean()) * px**2, 3) if areas.size else 0.0,
        "mean_equiv_diameter_px": round(float(diams.mean()), 3) if diams.size else 0.0,
        "mean_equiv_diameter_um": round(float(diams.mean()) * px, 3) if diams.size else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", nargs="*", help="Samples to include (default: all discovered)")
    args = parser.parse_args()

    sample_names = args.samples or config.SAMPLE_NAMES
    rows = []

    for sample in sample_names:
        print("=" * 65)
        print(f"SAMPLE: {sample}")
        print("=" * 65)

        for method in METHODS:
            mask_dir = config.MASKS_DIR / sample / method
            mask_vol = _load_mask_volume(mask_dir)
            if mask_vol is None:
                print(f"  [Skip] No '{method}' masks found for {sample}")
                continue
            print(f"  [{method}] volume shape {mask_vol.shape}")
            rows.append(summarize_volume(mask_vol, method, sample))

        # U-Net prediction, if this sample has one under the active experiment
        pred_path = (config.REPO_ROOT / "results" / "unet_predictions" /
                     EXPERIMENT_NAME / f"{sample}_mask.npy")
        if pred_path.exists():
            mask_vol = np.load(pred_path)
            print(f"  [unet:{EXPERIMENT_NAME}] volume shape {mask_vol.shape}")
            rows.append(summarize_volume(mask_vol, f"unet_{EXPERIMENT_NAME}", sample))
        else:
            print(f"  [Skip] No U-Net prediction found for {sample} "
                  f"(experiment '{EXPERIMENT_NAME}')")

    if not rows:
        print("\nNo data found — nothing written.")
        return

    out_path = config.REPO_ROOT / "results" / f"comparison_table_{EXPERIMENT_NAME}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
