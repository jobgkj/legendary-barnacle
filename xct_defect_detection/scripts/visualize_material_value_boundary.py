"""
=============================================================================
visualize_material_value_boundary.py — sample_06 / sample_07, preprocessed
(BHC+ring) data. Draws the boundary at the point pixel intensity crosses
relative to the material's OWN average value (robust solid-level
reference, air_threshold = material_avg * 0.5) rather than the fixed
absolute threshold=30 used in visualize_material_boundary.py.

Run from repository root:
    python scripts/visualize_material_value_boundary.py
=============================================================================
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import warnings
import numpy as np
import tifffile as tiff
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.measure import find_contours

import config
from src.sample_mask import compute_robust_solid_level, detect_sample_mask_robust

OUT_DIR = config.REPO_ROOT / "results" / "material_value_boundary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    ("sample_06", 734, (55, 1467)),
    ("sample_07", 736, (69, 1471)),
]

for sample_name, mid_idx, good_range in SAMPLES:
    proc_dir = config.REPO_ROOT / "data" / "processed_bhc_ring" / sample_name
    files = sorted(list(proc_dir.glob("*.tif")) + list(proc_dir.glob("*.tiff")))
    if not files:
        print(f"[skip] {sample_name}: no processed files found")
        continue

    print(f"\n=== {sample_name} (preprocessed) ===")
    robust_level = compute_robust_solid_level(files, list(good_range), n_sample=15)
    air_threshold = robust_level * 0.5
    print(f"  Material pixel average: {robust_level:.1f} / 255")
    print(f"  Boundary threshold (material avg x 0.5): {air_threshold:.1f}")

    img = tiff.imread(files[mid_idx])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = detect_sample_mask_robust(img, robust_level)
    coverage = float(mask.mean())
    print(f"  Slice {mid_idx}: boundary coverage = {coverage*100:.1f}% of frame")

    contours = find_contours(mask.astype(float), 0.5)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0d0d0d")
    ax.imshow(img, cmap="gray")
    for c in contours:
        ax.plot(c[:, 1], c[:, 0], color="#ff6a3d", linewidth=1.6)
    ax.set_title(
        f"{sample_name} (preprocessed)  —  slice {mid_idx}\n"
        f"Material avg: {robust_level:.1f}/255  —  boundary @ {air_threshold:.1f} "
        f"(coverage: {coverage*100:.1f}%)",
        color="white", fontsize=11
    )
    ax.axis("off")
    ax.set_facecolor("#0d0d0d")

    out_path = OUT_DIR / f"{sample_name}_material_value_boundary.png"
    fig.savefig(out_path, dpi=140, facecolor="#0d0d0d", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")

print(f"\nDone. -> {OUT_DIR}")
