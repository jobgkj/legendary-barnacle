"""
=============================================================================
visualize_material_boundary.py — for a handful of samples: report the
robust solid-material average intensity, and visualise the detected
solid<->air boundary overlaid on an actual slice.

Run from repository root:
    python scripts/visualize_material_boundary.py
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
from src.sample_mask import detect_sample_mask, compute_robust_solid_level

OUT_DIR = config.REPO_ROOT / "results" / "material_boundary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (sample, mid-slice index, "good range" for the robust reference)
SAMPLES = [
    ("sample_01", 450, (0, 899)),
    ("sample_05", 450, (0, 899)),
    ("sample_06", 734, (55, 1467)),   # correlation-verified reliable range
    ("sample_07", 736, (69, 1471)),   # correlation-verified reliable range
]

for sample_name, mid_idx, good_range in SAMPLES:
    proc_dir = config.REPO_ROOT / "data" / "processed_bhc_ring" / sample_name
    files = sorted(list(proc_dir.glob("*.tif")) + list(proc_dir.glob("*.tiff")))
    if not files:
        print(f"[skip] {sample_name}: no processed files found")
        continue

    print(f"\n=== {sample_name} ===")
    robust_level = compute_robust_solid_level(files, list(good_range), n_sample=15)
    print(f"  Material (solid) average intensity: {robust_level:.1f} / 255")

    img = tiff.imread(files[mid_idx])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = detect_sample_mask(img, return_circle=False)
    coverage = float(mask.mean())
    print(f"  Slice {mid_idx}: boundary coverage = {coverage*100:.1f}% of frame")

    contours = find_contours(mask.astype(float), 0.5)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0d0d0d")
    ax.imshow(img, cmap="gray")
    for c in contours:
        ax.plot(c[:, 1], c[:, 0], color="#ff6a3d", linewidth=1.6)
    ax.set_title(
        f"{sample_name}  —  slice {mid_idx}\n"
        f"Material avg intensity: {robust_level:.1f}/255  —  boundary coverage: {coverage*100:.1f}%",
        color="white", fontsize=11
    )
    ax.axis("off")
    ax.set_facecolor("#0d0d0d")

    out_path = OUT_DIR / f"{sample_name}_material_boundary.png"
    fig.savefig(out_path, dpi=140, facecolor="#0d0d0d", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")

print(f"\nDone. -> {OUT_DIR}")
