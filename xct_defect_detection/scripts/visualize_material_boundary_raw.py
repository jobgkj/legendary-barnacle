"""
=============================================================================
visualize_material_boundary_raw.py — same as visualize_material_boundary.py,
but run on RAW (unprocessed, no BHC/ring/denoise) slices instead of the
BHC+ring processed data, for comparison.

16-bit raw slices are converted to uint8 via per-slice min-max
normalisation (src/io.py::_normalize_to_uint8) before boundary
detection, since detect_sample_mask requires uint8 input.

Run from repository root:
    python scripts/visualize_material_boundary_raw.py
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
from src.io import _normalize_to_uint8

OUT_DIR = config.REPO_ROOT / "results" / "material_boundary_raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    ("sample_06", 734, (55, 1467)),
    ("sample_07", 736, (69, 1471)),
]


def load_raw_uint8(path):
    raw = tiff.imread(path)
    return _normalize_to_uint8(raw)


for sample_name, mid_idx, good_range in SAMPLES:
    raw_dir = config.RAW_DATA_DIR / sample_name
    files = sorted(list(raw_dir.glob("*.tif")) + list(raw_dir.glob("*.tiff")))
    if not files:
        print(f"[skip] {sample_name}: no raw files found")
        continue

    print(f"\n=== {sample_name} (RAW) ===")

    # compute_robust_solid_level expects tiff.imread(files[i]) to already
    # be uint8 -- raw files are 16-bit, so sample manually here instead
    # of calling it directly on the raw file list.
    lo, hi = good_range
    sample_idx = np.linspace(lo, hi, 15, dtype=int)
    solid_vals = []
    for i in sample_idx:
        img = load_raw_uint8(files[i])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask = detect_sample_mask(img, return_circle=False)
        if mask.sum() > 0:
            solid_vals.append(img[mask.astype(bool)])
    robust_level = float(np.median(np.concatenate(solid_vals)))
    print(f"  Material (solid) average intensity (raw, normalised to uint8): {robust_level:.1f} / 255")

    img = load_raw_uint8(files[mid_idx])
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
        f"{sample_name} (RAW)  —  slice {mid_idx}\n"
        f"Material avg intensity: {robust_level:.1f}/255  —  boundary coverage: {coverage*100:.1f}%",
        color="white", fontsize=11
    )
    ax.axis("off")
    ax.set_facecolor("#0d0d0d")

    out_path = OUT_DIR / f"{sample_name}_material_boundary_raw.png"
    fig.savefig(out_path, dpi=140, facecolor="#0d0d0d", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")

print(f"\nDone. -> {OUT_DIR}")
