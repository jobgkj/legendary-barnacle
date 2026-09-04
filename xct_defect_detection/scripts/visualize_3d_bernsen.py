"""
=============================================================================
visualize_3d_bernsen.py — 3D render of the CLASSICAL Bernsen result for
one sample, using the exact same BHC+ring shape-aware pipeline data (and
the same plotting code) as predict_and_visualize_3d.py, for a true
apples-to-apples comparison against the U-Net's 3D render.

Reuses data/cache_bhc_ring_shapeaware/<sample>_{volume,mask}.npy directly
-- that cache already IS (processed volume, Bernsen mask) from the
current pipeline, since load_cache(name, mask_method="bernsen") is what
built it. No new thresholding is computed here.

Note on filenames: predict_and_visualize_3d.py's visualize_3d() hardcodes
"_unet_" in every output filename regardless of what mask it's given —
harmless here (the mask is Bernsen's, not the U-Net's) but left as-is
rather than forking that function; the sample name passed in makes the
source unambiguous (e.g. sample_07_bernsen_classical_unet_3d_mesh.png).

Run from repository root:
    python scripts/visualize_3d_bernsen.py sample_07
=============================================================================
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

import config
from config import create_dirs, EXPERIMENT_NAME
from data.cache import load_cache
from scripts.predict_and_visualize_3d import visualize_3d, FIGURE_DIR

create_dirs()
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sample_name = sys.argv[1] if len(sys.argv) > 1 else "sample_07"

print(f"Experiment (figures namespace): {EXPERIMENT_NAME}")
print(f"PROCESSED_DATA_DIR = {config.PROCESSED_DATA_DIR}")
print(f"MASKS_DIR           = {config.MASKS_DIR}")
print(f"CACHE_DIR            = {config.CACHE_DIR}")
print(f"Sample: {sample_name}\n")

volume, mask = load_cache(sample_name, mask_method="bernsen")
volume = np.asarray(volume, dtype=np.float32)
mask   = np.asarray(mask)

defect_frac = mask.mean() * 100
print(f"Bernsen defect fraction: {defect_frac:.3f}%")

print("[3D] Building Bernsen visualisations ...")
visualize_3d(f"{sample_name}_bernsen_classical", volume, mask)
print(f"\nDone. Figures -> {FIGURE_DIR}")
