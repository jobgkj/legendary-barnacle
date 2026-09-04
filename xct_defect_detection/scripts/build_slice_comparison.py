"""
=============================================================================
build_slice_comparison.py — assemble side-by-side comparison figures
(Preprocessed | Bernsen | 2.5D U-Net | 2D U-Net) from the .npz files
saved by predict_slices_only.py for each checkpoint.

Run from repository root:
    python scripts/build_slice_comparison.py sample_02:450 sample_07:737
=============================================================================
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

EXP_25D = "bhc_ring_bce_shapeaware"
EXP_2D  = "plain2d_sample02_vs_sample05"
OUT_DIR = config.REPO_ROOT / "results" / "slice_compare" / "combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

specs = sys.argv[1:] or ["sample_02:450", "sample_07:737"]

for spec in specs:
    sample_name, idx_str = spec.split(":")
    idx = int(idx_str)

    p_25d = config.REPO_ROOT / "results" / "slice_compare" / EXP_25D / f"{sample_name}_{idx}.npz"
    p_2d  = config.REPO_ROOT / "results" / "slice_compare" / EXP_2D  / f"{sample_name}_{idx}.npz"

    if not p_25d.exists() or not p_2d.exists():
        print(f"[skip] {sample_name}:{idx} -- missing {'25D' if not p_25d.exists() else '2D'} data")
        continue

    d25 = np.load(p_25d)
    d2  = np.load(p_2d)

    img       = d25["preprocessed"]
    bernsen   = d25["bernsen"]
    pred_25d  = d25["pred"]
    pred_2d   = d2["pred"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), facecolor="#0d0d0d")
    panels = [
        (img, "gray", f"{sample_name} slice {idx}\nPreprocessed", None),
        (img, "gray", f"Bernsen\n({bernsen.mean()*100:.2f}% defect)", bernsen),
        (img, "gray", f"2.5D U-Net (bhc_ring)\n({pred_25d.mean()*100:.2f}% defect)", pred_25d),
        (img, "gray", f"2D U-Net (sample_02-only)\n({pred_2d.mean()*100:.2f}% defect)", pred_2d),
    ]
    for ax, (base, cmap, title, overlay) in zip(axes, panels):
        ax.imshow(base, cmap=cmap, vmin=0, vmax=1)
        if overlay is not None:
            ax.imshow(overlay, cmap="Reds", alpha=0.45, vmin=0, vmax=1)
        ax.set_title(title, color="white", fontsize=12)
        ax.axis("off")
        ax.set_facecolor("#0d0d0d")

    plt.tight_layout()
    out_path = OUT_DIR / f"{sample_name}_{idx}_comparison.png"
    fig.savefig(out_path, dpi=130, facecolor="#0d0d0d", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")

print(f"\nDone. -> {OUT_DIR}")
