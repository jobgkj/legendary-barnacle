"""
=============================================================================
Thesis Preprocessing Analysis
=============================================================================
Produces thesis-ready outputs for all 5 preprocessing analysis tasks:

    Task 1  Patch extraction (64x64 and 128x128)
    Task 2  Augmentation validation on extracted patches
    Task 3  Before/after visualisation for each preprocessing step
    Task 4  Statistical analysis — histograms, noise, SNR measurements
    Task 5  Parameter tuning comparison (NLM, BHC, ring filter)

Run from project root:
    python thesis_analysis.py

Outputs saved to:  thesis_outputs/
=============================================================================
"""

import os
import sys
import time
import numpy as np
import tifffile as tiff
import glob
import matplotlib
matplotlib.use("Agg")                          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from scipy.ndimage import median_filter, gaussian_filter
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import view_as_windows

# ── Project path ────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_DIR        = os.path.join(ROOT, r"data\tiff_stack")
PROCESSED_DIR  = os.path.join(ROOT, r"data\tiff_output")
OUT_DIR        = os.path.join(ROOT, "thesis_outputs")
PATCH_DIR_64   = os.path.join(OUT_DIR, "patches_64x64")
PATCH_DIR_128  = os.path.join(OUT_DIR, "patches_128x128")

os.makedirs(OUT_DIR,       exist_ok=True)
os.makedirs(PATCH_DIR_64,  exist_ok=True)
os.makedirs(PATCH_DIR_128, exist_ok=True)

PLOT_DPI   = 150
N_SLICES   = 10      # number of slices to sample for analysis
SLICE_STEP = None    # auto-computed from volume


# =============================================================================
# Helpers
# =============================================================================

def load_stack(folder: str, max_slices: int = None) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(folder, "*.tif")))
    if not files:
        raise RuntimeError(f"No .tif files found in '{folder}'")
    if max_slices:
        files = files[:max_slices]
    slices = [tiff.imread(f).astype(np.float32) for f in files]
    return np.stack(slices, axis=0)


def save_fig(fig, name: str):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Saved] {name}")
    return path


def sample_slices(volume: np.ndarray, n: int = N_SLICES) -> list[int]:
    total = volume.shape[0]
    return list(np.linspace(0, total - 1, n, dtype=int))


# =============================================================================
# Preprocessing steps (self-contained so we can apply one at a time)
# =============================================================================

def step_normalize(v: np.ndarray, p_low=1, p_high=99) -> np.ndarray:
    lo, hi = np.percentile(v, p_low), np.percentile(v, p_high)
    return np.clip((v - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)


def step_bhc(v: np.ndarray, degree: int = 3) -> np.ndarray:
    if v.ndim == 3:
        profile = v.mean(axis=(1, 2))
        x       = np.arange(len(profile))
        trend   = np.polyval(np.polyfit(x, profile, degree), x)
        v       = np.clip(v - (trend - trend.mean())[:, None, None], 0, 1)
    return v.astype(np.float32)


def step_ring(v: np.ndarray, radius: int = 15) -> np.ndarray:
    def _fix(slc):
        h, w   = slc.shape
        cy, cx = h // 2, w // 2
        yi, xi = np.indices((h, w))
        r      = np.sqrt((xi - cx)**2 + (yi - cy)**2).astype(np.int32)
        r_max  = r.max() + 1
        rm     = np.array([slc[r == ri].mean() if (r == ri).any() else 0.
                           for ri in range(r_max)])
        rs     = median_filter(rm, size=radius)
        corr   = rs[np.clip(r, 0, r_max-1)] - rm[np.clip(r, 0, r_max-1)]
        return np.clip(slc + corr, 0, 1).astype(np.float32)
    if v.ndim == 3:
        return np.stack([_fix(v[i]) for i in range(v.shape[0])])
    return _fix(v)


def step_nlm(v: np.ndarray, h_factor: float = 1.0,
             patch_size: int = 5, patch_dist: int = 6) -> np.ndarray:
    def _denoise(slc):
        sigma = estimate_sigma(slc)
        return denoise_nl_means(slc, h=h_factor * sigma,
                                patch_size=patch_size,
                                patch_distance=patch_dist,
                                fast_mode=True).astype(np.float32)
    if v.ndim == 3:
        return np.stack([_denoise(v[i]) for i in range(v.shape[0])])
    return _denoise(v)


def compute_snr(signal: np.ndarray) -> float:
    mu  = signal.mean()
    std = signal.std()
    return float(mu / std) if std > 0 else 0.0


def compute_cnr(signal: np.ndarray, background: np.ndarray) -> float:
    return float(abs(signal.mean() - background.mean()) /
                 (np.sqrt(signal.std()**2 + background.std()**2) + 1e-8))


# =============================================================================
# TASK 1 — Patch Extraction
# =============================================================================

def task1_patch_extraction(processed: np.ndarray):
    print("\n" + "="*60)
    print("  TASK 1 — Patch Extraction")
    print("="*60)

    stats = {}
    for patch_size, patch_dir in [(64, PATCH_DIR_64), (128, PATCH_DIR_128)]:
        count = 0
        stride = patch_size // 2           # 50 % overlap

        sample_idxs = sample_slices(processed, n=20)
        patches_saved = []

        for si in sample_idxs:
            slc = processed[si]
            h, w = slc.shape

            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    patch = slc[y:y+patch_size, x:x+patch_size]

                    # Skip near-empty patches (mostly background)
                    if patch.std() < 0.01:
                        continue

                    fname = os.path.join(
                        patch_dir, f"slice{si:04d}_y{y:04d}_x{x:04d}.tif"
                    )
                    tiff.imwrite(fname, patch)
                    count += 1
                    if len(patches_saved) < 9:
                        patches_saved.append((patch, si, y, x))

        stats[patch_size] = count
        print(f"  [{patch_size}x{patch_size}] Extracted {count} patches "
              f"→ {patch_dir}")

        # ── Figure: patch grid ──────────────────────────────────────────────
        fig, axes = plt.subplots(3, 3, figsize=(9, 9),
                                  facecolor="#0d0d0d")
        fig.suptitle(f"Extracted Patches — {patch_size}×{patch_size}px\n"
                     f"(50% overlap, {count} total patches from 20 slices)",
                     color="white", fontsize=13, y=1.01)
        for ax, (patch, si, y, x) in zip(axes.ravel(), patches_saved):
            ax.imshow(patch, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"Slice {si} | y={y} x={x}",
                         color="#aaaaaa", fontsize=7)
            ax.axis("off")
        for ax in axes.ravel()[len(patches_saved):]:
            ax.axis("off")
        plt.tight_layout()
        save_fig(fig, f"task1_patches_{patch_size}x{patch_size}.png")

    # ── Summary bar chart ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    bars = ax.bar(["64×64", "128×128"], list(stats.values()),
                  color=["#e05c5c", "#5ca8e0"], width=0.5)
    for bar, val in zip(bars, stats.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{val:,}", ha="center", color="white", fontsize=11)
    ax.set_title("Patch Count by Patch Size", color="white", fontsize=13)
    ax.set_ylabel("Number of Patches", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333333")
    plt.tight_layout()
    save_fig(fig, "task1_patch_count_summary.png")

    return stats


# =============================================================================
# TASK 2 — Augmentation Validation
# =============================================================================

def task2_augmentation(processed: np.ndarray):
    print("\n" + "="*60)
    print("  TASK 2 — Augmentation Validation")
    print("="*60)

    try:
        from xct_defect_detection.data.augmentation import apply_augmentation
    except ImportError:
        print("  [WARN] Could not import augmentation.py — using fallback.")
        def apply_augmentation(img, msk):
            if np.random.random() < 0.5:
                img, msk = np.fliplr(img), np.fliplr(msk)
            k = np.random.choice([0,1,2,3])
            return np.rot90(img, k), np.rot90(msk, k)

    # Pick a representative slice and extract one patch
    mid = processed.shape[0] // 2
    slc = processed[mid]
    ps  = 128
    h, w = slc.shape
    y0, x0 = (h - ps) // 2, (w - ps) // 2
    patch  = slc[y0:y0+ps, x0:x0+ps]
    dummy_mask = (patch > 0.5).astype(np.uint8)

    aug_names = [
        "Original",
        "Aug 1", "Aug 2", "Aug 3",
        "Aug 4", "Aug 5", "Aug 6",
        "Aug 7", "Aug 8",
    ]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10), facecolor="#0d0d0d")
    fig.suptitle("Augmentation Pipeline Validation\n"
                 "(same patch, 8 independent augmentations)",
                 color="white", fontsize=13)

    axes.ravel()[0].imshow(patch, cmap="gray", vmin=0, vmax=1)
    axes.ravel()[0].set_title("Original", color="#aaaaaa", fontsize=9)
    axes.ravel()[0].axis("off")

    for i in range(1, 9):
        aug_img, _ = apply_augmentation(patch.copy(), dummy_mask.copy())
        axes.ravel()[i].imshow(aug_img, cmap="gray", vmin=0, vmax=1)
        axes.ravel()[i].set_title(aug_names[i], color="#aaaaaa", fontsize=9)
        axes.ravel()[i].axis("off")

    plt.tight_layout()
    save_fig(fig, "task2_augmentation_validation.png")
    print("  Augmentation validation complete.")


# =============================================================================
# TASK 3 — Before / After Visualisation
# =============================================================================

def task3_before_after(raw: np.ndarray):
    print("\n" + "="*60)
    print("  TASK 3 — Before/After Preprocessing Visualisation")
    print("="*60)

    mid = raw.shape[0] // 2
    slc_raw  = raw[mid]

    steps = {}
    print("  Applying steps sequentially ...")
    v1 = step_normalize(raw);         steps["1_Normalised"]          = v1[mid]
    v2 = step_bhc(v1);                steps["2_BHC_Corrected"]       = v2[mid]
    v3 = step_ring(v2);               steps["3_Ring_Suppressed"]     = v3[mid]
    v4 = step_nlm(v3);                steps["4_NLM_Denoised"]        = v4[mid]

    all_steps = {"0_Raw": slc_raw, **steps}
    labels    = ["Raw", "Normalised", "BHC\nCorrected",
                 "Ring\nSuppressed", "NLM\nDenoised"]

    # ── Panel: side-by-side slices ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), facecolor="#0d0d0d")
    fig.suptitle(f"Preprocessing Pipeline — Slice {mid}\n"
                 f"(Volume: {raw.shape[0]}×{raw.shape[1]}×{raw.shape[2]})",
                 color="white", fontsize=13)
    for ax, (name, slc), label in zip(axes, all_steps.items(), labels):
        ax.imshow(slc, cmap="gray")
        ax.set_title(label, color="white", fontsize=10, pad=6)
        ax.axis("off")
        # intensity range annotation
        ax.text(0.5, -0.06,
                f"[{slc.min():.3f}, {slc.max():.3f}]",
                transform=ax.transAxes, ha="center",
                color="#888888", fontsize=8)
    plt.tight_layout()
    save_fig(fig, "task3_before_after_slices.png")

    # ── Panel: zoomed ROI comparison ────────────────────────────────────────
    h, w = slc_raw.shape
    y0, x0 = h//4, w//4
    roi_h, roi_w = h//4, w//4

    fig, axes = plt.subplots(1, 5, figsize=(20, 5), facecolor="#0d0d0d")
    fig.suptitle("ROI Zoom — Centre Quarter of Each Preprocessing Stage",
                 color="white", fontsize=13)
    for ax, (name, slc), label in zip(axes, all_steps.items(), labels):
        roi = slc[y0:y0+roi_h, x0:x0+roi_w]
        ax.imshow(roi, cmap="gray")
        ax.set_title(label, color="white", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "task3_before_after_roi_zoom.png")

    print("  Before/after visualisation complete.")
    return v4   # return fully preprocessed volume


# =============================================================================
# TASK 4 — Statistical Analysis
# =============================================================================

def task4_statistics(raw: np.ndarray, processed: np.ndarray):
    print("\n" + "="*60)
    print("  TASK 4 — Statistical Analysis")
    print("="*60)

    sample_idx = sample_slices(raw, n=N_SLICES)

    # ── Histogram comparison ─────────────────────────────────────────────────
    raw_flat  = np.concatenate([raw[i].ravel()       for i in sample_idx])
    proc_flat = np.concatenate([processed[i].ravel() for i in sample_idx])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d0d0d")
    fig.suptitle("Intensity Histograms — Raw vs Preprocessed\n"
                 f"(sampled from {N_SLICES} slices)",
                 color="white", fontsize=13)
    for ax, data, label, color in zip(
        axes,
        [raw_flat, proc_flat],
        ["Raw (float32)", "Preprocessed (normalised)"],
        ["#e05c5c", "#5ca8e0"]
    ):
        ax.set_facecolor("#0d0d0d")
        ax.hist(data, bins=256, color=color, alpha=0.85, edgecolor="none")
        ax.set_title(label, color="white", fontsize=11)
        ax.set_xlabel("Intensity", color="white")
        ax.set_ylabel("Frequency", color="white")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333333")
        # Stats annotation
        ax.axvline(np.mean(data), color="white", lw=1.2, ls="--",
                   label=f"μ={np.mean(data):.3f}")
        ax.axvline(np.median(data), color="yellow", lw=1.2, ls=":",
                   label=f"med={np.median(data):.3f}")
        ax.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)
    plt.tight_layout()
    save_fig(fig, "task4_histograms.png")

    # ── SNR and noise per slice ──────────────────────────────────────────────
    snr_raw  = [compute_snr(raw[i])       for i in sample_idx]
    snr_proc = [compute_snr(processed[i]) for i in sample_idx]
    std_raw  = [raw[i].std()              for i in sample_idx]
    std_proc = [processed[i].std()        for i in sample_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d0d0d")
    fig.suptitle("Noise & SNR Analysis — Per Slice Comparison",
                 color="white", fontsize=13)

    for ax in axes:
        ax.set_facecolor("#0d0d0d")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333333")

    axes[0].plot(sample_idx, snr_raw,  "o-", color="#e05c5c",
                 label="Raw",          lw=2)
    axes[0].plot(sample_idx, snr_proc, "s-", color="#5ca8e0",
                 label="Preprocessed", lw=2)
    axes[0].set_title("Signal-to-Noise Ratio per Slice",
                      color="white", fontsize=11)
    axes[0].set_xlabel("Slice Index", color="white")
    axes[0].set_ylabel("SNR",         color="white")
    axes[0].legend(facecolor="#1a1a1a", labelcolor="white")

    axes[1].plot(sample_idx, std_raw,  "o-", color="#e05c5c",
                 label="Raw",          lw=2)
    axes[1].plot(sample_idx, std_proc, "s-", color="#5ca8e0",
                 label="Preprocessed", lw=2)
    axes[1].set_title("Noise Std Dev per Slice",
                      color="white", fontsize=11)
    axes[1].set_xlabel("Slice Index", color="white")
    axes[1].set_ylabel("Std Dev",     color="white")
    axes[1].legend(facecolor="#1a1a1a", labelcolor="white")

    plt.tight_layout()
    save_fig(fig, "task4_snr_noise.png")

    # ── Summary stats table ──────────────────────────────────────────────────
    stats_table = {
        "Metric":       ["Mean",      "Std Dev",    "Min",
                         "Max",       "SNR (mean)"],
        "Raw":          [f"{raw_flat.mean():.4f}",
                         f"{raw_flat.std():.4f}",
                         f"{raw_flat.min():.4f}",
                         f"{raw_flat.max():.4f}",
                         f"{np.mean(snr_raw):.3f}"],
        "Preprocessed": [f"{proc_flat.mean():.4f}",
                         f"{proc_flat.std():.4f}",
                         f"{proc_flat.min():.4f}",
                         f"{proc_flat.max():.4f}",
                         f"{np.mean(snr_proc):.3f}"],
    }

    fig, ax = plt.subplots(figsize=(8, 3), facecolor="#0d0d0d")
    ax.axis("off")
    fig.suptitle("Statistical Summary — Raw vs Preprocessed",
                 color="white", fontsize=13)
    tbl = ax.table(
        cellText=list(zip(
            stats_table["Metric"],
            stats_table["Raw"],
            stats_table["Preprocessed"]
        )),
        colLabels=["Metric", "Raw", "Preprocessed"],
        cellLoc="center", loc="center",
        colColours=["#222222","#3a1515","#152233"],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#444444")
        cell.set_text_props(color="white")
        cell.set_facecolor("#0d0d0d" if row > 0 else
                           ["#222222","#3a1515","#152233"][col])
    plt.tight_layout()
    save_fig(fig, "task4_stats_table.png")

    print(f"  Mean SNR improvement: "
          f"{np.mean(snr_raw):.3f} → {np.mean(snr_proc):.3f}")
    print(f"  Mean noise reduction: "
          f"{np.mean(std_raw):.4f} → {np.mean(std_proc):.4f}")


# =============================================================================
# TASK 5 — Parameter Tuning
# =============================================================================

def task5_parameter_tuning(raw: np.ndarray):
    print("\n" + "="*60)
    print("  TASK 5 — Parameter Tuning Comparison")
    print("="*60)

    mid  = raw.shape[0] // 2
    norm = step_normalize(raw)
    slc  = norm[mid]

    # ── NLM h_factor tuning ─────────────────────────────────────────────────
    print("  Testing NLM h_factor values ...")
    h_values = [0.5, 1.0, 1.5, 2.0]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor="#0d0d0d")
    fig.suptitle("NLM Denoising — h_factor Comparison\n"
                 "(higher h = more smoothing, less detail)",
                 color="white", fontsize=13)
    sigma = estimate_sigma(slc)
    for ax, h in zip(axes, h_values):
        denoised = denoise_nl_means(slc, h=h*sigma,
                                    patch_size=5, patch_distance=6,
                                    fast_mode=True).astype(np.float32)
        snr = compute_snr(denoised)
        ax.imshow(denoised, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"h={h}×σ\nSNR={snr:.2f}",
                     color="white", fontsize=10)
        ax.axis("off")
        ax.set_facecolor("#0d0d0d")
    plt.tight_layout()
    save_fig(fig, "task5_nlm_h_tuning.png")

    # ── BHC polynomial degree tuning ────────────────────────────────────────
    print("  Testing BHC polynomial degrees ...")
    degrees  = [1, 2, 3, 5]
    bhc_snrs = []
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor="#0d0d0d")
    fig.suptitle("Beam Hardening Correction — Polynomial Degree Comparison",
                 color="white", fontsize=13)
    for ax, deg in zip(axes, degrees):
        corrected = step_bhc(norm, degree=deg)
        snr       = compute_snr(corrected[mid])
        bhc_snrs.append(snr)
        ax.imshow(corrected[mid], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Degree {deg}\nSNR={snr:.2f}",
                     color="white", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "task5_bhc_degree_tuning.png")

    # ── Ring filter radius tuning ────────────────────────────────────────────
    print("  Testing ring filter radius values ...")
    bhc_vol = step_bhc(norm, degree=3)
    radii   = [5, 10, 15, 25]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor="#0d0d0d")
    fig.suptitle("Ring Artefact Suppression — Filter Radius Comparison",
                 color="white", fontsize=13)
    for ax, r in zip(axes, radii):
        ring_suppressed = step_ring(bhc_vol, radius=r)
        snr             = compute_snr(ring_suppressed[mid])
        ax.imshow(ring_suppressed[mid], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Radius={r}\nSNR={snr:.2f}",
                     color="white", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "task5_ring_radius_tuning.png")

    # ── Combined SNR comparison bar chart ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    x      = np.arange(4)
    width  = 0.25
    snr_nlm  = []
    sigma    = estimate_sigma(slc)
    for h in h_values:
        d = denoise_nl_means(slc, h=h*sigma, patch_size=5,
                             patch_distance=6, fast_mode=True)
        snr_nlm.append(compute_snr(d))

    ax.bar(x - width, snr_nlm,  width, label="NLM h_factor",
           color="#e05c5c", alpha=0.85)
    ax.bar(x,         bhc_snrs, width, label="BHC degree",
           color="#5ca8e0", alpha=0.85)

    ax.set_title("Parameter Tuning — SNR Comparison",
                 color="white", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([str(h_values[i]) + " / deg " + str(degrees[i])
                        for i in range(4)], color="white", fontsize=8)
    ax.set_ylabel("SNR", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333333")
    ax.legend(facecolor="#1a1a1a", labelcolor="white")
    plt.tight_layout()
    save_fig(fig, "task5_parameter_snr_comparison.png")

    print("  Parameter tuning complete.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("=" * 60)
    print("  XCT Thesis Preprocessing Analysis")
    print("  University West — PODFAM Research Project 2026")
    print("=" * 60)
    print(f"\n  Output directory: {OUT_DIR}")

    # ── Load raw volume (sample first 50 slices for speed on tasks 3-5) ─────
    print("\n[Loading] Raw TIFF stack ...")
    t0  = time.time()
    raw = load_stack(RAW_DIR, max_slices=50)
    print(f"  Loaded {raw.shape[0]} slices in {time.time()-t0:.1f}s")
    print(f"  Shape: {raw.shape}  |  Range: [{raw.min():.1f}, {raw.max():.1f}]")

    # ── Load preprocessed volume (for tasks 1, 2, 4) ────────────────────────
    print("\n[Loading] Preprocessed TIFF stack ...")
    t0        = time.time()
    processed = load_stack(PROCESSED_DIR, max_slices=50)
    print(f"  Loaded {processed.shape[0]} slices in {time.time()-t0:.1f}s")

    # ── Run all tasks ────────────────────────────────────────────────────────
    task1_patch_extraction(processed)
    task2_augmentation(processed)
    task3_before_after(raw)
    task4_statistics(raw, processed)
    task5_parameter_tuning(raw)

    print()
    print("=" * 60)
    print("  ALL TASKS COMPLETE")
    print(f"  All outputs saved to: {OUT_DIR}")
    print("=" * 60)
    print()
    print("  Files generated:")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith(".png"):
            print(f"    → {f}")
    print()


if __name__ == "__main__":
    main()
