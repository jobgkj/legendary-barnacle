"""
=============================================================================
predict_and_visualize_3d.py — Run the trained 2D U-Net on full samples and
build a 3D reconstruction from its predictions
=============================================================================
Loads artifacts/best_model_<EXPERIMENT_NAME>.pt (trained by pipeline.py —
see config.py's EXPERIMENT_NAME) and runs sliding-window inference over
every slice of one or more samples, then renders the predicted defect
volume in 3D (marching-cubes mesh + ghost shell, scatter, orthographic
MIPs, slice mosaic) — the same visual style as models/thesis_analysis.py's
Task 5/6, but driven by the U-Net's predictions instead of classical
thresholding.

Output is namespaced by EXPERIMENT_NAME so results from a different
training run (different TRAINING_SAMPLES_OVERRIDE / MAX_TRAINING_SAMPLES
in config.py) never overwrite this one's:
    results/unet_predictions/<EXPERIMENT_NAME>/<sample>/pred_XXXX.tif
    results/unet_predictions/<EXPERIMENT_NAME>/<sample>_mask.npy
    results/figures/<EXPERIMENT_NAME>/<sample>_unet_*.png

Defaults to running on whichever samples this experiment's training run
did *not* use (see pipeline.py::select_training_samples) — pass sample
names explicitly to override, or --all for every discovered sample.

Run from repository root:
    python scripts/predict_and_visualize_3d.py sample_04 sample_05
    python scripts/predict_and_visualize_3d.py --all
=============================================================================
"""
import sys
import argparse
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# See pipeline.py for why this is needed: non-ASCII prints (banners, symbols)
# otherwise crash under a non-UTF-8 console/redirect encoding (e.g. Windows cp1252).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import tifffile as tiff
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter

import config
from config import create_dirs, PATCH_SIZE, PATCH_STRIDE, DICE_THRESHOLD, DEVICE, EXPERIMENT_NAME, UNET_INPUT_SLICES
from data.cache import load_cache
from models.unet2d import get_model
from pipeline import select_training_samples
from src.sample_mask import detect_sample_mask

INFER_BATCH_SIZE = 32
PRED_DIR    = config.REPO_ROOT / "results" / "unet_predictions" / EXPERIMENT_NAME
FIGURE_DIR  = config.REPO_ROOT / "results" / "figures" / EXPERIMENT_NAME


def _patch_grid(h: int, w: int, p: int, s: int) -> list[tuple[int, int]]:
    """Sliding-window top-left corners covering the full slice, edges included."""
    ys = list(range(0, max(h - p, 0) + 1, s))
    xs = list(range(0, max(w - p, 0) + 1, s))
    if ys[-1] != h - p:
        ys.append(h - p)
    if xs[-1] != w - p:
        xs.append(w - p)
    return [(y, x) for y in ys for x in xs]


def _slice_shape_mask(img01: np.ndarray) -> np.ndarray:
    """
    Shape-aware "inside sample" mask for one slice (float32 in [0,1]).
    Same hybrid logic as scripts/regenerate_masks_shape_aware_full.py:
    fixed air_threshold=30 first, falling back to a per-slice adaptive
    Otsu threshold only when the fixed one implausibly finds ~no
    background (>95% coverage) — the known failure mode on transition
    slices near the top/bottom of a scan (support structure rather than
    open air). Returns a bool array, True = inside the sample.
    """
    img_u8 = (img01 * 255.0).clip(0, 255).astype(np.uint8)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        try:
            mask = detect_sample_mask(img_u8, return_circle=False)
        except ValueError:
            # No sample region detected at all -- nothing to keep.
            return np.zeros(img01.shape, dtype=bool)
    if float(mask.mean()) > 0.95:
        try:
            mask = detect_sample_mask(img_u8, return_circle=False, adaptive_threshold=True)
        except ValueError:
            return np.zeros(img01.shape, dtype=bool)
    return mask.astype(bool)


def predict_volume(model, volume: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Sliding-window inference over every slice of `volume` (N, H, W) float32
    in [0, 1]. Each slice is inferred from a 2.5D stack of UNET_INPUT_SLICES
    adjacent slices (matching training — see
    data/dataset.py::XCTPatchDataset._get_slice_stack), with neighbours past
    a volume edge clamped rather than zero-padded. Overlapping patch
    predictions are averaged before thresholding. Returns a uint8 binary
    mask volume of the same shape.

    Background (air / support structure outside the part) is zeroed out
    of the input BEFORE the U-Net sees it, and masked out of the output
    prediction again afterwards — on slices with little or no metal in
    frame (e.g. transition slices near the top/bottom of a scan), the
    network was observed to produce spurious "defect" predictions on
    that background; both the mask on the input and the belt-and-suspenders
    mask on the output prevent that region from ever contributing.
    """
    n, h, w = volume.shape
    p, s = PATCH_SIZE, PATCH_STRIDE
    corners = _patch_grid(h, w, p, s)
    half = UNET_INPUT_SLICES // 2

    mask_vol = np.zeros((n, h, w), dtype=np.uint8)
    n_masked_slices = 0

    with torch.no_grad():
        for i in range(n):
            shape_mask = _slice_shape_mask(np.asarray(volume[i], dtype=np.float32))
            if not shape_mask.any():
                # Nothing identifiable as sample on this slice -- skip
                # inference entirely, leave it as all-background (0).
                n_masked_slices += 1
                if (i + 1) % 50 == 0 or (i + 1) == n:
                    print(f"      Inferred {i+1}/{n} slices ...", end="\r")
                continue

            neighbour_idxs = [
                int(np.clip(i + offset, 0, n - 1))
                for offset in range(-half, half + 1)
            ]
            slc_stack = np.stack(
                [np.asarray(volume[j], dtype=np.float32) * shape_mask for j in neighbour_idxs],
                axis=0
            )  # (N, H, W) -- background zeroed before the model sees it

            prob_sum = np.zeros((h, w), dtype=np.float32)
            count    = np.zeros((h, w), dtype=np.float32)

            for batch_start in range(0, len(corners), INFER_BATCH_SIZE):
                batch_corners = corners[batch_start:batch_start + INFER_BATCH_SIZE]
                patches = np.stack(
                    [slc_stack[:, y:y+p, x:x+p] for y, x in batch_corners], axis=0
                )  # (B, N, P, P)
                x_t = torch.from_numpy(patches).float().to(device)
                preds = model(x_t).squeeze(1).cpu().numpy()  # (B, P, P)

                for (y, x), pred in zip(batch_corners, preds):
                    prob_sum[y:y+p, x:x+p] += pred
                    count[y:y+p, x:x+p]    += 1.0

            prob = prob_sum / np.maximum(count, 1e-6)
            pred_mask = (prob >= DICE_THRESHOLD).astype(np.uint8)
            mask_vol[i] = pred_mask * shape_mask  # belt-and-suspenders on the output too

            if (i + 1) % 50 == 0 or (i + 1) == n:
                print(f"      Inferred {i+1}/{n} slices ...", end="\r")
    print()
    if n_masked_slices:
        print(f"      [Note] {n_masked_slices} slice(s) had no detectable sample region "
              f"-- inference skipped, left as background.")

    return mask_vol


def save_predictions(sample_name: str, mask_vol: np.ndarray) -> None:
    out_dir = PRED_DIR / sample_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(mask_vol.shape[0]):
        tiff.imwrite(out_dir / f"pred_{i:04d}.tif", mask_vol[i])
    np.save(PRED_DIR / f"{sample_name}_mask.npy", mask_vol)
    print(f"      Saved {mask_vol.shape[0]} slice masks -> {out_dir}")
    print(f"      Saved stacked mask -> {PRED_DIR / f'{sample_name}_mask.npy'}")


def dark_ax(ax):
    ax.set_facecolor("#0d0d0d")
    ax.tick_params(colors="white")
    if hasattr(ax, "spines"):
        for spine in ax.spines.values():
            spine.set_color("#333333")


def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"      [Saved] {path.name}")


def _cap_mesh_triangles(f: np.ndarray, max_triangles: int, label: str) -> np.ndarray:
    """
    Randomly subsample a marching_cubes face array down to max_triangles.

    Voxel-count downsampling before marching_cubes controls how much work
    marching_cubes itself does, but triangle count depends on surface
    complexity, not input voxel count — a highly convoluted/porous shape
    (e.g. ~50% of a volume flagged as "defect") can still produce millions
    of triangles even from a heavily downsampled voxel grid. matplotlib's
    Poly3DCollection carries substantial per-triangle overhead in its own
    3D depth-sorting step, so a mesh that marching_cubes builds without
    trouble can still raise MemoryError purely from matplotlib trying to
    render it. This caps the actual triangle count actually handed to
    plot_trisurf, independent of how it got that large — a visualisation
    only needs a representative subset of faces, not every single one.
    """
    if len(f) <= max_triangles:
        return f
    print(f"      [Note] {label} mesh has {len(f):,} triangles — "
          f"subsampling to {max_triangles:,} for rendering.")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(f), size=max_triangles, replace=False)
    return f[idx]


def visualize_3d(sample_name: str, volume: np.ndarray, mask_vol: np.ndarray) -> None:
    """Marching-cubes mesh + ghost shell, scatter, orthographic MIPs, slice mosaic."""
    defect_mask = mask_vol.astype(bool)
    n_slices = volume.shape[0]

    # Hard cap on triangles handed to matplotlib, regardless of how a mesh
    # got that large (see _cap_mesh_triangles docstring) — applied to both
    # the defect mesh and the solid ghost-shell mesh below.
    MAX_TRIANGLES = 300_000

    # ── 1. Solid mesh + ghost shell ─────────────────────────────────────
    fig = plt.figure(figsize=(12, 10), facecolor="#0d0d0d")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0d0d0d")
    try:
        if np.any(defect_mask):
            # Choose a downsample stride from the ACTUAL defect voxel count,
            # not a fixed 2x: a fixed factor is only safe for a "normal" low
            # defect fraction. A sample where the detector over-predicts
            # heavily (e.g. reproducing a known Bernsen-labelling artifact —
            # see config.SUSPICIOUS_DEFECT_FRACTION) can have 30-50%+ of the
            # whole volume flagged as defect; even a 2x downsample still
            # leaves tens of millions of voxels, which produces a triangle
            # mesh large enough that matplotlib's 3D Z-sort raises
            # MemoryError during rendering, even though marching_cubes
            # itself completes fine. Doubling the stride keeps the
            # downsampled defect-voxel count under DEFECT_MESH_VOXEL_BUDGET
            # regardless of how much of the volume is flagged, capped at
            # MAX_STRIDE so the mesh never silently degrades to nothing.
            DEFECT_MESH_VOXEL_BUDGET = 4_000_000
            MAX_STRIDE = 16
            n_defect_voxels = int(defect_mask.sum())
            stride = 2
            while (n_defect_voxels / stride**3) > DEFECT_MESH_VOXEL_BUDGET and stride < MAX_STRIDE:
                stride *= 2
            if stride > 2:
                print(f"      [Note] {100*defect_mask.mean():.1f}% of volume flagged as "
                      f"defect — using {stride}x downsample for the defect mesh "
                      f"(instead of the usual 2x) to keep it renderable.")

            # Downsample BEFORE smoothing, not after: gaussian_filter on the
            # full-resolution volume still allocates a full-size float32
            # array (and scipy's own working buffers) even if the mesh
            # itself is later built from a downsampled slice of it — for a
            # ~1470-slice volume that's several GB just for this one array,
            # which is what was actually driving system memory to
            # exhaustion (not the mesh/render step downsampling alone fixed).
            downsampled_mask = defect_mask[::stride, ::stride, ::stride]
            smoothed = gaussian_filter(downsampled_mask.astype(np.float32), sigma=0.8)
            # Scale vertices back up by `stride` to keep the same spatial
            # units as the (undownsampled) volume axes.
            v, f, _, _ = marching_cubes(smoothed, level=0.5)
            f = _cap_mesh_triangles(f, MAX_TRIANGLES, "defect")
            ax.plot_trisurf(v[:, 0]*stride, v[:, 1]*stride, v[:, 2]*stride, triangles=f,
                             color="#ff4444", alpha=0.9, linewidth=0, antialiased=True)

        # Same fix as the defect mesh above: downsample the volume BEFORE
        # smoothing, not after, so gaussian_filter never has to allocate a
        # full-resolution float32 copy of the whole volume. Otsu's
        # threshold is computed on this same downsampled+smoothed volume —
        # a representative 2x-downsampled sample gives an equivalent
        # threshold for this purpose (a coarse "ghost shell" outline, not a
        # precision measurement).
        smoothed_solid = gaussian_filter(volume[::2, ::2, ::2], sigma=0.8)
        vs, fs, _, _ = marching_cubes(
            smoothed_solid, level=threshold_otsu(smoothed_solid)
        )
        fs = _cap_mesh_triangles(fs, MAX_TRIANGLES, "solid")
        ax.plot_trisurf(vs[:, 0]*2, vs[:, 1]*2, vs[:, 2]*2, triangles=fs,
                         color="#cccccc", alpha=0.08, linewidth=0)
    except Exception as e:
        print(f"      [Warning] mesh generation failed: {e}")

    ax.set_title(f"U-Net Predicted Defects — {sample_name} ({n_slices} slices)",
                 color="white", pad=20)
    dark_ax(ax)
    save_fig(fig, FIGURE_DIR / f"{sample_name}_unet_3d_mesh.png")

    # ── 2. Defect scatter ────────────────────────────────────────────────
    # np.where(defect_mask) materialises one int64 coordinate per True
    # voxel BEFORE any subsampling -- fine for real (sparse) pore volumes,
    # but a badly-generalising model can flag a large fraction of the
    # whole volume as "defect" (seen on out-of-distribution samples),
    # which then tries to allocate tens of GB just for those coordinates.
    # Stride the mask down first so the candidate-point count stays
    # bounded regardless of how much of the volume is flagged, then
    # randomly subsample as before -- this is a scatter visualisation,
    # not a precision measurement, so coarse striding is fine.
    n_defect = int(defect_mask.sum())
    stride = 1
    while n_defect / (stride ** 3) > 2_000_000 and stride < 16:
        stride *= 2
    if stride > 1:
        print(f"      [Note] {n_defect:,} defect voxels — striding by {stride}x "
              f"before building the scatter plot to keep it renderable.")
    strided_mask = defect_mask[::stride, ::stride, ::stride]
    pz, py, px = np.where(strided_mask)
    pz, py, px = pz * stride, py * stride, px * stride
    if len(pz) > 0:
        if len(pz) > 20_000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(pz), 20_000, replace=False)
            pz, py, px = pz[idx], py[idx], px[idx]
        fig_sc = plt.figure(figsize=(9, 7), facecolor="#0d0d0d")
        ax_sc = fig_sc.add_subplot(111, projection="3d")
        ax_sc.set_facecolor("#0d0d0d")
        sc = ax_sc.scatter(px, py, pz, c=pz, cmap="Reds", s=1, alpha=0.6)
        plt.colorbar(sc, ax=ax_sc, label="Slice depth (Z)", shrink=0.5, pad=0.1)
        ax_sc.set_title(f"U-Net Defect Scatter — {sample_name} ({len(pz):,} pts)",
                        color="white")
        dark_ax(ax_sc)
        save_fig(fig_sc, FIGURE_DIR / f"{sample_name}_unet_defect_scatter.png")

    # ── 3. Orthographic MIPs ─────────────────────────────────────────────
    fig_op, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor="#0d0d0d")
    views   = [volume.max(axis=0), volume.max(axis=1), volume.max(axis=2)]
    d_views = [defect_mask.sum(axis=0), defect_mask.sum(axis=1), defect_mask.sum(axis=2)]
    labels  = ["XY (Top)", "XZ (Front)", "YZ (Side)"]
    for col in range(3):
        axes[0, col].imshow(views[col], cmap="gray")
        axes[0, col].set_title(f"{labels[col]} — MIP", color="white", fontsize=10)
        axes[0, col].axis("off")
        axes[1, col].imshow(d_views[col], cmap="hot")
        axes[1, col].set_title(f"{labels[col]} — U-Net Defect Density",
                               color="white", fontsize=10)
        axes[1, col].axis("off")
    plt.tight_layout()
    save_fig(fig_op, FIGURE_DIR / f"{sample_name}_unet_orthographic_mip.png")

    # ── 4. Slice mosaic (up to 20 evenly spaced) ────────────────────────
    step = max(1, n_slices // 20)
    shown = list(range(0, n_slices, step))[:20]
    n_cols = min(5, len(shown))
    n_rows = (len(shown) + n_cols - 1) // n_cols
    fig_mo, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows),
                                 facecolor="#0d0d0d")
    axes = np.array(axes).reshape(n_rows, n_cols)
    for i, sl_idx in enumerate(shown):
        row, col = divmod(i, n_cols)
        axes[row, col].imshow(volume[sl_idx], cmap="gray", vmin=0, vmax=1)
        axes[row, col].imshow(defect_mask[sl_idx], cmap="Reds", alpha=0.4)
        axes[row, col].set_title(f"Slice {sl_idx}", color="#aaaaaa", fontsize=8)
        axes[row, col].axis("off")
    for i in range(len(shown), n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].axis("off")
    plt.tight_layout()
    save_fig(fig_mo, FIGURE_DIR / f"{sample_name}_unet_slice_mosaic.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", nargs="*", help="Sample names to run (default: samples not used in training)")
    parser.add_argument("--all", action="store_true", help="Run every discovered sample")
    parser.add_argument("--checkpoint", default=None,
                         help=f"Path to a .pt checkpoint "
                              f"(default: artifacts/best_model_{EXPERIMENT_NAME}.pt)")
    args = parser.parse_args()

    create_dirs()
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        sample_names = config.SAMPLE_NAMES
    elif args.samples:
        sample_names = args.samples
    else:
        trained_on = set(select_training_samples())
        held_out = [n for n in config.SAMPLE_NAMES if n not in trained_on]
        sample_names = held_out or config.SAMPLE_NAMES
        print(f"No samples given — defaulting to samples not used in "
              f"experiment '{EXPERIMENT_NAME}': {sample_names}")

    ckpt_path = Path(args.checkpoint) if args.checkpoint \
        else config.CKPT_DIR / f"best_model_{EXPERIMENT_NAME}.pt"
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        print("  Run pipeline.py first to train and save a model "
              f"(experiment '{EXPERIMENT_NAME}'), or pass --checkpoint "
              "to point at a different one.")
        return

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Checkpoint: {ckpt_path}")

    model = get_model()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
          f"val_dice={ckpt.get('val_dice', float('nan')):.4f}\n")

    for name in sample_names:
        print("=" * 65)
        print(f"SAMPLE: {name}")
        print("=" * 65)
        try:
            volume, _ = load_cache(name, mask_method="bernsen")
            volume = np.asarray(volume, dtype=np.float32)
        except Exception as e:
            print(f"  [ERROR] Could not load cache for '{name}': {e}")
            continue

        print(f"  [Inference] Running U-Net over {volume.shape[0]} slices ...")
        mask_vol = predict_volume(model, volume, device)
        defect_frac = mask_vol.mean() * 100
        print(f"  [Inference] Done — predicted defect fraction: {defect_frac:.3f}%")

        save_predictions(name, mask_vol)

        print(f"  [3D] Building visualisations ...")
        visualize_3d(name, volume, mask_vol)
        print()

    print("=" * 65)
    print("ALL SAMPLES COMPLETE")
    print(f"  Predictions -> {PRED_DIR}")
    print(f"  Figures     -> {FIGURE_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
