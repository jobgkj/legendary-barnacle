"""
=============================================================================
generate_3d_defects.py — Full Volume 3D Defect Detection (All Samples)
=============================================================================

Processes the complete TIFF stack for every sample in data/raw/:
  1. Streams all slices (constant memory per slice)
  2. Preprocesses each slice (median + NLM)
  3. Detects circular sample mask (removes air background)
  4. Applies Bernsen thresholding with auto-DCT (Kim et al. 2017)
  5. Builds a 3D defect volume from all binary masks
  6. Generates interactive Plotly HTML outputs:
       - Defect scatter     (pore voxels coloured by depth)
       - Isosurface overlay (translucent sample surface + pores)
       - Porosity profile   (slice-wise porosity along Z)

Outputs saved to: results/figures/<sample_name>/3d/

Run from project root:
    python scripts/generate_3d_defects.py
=============================================================================
"""
import sys
from pathlib import Path

# Add project root to path so config.py can be found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
import time
import numpy as np
import tifffile as tiff

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu as sk_otsu

import config
from config import create_dirs
from src.preprocess import preprocess_slice
from src.thresholding import bernsen
from src.sample_mask import detect_sample_mask_stack, build_circle_mask


# =============================================================================
# Settings
# =============================================================================

MAX_POINTS  = 80_000   # max scatter points per plot (browser performance)
RANDOM_SEED = 42
DOWNSAMPLE  = 2        # volume downsample for isosurface (1 = full res)


# =============================================================================
# Full volume streaming — Bernsen only
# =============================================================================

def build_defect_volume(sample_name: str) -> tuple:
    """
    Stream all slices, preprocess, apply sample mask and Bernsen
    thresholding, accumulate into a 3D binary defect volume.

    Parameters
    ----------
    sample_name : str

    Returns
    -------
    tuple
        defect_volume : np.ndarray (Z, H, W) uint8  — 1=pore, 0=solid
        prep_volume   : np.ndarray (Z, H, W) float32 — normalised [0,1]
        porosity      : float — 3D porosity as percentage
        n_slices      : int   — number of slices processed
    """
    raw_dir    = config.RAW_DATA_DIR / sample_name
    tiff_files = sorted(
        list(raw_dir.glob("*.tif")) + list(raw_dir.glob("*.tiff"))
    )
    if not tiff_files:
        raise ValueError(f"No TIFF files in {raw_dir}")

    total = len(tiff_files)
    print(f"    Found {total} slices")

    # ------------------------------------------------------------------
    # Detect circular sample mask once for the whole stack
    # ------------------------------------------------------------------
    print("    Detecting sample boundary ...")
    try:
        cx, cy, radius, _ = detect_sample_mask_stack(
            tiff_files, n_sample_slices=7
        )
        print(
            f"    Circle: centre=({cx:.1f}, {cy:.1f}), "
            f"radius={radius:.1f}px"
        )
        has_mask = True
    except Exception as e:
        warnings.warn(f"    Sample mask detection failed: {e} — no masking.")
        cx = cy = radius = None
        has_mask = False

    # ------------------------------------------------------------------
    # Stream slices
    # ------------------------------------------------------------------
    defect_slices = []
    prep_slices   = []
    skipped       = 0

    for idx, f in enumerate(tiff_files):
        print(f"    Slice {idx + 1}/{total}", end="\r")

        # Load
        try:
            raw = tiff.imread(f).astype(np.float32)
        except Exception as e:
            warnings.warn(f"Skipping {f.name}: {e}")
            skipped += 1
            continue

        if raw.ndim != 2:
            skipped += 1
            continue

        # Preprocess
        try:
            prep = preprocess_slice(raw)
        except Exception as e:
            warnings.warn(f"Preprocessing failed for {f.name}: {e}")
            skipped += 1
            continue

        # Build sample mask for this slice
        sample_mask = None
        if has_mask:
            h, w = prep.shape
            sample_mask = build_circle_mask(
                h, w,
                cx=cx, cy=cy,
                radius=radius,
                erosion_radius=config.SAMPLE_MASK_EROSION_RADIUS,
            )

        # Bernsen with auto-DCT
        try:
            mask = bernsen(prep, sample_mask=sample_mask)
        except Exception as e:
            warnings.warn(f"Thresholding failed for {f.name}: {e}")
            skipped += 1
            continue

        defect_slices.append(mask)
        prep_slices.append(prep.astype(np.float32) / 255.0)

    if not defect_slices:
        raise ValueError(f"No valid slices processed for {sample_name}.")

    processed     = total - skipped
    defect_volume = np.stack(defect_slices, axis=0)
    prep_volume   = np.stack(prep_slices,   axis=0)

    pore_vox   = int(defect_volume.sum())
    total_vox  = int(defect_volume.size)
    porosity   = pore_vox / total_vox * 100

    print(
        f"\n    Done — {processed}/{total} slices  |  "
        f"3D porosity: {porosity:.3f}%  |  "
        f"{pore_vox:,} pore voxels"
    )

    return defect_volume, prep_volume, porosity, processed


# =============================================================================
# Plotly — Defect scatter
# =============================================================================

def plotly_defect_scatter(
    defect_volume: np.ndarray,
    sample_name:   str,
    porosity:      float,
    out_dir,
):
    """
    Interactive 3D scatter of pore voxel locations.
    Coloured by Z depth (slice index).
    """
    pz, py, px = np.where(defect_volume == 1)

    if len(pz) == 0:
        print(f"    [scatter] No pores detected in {sample_name} — skipping.")
        return

    # Subsample for browser performance
    if len(pz) > MAX_POINTS:
        np.random.seed(RANDOM_SEED)
        idx        = np.random.choice(len(pz), MAX_POINTS, replace=False)
        pz, py, px = pz[idx], py[idx], px[idx]

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=px.tolist(),
        y=py.tolist(),
        z=pz.tolist(),
        mode="markers",
        marker=dict(
            size=1.5,
            color=pz.tolist(),
            colorscale="Reds",
            opacity=0.7,
            colorbar=dict(
                title="Slice (Z)",
                thickness=15,
                titlefont=dict(color="white"),
                tickfont=dict(color="white"),
            ),
        ),
        name="Pore voxels",
    ))

    fig.update_layout(
        title=dict(
            text=(
                f"3D Defect Map — {sample_name}<br>"
                f"<sup>Bernsen (auto-DCT)  |  "
                f"3D Porosity: {porosity:.3f}%  |  "
                f"{len(pz):,} voxels shown</sup>"
            ),
            font=dict(color="white"),
        ),
        scene=dict(
            bgcolor="black",
            xaxis=dict(showgrid=False, zeroline=False,
                       title="X", titlefont=dict(color="white"),
                       tickfont=dict(color="white")),
            yaxis=dict(showgrid=False, zeroline=False,
                       title="Y", titlefont=dict(color="white"),
                       tickfont=dict(color="white")),
            zaxis=dict(showgrid=False, zeroline=False,
                       title="Slice (Z)", titlefont=dict(color="white"),
                       tickfont=dict(color="white")),
        ),
        paper_bgcolor="black",
        font=dict(color="white"),
        legend=dict(bgcolor="black", font=dict(color="white")),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    out_path = out_dir / f"{sample_name}_defect_scatter.html"
    fig.write_html(out_path)
    print(f"    [Saved] {out_path.name}")


# =============================================================================
# Plotly — Isosurface + pore scatter
# =============================================================================

def plotly_isosurface(
    prep_volume:   np.ndarray,
    defect_volume: np.ndarray,
    sample_name:   str,
    porosity:      float,
    out_dir,
):
    """
    Translucent sample surface (isosurface) with pore scatter overlay.
    Downsampled for browser performance.
    """
    s      = slice(None, None, DOWNSAMPLE)
    vol_ds = gaussian_filter(prep_volume[s, s, s].astype(np.float32),
                             sigma=0.5)
    def_ds = defect_volume[s, s, s]

    # Build coordinate grids
    zi, yi, xi = np.mgrid[
        0:vol_ds.shape[0],
        0:vol_ds.shape[1],
        0:vol_ds.shape[2],
    ]

    thresh = float(sk_otsu(vol_ds))

    # Pore scatter (downsampled coords scaled back to original space)
    pz, py, px = np.where(def_ds == 1)
    if len(pz) > MAX_POINTS:
        np.random.seed(RANDOM_SEED)
        idx        = np.random.choice(len(pz), MAX_POINTS, replace=False)
        pz, py, px = pz[idx], py[idx], px[idx]

    # Scale back to original voxel coordinates for consistent axes
    pz_orig = pz * DOWNSAMPLE
    py_orig = py * DOWNSAMPLE
    px_orig = px * DOWNSAMPLE

    fig = go.Figure()

    # Translucent sample surface
    fig.add_trace(go.Isosurface(
        x=xi.flatten().tolist(),
        y=yi.flatten().tolist(),
        z=zi.flatten().tolist(),
        value=vol_ds.flatten().tolist(),
        isomin=float(thresh * 0.95),
        isomax=float(thresh * 1.05),
        surface_count=1,
        opacity=0.12,
        colorscale="Greys",
        caps=dict(x_show=False, y_show=False, z_show=False),
        showscale=False,
        name="Sample surface",
    ))

    # Pore scatter
    if len(pz) > 0:
        fig.add_trace(go.Scatter3d(
            x=px_orig.tolist(),
            y=py_orig.tolist(),
            z=pz_orig.tolist(),
            mode="markers",
            marker=dict(
                size=2,
                color=pz_orig.tolist(),
                colorscale="Reds",
                opacity=0.8,
                colorbar=dict(
                    title="Slice (Z)",
                    thickness=15,
                    titlefont=dict(color="white"),
                    tickfont=dict(color="white"),
                ),
            ),
            name="Pores",
        ))

    fig.update_layout(
        title=dict(
            text=(
                f"3D Sample + Defects — {sample_name}<br>"
                f"<sup>Bernsen (auto-DCT)  |  "
                f"Porosity: {porosity:.3f}%</sup>"
            ),
            font=dict(color="white"),
        ),
        scene=dict(
            bgcolor="black",
            xaxis=dict(showgrid=False, zeroline=False,
                       title="X", titlefont=dict(color="white"),
                       tickfont=dict(color="white")),
            yaxis=dict(showgrid=False, zeroline=False,
                       title="Y", titlefont=dict(color="white"),
                       tickfont=dict(color="white")),
            zaxis=dict(showgrid=False, zeroline=False,
                       title="Z", titlefont=dict(color="white"),
                       tickfont=dict(color="white")),
        ),
        paper_bgcolor="black",
        font=dict(color="white"),
        legend=dict(bgcolor="black", font=dict(color="white")),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    out_path = out_dir / f"{sample_name}_isosurface.html"
    fig.write_html(out_path)
    print(f"    [Saved] {out_path.name}")


# =============================================================================
# Plotly — Porosity profile along Z
# =============================================================================

def plotly_porosity_profile(
    defect_volume: np.ndarray,
    sample_name:   str,
    porosity:      float,
    out_dir,
):
    """
    Slice-wise porosity profile along the build (Z) direction.
    Matches Kim et al. (2017) Fig. 17–20.
    """
    n_slices = defect_volume.shape[0]
    por_per_slice = [
        float(defect_volume[z].sum()) / defect_volume[z].size * 100
        for z in range(n_slices)
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(n_slices)),
        y=por_per_slice,
        mode="lines",
        line=dict(color="#e05c5c", width=1.5),
        name="Slice porosity",
    ))

    fig.add_hline(
        y=porosity,
        line=dict(color="white", dash="dash", width=1.5),
        annotation_text=f"Global: {porosity:.3f}%",
        annotation_font_color="white",
    )

    fig.update_layout(
        title=dict(
            text=(
                f"Porosity Profile — Build Direction (Z) — {sample_name}<br>"
                f"<sup>Bernsen (auto-DCT)  |  "
                f"Global 3D porosity: {porosity:.3f}%</sup>"
            ),
            font=dict(color="white"),
        ),
        xaxis=dict(
            title="Slice Index (Z)",
            titlefont=dict(color="white"),
            tickfont=dict(color="white"),
            gridcolor="#333333",
        ),
        yaxis=dict(
            title="Porosity (%)",
            titlefont=dict(color="white"),
            tickfont=dict(color="white"),
            gridcolor="#333333",
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        legend=dict(bgcolor="black", font=dict(color="white")),
        margin=dict(l=60, r=20, t=80, b=60),
    )

    out_path = out_dir / f"{sample_name}_porosity_profile.html"
    fig.write_html(out_path)
    print(f"    [Saved] {out_path.name}")


# =============================================================================
# Plotly — All samples porosity summary
# =============================================================================

def plotly_summary(all_results: list, out_dir):
    """
    Bar chart comparing 3D porosity across all samples.
    """
    samples    = [r["sample_name"] for r in all_results]
    porosities = [r["porosity"]    for r in all_results]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=samples,
        y=porosities,
        marker=dict(
            color=porosities,
            colorscale="Reds",
            showscale=True,
            colorbar=dict(
                title="Porosity (%)",
                titlefont=dict(color="white"),
                tickfont=dict(color="white"),
            ),
        ),
        text=[f"{p:.3f}%" for p in porosities],
        textposition="outside",
        textfont=dict(color="white"),
        name="3D Porosity",
    ))

    fig.update_layout(
        title=dict(
            text=(
                "3D Porosity Summary — All Samples<br>"
                "<sup>Bernsen (auto-DCT, Kim et al. 2017)</sup>"
            ),
            font=dict(color="white"),
        ),
        xaxis=dict(
            title="Sample",
            titlefont=dict(color="white"),
            tickfont=dict(color="white"),
            gridcolor="#333333",
        ),
        yaxis=dict(
            title="3D Porosity (%)",
            titlefont=dict(color="white"),
            tickfont=dict(color="white"),
            gridcolor="#333333",
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        margin=dict(l=60, r=20, t=80, b=60),
    )

    out_path = out_dir / "all_samples_3d_porosity_summary.html"
    fig.write_html(out_path)
    print(f"\n  [Summary] Saved → {out_path.name}")


# =============================================================================
# Main
# =============================================================================

def main():
    create_dirs()

    print()
    print("=" * 65)
    print("  3D Defect Volume Generator — All Samples")
    print("  Method: Bernsen (auto-DCT, Kim et al. 2017)")
    print("=" * 65)

    # Discover all samples
    sample_dirs = sorted(
        d for d in config.RAW_DATA_DIR.iterdir() if d.is_dir()
    )

    if not sample_dirs:
        print(f"No samples found in {config.RAW_DATA_DIR}")
        return

    print(f"\n  Found {len(sample_dirs)} sample(s): "
          f"{[d.name for d in sample_dirs]}\n")

    all_results  = []
    total_start  = time.time()
    summary_dir  = config.FIGURES_DIR / "3d_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    for sample_dir in sample_dirs:
        sample_name = sample_dir.name
        out_dir     = config.FIGURES_DIR / sample_name / "3d"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*65}")
        print(f"  [{sample_name}]")
        print(f"{'='*65}")

        t0 = time.time()

        try:
            # Build full defect volume
            defect_volume, prep_volume, porosity, n_slices = \
                build_defect_volume(sample_name)

            # Generate Plotly outputs
            print(f"\n  Generating Plotly outputs ...")
            plotly_defect_scatter(
                defect_volume, sample_name, porosity, out_dir
            )
            plotly_isosurface(
                prep_volume, defect_volume, sample_name, porosity, out_dir
            )
            plotly_porosity_profile(
                defect_volume, sample_name, porosity, out_dir
            )

            all_results.append({
                "sample_name": sample_name,
                "porosity":    porosity,
                "n_slices":    n_slices,
            })

            print(
                f"\n  Done in {time.time()-t0:.1f}s  |  "
                f"Porosity: {porosity:.3f}%"
            )

        except Exception as e:
            print(f"\n  [ERROR] {sample_name} failed: {e}")
            continue

    # ------------------------------------------------------------------
    # Cross-sample summary
    # ------------------------------------------------------------------
    if len(all_results) > 1:
        print(f"\n{'='*65}")
        print("  Generating cross-sample summary ...")
        plotly_summary(all_results, summary_dir)

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    elapsed = time.time() - total_start

    print()
    print("=" * 65)
    print(f"  ALL SAMPLES COMPLETE — {elapsed:.1f}s total")
    print("=" * 65)
    print()
    print(f"  {'Sample':<20} {'Porosity':>12} {'Slices':>10}")
    print(f"  {'-'*44}")
    for r in all_results:
        print(
            f"  {r['sample_name']:<20} "
            f"{r['porosity']:>11.3f}%  "
            f"{r['n_slices']:>8}"
        )
    print()
    print(f"  Output: {config.FIGURES_DIR}")
    print()


if __name__ == "__main__":
    main()
