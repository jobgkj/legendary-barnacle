"""
3D Interactive XCT Volume Visualiser with Defect Highlighting
Run from project root:
    python visualize.py
"""

import os
import numpy as np
import tifffile as tiff
import glob
import plotly.graph_objects as go

# ── Config ─────────────────────────────────────────────────────────────────
VOLUME_DIR = r"data\tiff_output"       # preprocessed slices
MASK_DIR   = r"data\tiff_masks"        # pseudo-label masks (if available)
DOWNSAMPLE = 4                         # reduce resolution for speed (2, 4, or 8)

# ── Load volume ─────────────────────────────────────────────────────────────
def load_stack(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.tif")))
    return np.stack([tiff.imread(f) for f in files], axis=0)

print("Loading preprocessed volume ...")
volume = load_stack(VOLUME_DIR)
print(f"  Volume shape: {volume.shape}")

# Downsample for performance
volume = volume[::DOWNSAMPLE, ::DOWNSAMPLE, ::DOWNSAMPLE]
print(f"  Downsampled to: {volume.shape}")

# ── Load or generate mask ───────────────────────────────────────────────────
if os.path.isdir(MASK_DIR) and glob.glob(os.path.join(MASK_DIR, "*.tif")):
    print("Loading pseudo-label masks ...")
    mask = load_stack(MASK_DIR)
    mask = mask[::DOWNSAMPLE, ::DOWNSAMPLE, ::DOWNSAMPLE]
else:
    print("No masks found — generating Otsu threshold mask ...")
    from skimage.filters import threshold_otsu
    thresh = threshold_otsu(volume)
    mask   = (volume < thresh).astype(np.uint8)   # defects = dark voxels

# ── Get defect voxel coordinates ────────────────────────────────────────────
print("Extracting defect coordinates ...")
z, y, x = np.where(mask > 0)

# Subsample points for performance (max 50k points)
max_pts = 50_000
if len(z) > max_pts:
    idx = np.random.choice(len(z), max_pts, replace=False)
    z, y, x = z[idx], y[idx], x[idx]

print(f"  Defect voxels to render: {len(z):,}")

# ── Build figure ─────────────────────────────────────────────────────────────
print("Building 3D visualisation ...")

fig = go.Figure()

# Volume outline box
D, H, W = volume.shape
fig.add_trace(go.Scatter3d(
    x=[0, W, W, 0, 0, 0, W, W, 0, 0, W, W, W, W, 0, 0],
    y=[0, 0, H, H, 0, 0, 0, H, H, 0, 0, 0, H, H, H, H],
    z=[0, 0, 0, 0, 0, D, D, D, D, D, D, 0, 0, D, D, 0],
    mode="lines",
    line=dict(color="lightgrey", width=2),
    name="Volume boundary"
))

# Defect points
fig.add_trace(go.Scatter3d(
    x=x, y=y, z=z,
    mode="markers",
    marker=dict(
        size=1.5,
        color=z,                        # colour by depth
        colorscale="Reds",
        opacity=0.4,
        colorbar=dict(title="Depth (slice)")
    ),
    name="Defects"
))

fig.update_layout(
    title="XCT Volume — Defect Map (Interactive 3D)",
    scene=dict(
        xaxis_title="X (px)",
        yaxis_title="Y (px)",
        zaxis_title="Slice",
        bgcolor="black",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        zaxis=dict(showgrid=False),
    ),
    paper_bgcolor="black",
    font_color="white",
    legend=dict(bgcolor="black")
)

# ── Save and open ─────────────────────────────────────────────────────────────
out = "defect_map_3d.html"
fig.write_html(out)
print(f"\n  Saved → '{out}'")
print("  Open it in your browser to interact with the 3D model.")
