import os
import glob
import numpy as np
import torch
import torch.nn as nn
import tifffile as tiff
import pyvista as pv
from skimage.measure import marching_cubes

# ===============================
# 3D U-NET (compact & stable)
# ===============================
class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv3D(1, 32)
        self.enc2 = DoubleConv3D(32, 64)
        self.enc3 = DoubleConv3D(64, 128)
        self.pool = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv3D(128, 256)

        self.up3 = nn.ConvTranspose3d(256, 128, 2, 2)
        self.dec3 = DoubleConv3D(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, 2, 2)
        self.dec2 = DoubleConv3D(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, 2, 2)
        self.dec1 = DoubleConv3D(64, 32)

        self.out = nn.Conv3d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)

# ===============================
# METRICS (VOXEL-LEVEL)
# ===============================
def dice_score(gt, pred, eps=1e-7):
    inter = (gt * pred).sum()
    return (2 * inter + eps) / (gt.sum() + pred.sum() + eps)

def iou_score(gt, pred, eps=1e-7):
    inter = (gt * pred).sum()
    union = gt.sum() + pred.sum() - inter
    return (inter + eps) / (union + eps)

# ===============================
# LOAD TIFF STACK
# ===============================
def load_tiff_stack(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.tif")))
    if len(files) == 0:
        raise RuntimeError(f"No TIFF files in {folder}")
    slices = [tiff.imread(f) for f in files]
    return np.stack(slices).astype(np.float32)

# ===============================
# 3D VISUALIZATION
# ===============================
def visualize_3d(gt, pred):
    overlap = gt & pred

    plotter = pv.Plotter()
    plotter.set_background("black")

    if gt.sum() > 0:
        v, f, _, _ = marching_cubes(gt, 0.5)
        plotter.add_mesh(
            pv.PolyData(v, f),
            color="green",
            opacity=0.35,
            label="Ground Truth"
        )

    if pred.sum() > 0:
        v, f, _, _ = marching_cubes(pred, 0.5)
        plotter.add_mesh(
            pv.PolyData(v, f),
            color="red",
            opacity=0.35,
            label="Prediction"
        )

    if overlap.sum() > 0:
        v, f, _, _ = marching_cubes(overlap, 0.5)
        plotter.add_mesh(
            pv.PolyData(v, f),
            color="yellow",
            opacity=0.9,
            label="Overlap"
        )

    plotter.add_axes()
    plotter.add_legend()
    plotter.show()

# ===============================
# CONFIG
# ===============================
VOLUME_DIR = "dataset/volume"
LABEL_DIR  = "dataset/label"
MODEL_PATH = "dataset/model.pth"
THRESHOLD  = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# LOAD DATA
# ===============================
volume = load_tiff_stack(VOLUME_DIR)
label  = load_tiff_stack(LABEL_DIR)

# Industrial CT normalization (robust to metal/artifacts)
p1, p99 = np.percentile(volume, [1, 99])
volume = np.clip((volume - p1) / (p99 - p1 + 1e-7), 0, 1)
label = (label > 0).astype(np.float32)

# To tensors (1, 1, D, H, W)
volume_t = torch.tensor(volume)[None, None].to(DEVICE)
label_t  = torch.tensor(label)[None, None].to(DEVICE)

# ===============================
# LOAD MODEL
# ===============================
model = UNet3D().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ===============================
# INFERENCE
# ===============================
with torch.no_grad():
    logits = model(volume_t)
    pred = torch.sigmoid(logits)
    pred = (pred > THRESHOLD).float()

# ===============================
# METRICS
# ===============================
dice = dice_score(label_t, pred).item()
iou  = iou_score(label_t, pred).item()

print("\n==============================")
print("INDUSTRIAL CT – SINGLE PART")
print("==============================")
print(f"Volume shape : {volume.shape}")
print(f"Dice score   : {dice:.4f}")
print(f"IoU score    : {iou:.4f}")

# ===============================
# 3D VIEW
# ===============================
gt_np = label_t.cpu().numpy()[0, 0]
pred_np = pred.cpu().numpy()[0, 0]
visualize_3d(gt_np, pred_np)
