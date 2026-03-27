import os
import glob
import numpy as np
import torch
import torch.nn as nn
import tifffile as tiff
import pyvista as pv
from skimage.measure import marching_cubes
from itertools import product

###############################################################################
#                              MODEL DEFINITIONS
###############################################################################

# -------- 3D U-NET --------
class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
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
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


# -------- 2D U-NET BASELINE --------
class DoubleConv2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv2D(1, 32)
        self.enc2 = DoubleConv2D(32, 64)
        self.enc3 = DoubleConv2D(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv2D(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = DoubleConv2D(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = DoubleConv2D(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = DoubleConv2D(64, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)

###############################################################################
#                                 METRICS
###############################################################################

def dice_score(gt, pred, eps=1e-7):
    inter = (gt * pred).sum()
    return (2 * inter + eps) / (gt.sum() + pred.sum() + eps)


def iou_score(gt, pred, eps=1e-7):
    inter = (gt * pred).sum()
    union = gt.sum() + pred.sum() - inter
    return (inter + eps) / (union + eps)

###############################################################################
#                               DATA LOADING
###############################################################################

def load_tiff_stack(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.tif")))
    if not files:
        raise RuntimeError(f"No TIFFs found in {folder}")
    return np.stack([tiff.imread(f) for f in files]).astype(np.float32)

###############################################################################
#                        3D PATCH-BASED INFERENCE
###############################################################################

def sliding_window_3d(model, volume, patch=(64,64,64), overlap=0.5, device="cpu"):
    ps = np.array(patch)
    stride = (ps * (1 - overlap)).astype(int)
    out = np.zeros(volume.shape, np.float32)
    cnt = np.zeros(volume.shape, np.float32)

    Z, Y, X = volume.shape
    zs = list(range(0, Z - ps[0] + 1, stride[0])) + [Z - ps[0]]
    ys = list(range(0, Y - ps[1] + 1, stride[1])) + [Y - ps[1]]
    xs = list(range(0, X - ps[2] + 1, stride[2])) + [X - ps[2]]

    with torch.no_grad():
        for z, y, x in product(zs, ys, xs):
            patch_v = volume[z:z+ps[0], y:y+ps[1], x:x+ps[2]]
            inp = torch.tensor(patch_v)[None,None].to(device)
            pred = torch.sigmoid(model(inp)).cpu().numpy()[0,0]
            out[z:z+ps[0], y:y+ps[1], x:x+ps[2]] += pred
            cnt[z:z+ps[0], y:y+ps[1], x:x+ps[2]] += 1

    return out / np.maximum(cnt, 1)

###############################################################################
#                         2D SLICE-WISE INFERENCE
###############################################################################

def inference_2d(model, volume, device="cpu"):
    D, H, W = volume.shape
    prob = np.zeros((D,H,W), np.float32)
    with torch.no_grad():
        for z in range(D):
            inp = torch.tensor(volume[z])[None,None].to(device)
            prob[z] = torch.sigmoid(model(inp)).cpu().numpy()[0,0]
    return prob

###############################################################################
#                               VISUALIZATION
###############################################################################

def visualize_3d(gt, pred):
    plotter = pv.Plotter()
    plotter.set_background("black")

    for arr, color, label, op in [
        (gt, "green", "GT", 0.4),
        (pred, "red", "Prediction", 0.4),
        (gt & pred, "yellow", "Overlap", 0.9)
    ]:
        if arr.sum() > 0:
            v,f,_,_ = marching_cubes(arr, 0.5)
            plotter.add_mesh(pv.PolyData(v,f), color=color, opacity=op, label=label)

    plotter.add_legend()
    plotter.show()

###############################################################################
#                                 MAIN
###############################################################################

if __name__ == "__main__":

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    THRESHOLD = 0.4
    PATCH = (64,64,64)

    VOL_DIR = "dataset/volume"
    LAB_DIR = "dataset/label"
    MODEL_3D = "dataset/model_3d.pth"
    MODEL_2D = "dataset/model_2d.pth"

    volume = load_tiff_stack(VOL_DIR)
    label = load_tiff_stack(LAB_DIR)

    # XCT normalization
    p1, p99 = np.percentile(volume, [1,99])
    volume = np.clip((volume - p1)/(p99-p1+1e-7), 0, 1)
    label = (label > 0).astype(np.float32)

    # Load models
    model3d = UNet3D().to(DEVICE)
    model2d = UNet2D().to(DEVICE)
    model3d.load_state_dict(torch.load(MODEL_3D, map_location=DEVICE))
    model2d.load_state_dict(torch.load(MODEL_2D, map_location=DEVICE))
    model3d.eval()
    model2d.eval()

    # Inference
    prob3d = sliding_window_3d(model3d, volume, PATCH, 0.5, DEVICE)
    prob2d = inference_2d(model2d, volume, DEVICE)

    pred3d = (prob3d > THRESHOLD).astype(np.float32)
    pred2d = (prob2d > THRESHOLD).astype(np.float32)

    # Metrics
    d2 = dice_score(torch.tensor(label), torch.tensor(pred2d)).item()
    i2 = iou_score(torch.tensor(label), torch.tensor(pred2d)).item()
    d3 = dice_score(torch.tensor(label), torch.tensor(pred3d)).item()
    i3 = iou_score(torch.tensor(label), torch.tensor(pred3d)).item()

    print("\n==============================")
    print("2D vs 3D U-NET COMPARISON")
    print("==============================")
    print(f"2D U-Net → Dice {d2:.4f}, IoU {i2:.4f}")
    print(f"3D U-Net → Dice {d3:.4f}, IoU {i3:.4f}")

    visualize_3d(label.astype(np.uint8), pred3d.astype(np.uint8))
