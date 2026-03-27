import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tifffile as tiff
from torch.utils.data import Dataset, DataLoader

########################################
# LOSS
########################################

class DiceBCELoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum()
        dice = (2 * inter + self.eps) / (probs.sum() + targets.sum() + self.eps)
        return self.bce(logits, targets) + (1 - dice)

########################################
# LOAD & NORMALIZE XCT
########################################

def load_volume(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.tif")))
    if not files:
        raise RuntimeError(f"No TIFF files found in {folder}")
    vol = np.stack([tiff.imread(f) for f in files]).astype(np.float32)
    p1, p99 = np.percentile(vol, [1, 99])
    vol = np.clip((vol - p1) / (p99 - p1 + 1e-7), 0, 1)
    return vol

########################################
# DATASETS
########################################

class XCTDataset2D(Dataset):
    """Slice-wise 2D dataset (baseline)"""
    def __init__(self, vol, lab):
        self.vol = vol
        self.lab = lab

    def __len__(self):
        return self.vol.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.vol[idx])[None]
        y = torch.tensor(self.lab[idx])[None]
        return x, y


class XCTDataset3D(Dataset):
    """Patch-based 3D dataset"""
    def __init__(self, vol, lab, patch=(64, 64, 64)):
        self.vol = vol
        self.lab = lab
        self.ps = patch
        self.shape = vol.shape

    def __len__(self):
        return 3000  # nominal patches

    def __getitem__(self, _):
        z = random.randint(0, self.shape[0] - self.ps[0])
        y = random.randint(0, self.shape[1] - self.ps[1])
        x = random.randint(0, self.shape[2] - self.ps[2])

        v = self.vol[z:z+self.ps[0], y:y+self.ps[1], x:x+self.ps[2]]
        l = self.lab[z:z+self.ps[0], y:y+self.ps[1], x:x+self.ps[2]]

        return torch.tensor(v)[None], torch.tensor(l)[None]

########################################
# MODELS (IMPORT OR PASTE)
########################################

from models import UNet2D, UNet3D
# OR paste the UNet2D / UNet3D definitions here if not modularized

########################################
# TRAINING LOOP
########################################

def train_model(model, loader, epochs, lr, device, save_path):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = DiceBCELoss()

    for ep in range(epochs):
        model.train()
        total_loss = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {ep+1:03d}/{epochs} | Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved → {save_path}")

########################################
# MAIN
########################################

if __name__ == "__main__":

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    VOL_DIR = "dataset/volume"
    LAB_DIR = "dataset/label"

    vol = load_volume(VOL_DIR)
    lab = (load_volume(LAB_DIR) > 0).astype(np.float32)

    # ========= TRAIN 2D =========
    ds2d = XCTDataset2D(vol, lab)
    dl2d = DataLoader(ds2d, batch_size=8, shuffle=True)

    model2d = UNet2D()
    train_model(
        model2d, dl2d,
        epochs=150,
        lr=1e-3,
        device=DEVICE,
        save_path="dataset/model_2d.pth"
    )

    # ========= TRAIN 3D =========
    ds3d = XCTDataset3D(vol, lab)
    dl3d = DataLoader(ds3d, batch_size=1, shuffle=True)

    model3d = UNet3D()
    train_model(
        model3d, dl3d,
        epochs=200,
        lr=1e-4,
        device=DEVICE,
        save_path="dataset/model_3d.pth"
    )
``
