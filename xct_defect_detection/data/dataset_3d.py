# dataset_3d.py
import numpy as np
import torch
from torch.utils.data import Dataset
from data.augmentation import apply_augmentation


class XCTPatchDataset3D(Dataset):
    def __init__(
        self,
        volumes,
        masks,
        patch_size=(16, 128, 128),
        augment=False
    ):
        self.volumes = volumes
        self.masks = masks
        self.ps = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.volumes) * 200   # sampling factor

    def __getitem__(self, idx):
        v = self.volumes[idx % len(self.volumes)]
        m = self.masks[idx % len(self.masks)]

        D, H, W = v.shape
        pd, ph, pw = self.ps

        z = np.random.randint(0, D - pd)
        y = np.random.randint(0, H - ph)
        x = np.random.randint(0, W - pw)

        vp = v[z:z+pd, y:y+ph, x:x+pw]
        mp = m[z:z+pd, y:y+ph, x:x+pw]

        # ✅ slice-wise augmentation
        if self.augment:
            for i in range(pd):
                vp[i], mp[i] = apply_augmentation(vp[i], mp[i])

        vp = torch.from_numpy(vp).unsqueeze(0).float()  # (1, D, H, W)
        mp = torch.from_numpy(mp).unsqueeze(0).float()

        return vp, mp
