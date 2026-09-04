"""
=============================================================================
predict_slices_only.py — run one checkpoint's inference on a HANDFUL of
named slices only (not a full volume), for building small comparison
figures cheaply. Saves each slice's preprocessed image + prediction as
.npy under results/slice_compare/<EXPERIMENT_NAME>/<sample>_<index>.npz.

Run from repository root (set XCT_* env vars for the checkpoint/data you
want, same as predict_and_visualize_3d.py):
    python scripts/predict_slices_only.py sample_02:450 sample_07:737
=============================================================================
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

import config
from config import DEVICE, EXPERIMENT_NAME, PATCH_SIZE, PATCH_STRIDE, DICE_THRESHOLD, UNET_INPUT_SLICES
from data.cache import load_cache
from models.unet2d import get_model
from scripts.predict_and_visualize_3d import _patch_grid, _slice_shape_mask

OUT_DIR = config.REPO_ROOT / "results" / "slice_compare" / EXPERIMENT_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

ckpt_path = config.CKPT_DIR / f"best_model_{EXPERIMENT_NAME}.pt"
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
model = get_model()
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.to(device).eval()
print(f"Loaded '{EXPERIMENT_NAME}' (UNET_INPUT_SLICES={UNET_INPUT_SLICES}) "
      f"from epoch {ckpt.get('epoch', '?')}")

specs = sys.argv[1:] or ["sample_02:450", "sample_07:737"]

for spec in specs:
    sample_name, idx_str = spec.split(":")
    idx = int(idx_str)
    print(f"\n{sample_name} slice {idx} ...")

    volume, bernsen_mask = load_cache(sample_name, mask_method="bernsen")
    n = volume.shape[0]
    idx = min(idx, n - 1)
    h, w = volume.shape[1], volume.shape[2]

    half = UNET_INPUT_SLICES // 2
    neighbour_idxs = [int(np.clip(idx + off, 0, n - 1)) for off in range(-half, half + 1)]

    shape_mask = _slice_shape_mask(np.asarray(volume[idx], dtype=np.float32))
    slc_stack = np.stack(
        [np.asarray(volume[j], dtype=np.float32) * shape_mask for j in neighbour_idxs], axis=0
    )

    corners = _patch_grid(h, w, PATCH_SIZE, PATCH_STRIDE)
    prob_sum = np.zeros((h, w), dtype=np.float32)
    count    = np.zeros((h, w), dtype=np.float32)
    with torch.no_grad():
        for bstart in range(0, len(corners), 32):
            bc = corners[bstart:bstart + 32]
            patches = np.stack(
                [slc_stack[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE] for y, x in bc], axis=0
            )
            x_t = torch.from_numpy(patches).float().to(device)
            preds = model(x_t).squeeze(1).cpu().numpy()
            for (y, x), pred in zip(bc, preds):
                prob_sum[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += pred
                count[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1.0
    prob = prob_sum / np.maximum(count, 1e-6)
    pred_mask = ((prob >= DICE_THRESHOLD).astype(np.uint8)) * shape_mask.astype(np.uint8)

    out_path = OUT_DIR / f"{sample_name}_{idx}.npz"
    np.savez_compressed(
        out_path,
        preprocessed=np.asarray(volume[idx], dtype=np.float32),
        bernsen=np.asarray(bernsen_mask[idx], dtype=np.uint8),
        pred=pred_mask,
    )
    print(f"  defect fraction (pred): {pred_mask.mean()*100:.3f}%  -> {out_path}")

print(f"\nDone. Saved under {OUT_DIR}")
