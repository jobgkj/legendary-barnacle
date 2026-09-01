"""
export_metrics.py — Run pipeline and export metrics to CSV
"""
import sys
from pathlib import Path

# Add project root to path so config.py can be found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathlib import Path
import csv
import tifffile as tiff

import config
from src.preprocess import preprocess_slice
from src.thresholding import otsu, yen, bernsen
from src.metrics import summarize_slice


def main():
    raw_dir = config.RAW_DATA_DIR / config.SAMPLE_NAME
    out_csv = config.METRICS_DIR / f"{config.SAMPLE_NAME}_metrics.csv"
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    files = sorted(list(raw_dir.glob("*.tif")) + list(raw_dir.glob("*.tiff")))

    for idx, f in enumerate(files):
        img = tiff.imread(f)
        img_pre = preprocess_slice(img)

        masks = {
            "otsu": otsu(img_pre),
            "yen": yen(img_pre),
            "bernsen": bernsen(
                img_pre,
                radius=config.BERNSEN_RADIUS,
                DCT=config.BERNSEN_DCT,
            ),
        }

        for method, mask in masks.items():
            stats = summarize_slice(mask)
            stats["sample"] = config.SAMPLE_NAME
            stats["slice"] = idx
            stats["method"] = method
            rows.append(stats)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Metrics saved to: {out_csv}")


if __name__ == "__main__":
    main()
