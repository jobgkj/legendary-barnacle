"""
=============================================================================
gui_app.py — Desktop GUI: pick a TIFF folder, run U-Net inference, view in 3D
=============================================================================
Select a folder of XCT TIFF slices, preprocesses them with the same pipeline
used elsewhere in this project (src/preprocess.py), runs the trained 2.5D
U-Net over the volume slice-by-slice (same sliding-window logic as
scripts/predict_and_visualize_3d.py), reconstructs the predicted defect
volume into a 3D mesh (marching cubes), and displays it in an embedded,
mouse-controllable 3D viewer — drag to rotate, scroll to zoom, and the
toolbar's pan/zoom tools work too.

Run from project root:
    python gui_app.py
=============================================================================
"""

import os
import sys
import queue
import threading
import traceback
from pathlib import Path

# See pipeline.py for why this is needed: non-ASCII prints (banners, symbols)
# otherwise crash under a non-UTF-8 console/redirect encoding (e.g. Windows cp1252).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import tifffile as tiff
import torch

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter

import config
from config import PATCH_SIZE, PATCH_STRIDE, DICE_THRESHOLD, DEVICE, EXPERIMENT_NAME, UNET_INPUT_SLICES
from src.preprocess import preprocess_slice
from src.metrics import pore_properties
from models.unet2d import get_model

INFER_BATCH_SIZE = 32
DEFAULT_CKPT = config.CKPT_DIR / f"best_model_{EXPERIMENT_NAME}.pt"


# =============================================================================
# Inference helpers (same sliding-window approach as scripts/predict_and_visualize_3d.py)
# =============================================================================

def _patch_grid(h: int, w: int, p: int, s: int) -> list[tuple[int, int]]:
    ys = list(range(0, max(h - p, 0) + 1, s))
    xs = list(range(0, max(w - p, 0) + 1, s))
    if ys[-1] != h - p:
        ys.append(h - p)
    if xs[-1] != w - p:
        xs.append(w - p)
    return [(y, x) for y in ys for x in xs]


def load_and_preprocess_folder(folder: Path, progress_cb) -> np.ndarray:
    """Load every TIFF in `folder`, preprocess each slice, stack to a
    float32 [0, 1] volume (N, H, W). progress_cb(done, total) is called
    after each slice."""
    files = sorted(
        list(folder.glob("*.tif")) + list(folder.glob("*.tiff"))
    )
    if not files:
        raise FileNotFoundError(f"No .tif/.tiff files found in {folder}")

    first = preprocess_slice(tiff.imread(files[0]))
    h, w = first.shape
    volume = np.empty((len(files), h, w), dtype=np.float32)
    volume[0] = first.astype(np.float32) / 255.0

    for i, f in enumerate(files[1:], start=1):
        img = tiff.imread(f)
        pre = preprocess_slice(img)
        volume[i] = pre.astype(np.float32) / 255.0
        progress_cb(i + 1, len(files))

    progress_cb(len(files), len(files))
    return volume


def predict_volume(model, volume: np.ndarray, device: torch.device, progress_cb) -> np.ndarray:
    """Sliding-window U-Net inference over every slice, using a 2.5D stack
    of UNET_INPUT_SLICES adjacent slices as input per slice (matching
    training — see data/dataset.py::XCTPatchDataset._get_slice_stack), with
    neighbours past a volume edge clamped rather than zero-padded. Returns
    a uint8 binary mask volume the same shape as `volume`."""
    n, h, w = volume.shape
    p, s = PATCH_SIZE, PATCH_STRIDE
    corners = _patch_grid(h, w, p, s)
    half = UNET_INPUT_SLICES // 2
    mask_vol = np.zeros((n, h, w), dtype=np.uint8)

    with torch.no_grad():
        for i in range(n):
            neighbour_idxs = [
                int(np.clip(i + offset, 0, n - 1))
                for offset in range(-half, half + 1)
            ]
            slc_stack = np.stack(
                [np.asarray(volume[j], dtype=np.float32) for j in neighbour_idxs],
                axis=0
            )  # (N, H, W)

            prob_sum = np.zeros((h, w), dtype=np.float32)
            count = np.zeros((h, w), dtype=np.float32)

            for batch_start in range(0, len(corners), INFER_BATCH_SIZE):
                batch_corners = corners[batch_start:batch_start + INFER_BATCH_SIZE]
                patches = np.stack(
                    [slc_stack[:, y:y + p, x:x + p] for y, x in batch_corners], axis=0
                )  # (B, N, P, P)
                x_t = torch.from_numpy(patches).float().to(device)
                preds = model(x_t).squeeze(1).cpu().numpy()
                for (y, x), pred in zip(batch_corners, preds):
                    prob_sum[y:y + p, x:x + p] += pred
                    count[y:y + p, x:x + p] += 1.0

            prob = prob_sum / np.maximum(count, 1e-6)
            mask_vol[i] = (prob >= DICE_THRESHOLD).astype(np.uint8)
            progress_cb(i + 1, n)

    return mask_vol


def compute_defect_metrics(mask_vol: np.ndarray, progress_cb) -> dict:
    """Pool per-slice pore properties (src/metrics.py) across the whole
    predicted volume — same 2D connected-component approach used
    elsewhere in this project (e.g. scripts/compare_methods_report.py)."""
    n = mask_vol.shape[0]
    all_areas, all_diams = [], []

    for i in range(n):
        props = pore_properties(mask_vol[i])
        all_areas.append(props["areas"])
        all_diams.append(props["equivalent_diameters"])
        progress_cb(i + 1, n)

    areas = np.concatenate(all_areas) if all_areas else np.array([])
    diams = np.concatenate(all_diams) if all_diams else np.array([])

    return {
        "defect_fraction_pct": float(mask_vol.mean()) * 100,
        "pore_count": int(areas.size),
        "mean_pore_area_px": float(areas.mean()) if areas.size else 0.0,
        "mean_equiv_diameter_px": float(diams.mean()) if diams.size else 0.0,
    }


# =============================================================================
# GUI
# =============================================================================

class XCTViewerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("XCT Defect Viewer — U-Net 3D Reconstruction")
        self.root.geometry("1100x800")

        self.folder_var = tk.StringVar()
        self.ckpt_var = tk.StringVar(value=str(DEFAULT_CKPT))
        self.status_var = tk.StringVar(value="Select a folder of TIFF slices to begin.")
        self.metric_defect_frac = tk.StringVar(value="—")
        self.metric_pore_count = tk.StringVar(value="—")
        self.metric_mean_area = tk.StringVar(value="—")
        self.metric_mean_diam = tk.StringVar(value="—")

        self.msg_queue: queue.Queue = queue.Queue()
        self.worker_thread: threading.Thread | None = None

        self._build_widgets()
        self.root.after(100, self._poll_queue)

    # ------------------------------------------------------------------
    def _build_widgets(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="TIFF folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.folder_var, width=70).grid(row=0, column=1, padx=5)
        ttk.Button(top, text="Browse…", command=self._browse_folder).grid(row=0, column=2)

        ttk.Label(top, text="Checkpoint (.pt):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(top, textvariable=self.ckpt_var, width=70).grid(row=1, column=1, padx=5, pady=(6, 0))
        ttk.Button(top, text="Browse…", command=self._browse_ckpt).grid(row=1, column=2, pady=(6, 0))

        run_row = ttk.Frame(self.root, padding=(10, 0))
        run_row.pack(side=tk.TOP, fill=tk.X)
        self.run_btn = ttk.Button(run_row, text="Run Inference && Show 3D", command=self._on_run)
        self.run_btn.pack(side=tk.LEFT)
        self.export_btn = ttk.Button(run_row, text="Export Image…", command=self._export_image, state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=8)

        progress_row = ttk.Frame(self.root, padding=10)
        progress_row.pack(side=tk.TOP, fill=tk.X)
        self.progress = ttk.Progressbar(progress_row, mode="determinate", length=400)
        self.progress.pack(side=tk.LEFT)
        ttk.Label(progress_row, textvariable=self.status_var).pack(side=tk.LEFT, padx=10)

        # Metrics panel — populated after a run completes (defect fraction,
        # pore count, mean area, mean equivalent diameter — src/metrics.py,
        # pooled across every slice in the predicted volume).
        metrics_row = ttk.LabelFrame(self.root, text="Defect metrics (this volume)", padding=10)
        metrics_row.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 5))

        def _metric_cell(parent, col, label, var):
            cell = ttk.Frame(parent)
            cell.grid(row=0, column=col, padx=20, sticky="w")
            ttk.Label(cell, text=label, font=("TkDefaultFont", 9)).pack(anchor="w")
            ttk.Label(cell, textvariable=var, font=("TkDefaultFont", 13, "bold")).pack(anchor="w")

        _metric_cell(metrics_row, 0, "Defect fraction", self.metric_defect_frac)
        _metric_cell(metrics_row, 1, "Pore count", self.metric_pore_count)
        _metric_cell(metrics_row, 2, "Mean pore area (px²)", self.metric_mean_area)
        _metric_cell(metrics_row, 3, "Mean equiv. diameter (px)", self.metric_mean_diam)

        # Embedded 3D canvas — mouse-drag rotates, scroll-wheel zooms (native
        # Axes3D behaviour), and the toolbar below adds pan/zoom/save tools.
        fig_frame = ttk.Frame(self.root)
        fig_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = plt.figure(figsize=(8, 7), facecolor="#0d0d0d")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("#0d0d0d")
        self._style_axes("Select a folder and click Run to reconstruct defects in 3D")

        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, fig_frame)
        toolbar.update()

    def _style_axes(self, title: str):
        self.ax.set_facecolor("#0d0d0d")
        self.ax.tick_params(colors="white")
        for spine in getattr(self.ax, "spines", {}).values():
            spine.set_color("#333333")
        self.ax.set_title(title, color="white", pad=15)

    # ------------------------------------------------------------------
    def _browse_folder(self):
        path = filedialog.askdirectory(title="Select folder of TIFF slices")
        if path:
            self.folder_var.set(path)

    def _browse_ckpt(self):
        path = filedialog.askopenfilename(
            title="Select model checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.ckpt_var.set(path)

    # ------------------------------------------------------------------
    def _on_run(self):
        folder = self.folder_var.get().strip()
        ckpt = self.ckpt_var.get().strip()

        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Invalid folder", "Please select a valid folder of TIFF slices.")
            return
        if not ckpt or not os.path.isfile(ckpt):
            messagebox.showerror("Invalid checkpoint", f"Checkpoint not found:\n{ckpt}")
            return
        if self.worker_thread and self.worker_thread.is_alive():
            return  # already running

        self.run_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.DISABLED)
        self.progress.config(value=0)
        self.status_var.set("Starting…")

        self.worker_thread = threading.Thread(
            target=self._worker, args=(Path(folder), Path(ckpt)), daemon=True
        )
        self.worker_thread.start()

    def _worker(self, folder: Path, ckpt_path: Path):
        """Runs on a background thread — never touch Tkinter widgets directly
        here, only push messages onto self.msg_queue."""
        try:
            def preprocess_progress(done, total):
                self.msg_queue.put(("progress", done, total, f"Preprocessing slice {done}/{total}"))

            self.msg_queue.put(("status", "Loading and preprocessing TIFF slices…"))
            volume = load_and_preprocess_folder(folder, preprocess_progress)

            self.msg_queue.put(("status", f"Loading model checkpoint ({ckpt_path.name})…"))
            device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
            model = get_model()
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            model.to(device).eval()

            def infer_progress(done, total):
                self.msg_queue.put(("progress", done, total, f"Running U-Net inference {done}/{total}"))

            self.msg_queue.put(("status", f"Running inference on {device}…"))
            mask_vol = predict_volume(model, volume, device, infer_progress)

            if not np.any(mask_vol):
                self.msg_queue.put(("error", "No defects were detected in this volume — nothing to display."))
                return

            def metrics_progress(done, total):
                self.msg_queue.put(("progress", done, total, f"Computing pore metrics {done}/{total}"))

            self.msg_queue.put(("status", "Computing defect metrics…"))
            metrics = compute_defect_metrics(mask_vol, metrics_progress)
            self.msg_queue.put(("metrics", metrics))

            self.msg_queue.put(("status", f"Building 3D mesh (defect fraction {metrics['defect_fraction_pct']:.3f}%)…"))
            smoothed = gaussian_filter(mask_vol.astype(np.float32), sigma=0.8)
            verts, faces, _, _ = marching_cubes(smoothed, level=0.5)

            self.msg_queue.put(("mesh", verts, faces, metrics, mask_vol.shape))
            self.msg_queue.put(("status", "Done."))
        except Exception as e:
            self.msg_queue.put(("error", f"{e}\n\n{traceback.format_exc()}"))

    # ------------------------------------------------------------------
    def _poll_queue(self):
        try:
            while True:
                item = self.msg_queue.get_nowait()
                kind = item[0]

                if kind == "status":
                    self.status_var.set(item[1])
                elif kind == "progress":
                    _, done, total, label = item
                    self.progress.config(maximum=total, value=done)
                    self.status_var.set(label)
                elif kind == "metrics":
                    self._update_metrics_panel(item[1])
                elif kind == "mesh":
                    _, verts, faces, metrics, shape = item
                    self._draw_mesh(verts, faces, metrics, shape)
                    self.run_btn.config(state=tk.NORMAL)
                    self.export_btn.config(state=tk.NORMAL)
                elif kind == "error":
                    self.status_var.set("Error — see dialog.")
                    self.run_btn.config(state=tk.NORMAL)
                    messagebox.showerror("Failed", item[1])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _update_metrics_panel(self, metrics: dict):
        self.metric_defect_frac.set(f"{metrics['defect_fraction_pct']:.4f}%")
        self.metric_pore_count.set(f"{metrics['pore_count']:,}")
        self.metric_mean_area.set(f"{metrics['mean_pore_area_px']:.2f}")
        self.metric_mean_diam.set(f"{metrics['mean_equiv_diameter_px']:.2f}")

    def _draw_mesh(self, verts, faces, metrics, shape):
        self.ax.clear()
        # Defect zones are highlighted by construction: the mesh is built
        # only from the U-Net's predicted defect voxels, rendered in red
        # against the dark background — solid material is not shown.
        self.ax.plot_trisurf(
            verts[:, 0], verts[:, 1], verts[:, 2],
            triangles=faces, color="#ff4444", alpha=0.9,
            linewidth=0, antialiased=True,
        )
        self._style_axes(
            f"Predicted defects — {shape[0]} slices, "
            f"defect fraction {metrics['defect_fraction_pct']:.3f}%, "
            f"{metrics['pore_count']:,} pores"
        )
        self.canvas.draw()

    def _export_image(self):
        path = filedialog.asksaveasfilename(
            title="Save current view as image",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
        )
        if path:
            self.fig.savefig(path, dpi=150, facecolor=self.fig.get_facecolor())
            messagebox.showinfo("Saved", f"Saved to:\n{path}")


def main():
    root = tk.Tk()
    XCTViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
