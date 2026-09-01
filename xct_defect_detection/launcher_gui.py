"""
=============================================================================
launcher_gui.py — Basic control panel: one button per pipeline stage
=============================================================================
A simple Tkinter front-end for the scripts in this repository. Each button
runs the corresponding script as a subprocess (from the project root, using
the same Python interpreter this launcher runs under) and streams its
stdout/stderr live into the log pane below. Only one script runs at a time.

This does not replace gui_app.py (the interactive inference + 3D viewer) —
it complements it, and has a button to open it too.

Run from project root:
    python launcher_gui.py
=============================================================================
"""

import os
import sys
import subprocess
import threading
import queue
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

ROOT = Path(__file__).resolve().parent

# label -> (script path relative to ROOT, tooltip)
ACTIONS = [
    ("1. Preprocess all samples",
     "scripts/run_preprocess.py",
     "Runs the production 3-stage pipeline (percentile norm, median filter, "
     "adaptive NLM) over every sample in data/raw/."),
    ("2. Run full pipeline (train U-Net)",
     "pipeline.py",
     "Preprocessing + Bernsen pseudo-labels + patch extraction + U-Net "
     "training, using the settings currently in config.py."),
    ("3. Compare classical vs. U-Net",
     "scripts/compare_methods_report.py",
     "Generates the Bernsen/Otsu/Yen/U-Net comparison table and figures."),
    ("4. Predict + 3D visualise all samples",
     "scripts/predict_and_visualize_3d.py",
     "Runs the trained U-Net over every sample and reconstructs 3D "
     "defect/solid meshes."),
    ("5. Open interactive inference GUI",
     "gui_app.py",
     "Opens gui_app.py in a separate window: pick a folder, run inference, "
     "view the result in an interactive 3D viewer."),
]


class LauncherGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("XCT Pipeline — Launcher")
        root.geometry("760x520")

        self.proc = None
        self.out_queue: "queue.Queue[str]" = queue.Queue()

        header = ttk.Label(
            root, text="XCT Defect Detection — Pipeline Launcher",
            font=("Segoe UI", 13, "bold")
        )
        header.pack(pady=(12, 2))
        sub = ttk.Label(
            root, text=f"Project root: {ROOT}", foreground="#666"
        )
        sub.pack(pady=(0, 10))

        btn_frame = ttk.Frame(root)
        btn_frame.pack(fill="x", padx=16)

        for label, script, tip in ACTIONS:
            row = ttk.Frame(btn_frame)
            row.pack(fill="x", pady=3)
            b = ttk.Button(
                row, text=label, width=34,
                command=lambda s=script: self.run_script(s)
            )
            b.pack(side="left")
            ttk.Label(row, text=tip, foreground="#666",
                      wraplength=420, justify="left").pack(side="left", padx=10)

        ctrl_frame = ttk.Frame(root)
        ctrl_frame.pack(fill="x", padx=16, pady=(10, 4))
        self.stop_btn = ttk.Button(
            ctrl_frame, text="Stop running script", command=self.stop_script,
            state="disabled"
        )
        self.stop_btn.pack(side="left")
        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(ctrl_frame, textvariable=self.status_var,
                  foreground="#2f6feb").pack(side="left", padx=12)

        ttk.Label(root, text="Output:", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", padx=16, pady=(8, 0)
        )
        self.log = scrolledtext.ScrolledText(
            root, height=18, bg="#111", fg="#ddd",
            font=("Consolas", 9), state="disabled"
        )
        self.log.pack(fill="both", expand=True, padx=16, pady=(2, 12))

        self.root.after(150, self._drain_queue)

    # -------------------------------------------------------------------
    def _log(self, text: str):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _drain_queue(self):
        try:
            while True:
                line = self.out_queue.get_nowait()
                self._log(line)
        except queue.Empty:
            pass
        self.root.after(150, self._drain_queue)

    # -------------------------------------------------------------------
    def run_script(self, script_rel: str):
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning(
                "Busy", "A script is already running. Stop it first."
            )
            return

        script_path = ROOT / script_rel
        if not script_path.exists():
            messagebox.showerror("Not found", f"Missing: {script_path}")
            return

        self._log(f"\n{'='*70}\n$ python {script_rel}\n{'='*70}\n")
        self.status_var.set(f"Running {script_rel} ...")
        self.stop_btn.configure(state="normal")

        def worker():
            try:
                self.proc = subprocess.Popen(
                    [sys.executable, str(script_path)],
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                for line in self.proc.stdout:
                    self.out_queue.put(line)
                code = self.proc.wait()
                self.out_queue.put(f"\n[exit code {code}]\n")
            except Exception as e:
                self.out_queue.put(f"\n[launcher error] {e}\n")
            finally:
                self.status_var.set("Idle.")
                self.stop_btn.configure(state="disabled")
                self.proc = None

        threading.Thread(target=worker, daemon=True).start()

    def stop_script(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            self._log("\n[stopped by user]\n")
        self.status_var.set("Idle.")
        self.stop_btn.configure(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = LauncherGUI(root)
    root.mainloop()
