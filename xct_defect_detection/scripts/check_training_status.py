"""
=============================================================================
check_training_status.py — Quick status check for a pipeline.py run
=============================================================================
Reports:
    - The active/most recent MLflow run for this project's experiment
      (status, start time, elapsed duration, latest logged metrics)
    - Current GPU utilisation/memory/power (via nvidia-smi, if available)

MLflow logs metrics once per completed epoch (see training/trainer.py), so
this is a more reliable progress signal than reading the training process's
own console output, which only prints once an epoch finishes.

Run from repository root (works while training is running, or after):
    python scripts/check_training_status.py
Add --watch to refresh every 30s until interrupted:
    python scripts/check_training_status.py --watch
=============================================================================
"""
import sys
import time
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def gpu_status() -> str:
    if shutil.which("nvidia-smi") is None:
        return "  (nvidia-smi not found — GPU status unavailable)"
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        util, mem_used, mem_total, power = [
            x.strip() for x in out.stdout.strip().split(",")
        ]
        return (f"  GPU utilisation : {util}%\n"
                f"  GPU memory      : {mem_used} / {mem_total} MiB\n"
                f"  GPU power draw  : {power} W")
    except Exception as e:
        return f"  (could not read GPU status: {e})"


def mlflow_status() -> str:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        return "  (mlflow not installed)"

    mlflow.set_tracking_uri(config.MLFLOW_URI)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    if experiment is None:
        return f"  No experiment named '{config.MLFLOW_EXPERIMENT}' found yet."

    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return "  No runs logged yet."

    run = runs[0]
    info = run.info
    started = datetime.fromtimestamp(info.start_time / 1000, tz=timezone.utc)
    elapsed = datetime.now(timezone.utc) - started
    end_note = ""
    if info.end_time:
        ended = datetime.fromtimestamp(info.end_time / 1000, tz=timezone.utc)
        elapsed = ended - started
        end_note = f"  Ended         : {ended.isoformat(timespec='seconds')}"

    lines = [
        f"  Run ID        : {info.run_id}",
        f"  Status        : {info.status}",
        f"  Started       : {started.isoformat(timespec='seconds')}",
        f"  Elapsed       : {elapsed}",
    ]
    if end_note:
        lines.append(end_note)

    params = run.data.params
    if params:
        lines.append(f"  Batch size    : {params.get('batch_size', '?')}")
        lines.append(f"  Num epochs    : {params.get('num_epochs', '?')}")
        lines.append(f"  Loss function : {params.get('loss_function', '?')}")

    metrics = run.data.metrics
    if metrics:
        # metrics dict holds only the LATEST value per key — good enough
        # for a status snapshot (full history via client.get_metric_history
        # if ever needed).
        lines.append("")
        lines.append("  Latest logged metrics:")
        for key in ["train_loss", "train_dice", "train_iou", "train_recall",
                    "val_loss", "val_dice", "val_iou", "val_recall",
                    "learning_rate"]:
            if key in metrics:
                lines.append(f"    {key:15s}: {metrics[key]:.4f}")

        history = client.get_metric_history(info.run_id, "val_dice")
        if history:
            lines.append("")
            lines.append(f"  Epochs completed so far: {len(history)}")
            if len(history) >= 2:
                per_epoch = (history[-1].timestamp - history[0].timestamp) / \
                            (len(history) - 1) / 1000.0
                lines.append(f"  Avg. time per epoch     : {per_epoch/60:.1f} min")
    else:
        lines.append("")
        lines.append("  No epoch has completed yet — still on the first epoch.")

    return "\n".join(lines)


def print_status():
    print("=" * 65)
    print(f"  Training status check — {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 65)
    print("\n[MLflow]")
    print(mlflow_status())
    print("\n[GPU]")
    print(gpu_status())
    print()


def main():
    watch = "--watch" in sys.argv
    if not watch:
        print_status()
        return

    try:
        while True:
            print_status()
            time.sleep(30)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
