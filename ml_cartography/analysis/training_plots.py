"""Training-run charts saved to disk (independent of W&B)."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIR = _REPO_ROOT / ".cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import matplotlib.pyplot as plt

from ml_cartography.utils.io import read_jsonl


def save_training_curve_plot(
    metrics_path: Path,
    output_path: Path,
    *,
    title: str = "Training progress",
) -> Path:
    """Plot train loss and validation accuracy per epoch from training_metrics.jsonl."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(metrics_path) if metrics_path.is_file() else []
    if not rows:
        return output_path

    epochs = [int(r["epoch"]) for r in rows]
    losses = [float(r.get("train_loss", 0.0)) for r in rows]
    val_accs = [float(r.get("val_accuracy", 0.0)) for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, losses, "r-o", label="train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color="red")
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.grid(alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_accs, "g-s", label="val accuracy")
    ax2.set_ylabel("Validation accuracy", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
