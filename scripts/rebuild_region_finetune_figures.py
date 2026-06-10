#!/usr/bin/env python3
"""Rebuild local charts for finished region-finetune training runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from ml_cartography.analysis.data_map import save_data_map_plot
from ml_cartography.analysis.training_plots import save_training_curve_plot
from ml_cartography.training.dynamic_cartography import records_to_coordinates
from ml_cartography.utils.io import read_jsonl
from scripts.run_cartography_experiment import _build_figures, _collect_dynamics_from_logs

PAPER_STYLE_DATASETS = ("snli", "mnli", "qnli", "winogrande")


def rebuild_run(run_dir: Path, *, dataset: str, per_epoch: bool) -> None:
    log_path = run_dir / "epoch_predictions.jsonl"
    if not log_path.is_file():
        print(f"skip {run_dir}: missing epoch_predictions.jsonl")
        return

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(log_path)
    coordinates = _collect_dynamics_from_logs(log_path)
    exp = SimpleNamespace(
        figures_dir=figures_dir,
        regions_path=lambda: figures_dir / "cartography_with_regions.jsonl",
    )
    _build_figures(exp, coordinates, task="snli", dataset=dataset)

    metrics_path = run_dir / "training_metrics.jsonl"
    if metrics_path.is_file():
        save_training_curve_plot(
            metrics_path,
            figures_dir / "training_curve.png",
            title=f"{run_dir.parent.name} — training curve",
        )

    if per_epoch:
        max_epoch = max(int(r["epoch"]) for r in records)
        paper_style = dataset.lower() in PAPER_STYLE_DATASETS
        for epoch in range(1, max_epoch + 1):
            coords = records_to_coordinates(records, max_epoch=epoch)
            title = f"{dataset.upper()} data map (through epoch {epoch})"
            save_data_map_plot(
                coords,
                figures_dir / f"epoch_{epoch:03d}_data_map_correctness.png",
                color_by="correctness" if paper_style else "region",
                title=title,
            )
            if paper_style:
                save_data_map_plot(
                    coords,
                    figures_dir / f"epoch_{epoch:03d}_data_map_regions.png",
                    color_by="region",
                    title=f"{title} (by region)",
                )

    pngs = sorted(figures_dir.glob("*.png"))
    print(f"{run_dir.name}: {len(pngs)} charts -> {figures_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("data/processed/region_finetune_winogrande"),
        help="Region finetune output root (contains training_runs/)",
    )
    parser.add_argument("--dataset", default="winogrande")
    parser.add_argument(
        "--per-epoch",
        action="store_true",
        help="Also write epoch_NNN_data_map_*.png for each epoch",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    runs = sorted(root.glob("training_runs/*/restart_*"))
    if not runs:
        raise SystemExit(f"no training runs under {root / 'training_runs'}")

    for run_dir in runs:
        if not (run_dir / "summary.json").is_file():
            continue
        rebuild_run(run_dir, dataset=args.dataset, per_epoch=args.per_epoch)


if __name__ == "__main__":
    main()
