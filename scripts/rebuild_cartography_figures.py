#!/usr/bin/env python3
"""Rebuild data maps / coordinates from a finished run's epoch_predictions.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.run_cartography_experiment import _build_figures, _collect_dynamics_from_logs
from ml_cartography.analysis.data_map import save_data_map_plot
from ml_cartography.training.dynamic_cartography import records_to_coordinates
from ml_cartography.training.experiment_run import ExperimentPaths
from ml_cartography.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Experiment run root (contains dynamics/epoch_predictions.jsonl)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name for plot styling (default: read from config.json)",
    )
    parser.add_argument(
        "--per-epoch-maps",
        action="store_true",
        help="Also write dynamics/snapshots/epoch_*_data_map.png for each epoch",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    log_path = run_dir / "dynamics" / "epoch_predictions.jsonl"
    if not log_path.is_file():
        raise SystemExit(f"missing {log_path}")

    dataset = args.dataset
    cfg_path = run_dir / "config.json"
    if dataset is None and cfg_path.is_file():
        with cfg_path.open(encoding="utf-8") as f:
            dataset = json.load(f).get("dataset", "snli")
    dataset = dataset or "snli"

    records = read_jsonl(log_path)
    max_epoch = max(int(r["epoch"]) for r in records)
    coordinates = _collect_dynamics_from_logs(log_path)

    dynamics_dir = run_dir / "dynamics"
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = dynamics_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(dynamics_dir / "cartography_coordinates.jsonl", coordinates)

    exp = SimpleNamespace(
        figures_dir=figures_dir,
        dynamics_dir=dynamics_dir,
        snapshots_dir=snapshots_dir,
        regions_path=lambda: dynamics_dir / "cartography_with_regions.jsonl",
    )
    _build_figures(exp, coordinates, task="snli", dataset=dataset)

    if args.per_epoch_maps:
        paper = dataset in ("snli", "mnli", "qnli", "winogrande")
        for epoch in range(1, max_epoch + 1):
            coords = records_to_coordinates(records, max_epoch=epoch)
            save_data_map_plot(
                coords,
                snapshots_dir / f"epoch_{epoch:03d}_data_map.png",
                color_by="correctness" if paper else "region",
                title=f"{dataset.upper()} data map (epoch {epoch})",
            )

    print(f"rebuilt figures in {figures_dir}")
    for p in sorted(figures_dir.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
