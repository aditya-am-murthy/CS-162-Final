"""Post-train: figures, preference maps, dynamic trajectory plots → results/<run_id>/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

from ml_cartography.analysis.data_map import annotate_regions
from ml_cartography.analysis.paper_figures import (
    generate_all_insight_figures,
    plot_data_map_fig1,
)
from ml_cartography.training.experiment_run import ExperimentPaths, RESULTS_ROOT
from ml_cartography.utils.io import read_jsonl, write_jsonl


def plot_region_trajectories(
    trajectories: List[Dict],
    output_path: Path,
    max_guids: int = 40,
) -> None:
    """How regions change across snapshot epochs (Idea #2)."""
    region_to_y = {
        "easy_to_learn": 2,
        "ambiguous": 1,
        "hard_to_learn": 0,
        "mixed": 0.5,
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    for row in trajectories[:max_guids]:
        hist = row.get("history", [])
        if len(hist) < 2:
            continue
        xs = [h["epoch"] for h in hist]
        ys = [region_to_y.get(h["region"], 0.5) for h in hist]
        ax.plot(xs, ys, alpha=0.35, linewidth=1)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["hard", "ambiguous", "easy"])
    ax.set_xlabel("Training epoch (snapshot)")
    ax.set_ylabel("Region")
    ax.set_title("Dynamic data map: region trajectories (sample)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_preference_data_map(rows: List[Dict], output_path: Path) -> None:
    plot_data_map_fig1(
        rows,
        output_path,
        title="Preference Data Map (reward margin dynamics)",
        dataset_label="UltraFeedback-style preferences",
    )


def finalize_experiment(
    paths: ExperimentPaths,
    task_type: str = "classification",
    publish: bool = True,
) -> Path:
    """Build figures under experiment root and publish to results/<run_id>/."""
    coords_path = paths.coordinates_path()
    if not coords_path.is_file():
        logs = paths.epoch_logs_path()
        if logs.is_file():
            from ml_cartography.core.dynamics import build_record, add_epoch_observation, summarize_record
            from collections import defaultdict

            records_map = {}
            for row in read_jsonl(logs):
                g = row["guid"]
                if g not in records_map:
                    records_map[g] = build_record(g, int(row["gold_label"]))
                add_epoch_observation(
                    records_map[g],
                    float(row["prob_gold"]),
                    int(row["pred_label"]),
                )
            coords = [summarize_record(r) for r in records_map.values()]
            write_jsonl(coords_path, coords)
        else:
            raise FileNotFoundError(f"no dynamics at {paths.dynamics_dir}")

    rows = annotate_regions(read_jsonl(coords_path))
    write_jsonl(paths.regions_path(), rows)

    fig_dir = paths.figures_dir
    if task_type == "preference":
        plot_preference_data_map(rows, fig_dir / "preference_data_map.png")
    else:
        clean, noised = rows, rows
        try:
            from ml_cartography.data.synthetic_cartography import apply_noise_shift

            clean, noised = apply_noise_shift(rows, 0.01, seed=42)
        except Exception:
            pass
        generate_all_insight_figures(rows, fig_dir, clean_for_noise=clean, noised_for_noise=noised)

    traj_path = paths.trajectories_path()
    if traj_path.is_file():
        plot_region_trajectories(
            read_jsonl(traj_path),
            fig_dir / "dynamic_region_trajectories.png",
        )

    manifest = {
        "run_id": paths.run_id,
        "task_type": task_type,
        "coordinates": str(coords_path),
        "regions": str(paths.regions_path()),
        "figures_dir": str(fig_dir),
        "models_dir": str(paths.models_dir),
    }
    if paths.config_path.is_file():
        with paths.config_path.open("r", encoding="utf-8") as f:
            manifest["config"] = json.load(f)
    paths.write_manifest(manifest)

    if publish:
        return paths.publish_to_results(RESULTS_ROOT)
    return paths.root
