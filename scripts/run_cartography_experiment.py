#!/usr/bin/env python3
"""
End-to-end cartography experiment: train → dynamics → maps → timestamped results/.

Tasks:
  snli          — classification cartography (DistilBERT, RoBERTa, Llama, Mistral)
  preference    — Preference Data Maps (Idea #1)
  instruction   — instruction-tuning dynamics (Idea #1)
  dynamic       — SNLI + iterative snapshots + curriculum (Idea #2)

Examples:
  python scripts/run_cartography_experiment.py --task snli --preset distilbert
  python scripts/run_cartography_experiment.py --task preference --preset distilbert --max-train-samples 2000
  python scripts/run_cartography_experiment.py --task dynamic --preset roberta-base --curriculum-after-epoch 2
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse
import json
import shutil
from collections import Counter

from ml_cartography.analysis.data_map import annotate_regions, save_data_map_plot
from ml_cartography.analysis.preference_map import (
    annotate_preference_regions,
    save_preference_map_plot,
)
from ml_cartography.core.dynamics import (
    add_epoch_observation,
    build_record,
    summarize_record,
)
from ml_cartography.core.preference_dynamics import preference_epoch_rows_to_coordinates
from ml_cartography.training.dynamic_cartography import build_region_trajectories
from ml_cartography.training.experiment_run import ExperimentPaths
from ml_cartography.training.glue_trainer import (
    MODEL_PRESETS,
    TrainConfig,
    apply_preset_defaults,
    train_and_collect_dynamics,
)
from ml_cartography.training.instruction_trainer import (
    InstructionTrainConfig,
    train_and_collect_instruction_dynamics,
)
from ml_cartography.training.preference_trainer import (
    PreferenceTrainConfig,
    train_and_collect_preference_dynamics,
)
from ml_cartography.utils.io import read_jsonl, write_jsonl
from scripts.common import add_wandb_args, finish_wandb, init_wandb, load_hf_credentials


def _collect_dynamics_from_logs(log_path: Path) -> list[dict]:
    records = {}
    for row in read_jsonl(log_path):
        guid = str(row["guid"])
        gold = int(row.get("gold_label", 0))
        if guid not in records:
            records[guid] = build_record(guid, gold)
        add_epoch_observation(
            records[guid],
            float(row["prob_gold"]),
            int(row.get("pred_label", 0)),
        )
    return [summarize_record(r) for r in records.values()]


def _build_figures(exp: ExperimentPaths, coordinates: list[dict], task: str) -> None:
    if task == "preference":
        tagged = annotate_preference_regions(coordinates)
        plot_path = exp.figures_dir / "preference_data_map.png"
        save_preference_map_plot(tagged, plot_path)
        write_jsonl(exp.regions_path(), tagged)
    else:
        tagged = annotate_regions(coordinates)
        plot_path = exp.figures_dir / "data_map.png"
        save_data_map_plot(tagged, plot_path)
        write_jsonl(exp.regions_path(), tagged)

    region_counts = Counter(r["region"] for r in tagged)
    with (exp.figures_dir / "region_counts.json").open("w", encoding="utf-8") as f:
        json.dump(dict(region_counts), f, indent=2)


def _export_high_variability_subset(exp: ExperimentPaths, tagged: list[dict], ratio: float = 0.33) -> Path:
    """Filter ambiguous / high-variability examples for DPO-style training (Idea #1)."""
    ambiguous = [r for r in tagged if "ambiguous" in r.get("region", "")]
    ambiguous.sort(key=lambda r: float(r["variability"]), reverse=True)
    k = max(1, int(len(ambiguous) * ratio)) if ambiguous else 0
    subset = ambiguous[:k] if k else sorted(tagged, key=lambda r: float(r["variability"]), reverse=True)[
        : max(1, int(len(tagged) * ratio))
    ]
    out = exp.dynamics_dir / "subset_high_variability.jsonl"
    write_jsonl(out, [{"guid": r["guid"], "region": r["region"]} for r in subset])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["snli", "preference", "instruction", "dynamic"],
        default="snli",
    )
    parser.add_argument("--preset", choices=list(MODEL_PRESETS.keys()), default="distilbert")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--max-eval-samples", type=int, default=2000)
    parser.add_argument("--curriculum-after-epoch", type=int, default=0)
    parser.add_argument("--run-id", default=None, help="Override timestamp run id")
    parser.add_argument("--no-publish", action="store_true", help="Skip copy to results/")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading for large models")
    add_wandb_args(parser)
    args = parser.parse_args()
    load_hf_credentials()

    model_name = args.model_name or MODEL_PRESETS[args.preset]
    task_slug = f"{args.task}_{args.preset}"
    exp = ExperimentPaths.create(task_slug=task_slug, run_id=args.run_id)

    config = {
        "task": args.task,
        "preset": args.preset,
        "model_name": model_name,
        "epochs": args.epochs,
        "max_train_samples": args.max_train_samples,
    }
    exp.write_config(config)

    init_wandb(
        args,
        job_type=f"cartography_{args.task}",
        config={**config, "run_id": exp.run_id},
    )
    import wandb

    wandb_run = wandb.run
    metrics_log = exp.training_metrics_path()

    summary = {}
    if args.task in ("snli", "dynamic"):
        cfg = TrainConfig(
            model_name=model_name,
            epochs=args.epochs,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            output_logs=exp.epoch_logs_path(),
            checkpoint_dir=exp.checkpoints_dir,
            snapshot_dir=exp.snapshots_dir,
            dynamic_snapshots=True,
            curriculum_after_epoch=args.curriculum_after_epoch if args.task == "dynamic" else 0,
        )
        if args.batch_size is None:
            cfg = apply_preset_defaults(cfg, args.preset)
        else:
            cfg.batch_size = args.batch_size
        if args.no_4bit:
            cfg.load_in_4bit = False
        summary = train_and_collect_dynamics(cfg, wandb_run=wandb_run, metrics_log=metrics_log)

        coordinates = _collect_dynamics_from_logs(exp.epoch_logs_path())
        write_jsonl(exp.coordinates_path(), coordinates)

        snap_dir = exp.snapshots_dir
        if snap_dir.is_dir():
            snapshots = sorted(snap_dir.glob("epoch_*_coordinates.jsonl"))
            if snapshots:
                pairs = []
                for p in snapshots:
                    epoch = int(p.stem.split("_")[1])
                    pairs.append((epoch, p))
                trajectories = build_region_trajectories(pairs)
                write_jsonl(exp.trajectories_path(), trajectories)
                if wandb_run:
                    wandb.log({"num_trajectory_examples": len(trajectories)})

        _build_figures(exp, coordinates, task="snli")
        tagged = read_jsonl(exp.regions_path())
        subset_path = _export_high_variability_subset(exp, tagged)
        summary["high_variability_subset"] = str(subset_path)

    elif args.task == "preference":
        pcfg = PreferenceTrainConfig(
            model_name=model_name,
            epochs=args.epochs,
            max_samples=args.max_train_samples,
            output_logs=exp.epoch_logs_path(),
            snapshot_dir=exp.snapshots_dir,
            checkpoint_dir=exp.models_dir,
        )
        if args.batch_size:
            pcfg.batch_size = args.batch_size
        summary = train_and_collect_preference_dynamics(
            pcfg, wandb_run=wandb_run, metrics_log=metrics_log
        )
        coordinates = preference_epoch_rows_to_coordinates(read_jsonl(exp.epoch_logs_path()))
        write_jsonl(exp.coordinates_path(), coordinates)
        _build_figures(exp, coordinates, task="preference")
        tagged = read_jsonl(exp.regions_path())
        _export_high_variability_subset(exp, tagged)

    elif args.task == "instruction":
        icfg = InstructionTrainConfig(
            model_name=model_name,
            epochs=args.epochs,
            max_samples=args.max_train_samples,
            output_logs=exp.epoch_logs_path(),
            snapshot_dir=exp.snapshots_dir,
        )
        if args.batch_size:
            icfg.batch_size = args.batch_size
        summary = train_and_collect_instruction_dynamics(
            icfg, wandb_run=wandb_run, metrics_log=metrics_log
        )
        coordinates = _collect_dynamics_from_logs(exp.epoch_logs_path())
        write_jsonl(exp.coordinates_path(), coordinates)
        _build_figures(exp, coordinates, task="snli")

    manifest = {
        "run_id": exp.run_id,
        "task": args.task,
        "preset": args.preset,
        "summary": summary,
        "paths": {
            "epoch_logs": str(exp.epoch_logs_path()),
            "coordinates": str(exp.coordinates_path()),
            "regions": str(exp.regions_path()),
            "figures": str(exp.figures_dir),
        },
    }
    exp.write_manifest(manifest)

    if wandb_run:
        for fig in exp.figures_dir.glob("*.png"):
            wandb.log({f"artifact/{fig.name}": wandb.Image(str(fig))})
        wandb.log(summary)

    results_dest = None
    if not args.no_publish:
        results_dest = exp.publish_to_results()
        print(f"published -> {results_dest}")

    print(json.dumps(manifest, indent=2))
    finish_wandb()


if __name__ == "__main__":
    main()
