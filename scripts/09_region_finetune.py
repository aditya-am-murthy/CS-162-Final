#!/usr/bin/env python3
"""
§3 fixed 33% strategy comparison

Exports one 33% subset per selection strategy and optionally retrains on each.

Strategies (from paper §3):
  high_variability   — ambiguous region (top variability)
  low_confidence     — hard-to-learn (bottom confidence)
  high_confidence    — easy-to-learn (top confidence)
  low_variability    — stable region (bottom variability)
  high_correctness   — always-correct examples
  low_correctness    — rarely-correct examples
  random             — random 33% baseline
  full               — full training set (100%)

Use --train to retrain on each exported subset.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import sys
from collections import Counter
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from tqdm import tqdm

from ml_cartography.experiments.selection import (
    proxy_eval_scores,
    select_by_strategy,
    subset_summary,
)
from ml_cartography.utils.io import read_jsonl, write_json, write_jsonl
from scripts.common import (
    add_wandb_args,
    finish_wandb,
    init_wandb,
    load_hf_credentials,
    load_pipeline_config,
    use_wandb,
)


STRATEGIES = (
    "high_variability",
    "low_confidence",
    "high_confidence",
    "low_variability",
    "high_correctness",
    "low_correctness",
    "random",
    "full",
)

KEEP_RATIO = 0.33


def _region_counts(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(str(r.get("region", "missing")) for r in rows))


def _annotate_regions(rows: list[dict]) -> list[dict]:
    """Assign regions when the input JSONL lacks them."""
    import statistics

    conf = [float(r["confidence"]) for r in rows]
    var = [float(r["variability"]) for r in rows]

    def quantile(vals: list[float], q: float) -> float:
        ordered = sorted(vals)
        pos = (len(ordered) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] * (1.0 - pos + lo) + ordered[hi] * (pos - lo)

    conf_q25 = quantile(conf, 0.25)
    conf_q75 = quantile(conf, 0.75)
    var_q40 = quantile(var, 0.40)
    var_q80 = quantile(var, 0.80)

    tagged = []
    for row in rows:
        c = float(row["confidence"])
        v = float(row["variability"])
        if c >= conf_q75 and v <= var_q40:
            region = "easy_to_learn"
        elif c <= conf_q25 and v <= var_q40:
            region = "hard_to_learn"
        elif v >= var_q80:
            region = "ambiguous"
        else:
            region = "mixed"
        tagged.append({**row, "region": region})
    return tagged


def _train_command(
    *,
    subset_path: Path,
    output_log: Path,
    dataset: str,
    preset: str,
    model_name: str | None,
    epochs: int,
    max_eval_samples: int,
    winogrande_config: str,
    seed: int,
    no_wandb: bool,
) -> list[str]:
    cmd = [
        "python",
        "scripts/train_and_collect_dynamics.py",
        "--dataset", dataset,
        "--preset", preset,
        "--epochs", str(epochs),
        "--max-train-samples", "0",
        "--max-eval-samples", str(max_eval_samples),
        "--subset-file", str(subset_path),
        "--output", str(output_log),
        "--seed", str(seed),
    ]
    if dataset == "winogrande":
        cmd.extend(["--winogrande-config", winogrande_config])
    if model_name:
        cmd.extend(["--model-name", model_name])
    if no_wandb:
        cmd.append("--no-wandb")
    return cmd


def _write_commands(path: Path, commands: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n\n")
        for cmd in commands:
            f.write(" ".join(shlex.quote(p) for p in cmd))
            f.write("\n")
    path.chmod(0o755)


def _write_manifest_csv(path: Path, entries: list[dict]) -> None:
    if not entries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "strategy",
        "keep_ratio",
        "seed",
        "count",
        "mean_confidence",
        "mean_variability",
        "mean_correctness",
        "proxy_id_accuracy",
        "proxy_ood_accuracy",
        "path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow({key: entry.get(key) for key in fieldnames})


def _entry(
    *,
    name: str,
    strategy: str,
    keep_ratio: float,
    rows: list[dict],
    path: Path,
    seed: int,
) -> dict:
    write_jsonl(path, rows)
    summary = subset_summary(rows)
    scores = proxy_eval_scores(rows)
    return {
        "name": name,
        "strategy": strategy,
        "keep_ratio": keep_ratio,
        "seed": seed,
        "count": int(summary["count"]),
        "mean_confidence": summary["mean_confidence"],
        "mean_variability": summary["mean_variability"],
        "mean_correctness": summary["mean_correctness"],
        "proxy_id_accuracy": scores["proxy_id_accuracy"],
        "proxy_ood_accuracy": scores["proxy_ood_accuracy"],
        "region_counts": _region_counts(rows),
        "path": str(path),
    }


def _train_subsets(args: argparse.Namespace, entries: list[dict]) -> list[dict]:
    from ml_cartography.training.glue_trainer import (
        MODEL_PRESETS,
        TrainConfig,
        apply_preset_defaults,
        load_guids_from_jsonl,
        train_and_collect_dynamics,
    )

    load_hf_credentials()
    model_name = args.model_name or MODEL_PRESETS.get(args.preset, args.preset)
    train_entries = entries[: args.limit_training_runs] if args.limit_training_runs else entries
    results: list[dict] = []

    for entry in tqdm(train_entries, desc="retraining subsets"):
        subset_path = Path(entry["path"])
        for restart_idx in range(args.restarts):
            run_seed = args.seed + restart_idx
            run_dir = (
                args.output_dir
                / "training_runs"
                / entry["name"]
                / f"restart_{restart_idx:02d}"
            )
            cfg = TrainConfig(
                dataset=args.dataset,
                model_name=model_name,
                max_train_samples=None,
                max_eval_samples=None if args.max_eval_samples == 0 else args.max_eval_samples,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                max_length=args.max_length,
                seed=run_seed,
                fp16=not args.no_fp16,
                output_logs=run_dir / "epoch_predictions.jsonl",
                checkpoint_dir=run_dir / "checkpoints" if args.save_checkpoints else None,
                subset_guids=load_guids_from_jsonl(subset_path),
                snapshot_dir=run_dir / "snapshots",
                dynamic_snapshots=False,
                winogrande_config=args.winogrande_config,
            )
            cfg = apply_preset_defaults(cfg, args.preset)
            if args.batch_size is not None:
                cfg.batch_size = args.batch_size
            if args.no_4bit:
                cfg.load_in_4bit = False

            summary = train_and_collect_dynamics(cfg)
            result = {
                "subset": entry["name"],
                "strategy": entry["strategy"],
                "restart": restart_idx,
                "seed": run_seed,
                "subset_path": str(subset_path),
                **summary,
            }
            results.append(result)

            if use_wandb(args):
                import wandb

                wandb.log(
                    {
                        "train/subset": entry["name"],
                        "train/strategy": entry["strategy"],
                        "train/restart": restart_idx,
                        "train/final_val_accuracy": summary["final_val_accuracy"],
                        "train/num_train": summary["num_train"],
                    }
                )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/example_pipeline.json"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/region_subsets"))
    parser.add_argument("--keep-ratio", type=float, default=KEEP_RATIO)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(STRATEGIES),
        choices=list(STRATEGIES),
        metavar="STRATEGY",
        help=f"Strategies to run. Default: all 8. Choices: {STRATEGIES}",
    )
    parser.add_argument("--train", action="store_true", help="Retrain on each exported subset.")
    parser.add_argument("--dataset", choices=["snli", "mnli", "qnli", "winogrande"], default="winogrande")
    parser.add_argument("--preset", default="roberta-large")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--winogrande-config", default="winogrande_xl")
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--limit-training-runs", type=int, default=0)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    add_wandb_args(parser)
    args = parser.parse_args()

    if args.keep_ratio <= 0.0 or args.keep_ratio > 1.0:
        raise ValueError("--keep-ratio must be in (0, 1]")
    if args.restarts < 1:
        raise ValueError("--restarts must be at least 1")

    cfg = load_pipeline_config(args.config)
    input_path = args.input or Path(cfg["coordinates_with_regions_output"])
    rows = read_jsonl(input_path)
    if not rows:
        raise ValueError(f"no rows found in {input_path}")
    if any("region" not in row for row in rows):
        rows = _annotate_regions(rows)

    init_wandb(
        args,
        job_type="region_finetune",
        config={
            "input": str(input_path),
            "output_dir": str(args.output_dir),
            "keep_ratio": args.keep_ratio,
            "strategies": args.strategies,
            "train": args.train,
        },
    )

    subsets_dir = args.output_dir / "subsets"
    logs_dir = args.output_dir / "command_logs"
    subsets_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    commands: list[list[str]] = []

    for strategy in tqdm(args.strategies, desc="selecting subsets"):
        keep_ratio = args.keep_ratio if strategy != "full" else 1.0
        subset = select_by_strategy(rows, strategy=strategy, keep_ratio=keep_ratio, seed=args.seed)
        slug = "33pct" if args.keep_ratio == KEEP_RATIO else f"{int(args.keep_ratio * 100)}pct"
        name = f"{strategy}_{slug}" if strategy != "full" else "full"
        path = subsets_dir / f"{name}.jsonl"

        entry = _entry(
            name=name,
            strategy=strategy,
            keep_ratio=keep_ratio,
            rows=subset,
            path=path,
            seed=args.seed,
        )
        entries.append(entry)

        commands.append(
            _train_command(
                subset_path=path,
                output_log=logs_dir / f"{name}.jsonl",
                dataset=args.dataset,
                preset=args.preset,
                model_name=args.model_name,
                epochs=args.epochs,
                max_eval_samples=args.max_eval_samples,
                winogrande_config=args.winogrande_config,
                seed=args.seed,
                no_wandb=args.no_wandb,
            )
        )

    manifest = {
        "paper_task": "Fixed 33% Strategy Comparison (§3)",
        "input": str(input_path),
        "output_dir": str(args.output_dir),
        "dataset": args.dataset,
        "preset": args.preset,
        "keep_ratio": args.keep_ratio,
        "seed": args.seed,
        "entries": entries,
    }
    manifest_path = args.output_dir / "manifest.json"
    csv_path = args.output_dir / "manifest.csv"
    commands_path = args.output_dir / "train_commands.sh"
    write_json(manifest_path, manifest)
    _write_manifest_csv(csv_path, entries)
    _write_commands(commands_path, commands)

    if use_wandb(args):
        import wandb

        for entry in entries:
            wandb.log(
                {
                    f"{entry['strategy']}/count": entry["count"],
                    f"{entry['strategy']}/mean_confidence": entry["mean_confidence"],
                    f"{entry['strategy']}/mean_variability": entry["mean_variability"],
                    f"{entry['strategy']}/proxy_id_accuracy": entry["proxy_id_accuracy"],
                    f"{entry['strategy']}/proxy_ood_accuracy": entry["proxy_ood_accuracy"],
                }
            )

    train_results = []
    if args.train:
        train_results = _train_subsets(args, entries)
        results_path = args.output_dir / "train_results.json"
        write_json(results_path, {"results": train_results})
        manifest["train_results"] = train_results
        write_json(manifest_path, manifest)

    print(f"wrote {len(entries)} subset definitions")
    print(f"manifest: {manifest_path}")
    print(f"csv: {csv_path}")
    print(f"commands: {commands_path}")
    if args.train:
        print(f"trained {len(train_results)} runs -> {args.output_dir / 'train_results.json'}")

    finish_wandb()


if __name__ == "__main__":
    main()
