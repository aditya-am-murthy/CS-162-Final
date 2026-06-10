#!/usr/bin/env python3
"""
easy-to-learn script

default does what the paper said:
  1. top ambiguous only subsets at 50%, 33%, 25%, 17%, 10%, 5%, and 1%
  2. matched random subsets with the same sizes
  3. mixtures made by replacing part of the 17% ambiguous core with easy examples

use --train to retrain with the exported subset files using the repo existing trainer
"""

from __future__ import annotations

import argparse
import csv
import random
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from tqdm import tqdm

from ml_cartography.experiments.selection import (
    ambiguous_ablation_subset,
    proxy_eval_scores,
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


DEFAULT_AMBIGUOUS_RATIOS = (0.50, 0.33, 0.25, 0.17, 0.10, 0.05, 0.01)
DEFAULT_REPLACE_RATIOS = (0.0, 0.10, 0.25, 0.50, 0.75)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _annotate_regions(rows: list[dict]) -> list[dict]:
    conf = [float(r["confidence"]) for r in rows]
    var = [float(r["variability"]) for r in rows]
    thresholds = {
        "easy_confidence_min": _quantile(conf, 0.75),
        "hard_confidence_max": _quantile(conf, 0.25),
        "low_variability_max": _quantile(var, 0.40),
        "ambiguous_variability_min": _quantile(var, 0.80),
    }
    thresholds["easy_confidence_min"] = max(
        thresholds["easy_confidence_min"],
        thresholds["hard_confidence_max"] + 0.05,
    )
    thresholds["ambiguous_variability_min"] = max(
        thresholds["ambiguous_variability_min"],
        thresholds["low_variability_max"] + 0.02,
    )

    tagged = []
    for row in rows:
        confidence = float(row["confidence"])
        variability = float(row["variability"])
        if (
            confidence >= thresholds["easy_confidence_min"]
            and variability <= thresholds["low_variability_max"]
        ):
            region = "easy_to_learn"
        elif (
            confidence <= thresholds["hard_confidence_max"]
            and variability <= thresholds["low_variability_max"]
        ):
            region = "hard_to_learn"
        elif variability >= thresholds["ambiguous_variability_min"]:
            region = "ambiguous"
        else:
            region = "mixed"
        tagged.append({**row, "region": region})
    return tagged


def _ratio_slug(ratio: float) -> str:
    pct = ratio * 100
    text = f"{pct:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{text}pct"


def _validate_ratios(name: str, ratios: Iterable[float]) -> list[float]:
    clean = [float(r) for r in ratios]
    bad = [r for r in clean if r < 0.0 or r > 1.0]
    if bad:
        raise ValueError(f"{name} ratios must be in [0, 1], got {bad}")
    return clean


def _sample_exact(rows: list[dict], count: int, seed: int) -> list[dict]:
    if count <= 0:
        return []
    copied = list(rows)
    random.Random(seed).shuffle(copied)
    return copied[: min(count, len(copied))]


def _easy_rank_key(row: dict) -> tuple[float, float, float]:
    return (
        -float(row.get("confidence", 0.0)),
        float(row.get("variability", 0.0)),
        -float(row.get("correctness", 0.0)),
    )


def _easy_pool(rows: list[dict], exclude_guids: set[str], easy_source: str) -> list[dict]:
    candidates = [r for r in rows if str(r["guid"]) not in exclude_guids]
    region_easy = [r for r in candidates if r.get("region") == "easy_to_learn"]
    if easy_source == "region":
        return list(region_easy)
    if easy_source == "ranked":
        return sorted(candidates, key=_easy_rank_key)
    if easy_source == "region_then_ranked":
        return sorted(region_easy, key=_easy_rank_key) or sorted(candidates, key=_easy_rank_key)
    raise ValueError(f"unknown easy_source: {easy_source}")


def _replace_with_easy(
    ambiguous_core: list[dict],
    all_rows: list[dict],
    replace_ratio: float,
    seed: int,
    easy_source: str,
) -> list[dict]:
    if not ambiguous_core or replace_ratio <= 0:
        return list(ambiguous_core)

    n_replace = min(len(ambiguous_core), max(1, int(len(ambiguous_core) * replace_ratio)))
    rng = random.Random(seed)
    drop_indices = set(rng.sample(range(len(ambiguous_core)), n_replace))
    kept = [row for idx, row in enumerate(ambiguous_core) if idx not in drop_indices]

    exclude = {str(r["guid"]) for r in kept}
    easy = _easy_pool(all_rows, exclude_guids=exclude, easy_source=easy_source)
    easy = easy[:n_replace]
    return kept + easy


def _region_counts(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(str(r.get("region", "missing")) for r in rows))


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
        "--dataset",
        dataset,
        "--preset",
        preset,
        "--epochs",
        str(epochs),
        "--max-train-samples",
        "0",
        "--max-eval-samples",
        str(max_eval_samples),
        "--subset-file",
        str(subset_path),
        "--output",
        str(output_log),
        "--seed",
        str(seed),
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
            f.write(" ".join(shlex.quote(part) for part in cmd))
            f.write("\n")
    path.chmod(0o755)


def _write_manifest_csv(path: Path, entries: list[dict]) -> None:
    if not entries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "group",
        "ambiguous_ratio",
        "replace_ratio",
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
    group: str,
    rows: list[dict],
    path: Path,
    ambiguous_ratio: float | None,
    replace_ratio: float | None,
    seed: int,
) -> dict:
    write_jsonl(path, rows)
    summary = subset_summary(rows)
    scores = proxy_eval_scores(rows)
    return {
        "name": name,
        "group": group,
        "ambiguous_ratio": ambiguous_ratio,
        "replace_ratio": replace_ratio,
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
            run_dir = args.output_dir / "training_runs" / entry["name"] / f"restart_{restart_idx:02d}"
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
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/easy_role"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ambiguous-ratios",
        type=float,
        nargs="+",
        default=list(DEFAULT_AMBIGUOUS_RATIOS),
    )
    parser.add_argument(
        "--replace-ratios",
        type=float,
        nargs="+",
        default=list(DEFAULT_REPLACE_RATIOS),
    )
    parser.add_argument(
        "--core-ambiguous-ratio",
        type=float,
        default=0.17,
        help="Ambiguous core used for the easy-replacement sweep.",
    )
    parser.add_argument(
        "--easy-source",
        choices=["region", "ranked", "region_then_ranked"],
        default="region_then_ranked",
        help="How to choose easy examples for replacement.",
    )
    parser.add_argument("--skip-random-baseline", action="store_true")
    parser.add_argument("--train", action="store_true", help="Retrain on each exported subset.")
    parser.add_argument("--dataset", choices=["snli", "mnli", "qnli", "winogrande"], default="winogrande")
    parser.add_argument("--preset", default="roberta-base")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--winogrande-config", default="winogrande_xl")
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--limit-training-runs", type=int, default=0)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    add_wandb_args(parser)
    args = parser.parse_args()

    ambiguous_ratios = _validate_ratios("ambiguous", args.ambiguous_ratios)
    replace_ratios = _validate_ratios("replace", args.replace_ratios)
    _validate_ratios("core ambiguous", [args.core_ambiguous_ratio])
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
        job_type="role_easy_to_learn",
        config={
            "input": str(input_path),
            "output_dir": str(args.output_dir),
            "ambiguous_ratios": ambiguous_ratios,
            "replace_ratios": replace_ratios,
            "core_ambiguous_ratio": args.core_ambiguous_ratio,
            "easy_source": args.easy_source,
            "train": args.train,
        },
    )

    subsets_dir = args.output_dir / "subsets"
    logs_dir = args.output_dir / "command_logs"
    entries: list[dict] = []
    commands: list[list[str]] = []

    for ratio in tqdm(ambiguous_ratios, desc="ambiguous-only and random baselines"):
        slug = _ratio_slug(ratio)
        ambiguous = ambiguous_ablation_subset(rows, ambiguous_ratio=ratio, seed=args.seed)
        name = f"ambiguous_only_{slug}"
        path = subsets_dir / f"{name}.jsonl"
        entry = _entry(
            name=name,
            group="ambiguous_only",
            rows=ambiguous,
            path=path,
            ambiguous_ratio=ratio,
            replace_ratio=None,
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

        if not args.skip_random_baseline:
            random_subset = _sample_exact(rows, count=len(ambiguous), seed=args.seed + int(ratio * 10000))
            name = f"random_matched_{slug}"
            path = subsets_dir / f"{name}.jsonl"
            entry = _entry(
                name=name,
                group="random_matched",
                rows=random_subset,
                path=path,
                ambiguous_ratio=ratio,
                replace_ratio=None,
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

    core = ambiguous_ablation_subset(
        rows,
        ambiguous_ratio=args.core_ambiguous_ratio,
        seed=args.seed,
    )
    core_slug = _ratio_slug(args.core_ambiguous_ratio)
    for replace_ratio in tqdm(replace_ratios, desc="easy replacement sweep"):
        replace_slug = _ratio_slug(replace_ratio)
        mixed = _replace_with_easy(
            core,
            rows,
            replace_ratio=replace_ratio,
            seed=args.seed + int(replace_ratio * 10000),
            easy_source=args.easy_source,
        )
        name = f"ambiguous_{core_slug}_replace_easy_{replace_slug}"
        path = subsets_dir / f"{name}.jsonl"
        entry = _entry(
            name=name,
            group="easy_replacement",
            rows=mixed,
            path=path,
            ambiguous_ratio=args.core_ambiguous_ratio,
            replace_ratio=replace_ratio,
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
        "paper_task": "Role of Easy-to-Learn Instances",
        "input": str(input_path),
        "output_dir": str(args.output_dir),
        "dataset": args.dataset,
        "preset": args.preset,
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
                    f"{entry['group']}/count": entry["count"],
                    f"{entry['group']}/mean_confidence": entry["mean_confidence"],
                    f"{entry['group']}/mean_variability": entry["mean_variability"],
                    f"{entry['group']}/proxy_id_accuracy": entry["proxy_id_accuracy"],
                    f"{entry['group']}/proxy_ood_accuracy": entry["proxy_ood_accuracy"],
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
