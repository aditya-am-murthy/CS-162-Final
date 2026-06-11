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
import json
import os
import random
import shlex
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    load_wandb_credentials,
    resolve_hf_token,
    use_wandb,
)

TRAIN_SCRIPT = _root / "scripts" / "train_and_collect_dynamics.py"

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


def _parse_gpu_list(gpus: str) -> List[int]:
    ids = [int(part.strip()) for part in gpus.split(",") if part.strip()]
    if not ids:
        raise ValueError("--gpus must list at least one GPU id, e.g. 0,1,2,3")
    return ids


def _child_env(gpu_id: int) -> Dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    token, _ = resolve_hf_token(_root / "hf_credentials.txt")
    if token:
        env["HF_TOKEN"] = token
        env["HUGGING_FACE_HUB_TOKEN"] = token
    wandb_creds = load_wandb_credentials()
    if wandb_creds.get("api_key"):
        env["WANDB_API_KEY"] = wandb_creds["api_key"]
    return env


def _wandb_prefix(args: argparse.Namespace) -> str:
    return args.wandb_run_name or f"easy_role_{args.dataset}"


def _wandb_group(args: argparse.Namespace) -> str:
    return args.wandb_group or _wandb_prefix(args)


def _wandb_run_name(args: argparse.Namespace, entry_name: str, restart_idx: int) -> str:
    return f"{_wandb_prefix(args)}_{entry_name}_r{restart_idx:02d}"


def _job_key(entry: dict, restart_idx: int) -> str:
    return f"{entry['name']}_r{restart_idx:02d}"


def _tail_jsonl(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    last: Optional[dict] = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last


def _sync_active_jobs(
    args: argparse.Namespace,
    active: Dict[int, Tuple[Any, ...]],
    last_epoch: Dict[str, int],
    epoch_bar: tqdm,
    jobs_done: int,
) -> None:
    payload: Dict[str, Any] = {
        "orchestrator/jobs_active": len(active),
        "orchestrator/jobs_completed": jobs_done,
    }
    postfix: List[str] = []
    for gpu_id, (_proc, entry, restart_idx, run_dir, *_rest) in active.items():
        latest = _tail_jsonl(run_dir / "training_metrics.jsonl")
        key = _job_key(entry, restart_idx)
        if not latest:
            postfix.append(f"gpu{gpu_id}:{key}:load")
            continue
        ep = int(latest.get("epoch", 0))
        prev = last_epoch.get(key, 0)
        if ep > prev:
            epoch_bar.update(ep - prev)
            last_epoch[key] = ep
            val_acc_new = latest.get("val_accuracy")
            loss_new = latest.get("train_loss")
            msg = f"[epoch {ep}/{args.epochs}] {key} (gpu {gpu_id})"
            if loss_new is not None:
                msg += f" loss={float(loss_new):.4f}"
            if val_acc_new is not None:
                msg += f" val_acc={float(val_acc_new):.4f}"
            tqdm.write(msg)
        val_acc = latest.get("val_accuracy")
        train_loss = latest.get("train_loss")
        postfix.append(f"gpu{gpu_id}:{key}e{ep}/{args.epochs}")
        payload[f"active/{key}/epoch"] = ep
        if val_acc is not None:
            payload[f"active/{key}/val_accuracy"] = float(val_acc)
        if train_loss is not None:
            payload[f"active/{key}/train_loss"] = float(train_loss)

    if postfix:
        epoch_bar.set_postfix_str(" | ".join(postfix[:5]), refresh=False)
    if use_wandb(args) and len(payload) > 2:
        import wandb

        wandb.log(payload, step=epoch_bar.n)


def _train_command(
    *,
    subset_path: Path,
    output_log: Path,
    summary_path: Path,
    metrics_path: Path,
    figures_dir: Path,
    dataset: str,
    preset: str,
    model_name: str | None,
    epochs: int,
    batch_size: int | None,
    learning_rate: float,
    max_length: int,
    max_eval_samples: int,
    winogrande_config: str,
    seed: int,
    subset_name: str,
    subset_strategy: str,
    wandb_run_name: str | None,
    wandb_group: str | None,
    wandb_project: str,
    wandb_entity: str | None,
    no_wandb: bool,
    no_fp16: bool,
    no_4bit: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--dataset",
        dataset,
        "--preset",
        preset,
        "--epochs",
        str(epochs),
        "--learning-rate",
        str(learning_rate),
        "--max-length",
        str(max_length),
        "--max-train-samples",
        "0",
        "--max-eval-samples",
        str(max_eval_samples),
        "--subset-file",
        str(subset_path),
        "--output",
        str(output_log),
        "--summary-out",
        str(summary_path),
        "--metrics-out",
        str(metrics_path),
        "--figures-dir",
        str(figures_dir),
        "--subset-name",
        subset_name,
        "--subset-strategy",
        subset_strategy,
        "--seed",
        str(seed),
        "--wandb-project",
        wandb_project,
    ]
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    if dataset == "winogrande":
        cmd.extend(["--winogrande-config", winogrande_config])
    if model_name:
        cmd.extend(["--model-name", model_name])
    if wandb_run_name:
        cmd.extend(["--wandb-run-name", wandb_run_name])
    if wandb_group:
        cmd.extend(["--wandb-group", wandb_group])
    if wandb_entity:
        cmd.extend(["--wandb-entity", wandb_entity])
    if no_wandb:
        cmd.append("--no-wandb")
    if no_fp16:
        cmd.append("--no-fp16")
    if no_4bit:
        cmd.append("--no-4bit")
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


def _training_jobs(
    args: argparse.Namespace,
    entries: list[dict],
) -> List[Tuple[dict, int, Path, Path, Path, list[str]]]:
    train_entries = entries[: args.limit_training_runs] if args.limit_training_runs else entries
    jobs: List[Tuple[dict, int, Path, Path, Path, list[str]]] = []
    for entry in train_entries:
        subset_path = Path(entry["path"])
        for restart_idx in range(args.restarts):
            run_seed = args.seed + restart_idx
            run_dir = (
                args.output_dir
                / "training_runs"
                / entry["name"]
                / f"restart_{restart_idx:02d}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            output_log = run_dir / "epoch_predictions.jsonl"
            summary_path = run_dir / "summary.json"
            metrics_path = run_dir / "training_metrics.jsonl"
            figures_dir = run_dir / "figures"
            cmd = _train_command(
                subset_path=subset_path,
                output_log=output_log,
                summary_path=summary_path,
                metrics_path=metrics_path,
                figures_dir=figures_dir,
                dataset=args.dataset,
                preset=args.preset,
                model_name=args.model_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_length=args.max_length,
                max_eval_samples=args.max_eval_samples,
                winogrande_config=args.winogrande_config,
                seed=run_seed,
                subset_name=entry["name"],
                subset_strategy=entry["group"],
                wandb_run_name=_wandb_run_name(args, entry["name"], restart_idx),
                wandb_group=_wandb_group(args),
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                no_wandb=args.no_wandb,
                no_fp16=args.no_fp16,
                no_4bit=args.no_4bit,
            )
            jobs.append((entry, restart_idx, run_dir, output_log, summary_path, cmd))
    return jobs


def _run_job_subprocess(cmd: list[str], gpu_id: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(_root),
        env=_child_env(gpu_id),
        text=True,
        capture_output=True,
    )


def _collect_job_result(
    entry: dict,
    restart_idx: int,
    run_dir: Path,
    summary_path: Path,
    gpu_id: int,
    proc: subprocess.CompletedProcess[str],
) -> dict:
    if proc.returncode != 0:
        err_tail = (proc.stderr or proc.stdout or "")[-4000:]
        if (run_dir / "train.log").is_file():
            err_tail = (run_dir / "train.log").read_text(encoding="utf-8")[-4000:]
        raise RuntimeError(
            f"training failed: {entry['name']} restart {restart_idx} on cuda:{gpu_id}\n{err_tail}"
        )
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {}
    return {
        "subset": entry["name"],
        "group": entry["group"],
        "restart": restart_idx,
        "seed": int(entry.get("seed", 0)) + restart_idx,
        "subset_path": entry["path"],
        "gpu_id": gpu_id,
        "run_dir": str(run_dir),
        **summary,
    }


def _train_subsets_parallel(
    args: argparse.Namespace,
    entries: list[dict],
) -> list[dict]:
    gpu_ids = _parse_gpu_list(args.gpus)
    jobs = _training_jobs(args, entries)
    results: list[dict] = []
    pending = list(jobs)
    active: Dict[int, Tuple[Any, ...]] = {}

    total_epochs = len(jobs) * args.epochs
    last_epoch: Dict[str, int] = {}
    jobs_done = 0

    print(f"training {len(jobs)} jobs ({total_epochs} epochs total) across GPUs {gpu_ids}")

    epoch_bar = tqdm(total=total_epochs, desc="training epochs", unit="ep")
    job_bar = tqdm(total=len(jobs), desc="jobs completed", unit="job", position=1)

    try:
        while pending or active:
            for gpu_id in gpu_ids:
                if gpu_id in active or not pending:
                    continue
                job = pending.pop(0)
                entry, restart_idx, run_dir, _output_log, summary_path, cmd = job
                log_path = run_dir / "train.log"
                print(
                    f"\n>>> launch cuda:{gpu_id} [{entry['name']} restart {restart_idx}]"
                )
                log_f = log_path.open("w", encoding="utf-8")
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(_root),
                    env=_child_env(gpu_id),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                active[gpu_id] = (
                    proc,
                    entry,
                    restart_idx,
                    run_dir,
                    summary_path,
                    job,
                    log_f,
                )

            _sync_active_jobs(args, active, last_epoch, epoch_bar, jobs_done)

            finished: List[int] = []
            for gpu_id, (
                proc,
                entry,
                restart_idx,
                run_dir,
                summary_path,
                _job,
                log_f,
            ) in active.items():
                rc = proc.poll()
                if rc is None:
                    continue
                finished.append(gpu_id)
                log_f.close()
                err_tail = ""
                if rc != 0 and (run_dir / "train.log").is_file():
                    err_tail = (run_dir / "train.log").read_text(encoding="utf-8")[-4000:]
                completed = subprocess.CompletedProcess(
                    args=proc.args,
                    returncode=rc,
                    stdout=err_tail,
                    stderr="",
                )
                result = _collect_job_result(
                    entry, restart_idx, run_dir, summary_path, gpu_id, completed
                )
                results.append(result)
                key = _job_key(entry, restart_idx)
                remaining = args.epochs - last_epoch.get(key, 0)
                if remaining > 0:
                    epoch_bar.update(remaining)
                    last_epoch[key] = args.epochs
                jobs_done += 1
                job_bar.update(1)
                if use_wandb(args):
                    import wandb

                    wandb.log(
                        {
                            "orchestrator/jobs_completed": jobs_done,
                            f"final/{key}/val_accuracy": result.get("final_val_accuracy"),
                            f"final/{key}/num_train": result.get("num_train"),
                        },
                        step=epoch_bar.n,
                    )

            for gpu_id in finished:
                del active[gpu_id]

            if active and not finished:
                time.sleep(5)
    finally:
        epoch_bar.close()
        job_bar.close()

    return results


def _train_subsets_sequential(
    args: argparse.Namespace,
    entries: list[dict],
) -> list[dict]:
    gpu_ids = _parse_gpu_list(args.gpus)
    gpu_id = gpu_ids[0]
    jobs = _training_jobs(args, entries)
    results: list[dict] = []

    for entry, restart_idx, run_dir, _output_log, summary_path, cmd in tqdm(
        jobs, desc="retraining subsets"
    ):
        print(f"\n>>> cuda:{gpu_id} [{entry['name']} restart {restart_idx}]")
        proc = _run_job_subprocess(cmd, gpu_id)
        results.append(
            _collect_job_result(entry, restart_idx, run_dir, summary_path, gpu_id, proc)
        )

    return results


def _train_subsets(args: argparse.Namespace, entries: list[dict]) -> list[dict]:
    load_hf_credentials()
    if args.sequential:
        return _train_subsets_sequential(args, entries)
    return _train_subsets_parallel(args, entries)


def _export_train_command(
    args: argparse.Namespace,
    *,
    entry: dict,
    subset_path: Path,
    run_dir: Path,
    restart_idx: int = 0,
) -> list[str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    run_seed = args.seed + restart_idx
    return _train_command(
        subset_path=subset_path,
        output_log=run_dir / "epoch_predictions.jsonl",
        summary_path=run_dir / "summary.json",
        metrics_path=run_dir / "training_metrics.jsonl",
        figures_dir=run_dir / "figures",
        dataset=args.dataset,
        preset=args.preset,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        max_eval_samples=args.max_eval_samples,
        winogrande_config=args.winogrande_config,
        seed=run_seed,
        subset_name=entry["name"],
        subset_strategy=entry["group"],
        wandb_run_name=_wandb_run_name(args, entry["name"], restart_idx),
        wandb_group=_wandb_group(args),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        no_wandb=args.no_wandb,
        no_fp16=args.no_fp16,
        no_4bit=args.no_4bit,
    )


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
    parser.add_argument(
        "--gpus",
        default="0,1,2,3,4",
        help="Comma-separated GPU ids for parallel training (e.g. 0,1,2,3,4)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run training jobs one at a time on the first GPU in --gpus",
    )
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

    orchestrator_name = (
        f"{_wandb_prefix(args)}_orchestrator" if args.train else _wandb_prefix(args)
    )
    saved_run_name = args.wandb_run_name
    args.wandb_run_name = orchestrator_name
    if not args.wandb_group:
        args.wandb_group = _wandb_group(args)
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
            "gpus": args.gpus,
            "epochs": args.epochs,
        },
    )
    args.wandb_run_name = saved_run_name

    subsets_dir = args.output_dir / "subsets"
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

        run_dir = args.output_dir / "training_runs" / name / "restart_00"
        commands.append(
            _export_train_command(
                args,
                entry=entry,
                subset_path=path,
                run_dir=run_dir,
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
            run_dir = args.output_dir / "training_runs" / name / "restart_00"
            commands.append(
                _export_train_command(
                    args,
                    entry=entry,
                    subset_path=path,
                    run_dir=run_dir,
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
        run_dir = args.output_dir / "training_runs" / name / "restart_00"
        commands.append(
            _export_train_command(
                args,
                entry=entry,
                subset_path=path,
                run_dir=run_dir,
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
        write_json(results_path, {"results": train_results, "gpus": args.gpus})
        manifest["train_results"] = train_results
        manifest["gpus"] = args.gpus
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
