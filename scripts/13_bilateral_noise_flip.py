#!/usr/bin/env python3
"""
Extra experiment #4: bilateral 1% label-flip (easy vs hard arms).

Extends paper §5 with matched easy/hard injection, region transition matrices,
detector cross-eval, and antisymmetric confidence shifts.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from ml_cartography.analysis.extension_figures import (
    plot_extra_04_bilateral_transitions,
    plot_extra_04_detector_cross_eval,
    plot_extra_04_recovery_bars,
    plot_extra_04_summary_table,
)
from ml_cartography.experiments.bilateral_noise import (
    bilateral_comparison,
    detector_cross_eval,
    region_transition_matrix,
    select_flip_arm,
    shift_rows_for_arm,
    summarize_arm_shift,
    train_confidence_detector,
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


def _collect_dynamics_from_logs(log_path: Path) -> list[dict]:
    from ml_cartography.core.dynamics import (
        add_epoch_observation,
        build_record,
        summarize_record,
    )

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


def _parse_gpu_list(gpus: str) -> List[int]:
    ids = [int(p.strip()) for p in gpus.split(",") if p.strip()]
    if not ids:
        raise ValueError("--gpus must list at least one GPU id")
    return ids


def _child_env(gpu_ids: List[int]) -> Dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    token, _ = resolve_hf_token(_root / "hf_credentials.txt")
    if token:
        env["HF_TOKEN"] = token
        env["HUGGING_FACE_HUB_TOKEN"] = token
    wandb_creds = load_wandb_credentials()
    if wandb_creds.get("api_key"):
        env["WANDB_API_KEY"] = wandb_creds["api_key"]
    return env


def _train_command(
    *,
    arm: str,
    args: argparse.Namespace,
    overrides_path: Path,
    run_dir: Path,
    seed: int,
    restart_idx: int,
) -> List[str]:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--dataset",
        args.dataset,
        "--preset",
        args.preset,
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--max-length",
        str(args.max_length),
        "--max-train-samples",
        str(args.max_train_samples),
        "--max-eval-samples",
        str(args.max_eval_samples),
        "--label-overrides",
        str(overrides_path),
        "--output",
        str(run_dir / "epoch_predictions.jsonl"),
        "--summary-out",
        str(run_dir / "training_summary.json"),
        "--metrics-out",
        str(run_dir / "training_metrics.jsonl"),
        "--figures-dir",
        str(run_dir / "figures"),
        "--subset-name",
        f"bilateral_{arm}_flip",
        "--subset-strategy",
        f"noise_injection_{arm}",
        "--seed",
        str(seed),
        "--wandb-project",
        args.wandb_project,
    ]
    if args.dataset == "winogrande":
        cmd.extend(["--winogrande-config", args.winogrande_config])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.model_name:
        cmd.extend(["--model-name", args.model_name])
    if args.wandb_run_name:
        cmd.extend(["--wandb-run-name", f"{args.wandb_run_name}_{arm}_r{restart_idx:02d}"])
    if args.wandb_group:
        cmd.extend(["--wandb-group", args.wandb_group])
    if args.wandb_entity:
        cmd.extend(["--wandb-entity", args.wandb_entity])
    if args.no_wandb:
        cmd.append("--no-wandb")
    if args.no_fp16:
        cmd.append("--no-fp16")
    if args.no_4bit:
        cmd.append("--no-4bit")
    if getattr(args, "multi_gpu", False):
        cmd.append("--multi-gpu")
    return cmd


def _tail_jsonl(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    last = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last


def _run_training_jobs(
    args: argparse.Namespace,
    jobs: List[Tuple[str, str, int, List[str], Path, List[int]]],
) -> Dict[str, dict]:
    """Run training jobs. Each job is (job_key, arm, restart_idx, cmd, run_dir, gpu_ids)."""
    results: Dict[str, dict] = {}
    total_epochs = len(jobs) * args.epochs
    epoch_bar = tqdm(total=total_epochs, desc="bilateral epochs", unit="ep")

    try:
        for job_key, arm, restart_idx, cmd, run_dir, gpu_ids in jobs:
            log_path = run_dir / "train.log"
            gpu_label = ",".join(str(g) for g in gpu_ids)
            print(f"\n>>> launch cuda:[{gpu_label}] [{job_key}]")
            if args.multi_gpu and len(gpu_ids) > 1:
                print(f"    DataParallel across {len(gpu_ids)} GPUs")
            last_epoch = 0
            with log_path.open("w", encoding="utf-8") as log_f:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(_root),
                    env=_child_env(gpu_ids),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                while proc.poll() is None:
                    latest = _tail_jsonl(run_dir / "training_metrics.jsonl")
                    if latest:
                        ep = int(latest.get("epoch", 0))
                        if ep > last_epoch:
                            epoch_bar.update(ep - last_epoch)
                            last_epoch = ep
                            val_acc = latest.get("val_accuracy")
                            loss = latest.get("train_loss")
                            msg = f"[epoch {ep}/{args.epochs}] {job_key} (gpus {gpu_label})"
                            if loss is not None:
                                msg += f" loss={float(loss):.4f}"
                            if val_acc is not None:
                                msg += f" val_acc={float(val_acc):.4f}"
                            tqdm.write(msg)
                            if use_wandb(args):
                                import wandb

                                wandb.log(
                                    {
                                        f"active/{job_key}/epoch": ep,
                                        f"active/{job_key}/val_accuracy": float(val_acc)
                                        if val_acc is not None
                                        else None,
                                    },
                                    step=epoch_bar.n,
                                )
                    epoch_bar.set_postfix_str(f"{job_key} e{last_epoch}/{args.epochs}", refresh=False)
                    time.sleep(5)

            rc = proc.returncode
            if rc != 0:
                tail = log_path.read_text(encoding="utf-8")[-3000:] if log_path.is_file() else ""
                raise RuntimeError(f"{job_key} failed (exit {rc}):\n{tail}")
            summary_path = run_dir / "training_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
            results[job_key] = {
                "run_dir": str(run_dir),
                "arm": arm,
                "restart_idx": restart_idx,
                **summary,
            }
            remaining = args.epochs - last_epoch
            if remaining > 0:
                epoch_bar.update(remaining)
            if use_wandb(args):
                import wandb

                wandb.log(
                    {
                        f"final/{job_key}/val_accuracy": summary.get("final_val_accuracy"),
                        f"final/{job_key}/num_train": summary.get("num_train"),
                    },
                    step=epoch_bar.n,
                )
    finally:
        epoch_bar.close()

    return results


def _load_existing_arm(arm_dir: Path, arm: str) -> Optional[dict]:
    coords = arm_dir / "cartography_coordinates.jsonl"
    log_path = arm_dir / "epoch_predictions.jsonl"
    if coords.is_file():
        return {
            "run_dir": str(arm_dir),
            "coordinates_path": str(coords),
            "log_path": str(log_path if log_path.is_file() else ""),
        }
    alt_coords = arm_dir / "noised_cartography_coordinates.jsonl"
    if alt_coords.is_file():
        return {
            "run_dir": str(arm_dir),
            "coordinates_path": str(alt_coords),
            "log_path": str(arm_dir / "noised_epoch_predictions.jsonl"),
        }
    return None


def _load_restart_run(run_dir: Path) -> Optional[dict]:
    coords_path = run_dir / "cartography_coordinates.jsonl"
    log_path = run_dir / "epoch_predictions.jsonl"
    if not coords_path.is_file() and log_path.is_file():
        coords = _collect_dynamics_from_logs(log_path)
        write_jsonl(coords_path, coords)
    if not coords_path.is_file():
        return None
    summary_path = run_dir / "training_summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )
    return {
        "run_dir": str(run_dir),
        "coordinates_path": str(coords_path),
        **summary,
    }


def _aggregate_hard_shifts(
    original_rows: list[dict],
    hard_guids: set[str],
    restart_runs: list[dict],
) -> tuple[list[dict], dict]:
    """Average per-guid shift metrics across hard-arm restarts."""
    from collections import defaultdict

    by_guid: dict[str, list[dict]] = defaultdict(list)
    for run in restart_runs:
        coords = read_jsonl(Path(run["coordinates_path"]))
        for row in shift_rows_for_arm(original_rows, coords, hard_guids, arm="hard"):
            by_guid[str(row["guid"])].append(row)

    pooled: list[dict] = []
    for guid, rows in by_guid.items():
        pooled.append(
            {
                "guid": guid,
                "arm": "hard",
                "flipped_hard": True,
                "confidence_before": float(np.mean([r["confidence_before"] for r in rows])),
                "confidence_after": float(np.mean([r["confidence_after"] for r in rows])),
                "variability_before": float(np.mean([r["variability_before"] for r in rows])),
                "variability_after": float(np.mean([r["variability_after"] for r in rows])),
                "correctness_before": float(np.mean([r["correctness_before"] for r in rows])),
                "correctness_after": float(np.mean([r["correctness_after"] for r in rows])),
                "region_before": rows[0]["region_before"],
                "n_restarts": len(rows),
            }
        )
    for row in pooled:
        row["confidence_delta"] = row["confidence_after"] - row["confidence_before"]
        row["variability_delta"] = row["variability_after"] - row["variability_before"]
        row["recovered"] = row["confidence_delta"] > 0.05
        row["degraded"] = row["confidence_delta"] < -0.05

    per_restart = [summarize_arm_shift(
        shift_rows_for_arm(
            original_rows,
            read_jsonl(Path(run["coordinates_path"])),
            hard_guids,
            arm="hard",
        ),
        arm="hard",
    ) for run in restart_runs]

    agg = summarize_arm_shift(pooled, arm="hard")
    agg["n_restarts"] = len(restart_runs)
    if per_restart:
        for key in ("confidence_delta_mean", "pct_recovered", "pct_easyward"):
            vals = [float(r[key]) for r in per_restart if key in r]
            if vals:
                agg[f"{key}_std"] = float(np.std(vals))
    return pooled, agg


def _analyze_and_plot(
    *,
    original_rows: list[dict],
    easy_shift: list[dict],
    hard_shift: list[dict],
    easy_retrained: list[dict],
    hard_retrained: list[dict],
    easy_guids: set[str],
    hard_guids: set[str],
    output_dir: Path,
) -> dict:
    easy_summary = summarize_arm_shift(easy_shift, arm="easy")
    hard_summary = summarize_arm_shift(hard_shift, arm="hard")
    comparison = bilateral_comparison(easy_summary, hard_summary)

    easy_mat = region_transition_matrix(easy_shift)
    hard_mat = region_transition_matrix(hard_shift)
    np.savetxt(output_dir / "region_transition_easy.csv", easy_mat, fmt="%d", delimiter=",")
    np.savetxt(output_dir / "region_transition_hard.csv", hard_mat, fmt="%d", delimiter=",")

    easy_det_summary, easy_predict = train_confidence_detector(
        easy_retrained, easy_guids, seed=42
    )
    hard_det_summary, hard_predict = train_confidence_detector(
        hard_retrained, hard_guids, seed=43
    )
    cross_eval = detector_cross_eval(
        original_rows,
        easy_predict,
        hard_predict,
        easy_guids,
        hard_guids,
    )

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_extra_04_bilateral_transitions(
        easy_mat, hard_mat, figures_dir / "bilateral_region_transitions.png"
    )
    plot_extra_04_detector_cross_eval(cross_eval, figures_dir / "detector_cross_eval.png")
    plot_extra_04_recovery_bars(
        easy_summary, hard_summary, comparison, figures_dir / "bilateral_recovery_bars.png"
    )
    plot_extra_04_summary_table(
        easy_summary,
        hard_summary,
        comparison,
        cross_eval,
        figures_dir / "bilateral_metrics_table.png",
    )

    analysis = {
        "easy_arm": easy_summary,
        "hard_arm": hard_summary,
        "comparison": comparison,
        "easy_detector": easy_det_summary,
        "hard_detector": hard_det_summary,
        "detector_cross_eval": cross_eval,
    }
    write_json(output_dir / "analysis_summary.json", analysis)
    write_jsonl(output_dir / "easy_shift.jsonl", easy_shift)
    write_jsonl(output_dir / "hard_shift.jsonl", hard_shift)
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/example_pipeline.json"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/bilateral_noise_flip"))
    parser.add_argument("--flip-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reuse-easy-dir",
        type=Path,
        default=Path("data/processed/noise_detection_paper"),
        help="Reuse completed paper §5 easy-flip run instead of retraining easy arm.",
    )
    parser.add_argument("--train", action="store_true", help="Run retraining for missing arms.")
    parser.add_argument("--arms", nargs="+", default=["easy", "hard"], choices=["easy", "hard"])
    parser.add_argument("--dataset", choices=["snli", "mnli", "qnli", "winogrande"], default="winogrande")
    parser.add_argument("--preset", default="roberta-large")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--winogrande-config", default="winogrande_xl")
    parser.add_argument(
        "--gpus",
        default="0,1,2,3,4",
        help="Comma-separated GPU ids; hard arm uses all of them with DataParallel.",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=1,
        help="Hard-arm training runs to average (default 1 = single multi-GPU run).",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        default=True,
        help="Spread hard-arm training across all --gpus via DataParallel (default on).",
    )
    parser.add_argument(
        "--no-multi-gpu",
        action="store_false",
        dest="multi_gpu",
        help="Train hard arm on a single GPU only.",
    )
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--analyze-only", action="store_true", help="Skip training; analyze existing arm outputs.")
    add_wandb_args(parser)
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    default_input = _root / "results/20260609_074628_snli_winogrande_roberta-large/dynamics/cartography_with_regions.jsonl"
    input_path = args.input or (default_input if default_input.is_file() else Path(cfg["coordinates_with_regions_output"]))
    if input_path.is_file():
        sample = read_jsonl(input_path)[:1]
        if sample and not str(sample[0].get("guid", "")).startswith("winogrande-train-"):
            raise ValueError(
                f"{input_path} guids do not match WinoGrande training ids "
                "(expected winogrande-train-*). Use dynamics/cartography_with_regions.jsonl."
            )
    original_rows = read_jsonl(input_path)
    if not original_rows:
        raise ValueError(f"no rows in {input_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    easy_dir = args.output_dir / "easy_arm"
    hard_dir = args.output_dir / "hard_arm"
    easy_dir.mkdir(parents=True, exist_ok=True)
    hard_dir.mkdir(parents=True, exist_ok=True)

    reuse_easy_overrides = args.reuse_easy_dir / "label_overrides.json"
    if reuse_easy_overrides.is_file():
        payload = json.loads(reuse_easy_overrides.read_text(encoding="utf-8"))
        easy_overrides = {str(k): int(v) for k, v in payload.get("label_overrides", payload).items()}
        easy_flips = read_jsonl(args.reuse_easy_dir / "flipped_examples.jsonl")
        if not easy_flips:
            easy_flips, easy_overrides = select_flip_arm(
                original_rows, arm="easy", flip_ratio=args.flip_ratio, dataset=args.dataset, seed=args.seed
            )
    else:
        easy_flips, easy_overrides = select_flip_arm(
            original_rows, arm="easy", flip_ratio=args.flip_ratio, dataset=args.dataset, seed=args.seed
        )

    hard_override_path = hard_dir / "label_overrides.json"
    if hard_override_path.is_file() and args.analyze_only:
        payload = json.loads(hard_override_path.read_text(encoding="utf-8"))
        hard_overrides = {str(k): int(v) for k, v in payload.get("label_overrides", payload).items()}
        hard_flips = read_jsonl(hard_dir / "flipped_examples.jsonl")
    else:
        hard_flips, hard_overrides = select_flip_arm(
            original_rows, arm="hard", flip_ratio=args.flip_ratio, dataset=args.dataset, seed=args.seed + 1
        )

    easy_guids = set(easy_overrides)
    hard_guids = set(hard_overrides)

    write_jsonl(easy_dir / "flipped_examples.jsonl", easy_flips)
    write_json(easy_dir / "label_overrides.json", {"label_overrides": easy_overrides})
    write_jsonl(hard_dir / "flipped_examples.jsonl", hard_flips)
    write_json(hard_dir / "label_overrides.json", {"label_overrides": hard_overrides})

    init_wandb(
        args,
        job_type="extra_bilateral_noise",
        config={
            "flip_ratio": args.flip_ratio,
            "num_easy_flipped": len(easy_flips),
            "num_hard_flipped": len(hard_flips),
            "dataset": args.dataset,
            "preset": args.preset,
        },
    )

    arm_results: Dict[str, dict] = {}

    reused_easy = _load_existing_arm(args.reuse_easy_dir, "easy")
    if reused_easy:
        arm_results["easy"] = reused_easy
        print(f"reusing easy arm from {args.reuse_easy_dir}")

    hard_restart_runs: list[dict] = []
    for restart_idx in range(args.restarts):
        run_dir = hard_dir / "training_runs" / f"restart_{restart_idx:02d}"
        loaded = _load_restart_run(run_dir)
        if loaded:
            loaded["restart_idx"] = restart_idx
            hard_restart_runs.append(loaded)
    if hard_restart_runs:
        print(f"reusing {len(hard_restart_runs)} hard-arm restart(s) from {hard_dir}")

    gpu_ids = _parse_gpu_list(args.gpus)
    hard_gpu_ids = gpu_ids if args.multi_gpu else [gpu_ids[0]]

    if args.train and not args.analyze_only:
        jobs: List[Tuple[str, str, int, List[str], Path, List[int]]] = []
        if "easy" in args.arms and "easy" not in arm_results:
            run_dir = easy_dir / "training_runs" / "restart_00"
            run_dir.mkdir(parents=True, exist_ok=True)
            easy_args = argparse.Namespace(**{**vars(args), "multi_gpu": False})
            jobs.append(
                (
                    "easy_r00",
                    "easy",
                    0,
                    _train_command(
                        arm="easy",
                        args=easy_args,
                        overrides_path=easy_dir / "label_overrides.json",
                        run_dir=run_dir,
                        seed=args.seed,
                        restart_idx=0,
                    ),
                    run_dir,
                    [gpu_ids[0]],
                )
            )
        if "hard" in args.arms:
            existing_idxs = {int(r.get("restart_idx", -1)) for r in hard_restart_runs}
            for restart_idx in range(args.restarts):
                if restart_idx in existing_idxs:
                    continue
                run_dir = hard_dir / "training_runs" / f"restart_{restart_idx:02d}"
                run_dir.mkdir(parents=True, exist_ok=True)
                jobs.append(
                    (
                        f"hard_r{restart_idx:02d}",
                        "hard",
                        restart_idx,
                        _train_command(
                            arm="hard",
                            args=args,
                            overrides_path=hard_dir / "label_overrides.json",
                            run_dir=run_dir,
                            seed=args.seed + restart_idx,
                            restart_idx=restart_idx,
                        ),
                        run_dir,
                        hard_gpu_ids,
                    )
                )
        if jobs:
            trained = _run_training_jobs(args, jobs)
            for job_key, summary in trained.items():
                run_dir = Path(summary["run_dir"])
                coords = _collect_dynamics_from_logs(run_dir / "epoch_predictions.jsonl")
                write_jsonl(run_dir / "cartography_coordinates.jsonl", coords)
                entry = {
                    "run_dir": str(run_dir),
                    "coordinates_path": str(run_dir / "cartography_coordinates.jsonl"),
                    **summary,
                }
                if summary.get("arm") == "easy":
                    arm_results["easy"] = entry
                else:
                    hard_restart_runs.append(entry)

    if hard_restart_runs:
        hard_restart_runs = sorted(hard_restart_runs, key=lambda r: int(r.get("restart_idx", 0)))
        arm_results["hard"] = {
            "restarts": hard_restart_runs,
            "n_restarts": len(hard_restart_runs),
            "coordinates_path": hard_restart_runs[0]["coordinates_path"],
            "run_dir": hard_restart_runs[0]["run_dir"],
        }

    if "easy" not in arm_results or "hard" not in arm_results:
        print("missing arm results — run with --train or provide completed easy/hard arm dirs")
        write_json(
            args.output_dir / "manifest.json",
            {
                "status": "pending_training",
                "easy_flips": len(easy_flips),
                "hard_flips": len(hard_flips),
                "arms_ready": list(arm_results.keys()),
            },
        )
        finish_wandb()
        return

    easy_coords_path = Path(arm_results["easy"]["coordinates_path"])
    hard_coords_path = Path(arm_results["hard"]["coordinates_path"])
    if not easy_coords_path.is_file():
        easy_coords_path = args.reuse_easy_dir / "noised_cartography_coordinates.jsonl"
    easy_retrained = read_jsonl(easy_coords_path)
    hard_restart_list = arm_results["hard"].get("restarts") or [arm_results["hard"]]
    hard_shift, hard_summary_agg = _aggregate_hard_shifts(
        original_rows, hard_guids, hard_restart_list
    )
    hard_retrained = read_jsonl(Path(hard_restart_list[0]["coordinates_path"]))

    easy_shift = shift_rows_for_arm(original_rows, easy_retrained, easy_guids, arm="easy")

    analysis = _analyze_and_plot(
        original_rows=original_rows,
        easy_shift=easy_shift,
        hard_shift=hard_shift,
        easy_retrained=easy_retrained,
        hard_retrained=hard_retrained,
        easy_guids=easy_guids,
        hard_guids=hard_guids,
        output_dir=args.output_dir,
    )
    analysis["hard_arm"] = hard_summary_agg
    analysis["comparison"] = bilateral_comparison(analysis["easy_arm"], hard_summary_agg)
    analysis["hard_restarts"] = [
        summarize_arm_shift(
            shift_rows_for_arm(
                original_rows,
                read_jsonl(Path(run["coordinates_path"])),
                hard_guids,
                arm="hard",
            ),
            arm="hard",
        )
        for run in hard_restart_list
    ]
    write_json(args.output_dir / "analysis_summary.json", analysis)

    manifest = {
        "extra_experiment": 4,
        "title": "Bilateral 1% label-flip mislabeling probe",
        "flip_ratio": args.flip_ratio,
        "num_easy_flipped": len(easy_flips),
        "num_hard_flipped": len(hard_flips),
        "input": str(input_path),
        "output_dir": str(args.output_dir),
        "analysis": analysis,
        "arm_results": arm_results,
        "figures": {
            "region_transitions": str(args.output_dir / "figures/bilateral_region_transitions.png"),
            "detector_cross_eval": str(args.output_dir / "figures/detector_cross_eval.png"),
            "recovery_bars": str(args.output_dir / "figures/bilateral_recovery_bars.png"),
            "metrics_table": str(args.output_dir / "figures/bilateral_metrics_table.png"),
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)

    if use_wandb(args):
        import wandb

        wandb.log(
            {
                "bilateral/antisymmetric_gap": analysis["comparison"]["antisymmetric_gap"],
                "bilateral/supports_hypothesis": float(
                    analysis["comparison"]["supports_mislabel_hypothesis"]
                ),
                "bilateral/easy_conf_delta": analysis["easy_arm"]["confidence_delta_mean"],
                "bilateral/hard_conf_delta": analysis["hard_arm"]["confidence_delta_mean"],
            }
        )

    print(f"analysis -> {args.output_dir / 'analysis_summary.json'}")
    print(f"figures -> {args.output_dir / 'figures'}")
    print(
        f"antisymmetric gap: {analysis['comparison']['antisymmetric_gap']:+.3f} | "
        f"hypothesis supported: {analysis['comparison']['supports_mislabel_hypothesis']}"
    )
    finish_wandb()


if __name__ == "__main__":
    main()
