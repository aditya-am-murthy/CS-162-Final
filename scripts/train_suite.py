#!/usr/bin/env python3
"""
Full training suite: SNLI encoders + Llama/Mistral + preference RLHF + dynamic maps.

Writes timestamped runs under experiments/runs/<timestamp>_<task>/
and publishes artifacts to results/<timestamp>_<task>/.

Examples:
  python scripts/train_suite.py --all
  python scripts/train_suite.py --snli-encoders --dynamic
  python scripts/train_suite.py --only snli_distilbert --max-train-samples 2000 --epochs 3
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse
import json
from typing import Dict, List

from ml_cartography.training.causal_lm_trainer import (
    CAUSAL_PRESETS,
    CausalTrainConfig,
    train_causal_dynamics,
)
from ml_cartography.training.experiment_run import ExperimentPaths
from ml_cartography.training.finalize_run import finalize_experiment
from ml_cartography.training.glue_trainer import MODEL_PRESETS, TrainConfig, train_and_collect_dynamics
from ml_cartography.training.preference_trainer import (
    PreferenceTrainConfig,
    train_preference_dynamics,
)
from scripts.common import add_wandb_args, finish_wandb, init_wandb

SNLI_ENCODER_JOBS = {
    "snli_distilbert": {"preset": "distilbert", "batch_size": 32},
    "snli_roberta_base": {"preset": "roberta-base", "batch_size": 16},
}

SNLI_CAUSAL_JOBS = {
    "snli_llama_3_2_1b": {"preset": "llama-3.2-1b", "batch_size": 4},
    "snli_mistral_7b": {"preset": "mistral-7b", "batch_size": 2},
}


def _plan_jobs(args: argparse.Namespace) -> List[str]:
    if args.only:
        return [args.only]
    planned: List[str] = []
    if args.all or args.snli_encoders:
        planned.extend(SNLI_ENCODER_JOBS.keys())
    if args.all or args.snli_causal:
        planned.extend(SNLI_CAUSAL_JOBS.keys())
    if args.all or args.preference:
        planned.append("preference_ultrafeedback")
    if args.all or args.instruction:
        planned.append("instruction_alpaca")
    return planned


def _run_snli_encoder(job_id: str, preset: str, args: argparse.Namespace, dynamic: bool) -> Dict:
    paths = ExperimentPaths.create(job_id)
    cfg = TrainConfig(
        model_name=MODEL_PRESETS[preset],
        max_train_samples=None if args.max_train_samples == 0 else args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        epochs=args.epochs,
        batch_size=SNLI_ENCODER_JOBS[job_id]["batch_size"],
        fp16=not args.no_fp16,
        paths=paths,
        map_interval=args.map_interval,
        curriculum_enabled=dynamic,
    )
    paths.write_config(
        {
            "job_id": job_id,
            "type": "snli_encoder",
            "preset": preset,
            "dynamic_cartography": dynamic,
        }
    )
    summary = train_and_collect_dynamics(cfg)
    summary["run_id"] = paths.run_id
    finalize_experiment(paths, task_type="classification")
    return summary


def _run_snli_causal(job_id: str, preset: str, args: argparse.Namespace, dynamic: bool) -> Dict:
    paths = ExperimentPaths.create(job_id)
    cfg = CausalTrainConfig(
        model_name=CAUSAL_PRESETS[preset],
        task="snli",
        max_train_samples=None if args.max_train_samples == 0 else args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        epochs=args.epochs,
        batch_size=SNLI_CAUSAL_JOBS[job_id]["batch_size"],
        load_in_4bit=not args.no_4bit,
        use_lora=not args.no_lora,
        paths=paths,
        map_interval=args.map_interval,
        curriculum_enabled=dynamic,
    )
    paths.write_config({"job_id": job_id, "type": "snli_causal_lora", "preset": preset})
    summary = train_causal_dynamics(cfg)
    summary["run_id"] = paths.run_id
    finalize_experiment(paths, task_type="causal_snli")
    return summary


def _run_preference(args: argparse.Namespace, dynamic: bool) -> Dict:
    paths = ExperimentPaths.create("preference_ultrafeedback")
    cfg = PreferenceTrainConfig(
        model_name=args.preference_model or CAUSAL_PRESETS["llama-3.2-1b"],
        max_train_samples=args.preference_samples,
        epochs=args.epochs,
        load_in_4bit=not args.no_4bit,
        use_lora=not args.no_lora,
        paths=paths,
        map_interval=args.map_interval,
        curriculum_enabled=dynamic,
    )
    paths.write_config({"job_id": "preference_ultrafeedback", "type": "preference_rlhf", "idea": "Idea #1"})
    summary = train_preference_dynamics(cfg)
    summary["run_id"] = paths.run_id
    finalize_experiment(paths, task_type="preference")
    return summary


def _run_instruction(args: argparse.Namespace, dynamic: bool) -> Dict:
    paths = ExperimentPaths.create("instruction_alpaca")
    cfg = CausalTrainConfig(
        model_name=args.preference_model or CAUSAL_PRESETS["llama-3.2-1b"],
        task="alpaca",
        max_train_samples=args.instruction_samples,
        epochs=args.epochs,
        paths=paths,
        map_interval=args.map_interval,
        curriculum_enabled=dynamic,
        load_in_4bit=not args.no_4bit,
    )
    paths.write_config({"job_id": "instruction_alpaca", "type": "instruction_tuning", "idea": "Idea #1"})
    summary = train_causal_dynamics(cfg)
    summary["run_id"] = paths.run_id
    finalize_experiment(paths, task_type="instruction")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--snli-encoders", action="store_true")
    parser.add_argument("--snli-causal", action="store_true")
    parser.add_argument("--preference", action="store_true")
    parser.add_argument("--instruction", action="store_true")
    parser.add_argument("--dynamic", action="store_true", help="Idea #2: snapshots + curriculum")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--max-eval-samples", type=int, default=2000)
    parser.add_argument("--map-interval", type=int, default=1)
    parser.add_argument("--preference-samples", type=int, default=4000)
    parser.add_argument("--instruction-samples", type=int, default=4000)
    parser.add_argument("--preference-model", default=None)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--no-lora", action="store_true")
    add_wandb_args(parser)
    args = parser.parse_args()

    jobs = _plan_jobs(args)
    if not jobs:
        parser.error("Pick jobs: --all, --snli-encoders, --snli-causal, --preference, --instruction, or --only")

    init_wandb(args, job_type="train_suite", config={"jobs": jobs})
    dynamic = args.dynamic or args.all
    summaries: Dict[str, Dict] = {}

    for job in jobs:
        print(f"\n========== {job} ==========")
        if job in SNLI_ENCODER_JOBS:
            summaries[job] = _run_snli_encoder(
                job, SNLI_ENCODER_JOBS[job]["preset"], args, dynamic
            )
        elif job in SNLI_CAUSAL_JOBS:
            summaries[job] = _run_snli_causal(
                job, SNLI_CAUSAL_JOBS[job]["preset"], args, dynamic
            )
        elif job == "preference_ultrafeedback":
            summaries[job] = _run_preference(args, dynamic)
        elif job == "instruction_alpaca":
            summaries[job] = _run_instruction(args, dynamic)
        else:
            raise ValueError(f"unknown job id: {job}")

    index_path = _root / "results" / "experiment_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index: List[Dict] = []
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
    for job, summary in summaries.items():
        index.append({"job": job, "run_id": summary.get("run_id"), "summary": summary})
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"\nDone. Runs published under results/<timestamp>_<task>/")
    print(f"Index: {index_path}")
    finish_wandb()


if __name__ == "__main__":
    main()
