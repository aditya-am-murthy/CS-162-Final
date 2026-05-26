#!/usr/bin/env python3
"""
Train all four SNLI models sequentially (Colab T4 / local GPU).

Models: DistilBERT, RoBERTa-base, Llama-3.2-1B, Ministral-3B (Unsloth 4-bit).

Usage:
  python scripts/train_all_models.py --max-train-samples 10000 --epochs 5
  bash scripts/tmux_train_suite.sh   # detached sessions with wandb

Each run publishes to results/<timestamp>_<preset>/ and logs to W&B.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse
import json

from scripts.common import add_wandb_args

ALL_PRESETS = ["distilbert", "roberta-base", "llama-3.2-1b", "ministral-3b"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--presets",
        nargs="+",
        default=ALL_PRESETS,
        choices=ALL_PRESETS,
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--max-eval-samples", type=int, default=2000)
    parser.add_argument("--task", default="dynamic", choices=["snli", "dynamic"])
    parser.add_argument("--curriculum-after-epoch", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    add_wandb_args(parser)
    args = parser.parse_args()

    script = _root / "scripts" / "run_cartography_experiment.py"
    results = []

    for preset in args.presets:
        cmd = [
            sys.executable,
            str(script),
            "--task",
            args.task,
            "--preset",
            preset,
            "--epochs",
            str(args.epochs),
            "--max-train-samples",
            str(args.max_train_samples),
            "--max-eval-samples",
            str(args.max_eval_samples),
            "--curriculum-after-epoch",
            str(args.curriculum_after_epoch),
            "--wandb-run-name",
            f"snli_{preset}",
        ]
        if args.no_wandb:
            cmd.append("--no-wandb")
        if args.wandb_project:
            cmd.extend(["--wandb-project", args.wandb_project])
        if args.wandb_entity:
            cmd.extend(["--wandb-entity", args.wandb_entity])

        print(f"\n=== training {preset} ===")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, cwd=str(_root))
        results.append({"preset": preset, "returncode": proc.returncode})

    summary_path = _root / "experiments" / "train_all_models_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
