#!/usr/bin/env python3
"""
Idea #2 — Dynamic cartography on DistilBERT (single GPU, W&B metrics).

Runs dynamic SNLI training with per-epoch snapshots, curriculum reweighting,
and Idea #2 movement / learnability-vs-compute metrics logged to W&B.

Usage:
  python scripts/idea2_distilbert_suite.py
  python scripts/idea2_distilbert_suite.py --max-train-samples 10000 --epochs 5
  python scripts/idea2_distilbert_suite.py --gpu-id 0 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch

from scripts.common import add_wandb_args, load_hf_credentials, load_wandb_credentials, resolve_hf_token

EXPERIMENT_SCRIPT = _root / "scripts" / "run_cartography_experiment.py"
DEFAULT_DATA_ROOT = _root / "data" / "idea2"


def _child_env(gpu_id: int) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    token, _ = resolve_hf_token(_root / "hf_credentials.txt")
    if token:
        env["HF_TOKEN"] = token
        env["HUGGING_FACE_HUB_TOKEN"] = token
    wandb_creds = load_wandb_credentials()
    if wandb_creds.get("api_key"):
        env["WANDB_API_KEY"] = wandb_creds["api_key"]
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=10000)
    parser.add_argument("--max-eval-samples", type=int, default=2000)
    parser.add_argument("--curriculum-after-epoch", type=int, default=2)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    add_wandb_args(parser)
    args = parser.parse_args()
    run_name = args.wandb_run_name or f"idea2_dynamic_distilbert_{args.epochs}ep"

    load_hf_credentials()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Idea #2 DistilBERT run")
    print(f"GPU cuda:{args.gpu_id} — {torch.cuda.get_device_name(args.gpu_id)}")

    cmd = [
        sys.executable,
        str(EXPERIMENT_SCRIPT),
        "--task",
        "dynamic",
        "--preset",
        "distilbert",
        "--epochs",
        str(args.epochs),
        "--max-train-samples",
        str(args.max_train_samples),
        "--max-eval-samples",
        str(args.max_eval_samples),
        "--curriculum-after-epoch",
        str(args.curriculum_after_epoch),
        "--wandb-run-name",
        run_name,
    ]
    if args.no_wandb:
        cmd.append("--no-wandb")
    if args.no_publish:
        cmd.append("--no-publish")
    if args.wandb_project:
        cmd.extend(["--wandb-project", args.wandb_project])
    if args.wandb_entity:
        cmd.extend(["--wandb-entity", args.wandb_entity])

    print("\n########## Idea #2 — DistilBERT dynamic cartography ##########")
    print(" ".join(cmd))
    if args.dry_run:
        return

    rc = subprocess.run(cmd, cwd=str(_root), env=_child_env(args.gpu_id)).returncode
    if rc != 0:
        raise SystemExit(rc)

    slug = "dynamic_distilbert"
    runs = sorted(
        (_root / "experiments" / "runs").glob(f"*_{slug}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if runs:
        src = runs[0]
        dest = args.data_root / f"distilbert_{args.epochs}ep"
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        for name in ("config.json", "manifest.json"):
            if (src / name).is_file():
                shutil.copy2(src / name, dest / name)
        for sub in ("dynamics", "figures", "logs", "models"):
            if (src / sub).is_dir():
                shutil.copytree(src / sub, dest / sub)
        meta = {
            "idea": "Idea #2",
            "preset": "distilbert",
            "epochs": args.epochs,
            "archived_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "experiment_run": str(src),
        }
        with (dest / "archive_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"\narchived -> {dest}")


if __name__ == "__main__":
    main()
