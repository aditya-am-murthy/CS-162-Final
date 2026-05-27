#!/usr/bin/env python3
"""
Idea #1 — Preference Data Maps + instruction-tuning dynamics on 3 GPUs in parallel.

Runs three `run_cartography_experiment.py` jobs (one per GPU):

  cuda:0 → preference + distilbert   (UltraFeedback-style pairs)
  cuda:1 → preference + roberta-base
  cuda:2 → instruction + llama-3.2-1b (Alpaca-style; causal LM)

Archives to data/idea1/<task>_<preset>/ and publishes to results/.

Usage:
  python scripts/idea1_multi_gpu_suite.py
  python scripts/idea1_multi_gpu_suite.py --max-train-samples 3000 --epochs 5
  python scripts/idea1_multi_gpu_suite.py --dry-run
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
from typing import Any, Dict, List, Optional

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    import unsloth  # noqa: F401
except ImportError:
    pass

import torch

from scripts.common import add_wandb_args, load_hf_credentials, load_wandb_credentials, resolve_hf_token

EXPERIMENT_SCRIPT = _root / "scripts" / "run_cartography_experiment.py"
EXPERIMENTS_ROOT = _root / "experiments" / "runs"
DEFAULT_DATA_ROOT = _root / "data" / "idea1"

# preference uses sequence-classification heads (encoders only)
PREFERENCE_PRESETS = ["distilbert", "roberta-base"]
INSTRUCTION_PRESET = "llama-3.2-1b"

IDEA1_JOBS: List[Dict[str, Any]] = [
    {
        "label": "preference_distilbert",
        "task": "preference",
        "preset": "distilbert",
        "gpu_id": 0,
    },
    {
        "label": "preference_roberta",
        "task": "preference",
        "preset": "roberta-base",
        "gpu_id": 1,
    },
    {
        "label": "instruction_llama",
        "task": "instruction",
        "preset": INSTRUCTION_PRESET,
        "gpu_id": 2,
    },
]
NUM_GPUS = 3


def _child_env(gpu_id: int) -> Dict[str, str]:
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


def _check_gpus() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    if torch.cuda.device_count() < NUM_GPUS:
        raise RuntimeError(f"Need {NUM_GPUS} GPUs, found {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  cuda:{i} — {torch.cuda.get_device_name(i)}")


def _build_cmd(
    job: Dict[str, Any],
    args: argparse.Namespace,
) -> List[str]:
    cmd = [
        sys.executable,
        str(EXPERIMENT_SCRIPT),
        "--task",
        job["task"],
        "--preset",
        job["preset"],
        "--epochs",
        str(args.epochs),
        "--max-train-samples",
        str(args.max_train_samples),
        "--max-eval-samples",
        "0",
        "--curriculum-after-epoch",
        "0",
        "--wandb-run-name",
        f"idea1_{job['label']}",
    ]
    if args.no_wandb:
        cmd.append("--no-wandb")
    if args.no_publish:
        cmd.append("--no-publish")
    if args.wandb_project:
        cmd.extend(["--wandb-project", args.wandb_project])
    if args.wandb_entity:
        cmd.extend(["--wandb-entity", args.wandb_entity])
    return cmd


def _latest_run(task: str, preset: str) -> Optional[Path]:
    slug = f"{task}_{preset}"
    hits = sorted(
        (p for p in EXPERIMENTS_ROOT.glob(f"*_{slug}") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return hits[0] if hits else None


def _archive(task: str, preset: str, data_root: Path) -> Path:
    src = _latest_run(task, preset)
    if src is None:
        raise FileNotFoundError(f"no run for {task}_{preset}")
    dest = data_root / f"{task}_{preset}"
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
        "idea": "Idea #1",
        "task": task,
        "preset": preset,
        "experiment_run": str(src),
        "archived_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with (dest / "archive_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"archived -> {dest}")
    return dest


def _run_job(job: Dict[str, Any], args: argparse.Namespace) -> int:
    gpu_id = job["gpu_id"]
    cmd = _build_cmd(job, args)
    print(f"\n=== [{job['label']}] cuda:{gpu_id} ===")
    print(" ".join(cmd))
    if args.dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(_root), env=_child_env(gpu_id))
    if proc.returncode != 0:
        return proc.returncode
    if not args.dry_run:
        _archive(job["task"], job["preset"], args.data_root)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=3000)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    add_wandb_args(parser)
    args = parser.parse_args()

    load_hf_credentials()
    token, src = resolve_hf_token(_root / "hf_credentials.txt")
    print(f"HF credentials: {src if token else 'MISSING'}")
    print(f"\n########## Idea #1 — 3 GPUs in parallel ##########")
    _check_gpus()

    if args.dry_run:
        for job in IDEA1_JOBS:
            _run_job(job, args)
        return

    procs: List[tuple[Dict[str, Any], subprocess.Popen]] = []
    for job in IDEA1_JOBS:
        cmd = _build_cmd(job, args)
        print(f"\n>>> launch cuda:{job['gpu_id']} [{job['label']}]")
        procs.append(
            (
                job,
                subprocess.Popen(cmd, cwd=str(_root), env=_child_env(job["gpu_id"])),
            )
        )

    summary: List[Dict[str, Any]] = []
    for job, proc in procs:
        rc = proc.wait()
        entry = {"label": job["label"], "gpu_id": job["gpu_id"], "returncode": rc}
        if rc == 0:
            try:
                dest = _archive(job["task"], job["preset"], args.data_root)
                entry["data_dir"] = str(dest)
            except Exception as e:
                entry["archive_error"] = str(e)
                rc = 1
        summary.append(entry)
        if rc != 0:
            raise RuntimeError(f"Idea #1 job failed: {job['label']} (exit {rc})")

    manifest = args.data_root / "manifest.json"
    args.data_root.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nIdea #1 complete. Artifacts: {args.data_root}/")
    print(f"manifest -> {manifest}")


if __name__ == "__main__":
    main()
