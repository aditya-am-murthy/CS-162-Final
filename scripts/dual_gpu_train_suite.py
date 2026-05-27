#!/usr/bin/env python3
"""
3-GPU training suite (local replacement for notebooks/colab_train_suite.ipynb).

Phase 1 — smoke tests (no W&B, not archived), all three GPUs in parallel:
  - DistilBERT SNLI (500 train, 2 epochs) on cuda:0
  - Llama 3.2 1B mini (200 train / 200 val, 1 epoch) on cuda:1
  - Ministral 3 3B mini (200 train / 200 val, 1 epoch) on cuda:2

Phase 2 — full SNLI dynamic training (W&B), three parallel workers:
  - cuda:0 → distilbert, roberta-base (sequential on one GPU)
  - cuda:1 → llama-3.2-1b
  - cuda:2 → ministral-3b

Trained weights + dynamics are copied to data/trained_models/<preset>/ for later
Idea #1 (preference) and Idea #2 (dynamic curriculum) runs.

Usage:
  python scripts/dual_gpu_train_suite.py
  python scripts/dual_gpu_train_suite.py --skip-smoke
  python scripts/dual_gpu_train_suite.py --smoke-only
  python scripts/dual_gpu_train_suite.py --train-only --max-train-samples 10000 --epochs 5
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
from typing import Any, Dict, List, Optional, Sequence

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    import unsloth  # noqa: F401 — before transformers when ministral runs
except ImportError:
    pass

import torch

from scripts.common import add_wandb_args, load_hf_credentials, load_wandb_credentials, resolve_hf_token

EXPERIMENT_SCRIPT = _root / "scripts" / "run_cartography_experiment.py"
EXPERIMENTS_ROOT = _root / "experiments" / "runs"
DEFAULT_DATA_ROOT = _root / "data" / "trained_models"

ALL_PRESETS = ["distilbert", "roberta-base", "llama-3.2-1b", "ministral-3b"]

# Encoders share GPU 0; each large model gets its own GPU.
GPU_PRESET_SPLIT: Dict[int, List[str]] = {
    0: ["distilbert", "roberta-base"],
    1: ["llama-3.2-1b"],
    2: ["ministral-3b"],
}
NUM_TRAIN_GPUS = len(GPU_PRESET_SPLIT)

SMOKE_JOBS: List[Dict[str, Any]] = [
    {
        "label": "distilbert_snli",
        "preset": "distilbert",
        "task": "snli",
        "max_train_samples": 500,
        "max_eval_samples": 200,
        "epochs": 2,
        "curriculum_after_epoch": 0,
        "gpu_id": 0,
    },
    {
        "label": "llama_mini",
        "preset": "llama-3.2-1b",
        "task": "dynamic",
        "max_train_samples": 200,
        "max_eval_samples": 200,
        "epochs": 1,
        "curriculum_after_epoch": 0,
        "gpu_id": 1,
    },
    {
        "label": "ministral_mini",
        "preset": "ministral-3b",
        "task": "dynamic",
        "max_train_samples": 200,
        "max_eval_samples": 200,
        "epochs": 1,
        "curriculum_after_epoch": 0,
        "gpu_id": 2,
    },
]


def _check_train_deps() -> None:
    """Llama (4-bit) and Ministral (Unsloth bnb-4bit) need bitsandbytes + unsloth."""
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "bitsandbytes is required for llama-3.2-1b and ministral-3b. "
            "Install with: pip install -U 'bitsandbytes>=0.46.1'"
        ) from e
    try:
        import unsloth  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "unsloth is required for ministral-3b (Unsloth 4-bit checkpoint). "
            "Install with: pip install unsloth"
        ) from e


def _check_gpus(required: int = NUM_TRAIN_GPUS) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. On Colab: T4 runtime + Restart session, then "
            "bash scripts/colab_setup.sh. Locally: install a CUDA-enabled PyTorch build."
        )
    count = torch.cuda.device_count()
    if count < required:
        raise RuntimeError(
            f"Need at least {required} CUDA devices, found {count}. "
            f"Available: {[torch.cuda.get_device_name(i) for i in range(count)]}"
        )
    print(f"CUDA devices ({count}):")
    for i in range(count):
        print(f"  cuda:{i} — {torch.cuda.get_device_name(i)}")


def _child_env(gpu_id: int) -> Dict[str, str]:
    """Subprocess env: pin one GPU and inject HF/W&B tokens from repo credential files."""
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


def _build_experiment_cmd(
    *,
    preset: str,
    task: str,
    max_train_samples: int,
    max_eval_samples: int,
    epochs: int,
    curriculum_after_epoch: int,
    no_wandb: bool,
    no_publish: bool,
    wandb_run_name: Optional[str],
    wandb_project: Optional[str],
    wandb_entity: Optional[str],
) -> List[str]:
    cmd = [
        sys.executable,
        str(EXPERIMENT_SCRIPT),
        "--task",
        task,
        "--preset",
        preset,
        "--epochs",
        str(epochs),
        "--max-train-samples",
        str(max_train_samples),
        "--max-eval-samples",
        str(max_eval_samples),
        "--curriculum-after-epoch",
        str(curriculum_after_epoch),
    ]
    if no_wandb:
        cmd.append("--no-wandb")
    if no_publish:
        cmd.append("--no-publish")
    if wandb_run_name:
        cmd.extend(["--wandb-run-name", wandb_run_name])
    if wandb_project:
        cmd.extend(["--wandb-project", wandb_project])
    if wandb_entity:
        cmd.extend(["--wandb-entity", wandb_entity])
    return cmd


def run_experiment(
    gpu_id: int,
    job: Dict[str, Any],
    *,
    no_wandb: bool,
    no_publish: bool,
    wandb_run_name: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    dry_run: bool = False,
    wait: bool = True,
) -> int | subprocess.Popen:
    cmd = _build_experiment_cmd(
        preset=job["preset"],
        task=job["task"],
        max_train_samples=job["max_train_samples"],
        max_eval_samples=job["max_eval_samples"],
        epochs=job["epochs"],
        curriculum_after_epoch=job.get("curriculum_after_epoch", 0),
        no_wandb=no_wandb,
        no_publish=no_publish,
        wandb_run_name=wandb_run_name,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
    )
    label = job.get("label") or job["preset"]
    print(f"\n=== [{label}] cuda:{gpu_id} ===")
    print(" ".join(cmd))
    if dry_run:
        return 0
    proc = subprocess.Popen(cmd, cwd=str(_root), env=_child_env(gpu_id))
    if not wait:
        return proc
    return proc.wait()


def _latest_experiment_dir(task: str, preset: str) -> Optional[Path]:
    slug = f"{task}_{preset}"
    candidates = sorted(
        (p for p in EXPERIMENTS_ROOT.glob(f"*_{slug}") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def archive_run_to_data(
    *,
    preset: str,
    task: str,
    data_root: Path,
    run_dir: Optional[Path] = None,
) -> Path:
    """Copy final model weights + dynamics into data/trained_models/<preset>/."""
    src = run_dir or _latest_experiment_dir(task, preset)
    if src is None or not src.is_dir():
        raise FileNotFoundError(
            f"No experiment run found for task={task!r} preset={preset!r} under {EXPERIMENTS_ROOT}"
        )

    dest = data_root / preset
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    for name in ("config.json", "manifest.json"):
        if (src / name).is_file():
            shutil.copy2(src / name, dest / name)

    for sub in ("dynamics", "figures", "logs"):
        if (src / sub).is_dir():
            shutil.copytree(src / sub, dest / sub)

    models_src = src / "models"
    if models_src.is_dir():
        shutil.copytree(models_src, dest / "models")

    meta = {
        "preset": preset,
        "task": task,
        "experiment_run": str(src),
        "archived_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "paths": {
            "root": str(dest),
            "epoch_logs": str(dest / "dynamics" / "epoch_predictions.jsonl"),
            "coordinates": str(dest / "dynamics" / "cartography_coordinates.jsonl"),
            "regions": str(dest / "dynamics" / "cartography_with_regions.jsonl"),
            "model_final": str(dest / "models" / "final"),
        },
    }
    with (dest / "archive_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"archived {preset} -> {dest}")
    return dest


def _update_suite_manifest(
    data_root: Path,
    entries: List[Dict[str, Any]],
) -> None:
    manifest_path = data_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    existing: List[Dict[str, Any]] = []
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
    existing.extend(entries)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    print(f"wrote suite manifest -> {manifest_path}")


def run_smoke_tests(args: argparse.Namespace) -> List[Dict[str, Any]]:
    print("\n########## Phase 1: smoke tests (no W&B, 3 GPUs in parallel) ##########")
    results: List[Dict[str, Any]] = []

    if args.dry_run:
        for job in SMOKE_JOBS:
            rc = run_experiment(
                job["gpu_id"],
                job,
                no_wandb=True,
                no_publish=True,
                dry_run=True,
            )
            results.append({"phase": "smoke", "label": job["label"], "returncode": rc})
        print("\nAll smoke tests passed.")
        return results

    pending: List[tuple[Dict[str, Any], subprocess.Popen]] = []
    for job in SMOKE_JOBS:
        proc = run_experiment(
            job["gpu_id"],
            job,
            no_wandb=True,
            no_publish=True,
            wait=False,
        )
        assert isinstance(proc, subprocess.Popen)
        pending.append((job, proc))

    for job, proc in pending:
        rc = proc.wait()
        results.append(
            {
                "phase": "smoke",
                "label": job["label"],
                "gpu_id": job["gpu_id"],
                "returncode": rc,
            }
        )
        if rc != 0:
            raise RuntimeError(
                f"Smoke test failed: {job['label']} on cuda:{job['gpu_id']} (exit {rc})"
            )
    print("\nAll smoke tests passed.")
    return results


def _worker_train_presets(
    gpu_id: int,
    presets: Sequence[str],
    args: argparse.Namespace,
) -> int:
    task = args.task
    archived: List[Dict[str, Any]] = []
    for preset in presets:
        job = {
            "preset": preset,
            "task": task,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "epochs": args.epochs,
            "curriculum_after_epoch": args.curriculum_after_epoch,
            "label": f"train_{preset}",
        }
        rc = run_experiment(
            gpu_id,
            job,
            no_wandb=args.no_wandb,
            no_publish=args.no_publish,
            wandb_run_name=args.wandb_run_name or f"snli_{preset}_gpu{gpu_id}",
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            dry_run=args.dry_run,
        )
        if isinstance(rc, subprocess.Popen):
            rc = rc.wait()
        if rc != 0:
            return rc
        if args.dry_run:
            continue
        dest = archive_run_to_data(
            preset=preset,
            task=task,
            data_root=args.data_root,
        )
        archived.append(
            {
                "preset": preset,
                "task": task,
                "gpu_id": gpu_id,
                "data_dir": str(dest),
            }
        )
    if archived:
        _update_suite_manifest(args.data_root, archived)
    return 0


def run_full_training(args: argparse.Namespace) -> List[Dict[str, Any]]:
    print(f"\n########## Phase 2: full training ({NUM_TRAIN_GPUS} GPUs) ##########")
    print(
        f"task={args.task} samples={args.max_train_samples} epochs={args.epochs} "
        f"curriculum_after_epoch={args.curriculum_after_epoch}"
    )
    for gpu_id, presets in sorted(GPU_PRESET_SPLIT.items()):
        print(f"  cuda:{gpu_id} -> {', '.join(presets)}")

    if args.dry_run:
        for gpu_id, presets in GPU_PRESET_SPLIT.items():
            _worker_train_presets(gpu_id, presets, args)
        return []

    procs: List[subprocess.Popen] = []
    for gpu_id, presets in GPU_PRESET_SPLIT.items():
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--gpu-id",
            str(gpu_id),
            "--presets",
            *presets,
            "--task",
            args.task,
            "--max-train-samples",
            str(args.max_train_samples),
            "--max-eval-samples",
            str(args.max_eval_samples),
            "--epochs",
            str(args.epochs),
            "--curriculum-after-epoch",
            str(args.curriculum_after_epoch),
            "--data-root",
            str(args.data_root),
        ]
        if args.no_wandb:
            cmd.append("--no-wandb")
        if args.no_publish:
            cmd.append("--no-publish")
        if args.wandb_project:
            cmd.extend(["--wandb-project", args.wandb_project])
        if args.wandb_entity:
            cmd.extend(["--wandb-entity", args.wandb_entity])
        if args.wandb_run_name:
            cmd.extend(["--wandb-run-name", args.wandb_run_name])

        print(f"\n>>> launching worker cuda:{gpu_id}: {' '.join(cmd)}")
        procs.append(subprocess.Popen(cmd, cwd=str(_root), env=_child_env(gpu_id)))

    results: List[Dict[str, Any]] = []
    for gpu_id, proc in zip(GPU_PRESET_SPLIT.keys(), procs):
        rc = proc.wait()
        results.append({"phase": "train", "gpu_id": gpu_id, "returncode": rc})
        if rc != 0:
            raise RuntimeError(f"GPU worker cuda:{gpu_id} failed (exit {rc})")

    print(f"\nFull training complete. Models saved under {args.data_root}/")
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Where to archive trained model weights + dynamics",
    )
    parser.add_argument("--task", default="dynamic", choices=["snli", "dynamic"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=10000)
    parser.add_argument("--max-eval-samples", type=int, default=2000)
    parser.add_argument("--curriculum-after-epoch", type=int, default=2)
    parser.add_argument("--no-publish", action="store_true", help="Skip results/ publish")
    add_wandb_args(parser)

    # Internal worker mode (one GPU, sequential presets).
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--presets",
        nargs="+",
        default=None,
        choices=ALL_PRESETS,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    load_hf_credentials()
    token, token_source = resolve_hf_token(_root / "hf_credentials.txt")
    if token:
        print(f"HF credentials: loaded from {token_source} (hf_credentials.txt)")
    else:
        print(
            "WARNING: no HF token in hf_credentials.txt — gated models (Llama, Ministral) will fail.",
            file=sys.stderr,
        )

    if args.worker:
        if not args.presets:
            raise SystemExit("--worker requires --presets")
        sys.exit(_worker_train_presets(args.gpu_id, args.presets, args))

    _check_gpus(required=NUM_TRAIN_GPUS)
    _check_train_deps()

    summary_path = _root / "experiments" / "dual_gpu_train_suite_summary.json"
    summary: Dict[str, Any] = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    run_smoke = not args.skip_smoke and not args.train_only
    run_train = not args.smoke_only

    if run_smoke:
        summary["smoke"] = run_smoke_tests(args)

    if run_train:
        summary["train"] = run_full_training(args)

    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote run summary -> {summary_path}")


if __name__ == "__main__":
    main()
