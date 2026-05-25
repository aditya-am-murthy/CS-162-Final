#!/usr/bin/env python3
"""
Fine-tune a transformer on SNLI and write per-epoch prediction logs for cartography.

Requires GPU for reasonable speed (CPU works on small --max-train-samples).

Examples:
  # fast smoke test (~2-5 min on GPU)
  python scripts/train_and_collect_dynamics.py --preset distilbert --max-train-samples 2000 --epochs 3

  # stronger model, more data
  python scripts/train_and_collect_dynamics.py --preset roberta-base --max-train-samples 50000 --epochs 5

  # train only on a cartography subset (paper section 3 style)
  python scripts/train_and_collect_dynamics.py --preset distilbert \\
    --subset-file data/processed/selected_high_variability_33pct.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse
import json

from ml_cartography.training.glue_trainer import (
    MODEL_PRESETS,
    TrainConfig,
    load_guids_from_jsonl,
    train_and_collect_dynamics,
)
from scripts.common import add_wandb_args, finish_wandb, init_wandb, load_pipeline_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--preset",
        choices=list(MODEL_PRESETS.keys()),
        default="distilbert",
        help="Model shortcut (distilbert is smallest/fastest)",
    )
    parser.add_argument("--model-name", default=None, help="Override HuggingFace model id")
    parser.add_argument("--dataset", default="snli", choices=["snli"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=20000,
        help="Cap train size for faster runs (omit by passing 0 to use all SNLI train)",
    )
    parser.add_argument("--max-eval-samples", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--subset-file",
        type=Path,
        default=None,
        help="JSONL with guid field; train only on those examples",
    )
    parser.add_argument("--no-fp16", action="store_true", help="Disable mixed precision")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    add_wandb_args(parser)
    args = parser.parse_args()

    model_name = args.model_name or MODEL_PRESETS[args.preset]
    max_train = None if args.max_train_samples == 0 else args.max_train_samples

    output = args.output
    if output is None:
        output = Path(f"data/raw/epoch_predictions_{args.dataset}_{args.preset}.jsonl")

    subset_guids = None
    if args.subset_file:
        subset_guids = load_guids_from_jsonl(args.subset_file)

    cfg = TrainConfig(
        dataset=args.dataset,
        model_name=model_name,
        max_train_samples=max_train,
        max_eval_samples=args.max_eval_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        fp16=not args.no_fp16,
        output_logs=output,
        checkpoint_dir=args.checkpoint_dir,
        subset_guids=subset_guids,
    )

    if args.config:
        with args.config.open("r", encoding="utf-8") as f:
            file_cfg = json.load(f)
        for key in (
            "model_name",
            "epochs",
            "batch_size",
            "learning_rate",
            "max_train_samples",
            "max_eval_samples",
            "output_logs",
        ):
            if key in file_cfg:
                val = file_cfg[key]
                if key == "output_logs":
                    val = Path(val)
                setattr(cfg, key, val)

    init_wandb(
        args,
        job_type="train_collect_dynamics",
        config={
            "model": cfg.model_name,
            "epochs": cfg.epochs,
            "max_train_samples": cfg.max_train_samples,
            "subset_file": str(args.subset_file) if args.subset_file else None,
        },
    )

    print(f"model: {cfg.model_name}")
    print(f"device: will use cuda if available")
    summary = train_and_collect_dynamics(cfg)

    if not args.no_wandb:
        import wandb

        wandb.log(summary)
        wandb.save(str(cfg.output_logs))

    print(json.dumps(summary, indent=2))
    print(f"\nnext: python scripts/01_collect_dynamics.py --input {cfg.output_logs}")
    print(f"      python scripts/07_generate_insight_figures.py --input data/processed/cartography_coordinates.jsonl")
    finish_wandb()


if __name__ == "__main__":
    main()
