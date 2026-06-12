#!/usr/bin/env python3


from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse
import json
from types import SimpleNamespace

from ml_cartography.analysis.training_plots import save_training_curve_plot
from ml_cartography.training.glue_trainer import (
    MODEL_PRESETS,
    TrainConfig,
    apply_preset_defaults,
    load_guids_from_jsonl,
    train_and_collect_dynamics,
)

from scripts.common import (
    add_wandb_args,
    finish_wandb,
    init_wandb,
    load_hf_credentials,
    use_wandb,
)


def _load_label_overrides(path: Path) -> dict[str, int]:
    if path.suffix == ".jsonl":
        overrides: dict[str, int] = {}
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                overrides[str(row["guid"])] = int(row.get("new_label", row["label"]))
        return overrides
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "label_overrides" in payload:
        payload = payload["label_overrides"]
    return {str(k): int(v) for k, v in payload.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--preset",
        choices=list(MODEL_PRESETS.keys()),
        default="distilbert",
        help="Model shortcut (distilbert, roberta-base, llama-3.2-1b, ministral-3b)",
    )
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit for large models")
    parser.add_argument("--model-name", default=None, help="Override HuggingFace model id")
    parser.add_argument(
        "--dataset",
        default="snli",
        choices=["snli", "mnli", "qnli", "winogrande"],
    )
    parser.add_argument(
        "--winogrande-config",
        default="winogrande_xl",
        help="HF config for allenai/winogrande",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument(
        "--label-overrides",
        type=Path,
        default=None,
        help="JSON/JSONL mapping guid to replacement train label",
    )
    parser.add_argument("--no-fp16", action="store_true", help="Disable mixed precision")
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Spread one training run across all visible CUDA devices (DataParallel)",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Write training summary JSON to this path",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Append per-epoch metrics JSONL (for orchestrator progress polling)",
    )
    parser.add_argument("--subset-name", default=None, help="Subset label for W&B config")
    parser.add_argument("--subset-strategy", default=None, help="Selection strategy for W&B config")
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Save per-epoch and final data maps + training curve here",
    )
    add_wandb_args(parser)
    args = parser.parse_args()
    load_hf_credentials()

    model_name = args.model_name or MODEL_PRESETS[args.preset]
    max_train = None if args.max_train_samples == 0 else args.max_train_samples
    max_eval = None if args.max_eval_samples == 0 else args.max_eval_samples

    output = args.output
    if output is None:
        output = Path(f"data/raw/epoch_predictions_{args.dataset}_{args.preset}.jsonl")

    subset_guids = None
    if args.subset_file:
        subset_guids = load_guids_from_jsonl(args.subset_file)
    label_overrides = _load_label_overrides(args.label_overrides) if args.label_overrides else None

    cfg = TrainConfig(
        dataset=args.dataset,
        model_name=model_name,
        max_train_samples=max_train,
        max_eval_samples=max_eval,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
        fp16=not args.no_fp16,
        output_logs=output,
        checkpoint_dir=args.checkpoint_dir,
        subset_guids=subset_guids,
        label_overrides=label_overrides,
        winogrande_config=args.winogrande_config,
        snapshot_dir=args.figures_dir,
        use_data_parallel=args.multi_gpu,
    )
    cfg = apply_preset_defaults(cfg, args.preset)
    if args.no_4bit:
        cfg.load_in_4bit = False

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
            "dataset": cfg.dataset,
            "epochs": cfg.epochs,
            "max_train_samples": cfg.max_train_samples,
            "subset_file": str(args.subset_file) if args.subset_file else None,
            "label_overrides": str(args.label_overrides) if args.label_overrides else None,
            "subset_name": args.subset_name,
            "subset_strategy": args.subset_strategy,
            "seed": cfg.seed,
        },
    )

    print(f"model: {cfg.model_name}")
    print(f"device: will use cuda if available")
    wandb_run = None
    if use_wandb(args):
        import wandb

        wandb_run = wandb.run
    summary = train_and_collect_dynamics(
        cfg,
        wandb_run=wandb_run,
        metrics_log=args.metrics_out,
    )

    if use_wandb(args):
        import wandb

        wandb.log({"final_val_accuracy": summary.get("final_val_accuracy"), **summary})
        wandb.save(str(cfg.output_logs))
        if args.metrics_out and args.metrics_out.is_file():
            wandb.save(str(args.metrics_out))

    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    if args.figures_dir and cfg.output_logs.is_file():
        from scripts.run_cartography_experiment import (
            _build_figures,
            _collect_dynamics_from_logs,
        )

        args.figures_dir.mkdir(parents=True, exist_ok=True)
        coords = _collect_dynamics_from_logs(cfg.output_logs)
        exp = SimpleNamespace(
            figures_dir=args.figures_dir,
            regions_path=lambda: args.figures_dir / "cartography_with_regions.jsonl",
        )
        _build_figures(exp, coords, task="snli", dataset=args.dataset)
        if args.metrics_out and args.metrics_out.is_file():
            subset_label = args.subset_name or args.subset_strategy or args.dataset
            save_training_curve_plot(
                args.metrics_out,
                args.figures_dir / "training_curve.png",
                title=f"{subset_label} — training curve",
            )
        print(f"figures saved under {args.figures_dir}")

    print(json.dumps(summary, indent=2))
    print(f"\nnext: python scripts/01_collect_dynamics.py --input {cfg.output_logs}")
    print(f"      python scripts/07_generate_insight_figures.py --input data/processed/cartography_coordinates.jsonl")
    finish_wandb()


if __name__ == "__main__":
    main()
