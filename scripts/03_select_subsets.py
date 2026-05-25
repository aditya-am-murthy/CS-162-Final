#!/usr/bin/env python3
"""
Experiment §3: data selection by cartography strategy (top 33% by default).

Trains fresh models on each subset in the full paper; this script materializes
subsets and logs proxy ID/OOD metrics for monitoring. Re-train with your own
trainer using the exported guid list.
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse

from tqdm import tqdm

from ml_cartography.experiments.selection import (
    SELECTION_STRATEGIES,
    proxy_eval_scores,
    select_by_strategy,
    subset_summary,
)
from ml_cartography.utils.io import read_jsonl, write_jsonl
from scripts.common import add_wandb_args, finish_wandb, init_wandb, load_pipeline_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/example_pipeline.json"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--strategy",
        choices=SELECTION_STRATEGIES,
        default="region_ambiguous",
        help="Selection strategy from experiments.md §3",
    )
    parser.add_argument("--keep-ratio", type=float, default=0.33)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-all-strategies",
        action="store_true",
        help="Run every §3 strategy and log comparison table to W&B",
    )
    add_wandb_args(parser)
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    input_path = args.input or Path(cfg["coordinates_with_regions_output"])
    output_path = args.output or Path(
        f"data/processed/selected_{args.strategy}_{int(args.keep_ratio * 100)}pct.jsonl"
    )

    rows = read_jsonl(input_path)
    strategies = list(SELECTION_STRATEGIES) if args.run_all_strategies else [args.strategy]

    init_wandb(
        args,
        job_type="select_subsets",
        config={
            "strategies": strategies,
            "keep_ratio": args.keep_ratio,
            "input": str(input_path),
        },
    )

    for strategy in tqdm(strategies, desc="selection strategies"):
        subset = select_by_strategy(rows, strategy=strategy, keep_ratio=args.keep_ratio, seed=args.seed)
        summary = subset_summary(subset)
        scores = proxy_eval_scores(subset)

        out = output_path
        if args.run_all_strategies:
            out = Path(f"data/processed/selected_{strategy}_{int(args.keep_ratio * 100)}pct.jsonl")
        write_jsonl(out, subset)

        if not args.no_wandb:
            import wandb

            wandb.log(
                {
                    f"{strategy}/subset_count": summary["count"],
                    f"{strategy}/mean_confidence": summary["mean_confidence"],
                    f"{strategy}/mean_variability": summary["mean_variability"],
                    f"{strategy}/mean_correctness": summary["mean_correctness"],
                    f"{strategy}/proxy_id_accuracy": scores["proxy_id_accuracy"],
                    f"{strategy}/proxy_ood_accuracy": scores["proxy_ood_accuracy"],
                }
            )

        print(
            f"[{strategy}] n={int(summary['count'])} "
            f"proxy_id={scores['proxy_id_accuracy']:.3f} "
            f"proxy_ood={scores['proxy_ood_accuracy']:.3f} -> {out}"
        )

    finish_wandb()


if __name__ == "__main__":
    main()
