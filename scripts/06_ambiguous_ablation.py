#!/usr/bin/env python3
"""
Experiment §4: role of easy-to-learn examples — ambiguous-only subset sweeps.

Sweeps ambiguous fraction (50% .. 1%) and optional easy-example replacement
within a fixed ambiguous core (default 17%).
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
    ambiguous_ablation_subset,
    proxy_eval_scores,
    replace_with_easy_examples,
    subset_summary,
)
from ml_cartography.utils.io import read_jsonl, write_jsonl
from scripts.common import add_wandb_args, finish_wandb, init_wandb, load_pipeline_config

AMBIGUOUS_RATIOS = (0.50, 0.33, 0.25, 0.17, 0.10, 0.05, 0.01)
REPLACE_RATIOS = (0.0, 0.10, 0.25, 0.50)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/example_pipeline.json"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--core-ambiguous-ratio",
        type=float,
        default=0.17,
        help="Ambiguous core size for replacement ablation (paper uses 17%)",
    )
    add_wandb_args(parser)
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    input_path = args.input or Path(cfg["coordinates_with_regions_output"])
    out_dir = Path("data/processed/ablation")

    init_wandb(
        args,
        job_type="ambiguous_ablation",
        config={"core_ambiguous_ratio": args.core_ambiguous_ratio},
    )

    rows = read_jsonl(input_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ratio in tqdm(AMBIGUOUS_RATIOS, desc="ambiguous ratio sweep"):
        subset = ambiguous_ablation_subset(rows, ambiguous_ratio=ratio, seed=args.seed)
        summary = subset_summary(subset)
        scores = proxy_eval_scores(subset)
        path = out_dir / f"ambiguous_only_{int(ratio * 100):02d}pct.jsonl"
        write_jsonl(path, subset)

        if not args.no_wandb:
            import wandb

            wandb.log(
                {
                    "ambiguous_sweep/ratio": ratio,
                    "ambiguous_sweep/count": summary["count"],
                    "ambiguous_sweep/proxy_id": scores["proxy_id_accuracy"],
                    "ambiguous_sweep/proxy_ood": scores["proxy_ood_accuracy"],
                }
            )
        print(
            f"ambiguous {ratio:.0%}: n={int(summary['count'])} "
            f"proxy_id={scores['proxy_id_accuracy']:.3f} "
            f"proxy_ood={scores['proxy_ood_accuracy']:.3f}"
        )

    core = ambiguous_ablation_subset(rows, ambiguous_ratio=args.core_ambiguous_ratio, seed=args.seed)
    for replace_ratio in tqdm(REPLACE_RATIOS, desc="easy replacement sweep"):
        mixed = replace_with_easy_examples(
            core, rows, replace_ratio=replace_ratio, seed=args.seed
        )
        summary = subset_summary(mixed)
        scores = proxy_eval_scores(mixed)
        path = out_dir / f"ambiguous17_replace_easy_{int(replace_ratio * 100)}pct.jsonl"
        write_jsonl(path, mixed)

        if not args.no_wandb:
            import wandb

            wandb.log(
                {
                    "replacement_sweep/replace_ratio": replace_ratio,
                    "replacement_sweep/count": summary["count"],
                    "replacement_sweep/proxy_id": scores["proxy_id_accuracy"],
                    "replacement_sweep/proxy_ood": scores["proxy_ood_accuracy"],
                }
            )
        print(
            f"replace easy {replace_ratio:.0%} in 17% core: n={int(summary['count'])} "
            f"proxy_id={scores['proxy_id_accuracy']:.3f} "
            f"proxy_ood={scores['proxy_ood_accuracy']:.3f}"
        )

    finish_wandb()


if __name__ == "__main__":
    main()
