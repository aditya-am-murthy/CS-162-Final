#!/usr/bin/env python3
"""Generate paper-style insight figures under results/."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse

from tqdm import tqdm

from ml_cartography.analysis.data_map import annotate_regions
from ml_cartography.analysis.paper_figures import generate_all_insight_figures
from ml_cartography.data.synthetic_cartography import (
    apply_noise_shift,
    generate_paper_like_coordinates,
)
from ml_cartography.utils.io import read_jsonl, write_jsonl
from scripts.common import add_wandb_args, finish_wandb, init_wandb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="coordinates JSONL (default: generate paper-like synthetic)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument("--num-examples", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-ratio", type=float, default=0.01)
    add_wandb_args(parser)
    args = parser.parse_args()

    init_wandb(
        args,
        job_type="insight_figures",
        config={"output_dir": str(args.output_dir), "num_examples": args.num_examples},
    )

    if args.input and args.input.is_file():
        rows = annotate_regions(read_jsonl(args.input))
        print(f"loaded {len(rows)} rows from {args.input}")
    else:
        rows = generate_paper_like_coordinates(args.num_examples, seed=args.seed)
        rows = annotate_regions(rows)
        cache = Path("data/processed/paper_like_coordinates.jsonl")
        write_jsonl(cache, rows)
        print(f"generated {len(rows)} paper-like coordinates -> {cache}")

    clean, noised = apply_noise_shift(rows, noise_ratio=args.noise_ratio, seed=args.seed)

    paths = []
    for _ in tqdm([1], desc="rendering figures"):
        paths = generate_all_insight_figures(
            rows,
            args.output_dir,
            clean_for_noise=clean,
            noised_for_noise=noised,
        )

    if not args.no_wandb:
        import wandb

        for p in paths:
            wandb.log({f"results/{p.stem}": wandb.Image(str(p))})

    print(f"\nwrote {len(paths)} figures to {args.output_dir.resolve()}/")
    for p in paths:
        print(f"  {p.name}")
    finish_wandb()


if __name__ == "__main__":
    main()
