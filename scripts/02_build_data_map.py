#!/usr/bin/env python3
"""
Experiment §2 (part 2): assign cartography regions and plot the data map.

Also writes density-style summary stats for confidence, variability, correctness.
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse
from collections import Counter

import matplotlib.pyplot as plt
from tqdm import tqdm

from ml_cartography.analysis.data_map import annotate_regions, save_data_map_plot
from ml_cartography.utils.io import read_jsonl, write_jsonl
from scripts.common import add_wandb_args, finish_wandb, init_wandb, load_pipeline_config


def _save_histogram(values: list[float], title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=30, color="#5c6bc0", alpha=0.85)
    plt.title(title)
    plt.xlabel(title.split()[0].lower())
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/example_pipeline.json"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--plot-output", type=Path, default=None)
    add_wandb_args(parser)
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    input_path = args.input or Path(cfg["coordinates_output"])
    output_path = args.output or Path(cfg["coordinates_with_regions_output"])
    plot_path = args.plot_output or Path(cfg["data_map_plot_output"])
    hist_dir = Path("data/outputs/histograms")

    init_wandb(args, job_type="build_data_map", config={"input": str(input_path)})

    rows = read_jsonl(input_path)
    tagged = []
    for row in tqdm(rows, desc="tagging regions"):
        tagged.append(dict(row))
    tagged = annotate_regions(tagged)
    write_jsonl(output_path, tagged)
    save_data_map_plot(tagged, plot_path)

    conf = [float(r["confidence"]) for r in tagged]
    var = [float(r["variability"]) for r in tagged]
    corr = [float(r["correctness"]) for r in tagged]
    region_counts = Counter(r["region"] for r in tagged)

    conf_hist = hist_dir / "confidence_hist.png"
    var_hist = hist_dir / "variability_hist.png"
    corr_hist = hist_dir / "correctness_hist.png"
    _save_histogram(conf, "Confidence distribution", conf_hist)
    _save_histogram(var, "Variability distribution", var_hist)
    _save_histogram(corr, "Correctness distribution", corr_hist)

    if not args.no_wandb:
        import wandb

        wandb.log(
            {
                "data_map": wandb.Image(str(plot_path)),
                "hist_confidence": wandb.Image(str(conf_hist)),
                "hist_variability": wandb.Image(str(var_hist)),
                "hist_correctness": wandb.Image(str(corr_hist)),
                **{f"region_count/{k}": v for k, v in region_counts.items()},
            }
        )
        wandb.save(str(output_path))

    print(f"tagged {len(tagged)} examples -> {output_path}")
    print(f"data map -> {plot_path}")
    finish_wandb()


if __name__ == "__main__":
    main()
