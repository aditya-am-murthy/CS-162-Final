#!/usr/bin/env python3
"""Plot paper figures and tables from exported experiment metrics CSVs."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from ml_cartography.analysis.paper_figures import (
    plot_fig4_noise_shift,
    plot_fig5_agreement_heatmap,
)
from ml_cartography.analysis.paper_tables import (
    plot_fig3_from_easy_role,
    plot_table2_from_metrics,
)
from ml_cartography.utils.io import read_jsonl
from scripts.common import add_wandb_args, finish_wandb, init_wandb, use_wandb


def _mean_std(series: pd.Series) -> tuple[float, float]:
    values = series.dropna().astype(float)
    if values.empty:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values.iloc[0]), 0.0
    return float(values.mean()), float(values.std(ddof=1))


def _ratio_from_name(name: str) -> float | None:
    match = re.search(r"(\d+)pct", name)
    if match:
        return int(match.group(1)) / 100.0
    if "ambiguous_only" in name:
        return None
    return None


def plot_fig4_from_shift_jsonl(shift_path: Path, output_path: Path) -> None:
    rows = read_jsonl(shift_path)
    if rows and "confidence_before" in rows[0]:
        clean = [
            {
                "confidence": r["confidence_before"],
                "variability": r["variability_before"],
                "was_noised": False,
            }
            for r in rows
        ]
        noised = [
            {
                "confidence": r["confidence_after"],
                "variability": r["variability_after"],
                "was_noised": bool(r.get("injected_noise", r.get("was_noised"))),
            }
            for r in rows
        ]
    else:
        clean = [r for r in rows if not r.get("was_noised")]
        noised = rows
    plot_fig4_noise_shift(clean, noised, output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("results/experiment_metrics_summary.csv"),
    )
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=Path("results/experiment_metrics_history.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper_plots_from_metrics"),
    )
    parser.add_argument(
        "--fig5-input",
        type=Path,
        default=None,
        help="SNLI coordinates JSONL for Fig 5 heatmap.",
    )
    parser.add_argument(
        "--fig4-shift-jsonl",
        type=Path,
        default=Path("data/processed/noise_detection_paper/before_after_shift.jsonl"),
    )
    parser.add_argument("--experiment-tag", default=None, help="Filter summary rows to this experiment tag.")
    add_wandb_args(parser)
    args = parser.parse_args()

    init_wandb(
        args,
        job_type="plot_from_metrics_csv",
        config={"summary_csv": str(args.summary_csv), "output_dir": str(args.output_dir)},
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame()
    if args.summary_csv.is_file() and args.summary_csv.stat().st_size > 0:
        try:
            summary = pd.read_csv(args.summary_csv)
        except pd.errors.EmptyDataError:
            summary = pd.DataFrame()
    wandb_summary = Path("results/wandb_metrics_summary.csv")
    if summary.empty and wandb_summary.is_file():
        summary = pd.read_csv(wandb_summary)
    if args.experiment_tag and not summary.empty and "experiment_tag" in summary.columns:
        summary = summary[summary["experiment_tag"].astype(str).str.contains(args.experiment_tag, na=False)]

    outputs: list[Path] = []
    easy_role_results = Path("data/processed/easy_role/train_results.json")
    easy_role_manifest = Path("data/processed/easy_role/manifest.json")
    fig3_path = args.output_dir / "fig03_easy_to_learn_measured.png"
    if plot_fig3_from_easy_role(easy_role_results, easy_role_manifest, fig3_path):
        outputs.append(fig3_path)

    table2_metrics = Path("results/region_metrics_table2.json")
    table2_path = args.output_dir / "table02_winogrande_measured.png"
    table2_tex = args.output_dir / "table02_winogrande_measured.tex"
    if plot_table2_from_metrics(table2_metrics, table2_path, table2_tex):
        outputs.append(table2_path)

    if args.fig4_shift_jsonl.is_file():
        fig4_path = args.output_dir / "fig04_noise_shift_measured.png"
        plot_fig4_from_shift_jsonl(args.fig4_shift_jsonl, fig4_path)
        outputs.append(fig4_path)

    if args.fig5_input and args.fig5_input.is_file():
        fig5_path = args.output_dir / "fig05_agreement_heatmap_measured.png"
        plot_fig5_agreement_heatmap(read_jsonl(args.fig5_input), fig5_path)
        outputs.append(fig5_path)

    manifest = {
        "outputs": [str(p) for p in outputs],
        "summary_rows": int(len(summary)),
    }
    manifest_path = args.output_dir / "plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if use_wandb(args):
        import wandb

        for path in outputs:
            wandb.log({f"measured/{path.stem}": wandb.Image(str(path))})

    print(f"wrote {len(outputs)} plots to {args.output_dir.resolve()}/")
    for path in outputs:
        print(f"  {path.name}")
    finish_wandb()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
