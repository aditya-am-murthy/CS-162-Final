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
    COLOR_AMBIG,
    COLOR_CORRECT,
    _save,
    _style_axes,
    plot_fig4_noise_shift,
    plot_fig5_agreement_heatmap,
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


def plot_fig3_from_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Build Fig 3 panels from script-08 easy_role summary rows."""
    work = df.copy()
    if "subset_name" not in work.columns:
        work["subset_name"] = work.get("experiment_id", pd.Series(dtype=str))

    amb_rows = work[work["subset_name"].astype(str).str.contains("ambiguous_only_", na=False)]
    rand_rows = work[work["subset_name"].astype(str).str.contains("random_matched_", na=False)]
    repl_rows = work[work["subset_name"].astype(str).str.contains("replace_easy_", na=False)]

    def _curve(frame: pd.DataFrame) -> tuple[list[float], list[float]]:
        points: list[tuple[float, float]] = []
        for _, row in frame.iterrows():
            name = str(row.get("subset_name", ""))
            ratio = _ratio_from_name(name)
            if ratio is None:
                continue
            acc = row.get("final_val_accuracy", row.get("val_accuracy"))
            if pd.isna(acc):
                continue
            points.append((ratio, float(acc)))
        points.sort(key=lambda item: item[0], reverse=True)
        if not points:
            return [], []
        xs, ys = zip(*points)
        return [x * 100 for x in xs], list(ys)

    amb_x, amb_y = _curve(amb_rows)
    rand_x, rand_y = _curve(rand_rows)

    repl_points: list[tuple[float, float, float]] = []
    for _, row in repl_rows.iterrows():
        name = str(row.get("subset_name", ""))
        match = re.search(r"replace_easy_(\d+)pct", name)
        if not match:
            continue
        frac = int(match.group(1)) / 100.0
        acc = row.get("final_val_accuracy", row.get("val_accuracy"))
        if pd.isna(acc):
            continue
        repl_points.append((frac, float(acc), float(row.get("proxy_ood_accuracy", float("nan")))))
    repl_points.sort(key=lambda item: item[0])

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    ax = axes[0]
    _style_axes(ax)
    if rand_x:
        ax.plot(rand_x, rand_y, "o-", color="#888888", label="Random", lw=2)
    if amb_x:
        ax.plot(amb_x, amb_y, "s-", color=COLOR_AMBIG, label="Top ambiguous", lw=2)
    ax.set_xscale("log")
    if amb_x or rand_x:
        ticks = sorted(set(amb_x + rand_x), reverse=True)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(t)) for t in ticks])
    ax.set_xlabel("% Train (ambiguous subset)")
    ax.set_ylabel("Val accuracy (ID proxy)")
    ax.set_title("WinoGrande ID")
    ax.legend(fontsize=8)

    ax = axes[1]
    _style_axes(ax)
    ax.text(
        0.5,
        0.5,
        "OOD (WSC) requires separate eval hook\nproxy_ood in manifest only",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
    )
    ax.set_title("WSC OOD (pending)")

    ax = axes[2]
    _style_axes(ax)
    if repl_points:
        xs = [p[0] for p in repl_points]
        ys = [p[1] for p in repl_points]
        ax.plot(xs, ys, "s-", color=COLOR_CORRECT, label="Replacement (ID)", lw=2)
        ood = [p[2] for p in repl_points if not math.isnan(p[2])]
        if ood:
            ax.plot(xs[: len(ood)], ood, "s:", color=COLOR_AMBIG, label="Replacement (OOD proxy)", lw=2)
    ax.set_xlabel("Easy replacement fraction (17% ambiguous core)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Replacement ablation")
    ax.legend(fontsize=8)

    fig.suptitle("Fig 3 from measured results", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, output_path)


def plot_table2_from_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart for Table 2 strategies from script-09 region finetune summaries."""
    work = df.copy()
    if "strategy" not in work.columns and "subset_strategy" in work.columns:
        work["strategy"] = work["subset_strategy"]
    if "strategy" not in work.columns:
        work["strategy"] = work.get("subset_name", pd.Series(dtype=str))

    grouped = (
        work.groupby("strategy", dropna=False)["final_val_accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    _style_axes(ax)
    labels = grouped["strategy"].astype(str).tolist()
    means = grouped["mean"].astype(float).tolist()
    stds = grouped["std"].fillna(0.0).astype(float).tolist()
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color=COLOR_CORRECT, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Val accuracy (mean ± std)")
    ax.set_title("Table 2 — WinoGrande 33% subset selection (measured)")
    fig.tight_layout()
    _save(fig, output_path)

    latex_path = output_path.with_suffix(".tex")
    lines = [
        r"\begin{tabular}{lcc}",
        r"Strategy & ID val acc & $n$ \\",
        r"\hline",
    ]
    for _, row in grouped.iterrows():
        lines.append(
            f"{row['strategy']} & {row['mean']:.3f} $\\pm$ {row['std']:.3f} & {int(row['count'])} \\\\"
        )
    lines.append(r"\end{tabular}")
    latex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_fig4_from_shift_jsonl(shift_path: Path, output_path: Path) -> None:
    rows = read_jsonl(shift_path)
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
    summary = pd.read_csv(args.summary_csv) if args.summary_csv.is_file() else pd.DataFrame()
    if args.experiment_tag and not summary.empty and "experiment_tag" in summary.columns:
        summary = summary[summary["experiment_tag"].astype(str).str.contains(args.experiment_tag, na=False)]

    outputs: list[Path] = []
    if not summary.empty:
        fig3_path = args.output_dir / "fig03_easy_to_learn_measured.png"
        plot_fig3_from_summary(summary, fig3_path)
        outputs.append(fig3_path)

        table2_path = args.output_dir / "table02_winogrande_measured.png"
        table2_summary = summary
        if "experiment_tag" in summary.columns:
            mask = summary["experiment_tag"].astype(str).str.contains("table2", case=False, na=False)
            if mask.any():
                table2_summary = summary[mask]
        plot_table2_from_summary(table2_summary, table2_path)
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
