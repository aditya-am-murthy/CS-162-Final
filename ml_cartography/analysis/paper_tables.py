"""Render paper-style selection tables and Fig 3 from measured experiment JSON."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ml_cartography.analysis.paper_figures import (
    COLOR_AMBIG,
    COLOR_CORRECT,
    _save,
    _style_axes,
)

TABLE2_ROWS: List[Tuple[str, str]] = [
    ("full", "100% train"),
    ("random", "random"),
    ("high_correctness", "high-correctness"),
    ("high_confidence", "high-confidence"),
    ("low_variability", "low-variability"),
    ("low_correctness", "low-correctness"),
    ("low_confidence", "hard-to-learn"),
    ("high_variability", "ambiguous"),
]

TABLE34_ROWS: List[Tuple[str, str]] = [
    ("full", "100% train"),
    ("random", "random"),
    ("low_confidence", "hard-to-learn"),
    ("high_variability", "ambiguous"),
]


def _load_metrics(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_pct(value: float, std: float = 0.0) -> str:
    if std > 0:
        return f"{value * 100:.1f}$_{{{std * 100:.1f}}}$"
    return f"{value * 100:.1f}"


def _table_rows(
    metrics: dict[str, dict],
    order: List[Tuple[str, str]],
) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    for key, label in order:
        m = metrics.get(key)
        if not m:
            continue
        id_val = float(m.get("val_accuracy", 0.0))
        ood_val = float(m.get("proxy_ood_accuracy", 0.0))
        std = float(m.get("seed_std", 0.0))
        rows.append((label, _fmt_pct(id_val, std), _fmt_pct(ood_val, std)))
    return rows


def write_selection_table_tex(
    rows: List[Tuple[str, str, str]],
    output_path: Path,
    *,
    id_header: str = "ID",
    ood_header: str = "OOD",
) -> None:
    lines = [
        r"\begin{tabular}{lcc}",
        rf"Selection & {id_header} & {ood_header} \\",
        r"\hline",
    ]
    for label, id_cell, ood_cell in rows:
        lines.append(f"{label} & {id_cell} & {ood_cell} \\\\")
    lines.append(r"\end{tabular}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_selection_table_image(
    rows: List[Tuple[str, str, str]],
    output_path: Path,
    *,
    title: str,
    id_header: str = "ID",
    ood_header: str = "OOD",
) -> None:
    if not rows:
        return

    def _plain(texish: str) -> str:
        return texish.replace("$", "").replace("_", " ").replace("{", "").replace("}", "")

    cell_text = [[label, _plain(id_v), _plain(ood_v)] for label, id_v, ood_v in rows]
    n_rows = len(cell_text)
    fig_h = max(2.8, 0.38 * n_rows + 1.2)
    fig, ax = plt.subplots(figsize=(6.5, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=["Selection", id_header, ood_header],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e8eef5")
            cell.set_text_props(weight="bold")
        elif col == 0:
            cell.set_facecolor("#f7f9fc")
    ax.set_title(title, fontsize=12, pad=14)
    _save(fig, output_path)


def plot_table2_from_metrics(metrics_path: Path, output_png: Path, output_tex: Path) -> bool:
    if not metrics_path.is_file():
        return False
    metrics = _load_metrics(metrics_path)
    rows = _table_rows(metrics, TABLE2_ROWS)
    if not rows:
        return False
    render_selection_table_image(
        rows,
        output_png,
        title="Table 2 — WinoGrande selection (measured)",
        id_header="WinoG. Val. (ID)",
        ood_header="WSC (OOD)",
    )
    write_selection_table_tex(
        rows,
        output_tex,
        id_header="WinoG. Val. (ID)",
        ood_header="WSC (OOD)",
    )
    return True


def plot_table34_from_metrics(
    metrics_path: Path,
    output_png: Path,
    output_tex: Path,
    *,
    table_num: int,
    dataset: str,
    id_header: str,
    ood_header: str,
) -> bool:
    if not metrics_path.is_file():
        return False
    metrics = _load_metrics(metrics_path)
    rows = _table_rows(metrics, TABLE34_ROWS)
    if not rows:
        return False
    render_selection_table_image(
        rows,
        output_png,
        title=f"Table {table_num} — {dataset.upper()} selection (measured)",
        id_header=id_header,
        ood_header=ood_header,
    )
    write_selection_table_tex(rows, output_tex, id_header=id_header, ood_header=ood_header)
    return True


def _pct_from_subset(name: str) -> float | None:
    match = re.search(r"(\d+)pct", name)
    return int(match.group(1)) / 100.0 if match else None


def plot_fig3_from_easy_role(
    results_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> bool:
    if not results_path.is_file() or not manifest_path.is_file():
        return False

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results_blob = json.loads(results_path.read_text(encoding="utf-8"))
    results = results_blob.get("results", results_blob if isinstance(results_blob, list) else [])
    manifest_idx = {e["name"]: e for e in manifest.get("entries", [])}

    amb_points: list[tuple[float, float, float]] = []
    rand_points: list[tuple[float, float, float]] = []
    repl_points: list[tuple[float, float, float]] = []

    for row in results:
        subset = str(row.get("subset", ""))
        entry = manifest_idx.get(subset, {})
        id_acc = float(row.get("final_val_accuracy", 0.0))
        ood_acc = float(entry.get("proxy_ood_accuracy", float("nan")))
        group = str(row.get("group", ""))

        if group == "ambiguous_only":
            pct = _pct_from_subset(subset)
            if pct is not None:
                amb_points.append((pct, id_acc, ood_acc))
        elif group == "random_matched":
            pct = _pct_from_subset(subset)
            if pct is not None:
                rand_points.append((pct, id_acc, ood_acc))
        elif group == "easy_replacement":
            match = re.search(r"replace_easy_(\d+)pct", subset)
            if match:
                frac = int(match.group(1)) / 100.0
                repl_points.append((frac, id_acc, ood_acc))

    if not amb_points and not rand_points and not repl_points:
        return False

    amb_points.sort(key=lambda x: x[0], reverse=True)
    rand_points.sort(key=lambda x: x[0], reverse=True)
    repl_points.sort(key=lambda x: x[0])

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))

    ax = axes[0]
    _style_axes(ax)
    if rand_points:
        ax.plot(
            [p[0] * 100 for p in rand_points],
            [p[1] for p in rand_points],
            "o-",
            color="#888888",
            label="Random",
            lw=2,
        )
    if amb_points:
        ax.plot(
            [p[0] * 100 for p in amb_points],
            [p[1] for p in amb_points],
            "s-",
            color=COLOR_AMBIG,
            label="Top ambiguous",
            lw=2,
        )
    if amb_points or rand_points:
        ticks = sorted({int(p[0] * 100) for p in amb_points + rand_points}, reverse=True)
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel("% Train (ambiguous subset)")
    ax.set_ylabel("Val accuracy (ID)")
    ax.set_title("WinoGrande ID")
    ax.legend(fontsize=8)

    ax = axes[1]
    _style_axes(ax)
    rand_ood = [(p[0], p[2]) for p in rand_points if not np.isnan(p[2])]
    if rand_ood:
        ax.plot(
            [p[0] * 100 for p in rand_ood],
            [p[1] for p in rand_ood],
            "o-",
            color="#888888",
            label="Random",
            lw=2,
        )
    if amb_points:
        ood_amb = [(p[0], p[2]) for p in amb_points if not np.isnan(p[2])]
        if ood_amb:
            ax.plot(
                [p[0] * 100 for p in ood_amb],
                [p[1] for p in ood_amb],
                "s-",
                color=COLOR_AMBIG,
                label="Top ambiguous",
                lw=2,
            )
    if amb_points or rand_points:
        ticks = sorted({int(p[0] * 100) for p in amb_points + rand_points}, reverse=True)
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel("% Train (ambiguous subset)")
    ax.set_ylabel("WSC accuracy (OOD proxy)")
    ax.set_title("WSC OOD")
    ax.legend(fontsize=8)

    ax = axes[2]
    _style_axes(ax)
    if repl_points:
        labels = ["None", "1/10", "1/5", "1/4", "1/3", "1/2"]
        frac_to_label = {0.0: "None", 0.1: "1/10", 0.2: "1/5", 0.25: "1/4", 0.33: "1/3", 0.5: "1/2"}
        xs = [p[0] for p in repl_points]
        ys = [p[1] for p in repl_points]
        ax.plot(xs, ys, "s-", color=COLOR_CORRECT, label="Replacement (ID)", lw=2)
        repl_ood = [(p[0], p[2]) for p in repl_points if not np.isnan(p[2])]
        if repl_ood:
            ax.plot(
                [p[0] for p in repl_ood],
                [p[1] for p in repl_ood],
                "s:",
                color=COLOR_AMBIG,
                label="Replacement (OOD proxy)",
                lw=2,
            )
        ax.set_xticks([p[0] for p in repl_points])
        ax.set_xticklabels(
            [frac_to_label.get(round(p[0], 2), f"{p[0]:.0%}") for p in repl_points],
            fontsize=8,
        )
    ax.set_xlabel("Easy replacement fraction (17% ambiguous core)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Replacement ablation")
    ax.legend(fontsize=8)

    fig.suptitle("Role of easy-to-learn examples (Fig 3)", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, output_path)
    return True
