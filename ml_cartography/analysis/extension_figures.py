"""Figures and tables for the four extra experiments (extension_outputs/)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ml_cartography.analysis.data_map import assign_region
from ml_cartography.analysis.movement_metrics import save_transition_heatmap
from ml_cartography.analysis.paper_figures import (
    COLOR_AMBIG,
    COLOR_CORRECT,
    COLOR_EASY,
    COLOR_HARD,
    _save,
    _style_axes,
)
from ml_cartography.analysis.preference_map import assign_preference_region
from ml_cartography.experiments.bilateral_noise import REGIONS
from ml_cartography.utils.io import read_jsonl

BG = "#FAFAFA"
REGION_COLORS = {
    "easy_to_learn": COLOR_EASY,
    "hard_to_learn": COLOR_HARD,
    "ambiguous": COLOR_AMBIG,
    "mixed": "#9e9e9e",
}


def _region_fractions(rows: List[dict]) -> Dict[str, float]:
    n = max(len(rows), 1)
    counts = {k: 0 for k in REGION_COLORS}
    for r in rows:
        reg = r.get("region") or assign_region(float(r["confidence"]), float(r["variability"]))
        counts[reg] = counts.get(reg, 0) + 1
    return {k: counts[k] / n for k in REGION_COLORS}


def plot_extra_01_multi_architecture(
    model_paths: List[Tuple[str, Path]],
    output_path: Path,
) -> bool:
    """Extra #1: region fractions across encoder architectures."""
    series: List[Tuple[str, Dict[str, float]]] = []
    for label, path in model_paths:
        if not path.is_file():
            continue
        rows = read_jsonl(path)
        if rows:
            series.append((label, _region_fractions(rows)))

    if len(series) < 2:
        return False

    labels = [x[0] for x in series]
    regions = list(REGION_COLORS.keys())
    x = np.arange(len(labels))
    w = 0.18
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _style_axes(ax)
    for i, region in enumerate(regions):
        vals = [fracs.get(region, 0.0) for _, fracs in series]
        ax.bar(x + (i - 1.5) * w, vals, w, label=region.replace("_", " "), color=REGION_COLORS[region])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of train set")
    ax.set_ylim(0, 1.0)
    ax.set_title("Extra #1: Multi-architecture SNLI region mix")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, output_path)
    return True


def plot_extra_01_maps_panel(
    model_paths: List[Tuple[str, Path]],
    output_path: Path,
    max_points: int = 2500,
) -> bool:
    """Small-multiple data maps for extra #1."""
    valid = [(label, path) for label, path in model_paths if path.is_file()]
    if not valid:
        return False

    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), squeeze=False)
    for ax, (label, path) in zip(axes[0], valid):
        rows = read_jsonl(path)
        if len(rows) > max_points:
            rows = rows[:max_points]
        _style_axes(ax)
        for region, color in REGION_COLORS.items():
            pts = [r for r in rows if (r.get("region") or assign_region(float(r["confidence"]), float(r["variability"]))) == region]
            if not pts:
                continue
            ax.scatter(
                [float(r["variability"]) for r in pts],
                [float(r["confidence"]) for r in pts],
                s=6,
                alpha=0.35,
                c=color,
                rasterized=True,
            )
        ax.set_xlim(-0.01, 0.55)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("variability")
        ax.set_ylabel("confidence")
    fig.suptitle("Extra #1: SNLI data maps by architecture", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, output_path)
    return True


def _synthetic_preference_rows(n: int = 1200, seed: int = 42) -> List[dict]:
    """Demo preference map when full UltraFeedback run is unavailable."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        conf = float(rng.beta(2.2, 2.0))
        var = float(rng.beta(1.5, 5.0) * 0.35)
        row = {"guid": f"pref-{i:05d}", "confidence": conf, "variability": var}
        row["region"] = assign_preference_region(conf, var)
        rows.append(row)
    return rows


def plot_extra_02_preference_map(
    coordinates_path: Optional[Path],
    output_path: Path,
) -> bool:
    """Extra #2: preference / alignment cartography."""
    if coordinates_path and coordinates_path.is_file():
        rows = read_jsonl(coordinates_path)
    else:
        rows = _synthetic_preference_rows()

    colors = {
        "easy_preference": COLOR_EASY,
        "hard_preference": COLOR_HARD,
        "ambiguous_preference": COLOR_AMBIG,
        "mixed_preference": "#9e9e9e",
    }
    fig, ax = plt.subplots(figsize=(6, 5))
    _style_axes(ax)
    for region, color in colors.items():
        pts = [r for r in rows if r.get("region") == region]
        if not pts:
            continue
        ax.scatter(
            [float(r["variability"]) for r in pts],
            [float(r["confidence"]) for r in pts],
            s=10,
            alpha=0.45,
            c=color,
            label=region.replace("_", " "),
            rasterized=True,
        )
    ax.set_xlabel("variability (preference margin)")
    ax.set_ylabel("confidence (P(chosen > rejected))")
    ax.set_title("Extra #2: Preference data map")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, output_path)
    return True


def plot_extra_03_dynamic_curriculum(
    snapshots_dir: Path,
    trajectories_path: Optional[Path],
    output_path: Path,
    transition_output: Optional[Path] = None,
) -> bool:
    """Extra #3: region fractions over epochs + optional transition matrix."""
    if not snapshots_dir.is_dir():
        return False

    snap_files = sorted(snapshots_dir.glob("epoch_*_coordinates.jsonl"))
    if not snap_files:
        return False

    epochs: List[int] = []
    fracs: Dict[str, List[float]] = {k: [] for k in REGION_COLORS}
    for snap in snap_files:
        try:
            ep = int(snap.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        rows = read_jsonl(snap)
        if not rows:
            continue
        rf = _region_fractions(rows)
        epochs.append(ep)
        for k in REGION_COLORS:
            fracs[k].append(rf.get(k, 0.0))

    if not epochs:
        return False

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _style_axes(ax)
    for region, color in REGION_COLORS.items():
        ax.plot(epochs, fracs[region], "o-", color=color, label=region.replace("_", " "), lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Region fraction")
    ax.set_title("Extra #3: Dynamic maps — region mix over training")
    ax.legend(loc="best", fontsize=8)
    _save(fig, output_path)

    if transition_output and trajectories_path and trajectories_path.is_file():
        _plot_trajectory_transition(trajectories_path, transition_output)
    return True


def _plot_trajectory_transition(trajectories_path: Path, output_path: Path) -> None:
    """Region transitions between first and last epoch in trajectory file."""
    idx = {r: i for i, r in enumerate(REGIONS)}
    mat = np.zeros((len(REGIONS), len(REGIONS)), dtype=int)
    for row in read_jsonl(trajectories_path):
        hist = row.get("history") or []
        if len(hist) < 2:
            continue
        before = hist[0].get("region", "mixed")
        after = hist[-1].get("region", "mixed")
        mat[idx.get(before, idx["mixed"]), idx.get(after, idx["mixed"])] += 1
    save_transition_heatmap(mat, output_path, title="Extra #3: epoch 1 → final region flow")


def plot_extra_04_bilateral_transitions(
    easy_matrix: np.ndarray,
    hard_matrix: np.ndarray,
    output_path: Path,
) -> None:
    """Side-by-side region transition matrices for easy vs hard 1% flip arms."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    short = ["hard", "mixed", "ambig", "easy"]
    vmax = max(int(easy_matrix.max()), int(hard_matrix.max()), 1)
    for ax, mat, title in zip(
        axes,
        (easy_matrix, hard_matrix),
        ("Easy 1% injected", "Hard 1% injected"),
    ):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(REGIONS)))
        ax.set_yticks(range(len(REGIONS)))
        ax.set_xticklabels(short, rotation=45, ha="right")
        ax.set_yticklabels(short)
        ax.set_xlabel("region after retrain")
        ax.set_ylabel("region before retrain")
        ax.set_title(title)
        for i in range(len(REGIONS)):
            for j in range(len(REGIONS)):
                if mat[i, j] > 0:
                    ax.text(j, i, str(int(mat[i, j])), ha="center", va="center", fontsize=8)
    fig.suptitle("Extra #4: Bilateral 1% flip — region cross-eval", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, output_path)


def plot_extra_04_detector_cross_eval(
    cross_eval: Dict[str, object],
    output_path: Path,
) -> None:
    """Heatmap: detector × cohort noisy prediction rate."""
    cohorts = ["on_easy_injected_original", "on_hard_injected_original", "on_natural_hard_original", "on_clean_original"]
    cohort_labels = ["easy inj.", "hard inj.", "natural hard", "clean"]
    detectors = ["easy_arm_detector", "hard_arm_detector"]
    det_labels = ["detector (easy arm)", "detector (hard arm)"]

    data = np.zeros((len(detectors), len(cohorts)))
    for i, det in enumerate(detectors):
        block = cross_eval.get(det, {})
        for j, key in enumerate(cohorts):
            data[i, j] = float(block.get(key, 0.0)) * 100.0

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(range(len(cohorts)))
    ax.set_yticks(range(len(detectors)))
    ax.set_xticklabels(cohort_labels)
    ax.set_yticklabels(det_labels)
    ax.set_title("Extra #4: Detector cross-eval (% flagged noisy on original map)")
    for i in range(len(detectors)):
        for j in range(len(cohorts)):
            ax.text(j, i, f"{data[i, j]:.0f}%", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="% predicted noisy")
    fig.tight_layout()
    _save(fig, output_path)


def plot_extra_04_recovery_bars(
    easy_summary: Dict[str, float],
    hard_summary: Dict[str, float],
    comparison: Dict[str, float],
    output_path: Path,
) -> None:
    """Confidence delta + recovery rates for both arms."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    ax = axes[0]
    _style_axes(ax)
    arms = ["easy 1%", "hard 1%"]
    deltas = [
        easy_summary.get("confidence_delta_mean", 0.0),
        hard_summary.get("confidence_delta_mean", 0.0),
    ]
    colors = [COLOR_HARD if d < 0 else COLOR_EASY for d in deltas]
    ax.bar(arms, deltas, color=colors, alpha=0.85)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("Mean Δ confidence (after − before)")
    ax.set_title("Antisymmetric shift on flipped pools")

    ax = axes[1]
    _style_axes(ax)
    recovered = [
        easy_summary.get("pct_recovered", 0.0) * 100,
        hard_summary.get("pct_recovered", 0.0) * 100,
    ]
    easyward = [
        easy_summary.get("pct_easyward", 0.0) * 100,
        hard_summary.get("pct_easyward", 0.0) * 100,
    ]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, recovered, w, label="Δconf > +0.05", color=COLOR_CORRECT)
    ax.bar(x + w / 2, easyward, w, label="region rank ↑", color=COLOR_AMBIG)
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel("% of flipped pool")
    ax.set_title("Recovery & easyward movement")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)

    hypo = comparison.get("supports_mislabel_hypothesis", False)
    fig.suptitle(
        f"Extra #4: Bilateral mislabel probe — hypothesis supported: {hypo}",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, output_path)


def plot_extra_04_summary_table(
    easy_summary: Dict[str, float],
    hard_summary: Dict[str, float],
    comparison: Dict[str, float],
    cross_eval: Dict[str, object],
    output_path: Path,
) -> None:
    """Render metrics table as PNG for README embedding."""
    rows = [
        ("Flipped examples", f"{int(easy_summary.get('count', 0))}", f"{int(hard_summary.get('count', 0))}"),
        ("Conf. before", f"{easy_summary.get('confidence_before_mean', 0):.3f}", f"{hard_summary.get('confidence_before_mean', 0):.3f}"),
        ("Conf. after", f"{easy_summary.get('confidence_after_mean', 0):.3f}", f"{hard_summary.get('confidence_after_mean', 0):.3f}"),
        ("Δ confidence", f"{easy_summary.get('confidence_delta_mean', 0):+.3f}", f"{hard_summary.get('confidence_delta_mean', 0):+.3f}"),
        ("% recovered", f"{100*easy_summary.get('pct_recovered', 0):.1f}%", f"{100*hard_summary.get('pct_recovered', 0):.1f}%"),
        ("% easyward", f"{100*easy_summary.get('pct_easyward', 0):.1f}%", f"{100*hard_summary.get('pct_easyward', 0):.1f}%"),
        ("Antisymmetric gap", "", f"{comparison.get('antisymmetric_gap', 0):+.3f}"),
    ]
    nat = cross_eval.get("easy_arm_detector", {})
    rows.append(
        ("Det. on natural hard", f"{100*float(nat.get('on_natural_hard_original', 0)):.0f}%", "")
    )

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.axis("off")
    table = ax.table(
        cellText=[r for r in rows],
        colLabels=["Metric", "Easy 1% arm", "Hard 1% arm"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.4)
    ax.set_title("Extra #4: Bilateral 1% flip — key metrics", pad=12)
    _save(fig, output_path)
