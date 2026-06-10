"""Tools for plotting and slicing cartography data maps."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Literal, Tuple

RegionMode = Literal["adaptive", "absolute", "equal_thirds"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIR = _REPO_ROOT / ".cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

# Paper-style subset fractions (§3): ambiguous ≈ top third by variability;
# easy/hard are the confident vs struggling tails within a low-variability band.
REGION_QUANTILES = {
    "ambiguous_var_rank_min": 0.80,
    "low_variability_rank_max": 0.40,
    "easy_conf_rank_min": 0.75,
    "hard_conf_rank_max": 0.25,
}

# Mutually exclusive ~30% / 30% / 30% of examples (remainder → mixed).
EQUAL_THIRDS_FRACTION = 0.30

# Legacy absolute cutoffs (SNLI-like); only used when explicitly passed to annotate_regions.
DEFAULT_THRESHOLDS = {
    "easy_confidence_min": 0.75,
    "hard_confidence_max": 0.35,
    "low_variability_max": 0.10,
    "ambiguous_variability_min": 0.18,
}


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    n = len(values)
    if n <= 1:
        return np.zeros(n, dtype=float)
    return (rankdata(values, method="average") - 1.0) / (n - 1.0)


def fit_region_thresholds(
    rows: List[Dict],
    region_mode: RegionMode = "adaptive",
) -> Dict[str, float]:
    """
    Guide-line cutoffs for plot overlays.

    ``absolute`` returns paper-style fixed axis limits; other modes return
    quantiles aligned with the active rank assignment rule.
    """
    if not rows:
        return dict(DEFAULT_THRESHOLDS)

    if region_mode == "absolute":
        return dict(DEFAULT_THRESHOLDS)

    conf = np.array([float(r["confidence"]) for r in rows], dtype=float)
    var = np.array([float(r["variability"]) for r in rows], dtype=float)

    if region_mode == "equal_thirds":
        f = EQUAL_THIRDS_FRACTION
        return {
            "easy_confidence_min": float(np.quantile(conf, 1.0 - f)),
            "hard_confidence_max": float(np.quantile(conf, f)),
            "low_variability_max": float(np.quantile(var, 1.0 - f)),
            "ambiguous_variability_min": float(np.quantile(var, 1.0 - f)),
        }

    var_rank = _percentile_ranks(var)
    low_var = var_rank <= REGION_QUANTILES["low_variability_rank_max"]
    conf_low = conf[low_var] if low_var.any() else conf

    return {
        "easy_confidence_min": float(
            np.quantile(conf_low, REGION_QUANTILES["easy_conf_rank_min"])
        ),
        "hard_confidence_max": float(
            np.quantile(conf_low, REGION_QUANTILES["hard_conf_rank_max"])
        ),
        "low_variability_max": float(
            np.quantile(var, REGION_QUANTILES["low_variability_rank_max"])
        ),
        "ambiguous_variability_min": float(
            np.quantile(var, REGION_QUANTILES["ambiguous_var_rank_min"])
        ),
    }


def assign_region(
    confidence: float,
    variability: float,
    thresholds: Dict[str, float] | None = None,
) -> str:
    """Assign a region using absolute axis cutoffs (legacy / explicit override)."""
    limits = thresholds or DEFAULT_THRESHOLDS
    if (
        confidence >= limits["easy_confidence_min"]
        and variability <= limits["low_variability_max"]
    ):
        return "easy_to_learn"
    if (
        confidence <= limits["hard_confidence_max"]
        and variability <= limits["low_variability_max"]
    ):
        return "hard_to_learn"
    if variability >= limits["ambiguous_variability_min"]:
        return "ambiguous"
    return "mixed"


def _assign_region_adaptive(
    var_rank: float,
    conf_rank_in_low_var: float | None,
    *,
    in_low_var: bool,
) -> str:
    q = REGION_QUANTILES
    if in_low_var and conf_rank_in_low_var is not None:
        if conf_rank_in_low_var >= q["easy_conf_rank_min"]:
            return "easy_to_learn"
        if conf_rank_in_low_var <= q["hard_conf_rank_max"]:
            return "hard_to_learn"
    if var_rank >= q["ambiguous_var_rank_min"]:
        return "ambiguous"
    return "mixed"


def _annotate_regions_adaptive(rows: List[Dict]) -> List[Dict]:
    conf = np.array([float(r["confidence"]) for r in rows], dtype=float)
    var = np.array([float(r["variability"]) for r in rows], dtype=float)
    var_rank = _percentile_ranks(var)
    low_var = var_rank <= REGION_QUANTILES["low_variability_rank_max"]
    conf_rank_local = np.full(len(rows), np.nan, dtype=float)
    if low_var.any():
        conf_rank_local[low_var] = _percentile_ranks(conf[low_var])

    tagged = []
    for i, row in enumerate(rows):
        new_row = dict(row)
        new_row["region"] = _assign_region_adaptive(
            float(var_rank[i]),
            None if np.isnan(conf_rank_local[i]) else float(conf_rank_local[i]),
            in_low_var=bool(low_var[i]),
        )
        tagged.append(new_row)
    return tagged


def _annotate_regions_absolute(
    rows: List[Dict],
    thresholds: Dict[str, float] | None = None,
) -> List[Dict]:
    limits = thresholds or DEFAULT_THRESHOLDS
    tagged: List[Dict] = []
    for row in rows:
        new_row = dict(row)
        new_row["region"] = assign_region(
            float(row["confidence"]),
            float(row["variability"]),
            thresholds=limits,
        )
        tagged.append(new_row)
    return tagged


def _annotate_regions_equal_thirds(rows: List[Dict]) -> List[Dict]:
    """~30% ambiguous (highest var), ~30% easy / ~30% hard from remainder; rest mixed."""
    n = len(rows)
    if n == 0:
        return []

    conf = np.array([float(r["confidence"]) for r in rows], dtype=float)
    var = np.array([float(r["variability"]) for r in rows], dtype=float)
    k = max(1, int(round(EQUAL_THIRDS_FRACTION * n)))

    regions = ["mixed"] * n
    by_var = np.argsort(-var)
    ambiguous_idx = set(by_var[:k].tolist())

    rest = [i for i in range(n) if i not in ambiguous_idx]
    by_conf_high = sorted(rest, key=lambda i: conf[i], reverse=True)
    easy_idx = set(by_conf_high[:k])

    rest2 = [i for i in rest if i not in easy_idx]
    by_conf_low = sorted(rest2, key=lambda i: conf[i])
    hard_idx = set(by_conf_low[: min(k, len(rest2))])

    for i in ambiguous_idx:
        regions[i] = "ambiguous"
    for i in easy_idx:
        regions[i] = "easy_to_learn"
    for i in hard_idx:
        regions[i] = "hard_to_learn"

    tagged = []
    for row, region in zip(rows, regions):
        new_row = dict(row)
        new_row["region"] = region
        tagged.append(new_row)
    return tagged


def annotate_regions(
    rows: List[Dict],
    thresholds: Dict[str, float] | None = None,
    *,
    region_mode: RegionMode = "adaptive",
) -> List[Dict]:
    """
    Tag each row with a cartography region.

    region_mode:
      - ``adaptive`` — rank percentiles (default pipeline)
      - ``absolute`` — fixed axis cutoffs (DEFAULT_THRESHOLDS unless thresholds set)
      - ``equal_thirds`` — ~30% ambiguous / easy / hard (mutually exclusive)
    """
    if thresholds is not None:
        return _annotate_regions_absolute(rows, thresholds)

    if region_mode == "absolute":
        return _annotate_regions_absolute(rows, DEFAULT_THRESHOLDS)
    if region_mode == "equal_thirds":
        return _annotate_regions_equal_thirds(rows)
    return _annotate_regions_adaptive(rows)


def summarize_regions(rows: List[Dict], thresholds: Dict[str, float]) -> Dict[str, object]:
    counts: Dict[str, int] = {
        "easy_to_learn": 0,
        "hard_to_learn": 0,
        "ambiguous": 0,
        "mixed": 0,
    }
    for row in rows:
        counts[row["region"]] = counts.get(row["region"], 0) + 1

    total = max(1, len(rows))
    shares = {key: value / total for key, value in counts.items()}
    return {
        "thresholds": thresholds,
        "counts": counts,
        "shares": shares,
        "num_examples": len(rows),
    }


def prepare_region_annotations(
    rows: List[Dict],
    thresholds: Dict[str, float] | None = None,
    *,
    region_mode: RegionMode = "adaptive",
) -> Tuple[List[Dict], Dict[str, object]]:
    if thresholds is not None:
        tagged = annotate_regions(rows, thresholds=thresholds)
        limits = thresholds
    else:
        tagged = annotate_regions(rows, region_mode=region_mode)
        limits = fit_region_thresholds(rows, region_mode=region_mode)
    summary = summarize_regions(tagged, thresholds=limits)
    summary["region_mode"] = region_mode if thresholds is None else "absolute"
    return tagged, summary


REGION_DISPLAY_NAMES = {
    "easy_to_learn": "Easy",
    "hard_to_learn": "Hard",
    "ambiguous": "Ambiguous",
    "mixed": "Mixed",
    "correct": "Mostly correct",
    "incorrect": "Often incorrect",
}

REGION_COLORS = {
    "easy_to_learn": "#4caf50",
    "hard_to_learn": "#f44336",
    "ambiguous": "#2196f3",
    "mixed": "#9e9e9e",
    "correct": "#4caf50",
    "incorrect": "#f44336",
}


def _inverse_density_weight(
    counts: np.ndarray,
    scale: float,
    *,
    penalty: str = "parabolic",
) -> np.ndarray:
    """
    Map counts to visibility weight in [0, 1]. Higher count -> lower weight.

    parabolic: (1 - (c/c_max)^2)^2 — strong fade for dense bins/groups
    log:       1 - log(1+c)/log(1+c_max) — legacy mild compression
    """
    scale = max(float(scale), 1.0)
    norm = np.clip(counts.astype(float) / scale, 0.0, 1.0)
    if penalty == "log":
        return 1.0 - np.log1p(counts.astype(float)) / np.log1p(scale)
    # parabolic with squared complement for a steeper penalty on crowds
    base = 1.0 - norm**2
    return base**2


def compute_density_alphas(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    *,
    bins: int = 48,
    alpha_min: float = 0.04,
    alpha_max: float = 0.98,
    penalty: str = "parabolic",
) -> np.ndarray:
    """
    Per-point opacity from inverse spatial density and group rarity.

    Dense regions (many overlapping green points) get lower alpha; sparse bins and
    rare groups (e.g. hard_to_learn) get higher alpha so they remain visible.
    """
    n = len(x)
    if n == 0:
        return np.array([], dtype=float)

    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    xi = np.clip(np.digitize(x, x_edges, right=False) - 1, 0, bins - 1)
    yi = np.clip(np.digitize(y, y_edges, right=False) - 1, 0, bins - 1)
    spatial_counts = hist[xi, yi]
    spatial_max = float(spatial_counts.max()) if spatial_counts.size else 1.0
    spatial_weight = _inverse_density_weight(spatial_counts, spatial_max, penalty=penalty)

    if groups is not None:
        unique, inverse = np.unique(groups, return_inverse=True)
        group_totals = np.bincount(inverse, minlength=len(unique)).astype(float)
        group_counts = group_totals[inverse]
        group_weight = _inverse_density_weight(group_counts, float(n), penalty=penalty)
    else:
        group_weight = np.ones(n, dtype=float)

    combined = np.maximum(spatial_weight, group_weight)
    return alpha_min + (alpha_max - alpha_min) * combined


def _group_counts_for_plot(rows: List[Dict], color_by: str) -> List[Tuple[str, str, int]]:
    """Return (key, display_label, count) in stable display order."""
    if color_by == "correctness":
        keys = ["correct", "incorrect"]
        membership = {
            "correct": [
                r
                for r in rows
                if float(r.get("correctness", 0.0)) >= 0.67
            ],
            "incorrect": [
                r
                for r in rows
                if float(r.get("correctness", 0.0)) < 0.67
            ],
        }
    else:
        keys = ["easy_to_learn", "hard_to_learn", "ambiguous", "mixed"]
        membership = {k: [r for r in rows if r.get("region") == k] for k in keys}

    return [(k, REGION_DISPLAY_NAMES.get(k, k), len(membership[k])) for k in keys]


def _draw_group_stats_table(
    ax: plt.Axes,
    rows: List[Dict],
    color_by: str,
) -> None:
    """Mini legend table: group name, count, and % of population."""
    total = max(len(rows), 1)
    groups = _group_counts_for_plot(rows, color_by)
    cell_text = [
        [label, f"{count:,}", f"{100.0 * count / total:.1f}%"]
        for _key, label, count in groups
    ]

    table = ax.table(
        cellText=cell_text,
        colLabels=["Group", "Count", "%"],
        loc="lower right",
        cellLoc="left",
        bbox=[0.54, 0.03, 0.43, 0.22 if color_by == "correctness" else 0.30],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)

    for row_idx, (key, _label, _count) in enumerate(groups, start=1):
        color = REGION_COLORS.get(key, "#9e9e9e")
        cell = table[row_idx, 0]
        cell.set_facecolor(color)
        cell.set_alpha(0.35)
        cell.get_text().set_color("#1a1a1a")
        cell.get_text().set_weight("bold")

    for col in range(3):
        table[0, col].set_facecolor("#f5f5f5")
        table[0, col].get_text().set_weight("bold")

    table.set_zorder(10)


def _scatter_with_alphas(
    x: np.ndarray,
    y: np.ndarray,
    colors: List[str],
    alphas: np.ndarray,
    *,
    sizes: np.ndarray | None = None,
    markers: List[str] | None = None,
    default_size: float = 20.0,
) -> None:
    """Draw low-opacity points first so rare high-opacity points sit on top."""
    order = np.argsort(alphas)
    x_ord = x[order]
    y_ord = y[order]
    c_ord = [colors[i] for i in order]
    a_ord = alphas[order]
    if sizes is not None:
        s_ord = sizes[order]
    else:
        s_ord = np.full(len(order), default_size)
    if markers is None:
        plt.scatter(x_ord, y_ord, c=c_ord, alpha=a_ord, s=s_ord, linewidths=0)
        return
    m_ord = [markers[i] for i in order]
    for marker in sorted(set(m_ord)):
        mask = np.array([m == marker for m in m_ord])
        plt.scatter(
            x_ord[mask],
            y_ord[mask],
            c=[c_ord[i] for i in range(len(c_ord)) if mask[i]],
            alpha=a_ord[mask],
            s=s_ord[mask],
            marker=marker,
            linewidths=0.6 if marker == "x" else 0,
        )


def save_data_map_plot(
    rows: List[Dict],
    output_path: Path,
    *,
    color_by: str = "region",
    title: str = "Dataset Cartography Data Map",
    thresholds: Dict[str, float] | None = None,
    opacity_mode: str | None = None,
    density_penalty: str = "parabolic",
    show_stats_table: bool = False,
) -> None:
    """Plot data map; use color_by='correctness' for paper Fig. 1 style (green/red)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    limits = thresholds or fit_region_thresholds(rows)
    x = np.array([float(r["variability"]) for r in rows])
    y = np.array([float(r["confidence"]) for r in rows])
    use_density = opacity_mode == "density"
    flat_alpha = 0.85 if use_density else None

    fig, ax = plt.subplots(figsize=(8, 6))
    if color_by == "correctness":
        correct = np.array([float(r.get("correctness", 0.0)) >= 0.67 for r in rows])
        if use_density:
            alphas = compute_density_alphas(
                x,
                y,
                groups=np.where(correct, "correct", "incorrect"),
                penalty=density_penalty,
            )
            colors = ["#4caf50" if c else "#f44336" for c in correct]
            markers = ["o" if c else "x" for c in correct]
            sizes = np.where(correct, 16.0, 32.0)
            _scatter_with_alphas(x, y, colors, alphas, sizes=sizes, markers=markers)
        else:
            if correct.any():
                ax.scatter(
                    x[correct],
                    y[correct],
                    c="#4caf50",
                    alpha=0.45,
                    s=18,
                    label="mostly correct",
                )
            if (~correct).any():
                ax.scatter(
                    x[~correct],
                    y[~correct],
                    c="#f44336",
                    alpha=0.5,
                    s=22,
                    marker="x",
                    label="often incorrect",
                )
            ax.legend(loc="upper left", fontsize=9)
    else:
        regions = np.array([r.get("region", "mixed") for r in rows])
        colors = [REGION_COLORS.get(r, "#9e9e9e") for r in regions]
        if use_density:
            alphas = compute_density_alphas(
                x, y, groups=regions, penalty=density_penalty
            )
            _scatter_with_alphas(x, y, colors, alphas, default_size=18.0)
        else:
            ax.scatter(x, y, c=colors, alpha=flat_alpha or 0.7, s=22)

    ax.axvline(
        limits["low_variability_max"],
        color="#616161",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )
    ax.axvline(
        limits["ambiguous_variability_min"],
        color="#1565c0",
        linestyle=":",
        linewidth=1,
        alpha=0.8,
    )
    ax.axhline(
        limits["easy_confidence_min"],
        color="#2e7d32",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )
    ax.axhline(
        limits["hard_confidence_max"],
        color="#c62828",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )
    ax.set_xlabel("Variability (std of gold-label probability)")
    ax.set_ylabel("Confidence (mean gold-label probability)")
    ax.set_title(title)
    ax.grid(alpha=0.2)

    if show_stats_table:
        _draw_group_stats_table(ax, rows, color_by)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
