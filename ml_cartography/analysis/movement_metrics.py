"""Idea #2 metrics: region movement toward easy-to-learn and learnability vs compute."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ml_cartography.analysis.data_map import assign_region

REGION_RANK = {
    "hard_to_learn": 0,
    "mixed": 1,
    "ambiguous": 2,
    "easy_to_learn": 3,
}

REGIONS = list(REGION_RANK.keys())


def _index_by_guid(rows: List[Dict]) -> Dict[str, Dict]:
    return {str(r["guid"]): r for r in rows}


def region_fractions(coordinates: List[Dict]) -> Dict[str, float]:
    n = max(len(coordinates), 1)
    counts = {r: 0 for r in REGIONS}
    for row in coordinates:
        region = row.get("region") or assign_region(
            float(row["confidence"]), float(row["variability"])
        )
        counts[region] = counts.get(region, 0) + 1
    return {f"region_frac/{k}": counts[k] / n for k in REGIONS}


def learnability_index(coordinates: List[Dict]) -> float:
    """Higher = more examples confidently and stably correct (paper-style learnability)."""
    if not coordinates:
        return 0.0
    scores = []
    for row in coordinates:
        conf = float(row["confidence"])
        corr = float(row.get("correctness", 0.0))
        scores.append(0.6 * conf + 0.4 * corr)
    return float(np.mean(scores))


def compute_epoch_movement(
    prev_coords: Optional[List[Dict]],
    curr_coords: List[Dict],
) -> Dict[str, float]:
    """
    Movement metrics between consecutive snapshot epochs.

    - pct_newly_easy: not easy last epoch → easy now
    - pct_easyward_step: region rank increased (toward easy_to_learn)
    - mean_confidence_delta / mean_variability_delta
    - easyward_score: mean Δconf − 0.5·Δvar (positive → toward easy quadrant)
    """
    out: Dict[str, float] = {}
    out.update(region_fractions(curr_coords))
    out["learnability/index"] = learnability_index(curr_coords)

    if not prev_coords:
        return out

    prev = _index_by_guid(prev_coords)
    curr = _index_by_guid(curr_coords)
    common = set(prev) & set(curr)
    if not common:
        return out

    newly_easy = easyward_step = 0
    conf_deltas: List[float] = []
    var_deltas: List[float] = []
    easyward_scores: List[float] = []
    transition = np.zeros((len(REGIONS), len(REGIONS)), dtype=int)

    for guid in common:
        p, c = prev[guid], curr[guid]
        pr = p.get("region") or assign_region(float(p["confidence"]), float(p["variability"]))
        cr = c.get("region") or assign_region(float(c["confidence"]), float(c["variability"]))
        transition[REGION_RANK[pr], REGION_RANK[cr]] += 1

        if pr != "easy_to_learn" and cr == "easy_to_learn":
            newly_easy += 1
        if REGION_RANK[cr] > REGION_RANK[pr]:
            easyward_step += 1

        d_conf = float(c["confidence"]) - float(p["confidence"])
        d_var = float(c["variability"]) - float(p["variability"])
        conf_deltas.append(d_conf)
        var_deltas.append(d_var)
        easyward_scores.append(d_conf - 0.5 * d_var)

    n = len(common)
    out["movement/pct_newly_easy"] = newly_easy / n
    out["movement/pct_easyward_region_step"] = easyward_step / n
    out["movement/mean_confidence_delta"] = float(np.mean(conf_deltas))
    out["movement/mean_variability_delta"] = float(np.mean(var_deltas))
    out["movement/easyward_score"] = float(np.mean(easyward_scores))
    out["movement/pct_confidence_up"] = sum(1 for d in conf_deltas if d > 0) / n
    out["learnability/delta"] = out["learnability/index"] - learnability_index(prev_coords)
    out["_transition_matrix"] = transition  # internal; stripped before wandb log
    return out


def compute_learnability_efficiency(
    *,
    delta_learnability: float,
    optimizer_steps: int,
    trainable_params: int,
    batch_size: int,
    seq_length: int,
) -> Dict[str, float]:
    """
    Learnability gain per unit compute.

    compute_units = optimizer_steps × trainable_params (weight-update proxy)
    token_steps   = optimizer_steps × batch_size × seq_length
    """
    param_units = max(optimizer_steps * trainable_params, 1)
    token_steps = max(optimizer_steps * batch_size * seq_length, 1)
    return {
        "compute/optimizer_steps": float(optimizer_steps),
        "compute/trainable_params": float(trainable_params),
        "compute/param_update_units": float(param_units),
        "compute/token_steps": float(token_steps),
        "learnability/per_1m_param_updates": delta_learnability / (param_units / 1e6),
        "learnability/per_1b_token_steps": delta_learnability / (token_steps / 1e9),
    }


def summarize_trajectories(trajectories: List[Dict]) -> Dict[str, float]:
    """Aggregate per-example histories: monotonic easyward paths, final-easy rate."""
    if not trajectories:
        return {}

    final_easy = monotonic_conf = ended_easy = 0
    total_easyward_steps = total_steps = 0

    for traj in trajectories:
        hist = traj.get("history", [])
        if not hist:
            continue
        confs = [float(h["confidence"]) for h in hist]
        regions = [h.get("region", "mixed") for h in hist]

        if regions[-1] == "easy_to_learn":
            ended_easy += 1
        if len(hist) >= 2 and all(confs[i] <= confs[i + 1] for i in range(len(confs) - 1)):
            monotonic_conf += 1

        for i in range(1, len(hist)):
            total_steps += 1
            pr, cr = regions[i - 1], regions[i]
            if REGION_RANK.get(cr, 1) > REGION_RANK.get(pr, 1):
                total_easyward_steps += 1

    n = len(trajectories)
    return {
        "trajectory/pct_ended_easy": ended_easy / max(n, 1),
        "trajectory/pct_monotonic_confidence": monotonic_conf / max(n, 1),
        "trajectory/pct_easyward_region_steps": total_easyward_steps / max(total_steps, 1),
    }


def save_transition_heatmap(
    transition: np.ndarray,
    output_path: Path,
    title: str = "Region transitions (prev → curr)",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_sums = transition.sum(axis=1, keepdims=True)
    norm = np.divide(transition, row_sums, where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(REGIONS)))
    ax.set_yticks(range(len(REGIONS)))
    short = ["hard", "mixed", "ambig", "easy"]
    ax.set_xticklabels(short, rotation=45, ha="right")
    ax.set_yticklabels(short)
    ax.set_xlabel("Region at epoch t")
    ax.set_ylabel("Region at epoch t−1")
    ax.set_title(title)
    for i in range(len(REGIONS)):
        for j in range(len(REGIONS)):
            count = int(transition[i, j])
            ax.text(
                j,
                i,
                str(count),
                ha="center",
                va="center",
                fontsize=8,
                color="#555555" if count == 0 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, label="P(col|row)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_learnability_vs_compute_plot(
    history: List[Dict],
    output_path: Path,
) -> Path:
    """Cumulative learnability vs cumulative param-update units."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return output_path

    cum_compute = []
    cum_learn = []
    cc = cl = 0.0
    for row in history:
        cc += float(row.get("compute/param_update_units", 0))
        cl += float(row.get("learnability/delta", row.get("learnability/index", 0.0)))
        cum_compute.append(cc)
        cum_learn.append(cl)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(range(1, len(cum_learn) + 1), cum_learn, "g-o", label="Cumulative Δ learnability")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cumulative learnability gain", color="green")
    ax1.tick_params(axis="y", labelcolor="green")
    ax1.grid(alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(range(1, len(cum_compute) + 1), cum_compute, "b--s", label="Cumulative param-update units")
    ax2.set_ylabel("Cumulative compute (param·steps)", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    ax1.set_title("Learnability improvement vs compute (Idea #2)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def strip_internal_keys(metrics: Dict) -> Dict[str, float]:
    return {k: v for k, v in metrics.items() if not k.startswith("_") and isinstance(v, (int, float))}
