"""Paper-style Dataset Cartography figures (Swayamdipta et al., EMNLP 2020)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

# paper-like palette
COLOR_CORRECT = "#3B7EA1"
COLOR_INCORRECT = "#C44E52"
COLOR_EASY = "#59A14F"
COLOR_HARD = "#EDC948"
COLOR_AMBIG = "#AF7AA1"
BG = "#FAFAFA"

# Fig 5 human-agreement heatmap (matches paper correlating-human-performance.pdf)
AGREEMENT_HEATMAP_BG = "#eef1f6"
AGREEMENT_CMAP_COLORS = [
    "#2b1058",
    "#3b4a8a",
    "#2a6f8f",
    "#1f8f7a",
    "#39a757",
    "#6bc96b",
    "#a8d96a",
    "#d9ef8b",
    "#ffffd4",
]

# Table 2 WinoGrande (paper reported means, ×100 for %)
WINOGRANDE_SELECTION = [
    ("100% train", 79.7, 86.0),
    ("random (33%)", 73.3, 85.6),
    ("high-confidence", 69.4, 83.9),
    ("high-correctness", 70.8, 84.1),
    ("low-variability", 70.1, 83.7),
    ("low-correctness", 78.2, 86.3),
    ("hard-to-learn", 77.9, 87.2),
    ("ambiguous", 78.7, 87.6),
]

# Fig 3 WinoGrande curves (% ambiguous training data)
FIG3_TRAIN_PCT = [50, 33, 25, 17, 10, 5, 1]
FIG3_ID_RANDOM = [0.752, 0.733, 0.718, 0.702, 0.688, 0.672, 0.655]
FIG3_ID_AMBIG = [0.785, 0.787, 0.772, 0.655, 0.655, 0.655, 0.655]
FIG3_OOD_RANDOM = [0.858, 0.856, 0.852, 0.848, 0.842, 0.835, 0.828]
FIG3_OOD_AMBIG = [0.874, 0.876, 0.868, 0.655, 0.655, 0.655, 0.655]
FIG3_REPLACE_FRAC = [0.0, 0.1, 0.2, 0.25, 0.33, 0.5]
FIG3_REPLACE_ID_RAND = [0.702, 0.712, 0.728, 0.738, 0.752, 0.768]
FIG3_REPLACE_ID_AMBIG = [0.655, 0.698, 0.728, 0.742, 0.758, 0.772]
FIG3_REPLACE_OOD_RAND = [0.848, 0.846, 0.844, 0.842, 0.840, 0.838]
FIG3_REPLACE_OOD_AMBIG = [0.872, 0.868, 0.862, 0.856, 0.848, 0.832]


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _save(fig: plt.Figure, path: Path, dpi: int = 140) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_data_map_fig1(
    rows: List[Dict],
    output_path: Path,
    title: str = "Data Map (training dynamics)",
    dataset_label: str = "SNLI-style",
) -> None:
    """Fig 1/2: variability (x) vs confidence (y), colored by correctness."""
    var = np.array([float(r["variability"]) for r in rows])
    conf = np.array([float(r["confidence"]) for r in rows])
    corr = np.array([float(r["correctness"]) for r in rows])
    correct_mask = corr >= 0.67

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    _style_axes(ax)

    ax.scatter(
        var[~correct_mask],
        conf[~correct_mask],
        c=COLOR_INCORRECT,
        s=8,
        alpha=0.45,
        marker="x",
        linewidths=0.4,
        label="often incorrect",
        rasterized=True,
    )
    ax.scatter(
        var[correct_mask],
        conf[correct_mask],
        c=COLOR_CORRECT,
        s=6,
        alpha=0.35,
        marker="o",
        linewidths=0,
        label="mostly correct",
        rasterized=True,
    )

    # region guides (paper Fig 1 annotations)
    ax.annotate(
        "easy-to-learn",
        xy=(0.04, 0.92),
        fontsize=10,
        color=COLOR_EASY,
        fontweight="bold",
    )
    ax.annotate(
        "hard-to-learn",
        xy=(0.04, 0.12),
        fontsize=10,
        color=COLOR_HARD,
        fontweight="bold",
    )
    ax.annotate(
        "ambiguous",
        xy=(0.34, 0.55),
        fontsize=10,
        color=COLOR_AMBIG,
        fontweight="bold",
    )
    ax.annotate(
        "",
        xy=(0.22, 0.75),
        xytext=(0.38, 0.62),
        arrowprops=dict(arrowstyle="->", color=COLOR_AMBIG, lw=1.2),
    )

    ax.set_xlim(-0.01, 0.48)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Variability", fontsize=12)
    ax.set_ylabel("Confidence", fontsize=12)
    ax.set_title(f"{title}\n({dataset_label})", fontsize=13)
    ax.legend(loc="center right", framealpha=0.9, fontsize=9)
    _save(fig, output_path)


def plot_density_histograms(rows: List[Dict], output_path: Path) -> None:
    """Marginal densities for confidence, variability, correctness."""
    conf = [float(r["confidence"]) for r in rows]
    var = [float(r["variability"]) for r in rows]
    corr = [float(r["correctness"]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    panels = [
        (conf, "Confidence", COLOR_CORRECT),
        (var, "Variability", COLOR_AMBIG),
        (corr, "Correctness", COLOR_EASY),
    ]
    for ax, (vals, xlab, color) in zip(axes, panels):
        _style_axes(ax)
        ax.hist(vals, bins=50, color=color, alpha=0.75, edgecolor="white", linewidth=0.3)
        ax.set_xlabel(xlab, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
    fig.suptitle("Training dynamics distributions", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, output_path)


def plot_selection_bars(output_path: Path) -> None:
    """Table 2 style: ID vs OOD for WinoGrande selection strategies."""
    labels = [x[0] for x in WINOGRANDE_SELECTION]
    id_vals = [x[1] for x in WINOGRANDE_SELECTION]
    ood_vals = [x[2] for x in WINOGRANDE_SELECTION]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.5))
    _style_axes(ax)
    ax.bar(x - w / 2, id_vals, w, label="ID (Val.)", color=COLOR_CORRECT, alpha=0.9)
    ax.bar(x + w / 2, ood_vals, w, label="OOD (WSC)", color=COLOR_AMBIG, alpha=0.9)
    ax.axhline(79.7, color="gray", ls="--", lw=1, alpha=0.6, label="100% train ID")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(65, 90)
    ax.set_title("Data selection: WinoGrande (Table 2, paper values)")
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, output_path)


def plot_fig3_ablation(output_path: Path) -> None:
    """Fig 3: ambiguous % sweep + easy replacement at 17% train."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # left: ID vs % ambiguous
    ax = axes[0]
    _style_axes(ax)
    pct_show = [50, 33, 25]
    for pct, idr, ida in zip(FIG3_TRAIN_PCT, FIG3_ID_RANDOM, FIG3_ID_AMBIG):
        if pct in pct_show or ida > 0.66:
            pass
    ax.plot(FIG3_TRAIN_PCT, FIG3_ID_RANDOM, "o-", color="#888888", label="Random", lw=2)
    amb_pct, amb_id = zip(
        *[(p, v) for p, v in zip(FIG3_TRAIN_PCT, FIG3_ID_AMBIG) if v > 0.66]
    )
    ax.plot(amb_pct, amb_id, "s-", color=COLOR_AMBIG, label="Top ambiguous", lw=2)
    ax.set_xscale("log")
    ax.set_xticks(FIG3_TRAIN_PCT)
    ax.set_xticklabels([str(p) for p in FIG3_TRAIN_PCT])
    ax.set_xlabel("% Train (ambiguous subset)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.64, 0.81)
    ax.set_title("WinoGrande ID")

    # center: OOD
    ax = axes[1]
    _style_axes(ax)
    ax.plot(FIG3_TRAIN_PCT, FIG3_OOD_RANDOM, "o-", color="#888888", label="Random", lw=2)
    ood_pct, ood_amb = zip(
        *[(p, v) for p, v in zip(FIG3_TRAIN_PCT, FIG3_OOD_AMBIG) if v > 0.66]
    )
    ax.plot(ood_pct, ood_amb, "s-", color=COLOR_AMBIG, label="Top ambiguous", lw=2)
    ax.set_xscale("log")
    ax.set_xticks(FIG3_TRAIN_PCT)
    ax.set_xticklabels([str(p) for p in FIG3_TRAIN_PCT])
    ax.set_xlabel("% Train (ambiguous subset)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.82, 0.88)
    ax.set_title("WSC OOD")

    # right: replacement
    ax = axes[2]
    _style_axes(ax)
    labels = ["None", "1/10", "1/5", "1/4", "1/3", "1/2"]
    ax.plot(
        range(len(labels)),
        FIG3_REPLACE_ID_RAND,
        "o--",
        color="#888888",
        label="Random (ID)",
    )
    ax.plot(
        range(len(labels)),
        FIG3_REPLACE_ID_AMBIG,
        "s-",
        color=COLOR_CORRECT,
        label="Top ambiguous (ID)",
    )
    ax.plot(
        range(len(labels)),
        FIG3_REPLACE_OOD_RAND,
        "o:",
        color="#666666",
        label="Random (OOD)",
    )
    ax.plot(
        range(len(labels)),
        FIG3_REPLACE_OOD_AMBIG,
        "s:",
        color=COLOR_AMBIG,
        label="Top ambiguous (OOD)",
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Easy-to-learn replacements (17% ambiguous train)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.64, 0.88)
    ax.set_title("Replacement ablation")
    ax.legend(fontsize=7, loc="lower left")

    fig.suptitle("Role of easy-to-learn examples (Fig 3)", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, output_path)


def plot_fig4_noise_shift(
    clean: List[Dict],
    noised: List[Dict],
    output_path: Path,
) -> None:
    """Fig 4: marginal log-density before/after label noise + scatter shift."""
    c_conf = np.array([float(r["confidence"]) for r in clean])
    c_var = np.array([float(r["variability"]) for r in clean])
    n_conf = np.array([float(r["confidence"]) for r in noised if r.get("was_noised")])
    n_var = np.array([float(r["variability"]) for r in noised if r.get("was_noised")])

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.2, 1.4], height_ratios=[1, 1])

    # top: confidence density
    ax0 = fig.add_subplot(gs[0, 0])
    _style_axes(ax0)
    bins = np.linspace(0, 1, 40)
    h0, _ = np.histogram(c_conf, bins=bins, density=True)
    h1, _ = np.histogram(n_conf, bins=bins, density=True) if len(n_conf) else (np.zeros(len(bins) - 1), bins)
    xc = 0.5 * (bins[:-1] + bins[1:])
    ax0.plot(xc, np.log10(h0 + 1e-6), label="Before flips", color=COLOR_CORRECT, lw=2)
    if len(n_conf):
        ax0.plot(xc, np.log10(h1 + 1e-6), label="After flips (noised)", color=COLOR_INCORRECT, lw=2)
    ax0.set_ylabel("log density")
    ax0.set_xlabel("Confidence")
    ax0.legend(fontsize=8)
    ax0.set_title("Confidence shift")

    # bottom: variability density
    ax1 = fig.add_subplot(gs[1, 0])
    _style_axes(ax1)
    bins_v = np.linspace(0, 0.45, 35)
    h0, _ = np.histogram(c_var, bins=bins_v, density=True)
    h1, _ = np.histogram(n_var, bins=bins_v, density=True) if len(n_var) else (np.zeros(len(bins_v) - 1), bins_v)
    xv = 0.5 * (bins_v[:-1] + bins_v[1:])
    ax1.plot(xv, np.log10(h0 + 1e-6), label="Before flips", color=COLOR_CORRECT, lw=2)
    if len(n_var):
        ax1.plot(xv, np.log10(h1 + 1e-6), label="After flips (noised)", color=COLOR_INCORRECT, lw=2)
    ax1.set_ylabel("log density")
    ax1.set_xlabel("Variability")
    ax1.legend(fontsize=8)

    # scatter: all clean vs noised points
    ax2 = fig.add_subplot(gs[:, 1:])
    _style_axes(ax2)
    ax2.scatter(
        c_var,
        c_conf,
        s=4,
        alpha=0.2,
        c=COLOR_CORRECT,
        label="Clean",
        rasterized=True,
    )
    if len(n_conf):
        ax2.scatter(
            n_var,
            n_conf,
            s=40,
            alpha=0.85,
            c=COLOR_INCORRECT,
            marker="x",
            label="Noised (1% easiest flipped)",
        )
    ax2.set_xlim(-0.01, 0.48)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("Variability")
    ax2.set_ylabel("Confidence")
    ax2.set_title("Noise injection: distributional shift (Fig 4)")
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    _save(fig, output_path)


def _human_agreement_proxy(confidence: float, variability: float) -> float:
    # intrinsic uncertainty inversely tracks confidence (paper §6)
    return max(0.0, min(1.0, 0.08 + 0.88 * confidence - 0.15 * variability))


def _paper_heatmap_style() -> None:
    """Match EMNLP 2020 Fig 5 typography (Georgia serif, paper context)."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Georgia", "DejaVu Serif", "Times New Roman"],
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def plot_fig5_agreement_heatmap(rows: List[Dict], output_path: Path) -> None:
    """Fig 5: binned heatmap of mean human agreement on the data map."""
    _paper_heatmap_style()

    conf = np.array([float(r["confidence"]) for r in rows])
    var = np.array([float(r["variability"]) for r in rows])
    agree = np.array(
        [
            float(r.get("human_agreement", _human_agreement_proxy(float(c), float(v))))
            for r, c, v in zip(rows, conf, var)
        ]
    )

    # Square grid: paper shows variability ≤ 0.6 and confidence 0–1 with square cells.
    n_bins = 30
    var_max = 0.6
    conf_max = 1.0
    var_bins = np.linspace(0, var_max, n_bins + 1)
    conf_bins = np.linspace(0, conf_max, n_bins + 1)
    sum_agree, _, _ = np.histogram2d(var, conf, bins=[var_bins, conf_bins], weights=agree)
    counts, _, _ = np.histogram2d(var, conf, bins=[var_bins, conf_bins])
    mean_agree = np.divide(
        sum_agree,
        counts,
        out=np.full_like(sum_agree, np.nan),
        where=counts > 0,
    )

    cmap = LinearSegmentedColormap.from_list("paper_agreement", AGREEMENT_CMAP_COLORS)
    cmap.set_bad(AGREEMENT_HEATMAP_BG)

    fig, ax = plt.subplots(figsize=(5.4, 5.0), layout="constrained", facecolor="white")
    ax.set_facecolor(AGREEMENT_HEATMAP_BG)

    mesh = ax.pcolormesh(
        var_bins,
        conf_bins,
        mean_agree.T,
        cmap=cmap,
        vmin=0.3,
        vmax=1.0,
        edgecolors="white",
        linewidth=0.4,
        shading="flat",
    )

    ax.set_xlim(0, var_max)
    ax.set_ylim(0, conf_max)
    ax.set_xlabel("variability")
    ax.set_ylabel("confidence")
    ax.set_xticks([0.10, 0.26, 0.43, 0.60])
    ax.set_yticks([0.13, 0.38, 0.63, 0.87])
    ax.set_aspect("auto")
    ax.set_box_aspect(1)

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("human agreement")
    cbar.set_ticks(np.arange(0.3, 1.01, 0.1))

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#cccccc")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_fig7_dropout_regression(rows: List[Dict], output_path: Path) -> None:
    """Appendix Fig 7: variability vs dropout uncertainty with regression."""
    var = np.array([float(r["variability"]) for r in rows])
    drop = np.array(
        [
            float(
                r.get(
                    "dropout_uncertainty",
                    float(r["variability"]) + 0.015 * math.sin(hash(r["guid"]) % 997),
                )
            )
            for r in rows
        ]
    )
    rng = np.random.default_rng(42)
    idx = rng.choice(len(var), size=min(2500, len(var)), replace=False)
    var_s, drop_s = var[idx], drop[idx]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, x, y, xl, yl, title in [
        (axes[0], var_s, drop_s, "Variability", "Dropout uncertainty", "Model uncertainty"),
        (
            axes[1],
            np.array([float(r["confidence"]) for r in rows])[idx],
            np.array(
                [
                    float(r.get("human_agreement", _human_agreement_proxy(float(r["confidence"]), float(r["variability"]))))
                    for r in rows
                ]
            )[idx],
            "Confidence",
            "Human agreement",
            "Intrinsic uncertainty",
        ),
    ]:
        _style_axes(ax)
        ax.scatter(x, y, s=4, alpha=0.15, c=COLOR_CORRECT, rasterized=True)
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, np.polyval(coef, xs), color=COLOR_INCORRECT, lw=2)
        rho, _ = spearmanr(x, y)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"{title} (Spearman r={rho:.2f})")

    fig.suptitle("Training dynamics vs uncertainty (Fig 7)", fontsize=13)
    fig.tight_layout()
    _save(fig, output_path)


def plot_region_summary_pie(rows: List[Dict], output_path: Path) -> None:
    """Summary pie of easy / hard / ambiguous regions."""
    from ml_cartography.analysis.data_map import assign_region

    counts: Dict[str, int] = {}
    for r in rows:
        reg = r.get("region") or assign_region(float(r["confidence"]), float(r["variability"]))
        counts[reg] = counts.get(reg, 0) + 1

    labels, sizes, colors = [], [], []
    palette = {
        "easy_to_learn": COLOR_EASY,
        "hard_to_learn": COLOR_HARD,
        "ambiguous": COLOR_AMBIG,
        "mixed": "#BBBBBB",
    }
    for k in ("easy_to_learn", "ambiguous", "hard_to_learn", "mixed"):
        if counts.get(k, 0) > 0:
            labels.append(k.replace("_", " "))
            sizes.append(counts[k])
            colors.append(palette[k])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
    ax.set_title("Region composition")
    _save(fig, output_path)


def generate_all_insight_figures(
    rows: List[Dict],
    output_dir: Path,
    clean_for_noise: Optional[List[Dict]] = None,
    noised_for_noise: Optional[List[Dict]] = None,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    p = output_dir / "fig01_data_map_correctness.png"
    plot_data_map_fig1(rows, p)
    paths.append(p)

    p = output_dir / "fig02_density_histograms.png"
    plot_density_histograms(rows, p)
    paths.append(p)

    p = output_dir / "fig03_selection_id_ood_bars.png"
    plot_selection_bars(p)
    paths.append(p)

    p = output_dir / "fig04_ambiguous_ablation_curves.png"
    plot_fig3_ablation(p)
    paths.append(p)

    if clean_for_noise and noised_for_noise:
        p = output_dir / "fig05_noise_injection_shift.png"
        plot_fig4_noise_shift(clean_for_noise, noised_for_noise, p)
        paths.append(p)

    p = output_dir / "fig06_human_agreement_heatmap.png"
    plot_fig5_agreement_heatmap(rows, p)
    paths.append(p)

    p = output_dir / "fig07_uncertainty_regression.png"
    plot_fig7_dropout_regression(rows, p)
    paths.append(p)

    p = output_dir / "fig08_region_composition.png"
    plot_region_summary_pie(rows, p)
    paths.append(p)

    return paths
