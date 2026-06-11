# CS-162 Final: Dataset Cartography Reproduction

Educational reimplementation of [*Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics*](https://aclanthology.org/2020.emnlp-main.746/) (Swayamdipta et al., EMNLP 2020).

We train classifiers, log **per-epoch training dynamics** (confidence, variability, correctness), build **data maps**, and rerun the paper’s downstream analyses: subset selection (§3), ambiguous scaling (§4), noise detection (§5), and uncertainty / human agreement (§6).

**Canonical figures and tables** for the write-up live in [`paper_outputs/`](paper_outputs/) (see [`paper_outputs/manifest.json`](paper_outputs/manifest.json)).

---

## Primary objective

Reproduce the paper’s central claim: **training dynamics reveal structure in a dataset** that standard accuracy alone hides. Concretely we aim to:

1. **Map** each example by variability (x) and confidence (y), and label regions as easy-to-learn, hard-to-learn, or ambiguous.
2. **Select** 33% subsets by strategy and measure **in-distribution (ID)** vs **out-of-distribution (OOD)** performance (Tables 2–4).
3. **Ablate** how much ambiguous vs easy-to-learn data is needed for optimization and generalization (Figure 3).
4. **Diagnose** label noise and intrinsic uncertainty via dynamics (Figures 4–5).

The pipeline is modular (`scripts/00`–`12` + tmux launchers in [`scripts/EXPERIMENT_LAUNCHERS.md`](scripts/EXPERIMENT_LAUNCHERS.md)), logs to **Weights & Biases**, and exports reproducible artifacts under `results/` and `paper_outputs/`.

---

## Key findings (measured on this repo)

Values below come from our **RoBERTa-large / RoBERTa-base** runs (often 1 seed and subsampled NLI where noted). OOD scores use **proxy evaluators** (WSC-style for WinoGrande, diagnostics proxy for NLI) unless a full benchmark hook is wired. Absolute numbers may differ from the paper’s 3-seed RoBERTa-large tables; **trends** match the original work.

### 1. Data maps expose three stable regions (§2)

SNLI and WinoGrande both show the paper’s wedge shape: a dense **easy-to-learn** cloud (high confidence, low variability), a **hard-to-learn** tail (low confidence), and **ambiguous** examples along the high-variability edge.

| Figure | Dataset | Artifact |
|--------|---------|----------|
| Fig 1 | SNLI | [`paper_outputs/Figure_01_snli_data_map.png`](paper_outputs/Figure_01_snli_data_map.png) |
| Fig 2 | WinoGrande | [`paper_outputs/Figure_02_winogrande_data_map.png`](paper_outputs/Figure_02_winogrande_data_map.png) |

![SNLI data map](paper_outputs/Figure_01_snli_data_map.png)

![WinoGrande data map](paper_outputs/Figure_02_winogrande_data_map.png)

### 2. Ambiguous / hard-to-learn subsets help OOD at 33% train (§3)

On WinoGrande (Table 2), selecting the top-variability (**ambiguous**) or low-confidence (**hard-to-learn**) third matches or beats **random 33%** on our OOD proxy while using far less data. Conversely, **high-confidence** and **low-variability** selections underperform random on ID—mirroring the paper’s “overconfident, easy” failure modes.

| Strategy | ID (%) | OOD proxy (%) |
|----------|--------|---------------|
| 100% train | 79.9 | 61.2 |
| random (33%) | 75.1 | 61.2 |
| ambiguous (33%) | 75.1 | **63.2** |
| hard-to-learn (33%) | 74.8 | **63.2** |
| high-confidence (33%) | 67.1 | 60.0 |

![Table 2 — WinoGrande selection](paper_outputs/Table_02_winogrande_selection.png)

Subsampled NLI finetunes (Table 3–4) show the same pattern at a smaller scale: **ambiguous** subsets can lift OOD proxy vs random, while **hard-to-learn** subsets hurt ID when the train budget is tight.

![Table 3 — SNLI](paper_outputs/Table_03_snli_selection.png)

![Table 4 — QNLI](paper_outputs/Table_04_qnli_selection.png)

### 3. Easy-to-learn examples are required for optimization (§4)

Figure 3 sweeps the fraction of **ambiguous-only** WinoGrande training data (50% → 1%) against size-matched random baselines, then replaces easy examples inside a fixed 17% ambiguous core.

- **ID** curves stay near chance when the ambiguous fraction is very small—models fail to optimize on ambiguous-only slices alone.
- **OOD proxy** is highest when training on the smallest ambiguous fractions (high-variability tail), but at the cost of unstable ID learning.
- **Replacing** even 10% of the 17% ambiguous core with easy-to-learn examples improves ID without fully sacrificing OOD—consistent with the paper’s “optimization vs generalization” trade-off.

![Figure 3 — easy-to-learn role](paper_outputs/Figure_03_easy_to_learn_role.png)

### 4. Label noise shifts dynamics toward hard-to-learn (§5)

Injecting 1% label flips into easy examples and retraining moves the flipped points toward **lower confidence** and **higher variability** in the before/after histograms (Figure 4)—the same distributional shift the paper uses to motivate automatic noise detection.

![Figure 4 — noise shift](paper_outputs/Figure_04_noise_shift.png)

### 5. Confidence tracks human agreement (§6)

The Figure 5 heatmap bins examples by confidence × variability and colors cells by mean human agreement (or a deterministic proxy when multi-annotator labels are unavailable). **High-confidence regions align with high annotator agreement**, supporting the paper’s link between dynamics and intrinsic uncertainty.

![Figure 5 — human agreement](paper_outputs/Figure_05_human_agreement_heatmap.png)

---

## Extensions beyond the paper

These experiments are **not** in Swayamdipta et al. (2020); they stress-test the same dynamics machinery and produced supporting artifacts in `experiments/runs/` and `results/`.

| Extension | What we did | Why it matters |
|-----------|-------------|----------------|
| **Multi-architecture SNLI maps** | DistilBERT, RoBERTa-base/large, Llama-3.2-1B, Ministral-3B on SNLI | Shows which map shapes are architecture-dependent vs dataset-intrinsic (paper App. C.1 explores weaker encoders). |
| **Idea #1 — Preference / instruction maps** | `--task preference` and `--task instruction` in `run_cartography_experiment.py` | Extends cartography to **alignment-style** pair data and instruction tuning—high-variability pairs export to `subset_high_variability.jsonl` for selective / DPO-style training. |
| **Idea #2 — Dynamic maps + curriculum** | `--task dynamic`: per-epoch snapshots, region trajectories, ambiguous upweighting after epoch 2 | Tests whether **time-varying** maps and adaptive sampling improve learning vs a single post-hoc map. |
| **Multi-GPU orchestration** | `dual_gpu_train_suite.py`, `run_exp_*` tmux launchers, 5× RTX 3090 layout | Makes full reproduction feasible: parallel map collection + subset finetuning. |
| **Streamlit results browser** | `apps/streamlit_app.py` | Explores `results/<run_id>/` dynamics, maps, and metrics without digging through JSONL. |
| **Subregion compute–gain study** | `run_exp_10_subregion_compute_gain.sh` | Quantifies whether finetuning on region-only subsets is more **compute-efficient** than full-data training. |

Deeper write-ups: [`results/report.md`](results/report.md), [`docs/paper-summary.md`](docs/paper-summary.md).

---

## Repository organization

```
CS-162-Final/
├── ml_cartography/           # Core library
│   ├── core/                 # Dynamics aggregation (confidence, variability, correctness)
│   ├── analysis/             # Maps, regions, paper figures/tables (paper_figures.py, paper_tables.py)
│   ├── experiments/          # Subset selection, noise, uncertainty helpers
│   ├── training/             # GLUE/WinoGrande trainers, curriculum, snapshots
│   └── utils/                # I/O, W&B helpers
│
├── scripts/                  # Runnable experiments
│   ├── 00–07               # Numbered paper pipeline (toy → insight figures)
│   ├── 08–12               # Fig 3/4/5 + region finetune + noise detection
│   ├── run_exp_01–10       # One-line tmux launchers per paper figure/table
│   ├── run_cartography_experiment.py   # Main train → dynamics → maps entry point
│   ├── collect_paper_outputs.py        # Copy/rename artifacts → paper_outputs/
│   ├── plot_from_metrics_csv.py        # Regenerate measured figs from JSON/CSV
│   └── lib/                  # Shared conda/GPU/env defaults
│
├── paper_outputs/            # ★ Figures & tables for the final report (Figure_*, Table_*)
├── results/                  # Timestamped run folders + region_metrics_*.json + plots
├── data/processed/           # Subsets, finetune manifests, easy_role / table3 / table4 runs
├── experiments/runs/         # Working training directories (gitignored)
├── configs/                  # JSON training presets
├── apps/streamlit_app.py     # Interactive results explorer
├── docs/                     # Paper notes, setup guides (local-docs/ is gitignored)
└── notebooks/                # Colab train suite
```

**Data flow:** train with `run_cartography_experiment.py` or `09_region_finetune.py` → dynamics JSONL under `experiments/runs/` or `data/processed/` → metrics via `11_region_metrics.py` → plots via `plot_from_metrics_csv.py` → collect with `collect_paper_outputs.py` → `paper_outputs/`.

---

## Quick start

### Environment

```bash
bash scripts/setup_conda_env.sh
conda activate cs162-cartography
# optional: copy wandb_credentials.example.txt → wandb_credentials.txt
```

### Reproduce paper outputs (after training artifacts exist)

```bash
conda activate cs162-cartography

# Regenerate measured figures from easy_role + region metrics
python scripts/plot_from_metrics_csv.py --no-wandb

# Copy/render everything into paper_outputs/
python scripts/collect_paper_outputs.py
```

### Run individual paper experiments (tmux + W&B)

```bash
bash scripts/run_exp_04_table2_winogrande.sh   # Table 2
bash scripts/run_exp_05_fig3_easy_to_learn.sh    # Figure 3
bash scripts/run_exp_06_fig4_noise.sh            # Figure 4
bash scripts/run_exp_07_fig5_uncertainty.sh      # Figure 5
bash scripts/run_exp_08_table3_snli_mnli.sh      # Table 3
bash scripts/run_exp_09_table4_qnli.sh           # Table 4
```

See [`scripts/EXPERIMENT_LAUNCHERS.md`](scripts/EXPERIMENT_LAUNCHERS.md) for hyperparameters and GPU layout.

### Train dynamics from scratch

```bash
conda activate cs162-cartography
pip install -r requirements-train.txt

# Single SNLI/WinoGrande run → results/<timestamp>_<task>_<preset>/
python scripts/run_cartography_experiment.py --task snli --preset roberta-base --epochs 3

# Full numbered pipeline on toy data
python scripts/run_all_experiments.py --no-wandb
```

### Streamlit explorer

```bash
streamlit run apps/streamlit_app.py
```

---

## Paper pipeline scripts (§2–§6)

| Script | Paper | Role |
|--------|-------|------|
| `scripts/01_collect_dynamics.py` | §2 | Confidence, variability, correctness |
| `scripts/02_build_data_map.py` | §2 | Regions + scatter + histograms |
| `scripts/03_select_subsets.py` | §3 | Top-33% selection strategies |
| `scripts/08_role_easy_to_learn.py` | §4 | Ambiguous % sweep + easy replacement |
| `scripts/09_region_finetune.py` | §3 | Retrain on each subset (Tables 2–4) |
| `scripts/11_region_metrics.py` | §3 | ID/OOD proxy metrics per strategy |
| `scripts/12_noise_detection_paper.py` | §5 | Label-noise injection + detector |
| `scripts/05_uncertainty_checks.py` | §6 | Human agreement correlations |

---

## GPU training notes

| Preset | Model | Notes |
|--------|--------|-------|
| `distilbert` | DistilBERT | Fast smoke tests |
| `roberta-base` | RoBERTa-base | Subset / deadline experiments |
| `roberta-large` | RoBERTa-large | Closest to paper default |
| `llama-3.2-1b` | Llama 3.2 1B (4-bit) | Causal LM head on NLI |
| `ministral-3b` | Ministral 3 3B (4-bit) | Unsloth checkpoint |

- **Colab:** `notebooks/colab_train_suite.ipynb` or `bash scripts/colab_setup.sh` — use `requirements-colab.txt` (not `requirements-train.txt`) to keep CUDA torch. See [docs/COLAB_CUDA.md](docs/COLAB_CUDA.md).
- **Local multi-GPU:** `bash scripts/tmux_train_suite.sh` or `python scripts/dual_gpu_train_suite.py`
- **Hugging Face gated models:** [docs/HUGGINGFACE_SETUP.md](docs/HUGGINGFACE_SETUP.md)

Use `--max-train-samples 0` for the full SNLI train split (~550k; slow).

---

## Citation

```bibtex
@inproceedings{swayamdipta2020dataset,
  title={Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics},
  author={Swayamdipta, Swabha and Schwartz, Roy and Lourie, Nicholas and Wang, Yizhong and Hajishirzi, Hannaneh and Smith, Noah A. and Choi, Yejin},
  booktitle={EMNLP},
  year={2020}
}
```
