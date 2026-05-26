# Dataset Cartography — Experiment Report

**Paper:** *Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics* (Swayamdipta et al., EMNLP 2020)  
**Project:** CS-162 Final — educational reimplementation  
**Figures:** Paper-style synthetic figures live in `results/20260525_000000_baseline_synthetic/`. New GPU runs publish timestamped folders via `scripts/run_cartography_experiment.py` (see `results/runs_index.json`).

---

## Overview

This project reproduces the core ideas of Dataset Cartography: using **training dynamics** from a single model training run to assign each example coordinates on a **data map** (confidence vs. variability), diagnose dataset regions, guide data selection, detect label noise, and relate dynamics to uncertainty.

The pipeline runs in a conda environment (`cs162-cartography`) with experiment scripts under `scripts/`. Progress and metrics can be logged to [Weights & Biases](https://wandb.ai) via `wandb_credentials.txt`.

**Data source:** Figures can be generated from **paper-shaped synthetic coordinates** (default in `07_generate_insight_figures.py`) or from **real training dynamics** after GPU fine-tuning (see below).

**Training in this repo:** `scripts/run_cartography_experiment.py` and `scripts/train_all_models.py` fine-tune **DistilBERT, RoBERTa-base, Llama-3.2-1B, and Mistral-7B** on SNLI (4-bit on T4 for large models), with W&B logging and timestamped `results/<run_id>/` artifacts. Legacy entry: `scripts/train_and_collect_dynamics.py`. Selection bar charts (`fig03`, `fig04`) still use **published WinoGrande numbers** unless you add full retrain-on-subset runs yourself.

**Follow-up ideas implemented:**
- **Idea #1 (Preference / IT maps):** `--task preference` (UltraFeedback-style pairs) and `--task instruction` (Alpaca); high-variability subsets exported for alignment experiments.
- **Idea #2 (Dynamic maps):** `--task dynamic` saves per-epoch snapshots, region trajectories, and adaptive curriculum reweighting after `--curriculum-after-epoch`.

---

## Experiments Conducted

### 1. Training dynamics and data maps (Paper Section 2)

**Goal:** Map every training example by how the model’s belief in the gold label evolves across epochs.

**Method (this repo):**
- Input: per-epoch logs with `guid`, `epoch`, `gold_label`, `pred_label`, `prob_gold` (`scripts/01_collect_dynamics.py`), **or** direct synthetic coordinates (`ml_cartography/data/synthetic_cartography.py`).
- For each example, compute:
  - **Confidence** — mean predicted probability of the gold label across epochs.
  - **Variability** — standard deviation of that probability across epochs.
  - **Correctness** — fraction of epochs where the predicted label matches the gold label.
- Assign **regions**: easy-to-learn (high confidence, low variability), hard-to-learn (low confidence, low variability), ambiguous (high variability).

**Scripts:** `00_generate_toy_epoch_logs.py`, `01_collect_dynamics.py`, `02_build_data_map.py`, `07_generate_insight_figures.py`

**Figures:**
| File | Description |
|------|-------------|
| `20260525_000000_baseline_synthetic/fig01_data_map_correctness.png` | Main data map: variability (x) vs. confidence (y), points colored by correctness; annotated easy / hard / ambiguous regions |
| `fig02_density_histograms.png` | Marginal distributions of confidence, variability, and correctness |
| `fig08_region_composition.png` | Share of examples in each region (~68% easy, ~14% hard, ~18% ambiguous in synthetic run) |

**Insights:**
- Strong models produce a characteristic **bell-shaped** envelope: a dense **easy-to-learn** cluster (top-left), a smaller **hard-to-learn** cluster (bottom-left), and an **ambiguous** band along higher variability (often mid confidence).
- Easy-to-learn examples dominate the dataset; they are important for **optimization** but less so for OOD generalization in later experiments.
- Data maps are cheap to build after one training run and give a actionable view of dataset structure.

---

### 2. Data selection by region (Paper Section 3)

**Goal:** Test whether training only on examples from specific map regions changes in-distribution (ID) vs. out-of-distribution (OOD) performance.

**Method (paper):** Retrain RoBERTa-large from scratch on the top **33%** of examples under each strategy (ambiguous, hard-to-learn, high-confidence, random, etc.), evaluate on validation (ID) and challenge sets (OOD).

**Method (this repo):**
- `scripts/03_select_subsets.py` exports subset JSONL files for each strategy.
- `fig03_selection_id_ood_bars.png` plots **WinoGrande results from Table 2 in the paper** (not retrained in this repo).

**Figure:**
| File | Description |
|------|-------------|
| `fig03_selection_id_ood_bars.png` | Grouped bars: ID (validation) vs. OOD (WSC) accuracy per selection strategy |

**Insights (from paper, visualized here):**
- **Ambiguous** (highest variability) subset achieves the **best OOD** score (87.6% WSC) while using only one-third of the data — **above the 100% training baseline** (86.0% OOD).
- **Hard-to-learn** and **low-correctness** subsets also help OOD; **high-confidence**, **high-correctness**, and **low-variability** subsets perform **below random** on OOD.
- Challenging examples (ambiguous + hard) drive robustness; “easy” subsets hurt generalization even when ID accuracy looks acceptable.

---

### 3. Role of easy-to-learn examples (Paper Section 4)

**Goal:** Understand whether ambiguous-only training sets are enough to learn, and whether mixing in easy examples helps optimization at a cost to OOD performance.

**Method (paper):** On WinoGrande, train on decreasing fractions of the most ambiguous examples (50% down to 1%); ablate replacing part of a fixed 17% ambiguous set with easy-to-learn examples.

**Method (this repo):**
- `scripts/06_ambiguous_ablation.py` builds subset files and logs proxy metrics.
- `fig04_ambiguous_ablation_curves.png` uses **paper-reported curve shapes** (Fig. 3).

**Figure:**
| File | Description |
|------|-------------|
| `fig04_ambiguous_ablation_curves.png` | Left/center: ID and OOD accuracy vs. % ambiguous training data; right: effect of replacing easy examples into a 17% ambiguous core |

**Insights:**
- Training on **very small ambiguous-only** sets (&lt;25%) often **fails to optimize** (chance-level / majority baseline) — ambiguous points alone are not sufficient.
- **Random** subsets of the same size still learn, but with weaker OOD performance as data shrinks.
- Adding a **small fraction of easy-to-learn** examples to an ambiguous core restores ID accuracy but **reduces OOD** as the easy fraction grows — a tradeoff between optimization and generalization.

---

### 4. Detecting mislabeled examples (Paper Section 5)

**Goal:** Show that label noise moves examples on the map and that simple models can flag likely errors.

**Method (this repo):**
- Flip labels on the **1% highest-confidence (easiest)** examples (`ml_cartography/experiments/noise_injection.py`).
- Shift their dynamics toward **lower confidence** and **higher variability** (simulating post-retrain behavior).
- `scripts/04_detect_mislabeled.py` trains a **logistic regression** detector on confidence, variability, and correctness.

**Figure:**
| File | Description |
|------|-------------|
| `fig05_noise_injection_shift.png` | Log-density of confidence/variability before vs. after noise; scatter highlighting shifted noised points |

**Insights:**
- Noised (mislabeled) examples migrate toward the **hard-to-learn** region after retraining dynamics are updated.
- The **hard-to-learn** zone is enriched for labeling errors; confidence-based screening is a cheap cleaning signal (paper reports ~67–76% precision with human validation on SNLI / WinoGrande).
- Data maps support **dataset hygiene** without manual audit of the full corpus.

---

### 5. Connection to uncertainty (Paper Section 6)

**Goal:** Relate training dynamics to human disagreement (intrinsic uncertainty) and model uncertainty (dropout).

**Method (this repo):**
- `scripts/05_uncertainty_checks.py` computes Spearman correlation between dynamics and agreement / dropout proxies.
- Heatmap and regression figures use a **human-agreement proxy** tied to confidence when multi-annotator labels are unavailable.

**Figures:**
| File | Description |
|------|-------------|
| `fig06_human_agreement_heatmap.png` | Binned data map colored by mean human agreement (high agreement with high confidence) |
| `fig07_uncertainty_regression.png` | Scatter + trend: variability vs. dropout uncertainty; confidence vs. human agreement |

**Insights:**
- **Confidence** tracks **intrinsic uncertainty** (human agreement): when annotators agree, the model is consistently confident on the gold label.
- **Variability** tracks **model uncertainty** (flip-flopping predictions across epochs), aligned with dropout-based uncertainty in the paper.
- Training dynamics are a **lightweight alternative** to expensive ensemble or dropout uncertainty estimates.

---

## Summary of Main Takeaways

1. **One training run** yields a map that diagnoses an entire dataset.
2. **Ambiguous** examples are the best bet for **OOD robustness**; they often beat training on 100% of the data with only 33% selected.
3. **Easy-to-learn** examples are numerous and needed for **stable training**, but oversampling them hurts OOD performance.
4. **Hard-to-learn** examples flag **noise and difficulty**; useful for cleaning and selective training.
5. The field’s focus can shift from **more data** to **better-chosen data** using map coordinates.

---

## How to Reproduce

**Synthetic pipeline (no GPU):**
```bash
conda activate cs162-cartography
python scripts/run_all_experiments.py
python scripts/07_generate_insight_figures.py
```

**Trained dynamics (GPU):**
```bash
pip install -r requirements-train.txt
python scripts/train_and_collect_dynamics.py --preset distilbert --max-train-samples 20000
python scripts/01_collect_dynamics.py --input data/raw/epoch_predictions_snli_distilbert.jsonl
python scripts/02_build_data_map.py
python scripts/07_generate_insight_figures.py --input data/processed/cartography_with_regions.jsonl
```

---

## Figure Index

| Figure | Paper reference | Experiment |
|--------|-----------------|------------|
| `fig01_data_map_correctness.png` | Fig. 1–2 | Data maps |
| `fig02_density_histograms.png` | §2 histograms | Training dynamics distributions |
| `fig03_selection_id_ood_bars.png` | Table 2 | Data selection (WinoGrande) |
| `fig04_ambiguous_ablation_curves.png` | Fig. 3 | Easy vs. ambiguous ablation |
| `fig05_noise_injection_shift.png` | Fig. 4 | Mislabeled / noise detection |
| `fig06_human_agreement_heatmap.png` | Fig. 5 | Uncertainty — human agreement |
| `fig07_uncertainty_regression.png` | Fig. 7 (appendix) | Uncertainty — dropout regression |
| `fig08_region_composition.png` | §2 | Region composition summary |

---

## References

- Swayamdipta, S., Schwartz, R., Bauchnik, S., et al. (2020). *Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics.* EMNLP 2020.
- Official implementation: [allenai/cartography](https://github.com/allenai/cartography)
