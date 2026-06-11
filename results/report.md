# CS-162 Dataset Cartography — Experiment Report

**Paper:** *Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics* (Swayamdipta et al., EMNLP 2020)  
<<<<<<< Updated upstream
**Project:** CS-162 Final — educational reimplementation with extensions  
**Primary training entry point:** `scripts/dual_gpu_train_suite.py` (local 3-GPU suite; replaces `notebooks/colab_train_suite.ipynb`)

---

## What `dual_gpu_train_suite.py` runs

This script orchestrates **baseline SNLI cartography training** for four model presets, then archives artifacts under `data/trained_models/<preset>/` so later runs (Idea #1 / Idea #2) can reuse trained weights without re-running the full suite.

### Phase 1 — Smoke tests (no Weights & Biases)

All three jobs launch **in parallel** on separate GPUs to verify loading, CUDA, Hugging Face auth, and 4-bit dependencies before a long run.

| GPU | Job | Task | Model | Train / val | Epochs |
|-----|-----|------|-------|-------------|--------|
| cuda:0 | `distilbert_snli` | `snli` | DistilBERT | 500 / 200 | 2 |
| cuda:1 | `llama_mini` | `dynamic` | Llama 3.2 1B (4-bit) | 200 / 200 | 1 |
| cuda:2 | `ministral_mini` | `dynamic` | Ministral 3 3B (Unsloth 4-bit) | 200 / 200 | 1 |
=======
**Project:** CS-162 Final — training, dynamic maps, preference cartography, Streamlit-ready artifacts

---

## Results layout

| Path | Purpose |
|------|---------|
| `results/report.md` | This file — constant overview (not tied to one run) |
| `results/<YYYYMMDD_HHMMSS>_<task>/` | One folder per experiment run (timestamped) |
| `results/experiment_index.json` | Index of all completed jobs and run IDs |
| `experiments/runs/<run_id>/` | Full training workspace (checkpoints, logs, dynamics) |

Each timestamped folder contains:

```
results/20260520_143022_snli_distilbert/
  manifest.json          # metadata for Streamlit / reporting
  config.json            # hyperparameters
  summary.json           # device, paths, metrics
  dynamics/
    epoch_predictions.jsonl
    cartography_coordinates.jsonl
    cartography_with_regions.jsonl
    region_trajectories.jsonl    # Idea #2 (if --dynamic)
    snapshots/epoch_*_coordinates.jsonl
  figures/               # data maps, preference maps, trajectory plots
  logs/training_metrics.jsonl
  models/final/          # saved model + tokenizer
```
>>>>>>> Stashed changes

- **No W&B logging** (`--no-wandb`)
- **Not archived** to `data/trained_models/` (ephemeral under `experiments/runs/`)

<<<<<<< Updated upstream
### Phase 2 — Full training (W&B enabled by default)

Three **parallel worker processes**, each pinned to one physical GPU via `CUDA_VISIBLE_DEVICES`:

| GPU | Models (order) | Task | Default hyperparameters |
|-----|----------------|------|-------------------------|
| cuda:0 | DistilBERT → RoBERTa-base | `dynamic` | 10k train, 2k val, 5 epochs |
| cuda:1 | Llama 3.2 1B | `dynamic` | same |
| cuda:2 | Ministral 3 3B | `dynamic` | same |

Each run invokes `scripts/run_cartography_experiment.py`, which:

1. Fine-tunes on **SNLI** and logs per-epoch gold-label probabilities  
2. Builds **cartography coordinates** (confidence, variability, correctness) and region labels  
3. Saves **per-epoch map snapshots** and optional **curriculum reweighting** (Idea #2; see below)  
4. Writes figures, dynamics JSONL, and model checkpoints  
5. Copies a full artifact tree to `data/trained_models/<preset>/` plus an entry in `data/trained_models/manifest.json`

**Default command (matches Colab overnight suite):**

```bash
conda activate cs162-cartography
python scripts/dual_gpu_train_suite.py
# equivalent training only:
python scripts/dual_gpu_train_suite.py --train-only --max-train-samples 10000 --epochs 5
```

**Related scripts (not run by the suite itself):**

| Script | Role |
|--------|------|
| `scripts/run_cartography_experiment.py` | Single experiment: train → maps → `results/<run_id>/` |
| `scripts/train_all_models.py` | Same four models, **sequential** on one machine |
| Colab notebook cells | Idea #1 preference / Idea #2 dynamic — run **after** Phase 2 archives exist |

---

## What the paper does (baseline we reproduce)

The paper trains a **single encoder** (e.g. RoBERTa-large) on a task such as SNLI or WinoGrande, records **training dynamics** each epoch (probability of the gold label, predicted label), and plots a **data map**:

- **x:** variability (std of gold-label probability across epochs)  
- **y:** confidence (mean gold-label probability)  
- **Regions:** easy-to-learn, hard-to-learn, ambiguous  

Downstream uses include subset selection (§3), ambiguous-only ablations (§4), mislabel detection (§5), and uncertainty analysis (§6). The numbered pipeline `scripts/00`–`07` implements these analyses on logged dynamics (synthetic or real).

---

## Extensions beyond the paper (what we add)

These are **not** in Swayamdipta et al. (2020); they are course/project add-ons built on the same dynamics machinery.

| Extension | Paper? | Summary |
|-----------|--------|---------|
| **Four model families** | Partially | Paper focuses on one model per task; we run DistilBERT, RoBERTa-base, Llama 3.2 1B, and Ministral 3 3B on SNLI for comparison |
| **Causal / 4-bit LMs on NLI** | No | Llama (bitsandbytes NF4) and Ministral (Unsloth pre-quantized checkpoint) as sequence-classification heads on SNLI |
| **3-GPU parallel orchestration** | No | Smoke + full training split across GPUs; encoders share GPU 0, one LLM per GPU |
| **Timestamped `results/` + `data/trained_models/`** | No | Streamlit-ready run folders and a stable archive for follow-up experiments |
| **W&B integration** | No | Metrics and map images logged per run (disabled for smoke tests) |
| **Idea #1 — Preference / instruction maps** | No | Cartography on preference pairs and instruction data (see below) |
| **Idea #2 — Dynamic maps + curriculum** | No | Per-epoch snapshots, region trajectories, adaptive sampling (see below) |

**Practical deviations from paper protocol:**

- **Subset size:** default 10k SNLI train samples (not full ~550k) for feasible GPU time  
- **Task:** suite Phase 2 uses `--task dynamic` (snapshots + curriculum), not plain `snli`  
- **Selection / OOD bars (`fig03`, `fig04`):** still use **published WinoGrande numbers** unless you add full retrain-on-subset jobs  
- **Model scale:** Ministral 3 3B instead of paper’s RoBERTa-large / BERT-era encoders for the “large model” slot  

---

## Idea #1 — Preference Data Maps (alignment / RLHF-style data)

**Motivation:** The paper maps **classification** dynamics. Modern alignment pipelines use **preference pairs** (chosen vs. rejected). Idea #1 asks: *Can we build a “Preference Data Map” using the same confidence/variability ideas on pair-level training?*

**What it does in this repo:**

- **Task:** `--task preference` in `run_cartography_experiment.py`  
- **Data:** UltraFeedback-style preference pairs (or synthetic pairs in tests)  
- **Dynamics:** Per-epoch signals on whether the model ranks the **chosen** response above the **rejected** (margin / probability on the preferred side)  
- **Map:** Preference-specific regions (e.g. clear preference, borderline, inconsistent) via `ml_cartography/analysis/preference_map.py`  
- **Export:** `dynamics/subset_high_variability.jsonl` — ambiguous / high-variability pairs for **DPO-style** or selective training  

**Optional variant:** `--task instruction` logs dynamics on **instruction-tuning** (Alpaca-style) examples for the same diagnostic view on IT data.

**How to run (after baseline weights exist in `data/trained_models/`):**

```bash
python scripts/run_cartography_experiment.py --task preference --preset distilbert \
  --max-train-samples 3000 --epochs 5 --wandb-run-name colab_preference
```

**Not run by `dual_gpu_train_suite.py`** — execute separately once SNLI baselines are archived.

---

## Idea #2 — Dynamic cartography (snapshots, trajectories, curriculum)

**Motivation:** The paper’s map uses **one** aggregate point per example after full training. Idea #2 treats the map as **time-varying**: where each example sits after each epoch, how **regions change** over training, and whether **reweighting** the training sampler toward ambiguous examples improves learning.

**What it does in this repo (`--task dynamic`):**

1. **Per-epoch snapshots** — `dynamics/snapshots/epoch_XXX_coordinates.jsonl` after each epoch  
2. **Region trajectories** — `dynamics/region_trajectories.jsonl` (easy → ambiguous → hard paths per `guid`)  
3. **Adaptive curriculum** — after `--curriculum-after-epoch` (default **2** in the full suite):
   - Upweight **ambiguous** examples (`ambiguous_boost`, default 2.5×)  
   - Downweight **easy** examples (`easy_scale`, default 0.4×)  
   - Slight boost for **hard** examples (`hard_boost`, default 1.2×)  
   - Implemented in `ml_cartography/training/dynamic_cartography.py` via `WeightedRandomSampler`  

**Phase 2 of `dual_gpu_train_suite.py` always uses `dynamic`** with `curriculum_after_epoch=2` and 5 epochs, so every archived model includes snapshots + curriculum behavior.

**Standalone Idea #2 example (e.g. RoBERTa only):**

```bash
python scripts/run_cartography_experiment.py --task dynamic --preset roberta-base \
  --curriculum-after-epoch 2 --epochs 5 --wandb-run-name colab_dynamic
=======
## Training suite (`scripts/train_suite.py`)

Single entry point for all models and follow-up ideas:

```bash
conda activate cs162-cartography
pip install -r requirements-train.txt

# Everything (encoders + Llama + Mistral + preference + instruction + dynamic maps)
python scripts/train_suite.py --all --dynamic

# SNLI encoder models only
python scripts/train_suite.py --snli-encoders --dynamic

# One job
python scripts/train_suite.py --only snli_distilbert --max-train-samples 20000 --epochs 5
```

### Models trained on SNLI

| Job ID | Model | Type |
|--------|--------|------|
| `snli_distilbert` | DistilBERT | Encoder classifier |
| `snli_roberta_base` | RoBERTa-base | Encoder classifier |
| `snli_llama_3_2_1b` | Llama-3.2-1B-Instruct | Causal LM + LoRA (4-bit) |
| `snli_mistral_7b` | Mistral-7B-Instruct | Causal LM + LoRA (4-bit) |

Paper default is RoBERTa-**large**; use `--only` with `roberta-large` via `train_and_collect_dynamics.py` if you have 24GB+ VRAM.

---

## Idea #1: Preference & instruction data maps

**Goal:** Extend cartography to RLHF/DPO-style preference data and instruction tuning.

**Runs:**
- `preference_ultrafeedback` — `HuggingFaceH4/ultrafeedback_binarized` pairs (chosen/rejected)
- `instruction_alpaca` — `yahma/alpaca-cleaned` prompt–response SFT

**Dynamics:** Each epoch logs reward margin (log-prob chosen − log-prob rejected) or response log-prob; aggregated into **confidence** and **variability** like the paper.

**Outputs:** `preference_data_map.png`, coordinates under `dynamics/`, high-variability pairs for DPO filtering (export via subset scripts).

**Hypothesis:** High-variability preferences are most informative for alignment (less sycophancy, better reasoning) — test with MT-Bench / human eval after DPO on filtered subsets.

---

## Idea #2: Dynamic / iterative data maps

**Goal:** Recompute maps during training; track region movement; adaptive curriculum.

**Enabled with:** `--dynamic` on `train_suite.py`

**Mechanism:**
1. Every `--map-interval` epochs → snapshot `dynamics/snapshots/epoch_XXX_coordinates.jsonl`
2. `region_trajectories.jsonl` — per-example path easy → ambiguous → hard
3. **Curriculum sampler** — upsample ambiguous (2.5×), downsample easy (0.4×) before next epoch
4. Figure: `dynamic_region_trajectories.png`

**Use case:** Continual / multi-stage fine-tuning where static maps miss examples that change role over time.

---

## Core experiments (paper §2–§6)

| § | Experiment | Script / run |
|---|------------|----------------|
| 2 | Training dynamics + data maps | All `snli_*` jobs + `07_generate_insight_figures.py` |
| 3 | Data selection by region | `03_select_subsets.py` on trained coordinates |
| 4 | Easy vs ambiguous ablation | `06_ambiguous_ablation.py` |
| 5 | Mislabeled detection | `04_detect_mislabeled.py` |
| 6 | Uncertainty / agreement | `05_uncertainty_checks.py` |

After training, regenerate figures for a specific run:

```bash
python scripts/07_generate_insight_figures.py \
  --input results/<run_id>/dynamics/cartography_with_regions.jsonl \
  --run-id <run_id>
>>>>>>> Stashed changes
```

---

<<<<<<< Updated upstream
## Artifact layout

| Location | Contents |
|----------|----------|
| `experiments/runs/<timestamp>_<task>_<preset>/` | Raw run: config, dynamics, figures, checkpoints |
| `results/<timestamp>_<task>_<preset>/` | Published copy (if not `--no-publish`) |
| `data/trained_models/<preset>/` | **Archive** after Phase 2: models, dynamics, maps, `archive_meta.json` |
| `data/trained_models/manifest.json` | Index of archived presets for downstream Idea #1 / #2 |
| `experiments/dual_gpu_train_suite_summary.json` | Smoke + train exit codes and timing |

---

## Suggested workflow

1. **Install deps:** `pip install -r requirements-train.txt` (`bitsandbytes`, `unsloth`, etc.)  
2. **Credentials:** `hf_credentials.txt` (and optional `wandb_credentials.txt`) in repo root  
3. **Run suite:** `python scripts/dual_gpu_train_suite.py`  
4. **Idea #1:** preference / instruction experiments using archived encoders or fresh runs  
5. **Analysis:** `scripts/02_build_data_map.py`, `07_generate_insight_figures.py`, or Streamlit over `results/`  
=======
## Streamlit app (preview)

```bash
pip install streamlit
streamlit run apps/streamlit_app.py
```

Loads `results/<run_id>/` — future: upload dataset → trigger `train_suite` → filter regions interactively.

---

## Main insights (paper + project)

1. **Data maps** from one training run diagnose easy / hard / ambiguous regions.
2. **Ambiguous** points often maximize OOD gains when used as 33% train subsets.
3. **Easy** points dominate and stabilize optimization but can hurt OOD if over-sampled.
4. **Hard** points correlate with label noise; confidence-based filtering helps cleaning.
5. **Preference maps** (Idea #1) apply the same logic to chosen/rejected margins for alignment data.
6. **Dynamic maps** (Idea #2) show examples migrating between regions and support adaptive curricula.
>>>>>>> Stashed changes

---

## References

<<<<<<< Updated upstream
- Swayamdipta, S., et al. (2020). *Dataset Cartography.* EMNLP 2020.  
- [allenai/cartography](https://github.com/allenai/cartography)  
- Course pipeline: `README.md`, `scripts/00`–`07`, synthetic figures under `results/20260525_000000_baseline_synthetic/`
=======
- Swayamdipta et al. (2020). EMNLP. [Paper](https://aclanthology.org/2020.emnlp-main.746/)
- [allenai/cartography](https://github.com/allenai/cartography)
>>>>>>> Stashed changes
