# CS-162 Dataset Cartography — Experiment Report

**Paper:** *Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics* (Swayamdipta et al., EMNLP 2020)  
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

- **No W&B logging** (`--no-wandb`)
- **Not archived** to `data/trained_models/` (ephemeral under `experiments/runs/`)

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
```

---

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

---

## References

- Swayamdipta, S., et al. (2020). *Dataset Cartography.* EMNLP 2020.  
- [allenai/cartography](https://github.com/allenai/cartography)  
- Course pipeline: `README.md`, `scripts/00`–`07`, synthetic figures under `results/20260525_000000_baseline_synthetic/`
