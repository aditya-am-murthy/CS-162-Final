# CS-162 Final: Simple Dataset Cartography

This repository recreates the *Dataset Cartography* paper.
It is an original educational reimplementation with student-friendly code.

## Project Layout

- `ml_cartography/core/` - core training-dynamics logic.
- `ml_cartography/analysis/` - data-map region labeling and plotting.
- `ml_cartography/experiments/` - subset selection, noise detection, uncertainty checks.
- `scripts/` - numbered end-to-end pipeline commands (tqdm + Weights & Biases).
- `data/raw/` - input per-epoch prediction logs.
- `data/processed/` - generated coordinates and subsets.
- `data/outputs/` - intermediate plots from the pipeline.
- `results/` - final paper-style insight visualizations.
- `configs/` - simple JSON pipeline examples.

## Setup (conda)

Requires [Miniconda](https://docs.anaconda.com/miniconda/) or Anaconda. From the repo root:

```bash
bash scripts/setup_conda_env.sh
conda activate cs162-cartography
# edit wandb_credentials.txt in the repo root (see wandb_credentials.example.txt)
# optional: wandb login
# use --no-wandb to skip logging entirely
```

If `conda` is not on your PATH after a fresh Miniconda install:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cs162-cartography
```

Manual alternative:

```bash
conda env create -f environment.yml
conda activate cs162-cartography
```

`requirements.txt` mirrors pip-only deps for non-conda installs.

## Experiment Scripts (maps to `docs/local-docs/experiments.md`)

| Script | Paper section | What it does |
|--------|---------------|--------------|
| `scripts/00_generate_toy_epoch_logs.py` | — | Synthetic per-epoch logs for local testing |
| `scripts/01_collect_dynamics.py` | §2 | Confidence, variability, correctness |
| `scripts/02_build_data_map.py` | §2 | Regions + scatter plot + histograms |
| `scripts/03_select_subsets.py` | §3 | Top-33% selection strategies (+ proxy metrics) |
| `scripts/04_detect_mislabeled.py` | §5 | Noise injection + linear detector |
| `scripts/05_uncertainty_checks.py` | §6 | Spearman vs human agreement / dropout proxy |
| `scripts/06_ambiguous_ablation.py` | §4 | Ambiguous-only sweeps + easy replacement |
| `scripts/07_generate_insight_figures.py` | §2–§6 | Paper-style figures → `results/` |
| `scripts/run_all_experiments.py` | all | Runs the full pipeline |

Run everything (monitor on [wandb.ai](https://wandb.ai)):

```bash
conda activate cs162-cartography
python scripts/run_all_experiments.py --wandb-project cs162-dataset-cartography
```

Or step by step:

```bash
conda activate cs162-cartography
python scripts/00_generate_toy_epoch_logs.py
python scripts/01_collect_dynamics.py --input data/raw/epoch_predictions_toy.jsonl
python scripts/02_build_data_map.py
python scripts/03_select_subsets.py --run-all-strategies
python scripts/04_detect_mislabeled.py
python scripts/05_uncertainty_checks.py
python scripts/06_ambiguous_ablation.py
```

## GPU training (real dynamics)

Fine-tune on **SNLI** and write per-epoch logs the cartography pipeline expects:

```bash
conda activate cs162-cartography
pip install -r requirements-train.txt
# CUDA example (pick matching wheel from https://pytorch.org):
# pip install torch --index-url https://download.pytorch.org/whl/cu124

# fast (~minutes on one GPU): DistilBERT, 20k examples
python scripts/train_and_collect_dynamics.py --preset distilbert --max-train-samples 20000 --epochs 5

# larger: RoBERTa-base, 50k examples (use smaller batch if OOM)
python scripts/train_and_collect_dynamics.py --config configs/train_snli_roberta_base.json

# then cartography on *trained* logs (not toy synthetic)
python scripts/01_collect_dynamics.py --input data/raw/epoch_predictions_snli_distilbert.jsonl
python scripts/02_build_data_map.py --input data/processed/cartography_coordinates.jsonl
python scripts/07_generate_insight_figures.py --input data/processed/cartography_with_regions.jsonl
```

| Preset | Model | VRAM (rough) | Notes |
|--------|--------|--------------|--------|
| `distilbert` | DistilBERT | ~4–6 GB | Best default for 1 GPU |
| `roberta-base` | RoBERTa-base | ~8–12 GB | Paper-like, smaller than large |
| `roberta-large` | RoBERTa-large | ~24 GB+ | Paper default; reduce batch / samples |

Train on a **cartography subset** (after steps 01–03 on full trained logs):

```bash
python scripts/train_and_collect_dynamics.py --preset distilbert \\
  --subset-file data/processed/selected_high_variability_33pct.jsonl \\
  --output data/raw/epoch_predictions_subset_ambiguous.jsonl
```

Use `--max-train-samples 0` for the full SNLI train split (~550k; slow). Paper uses RoBERTa-**large** for 5–6 epochs; this repo supports smaller models for coursework GPUs.