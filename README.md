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

## Streamlit App

First install the base dependencies, then launch the read-only results explorer:

```bash
conda activate cs162-cartography
pip install -r requirements.txt
streamlit run apps/streamlit_app.py
```

The app will read the published experiment values from `results/<run_id>/`.

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
| `scripts/08_role_easy_to_learn.py` | §4 | WinoGrande ambiguous-size sweep + random baselines + easy replacements |
| `scripts/12_noise_detection_paper.py` | §5 | Label-noise injection + confidence-only noise detector |
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
python scripts/08_role_easy_to_learn.py --no-wandb
python scripts/12_noise_detection_paper.py --no-wandb
```

<<<<<<< Updated upstream
For the WinoGrande easy-to-learn experiment, the default command exports subset
files plus `data/processed/easy_role/train_commands.sh`. Add `--train` to launch
the retraining sweep from the same script.

## GPU training (real dynamics)
=======
## Full training suite (multi-model + Ideas #1/#2)

```bash
pip install -r requirements-train.txt
python scripts/train_suite.py --all --dynamic   # 4 SNLI models + preference + instruction
python scripts/train_suite.py --snli-encoders --only snli_distilbert
```

Artifacts: `experiments/runs/<timestamp>_<task>/` → published to `results/<timestamp>_<task>/`.  
See `results/report.md` and `configs/train_suite.json`.

| Job | Model |
|-----|--------|
| `snli_distilbert` | DistilBERT |
| `snli_roberta_base` | RoBERTa-base |
| `snli_llama_3_2_1b` | Llama-3.2-1B (LoRA) |
| `snli_mistral_7b` | Mistral-7B (LoRA) |
| `preference_ultrafeedback` | Preference data maps (Idea #1) |
| `instruction_alpaca` | Instruction tuning maps (Idea #1) |

`--dynamic` enables iterative snapshots + curriculum (Idea #2).

## GPU training (single model)
>>>>>>> Stashed changes

### Google Colab (T4)

1. **Runtime → T4 GPU → Restart session** (restart after enabling GPU).
2. Run `notebooks/colab_train_suite.ipynb` or `bash scripts/colab_setup.sh`.
3. Verify: `python scripts/check_cuda.py` → `cuda.is_available()=True`.

Do **not** `pip install -r requirements-train.txt` on Colab — it installs **CPU torch** and disables the GPU. Use `requirements-colab.txt` via `colab_setup.sh`. See [docs/COLAB_CUDA.md](docs/COLAB_CUDA.md).

4. Add `wandb_credentials.txt` / `hf_credentials.txt` (see examples).
5. Train: `python scripts/train_all_models.py --max-train-samples 10000 --epochs 5`

### Local / SSH with tmux

```bash
bash scripts/tmux_train_suite.sh   # one window per model
# or sequential:
python scripts/train_all_models.py --epochs 5
```

### Single experiment (recommended)

Writes dynamics, **dynamic epoch snapshots**, data-map PNGs, and copies everything to `results/<timestamp>_<task>_<preset>/`:

```bash
conda activate cs162-cartography
pip install -r requirements-train.txt

# SNLI + dynamic curriculum (Idea #2)
python scripts/run_cartography_experiment.py --task dynamic --preset distilbert --epochs 5

# Preference Data Maps — UltraFeedback / synthetic (Idea #1)
python scripts/run_cartography_experiment.py --task preference --preset distilbert --max-train-samples 3000

# Instruction-tuning dynamics — Alpaca (Idea #1)
python scripts/run_cartography_experiment.py --task instruction --preset distilbert --max-train-samples 2000
```

Legacy one-shot trainer (logs only, no timestamped `results/`):

```bash
python scripts/train_and_collect_dynamics.py --preset distilbert --max-train-samples 20000 --epochs 5
```

| Preset | Model | Colab T4 | Notes |
|--------|--------|----------|--------|
| `distilbert` | DistilBERT | yes | Fastest smoke test |
| `roberta-base` | RoBERTa-base | yes | Paper-like encoder |
| `llama-3.2-1b` | Llama 3.2 1B | yes (4-bit) | Accept [HF license](https://huggingface.co/meta-llama/Llama-3.2-1B) |
| `ministral-3b` | Unsloth Ministral 3 3B 4-bit | yes (~8GB) | `configs/train_snli_mistral.json` |

**Hugging Face:** see [docs/HUGGINGFACE_SETUP.md](docs/HUGGINGFACE_SETUP.md) — token + gated Llama acceptance.

### Results layout (Streamlit-ready)

- `results/report.md` — constant project report (this file stays at the root).
- `results/<YYYYMMDD_HHMMSS>_<task>_<preset>/` — per-run artifacts: `dynamics/`, `figures/`, `models/`, `manifest.json`.
- `results/runs_index.json` — index of baseline / migrated runs.

High-variability preference/SNLI subsets for DPO-style training are exported to `dynamics/subset_high_variability.jsonl` inside each run folder.

Use `--max-train-samples 0` for the full SNLI train split (~550k; slow).
