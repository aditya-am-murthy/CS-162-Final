# Dataset Cartography — Re-implementation & Extension

CS-162 final project built on [Swayamdipta et al., EMNLP 2020](https://aclanthology.org/2020.emnlp-main.746/). This repo **reproduces the paper’s analyses** (data maps, subset selection, ablations, noise & uncertainty) in a modular, student-readable codebase—and **extends** cartography to new models, tasks, and training regimes.

All report figures/tables: [`paper_outputs/`](paper_outputs/)

---

## Abstract

We re-implemented Dataset Cartography end-to-end: per-epoch dynamics logging, adaptive region labeling, subset export, multi-GPU retraining, and paper-style plotting. On WinoGrande and subsampled NLI runs we recover the paper’s qualitative trends—**easy / hard / ambiguous** map regions; **ambiguous** and **hard-to-learn** 33% slices beating high-confidence baselines on OOD proxies; **easy-to-learn** examples required for optimization in small ambiguous-only sets; label-noise shifts toward low confidence; confidence correlating with human agreement.

**Our contributions (beyond the paper):** (1) a modular `ml_cartography/` library + numbered `scripts/00–12` pipeline with tmux launchers and W&B/CSV export; (2) **multi-architecture** SNLI maps (DistilBERT, RoBERTa, Llama-3.2-1B, Ministral-3B); (3) **preference & instruction** cartography (Idea #1) and **dynamic per-epoch maps + curriculum sampling** (Idea #2); (4) a **Streamlit** run browser; (5) automated `paper_outputs/` collection with measured tables (ID + OOD columns, not ad-hoc bar charts).

---

## Results gallery

<sub>Click filenames in [`paper_outputs/`](paper_outputs/) for full resolution.</sub>

**Data maps (§2)** — SNLI · WinoGrande

<p align="center">
  <img src="paper_outputs/Figure_01_snli_data_map.png" width="300" alt="Fig 1 SNLI" />
  &nbsp;
  <img src="paper_outputs/Figure_02_winogrande_data_map.png" width="300" alt="Fig 2 WinoGrande" />
</p>

**Subset selection (§3)** — ambiguous/hard-to-learn 33% ≈ random on ID, better OOD proxy vs high-confidence

<p align="center">
  <img src="paper_outputs/Table_02_winogrande_selection.png" width="320" alt="Table 2" />
  <img src="paper_outputs/Table_03_snli_selection.png" width="240" alt="Table 3 SNLI" />
  <img src="paper_outputs/Table_04_qnli_selection.png" width="240" alt="Table 4 QNLI" />
</p>

**Ambiguous scaling & noise (§4–§5)**

<p align="center">
  <img src="paper_outputs/Figure_03_easy_to_learn_role.png" width="480" alt="Fig 3" />
</p>

<p align="center">
  <img src="paper_outputs/Figure_04_noise_shift.png" width="300" alt="Fig 4" />
</p>

**Human agreement heatmap (§6)** — confidence tracks annotator agreement

<p align="center">
  <img src="paper_outputs/Figure_05_human_agreement_heatmap.png" width="280" alt="Fig 5" />
</p>

| Takeaway | Measured signal |
|----------|-----------------|
| Map structure | Three regions stable across SNLI & WinoGrande |
| Table 2 | ambiguous OOD **63.2%** vs random **61.2%**; high-confidence ID **67.1%** |
| Fig 3 | Tiny ambiguous-only sets ≈ chance ID; easy replacement restores learning |
| Fig 4 | Noised easy examples shift to low-confidence / higher-variability |
| Fig 5 | High-confidence bins → high human agreement |

*Values from our RoBERTa runs (often 1 seed; NLI subsampled). OOD = proxy evaluators. Trends align with the paper; absolute numbers differ from 3-seed RoBERTa-large tables.*

---

## Extensions

| Extension | Significance |
|-----------|--------------|
| Multi-architecture maps | Separates dataset structure from encoder capacity |
| Idea #1: preference / instruction maps | Cartography for alignment & IT data; exports high-var pairs for selective training |
| Idea #2: dynamic maps + curriculum | Per-epoch snapshots + ambiguous upweighting after epoch 2 |
| `run_exp_*` + 5-GPU orchestration | Reproducible tmux launchers per paper figure/table |
| Streamlit app | Browse `results/<run_id>/` without reading JSONL |

Details: [`results/report.md`](results/report.md) · Launchers: [`scripts/EXPERIMENT_LAUNCHERS.md`](scripts/EXPERIMENT_LAUNCHERS.md)

---

## Repo layout

```
ml_cartography/     core library (dynamics, maps, paper_figures, paper_tables)
scripts/            00–12 pipeline, run_exp_01–10, collect_paper_outputs.py
paper_outputs/      Figure_* and Table_* for the write-up
results/            timestamped runs + region_metrics_*.json
data/processed/     subset manifests & finetune artifacts
apps/streamlit_app.py
```

**Regenerate outputs:** `python scripts/plot_from_metrics_csv.py --no-wandb && python scripts/collect_paper_outputs.py`

---

## Quick start

```bash
bash scripts/setup_conda_env.sh && conda activate cs162-cartography
pip install -r requirements-train.txt   # local GPU training

python scripts/run_cartography_experiment.py --task snli --preset roberta-base --epochs 3
python scripts/collect_paper_outputs.py
streamlit run apps/streamlit_app.py
```

Paper experiments: `bash scripts/run_exp_04_table2_winogrande.sh` (etc.) — see [`scripts/EXPERIMENT_LAUNCHERS.md`](scripts/EXPERIMENT_LAUNCHERS.md).

Colab: `notebooks/colab_train_suite.ipynb` · HF setup: [`docs/HUGGINGFACE_SETUP.md`](docs/HUGGINGFACE_SETUP.md)

---

## Citation

```bibtex
@inproceedings{swayamdipta2020dataset,
  title={Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics},
  author={Swayamdipta, Swabha and others},
  booktitle={EMNLP}, year={2020}
}
```
