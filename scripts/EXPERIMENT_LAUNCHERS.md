# Paper reproduction launchers

One-line bash scripts for every experiment in `docs/local-docs/remaining-experiments.md`.
Each launcher runs in **tmux**, logs to **wandb**, streams **per-epoch progress** (via scripts 08/09/12),
and exports metrics to **local CSV** when finished.

## Quick start

```bash
# Highest priority (WinoGrande Table 2 + Fig 3)
bash scripts/run_exp_04_table2_winogrande.sh
bash scripts/run_exp_05_fig3_easy_to_learn.sh

# Export all metrics to CSV (local JSONL + wandb)
bash scripts/export_all_metrics.sh paper-reproduction

# Plot measured curves from CSV
bash scripts/plot_from_metrics_csv.py --experiment-tag table2
```

## Global hyperparameters (env vars)

| Variable | Default | Description |
|----------|---------|-------------|
| `EXP_PRESET` | `roberta-large` | Model preset |
| `EXP_EPOCHS_WINO` | `6` | WinoGrande epochs |
| `EXP_EPOCHS_SNLI` | `6` | SNLI epochs |
| `EXP_EPOCHS_MNLI` | `5` | MNLI epochs |
| `EXP_EPOCHS_QNLI` | `5` | QNLI epochs |
| `EXP_BATCH_SIZE_WINO` | `64` | WinoGrande batch size |
| `EXP_BATCH_SIZE_NLI` | `96` | NLI batch size |
| `EXP_MAX_TRAIN_SAMPLES` | `0` | `0` = full dataset |
| `EXP_MAX_EVAL_SAMPLES` | `0` | `0` = full eval |
| `EXP_RESTARTS` | `3` | Seeds per strategy (paper §3) |
| `EXP_GPUS` | `0,1,2,3,4` | Parallel training GPUs |
| `EXP_INPUT` | WinoGrande adaptive map | `cartography_with_regions.jsonl` |
| `EXP_WANDB_GROUP` | `paper-reproduction` | W&B group tag |

Example:

```bash
RESTARTS=3 GPUS=0,1,2 EPOCHS=6 bash scripts/run_exp_04_table2_winogrande.sh
```

## Experiment scripts

| Script | Paper target | Python driver |
|--------|--------------|---------------|
| `run_exp_01_datamap_snli.sh` | Fig 1 SNLI map | `run_cartography_experiment.py` |
| `run_exp_02_datamap_mnli_qnli.sh` | Appendix maps | same (2 GPUs) |
| `run_exp_03_polish_maps.sh` | Map polish + insight figs | `rebuild_fixed_maps.py`, `07` |
| `run_exp_04_table2_winogrande.sh` | Table 2 | `09_region_finetune.py`, `11` |
| `run_exp_05_fig3_easy_to_learn.sh` | Fig 3 | `08_role_easy_to_learn.py` |
| `run_exp_06_fig4_noise.sh` | Fig 4 | `12_noise_detection_paper.py` |
| `run_exp_07_fig5_uncertainty.sh` | Fig 5 | `05`, `plot_from_metrics_csv.py` |
| `run_exp_08_table3_snli_mnli.sh` | Table 3 | `09_region_finetune.py` |
| `run_exp_09_table4_qnli.sh` | Table 4 | `09_region_finetune.py`, `11` |
| `run_exp_10_subregion_compute_gain.sh` | Compute vs gain | `09` epoch sweep |
| `run_exp_11_extra_bilateral_noise.sh` | **Extra #4** bilateral 1% flip | `13_bilateral_noise_flip.py` |
| `run_all_remaining_experiments.sh` | All of the above | tmux multi-window queue |

## Extra experiments (beyond paper)

| # | Script / collector | Output |
|---|-------------------|--------|
| 1 Multi-architecture maps | `collect_extension_outputs.py` | `extension_outputs/Extra_01_*` |
| 2 Preference cartography | same | `Extra_02_preference_data_map.png` |
| 3 Dynamic curriculum | same | `Extra_03_dynamic_*` |
| 4 Bilateral 1% flip | `run_exp_11_extra_bilateral_noise.sh` | `Extra_04_bilateral_*` |

```bash
# Extra #4: easy arm reuses Fig 4 run; hard arm runs 5 restarts on GPUs 0–4
GPUS=0,1,2,3,4 RESTARTS=5 bash scripts/run_exp_11_extra_bilateral_noise.sh
python scripts/collect_extension_outputs.py
```

## CSV outputs

| File | Contents |
|------|----------|
| `results/experiment_metrics_history.csv` | Per-epoch metrics from all `training_metrics.jsonl` |
| `results/experiment_metrics_summary.csv` | One row per training run (final metrics) |
| `results/wandb_metrics_history.csv` | W&B cloud/local cache history |
| `results/paper_plots_from_metrics/` | Fig 3/4/5/Table 2 plots from measured data |
