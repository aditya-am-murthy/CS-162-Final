#!/usr/bin/env bash
# Table 4: QNLI 33% ambiguous/hard vs random/100% (paper §3). One-line:
#   bash scripts/run_exp_09_table4_qnli.sh
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/lib/experiment_env.sh"
exp_activate_conda
exp_apply_paper_defaults

SESSION="${SESSION:-cs162-exp-table4}"
QNLI_INPUT="${QNLI_INPUT:-results/20260527_051157_snli_qnli_roberta-base/dynamics/cartography_with_regions.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed/table4_qnli}"
EPOCHS="${EPOCHS:-${EXP_EPOCHS_QNLI}}"
BATCH_SIZE="${BATCH_SIZE:-${EXP_BATCH_SIZE_NLI}}"
RESTARTS="${RESTARTS:-${EXP_RESTARTS}}"
GPUS="${GPUS:-${EXP_GPUS}}"
MAX_TRAIN="${MAX_TRAIN:-${EXP_MAX_TRAIN_SAMPLES}}"
WANDB_NAME="${WANDB_NAME:-table4_qnli}"
WANDB_GROUP="${WANDB_GROUP:-table4-qnli}"
LOG="$(exp_log_file exp_09_table4)"
ROOT="$(exp_repo_root)"

CMD="cd '$ROOT' && $PYTHON scripts/09_region_finetune.py \
    --train --dataset qnli --preset '$EXP_PRESET' \
    --epochs '$EPOCHS' --batch-size '$BATCH_SIZE' \
    --learning-rate '$EXP_LEARNING_RATE' \
    --max-train-samples '$MAX_TRAIN' --max-eval-samples '$EXP_MAX_EVAL_SAMPLES' \
    --strategies high_variability low_confidence random full \
    --restarts '$RESTARTS' --gpus '$GPUS' \
    --input '$QNLI_INPUT' --output-dir '$OUTPUT_DIR' \
    --wandb-run-name '$WANDB_NAME' --wandb-group '$WANDB_GROUP' && \
  $PYTHON scripts/11_region_metrics.py \
    --results '$OUTPUT_DIR/train_results.json' \
    --manifest '$OUTPUT_DIR/manifest.json' \
    --output results/region_metrics_table4.json \
    --wandb-group '$WANDB_GROUP' --wandb-run-name '${WANDB_NAME}_metrics' && \
  bash scripts/export_all_metrics.sh '$WANDB_GROUP' && \
  bash scripts/plot_from_metrics_csv.py --experiment-tag table4 && \
  echo '=== Table 4 complete ==='; bash"

exp_tmux_start "$SESSION" table4 "$CMD 2>&1 | tee '$LOG'"
exp_print_launch_summary "$SESSION" \
  "GPUs: $GPUS | restarts: $RESTARTS | epochs: $EPOCHS" \
  "input: $QNLI_INPUT" \
  "output: $OUTPUT_DIR" \
  "log: $LOG" \
  "W&B: $WANDB_NAME (group $WANDB_GROUP)"
