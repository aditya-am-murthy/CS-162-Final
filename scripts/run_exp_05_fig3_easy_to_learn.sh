#!/usr/bin/env bash
# Fig 3: ambiguous scaling + easy replacement curves (paper §4). One-line:
#   bash scripts/run_exp_05_fig3_easy_to_learn.sh
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/lib/experiment_env.sh"
exp_activate_conda
exp_apply_paper_defaults
exp_require_input

SESSION="${SESSION:-cs162-exp-fig3}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed/easy_role}"
EPOCHS="${EPOCHS:-${EXP_EPOCHS_WINO}}"
BATCH_SIZE="${BATCH_SIZE:-${EXP_BATCH_SIZE_WINO}}"
RESTARTS="${RESTARTS:-${EXP_RESTARTS}}"
GPUS="${GPUS:-${EXP_GPUS}}"
WANDB_NAME="${WANDB_NAME:-fig3_easy_to_learn}"
WANDB_GROUP="${WANDB_GROUP:-fig3-winogrande}"
LOG="$(exp_log_file exp_05_fig3)"
ROOT="$(exp_repo_root)"

CMD="cd '$ROOT' && $PYTHON scripts/08_role_easy_to_learn.py \
    --train --dataset winogrande --preset '$EXP_PRESET' \
    --epochs '$EPOCHS' --batch-size '$BATCH_SIZE' \
    --learning-rate '$EXP_LEARNING_RATE' \
    --max-train-samples '$EXP_MAX_TRAIN_SAMPLES' \
    --max-eval-samples '$EXP_MAX_EVAL_SAMPLES' \
    --winogrande-config '$EXP_WINOGRANDE_CONFIG' \
    --restarts '$RESTARTS' --gpus '$GPUS' \
    --ambiguous-ratios 0.50 0.33 0.25 0.17 0.10 0.05 0.01 \
    --replace-ratios 0.0 0.10 0.25 0.50 0.75 \
    --core-ambiguous-ratio 0.17 \
    --input '$EXP_INPUT' --output-dir '$OUTPUT_DIR' \
    --wandb-run-name '$WANDB_NAME' --wandb-group '$WANDB_GROUP' && \
  bash scripts/export_all_metrics.sh '$WANDB_GROUP' && \
  bash scripts/plot_from_metrics_csv.py --experiment-tag fig3 && \
  echo '=== Fig 3 complete ==='; bash"

exp_tmux_start "$SESSION" fig3 "$CMD 2>&1 | tee '$LOG'"
exp_print_launch_summary "$SESSION" \
  "GPUs: $GPUS | restarts: $RESTARTS" \
  "epochs: $EPOCHS | preset: $EXP_PRESET" \
  "output: $OUTPUT_DIR" \
  "log: $LOG" \
  "W&B: $WANDB_NAME (group $WANDB_GROUP)"
