#!/usr/bin/env bash
# §6 extension: gain-per-compute on region-only subsets (paper repro plan item 6). One-line:
#   bash scripts/run_exp_10_subregion_compute_gain.sh
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/lib/experiment_env.sh"
exp_activate_conda
exp_apply_paper_defaults
exp_require_input

SESSION="${SESSION:-cs162-exp-compute-gain}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed/subregion_compute_gain}"
GPUS="${GPUS:-${EXP_GPUS}}"
RESTARTS="${RESTARTS:-1}"
EPOCH_SWEEP="${EPOCH_SWEEP:-1 2 3 4 5 6}"
BATCH_SIZE="${BATCH_SIZE:-${EXP_BATCH_SIZE_WINO}}"
WANDB_GROUP="${WANDB_GROUP:-compute-gain}"
LOG="$(exp_log_file exp_10_compute_gain)"
ROOT="$(exp_repo_root)"

CMD="cd '$ROOT' && \
  for EPOCHS in $EPOCH_SWEEP; do \
    for STRATEGY in high_variability low_confidence high_confidence random; do \
      echo \">>> epochs=\$EPOCHS strategy=\$STRATEGY\"; \
      $PYTHON scripts/09_region_finetune.py --train --dataset winogrande \
        --preset '$EXP_PRESET' --epochs \"\$EPOCHS\" --batch-size '$BATCH_SIZE' \
        --learning-rate '$EXP_LEARNING_RATE' \
        --strategies \"\$STRATEGY\" --restarts '$RESTARTS' --gpus '$GPUS' \
        --input '$EXP_INPUT' \
        --output-dir '$OUTPUT_DIR/ep\${EPOCHS}_\${STRATEGY}' \
        --wandb-run-name compute_gain_e\${EPOCHS}_\${STRATEGY} \
        --wandb-group '$WANDB_GROUP'; \
    done; \
  done && \
  bash scripts/export_all_metrics.sh '$WANDB_GROUP' && \
  echo '=== compute-gain sweep complete ==='; bash"

exp_tmux_start "$SESSION" compute-gain "$CMD 2>&1 | tee '$LOG'"
exp_print_launch_summary "$SESSION" \
  "epoch sweep: $EPOCH_SWEEP" \
  "strategies: high_variability low_confidence high_confidence random" \
  "GPUs: $GPUS" \
  "output: $OUTPUT_DIR" \
  "log: $LOG" \
  "W&B group: $WANDB_GROUP"
