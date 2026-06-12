#!/usr/bin/env bash
# Extra experiment #4: bilateral 1% easy/hard label flip (§5 extension). One-line:
#   bash scripts/run_exp_11_extra_bilateral_noise.sh
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/lib/experiment_env.sh"
exp_activate_conda
exp_apply_paper_defaults
ROOT="$(exp_repo_root)"
export EXP_INPUT="${EXP_INPUT:-$ROOT/results/20260609_074628_snli_winogrande_roberta-large/dynamics/cartography_with_regions.jsonl}"
exp_require_input

SESSION="${SESSION:-cs162-extra-exp4-bilateral}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed/bilateral_noise_flip}"
EASY_REUSE="${EASY_REUSE:-data/processed/noise_detection_paper}"
GPUS="${GPUS:-0,1,2,3,4}"
RESTARTS="${RESTARTS:-5}"
EPOCHS="${EPOCHS:-${EXP_EPOCHS_WINO}}"
BATCH_SIZE="${BATCH_SIZE:-${EXP_BATCH_SIZE_WINO}}"
FLIP_RATIO="${FLIP_RATIO:-0.01}"
WANDB_NAME="${WANDB_NAME:-extra04_bilateral_noise}"
WANDB_GROUP="${WANDB_GROUP:-extra-exp4-bilateral}"
LOG="$(exp_log_file exp_11_bilateral_noise)"

# Reuse paper §5 easy arm when available; train hard arm (and easy if missing).
CMD="cd '$ROOT' && \
  $PYTHON scripts/13_bilateral_noise_flip.py \
    --train --dataset winogrande --preset '$EXP_PRESET' \
    --epochs '$EPOCHS' --batch-size '$BATCH_SIZE' \
    --learning-rate '$EXP_LEARNING_RATE' \
    --flip-ratio '$FLIP_RATIO' \
    --input '$EXP_INPUT' --output-dir '$OUTPUT_DIR' \
    --reuse-easy-dir '$EASY_REUSE' \
    --gpus '$GPUS' --restarts '$RESTARTS' \
    --wandb-run-name '$WANDB_NAME' --wandb-group '$WANDB_GROUP' && \
  $PYTHON scripts/collect_extension_outputs.py && \
  bash scripts/export_all_metrics.sh '$WANDB_GROUP' && \
  echo '=== Extra #4 bilateral noise complete ==='; bash"

exp_tmux_start "$SESSION" extra-exp4 "$CMD 2>&1 | tee '$LOG'"
exp_print_launch_summary "$SESSION" \
  "GPUs: $GPUS | hard restarts: $RESTARTS (parallel)" \
  "flip_ratio: $FLIP_RATIO | epochs: $EPOCHS" \
  "reuse easy: $EASY_REUSE" \
  "output: $OUTPUT_DIR" \
  "extensions: extension_outputs/Extra_04_*" \
  "log: $LOG" \
  "W&B: $WANDB_NAME (group $WANDB_GROUP)"
