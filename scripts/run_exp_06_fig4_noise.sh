#!/usr/bin/env bash
# Fig 4: 1% easy-label noise injection + retrain (paper §5). One-line:
#   bash scripts/run_exp_06_fig4_noise.sh
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/lib/experiment_env.sh"
exp_activate_conda
exp_apply_paper_defaults
[[ -f "$(dirname "$0")/lib/fast_5hour.env" ]] && source "$(dirname "$0")/lib/fast_5hour.env"
exp_require_input

if [[ "${SKIP_TRAIN_FIG4:-0}" == "1" ]]; then
  echo "SKIP_TRAIN_FIG4=1 — using existing results in ${FIG4_OUTPUT:-data/processed/noise_detection_paper}"
  ROOT="$(exp_repo_root)"
  OUT="${FIG4_OUTPUT:-data/processed/noise_detection_paper}"
  bash "$ROOT/scripts/export_all_metrics.sh" pranav-scripts
  "$PYTHON" "$ROOT/scripts/plot_from_metrics_csv.py" \
    --fig4-shift-jsonl "$OUT/before_after_shift.jsonl" --experiment-tag fig4 --no-wandb
  echo "=== Fig 4 complete (existing runs) ==="
  exit 0
fi

SESSION="${SESSION:-cs162-exp-fig4}"
GPU="${GPU:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed/noise_detection_paper}"
EPOCHS="${EPOCHS:-${EXP_EPOCHS_WINO}}"
BATCH_SIZE="${BATCH_SIZE:-${EXP_BATCH_SIZE_WINO}}"
NOISE_RATIO="${NOISE_RATIO:-0.01}"
WANDB_NAME="${WANDB_NAME:-fig4_noise_winogrande}"
WANDB_GROUP="${WANDB_GROUP:-fig4-noise}"
LOG="$(exp_log_file exp_06_fig4)"
ROOT="$(exp_repo_root)"

CMD="cd '$ROOT' && export CUDA_VISIBLE_DEVICES='$GPU' PYTHONUNBUFFERED=1 && \
  $PYTHON scripts/12_noise_detection_paper.py \
    --train --dataset winogrande --preset '$EXP_PRESET' \
    --epochs '$EPOCHS' --batch-size '$BATCH_SIZE' \
    --learning-rate '$EXP_LEARNING_RATE' \
    --noise-ratio '$NOISE_RATIO' \
    --input '$EXP_INPUT' --output-dir '$OUTPUT_DIR' \
    --wandb-run-name '$WANDB_NAME' --wandb-group '$WANDB_GROUP' && \
  bash scripts/export_all_metrics.sh '$WANDB_GROUP' && \
  bash scripts/plot_from_metrics_csv.py \
    --fig4-shift-jsonl '$OUTPUT_DIR/before_after_shift.jsonl' \
    --experiment-tag fig4 && \
  echo '=== Fig 4 complete ==='; bash"

exp_tmux_start "$SESSION" fig4 "$CMD 2>&1 | tee '$LOG'"
exp_print_launch_summary "$SESSION" \
  "GPU: cuda:$GPU" \
  "noise_ratio: $NOISE_RATIO | epochs: $EPOCHS" \
  "output: $OUTPUT_DIR" \
  "log: $LOG" \
  "W&B: $WANDB_NAME (group $WANDB_GROUP)"
