#!/usr/bin/env bash
# Fig 1: SNLI RoBERTa-large data map (paper §2). One-line:
#   bash scripts/run_exp_01_datamap_snli.sh
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/lib/experiment_env.sh"
exp_activate_conda
exp_apply_paper_defaults
if [[ "${SUBSET_MODE:-0}" == "1" ]]; then
  source "$(dirname "$0")/lib/subset_training.env"
elif [[ -f "$(dirname "$0")/lib/fast_5hour.env" ]]; then
  source "$(dirname "$0")/lib/fast_5hour.env"
fi

SESSION="${SESSION:-cs162-exp-fig1-snli}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-${EXP_EPOCHS_SNLI}}"
MAX_TRAIN="${MAX_TRAIN:-${EXP_MAX_TRAIN_SAMPLES}}"
MAX_EVAL="${MAX_EVAL:-${EXP_MAX_EVAL_SAMPLES}}"
BATCH_SIZE="${BATCH_SIZE:-${EXP_BATCH_SIZE_NLI}}"
WANDB_NAME="${WANDB_NAME:-fig1_snli_roberta_large}"
WANDB_GROUP="${WANDB_GROUP:-${EXP_WANDB_GROUP}}"
LOG="$(exp_log_file exp_01_datamap_snli)"
ROOT="$(exp_repo_root)"

CMD="cd '$ROOT' && export CUDA_VISIBLE_DEVICES='$GPU' PYTHONUNBUFFERED=1 && \
  $PYTHON scripts/run_cartography_experiment.py \
    --task snli --dataset snli --preset '$EXP_PRESET' \
    --epochs '$EPOCHS' --batch-size '$BATCH_SIZE' \
    --max-train-samples '$MAX_TRAIN' --max-eval-samples '$MAX_EVAL' \
    --wandb-run-name '$WANDB_NAME' --wandb-group '$WANDB_GROUP' && \
  $PYTHON scripts/rebuild_fixed_maps.py \"\$(ls -dt experiments/runs/*_snli_* | head -1)\" && \
  bash scripts/export_all_metrics.sh '$WANDB_GROUP' && \
  echo '=== Fig 1 SNLI map complete ==='; bash"

exp_tmux_start "$SESSION" fig1-snli "$CMD 2>&1 | tee '$LOG'"
exp_print_launch_summary "$SESSION" \
  "GPU: cuda:$GPU" \
  "epochs: $EPOCHS | train=$MAX_TRAIN | batch=$BATCH_SIZE" \
  "log: $LOG" \
  "W&B: $WANDB_NAME (group $WANDB_GROUP)"
