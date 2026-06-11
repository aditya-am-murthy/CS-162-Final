#!/usr/bin/env bash
# Appendix data maps: MNLI + QNLI RoBERTa-large (paper §2). One-line:
#   bash scripts/run_exp_02_datamap_mnli_qnli.sh
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/lib/experiment_env.sh"
exp_activate_conda
exp_apply_paper_defaults
if [[ "${SUBSET_MODE:-0}" == "1" ]]; then
  source "$(dirname "$0")/lib/subset_training.env"
fi

SESSION="${SESSION:-cs162-exp-maps-mnli-qnli}"
GPU_MNLI="${GPU_MNLI:-0}"
GPU_QNLI="${GPU_QNLI:-1}"
MAX_TRAIN="${MAX_TRAIN:-${EXP_MAX_TRAIN_SAMPLES}}"
MAX_EVAL="${MAX_EVAL:-${EXP_MAX_EVAL_SAMPLES}}"
BATCH_SIZE="${BATCH_SIZE:-${EXP_BATCH_SIZE_NLI}}"
WANDB_GROUP="${WANDB_GROUP:-${EXP_WANDB_GROUP}}"
ROOT="$(exp_repo_root)"

_run_one() {
  local dataset="$1" gpu="$2" epochs="$3" log_name="$4" wandb_name="$5"
  local log
  log="$(exp_log_file "$log_name")"
  cat <<EOF
cd '$ROOT' && export CUDA_VISIBLE_DEVICES='$gpu' PYTHONUNBUFFERED=1 && \
  $PYTHON scripts/run_cartography_experiment.py \
    --task snli --dataset '$dataset' --preset '$EXP_PRESET' \
    --epochs '$epochs' --batch-size '$BATCH_SIZE' \
    --max-train-samples '$MAX_TRAIN' --max-eval-samples '$MAX_EVAL' \
    --wandb-run-name '$wandb_name' --wandb-group '$WANDB_GROUP' && \
  $PYTHON scripts/rebuild_fixed_maps.py "\$(ls -dt experiments/runs/*_${dataset}_* | head -1)" \
  2>&1 | tee '$log'
EOF
}

exp_tmux_multi_window "$SESSION" \
  mnli "$(_run_one mnli "$GPU_MNLI" "${EXP_EPOCHS_MNLI}" exp_02_mnli_map fig_mnli_roberta_large); bash" \
  qnli "$(_run_one qnli "$GPU_QNLI" "${EXP_EPOCHS_QNLI}" exp_02_qnli_map fig_qnli_roberta_large); \
  bash scripts/export_all_metrics.sh '$WANDB_GROUP'; echo '=== MNLI+QNLI maps complete ==='; bash"

exp_print_launch_summary "$SESSION" \
  "mnli: cuda:$GPU_MNLI (${EXP_EPOCHS_MNLI} ep)" \
  "qnli: cuda:$GPU_QNLI (${EXP_EPOCHS_QNLI} ep)" \
  "train=$MAX_TRAIN | batch=$BATCH_SIZE" \
  "W&B group: $WANDB_GROUP"
