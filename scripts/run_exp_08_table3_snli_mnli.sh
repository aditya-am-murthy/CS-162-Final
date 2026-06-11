#!/usr/bin/env bash
# Table 3: SNLI + MultiNLI 33% ambiguous/hard vs random/100% (paper §3). One-line:
#   bash scripts/run_exp_08_table3_snli_mnli.sh
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/lib/experiment_env.sh"
exp_activate_conda
exp_apply_paper_defaults
if [[ "${SUBSET_MODE:-0}" == "1" ]]; then
  source "$(dirname "$0")/lib/subset_training.env"
fi

SESSION="${SESSION:-cs162-exp-table3}"
OUTPUT_BASE="${OUTPUT_BASE:-data/processed/table3_snli_mnli}"
SNLI_INPUT="${SNLI_INPUT:-$EXP_INPUT}"
MNLI_INPUT="${MNLI_INPUT:-results/20260527_051157_snli_mnli_roberta-base/dynamics/cartography_with_regions.jsonl}"
RESTARTS="${RESTARTS:-${EXP_RESTARTS}}"
GPUS="${GPUS:-${EXP_GPUS}}"
MAX_TRAIN="${MAX_TRAIN:-${EXP_MAX_TRAIN_SAMPLES}}"
WANDB_GROUP="${WANDB_GROUP:-table3-snli-mnli}"
LOG="$(exp_log_file exp_08_table3)"
ROOT="$(exp_repo_root)"

_run_dataset() {
  local dataset="$1" input="$2" epochs="$3" out="$4" wandb="$5"
  cat <<EOF
$PYTHON scripts/09_region_finetune.py --train --dataset '$dataset' --preset '$EXP_PRESET' \
  --epochs '$epochs' --batch-size '$(exp_dataset_batch_size "$dataset")' \
  --learning-rate '$EXP_LEARNING_RATE' \
  --max-train-samples '$MAX_TRAIN' --max-eval-samples '$EXP_MAX_EVAL_SAMPLES' \
  --strategies high_variability low_confidence random full \
  --restarts '$RESTARTS' --gpus '$GPUS' \
  --input '$input' --output-dir '$out' \
  --wandb-run-name '${wandb}' --wandb-group '$WANDB_GROUP'
EOF
}

exp_tmux_multi_window "$SESSION" \
  snli "cd '$ROOT' && $(_run_dataset snli "$SNLI_INPUT" "${EXP_EPOCHS_SNLI}" "$OUTPUT_BASE/snli" table3_snli) \
  2>&1 | tee '$(exp_log_file exp_08_table3_snli)'; bash" \
  mnli "cd '$ROOT' && $(_run_dataset mnli "$MNLI_INPUT" "${EXP_EPOCHS_MNLI}" "$OUTPUT_BASE/mnli" table3_mnli) \
  2>&1 | tee '$(exp_log_file exp_08_table3_mnli)'; \
  bash scripts/export_all_metrics.sh '$WANDB_GROUP'; \
  bash scripts/plot_from_metrics_csv.py --experiment-tag table3; \
  echo '=== Table 3 complete ==='; bash"

exp_print_launch_summary "$SESSION" \
  "GPUs: $GPUS | restarts: $RESTARTS" \
  "snli input: $SNLI_INPUT" \
  "mnli input: $MNLI_INPUT" \
  "output: $OUTPUT_BASE" \
  "log: $LOG" \
  "W&B group: $WANDB_GROUP"
