#!/usr/bin/env bash
# Subsampled training for exp 01, 02, 08, 09 — multi-GPU parallel layout.
# One-line: bash scripts/run_exp_subset_final_four.sh
#
# GPU plan (5x RTX 3090):
#   Phase 1 (~45m): SNLI cuda:0 | MNLI cuda:1 | QNLI cuda:2  (all parallel)
#   Phase 2 (~35m): Table3-SNLI gpus 0,1 | Table3-MNLI gpus 2,3 | Table4-QNLI cuda:4
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/lib/experiment_env.sh"
source "$(dirname "$0")/lib/subset_training.env"
exp_activate_conda
exp_apply_paper_defaults

ROOT="$(exp_repo_root)"
cd "$ROOT"
LOG="$ROOT/logs/subset_final_four.log"
MASTER_SESSION="${MASTER_SESSION:-cs162-subset-four}"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

_wait_log() {
  local file="$1" marker="$2"
  while [[ ! -f "$file" ]] || ! grep -qF "$marker" "$file"; do sleep 20; done
}

_resolve_map_input() {
  local dataset="$1"
  local published run
  for published in $(ls -dt "$ROOT/results"/*_"${dataset}"_* 2>/dev/null); do
    case "$dataset" in
      snli)
        [[ "$(basename "$published")" == *"_snli_mnli_"* ]] && continue
        [[ "$(basename "$published")" == *"_snli_qnli_"* ]] && continue
        ;;
      mnli) [[ "$(basename "$published")" != *"_snli_mnli_"* ]] && continue ;;
      qnli) [[ "$(basename "$published")" != *"_snli_qnli_"* ]] && [[ "$(basename "$published")" != *"_qnli_"* ]] && continue ;;
    esac
    if [[ -f "$published/dynamics/cartography_with_regions.jsonl" ]]; then
      echo "$published/dynamics/cartography_with_regions.jsonl"
      return
    fi
  done
  for run in $(ls -dt "$ROOT/experiments/runs"/*_"${dataset}"_* 2>/dev/null); do
    case "$dataset" in
      snli)
        [[ "$(basename "$run")" == *"_snli_mnli_"* ]] && continue
        [[ "$(basename "$run")" == *"_snli_qnli_"* ]] && continue
        ;;
      mnli) [[ "$(basename "$run")" != *"_snli_mnli_"* ]] && continue ;;
      qnli) [[ "$(basename "$run")" != *"_snli_qnli_"* ]] && [[ "$(basename "$run")" != *"_qnli_"* ]] && continue ;;
    esac
    if [[ -f "$run/dynamics/cartography_with_regions.jsonl" ]]; then
      echo "$run/dynamics/cartography_with_regions.jsonl"
      return
    fi
  done
  echo ""
}

_run_map() {
  local dataset="$1" gpu="$2" epochs="$3" log_name="$4" wandb_name="$5"
  local marker="=== ${dataset} map complete ==="
  cat <<EOF
cd '$ROOT' && export CUDA_VISIBLE_DEVICES='$gpu' PYTHONUNBUFFERED=1 && \
  $PYTHON scripts/run_cartography_experiment.py \
    --task snli --dataset '$dataset' --preset '$EXP_PRESET' \
    --epochs '$epochs' --batch-size '$EXP_BATCH_SIZE_NLI' \
    --max-train-samples '$EXP_MAX_TRAIN_SAMPLES' --max-eval-samples '$EXP_MAX_EVAL_SAMPLES' \
    --wandb-run-name '$wandb_name' --wandb-group '$EXP_WANDB_GROUP' && \
  $PYTHON scripts/rebuild_fixed_maps.py "\$(ls -dt experiments/runs/*_${dataset}_* | head -1)" && \
  echo '$marker'
EOF
}

_run_table3_dataset() {
  local dataset="$1" input="$2" gpus="$3" log_name="$4" wandb="$5"
  cat <<EOF
cd '$ROOT' && $PYTHON scripts/09_region_finetune.py --train --dataset '$dataset' \
  --preset '$EXP_PRESET' --epochs '$(exp_dataset_epochs "$dataset")' \
  --batch-size '$(exp_dataset_batch_size "$dataset")' \
  --learning-rate '$EXP_LEARNING_RATE' \
  --max-train-samples '$EXP_MAX_TRAIN_SAMPLES' --max-eval-samples '$EXP_MAX_EVAL_SAMPLES' \
  --strategies high_variability low_confidence random full \
  --restarts '$EXP_RESTARTS' --gpus '$gpus' \
  --input '$input' --output-dir '$ROOT/data/processed/table3_snli_mnli/${dataset}' \
  --wandb-run-name '${wandb}' --wandb-group '$EXP_WANDB_GROUP' && \
  $PYTHON scripts/11_region_metrics.py \
    --results '$ROOT/data/processed/table3_snli_mnli/${dataset}/train_results.json' \
    --manifest '$ROOT/data/processed/table3_snli_mnli/${dataset}/manifest.json' \
    --output '$ROOT/results/region_metrics_table3_${dataset}.json' --no-wandb && \
  echo '=== Table 3 ${dataset} complete ==='
EOF
}

log "=== Subset final four (multi-GPU) ==="
log "Phase 1 GPUs: SNLI=${GPU_SNLI_MAP} MNLI=${GPU_MNLI_MAP} QNLI=${GPU_QNLI_MAP} (parallel)"
log "Phase 2 GPUs: T3-SNLI=${GPUS_TABLE3_SNLI} T3-MNLI=${GPUS_TABLE3_MNLI} T4-QNLI=${GPUS_TABLE4_QNLI} (parallel)"
log "train=${EXP_MAX_TRAIN_SAMPLES} eval=${EXP_MAX_EVAL_SAMPLES} epochs=3 | wall ~1.5-2h"
log ""

# --- Phase 1: all three map jobs in parallel ---
if [[ "${SKIP_EXP_01}" == "0" || "${SKIP_EXP_02}" == "0" ]]; then
  log "Phase 1: launching map collection on 3 GPUs in parallel"
  tmux has-session -t "$MASTER_SESSION" 2>/dev/null && tmux kill-session -t "$MASTER_SESSION"
  tmux new-session -d -s "$MASTER_SESSION" -n maps

  if [[ "${SKIP_EXP_01}" == "0" ]]; then
    tmux new-window -t "$MASTER_SESSION" -n snli-map \
      "$(_run_map snli "$GPU_SNLI_MAP" "$EXP_EPOCHS_SNLI" exp_01_datamap_snli fig1_snli_subset) 2>&1 | tee '$ROOT/logs/exp_01_datamap_snli.log'; bash"
  fi
  if [[ "${SKIP_EXP_02}" == "0" ]]; then
    tmux new-window -t "$MASTER_SESSION" -n mnli-map \
      "$(_run_map mnli "$GPU_MNLI_MAP" "$EXP_EPOCHS_MNLI" exp_02_mnli_map fig_mnli_subset) 2>&1 | tee '$ROOT/logs/exp_02_mnli_map.log'; bash"
    tmux new-window -t "$MASTER_SESSION" -n qnli-map \
      "$(_run_map qnli "$GPU_QNLI_MAP" "$EXP_EPOCHS_QNLI" exp_02_qnli_map fig_qnli_subset) 2>&1 | tee '$ROOT/logs/exp_02_qnli_map.log'; bash"
  fi

  [[ "${SKIP_EXP_01}" == "0" ]] && _wait_log "$ROOT/logs/exp_01_datamap_snli.log" "snli map complete"
  [[ "${SKIP_EXP_02}" == "0" ]] && _wait_log "$ROOT/logs/exp_02_qnli_map.log" "qnli map complete"
  log "Phase 1 complete"
fi

SNLI_INPUT="$(_resolve_map_input snli)"
MNLI_INPUT="$(_resolve_map_input mnli)"
QNLI_INPUT="$(_resolve_map_input qnli)"
[[ -z "$SNLI_INPUT" ]] && SNLI_INPUT="results/20260611_224924_snli_roberta-base/dynamics/cartography_with_regions.jsonl"
[[ -z "$MNLI_INPUT" ]] && MNLI_INPUT="results/20260527_051157_snli_mnli_roberta-base/dynamics/cartography_with_regions.jsonl"
[[ -z "$QNLI_INPUT" ]] && QNLI_INPUT="results/20260527_051157_snli_qnli_roberta-base/dynamics/cartography_with_regions.jsonl"

# --- Phase 2: Table 3 (snli+mnli) + Table 4 (qnli) all parallel ---
if [[ "${SKIP_EXP_08}" == "0" || "${SKIP_EXP_09}" == "0" ]]; then
  log "Phase 2: Table 3 + Table 4 on GPUs ${GPUS_TABLE3_SNLI}|${GPUS_TABLE3_MNLI}|${GPUS_TABLE4_QNLI}"
  if ! tmux has-session -t "$MASTER_SESSION" 2>/dev/null; then
    tmux new-session -d -s "$MASTER_SESSION" -n readme "echo 'Phase 2 finetune'; bash"
  fi

  if [[ "${SKIP_EXP_08}" == "0" ]]; then
    tmux new-window -t "$MASTER_SESSION" -n t3-snli \
      "$(_run_table3_dataset snli "$SNLI_INPUT" "$GPUS_TABLE3_SNLI" exp_08_table3_snli table3_snli_subset) 2>&1 | tee '$ROOT/logs/exp_08_table3_snli.log'; bash"
    tmux new-window -t "$MASTER_SESSION" -n t3-mnli \
      "$(_run_table3_dataset mnli "$MNLI_INPUT" "$GPUS_TABLE3_MNLI" exp_08_table3_mnli table3_mnli_subset) 2>&1 | tee '$ROOT/logs/exp_08_table3_mnli.log'; bash"
  fi
  if [[ "${SKIP_EXP_09}" == "0" ]]; then
    tmux new-window -t "$MASTER_SESSION" -n t4-qnli \
      "cd '$ROOT' && $PYTHON scripts/09_region_finetune.py --train --dataset qnli \
        --preset '$EXP_PRESET' --epochs '${EXP_EPOCHS_QNLI}' --batch-size '$EXP_BATCH_SIZE_NLI' \
        --learning-rate '$EXP_LEARNING_RATE' \
        --max-train-samples '$EXP_MAX_TRAIN_SAMPLES' --max-eval-samples '$EXP_MAX_EVAL_SAMPLES' \
        --strategies high_variability low_confidence random full \
        --restarts '$EXP_RESTARTS' --gpus '$GPUS_TABLE4_QNLI' \
        --input '$QNLI_INPUT' --output-dir '$ROOT/data/processed/table4_qnli' \
        --wandb-run-name table4_qnli_subset --wandb-group '$EXP_WANDB_GROUP' && \
      $PYTHON scripts/11_region_metrics.py \
        --results '$ROOT/data/processed/table4_qnli/train_results.json' \
        --manifest '$ROOT/data/processed/table4_qnli/manifest.json' \
        --output results/region_metrics_table4.json --no-wandb && \
      echo '=== Table 4 complete ===' 2>&1 | tee '$ROOT/logs/exp_09_table4.log'; bash"
  fi

  [[ "${SKIP_EXP_08}" == "0" ]] && _wait_log "$ROOT/logs/exp_08_table3_snli.log" "Table 3 snli complete"
  [[ "${SKIP_EXP_08}" == "0" ]] && _wait_log "$ROOT/logs/exp_08_table3_mnli.log" "Table 3 mnli complete"
  [[ "${SKIP_EXP_09}" == "0" ]] && _wait_log "$ROOT/logs/exp_09_table4.log" "Table 4 complete"
  log "Phase 2 complete"
fi

log "Collecting paper outputs..."
"$PYTHON" scripts/export_experiment_metrics_csv.py >>"$LOG" 2>&1
"$PYTHON" scripts/collect_paper_outputs.py >>"$LOG" 2>&1
log "=== subset final four complete ==="
log "attach: tmux attach -t $MASTER_SESSION"
