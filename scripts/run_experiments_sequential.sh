#!/usr/bin/env bash
# Run paper reproduction experiments one at a time in tmux.
# Usage: bash scripts/run_experiments_sequential.sh
# For deadline runs use: bash scripts/run_experiments_5hour.sh
set -euo pipefail
if [[ "${DEADLINE_MODE:-}" == "1" ]]; then
  echo "Use scripts/run_experiments_5hour.sh for the 5-hour deadline plan." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/lib/experiment_env.sh"
exp_activate_conda
exp_apply_paper_defaults

ROOT="$(exp_repo_root)"
cd "$ROOT"
MASTER_SESSION="${MASTER_SESSION:-cs162-paper-sequential}"
PROGRESS_LOG="$ROOT/logs/sequential_experiments.log"

declare -A EXP_ETA=(
  [01]="10-24 h (full SNLI RoBERTa-large, 6 ep, 1 GPU)"
  [02]="6-14 h (MNLI+QNLI maps in parallel on 2 GPUs)"
  [03]="10-20 min (rebuild maps + insight figures, no training)"
  [04]="1.5-2.5 h (24 jobs: 8 strategies x 3 seeds, 5 GPUs)"
  [05]="2.5-4 h (57 jobs: 19 subsets x 3 seeds, 5 GPUs)"
  [06]="15-30 min (single noised retrain + detector)"
  [07]="5-15 min (uncertainty correlations + Fig 5 plot)"
  [08]="12-36 h (SNLI+MNLI subset retraining; SNLI dominates)"
  [09]="4-8 h (12 QNLI jobs: 4 strategies x 3 seeds, 5 GPUs)"
  [10]="4-6 h (24 jobs: 4 strategies x 6 epoch sweeps, sequential)"
)

declare -A EXP_LOG=(
  [01]="exp_01_datamap_snli"
  [02]="exp_02_mnli_map"   # completes when mnli window done; also exp_02_qnli_map
  [03]="exp_03_polish_maps"
  [04]="exp_04_table2"
  [05]="exp_05_fig3"
  [06]="exp_06_fig4"
  [07]="exp_07_fig5"
  [08]="exp_08_table3_snli"  # longer of snli/mnli windows
  [09]="exp_09_table4"
  [10]="exp_10_compute_gain"
)

declare -A EXP_DONE=(
  [01]="Fig 1 SNLI map complete"
  [02]="MNLI+QNLI maps complete"
  [03]="map polish complete"
  [04]="Table 2 complete"
  [05]="Fig 3 complete"
  [06]="Fig 4 complete"
  [07]="Fig 5 complete"
  [08]="Table 3 complete"
  [09]="Table 4 complete"
  [10]="compute-gain sweep complete"
)

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS_LOG"
}

wait_for_log() {
  local log_file="$1"
  local marker="$2"
  local poll="${3:-30}"
  while true; do
    if [[ -f "$log_file" ]] && grep -qF "$marker" "$log_file"; then
      return 0
    fi
    sleep "$poll"
  done
}

run_one() {
  local id="$1"
  local script="run_exp_${id}_"*
  script=$(ls "$ROOT/scripts"/run_exp_"${id}"_*.sh 2>/dev/null | head -1)
  local log_name="${EXP_LOG[$id]}"
  local log_file="$ROOT/logs/${log_name}.log"
  local marker="${EXP_DONE[$id]}"
  local eta="${EXP_ETA[$id]}"

  log "START exp $id: $(basename "$script") | ETA: $eta"
  : > "$log_file"
  bash "$script" >> "$PROGRESS_LOG" 2>&1 || true

  if [[ "$id" == "02" ]]; then
    wait_for_log "$ROOT/logs/exp_02_qnli_map.log" "$marker"
  elif [[ "$id" == "08" ]]; then
    wait_for_log "$ROOT/logs/exp_08_table3_mnli.log" "$marker"
  else
    wait_for_log "$log_file" "$marker"
  fi
  log "DONE  exp $id ($(basename "$script"))"
}

log "=== Sequential paper reproduction queue ==="
log "Hardware: 5x RTX 3090 | preset=$EXP_PRESET | restarts=$EXP_RESTARTS"
total_low=0
total_high=0
for id in 01 02 03 04 05 06 07 08 09 10; do
  log "  exp $id ETA: ${EXP_ETA[$id]}"
done
log "Estimated total wall time: ~45-90 h (SNLI/MNLI full-data maps dominate)"
log ""

for id in 01 02 03 04 05 06 07 08 09 10; do
  run_one "$id"
done

log "=== All experiments finished ==="
bash "$ROOT/scripts/export_all_metrics.sh" paper-reproduction-all
bash "$ROOT/scripts/plot_from_metrics_csv.py" --experiment-tag paper-reproduction
