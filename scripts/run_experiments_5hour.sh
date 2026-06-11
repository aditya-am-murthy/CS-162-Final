#!/usr/bin/env bash
# 5-hour deadline queue: reuse existing runs, skip expensive NLI training.
# One-line: bash scripts/run_experiments_5hour.sh
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/lib/experiment_env.sh"
source "$(cd "$(dirname "$0")" && pwd)/lib/fast_5hour.env"
exp_activate_conda
exp_apply_paper_defaults

ROOT="$(exp_repo_root)"
cd "$ROOT"
SESSION="${SESSION:-cs162-deadline-5h}"
LOG="$ROOT/logs/deadline_5hour.log"
mkdir -p "$ROOT/logs" "$ROOT/results/paper_plots_from_metrics"

log() { echo "[$(date '+%H:%M:%S')] $*" >>"$LOG"; echo "[$(date '+%H:%M:%S')] $*"; }

log "=== 5-hour deadline plan ==="
log "Reuse: Table2 (8/8), Fig3 (19/19), Fig4 (noise), Fig2 (WinoGrande map)"
log "Run:   polish maps, export CSV, plot all figures, Fig5 heatmap"
log "Optional: subsampled SNLI map (8k train, 3 ep) if SKIP_EXP_01=0"
log "Skip:  full SNLI/MNLI, Tables 3/4, compute-gain sweep"
log ""
log "Phase 1 (~30 min): export + plot from existing artifacts"
log "Phase 2 (~45 min): optional subsampled Fig1 SNLI map"
log "Phase 3 (~0 min):  training skipped — using pranav-scripts runs"
log "Total budget:      ~1-1.5 h active GPU; rest is plotting/export"
log ""

# --- Phase 1: no training ---
log "Phase 1a: export all existing metrics"
"$PYTHON" scripts/export_experiment_metrics_csv.py \
  --output "$ROOT/results/experiment_metrics_history.csv" \
  --summary-output "$ROOT/results/experiment_metrics_summary.csv" >>"$LOG" 2>&1
bash scripts/export_all_metrics.sh pranav-scripts >>"$LOG" 2>&1 || true

log "Phase 1b: Table 2 metrics from existing region finetune (8 strategies)"
"$PYTHON" scripts/11_region_metrics.py \
  --results "$TABLE2_OUTPUT/train_results.json" \
  --manifest "$TABLE2_OUTPUT/manifest.json" \
  --output results/region_metrics_table2.json \
  --no-wandb >>"$LOG" 2>&1

log "Phase 1c: polish WinoGrande maps + insight figures"
"$PYTHON" scripts/rebuild_fixed_maps.py \
  results/20260609_074628_snli_winogrande_roberta-large >>"$LOG" 2>&1
MAP_INPUT="results/20260609_074628_snli_winogrande_roberta-large/fixed-maps/adaptive/20260609_074628_cartography_with_regions.jsonl"
"$PYTHON" scripts/07_generate_insight_figures.py \
  --input "$MAP_INPUT" \
  --run-id 20260609_074628_snli_winogrande_roberta-large \
  --wandb-group deadline-5h --wandb-run-name polish_maps_deadline \
  --no-wandb >>"$LOG" 2>&1 || true

log "Phase 1d: plot Fig 3/4/5 + Table 2 from measured data"
FIG5_INPUT="${FIG5_INPUT:-$MAP_INPUT}"
"$PYTHON" scripts/plot_from_metrics_csv.py \
  --summary-csv results/experiment_metrics_summary.csv \
  --output-dir results/paper_plots_from_metrics \
  --fig4-shift-jsonl "$FIG4_OUTPUT/before_after_shift.jsonl" \
  --fig5-input "$FIG5_INPUT" \
  --no-wandb >>"$LOG" 2>&1

log "Phase 1e: uncertainty correlations (Fig 5 stats)"
"$PYTHON" scripts/05_uncertainty_checks.py \
  --input "$FIG5_INPUT" \
  --wandb-group deadline-5h --wandb-run-name fig5_uncertainty \
  --no-wandb >>"$LOG" 2>&1

log "Phase 1 complete."

# --- Phase 2: optional quick SNLI map ---
if [[ "${SKIP_EXP_01}" == "0" ]]; then
  log "Phase 2: subsampled SNLI map (${EXP_MAX_TRAIN_SAMPLES} train, ${EXP_EPOCHS_SNLI} ep)"
  export SESSION=cs162-deadline-snli
  export GPU=1
  export EPOCHS="${EXP_EPOCHS_SNLI}"
  export MAX_TRAIN="${EXP_MAX_TRAIN_SAMPLES}"
  export MAX_EVAL="${EXP_MAX_EVAL_SAMPLES}"
  export WANDB_GROUP=deadline-5h
  export WANDB_NAME=fig1_snli_subsampled
  bash scripts/run_exp_01_datamap_snli.sh >>"$LOG" 2>&1
  log "Waiting for subsampled SNLI map (~45 min)..."
  wait_for_log() { local f="$1" m="$2"; while [[ ! -f "$f" ]] || ! grep -qF "$m" "$f"; do sleep 30; done; }
  wait_for_log "$ROOT/logs/exp_01_datamap_snli.log" "Fig 1 SNLI map complete"
  SNLI_RUN="$(ls -dt experiments/runs/*_snli_* 2>/dev/null | head -1)"
  if [[ -n "$SNLI_RUN" ]]; then
    "$PYTHON" scripts/rebuild_fixed_maps.py "$SNLI_RUN" >>"$LOG" 2>&1
    SNLI_MAP="$SNLI_RUN/fixed-maps/adaptive/"*_cartography_with_regions.jsonl
    "$PYTHON" scripts/plot_from_metrics_csv.py \
      --fig5-input "$SNLI_MAP" \
      --output-dir results/paper_plots_from_metrics \
      --no-wandb >>"$LOG" 2>&1 || true
  fi
else
  log "Phase 2 skipped (SKIP_EXP_01=1). Use existing subsampled map: $SNLI_MAP_RUN"
  if [[ -d "$SNLI_MAP_RUN" ]]; then
    "$PYTHON" scripts/rebuild_fixed_maps.py "$SNLI_MAP_RUN" >>"$LOG" 2>&1 || true
  fi
fi

# --- Phase 3: training skipped (existing pranav runs) ---
if [[ "${SKIP_TRAIN_TABLE2}" == "1" ]]; then
  log "Phase 3: Table 2 training skipped — using $TABLE2_OUTPUT (8 strategies, roberta-base)"
fi
if [[ "${SKIP_TRAIN_FIG3}" == "1" ]]; then
  log "Phase 3: Fig 3 training skipped — using $FIG3_OUTPUT (19 subsets, roberta-base)"
fi
if [[ "${SKIP_TRAIN_FIG4}" == "1" ]]; then
  log "Phase 3: Fig 4 training skipped — using $FIG4_OUTPUT"
fi

log "Phase 4: final export"
"$PYTHON" scripts/export_experiment_metrics_csv.py \
  --output "$ROOT/results/experiment_metrics_history.csv" \
  --summary-output "$ROOT/results/experiment_metrics_summary.csv" >>"$LOG" 2>&1
bash scripts/export_all_metrics.sh pranav-scripts >>"$LOG" 2>&1 || true
"$PYTHON" scripts/plot_from_metrics_csv.py \
  --summary-csv results/experiment_metrics_summary.csv \
  --output-dir results/paper_plots_from_metrics \
  --fig4-shift-jsonl "$FIG4_OUTPUT/before_after_shift.jsonl" \
  --fig5-input "$FIG5_INPUT" \
  --no-wandb >>"$LOG" 2>&1

log "=== 5-hour deadline queue complete ==="
log "Outputs:"
log "  results/experiment_metrics_summary.csv"
log "  results/region_metrics_table2.json"
log "  results/paper_plots_from_metrics/"
log "  results/20260609_074628_snli_winogrande_roberta-large/fixed-maps/"
