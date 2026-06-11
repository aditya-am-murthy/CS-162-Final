#!/usr/bin/env bash
# Master launcher: queue all paper-reproduction experiments in tmux windows.
# One-line: bash scripts/run_all_remaining_experiments.sh
#
# Override hyperparams globally:
#   EXP_EPOCHS_WINO=6 RESTARTS=3 GPUS=0,1,2,3,4 bash scripts/run_all_remaining_experiments.sh
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/lib/experiment_env.sh"
exp_activate_conda
exp_apply_paper_defaults

SESSION="${SESSION:-cs162-paper-repro}"
ROOT="$(exp_repo_root)"
WANDB_GROUP="${WANDB_GROUP:-paper-reproduction-all}"
SKIP_DATAMAPS="${SKIP_DATAMAPS:-0}"
SKIP_TABLE2="${SKIP_TABLE2:-0}"
SKIP_FIG3="${SKIP_FIG3:-0}"
SKIP_FIG4="${SKIP_FIG4:-0}"
SKIP_FIG5="${SKIP_FIG5:-0}"
SKIP_TABLE3="${SKIP_TABLE3:-0}"
SKIP_TABLE4="${SKIP_TABLE4:-0}"
SKIP_COMPUTE="${SKIP_COMPUTE:-0}"

exp_tmux_ensure_free "$SESSION"

_launch() {
  local window="$1" script="$2"
  tmux new-window -t "$SESSION" -n "$window" \
    "cd '$ROOT' && export EXP_NO_TMUX=1 WANDB_GROUP='$WANDB_GROUP' SESSION='${SESSION}-${window}' && \
     bash scripts/$script 2>&1 | tee logs/master_${window}.log; bash"
}

tmux new-session -d -s "$SESSION" -n readme \
  "cd '$ROOT' && echo 'Paper reproduction queue — attach with: tmux attach -t $SESSION' && \
   echo 'Export metrics anytime: bash scripts/export_all_metrics.sh $WANDB_GROUP' && bash"

[[ "$SKIP_DATAMAPS" == "0" ]] && _launch fig1-snli run_exp_01_datamap_snli.sh
[[ "$SKIP_DATAMAPS" == "0" ]] && _launch maps-mnli-qnli run_exp_02_datamap_mnli_qnli.sh
_launch polish-maps run_exp_03_polish_maps.sh
[[ "$SKIP_TABLE2" == "0" ]] && _launch table2 run_exp_04_table2_winogrande.sh
[[ "$SKIP_FIG3" == "0" ]] && _launch fig3 run_exp_05_fig3_easy_to_learn.sh
[[ "$SKIP_FIG4" == "0" ]] && _launch fig4 run_exp_06_fig4_noise.sh
[[ "$SKIP_FIG5" == "0" ]] && _launch fig5 run_exp_07_fig5_uncertainty.sh
[[ "$SKIP_TABLE3" == "0" ]] && _launch table3 run_exp_08_table3_snli_mnli.sh
[[ "$SKIP_TABLE4" == "0" ]] && _launch table4 run_exp_09_table4_qnli.sh
[[ "$SKIP_COMPUTE" == "0" ]] && _launch compute-gain run_exp_10_subregion_compute_gain.sh

tmux new-window -t "$SESSION" -n export-metrics \
  "cd '$ROOT' && bash scripts/export_all_metrics.sh '$WANDB_GROUP' && \
   bash scripts/plot_from_metrics_csv.py --experiment-tag '$WANDB_GROUP' && bash"

exp_print_launch_summary "$SESSION" \
  "queued experiments from remaining-experiments.md" \
  "global W&B group: $WANDB_GROUP" \
  "export CSV: results/experiment_metrics_{history,summary}.csv" \
  "plots: results/paper_plots_from_metrics/"
