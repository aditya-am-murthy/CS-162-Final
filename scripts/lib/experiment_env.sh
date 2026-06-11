#!/usr/bin/env bash
# Shared helpers for paper-reproduction experiment launchers.
# Source from run_exp_*.sh scripts — do not execute directly.

if [[ -n "${EXP_ENV_LOADED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
EXP_ENV_LOADED=1

exp_repo_root() {
  if [[ -n "${EXP_ROOT:-}" ]]; then
    printf '%s\n' "$EXP_ROOT"
    return
  fi
  local src="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
  EXP_ROOT="$(cd "$(dirname "$src")/../.." && pwd)"
  printf '%s\n' "$EXP_ROOT"
}

exp_activate_conda() {
  local root
  root="$(exp_repo_root)"
  cd "$root"
  export PYTHONUNBUFFERED=1
  export CONDA_ENV="${CONDA_ENV:-cs162-cartography}"
  if [[ -z "${PYTHON:-}" ]]; then
    if command -v conda &>/dev/null; then
      # shellcheck disable=SC1091
      source "$(conda info --base)/etc/profile.d/conda.sh"
      conda activate "$CONDA_ENV"
      PYTHON="$(command -v python)"
    else
      PYTHON="python3"
    fi
  fi
  export PYTHON
  mkdir -p "$root/logs" "$root/results"
}

exp_default_input() {
  local root dataset
  root="$(exp_repo_root)"
  dataset="${1:-winogrande}"
  if [[ "$dataset" == "winogrande" ]]; then
    printf '%s/results/20260609_074628_snli_winogrande_roberta-large/fixed-maps/adaptive/20260609_074628_cartography_with_regions.jsonl\n' "$root"
  else
    printf '%s/results/20260609_074628_snli_winogrande_roberta-large/dynamics/cartography_with_regions.jsonl\n' "$root"
  fi
}

exp_apply_paper_defaults() {
  export EXP_PRESET="${EXP_PRESET:-roberta-large}"
  export EXP_LEARNING_RATE="${EXP_LEARNING_RATE:-1e-5}"
  export EXP_MAX_TRAIN_SAMPLES="${EXP_MAX_TRAIN_SAMPLES:-0}"
  export EXP_MAX_EVAL_SAMPLES="${EXP_MAX_EVAL_SAMPLES:-0}"
  export EXP_MAX_LENGTH="${EXP_MAX_LENGTH:-256}"
  export EXP_RESTARTS="${EXP_RESTARTS:-3}"
  export EXP_GPUS="${EXP_GPUS:-0,1,2,3,4}"
  export EXP_WANDB_GROUP="${EXP_WANDB_GROUP:-paper-reproduction}"
  export EXP_WINOGRANDE_CONFIG="${EXP_WINOGRANDE_CONFIG:-winogrande_xl}"
  export EXP_BATCH_SIZE_WINO="${EXP_BATCH_SIZE_WINO:-64}"
  export EXP_BATCH_SIZE_NLI="${EXP_BATCH_SIZE_NLI:-96}"
  export EXP_EPOCHS_WINO="${EXP_EPOCHS_WINO:-6}"
  export EXP_EPOCHS_SNLI="${EXP_EPOCHS_SNLI:-6}"
  export EXP_EPOCHS_MNLI="${EXP_EPOCHS_MNLI:-5}"
  export EXP_EPOCHS_QNLI="${EXP_EPOCHS_QNLI:-5}"
  export EXP_INPUT="${EXP_INPUT:-$(exp_default_input winogrande)}"
}

exp_dataset_epochs() {
  case "$1" in
    winogrande) printf '%s\n' "${EXP_EPOCHS_WINO}" ;;
    snli) printf '%s\n' "${EXP_EPOCHS_SNLI}" ;;
    mnli) printf '%s\n' "${EXP_EPOCHS_MNLI}" ;;
    qnli) printf '%s\n' "${EXP_EPOCHS_QNLI}" ;;
    *) printf '%s\n' "${EXP_EPOCHS_WINO}" ;;
  esac
}

exp_dataset_batch_size() {
  case "$1" in
    winogrande) printf '%s\n' "${EXP_BATCH_SIZE_WINO}" ;;
    *) printf '%s\n' "${EXP_BATCH_SIZE_NLI}" ;;
  esac
}

exp_require_input() {
  if [[ ! -f "$EXP_INPUT" ]]; then
    echo "missing dynamics input: $EXP_INPUT" >&2
    echo "set EXP_INPUT= to cartography_with_regions.jsonl" >&2
    exit 1
  fi
}

exp_log_file() {
  local name="$1"
  local root
  root="$(exp_repo_root)"
  printf '%s/logs/%s.log\n' "$root" "$name"
}

exp_export_metrics() {
  local root tag
  root="$(exp_repo_root)"
  tag="${1:-${EXP_WANDB_GROUP:-paper-reproduction}}"
  "$PYTHON" "$root/scripts/export_experiment_metrics_csv.py" \
    --tag "$tag" \
    --output "$root/results/experiment_metrics_history.csv" \
    --summary-output "$root/results/experiment_metrics_summary.csv" \
    2>&1 | tee -a "$(exp_log_file "export_metrics")"
}

exp_tmux_ensure_free() {
  local session="$1"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session '$session' already exists — attach: tmux attach -t $session" >&2
    exit 1
  fi
}

exp_tmux_start() {
  local session="$1"
  local window="$2"
  local cmd="$3"
  if [[ "${EXP_NO_TMUX:-0}" == "1" ]]; then
    eval "$cmd"
    return
  fi
  exp_tmux_ensure_free "$session"
  tmux new-session -d -s "$session" -n "$window" "$cmd"
}

exp_tmux_start_window() {
  local session="$1"
  local window="$2"
  local cmd="$3"
  if [[ "${EXP_NO_TMUX:-0}" == "1" ]]; then
    eval "$cmd"
    return
  fi
  tmux new-window -t "$session" -n "$window" "$cmd"
}

exp_tmux_multi_window() {
  local session="$1"
  shift
  if [[ "${EXP_NO_TMUX:-0}" == "1" ]]; then
    while (($# >= 2)); do
      eval "$2"
      shift 2
    done
    return
  fi
  exp_tmux_ensure_free "$session"
  local first_window="$1"
  local first_cmd="$2"
  shift 2
  tmux new-session -d -s "$session" -n "$first_window" "$first_cmd"
  while (($# >= 2)); do
    tmux new-window -t "$session" -n "$1" "$2"
    shift 2
  done
}

exp_print_launch_summary() {
  local session="$1"
  shift
  echo "Started tmux session: $session"
  echo "  attach: tmux attach -t $session"
  while (($# > 0)); do
    echo "  $1"
    shift
  done
}
