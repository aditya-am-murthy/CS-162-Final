#!/usr/bin/env python3
"""Export combined metrics from local W&B runs to CSV."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.common import DEFAULT_CREDENTIALS_PATH, load_wandb_credentials

REPO_ROOT = _root
DEFAULT_WANDB_DIR = REPO_ROOT / "wandb"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "wandb_metrics_history.csv"
DEFAULT_SUMMARY_OUTPUT = REPO_ROOT / "results" / "wandb_metrics_summary.csv"

_SKIP_PREFIXES = ("charts/", "idea2/", "artifact/", "results/")
_SKIP_KEYS = {"_wandb"}


def _arg_value(args: list[str], flag: str) -> str | None:
    for idx, value in enumerate(args):
        if value == flag and idx + 1 < len(args):
            return args[idx + 1]
    return None


def _run_id_from_dir(run_dir: Path) -> str:
    return run_dir.name.rsplit("-", 1)[-1]


def _run_date_from_dir(run_dir: Path) -> str:
    match = re.match(r"run-(\d{8})_", run_dir.name)
    return match.group(1) if match else ""


def _load_metadata(run_dir: Path) -> dict[str, Any]:
    meta_path = run_dir / "files" / "wandb-metadata.json"
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text())


def _load_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "files" / "wandb-summary.json"
    if not summary_path.is_file():
        return {}
    return json.loads(summary_path.read_text())


def _flatten_numeric_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _SKIP_KEYS or key.startswith(_SKIP_PREFIXES):
            continue
        if isinstance(value, (int, float, bool)) and not isinstance(value, bool):
            flat[key] = value
        elif isinstance(value, bool):
            flat[key] = int(value)
        elif isinstance(value, str) and value.replace(".", "", 1).isdigit():
            flat[key] = float(value)
    return flat


def _run_metadata_row(run_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    args = metadata.get("args", [])
    return {
        "wandb_run_dir": run_dir.name,
        "wandb_run_id": _run_id_from_dir(run_dir),
        "wandb_run_name": _arg_value(args, "--wandb-run-name"),
        "wandb_group": _arg_value(args, "--wandb-group"),
        "wandb_project": _arg_value(args, "--wandb-project"),
        "dataset": _arg_value(args, "--dataset"),
        "preset": _arg_value(args, "--preset"),
        "subset_name": _arg_value(args, "--subset-name"),
        "subset_strategy": _arg_value(args, "--subset-strategy"),
        "program": Path(metadata.get("program", "")).name or None,
        "started_at": metadata.get("startedAt"),
        "run_date": _run_date_from_dir(run_dir),
    }


def _select_numeric_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep: list[str] = []
    for col in df.columns:
        if col in _SKIP_KEYS or col.startswith(_SKIP_PREFIXES):
            continue
        if col in {"_step", "_timestamp", "_runtime", "epoch"}:
            keep.append(col)
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            keep.append(col)
    if not keep:
        return pd.DataFrame()
    return df[keep].copy()


def _history_from_local_jsonl(metrics_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with metrics_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "epoch" in df.columns and "_step" not in df.columns:
        df["_step"] = df["epoch"]
    return _select_numeric_history_columns(df)


def _history_from_api(api: Any, entity: str, project: str, run_id: str) -> pd.DataFrame:
    run = api.run(f"{entity}/{project}/{run_id}")
    history = run.history(samples=10_000, pandas=True)
    if history is None or history.empty:
        return pd.DataFrame()
    return _select_numeric_history_columns(history)


def _iter_run_dirs(
    wandb_dir: Path,
    *,
    since: str | None,
    group: str | None,
    max_runs: int | None,
) -> list[Path]:
    run_dirs = sorted(
        [path for path in wandb_dir.glob("run-*") if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )
    selected: list[Path] = []
    for run_dir in run_dirs:
        run_date = _run_date_from_dir(run_dir)
        if since and run_date and run_date < since:
            continue
        if group:
            metadata = _load_metadata(run_dir)
            args = metadata.get("args", [])
            run_group = _arg_value(args, "--wandb-group")
            if run_group != group:
                continue
        selected.append(run_dir)
        if max_runs is not None and len(selected) >= max_runs:
            break
    return selected


def export_metrics(
    *,
    wandb_dir: Path,
    output: Path,
    summary_output: Path,
    since: str | None,
    group: str | None,
    max_runs: int | None,
    use_api: bool,
) -> tuple[Path, Path]:
    creds = load_wandb_credentials(DEFAULT_CREDENTIALS_PATH)
    entity = creds.get("entity")
    project = creds.get("project", "cs162-dataset-cartography")

    api = None
    if use_api:
        import wandb

        api = wandb.Api()

    run_dirs = _iter_run_dirs(wandb_dir, since=since, group=group, max_runs=max_runs)
    if not run_dirs:
        raise SystemExit("No wandb runs matched the requested filters.")

    history_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        metadata = _load_metadata(run_dir)
        meta_row = _run_metadata_row(run_dir, metadata)
        summary_row = {**meta_row, **_flatten_numeric_metrics(_load_summary(run_dir))}
        summary_rows.append(summary_row)

        metrics_path_raw = _arg_value(metadata.get("args", []), "--metrics-out")
        history = pd.DataFrame()
        if metrics_path_raw:
            metrics_path = Path(metrics_path_raw)
            if not metrics_path.is_absolute():
                metrics_path = REPO_ROOT / metrics_path
            if metrics_path.is_file():
                history = _history_from_local_jsonl(metrics_path)

        if history.empty and use_api and api is not None and entity:
            run_id = meta_row["wandb_run_id"]
            try:
                history = _history_from_api(api, entity, project, run_id)
            except Exception as exc:  # pragma: no cover - network/API edge cases
                print(f"warn: could not fetch history for {run_id}: {exc}", file=sys.stderr)

        if history.empty:
            continue

        for key, value in meta_row.items():
            history[key] = value
        history_frames.append(history)

    output.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_output, index=False)

    if history_frames:
        history_df = pd.concat(history_frames, ignore_index=True, sort=False)
        meta_cols = [
            "wandb_run_dir",
            "wandb_run_id",
            "wandb_run_name",
            "wandb_group",
            "wandb_project",
            "dataset",
            "preset",
            "subset_name",
            "subset_strategy",
            "program",
            "started_at",
            "run_date",
        ]
        metric_cols = [col for col in history_df.columns if col not in meta_cols]
        preferred = [col for col in meta_cols if col in history_df.columns]
        step_cols = [col for col in ("_step", "epoch", "_timestamp", "_runtime") if col in metric_cols]
        other_cols = sorted(col for col in metric_cols if col not in step_cols)
        history_df = history_df[preferred + step_cols + other_cols]
        history_df.to_csv(output, index=False)
    else:
        output.write_text("")

    return output, summary_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-dir", type=Path, default=DEFAULT_WANDB_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument(
        "--since",
        default="20260609",
        help="Only include runs from this date onward (YYYYMMDD). Use 'all' for every local run.",
    )
    parser.add_argument("--group", default=None, help="Filter to a specific --wandb-group value.")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Only use local training_metrics.jsonl files; skip cloud history fallback.",
    )
    args = parser.parse_args()

    since = None if args.since == "all" else args.since
    output, summary_output = export_metrics(
        wandb_dir=args.wandb_dir,
        output=args.output,
        summary_output=args.summary_output,
        since=since,
        group=args.group,
        max_runs=args.max_runs,
        use_api=not args.no_api,
    )

    history_rows = max(sum(1 for _ in output.open()) - 1, 0) if output.is_file() else 0
    summary_rows = max(sum(1 for _ in summary_output.open()) - 1, 0)
    print(f"Wrote {history_rows} history rows to {output}")
    print(f"Wrote {summary_rows} run summaries to {summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
