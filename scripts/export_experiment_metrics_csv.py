#!/usr/bin/env python3
"""Aggregate per-epoch training metrics from all experiment output directories."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

REPO_ROOT = _root

DEFAULT_SEARCH_ROOTS = (
    REPO_ROOT / "data" / "processed" / "easy_role",
    REPO_ROOT / "data" / "processed" / "region_finetune_winogrande",
    REPO_ROOT / "data" / "processed" / "region_subsets",
    REPO_ROOT / "data" / "processed" / "noise_detection_paper",
    REPO_ROOT / "data" / "processed" / "subregion_compute_gain",
    REPO_ROOT / "data" / "processed" / "table3_snli_mnli",
    REPO_ROOT / "data" / "processed" / "table4_qnli",
    REPO_ROOT / "experiments" / "runs",
    REPO_ROOT / "results",
)

METRICS_FILENAMES = (
    "training_metrics.jsonl",
    "noised_training_metrics.jsonl",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _infer_experiment_id(metrics_path: Path) -> str:
    parts = metrics_path.parts
    if "training_runs" in parts:
        idx = parts.index("training_runs")
        if idx + 2 < len(parts):
            return "/".join(parts[idx + 1 : idx + 3])
    if "logs" in parts and parts[-1] in METRICS_FILENAMES:
        run_idx = next((i for i, p in enumerate(parts) if p in {"runs", "results"}), None)
        if run_idx is not None and run_idx + 1 < len(parts):
            return parts[run_idx + 1]
    return metrics_path.parent.name


def _parse_run_meta(metrics_path: Path, summary_path: Path | None) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "metrics_path": str(metrics_path),
        "experiment_id": _infer_experiment_id(metrics_path),
    }
    if summary_path and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for key in (
            "dataset",
            "preset",
            "subset_name",
            "subset_strategy",
            "strategy",
            "final_val_accuracy",
            "num_train",
            "num_val",
            "epochs",
            "seed",
        ):
            if key in summary:
                meta[key] = summary[key]
    parts = metrics_path.parts
    if "restart_" in parts[-2]:
        meta["restart"] = parts[-2]
    if "training_runs" in parts:
        idx = parts.index("training_runs")
        if idx + 1 < len(parts):
            meta["subset_name"] = meta.get("subset_name") or parts[idx + 1]
    exp_root = metrics_path.parents[2] if "training_runs" in parts else metrics_path.parents[1]
    manifest = exp_root / "manifest.json"
    train_results = exp_root / "train_results.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        meta["experiment_tag"] = payload.get("tag") or exp_root.name
        meta["wandb_group"] = payload.get("wandb_group")
    else:
        meta["experiment_tag"] = exp_root.name
    if train_results.is_file() and "final_val_accuracy" not in meta:
        results = json.loads(train_results.read_text(encoding="utf-8"))
        subset = meta.get("subset_name")
        for row in results:
            if row.get("subset") == subset or row.get("name") == subset:
                meta["final_val_accuracy"] = row.get("final_val_accuracy")
                meta["strategy"] = row.get("strategy") or row.get("subset_strategy")
                break
    return meta


def _find_metrics_files(roots: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for name in METRICS_FILENAMES:
            for path in root.rglob(name):
                if path in seen:
                    continue
                seen.add(path)
                found.append(path)
    return sorted(found)


def _filter_by_tag(paths: list[Path], tag: str | None) -> list[Path]:
    if not tag:
        return paths
    kept: list[Path] = []
    for path in paths:
        exp_root = path.parents[2] if "training_runs" in path.parts else path.parents[1]
        manifest = exp_root / "manifest.json"
        if manifest.is_file():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            manifest_tag = payload.get("tag") or payload.get("wandb_group") or exp_root.name
            if tag in {manifest_tag, payload.get("wandb_group"), exp_root.name}:
                kept.append(path)
                continue
        if tag in str(path):
            kept.append(path)
    return kept


def export_metrics(
    *,
    search_roots: list[Path],
    output: Path,
    summary_output: Path,
    tag: str | None,
    since: str | None,
) -> tuple[int, int]:
    metrics_files = _find_metrics_files(search_roots)
    metrics_files = _filter_by_tag(metrics_files, tag)

    history_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for metrics_path in metrics_files:
        if since:
            match = re.search(r"(\d{8})", str(metrics_path))
            if match and match.group(1) < since:
                continue

        summary_path = metrics_path.parent / "summary.json"
        meta = _parse_run_meta(metrics_path, summary_path)
        rows = _load_jsonl(metrics_path)
        if not rows:
            continue

        history = pd.DataFrame(rows)
        if "epoch" in history.columns and "_step" not in history.columns:
            history["_step"] = history["epoch"]
        for key, value in meta.items():
            history[key] = value
        history_frames.append(history)

        last = rows[-1]
        summary = {
            **meta,
            **{k: v for k, v in last.items() if isinstance(v, (int, float))},
            "num_epochs_logged": len(rows),
        }
        if summary_path.is_file():
            summary.update(
                {
                    k: v
                    for k, v in json.loads(summary_path.read_text(encoding="utf-8")).items()
                    if isinstance(v, (int, float, str))
                }
            )
        summary_rows.append(summary)

    output.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_output, index=False)

    history_count = 0
    if history_frames:
        history_df = pd.concat(history_frames, ignore_index=True, sort=False)
        meta_cols = [
            "experiment_tag",
            "experiment_id",
            "wandb_group",
            "subset_name",
            "subset_strategy",
            "strategy",
            "dataset",
            "preset",
            "restart",
            "metrics_path",
        ]
        step_cols = [c for c in ("_step", "epoch", "_timestamp", "_runtime") if c in history_df.columns]
        preferred = [c for c in meta_cols if c in history_df.columns]
        other_cols = sorted(
            c for c in history_df.columns if c not in preferred and c not in step_cols
        )
        history_df = history_df[preferred + step_cols + other_cols]
        history_df.to_csv(output, index=False)
        history_count = len(history_df)
    else:
        output.write_text("")

    return history_count, len(summary_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-root",
        action="append",
        type=Path,
        default=[],
        help="Experiment output root to scan (repeatable).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "experiment_metrics_history.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=REPO_ROOT / "results" / "experiment_metrics_summary.csv",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Filter to runs whose manifest tag / wandb_group / path contains this value.",
    )
    parser.add_argument("--since", default=None, help="Only include paths with YYYYMMDD >= this date.")
    args = parser.parse_args()

    roots = args.search_root or list(DEFAULT_SEARCH_ROOTS)
    history_count, summary_count = export_metrics(
        search_roots=roots,
        output=args.output,
        summary_output=args.summary_output,
        tag=args.tag,
        since=args.since,
    )
    print(f"Wrote {history_count} history rows to {args.output}")
    print(f"Wrote {summary_count} run summaries to {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
