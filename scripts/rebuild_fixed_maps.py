#!/usr/bin/env python3
"""Rebuild data maps for multiple region-assignment schemes and density opacity."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from ml_cartography.analysis.data_map import (
    RegionMode,
    annotate_regions,
    fit_region_thresholds,
    prepare_region_annotations,
    save_data_map_plot,
)
from ml_cartography.utils.io import read_jsonl, write_jsonl
from scripts.run_cartography_experiment import _collect_dynamics_from_logs

REGION_VARIANTS: list[tuple[str, RegionMode, str]] = [
    ("adaptive", "adaptive", "adaptive rank regions"),
    ("hard-limit", "absolute", "hard limit (paper axis cutoffs)"),
    ("equal-thirds", "equal_thirds", "30% / 30% / 30% rank split"),
]


def _run_prefix(run_dir: Path) -> str:
    """Use YYYYMMDD_HHMMSS from run folder names like 20260609_074628_snli_..."""
    parts = run_dir.name.split("_")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return f"{parts[0]}_{parts[1]}"
    return run_dir.name


def _coords_through_epoch(log_path: Path, max_epoch: int) -> list[dict]:
    records = read_jsonl(log_path)
    filtered = [r for r in records if int(r["epoch"]) <= max_epoch]
    by_guid: dict = {}
    from ml_cartography.core.dynamics import (
        add_epoch_observation,
        build_record,
        summarize_record,
    )

    for row in filtered:
        guid = row["guid"]
        if guid not in by_guid:
            by_guid[guid] = build_record(guid, int(row["gold_label"]))
        add_epoch_observation(
            by_guid[guid],
            float(row["prob_gold"]),
            int(row["pred_label"]),
        )
    return [summarize_record(r) for r in by_guid.values()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run root with dynamics/epoch_predictions.jsonl")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root (default: <run_dir>/fixed-maps)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    log_path = run_dir / "dynamics" / "epoch_predictions.jsonl"
    if not log_path.is_file():
        raise SystemExit(f"missing {log_path}")

    cfg_path = run_dir / "config.json"
    dataset = "snli"
    if cfg_path.is_file():
        with cfg_path.open(encoding="utf-8") as f:
            dataset = json.load(f).get("dataset", dataset)

    prefix = _run_prefix(run_dir)
    root_out = (args.output_dir or run_dir / "fixed-maps").resolve()
    root_out.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(log_path)
    max_epoch = max(int(r["epoch"]) for r in records)
    titles = {
        "snli": "SNLI",
        "mnli": "MultiNLI",
        "qnli": "QNLI",
        "winogrande": "WinoGrande",
    }
    ds_title = titles.get(dataset, dataset.upper())

    written: list[Path] = []
    summaries: dict[str, dict] = {}

    for folder_name, region_mode, label in REGION_VARIANTS:
        out_dir = root_out / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)

        def write_maps(coords: list[dict], stem: str, subtitle: str) -> None:
            tagged = annotate_regions(coords, region_mode=region_mode)
            guide = fit_region_thresholds(coords, region_mode=region_mode)
            for color_by, suffix in (("correctness", "correctness"), ("region", "regions")):
                path = out_dir / f"{prefix}_{stem}_{suffix}.png"
                save_data_map_plot(
                    tagged,
                    path,
                    color_by=color_by,
                    title=f"{ds_title} {subtitle} ({label})",
                    thresholds=guide,
                    opacity_mode="density",
                    density_penalty="parabolic",
                    show_stats_table=True,
                )
                written.append(path)

        final_coords = _collect_dynamics_from_logs(log_path)
        write_maps(final_coords, "data_map", "data map (final)")

        for epoch in range(1, max_epoch + 1):
            coords = _coords_through_epoch(log_path, epoch)
            write_maps(coords, f"epoch{epoch:02d}", f"data map (epoch {epoch})")

        tagged, summary = prepare_region_annotations(final_coords, region_mode=region_mode)
        write_jsonl(out_dir / f"{prefix}_cartography_with_regions.jsonl", tagged)
        summary_path = out_dir / f"{prefix}_region_counts.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "region_mode": region_mode,
                    "label": label,
                    "thresholds": summary["thresholds"],
                    "counts": summary["counts"],
                    "shares": summary["shares"],
                    "num_examples": summary["num_examples"],
                },
                f,
                indent=2,
            )
        summaries[folder_name] = dict(Counter(r["region"] for r in tagged))

    print(f"wrote {len(written)} maps under {root_out}")
    for folder_name, _, label in REGION_VARIANTS:
        print(f"\n[{folder_name}] {label}")
        print(f"  counts: {summaries[folder_name]}")
        variant_dir = root_out / folder_name
        for p in sorted(variant_dir.glob("*.png")):
            print(f"  {folder_name}/{p.name}")


if __name__ == "__main__":
    main()
