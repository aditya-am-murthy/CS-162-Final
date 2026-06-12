#!/usr/bin/env python3
"""Collect extra experiment figures into extension_outputs/."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from ml_cartography.analysis.extension_figures import (
    plot_extra_01_maps_panel,
    plot_extra_01_multi_architecture,
    plot_extra_02_preference_map,
    plot_extra_03_dynamic_curriculum,
    plot_extra_04_bilateral_transitions,
    plot_extra_04_detector_cross_eval,
    plot_extra_04_recovery_bars,
    plot_extra_04_summary_table,
)
from ml_cartography.experiments.bilateral_noise import region_transition_matrix
from ml_cartography.utils.io import read_jsonl

REPO = _root
OUT = REPO / "extension_outputs"
MANIFEST_PATH = OUT / "manifest.json"


def _shrink(path: Path, max_width: int = 640) -> None:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w > max_width:
        nh = max(1, int(h * max_width / w))
        img = img.resize((max_width, nh), Image.Resampling.LANCZOS)
        img.save(path, optimize=True)


def _copy(src: Path | None, name: str, manifest: dict, note: str = "") -> None:
    if src is None or not src.is_file():
        manifest[name] = {"status": "missing", "note": note}
        return
    shutil.copy2(src, OUT / name)
    manifest[name] = {"status": "ok", "source": str(src.relative_to(REPO)), "note": note}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"extension_outputs_dir": str(OUT.relative_to(REPO))}

    # Extra #1 — multi-architecture SNLI maps
    arch_paths = [
        ("DistilBERT", REPO / "data/trained_models/distilbert/dynamics/cartography_with_regions.jsonl"),
        ("RoBERTa-base", REPO / "data/trained_models/roberta-base/dynamics/cartography_with_regions.jsonl"),
    ]
    e1_bars = OUT / "Extra_01_multi_architecture_regions.png"
    e1_maps = OUT / "Extra_01_multi_architecture_maps.png"
    if plot_extra_01_multi_architecture(arch_paths, e1_bars):
        manifest["Extra_01_multi_architecture_regions.png"] = {"status": "ok"}
    else:
        manifest["Extra_01_multi_architecture_regions.png"] = {"status": "missing"}
    if plot_extra_01_maps_panel(arch_paths, e1_maps):
        manifest["Extra_01_multi_architecture_maps.png"] = {"status": "ok"}
    else:
        manifest["Extra_01_multi_architecture_maps.png"] = {"status": "missing"}

    # Extra #2 — preference cartography
    pref_path = REPO / "data/processed/preference_ultrafeedback/dynamics/cartography_coordinates.jsonl"
    e2 = OUT / "Extra_02_preference_data_map.png"
    plot_extra_02_preference_map(pref_path if pref_path.is_file() else None, e2)
    manifest["Extra_02_preference_data_map.png"] = {
        "status": "ok",
        "note": "synthetic demo" if not pref_path.is_file() else "measured",
    }

    # Extra #3 — dynamic curriculum
    snap_dir = REPO / "data/trained_models/roberta-base/dynamics/snapshots"
    traj = REPO / "data/trained_models/roberta-base/dynamics/region_trajectories.jsonl"
    e3_curve = OUT / "Extra_03_dynamic_region_mix.png"
    e3_trans = OUT / "Extra_03_dynamic_region_transition.png"
    if plot_extra_03_dynamic_curriculum(snap_dir, traj, e3_curve, e3_trans):
        manifest["Extra_03_dynamic_region_mix.png"] = {"status": "ok"}
        manifest["Extra_03_dynamic_region_transition.png"] = {
            "status": "ok" if e3_trans.is_file() else "missing"
        }
    else:
        manifest["Extra_03_dynamic_region_mix.png"] = {"status": "missing"}

    # Extra #4 — bilateral 1% flip
    bilateral_dir = REPO / "data/processed/bilateral_noise_flip"
    analysis_path = bilateral_dir / "analysis_summary.json"
    if analysis_path.is_file():
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        easy_shift = read_jsonl(bilateral_dir / "easy_shift.jsonl")
        hard_shift = read_jsonl(bilateral_dir / "hard_shift.jsonl")
        plot_extra_04_bilateral_transitions(
            region_transition_matrix(easy_shift),
            region_transition_matrix(hard_shift),
            OUT / "Extra_04_bilateral_region_transitions.png",
        )
        plot_extra_04_detector_cross_eval(
            analysis["detector_cross_eval"],
            OUT / "Extra_04_detector_cross_eval.png",
        )
        plot_extra_04_recovery_bars(
            analysis["easy_arm"],
            analysis["hard_arm"],
            analysis["comparison"],
            OUT / "Extra_04_bilateral_recovery.png",
        )
        plot_extra_04_summary_table(
            analysis["easy_arm"],
            analysis["hard_arm"],
            analysis["comparison"],
            analysis["detector_cross_eval"],
            OUT / "Extra_04_bilateral_metrics_table.png",
        )
        for name in (
            "Extra_04_bilateral_region_transitions.png",
            "Extra_04_detector_cross_eval.png",
            "Extra_04_bilateral_recovery.png",
            "Extra_04_bilateral_metrics_table.png",
        ):
            manifest[name] = {"status": "ok", "source": str(bilateral_dir.relative_to(REPO))}
    else:
        fig_dir = bilateral_dir / "figures"
        for src_name, dest_name in (
            ("bilateral_region_transitions.png", "Extra_04_bilateral_region_transitions.png"),
            ("detector_cross_eval.png", "Extra_04_detector_cross_eval.png"),
            ("bilateral_recovery_bars.png", "Extra_04_bilateral_recovery.png"),
            ("bilateral_metrics_table.png", "Extra_04_bilateral_metrics_table.png"),
        ):
            _copy(fig_dir / src_name if fig_dir.is_dir() else None, dest_name, manifest, "run scripts/run_exp_11_extra_bilateral_noise.sh")

    for png in OUT.glob("*.png"):
        _shrink(png)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Collected extension outputs in {OUT}/")
    for name, info in sorted(manifest.items()):
        if name == "extension_outputs_dir":
            continue
        status = info.get("status", "?") if isinstance(info, dict) else "ok"
        print(f"  [{status}] {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
