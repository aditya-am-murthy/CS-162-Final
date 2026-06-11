#!/usr/bin/env python3
"""Copy figures and tables into paper_outputs/ with paper Figure/Table numbering."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from ml_cartography.analysis.paper_tables import (
    plot_fig3_from_easy_role,
    plot_table2_from_metrics,
    plot_table34_from_metrics,
)

REPO = _root
OUT = REPO / "paper_outputs"
MANIFEST_PATH = OUT / "manifest.json"


def _first_existing(*candidates: Path) -> Path | None:
    for path in candidates:
        if path and path.is_file():
            return path
    return None


def _latest_glob(root: Path, pattern: str) -> Path | None:
    if not root.is_dir():
        return None
    matches = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _copy(src: Path | None, dest_name: str, manifest: dict, note: str = "") -> None:
    if src is None:
        manifest[dest_name] = {"status": "missing", "note": note}
        return
    dest = OUT / dest_name
    shutil.copy2(src, dest)
    manifest[dest_name] = {"status": "ok", "source": str(src.relative_to(REPO)), "note": note}


def _shrink_png(path: Path, max_width: int = 640) -> None:
    """Keep paper_outputs web-friendly; Fig 5 is cropped to a square."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    if "Figure_05" in path.name:
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        w = h = side
    if w > max_width:
        nh = max(1, int(h * max_width / w))
        img = img.resize((max_width, nh), Image.Resampling.LANCZOS)
    img.save(path, optimize=True)


def _export_table_csv(metrics_path: Path, output_csv: Path) -> bool:
    if not metrics_path.is_file():
        return False
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = []
    for strategy, m in metrics.items():
        rows.append({"strategy": strategy, **m})
    if not rows:
        return False
    import pandas as pd

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return True


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"paper_outputs_dir": str(OUT.relative_to(REPO))}

    wg = REPO / "results/20260609_074628_snli_winogrande_roberta-large"
    plots = REPO / "results/paper_plots_from_metrics"
    insight = wg / "figures"
    fixed = wg / "fixed-maps/adaptive"

    # Resolve latest subsampled SNLI/MNLI/QNLI maps if present
    snli_run = _latest_glob(REPO / "results", "*_snli_*")
    mnli_run = _latest_glob(REPO / "results", "*_mnli_*")
    qnli_run = _latest_glob(REPO / "results", "*_qnli_*")

    fig1 = _first_existing(
        snli_run / "fixed-maps/adaptive" / "data_map_correctness.png" if snli_run else None,
        _latest_glob(snli_run / "fixed-maps/adaptive", "*_data_map_correctness.png") if snli_run else None,
        snli_run / "figures/data_map.png" if snli_run else None,
        REPO / "results/20260527_051157_snli_mnli_roberta-base/figures/data_map.png",
        insight / "fig01_data_map_correctness.png",
    )

    fig2 = _first_existing(
        fixed / "20260609_074628_data_map_correctness.png",
        fixed / "20260609_074628_data_map_regions.png",
        wg / "figures/data_map_regions.png",
        insight / "fig01_data_map_correctness.png",
    )

    easy_role_results = REPO / "data/processed/easy_role/train_results.json"
    easy_role_manifest = REPO / "data/processed/easy_role/manifest.json"
    fig3_out = plots / "fig03_easy_to_learn_measured.png"
    if plot_fig3_from_easy_role(easy_role_results, easy_role_manifest, fig3_out):
        fig3 = fig3_out
    else:
        fig3 = _first_existing(
            plots / "fig03_easy_to_learn_measured.png",
            insight / "fig04_ambiguous_ablation_curves.png",
        )
    fig4 = _first_existing(
        plots / "fig04_noise_shift_measured.png",
        insight / "fig05_noise_injection_shift.png",
    )
    fig5 = _first_existing(
        plots / "fig05_agreement_heatmap_measured.png",
        insight / "fig06_human_agreement_heatmap.png",
    )

    _copy(fig1, "Figure_01_snli_data_map.png", manifest, "SNLI cartography map")
    _copy(fig2, "Figure_02_winogrande_data_map.png", manifest, "WinoGrande cartography map")
    _copy(fig3, "Figure_03_easy_to_learn_role.png", manifest, "Ambiguous scaling + replacement")
    _copy(fig4, "Figure_04_noise_shift.png", manifest, "Label-noise before/after histograms")
    _copy(fig5, "Figure_05_human_agreement_heatmap.png", manifest, "Human agreement on data map")

    # Table 2
    table2_metrics = REPO / "results/region_metrics_table2.json"
    table2_png = OUT / "Table_02_winogrande_selection.png"
    table2_tex = OUT / "Table_02_winogrande_selection.tex"
    if plot_table2_from_metrics(table2_metrics, table2_png, table2_tex):
        manifest["Table_02_winogrande_selection.png"] = {
            "status": "ok",
            "source": str(table2_metrics.relative_to(REPO)),
        }
        manifest["Table_02_winogrande_selection.tex"] = {
            "status": "ok",
            "source": str(table2_metrics.relative_to(REPO)),
        }
    else:
        manifest["Table_02_winogrande_selection.png"] = {"status": "missing"}
        manifest["Table_02_winogrande_selection.tex"] = {"status": "missing"}
    _export_table_csv(table2_metrics, OUT / "Table_02_winogrande_selection.csv")
    if table2_metrics.is_file():
        manifest["Table_02_winogrande_selection.csv"] = {
            "status": "ok",
            "source": str(table2_metrics.relative_to(REPO)),
        }

    # Table 3
    if (REPO / "data/processed/table3_snli_mnli/snli").is_dir():
        for tag in ("snli", "mnli"):
            results = REPO / f"data/processed/table3_snli_mnli/{tag}/train_results.json"
            manifest_path = REPO / f"data/processed/table3_snli_mnli/{tag}/manifest.json"
            metrics_out = REPO / f"results/region_metrics_table3_{tag}.json"
            if results.is_file() and manifest_path.is_file():
                import subprocess

                subprocess.run(
                    [
                        sys.executable,
                        str(REPO / "scripts/11_region_metrics.py"),
                        "--results",
                        str(results),
                        "--manifest",
                        str(manifest_path),
                        "--output",
                        str(metrics_out),
                        "--no-wandb",
                    ],
                    check=False,
                    cwd=REPO,
                )
                ood_header = "Diagnostics (OOD proxy)"
                id_header = "Test" if tag == "snli" else "Matched"
                plot_table34_from_metrics(
                    metrics_out,
                    OUT / f"Table_03_{tag}_selection.png",
                    OUT / f"Table_03_{tag}_selection.tex",
                    table_num=3,
                    dataset=tag,
                    id_header=id_header,
                    ood_header=ood_header,
                )
                _export_table_csv(metrics_out, OUT / f"Table_03_{tag}_selection.csv")
                manifest[f"Table_03_{tag}_selection.png"] = {"status": "ok", "source": str(metrics_out.relative_to(REPO))}

    # Table 4
    table4_metrics = REPO / "results/region_metrics_table4.json"
    if not table4_metrics.is_file():
        t4_results = REPO / "data/processed/table4_qnli/train_results.json"
        t4_manifest = REPO / "data/processed/table4_qnli/manifest.json"
        if t4_results.is_file() and t4_manifest.is_file():
            import subprocess

            subprocess.run(
                [
                    sys.executable,
                    str(REPO / "scripts/11_region_metrics.py"),
                    "--results",
                    str(t4_results),
                    "--manifest",
                    str(t4_manifest),
                    "--output",
                    str(table4_metrics),
                    "--no-wandb",
                ],
                check=False,
                cwd=REPO,
            )
    if table4_metrics.is_file():
        plot_table34_from_metrics(
            table4_metrics,
            OUT / "Table_04_qnli_selection.png",
            OUT / "Table_04_qnli_selection.tex",
            table_num=4,
            dataset="qnli",
            id_header="QNLI Val.",
            ood_header="Adversarial (OOD proxy)",
        )
        _export_table_csv(table4_metrics, OUT / "Table_04_qnli_selection.csv")
        manifest["Table_04_qnli_selection.png"] = {
            "status": "ok",
            "source": str(table4_metrics.relative_to(REPO)),
        }
    else:
        manifest["Table_04_qnli_selection.png"] = {"status": "missing", "note": "run exp 09"}

    # Appendix maps from subset runs
    if mnli_run:
        m = _latest_glob(mnli_run / "fixed-maps/adaptive", "*_data_map_correctness.png")
        _copy(m, "Appendix_Figure_mnli_data_map.png", manifest, "Subsampled MNLI map")
    if qnli_run:
        q = _latest_glob(qnli_run / "fixed-maps/adaptive", "*_data_map_correctness.png")
        _copy(q, "Appendix_Figure_qnli_data_map.png", manifest, "Subsampled QNLI map")

    for png in OUT.glob("*.png"):
        _shrink_png(png)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Collected paper outputs in {OUT}/")
    for name, info in sorted(manifest.items()):
        if name == "paper_outputs_dir":
            continue
        status = info.get("status", "?") if isinstance(info, dict) else "ok"
        print(f"  [{status}] {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
