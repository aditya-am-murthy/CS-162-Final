#!/usr/bin/env python3
"""
Streamlit shell for Dataset Cartography (upload → train → map → filter).

Run: streamlit run apps/streamlit_app.py

Expects completed runs under results/<run_id>/ with manifest.json + dynamics/.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def _list_runs() -> list[str]:
    if not RESULTS.is_dir():
        return []
    return sorted(
        [p.name for p in RESULTS.iterdir() if p.is_dir()],
        reverse=True,
    )


def main() -> None:
    st.set_page_config(page_title="Dataset Cartography", layout="wide")
    st.title("Dataset Cartography Explorer")
    st.caption("Select a timestamped experiment run from results/")

    runs = _list_runs()
    if not runs:
        st.warning("No runs in results/ yet. Run: python scripts/train_suite.py --snli-encoders")
        return

    run_id = st.selectbox("Experiment run", runs)
    run_dir = RESULTS / run_id
    manifest_path = run_dir / "manifest.json"

    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as f:
            st.json(json.load(f))

    fig_dir = run_dir / "figures"
    if fig_dir.is_dir():
        for img in sorted(fig_dir.glob("*.png")):
            st.image(str(img), caption=img.name, use_container_width=True)

    dyn = run_dir / "dynamics" / "cartography_with_regions.jsonl"
    if dyn.is_file():
        st.subheader("Filter by region (preview)")
        region = st.selectbox("Region", ["ambiguous", "easy_to_learn", "hard_to_learn", "mixed"])
        lines = []
        with dyn.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("region") == region:
                    lines.append(row)
                if len(lines) >= 50:
                    break
        st.dataframe(lines)


if __name__ == "__main__":
    main()
