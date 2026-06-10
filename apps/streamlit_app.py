#!/usr/bin/env python3
"""
Streamlit shell for Dataset Cartography (upload → train → map → filter).

Run: streamlit run apps/streamlit_app.py

Expects completed runs under results/<run_id>/ with manifest.json + dynamics/.
"""

from __future__ import annotations

import streamlit as st

from results_loader import (
    get_run_dir,
    has_region_rows,
    list_figure_paths,
    list_runs,
    load_manifest,
    preview_region_rows,
)


def main() -> None:
    st.set_page_config(page_title="Dataset Cartography", layout="wide")
    st.title("Dataset Cartography Explorer")
    st.caption("Select a timestamped experiment run from results/")

    runs = list_runs()
    if not runs:
        st.warning("No runs in results/ yet. Run: python scripts/train_suite.py --snli-encoders")
        return

    run_id = st.selectbox("Experiment run", runs)
    run_dir = get_run_dir(run_id)
    manifest = load_manifest(run_dir)

    if manifest is not None:
        st.json(manifest)

    for img in list_figure_paths(run_dir):
        st.image(str(img), caption=img.name, use_container_width=True)

    if has_region_rows(run_dir):
        st.subheader("Filter by region (preview)")
        region = st.selectbox("Region", ["ambiguous", "easy_to_learn", "hard_to_learn", "mixed"])
        st.dataframe(preview_region_rows(run_dir, region=region, limit=50))


if __name__ == "__main__":
    main()
