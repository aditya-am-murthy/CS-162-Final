#!/usr/bin/env python3
"""
Streamlit shell for Dataset Cartography (upload → train → map → filter).

Run: streamlit run apps/streamlit_app.py

Expects completed runs under results/<run_id>/ with manifest.json + dynamics/.
"""

from __future__ import annotations

import streamlit as st

from results_loader import (
    get_run_kind,
    get_run_dir,
    has_region_rows,
    list_figure_paths,
    list_runs,
    load_config,
    load_manifest,
    load_summary,
    preview_region_rows,
    run_exists,
    summarize_report_rows,
)


def _render_key_value_metadata(title: str, payload: dict | None) -> None:
    st.subheader(title)
    if not payload:
        st.info(f"No {title.lower()} available for this selection.")
        return
    st.json(payload)


def _render_json_report(payload: dict | None) -> None:
    st.subheader("Report Summary")
    if not payload:
        st.warning("This JSON report could not be loaded.")
        return

    summary = summarize_report_rows(payload)
    if summary:
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", summary["num_rows"])
        col2.metric("Datasets", len(summary["datasets"]))
        best_accuracy = summary["best_accuracy"]
        col3.metric(
            "Best Val Accuracy",
            f"{best_accuracy:.4f}" if best_accuracy is not None else "N/A",
        )
        st.caption(
            "Datasets: "
            + ", ".join(summary["datasets"])
            + " | Strategies: "
            + ", ".join(summary["strategies"])
        )

    results_rows = payload.get("results")
    if isinstance(results_rows, list) and results_rows:
        st.subheader("Report Rows")
        st.dataframe(results_rows, use_container_width=True)
    else:
        st.info("This JSON file does not contain a tabular `results` list.")

    with st.expander("Raw JSON", expanded=False):
        st.json(payload)


def main() -> None:
    st.set_page_config(page_title="Dataset Cartography", layout="wide")
    st.title("Dataset Cartography Explorer")
    st.caption("Browse published runs and JSON result artifacts from `results/`.")

    runs = list_runs()
    if not runs:
        st.warning("No result artifacts found in `results/` yet.")
        st.info(
            "The app accepts either `results/<run_id>/` folders or root-level JSON reports such as `results/train_results.json`."
        )
        return

    run_id = st.selectbox("Experiment run", runs)
    run_dir = get_run_dir(run_id)
    run_kind = get_run_kind(run_dir)

    if not run_exists(run_id):
        st.error(f"Selected artifact does not exist: {run_id}")
        return

    st.caption(f"Selected artifact: `{run_id}` ({run_kind})")

    manifest = load_manifest(run_dir)
    if run_kind == "json_report":
        _render_json_report(manifest)
        return

    config = load_config(run_dir)
    summary = load_summary(run_dir)

    col1, col2, col3 = st.columns(3)
    col1.metric("Figures", len(list_figure_paths(run_dir)))
    col2.metric("Region Preview", "Available" if has_region_rows(run_dir) else "Missing")
    col3.metric("Manifest", "Available" if manifest is not None else "Missing")

    _render_key_value_metadata("Manifest", manifest)
    _render_key_value_metadata("Config", config)
    _render_key_value_metadata("Summary", summary)

    figure_paths = list_figure_paths(run_dir)
    st.subheader("Figures")
    if figure_paths:
        for img in figure_paths:
            st.image(str(img), caption=img.name, use_container_width=True)
    else:
        st.info("No PNG figures were found for this run.")

    if has_region_rows(run_dir):
        st.subheader("Filter by region (preview)")
        region = st.selectbox("Region", ["ambiguous", "easy_to_learn", "hard_to_learn", "mixed"])
        st.dataframe(preview_region_rows(run_dir, region=region, limit=50))
    else:
        st.info("No `cartography_with_regions.jsonl` preview is available for this run.")


if __name__ == "__main__":
    main()
