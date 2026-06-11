#!/usr/bin/env python3
"""
Streamlit shell for Dataset Cartography (upload → train → map → filter).

Run: streamlit run apps/streamlit_app.py

Expects completed runs under results/<run_id>/ with manifest.json + dynamics/.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from results_loader import (
    get_run_kind,
    get_run_dir,
    has_region_rows,
    jsonl_has_region_field,
    list_dynamics_files,
    list_fixed_map_images,
    list_fixed_map_json,
    list_fixed_map_modes,
    list_figure_paths,
    list_log_files,
    list_model_files,
    list_region_values,
    list_runs,
    list_snapshot_images,
    list_snapshot_jsonl,
    load_config,
    load_manifest,
    load_json_from_path,
    load_region_counts,
    load_summary,
    load_training_metrics,
    preview_jsonl_path,
    preview_region_rows_from_path,
    run_exists,
    summarize_report_rows,
)


def _render_key_value_metadata(title: str, payload: dict | None) -> None:
    st.subheader(title)
    if not payload:
        st.info(f"No {title.lower()} available for this selection.")
        return
    st.json(payload)


def _path_label(run_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return path.name


def _render_image_gallery(run_dir: Path, paths: list[Path], title: str, key: str) -> None:
    st.subheader(title)
    if not paths:
        st.info(f"No assets found for {title.lower()}.")
        return
    labels = [_path_label(run_dir, path) for path in paths]
    selected_label = st.selectbox(title, labels, key=key)
    selected_path = paths[labels.index(selected_label)]
    st.image(str(selected_path), caption=selected_label, use_container_width=True)


def _render_image_preview(run_dir: Path, path: Path | None, title: str) -> None:
    st.subheader(title)
    if path is None or not path.is_file():
        st.info(f"No file is available for {title.lower()}.")
        return
    st.image(str(path), caption=_path_label(run_dir, path), use_container_width=True)


def _render_jsonl_preview_path(
    run_dir: Path,
    path: Path | None,
    title: str,
    limit: int = 50,
) -> None:
    st.subheader(title)
    if path is None or not path.is_file():
        st.info(f"No file is available for {title.lower()}.")
        return
    rows = preview_jsonl_path(path, limit=limit)
    if rows:
        st.dataframe(rows, use_container_width=True)
    else:
        st.info(f"No preview rows found in `{_path_label(run_dir, path)}`.")


def _render_file_list(run_dir: Path, paths: list[Path], title: str) -> None:
    st.subheader(title)
    if not paths:
        st.info(f"No files found for {title.lower()}.")
        return
    for path in paths:
        st.code(_path_label(run_dir, path))


def _render_training_metrics(run_dir: Path) -> None:
    st.subheader("Training Metrics")
    metric_rows = load_training_metrics(run_dir)
    if not metric_rows:
        st.info("No `logs/training_metrics.jsonl` data is available for this run.")
        return

    st.dataframe(metric_rows, use_container_width=True)
    first_row = metric_rows[0]
    numeric_keys = [
        key
        for key, value in first_row.items()
        if key != "epoch" and isinstance(value, (int, float))
    ]
    if "epoch" in first_row and numeric_keys:
        chart_rows = []
        for row in metric_rows:
            chart_row = {"epoch": row.get("epoch")}
            for key in numeric_keys:
                value = row.get(key)
                if isinstance(value, (int, float)):
                    chart_row[key] = value
            chart_rows.append(chart_row)
        st.line_chart(chart_rows, x="epoch")


def _render_fixed_maps(run_dir: Path) -> None:
    st.subheader("Fixed Maps")
    modes = list_fixed_map_modes(run_dir)
    if not modes:
        st.info("No `fixed-maps/` directory is available for this run.")
        return

    mode = st.selectbox("Fixed map mode", modes, key="fixed_map_mode")
    images = list_fixed_map_images(run_dir, mode)
    json_files = list_fixed_map_json(run_dir, mode)

    if images:
        _render_image_gallery(run_dir, images, f"{mode} images", key=f"{mode}_images")
    else:
        st.info(f"No images found under `fixed-maps/{mode}/`.")

    if json_files:
        labels = [_path_label(run_dir, path) for path in json_files]
        selected_label = st.selectbox(
            f"{mode} JSON artifacts",
            labels,
            key=f"{mode}_json",
        )
        selected_path = json_files[labels.index(selected_label)]
        if selected_path.suffix == ".json":
            payload = load_json_from_path(selected_path)
            st.json(payload if payload is not None else {})
        else:
            rows = preview_jsonl_path(selected_path, limit=50)
            st.dataframe(rows, use_container_width=True)
    else:
        st.info(f"No JSON or JSONL artifacts found under `fixed-maps/{mode}/`.")


def _selected_path_from_labels(
    paths: list[Path],
    selected_label: str | None,
    run_dir: Path,
) -> Path | None:
    if not paths or selected_label is None:
        return None
    labels = {_path_label(run_dir, path): path for path in paths}
    return labels.get(selected_label)


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

    with st.sidebar:
        st.header("Navigation")
        run_id = st.selectbox("Run or report", runs)
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
    region_counts = load_region_counts(run_dir)
    dynamics_files = list_dynamics_files(run_dir)
    snapshot_images = list_snapshot_images(run_dir)
    snapshot_jsonl = list_snapshot_jsonl(run_dir)
    figure_paths = list_figure_paths(run_dir)
    log_files = list_log_files(run_dir)
    model_files = list_model_files(run_dir)
    dynamics_jsonl = [path for path in dynamics_files if path.suffix == ".jsonl"]
    fixed_map_modes = list_fixed_map_modes(run_dir)

    with st.sidebar:
        st.header("Artifacts")
        if dynamics_jsonl:
            dynamics_labels = [_path_label(run_dir, path) for path in dynamics_jsonl]
            selected_dynamics_label = st.selectbox(
                "Dynamics JSONL",
                dynamics_labels,
                key="sidebar_dynamics_jsonl",
            )
        else:
            st.caption("Dynamics JSONL: not available")
            selected_dynamics_label = None

        if snapshot_images:
            snapshot_image_labels = [_path_label(run_dir, path) for path in snapshot_images]
            selected_snapshot_image_label = st.selectbox(
                "Snapshot Image",
                snapshot_image_labels,
                key="sidebar_snapshot_image",
            )
        else:
            st.caption("Snapshot images: not available")
            selected_snapshot_image_label = None

        if snapshot_jsonl:
            snapshot_jsonl_labels = [_path_label(run_dir, path) for path in snapshot_jsonl]
            selected_snapshot_jsonl_label = st.selectbox(
                "Snapshot JSONL",
                snapshot_jsonl_labels,
                key="sidebar_snapshot_jsonl",
            )
        else:
            st.caption("Snapshot JSONL: not available")
            selected_snapshot_jsonl_label = None

        if fixed_map_modes:
            selected_fixed_map_mode = st.selectbox(
                "Fixed Map Mode",
                fixed_map_modes,
                key="sidebar_fixed_map_mode",
            )
        else:
            st.caption("Fixed maps: not available")
            selected_fixed_map_mode = None

    selected_dynamics_path = _selected_path_from_labels(
        dynamics_jsonl,
        selected_dynamics_label,
        run_dir,
    )
    selected_snapshot_image_path = _selected_path_from_labels(
        snapshot_images,
        selected_snapshot_image_label,
        run_dir,
    )
    selected_snapshot_jsonl_path = _selected_path_from_labels(
        snapshot_jsonl,
        selected_snapshot_jsonl_label,
        run_dir,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Figures", len(figure_paths))
    col2.metric("Region Preview", "Available" if has_region_rows(run_dir) else "Missing")
    col3.metric("Manifest", "Available" if manifest is not None else "Missing")
    col4.metric("Snapshot Images", len(snapshot_images))

    overview_tab, dynamics_tab, snapshots_tab, fixed_maps_tab, logs_tab, models_tab = st.tabs(
        ["Overview", "Dynamics", "Snapshots", "Fixed Maps", "Logs", "Models"]
    )

    with overview_tab:
        _render_key_value_metadata("Manifest", manifest)
        _render_key_value_metadata("Config", config)
        _render_key_value_metadata("Summary", summary)
        _render_key_value_metadata("Region Counts", region_counts)
        _render_image_gallery(run_dir, figure_paths, "Figures", key="figures_gallery")

    with dynamics_tab:
        _render_file_list(run_dir, dynamics_files, "Dynamics Files")
        st.caption(
            "Selected dynamics artifact: "
            + (
                f"`{_path_label(run_dir, selected_dynamics_path)}`"
                if selected_dynamics_path is not None
                else "not available"
            )
        )
        if selected_dynamics_path is not None and jsonl_has_region_field(selected_dynamics_path):
            region_options = list_region_values(selected_dynamics_path)
            if not region_options:
                st.info("This dynamics artifact has no detectable region values.")
            else:
                st.subheader("Filter by Region")
                region = st.selectbox(
                    "Region",
                    region_options,
                    key="region_filter",
                )
                region_rows = preview_region_rows_from_path(
                    selected_dynamics_path,
                    region=region,
                    limit=50,
                )
                if region_rows:
                    st.dataframe(region_rows, use_container_width=True)
                else:
                    st.info(f"No rows found for region `{region}` in the selected artifact.")
        elif selected_dynamics_path is not None:
            st.info("The selected dynamics artifact does not contain a `region` field.")
        elif has_region_rows(run_dir):
            st.subheader("Filter by Region")
            st.info("A region-labeled dynamics artifact exists, but no file is currently selectable.")
        else:
            st.info("No `cartography_with_regions.jsonl` preview is available for this run.")
        _render_jsonl_preview_path(
            run_dir,
            selected_dynamics_path,
            "Dynamics JSONL Preview",
        )

    with snapshots_tab:
        _render_image_preview(
            run_dir,
            selected_snapshot_image_path,
            "Snapshot Images",
        )
        _render_jsonl_preview_path(
            run_dir,
            selected_snapshot_jsonl_path,
            "Snapshot JSONL Preview",
        )

    with fixed_maps_tab:
        if selected_fixed_map_mode is None:
            st.subheader("Fixed Maps")
            st.info("No `fixed-maps/` directory is available for this run.")
        else:
            st.subheader("Fixed Maps")
            st.caption(f"Selected fixed-map mode: `{selected_fixed_map_mode}`")
            images = list_fixed_map_images(run_dir, selected_fixed_map_mode)
            json_files = list_fixed_map_json(run_dir, selected_fixed_map_mode)

            if images:
                _render_image_gallery(
                    run_dir,
                    images,
                    f"{selected_fixed_map_mode} images",
                    key=f"{selected_fixed_map_mode}_images",
                )
            else:
                st.info(f"No images found under `fixed-maps/{selected_fixed_map_mode}/`.")

            if json_files:
                labels = [_path_label(run_dir, path) for path in json_files]
                selected_label = st.selectbox(
                    f"{selected_fixed_map_mode} JSON artifacts",
                    labels,
                    key=f"{selected_fixed_map_mode}_json",
                )
                selected_path = _selected_path_from_labels(json_files, selected_label, run_dir)
                if selected_path is None:
                    st.info("The selected fixed-map artifact is not available.")
                elif selected_path.suffix == ".json":
                    payload = load_json_from_path(selected_path)
                    if payload is None:
                        st.info("This JSON artifact could not be loaded.")
                    else:
                        st.json(payload)
                else:
                    rows = preview_jsonl_path(selected_path, limit=50)
                    if rows:
                        st.dataframe(rows, use_container_width=True)
                    else:
                        st.info("This JSONL artifact has no previewable rows.")
            else:
                st.info(f"No JSON or JSONL artifacts found under `fixed-maps/{selected_fixed_map_mode}/`.")

    with logs_tab:
        _render_file_list(run_dir, log_files, "Log Files")
        _render_training_metrics(run_dir)

    with models_tab:
        _render_file_list(run_dir, model_files, "Model Files")


if __name__ == "__main__":
    main()
