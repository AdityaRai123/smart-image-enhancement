"""
Smart Multimedia Image Enhancement Assistant - Streamlit application.

Run with:  streamlit run app.py

The page follows the five-stage architecture of the Review 1 report:
    1. Image Ingestion  2. Quality Analysis  3. Decision Engine
    4. Enhancement Execution  5. Evaluation & Presentation
All logic lives in ``src/``; this file only wires the UI together.
"""

from __future__ import annotations

import logging
import time

import streamlit as st

from config import APP
from src.pipeline import MODE_ADAPTIVE
from src.ui import components as ui
from src.ui.session import STAGES, get_sr_backend, load_input, process, run_key
from src.ui.sidebar import render_sidebar
from src.utils.image_utils import ImageLoadError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

st.set_page_config(
    page_title="Smart Image Enhancement Assistant",
    page_icon="\N{FRAME WITH PICTURE}",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    ui.render_header()
    state = render_sidebar()
    tracker = st.empty()
    statuses = ["pending"] * len(STAGES)
    tracker.html(ui.stage_tracker_html(statuses))

    # ---- Stage 1: ingestion ----------------------------------------------------------
    start = time.perf_counter()
    try:
        loaded = load_input(state)
    except ImageLoadError as exc:
        statuses[0] = "error"
        tracker.html(ui.stage_tracker_html(statuses))
        st.error(f"Could not load the image: {exc}")
        return
    if loaded is None:
        st.info(
            "Upload an image in the sidebar (JPG, PNG or WebP), or switch the image source to "
            "'Built-in sample + synthetic degradation' for a demo with a ground-truth reference."
        )
        return
    ingestion_ms = (time.perf_counter() - start) * 1000.0
    statuses[0] = "done"
    tracker.html(ui.stage_tracker_html(statuses, {"Image Ingestion": ingestion_ms}))

    sr = get_sr_backend(state)
    if state.sr_enabled:
        (st.sidebar.success if sr.ai_ready else st.sidebar.warning)(sr.message)

    # ---- Stages 2-5 (cached per image + settings) -------------------------------------
    key = run_key(loaded, state, sr)
    run = st.session_state.get("run") if st.session_state.get("run_key") == key else None
    if run is None:
        def on_stage(index: int, status: str) -> None:
            statuses[index] = status
            tracker.html(ui.stage_tracker_html(statuses))

        try:
            with st.status("Processing image...", expanded=False) as status_box:
                run = process(loaded, state, sr, on_stage)
                status_box.update(label="Processing complete", state="complete")
        except Exception as exc:  # never let one image crash the app
            logger.exception("Processing failed")
            statuses = [s if s == "done" else "error" for s in statuses]
            tracker.html(ui.stage_tracker_html(statuses))
            st.error(f"Processing failed: {exc}")
            return
        st.session_state["run"], st.session_state["run_key"] = run, key

    stage_times = {"Image Ingestion": ingestion_ms, **run.stage_ms}
    tracker.html(ui.stage_tracker_html(["done"] * len(STAGES), stage_times))

    adaptive = run.comparison.results[MODE_ADAPTIVE]
    decision = adaptive.decision
    shown = run.comparison.results[state.display_mode]
    image = loaded.ingested.image
    for error in run.errors:
        st.warning(error)

    ui.render_ingestion(loaded.ingested, loaded.reference_label)
    st.divider()
    ui.render_quality(run.quality, decision, image)
    st.divider()
    ui.render_decision(decision, state.thresholds, state.thresholds_modified)
    st.divider()
    ui.render_execution(adaptive, state.keep_intermediates)
    st.divider()
    ui.render_before_after(image, shown, shown.total_ms)
    ui.render_histograms(image, shown)
    ui.render_metrics(run.comparison, state.display_mode, run.quality)
    ui.render_baselines(run.comparison, image)
    st.divider()
    ui.render_he_demo(image)
    ui.render_technical_details(decision, state.thresholds, sr.message, image, stage_times)
    ui.render_background_removal(image)
    st.caption(f"{APP.title} - BITE314L Multimedia Systems course project. All values are computed live.")


main()
