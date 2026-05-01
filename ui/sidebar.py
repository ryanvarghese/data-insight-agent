"""
ui/sidebar.py — Sidebar layout and controls.

Orchestrates the complete user flow:
  Upload → Clean → Schema → Anomalies → Trends
  → Run Analysis → Display Progress → Show Summary

Calls into core/ modules for all business logic.
Stores nothing itself — reads and writes only
through st.session_state.

Sidebar sections in display order:
  1. File uploader
  2. Run Analysis button (primary action — first
     visible element after upload so users see it
     immediately without scrolling)
  3. File summary with color-coded column badges
  4. Color legend explaining badge colors
  5. Anomaly alert (red if found, green if none)
  6. Round-by-round reasoning expanders
  7. Analysis completion summary
"""

import html
from typing import List

import pandas as pd
import streamlit as st

from core.analysis import run_analysis_loop
from core.anomalies import detect_anomalies
from core.loader import load_and_clean
from core.schema import extract_schema
from core.trends import precompute_trends


def _format_badges(columns: List[str], badge_class: str) -> str:
    """Render compact badge list for sidebar summary."""
    if not columns:
        return '<span class="sidebar-badge {}">None</span>'.format(badge_class)
    display_cols = columns if len(columns) <= 5 else columns[:4]
    badges = "".join(
        f'<span class="sidebar-badge {badge_class}">{html.escape(str(col), quote=False)}</span>'
        for col in display_cols
    )
    remaining = len(columns) - len(display_cols)
    if len(columns) > 5 and remaining > 0:
        badges += f'<span class="sidebar-badge {badge_class}">+{remaining} more</span>'
    return badges


def _format_anomaly_value(value) -> str:
    """Format anomaly values for compact display."""
    if pd.isna(value):
        return "NaN"
    if isinstance(value, (int, float)):
        return f"{value:,.2f}"
    return str(value)


def _sanitize_text(text: str) -> str:
    """Escape text for safe HTML rendering in markdown cards."""
    if not text:
        return ""
    return html.escape(str(text), quote=False).replace("\n", "<br>")


def _handle_file_upload(uploaded_file) -> None:
    """Process uploaded file, then populate schema and anomaly state."""
    # Guard clause: only reprocess if the filename
    # changed. Without this, the entire cleaning
    # pipeline re-runs on every Streamlit rerun
    # triggered by any UI interaction (button click,
    # expander open etc.) which would be very slow.
    if uploaded_file is None:
        return
    try:
        load_and_clean(uploaded_file)
        schema = extract_schema(st.session_state["df"])
        anomalies = detect_anomalies(st.session_state["df"], schema)
        st.session_state["schema"] = schema
        st.session_state["anomalies"] = anomalies
        st.session_state["uploaded_file_name"] = uploaded_file.name
        st.session_state.pop("analysis_meta", None)
        st.session_state.pop("round_history", None)
    except Exception as exc:
        st.error(f"Unable to load file: {exc}")
        st.session_state.pop("df", None)
        st.session_state.pop("basic_stats", None)
        st.session_state.pop("identifier_cols", None)
        st.session_state.pop("schema", None)
        st.session_state.pop("anomalies", None)
        st.session_state.pop("uploaded_file_name", None)
        st.session_state.pop("analysis_meta", None)
        st.session_state.pop("round_history", None)


def _render_round_progress(round_history_placeholder) -> None:
    """Render round-by-round reasoning transparency section."""
    # Each completed round shows as a collapsed expander
    # containing Claude's REASONING text — the explanation
    # of WHY it chose that computation at that point.
    # This makes the AI's analytical process transparent:
    # users see genuine progressive reasoning, not a
    # black box producing outputs from nowhere.
    #
    # In-progress round shown as plain text (no expander)
    # because there's no reasoning to show yet.
    round_history = st.session_state.get("round_history", [])
    with round_history_placeholder.container():
        if not round_history:
            return
        for item in round_history:
            round_no = item.get("round", "?")
            status = item.get("status", "")
            reasoning = _sanitize_text(item.get("reasoning", ""))
            if status == "in_progress":
                st.markdown(f"🔄 Round {round_no} in progress...")
            else:
                with st.expander(f"Round {round_no} ✅", expanded=False):
                    st.markdown(
                        f'<div style="font-size:12px;color:#8b949e;">{reasoning}</div>',
                        unsafe_allow_html=True,
                    )


def _render_file_summary() -> None:
    """Render compact file summary card with typed column badges."""
    if "schema" not in st.session_state:
        return

    schema = st.session_state["schema"]
    file_name = st.session_state.get("uploaded_file_name", "Unknown")
    # Color coding maps to analytical role:
    # Blue   (#1f3a6e) = numeric metrics — what you measure
    # Green  (#196c2e) = categorical dimensions — how you group
    # Orange (#7d4e00) = datetime columns — when things happen
    #
    # First 4 columns shown per type, then "+N more" badge
    # to keep the sidebar compact on wide datasets.
    st.markdown(
        f"""
        <div class="card">
            <div class="summary-file-name">{html.escape(str(file_name), quote=False)}</div>
            <div class="summary-line">{schema["shape"]["rows"]} rows • {schema["shape"]["columns"]} columns</div>
            <div class="badge-row">{_format_badges(schema["numeric_columns"], "badge-num")}</div>
            <div class="badge-row">{_format_badges(schema["categorical_columns"], "badge-cat")}</div>
            <div class="badge-row">{_format_badges(schema["datetime_columns"], "badge-dt")}</div>
            <div class="badge-legend-row">
                <span class="badge-legend-item"><span class="badge-legend-dot legend-num"></span>Numeric</span>
                <span class="badge-legend-item"><span class="badge-legend-dot legend-cat"></span>Categorical</span>
                <span class="badge-legend-item"><span class="badge-legend-dot legend-dt"></span>Datetime</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    datetime_cols = st.session_state.get("schema", {}).get("datetime_columns", [])
    if not datetime_cols:
        st.warning(
            "⚠️ No datetime columns detected. "
            "Trend analysis will be limited. "
            "Check that date columns are in a standard "
            "format (DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD)."
        )


def _render_anomaly_section() -> None:
    """Render anomaly count alert and optional row-level details."""
    # Red alert for anomalies, green for clean data.
    # The visual distinction makes data quality status
    # immediately obvious without reading the number.
    st.markdown("### Analysis Controls")
    anomalies = st.session_state.get("anomalies", [])
    if anomalies:
        st.markdown(
            f'<div class="alert-badge alert-danger">⚠️ {len(anomalies)} anomalies detected</div>',
            unsafe_allow_html=True,
        )
        with st.expander("View details", expanded=False):
            for anomaly in anomalies:
                row_index = anomaly.get("row_index", "Unknown")
                if "column" in anomaly:
                    column = anomaly["column"]
                    value = _format_anomaly_value(anomaly.get("value"))
                    z_score = float(anomaly.get("z_score", 0.0))
                    direction = "above" if z_score > 0 else "below"
                    st.write(
                        f"Row {row_index} — {column}: {value} ({abs(z_score):.1f}σ {direction} mean)"
                    )
                else:
                    st.write(
                        f"Row {row_index} — Multivariate numeric pattern anomaly (IsolationForest)"
                    )
    elif "schema" in st.session_state:
        st.markdown(
            '<div class="alert-badge alert-success">✅ No anomalies detected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Upload a file to run anomaly checks.")


def _render_completion_summary(analysis_status_placeholder) -> None:
    """Render compact post-analysis summary card."""
    # Compact summary replaces the round-by-round
    # progress once analysis finishes. Shows counts
    # not details to keep the sidebar clean after
    # analysis completes.
    if "analysis_meta" not in st.session_state:
        return
    meta = st.session_state["analysis_meta"]
    analysis_status_placeholder.markdown(
        f"""
        <div class="card">
            <strong>✅ Analysis complete</strong><br>
            {meta.get("insights_count", 0)} insights generated<br>
            {meta.get("recommendations_count", 0)} recommendations<br>
            Analysed in {meta.get("rounds_used", 0)} rounds
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Render the full sidebar and orchestrate upload/analysis actions."""
    with st.sidebar:
        st.title("Data-to-Insight Agent")
        uploaded_file = st.file_uploader("Upload data file", type=["csv", "xlsx"])
        _handle_file_upload(uploaded_file)

        can_run_analysis = "df" in st.session_state and "schema" in st.session_state
        analysis_status_placeholder = st.empty()
        round_history_placeholder = st.empty()

        if st.button("Run Analysis", use_container_width=True, disabled=not can_run_analysis):
            # Guard: ensure pipeline has fully completed
            # before allowing analysis to run.
            # If df or schema is missing, rerun the pipeline
            # first then stop — user clicks again to analyze.
            if st.session_state.get("df") is None or \
               st.session_state.get("schema") is None or \
               st.session_state.get("basic_stats") is None:

                if uploaded_file:
                    with st.spinner("Preparing data first..."):
                        df = load_and_clean(uploaded_file)
                        st.session_state["schema"] = extract_schema(df)
                        st.session_state["anomalies"] = detect_anomalies(
                            df,
                            st.session_state["schema"],
                        )
                        precompute_trends(df, st.session_state["schema"])

                st.warning(
                    "Data is ready. Click Run Analysis again."
                )
                st.stop()

            progress_state = {"completed": 0, "in_progress": 1}
            st.session_state["round_history"] = []

            def _render_progress() -> None:
                lines = ['<div class="card"><strong>Analyzing your data...</strong><br><br>']
                for idx in range(1, 6):
                    if idx <= progress_state["completed"]:
                        lines.append(f"✅ Round {idx} complete<br>")
                    elif idx == progress_state["in_progress"]:
                        lines.append(f"🔄 Round {idx} in progress...<br>")
                    else:
                        break
                lines.append("</div>")
                analysis_status_placeholder.markdown("".join(lines), unsafe_allow_html=True)

            def _progress_callback(round_number: int, state: str) -> None:
                if state == "in_progress":
                    progress_state["in_progress"] = round_number
                elif state == "complete":
                    progress_state["completed"] = max(progress_state["completed"], round_number)
                    progress_state["in_progress"] = round_number + 1
                _render_progress()
                _render_round_progress(round_history_placeholder)

            with st.spinner("Analyzing your data..."):
                _render_progress()
                result = run_analysis_loop(
                    st.session_state["df"],
                    st.session_state["schema"],
                    st.session_state.get("anomalies", []),
                    progress_callback=_progress_callback,
                )
                st.session_state["analysis_result"] = result
                st.session_state["analysis_meta"] = {
                    "insights_count": len(result.get("insights", [])),
                    "recommendations_count": len(result.get("recommendations", [])),
                    "rounds_used": int(result.get("rounds_used", 0)),
                }

        _render_completion_summary(analysis_status_placeholder)
        _render_round_progress(round_history_placeholder)
        _render_file_summary()
        _render_anomaly_section()
