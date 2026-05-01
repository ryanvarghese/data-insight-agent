"""
ui/insights_tab.py — Insights tab rendering.

Displays three sections from the analysis result:
  KEY INSIGHTS        → what the AI found
  RECOMMENDED ACTIONS → what to do about it
  ANOMALY FLAGS       → statistically unusual rows

Each card has a collapsed "Why?" expander showing
Claude's reasoning for that specific finding —
making the AI's logic transparent and auditable.

Dollar signs ($) are escaped before rendering to
prevent Streamlit from interpreting them as LaTeX
math delimiters which corrupts number formatting.
"""

import html

import pandas as pd
import streamlit as st


def _sanitize_answer_text(text):
    """Escape answer text for safe card rendering."""
    if not text:
        return ""
    text = str(text)
    text = html.escape(text, quote=False)
    text = text.replace("\n", "<br>")
    return text


def _normalize_reasoned_items(items):
    """Normalize item list into dicts with text/why keys."""
    normalized = []
    if not isinstance(items, list):
        return normalized
    for item in items:
        if isinstance(item, dict):
            text_value = str(item.get("text", "")).strip()
            why_value = item.get("why")
            why_value = str(why_value).strip() if why_value is not None else None
            normalized.append({"text": text_value, "why": why_value if why_value else None})
        else:
            normalized.append({"text": str(item), "why": None})
    return normalized


def _render_metric_row(schema: dict, anomalies: list, analysis_result) -> None:
    """Render the top metrics row."""
    # Shown even before analysis runs — gives users
    # immediate quantitative context about their dataset
    # (row count, anomaly count) as soon as they upload.
    # Insights Generated shows 0 until analysis completes.
    total_rows = schema.get("shape", {}).get("rows", 0) if isinstance(schema, dict) else 0
    insights_count = 0
    if isinstance(analysis_result, dict):
        insights_count = len(analysis_result.get("insights", []))

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Total Rows in dataset", int(total_rows))
    metric_col_2.metric("Anomalies Detected", len(anomalies))
    metric_col_3.metric("Insights Generated", int(insights_count))


def _render_card(icon: str, css_class: str, text: str) -> None:
    """Render one styled card line."""
    # Escape $ before rendering. Streamlit's markdown
    # parser treats $text$ as LaTeX math expressions.
    # Without escaping, "$456" renders as italic math
    # font and "$22,638" becomes completely garbled.
    st.markdown(
        f'<div class="{css_class}">{icon} {_sanitize_answer_text(text)}</div>',
        unsafe_allow_html=True,
    )


def _render_insights_section(insights):
    """Render key insights cards with optional why-expander."""
    # Filter out protocol artifacts before rendering.
    # If JSON parsing partially failed, Claude's raw
    # REASONING: and REQUEST: lines might appear in
    # the insights list. These prefixes identify and
    # remove them so they never reach the UI.
    filtered_insights = [
        i for i in insights
        if not any(
            i.get("text", "").startswith(prefix)
            for prefix in [
                "REASONING:", "REQUEST:",
                "I need to complete",
                "I need to", "Let me",
                "ANALYSIS_COMPLETE",
            ]
        )
    ]
    st.markdown("### Key Insights")
    for insight in filtered_insights:
        _render_card("💡", "insight-card", insight.get("text", ""))
        insight_why = insight.get("why")
        if insight_why:
            with st.expander("💬 Why this insight?", expanded=False):
                st.markdown(
                    f'<div style="font-size:12px;color:#8b949e;">{_sanitize_answer_text(insight_why)}</div>',
                    unsafe_allow_html=True,
                )


def _render_recommendations_section(recommendations):
    """Render recommendations cards with optional why-expander."""
    st.markdown("### Recommended Actions")
    for recommendation in recommendations:
        _render_card("🎯", "rec-card", recommendation.get("text", ""))
        rec_why = recommendation.get("why")
        if rec_why:
            with st.expander("💬 Why this recommendation?", expanded=False):
                st.markdown(
                    f'<div style="font-size:12px;color:#8b949e;">{_sanitize_answer_text(rec_why)}</div>',
                    unsafe_allow_html=True,
                )


def _render_anomaly_section(anomaly_commentary, anomalies):
    """Render anomaly commentary cards and raw anomaly table."""
    if not anomaly_commentary:
        return
    st.markdown("### Anomaly Flags")
    for item in anomaly_commentary:
        _render_card("⚠️", "anomaly-card", item.get("text", ""))
        anomaly_why = item.get("why")
        if anomaly_why:
            with st.expander("💬 Why was this flagged?", expanded=False):
                st.markdown(
                    f'<div style="font-size:12px;color:#8b949e;">{_sanitize_answer_text(anomaly_why)}</div>',
                    unsafe_allow_html=True,
                )
    with st.expander("View anomaly details"):
        st.dataframe(pd.DataFrame(anomalies), use_container_width=True)


def render_insights_tab() -> None:
    """Render the complete insights tab."""
    schema = st.session_state.get("schema", {})
    anomalies = st.session_state.get("anomalies", [])
    analysis_result = st.session_state.get("analysis_result")

    _render_metric_row(schema, anomalies, analysis_result)

    if isinstance(analysis_result, dict):
        insights = _normalize_reasoned_items(analysis_result.get("insights", []))
        recommendations = _normalize_reasoned_items(analysis_result.get("recommendations", []))
        anomaly_commentary = _normalize_reasoned_items(analysis_result.get("anomaly_commentary", []))
        _render_insights_section(insights)
        _render_recommendations_section(recommendations)
        _render_anomaly_section(anomaly_commentary, anomalies)
    else:
        if "schema" in st.session_state:
            st.info("File uploaded. Click Run Analysis to generate insights.")
        else:
            st.info("Upload a file and click Run Analysis to generate insights.")
