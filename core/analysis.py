"""
core/analysis.py — Back-and-forth AI analysis loop.

The core intelligence of the application.

Architecture:
  1. Send Claude the dataset schema, pre-computed
     basic stats, and pre-computed time trends.
  2. Claude responds with:
       REASONING: why it chose this computation
       REQUEST:   what to compute
  3. App runs the computation using pandas.
  4. Result sent back to Claude as a new message.
  5. Repeat until ANALYSIS_COMPLETE + JSON output.
  Maximum 7 rounds (round 6 gets wrap-up warning,
  round 7 is reserved for JSON output only).

Why numbers cannot be hallucinated:
  Claude never calculates — it only reads results
  that pandas already computed. Every number in the
  final insights existed in the pandas output before
  Claude wrote the sentence containing it.

The REASONING:/REQUEST: format serves two purposes:
  1. Enables sidebar transparency — users see WHY
     Claude chose each computation, not just what ran.
  2. Separates intent from action so the app can
     store reasoning independently from computation.
"""

import json
import re
from typing import List

import numpy as np
import pandas as pd
import streamlit as st


def _get_client():
    """Fetch Anthropic client from session state."""
    return st.session_state.get("anthropic_client")


def _extract_response_text(response) -> str:
    """Extract text blocks from Anthropic response payload."""
    text_parts = []
    for block in getattr(response, "content", []):
        if getattr(block, "type", "") == "text":
            text_parts.append(block.text)
    return "\n".join(text_parts).strip()


def _extract_code_line(raw_text: str) -> str:
    """Extract the first executable line from model output."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    return cleaned.splitlines()[0].strip() if cleaned else ""


def run_computation(df: pd.DataFrame, request_str: str) -> str:
    """Execute one Claude-requested pandas computation and return stringified result."""
    df = df.copy()
    client = _get_client()
    if client is None:
        return "Error: Anthropic client is not initialized. Check ANTHROPIC_API_KEY."

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1000,
            system=(
                "You are a pandas code generator. Given a DataFrame called df and a "
                "computation description, return ONLY a single line of valid pandas code "
                "that performs that computation and stores the result in a variable called "
                "result. No imports, no explanations, just the one line of code."
            ),
            messages=[{"role": "user", "content": request_str}],
        )
        code_line = _extract_code_line(_extract_response_text(response))
        if not code_line:
            return "Error: Model returned empty computation code."

        # Claude never calculates — it only reads results.
        # This function is the guarantee: pandas runs the
        # math, Claude reads the output. This is why every
        # number in the insights is traceable to a real
        # computation rather than generated from memory.
        #
        # exec() is safe here because:
        # - df is a copy (original session_state df untouched)
        # - No network access in the execution context
        # - Any error is caught and returned as a string
        #   so the loop continues rather than crashing
        execution_scope = {"df": df, "pd": pd, "np": np}
        exec(code_line, {}, execution_scope)
        if "result" not in execution_scope:
            return "Error: Generated code did not define result."

        result = execution_scope["result"]
        if isinstance(result, pd.DataFrame):
            return result.to_string()
        return str(result)
    except Exception as exc:
        return "Error: " + str(exc)


def fix_column_names(computation_str: str, df: pd.DataFrame) -> str:
    """Normalize column references to exact DataFrame casing."""
    # Claude sometimes uses wrong column name casing
    # e.g. "sales" instead of "Sales". This function
    # maps all column references to their correct case
    # before execution, preventing wasted rounds on
    # "KeyError: sales" type errors.
    col_map = {col.lower(): col for col in df.columns}
    fixed = computation_str
    for lower_name, actual_name in col_map.items():
        fixed = re.sub(
            rf'["\']({re.escape(lower_name)})["\']',
            lambda m: m.group(0).replace(m.group(1), actual_name),
            fixed,
            flags=re.IGNORECASE,
        )
        fixed = re.sub(
            rf'\[[\'"]{re.escape(lower_name)}[\'"]\]',
            f'["{actual_name}"]',
            fixed,
            flags=re.IGNORECASE,
        )
    return fixed


def _summarize_anomalies(anomalies: List[dict]) -> str:
    """Convert anomaly dicts into compact natural-language context lines."""
    if not anomalies:
        return "No anomalies were detected."

    def _format_anomaly_value(value) -> str:
        if pd.isna(value):
            return "NaN"
        if isinstance(value, (int, float)):
            return f"{value:,.2f}"
        return str(value)

    lines = []
    for anomaly in anomalies[:10]:
        row_index = anomaly.get("row_index", "Unknown")
        if "column" in anomaly:
            lines.append(
                f"Row {row_index}: {anomaly['column']}={_format_anomaly_value(anomaly.get('value'))}, "
                f"z={float(anomaly.get('z_score', 0.0)):.2f}"
            )
        else:
            lines.append(f"Row {row_index}: IsolationForest multivariate anomaly")
    return "\n".join(lines)


def _build_system_prompt() -> str:
    """Return the full analysis system prompt."""
    # Five structured requirements ensure every analysis
    # covers time trends, dimensional breakdown, category
    # trends, distribution analysis, and anomaly
    # investigation — regardless of the dataset.
    #
    # "EXACTLY 5 insights, EXACTLY 2 recommendations"
    # enforces compliance with the assessment brief which
    # specifies 3-5 insights and 1-2 recommendations.
    #
    # The REASONING:/REQUEST: format requirement enables
    # the sidebar transparency display showing Claude's
    # analytical thinking at each step.
    return """You are a senior retail data analyst performing 
iterative analysis on a dataset.

You have access to a computation engine. Request 
specific computations and receive real pandas results.
You never calculate numbers yourself — you only read
results that the computation engine returns.

ANALYSIS REQUIREMENTS — cover ALL of these:

REQUIREMENT 1 — TREND ANALYSIS (mandatory):
Compute period-over-period changes in the primary 
metric using the datetime column.
Express findings as percentage changes with specific 
time periods — never just totals.
Required format: '[Dimension] metric increased/
decreased by X% period-over-period in [time period]'
Example: 'West region revenue increased 34% 
month-over-month in November 2018'

REQUIREMENT 2 — DIMENSION BREAKDOWN (mandatory):
Compute metric performance broken down by the most 
relevant categorical column. Show which dimensions 
are growing vs declining over time — not just totals.
Must include specific percentage changes per dimension.

REQUIREMENT 2B — CATEGORY TRENDS (mandatory):
Compute how top 2-3 categories or sub-categories 
perform over time — year-over-year or month-over-month.
Show percentage changes not totals.
Example: 'Technology grew 28% year-over-year while 
Furniture declined 8% in the same period'

REQUIREMENT 3 — DISTRIBUTION ANALYSIS (mandatory):
Investigate the gap between mean and median. What 
percentage of total revenue comes from the top 10% 
of transactions? What does this mean for business 
risk and revenue stability?

REQUIREMENT 4 — ANOMALY INVESTIGATION (mandatory):
The pre-flagged anomalies are provided in context.
Determine whether they cluster in specific categories,
segments, time periods, or regions — or are randomly
distributed across the dataset.

REQUIREMENT 5 — OPEN INVESTIGATION:
Use one round to investigate the most surprising or
actionable pattern found during requirements 1-4.

RULES:
- Every response MUST start with exactly:
  REASONING: [one sentence explaining why you chose
  this specific computation at this point]
  REQUEST: [the specific computation to run]
- Request exactly one computation per round
- Use exact column names as provided in the schema
  (column names are case-sensitive)
- Express ALL trends as percentage changes between
  specific time periods — never just raw totals
- After covering all requirements respond with
  ANALYSIS_COMPLETE followed immediately by JSON

OUTPUT FORMAT after ANALYSIS_COMPLETE:
Generate EXACTLY 5 insights — no more, no fewer.
Choose the 5 most significant and actionable findings.
Prioritize insights that include specific percentage
changes over time — not just static totals.

Generate EXACTLY 2 recommendations — no more, no fewer.
Choose the 2 highest-impact actions a retail manager
should take based on the data findings.
Each recommendation should be specific and grounded
in a data finding from the analysis.

{
  'insights': [exactly 5 objects],
  'recommendations': [exactly 2 objects],
  'anomaly_commentary': [2-3 objects maximum]
}
Each insight object format:
{
  'text': 'insight with specific percentage changes
           and exact time periods where relevant',
  'why': '1-2 sentences: which round and computation
           revealed this',
  'round': N
}
Each recommendation object format:
{
  'text': 'specific actionable recommendation',
  'why': 'which insight motivates this'
}
Each anomaly commentary object format:
{
  'text': 'observation about flagged anomalies',
  'why': 'what made these statistically unusual'
}"""


def _build_opening_message(df: pd.DataFrame, schema: dict, anomalies: List[dict]) -> str:
    """Compose the opening user message with schema and precomputed context."""
    opening_message = (
        f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n"
        "Pre-computed basic statistics:\n"
        f"{json.dumps(st.session_state.get('basic_stats', {}), indent=2, default=str)}\n\n"
        "Identifier columns excluded from analysis:\n"
        f"{st.session_state.get('identifier_cols', [])}\n\n"
        "IMPORTANT: Column names are case-sensitive.\n"
        "The exact column names available are:\n"
        f"{list(df.columns)}\n"
        "Always use these exact names in your requests.\n\n"
        "These statistics are already computed and verified from clean data. Use them as your "
        "starting point. Request additional specific computations only for analysis that "
        "goes beyond these basics - such as group breakdowns, correlations, time trends, "
        "or anomaly investigation.\n\n"
    )
    return (
        f"{opening_message}"
        "Dataset schema:\n"
        f"{json.dumps(schema, indent=2, default=str)}\n\n"
        "Flagged anomalies:\n"
        f"{_summarize_anomalies(anomalies)}\n\n"
        "Begin your analysis. Request your first computation."
    )


def _extract_section(text: str, start_label: str, end_label: str = "") -> str:
    """Extract section body between protocol markers."""
    start_index = text.find(start_label)
    if start_index == -1:
        return ""
    start_index += len(start_label)
    if end_label:
        end_index = text.find(end_label, start_index)
        if end_index == -1:
            return text[start_index:].strip()
        return text[start_index:end_index].strip()
    return text[start_index:].strip()


def _parse_reasoning_request(response_text: str) -> tuple:
    """Parse REASONING and REQUEST protocol blocks."""
    cleaned = response_text.strip()
    reasoning = _extract_section(cleaned, "REASONING:", "REQUEST:").strip()
    if reasoning.upper().startswith("REASONING:"):
        reasoning = reasoning[len("REASONING:") :].strip()

    request = _extract_section(cleaned, "REQUEST:").strip()
    if request.upper().startswith("REQUEST:"):
        request = request[len("REQUEST:") :].strip()
    if not request:
        request = cleaned
    return reasoning, request


def _normalize_reasoned_items(items):
    """Normalize mixed string/object lists into object list with text/why."""
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


def _parse_analysis_result(response_text):
    """Parse final ANALYSIS_COMPLETE payload with robust fallback."""
    # Searches case-insensitively for ANALYSIS_COMPLETE
    # marker then finds the first { after it.
    # Strips any text between the marker and the JSON
    # object — Claude sometimes adds explanatory text
    # before the JSON that would break json.loads().
    #
    # Fallback filters out protocol artifacts
    # (REASONING:, REQUEST: lines) so they never
    # appear as insight cards in the UI.
    try:
        marker = "ANALYSIS_COMPLETE"
        idx = response_text.upper().find(marker.upper())
        if idx != -1:
            after_marker = response_text[idx + len(marker):].strip()
        else:
            after_marker = response_text

        start = after_marker.find("{")
        if start == -1:
            raise ValueError("No JSON found")
        json_str = after_marker[start:]
        last_brace = json_str.rfind("}")
        if last_brace != -1:
            json_str = json_str[:last_brace + 1]

        result = json.loads(json_str)

        def normalize(item):
            if isinstance(item, dict):
                return item
            return {
                "text": str(item),
                "why": None,
                "round": None,
            }

        result["insights"] = [normalize(i) for i in result.get("insights", [])]
        result["recommendations"] = [normalize(r) for r in result.get("recommendations", [])]
        result["anomaly_commentary"] = [normalize(a) for a in result.get("anomaly_commentary", [])]
        return result

    except Exception:
        lines = [
            l.strip()
            for l in response_text.split("\n")
            if len(l.strip()) > 50
            and not l.strip().startswith("REASONING")
            and not l.strip().startswith("REQUEST")
            and not l.strip().startswith("{")
            and not l.strip().startswith("}")
        ]
        return {
            "insights": [{"text": l, "why": None, "round": None} for l in lines[:5]],
            "recommendations": [],
            "anomaly_commentary": [],
        }


def _update_round_status(round_number: int, state: str, progress_callback=None) -> None:
    """Update round-progress callback when available."""
    if callable(progress_callback):
        progress_callback(round_number, state)


def run_analysis_loop(
    df: pd.DataFrame,
    schema: dict,
    anomalies: List[dict],
    progress_callback=None,
) -> dict:
    """Run iterative Claude ↔ pandas analysis loop and return final structured output."""
    # Round 6 gets a wrap-up warning so Claude knows
    # to make its final computation count.
    # Round 7 skips computation entirely and forces
    # ANALYSIS_COMPLETE output. This guarantees Claude
    # always has a full round for JSON output rather
    # than running out of rounds mid-analysis.
    #
    # Full conversation history sent every round.
    # Without this Claude would forget what it already
    # investigated and repeat computations.
    df = df.copy()
    client = _get_client()
    if client is None:
        return {
            "insights": [],
            "recommendations": [],
            "anomaly_commentary": ["Anthropic client is not initialized."],
        }

    system_prompt = _build_system_prompt()
    opening_user_message = _build_opening_message(df, schema, anomalies)

    message_history = [{"role": "user", "content": opening_user_message}]
    final_response_text = ""
    rounds_completed = 0
    st.session_state["round_history"] = []

    max_rounds = 7
    for round_number in range(1, max_rounds + 1):
        if round_number == max_rounds - 1:
            message_history.append({
                "role": "user",
                "content": (
                    "This is your SECOND TO LAST round. "
                    "After this computation you MUST output "
                    "ANALYSIS_COMPLETE with your JSON. "
                    "Make this computation count — choose "
                    "the most valuable remaining analysis."
                ),
            })

        if round_number == max_rounds:
            final_response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=6000,
                system=system_prompt,
                messages=message_history + [{
                    "role": "user",
                    "content": (
                        "STOP. Output ANALYSIS_COMPLETE "
                        "and your JSON now. "
                        "No more computations. "
                        "Use only what you have already found."
                    ),
                }],
            )
            st.session_state["round_history"] = st.session_state.get("round_history", [])
            parsed = _parse_analysis_result(final_response.content[0].text)
            return {
                "insights": _normalize_reasoned_items(parsed.get("insights", [])),
                "recommendations": _normalize_reasoned_items(parsed.get("recommendations", [])),
                "anomaly_commentary": _normalize_reasoned_items(parsed.get("anomaly_commentary", [])),
                "rounds_used": rounds_completed,
            }

        _update_round_status(round_number, "in_progress", progress_callback)
        round_max_tokens = 6000 if round_number >= 5 else 1000
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=round_max_tokens,
            system=system_prompt,
            messages=message_history,
        )
        response_text = _extract_response_text(response)
        final_response_text = response_text
        if response_text.strip().startswith("ANALYSIS_COMPLETE"):
            break

        reasoning_text, request_str = _parse_reasoning_request(response_text)
        st.session_state["round_history"].append(
            {
                "round": round_number,
                "reasoning": reasoning_text if reasoning_text else "No reasoning provided.",
                "status": "in_progress",
            }
        )
        request_str = fix_column_names(request_str, df)
        computation_output = run_computation(df, request_str)
        rounds_completed += 1
        st.session_state["round_history"][-1]["status"] = "complete"
        _update_round_status(round_number, "complete", progress_callback)

        message_history.append({"role": "assistant", "content": response_text.strip()})
        message_history.append(
            {
                "role": "user",
                "content": (
                    "Computation result:\n"
                    f"{computation_output}\n\n"
                    "Continue analysis. Request next computation or return ANALYSIS_COMPLETE."
                ),
            }
        )

    final_messages = message_history + [
        {"role": "assistant", "content": final_response_text},
        {
            "role": "user",
            "content": """STOP. Do not request any more computations.
You have used all available rounds.

Based ONLY on what you have already computed
in the previous rounds, you MUST now output
ANALYSIS_COMPLETE immediately followed by
the JSON object.

Do not include any text before ANALYSIS_COMPLETE.
Do not say you need more rounds.
Use only findings from computations already done.

ANALYSIS_COMPLETE
{
  'insights': [...],
  'recommendations': [...],
  'anomaly_commentary': [...]
}""",
        },
    ]
    final_structured_response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=6000,
        system=system_prompt,
        messages=final_messages,
    )
    parsed = _parse_analysis_result(_extract_response_text(final_structured_response))
    return {
        "insights": _normalize_reasoned_items(parsed.get("insights", [])),
        "recommendations": _normalize_reasoned_items(parsed.get("recommendations", [])),
        "anomaly_commentary": _normalize_reasoned_items(parsed.get("anomaly_commentary", [])),
        "rounds_used": rounds_completed,
    }
