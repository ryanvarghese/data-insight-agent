"""
ui/query_tab.py — Natural language query interface.

Lets users ask plain English questions about their
dataset and receive specific, grounded answers.

Uses a simplified single-round version of the main
analysis loop:
  Round 1: Claude decides what computation answers
           the question — or flags OUT_OF_SCOPE
  Round 2: Claude reads the result and writes a
           plain English answer

Out-of-scope handling prevents hallucination:
  Questions about topics not in the dataset columns
  return a clear message explaining what IS available
  rather than Claude attempting to answer incorrectly
  from training data memory.
"""

import html
import json
from datetime import datetime

import pandas as pd
import streamlit as st

from core.analysis import run_computation


def _sanitize_answer_text(text):
    """Escape answer text for safe HTML rendering."""
    if not text:
        return ""
    text = str(text)
    text = html.escape(text, quote=False)
    text = text.replace("\n", "<br>")
    return text


def _extract_response_text(response) -> str:
    """Extract text blocks from Anthropic response object."""
    text_parts = []
    for block in getattr(response, "content", []):
        if getattr(block, "type", "") == "text":
            text_parts.append(block.text)
    return "\n".join(text_parts).strip()


def _extract_section(text: str, start_label: str, end_label: str = "") -> str:
    """Extract section body between markers."""
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


def _safe_rerun() -> None:
    """Rerun helper supporting old/new streamlit APIs."""
    rerun_fn = getattr(st, "rerun", None)
    if callable(rerun_fn):
        rerun_fn()
        return
    experimental_rerun_fn = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun_fn):
        experimental_rerun_fn()


def _answer_question(df: pd.DataFrame, schema: dict, question: str) -> dict:
    """Run two-round Q&A loop: compute step then plain-English answer."""
    # Two-round mini loop mirrors the main analysis loop:
    # Round 1 determines WHAT to compute (or OUT_OF_SCOPE)
    # Round 2 interprets the result into plain English
    # Same guarantee applies: Claude reads pandas output,
    # it does not generate numbers from memory.
    df = df.copy()
    identifier_cols = st.session_state.get("identifier_cols", [])
    if isinstance(identifier_cols, list):
        valid_drop_cols = [col for col in identifier_cols if col in df.columns]
        if valid_drop_cols:
            df = df.drop(columns=valid_drop_cols)

    client = st.session_state.get("anthropic_client")
    if client is None:
        return {
            "answer": "Anthropic client is not initialized. Check ANTHROPIC_API_KEY.",
            "chart_spec": None,
            "computation_used": "",
            "raw_result": "",
            "out_of_scope": False,
        }

    today_str = datetime.now().strftime("%Y-%m-%d")
    system_prompt = (
        "You are a retail data analyst. You will answer a specific question "
        "about a dataset by requesting one computation, receiving the result, "
        "and then writing a clear plain English answer.\n\n"
        f"Today's date is {today_str}.\n\n"
        "First, respond with exactly: COMPUTE: [single pandas computation "
        "that answers the question]\n\n"
        "After receiving the result, respond with:\n"
        "ANSWER: [2-3 sentence plain English answer]\n\n"
        "If the question cannot be answered using the available columns "
        "in this dataset, do not attempt to find workarounds or guess. "
        "Instead respond with exactly:\n"
        "OUT_OF_SCOPE: This dataset contains [list the actual columns] "
        "and cannot answer questions about [topic of the question]. "
        "Try asking about sales, regions, categories, segments, or time trends."
    )

    first_user_message = (
        f"Schema:\n{json.dumps(schema, indent=2, default=str)}\n\n"
        f"Question:\n{question}"
    )

    try:
        compute_response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1000,
            system=system_prompt,
            messages=[{"role": "user", "content": first_user_message}],
        )
        compute_text = _extract_response_text(compute_response)

        # OUT_OF_SCOPE is a hard protocol string Claude must
        # return if the question cannot be answered from the
        # available columns. This prevents Claude from
        # hallucinating answers about data that doesn't exist
        # in the dataset (e.g. asking about artists when the
        # dataset only contains retail sales data).
        if compute_text.strip().startswith("OUT_OF_SCOPE:"):
            return {
                "answer": compute_text.strip(),
                "chart_spec": None,
                "computation_used": None,
                "raw_result": None,
                "out_of_scope": True,
            }

        computation_line = _extract_section(compute_text, "COMPUTE:")
        computation_used = computation_line.splitlines()[0].strip() if computation_line else ""
        if not computation_used:
            return {
                "answer": "Unable to compute an answer for that question.",
                "chart_spec": None,
                "computation_used": "",
                "raw_result": "",
                "out_of_scope": False,
            }

        raw_result = run_computation(df, computation_used)
        second_user_message = (
            f"Question:\n{question}\n\n"
            f"COMPUTATION_USED:\n{computation_used}\n\n"
            f"RESULT:\n{raw_result}\n\n"
            "Now provide your final response in the requested ANSWER format."
        )
        final_response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1000,
            system=system_prompt,
            messages=[{"role": "user", "content": second_user_message}],
        )
        final_text = _extract_response_text(final_response)
        answer_text = _extract_section(final_text, "ANSWER:").strip()
        if answer_text.upper().startswith("ANSWER:"):
            answer_text = answer_text[len("ANSWER:"):].strip()

        return {
            "answer": answer_text if answer_text else final_text.strip(),
            "chart_spec": None,
            "computation_used": computation_used,
            "raw_result": raw_result,
            "out_of_scope": False,
        }
    except Exception as exc:
        return {
            "answer": f"Error while answering question: {exc}",
            "chart_spec": None,
            "computation_used": "",
            "raw_result": "",
            "out_of_scope": False,
        }


def _render_example_buttons():
    """Render sample question buttons that prefill input text."""
    # Example buttons pre-fill the text input via
    # session_state on the next rerun. Shows users what
    # kinds of questions work well without restricting
    # what they can actually ask.
    example_questions = [
        "Which region performed best last quarter?",
        "What was the worst performing week?",
        "Are there any unusual spikes in revenue?",
        "Which product has the most consistent sales?",
    ]
    for idx, sample_question in enumerate(example_questions):
        if st.button(sample_question, key=f"example_q_{idx}", use_container_width=True):
            st.session_state["question_input"] = sample_question
            _safe_rerun()


def _render_query_history():
    """Render compact recent query history section."""
    # Capped at 3 most recent Q&A pairs. Enough for
    # reference during a session without cluttering
    # the interface with a long scrollable history.
    with st.expander("Recent Question History"):
        for item in st.session_state.get("query_history", []):
            st.markdown(f"**Q:** {item.get('question', '')}")
            st.write(f"A: {item.get('answer', '')}")


def render_query_tab() -> None:
    """Render the complete Ask a Question tab."""
    df = st.session_state.get("df")
    schema = st.session_state.get("schema", {})
    if "query_history" not in st.session_state:
        st.session_state["query_history"] = []
    if "latest_qa_result" not in st.session_state:
        st.session_state["latest_qa_result"] = None
    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""

    st.markdown("### Ask a Question")
    _render_example_buttons()

    question = st.text_input("Ask anything about your data...", key="question_input")
    can_ask = isinstance(df, pd.DataFrame) and isinstance(schema, dict) and bool(question.strip())
    if st.button("Get Answer", disabled=not can_ask, use_container_width=True):
        with st.spinner("Thinking..."):
            qa_result = _answer_question(df, schema, question.strip())
            st.session_state["latest_qa_result"] = qa_result
            history = st.session_state.get("query_history", [])
            history.insert(
                0,
                {
                    "question": question.strip(),
                    "answer": qa_result.get("answer", ""),
                },
            )
            st.session_state["query_history"] = history[:3]

    latest_qa_result = st.session_state.get("latest_qa_result")
    if isinstance(latest_qa_result, dict):
        if latest_qa_result.get("out_of_scope"):
            st.markdown(
                f'<div class="anomaly-card">⚠️ {_sanitize_answer_text(latest_qa_result.get("answer", ""))}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="insight-card">{_sanitize_answer_text(latest_qa_result.get("answer", ""))}</div>',
                unsafe_allow_html=True,
            )
            with st.expander("How was this computed?"):
                st.markdown("**Computation Used**")
                st.code(latest_qa_result.get("computation_used", ""))
                st.markdown("**Raw Result**")
                st.code(latest_qa_result.get("raw_result", ""))
    elif not isinstance(df, pd.DataFrame):
        st.info("Upload a file first to ask questions about your data.")

    _render_query_history()
