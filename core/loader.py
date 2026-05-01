"""
core/loader.py — File loading and data cleaning.

Runs once on file upload. Produces a clean,
correctly-typed DataFrame stored in
st.session_state["df"].

Every other module reads from this via df.copy()
and never modifies session_state["df"] directly.

Cleaning pipeline in order:
  1. Read CSV or Excel into raw DataFrame
  2. Strip whitespace from column names
  3. Drop completely empty rows and columns
  4. Detect and convert datetime columns
  5. Detect and convert numeric columns
     (strips $, commas, %, accounting negatives)
  6. Standardize string null representations
  7. Detect identifier columns statistically
  8. Compute basic stats for every column
"""

import numpy as np
import pandas as pd
import streamlit as st


def load_and_clean(uploaded_file):
    """Read the uploaded file and run the full cleaning pipeline."""
    if uploaded_file.name.lower().endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file)

    df = raw_df.copy()
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")
    df = df.reset_index(drop=True)

    df = _convert_datetime_columns(df)
    df = _convert_numeric_columns(df)
    df = _clean_string_columns(df)
    identifier_cols = _detect_identifier_columns(df)
    basic_stats = _compute_basic_stats(df, identifier_cols)

    st.session_state["df"] = df.copy()
    st.session_state["basic_stats"] = basic_stats
    st.session_state["identifier_cols"] = identifier_cols
    return df


def _convert_datetime_columns(df):
    """Convert non-numeric columns to datetimes when parse confidence is high."""
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        # Try multiple explicit formats before falling back
        # to pandas auto-detection. train.csv uses DD/MM/YYYY
        # which pandas misreads as MM/DD/YYYY by default,
        # producing invalid dates or keeping them as strings.
        # Trying formats explicitly and picking the one with
        # the highest parse success rate fixes this.
        formats_to_try = [
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%Y-%m-%d",
            "%d-%m-%Y",
            "%Y/%m/%d",
            "%d %b %Y",
            "%b %d %Y",
            None,
        ]

        best_result = None
        best_success_rate = 0

        for fmt in formats_to_try:
            try:
                if fmt is None:
                    converted = pd.to_datetime(
                        df[col],
                        infer_datetime_format=True,
                        errors="coerce",
                    )
                else:
                    converted = pd.to_datetime(
                        df[col],
                        format=fmt,
                        errors="coerce",
                    )
                success_rate = converted.notna().sum() / len(df) if len(df) else 0
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_result = converted
            except Exception:
                continue

        if best_result is not None and best_success_rate > 0.7:
            df[col] = best_result

    return df


def _convert_numeric_columns(df):
    """Convert non-datetime, non-numeric columns to numeric when parse confidence is high."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            # Strip currency symbols ($, £, €), thousands
            # separators (,), and percentage signs (%) before
            # numeric conversion. Many real-world financial
            # exports include these formatting characters.
            # Accounting negatives like (123) mean -123 —
            # the regex converts these before pd.to_numeric().
            cleaned = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(r"[$,\s%£€¥]", "", regex=True)
                .str.replace(r"^\((.+)\)$", r"-\1", regex=True)
            )
            converted = pd.to_numeric(cleaned, errors="coerce")
            success_rate = converted.notna().sum() / len(df) if len(df) else 0
            if success_rate > 0.7:
                df[col] = converted
        except Exception:
            pass
    return df


def _clean_string_columns(df):
    """Normalize string values and common null representations."""
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(
            ["nan", "none", "null", "n/a", "na", "-", ""],
            np.nan,
            regex=False,
        )
    return df


def _detect_identifier_columns(df):
    """Detect likely identifier columns using statistical properties only."""

    def is_identifier_column(series: pd.Series) -> bool:
        # Detects identifiers statistically, never by name.
        # This works on any dataset regardless of column
        # naming conventions (Row ID, employee_id, SKU etc.)
        #
        # Two signals:
        # 1. cardinality_ratio > 0.9 — almost every row has
        #    a unique value, which is the definition of an
        #    identifier column like a primary key.
        # 2. Sequential whole numbers — auto-increment IDs
        #    like 1, 2, 3, 4... from database exports.
        if not pd.api.types.is_numeric_dtype(series):
            return False
        clean = series.dropna()
        if len(clean) == 0:
            return False

        cardinality_ratio = clean.nunique() / len(clean)
        if cardinality_ratio > 0.9:
            return True
        try:
            is_whole = (clean == clean.round()).all()
            sorted_vals = clean.sort_values().values
            is_sequential = len(sorted_vals) > 1 and np.all(np.diff(sorted_vals) == 1)
            if is_whole and is_sequential:
                return True
        except Exception:
            pass
        return False

    return [col for col in df.columns if is_identifier_column(df[col])]


def _compute_basic_stats(df, identifier_cols):
    """Pre-compute reusable stats for all non-identifier columns."""
    basic_stats = {}
    for col in df.columns:
        if col in identifier_cols:
            continue

        # Pre-computes stats for every column before the
        # analysis loop runs. Passed to Claude in the opening
        # message so it starts with quantitative context
        # rather than wasting rounds on basic questions
        # like "what is the mean Sales value?".
        #
        # Skewness > 1 = right-skewed distribution.
        # Skewness > 3 = highly skewed (like retail sales).
        # Skewness of 12.98 in this dataset = extreme skew
        # where a few large orders dominate total revenue.
        if pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].dropna()
            if len(series) == 0:
                basic_stats[col] = {
                    "type": "numeric",
                    "count": 0,
                    "mean": None,
                    "median": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "sum": 0.0,
                    "null_count": int(df[col].isna().sum()),
                    "skewness": None,
                }
            else:
                basic_stats[col] = {
                    "type": "numeric",
                    "count": int(series.count()),
                    "mean": round(float(series.mean()), 2),
                    "median": round(float(series.median()), 2),
                    "std": round(float(series.std()), 2) if series.count() > 1 else 0.0,
                    "min": round(float(series.min()), 2),
                    "max": round(float(series.max()), 2),
                    "sum": round(float(series.sum()), 2),
                    "null_count": int(df[col].isna().sum()),
                    "skewness": round(float(series.skew()), 2) if series.count() > 2 else 0.0,
                }
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            series = df[col].dropna()
            if len(series) == 0:
                basic_stats[col] = {
                    "type": "datetime",
                    "min": None,
                    "max": None,
                    "range_days": None,
                    "null_count": int(df[col].isna().sum()),
                }
            else:
                basic_stats[col] = {
                    "type": "datetime",
                    "min": str(series.min().date()),
                    "max": str(series.max().date()),
                    "range_days": int((series.max() - series.min()).days),
                    "null_count": int(df[col].isna().sum()),
                }
        else:
            series = df[col].dropna()
            basic_stats[col] = {
                "type": "categorical",
                "unique_values": int(series.nunique()),
                "top_5": series.value_counts().head(5).to_dict(),
                "null_count": int(df[col].isna().sum()),
            }
    return basic_stats
