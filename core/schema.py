"""
core/schema.py — Column type classification.

Inspects the cleaned DataFrame and produces a schema
dict categorizing every column by analytical role:

  numeric_columns     → measurable metrics (Sales)
  categorical_columns → grouping dimensions (Region)
  datetime_columns    → time axes (Order Date)

Used by every other module so they understand the
dataset structure without re-inspecting the DataFrame.
"""

import pandas as pd
import streamlit as st


def extract_schema(df: pd.DataFrame) -> dict:
    """Classify DataFrame columns into analysis-friendly groups."""
    df = df.copy()
    datetime_columns = []
    identifier_cols = st.session_state.get("identifier_cols", [])
    if not isinstance(identifier_cols, list):
        identifier_cols = []

    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            datetime_columns.append(column)

    object_columns = df.select_dtypes(include=["object"]).columns.tolist()
    for column in object_columns:
        non_null_count = int(df[column].notna().sum())
        if non_null_count == 0:
            continue

        parsed_series = pd.to_datetime(df[column], errors="coerce")
        success_ratio = float(parsed_series.notna().sum()) / float(non_null_count)

        if success_ratio > 0.8:
            df[column] = parsed_series
            datetime_columns.append(column)

    numeric_columns = [
        column
        for column in df.columns
        if (pd.api.types.is_float_dtype(df[column]) or pd.api.types.is_integer_dtype(df[column]))
        and column not in identifier_cols
    ]

    # Categorical columns capped at cardinality < 50.
    # High-cardinality columns like Product Name (hundreds
    # of unique values) are excluded — grouping by them
    # produces unreadable results with too many categories.
    # They stay in df but are excluded from analysis.
    categorical_columns = [
        column
        for column in df.columns
        if pd.api.types.is_object_dtype(df[column]) and df[column].nunique(dropna=True) < 50
    ]

    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": df.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "sample_rows": df.head().to_dict("records"),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": sorted(set(datetime_columns)),
        "null_counts": {column: int(df[column].isna().sum()) for column in df.columns},
        "cardinality": {
            column: int(df[column].nunique(dropna=True)) for column in categorical_columns
        },
    }
