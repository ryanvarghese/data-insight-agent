"""
core/trends.py — Time trend precomputation.

Computes period-over-period metric changes before
the analysis loop runs. Passed to Claude as opening
context so it starts with quantitative trend data
rather than discovering basics in early rounds.

Fully dynamic — never hardcodes column names.
Detects which columns are time axes, metrics, and
grouping dimensions automatically from the schema.

Automatically detects natural time granularity:
  Daily data   → day-over-day analysis
  Weekly data  → week-over-week analysis
  Monthly data → month-over-month analysis
  Yearly data  → year-over-year analysis
"""

import pandas as pd


def _detect_frequency(time_series: pd.Series) -> str:
    """Infer natural period frequency from date spacing."""
    # Uses median gap between consecutive dates rather
    # than mean — median is robust to large gaps caused
    # by missing data periods which would skew the mean.
    cleaned = pd.to_datetime(time_series, errors="coerce").dropna().sort_values()
    if len(cleaned) < 3:
        return "M"
    day_gaps = cleaned.diff().dropna().dt.days
    median_gap = day_gaps.median()
    if median_gap <= 2:
        return "D"
    if median_gap <= 10:
        return "W"
    if median_gap <= 45:
        return "M"
    return "Y"


def _compute_overall_trend(df: pd.DataFrame, time_col: str, metric_col: str, freq: str) -> dict:
    """Compute aggregate trend changes for the primary metric."""
    work = df[[time_col, metric_col]].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
    work = work.dropna(subset=[time_col, metric_col])
    if work.empty:
        return {}

    grouped = (
        work.set_index(time_col)[metric_col]
        .resample(freq)
        .sum()
        .dropna()
    )
    if len(grouped) < 2:
        return {}

    latest = grouped.iloc[-1]
    previous = grouped.iloc[-2]
    pct_change = ((latest - previous) / previous * 100) if previous != 0 else None
    return {
        "latest_period_value": float(latest),
        "previous_period_value": float(previous),
        "period_over_period_change_pct": float(pct_change) if pct_change is not None else None,
    }


def _compute_dimension_trends(
    df: pd.DataFrame,
    time_col: str,
    metric_col: str,
    dimension_col: str,
    freq: str,
) -> dict:
    """Compute period-over-period trends per selected dimension."""
    # suitable_dims filters to 2-20 unique values.
    # Too few (1 value) = nothing to compare.
    # Too many (50+) = unreadable output with too many
    # groups for Claude to reason about meaningfully.
    work = df[[time_col, metric_col, dimension_col]].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
    work = work.dropna(subset=[time_col, metric_col, dimension_col])
    if work.empty:
        return {}

    unique_count = work[dimension_col].nunique(dropna=True)
    if unique_count < 2 or unique_count > 20:
        return {}

    trends = {}
    for dim_value, dim_df in work.groupby(dimension_col):
        series = dim_df.set_index(time_col)[metric_col].resample(freq).sum().dropna()
        if len(series) < 2:
            continue
        latest = series.iloc[-1]
        previous = series.iloc[-2]
        pct_change = ((latest - previous) / previous * 100) if previous != 0 else None
        trends[str(dim_value)] = (
            float(pct_change) if pct_change is not None else None
        )
    return trends


def precompute_trends(df: pd.DataFrame, schema: dict) -> dict:
    """Precompute overall and dimension trends for analysis context."""
    df = df.copy()
    datetime_cols = schema.get("datetime_columns", [])
    numeric_cols = schema.get("numeric_columns", [])
    categorical_cols = schema.get("categorical_columns", [])
    if not datetime_cols or not numeric_cols:
        return {}

    time_col = datetime_cols[0]
    metric_col = numeric_cols[0]
    freq = _detect_frequency(df[time_col])
    result = {
        "time_column": time_col,
        "metric_column": metric_col,
        "frequency": freq,
        "overall": _compute_overall_trend(df, time_col, metric_col, freq),
        "dimensions": {},
    }
    for dim_col in categorical_cols[:3]:
        dim_trend = _compute_dimension_trends(df, time_col, metric_col, dim_col, freq)
        if dim_trend:
            result["dimensions"][dim_col] = dim_trend
    return result
