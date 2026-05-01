"""
core/anomalies.py — Statistical anomaly detection.

Two complementary methods:

Z-SCORE (univariate):
  Flags values beyond 3.5 standard deviations from
  the column mean. Good for obvious single-column
  outliers like an unusually large Sales value.

ISOLATION FOREST (multivariate):
  Flags rows anomalous across multiple columns
  simultaneously using sklearn's ensemble method.
  Good for subtle anomalies no single column catches.

Results are deduplicated by row index, sorted by
z-score severity, and capped at 10 to surface only
the most statistically significant outliers.
"""

from typing import List

import pandas as pd
from sklearn.ensemble import IsolationForest


def _zscore_detection(df: pd.DataFrame, numeric_cols: List[str]) -> List[dict]:
    """Detect univariate outliers for each numeric column using z-score."""
    anomalies = []
    for column in numeric_cols:
        # Threshold of 3.5σ is stricter than the commonly
        # used 2.5σ. At 2.5σ this dataset flagged 168
        # anomalies — too many to be useful. At 3.5σ we
        # get 10 genuinely extraordinary transactions.
        # Z-score formula: z = (value - mean) / std_dev
        numeric_series = pd.to_numeric(df[column], errors="coerce")
        mean_value = numeric_series.mean()
        std_value = numeric_series.std()
        if pd.isna(std_value) or std_value == 0:
            continue

        z_scores = (numeric_series - mean_value) / std_value
        anomaly_mask = z_scores.abs() > 3.5
        for row_index, z_score in z_scores[anomaly_mask].items():
            value = numeric_series.loc[row_index]
            anomalies.append(
                {
                    "row_index": int(row_index) if isinstance(row_index, int) else row_index,
                    "column": column,
                    "value": value.item() if hasattr(value, "item") else value,
                    "z_score": float(z_score),
                    "direction": "high" if z_score > 0 else "low",
                    "context": df.loc[row_index].to_dict(),
                }
            )
    return anomalies


def _isolation_forest_detection(df: pd.DataFrame, numeric_cols: List[str]) -> List[dict]:
    """Detect multivariate anomalies using IsolationForest."""
    anomalies = []
    if len(numeric_cols) < 2 or df.shape[0] <= 20:
        return anomalies

    iso_input = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    iso_input = iso_input.dropna(how="all")
    if iso_input.empty:
        return anomalies

    iso_input = iso_input.fillna(iso_input.mean())
    # contamination=0.05 tells the model to expect
    # roughly 5% anomalous rows. random_state=42
    # ensures identical results on every run.
    # Returns -1 for anomalies, 1 for normal rows.
    # z_score remains a lower-priority placeholder so
    # these rows rank below strong z-score outliers.
    model = IsolationForest(contamination=0.05, random_state=42)
    predictions = model.fit_predict(iso_input)

    for row_index, prediction in zip(iso_input.index, predictions):
        if prediction == -1:
            row_values = {column: iso_input.loc[row_index, column] for column in numeric_cols}
            anomalies.append(
                {
                    "row_index": int(row_index) if isinstance(row_index, int) else row_index,
                    "numeric_values": row_values,
                    "z_score": 0.0,
                }
            )
    return anomalies


def detect_anomalies(df: pd.DataFrame, schema: dict) -> List[dict]:
    """Combine z-score and isolation-forest anomalies into a deduplicated top list."""
    df = df.copy()
    numeric_columns = schema.get("numeric_columns", [])
    zscore_anomalies = _zscore_detection(df, numeric_columns)
    isolation_anomalies = _isolation_forest_detection(df, numeric_columns)

    combined = zscore_anomalies + isolation_anomalies
    deduped_by_row = {}
    for anomaly in combined:
        row_key = anomaly.get("row_index")
        current_best = deduped_by_row.get(row_key)
        current_score = (
            abs(current_best.get("z_score", 0.0)) if current_best is not None else -1.0
        )
        new_score = abs(anomaly.get("z_score", 0.0))
        if current_best is None or new_score > current_score:
            deduped_by_row[row_key] = anomaly

    sorted_anomalies = sorted(
        deduped_by_row.values(),
        key=lambda anomaly: abs(anomaly.get("z_score", 0.0)),
        reverse=True,
    )
    return sorted_anomalies[:10]
