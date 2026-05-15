"""
evaluate.py
Centralised evaluation metrics for sales forecasting.
All metrics computed here — imported by train.py and pipelines.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error — penalises large errors heavily."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error — average absolute deviation in sales units."""
    return float(mean_absolute_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """
    Mean Absolute Percentage Error.
    eps guards against division by zero on near-zero true values.
    Returns a percentage (e.g. 8.5 means 8.5% average error).
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true > eps                         # ignore near-zero rows
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² (coefficient of determination).
    1.0 = perfect, 0.0 = predicts the mean, <0 = worse than mean.
    """
    return float(r2_score(y_true, y_pred))


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """Compute all four metrics and return as a dict."""
    metrics = {
        "rmse": rmse(y_true, y_pred),
        "mae":  mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2":   r2(y_true, y_pred),
    }
    prefix = f"[{label}] " if label else ""
    logger.info(
        f"{prefix}RMSE={metrics['rmse']:,.1f}  "
        f"MAE={metrics['mae']:,.1f}  "
        f"MAPE={metrics['mape']:.2f}%  "
        f"R²={metrics['r2']:.4f}"
    )
    return metrics


def per_store_metrics(df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Compute RMSE and MAPE broken down per store.
    Useful for spotting which stores the model struggles with.
    """
    df = df.copy()
    df["y_pred"] = y_pred
    rows = []
    for store_id, group in df.groupby("store_id"):
        rows.append({
            "store_id": store_id,
            "n_rows": len(group),
            "rmse": rmse(group["sales"].values, group["y_pred"].values),
            "mape": mape(group["sales"].values, group["y_pred"].values),
            "r2":   r2(group["sales"].values, group["y_pred"].values),
        })
    result = pd.DataFrame(rows).sort_values("rmse", ascending=False)
    return result
