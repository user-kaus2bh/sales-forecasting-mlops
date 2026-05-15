"""
data_pipeline.py
Full data pipeline for sales forecasting:
  - Load raw data
  - Clean and validate
  - Feature engineering (temporal, lag, rolling, store features)
  - Train/validation/test split
  - Save processed datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ─── 1. LOAD ──────────────────────────────────────────────────────────────────

def load_data(path: Path = RAW_DIR / "sales_data.csv") -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


# ─── 2. CLEAN & VALIDATE ──────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning data...")
    original_len = len(df)

    # Remove rows where store is closed (sales = 0, open = 0)
    df = df[~((df["open"] == 0) & (df["sales"] == 0))].copy()
    logger.info(f"Removed {original_len - len(df):,} closed-store rows")

    # Drop rows with negative sales
    df = df[df["sales"] >= 0].copy()

    # Fill missing competition_distance with median per store_type
    df["competition_distance"] = df.groupby("store_type")["competition_distance"].transform(
        lambda x: x.fillna(x.median())
    )

    # Encode state_holiday as binary
    df["is_state_holiday"] = (df["state_holiday"] != "0").astype(int)

    # Ensure correct dtypes
    df["date"] = pd.to_datetime(df["date"])
    df["store_id"] = df["store_id"].astype(int)

    # Sort
    df = df.sort_values(["store_id", "date"]).reset_index(drop=True)

    logger.info(f"Clean dataset: {len(df):,} rows")
    return df


# ─── 3. FEATURE ENGINEERING ───────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Engineering features...")

    df = df.copy()

    # ── Temporal features ──
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["day"]         = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek          # 0=Mon, 6=Sun
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"]     = df["date"].dt.quarter
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"]   = df["date"].dt.is_month_end.astype(int)

    # Sine/cosine encoding for cyclical features
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]    = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # ── Lag features (per store) ──
    lag_days = [1, 7, 14, 30]
    for lag in lag_days:
        df[f"sales_lag_{lag}"] = df.groupby("store_id")["sales"].shift(lag)

    # ── Rolling window features (per store) ──
    for window in [7, 14, 30]:
        df[f"sales_rolling_mean_{window}"] = (
            df.groupby("store_id")["sales"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"sales_rolling_std_{window}"] = (
            df.groupby("store_id")["sales"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
        )

    # Rolling promo effect
    df["promo_rolling_7"] = (
        df.groupby("store_id")["promo"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).sum())
    )

    # ── Store-level aggregated features ──
    store_stats = (
        df[df["sales"] > 0]
        .groupby("store_id")["sales"]
        .agg(store_avg_sales="mean", store_median_sales="median", store_std_sales="std")
        .reset_index()
    )
    df = df.merge(store_stats, on="store_id", how="left")

    # ── Encode categoricals ──
    df["store_type_enc"]  = df["store_type"].map({"a": 0, "b": 1, "c": 2, "d": 3}).fillna(0).astype(int)
    df["assortment_enc"]  = df["assortment"].map({"basic": 0, "extra": 1, "extended": 2}).fillna(0).astype(int)

    # Log-transform competition distance (right-skewed)
    df["log_competition_distance"] = np.log1p(df["competition_distance"])

    # Drop raw columns we've encoded
    df.drop(columns=["state_holiday", "store_type", "assortment"], inplace=True)

    logger.info(f"Feature-engineered dataset: {df.shape[1]} columns")
    return df


# ─── 4. SPLIT ─────────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame, val_months: int = 3, test_months: int = 3):
    """
    Time-based split to avoid data leakage:
      - Test  : last `test_months` months
      - Val   : `val_months` before test
      - Train : everything before val
    """
    logger.info("Splitting into train / val / test...")

    max_date = df["date"].max()
    test_start = max_date - pd.DateOffset(months=test_months)
    val_start  = test_start - pd.DateOffset(months=val_months)

    train = df[df["date"] < val_start].copy()
    val   = df[(df["date"] >= val_start) & (df["date"] < test_start)].copy()
    test  = df[df["date"] >= test_start].copy()

    logger.info(f"Train : {len(train):,} rows  ({train['date'].min().date()} → {train['date'].max().date()})")
    logger.info(f"Val   : {len(val):,}  rows  ({val['date'].min().date()} → {val['date'].max().date()})")
    logger.info(f"Test  : {len(test):,}  rows  ({test['date'].min().date()} → {test['date'].max().date()})")

    return train, val, test


# ─── 5. SAVE ──────────────────────────────────────────────────────────────────

def save_splits(train, val, test):
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)

    # Save feature list for reproducibility
    feature_cols = [c for c in train.columns if c not in ["date", "sales", "customers", "open"]]
    meta = {
        "features": feature_cols,
        "target": "sales",
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "train_date_range": [str(train["date"].min().date()), str(train["date"].max().date())],
        "val_date_range":   [str(val["date"].min().date()),   str(val["date"].max().date())],
        "test_date_range":  [str(test["date"].min().date()),  str(test["date"].max().date())],
    }
    with open(PROCESSED_DIR / "feature_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved train/val/test to {PROCESSED_DIR}")
    logger.info(f"Feature count: {len(feature_cols)}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run_pipeline():
    df_raw = load_data()
    df_clean = clean_data(df_raw)
    df_features = engineer_features(df_clean)

    # Drop rows with NaN from lag features (first N rows per store)
    df_features.dropna(inplace=True)
    logger.info(f"After dropping lag NaNs: {len(df_features):,} rows")

    train, val, test = split_data(df_features)
    save_splits(train, val, test)
    logger.info("Pipeline complete.")
    return train, val, test


if __name__ == "__main__":
    run_pipeline()
