"""
generate_data.py
Generates a realistic synthetic retail sales dataset for MLOps forecasting project.
Mimics Rossmann-style store sales with seasonality, promotions, holidays, and noise.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

N_STORES = 10
START_DATE = "2020-01-01"
END_DATE = "2023-12-31"


def generate_sales(n_stores: int = N_STORES) -> pd.DataFrame:
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    records = []

    store_configs = {
        i: {
            "base_sales": np.random.randint(3000, 12000),
            "store_type": np.random.choice(["a", "b", "c", "d"]),
            "assortment": np.random.choice(["basic", "extra", "extended"]),
            "competition_distance": np.random.randint(200, 20000),
        }
        for i in range(1, n_stores + 1)
    }

    for store_id, cfg in store_configs.items():
        base = cfg["base_sales"]

        for date in dates:
            # Closed on Sunday (simulate real retail pattern)
            if date.weekday() == 6:
                records.append({
                    "date": date,
                    "store_id": store_id,
                    "sales": 0,
                    "customers": 0,
                    "open": 0,
                    "promo": 0,
                    "state_holiday": "0",
                    "school_holiday": 0,
                    "store_type": cfg["store_type"],
                    "assortment": cfg["assortment"],
                    "competition_distance": cfg["competition_distance"],
                    "promo2": 0,
                })
                continue

            # Day-of-week pattern
            dow_factor = {0: 1.10, 1: 0.95, 2: 0.90, 3: 0.92, 4: 1.05, 5: 1.20}[date.weekday()]

            # Monthly seasonality (Nov/Dec boost for holidays)
            month_factor = {
                1: 0.85, 2: 0.80, 3: 0.90, 4: 0.92, 5: 0.95,
                6: 1.00, 7: 1.05, 8: 1.00, 9: 0.95, 10: 0.98,
                11: 1.20, 12: 1.45
            }[date.month]

            # Promotions
            promo = int(np.random.random() < 0.45)
            promo_factor = 1.30 if promo else 1.0

            # Promo2 (recurring promotion)
            promo2 = int(np.random.random() < 0.3)

            # Holidays
            state_holiday = np.random.choice(["0", "a", "b", "c"], p=[0.93, 0.03, 0.02, 0.02])
            school_holiday = int(date.month in [6, 7, 8] or np.random.random() < 0.05)
            holiday_factor = 0.60 if state_holiday != "0" else 1.0

            # Year-over-year growth trend (2% per year)
            year_delta = (date.year - 2020)
            trend_factor = 1 + 0.02 * year_delta

            # Random noise
            noise = np.random.normal(1.0, 0.08)

            raw_sales = base * dow_factor * month_factor * promo_factor * holiday_factor * trend_factor * noise
            sales = max(0, int(raw_sales))
            customers = max(0, int(sales / np.random.uniform(8, 15)))

            records.append({
                "date": date,
                "store_id": store_id,
                "sales": sales,
                "customers": customers,
                "open": 1,
                "promo": promo,
                "promo2": promo2,
                "state_holiday": state_holiday,
                "school_holiday": school_holiday,
                "store_type": cfg["store_type"],
                "assortment": cfg["assortment"],
                "competition_distance": cfg["competition_distance"],
            })

    df = pd.DataFrame(records)
    df = df.sort_values(["store_id", "date"]).reset_index(drop=True)
    return df


def main():
    logger.info("Generating synthetic sales data...")
    df = generate_sales(N_STORES)

    out_path = RAW_DIR / "sales_data.csv"
    df.to_csv(out_path, index=False)

    logger.info(f"Saved {len(df):,} rows to {out_path}")
    logger.info(f"Date range : {df['date'].min().date()} → {df['date'].max().date()}")
    logger.info(f"Stores     : {df['store_id'].nunique()}")
    logger.info(f"Sales range: {df['sales'].min():,} – {df['sales'].max():,}")
    logger.info(f"Columns    : {list(df.columns)}")
    return df


if __name__ == "__main__":
    main()
