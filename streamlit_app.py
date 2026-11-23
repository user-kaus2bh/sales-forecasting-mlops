"""
streamlit_app.py — self-contained Sales Forecasting app for Streamlit Cloud.

No FastAPI server, MLflow, DVC or Prefect needed. On first launch it
generates the synthetic data, runs the existing pipeline code in src/,
trains a LightGBM model (~10 s) and caches everything.

Run locally:   streamlit run streamlit_app.py
"""

import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data_pipeline import clean_data, engineer_features, split_data  # noqa: E402
from evaluate import evaluate_all  # noqa: E402
from generate_data import generate_sales  # noqa: E402

st.set_page_config(page_title="Sales Forecasting", page_icon="📊", layout="wide")

EXCLUDE = {"date", "sales", "customers", "open"}
TARGET = "sales"


# ── Build data + model once ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="First run: generating data and training model (~15 s)…")
def build():
    raw = generate_sales()
    clean = clean_data(raw)                       # keeps store_type / assortment
    store_info = (
        clean.groupby("store_id")[["store_type", "assortment", "competition_distance"]]
        .first()
        .reset_index()
        .set_index("store_id")
    )

    feat = engineer_features(clean).dropna().reset_index(drop=True)
    features = [c for c in feat.columns if c not in EXCLUDE]
    train, val, test = split_data(feat)

    model = LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=63,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=-1,
    )
    model.fit(train[features], train[TARGET])

    metrics = {}
    for name, df in (("Validation", val), ("Test", test)):
        pred = model.predict(df[features]).clip(0)
        metrics[name] = evaluate_all(df[TARGET].values, pred, name)

    test_pred = pd.DataFrame({
        "date": test["date"].values, "store_id": test["store_id"].values,
        "Actual": test[TARGET].values,
        "Predicted": model.predict(test[features]).clip(0),
    })

    # per-store stats (same definition as data_pipeline.engineer_features)
    store_stats = feat[["store_id", "store_avg_sales", "store_median_sales",
                        "store_std_sales"]].drop_duplicates("store_id").set_index("store_id")

    importance = (pd.Series(model.feature_importances_, index=features)
                  .sort_values(ascending=False))

    return dict(model=model, features=features, feat=feat, store_info=store_info,
                store_stats=store_stats, metrics=metrics, test_pred=test_pred,
                importance=importance, raw=raw,
                last_date=feat["date"].max())


def forecast(art, store_id, start, days, promo_on, school_hol):
    """Recursive forecast. Lags/rolling use the last *open-day* sales of the store,
    exactly like data_pipeline.py (shift by rows, Sundays removed)."""
    feat, model, features = art["feat"], art["model"], art["features"]
    info = art["store_info"].loc[store_id]
    stats = art["store_stats"].loc[store_id]
    hist_df = feat[feat["store_id"] == store_id].sort_values("date")
    hist_df = hist_df[hist_df["date"] < pd.Timestamp(start)]
    if len(hist_df) < 30:
        hist_df = feat[feat["store_id"] == store_id].sort_values("date")

    sales = list(hist_df[TARGET].values)
    promos = list(hist_df["promo"].values)
    promo2 = int(hist_df["promo2"].iloc[-1])

    rows = []
    for i in range(days):
        d = pd.Timestamp(start) + pd.Timedelta(days=i)
        if d.weekday() == 6:                       # stores closed on Sunday
            rows.append({"Date": d.date(), "Day": d.strftime("%A"),
                         "Predicted Sales": 0.0, "Note": "Closed (Sunday)"})
            continue
        s = np.array(sales)
        x = {
            "store_id": store_id, "promo": int(promo_on), "promo2": promo2,
            "school_holiday": int(school_hol),
            "competition_distance": float(info["competition_distance"]),
            "is_state_holiday": 0, "year": d.year, "month": d.month, "day": d.day,
            "day_of_week": d.weekday(), "week_of_year": int(d.isocalendar()[1]),
            "quarter": d.quarter, "is_weekend": int(d.weekday() >= 5),
            "is_month_start": int(d.is_month_start), "is_month_end": int(d.is_month_end),
            "month_sin": np.sin(2 * np.pi * d.month / 12),
            "month_cos": np.cos(2 * np.pi * d.month / 12),
            "dow_sin": np.sin(2 * np.pi * d.weekday() / 7),
            "dow_cos": np.cos(2 * np.pi * d.weekday() / 7),
            "promo_rolling_7": float(sum(promos[-7:]) + 0),
            "store_avg_sales": stats["store_avg_sales"],
            "store_median_sales": stats["store_median_sales"],
            "store_std_sales": stats["store_std_sales"],
            "store_type_enc": {"a": 0, "b": 1, "c": 2, "d": 3}[info["store_type"]],
            "assortment_enc": {"basic": 0, "extra": 1, "extended": 2}[info["assortment"]],
            "log_competition_distance": np.log1p(float(info["competition_distance"])),
        }
        for lag in (1, 7, 14, 30):
            x[f"sales_lag_{lag}"] = s[-lag]
        for w in (7, 14, 30):
            x[f"sales_rolling_mean_{w}"] = s[-w:].mean()
            x[f"sales_rolling_std_{w}"] = s[-w:].std(ddof=1)
        pred = float(max(0, model.predict(pd.DataFrame([x])[features])[0]))
        sales.append(pred)
        promos.append(int(promo_on))
        rows.append({"Date": d.date(), "Day": d.strftime("%A"),
                     "Predicted Sales": round(pred, 0), "Note": ""})
    return pd.DataFrame(rows)


# ── UI ─────────────────────────────────────────────────────────────────────────
art = build()
last = art["last_date"].date()

st.title("📊 Sales Forecasting Dashboard")
st.caption("LightGBM · 10 stores · synthetic retail data (2020–2023) · trained live in this app")

tab1, tab2, tab3 = st.tabs(["🔮 Forecast", "📈 Model Performance", "🗂 Data"])

with tab1:
    c1, c2, c3 = st.columns(3)
    store_id = c1.selectbox("Store", list(range(1, 11)), index=2)
    start = c2.date_input("Forecast start date", value=last + timedelta(days=1),
                          min_value=last - timedelta(days=60))
    days = c3.slider("Days to forecast", 1, 28, 7)

    c4, c5 = st.columns(2)
    promo = c4.checkbox("Promotion running", value=True)
    school = c5.checkbox("School holiday", value=False)

    info = art["store_info"].loc[store_id]
    st.caption(f"Store {store_id}: type **{info['store_type']}**, assortment "
               f"**{info['assortment']}**, competitor **{int(info['competition_distance']):,} m** away. "
               f"History available until **{last}**.")

    fc = forecast(art, store_id, start, days, promo, school)
    open_fc = fc[fc["Note"] == ""]

    m1, m2, m3 = st.columns(3)
    m1.metric("Total forecast", f"{open_fc['Predicted Sales'].sum():,.0f}")
    m2.metric("Avg per open day", f"{open_fc['Predicted Sales'].mean():,.0f}")
    m3.metric("Open days", len(open_fc))

    past = art["feat"][(art["feat"]["store_id"] == store_id)
                       & (art["feat"]["date"] < pd.Timestamp(start))].tail(60)
    fig = go.Figure()
    fig.add_scatter(x=past["date"], y=past[TARGET], name="History", mode="lines")
    fig.add_scatter(x=pd.to_datetime(fc["Date"]), y=fc["Predicted Sales"],
                    name="Forecast", mode="lines+markers", line=dict(dash="dash"))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Sales")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(fc, width="stretch", hide_index=True)
    st.caption("Multi-day forecasts are recursive: each day's prediction feeds the next day's lag features.")

with tab2:
    st.subheader("Hold-out metrics")
    mt = pd.DataFrame(art["metrics"]).T.rename(columns=str.upper)
    st.dataframe(mt.style.format({"RMSE": "{:,.1f}", "MAE": "{:,.1f}",
                                  "MAPE": "{:.2f}%", "R2": "{:.4f}"}),
                 width="stretch")

    tp = art["test_pred"]
    sid = st.selectbox("Store for test-set chart", list(range(1, 11)), index=0, key="ts")
    t = tp[tp["store_id"] == sid]
    fig2 = go.Figure()
    fig2.add_scatter(x=t["date"], y=t["Actual"], name="Actual")
    fig2.add_scatter(x=t["date"], y=t["Predicted"], name="Predicted", line=dict(dash="dash"))
    fig2.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, width="stretch")

    st.subheader("Top feature importance")
    top = art["importance"].head(15)[::-1]
    fig3 = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h"))
    fig3.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig3, width="stretch")

with tab3:
    st.subheader("Raw data sample")
    st.dataframe(art["raw"].tail(200), width="stretch", hide_index=True)
    monthly = (art["raw"].assign(month=lambda d: d["date"].dt.to_period("M").dt.to_timestamp())
               .groupby("month")["sales"].sum())
    st.subheader("Total monthly sales (all stores)")
    st.line_chart(monthly)
