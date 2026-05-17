"""
main.py
Phase 4 — FastAPI Prediction API

Endpoints:
  GET  /              → welcome message
  GET  /health        → model status + monitoring metrics
  POST /predict       → predict sales for one store-day
  POST /predict/batch → predict for multiple store-days
  GET  /model/info    → model metadata

Run locally:
    uvicorn app.main:app --reload --port 8000

Then open:
    http://localhost:8000/docs  ← auto-generated Swagger UI
"""

import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
MODELS    = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"
REPORTS   = PROCESSED / "reports"
META_PATH = PROCESSED / "feature_meta.json"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ── Load model + metadata at startup ──────────────────────────────────────────
def load_model():
    """Load the best trained model from Phase 2 or Phase 3."""
    for candidate in ["lightgbm_tuned.pkl", "pipeline_best_model.pkl",
                      "lightgbm.pkl", "xgboost_tuned.pkl", "xgboost.pkl"]:
        path = MODELS / candidate
        if path.exists():
            logger.info(f"Loading model: {path.name}")
            return joblib.load(path), path.name
    raise FileNotFoundError(
        "No trained model found in models/. Run src/train.py first."
    )

def load_meta():
    if not META_PATH.exists():
        raise FileNotFoundError("feature_meta.json not found. Run src/data_pipeline.py first.")
    return json.loads(META_PATH.read_text())

def load_best_model_info():
    p = MODELS / "best_model.json"
    return json.loads(p.read_text()) if p.exists() else {}

def load_monitoring_summary():
    p = REPORTS / "monitoring_summary.json"
    return json.loads(p.read_text()) if p.exists() else {}

# Global model state
MODEL, MODEL_NAME = load_model()
META    = load_meta()
FEATURES = META["features"]

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sales Forecasting API",
    description=(
        "MLOps end-to-end sales forecasting API. "
        "Predicts daily store sales using a LightGBM model trained on 4 years of retail data."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic schemas ───────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Input schema for a single-day sales prediction."""
    store_id:             int   = Field(..., ge=1, le=100,  description="Store ID (1–100)")
    date:                 str   = Field(...,                description="Date in YYYY-MM-DD format")
    promo:                int   = Field(0,  ge=0, le=1,    description="1 if promotion running, 0 otherwise")
    promo2:               int   = Field(0,  ge=0, le=1,    description="1 if recurring promo, 0 otherwise")
    school_holiday:       int   = Field(0,  ge=0, le=1,    description="1 if school holiday")
    competition_distance: float = Field(1000.0, ge=0,      description="Distance to nearest competitor (metres)")
    store_type:           str   = Field("a",               description="Store type: a, b, c, or d")
    assortment:           str   = Field("basic",           description="Assortment: basic, extra, or extended")
    sales_lag_1:          Optional[float] = Field(None,    description="Yesterday's sales (optional)")
    sales_lag_7:          Optional[float] = Field(None,    description="Sales 7 days ago (optional)")

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("date must be in YYYY-MM-DD format")

    @field_validator("store_type")
    @classmethod
    def validate_store_type(cls, v):
        if v not in ("a", "b", "c", "d"):
            raise ValueError("store_type must be a, b, c, or d")
        return v

    @field_validator("assortment")
    @classmethod
    def validate_assortment(cls, v):
        if v not in ("basic", "extra", "extended"):
            raise ValueError("assortment must be basic, extra, or extended")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "store_id": 3, "date": "2024-03-15", "promo": 1,
                "promo2": 0, "school_holiday": 0,
                "competition_distance": 1200.0,
                "store_type": "a", "assortment": "basic",
                "sales_lag_1": 8500.0, "sales_lag_7": 7900.0
            }
        }
    }


class PredictResponse(BaseModel):
    store_id:          int
    date:              str
    predicted_sales:   float
    model_name:        str
    confidence_note:   str


class BatchPredictRequest(BaseModel):
    requests: List[PredictRequest] = Field(..., max_length=100)


class BatchPredictResponse(BaseModel):
    predictions:    List[PredictResponse]
    total_requests: int
    model_name:     str


# ── Feature engineering for inference ─────────────────────────────────────────

def build_features(req: PredictRequest) -> pd.DataFrame:
    """
    Reconstruct the 36 features from a prediction request.
    Mirrors the logic in data_pipeline.py — must stay in sync.
    """
    dt = datetime.strptime(req.date, "%Y-%m-%d")

    store_type_enc  = {"a": 0, "b": 1, "c": 2, "d": 3}.get(req.store_type, 0)
    assortment_enc  = {"basic": 0, "extra": 1, "extended": 2}.get(req.assortment, 0)
    log_comp_dist   = float(np.log1p(req.competition_distance))

    # Use provided lags or sensible defaults based on store type
    default_lag = {0: 7500, 1: 10000, 2: 6000, 3: 5000}.get(store_type_enc, 7000)
    lag1  = req.sales_lag_1  if req.sales_lag_1  is not None else float(default_lag)
    lag7  = req.sales_lag_7  if req.sales_lag_7  is not None else float(default_lag * 0.95)
    lag14 = float(default_lag * 0.93)
    lag30 = float(default_lag * 0.90)

    # Rolling features derived from lags
    rolling_mean_7  = (lag1 + lag7) / 2
    rolling_mean_14 = (lag1 + lag7 + lag14) / 3
    rolling_mean_30 = (lag1 + lag7 + lag14 + lag30) / 4
    rolling_std_7   = float(np.std([lag1, lag7]))
    rolling_std_14  = float(np.std([lag1, lag7, lag14]))
    rolling_std_30  = float(np.std([lag1, lag7, lag14, lag30]))

    # Store aggregates (use lag mean as proxy at inference time)
    store_avg    = rolling_mean_30
    store_median = rolling_mean_7
    store_std    = rolling_std_30

    row = {
        "store_id":                   req.store_id,
        "promo":                      req.promo,
        "promo2":                     req.promo2,
        "school_holiday":             req.school_holiday,
        "competition_distance":       req.competition_distance,
        "is_state_holiday":           0,
        "year":                       dt.year,
        "month":                      dt.month,
        "day":                        dt.day,
        "day_of_week":                dt.weekday(),
        "week_of_year":               dt.isocalendar()[1],
        "quarter":                    (dt.month - 1) // 3 + 1,
        "is_weekend":                 int(dt.weekday() >= 5),
        "is_month_start":             int(dt.day == 1),
        "is_month_end":               int(dt.day in (28, 29, 30, 31)),
        "month_sin":                  float(np.sin(2 * np.pi * dt.month / 12)),
        "month_cos":                  float(np.cos(2 * np.pi * dt.month / 12)),
        "dow_sin":                    float(np.sin(2 * np.pi * dt.weekday() / 7)),
        "dow_cos":                    float(np.cos(2 * np.pi * dt.weekday() / 7)),
        "sales_lag_1":                lag1,
        "sales_lag_7":                lag7,
        "sales_lag_14":               lag14,
        "sales_lag_30":               lag30,
        "sales_rolling_mean_7":       rolling_mean_7,
        "sales_rolling_std_7":        rolling_std_7,
        "sales_rolling_mean_14":      rolling_mean_14,
        "sales_rolling_std_14":       rolling_std_14,
        "sales_rolling_mean_30":      rolling_mean_30,
        "sales_rolling_std_30":       rolling_std_30,
        "promo_rolling_7":            float(req.promo),
        "store_avg_sales":            store_avg,
        "store_median_sales":         store_median,
        "store_std_sales":            store_std,
        "store_type_enc":             store_type_enc,
        "assortment_enc":             assortment_enc,
        "log_competition_distance":   log_comp_dist,
    }

    df = pd.DataFrame([row])

    # Ensure all 36 training features are present in the right order
    missing = [f for f in FEATURES if f not in df.columns]
    for m in missing:
        df[m] = 0.0
    return df[FEATURES]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    return {
        "message": "Sales Forecasting API — Phase 4",
        "docs":    "/docs",
        "health":  "/health",
        "predict": "POST /predict",
    }


@app.get("/health", tags=["General"])
def health():
    """Returns model status, performance metrics, and drift summary."""
    monitoring = load_monitoring_summary()
    best_info  = load_best_model_info()

    quality = monitoring.get("quality", {})
    drift   = monitoring.get("drift",   {})

    return {
        "status":        "healthy",
        "model_loaded":  MODEL_NAME,
        "model_source":  best_info.get("model_name", "unknown"),
        "val_rmse":      best_info.get("val_rmse", None),
        "performance": {
            "rmse": quality.get("rmse"),
            "mape": quality.get("mape"),
            "r2":   quality.get("r2"),
        },
        "monitoring": {
            "drift_detected":    drift.get("data_drift_detected"),
            "drifted_features":  drift.get("n_drifted_features"),
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Predictions"])
def predict(req: PredictRequest):
    """
    Predict daily sales for one store on one date.

    Provide lag features (sales_lag_1, sales_lag_7) for best accuracy.
    If not provided, sensible defaults are used based on store type.
    """
    try:
        df   = build_features(req)
        pred = float(MODEL.predict(df)[0])
        pred = max(0.0, round(pred, 2))

        note = (
            "Lag features provided — high confidence prediction."
            if req.sales_lag_1 is not None
            else "Lag features not provided — using defaults. Provide sales_lag_1 for better accuracy."
        )

        logger.info(f"Predicted store={req.store_id} date={req.date} promo={req.promo} → {pred:,.0f}")

        return PredictResponse(
            store_id=req.store_id,
            date=req.date,
            predicted_sales=pred,
            model_name=MODEL_NAME,
            confidence_note=note,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Predictions"])
def predict_batch(req: BatchPredictRequest):
    """Predict sales for multiple store-days in one request (max 100)."""
    results = []
    for r in req.requests:
        try:
            df   = build_features(r)
            pred = float(MODEL.predict(df)[0])
            pred = max(0.0, round(pred, 2))
            results.append(PredictResponse(
                store_id=r.store_id, date=r.date,
                predicted_sales=pred, model_name=MODEL_NAME,
                confidence_note="batch prediction",
            ))
        except Exception as e:
            logger.error(f"Batch item error: {e}")
            results.append(PredictResponse(
                store_id=r.store_id, date=r.date,
                predicted_sales=-1.0, model_name=MODEL_NAME,
                confidence_note=f"Error: {str(e)}",
            ))

    return BatchPredictResponse(
        predictions=results,
        total_requests=len(results),
        model_name=MODEL_NAME,
    )


@app.get("/model/info", tags=["Model"])
def model_info():
    """Returns model metadata, feature list, and training configuration."""
    best = load_best_model_info()
    return {
        "model_file":    MODEL_NAME,
        "model_name":    best.get("model_name", "unknown"),
        "val_rmse":      best.get("val_rmse"),
        "feature_count": len(FEATURES),
        "features":      FEATURES,
        "target":        META.get("target"),
        "train_rows":    META.get("train_rows"),
        "val_rows":      META.get("val_rows"),
        "test_rows":     META.get("test_rows"),
        "train_range":   META.get("train_date_range"),
        "val_range":     META.get("val_date_range"),
        "test_range":    META.get("test_date_range"),
    }
