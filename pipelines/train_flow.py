"""
train_flow.py
Phase 3 — Prefect Orchestration Pipeline

Wraps the entire ML workflow into Prefect tasks and a flow:
  - Task 1: generate_data      → raw CSV
  - Task 2: run_data_pipeline  → clean + features + splits
  - Task 3: train_models       → MLflow experiments
  - Task 4: run_monitoring     → Evidently drift + quality reports
  - Task 5: validate_model     → pass/fail gate before "deploy"

Run:
    python pipelines/train_flow.py

Prefect will log each task's start, end, duration, and status.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

from prefect import flow, task, get_run_logger
from prefect.task_runners import ThreadPoolTaskRunner

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── Task 1: Generate raw data ─────────────────────────────────────────────────

@task(name="generate-raw-data", retries=2, retry_delay_seconds=5)
def generate_data_task():
    logger = get_run_logger()
    logger.info("Generating synthetic sales data...")

    from generate_data import main as gen_main
    gen_main()

    raw_path = ROOT / "data" / "raw" / "sales_data.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Expected {raw_path} — generation failed.")

    import pandas as pd
    df = pd.read_csv(raw_path)
    logger.info(f"Raw data ready: {len(df):,} rows")
    return str(raw_path)


# ── Task 2: Run data pipeline ─────────────────────────────────────────────────

@task(name="run-data-pipeline", retries=1)
def data_pipeline_task(raw_path: str):
    logger = get_run_logger()
    logger.info("Running data pipeline: clean → features → split...")

    from data_pipeline import run_pipeline
    train, val, test = run_pipeline()

    logger.info(f"Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    meta_path = ROOT / "data" / "processed" / "feature_meta.json"
    meta = json.loads(meta_path.read_text())
    logger.info(f"Features engineered: {len(meta['features'])}")
    return {"train_rows": len(train), "val_rows": len(val), "test_rows": len(test)}


# ── Task 3: Train models ──────────────────────────────────────────────────────

@task(name="train-models", retries=1)
def train_models_task():
    logger = get_run_logger()
    logger.info("Training models with MLflow tracking...")

    # Import and run training
    import mlflow
    import json as j
    import pandas as pd
    import numpy as np
    import joblib
    import warnings
    warnings.filterwarnings("ignore")

    sys.path.insert(0, str(ROOT / "src"))
    from evaluate import evaluate_all

    meta = j.loads((ROOT / "data" / "processed" / "feature_meta.json").read_text())
    features, target = meta["features"], meta["target"]

    train = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
    val   = pd.read_csv(ROOT / "data" / "processed" / "val.csv")
    test  = pd.read_csv(ROOT / "data" / "processed" / "test.csv")

    X_tr, y_tr = train[features].values, train[target].values
    X_va, y_va = val[features].values,   val[target].values
    X_te, y_te = test[features].values,  test[target].values

    mlflow.set_tracking_uri(str(ROOT / "mlruns"))
    mlflow.set_experiment("sales_forecasting_pipeline")

    results = {}

    # XGBoost
    from xgboost import XGBRegressor
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8, verbosity=0, random_state=42)
    xgb.fit(X_tr, y_tr)
    vp = xgb.predict(X_va).clip(0)
    xgb_metrics = evaluate_all(y_va, vp, "XGB-pipeline")
    results["xgboost"] = xgb_metrics["rmse"]

    with mlflow.start_run(run_name="pipeline_xgboost"):
        mlflow.log_params({"model": "xgboost", "source": "prefect_pipeline"})
        mlflow.log_metrics({f"val_{k}": v for k, v in xgb_metrics.items()})

    # LightGBM
    from lightgbm import LGBMRegressor
    lgb = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                        num_leaves=63, subsample=0.8, verbosity=-1, random_state=42)
    lgb.fit(X_tr, y_tr)
    vp2 = lgb.predict(X_va).clip(0)
    lgb_metrics = evaluate_all(y_va, vp2, "LGB-pipeline")
    results["lightgbm"] = lgb_metrics["rmse"]

    with mlflow.start_run(run_name="pipeline_lightgbm"):
        mlflow.log_params({"model": "lightgbm", "source": "prefect_pipeline"})
        mlflow.log_metrics({f"val_{k}": v for k, v in lgb_metrics.items()})

    # Save best
    best_name = min(results, key=results.get)
    best_model = lgb if best_name == "lightgbm" else xgb
    best_rmse  = results[best_name]

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(best_model, ROOT / "models" / "pipeline_best_model.pkl")

    best_info = {"model_name": best_name, "val_rmse": best_rmse,
                 "all_results": results, "source": "prefect_pipeline"}
    (ROOT / "models" / "best_model.json").write_text(j.dumps(best_info, indent=2))

    logger.info(f"Best model: {best_name}  RMSE={best_rmse:,.1f}")
    return best_info


# ── Task 4: Run Evidently monitoring ──────────────────────────────────────────

@task(name="run-monitoring", retries=1)
def monitoring_task():
    logger = get_run_logger()
    logger.info("Running Evidently data drift + model quality reports...")

    from monitoring import run_monitoring
    results = run_monitoring()

    logger.info(f"Data drift detected: {results['data_drift_detected']}")
    logger.info(f"Drifted features: {results['n_drifted_features']}")
    logger.info(f"Model RMSE on reference: {results.get('rmse', 'N/A')}")
    return results


# ── Task 5: Model validation gate ─────────────────────────────────────────────

@task(name="validate-model")
def validate_model_task(train_result: dict, rmse_threshold: float = 1200.0):
    logger = get_run_logger()
    val_rmse = train_result.get("val_rmse", 9999)

    logger.info(f"Validation gate — RMSE: {val_rmse:,.1f}  Threshold: {rmse_threshold:,.1f}")

    if val_rmse > rmse_threshold:
        raise ValueError(
            f"Model FAILED validation gate. "
            f"RMSE {val_rmse:,.1f} exceeds threshold {rmse_threshold:,.1f}. "
            f"Do NOT deploy."
        )

    logger.info(f"Model PASSED validation gate. Ready for deployment.")
    return {"passed": True, "val_rmse": val_rmse, "threshold": rmse_threshold}


# ── Main flow ─────────────────────────────────────────────────────────────────

@flow(
    name="sales-forecasting-pipeline",
    description="End-to-end sales forecasting: data → features → train → monitor → validate",
)
def sales_forecasting_flow(
    run_data_gen: bool = True,
    rmse_threshold: float = 1200.0,
):
    logger = get_run_logger()
    logger.info("=" * 50)
    logger.info("Sales Forecasting Pipeline — Phase 3")
    logger.info("=" * 50)

    # Step 1 — Generate data
    if run_data_gen:
        raw_path = generate_data_task()
    else:
        raw_path = str(ROOT / "data" / "raw" / "sales_data.csv")
        logger.info("Skipping data generation — using existing raw data.")

    # Step 2 — Data pipeline
    pipeline_stats = data_pipeline_task(raw_path)

    # Step 3 — Train models
    train_result = train_models_task()

    # Step 4 — Monitoring
    monitor_result = monitoring_task()

    # Step 5 — Validate
    validation = validate_model_task(train_result, rmse_threshold)

    logger.info("=" * 50)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Train rows : {pipeline_stats['train_rows']:,}")
    logger.info(f"  Best model : {train_result['model_name']}")
    logger.info(f"  Val RMSE   : {train_result['val_rmse']:,.1f}")
    logger.info(f"  Drift      : {monitor_result['data_drift_detected']}")
    logger.info(f"  Gate       : {'PASSED' if validation['passed'] else 'FAILED'}")
    logger.info("=" * 50)

    return {
        "pipeline_stats": pipeline_stats,
        "train_result": train_result,
        "monitor_result": monitor_result,
        "validation": validation,
    }


if __name__ == "__main__":
    result = sales_forecasting_flow(run_data_gen=False)
    print("\nPipeline finished:", result)
