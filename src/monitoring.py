"""
monitoring.py
Phase 3 — Data Drift & Model Quality Monitoring with Evidently

Generates two reports:
  1. Data Drift Report    — compares train vs val feature distributions
  2. Model Quality Report — RMSE, MAE, MAPE breakdown on val set

Reports saved as HTML to data/processed/reports/

Run standalone:
    python src/monitoring.py
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS   = PROCESSED / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
META_PATH = PROCESSED / "feature_meta.json"
MODELS    = ROOT / "models"


def load_data():
    meta     = json.loads(META_PATH.read_text())
    features = meta["features"]
    target   = meta["target"]

    train = pd.read_csv(PROCESSED / "train.csv", parse_dates=["date"])
    val   = pd.read_csv(PROCESSED / "val.csv",   parse_dates=["date"])

    return train, val, features, target, meta


def run_data_drift_report(train: pd.DataFrame, val: pd.DataFrame, features: list):
    """
    Evidently data drift report.
    Compares feature distributions between training set (reference)
    and validation set (current). Flags features that shifted significantly.
    """
    logger.info("Running data drift analysis...")

    try:
        import evidently
        from evidently import Dataset, DataDefinition, ColumnType, Report
        from evidently.presets import DataDriftPreset

        # Build column definitions for numeric features
        col_defs = {f: ColumnType.Numerical for f in features}
        data_def = DataDefinition(columns=col_defs)

        reference = Dataset(
            data=train[features].sample(min(2000, len(train)), random_state=42),
            data_definition=data_def
        )
        current = Dataset(
            data=val[features].copy(),
            data_definition=data_def
        )

        report = Report(metrics=[DataDriftPreset()])
        run = report.run(reference=reference, current=current)

        report_path = str(REPORTS / "data_drift_report.html")
        run.save_html(report_path)
        logger.info(f"Data drift report → {report_path}")

        # Extract drift summary from run dict
        run_dict = run.dict() if hasattr(run, "dict") else {}
        drifted_count = 0
        for metric_res in run_dict.get("metrics", []):
            v = metric_res.get("value", {})
            if isinstance(v, dict) and "number_of_drifted_columns" in v:
                drifted_count = v["number_of_drifted_columns"]
                break

        drift_results = {
            "data_drift_detected": drifted_count > 0,
            "n_drifted_features":  drifted_count,
            "drift_share":         drifted_count / max(len(features), 1),
        }

        logger.info(f"Drift detected: {drift_results['data_drift_detected']}")
        logger.info(f"Drifted features: {drift_results['n_drifted_features']} / {len(features)}")
        return drift_results

    except Exception as e:
        logger.warning(f"Evidently drift report failed ({e}). Falling back to manual analysis.")
        return _manual_drift_check(train, val, features)


def _manual_drift_check(train: pd.DataFrame, val: pd.DataFrame, features: list) -> dict:
    """Fallback: simple statistical drift detection using KS test."""
    from scipy import stats

    drifted = []
    for feat in features:
        if feat in train.columns and feat in val.columns:
            try:
                stat, p_value = stats.ks_2samp(
                    train[feat].dropna().values,
                    val[feat].dropna().values
                )
                if p_value < 0.05:          # statistically significant shift
                    drifted.append(feat)
            except Exception:
                pass

    result = {
        "data_drift_detected": len(drifted) > 0,
        "n_drifted_features":  len(drifted),
        "drift_share":         len(drifted) / len(features),
        "drifted_features":    drifted[:10],   # top 10
        "method":              "ks_test_fallback",
    }
    logger.info(f"Manual KS drift: {len(drifted)} features drifted")
    return result


def run_model_quality_report(
    train: pd.DataFrame, val: pd.DataFrame,
    features: list, target: str
):
    """
    Evidently model quality report.
    Evaluates regression metrics (RMSE, MAE, MAPE) on the val set.
    """
    logger.info("Running model quality report...")

    try:
        import joblib
        model_path = MODELS / "lightgbm_tuned.pkl"
        if not model_path.exists():
            model_path = MODELS / "pipeline_best_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError("No trained model found.")

        model = joblib.load(model_path)
        val_copy = val.copy()
        val_copy["prediction"] = model.predict(val[features]).clip(0)

        # Compute metrics manually for the report
        y_true = val_copy[target].values
        y_pred = val_copy["prediction"].values
        mask   = y_true > 1.0

        rmse_val = float(np.sqrt(((y_true - y_pred)**2).mean()))
        mae_val  = float(np.abs(y_true - y_pred).mean())
        mape_val = float(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]).mean() * 100)
        r2_val   = float(1 - ((y_true - y_pred)**2).sum() / ((y_true - y_true.mean())**2).sum())

        # Try Evidently regression report
        try:
            from evidently.report import Report
            from evidently.metric_preset import RegressionPreset

            ref_sample = train.copy()
            ref_sample["prediction"] = model.predict(train[features]).clip(0)

            report = Report(metrics=[RegressionPreset()])
            report.run(
                reference_data=ref_sample[[target, "prediction"]].sample(
                    min(2000, len(ref_sample)), random_state=42),
                current_data=val_copy[[target, "prediction"]]
            )
            report_path = str(REPORTS / "model_quality_report.html")
            report.save_html(report_path)
            logger.info(f"Model quality report → {report_path}")

        except Exception as e2:
            logger.warning(f"Evidently regression report failed ({e2}). Metrics computed manually.")

        metrics = {"rmse": rmse_val, "mae": mae_val, "mape": mape_val, "r2": r2_val}
        logger.info(f"Model quality — RMSE={rmse_val:,.1f}  MAE={mae_val:,.1f}  "
                    f"MAPE={mape_val:.2f}%  R²={r2_val:.4f}")
        return metrics

    except Exception as e:
        logger.warning(f"Model quality report skipped: {e}")
        return {"rmse": None, "mae": None, "mape": None, "r2": None}


def save_monitoring_summary(drift_results: dict, quality_metrics: dict):
    """Save a JSON summary of monitoring results."""
    summary = {
        "drift": drift_results,
        "quality": quality_metrics,
        "reports_dir": str(REPORTS),
    }
    out_path = REPORTS / "monitoring_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info(f"Monitoring summary → {out_path}")
    return summary


def run_monitoring():
    """Main entry point — called by Prefect task and standalone."""
    train, val, features, target, meta = load_data()

    drift_results   = run_data_drift_report(train, val, features)
    quality_metrics = run_model_quality_report(train, val, features, target)

    summary = save_monitoring_summary(drift_results, quality_metrics)

    # Merge for return
    return {**drift_results, **quality_metrics, "reports_dir": str(REPORTS)}


if __name__ == "__main__":
    results = run_monitoring()
    print("\n=== Monitoring Results ===")
    for k, v in results.items():
        if k != "reports_dir":
            print(f"  {k}: {v}")
    print(f"\nReports saved to: {results['reports_dir']}")
