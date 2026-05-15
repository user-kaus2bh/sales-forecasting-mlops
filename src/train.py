"""
train.py
Phase 2 — Model Training + Experiment Tracking

Trains three models in sequence:
  1. Linear Regression (baseline)
  2. XGBoost
  3. LightGBM

Each run is logged to MLflow with:
  - Parameters
  - Metrics (RMSE, MAE, MAPE, R²) on val and test sets
  - Feature importance plot
  - Trained model artifact

After all runs, the best model (lowest val RMSE) is registered
in the MLflow Model Registry as "sales_forecaster".

Usage:
    python src/train.py                    # train all models
    python src/train.py --tune             # + Optuna tuning (slower)
    python src/train.py --model xgboost   # train one model only
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from evaluate import evaluate_all, per_store_metrics

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
CONFIG_PATH = ROOT / "configs" / "model_config.yaml"
META_PATH = PROCESSED / "feature_meta.json"


# ── Load config & data ─────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_splits(meta: dict):
    features = meta["features"]
    target = meta["target"]

    train = pd.read_csv(PROCESSED / "train.csv", parse_dates=["date"])
    val   = pd.read_csv(PROCESSED / "val.csv",   parse_dates=["date"])
    test  = pd.read_csv(PROCESSED / "test.csv",  parse_dates=["date"])

    X_train, y_train = train[features].values, train[target].values
    X_val,   y_val   = val[features].values,   val[target].values
    X_test,  y_test  = test[features].values,  test[target].values

    logger.info(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), train, val, test, features


# ── MLflow setup ───────────────────────────────────────────────────────────────

def setup_mlflow(cfg: dict):
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    logger.info(f"MLflow experiment: {cfg['mlflow']['experiment_name']}")


# ── Feature importance plot ────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list, model_name: str) -> str:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return None

    idx = np.argsort(importances)[-20:]           # top 20 features
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color="steelblue", alpha=0.85
    )
    ax.set_title(f"Feature importance — {model_name}")
    ax.set_xlabel("Importance score")
    plt.tight_layout()

    path = str(MODELS_DIR / f"feature_importance_{model_name}.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Predictions plot ───────────────────────────────────────────────────────────

def plot_predictions(y_true, y_pred, title: str, model_name: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Actual vs predicted (first 200 points for clarity)
    n = min(200, len(y_true))
    axes[0].plot(y_true[:n], label="Actual", alpha=0.8, linewidth=1)
    axes[0].plot(y_pred[:n], label="Predicted", alpha=0.8, linewidth=1, linestyle="--")
    axes[0].set_title(f"Actual vs Predicted — {title}")
    axes[0].legend()
    axes[0].set_ylabel("Sales")

    # Residual scatter
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=8, color="coral")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Residuals vs Predicted")
    axes[1].set_xlabel("Predicted Sales")
    axes[1].set_ylabel("Residual (Actual − Predicted)")

    plt.tight_layout()
    path = str(MODELS_DIR / f"predictions_{model_name}.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


# ── 1. BASELINE — Linear Regression ───────────────────────────────────────────

def train_baseline(splits, cfg, features):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits
    logger.info("=" * 60)
    logger.info("Training: Linear Regression (baseline)")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)

    val_pred  = model.predict(X_val_s).clip(0)
    test_pred = model.predict(X_test_s).clip(0)

    val_metrics  = evaluate_all(y_val,  val_pred,  "LinearReg VAL")
    test_metrics = evaluate_all(y_test, test_pred, "LinearReg TEST")

    with mlflow.start_run(run_name="linear_regression") as run:
        mlflow.log_params({"model": "linear_regression", "scaler": "standard"})
        mlflow.log_metrics({f"val_{k}":  v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        pred_path = plot_predictions(y_val, val_pred, "LinearReg", "linear_regression")
        mlflow.log_artifact(pred_path)

        mlflow.sklearn.log_model(model, artifact_path="model")
        run_id = run.info.run_id

    joblib.dump({"model": model, "scaler": scaler}, MODELS_DIR / "linear_regression.pkl")
    logger.info(f"Baseline run_id: {run_id}")
    return val_metrics["rmse"], run_id


# ── 2. XGBoost ────────────────────────────────────────────────────────────────

def train_xgboost(splits, cfg, features, params: dict = None):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits
    logger.info("=" * 60)
    logger.info("Training: XGBoost")

    if params is None:
        params = cfg["models"]["xgboost"]["params"]
    params["random_state"] = cfg["training"]["random_seed"]

    model = XGBRegressor(
        **params,
        eval_metric="rmse",
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_pred  = model.predict(X_val).clip(0)
    test_pred = model.predict(X_test).clip(0)

    val_metrics  = evaluate_all(y_val,  val_pred,  "XGBoost VAL")
    test_metrics = evaluate_all(y_test, test_pred, "XGBoost TEST")

    with mlflow.start_run(run_name="xgboost") as run:
        mlflow.log_params({"model": "xgboost", **params})
        mlflow.log_metrics({f"val_{k}":  v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        fi_path   = plot_feature_importance(model, features, "xgboost")
        pred_path = plot_predictions(y_val, val_pred, "XGBoost", "xgboost")
        if fi_path:
            mlflow.log_artifact(fi_path)
        mlflow.log_artifact(pred_path)

        mlflow.xgboost.log_model(model, artifact_path="model")
        run_id = run.info.run_id

    joblib.dump(model, MODELS_DIR / "xgboost.pkl")
    logger.info(f"XGBoost run_id: {run_id}")
    return val_metrics["rmse"], run_id


# ── 3. LightGBM ───────────────────────────────────────────────────────────────

def train_lightgbm(splits, cfg, features, params: dict = None):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits
    logger.info("=" * 60)
    logger.info("Training: LightGBM")

    if params is None:
        params = cfg["models"]["lightgbm"]["params"]
    params["random_state"] = cfg["training"]["random_seed"]

    model = LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[],
    )

    val_pred  = model.predict(X_val).clip(0)
    test_pred = model.predict(X_test).clip(0)

    val_metrics  = evaluate_all(y_val,  val_pred,  "LightGBM VAL")
    test_metrics = evaluate_all(y_test, test_pred, "LightGBM TEST")

    with mlflow.start_run(run_name="lightgbm") as run:
        mlflow.log_params({"model": "lightgbm", **params})
        mlflow.log_metrics({f"val_{k}":  v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        fi_path   = plot_feature_importance(model, features, "lightgbm")
        pred_path = plot_predictions(y_val, val_pred, "LightGBM", "lightgbm")
        if fi_path:
            mlflow.log_artifact(fi_path)
        mlflow.log_artifact(pred_path)

        mlflow.lightgbm.log_model(model, artifact_path="model")
        run_id = run.info.run_id

    joblib.dump(model, MODELS_DIR / "lightgbm.pkl")
    logger.info(f"LightGBM run_id: {run_id}")
    return val_metrics["rmse"], run_id


# ── 4. Optuna Hyperparameter Tuning ───────────────────────────────────────────

def tune_xgboost(splits, cfg, features, n_trials: int = None):
    (X_train, y_train), (X_val, y_val), _ = splits
    n_trials = n_trials or cfg["training"]["n_optuna_trials"]
    search   = cfg["optuna"]["xgboost"]
    logger.info(f"Optuna tuning XGBoost — {n_trials} trials")

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators",     *search["n_estimators"]),
            "learning_rate":     trial.suggest_float("learning_rate",  *search["learning_rate"], log=True),
            "max_depth":         trial.suggest_int("max_depth",        *search["max_depth"]),
            "subsample":         trial.suggest_float("subsample",      *search["subsample"]),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", *search["colsample_bytree"]),
            "min_child_weight":  trial.suggest_int("min_child_weight", *search["min_child_weight"]),
            "reg_alpha":         trial.suggest_float("reg_alpha",      *search["reg_alpha"]),
            "reg_lambda":        trial.suggest_float("reg_lambda",     *search["reg_lambda"]),
            "random_state":      cfg["training"]["random_seed"],
            "verbosity":         0,
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  verbose=False)
        pred = model.predict(X_val).clip(0)
        return float(np.sqrt(((y_val - pred) ** 2).mean()))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    logger.info(f"Best XGBoost params: {best}")
    logger.info(f"Best val RMSE: {study.best_value:,.1f}")

    with mlflow.start_run(run_name="xgboost_tuned") as run:
        mlflow.log_params({"model": "xgboost_tuned", **best})
        mlflow.log_metric("optuna_best_val_rmse", study.best_value)
        mlflow.log_metric("n_trials", n_trials)
        run_id = run.info.run_id

    return best, study.best_value, run_id


def tune_lightgbm(splits, cfg, features, n_trials: int = None):
    (X_train, y_train), (X_val, y_val), _ = splits
    n_trials = n_trials or cfg["training"]["n_optuna_trials"]
    search   = cfg["optuna"]["lightgbm"]
    logger.info(f"Optuna tuning LightGBM — {n_trials} trials")

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators",    *search["n_estimators"]),
            "learning_rate":    trial.suggest_float("learning_rate", *search["learning_rate"], log=True),
            "max_depth":        trial.suggest_int("max_depth",       *search["max_depth"]),
            "num_leaves":       trial.suggest_int("num_leaves",      *search["num_leaves"]),
            "subsample":        trial.suggest_float("subsample",     *search["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *search["colsample_bytree"]),
            "min_child_samples":trial.suggest_int("min_child_samples", *search["min_child_samples"]),
            "reg_alpha":        trial.suggest_float("reg_alpha",     *search["reg_alpha"]),
            "reg_lambda":       trial.suggest_float("reg_lambda",    *search["reg_lambda"]),
            "random_state":     cfg["training"]["random_seed"],
            "verbosity":        -1,
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[])
        pred = model.predict(X_val).clip(0)
        return float(np.sqrt(((y_val - pred) ** 2).mean()))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    logger.info(f"Best LightGBM params: {best}")
    logger.info(f"Best val RMSE: {study.best_value:,.1f}")

    with mlflow.start_run(run_name="lightgbm_tuned") as run:
        mlflow.log_params({"model": "lightgbm_tuned", **best})
        mlflow.log_metric("optuna_best_val_rmse", study.best_value)
        mlflow.log_metric("n_trials", n_trials)
        run_id = run.info.run_id

    return best, study.best_value, run_id


# ── 5. Model Registry ─────────────────────────────────────────────────────────

def register_best_model(results: list, cfg: dict):
    """
    results = [(model_name, val_rmse, run_id), ...]
    Registers the best model (lowest val RMSE) in MLflow Model Registry.
    """
    best = min(results, key=lambda x: x[1])
    model_name, best_rmse, run_id = best

    logger.info("=" * 60)
    logger.info(f"Best model: {model_name}  |  Val RMSE: {best_rmse:,.1f}")
    logger.info(f"Registering as '{cfg['mlflow']['registered_model_name']}'")

    model_uri = f"runs:/{run_id}/model"
    try:
        mv = mlflow.register_model(
            model_uri=model_uri,
            name=cfg["mlflow"]["registered_model_name"],
        )
        logger.info(f"Registered: version {mv.version}")
    except Exception as e:
        logger.warning(f"Model registry step skipped: {e}")

    # Save best model info for Phase 3/4
    best_info = {
        "model_name": model_name,
        "run_id": run_id,
        "val_rmse": best_rmse,
        "registered_model": cfg["mlflow"]["registered_model_name"],
    }
    with open(MODELS_DIR / "best_model.json", "w") as f:
        json.dump(best_info, f, indent=2)
    logger.info(f"Saved best model info → models/best_model.json")
    return best_info


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run Optuna tuning")
    parser.add_argument("--model", default="all",
                        choices=["all", "baseline", "xgboost", "lightgbm"],
                        help="Which model to train")
    parser.add_argument("--trials", type=int, default=30,
                        help="Number of Optuna trials per model")
    args = parser.parse_args()

    cfg  = load_config()
    meta = json.load(open(META_PATH))
    setup_mlflow(cfg)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), \
        train_df, val_df, test_df, features = load_splits(meta)

    splits = (
        (X_train, y_train),
        (X_val,   y_val),
        (X_test,  y_test),
    )

    results = []   # [(model_name, val_rmse, run_id)]

    # ── Baseline ──────────────────────────────────────────────────────────────
    if args.model in ("all", "baseline"):
        rmse_val, run_id = train_baseline(splits, cfg, features)
        results.append(("linear_regression", rmse_val, run_id))

    # ── XGBoost ───────────────────────────────────────────────────────────────
    if args.model in ("all", "xgboost"):
        rmse_val, run_id = train_xgboost(splits, cfg, features)
        results.append(("xgboost", rmse_val, run_id))

        if args.tune:
            best_params, best_rmse, tuned_run_id = tune_xgboost(
                splits, cfg, features, n_trials=args.trials)
            # Train final model with best params
            rmse_tuned, tuned_run_id = train_xgboost(
                splits, cfg, features, params=best_params)
            results.append(("xgboost_tuned", rmse_tuned, tuned_run_id))

    # ── LightGBM ──────────────────────────────────────────────────────────────
    if args.model in ("all", "lightgbm"):
        rmse_val, run_id = train_lightgbm(splits, cfg, features)
        results.append(("lightgbm", rmse_val, run_id))

        if args.tune:
            best_params, best_rmse, tuned_run_id = tune_lightgbm(
                splits, cfg, features, n_trials=args.trials)
            rmse_tuned, tuned_run_id = train_lightgbm(
                splits, cfg, features, params=best_params)
            results.append(("lightgbm_tuned", rmse_tuned, tuned_run_id))

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    for name, rmse_val, run_id in sorted(results, key=lambda x: x[1]):
        marker = " ← best" if rmse_val == min(r[1] for r in results) else ""
        logger.info(f"  {name:<25} Val RMSE: {rmse_val:>8,.1f}{marker}")

    # ── Register best ─────────────────────────────────────────────────────────
    if results:
        register_best_model(results, cfg)

    logger.info("Phase 2 complete.")


if __name__ == "__main__":
    main()
