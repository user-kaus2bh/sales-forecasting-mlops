"""
test_train.py
Unit tests for model training and evaluation functions (Phase 2 + Phase 3).

Tests cover:
  - Metric calculations
  - Model output shapes
  - No NaN predictions
  - Model improves on baseline

Run: pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import rmse, mae, mape, r2, evaluate_all


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_predictions():
    y = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
    return y, y.copy()   # actual == predicted


@pytest.fixture
def typical_predictions():
    y_true = np.array([5000.0, 8000.0, 6000.0, 9000.0, 7000.0])
    y_pred = np.array([5200.0, 7800.0, 6300.0, 8700.0, 7100.0])
    return y_true, y_pred


@pytest.fixture
def small_dataset():
    """Load real train/val if available, otherwise generate tiny synthetic."""
    train_path = ROOT / "data" / "processed" / "train.csv"
    val_path   = ROOT / "data" / "processed" / "val.csv"
    meta_path  = ROOT / "data" / "processed" / "feature_meta.json"

    if train_path.exists() and val_path.exists() and meta_path.exists():
        import json
        meta     = json.loads(meta_path.read_text())
        features = meta["features"]
        target   = meta["target"]
        train    = pd.read_csv(train_path).head(500)
        val      = pd.read_csv(val_path).head(100)
        return train, val, features, target
    else:
        pytest.skip("Processed data not found — run data_pipeline.py first")


# ── Metric tests ──────────────────────────────────────────────────────────────

class TestMetrics:

    def test_rmse_perfect_predictions(self, perfect_predictions):
        """RMSE must be 0 when predictions equal actuals."""
        y_true, y_pred = perfect_predictions
        assert rmse(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)

    def test_rmse_positive(self, typical_predictions):
        """RMSE must always be non-negative."""
        y_true, y_pred = typical_predictions
        assert rmse(y_true, y_pred) >= 0

    def test_rmse_penalises_large_errors(self):
        """RMSE with one large error must exceed MAE with same total error."""
        y_true = np.array([1000.0, 1000.0, 1000.0, 1000.0])
        y_small = np.array([1100.0, 1100.0, 1100.0, 1100.0])   # uniform 100 error
        y_large = np.array([1400.0, 1000.0, 1000.0, 1000.0])   # one 400 error
        # Both have same MAE=100 but large-error has higher RMSE
        assert rmse(y_true, y_large) > rmse(y_true, y_small)

    def test_mae_perfect_predictions(self, perfect_predictions):
        """MAE must be 0 for perfect predictions."""
        y_true, y_pred = perfect_predictions
        assert mae(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)

    def test_mae_less_than_or_equal_rmse(self, typical_predictions):
        """MAE <= RMSE always (Jensen's inequality)."""
        y_true, y_pred = typical_predictions
        assert mae(y_true, y_pred) <= rmse(y_true, y_pred) + 1e-6

    def test_mape_perfect_predictions(self, perfect_predictions):
        """MAPE must be 0 for perfect predictions."""
        y_true, y_pred = perfect_predictions
        assert mape(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)

    def test_mape_percentage_range(self, typical_predictions):
        """MAPE must be a percentage value (0–100 range for reasonable predictions)."""
        y_true, y_pred = typical_predictions
        result = mape(y_true, y_pred)
        assert 0 <= result <= 100, f"MAPE {result} outside [0, 100]"

    def test_mape_handles_near_zero(self):
        """MAPE must not crash when actuals are near zero."""
        y_true = np.array([0.0, 0.5, 1000.0])
        y_pred = np.array([100.0, 200.0, 1100.0])
        result = mape(y_true, y_pred)   # should not raise ZeroDivisionError
        assert result >= 0

    def test_r2_perfect_predictions(self, perfect_predictions):
        """R² must be exactly 1.0 for perfect predictions."""
        y_true, y_pred = perfect_predictions
        assert r2(y_true, y_pred) == pytest.approx(1.0, abs=1e-6)

    def test_r2_predicting_mean(self):
        """R² must be 0.0 when model always predicts the mean."""
        y_true = np.array([1000.0, 2000.0, 3000.0, 4000.0])
        y_pred = np.full_like(y_true, y_true.mean())
        assert r2(y_true, y_pred) == pytest.approx(0.0, abs=1e-4)

    def test_r2_between_neg_inf_and_one(self, typical_predictions):
        """R² must be <= 1.0."""
        y_true, y_pred = typical_predictions
        assert r2(y_true, y_pred) <= 1.0

    def test_evaluate_all_returns_all_keys(self, typical_predictions):
        """evaluate_all must return all 4 metrics."""
        y_true, y_pred = typical_predictions
        result = evaluate_all(y_true, y_pred, label="test")
        for key in ["rmse", "mae", "mape", "r2"]:
            assert key in result, f"Missing metric: {key}"
            assert isinstance(result[key], float), f"{key} must be a float"

    def test_evaluate_all_no_nan(self, typical_predictions):
        """No metric should return NaN for well-formed predictions."""
        y_true, y_pred = typical_predictions
        result = evaluate_all(y_true, y_pred)
        for key, val in result.items():
            assert not np.isnan(val), f"Metric {key} returned NaN"


# ── Model tests ───────────────────────────────────────────────────────────────

class TestModelTraining:

    def test_xgboost_trains_and_predicts(self, small_dataset):
        """XGBoost must train without errors and produce valid predictions."""
        train, val, features, target = small_dataset
        from xgboost import XGBRegressor

        model = XGBRegressor(n_estimators=10, verbosity=0, random_state=42)
        model.fit(train[features], train[target])
        preds = model.predict(val[features])

        assert len(preds) == len(val), "Prediction count must match val row count"
        assert not np.any(np.isnan(preds)), "Predictions must not contain NaN"
        assert not np.any(np.isinf(preds)), "Predictions must not contain Inf"

    def test_lightgbm_trains_and_predicts(self, small_dataset):
        """LightGBM must train without errors and produce valid predictions."""
        train, val, features, target = small_dataset
        from lightgbm import LGBMRegressor
        import warnings
        warnings.filterwarnings("ignore")

        model = LGBMRegressor(n_estimators=10, verbosity=-1, random_state=42)
        model.fit(train[features], train[target])
        preds = model.predict(val[features])

        assert len(preds) == len(val)
        assert not np.any(np.isnan(preds))
        assert not np.any(np.isinf(preds))

    def test_tree_model_beats_baseline(self, small_dataset):
        """XGBoost or LightGBM must achieve lower RMSE than a mean baseline."""
        train, val, features, target = small_dataset
        from xgboost import XGBRegressor

        # Mean baseline — predicts the training mean for every row
        baseline_pred = np.full(len(val), train[target].mean())
        baseline_rmse = rmse(val[target].values, baseline_pred)

        model = XGBRegressor(n_estimators=50, verbosity=0, random_state=42)
        model.fit(train[features], train[target])
        model_pred = model.predict(val[features]).clip(0)
        model_rmse = rmse(val[target].values, model_pred)

        assert model_rmse < baseline_rmse, (
            f"Model RMSE ({model_rmse:.1f}) must beat mean baseline ({baseline_rmse:.1f})"
        )

    def test_predictions_are_non_negative(self, small_dataset):
        """After clip(0), all predictions must be >= 0."""
        train, val, features, target = small_dataset
        from xgboost import XGBRegressor

        model = XGBRegressor(n_estimators=10, verbosity=0, random_state=42)
        model.fit(train[features], train[target])
        preds = model.predict(val[features]).clip(0)

        assert (preds >= 0).all(), "Clipped predictions must all be non-negative"

    def test_model_r2_above_zero(self, small_dataset):
        """Model must explain more variance than the mean (R² > 0)."""
        train, val, features, target = small_dataset
        from lightgbm import LGBMRegressor
        import warnings
        warnings.filterwarnings("ignore")

        model = LGBMRegressor(n_estimators=50, verbosity=-1, random_state=42)
        model.fit(train[features], train[target])
        preds = model.predict(val[features]).clip(0)

        r2_score = r2(val[target].values, preds)
        assert r2_score > 0, f"R² = {r2_score:.4f} — model worse than mean baseline"

    def test_feature_importance_available(self, small_dataset):
        """Tree models must expose feature_importances_ attribute."""
        train, val, features, target = small_dataset
        from xgboost import XGBRegressor

        model = XGBRegressor(n_estimators=10, verbosity=0, random_state=42)
        model.fit(train[features], train[target])

        assert hasattr(model, "feature_importances_"), \
            "Model must have feature_importances_ attribute"
        fi = model.feature_importances_
        assert len(fi) == len(features), \
            "feature_importances_ length must match feature count"
        assert (fi >= 0).all(), "All importance scores must be non-negative"


# ── Best model JSON tests ──────────────────────────────────────────────────────

class TestBestModelArtifact:

    def test_best_model_json_exists(self):
        """models/best_model.json must exist after training."""
        path = ROOT / "models" / "best_model.json"
        assert path.exists(), "best_model.json not found — run train.py first"

    def test_best_model_json_structure(self):
        """best_model.json must have required fields."""
        import json
        path = ROOT / "models" / "best_model.json"
        if not path.exists():
            pytest.skip("best_model.json not found")
        data = json.loads(path.read_text())
        assert "model_name" in data, "Missing 'model_name' key"
        assert "val_rmse"   in data, "Missing 'val_rmse' key"
        assert isinstance(data["val_rmse"], float), "val_rmse must be a float"

    def test_best_model_rmse_reasonable(self):
        """Best model RMSE must be within a sane range for this dataset."""
        import json
        path = ROOT / "models" / "best_model.json"
        if not path.exists():
            pytest.skip("best_model.json not found")
        data  = json.loads(path.read_text())
        rmse_v = data["val_rmse"]
        assert 100 < rmse_v < 5000, \
            f"RMSE {rmse_v} outside expected range — possible training failure"
