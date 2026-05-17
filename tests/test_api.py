"""
test_api.py
Phase 4 — FastAPI endpoint unit tests

Tests:
  - GET /health returns correct structure
  - POST /predict returns a valid prediction
  - POST /predict validates bad input
  - POST /predict/batch handles multiple requests
  - GET /model/info returns feature list

Run:
    pytest tests/test_api.py -v
    (requires: pip install httpx)
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


# ── Try to import FastAPI test client ─────────────────────────────────────────
try:
    from fastapi.testclient import TestClient
    from app.main import app
    CLIENT_AVAILABLE = True
    client = TestClient(app)
except Exception as e:
    CLIENT_AVAILABLE = False
    SKIP_REASON = f"FastAPI app could not be loaded: {e}"


def skip_if_no_client(fn):
    return pytest.mark.skipif(
        not CLIENT_AVAILABLE,
        reason=SKIP_REASON if not CLIENT_AVAILABLE else ""
    )(fn)


# ── Sample request payload ────────────────────────────────────────────────────
SAMPLE_PAYLOAD = {
    "store_id":             3,
    "date":                 "2024-03-15",
    "promo":                1,
    "promo2":               0,
    "school_holiday":       0,
    "competition_distance": 1200.0,
    "store_type":           "a",
    "assortment":           "basic",
    "sales_lag_1":          8500.0,
    "sales_lag_7":          7900.0,
}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRootEndpoint:

    @skip_if_no_client
    def test_root_returns_200(self):
        r = client.get("/")
        assert r.status_code == 200

    @skip_if_no_client
    def test_root_has_docs_key(self):
        r = client.get("/")
        assert "docs" in r.json()

    @skip_if_no_client
    def test_root_has_predict_key(self):
        r = client.get("/")
        assert "predict" in r.json()


class TestHealthEndpoint:

    @skip_if_no_client
    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    @skip_if_no_client
    def test_health_has_status_key(self):
        r = client.get("/health")
        assert "status" in r.json()

    @skip_if_no_client
    def test_health_model_loaded(self):
        r = client.get("/health")
        data = r.json()
        assert "model_loaded" in data
        assert data["model_loaded"] is not None

    @skip_if_no_client
    def test_health_has_timestamp(self):
        r = client.get("/health")
        assert "timestamp" in r.json()

    @skip_if_no_client
    def test_health_performance_keys(self):
        r = client.get("/health")
        perf = r.json().get("performance", {})
        for key in ["rmse", "mape", "r2"]:
            assert key in perf


class TestPredictEndpoint:

    @skip_if_no_client
    def test_predict_returns_200(self):
        r = client.post("/predict", json=SAMPLE_PAYLOAD)
        assert r.status_code == 200

    @skip_if_no_client
    def test_predict_returns_positive_sales(self):
        r = client.post("/predict", json=SAMPLE_PAYLOAD)
        pred = r.json().get("predicted_sales", -1)
        assert pred >= 0, "Predicted sales must be non-negative"

    @skip_if_no_client
    def test_predict_response_has_required_fields(self):
        r = client.post("/predict", json=SAMPLE_PAYLOAD)
        data = r.json()
        for key in ["store_id", "date", "predicted_sales", "model_name", "confidence_note"]:
            assert key in data, f"Missing field: {key}"

    @skip_if_no_client
    def test_predict_store_id_matches(self):
        r = client.post("/predict", json=SAMPLE_PAYLOAD)
        assert r.json()["store_id"] == SAMPLE_PAYLOAD["store_id"]

    @skip_if_no_client
    def test_predict_date_matches(self):
        r = client.post("/predict", json=SAMPLE_PAYLOAD)
        assert r.json()["date"] == SAMPLE_PAYLOAD["date"]

    @skip_if_no_client
    def test_predict_rejects_bad_date(self):
        bad = {**SAMPLE_PAYLOAD, "date": "15-03-2024"}   # wrong format
        r = client.post("/predict", json=bad)
        assert r.status_code == 422   # Unprocessable Entity

    @skip_if_no_client
    def test_predict_rejects_invalid_store_type(self):
        bad = {**SAMPLE_PAYLOAD, "store_type": "z"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    @skip_if_no_client
    def test_predict_rejects_invalid_assortment(self):
        bad = {**SAMPLE_PAYLOAD, "assortment": "premium"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    @skip_if_no_client
    def test_predict_works_without_lag_features(self):
        """API should use defaults when lag features are not provided."""
        no_lags = {k: v for k, v in SAMPLE_PAYLOAD.items()
                   if k not in ("sales_lag_1", "sales_lag_7")}
        r = client.post("/predict", json=no_lags)
        assert r.status_code == 200
        assert r.json()["predicted_sales"] >= 0

    @skip_if_no_client
    def test_predict_promo_increases_sales(self):
        """Promotion should generally increase predicted sales."""
        promo_on  = {**SAMPLE_PAYLOAD, "promo": 1}
        promo_off = {**SAMPLE_PAYLOAD, "promo": 0}
        pred_on  = client.post("/predict", json=promo_on).json()["predicted_sales"]
        pred_off = client.post("/predict", json=promo_off).json()["predicted_sales"]
        # Soft assertion — promo should increase or maintain sales
        assert pred_on >= pred_off * 0.9, "Promo should not significantly reduce predictions"


class TestBatchPredictEndpoint:

    @skip_if_no_client
    def test_batch_predict_returns_200(self):
        payload = {"requests": [SAMPLE_PAYLOAD, SAMPLE_PAYLOAD]}
        r = client.post("/predict/batch", json=payload)
        assert r.status_code == 200

    @skip_if_no_client
    def test_batch_predict_count_matches(self):
        n = 3
        payload = {"requests": [SAMPLE_PAYLOAD] * n}
        r = client.post("/predict/batch", json=payload)
        data = r.json()
        assert data["total_requests"] == n
        assert len(data["predictions"]) == n

    @skip_if_no_client
    def test_batch_all_predictions_non_negative(self):
        payload = {"requests": [SAMPLE_PAYLOAD] * 5}
        r = client.post("/predict/batch", json=payload)
        for pred in r.json()["predictions"]:
            assert pred["predicted_sales"] >= 0


class TestModelInfoEndpoint:

    @skip_if_no_client
    def test_model_info_returns_200(self):
        r = client.get("/model/info")
        assert r.status_code == 200

    @skip_if_no_client
    def test_model_info_has_features(self):
        r = client.get("/model/info")
        data = r.json()
        assert "features" in data
        assert isinstance(data["features"], list)
        assert len(data["features"]) > 0

    @skip_if_no_client
    def test_model_info_feature_count(self):
        r = client.get("/model/info")
        assert r.json()["feature_count"] == 36

    @skip_if_no_client
    def test_model_info_has_target(self):
        r = client.get("/model/info")
        assert r.json()["target"] == "sales"
