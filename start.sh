#!/bin/bash
set -e

echo "==> Generating data..."
python src/generate_data.py

echo "==> Running data pipeline..."
python src/data_pipeline.py

echo "==> Training model..."
export MLFLOW_ALLOW_FILE_STORE=true
python src/train.py --model xgboost

echo "==> Running monitoring..."
python src/monitoring.py || echo '{"drift":{},"quality":{}}' > data/processed/reports/monitoring_summary.json

echo "==> Starting API..."
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
