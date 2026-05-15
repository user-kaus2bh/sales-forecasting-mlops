# Sales Forecasting MLOps

An end-to-end MLOps project for retail sales forecasting — built across 4 days, covering data pipelines, experiment tracking, model monitoring, and production deployment.

## Project structure

```
sales-forecasting-mlops/
├── data/
│   ├── raw/                  # Raw generated data (DVC tracked)
│   ├── processed/            # Cleaned + feature-engineered splits
│   └── external/             # Any external data sources
├── notebooks/
│   └── eda.ipynb             # Phase 1: Exploratory data analysis
├── src/
│   ├── generate_data.py      # Synthetic data generator
│   ├── data_pipeline.py      # Clean + feature engineer + split
│   ├── train.py              # Phase 2: Model training
│   └── evaluate.py           # Phase 2: Evaluation metrics
├── pipelines/
│   └── train_flow.py         # Phase 3: Prefect orchestration flow
├── app/
│   ├── main.py               # Phase 4: FastAPI prediction endpoint
│   └── dashboard.py          # Phase 4: Streamlit dashboard
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_train.py
│   └── test_api.py
├── configs/
│   ├── data_config.yaml
│   └── model_config.yaml
├── models/                   # Saved model artifacts
├── .github/workflows/        # CI/CD GitHub Actions
├── dvc.yaml                  # DVC pipeline stages
├── Dockerfile
├── requirements.txt
└── README.md
```

## Phases

| Day | Phase | What gets built |
|-----|-------|----------------|
| 1 | Data Pipeline & EDA | Data generation, cleaning, feature engineering, EDA |
| 2 | Model Training | XGBoost/LightGBM, MLflow tracking, hyperparameter tuning |
| 3 | MLOps Infra | Prefect flows, Evidently monitoring, unit tests |
| 4 | Deployment | FastAPI, Docker, Streamlit, GitHub Actions CI/CD |

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/your-username/sales-forecasting-mlops.git
cd sales-forecasting-mlops
pip install -r requirements.txt

# 2. Generate data and run pipeline
python src/generate_data.py
python src/data_pipeline.py

# 3. Run EDA notebook
jupyter notebook notebooks/eda.ipynb

# 4. Train model (Phase 2)
python src/train.py

# 5. Start API (Phase 4)
uvicorn app.main:app --reload
```

## Tech stack

- **Data**: Pandas, NumPy, DVC
- **Modelling**: XGBoost, LightGBM, scikit-learn, Optuna
- **Tracking**: MLflow
- **Orchestration**: Prefect
- **Monitoring**: Evidently
- **API**: FastAPI + Uvicorn
- **Dashboard**: Streamlit
- **CI/CD**: GitHub Actions
- **Containerisation**: Docker

## Dataset

Synthetic retail sales dataset (Rossmann-inspired) with:
- 10 stores × 4 years daily data
- Features: promotions, holidays, store type, competition distance
- Engineered features: lag values, rolling means, cyclical encodings

## Author

Built as a MLOps  project.
