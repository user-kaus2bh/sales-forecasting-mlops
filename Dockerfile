FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi uvicorn pandas numpy scikit-learn \
    lightgbm xgboost joblib pyyaml scipy mlflow optuna

# Copy code and configs
COPY app/     ./app/
COPY src/     ./src/
COPY configs/ ./configs/

# Copy models folder (only the .json file, not .pkl files)
COPY models/best_model.json ./models/best_model.json

# Create required directories
RUN mkdir -p data/processed/reports data/raw

# Startup script — generates data, trains model, then starts API
COPY start.sh .
RUN chmod +x start.sh

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["./start.sh"]