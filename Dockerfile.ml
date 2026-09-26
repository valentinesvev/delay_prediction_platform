# ML-сервис: инференс ансамбля (CatBoost + LightGBM + XGBoost + PyTorch GRU), Swagger на :8001/docs
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_DIR=/app/artifacts/model
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-ml.txt .
# CPU-сборка PyTorch (~200 МБ вместо ~2 ГБ с CUDA); для GPU-хоста уберите --index-url
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-ml.txt
COPY ml/ ml/
COPY artifacts/model/ artifacts/model/

EXPOSE 8001
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD curl -fs http://localhost:8001/health || exit 1
CMD ["uvicorn", "ml.service:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
