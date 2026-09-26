"""HTTP API для готовых прогнозов; расчёт модели выполняется в ml.worker."""

import os
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from backend.predictions import latest_for_all, latest_for_vehicle

router = APIRouter()


def _timestamp(value):
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc)
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value.replace(tzinfo=value.tzinfo or timezone.utc).astimezone(timezone.utc)


def _present(row):
    result = dict(row)
    forecast_at = _timestamp(result["t_forecast"])
    predicted_at = _timestamp(result["predicted_at"])
    max_age = float(os.getenv("PREDICTION_MAX_AGE_SECONDS", "90"))
    if max_age <= 0:
        raise ValueError("PREDICTION_MAX_AGE_SECONDS должен быть положительным")
    age = time.time() - min(forecast_at.timestamp(), predicted_at.timestamp())
    result["status"] = "stale" if age > max_age else "ready"
    for key in ("t_forecast", "predicted_at", "target_time_plan", "predicted_arrival"):
        result[key] = _timestamp(result[key]).isoformat()
    result["degraded"] = bool(result["degraded"])
    return result


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/predictions/latest")
def get_latest_predictions():
    try:
        rows = latest_for_all()
        predictions = [_present(row) for row in rows]
        status = ("unavailable" if not predictions else
                  "ready" if any(p["status"] == "ready" for p in predictions) else "stale")
        return {"status": status, "predictions": predictions}
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="База прогнозов недоступна") from exc


@router.get("/predictions/{tr_id}/latest")
def get_latest_for_vehicle(tr_id: int):
    try:
        row = latest_for_vehicle(tr_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Прогноз для ТС не найден")
        return _present(row)
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="База прогнозов недоступна") from exc
