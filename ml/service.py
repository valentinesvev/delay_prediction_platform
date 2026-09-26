"""ML-сервис: HTTP-обёртка над :class:`ml.predictor.DelayPredictor`.

Отдельный контейнер/процесс (не монолит с Backend). Backend держит по каждому
ТС окно телеметрии (последние ~25 минут) и плановое расписание, раз в N секунд
отправляет состояние сюда и получает прогноз + риск + причину.

Запуск::

    uvicorn ml.service:app --host 0.0.0.0 --port 8001

Swagger: ``http://localhost:8001/docs``.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml.feature_engineering import VehiclePlan, VehicleTrack, select_target
from ml.predictor import DEFAULT_MODEL_DIR, DEFAULT_MODEL_TYPE, DelayPredictor, baseline_prediction

TimeLike = Union[float, str]


def to_unix(t: TimeLike) -> float:
    """Unix-секунды или ISO-строка (без зоны трактуется как UTC, как в датасете)."""
    if isinstance(t, (int, float)):
        return float(t)
    dt = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


# ---------------------------------------------------------------- схемы API
class PlanStop(BaseModel):
    """Плановая остановка из расписания (``schedule_plan.csv``)."""

    stop_id: int = Field(..., description="tt_action_item_id")
    time_begin: TimeLike = Field(..., description="плановое прибытие (unix-сек или ISO)")
    lon: float
    lat: float
    manual_fill: bool = False


class TelemetryPoint(BaseModel):
    """Навигационная точка (ячейка NDTP G6CellNav00 после декодирования)."""

    t: TimeLike = Field(..., description="event_time (unix-сек или ISO)")
    lon: Optional[float] = None
    lat: Optional[float] = None
    speed: Optional[float] = Field(None, description="км/ч")
    valid: bool = True


class VehicleState(BaseModel):
    """Состояние ТС на момент прогноза ``T``."""

    tr_id: int
    T: TimeLike = Field(..., description="момент прогноза; телеметрия после T игнорируется")
    cur_dev_s: Optional[float] = Field(None, description="задержка на последней пройденной остановке, если известна")
    target_stop_id: Optional[int] = Field(None, description="если не задан — первая остановка в окне T+10…15 мин")
    plan: list[PlanStop] = Field(..., description="план ТС на весь день (или минимум с начала текущего рейса до T+30 мин)")
    telemetry: list[TelemetryPoint] = Field(default_factory=list, description="окно телеметрии, последние 30 минут до T")


class PredictionOut(BaseModel):
    tr_id: int
    T: float
    target_stop_id: int
    target_time_plan: float
    predicted_arrival: float
    predicted_delay_s: float
    interval_s: tuple[float, float] = Field(..., description="интервал прогноза задержки 10–90%, сек")
    mode: str = Field(..., description="hint — с cur_dev_s, stream — только телеметрия")
    degraded: bool = Field(..., description="нет свежей телеметрии: прогноз по последнему состоянию")
    inference_ms: float
    model_used: str = Field(..., description="ensemble | boosting | catboost | torch | baseline")
    fallback_reason: Optional[str] = Field(None, description="почему ответил бейзлайн (если ответил)")
    predicted_at: float = Field(..., description="когда сделан прогноз, unix-сек (время сервера)")


class SkippedOut(BaseModel):
    tr_id: int
    reason: str


class BatchOut(BaseModel):
    predictions: list[PredictionOut]
    skipped: list[SkippedOut]
    total_ms: float


# ---------------------------------------------------------------- приложение
_predictor: Optional[DelayPredictor] = None
_load_error: Optional[str] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Прогрев: модель загружается при старте, чтобы первый запрос был быстрым."""
    get_predictor()
    yield


app = FastAPI(title="Delay Prediction — ML service", version="1.0.0", lifespan=lifespan,
              description="Регрессия: прогноз задержки ТС (сек) на первой остановке в окне T+10…15 мин. "
                          "Ансамбль CatBoost + LightGBM + XGBoost + PyTorch GRU; при сбое — бейзлайн "
                          "«прогноз = текущая задержка». Источник данных: запрос (`/predict*`) или база "
                          "(`/predict/from-db`, фоновый воркер `python -m ml.worker`).")


def get_predictor() -> Optional[DelayPredictor]:
    """Загрузить модель один раз. Если не удалось — ``None``, и сервис отвечает бейзлайном."""
    global _predictor, _load_error
    if _predictor is None and _load_error is None:
        try:
            _predictor = DelayPredictor(Path(os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR)),
                                        os.getenv("MODEL_TYPE", DEFAULT_MODEL_TYPE))
        except Exception as e:
            _load_error = f"модель не загрузилась: {type(e).__name__}: {e}"
            print("[ml.service]", _load_error)
    return _predictor


def _run(states: list[dict]):
    """Прогноз с защитой: любая ошибка → бейзлайн «прогноз = текущая задержка»."""
    p = get_predictor()
    if p is not None:
        try:
            return p.predict_states(states)
        except Exception as e:
            reason = f"сбой предиктора: {type(e).__name__}: {e}"
    else:
        reason = _load_error or "модель не загружена"
    return [baseline_prediction(st, None, "hint" if st.get("cur_dev_s") is not None else "stream", reason)
            for st in states]


def _to_state(s: VehicleState) -> dict:
    """Pydantic-состояние → структуры ML-модуля; проверка наличия цели в окне."""
    plan = VehiclePlan.from_arrays(
        [x.stop_id for x in s.plan], [to_unix(x.time_begin) for x in s.plan],
        [x.lon for x in s.plan], [x.lat for x in s.plan], [float(x.manual_fill) for x in s.plan])
    tel = s.telemetry
    track = VehicleTrack.from_arrays(
        [to_unix(x.t) for x in tel], [np.nan if x.lon is None else x.lon for x in tel],
        [np.nan if x.lat is None else x.lat for x in tel],
        [np.nan if x.speed is None else x.speed for x in tel], [x.valid for x in tel])
    T = to_unix(s.T)
    tidx = None
    if s.target_stop_id is not None:
        hit = np.where(plan.stop_ids == s.target_stop_id)[0]
        if not len(hit):
            raise ValueError("target_stop_id нет в плане")
        tidx = int(hit[0])
    elif select_target(plan, T) is None:
        raise ValueError("нет плановой остановки в окне T+10…15 мин")
    return {"tr_id": s.tr_id, "plan": plan, "track": track, "T": T, "cur_dev_s": s.cur_dev_s, "target_idx": tidx}


def _out(r) -> PredictionOut:
    return PredictionOut(**{**r.__dict__, "predicted_arrival": r.predicted_arrival})


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if _predictor is not None else "degraded", "model_loaded": _predictor is not None,
            "model_type": _predictor.model_type if _predictor else "baseline",
            "nn_loaded": bool(_predictor and _predictor.uses_nn), "load_error": _load_error}


@app.get("/model/info")
def model_info() -> dict:
    """Версия модели, признаки, метрики лидерборда."""
    p = get_predictor()
    if p is None:
        raise HTTPException(status_code=503, detail=_load_error)
    info = {"model_type": p.model_type, "nn_weights": p.w_nn, "ensemble": p.meta["ensemble"], "n_features": len(p.features), "features": p.features}
    for mode in ("hint", "stream"):
        f = p.model_dir / f"leaderboard_{mode}.csv"
        if f.exists():
            import pandas as pd
            info[f"leaderboard_{mode}"] = pd.read_csv(f).to_dict(orient="records")
    return info


@app.post("/predict", response_model=PredictionOut)
def predict(state: VehicleState) -> PredictionOut:
    """Прогноз для одного ТС."""
    try:
        st = _to_state(state)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return _out(_run([st])[0])


@app.post("/predict/batch", response_model=BatchOut)
def predict_batch(states: list[VehicleState]) -> BatchOut:
    """Прогноз для всего парка за один вызов; ТС без цели в окне попадают в ``skipped``."""
    t0 = time.perf_counter()
    ok, skipped = [], []
    for s in states:
        try:
            ok.append(_to_state(s))
        except ValueError as e:
            skipped.append(SkippedOut(tr_id=s.tr_id, reason=str(e)))
    preds = _run(ok) if ok else []
    return BatchOut(predictions=[_out(r) for r in preds], skipped=skipped,
                    total_ms=round((time.perf_counter() - t0) * 1000, 2))


class DBCycleOut(BaseModel):
    T: Optional[float] = Field(None, description="момент потока, на который сделан прогноз")
    written: int
    predictions: list[PredictionOut]
    skipped: list[SkippedOut]
    timing_ms: dict


@app.post("/predict/from-db", response_model=DBCycleOut)
def predict_from_db(write: bool = True, T: Optional[TimeLike] = None) -> DBCycleOut:
    """Один цикл по базе: прочитать телеметрию и план → прогноз по всем ТС → записать в ``predictions``.

    То же, что делает фоновый воркер ``python -m ml.worker``; удобно для проверки из Swagger.
    ``T`` по умолчанию — последняя точка телеметрии в базе.
    """
    from ml.db import DBConfig, make_engine, run_cycle

    cfg = DBConfig()
    if not cfg.url:
        raise HTTPException(status_code=503, detail="DATABASE_URL не задан")
    try:
        r = run_cycle(get_predictor(), make_engine(cfg), cfg, None if T is None else to_unix(T), write)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"база недоступна: {type(e).__name__}: {e}")
    return DBCycleOut(T=r["T"], written=r["written"], predictions=[_out(p) for p in r["predictions"]],
                      skipped=[SkippedOut(**k) for k in r["skipped"]], timing_ms=r["timing_ms"])
