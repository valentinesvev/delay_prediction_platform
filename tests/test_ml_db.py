"""ML ↔ база: воркер читает телеметрию и план из БД и пишет прогнозы в ``predictions``."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("catboost")

MODEL_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "model"
T0 = pd.Timestamp("2026-01-06 12:00:00")


@pytest.fixture()
def db(tmp_path):
    from sqlalchemy import create_engine

    url = f"sqlite:///{tmp_path / 'test.db'}"
    eng = create_engine(url)
    t_plan = [T0 - pd.Timedelta(seconds=1800 - 90 * i) for i in range(60)]
    pd.DataFrame({
        "tt_action_item_id": 1000 + np.arange(60), "tr_id": 1, "time_begin": t_plan,
        "geom": [f"POINT ({37.5 + 0.006 * i} 55.75)" for i in range(60)], "manual_fill": False,
    }).to_sql("schedule_plan", eng, index=False)
    ts = [T0 - pd.Timedelta(seconds=1500 - 15 * k) for k in range(101)]
    pos = [((t - T0).total_seconds() + 1800 - 60) / 90 for t in ts]  # опаздывает на 60 с
    pd.DataFrame({
        "tr_id": [1] * 101 + [2], "event_time": ts + [T0], "receive_time": ts + [T0],
        "location_valid": True, "lon": [37.5 + 0.006 * p for p in pos] + [37.6], "lat": 55.75, "speed": 18.0,
    }).to_sql("telemetry", eng, index=False)
    return url


def test_cycle_reads_db_and_writes_predictions(db, monkeypatch):
    from ml.db import DBConfig, make_engine, run_cycle
    from ml.predictor import DelayPredictor

    monkeypatch.setenv("DATABASE_URL", db)
    cfg = DBConfig()
    eng = make_engine(cfg)
    r = run_cycle(DelayPredictor(MODEL_DIR), eng, cfg)
    assert r["T"] == pytest.approx(T0.timestamp())            # T = последняя точка в базе
    assert r["written"] == 1 and r["predictions"][0].tr_id == 1
    assert r["skipped"] == [{"tr_id": 2, "reason": "нет планового расписания"}]
    p = r["predictions"][0]
    assert 600 < p.target_time_plan - p.T <= 900              # горизонт 10–15 мин
    rows = pd.read_sql("select * from predictions", eng)
    assert len(rows) == 1 and rows.loc[0, "model_used"] in ("ensemble", "boosting")


def test_cycle_without_model_writes_baseline(db, monkeypatch):
    from ml.db import DBConfig, make_engine, run_cycle

    monkeypatch.setenv("DATABASE_URL", db)
    cfg = DBConfig()
    r = run_cycle(None, make_engine(cfg), cfg)
    assert r["predictions"][0].model_used == "baseline"
    assert r["predictions"][0].fallback_reason


def test_service_from_db_endpoint(db, monkeypatch):
    from fastapi.testclient import TestClient
    from ml.service import app

    monkeypatch.setenv("DATABASE_URL", db)
    with TestClient(app) as c:
        body = c.post("/predict/from-db").json()
    assert body["written"] == 1 and body["predictions"][0]["tr_id"] == 1
