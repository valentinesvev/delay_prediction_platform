"""Интеграционный контракт между таблицей ML и FastAPI бэкенда."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select

from backend.api import app
from ml.db import DBConfig, predictions_table


@pytest.fixture
def prediction_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'shared.db'}")
    monkeypatch.setenv("PREDICTION_MAX_AGE_SECONDS", "90")
    monkeypatch.delenv("ML_PREDICTIONS_TABLE", raising=False)
    engine = create_engine(f"sqlite:///{tmp_path / 'shared.db'}")
    table = predictions_table(DBConfig())
    table.metadata.create_all(engine)
    yield engine, table
    engine.dispose()


def add_prediction(engine, table, tr_id, age=10, delay=42, forecast_offset=0):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    forecast = now - timedelta(seconds=age + forecast_offset)
    with engine.begin() as connection:
        connection.execute(table.insert().values(
            tr_id=tr_id, t_forecast=forecast,
            predicted_at=now - timedelta(seconds=age),
            target_stop_id=tr_id * 100,
            target_time_plan=forecast + timedelta(minutes=12),
            predicted_delay_s=delay,
            predicted_arrival=forecast + timedelta(minutes=12, seconds=delay),
            interval_lo_s=delay - 20, interval_hi_s=delay + 20,
            model_used="ensemble", degraded=False,
        ))


def test_missing_database_does_not_create_it(tmp_path, monkeypatch):
    path = tmp_path / "absent.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")
    with TestClient(app) as client:
        assert client.get("/predictions/latest").json() == {"status": "unavailable", "predictions": []}
        assert client.get("/predictions/5/latest").status_code == 404
    assert not path.exists()


def test_latest_per_vehicle_freshness_and_no_writes(prediction_db):
    engine, table = prediction_db
    add_prediction(engine, table, 1, age=60, delay=2)
    add_prediction(engine, table, 1, age=10, delay=15)
    add_prediction(engine, table, 2, age=120, delay=-3)
    with engine.connect() as connection:
        count_before = len(connection.execute(select(table)).all())
    with TestClient(app) as client:
        response = client.get("/predictions/latest")
        assert response.status_code == 200
        rows = response.json()["predictions"]
        assert [row["tr_id"] for row in rows] == [1, 2]
        assert rows[0]["predicted_delay_s"] == 15
        assert rows[0]["status"] == "ready"
        assert rows[0]["target_time_plan"].endswith("+00:00")
        assert rows[1]["status"] == "stale"
        assert client.get("/predictions/1/latest").json()["predicted_delay_s"] == 15
        assert client.get("/predictions/3/latest").status_code == 404
    with engine.connect() as connection:
        assert len(connection.execute(select(table)).all()) == count_before


def test_older_forecast_is_stale_even_if_just_calculated(prediction_db):
    engine, table = prediction_db
    add_prediction(engine, table, 9, age=1, forecast_offset=300)
    with TestClient(app) as client:
        assert client.get("/predictions/9/latest").json()["status"] == "stale"


def test_second_write_of_same_cycle_does_not_duplicate(prediction_db):
    from ml.db import write_predictions
    from ml.predictor import Prediction

    engine, table = prediction_db
    now = datetime.now(timezone.utc).timestamp()
    prediction = Prediction(
        tr_id=7, T=now, target_stop_id=70, target_time_plan=now + 700,
        predicted_delay_s=25, interval_s=(5, 45),
    )
    cfg = DBConfig()
    assert write_predictions(engine, cfg, [prediction]) == 1
    assert write_predictions(engine, cfg, [prediction]) == 0
    with engine.connect() as connection:
        assert len(connection.execute(select(table)).all()) == 1
