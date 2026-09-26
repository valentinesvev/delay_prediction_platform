"""Проверки ML-сервиса: контракт API, горизонт 10–15 минут, деградация, latency, работа с базой.

Требуют обученных артефактов в ``artifacts/model`` (иначе тесты пропускаются).
"""

from pathlib import Path

import pytest

MODEL_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "model"
pytest.importorskip("catboost")
pytest.importorskip("lightgbm")
pytest.importorskip("xgboost")
pytestmark = pytest.mark.skipif(not (MODEL_DIR / "meta.json").exists(), reason="нет обученной модели")

T0 = 1767700800.0  # 2026-01-06 12:00:00 UTC


def _state(with_telemetry=True, cur_dev=60.0, gap=0.0):
    # прямой маршрут: остановка каждые 90 с, ~400 м
    plan = [{"stop_id": 1000 + i, "time_begin": T0 - 1800 + 90 * i,
             "lon": 37.50 + 0.006 * i, "lat": 55.75} for i in range(60)]
    tel = []
    if with_telemetry:
        for k in range(100):
            t = T0 - 1500 + 15 * k - gap
            pos = (t - 60 - (T0 - 1800)) / 90  # идём с опозданием 60 с
            tel.append({"t": t, "lon": 37.50 + 0.006 * pos, "lat": 55.75, "speed": 18.0, "valid": True})
    return {"tr_id": 1, "T": T0, "cur_dev_s": cur_dev, "plan": plan, "telemetry": tel}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from ml.service import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_contract_and_horizon(client):
    r = client.post("/predict", json=_state())
    assert r.status_code == 200, r.text
    body = r.json()
    lead = body["target_time_plan"] - body["T"]
    assert 600 < lead <= 900                      # цель строго в окне T+10…15 мин
    lo, hi = body["interval_s"]
    assert lo <= body["predicted_delay_s"] <= hi
    assert body["mode"] == "hint"
    # ML — только регрессия: ни цвета, ни причин, ни вероятности в ответе нет
    assert not {"risk", "causes", "p_late"} & set(body)


def test_stream_mode_without_hint(client):
    s = _state(cur_dev=None)
    body = client.post("/predict", json=s).json()
    assert body["mode"] == "stream"


def test_degraded_when_no_telemetry(client):
    body = client.post("/predict", json=_state(with_telemetry=False)).json()
    assert body["degraded"] is True                # сервис не падает, работает по последнему состоянию


def test_no_target_in_window(client):
    s = _state()
    s["plan"] = s["plan"][:5]
    assert client.post("/predict", json=s).status_code == 422


def test_batch_latency(client):
    r = client.post("/predict/batch", json=[_state() for _ in range(50)]).json()
    assert len(r["predictions"]) == 50
    assert r["total_ms"] / 50 < 200                # < 0.2 с на ТС


def test_response_has_model_and_timestamp(client):
    import time
    body = client.post("/predict", json=_state()).json()
    assert body["model_used"] in {"ensemble", "boosting", "catboost", "torch", "baseline"}
    assert abs(body["predicted_at"] - time.time()) < 60
    assert body["fallback_reason"] is None


def test_model_type_switch():
    from ml.predictor import DelayPredictor
    from ml.feature_engineering import VehiclePlan, VehicleTrack
    s = _state()
    plan = VehiclePlan.from_arrays([x["stop_id"] for x in s["plan"]], [x["time_begin"] for x in s["plan"]],
                                   [x["lon"] for x in s["plan"]], [x["lat"] for x in s["plan"]])
    tel = s["telemetry"]
    track = VehicleTrack.from_arrays([x["t"] for x in tel], [x["lon"] for x in tel], [x["lat"] for x in tel],
                                     [x["speed"] for x in tel], [True] * len(tel))
    st = {"tr_id": 1, "plan": plan, "track": track, "T": T0, "cur_dev_s": 60.0}
    for mt in ("ensemble", "boosting", "catboost", "torch", "baseline"):
        r = DelayPredictor(MODEL_DIR, mt).predict_states([st])[0]
        assert r.model_used in (mt, "boosting")  # ensemble без torch честно пишет boosting
        if mt == "baseline":
            assert r.predicted_delay_s == 60.0


def test_fallback_to_baseline_on_model_error(client, monkeypatch):
    from ml import service
    p = service.get_predictor()

    def boom(*a, **k):
        raise RuntimeError("model crashed")

    monkeypatch.setattr(p, "predict_frame", boom)
    body = client.post("/predict", json=_state(cur_dev=75.0)).json()
    assert body["model_used"] == "baseline"
    assert body["predicted_delay_s"] == 75.0
    assert "model crashed" in body["fallback_reason"]
