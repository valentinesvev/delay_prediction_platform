"""Контракт NDTP → общая SQLite → ML → API и границы очистки."""
import json
import struct
import time

import pytest
from fastapi.testclient import TestClient

from backend.api import app
from ndtp_ingestion.db_init import connect, init_db
from ndtp_ingestion.load_schedule import load_schedule
from ndtp_ingestion.server import NPL, NPH, NAV, crc16, decode, save_point, cleanup, stamp


@pytest.fixture
def live_db(tmp_path, monkeypatch):
    monkeypatch.setenv('DATABASE_URL', f'sqlite:///{tmp_path / "live.db"}')
    monkeypatch.setenv('ML_TIME_MODE', 'wall')
    init_db()
    return tmp_path


def frame(now, unit=123, flags=0xe0, speed=20):
    body = bytes([0, 0]) + NAV.pack(int(now), 375000000, 557500000, flags, 100, speed, 20, 90, 0, 0, 10, 1)
    payload = NPH.pack(1, 101, 1, 2) + body
    return NPL.pack(0x7e7e, len(payload), 0, crc16(payload), 2, unit, 0), payload


def test_decode_identity_crc_validation_and_duplicates(live_db):
    now = time.time()
    header, payload = frame(now)
    point = decode(header, payload, {'123': 42}, now)
    assert point[:2] == (42, 123)
    assert point[4:7] == (37.5, 55.75, 20)
    assert save_point(point) == 1
    assert save_point(point) == 0
    with pytest.raises(ValueError, match='CRC'):
        decode(header, payload[:-1] + b'\x02', {'123': 42}, now)
    with pytest.raises(ValueError, match='соответствия'):
        decode(header, payload, {}, now)
    for flags, speed in [(0x60, 20), (0xe0, 121), (0x80, 20)]:
        header, payload = frame(now, flags=flags, speed=speed)
        with pytest.raises(ValueError, match='координаты'):
            decode(header, payload, {'123': 42}, now)
    header, payload = frame(now - 7201)
    with pytest.raises(ValueError, match='Время'):
        decode(header, payload, {'123': 42}, now)


def test_schedule_geom_no_fact_and_idempotency(live_db):
    path = live_db / 'schedule.csv'
    path.write_text('tt_action_item_id,tr_id,time_begin,time_fact_begin,geom,building_address\n'
                    '10,42,2026-09-26 12:00:00,2026-09-26 12:10:00,POINT (37.5 55.75),Остановка\n')
    assert load_schedule(path) == 1
    assert load_schedule(path) == 1
    with connect() as conn:
        rows = conn.execute('SELECT * FROM schedule_plan').fetchall()
        assert len(rows) == 1
        assert rows[0]['geom'] == 'POINT (37.5 55.75)'
        assert 'time_fact_begin' not in rows[0].keys()


def test_real_packet_to_ml_and_fleet(live_db):
    from ml.db import DBConfig, make_engine, run_cycle
    now = time.time()
    for age in (30, 20, 10):
        save_point(decode(*frame(now - age), {'123': 42}, now - age))
    with connect() as conn:
        for offset in range(-900, 1000, 90):
            conn.execute('INSERT INTO schedule_plan VALUES (?, ?, ?, ?, ?, ?)',
                         (offset+1000, 42, stamp(now+offset), False, 'POINT (37.5 55.75)', 'Тестовая остановка'))
    engine = make_engine(DBConfig())
    try:
        result = run_cycle(None, engine, DBConfig(), T=now)
        assert result['written'] == 1
        assert run_cycle(None, engine, DBConfig(), T=now)['written'] == 0
    finally:
        engine.dispose()
    with TestClient(app) as client:
        response = client.get('/vehicles/active')
        assert response.status_code == 200, response.text
        vehicle = response.json()['vehicles'][0]
        assert vehicle['tr_id'] == 42 and vehicle['unit_id'] == 123
        assert vehicle['status'] == 'on_route'
        assert vehicle['current_delay_s'] is not None
        assert vehicle['current_stop']['name'] == 'Тестовая остановка'
        assert vehicle['prediction']['model_used'] == 'baseline'


def test_fleet_shows_vehicle_without_plan_or_prediction(live_db):
    now = time.time()
    save_point(decode(*frame(now), {'123': 42}, now))
    with TestClient(app) as client:
        rows = client.get('/vehicles/active').json()['vehicles']
        assert len(rows) == 1 and rows[0]['status'] == 'no_plan'
        assert rows[0]['current_delay_s'] is None and rows[0]['prediction'] is None
    with connect() as conn:
        conn.execute('UPDATE telemetry SET event_time=?', (stamp(now - 121),))
    with TestClient(app) as client:
        assert client.get('/vehicles/active').json()['vehicles'] == []


def test_cleanup_retention_and_receive_time(live_db):
    from ml.db import DBConfig, make_engine, write_predictions
    from ml.predictor import Prediction
    now = time.time()
    with connect() as conn:
        # Старое событие, недавно полученное: сохраняется по receive_time.
        for tr, event, receive in [(1, now-8000, now-10), (2, now-8000, now-7201)]:
            conn.execute('INSERT INTO telemetry VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                         (tr, tr, stamp(event), stamp(receive), 37.5, 55.75, 0, True))
        for stop, seconds in [(1, -21601), (2, -21599), (3, 3600)]:
            conn.execute('INSERT INTO schedule_plan VALUES (?, ?, ?, ?, ?, ?)',
                         (stop, 1, stamp(now+seconds), False, 'POINT (37.5 55.75)', None))
    engine = make_engine(DBConfig())
    try:
        for i, age in enumerate([86401, 86399]):
            prediction = Prediction(tr_id=i+1, T=now-age, predicted_at=now-age,
                                    target_stop_id=1, target_time_plan=now-age+700,
                                    predicted_delay_s=10, interval_s=(0, 20))
            write_predictions(engine, DBConfig(), [prediction])
    finally:
        engine.dispose()
    assert cleanup(now) == {'telemetry': 1, 'schedule_plan': 1, 'predictions': 1}
    with connect() as conn:
        assert conn.execute('SELECT tr_id FROM telemetry').fetchone()[0] == 1
