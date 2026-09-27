"""Replay clocks, isolated runs and HTTP reads using real separate SQLite files."""
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.engine import make_url

from analytics.routes import import_route
from backend.api import app
from ml.db import DBConfig, current_T
from ndtp_ingestion.db_init import init_db
from scripts import replay as coordinator
from storage import replay
from storage.database import existing_engine

BASE = datetime(2026, 1, 6, 7, 23, tzinfo=timezone.utc)


def stamp(seconds):
    return (BASE+timedelta(seconds=seconds)).replace(tzinfo=None).strftime('%Y-%m-%d %H:%M:%S.%f')


@pytest.fixture
def history(tmp_path, monkeypatch):
    source = tmp_path / 'input.db'
    output = tmp_path / 'results.db'
    monkeypatch.setenv('DATABASE_URL', f'sqlite:///{source}')
    monkeypatch.setenv('RESULTS_DATABASE_URL', f'sqlite:///{output}')
    monkeypatch.setenv('DATA_TIME_MODE', 'stream')
    monkeypatch.delenv('REPLAY_STATE_DIR', raising=False)
    init_db()
    with sqlite3.connect(source) as c:
        for i in range(9):
            c.execute('INSERT INTO telemetry VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                      (42, 142, stamp(i*15), stamp(i*15), 37.501+i*.0008, 55.75, 12, 1))
        for i, seconds in enumerate([600, 700, 800]):
            c.execute('INSERT INTO schedule_plan VALUES (?, ?, ?, ?, ?, ?)',
                      (i+1, 42, stamp(seconds), 0, f'POINT ({37.51+i*.01} 55.75)', f'Stop {i}'))
    import_route(dict(route_id='r', route_name='Test', is_demo=True,
        directions=[dict(direction_id='a', stops=[dict(stop_id='a', name='A', lat=55.75, lon=37.5),
                                                 dict(stop_id='b', name='B', lat=55.75, lon=37.51)])],
        assignments=[dict(tr_id=42, direction_id='a', valid_from=BASE.isoformat(),
                          valid_to=(BASE+timedelta(hours=1)).isoformat())]), f'sqlite:///{source}')
    monkeypatch.setenv('REPLAY_STATE_DIR', str(tmp_path / 'clock'))
    return source, output


def start(seconds=0):
    return coordinator.create_session(dict(id=f'request-{seconds}', start=BASE.timestamp()+seconds))


def test_beginning_seek_and_state_persistence(history):
    source, original_results = history
    # Flush setup's WAL first so subsequent byte comparisons are meaningful.
    with closing(sqlite3.connect(source)) as c:
        c.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    before = source.read_bytes()
    with TestClient(app) as client:
        assert client.get('/vehicles/active').status_code == 503
        assert not original_results.exists()
        status = client.get('/replay/status').json()
        assert status['current'] is None
        assert status['first'] == BASE.isoformat()
        response = client.post('/replay/start', json={})
        assert response.status_code == 202
        state = coordinator.create_session(replay.read_json('request.json'))
        state = coordinator.advance(state, None, 10)
        assert state['cursor'] == BASE.timestamp()
        # A process restart reads this same cursor, not MAX(telemetry.event_time).
        restored = replay.read_json('state.json')
        engine = existing_engine(DBConfig().url, readonly=True)
        try:
            assert current_T(engine, DBConfig()) == BASE.timestamp()
        finally:
            engine.dispose()
        advanced = coordinator.advance(restored, None, 10)
        assert advanced['cursor'] == BASE.timestamp()+10
        snapshot = client.get('/vehicles/active')
        assert snapshot.status_code == 200
        assert snapshot.json()['reference_time'] == (BASE+timedelta(seconds=10)).isoformat()
        assert snapshot.json()['replay']['session'] == state['session']
        assert len(snapshot.json()['vehicles']) == 1
        assert client.get('/analytics/segments').json()['status'] == 'ready'
    assert source.read_bytes() == before
    assert not original_results.exists()


def test_new_run_isolates_predictions_and_analytics(history):
    old = coordinator.advance(start(60), None, 10)
    with TestClient(app) as client:
        predictions = client.get('/predictions/latest').json()['predictions']
        assert predictions and all(p['status'] == 'ready' for p in predictions)
        assert client.get('/analytics/segments').json()['segments']
        old_url = replay.results_url()
        assert client.post('/replay/start', json={'at': BASE.isoformat()}).status_code == 202
        new = coordinator.create_session(replay.read_json('request.json'))
        assert new['session'] != old['session']
        assert client.get('/predictions/latest').json()['predictions'] == []
        assert client.get('/analytics/segments').json()['segments'] == []
        assert client.get('/analytics/alerts').json()['alerts'] == []
        with replay.scope(old):
            assert replay.results_url() == old_url
            assert replay.current_time() == BASE.timestamp()+60
        with sqlite3.connect(make_url(old_url).database) as c:
            assert c.execute('SELECT COUNT(*) FROM predictions').fetchone()[0] > 0


def test_failed_step_does_not_publish_future_and_retry_is_idempotent(history, monkeypatch):
    state = coordinator.advance(start(60), None, 10)
    real_analytics = coordinator.analytics_cycle
    def unavailable(*args):
        raise RuntimeError('Temporary results write failure')
    monkeypatch.setattr(coordinator, 'analytics_cycle', unavailable)
    with pytest.raises(RuntimeError):
        coordinator.advance(state, None, 10)
    pending = replay.read_json('state.json')
    assert pending['cursor'] == BASE.timestamp()+60
    assert pending['pending'] == BASE.timestamp()+70
    with TestClient(app) as client:
        forecasts = client.get('/predictions/latest').json()['predictions']
        assert forecasts
        assert all(datetime.fromisoformat(p['t_forecast']).timestamp() <= pending['cursor'] for p in forecasts)
    monkeypatch.setattr(coordinator, 'analytics_cycle', real_analytics)
    finished = coordinator.advance(pending, None, 10)
    assert finished['cursor'] == BASE.timestamp()+70
    assert 'pending' not in finished and finished['error'] is None
    with sqlite3.connect(make_url(replay.results_url()).database) as c:
        assert c.execute('SELECT COUNT(*) FROM predictions').fetchone()[0] == c.execute(
            'SELECT COUNT(*) FROM (SELECT DISTINCT tr_id,t_forecast,target_stop_id FROM predictions)').fetchone()[0]


def test_time_validation_and_end_of_history(history):
    with TestClient(app) as client:
        for invalid in ['2026-01-06T07:23:00', 'not-a-time', '2025-01-01T00:00:00Z']:
            assert client.post('/replay/start', json={'at': invalid}).status_code == 422
        assert replay.read_json('request.json') is None
        assert client.post('/replay/start', json={'at': '2026-01-06T10:24:50+03:00'}).status_code == 202
        state = coordinator.create_session(replay.read_json('request.json'))
        state = coordinator.advance(state, None, 60)
        state = coordinator.advance(state, None, 60)
        assert state['cursor'] == BASE.timestamp()+120
        assert state['status'] == 'finished'


def test_missing_session_database_is_explicit(history):
    state = start()
    from pathlib import Path
    path = Path(make_url(replay.session_url(state['session'])).database)
    path.unlink()
    with TestClient(app) as client:
        assert client.get('/vehicles/active').status_code == 503
        assert client.get('/analytics/segments').status_code == 503
    assert not path.exists()


def test_live_mode_rejects_replay_control(history, monkeypatch):
    monkeypatch.setenv('DATA_TIME_MODE', 'wall')
    with TestClient(app) as client:
        assert client.get('/replay/status').json() == {'enabled': False}
        assert client.post('/replay/start', json={}).status_code == 409
