import time
import pytest
from fastapi.testclient import TestClient
from backend.api import app
from backend import prediction_worker as worker
from data.storage import initialize_input, initialize_results, database, latest_prediction
from data.ingestion import ingest_once


def setup_databases(tmp_path, monkeypatch):
    monkeypatch.setenv('DEMO_DATA_DIR', str(tmp_path))
    initialize_input()
    initialize_results()


def test_pipeline_and_read_only_api(tmp_path, monkeypatch):
    setup_databases(tmp_path, monkeypatch)
    now = time.time()
    assert ingest_once(1800, now - 1) == 0
    assert ingest_once(1800, now) == 1
    assert worker.run_prediction_cycle(1800, 1800, now) == 1
    with database('incoming.sqlite3', readonly=True) as db:
        assert db.execute('SELECT COUNT(*) FROM schedule').fetchone()[0] == 0
    before = latest_prediction()
    with TestClient(app) as client:
        for _ in range(2):
            result = client.get('/predictions/latest').json()
            assert result['state'] == 1
            assert result['delay'] == 0
            assert result['status'] == 'ready'
            assert result['predicted_at']
    assert latest_prediction() == before
    with database('results.sqlite3', readonly=True) as db:
        assert db.execute('SELECT COUNT(*) FROM predictions').fetchone()[0] == 1
    monkeypatch.setenv('PREDICTION_MAX_AGE_SECONDS', '0.000001')
    with TestClient(app) as client:
        assert client.get('/predictions/latest').json()['status'] == 'stale'


def test_retention_and_counter_survives_restart(tmp_path, monkeypatch):
    setup_databases(tmp_path, monkeypatch)
    ingest_once(10, 100)
    worker.run_prediction_cycle(10, 10, 100)
    initialize_input()
    assert ingest_once(10, 120) == 1
    worker.run_prediction_cycle(10, 10, 120)
    with database('incoming.sqlite3', readonly=True) as db:
        assert db.execute('SELECT COUNT(*) FROM validated_events').fetchone()[0] == 1
    with database('results.sqlite3', readonly=True) as db:
        assert db.execute('SELECT COUNT(*) FROM features').fetchone()[0] == 1
        assert db.execute('SELECT COUNT(*) FROM predictions').fetchone()[0] == 1
    assert worker.run_prediction_cycle(10, 10, 140) is None
    assert latest_prediction() is None


def test_failure_does_not_publish_partial_result(tmp_path, monkeypatch):
    setup_databases(tmp_path, monkeypatch)
    ingest_once(100, 100)
    worker.run_prediction_cycle(100, 100, 100)
    def fail(*args):
        raise RuntimeError('model failed')
    monkeypatch.setattr(worker, 'predict', fail)
    with pytest.raises(RuntimeError):
        worker.run_prediction_cycle(100, 100, 101)
    assert latest_prediction()['predicted_at'] == 100
    with database('results.sqlite3', readonly=True) as db:
        assert db.execute('SELECT COUNT(*) FROM features').fetchone()[0] == 1
