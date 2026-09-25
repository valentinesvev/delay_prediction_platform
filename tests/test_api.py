from fastapi.testclient import TestClient
from backend.api import app
from data.storage import database, initialize_results
import pytest


def test_api_read_only_when_empty(tmp_path, monkeypatch):
    monkeypatch.setenv('DEMO_DATA_DIR', str(tmp_path / 'absent'))
    with TestClient(app) as client:
        assert client.get('/health').json() == {'status': 'ok'}
        assert client.get('/').status_code == 200
        assert client.get('/predictions/latest').json() == {'status': 'unavailable', 'state': None}
        assert client.get('/predictions/average').json() == {'status': 'unavailable', 'average': None, 'count': 0}
        assert client.post('/predict', json={'current_delay': 1}).status_code == 404
    assert not (tmp_path / 'absent').exists()


@pytest.mark.parametrize('values, expected', [
    ([], {'status': 'unavailable', 'average': None, 'count': 0}),
    ([0], {'status': 'ready', 'average': 0, 'count': 1}),
    ([2, 5], {'status': 'ready', 'average': 3.5, 'count': 2}),
    ([1000, 2000, *range(10)], {'status': 'ready', 'average': 4.5, 'count': 10}),
])
def test_average_latest_predictions(tmp_path, monkeypatch, values, expected):
    monkeypatch.setenv('DEMO_DATA_DIR', str(tmp_path))
    initialize_results()
    with database('results.sqlite3') as db:
        db.executemany('INSERT INTO predictions(state, delay, data_as_of, predicted_at) VALUES (?, 0, 100, 100)',
                       [(value,) for value in values])
    before = (tmp_path / 'results.sqlite3').read_bytes()
    with TestClient(app) as client:
        response = client.get('/predictions/average')
        assert response.status_code == 200
        assert response.json() == expected
    assert (tmp_path / 'results.sqlite3').read_bytes() == before


def test_average_before_worker_creates_tables(tmp_path, monkeypatch):
    monkeypatch.setenv('DEMO_DATA_DIR', str(tmp_path))
    with database('results.sqlite3'):
        pass
    with TestClient(app) as client:
        assert client.get('/predictions/average').json() == {
            'status': 'unavailable', 'average': None, 'count': 0,
        }
