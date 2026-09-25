import time
import pytest
from fastapi.testclient import TestClient
from backend.api import app
from backend import prediction_worker as worker
from backend import prediction
from data.storage import initialize_input, initialize_results, database, latest_prediction
from data.ingestion import ingest_once


def setup_databases(tmp_path, monkeypatch):
    monkeypatch.setenv('DEMO_DATA_DIR', str(tmp_path))
    monkeypatch.setattr(prediction, 'MODEL_TYPE', 'baseline')
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
        raise RuntimeError('save failed')
    monkeypatch.setattr(worker, 'save_prediction', fail)
    with pytest.raises(RuntimeError):
        worker.run_prediction_cycle(100, 100, 101)
    assert latest_prediction()['predicted_at'] == 100
    with database('results.sqlite3', readonly=True) as db:
        assert db.execute('SELECT COUNT(*) FROM features').fetchone()[0] == 1


@pytest.mark.parametrize('scenario', ['ok', 'no_data', 'features_error', 'predict_error'])
def test_cycle_timing_log(tmp_path, monkeypatch, caplog, scenario):
    from types import SimpleNamespace

    setup_databases(tmp_path, monkeypatch)
    if scenario != 'no_data':
        ingest_once(100, 100)
    clock = [0.0]
    monkeypatch.setattr(worker, 'time', SimpleNamespace(perf_counter=lambda: clock[0]))
    original_features = worker.generate_features
    original_cleanup = worker.cleanup_results

    def features(event):
        clock[0] += 0.125
        if scenario == 'features_error':
            raise RuntimeError('features failed')
        return original_features(event)

    def predict(*args):
        clock[0] += 0.250
        if scenario == 'predict_error':
            raise RuntimeError('predict failed')
        return {'state': args[0]['state'], 'delay': args[0]['delay']}

    def cleanup(*args):
        clock[0] += 0.500
        return original_cleanup(*args)

    monkeypatch.setattr(worker, 'generate_features', features)
    monkeypatch.setattr(worker, 'predict', predict)
    monkeypatch.setattr(worker, 'cleanup_results', cleanup)
    with caplog.at_level('INFO', logger=worker.__name__):
        if scenario.endswith('_error'):
            with pytest.raises(RuntimeError):
                worker.run_prediction_cycle(100, 100, 100)
            assert latest_prediction() is None
        else:
            worker.run_prediction_cycle(100, 100, 100)
    records = [record for record in caplog.records if record.name == worker.__name__ and record.msg.startswith('cycle ')]
    assert len(records) == 1
    status, buses, features_ms, predict_ms, total_ms, state = records[0].args
    assert status == ('error' if scenario.endswith('_error') else 'no_data' if scenario == 'no_data' else 'ok')
    assert buses == (0 if scenario == 'no_data' else 1)
    assert features_ms == (0 if scenario == 'no_data' else 125)
    assert predict_ms == (250 if scenario in ('ok', 'predict_error') else 0)
    assert total_ms == features_ms + predict_ms + 500
    assert state == (0 if scenario == 'ok' else None)

@pytest.mark.parametrize('available', [True, False])
def test_model_or_baseline_prediction(tmp_path, monkeypatch, caplog, available):
    setup_databases(tmp_path, monkeypatch)
    ingest_once(100, 100)
    monkeypatch.setattr(prediction, 'MODEL_TYPE', 'catboost')
    if available:
        monkeypatch.setattr(prediction, 'predict_catboost', lambda features: {'state': 42, 'delay': 12.5})
        def unexpected_baseline(features):
            pytest.fail('baseline must not run when model succeeds')
        monkeypatch.setattr(prediction, 'predict_baseline', unexpected_baseline)
    with caplog.at_level('INFO'):
        assert worker.run_prediction_cycle(100, 100, 100) == (42 if available else 0)
    result = latest_prediction()
    assert result['state'] == (42 if available else 0)
    assert result['delay'] == (12.5 if available else 0)
    assert result['data_as_of'] == 100
    assert ('source=catboost' if available else 'source=baseline') in caplog.text


def test_baseline_failure_preserves_previous_prediction(tmp_path, monkeypatch):
    setup_databases(tmp_path, monkeypatch)
    ingest_once(100, 100)
    worker.run_prediction_cycle(100, 100, 100)
    before = latest_prediction()
    def fail(features):
        raise RuntimeError('baseline failed')
    monkeypatch.setattr(prediction, 'predict_baseline', fail)
    with pytest.raises(RuntimeError, match='baseline failed'):
        worker.run_prediction_cycle(100, 100, 101)
    assert latest_prediction() == before


@pytest.mark.parametrize('model_type', ['catboost', 'torch'])
@pytest.mark.parametrize('fails', [False, True])
def test_predict_dispatch_and_fallback(monkeypatch, model_type, fails):
    monkeypatch.setattr(prediction, 'MODEL_TYPE', model_type)
    features = {'state': 7, 'delay': 2}
    def model(values):
        values['state'] = 99
        if fails:
            raise RuntimeError('inference failed')
        return {'state': 42, 'delay': 5}
    monkeypatch.setattr(prediction, 'predict_' + model_type, model)
    assert prediction.predict(features) == (features if fails else {'state': 42, 'delay': 5})
    assert features['state'] == 7


@pytest.mark.parametrize('model_type', ['baseline', 'unknown'])
def test_predict_without_model(monkeypatch, model_type):
    monkeypatch.setattr(prediction, 'MODEL_TYPE', model_type)
    assert prediction.predict({'state': 7, 'delay': 2}) == {'state': 7, 'delay': 2}
