"""Независимый worker: python -m backend.prediction_worker."""
import logging
import os
import time
from backend.prediction import predict
from data.storage import database, directory, initialize_results


logger = logging.getLogger(__name__)

def generate_features(event):
    return {'state': event['state'], 'delay': event['delay'], 'data_as_of': event['event_at']}


def save_prediction(db, prediction, features, now):
    db.execute('INSERT INTO predictions(state, delay, data_as_of, predicted_at) VALUES (?, ?, ?, ?)',
               (prediction['state'], prediction['delay'], features['data_as_of'], now))


def cleanup_results(retention_seconds, now):
    with database('results.sqlite3') as db:
        db.execute('DELETE FROM features WHERE created_at < ?', (now - retention_seconds,))
        db.execute('DELETE FROM predictions WHERE predicted_at < ?', (now - retention_seconds,))


def run_prediction_cycle(retention_seconds, input_retention_seconds, now=None):
    started = time.perf_counter()
    buses = 0
    features_ms = predict_ms = 0.0
    state = None
    status = 'no_data'
    now = time.time() if now is None else now
    try:
        if not (directory() / 'incoming.sqlite3').exists():
            return None
        with database('incoming.sqlite3', readonly=True) as db:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='validated_events'").fetchone():
                return None
            event = db.execute('SELECT * FROM validated_events WHERE event_at >= ? ORDER BY id DESC LIMIT 1', (now - input_retention_seconds,)).fetchone()
        if event is None:
            return None
        buses = 1
        phase_started = time.perf_counter()
        try:
            features = generate_features(event)
        finally:
            features_ms = (time.perf_counter() - phase_started) * 1000
        phase_started = time.perf_counter()
        try:
            prediction = predict(features)
        finally:
            predict_ms = (time.perf_counter() - phase_started) * 1000
        with database('results.sqlite3') as db:
            db.execute('INSERT INTO features(state, delay, data_as_of, created_at) VALUES (?, ?, ?, ?)',
                       (features['state'], features['delay'], features['data_as_of'], now))
            save_prediction(db, prediction, features, now)
        state = prediction['state']
        status = 'ok'
        return state
    except Exception:
        status = 'error'
        raise
    finally:
        try:
            cleanup_results(retention_seconds, now)
        except Exception:
            status = 'error'
            raise
        finally:
            logger.info(
                'cycle status=%s buses=%d features_ms=%.3f predict_ms=%.3f total_ms=%.3f state=%s',
                status, buses, features_ms, predict_ms,
                (time.perf_counter() - started) * 1000, state,
            )


def main():
    interval = float(os.getenv('PREDICTION_INTERVAL_SECONDS', '3'))
    retention = float(os.getenv('RESULT_RETENTION_SECONDS', '1800'))
    input_retention = float(os.getenv('INPUT_RETENTION_SECONDS', '1800'))
    if min(interval, retention, input_retention) <= 0:
        raise ValueError('Интервал и сроки хранения должны быть положительными')
    initialize_results()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s prediction worker: %(message)s')
    while True:
        started = time.monotonic()
        try:
            run_prediction_cycle(retention, input_retention)
        except Exception:
            logging.exception('Ошибка расчёта; следующий цикл будет повторён')
        elapsed = time.monotonic() - started
        # Пропускаем истёкшие интервалы, не запускаем перекрывающиеся расчёты.
        time.sleep(interval - elapsed % interval)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
