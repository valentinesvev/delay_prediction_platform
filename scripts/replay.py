"""Advance ML, analytics and the API cursor together through historical inputs."""
import fcntl
import logging
import math
import os
import signal
import threading
import uuid

from analytics.results import initialize as initialize_analytics
from analytics.worker import run_cycle as analytics_cycle
from ml.db import DBConfig, run_cycle as ml_cycle
from ml.results_db import initialize as initialize_predictions
from ml.worker import load_predictor
from storage import replay
from storage.database import existing_engine

log = logging.getLogger('replay')


def create_session(request):
    first, last = replay.bounds()
    start = request['start']
    if not first <= start <= last:
        raise ValueError('Replay start is outside the input history')
    session = uuid.uuid4().hex
    cfg = DBConfig(results_url=replay.session_url(session), time_mode='stream')
    initialize_predictions(cfg)
    initialize_analytics(cfg)
    # Publish only after both schemas exist. Older session files are preserved.
    state = dict(session=session, request_id=request['id'], first=first, last=last,
                 cursor=start, status='initializing', error=None)
    replay.write_json('state.json', state)
    return state


def advance(state, predictor, step_s):
    target = state['cursor'] if state['status'] == 'initializing' else min(state['last'], state['cursor'] + step_s)
    # Persist the pending step so a process restart retries that same step.
    target = state.get('pending', target)
    pending = {**state, 'pending': target, 'error': None}
    replay.write_json('state.json', pending)
    cfg = DBConfig(results_url=replay.session_url(state['session']), time_mode='stream')
    engine = existing_engine(cfg.url, readonly=True)
    try:
        ml_cycle(predictor, engine, cfg, target)
        analytics_cycle(target, cfg)
    except Exception as exc:
        replay.write_json('state.json', {**pending, 'error': f'{type(exc).__name__}: {exc}'})
        raise
    finally:
        engine.dispose()
    completed = {**state, 'cursor': target, 'status': 'finished' if target >= state['last'] else 'running', 'error': None}
    completed.pop('pending', None)
    replay.write_json('state.json', completed)
    return completed


def main():
    logging.basicConfig(level=logging.INFO)
    if not replay.enabled():
        raise ValueError('Set DATA_TIME_MODE=stream and REPLAY_STATE_DIR')
    step = float(os.getenv('REPLAY_STEP_S', '10'))
    interval = float(os.getenv('REPLAY_TICK_S', '1'))
    if not (math.isfinite(step) and 0 < step <= 60 and math.isfinite(interval) and interval > 0):
        raise ValueError('REPLAY_STEP_S must be in (0, 60], REPLAY_TICK_S must be positive')
    replay.directory().mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    with (replay.directory() / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        predictor = load_predictor()
        while not stop.is_set():
            try:
                state = replay.read_json('state.json')
                request = replay.read_json('request.json')
                if not state and not request:
                    first, _ = replay.bounds()
                    request = dict(id=uuid.uuid4().hex, start=first)
                if request and (not state or request['id'] != state['request_id']):
                    state = create_session(request)
                    log.info('Replay session %s starts at %s', state['session'], replay.iso(state['cursor']))
                if state['status'] != 'finished':
                    state = advance(state, predictor, step)
                    log.info('Replay time: %s', replay.iso(state['cursor']))
            except Exception:
                log.exception('Replay step failed; will retry')
            stop.wait(interval)


if __name__ == '__main__':
    main()
