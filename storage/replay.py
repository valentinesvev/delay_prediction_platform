"""Shared replay cursor and control requests. No model imports or new DB tables."""
import json
import math
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.engine import make_url

_UNSET = object()
_snapshot = ContextVar('replay_snapshot', default=_UNSET)


def enabled():
    from storage.database import data_time_mode
    return bool(os.getenv('REPLAY_STATE_DIR')) and data_time_mode() == 'stream'


def directory():
    if not enabled():
        raise ValueError('Historical replay is not configured')
    return Path(os.environ['REPLAY_STATE_DIR'])


def read_json(name):
    try:
        return json.loads((directory() / name).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None


def write_json(name, value):
    root = directory()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=root, delete=False) as f:
        temporary = Path(f.name)
        try:
            json.dump(value, f, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
            os.replace(temporary, root / name)
        finally:
            temporary.unlink(missing_ok=True)


def snapshot():
    if not enabled():
        return None
    value = _snapshot.get()
    return read_json('state.json') if value is _UNSET else value


@contextmanager
def scope(value):
    token = _snapshot.set(value)
    try:
        yield
    finally:
        _snapshot.reset(token)


def current_time():
    value = snapshot()
    if not value:
        if enabled():
            raise ValueError('Historical replay is initializing')
        return None
    result = float(value['cursor'])
    if not math.isfinite(result):
        raise ValueError('Invalid replay cursor')
    return result


def session_url(session):
    if not re.fullmatch(r'[0-9a-f]{32}', session):
        raise ValueError('Invalid replay session')
    base = make_url(os.environ.get('RESULTS_DATABASE_URL', ''))
    if base.get_backend_name() != 'sqlite' or not base.database or base.database == ':memory:':
        raise ValueError('Replay requires a persistent RESULTS_DATABASE_URL SQLite file')
    path = Path(base.database).resolve().with_name(f'replay-{session}.db')
    return base.set(database=str(path)).render_as_string(hide_password=False)


def results_url():
    if not enabled():
        return os.getenv('RESULTS_DATABASE_URL', '')
    value = snapshot()
    if not value:
        raise ValueError('Historical replay is initializing')
    return session_url(value['session'])


def iso(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def bounds():
    from sqlalchemy import text
    from storage.database import existing_engine, quote_identifier
    engine = existing_engine(os.environ['DATABASE_URL'], readonly=True)
    try:
        table = quote_identifier(os.getenv('ML_TELEMETRY_TABLE', 'telemetry'))
        column = quote_identifier(os.getenv('ML_TEL_TIME', 'event_time'))
        with engine.connect() as conn:
            first, last = conn.execute(text(f'SELECT MIN({column}), MAX({column}) FROM {table}')).one()
        if first is None:
            raise ValueError('В исторической базе нет телеметрии')
        def timestamp(value):
            if not isinstance(value, datetime):
                value = datetime.fromisoformat(str(value))
            return value.replace(tzinfo=value.tzinfo or timezone.utc).timestamp()
        return timestamp(first), timestamp(last)
    finally:
        engine.dispose()


def request_start(at=None):
    first, last = bounds()
    if at is None:
        start = first
    else:
        value = datetime.fromisoformat(at.replace('Z', '+00:00'))
        if value.tzinfo is None:
            raise ValueError('Укажите часовой пояс времени, например Z или +03:00')
        start = value.timestamp()
    if not first <= start <= last:
        raise ValueError(f'Время должно быть между {iso(first)} и {iso(last)}')
    request = dict(id=uuid.uuid4().hex, start=start)
    write_json('request.json', request)
    return request
