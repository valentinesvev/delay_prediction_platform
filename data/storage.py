"""Две базы учебного сервиса. API открывает базу результатов только для чтения."""
from contextlib import contextmanager
from pathlib import Path
import os
import sqlite3


def directory():
    return Path(os.getenv('DEMO_DATA_DIR', str(Path(__file__).resolve().parent / 'local' / 'worker_demo'))).resolve()


@contextmanager
def database(name, readonly=False):
    path = directory() / name
    if readonly:
        connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=10)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_input():
    with database('incoming.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS validated_events (id INTEGER PRIMARY KEY AUTOINCREMENT, state INTEGER NOT NULL, delay REAL NOT NULL, event_at REAL NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS schedule (id INTEGER PRIMARY KEY, scheduled_at REAL)')
        db.execute('CREATE TABLE IF NOT EXISTS generator_state (id INTEGER PRIMARY KEY CHECK(id=1), next_value INTEGER NOT NULL)')
        db.execute('INSERT OR IGNORE INTO generator_state VALUES (1, 0)')


def initialize_results():
    with database('results.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS features (id INTEGER PRIMARY KEY AUTOINCREMENT, state INTEGER NOT NULL, delay REAL NOT NULL, data_as_of REAL NOT NULL, created_at REAL NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, state INTEGER NOT NULL, delay REAL NOT NULL, data_as_of REAL NOT NULL, predicted_at REAL NOT NULL)')


def latest_prediction():
    if not (directory() / 'results.sqlite3').exists():
        return None
    with database('results.sqlite3', readonly=True) as db:
        # Worker может ещё создавать таблицы при первом запуске.
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='predictions'").fetchone():
            return None
        row = db.execute('SELECT state, delay, data_as_of, predicted_at FROM predictions ORDER BY id DESC LIMIT 1').fetchone()
        return dict(row) if row else None


def average_latest_predictions():
    empty = {'average': None, 'count': 0}
    if not (directory() / 'results.sqlite3').exists():
        return empty
    with database('results.sqlite3', readonly=True) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='predictions'").fetchone():
            return empty
        row = db.execute('''
            SELECT AVG(state) AS average, COUNT(*) AS count
            FROM (SELECT state FROM predictions ORDER BY id DESC LIMIT 10)
        ''').fetchone()
        return dict(row)
