"""Общая SQLite-база обработчика, ML и API."""
import os
import sqlite3
from pathlib import Path
from sqlalchemy.engine import make_url


def db_path():
    url = make_url(os.environ['DATABASE_URL'])
    if url.get_backend_name() != 'sqlite' or not url.database or url.database == ':memory:':
        raise ValueError('Обработчик поддерживает файловую SQLite-базу')
    return Path(url.database).resolve()


def connect():
    conn = sqlite3.connect(db_path(), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    from ml.db import DBConfig, make_engine, predictions_table
    db_path().parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS telemetry (
                tr_id INTEGER NOT NULL, unit_id INTEGER NOT NULL,
                event_time DATETIME NOT NULL, receive_time DATETIME NOT NULL,
                lon REAL NOT NULL, lat REAL NOT NULL, speed REAL NOT NULL,
                location_valid BOOLEAN NOT NULL, PRIMARY KEY (tr_id, event_time));
            CREATE TABLE IF NOT EXISTS schedule_plan (
                tt_action_item_id INTEGER NOT NULL, tr_id INTEGER NOT NULL,
                time_begin DATETIME NOT NULL, manual_fill BOOLEAN NOT NULL,
                geom TEXT NOT NULL, building_address TEXT,
                PRIMARY KEY (tr_id, tt_action_item_id, time_begin));
            CREATE INDEX IF NOT EXISTS ix_tel_time ON telemetry(event_time);
            CREATE INDEX IF NOT EXISTS ix_tel_receive ON telemetry(receive_time);
            CREATE INDEX IF NOT EXISTS ix_schedule_time ON schedule_plan(time_begin);
        ''')
        for table, required in [('telemetry', {'unit_id'}), ('schedule_plan', {'geom', 'building_address'})]:
            fields = {r['name'] for r in conn.execute(f'PRAGMA table_info({table})')}
            if not required <= fields:
                raise ValueError(f'{table}: несовместимая старая схема; укажите новую базу через --db')
    cfg = DBConfig()
    engine = make_engine(cfg)
    try:
        predictions_table(cfg).metadata.create_all(engine)
        with connect() as conn:
            conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS uq_predictions_vehicle_time_stop '
                         'ON predictions(tr_id, t_forecast, target_stop_id)')
    finally:
        engine.dispose()


if __name__ == '__main__':
    init_db()
