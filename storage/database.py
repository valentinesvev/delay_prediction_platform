"""Connections and common settings, without model or analytics imports."""
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url


def data_time_mode():
    """Keep old local commands working; new deployments use DATA_TIME_MODE."""
    return os.getenv('DATA_TIME_MODE') or os.getenv('ML_TIME_MODE', 'stream')


def results_database_url():
    from storage.replay import results_url
    return results_url()


@dataclass
class DatabaseConfig:
    url: str = field(default_factory=lambda: os.getenv('DATABASE_URL', ''))
    results_url: str = field(default_factory=results_database_url)
    pred_table: str = field(default_factory=lambda: os.getenv('ML_PREDICTIONS_TABLE', 'predictions'))
    time_mode: str = field(default_factory=data_time_mode)


def create_database_engine(url):
    if not url:
        raise ValueError('Database URL is required')
    return create_engine(url, pool_pre_ping=True)


def existing_engine(url, *, readonly=False):
    """SQLite URI mode prevents accidental creation, including a check/open race."""
    parsed = make_url(url)
    if parsed.get_backend_name() == 'sqlite':
        if not parsed.database or parsed.database == ':memory:':
            raise ValueError('Database must be a persistent SQLite file')
        uri = Path(parsed.database).resolve().as_uri()
        parsed = parsed.set(database=uri, query={**parsed.query, 'mode': 'ro' if readonly else 'rw', 'uri': 'true'})
    return create_database_engine(parsed)


def utc_datetime(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).replace(tzinfo=None)


def timestamp_parameter(engine, seconds):
    value = utc_datetime(seconds)
    return value.strftime('%Y-%m-%d %H:%M:%S.%f') if engine.dialect.name == 'sqlite' else value


def quote_identifier(name):
    if not name or '\x00' in name:
        raise ValueError('Invalid SQL identifier')
    return '"' + name.replace('"', '""') + '"'
