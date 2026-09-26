"""Чтение таблицы прогнозов из общей базы. Подключение не создаёт таблиц."""

import os
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url


COLUMNS = (
    "id", "tr_id", "t_forecast", "predicted_at", "target_stop_id",
    "target_time_plan", "predicted_delay_s", "predicted_arrival",
    "interval_lo_s", "interval_hi_s", "model_used", "fallback_reason", "degraded",
)


def _table():
    name = os.getenv("ML_PREDICTIONS_TABLE", "predictions")
    if not name or "\x00" in name:
        raise ValueError("Некорректное имя таблицы прогнозов")
    return name


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def _read(sql, params=None):
    url = os.getenv("DATABASE_URL")
    if not url:
        return []
    parsed_url = make_url(url)
    if (parsed_url.get_backend_name() == "sqlite" and parsed_url.database
            and parsed_url.database != ":memory:" and not Path(parsed_url.database).exists()):
        return []
    engine = create_engine(url, pool_pre_ping=True)
    try:
        table = _table()
        with engine.connect() as connection:
            if not inspect(connection).has_table(table):
                return []
            return [dict(row) for row in connection.execute(text(sql.format(table=_quote(table))), params or {}).mappings()]
    finally:
        engine.dispose()


def latest_for_all():
    columns = ", ".join(_quote(c) for c in COLUMNS)
    return _read(f"""
        SELECT {columns} FROM (
            SELECT {columns}, ROW_NUMBER() OVER (
                PARTITION BY tr_id ORDER BY t_forecast DESC, id DESC
            ) AS row_number FROM {{table}}
        ) AS ranked WHERE row_number = 1 ORDER BY tr_id
    """)


def latest_for_vehicle(tr_id):
    columns = ", ".join(_quote(c) for c in COLUMNS)
    rows = _read(f"SELECT {columns} FROM {{table}} WHERE tr_id = :tr_id "
                 "ORDER BY t_forecast DESC, id DESC LIMIT 1", {"tr_id": tr_id})
    return rows[0] if rows else None
