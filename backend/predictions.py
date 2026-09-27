"""Чтение таблицы прогнозов из базы результатов. Подключение не создаёт таблиц."""

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
    from storage.database import results_database_url
    explicit = results_database_url()
    url = explicit or os.getenv("DATABASE_URL")
    if not url:
        return []
    parsed_url = make_url(url)
    if (parsed_url.get_backend_name() == "sqlite" and parsed_url.database
            and parsed_url.database != ":memory:" and not Path(parsed_url.database).exists() and not explicit):
        return []
    from storage.database import existing_engine
    engine = existing_engine(url, readonly=True) if parsed_url.database != ":memory:" else create_engine(url)
    try:
        table = _table()
        with engine.connect() as connection:
            if not explicit and not inspect(connection).has_table(table):
                return []
            return [dict(row) for row in connection.execute(text(sql.format(table=_quote(table))), params or {}).mappings()]
    finally:
        engine.dispose()


def latest_for_all():
    columns = ", ".join(_quote(c) for c in COLUMNS)
    condition, params = _replay_cutoff()
    return _read(f"""
        SELECT {columns} FROM (
            SELECT {columns}, ROW_NUMBER() OVER (
                PARTITION BY tr_id ORDER BY t_forecast DESC, id DESC
            ) AS row_number FROM {{table}} {condition}
        ) AS ranked WHERE row_number = 1 ORDER BY tr_id
    """, params)


def latest_for_vehicle(tr_id):
    columns = ", ".join(_quote(c) for c in COLUMNS)
    condition, params = _replay_cutoff()
    extra = condition.replace('WHERE', 'AND', 1) if condition else ''
    rows = _read(f"SELECT {columns} FROM {{table}} WHERE tr_id = :tr_id "
                 f"{extra} ORDER BY t_forecast DESC, id DESC LIMIT 1", {"tr_id": tr_id, **params})
    return rows[0] if rows else None


def _replay_cutoff():
    from storage import replay
    from storage.database import utc_datetime
    if not replay.enabled():
        return '', {}
    return 'WHERE t_forecast <= :replay_time', {'replay_time': utc_datetime(replay.current_time()).strftime('%Y-%m-%d %H:%M:%S.%f')}
