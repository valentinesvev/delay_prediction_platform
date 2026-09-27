"""Read completed segment calculations from Base 2 for the dispatcher."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from sqlalchemy import func, inspect, select
from sqlalchemy.exc import SQLAlchemyError

from analytics.results import alerts, segment_state
from storage.database import existing_engine, results_database_url, utc_datetime
from storage import replay


router = APIRouter(prefix="/analytics", tags=["segment analytics"])


def _serialize(row):
    result = dict(row)
    for key, value in result.items():
        if isinstance(value, datetime):
            result[key] = value.replace(tzinfo=value.tzinfo or timezone.utc).isoformat()
    return result


def _read(table, statement):
    url = results_database_url()
    if not url:
        raise HTTPException(503, "База результатов не настроена")
    try:
        engine = existing_engine(url, readonly=True)
        try:
            with engine.connect() as conn:
                if not inspect(conn).has_table(table.name):
                    raise HTTPException(503, "Таблицы аналитики ещё не созданы")
                return [_serialize(row) for row in conn.execute(statement).mappings()]
        finally:
            engine.dispose()
    except SQLAlchemyError as exc:
        raise HTTPException(503, "База результатов недоступна") from exc


@router.get("/segments")
def latest_segments():
    latest_query = select(segment_state.c.segment_id, func.max(segment_state.c.calculated_at).label('latest_at'))
    if replay.enabled():
        latest_query = latest_query.where(segment_state.c.calculated_at <= utc_datetime(replay.current_time()))
    latest = latest_query.group_by(segment_state.c.segment_id).subquery()
    statement = (select(segment_state)
                 .join(latest, (segment_state.c.segment_id == latest.c.segment_id)
                       & (segment_state.c.calculated_at == latest.c.latest_at))
                 .order_by(segment_state.c.route_name, segment_state.c.direction_id,
                           segment_state.c.segment_id))
    rows = _read(segment_state, statement)
    return {"status": "ready" if rows else "empty", "segments": rows}


@router.get("/alerts")
def active_alerts():
    statement = select(alerts).where(alerts.c.status == 'ACTIVE')
    if replay.enabled():
        statement = statement.where(alerts.c.updated_at <= utc_datetime(replay.current_time()))
    rows = _read(alerts, statement.order_by(alerts.c.updated_at.desc()))
    return {"status": "ready" if rows else "empty", "alerts": rows}
