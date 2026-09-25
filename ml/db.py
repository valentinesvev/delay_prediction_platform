"""Работа ML-модуля с базой: чтение телеметрии и плана, запись прогнозов.

Backend наполняет базу из входных данных (NDTP-поток / исторический датасет),
ML-модуль по расписанию читает из неё окно телеметрии и плановое расписание,
считает прогнозы и пишет их в таблицу ``predictions``. Оттуда их берут
модуль риска/причин и дашборд.

Имена таблиц и колонок настраиваются переменными окружения (значения по
умолчанию совпадают с колонками CSV хакатона), поэтому под реальную схему
базы код менять не нужно — только ``.env``:

=========================  ===========================  =====================================
Переменная                 По умолчанию                 Смысл
=========================  ===========================  =====================================
``DATABASE_URL``           —                            SQLAlchemy URL, напр.
                                                        ``postgresql+psycopg2://u:p@db:5432/x``
``ML_TELEMETRY_TABLE``     ``telemetry``                таблица телеметрии
``ML_TEL_TR_ID``           ``tr_id``                    ID ТС
``ML_TEL_TIME``            ``event_time``               время точки
``ML_TEL_RECEIVE``         ``receive_time``             время приёма ('' — нет колонки)
``ML_TEL_LON/LAT``         ``lon`` / ``lat``            координаты, градусы
``ML_TEL_SPEED``           ``speed``                    скорость, км/ч
``ML_TEL_VALID``           ``location_valid``           валидность координат
``ML_SCHEDULE_TABLE``      ``schedule_plan``            плановое расписание (**без факта!**)
``ML_SCH_STOP``            ``tt_action_item_id``        ID планового прибытия (остановки)
``ML_SCH_TR_ID``           ``tr_id``                    ID ТС
``ML_SCH_TIME``            ``time_begin``               плановое время прибытия
``ML_SCH_GEOM``            ``geom``                     ``POINT (lon lat)`` ('' — брать LON/LAT)
``ML_SCH_LON/LAT``         ``lon`` / ``lat``            координаты остановки, если нет geom
``ML_SCH_MANUAL``          ``manual_fill``              флаг ('' — нет колонки)
``ML_CURDEV_TABLE``        ``''`` (нет)                 необязательно: текущая задержка ТС
``ML_CURDEV_TR_ID/VALUE``  ``tr_id`` / ``cur_dev_s``    колонки этой таблицы
``ML_PREDICTIONS_TABLE``   ``predictions``              куда писать прогнозы
``ML_TIME_MODE``           ``stream``                   ``stream`` — T = последняя точка в
                                                        базе (реплей), ``wall`` — текущее время
``ML_WINDOW_S``            ``2700``                     окно телеметрии, сек (45 мин)
=========================  ===========================  =====================================

Все времена в базе трактуются как **UTC** (как в датасете).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from ml.feature_engineering import VehiclePlan, VehicleTrack, select_target

PLAN_BACK_S = 6 * 3600   # план берём за 6 ч назад (начало текущего рейса) …
PLAN_FWD_S = 2 * 3600    # … и на 2 ч вперёд (цель в T+10…15 мин и конец рейса)


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class DBConfig:
    """Настройки доступа к базе; по умолчанию — из переменных окружения."""

    url: str = field(default_factory=lambda: _env("DATABASE_URL", ""))
    tel_table: str = field(default_factory=lambda: _env("ML_TELEMETRY_TABLE", "telemetry"))
    tel_tr: str = field(default_factory=lambda: _env("ML_TEL_TR_ID", "tr_id"))
    tel_time: str = field(default_factory=lambda: _env("ML_TEL_TIME", "event_time"))
    tel_receive: str = field(default_factory=lambda: _env("ML_TEL_RECEIVE", "receive_time"))
    tel_lon: str = field(default_factory=lambda: _env("ML_TEL_LON", "lon"))
    tel_lat: str = field(default_factory=lambda: _env("ML_TEL_LAT", "lat"))
    tel_speed: str = field(default_factory=lambda: _env("ML_TEL_SPEED", "speed"))
    tel_valid: str = field(default_factory=lambda: _env("ML_TEL_VALID", "location_valid"))
    sch_table: str = field(default_factory=lambda: _env("ML_SCHEDULE_TABLE", "schedule_plan"))
    sch_stop: str = field(default_factory=lambda: _env("ML_SCH_STOP", "tt_action_item_id"))
    sch_tr: str = field(default_factory=lambda: _env("ML_SCH_TR_ID", "tr_id"))
    sch_time: str = field(default_factory=lambda: _env("ML_SCH_TIME", "time_begin"))
    sch_geom: str = field(default_factory=lambda: _env("ML_SCH_GEOM", "geom"))
    sch_lon: str = field(default_factory=lambda: _env("ML_SCH_LON", "lon"))
    sch_lat: str = field(default_factory=lambda: _env("ML_SCH_LAT", "lat"))
    sch_manual: str = field(default_factory=lambda: _env("ML_SCH_MANUAL", "manual_fill"))
    curdev_table: str = field(default_factory=lambda: _env("ML_CURDEV_TABLE", ""))
    curdev_tr: str = field(default_factory=lambda: _env("ML_CURDEV_TR_ID", "tr_id"))
    curdev_value: str = field(default_factory=lambda: _env("ML_CURDEV_VALUE", "cur_dev_s"))
    pred_table: str = field(default_factory=lambda: _env("ML_PREDICTIONS_TABLE", "predictions"))
    time_mode: str = field(default_factory=lambda: _env("ML_TIME_MODE", "stream"))
    window_s: float = field(default_factory=lambda: float(_env("ML_WINDOW_S", "2700")))


def make_engine(cfg: DBConfig):
    """SQLAlchemy engine с проверкой соединения перед использованием (переживает рестарт БД)."""
    from sqlalchemy import create_engine

    if not cfg.url:
        raise RuntimeError("DATABASE_URL не задан")
    return create_engine(cfg.url, pool_pre_ping=True, future=True)


def _unix(s: pd.Series) -> np.ndarray:
    """Колонка времени (datetime/строка/unix) → unix-секунды; наивное время = UTC."""
    if pd.api.types.is_numeric_dtype(s):
        return s.to_numpy(dtype=float)
    dt = pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
    return ((dt - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def _ts(t: float) -> datetime:
    """unix → наивный datetime UTC (так он сравнивается с колонками без зоны)."""
    return datetime.fromtimestamp(t, tz=timezone.utc).replace(tzinfo=None)


def _param(engine, t: float):
    """Параметр времени для WHERE. В SQLite время хранится строкой — сравниваем в том же формате."""
    d = _ts(t)
    return d.strftime("%Y-%m-%d %H:%M:%S.%f") if engine.dialect.name == "sqlite" else d


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def current_T(engine, cfg: DBConfig) -> Optional[float]:
    """Момент прогноза: последняя точка телеметрии в базе (stream) или текущее время (wall)."""
    if cfg.time_mode == "wall":
        return time.time()
    from sqlalchemy import text

    with engine.connect() as c:
        v = c.execute(text(f"SELECT MAX({_q(cfg.tel_time)}) FROM {_q(cfg.tel_table)}")).scalar()
    if v is None:
        return None
    return float(_unix(pd.Series([v]))[0])


def load_states(engine, cfg: DBConfig, T: float) -> tuple[list[dict], list[dict]]:
    """Прочитать окно телеметрии и план, собрать состояния ТС для :class:`DelayPredictor`.

    Returns:
        ``(states, skipped)``: состояния ТС, у которых есть план и цель в окне
        T+10…15 мин, и список пропущенных ТС с причиной.
    """
    from sqlalchemy import text

    tcols = [cfg.tel_tr, cfg.tel_time, cfg.tel_lon, cfg.tel_lat, cfg.tel_speed, cfg.tel_valid]
    if cfg.tel_receive:
        tcols.append(cfg.tel_receive)
    tel_sql = (f"SELECT {', '.join(map(_q, tcols))} FROM {_q(cfg.tel_table)} "
               f"WHERE {_q(cfg.tel_time)} > :lo AND {_q(cfg.tel_time)} <= :hi")
    with engine.connect() as c:
        tel = pd.read_sql(text(tel_sql), c, params={"lo": _param(engine, T - cfg.window_s), "hi": _param(engine, T)})
    if tel.empty:
        return [], []
    tr_ids = sorted(int(x) for x in tel[cfg.tel_tr].unique())

    scols = [cfg.sch_stop, cfg.sch_tr, cfg.sch_time]
    scols += [cfg.sch_geom] if cfg.sch_geom else [cfg.sch_lon, cfg.sch_lat]
    if cfg.sch_manual:
        scols.append(cfg.sch_manual)
    sch_sql = (f"SELECT {', '.join(map(_q, scols))} FROM {_q(cfg.sch_table)} "
               f"WHERE {_q(cfg.sch_time)} >= :lo AND {_q(cfg.sch_time)} <= :hi")
    with engine.connect() as c:
        sch = pd.read_sql(text(sch_sql), c, params={"lo": _param(engine, T - PLAN_BACK_S), "hi": _param(engine, T + PLAN_FWD_S)})
        curdev = {}
        if cfg.curdev_table:
            cd = pd.read_sql(text(f"SELECT {_q(cfg.curdev_tr)}, {_q(cfg.curdev_value)} FROM {_q(cfg.curdev_table)}"), c)
            curdev = {int(a): float(b) for a, b in zip(cd[cfg.curdev_tr], cd[cfg.curdev_value]) if pd.notna(b)}

    if cfg.sch_geom:
        xy = sch[cfg.sch_geom].astype(str).str.extract(r"POINT\s*\(([-\d.]+)\s+([-\d.]+)\)").astype(float)
        sch["_lon"], sch["_lat"] = xy[0], xy[1]
    else:
        sch["_lon"], sch["_lat"] = sch[cfg.sch_lon].astype(float), sch[cfg.sch_lat].astype(float)
    sch["_t"] = _unix(sch[cfg.sch_time])
    sch["_manual"] = (sch[cfg.sch_manual].astype(str).str.lower().isin(["true", "1", "t"]).astype(float)
                      if cfg.sch_manual else 0.0)

    t_ev = _unix(tel[cfg.tel_time])
    t_rx = _unix(tel[cfg.tel_receive]) if cfg.tel_receive else t_ev
    tel["_t"] = np.fmax(t_ev, np.nan_to_num(t_rx, nan=-np.inf))  # точка известна, когда и произошла, и получена
    tel["_valid"] = tel[cfg.tel_valid].astype(str).str.lower().isin(["true", "1", "t"])

    states, skipped = [], []
    plans = {int(k): g for k, g in sch.groupby(cfg.sch_tr)}
    for tr_id, g in tel.groupby(cfg.tel_tr):
        tr_id = int(tr_id)
        if tr_id not in plans:
            skipped.append({"tr_id": tr_id, "reason": "нет планового расписания"})
            continue
        p = plans[tr_id]
        plan = VehiclePlan.from_arrays(p[cfg.sch_stop].to_numpy(), p["_t"].to_numpy(), p["_lon"].to_numpy(),
                                       p["_lat"].to_numpy(), np.asarray(p["_manual"], dtype=float))
        if select_target(plan, T) is None:
            skipped.append({"tr_id": tr_id, "reason": "нет плановой остановки в окне T+10…15 мин"})
            continue
        track = VehicleTrack.from_arrays(g["_t"].to_numpy(), g[cfg.tel_lon].to_numpy(dtype=float),
                                         g[cfg.tel_lat].to_numpy(dtype=float),
                                         g[cfg.tel_speed].to_numpy(dtype=float), g["_valid"].to_numpy())
        states.append({"tr_id": tr_id, "plan": plan, "track": track, "T": T, "cur_dev_s": curdev.get(tr_id)})
    return states, skipped


def predictions_table(cfg: DBConfig):
    """Описание таблицы прогнозов (создаётся автоматически, если её нет)."""
    from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, Integer, MetaData, String, Table

    md = MetaData()
    return Table(
        cfg.pred_table, md,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("tr_id", BigInteger, index=True, nullable=False),
        Column("t_forecast", DateTime, index=True, nullable=False, comment="момент потока T, UTC"),
        Column("predicted_at", DateTime, nullable=False, comment="когда сделан прогноз, UTC"),
        Column("target_stop_id", BigInteger, nullable=False),
        Column("target_time_plan", DateTime, nullable=False),
        Column("predicted_delay_s", Float, nullable=False),
        Column("predicted_arrival", DateTime, nullable=False),
        Column("interval_lo_s", Float),
        Column("interval_hi_s", Float),
        Column("model_used", String(16), nullable=False),
        Column("fallback_reason", String(500)),
        Column("degraded", Boolean, nullable=False),
    )


def write_predictions(engine, cfg: DBConfig, preds) -> int:
    """Дописать прогнозы в таблицу ``predictions`` (история сохраняется)."""
    tbl = predictions_table(cfg)
    tbl.metadata.create_all(engine, checkfirst=True)
    rows = [{
        "tr_id": p.tr_id, "t_forecast": _ts(p.T), "predicted_at": _ts(p.predicted_at),
        "target_stop_id": p.target_stop_id, "target_time_plan": _ts(p.target_time_plan),
        "predicted_delay_s": p.predicted_delay_s, "predicted_arrival": _ts(p.predicted_arrival),
        "interval_lo_s": p.interval_s[0], "interval_hi_s": p.interval_s[1], "model_used": p.model_used,
        "fallback_reason": (p.fallback_reason or None) and p.fallback_reason[:500], "degraded": p.degraded,
    } for p in preds]
    if rows:
        with engine.begin() as c:
            c.execute(tbl.insert(), rows)
    return len(rows)


def run_cycle(predictor, engine, cfg: DBConfig, T: Optional[float] = None, write: bool = True) -> dict:
    """Один цикл: прочитать базу → прогноз по всем ТС → записать в ``predictions``.

    ``predictor`` — :class:`ml.predictor.DelayPredictor` или ``None`` (тогда бейзлайн).
    """
    from ml.predictor import baseline_prediction

    t0 = time.perf_counter()
    if T is None:
        T = current_T(engine, cfg)
    if T is None:
        return {"T": None, "predictions": [], "skipped": [], "written": 0, "timing_ms": {}}
    states, skipped = load_states(engine, cfg, T)
    t_read = time.perf_counter()
    if predictor is not None:
        try:
            preds = predictor.predict_states(states) if states else []
        except Exception as e:  # сбой модели — бейзлайн, цикл не падает
            preds = [baseline_prediction(s, None, "hint" if s.get("cur_dev_s") is not None else "stream",
                                         f"сбой предиктора: {type(e).__name__}: {e}") for s in states]
    else:
        preds = [baseline_prediction(s, None, "hint" if s.get("cur_dev_s") is not None else "stream",
                                     "модель не загружена") for s in states]
    t_pred = time.perf_counter()
    written = write_predictions(engine, cfg, preds) if write else 0
    t_end = time.perf_counter()
    return {"T": T, "predictions": preds, "skipped": skipped, "written": written,
            "timing_ms": {"read": round((t_read - t0) * 1000, 1), "predict": round((t_pred - t_read) * 1000, 1),
                          "write": round((t_end - t_pred) * 1000, 1), "total": round((t_end - t0) * 1000, 1)}}
