"""Обзор ТС по телеметрии; ML-прогнозы только читаются из базы."""
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.engine import make_url

from backend.predictions import latest_for_all
from ml.db import DBConfig, _param, _q, _unix, current_T, make_engine
from ml.feature_engineering import VehiclePlan, VehicleTrack, _align


def fleet_snapshot():
    cfg = DBConfig()
    historical = cfg.time_mode == 'stream'
    result = dict(mode='historical' if historical else 'live', demo_plan=os.getenv('DEMO_PLAN') == '1',
                  reference_time=None, vehicles=[], active_window_s=120)
    if not cfg.url:
        return result
    url = make_url(cfg.url)
    if url.get_backend_name() == 'sqlite' and url.database != ':memory:' and not Path(url.database).exists():
        return result
    engine = make_engine(cfg)
    try:
        with engine.connect() as conn:
            schema = inspect(conn)
            if not schema.has_table(cfg.tel_table):
                return result
            T = current_T(engine, cfg) if historical else time.time()
            if T is None:
                return result
            result['reference_time'] = pd.Timestamp(T, unit='s', tz='UTC').isoformat()
            tel = pd.read_sql(text(f'SELECT * FROM {_q(cfg.tel_table)} WHERE {_q(cfg.tel_time)} >= :lo '
                                   f'AND {_q(cfg.tel_time)} <= :hi'), conn,
                              params={'lo': _param(engine, T - 600), 'hi': _param(engine, T)})
            if tel.empty:
                return result
            tel['_t'] = _unix(tel[cfg.tel_time])
            tel['_received'] = _unix(tel[cfg.tel_receive]) if cfg.tel_receive else tel['_t']
            tel = tel[(tel['_received'] <= T) & (tel['_t'] <= T)]
            sch = pd.DataFrame()
            if schema.has_table(cfg.sch_table):
                sch = pd.read_sql(text(f'SELECT * FROM {_q(cfg.sch_table)} WHERE {_q(cfg.sch_time)} >= :lo '
                                       f'AND {_q(cfg.sch_time)} <= :hi'), conn,
                                  params={'lo': _param(engine, T - 21600), 'hi': _param(engine, T + 7200)})
        plans = {}
        if not sch.empty:
            sch['_t'] = _unix(sch[cfg.sch_time])
            if cfg.sch_geom:
                xy = sch[cfg.sch_geom].str.extract(r'POINT\s*\(([-\d.]+)\s+([-\d.]+)\)').astype(float)
                sch['_lon'], sch['_lat'] = xy[0], xy[1]
            else:
                sch['_lon'], sch['_lat'] = sch[cfg.sch_lon], sch[cfg.sch_lat]
            plans = {int(k): g.sort_values('_t') for k, g in sch.groupby(cfg.sch_tr)}
        from backend.endpoints import _present
        predictions = {int(row['tr_id']): _present(row) for row in latest_for_all()}
        for tr_id, group in tel.groupby(cfg.tel_tr):
            group = group.sort_values('_t')
            last = group.iloc[-1]
            if T - float(last['_t']) > result['active_window_s']:
                continue
            row = dict(tr_id=int(tr_id), unit_id=int(last['unit_id']) if 'unit_id' in last and pd.notna(last['unit_id']) else None,
                       event_time=pd.Timestamp(last['_t'], unit='s', tz='UTC').isoformat(),
                       speed_kmh=float(last[cfg.tel_speed]), current_delay_s=None,
                       delay_source=None, current_stop=None, next_stop=None, status='no_plan',
                       prediction=predictions.get(int(tr_id)))
            plan_rows = plans.get(int(tr_id))
            if plan_rows is not None and len(plan_rows):
                plan = VehiclePlan.from_arrays(plan_rows[cfg.sch_stop], plan_rows['_t'], plan_rows['_lon'], plan_rows['_lat'])
                valid = group[cfg.tel_valid].astype(str).str.lower().isin(['1', 'true', 't'])
                track = VehicleTrack.from_arrays(group['_t'], group[cfg.tel_lon], group[cfg.tel_lat], group[cfg.tel_speed], valid).upto(T)
                row['status'] = 'on_route' if plan.t[0] <= T <= plan.t[-1] else 'outside_schedule'
                delta, cost, _ = _align(plan, track, T, 600, 0)
                if math.isfinite(delta) and cost <= 500 and row['status'] == 'on_route':
                    row['current_delay_s'] = round(delta, 1)
                    row['delay_source'] = 'gps_plan_estimate'
                elif math.isfinite(cost) and cost > 500:
                    row['status'] = 'off_route'
                def stop_at(index):
                    stop = plan_rows.iloc[index]
                    name = stop.get('building_address')
                    return dict(id=int(stop[cfg.sch_stop]), name=str(name) if pd.notna(name) else None,
                                planned_time=pd.Timestamp(stop['_t'], unit='s', tz='UTC').isoformat())
                if len(track.t):
                    distances = np.hypot(plan.x - track.x[-1], plan.y - track.y[-1])
                    near = int(np.argmin(distances))
                    if distances[near] <= 60:
                        row['current_stop'] = stop_at(near)
                next_index = int(np.searchsorted(plan.t, T - (row['current_delay_s'] or 0), side='right'))
                if next_index < len(plan.t):
                    row['next_stop'] = stop_at(next_index)
                if row['prediction']:
                    target = plan_rows[plan_rows[cfg.sch_stop] == row['prediction']['target_stop_id']]
                    name = target.iloc[0].get('building_address') if len(target) else None
                    row['prediction']['target_stop_name'] = str(name) if pd.notna(name) else None
            result['vehicles'].append(row)
        return result
    finally:
        engine.dispose()
