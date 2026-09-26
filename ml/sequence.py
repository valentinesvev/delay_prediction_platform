"""Последовательность телеметрии для нейросети (PyTorch).

Окно — последние 30 минут до момента ``T`` на равномерной сетке 15 с
(120 шагов). На каждом шаге берётся последняя валидная GPS-точка из интервала
``(g − 15 с, g]``; позиция между точками протягивается вперёд (ffill).

Каналы шага:

0. ``has_fix``    — была ли свежая точка на шаге (разрывы связи видны сети);
1. ``speed``      — скорость, км/ч / 40;
2. ``step_m``     — перемещение за шаг, м / 150;
3. ``plan_gap``   — расстояние до планового положения в момент ``g − dev_ref``, м / 500
   (dev_ref — текущая оценка отклонения); показывает, как ТС «уходит» от нитки графика;
4. ``stop_dist``  — расстояние до ближайшей плановой остановки (±30 мин), м / 300:
   различает стоянку на остановке и стоянку в заторе;
5. ``has_plan``   — есть ли у ТС план (для предобучения на ТС без расписания);
6. ``pos``        — положение шага в окне, 0…1.

Только данные с ``t ≤ T`` — анти-утечка та же, что у табличных признаков.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ml.feature_engineering import VehiclePlan, VehicleTrack

SEQ_STEP_S = 15.0
SEQ_LEN = 120  # 30 минут
N_CHANNELS = 7


def build_sequence(plan: Optional[VehiclePlan], track: VehicleTrack, T: float, dev_ref: float) -> np.ndarray:
    """Собрать матрицу ``(SEQ_LEN, N_CHANNELS)`` float32 для момента ``T``."""
    g = T - SEQ_STEP_S * np.arange(SEQ_LEN - 1, -1, -1)
    out = np.zeros((SEQ_LEN, N_CHANNELS), dtype=np.float32)
    out[:, 6] = np.linspace(0, 1, SEQ_LEN)
    k_end = np.searchsorted(track.t, T, side="right")
    t, x, y, sp = track.t[:k_end], track.x[:k_end], track.y[:k_end], track.speed[:k_end]
    if len(t) == 0:
        return out
    # индекс последней точки ≤ g (для ffill) и была ли точка на шаге
    idx = np.searchsorted(t, g, side="right") - 1
    known = idx >= 0
    idc = np.clip(idx, 0, None)
    fresh = known & (t[idc] > g - SEQ_STEP_S)
    px, py = np.where(known, x[idc], np.nan), np.where(known, y[idc], np.nan)
    out[:, 0] = fresh
    s = np.where(fresh, sp[idc], 0.0)
    out[:, 1] = np.nan_to_num(s, nan=0.0) / 40.0
    step = np.hypot(np.diff(px, prepend=np.nan), np.diff(py, prepend=np.nan))
    out[:, 2] = np.clip(np.nan_to_num(step, nan=0.0) / 150.0, 0, 5)
    if plan is not None and len(plan.t) >= 2:
        out[:, 5] = 1.0
        ref = 0.0 if dev_ref is None or not np.isfinite(dev_ref) else float(dev_ref)
        qx, qy = plan.position_at(g - ref)
        d = np.hypot(px - qx, py - qy)
        out[:, 3] = np.clip(np.nan_to_num(d / 500.0, nan=0.0), 0, 5)
        near = (plan.t >= T - 3600) & (plan.t <= T + 1800)
        if near.any():
            dd = np.hypot(px[:, None] - plan.x[near][None, :], py[:, None] - plan.y[near][None, :]).min(axis=1)
            out[:, 4] = np.clip(np.nan_to_num(dd / 300.0, nan=0.0), 0, 3)
    out[~known, 1:5] = 0.0
    return out


def pretext_targets(plan: Optional[VehiclePlan], track: VehicleTrack, T: float, dev_ref: float) -> np.ndarray:
    """Цели самообучения по **будущей** телеметрии (только для предобучения!).

    Возвращает 5 чисел (NaN — цель недоступна, в лоссе маскируется):
    средняя скорость за 5 и 10 мин вперёд (/40), путь за 10 мин (/3000 м),
    доля стоянок за 10 мин, отставание от плановой позиции через 10 мин (/500 м).
    """
    res = np.full(5, np.nan, dtype=np.float32)
    lo, hi5, hi10 = np.searchsorted(track.t, [T, T + 300, T + 600], side="right")
    if hi10 - lo < 10:
        return res
    sp5, sp10 = track.speed[lo:hi5], track.speed[lo:hi10]
    if np.isfinite(sp5).any():
        res[0] = np.nanmean(sp5) / 40.0
    if np.isfinite(sp10).any():
        res[1] = np.nanmean(sp10) / 40.0
        res[3] = np.mean(sp10[np.isfinite(sp10)] <= 3)
    xs, ys = track.x[lo:hi10], track.y[lo:hi10]
    res[2] = np.hypot(np.diff(xs), np.diff(ys)).sum() / 3000.0
    if plan is not None and len(plan.t) >= 2 and np.isfinite(dev_ref):
        qx, qy = plan.position_at(np.array([track.t[hi10 - 1] - dev_ref]))
        res[4] = min(np.hypot(xs[-1] - qx[0], ys[-1] - qy[0]) / 500.0, 5.0)
    return res
