"""Построение признаков для прогноза задержки ТС на горизонте 10–15 минут.

Один и тот же код используется:

* офлайн — при сборке обучающей выборки из CSV (``ml.dataset``);
* онлайн — в ML-сервисе, когда Backend присылает окно телеметрии и план ТС.

Правило честности: функция :func:`compute_features` получает только телеметрию
с ``event_time <= T`` (и ``receive_time <= T``) и **плановое** расписание.
Фактические времена прибытия из ``schedule.csv`` не используются нигде.

Ключевые идеи признаков:

1. **Выравнивание трека по графику** (schedule alignment). Плановое положение ТС
   в момент времени ``t`` получаем линейной интерполяцией координат остановок
   по плановым временам. Ищем сдвиг ``δ``, при котором фактический трек за
   последние N минут лучше всего совпадает с плановым положением в ``t − δ``.
   ``δ`` — оценка текущего отклонения от графика «прямо сейчас», свежее, чем
   ``cur_dev_s`` (который относится к последней пройденной остановке).
2. **Детекция проходов остановок** по телеметрии (точка минимального расстояния
   до остановки) → ряд отклонений на последних остановках и его тренд.
3. **Динамика движения**: скорость, доля стоянок, текущая стоянка (простой),
   разрывы связи.
4. **План впереди**: сколько остановок и какой запас времени до целевой
   остановки, есть ли отстой/разворот (большой разрыв в плане), плановая
   скорость на участке против фактической.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

EARTH_M_PER_DEG = 111_320.0
COS_LAT = float(np.cos(np.radians(55.75)))  # Москва: локальная равнопромежуточная проекция

HORIZON_MIN_S = 10 * 60
HORIZON_MAX_S = 15 * 60

# Сетка сдвигов для выравнивания трека по графику (сек): от −10 до +15 минут.
DELTA_GRID = np.arange(-600, 901, 10, dtype=float)
MAX_SPEED_KMH = 120.0
LAYOVER_GAP_S = 240.0     # разрыв в плане ≥ 4 мин считаем отстоем/разворотом
TYPICAL_SPEED_MS = 5.0    # ~18 км/ч — типичная скорость движения между остановками
DWELL_S = 20.0            # типичная стоянка на остановке

FEATURE_NAMES: list[str] = [
    # подсказка из условия
    "cur_dev_s", "cur_dev_missing", "last_plan_stop_age_s",
    # выравнивание по графику
    "align_dev_3m", "align_dev_10m", "align_dev_20m",
    "align_cost_3m", "align_cost_10m", "align_sharp_10m",
    "align_trend", "align_minus_cur",
    # детекция проходов остановок
    "det_n", "det_last_dev", "det_mean3_dev", "det_slope", "det_last_age_s",
    "det_proj_dev", "det_minus_cur",
    # движение
    "spd_mean_5m", "spd_mean_10m", "spd_moving_10m", "stop_share_10m",
    "cur_dwell_s", "last_speed", "gap_last_fix_s", "n_fix_10m", "valid_share_10m",
    "dist_moved_5m",
    # план впереди
    "lead_s", "n_stops_to_target", "target_gap_prev_s", "max_gap_ahead_s",
    "plan_dist_to_target_m", "plan_speed_ahead_kmh", "req_time_at_cur_speed_s",
    "slack_s", "proj_dev_speed", "target_manual", "plan_dist_from_now_m",
    # структура рейса: отстой/разворот поглощает задержку
    "layover_ahead", "layover_ahead_s", "excess_plan_ahead_s", "buffer_ahead_s",
    "trip_elapsed_s", "trip_stops_done", "trip_remaining_s", "target_after_layover",
    "cur_dev_minus_buffer",
    # время
    "hour",
]


@dataclass
class VehiclePlan:
    """Плановое расписание одного ТС (отсортировано по времени).

    Attributes:
        stop_ids: ``tt_action_item_id`` остановок.
        t: плановое время прибытия, unix-секунды.
        x, y: координаты остановки в метрах (локальная проекция).
        manual: флаг ``manual_fill``.
    """

    stop_ids: np.ndarray
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    manual: np.ndarray

    @classmethod
    def from_arrays(cls, stop_ids, t, lon, lat, manual=None) -> "VehiclePlan":
        """Собрать план из массивов; координаты в градусах, время в unix-сек."""
        order = np.argsort(np.asarray(t, dtype=float), kind="stable")
        lon = np.asarray(lon, dtype=float)[order]
        lat = np.asarray(lat, dtype=float)[order]
        man = np.zeros(len(order)) if manual is None else np.asarray(manual, dtype=float)[order]
        return cls(
            stop_ids=np.asarray(stop_ids)[order],
            t=np.asarray(t, dtype=float)[order],
            x=lon * COS_LAT * EARTH_M_PER_DEG,
            y=lat * EARTH_M_PER_DEG,
            manual=man,
        )

    def position_at(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Плановое положение ТС в заданные моменты (линейная интерполяция)."""
        return np.interp(times, self.t, self.x), np.interp(times, self.t, self.y)


@dataclass
class VehicleTrack:
    """Телеметрия одного ТС (только валидные точки, отсортированы по времени)."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    speed: np.ndarray
    n_invalid_t: np.ndarray  # времена невалидных пакетов (для доли валидности)

    @classmethod
    def from_arrays(cls, t, lon, lat, speed, valid) -> "VehicleTrack":
        """Очистка: невалидные координаты, выбросы скорости, координаты вне Москвы."""
        t = np.asarray(t, dtype=float)
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        speed = np.asarray(speed, dtype=float)
        valid = np.asarray(valid).astype(bool)
        ok = (
            valid & np.isfinite(lon) & np.isfinite(lat)
            & (lat > 55.0) & (lat < 56.5) & (lon > 36.5) & (lon < 38.5)
        )
        order = np.argsort(t[ok], kind="stable")
        sp = speed[ok][order]
        sp = np.where((sp >= 0) & (sp <= MAX_SPEED_KMH), sp, np.nan)
        return cls(
            t=t[ok][order],
            x=lon[ok][order] * COS_LAT * EARTH_M_PER_DEG,
            y=lat[ok][order] * EARTH_M_PER_DEG,
            speed=sp,
            n_invalid_t=np.sort(t[~ok]),
        )

    def upto(self, T: float) -> "VehicleTrack":
        """Срез телеметрии, доступной на момент ``T`` (анти-утечка)."""
        k = np.searchsorted(self.t, T, side="right")
        j = np.searchsorted(self.n_invalid_t, T, side="right")
        return VehicleTrack(self.t[:k], self.x[:k], self.y[:k], self.speed[:k], self.n_invalid_t[:j])


def select_target(plan: VehiclePlan, T: float) -> Optional[tuple[int, float]]:
    """Первая остановка с плановым временем в окне ``(T+10 мин, T+15 мин]``.

    Returns:
        ``(индекс в плане, плановое время)`` или ``None``, если в окне нет остановок.
    """
    idx = np.where((plan.t > T + HORIZON_MIN_S) & (plan.t <= T + HORIZON_MAX_S))[0]
    if len(idx) == 0:
        return None
    return int(idx[0]), float(plan.t[idx[0]])


def _align(plan: VehiclePlan, tr: VehicleTrack, T: float, window_s: float, prior: float):
    """Найти сдвиг ``δ`` (сек), при котором трек лучше всего совпадает с планом.

    Returns:
        (delta, cost_m, sharpness_m) или (nan, nan, nan), если точек мало.
    """
    m = tr.t >= T - window_s
    if m.sum() < 3:
        return np.nan, np.nan, np.nan
    tt, xx, yy = tr.t[m], tr.x[m], tr.y[m]
    shifted = tt[None, :] - DELTA_GRID[:, None]  # (G, N)
    px = np.interp(shifted, plan.t, plan.x)
    py = np.interp(shifted, plan.t, plan.y)
    d = np.hypot(px - xx[None, :], py - yy[None, :])
    cost = np.median(d, axis=1)
    # слабая регуляризация к априорной оценке — разрешает неоднозначность на стоянках
    reg = cost + 0.02 * np.abs(DELTA_GRID - prior)
    i = int(np.argmin(reg))
    return float(DELTA_GRID[i]), float(cost[i]), float(np.percentile(cost, 50) - cost[i])


def _detect_passes(plan: VehiclePlan, tr: VehicleTrack, T: float, dev_guess: float):
    """Оценить фактические отклонения на остановках, пройденных до ``T``.

    Для каждой остановки с плановым временем в последние 40 минут ищем точку
    трека, ближайшую к остановке, в окне ``план + dev_guess ± 5 мин``.
    Проход засчитывается, если минимум < 60 м и ТС уже отъехало (минимум не на
    последней точке). Возвращает массивы (плановые времена, отклонения).
    """
    if len(tr.t) < 3:
        return np.array([]), np.array([])
    g = 0.0 if not np.isfinite(dev_guess) else dev_guess
    sel = np.where((plan.t >= T - 2400) & (plan.t <= T))[0]
    tps, devs = [], []
    for i in sel:
        tp = plan.t[i]
        lo, hi = np.searchsorted(tr.t, [tp + g - 300, min(tp + g + 300, T)])
        if hi - lo < 2:
            continue
        d = np.hypot(tr.x[lo:hi] - plan.x[i], tr.y[lo:hi] - plan.y[i])
        k = int(np.argmin(d))
        if d[k] > 60 or (lo + k) >= len(tr.t) - 1:
            continue
        tps.append(tp)
        devs.append(tr.t[lo + k] - tp)
    return np.asarray(tps), np.asarray(devs)


def compute_features(
    plan: VehiclePlan,
    track: VehicleTrack,
    T: float,
    target_idx: Optional[int] = None,
    cur_dev_s: Optional[float] = None,
) -> dict:
    """Рассчитать вектор признаков для прогнозной точки ``(ТС, T)``.

    Args:
        plan: плановое расписание ТС (весь день или окно ±1 ч вокруг T).
        track: телеметрия ТС; внутри берётся срез ``t <= T``.
        T: момент прогноза, unix-секунды.
        target_idx: индекс целевой остановки в ``plan``; если ``None`` —
            выбирается автоматически по правилу горизонта 10–15 минут.
        cur_dev_s: подсказка «задержка на последней пройденной остановке»;
            в онлайне может отсутствовать — тогда берётся оценка по телеметрии.

    Returns:
        Словарь ``{имя признака: значение}`` в порядке :data:`FEATURE_NAMES`
        плюс служебные ключи ``_target_idx`` и ``_target_t``.
    """
    tr = track.upto(T)
    f: dict = {k: np.nan for k in FEATURE_NAMES}

    if target_idx is None:
        sel = select_target(plan, T)
        if sel is None:
            raise ValueError("Нет плановой остановки в окне T+10…15 мин")
        target_idx = sel[0]
    tgt_t = float(plan.t[target_idx])

    # --- подсказка cur_dev и возраст последней плановой остановки
    k_last = int(np.searchsorted(plan.t, T, side="right")) - 1
    f["last_plan_stop_age_s"] = T - plan.t[k_last] if k_last >= 0 else np.nan
    has_cur = cur_dev_s is not None and np.isfinite(cur_dev_s)
    prior = float(cur_dev_s) if has_cur else 0.0

    # --- выравнивание трека по графику
    a3 = _align(plan, tr, T, 180, prior)
    a10 = _align(plan, tr, T, 600, prior)
    a20 = _align(plan, tr, T, 1200, prior)
    f["align_dev_3m"], f["align_cost_3m"], _ = a3
    f["align_dev_10m"], f["align_cost_10m"], f["align_sharp_10m"] = a10
    f["align_dev_20m"] = a20[0]
    f["align_trend"] = a3[0] - a20[0]

    # --- детекция проходов остановок
    guess = a10[0] if np.isfinite(a10[0]) else prior
    tps, devs = _detect_passes(plan, tr, T, guess)
    f["det_n"] = float(len(devs))
    if len(devs):
        f["det_last_dev"] = devs[-1]
        f["det_mean3_dev"] = float(np.mean(devs[-3:]))
        f["det_last_age_s"] = T - (tps[-1] + devs[-1])
        if len(devs) >= 3 and np.ptp(tps) > 0:
            slope = np.polyfit(tps, devs, 1)[0]
            f["det_slope"] = float(np.clip(slope, -1, 1))
            f["det_proj_dev"] = devs[-1] + f["det_slope"] * (tgt_t - tps[-1])

    # если подсказки нет (онлайн) — используем лучшую оценку по телеметрии
    if not has_cur:
        est = f["det_last_dev"] if np.isfinite(f["det_last_dev"]) else a10[0]
        cur = est if np.isfinite(est) else 0.0
        f["cur_dev_missing"] = 1.0
    else:
        cur = float(cur_dev_s)
        f["cur_dev_missing"] = 0.0
    f["cur_dev_s"] = cur
    f["align_minus_cur"] = f["align_dev_10m"] - cur
    f["det_minus_cur"] = f["det_last_dev"] - cur

    # --- динамика движения
    for w, name in [(300, "spd_mean_5m"), (600, "spd_mean_10m")]:
        m = tr.t >= T - w
        if m.any():
            f[name] = float(np.nanmean(tr.speed[m])) if np.isfinite(tr.speed[m]).any() else np.nan
    m10 = tr.t >= T - 600
    f["n_fix_10m"] = float(m10.sum())
    n_inv = float(((tr.n_invalid_t >= T - 600)).sum())
    f["valid_share_10m"] = f["n_fix_10m"] / (f["n_fix_10m"] + n_inv) if (f["n_fix_10m"] + n_inv) > 0 else np.nan
    if m10.any():
        sp = tr.speed[m10]
        mv = sp[np.isfinite(sp) & (sp > 3)]
        f["spd_moving_10m"] = float(mv.mean()) if len(mv) else 0.0
        f["stop_share_10m"] = float(np.mean(sp[np.isfinite(sp)] <= 3)) if np.isfinite(sp).any() else np.nan
    if len(tr.t):
        f["gap_last_fix_s"] = T - tr.t[-1]
        f["last_speed"] = tr.speed[-1]
        # текущая стоянка: сколько секунд ТС не сдвигалось дальше 30 м
        d_back = np.hypot(tr.x - tr.x[-1], tr.y - tr.y[-1])
        far = np.where(d_back > 30)[0]
        f["cur_dwell_s"] = (T - tr.t[far[-1] + 1]) if len(far) and far[-1] + 1 < len(tr.t) else (
            T - tr.t[0] if not len(far) else 0.0)
        m5 = np.where(tr.t >= T - 300)[0]
        if len(m5) >= 2:
            seg = np.hypot(np.diff(tr.x[m5]), np.diff(tr.y[m5]))
            f["dist_moved_5m"] = float(seg.sum())

    # --- план впереди
    f["lead_s"] = tgt_t - T
    f["n_stops_to_target"] = float(target_idx - k_last)
    f["target_gap_prev_s"] = tgt_t - plan.t[target_idx - 1] if target_idx > 0 else np.nan
    lo = max(k_last, 0)
    gaps = np.diff(plan.t[lo:target_idx + 1])
    f["max_gap_ahead_s"] = float(gaps.max()) if len(gaps) else 0.0
    f["target_manual"] = float(plan.manual[target_idx])
    f["hour"] = ((T + 3 * 3600) % 86400) / 3600.0  # МСК

    # «где по плану» ТС сейчас: плановое время T − δ; путь по плану до цели
    dev_now = a10[0] if np.isfinite(a10[0]) else cur
    t_plan_now = T - dev_now
    ts = np.concatenate([[t_plan_now], plan.t[(plan.t > t_plan_now) & (plan.t <= tgt_t)]])
    px, py = plan.position_at(ts)
    dist = float(np.hypot(np.diff(px), np.diff(py)).sum()) if len(ts) > 1 else 0.0
    f["plan_dist_to_target_m"] = dist
    f["slack_s"] = tgt_t - t_plan_now  # сколько времени по плану на оставшийся путь
    f["plan_speed_ahead_kmh"] = dist / f["slack_s"] * 3.6 if f["slack_s"] > 30 else np.nan
    ts2 = np.concatenate([[T], plan.t[(plan.t > T) & (plan.t <= tgt_t)]])
    px2, py2 = plan.position_at(ts2)
    f["plan_dist_from_now_m"] = float(np.hypot(np.diff(px2), np.diff(py2)).sum()) if len(ts2) > 1 else 0.0
    v = f["spd_mean_10m"]
    if np.isfinite(v) and v > 1:
        f["req_time_at_cur_speed_s"] = dist / (v / 3.6)
        f["proj_dev_speed"] = dev_now + (f["req_time_at_cur_speed_s"] - f["slack_s"])

    # --- структура рейса (только по плану)
    lo = max(k_last, 0)
    seg_t = np.diff(plan.t[lo:target_idx + 1])
    seg_d = np.hypot(np.diff(plan.x[lo:target_idx + 1]), np.diff(plan.y[lo:target_idx + 1]))
    lay = seg_t >= LAYOVER_GAP_S
    f["layover_ahead"] = float(lay.any())
    f["layover_ahead_s"] = float(seg_t[lay].sum()) if lay.any() else 0.0
    # запас плана: плановое время минус «физически нужное» время на участке
    need = seg_d / TYPICAL_SPEED_MS + DWELL_S
    f["excess_plan_ahead_s"] = float(np.clip(seg_t - need, 0, None).sum()) if len(seg_t) else 0.0
    f["buffer_ahead_s"] = float((seg_t - need).sum()) if len(seg_t) else 0.0
    f["cur_dev_minus_buffer"] = cur - f["excess_plan_ahead_s"]
    f["target_after_layover"] = float(f["target_gap_prev_s"] >= LAYOVER_GAP_S) if np.isfinite(f["target_gap_prev_s"]) else np.nan
    all_gaps = np.diff(plan.t)
    lay_idx = np.where(all_gaps >= LAYOVER_GAP_S)[0]  # отстой между i и i+1
    starts = lay_idx[lay_idx + 1 <= k_last] + 1 if k_last >= 0 else np.array([], int)
    trip_start = int(starts[-1]) if len(starts) else 0
    f["trip_elapsed_s"] = T - plan.t[trip_start]
    f["trip_stops_done"] = float(k_last - trip_start + 1) if k_last >= 0 else 0.0
    ends = lay_idx[lay_idx >= max(k_last, 0)]
    f["trip_remaining_s"] = (plan.t[ends[0]] - T) if len(ends) else (plan.t[-1] - T)

    f["_target_idx"] = target_idx
    f["_target_t"] = tgt_t
    return f
