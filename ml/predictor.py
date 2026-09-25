"""Предикторы задержки.

* :class:`BaselinePredictor` — исходный бейзлайн (прогноз = текущая задержка),
  оставлен для совместимости с текущим Backend и тестами.
* :class:`DelayPredictor` — боевая модель: регрессия задержки (ансамбль CatBoost +
  LightGBM + XGBoost + нейросеть PyTorch) и интервал прогноза (квантили CatBoost).

Цвет риска и причину задержки ML-модуль не считает — это отдельный модуль,
который берёт прогнозы из таблицы ``predictions``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from ml.feature_engineering import FEATURE_NAMES, VehiclePlan, VehicleTrack, compute_features
from ml.features import DelayFeatures

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "model"

# Нет свежих точек дольше этого — прогноз помечается degraded (по последнему состоянию)
STALE_TELEMETRY_S = 120.0
# Конформная калибровка интервала: сырые квантили CatBoost покрывают ~66% на holdout,
# при растяжении полуширины в 1.5 раза — ~80% (номинал 10–90%).
INTERVAL_SCALE = 1.5

# Переключатель модели (переменная окружения MODEL_TYPE):
#   ensemble — бустинги + нейросеть (по умолчанию, лучшее качество)
#   boosting — среднее CatBoost + LightGBM + XGBoost
#   catboost — только CatBoost
#   torch    — только нейросеть PyTorch
#   baseline — прогноз = текущая задержка (cur_dev_s)
# При любой ошибке выбранной модели ответ считается бейзлайном, сервис не падает.
MODEL_TYPES = ("ensemble", "boosting", "catboost", "torch", "baseline")
DEFAULT_MODEL_TYPE = os.getenv("MODEL_TYPE", "ensemble")

class Predictor(Protocol):
    """Единый интерфейс baseline и будущей ML-модели."""

    def predict(self, features: DelayFeatures) -> float:
        ...


class BaselinePredictor:
    def predict(self, features: DelayFeatures) -> float:
        """Прогноз будущей задержки равен текущей задержке в минутах."""
        return features.current_delay


@dataclass
class Prediction:
    """Результат прогноза для одной точки ``(ТС, T)``."""

    tr_id: int
    T: float
    target_stop_id: int
    target_time_plan: float
    predicted_delay_s: float
    interval_s: tuple[float, float]
    mode: str = "hint"
    degraded: bool = False
    inference_ms: float = 0.0
    model_used: str = "ensemble"
    fallback_reason: Optional[str] = None
    predicted_at: float = field(default_factory=time.time)

    @property
    def predicted_arrival(self) -> float:
        """Прогнозируемое время прибытия на целевую остановку (unix-сек)."""
        return self.target_time_plan + self.predicted_delay_s


class DelayPredictor:
    """Загрузка обученных артефактов и прогноз.

    Два режима:

    * ``hint`` — есть ``cur_dev_s`` (офлайн-датасет / Backend посчитал сам);
    * ``stream`` — подсказки нет, всё оценивается по телеметрии и плану.
    """

    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR, model_type: str = DEFAULT_MODEL_TYPE):
        if model_type not in MODEL_TYPES:
            raise ValueError(f"MODEL_TYPE должен быть одним из {MODEL_TYPES}, получено {model_type!r}")
        self.model_type = model_type
        import catboost as cb
        import lightgbm as lgb
        import xgboost as xgb

        self.model_dir = Path(model_dir)
        self.meta = json.loads((self.model_dir / "meta.json").read_text())
        self.features = self.meta["features"]
        assert self.features == FEATURE_NAMES, "Модель обучена на другом наборе признаков"
        self.reg, self.q = {}, {}
        for mode in ("hint", "stream"):
            self.reg[mode] = []
            for n in self.meta["ensemble"]:
                stem = self.model_dir / f"{mode}_reg_{n}"
                if n == "catboost":
                    m = cb.CatBoostRegressor()
                    m.load_model(str(stem.with_suffix(".cbm")))
                elif n == "lightgbm":
                    m = lgb.Booster(model_file=str(stem.with_suffix(".txt")))
                else:
                    m = xgb.XGBRegressor()
                    m.load_model(str(stem.with_suffix(".json")))
                self.reg[mode].append(m)
            self.q[mode] = []
            for q in (10, 90):
                m = cb.CatBoostRegressor()
                m.load_model(str(self.model_dir / f"{mode}_q{q}.cbm"))
                self.q[mode].append(m)
        self._load_nn()

    def _load_nn(self) -> None:
        """Нейросеть (PyTorch) — необязательный член ансамбля.

        Загружается, если в ``meta.json`` есть секция ``nn`` и установлен torch;
        иначе сервис работает на одних бустингах (graceful degradation).
        """
        self.nn, self.w_nn = {}, {}
        for mode, spec in self.meta.get("nn", {}).items():
            if spec.get("w_nn", 0) <= 0:
                continue
            try:
                import torch  # noqa: F401
                from ml.nn_model import load_net
                self.nn[mode] = [load_net(self.model_dir / f) for f in spec["files"]]
                self.w_nn[mode] = float(spec["w_nn"])
            except Exception as e:  # pragma: no cover - нет torch / файлов
                print(f"[DelayPredictor] NN для режима {mode} не загружена: {e}")

    @property
    def uses_nn(self) -> bool:
        return bool(self.nn)

    # ------------------------------------------------------------ табличный прогноз
    def predict_frame(self, X, mode: str = "hint", seqs: Optional[np.ndarray] = None,
                      model_type: Optional[str] = None) -> dict:
        """Прогноз по готовой матрице признаков (DataFrame в порядке FEATURE_NAMES).

        Args:
            X: табличные признаки.
            mode: ``hint`` | ``stream``.
            seqs: последовательности телеметрии ``(N, 120, 7)`` для нейросети;
                если ``None`` или сеть не загружена — только бустинги.
            model_type: одна из :data:`MODEL_TYPES`; по умолчанию ``self.model_type``.

        Returns:
            словарь массивов ``delay``, ``q10``, ``q90`` (границы интервала), ``resid_gb``, ``resid_nn``.
        """
        mt = model_type or self.model_type
        X = X[self.features]
        base = X["cur_dev_s"].to_numpy(dtype=float)

        def _nn():
            if seqs is None or mode not in self.nn:
                raise RuntimeError("нейросеть недоступна (нет torch/весов или последовательностей)")
            from ml.nn_model import predict_net
            Xn = X.to_numpy(dtype=float)
            return np.mean([predict_net(n, sc, seqs, Xn, base) for n, sc in self.nn[mode]], axis=0)

        resid_gb = resid_nn = None
        if mt == "baseline":
            resid = np.zeros(len(X))
        elif mt == "catboost":
            resid = resid_gb = self.reg[mode][self.meta["ensemble"].index("catboost")].predict(X)
        elif mt == "torch":
            resid = resid_nn = _nn()
        else:
            resid = resid_gb = np.mean([m.predict(X) for m in self.reg[mode]], axis=0)
            if mt == "ensemble" and seqs is not None and mode in self.nn:
                resid_nn = _nn()
                w = self.w_nn[mode]
                resid = (1 - w) * resid_gb + w * resid_nn
        delay = np.clip(base + resid, -600, 900)
        q10 = base + self.q[mode][0].predict(X)
        q90 = base + self.q[mode][1].predict(X)
        lo = delay - np.clip(delay - q10, 0, None) * INTERVAL_SCALE
        hi = delay + np.clip(q90 - delay, 0, None) * INTERVAL_SCALE
        return {"delay": delay, "q10": lo, "q90": hi, "resid_gb": resid_gb, "resid_nn": resid_nn}

    # ------------------------------------------------------------ онлайн-прогноз
    def predict_states(self, states: list[dict]) -> list[Prediction]:
        """Батчевый онлайн-прогноз по «сырым» состояниям ТС.

        Каждый элемент ``states`` — словарь с ключами ``tr_id``, ``plan``
        (:class:`VehiclePlan`), ``track`` (:class:`VehicleTrack`), ``T``,
        опционально ``cur_dev_s`` и ``target_idx``. Признаки считаются по каждому
        ТС, а модели вызываются один раз на весь батч (отдельно для режимов
        ``hint``/``stream``) — это держит latency на парк в десятки миллисекунд.

        Целевая остановка по умолчанию — первая с плановым временем
        в ``(T+10, T+15]`` мин: горизонт соблюдается по построению.
        """
        import pandas as pd

        t0 = time.perf_counter()
        feats, modes, feat_err = [], [], {}
        for i, st in enumerate(states):
            cur = st.get("cur_dev_s")
            try:
                feats.append(compute_features(st["plan"], st["track"], float(st["T"]), st.get("target_idx"), cur))
            except ValueError:
                raise  # нет цели в окне 10–15 мин — это не сбой, сервис вернёт skipped
            except Exception as e:  # сломались признаки одного ТС — бейзлайн только для него
                feats.append(None)
                feat_err[i] = f"features: {type(e).__name__}: {e}"
            modes.append("hint" if cur is not None else "stream")
        t_feat = time.perf_counter()
        out: list[Optional[Prediction]] = [None] * len(states)
        for mode in ("hint", "stream"):
            idx = [i for i, m in enumerate(modes) if m == mode and feats[i] is not None]
            if not idx:
                continue
            try:
                X = pd.DataFrame([{k: feats[i][k] for k in self.features} for i in idx])
                seqs = None
                if self.model_type in ("ensemble", "torch") and mode in self.nn:
                    from ml.sequence import build_sequence
                    seqs = np.stack([build_sequence(states[i]["plan"], states[i]["track"], float(states[i]["T"]),
                                                    feats[i]["cur_dev_s"]) for i in idx])
                r = self.predict_frame(X, mode, seqs)
                used = self.model_type
                if used == "ensemble" and r["resid_nn"] is None:
                    used = "boosting"
            except Exception as e:  # любая ошибка модели → бейзлайн, сервис не падает
                for i in idx:
                    feat_err[i] = f"model {self.model_type}: {type(e).__name__}: {e}"
                continue
            for j, i in enumerate(idx):
                f, st = feats[i], states[i]
                delay = float(r["delay"][j])
                gap = f["gap_last_fix_s"]
                out[i] = Prediction(
                    tr_id=int(st["tr_id"]), T=float(st["T"]),
                    target_stop_id=int(st["plan"].stop_ids[f["_target_idx"]]),
                    target_time_plan=float(f["_target_t"]),
                    predicted_delay_s=round(delay, 1),
                    interval_s=(round(float(r["q10"][j]), 1), round(float(r["q90"][j]), 1)),
                    mode=mode, degraded=bool(not np.isfinite(gap) or gap > STALE_TELEMETRY_S),
                    model_used=used,
                )
        for i, reason in feat_err.items():
            out[i] = baseline_prediction(states[i], feats[i], modes[i], reason)
        total_ms = (time.perf_counter() - t0) * 1000
        per = total_ms / max(len(states), 1)
        for p_ in out:
            p_.inference_ms = round(per, 2)
        self.last_timing = {"features_ms": round((t_feat - t0) * 1000, 2), "total_ms": round(total_ms, 2)}
        return out  # type: ignore[return-value]

    def predict_state(self, tr_id: int, plan: VehiclePlan, track: VehicleTrack, T: float,
                      cur_dev_s: Optional[float] = None, target_idx: Optional[int] = None) -> Prediction:
        """Прогноз для одного ТС (обёртка над :meth:`predict_states`)."""
        return self.predict_states([{"tr_id": tr_id, "plan": plan, "track": track, "T": T,
                                     "cur_dev_s": cur_dev_s, "target_idx": target_idx}])[0]


def baseline_prediction(st: dict, f: Optional[dict], mode: str, reason: str) -> Prediction:
    """Бейзлайн «прогноз = текущая задержка» — ответ, когда модель или признаки сломались.

    Текущая задержка: ``cur_dev_s`` от Backend, иначе оценка по телеметрии
    (если признаки успели посчитаться), иначе 0.
    """
    from ml.feature_engineering import select_target

    plan, T = st["plan"], float(st["T"])
    cur = st.get("cur_dev_s")
    if cur is None and f is not None and np.isfinite(f.get("cur_dev_s", np.nan)):
        cur = f["cur_dev_s"]
    cur = float(cur) if cur is not None else 0.0
    if st.get("target_idx") is not None:
        tidx = int(st["target_idx"])
    else:
        sel = select_target(plan, T)
        tidx = sel[0] if sel else int(np.argmin(np.abs(plan.t - (T + 720))))
    return Prediction(
        tr_id=int(st["tr_id"]), T=T, target_stop_id=int(plan.stop_ids[tidx]), target_time_plan=float(plan.t[tidx]),
        predicted_delay_s=round(cur, 1), interval_s=(round(cur - 120, 1), round(cur + 120, 1)),
        mode=mode, degraded=True, model_used="baseline", fallback_reason=reason,
    )
