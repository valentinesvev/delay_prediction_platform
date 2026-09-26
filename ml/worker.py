"""Фоновый воркер ML: раз в N секунд читает базу, прогнозирует, пишет ``predictions``.

Запуск::

    DATABASE_URL=postgresql+psycopg2://user:pass@db:5432/transport python -m ml.worker
    python -m ml.worker --once            # один цикл (для проверки)
    python -m ml.worker --interval 30     # период, сек (по умолчанию ML_INTERVAL_S или 30)

Надёжность: ошибки базы/модели логируются, воркер не падает и повторяет
попытку с растущей паузой (до 60 с); после восстановления базы продолжает
работу сам. Модель выбирается переменной ``MODEL_TYPE``; если модель не
загрузилась — пишет бейзлайн («прогноз = текущая задержка») с причиной
в ``fallback_reason``.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time
from pathlib import Path

from ml.db import DBConfig, current_T, make_engine, run_cycle

log = logging.getLogger("ml.worker")
_stop = False
_wake = threading.Event()


def _handle_stop(*_):
    global _stop
    _stop = True
    _wake.set()


def load_predictor():
    """Модель или ``None`` (тогда воркер работает бейзлайном)."""
    from ml.predictor import DEFAULT_MODEL_DIR, DEFAULT_MODEL_TYPE, DelayPredictor

    model_type = os.getenv("MODEL_TYPE", DEFAULT_MODEL_TYPE)
    if model_type == "baseline":
        log.info("MODEL_TYPE=baseline: использую бейзлайн без загрузки артефактов")
        return None
    try:
        p = DelayPredictor(Path(os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR)), model_type)
        log.info("модель загружена: MODEL_TYPE=%s, нейросеть=%s", p.model_type, p.uses_nn)
        return p
    except Exception as e:
        log.error("модель не загрузилась, работаю бейзлайном: %s", e)
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="выполнить один цикл и выйти")
    ap.add_argument("--interval", type=float, default=float(os.getenv("ML_INTERVAL_S", "30")))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    cfg = DBConfig()
    predictor = load_predictor()
    engine = None
    backoff = 1.0
    last_T = None
    while not _stop:
        t0 = time.time()
        try:
            engine = engine or make_engine(cfg)
            T = current_T(engine, cfg)
            if T is None:
                log.info("в базе нет телеметрии — жду")
            elif T == last_T and cfg.time_mode == "stream":
                log.info("новых данных нет (T не изменился) — пропускаю цикл")
            else:
                r = run_cycle(predictor, engine, cfg, T)
                n_base = sum(p.model_used == "baseline" for p in r["predictions"])
                log.info("T=%s: прогнозов %d (бейзлайн %d), пропущено ТС %d, время %s",
                         time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(r["T"])), r["written"], n_base,
                         len(r["skipped"]), r["timing_ms"])
                last_T = r["T"]
            backoff = 1.0
        except Exception as e:  # база недоступна и т.п. — не падаем
            log.error("цикл не выполнен: %s: %s — повтор через %.0f с", type(e).__name__, e, backoff)
            if args.once:
                raise
            _wake.wait(backoff)
            backoff = min(backoff * 2, 60.0)
            continue
        if args.once:
            break
        _wake.wait(max(0.0, args.interval - (time.time() - t0)))


if __name__ == "__main__":
    main()
