# ML-модуль

Рабочий цикл: `ml.worker` читает из общей базы `telemetry` и `schedule_plan`, через `ml.db.run_cycle` строит признаки и вызывает `ml.predictor.DelayPredictor`, затем записывает историю прогнозов в таблицу `predictions`. Один цикл охватывает несколько транспортных средств. Модель загружается при запуске worker; при её недоступности используется baseline.

Замеры `read`, `predict`, `write`, `total` (мс) входят в журнал worker; `DelayPredictor.last_timing` содержит замеры построения признаков и инференса. Расчётные признаки в рабочем цикле не сохраняются. Отдельные файлы `artifacts/features/*.csv` относятся к обучению и создаются `ml.dataset`.

Для ручной проверки модели есть отдельное приложение `ml.service` с HTTP-методами `POST /predict`, `POST /predict/batch` и `POST /predict/from-db`. Последний метод по умолчанию добавляет прогнозы в БД. Формат и метрики обучения описаны в [README_ML.md](../README_ML.md).
