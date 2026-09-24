# Delay Prediction Platform

Минимальный Python-проект для ML-хакатона. Baseline прогнозирует будущую
задержку как текущую задержку. Все задержки измеряются в минутах.

## Структура

- `backend/` — приложение FastAPI, endpoints и схемы HTTP-запросов.
- `data/` — заготовки получения, валидации, хранения и удаления данных.
- `ml/` — feature pipeline, интерфейс `Predictor` и `BaselinePredictor`.
- `tests/` — автоматические проверки baseline и API.
- `frontend/` — место для будущего интерфейса диспетчера.
- `.github/workflows/` — место для будущих workflows автоматических тестов.

## Запуск

Используйте Python 3.9 или новее. Команды выполняются из корня проекта:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
```

Документация API: http://127.0.0.1:8000/docs.
Проверка доступности: `GET /health`.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"current_delay": 12.5}'
```

Ответ: `{"predicted_delay": 12.5}`.

## Baseline

```python
from ml.features import build_features
from ml.predictor import BaselinePredictor, Predictor

predictor: Predictor = BaselinePredictor()
prediction = predictor.predict(build_features(current_delay=12.5))
assert prediction == 12.5
```

Будущая модель должна реализовать тот же метод
`predict(features: DelayFeatures) -> float`. Обучение, доступ к источникам,
хранилище и полноценный frontend пока не реализованы.

## Тесты

```bash
python -m pytest -q
```

## Конфигурация и локальные файлы

Сейчас настройки окружения не требуются. `.env.example` содержит только
безопасный шаблон; загрузка `.env` пока не реализована. Не добавляйте секреты
в исходники. Реальные `.env` и `.env.*` исключены из git, кроме `.env.example`.
Локальные наборы данных размещайте в `data/raw/` и `data/processed/`,
артефакты моделей — в `artifacts/`: эти каталоги также исключены из git.
