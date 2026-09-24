# Delay Prediction Platform

!!!!
Если кто-то ставит новую библиотеку (pip install), он сразу добавляет её в requirements.txt и коммитит это изменение.
Все импортируемые библиотеки должны быть тоже добавлены в requirements.txt. Если нет - добавьте пожалуйста.
Устанавливайте себе общее окружение

Клонируйте себе репозиторий:

git clone https://github.com/valentinesvev/delay_prediction_platform.git
cd delay_prediction_platform

Загрузите обновления, зайдите в свою ветеку: 

git switch main
git pull

git switch feature/ *название вашей ветки*
git merge main
(Командой git branch проверьте где вы)

Установите окружение:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Для работы в Jupyter запустите это:

pip install ipykernel 
python -m ipykernel install --user --name delay-platform --display-name "Python (delay-platform)" 
Jupyter lab
(В окне Jupyter выберите окружение)

При повторном запуске:

cd delay_prediction_platform

git switch main
git pull

git switch feature/ml
git merge main

source .venv/bin/activate
jupyter lab




## Структура

- `backend/` — приложение FastAPI, endpoints и схемы HTTP-запросов.
- `data/` — заготовки получения, валидации, хранения и удаления данных.
- `ml/` — feature pipeline и модель прогнозирования
- `tests/` — автоматические проверки
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
