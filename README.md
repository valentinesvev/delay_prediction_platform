# Delay Prediction Platform

## Работа с репозиторием и окружением

> **Важно:** если вы устанавливаете новую внешнюю библиотеку через `pip install`, добавьте её в `requirements.txt` и закоммитьте это изменение вместе с кодом. Стандартные библиотеки Python (`os`, `json`, `sys` и т. п.) в `requirements.txt` добавлять не нужно.

### Первый запуск

#### 1. Клонируйте репозиторий

```bash
git clone https://github.com/valentinesvev/delay_prediction_platform.git
cd delay_prediction_platform
```

#### 2. Перейдите в свою ветку

Если ветка уже создана для вас на GitHub:

```bash
git fetch
git switch --track origin/feature/НАЗВАНИЕ_ВЕТКИ
```

Проверьте, в какой ветке вы находитесь:

```bash
git branch
```

Звёздочка `*` должна стоять напротив вашей ветки.

#### 3. Создайте и активируйте виртуальное окружение

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Работа в Jupyter

Если вы используете Jupyter, один раз установите и зарегистрируйте kernel проекта:

```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name delay-platform --display-name "Python (delay-platform)"
jupyter lab
```

В Jupyter выберите kernel **Python (delay-platform)**.

### Последующие запуски

Перед началом работы обновите `main`, затем подтяните его изменения в свою ветку:

```bash
cd delay_prediction_platform

git switch main
git pull

git switch feature/НАЗВАНИЕ_ВЕТКИ
git merge main
```

После этого активируйте окружение и запускайте Jupyter:

```bash
source .venv/bin/activate
jupyter lab
```

Если после обновления изменился `requirements.txt`, дополнительно выполните:

```bash
pip install -r requirements.txt
```

### Как сохранять свою работу

После законченного небольшого этапа:

```bash
git status
git add .
git commit -m "Коротко опишите, что сделано"
git push
```

Не работайте напрямую в `main`: изменения из рабочих веток добавляем в `main` через Pull Request.

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
