# Delay Prediction Platform

## Работа с репозиторием

> **Важно:** если вы загружаете новую библиотеку, добавьте её в `requirements.txt` и закоммитьте это изменение. 

#### 1. Клонируйте репозиторий

```bash
git clone https://github.com/valentinesvev/delay_prediction_platform.git
cd delay_prediction_platform
```

#### 2. Перейдите в свою ветку

Если ветка уже создана на GitHub:

```bash
git fetch
git switch --track origin/feature/НАЗВАНИЕ_ВЕТКИ
```

Проверьте, в какой ветке вы находитесь:

```bash
git branch
```

Звёздочка `*` должна стоять напротив вашей ветки.

Перед началом работы обновите `main`, затем подтяните его изменения в свою ветку:

```bash
git switch main
git pull

git switch feature/НАЗВАНИЕ_ВЕТКИ
git merge main
```

Если после обновления изменился `requirements.txt`, дополнительно выполните:

```bash
pip install -r requirements.txt
```

### Как сохранять свою работу

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
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload

...
```

## Тесты

```bash
python -m pytest -q
```
