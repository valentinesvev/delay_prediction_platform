# Запуск проекта

## 1. Клонировать проект

```bash
git clone <URL_РЕПОЗИТОРИЯ>
cd delay_prediction_platform
```

## 2. Live-режим — первый запуск

Положить эмулятор сюда:

```text
data/dataset/ndtp-telemetry-emulator.tar
```

Один раз загрузить образ:

```bash
docker load -i data/dataset/ndtp-telemetry-emulator.tar
```

Запустить:

```bash
docker compose up -d --build
```

Открыть:

```text
http://127.0.0.1:8000/
```

Проверить:

```bash
docker compose ps -a
```

Остановить:

```bash
docker compose down
```

## 3. Historical-режим

Эмулятор не нужен.

Запустить:

```bash
PORT=8002 docker compose -f compose.historical.yaml up -d --build
```

Открыть:

```text
http://127.0.0.1:8002/
```

Проверить:

```bash
docker compose -f compose.historical.yaml ps -a
```

Посмотреть ход replay:

```bash
docker compose -f compose.historical.yaml logs -f replay
```

Остановить:

```bash
docker compose -f compose.historical.yaml down
```

После первого `docker load` для live-режима архив каждый раз загружать не нужно: Docker уже хранит образ `ndtp-telemetry-emulator:1.0` локально.
