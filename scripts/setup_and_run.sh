#!/usr/bin/env bash
# Используем существующее окружение; сеть нужна только с --install.
set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_path="${PYTHON_BIN:-$project_root/.venv/bin/python}"
if [[ ! -x "$python_path" ]]; then
  echo 'Не найден Python в .venv. Укажите PYTHON_BIN или создайте окружение Python 3.12.' >&2
  exit 1
fi
exec "$python_path" -m scripts.setup "$@"
