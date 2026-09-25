#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${PYTHON_BIN:-.venv/bin/python}"
pids=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait || true
}
trap cleanup EXIT
trap 'exit 0' INT TERM
"$python_bin" -m data.ingestion &
pids+=("$!")
"$python_bin" -m backend.prediction_worker &
pids+=("$!")
"$python_bin" -m uvicorn backend.api:app --host 127.0.0.1 --port "${PORT:-8000}" &
pids+=("$!")
wait -n "${pids[@]}"
