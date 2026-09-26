"""Запуск ML worker и FastAPI на одной базе: python scripts/run_demo.py."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def stop_processes(processes):
    """Останавливаем процессы и обязательно дожидаемся их завершения."""
    for process in processes:
        if process.poll() is None:
            try:
                if os.name == 'nt':
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    process.send_signal(signal.SIGINT)
            except OSError:
                pass
    deadline = time.monotonic() + 5
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=os.getenv('PORT', '8000'),
                        help='Порт API (по умолчанию PORT или 8000)')
    parser.add_argument('--ingestion', action='store_true', help='Запустить TCP-приёмник NDTP')
    parser.add_argument('--emulator-config', help='Запустить Docker-эмулятор с этим JSON')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Порт должен быть от 1 до 65535')

    if not os.getenv('DATABASE_URL'):
        parser.error('Нужен DATABASE_URL: сначала создайте/заполните общую базу')
    commands = [
        ['-m', 'ml.worker'],
        ['-m', 'uvicorn', 'backend.api:app', '--host', '127.0.0.1',
         '--port', str(args.port)],
    ]
    if args.ingestion:
        commands.insert(0, ['-m', 'ndtp_ingestion.server'])
    if args.emulator_config:
        commands.append(['scripts/run_emulator.py', args.emulator_config])
    processes = []
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    signals = [signal.SIGINT, signal.SIGTERM]
    if os.name == 'nt':
        signals.append(signal.SIGBREAK)
    previous = {sig: signal.signal(sig, request_stop) for sig in signals}
    try:
        for command in commands:
            if stopping:
                return 0
            processes.append(subprocess.Popen(
                [sys.executable, '-u', *command], cwd=ROOT,
                start_new_session=os.name != 'nt',
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
            ))
        print(f'Демо запускается: http://127.0.0.1:{args.port}. Остановка: Ctrl+C.',
              flush=True)
        while not stopping:
            for command, process in zip(commands, processes):
                code = process.poll()
                if code is not None:
                    print(f'{command[1]} завершился (код {code}). Останавливаем демо.',
                          file=sys.stderr)
                    return code if code >= 0 else 1
            time.sleep(0.2)
        return 0
    except OSError as exc:
        print(f'Не удалось запустить демо: {exc}', file=sys.stderr)
        return 1
    finally:
        stop_processes(processes)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    sys.exit(main())
