"""Управляемый Docker-эмулятор; Linux host network соединяет его с локальным TCP."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
STOP = False


def stop(*_):
    global STOP
    STOP = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    if not sys.platform.startswith('linux'):
        parser.error('Автозапуск Docker поддержан на Linux; используйте --no-emulator на других ОС')
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    config = json.loads(args.config.read_text())
    port = int(os.getenv('EMULATOR_API_PORT', '18080'))
    # Не перезаписываем конфигурацию чужого работающего эмулятора.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))
    image = 'ndtp-telemetry-emulator:1.0'
    subprocess.run(['docker', 'info'], check=True, stdout=subprocess.DEVNULL)
    if subprocess.run(['docker', 'image', 'inspect', image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        archive = ROOT / 'data/dataset/ndtp-telemetry-emulator.tar'
        if not archive.exists():
            parser.error(f'Нет образа или архива {archive}')
        subprocess.run(['docker', 'load', '-i', str(archive)], check=True)
    name = f'delay-emulator-{os.getpid()}'
    created = False
    try:
        subprocess.run(['docker', 'run', '-d', '--rm', '--network', 'host', '--name', name,
                        '-e', f'SERVER_PORT={port}', image], check=True)
        created = True
        deadline = time.monotonic() + 90
        while not STOP:
            if time.monotonic() > deadline:
                raise TimeoutError('Эмулятор/приёмник не готовы за 90 секунд')
            try:
                with socket.create_connection((config['targetHost'], config['targetPort']), timeout=1):
                    pass
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/cells', timeout=2):
                    pass
                break
            except OSError:
                time.sleep(1)
        if STOP:
            return
        request = urllib.request.Request(f'http://127.0.0.1:{port}/api/config',
            data=json.dumps(config).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=10) as response:
            print(f'Эмулятор передаёт {len(config["units"])} устройств: HTTP {response.status}', flush=True)
        while not STOP:
            state = subprocess.check_output(['docker', 'inspect', '-f', '{{.State.Running}}', name], text=True).strip()
            if state != 'true':
                raise RuntimeError('Контейнер эмулятора остановлен')
            time.sleep(1)
    finally:
        if created:
            subprocess.run(['docker', 'stop', '--time', '3', name], check=False, stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
