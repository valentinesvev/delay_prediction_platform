"""Подготовка исторического или живого запуска без переустановки зависимостей."""
import argparse
import csv
import importlib
import json
import os
import sqlite3
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def demo_plan(directory):
    """Синтетический план для проверки связности; не реальное расписание."""
    from ndtp_ingestion.server import stamp
    directory.mkdir(parents=True, exist_ok=True)
    units = [1166336, 1166337, 1166338]
    mapping = {str(unit): 900001 + i for i, unit in enumerate(units)}
    now = time.time()
    path = directory / 'schedule.csv'
    with path.open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=['tt_action_item_id', 'tr_id', 'time_begin', 'manual_fill', 'geom', 'building_address'])
        writer.writeheader()
        for unit in units:
            # Начальная позиция автогенератора из спецификации эмулятора.
            lon, lat = 37.50 + (unit % 1000) / 10000, 55.70 + (unit % 1000) / 10000
            for step in range(-20, 81):
                writer.writerow(dict(tt_action_item_id=mapping[str(unit)] * 1000 + step + 20,
                    tr_id=mapping[str(unit)], time_begin=stamp(now + step * 90), manual_fill=False,
                    geom=f'POINT ({lon + step * .003} {lat})', building_address=f'Демо-остановка {step + 21}'))
    map_path = directory / 'units.json'
    map_path.write_text(json.dumps(mapping), encoding='utf-8')
    config_path = directory / 'emulator.json'
    config_path.write_text(json.dumps(dict(targetHost='127.0.0.1', targetPort=int(os.getenv('NDTP_PORT', '9000')),
        units=[dict(unitId=u, intervalMs=2000, autoGenerate=True, cells=[]) for u in units])), encoding='utf-8')
    return path, map_path, config_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['historical', 'emulator'], default='historical')
    parser.add_argument('--db', type=Path, help='Отдельная SQLite-база; по умолчанию demo.db/live.db')
    parser.add_argument('--port', type=int, default=int(os.getenv('PORT', '8000')))
    parser.add_argument('--demo-plan', action='store_true', help='Явно создать синтетический план и 3 демо-ТС')
    parser.add_argument('--schedule', type=Path, help='Актуальный CSV плана, время UTC')
    parser.add_argument('--unit-map', type=Path, help='JSON: ID устройства → ID ТС')
    parser.add_argument('--emulator-config', type=Path)
    parser.add_argument('--no-emulator', action='store_true', help='Только приёмник; внешний эмулятор запускается отдельно')
    parser.add_argument('--install', action='store_true', help='Установить/дополнить зависимости (нужен интернет)')
    parser.add_argument('--test', action='store_true', help='Запустить тесты перед стартом')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error('Проверенная версия Python — 3.12. Укажите её через PYTHON_BIN')
    if not 1 <= args.port <= 65535:
        parser.error('Порт должен быть от 1 до 65535')
    if args.demo_plan and (args.mode != 'emulator' or args.schedule or args.unit_map or args.emulator_config):
        parser.error('--demo-plan используется только в emulator, без собственных входных файлов')
    if args.mode == 'emulator' and not args.demo_plan:
        if not args.schedule or not args.unit_map or (not args.no_emulator and not args.emulator_config):
            parser.error('Нужны --schedule, --unit-map и --emulator-config (или --no-emulator); для демо: --demo-plan')
    os.chdir(ROOT)
    if args.install:
        if sys.platform.startswith('linux'):
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', '--index-url', 'https://download.pytorch.org/whl/cpu'], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    for module in ['numpy', 'pandas', 'sklearn', 'catboost', 'lightgbm', 'xgboost', 'fastapi', 'pydantic', 'uvicorn', 'torch', 'onnx', 'sqlalchemy', 'psycopg2', 'httpx', 'pytest']:
        try:
            importlib.import_module(module)
        except ImportError as exc:
            parser.error(f'Не удалось импортировать {module}: {exc}. Повторите с --install')
    if args.test:
        subprocess.run([sys.executable, '-m', 'pytest', '-q'], check=True)
    db = (args.db or ROOT / 'artifacts' / ('live.db' if args.mode == 'emulator' else 'demo.db')).resolve()
    db.parent.mkdir(parents=True, exist_ok=True)
    os.environ['DATABASE_URL'] = 'sqlite:///' + str(db)
    command = [sys.executable, 'scripts/run_demo.py', '--port', str(args.port)]
    if args.mode == 'historical':
        os.environ['ML_TIME_MODE'] = 'stream'
        if not db.exists():
            subprocess.run([sys.executable, '-m', 'ml.csv_to_db', '--data', 'data/dataset', '--url', os.environ['DATABASE_URL']], check=True)
        else:
            with sqlite3.connect(db.as_uri() + '?mode=ro', uri=True) as conn:
                tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if not {'telemetry', 'schedule_plan'} <= tables:
                    parser.error('Существующая база не содержит telemetry/schedule_plan; укажите другую --db')
        print('Исторический режим: CSV загружается только при отсутствии базы.', flush=True)
    else:
        from ndtp_ingestion.db_init import init_db
        from ndtp_ingestion.load_schedule import load_schedule
        os.environ['ML_TIME_MODE'] = 'wall'
        os.environ['DEMO_PLAN'] = '1' if args.demo_plan else '0'
        os.environ.setdefault('ML_INTERVAL_S', '10')
        if args.demo_plan:
            # Новый файл для каждого запуска демонстрации: не смешиваем планы разных запусков.
            if args.db is None:
                db = ROOT / 'artifacts' / f'emulator-demo-{time.time_ns()}.db'
                os.environ['DATABASE_URL'] = 'sqlite:///' + str(db)
            args.schedule, args.unit_map, args.emulator_config = demo_plan(ROOT / 'artifacts' / f'emulator-{time.time_ns()}')
            print('ДЕМО: синтетический план и случайное движение; прогнозы не являются оценкой реального маршрута.', flush=True)
        os.environ['NDTP_UNIT_MAP'] = str(args.unit_map.resolve())
        # Проверить файл до запуска процессов.
        with args.unit_map.open(encoding='utf-8') as source:
            mapping = json.load(source)
        if not mapping:
            parser.error('Карта устройств пустая')
        init_db()
        print(f'План: {load_schedule(args.schedule)} остановок; историческая телеметрия НЕ загружается.', flush=True)
        command += ['--ingestion']
        if not args.no_emulator:
            command += ['--emulator-config', str(args.emulator_config.resolve())]
    print(f'База: {db}', flush=True)
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
