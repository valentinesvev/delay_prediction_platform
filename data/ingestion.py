"""Независимый генератор: python -m data.ingestion."""
import logging
import os
import time
from data.storage import database, initialize_input


def ingest_once(retention_seconds, now=None):
    now = time.time() if now is None else now
    with database('incoming.sqlite3') as db:
        db.execute('BEGIN IMMEDIATE')
        value = db.execute('SELECT next_value FROM generator_state WHERE id=1').fetchone()[0]
        # Расписание пусто; текущая задержка в этом демо всегда равна нулю.
        db.execute('INSERT INTO validated_events(state, delay, event_at) VALUES (?, 0, ?)', (value, now))
        db.execute('UPDATE generator_state SET next_value=? WHERE id=1', (value + 1,))
        db.execute('DELETE FROM validated_events WHERE event_at < ?', (now - retention_seconds,))
    return value


def main():
    interval = float(os.getenv('INPUT_INTERVAL_SECONDS', '1'))
    retention = float(os.getenv('INPUT_RETENTION_SECONDS', '1800'))
    if interval <= 0 or retention <= 0:
        raise ValueError('Интервал и срок хранения должны быть положительными')
    initialize_input()
    logging.basicConfig(level=logging.INFO, format='ingestion: %(message)s')
    while True:
        try:
            logging.info('state=%s', ingest_once(retention))
        except Exception:
            logging.exception('Ошибка приёма данных')
        time.sleep(interval)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
