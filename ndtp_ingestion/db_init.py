import sqlite3
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def init_db():
    # Создаем файл базы данных (или подключаемся, если он уже есть)
    conn = sqlite3.connect('transport_data.db')
    cursor = conn.cursor()

    logging.info("Создание таблицы telemetry...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS telemetry (
            tr_id INTEGER,
            event_time DATETIME,
            receive_time DATETIME,
            lon REAL,
            lat REAL,
            speed REAL,
            location_valid BOOLEAN,
            PRIMARY KEY (tr_id, event_time)
        )
    ''')

    logging.info("Создание таблицы schedule_plan...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS schedule_plan (
            tt_action_item_id INTEGER,
            tr_id INTEGER,
            time_begin DATETIME,
            manual_fill BOOLEAN,
            lon REAL,
            lat REAL
        )
    ''')

    logging.info("Создание таблицы predictions...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tr_id INTEGER,
            t_forecast DATETIME,
            predicted_at DATETIME,
            target_stop_id INTEGER,
            target_time_plan DATETIME,
            predicted_delay_s REAL,
            predicted_arrival DATETIME,
            interval_lo_s REAL,
            interval_hi_s REAL,
            model_used TEXT,
            fallback_reason TEXT,
            degraded BOOLEAN
        )
    ''')

    # Создаем индексы для быстрого поиска (согласно требованиям ML)
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_telemetry_event ON telemetry(event_time)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_schedule_time ON schedule_plan(time_begin)')
    
    conn.commit()
    conn.close()
    logging.info("База данных 'transport_data.db' успешно инициализирована.")

if __name__ == '__main__':
    init_db()