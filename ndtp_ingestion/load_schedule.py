import csv
import sqlite3
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_schedule():
    conn = sqlite3.connect('transport_data.db')
    cursor = conn.cursor()
    
    try:
        with open('schedule.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                # Вставляем данные, игнорируя фактическое время time_fact_begin (требование ML)
                cursor.execute('''
                    INSERT INTO schedule_plan (tt_action_item_id, tr_id, time_begin, manual_fill, lon, lat)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    row.get('tt_action_item_id'),
                    row.get('tr_id'),
                    row.get('time_begin'),
                    row.get('manual_fill') == 'True',
                    row.get('lon'),
                    row.get('lat')
                ))
                count += 1
        
        conn.commit()
        logging.info(f"Успешно загружено {count} строк в таблицу schedule_plan.")
    except Exception as e:
        logging.error(f"Ошибка при загрузке расписания: {e}")
    finally:
        conn.close()

if __name__ == '__main__':
    load_schedule()