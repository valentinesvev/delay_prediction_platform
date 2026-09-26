"""Импорт только плана; фактическое прибытие никогда не загружается."""
import argparse
import csv
import math
import re
from datetime import datetime, timezone
from ndtp_ingestion.db_init import connect, init_db


def load_schedule(path):
    rows = []
    with open(path, encoding='utf-8-sig', newline='') as source:
        for row in csv.DictReader(source):
            geom = row.get('geom')
            if geom:
                match = re.fullmatch(r'\s*POINT\s*\(([-\d.]+)\s+([-\d.]+)\)\s*', geom)
                if not match:
                    raise ValueError(f'Некорректная геометрия: {geom}')
                lon, lat = map(float, match.groups())
            else:
                lon, lat = float(row['lon']), float(row['lat'])
            if not (math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90):
                raise ValueError('Некорректные координаты плана')
            when = datetime.fromisoformat(row['time_begin'])
            when = when.replace(tzinfo=when.tzinfo or timezone.utc).astimezone(timezone.utc)
            rows.append((int(row['tt_action_item_id']), int(row['tr_id']),
                         when.strftime('%Y-%m-%d %H:%M:%S.%f'),
                         row.get('manual_fill', '').lower() in ('true', '1', 't'),
                         f'POINT ({lon} {lat})', row.get('building_address') or None))
    if not rows:
        raise ValueError('Расписание пустое')
    with connect() as conn:
        conn.executemany('''INSERT INTO schedule_plan VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(tr_id, tt_action_item_id, time_begin) DO UPDATE SET
            manual_fill=excluded.manual_fill, geom=excluded.geom,
            building_address=excluded.building_address''', rows)
    return len(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path')
    args = parser.parse_args()
    init_db()
    print(f'Загружено {load_schedule(args.path)} плановых остановок')
