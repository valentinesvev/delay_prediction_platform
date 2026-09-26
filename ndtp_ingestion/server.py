"""TCP-приёмник NDTP 6.2 для поставляемого эмулятора."""
import asyncio
import json
import logging
import os
import struct
import time
from datetime import datetime, timezone
from ndtp_ingestion.db_init import connect, init_db

log = logging.getLogger('ndtp')
NPL = struct.Struct('<HHHHBIH')
NPH = struct.Struct('<HHHI')
NAV = struct.Struct('<IIIBBHHHHHBB')


def stamp(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')


def crc16(data):
    crc = 0xffff
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xa001 if crc & 1 else 0)
    return ((crc & 255) << 8) | (crc >> 8)


def decode(header, payload, unit_map, now=None):
    now = time.time() if now is None else now
    sig, size, flags, crc, kind, unit, _ = NPL.unpack(header)
    if sig != 0x7e7e or kind != 2 or flags != 0 or size != len(payload) or size < NPH.size:
        raise ValueError('Некорректный NPL')
    if crc16(payload) != crc:
        raise ValueError('Ошибка CRC')
    service, kind, _, _ = NPH.unpack_from(payload)
    body = payload[NPH.size:]
    if kind == 100 and service == 0:
        if len(body) != 18:
            raise ValueError('Некорректный handshake')
        major, minor, _, peer, _, _ = struct.unpack('<HHHIII', body)
        if (major, minor, peer) != (6, 2, unit):
            raise ValueError('Некорректная версия/устройство handshake')
        return None
    if service != 1 or kind != 101 or len(body) < 28 or body[0] != 0:
        raise ValueError('Нет первой навигационной ячейки')
    if str(unit) not in unit_map:
        raise ValueError(f'Нет соответствия unit_id={unit} → tr_id')
    timestamp, lon, lat, navflags, _, speed, *_ = NAV.unpack_from(body, 2)
    lon = lon / 1e7 * (1 if navflags & 0x40 else -1)
    lat = lat / 1e7 * (1 if navflags & 0x20 else -1)
    if not navflags & 0x80 or not (36.5 < lon < 38.5 and 55 < lat < 56.5) or speed > 120:
        raise ValueError('Невалидные координаты/скорость (область модели — Москва)')
    if not now - 7200 <= timestamp <= now + 60:
        raise ValueError('Время точки вне допустимого окна: -2 часа … +60 секунд')
    return (int(unit_map[str(unit)]), unit, stamp(timestamp), stamp(now), lon, lat, speed, True)


def save_point(point):
    with connect() as conn:
        result = conn.execute('''INSERT INTO telemetry
            (tr_id, unit_id, event_time, receive_time, lon, lat, speed, location_valid)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(tr_id, event_time) DO NOTHING''', point)
        return result.rowcount


def retentions():
    telemetry_s = float(os.getenv('TELEMETRY_RETENTION_S', '7200'))
    schedule_s = float(os.getenv('SCHEDULE_RETENTION_S', '21600'))
    predictions_s = float(os.getenv('PREDICTIONS_RETENTION_S', '86400'))
    if telemetry_s < float(os.getenv('ML_WINDOW_S', '2700')) + 60 or schedule_s < 21600 or predictions_s <= 0:
        raise ValueError('Сроки хранения слишком малы для ML')
    return telemetry_s, schedule_s, predictions_s


def cleanup(now=None):
    now = time.time() if now is None else now
    telemetry_s, schedule_s, predictions_s = retentions()
    result = {}
    with connect() as conn:
        for table, field, age in [('telemetry', 'receive_time', telemetry_s),
                                  ('schedule_plan', 'time_begin', schedule_s),
                                  ('predictions', 'predicted_at', predictions_s)]:
            result[table] = conn.execute(f'DELETE FROM {table} WHERE {field} < ?', (stamp(now - age),)).rowcount
    return result


async def garbage_collector():
    interval = float(os.getenv('CLEANUP_INTERVAL_S', '600'))
    if interval <= 0:
        raise ValueError('CLEANUP_INTERVAL_S должен быть положительным')
    while True:
        try:
            log.info('Очистка: %s', await asyncio.to_thread(cleanup))
        except Exception:
            log.exception('Ошибка очистки')
        await asyncio.sleep(interval)


async def handle_client(reader, writer, unit_map):
    peer = writer.get_extra_info('peername')
    log.info('Подключение %s', peer)
    try:
        while True:
            header = await reader.readexactly(NPL.size)
            sig, size, *_ = NPL.unpack(header)
            if sig != 0x7e7e or size < NPH.size:
                raise ValueError('Некорректная граница пакета')
            payload = await reader.readexactly(size)
            try:
                point = decode(header, payload, unit_map)
                if point:
                    written = await asyncio.to_thread(save_point, point)
                    log.debug('ТС %s: %s', point[0], 'записано' if written else 'дубль')
            except ValueError as exc:
                log.warning('Пакет отклонён: %s', exc)
    except (asyncio.IncompleteReadError, ConnectionError):
        pass
    except Exception:
        log.exception('Ошибка соединения %s', peer)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except ConnectionError:
            pass


async def main():
    init_db()
    with open(os.environ['NDTP_UNIT_MAP'], encoding='utf-8') as source:
        unit_map = json.load(source)
    if not unit_map or any(int(k) < 0 or int(v) <= 0 for k, v in unit_map.items()):
        raise ValueError('Нужен непустой JSON unit_id → tr_id')
    retentions()
    if float(os.getenv('CLEANUP_INTERVAL_S', '600')) <= 0:
        raise ValueError('CLEANUP_INTERVAL_S должен быть положительным')
    collector = asyncio.create_task(garbage_collector())
    server = await asyncio.start_server(lambda r, w: handle_client(r, w, unit_map),
                                      os.getenv('NDTP_HOST', '0.0.0.0'), int(os.getenv('NDTP_PORT', '9000')))
    log.info('NDTP слушает %s; устройств в карте: %s', server.sockets[0].getsockname(), len(unit_map))
    try:
        async with server:
            await server.serve_forever()
    finally:
        collector.cancel()
        await asyncio.gather(collector, return_exceptions=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
