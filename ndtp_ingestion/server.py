import asyncio
import logging
import struct
import sqlite3
import time
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_NAME = 'transport_data.db'

def save_telemetry_to_db(tr_id, event_time_sec, lon, lat, speed, location_valid):
    """Сохраняет валидную точку в базу данных."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Конвертируем unix_timestamp в строку datetime UTC
        event_time = datetime.fromtimestamp(event_time_sec, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        receive_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        
        cursor.execute('''
            INSERT OR IGNORE INTO telemetry 
            (tr_id, event_time, receive_time, lon, lat, speed, location_valid)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (tr_id, event_time, receive_time, lon, lat, speed, location_valid))
        
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Ошибка записи в БД: {e}")

async def garbage_collector():
    """Фоновый процесс: удаляет старые данные из базы (храним только последние 2 часа телеметрии)."""
    while True:
        try:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            
            # Удаляем телеметрию старше 2 часов от текущего времени получения
            cursor.execute("DELETE FROM telemetry WHERE receive_time < datetime('now', '-2 hours')")
            deleted = cursor.rowcount
            if deleted > 0:
                logging.info(f"Сборщик мусора: удалено {deleted} старых записей телеметрии.")
            
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Ошибка сборщика мусора: {e}")
        
        await asyncio.sleep(600) # Запускаем каждые 10 минут

async def handle_client(reader, writer):
    addr = writer.get_extra_info('peername')
    logging.info(f"Подключен эмулятор: {addr}")
    
    try:
        while True:
            # Читаем NPL (15 байт)
            npl_data = await reader.readexactly(15)
            # Распаковка NPL: signature(H), dataSize(H), flags(H), crc(H), type(B), peerAddress(I), requestId(H)
            npl = struct.unpack('<H H H H B I H', npl_data)
            
            if npl[0] != 0x7E7E: # Проверка сигнатуры
                logging.warning("Неверная сигнатура пакета. Отключаем.")
                break
                
            unit_id = npl[5]
            data_size = npl[1]
            
            # Читаем NPH и тело пакета
            payload_data = await reader.readexactly(data_size)
            
            # Распаковка NPH (10 байт): serviceId(H), type(H), flags(H), requestId(I)
            nph = struct.unpack('<H H H I', payload_data[:10])
            nph_type = nph[1]
            
            body = payload_data[10:]
            
            if nph_type == 100:
                logging.info(f"[{unit_id}] Получен Handshake (CONN_REQUEST).")
                # В ответ ничего слать не нужно по спецификации эмулятора
            
            elif nph_type == 101:
                # Realtime пакет, парсим ячейки
                offset = 0
                while offset < len(body):
                    cell_type = body[offset]
                    cell_num = body[offset+1]
                    offset += 2
                    
                    if cell_type == 0: # G6CellNav00 (26 байт)
                        cell_data = body[offset:offset+26]
                        if len(cell_data) == 26:
                            nav = struct.unpack('<I I I B B H H H H H B B', cell_data)
                            
                            timestamp = nav[0]
                            lon_raw = nav[1]
                            lat_raw = nav[2]
                            flags = nav[3]
                            speed = nav[5]
                            
                            # Математика из документации:
                            lon = lon_raw / 10000000.0
                            lat = lat_raw / 10000000.0
                            valid = bool(flags & 0x80) # extraDopBit7 (валидность)
                            
                            if valid and lon > 0 and lat > 0:
                                logging.info(f"[{unit_id}] Телеметрия: {lat:.6f}, {lon:.6f} | {speed} км/ч")
                                save_telemetry_to_db(unit_id, timestamp, lon, lat, speed, valid)
                        offset += 26
                    else:
                        # Пропускаем неизвестные ячейки (нам нужна только навигация)
                        break 
                        
    except asyncio.IncompleteReadError:
        logging.info(f"Отключение эмулятора: {addr}")
    except Exception as e:
        logging.error(f"Ошибка соединения {addr}: {e}")
    finally:
        writer.close()
        await writer.wait_closed()

async def main():
    # Запускаем фоновый сборщик мусора
    asyncio.create_task(garbage_collector())
    
    server = await asyncio.start_server(handle_client, '0.0.0.0', 9000)
    logging.info("TCP-сервер (NDTP Ingestion) запущен. Ожидание эмулятора...")
    async with server:
        await server.serve_forever()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Сервер остановлен.")