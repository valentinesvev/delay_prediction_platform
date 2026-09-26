# Используем официальный легкий образ Python
FROM python:3.12-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем наши скрипты и файл расписания
COPY server.py .
COPY db_init.py .
COPY load_schedule.py .
COPY schedule.csv .

# Открываем порт 9000 для эмулятора
EXPOSE 9000

# При старте контейнера: создаем БД -> грузим расписание -> запускаем сервер
CMD ["sh", "-c", "python3 db_init.py && python3 load_schedule.py && python3 server.py"]