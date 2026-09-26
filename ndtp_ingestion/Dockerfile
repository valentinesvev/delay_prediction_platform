# Сборка из корня: docker build -f ndtp_ingestion/Dockerfile -t ndtp-server .
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir 'numpy>=2,<3' 'pandas>=2.2,<4' 'sqlalchemy>=2,<3'
COPY ml/ ml/
COPY ndtp_ingestion/ ndtp_ingestion/
EXPOSE 9000
CMD ["python", "-m", "ndtp_ingestion.server"]
