FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./backend/
COPY dashboard/ ./dashboard/
COPY models/ ./models/
COPY logs/ ./logs/
COPY datasets/processed/sample_streams.json ./datasets/processed/sample_streams.json
EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
