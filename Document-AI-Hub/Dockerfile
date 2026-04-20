# Multi-stage Dockerfile for DocumentAI-Hub
FROM python:3.11-slim as builder
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential libpq-dev curl gcc g++ --no-install-recommends && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

FROM python:3.11-slim as runner
RUN apt-get update && apt-get install -y libpq5 tesseract-ocr ffmpeg curl --no-install-recommends && rm -rf /var/lib/apt/lists/*
ENV PATH="/install/bin:$PATH"
COPY --from=builder /install /usr/local
WORKDIR /app
COPY backend/src ./app
COPY .env ./
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
