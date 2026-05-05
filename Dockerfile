# Этап 1: Сборка фронтенда
FROM node:18-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
RUN npm run build

# Этап 2: Бэкенд + статика
FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Установка Python-зависимостей
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода бэкенда
COPY backend/ .

# Копирование собранного фронтенда
COPY --from=frontend-build /app/frontend/dist ./static

# HF Spaces требует порт 7860
EXPOSE 7860

# Запуск
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]