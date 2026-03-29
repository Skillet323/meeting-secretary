# Meeting Secretary

Автоматическая расшифровка совещаний, извлечение задач и оценка качества.

## Стек

- **Backend**: FastAPI, Whisper (OpenAI), Transformers (T5), SQLModel
- **Frontend**: React + Vite, Material-UI
- **DB**: SQLite

## Запуск

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
# Скопируйте .env.example в .env и при необходимости отредактируйте
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API: http://localhost:8000  
Docs: http://localhost:8000/docs

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Фронтенд: http://localhost:5173

## Как использовать

1. Загрузите аудиофайл через UI (кнопка Upload)
2. Система выполнит:
   - Распознавание речи (Whisper)
   - Извлечение задач (rule-based или T5)
   - Сохранение транскрипта и задач в базу
3. Просмотрите transcript и задачи на вкладке "Transcript"
4. Оцените качество на вкладке "Metrics" (требуются gold annotations)

## Gold Standard и оценка

В папке `gold_annotations/` расположены 5 размеченных встреч (AMI ES2002a-ES2006a). Каждая содержит `transcript` и `tasks` (gold).

Для генерации метрик по этим gold стандартам:

```bash
cd backend
python create_eval_simple.py
```

Затем обновите фронтенд: вкладка Metrics покажет Task Set F1, Assignee Accuracy, Deadline Accuracy.

## Конфигурация

Настройки в `backend/.env`:

```
WHISPER_MODEL=small
WHISPER_DEVICE=cuda   # cpu если нет GPU
WHISPER_COMPUTE_TYPE=float32  # для CUDA
TASK_MODEL=google/flan-t5-base  # или rule-based если не задан
USE_OLLAMA=false
```

Для загрузки T5 нужен Hugging Face токен (установите `HF_TOKEN` в переменных окружения).

## Структура проекта

```
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   ├── models.py
│   │   ├── db.py
│   │   ├── config.py
│   │   └── services/
│   ├── gold_annotations/    # gold стандарты (примеры)
│   ├── .env.example
│   └── requirements.txt
├── frontend/
│   ├── src/
│   └── package.json
├── README.md
└── .gitignore
```

