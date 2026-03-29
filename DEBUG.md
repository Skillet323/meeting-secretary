# Debugging Meeting Secretary

## Quick Start

1. Install dependencies:
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

2. Configure `.env` (copy from `.env.example`):
- Set `WHISPER_DEVICE=cpu` if no GPU (or `cuda` if GPU available)
- Set `WHISPER_COMPUTE_TYPE=float32` for CUDA compatibility
- Set `HF_TOKEN` environment variable (for T5 download) or set in `.env`

3. Run backend:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Check health: `curl http://localhost:8000/health`

4. Run frontend:
```bash
cd frontend
npm install
npm run dev
```

## Common Issues

### 1. T5 model not loading (HF_TOKEN)
- Set `HF_TOKEN` in environment or `.env`
- First run will download ~200MB model (cached in `~/.cache/huggingface`)

### 2. Whisper error: "expected scalar type Float but found Half"
- Cause: `WHISPER_COMPUTE_TYPE=float16` incompatible with CUDA 13+
- Fix: set `WHISPER_COMPUTE_TYPE=float32` or use `WHISPER_DEVICE=cpu`

### 3. 500 error on `/evaluations`
- Check DB tables exist: `evaluation_run`, `evaluation_metric`
- If missing, ensure `EvaluationRun` and `EvaluationMetric` are imported in `app/main.py`
- Delete `meeting_secretary.db` and restart to recreate tables

### 4. No tasks extracted
- For English: rule-based works; T5 should produce JSON output. Check logs for `[TASK]` messages.
- Ensure `TASK_MODEL` is set and model downloaded.
- Check that `MAX_TASK_MODEL_TOKENS` is sufficient (1024 default).

### 5. Frontend shows white page / JS errors
- Check browser console for missing imports (e.g., `List` from MUI). Ensure `App.jsx` imports all MUI components used.
- Do a hard refresh (Ctrl+F5).

## Logs

- Backend logs appear in console where `uvicorn` runs.
- For debugging, you can test imports directly:
```bash
cd backend
.venv\Scripts\activate
python -c "from app.services.transcription import get_model; get_model()"
```
- Frontend logs in browser console.

## Database Inspection

```bash
cd backend
sqlite3 meeting_secretary.db
.tables
SELECT * FROM meeting;
SELECT * FROM task;
SELECT * FROM evaluation_run;
```

## Testing Endpoints

- Health: `GET /health`
- List meetings: `GET /meetings`
- Get meeting: `GET /meeting/{id}`
- Upload: `POST /upload_meeting` (multipart/form-data)
- Gold standards: `GET /gold_standards`
- Evaluations: `GET /evaluations`
- Evaluation details: `GET /evaluation/{id}`

Use `curl` or Postman.

## Performance

- Whisper `small` on CPU: ~2-5x real-time (depends on CPU)
- T5-base task extraction: ~1-2 seconds per transcript (GPU) or 5-10s (CPU)
- Total latency dominated by ASR.

## Reprovisioning

If you need to reset DB:
```bash
cd backend
del meeting_secretary.db
# restart server, then load gold and create meetings via scripts:
python populate_test_db.py
```

## Known Limitations

- BERTScore and ROUGE-L disabled to avoid HF downloads; simple token-overlap used instead.
- Russian language not well supported by rule-based; T5 may handle if fine-tuned.
- No speaker diarization (disabled).
- No webhooks or external integrations (code exists but not configured).
