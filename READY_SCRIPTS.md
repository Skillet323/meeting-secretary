# Updated scripts and runtime changes

## Backend
- `backend/app/services/task_extraction.py`
  - OpenRouter extraction with stricter JSON parsing, `speaker_hint`, `source_snippet`, and immediate fallback on rate limits.
- `backend/app/services/transcription.py`
  - Speaker-labeled transcript output and inferred `speaker_aliases`.
- `backend/app/services/assignment_engine.py`
  - Better assignment using names, roles, speaker aliases, and safer fallback.
- `backend/app/services/evaluation.py`
  - Better matching, blended overall score, and extra metadata in evaluation details.
- `backend/app/api/routes.py`
  - Passes speaker transcript to task extraction, stores task debug metadata, and fixes processing flow.
- `backend/app/api/evaluation.py`
  - Evaluation summaries now expose WER/CER/F1/parse-stage/provider fields for the frontend.

## Frontend
- `frontend/src/App.jsx`
  - Expanded dashboard with Analytics tab and charts.
- `frontend/src/api.js`
  - Added `evaluateMeeting()` and `getEvaluationDetails()`.
- `frontend/vite.config.js`
  - Simplified proxy config.

## Batch scripts
- `data/build_ami_gold.py`
  - Works with the updated extractor and debug fields.
- `backend/scripts/import_ami_gold.py`
  - Imports generated JSON gold sets into the DB.

## Run order
1. Start backend.
2. Start frontend.
3. Upload audio.
4. Run evaluation from the Metrics tab or via `POST /evaluate/meeting/{id}`.
5. Use the Analytics tab to compare runs.
