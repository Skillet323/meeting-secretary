# Patch report

## Backend
- Connected FastAPI routers in `app/main.py` so `/upload_meeting`, `/meeting/{id}`, `/metrics`, `/evaluations`, `/gold_standards` work again.
- Added startup DB initialization and a root health endpoint.
- Added `/meetings` and fixed `/metrics` serialization for frontend use.
- Improved upload flow to store audio in a temp file instead of reading it directly into RAM.
- Added cleanup for temporary audio files.
- Added compatibility alias `_find_participant_by_name` for older tests.
- Fixed evaluation logic to use direct `Task` imports and store model names in evaluation runs.
- Extended audio preprocessing to accept file paths as well as bytes.
- Kept Whisper/T5 fallback behavior so the system can still work on weak hardware.

## Frontend
- Added `getMeetings()` API call.
- Dashboard now loads recent meetings in addition to metrics and evaluations.
- Kept all API calls on the `/api` path so they work both with Vite dev proxy and Nginx proxy.

## Dev/Docker
- Added `ffmpeg` and `libsndfile1` installation to backend Dockerfiles.
- Switched backend Docker defaults to CPU-safe Whisper settings.
- Fixed frontend Nginx config to proxy `/api` to backend.
- Fixed docker-compose frontend port mapping (`5173:80`).
