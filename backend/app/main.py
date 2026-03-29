from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .api.routes import router as core_router
from .api.evaluation import router as evaluation_router
from .db import init_db

# Increase max body size to 2GB for large audio files
class LimitUploadSize(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size: int):
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method == 'POST':
            if 'content-length' in request.headers:
                content_length = int(request.headers['content-length'])
                if content_length > self.max_upload_size:
                    return Response("File too large", status_code=413)
        return await call_next(request)

app = FastAPI(title="Meeting Secretary")
# Allow up to 2GB uploads (2 * 1024 * 1024 * 1024 bytes)
app.add_middleware(LimitUploadSize, max_upload_size=2 * 1024 * 1024 * 1024)

# In Codespaces and local dev the frontend usually talks to the backend through
# Vite/Nginx proxies, but having permissive CORS makes direct calls easier to debug.
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://super-zebra-g47qqrjr56p63q7g-5173.app.github.dev",
    "https://super-zebra-g47qqrjr56p63q7g-8000.app.github.dev",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.app\.github\.dev",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    init_db()


app.include_router(core_router)
app.include_router(evaluation_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "meeting-secretary"}
