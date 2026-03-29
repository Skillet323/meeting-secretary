from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router as core_router
from .api.evaluation import router as evaluation_router
from .db import init_db

app = FastAPI(title="Meeting Secretary")

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
