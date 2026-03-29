# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Meeting Secretary")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://super-zebra-g47qqrjr56p63q7g-5173.app.github.dev",
        "https://super-zebra-g47qqrjr56p63q7g-8000.app.github.dev",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)