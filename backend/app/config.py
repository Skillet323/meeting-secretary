# backend/app/config.py  —  обновлённая версия
# Добавлены поля для OpenRouter (task extraction) и HF_TOKEN (pyannote diarization).
# Все новые поля опциональны со значениями по умолчанию — старый .env продолжает работать.

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DATABASE_URL: str = "sqlite:///./meeting_secretary.db"

    # ── Whisper ASR ──────────────────────────────────────────────────────────
    WHISPER_MODEL: str = "small"         # tiny | base | small | medium | large-v3
    WHISPER_DEVICE: str = "cpu"          # "cuda" | "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"   # int8 (CPU, default) | float16 (GPU only)

    # ── Task extraction ───────────────────────────────────────────────────────
    # TASK_PROVIDER selects the extraction backend:
    #   "openrouter"  → call OpenRouter API (default; best quality, free tier available)
    #   "rules"       → pure heuristic, no network, useful for offline/CI runs
    TASK_PROVIDER: str = "openrouter"

    # OpenRouter settings (used when TASK_PROVIDER == "openrouter")
    # Free models: "openrouter/free" auto-selects, or pin to e.g.:
    #   "meta-llama/llama-3.3-70b-instruct:free"
    #   "google/gemma-3-27b-it:free"
    #   "mistralai/mistral-7b-instruct:free"
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_TASK_MODEL: str = "openrouter/free"

    # Legacy T5 fields — kept for backwards compatibility, no longer used by default
    TASK_MODEL: str = "google/flan-t5-base"
    MAX_TASK_MODEL_TOKENS: int = 1024
    USE_OLLAMA: bool = False
    OLLAMA_MODEL: str = "codellama:7b"
    OLLAMA_URL: str = "http://localhost:11434"

    # ── Speaker diarization ───────────────────────────────────────────────────
    DIARIZATION_ENABLED: bool = True
    # DIARIZATION_MODEL is now advisory only; actual selection is automatic:
    #   pyannote/speaker-diarization-3.1 (needs HF_TOKEN) → primary
    #   resemblyzer + SpectralClustering                   → fallback
    DIARIZATION_MODEL: str = "pyannote"

    # ── HuggingFace token (required for pyannote gated models) ───────────────
    # Set in .env or as a Codespace secret.
    # Get a free token at https://hf.co/settings/tokens, then accept the
    # conditions on:
    #   https://hf.co/pyannote/speaker-diarization-3.1
    #   https://hf.co/pyannote/segmentation-3.0
    HF_TOKEN: str | None = Field(default=None, description="HuggingFace access token")

    # ── Audio preprocessing ───────────────────────────────────────────────────
    AUDIO_TARGET_SR: int = 16000
    AUDIO_NORMALIZE: bool = True
    AUDIO_REDUCE_NOISE: bool = False

    # ── Quality thresholds ────────────────────────────────────────────────────
    MIN_TRANSCRIPT_CONFIDENCE: float = 0.6


settings = Settings()
