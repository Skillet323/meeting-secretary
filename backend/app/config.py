from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    DATABASE_URL: str = "sqlite:///./meeting_secretary.db"

    # Whisper ASR settings
    WHISPER_MODEL: str = "small"  # tiny, base, small, medium, large-v3
    WHISPER_DEVICE: str = "cpu"   # "cuda" or "cpu"
    WHISPER_COMPUTE_TYPE: str = "float16"  # for GPU: float16, int8_float16; for CPU: int8

    # Task extraction LLM settings
    TASK_MODEL: str = "google/flan-t5-base"
    MAX_TASK_MODEL_TOKENS: int = 1024
    USE_OLLAMA: bool = False
    OLLAMA_MODEL: str = "codellama:7b"
    OLLAMA_URL: str = "http://localhost:11434"

    # Diarization
    DIARIZATION_ENABLED: bool = True
    DIARIZATION_MODEL: str = "resemblyzer"

    # Audio preprocessing
    AUDIO_TARGET_SR: int = 16000
    AUDIO_NORMALIZE: bool = True
    AUDIO_REDUCE_NOISE: bool = False

    # Quality thresholds
    MIN_TRANSCRIPT_CONFIDENCE: float = 0.6

    # Hugging Face token (if needed for gated models)
    HF_TOKEN: str | None = Field(default=None, description="HuggingFace token")


settings = Settings()
