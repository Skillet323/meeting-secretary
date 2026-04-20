# backend/app/services/diarization.py
"""
Speaker diarization service.

Model strategy
--------------
Primary  : pyannote/speaker-diarization-3.1
           State-of-the-art, pure PyTorch (no onnxruntime), MIT-licensed.
           Requires a free HuggingFace token with model access:
               1. Go to hf.co/pyannote/speaker-diarization-3.1 and accept terms
               2. Go to hf.co/pyannote/segmentation-3.0 and accept terms
               3. Set HF_TOKEN in .env

           DER benchmarks (pyannote 3.1, no forgiveness collar):
               AMI (IHM)  ~18 %   AMI (Mix) ~27 %   AISHELL-4 ~14 %

Fallback : resemblyzer + SpectralClustering
           No token needed, but lower accuracy.
           Activated automatically when pyannote is unavailable or HF_TOKEN unset.

GitHub Codespaces note
----------------------
Add HF_TOKEN to repo/codespace secrets.
`pip install pyannote.audio` — ~200 MB, installs cleanly on CPU.
First run downloads model weights (~140 MB) and caches them in ~/.cache/huggingface.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from ..config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# pyannote 3.1 — primary pipeline
# ---------------------------------------------------------------------------

_pyannote_pipeline = None
_pyannote_available: Optional[bool] = None  # None = not yet tried


def _load_pyannote():
    """Lazy-load pyannote pipeline once per process."""
    global _pyannote_pipeline, _pyannote_available

    if _pyannote_available is False:
        return None
    if _pyannote_pipeline is not None:
        return _pyannote_pipeline

    token: str = settings.HF_TOKEN or os.environ.get("HF_TOKEN", "")
    if not token:
        logger.warning(
            "[DIARIZATION] HF_TOKEN not set — pyannote requires a HuggingFace token. "
            "Falling back to resemblyzer."
        )
        _pyannote_available = False
        return None

    try:
        from pyannote.audio import Pipeline  # type: ignore
        import torch  # type: ignore

        logger.info("[DIARIZATION] Loading pyannote/speaker-diarization-3.1 …")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )

        device = "cuda" if settings.WHISPER_DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
        pipeline.to(torch.device(device))

        _pyannote_pipeline = pipeline
        _pyannote_available = True
        logger.info("[DIARIZATION] pyannote loaded on %s", device.upper())
        return pipeline

    except Exception as exc:
        logger.warning(
            "[DIARIZATION] pyannote load failed (%s). Falling back to resemblyzer.", exc
        )
        _pyannote_available = False
        return None


def _diarize_pyannote(wav_path: str, n_speakers: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
    """Run pyannote diarization and return list of {start, end, speaker} dicts."""
    pipeline = _load_pyannote()
    if pipeline is None:
        return None

    try:
        kwargs: Dict[str, Any] = {}
        if n_speakers is not None:
            kwargs["num_speakers"] = n_speakers

        diarization = pipeline(wav_path, **kwargs)

        segments: List[Dict[str, Any]] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),  # e.g. "SPEAKER_00"
            })

        logger.info("[DIARIZATION] pyannote → %d segments", len(segments))
        return segments if segments else None

    except Exception as exc:
        logger.error("[DIARIZATION] pyannote inference failed: %s", exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# resemblyzer — fallback pipeline
# ---------------------------------------------------------------------------

def _diarize_resemblyzer(wav_path: str, n_speakers: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
    """
    Fallback diarizer using resemblyzer + SpectralClustering.
    Lower DER than pyannote, but zero-dependency on HF token.
    """
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav  # type: ignore
        import numpy as np  # type: ignore
        from sklearn.cluster import SpectralClustering  # type: ignore
    except ImportError:
        logger.warning("[DIARIZATION] resemblyzer/sklearn not installed; diarization disabled.")
        return None

    try:
        wav = preprocess_wav(wav_path)  # float32 numpy, 16 kHz
        encoder = VoiceEncoder()
        sr = 16_000
        win = int(1.5 * sr)
        hop = int(0.75 * sr)

        embeds: list = []
        timestamps: list = []
        for start in range(0, len(wav) - win + 1, hop):
            chunk = wav[start : start + win]
            embeds.append(encoder.embed_utterance(chunk))
            timestamps.append((start / sr, (start + win) / sr))

        if not embeds:
            return None

        X = np.vstack(embeds)
        k = n_speakers if n_speakers else min(5, max(1, int(len(X) ** 0.5)))
        k = min(k, len(X))  # can't have more clusters than samples

        labels = SpectralClustering(
            n_clusters=k,
            affinity="nearest_neighbors",
            random_state=0,
        ).fit_predict(X)

        segments = [
            {"start": st, "end": en, "speaker": f"SPEAKER_{int(lab):02d}"}
            for (st, en), lab in zip(timestamps, labels)
        ]
        logger.info("[DIARIZATION] resemblyzer → %d segments (%d speakers)", len(segments), k)
        return segments

    except Exception as exc:
        logger.error("[DIARIZATION] resemblyzer failed: %s", exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def diarize_audio(wav_path: str, n_speakers: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
    """
    Returns a list of {"start": float, "end": float, "speaker": str} dicts,
    or None if diarization is unavailable/failed.

    Tries pyannote first (best accuracy), falls back to resemblyzer.
    """
    # --- pyannote (primary) ---
    segments = _diarize_pyannote(wav_path, n_speakers)
    if segments is not None:
        return segments

    # --- resemblyzer (fallback) ---
    return _diarize_resemblyzer(wav_path, n_speakers)
