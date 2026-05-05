# backend/app/services/diarization.py
"""
Speaker diarization service.

Primary:
    pyannote/speaker-diarization-community-1
    Current pyannote README shows loading this pipeline with:
        Pipeline.from_pretrained(..., token="HUGGINGFACE_ACCESS_TOKEN")
    The older 3.1 pipeline is still gated and requires accepted terms.

Fallback:
    resemblyzer + SpectralClustering
    Lower accuracy, but no Hugging Face token required.

This module also provides helper utilities to infer speaker-name aliases
from transcript self-introductions like:
    "I'm Laura"
    "My name is David"
    "This is Craig"
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from ..config import settings

logger = logging.getLogger(__name__)

_pyannote_pipeline = None
_pyannote_available: Optional[bool] = None
_pyannote_model_id: Optional[str] = None


def _load_pyannote():
    global _pyannote_pipeline, _pyannote_available, _pyannote_model_id

    if _pyannote_available is False:
        return None
    if _pyannote_pipeline is not None:
        return _pyannote_pipeline

    token: str = settings.HF_TOKEN or os.environ.get("HF_TOKEN", "")
    if not token:
        logger.warning(
            "[DIARIZATION] HF_TOKEN not set — pyannote requires a HuggingFace token. Falling back to resemblyzer."
        )
        _pyannote_available = False
        return None

    try:
        from pyannote.audio import Pipeline  # type: ignore
        import torch  # type: ignore

        preferred_models = [
            "pyannote/speaker-diarization-community-1",
            "pyannote/speaker-diarization-3.1",
        ]

        pipeline = None
        last_error: Exception | None = None

        for model_id in preferred_models:
            try:
                logger.info("[DIARIZATION] Loading %s …", model_id)
                try:
                    pipeline = Pipeline.from_pretrained(model_id, token=token)
                except TypeError:
                    pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)
                _pyannote_model_id = model_id
                break
            except Exception as exc:
                last_error = exc
                logger.warning("[DIARIZATION] Failed to load %s: %s", model_id, exc)

        if pipeline is None:
            raise RuntimeError(f"All pyannote models failed to load: {last_error}")

        device = "cuda" if settings.WHISPER_DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
        pipeline.to(torch.device(device))

        _pyannote_pipeline = pipeline
        _pyannote_available = True
        logger.info("[DIARIZATION] pyannote loaded on %s (%s)", device.upper(), _pyannote_model_id)
        return pipeline

    except Exception as exc:
        logger.warning("[DIARIZATION] pyannote load failed (%s). Falling back to resemblyzer.", exc)
        _pyannote_available = False
        return None


def _merge_adjacent_segments(segments: List[Dict[str, Any]], max_gap: float = 0.25) -> List[Dict[str, Any]]:
    if not segments:
        return []

    merged: List[Dict[str, Any]] = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg["speaker"] == prev["speaker"] and float(seg["start"]) - float(prev["end"]) <= max_gap:
            prev["end"] = max(float(prev["end"]), float(seg["end"]))
        else:
            merged.append(seg.copy())
    return merged


def _diarize_pyannote(wav_path: str, n_speakers: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
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
            segments.append(
                {
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": str(speaker),
                }
            )

        segments = _merge_adjacent_segments(segments)
        logger.info("[DIARIZATION] pyannote → %d segments", len(segments))
        return segments if segments else None

    except Exception as exc:
        logger.error("[DIARIZATION] pyannote inference failed: %s", exc, exc_info=True)
        return None


def _diarize_resemblyzer(wav_path: str, n_speakers: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav  # type: ignore
        import numpy as np  # type: ignore
        from sklearn.cluster import SpectralClustering  # type: ignore
    except ImportError:
        logger.warning("[DIARIZATION] resemblyzer/sklearn not installed; diarization disabled.")
        return None

    try:
        wav = preprocess_wav(wav_path)
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
        duration_sec = len(wav) / sr

        # Very short files: avoid hallucinating multiple speakers.
        if n_speakers is None and (duration_sec < 120 or len(X) <= 4):
            segment = {"start": 0.0, "end": float(duration_sec), "speaker": "SPEAKER_00"}
            logger.info("[DIARIZATION] resemblyzer → 1 segment (single-speaker guard)")
            return [segment]

        if len(X) == 1:
            segment = {"start": float(timestamps[0][0]), "end": float(timestamps[0][1]), "speaker": "SPEAKER_00"}
            logger.info("[DIARIZATION] resemblyzer → 1 segment (1 speaker)")
            return [segment]

        if n_speakers and n_speakers > 0:
            k = min(n_speakers, len(X))
        else:
            k = min(5, max(1, int(len(X) ** 0.5)))
            k = min(k, len(X))

        if k <= 1:
            segments = [
                {"start": float(st), "end": float(en), "speaker": "SPEAKER_00"}
                for st, en in timestamps
            ]
            logger.info("[DIARIZATION] resemblyzer → %d segments (1 speaker)", len(segments))
            return _merge_adjacent_segments(segments)

        labels = SpectralClustering(
            n_clusters=k,
            affinity="nearest_neighbors",
            random_state=0,
        ).fit_predict(X)

        segments = [
            {"start": float(st), "end": float(en), "speaker": f"SPEAKER_{int(lab):02d}"}
            for (st, en), lab in zip(timestamps, labels)
        ]
        segments = _merge_adjacent_segments(segments)
        logger.info("[DIARIZATION] resemblyzer → %d segments (%d speakers)", len(segments), k)
        return segments

    except Exception as exc:
        logger.error("[DIARIZATION] resemblyzer failed: %s", exc, exc_info=True)
        return None


_NAME_PATTERNS = [
    r"\b(?:i'm|i am|my name is|this is|name's)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    r"\b(?:i am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
]


def extract_explicit_names_from_transcript(transcript: str) -> list[str]:
    names: list[str] = []
    if not transcript:
        return names

    for pattern in _NAME_PATTERNS:
        for m in re.finditer(pattern, transcript, flags=re.IGNORECASE):
            name = m.group(1).strip()
            if name and name not in names:
                names.append(name)
    return names


def infer_speaker_alias_map_from_transcript(transcript: str) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    if not transcript:
        return alias_map

    for raw_line in transcript.splitlines():
        m = re.match(r"^\s*((?:SPEAKER_\d+)|(?:Speaker\s+\d+)|(?:[A-Z]))\s*:\s*(.+)$", raw_line)
        if not m:
            continue

        speaker_label = m.group(1).strip()
        body = m.group(2).strip()

        for pattern in _NAME_PATTERNS:
            found = re.search(pattern, body, flags=re.IGNORECASE)
            if found:
                candidate = found.group(1).strip()
                if candidate and speaker_label not in alias_map:
                    alias_map[speaker_label] = candidate
                break

    return alias_map


def apply_speaker_aliases(
    segments: List[Dict[str, Any]],
    alias_map: dict[str, str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for seg in segments:
        speaker = str(seg.get("speaker", "")).strip()
        alias = alias_map.get(speaker)

        item = dict(seg)
        item["speaker_label"] = speaker
        item["speaker_name"] = alias
        item["speaker_display"] = alias or speaker
        out.append(item)
    return out


def diarize_audio(wav_path: str, n_speakers: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
    segments = _diarize_pyannote(wav_path, n_speakers)
    if segments is not None:
        return segments
    return _diarize_resemblyzer(wav_path, n_speakers)