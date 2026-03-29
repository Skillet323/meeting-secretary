"""
Transcription service with:
- Audio preprocessing
- Whisper ASR with diarization support
- Confidence scoring
- Segment-level timestamps
"""
from __future__ import annotations

import logging
import math
import os
import tempfile
from typing import Any, Dict, List, Optional, Union

import torch
import whisper

from .audio_preprocessing import preprocess_audio
from .diarization import diarize_audio
from ..config import settings

logger = logging.getLogger(__name__)

_model = None
_model_device = None
_model_dtype = None


def get_model():
    """Lazy-load Whisper model once per process."""
    global _model, _model_device, _model_dtype

    if _model is not None:
        return _model

    device = "cuda" if settings.WHISPER_DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
    _model_device = device

    logger.info("[TRANSCRIPTION] Loading Whisper model %s on %s", settings.WHISPER_MODEL, device)
    _model = whisper.load_model(settings.WHISPER_MODEL, device=device)

    try:
        _model_dtype = _model.model.dtype if hasattr(_model, "model") else torch.float32
    except Exception:
        _model_dtype = torch.float32

    logger.info("[TRANSCRIPTION] Model loaded - device=%s, dtype=%s", device, _model_dtype)
    return _model


def _segment_confidence(seg: Dict[str, Any]) -> Optional[float]:
    """
    Whisper usually exposes avg_logprob, not token-level probabilities.
    Convert avg_logprob to a [0..1] heuristic confidence.
    """
    for key in ("confidence", "avg_logprob"):
        value = seg.get(key)
        if isinstance(value, (int, float)):
            if key == "confidence":
                return float(max(0.0, min(1.0, value)))
            # avg_logprob is usually negative; exp() gives a soft score
            return float(max(0.0, min(1.0, math.exp(value))))
    return None


def merge_speakers(transcript_segments: List[Dict], speaker_segments: List[Dict]) -> List[Dict]:
    """
    Merge Whisper transcript segments with diarization output.
    Simple overlap-based assignment.
    """
    if not speaker_segments:
        return transcript_segments

    merged = []
    for ts in transcript_segments:
        start = ts["start"]
        end = ts["end"]
        best_speaker = None
        best_overlap = 0.0
        mid = (start + end) / 2

        for sp in speaker_segments:
            s_start = sp.get("start", 0)
            s_end = sp.get("end", 0)
            overlap = max(0, min(end, s_end) - max(start, s_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = sp.get("speaker")

        if best_speaker is None and speaker_segments:
            min_dist = float("inf")
            for sp in speaker_segments:
                sp_mid = (sp.get("start", 0) + sp.get("end", 0)) / 2
                dist = abs(mid - sp_mid)
                if dist < min_dist:
                    min_dist = dist
                    best_speaker = sp.get("speaker")

        ts_copy = ts.copy()
        speaker_label = best_speaker if best_speaker is not None else "Unknown"
        if isinstance(speaker_label, (int, float)):
            ts_copy["speaker"] = f"Speaker {int(speaker_label)}"
        else:
            ts_copy["speaker"] = str(speaker_label)
        merged.append(ts_copy)

    return merged


def transcribe_from_bytes(audio_source: Union[bytes, str], filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Transcribe audio with preprocessing and optional diarization.
    Returns:
    - text
    - segments
    - language
    - confidence
    - has_diarization
    """
    temp_wav = None
    try:
        try:
            temp_wav = preprocess_audio(audio_source, filename)
        except Exception as e:
            logger.warning("Preprocessing failed: %s; saving raw audio to temp file", e)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            if isinstance(audio_source, bytes):
                tmp.write(audio_source)
            else:
                with open(audio_source, "rb") as src:
                    tmp.write(src.read())
            tmp.close()
            temp_wav = tmp.name

        model = get_model()
        use_fp16 = _model_device == "cuda" and _model_dtype == torch.float16

        transcribe_kwargs = {
            "audio": temp_wav,
            "task": "transcribe",
            "word_timestamps": True,
            "temperature": 0.0,
            "verbose": False,
        }
        # Newer whisper versions accept fp16; older ones ignore it.
        transcribe_kwargs["fp16"] = use_fp16

        result = model.transcribe(**transcribe_kwargs)

        segments: List[Dict[str, Any]] = []
        total_confidence = 0.0
        count = 0

        for seg in result.get("segments", []):
            segment_data: Dict[str, Any] = {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": str(seg["text"]).strip(),
            }
            conf = _segment_confidence(seg)
            if conf is not None:
                segment_data["confidence"] = conf
                total_confidence += conf
                count += 1
            segments.append(segment_data)

        output: Dict[str, Any] = {
            "text": str(result.get("text", "")).strip(),
            "language": result.get("language"),
            "segments": segments,
            "confidence": total_confidence / count if count > 0 else None,
            "has_diarization": False,
        }

        # IMPORTANT: diarization must happen before temp file cleanup
        if settings.DIARIZATION_ENABLED and temp_wav and os.path.exists(temp_wav):
            try:
                logger.info("[TRANSCRIPTION] Starting diarization...")
                speaker_segments = diarize_audio(temp_wav)
                if speaker_segments:
                    output["segments"] = merge_speakers(output["segments"], speaker_segments)
                    output["has_diarization"] = True
                    logger.info("[TRANSCRIPTION] Diarization complete: %d speaker segments", len(speaker_segments))
                else:
                    logger.warning("[TRANSCRIPTION] Diarization returned no segments")
            except Exception as e:
                logger.error("[TRANSCRIPTION] Diarization failed: %s", e, exc_info=True)

        logger.info(
            "[TRANSCRIPTION] Completed: %d segments, language=%s, confidence=%s",
            len(output["segments"]),
            output["language"],
            output["confidence"],
        )
        return output

    finally:
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.unlink(temp_wav)
            except OSError as e:
                logger.warning("Failed to delete temp file %s: %s", temp_wav, e)
