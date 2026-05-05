"""Transcription service with audio preprocessing, Whisper ASR and diarization.

Enhancements over the baseline:
- normalized per-segment confidence
- diarization labels kept in segments
- a speaker-labeled transcript is generated for downstream task extraction
- lightweight speaker alias inference from self-introductions
"""
from __future__ import annotations

import logging
import math
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Union

import torch
import whisper

from .audio_preprocessing import preprocess_audio
from .diarization import diarize_audio
from ..config import settings

logger = logging.getLogger(__name__)

_model = None
_model_device: Optional[str] = None
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
    """Convert Whisper's avg_logprob to a soft [0..1] score."""
    for key in ("confidence", "avg_logprob"):
        value = seg.get(key)
        if isinstance(value, (int, float)):
            if key == "confidence":
                return float(max(0.0, min(1.0, value)))
            return float(max(0.0, min(1.0, math.exp(value))))
    return None


def _normalize_speaker_label(label: Any) -> str:
    text = str(label or "Unknown").strip()
    m = re.search(r"(\d+)", text)
    if text.upper().startswith("SPEAKER") and m:
        return f"SPEAKER_{int(m.group(1)):02d}"
    if text.lower().startswith("speaker") and m:
        return f"SPEAKER_{int(m.group(1)):02d}"
    return text or "Unknown"


def merge_speakers(transcript_segments: List[Dict[str, Any]], speaker_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge Whisper segments with diarization output using overlap matching."""
    if not speaker_segments:
        return transcript_segments

    merged: List[Dict[str, Any]] = []
    for ts in transcript_segments:
        start = float(ts["start"])
        end = float(ts["end"])
        best_speaker = None
        best_overlap = 0.0
        mid = (start + end) / 2

        for sp in speaker_segments:
            s_start = float(sp.get("start", 0.0))
            s_end = float(sp.get("end", 0.0))
            overlap = max(0.0, min(end, s_end) - max(start, s_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = sp.get("speaker")

        if best_speaker is None and speaker_segments:
            min_dist = float("inf")
            for sp in speaker_segments:
                sp_mid = (float(sp.get("start", 0.0)) + float(sp.get("end", 0.0))) / 2
                dist = abs(mid - sp_mid)
                if dist < min_dist:
                    min_dist = dist
                    best_speaker = sp.get("speaker")

        ts_copy = dict(ts)
        ts_copy["speaker"] = _normalize_speaker_label(best_speaker if best_speaker is not None else "Unknown")
        merged.append(ts_copy)

    return merged


def _looks_like_person_name(candidate: str) -> bool:
    candidate = (candidate or "").strip()
    if not candidate:
        return False

    # Only accept explicit-looking names, not ordinary phrases.
    if not re.fullmatch(r"[A-Z][a-z'\-]+(?:\s+[A-Z][a-z'\-]+){0,2}", candidate):
        return False

    bad_tokens = {
        "getting", "going", "doing", "waking", "waking up", "shadow", "everything",
        "problem", "meeting", "agenda", "project", "team", "design", "function",
        "control", "system", "pain", "hurt", "this", "that", "there", "here",
    }
    low = candidate.lower()
    return not any(tok in low for tok in bad_tokens)


def _infer_speaker_aliases(segments: List[Dict[str, Any]]) -> dict[str, str]:
    """Infer names from self-introductions in early diarized turns."""
    aliases: dict[str, str] = {}

    patterns = [
        r"(?:i'm|i am|my name is|this is|name's)\s+([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+)?)",
        r"(?:i am)\s+([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+)?)",
    ]

    for seg in segments[:40]:
        speaker = _normalize_speaker_label(seg.get("speaker") or "Unknown")
        text = str(seg.get("text") or "")
        if speaker in aliases:
            continue

        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if not m:
                continue
            candidate = m.group(1).strip()

            # Reject phrases like "I am getting out"
            if _looks_like_person_name(candidate) and len(candidate.split()) <= 3:
                aliases[speaker] = candidate
                break

    return aliases


def _apply_speaker_aliases(segments: List[Dict[str, Any]], alias_map: dict[str, str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for seg in segments:
        speaker = _normalize_speaker_label(seg.get("speaker") or "Unknown")
        alias = alias_map.get(speaker)

        item = dict(seg)
        item["speaker_label"] = speaker
        item["speaker_name"] = alias
        item["speaker_display"] = alias or speaker
        out.append(item)
    return out


def _segments_to_speaker_transcript(segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    last_speaker: Optional[str] = None

    for seg in segments:
        speaker = str(seg.get("speaker_display") or seg.get("speaker_name") or seg.get("speaker") or "Unknown").strip()
        text = str(seg.get("text") or "").strip()
        if not text:
            continue

        line = f"{speaker}: {text}"
        if speaker == last_speaker and lines:
            lines[-1] = f"{lines[-1]} {text}"
        else:
            lines.append(line)
            last_speaker = speaker

    return "\n".join(lines).strip()


def transcribe_from_bytes(audio_source: Union[bytes, str], filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Transcribe audio with preprocessing and optional diarization.
    """
    temp_wav: Optional[str] = None

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

        result = model.transcribe(
            audio=temp_wav,
            task="transcribe",
            word_timestamps=True,
            temperature=0.0,
            verbose=False,
            fp16=use_fp16,
        )

        whisper_segments: List[Dict[str, Any]] = []
        confidences: List[float] = []

        for seg in result.get("segments", []):
            segment_data: Dict[str, Any] = {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": str(seg["text"]).strip(),
            }
            conf = _segment_confidence(seg)
            if conf is not None:
                segment_data["confidence"] = conf
                confidences.append(conf)
            whisper_segments.append(segment_data)

        merged_segments = whisper_segments
        has_diarization = False

        if settings.DIARIZATION_ENABLED and temp_wav and os.path.exists(temp_wav):
            try:
                logger.info("[TRANSCRIPTION] Starting diarization...")
                speaker_segments = diarize_audio(temp_wav)
                if speaker_segments:
                    merged_segments = merge_speakers(whisper_segments, speaker_segments)
                    has_diarization = True
                    logger.info("[TRANSCRIPTION] Diarization complete: %d speaker segments", len(speaker_segments))
                else:
                    logger.warning("[TRANSCRIPTION] Diarization returned no segments")
            except Exception as e:
                logger.error("[TRANSCRIPTION] Diarization failed: %s", e, exc_info=True)

        speaker_aliases = _infer_speaker_aliases(merged_segments)
        enriched_segments = _apply_speaker_aliases(merged_segments, speaker_aliases)
        speaker_transcript = _segments_to_speaker_transcript(enriched_segments) or str(result.get("text", "")).strip()

        confidence = float(sum(confidences) / len(confidences)) if confidences else None

        output: Dict[str, Any] = {
            "text": str(result.get("text", "")).strip(),
            "language": result.get("language") or "en",
            "segments": enriched_segments,
            "speaker_transcript": speaker_transcript,
            "speaker_aliases": speaker_aliases,
            "confidence": confidence,
            "has_diarization": has_diarization,
        }

        logger.info(
            "[TRANSCRIPTION] Completed: %d segments, language=%s, confidence=%s, diarization=%s",
            len(enriched_segments),
            output["language"],
            output["confidence"],
            output["has_diarization"],
        )

        return output

    finally:
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.unlink(temp_wav)
            except OSError as e:
                logger.warning("Failed to delete temp file %s: %s", temp_wav, e)