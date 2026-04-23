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
from collections import defaultdict
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
    return text


def merge_speakers(transcript_segments: List[Dict], speaker_segments: List[Dict]) -> List[Dict]:
    """Merge Whisper segments with diarization output using overlap matching."""
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
        ts_copy["speaker"] = _normalize_speaker_label(best_speaker if best_speaker is not None else "Unknown")
        merged.append(ts_copy)

    return merged


def _segments_to_speaker_transcript(segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    last_speaker: Optional[str] = None
    last_line: Optional[str] = None

    for seg in segments:
        speaker = _normalize_speaker_label(seg.get("speaker") or "Unknown")
        text = str(seg.get("text") or "").strip()
        if not text:
            continue

        line = f"{speaker}: {text}"
        if speaker == last_speaker and lines:
            lines[-1] = f"{last_line} {text}" if last_line else line
            last_line = lines[-1]
        else:
            lines.append(line)
            last_speaker = speaker
            last_line = line
    return "\n".join(lines).strip()


def _infer_speaker_aliases(segments: List[Dict[str, Any]]) -> dict[str, str]:
    """Infer names from self-introductions in early diarized turns."""
    aliases: dict[str, str] = {}
    patterns = [
        r"(?:i'm|i am|my name is|this is)\s+([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+)?)",
        r"(?:i am)\s+([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+)?)",
    ]

    for seg in segments[:40]:  # introductions typically happen early
        speaker = _normalize_speaker_label(seg.get("speaker") or "Unknown")
        text = str(seg.get("text") or "")
        if speaker in aliases:
            continue
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                if len(candidate.split()) <= 3:
                    aliases[speaker] = candidate
                    break

    return aliases


def transcribe_from_bytes(audio_source: Union[bytes, str], filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Transcribe audio with preprocessing and optional diarization.
    Returns text, segments, speaker_transcript, aliases, language, confidence and diarization flag.
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
            "fp16": use_fp16,
        }
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
            "speaker_transcript": None,
            "speaker_aliases": {},
        }

        # IMPORTANT: diarization must happen before temp file cleanup
        if settings.DIARIZATION_ENABLED and temp_wav and os.path.exists(temp_wav):
            try:
                logger.info("[TRANSCRIPTION] Starting diarization...")
                speaker_segments = diarize_audio(temp_wav)
                if speaker_segments:
                    output["segments"] = merge_speakers(output["segments"], speaker_segments)
                    output["has_diarization"] = True
                    output["speaker_transcript"] = _segments_to_speaker_transcript(output["segments"])
                    output["speaker_aliases"] = _infer_speaker_aliases(output["segments"])
                    logger.info("[TRANSCRIPTION] Diarization complete: %d speaker segments", len(speaker_segments))
                else:
                    logger.warning("[TRANSCRIPTION] Diarization returned no segments")
            except Exception as e:
                logger.error("[TRANSCRIPTION] Diarization failed: %s", e, exc_info=True)

        if not output["speaker_transcript"]:
            if output["has_diarization"]:
                output["speaker_transcript"] = _segments_to_speaker_transcript(output["segments"]) or output["text"]
            else:
                output["speaker_transcript"] = output["text"]

        logger.info(
            "[TRANSCRIPTION] Completed: %d segments, language=%s, confidence=%s",
            len(output["segments"]),
            output["language"],
            output["confidence"],
        )
                # Infer speaker aliases only from explicit self-introductions.
        speaker_aliases = _infer_speaker_aliases(merged_segments)
        speaker_transcript = _segments_to_speaker_transcript(merged_segments)

        confidence_values = [
            c for c in (_segment_confidence(seg) for seg in merged_segments) if c is not None
        ]
        confidence = float(sum(confidence_values) / len(confidence_values)) if confidence_values else None

        return {
            "text": final_text.strip(),
            "language": result.get("language") or "en",
            "segments": merged_segments,
            "speaker_transcript": speaker_transcript,
            "speaker_aliases": speaker_aliases,
            "confidence": confidence,
            "has_diarization": bool(diarization_segments),
        }
        return output

    finally:
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.unlink(temp_wav)
            except OSError as e:
                logger.warning("Failed to delete temp file %s: %s", temp_wav, e)
