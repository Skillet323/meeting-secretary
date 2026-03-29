"""
Audio preprocessing for better transcription quality.
- Convert to 16kHz mono (optimal for Whisper)
- Volume normalization (RMS)
- Optional noise reduction
- Format conversion via pydub + ffmpeg
"""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

import numpy as np
import soundfile as sf

from ..config import settings

AudioSource = Union[bytes, str, os.PathLike]


def _load_audio_segment(audio_source: AudioSource, original_format: Optional[str] = None):
    if not PYDUB_AVAILABLE:
        raise ImportError("pydub is required for audio preprocessing. Install with: pip install pydub")

    if isinstance(audio_source, (str, os.PathLike)):
        path = str(audio_source)
        return AudioSegment.from_file(path, format=original_format)

    return AudioSegment.from_file(io.BytesIO(audio_source), format=original_format)


def load_audio_bytes(audio_bytes: bytes, original_format: Optional[str] = None) -> Tuple[np.ndarray, int]:
    """
    Backwards-compatible wrapper that loads audio from bytes and converts to
    numpy array (mono, target sample rate).
    Returns float32 normalized array in range [-1, 1].
    """
    return load_audio_source(audio_bytes, original_format=original_format)


def load_audio_source(audio_source: AudioSource, original_format: Optional[str] = None) -> Tuple[np.ndarray, int]:
    """
    Load audio from bytes or a file path and convert to numpy array.
    Returns float32 normalized array in range [-1, 1].
    """
    audio = _load_audio_segment(audio_source, original_format=original_format)

    # Convert to mono and target sample rate
    audio = audio.set_channels(1)
    if audio.frame_rate != settings.AUDIO_TARGET_SR:
        audio = audio.set_frame_rate(settings.AUDIO_TARGET_SR)

    # Normalize volume if configured
    if settings.AUDIO_NORMALIZE:
        audio = audio.normalize()

    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.sample_width == 2:
        samples = samples / (2**15)
    elif audio.sample_width == 4:
        samples = samples / (2**31)
    else:
        scale = float(2 ** (8 * audio.sample_width - 1))
        samples = samples / scale if scale else samples

    return samples, settings.AUDIO_TARGET_SR


def reduce_noise(samples: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Simple noise reduction using spectral gating.
    Requires noisereduce library (optional: pip install noisereduce).
    """
    try:
        import noisereduce as nr
        noise_sample = samples[: int(sr * 0.5)]
        reduced = nr.reduce_noise(
            y=samples,
            sr=sr,
            prop_decrease=0.8,
            stationary=True,
            n_std_thresh_stationary=1.5,
            n_mfcc=13,
            n_hop=512,
            n_fft=2048,
            noise_clip=noise_sample,
        )
        return reduced
    except Exception:
        return samples


def save_wave(samples: np.ndarray, sr: int = 16000, path: Optional[str] = None) -> str:
    """Save numpy array as WAV file. Returns path."""
    if path is None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        path = tmp.name
        tmp.close()

    sf.write(path, samples, sr, subtype="PCM_16")
    return path


def preprocess_audio(audio_source: AudioSource, original_filename: Optional[str] = None) -> str:
    """
    Full preprocessing pipeline. Converts raw audio to clean 16kHz mono WAV.
    Returns path to temporary WAV file (caller must delete).
    """
    ext = None
    if original_filename and "." in original_filename:
        ext = original_filename.rsplit(".", 1)[-1].lower()
    elif isinstance(audio_source, (str, os.PathLike)):
        ext = Path(audio_source).suffix.lstrip(".").lower() or None

    samples, sr = load_audio_source(audio_source, original_format=ext)

    if settings.AUDIO_REDUCE_NOISE:
        samples = reduce_noise(samples, sr)

    return save_wave(samples, sr)
