"""
Audio preprocessing for better transcription quality.
- Convert to 16kHz mono (optimal for Whisper)
- Volume normalization (RMS)
- Optional noise reduction
- Format conversion via pydub + ffmpeg
"""
import io
import tempfile
import os
from typing import Optional

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

import numpy as np
import soundfile as sf

from ..config import settings


def load_audio_bytes(audio_bytes: bytes, original_format: Optional[str] = None) -> np.ndarray:
    """
    Load audio from bytes and convert to numpy array (mono, target sample rate).
    Returns float32 normalized array in range [-1, 1].
    """
    if not PYDUB_AVAILABLE:
        raise ImportError("pydub is required for audio preprocessing. Install with: pip install pydub")

    # Load with pydub (auto-detects format)
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=original_format)

    # Convert to mono and target sample rate
    audio = audio.set_channels(1)
    if audio.frame_rate != settings.AUDIO_TARGET_SR:
        audio = audio.set_frame_rate(settings.AUDIO_TARGET_SR)

    # Normalize volume if configured
    if settings.AUDIO_NORMALIZE:
        audio = audio.normalize()

    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples = samples / (2**15) if audio.sample_width == 2 else samples / (2**31)

    return samples, settings.AUDIO_TARGET_SR


def reduce_noise(samples: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Simple noise reduction using spectral gating.
    Requires noisereduce library (optional: pip install noisereduce).
    """
    try:
        import noisereduce as nr
        # Estimate noise profile from first 0.5 seconds (assume silence/speech)
        noise_sample = samples[:int(sr * 0.5)]
        reduced = nr.reduce_noise(
            y=samples,
            sr=sr,
            prop_decrease=0.8,
            stationary=True,
            n_std_thresh_stationary=1.5,
            n_mfcc=13,
            n_hop=512,
            n_fft=2048,
            noise_clip=noise_sample
        )
        return reduced
    except ImportError:
        # Noise reduction not installed, return original
        return samples


def save_wave(samples: np.ndarray, sr: int = 16000, path: Optional[str] = None) -> str:
    """Save numpy array as WAV file. Returns path."""
    if path is None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        path = tmp.name
        tmp.close()

    sf.write(path, samples, sr, subtype='PCM_16')
    return path


def preprocess_audio(audio_bytes: bytes, original_filename: Optional[str] = None) -> str:
    """
    Full preprocessing pipeline. Converts raw audio to clean 16kHz mono WAV.
    Returns path to temporary WAV file (caller must delete).
    """
    # Guess format from filename extension
    ext = None
    if original_filename:
        ext = original_filename.split('.')[-1].lower() if '.' in original_filename else None

    # Load and convert
    samples, sr = load_audio_bytes(audio_bytes, original_format=ext)

    # Optional noise reduction
    if settings.AUDIO_REDUCE_NOISE:
        samples = reduce_noise(samples, sr)

    # Save to temp WAV
    path = save_wave(samples, sr)
    return path