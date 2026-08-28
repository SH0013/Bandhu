"""Audio processing and normalization utilities."""

from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal


class AudioProcessor:
    """Audio normalization, resampling, and waveform management."""

    @staticmethod
    def load_audio(path_or_bytes: str | Path | bytes, target_sr: int = 24000) -> tuple[np.ndarray, int]:
        """Load audio file or byte buffer and convert to mono float32 at target sample rate."""
        if isinstance(path_or_bytes, (str, Path)):
            audio, sr = sf.read(str(path_or_bytes), dtype="float32")
        else:
            audio, sr = sf.read(io.BytesIO(path_or_bytes), dtype="float32")

        # Convert stereo to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != target_sr:
            num_samples = int(len(audio) * target_sr / sr)
            audio = signal.resample(audio, num_samples).astype(np.float32)
            sr = target_sr

        # Peak normalization
        max_val = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
        if max_val > 1e-4:
            audio = audio / max_val * 0.95

        return audio.astype(np.float32), sr

    @staticmethod
    def save_wav(audio: np.ndarray, file_path: str | Path, sample_rate: int = 24000) -> Path:
        """Save float32 audio array to 16-bit PCM WAV."""
        out_path = Path(file_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio, sample_rate, subtype="PCM_16")
        return out_path

    @staticmethod
    def to_wav_bytes(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
        """Convert float32 audio array to in-memory WAV bytes."""
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()
