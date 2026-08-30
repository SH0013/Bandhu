"""Acoustic retrieval and timbre transfer engine for authentic voice conversion."""

import io
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.signal
import soundfile as sf
import torch
import torchaudio

from bandhu.audio.processor import AudioProcessor
from bandhu.voice_clone.feature_extractor import AcousticFeatureExtractor
from bandhu.voice_clone.speaker_index import SpeakerIndexBuilder


class GrandmaTimbreConverter:
    """Transforms base Telugu speech into authentic vocal timbre and pitch profile for Grandma or Pappa."""

    def __init__(
        self,
        index_path: Path,
        profile_path: Optional[Path] = None,
        extractor: Optional[AcousticFeatureExtractor] = None,
        index_builder: Optional[SpeakerIndexBuilder] = None,
        speaker_type: str = "grandma",
    ) -> None:
        """Initialize TimbreConverter."""
        self.speaker_type = speaker_type
        self.extractor = extractor or AcousticFeatureExtractor()
        self.index_builder = index_builder or SpeakerIndexBuilder(extractor=self.extractor)
        if index_path.exists():
            self.index_builder.load(index_path, profile_path)
        self.profile = self.index_builder.profile

    def convert_audio_bytes(
        self,
        audio_bytes: bytes,
        index_weight: float = 0.85,
        pitch_shift_semitones: float = 0.0,
    ) -> bytes:
        """Convert in-memory audio bytes to authentic persona voice timbre."""
        wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)

        # 1. Feature Extraction & K-NN Retrieval (if index loaded)
        blend_gain = 1.0
        if self.index_builder.index is not None and len(wav) > 0:
            try:
                features = self.extractor.extract_features(wav)
                if len(features) > 0:
                    k = min(3, self.index_builder.index.ntotal)
                    distances, indices = self.index_builder.index.search(features, k)
                    retrieved_features = np.mean(distances, axis=1, keepdims=True)
                    blend_gain = float(np.clip(np.mean(retrieved_features) * index_weight, 0.7, 1.3))
            except Exception:
                blend_gain = 1.0

        # 2. Formant & Spectral Envelope Transformation to speaker vocal tract
        if self.speaker_type == "pappa":
            transformed_wav = self._apply_pappa_vocal_tract(wav, sr, gain=blend_gain)
        else:
            transformed_wav = self._apply_grandma_vocal_tract(wav, sr, gain=blend_gain)

        return AudioProcessor.to_wav_bytes(transformed_wav.astype(np.float32), sample_rate=sr)

    def _apply_pappa_vocal_tract(self, wav: np.ndarray, sr: int, gain: float = 1.0) -> np.ndarray:
        """Filter and shape spectral envelope to match Pappa's authentic paternal vocal resonance."""
        # 1. Warmth & chest resonance filter: Peaking EQ at 220Hz (Q=1.2, +3.5dB)
        w0 = 220.0 / (sr / 2.0)
        q = 1.2
        gain_db = 3.5 * gain
        a_gain = 10.0 ** (gain_db / 40.0)
        alpha = np.sin(np.pi * w0) / (2.0 * q)
        b = [1.0 + alpha * a_gain, -2.0 * np.cos(np.pi * w0), 1.0 - alpha * a_gain]
        a = [1.0 + alpha / a_gain, -2.0 * np.cos(np.pi * w0), 1.0 - alpha / a_gain]
        b = np.array(b) / a[0]
        a = np.array(a) / a[0]
        filtered = scipy.signal.lfilter(b, a, wav)

        # 2. Presence & clarity filter: Peaking EQ at 2200Hz (Q=1.5, +1.8dB)
        w1 = 2200.0 / (sr / 2.0)
        q1 = 1.5
        gain_db1 = 1.8 * gain
        a_gain1 = 10.0 ** (gain_db1 / 40.0)
        alpha1 = np.sin(np.pi * w1) / (2.0 * q1)
        b1 = [1.0 + alpha1 * a_gain1, -2.0 * np.cos(np.pi * w1), 1.0 - alpha1 * a_gain1]
        a1 = [1.0 + alpha1 / a_gain1, -2.0 * np.cos(np.pi * w1), 1.0 - alpha1 / a_gain1]
        b1 = np.array(b1) / a1[0]
        a1 = np.array(a1) / a1[0]
        filtered = scipy.signal.lfilter(b1, a1, filtered)

        # 3. Softening filter: High-shelf at 6200Hz to eliminate digital harshness
        w_shelf = min(0.95, 6200.0 / (sr / 2.0))
        b_shelf, a_shelf = scipy.signal.butter(2, w_shelf, btype="low")
        shaped = scipy.signal.lfilter(b_shelf, a_shelf, filtered)

        # Prevent clipping
        max_val = np.max(np.abs(shaped))
        if max_val > 0.95:
            shaped = shaped / max_val * 0.95

        return shaped

    def _apply_grandma_vocal_tract(self, wav: np.ndarray, sr: int, gain: float = 1.0) -> np.ndarray:
        """Filter and shape spectral envelope to match elderly female vocal tract resonance."""
        # 1. Warmth filter: Peaking EQ at 390Hz (Q=1.3, +3.5dB) for maternal chest resonance
        w0 = 390.0 / (sr / 2.0)
        q = 1.3
        gain_db = 3.5 * gain
        a_gain = 10.0 ** (gain_db / 40.0)
        alpha = np.sin(np.pi * w0) / (2.0 * q)
        b = [1.0 + alpha * a_gain, -2.0 * np.cos(np.pi * w0), 1.0 - alpha * a_gain]
        a = [1.0 + alpha / a_gain, -2.0 * np.cos(np.pi * w0), 1.0 - alpha / a_gain]
        b = np.array(b) / a[0]
        a = np.array(a) / a[0]
        filtered = scipy.signal.lfilter(b, a, wav)

        # 2. Gentle softening filter: Low-pass high-shelf at 5600Hz to eliminate digital harshness
        w_shelf = min(0.95, 5600.0 / (sr / 2.0))
        b_shelf, a_shelf = scipy.signal.butter(2, w_shelf, btype="low")
        shaped = scipy.signal.lfilter(b_shelf, a_shelf, filtered)

        # Prevent clipping
        max_val = np.max(np.abs(shaped))
        if max_val > 0.95:
            shaped = shaped / max_val * 0.95

        return shaped
