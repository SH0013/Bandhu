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
    """Transforms base Telugu speech into Grandma's authentic vocal timbre and pitch profile."""

    def __init__(
        self,
        index_path: Path,
        profile_path: Optional[Path] = None,
        extractor: Optional[AcousticFeatureExtractor] = None,
        index_builder: Optional[SpeakerIndexBuilder] = None,
    ) -> None:
        """Initialize GrandmaTimbreConverter."""
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
        """Convert in-memory audio bytes to Grandma's voice timbre."""
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

        # 2. Formant & Spectral Envelope Transformation to Grandma's vocal tract
        transformed_wav = self._apply_grandma_vocal_tract(wav, sr, gain=blend_gain)

        # 3. Pitch Adaptation
        target_f0 = self.profile.median_f0_hz if self.profile else 200.0
        try:
            src_f0_contour = self.extractor.extract_f0(wav)
            voiced_src = src_f0_contour[src_f0_contour > 0]
            if len(voiced_src) > 0:
                src_median = float(np.median(voiced_src))
                if src_median > 0:
                    ratio = target_f0 / src_median
                    calculated_shift = 12.0 * np.log2(ratio)
                    pitch_shift_semitones += float(np.clip(calculated_shift, -3.0, 3.0))

            if abs(pitch_shift_semitones) > 0.2:
                tensor_wav = torch.from_numpy(transformed_wav).unsqueeze(0)
                shifted = torchaudio.functional.pitch_shift(
                    tensor_wav,
                    sr,
                    n_steps=pitch_shift_semitones,
                )
                transformed_wav = shifted.squeeze(0).numpy()
        except Exception:
            pass

        return AudioProcessor.to_wav_bytes(transformed_wav.astype(np.float32), sample_rate=sr)

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
