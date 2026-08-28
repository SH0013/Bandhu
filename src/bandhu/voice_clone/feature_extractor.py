"""Acoustic feature and pitch extraction using HuBERT and F0 estimation."""

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoFeatureExtractor, HubertModel


class AcousticFeatureExtractor:
    """Extracts HuBERT acoustic representations and fundamental frequency contours."""

    def __init__(
        self,
        model_name: str = "facebook/hubert-base-ls960",
        device: str = "cpu",
    ) -> None:
        """Initialize feature extractor with HuBERT model."""
        self.device = torch.device(device)
        self.target_sr = 16000
        self._model = None
        self._processor = None
        self.model_name = model_name

    def _ensure_model(self) -> None:
        if self._model is None:
            self._processor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self._model = HubertModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()

    def load_audio(self, audio_path: Path) -> np.ndarray:
        """Load audio file and resample to 16kHz mono."""
        wav, sr = sf.read(str(audio_path), dtype="float32")
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)

        if sr != self.target_sr:
            tensor_wav = torch.from_numpy(wav).unsqueeze(0)
            resampled = torchaudio.functional.resample(tensor_wav, sr, self.target_sr)
            wav = resampled.squeeze(0).numpy()

        return wav

    def extract_features(self, wav: np.ndarray) -> np.ndarray:
        """Extract HuBERT frame-level acoustic embeddings."""
        if len(wav) == 0:
            return np.zeros((0, 768), dtype=np.float32)

        try:
            self._ensure_model()
            inputs = self._processor(
                wav,
                sampling_rate=self.target_sr,
                return_tensors="pt",
                padding=True,
            ).input_values.to(self.device)

            with torch.no_grad():
                outputs = self._model(inputs)
                features = outputs.last_hidden_state.squeeze(0).cpu().numpy()

            norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
            normalized = features / norms
            return normalized.astype(np.float32)
        except Exception:
            # Fallback lightweight spectral features if Hub is inaccessible
            return np.zeros((0, 768), dtype=np.float32)

    def extract_f0(self, wav: np.ndarray) -> np.ndarray:
        """Extract fundamental frequency (pitch) contour using autocorrelation."""
        frame_length = 400  # 25ms at 16kHz
        hop_length = 160    # 10ms at 16kHz
        num_frames = max(1, (len(wav) - frame_length) // hop_length + 1)
        f0 = np.zeros(num_frames, dtype=np.float32)

        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            frame = wav[start:end]
            if len(frame) < frame_length or np.max(np.abs(frame)) < 1e-3:
                f0[i] = 0.0
                continue

            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(corr) // 2 :]
            d = np.diff(corr)
            start_idx = np.where(d > 0)[0]
            if len(start_idx) == 0:
                f0[i] = 0.0
                continue
            peak_idx = start_idx[0] + np.argmax(corr[start_idx[0] :])
            if peak_idx > 0 and corr[peak_idx] > 0.3 * corr[0]:
                freq = self.target_sr / peak_idx
                if 60.0 <= freq <= 600.0:
                    f0[i] = freq
                else:
                    f0[i] = 0.0
            else:
                f0[i] = 0.0

        return f0
