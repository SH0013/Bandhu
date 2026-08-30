"""IndicF5 Zero-Shot Voice Cloner for Bandhu Platform.

Ported from the proven grandma-voice-assistant implementation.
Uses AI4Bharat's IndicF5 (F5-TTS architecture) for zero-shot Telugu voice
cloning from a reference audio clip + transcript.

Requires GPU (CUDA) for real-time inference. Falls back gracefully when
GPU is unavailable, allowing the AdaptiveVoiceSynthesizer to use
Edge TTS + FAISS timbre conversion instead.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import numpy as np

from bandhu.audio.processor import AudioProcessor
from bandhu.config import settings

#: IndicF5 always emits 24 kHz audio.
INDICF5_SAMPLE_RATE = 24000
# Checkpoint inside the Hub repo
INDICF5_WEIGHTS_FILE = "model.safetensors"


def _ensure_torchaudio_soundfile_shim() -> None:
    """Shim torchaudio.load onto soundfile.

    Torchaudio 2.9+ routes torchaudio.load through torchcodec by default, which can fail
    on Windows with native DLL loading errors. This shim maps torchaudio.load onto
    soundfile to guarantee zero-error audio loading across all platforms.
    """
    try:
        import torchaudio
    except ImportError:
        return

    if getattr(torchaudio, "_soundfile_shim_active", False):
        return

    import soundfile as sf
    import torch

    orig_load = torchaudio.load

    def _shimmed_load(filepath: Any, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, int]:
        try:
            data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
            tensor = torch.from_numpy(data.T)
            return tensor, sr
        except Exception:
            try:
                return orig_load(filepath, *args, **kwargs)
            except Exception:
                from pydub import AudioSegment  # type: ignore[import-untyped]
                seg = AudioSegment.from_file(str(filepath))
                samples = np.array(seg.get_array_of_samples(), dtype=np.float32) / (
                    1 << (8 * seg.sample_width - 1)
                )
                if seg.channels > 1:
                    samples = samples.reshape((-1, seg.channels)).T
                else:
                    samples = samples.reshape((1, -1))
                return torch.from_numpy(samples), seg.frame_rate

    torchaudio.load = _shimmed_load
    torchaudio._soundfile_shim_active = True


def cuda_is_usable() -> bool:
    """Report whether CUDA can actually run kernels on this GPU.

    torch.cuda.is_available() alone is not enough: a torch build compiled for older
    architectures still reports True on a newer card, then fails at the first kernel
    launch. This checks the device's compute capability against the build's arch list.
    """
    try:
        import torch
    except ImportError:
        return False

    if not torch.cuda.is_available():
        return False

    try:
        major, minor = torch.cuda.get_device_capability(0)
        arch_list = torch.cuda.get_arch_list()
    except Exception:
        return False

    if not arch_list:
        return False

    target = f"{major}{minor}"
    return any(entry.rsplit("_", 1)[-1] == target for entry in arch_list)


class MissingReferenceTranscriptError(RuntimeError):
    """Raised when IndicF5 is asked to synthesize without a usable reference clip + transcript."""


class IndicF5VoiceCloner:
    """Zero-shot Telugu voice cloner backed by AI4Bharat's IndicF5.

    IndicF5 is an F5-TTS-architecture model loaded via transformers AutoModel with
    trust_remote_code=True from ai4bharat/IndicF5. Every call is zero-shot with three inputs:

    1. text -- the target Telugu text to speak,
    2. ref_audio_path -- a short, clean reference clip of the target speaker,
    3. ref_text -- the verbatim transcript of that reference clip.

    The model returns a 1-D waveform at 24 kHz, written as 16-bit PCM WAV.
    """

    SAMPLE_RATE: int = INDICF5_SAMPLE_RATE

    def __init__(
        self,
        repo_id: str = "ai4bharat/IndicF5",
        ref_audio: Path | None = None,
        ref_text: str | None = None,
        device: str | None = None,
        metadata_csv: Path | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self._device_override = device

        self.metadata_csv = metadata_csv or (settings.data_dir / "metadata.csv")

        self._model: Any = None
        self._load_lock = threading.Lock()

    @property
    def device(self) -> str:
        """Resolve the torch device to run inference on."""
        if self._device_override:
            return self._device_override
        return "cuda" if cuda_is_usable() else "cpu"

    def _load_model(self) -> Any:
        """Load and cache the IndicF5 model (thread-safe, once per process)."""
        _ensure_torchaudio_soundfile_shim()
        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:
                return self._model

            try:
                from transformers import AutoModel  # type: ignore[import-untyped]
            except ImportError as exc:
                raise RuntimeError(
                    "The IndicF5 backend requires 'transformers'. Install project "
                    "dependencies with: pip install -r requirements.txt"
                ) from exc

            try:
                model = AutoModel.from_pretrained(self.repo_id, trust_remote_code=True)
            except Exception as exc:
                # transformers >= 5 runs __init__ under a meta-device context.
                # IndicF5's remote model.py builds real tensors there, which collides.
                if "expected device meta" not in str(exc):
                    raise RuntimeError(self._load_failure_message(exc)) from exc
                try:
                    model = self._load_model_without_meta_init()
                except Exception as fallback_exc:
                    raise RuntimeError(self._load_failure_message(fallback_exc)) from fallback_exc

            # Disable aggressive silence chopping that drops quiet ending words
            if hasattr(model, "config") and hasattr(model.config, "remove_sil"):
                model.config.remove_sil = False

            self._model = model
            return model

    def _load_failure_message(self, exc: BaseException) -> str:
        """Build the actionable checklist shown when IndicF5 cannot be loaded."""
        return (
            f"Failed to load IndicF5 from '{self.repo_id}'.\n"
            "Checklist:\n"
            "  1. pip install --no-deps git+https://github.com/ai4bharat/IndicF5.git\n"
            "  2. torch + transformers installed\n"
            "  3. The Hub repo is gated -- accept the terms at "
            f"https://huggingface.co/{self.repo_id} and run 'hf auth login'\n"
            f"Underlying error: {exc}"
        )

    def _load_model_without_meta_init(self) -> Any:
        """Construct IndicF5 directly and load checkpoint, bypassing meta init."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoConfig  # type: ignore[import-untyped]
        from transformers.dynamic_module_utils import (  # type: ignore[import-untyped]
            get_class_from_dynamic_module,
        )

        config = AutoConfig.from_pretrained(self.repo_id, trust_remote_code=True)
        reference = getattr(config, "auto_map", {}).get("AutoModel")
        if not reference:
            raise RuntimeError(f"{self.repo_id} config has no auto_map['AutoModel'] entry")

        model_cls = get_class_from_dynamic_module(reference, self.repo_id)
        model = model_cls(config)

        state_dict = load_file(hf_hub_download(self.repo_id, INDICF5_WEIGHTS_FILE))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if len(state_dict) == len(unexpected):
            raise RuntimeError(
                f"No weights from {INDICF5_WEIGHTS_FILE} matched the IndicF5 module."
            )
        return model

    def _reference_from_metadata(self) -> tuple[Path, str] | None:
        """Pick the cleanest reference candidate that has a transcript from metadata.csv."""
        import csv

        try:
            if not self.metadata_csv.exists():
                return None
            with open(self.metadata_csv, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                best: tuple[Path, str] | None = None
                best_snr = -999.0
                for row in reader:
                    transcript = (row.get("transcript_te") or "").strip()
                    if not transcript:
                        continue
                    clip_file = row.get("clip_filename", "")
                    snr = float(row.get("snr_db", "0") or "0")
                    clip_path = settings.data_dir / "training_dataset" / clip_file
                    if not clip_path.exists():
                        clip_path = settings.data_dir / "reference_audio" / clip_file
                    if clip_path.exists() and snr > best_snr:
                        best = (clip_path, transcript)
                        best_snr = snr
                return best
        except Exception:
            return None

    def _resolve_reference(
        self,
        ref_audio: Path | None,
        ref_text: str | None,
    ) -> tuple[Path, str]:
        """Resolve the reference clip and its transcript for a synthesis call."""
        audio = ref_audio if ref_audio is not None else self.ref_audio
        text = ref_text if ref_text is not None else self.ref_text
        text = text.strip() if text else ""

        if audio is not None and text:
            audio_path = Path(audio)
            if not audio_path.exists():
                # Try relative to project root
                cand = settings.project_root / audio
                if cand.exists():
                    audio_path = cand
            if audio_path.exists():
                return audio_path, text

        # Check persona folders first before any legacy fallback
        personas_dir = settings.data_dir / "personas"
        if personas_dir.exists():
            for pdir in personas_dir.iterdir():
                if pdir.is_dir():
                    vprof_file = pdir / "voice_profile.json"
                    if vprof_file.exists():
                        try:
                            import json
                            vdata = json.loads(vprof_file.read_text(encoding="utf-8"))
                            t = (vdata.get("reference_transcript") or "").strip()
                            p = Path(vdata.get("reference_audio_path", ""))
                            if not p.is_absolute():
                                p = settings.project_root / p
                            if p.exists() and t:
                                audio = p
                                text = t
                                return audio, text
                        except Exception:
                            pass

        if audio is None or not text:
            from_metadata = self._reference_from_metadata()
            if from_metadata is not None:
                meta_audio, meta_text = from_metadata
                audio = audio if audio is not None else meta_audio
                text = text or meta_text

        if audio is None or not text:
            raise MissingReferenceTranscriptError(
                "IndicF5 needs a reference clip AND its exact transcript, but none was found."
            )

        audio = Path(audio)
        if not audio.exists():
            raise FileNotFoundError(f"Reference audio clip not found: {audio}")

        return audio, text

    @staticmethod
    def _to_mono_float32(raw: Any) -> np.ndarray:
        """Normalize the model's raw output into a 1-D float32 waveform in [-1, 1]."""
        if hasattr(raw, "detach"):  # torch.Tensor
            raw = raw.detach().cpu().numpy()

        audio = np.squeeze(np.asarray(raw))

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)

        if audio.ndim > 1:
            audio = audio.mean(axis=int(np.argmin(audio.shape)))

        if audio.size == 0:
            raise RuntimeError("IndicF5 returned an empty waveform.")

        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        peak = float(np.max(np.abs(audio)))
        if peak > 1.0:
            audio = audio / peak * 0.99

        return audio.astype(np.float32)

    @staticmethod
    def _enhance_clarity(audio: np.ndarray, sr: int) -> np.ndarray:
        """Restore pristine acoustic clarity and presence to IndicF5 output.

        Compensates for phone mic proximity effect, low-end boominess, and codec
        high-frequency attenuation through:
          1. 95 Hz Butterworth high-pass (removes sub-bass mud & rumble)
          2. 300 Hz boxiness cut (cleans chest resonance)
          3. 1.4 kHz - 4.8 kHz presence boost (enhances Telugu consonant clarity)
          4. Subtle upper-harmonic excitation (> 2.2 kHz)
          5. Transparent peak limiting with -1.0 dBFS headroom
        """
        from scipy import signal

        nyq = sr / 2.0

        # 1. 95 Hz high-pass filter to strip low-end mud / rumble
        sos_hp = signal.butter(2, min(95.0, nyq * 0.1) / nyq, btype='high', output='sos')
        wav_clean = signal.sosfilt(sos_hp, audio)

        # 2. Gentle dip at 300 Hz (boxiness / mud removal)
        b_mud, a_mud = signal.iirpeak(min(300.0, nyq * 0.3) / nyq, 1.2)
        mud_comp = signal.lfilter(b_mud, a_mud, wav_clean)
        wav_clean = wav_clean - 0.30 * mud_comp

        # 3. Speech presence band (1.4 kHz - 4.8 kHz) for consonant crispness
        sos_presence = signal.butter(2, [min(1400.0, nyq * 0.5) / nyq, min(4800.0, nyq * 0.85) / nyq], btype='band', output='sos')
        pres_comp = signal.sosfilt(sos_presence, wav_clean)

        # 4. Harmonic exciter for upper frequencies (> 2.2 kHz)
        sos_highs = signal.butter(2, min(2200.0, nyq * 0.6) / nyq, btype='high', output='sos')
        high_comp = signal.sosfilt(sos_highs, wav_clean)
        excited_highs = np.tanh(high_comp * 2.2) * 0.35

        # Combine: clean audio + presence boost + excited crispness
        audio_boosted = wav_clean + 0.65 * pres_comp + 0.75 * excited_highs

        # 5. Transparent peak normalization
        peak = float(np.max(np.abs(audio_boosted)))
        if peak > 1e-4:
            audio_boosted = audio_boosted / peak * 0.90

        return audio_boosted.astype(np.float32)

    def synthesize_sync(
        self,
        text: str,
        output_path: Path,
        ref_audio: Path | None = None,
        ref_text: str | None = None,
        speed: float | None = 0.80,
    ) -> Path:
        """Blocking IndicF5 synthesis with adjustable pacing."""
        if not text or not text.strip():
            raise ValueError("Cannot synthesize empty text.")

        ref_audio_path, resolved_ref_text = self._resolve_reference(ref_audio, ref_text)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model = self._load_model()
        if speed is not None and hasattr(model, "config"):
            model.config.speed = float(speed)

        try:
            import torch
            with torch.inference_mode():
                raw_audio = model(
                    text,
                    ref_audio_path=str(ref_audio_path),
                    ref_text=resolved_ref_text,
                )
        except Exception:
            raw_audio = model(
                text,
                ref_audio_path=str(ref_audio_path),
                ref_text=resolved_ref_text,
            )

        waveform = self._to_mono_float32(raw_audio)
        waveform = self._enhance_clarity(waveform, self.SAMPLE_RATE)
        AudioProcessor.save_wav(waveform, output_path, self.SAMPLE_RATE)
        return output_path

    async def synthesize(
        self,
        text: str,
        output_path: Path,
        ref_audio: Path | None = None,
        ref_text: str | None = None,
        speed: float | None = 0.80,
    ) -> Path:
        """Synthesize Telugu speech in the target speaker's voice with IndicF5.

        Inference is GPU-bound, so runs in a worker thread to keep the event loop responsive.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.synthesize_sync(text, output_path, ref_audio, ref_text, speed=speed),
        )
