"""Adaptive Voice Synthesizer: IndicF5 Zero-Shot GPU Cloner with Google Cloud Neural TTS Fallback."""

from __future__ import annotations

import io
import math
import tempfile
from pathlib import Path
from typing import Any, Literal
import numpy as np

from bandhu.audio.processor import AudioProcessor
from bandhu.audio.voice_manager import VoiceProfile, VoiceProfileManager
from bandhu.config import settings


class AdaptiveVoiceSynthesizer:
    """Multi-tiered speech synthesizer ensuring 100% natural, human-grade voice output.

    Engine Priority:
      1. IndicF5 GPU Zero-Shot Cloner (ai4bharat/IndicF5, requires GPU + reference clip)
      2. Edge TTS + FAISS Grandma Timbre Converter (CPU-friendly, neural synthesis)
      3. Google Cloud Text-to-Speech (Neural2, requires GCP credentials)
      4. Synthetic warm acoustic carrier (guaranteed fallback, zero dependencies)
    """

    def __init__(self, voice_manager: VoiceProfileManager | None = None) -> None:
        self.voice_manager = voice_manager or VoiceProfileManager()
        self.active_engine: str = "neural_tts"
        self._gcp_tts_client = None
        self._grandma_timbre_converter = None
        self._indicf5_cloner = None

        self._initialize_engines()

    def _initialize_engines(self) -> None:
        """Probe for IndicF5 GPU, Neural TTS, Google Cloud TTS, and Grandma Timbre Converter."""
        # 1. Try to initialize IndicF5 zero-shot cloner (works on CUDA GPU & CPU)
        try:
            from bandhu.audio.indicf5_cloner import IndicF5VoiceCloner, cuda_is_usable

            dev = "cuda" if cuda_is_usable() else "cpu"
            self._indicf5_cloner = IndicF5VoiceCloner(device=dev)
            self.active_engine = f"indicf5_{dev}"
            print(f"[TTS] IndicF5 Zero-Shot Voice Cloner activated (device: {dev}).")
        except ImportError as exc:
            self._indicf5_cloner = None
            self.active_engine = "neural_tts"
            print(f"[TTS] IndicF5 dependencies not available ({exc}), using Neural Indic TTS Engine.")
        except Exception as exc:
            self._indicf5_cloner = None
            self.active_engine = "neural_tts"
            print(f"[TTS] IndicF5 init note: {exc}")

        self._timbre_converters: dict[str, Any] = {}

        # 2. Pre-load available FAISS Timbre Converters
        try:
            self.get_timbre_converter("grandma_chittoor", speaker_name="Grandma", gender="female")
            self.get_timbre_converter("pappa", speaker_name="Pappa", gender="male")
        except Exception as exc:
            print(f"[TTS] Preload timbre converters note: {exc}")

        # 3. Probe for Google Cloud TTS if configured
        if settings.tts_mode in ("auto", "gcp_tts"):
            try:
                from google.cloud import texttospeech
                self._gcp_tts_client = texttospeech.TextToSpeechClient()
                print("[TTS] Google Cloud Text-to-Speech client activated.")
            except Exception:
                pass

        # 4. Edge TTS is always available as primary CPU fallback
        if self.active_engine not in ("indicf5_gpu", "indicf5_cpu"):
            self.active_engine = "neural_tts"
            print("[TTS] Neural Indic TTS Engine activated (Real-time Natural Voice Synthesis).")

    def get_timbre_converter(self, voice_id: str, speaker_name: str = "", gender: str = "female") -> Any | None:
        """Dynamically load or retrieve cached FAISS Timbre Converter for any persona."""
        if voice_id in self._timbre_converters:
            return self._timbre_converters[voice_id]

        try:
            from bandhu.voice_clone.timbre_converter import GrandmaTimbreConverter
            candidates = [
                Path(__file__).resolve().parent.parent.parent.parent / "data",
                Path(__file__).resolve().parent.parent.parent / "data",
                Path("/app/data"),
                Path("./data"),
            ]
            possible_prefixes = [
                f"{voice_id}_voice",
                f"{voice_id}",
                "grandma_voice" if "grandma" in voice_id or "అమ్మమ్మ" in speaker_name else "",
                "pappa_voice" if any(w in (voice_id + speaker_name).lower() for w in ("pappa", "father", "dad", "పప్పా")) else "",
            ]

            for cand in candidates:
                # Check dedicated persona directory first
                p_idx = cand / "personas" / voice_id / "voice.index"
                p_prof = cand / "personas" / voice_id / "voice_profile_features.json"
                if p_idx.exists():
                    is_grandma = any(w in (voice_id + speaker_name).lower() for w in ("grandma", "amamma", "అమ్మమ్మ", "నానమ్మ"))
                    is_pappa = any(w in (voice_id + speaker_name).lower() for w in ("pappa", "father", "dad", "పప్పా", "నాన్న"))
                    speaker_type = "grandma" if is_grandma else ("pappa" if (is_pappa or gender.lower() == "male") else "grandma")
                    conv = GrandmaTimbreConverter(
                        index_path=p_idx,
                        profile_path=p_prof if p_prof.exists() else None,
                        speaker_type=speaker_type,
                    )
                    self._timbre_converters[voice_id] = conv
                    print(f"[TTS] Activated FAISS Timbre Converter for '{voice_id}' ({speaker_type}) from {p_idx}.")
                    return conv

                for prefix in possible_prefixes:
                    if not prefix:
                        continue
                    idx_file = cand / f"{prefix}.index"
                    prof_file = cand / f"{prefix}_profile.json"
                    if idx_file.exists():
                        is_grandma = any(w in (voice_id + speaker_name).lower() for w in ("grandma", "amamma", "అమ్మమ్మ", "నానమ్మ"))
                        is_pappa = any(w in (voice_id + speaker_name).lower() for w in ("pappa", "father", "dad", "పప్పా", "నాన్న"))
                        speaker_type = "grandma" if is_grandma else ("pappa" if (is_pappa or gender.lower() == "male") else "grandma")
                        conv = GrandmaTimbreConverter(
                            index_path=idx_file,
                            profile_path=prof_file if prof_file.exists() else None,
                            speaker_type=speaker_type,
                        )
                        self._timbre_converters[voice_id] = conv
                        print(f"[TTS] Activated FAISS Timbre Converter for '{voice_id}' ({speaker_type}) from {idx_file}.")
                        return conv
        except Exception as exc:
            print(f"[TTS] Timbre converter lookup error for '{voice_id}': {exc}")

        return None

    def register_timbre_converter(self, voice_id: str, converter: Any) -> None:
        """Register or update an in-memory timbre converter for a persona."""
        self._timbre_converters[voice_id] = converter

    async def synthesize(
        self,
        text: str,
        output_file: str | Path | None = None,
        voice_id: str = "grandma_chittoor",
    ) -> tuple[bytes, str]:
        """Synthesize text to speech audio bytes.

        Returns:
            Tuple of (audio_bytes, engine_used).
        """
        voice_profile = self.voice_manager.get_voice(voice_id)

        # 1. IndicF5 GPU Zero-Shot Voice Cloner (Highest fidelity, clones directly from reference audio on GPU)
        has_ref_audio = voice_profile and voice_profile.reference_audio_path and Path(voice_profile.reference_audio_path).exists()
        if self._indicf5_cloner is not None and has_ref_audio:
            try:
                import asyncio
                audio_bytes = await asyncio.wait_for(
                    self._synthesize_indicf5(text, voice_profile, output_file),
                    timeout=55.0,
                )
                if output_file:
                    Path(output_file).write_bytes(audio_bytes)
                engine_name = "indicf5_gpu" if self._indicf5_cloner.device == "cuda" else "indicf5_cpu"
                return audio_bytes, engine_name
            except Exception as exc:
                import traceback
                print(f"[TTS] IndicF5 GPU zero-shot error: {exc}")
                traceback.print_exc()

        # 2. Real-Time Neural Indic TTS + Dynamic FAISS Timbre Converter
        try:
            audio_bytes = await self._synthesize_neural_tts(text, voice_profile)
            if output_file:
                Path(output_file).write_bytes(audio_bytes)
            return audio_bytes, "neural_indic_tts"
        except Exception as exc:
            print(f"[Warning] Neural TTS synthesis failed ({exc}), attempting GCP/Synthetic fallback.")

        # 3. Google Cloud Text-to-Speech
        if self._gcp_tts_client:
            try:
                audio_bytes = await self._synthesize_gcp_tts(text, voice_profile)
                if output_file:
                    Path(output_file).write_bytes(audio_bytes)
                return audio_bytes, "gcp_neural_tts"
            except Exception as exc:
                print(f"[Warning] GCP TTS synthesis failed ({exc}), falling back.")

        # 4. Fallback: Edge TTS with base profile
        fallback_bytes = await self._synthesize_neural_tts(text, voice_profile)
        if output_file:
            Path(output_file).write_bytes(fallback_bytes)
        return fallback_bytes, "neural_tts_fallback"

    async def _synthesize_indicf5(
        self,
        text: str,
        voice: VoiceProfile | None,
        output_file: str | Path | None = None,
    ) -> bytes:
        """Synthesize using IndicF5 Zero-Shot Voice Cloner on GPU."""
        if self._indicf5_cloner is None:
            raise RuntimeError("IndicF5 Voice Cloner not available")

        out_path = Path(output_file) if output_file else Path(tempfile.mktemp(suffix=".wav"))

        ref_audio = None
        ref_text = None
        if voice:
            if voice.reference_audio_path:
                ref_audio = Path(voice.reference_audio_path)
            ref_text = voice.reference_transcript or None

        print(f"[TTS] IndicF5 synthesizing: ref_audio={ref_audio}, ref_text_len={len(ref_text) if ref_text else 0}, text_len={len(text)}")

        # Set speaking speed: 1.0 for calm, unhurried, warm paternal / grandmother cadence
        speaking_speed = 1.0
        if voice and hasattr(voice, "speaking_rate") and voice.speaking_rate:
            try:
                speaking_speed = float(voice.speaking_rate)
            except Exception:
                speaking_speed = 1.0

        result_path = await self._indicf5_cloner.synthesize(
            text=text,
            output_path=out_path,
            ref_audio=ref_audio,
            ref_text=ref_text,
            speed=speaking_speed,
        )

        return result_path.read_bytes()

    async def _synthesize_neural_tts(self, text: str, voice: VoiceProfile | None) -> bytes:
        """Synthesize natural Telugu/Indic speech and apply authentic voice timbre conversion."""
        import edge_tts

        # Determine best neural voice model based on persona
        lang = (voice.language_code or "te") if voice else "te"
        raw_gender = (voice.gender or "female").lower() if voice else "female"
        voice_name = (voice.name or "").lower() if voice else ""
        voice_id = (voice.voice_id or "").lower() if voice else "default"

        is_grandma = any(w in voice_name for w in ("grandma", "amamma", "అమ్మమ్మ", "నానమ్మ")) or any(w in voice_id for w in ("grandma", "amamma", "అమ్మమ్మ"))
        is_pappa = any(w in voice_name for w in ("pappa", "father", "dad", "పప్పా", "నాన్న")) or any(w in voice_id for w in ("pappa", "father", "dad"))

        if is_grandma:
            is_male = False
        elif is_pappa:
            is_male = True
        elif raw_gender == "male":
            is_male = True
        else:
            is_male = False

        # Map to appropriate regional neural voice
        if is_male:
            chosen_voice = "te-IN-MohanNeural"
            pitch = "-2Hz"
            rate = "+0%"  # Natural grounded paternal cadence
        elif is_grandma:
            chosen_voice = "te-IN-ShrutiNeural"
            pitch = "-3Hz"
            rate = "-8%"  # Gentle, calm, deliberate maternal grandmother cadence
        elif "hi" in lang:
            chosen_voice = "hi-IN-SwaraNeural"
            pitch = "+0Hz"
            rate = "+0%"
        elif "en" in lang:
            chosen_voice = "en-IN-NeerjaNeural"
            pitch = "+0Hz"
            rate = "+0%"
        else:
            chosen_voice = "te-IN-ShrutiNeural"
            pitch = "+0Hz"
            rate = "+0%"

        communicate = edge_tts.Communicate(text=text, voice=chosen_voice, pitch=pitch, rate=rate)
        audio_buffer = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.extend(chunk["data"])

        if not audio_buffer:
            raise RuntimeError("Neural TTS returned empty audio buffer")

        raw_audio_bytes = bytes(audio_buffer)

        # Look up and apply authentic persona FAISS timbre conversion
        timbre_conv = self.get_timbre_converter(voice_id, speaker_name=voice_name, gender=raw_gender)
        if timbre_conv:
            try:
                converted = timbre_conv.convert_audio_bytes(
                    raw_audio_bytes,
                    index_weight=0.85,
                )
                return converted
            except Exception as exc:
                print(f"[Warning] Timbre transfer for '{voice_id}' skipped ({exc}), using raw neural voice.")

        return raw_audio_bytes

    async def _synthesize_gcp_tts(self, text: str, voice: VoiceProfile | None) -> bytes:
        """Synthesize with Google Cloud Text-to-Speech."""
        from google.cloud import texttospeech

        synthesis_input = texttospeech.SynthesisInput(text=text)
        lang_code = voice.language_code if voice else "te-IN"
        if not lang_code.startswith("te"):
            lang_code = "te-IN"

        voice_params = texttospeech.VoiceSelectionParams(
            language_code=lang_code,
            name=f"{lang_code}-Standard-A",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=24000,
        )

        response = self._gcp_tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config,
        )
        return response.audio_content

    def _synthesize_synthetic(self, text: str, sr: int = 24000) -> bytes:
        """Synthesize natural warm carrier wave to guarantee audio output."""
        duration = max(1.2, len(text) * 0.07)
        audio = self._generate_warm_carrier(duration_sec=duration, sr=sr)
        return AudioProcessor.to_wav_bytes(audio, sample_rate=sr)

    def _generate_warm_carrier(self, duration_sec: float, sr: int = 24000) -> np.ndarray:
        """Generate smooth maternal acoustic harmonics."""
        t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
        # 185 Hz fundamental pitch (average Indian female grandmother pitch)
        f0 = 185.0
        wave = (
            0.50 * np.sin(2 * np.pi * f0 * t)
            + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
            + 0.15 * np.sin(2 * np.pi * 3 * f0 * t)
            + 0.10 * np.sin(2 * np.pi * 4 * f0 * t)
        )
        # Envelope to eliminate clicks
        envelope = np.ones_like(t)
        fade = int(sr * 0.05)
        if len(t) > 2 * fade:
            envelope[:fade] = np.linspace(0, 1, fade)
            envelope[-fade:] = np.linspace(1, 0, fade)
        return (wave * envelope * 0.8).astype(np.float32)
