"""Unit tests for Bandhu Audio Processor and Adaptive Voice Synthesizer."""

import tempfile
from pathlib import Path
import pytest
import soundfile as sf
from bandhu.audio.processor import AudioProcessor
from bandhu.audio.tts import AdaptiveVoiceSynthesizer
from bandhu.audio.voice_manager import VoiceProfileManager


@pytest.fixture
def tts_engine(tmp_path: Path) -> AdaptiveVoiceSynthesizer:
    voice_mgr = VoiceProfileManager(storage_dir=tmp_path / "voices")
    return AdaptiveVoiceSynthesizer(voice_manager=voice_mgr)


@pytest.mark.asyncio
async def test_adaptive_synthesis_wav_bytes(tts_engine: AdaptiveVoiceSynthesizer) -> None:
    wav_bytes, engine = await tts_engine.synthesize("నాయనా కడుపునిండా తింటివా?")
    assert len(wav_bytes) > 1000
    assert engine in ("neural_indic_tts", "indicf5_gpu", "gcp_neural_tts", "synthetic_adaptive")


@pytest.mark.asyncio
async def test_audio_processor_save_and_load(tmp_path: Path, tts_engine: AdaptiveVoiceSynthesizer) -> None:
    out_file = tmp_path / "test_out.wav"
    wav_bytes, _ = await tts_engine.synthesize("కన్నా బాగుండావా", output_file=out_file)
    assert out_file.exists()

    audio, sr = AudioProcessor.load_audio(out_file, target_sr=24000)
    assert len(audio) > 0
    assert sr == 24000


def test_voice_profile_registration(tmp_path: Path) -> None:
    mgr = VoiceProfileManager(storage_dir=tmp_path / "voice_reg")
    vp = mgr.register_voice(
        voice_id="dad_telugu",
        name="నాన్న (Father)",
        reference_audio_path=str(tmp_path / "dad.wav"),
        reference_transcript="ఎలా ఉన్నావురా",
        gender="male",
    )
    assert vp.voice_id == "dad_telugu"
    assert mgr.get_voice("dad_telugu") is not None
