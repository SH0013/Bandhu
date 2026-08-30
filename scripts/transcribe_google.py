"""Transcribe Pappa's reference clips using Google Speech Recognition (Telugu)."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import speech_recognition as sr
from pathlib import Path
import soundfile as sf
import numpy as np
from scipy import signal as sig
import io

# Also convert the 3.5s clip to WAV first
for opus_name in ["custom_persona_57531d_5.opus"]:
    src = Path("data/reference_audio") / opus_name
    dst = Path("data/reference_audio") / opus_name.replace(".opus", "_24k.wav")
    if not dst.exists():
        data, sr2 = sf.read(str(src), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        num_samples = int(len(data) * 24000 / sr2)
        data_24k = sig.resample(data, num_samples).astype(np.float32)
        peak = np.max(np.abs(data_24k))
        if peak > 1e-4:
            data_24k = data_24k / peak * 0.89
        sf.write(str(dst), data_24k, 24000, subtype="PCM_16")
        print(f"Converted {opus_name} -> {dst.name}")

clips = [
    "data/reference_audio/custom_persona_57531d_5_24k.wav",
    "data/reference_audio/custom_persona_9451da_6_24k.wav",
    "data/reference_audio/custom_persona_4f6e6b_3_24k.wav",
    "data/reference_audio/custom_persona_39185f_4_24k.wav",
]

r = sr.Recognizer()
for clip in clips:
    try:
        with sr.AudioFile(clip) as source:
            audio = r.record(source)
        text = r.recognize_google(audio, language="te-IN")
        print(f"OK  {clip}")
        print(f"    -> {text}")
    except sr.UnknownValueError:
        print(f"ERR {clip}: Could not understand audio")
    except Exception as e:
        print(f"ERR {clip}: {e}")
