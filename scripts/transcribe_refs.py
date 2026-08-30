"""Transcribe Pappa's reference audio clips for IndicF5 voice cloning."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

clips = [
    "data/reference_audio/custom_persona_4f6e6b_3_24k.wav",
    "data/reference_audio/custom_persona_39185f_4_24k.wav",
    "data/reference_audio/custom_persona_9451da_6_24k.wav",
]

# Try OpenAI Whisper
try:
    import whisper
    print("Using OpenAI Whisper")
    model = whisper.load_model("base")
    for clip in clips:
        result = model.transcribe(clip, language="te")
        text = result["text"]
        print(f"{clip}: {text}")
    sys.exit(0)
except ImportError:
    print("whisper not available")

# Try Google Speech Recognition
try:
    import speech_recognition as sr2
    print("Using Google Speech Recognition")
    r = sr2.Recognizer()
    for clip in clips:
        with sr2.AudioFile(clip) as source:
            audio = r.record(source)
        try:
            text = r.recognize_google(audio, language="te-IN")
            print(f"{clip}: {text}")
        except Exception as e:
            print(f"{clip}: ERROR {e}")
    sys.exit(0)
except ImportError:
    print("speech_recognition not available")

print("No transcription engine available. Install openai-whisper or SpeechRecognition.")
