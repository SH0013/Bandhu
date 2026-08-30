"""Test IndicF5 zero-shot synthesis for Pappa via the chat API."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import urllib.request
import json

payload = {
    "message": "పప్పా అన్నం తిన్నారా?",
    "speaker_name": "కూతురు",
    "persona_id": "pappa",
    "generate_audio": True,
}

req = urllib.request.Request(
    "http://127.0.0.1:8080/api/chat",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)

try:
    with urllib.request.urlopen(req, timeout=45) as resp:
        res = json.loads(resp.read().decode())
        print(f"Reply: {res.get('reply_text', 'NO REPLY')}")
        print(f"Audio: {res.get('audio_url', 'NO AUDIO')}")
        print(f"Persona: {res.get('persona_name', '?')}")
        print(f"Model: {res.get('model', '?')}")
except Exception as e:
    print(f"ERROR: {e}")
