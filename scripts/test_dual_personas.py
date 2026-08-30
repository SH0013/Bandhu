"""Verify both Pappa and Grandma persona responses and voice synthesis isolation."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import urllib.request
import json

def test_chat(persona_id, msg, speaker):
    payload = {
        "message": msg,
        "speaker_name": speaker,
        "persona_id": persona_id,
        "generate_audio": True,
    }
    req = urllib.request.Request(
        "http://127.0.0.1:8080/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=40) as resp:
        res = json.loads(resp.read().decode())
        print(f"=== Persona: {persona_id} ===")
        print(f"  Persona Name: {res.get('persona_name')}")
        print(f"  Reply Text: {res.get('reply_text')}")
        print(f"  Audio URL: {res.get('audio_url')}")
        return res

print("1. Testing Pappa chat turn...")
res_pappa = test_chat("pappa", "పప్పా అన్నం తిన్నారా? ఎలా ఉన్నారు?", "కూతురు")

print("\n2. Testing Grandma chat turn...")
res_grandma = test_chat("grandma_chittoor", "అమ్మమ్మ ఎలా ఉన్నారు? ఏం వండారు?", "మనవడు")
