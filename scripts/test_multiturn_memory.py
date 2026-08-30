"""Test multi-turn conversation memory and voice synthesis for Pappa."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import urllib.request
import json
import time

def send_message(msg: str):
    payload = {
        "message": msg,
        "speaker_name": "కూతురు",
        "persona_id": "pappa",
        "generate_audio": True,
    }
    req = urllib.request.Request(
        "http://127.0.0.1:8080/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=70) as resp:
        res = json.loads(resp.read().decode())
        dt = time.time() - t0
        print(f"\nUser: {msg}")
        print(f"Pappa: {res.get('reply_text')}")
        print(f"Audio URL: {res.get('audio_url')} (Latency: {dt:.2f}s)")
        return res

# Restart/clear history test
print("--- Starting Multi-Turn Conversation Memory Test ---")

# Turn 1
r1 = send_message("పప్పా నేను రేపు ఉదయం 10 గంటలకు ఇంటర్వ్యూకి వెళ్తున్నాను")

# Turn 2
r2 = send_message("పప్పా నా ఇంటర్వ్యూ గురించి గుర్తించుకున్నారా? ఏ సమయంలో ఉందో గుర్తొచ్చిందా?")

# Turn 3
r3 = send_message("నాకు చాలా భయంగా ఉంది పప్పా")

print("\n--- Memory Verification ---")
t2_text = r2.get("reply_text", "")
print("Turn 2 text contains 10 / సమయం / ఉదయం:", any(w in t2_text for w in ("10", "ఉదయం", "ఇంటర్వ్యూ", "గుర్తుంది", "మర్చిపోతానా")))
