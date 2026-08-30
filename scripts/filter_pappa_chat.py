"""Parse WhatsApp chat strictly up to March 14th and extract authentic Pappa dialogue."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import re
import json
from pathlib import Path
from collections import Counter

raw_chat_path = Path(r"C:\Users\msree\Downloads\Compressed\WhatsApp Chat - Pappa ❤️\_chat.txt")
cutoff_date = "2026-03-14"

content = raw_chat_path.read_text(encoding="utf-8", errors="replace")
pattern = re.compile(r"\[(\d{4}-\d{2}-\d{2}),?\s*([^\]]+)\]\s*([^:]+):\s*(.*)")

pappa_messages = []
daughter_messages = []
dialogue_pairs = []

last_speaker = None
last_msg = None

for line in content.splitlines():
    m = pattern.search(line)
    if m:
        dt, time_str, sender, msg = m.groups()
        if dt > cutoff_date:
            continue  # Strictly ignore everything after March 14th!
        
        sender = sender.strip()
        msg = msg.strip()
        
        if any(marker in msg for marker in (
            "<attached:", "Missed video call", "Missed voice call",
            "Messages and calls are end-to-end", "You pinned a message",
            "deleted this message", "This message was deleted", "Voice call,", "Video call,"
        )):
            continue
        if not msg:
            continue
            
        if "Pappa" in sender:
            pappa_messages.append({"date": dt, "time": time_str, "text": msg})
            if last_speaker == "daughter" and last_msg:
                dialogue_pairs.append({"user": last_msg, "pappa": msg, "date": dt})
            last_speaker = "pappa"
            last_msg = msg
        else:
            daughter_messages.append({"date": dt, "time": time_str, "text": msg})
            last_speaker = "daughter"
            last_msg = msg

print(f"Total Pappa text messages strictly on/before {cutoff_date}: {len(pappa_messages)}")
print(f"Total Daughter text messages strictly on/before {cutoff_date}: {len(daughter_messages)}")
print(f"Dialogue pairs extracted: {len(dialogue_pairs)}")

all_pappa_text = " ".join([m["text"] for m in pappa_messages])
print("\nPet name counts in Pappa's pre-March 14 messages:")
for term in ["నానమ్మ", "డాడీ", "బేటా", "బంగారం", "దెయ్యం"]:
    print(f"  {term}: {all_pappa_text.count(term)}")

# Save filtered message list
pappa_dir = Path("data/personas/pappa")
(pappa_dir / "chat_exports").mkdir(parents=True, exist_ok=True)

out_txt = pappa_dir / "chat_exports" / "pappa_extracted_messages.txt"
out_txt.write_text("\n".join([m["text"] for m in pappa_messages]), encoding="utf-8")

out_json = pappa_dir / "chat_exports" / "pappa_analysis.json"
out_json.write_text(json.dumps({
    "cutoff_date": cutoff_date,
    "total_pappa_turns": len(pappa_messages),
    "frequent_terms": {
        "నానమ్మ": all_pappa_text.count("నానమ్మ"),
        "డాడీ": all_pappa_text.count("డాడీ"),
        "బేటా": all_pappa_text.count("బేటా"),
        "దెయ్యం": all_pappa_text.count("దెయ్యం"),
        "బంగారం": all_pappa_text.count("బంగారం"),
    },
    "dialogue_samples": dialogue_pairs[:50]
}, indent=2, ensure_ascii=False), encoding="utf-8")

print("\nSuccessfully updated data/personas/pappa/chat_exports/ with strict <= March 14 data.")
