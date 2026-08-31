"""Telegram long-polling client.

This runs alongside the FastAPI server (or by itself) and:
  1. Polls Telegram's getUpdates API for new messages
  2. Forwards each message to the Bandhu agent
  3. Sends the agent's reply back via sendMessage
  4. Resolves chat_id → persona_id per user (multi-user safe)
  5. Logs every interaction

Why this script instead of a webhook:
  - No HTTPS or public URL needed (works on localhost)
  - No tunnel / ngrok required
  - Set-and-forget: just run it in a background process

To use:
  1. Set TELEGRAM_BOT_TOKEN in .env
  2. Run:  python scripts/telegram_polling.py
  3. Open Telegram, message your bot, get replies

The script will remember your chat_id once you /start, and you can
add more users later by having them /start the bot too.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

# Ensure src/ is on the path (same as run_server.py)
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import httpx

from bandhu.agent.gemini_agent import BandhuGeminiAgent
from bandhu.api.chat_channels import (
    maybe_handle_command,
    parse_telegram_update,
)
from bandhu.config import settings
from bandhu.memory.store import MemoryStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("telegram_polling")

POLL_URL = "https://api.telegram.org/bot{}/getUpdates"
SEND_URL = "https://api.telegram.org/bot{}/sendMessage"

# Map chat_id -> persona_id (per-user persona, persisted in a tiny file)
PERSONA_STATE_FILE = Path(__file__).parent.parent / "data" / "telegram_personas.json"


def load_persona_map() -> dict[str, str]:
    if PERSONA_STATE_FILE.exists():
        try:
            return json.loads(PERSONA_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_persona_map(m: dict[str, str]) -> None:
    PERSONA_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PERSONA_STATE_FILE.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")


async def process_update(update: dict, agent: BandhuGeminiAgent, memory: MemoryStore) -> None:
    """Take a raw Telegram update, run it through the agent, send reply."""
    msg = parse_telegram_update(update)
    if msg is None:
        return
    log.info("TG <- %s (%s): %s", msg.user_name, msg.user_id, msg.text[:80])

    # Determine persona for this user (or use default)
    personas = load_persona_map()
    desired_pid = personas.get(msg.user_id) or os.getenv("TELEGRAM_DEFAULT_PERSONA", "grandma_chittoor")
    if agent.persona.persona_id != desired_pid:
        profile = memory.get_persona(desired_pid)
        if profile:
            agent.set_persona(profile)

    # Slash command?
    cmd_reply = maybe_handle_command(msg.text, agent)
    if cmd_reply is not None:
        # Special handling: if the user did /persona X, save their choice
        parts = msg.text.strip().split(maxsplit=1)
        if len(parts) >= 2 and parts[0].lower() == "/persona":
            new_pid = parts[1].strip()
            profile = memory.get_persona(new_pid)
            if profile:
                personas[msg.user_id] = new_pid
                save_persona_map(personas)
                log.info("TG  set persona for %s -> %s", msg.user_id, new_pid)
        reply_text = cmd_reply
    else:
        # Run the agent
        try:
            agent_res = await agent.reply(msg.text, speaker_name=msg.user_name or "Chat User")
            reply_text = agent_res.reply_text or "…"
            log.info("TG  tools=%d model=%s", len(agent_res.tools_executed or []), agent_res.model)
        except Exception as exc:
            log.exception("agent error: %s", exc)
            reply_text = "క్షమించు, కొద్దిగా ఇబ్బంది ఉంది. మళ్ళీ ప్రయత్నించు."

    # Send the reply (chunked if too long)
    token = settings.telegram_bot_token
    chunks = _split(reply_text, limit=4000)
    async with httpx.AsyncClient(timeout=15.0) as client:
        for chunk in chunks:
            r = await client.post(
                SEND_URL.format(token),
                json={
                    "chat_id": msg.user_id,
                    "text": chunk,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
            )
            if r.status_code != 200:
                log.warning("sendMessage failed %s: %s", r.status_code, r.text[:200])
            else:
                log.info("TG -> %s (%d chars)", msg.user_name, len(chunk))


def _split(text: str, limit: int = 4000) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            chunks.append(text[:limit])
            text = text[limit:]
        else:
            chunks.append(text[: cut + 1])
            text = text[cut + 1 :]
    return chunks


async def poll_loop(agent: BandhuGeminiAgent, memory: MemoryStore) -> None:
    if not settings.telegram_bot_token:
        log.error("TELEGRAM_BOT_TOKEN is not set. Edit your .env and retry.")
        return

    log.info("Telegram polling started for @%s", "Bandhu_gbot")
    log.info("Open Telegram, message the bot, watch this log.")
    log.info("Press Ctrl+C to stop.")
    print()

    offset = 0
    backoff = 1
    async with httpx.AsyncClient(timeout=35.0) as client:
        while True:
            try:
                r = await client.get(
                    POLL_URL.format(settings.telegram_bot_token),
                    params={"timeout": 30, "offset": offset, "allowed_updates": json.dumps(["message", "edited_message"])},
                )
                if r.status_code != 200:
                    log.warning("getUpdates HTTP %s: %s", r.status_code, r.text[:200])
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    continue
                backoff = 1
                data = r.json()
                for update in data.get("result", []):
                    offset = max(offset, update["update_id"] + 1)
                    try:
                        await process_update(update, agent, memory)
                    except Exception:
                        log.exception("error processing update %s", update.get("update_id"))
            except httpx.ReadTimeout:
                # Long-poll timeout is normal; just retry
                continue
            except Exception as exc:
                log.warning("polling error: %s", exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)


def main() -> int:
    if not settings.telegram_bot_token:
        print("ERROR: TELEGRAM_BOT_TOKEN is not set in your .env")
        return 1

    memory = MemoryStore()
    agent = BandhuGeminiAgent(memory_store=memory)
    # Set the default persona
    default_pid = os.getenv("TELEGRAM_DEFAULT_PERSONA", "grandma_chittoor")
    profile = memory.get_persona(default_pid)
    if profile:
        agent.set_persona(profile)
        log.info("Default persona: %s (%s)", profile.name, profile.persona_id)
    else:
        log.warning("Default persona '%s' not found in store; agent will use its built-in default.", default_pid)

    try:
        asyncio.run(poll_loop(agent, memory))
    except KeyboardInterrupt:
        log.info("Stopped by user.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
