"""Bidirectional chat channels for Bandhu: Telegram and WhatsApp (Meta Cloud API).

The web UI is just one channel. With this module, the same Bandhu agent
can be reached from any of:

  - Web UI              → POST /api/chat
  - Telegram bot        → POST /api/webhook/telegram
  - WhatsApp (Meta)     → POST /api/webhook/whatsapp

All three call `process_inbound_message(channel, user_id, text)` which
returns the agent's reply text. Each channel adapter is responsible for
sending that reply back to the user on its own platform.

Cost:
  - Telegram Bot API   — completely free, 30 msg/sec
  - WhatsApp Meta      — 1,000 service conversations/month free
  - Twilio             — NOT used here (costs ~$0.012/msg)

Setup:
  - Telegram:
      1. Message @BotFather on Telegram, /newbot
      2. Save the bot token
      3. Set webhook: https://<your-cloud-run>/api/webhook/telegram
         OR use long polling (no webhook needed for dev)
      4. Set TELEGRAM_BOT_TOKEN in .env
  - WhatsApp (Meta Cloud):
      1. Create a Meta Business account, get a WhatsApp Business number
      2. Set webhook: https://<your-cloud-run>/api/webhook/whatsapp
         with verify token = WHATSAPP_VERIFY_TOKEN in .env
      3. Set WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_ACCESS_TOKEN, and
         a real WHATSAPP_RECIPIENT_PHONE in .env (only for the demo
         send endpoint)
"""
from __future__ import annotations

import json
import logging
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any

import httpx

from bandhu.config import settings

logger = logging.getLogger("bandhu.chat_channels")


@dataclass
class InboundMessage:
    """A single message received from any chat channel."""
    channel: str          # "telegram" | "whatsapp" | "web"
    user_id: str          # phone number, telegram chat id, etc.
    user_name: str        # display name if known
    text: str             # the message text
    raw: dict[str, Any]   # original payload for debugging
    voice_file_id: str | None = None
    voice_mime_type: str = "audio/ogg"


@dataclass
class OutboundMessage:
    """A reply ready to be sent back on a channel."""
    channel: str
    recipient: str
    text: str
    parse_mode: str | None = None  # "HTML" | "MarkdownV2" | None


# ────────────────────────────────────────────────────────────────────
# 1. Channel-agnostic inbound parsing
# ────────────────────────────────────────────────────────────────────

def parse_telegram_update(payload: dict[str, Any]) -> InboundMessage | None:
    """Parse a Telegram Bot API update payload into an InboundMessage."""
    msg = payload.get("message") or payload.get("edited_message")
    if not msg:
        return None
    text = (msg.get("text") or msg.get("caption") or "").strip()
    voice = msg.get("voice") or msg.get("audio")
    voice_file_id = voice.get("file_id") if voice else None
    voice_mime_type = (voice.get("mime_type") if voice else None) or "audio/ogg"

    if not text and not voice_file_id:
        return None

    chat = msg.get("chat") or {}
    user = msg.get("from") or {}
    chat_id = str(chat.get("id") or user.get("id", ""))
    return InboundMessage(
        channel="telegram",
        user_id=chat_id,
        user_name=user.get("first_name") or user.get("username") or "Telegram User",
        text=text or "[Voice Note Received]",
        raw=payload,
        voice_file_id=voice_file_id,
        voice_mime_type=voice_mime_type,
    )


async def download_telegram_file(file_id: str) -> bytes | None:
    """Download audio file from Telegram Bot API via getFile."""
    import os
    token = settings.telegram_bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token or not file_id:
        return None
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            get_file_url = f"https://api.telegram.org/bot{token}/getFile?file_id={file_id}"
            res = await client.get(get_file_url)
            if res.status_code != 200:
                return None
            data = res.json()
            file_path = data.get("result", {}).get("file_path")
            if not file_path:
                return None
            download_url = f"https://api.telegram.org/file/bot{token}/{file_path}"
            dl_res = await client.get(download_url)
            if dl_res.status_code == 200:
                return dl_res.content
    except Exception as exc:
        logger.warning("Failed to download Telegram voice file %s: %s", file_id, exc)
    return None


async def send_telegram_chat_action(recipient: str, action: str = "record_voice") -> None:
    """Send typing or record_voice indicator to Telegram client."""
    import os
    token = settings.telegram_bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        return
    url = f"https://api.telegram.org/bot{token}/sendChatAction"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(url, json={"chat_id": recipient, "action": action})
    except Exception:
        pass


def parse_whatsapp_payload(payload: dict[str, Any]) -> list[InboundMessage]:
    """Parse a Meta WhatsApp Cloud API webhook payload.

    The payload can contain multiple messages, one per change entry.
    We return a list (usually length 0 or 1) so the caller can iterate.

    For Twilio's form-encoded body (legacy), use parse_twilio_form() instead.
    """
    messages: list[InboundMessage] = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            contacts = {c.get("wa_id"): c for c in value.get("contacts", [])}
            for msg in value.get("messages", []):
                if msg.get("type") != "text":
                    # We only handle text. Voice notes would need STT first.
                    continue
                wa_id = msg.get("from", "")
                contact = contacts.get(wa_id, {})
                name = contact.get("profile", {}).get("name", "WhatsApp User")
                text = msg.get("text", {}).get("body", "").strip()
                if not text:
                    continue
                messages.append(InboundMessage(
                    channel="whatsapp",
                    user_id=wa_id,
                    user_name=name,
                    text=text,
                    raw=msg,
                ))
    return messages


def parse_twilio_form(body_str: str) -> InboundMessage | None:
    """Parse a Twilio form-encoded WhatsApp body (legacy path)."""
    form = urllib.parse.parse_qs(body_str)
    text = (form.get("Body", [""])[0] or "").strip()
    sender = (form.get("From", ["WhatsApp User"])[0] or "").replace("whatsapp:", "")
    if not text:
        return None
    return InboundMessage(
        channel="twilio",
        user_id=sender,
        user_name=sender or "WhatsApp User",
        text=text,
        raw={"Body": text, "From": sender},
    )


# ────────────────────────────────────────────────────────────────────
# Telegram Interactive Keyboards & Session Management
# ────────────────────────────────────────────────────────────────────

TELEGRAM_KEYBOARD = {
    "keyboard": [
        [{"text": "👵 అమ్మమ్మ (Grandma)"}, {"text": "👨 పప్పా (Pappa)"}],
        [{"text": "🎙️ Voice + Text"}, {"text": "💬 Text Only"}, {"text": "🔊 Voice Only"}],
        [{"text": "⚙️ Status & Mode"}, {"text": "📖 Help & Guide"}],
    ],
    "resize_keyboard": True,
    "is_persistent": True,
}

_user_sessions: dict[str, dict[str, Any]] = {}


def get_user_session(user_id: str) -> dict[str, Any]:
    """Retrieve or initialize persistent session preferences for a user."""
    if user_id not in _user_sessions:
        _user_sessions[user_id] = {
            "persona_id": "grandma_chittoor",
            "output_mode": "combined",  # "combined" | "text_only" | "voice_only"
            "user_name": "Family Member",
        }
    return _user_sessions[user_id]


# ────────────────────────────────────────────────────────────────────
# 2. Channel-agnostic outbound senders
# ────────────────────────────────────────────────────────────────────

async def send_telegram_message(
    recipient: str, text: str, reply_markup: dict[str, Any] | None = None
) -> str:
    """Send a text message to a Telegram chat with optional keyboard buttons."""
    import os
    token = settings.telegram_bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        return "telegram_not_configured"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    chunks = _split_for_telegram(text)
    last_id = "ok"
    markup = reply_markup or TELEGRAM_KEYBOARD
    async with httpx.AsyncClient(timeout=15.0) as client:
        for i, chunk in enumerate(chunks):
            payload: dict[str, Any] = {
                "chat_id": recipient,
                "text": chunk,
                "disable_web_page_preview": True,
            }
            # Attach keyboard to the last message chunk
            if i == len(chunks) - 1:
                payload["reply_markup"] = markup

            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                last_id = data.get("result", {}).get("message_id", "ok")
            else:
                logger.warning("Telegram send failed %s: %s", resp.status_code, resp.text[:200])
                return f"telegram_failed:{resp.status_code}:{resp.text[:120]}"
    return f"telegram:{last_id}"


async def send_telegram_voice(
    recipient: str, audio_bytes: bytes, caption: str = ""
) -> str:
    """Send a synthesized audio voice note directly to Telegram."""
    import os
    token = settings.telegram_bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        return "telegram_not_configured"
    url = f"https://api.telegram.org/bot{token}/sendVoice"
    async with httpx.AsyncClient(timeout=45.0) as client:
        files = {"voice": ("voice.wav", audio_bytes, "audio/wav")}
        data: dict[str, Any] = {"chat_id": recipient}
        if caption:
            data["caption"] = caption[:1024]
        resp = await client.post(url, data=data, files=files)
        if resp.status_code == 200:
            return "ok"
        logger.warning("Telegram sendVoice failed %s: %s", resp.status_code, resp.text[:200])
        return f"voice_failed:{resp.status_code}:{resp.text[:120]}"


def _split_for_telegram(text: str, limit: int = 4000) -> list[str]:
    """Split text into <= limit-char chunks, preferring newline boundaries."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
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


async def send_whatsapp_cloud_message(recipient: str, text: str) -> str:
    """Send a text message via Meta WhatsApp Cloud API (free tier)."""
    if not (settings.whatsapp_phone_number_id and settings.whatsapp_access_token):
        return "whatsapp_cloud_not_configured"
    url = f"https://graph.facebook.com/v20.0/{settings.whatsapp_phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_access_token}",
        "Content-Type": "application/json",
    }
    to = recipient.replace("whatsapp:", "").replace("+", "").strip()
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text[:4000]},
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code in (200, 201):
            return f"whatsapp_cloud:{resp.json().get('messages', [{}])[0].get('id', 'ok')}"
        return f"whatsapp_cloud_failed:{resp.status_code}:{resp.text[:120]}"


async def send_message(msg: OutboundMessage) -> str:
    """Dispatch an OutboundMessage to the correct channel sender."""
    if msg.channel == "telegram":
        return await send_telegram_message(msg.recipient, msg.text)
    if msg.channel == "whatsapp":
        return await send_whatsapp_cloud_message(msg.recipient, msg.text)
    return f"unknown_channel:{msg.channel}"


# ────────────────────────────────────────────────────────────────────
# 3. Interactive Command & Button Handling
# ────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────
# 3. Interactive Command & Button Handling
# ────────────────────────────────────────────────────────────────────

HELP_TEXT = """\
🌟 *Bandhu (బంధు) — Companion Chat & Voice Friend*

I am your AI family companion. You can chat with me in Telugu (ఉదా: బాగున్నావా) or Tanglish (e.g. *baaane unna nanamma*), or hold the mic to send a voice note!

*👥 Available Personas:*
• 👵 *అమ్మమ్మ (Grandma · Chittoor)* — Warm, caring Rayalaseema Telugu
• 👨 *పప్పా (Pappa · Telangana)* — Playful, protective Telangana father

*🎙️ Audio Modes:*
• 🎙️ *Voice + Text* — Cloned voice audio note + text message
• 💬 *Text Only* — Fast instant text messages
• 🔊 *Voice Only* — Direct audio voice notes

👉 _Tap any button below to switch persona or voice mode!_
"""


async def register_telegram_commands() -> bool:
    """Register native Telegram bottom-left menu commands via setMyCommands."""
    import os
    token = settings.telegram_bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        return False
    url = f"https://api.telegram.org/bot{token}/setMyCommands"
    commands = [
        {"command": "grandma", "description": "👵 Switch to Grandma (అమ్మమ్మ)"},
        {"command": "pappa", "description": "👨 Switch to Pappa (పప్పా)"},
        {"command": "voice", "description": "🎙️ Mode: Voice Audio Notes + Text"},
        {"command": "text", "description": "💬 Mode: Text Only (Fast)"},
        {"command": "status", "description": "⚙️ Current Persona & Mode"},
        {"command": "help", "description": "📖 Guide & Menu Buttons"},
    ]
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post(url, json={"commands": commands})
            return res.status_code == 200
    except Exception:
        return False


def maybe_handle_command(text: str, agent, user_id: str = "") -> str | None:
    """Handle slash commands and interactive button presses."""
    t_clean = text.strip()
    session = get_user_session(user_id) if user_id else {}

    # 1. Persona Switching Buttons & Commands
    if t_clean in ("👵 అమ్మమ్మ (Grandma)", "/grandma", "/amma", "grandma"):
        if session:
            session["persona_id"] = "grandma_chittoor"
        profile = agent.memory_store.get_persona("grandma_chittoor")
        if profile:
            agent.set_persona(profile)
        p_name = profile.name if profile else "అమ్మమ్మ (Grandma)"
        return (
            f"👵 *Active Persona: {p_name} (Chittoor)*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "స్వామి దయతో మంచిగా ఉండు నాయనా! నాతో ఏం మాట్లాడాలనుకుంటున్నావు బా?\n\n"
            "💡 _You can type in pure Telugu, Tanglish ('baaane unna'), or hold the mic 🎙️ to send a voice note!_"
        )

    if t_clean in ("👨 పప్పా (Pappa)", "/pappa", "/dad", "pappa"):
        if session:
            session["persona_id"] = "pappa"
        profile = agent.memory_store.get_persona("pappa")
        if profile:
            agent.set_persona(profile)
        p_name = profile.name if profile else "పప్పా (Pappa)"
        return (
            f"👨 *Active Persona: {p_name} (Telangana)*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "హాయ్ నానమ్మ! ఏం చేస్తున్నావ్ బేటా? తిన్నావా లేదా?\n\n"
            "💡 _Don't worry beta, antha manchigane avtadi! Text or voice note pampinchu._"
        )

    # 2. Voice / Output Mode Buttons & Commands
    if t_clean in ("🎙️ Voice + Text", "/voice_and_text", "/combined", "/voice", "/audio"):
        if session:
            session["output_mode"] = "combined"
        return (
            "🎙️ *Output Mode: Voice + Text [Active]*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "I will now send **both audio voice notes and text messages** for every turn!"
        )

    if t_clean in ("💬 Text Only", "/text", "/textonly"):
        if session:
            session["output_mode"] = "text_only"
        return (
            "💬 *Output Mode: Text Only [Active]*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "Fast text messaging active (audio generation paused for lightning-fast replies)."
        )

    if t_clean in ("🔊 Voice Only", "/voiceonly"):
        if session:
            session["output_mode"] = "voice_only"
        return (
            "🔊 *Output Mode: Voice Only [Active]*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "I will respond with **direct playable voice notes**!"
        )

    # 3. Status & Settings Button
    if t_clean in ("⚙️ Status & Mode", "/status", "/settings", "/mode", "mode", "status"):
        p_id = session.get("persona_id", "grandma_chittoor")
        cur_p = "👵 అమ్మమ్మ (Grandma · Chittoor)" if p_id == "grandma_chittoor" else "👨 పప్పా (Pappa · Telangana)"
        mode_map = {
            "combined": "🎙️ Voice + Text (Audio Notes Active)",
            "text_only": "💬 Text Only (Fast)",
            "voice_only": "🔊 Voice Only (Audio Notes)",
        }
        cur_mode = mode_map.get(session.get("output_mode", "combined"), "🎙️ Voice + Text")
        return (
            "⚙️ *Bandhu Companion Status:*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"• *Active Persona:* {cur_p}\n"
            f"• *Audio Mode:* {cur_mode}\n"
            f"• *AI Engine:* Google Gemini 3.7 Flash\n"
            f"• *Voice Cloner:* IndicF5 Zero-Shot Neural Cloner (24kHz Studio Audio)\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "👉 _Tap any button below to switch persona or voice mode!_"
        )

    # 4. Help, Start, Restart & Info
    if t_clean in ("📖 Help & Guide", "/help", "/start", "/restart", "/reset", "/h", "help", "start", "restart"):
        return HELP_TEXT

    if t_clean in ("/personas", "personas", "/persona"):
        ids = [p.persona_id for p in agent.memory_store.list_personas()] or ["grandma_chittoor", "pappa"]
        return (
            "👥 *Available Personas:*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            + "\n".join(f"• {i}" for i in ids)
            + "\n\n👉 _Tap the buttons below to switch instantly!_"
        )

    if t_clean.startswith("/persona "):
        arg = t_clean.split(maxsplit=1)[1].strip()
        profile = agent.memory_store.get_persona(arg)
        if not profile:
            return f"Unknown persona '{arg}'. Tap the buttons below to switch!"
        agent.set_persona(profile)
        if session:
            session["persona_id"] = profile.persona_id
        return (
            f"👥 *Active Persona: {profile.name} ({profile.relationship})*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"Switched to {profile.name} ({profile.relationship})."
        )

    if t_clean.startswith("/"):
        return f"Unknown command: {t_clean}. Tap the buttons below or try /help."

    return None
