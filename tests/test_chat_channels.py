"""Tests for the bidirectional chat-channel adapters (Telegram + WhatsApp)."""
from __future__ import annotations

import pytest

from bandhu.api.chat_channels import (
    OutboundMessage,
    maybe_handle_command,
    parse_telegram_update,
    parse_twilio_form,
    parse_whatsapp_payload,
    _split_for_telegram,
)
from bandhu.persona.models import PersonaProfile


# ───────────────────── Telegram parsing ─────────────────────

def test_parse_telegram_text_message() -> None:
    payload = {
        "update_id": 12345,
        "message": {
            "message_id": 1,
            "from": {"id": 999, "is_bot": False, "first_name": "Sai", "username": "saitest"},
            "chat": {"id": 999, "type": "private"},
            "date": 1700000000,
            "text": "అమ్మమ్మ నాకు జ్వరంగా ఉంది",
        },
    }
    m = parse_telegram_update(payload)
    assert m is not None
    assert m.channel == "telegram"
    assert m.user_id == "999"
    assert m.user_name == "Sai"
    assert m.text == "అమ్మమ్మ నాకు జ్వరంగా ఉంది"


def test_parse_telegram_ignores_non_text_updates() -> None:
    # Edited message with no text -> ignore
    payload = {"edited_message": {"message_id": 1, "date": 0}}
    assert parse_telegram_update(payload) is None
    # Callback query
    payload = {"callback_query": {"id": "1", "from": {"id": 1}}}
    assert parse_telegram_update(payload) is None
    # Empty body
    payload = {"message": {"from": {"id": 1, "first_name": "x"}, "text": ""}}
    assert parse_telegram_update(payload) is None


# ───────────────────── WhatsApp parsing ─────────────────────

def test_parse_whatsapp_meta_payload() -> None:
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "123",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"phone_number_id": "987"},
                            "contacts": [{"wa_id": "919999999999", "profile": {"name": "Sai"}}],
                            "messages": [
                                {
                                    "from": "919999999999",
                                    "id": "wamid.abc",
                                    "timestamp": "1700000000",
                                    "type": "text",
                                    "text": {"body": "Hi grandma"},
                                }
                            ],
                        },
                        "field": "messages",
                    }
                ],
            }
        ],
    }
    msgs = parse_whatsapp_payload(payload)
    assert len(msgs) == 1
    assert msgs[0].channel == "whatsapp"
    assert msgs[0].user_id == "919999999999"
    assert msgs[0].user_name == "Sai"
    assert msgs[0].text == "Hi grandma"


def test_parse_whatsapp_ignores_non_text() -> None:
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {"from": "1", "type": "audio", "audio": {"id": "x"}},
                                {"from": "1", "type": "image", "image": {"id": "y"}},
                            ]
                        }
                    }
                ]
            }
        ]
    }
    assert parse_whatsapp_payload(payload) == []


def test_parse_twilio_form() -> None:
    body = "Body=%E0%B0%85%E0%B0%AE%E0%B1%8D%E0%B0%AE%E0%B0%AE%E0%B1%8D%E0%B0%AE&From=whatsapp%3A%2B919876543210"
    m = parse_twilio_form(body)
    assert m is not None
    assert m.channel == "twilio"
    # parse_qs decodes percent-encoding; the Telugu text should round-trip
    assert "అమ్మమ్మ" in m.text
    assert m.user_id == "+919876543210"


def test_parse_twilio_form_empty() -> None:
    assert parse_twilio_form("") is None
    assert parse_twilio_form("Body=&From=") is None


# ───────────────────── Slash command handling ─────────────────────

class _FakeAgent:
    """Minimal stub exposing memory_store.list_personas / get_persona / set_persona."""

    def __init__(self, personas: list[PersonaProfile]) -> None:
        self._personas = {p.persona_id: p for p in personas}
        self.active = None
        # Build a real MemoryStore for list_personas to use
        from bandhu.memory.store import MemoryStore
        import tempfile, pathlib
        tmp = tempfile.mkdtemp()
        self.memory_store = MemoryStore(db_path=pathlib.Path(tmp) / "t.db", force_sqlite=True)
        for p in personas:
            self.memory_store.save_persona(p)

    def set_persona(self, profile: PersonaProfile) -> None:
        self.active = profile


def test_command_help() -> None:
    agent = _FakeAgent([PersonaProfile(persona_id="grandma_chittoor", name="Demo Amamma", relationship="Grandmother")])
    out = maybe_handle_command("/help", agent)
    assert out is not None and "companion chat" in out.lower()


def test_command_persona_switch() -> None:
    agent = _FakeAgent([
        PersonaProfile(persona_id="grandma_chittoor", name="Demo Amamma", relationship="Grandmother"),
        PersonaProfile(persona_id="pappa", name="Demo Nanna", relationship="Father"),
    ])
    out = maybe_handle_command("/persona pappa", agent)
    assert out is not None and "Demo Nanna" in out
    assert agent.active is not None and agent.active.persona_id == "pappa"


def test_command_unknown_persona() -> None:
    agent = _FakeAgent([PersonaProfile(persona_id="grandma_chittoor", name="x", relationship="y")])
    out = maybe_handle_command("/persona stranger", agent)
    assert out is not None and "Unknown persona" in out


def test_command_personas_list() -> None:
    agent = _FakeAgent([
        PersonaProfile(persona_id="grandma_chittoor", name="Demo Amamma", relationship="Grandmother"),
        PersonaProfile(persona_id="pappa", name="Demo Nanna", relationship="Father"),
    ])
    out = maybe_handle_command("/personas", agent)
    assert out is not None
    assert "grandma_chittoor" in out and "pappa" in out


def test_command_unknown_command_returns_message() -> None:
    agent = _FakeAgent([])
    out = maybe_handle_command("/whatever", agent)
    assert out is not None and "Unknown command" in out


def test_non_command_returns_none() -> None:
    agent = _FakeAgent([])
    # Regular chat messages should not be intercepted
    assert maybe_handle_command("hello grandma", agent) is None
    assert maybe_handle_command("అమ్మమ్మ ఎలా ఉన్నావు?", agent) is None


# ───────────────────── Telegram message splitter ─────────────────────

def test_split_short_text_single_chunk() -> None:
    assert _split_for_telegram("hi") == ["hi"]


def test_split_long_text_on_newlines() -> None:
    text = ("line1\n" * 200) + "last"  # ~1001 chars
    chunks = _split_for_telegram(text, limit=400)
    assert all(len(c) <= 400 for c in chunks)
    assert "".join(chunks) == text  # round-trips


def test_split_no_newlines_hard_cuts() -> None:
    text = "x" * 5000
    chunks = _split_for_telegram(text, limit=4000)
    assert all(len(c) <= 4000 for c in chunks)
    assert sum(len(c) for c in chunks) == 5000
