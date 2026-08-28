"""Unit tests for WhatsApp chat export parsing."""

from pathlib import Path
import pytest
from bandhu.persona.parser import WhatsAppChatParser


def test_parse_ios_format() -> None:
    raw = """[10/08/24, 08:30:15] అమ్మమ్మ: కన్నా లేచినావా?
[10/08/24, 08:32:00] సాయి: లేచాను అమ్మమ్మ.
[10/08/24, 08:33:10] అమ్మమ్మ: <Media omitted>
[10/08/24, 08:34:00] అమ్మమ్మ: రెండు ఇడ్లీలు తినరా నాయనా."""

    turns = WhatsAppChatParser.parse_text(raw)
    assert len(turns) == 3
    assert turns[0].speaker == "అమ్మమ్మ"
    assert turns[0].text == "కన్నా లేచినావా?"
    assert turns[1].speaker == "సాయి"
    assert turns[2].speaker == "అమ్మమ్మ"
    assert turns[2].text == "రెండు ఇడ్లీలు తినరా నాయనా."


def test_parse_android_format() -> None:
    raw = """10/08/2024, 08:30 - Mom: Did you take your medicines?
10/08/2024, 08:32 - Alex: Yes mom, just took them.
10/08/2024, 08:35 - Mom: Good boy, take care."""

    turns = WhatsAppChatParser.parse_text(raw)
    assert len(turns) == 3
    assert turns[0].speaker == "Mom"
    assert turns[0].text == "Did you take your medicines?"
    assert turns[1].speaker == "Alex"


def test_parse_sample_file() -> None:
    sample_path = Path("data/sample_chats/sample_telugu_grandma_chat.txt")
    turns = WhatsAppChatParser.parse_file(sample_path)
    assert len(turns) == 6
    speakers = WhatsAppChatParser.get_speakers(turns)
    assert "అమ్మమ్మ" in speakers
    assert speakers["అమ్మమ్మ"] == 4
