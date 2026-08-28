"""Unit tests for Persona profile generation and extraction."""

import pytest
from bandhu.persona.extractor import PersonaExtractor
from bandhu.persona.parser import WhatsAppChatParser


@pytest.mark.asyncio
async def test_persona_heuristic_extraction() -> None:
    raw = """[10/08/24, 08:30:15] అమ్మమ్మ: కన్నా లేచినావా? టిఫిన్ తింటివా లేదా?
[10/08/24, 08:33:10] అమ్మమ్మ: ఒట్టి కాఫీ తాగితే కడుపు మండుతాది బా. వేడివేడిగా రెండు ఇడ్లీలు తినరా నాయనా.
[10/08/24, 14:15:22] అమ్మమ్మ: నేను మన ఇంట్లో రాగి సంగటి, నాటుకోడి పులుసు జేస్తిని బా.
[10/08/24, 14:19:05] అమ్మమ్మ: నేను ఘాటుగా మిరియాల కషాయం కాసిస్తా బా, తాగి పడుకో వెంటనే తగ్గిపోతాది. స్వామి దయతో చల్లగా ఉండాలి నాయనా."""

    turns = WhatsAppChatParser.parse_text(raw)
    profile = await PersonaExtractor.extract_profile_from_turns(
        turns=turns,
        target_speaker="అమ్మమ్మ",
        relationship="Grandmother",
        language="Telugu",
        dialect="Rayalaseema / Chittoor",
    )

    assert profile.name == "అమ్మమ్మ"
    assert profile.relationship == "Grandmother"
    assert profile.language == "Telugu"
    assert any("కన్నా" in p for p in profile.pet_names) or any("నాయనా" in p for p in profile.pet_names)
    assert len(profile.frequent_catchphrases) >= 1

    # Test prompt generation
    prompt = profile.generate_system_instruction()
    assert "అమ్మమ్మ" in prompt
    assert len(prompt) >= 20
