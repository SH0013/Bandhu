"""Unit tests for Bandhu Gemini Agent & Dynamic Persona Engine."""

import gc
from pathlib import Path
import pytest
from bandhu.agent.gemini_agent import BandhuGeminiAgent
from bandhu.memory.store import MemoryStore
from bandhu.persona.models import PersonaProfile


@pytest.fixture
def agent(tmp_path: Path) -> BandhuGeminiAgent:
    db_path = tmp_path / "agent_test.db"
    store = MemoryStore(db_path=db_path, force_sqlite=True)
    ag = BandhuGeminiAgent(memory_store=store)
    yield ag
    del ag
    del store
    gc.collect()


@pytest.mark.asyncio
async def test_agent_default_grandma_persona(agent: BandhuGeminiAgent) -> None:
    assert agent.persona.persona_id == "grandma_chittoor"
    assert "అమ్మమ్మ" in agent.persona.name


@pytest.mark.asyncio
async def test_agent_health_concern_turn(agent: BandhuGeminiAgent) -> None:
    response = await agent.reply("అమ్మమ్మ నాకు విపరీతమైన జ్వరం, తలనొప్పిగా ఉంది", speaker_name="Sai")
    assert len(response.reply_text) > 0
    assert any(kw in response.reply_text for kw in ("మిరియాల కషాయం", "కషాయం", "జ్వరం", "తలనొప్పి", "మిరియాలు", "బాగులేదా", "నొప్పులు", "అయ్యో"))
    assert len(response.tools_executed) >= 1

    tool_names = [t["name"] for t in response.tools_executed]
    assert "analyze_and_dispatch_health_alert" in tool_names

    # Check alert logged in memory
    alerts = agent.memory_store.list_health_alerts("grandma_chittoor")
    assert len(alerts) >= 1


@pytest.mark.asyncio
async def test_agent_dynamic_persona_switch(agent: BandhuGeminiAgent) -> None:
    new_profile = PersonaProfile(
        persona_id="mother_telugu",
        name="అమ్మ (Amma)",
        relationship="Mother",
        language="Telugu",
        pet_names=["బంగారం"],
        frequent_catchphrases=["జాగ్రత్త"],
    )
    agent.set_persona(new_profile)
    assert agent.persona.persona_id == "mother_telugu"

    response = await agent.reply("హలో అమ్మ, ఎలా ఉన్నావు?", speaker_name="Sai")
    assert len(response.reply_text) > 0
    assert response.persona_id == "mother_telugu"
    assert response.thought_stream is not None
    assert "Gemini 3.7 Flash" in response.thought_stream


@pytest.mark.asyncio
async def test_agent_thought_stream(agent: BandhuGeminiAgent) -> None:
    response = await agent.reply("నాకు కాస్త తలనొప్పిగా ఉంది", speaker_name="TestUser")
    assert response.thought_stream is not None
    assert len(response.thought_stream) > 10
    assert "Reasoning Chain" in response.thought_stream

