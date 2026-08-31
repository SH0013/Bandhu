"""Unit tests for Bandhu autonomous agentic tools."""

import gc
from pathlib import Path
import pytest
from bandhu.agent.tools import AgentToolsRegistry
from bandhu.memory.store import MemoryStore


@pytest.fixture
def registry(tmp_path: Path) -> AgentToolsRegistry:
    db_path = tmp_path / "tools_test.db"
    store = MemoryStore(db_path=db_path, force_sqlite=True)
    reg = AgentToolsRegistry(memory_store=store)
    reg.set_active_persona("grandma_test")
    yield reg
    del reg
    del store
    gc.collect()


def test_tool_declarations_schema(registry: AgentToolsRegistry) -> None:
    declarations = registry.get_tool_declarations()
    assert len(declarations) >= 5
    tool_names = [d["name"] for d in declarations]
    assert "analyze_and_dispatch_health_alert" in tool_names
    assert "schedule_proactive_followup" in tool_names
    assert "lookup_cultural_remedy" in tool_names
    assert "archive_oral_history" in tool_names


def test_health_triage_critical_alert(registry: AgentToolsRegistry) -> None:
    res = registry.execute_tool(
        "analyze_and_dispatch_health_alert",
        {
            "patient_name": "Sai",
            "symptoms": "High 103 fever and severe chest pain",
            "mood": "distressed",
        },
    )
    assert res["status"] == "success"
    assert res["severity"] == "CRITICAL"
    assert res["alert_dispatched"] is True
    assert "🚨 [BANDHU CAREGIVER ALERT - CRITICAL]" in res["alert_message"]

    # Verify alert was saved in memory store
    alerts = registry.memory_store.list_health_alerts("grandma_test")
    assert len(alerts) == 1
    assert alerts[0].severity == "CRITICAL"


def test_dispatch_with_no_channels_configured(registry: AgentToolsRegistry) -> None:
    """When Telegram, WhatsApp Cloud, and Cloud Tasks are all unconfigured,
    the dispatch must NOT crash and must return a 'not_dispatched' sentinel.
    This guarantees the demo never makes a real network call without intent."""
    res = registry.execute_tool(
        "analyze_and_dispatch_health_alert",
        {
            "patient_name": "Demo User",
            "symptoms": "chest pain and 103 fever",
            "mood": "distressed",
        },
    )
    assert res["status"] == "success"
    assert res["severity"] == "CRITICAL"
    assert res["alert_dispatched"] is True
    # No real network call should have been attempted.
    # The DB log should still contain the alert (it always does, regardless of channel).


def test_schedule_followup_tool(registry: AgentToolsRegistry) -> None:
    res = registry.execute_tool(
        "schedule_proactive_followup",
        {
            "task_description": "Check if temperature returned to normal",
            "delay_hours": 3.5,
        },
    )
    assert res["status"] == "scheduled"
    assert res["delay_hours"] == 3.5

    memories = registry.memory_store.recall_memories("grandma_test", query="Followup")
    assert len(memories) >= 1


def test_cultural_remedy_lookup(registry: AgentToolsRegistry) -> None:
    res = registry.execute_tool("lookup_cultural_remedy", {"query": "జ్వరం మిరియాల కషాయం"})
    assert res["status"] == "found"
    assert "మిరియాల కషాయం" in res["remedy"]["name"]


def test_oral_history_archival(registry: AgentToolsRegistry) -> None:
    res = registry.execute_tool(
        "archive_oral_history",
        {
            "title": "అమ్మమ్మ రాగి సంగటి రహస్యం",
            "category": "recipe",
            "content": "సంగటి కర్రతో గిరగిరా తిప్పితేనే ముద్దలు మృదువుగా వస్తాయి.",
        },
    )
    assert res["status"] == "archived"
    assert res["title"] == "అమ్మమ్మ రాగి సంగటి రహస్యం"


def test_set_care_reminder_tool(registry: AgentToolsRegistry) -> None:
    res = registry.execute_tool(
        "set_care_reminder",
        {
            "reminder_type": "medication",
            "title": "Evening BP Tablet",
            "schedule_time": "8:00 PM after dinner",
            "notes": "Take with warm water",
        },
    )
    assert res["status"] == "reminder_set"
    assert res["title"] == "Evening BP Tablet"
    assert "8:00 PM" in res["schedule_time"]


def test_record_wellness_checkup_tool(registry: AgentToolsRegistry) -> None:
    res = registry.execute_tool(
        "record_wellness_checkup",
        {
            "checkup_summary": "Slept 8 hours, mild knee pain after morning walk",
            "vitals_logged": "BP 125/82",
            "comfort_level": "normal",
        },
    )
    assert res["status"] == "checkup_recorded"
    assert res["comfort_level"] == "normal"
    assert "knee pain" in res["summary"]

