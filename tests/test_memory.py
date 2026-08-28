"""Unit tests for Bandhu Unified Memory Store."""

import gc
import tempfile
from pathlib import Path
import pytest
from bandhu.memory.store import MemoryStore
from bandhu.persona.models import PersonaProfile


@pytest.fixture
def temp_store(tmp_path: Path) -> MemoryStore:
    db_path = tmp_path / "test_store.db"
    store = MemoryStore(db_path=db_path, force_sqlite=True)
    yield store
    del store
    gc.collect()


def test_persona_crud(temp_store: MemoryStore) -> None:
    profile = PersonaProfile(
        persona_id="test_grandma",
        name="అమ్మమ్మ",
        relationship="Grandmother",
        language="Telugu",
        frequent_catchphrases=["బా", "నాయనా"],
    )
    saved = temp_store.save_persona(profile)
    assert saved.persona_id == "test_grandma"

    retrieved = temp_store.get_persona("test_grandma")
    assert retrieved is not None
    assert retrieved.name == "అమ్మమ్మ"
    assert "బా" in retrieved.frequent_catchphrases

    all_personas = temp_store.list_personas()
    assert len(all_personas) == 1


def test_memory_storage_and_recall(temp_store: MemoryStore) -> None:
    temp_store.store_memory(
        persona_id="test_grandma",
        category="health",
        topic="Fever history",
        details="Grandchild reported fever on Sunday",
        importance=2,
    )
    temp_store.store_memory(
        persona_id="test_grandma",
        category="preference",
        topic="Favorite dish",
        details="Loves Ragi Sangati with chicken pulusu",
        importance=1,
    )

    recalled = temp_store.recall_memories("test_grandma", query="Fever")
    assert len(recalled) == 1
    assert recalled[0].topic == "Fever history"

    all_mem = temp_store.recall_memories("test_grandma")
    assert len(all_mem) == 2


def test_health_logging_and_alert_dispatch(temp_store: MemoryStore) -> None:
    log = temp_store.log_health(
        persona_id="test_grandma",
        speaker_name="Sai",
        mood="tired",
        health_status="103 fever and severe shivering",
        severity="CRITICAL",
        vital_symptoms=["high_fever", "shivering"],
        notes="Urgent care needed",
    )
    assert log.severity == "CRITICAL"

    alert = temp_store.record_emergency_alert(
        persona_id="test_grandma",
        patient_name="Sai",
        severity="CRITICAL",
        symptoms="103 fever",
        dispatched_to="+919876543210",
        channel="whatsapp",
        alert_payload="EMERGENCY ALERT: Sai reported 103 fever.",
    )
    assert alert.severity == "CRITICAL"

    alerts = temp_store.list_health_alerts("test_grandma")
    assert len(alerts) == 1
    assert alerts[0].patient_name == "Sai"
