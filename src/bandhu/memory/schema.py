"""Schemas for memory records, health logs, emergency alerts, and oral histories."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class MemoryRecord:
    """A long-term contextual memory entry."""

    persona_id: str
    category: str  # health, family, preference, routine, general
    topic: str
    details: str
    importance: int = 1  # 1 (normal) to 3 (critical)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HealthLogRecord:
    """A health symptom or daily wellbeing log."""

    persona_id: str
    speaker_name: str
    mood: str
    health_status: str
    severity: Literal["LOW", "MEDIUM", "CRITICAL"] = "LOW"
    vital_symptoms: list[str] = field(default_factory=list)
    notes: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EmergencyAlertRecord:
    """An emergency alert dispatched to the primary caregiver."""

    persona_id: str
    patient_name: str
    severity: str
    symptoms: str
    dispatched_to: str
    channel: str  # whatsapp, sms, mock
    alert_payload: str
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OralHistoryRecord:
    """An archived oral family story, folklore, or traditional recipe."""

    persona_id: str
    title: str
    category: str  # recipe, folklore, tradition, life_story
    content: str
    audio_url: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
