"""Unified Memory Bank supporting Google Cloud Firestore and SQLite development fallback."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from bandhu.config import settings
from bandhu.memory.schema import (
    EmergencyAlertRecord,
    HealthLogRecord,
    MemoryRecord,
    OralHistoryRecord,
)
from bandhu.persona.models import PersonaProfile


class MemoryStore:
    """Persistent storage engine for personas, memories, health logs, and emergency alerts."""

    def __init__(self, db_path: Path | str | None = None, force_sqlite: bool = False) -> None:
        self.use_firestore = False
        self.firestore_client = None

        if not force_sqlite and not settings.use_sqlite_fallback:
            try:
                from google.cloud import firestore  # type: ignore[import]

                self.firestore_client = firestore.Client(project=settings.gcp_project_id)
                self.use_firestore = True
                print(f"[Firestore] Connected to Google Cloud Firestore (Project: {settings.gcp_project_id})")
            except Exception as exc:
                print(f"[Memory] Firestore connection not available ({exc}), using SQLite local fallback.")
                self.use_firestore = False

        if not self.use_firestore:
            if db_path is None:
                self.db_path = settings.data_dir / "bandhu_store.db"
            else:
                self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        """Create relational SQLite tables if not existing."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS personas (
                    persona_id TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    persona_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    details TEXT NOT NULL,
                    importance INTEGER DEFAULT 1,
                    timestamp TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS health_logs (
                    id TEXT PRIMARY KEY,
                    persona_id TEXT NOT NULL,
                    speaker_name TEXT NOT NULL,
                    mood TEXT NOT NULL,
                    health_status TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    vital_symptoms_json TEXT,
                    notes TEXT,
                    timestamp TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS emergency_alerts (
                    id TEXT PRIMARY KEY,
                    persona_id TEXT NOT NULL,
                    patient_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    symptoms TEXT NOT NULL,
                    dispatched_to TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    alert_payload TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS oral_histories (
                    id TEXT PRIMARY KEY,
                    persona_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    audio_url TEXT,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.commit()

    # --- Persona Operations ---

    def save_persona(self, profile: PersonaProfile) -> PersonaProfile:
        """Save or update persona profile."""
        if self.use_firestore and self.firestore_client:
            doc_ref = self.firestore_client.collection("bandhu_personas").document(profile.persona_id)
            doc_ref.set(profile.to_dict())
            return profile

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO personas (persona_id, data_json, updated_at)
                VALUES (?, ?, ?)
                """,
                (profile.persona_id, json.dumps(profile.to_dict(), ensure_ascii=False), profile.created_at),
            )
            conn.commit()
        return profile

    def get_persona(self, persona_id: str) -> PersonaProfile | None:
        """Retrieve persona profile by ID."""
        if self.use_firestore and self.firestore_client:
            doc = self.firestore_client.collection("bandhu_personas").document(persona_id).get()
            if doc.exists:
                return PersonaProfile.from_dict(doc.to_dict() or {})
            return None

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT data_json FROM personas WHERE persona_id = ?", (persona_id,))
            row = cursor.fetchone()
            if row:
                return PersonaProfile.from_dict(json.loads(row[0]))
        return None

    def list_personas(self) -> list[PersonaProfile]:
        """List all registered personas."""
        if self.use_firestore and self.firestore_client:
            docs = self.firestore_client.collection("bandhu_personas").stream()
            return [PersonaProfile.from_dict(doc.to_dict()) for doc in docs]

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT data_json FROM personas ORDER BY updated_at DESC")
            rows = cursor.fetchall()
            return [PersonaProfile.from_dict(json.loads(row[0])) for row in rows]

    def delete_persona(self, persona_id: str) -> bool:
        """Delete persona profile by ID."""
        if self.use_firestore and self.firestore_client:
            self.firestore_client.collection("bandhu_personas").document(persona_id).delete()
            return True

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM personas WHERE persona_id = ?", (persona_id,))
            conn.commit()
        return True

    # --- Memory Operations ---

    def store_memory(
        self,
        persona_id: str,
        category: str,
        topic: str,
        details: str,
        importance: int = 1,
    ) -> MemoryRecord:
        """Store a long-term personal or family memory."""
        record = MemoryRecord(
            persona_id=persona_id,
            category=category.strip().lower(),
            topic=topic.strip(),
            details=details.strip(),
            importance=importance,
        )

        if self.use_firestore and self.firestore_client:
            self.firestore_client.collection("bandhu_memories").document(record.id).set(record.to_dict())
            return record

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO memories (id, persona_id, category, topic, details, importance, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (record.id, record.persona_id, record.category, record.topic, record.details, record.importance, record.timestamp),
            )
            conn.commit()
        return record

    def recall_memories(
        self, persona_id: str, query: str = "", limit: int = 5
    ) -> list[MemoryRecord]:
        """Recall relevant memories for a specific persona."""
        if self.use_firestore and self.firestore_client:
            query_ref = self.firestore_client.collection("bandhu_memories").where("persona_id", "==", persona_id)
            docs = query_ref.order_by("importance", direction="DESCENDING").limit(limit).stream()
            return [MemoryRecord(**doc.to_dict()) for doc in docs]

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            if query.strip():
                pattern = f"%{query.strip()}%"
                cursor.execute(
                    """
                    SELECT id, persona_id, category, topic, details, importance, timestamp
                    FROM memories
                    WHERE persona_id = ? AND (topic LIKE ? OR details LIKE ? OR category LIKE ?)
                    ORDER BY importance DESC, timestamp DESC
                    LIMIT ?
                    """,
                    (persona_id, pattern, pattern, pattern, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, persona_id, category, topic, details, importance, timestamp
                    FROM memories
                    WHERE persona_id = ?
                    ORDER BY importance DESC, timestamp DESC
                    LIMIT ?
                    """,
                    (persona_id, limit),
                )
            rows = cursor.fetchall()
            return [
                MemoryRecord(
                    id=r[0], persona_id=r[1], category=r[2], topic=r[3], details=r[4], importance=r[5], timestamp=r[6]
                )
                for r in rows
            ]

    # --- Health & Emergency Alert Operations ---

    def log_health(
        self,
        persona_id: str,
        speaker_name: str,
        mood: str,
        health_status: str,
        severity: str = "LOW",
        vital_symptoms: list[str] | None = None,
        notes: str = "",
    ) -> HealthLogRecord:
        """Log health symptoms or daily wellbeing."""
        vitals = vital_symptoms or []
        record = HealthLogRecord(
            persona_id=persona_id,
            speaker_name=speaker_name,
            mood=mood,
            health_status=health_status,
            severity=severity,  # type: ignore[arg-type]
            vital_symptoms=vitals,
            notes=notes,
        )

        if self.use_firestore and self.firestore_client:
            self.firestore_client.collection("bandhu_health_logs").document(record.id).set(record.to_dict())
            return record

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO health_logs (id, persona_id, speaker_name, mood, health_status, severity, vital_symptoms_json, notes, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (record.id, record.persona_id, record.speaker_name, record.mood, record.health_status, record.severity, json.dumps(vitals), record.notes, record.timestamp),
            )
            conn.commit()
        return record

    def record_emergency_alert(
        self,
        persona_id: str,
        patient_name: str,
        severity: str,
        symptoms: str,
        dispatched_to: str,
        channel: str,
        alert_payload: str,
    ) -> EmergencyAlertRecord:
        """Record an alert dispatched to caregivers."""
        record = EmergencyAlertRecord(
            persona_id=persona_id,
            patient_name=patient_name,
            severity=severity,
            symptoms=symptoms,
            dispatched_to=dispatched_to,
            channel=channel,
            alert_payload=alert_payload,
        )

        if self.use_firestore and self.firestore_client:
            self.firestore_client.collection("bandhu_emergency_alerts").document(record.id).set(record.to_dict())
            return record

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO emergency_alerts (id, persona_id, patient_name, severity, symptoms, dispatched_to, channel, alert_payload, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (record.id, record.persona_id, record.patient_name, record.severity, record.symptoms, record.dispatched_to, record.channel, record.alert_payload, record.timestamp),
            )
            conn.commit()
        return record

    def list_health_alerts(self, persona_id: str, limit: int = 10) -> list[EmergencyAlertRecord]:
        """List past emergency alerts."""
        if self.use_firestore and self.firestore_client:
            docs = self.firestore_client.collection("bandhu_emergency_alerts").where("persona_id", "==", persona_id).order_by("timestamp", direction="DESCENDING").limit(limit).stream()
            return [EmergencyAlertRecord(**doc.to_dict()) for doc in docs]

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, persona_id, patient_name, severity, symptoms, dispatched_to, channel, alert_payload, timestamp
                FROM emergency_alerts
                WHERE persona_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (persona_id, limit),
            )
            rows = cursor.fetchall()
            return [
                EmergencyAlertRecord(
                    id=r[0], persona_id=r[1], patient_name=r[2], severity=r[3], symptoms=r[4], dispatched_to=r[5], channel=r[6], alert_payload=r[7], timestamp=r[8]
                )
                for r in rows
            ]

    # --- Oral History Archival ---

    def archive_oral_history(
        self,
        persona_id: str,
        title: str,
        category: str,
        content: str,
        audio_url: str = "",
    ) -> OralHistoryRecord:
        """Archive a family folklore, recipe, or life story."""
        record = OralHistoryRecord(
            persona_id=persona_id,
            title=title,
            category=category,
            content=content,
            audio_url=audio_url,
        )

        if self.use_firestore and self.firestore_client:
            self.firestore_client.collection("bandhu_oral_histories").document(record.id).set(record.to_dict())
            return record

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO oral_histories (id, persona_id, title, category, content, audio_url, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (record.id, record.persona_id, record.title, record.category, record.content, record.audio_url, record.timestamp),
            )
            conn.commit()
        return record
