"""Autonomous Agentic Tools for Bandhu Platform."""

from __future__ import annotations

import inspect
import json
import os
from datetime import datetime
from typing import Any, Callable

from bandhu.config import settings
from bandhu.memory.store import MemoryStore

# Rayalaseema & Traditional Home Remedies & Recipes Knowledge Base
CULTURAL_REMEDIES: dict[str, dict[str, str]] = {
    "kashayam": {
        "name": "అమ్మమ్మ మిరియాల కషాయం (Grandma's Pepper Kashayam)",
        "benefit": "జలుబు, దగ్గు, గొంతు నొప్పి, జ్వరం వెంటనే ఉపశమిస్తాయి.",
        "ingredients": "మిరియాలు (1 చెంచా), తులసి ఆకులు (10), శొంఠి ముక్క, జీలకర్ర, బెల్లం ముక్క, నీళ్ళు (2 గ్లాసులు).",
        "preparation": "మిరియాలు, శొంఠి దంచి నీటిలో వేసి సగానికి మరిగించి, బెల్లం, తులసి వేసి వడకట్టి వేడిగా తాగాలి.",
    },
    "ragi_sangati": {
        "name": "రాయలసీమ రాగి సంగటి (Ragi Sangati)",
        "benefit": "శరీరానికి అపారమైన బలాన్ని, చలువను, రోగనిరోధక శక్తిని ఇస్తుంది.",
        "ingredients": "రాగి పిండి (1 కప్పు), బియ్యం (1/2 కప్పు), నీళ్ళు (3 కప్పులు), ఉప్పు, నెయ్యి.",
        "preparation": "బియ్యం బాగా మెత్తగా ఉడికించి, రాగి పిండిని పైన చల్లి ఆవిరి మీద ఉడికించి సంగటి ముద్దలు చేసుకోవాలి.",
    },
    "natukodi_pulusu": {
        "name": "నాటుకోడి పులుసు (Natukodi Pulusu)",
        "benefit": "జలుబు, ఒంటి నొప్పులు ఉన్నప్పుడు ఘాటైన మసాలాతో ఉపశమనం ఇస్తుంది.",
        "ingredients": "నాటుకోడి మాంసం, ధనియాలు, ఎండుమిర్చి, దాల్చినచెక్క, లవంగాలు, ఉల్లిపాయలు.",
        "preparation": "మసాలాను రోట్లో నూరి, నూనెలో వేయించి, ముక్కలు వేసి పులుసు చిక్కబడేవరకు మరిగించాలి.",
    },
    "pachi_pulusu": {
        "name": "రాయలసీమ పచ్చి పులుసు (Pachi Pulusu)",
        "benefit": "వేసవిలో వేడి తగ్గించి, జీర్ణశక్తిని పెంచుతుంది.",
        "ingredients": "చింతపండు రసం, కాల్చిన ఉల్లిపాయ, పచ్చిమిర్చి, బెల్లం, జీలకర్ర.",
        "preparation": "చింతపండు రసంలో కాల్చిన ఉల్లిపాయ, పచ్చిమిర్చి పిసికి, పచ్చి నూనె తాలింపు పెట్టాలి.",
    },
    "gongura_pachadi": {
        "name": "గోంగూర రోటి పచ్చడి (Gongura Pachadi)",
        "benefit": "ఐరన్, విటమిన్ సి సమృద్ధిగా అందిస్తుంది.",
        "ingredients": "పుల్ల గోంగూర, ఎండుమిర్చి, ధనియాలు, వెల్లుల్లి.",
        "preparation": "గోంగూరను మగ్గించి రోట్లో కచ్చాపచ్చాగా రుబ్బి వెల్లుల్లి తాలింపు వేయాలి.",
    },
}


class AgentToolsRegistry:
    """Manages declarations and execution of autonomous proactive tools."""

    def __init__(self, memory_store: MemoryStore | None = None) -> None:
        self.memory_store = memory_store or MemoryStore()
        self.active_persona_id: str = "default_persona"

    def set_active_persona(self, persona_id: str) -> None:
        """Set active persona context for tool operations."""
        self.active_persona_id = persona_id

    def get_tool_declarations(self) -> list[dict[str, Any]]:
        """Return Gemini / OpenAPI compatible tool declarations."""
        return [
            {
                "name": "analyze_and_dispatch_health_alert",
                "description": "Analyze health symptoms, compute medical severity (LOW/MEDIUM/CRITICAL), log vitals, and autonomously dispatch emergency WhatsApp/SMS alert to caregiver if necessary.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "patient_name": {"type": "STRING", "description": "Name of the person unwell"},
                        "symptoms": {"type": "STRING", "description": "Specific symptoms reported (e.g. 102 fever, chest pain, headache)"},
                        "mood": {"type": "STRING", "description": "Current mood (e.g. anxious, tired, distressed)"},
                    },
                    "required": ["patient_name", "symptoms"],
                },
            },
            {
                "name": "schedule_proactive_followup",
                "description": "Schedule an automated proactive check-in or medication reminder for the user after a specified duration.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "task_description": {"type": "STRING", "description": "Description of the reminder (e.g. 'Check if fever broke', 'Take BP tablet')"},
                        "delay_hours": {"type": "NUMBER", "description": "Hours after which to execute check-in (e.g. 2, 4, 12)"},
                    },
                    "required": ["task_description", "delay_hours"],
                },
            },
            {
                "name": "lookup_cultural_remedy",
                "description": "Search traditional Rayalaseema home remedies and heritage dishes (Kashayam for cold/fever, Ragi Sangati, Pachi Pulusu, Natukodi, Gongura).",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "query": {"type": "STRING", "description": "Symptom or recipe keyword (e.g. 'fever', 'cold', 'kashayam', 'sangati', 'chicken')"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "remember_fact",
                "description": "Store a long-term personal, family, routine, or medical detail for future recall.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "category": {"type": "STRING", "description": "Category (health, family, preference, routine)"},
                        "topic": {"type": "STRING", "description": "Short topic summary"},
                        "details": {"type": "STRING", "description": "Detailed information to remember"},
                    },
                    "required": ["topic", "details"],
                },
            },
            {
                "name": "recall_facts",
                "description": "Recall past personal memories, events, preferences, or medical history.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "query": {"type": "STRING", "description": "Keyword to search memories (e.g. 'fever', 'job', 'food')"},
                    },
                },
            },
            {
                "name": "archive_oral_history",
                "description": "Archive a spoken family story, traditional cooking recipe, or life wisdom into the digital family vault.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "title": {"type": "STRING", "description": "Title of the story or recipe"},
                        "category": {"type": "STRING", "description": "Category (recipe, folklore, tradition, life_story)"},
                        "content": {"type": "STRING", "description": "Full narrative or recipe instructions"},
                    },
                    "required": ["title", "category", "content"],
                },
            },
            {
                "name": "set_care_reminder",
                "description": "Set a routine or health reminder (e.g. taking medication, drinking water, eating meals on time, doctor appointments, evening walks).",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "reminder_type": {"type": "STRING", "description": "Type: medication, hydration, meal, appointment, walk, general"},
                        "title": {"type": "STRING", "description": "Short title of reminder (e.g. 'Morning BP Tablet', 'Afternoon Meal')"},
                        "schedule_time": {"type": "STRING", "description": "When to remind (e.g. 'after lunch', '8:00 PM', 'tomorrow morning')"},
                        "notes": {"type": "STRING", "description": "Additional affectionate instructions or notes"},
                    },
                    "required": ["reminder_type", "title", "schedule_time"],
                },
            },
            {
                "name": "record_wellness_checkup",
                "description": "Log a daily health/wellness checkup (sleep quality, pain levels, appetite, mood, or vitals like BP and temperature).",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "checkup_summary": {"type": "STRING", "description": "Overall summary of how user feels today"},
                        "vitals_logged": {"type": "STRING", "description": "Any specific vitals mentioned (e.g. BP 120/80, 98.6 temp, slept 7 hours)"},
                        "comfort_level": {"type": "STRING", "description": "Rating: excellent, normal, tired, uncomfortable, in_pain"},
                    },
                    "required": ["checkup_summary"],
                },
            },
        ]

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch and execute tool call."""
        tools_map: dict[str, Callable[..., dict[str, Any]]] = {
            "analyze_and_dispatch_health_alert": self._tool_analyze_health_alert,
            "schedule_proactive_followup": self._tool_schedule_followup,
            "lookup_cultural_remedy": self._tool_lookup_remedy,
            "remember_fact": self._tool_remember_fact,
            "recall_facts": self._tool_recall_facts,
            "archive_oral_history": self._tool_archive_oral_history,
            "set_care_reminder": self._tool_set_care_reminder,
            "record_wellness_checkup": self._tool_record_wellness_checkup,
        }

        if name not in tools_map:
            return {"error": f"Unknown tool: '{name}'"}

        try:
            return tools_map[name](**arguments)
        except Exception as exc:
            return {"error": f"Tool execution failure in '{name}': {exc}"}

    def _tool_analyze_health_alert(
        self, patient_name: str, symptoms: str, mood: str = "concerned"
    ) -> dict[str, Any]:
        """Compute severity, log to memory store, and trigger caregiver dispatch if needed."""
        sym_lower = symptoms.lower()

        critical_markers = ["103", "104", "102", "chest pain", "breathing", "unconscious", "bleeding", "severe pain", "గుండె", "శ్వాస"]
        medium_markers = ["fever", "headache", "vomiting", "shivering", "జ్వరం", "తలనొప్పి", "వాంతులు", "నొప్పులు", "కడుపునొప్పి"]

        if any(m in sym_lower for m in critical_markers):
            severity = "CRITICAL"
        elif any(m in sym_lower for m in medium_markers):
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # Log health record
        health_record = self.memory_store.log_health(
            persona_id=self.active_persona_id,
            speaker_name=patient_name,
            mood=mood,
            health_status=symptoms,
            severity=severity,
            vital_symptoms=[symptoms],
            notes="Logged by autonomous health triage agent",
        )

        alert_dispatched = False
        alert_payload = ""

        if severity in ("MEDIUM", "CRITICAL"):
            alert_payload = (
                f"🚨 [BANDHU CAREGIVER ALERT - {severity}]\n"
                f"Patient: {patient_name}\n"
                f"Symptoms: {symptoms}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                f"Action Recommended: Immediate caregiver follow-up advised."
            )
            self.memory_store.record_emergency_alert(
                persona_id=self.active_persona_id,
                patient_name=patient_name,
                severity=severity,
                symptoms=symptoms,
                dispatched_to=settings.caregiver_phone_number,
                channel="whatsapp",
                alert_payload=alert_payload,
            )
            alert_dispatched = True

            dispatch_result = self._dispatch_caregiver_alert(alert_payload)
            alert_payload = f"{alert_payload}\nDispatch: {dispatch_result}"

        return {
            "status": "success",
            "severity": severity,
            "patient_name": patient_name,
            "symptoms": symptoms,
            "alert_dispatched": alert_dispatched,
            "dispatched_to": settings.caregiver_phone_number if alert_dispatched else None,
            "alert_message": alert_payload if alert_dispatched else "Symptom severity is low; monitoring locally.",
        }

    def _dispatch_caregiver_alert(self, alert_payload: str) -> str:
        """Dispatch caregiver alert via Telegram (free), WhatsApp Cloud API (free tier), or GCP Cloud Tasks.

        Order of preference (all free / freemium):
          1. Telegram Bot API — completely free, no per-message cost, 30 msg/sec limit.
          2. Meta WhatsApp Cloud API (direct) — 1,000 service conversations/month free.
          3. Google Cloud Tasks — durable delay queue for retry; not a notification channel.

        Falls through silently if no channel is configured (returns "not_dispatched").
        """
        result = "not_dispatched"

        # ── 1) Telegram (primary, free, recommended) ──────────────────────────
        if settings.telegram_bot_token and settings.caregiver_telegram_chat_id:
            try:
                import httpx  # already in requirements.txt
                url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
                # Telegram's max message length is 4096 chars; truncate safely
                text = alert_payload if len(alert_payload) <= 4000 else alert_payload[:3990] + "\n…"
                resp = httpx.post(
                    url,
                    json={
                        "chat_id": settings.caregiver_telegram_chat_id,
                        "text": text,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return f"telegram:{data.get('result', {}).get('message_id', 'ok')}"
                result = f"telegram_failed:{resp.status_code}:{resp.text[:200]}"
                return result
            except Exception as exc:
                result = f"telegram_failed:{exc}"
                return result  # don't fall through if Telegram is configured but failed

        # ── 2) Meta WhatsApp Cloud API (direct, free tier) ────────────────────
        if (settings.whatsapp_phone_number_id
                and settings.whatsapp_access_token
                and settings.caregiver_phone_number):
            try:
                import httpx
                url = (
                    f"https://graph.facebook.com/v20.0/"
                    f"{settings.whatsapp_phone_number_id}/messages"
                )
                headers = {
                    "Authorization": f"Bearer {settings.whatsapp_access_token}",
                    "Content-Type": "application/json",
                }
                # Normalize "whatsapp:+91…" or "+91…" to bare "91…"
                to = settings.caregiver_phone_number.replace("whatsapp:", "").replace("+", "").strip()
                payload = {
                    "messaging_product": "whatsapp",
                    "to": to,
                    "type": "text",
                    "text": {"body": alert_payload[:4000]},
                }
                resp = httpx.post(url, headers=headers, json=payload, timeout=10.0)
                if resp.status_code in (200, 201):
                    return f"whatsapp_cloud:{resp.json().get('messages', [{}])[0].get('id', 'ok')}"
                result = f"whatsapp_cloud_failed:{resp.status_code}:{resp.text[:200]}"
                return result
            except Exception as exc:
                result = f"whatsapp_cloud_failed:{exc}"
                return result

        # ── 3) Google Cloud Tasks (durable retry queue) ───────────────────────
        if settings.gcp_project_id:
            try:
                from google.cloud import tasks_v2  # type: ignore[import]
                client = tasks_v2.CloudTasksClient()
                parent = client.queue_path(
                    settings.gcp_project_id,
                    os.getenv("GCP_SCHEDULER_LOCATION", "us-central1"),
                    os.getenv("BANDHU_TASK_QUEUE", "bandhu-care-queue"),
                )
                task = {
                    "http_request": {
                        "http_method": tasks_v2.HttpMethod.POST,
                        "url": os.getenv(
                            "BANDHU_DISPATCH_URL",
                            os.getenv("BANDHU_CLOUD_RUN_URL", "http://localhost:8080") + "/api/webhook/dispatch-caregiver",
                        ),
                        "headers": {"Content-Type": "application/json"},
                        "body": alert_payload.encode(),
                    }
                }
                resp = client.create_task(parent=parent, task=task)
                result = f"gcp_tasks:{resp.name}"
                return result
            except Exception as exc:
                result = f"gcp_tasks_failed:{exc}"

        return result

    def _tool_schedule_followup(
        self, task_description: str, delay_hours: float
    ) -> dict[str, Any]:
        """Register proactive check-in timer and schedule Cloud Scheduler task."""
        entry = self.memory_store.store_memory(
            persona_id=self.active_persona_id,
            category="routine",
            topic=f"Scheduled Followup: {task_description}",
            details=f"Scheduled check-in in {delay_hours} hours",
            importance=2,
        )

        schedule_result = "memory_only"
        if settings.gcp_project_id and delay_hours > 0:
            try:
                schedule_result = self._schedule_cloud_checkin(
                    task_description=task_description,
                    delay_hours=delay_hours,
                )
            except Exception as exc:
                schedule_result = f"scheduler_failed:{exc}"

        return {
            "status": "scheduled",
            "task": task_description,
            "delay_hours": delay_hours,
            "memory_id": entry.id,
            "schedule": schedule_result,
            "message": f"Autonomous check-in registered: '{task_description}' in {delay_hours}h",
        }

    def _schedule_cloud_checkin(self, task_description: str, delay_hours: float) -> str:
        """Create a delayed Cloud Tasks entry that triggers the proactive check-in webhook."""
        from google.cloud import tasks_v2  # type: ignore[import]
        from datetime import datetime, timedelta, timezone

        client = tasks_v2.CloudTasksClient()
        parent = client.queue_path(
            settings.gcp_project_id,
            os.getenv("GCP_SCHEDULER_LOCATION", "us-central1"),
            os.getenv("BANDHU_TASK_QUEUE", "bandhu-care-queue"),
        )
        scheduled_ts = datetime.now(timezone.utc) + timedelta(hours=delay_hours)
        payload = {
            "persona_id": self.active_persona_id,
            "task": task_description,
            "scheduled_at": scheduled_ts.isoformat(),
        }

        task = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": os.getenv(
                    "BANDHU_DISPATCH_URL",
                    os.getenv("BANDHU_CLOUD_RUN_URL", "http://localhost:8080") + "/api/cron/checkin",
                ),
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(payload).encode(),
                "schedule_time": scheduled_ts,
            }
        }
        resp = client.create_task(parent=parent, task=task)
        return f"gcp_tasks:{resp.name}"

    def _tool_lookup_remedy(self, query: str) -> dict[str, Any]:
        """Lookup traditional remedy from knowledge base."""
        q_norm = query.lower()
        for key, remedy in CULTURAL_REMEDIES.items():
            if (
                key in q_norm
                or ("fever" in q_norm and key == "kashayam")
                or ("cold" in q_norm and key == "kashayam")
                or ("cough" in q_norm and key == "kashayam")
                or ("జ్వరం" in q_norm and key == "kashayam")
                or ("జలుబు" in q_norm and key == "kashayam")
                or ("sangati" in q_norm and key == "ragi_sangati")
                or ("సంగటి" in q_norm and key == "ragi_sangati")
                or ("chicken" in q_norm and key == "natukodi_pulusu")
            ):
                return {"status": "found", "remedy": remedy}

        return {"status": "found", "remedy": CULTURAL_REMEDIES["kashayam"]}

    def _tool_remember_fact(
        self, topic: str, details: str, category: str = "general"
    ) -> dict[str, Any]:
        record = self.memory_store.store_memory(
            persona_id=self.active_persona_id,
            category=category,
            topic=topic,
            details=details,
        )
        return {"status": "success", "id": record.id, "topic": topic}

    def _tool_recall_facts(self, query: str = "") -> dict[str, Any]:
        records = self.memory_store.recall_memories(
            persona_id=self.active_persona_id, query=query, limit=5
        )
        return {
            "query": query,
            "count": len(records),
            "memories": [r.to_dict() for r in records],
        }

    def _tool_archive_oral_history(
        self, title: str, category: str, content: str
    ) -> dict[str, Any]:
        record = self.memory_store.archive_oral_history(
            persona_id=self.active_persona_id,
            title=title,
            category=category,
            content=content,
        )
        return {"status": "archived", "id": record.id, "title": title}

    def _tool_set_care_reminder(
        self, reminder_type: str, title: str, schedule_time: str, notes: str = ""
    ) -> dict[str, Any]:
        """Record and schedule a caring health, medication, or routine reminder."""
        entry = self.memory_store.store_memory(
            persona_id=self.active_persona_id,
            category="reminder",
            topic=f"[{reminder_type.upper()}] {title} @ {schedule_time}",
            details=f"Reminder scheduled: {title} at {schedule_time}. Notes: {notes}",
            importance=2,
        )
        return {
            "status": "reminder_set",
            "reminder_type": reminder_type,
            "title": title,
            "schedule_time": schedule_time,
            "memory_id": entry.id,
            "message": f"Care reminder set: '{title}' ({schedule_time})",
        }

    def _tool_record_wellness_checkup(
        self, checkup_summary: str, vitals_logged: str = "", comfort_level: str = "normal"
    ) -> dict[str, Any]:
        """Record a routine wellness checkup and log vitals to memory store."""
        health_record = self.memory_store.log_health(
            persona_id=self.active_persona_id,
            speaker_name="User",
            mood=comfort_level,
            health_status=checkup_summary,
            severity="LOW" if comfort_level in ("excellent", "normal") else "MEDIUM",
            vital_symptoms=[vitals_logged] if vitals_logged else [],
            notes=f"Wellness checkup: {checkup_summary}. Comfort: {comfort_level}",
        )
        return {
            "status": "checkup_recorded",
            "summary": checkup_summary,
            "comfort_level": comfort_level,
            "log_id": health_record.id,
            "message": f"Daily wellness checkup logged: {checkup_summary}",
        }

