"""Unit tests for Bandhu FastAPI REST and Webhook Endpoints."""

import pytest
from fastapi.testclient import TestClient
from bandhu.api.app import app

client = TestClient(app)


def test_healthz_endpoint() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "bandhu" in data["service"]


def test_chat_endpoint_text_and_audio() -> None:
    response = client.post(
        "/api/chat",
        json={
            "message": "అమ్మమ్మ నాకు జ్వరంగా ఉంది",
            "speaker_name": "Sai",
            "persona_id": "grandma_chittoor",
            "generate_audio": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "reply_text" in data
    assert len(data["reply_text"]) > 0
    assert data["audio_url"] is not None
    assert len(data["tools_executed"]) >= 1
    assert "thought_stream" in data
    assert data["thought_stream"] is not None


def test_upload_chat_endpoint() -> None:
    sample_chat = """[10/08/24, 08:30:15] అమ్మమ్మ: కన్నా లేచినావా?
[10/08/24, 08:33:10] అమ్మమ్మ: రెండు ఇడ్లీలు తినరా నాయనా."""

    response = client.post(
        "/api/persona/upload-chat",
        data={
            "chat_text": sample_chat,
            "target_speaker": "అమ్మమ్మ",
            "relationship": "Grandmother",
            "language": "Telugu",
            "dialect": "Rayalaseema / Chittoor",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_turns_parsed"] == 2


def test_whatsapp_webhook_twilio_format() -> None:
    response = client.post(
        "/api/webhook/whatsapp",
        data="Body=అమ్మమ్మ బాగుండావా&From=whatsapp:+919876543210",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 200
    assert "<Response>" in response.text
    assert "<Message>" in response.text


def test_proactive_cron_checkin() -> None:
    response = client.post("/api/cron/checkin")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "executed"


def test_scheduled_checkin_is_agent_initiated() -> None:
    # No request body is sent: the trigger is entirely server-side.
    response = client.post("/internal/scheduled-checkin")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "executed"
    assert data["user_initiated"] is False
    assert data["initiator"] == "cloud-scheduler"
    assert data["trigger"] == "proactive morning check-in"
    assert data["memory_id"]


def test_scheduled_checkin_rejects_unknown_persona() -> None:
    response = client.post("/internal/scheduled-checkin", params={"persona": "stranger"})
    assert response.status_code == 400


def test_list_personas() -> None:
    response = client.get("/api/personas")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "personas" in data
    assert len(data["personas"]) >= 1

