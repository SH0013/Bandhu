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


def test_list_personas() -> None:
    response = client.get("/api/personas")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "personas" in data
    assert len(data["personas"]) >= 1

