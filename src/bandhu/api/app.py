"""FastAPI Production Server for Bandhu Universal Agent Platform."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import uuid

import httpx
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from bandhu.agent.gemini_agent import BandhuGeminiAgent
from bandhu.api.chat_channels import (
    OutboundMessage,
    maybe_handle_command,
    parse_telegram_update,
    parse_twilio_form,
    parse_whatsapp_payload,
    send_message,
)
from bandhu.audio.stt import SpeechRecognizer
from bandhu.audio.tts import AdaptiveVoiceSynthesizer
from bandhu.audio.voice_manager import VoiceProfileManager
from bandhu.config import settings
from bandhu.memory.store import MemoryStore
from bandhu.persona.extractor import PersonaExtractor
from bandhu.persona.models import PersonaProfile
from bandhu.persona.parser import WhatsAppChatParser

app = FastAPI(
    title="Bandhu (బంధు) Agent Platform",
    description="Universal Voice Persona & Autonomous Proactive Agent Platform for Loved Ones",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Singletons
memory_store = MemoryStore()
voice_manager = VoiceProfileManager()
voice_synthesizer = AdaptiveVoiceSynthesizer(voice_manager=voice_manager)
speech_recognizer = SpeechRecognizer()

# Seed default personas (Pappa & Grandma) into MemoryStore
def _seed_personas() -> None:
    from bandhu.agent.prompts import CHITTOOR_GRANDMA_SYSTEM_PROMPT, HANUMAKONDA_PAPPA_SYSTEM_PROMPT

    pappa = PersonaProfile(
        persona_id="pappa",
        name="పప్పా (Pappa)",
        relationship="Father",
        language="Telugu",
        dialect_region="Telangana / Hanumakonda",
        tone="Deeply loving, caring father, playful & teasing, authentic WhatsApp conversational style",
        frequent_catchphrases=["లేచినవా నానమ్మ", "Don't worry బేటా", "అంతా మన మంచికే", "అన్నం తిన్నావా"],
        pet_names=["నానమ్మ", "డాడీ", "బేటా", "దెయ్యం"],
        key_topics=["ఆరోగ్యం & భోజనం", "కెరీర్ & ఉద్యోగం", "యోగక్షేమాలు"],
        voice_profile_id="pappa",
        custom_system_prompt=HANUMAKONDA_PAPPA_SYSTEM_PROMPT,
    )
    memory_store.save_persona(pappa)

    grandma = PersonaProfile(
        persona_id="grandma_chittoor",
        name="అమ్మమ్మ (Grandma)",
        relationship="Grandmother",
        language="Telugu",
        dialect_region="Rayalaseema / Chittoor",
        tone="Loving, traditional, maternal Rayalaseema dialect",
        frequent_catchphrases=["నాయనా", "తింటివా", "స్వామి దయతో చల్లగా ఉండాలి", "బాగుండావా మా Demo Lakshmi", "బాగుండావా మా Demo Priya", "బాగుండావా మా Demo Anjali", "బాగుండావా మా Demo Sita"],
        pet_names=["నాయనా", "తల్లీ", "మా Demo Lakshmi", "మా Demo Priya", "మా Demo Anjali", "మా Demo Sita", "మా Demo Radha", "మా Demo Devi"],
        key_topics=["ఆరోగ్యం (Health)", "భోజనం (Meals)", "యోగక్షేమాలు (Wellbeing)", "కుటుంబ జ్ఞాపకాలు", "కూతుళ్లు & మనవరాళ్లు"],
        voice_profile_id="grandma_chittoor",
        custom_system_prompt=CHITTOOR_GRANDMA_SYSTEM_PROMPT,
    )
    memory_store.save_persona(grandma)

_seed_personas()

# Initialize agent with Pappa persona as default
default_pappa = memory_store.get_persona("pappa")
gemini_agent = BandhuGeminiAgent(persona_profile=default_pappa, memory_store=memory_store)

# Ensure static directories
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
audio_output_dir = settings.data_dir / "output_audio"
audio_output_dir.mkdir(parents=True, exist_ok=True)

# Mount static files (logo, assets)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# --- Request & Response Models ---

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message text")
    speaker_name: str = Field(default="Grandchild", description="Name of the user speaking")
    persona_id: str = Field(default="pappa", description="Target persona profile ID")
    generate_audio: bool = Field(default=True, description="Whether to synthesize voice audio")


class PersonaUploadRequest(BaseModel):
    chat_text: str = Field(..., description="Raw text of WhatsApp chat export")
    target_speaker: str = Field(..., description="Name of the person to clone")
    relationship: str = Field(default="Grandmother", description="Relationship (e.g. Grandmother, Mother, Father, Friend)")
    language: str = Field(default="Telugu", description="Language of communication")
    dialect: str = Field(default="Rayalaseema / Chittoor", description="Dialect region")


class SavePersonaRequest(BaseModel):
    persona_id: str | None = Field(default=None, description="Optional existing persona ID")
    name: str = Field(..., description="Display name of the loved one / character")
    relationship: str = Field(default="Friend", description="Relationship (e.g. Best Friend, Mother, Father, Mentor)")
    language: str = Field(default="Telugu", description="Primary language")
    dialect_region: str = Field(default="Colloquial", description="Dialect or region")
    tone: str = Field(default="Warm, friendly, casual, authentic", description="Personality and tone")
    frequent_catchphrases: list[str] = Field(default_factory=list, description="Frequently used catchphrases or dialect words")
    pet_names: list[str] = Field(default_factory=list, description="Terms of endearment or addressing style")
    key_topics: list[str] = Field(default_factory=list, description="Topics or shared interests")
    care_instructions: str = Field(default="", description="Specific guidelines or memories")
    voice_profile_id: str = Field(default="default", description="Voice profile identifier")
    custom_system_prompt: str = Field(default="", description="Optional full custom system prompt")


# --- API Endpoints ---

@app.get("/healthz")
async def health_check() -> dict[str, Any]:
    """Health status check."""
    return {
        "status": "healthy",
        "service": "bandhu-agentic-cloud",
        "gemini_model": settings.gemini_model,
        "tts_engine": voice_synthesizer.active_engine,
        "environment": settings.environment,
    }


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest) -> dict[str, Any]:
    """Execute conversational turn with Gemini Agent, proactive tools, and speech synthesis."""
    try:
        # Strictly switch to the requested persona
        target_pid = request.persona_id or "pappa"
        profile = memory_store.get_persona(target_pid)
        if not profile:
            p_file = settings.data_dir / "personas" / target_pid / "profile.json"
            if p_file.exists():
                try:
                    p_data = json.loads(p_file.read_text(encoding="utf-8"))
                    profile = PersonaProfile.from_dict(p_data)
                    memory_store.save_persona(profile)
                except Exception:
                    pass

        if profile and profile.persona_id != gemini_agent.persona.persona_id:
            gemini_agent.set_persona(profile)
        elif not profile and target_pid == "pappa":
            _seed_personas()
            p_obj = memory_store.get_persona("pappa")
            if p_obj:
                gemini_agent.set_persona(p_obj)

        # 1. Agent Reasoning & Tool Dispatch (with multi-turn conversational memory)
        agent_res = await gemini_agent.reply(request.message, speaker_name=request.speaker_name)

        # 2. Adaptive Voice Synthesis with STRICT persona routing
        audio_url = None
        if request.generate_audio and agent_res.reply_text:
            audio_id = f"bandhu_turn_{uuid.uuid4().hex[:8]}.wav"
            out_path = audio_output_dir / audio_id
            try:
                _, engine = await asyncio.wait_for(
                    voice_synthesizer.synthesize(
                        text=agent_res.reply_text,
                        output_file=out_path,
                        voice_id=target_pid,
                    ),
                    timeout=60.0,
                )
                audio_url = f"/api/audio/{audio_id}"
            except Exception as exc:
                try:
                    safe_enc = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
                    err_str = str(exc).encode(safe_enc, errors="replace").decode(safe_enc)
                    print(f"[Warning] Synthesis error: {err_str}")
                except Exception:
                    pass

        return {
            "reply_text": agent_res.reply_text,
            "audio_url": audio_url,
            "tools_executed": agent_res.tools_executed,
            "persona_id": agent_res.persona_id,
            "persona_name": gemini_agent.persona.name,
            "model": agent_res.model,
            "thought_stream": agent_res.thought_stream,
        }
    except Exception as top_exc:
        # Fail gracefully with fallback response
        fallback_reply = (
            f"సరే {request.speaker_name or 'నాయనా'}, నీ మాటలు విన్నాను. "
            f"మన ఇంట్లో విశేషాలు చెప్పు నాయనా."
        )
        return {
            "reply_text": fallback_reply,
            "audio_url": None,
            "tools_executed": [],
            "persona_id": request.persona_id,
            "persona_name": gemini_agent.persona.name,
            "model": "fallback",
            "thought_stream": None,
        }


@app.post("/api/persona/upload-chat")
async def upload_chat_endpoint(
    file: UploadFile = File(None),
    chat_text: str = Form(None),
    target_speaker: str = Form(...),
    relationship: str = Form("Grandmother"),
    language: str = Form("Telugu"),
    dialect: str = Form("Rayalaseema / Chittoor"),
) -> dict[str, Any]:
    """Ingest WhatsApp chat export file or text, and extract persona profile."""
    content = ""
    if file:
        raw_bytes = await file.read()
        content = raw_bytes.decode("utf-8", errors="replace")
    elif chat_text:
        content = chat_text
    else:
        raise HTTPException(status_code=400, detail="Must provide either chat export file or chat_text")

    turns = WhatsAppChatParser.parse_text(content)
    if not turns:
        raise HTTPException(status_code=400, detail="Could not parse any messages from provided chat format.")

    profile = await PersonaExtractor.extract_profile_from_turns(
        turns=turns,
        target_speaker=target_speaker,
        relationship=relationship,
        language=language,
        dialect=dialect,
    )

    saved_profile = memory_store.save_persona(profile)
    # Activate newly created persona immediately
    gemini_agent.set_persona(saved_profile)
    return {
        "status": "success",
        "persona": saved_profile.to_dict(),
        "total_turns_parsed": len(turns),
        "target_speaker_turns": len(WhatsAppChatParser.extract_speaker_turns(turns, target_speaker)),
    }


@app.get("/api/personas")
async def list_personas_endpoint() -> dict[str, Any]:
    """List registered personas with Grandma (Rayalaseema) and Pappa (Hanumakonda) pre-seeded."""
    from bandhu.agent.prompts import CHITTOOR_GRANDMA_SYSTEM_PROMPT, HANUMAKONDA_PAPPA_SYSTEM_PROMPT

    grandma = PersonaProfile(
        persona_id="grandma_chittoor",
        name="అమ్మమ్మ (Grandma)",
        relationship="Grandmother",
        language="Telugu",
        dialect_region="Rayalaseema / Chittoor",
        tone="Loving, traditional, maternal Rayalaseema dialect",
        frequent_catchphrases=["నాయనా", "తింటివా", "స్వామి దయతో చల్లగా ఉండాలి", "బాగుండావా మా Demo Lakshmi", "బాగుండావా మా Demo Priya", "బాగుండావా మా Demo Anjali", "బాగుండావా మా Demo Sita"],
        pet_names=["నాయనా", "తల్లీ", "మా Demo Lakshmi", "మా Demo Priya", "మా Demo Anjali", "మా Demo Sita", "మా Demo Radha", "మా Demo Devi"],
        key_topics=["ఆరోగ్యం (Health)", "భోజనం (Meals)", "యోగక్షేమాలు (Wellbeing)", "కుటుంబ జ్ఞాపకాలు", "కూతుళ్లు & మనవరాళ్లు"],
        voice_profile_id="grandma_chittoor",
        custom_system_prompt=CHITTOOR_GRANDMA_SYSTEM_PROMPT,
    )
    memory_store.save_persona(grandma)

    pappa = memory_store.get_persona("pappa")
    # Always update Pappa with the authentic chat-derived prompt & pet names
    pappa = PersonaProfile(
        persona_id="pappa",
        name="పప్పా (Pappa)",
        relationship="Father",
        language="Telugu",
        dialect_region="Telangana / Hanumakonda",
        tone="Deeply loving, caring father, playful & teasing, authentic WhatsApp conversational style",
        frequent_catchphrases=["లేచినవా నానమ్మ", "Don't worry బేటా", "అంతా మన మంచికే", "అన్నం తిన్నావా"],
        pet_names=["నానమ్మ", "డాడీ", "బేటా", "బంగారం", "దెయ్యం"],
        key_topics=["ఆరోగ్యం & భోజనం", "కెరీర్ & ఉద్యోగం", "యోగక్షేమాలు"],
        voice_profile_id="pappa",
        custom_system_prompt=HANUMAKONDA_PAPPA_SYSTEM_PROMPT,
    )
    memory_store.save_persona(pappa)

    personas = memory_store.list_personas()
    return {
        "status": "success",
        "personas": [p.to_dict() for p in personas],
        "count": len(personas),
    }


@app.get("/api/persona/{persona_id}")
async def get_persona_endpoint(persona_id: str) -> dict[str, Any]:
    """Get single persona by ID."""
    profile = memory_store.get_persona(persona_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Persona not found")
    return {"status": "success", "persona": profile.to_dict()}


@app.post("/api/persona/save")
async def save_persona_endpoint(req: SavePersonaRequest) -> dict[str, Any]:
    """Create or update a custom persona profile."""
    pid = req.persona_id or f"persona_{uuid.uuid4().hex[:8]}"
    vpid = req.voice_profile_id or "default"
    if vpid == "default" and voice_manager.get_voice(pid):
        vpid = pid

    profile = PersonaProfile(
        persona_id=pid,
        name=req.name.strip(),
        relationship=req.relationship.strip(),
        language=req.language.strip() or "Telugu",
        dialect_region=req.dialect_region.strip() or "Colloquial",
        tone=req.tone.strip() or "Warm, authentic, expressive",
        frequent_catchphrases=req.frequent_catchphrases or [],
        pet_names=req.pet_names or [],
        key_topics=req.key_topics or [],
        care_instructions=req.care_instructions or "",
        voice_profile_id=vpid,
        custom_system_prompt=req.custom_system_prompt or "",
    )
    saved = memory_store.save_persona(profile)
    # Activate newly created persona immediately
    gemini_agent.set_persona(saved)
    return {"status": "success", "persona": saved.to_dict()}


@app.post("/api/persona/upload-audio")
async def upload_persona_audio(
    files: list[UploadFile] = File(None),
    file: UploadFile = File(None),
    voice_id: str = Form(...),
    name: str = Form("Custom Voice"),
    transcript: str = Form(""),
) -> dict[str, Any]:
    """Upload one or multiple reference voice samples (.wav / .mp3 / .m4a) for zero-shot voice cloning."""
    ref_dir = settings.data_dir / "reference_audio"
    ref_dir.mkdir(parents=True, exist_ok=True)

    all_uploads: list[UploadFile] = []
    if files:
        all_uploads.extend(files)
    if file and file not in all_uploads:
        all_uploads.append(file)

    if not all_uploads:
        raise HTTPException(status_code=400, detail="No audio files provided")

    saved_paths: list[str] = []
    for idx, ufile in enumerate(all_uploads):
        file_ext = Path(ufile.filename or f"clip_{idx}.wav").suffix.lower() or ".wav"
        out_filename = f"{voice_id}_{uuid.uuid4().hex[:6]}_{idx}{file_ext}"
        out_path = ref_dir / out_filename

        raw_bytes = await ufile.read()
        out_path.write_bytes(raw_bytes)
        saved_paths.append(str(out_path))

    primary_path = saved_paths[0] if saved_paths else ""

    # 1. Determine speaker gender from persona or name
    existing = memory_store.get_persona(voice_id)
    combined_info = f"{name} {voice_id} {existing.relationship if existing else ''}".lower()
    is_grandma = any(w in combined_info for w in ("grandma", "amamma", "అమ్మమ్మ", "నానమ్మ", "mother", "amma", "తల్లి", "sister"))
    is_male = not is_grandma and (
        any(w in combined_info.split() for w in ("male", "man", "boy")) or
        any(w in combined_info for w in ("father", "pappa", "dad", "brother", "grandpa", "thatha", "nanna", "మిత్రుడు", "రాహుల్", "బాబు", "తాతయ్య"))
    )
    gender = "male" if is_male else "female"

    vprofile = voice_manager.register_voice(
        voice_id=voice_id,
        name=name,
        reference_audio_path=primary_path,
        reference_transcript=transcript or "నమస్కారం",
        reference_audio_paths=saved_paths,
        gender=gender,
    )

    # 2. Automatically extract acoustic embeddings and build FAISS Timbre Index
    index_path = None
    try:
        from bandhu.voice_clone.speaker_index import SpeakerIndexBuilder
        builder = SpeakerIndexBuilder()
        idx_file, prof_file = builder.build_from_audio_paths(
            audio_paths=saved_paths,
            speaker_name=name,
            output_dir=settings.data_dir,
            index_name=f"{voice_id}_voice",
        )
        index_path = str(idx_file)

        # 3. Dynamically activate timbre converter in voice synthesizer
        from bandhu.voice_clone.timbre_converter import GrandmaTimbreConverter
        speaker_type = "pappa" if is_male else "grandma"
        conv = GrandmaTimbreConverter(
            index_path=idx_file,
            profile_path=prof_file,
            speaker_type=speaker_type,
        )
        voice_synthesizer.register_timbre_converter(voice_id, conv)
        print(f"[Upload] Automated FAISS voice index built & activated for '{voice_id}' ({len(saved_paths)} clips).")
    except Exception as exc:
        print(f"[Upload] Automated timbre index note for '{voice_id}': {exc}")

    # 4. Associate voice with persona in memory store if exists
    if existing:
        existing.voice_profile_id = voice_id
        memory_store.save_persona(existing)
        gemini_agent.set_persona(existing)

    return {
        "status": "success",
        "voice_profile": vprofile.to_dict(),
        "voice_id": voice_id,
        "persona": existing.to_dict() if existing else None,
        "total_clips_uploaded": len(saved_paths),
        "index_created": bool(index_path),
        "files": [Path(p).name for p in saved_paths],
    }


@app.delete("/api/persona/{persona_id}")
async def delete_persona_endpoint(persona_id: str) -> dict[str, Any]:
    """Delete a persona profile."""
    if persona_id == "grandma_chittoor":
        # Keep default grandma protected or allow reset
        pass
    success = memory_store.delete_persona(persona_id)
    return {"status": "success", "deleted": success, "persona_id": persona_id}


@app.get("/api/health-logs")
async def get_health_logs(persona_id: str = "grandma_chittoor") -> dict[str, Any]:
    """Fetch health logs and alerts for the family dashboard."""
    alerts = memory_store.list_health_alerts(persona_id)
    return {
        "persona_id": persona_id,
        "alerts_count": len(alerts),
        "recent_alerts": [a.to_dict() for a in alerts],
    }


@app.post("/api/cron/checkin")
async def proactive_checkin_cron() -> dict[str, Any]:
    """Asynchronous background check-in runner triggered by Cloud Scheduler."""
    print("[Cron] Background Proactive Check-in Triggered...")
    active_persona = gemini_agent.persona
    # Execute health inquiry turn
    checkin_prompt = "నాయనా ఈరోజు నీ ఒంట్లో ఎలా ఉంది? వేళకు భోజనం చేసి మాత్రలు వేసుకుంటివా లేదా బా?"
    res = await gemini_agent.reply(checkin_prompt, speaker_name="System Scheduler")

    return {
        "status": "executed",
        "persona_id": active_persona.persona_id,
        "checkin_prompt": checkin_prompt,
        "tools_executed": res.tools_executed,
    }


# Hardcoded system trigger for the daily Cloud Scheduler job. Intentionally not
# user-configurable: this endpoint accepts no message body.
SCHEDULED_CHECKIN_TRIGGER = "proactive morning check-in"
# Accepted persona keys -> seeded persona_id in the MemoryStore
SCHEDULED_CHECKIN_PERSONAS = {
    "pappa": "pappa",
    "grandma": "grandma_chittoor",
}


@app.post("/internal/scheduled-checkin")
async def scheduled_checkin(persona: str = "pappa") -> dict[str, Any]:
    """Agent-initiated daily check-in fired by Cloud Scheduler (no user request, no message body)."""
    if persona not in SCHEDULED_CHECKIN_PERSONAS:
        raise HTTPException(
            status_code=400,
            detail=f"persona must be one of {sorted(SCHEDULED_CHECKIN_PERSONAS)}",
        )
    resolved_id = SCHEDULED_CHECKIN_PERSONAS[persona]

    triggered_at = datetime.now(timezone.utc)
    print(
        f"[SCHEDULED-CHECKIN] {triggered_at.isoformat()} persona={resolved_id} "
        f"trigger='{SCHEDULED_CHECKIN_TRIGGER}' initiator=cloud-scheduler "
        f"user_request=none request_body=none",
        flush=True,
    )

    previous_persona = gemini_agent.persona
    target_persona = memory_store.get_persona(resolved_id) or previous_persona
    if target_persona.persona_id != previous_persona.persona_id:
        gemini_agent.set_persona(target_persona)

    try:
        res = await gemini_agent.reply(
            SCHEDULED_CHECKIN_TRIGGER,
            speaker_name="System Scheduler",
        )
    finally:
        if target_persona.persona_id != previous_persona.persona_id:
            gemini_agent.set_persona(previous_persona)

    record = memory_store.store_memory(
        persona_id=target_persona.persona_id,
        category="proactive_checkin",
        topic=f"Scheduled check-in {triggered_at.date().isoformat()}",
        details=res.reply_text,
        importance=3,
    )

    completed_at = datetime.now(timezone.utc)
    print(
        f"[SCHEDULED-CHECKIN] {completed_at.isoformat()} persona={target_persona.persona_id} "
        f"memory_id={record.id} stored_in=bandhu_memories "
        f"initiator=cloud-scheduler user_request=none",
        flush=True,
    )

    return {
        "status": "executed",
        "initiator": "cloud-scheduler",
        "user_initiated": False,
        "trigger": SCHEDULED_CHECKIN_TRIGGER,
        "persona_id": target_persona.persona_id,
        "triggered_at": triggered_at.isoformat(),
        "reply_text": res.reply_text,
        "memory_id": record.id,
        "tools_executed": res.tools_executed,
    }


@app.post("/api/webhook/dispatch-caregiver")
async def dispatch_caregiver_alert(request: Request) -> dict[str, Any]:
    """Receive and process caregiver emergency alert dispatched via Cloud Tasks."""
    try:
        body = await request.json()
    except Exception:
        body_bytes = await request.body()
        body = json.loads(body_bytes.decode("utf-8", errors="replace"))

    severity = body.get("severity", "UNKNOWN")
    patient = body.get("patient_name", "Unknown")
    symptoms = body.get("symptoms", "")

    # Log the dispatch to memory
    memory_store.record_emergency_alert(
        persona_id=body.get("persona_id", "grandma_chittoor"),
        patient_name=patient,
        severity=severity,
        symptoms=symptoms if isinstance(symptoms, str) else ", ".join(symptoms),
        dispatched_to=body.get("dispatched_to", settings.caregiver_phone_number),
        channel="cloud_tasks_webhook",
        alert_payload=json.dumps(body, ensure_ascii=False),
    )

    # TODO: Send actual WhatsApp/SMS notification via Twilio
    # For now, log and confirm receipt
    print(f"[CAREGIVER ALERT] Severity={severity} Patient={patient} Symptoms={symptoms}")

    return {
        "status": "dispatched",
        "severity": severity,
        "patient": patient,
        "message": f"Caregiver alert received and logged for {patient} ({severity})",
    }


@app.api_route("/api/webhook/whatsapp", methods=["GET", "POST"])
async def whatsapp_webhook(request: Request) -> Response:
    """WhatsApp incoming webhook handler supporting Meta Cloud API and Twilio.

    GET:  Meta webhook verification handshake.
    POST: Receive a message, run it through the agent, and reply on the
          same channel.

    For Meta Cloud API, the reply is sent via an async POST to the
    Graph API (see `send_whatsapp_cloud_message` in chat_channels.py).
    For Twilio, the reply is returned as TwiML XML (legacy).
    """
    # 1. Meta Webhook Verification (GET)
    if request.method == "GET":
        mode = request.query_params.get("hub.mode")
        token = request.query_params.get("hub.verify_token")
        challenge = request.query_params.get("hub.challenge")
        if mode == "subscribe" and token == settings.whatsapp_verify_token:
            return Response(content=challenge or "", media_type="text/plain")
        return Response(content="Verification failed", status_code=403)

    # 2. Incoming WhatsApp Message (POST)
    body_bytes = await request.body()
    body_str = body_bytes.decode("utf-8", errors="replace")

    # Detect Twilio form-encoded vs Meta JSON
    is_twilio = "Body=" in body_str and body_str.lstrip().startswith("Body=")
    inbound_messages: list = []
    if is_twilio:
        m = parse_twilio_form(body_str)
        if m:
            inbound_messages.append(m)
    else:
        try:
            payload = json.loads(body_str)
            inbound_messages = parse_whatsapp_payload(payload)
        except Exception:
            # Fallback: treat the whole body as the user text
            inbound_messages = []  # ignore; we won't synthesize a message

    if not inbound_messages:
        # Acknowledge with 200 so Meta doesn't retry, but log it
        return Response(content="no text message", media_type="text/plain", status_code=200)

    # Process each message; collect TwiML for Twilio, async-send for Meta
    twiml_responses: list[str] = []
    for msg in inbound_messages:
        reply_text = await _run_inbound(msg)
        if is_twilio:
            # Escape user-content into TwiML safely
            safe = (
                reply_text.replace("&", "&amp;")
                           .replace("<", "&lt;")
                           .replace(">", "&gt;")
            )
            twiml_responses.append(
                f"<Message><Body>{safe}</Body></Message>"
            )
        else:
            # Meta Cloud API: send reply via the Graph API
            await send_message(OutboundMessage(
                channel="whatsapp",
                recipient=msg.user_id,
                text=reply_text,
            ))

    if is_twilio:
        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f"<Response>{''.join(twiml_responses)}</Response>"
        )
        return Response(content=twiml, media_type="application/xml")

    # Meta path: 200 OK with empty body (Meta doesn't read it)
    return Response(content="ok", media_type="text/plain", status_code=200)


@app.get("/api/telegram/bot-info")
async def telegram_bot_info() -> dict[str, Any]:
    """Return the bot's username (if configured) so the UI can link to it."""
    if not settings.telegram_bot_token:
        return {"ok": False, "error": "TELEGRAM_BOT_TOKEN not configured"}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"https://api.telegram.org/bot{settings.telegram_bot_token}/getMe"
            )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("ok"):
                result = data.get("result", {})
                return {"ok": True, "username": result.get("username", ""), "first_name": result.get("first_name", "")}
        return {"ok": False, "error": "telegram_api_error"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ────────────────────────────────────────────────────────────────────
# Telegram bot webhook
# ────────────────────────────────────────────────────────────────────

@app.post("/api/webhook/telegram")
async def telegram_webhook(request: Request) -> dict[str, Any]:
    """Telegram Bot API webhook with voice note dispatch, persona tracking, and button menu."""
    bot_token = settings.telegram_bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not bot_token:
        return {"ok": False, "error": "TELEGRAM_BOT_TOKEN not configured"}

    try:
        payload = await request.json()
    except Exception:
        return {"ok": False, "error": "invalid JSON body"}

    msg = parse_telegram_update(payload)
    if msg is None:
        return {"ok": True, "ignored": True}

    from bandhu.api.chat_channels import (
        download_telegram_file,
        get_user_session,
        maybe_handle_command,
        send_telegram_chat_action,
        send_telegram_message,
        send_telegram_voice,
    )

    session = get_user_session(msg.user_id)
    cmd_reply = maybe_handle_command(msg.text, gemini_agent, user_id=msg.user_id)
    if cmd_reply is not None:
        await send_telegram_message(msg.user_id, cmd_reply)
        return {"ok": True, "channel": "telegram", "command": True}

    # Ensure agent is set to this user's active persona
    p_id = session.get("persona_id", "grandma_chittoor")
    profile = memory_store.get_persona(p_id)
    if profile:
        gemini_agent.set_persona(profile)

    # 1. Download inbound voice memo if present
    inbound_audio_bytes = None
    if msg.voice_file_id:
        await send_telegram_chat_action(msg.user_id, "record_voice")
        inbound_audio_bytes = await download_telegram_file(msg.voice_file_id)

    # 2. Execute Conversational Turn with Gemini (multimodal voice reasoning enabled)
    agent_res = await gemini_agent.reply(
        msg.text,
        speaker_name=msg.user_name or "Family Member",
        audio_bytes=inbound_audio_bytes,
        audio_mime_type=msg.voice_mime_type or "audio/ogg",
    )
    reply_text = agent_res.reply_text or "…"

    # 3. Formatted Persona Header Badge
    p_label = "👵 అమ్మమ్మ (Grandma · Chittoor)" if p_id == "grandma_chittoor" else "👨 పప్పా (Pappa · Telangana)"
    mode = session.get("output_mode", "combined")
    formatted_text = f"{p_label}\n────────────────────\n{reply_text}"

    # 4. Audio Voice Note Synthesis & Dispatch (if user spoke with voice OR mode is combined/voice_only)
    if msg.voice_file_id or mode in ("combined", "voice_only"):
        try:
            await send_telegram_chat_action(msg.user_id, "record_voice")
            v_id = profile.voice_profile_id if profile else "grandma_chittoor"
            audio_bytes, engine_used = await voice_synthesizer.synthesize(reply_text, voice_id=v_id)
            caption = formatted_text if mode == "voice_only" else ""
            await send_telegram_voice(msg.user_id, audio_bytes, caption=caption)
        except Exception as exc:
            import logging
            logging.getLogger("bandhu.telegram").warning("Voice synthesis note failed: %s", exc)

    # 5. Text Dispatch (for Combined or Text Only modes)
    if mode in ("combined", "text_only"):
        await send_telegram_message(msg.user_id, formatted_text)

    return {"ok": True, "channel": "telegram", "persona": p_id, "mode": mode}


# ────────────────────────────────────────────────────────────────────
# Shared inbound-message handler
# ────────────────────────────────────────────────────────────────────

async def _run_inbound(msg) -> str:
    """Common agent invocation for any chat channel.

    Handles slash commands, runs the agent, and returns the reply text.
    Per-channel formatting/quoting is the caller's responsibility.
    """
    cmd_reply = maybe_handle_command(msg.text, gemini_agent, user_id=msg.user_id)
    if cmd_reply is not None:
        return cmd_reply
    agent_res = await gemini_agent.reply(msg.text, speaker_name=msg.user_name or "Chat User")
    return agent_res.reply_text or "…"


# ────────────────────────────────────────────────────────────────────
# Optional: trigger an outbound message to a chat user
# (used for proactive check-ins, /api/cron/checkin, etc.)
# ────────────────────────────────────────────────────────────────────

class ProactiveChatRequest(BaseModel):
    channel: str = Field(..., description="'telegram' or 'whatsapp'")
    recipient: str = Field(..., description="Telegram chat_id or E.164 phone number")
    persona_id: str = Field(default="grandma_chittoor", description="Persona to use")
    prompt: str = Field(
        default="కన్నా ఈరోజు నీ ఒంట్లో ఎలా ఉంది? వేళకు భోజనం చేసి మాత్రలు వేసుకుంటివా లేదా?",
        description="System-side message to send to the agent",
    )


@app.post("/api/chat/proactive")
async def proactive_chat(req: ProactiveChatRequest) -> dict[str, Any]:
    """Send a server-initiated message to a chat user.

    Useful for the scheduled check-in flow: instead of /api/cron/checkin
    logging to a DB, you can call this to actually ping the user on
    their chat channel.
    """
    if req.channel not in ("telegram", "whatsapp"):
        raise HTTPException(status_code=400, detail="channel must be 'telegram' or 'whatsapp'")
    if req.persona_id and req.persona_id != gemini_agent.persona.persona_id:
        profile = memory_store.get_persona(req.persona_id)
        if profile:
            gemini_agent.set_persona(profile)
    agent_res = await gemini_agent.reply(req.prompt, speaker_name="Bandhu (proactive)")
    dispatch = await send_message(OutboundMessage(
        channel=req.channel,
        recipient=req.recipient,
        text=agent_res.reply_text or "…",
    ))
    return {
        "status": "sent",
        "channel": req.channel,
        "recipient": req.recipient,
        "dispatch": dispatch,
        "reply_chars": len(agent_res.reply_text or ""),
    }


# ────────────────────────────────────────────────────────────────────
# Conversation history per (channel, user)
# ────────────────────────────────────────────────────────────────────

@app.get("/api/conversations/{channel}/{user_id}")
async def get_conversation(channel: str, user_id: str) -> dict[str, Any]:
    """Return the in-memory conversation history for a specific chat user.

    The agent keeps per-turn history in `gemini_agent.history`, which is
    currently a single list (not partitioned per user). For multi-user
    production use this would be a separate dict; for the hackathon
    demo this returns the shared history plus a marker.
    """
    history = list(getattr(gemini_agent, "history", []) or [])
    return {
        "channel": channel,
        "user_id": user_id,
        "turn_count": len(history),
        "history": history[-20:],  # last 20 turns
        "note": "history is shared across all users in the current process; see chat_channels.parse_* for per-user wiring",
    }


@app.get("/api/audio/{filename}")
async def serve_audio(filename: str) -> FileResponse:
    """Serve synthesized voice WAV audio files."""
    file_path = audio_output_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path=file_path, media_type="audio/wav", filename=filename)


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard() -> HTMLResponse:
    """Serve Bandhu interactive web dashboard."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Bandhu Agent Platform is Running</h1>")


# ────────────────────────────────────────────────────────────────────
# Telegram long polling (runs when TELEGRAM_POLLING=true in .env)
# ────────────────────────────────────────────────────────────────────

_telegram_offset = 0


@app.on_event("startup")
async def _start_telegram_polling_if_enabled() -> None:
    """Register commands and optionally start Telegram long polling."""
    import os
    from bandhu.api.chat_channels import register_telegram_commands
    try:
        await register_telegram_commands()
    except Exception:
        pass

    polling_enabled = os.getenv("TELEGRAM_POLLING", "false").lower() == "true"
    if not polling_enabled or not settings.telegram_bot_token:
        return
    import asyncio
    task = asyncio.create_task(_telegram_long_poll_loop())
    task.set_name("telegram-polling")
    logger = logging.getLogger("bandhu.telegram_poll")
    logger.info("Telegram long polling started (webhook disabled)")


async def _telegram_long_poll_loop() -> None:
    """Long-poll Telegram Bot API for incoming messages (runs on local GPU)."""
    import asyncio
    import logging
    import os
    import httpx

    token = settings.telegram_bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        return

    logger = logging.getLogger("bandhu.telegram_poll")

    # Step 1: Delete any active webhook first to prevent 409 Conflict
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            del_resp = await client.post(f"https://api.telegram.org/bot{token}/deleteWebhook", json={"drop_pending_updates": False})
            logger.info("Telegram deleteWebhook result: %s", del_resp.text[:100])
    except Exception as e:
        logger.warning("Could not delete Telegram webhook before polling: %s", e)

    url = f"https://api.telegram.org/bot{token}/getUpdates"
    offset = 0

    while True:
        try:
            async with httpx.AsyncClient(timeout=35.0) as client:
                resp = await client.get(url, params={"offset": offset, "timeout": 20, "allowed_updates": ["message"]})
                if resp.status_code != 200:
                    logger.warning("Telegram getUpdates failed: %s %s", resp.status_code, resp.text[:200])
                    await asyncio.sleep(4)
                    continue
                data = resp.json()
                if not data.get("ok"):
                    logger.warning("Telegram getUpdates not ok: %s", data)
                    await asyncio.sleep(4)
                    continue

                for update in data.get("result", []):
                    offset = update.get("update_id", offset) + 1
                    msg = parse_telegram_update(update)
                    if msg is None:
                        continue

                    from bandhu.api.chat_channels import (
                        download_telegram_file,
                        get_user_session,
                        maybe_handle_command,
                        send_telegram_chat_action,
                        send_telegram_message,
                        send_telegram_voice,
                    )

                    session = get_user_session(msg.user_id)
                    cmd_reply = maybe_handle_command(msg.text, gemini_agent, user_id=msg.user_id)
                    if cmd_reply is not None:
                        await send_telegram_message(msg.user_id, cmd_reply)
                        continue

                    p_id = session.get("persona_id", "grandma_chittoor")
                    profile = memory_store.get_persona(p_id)
                    if profile:
                        gemini_agent.set_persona(profile)

                    inbound_audio_bytes = None
                    if msg.voice_file_id:
                        await send_telegram_chat_action(msg.user_id, "record_voice")
                        inbound_audio_bytes = await download_telegram_file(msg.voice_file_id)

                    await send_telegram_chat_action(msg.user_id, "typing")

                    output_mode = session.get("output_mode", "combined")
                    should_gen_audio = msg.voice_file_id or output_mode in ("combined", "voice_only")

                    agent_res = await gemini_agent.reply(
                        msg.text,
                        speaker_name=msg.user_name or "Family Member",
                        audio_bytes=inbound_audio_bytes,
                        audio_mime_type=msg.voice_mime_type or "audio/ogg",
                    )
                    reply_text = agent_res.reply_text or "…"

                    p_badge = "👵 అమ్మమ్మ (Grandma · Chittoor)" if p_id == "grandma_chittoor" else "👨 పప్పా (Pappa · Telangana)"
                    formatted_text = f"{p_badge}\n━━━━━━━━━━━━━━━━━━━━━\n{reply_text}"

                    if output_mode != "voice_only":
                        await send_telegram_message(msg.user_id, formatted_text)

                    if should_gen_audio:
                        await send_telegram_chat_action(msg.user_id, "record_voice")
                        try:
                            v_id = profile.voice_profile_id if profile else "grandma_chittoor"
                            audio_bytes, engine_used = await voice_synthesizer.synthesize(reply_text, voice_id=v_id)
                            caption = formatted_text if output_mode == "voice_only" else ""
                            await send_telegram_voice(msg.user_id, audio_bytes, caption=caption)
                        except Exception as exc:
                            logger.warning("Local voice synthesis failed: %s", exc)

        except Exception as exc:
            logger.error("Telegram polling error: %s", exc)
            await asyncio.sleep(3)
