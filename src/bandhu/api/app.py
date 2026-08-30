"""FastAPI Production Server for Bandhu Universal Agent Platform."""

from __future__ import annotations

import asyncio
import io
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from bandhu.agent.gemini_agent import BandhuGeminiAgent
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
        frequent_catchphrases=["నాయనా", "తింటివా", "స్వామి దయతో చల్లగా ఉండాలి", "బా"],
        pet_names=["కన్నా", "తల్లీ", "నాయనా", "బంగారుతల్లీ"],
        key_topics=["ఆరోగ్యం (Health)", "భోజనం (Meals)", "యోగక్షేమాలు (Wellbeing)"],
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
        }
    except Exception as top_exc:
        # Fail gracefully with fallback response
        fallback_reply = (
            f"సరే {request.speaker_name or 'కన్నా'}, నీ మాటలు విన్నాను. "
            f"మన ఇంట్లో విశేషాలు చెప్పు నాయనా."
        )
        return {
            "reply_text": fallback_reply,
            "audio_url": None,
            "tools_executed": [],
            "persona_id": request.persona_id,
            "persona_name": gemini_agent.persona.name,
            "model": "fallback",
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

    grandma = memory_store.get_persona("grandma_chittoor")
    if not grandma:
        grandma = PersonaProfile(
            persona_id="grandma_chittoor",
            name="అమ్మమ్మ (Grandma)",
            relationship="Grandmother",
            language="Telugu",
            dialect_region="Rayalaseema / Chittoor",
            tone="Loving, traditional, maternal Rayalaseema dialect",
            frequent_catchphrases=["నాయనా", "తింటివా", "స్వామి దయతో చల్లగా ఉండాలి", "బా"],
            pet_names=["కన్నా", "తల్లీ", "నాయనా", "బంగారుతల్లీ"],
            key_topics=["ఆరోగ్యం (Health)", "భోజనం (Meals)", "యోగక్షేమాలు (Wellbeing)"],
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
    is_male = any(w in combined_info for w in ("father", "pappa", "dad", "male", "brother", "grandpa", "thatha", "nanna", "మిత్రుడు", "రాహుల్", "బాబు", "తాతయ్య"))
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
    checkin_prompt = "కన్నా ఈరోజు నీ ఒంట్లో ఎలా ఉంది? వేళకు భోజనం చేసి మాత్రలు వేసుకుంటివా లేదా?"
    res = await gemini_agent.reply(checkin_prompt, speaker_name="System Scheduler")

    return {
        "status": "executed",
        "persona_id": active_persona.persona_id,
        "checkin_prompt": checkin_prompt,
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
    """WhatsApp incoming webhook handler supporting Meta Cloud API and Twilio."""
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

    user_text = ""
    speaker = "WhatsApp User"

    # Check if Twilio Form format
    if "Body=" in body_str:
        import urllib.parse
        form_data = urllib.parse.parse_qs(body_str)
        user_text = form_data.get("Body", [""])[0]
        speaker = form_data.get("From", ["WhatsApp User"])[0]
    else:
        # Meta JSON format
        try:
            payload = json.loads(body_str)
            entries = payload.get("entry", [])
            if entries:
                changes = entries[0].get("changes", [])
                if changes:
                    messages = changes[0].get("value", {}).get("messages", [])
                    if messages:
                        msg = messages[0]
                        user_text = msg.get("text", {}).get("body", "")
        except Exception:
            user_text = body_str

    if not user_text:
        user_text = "అమ్మమ్మ ఎలా ఉన్నారు?"

    # Execute Agent reply
    agent_res = await gemini_agent.reply(user_text, speaker_name=speaker)

    # Synthesize Audio
    audio_id = f"wa_voice_{uuid.uuid4().hex[:8]}.wav"
    out_path = audio_output_dir / audio_id
    await voice_synthesizer.synthesize(text=agent_res.reply_text, output_file=out_path)

    # Return Twilio TwiML / JSON
    twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>
        <Body>{agent_res.reply_text}</Body>
    </Message>
</Response>"""
    return Response(content=twiml_response, media_type="application/xml")


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
