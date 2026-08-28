# 👵 Bandhu (బంధు / बंधु)

> **Universal Voice Persona, Lifelong Companion & Empathetic Support Friend for Loved Ones**  
> *Built for the [All Things Agentic Hackathon](https://allthingsagentichackathon.devpost.com/) powered by Google Cloud & Gemini.*

---

## 🌟 Overview

**Bandhu** (*meaning "Loved One & Faithful Companion"*) is an empathetic voice persona and support companion platform that lets anyone preserve and connect with their loved ones (Grandparents, Parents, Childhood Mentors, or Dearest Friends) in their authentic voice, regional dialect, and conversational personality:

- ❤️ **Empathetic Companion & Confidant**: Talk freely about your day, work stress, feelings, or happy moments with a trusted companion who listens with deep maternal warmth and regional humor.
- 💬 **WhatsApp Chat Ingestion**: Ingests real WhatsApp `.txt` exports to clone vocabulary, pet names (`కన్నా`, `తల్లీ`, `నాయనా`, `బంగారుతల్లీ`), and regional catchphrases (`బా`, `జేస్తిని`).
- 🎙️ **Universal Voice Cloning**: Zero-Shot regional voice cloning using GPU **IndicF5** with zero-crash fallback to **Google Cloud Neural2 TTS**.
- 📖 **Living Memory & Cultural Lore**: Reminisces family memories, archives oral folklore, and preserves traditional recipes (Ragi Sangati, Pepper Kashayam).
- 🌿 **Proactive Care & Well-Being**: Gently checks in on your meals and rest, suggests comforting home remedies when you're tired, and autonomously alerts family caregivers in medical emergencies.
- ☁️ **Google Cloud Native**: Powered by the official `google-genai` SDK with **Gemini 2.5 Flash**, **Google Cloud Firestore**, and **Google Cloud Run**.

---

## 🚀 Quick Start (Local & Cloud)

### 1. Installation
```bash
# Clone and enter directory
cd bandhu-agentic-cloud

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file (or copy `.env.example`):
```bash
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.7-flash
GCP_PROJECT_ID=your_gcp_project_id
USE_SQLITE_FALLBACK=true
```

### 3. Run Automated Tests (26/26 passing)
```bash
pytest -v
```

### 4. Launch Web Dashboard & Companion Server
```bash
python run_server.py
# or: uvicorn bandhu.api.app:app --host 0.0.0.0 --port 8080
```
Open **[http://localhost:8080](http://localhost:8080)** in your browser!

---

## ☁️ 1-Click Google Cloud Run Deployment

Deploy directly to Google Cloud Run:

```powershell
# Windows PowerShell
.\deploy_cloudrun.ps1 -ProjectId "your-gcp-project-id" -Region "us-central1"
```

```bash
# Linux / macOS / Cloud Shell
./deploy_cloudrun.sh "your-gcp-project-id" "us-central1"
```

---

## 📁 Repository Structure

```text
bandhu-agentic-cloud/
├── Dockerfile                      # Production Google Cloud Run container
├── cloudbuild.yaml                 # Google Cloud Build automated pipeline
├── deploy_cloudrun.ps1             # 1-click deployment script (PowerShell)
├── deploy_cloudrun.sh              # 1-click deployment script (Bash)
├── pyproject.toml                  # Modern Python 3.12, strict dependencies & pytest
├── requirements.txt                # Lean dependencies (google-genai, fastapi, etc.)
├── ARCHITECTURE.md                 # System architecture & Mermaid data flows
├── DEMO_SCRIPT.md                  # 4-minute unedited live video walkthrough script
├── SUBMISSION.md                   # Complete Devpost submission answers
├── data/
│   ├── sample_chats/               # Sample WhatsApp export files
│   └── reference_audio/            # Preset voice reference clips
├── src/
│   └── bandhu/
│       ├── config.py               # Typed configuration and environment parsing
│       ├── persona/
│       │   ├── parser.py           # WhatsApp chat export (.txt) parser
│       │   ├── extractor.py        # Gemini-powered Persona & Dialect Extractor
│       │   └── models.py           # PersonaProfile and SpeakerTurn schemas
│       ├── agent/
│       │   ├── gemini_agent.py     # Official google-genai Agent & tool dispatcher
│       │   ├── prompts.py          # Authentic Chittoor Rayalaseema Telugu prompt
│       │   └── tools.py            # Typed tools (memory recall, recipes, health triage, alerts)
│       ├── memory/
│       │   ├── store.py            # Firestore Memory Bank with SQLite fallback
│       │   └── schema.py           # Memory, health logs, and alert schemas
│       ├── audio/
│       │   ├── tts.py              # Adaptive TTS (IndicF5 GPU + GCP Neural TTS)
│       │   ├── voice_manager.py    # Multi-voice profile manager
│       │   └── processor.py        # Audio resampling, RMS normalization, DSP
│       └── api/
│           ├── app.py              # FastAPI server (chat, WhatsApp webhook, cron)
│           └── static/
│               └── index.html      # Minimalist interactive companion dashboard
└── tests/
    ├── test_chat_parser.py         # Unit tests for WhatsApp chat parsing
    ├── test_persona_extractor.py   # Unit tests for persona extraction
    ├── test_gemini_agent.py        # Unit tests for Gemini agent & function calls
    ├── test_tools.py               # Unit tests for autonomous tool logic
    ├── test_memory.py              # Unit tests for Firestore/SQLite memory
    ├── test_tts.py                 # Unit tests for adaptive audio synthesis
    └── test_api.py                 # Unit tests for FastAPI REST & webhook endpoints
```

---

## 🏆 Hackathon Alignment

| Requirement | How Bandhu Meets It |
| :--- | :--- |
| **Gemini 2.5 Flash** | Official `google-genai` SDK with `client.models.generate_content`. |
| **Google Agent Framework** | Official Google GenAI SDK with native typed Function Declarations. |
| **Google Cloud Service** | Google Cloud Run, Google Cloud Firestore, and Google Cloud Storage. |
| **Collaborative Partner (40%)** | Empathetic emotional companion, spontaneous daily check-ins, folklore archiving, and proactive caregiver safety net. |
| **Production Readiness (30%)** | 100% passing test suite, multi-stage Dockerfile, reproducible setup. |
