"""Configuration settings for Bandhu Agent Platform (Zero-bloat, Pure Python)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class AppSettings:
    """Application-wide settings with environment variable overrides."""

    # Project Directories
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
    )
    data_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data"
    )

    # Gemini & Google GenAI SDK
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv(
            "GEMINI_API_KEY",
            os.getenv("OPENAI_API_KEY", ""),
        )
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
    )
    gemini_fallback_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
    )
    gemini_temperature: float = field(
        default_factory=lambda: float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
    )
    gemini_max_output_tokens: int = field(
        default_factory=lambda: int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1024"))
    )

    # Google Cloud Platform
    gcp_project_id: str = field(
        default_factory=lambda: os.getenv("GCP_PROJECT_ID", "bandhu-agentic-demo")
    )
    firestore_database: str = field(
        default_factory=lambda: os.getenv("FIRESTORE_DATABASE", "(default)")
    )
    gcs_audio_bucket: str = field(
        default_factory=lambda: os.getenv("GCS_AUDIO_BUCKET", "bandhu-audio-vault")
    )
    use_sqlite_fallback: bool = field(
        default_factory=lambda: os.getenv("USE_SQLITE_FALLBACK", "true").lower() == "true"
    )

    # WhatsApp & Twilio Integration
    whatsapp_phone_number_id: str = field(
        default_factory=lambda: os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
    )
    whatsapp_access_token: str = field(
        default_factory=lambda: os.getenv("WHATSAPP_ACCESS_TOKEN", "")
    )
    whatsapp_verify_token: str = field(
        default_factory=lambda: os.getenv("WHATSAPP_VERIFY_TOKEN", "bandhu_secure_verify_token_2026")
    )
    twilio_account_sid: str = field(
        default_factory=lambda: os.getenv("TWILIO_ACCOUNT_SID", "")
    )
    twilio_auth_token: str = field(
        default_factory=lambda: os.getenv("TWILIO_AUTH_TOKEN", "")
    )
    twilio_whatsapp_number: str = field(
        default_factory=lambda: os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
    )

    # Caregiver Emergency Notification
    caregiver_phone_number: str = field(
        default_factory=lambda: os.getenv("CAREGIVER_PHONE_NUMBER", "+919876543210")
    )
    caregiver_email: str = field(
        default_factory=lambda: os.getenv("CAREGIVER_EMAIL", "caregiver@bandhu.local")
    )

    # Audio & TTS Settings
    tts_mode: Literal["auto", "indicf5", "gcp_tts", "mock"] = field(
        default_factory=lambda: os.getenv("TTS_MODE", "auto")  # type: ignore[assignment]
    )
    tts_sample_rate: int = field(
        default_factory=lambda: int(os.getenv("TTS_SAMPLE_RATE", "24000"))
    )
    stt_sample_rate: int = field(
        default_factory=lambda: int(os.getenv("STT_SAMPLE_RATE", "16000"))
    )

    # Server Settings
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8080")))
    environment: str = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development")
    )

    def ensure_directories(self) -> None:
        """Create necessary data subdirectories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "sample_chats").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "reference_audio").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "output_audio").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "personas").mkdir(parents=True, exist_ok=True)


# Global settings singleton
settings = AppSettings()
settings.ensure_directories()
