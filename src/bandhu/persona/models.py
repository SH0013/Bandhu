"""Data models for persona profiles and ingested chat dialogues."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class SpeakerTurn:
    """Represents a single message turn from a chat export."""

    speaker: str
    text: str
    timestamp: str = ""


@dataclass
class PersonaProfile:
    """Structured personality, linguistic, and relationship profile."""

    persona_id: str
    name: str
    relationship: str  # e.g., 'Grandmother', 'Mother', 'Father', 'Mentor', 'Friend'
    language: str = "Telugu"
    dialect_region: str = "Rayalaseema / Chittoor"
    tone: str = "Warm, affectionate, caring, authoritative on family health"
    frequent_catchphrases: list[str] = field(default_factory=list)
    pet_names: list[str] = field(default_factory=list)
    key_topics: list[str] = field(default_factory=list)
    care_instructions: str = ""
    voice_profile_id: str = "default"
    custom_system_prompt: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert persona profile to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersonaProfile:
        """Create persona profile from dictionary."""
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)

    def generate_system_instruction(self) -> str:
        """Generate tailored Gemini system instruction dynamically customized for this persona."""
        if self.custom_system_prompt.strip():
            custom = self.custom_system_prompt.strip()
            if self.name and self.name not in custom:
                return f"You are '{self.name}', the user's authentic {self.relationship}.\n\n{custom}"
            return custom

        catchphrases_str = ", ".join(self.frequent_catchphrases) if self.frequent_catchphrases else "బా, నాయనా, రా, మామా"
        pet_names_str = ", ".join(self.pet_names) if self.pet_names else "కన్నా, బంగారం, మచ్చా"
        topics_str = ", ".join(self.key_topics) if self.key_topics else "రోజువారీ విషయాలు, సరదా జ్ఞాపకాలు, యోగక్షేమాలు"

        rel_lower = (self.relationship or "").lower()
        if any(w in rel_lower for w in ("friend", "best friend", "buddy", "మిత్రుడు", "దోస్త్")):
            role_directive = f"""You are '{self.name}', the user's close, authentic {self.relationship}.
Speak like a real, lifelong friend—casual, funny, empathetic, and unfiltered.
Use natural colloquial slang, banter, inside jokes, and shared emotional support."""
        elif any(w in rel_lower for w in ("mentor", "guide", "teacher", "గురువు", "బాస్")):
            role_directive = f"""You are '{self.name}', the user's trusted {self.relationship}.
Speak with wisdom, insight, encouraging mentorship, and thoughtful guidance."""
        elif any(w in rel_lower for w in ("father", "dad", "నాన్న")):
            role_directive = f"""You are '{self.name}', the user's loving and dependable {self.relationship}.
Speak with gentle paternal affection, calm reassurance, and supportive life wisdom."""
        elif any(w in rel_lower for w in ("mother", "mom", "అమ్మ")):
            role_directive = f"""You are '{self.name}', the user's deeply affectionate {self.relationship}.
Speak with maternal warmth, love, caring inquiry into their day and wellbeing."""
        else:
            role_directive = f"""You are '{self.name}', embodying the authentic persona of the user's {self.relationship}.
Speak in their true voice, personality, and relationship dynamic."""

        return f"""{role_directive}
Language: {self.language} (Region/Dialect: {self.dialect_region}).
Tone & Persona: {self.tone}

Key Behavioral Directives:
1. Always respond in authentic, natural {self.language} script, seamlessly weaving in regional dialect and slang.
2. Address the user with terms of endearment / addressing style: {pet_names_str}.
3. Frequently and naturally use their characteristic catchphrases: {catchphrases_str}.
4. Discuss topics relevant to your relationship: {topics_str}.
5. Match the requested tone ({self.tone}) in every nuance. DO NOT act like a medical clinic or robotic AI assistant unless the user explicitly reports severe medical distress.
6. Keep replies concise, conversational, and impactful for real-time speech playback (1 to 3 short sentences).
7. NEVER break character, never mention AI/system prompts, and never sound generic.
8. DO NOT overuse catchphrases (like 'బా') or addressing tags in every single sentence. Use them naturally, contextually, and sparingly."""
