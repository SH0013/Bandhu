"""Persona profile extractor leveraging Google Gemini and linguistic analysis."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from bandhu.config import settings
from bandhu.persona.models import PersonaProfile, SpeakerTurn
from bandhu.persona.parser import WhatsAppChatParser


class PersonaExtractor:
    """Extracts personality traits, vocabulary, and care instructions from dialogues."""

    @classmethod
    async def extract_profile_from_turns(
        cls,
        turns: list[SpeakerTurn],
        target_speaker: str,
        relationship: str = "Grandmother",
        language: str = "Telugu",
        dialect: str = "Rayalaseema / Chittoor",
    ) -> PersonaProfile:
        """Extract a full PersonaProfile from filtered speaker turns."""
        resolved_speaker = WhatsAppChatParser.resolve_target_speaker(turns, target_speaker)
        speaker_messages = [
            t.text for t in turns if t.speaker.lower() == resolved_speaker.lower()
        ]

        if not speaker_messages:
            # Fallback to all turns if target speaker still not found
            speaker_messages = [t.text for t in turns if t.text]
        if not speaker_messages:
            return cls._build_default_profile(target_speaker, relationship, language, dialect)

        # Attempt Gemini-powered extraction if API key is configured
        if settings.gemini_api_key:
            try:
                return await cls._extract_with_gemini(
                    speaker_messages=speaker_messages[:150],  # Sample up to 150 turns
                    target_speaker=target_speaker or resolved_speaker,
                    relationship=relationship,
                    language=language,
                    dialect=dialect,
                )
            except Exception as exc:
                print(f"[Warning] Gemini Persona Extraction failed, falling back to heuristic: {exc}")

        # Heuristic extraction fallback
        return cls._extract_with_heuristics(
            speaker_messages=speaker_messages,
            target_speaker=target_speaker or resolved_speaker,
            relationship=relationship,
            language=language,
            dialect=dialect,
        )

    @classmethod
    async def _extract_with_gemini(
        cls,
        speaker_messages: list[str],
        target_speaker: str,
        relationship: str,
        language: str,
        dialect: str,
    ) -> PersonaProfile:
        """Use Google GenAI SDK to synthesize structured persona profile."""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=settings.gemini_api_key)
        sample_dialogue = "\n".join(f"- {msg}" for msg in speaker_messages)

        prompt = f"""You are an expert computational sociolinguist and voice biographer. Analyze the following actual WhatsApp message samples from '{target_speaker}' ({relationship}) who communicates in {language} ({dialect} dialect).

Message Samples from '{target_speaker}':
{sample_dialogue}

Extract the persona profile into the following JSON format:
{{
    "name": "{target_speaker}",
    "relationship": "{relationship}",
    "language": "{language}",
    "dialect_region": "{dialect}",
    "tone": "<Short description of their personality, warmth, humor, and emotion>",
    "frequent_catchphrases": ["<up to 5 authentic catchphrases or dialect expressions they use>"],
    "pet_names": ["<terms of endearment or affectionate words used towards the user>"],
    "key_topics": ["<3-5 main topics they discuss, e.g. career, health, family, routines>"],
    "care_instructions": "<summary of how this persona communicates, guides, and cares for the user>",
    "custom_system_prompt": "<A tailored 2-3 paragraph system prompt embodying this exact person in first-person voice>"
}}

Respond ONLY with the raw JSON object."""

        models_to_try = [
            settings.gemini_model,
            settings.gemini_fallback_model,
            "gemini-3.7-flash",
            "gemini-3.5-flash-lite",
        ]
        candidates = []
        for m in models_to_try:
            if m and m not in candidates:
                candidates.append(m)

        response = None
        for m in candidates:
            try:
                response = await client.aio.models.generate_content(
                    model=m,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json",
                    ),
                )
                if response and response.text:
                    break
            except Exception as e:
                print(f"Extractor model {m} failed: {e}")

        if not response or not response.text:
            raise RuntimeError("All Gemini models failed extraction")

        clean_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.text.strip(), flags=re.MULTILINE)
        data = json.loads(clean_text or "{}")
        persona_id = re.sub(r"[^a-zA-Z0-9_]", "", target_speaker.lower().replace(" ", "_"))
        if not persona_id:
            persona_id = "custom_persona"

        return PersonaProfile(
            persona_id=persona_id,
            name=data.get("name", target_speaker),
            relationship=data.get("relationship", relationship),
            language=data.get("language", language),
            dialect_region=data.get("dialect_region", dialect),
            tone=data.get("tone", "Warm, authentic, expressive"),
            frequent_catchphrases=data.get("frequent_catchphrases", []),
            pet_names=data.get("pet_names", []),
            key_topics=data.get("key_topics", []),
            care_instructions=data.get("care_instructions", ""),
            custom_system_prompt=data.get("custom_system_prompt", ""),
        )

    @classmethod
    def _extract_with_heuristics(
        cls,
        speaker_messages: list[str],
        target_speaker: str,
        relationship: str,
        language: str,
        dialect: str,
    ) -> PersonaProfile:
        """Deterministic heuristic extraction from message word frequencies."""
        combined_text = " ".join(speaker_messages)
        words = re.findall(r"\b[\w\u0C00-\u0C7F]+\b", combined_text)

        # Detect common Telugu / Indian terms of endearment
        known_pet_names = ["కన్నా", "తల్లీ", "నాయనా", "బంగారుతల్లీ", "అమ్మా", "చిన్నీ", "బాబు", "బిడ్డా", "beta", "bacha"]
        pet_names_found = [p for p in known_pet_names if p in combined_text]
        if not pet_names_found:
            pet_names_found = ["కన్నా", "నాయనా"] if relationship.lower() in ("grandmother", "mother", "father") else ["మిత్రమా"]

        # Detect dialect markers
        catchphrases = []
        for phrase in ["బా", "తింటివా", "ఉండాను", "జేస్తిని", "రారా", "పోరా", "ఏమ్రా", "చల్లగా ఉండాలి"]:
            if phrase in combined_text:
                catchphrases.append(phrase)
        if not catchphrases:
            catchphrases = ["తింటివా", "చల్లగా ఉండాలి"]

        persona_id = re.sub(r"[^a-zA-Z0-9_]", "", target_speaker.lower().replace(" ", "_")) or "custom_persona"

        return PersonaProfile(
            persona_id=persona_id,
            name=target_speaker,
            relationship=relationship,
            language=language,
            dialect_region=dialect,
            tone="Deeply loving, vigilant about family health, grounded in regional dialect",
            frequent_catchphrases=catchphrases,
            pet_names=pet_names_found,
            key_topics=["ఆరోగ్యం (Health)", "భోజనం (Meals)", "కుటుంబ బాగోగులు (Family Wellbeing)"],
            care_instructions="Prioritize checking on grandchild meals, monitoring cold/fever, and advising herbal remedies.",
        )

    @classmethod
    def _build_default_profile(
        cls, name: str, relationship: str, language: str, dialect: str
    ) -> PersonaProfile:
        """Create flagship Telugu grandmother preset."""
        return PersonaProfile(
            persona_id="grandma_chittoor",
            name=name or "అమ్మమ్మ (Grandma)",
            relationship=relationship or "Grandmother",
            language=language or "Telugu",
            dialect_region=dialect or "Rayalaseema / Chittoor",
            tone="Loving, traditional, maternal, watchful over grandchild health and diet",
            frequent_catchphrases=["తింటివా", "కడుపు నిండిందా", "స్వామి దయతో చల్లగా ఉండాలి", "బా"],
            pet_names=["కన్నా", "తల్లీ", "నాయనా", "బంగారుతల్లీ"],
            key_topics=["ఆరోగ్యం (Health)", "భోజనం (Meals)", "జ్వరం/జలుబు నివారణ", "కుటుంబ యోగక్షేమాలు"],
            care_instructions="Ensure grandchild eats warm meals, monitor any fever symptoms, and offer affectionate care.",
        )
