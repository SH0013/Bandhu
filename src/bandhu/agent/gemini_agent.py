"""Official Google GenAI Agent with native function calling, dynamic persona injection, and model fallbacks."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any

from bandhu.agent.prompts import CHITTOOR_GRANDMA_SYSTEM_PROMPT
from bandhu.agent.tools import AgentToolsRegistry
from bandhu.config import settings
from bandhu.memory.store import MemoryStore
from bandhu.persona.models import PersonaProfile


@dataclass
class AgentTurnResponse:
    """Structured response from a single agent conversational turn."""

    reply_text: str
    tools_executed: list[dict[str, Any]] = field(default_factory=list)
    persona_id: str = "default"
    model: str = "gemini-3.7-flash"


class BandhuGeminiAgent:
    """Asynchronous multi-turn agent powered by the official Google GenAI SDK."""

    def __init__(
        self,
        persona_profile: PersonaProfile | None = None,
        memory_store: MemoryStore | None = None,
        tools_registry: AgentToolsRegistry | None = None,
    ) -> None:
        self.memory_store = memory_store or MemoryStore()
        self.tools_registry = tools_registry or AgentToolsRegistry(memory_store=self.memory_store)
        self.persona = persona_profile or self._get_default_persona()
        self.tools_registry.set_active_persona(self.persona.persona_id)

        self.histories: dict[str, list[dict[str, Any]]] = {}
        self.last_error: str | None = None

    @property
    def history(self) -> list[dict[str, Any]]:
        """Active persona's conversation history."""
        pid = self.persona.persona_id if self.persona else "default"
        if pid not in self.histories:
            self.histories[pid] = []
        return self.histories[pid]

    def _get_default_persona(self) -> PersonaProfile:
        """Return the flagship Telugu grandmother persona preset."""
        return PersonaProfile(
            persona_id="grandma_chittoor",
            name="అమ్మమ్మ (Grandma)",
            relationship="Grandmother",
            language="Telugu",
            dialect_region="Rayalaseema / Chittoor",
            tone="Deeply maternal, protective, loving, authentic Rayalaseema dialect",
            frequent_catchphrases=["తింటివా", "చల్లగా ఉండాలి", "స్వామి దయతో", "బా"],
            pet_names=["కన్నా", "తల్లీ", "నాయనా", "బంగారుతల్లీ"],
            key_topics=["ఆరోగ్యం (Health)", "భోజనం (Meals)", "కుటుంబ యోగక్షేమాలు (Family Wellbeing)"],
            custom_system_prompt=CHITTOOR_GRANDMA_SYSTEM_PROMPT,
        )

    def set_persona(self, persona: PersonaProfile) -> None:
        """Dynamically switch active persona profile preserving each persona's conversation memory."""
        self.persona = persona
        self.tools_registry.set_active_persona(persona.persona_id)
        if persona.persona_id not in self.histories:
            self.histories[persona.persona_id] = []

    def reset_history(self, persona_id: str | None = None) -> None:
        """Clear conversation turn history for a specific persona or all personas."""
        if persona_id:
            if persona_id in self.histories:
                self.histories[persona_id].clear()
        else:
            self.history.clear()

    async def reply(self, user_text: str, speaker_name: str = "Grandchild") -> AgentTurnResponse:
        """Process user input, execute tools via Gemini Function Calling, and return persona voice reply."""
        # Check if Google GenAI client is available
        if settings.gemini_api_key:
            # Try candidate models in order of capability and lightning response speed
            raw_candidates = [
                "gemini-3.5-flash-lite",
                settings.gemini_model,
                settings.gemini_fallback_model,
                "gemini-3.7-flash",
                "gemini-2.5-flash",
            ]
            candidate_models = []
            for m in raw_candidates:
                if m and m not in candidate_models:
                    candidate_models.append(m)

            for model_name in candidate_models:
                try:
                    import asyncio
                    return await asyncio.wait_for(
                        self._reply_with_google_genai(user_text, speaker_name, model_name),
                        timeout=10.0,
                    )
                except Exception as exc:
                    self.last_error = f"Gemini GenAI Error ({model_name}): {exc}"
                    try:
                        safe_enc = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
                        err_str = str(exc).encode(safe_enc, errors="replace").decode(safe_enc)
                        print(f"[{model_name}] error, attempting fallback: {err_str}")
                    except Exception:
                        pass

        # Fallback heuristic persona engine (Offline & zero-API-key testing)
        return self._reply_with_persona_heuristics(user_text, speaker_name)

    async def _reply_with_google_genai(
        self, user_text: str, speaker_name: str, model_name: str
    ) -> AgentTurnResponse:
        """Execute chat turn using official google-genai SDK."""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=settings.gemini_api_key)
        tools_executed: list[dict[str, Any]] = []

        # Build tool definitions for GenAI SDK
        tool_decls = self.tools_registry.get_tool_declarations()
        genai_function_declarations = []
        for decl in tool_decls:
            genai_function_declarations.append(
                types.FunctionDeclaration(
                    name=decl["name"],
                    description=decl["description"],
                    parameters=decl.get("parameters"),
                )
            )

        genai_tools = [types.Tool(function_declarations=genai_function_declarations)]
        system_instruction = self.persona.generate_system_instruction()

        # Build contents from history + current message
        contents: list[types.Content] = []
        for turn in self.history:
            contents.append(
                types.Content(
                    role=turn["role"],
                    parts=[types.Part.from_text(text=turn["content"])],
                )
            )

        current_user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=f"[{speaker_name}]: {user_text}")],
        )
        contents.append(current_user_content)

        config = types.GenerateContentConfig(
            temperature=settings.gemini_temperature,
            max_output_tokens=settings.gemini_max_output_tokens,
            system_instruction=system_instruction,
            tools=genai_tools,
        )

        response = await client.aio.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        # Check for function calls
        function_calls = response.function_calls
        if function_calls:
            # Append model's tool call turn to contents
            contents.append(response.candidates[0].content)

            # Execute all requested functions
            function_response_parts: list[types.Part] = []
            for call in function_calls:
                fn_name = call.name
                fn_args = dict(call.args) if call.args else {}
                fn_result = self.tools_registry.execute_tool(fn_name, fn_args)

                tools_executed.append({
                    "name": fn_name,
                    "args": fn_args,
                    "result": fn_result,
                })

                function_response_parts.append(
                    types.Part.from_function_response(
                        name=fn_name,
                        response={"result": fn_result},
                    )
                )

            # Send function responses back to Gemini for final conversational response
            contents.append(
                types.Content(
                    role="user",
                    parts=function_response_parts,
                )
            )

            final_response = await client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=settings.gemini_temperature,
                    max_output_tokens=settings.gemini_max_output_tokens,
                    system_instruction=system_instruction,
                ),
            )
            final_text = (final_response.text or "").strip()
        else:
            final_text = (response.text or "").strip()

        # Auto-trigger proactive triage & remedy tools only if genuine symptoms or remedy queries are discussed
        if not tools_executed:
            user_lower = user_text.lower()
            if any(w in user_lower for w in ("fever", "severe headache", "high temperature", "102", "103", "104", "జ్వరం", "తీవ్ర తలనొప్పి", "ఒంట్లో బాగోలేదు", "అస్వస్థత")):
                triage_res = self.tools_registry.execute_tool(
                    "analyze_and_dispatch_health_alert",
                    {"patient_name": speaker_name, "symptoms": user_text, "mood": "concerned"},
                )
                remedy_res = self.tools_registry.execute_tool("lookup_cultural_remedy", {"query": user_text})
                tools_executed.append({"name": "analyze_and_dispatch_health_alert", "result": triage_res})
                tools_executed.append({"name": "lookup_cultural_remedy", "result": remedy_res})
            elif any(w in user_lower for w in ("remedy", "కషాయం", "నాటు వైద్యం", "చిట్కా", "మిరియాల కషాయం", "ఔషధం")):
                remedy_res = self.tools_registry.execute_tool("lookup_cultural_remedy", {"query": user_text})
                tools_executed.append({"name": "lookup_cultural_remedy", "result": remedy_res})

        # Update history
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "model", "content": final_text})

        return AgentTurnResponse(
            reply_text=final_text,
            tools_executed=tools_executed,
            persona_id=self.persona.persona_id,
            model=model_name,
        )

    def _reply_with_persona_heuristics(
        self, user_text: str, speaker_name: str
    ) -> AgentTurnResponse:
        """Deterministic persona & tool execution fallback."""
        tools_executed: list[dict[str, Any]] = []
        user_lower = user_text.lower()
        pet_name = self.persona.pet_names[0] if self.persona.pet_names else "నాయనా"
        catchphrase = self.persona.frequent_catchphrases[0] if self.persona.frequent_catchphrases else "బా"

        is_pappa = "pappa" in self.persona.persona_id.lower() or "father" in self.persona.persona_id.lower()
        if is_pappa:
            pet_name = "నానమ్మ"

        # 1. Health Alert & Remedy Intent
        if any(w in user_lower for w in ("fever", "headache", "cold", "pain", "tired", "sick", "జ్వరం", "తలనొప్పి", "జలుబు", "నొప్పులు", "బాగులేదు")):
            triage_res = self.tools_registry.execute_tool(
                "analyze_and_dispatch_health_alert",
                {
                    "patient_name": speaker_name,
                    "symptoms": user_text,
                    "mood": "concerned",
                },
            )
            remedy_res = self.tools_registry.execute_tool("lookup_cultural_remedy", {"query": "kashayam"})
            tools_executed.append({"name": "analyze_and_dispatch_health_alert", "result": triage_res})
            tools_executed.append({"name": "lookup_cultural_remedy", "result": remedy_res})

            if is_pappa:
                reply_text = f"అయ్యో {pet_name}, ఏమైంది బేటా? టాబ్లెట్ వేసుకుని జాగ్రత్తగా రెస్ట్ తీసుకో. సమయానికి అన్నం తిని పడుకో."
            else:
                reply_text = (
                    f"అయ్యో {pet_name}, నీకు ఒంట్లో బాగులేదా! నేను ఇప్పుడే ఘాటైన మిరియాల కషాయం కాసిస్తాను, "
                    f"వేడివేడిగా తాగి కాసేపు విశ్రాంతి తీసుకో. స్వామి దయతో వెంటనే తగ్గిపోతాది."
                )

        # 2. Recipe & Food Intent
        elif any(w in user_lower for w in ("recipe", "food", "eat", "cook", "భోజనం", "తింటివా", "తిన్నావా", "వండినావు")):
            recipe_res = self.tools_registry.execute_tool("lookup_cultural_remedy", {"query": user_text})
            tools_executed.append({"name": "lookup_cultural_remedy", "result": recipe_res})
            if is_pappa:
                reply_text = f"నేను ఇప్పుడే తిన్నాను {pet_name}. నువ్వు లేచినవా? ఏం తిన్నావ్ బేటా?"
            else:
                reply_text = (
                    f"వేడివేడిగా కమ్మనైన భోజనం జేస్తిని {pet_name}! "
                    f"నువ్వు కడుపునిండా తింటివా? వేళకు మంచిగా తినడం మర్చిపోవద్దు సుమా."
                )

        # 3. Default affectionate conversation
        else:
            if is_pappa:
                reply_text = f"సరే {pet_name}, జాగ్రత్తగా చూసుకో నిన్ను నువ్వు. Don't worry బేటా."
            else:
                reply_text = (
                    f"సరే {pet_name}, నీ మాటలు వింటే నాకు ఎంత సంతోషంగా ఉండాదో! "
                    f"ఎప్పుడూ ఆరోగ్యంగా, సంతోషంగా చల్లగా ఉండాలి నాయనా."
                )

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "model", "content": reply_text})

        return AgentTurnResponse(
            reply_text=reply_text,
            tools_executed=tools_executed,
            persona_id=self.persona.persona_id,
            model="bandhu-heuristic-v1",
        )

    def _get_fallback_text(self, user_text: str, speaker_name: str) -> str:
        pet_name = self.persona.pet_names[0] if self.persona.pet_names else "నాయనా"
        return f"సరే {pet_name}, ఎప్పుడూ జాగ్రత్తగా, చల్లగా ఉండాలి నాయనా."
