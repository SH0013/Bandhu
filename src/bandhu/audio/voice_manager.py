"""Multi-voice profile manager for loved ones' cloned voices."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bandhu.config import settings


@dataclass
class VoiceProfile:
    """Represents reference audio and transcript for a cloned voice persona."""

    voice_id: str
    name: str
    reference_audio_path: str
    reference_transcript: str
    language_code: str = "te"
    gender: str = "female"
    sample_rate: int = 24000
    reference_audio_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VoiceProfileManager:
    """Manages voice registration and reference audio storage."""

    def __init__(self, storage_dir: Path | str | None = None) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir else settings.data_dir / "reference_audio"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_dir / "voice_profiles_index.json"
        self.profiles: dict[str, VoiceProfile] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load registered voice profiles index or register default grandma clips."""
        if self.index_file.exists():
            try:
                data = json.loads(self.index_file.read_text(encoding="utf-8"))
                for vid, vdata in data.items():
                    # Resolve relative paths against project root
                    raw_path = vdata.get("reference_audio_path", "")
                    if raw_path and not Path(raw_path).is_absolute():
                        resolved = settings.project_root / raw_path
                        if resolved.exists():
                            vdata["reference_audio_path"] = str(resolved)
                    self.profiles[vid] = VoiceProfile(**vdata)
            except Exception as exc:
                print(f"[Warning] Failed to read voice index: {exc}")

        # Ensure flagship Telugu Grandma reference clips are registered
        best_clip = self.storage_dir / "grandma_clip_0024.wav"
        if not best_clip.exists():
            best_clip = self.storage_dir / "grandma_clip_0021.wav"

        if "grandma_chittoor" not in self.profiles:
            self.register_voice(
                voice_id="grandma_chittoor",
                name="అమ్మమ్మ (Chittoor Telugu Grandma)",
                reference_audio_path=str(best_clip),
                reference_transcript="నేనేం చెయ్యాలనుకోలేదు నేను ఇంట్లో ఇంట్లోనే దిగాల అనుకున్నా",
                language_code="te",
                gender="female",
            )

        # Register alternate reference candidates if present
        ref_candidates = [
            ("grandma_chittoor_clip21", "grandma_clip_0021.wav", "ఆ పేరుతో ఇల్లు గడ్డి చేసుకున్నారంటే ఇంకా అంగడంతా యూరిని తినోడేది"),
            ("grandma_chittoor_clip24", "grandma_clip_0024.wav", "నేనేం చెయ్యాలనుకోలేదు నేను ఇంట్లో ఇంట్లోనే దిగాల అనుకున్నా"),
            ("grandma_chittoor_clip29", "grandma_clip_0029.wav", "అటనే అనుకుంటాను నాకెందుకు మడితే"),
            ("grandma_chittoor_clip79", "grandma_clip_0079.wav", "గేమ్ ఉండ విశేష అవు ఉంటాయికి ఇవే విశేష అవు"),
            ("grandma_chittoor_clip82", "grandma_clip_0082.wav", "గీతకు మర్ది కొడుకు"),
        ]

        for vid, fname, transcript in ref_candidates:
            cpath = self.storage_dir / fname
            if cpath.exists() and vid not in self.profiles:
                self.register_voice(
                    voice_id=vid,
                    name=f"అమ్మమ్మ ({fname})",
                    reference_audio_path=str(cpath),
                    reference_transcript=transcript,
                    language_code="te",
                    gender="female",
                )

    def _save_index(self) -> None:
        """Persist index to disk."""
        data = {vid: vp.to_dict() for vid, vp in self.profiles.items()}
        self.index_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def register_voice(
        self,
        voice_id: str,
        name: str,
        reference_audio_path: str,
        reference_transcript: str,
        language_code: str = "te",
        gender: str = "female",
        reference_audio_paths: list[str] | None = None,
    ) -> VoiceProfile:
        """Register or update a voice profile with one or more reference audio clips."""
        all_paths = list(reference_audio_paths or [])
        if reference_audio_path and reference_audio_path not in all_paths:
            all_paths.insert(0, reference_audio_path)

        profile = VoiceProfile(
            voice_id=voice_id,
            name=name,
            reference_audio_path=reference_audio_path or (all_paths[0] if all_paths else ""),
            reference_transcript=reference_transcript,
            language_code=language_code,
            gender=gender,
            reference_audio_paths=all_paths,
        )
        self.profiles[voice_id] = profile
        self._save_index()
        return profile

    def get_voice(self, voice_id: str) -> VoiceProfile | None:
        """Retrieve voice profile by ID."""
        return self.profiles.get(voice_id) or self.profiles.get("grandma_chittoor")

    def list_voices(self) -> list[VoiceProfile]:
        """List all available voice profiles."""
        return list(self.profiles.values())
