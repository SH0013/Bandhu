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
    """Manages voice registration and reference audio storage with per-persona folder isolation."""

    def __init__(self, storage_dir: Path | str | None = None) -> None:
        self.personas_dir = settings.data_dir / "personas"
        self.personas_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir = Path(storage_dir) if storage_dir else settings.data_dir / "reference_audio"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_dir / "voice_profiles_index.json"
        self.profiles: dict[str, VoiceProfile] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load registered voice profiles from per-persona folders and legacy index."""
        # 1. Scan dedicated per-persona folders (e.g. data/personas/pappa/voice_profile.json)
        if self.personas_dir.exists():
            for pdir in self.personas_dir.iterdir():
                if pdir.is_dir():
                    vprof_file = pdir / "voice_profile.json"
                    if vprof_file.exists():
                        try:
                            vdata = json.loads(vprof_file.read_text(encoding="utf-8"))
                            vid = vdata.get("voice_id") or pdir.name
                            ref_path = vdata.get("reference_audio_path", "")
                            if ref_path and not Path(ref_path).is_absolute():
                                ref_path = str(settings.project_root / ref_path)
                                vdata["reference_audio_path"] = ref_path
                            self.profiles[vid] = VoiceProfile(
                                voice_id=vid,
                                name=vdata.get("name", vid),
                                reference_audio_path=ref_path,
                                reference_transcript=vdata.get("reference_transcript", ""),
                                language_code=vdata.get("language_code", "te"),
                                gender=vdata.get("gender", "male" if "pappa" in vid else "female"),
                                sample_rate=vdata.get("sample_rate", 24000),
                            )
                            # Register alias if applicable
                            if vid == "pappa":
                                self.profiles["father"] = self.profiles[vid]
                            elif vid in ("grandma", "grandma_chittoor"):
                                self.profiles["grandma"] = self.profiles[vid]
                                self.profiles["grandma_chittoor"] = self.profiles[vid]
                        except Exception as exc:
                            print(f"[Warning] Failed to load voice profile from {vprof_file}: {exc}")

        # 2. Legacy fallback index if exists
        if self.index_file.exists():
            try:
                data = json.loads(self.index_file.read_text(encoding="utf-8"))
                for vid, vdata in data.items():
                    if vid not in self.profiles:
                        raw_path = vdata.get("reference_audio_path", "")
                        if raw_path and not Path(raw_path).is_absolute():
                            resolved = settings.project_root / raw_path
                            if resolved.exists():
                                vdata["reference_audio_path"] = str(resolved)
                        self.profiles[vid] = VoiceProfile(
                            voice_id=vid,
                            name=vdata.get("name", vid),
                            reference_audio_path=vdata.get("reference_audio_path", ""),
                            reference_transcript=vdata.get("reference_transcript", ""),
                            language_code=vdata.get("language_code", "te"),
                            gender=vdata.get("gender", "female"),
                            sample_rate=vdata.get("sample_rate", 24000),
                        )
            except Exception as exc:
                print(f"[Warning] Failed to read voice index: {exc}")

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
        """Register or update a voice profile with one or more reference audio clips in a dedicated folder."""
        # Create persona-specific folder
        p_folder = self.personas_dir / voice_id
        p_folder.mkdir(parents=True, exist_ok=True)
        (p_folder / "reference_audio").mkdir(parents=True, exist_ok=True)

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

        # Save per-persona profile
        (p_folder / "voice_profile.json").write_text(
            json.dumps(profile.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        self._save_index()
        return profile

    def get_voice(self, voice_id: str) -> VoiceProfile | None:
        """Retrieve voice profile by ID with strict persona routing."""
        vid_lower = (voice_id or "").lower()

        # Direct match
        if voice_id in self.profiles:
            return self.profiles[voice_id]

        # Pappa routing
        if any(w in vid_lower for w in ("pappa", "father", "dad", "పప్పా", "నాన్న")):
            return self.profiles.get("pappa") or self.profiles.get("father")

        # Grandma routing
        if any(w in vid_lower for w in ("grandma", "amamma", "అమ్మమ్మ", "grandmother")):
            return self.profiles.get("grandma_chittoor") or self.profiles.get("grandma")

        # Generic lookup
        for key, prof in self.profiles.items():
            if key.lower() == vid_lower or vid_lower in key.lower():
                return prof

        return None

    def list_voices(self) -> list[VoiceProfile]:
        """List all available voice profiles."""
        return list(self.profiles.values())
