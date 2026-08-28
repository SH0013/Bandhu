"""WhatsApp chat export (.txt) parser supporting multiple mobile OS formats."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TextIO

from bandhu.persona.models import SpeakerTurn


class WhatsAppChatParser:
    """Robust parser for exported WhatsApp .txt transcripts across iOS and Android formats."""

    # Regex patterns for iOS and Android WhatsApp exports
    # iOS: [22/08/23, 14:30:15] Person: Message or [22/8/23, 2:30:15 PM] Person: Message
    IOS_PATTERN = re.compile(
        r"^\[(?P<timestamp>\d{1,4}[/\.\-]\d{1,2}[/\.\-]\d{1,4},?\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]\s+(?P<speaker>[^:]+?):\s+(?P<text>.*)$"
    )

    # Android: 22/08/2023, 14:30 - Person: Message or 22/8/23, 2:30 pm - Person: Message
    ANDROID_PATTERN = re.compile(
        r"^(?P<timestamp>\d{1,4}[/\.\-]\d{1,2}[/\.\-]\d{1,4},?\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\s+-\s+(?P<speaker>[^:]+?):\s+(?P<text>.*)$"
    )

    # Flexible generic fallback pattern
    FALLBACK_PATTERN = re.compile(
        r"^\[?(?P<timestamp>\d{1,4}[/\.\-]\d{1,2}[/\.\-]\d{1,4},?\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*[-:]?\s*(?P<speaker>[^:\n]{1,60}?):\s+(?P<text>.*)$"
    )

    OMITTED_MEDIA_PATTERNS = (
        "<media omitted>",
        "<attached:",
        "image omitted",
        "video omitted",
        "audio omitted",
        "sticker omitted",
        "document omitted",
        "this message was deleted",
        "you deleted this message",
        "end-to-end encrypted",
        "messages and calls are end-to-end encrypted",
    )

    @classmethod
    def parse_file(cls, file_path: str | Path) -> list[SpeakerTurn]:
        """Parse chat log from file path."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"WhatsApp export file not found: {file_path}")

        content = path.read_text(encoding="utf-8", errors="replace")
        return cls.parse_text(content)

    @classmethod
    def parse_text(cls, text: str) -> list[SpeakerTurn]:
        """Parse raw text string containing chat export lines."""
        turns: list[SpeakerTurn] = []
        # Normalize invisible directional characters and narrow spaces
        sanitized_text = (
            text.replace("\u200e", "")
            .replace("\u200f", "")
            .replace("\u202a", "")
            .replace("\u202c", "")
            .replace("\u202f", " ")
            .replace("\xa0", " ")
        )
        lines = sanitized_text.splitlines()

        current_speaker: str | None = None
        current_timestamp: str = ""
        current_message_lines: list[str] = []

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            match = (
                cls.IOS_PATTERN.match(line_str)
                or cls.ANDROID_PATTERN.match(line_str)
                or cls.FALLBACK_PATTERN.match(line_str)
            )
            if match:
                # Flush previous message turn if exists
                if current_speaker and current_message_lines:
                    full_text = " ".join(current_message_lines).strip()
                    if not cls._is_omitted(full_text):
                        turns.append(
                            SpeakerTurn(
                                speaker=current_speaker,
                                text=full_text,
                                timestamp=current_timestamp,
                            )
                        )
                current_speaker = match.group("speaker").strip()
                current_timestamp = match.group("timestamp").strip()
                current_message_lines = [match.group("text").strip()]
            else:
                # Continuation of previous multi-line message
                if current_speaker is not None:
                    current_message_lines.append(line_str)

        # Flush final message
        if current_speaker and current_message_lines:
            full_text = " ".join(current_message_lines).strip()
            if not cls._is_omitted(full_text):
                turns.append(
                    SpeakerTurn(
                        speaker=current_speaker,
                        text=full_text,
                        timestamp=current_timestamp,
                    )
                )

        return turns

    @classmethod
    def get_speakers(cls, turns: list[SpeakerTurn]) -> dict[str, int]:
        """Return speaker names mapped to their message counts."""
        speakers: dict[str, int] = {}
        for turn in turns:
            speakers[turn.speaker] = speakers.get(turn.speaker, 0) + 1
        return speakers

    @classmethod
    def resolve_target_speaker(cls, turns: list[SpeakerTurn], target: str) -> str:
        """Find the matching speaker name from turns using exact or fuzzy match."""
        speakers = cls.get_speakers(turns)
        if not speakers:
            return target

        target_clean = target.strip().lower()

        # 1. Exact match
        for s in speakers:
            if s.strip().lower() == target_clean:
                return s

        # 2. Substring match (e.g. "Pappa" in "Pappa ❤️" or "Pappa (Home)")
        for s in speakers:
            if target_clean in s.lower() or s.lower() in target_clean:
                return s

        # 3. If 2 speakers (1-on-1 chat), pick the other speaker that is not 'you' or 'me'
        non_user_speakers = [
            s for s in speakers if s.lower() not in ("you", "me", "నేను", "self")
        ]
        if len(non_user_speakers) == 1:
            return non_user_speakers[0]

        # 4. Fallback to the speaker with most messages
        return max(speakers.items(), key=lambda x: x[1])[0]

    @classmethod
    def extract_speaker_turns(
        cls, turns: list[SpeakerTurn], target_speaker: str
    ) -> list[SpeakerTurn]:
        """Filter turns spoken exclusively by the target persona."""
        resolved = cls.resolve_target_speaker(turns, target_speaker)
        return [
            t for t in turns if t.speaker.lower() == resolved.lower()
        ]

    @classmethod
    def _is_omitted(cls, text: str) -> bool:
        """Check if message is placeholder/system noise."""
        lower = text.lower()
        return any(omitted in lower for omitted in cls.OMITTED_MEDIA_PATTERNS)
