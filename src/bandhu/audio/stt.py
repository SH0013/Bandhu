"""Speech-to-Text recognition module supporting Telugu and multilingual audio."""

from __future__ import annotations

import io
from pathlib import Path

from bandhu.config import settings


class SpeechRecognizer:
    """Multilingual transcriber for incoming voice notes."""

    def __init__(self) -> None:
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize Google Cloud Speech-to-Text client if credentials are available."""
        try:
            from google.cloud import speech  # type: ignore[import]
            self._client = speech.SpeechClient()
        except Exception:
            self._client = None

    async def transcribe(self, audio_path_or_bytes: str | Path | bytes, language_code: str = "te-IN") -> str:
        """Transcribe audio to Telugu / English text."""
        if isinstance(audio_path_or_bytes, (str, Path)):
            path = Path(audio_path_or_bytes)
            if not path.exists():
                return ""
            audio_bytes = path.read_bytes()
        else:
            audio_bytes = audio_path_or_bytes

        if not audio_bytes:
            return ""

        if self._client is None:
            return ""

        try:
            from google.cloud import speech  # type: ignore[import]
            from google.cloud.speech import RecognitionConfig, RecognitionAudio  # type: ignore[import]

            audio = RecognitionAudio(content=audio_bytes)
            config = RecognitionConfig(
                encoding=RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=language_code,
                alternative_language_codes=["en-IN"],
                enable_automatic_punctuation=True,
                model="latest_long",
            )

            response = self._client.recognize(config=config, audio=audio)
            transcripts = [result.alternatives[0].transcript for result in response.results if result.alternatives]
            return " ".join(transcripts).strip()
        except Exception:
            return ""
