"""Speaker acoustic indexing and FAISS vector database loader for voice cloning."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from bandhu.voice_clone.feature_extractor import AcousticFeatureExtractor


@dataclass
class SpeakerVoiceProfile:
    """Acoustic statistics and metadata for the indexed speaker."""

    speaker_name: str
    num_clips: int
    total_duration_sec: float
    feature_dim: int
    num_vectors: int
    median_f0_hz: float
    min_f0_hz: float
    max_f0_hz: float
    spectral_centroid_hz: float
    index_file_name: str


class SpeakerIndexBuilder:
    """Builds and loads FAISS vector index of speaker acoustic embeddings."""

    def __init__(
        self,
        extractor: Optional[AcousticFeatureExtractor] = None,
        feature_dim: int = 768,
    ) -> None:
        """Initialize SpeakerIndexBuilder."""
        self.extractor = extractor or AcousticFeatureExtractor()
        self.feature_dim = feature_dim
        self.index: Optional[faiss.IndexFlatIP] = None
        self.profile: Optional[SpeakerVoiceProfile] = None

    def load(self, index_path: Path, profile_path: Optional[Path] = None) -> None:
        """Load pre-built FAISS index and speaker profile."""
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")
        self.index = faiss.read_index(str(index_path))

        if profile_path and profile_path.exists():
            data = json.loads(profile_path.read_text(encoding="utf-8"))
            self.profile = SpeakerVoiceProfile(**data)
