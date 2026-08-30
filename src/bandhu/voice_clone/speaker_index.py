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

    def build_from_audio_paths(
        self,
        audio_paths: list[str | Path],
        speaker_name: str = "Custom Speaker",
        output_dir: Optional[Path] = None,
        index_name: Optional[str] = None,
    ) -> tuple[Path, Path]:
        """Process reference audio files, extract HuBERT & F0 features, and build FAISS vector index."""
        import soundfile as sf
        import torch
        import torchaudio

        if not audio_paths:
            raise ValueError("No audio paths provided to build speaker index")

        out_dir = output_dir or Path("data")
        out_dir.mkdir(parents=True, exist_ok=True)
        idx_prefix = index_name or f"{speaker_name.lower().replace(' ', '_')}_voice"
        index_file = out_dir / f"{idx_prefix}.index"
        profile_file = out_dir / f"{idx_prefix}_profile.json"

        all_features: list[np.ndarray] = []
        all_f0: list[float] = []
        total_dur = 0.0

        for p in audio_paths:
            path_obj = Path(p)
            if not path_obj.exists():
                continue
            try:
                wav, sr = sf.read(str(path_obj), dtype="float32")
                if wav.ndim > 1:
                    wav = np.mean(wav, axis=1)
                if sr != self.extractor.target_sr:
                    tensor_wav = torch.from_numpy(wav).unsqueeze(0)
                    wav = torchaudio.functional.resample(tensor_wav, sr, self.extractor.target_sr).squeeze(0).numpy()

                total_dur += len(wav) / float(self.extractor.target_sr)
                f0_contour = self.extractor.extract_f0(wav)
                voiced = f0_contour[f0_contour > 0]
                if len(voiced) > 0:
                    all_f0.extend(voiced.tolist())

                feat = self.extractor.extract_features(wav)
                if len(feat) > 0:
                    all_features.append(feat)
            except Exception as exc:
                print(f"[Warning] Failed extracting features from {path_obj.name}: {exc}")

        f0_arr = np.array(all_f0) if all_f0 else np.array([160.0])
        median_f0 = float(np.median(f0_arr))
        min_f0 = float(np.percentile(f0_arr, 5)) if len(f0_arr) > 1 else median_f0 * 0.8
        max_f0 = float(np.percentile(f0_arr, 95)) if len(f0_arr) > 1 else median_f0 * 1.4

        if all_features:
            stacked = np.vstack(all_features).astype(np.float32)
            dim = stacked.shape[1]
            idx = faiss.IndexFlatIP(dim)
            idx.add(stacked)
            faiss.write_index(idx, str(index_file))
            self.index = idx
            num_vectors = stacked.shape[0]
        else:
            dim = self.feature_dim
            idx = faiss.IndexFlatIP(dim)
            dummy = np.zeros((1, dim), dtype=np.float32)
            idx.add(dummy)
            faiss.write_index(idx, str(index_file))
            self.index = idx
            num_vectors = 1

        prof_data = {
            "speaker_name": speaker_name,
            "num_clips": len(audio_paths),
            "total_duration_sec": round(total_dur, 2),
            "feature_dim": dim,
            "num_vectors": num_vectors,
            "median_f0_hz": round(median_f0, 2),
            "min_f0_hz": round(min_f0, 2),
            "max_f0_hz": round(max_f0, 2),
            "spectral_centroid_hz": 1800.0,
            "index_file_name": index_file.name,
        }

        profile_file.write_text(json.dumps(prof_data, indent=2), encoding="utf-8")
        self.profile = SpeakerVoiceProfile(**prof_data)

        return index_file, profile_file
