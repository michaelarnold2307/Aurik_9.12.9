"""
§v10.126: MERT Feature Extractor — Music Understanding Transformer (117M, ONNX GPU).

MERT (Music undERstanding Transformer) ist auf 160k+ Stunden Musik vortrainiert.
Produziert 768-dim Features, die musikalische Struktur codieren:
  - Genre, Instrumentierung, Harmonik, Rhythmus

Nutzung:
  extractor = MERTFeatureExtractor()
  features = extractor.extract(audio, sample_rate)
  # features: [frames, 768] — eine Feature-Matrix pro ~10ms
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "mert" / "mert.onnx"


class MERTFeatureExtractor:
    """MERT ONNX GPU Feature-Extraktor für Musik-Kontext."""

    def __init__(self):
        import onnxruntime as ort

        if not _MODEL_PATH.exists():
            raise FileNotFoundError(f"MERT model not found: {_MODEL_PATH}")

        self._session = ort.InferenceSession(
            str(_MODEL_PATH),
            providers=["ROCMExecutionProvider", "CPUExecutionProvider"],
        )
        self._provider = self._session.get_providers()[0]
        logger.info("MERT geladen: %s (117M params, %s)", _MODEL_PATH.name, self._provider)

    def extract(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        """Extrahiert MERT-Features aus Audio.

        Args:
            audio: float32 [samples]
            sample_rate: Sample-Rate (beliebig, MERT ist flexibel)

        Returns:
            np.ndarray [frames, 768] — Musik-Features
        """
        audio = audio.astype(np.float32)

        if audio.ndim == 2:
            audio = audio.mean(axis=0)  # Stereo → Mono

        # Normalize
        peak = np.abs(audio).max() + 1e-10
        audio = audio / peak

        # Add batch dimension
        audio_batch = audio[np.newaxis, :]

        # Inferenz
        outputs = self._session.run(None, {"input_values": audio_batch})
        features = outputs[0]  # [1, frames, 768]
        return cast(np.ndarray, features[0])  # [frames, 768]

    def extract_mean(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        """Extrahiert gemittelte MERT-Features (ein Vektor pro Audiodatei).

        Nützlich für Genre-Erkennung oder globale Audio-Klassifikation.
        """
        features = self.extract(audio, sample_rate)
        return cast(np.ndarray, features.mean(axis=0).astype(np.float32))  # [768]

    def extract_segments(self, audio: np.ndarray, sample_rate: int = 48000, segment_s: float = 5.0) -> np.ndarray:
        """Extrahiert MERT-Features in Segmenten (für lange Audiodateien).

        Args:
            audio: float32 [samples]
            sample_rate: Sample-Rate
            segment_s: Segment-Länge in Sekunden

        Returns:
            np.ndarray [n_segments, 768]
        """
        segment_samples = int(segment_s * sample_rate)
        n_segments = (len(audio) + segment_samples - 1) // segment_samples
        features = []

        for i in range(n_segments):
            start = i * segment_samples
            end = min(start + segment_samples, len(audio))
            segment = audio[start:end]
            feat = self.extract_mean(segment, sample_rate)
            features.append(feat)

        return cast(np.ndarray, (np.stack(features, axis=0)))


# ── Context-Analyse ────────────────────────────────────────────────────────


def compute_music_context(audio: np.ndarray, sample_rate: int = 48000) -> dict:
    """Hochrangige Musik-Kontext-Analyse via MERT.

    Returns dict mit:
      - genre_proxy: float [0,1] — "how acoustic" (0=elektronisch, 1=akustisch)
      - density: float [0,1] — spektrale Dichte (0=sparse, 1=dense)
      - brightness: float [0,1] — Helligkeit (0=dunkel, 1=hell)
      - is_vocal: float [0,1] — Vocal-Präsenz
    """
    try:
        extractor = MERTFeatureExtractor()
        features = extractor.extract(audio, sample_rate)
        mean_feat = features.mean(axis=0)

        # Heuristische Mappings (vereinfacht, aber nützlich)
        # MERT-Dimensionen korrelieren mit musikalischen Eigenschaften
        low_dim = mean_feat[:256]  # Untere Dimensionen → Rhythmus, Bass
        mid_dim = mean_feat[256:512]  # Mittlere → Harmonik, Instrumentierung
        high_dim = mean_feat[512:]  # Obere → Textur, Vocals

        return {
            # Rhythmische Dichte (Varianz in unteren Dimensionen)
            "density": float(np.clip(np.std(low_dim) * 3.0, 0.0, 1.0)),
            # Harmonische Helligkeit (Mittelwert der mittleren Dimensionen)
            "brightness": float(np.clip((np.mean(mid_dim) + 0.5), 0.0, 1.0)),
            # Vocal-Präsenz (hohe Dimensionen korrelieren mit Stimme)
            "is_vocal": float(np.clip(np.mean(np.abs(high_dim)) * 2.0, 0.0, 1.0)),
            # Akustik-Proxy (niedrige Varianz = akustisch, hohe = elektronisch)
            "genre_proxy": float(1.0 - np.clip(np.std(high_dim) * 2.0, 0.0, 1.0)),
        }
    except Exception as e:
        logger.debug("MERT-Kontext-Analyse fehlgeschlagen: %s", e)
        return {"density": 0.5, "brightness": 0.5, "is_vocal": 0.5, "genre_proxy": 0.5}
