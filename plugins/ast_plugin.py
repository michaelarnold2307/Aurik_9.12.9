"""ast_plugin — AST (Audio Spectrogram Transformer) Plugin für Aurik.

AST (331M Parameter, Gong et al. 2021) ist ein Audio Spectrogram Transformer,
der auf AudioSet-527 für Audio-Tagging und Klassifikation trainiert wurde.

In Aurik dient AST als zentraler Klassifikator für:
  - PerceptualValidator — Goal-Validierung via AudioSet-Labels
  - DefectScanner — Defect-vs-Music-Discrimination (AST Pre-Filter)
  - EmotionalArcPreserver — Mood-Guided Mastering
  - Phase_53 SemanticAudio — Instrument/Genre-Tagging
  - EraClassifier — Era-Indikatoren via Instrumenten-Detektion

Modell:
    models/ast/ast_model.onnx (~294 KB) + ast_model.onnx.data (~346 MB)
    Input:  [batch, time, 128] float32 (Mel-Spectrogram @ 16 kHz)
    Output: [batch, 527] float32 (Sigmoid AudioSet Scores)

Dieses Plugin ist ein Thin-Wrapper um den zentralen AstAudioSetClassifier
(backend/core/ast_audio_set_classifier.py), der die ONNX-Inference und
das Lifecycle-Management übernimmt. Das Plugin stellt die standardisierte
get_ast_plugin() / get_loaded_ast_plugin() Schnittstelle bereit, die mit
dem Plugin-System (plugin_registry.py, plugin_lifecycle_manager.py)
kompatibel ist.

Spec §v10.304: Zentraler AST AudioSet-527 Classifier Hub.
Privacy: Kein Audio verlässt den Prozess. Reine ONNX-Inference.

Singleton-Pattern: get_ast_plugin() verwenden.
CPU-Only: CPUExecutionProvider.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_instance: AstPlugin | None = None


@dataclass
class AstResult:
    """Ergebnis der AST AudioSet-Klassifikation.

    Attributes:
        labels:     Liste von (label_name, confidence ∈ [0,1]) Tupeln
        top_k:      Top-K Labels sortiert nach Konfidenz
        embeddings: AST-Feature-Embedding (sofern vom Modell bereitgestellt)
        model_used: "ast_onnx" | "ast_unloaded"
    """

    labels: list[tuple[str, float]]
    top_k: list[tuple[str, float]]
    embeddings: np.ndarray
    model_used: str
    raw_scores: np.ndarray = field(default_factory=lambda: np.zeros(527, dtype=np.float32))


class AstPlugin:
    """AST Audio Spectrogram Transformer — Plugin-Wrapper für Aurik.

    Delegiert an den zentralen AstAudioSetClassifier (Singleton) in
    backend/core/ast_audio_set_classifier.py. Stellt die standardisierte
    get_*_plugin()-Schnittstelle bereit.

    Verwendung:
        plugin = get_ast_plugin()
        if plugin.is_loaded():
            result = plugin.classify(audio, sr=48000, top_k=15)
    """

    def __init__(self) -> None:
        self._classifier: Any = None
        self._classifier_loaded: bool = False
        self._init_classifier()

    def _init_classifier(self) -> None:
        """Lazy-Init: Bindet den zentralen AstAudioSetClassifier ein."""
        self._get_classifier: Any = None
        self._is_loaded_fn: Any = None
        try:
            from backend.core.ast_audio_set_classifier import (
                get_ast_classifier,
                is_ast_loaded,
            )

            self._get_classifier = get_ast_classifier
            self._is_loaded_fn = is_ast_loaded
            self._classifier_loaded = is_ast_loaded()
        except Exception as exc:
            logger.warning("AST Plugin: ast_audio_set_classifier nicht verfügbar: %s", exc)
            self._get_classifier = None
            self._is_loaded_fn = None
            self._classifier_loaded = False

    def is_loaded(self) -> bool:
        """True wenn AST ONNX geladen und bereit (non-invasiver Peek).

        Ruft KEINEN Lazy-Load aus — nur Statusabfrage des bereits
        existierenden AstAudioSetClassifier Singletons.
        """
        if self._is_loaded_fn is not None:
            try:
                return bool(self._is_loaded_fn())
            except Exception:
                return False
        return self._classifier_loaded

    def classify(
        self,
        audio: np.ndarray,
        sr: int = 48000,
        top_k: int = 15,
    ) -> AstResult:
        """Klassifiziert Audio via AST ONNX → AudioSet-527 Scores.

        Args:
            audio: float32 mono/stereo, 48000 Hz
            sr:    Sample-Rate (muss 48000 sein)
            top_k: Anzahl Top-K-Labels im Ergebnis

        Returns:
            AstResult mit Labels, Top-K, Embeddings.

        Raises:
            RuntimeError wenn AST Classifier nicht geladen ist.
        """
        if self._get_classifier is None:
            raise RuntimeError("AST Classifier nicht verfügbar — ast_audio_set_classifier fehlt")

        classifier = self._get_classifier()
        if not classifier.is_loaded():
            raise RuntimeError("AST ONNX Modell nicht geladen")

        # AstAudioSetClassifier.classify() gibt (labels, scores) zurück
        # wobei labels = [(name, conf), ...]
        result = classifier.classify(audio, sr, top_k=top_k)
        if result is None:
            return AstResult(
                labels=[],
                top_k=[],
                embeddings=np.zeros(768, dtype=np.float32),
                model_used="ast_unloaded",
            )

        # Bug-Reparatur: classifier.classify() liefert ein AstResult-Objekt
        # (top_k: list[tuple[int, str, float]]), KEIN (labels, scores)-Tupel.
        _labels = [(str(name), float(conf)) for _, name, conf in getattr(result, "top_k", [])]
        _top = _labels[:top_k]

        return AstResult(
            labels=_labels,
            top_k=_top,
            embeddings=np.zeros(768, dtype=np.float32),
            model_used="ast_onnx",
            raw_scores=np.asarray(getattr(result, "probs", np.zeros(527, dtype=np.float32)), dtype=np.float32),
        )

    def get_tags(self, audio: np.ndarray, sr: int = 48000, top_k: int = 15) -> dict[str, float]:
        """Convenience: Gibt Labels als {name: confidence} Dict zurück.

        Kompatibel mit BeatsPlugin.get_tags() / PannsPlugin.get_tags().
        """
        result = self.classify(audio, sr, top_k=top_k)
        return {label: conf for label, conf in result.labels}

    def discriminate_defect(
        self,
        defect_type: str,
        audio: np.ndarray,
        sr: int = 48000,
    ) -> float:
        """Prüft ob ein Defekt-Typ mit einem Musikinstrument kollidiert.

        Gibt die maximale Instrument-Konfidenz (0–1) für den gegebenen
        Defekt-Typ zurück. Werte ≥ 0.15 deuten auf eine Kollision hin
        (Defekt klingt wie ein Instrument → PRESERVE statt REPAIR).

        Delegiert an AstAudioSetClassifier.discriminate_defect().
        """
        if self._get_classifier is None:
            return 0.0
        classifier = self._get_classifier()
        if not classifier.is_loaded():
            return 0.0
        try:
            _disc = classifier.discriminate_defect(defect_type, audio, sr, time_s=0.0, severity=0.5)
            if _disc is None:
                return 0.0
            return float(_disc.instrument_confidence)
        except Exception:
            logger.warning("§V6 ML→DSP-Fallback: get_ast_instrument_confidence fehlgeschlagen → neutraler Return (0.0)")
            return 0.0

    def get_ast_musical_confidence(self, audio: np.ndarray, sr: int = 48000) -> float:
        """Berechnet die aggregierte 'Musical Confidence' über alle
        Musik-Instrument-Klassen hinweg. Verwendet von TapeHeadRepair,
        EchoRemoval und Azimuth-Guard zur Reduktion von False Positives.

        Returns: float ∈ [0, 1]
        """
        if self._get_classifier is None:
            return 0.0
        classifier = self._get_classifier()
        if not classifier.is_loaded():
            return 0.0
        try:
            result = classifier.classify(audio, sr, top_k=50)
            if result is None:
                return 0.0
            # Bug-Reparatur: classify() liefert AstResult mit .probs
            scores = np.asarray(getattr(result, "probs", np.zeros(527, dtype=np.float32)), dtype=np.float32)
            # Musik-Instrument-Indizes aus AudioSet (grobe Abdeckung)
            musical_indices = set(range(120, 250))  # Instruments-Bereich
            musical_conf = 0.0
            for idx in musical_indices:
                if idx < len(scores):
                    musical_conf = max(musical_conf, float(scores[idx]))
            return float(np.clip(musical_conf, 0.0, 1.0))
        except Exception:
            logger.warning("§V6 ML→DSP-Fallback: get_ast_musical_confidence fehlgeschlagen → neutraler Return (0.0)")
            return 0.0


# ---------------------------------------------------------------------------
# Singleton (§3.2 Double-Checked Locking)
# ---------------------------------------------------------------------------


def get_ast_plugin() -> AstPlugin:
    """Thread-sicherer Singleton-Accessor für das AST Plugin.

    Konform mit der plugins/*_plugin.py Konvention (analog zu
    get_beats_plugin(), get_mert_plugin(), get_demucs_plugin(), etc.).
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = AstPlugin()
    return _instance


def get_loaded_ast_plugin() -> AstPlugin | None:
    """Gibt das AST Plugin nur zurück wenn es bereits geladen ist.

    Non-invasiver Peek — KEIN Lazy-Load. Verwendet für
    ml_model_readiness-Checks ohne Deadlock-Risiko.
    """
    global _instance
    if _instance is not None and _instance.is_loaded():
        return _instance
    return None


def ast_classify(audio: np.ndarray, sr: int = 48000, top_k: int = 15) -> AstResult:
    """Convenience-Wrapper für get_ast_plugin().classify()."""
    return get_ast_plugin().classify(audio, sr, top_k=top_k)


def ast_get_tags(audio: np.ndarray, sr: int = 48000, top_k: int = 15) -> dict[str, float]:
    """Convenience-Wrapper für get_ast_plugin().get_tags()."""
    return get_ast_plugin().get_tags(audio, sr, top_k=top_k)
