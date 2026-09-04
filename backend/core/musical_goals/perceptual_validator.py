"""
Perceptual Validation System für Musical Goals.

Validiert technische Musical Goals Measurements gegen psychoakustische
Wahrnehmung durch ML-Modell und sammelt A/B-Test-Daten für kontinuierliche
Verbesserung.

Component 0.9.2: Perceptual Validation System
Impact: +1.5 Punkte - Perceptual Validation Guarantee
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None  # type: ignore[assignment]

torch = None  # type: ignore[assignment]
AutoFeatureExtractor = None  # type: ignore[assignment]
AutoModelForAudioClassification = None  # type: ignore[assignment]


def _load_torch_stack() -> bool:
    """Lädt optional torch/transformers dependencies only when needed."""
    global torch, AutoFeatureExtractor, AutoModelForAudioClassification

    if torch is not None and AutoFeatureExtractor is not None and AutoModelForAudioClassification is not None:
        return True

    try:
        import torch as _torch
        from transformers import (
            AutoFeatureExtractor as _AutoFeatureExtractor,
        )
        from transformers import (
            AutoModelForAudioClassification as _AutoModelForAudioClassification,
        )

        torch = _torch  # type: ignore[assignment]
        AutoFeatureExtractor = _AutoFeatureExtractor  # type: ignore[assignment]
        AutoModelForAudioClassification = _AutoModelForAudioClassification  # type: ignore[assignment]
        return True
    except (ImportError, OSError, Warning) as exc:
        logger.debug("§V6 AST-Perceptual-ONNX Import fehlgeschlagen — False zurückgegeben (torch/transformers): %s", exc)
        # Warning is included because pytest may escalate third-party deprecation
        # warnings to exceptions during optional dependency imports.
        return False


logger = logging.getLogger(__name__)

# Lokales AST-Modell-Verzeichnis — §13.3 bundled:true, kein HF-Download
_AST_LOCAL_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent / "models" / "ast_perceptual_base"
_AST_ONNX_PATH: Path = Path(__file__).resolve().parent.parent.parent.parent / "models" / "ast" / "ast_model.onnx"


@dataclass
class PerceptualScore:
    """
    Perceptual validation score für ein Musical Goal.

    Attributes:
        technical_score: Original technischer Score (0-1)
        psychoacoustic_score: Predicted perceptual score (0-1)
        confidence: Confidence des psychoacoustic models (0-1)
        adjusted_score: Gewichteter final score (0-1)
        requires_human: Ob menschliche validation empfohlen wird
    """

    technical_score: float
    psychoacoustic_score: float
    confidence: float
    adjusted_score: float
    requires_human: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ListeningTestRequest:
    """Request für menschliches Listening Test."""

    session_id: str
    audio_path: str
    goal_scores: dict[str, float]
    confidence_scores: dict[str, float]
    reason: str
    priority: str  # 'high', 'medium', 'low'
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ABTestSample:
    """A/B Test Sample für Training Data Collection."""

    sample_id: str
    audio_a_path: str
    audio_b_path: str
    goal: str
    score_a: float
    score_b: float
    predicted_preference: str  # 'A' or 'B'
    confidence: float
    collected_at: datetime = field(default_factory=datetime.now)
    human_preference: str | None = None  # Filled after human evaluation


class PerceptualValidator:
    """
    Validates Musical Goals gegen psychoakustische Wahrnehmung.

    Features:
    - Psychoakustisches ML-Modell (AST from HuggingFace)
    - Confidence Scoring für alle 7 Goals
    - A/B Test Data Collection
    - Listening Test Requirement Logic
    - Continuous Learning from Human Feedback

    Workflow:
    1. Technical Score berechnen (MusicalGoalsChecker)
    2. Psychoacoustic Score predicten (AST Model)
    3. Confidence Score berechnen
    4. Weighted Final Score berechnen (70% technical, 30% psychoacoustic)
    5. Entscheiden ob Listening Test nötig
    6. A/B Test Samples sammeln für Retraining
    """

    _GOAL_MAPPINGS: dict[str, list[int]] = {
        # §v10.304: Korrigierte AudioSet-527-Indizes (vorher: Speech-Klassen für Bass-Kraft).
        # Referenz: AudioSet-Ontologie (Gemmeke et al. 2017), validiert gegen
        # ast_audio_set_classifier.py _AUDIOSET_CLASS_NAMES.
        #
        # bass-kraft:     Low-frequency instruments → sub-bass energy
        "bass-kraft": [10, 73, 413],  # Bass drum, Double bass, Bass guitar
        # brillanz:        High-frequency percussive → presence/clarity
        "brillanz": [310, 311, 346],  # Hi-hat, Cymbal, Tambourine
        # waerme:          Mid-low harmonic instruments → warmth
        "waerme": [137, 363, 364],  # Cello, String section, Orchestra
        # natuerlichkeit:  Musical content (nicht Speech) → naturalness
        "natuerlichkeit": [15, 407, 137],  # Music, Singing, Cello
        # authentizitaet:  Source/genre authenticity markers
        "authentizitaet": [15, 28, 37],  # Music, Musical instrument, Independent music
        # emotionalitaet:  Mood classes → emotional impact
        "emotionalitaet": [399, 400, 401],  # Happy music, Sad music, Exciting music
        # transparenz:     High-frequency clarity + vocal presence
        "transparenz": [15, 407, 310],  # Music, Singing, Hi-hat
    }

    # §v10.304: Genre-adaptive Goal-Mappings.
    # Jazz → andere Referenz-Instrumente als Metal oder Klassik.
    _GENRE_GOAL_MAPPINGS: dict[str, dict[str, list[int]]] = {
        "jazz": {
            "bass-kraft": [73, 137, 10],  # Double bass, Cello, Bass drum
            "brillanz": [310, 196, 138],  # Hi-hat, Flute, Bell
            "waerme": [141, 137, 363],  # Piano, Cello, String section
            "emotionalitaet": [401, 403, 400],  # Sad, Tender, Happy
        },
        "classical_music": {
            "bass-kraft": [137, 73, 363],  # Cello, Double bass, String section
            "brillanz": [196, 138, 311],  # Flute, Bell, Cymbal
            "waerme": [363, 137, 141],  # String section, Cello, Piano
            "emotionalitaet": [403, 401, 400],  # Tender, Sad, Happy
        },
        "rock_music": {
            "bass-kraft": [10, 413, 143],  # Bass drum, Bass guitar, Electric guitar
            "brillanz": [311, 310, 143],  # Cymbal, Hi-hat, Electric guitar
            "waerme": [143, 141, 10],  # Electric guitar, Piano, Bass drum
            "emotionalitaet": [404, 400, 401],  # Exciting, Happy, Sad
        },
        "pop_music": {
            "bass-kraft": [413, 10, 141],  # Bass guitar, Bass drum, Piano
            "brillanz": [310, 311, 407],  # Hi-hat, Cymbal, Singing
            "waerme": [141, 407, 137],  # Piano, Singing, Cello
            "emotionalitaet": [400, 404, 401],  # Happy, Exciting, Sad
        },
        "electronic_music": {
            "bass-kraft": [10, 413, 299],  # Bass drum, Bass guitar, Synthesizer
            "brillanz": [311, 310, 299],  # Cymbal, Hi-hat, Synthesizer
            "waerme": [299, 141, 137],  # Synthesizer, Piano, Cello
            "emotionalitaet": [404, 400, 401],  # Exciting, Happy, Sad
        },
        "hip_hop": {
            "bass-kraft": [10, 413, 104],  # Bass drum, Bass guitar, Bass
            "brillanz": [310, 311, 407],  # Hi-hat, Cymbal, Singing
            "waerme": [407, 141, 143],  # Singing, Piano, Electric guitar
            "emotionalitaet": [404, 400, 357],  # Exciting, Happy, Hip hop
        },
    }

    def __init__(
        self,
        model_name: str = str(_AST_LOCAL_DIR),
        confidence_threshold: float = 0.7,
        ab_test_collection_rate: float = 0.1,
        ab_test_storage_path: Path | None = None,
    ):
        """
        Initialize Perceptual Validator.

        Args:
            model_name: Lokaler Pfad zum AST-Modell (models/ast_perceptual_base/)
            confidence_threshold: Minimum confidence für automatische validation
            ab_test_collection_rate: Fraction of samples für A/B testing (0-1)
            ab_test_storage_path: Path für A/B test data storage
        """
        self.confidence_threshold = confidence_threshold
        self.ab_test_collection_rate = ab_test_collection_rate
        self.ab_test_storage_path = ab_test_storage_path or Path("data/ab_tests")
        self.ab_test_storage_path.mkdir(parents=True, exist_ok=True)
        self.onnx_session = None
        self._onnx_input_name: str | None = None
        self._onnx_output_name: str | None = None
        self.feature_extractor = None
        self.model = None

        # Preferred local model path:
        # 0) §v10.304: Zentraler AST Classifier (ast_audio_set_classifier)
        # 1) ONNX bundle at models/ast/ast_model.onnx
        # 2) HF local directory at models/ast_perceptual_base
        _central_loaded = self._try_load_central_classifier()
        if _central_loaded or self._try_load_onnx_model():
            self.device = None
        else:
            self.device = torch.device("cpu") if _load_torch_stack() and torch is not None else None
            self._try_load_hf_model(model_name)

        # Statistics
        self.validation_count = 0
        self.listening_test_requests: list[ListeningTestRequest] = []
        self.ab_test_samples: list[ABTestSample] = []

    def _try_load_central_classifier(self) -> bool:
        """§v10.304: Versucht den zentralen AST Classifier zu verwenden.

        Wenn der AstAudioSetClassifier bereits geladen ist, wird seine
        Inference-Session wiederverwendet — keine zweite Modell-Instanz.
        """
        try:
            from backend.core.ast_audio_set_classifier import get_ast_classifier

            _clf = get_ast_classifier()
            if _clf.is_loaded():
                self.onnx_session = _clf._session  # type: ignore[assignment]
                self._onnx_input_name = _clf._input_name
                self._onnx_output_name = _clf._output_name
                self._ast_classifier = _clf  # Referenz für goal-spezifische Methoden
                logger.info("PerceptualValidator: verwende zentralen AST Classifier")
                return True
        except Exception as exc:
            logger.debug("Zentraler AST Classifier nicht verfügbar: %s", exc)
        return False

    def _try_load_onnx_model(self) -> bool:
        """Try loading bundled AST ONNX model (models/ast/ast_model.onnx)."""
        if not _AST_ONNX_PATH.is_file():
            return False
        try:
            from backend.core.ml_memory_budget import release, try_allocate
            from backend.core.plugin_lifecycle_manager import register_plugin

            if not try_allocate("ASTPerceptualONNX", 0.35):
                logger.warning("AST ONNX wurde wegen ML-Budgetlimit nicht geladen — DSP-Ersatzpfad aktiviert")
                return False

            try:
                import onnxruntime as ort

                sess_opts = ort.SessionOptions()
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_opts.intra_op_num_threads = max(1, os.cpu_count() or 1)
                sess_opts.inter_op_num_threads = 1
                session = ort.InferenceSession(
                    str(_AST_ONNX_PATH),
                    sess_options=sess_opts,
                    providers=["CPUExecutionProvider"],
                )
                self.onnx_session = session
                self._onnx_input_name = session.get_inputs()[0].name
                self._onnx_output_name = session.get_outputs()[0].name

                def _unload() -> None:
                    self.onnx_session = None
                    self._onnx_input_name = None
                    self._onnx_output_name = None
                    release("ASTPerceptualONNX")

                register_plugin("ASTPerceptualONNX", 0.35, _unload)
                logger.info("AST ONNX Modell geladen: %s", _AST_ONNX_PATH)
                return True
            except Exception:
                release("ASTPerceptualONNX")
                raise
        except Exception as exc:
            logger.debug("AST ONNX nicht verfügbar: %s", exc)
            self.onnx_session = None
            self._onnx_input_name = None
            self._onnx_output_name = None
            return False

    def _try_load_hf_model(self, model_name: str) -> None:
        """Fallback to local HF directory model if ONNX is not available."""
        # Psychoakustisches Modell laden — §13.3 Invariante: ausschließlich lokal (kein HF-Hub)
        # TRANSFORMERS_OFFLINE + HF_HUB_OFFLINE: blockieren Netzwerkzugriff in _load_hf_model()
        # Thread + 8 s Timeout + shutdown(wait=False): schützt gegen stale file locks
        import concurrent.futures as _cf
        import os as _os

        try:
            if not _load_torch_stack() or torch is None:
                raise ImportError("torch nicht verfügbar — DSP-Fallback aktiviert")
            # Lokales Modell-Verzeichnis prüfen — kein HF-Hub-Download (§13.3)
            if not Path(model_name).is_dir():
                raise OSError(f"Modell-Verzeichnis {model_name!r} nicht gefunden — DSP-Fallback aktiviert")

            from backend.core.ml_memory_budget import release, try_allocate
            from backend.core.plugin_lifecycle_manager import register_plugin

            if not try_allocate("ASTPerceptualHF", 0.35):
                logger.warning("AST HF wurde wegen ML-Budgetlimit nicht geladen — DSP-Ersatzpfad aktiviert")
                return

            def _load_hf_model() -> tuple:
                assert (
                    AutoFeatureExtractor is not None
                    and AutoModelForAudioClassification is not None
                    and torch is not None
                )
                _prev = _os.environ.get("TRANSFORMERS_OFFLINE", "")
                _prev_hf = _os.environ.get("HF_HUB_OFFLINE", "")
                _os.environ["TRANSFORMERS_OFFLINE"] = "1"
                _os.environ["HF_HUB_OFFLINE"] = "1"  # blockiert XET-Storage-Downloads
                try:
                    _fe = AutoFeatureExtractor.from_pretrained(model_name, local_files_only=True)  # nosec B615 — local_files_only=True, kein Download
                    _m = AutoModelForAudioClassification.from_pretrained(model_name, local_files_only=True)  # nosec B615 — local_files_only=True, kein Download
                    _m.eval()
                    _m.to(torch.device("cpu"))
                    return _fe, _m
                finally:
                    if _prev:
                        _os.environ["TRANSFORMERS_OFFLINE"] = _prev
                    else:
                        _os.environ.pop("TRANSFORMERS_OFFLINE", None)
                    if _prev_hf:
                        _os.environ["HF_HUB_OFFLINE"] = _prev_hf
                    else:
                        _os.environ.pop("HF_HUB_OFFLINE", None)

            _ex = _cf.ThreadPoolExecutor(max_workers=1)
            _fut = _ex.submit(_load_hf_model)
            try:
                self.feature_extractor, self.model = _fut.result(timeout=8.0)

                def _unload_hf() -> None:
                    self.feature_extractor = None
                    self.model = None
                    release("ASTPerceptualHF")

                register_plugin("ASTPerceptualHF", 0.35, _unload_hf)
                logger.info("Psychoakustisches Modell geladen: %s (CPU)", model_name)
            except _cf.TimeoutError:
                logger.warning("Modell-Laden Zeitlimit (8 s) — DSP-Ersatzpfad aktiviert")
                self.feature_extractor = None
                self.model = None
                release("ASTPerceptualHF")
            except Exception as _load_err:
                logger.warning("Modell-Laden Fehler: %s — DSP-Ersatzpfad", _load_err)
                self.feature_extractor = None
                self.model = None
                release("ASTPerceptualHF")
            finally:
                _ex.shutdown(wait=False)  # Hintergrund-Thread nicht blockierend beenden

        except OSError as e:
            # Expected: model directory not bundled yet — DSP fallback is the intended path.
            # Log at DEBUG to avoid noise in startup logs.
            logger.debug("PerceptualValidator: Modell nicht gefunden, DSP-Ersatzpfad aktiv (%s)", e)
            self.feature_extractor = None
            self.model = None
        except Exception as e:
            logger.warning("konnte nicht laden psychoacoustic model: %s", e)
            logger.debug("Perceptual Validierung will use Ersatzpfad heuristics")
            self.feature_extractor = None
            self.model = None

    def validate_goal(
        self,
        audio: np.ndarray,
        sr: int,
        goal_name: str,
        technical_score: float,
        metadata: dict[str, Any] | None = None,
    ) -> PerceptualScore:
        """
        Validate ein Musical Goal gegen psychoakustische Wahrnehmung.

        Args:
            audio: Audio signal (mono)
            sr: Sample rate
            goal_name: Name des Musical Goal ('bass-kraft', 'brillanz', etc.)
            technical_score: Technischer score von MusicalGoalsChecker
            metadata: Zusätzliche metadata (genre, medium_type, etc.)

        Returns:
            PerceptualScore mit allen validation details
        """
        self.validation_count += 1
        metadata = metadata or {}

        # Psychoacoustic Score predicten
        psychoacoustic_score, confidence = self._predict_psychoacoustic_score(audio, sr, goal_name, metadata)

        # Adjusted Score berechnen (weighted: 70% technical, 30% psychoacoustic)
        adjusted_score = 0.7 * technical_score + 0.3 * psychoacoustic_score

        # Check ob Listening Test nötig
        requires_human = self._requires_listening_test(
            confidence=confidence,
            technical_score=technical_score,
            psychoacoustic_score=psychoacoustic_score,
            goal_name=goal_name,
        )

        # A/B Test Sample sammeln (bei Rate%)
        if np.random.random() < self.ab_test_collection_rate:
            self._collect_ab_test_sample(
                audio, sr, goal_name, technical_score, psychoacoustic_score, confidence, metadata
            )

        return PerceptualScore(
            technical_score=technical_score,
            psychoacoustic_score=psychoacoustic_score,
            confidence=confidence,
            adjusted_score=adjusted_score,
            requires_human=requires_human,
            metadata={"goal_name": goal_name, "validation_id": self.validation_count, **metadata},
        )

    def validate_all_goals(
        self, audio: np.ndarray, sr: int, technical_scores: dict[str, float], metadata: dict[str, Any] | None = None
    ) -> dict[str, PerceptualScore]:
        """
        Validate alle 7 Musical Goals.

        Args:
            audio: Audio signal
            sr: Sample rate
            technical_scores: Dict mit allen technical scores
            metadata: Zusätzliche metadata

        Returns:
            Dict[goal_name, PerceptualScore] für alle Goals
        """
        results = {}
        for goal_name, technical_score in technical_scores.items():
            results[goal_name] = self.validate_goal(audio, sr, goal_name, technical_score, metadata)

        # Check ob overall Listening Test nötig
        if any(score.requires_human for score in results.values()):
            self._create_listening_test_request(audio, technical_scores, results, metadata)  # type: ignore[arg-type]

        return results

    def _predict_psychoacoustic_score(
        self, audio: np.ndarray, sr: int, goal_name: str, metadata: dict[str, Any]
    ) -> tuple[float, float]:
        """
        Predict psychoacoustic score using ML model.

        Returns:
            (psychoacoustic_score, confidence)
        """
        if self.onnx_session is not None:
            # §4.6b PLM-Active-Guard: prevent Emergency-Eviction during AST ONNX inference
            _plm_ast: object | None = None
            try:
                from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm_ast

                _plm_ast = _get_plm_ast()
                _plm_ast.set_active("ASTPerceptualONNX", True)
            except Exception as e:
                logger.warning("perceptual_validator.py::_predict_psychoacoustic_Wert Ersatzpfad: %s", e)
            try:
                _inp = self._prepare_ast_onnx_input(audio, sr)
                _res = self.onnx_session.run([self._onnx_output_name], {self._onnx_input_name: _inp})[0]
                logits = np.asarray(_res, dtype=np.float32)
                return self._map_onnx_output_to_goal(logits, goal_name)
            except Exception as e:
                logger.debug("AST ONNX prediction fehlgeschlagen: %s", e)
            finally:
                if _plm_ast is not None:
                    try:
                        _plm_ast.set_active("ASTPerceptualONNX", False)
                    except Exception as e:
                        logger.warning("perceptual_validator.py::_predict_psychoacoustic_Wert Ersatzpfad: %s", e)

        if self.model is None:
            # Fallback: Heuristic-based scoring
            return self._heuristic_psychoacoustic_score(audio, sr, goal_name, metadata)

        try:
            assert librosa is not None and torch is not None
            # Resample to model's expected sample rate (16kHz for AST)
            target_sr = 16000
            audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr) if sr != target_sr else audio

            # Extract features
            assert self.feature_extractor is not None
            inputs = self.feature_extractor(audio_resampled, sampling_rate=target_sr, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)

            # Map model outputs to goal-specific score
            # Das ist eine vereinfachte Mapping - in Production würde hier
            # ein fine-tuned model verwendet werden
            psychoacoustic_score, confidence = self._map_model_output_to_goal(probs, goal_name, metadata)

            return psychoacoustic_score, confidence

        except Exception as e:
            logger.warning("Psychoacoustic prediction fehlgeschlagen: %s", e)
            return self._heuristic_psychoacoustic_score(audio, sr, goal_name, metadata)

    def _map_model_output_to_goal(self, probs: Any, goal_name: str, metadata: dict[str, Any]) -> tuple[float, float]:
        """
        Map model probabilities zu Goal-specific score.

        NOTE: Dies ist eine Placeholder-Implementierung.
        In Production würde hier ein custom-trained model verwendet werden.
        """
        # Get max probability als confidence
        confidence = float(probs.max())

        # Average probability für relevante classes
        relevant_classes = self._GOAL_MAPPINGS.get(goal_name, [0])
        relevant_probs = probs[0, relevant_classes]
        psychoacoustic_score = float(relevant_probs.mean())

        # Normalize to 0-1 range
        psychoacoustic_score = np.clip(psychoacoustic_score * 2.0, 0.0, 1.0)

        return psychoacoustic_score, confidence

    def _prepare_ast_onnx_input(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Erstellt [1, 1024, 128] log-mel tensor for AST ONNX model."""
        if librosa is None:
            raise RuntimeError("librosa not available for AST ONNX preprocessing")
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=-1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if sr != 16000:
            x = librosa.resample(x, orig_sr=sr, target_sr=16000)

        # Short-signal guard: avoid librosa warnings when n_fft exceeds signal length.
        if len(x) < 64:
            x = np.pad(x, (0, 64 - len(x)))
        _n_fft = int(min(1024, len(x)))
        if _n_fft % 2 == 1:
            _n_fft -= 1
        _n_fft = max(64, _n_fft)
        _win_length = int(min(400, _n_fft))
        _hop_length = int(max(16, min(160, _win_length // 2)))

        mel = librosa.feature.melspectrogram(
            y=x,
            sr=16000,
            n_fft=_n_fft,
            hop_length=_hop_length,
            win_length=_win_length,
            n_mels=128,
            fmin=20,
            fmax=8000,
            power=2.0,
        )
        mel = librosa.power_to_db(np.maximum(mel, 1e-10), ref=np.max)
        feat = mel.T  # [frames, 128]

        target_frames = 1024
        if feat.shape[0] < target_frames:
            pad = np.zeros((target_frames - feat.shape[0], feat.shape[1]), dtype=np.float32)
            feat = np.concatenate([feat.astype(np.float32), pad], axis=0)
        else:
            feat = feat[:target_frames].astype(np.float32)

        mean = float(np.mean(feat))
        std = float(np.std(feat) + 1e-6)
        feat = (feat - mean) / std
        return feat[np.newaxis, :, :].astype(np.float32)  # type: ignore[no-any-return]

    def _map_onnx_output_to_goal(self, logits: np.ndarray, goal_name: str) -> tuple[float, float]:
        """Map AST ONNX logits [1, 527] to (score, confidence)."""
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError(f"Unexpected AST logits shape: {logits.shape}")
        x = logits[0]
        x = x - float(np.max(x))
        ex = np.exp(x)
        probs = ex / max(float(np.sum(ex)), 1e-12)

        confidence = float(np.max(probs))
        relevant_classes = self._GOAL_MAPPINGS.get(goal_name, [0])
        score = float(np.mean(probs[relevant_classes]))
        score = float(np.clip(score * 2.0, 0.0, 1.0))
        return score, confidence

    def _heuristic_psychoacoustic_score(
        self, audio: np.ndarray, sr: int, goal_name: str, metadata: dict[str, Any]
    ) -> tuple[float, float]:
        """
        Fallback heuristic-based psychoacoustic scoring.

        Returns:
            (psychoacoustic_score, confidence)
        """
        assert librosa is not None
        # Extract basic features
        rms = librosa.feature.rms(y=audio)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(audio)[0]

        # Goal-specific heuristics
        if goal_name == "bass-kraft":
            # Low spectral centroid = more bass
            score = 1.0 - np.clip(np.mean(spectral_centroid) / 4000, 0, 1)
            confidence = 0.5

        elif goal_name == "brillanz":
            # High spectral centroid = more brilliance
            score = np.clip(np.mean(spectral_centroid) / 4000, 0, 1)
            confidence = 0.5

        elif goal_name == "waerme":
            # Warmth Ratio E(200-800 Hz) / E(800-3000 Hz) — Moore & Glasberg 1983
            S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
            freqs_w = librosa.fft_frequencies(sr=sr, n_fft=2048)
            low_mask = (freqs_w >= 200) & (freqs_w <= 800)
            mid_mask = (freqs_w >= 800) & (freqs_w <= 3000)
            low_e = float(np.sum(S[low_mask, :] ** 2))
            mid_e = float(np.sum(S[mid_mask, :] ** 2))
            warmth_ratio = low_e / (mid_e + 1e-12)
            score = float(np.clip(warmth_ratio / 2.0, 0.0, 1.0))
            confidence = 0.55

        elif goal_name == "natuerlichkeit":
            # ZCR + Spectral Flatness combined (Fastl & Zwicker 2007)
            sf = librosa.feature.spectral_flatness(y=audio)[0]
            zcr_score = 1.0 - float(np.clip(np.mean(zcr), 0, 1))
            sf_score = float(np.clip(np.mean(sf) * 3.0, 0, 1))  # flat spectrum = natural
            score = 0.6 * zcr_score + 0.4 * sf_score
            confidence = 0.5

        elif goal_name == "emotionalitaet":
            # Dynamic range + spectral centroid variance (Juslin & Laukka 2003)
            dynamic_range = float(np.std(rms))
            centroid_var = float(np.std(spectral_centroid) / (np.mean(spectral_centroid) + 1e-10))
            score = float(np.clip(0.6 * dynamic_range * 10 + 0.4 * centroid_var * 5, 0, 1))
            confidence = 0.5

        elif goal_name == "transparenz":
            # Multi-Band Spectral Crest Factor (Moore & Glasberg 1983; ITU-T P.862)
            S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
            freqs_t = librosa.fft_frequencies(sr=sr, n_fft=2048)
            band_edges = [250, 500, 1000, 2000, 4000, 8000]
            crest_values = []
            for i in range(len(band_edges) - 1):
                bm = (freqs_t >= band_edges[i]) & (freqs_t < band_edges[i + 1])
                if np.any(bm):
                    band_energy = S[bm, :]
                    p95 = float(np.percentile(band_energy, 95))
                    p50 = float(np.percentile(band_energy, 50))
                    crest = p95 / (p50 + 1e-12)
                    crest_values.append(float(np.clip(crest / 10.0, 0.0, 1.0)))
            score = float(np.mean(crest_values)) if crest_values else 0.5
            confidence = 0.5

        else:  # 'authentizitaet' or unknown
            # Spectral correlation stability as authenticity proxy
            S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
            if S.shape[1] > 1:
                frame_corrs = []
                for i in range(min(S.shape[1] - 1, 50)):
                    _a, _b = S[:, i], S[:, i + 1]
                    # §VERBOTEN: np.corrcoef ohne std-Guard → RuntimeWarning bei near-constant STFT-Frames.
                    if np.std(_a) < 1e-12 or np.std(_b) < 1e-12:
                        continue  # silent frame → skip
                    _aa = _a - _a.mean()
                    _ba = _b - _b.mean()
                    _na = float(np.linalg.norm(_aa))
                    _nb = float(np.linalg.norm(_ba))
                    c = float(np.dot(_aa, _ba) / (_na * _nb + 1e-10))
                    if np.isfinite(c):
                        frame_corrs.append(c)
                score = float(np.mean(frame_corrs)) if frame_corrs else 0.7
            else:
                score = 0.7
            confidence = 0.4

        return float(score), float(confidence)

    def _requires_listening_test(
        self, confidence: float, technical_score: float, psychoacoustic_score: float, goal_name: str
    ) -> bool:
        """
        Entscheidet ob menschliches Listening Test nötig ist.

        Kriterien:
        - Low confidence (<70%)
        - Large discrepancy zwischen technical und psychoacoustic score
        - Critical violations (<60%)
        - Critical goals (Natürlichkeit, Authentizität)
        """
        # Low confidence
        if confidence < self.confidence_threshold:
            return True

        # Large discrepancy (>20% difference)
        discrepancy = abs(technical_score - psychoacoustic_score)
        if discrepancy > 0.2:
            return True

        # Critical violations
        if technical_score < 0.6 or psychoacoustic_score < 0.6:
            return True

        # Critical goals always require higher scrutiny
        critical_goals = ["natuerlichkeit", "authentizitaet"]
        return bool(goal_name in critical_goals and confidence < 0.85)

    def _create_listening_test_request(
        self,
        audio: np.ndarray,
        technical_scores: dict[str, float],
        perceptual_scores: dict[str, PerceptualScore],
        metadata: dict[str, Any],
    ) -> None:
        """Create Listening Test Request für menschliche validation."""
        metadata = metadata or {}  # Handle None

        # Find lowest confidence scores
        confidence_scores = {goal: score.confidence for goal, score in perceptual_scores.items()}
        min_confidence = min(confidence_scores.values())

        # Determine priority
        if min_confidence < 0.5:
            priority = "high"
        elif min_confidence < 0.7:
            priority = "medium"
        else:
            priority = "low"

        # Create reason
        low_conf_goals = [goal for goal, conf in confidence_scores.items() if conf < self.confidence_threshold]
        reason = f"Low confidence for goals: {', '.join(low_conf_goals)}"

        request = ListeningTestRequest(
            session_id=metadata.get("session_id", "unknown"),
            audio_path=metadata.get("audio_path", "unknown"),
            goal_scores=technical_scores,
            confidence_scores=confidence_scores,
            reason=reason,
            priority=priority,
        )

        self.listening_test_requests.append(request)
        logger.info("erstellt listening test request (priority=%s): %s", priority, reason)

    def _collect_ab_test_sample(
        self,
        audio: np.ndarray,
        sr: int,
        goal_name: str,
        technical_score: float,
        psychoacoustic_score: float,
        confidence: float,
        metadata: dict[str, Any],
    ) -> None:
        """
        Sammelt A/B Test Sample für Training Data Collection.

        Strategy: Sammle pairs von audio wo model uncertain ist (confidence < 0.8)
        für spätere menschliche evaluation.
        """
        if confidence > 0.8:
            return  # Only collect uncertain samples

        sample_id = f"ab_{goal_name}_{datetime.now().timestamp()}"

        # Store sample info (actual audio files würden separat gespeichert)
        sample = ABTestSample(
            sample_id=sample_id,
            audio_a_path=metadata.get("audio_path", "unknown"),
            audio_b_path=metadata.get("reference_path", metadata.get("audio_path", "unknown")),
            goal=goal_name,
            score_a=technical_score,
            score_b=psychoacoustic_score,
            predicted_preference="A" if technical_score > psychoacoustic_score else "B",
            confidence=confidence,
        )

        self.ab_test_samples.append(sample)

        # Save to disk
        sample_file = self.ab_test_storage_path / f"{sample_id}.json"
        with open(sample_file, "w") as f:
            json.dump(
                {
                    "sample_id": sample.sample_id,
                    "audio_a_path": sample.audio_a_path,
                    "audio_b_path": sample.audio_b_path,
                    "goal": sample.goal,
                    "score_a": sample.score_a,
                    "score_b": sample.score_b,
                    "predicted_preference": sample.predicted_preference,
                    "confidence": sample.confidence,
                    "collected_at": sample.collected_at.isoformat(),
                },
                f,
                indent=2,
            )

        logger.debug("Collected A/B test sample: %s", sample_id)

    def get_listening_test_queue(self, priority: str | None = None, limit: int = 10) -> list[ListeningTestRequest]:
        """
        Gibt zurück: queue of pending listening test requests.

        Args:
            priority: Filter by priority ('high', 'medium', 'low')
            limit: Maximum number of requests to return

        Returns:
            List of ListeningTestRequest
        """
        requests = self.listening_test_requests

        if priority:
            requests = [r for r in requests if r.priority == priority]

        # Sort by priority (high > medium > low) and creation time
        priority_order = {"high": 0, "medium": 1, "low": 2}
        requests.sort(key=lambda r: (priority_order[r.priority], r.created_at))

        return requests[:limit]

    def submit_listening_test_result(  # type: ignore[return,return-value]
        self, session_id: str, human_scores: dict[str, float], comments: str | None = None
    ) -> dict[str, Any]:
        """
        Submit menschliche listening test results.

        Args:
            session_id: Session ID des requests
            human_scores: Dict[goal_name, human_score]
            comments: Optional feedback comments
        """
        # Find corresponding request
        request = next((r for r in self.listening_test_requests if r.session_id == session_id), None)

        if not request:
            logger.warning("No listening test request found for Sitzung %s", session_id)
            return  # type: ignore[return-value]

        # Store result
        result_file = self.ab_test_storage_path / f"listening_test_{session_id}.json"
        with open(result_file, "w") as f:
            json.dump(
                {
                    "session_id": session_id,
                    "audio_path": request.audio_path,
                    "technical_scores": request.goal_scores,
                    "human_scores": human_scores,
                    "confidence_scores": request.confidence_scores,
                    "comments": comments,
                    "submitted_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        # Remove from queue
        self.listening_test_requests.remove(request)
        logger.info("Listening test Ergebnis submitted for Sitzung %s", session_id)

    def get_statistics(self) -> dict[str, Any]:
        """Gibt zurück: validation statistics."""
        return {
            "total_validations": self.validation_count,
            "listening_test_requests": len(self.listening_test_requests),
            "ab_test_samples_collected": len(self.ab_test_samples),
            "listening_test_queue_by_priority": {
                "high": len([r for r in self.listening_test_requests if r.priority == "high"]),
                "medium": len([r for r in self.listening_test_requests if r.priority == "medium"]),
                "low": len([r for r in self.listening_test_requests if r.priority == "low"]),
            },
        }

    def is_loaded(self) -> bool:
        """Ob ein echtes psychoakustisches Modell aktiv ist (zentraler AST, ONNX oder HF).

        Ohne aktives Modell fällt `validate_goal()` auf DSP-Heuristiken zurück (§G8-Transparenz-
        konform per `logger.warning()` an den jeweiligen Ladeversuchen) — dieser Wert spiegelt
        genau diesen Zustand für `ml_model_readiness.py` wider.
        """
        return self.onnx_session is not None or self.model is not None


_validator: PerceptualValidator | None = None


def get_perceptual_validator() -> PerceptualValidator:
    """Gibt die globale PerceptualValidator-Instanz zurück (Singleton).

    Kein per-Song-Zustand (§V8 (copilot-instructions.md)): `ab_test_samples`/`listening_test_requests` sind
    projektweite Trainingsdaten-Sammlung, keine Song-spezifischen Audio-Daten —
    ein Singleton ist hier architektonisch korrekt, analog `get_ast_classifier()`.
    """
    global _validator
    if _validator is None:
        _validator = PerceptualValidator()
    return _validator
