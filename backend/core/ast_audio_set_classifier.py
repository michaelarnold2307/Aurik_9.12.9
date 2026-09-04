"""
§v10.304 — AST AudioSet-527 Classifier: Zentraler Hub für Aurik.

Ersetzt die fragmentierte AST-Nutzung im PerceptualValidator durch einen
einheitlichen Singleton-Classifier, den ALLE Subsysteme konsumieren:
  - PerceptualValidator (Goal-Validierung)
  - DefectScanner (Defect-vs-Music-Discrimination)
  - EmotionalArcPreserver (Mood-Guided Mastering)
  - Phase_53 SemanticAudio (Instrument/Genre-Tagging)
  - EraClassifier (Era-Indikatoren via Instrumente)

AudioSet-Ontologie: 527 Klassen, ONNX-Inference (CPUExecutionProvider).

Privacy: Kein Audio verlässt den Prozess. Reine ONNX-Inference.

Spec: §v10.304, ersetzt §v10.303 AST-Tiefenanalyse-Empfehlungen.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------

_AST_ONNX_PATH: Path = Path(__file__).resolve().parent.parent.parent / "models" / "ast" / "ast_model.onnx"

# ---------------------------------------------------------------------------
# AudioSet-527 Label-Map (gekürzt auf die für Aurik relevanten Klassen)
# Vollständige Ontologie: Gemmeke et al. 2017, AudioSet
# ---------------------------------------------------------------------------

AUDIOSET_LABELS: dict[int, str] = {
    # ── Human sounds (0–50) ─────────────────────────────────────────────
    0: "speech",
    1: "male_speech",
    2: "female_speech",
    4: "child_speech",
    7: "yell",
    20: "singing",
    21: "choir",
    25: "whispering",
    30: "humming",
    # ── Musical instruments (100–250) ────────────────────────────────────
    10: "bass_drum",
    104: "bass_guitar",
    105: "violin",
    121: "snare_drum",
    137: "cello",
    138: "bell",
    141: "piano",
    143: "electric_guitar",
    196: "flute",
    299: "synthesizer",
    310: "hi_hat",
    311: "cymbal",
    # ── Music genre (350–400) ────────────────────────────────────────────
    350: "rock_music",
    351: "pop_music",
    352: "jazz",
    353: "classical_music",
    354: "country",
    355: "blues",
    356: "electronic_music",
    357: "hip_hop",
    358: "reggae",
    359: "folk_music",
    360: "soul_music",
    361: "latin_music",
    # ── Music mood/emotion (400–450) ─────────────────────────────────────
    400: "happy_music",
    401: "sad_music",
    402: "scary_music",
    403: "tender_music",
    404: "exciting_music",
    # ── Audio quality (450–490) ──────────────────────────────────────────
    450: "high_audio_quality",
    451: "low_audio_quality",
    452: "noise",
    453: "distortion",
    454: "echo",
    # ── Sound effects / defects (500–527) ────────────────────────────────
    504: "crackle",
    505: "pop",
    506: "hiss",
    507: "rumble",
    508: "buzz",
    509: "hum",
    510: "click",
    511: "wow",
    512: "flutter",
}

# ── Goal-Mappings: korrekte AudioSet-Indizes für Musical Goals ──────────

CORRECTED_GOAL_MAPPINGS: dict[str, list[int]] = {
    # Bass-Kraft: Bass drum (10), Bass guitar (104), Cello (137)
    "bass-kraft": [10, 104, 137],
    # Brillanz: Hi-hat (310), Cymbal (311), Bell (138)
    "brillanz": [310, 311, 138],
    # Wärme: Cello (137), Piano (141), Bass guitar (104)
    "waerme": [137, 141, 104],
    # Natürlichkeit: High audio quality (450), Classical (353), Folk (359)
    "natuerlichkeit": [450, 353, 359],
    # Authentizität: Analog-indikative Klassen
    "authentizitaet": [504, 506, 507],  # crackle, hiss, rumble (analog markers)
    # Emotionalität: Happy music (400), Sad music (401), Tender (403), Exciting (404)
    "emotionalitaet": [400, 401, 403, 404],
    # Transparenz: Piano (141), Flute (196), Classical (353)
    "transparenz": [141, 196, 353],
}

# ── Defect-vs-Music-Discriminator-Mappings ──────────────────────────────

# Jeder DefectScanner-Defekttyp bekommt eine Liste von AudioSet-Instrumenten.
# Wenn ein Defekt auf einem dieser Instrumente liegt → PRESERVE (kein Defekt).
DEFECT_INSTRUMENT_DISCRIMINATOR: dict[str, list[int]] = {
    "crackle": [121, 141, 137, 143],  # snare, piano, cello, e-guitar
    "click": [121, 141, 104, 143],  # snare, piano, bass, e-guitar
    "hiss": [310, 311, 196],  # hi-hat, cymbal, flute
    "rumble": [10, 104, 299],  # bass drum, bass guitar, synth
    "buzz": [143, 299],  # e-guitar, synth
    "pop": [121, 141],  # snare, piano
    "wow": [105, 196, 299],  # violin, flute, synth (vibrato)
    "flutter": [105, 196, 299],  # violin, flute, synth (vibrato)
}

# ── Emotion-Guided-Mastering-Mappings ───────────────────────────────────

# DSP-Parameter: {dynamics_scale, presence_db, space_scale, groove_ms}
EMOTION_DSP_PARAMS: dict[str, dict[str, float]] = {
    "happy_music": {"dyn": 1.10, "pres_db": 1.0, "space": 1.00, "groove": -2.0},
    "sad_music": {"dyn": 0.65, "pres_db": -0.5, "space": 1.20, "groove": 5.0},
    "scary_music": {"dyn": 0.50, "pres_db": -2.0, "space": 1.30, "groove": 10.0},
    "tender_music": {"dyn": 0.60, "pres_db": -1.0, "space": 1.20, "groove": 3.0},
    "exciting_music": {"dyn": 1.20, "pres_db": 1.5, "space": 0.90, "groove": -3.0},
}

# ── Era-Indikatoren via Instrumente ─────────────────────────────────────

ERA_INSTRUMENT_INDICATORS: dict[str, tuple[int, int]] = {
    "harpsichord": (1600, 1780),
    "pipe_organ": (1600, 1900),
    "accordion": (1900, 1950),
    "theremin": (1920, 1950),
    "synthesizer": (1970, 2100),
    "drum_machine": (1980, 2100),
    "sampler": (1985, 2100),
    "turntable": (1980, 2100),
    "autotune": (1995, 2100),
}
# Leider hat AudioSet keine direkten Klassen für alle diese Instrumente.
# Wir verwenden die nächstbesten: electric_guitar (143) für 1950+,
# synthesizer (299) für 1970+, drum_machine fehlt → bass_drum+hi_hat (10+310) als Proxy.
ERA_AUDIOSET_PROXIES: dict[int, tuple[int, int]] = {
    141: (1700, 2100),  # piano: immer präsent
    143: (1950, 2100),  # electric guitar: ≥1950
    299: (1970, 2100),  # synthesizer: ≥1970
    357: (1980, 2100),  # hip_hop: ≥1980
}


@dataclass
class AstResult:
    """Ergebnis einer AST AudioSet-527 Inferenz."""

    logits: np.ndarray  # [527,] raw logits
    probs: np.ndarray  # [527,] softmax probabilities
    top_k: list[tuple[int, str, float]]  # (index, label, probability)
    model_used: str  # "ast_onnx" | "fallback"
    inference_time_ms: float

    def get_prob(self, class_index: int) -> float:
        """Gibt die Wahrscheinlichkeit für eine AudioSet-Klasse zurück."""
        if 0 <= class_index < len(self.probs):
            return float(self.probs[class_index])
        return 0.0

    def get_probs(self, class_indices: list[int]) -> float:
        """Mittlere Wahrscheinlichkeit über mehrere Klassen."""
        if not class_indices:
            return 0.0
        return float(np.mean([self.get_prob(i) for i in class_indices]))


@dataclass
class AstDefectDiscrimination:
    """Ergebnis der Defekt-vs-Musik-Unterscheidung."""

    defect_type: str
    time_s: float
    defect_severity: float
    instrument_label: str
    instrument_confidence: float
    is_musical: bool  # True = intentional music, nicht entfernen
    preserve_reason: str


class AstAudioSetClassifier:
    """§v10.304: Singleton AST AudioSet-527 Classifier für alle Aurik-Subsysteme.

    Lädt models/ast/ast_model.onnx (ONNX, CPUExecutionProvider).
    Verarbeitet Audio in 10 s-Fenstern (128-Mel × 1024-Frames).
    Gibt Top-K AudioSet-Labels mit Konfidenzen zurück.

    Usage:
        clf = get_ast_classifier()
        result = clf.classify(audio, sr)
        is_snare = result.get_prob(121) > 0.3  # snare drum
    """

    _AST_SR: int = 16_000
    _N_MELS: int = 128
    _N_FFT: int = 1024
    _HOP: int = 160  # 10 ms at 16 kHz
    _TARGET_FRAMES: int = 1024  # ~10.24 s

    def __init__(self) -> None:
        self._session: Any = None
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._lock = threading.Lock()
        self._load_onnx()

    def _load_onnx(self) -> None:
        """Lädt AST ONNX-Modell mit CPUExecutionProvider."""
        if not _AST_ONNX_PATH.is_file():
            logger.info("AST ONNX nicht gefunden: %s — DSP-Ersatzpfad aktiv", _AST_ONNX_PATH)
            return
        try:
            from backend.core.ml_memory_budget import release, try_allocate

            if not try_allocate("ASTAudioSetClassifier", 0.35):
                logger.warning("AST: ML-Grenze erschöpft — DSP-Ersatzpfad aktiv")
                return

            try:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
                opts.inter_op_num_threads = 1
                self._session = ort.InferenceSession(
                    str(_AST_ONNX_PATH),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self._input_name = self._session.get_inputs()[0].name
                self._output_name = self._session.get_outputs()[0].name

                def _unload() -> None:
                    self._session = None
                    self._input_name = None
                    self._output_name = None
                    try:
                        release("ASTAudioSetClassifier")
                    except Exception:
                        logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

                from backend.core.plugin_lifecycle_manager import register_plugin

                register_plugin("ASTAudioSetClassifier", 0.35, _unload)
                logger.info("§v10.304 AST AudioSet Classifier geladen (models/ast/ast_model.onnx)")
            except Exception:
                release("ASTAudioSetClassifier")
                raise
        except Exception as exc:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            logger.debug("AST ONNX laden fehlgeschlagen: %s — DSP-Ersatzpfad aktiv", exc)
            self._session = None

    def is_loaded(self) -> bool:
        """True wenn ONNX-Modell geladen und bereit."""
        return self._session is not None

    def classify(
        self,
        audio: np.ndarray,
        sr: int,
        top_k: int = 10,
    ) -> AstResult:
        """Klassifiziert Audio mit AST AudioSet-527.

        Args:
            audio: float32 ndarray, mono (N,) oder stereo (N, 2)
            sr:    Sample rate (beliebig, wird resampled)
            top_k: Anzahl Top-K Labels

        Returns:
            AstResult mit Logits, Probs und Top-K Labels.
        """
        if self._session is None:
            return self._fallback_result()

        import time

        t0 = time.perf_counter()

        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

            _plm = get_plugin_lifecycle_manager()
            _plm.set_active("ASTAudioSetClassifier", True)
        except Exception:
            _plm = None

        try:
            # Mono + resample to 16 kHz
            audio_np = np.asarray(audio, dtype=np.float32)
            mono = audio_np.mean(axis=-1) if audio_np.ndim > 1 else audio_np
            if sr != self._AST_SR:
                mono = self._resample(mono, sr, self._AST_SR)

            # Compute 128-mel spectrogram, shape (128, 1024)
            mel_spec = self._compute_mel_spectrogram(mono)

            # Run ONNX inference
            input_tensor = mel_spec[np.newaxis, :, :].astype(np.float32)  # (1, 128, 1024)
            logits_raw = self._session.run([self._output_name], {self._input_name: input_tensor})[0]  # (1, 527)
            logits = np.asarray(logits_raw[0], dtype=np.float32)

            # Softmax
            logits_shifted = logits - float(np.max(logits))
            ex = np.exp(logits_shifted)
            probs = ex / max(float(np.sum(ex)), 1e-12)

            # Top-K
            top_indices = np.argsort(probs)[::-1][:top_k]
            top_k_list: list[tuple[int, str, float]] = [
                (int(i), AUDIOSET_LABELS.get(int(i), f"class_{i}"), float(probs[i])) for i in top_indices
            ]

            dt_ms = (time.perf_counter() - t0) * 1000.0
            return AstResult(
                logits=logits,
                probs=probs,
                top_k=top_k_list,
                model_used="ast_onnx",
                inference_time_ms=dt_ms,
            )
        except Exception as exc:
            logger.debug("AST classify fehlgeschlagen: %s", exc)
            return self._fallback_result()
        finally:
            if _plm is not None:
                try:
                    _plm.set_active("ASTAudioSetClassifier", False)
                except Exception:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

    def classify_segment(
        self,
        audio: np.ndarray,
        sr: int,
        start_s: float,
        end_s: float,
    ) -> AstResult:
        """Klassifiziert ein Zeitsegment des Audios."""
        i0 = max(0, int(start_s * sr))
        i1 = min(audio.shape[-1] if audio.ndim > 1 else len(audio), int(end_s * sr) + 1)
        if i1 <= i0:
            return self._fallback_result()
        segment = audio[..., i0:i1] if audio.ndim > 1 else audio[i0:i1]
        return self.classify(segment, sr)

    def discriminate_defect(
        self,
        defect_type: str,
        audio: np.ndarray,
        sr: int,
        time_s: float,
        severity: float,
    ) -> AstDefectDiscrimination | None:
        """§v10.304: Prüft ob ein erkannter Defekt tatsächlich Musik ist.

        Args:
            defect_type: DefectScanner Defekttyp (crackle, click, hiss, …)
            audio:       Volles Audio (für Kontext-Fenster)
            sr:          Sample rate
            time_s:      Zeitpunkt des Defekts
            severity:    DefectScanner severity

        Returns:
            AstDefectDiscrimination oder None wenn AST nicht geladen.
        """
        if self._session is None:
            return None

        instrument_indices = DEFECT_INSTRUMENT_DISCRIMINATOR.get(defect_type)
        if not instrument_indices:
            return None

        # Segment um den Defekt herum (1s Fenster)
        result = self.classify_segment(audio, sr, max(0, time_s - 0.5), time_s + 0.5)
        if result.model_used == "fallback":
            return None

        # Bestes Instrument im Fenster
        best_idx = -1
        best_conf = 0.0
        for idx in instrument_indices:
            conf = result.get_prob(idx)
            if conf > best_conf:
                best_conf = conf
                best_idx = idx

        is_musical = best_conf >= 0.15  # Schwelle: 15 % Konfidenz
        label = AUDIOSET_LABELS.get(best_idx, "unknown") if best_idx >= 0 else "unknown"
        reason = (
            f"{defect_type} on {label} ({best_conf:.2f}) → PRESERVE"
            if is_musical
            else f"{defect_type} ({best_conf:.2f}) → REPAIR"
        )

        return AstDefectDiscrimination(
            defect_type=defect_type,
            time_s=time_s,
            defect_severity=severity,
            instrument_label=label,
            instrument_confidence=best_conf,
            is_musical=is_musical,
            preserve_reason=reason,
        )

    def get_emotion_profile(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> dict[str, float] | None:
        """§v10.304: Ermittelt das emotionale Profil aus AST Mood-Labels.

        Returns:
            DSP-Parameter-Dict oder None wenn AST nicht geladen.
        """
        if self._session is None:
            return None

        result = self.classify(audio, sr, top_k=10)
        if result.model_used == "fallback":
            return None

        # Dominante Emotion finden
        emotion_probs = {
            label: result.get_prob(idx) for idx, label in AUDIOSET_LABELS.items() if idx in range(400, 405)
        }
        if not emotion_probs:
            return None

        best_emotion = max(emotion_probs, key=emotion_probs.get)  # type: ignore[arg-type]
        best_conf = emotion_probs[best_emotion]

        if best_conf < 0.10:
            return None

        params = EMOTION_DSP_PARAMS.get(best_emotion, {})
        params["emotion_label"] = best_emotion  # type: ignore[assignment]
        params["emotion_confidence"] = best_conf
        return params

    def get_era_constraints(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> tuple[int | None, int | None]:
        """§v10.304: Era-Constraints aus AST Instrument-Indikatoren.

        Returns:
            (min_possible_decade, max_possible_decade) oder (None, None).
        """
        if self._session is None:
            return None, None

        result = self.classify(audio, sr, top_k=20)
        if result.model_used == "fallback":
            return None, None

        min_decade = None
        max_decade = None
        for idx, (era_min, era_max) in ERA_AUDIOSET_PROXIES.items():
            if result.get_prob(idx) >= 0.15:
                if min_decade is None or era_min > min_decade:
                    min_decade = era_min
                if max_decade is None or era_max < max_decade:
                    max_decade = era_max
        return min_decade, max_decade

    # ── DSP helpers ────────────────────────────────────────────────────────

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample mit scipy (wenn verfügbar) oder np.interp-Fallback."""
        if orig_sr == target_sr:
            return cast(np.ndarray, audio.astype(np.float32))
        try:
            import scipy.signal

            num = int(len(audio) * target_sr / orig_sr)
            return scipy.signal.resample(audio, num).astype(np.float32)  # type: ignore[no-any-return]
        except Exception as exc:
            logger.debug("§V6 scipy.signal.resample fehlgeschlagen — Linear-Interpolation Fallback: %s", exc)
            # Linear interpolation fallback
            n = len(audio)
            x_old = np.linspace(0, n - 1, n)
            x_new = np.linspace(0, n - 1, int(n * target_sr / orig_sr))
            return cast(np.ndarray, (np.interp(x_new, x_old, audio).astype(np.float32)))

    def _compute_mel_spectrogram(self, audio_16k: np.ndarray) -> np.ndarray:
        """Berechnet 128-Mel-Spektrogramm (1024 Frames) für AST Input.

        AST erwartet: [1, 1024, 128] (batch, time, mel) oder [1, 128, 1024] (batch, mel, time).
        Je nach exportierter ONNX-Variante. Wir erzeugen (128, 1024).
        """
        n = len(audio_16k)
        # Pad/crop to get ~1024 frames (10.24 s at 10 ms hop)
        target_samples = self._TARGET_FRAMES * self._HOP
        if n < target_samples:
            audio_16k = np.pad(audio_16k, (0, target_samples - n))
        elif n > target_samples:
            audio_16k = audio_16k[:target_samples]

        try:
            import librosa

            mel = librosa.feature.melspectrogram(
                y=audio_16k.astype(np.float64),
                sr=self._AST_SR,
                n_fft=self._N_FFT,
                hop_length=self._HOP,
                n_mels=self._N_MELS,
                fmin=20,
                fmax=8000,
                power=2,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
            # Normalize to [0, 1]
            mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-12)
            return mel_norm.astype(np.float32)[:, : self._TARGET_FRAMES]  # type: ignore[no-any-return]
        except Exception as exc:
            logger.debug("§V6 librosa.feature.melspectrogram fehlgeschlagen — STFT-Fallback aktiviert: %s", exc)
            # Minimal fallback: STFT-based mel approximation
            return self._compute_mel_fallback(audio_16k)

    def _compute_mel_fallback(self, audio_16k: np.ndarray) -> np.ndarray:
        """STFT-basierte Mel-Spektrogramm-Approximation (kein librosa)."""
        n_fft = self._N_FFT
        hop = self._HOP
        n_frames = min(self._TARGET_FRAMES, (len(audio_16k) - n_fft) // hop + 1)
        window = np.hanning(n_fft)
        mel_spec = np.zeros((self._N_MELS, n_frames), dtype=np.float32)

        for i in range(n_frames):
            start = i * hop
            frame = audio_16k[start : start + n_fft] * window
            spec = np.abs(np.fft.rfft(frame, n=n_fft))
            power = spec**2
            # Simple mel binning (approximate)
            mel_bins = self._N_MELS
            mel_points = np.linspace(0, n_fft // 2, mel_bins + 1, dtype=np.int32)
            for m in range(mel_bins):
                mel_spec[m, i] = float(np.mean(power[mel_points[m] : mel_points[m + 1]]))

        # Normalize
        mel_spec = np.log1p(mel_spec)
        mn, mx = mel_spec.min(), mel_spec.max()
        if mx > mn:
            mel_spec = (mel_spec - mn) / (mx - mn + 1e-12)
        return cast(np.ndarray, mel_spec.astype(np.float32))

    @staticmethod
    def _fallback_result() -> AstResult:
        """Gibt leeres Fallback-Ergebnis zurück."""
        return AstResult(
            logits=np.zeros(527, dtype=np.float32),
            probs=np.ones(527, dtype=np.float32) / 527.0,
            top_k=[(0, "speech", 0.0)],
            model_used="fallback",
            inference_time_ms=0.0,
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: AstAudioSetClassifier | None = None
_lock = threading.Lock()


def get_ast_classifier() -> AstAudioSetClassifier:
    """Gibt den AST AudioSet Classifier Singleton zurück."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = AstAudioSetClassifier()
    return _instance


def is_ast_loaded() -> bool:
    """True wenn AST ONNX geladen und bereit (non-invasiver Peek, KEIN Lazy-Load).

    KRITISCH: Ruft NICHT get_ast_classifier() auf — das würde den Singleton
    konstruieren (ONNX-Load + try_allocate()) als Nebeneffekt eines reinen
    Readiness-Checks. Da ml_model_readiness._validate_all_checks() diesen
    Check beim ERSTEN Import von ml_model_readiness ausführt, und dieser
    Import selbst aus try_allocate() heraus ausgelöst wird (siehe
    ast_audio_set_classifier._load_onnx() → ml_memory_budget.try_allocate()),
    führte ein Aufruf von get_ast_classifier() hier zu einem Re-Entrant-
    Deadlock auf demselben, bereits vom konstruierenden Thread gehaltenen
    `_lock` (self-deadlock, futex_do_wait — reproduziert 2026-08-06).
    Analog zu _probe_plugin() in ml_model_readiness.py: nur den bereits
    existierenden Zustand abfragen, niemals konstruieren.
    """
    return _instance is not None and _instance.is_loaded()
