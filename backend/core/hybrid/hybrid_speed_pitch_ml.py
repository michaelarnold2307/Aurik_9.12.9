"""
Hybrid Speed/Pitch Detection - AURIK 9.0 Phase 31 ML-Hybrid
============================================================

Zwei-Stufen-Pitch-Detektion: pYIN (DSP) + CREPE (ML) für Geschwindigkeitskorrektur.
§4.2-konform: klassisches YIN (de Cheveigné 2002) ist VERBOTEN als primäre Methode.

Architektur:
1. Stufe 1: pYIN Pitch-Detektion (Mauch & Dixon 2014) via librosa.pyin
   - HMM-basierte Voiced/Unvoiced-Klassifikation
   - Globale Pitch-Schätzung (gemittelt über 2-5 Sekunden)
   - Robust bei verrauschten Signalen

2. Stufe 2: RMVPE/PESTO ML Pitch-Detektion
   - CNN-basiertes Pitch-Tracking
   - ±1 Cent Genauigkeit
   - Verhindert Oktavfehler
   - Besser bei komplexen/harmonisch dichten Signalen

Strategy-Modi:
- PYIN_ONLY: Schnelle pYIN-DSP-Detektion (FAST-Modus)
- CREPE_ONLY: Reines ML (kein DSP-Preprocessing)
- ADAPTIVE: pYIN -> ML falls Konfidenz < 0.7 (BALANCED-Modus)
- HYBRID: Immer pYIN + ML kombiniert (MAXIMUM-Modus)

VERBOTEN (§4.2): klassisches YIN (de Cheveigné 2002) als primäre Methode
Erfüllt: Mauch & Dixon (2014) pYIN mit HMM Voiced-Klassifikation

Use Case: Speed/Pitch Correction
- Detect global pitch (average over full audio)
- Compare to reference (typically A440)
- Calculate speed ratio for correction
- Apply time-stretching + pitch-shifting

Difference from Phase 12 (Wow/Flutter):
- Phase 12: Time-varying pitch (short segments, wow/flutter detection)
- Phase 31: Global pitch (full audio average, speed correction)

Author: Aurik 10.0.0 Development Team
Version: 1.0.0
Date: 16. Februar 2026
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PitchDetectionStrategy(Enum):
    """Pitch detection strategy selection."""

    PYIN_ONLY = "pyin_only"  # pYIN DSP (Mauch & Dixon 2014) — §4.2 konform
    CREPE_ONLY = "crepe_only"  # Pure ML CNN
    ADAPTIVE = "adaptive"  # Auto-select based on confidence
    HYBRID = "hybrid"  # pYIN + CREPE combined
    # Backward-Alias (deprecated, wird auf PYIN_ONLY gemappt)
    YIN_ONLY = "pyin_only"  # Alias → PYIN_ONLY


@dataclass
class SpeedPitchConfig:
    """Configuration for hybrid speed/pitch detection."""

    strategy: PitchDetectionStrategy = PitchDetectionStrategy.ADAPTIVE
    pyin_confidence_threshold: float = 0.4  # pYIN: Mindest-voiced_fraction × mean_prob
    crepe_model: str = "full"  # CREPE model size
    confidence_threshold: float = 0.7  # Skip CREPE if pYIN confidence > this
    averaging_window: float = 2.0  # Seconds to average for global pitch
    # Backward-Alias
    yin_threshold: float = 0.15  # unused — nur für Kompatibilität


@dataclass
class SpeedPitchResult:
    """Result from hybrid speed/pitch detection."""

    detected_pitch: float  # Globaler Pitch-Schätzwert (Hz)
    confidence: float  # Gesamtkonfidenz (0-1)
    strategy_used: PitchDetectionStrategy
    pyin_applied: bool  # pYIN (Mauch & Dixon 2014) angewendet
    crepe_applied: bool
    processing_time: float
    pyin_pitch: float | None = None
    pyin_confidence: float | None = None
    crepe_pitch: float | None = None
    crepe_confidence: float | None = None
    metadata: dict[str, Any] | None = None

    @property
    def yin_applied(self) -> bool:
        """Backward-Alias für pyin_applied."""
        return self.pyin_applied

    @property
    def yin_pitch(self) -> float | None:
        """Backward-Alias für pyin_pitch."""
        return self.pyin_pitch

    @property
    def yin_confidence(self) -> float | None:
        """Backward-Alias für pyin_confidence."""
        return self.pyin_confidence


class HybridSpeedPitch:
    """
    Hybrid Speed/Pitch Detection: YIN + CREPE.

    Detects global pitch for speed correction using adaptive YIN/CREPE strategy.
    """

    # §v10.303: Klassen-Variablen für Chunk-übergreifende Persistenz.
    # Phase 31 erzeugt pro Chunk eine neue Instanz — ohne diese Variablen
    # gehen pYIN-Disable und CREPE-Fallback nach Chunk 1 verloren.
    _pyin_disabled: bool = False
    _last_valid_crepe: tuple[float, float] | None = None

    def __init__(self, config: SpeedPitchConfig | None = None) -> None:
        """
        Initialisiert hybrid speed/pitch detector.

        Args:
            config: Speed/pitch configuration
        """
        self.config = config or SpeedPitchConfig()

        # Lazy-load ML pitch plugin chain
        self.crepe = None
        if self.config.strategy in [
            PitchDetectionStrategy.CREPE_ONLY,
            PitchDetectionStrategy.HYBRID,
            PitchDetectionStrategy.ADAPTIVE,
        ]:
            self._init_crepe()

    def _init_crepe(self) -> None:
        """Initialisiert pitch plugin: FCPE -> RMVPE -> PESTO -> pYIN (§4.4 Spec).

        Order: Tier-1 FCPE, Tier-2 RMVPE (Wei et al. ICASSP 2023, ~30 % lower pitch
        error for vocals), Tier-3 PESTO, Tier-4 pYIN (DSP-Endfall, §4.4).
        CREPE als Produktions-Tier VERBOTEN (Spec 04, Z. 1129) — Rev. 2026-08-16.
        VERBOTEN: FCPE -> CREPE -> RMVPE (RMVPE muss vor CREPE stehen — §4.4).
        """
        try:
            from plugins.fcpe_plugin import get_fcpe_plugin

            _fcpe_plugin = get_fcpe_plugin()
            if _fcpe_plugin is None:
                raise RuntimeError("FCPE plugin factory returned None")
            self.crepe = _fcpe_plugin  # type: ignore[assignment]
            logger.info(
                "FCPE plugin geladen for Verarbeitungsschritt 31 speed/pitch detection (model=%s)",
                getattr(_fcpe_plugin, "model_used", "fcpe"),
            )
            return
        except Exception as e:
            logger.debug("FCPE nicht verfügbar (%s) — RMVPE-Ersatzpfad (§4.4 Tier-2)", e)
        # Tier-2: RMVPE — before CREPE per §4.4 (30 % lower pitch error, Wei ICASSP 2023)
        try:
            from plugins.rmvpe_plugin import get_rmvpe_plugin

            self.crepe = get_rmvpe_plugin()  # type: ignore[assignment]
            logger.info("RMVPE plugin geladen for Verarbeitungsschritt 31 speed/pitch detection (§4.4 Tier-2)")
            return
        except Exception as e:
            logger.debug("RMVPE nicht verfügbar (%s) — CREPE-Ersatzpfad (§4.4 Tier-3)", e)
        # Tier-3: PESTO
        try:
            from plugins.pesto_plugin import get_pesto_plugin  # pylint: disable=no-name-in-module

            self.crepe = get_pesto_plugin()  # type: ignore[assignment]
            logger.info("PESTO plugin geladen for Verarbeitungsschritt 31 speed/pitch detection (§4.4 Tier-3)")
            return
        except Exception as e:
            logger.debug("PESTO nicht verfügbar (%s) — pYIN-DSP-Ersatzpfad (§4.4 Tier-4)", e)
        # Tier-4: pYIN (DSP-Endfall, §4.4). CREPE als Produktions-Tier VERBOTEN
        # (Spec 04, Z. 1129) — Rev. 2026-08-16.
        logger.warning("Kein Pitch-ML-Plugin verfügbar — pYIN-DSP-Ersatzpfad (§V6 (copilot-instructions.md))")
        try:
            from backend.core.fallback_auditor import get_fallback_auditor

            get_fallback_auditor().record("PitchDetection", "fcpe_rmvpe_pesto", "pyin_dsp", "all_ml_unavailable")
        except Exception:
            logger.debug("FallbackAuditor nicht verfügbar (unkritisch)", exc_info=True)
        self.crepe = None

    def detect_global_pitch(self, audio: np.ndarray, sample_rate: int = 48000) -> SpeedPitchResult:
        """
        Erkennt global pitch using hybrid YIN + CREPE.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate in Hz

        Returns:
            SpeedPitchResult with global pitch and metadata
        """
        import time

        start_time = time.time()

        # Convert to mono if stereo
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)

        strategy = self.config.strategy
        pyin_applied = False
        crepe_applied = False
        metadata = {}

        pyin_pitch = None
        pyin_confidence = None
        crepe_pitch = None
        crepe_confidence = None

        # Stufe 1: pYIN-Detektion (Mauch & Dixon 2014) — §4.2 konform
        # §v10.303: Nach erstem massivem Fail (conf < 0.15) pYIN für diesen Song deaktivieren.
        # pYIN confidence 0.08 auf cassette ist nutzlos — spart 2× 4s pro Folgelauf.
        if HybridSpeedPitch._pyin_disabled:
            pyin_pitch = 0.0
            pyin_confidence = 0.0
            logger.debug("pYIN: deaktiviert (vorheriger Fail conf < 0.15)")
        elif strategy in [
            PitchDetectionStrategy.PYIN_ONLY,
            PitchDetectionStrategy.ADAPTIVE,
            PitchDetectionStrategy.HYBRID,
        ]:
            logger.info("Stufe 1: pYIN-Globalpitch-Detektion (Mauch & Dixon 2014)...")
            pyin_pitch, pyin_confidence = self._apply_pyin_global(audio, sample_rate)
            pyin_applied = True

            metadata["pyin"] = {"pitch": float(pyin_pitch), "confidence": float(pyin_confidence)}

            logger.info("pYIN: pitch=%.2f Hz, confidence=%.3f", pyin_pitch, pyin_confidence)
            # §v10.303: Bei massivem Fail (conf < 0.15) pYIN für diesen Song deaktivieren
            if pyin_confidence < 0.15:
                HybridSpeedPitch._pyin_disabled = True
                logger.info("pYIN confidence %.3f < 0.15 → pYIN für diesen Song deaktiviert", pyin_confidence)

        # Stufe 2: ML-Detektion (bedingt: RMVPE/PESTO/CREPE)
        should_apply_crepe = False

        if strategy == PitchDetectionStrategy.CREPE_ONLY or strategy == PitchDetectionStrategy.HYBRID:
            should_apply_crepe = True
        elif strategy == PitchDetectionStrategy.ADAPTIVE:
            # CREPE anwenden wenn pYIN-Konfidenz niedrig
            if (pyin_confidence or 0.0) < self.config.confidence_threshold:
                logger.info(
                    f"pYIN confidence {pyin_confidence:.3f} < {self.config.confidence_threshold} -> ML anwenden"
                )
                should_apply_crepe = True
            else:
                logger.info("pYIN confidence %.3f ausreichend -> ML überspringen", pyin_confidence)

        if should_apply_crepe and self.crepe is not None:
            logger.info("Stufe 2: ML-Pitch-Detektion (RMVPE/PESTO/CREPE)...")
            crepe_pitch, crepe_confidence = self._apply_crepe_global(audio, sample_rate)
            crepe_applied = True

            metadata["crepe"] = {"pitch": float(crepe_pitch), "confidence": float(crepe_confidence)}

            logger.info("CREPE: pitch=%.2f Hz, confidence=%.3f", crepe_pitch, crepe_confidence)
            # §v10.303: CREPE-Null-Ergebnis (pitch≈0, conf≈0) → letztes gültiges Ergebnis verwenden
            if crepe_pitch < 1.0 and crepe_confidence < 0.01:
                if HybridSpeedPitch._last_valid_crepe is not None:
                    crepe_pitch, crepe_confidence = HybridSpeedPitch._last_valid_crepe
                    logger.info(
                        "CREPE Null-Ergebnis → letztes gültiges übernommen: pitch=%.2f Hz, conf=%.3f",
                        crepe_pitch,
                        crepe_confidence,
                    )
                else:
                    logger.info("CREPE Null-Ergebnis — kein vorheriges gültiges Ergebnis, verwerfe")
            elif crepe_confidence > 0.3:
                HybridSpeedPitch._last_valid_crepe = (float(crepe_pitch), float(crepe_confidence))

        # Finaler Pitch-Schätzwert
        final_pitch, final_confidence = self._combine_estimates(
            pyin_pitch, pyin_confidence, crepe_pitch, crepe_confidence, strategy
        )

        processing_time = time.time() - start_time

        logger.info(
            "Hybrid Speed/Pitch Detektion abgeschlossen: "
            f"pitch={final_pitch:.2f} Hz, confidence={final_confidence:.3f}, "
            f"strategy={strategy.value}, pYIN={pyin_applied}, CREPE={crepe_applied}, "
            f"time={processing_time:.2f}s"
        )

        return SpeedPitchResult(
            detected_pitch=final_pitch,
            confidence=final_confidence,
            strategy_used=strategy,
            pyin_applied=pyin_applied,
            crepe_applied=crepe_applied,
            processing_time=processing_time,
            pyin_pitch=pyin_pitch,
            pyin_confidence=pyin_confidence,
            crepe_pitch=crepe_pitch,
            crepe_confidence=crepe_confidence,
            metadata=metadata,
        )

    def _apply_pyin_global(self, audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
        """
        pYIN-basierte Globalpitch-Detektion (Mauch & Dixon 2014).

        Algorithmus:
            1. librosa.pyin: HMM-basierte voiced/unvoiced-Klassifikation
               f0[t], voiced_flag[t], voiced_probs[t] pro Frame
            2. Voiced-Frames filtern: f0[voiced_flag]
            3. Globaler Pitch = Median(voiced_f0)
            4. Konfidenz = voiced_fraction × mean(voiced_probs) ∈ [0, 1]

        Referenz: Mauch & Dixon (2014) pYIN, ISMIR

        Args:
            audio: Mono-Audio (float32, normalisiert)
            sample_rate: Abtastrate in Hz

        Returns:
            (global_pitch_hz, confidence) — (0.0, 0.0) bei keinem Pitch
        """
        import librosa

        # Auf Fensterlänge begrenzen (max 5 s) für Performance
        max_samples = int(5 * sample_rate)
        segment = audio[:max_samples].astype(np.float32)
        segment = np.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0)

        if len(segment) < 2048 or np.max(np.abs(segment)) < 1e-8:
            return 0.0, 0.0

        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                segment,
                fmin=librosa.note_to_hz("C2"),  # type: ignore[arg-type]  # ~65 Hz
                fmax=librosa.note_to_hz("C7"),  # type: ignore[arg-type]  # ~2093 Hz
                sr=sample_rate,
                frame_length=2048,
                hop_length=512,
            )

            voiced_f0 = f0[voiced_flag]
            voiced_p = voiced_probs[voiced_flag]

            if len(voiced_f0) == 0:
                return 0.0, 0.0

            global_pitch = float(np.median(voiced_f0))
            voiced_fraction = len(voiced_f0) / max(1, len(f0))
            mean_prob = float(np.mean(voiced_p))
            confidence = float(np.clip(voiced_fraction * mean_prob, 0.0, 1.0))

            return global_pitch, confidence

        except Exception as e:
            logger.debug("pYIN-Fensteranalyse fehlgeschlagen: %s, Notfall-librosa.yin", e)
            try:
                f0_yin = librosa.yin(segment, fmin=60, fmax=800, sr=sample_rate)
                valid = f0_yin[(f0_yin > 0) & np.isfinite(f0_yin)]
                if len(valid) == 0:
                    return 0.0, 0.0
                return float(np.median(valid)), 0.35  # Feste niedrige Konfidenz
            except Exception as e:
                logger.warning("hybrid_speed_pitch_ml.py::_anwenden_pyin_global Ersatzpfad: %s", e)
                return 0.0, 0.0

    def _apply_crepe_global(self, audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
        """
        Wendet an: ML pitch detector (RMVPE/PESTO/CREPE) for global pitch detection.

        Args:
            audio: Mono audio
            sample_rate: Sample rate

        Returns:
            (global_pitch, confidence)
        """
        if self.crepe is None:
            logger.warning("ML-Pitch-Plugin nicht verfügbar, Ersatzpfad auf pYIN")
            return self._apply_pyin_global(audio, sample_rate)

        try:
            # Run selected ML model on full audio (returns time-series)
            result = self.crepe.analyze(audio, sample_rate)

            # Extract pitch trajectory and confidence
            pitch_trajectory = result.f0_hz
            confidence_trajectory = result.voiced_prob

            if len(pitch_trajectory) == 0:
                return 0.0, 0.0

            # Filter valid estimates
            valid_mask = (confidence_trajectory > 0.5) & (pitch_trajectory > 0)
            valid_pitches = pitch_trajectory[valid_mask]
            valid_confidences = confidence_trajectory[valid_mask]

            if len(valid_pitches) == 0:
                return 0.0, 0.0

            # Weighted average
            global_pitch = np.average(valid_pitches, weights=valid_confidences)
            global_confidence = np.mean(valid_confidences)

            return float(global_pitch), float(global_confidence)

        except Exception as e:
            logger.error("ML pitch processing fehlgeschlagen: %s", e)
            return 0.0, 0.0

    def _combine_estimates(
        self,
        pyin_pitch: float | None,
        pyin_confidence: float | None,
        crepe_pitch: float | None,
        crepe_confidence: float | None,
        strategy: PitchDetectionStrategy,
    ) -> tuple[float, float]:
        """
        pYIN- und CREPE-Schätzwerte gemäß Strategie kombinieren.

        Args:
            pyin_pitch: pYIN-Pitch-Schätzwert
            pyin_confidence: pYIN-Konfidenz
            crepe_pitch: CREPE-Pitch-Schätzwert
            crepe_confidence: CREPE-Konfidenz
            strategy: Detektions-Strategie

        Returns:
            (final_pitch, final_confidence)
        """
        if strategy in [PitchDetectionStrategy.PYIN_ONLY, PitchDetectionStrategy.YIN_ONLY]:
            return pyin_pitch or 0.0, pyin_confidence or 0.0

        elif strategy == PitchDetectionStrategy.CREPE_ONLY:
            return crepe_pitch or 0.0, crepe_confidence or 0.0

        elif strategy == PitchDetectionStrategy.ADAPTIVE:
            # CREPE wenn verfügbar, sonst pYIN
            if crepe_pitch is not None and crepe_confidence is not None:
                return crepe_pitch, crepe_confidence
            else:
                return pyin_pitch or 0.0, pyin_confidence or 0.0

        elif strategy == PitchDetectionStrategy.HYBRID:
            # Gewichtete Kombination
            if pyin_pitch and crepe_pitch:
                total_conf = (pyin_confidence or 0.0) + (crepe_confidence or 0.0)
                if total_conf > 0:
                    weighted_pitch = (
                        pyin_pitch * (pyin_confidence or 0.0) + crepe_pitch * (crepe_confidence or 0.0)
                    ) / total_conf
                    combined_confidence = total_conf / 2
                    return weighted_pitch, combined_confidence

            # Fallback: best verfügbar
            if crepe_pitch is not None:
                return crepe_pitch, crepe_confidence or 0.0
            elif pyin_pitch is not None:
                return pyin_pitch, pyin_confidence or 0.0

        return 0.0, 0.0
