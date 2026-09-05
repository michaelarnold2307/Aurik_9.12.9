"""
Multi-Pass Processing Strategy & Adaptive Selection

Implements GAP #1: Multi-Pass Strategy für vollautomatisches Audio-Processing.
Ein weltklasse-automatisches System sollte mehrere Ansätze testen und automatisch
den besten wählen basierend auf objektiven Metriken.

Architecture:
1. ProcessingVariant - Beschreibt eine Processing-Strategie mit Parametern
2. ObjectiveScorer - Bewertet Resultate via VERSA, DNSMOS, Musical Goals
3. MultiPassEngine - Führt Audio durch mehrere Varianten, wählt beste
4. ConfidenceCalculator - Berechnet Confidence basierend auf variance

Author: AURIK Development Team
Version: 1.0
Date: 2026-02-10
"""

import copy
import logging
import time
import traceback as _tb
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np

from backend.core.processing_modes import ProcessingConfig, ProcessingMode, get_processing_config

logger = logging.getLogger(__name__)

# Singleton IAQS-Instanz (kein State, daher Thread-safe)
# _iaqs is lazily initialized via _get_iaqs() below after IntrinsicAudioQualityScorer is defined
_iaqs = None


def _get_iaqs():
    global _iaqs
    if _iaqs is None:
        _iaqs = IntrinsicAudioQualityScorer()
    return _iaqs


class VariantStrategy(Enum):
    """Strategien für ProcessingVarianten."""

    CONSERVATIVE = "conservative"
    """Conservative: Minimal processing, max preservation."""

    BALANCED = "balanced"
    """Balanced: Moderate processing, good quality/authenticity trade-off."""

    AGGRESSIVE = "aggressive"
    """Aggressive: Strong processing, maximum quality improvement."""

    GENTLE_DENOISE = "gentle_denoise"
    """Focus: Ultra-gentle denoising, preserve everything else."""

    STRONG_DYNAMICS = "strong_dynamics"
    """Focus: Strong dynamics control, moderate other processing."""

    NATURALNESS_FIRST = "naturalness_first"
    """Focus: Maximale Natürlichkeit — minimales Processing, volle Authentizität."""


@dataclass
class ProcessingVariant:
    """
    Beschreibt eine Processing-Variante mit spezifischen Parametern.

    Jede Variante ist eine andere "Strategie" das Material zu restaurieren.
    Multi-Pass Engine testet 3-5 Varianten und wählt die beste.
    """

    name: str
    """Eindeutiger Name (z.B. 'conservative', 'balanced', 'aggressive')."""

    strategy: VariantStrategy
    """High-level Strategie-Typ."""

    config: ProcessingConfig
    """Konkrete Processing-Parameter."""

    description: str = ""
    """Optionale Beschreibung der Variante."""

    weight: float = 1.0
    """Gewichtung im Scoring (default: 1.0 = equal weight)."""

    def to_dict(self) -> dict[str, Any]:
        """Konvertiere zu Dictionary für Serialisierung."""
        return {
            "name": self.name,
            "strategy": self.strategy.value,
            "config": asdict(self.config),
            "description": self.description,
            "weight": self.weight,
        }

    @classmethod
    def create_conservative(cls, base_mode: ProcessingMode = ProcessingMode.RESTORATION) -> "ProcessingVariant":
        """
        Erstelle Conservative Variante: Minimal processing.

        Gut für:
        - Material mit wenig Defekten
        - Maximale Authentizität gefragt
        - Archival/Forensic use cases
        """
        config = copy.deepcopy(get_processing_config(base_mode))

        # Ultra-conservative adjustments
        config.denoise_strength *= 0.5
        config.declip_strength *= 0.7
        config.click_removal_sensitivity *= 0.6
        config.compression_ratio = min(config.compression_ratio, 2.5)
        config.preserve_breaths = True
        config.preserve_room_tone = True
        config.preserve_analog_character = True

        return cls(
            name="conservative",
            strategy=VariantStrategy.CONSERVATIVE,
            config=config,
            description="Minimal processing, maximum preservation",
            weight=1.0,
        )

    @classmethod
    def create_balanced(cls, base_mode: ProcessingMode = ProcessingMode.RESTORATION) -> "ProcessingVariant":
        """
        Erstelle Balanced Variante: Standard processing.

        Gut für:
        - Typisches Material
        - Balance zwischen Quality und Authenticity
        - Default Ansatz
        """
        config = copy.deepcopy(get_processing_config(base_mode))

        # Balanced = use base config as-is

        return cls(
            name="balanced",
            strategy=VariantStrategy.BALANCED,
            config=config,
            description="Balanced processing, good quality/authenticity trade-off",
            weight=1.0,
        )

    @classmethod
    def create_aggressive(cls, base_mode: ProcessingMode = ProcessingMode.STUDIO_2026) -> "ProcessingVariant":
        """
        Erstelle Aggressive Variante: Strong processing.

        Gut für:
        - Stark defektes Material
        - Maximum Quality improvement gefragt
        - Competitive sound
        """
        config = copy.deepcopy(get_processing_config(base_mode))

        # Aggressive adjustments
        config.denoise_strength *= 1.3
        config.declip_strength *= 1.2
        config.click_removal_sensitivity *= 1.2
        config.compression_ratio *= 1.3
        config.enhancement_strength = min(config.enhancement_strength * 1.3, 1.0)

        # Clamp to valid ranges
        config.denoise_strength = min(config.denoise_strength, 1.0)
        config.declip_strength = min(config.declip_strength, 1.0)
        config.click_removal_sensitivity = min(config.click_removal_sensitivity, 1.0)

        return cls(
            name="aggressive",
            strategy=VariantStrategy.AGGRESSIVE,
            config=config,
            description="Strong processing, maximum quality improvement",
            weight=1.0,
        )

    @classmethod
    def create_gentle_denoise(cls, base_mode: ProcessingMode = ProcessingMode.RESTORATION) -> "ProcessingVariant":
        """
        Erstelle Gentle Denoise Variante: Ultra-gentle noise reduction only.

        Gut für:
        - Leichtes Background Noise
        - Preservation kritisch
        - Vintage recordings
        """
        config = copy.deepcopy(get_processing_config(base_mode))

        # Focus on gentle denoising
        config.denoise_strength = 0.15  # Ultra-gentle
        config.declip_strength = 0.3  # Minimal
        config.click_removal_sensitivity = 0.3  # Minimal
        config.compression_ratio = 1.5  # Very light compression
        config.preserve_breaths = True
        config.preserve_room_tone = True
        config.preserve_analog_character = True

        return cls(
            name="gentle_denoise",
            strategy=VariantStrategy.GENTLE_DENOISE,
            config=config,
            description="Ultra-gentle denoising, preserve everything else",
            weight=1.0,
        )

    @classmethod
    def create_strong_dynamics(cls, base_mode: ProcessingMode = ProcessingMode.STUDIO_2026) -> "ProcessingVariant":
        """
        Erstelle Strong Dynamics Variante: Focus on dynamics control.

        Gut für:
        - Extreme dynamic range
        - Competitive loudness
        - Modern streaming
        """
        config = copy.deepcopy(get_processing_config(base_mode))

        # Focus on dynamics
        config.compression_ratio = 6.0  # Strong compression
        config.compression_threshold_db = -30.0  # Lower threshold
        config.target_lufs = -14.0  # Streaming standard
        config.denoise_strength = 0.4  # Moderate
        config.preserve_breaths = False  # Allow breath reduction for loudness

        return cls(
            name="strong_dynamics",
            strategy=VariantStrategy.STRONG_DYNAMICS,
            config=config,
            description="Strong dynamics control, moderate other processing",
            weight=1.0,
        )

    @classmethod
    def create_naturalness_first(cls, base_mode: ProcessingMode = ProcessingMode.RESTORATION) -> "ProcessingVariant":
        """
        Erstelle Naturalness-First Variante: Maximale Natürlichkeit.

        Gut für:
        - Vintage-Aufnahmen mit Charakter
        - Archivierung (originaler Charakter erhalten)
        - Material wo Over-Processing schlimmer wäre als Rest-Artefakte

        Strategie:
        - Enhancement auf 0% (keine künstlichen Verbesserungen)
        - Nur defekte Stellen korrigieren (Klicks, Clips) — sehr sanft
        - Raum-Ton, Atem, Analogcharakter vollständig erhalten
        - Kompression fast abgeschaltet (natürliche Dynamik)
        """
        config = copy.deepcopy(get_processing_config(base_mode))

        # Naturalness: so wenig Eingriff wie möglich
        config.denoise_strength = 0.08  # Fast kein Rauschen entfernen
        config.declip_strength = 0.25  # Nur harte Clips korrigieren
        config.click_removal_sensitivity = 0.20  # Nur deutliche Klicks
        config.compression_ratio = 1.2  # Annähernd keine Kompression
        config.enhancement_strength = 0.0  # Keine künstliche Verbesserung
        config.preserve_breaths = True
        config.preserve_room_tone = True
        config.preserve_analog_character = True

        return cls(
            name="naturalness_first",
            strategy=VariantStrategy.NATURALNESS_FIRST,
            config=config,
            description="Maximale Natürlichkeit: minimales Processing, volle Authentizität",
            weight=1.05,  # Leicht bevorzugt bei RESTORATION
        )


@dataclass
class ObjectiveScore:
    """
    Objektive Quality Scores für ein verarbeitetes Audio.

    Kombiniert:
    - VERSA (non-reference MOS, 1.0–5.0 Skala, §4.4)
    - DNSMOS (Speech Quality, 3.5-5.0 = sehr gut)
    - Musical Goals (0.0-1.0, >0.7 = excellent)
    - Signal Stats (SNR, THD, etc.)
    """

    # === Perceptual Metrics ===
    versa_score: float = 0.0
    """VERSA Score (normiert 0.0-1.0, höher=besser, §4.4)."""

    cdpam_score: float = 0.0
    """Legacy alias for VERSA migration compatibility (historically lower=better)."""

    dnsmos_score: float = 0.0
    """DNSMOS P.835 (1.0-5.0, higher=better, >3.5=good)."""

    # === Musical Goals ===
    musical_goals_avg: float = 0.0
    """Average Musical Goals Score (0.0-1.0, higher=better)."""

    musical_goals_min: float = 0.0
    """Minimum Musical Goals Score (detectiert worst-case goal)."""

    # === Signal Statistics ===
    snr_db: float = 0.0
    """Signal-to-Noise Ratio in dB (higher=better)."""

    thd_percent: float = 0.0
    """Total Harmonic Distortion in % (lower=better)."""

    # === IAQS Holistic Score ===
    iaqs_total: float = 0.0
    """Intrinsic Audio Quality Score gesamt (0.0-1.0, higher=better)."""

    # === Flags: welche Metriken tatsächlich berechnet wurden ===
    iaqs_active: bool = False
    """True wenn IAQS tatsächlich berechnet (nicht Default-Wert)."""

    versa_active: bool = False
    """True wenn VERSA tatsächlich berechnet (nicht Default-Wert)."""

    dnsmos_active: bool = False
    """True wenn DNSMOS tatsächlich berechnet (nicht Default-Wert)."""

    # === Composite Scores ===
    composite_score: float = 0.0
    """Weighted composite score (0.0-1.0, higher=better)."""

    # === §v10 HPE: Psychoakustische Angenehmheit ===
    pleasantness_score: float = 0.0
    """HPE Pleasantness Score (0.0-1.0). DAS ist, worauf es für menschliche Ohren ankommt."""
    pleasantness_active: bool = False
    """True wenn HPE tatsächlich berechnet wurde."""
    pleasantness_label: str = ""
    """Mensch-lesbare Bewertung: 'Sehr angenehm', 'Angenehm', 'Neutral', 'Anstrengend'."""
    pleasantness_delta_vs_original: float = 0.0
    """Veränderung der Angenehmheit gegenüber dem Original."""

    confidence: float = 0.0
    """Confidence in this score (0.0-1.0, based on consistency)."""

    # === Metadata ===
    variant_name: str = ""
    """Name der Variante die diesen Score produziert hat."""

    processing_time_sec: float = 0.0
    """Processing time in Sekunden."""

    def __post_init__(self) -> None:
        """Keep legacy cdpam/versa fields consistent for backward compatibility."""
        if self.versa_score <= 0.0 and self.cdpam_score > 0.0:
            # Legacy CDPAM distance (lower=better) to quality-like [0,1] score.
            self.versa_score = float(np.clip(1.0 - self.cdpam_score, 0.0, 1.0))
        elif self.cdpam_score <= 0.0 and self.versa_score > 0.0:
            self.cdpam_score = float(np.clip(1.0 - self.versa_score, 0.0, 1.0))

    def to_dict(self) -> dict[str, Any]:
        """Konvertiere zu Dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        """Human-readable representation."""
        hpe_str = f" HPE={self.pleasantness_score:.3f}({self.pleasantness_label})" if self.pleasantness_active else ""
        return (
            f"ObjectiveScore(variant='{self.variant_name}', "
            f"composite={self.composite_score:.3f},"
            f"{hpe_str}"
            f" confidence={self.confidence:.2f}, "
            f"VERSA={self.versa_score:.3f}, "
            f"MG_avg={self.musical_goals_avg:.2f}, "
            f"SNR={self.snr_db:.1f}dB)"
        )


class IntrinsicAudioQualityScorer:
    """§v10 Leichtgewichtiger Scorer für Multi-Pass-Varianten-Evaluation.

    Berechnet einen einfachen Composite-Score aus:
    - RMS-Erhalt (keine Pegel-Explosion)
    - Peak-Erhalt (kein Clipping)
    - SNR-Verbesserung (Signal/Rausch-Abstand)
    - Spektrale Ähnlichkeit (keine drastische Klangfarben-Änderung)
    """

    def score(self, original: np.ndarray, processed: np.ndarray, sr: int) -> float:
        import numpy as np

        orig = np.asarray(original, dtype=np.float64)
        proc = np.asarray(processed, dtype=np.float64)
        # Mono für Vergleich
        if orig.ndim > 1:
            orig = orig.mean(axis=-1) if orig.shape[-1] <= 2 else orig.mean(axis=0)
        if proc.ndim > 1:
            proc = proc.mean(axis=-1) if proc.shape[-1] <= 2 else proc.mean(axis=0)
        min_len = min(len(orig), len(proc))
        orig, proc = orig[:min_len], proc[:min_len]

        # 1. RMS-Erhalt (Score 0-25)
        rms_orig = float(np.sqrt(np.mean(orig**2)) + 1e-12)
        rms_proc = float(np.sqrt(np.mean(proc**2)) + 1e-12)
        rms_ratio = min(rms_proc, rms_orig) / max(rms_proc, rms_orig, 1e-12)
        rms_score = 25.0 * rms_ratio

        # 2. Peak-Erhalt (Score 0-25)
        float(np.max(np.abs(orig)))
        peak_proc = float(np.max(np.abs(proc)))
        peak_ok = 1.0 if peak_proc < 0.99 else 0.5 if peak_proc < 1.0 else 0.0
        peak_score = 25.0 * peak_ok

        # 3. SNR-Verbesserung (Score 0-25)
        noise = proc - orig
        noise_power = float(np.mean(noise**2)) + 1e-12
        signal_power = float(np.mean(orig**2)) + 1e-12
        snr_db = 10.0 * np.log10(signal_power / noise_power)
        snr_score = 25.0 * min(1.0, max(0.0, (snr_db + 10.0) / 40.0))

        # 4. Spektrale Ähnlichkeit (Score 0-25)
        n_fft = min(2048, min_len // 4)
        if n_fft >= 64:
            spec_orig = np.abs(np.fft.rfft(orig[: n_fft * 10] * np.hanning(n_fft * 10)))[: n_fft // 2]
            spec_proc = np.abs(np.fft.rfft(proc[: n_fft * 10] * np.hanning(n_fft * 10)))[: n_fft // 2]
            spec_corr = float(np.corrcoef(spec_orig, spec_proc)[0, 1]) if len(spec_orig) > 1 else 1.0
            spec_score = 25.0 * max(0.0, spec_corr)
        else:
            spec_score = 25.0

        return rms_score + peak_score + snr_score + spec_score  # type: ignore[no-any-return]


class ObjectiveScorer:
    """
    Bewertet Audio via objektive Metriken.

    Integriert:
    - VERSA Plugin (wenn verfügbar, §4.4)
    - DNSMOS Plugin (wenn available)
    - Musical Goals Checker
    - Enhanced Metrics (SNR, THD, etc.)
    """

    def __init__(
        self,
        enable_versa: bool | None = None,
        enable_dnsmos: bool = False,
        enable_musical_goals: bool = True,
        enable_cdpam: bool | None = None,
    ):
        """
        Initialize ObjectiveScorer.

        Args:
            enable_versa: Enable VERSA scoring (wenn Plugin verfügbar)
            enable_dnsmos: PERMANENT FALSE — DNSMOS P.835 ist auf Sprachkorpus trainiert
                           (16 kHz DNS-Challenge) und ist VERBOTEN als Musik-Metrik (§10.2).
                           Parameter bleibt aus Rückwärtskompatibilität erhalten, ist aber wirkungslos.
            enable_musical_goals: Enable Musical Goals scoring
            enable_cdpam: Legacy alias (mapped to VERSA toggle for compatibility)
        """
        if enable_versa is None:
            enable_versa = True if enable_cdpam is None else bool(enable_cdpam)

        self.enable_versa = enable_versa
        self.enable_cdpam = enable_versa  # Backward-compatible attribute alias
        self.enable_dnsmos = enable_dnsmos
        self.enable_musical_goals = enable_musical_goals

        # Try to import plugins (may fail if not installed)
        # §4.4: VERSA 2024 ist primäre non-reference MOS-Metrik
        self.versa_plugin = None
        self.dnsmos_plugin = None

        if enable_versa:
            try:
                from plugins.versa_plugin import (  # pylint: disable=import-outside-toplevel
                    get_loaded_versa_plugin,
                    get_versa_plugin,
                )

                self.versa_plugin = get_loaded_versa_plugin()
                if self.versa_plugin is None:
                    self.versa_plugin = get_versa_plugin()
                logger.info("✓ VERSA Plugin geladen (§4.4, non-Referenz MOS)")
            except Exception as e:
                logger.warning("⚠ VERSA Plugin not verfuegbar: %s", e)

        # §10.2: DNSMOS-Plugin wird nicht geladen — DNSMOS P.835 ist auf Sprachkorpus
        # trainiert (16 kHz DNS-Challenge) und ist VERBOTEN als Musik-Qualitätsmetrik.

    def score(
        self,
        audio: np.ndarray,
        sample_rate: int,
        variant_name: str,
        reference_audio: np.ndarray | None = None,
        processing_time_sec: float = 0.0,
    ) -> ObjectiveScore:
        """
        Bewertet audio via objektive Metriken.

        Args:
            audio: Processed audio array
            sample_rate: Sample rate
            variant_name: Name der Variante
            reference_audio: Nicht genutzt (VERSA ist non-reference
            processing_time_sec: Processing time

        Returns:
            ObjectiveScore mit allen Metriken
        """
        # Non-reference-Scorer: Parameter bleibt für API-Kompatibilität erhalten.
        del reference_audio
        score = ObjectiveScore(variant_name=variant_name, processing_time_sec=processing_time_sec)

        # === 1. VERSA: non-reference MOS (§4.4 VERSA §4.4) ===
        if self.enable_versa and self.versa_plugin is not None:
            try:
                proc_arr = np.asarray(audio, dtype=np.float32)
                if proc_arr.ndim == 2:
                    # Accept both layouts: [N, C] and [C, N].
                    if proc_arr.shape[0] <= 2 and proc_arr.shape[1] > proc_arr.shape[0]:
                        proc_arr = proc_arr.mean(axis=0)
                    elif proc_arr.shape[1] <= 2 and proc_arr.shape[0] > proc_arr.shape[1]:
                        proc_arr = proc_arr.mean(axis=1)
                    else:
                        proc_arr = proc_arr.mean(axis=-1)
                proc_arr = np.nan_to_num(proc_arr, nan=0.0, posinf=0.0, neginf=0.0)
                versa_result = self.versa_plugin.score(proc_arr, sample_rate)
                # MOS [1,5] → [0,1] skaliert → versa_score (§4.4)
                score.versa_score = float(np.clip((versa_result.mos - 1.0) / 4.0, 0.0, 1.0))
                score.versa_active = True

            except Exception as e:
                logger.warning("VERSA scoring fehlgeschlagen: %s", e)
                score.versa_score = 0.5  # Neutral default, versa_active bleibt False

        # §10.2: DNSMOS-Berechnung deaktiviert — Sprach-Metrik verboten für Musikrestaurierung.
        # score.dnsmos_active bleibt False; score.dnsmos_score bleibt 0.0.

        # === 3. Musical Goals ===
        if self.enable_musical_goals:
            _mg_loaded = False
            try:
                from backend.core.musical_goals.musical_goals_metrics import (  # pylint: disable=import-outside-toplevel
                    get_checker,
                )

                checker = get_checker()  # §v10.101: Singleton → Cache wirkt
                goals_scores = checker.measure_all(audio, sample_rate)

                if goals_scores:
                    values = list(goals_scores.values())
                    score.musical_goals_avg = float(np.mean(values))
                    score.musical_goals_min = float(np.min(values))
                    _mg_loaded = True

            except Exception as e:
                logger.debug("MusicalGoalsChecker nicht verfügbar (%s) — IAQS-Ersatzpfad", e)

            if not _mg_loaded:
                # Intrinsischer Fallback: IAQS liefert psychoakustisch fundierte Scores
                try:
                    iaqs = _get_iaqs().score(audio, sample_rate)
                    # harmonicity + bark_balance + spectral_regularity ≈ musikalische Güte
                    mg_approx = (iaqs.harmonicity + iaqs.bark_balance + iaqs.spectral_regularity) / 3.0
                    score.musical_goals_avg = float(np.clip(mg_approx, 0.0, 1.0))
                    score.musical_goals_min = float(min(iaqs.harmonicity, iaqs.bark_balance, iaqs.spectral_regularity))
                    logger.debug("IAQS Musical-Goals-Ersatzpfad: avg=%.3f", score.musical_goals_avg)
                except Exception as e2:
                    logger.warning("IAQS-Ersatzpfad fehlgeschlagen: %s", e2)
                    score.musical_goals_avg = 0.5
                    score.musical_goals_min = 0.5

        # === 4. Signal Statistics + IAQS Holistic (primär, keine externen Deps) ===
        try:
            iaqs_stats = _get_iaqs().score(audio, sample_rate)
            score.snr_db = float(iaqs_stats.snr_estimate)
            score.thd_percent = float(iaqs_stats.thd_estimate_pct)
            # IAQS gesamt Score direkt aus bereits berechnetem Objekt (kein Doppel-Call)
            score.iaqs_total = float(np.clip(iaqs_stats.overall, 0.0, 1.0))
            score.iaqs_active = True
            logger.debug("IAQS: SNR=%.1f dB, THD=%.2f%%, Total=%.3f", score.snr_db, score.thd_percent, score.iaqs_total)
        except Exception as e:
            logger.warning("IAQS Signal statistics fehlgeschlagen: %s", e)
            # Letzter Fallback: EnhancedMetrics Backend
            try:
                from backend.core.enhanced_metrics import EnhancedMetrics  # pylint: disable=import-outside-toplevel

                metrics = EnhancedMetrics()
                try:
                    score.snr_db = metrics.compute_snr(audio, sample_rate)
                except Exception:
                    logger.debug("multi_pass_strategy: SNR computation failed, using default 20.0", exc_info=True)
                    score.snr_db = 20.0
                try:
                    score.thd_percent = metrics.compute_thd(audio, sample_rate)
                except Exception:
                    logger.debug("multi_pass_strategy: THD computation failed, using default 1.0", exc_info=True)
                    score.thd_percent = 1.0
            except Exception as e2:
                logger.warning("EnhancedMetrics auch nicht verfügbar: %s", e2)
                score.snr_db = 20.0
                score.thd_percent = 1.0

        # === 5. §v10 HPE: Psychoakustische Angenehmheit (PRIMÄR) ===
        try:
            from backend.core.human_pleasantness_estimator import (
                compute_pleasantness,
            )

            hpe_result = compute_pleasantness(audio, sample_rate)
            score.pleasantness_score = float(hpe_result.score)
            score.pleasantness_active = True
            score.pleasantness_label = hpe_result.label
            # Wenn Referenz-Audio verfügbar, berechne Delta
            # (reference_audio wird als Parameter durchgereicht, kann None sein)
            logger.debug("HPE: Wert=%.3f Label=%s", score.pleasantness_score, score.pleasantness_label)
        except Exception as e:
            logger.debug("HPE nicht verfügbar: %s", e)
            score.pleasantness_score = 0.5
            score.pleasantness_active = False

        # === 6. Composite Score (weighted) ===
        score.composite_score = self._calculate_composite(score)

        # === 7. Confidence (ConfidenceCalculator, §v10 Multi-Pass Selection) ===
        # Hinweis: Dieser Score ist ein Einzel-Ergebnis; Confidence wird später
        # in MultiPassEngine.calculate_confidence() berechnet (Vergleich aller Varianten).
        score.confidence = 0.5  # Neutraler Startwert; wird durch MultiPassEngine überschrieben

        return score

    def _calculate_composite(self, score: ObjectiveScore) -> float:
        """
        §v10 Berechne weighted composite score — HPE als PRIMÄRE Dimension.

        Gewichtung:
        - HPE Pleasantness (wenn aktiv):       35%  ← PRIMÄR: Menschlicher Wohlklang
        - Musical Goals Average:               20%
        - VERSA (normiert, wenn aktiv):        15%
        - IAQS-Gesamt (wenn aktiv):            15%
        - SNR (normalisiert):                   8%
        - THD (invertiert, normalisiert):       7%

        Wenn HPE nicht verfügbar: Gewicht wird umverteilt.

        Result: 0.0-1.0, higher=better
        """
        base_hpe = 0.35
        base_mg = 0.20
        base_iaqs = 0.15
        base_versa = 0.15
        base_snr = 0.08
        base_thd = 0.07

        freed = 0.0
        if not score.versa_active:
            freed += base_versa
        if not score.iaqs_active:
            freed += base_iaqs
        if not score.pleasantness_active:
            freed += base_hpe

        w_hpe = base_hpe if score.pleasantness_active else 0.0
        w_iaqs = base_iaqs if score.iaqs_active else 0.0
        w_mg = base_mg + freed * 0.45
        w_snr = base_snr + freed * 0.35
        w_thd = base_thd + freed * 0.20

        composite = 0.0

        # §v10 HPE — das wichtigste Kriterium
        if score.pleasantness_active:
            composite += w_hpe * score.pleasantness_score

        # IAQS Gesamt (0.0-1.0)
        if score.iaqs_active:
            composite += w_iaqs * score.iaqs_total

        # Musical Goals (0.0-1.0)
        composite += w_mg * score.musical_goals_avg

        # VERSA (normiert 0.0–1.0)
        if score.versa_active:
            versa_norm = min(score.versa_score, 1.0)
            composite += base_versa * versa_norm

        # SNR (typ. 10-40 dB → 0.0-1.0)
        snr_norm = min(max((score.snr_db - 10.0) / 30.0, 0.0), 1.0)
        composite += w_snr * snr_norm

        # THD (0-10% → 0.0-1.0, lower=better → invert)
        thd_norm = 1.0 - min(score.thd_percent / 10.0, 1.0)
        composite += w_thd * thd_norm

        return min(max(composite, 0.0), 1.0)


class ConfidenceCalculator:
    """
    Berechnet Confidence für Multi-Pass Selection.

    Confidence basiert auf:
    - Variance zwischen top-2 Varianten (geringe variance = high confidence)
    - Consistency across Metriken (alle Metriken zeigen gleiche Variante = high)
    - Absolute composite score (sehr hoher Score = high confidence)
    """

    @staticmethod
    def calculate_confidence(scores: list[ObjectiveScore], best_score: ObjectiveScore) -> float:
        """
        Berechne Confidence für best_score Selection.

        Args:
            scores: Liste aller Scores (sorted by composite, descending)
            best_score: Der beste Score

        Returns:
            Confidence (0.0-1.0), 1.0 = very confident, 0.0 = guessing
        """
        if len(scores) < 2:
            return 0.5  # Not enough data for confidence

        # === 1. Top-2 Variance ===
        # Je größer der Gap zwischen #1 und #2, desto confidenter
        top1_composite = scores[0].composite_score
        top2_composite = scores[1].composite_score

        gap = top1_composite - top2_composite

        # Gap > 0.15 → very confident
        # Gap < 0.05 → not confident
        variance_confidence = min(gap / 0.15, 1.0)

        # === 2. Absolute Quality ===
        # Sehr hoher composite score = zusätzliche confidence
        absolute_confidence = best_score.composite_score

        # === 3. Metric Consistency ===
        # How much do individual metrics agree on the ranking?
        # High agreement (best variant leads on MG avg AND worst-case goal) → high consistency.
        mg_lead = scores[0].musical_goals_avg - scores[1].musical_goals_avg
        mg_min_lead = scores[0].musical_goals_min - scores[1].musical_goals_min
        # Map ±0.2-range to [0, 1] with centre 0.5
        consistency = min(max(0.5 + (mg_lead + mg_min_lead) * 2.5, 0.0), 1.0)

        # === Final Confidence (weighted) ===
        confidence = 0.50 * variance_confidence + 0.30 * absolute_confidence + 0.20 * consistency
        # NaN/Inf-Guard (§3.1)
        confidence = np.nan_to_num(confidence, nan=0.5, posinf=1.0, neginf=0.0)
        return float(np.clip(confidence, 0.0, 1.0))


class MultiPassEngine:
    """
    Multi-Pass Processing Engine.

    Führt Audio durch mehrere ProcessingVarianten,
    scored jede Variante via objektive Metriken,
    wählt automatisch die beste.

    Usage:
        engine = MultiPassEngine()
        result = engine.process_with_variants(
            audio, sample_rate,
            variants=[
                ProcessingVariant.create_conservative(),
                ProcessingVariant.create_balanced(),
                ProcessingVariant.create_aggressive()
            ]
        )

        best_audio = result["audio"]
        best_variant_name = result["variant_name"]
        confidence = result["confidence"]
    """

    def __init__(self, scorer: ObjectiveScorer | None = None):
        """
        Initialisiert MultiPassEngine.

        Args:
            scorer: ObjectiveScorer instance (creates default if None)
        """
        self.scorer = scorer or ObjectiveScorer()
        self.confidence_calc = ConfidenceCalculator()
        self._restorer: Any | None = None  # gecachte UnifiedRestorerV3-Instanz (einmalig laden)

    def process_with_variants(
        self,
        audio: np.ndarray,
        sample_rate: int,
        variants: list[ProcessingVariant],
        reference_audio: np.ndarray | None = None,
        process_func: Callable | None = None,
        progress_callback=None,
    ) -> dict[str, Any]:
        """
        Process audio mit mehreren Varianten, wähle beste.

        Args:
            audio: Input audio array
            sample_rate: Sample rate
            variants: Liste von ProcessingVarianten zu testen
            reference_audio: Nicht genutzt (VERSA ist non-reference scoring
            process_func: Custom processing function (audio, config) -> processed_audio
                         Falls None, wird UnifiedRestorerV3 verwendet

        Returns:
            Dict mit:
            - "audio": Best processed audio
            - "variant_name": Name der besten Variante
            - "variant_strategy": Strategy enum
            - "confidence": Confidence (0.0-1.0)
            - "composite_score": Composite score
            - "all_scores": Liste aller ObjectiveScores
            - "processing_times": Dict variant_name -> time_sec
        """
        if len(variants) == 0:
            raise ValueError("Mindestens 1 ProcessingVariant erforderlich")

        logger.info("🎯 Multi-Pass Processing: testing %s variants...", len(variants))

        # Default processing function
        if process_func is None:
            process_func = self._default_process_func

        # Process mit jeder Variante
        results = []
        processing_times = {}

        for variant in variants:
            try:
                logger.info("  Processing with '%s' (%s)...", variant.name, variant.strategy.value)
                logger.debug("[MPASS] Starte Variante '%s' …", variant.name)

                # —— Emit variant start — real-time progress
                _vi = variants.index(variant)
                _vn = max(len(variants), 1)
                _vpct = int(100 * _vi / _vn)
                _cpct = int(100 * (_vi + 1) / _vn)
                if progress_callback is not None:
                    try:
                        progress_callback(
                            _vpct,
                            f"Variante {_vi + 1}/{_vn}: '{variant.name}' wird bewertet …",
                            0.0,
                        )
                    except Exception as _exc:
                        logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

                # Sub-progress: map UV3's 0–100 into this variant's slice of the outer bar
                def _make_sub_cb(_base: int, _span: int):
                    def _sub_progress(pct, phase, elapsed=0.0):
                        if progress_callback is not None:
                            try:
                                progress_callback(_base + int(pct * _span / 100), phase, elapsed)
                            except Exception as _exc:
                                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

                    return _sub_progress

                _sub_cb = _make_sub_cb(_vpct, max(1, _cpct - _vpct)) if progress_callback is not None else None

                start_time = time.time()
                try:
                    processed_audio = process_func(audio, sample_rate, variant.config, progress_callback=_sub_cb)
                except TypeError:
                    # Custom process_func without progress_callback support — graceful fallback
                    processed_audio = process_func(audio, sample_rate, variant.config)
                proc_time = time.time() - start_time
                logger.debug("[MPASS] Variante '%s' fertig in %.1fs", variant.name, proc_time)

                processing_times[variant.name] = proc_time

                # Score result
                score = self.scorer.score(
                    audio=processed_audio,
                    sample_rate=sample_rate,
                    variant_name=variant.name,
                    reference_audio=reference_audio,
                    processing_time_sec=proc_time,
                )

                results.append({"audio": processed_audio, "variant": variant, "score": score})

                # Emit score result for frontend variant-ranking display
                if progress_callback is not None:
                    try:
                        _cpct = int(100 * (_vi + 1) / _vn)
                        _mos_v = getattr(score, "mos", None)
                        _score_str = f"MOS {_mos_v:.2f}" if _mos_v is not None else f"Score {score.composite_score:.3f}"
                        progress_callback(
                            _cpct,
                            f"Variante {_vi + 1}/{_vn}: '{variant.name}' → {_score_str} ✓",
                            proc_time,
                        )
                    except Exception as _exc:
                        logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

                logger.info("    → %s", score)

            except Exception as e:
                logger.error(
                    "  ❌ Variant '%s' fehlgeschlagen: %s",
                    variant.name,
                    e,
                    exc_info=True,
                )
                continue

        if len(results) == 0:
            raise RuntimeError("Alle Varianten sind fehlgeschlagen")

        # Sort by composite score (descending)
        results.sort(key=lambda x: x["score"].composite_score, reverse=True)  # type: ignore[union-attr]

        # Best result
        best = results[0]
        best_variant = best["variant"]
        best_score = best["score"]

        # Calculate confidence
        all_scores = [r["score"] for r in results]
        confidence = self.confidence_calc.calculate_confidence(all_scores, best_score)  # type: ignore[arg-type]
        best_score.confidence = confidence  # type: ignore[union-attr]

        logger.info(
            "✅ Best variant: '%s' (Wert=%.3f, confidence=%.2f)",
            best_variant.name,  # type: ignore[union-attr]
            best_score.composite_score,  # type: ignore[union-attr]
            confidence,
        )

        return {
            "audio": best["audio"],
            "best_audio": best["audio"],  # Alias für Rückwärtskompatibilität
            "success": True,
            "variant_name": best_variant.name,  # type: ignore[union-attr]
            "variant_strategy": best_variant.strategy,  # type: ignore[union-attr]
            "confidence": confidence,
            "composite_score": best_score.composite_score,  # type: ignore[union-attr]
            "best_score": best_score,
            "all_scores": all_scores,
            "processing_times": processing_times,
        }

    @staticmethod
    def _derive_restore_mode(config: ProcessingConfig) -> str:
        """Map variant-level ProcessingConfig to a coarse UV3 restore mode.

        Mapping ensures each standard variant gets a distinct UV3 mode so that
        Multi-Pass evaluation is actually comparing different processing depths:
          - denoise < 0.10                              → "fast"        (naturalness_first)
          - denoise ≤ 0.20                              → "balanced"    (conservative)
          - denoise ≥ 0.60 / enh ≥ 0.65 / comp ≥ 5.0  → "maximum"     (aggressive)
          - else                                        → "restoration" (balanced variant)
        """
        # Near-zero denoise: treat as minimal/fast path.
        if config.denoise_strength < 0.10:
            return "fast"

        # Heavy profiles: use MAXIMUM path.
        if config.denoise_strength >= 0.60 or config.enhancement_strength >= 0.65 or config.compression_ratio >= 5.0:
            return "maximum"

        # Light-conservative profiles: use BALANCED (less than full restoration phases).
        if config.denoise_strength <= 0.20:
            return "balanced"

        # Default profile: restoration quality path.
        return "restoration"

    def _default_process_func(
        self,
        audio: np.ndarray,
        sample_rate: int,
        config: ProcessingConfig,
        *,
        progress_callback=None,
    ) -> np.ndarray:
        """
        Default processing function using UnifiedRestorerV3.
        UnifiedRestorerV3 wird einmalig instanziiert und gecacht (ML-Modelle nur 1× laden).

        Args:
            audio: Input audio
            sample_rate: Sample rate
            config: ProcessingConfig
            progress_callback: Optional callable(pct, phase, elapsed_s) — forwarded to UV3.

        Returns:
            Processed audio (np.ndarray)
        """

        try:
            from backend.core.unified_restorer_v3 import UnifiedRestorerV3  # pylint: disable=import-outside-toplevel

            # Einmalig laden — alle Varianten nutzen dieselbe Instanz
            restorer = self._restorer
            if restorer is None:
                logger.info("Initialisiere UnifiedRestorerV3 (einmalig)...")
                restorer = UnifiedRestorerV3()
                self._restorer = restorer

            _restore_mode = self._derive_restore_mode(config)
            logger.debug(
                "[MPASS] wiederherstellen(Betriebsart=%s) auf voller Programmlänge (%.0fs) …",
                _restore_mode,
                len(audio) / sample_rate,
            )
            result = restorer.restore(  # type: ignore[union-attr,attr-defined]
                audio=audio,
                sample_rate=sample_rate,
                mode=_restore_mode,
                progress_callback=progress_callback,
            )

            # V3 gibt RestorationResult zurück — audio-Array extrahieren
            if hasattr(result, "audio") and result.audio is not None:
                processed_audio: np.ndarray = np.asarray(result.audio, dtype=np.float32)
                return processed_audio
            raise RuntimeError("UnifiedRestorerV3 returned no audio payload for variant evaluation")

        except Exception as e:
            logger.error("Default processing fehlgeschlagen: %s\n%s", e, _tb.format_exc())
            raise RuntimeError("MultiPass default processing failed") from e


def create_default_variants(
    base_mode: ProcessingMode = ProcessingMode.RESTORATION, num_variants: int = 3
) -> list[ProcessingVariant]:
    """
    Erstelle default Varianten für Multi-Pass Processing.

    Args:
        base_mode: Base ProcessingMode
        num_variants: Anzahl Varianten (3 or 5)

    Returns:
        Liste von ProcessingVarianten
    """
    if num_variants == 3:
        # Conservative, Balanced, Aggressive
        return [
            ProcessingVariant.create_conservative(base_mode),
            ProcessingVariant.create_balanced(base_mode),
            ProcessingVariant.create_aggressive(base_mode),
        ]
    elif num_variants == 5:
        # Full spectrum
        return [
            ProcessingVariant.create_conservative(base_mode),
            ProcessingVariant.create_gentle_denoise(base_mode),
            ProcessingVariant.create_balanced(base_mode),
            ProcessingVariant.create_aggressive(base_mode),
            ProcessingVariant.create_strong_dynamics(ProcessingMode.STUDIO_2026),
        ]
    else:
        raise ValueError(f"num_variants muss 3 oder 5 sein, nicht {num_variants}")


def _run_demo() -> None:
    """Lokaler Demo-Entrypoint für manuelle Multi-Pass-Validierung."""
    import soundfile as sf  # pylint: disable=import-outside-toplevel

    from backend.file_import import load_audio_file  # pylint: disable=import-outside-toplevel

    # Load test audio
    _res = load_audio_file("test_audio/test_input.wav")
    if _res is None:
        raise RuntimeError("test_audio/test_input.wav konnte nicht geladen werden")
    demo_audio, demo_sr = np.asarray(_res["audio"], dtype=np.float32), int(_res["sr"])

    # Create variants
    demo_variants = create_default_variants(base_mode=ProcessingMode.RESTORATION, num_variants=3)

    # Run Multi-Pass
    engine = MultiPassEngine()
    demo_result = engine.process_with_variants(audio=demo_audio, sample_rate=demo_sr, variants=demo_variants)

    # Save best result
    sf.write("test_output/multipass_best.wav", demo_result["audio"], demo_sr)

    logger.debug("\n✅ Best Variant: %s", demo_result["variant_name"])
    logger.debug("   Composite Wert: %.3f", demo_result["composite_score"])
    logger.debug("   Confidence: %.2f", demo_result["confidence"])
    logger.debug("\n📊 All Scores:")
    for variant_score in demo_result["all_scores"]:
        logger.debug("   %s", variant_score)


# === Example Usage ===
if __name__ == "__main__":
    _run_demo()
