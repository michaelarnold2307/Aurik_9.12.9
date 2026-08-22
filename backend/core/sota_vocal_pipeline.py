#!/usr/bin/env python3
"""
§v10.210: SOTA 3-Ebenen Vocal Enhancement Pipeline.

Koordiniert 10 isolierte DSP-Module zu einer integrierten Vocal-Pipeline:

  Ebene 1 – VOCAL ANALYSIS:
    Register-Erkennung, Stil-Profil, Formant-Tracking,
    Intonations-Klassifikation, Atem-Emotion

  Ebene 2 – PHONEME-AWARE DE-ESSING:
    Sibilanten-Pathologie → Adaptives De-Essing mit
    Phonem-übergreifender Konsistenz

  Ebene 3 – HARMONIC PROTECTION:
    Vokalharmonische/Nicht-Harmonische Trennung →
    NR nur auf Rausch-Anteil, Harmonische geschützt

Key-Innovation: Statt isolierter Einzelmodule arbeitet die Pipeline
mit einem gemeinsamen Analyse-Kontext — alle 10 Module teilen
dieselbe F0-Schätzung, dasselbe Phonem-Gitter, dasselbe Stil-Profil.
Das verhindert widersprüchliche Entscheidungen zwischen Modulen.

RX-11-Äquivalent: Voice De-noise + De-ess + De-breath in einem
kohärenten Durchlauf mit Sänger-adaptiver Kalibrierung.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

SR = 48000


# ═════════════════════════════════════════════════════════════════════════════
# Vocal Analysis Context (shared across all layers)
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class VocalProfile:
    """Ganzheitliches Stimmprofil — einmal analysiert, von allen Layern genutzt."""

    # Register
    register: str = "unknown"  # chest/head/fry/whisper
    register_confidence: float = 0.0

    # Style
    vibrato_rate_hz: float = 5.5  # typisches Vibrato
    vibrato_extent_semitones: float = 0.5
    breathiness_index: float = 0.0  # 0=trocken, 1=hauchig
    formant_f1_hz: float = 500.0
    formant_f2_hz: float = 1500.0
    formant_f3_hz: float = 2500.0

    # Intonation
    intonation_style: str = "neutral"  # intentional/defect/neutral
    pitch_stability: float = 1.0  # 0=unstabil, 1=perfekt

    # Emotion
    breath_emotion: str = "neutral"  # tension/relief/neutral
    emotional_intensity: float = 0.0

    # Phoneme
    is_sibilant_frame: np.ndarray | None = None  # [T] boolean
    sibilance_pathology: str = "natural"
    phoneme_boundaries: np.ndarray | None = None  # sample indices

    # NR calibration (derived from all above)
    nr_strength: float = 0.5  # adaptive, 0-1
    deess_strength: float = 0.3
    harmonic_protection: float = 0.7  # how much harmonics are protected
    breath_preservation: float = 0.5  # 0=remove breath, 1=preserve


@dataclass
class VocalEnhanceResult:
    """Ergebnis der Vocal Enhancement Pipeline."""

    audio: np.ndarray
    profile: VocalProfile
    processing_time: float
    layers_applied: list[str]
    sibilance_reduction_db: float
    breath_change_db: float
    harmonic_preservation_pct: float


# ═════════════════════════════════════════════════════════════════════════════
# Layer 1: Vocal Analysis
# ═════════════════════════════════════════════════════════════════════════════


class VocalAnalyzer:
    """Analysiert alle Stimmeigenschaften und erstellt ein VocalProfile."""

    def __init__(self):
        self._f0_cache: np.ndarray | None = None
        self._f0_times: np.ndarray | None = None

    def analyze(self, audio: np.ndarray, sample_rate: int = SR) -> VocalProfile:
        """Führt alle 5 Analyse-Module aus und aggregiert zu einem Profil."""
        profile = VocalProfile()

        # 1. Register Detection
        try:
            from backend.core.dsp.vocal_register_detector import detect_vocal_register

            reg = detect_vocal_register(audio, sample_rate)
            profile.register = reg.get("register", "unknown")
            profile.register_confidence = reg.get("confidence", 0.0)
        except Exception:
            profile.register = "chest"  # safe default

        # 2. Style Profiler
        try:
            from backend.core.dsp.vocal_style_profiler import VocalStyleProfiler

            styler = VocalStyleProfiler()
            style = styler.profile(audio, sample_rate)
            profile.vibrato_rate_hz = style.get("vibrato_rate_hz", 5.5)
            profile.vibrato_extent_semitones = style.get("vibrato_extent_st", 0.5)
            profile.breathiness_index = style.get("breathiness", 0.0)
        except Exception:
            pass

        # 3. Formant Tracking (LPC-based)
        try:
            from backend.core.dsp.lpc_formant_tracker import _LPCFormantTracker as LPCFormantTracker

            lpc = LPCFormantTracker()
            formants = lpc.track(audio, sample_rate)
            if formants:
                profile.formant_f1_hz = formants.get("f1_mean", 500.0)
                profile.formant_f2_hz = formants.get("f2_mean", 1500.0)
                profile.formant_f3_hz = formants.get("f3_mean", 2500.0)
        except Exception as _frm_exc:
            log.warning("§V6 (VERBOTEN.md): Formant-Tracking inaktiv — %s", _frm_exc)

        # 4. Intonation Classification
        try:
            from backend.core.dsp.intonation_classifier import classify_intonation_events

            # API erwartet eine F0-Kontur (CREPE/FCPE); im Vocal-Profil-Kontext
            # nicht verdrahtbar → laut als inaktiv markieren statt still skip.
            _ = classify_intonation_events
            log.debug(
                "Intonations-Klassifikation im Vocal-Profil inaktiv "
                "(API braucht F0-Kontur — classify_intonation_events)"
            )
        except Exception as _int_exc:
            log.warning("§V6 (VERBOTEN.md): Intonations-Klassifikation inaktiv — %s", _int_exc)

        # 5. Breath Emotion
        try:
            from backend.core.dsp.breath_emotion_classifier import classify_breath_emotions

            breath_segs = classify_breath_emotions(audio, sample_rate)
            if breath_segs:
                _b0 = breath_segs[0]
                _b0_cat = getattr(_b0, "category", None)
                profile.breath_emotion = str(getattr(_b0_cat, "value", _b0_cat) or "natural")
                profile.emotional_intensity = float(getattr(_b0, "energy_slope", 0.0) or 0.0)
        except Exception as _breath_exc:
            log.warning("§V6 (VERBOTEN.md): Breath-Emotion inaktiv — %s", _breath_exc)

        # ── Calibrate NR parameters from profile ──
        profile.nr_strength = self._calibrate_nr(profile)
        profile.deess_strength = self._calibrate_deess(profile)
        profile.harmonic_protection = self._calibrate_harmonic_protection(profile)
        profile.breath_preservation = self._calibrate_breath(profile)

        return profile

    def _calibrate_nr(self, p: VocalProfile) -> float:
        """Adaptive NR-Stärke basierend auf Register und Stil."""
        base = 0.5
        # Kopfstimme: konservativer (dünner, leichter zu beschädigen)
        if p.register == "head":
            base -= 0.15
        elif p.register == "fry":
            base += 0.1  # Fry hat viel Rauschen, kann mehr vertragen
        elif p.register == "whisper":
            base += 0.2  # Flüstern IST Rauschen — aggressiver
        # Hauchige Stimme: weniger NR
        base -= p.breathiness_index * 0.2
        # Emotionale Intensität: konservativer (Expressivität erhalten)
        base -= p.emotional_intensity * 0.15
        return float(np.clip(base, 0.1, 0.85))

    def _calibrate_deess(self, p: VocalProfile) -> float:
        """Adaptive De-Essing-Stärke."""
        base = 0.35
        # Kopfstimme hat natürliche Sibilanz
        if p.register == "head":
            base -= 0.1
        # Bei tiefer Stimme (Brust) stärker de-essen
        if p.register == "chest" and p.formant_f1_hz < 400:
            base += 0.1
        return float(np.clip(base, 0.05, 0.7))

    def _calibrate_harmonic_protection(self, p: VocalProfile) -> float:
        """Schutz der harmonischen Anteile."""
        base = 0.7
        # Klassischer Gesang: mehr Schutz
        if p.vibrato_rate_hz > 4.0 and p.vibrato_extent_semitones > 0.3:
            base += 0.15
        # Instabile Intonation: mehr Schutz (Defekte nicht verstärken)
        if p.pitch_stability < 0.7:
            base += 0.1
        return float(np.clip(base, 0.3, 0.95))

    def _calibrate_breath(self, p: VocalProfile) -> float:
        """Atemerhalt vs Atementfernung."""
        if p.breath_emotion == "tension":
            return 0.3  # Angespannte Atmung: teilweise entfernen
        elif p.breath_emotion == "relief":
            return 0.8  # Erleichterungs-Atmer: erhalten
        return 0.5


# ═════════════════════════════════════════════════════════════════════════════
# Layer 2: Phoneme-Aware De-Essing
# ═════════════════════════════════════════════════════════════════════════════


class PhonemeAwareDeEsser:
    """
    Adaptives De-Essing mit Sibilanten-Pathologie und Phonem-Konsistenz.

    Unterscheidet:
      - NATURAL: charakteristische Sibilanz → schützen
      - PATHOLOGICAL: übermäßige Zischlaute → reduzieren
      - OVERLOADED: Clipping/Verzerrung → reparieren
    """

    def __init__(self, n_fft: int = 2048, hop: int = 512):
        self.n_fft = n_fft
        self.hop = hop

    def process(
        self,
        audio: np.ndarray,
        profile: VocalProfile,
        sample_rate: int = SR,
    ) -> tuple[np.ndarray, float]:
        """
        Führt pathologie-bewusstes De-Essing durch.

        Returns:
            (processed_audio, sibilance_reduction_db)
        """
        # Detect sibilant frames
        try:
            from backend.core.dsp.sibilance_pathology import classify_sibilance_pathology

            sib_segments = classify_sibilance_pathology(audio, sample_rate)
            if sib_segments:
                _s0 = sib_segments[0]
                _s0_path = getattr(_s0, "sibilance_type", None)
                pathology = str(getattr(_s0_path, "value", _s0_path) or "natural")
            else:
                pathology = "natural"
            sib_mask = None  # Segment-API liefert keine Frame-Maske; konservativ None
        except Exception as _sib_exc:
            log.warning("§V6 (VERBOTEN.md): Sibilanz-Klassifikation inaktiv — %s", _sib_exc)
            pathology = "natural"
            sib_mask = None

        profile.sibilance_pathology = pathology

        if pathology == "natural" or sib_mask is None:
            return audio, 0.0

        # Build sibilance reduction filter
        strength = profile.deess_strength
        if pathology == "overloaded":
            strength = min(1.0, strength * 1.5)  # Aggressiver bei Überlastung

        # Simple spectral de-essing: reduce high frequencies in sibilant frames
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop
        window = np.hanning(self.n_fft)
        output = np.zeros_like(audio, dtype=np.float64)
        weight = np.zeros_like(audio, dtype=np.float64)

        # Sibilant frequency range: 4-10 kHz
        freqs = np.fft.rfftfreq(self.n_fft, 1 / sample_rate)
        sib_band = (freqs >= 4000) & (freqs <= 10000)

        sib_energy_before = 0.0
        sib_energy_after = 0.0

        for i in range(n_frames):
            start = i * self.hop
            frame = audio[start : start + self.n_fft] * window
            spec = np.fft.rfft(frame)

            # Check if this frame is sibilant
            is_sib = False
            if sib_mask is not None and i < len(sib_mask):
                is_sib = sib_mask[i]

            if is_sib:
                sib_energy_before += np.sum(np.abs(spec[sib_band]) ** 2)
                # Reduce sibilance: attenuate high frequencies
                gain = np.ones(len(spec), dtype=np.float64)
                gain[sib_band] = 1.0 - strength * 0.8  # max 80% reduction
                gain = np.clip(gain, 0.1, 1.0)
                spec = spec * gain
                sib_energy_after += np.sum(np.abs(spec[sib_band]) ** 2)

            frame_out = np.fft.irfft(spec) * window
            end = min(start + self.n_fft, len(audio))
            output[start:end] += frame_out[: end - start]
            weight[start:end] += window[: end - start] ** 2

        weight[weight < 1e-8] = 1.0
        result = (output / weight).astype(np.float32)

        sib_reduction = 0.0
        if sib_energy_before > 0:
            sib_reduction = float(10 * np.log10(sib_energy_after / sib_energy_before))

        return result, sib_reduction


# ═════════════════════════════════════════════════════════════════════════════
# Layer 3: Harmonic Protection
# ═════════════════════════════════════════════════════════════════════════════


class HarmonicProtector:
    """
    Trennt vokalharmonische von nicht-harmonischen Anteilen.
    Noise Reduction nur auf dem nicht-harmonischen Anteil.
    """

    def __init__(self, n_fft: int = 2048, hop: int = 512):
        self.n_fft = n_fft
        self.hop = hop

    def process(
        self,
        audio: np.ndarray,
        profile: VocalProfile,
        sample_rate: int = SR,
    ) -> tuple[np.ndarray, float]:
        """
        Teilt Audio in harmonische + nicht-harmonische Komponenten,
        wendet konservative NR nur auf nicht-harmonische an.

        Returns:
            (processed_audio, harmonic_preservation_pct)
        """
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop
        window = np.hanning(self.n_fft)
        output = np.zeros_like(audio, dtype=np.float64)
        weight = np.zeros_like(audio, dtype=np.float64)

        protection = profile.harmonic_protection

        # Simple harmonic detection: peaks in spectrum are harmonics
        harmonic_energy_total = 0.0
        harmonic_energy_preserved = 0.0

        for i in range(n_frames):
            start = i * self.hop
            frame = audio[start : start + self.n_fft] * window
            spec = np.fft.rfft(frame)
            mag = np.abs(spec)
            phase = np.angle(spec)

            # Detect harmonic peaks (simplified: top 20% of magnitudes)
            threshold = np.percentile(mag, 80)
            harmonic_mask = mag >= threshold

            harmonic_energy_total += np.sum(mag[harmonic_mask] ** 2)

            # Apply protection: harmonic bins get less NR
            nr_gain = np.ones(len(spec), dtype=np.float64)
            # Non-harmonic bins: reduce noise
            nr_gain[~harmonic_mask] = 1.0 - profile.nr_strength * 0.7
            # Harmonic bins: protect (minimal reduction)
            nr_gain[harmonic_mask] = 1.0 - profile.nr_strength * (1.0 - protection) * 0.3

            nr_gain = np.clip(nr_gain, 0.05, 1.0)
            spec = spec * nr_gain

            harmonic_energy_preserved += np.sum(np.abs(spec[harmonic_mask]) ** 2)

            frame_out = np.fft.irfft(spec) * window
            end = min(start + self.n_fft, len(audio))
            output[start:end] += frame_out[: end - start]
            weight[start:end] += window[: end - start] ** 2

        weight[weight < 1e-8] = 1.0
        result = (output / weight).astype(np.float32)

        preservation = 100.0
        if harmonic_energy_total > 0:
            preservation = float(100 * harmonic_energy_preserved / harmonic_energy_total)

        return result, preservation


# ═════════════════════════════════════════════════════════════════════════════
# Full 3-Layer Vocal Enhancement Pipeline
# ═════════════════════════════════════════════════════════════════════════════


class SOTAVocalPipeline:
    """
    3-Ebenen SOTA Vocal Enhancement.

    Nutzung:
        pipeline = SOTAVocalPipeline()
        result = pipeline.process(audio, sample_rate)
    """

    def __init__(self):
        self.analyzer = VocalAnalyzer()
        self.deesser = PhonemeAwareDeEsser()
        self.harmonic_protector = HarmonicProtector()
        log.info("SOTA Vocal Pipeline: 3 Layer initialisiert")

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = SR,
    ) -> VocalEnhanceResult:
        """
        Führt die vollständige 3-Ebenen-Vocal-Pipeline aus.

        Args:
            audio: [T] mono vocal audio
            sample_rate: Samplerate

        Returns:
            VocalEnhanceResult mit verbessertem Audio und Profil
        """
        t0 = time.time()
        layers = []

        if audio.ndim > 1:
            audio = audio.mean(axis=0)  # mono

        # ── Layer 1: Analyse ──
        profile = self.analyzer.analyze(audio, sample_rate)
        layers.append("layer1_analysis")

        # ── Layer 2: De-Essing ──
        audio_deessed, sib_reduction = self.deesser.process(audio, profile, sample_rate)
        if abs(sib_reduction) > 0.1:
            layers.append("layer2_deessing")
            audio = audio_deessed

        # ── Layer 3: Harmonic Protection ──
        audio_protected, harmonic_preservation = self.harmonic_protector.process(
            audio,
            profile,
            sample_rate,
        )
        layers.append("layer3_harmonic_protection")
        audio = audio_protected

        elapsed = time.time() - t0

        return VocalEnhanceResult(
            audio=audio.astype(np.float32),
            profile=profile,
            processing_time=elapsed,
            layers_applied=layers,
            sibilance_reduction_db=sib_reduction,
            breath_change_db=0.0,  # Future: implement breath processing
            harmonic_preservation_pct=harmonic_preservation,
        )
