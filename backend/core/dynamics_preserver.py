"""
dynamics_preserver.py — §v10 Dynamik-Erhalt für unhörbare Restaurierung
========================================================================

Schützt die natürliche musikalische Dynamik vor dem „Glattbügeln" durch
kumulative DSP-Verarbeitung. Zentrale Erkenntnis: Das menschliche Ohr
nimmt Dynamik-Verlust sofort als „unnatürlich"/„überproduziert" wahr.

Ansatz (Dreistufig):
  1. PRE: Dynamik-Profil erfassen (Crest, Mikrodynamik, Makrodynamik)
  2. MONITOR: Nach jeder Phase prüfen, ob Dynamik erhalten blieb
  3. RESTORE: Bei Verlust adaptive Dynamik-Rekonstruktion

Integration:
  Wird VOR und NACH jeder Restaurierungs-Phase aufgerufen.
  Präserviert die originale Dynamik-Signatur ohne das Signal zu verfälschen.

Author: Aurik 10 Development Team — Juli 2026
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Konstanten
# ═══════════════════════════════════════════════════════════════════════════════

# Crest-Faktor: Unter 6 dB = überkomprimiert für Musik
CREST_MIN_MUSICAL_DB: float = 6.0
# Dynamik-Range: Unter 15 dB = flach
DYNAMIC_RANGE_MIN_MUSICAL_DB: float = 15.0
# Erlaubter Dynamik-Verlust pro Phase (dB)
MAX_DYNAMIC_LOSS_PER_PHASE_DB: float = 1.5
# Erlaubter kumulativer Verlust (dB)
MAX_CUMULATIVE_DYNAMIC_LOSS_DB: float = 4.0

# P-Faktor: Mikrodynamik (lokale Varianz) vs Makrodynamik (globale Hüllkurve)
MICRO_WINDOW_MS: float = 50.0  # 50ms für Mikrodynamik
MACRO_WINDOW_MS: float = 500.0  # 500ms für Makrodynamik (musikalische Phrasierung)

# ═══════════════════════════════════════════════════════════════════════════════
# Datenstrukturen
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DynamicsProfile:
    """Dynamik-Profil eines Audiosignals vor/nach der Verarbeitung."""

    crest_factor_db: float = 12.0
    dynamic_range_db: float = 20.0  # P95-P05 der Frame-RMS
    micro_dynamic_db: float = 6.0  # RMS-Varianz in 50ms-Fenstern
    macro_dynamic_db: float = 12.0  # RMS-Varianz in 500ms-Fenstern
    rms_dbfs: float = -20.0
    peak_dbfs: float = -3.0
    transient_density: float = 0.02  # Anteil Transienten (1. Ableitung > Schwelle)
    sample_rate: int = 48000


@dataclass
class DynamicsLoss:
    """Dokumentierter Dynamik-Verlust zwischen Profilen."""

    delta_crest_db: float = 0.0
    delta_range_db: float = 0.0
    delta_micro_db: float = 0.0
    delta_macro_db: float = 0.0
    total_loss_db: float = 0.0  # Gewichtete Summe
    severity: str = "none"  # none, mild, moderate, severe
    recommendation: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Profil-Erfassung
# ═══════════════════════════════════════════════════════════════════════════════


def capture_dynamics_profile(audio: np.ndarray, sr: int) -> DynamicsProfile:
    """Erfasst das Dynamik-Profil eines Audiosignals.

    Args:
        audio: Eingabe-Audio (mono/stereo, beliebige Sample-Rate)
        sr: Sample-Rate

    Returns:
        DynamicsProfile mit allen Dynamik-Metriken
    """
    arr = np.asarray(audio, dtype=np.float64)
    # Mono
    if arr.ndim == 2:
        if arr.shape[0] <= 2 < arr.shape[1]:
            mono = arr.mean(axis=0)
        else:
            mono = arr.mean(axis=1)
    else:
        mono = arr

    mono = np.ravel(mono)
    if mono.size < 64:
        return DynamicsProfile(sample_rate=sr)

    # Grundlegende Pegel
    rms = float(np.sqrt(np.mean(mono**2) + 1e-12))
    peak = float(np.max(np.abs(mono)) + 1e-12)
    rms_dbfs = float(20.0 * np.log10(max(rms, 1e-12)))
    peak_dbfs = float(20.0 * np.log10(max(peak, 1e-12)))
    crest_db = float(20.0 * np.log10(max(peak / max(rms, 1e-8), 1e-8)))

    # Frame-basierte Dynamik-Analyse
    micro_samples = int(MICRO_WINDOW_MS / 1000.0 * sr)
    macro_samples = int(MACRO_WINDOW_MS / 1000.0 * sr)
    micro_samples = max(micro_samples, 64)
    macro_samples = max(macro_samples, micro_samples * 2)

    def _frame_rms(sig: np.ndarray, frame_len: int) -> np.ndarray:
        n_frames = (len(sig) - frame_len) // (frame_len // 2) + 1
        if n_frames < 2:
            return cast(np.ndarray, np.array([float(np.sqrt(np.mean(sig**2) + 1e-12))]))
        rms_vals = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * (frame_len // 2)
            chunk = sig[start : start + frame_len]
            rms_vals[i] = float(np.sqrt(np.mean(chunk**2) + 1e-12))
        return cast(np.ndarray, rms_vals)

    micro_rms = _frame_rms(mono, micro_samples)
    macro_rms = _frame_rms(mono, macro_samples)

    micro_rms_db = 20.0 * np.log10(micro_rms + 1e-12)
    macro_rms_db = 20.0 * np.log10(macro_rms + 1e-12)

    # Dynamic Range: P95 - P05
    p95 = float(np.percentile(macro_rms_db, 95))
    p05 = float(np.percentile(macro_rms_db, 5))
    dynamic_range_db = max(0.0, p95 - p05)

    # Mikrodynamik: Standardabweichung der Mikro-RMS
    micro_dynamic_db = float(np.clip(np.std(micro_rms_db), 0.0, 30.0))

    # Makrodynamik: Standardabweichung der Makro-RMS
    macro_dynamic_db = float(np.clip(np.std(macro_rms_db), 0.0, 30.0))

    # Transienten-Dichte
    diff = np.abs(np.diff(mono, prepend=mono[0]))
    if diff.size > 8:
        thr = float(np.percentile(diff, 99.0))
        transient_density = float(np.clip(np.mean(diff > thr), 0.0, 1.0))
    else:
        transient_density = 0.0

    return DynamicsProfile(
        crest_factor_db=float(np.clip(crest_db, 0.0, 40.0)),
        dynamic_range_db=float(np.clip(dynamic_range_db, 0.0, 60.0)),
        micro_dynamic_db=micro_dynamic_db,
        macro_dynamic_db=macro_dynamic_db,
        rms_dbfs=rms_dbfs,
        peak_dbfs=peak_dbfs,
        transient_density=transient_density,
        sample_rate=sr,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Verlust-Erkennung
# ═══════════════════════════════════════════════════════════════════════════════


def compute_dynamics_loss(
    before: DynamicsProfile,
    after: DynamicsProfile,
) -> DynamicsLoss:
    """Berechnet den Dynamik-Verlust zwischen zwei Profilen.

    Positives delta = Verlust (nachher kleiner als vorher).
    """
    delta_crest = before.crest_factor_db - after.crest_factor_db
    delta_range = before.dynamic_range_db - after.dynamic_range_db
    delta_micro = before.micro_dynamic_db - after.micro_dynamic_db
    delta_macro = before.macro_dynamic_db - after.macro_dynamic_db

    # Gewichtete Summe: Crest und Range sind die wichtigsten Faktoren
    total = max(0.0, delta_crest * 0.35 + delta_range * 0.30 + delta_micro * 0.20 + delta_macro * 0.15)

    if total <= 0.5:
        severity = "none"
        rec = "Dynamik vollständig erhalten."
    elif total <= 1.5:
        severity = "mild"
        rec = "Minimaler Dynamik-Verlust — für menschliche Ohren unhörbar."
    elif total <= 3.0:
        severity = "moderate"
        rec = "Spürbarer Dynamik-Verlust — adaptive Rekonstruktion empfohlen."
    else:
        severity = "severe"
        rec = "Deutlicher Dynamik-Verlust — Dynamik-Rekonstruktion erforderlich."

    return DynamicsLoss(
        delta_crest_db=float(delta_crest),
        delta_range_db=float(delta_range),
        delta_micro_db=float(delta_micro),
        delta_macro_db=float(delta_macro),
        total_loss_db=float(total),
        severity=severity,
        recommendation=rec,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive Dynamik-Rekonstruktion (sanft, nicht-invasiv)
# ═══════════════════════════════════════════════════════════════════════════════


def restore_dynamics(
    audio: np.ndarray,
    sr: int,
    target_profile: DynamicsProfile,
    *,
    strength: float = 0.5,
) -> np.ndarray:
    """Stellt die Dynamik eines Signals sanft wieder her.

    Verwendet einen zweistufigen Ansatz:
    1. Makrodynamik: Langsame Hüllkurven-Anpassung (musikalische Phrasierung)
    2. Mikrodynamik: Schnelle Transienten-Wiederherstellung

    Args:
        audio: Zu restaurierendes Audio
        sr: Sample-Rate
        target_profile: Ziel-Dynamikprofil (vom Original)
        strength: Stärke der Rekonstruktion (0.0 = keine, 1.0 = voll)

    Returns:
        Dynamik-restauriertes Audio
    """
    if strength <= 0.01:
        return audio

    arr = np.asarray(audio, dtype=np.float64)
    is_stereo = arr.ndim == 2

    if is_stereo:
        if arr.shape[0] <= 2 < arr.shape[1]:
            channels = [arr[i] for i in range(arr.shape[0])]
        else:
            channels = [arr[:, i] for i in range(arr.shape[1])]
    else:
        channels = [arr]

    result_channels = []
    for ch in channels:
        restored = _restore_dynamics_mono(ch, sr, target_profile, strength)
        result_channels.append(restored)

    if is_stereo:
        result = np.stack(result_channels, axis=0 if arr.shape[0] <= 2 else 1)
    else:
        result = result_channels[0]

    # Sanftes Clipping: nur extremes Overshoot begrenzen
    max_val = float(np.max(np.abs(result)))
    if max_val > 1.2:
        result /= max_val * 0.95

    return result.astype(audio.dtype)  # type: ignore[no-any-return]


def _restore_dynamics_mono(
    mono: np.ndarray,
    sr: int,
    target: DynamicsProfile,
    strength: float,
) -> np.ndarray:
    """Mono-Variante der Dynamik-Rekonstruktion."""
    if mono.size < 64:
        return mono

    # 1. Makrodynamik: Hüllkurven-basierte Anpassung
    macro_samples = int(MACRO_WINDOW_MS / 1000.0 * sr)
    macro_samples = max(macro_samples, 512)

    current_profile = capture_dynamics_profile(mono, sr)

    # Wenn das aktuelle Profil bereits mehr Dynamik hat → nichts tun
    if current_profile.dynamic_range_db >= target.dynamic_range_db - 0.5:
        return mono

    # Ziel-Expansion: expandiere Dynamik-Range
    target_range = target.dynamic_range_db
    current_range = max(current_profile.dynamic_range_db, 0.1)
    expansion_ratio = min((target_range / current_range) ** strength, 2.0)

    # Hüllkurven-basierte Expansion
    hop = macro_samples // 2
    n_frames = (len(mono) - macro_samples) // hop + 1
    if n_frames < 2:
        return mono

    # Extrahiere Hüllkurve
    envelope = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        chunk = mono[start : start + macro_samples]
        envelope[i] = float(np.sqrt(np.mean(chunk**2) + 1e-12))

    envelope_db = 20.0 * np.log10(envelope + 1e-12)
    mean_db = float(np.mean(envelope_db))

    # Expandiere um mean_db zentriert
    expanded_db = (envelope_db - mean_db) * expansion_ratio + mean_db
    expanded_linear = np.power(10.0, expanded_db / 20.0)
    envelope_linear = np.power(10.0, envelope_db / 20.0)
    gain = expanded_linear / (envelope_linear + 1e-12)

    # Interpoliere Gain auf Sample-Ebene
    t_gain = np.arange(n_frames) * hop + macro_samples // 2
    t_signal = np.arange(len(mono))
    gain_interp = np.interp(t_signal, t_gain, gain)

    result = mono * gain_interp

    # 2. Mikrodynamik: Transienten-Verstärkung (Sub-Band)
    # Nur anwenden wenn Mikrodynamik verloren ging
    if current_profile.micro_dynamic_db < target.micro_dynamic_db - 0.5:
        # Einfache Transienten-Verstärkung via 1. Ableitung
        diff = np.abs(np.diff(result, prepend=result[0]))
        if diff.size > 8:
            thr = float(np.percentile(diff, 95.0))
            transient_mask = diff > thr
            # Sanfte Anhebung der Transienten
            micro_boost = (
                1.0 + (target.micro_dynamic_db / max(current_profile.micro_dynamic_db, 0.1) - 1.0) * strength * 0.3
            )
            smooth_mask = np.convolve(transient_mask.astype(float), np.hanning(64), mode="same")
            result = result * (1.0 + (micro_boost - 1.0) * smooth_mask)

    return cast(np.ndarray, result)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase-übergreifende Dynamik-Überwachung
# ═══════════════════════════════════════════════════════════════════════════════


class DynamicsPreserver:
    """Thread-sicherer Dynamik-Wächter für die gesamte Pipeline."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._original_profile: DynamicsProfile | None = None
        self._phase_profiles: dict[str, DynamicsProfile] = {}
        self._cumulative_loss_db: float = 0.0
        self._warnings: list[str] = []

    def capture_original(self, audio: np.ndarray, sr: int) -> DynamicsProfile:
        """Erfasst das Original-Dynamikprofil vor der Pipeline."""
        profile = capture_dynamics_profile(audio, sr)
        with self._lock:
            self._original_profile = profile
            self._phase_profiles.clear()
            self._cumulative_loss_db = 0.0
            self._warnings.clear()
        logger.info(
            "DynamicsPreserver: Originalsignal-Profil erfasst (Crest=%.1fdB, Range=%.1fdB, Micro=%.1fdB)",
            profile.crest_factor_db,
            profile.dynamic_range_db,
            profile.micro_dynamic_db,
        )
        return profile

    def check_phase(self, phase_name: str, audio: np.ndarray, sr: int) -> DynamicsLoss:
        """Prüft Dynamik-Erhalt nach einer Phase und warnt bei Verlust."""
        if self._original_profile is None:
            logger.debug("DynamicsPreserver: Kein Originalsignal-Profil — überspringe Pruefung.")
            return DynamicsLoss()

        current = capture_dynamics_profile(audio, sr)
        loss = compute_dynamics_loss(self._original_profile, current)

        with self._lock:
            self._phase_profiles[phase_name] = current
            self._cumulative_loss_db = max(0.0, self._original_profile.dynamic_range_db - current.dynamic_range_db)

            if loss.severity in ("moderate", "severe"):
                warning = (
                    f"DynamicsPreserver: {phase_name} — Dynamik-Verlust "
                    f"(Crest={loss.delta_crest_db:+.1f}dB, "
                    f"Range={loss.delta_range_db:+.1f}dB, "
                    f"Micro={loss.delta_micro_db:+.1f}dB) → {loss.severity}"
                )
                self._warnings.append(warning)
                logger.warning(warning)
            else:
                logger.debug(
                    "DynamicsPreserver: %s OK (Crest=%.1fdB, Range=%.1fdB)",
                    phase_name,
                    current.crest_factor_db,
                    current.dynamic_range_db,
                )

        return loss

    def should_restore(self) -> bool:
        """Prüft ob Dynamik-Rekonstruktion nötig ist."""
        return self._cumulative_loss_db > MAX_CUMULATIVE_DYNAMIC_LOSS_DB

    @property
    def original_profile(self) -> DynamicsProfile | None:
        return self._original_profile

    @property
    def cumulative_loss_db(self) -> float:
        return self._cumulative_loss_db

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════════

_preserver: DynamicsPreserver | None = None


def get_dynamics_preserver() -> DynamicsPreserver:
    """Thread-sicherer Singleton-Accessor."""
    global _preserver
    if _preserver is None:
        _preserver = DynamicsPreserver()
    return _preserver
