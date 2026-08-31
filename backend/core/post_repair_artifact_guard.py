#!/usr/bin/env python3
"""
§v10.610: Post-Repair Artifact Guard — Pumping- und Verzerrungsschutz für die SOTA-Kette.

Problem: Die SOTA-Pipelines (Denoise, Vocal, Repair, Inpainting) haben keine
eigenen Artefakt-Guards. Wenn der DiT zu aggressiv rekonstruiert oder der
Denoiser zu stark subtrahiert, entstehen Pumping/Verzerrung — und die globalen
Guards in unified_restorer_v3 laufen erst NACH der ganzen Kette.

Lösung: Der PostRepairArtifactGuard läuft NACH JEDEM Repair-Schritt:
  1. Formant-Drift-Check (vocal_overprocessing_detector) — erkennt
     Verzerrung durch zu aggressives Processing
  2. TruePeak-Check — erkennt Clipping/Übersteuerung
  3. Pumping-Check (Gain-Modulation im Zeitbereich) — erkennt Atmung
  4. Bei Verstoß: automatische Strength-Reduktion (Blend Richtung Original)

Integration: In CoordinatedRepair.execute() nach jedem Schritt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, cast

import numpy as np

log = logging.getLogger(__name__)

SR = 48000

# Schwellwerte
TRUEPEAK_LIMIT_DBFS = 0.0  # 0 dBFS — darüber = Clipping
TRUEPEAK_WARN_DBFS = -0.5  # Warnschwelle
PUMPING_GAIN_MODULATION_MAX = 0.15  # max. 15% Gain-Modulation pro 100ms
FORMANT_DRIFT_MAX = 0.08  # max. 8% Formant-Drift
SPECTRAL_NOISE_RISE_MAX_DB = 1.5  # §v10.850: max. +1.5 dB Rauschfloor-Anstieg (8-20 kHz)


@dataclass
class GuardResult:
    """Ergebnis einer Artefakt-Prüfung."""

    passed: bool
    truepeak_dbfs: float
    pumping_index: float  # 0-1, 0 = kein Pumpen
    formant_drift: float  # 0-1, 0 = keine Drift
    spectral_noise_rise_db: float = 0.0  # §v10.850: Anstieg des Rauschfloors
    violations: list[str] = field(default_factory=list)
    blended_back: bool = False  # Wurde Strength automatisch reduziert?


class PostRepairArtifactGuard:
    """
    Prüft nach jedem Repair-Schritt auf Pumping und Verzerrung.

    Nutzung:
        guard = PostRepairArtifactGuard()
        result = guard.check(audio_pre, audio_post, sr, phase_id)
        if not result.passed:
            audio_post = guard.blend_back(audio_pre, audio_post, 0.7)
    """

    def __init__(self):
        self._overprocessing = None
        self._init_detectors()

    def _init_detectors(self):
        try:
            from backend.core.vocal_overprocessing_detector import VocalOverprocessingDetector

            self._overprocessing = VocalOverprocessingDetector()
            log.debug("Artifact Guard: VocalOverprocessingDetector geladen")
        except Exception as exc:
            log.debug("Artifact Guard: Overprocessing-Detektor nicht verfügbar (%s)", exc)

    def check(
        self,
        audio_pre: np.ndarray,
        audio_post: np.ndarray,
        sr: int = SR,
        phase_id: str = "unknown",
    ) -> GuardResult:
        """
        Führt alle drei Artefakt-Checks durch.

        Returns:
            GuardResult mit passed-Flag und Metriken.
        """
        violations: list[str] = []

        # ── Check 1: TruePeak (Clipping) — RELATIV, nicht absolut ──
        # §v10.860: Vorher wurde jede Datei geflaggt, die bereits mit
        # -0.4 dBFS gemastert war (Corpus!). Jetzt: nur flaggen, wenn der
        # Schritt den Peak um >0.1 dB ERHÖHT hat.
        truepeak = float(np.abs(audio_post).max())
        truepeak_dbfs = float(20 * np.log10(truepeak + 1e-10))
        pre_peak = float(np.abs(audio_pre).max())
        pre_peak_dbfs = float(20 * np.log10(pre_peak + 1e-10))
        peak_delta = truepeak_dbfs - pre_peak_dbfs
        if truepeak_dbfs > TRUEPEAK_LIMIT_DBFS:
            violations.append(f"truepeak_overflow_{truepeak_dbfs:+.1f}dBFS")
        elif peak_delta > 0.1:
            violations.append(f"truepeak_rise_{peak_delta:+.2f}dB")

        # ── Check 2: Pumping (Gain-Modulation) ──
        pumping_index = self._measure_pumping(audio_pre, audio_post, sr)
        if pumping_index > PUMPING_GAIN_MODULATION_MAX:
            violations.append(f"pumping_{pumping_index:.2f}")

        # ── Check 3: Formant-Drift (Verzerrung) ──
        formant_drift = 0.0
        if self._overprocessing is not None:
            try:
                result = self._overprocessing.check_formant_drift(
                    vocals_pre=audio_pre,
                    vocals_post=audio_post,
                    sr=sr,
                    phase_id=phase_id,
                )
                if result is not None:
                    drift = getattr(result, "drift", None) or getattr(result, "score", None)
                    if drift is not None:
                        formant_drift = float(drift)
            except Exception:
                pass

        if formant_drift > FORMANT_DRIFT_MAX:
            violations.append(f"formant_drift_{formant_drift:.2f}")

        # ── Check 4: Spektraler Rauschfloor (8–20 kHz) ──
        # §v10.850: Banquet & Co. können spektralen Schaden anrichten, ohne
        # die Signalenergie zu ändern. Misst den Hochfrequenz-Rauschfloor.
        spectral_rise = self._measure_spectral_noise_rise(audio_pre, audio_post, sr)
        if spectral_rise > SPECTRAL_NOISE_RISE_MAX_DB:
            violations.append(f"spectral_noise_rise_{spectral_rise:+.1f}dB")

        passed = len(violations) == 0

        return GuardResult(
            passed=passed,
            truepeak_dbfs=truepeak_dbfs,
            pumping_index=pumping_index,
            formant_drift=formant_drift,
            spectral_noise_rise_db=spectral_rise,
            violations=violations,
        )

    def _measure_pumping(self, pre: np.ndarray, post: np.ndarray, sr: int) -> float:
        """
        Misst Gain-Modulation: wie stark schwankt die Verstärkung im Zeitverlauf?
        Pumping = periodisches An-/Abschwellen der Lautstärke.
        """
        if len(pre) == 0 or len(post) == 0:
            return 0.0

        # Frame-Energien (100 ms Fenster)
        frame_len = sr // 10  # 100 ms
        n_frames = max(1, min(len(pre), len(post)) // frame_len)

        pre_env = np.zeros(n_frames, dtype=np.float64)
        post_env = np.zeros(n_frames, dtype=np.float64)

        for i in range(n_frames):
            s = i * frame_len
            e = s + frame_len
            pre_env[i] = np.sqrt(np.mean(pre[s:e] ** 2) + 1e-10)
            post_env[i] = np.sqrt(np.mean(post[s:e] ** 2) + 1e-10)

        # Gain = post/pre pro Frame
        gain = post_env / (pre_env + 1e-10)

        # Pumping-Index = Variationskoeffizient des Gains
        if gain.std() > 0:
            pumping = float(gain.std() / (gain.mean() + 1e-10))
        else:
            pumping = 0.0

        return min(pumping, 1.0)

    def _measure_spectral_noise_rise(self, pre: np.ndarray, post: np.ndarray, sr: int) -> float:
        """Misst den Anstieg des Hochfrequenz-Rauschfloors (8–20 kHz) in dB.

        §v10.850: Banquet & Co. können spektralen Schaden anrichten, ohne die
        Gesamtenergie zu ändern. Vergleich: 20. Perzentil des HF-Spektrums
        (Rauschfloor, nicht Signalpeaks) vor/nach dem Schritt.
        """
        pre = np.asarray(pre, dtype=np.float64)
        post = np.asarray(post, dtype=np.float64)
        if len(pre) < 1024 or len(post) < 1024:
            return 0.0

        def hf_noise_floor(x: np.ndarray) -> float:
            n_fft = min(2048, len(x))
            window = np.hanning(n_fft)
            n_frames = max(1, len(x) // (n_fft // 2))
            spec = np.zeros((n_frames, n_fft // 2 + 1))
            for i in range(n_frames):
                s = i * (n_fft // 2)
                if s + n_fft > len(x):
                    continue  # letzter unvollständiger Frame
                frame = x[s : s + n_fft] * window
                spec[i] = np.abs(np.fft.rfft(frame)) + 1e-12
            freqs = np.fft.rfftfreq(n_fft, 1 / sr)
            hf_mask = freqs >= 8000
            hf = spec[:, hf_mask]
            # 20. Perzentil = Rauschfloor (robust gegen Signal-Peaks)
            return float(np.percentile(hf, 20))

        floor_pre = hf_noise_floor(pre)
        floor_post = hf_noise_floor(post)
        if floor_pre <= 0:
            return 0.0
        return float(20 * np.log10(floor_post / floor_pre))

    def blend_back(
        self,
        audio_pre: np.ndarray,
        audio_post: np.ndarray,
        blend_ratio: float = 0.7,
    ) -> np.ndarray:
        """
        Reduziert die Strength automatisch: blend_ratio Anteil Original,
        (1 - blend_ratio) Anteil prozessiert.
        """
        return cast(np.ndarray, (blend_ratio * audio_pre + (1 - blend_ratio) * audio_post).astype(np.float32))

    def normalize_truepeak(self, audio: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
        """Begrenzt TruePeak auf target_dbfs."""
        peak = float(np.abs(audio).max())
        if peak <= 0:
            return audio
        target = 10 ** (target_dbfs / 20)
        if peak > target:
            return cast(np.ndarray, (audio * (target / peak)).astype(np.float32))
        return audio


# ═════════════════════════════════════════════════════════════════════════════
# Integration in Coordinated Repair
# ═════════════════════════════════════════════════════════════════════════════


def run_post_repair_guard(
    audio_pre: np.ndarray,
    audio_post: np.ndarray,
    sr: int = SR,
    phase_id: str = "unknown",
) -> tuple[np.ndarray, GuardResult]:
    """
    Convenience-Funktion: Prüft und korrigiert automatisch.

    Returns:
        (korrigiertes_audio, GuardResult)
    """
    guard = PostRepairArtifactGuard()
    result = guard.check(audio_pre, audio_post, sr, phase_id)

    if not result.passed:
        # Automatische Korrektur
        corrected = guard.blend_back(audio_pre, audio_post, 0.7)
        corrected = guard.normalize_truepeak(corrected)
        result.blended_back = True
        log.warning(
            "Artifact Guard %s: Verstöße %s → Strength reduziert (Blend 70/30)",
            phase_id,
            result.violations,
        )
        return corrected, result

    return audio_post, result
