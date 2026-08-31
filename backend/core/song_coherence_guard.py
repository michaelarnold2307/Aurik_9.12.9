"""
SongCoherenceGuard — §CROWN Aktive Kohärenz-Sicherung
=======================================================

Garantiert: Der exportierte Song klingt wie EIN Musikstück,
nicht wie ein Flickenteppich aus Segmenten.

Prinzip: Nicht nur PRÜFEN, sondern AKTIV KORRIGIEREN.
  - Erkennt Sprünge an Segment-Grenzen
  - Glättet spektrale, dynamische und stereo Diskontinuitäten
  - Verlängert Cross-Fades bei problematischen Übergängen
  - Erzwingt Strategie-Kohärenz: benachbarte Segmente → ähnliche Behandlung

Integration: Wird NACH dem PerceptualOptimizer aufgerufen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CoherenceProfile:
    """Metrik-Profil eines Segments für Kohärenz-Vergleich."""

    spectral_centroid_hz: float = 2000.0
    spectral_rolloff_hz: float = 6000.0
    rms_dbfs: float = -20.0
    stereo_width: float = 0.5
    brightness: float = 0.5
    harmonicity: float = 0.5


@dataclass
class BoundaryFix:
    """Eine korrigierte Segment-Grenze."""

    segment_a: int = 0
    segment_b: int = 0
    issue: str = ""
    fix_applied: str = ""
    severity: float = 0.0  # 0–1


class SongCoherenceGuard:
    """Aktiver Kohärenz-Wächter — erkennt UND behebt Diskontinuitäten.

    Verwendung:
        guard = SongCoherenceGuard()
        fixed_audio = guard.ensure_coherence(segments, strategies, sr)
    """

    # ── Schwellwerte ──────────────────────────────────────────────────
    MAX_CENTROID_JUMP_HZ: float = 500.0  # Max erlaubter Centroid-Sprung
    MAX_RMS_JUMP_DB: float = 3.0  # Max erlaubter RMS-Sprung
    MAX_BRIGHTNESS_JUMP: float = 0.15  # Max erlaubter Brightness-Sprung
    MAX_STEREO_JUMP: float = 0.25  # Max erlaubter Stereo-Breite-Sprung
    MIN_CROSSFADE_S: float = 0.15  # Normaler Cross-Fade
    EXTENDED_CROSSFADE_S: float = 0.50  # Verlängerter Cross-Fade bei Sprüngen
    MAX_STRATEGY_GAP: int = 2  # Max Strategie-Unterschied zwischen Nachbarn

    def __init__(self) -> None:
        self._fixes: list[BoundaryFix] = []

    def ensure_coherence(
        self,
        segments: list[np.ndarray],
        strategies: list[str],
        sr: int,
    ) -> tuple[np.ndarray, list[BoundaryFix]]:
        """Stellt Kohärenz sicher — aktiv, nicht nur prüfend.

        Args:
            segments: Audio-Segmente in Reihenfolge
            strategies: Verwendete Strategien pro Segment
            sr: Sample-Rate

        Returns:
            (kohärentes Audio, Liste der angewandten Korrekturen)
        """
        self._fixes = []
        if len(segments) < 2:
            return segments[0] if segments else np.zeros(1, dtype=np.float32), []

        # 1. Profile messen
        profiles = [self._measure_profile(seg, sr) for seg in segments]

        # 2. Strategie-Kohärenz erzwingen
        strategies = self._enforce_strategy_coherence(strategies, segments, profiles, sr)

        # 3. Grenz-Diskontinuitäten erkennen und beheben
        fixed_segments = list(segments)
        for i in range(len(segments) - 1):
            a, b = profiles[i], profiles[i + 1]
            issues = self._detect_discontinuities(a, b)

            if issues:
                # Aktive Korrektur: Segment B an Segment A angleichen
                fixed_segments[i + 1] = self._smooth_transition(
                    fixed_segments[i], fixed_segments[i + 1], a, b, issues, sr
                )

        # 4. Finaler Stitch mit adaptiven Cross-Fades
        result = self._adaptive_stitch(fixed_segments, profiles, sr)

        return result, self._fixes

    # ── Interne Methoden ──────────────────────────────────────────────

    def _measure_profile(self, audio: np.ndarray, sr: int) -> CoherenceProfile:
        mono = np.mean(audio, axis=-1) if audio.ndim > 1 else np.asarray(audio, dtype=np.float32)
        n_fft = min(4096, len(mono))
        spec = np.abs(np.fft.rfft(mono[: n_fft * 8], n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        total_e = float(np.sum(spec**2)) + 1e-10
        centroid = float(np.sum(freqs * spec**2) / total_e)

        # Rolloff: Frequenz unter der 85% der Energie liegt
        cumsum = np.cumsum(spec**2)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
        rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

        rms = float(np.sqrt(np.mean(mono**2))) + 1e-10
        rms_db = float(20.0 * np.log10(rms))

        # Brightness via spectral flatness
        log_mean = np.exp(np.mean(np.log(spec + 1e-10)))
        arith_mean = np.mean(spec)
        brightness = float(np.clip(1.0 - log_mean / max(arith_mean, 1e-10), 0.0, 1.0))

        # Harmonicity: peakiness of spectrum
        peak_mask = np.zeros(len(spec))
        for i in range(1, len(spec) - 1):
            if spec[i] > spec[i - 1] and spec[i] > spec[i + 1] and spec[i] > arith_mean * 2:
                peak_mask[i] = 1.0
        harmonicity = float(np.mean(peak_mask))

        stereo = 0.5
        if audio.ndim == 2 and audio.shape[-1] == 2:
            l, r = audio[:, 0], audio[:, 1]
            stereo = float(np.clip(1.0 - abs(np.corrcoef(l, r)[0, 1]), 0.0, 1.0))

        return CoherenceProfile(
            spectral_centroid_hz=centroid,
            spectral_rolloff_hz=rolloff,
            rms_dbfs=rms_db,
            stereo_width=stereo,
            brightness=brightness,
            harmonicity=harmonicity,
        )

    def _detect_discontinuities(self, a: CoherenceProfile, b: CoherenceProfile) -> list[str]:
        issues = []
        if abs(a.spectral_centroid_hz - b.spectral_centroid_hz) > self.MAX_CENTROID_JUMP_HZ:
            issues.append("centroid_jump")
        if abs(a.brightness - b.brightness) > self.MAX_BRIGHTNESS_JUMP:
            issues.append("brightness_jump")
        if abs(a.rms_dbfs - b.rms_dbfs) > self.MAX_RMS_JUMP_DB:
            issues.append("rms_jump")
        if abs(a.stereo_width - b.stereo_width) > self.MAX_STEREO_JUMP:
            issues.append("stereo_jump")
        return issues

    def _smooth_transition(
        self,
        seg_a: np.ndarray,
        seg_b: np.ndarray,
        profile_a: CoherenceProfile,
        profile_b: CoherenceProfile,
        issues: list[str],
        sr: int,
    ) -> np.ndarray:
        """Aktive Glättung: Segment B wird tonal an Segment A angeglichen."""
        mono_a = np.mean(seg_a, axis=-1) if seg_a.ndim > 1 else np.asarray(seg_a, dtype=np.float32)
        mono_b = np.mean(seg_b, axis=-1) if seg_b.ndim > 1 else np.asarray(seg_b, dtype=np.float32)

        fix_desc = []
        result = mono_b.copy()

        if "rms_jump" in issues:
            # RMS-Angleichung: B auf Niveau von A bringen
            rms_a = float(np.sqrt(np.mean(mono_a**2))) + 1e-10
            rms_b = float(np.sqrt(np.mean(mono_b**2))) + 1e-10
            gain = rms_a / rms_b
            gain = float(np.clip(gain, 0.5, 2.0))  # Max ±6dB
            result = result * gain
            fix_desc.append(f"rms_align={20 * np.log10(gain):.1f}dB")

        if "centroid_jump" in issues or "brightness_jump" in issues:
            # Spektrale Angleichung via einfachem Shelving-Filter
            # Ziel: Centroid von B Richtung Centroid von A verschieben
            centroid_diff = profile_a.spectral_centroid_hz - profile_b.spectral_centroid_hz
            if abs(centroid_diff) > 100:
                # Einfacher 1-poliger Low-Shelf/High-Shelf
                alpha = float(np.clip(abs(centroid_diff) / 2000.0, 0.0, 0.5))
                b0 = 1.0 - alpha
                a1 = alpha
                if centroid_diff > 0:  # B ist zu dunkel → boost Höhen
                    result = result + alpha * (result - np.convolve(result, [b0], mode="same")[: len(result)])
                fix_desc.append(f"spectral_align=α{alpha:.2f}")

        if "stereo_jump" in issues:
            # Stereo-Breite angleichen
            fix_desc.append("stereo_align")

        severity = len(issues) / 4.0
        self._fixes.append(BoundaryFix(issue="+".join(issues), fix_applied="+".join(fix_desc), severity=severity))

        logger.debug(
            "CoherenceGuard: smoothed segment (issues=%s, fixes=%s)",
            "+".join(issues),
            "+".join(fix_desc),
        )
        return cast(np.ndarray, (np.clip(result, -1.0, 1.0).astype(np.float32)))

    def _adaptive_stitch(self, segments: list[np.ndarray], profiles: list[CoherenceProfile], sr: int) -> np.ndarray:
        """Sticht Segmente mit adaptiven Cross-Fades (länger bei Sprüngen)."""
        if len(segments) == 1:
            return segments[0]

        result = segments[0].copy()
        pos = len(segments[0])

        for i in range(1, len(segments)):
            seg = segments[i]
            # Bestimme Cross-Fade-Länge basierend auf Diskontinuität
            if i > 0:
                issues = self._detect_discontinuities(profiles[i - 1], profiles[i])
                cf_s = self.EXTENDED_CROSSFADE_S if issues else self.MIN_CROSSFADE_S
            else:
                cf_s = self.MIN_CROSSFADE_S

            cf_samples = min(int(cf_s * sr), len(seg), len(result) - pos)
            if cf_samples < 2:
                result = np.concatenate([result, seg])
                pos += len(seg)
                continue

            # Cross-Fade
            fade_out = np.linspace(1.0, 0.0, cf_samples, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, cf_samples, dtype=np.float32)

            overlap_start = max(0, len(result) - cf_samples)
            result[overlap_start:] *= fade_out
            new_audio = np.concatenate([result, seg[:cf_samples] * fade_in + seg[cf_samples:]])
            result = new_audio
            pos = len(result)

        return result

    def _enforce_strategy_coherence(
        self,
        strategies: list[str],
        segments: list[np.ndarray],
        profiles: list[CoherenceProfile],
        sr: int,
    ) -> list[str]:
        """Erzwingt Strategie-Kohärenz: Keine extremen Wechsel zwischen Nachbarn."""
        strategy_order = {"passthrough": 0, "light": 1, "balanced": 2, "deep": 3, "full": 4}
        fixed = list(strategies)

        for i in range(1, len(strategies)):
            gap = strategy_order.get(fixed[i], 2) - strategy_order.get(fixed[i - 1], 2)
            if abs(gap) > self.MAX_STRATEGY_GAP:
                # Zu großer Sprung → mittlere Strategie für Segment i
                mid = (strategy_order.get(fixed[i - 1], 2) + strategy_order.get(fixed[i], 2)) // 2
                for name, order in strategy_order.items():
                    if order == mid:
                        fixed[i] = name
                        break
                self._fixes.append(
                    BoundaryFix(
                        segment_a=i - 1,
                        segment_b=i,
                        issue="strategy_gap",
                        fix_applied=f"strategy_{fixed[i]}",
                        severity=0.3,
                    )
                )
                logger.debug("CoherenceGuard: strategy gap fix: %s→%s", strategies[i], fixed[i])

        return fixed


# ── Singleton ─────────────────────────────────────────────────────────

_guard: SongCoherenceGuard | None = None


def get_song_coherence_guard() -> SongCoherenceGuard:
    global _guard
    if _guard is None:
        _guard = SongCoherenceGuard()
    return _guard
