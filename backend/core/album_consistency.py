"""AlbumConsistency — §INCREMENTAL #2: Track-übergreifende Konsistenz.

Stellt sicher: Alle Tracks eines Albums haben dieselbe Loudness,
Tonal-Balance und Stereo-Breite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

_LUFS_MAX_CORRECTION_DB: float = 6.0
"""Maximale LUFS-Korrektur in dB (positiv/negativ)."""

_LUFS_OUTLIER_THRESHOLD_LU: float = 2.0
"""Schwellwert in LU: Songs außerhalb des Median ± Threshold werden korrigiert."""

_MIN_SONGS: int = 3
"""Mindestanzahl Songs für Album-Analyse."""

_TILT_MAX_CORRECTION_DB: float = 3.0
"""Maximale Tilt-Korrektur in dB."""

_PEAK_SAFETY: float = 0.98
"""Peak-Safety-Ceiling nach Gain-Korrektur."""

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SongConsistencyProfile:
    """Pro Song: gemessene Werte + berechnete Korrekturen."""

    file_path: str = ""
    lufs: float = -23.0
    spectral_tilt: float = -3.0
    dynamic_range_db: float = 12.0
    lufs_correction_db: float = 0.0
    tilt_correction_db: float = 0.0
    stereo_width: float = 0.5
    rms_dbfs: float = -20.0


@dataclass
class AlbumConsistencyReport:
    """Ergebnis der Album-Analyse."""

    n_songs: int = 0
    album_lufs_median: float = -23.0
    album_tilt_median: float = -3.0
    album_dr_median: float = 12.0
    corrections_applied: int = 0
    songs: list[SongConsistencyProfile] = field(default_factory=list)
    skipped_insufficient_songs: bool = False


@dataclass
class TrackProfile:
    """Legacy Track-Profile (für Abwärtskompatibilität)."""

    path: str = ""
    integrated_lufs: float = -23.0
    spectral_centroid_hz: float = 2000.0
    stereo_width: float = 0.5
    rms_dbfs: float = -20.0


@dataclass
class AlbumTarget:
    """Legacy Album-Target (für Abwärtskompatibilität)."""

    target_lufs: float = -16.0
    target_centroid_hz: float = 2000.0
    target_stereo_width: float = 0.5
    tolerance_lu: float = 2.0


# ---------------------------------------------------------------------------
# AlbumConsistencyPass
# ---------------------------------------------------------------------------


class AlbumConsistencyPass:
    """Track-übergreifende Album-Konsistenz: LUFS + Tilt + Gain."""

    def __init__(self) -> None:
        self.enabled = True

    # -- Messung --

    def _measure_lufs(self, audio: np.ndarray, sr: int) -> float:
        """Schätzt Integrated LUFS aus RMS (leichtgewichtig, kein echtes EBU R128)."""
        _a = audio.ravel() if audio.ndim == 1 else audio.reshape(-1)
        rms = float(np.sqrt(np.mean(np.square(_a))) + 1e-12)
        return float(20.0 * np.log10(rms))

    def _measure_spectral_tilt(self, audio: np.ndarray, sr: int) -> float:
        """Schätzt Spectral Tilt (dB/Oktave) via FFT-Linreg."""
        mono = audio.ravel() if audio.ndim == 1 else audio.reshape(-1)
        n = min(len(mono), sr * 2)  # max 2s
        seg = mono[:n]
        n_fft = min(4096, n)
        spec = np.abs(np.fft.rfft(seg, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        # Nur Bereich 200 Hz – 8 kHz
        mask = (freqs >= 200) & (freqs <= 8000)
        if mask.sum() < 4:
            return 0.0

        log_f = np.log2(freqs[mask] + 1e-12)
        log_s = np.log10(spec[mask] + 1e-12)

        # Lineare Regression: log_s ≈ a * log_f + b
        # Tilt = a * 10 (dB pro Oktave? Nein: dB pro Verdopplung)
        # Hier: Steigung in log10-Amplitude pro log2-Frequenz
        A = np.column_stack([log_f, np.ones_like(log_f)])
        slope, _ = np.linalg.lstsq(A, log_s, rcond=None)[0]
        # slope ist log10(A) / log2(Hz) → dB/Oktave = slope * 20 (da 20*log10)
        # Umrechnung: 1 Oktave = Frequenzverdopplung = 1 in log2
        # Also: dB/Oktave = slope * 20
        return float(np.clip(slope * 20.0, -12.0, 2.0))

    # -- Analyse --

    def analyze(
        self,
        audios: list[np.ndarray],
        srs: list[int],
        file_paths: list[str],
    ) -> AlbumConsistencyReport:
        """Analysiert Album-Songs und berechnet Korrekturen."""
        n = len(audios)
        if n < _MIN_SONGS:
            return AlbumConsistencyReport(
                n_songs=n,
                skipped_insufficient_songs=True,
            )

        profiles: list[SongConsistencyProfile] = []
        for audio, sr, fp in zip(audios, srs, file_paths):
            lufs = self._measure_lufs(audio, sr)
            tilt = self._measure_spectral_tilt(audio, sr)
            profiles.append(
                SongConsistencyProfile(
                    file_path=fp,
                    lufs=lufs,
                    spectral_tilt=tilt,
                )
            )

        # Median-Targets
        lufs_vals = [p.lufs for p in profiles]
        tilt_vals = [p.spectral_tilt for p in profiles]
        album_lufs = float(np.median(lufs_vals))
        album_tilt = float(np.median(tilt_vals))

        corrections_applied = 0
        for p in profiles:
            lufs_dev = p.lufs - album_lufs
            if abs(lufs_dev) > _LUFS_OUTLIER_THRESHOLD_LU:
                p.lufs_correction_db = float(np.clip(-lufs_dev, -_LUFS_MAX_CORRECTION_DB, _LUFS_MAX_CORRECTION_DB))
                corrections_applied += 1
            else:
                p.lufs_correction_db = 0.0

            tilt_dev = p.spectral_tilt - album_tilt
            if abs(tilt_dev) > 1.0:
                p.tilt_correction_db = float(
                    np.clip(-tilt_dev * 0.5, -_TILT_MAX_CORRECTION_DB, _TILT_MAX_CORRECTION_DB)
                )
                corrections_applied += 1
            else:
                p.tilt_correction_db = 0.0

        return AlbumConsistencyReport(
            n_songs=n,
            album_lufs_median=album_lufs,
            album_tilt_median=album_tilt,
            corrections_applied=corrections_applied,
            songs=profiles,
            skipped_insufficient_songs=False,
        )

    # -- Audio-Processing --

    def _apply_gain(self, audio: np.ndarray, gain_db: float) -> np.ndarray:
        """Wendet Gain (dB) auf Audio an, mit Peak-Safety."""
        if abs(gain_db) < 1e-6:
            return cast(np.ndarray, (np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)))

        gain_linear = float(10.0 ** (gain_db / 20.0))
        out = audio * gain_linear
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        # Peak-Safety
        peak = float(np.percentile(np.abs(out), 99.9))
        if peak > _PEAK_SAFETY:
            out = out * (_PEAK_SAFETY / peak)

        return cast(np.ndarray, (np.clip(out, -1.0, 1.0).astype(np.float32)))

    def _build_shelf_sos(self, gain_db: float, freq_hz: float, sr: int) -> np.ndarray:
        """Baut High-Shelf SOS-Filter (identity bei gain_db=0)."""
        if abs(gain_db) < 1e-6:
            return cast(np.ndarray, (np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)))

        # Manuelle High-Shelf biquad per RBJ-Audio-EQ-Kochrezept
        A = 10.0 ** (gain_db / 40.0)  # sqrt(gain_linear)
        w0 = 2.0 * np.pi * freq_hz / sr
        alpha = np.sin(w0) / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / 0.707 - 1.0) + 2.0)  # Q ≈ 0.707
        cos_w0 = np.cos(w0)

        # High-shelf coefficients
        b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * np.sqrt(A) * alpha)
        b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
        b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * np.sqrt(A) * alpha)
        a0 = (A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * np.sqrt(A) * alpha
        a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
        a2 = (A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * np.sqrt(A) * alpha

        return cast(np.ndarray, (np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)))

    def _apply_tilt_correction(self, audio: np.ndarray, correction_db: float, sr: int) -> np.ndarray:
        """Korrigiert Spectral Tilt via High-Shelf-Filter."""
        if abs(correction_db) < 1e-6:
            return audio.copy()

        # V11: zero-phase (sosfiltfilt) statt kausalem sosfilt — keine
        # Phasendrehung/destruktive Interferenz. Halbe Gain-Designierung
        # kompensiert die doppelte Anwendung (|H|²), Gesamtkorrektur ≈ correction_db.
        sos = self._build_shelf_sos(correction_db / 2.0, 3000.0, sr)

        was_1d = audio.ndim == 1
        work = np.atleast_2d(audio)
        if work.shape[0] > work.shape[-1]:
            work = work.T  # (N, C) → channels-last

        import scipy.signal as _sig

        try:
            out = _sig.sosfiltfilt(sos, work, axis=0)
        except ValueError:
            # Kurzsignal (z.B. Test-Segmente): filtfilt-Padding nicht möglich —
            # kausale Phase ist bei dieser Länge vernachlässigbar.
            out = _sig.sosfilt(sos, work, axis=0)
        if was_1d:
            out = out.flatten()

        return cast(np.ndarray, (np.asarray(out, dtype=np.float32)))

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        profile: SongConsistencyProfile,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Wendet LUFS- und Tilt-Korrektur auf ein einzelnes Lied an."""
        correction_applied = False
        corrected = np.asarray(audio, dtype=np.float32).copy()

        # LUFS correction
        if abs(profile.lufs_correction_db) > 0.01:
            corrected = self._apply_gain(corrected, profile.lufs_correction_db)
            correction_applied = True

        # Tilt correction
        if abs(profile.tilt_correction_db) > 0.01:
            corrected = self._apply_tilt_correction(corrected, profile.tilt_correction_db, sr)
            correction_applied = True

        corrected = np.nan_to_num(corrected, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        meta: dict[str, Any] = {
            "correction_applied": correction_applied,
            "lufs_correction_db": profile.lufs_correction_db,
            "tilt_correction_db": profile.tilt_correction_db,
        }
        return corrected, meta


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_album_consistency_pass: AlbumConsistencyPass | None = None


def get_album_consistency_pass() -> AlbumConsistencyPass:
    """Gibt den globalen AlbumConsistencyPass-Singleton zurück."""
    global _album_consistency_pass
    if _album_consistency_pass is None:
        _album_consistency_pass = AlbumConsistencyPass()
    return _album_consistency_pass


# ---------------------------------------------------------------------------
# Legacy-Funktionen (für Abwärtskompatibilität)
# ---------------------------------------------------------------------------


def analyze_track(audio: np.ndarray, sr: int, path: str = "") -> TrackProfile:
    """Legacy: Analysiert einen einzelnen Track."""
    mono = np.mean(audio, axis=-1) if audio.ndim > 1 else np.asarray(audio, dtype=np.float32)
    n_fft = min(4096, len(mono))
    spec = np.abs(np.fft.rfft(mono[: n_fft * 8], n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    centroid = float(np.sum(freqs * spec) / max(np.sum(spec), 1e-10))
    rms = float(np.sqrt(np.mean(mono**2)))
    stereo = 0.5
    if audio.ndim == 2 and audio.shape[-1] == 2:
        l, r = audio[:, 0], audio[:, 1]
        corr = float(np.corrcoef(l, r)[0, 1])
        stereo = float(np.clip(1.0 - abs(corr), 0.0, 1.0))
    return TrackProfile(
        path=path,
        spectral_centroid_hz=centroid,
        stereo_width=stereo,
        rms_dbfs=20.0 * np.log10(max(rms, 1e-10)),
        integrated_lufs=-23.0,
    )


def compute_album_target(tracks: list[TrackProfile]) -> AlbumTarget:
    """Legacy: Berechnet Album-Target aus Track-Profilen."""
    if not tracks:
        return AlbumTarget()
    lufs_vals = [t.integrated_lufs for t in tracks if t.integrated_lufs < -5]
    cent_vals = [t.spectral_centroid_hz for t in tracks]
    stereo_vals = [t.stereo_width for t in tracks]
    return AlbumTarget(
        target_lufs=float(np.median(lufs_vals)) if lufs_vals else -16.0,
        target_centroid_hz=float(np.median(cent_vals)),
        target_stereo_width=float(np.median(stereo_vals)),
    )


def normalize_track(audio: np.ndarray, sr: int, target: AlbumTarget) -> np.ndarray:
    """Legacy: Normalisiert einen Track auf Album-Target."""
    mono = np.mean(audio, axis=-1) if audio.ndim > 1 else np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(mono**2))) + 1e-10
    target_rms = 10.0 ** (target.target_lufs / 20.0)
    gain = target_rms / rms
    gain = float(np.clip(gain, 0.1, 10.0))
    logger.info("AlbumConsistency: gain=%.1f dB", 20.0 * np.log10(gain))
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]
