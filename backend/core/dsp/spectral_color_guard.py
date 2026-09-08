"""§SCK (V24) Spektralfarbe-Korrelations-Guard.

Prüft nach EQ/NR-Phasen ob die 1/3-Oktav-Spektralfarbe im Band 200–8000 Hz
erhalten bleibt. Korrelation < 0.97 → Phase-Strength − 30 % (WARNING, kein Rollback).

Kanonische Nutzung (UV3 post-phase hook):
    from backend.core.dsp.spectral_color_guard import check_spectral_color_preservation, SpectralColorResult
    result = check_spectral_color_preservation(pre, post, sr)
    if not result.ok:
        # Wet-Blend 70/30 anwenden (Strength − 30 %)
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from backend.core.residuum_masking import estimate_delta_masking_jnd_db

logger = logging.getLogger(__name__)

# 1/3-Oktav-Mittelpunkte 160 Hz bis 10 kHz (ISO 266)
_THIRD_OCT_CENTERS_HZ: list[float] = [
    160.0,
    200.0,
    250.0,
    315.0,
    400.0,
    500.0,
    630.0,
    800.0,
    1000.0,
    1250.0,
    1600.0,
    2000.0,
    2500.0,
    3150.0,
    4000.0,
    5000.0,
    6300.0,
    8000.0,
]

# Korrelationsschwellwert
SPECTRAL_COLOR_THRESHOLD = 0.97


@dataclass
class SpectralColorResult:
    """Ergebnis der Spektralfarbe-Prüfung.

    Attributes:
        correlation: Pearson-Korrelation der 1/3-Oktav-Profile pre/post [0..1].
        ok: True wenn correlation >= 0.97.
        pre_profile_db: 1/3-Oktav-Profil vor Phase (dB).
        post_profile_db: 1/3-Oktav-Profil nach Phase (dB).
    """

    correlation: float
    ok: bool
    pre_profile_db: list[float]
    post_profile_db: list[float]


def _third_octave_profile(audio_mono: np.ndarray, sr: int) -> np.ndarray:
    """Berechnet 1/3-Oktav-Profil (dB) für ISO-266-Mittenfrequenzen."""
    n_fft = min(16384, len(audio_mono))
    if n_fft < 256:
        return np.zeros(len(_THIRD_OCT_CENTERS_HZ), dtype=np.float32)  # type: ignore[no-any-return]

    spectrum = np.abs(np.fft.rfft(audio_mono[:n_fft].astype(np.float32), n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    profile = np.zeros(len(_THIRD_OCT_CENTERS_HZ), dtype=np.float32)
    factor = 10.0 ** (1.0 / 20.0)  # 1/3-Oktav-Bandbreite: fc/√2^(1/3)

    for i, fc in enumerate(_THIRD_OCT_CENTERS_HZ):
        lo = fc / factor
        hi = fc * factor
        mask = (freqs >= lo) & (freqs <= hi)
        if mask.sum() == 0:
            continue
        band_energy = float(np.mean(spectrum[mask]) + 1e-14)
        profile[i] = float(10.0 * np.log10(band_energy))

    return np.nan_to_num(profile, nan=-120.0, posinf=0.0, neginf=-120.0)  # type: ignore[no-any-return]


def check_spectral_color_preservation(
    pre: np.ndarray,
    post: np.ndarray,
    sr: int,
    *,
    threshold: float = 0.97,
) -> SpectralColorResult:
    """Prüft ob die 1/3-Oktav-Spektralfarbe zwischen pre und post erhalten bleibt.

    Args:
        pre: Audio vor der Phase. Shape [N] oder [2, N].
        post: Audio nach der Phase (same shape as pre).
        sr: Sample-Rate (muss 48000 sein).
        threshold: Korrelations-Schwellwert. None → auto-adaptiv aus CalibrationContext.
                   §v10.303: 0.97 für depth=1, 0.73 für depth=4, linear interpoliert.

    Returns:
        SpectralColorResult. ok=False wenn correlation < threshold.
    """
    assert sr == 48000
    # §v10.303: Restorability-adaptiver Default — bei degraded Material ist
    # spektrale Abweichung vom Input ERWÜNSCHT (Rauschen wurde entfernt, BW erweitert).
    if threshold == 0.97:  # Default-Wert → auto-adaptiv
        try:
            from backend.core.calibration_context import get_calibration_context

            _ctx = get_calibration_context()
            if _ctx is not None:
                _rs = float(np.clip(_ctx.restorability_score, 0.0, 100.0))
                threshold = float(np.clip(0.97 - (0.97 - 0.70) * (100.0 - _rs) / 100.0, 0.70, 0.97))
        except Exception:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            pass  # Fallback: 0.97
    _empty = [0.0] * len(_THIRD_OCT_CENTERS_HZ)
    _fallback = SpectralColorResult(correlation=1.0, ok=True, pre_profile_db=_empty, post_profile_db=_empty)

    try:
        pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        post = np.nan_to_num(post, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if pre.shape != post.shape or pre.size < 256:
            return _fallback

        # §2.51 Stereo-Axis-Invariante: channels-last (N,2) korrekt behandeln
        if pre.ndim == 2:
            _ax = 0 if pre.shape[0] <= 2 else 1
            pre_mono = pre.mean(axis=_ax)
        else:
            pre_mono = pre
        if post.ndim == 2:
            _ax = 0 if post.shape[0] <= 2 else 1
            post_mono = post.mean(axis=_ax)
        else:
            post_mono = post

        pre_profile = _third_octave_profile(pre_mono, sr)
        post_profile = _third_octave_profile(post_mono, sr)

        # Pearson-Korrelation über 1/3-Oktav-Profil
        pre_std = float(np.std(pre_profile))
        post_std = float(np.std(post_profile))

        # Guard: Wenn pre-Profil spektral flach ist (z.B. weißes Rauschen / Tape-Hiss),
        # ist die Pearson-Korrelation mathematisch undefiniert (0/0) → fallback ok=True.
        # Physikalisch: ein flaches Pre-Spektrum hat keine definierbare Spektralfarbe,
        # daher ist die Preservierungs-Prüfung nicht sinnvoll anwendbar.
        if pre_std < 0.5:
            return _fallback

        corr = float(np.mean((pre_profile - np.mean(pre_profile)) * (post_profile - np.mean(post_profile))))
        corr /= (pre_std + 1e-9) * (post_std + 1e-9)
        corr = float(np.clip(np.nan_to_num(corr, nan=1.0), -1.0, 1.0))

        # §P1-3 (Hörordnung Ebene 2): Ist der Phasen-Delta lokal maskiert,
        # darf die Spektralfarbe weiter abweichen, bevor der Guard greift.
        # Korrelation ist keine dB-Größe → beschränkte Relaxation
        # (max. 0,20 bei voller 6-dB-JND, linear).
        _jnd = estimate_delta_masking_jnd_db(pre, post, sr, freq_range_hz=(160.0, 8000.0))
        _relax = float(np.clip(_jnd.jnd_db / 30.0, 0.0, 0.20))
        _eff_threshold = float(np.clip(threshold - _relax, 0.0, 1.0))

        ok = corr >= _eff_threshold

        if not ok:
            logger.info(
                "§V24 (Spec-Vintage-Guard) Spektralfarbe: Korrelation=%.3f < %.3f (Schwelle %.3f, §P1-3 JND=%.2f dB → Relaxation %.3f) → Verarbeitungsschritt-Strength − 30 %% (WARNING)",
                corr,
                _eff_threshold,
                threshold,
                _jnd.jnd_db,
                _relax,
            )

        return SpectralColorResult(
            correlation=round(corr, 4),
            ok=ok,
            pre_profile_db=[round(float(v), 2) for v in pre_profile],
            post_profile_db=[round(float(v), 2) for v in post_profile],
        )

    except Exception as exc:
        logger.debug("Pruefung_spectral_color_preservation nicht blockierend: %s", exc)
        return _fallback
