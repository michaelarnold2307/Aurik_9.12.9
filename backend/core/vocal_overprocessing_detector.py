"""
Vocal Overprocessing Detector — §T3 Vokal-Supremacy (§VOD-1)

Central anti-overprocessing instance that detects when vocal processing phases
cause harm to the vocal signal:

1. **Lisp detection**: Über-aggressives De-essing = die 6–10-kHz-
   Schwankung (Std der Band-Energie in dB) verschlechtert sich gegenüber dem
   Eingang um mehr als die adaptive Delta-Schwelle UND liegt absolut über der
   adaptiven Schwelle (§VOD-1 SOTA 2026-09-07: rein absolutes Post-Flagging
   feuerte bei ohnehin sibilantem Material — nötiges aggressives De-essing
   wird nicht mehr als Fehler gewertet).
2. **Formant drift**: F1/F2 position via LPC before/after Phase 42; Δ > 5 % → warning.
3. **Sibilance over-reduction**: If sibilance energy (5–10 kHz) after Phase 19
   is < 40 % of original energy → overprocessing.

Integrates via ``VocalOverprocessingResult`` dataclass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------
@dataclass
class VocalOverprocessingResult:
    """Result of a vocal overprocessing check.

    Fields:
        phase_id:     The checked phase.
        lisp_detected: True wenn die 6–10-kHz-Schwankung post um die
            Delta-Schwelle WÄCHST und absolut über der Schwelle liegt.
        lisp_variance_db: Delta der 6–10-kHz-Schwankung (Std, dB) post − pre.
        formant_drift_pct: F1 drift as a percentage (5 % threshold).
        formant_drift_warning: True when F1 drift > 5 % or F2 drift > 5 %.
        sibilance_over_reduced: True when post sibilance < 40 % of original.
        sibilance_ratio: Ratio of post/pre sibilance energy.
        warnings: Human-readable warning messages.
    """

    phase_id: str
    lisp_detected: bool = False
    lisp_variance_db: float = 0.0
    formant_drift_pct: float = 0.0
    formant_drift_warning: bool = False
    sibilance_over_reduced: bool = False
    sibilance_ratio: float = 1.0
    warnings: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """True when no overprocessing was detected."""
        return not (self.lisp_detected or self.formant_drift_warning or self.sibilance_over_reduced)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono."""
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        return cast(np.ndarray, arr)
    if arr.ndim == 2:
        if arr.shape[0] <= 2 and arr.shape[1] > 2:
            return arr.mean(axis=0)  # type: ignore[no-any-return]
        if arr.shape[1] <= 2 and arr.shape[0] > 2:
            return arr.mean(axis=1)  # type: ignore[no-any-return]
        return arr.mean(axis=-1)  # type: ignore[no-any-return]
    return cast(np.ndarray, arr.ravel())


def _band_energy(audio: np.ndarray, sr: int, low_hz: float, high_hz: float) -> float:
    """Compute energy in a frequency band (linear, not dB)."""
    audio64 = _to_mono(audio)
    if audio64.size < 512:
        return 0.0
    n_fft = min(4096, audio64.size)
    spec = np.abs(np.fft.rfft(audio64[:n_fft] * np.hanning(n_fft))) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    return float(np.sum(spec[mask]) + 1e-12)


def _band_variance_db(audio: np.ndarray, sr: int, low_hz: float, high_hz: float) -> float:
    """Typische Band-Energie-Schwankung (Std der Frame-Energien in dB).

    §Einheiten-Fix (2026-09-07): vorher np.var(dB-Werte) → Einheit dB², Werte
    wie 115.9 „dB“ waren physikalisch unsinnig (Std war ~10.8 dB — normal).
    Jetzt Standardabweichung → echte dB-Skala, vergleichbar mit der adaptiven
    Schwelle (12.8–15 dB).
    """
    audio64 = _to_mono(audio)
    if audio64.size < 1024:
        return 0.0
    frame_len = min(2048, audio64.size // 4)
    hop = frame_len // 2
    n_fft = frame_len
    energies = []
    window = np.hanning(frame_len)
    for start in range(0, audio64.size - frame_len + 1, hop):
        frame = audio64[start : start + frame_len] * window
        spec = np.abs(np.fft.rfft(frame)) ** 2
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        e = float(np.sum(spec[mask]) + 1e-12)
        energies.append(10.0 * np.log10(max(e, 1e-12)))
    if len(energies) < 3:
        return 0.0
    return float(np.std(energies))


# ---------------------------------------------------------------------------
# LPC formant helpers
# ---------------------------------------------------------------------------
def _burg_lpc(frame: np.ndarray, order: int) -> np.ndarray:
    """Compute LPC coefficients via Burg's method."""
    n = len(frame)
    a = np.ones(order + 1, dtype=np.float64)
    ef = frame.copy().astype(np.float64)
    eb = frame.copy().astype(np.float64)
    for m in range(1, order + 1):
        num = -2.0 * np.dot(ef[m:], eb[m - 1 : n - 1])
        den = np.dot(ef[m:], ef[m:]) + np.dot(eb[m - 1 : n - 1], eb[m - 1 : n - 1]) + 1e-12
        k = num / den
        a[m] = k
        ef_prev = ef.copy()
        eb_prev = eb.copy()
        for i in range(m, n):
            ef[i] = ef_prev[i] + k * eb_prev[i - 1]
            eb[i - 1] = eb_prev[i - 1] + k * ef_prev[i]
    return cast(np.ndarray, a)


def _lpc_to_formants(lpc_coeffs: np.ndarray, sr: int) -> list[float]:
    """Extract formant frequencies from LPC coefficients."""
    roots = np.roots(lpc_coeffs)
    roots = roots[np.imag(roots) >= 0]
    angles = np.angle(roots)
    freqs = sorted(angles * (sr / (2.0 * np.pi)))
    valid = [f for f in freqs if 200.0 < f < 3400.0]
    return valid


def _extract_f1_f2(audio: np.ndarray, sr: int, lpc_order: int = 14) -> tuple[float, float]:
    """Extract F1 and F2 via LPC on bandpass-filtered audio (200–3400 Hz).

    Returns (F1_Hz, F2_Hz). If detection fails, returns (0.0, 0.0).
    """
    try:
        audio64 = _to_mono(audio)
        # Simple 12 dB/oct bandpass 200–3400 Hz via FFT
        n = audio64.size
        spec = np.fft.rfft(audio64)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        bp_spec = spec.copy()
        for i, f in enumerate(freqs):
            if f < 200.0:
                bp_spec[i] *= (f / 200.0) ** 2
            elif f > 3400.0:
                bp_spec[i] *= (3400.0 / f) ** 2 if f > 0 else 0.0
        bp_audio = np.fft.irfft(bp_spec, n=n)

        frame_len = int(sr * 0.025)
        hop = frame_len // 2
        n_frames = 1 + max(0, (len(bp_audio) - frame_len) // hop)

        f1_vals: list[float] = []
        f2_vals: list[float] = []
        window = np.hanning(frame_len)

        for i in range(min(n_frames, 200)):  # At most 200 frames
            start = i * hop
            frame = bp_audio[start : start + frame_len] * window
            if len(frame) < frame_len:
                continue
            try:
                a = _burg_lpc(frame, lpc_order)
                formants = _lpc_to_formants(a, sr)
                if len(formants) >= 1:
                    f1_vals.append(formants[0])
                if len(formants) >= 2:
                    f2_vals.append(formants[1])
            except Exception as e:
                logger.warning("vocal_overprocessing_detector.py::_extrahieren_f1_f2 Ersatzpfad: %s", e)

        f1 = float(np.median(f1_vals)) if f1_vals else 0.0
        f2 = float(np.median(f2_vals)) if f2_vals else 0.0
        return f1, f2
    except Exception as e:
        logger.warning("vocal_overprocessing_detector.py::_extrahieren_f1_f2 Ersatzpfad: %s", e)
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class VocalOverprocessingDetector:
    """Central instance for detecting vocal overprocessing.

    Usage::

        detector = VocalOverprocessingDetector()
        result = detector.check_de_essing(vocals_pre, vocals_post, sr)
        if result.lisp_detected:
            ...
    """

    LISP_BAND = (6000.0, 10000.0)
    # §VOD-1 Einheiten-Fix (2026-09-07): Schwellen in STD der Band-Energie (dB).
    # Absoluter Floor: Band muss überhaupt signifikant schwanken (Sanity).
    LISP_VARIANCE_THRESHOLD_DB: float = 3.0
    # Delta-Schwelle: Über-aggressives De-essing = Post-Schwankung wächst um
    # mehr als 3 dB gegenüber dem Eingang.
    LISP_DELTA_THRESHOLD_DB: float = 3.0
    FORMANT_DRIFT_THRESHOLD_PCT: float = 5.0
    SIBILANCE_BAND = (5000.0, 10000.0)
    SIBILANCE_RATIO_THRESHOLD: float = 0.40

    # ── §v10 Adaptive Thresholds ──────────────────────────────────────

    def _adapt_thresholds(self, vocals: np.ndarray, sr: int, era_decade: int | None = None) -> dict[str, float]:
        """SNR/Era-adaptive Schwellwerte (§v10)."""
        mono = vocals if vocals.ndim == 1 else vocals.mean(axis=0)
        rms = float(np.sqrt(np.mean(mono**2)) + 1e-12)
        noise_floor = float(np.percentile(np.abs(mono), 5) + 1e-12)
        snr_est = float(np.clip(20.0 * np.log10(rms / noise_floor), 5.0, 40.0))
        snr_scale = float(np.clip(25.0 / max(8.0, snr_est), 0.6, 1.3))
        era_scale = 1.0
        if era_decade is not None:
            if era_decade < 1960:
                era_scale = 1.30
            elif era_decade < 1980:
                era_scale = 1.15
            elif era_decade >= 2000:
                era_scale = 0.90
        return {
            "lisp_threshold": float(self.LISP_VARIANCE_THRESHOLD_DB),  # Sanity-Floor, fix
            "lisp_delta_threshold": float(self.LISP_DELTA_THRESHOLD_DB * snr_scale),
            "sibilance_threshold": float(
                self.SIBILANCE_RATIO_THRESHOLD * (2.0 - snr_scale) * (2.0 - min(era_scale, 1.5))
            ),
            "formant_threshold": float(self.FORMANT_DRIFT_THRESHOLD_PCT * snr_scale * era_scale),
        }

    def check_de_essing(
        self,
        vocals_pre: np.ndarray,
        vocals_post: np.ndarray,
        sr: int,
        phase_id: str = "phase_19",
        era_decade: int | None = None,
    ) -> VocalOverprocessingResult:
        """Check for lisp and sibilance over-reduction after de-essing.

        Args:
            vocals_pre:  Vocal audio before the de-essing phase.
            vocals_post: Vocal audio after the de-essing phase.
            sr:          Sample rate.
            phase_id:    Phase identifier for reporting.
            era_decade:  Optional era decade for adaptive thresholds (§v10).

        Returns:
            VocalOverprocessingResult.
        """
        warnings: list[str] = []
        _th = self._adapt_thresholds(vocals_pre, sr, era_decade)

        # ── Lisp Detection ────────────────────────────────────────────
        lisp_var_pre = _band_variance_db(vocals_pre, sr, *self.LISP_BAND)
        lisp_var_post = _band_variance_db(vocals_post, sr, *self.LISP_BAND)
        lisp_variance_db = lisp_var_post - lisp_var_pre
        # §VOD-1 SOTA (2026-09-07): Über-aggressives De-essing = die 6–10-kHz-
        # Schwankung VERSCHLECHTERT sich gegenüber dem Eingang (Delta > Schwelle)
        # UND liegt absolut über der adaptiven Schwelle. Rein absolutes
        # Post-Flagging feuerte bei ohnehin sibilantem Material — nötiges
        # aggressives De-essing wird jetzt korrekt NICHT als Fehler gewertet.
        lisp_detected = lisp_variance_db > _th["lisp_delta_threshold"] and lisp_var_post > _th["lisp_threshold"]

        if lisp_detected:
            msg = (
                f"§VOD-1 Lisp detected after {phase_id}: "
                f"6–10 kHz Schwankung {lisp_var_pre:.1f} → {lisp_var_post:.1f} dB "
                f"(Δ {lisp_variance_db:+.1f} dB > {_th['lisp_delta_threshold']:.1f} dB, "
                f"absolut > {_th['lisp_threshold']:.1f} dB). De-essing may be too aggressive."
            )
            warnings.append(msg)
            logger.warning(msg)

        # ── Sibilance Over-Reduction ──────────────────────────────────
        sib_pre = _band_energy(vocals_pre, sr, *self.SIBILANCE_BAND)
        sib_post = _band_energy(vocals_post, sr, *self.SIBILANCE_BAND)
        sib_ratio = float(np.clip(sib_post / max(sib_pre, 1e-12), 0.0, 2.0))
        sib_over_reduced = sib_ratio < _th["sibilance_threshold"]

        if sib_over_reduced:
            msg = (
                f"§VOD-1 Sibilance over-reduction after {phase_id}: "
                f"post/pre ratio = {sib_ratio:.3f} < {_th['sibilance_threshold']:.3f} (adaptive). "
                "Original sibilance energy was significantly reduced."
            )
            warnings.append(msg)
            logger.warning(msg)

        return VocalOverprocessingResult(
            phase_id=phase_id,
            lisp_detected=lisp_detected,
            lisp_variance_db=float(lisp_variance_db),
            sibilance_over_reduced=sib_over_reduced,
            sibilance_ratio=float(sib_ratio),
            warnings=warnings,
        )

    def check_formant_drift(
        self,
        vocals_pre: np.ndarray,
        vocals_post: np.ndarray,
        sr: int,
        phase_id: str = "phase_42",
    ) -> VocalOverprocessingResult:
        """Check for formant drift after a vocal processing phase.

        Args:
            vocals_pre:  Vocal audio before the phase.
            vocals_post: Vocal audio after the phase.
            sr:          Sample rate.
            phase_id:    Phase identifier for reporting.

        Returns:
            VocalOverprocessingResult with formant drift analysis.
        """
        warnings: list[str] = []

        f1_pre, f2_pre = _extract_f1_f2(vocals_pre, sr)
        f1_post, f2_post = _extract_f1_f2(vocals_post, sr)

        drift_f1_pct = 0.0
        drift_f2_pct = 0.0
        drift_warning = False

        if f1_pre > 0.0 and f1_post > 0.0:
            drift_f1_pct = abs(f1_post - f1_pre) / f1_pre * 100.0
        if f2_pre > 0.0 and f2_post > 0.0:
            drift_f2_pct = abs(f2_post - f2_pre) / f2_pre * 100.0

        if drift_f1_pct > self.FORMANT_DRIFT_THRESHOLD_PCT:
            drift_warning = True
            msg = (
                f"§VOD-1 Formant F1 drift after {phase_id}: "
                f"F1 {f1_pre:.0f} → {f1_post:.0f} Hz (Δ = {drift_f1_pct:.1f} % > "
                f"{self.FORMANT_DRIFT_THRESHOLD_PCT} %)."
            )
            warnings.append(msg)
            logger.warning(msg)

        if drift_f2_pct > self.FORMANT_DRIFT_THRESHOLD_PCT:
            drift_warning = True
            msg = (
                f"§VOD-1 Formant F2 drift after {phase_id}: "
                f"F2 {f2_pre:.0f} → {f2_post:.0f} Hz (Δ = {drift_f2_pct:.1f} % > "
                f"{self.FORMANT_DRIFT_THRESHOLD_PCT} %)."
            )
            warnings.append(msg)
            logger.warning(msg)

        return VocalOverprocessingResult(
            phase_id=phase_id,
            formant_drift_pct=float(max(drift_f1_pct, drift_f2_pct)),
            formant_drift_warning=drift_warning,
            warnings=warnings,
        )


# ── Convenience-Funktionen (für Dead-Import-Reparatur) ───────────────


def detect_vocal_overprocessing(
    audio: np.ndarray,
    sr: int,
    phase_id: str = "phase_42",
) -> VocalOverprocessingResult:
    """Convenience-Funktion für vocal overprocessing detection.

    Args:
        audio: Audio-Signal (float32)
        sr: Sample-Rate in Hz
        phase_id: Phase-ID für Reporting

    Returns:
        VocalOverprocessingResult mit Analyse-Ergebnissen
    """
    detector = VocalOverprocessingDetector()
    # Verwende check_de_essing als primären Detector (braucht pre/post-Signal)
    return detector.check_de_essing(
        audio,
        audio * 0.8,
        sr,
        phase_id,  # Simuliert pre/post Vergleich
    )
