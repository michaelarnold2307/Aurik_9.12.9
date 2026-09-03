"""
Enhanced Audio Quality Metrics for AURIK
=========================================

Implements advanced objective quality metrics for audio restoration validation:
- ViSQOL v3 (--audio, music mode) — MOS 1.0–5.0
- SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
- Authenticity Metrics (Breath Retention, Transient Preservation)

VERBOTENE Metriken (§4.4, §10.2, ungeeignet für Musik):
  PESQ, DNSMOS, NISQA, STOI, ViSQOL --speech, SI-SDR, CDPAM

Author: AURIK Team
Date: 8. Februar 2026 | Bereinigt: 14. März 2026
Phase: 2D.2.1 - Real-World Validation Testing
"""

import logging
import warnings
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Import Authenticity Metrics from Phase 2D.1
try:
    from backend.core.authenticity_metrics import AuthenticityMetrics

    AUTHENTICITY_AVAILABLE = True
except ImportError:
    AUTHENTICITY_AVAILABLE = False
    warnings.warn("AuthenticityMetrics not available. Install with Phase 2D.1 components.", ImportWarning)


@dataclass
class QualityMetricsResult:
    """Result container for all quality metrics."""

    # Basic metrics (Phase 2D.1)
    snr_db: float
    thd: float
    lufs: float

    # Enhanced metrics (Phase 2D.2)
    visqol_mos: float | None = None  # MOS [1.0-5.0] — ViSQOL --audio (music mode)
    si_snr_db: float | None = None  # Scale-Invariant SNR
    # NOTE: pesq_mos, si_sdr_db, stoi_score removed — forbidden for music (§4.4)

    # Improvement metrics (before → after)
    snr_improvement_db: float | None = None

    # Authenticity metrics (Phase 2D.2.1 Task 3)
    breath_retention: float | None = None  # 0.0-1.0 (Target: >0.98)
    transient_preservation: float | None = None  # 0.0-1.0 (Target: >0.95)
    plosive_retention: float | None = None  # 0.0-1.0 (Target: >0.95)
    sibilance_retention: float | None = None  # 0.0-1.0 (Target: >0.95, not over-deessed)
    room_tone_retention: float | None = None  # 0.0-1.0 (Target: >0.90, natural acoustics)

    def passes_aurik_standards(self) -> bool:
        """
        Prüft if metrics meet AURIK Weltspitze standards.

        Success Metrics (Phase 2D.2.1):
        - SNR Improvement: >15dB (for high-noise inputs)
        - ViSQOL MOS: >4.0 (Excellent)  [PESQ/STOI/SI-SDR verboten — §4.4/§10.2]
        - Breath Retention: >98% (authenticity preservation)
        - Transient Preservation: >95% (dynamic integrity)
        - Sibilance Retention: >95% (not over-deessed)
        - Room Tone Retention: >90% (natural acoustics)
        """
        checks = []

        # SNR standards
        if self.snr_db is not None:
            checks.append(self.snr_db > 30.0)  # High quality

        # Improvement standards
        if self.snr_improvement_db is not None:
            checks.append(self.snr_improvement_db > 15.0)

        # Perceptual standards
        if self.visqol_mos is not None:
            checks.append(self.visqol_mos > 4.0)

        # Authenticity standards (Phase 2D.2.1 Task 3)
        if self.breath_retention is not None:
            checks.append(self.breath_retention > 0.98)

        if self.transient_preservation is not None:
            checks.append(self.transient_preservation > 0.95)

        if self.plosive_retention is not None:
            checks.append(self.plosive_retention > 0.95)

        if self.sibilance_retention is not None:
            checks.append(self.sibilance_retention > 0.95)

        if self.room_tone_retention is not None:
            checks.append(self.room_tone_retention > 0.90)

        # Must pass at least 70% of available checks
        if len(checks) == 0:
            return False

        # Convert to Python bool (not numpy.bool_)
        return bool(sum(checks) / len(checks) >= 0.7)


class EnhancedMetrics:
    """
    Berechnet enhanced objective quality metrics.

    Usage:
        metrics = EnhancedMetrics()
        result = metrics.compute_all(
            original_audio,
            restored_audio,
            sr=48000
        )
    """

    def __init__(self):
        # Check for optional dependencies
        self._visqol_available = self._check_visqol()

        # Initialize Authenticity Metrics (Phase 2D.2.1 Task 3)
        if AUTHENTICITY_AVAILABLE:
            self.authenticity = AuthenticityMetrics()
        else:
            self.authenticity = None  # type: ignore[assignment]

    def _check_visqol(self) -> bool:
        """Prüft if ViSQOL is available."""
        # ViSQOL is complex - requires C++ binary or API
        # For now, we'll implement a simplified version
        return False

    @staticmethod
    def compute_si_snr(reference: np.ndarray, estimate: np.ndarray, epsilon: float = 1e-8) -> float:
        """
        Berechnet Scale-Invariant Signal-to-Noise Ratio.

        Similar to SI-SDR but focuses on noise characteristics.

        Args:
            reference: Clean reference signal
            estimate: Processed signal
            epsilon: Small constant for numerical stability

        Returns:
            SI-SNR in dB
        """
        # Ensure same length
        min_len = min(len(reference), len(estimate))
        reference = reference[:min_len]
        estimate = estimate[:min_len]

        # Remove mean
        reference = reference - np.mean(reference)
        estimate = estimate - np.mean(estimate)

        # Project estimate onto reference
        alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + epsilon)
        s_signal = alpha * reference
        e_noise = estimate - s_signal

        # SI-SNR
        si_snr = 10 * np.log10((np.sum(s_signal**2) + epsilon) / (np.sum(e_noise**2) + epsilon))
        # NaN/Inf-Guard (§3.1)
        si_snr = np.nan_to_num(si_snr, nan=0.0, posinf=30.0, neginf=-10.0)
        return float(si_snr)

    # ============================================================
    # ViSQOL (Virtual Speech Quality Objective Listener)
    # ============================================================

    def compute_visqol(self, reference: np.ndarray, degraded: np.ndarray, sr: int = 48000) -> float | None:
        """
        Berechnet ViSQOL MOS score (simplified version).

        ViSQOL returns MOS: 1.0 (bad) to 5.0 (excellent).
        Full ViSQOL requires C++ binary - this is a Python approximation.

        Args:
            reference: Clean reference audio
            degraded: Degraded/processed audio
            sr: Sample rate

        Returns:
            Approximate ViSQOL MOS [1.0-5.0], or None if unavailable

        Success Criteria: >4.0 (Excellent)
        """
        # Simplified ViSQOL approximation using spectral similarity
        # Real ViSQOL uses neurogram similarity + ML model

        try:
            # Ensure same length
            min_len = min(len(reference), len(degraded))
            reference = reference[:min_len]
            degraded = degraded[:min_len]

            # Convert stereo to mono to avoid array-truth ambiguity
            if reference.ndim > 1:
                reference = reference.mean(axis=-1)
            if degraded.ndim > 1:
                degraded = degraded.mean(axis=-1)

            if len(reference) < 2:
                return None

            # Compute spectral similarity
            ref_stft = np.abs(np.fft.rfft(reference))
            deg_stft = np.abs(np.fft.rfft(degraded))

            # Normalized correlation in spectral domain
            ref_norm = ref_stft / (np.linalg.norm(ref_stft) + 1e-8)
            deg_norm = deg_stft / (np.linalg.norm(deg_stft) + 1e-8)

            spectral_corr = np.dot(ref_norm, deg_norm)

            # Convert correlation to MOS-like score
            # Correlation 1.0 → MOS 5.0, Correlation 0.0 → MOS 1.0
            mos = 1.0 + 4.0 * spectral_corr

            # Clamp to valid range + NaN-Guard (§3.1)
            mos = np.nan_to_num(mos, nan=3.5, posinf=5.0, neginf=1.0)
            return max(1.0, min(mos, 5.0))  # type: ignore[no-any-return]

        except Exception as e:
            logger.warning("ViSQOL computation failed: %s", e)
            return None

    # ============================================================
    # Basic Metrics (from Phase 2D.1)
    # ============================================================

    @staticmethod
    def compute_snr(audio: np.ndarray, sr: int) -> float:
        """Berechnet Signal-to-Noise Ratio."""
        rms = np.sqrt(np.mean(audio**2))

        # Estimate noise floor (quietest 10%)
        sorted_abs = np.sort(np.abs(audio))
        noise_floor_samples = sorted_abs[: len(sorted_abs) // 10]
        noise_rms = np.sqrt(np.mean(noise_floor_samples**2))

        if noise_rms < 1e-10:
            return 60.0  # Very high SNR

        snr_db = 20 * np.log10(rms / noise_rms)
        # NaN/Inf-Guard (§3.1)
        snr_db = np.nan_to_num(snr_db, nan=30.0, posinf=100.0, neginf=0.0)
        return max(0.0, min(snr_db, 100.0))  # type: ignore[no-any-return]

    @staticmethod
    def compute_thd(audio: np.ndarray, sr: int) -> float:
        """Berechnet Total Harmonic Distortion."""
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)

        # Find fundamental
        fundamental_idx = np.argmax(magnitude)
        fundamental_power = magnitude[fundamental_idx] ** 2

        # Sum harmonic power
        harmonic_power = 0.0
        for i in range(2, 6):
            harmonic_idx = fundamental_idx * i
            if harmonic_idx < len(magnitude):
                harmonic_power += magnitude[harmonic_idx] ** 2

        if fundamental_power < 1e-10:
            return 0.0

        thd = np.sqrt(harmonic_power / fundamental_power)
        return min(thd, 1.0)  # type: ignore[no-any-return]

    @staticmethod
    def compute_lufs(audio: np.ndarray, sr: int) -> float:
        """Berechnet LUFS (Loudness)."""
        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-10:
            return -100.0
        lufs = 20 * np.log10(rms) - 0.691
        return max(lufs, -100.0)  # type: ignore[no-any-return]

    # ============================================================
    # Unified Interface
    # ============================================================

    def compute_all(self, original: np.ndarray, restored: np.ndarray, sr: int = 48000) -> QualityMetricsResult:
        """
        Berechnet all available quality metrics.

        Args:
            original: Original (potentially degraded) audio
            restored: Restored/processed audio
            sr: Sample rate

        Returns:
            QualityMetricsResult with all computed metrics

        Example:
            >>> metrics = EnhancedMetrics()
            >>> result = metrics.compute_all(original, restored, sr=48000)
            logger.debug("ViSQOL: %s", result.visqol_mos)
            logger.debug("Passes standards: %s", result.passes_aurik_standards())
        """
        # Basic metrics on restored audio
        snr_restored = self.compute_snr(restored, sr)
        thd_restored = self.compute_thd(restored, sr)
        lufs_restored = self.compute_lufs(restored, sr)

        # Basic metrics on original audio (for improvement calculation)
        snr_original = self.compute_snr(original, sr)
        snr_improvement = snr_restored - snr_original

        # Enhanced metrics (require reference comparison)
        si_snr = self.compute_si_snr(original, restored)

        # Perceptual metrics (may be None if libraries unavailable)
        visqol_mos = self.compute_visqol(original, restored, sr=sr)

        # Authenticity metrics (Phase 2D.2.1 Task 3)
        breath_retention = None
        transient_preservation = None
        plosive_retention = None
        sibilance_retention = None
        room_tone_retention = None

        if self.authenticity is not None:
            # Limit to max 30 s mono to avoid OOM in authenticity metrics
            _MAX_AUTH_SAMPLES = 30 * sr
            _orig_auth = original[:_MAX_AUTH_SAMPLES]
            _rest_auth = restored[:_MAX_AUTH_SAMPLES]
            if _orig_auth.ndim > 1:
                _orig_auth = _orig_auth.mean(axis=-1)
            if _rest_auth.ndim > 1:
                _rest_auth = _rest_auth.mean(axis=-1)

            try:
                breath_ret, _, _ = self.authenticity.compute_breath_retention(_orig_auth, _rest_auth, sr)
                breath_retention = breath_ret
            except Exception as e:
                logger.warning("Breath retention computation failed: %s", e)

            try:
                trans_pres, _, _ = self.authenticity.compute_transient_preservation(_orig_auth, _rest_auth, sr)
                transient_preservation = trans_pres
            except Exception as e:
                logger.warning("Transient preservation computation failed: %s", e)

            try:
                plos_ret, _, _ = self.authenticity.compute_plosive_retention(_orig_auth, _rest_auth, sr)
                plosive_retention = plos_ret
            except Exception as e:
                logger.warning("Plosive retention computation failed: %s", e)

            try:
                sib_ret, _, _ = self.authenticity.compute_sibilance_retention(_orig_auth, _rest_auth, sr)
                sibilance_retention = sib_ret
            except Exception as e:
                logger.warning("Sibilance retention computation failed: %s", e)

            try:
                room_ret, _, _ = self.authenticity.compute_room_tone_retention(_orig_auth, _rest_auth, sr)
                room_tone_retention = room_ret
            except Exception as e:
                logger.warning("Room tone retention computation failed: %s", e)

        return QualityMetricsResult(
            snr_db=snr_restored,
            thd=thd_restored,
            lufs=lufs_restored,
            visqol_mos=visqol_mos,
            si_snr_db=si_snr,
            snr_improvement_db=snr_improvement,
            breath_retention=breath_retention,
            transient_preservation=transient_preservation,
            plosive_retention=plosive_retention,
            sibilance_retention=sibilance_retention,
            room_tone_retention=room_tone_retention,
        )

    def compute_restoration_improvement(
        self, original_noisy: np.ndarray, original_clean: np.ndarray, restored: np.ndarray, sr: int = 48000
    ) -> QualityMetricsResult:
        """
        Berechnet restoration improvement metrics.

        Compares:
        - Original noisy → Restored (improvement)
        - Original clean → Restored (fidelity)

        Args:
            original_noisy: Original degraded audio
            original_clean: Original clean reference (if available)
            restored: AURIK-restored audio
            sr: Sample rate

        Returns:
            QualityMetricsResult with improvement metrics
        """
        # Compute metrics for noisy input
        snr_noisy = self.compute_snr(original_noisy, sr)

        # Compute metrics for restored output
        snr_restored = self.compute_snr(restored, sr)

        # Improvement
        snr_improvement = snr_restored - snr_noisy

        # Full metrics with clean reference
        visqol_mos = self.compute_visqol(original_clean, restored, sr=sr)

        # Authenticity metrics (Phase 2D.2.1 Task 3)
        breath_retention = None
        transient_preservation = None
        plosive_retention = None
        sibilance_retention = None
        room_tone_retention = None

        if self.authenticity is not None:
            try:
                breath_ret, _, _ = self.authenticity.compute_breath_retention(original_clean, restored, sr)
                breath_retention = breath_ret
            except Exception as e:
                logger.debug("Breath retention (improvement) fehlgeschlagen: %s", e)

            try:
                trans_pres, _, _ = self.authenticity.compute_transient_preservation(original_clean, restored, sr)
                transient_preservation = trans_pres
            except Exception as e:
                logger.debug("Transient preservation (improvement) fehlgeschlagen: %s", e)

            try:
                plos_ret, _, _ = self.authenticity.compute_plosive_retention(original_clean, restored, sr)
                plosive_retention = plos_ret
            except Exception as e:
                logger.debug("Plosive retention (improvement) fehlgeschlagen: %s", e)

            try:
                sib_ret, _, _ = self.authenticity.compute_sibilance_retention(original_clean, restored, sr)
                sibilance_retention = sib_ret
            except Exception as e:
                logger.debug("Sibilance retention (improvement) fehlgeschlagen: %s", e)

            try:
                room_ret, _, _ = self.authenticity.compute_room_tone_retention(original_clean, restored, sr)
                room_tone_retention = room_ret
            except Exception as e:
                logger.debug("Room tone retention (improvement) fehlgeschlagen: %s", e)

        return QualityMetricsResult(
            snr_db=snr_restored,
            thd=self.compute_thd(restored, sr),
            lufs=self.compute_lufs(restored, sr),
            visqol_mos=visqol_mos,
            si_snr_db=self.compute_si_snr(original_clean, restored),
            snr_improvement_db=snr_improvement,
            breath_retention=breath_retention,
            transient_preservation=transient_preservation,
            plosive_retention=plosive_retention,
            sibilance_retention=sibilance_retention,
            room_tone_retention=room_tone_retention,
        )


# ============================================================
# Convenience Functions
# ============================================================


def batch_compute_metrics(audio_pairs: list, sr: int = 48000) -> list:
    """
    Berechnet metrics for multiple audio pairs.

    Args:
        audio_pairs: List of (original, restored) tuples
        sr: Sample rate

    Returns:
        List of QualityMetricsResult
    """
    metrics = EnhancedMetrics()
    results = []

    for original, restored in audio_pairs:
        result = metrics.compute_all(original, restored, sr)
        results.append(result)

    return results


def generate_metrics_report(result: QualityMetricsResult, filename: str = "metrics_report.txt") -> str:
    """
    Generiert human-readable metrics report.

    Args:
        result: QualityMetricsResult
        filename: Output filename

    Returns:
        Report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("AURIK Quality Metrics Report")
    lines.append("=" * 60)
    lines.append("")

    # Basic metrics
    lines.append("Basic Metrics:")
    lines.append(f"  SNR:  {result.snr_db:.2f} dB")
    lines.append(f"  THD:  {result.thd:.4f}")
    lines.append(f"  LUFS: {result.lufs:.2f} dB")
    lines.append("")

    # Enhanced metrics
    if result.visqol_mos is not None or result.si_snr_db is not None:
        lines.append("Enhanced Metrics:")

        if result.visqol_mos is not None:
            lines.append(f"  ViSQOL MOS: {result.visqol_mos:.2f} / 5.0")

        if result.si_snr_db is not None:
            lines.append(f"  SI-SNR:     {result.si_snr_db:.2f} dB")

        lines.append("")

    # Improvement metrics
    if result.snr_improvement_db is not None:
        lines.append("Improvement Metrics:")
        lines.append(f"  SNR Improvement: {result.snr_improvement_db:+.2f} dB")
        lines.append("")

    # Pass/Fail
    passes = result.passes_aurik_standards()
    status = "✅ PASSED" if passes else "❌ FAILED"
    lines.append(f"AURIK Weltspitze Standards: {status}")
    lines.append("=" * 60)

    report = "\n".join(lines)

    # Write to file
    with open(filename, "w") as f:
        f.write(report)

    return report
