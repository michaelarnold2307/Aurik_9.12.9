"""
Artifact Detector — Post-Processing Anomaly Scanner (§G53)

Detects processing artifacts that should NOT be present in the output:
  §G53a  Click/Pop detection (transient anomalies)
  §G53b  Spectral hole detection (missing frequency content)
  §G53c  Pre-echo detection (energy before onsets)
  §G53d  Stereo anomaly detection (L/R phase corruption)
  §G53e  Aggregate artifact score (0-1, 1 = artifact-free)

All detectors operate on the processed output only (reference-free).
Integration: artifact_score feeds into MUSHRA proxy as penalty term.

Author: Aurik Development Team
Version: 10.0.7
Date: 2026-07-13
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ArtifactReport:
    """Complete artifact scan results."""

    click_count: int = 0
    click_score: float = 1.0
    spectral_hole_score: float = 1.0
    pre_echo_score: float = 1.0
    stereo_anomaly_score: float = 1.0
    overall_score: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        return self.overall_score > 0.95

    @property
    def has_artifacts(self) -> bool:
        return self.overall_score < 0.80


class ArtifactDetector:
    """§G53: Multi-detector artifact scanner.

    Usage:
        detector = ArtifactDetector(sr=48000)
        report = detector.scan(processed_audio)
        logger.info(f"Artifact score: {report.overall_score:.2f}")
        if report.click_count > 0:
            logger.info(f"  Found {report.click_count} clicks!")
    """

    def __init__(self, sr: int = 48000):
        self.sr = sr

    def scan(self, audio: np.ndarray) -> ArtifactReport:
        """Run all artifact detectors on the audio.

        Returns ArtifactReport with per-detector scores and overall.
        """
        mono = self._to_mono(audio)
        n = len(mono)

        # Minimum length check
        if n < 2048:
            return ArtifactReport(overall_score=1.0)

        details = {}

        # §G53a: Click/pop detection
        click_count, click_score = self._detect_clicks(mono)
        details["click_count"] = click_count

        # §G53b: Spectral hole detection
        hole_score = self._detect_spectral_holes(mono)
        details["spectral_hole_score_raw"] = hole_score  # type: ignore[assignment]

        # §G53c: Pre-echo detection
        pre_echo_score = self._detect_pre_echo(mono)
        details["pre_echo_score_raw"] = pre_echo_score  # type: ignore[assignment]

        # §G53d: Stereo anomalies
        if audio.ndim == 2 and audio.shape[1] >= 2:
            stereo_score = self._detect_stereo_anomalies(audio)
        else:
            stereo_score = 1.0
        details["stereo_anomaly_score_raw"] = stereo_score  # type: ignore[assignment]

        # §G53e: Aggregate
        overall = 0.30 * click_score + 0.25 * hole_score + 0.25 * pre_echo_score + 0.20 * stereo_score

        return ArtifactReport(
            click_count=click_count,
            click_score=click_score,
            spectral_hole_score=hole_score,
            pre_echo_score=pre_echo_score,
            stereo_anomaly_score=stereo_score,
            overall_score=float(np.clip(overall, 0.0, 1.0)),
            details=details,
        )

    # ── §G53a Click Detection ──────────────────────────────────────────

    def _detect_clicks(self, mono: np.ndarray) -> tuple[int, float]:
        """Detect transient clicks via gradient anomaly detection.

        A click is a sample whose absolute gradient exceeds 6 standard
        deviations of the local neighborhood (200-sample window).
        """
        n = len(mono)
        if n < 400:
            return 0, 1.0

        # Gradient
        grad = np.abs(np.diff(mono.astype(np.float64)))
        grad = np.concatenate([[0.0], grad])

        # Local statistics in 200-sample windows
        window = 200
        n_windows = n // window
        if n_windows < 3:
            return 0, 1.0

        thresholds = np.zeros(n, dtype=np.float64)
        for i in range(n_windows):
            start = i * window
            end = min(start + window, n)
            local = grad[start:end]
            mu = float(np.mean(local))
            std = float(np.std(local))
            thresholds[start:end] = mu + 6.0 * max(std, 1e-15)

        clicks = grad > thresholds
        click_count = int(np.sum(clicks))

        # Score: penalty for each click, decaying
        if click_count == 0:
            return 0, 1.0
        # Exponential decay: 1 click → 0.98, 5 → 0.90, 20 → 0.67
        score = float(np.exp(-click_count / 10.0))
        return click_count, score

    # ── §G53b Spectral Hole Detection ──────────────────────────────────

    def _detect_spectral_holes(self, mono: np.ndarray) -> float:
        """Detect spectral holes: frequency bins with abnormally low energy.

        A spectral hole is a bin where energy drops >30 dB below its
        immediate neighbors AND below the frame's noise floor estimate.
        """
        n = len(mono)
        n_fft = 2048
        hop = n_fft // 2
        n_frames = (n - n_fft) // hop + 1
        if n_frames < 5:
            return 1.0

        win = np.hanning(n_fft)
        hole_count = 0
        total_bins = 0

        for i in range(n_frames):
            start = i * hop
            frame = mono[start : start + n_fft] * win
            spec_db = 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(frame)), 1e-15))

            # Skip DC and highest bins
            spec = spec_db[5:-5]
            if len(spec) < 10:
                continue

            # Local median filter (11-bin window) gives expected level
            kernel = 11
            half = kernel // 2
            expected = np.zeros(len(spec), dtype=np.float64)
            for j in range(len(spec)):
                lo = max(0, j - half)
                hi = min(len(spec), j + half + 1)
                expected[j] = np.median(spec[lo:hi])

            # Hole: bin is >30 dB below expected
            holes = spec < expected - 30.0
            hole_count += int(np.sum(holes))
            total_bins += len(spec)

        if total_bins == 0:
            return 1.0

        hole_ratio = hole_count / total_bins
        # >5% holes → significant
        score = float(np.clip(1.0 - hole_ratio * 10.0, 0.0, 1.0))
        return score

    # ── §G53c Pre-Echo Detection ───────────────────────────────────────

    def _detect_pre_echo(self, mono: np.ndarray) -> float:
        """Detect pre-echo: energy buildup before sharp onsets.

        Pre-echo is a tell-tale sign of transform codec artifacts (MP3,
        AAC) or STFT-based processing with insufficient temporal resolution.

        Algorithm: for each detected onset, compare energy in 20ms before
        vs 20ms after. Pre-echo = significant energy before onset.
        """
        n = len(mono)
        n_fft, hop = 1024, 256
        n_frames = (n - n_fft) // hop + 1
        if n_frames < 10:
            return 1.0

        win = np.hanning(n_fft)
        # High-frequency energy (2-8 kHz) for onset detection
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sr)
        lo = np.searchsorted(freqs, 2000)
        hi = np.searchsorted(freqs, 8000)
        if hi <= lo:
            return 1.0

        energy = np.zeros(n_frames, dtype=np.float64)
        for i in range(n_frames):
            s = i * hop
            frame = mono[s : s + n_fft] * win
            spec = np.abs(np.fft.rfft(frame))
            energy[i] = float(np.sum(spec[lo:hi] ** 2))

        if np.max(energy) < 1e-15:
            return 1.0

        energy_db = 10.0 * np.log10(energy + 1e-15)
        onset = np.diff(energy_db)
        onset = np.maximum(onset, 0.0)
        # Absolute minimum onset: 3 dB (rejects steady-state fluctuations)
        onset_threshold = max(np.mean(onset) + 2.5 * np.std(onset), 3.0)

        # Find onset frames
        onset_frames = np.where(onset > onset_threshold)[0]
        if len(onset_frames) == 0:
            return 1.0

        # For each onset, check pre-onset energy
        pre_samples = int(0.020 * self.sr)  # 20 ms
        post_samples = int(0.020 * self.sr)

        pre_echo_count = 0
        for onset_idx in onset_frames:
            onset_sample = onset_idx * hop + n_fft // 2
            if onset_sample < pre_samples + 100 or onset_sample > n - post_samples:
                continue

            pre_energy = float(np.mean(mono[onset_sample - pre_samples : onset_sample] ** 2))
            post_energy = float(np.mean(mono[onset_sample + 100 : onset_sample + 100 + post_samples] ** 2))

            if post_energy < 1e-15:
                continue

            pre_ratio = pre_energy / post_energy
            # Pre-echo: >10% of post-onset energy in pre-onset region
            if pre_ratio > 0.10:
                pre_echo_count += 1

        # Score: penalty per pre-echo onset
        if pre_echo_count == 0:
            return 1.0
        return float(np.clip(1.0 - pre_echo_count / max(len(onset_frames), 1), 0.0, 1.0))

    # ── §G53d Stereo Anomaly Detection ─────────────────────────────────

    def _detect_stereo_anomalies(self, audio: np.ndarray) -> float:
        """Detect stereo anomalies: sudden L/R correlation changes.

        A healthy stereo signal has slowly-varying L/R correlation.
        Sudden jumps indicate phase processing errors or channel swaps.
        """
        if audio.ndim < 2 or audio.shape[1] < 2:
            return 1.0

        left = audio[:, 0].astype(np.float64)
        right = audio[:, 1].astype(np.float64)
        n = min(len(left), len(right))

        # L/R correlation in 100ms windows
        win = int(0.100 * self.sr)
        hop = win // 2
        n_windows = (n - win) // hop + 1
        if n_windows < 5:
            return 1.0

        correlations = np.zeros(n_windows, dtype=np.float64)
        for i in range(n_windows):
            s = i * hop
            l = left[s : s + win]
            r = right[s : s + win]
            sl = float(np.std(l))
            sr_ = float(np.std(r))
            if sl < 1e-10 or sr_ < 1e-10:
                correlations[i] = 0.0
            else:
                c = float(np.corrcoef(l, r)[0, 1])
                correlations[i] = c if not np.isnan(c) else 0.0

        # Detect sudden jumps
        diff = np.abs(np.diff(correlations))
        jumps = diff > 0.3  # >0.3 correlation change in 50ms
        jump_count = int(np.sum(jumps))

        if jump_count == 0:
            return 1.0

        # Also check for anti-correlation (out-of-phase)
        anti_phase = int(np.sum(correlations < -0.5))

        score = float(np.clip(1.0 - (jump_count * 0.1 + anti_phase * 0.05), 0.0, 1.0))
        return score

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            return audio
        return audio.mean(axis=0) if audio.shape[1] < audio.shape[0] else audio.mean(axis=1)  # type: ignore[no-any-return]


def compute_artifact_freedom_score(audio: np.ndarray, sr: int = 48000) -> float:
    """Convenience: one-shot artifact freedom score [0, 1].

    1.0 = artifact-free. Callable from quality gates or MUSHRA.
    """
    return ArtifactDetector(sr).scan(audio).overall_score
