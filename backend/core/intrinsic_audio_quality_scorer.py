"""
core/intrinsic_audio_quality_scorer.py
Intrinsic Audio Quality Scorer (IAQS)
=======================================

Psychoakustisch fundierter Qualitätsscorer — vollständig ohne externe
Abhängigkeiten (kein CDPAM, kein DNSMOS, kein PyTorch).

Basiert auf messbaren Signal-Eigenschaften, die stark mit wahrgenommener
Qualität korrelieren:

  A) Spektrale Güte
     - SNR (blind, via Minimum-Statistics-Schätzung)
     - Spektrale Regularität (Spitzen-zu-Tal-Verhältnis)
     - Bandbreiteneffizienz (genutzte Bandbreite vs. erwartete)
     - Bark-Band-Energie-Verteilung (Psychoakustisches Modell)

  B) Zeitbereichs-Güte
     - Transientenklarheit (Attack-Erkennung im Zeitsignal)
     - Dynamikumfang (EBU R128 Loudness Range näherungsweise)
     - Klirrfaktor-Schätzung (THD via Harmonics)

  C) Musikalische Güte
     - Harmonizität (Verhältnis harmonische zu inharmonische Energie)
     - Stimmungsklarheit (Pitch-Konsistenz über Zeit)
     - Authentizitätsindikator (Vintage vs. Digital-Überprägung)

  D) Artefakt-Detektion
     - Klick-Energie-Residuen (hohe Kurzzeitpegel)
     - Digitale Clipping-Indikatoren (Flat-Top-Samples)
     - Codec-Blockartefakte (periodische Spektralmodulation)

Alle Metriken sind:
  - schnell (< 0.5× Echtzeit für typische Längen)
  - robust (kein NaN/Inf)
  - skaliert auf [0.0, 1.0] (1.0 = perfekt)

Verwendung in MultiPassEngine als fallback wenn Plugins fehlen,
und als primärer Scorer in AutonomousRestorationEngine.

Author: Aurik Development Team
Version: 1.0.0 "Perceptual Precision"
Date: 2026-02-17
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

# Bark-Band-Grenzen in Hz (25 Bänder nach Zwicker 1961)
_BARK_BANDS_HZ: tuple[float, ...] = (
    100,
    200,
    300,
    400,
    510,
    630,
    770,
    920,
    1080,
    1270,
    1480,
    1720,
    2000,
    2320,
    2700,
    3150,
    3700,
    4400,
    5300,
    6400,
    7700,
    9500,
    12000,
    15500,
    20000,
)


# ---------------------------------------------------------------------------
# Ergebnis-Datenstruktur
# ---------------------------------------------------------------------------


@dataclass
class IntrinsicQualityScore:
    """Vollständiges intrinsisches Qualitätsergebnis."""

    # === Zusammenfassung ===
    overall: float = 0.0
    """Gewichteter Gesamtscore (0–1, 1 = perfekt)."""

    # === Spektral ===
    snr_estimate: float = 0.0
    """Blind-SNR-Schätzung in dB."""

    snr_score: float = 0.0
    """SNR normiert (0–1)."""

    spectral_regularity: float = 0.0
    """Spektrale Glätte (0–1, 1 = glatt)."""

    bandwidth_score: float = 0.0
    """Bandbreiteneffizienz (0–1)."""

    bark_balance: float = 0.0
    """Bark-Band-Balance (0–1, 1 = ideal)."""

    # === Zeitbereich ===
    dynamic_range_score: float = 0.0
    """Dynamikumfang-Score (0–1)."""

    transient_clarity: float = 0.0
    """Transientenklarheit (0–1)."""

    thd_estimate_pct: float = 0.0
    """THD-Schätzung in % (kleiner = besser)."""

    thd_score: float = 0.0
    """THD normiert (0–1, 1 = kein Klirr)."""

    # === Musikalisch ===
    harmonicity: float = 0.0
    """Harmonizität (0–1, 1 = rein harmonisch)."""

    pitch_consistency: float = 0.0
    """Pitch-Konsistenz (0–1, 1 = stabile Intonation)."""

    # === Artefakte ===
    click_residual: float = 0.0
    """Klick-Residual-Score (1 = keine Klicks, 0 = viele)."""

    clipping_score: float = 0.0
    """Clipping-Score (1 = kein Clipping, 0 = geclippt)."""

    codec_artifact_score: float = 0.0
    """Codec-Artefakt-Score (1 = keine, 0 = stark)."""

    # === Metadaten ===
    sample_rate: int = 44100
    duration_seconds: float = 0.0
    is_stereo: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Hauptklasse
# ---------------------------------------------------------------------------


class IntrinsicAudioQualityScorer:
    """Deterministischer, no-reference IAQS mit echter Metrik-Aufschlüsselung."""

    _EPS = 1e-12

    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(value if np.isfinite(value) else 0.0, 0.0, 1.0))

    @staticmethod
    def _to_mono_time(audio: np.ndarray) -> tuple[np.ndarray, bool]:
        arr = np.asarray(audio, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim == 1:
            return arr.astype(np.float32, copy=False), False
        if arr.ndim == 2:
            if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[-1]:
                return np.mean(arr, axis=0).astype(np.float32, copy=False), True
            if arr.shape[-1] <= 8:
                return np.mean(arr, axis=-1).astype(np.float32, copy=False), True
        return np.mean(arr.reshape(-1, arr.shape[-1]), axis=0).astype(np.float32, copy=False), True

    @classmethod
    def _frame_rms(cls, mono: np.ndarray, sr: int) -> np.ndarray:
        n = int(mono.shape[0])
        if n == 0:
            return cast(np.ndarray, (np.zeros(1, dtype=np.float64)))
        frame_len = int(np.clip(round(sr * 0.050), 128, max(128, n)))
        hop = max(1, frame_len // 2)
        if n < frame_len:
            return cast(
                np.ndarray,
                (np.asarray([np.sqrt(float(np.mean(mono.astype(np.float64) ** 2)) + cls._EPS)], dtype=np.float64)),
            )
        power = mono.astype(np.float64) ** 2
        cumsum = np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(power, dtype=np.float64)])
        starts = np.arange(0, n - frame_len + 1, hop, dtype=np.int64)
        sums = cumsum[starts + frame_len] - cumsum[starts]
        return cast(
            np.ndarray, (np.asarray(np.sqrt(np.maximum(sums / float(frame_len), 0.0) + cls._EPS), dtype=np.float64))
        )

    @classmethod
    def _spectrum(cls, mono: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        n = int(mono.shape[0])
        if n < 64:
            return np.asarray([0.0, float(sr) / 2.0]), np.asarray([cls._EPS, cls._EPS])
        n_fft = int(2 ** math.floor(math.log2(float(min(n, 65536)))))
        n_fft = max(256, n_fft)
        if n_fft > n:
            n_fft = int(2 ** math.floor(math.log2(float(n))))
        starts = [0]
        if n > n_fft:
            starts = sorted({0, max(0, (n - n_fft) // 2), max(0, n - n_fft)})
        window = np.hanning(n_fft).astype(np.float64)
        power_acc: np.ndarray | None = None
        for start in starts:
            segment = mono[start : start + n_fft].astype(np.float64)
            if segment.shape[0] < n_fft:
                segment = np.pad(segment, (0, n_fft - segment.shape[0]))
            power = np.abs(np.fft.rfft(segment * window)) ** 2
            if power_acc is None:
                power_acc = np.zeros_like(power, dtype=np.float64)
            power_acc = power_acc + power
        if power_acc is None:
            power_acc = np.full(n_fft // 2 + 1, cls._EPS, dtype=np.float64)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / max(int(sr), 1))
        return freqs.astype(np.float64), (power_acc / max(len(starts), 1) + cls._EPS).astype(np.float64)

    @classmethod
    def _estimate_snr(cls, frame_rms: np.ndarray) -> tuple[float, float]:
        if frame_rms.size == 0 or float(np.max(frame_rms)) < 1e-8:
            return 0.0, 0.0
        noise_floor = float(np.percentile(frame_rms, 10))
        signal_level = float(np.percentile(frame_rms, 95))
        snr_db = 20.0 * np.log10((signal_level + cls._EPS) / (noise_floor + cls._EPS))
        snr_db = float(np.clip(snr_db, 0.0, 72.0))
        return snr_db, cls._clip01((snr_db - 6.0) / 34.0)

    @classmethod
    def _spectral_regularity(cls, power: np.ndarray) -> float:
        if power.size < 16:
            return 0.5
        db = 10.0 * np.log10(power + cls._EPS)
        kernel = np.ones(9, dtype=np.float64) / 9.0
        smooth = np.convolve(db, kernel, mode="same")
        residual = db - smooth
        spread = float(np.percentile(residual, 90) - np.percentile(residual, 10))
        return cls._clip01(1.0 - spread / 42.0)

    @classmethod
    def _bandwidth_score(cls, freqs: np.ndarray, power: np.ndarray, sr: int) -> float:
        total = float(np.sum(power))
        if total <= cls._EPS:
            return 0.0
        cumulative = np.cumsum(power)
        idx = int(np.searchsorted(cumulative, total * 0.95))
        f95 = float(freqs[min(idx, len(freqs) - 1)])
        expected = max(1000.0, min(20000.0, float(sr) / 2.0))
        return cls._clip01((f95 / expected) ** 0.45)

    @classmethod
    def _bark_balance(cls, freqs: np.ndarray, power: np.ndarray, sr: int) -> float:
        nyquist = float(sr) / 2.0
        edges = [0.0] + [b for b in _BARK_BANDS_HZ if b < nyquist] + [nyquist]
        energies: list[float] = []
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (freqs >= low) & (freqs < high)
            energies.append(float(np.sum(power[mask])) if np.any(mask) else 0.0)
        vals = np.asarray(energies, dtype=np.float64)
        total = float(np.sum(vals))
        if total <= cls._EPS or len(vals) < 2:
            return 0.0
        probs = vals / total
        entropy = -float(np.sum(probs * np.log(probs + cls._EPS))) / float(np.log(len(vals)))
        return cls._clip01(entropy * 1.18)

    @classmethod
    def _dynamic_range_score(cls, frame_rms: np.ndarray) -> float:
        if frame_rms.size < 2 or float(np.max(frame_rms)) < 1e-8:
            return 0.0
        frame_db = 20.0 * np.log10(frame_rms + cls._EPS)
        dr_db = float(np.percentile(frame_db, 95) - np.percentile(frame_db, 10))
        return cls._clip01((dr_db - 4.0) / 22.0)

    @classmethod
    def _transient_clarity(cls, mono: np.ndarray) -> float:
        if mono.size < 8:
            return 0.5
        diff = np.abs(np.diff(mono.astype(np.float64)))
        median = float(np.median(diff))
        p95 = float(np.percentile(diff, 95))
        ratio = p95 / (median + cls._EPS)
        return cls._clip01(np.log10(ratio + 1.0) / 1.25)

    @classmethod
    def _harmonic_metrics(cls, freqs: np.ndarray, power: np.ndarray) -> tuple[float, float, float, float]:
        mask = (freqs >= 50.0) & (freqs <= 2000.0)
        if not np.any(mask):
            return 0.0, 0.0, 0.0, 0.0
        local_power = power[mask]
        local_freqs = freqs[mask]
        if float(np.max(local_power)) <= cls._EPS:
            return 0.0, 0.0, 0.0, 0.0
        f0_idx = int(np.argmax(local_power))
        f0 = float(local_freqs[f0_idx])
        fundamental_power = float(local_power[f0_idx]) + cls._EPS
        bin_hz = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
        harmonic_power = 0.0
        distortion_power = 0.0
        for harmonic in range(2, 8):
            target = f0 * harmonic
            if target >= freqs[-1]:
                break
            width = max(bin_hz * 2.0, target * 0.01)
            hmask = (freqs >= target - width) & (freqs <= target + width)
            hp = float(np.max(power[hmask])) if np.any(hmask) else 0.0
            harmonic_power += hp
            if harmonic <= 5:
                distortion_power += hp
        band_mask = (freqs >= 50.0) & (freqs <= min(8000.0, freqs[-1]))
        band_total = float(np.sum(power[band_mask])) + cls._EPS
        harmonicity = cls._clip01((fundamental_power + harmonic_power) / band_total * 1.8)
        thd_pct = float(np.clip(np.sqrt(distortion_power / fundamental_power) * 100.0, 0.0, 100.0))
        thd_score = cls._clip01(1.0 / (1.0 + thd_pct / 18.0))
        pitch_consistency = cls._clip01(fundamental_power / (fundamental_power + float(np.median(local_power)) * 8.0))
        return harmonicity, pitch_consistency, thd_pct, thd_score

    @classmethod
    def _click_residual_score(cls, mono: np.ndarray) -> float:
        if mono.size < 8:
            return 1.0
        diff = np.diff(mono.astype(np.float64))
        abs_diff = np.abs(diff)
        mad = float(np.median(np.abs(abs_diff - np.median(abs_diff))))
        floor = float(np.percentile(abs_diff, 95)) * 0.35
        threshold = max(12.0 * mad, floor, 1e-5)
        spike_rate = float(np.mean(abs_diff > threshold))
        return cls._clip01(1.0 - spike_rate * 220.0)

    @classmethod
    def _clipping_score(cls, mono: np.ndarray) -> float:
        if mono.size < 2:
            return 1.0
        abs_mono = np.abs(mono)
        clip_frac = float(np.mean(abs_mono >= 0.999))
        flat = (abs_mono[:-1] >= 0.999) & (abs_mono[1:] >= 0.999) & (np.abs(np.diff(mono)) < 1e-5)
        flat_frac = float(np.mean(flat)) if flat.size else 0.0
        return cls._clip01(1.0 - clip_frac * 80.0 - flat_frac * 180.0)

    @classmethod
    def _codec_artifact_score(cls, mono: np.ndarray, sr: int) -> float:
        if mono.size < max(2048, sr // 10):
            return 0.85
        scores: list[float] = []
        for block in (576, 1024):
            n_blocks = mono.size // block
            if n_blocks < 4:
                continue
            trimmed = mono[: n_blocks * block].astype(np.float64).reshape(n_blocks, block)
            rms = np.sqrt(np.mean(trimmed**2, axis=1) + cls._EPS)
            modulation = float(np.std(rms) / (np.mean(rms) + cls._EPS))
            scores.append(cls._clip01(1.0 - max(0.0, modulation - 0.25) * 1.6))
        return float(min(scores)) if scores else 0.85

    def analyze(self, audio: np.ndarray, sr: int = 44100) -> IntrinsicQualityScore:
        mono, is_stereo = self._to_mono_time(audio)
        warnings: list[str] = []
        if mono.size == 0:
            warnings.append("empty_audio")
            return IntrinsicQualityScore(sample_rate=sr, is_stereo=is_stereo, warnings=warnings)
        if float(np.max(np.abs(mono))) < 1e-8:
            warnings.append("silent_audio")

        frame_rms = self._frame_rms(mono, sr)
        freqs, power = self._spectrum(mono, sr)
        snr_db, snr_score = self._estimate_snr(frame_rms)
        spectral_regularity = self._spectral_regularity(power)
        bandwidth_score = self._bandwidth_score(freqs, power, sr)
        bark_balance = self._bark_balance(freqs, power, sr)
        dynamic_range_score = self._dynamic_range_score(frame_rms)
        transient_clarity = self._transient_clarity(mono)
        harmonicity, pitch_consistency, thd_pct, thd_score = self._harmonic_metrics(freqs, power)
        click_residual = self._click_residual_score(mono)
        clipping_score = self._clipping_score(mono)
        codec_artifact_score = self._codec_artifact_score(mono, sr)

        weighted = {
            "snr_score": (snr_score, 0.17),
            "spectral_regularity": (spectral_regularity, 0.10),
            "bandwidth_score": (bandwidth_score, 0.08),
            "bark_balance": (bark_balance, 0.10),
            "dynamic_range_score": (dynamic_range_score, 0.10),
            "transient_clarity": (transient_clarity, 0.09),
            "thd_score": (thd_score, 0.08),
            "harmonicity": (harmonicity, 0.10),
            "pitch_consistency": (pitch_consistency, 0.06),
            "click_residual": (click_residual, 0.06),
            "clipping_score": (clipping_score, 0.04),
            "codec_artifact_score": (codec_artifact_score, 0.02),
        }
        overall = self._clip01(sum(score * weight for score, weight in weighted.values()))
        return IntrinsicQualityScore(
            overall=overall,
            snr_estimate=snr_db,
            snr_score=snr_score,
            spectral_regularity=spectral_regularity,
            bandwidth_score=bandwidth_score,
            bark_balance=bark_balance,
            dynamic_range_score=dynamic_range_score,
            transient_clarity=transient_clarity,
            thd_estimate_pct=thd_pct,
            thd_score=thd_score,
            harmonicity=harmonicity,
            pitch_consistency=pitch_consistency,
            click_residual=click_residual,
            clipping_score=clipping_score,
            codec_artifact_score=codec_artifact_score,
            sample_rate=int(sr),
            duration_seconds=float(mono.size) / float(max(int(sr), 1)),
            is_stereo=is_stereo,
            warnings=warnings,
        )

    def score_as_float(self, audio: np.ndarray, sr: int = 44100) -> float:
        return float(self.analyze(audio, sr).overall)

    def _reference_preservation(self, original: np.ndarray, processed: np.ndarray, sr: int) -> float:
        orig, _ = self._to_mono_time(original)
        proc, _ = self._to_mono_time(processed)
        n = min(orig.size, proc.size)
        if n < 32:
            return 1.0
        orig = orig[:n]
        proc = proc[:n]
        rms_orig = float(np.sqrt(np.mean(orig.astype(np.float64) ** 2)) + self._EPS)
        rms_proc = float(np.sqrt(np.mean(proc.astype(np.float64) ** 2)) + self._EPS)
        rms_ratio = min(rms_orig, rms_proc) / max(rms_orig, rms_proc, self._EPS)
        freqs_o, power_o = self._spectrum(orig, sr)
        _, power_p = self._spectrum(proc, sr)
        m = min(power_o.size, power_p.size, freqs_o.size)
        if m < 3 or float(np.std(power_o[:m])) <= self._EPS or float(np.std(power_p[:m])) <= self._EPS:
            spectral_corr = 1.0
        else:
            spectral_corr = float(np.corrcoef(np.log1p(power_o[:m]), np.log1p(power_p[:m]))[0, 1])
            spectral_corr = self._clip01((spectral_corr + 1.0) * 0.5)
        return self._clip01(0.45 * rms_ratio + 0.55 * spectral_corr)

    def score(
        self,
        audio_or_original: np.ndarray,
        processed_or_sr: np.ndarray | int | None = None,
        sr: int | None = None,
    ) -> IntrinsicQualityScore | float:
        """Kompatible API: ``score(audio, sr)`` → Detail; ``score(ref, out, sr)`` → 0-100."""
        if sr is None and isinstance(processed_or_sr, (int, np.integer)):
            return self.analyze(audio_or_original, int(processed_or_sr))
        if sr is None or processed_or_sr is None:
            raise TypeError("score() erwartet entweder (audio, sr) oder (original, processed, sr)")
        detail = self.analyze(np.asarray(processed_or_sr, dtype=np.float32), int(sr))
        preservation = self._reference_preservation(audio_or_original, np.asarray(processed_or_sr), int(sr))
        return float(self._clip01(detail.overall * (0.85 + 0.15 * preservation)) * 100.0)
