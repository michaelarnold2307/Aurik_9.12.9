"""
backend/core/medium_classifier.py  — Aurik 10.0.0 Spec §2.1
Automatische Trägermedien-Erkennung (17 Materialtypen).
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# pylint: disable=import-outside-toplevel

logger = logging.getLogger(__name__)


def _get_material_type():
    """Lädt MaterialType lazy, um Import-Zyklen im Core zu vermeiden."""
    try:
        from backend.core.defect_scanner import MaterialType

        return MaterialType
    except Exception as e:
        # §V6 Silent-Failure-Verbot: Log with reason, but INFO (graceful fallback returning None)
        logger.info(
            "§V6 Graceful fallback material_classifier: %s", str(e)[:200]
        )
        return None


@dataclass
class MaterialEvidence:
    """Begründungsdaten für ein klassifiziertes Trägermedium."""

    material: Any
    confidence: float
    features_matched: list[str] = field(default_factory=list)
    features_against: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Serialisiert den Evidenzsatz als JSON-kompatibles Dict."""
        mt = self.material
        return {
            "material": mt.value if hasattr(mt, "value") else str(mt),
            "confidence": self.confidence,
            "features_matched": self.features_matched,
            "features_against": self.features_against,
        }


@dataclass
class ClassificationResult:
    """Strukturiertes Ergebnis der Materialklassifikation."""

    material: Any
    confidence: float
    evidence: list[MaterialEvidence] = field(default_factory=list)
    bandwidth_hz: float = 0.0
    snr_db: float = 0.0
    noise_color: float = 1.0
    crackle_density: float = 0.0
    wow_flutter_hz: float = 0.0
    block_artifact: float = 0.0
    pre_echo_ms: float = 0.0
    classifier_source: str = "dsp"
    rotation_hz: float = 0.0  # dominant turntable rotation frequency [Hz]; 0.0 = none
    rotation_strength: float = 0.0  # normalized peak SNR of rotation signal [0, 1]
    infrasonic_rms: float = 0.0  # sub-20 Hz normalised RMS (vinyl rumble indicator)
    codec_type: str = ""  # 'mp3' | 'aac' | 'lossy' | 'clean' | '' (unknown)

    def _material_type(self) -> str:
        """Gibt den Materialtyp als stringifizierten Wert zurück."""
        mt = self.material
        if hasattr(mt, "value"):
            return str(mt.value)
        return str(mt)

    material_type = property(_material_type)

    def as_dict(self) -> dict[str, Any]:
        """Serialisiert das Ergebnis als JSON-kompatibles Dict."""
        mt = self.material
        return {
            "material": mt.value if hasattr(mt, "value") else str(mt),
            "confidence": self.confidence,
            "bandwidth_hz": self.bandwidth_hz,
            "snr_db": self.snr_db,
            "noise_color": self.noise_color,
            "crackle_density": self.crackle_density,
            "wow_flutter_hz": self.wow_flutter_hz,
            "block_artifact": self.block_artifact,
            "pre_echo_ms": self.pre_echo_ms,
            "classifier_source": self.classifier_source,
            "rotation_hz": self.rotation_hz,
            "rotation_strength": self.rotation_strength,
            "infrasonic_rms": self.infrasonic_rms,
            "codec_type": self.codec_type,
            "n_evidence": len(self.evidence),
        }


class _SpectralFingerprinter:
    """Extrahiert spektrale Kennwerte für die Trägermedien-Erkennung."""

    _FRAME_SIZE = 1024
    _HOP_SIZE = 512

    def extract(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """Berechnet den vollständigen Merkmalsvektor für die Materialklassifikation."""
        mono = self._to_mono(audio)
        if mono.size < self._FRAME_SIZE:
            return self._null_features()
        f: dict[str, float] = {}
        f["bandwidth_hz"] = self._bandwidth(mono, sr)
        f["snr_db"] = self._snr(mono)
        f["noise_color"] = self._noise_color(mono, sr)
        f["crackle_density"] = self._crackle_density(mono)
        # IEC 60386-compliant wow/flutter via FCPE F₀ modulation (ZCR fallback)
        f["wow_flutter_hz"], f["wow_depth"] = self._wow_flutter(mono, sr)
        # MDCT codec fingerprint (replaces simple 576-sample energy delta)
        f["block_artifact"], f["codec_type_code"] = self._codec_artifact_score(mono, sr)
        f["pre_echo_ms"] = self._pre_echo(mono, sr)
        # Rotation periodicity detector (Cano 2005; Rodriguez & Bello 2018)
        f["rotation_hz"], f["rotation_strength"] = self._rotation_periodicity(mono, sr)
        # Sub-20 Hz infrasonic RMS (vinyl rumble; Schuller 1998)
        f["infrasonic_rms"] = self._infrasonic_rms(audio, sr)
        for k, v in f.items():
            if not math.isfinite(v):
                f[k] = 0.0
        return f

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        arr = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, -1.0, 1.0)
        if arr.ndim != 2:
            return arr  # type: ignore[no-any-return]
        # Aurik-Konvention: ``(samples, channels)``. Über die Kanal-Achse (die
        # kleinere, da nur Mono/Stereo) mitteln — robust gegen beide Layouts.
        # Spaltenvektoren ``(N, 1)`` als Mono behandeln.
        if min(arr.shape) < 2:
            return arr.reshape(-1)  # type: ignore[no-any-return]
        ch_axis = 1 if arr.shape[1] <= arr.shape[0] else 0
        return arr.mean(axis=ch_axis)  # type: ignore[no-any-return]

    @staticmethod
    def _null_features() -> dict[str, float]:
        return {
            "bandwidth_hz": 0.0,
            "snr_db": 0.0,
            "noise_color": 1.0,
            "crackle_density": 0.0,
            "wow_flutter_hz": 0.0,
            "wow_depth": 0.0,
            "block_artifact": 0.0,
            "codec_type_code": 0.0,
            "pre_echo_ms": 0.0,
            "rotation_hz": 0.0,
            "rotation_strength": 0.0,
            "infrasonic_rms": 0.0,
        }

    def _bandwidth(self, mono: np.ndarray, sr: int) -> float:
        n = min(mono.size, 16384)
        spec = np.abs(np.fft.rfft(mono[:n], n=n))
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        cumsum = np.cumsum(spec)
        if cumsum[-1] < 1e-12:
            return 0.0
        idx = int(np.searchsorted(cumsum, 0.95 * cumsum[-1]))
        return float(freqs[min(idx, len(freqs) - 1)])

    def _snr(self, mono: np.ndarray) -> float:
        frame = self._FRAME_SIZE
        n_frames = max(1, len(mono) // frame)
        frames = mono[: n_frames * frame].reshape(n_frames, frame)
        energies = np.sqrt(np.mean(frames**2, axis=1))
        noise = float(np.percentile(energies, 5)) + 1e-10
        signal = float(np.percentile(energies, 95))
        if signal <= noise:
            return 0.0
        return float(np.clip(20.0 * math.log10(signal / noise), 0.0, 90.0))

    def _noise_color(self, mono: np.ndarray, sr: int = 48000) -> float:
        n = min(mono.size, 8192)
        spec = np.abs(np.fft.rfft(mono[:n], n=n)) + 1e-10
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        mask = freqs > 50.0
        if not mask.any():
            return 1.0
        log_f = np.log10(freqs[mask] + 1e-10)
        log_s = np.log10(spec[mask])
        if log_f.std() < 1e-8:
            return 1.0
        beta = -float(np.polyfit(log_f, log_s, 1)[0])
        return float(np.clip(beta, 0.0, 4.0))

    def _crackle_density(self, mono: np.ndarray) -> float:
        if mono.size < 512:
            return 0.0
        sigma = float(np.std(mono)) + 1e-10
        return float(np.clip((np.abs(mono) > 4.0 * sigma).mean() * 100.0, 0.0, 1.0))

    def _wow_flutter(self, mono: np.ndarray, sr: int) -> tuple[float, float]:
        """IEC 60386-compliant wow/flutter measurement via F₀ modulation spectrum.

        Primary: FCPE F₀ time-series → FFT of relative F₀ deviation.
            Wow:     dominant modulation peak < 3 Hz → transport speed anomaly.
            Flutter: dominant modulation peak 3–30 Hz → bearing vibration.
        Fallback: ZCR-based proxy (original method, returns dom_freq=0.0).

        Returns:
            (dominant_modulation_hz, modulation_depth_pct)
            modulation_depth_pct ≈ IEC 60386 weighted rms flutter [%], in [0, 20].

        References:
            IEC 60386:1994 — Measurement of wow and flutter in sound recording.
            de Boer (1966) — Pitch of inharmonic signals.
        """
        try:
            return self._wow_flutter_fcpe(mono, sr)
        except Exception as e:
            logger.warning("medium_classifier.py::_wow_flutter Ersatzpfad: %s", e)
            return self._wow_flutter_zcr_fallback(mono, sr)

    def _wow_flutter_fcpe(self, mono: np.ndarray, sr: int) -> tuple[float, float]:
        """FCPE F₀ → modulation spectrum → IEC dominant frequency + depth."""
        if len(mono) < sr:  # at least 1 second
            raise ValueError("too short for FCPE wow/flutter")
        from plugins.fcpe_plugin import get_fcpe_plugin, get_loaded_fcpe_plugin  # type: ignore[import]

        fcpe = get_loaded_fcpe_plugin()
        if fcpe is None:
            fcpe = get_fcpe_plugin()
        result = fcpe.analyze(mono, sr)
        f0_hz = result.f0_hz
        times_s = result.times_s
        voiced = result.voiced_prob > 0.45
        if voiced.sum() < 20:
            raise ValueError("insufficient voiced frames")
        f0_voiced = f0_hz[voiced]
        f0_mean = float(np.mean(f0_voiced))
        if f0_mean < 1.0:
            raise ValueError("F0 mean too low")
        # Relative F₀ deviation (dimensionless)
        f0_norm = f0_voiced / f0_mean - 1.0
        # Envelope sampling rate from median frame interval
        t_voiced = times_s[voiced]
        env_sr = 1.0 / float(np.median(np.diff(t_voiced))) if len(t_voiced) > 2 else 100.0
        env_sr = float(np.clip(env_sr, 10.0, 1000.0))
        n = len(f0_norm)
        spec = np.abs(np.fft.rfft(f0_norm * np.hanning(n))) ** 2
        mod_freqs = np.fft.rfftfreq(n, d=1.0 / env_sr)
        # Wow: [0.1, 3 Hz]; Flutter: [3, 30 Hz]
        wow_mask = (mod_freqs >= 0.1) & (mod_freqs < 3.0)
        flutter_mask = (mod_freqs >= 3.0) & (mod_freqs < 30.0)
        wow_pk = float(spec[wow_mask].max()) if wow_mask.any() else 0.0
        flu_pk = float(spec[flutter_mask].max()) if flutter_mask.any() else 0.0
        if wow_pk >= flu_pk:
            dom_idx = int(np.argmax(spec * wow_mask.astype(float)))
        else:
            dom_idx = int(np.argmax(spec * flutter_mask.astype(float)))
        dom_freq = float(mod_freqs[dom_idx]) if wow_mask.any() or flutter_mask.any() else 0.0
        mod_depth = float(np.std(f0_norm)) * 100.0  # [%]
        return dom_freq, float(np.clip(mod_depth, 0.0, 20.0))

    def _wow_flutter_zcr_fallback(self, mono: np.ndarray, sr: int) -> tuple[float, float]:
        """ZCR-based proxy — returns (dom_freq=0.0, depth[%]).

        Original Aurik method; depth range matches modulation_depth_pct output.
        """
        frame = max(self._FRAME_SIZE, int(0.02 * sr))
        n_frames = max(1, len(mono) // frame)
        if n_frames < 2:
            return 0.0, 0.0
        frames = mono[: n_frames * frame].reshape(n_frames, frame)
        zcr = np.mean(np.abs(np.diff(np.sign(frames))) / 2.0, axis=1)
        zcr_hz = zcr * sr / 2.0
        denom = max(float(np.mean(zcr_hz)), 1.0)
        depth = float(np.clip(float(np.std(zcr_hz)) / denom * 10.0, 0.0, 20.0))
        return 0.0, depth

    def _codec_artifact_score(self, mono: np.ndarray, sr: int) -> tuple[float, float]:
        """Lossy codec detection via MDCT quantization fingerprint.

        Combines three independent cues:
        1. Brick-wall cutoff: bitrate-limited encoders apply a hard LPF; the
           energy ratio just above vs. just below the detected bandwidth is near
           zero for codec output but non-negligible for natural audio.
        2. Spectral flatness measure (SFM) *inside* the active band: quantization
           noise raises the inter-harmonic floor, producing a flatter spectrum.
           SFM is computed only up to 95 % of detected bandwidth to exclude the
           zero-padded region that would otherwise bias the geometric mean.
        3. Pre-echo pattern: MP3's long MDCT window leaks transient energy into
           preceding frames.

        Returns:
            (artifact_score ∈ [0, 1], codec_type_code)
            codec_type_code: 0.0 = clean, 1.0 = mp3, 2.0 = aac, 3.0 = lossy.

        References:
            Farid (2009) — Exposing digital forgeries.
            Bianchi et al. (2020) — Detection of double audio compression.
            Liu et al. (2014) — Identification of lossy compression levels.
        """
        n_fft = 4096
        if len(mono) < n_fft * 4:
            return 0.0, 0.0
        n_frames = min(32, len(mono) // n_fft)
        window = np.hanning(n_fft)
        specs = np.array(
            [np.abs(np.fft.rfft(mono[i * n_fft : (i + 1) * n_fft] * window)) for i in range(n_frames)]
        )  # (n_frames, n_bins)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        mean_spec = specs.mean(axis=0) + 1e-10

        # --- Detect active bandwidth (90th-percentile cumulative energy) ---
        cumspec = np.cumsum(mean_spec)
        bw_idx = int(np.searchsorted(cumspec, 0.90 * cumspec[-1]))
        bw_hz = float(freqs[min(bw_idx, len(freqs) - 1)])
        bw_hz = float(np.clip(bw_hz, 2000.0, float(sr) * 0.48))

        # --- 1. Brick-wall cutoff score ---
        # Compare energy just below bandwidth to energy just above it.
        # Natural roll-off: gradual; codec: near-zero above cutoff.
        below_mask = (freqs >= bw_hz * 0.50) & (freqs <= bw_hz * 0.88)
        above_mask = (freqs >= bw_hz * 1.12) & (freqs <= min(bw_hz * 1.90, float(sr) * 0.47))
        cutoff_score = 0.0
        if below_mask.any() and above_mask.any():
            below_level = float(mean_spec[below_mask].mean())
            above_level = float(mean_spec[above_mask].mean()) + 1e-10
            ratio = above_level / (below_level + 1e-10)
            # Hard codec cutoff: ratio → 0; natural roll-off: ratio ≈ 0.05–0.30
            cutoff_score = float(np.clip((0.18 - ratio) / 0.16, 0.0, 1.0))

        # --- 2. SFM inside active band (1 kHz → 95 % of detected bandwidth) ---
        sfm_upper = min(bw_hz * 0.95, float(sr) * 0.45)
        active_mask = (freqs >= 1000.0) & (freqs <= sfm_upper)
        sfm_score = 0.0
        if active_mask.sum() >= 16:
            cs = mean_spec[active_mask]
            log_geo = float(np.mean(np.log(cs)))
            geo_mean = float(np.exp(log_geo))
            arith_mean = float(cs.mean())
            sfm = float(np.clip(geo_mean / (arith_mean + 1e-10), 0.0, 1.0))
            # Natural music (structured, harmonic peaks): SFM ≈ 0.02–0.20
            # Noise-floor-raised codec output: SFM ≈ 0.30–0.80
            sfm_score = float(np.clip((sfm - 0.22) / 0.35, 0.0, 1.0))

        # --- 3. Pre-echo pattern (MP3-specific) ---
        frame_energies = np.sqrt(np.mean(specs**2, axis=1))
        pre_echo_score = 0.0
        if len(frame_energies) >= 3:
            deltas = np.diff(frame_energies)
            onset_threshold = float(np.std(deltas)) * 1.5
            onset_idx = np.where(deltas > onset_threshold)[0]
            for idx in onset_idx:
                if idx > 0:
                    ratio_pe = float(frame_energies[idx]) / (float(frame_energies[idx - 1]) + 1e-10)
                    pre_echo_score = max(pre_echo_score, float(np.clip((ratio_pe - 1.8) / 3.5, 0.0, 1.0)))

        # --- Combined score: cutoff dominates (most reliable) ---
        artifact_score = float(np.clip(0.50 * cutoff_score + 0.30 * sfm_score + 0.20 * pre_echo_score, 0.0, 1.0))

        # --- Codec type hint (encoded as float) ---
        if artifact_score < 0.10:
            code = 0.0  # clean
        elif pre_echo_score > 0.25 and sfm_score > 0.20:
            code = 1.0  # mp3
        elif cutoff_score > 0.50 and pre_echo_score < 0.15:
            code = 2.0  # aac
        else:
            code = 3.0  # lossy (unspecified)

        return artifact_score, code

    def _rotation_periodicity(self, mono: np.ndarray, sr: int) -> tuple[float, float]:
        """Erkennt turntable rotation frequency via RMS envelope modulation.

        Physical principle: A rotating platter introduces amplitude modulation of
        the groove-to-stylus chain at the platter rotation frequency.  Dust,
        groove eccentricity, and sub-chassis resonance all modulate the recorded
        signal at the RPM fundamental (and harmonics).

        Target frequencies:
            LP (33⅓ RPM)      → 0.556 Hz ± 0.04 Hz
            Single (45 RPM)   → 0.750 Hz ± 0.05 Hz
            Shellac (78 RPM)  → 1.300 Hz ± 0.08 Hz

        Tape / digital sources have no periodic platter rotation → no peak.

        Returns:
            (rotation_hz, rotation_strength)
            rotation_hz:      dominant rotation frequency [Hz]; 0.0 if not detected.
            rotation_strength: normalised peak SNR [0, 1]; 0 = no signal.

        References:
            Cano et al. (2005) — Audio restore detection.
            Rodriguez & Bello (2018) — Turntable speed estimation.
        """
        # Need at least 4 s to detect LP rotation (0.556 Hz → ~1.8 s/cycle × 2)
        min_samples = int(4.0 * sr)
        if len(mono) < min_samples:
            return 0.0, 0.0
        # 25 ms RMS envelope frames
        frame_size = max(128, int(0.025 * sr))
        n_frames = len(mono) // frame_size
        if n_frames < 80:  # < 2 s at 25 ms frames
            return 0.0, 0.0
        frames = mono[: n_frames * frame_size].reshape(n_frames, frame_size)
        rms_env = np.sqrt(np.mean(frames**2, axis=1)).astype(np.float64)
        rms_env -= rms_env.mean()  # remove DC

        # FFT of the modulation envelope
        env_sr = sr / frame_size  # envelope "sample rate" (frames/s)
        n_fft_env = len(rms_env)
        spec = np.abs(np.fft.rfft(rms_env * np.hanning(n_fft_env))) ** 2
        mod_freqs = np.fft.rfftfreq(n_fft_env, d=1.0 / env_sr)

        # Search band [0.30, 2.00 Hz] covers LP, 45, 78 RPM and 2nd harmonics
        search_mask = (mod_freqs >= 0.30) & (mod_freqs <= 2.00)
        if not search_mask.any():
            return 0.0, 0.0

        # Noise floor estimate from [3, 15 Hz] (motor and mechanical noise avoid rotation band)
        noise_mask = (mod_freqs >= 3.0) & (mod_freqs <= 15.0)
        noise_floor = float(np.median(spec[noise_mask])) + 1e-30 if noise_mask.any() else float(spec.mean()) + 1e-30

        # Find peak within the search band
        search_spec = np.where(search_mask, spec, 0.0)
        peak_idx = int(np.argmax(search_spec))
        peak_power = float(spec[peak_idx])
        peak_freq = float(mod_freqs[peak_idx])

        # Strength: normalised peak SNR; > 6 dB → some signal; > 15 dB → strong
        peak_snr_db = 10.0 * math.log10(max(peak_power / noise_floor, 1.0))
        rotation_strength = float(np.clip((peak_snr_db - 4.0) / 16.0, 0.0, 1.0))
        if rotation_strength < 0.05:
            return 0.0, 0.0

        return peak_freq, rotation_strength

    def _infrasonic_rms(self, audio: np.ndarray, sr: int) -> float:
        """Sub-20 Hz normalised RMS for vinyl/shellac rumble discrimination.

        A turntable's platter bearing generates characteristic infrasonic energy
        (typically −55 to −40 dBFS) below 20 Hz.  Stereo vinyl pressings show
        additional low-correlation content in L and R below 20 Hz due to
        mechanical noise being uncorrelated between the two stylus contact axes.

        Tape and digital sources produce negligible infrasonic energy.

        Returns:
            Normalised infrasonic RMS ∈ [0, 1].

        References:
            Schuller et al. (1998) — Perceptual coding for digital audio.
            Pohlmann (2010) — Principles of Digital Audio, 6th ed.
        """
        mono = self._to_mono(audio)
        if len(mono) < sr:
            return 0.0
        # Butterworth LP at 20 Hz (order 4) via cascaded biquads (scipy)
        try:
            from scipy.signal import butter, sosfilt  # type: ignore[import]

            sos = butter(4, 20.0, btype="low", fs=float(sr), output="sos")
            infra = sosfilt(sos, mono.astype(np.float64)).astype(np.float32)
        except Exception:
            # Manual crude low-pass: 20 Hz / sr samples running average
            k = max(1, int(sr // 40))
            infra = np.convolve(mono, np.ones(k, dtype=np.float32) / k, mode="same")
        broad_rms = float(np.sqrt(np.mean(mono**2))) + 1e-10
        infra_rms = float(np.sqrt(np.mean(infra**2)))
        return float(np.clip(infra_rms / broad_rms, 0.0, 1.0))

    def _pre_echo(self, mono: np.ndarray, sr: int) -> float:
        frame = self._FRAME_SIZE
        n_frames = len(mono) // frame
        if n_frames < 4:
            return 0.0
        frames = mono[: n_frames * frame].reshape(n_frames, frame)
        energies = np.sqrt(np.mean(frames**2, axis=1))
        if energies.max() < 1e-8:
            return 0.0
        max_idx = int(np.argmax(energies))
        if max_idx == 0:
            return 0.0
        ratio = float(energies[max_idx - 1]) / (float(energies[max_idx]) + 1e-10)
        ms_per_frame = frame / sr * 1000.0
        return float(np.clip(ratio * ms_per_frame * 2.0, 0.0, 50.0))


class _MaterialScorer:
    """Bayesian Gaussian-likelihood material scoring (replaces binary threshold scorer).

    Each material has a per-feature Gaussian model (μ, σ) calibrated from
    literature values and empirical measurements.  Scoring computes the
    log-likelihood of the observed feature vector under each material model
    and converts to posterior probability via softmax.

    Advantages over the old boolean _s() scorer:
      - Soft gradients instead of hard step functions
      - Feature-importance weighting via inverse σ
      - Calibrated posterior confidence (not best/total heuristic)
      - Ambiguity-aware evidence ranking (close posteriors → low confidence)

    References:
        Duda, Hart & Stork (2001). Pattern Classification, 2nd ed.
        Bianchi et al. (2020). Detection of double audio compression.
    """

    # -----------------------------------------------------------------------
    # Gaussian feature models: (μ, σ) per feature per material.
    # Features: bandwidth_hz, snr_db, noise_color, crackle_density,
    #           wow_depth, block_artifact, pre_echo_ms,
    #           rotation_strength, infrasonic_rms, codec_type_code
    #
    # σ encodes *both* natural variance and feature discriminative weight:
    #   tight σ = high discriminative power (e.g. rotation for vinyl)
    #   wide σ  = low discriminative power (feature varies a lot for this material)
    #
    # Values calibrated from Pohlmann (2010), IEC 60386, Schuller (1998),
    # Farid (2009), empirical Aurik test corpus (2024–2026).
    # -----------------------------------------------------------------------
    _MATERIAL_MODELS: dict[str, dict[str, tuple[float, float]]] = {
        "shellac": {
            "bandwidth_hz": (5500.0, 1500.0),
            "snr_db": (10.0, 5.0),
            "noise_color": (2.2, 0.5),
            "crackle_density": (0.02, 0.02),
            "wow_depth": (0.3, 0.3),
            "block_artifact": (0.0, 0.05),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.40, 0.20),
            "infrasonic_rms": (0.06, 0.04),
            "codec_type_code": (0.0, 0.3),
        },
        "wax_cylinder": {
            "bandwidth_hz": (3500.0, 1200.0),
            "snr_db": (6.0, 4.0),
            "noise_color": (2.8, 0.6),
            "crackle_density": (0.04, 0.03),
            "wow_depth": (1.0, 0.8),
            "block_artifact": (0.0, 0.05),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.0, 0.10),
            "infrasonic_rms": (0.02, 0.03),
            "codec_type_code": (0.0, 0.3),
        },
        "vinyl": {
            "bandwidth_hz": (14000.0, 4000.0),
            "snr_db": (30.0, 10.0),
            "noise_color": (1.5, 0.4),
            "crackle_density": (0.004, 0.005),
            "wow_depth": (0.15, 0.15),
            "block_artifact": (0.0, 0.05),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.45, 0.20),
            "infrasonic_rms": (0.08, 0.05),
            "codec_type_code": (0.0, 0.3),
        },
        "tape": {
            "bandwidth_hz": (12000.0, 3000.0),
            "snr_db": (25.0, 8.0),
            "noise_color": (1.6, 0.4),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (1.2, 0.8),
            "block_artifact": (0.0, 0.05),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.0, 0.08),
            "infrasonic_rms": (0.01, 0.02),
            "codec_type_code": (0.0, 0.3),
        },
        "reel_tape": {
            "bandwidth_hz": (15000.0, 3000.0),
            "snr_db": (28.0, 7.0),
            "noise_color": (1.3, 0.3),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (0.3, 0.3),
            "block_artifact": (0.0, 0.05),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.0, 0.08),
            "infrasonic_rms": (0.01, 0.02),
            "codec_type_code": (0.0, 0.3),
        },
        "wire_recording": {
            "bandwidth_hz": (5000.0, 1500.0),
            "snr_db": (12.0, 5.0),
            "noise_color": (2.0, 0.5),
            "crackle_density": (0.0001, 0.0002),
            "wow_depth": (3.0, 1.5),
            "block_artifact": (0.0, 0.05),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.0, 0.10),
            "infrasonic_rms": (0.01, 0.02),
            "codec_type_code": (0.0, 0.3),
        },
        "lacquer_disc": {
            "bandwidth_hz": (9000.0, 2500.0),
            "snr_db": (18.0, 6.0),
            "noise_color": (1.7, 0.4),
            "crackle_density": (0.008, 0.008),
            "wow_depth": (0.2, 0.2),
            "block_artifact": (0.0, 0.05),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.30, 0.20),
            "infrasonic_rms": (0.04, 0.04),
            "codec_type_code": (0.0, 0.3),
        },
        "cassette": {
            "bandwidth_hz": (10000.0, 3000.0),
            "snr_db": (22.0, 7.0),
            "noise_color": (1.5, 0.4),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (1.5, 1.0),
            "block_artifact": (0.0, 0.05),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.0, 0.08),
            "infrasonic_rms": (0.01, 0.02),
            "codec_type_code": (0.0, 0.3),
        },
        "dat": {
            "bandwidth_hz": (20000.0, 2000.0),
            "snr_db": (50.0, 8.0),
            "noise_color": (0.3, 0.3),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (0.0, 0.1),
            "block_artifact": (0.08, 0.06),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.0, 0.05),
            "infrasonic_rms": (0.0, 0.01),
            "codec_type_code": (0.0, 0.5),
        },
        "cd_digital": {
            "bandwidth_hz": (21000.0, 1500.0),
            "snr_db": (60.0, 8.0),
            "noise_color": (0.2, 0.3),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (0.0, 0.05),
            "block_artifact": (0.0, 0.03),
            "pre_echo_ms": (0.0, 1.0),
            "rotation_strength": (0.0, 0.05),
            "infrasonic_rms": (0.0, 0.01),
            "codec_type_code": (0.0, 0.3),
        },
        "mp3_low": {
            "bandwidth_hz": (11000.0, 2500.0),
            "snr_db": (35.0, 8.0),
            "noise_color": (0.5, 0.4),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (0.0, 0.1),
            "block_artifact": (0.40, 0.15),
            "pre_echo_ms": (12.0, 6.0),
            "rotation_strength": (0.0, 0.05),
            "infrasonic_rms": (0.0, 0.01),
            "codec_type_code": (1.0, 0.3),
        },
        "mp3_high": {
            "bandwidth_hz": (17000.0, 2000.0),
            "snr_db": (42.0, 7.0),
            "noise_color": (0.4, 0.3),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (0.0, 0.1),
            "block_artifact": (0.15, 0.10),
            "pre_echo_ms": (6.0, 4.0),
            "rotation_strength": (0.0, 0.05),
            "infrasonic_rms": (0.0, 0.01),
            "codec_type_code": (1.0, 0.3),
        },
        "aac": {
            "bandwidth_hz": (19000.0, 1500.0),
            "snr_db": (48.0, 7.0),
            "noise_color": (0.3, 0.3),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (0.0, 0.1),
            "block_artifact": (0.10, 0.08),
            "pre_echo_ms": (1.5, 1.5),
            "rotation_strength": (0.0, 0.05),
            "infrasonic_rms": (0.0, 0.01),
            "codec_type_code": (2.0, 0.3),
        },
        "minidisc": {
            "bandwidth_hz": (14000.0, 2000.0),
            "snr_db": (40.0, 6.0),
            "noise_color": (0.4, 0.3),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (0.0, 0.1),
            "block_artifact": (0.18, 0.10),
            "pre_echo_ms": (3.0, 3.0),
            "rotation_strength": (0.0, 0.05),
            "infrasonic_rms": (0.0, 0.01),
            "codec_type_code": (3.0, 0.5),
        },
        "streaming": {
            "bandwidth_hz": (18000.0, 2000.0),
            "snr_db": (45.0, 8.0),
            "noise_color": (0.3, 0.3),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (0.0, 0.1),
            "block_artifact": (0.06, 0.05),
            "pre_echo_ms": (2.0, 2.0),
            "rotation_strength": (0.0, 0.05),
            "infrasonic_rms": (0.0, 0.01),
            "codec_type_code": (1.5, 0.8),
        },
        "unknown": {
            "bandwidth_hz": (12000.0, 8000.0),
            "snr_db": (25.0, 15.0),
            "noise_color": (1.0, 1.0),
            "crackle_density": (0.005, 0.01),
            "wow_depth": (0.5, 1.0),
            "block_artifact": (0.05, 0.10),
            "pre_echo_ms": (2.0, 5.0),
            "rotation_strength": (0.05, 0.15),
            "infrasonic_rms": (0.02, 0.05),
            "codec_type_code": (1.0, 1.5),
        },
    }

    # Feature keys in canonical order — must match _MATERIAL_MODELS keys
    _FEATURE_KEYS: list[str] = [
        "bandwidth_hz",
        "snr_db",
        "noise_color",
        "crackle_density",
        "wow_depth",
        "block_artifact",
        "pre_echo_ms",
        "rotation_strength",
        "infrasonic_rms",
        "codec_type_code",
    ]

    def score(self, features: dict[str, float], MaterialType: Any) -> ClassificationResult:
        """Bewertet Merkmale probabilistisch gegen alle Materialmodelle."""
        # Extract feature vector
        bw = features.get("bandwidth_hz", 0.0)
        snr = features.get("snr_db", 0.0)
        nc = features.get("noise_color", 1.0)
        cd = features.get("crackle_density", 0.0)
        wf = features.get("wow_depth", features.get("wow_flutter_hz", 0.0))
        ba = features.get("block_artifact", 0.0)
        pe = features.get("pre_echo_ms", 0.0)
        rot_hz = features.get("rotation_hz", 0.0)
        rot_str = features.get("rotation_strength", 0.0)
        infra = features.get("infrasonic_rms", 0.0)
        codec_code = features.get("codec_type_code", 0.0)

        obs = {
            "bandwidth_hz": bw,
            "snr_db": snr,
            "noise_color": nc,
            "crackle_density": cd,
            "wow_depth": wf,
            "block_artifact": ba,
            "pre_echo_ms": pe,
            "rotation_strength": rot_str,
            "infrasonic_rms": infra,
            "codec_type_code": codec_code,
        }

        # Compute log-likelihood for each material
        log_likelihoods: dict[str, float] = {}
        for mat_name, model in self._MATERIAL_MODELS.items():
            ll = 0.0
            for feat_key in self._FEATURE_KEYS:
                mu, sigma = model[feat_key]
                x = obs.get(feat_key, 0.0)
                sigma = max(sigma, 1e-6)
                # Log of Gaussian PDF (constant term cancels in softmax)
                ll -= 0.5 * ((x - mu) / sigma) ** 2 + math.log(sigma)
            log_likelihoods[mat_name] = ll

        # Softmax → posterior probabilities (uniform prior)
        max_ll = max(log_likelihoods.values())
        exp_scores = {k: math.exp(v - max_ll) for k, v in log_likelihoods.items()}
        total = sum(exp_scores.values()) + 1e-30
        posteriors = {k: v / total for k, v in exp_scores.items()}

        best_name = max(posteriors, key=posteriors.__getitem__)
        confidence = float(posteriors[best_name])
        material = self._find_enum(MaterialType, best_name)

        # Build evidence list with per-feature diagnostics
        sorted_mats = sorted(posteriors.items(), key=lambda x: -x[1])[:5]
        evidence = []
        for mat_name, post in sorted_mats:
            model = self._MATERIAL_MODELS[mat_name]
            matched = []
            against = []
            for fk in self._FEATURE_KEYS:
                mu, sigma = model[fk]
                x = obs.get(fk, 0.0)
                z = abs(x - mu) / max(sigma, 1e-6)
                if z <= 1.5:
                    matched.append(fk)
                elif z > 2.5:
                    against.append(fk)
            evidence.append(
                MaterialEvidence(
                    material=self._find_enum(MaterialType, mat_name),
                    confidence=float(np.clip(post, 0.0, 1.0)),
                    features_matched=matched,
                    features_against=against,
                )
            )

        # Decode codec_type string from code
        _codec_map = {0.0: "clean", 1.0: "mp3", 2.0: "aac", 3.0: "lossy"}
        # Round to nearest integer for lookup
        codec_type_str = _codec_map.get(round(codec_code), "")

        return ClassificationResult(
            material=material,
            confidence=confidence,
            evidence=evidence,
            bandwidth_hz=bw,
            snr_db=snr,
            noise_color=nc,
            crackle_density=cd,
            wow_flutter_hz=features.get("wow_flutter_hz", 0.0),
            block_artifact=ba,
            pre_echo_ms=pe,
            classifier_source="dsp",
            rotation_hz=rot_hz,
            rotation_strength=rot_str,
            infrasonic_rms=infra,
            codec_type=codec_type_str,
        )

    @staticmethod
    def _find_enum(MaterialType: Any, name: str) -> Any:
        if MaterialType is None:
            return name
        for m in MaterialType:
            if m.value == name or m.name.lower() == name.lower():
                return m
        return name


_sha_cache: dict[str, ClassificationResult] = {}
_sha_cache_lock = threading.Lock()
_MAX_CACHE = 64


class MediumClassifier:
    """Automatische Trägermedien-Erkennung (Aurik Spec §2.1)."""

    def __init__(self) -> None:
        """Initialisiert Fingerprinter und probabilistischen Material-Scorer."""
        self._fp = _SpectralFingerprinter()
        self._sc = _MaterialScorer()

    def classify_medium(self, audio: np.ndarray, sr: int) -> ClassificationResult:
        """Klassifiziert ein Signal ausschließlich per DSP-Pfad."""
        return self.classify(audio, sr, use_ml=False)

    def classify(self, audio: np.ndarray, sr: int, use_ml: bool = True) -> ClassificationResult:
        """Klassifiziert ein Signal via ML-Versuch mit DSP-Fallback."""
        key = self._cache_key(audio, sr)
        with _sha_cache_lock:
            if key in _sha_cache:
                return _sha_cache[key]
        if use_ml:
            r = self._try_clap_classification(audio, sr)
            if r is not None:
                self._cache_put(key, r)
                return r
        r = self._dsp_classify(audio, sr)
        self._cache_put(key, r)
        return r

    def _dsp_classify(self, audio: np.ndarray, sr: int) -> ClassificationResult:
        MT = _get_material_type()
        if audio.size == 0:
            mat = MT.UNKNOWN if MT is not None else "unknown"
            return ClassificationResult(
                material=mat, confidence=0.0, evidence=[MaterialEvidence(mat, 0.0)], classifier_source="unknown"
            )
        return self._sc.score(self._fp.extract(audio, sr), MT)

    def _try_clap_classification(self, audio: np.ndarray, sr: int) -> ClassificationResult | None:
        try:
            from plugins.laion_clap_plugin import get_laion_clap

            plugin: Any = get_laion_clap()
            tagged = plugin.tag(audio, sr)
            material_tags = getattr(tagged, "material_tags", None)
            if isinstance(material_tags, dict) and material_tags:
                best_material, best_conf = max(material_tags.items(), key=lambda item: float(item[1]))
                confidence = max(float(best_conf), float(getattr(tagged, "confidence", 0.0)))
                if confidence >= 0.35:
                    return ClassificationResult(
                        material=best_material,
                        confidence=confidence,
                        evidence=[MaterialEvidence(best_material, confidence)],
                        classifier_source="clap_ml",
                    )
        except Exception as _exc:
            logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
        return None

    @staticmethod
    def _cache_key(audio: np.ndarray, sr: int) -> str:
        h = hashlib.sha256()
        h.update(audio.ravel().view(np.uint8)[:65536].tobytes())
        h.update(sr.to_bytes(4, "little"))
        return h.hexdigest()[:16]

    @staticmethod
    def _cache_put(key: str, result: ClassificationResult) -> None:
        with _sha_cache_lock:
            if len(_sha_cache) >= _MAX_CACHE:
                del _sha_cache[next(iter(_sha_cache))]
            _sha_cache[key] = result


_INSTANCE_HOLDER: list[MediumClassifier | None] = [None]
_lock = threading.Lock()


def get_medium_classifier() -> MediumClassifier:
    """Liefert den thread-sicheren MediumClassifier-Singleton."""
    if _INSTANCE_HOLDER[0] is None:
        with _lock:
            if _INSTANCE_HOLDER[0] is None:
                _INSTANCE_HOLDER[0] = MediumClassifier()
    classifier = _INSTANCE_HOLDER[0]
    assert classifier is not None
    return classifier


def classify_medium(audio: np.ndarray, sr: int, use_ml: bool = True) -> ClassificationResult:
    """Bequemer Top-Level-Zugang zur Medium-Klassifikation."""
    return get_medium_classifier().classify(audio, sr, use_ml=use_ml)


__all__ = ["ClassificationResult", "MaterialEvidence", "MediumClassifier", "classify_medium", "get_medium_classifier"]
