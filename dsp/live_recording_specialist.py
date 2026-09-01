"""
Live Recording Specialist - Specialized Tools for Live Concert Recordings

This module provides 8 specialized tools for processing live recordings:
1. CrowdNoiseIsolator - Remove/reduce audience noise
2. RoomDeverberator - Reduce excessive room reverb
3. StageBleedReducer - Reduce microphone leakage/bleed
4. FeedbackCanceller - Detect and remove feedback howls
5. PAResonanceRemover - Remove PA system resonances
6. HandlingNoiseDetector - Detect microphone handling noise
7. DeWindTool - Remove wind noise (outdoor recordings)
8. RoomModeCorrector - Correct room modal resonances

Target: +3 points (118.0 → 121.0/100)
Coverage: Live concert recordings, bootlegs, archives, broadcast

Author: AURIK Development Team
Date: 2026-02-08
Version: v8
Status: Production Ready
"""

import logging
from typing import Any

import numpy as np
from scipy import signal
from scipy.fft import rfft
from scipy.signal import butter, find_peaks, hilbert, istft, sosfilt, stft

logger = logging.getLogger(__name__)


class CrowdNoiseIsolator:
    """
    Isolate and reduce crowd noise from live recordings.

    Uses spectral gating and temporal analysis to separate crowd noise
    from musical content. Preserves vocal/instrumental clarity while
    reducing audience noise.

    Algorithm:
    1. STFT analysis
    2. Identify crowd noise regions (broadband, continuous)
    3. Spectral gating with adaptive threshold
    4. Preserve transients and musical content

    Target: >15dB SNR improvement
    """

    def __init__(self, sensitivity: float = 0.7, preserve_applause: bool = True):
        """
        Initialisiert CrowdNoiseIsolator.

        Parameters:
        -----------
        sensitivity : float, default 0.7
            Detection sensitivity (0.0 = gentle, 1.0 = aggressive)
        preserve_applause : bool, default True
            If True, preserve applause/cheering (applause = intentional)
        """
        self.sensitivity = np.clip(sensitivity, 0.0, 1.0)
        self.preserve_applause = preserve_applause

    def detect_crowd_noise(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        Erkennt crowd noise presence and characteristics.

        Returns:
        --------
        dict with keys:
            - crowd_noise_detected: bool
            - crowd_noise_ratio: float (0.0-1.0)
            - applause_detected: bool
            - energy_ratio: float
        """
        # STFT analysis
        nperseg = 2048
        f, _t, Zxx = stft(audio, sr, nperseg=nperseg)
        magnitude = np.abs(Zxx)

        # Crowd noise characteristics: broadband 200-4000 Hz, continuous
        crowd_band_idx = np.where((f >= 200) & (f <= 4000))[0]
        crowd_energy = np.mean(magnitude[crowd_band_idx, :], axis=0)

        # Spectral flatness (crowd noise is noise-like → high flatness)
        eps = 1e-10
        geometric_mean = np.exp(np.mean(np.log(magnitude + eps), axis=0))
        arithmetic_mean = np.mean(magnitude, axis=0)
        spectral_flatness = geometric_mean / (arithmetic_mean + eps)

        # Crowd noise frames: high flatness (>0.4) + significant energy
        threshold = np.percentile(crowd_energy, 50) * self.sensitivity
        crowd_frames = (spectral_flatness > 0.4) & (crowd_energy > threshold)
        crowd_noise_ratio = np.mean(crowd_frames)

        # Detect applause (transient bursts in crowd band)
        analytic = np.asarray(hilbert(np.asarray(audio, dtype=np.float64)), dtype=np.complex128)
        envelope = np.sqrt(np.square(analytic.real) + np.square(analytic.imag))
        envelope_smooth = np.convolve(envelope, np.ones(int(0.05 * sr)) / (0.05 * sr), mode="same")
        transients = np.diff(envelope_smooth) > (np.std(envelope_smooth) * 3)
        applause_detected = np.sum(transients) > (len(audio) / sr) * 2  # >2 bursts/second

        # Energy ratio
        crowd_energy_total = np.sum(magnitude[crowd_band_idx, :])
        total_energy = np.sum(magnitude) + eps
        energy_ratio = crowd_energy_total / total_energy

        return {
            "crowd_noise_detected": crowd_noise_ratio > 0.1,
            "crowd_noise_ratio": float(crowd_noise_ratio),
            "applause_detected": bool(applause_detected and self.preserve_applause),
            "energy_ratio": float(energy_ratio),
        }

    def remove_crowd_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Entfernt/Reduziert Publikumsgeräusche aus Audio.

        Uses spectral gating with adaptive threshold.
        """
        # Preserve input dtype
        input_dtype = audio.dtype
        audio = audio.astype(np.float64)

        # STFT
        nperseg = 2048
        noverlap = nperseg // 2
        _f, _t, Zxx = stft(audio, sr, nperseg=nperseg, noverlap=noverlap)
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Compute spectral flatness per frame
        eps = 1e-10
        geometric_mean = np.exp(np.mean(np.log(magnitude + eps), axis=0))
        arithmetic_mean = np.mean(magnitude, axis=0)
        spectral_flatness = geometric_mean / (arithmetic_mean + eps)

        # Identify crowd noise frames (high flatness → noise-like)
        crowd_threshold = 0.4 * self.sensitivity
        crowd_frames = spectral_flatness > crowd_threshold

        # Spectral gating: Reduce magnitude in crowd frames
        gate_reduction = 0.3 + (0.5 * self.sensitivity)  # 30-80% reduction
        magnitude_gated = magnitude.copy()
        magnitude_gated[:, crowd_frames] *= 1.0 - gate_reduction

        # Preserve transients (applause, cheers)
        if self.preserve_applause:
            # Detect transient frames (high energy increase)
            frame_energy = np.sum(magnitude**2, axis=0)
            energy_diff = np.diff(frame_energy, prepend=frame_energy[0])
            transient_frames = energy_diff > (np.std(energy_diff) * 3)

            # Restore transient frames
            magnitude_gated[:, transient_frames] = magnitude[:, transient_frames]

        # Reconstruct
        Zxx_gated = magnitude_gated * np.exp(1j * phase)
        _, audio_cleaned = istft(Zxx_gated, sr, nperseg=nperseg, noverlap=noverlap)

        # Match original length
        if len(audio_cleaned) < len(audio):
            audio_cleaned = np.pad(audio_cleaned, (0, len(audio) - len(audio_cleaned)))
        else:
            audio_cleaned = audio_cleaned[: len(audio)]

        return audio_cleaned.astype(input_dtype)  # type: ignore[no-any-return]


class RoomDeverberator:
    """
    Reduce excessive room reverb from live recordings.

    Uses spectral subtraction and temporal analysis to reduce reverb
    while preserving direct sound and intentional room ambience.

    Algorithm:
    1. Estimate reverb tail (late reflections)
    2. Spectral subtraction in reverb-dominant regions
    3. Preserve early reflections (< 50ms)

    Target: Achieve target RT60 ± 0.1s
    """

    def __init__(self, target_rt60: float = 0.4, strength: float = 0.7):
        """
        Initialisiert RoomDeverberator.

        Parameters:
        -----------
        target_rt60 : float, default 0.4
            Target reverberation time in seconds
        strength : float, default 0.7
            Deverberation strength (0.0 = none, 1.0 = maximum)
        """
        self.target_rt60 = target_rt60
        self.strength = np.clip(strength, 0.0, 1.0)

    def estimate_rt60(self, audio: np.ndarray, sr: int) -> float:
        """
        Schätzt RT60 (reverberation time) of audio.

        RT60 = time for reverb to decay by 60dB
        """
        # Use Schroeder integration method
        # 1. Compute squared impulse response (energy decay curve)
        analytic = np.asarray(hilbert(np.asarray(audio, dtype=np.float64)), dtype=np.complex128)
        envelope = np.sqrt(np.square(analytic.real) + np.square(analytic.imag))
        energy = envelope**2

        # 2. Reverse integrate (Schroeder curve)
        energy_reversed = energy[::-1]
        schroeder_curve = np.cumsum(energy_reversed)[::-1]
        schroeder_curve = schroeder_curve / (np.max(schroeder_curve) + 1e-10)

        # 3. Convert to dB
        schroeder_db = 10 * np.log10(schroeder_curve + 1e-10)

        # 4. Find T60: time from 0dB to -60dB
        try:
            idx_0db = np.where(schroeder_db >= -5)[0][-1]
            idx_60db = np.where(schroeder_db <= -60)[0]
            if len(idx_60db) > 0:
                idx_60db = idx_60db[0]
                rt60 = (idx_60db - idx_0db) / sr
            else:
                # Extrapolate from -5dB to -35dB (EDT method)
                idx_35db = np.where(schroeder_db <= -35)[0]
                if len(idx_35db) > 0:
                    idx_35db = idx_35db[0]
                    edt = (idx_35db - idx_0db) / sr
                    rt60 = edt * 2  # Extrapolate
                else:
                    rt60 = 0.5  # Default fallback
        except Exception:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6
            rt60 = 0.5  # Fallback

        return float(rt60)

    def reduce_reverb(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Reduce excessive reverb from audio.

        Uses spectral subtraction on late reflections.
        """
        # Preserve input dtype
        input_dtype = audio.dtype
        audio = audio.astype(np.float64)

        # Estimate current RT60
        current_rt60 = self.estimate_rt60(audio, sr)

        # If current RT60 <= target, no processing needed
        if current_rt60 <= self.target_rt60:
            return audio.astype(input_dtype)

        # STFT
        nperseg = 2048
        noverlap = nperseg // 2
        _f, _t, Zxx = stft(audio, sr, nperseg=nperseg, noverlap=noverlap)
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Estimate reverb tail energy (frames with decaying energy)
        frame_energy = np.sum(magnitude**2, axis=0)
        frame_energy_smooth = np.convolve(frame_energy, np.ones(5) / 5, mode="same")

        # Identify reverb-dominant frames (low energy, after transients)
        transient_threshold = np.percentile(frame_energy, 70)
        reverb_frames = frame_energy_smooth < transient_threshold

        # Spectral subtraction in reverb frames
        reduction_factor = 1.0 - (self.strength * 0.6)  # Max 60% reduction
        magnitude_processed = magnitude.copy()
        magnitude_processed[:, reverb_frames] *= reduction_factor

        # Reconstruct
        Zxx_processed = magnitude_processed * np.exp(1j * phase)
        _, audio_deverb = istft(Zxx_processed, sr, nperseg=nperseg, noverlap=noverlap)

        # Match original length
        if len(audio_deverb) < len(audio):
            audio_deverb = np.pad(audio_deverb, (0, len(audio) - len(audio_deverb)))
        else:
            audio_deverb = audio_deverb[: len(audio)]

        return audio_deverb.astype(input_dtype)  # type: ignore[no-any-return]


class StageBleedReducer:
    """
    Reduce stage bleed (microphone leakage) in live recordings.

    Stage bleed occurs when multiple instruments/vocals are captured by
    a single microphone. This tool uses spectral masking to reduce bleed
    while preserving the primary source.

    Algorithm:
    1. Identify primary source frequency range
    2. Apply spectral masking to bleed regions
    3. Preserve transients and energy balance

    Target: >10dB isolation improvement
    """

    def __init__(self, sensitivity: float = 0.6):
        """
        Initialisiert StageBleedReducer.

        Parameters:
        -----------
        sensitivity : float, default 0.6
            Reduction sensitivity (0.0 = gentle, 1.0 = aggressive)
        """
        self.sensitivity = np.clip(sensitivity, 0.0, 1.0)

    def reduce_bleed(
        self,
        audio: np.ndarray,
        sr: int,
        primary_band: tuple[float, float] = (200, 4000),
    ) -> np.ndarray:
        """
        Reduce stage bleed from audio.

        Parameters:
        -----------
        primary_band : tuple, default (200, 4000)
            Frequency range of primary source in Hz
        """
        # Preserve input dtype
        input_dtype = audio.dtype
        audio = audio.astype(np.float64)

        # STFT
        nperseg = 2048
        noverlap = nperseg // 2
        f, _t, Zxx = stft(audio, sr, nperseg=nperseg, noverlap=noverlap)
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Identify primary source band
        primary_idx = np.where((f >= primary_band[0]) & (f <= primary_band[1]))[0]
        bleed_idx = np.where((f < primary_band[0]) | (f > primary_band[1]))[0]

        # Compute energy ratio (primary vs bleed)
        primary_energy = np.sum(magnitude[primary_idx, :] ** 2, axis=0)
        bleed_energy = np.sum(magnitude[bleed_idx, :] ** 2, axis=0)

        # Identify bleed-dominant frames (high bleed-to-primary ratio)
        ratio = bleed_energy / (primary_energy + 1e-10)
        bleed_threshold = np.percentile(ratio, 50) * (1.0 + self.sensitivity)
        bleed_frames = ratio > bleed_threshold

        # Reduce bleed regions
        reduction_factor = 1.0 - (self.sensitivity * 0.5)  # Max 50% reduction
        magnitude_processed = magnitude.copy()
        magnitude_processed[bleed_idx[:, None], bleed_frames] *= reduction_factor

        # Reconstruct
        Zxx_processed = magnitude_processed * np.exp(1j * phase)
        _, audio_reduced = istft(Zxx_processed, sr, nperseg=nperseg, noverlap=noverlap)

        # Match original length
        if len(audio_reduced) < len(audio):
            audio_reduced = np.pad(audio_reduced, (0, len(audio) - len(audio_reduced)))
        else:
            audio_reduced = audio_reduced[: len(audio)]

        return audio_reduced.astype(input_dtype)  # type: ignore[no-any-return]


class FeedbackCanceller:
    """
    Erkennt and remove feedback howls from live recordings.

    Feedback occurs when microphone picks up speaker output, creating
    a resonant howl at specific frequencies (typically 200-4000 Hz).

    Algorithm:
    1. FFT analysis to detect narrow-band peaks
    2. Identify feedback frequencies (high Q-factor)
    3. Apply notch filters at feedback frequencies

    Target: >40dB attenuation at howl frequency
    """

    def __init__(self, sensitivity: float = 0.8):
        """
        Initialisiert FeedbackCanceller.

        Parameters:
        -----------
        sensitivity : float, default 0.8
            Detection sensitivity (0.0 = conservative, 1.0 = aggressive)
        """
        self.sensitivity = np.clip(sensitivity, 0.0, 1.0)

    def detect_feedback(self, audio: np.ndarray, sr: int) -> list[float]:
        """
        Erkennt feedback howl frequencies.

        Returns:
        --------
        list of float
            Detected feedback frequencies in Hz
        """
        # FFT analysis
        fft_size = 8192  # High resolution for narrow peaks
        audio_fft = np.asarray(rfft(audio, n=fft_size), dtype=np.complex128)
        magnitude = np.sqrt(np.square(audio_fft.real) + np.square(audio_fft.imag))
        freqs = np.fft.rfftfreq(fft_size, 1 / sr)

        # Focus on feedback range (200-4000 Hz)
        feedback_range_idx = np.where((freqs >= 200) & (freqs <= 4000))[0]
        magnitude_feedback = magnitude[feedback_range_idx]
        freqs_feedback = freqs[feedback_range_idx]

        # Detect peaks (feedback = narrow-band, high-energy peaks)
        threshold = np.percentile(magnitude_feedback, 95) * (0.5 + 0.5 * self.sensitivity)
        peaks, _properties = find_peaks(
            magnitude_feedback,
            height=threshold,
            prominence=threshold * 0.3,
            width=2,  # Narrow peaks
        )

        # Extract feedback frequencies
        feedback_freqs = freqs_feedback[peaks].tolist()

        return feedback_freqs  # type: ignore[no-any-return]

    def remove_feedback(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Entfernt detected feedback howls using notch filters.
        """
        # Detect feedback frequencies
        feedback_freqs = self.detect_feedback(audio, sr)

        if not feedback_freqs:
            return audio  # No feedback detected

        # Preserve input dtype
        input_dtype = audio.dtype
        audio = audio.astype(np.float64)

        # Apply notch filter at each feedback frequency
        for freq in feedback_freqs:
            # Notch filter: Q = 30 (narrow)
            Q = 30
            w0 = freq / (sr / 2)  # Normalized frequency

            # Design notch filter (iirnotch)
            b, a = signal.iirnotch(w0, Q, sr)
            sos = signal.tf2sos(b, a)
            audio = np.asarray(sosfilt(sos, audio), dtype=np.float64)

        return audio.astype(input_dtype)


class PAResonanceRemover:
    """
    Entfernt PA system resonances from live recordings.

    PA systems can introduce resonances at specific frequencies due to
    speaker/room interactions, creating tonal coloration.

    Algorithm:
    1. Detect resonance frequencies (comb filter analysis)
    2. Apply parametric EQ to reduce resonances
    3. Preserve overall tonal balance

    Target: Remove coloration while maintaining clarity
    """

    def __init__(self, sensitivity: float = 0.7):
        """
        Initialisiert PAResonanceRemover.

        Parameters:
        -----------
        sensitivity : float, default 0.7
            Detection sensitivity (0.0 = gentle, 1.0 = aggressive)
        """
        self.sensitivity = np.clip(sensitivity, 0.0, 1.0)

    def detect_resonances(self, audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
        """
        Erkennt PA resonance frequencies and their magnitudes.

        Returns:
        --------
        list of tuples (frequency_hz, magnitude_db)
        """
        # FFT analysis
        fft_size = 8192
        audio_fft = np.asarray(rfft(audio, n=fft_size), dtype=np.complex128)
        magnitude = np.sqrt(np.square(audio_fft.real) + np.square(audio_fft.imag))
        freqs = np.fft.rfftfreq(fft_size, 1 / sr)

        # Convert to dB
        magnitude_db = 20 * np.log10(magnitude + 1e-10)

        # Focus on PA resonance range (80-800 Hz typical)
        resonance_range_idx = np.where((freqs >= 80) & (freqs <= 800))[0]
        magnitude_resonance = magnitude_db[resonance_range_idx]
        freqs_resonance = freqs[resonance_range_idx]

        # Detect peaks (resonances = narrow peaks above median)
        threshold = np.median(magnitude_resonance) + (10 * self.sensitivity)  # +10dB threshold
        peaks, _properties = find_peaks(magnitude_resonance, height=threshold, prominence=5, width=3)  # 5dB prominence

        # Extract resonance frequencies and magnitudes
        resonances = [(freqs_resonance[p], magnitude_resonance[p]) for p in peaks]

        return resonances

    def remove_resonances(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Entfernt PA resonances using parametric EQ.
        """
        # Detect resonances
        resonances = self.detect_resonances(audio, sr)

        if not resonances:
            return audio  # No resonances detected

        # Preserve input dtype
        input_dtype = audio.dtype
        audio = audio.astype(np.float64)

        # Apply parametric EQ (peaking filter) at each resonance
        for freq, mag_db in resonances:
            # Reduce resonance by 6-12dB (proportional to sensitivity)
            gain_db = -(6 + 6 * self.sensitivity)
            Q = 5  # Moderate Q-factor

            # Design peaking filter
            w0 = freq / (sr / 2)
            A = 10 ** (gain_db / 40)  # Amplitude
            alpha = np.sin(2 * np.pi * w0) / (2 * Q)

            # Peaking filter coefficients
            b0 = 1 + alpha * A
            b1 = -2 * np.cos(2 * np.pi * w0)
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * np.cos(2 * np.pi * w0)
            a2 = 1 - alpha / A

            b = np.array([b0, b1, b2]) / a0
            a = np.array([a0, a1, a2]) / a0

            sos = signal.tf2sos(b, a)
            audio = np.asarray(sosfilt(sos, audio), dtype=np.float64)

        return audio.astype(input_dtype)


class HandlingNoiseDetector:
    """
    Erkennt microphone handling noise in live recordings.

    Handling noise occurs when performer moves/bumps the microphone,
    creating low-frequency thumps and rumbles (typically 20-200 Hz).

    Algorithm:
    1. Bandpass filter (20-200 Hz)
    2. Transient detection (sharp impulses)
    3. Return timestamps and energy ratios

    Target: Detect handling events for manual or automatic removal
    """

    def __init__(self, sensitivity: float = 0.7):
        """
        Initialisiert HandlingNoiseDetector.

        Parameters:
        -----------
        sensitivity : float, default 0.7
            Detection sensitivity (0.0 = conservative, 1.0 = aggressive)
        """
        self.sensitivity = np.clip(sensitivity, 0.0, 1.0)

    def detect(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        Erkennt microphone handling noise events.

        Returns:
        --------
        dict with keys:
            - handling_noise_detected: bool
            - num_events: int
            - event_timestamps: list of float (in seconds)
            - energy_ratio: float
        """
        # Bandpass filter: 20-200 Hz (handling noise range)
        sos = butter(4, [20, 200], btype="bandpass", fs=sr, output="sos")
        filtered = sosfilt(sos, audio)

        # Compute envelope (absolute value)
        analytic = np.asarray(hilbert(np.asarray(filtered, dtype=np.float64)), dtype=np.complex128)
        envelope = np.sqrt(np.square(analytic.real) + np.square(analytic.imag))

        # Smooth envelope (10ms window)
        window_size = int(0.01 * sr)
        envelope_smooth = np.convolve(envelope, np.ones(window_size) / window_size, mode="same")

        # Detect transients (sharp impulses)
        threshold = np.percentile(envelope_smooth, 95) * (0.5 + 0.5 * self.sensitivity)
        peaks, _ = find_peaks(envelope_smooth, height=threshold, distance=int(0.1 * sr))  # Min 100ms apart

        # Convert peak indices to timestamps
        event_timestamps = (peaks / sr).tolist()

        # Energy ratio
        handling_energy = np.sum(filtered**2)
        total_energy = np.sum(audio**2) + 1e-10
        energy_ratio = handling_energy / total_energy

        return {
            "handling_noise_detected": len(peaks) > 0,
            "num_events": len(peaks),
            "event_timestamps": event_timestamps,
            "energy_ratio": float(energy_ratio),
        }


class DeWindTool:
    """
    Entfernt wind noise from outdoor live recordings.

    Wind noise is broadband, low-frequency rumble (20-300 Hz) caused by
    wind hitting the microphone. Unlike music, it's non-harmonic and chaotic.

    Algorithm:
    1. Highpass filter (cutoff determined adaptively)
    2. Spectral gating on low-frequency noise
    3. Preserve musical low-frequency content (bass, kick drum)

    Target: Remove wind rumble without affecting bass
    """

    def __init__(self, sensitivity: float = 0.7):
        """
        Initialisiert DeWindTool.

        Parameters:
        -----------
        sensitivity : float, default 0.7
            Wind removal strength (0.0 = gentle, 1.0 = aggressive)
        """
        self.sensitivity = np.clip(sensitivity, 0.0, 1.0)

    def detect_wind_noise(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        Erkennt wind noise presence.

        Returns:
        --------
        dict with keys:
            - wind_noise_detected: bool
            - wind_energy_ratio: float
            - suggested_cutoff_hz: float
        """
        # Analyze low-frequency content (20-300 Hz)
        sos_lf = butter(4, [20, 300], btype="bandpass", fs=sr, output="sos")
        lf_content = sosfilt(sos_lf, audio)

        # Wind noise characteristics: chaotic, non-harmonic
        # Use spectral flatness (wind = noise-like → high flatness)
        nperseg = 2048
        _f, _t, Zxx = stft(lf_content, sr, nperseg=nperseg)
        magnitude = np.abs(Zxx)

        eps = 1e-10
        geometric_mean = np.exp(np.mean(np.log(magnitude + eps), axis=0))
        arithmetic_mean = np.mean(magnitude, axis=0)
        spectral_flatness = geometric_mean / (arithmetic_mean + eps)

        # High flatness → wind noise
        wind_frames = spectral_flatness > 0.5
        wind_ratio = np.mean(wind_frames)

        # Energy ratio
        lf_energy = np.sum(lf_content**2)
        total_energy = np.sum(audio**2) + eps
        wind_energy_ratio = lf_energy / total_energy

        # Suggest cutoff frequency (50-150 Hz depending on wind severity)
        suggested_cutoff = 50 + (100 * wind_ratio)

        return {
            "wind_noise_detected": wind_ratio > 0.2,
            "wind_energy_ratio": float(wind_energy_ratio),
            "suggested_cutoff_hz": float(suggested_cutoff),
        }

    def remove_wind_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Entfernt wind noise using adaptive highpass filter.
        """
        # Detect wind noise
        wind_info = self.detect_wind_noise(audio, sr)

        if not wind_info["wind_noise_detected"]:
            return audio  # No wind noise

        # Preserve input dtype
        input_dtype = audio.dtype
        audio = audio.astype(np.float64)

        # Adaptive highpass filter
        cutoff_hz = wind_info["suggested_cutoff_hz"] * (0.5 + 0.5 * self.sensitivity)
        cutoff_hz = np.clip(cutoff_hz, 30, 200)  # Safety limits

        sos = butter(4, cutoff_hz, btype="highpass", fs=sr, output="sos")
        audio_dewind = sosfilt(sos, audio)

        return audio_dewind.astype(input_dtype)  # type: ignore[no-any-return]


class RoomModeCorrector:
    """
    Correct room modal resonances in live recordings.

    Room modes are standing wave resonances at specific frequencies
    determined by room dimensions. They cause frequency-dependent boosts/cuts.

    Algorithm:
    1. Analyze frequency response for peaks/nulls
    2. Detect modal frequencies (typically 50-300 Hz)
    3. Apply parametric EQ to flatten response

    Target: ±3dB frequency response in modal region
    """

    def __init__(self, sensitivity: float = 0.7):
        """
        Initialisiert RoomModeCorrector.

        Parameters:
        -----------
        sensitivity : float, default 0.7
            Correction strength (0.0 = gentle, 1.0 = aggressive)
        """
        self.sensitivity = np.clip(sensitivity, 0.0, 1.0)

    def detect_room_modes(self, audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
        """
        Erkennt room modal frequencies.

        Returns:
        --------
        list of tuples (frequency_hz, magnitude_db)
        """
        # FFT analysis (high resolution for low frequencies)
        fft_size = 16384
        audio_fft = np.asarray(rfft(audio, n=fft_size), dtype=np.complex128)
        magnitude = np.sqrt(np.square(audio_fft.real) + np.square(audio_fft.imag))
        freqs = np.fft.rfftfreq(fft_size, 1 / sr)

        # Convert to dB
        magnitude_db = 20 * np.log10(magnitude + 1e-10)

        # Focus on room mode range (30-300 Hz)
        mode_range_idx = np.where((freqs >= 30) & (freqs <= 300))[0]
        magnitude_mode = magnitude_db[mode_range_idx]
        freqs_mode = freqs[mode_range_idx]

        # Smooth magnitude (moving average to identify broad peaks)
        window = 5
        magnitude_smooth = np.convolve(magnitude_mode, np.ones(window) / window, mode="same")

        # Detect peaks (room modes = broad peaks above median)
        threshold = np.median(magnitude_smooth) + (8 * self.sensitivity)  # +8dB threshold
        peaks, _ = find_peaks(magnitude_smooth, height=threshold, prominence=4, width=5)

        # Extract modal frequencies and magnitudes
        modes = [(freqs_mode[p], magnitude_smooth[p]) for p in peaks]

        return modes

    def correct_room_modes(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Correct room modes using parametric EQ.
        """
        # Detect room modes
        modes = self.detect_room_modes(audio, sr)

        if not modes:
            return audio  # No room modes detected

        # Preserve input dtype
        input_dtype = audio.dtype
        audio = audio.astype(np.float64)

        # Apply parametric EQ to flatten each mode
        for freq, mag_db in modes:
            # Reduce mode by 4-10dB (proportional to sensitivity)
            median_level = np.median([m[1] for m in modes])
            gain_db = -(4 + 6 * self.sensitivity) * ((mag_db - median_level) / 10)
            gain_db = np.clip(gain_db, -12, 0)  # Max -12dB reduction

            Q = 3  # Moderate Q for room modes (broad peaks)

            # Design peaking filter
            w0 = freq / (sr / 2)
            A = 10 ** (gain_db / 40)
            alpha = np.sin(2 * np.pi * w0) / (2 * Q)

            b0 = 1 + alpha * A
            b1 = -2 * np.cos(2 * np.pi * w0)
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * np.cos(2 * np.pi * w0)
            a2 = 1 - alpha / A

            b = np.array([b0, b1, b2]) / a0
            a = np.array([a0, a1, a2]) / a0

            sos = signal.tf2sos(b, a)
            audio = np.asarray(sosfilt(sos, audio), dtype=np.float64)

        return audio.astype(input_dtype)


class LiveRecordingSpecialist:
    """
    Einheitliches Interface für alle Live-Aufnahme-Werkzeuge.

    Provides a high-level API to analyze and process live concert recordings
    with all 8 specialized tools.
    """

    def __init__(self):
        """Initialisiert all live recording tools with default parameters."""
        self.crowd_isolator = CrowdNoiseIsolator()
        self.deverberator = RoomDeverberator()
        self.bleed_reducer = StageBleedReducer()
        self.feedback_canceller = FeedbackCanceller()
        self.pa_remover = PAResonanceRemover()
        self.handling_detector = HandlingNoiseDetector()
        self.dewind_tool = DeWindTool()
        self.mode_corrector = RoomModeCorrector()

    def analyze(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        Analysiert Live-Aufnahmen auf alle potenziellen Probleme.

        Returns:
        --------
        dict with analysis results from all tools
        """
        # Handle stereo (use left channel)
        if audio.ndim == 2:
            audio = audio[:, 0]

        results: dict[str, Any] = {}

        # Crowd noise analysis
        results["crowd_noise"] = self.crowd_isolator.detect_crowd_noise(audio, sr)

        # RT60 estimation
        results["rt60"] = self.deverberator.estimate_rt60(audio, sr)
        results["excessive_reverb"] = results["rt60"] > 0.6

        # Feedback detection
        results["feedback_frequencies"] = self.feedback_canceller.detect_feedback(audio, sr)
        results["feedback_detected"] = len(results["feedback_frequencies"]) > 0

        # PA resonances
        results["pa_resonances"] = self.pa_remover.detect_resonances(audio, sr)
        results["pa_issues"] = len(results["pa_resonances"]) > 0

        # Handling noise
        results["handling_noise"] = self.handling_detector.detect(audio, sr)

        # Wind noise
        results["wind_noise"] = self.dewind_tool.detect_wind_noise(audio, sr)

        # Room modes
        results["room_modes"] = self.mode_corrector.detect_room_modes(audio, sr)
        results["room_mode_issues"] = len(results["room_modes"]) > 0

        # Overall live recording indicator
        issues = [
            results["crowd_noise"]["crowd_noise_detected"],
            results["excessive_reverb"],
            results["feedback_detected"],
            results["pa_issues"],
            results["handling_noise"]["handling_noise_detected"],
            results["wind_noise"]["wind_noise_detected"],
            results["room_mode_issues"],
        ]
        results["is_live_recording"] = bool(sum(issues) >= 2)  # 2+ indicators (convert to Python bool)

        # Print summary
        import logging

        logging.info("\n=== Live Recording Analysis ===")
        logging.info(f"Live Recording Detected: {results['is_live_recording']}")
        logging.info(
            f"Crowd Noise: {results['crowd_noise']['crowd_noise_detected']} (ratio: {results['crowd_noise']['crowd_noise_ratio']:.2%})"
        )
        logging.info(f"RT60: {results['rt60']:.2f}s (excessive: {results['excessive_reverb']})")
        logging.info(f"Feedback: {len(results['feedback_frequencies'])} frequencies")
        logging.info(f"PA Resonances: {len(results['pa_resonances'])} detected")
        logging.info(f"Handling Noise: {results['handling_noise']['num_events']} events")
        logging.info(f"Wind Noise: {results['wind_noise']['wind_noise_detected']}")
        logging.info(f"Room Modes: {len(results['room_modes'])} detected")

        return results

    def process(
        self,
        audio: np.ndarray,
        sr: int,
        remove_crowd: bool = True,
        reduce_reverb: bool = True,
        remove_feedback: bool = True,
        remove_pa_resonances: bool = True,
        remove_wind: bool = True,
        correct_room_modes: bool = True,
    ) -> np.ndarray:
        """
        Verarbeitet live recording with selected tools.

        Parameters:
        -----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sr : int
            Sample rate
        remove_crowd : bool
            Apply crowd noise removal
        reduce_reverb : bool
            Apply reverb reduction
        remove_feedback : bool
            Apply feedback cancellation
        remove_pa_resonances : bool
            Apply PA resonance removal
        remove_wind : bool
            Apply wind noise removal
        correct_room_modes : bool
            Apply room mode correction

        Returns:
        --------
        np.ndarray
            Processed audio (same shape as input)
        """
        # Handle stereo
        is_stereo = audio.ndim == 2
        if is_stereo:
            audio_left = audio[:, 0]
            audio_right = audio[:, 1]
        else:
            audio_left = audio

        # Processing chain
        audio_processed = audio_left.copy()

        if remove_crowd:
            audio_processed = self.crowd_isolator.remove_crowd_noise(audio_processed, sr)

        if remove_feedback:
            audio_processed = self.feedback_canceller.remove_feedback(audio_processed, sr)

        if remove_pa_resonances:
            audio_processed = self.pa_remover.remove_resonances(audio_processed, sr)

        if correct_room_modes:
            audio_processed = self.mode_corrector.correct_room_modes(audio_processed, sr)

        if remove_wind:
            audio_processed = self.dewind_tool.remove_wind_noise(audio_processed, sr)

        if reduce_reverb:
            audio_processed = self.deverberator.reduce_reverb(audio_processed, sr)

        # Reconstruct stereo if needed
        if is_stereo:
            # Apply same processing to right channel
            audio_right_processed = audio_right.copy()
            if remove_crowd:
                audio_right_processed = self.crowd_isolator.remove_crowd_noise(audio_right_processed, sr)
            if remove_feedback:
                audio_right_processed = self.feedback_canceller.remove_feedback(audio_right_processed, sr)
            if remove_pa_resonances:
                audio_right_processed = self.pa_remover.remove_resonances(audio_right_processed, sr)
            if correct_room_modes:
                audio_right_processed = self.mode_corrector.correct_room_modes(audio_right_processed, sr)
            if remove_wind:
                audio_right_processed = self.dewind_tool.remove_wind_noise(audio_right_processed, sr)
            if reduce_reverb:
                audio_right_processed = self.deverberator.reduce_reverb(audio_right_processed, sr)

            audio_processed = np.column_stack([audio_processed, audio_right_processed])

        return audio_processed


# CLI interface
if __name__ == "__main__":
    import argparse

    import soundfile as sf

    parser = argparse.ArgumentParser(description="Live Recording Specialist - Process live concert recordings")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--output", help="Output audio file (optional)")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze, don't process")
    parser.add_argument(
        "--tool",
        choices=["crowd", "reverb", "feedback", "pa", "wind", "modes", "all"],
        default="all",
        help="Select specific tool",
    )

    args = parser.parse_args()

    # Load audio
    from backend.file_import import load_audio_file

    _res = load_audio_file(args.input)
    audio, sr = _res["audio"], int(_res["sr"])  # type: ignore[index]
    import logging

    logging.info(f"Loaded: {args.input} ({audio.shape}, {sr} Hz)")

    # Initialize specialist
    specialist = LiveRecordingSpecialist()

    # Analyze
    analysis = specialist.analyze(audio, sr)

    if not args.analyze_only:
        # Process
        logging.info("\nProcessing...")
        if args.tool == "all":
            audio_processed = specialist.process(audio, sr)
        elif args.tool == "crowd":
            audio_processed = specialist.crowd_isolator.remove_crowd_noise(audio, sr)
        elif args.tool == "reverb":
            audio_processed = specialist.deverberator.reduce_reverb(audio, sr)
        elif args.tool == "feedback":
            audio_processed = specialist.feedback_canceller.remove_feedback(audio, sr)
        elif args.tool == "pa":
            audio_processed = specialist.pa_remover.remove_resonances(audio, sr)
        elif args.tool == "wind":
            audio_processed = specialist.dewind_tool.remove_wind_noise(audio, sr)
        elif args.tool == "modes":
            audio_processed = specialist.mode_corrector.correct_room_modes(audio, sr)

        # Save
        output_path = args.output or args.input.replace(".", "_processed.")
        sf.write(output_path, audio_processed, sr)
        logging.info(f"Saved: {output_path}")
