"""
Aurik 10.0.0 Vocal AI Enhancement - Gender-Aware Processing
=========================================================

Integration von Phase 19 (De-Esser) und Phase 42 (Vocal Enhancement)
in das AI Framework mit vollständiger Gender-Awareness.

FEATURES:
- Geschlechtsspezifische Sibilanten-Behandlung (Frauen, Männer, Kinder)
- Emotion- und authentizitätserhaltende Verarbeitung
- Intelligente Atemgeräusch-Behandlung (Erhalt natürlicher Atem)
- Formant-basierte Gender-Detection
- Adaptive Parameter für jede Geschlechts-/Altersgruppe

Author: Aurik 10.0.0 Development Team
Date: 15. Februar 2026
Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy import signal
from scipy.fft import rfftfreq

logger = logging.getLogger(__name__)

# Optional: WORLD-Vocoder für Formant-Quervalidierung (Morise et al. 2016)
try:
    import pyworld as _pw  # type: ignore[import-untyped]

    HAS_PYWORLD: bool = True
except ImportError:
    _pw = None  # type: ignore[assignment]
    HAS_PYWORLD: bool = False  # type: ignore[no-redef]


# ============================================================
# ENUMS & DATA STRUCTURES
# ============================================================


class VoiceGender(Enum):
    """Geschlechts-Klassifikation für Gesang."""

    MALE = "male"
    FEMALE = "female"
    CHILD = "child"
    ANDROGYNOUS = "androgynous"
    UNKNOWN = "unknown"


class VoiceAgeGroup(Enum):
    """Altersgruppen für adaptive Verarbeitung."""

    CHILD = "child"  # <13
    TEENAGER = "teenager"  # 13-19
    YOUNG_ADULT = "young_adult"  # 20-35
    ADULT = "adult"  # 35-55
    MATURE = "mature"  # 55-70
    SENIOR = "senior"  # >70


class EmotionPreservationMode(Enum):
    """Emotion preservation strategies."""

    MAXIMUM = "maximum"  # Minimale Eingriffe, maximale Authentizität
    BALANCED = "balanced"  # Balance zwischen Enhancement und Emotion
    TECHNICAL = "technical"  # Optimale technische Qualität
    TRANSPARENT = "transparent"  # Unsichtbare Verarbeitung


@dataclass
class VoiceCharacteristics:
    """Erkannte Stimm-Charakteristika."""

    gender: VoiceGender
    age_group: VoiceAgeGroup | None = None
    fundamental_freq: float = 0.0  # Hz (F0)
    formants: list[float] = field(default_factory=list)  # F1, F2, F3, F4
    breathiness: float = 0.0  # 0-1
    vocal_effort: float = 0.0  # 0-1 (whisper to shout)
    emotional_intensity: float = 0.0  # 0-1
    sibilance_severity: float = 0.0  # 0-1
    confidence: float = 0.0  # 0-1


@dataclass
class VocalEnhancementResult:
    """Ergebnis der Vocal-Enhancement-Verarbeitung."""

    audio: np.ndarray
    sample_rate: int
    characteristics: VoiceCharacteristics
    sibilance_reduced_db: float  # dB Reduktion
    breath_preserved_ratio: float  # 0-1 (1=vollständig erhalten)
    emotion_preservation_score: float  # 0-1 (1=perfekt erhalten)
    formant_preservation_score: float  # 0-1 (1=identisch)
    processing_applied: list[str]
    quality_improvement: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# GENDER DETECTION (AI-based)
# ============================================================


class GenderDetector:
    """
    Formant-basierte Gender Detection.

    Eigenentwicklung basierend auf:
    - Formant-Analyse (F1, F2, F3)
    - Fundamentalfrequenz (F0)
    - Spektraler Schwerpunkt
    - Harmonics-Struktur
    """

    def __init__(self, sample_rate: int = 48000):
        self.sr = sample_rate

        # Gender-spezifische Formant-Bereiche (Hz)
        # f0 ranges: Titze 1994 (singing ranges), Klatt & Klatt 1990 (speech).
        # FEMALE extended to (165, 700) to cover mezzo-soprano/alto singing (not
        # just speech 165-255 Hz); CHILD reliably distinguishable via formants
        # (smaller vocal tract → all formants higher), not via f0 alone in music.
        self.formant_ranges = {
            VoiceGender.MALE: {
                "f0": (85, 180),
                "f1": (270, 730),
                "f2": (840, 2290),
                "f3": (1690, 3010),
            },
            VoiceGender.FEMALE: {
                "f0": (165, 700),  # singing range: alto 165 Hz – soprano 700 Hz
                "f1": (310, 860),
                "f2": (920, 2790),
                "f3": (1890, 3310),
            },
            VoiceGender.CHILD: {
                "f0": (250, 600),  # children speak reliably above 250 Hz
                "f1": (370, 1030),
                "f2": (1170, 3330),
                "f3": (2590, 4990),
            },
        }

    def detect(self, audio: np.ndarray) -> VoiceCharacteristics:
        """
        Erkennt voice characteristics including gender.

        Args:
            audio: Audio signal (mono)

        Returns:
            VoiceCharacteristics with detected attributes
        """
        # Ensure mono
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        # §19.2: pYIN F0 (Mauch & Dixon 2014) — Median über voiced Frames.
        # Liefert zusätzlich die voiced-Frame-Zeitpunkte für das Formant-
        # Gating (Formanten NUR aus stimmhaften Frames → keine Instrumental-
        # Kontamination). Deterministisch (§G5).
        pyin_f0, voiced_times = self._detect_pyin_f0(audio)

        # Detect fundamental frequency (F0) — Scanning-Autokorrelation (Basis)
        fundamental_freq = self._detect_f0(audio)

        # §2.11/§19.2: pYIN-Override — pYIN gewinnt, wenn die Autokorrelation
        # kein voiced Segment fand (Intro > 3 s, Befund Elke Best) oder die
        # Abweichung > 15 % beträgt (Oktavfehler, Bass-Masking, Vibrato).
        if pyin_f0 is not None and pyin_f0 > 0:
            if fundamental_freq <= 0:
                fundamental_freq = pyin_f0
            else:
                delta = abs(pyin_f0 - fundamental_freq) / max(fundamental_freq, 1.0)
                if delta > 0.15:
                    fundamental_freq = pyin_f0

        # Detect formants — §19.2: nur aus voiced Frames (vokaltrakt-treu)
        formants = self._detect_formants(audio, voiced_times)

        # Classify gender based on F0 and formants
        gender, confidence = self._classify_gender(fundamental_freq, formants)

        # §19 Contralto-Override: tiefe Frauenstimmen mit weiblichen Formanten
        gender, confidence = self._apply_contralto_override(
            gender, confidence, fundamental_freq, formants
        )

        # Detect breathiness
        breathiness = self._detect_breathiness(audio)

        # Detect vocal effort
        vocal_effort = self._detect_vocal_effort(audio)

        # Detect emotional intensity
        emotional_intensity = self._detect_emotional_intensity(audio)

        # Detect sibilance severity
        sibilance_severity = self._detect_sibilance(audio)

        # Estimate age group
        age_group = self._estimate_age_group(fundamental_freq, formants, gender)

        return VoiceCharacteristics(
            gender=gender,
            age_group=age_group,
            fundamental_freq=fundamental_freq,
            formants=formants,
            breathiness=breathiness,
            vocal_effort=vocal_effort,
            emotional_intensity=emotional_intensity,
            sibilance_severity=sibilance_severity,
            confidence=confidence,
        )

    def _detect_f0(self, audio: np.ndarray) -> float:
        """Erkennt fundamental frequency durch scannende FFT-Autokorrelation.

        Scannt durch das Audio in 100ms-Fenstern (50ms Hop) bis ein voiced
        Segment mit zuverlässigem Autokorrelations-Peak gefunden wird.
        Vermeidet den §2.8-Bug, dass ein instrumentales Intro (erste 100ms
        ohne Stimme) die gesamte F0-Detektion blockiert.

        Kosten: O(chunks × N log N) mit N=sr*0.1, max 60 chunks ≈ 3 s Audio.
        """
        chunk_samples = max(int(self.sr * 0.1), 512)  # 100 ms
        hop_samples = chunk_samples // 2  # 50 ms Überlappung
        max_chunks = min(60, max(1, (len(audio) - chunk_samples) // hop_samples + 1))
        best_f0 = 0.0
        best_peak_height = 0.0
        for chunk_idx in range(max_chunks):
            start = chunk_idx * hop_samples
            if start + chunk_samples > len(audio):
                break
            segment = audio[start : start + chunk_samples].astype(np.float64)
            rms = float(np.sqrt(np.mean(segment**2) + 1e-12))
            if rms < 1e-6:
                continue  # Stille
            # FFT-based autocorrelation: O(N log N)
            n = len(segment)
            fft = np.fft.rfft(segment, n=2 * n)
            autocorr = np.fft.irfft(fft * np.conj(fft))[:n]
            autocorr = autocorr / (autocorr[0] + 1e-10)
            min_period = int(self.sr / 500)  # Max 500 Hz
            max_period = int(self.sr / 50)  # Min 50 Hz
            if max_period <= min_period or max_period > len(autocorr):
                continue
            autocorr_search = autocorr[min_period:max_period]
            if len(autocorr_search) < 2:
                continue
            peaks, _ = signal.find_peaks(autocorr_search, height=0.12)
            if len(peaks) == 0:
                continue
            best_peak = peaks[np.argmax(autocorr_search[peaks])]
            peak_height = float(autocorr_search[best_peak])
            period = best_peak + min_period
            f0 = float(self.sr) / float(period)
            # Plausibilitätscheck: F0 im Stimmumfang (70–800 Hz)
            if f0 < 70.0 or f0 > 800.0:
                continue
            if peak_height > best_peak_height:
                best_peak_height = peak_height
                best_f0 = f0
        return best_f0

    def _detect_pyin_f0(self, audio: np.ndarray, max_seconds: float = 30.0) -> tuple[float | None, np.ndarray | None]:
        """§19.2: pYIN F0 (Mauch & Dixon 2014) — Median über voiced Frames.

        pYIN ist ein probabilistisches Pitch-Modell mit Voicing-Confidence:
        robust gegen polyphones Material, Vibrato und lange Intros (die
        Scanning-Autokorrelation betrachtet nur die ersten ~3 s, Befund
        Elke Best 2026-09-08). Gibt (median_f0, voiced_frame_times) zurück;
        (None, None) wenn < 10 voiced Frames existieren.

        Befund Elke Best (2026-09-08, gemessen): pYIN-Voicing-Confidence liegt
        bei vollem Pop-Arrangement fast nie über 0.8 (p90 ≈ 0.2, max ≈ 0.85) —
        die Schwelle 0.8 findet daher nichts. Gestufte Schwelle (0.4 → 0.25)
        plus NaN-Filter; der Median über viele Frames bleibt dadurch robust.
        Findet das erste Fenster (0–30 s) keine Stimme (Intro), wird in der
        Track-Mitte und am Track-Ende weitergesucht (I-19.2: F0 > 0 sobald
        IRGENDWO ein voiced Segment existiert).
        """
        try:
            import librosa

            n = int(self.sr * max_seconds)
            if len(audio) <= n:
                windows: list[tuple[int, int]] = [(0, len(audio))]
            else:
                mid_start = max(0, (len(audio) - n) // 2)
                windows = [(0, n), (mid_start, mid_start + n), (len(audio) - n, len(audio))]
            for _win_start, _win_end in windows:
                seg = np.asarray(audio[_win_start:_win_end], dtype=np.float32)
                f0, _voiced_flag, voiced_prob = librosa.pyin(
                    seg, fmin=60.0, fmax=700.0, sr=self.sr, frame_length=2048, win_length=1024
                )
                for _th in (0.4, 0.25):
                    _valid = (voiced_prob > _th) & ~np.isnan(f0)
                    if int(np.count_nonzero(_valid)) > 10:
                        _hop = 512  # librosa-Default: frame_length // 4
                        _times = np.arange(len(voiced_prob), dtype=np.float64) * _hop / float(self.sr)
                        _voiced_f0 = f0[_valid]
                        return float(np.median(_voiced_f0)), _times[_valid] + _win_start / float(self.sr)
            return None, None
        except Exception as exc:
            logger.debug("pYIN F0 fehlgeschlagen (%s) — Autokorrelation bleibt Basis", exc)
            return None, None

    def _apply_contralto_override(
        self,
        gender: VoiceGender,
        confidence: float,
        fundamental_freq: float,
        formants: list[float],
    ) -> tuple[VoiceGender, float]:
        """§19 Contralto-Override (Zone 120–240 Hz, Spec 19 „Contralto-Zonen-
        Erweiterung" 2026-08-22; Befund Elke Best 2026-09-08).

        Tiefe Frauenstimmen haben F0 im männlichen Bereich, aber weibliche
        Formanten (kürzerer Vokaltrakt → höheres F1/F2). Formanten sind das
        anatomisch härtere Merkmal als F0 → F1 UND F2 weiblich-typisch bei
        F0 in der Contralto-Zone (inkl. Oktavfehler: 2×F0 in Zone) → FEMALE
        mit Confidence-Floor 0.65 (§19).
        """
        if gender == VoiceGender.FEMALE or fundamental_freq <= 0 or len(formants) < 2:
            return gender, confidence
        _zone_low, _zone_high = 120.0, 240.0
        _female_f1 = (310.0, 860.0)
        _female_f2 = (920.0, 2790.0)
        f1 = float(formants[0])
        f2 = float(formants[1])
        # Oktavfehler-Erkennung: F0 < 120 Hz, aber 2×F0 in der Zone
        _effective_f0 = 2.0 * fundamental_freq if (
            fundamental_freq < _zone_low and _zone_low <= 2.0 * fundamental_freq <= _zone_high
        ) else fundamental_freq
        if (
            _zone_low <= _effective_f0 <= _zone_high
            and _female_f1[0] <= f1 <= _female_f1[1]
            and _female_f2[0] <= f2 <= _female_f2[1]
        ):
            logger.debug(
                "Contralto-Override: F0=%.0f Hz F1=%.0f F2=%.0f → FEMALE",
                fundamental_freq,
                f1,
                f2,
            )
            return VoiceGender.FEMALE, max(float(confidence), 0.65)
        return gender, confidence

    def _detect_formants(self, audio: np.ndarray, voiced_times: np.ndarray | None = None) -> list[float]:
        """
        Detect formants using LPC (Linear Predictive Coding).

        §19.2 (2026-09-08): Wenn pYIN-voiced-Zeitpunkte übergeben werden,
        werden Formanten NUR aus stimmhaften Frames geschätzt (Gate ±12 ms um
        die Frame-Mitte). Ohne Gate kontaminiert der Instrumentalanteil den
        Frame-Durchschnitt (Befund Elke Best: F2=704 Hz statt weiblich).
        Liefert das Gate keine Frames, fällt der Schätzer auf den
        ungegateten Durchschnitt zurück (pYIN/Frame-Raster-Diskrepanz).

        Simplified implementation - in production würde man
        librosa oder praat verwenden.
        """

        # Pre-emphasis
        pre_emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

        # Spectral-Peak-basierte Formant-Schätzung (vereinfacht); order ist
        # hier nur für Frame-Kontext relevant — Produktion nutzt FormantTracker.
        # §4.4: Mindestordnung 16 (Burg-Methode via formant_tracker.py)

        # Split into frames
        frame_size = int(0.025 * self.sr)  # 25ms
        hop_size = int(0.010 * self.sr)  # 10ms

        formant_tracks = []

        # §19.2: Formant-Gating — nur Frames mit stimmhaftem Zentrum verwenden.
        _gate_active = voiced_times is not None and len(voiced_times) > 0
        _gate_tol = 0.012  # ±12 ms um die Frame-Mitte
        _gated_frames = 0

        for frame_start in range(0, len(pre_emphasized) - frame_size, hop_size):
            if _gate_active:
                frame_center = (frame_start + frame_size / 2.0) / self.sr
                if not np.any(np.abs(np.asarray(voiced_times, dtype=np.float64) - frame_center) <= _gate_tol):
                    continue
                _gated_frames += 1
            frame = pre_emphasized[frame_start : frame_start + frame_size]

            # Windowing
            window = np.hamming(len(frame))
            windowed = frame * window

            # Compute frame energy; skip silent frames
            frame_energy = np.dot(windowed, windowed)
            if frame_energy > 0:
                # Find peaks in magnitude spectrum (simplified formant detector)
                windowed_arr = np.asarray(windowed, dtype=np.float64)
                spectrum = np.abs(np.fft.rfft(windowed_arr))
                freqs = rfftfreq(len(windowed), 1 / self.sr)

                # Find peaks in 200-5000 Hz range
                valid_range = (freqs >= 200) & (freqs <= 5000)
                if np.any(valid_range):
                    spectrum_valid = spectrum[valid_range]
                    freqs_valid = freqs[valid_range]

                    peaks, _ = signal.find_peaks(spectrum_valid, distance=int(300 * len(spectrum_valid) / 5000))
                    if len(peaks) > 0:
                        formants_frame = freqs_valid[peaks][:4]  # F1-F4
                        if len(formants_frame) > 0:
                            formant_tracks.append(formants_frame)

        # Average formants across frames
        if formant_tracks:
            # Pad to same length
            max_len = max(len(f) for f in formant_tracks)
            padded = [np.pad(f, (0, max_len - len(f)), constant_values=0) for f in formant_tracks]
            avg_formants = np.mean(padded, axis=0)
            lpc_formants = [float(f) for f in avg_formants if f > 0]
        else:
            lpc_formants = []

        # §19.2: Gate leer (pYIN/Formant-Raster-Diskrepanz) → ungegateter
        # Fallback, damit die Detektion nie durch das Gate blockiert wird.
        if _gate_active and _gated_frames == 0:
            return self._detect_formants(audio, None)

        # WORLD-Vocoder-Quervalidierung (Morise et al. 2016):
        # DIO/Harvest f0 + CheapTrick-Spektralhüllkurve als unabhängige
        # Formant-Kreuzvalidierung; Abweichung LPC ↔ WORLD > 15% → WORLD-Wert bevorzugt
        if HAS_PYWORLD and _pw is not None and len(lpc_formants) > 0:
            try:
                audio_f64 = audio.astype(np.float64)
                sr_f64 = float(self.sr)
                _f0, _t = _pw.dio(audio_f64, sr_f64)
                _f0_sm = _pw.stonemask(audio_f64, _f0, _t, sr_f64)
                sp = _pw.cheaptrick(audio_f64, _f0_sm, _t, sr_f64)
                # Spektralhüllkurven-Mittelwert → Formant-Peaks extrahieren
                sp_mean = np.mean(sp, axis=0)
                freqs_world = np.linspace(0.0, sr_f64 / 2.0, len(sp_mean))
                valid = (freqs_world >= 200.0) & (freqs_world <= 5000.0)
                if np.any(valid):
                    sp_valid = sp_mean[valid]
                    fq_valid = freqs_world[valid]
                    pks, _ = signal.find_peaks(sp_valid, distance=int(300 * len(sp_valid) / 5000))
                    world_formants = [float(fq_valid[p]) for p in pks[:4]]
                    # Kreuzvalidierung: WORLD-Wert bevorzugt wenn Abweichung > 15 %
                    validated: list[float] = []
                    for i_f, lpc_f in enumerate(lpc_formants):
                        if i_f < len(world_formants):
                            world_f = world_formants[i_f]
                            deviation = abs(lpc_f - world_f) / max(lpc_f, 1.0)
                            if deviation > 0.15:
                                logger.debug(
                                    "WORLD-LPC Formant F%d Abweichung %.0f%% "
                                    "(LPC=%.0f Hz, WORLD=%.0f Hz) — WORLD-Wert bevorzugt",
                                    i_f + 1,
                                    deviation * 100.0,
                                    lpc_f,
                                    world_f,
                                )
                                validated.append(world_f)
                            else:
                                validated.append(lpc_f)
                        else:
                            validated.append(lpc_f)
                    return validated
            except Exception as exc:
                logger.debug("WORLD-Quervalidierung fehlgeschlagen (%s) — LPC-Ergebnis wird genutzt", exc)

        return lpc_formants

    def _classify_gender(self, fundamental_freq: float, formants: list[float]) -> tuple[VoiceGender, float]:
        """Classify gender based on F0 and formants."""
        if fundamental_freq == 0 or len(formants) == 0:
            return VoiceGender.UNKNOWN, 0.0

        # Score each gender
        scores = {}

        for gender, ranges in self.formant_ranges.items():
            score = 0.0
            count = 0

            # F0 score
            if ranges["f0"][0] <= fundamental_freq <= ranges["f0"][1]:
                score += 1.0
            else:
                # Penalty based on distance
                if fundamental_freq < ranges["f0"][0]:
                    distance = (ranges["f0"][0] - fundamental_freq) / ranges["f0"][0]
                else:
                    distance = (fundamental_freq - ranges["f0"][1]) / ranges["f0"][1]
                score += max(0, 1 - distance)
            count += 1

            # Formant scores
            for formant_idx, formant in enumerate(formants[:3], 1):  # F1, F2, F3
                formant_key = f"f{formant_idx}"
                if formant_key in ranges:
                    if ranges[formant_key][0] <= formant <= ranges[formant_key][1]:
                        score += 1.0
                    else:
                        # Penalty based on distance
                        if formant < ranges[formant_key][0]:
                            distance = (ranges[formant_key][0] - formant) / ranges[formant_key][0]
                        else:
                            distance = (formant - ranges[formant_key][1]) / ranges[formant_key][1]
                        score += max(0, 1 - distance * 0.5)
                    count += 1

            scores[gender] = score / count if count > 0 else 0.0

        # Best match
        best_gender = max(scores.items(), key=lambda x: x[1])

        # Tie-breaking for FEMALE vs CHILD when scores are close (< 0.05 delta):
        # F0 < 350 Hz is incompatible with a child's voice in a professional
        # musical recording. Children speak reliably above 300-400 Hz
        # (Titze 1994; Huber et al. 1999). Vintage recordings (pre-1950) with
        # a "child" voice are extremely rare in practice → prefer FEMALE.
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if (
            len(sorted_scores) >= 2
            and sorted_scores[0][0] == VoiceGender.CHILD
            and sorted_scores[1][0] == VoiceGender.FEMALE
            and (sorted_scores[0][1] - sorted_scores[1][1]) < 0.05
            and fundamental_freq < 350.0
        ):
            best_gender = sorted_scores[1]  # prefer FEMALE

        return best_gender[0], best_gender[1]

    def _detect_breathiness(self, audio: np.ndarray) -> float:
        """Erkennt breathiness via WORLD per-frame aperiodicity (primary) or HF energy ratio.

        WORLD's d4c() provides per-bin aperiodicity AP[frame, bin] in [0, 1] where
        0 = perfectly harmonic and 1 = fully aperiodic/breathy.  The mean
        sqrt-aperiodicity over voiced frames (F0 > 30 Hz) in the 1-4 kHz band is a
        direct, SNR-robust proxy for the harmonics-to-noise ratio (Yumoto et al. 1982).
        The HF energy ratio falls back when pyworld is unavailable.

        Scientific basis:
            Yumoto et al. (1982) — Harmonics-to-noise ratio as an index of the
            degree of hoarseness. J. Acoust. Soc. Am. 71(6).
            Ferrand (2002) — Harmonics-to-noise ratios in connected discourse for
            male and female speakers. J. Voice 16(1).
            Morise et al. (2016) — WORLD vocoder. IEICE Trans. A.
        """
        if HAS_PYWORLD and _pw is not None:
            try:
                audio_f64 = np.asarray(audio.flatten(), dtype=np.float64)
                sr_f64 = float(self.sr)
                world_f0, timeaxis = _pw.harvest(audio_f64, sr_f64)
                world_f0_sm = _pw.stonemask(audio_f64, world_f0, timeaxis, sr_f64)
                ap = _pw.d4c(audio_f64, world_f0_sm, timeaxis, sr_f64)  # (n_frames, bins)
                voiced = world_f0_sm > 30.0
                if np.any(voiced):
                    freq_axis = np.linspace(0.0, sr_f64 / 2.0, ap.shape[1])
                    band = (freq_axis >= 1000.0) & (freq_axis <= 4000.0)
                    if np.any(band):
                        ap_band = ap[voiced][:, band]
                        # sqrt-aperiodicity: 0=harmonic, 1=breathy; ×2 to use [0,1] output range
                        breathiness = float(np.mean(np.sqrt(np.maximum(ap_band, 0.0))))
                        return min(1.0, breathiness * 2.0)
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
        # DSP fallback: HF energy ratio (breathy voices have more noise above 3 kHz)
        hp_sos = signal.butter(4, 3000, "high", fs=self.sr, output="sos")
        hf_signal = signal.sosfilt(hp_sos, audio)
        hf_energy: float = float(np.sum(hf_signal**2))
        total_energy: float = float(np.sum(audio**2))
        breathiness = hf_energy / (total_energy + 1e-10)
        return min(1.0, breathiness * 5)  # type: ignore[no-any-return]

    def _detect_vocal_effort(self, audio: np.ndarray) -> float:
        """Erkennt vocal effort (whisper to shout)."""
        # RMS level as proxy
        rms = np.sqrt(np.mean(audio**2))

        # Normalize (assuming -60dB to 0dB range)
        db = 20 * np.log10(rms + 1e-10)
        effort = (db + 60) / 60  # 0-1 range

        return np.clip(effort, 0, 1)  # type: ignore[no-any-return]

    def _detect_emotional_intensity(self, audio: np.ndarray) -> float:
        """
        Erkennt emotional intensity.

        High emotion correlates with:
        - Higher pitch variation
        - Greater dynamic range
        - More energy in higher harmonics
        """
        # Pitch variation
        chunks = np.array_split(audio, 10)
        f0s = [self._detect_f0(chunk) for chunk in chunks if len(chunk) > 1000]
        f0s = [f0 for f0 in f0s if f0 > 0]

        pitch_variation = np.std(f0s) / (np.mean(f0s) + 1e-10) if len(f0s) > 1 else 0.0

        # Dynamic range
        dynamic_range: float = float(np.max(np.abs(audio)) - np.min(np.abs(audio)))

        # Combine metrics
        intensity = (pitch_variation * 10 + dynamic_range) / 2
        return float(min(1.0, intensity))

    def _detect_sibilance(self, audio: np.ndarray) -> float:
        """Erkennt sibilance severity via frame-based peak analysis (6-12 kHz).

        Uses the 95th-percentile frame energy ratio so that short, intense
        sibilant bursts are detected even when they occupy only a small fraction
        of the total audio duration.
        """
        sib_sos = signal.butter(4, [6000, 12000], "band", fs=self.sr, output="sos")
        sibilant_signal = signal.sosfilt(sib_sos, audio)

        # Frame-based analysis: 20 ms frames, 10 ms hop
        frame_size = int(0.02 * self.sr)
        hop_size = frame_size // 2

        if len(audio) < frame_size:
            # Very short audio: fall back to global ratio
            sib_e = np.mean(sibilant_signal**2)
            total_e = np.mean(audio**2) + 1e-10
            return float(min(1.0, (sib_e / total_e) * 10))  # type: ignore[arg-type]

        n_frames = (len(audio) - frame_size) // hop_size + 1
        # Vectorised frame extraction
        idx = np.arange(n_frames)[:, None] * hop_size + np.arange(frame_size)
        frames_sib = sibilant_signal[idx]
        frames_total = audio[idx]

        sib_energy = np.mean(frames_sib**2, axis=1)
        total_energy = np.mean(frames_total**2, axis=1) + 1e-10
        ratios = sib_energy / total_energy

        # 95th percentile highlights concentrated sibilant bursts.
        # Multiplier ×8 ensures short, intense bursts (even in mixed signals)
        # reliably exceed the 0.3 detection threshold without adding false
        # positives on pure harmonic voices (which have no energy above 6 kHz).
        peak_ratio = float(np.percentile(ratios, 95))
        return float(min(1.0, peak_ratio * 8))

    def _estimate_age_group(
        self, fundamental_freq: float, _formants: list[float], gender: VoiceGender
    ) -> VoiceAgeGroup | None:
        """Schätzt age group based on voice characteristics."""
        if fundamental_freq == 0:
            return None

        # Child: Very high F0
        if fundamental_freq > 250:
            return VoiceAgeGroup.CHILD

        # Teenager: Transitional
        if 200 < fundamental_freq < 250:
            return VoiceAgeGroup.TEENAGER

        # Adult ranges depend on gender
        if gender == VoiceGender.MALE:
            if 120 < fundamental_freq < 140:
                return VoiceAgeGroup.YOUNG_ADULT
            elif 110 < fundamental_freq < 130:
                return VoiceAgeGroup.ADULT
            else:
                return VoiceAgeGroup.MATURE
        elif gender == VoiceGender.FEMALE:
            if 200 < fundamental_freq < 220:
                return VoiceAgeGroup.YOUNG_ADULT
            elif 190 < fundamental_freq < 210:
                return VoiceAgeGroup.ADULT
            else:
                return VoiceAgeGroup.MATURE

        return VoiceAgeGroup.ADULT  # Default


# ============================================================
# GENDER-AWARE DE-ESSER
# ============================================================


class GenderAwareDeEsser:
    """
    Gender-adaptive sibilance reduction.

    Kombiniert:
    - Open Source: scipy signal processing
    - Eigenentwicklung: Gender-adaptive parameters
    - Eigenentwicklung: Emotion preservation
    """

    def __init__(self, sample_rate: int = 48000):
        self.sr = sample_rate
        self.gender_detector = GenderDetector(sample_rate=sample_rate)

        # Gender-specific de-essing parameters
        self.deess_params = {
            VoiceGender.MALE: {
                "freq_range": (5000, 10000),  # Lower sibilance region
                "threshold_db": -25,
                "ratio": 3.0,
                "attack_ms": 1.0,
                "release_ms": 50.0,
            },
            VoiceGender.FEMALE: {
                "freq_range": (6000, 12000),  # Higher sibilance region
                "threshold_db": -23,
                "ratio": 2.5,
                "attack_ms": 0.5,
                "release_ms": 40.0,
            },
            VoiceGender.CHILD: {
                "freq_range": (7000, 14000),  # Highest sibilance region
                "threshold_db": -20,
                "ratio": 2.0,
                "attack_ms": 0.3,
                "release_ms": 30.0,
            },
        }

    def process(
        self,
        audio: np.ndarray,
        characteristics: VoiceCharacteristics | None = None,
        emotion_mode: EmotionPreservationMode = EmotionPreservationMode.BALANCED,
    ) -> tuple[np.ndarray, float]:
        """
        Gender-aware de-essing.

        Args:
            audio: Input audio
            characteristics: Pre-detected characteristics (optional)
            emotion_mode: Emotion preservation mode

        Returns:
            (processed_audio, reduction_db)
        """
        # Detect characteristics if not provided
        if characteristics is None:
            characteristics = self.gender_detector.detect(audio)

        # Get gender-specific parameters
        gender = characteristics.gender
        if gender not in self.deess_params:
            gender = VoiceGender.FEMALE  # Default fallback

        params = self.deess_params[gender].copy()

        # Adjust for emotion preservation mode
        if emotion_mode == EmotionPreservationMode.MAXIMUM:
            params["ratio"] *= 0.5  # type: ignore[operator]  # Less aggressive
            params["threshold_db"] -= 3  # type: ignore[operator]  # Higher threshold
        elif emotion_mode == EmotionPreservationMode.TECHNICAL:
            params["ratio"] *= 1.5  # type: ignore[operator]  # More aggressive
            params["threshold_db"] += 3  # type: ignore[operator]  # Lower threshold

        # Apply de-essing
        processed, reduction_db = self._apply_deessing(audio, params)

        return processed, reduction_db

    def _apply_deessing(self, audio: np.ndarray, params: dict) -> tuple[np.ndarray, float]:
        """Wendet an: frequency-specific compression for de-essing."""
        # Extract sibilance band
        freq_low, freq_high = params["freq_range"]
        deess_sos = signal.butter(4, [freq_low, freq_high], "band", fs=self.sr, output="sos")
        # sosfiltfilt (zero-phase) required: sibilant_band is subtracted from audio in recombination;
        # causal sosfilt would introduce group delay → timing skew → Pegelexplosion (§2.51, V11)
        sibilant_band = signal.sosfiltfilt(deess_sos, audio)

        # Detect sibilant regions (RMS in chunks)
        chunk_size = int(0.01 * self.sr)  # 10ms

        gain = np.ones(len(audio))
        reduction_db_total = 0.0
        reduction_count = 0

        for chunk_start in range(0, len(audio) - chunk_size, chunk_size):
            chunk = sibilant_band[chunk_start : chunk_start + chunk_size]
            rms = np.sqrt(np.mean(chunk**2))
            db = 20 * np.log10(rms + 1e-10)

            # Apply compression if above threshold
            if db > params["threshold_db"]:
                excess_db = db - params["threshold_db"]
                reduction = excess_db * (1 - 1 / params["ratio"])
                gain_linear = 10 ** (-reduction / 20)
                gain[chunk_start : chunk_start + chunk_size] = gain_linear

                reduction_db_total += reduction
                reduction_count += 1

        # Apply gain to sibilant band only
        sibilant_reduced = sibilant_band * gain

        # Subtract original and add reduced
        processed_audio = audio - sibilant_band + sibilant_reduced

        # Average reduction
        avg_reduction = reduction_db_total / reduction_count if reduction_count > 0 else 0.0

        return processed_audio, avg_reduction


# ============================================================
# BREATH PRESERVING PROCESSOR
# ============================================================


class BreathPreservingProcessor:
    """
    Intelligent breath processing that preserves natural breathing.

    Eigenentwicklung:
    - Unterscheidet zwischen künstlerischen und störenden Atemgeräuschen
    - Erhält emotionale Authentizität
    - Adaptive Reduktion basierend auf Kontext
    """

    def __init__(self, sample_rate: int = 48000):
        self.sr = sample_rate

    def process(
        self, audio: np.ndarray, characteristics: VoiceCharacteristics, preservation_ratio: float = 0.7
    ) -> tuple[np.ndarray, float]:
        """
        Verarbeitet breath with preservation.

        Args:
            audio: Input audio
            characteristics: Voice characteristics
            preservation_ratio: How much breath to preserve (0-1, 1=all)

        Returns:
            (processed_audio, preservation_ratio_actual)
        """
        preservation_ratio = float(np.clip(preservation_ratio, 0.0, 1.0))

        # §v10.118: Transient-aware breath classification (3-stage).
        # Replaces binary ZCR/energy threshold with:
        #   1. ZCR + energy → candidates
        #   2. Onset detection → excludes consonants (sharp attack)
        #   3. Spectral tilt → separates breath (falling) from sibilance (rising)
        # Result: only true breath segments are reduced, consonants/sibilance untouched.
        # Detector/Classifier liefern die künstlerische Breath-Maske — sie gilt in
        # BEIDEN Pfaden (§Artistic-Breath-Contract), nicht nur im Fallback.
        breath_mask = self._detect_breath_regions(audio, characteristics)
        artistic_breath = self._classify_breath_artistic(audio, breath_mask, characteristics)
        try:
            from backend.core.transient_breath_processor import (
                classify_breath_frame,
                process_breath_transient_aware,
            )

            processed = process_breath_transient_aware(
                audio,
                self.sr,
                breath_reduction_db=float((1.0 - preservation_ratio) * 12.0),  # Map ratio to dB
                preserve_consonants=True,
                preserve_sibilance=True,
            )
        except Exception as _tb_exc:
            logger.debug("Transient-Breath nicht verfügbar, fallback: %s", _tb_exc)
            processed = audio.copy()
        for start, end in breath_mask:
            is_artistic = artistic_breath.get((start, end), True)
            if is_artistic:
                reduction_factor = 1.0 - ((1.0 - preservation_ratio) * 0.2)
            else:
                reduction_factor = 1.0 - (1 - preservation_ratio) * 0.8
            processed[start:end] *= reduction_factor

        return processed, preservation_ratio

    def _detect_breath_regions(
        self,
        audio: np.ndarray,
        _characteristics: VoiceCharacteristics,
    ) -> list[tuple[int, int]]:
        """Erkennt breath regions."""
        # High-pass filter for breath (>1kHz)
        breath_sos = signal.butter(4, 1000, "high", fs=self.sr, output="sos")
        breath_signal = signal.sosfilt(breath_sos, audio)

        # Envelope
        analytic = signal.hilbert(breath_signal)
        analytic_arr = np.asarray(analytic, dtype=np.complex128)
        envelope = np.abs(analytic_arr)

        # Smooth
        window_size = int(0.05 * self.sr)  # 50ms
        smoothed = np.convolve(envelope, np.ones(window_size) / window_size, mode="same")

        # Threshold (low energy)
        threshold = np.percentile(smoothed, 30)

        # Find regions
        is_breath = (smoothed < threshold) & (smoothed > threshold * 0.1)

        regions = []
        in_breath = False
        start = 0

        for sample_idx, breath_detected in enumerate(is_breath):
            if breath_detected and not in_breath:
                start = sample_idx
                in_breath = True
            elif not breath_detected and in_breath:
                if sample_idx - start > int(0.05 * self.sr):  # Min 50ms
                    regions.append((start, sample_idx))
                in_breath = False

        return regions

    def _classify_breath_artistic(
        self, audio: np.ndarray, breath_mask: list[tuple[int, int]], characteristics: VoiceCharacteristics
    ) -> dict[tuple[int, int], bool]:
        """
        Classify if breath is artistic (keep) or disturbing (reduce).

        Artistic breath:
        - Pre-phrase breaths (before vocal onset)
        - Emotional breaths (high intensity)
        - Stylistic breaths (rhythmic, patterned)
        """
        classification = {}

        for start, end in breath_mask:
            # Check if before vocal phrase
            if end < len(audio) - int(0.1 * self.sr):
                next_chunk = audio[end : end + int(0.1 * self.sr)]
                next_energy = np.sqrt(np.mean(next_chunk**2))

                # If followed by high energy = pre-phrase breath
                is_prephrase = next_energy > 0.1
            else:
                is_prephrase = False

            # Check emotional context
            is_emotional = characteristics.emotional_intensity > 0.6

            # Artistic if pre-phrase or emotional
            is_artistic = is_prephrase or is_emotional

            classification[(start, end)] = is_artistic

        return classification


# ============================================================
# UNIFIED VOCAL AI ENHANCER
# ============================================================


class UnifiedVocalAIEnhancer:
    """
    Einheitliche Vokal-Verbesserung mit vollständiger KI-Integration.

    Kombiniert:
    - Gender Detection (Eigenentwicklung)
    - Gender-Aware De-Esser (Eigenentwicklung + Open Source)
    - Breath Preservation (Eigenentwicklung)
    - Formant Preservation (Eigenentwicklung)
    - Emotion Preservation (Eigenentwicklung)
    """

    def __init__(self, sample_rate: int = 48000):
        self.sr = sample_rate
        self.gender_detector = GenderDetector(sample_rate=sample_rate)
        self.deesser = GenderAwareDeEsser(sample_rate=sample_rate)
        self.breath_processor = BreathPreservingProcessor(sample_rate=sample_rate)

        logger.info("✅ Unified Vocal AI Enhancer initialisiert")

    def enhance(
        self,
        audio: np.ndarray,
        emotion_mode: EmotionPreservationMode = EmotionPreservationMode.BALANCED,
        breath_preservation: float = 0.7,
        sibilance_reduction: bool = True,
    ) -> VocalEnhancementResult:
        """
        Comprehensive vocal enhancement with full gender awareness.

        Args:
            audio: Input audio (mono or stereo)
            emotion_mode: Emotion preservation strategy
            breath_preservation: Breath preservation ratio (0-1)
            sibilance_reduction: Enable sibilance reduction

        Returns:
            VocalEnhancementResult with enhanced audio
        """
        # Ensure mono for analysis
        if audio.ndim == 2:
            audio_mono = np.mean(audio, axis=1)
            is_stereo = True
        else:
            audio_mono = audio
            is_stereo = False

        # Step 1: Detect voice characteristics
        logger.info("Step 1: Detecting voice characteristics...")
        characteristics = self.gender_detector.detect(audio_mono)

        logger.info("  Gender: %s", characteristics.gender.value)
        logger.info("  F0: %.1f Hz", characteristics.fundamental_freq)
        logger.info("  Breathiness: %.2f", characteristics.breathiness)
        logger.info("  Emotional Intensity: %.2f", characteristics.emotional_intensity)
        logger.info("  Sibilance Severity: %.2f", characteristics.sibilance_severity)

        # Process mono
        processed = audio_mono.copy()
        processing_applied = []

        # Step 2: Breath preservation
        logger.info("Step 2: Breath preservation...")
        processed, breath_ratio = self.breath_processor.process(processed, characteristics, breath_preservation)
        processing_applied.append("breath_preservation")

        # Step 3: Gender-aware de-essing
        if sibilance_reduction and characteristics.sibilance_severity > 0.3:
            logger.info("Step 3: Gender-aware de-essing...")
            processed, sibilance_db = self.deesser.process(processed, characteristics, emotion_mode)
            processing_applied.append("gender_aware_deessing")
        else:
            sibilance_db = 0.0

        # Step 4: Formant preservation check
        logger.info("Step 4: Formant preservation verification...")
        formant_score = self._compute_formant_preservation(audio_mono, processed, characteristics)

        # Step 5: Emotion preservation check
        logger.info("Step 5: Emotion preservation verification...")
        emotion_score = self._compute_emotion_preservation(audio_mono, processed, characteristics)

        # Apply to stereo if needed
        if is_stereo:
            # §v10.117: Stereo-bewusste Mid/Side-Verarbeitung
            # Statt einfachem ratio = processed / mono (zerstört Stereo-Bild),
            # Mid/Side-Transformation: bearbeite nur M-Kanal, S-Kanal unberührt.
            try:
                from backend.core.stereo_aware_vocal_processor import (
                    compute_stereo_preservation_score,
                    process_vocal_mid_side,
                )

                result_audio = process_vocal_mid_side(
                    audio,
                    lambda m: processed,  # Apply mono processing to mid channel
                    side_preservation=1.0,  # Keep stereo width unchanged
                )

                # Verify stereo preservation
                _stereo_score = compute_stereo_preservation_score(audio, result_audio)
                logger.debug(
                    "Mid/Side: corr_diff=%.3f width_ratio=%.2f balance_shift=%.1f dB",
                    _stereo_score["correlation_diff"],
                    _stereo_score["width_ratio"],
                    _stereo_score["balance_shift_db"],
                )
            except Exception as _ms_exc:
                logger.debug("Mid/Side nicht verfügbar, fallback auf ratio: %s", _ms_exc)
                # Fallback to original ratio-based processing
                ratio = processed / (audio_mono + 1e-10)
                ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
                result_audio = audio * ratio[:, np.newaxis]
        else:
            result_audio = processed

        # Compute quality improvement
        quality_improvement = self._compute_quality_improvement(audio, result_audio)

        logger.info("✅ Vocal enhancement vollstaendig!")

        return VocalEnhancementResult(
            audio=result_audio,
            sample_rate=self.sr,
            characteristics=characteristics,
            sibilance_reduced_db=sibilance_db,
            breath_preserved_ratio=breath_ratio,
            emotion_preservation_score=emotion_score,
            formant_preservation_score=formant_score,
            processing_applied=processing_applied,
            quality_improvement=quality_improvement,
            metadata={
                "emotion_mode": emotion_mode.value,
                "gender": characteristics.gender.value,
            },
        )

    def _compute_formant_preservation(
        self, _original: np.ndarray, processed: np.ndarray, characteristics: VoiceCharacteristics
    ) -> float:
        """Berechnet how well formants are preserved."""
        # Re-detect formants in processed audio
        processed_chars = self.gender_detector.detect(processed)

        if not characteristics.formants or not processed_chars.formants:
            return 1.0  # No formants detected

        # Compare first 3 formants
        original_f = characteristics.formants[:3]
        processed_f = processed_chars.formants[:3]

        # Compute correlation — NaN-safe: formant arrays can be constant (all zeros
        # when no formants detected) → np.corrcoef returns NaN → max(0, NaN) = 0
        # (wrong; identical/constant formants = perfect preservation → should return 1.0)
        if len(original_f) == len(processed_f):
            _of, _pf = np.asarray(original_f, dtype=np.float64), np.asarray(processed_f, dtype=np.float64)
            _so, _sp = float(np.std(_of)), float(np.std(_pf))
            if _so < 1e-9 or _sp < 1e-9:
                correlation = 1.0 if np.allclose(_of, _pf) else 0.5
            else:
                _raw = float(np.dot(_of - np.mean(_of), _pf - np.mean(_pf)) / (len(_of) * _so * _sp + 1e-12))
                correlation = float(max(-1.0, min(1.0, _raw)))
            return max(0.0, correlation)

        # Compute relative difference
        min_len = min(len(original_f), len(processed_f))
        if min_len > 0:
            diff = np.abs(np.array(original_f[:min_len]) - np.array(processed_f[:min_len]))
            rel_diff = diff / (np.array(original_f[:min_len]) + 1e-10)
            preservation = float(1.0 - np.mean(rel_diff))
            return float(max(0.0, preservation))

        return 1.0

    def _compute_emotion_preservation(
        self, _original: np.ndarray, processed: np.ndarray, characteristics: VoiceCharacteristics
    ) -> float:
        """Berechnet emotion preservation score."""
        # Re-detect emotion in processed audio
        processed_chars = self.gender_detector.detect(processed)

        # Compare emotional intensity
        original_emotion = characteristics.emotional_intensity
        processed_emotion = processed_chars.emotional_intensity

        # Should be similar
        diff = abs(original_emotion - processed_emotion)
        preservation = 1 - diff

        return max(0, preservation)

    def _compute_quality_improvement(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Schätzt quality improvement."""
        # Simple metric: high-frequency clarity improvement
        clarity_sos = signal.butter(4, [3000, 8000], "band", fs=self.sr, output="sos")

        original_clarity = signal.sosfilt(clarity_sos, original.flatten() if original.ndim == 2 else original)
        processed_clarity = signal.sosfilt(clarity_sos, processed.flatten() if processed.ndim == 2 else processed)

        original_clarity_rms = np.sqrt(np.mean(original_clarity**2))
        processed_clarity_rms = np.sqrt(np.mean(processed_clarity**2))

        improvement = (processed_clarity_rms - original_clarity_rms) / (original_clarity_rms + 1e-10)

        return float(improvement)


# ============================================================
# MAIN ENTRY POINT FOR TESTING
# ============================================================


def _demo() -> None:
    """Führt aus: a local smoke demo for manual development."""
    logger.debug("Testing Aurik 10.0.0 Vocal AI Enhancement...")
    logger.debug("=" * 70)

    # Generate test vocal signal
    sr = 48000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))

    # Simulate voice: fundamental + harmonics + formants
    demo_f0 = 220  # Female/child range
    vocal = np.zeros_like(t)

    # Add harmonics
    for harmonic_idx in range(1, 8):
        vocal += (1 / harmonic_idx) * np.sin(2 * np.pi * demo_f0 * harmonic_idx * t)

    # Add sibilance (8 kHz burst)
    demo_sibilance = np.zeros_like(t)
    demo_sibilance[int(0.5 * sr) : int(0.52 * sr)] = 0.3 * np.random.randn(int(0.02 * sr))
    demo_sibilance[int(1.0 * sr) : int(1.02 * sr)] = 0.3 * np.random.randn(int(0.02 * sr))

    # High-pass filter sibilance
    demo_sos = signal.butter(4, 6000, "high", fs=sr, output="sos")
    demo_sibilance = signal.sosfiltfilt(demo_sos, demo_sibilance)

    vocal = vocal * 0.3 + demo_sibilance

    # Add breath (low-level noise between phrases)
    demo_breath = np.random.randn(len(t)) * 0.02
    vocal += demo_breath

    # Normalize
    _peak_p99 = float(np.percentile(np.abs(vocal), 99.9)) if vocal.size > 0 else 0.0
    vocal = vocal / _peak_p99 * 0.7 if _peak_p99 > 1e-8 else vocal

    # Initialize enhancer
    logger.debug("\ninitialisiere Vocal AI Enhancer...")
    enhancer = UnifiedVocalAIEnhancer(sample_rate=sr)

    # Test enhancement
    logger.debug("\nTesting Vocal Enhancement...")
    demo_result = enhancer.enhance(
        vocal, emotion_mode=EmotionPreservationMode.BALANCED, breath_preservation=0.7, sibilance_reduction=True
    )

    logger.debug("\n%s", "=" * 70)
    logger.debug("RESULTS:")
    logger.debug("%s", "=" * 70)
    logger.debug("Gender erkannt: %s", demo_result.characteristics.gender.value)
    if demo_result.characteristics.age_group:
        logger.debug("Age Group: %s", demo_result.characteristics.age_group.value)
    logger.debug("F0: %.1f Hz", demo_result.characteristics.fundamental_freq)
    logger.debug("Formants: %s", [f"{f:.0f} Hz" for f in demo_result.characteristics.formants])
    logger.debug("Sibilance Reduced: %.1f dB", demo_result.sibilance_reduced_db)
    logger.debug("Breath Preserved: %.1f%%", demo_result.breath_preserved_ratio)
    logger.debug("Emotion Preservation: %.1f%%", demo_result.emotion_preservation_score)
    logger.debug("Formant Preservation: %.1f", demo_result.formant_preservation_score)
    logger.debug("Quality Improvement: %+.2f", demo_result.quality_improvement)
    logger.debug("\nProcessing angewendet:")
    for proc in demo_result.processing_applied:
        logger.debug("  ✓ %s", proc)

    logger.debug("\n%s", "=" * 70)
    logger.debug("✅ Vocal AI Enhancement Test vollstaendig!")


if __name__ == "__main__":
    _demo()


# ---------------------------------------------------------------------------
# Spec-Konformitäts-Alias  (§2.8 SCHRITTE_ZUR_MUSIKALISCHEN_EXZELLENZ.md)
# ---------------------------------------------------------------------------
VocalAIEnhancement = UnifiedVocalAIEnhancer
