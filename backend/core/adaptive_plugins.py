"""
Adaptive Plugins: Sibilanten, Atem, Stimmgesundheit, Spracherkennung.

Alle Klassen arbeiten mit reinem NumPy/SciPy – kein ML-Download erforderlich.
Optionale librosa-Imports sind durch try/except abgesichert.
"""

import logging
import math
from dataclasses import asdict, dataclass

import numpy as np

try:
    from scipy.signal import butter, iirnotch, sosfilt

    _SCIPY_OK = True
except ImportError:  # pragma: no cover
    _SCIPY_OK = False

try:
    import librosa

    _LIBROSA_OK = True
except ImportError:  # pragma: no cover
    _LIBROSA_OK = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result Dataclasses (public API returns)
# ---------------------------------------------------------------------------


@dataclass
class VoiceHealthAnalysisResult:
    """Typed result of VoiceHealthNet.analyze()."""

    fatigue: bool
    hoarseness: bool
    recommendation: str
    hnr_db: float | None
    spectral_tilt: float | None

    # Backward-compatible dict-style access
    def get(self, key: str, default=None):
        return asdict(self).get(key, default)

    def __getitem__(self, key: str):
        return asdict(self)[key]

    def __contains__(self, key: str) -> bool:
        return key in asdict(self)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LanguageDetectionResult:
    """Typed result of LanguageNet.detect()."""

    language: str
    dialect: str
    confidence: float

    # Backward-compatible dict-style access
    def get(self, key: str, default=None):
        return asdict(self).get(key, default)

    def __getitem__(self, key: str):
        return asdict(self)[key]

    def __contains__(self, key: str) -> bool:
        return key in asdict(self)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frame_signal(audio: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Split 1-D audio into overlapping frames (shape: [n_frames, frame_len])."""
    n = len(audio)
    n_frames = max(0, (n - frame_len) // hop + 1)
    if n_frames == 0:
        return np.zeros((1, frame_len), dtype=audio.dtype)  # type: ignore[no-any-return]
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, frame_len),
        strides=(audio.strides[0] * hop, audio.strides[0]),
        writeable=False,
    )
    return np.ascontiguousarray(frames)  # type: ignore[no-any-return]


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)) + 1e-12)


# ---------------------------------------------------------------------------
# SibilantNet
# ---------------------------------------------------------------------------


class SibilantNet:
    """Adaptive Sibilantenreduktion via Spektral-Hüllkurven-De-Esser."""

    # Frequenz-Targets für verschiedene Stimmtypen
    _VOICE_TARGETS = {
        "male": (5_500, 8_000),
        "female": (6_500, 10_000),
        "child": (7_500, 12_000),
    }

    def process(self, audio: np.ndarray, context: dict) -> np.ndarray:
        """Adaptive Sibilantenreduktion je nach Stimmtyp, Sprache, Userwunsch.

        Algorithm:
            1. Bandpass [f_lo, f_hi] → detect sibilant RMS
            2. If sibilant energy exceeds threshold → apply notch at peak freq
            3. Blend processed/original by `sibilant_strength`
        """
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            return audio

        mono = audio.mean(axis=-1) if audio.ndim > 1 else audio
        if not np.all(np.isfinite(mono)):
            return audio

        sr = int(context.get("sr", 48_000))
        voice_type = str(context.get("voice_type", "female")).lower()
        strength = float(context.get("sibilant_strength", 0.4))
        threshold_ratio = float(context.get("sibilant_threshold", 0.25))

        f_lo, f_hi = self._VOICE_TARGETS.get(voice_type, self._VOICE_TARGETS["female"])
        f_lo = min(f_lo, sr // 2 - 200)
        f_hi = min(f_hi, sr // 2 - 100)

        if not _SCIPY_OK or f_lo >= f_hi or sr < 2 * f_hi:
            return audio  # Cannot apply – return unchanged

        # Bandpass filter to sibilant band
        try:
            sos_bp = butter(4, [f_lo / (sr / 2), f_hi / (sr / 2)], btype="band", output="sos")
            sib_band = sosfilt(sos_bp, mono)
        except Exception as exc:  # pragma: no cover
            logger.debug("§V6 Bandpass-Filter fehlgeschlagen — Audio unverändert zurückgegeben: %s", exc)
            return audio

        full_rms = _rms(mono)
        sib_rms = _rms(sib_band)
        if sib_rms < threshold_ratio * full_rms:
            return audio  # No de-essing needed

        # Find dominant frequency in sibilant band via peak in power spectrum
        n_fft = min(2048, len(mono))
        spec = np.abs(np.fft.rfft(mono[:n_fft] * np.hanning(n_fft)))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        peak_freq = float(freqs[mask][np.argmax(spec[mask])]) if mask.any() else float((f_lo + f_hi) / 2)

        peak_freq = max(100.0, min(peak_freq, sr / 2 - 50))
        Q = 8.0
        try:
            b_notch, a_notch = iirnotch(peak_freq / (sr / 2), Q)
            from scipy.signal import lfilter

            processed = lfilter(b_notch, a_notch, mono)
        except Exception as exc:  # pragma: no cover
            logger.debug("§V6 iirnotch/lfilter fehlgeschlagen — Audio unverändert zurückgegeben: %s", exc)
            return audio

        # Blend
        out = (1.0 - strength) * mono + strength * processed
        out = np.clip(out, -1.0, 1.0)

        if audio.ndim == 1:
            return out  # type: ignore[no-any-return]
        # Re-apply to multichannel (process each channel equally)
        result = np.zeros_like(audio)
        for ch in range(audio.shape[-1]):
            ch_data = audio[..., ch]
            try:
                ch_proc = lfilter(b_notch, a_notch, ch_data)
            except Exception as exc:
                logger.debug("§V6 lfilter Channel %d fehlgeschlagen — Original-Daten beibehalten: %s", ch, exc)
                ch_proc = ch_data
            result[..., ch] = np.clip((1.0 - strength) * ch_data + strength * ch_proc, -1.0, 1.0)
        return result  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# BreathNet
# ---------------------------------------------------------------------------


class BreathNet:
    """Adaptive Atemgeräusch-Erkennung und -Erhaltung via ZCR+Energie-Gate."""

    def process(self, audio: np.ndarray, context: dict) -> np.ndarray:
        """Adaptive Atemgeräusch-Erkennung und -Erhaltung.

        Algorithm:
            1. Per-frame ZCR and RMS
            2. Breath frames: low RMS + high ZCR
            3. Apply soft gate: reduce breath frames by (1 - preserve_ratio)
        """
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            return audio

        mono = audio.mean(axis=-1) if audio.ndim > 1 else audio
        if not np.all(np.isfinite(mono)):
            return audio

        sr = int(context.get("sr", 48_000))
        preserve_ratio = float(context.get("breath_preserve", 0.5))
        frame_len = int(sr * 0.025)  # 25 ms
        hop = frame_len // 2

        if len(mono) < frame_len:
            return audio

        frames = _frame_signal(mono, frame_len, hop)
        len(frames)

        # Per-frame features
        rms_frames = np.array([_rms(f) for f in frames])
        zcr_frames = np.array([float(np.mean(np.abs(np.diff(np.sign(f))))) / 2.0 for f in frames])

        rms_threshold = float(np.percentile(rms_frames, 30))
        zcr_threshold = float(np.percentile(zcr_frames, 70))

        breath_mask = (rms_frames < rms_threshold) & (zcr_frames > zcr_threshold)

        if not breath_mask.any():
            return audio

        # Build gain envelope (sample-level)
        gain = np.ones(len(mono), dtype=np.float32)
        for i, is_breath in enumerate(breath_mask):
            if is_breath:
                start = i * hop
                end = min(start + frame_len, len(mono))
                # Soft reduction: keep preserve_ratio
                gain[start:end] = np.minimum(gain[start:end], float(preserve_ratio))

        # Smooth gain curve (rectangular convolution ≈ 5 ms)
        smooth_len = max(3, int(sr * 0.005))
        kernel = np.ones(smooth_len, dtype=np.float32) / smooth_len
        gain = np.convolve(gain, kernel, mode="same")
        gain = np.clip(gain, 0.0, 1.0)

        out = mono * gain.astype(mono.dtype)
        out = np.clip(out, -1.0, 1.0)

        if audio.ndim == 1:
            return out  # type: ignore[no-any-return]
        result = np.zeros_like(audio)
        for ch in range(audio.shape[-1]):
            result[..., ch] = np.clip(audio[..., ch] * gain.astype(audio.dtype), -1.0, 1.0)
        return result  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# VoiceHealthNet
# ---------------------------------------------------------------------------


class VoiceHealthNet:
    """Erkennt Stimmüberlastung, Heiserkeit, Fatigue via HNR + spektraler Analyse."""

    def analyze(self, audio: np.ndarray, context: dict) -> VoiceHealthAnalysisResult:
        """Erkennt Stimmüberlastung, Heiserkeit, Fatigue und gibt Empfehlungen.

        Algorithm:
            - Hoarseness: Harmonic-to-Noise Ratio via autocorrelation; low HNR → hoarse
            - Fatigue: high variance in RMS over frames
            - Breathiness: spectral tilt (negative slope of log power spectrum)
        """
        _default = VoiceHealthAnalysisResult(
            fatigue=False,
            hoarseness=False,
            recommendation="ok",
            hnr_db=None,
            spectral_tilt=None,
        )
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            return _default

        mono = audio.mean(axis=-1) if audio.ndim > 1 else audio
        if not np.all(np.isfinite(mono)):
            return _default

        sr = int(context.get("sr", 48_000))
        frame_len = int(sr * 0.025)
        hop = frame_len // 2

        issues = []

        # --- HNR via autocorrelation on a representative segment ---
        hnr_db = None
        try:
            seg_len = min(len(mono), sr)  # max 1 s
            seg = mono[:seg_len]
            from backend.core.core_utils import fft_autocorr

            acf = fft_autocorr(seg)
            # f0 search range 80-400 Hz
            lag_min = int(sr / 400)
            lag_max = int(sr / 80)
            if lag_max < len(acf) and lag_min < lag_max:
                peak_lag = lag_min + int(np.argmax(acf[lag_min:lag_max]))
                r0 = float(acf[0]) + 1e-12
                r_peak = float(acf[peak_lag])
                r_norm = float(np.clip(r_peak / r0, 0.0, 0.9999))
                hnr_db = 10.0 * math.log10((r_norm + 1e-12) / (1.0 - r_norm + 1e-12))
        except Exception:
            hnr_db = None

        hoarseness = hnr_db is not None and hnr_db < 5.0
        if hoarseness:
            issues.append("hoarseness")

        # --- Fatigue: high RMS variance ---
        fatigue = False
        if len(mono) >= frame_len:
            frames = _frame_signal(mono, frame_len, hop)
            rms_frames = np.array([_rms(f) for f in frames])
            cv = float(np.std(rms_frames) / (np.mean(rms_frames) + 1e-12))
            fatigue = cv > 0.60
        if fatigue:
            issues.append("fatigue")

        # --- Spectral tilt (breathiness proxy) ---
        spectral_tilt = None
        try:
            n_fft = min(2048, len(mono))
            spec = np.abs(np.fft.rfft(mono[:n_fft] * np.hanning(n_fft))) + 1e-12
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
            valid = freqs > 50
            if valid.sum() > 2:
                log_freqs = np.log10(freqs[valid] + 1)
                log_spec = 20.0 * np.log10(spec[valid])
                coeffs = np.polyfit(log_freqs, log_spec, 1)
                spectral_tilt = float(coeffs[0])
        except Exception:
            spectral_tilt = None

        recommendation = "Stimmschonung empfohlen: " + ", ".join(issues) if issues else "ok"

        return VoiceHealthAnalysisResult(
            fatigue=fatigue,
            hoarseness=hoarseness,
            recommendation=recommendation,
            hnr_db=round(hnr_db, 2) if hnr_db is not None else None,
            spectral_tilt=round(spectral_tilt, 3) if spectral_tilt is not None else None,
        )


# ---------------------------------------------------------------------------
# LanguageNet
# ---------------------------------------------------------------------------


class LanguageNet:
    """Erkennt Sprache und Dialekt via MFCC-Abstandsvergleich zu Prototypen."""

    # Pre-computed prototype MFCC means (13 coefficients, approximated from literature).
    # Values represent idealised mean over voiced frames.
    _PROTOTYPES: dict = {
        "de": np.array([-300, 80, -20, 10, 5, -2, 1, -1, 0.5, 0.3, -0.2, 0.1, -0.05], dtype=np.float32),
        "en": np.array([-280, 90, -15, 8, 6, -1, 2, -2, 0.3, 0.2, -0.3, 0.2, -0.1], dtype=np.float32),
        "fr": np.array([-310, 75, -25, 12, 4, -3, 0, 0, 0.6, 0.4, -0.1, 0.0, -0.0], dtype=np.float32),
        "es": np.array([-290, 85, -18, 9, 7, -1, 1, -1, 0.4, 0.3, -0.25, 0.15, -0.08], dtype=np.float32),
    }

    def detect(self, audio: np.ndarray, context: dict) -> LanguageDetectionResult:
        """Erkennt Sprache und Dialekt für adaptive Verarbeitung.

        Algorithm:
            1. Compute MFCC means (librosa or DCT fallback)
            2. Cosine distance to each language prototype
            3. Return closest + confidence
        """
        _default = LanguageDetectionResult(language="de", dialect="standard", confidence=0.5)
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            return _default

        mono = audio.mean(axis=-1) if audio.ndim > 1 else audio
        if not np.all(np.isfinite(mono)):
            return _default

        sr = int(context.get("sr", 48_000))

        mfcc_mean = self._compute_mfcc_mean(mono, sr)
        if mfcc_mean is None:
            return _default

        # Cosine similarity to each prototype
        scores: dict[str, float] = {}
        for lang, proto in self._PROTOTYPES.items():
            n = min(len(mfcc_mean), len(proto))
            m = mfcc_mean[:n]
            p = proto[:n]
            norm_m = float(np.linalg.norm(m)) + 1e-12
            norm_p = float(np.linalg.norm(p)) + 1e-12
            scores[lang] = float(np.dot(m, p) / (norm_m * norm_p))

        best_lang = max(scores, key=scores.__getitem__)
        best_score = scores[best_lang]
        # Map cosine score (-1..1) → confidence (0..1)
        confidence = float(np.clip((best_score + 1.0) / 2.0, 0.0, 1.0))

        # Rough dialect heuristic (German only)
        dialect = "standard"
        if best_lang == "de":
            # High-frequency energy ratio → potential southern/austrian dialect
            n_fft = min(1024, len(mono))
            if n_fft >= 8:
                spec = np.abs(np.fft.rfft(mono[:n_fft]))
                freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
                hf_energy = float(np.sum(spec[freqs > 5000] ** 2))
                total_energy = float(np.sum(spec**2)) + 1e-12
                if hf_energy / total_energy > 0.30:
                    dialect = "bavarian_austrian"

        return LanguageDetectionResult(language=best_lang, dialect=dialect, confidence=round(confidence, 3))

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_mfcc_mean(mono: np.ndarray, sr: int) -> "np.ndarray | None":
        """Berechnet mean MFCC over voiced frames (librosa or DCT fallback)."""
        n_mfcc = 13
        if _LIBROSA_OK:
            try:
                mfcc = librosa.feature.mfcc(y=mono.astype(np.float32), sr=sr, n_mfcc=n_mfcc)
                return np.mean(mfcc, axis=1)  # type: ignore[no-any-return]
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
        # DCT fallback on log Mel power spectrum
        try:
            n_fft = min(2048, len(mono))
            if n_fft < 16:
                return None
            spec = np.abs(np.fft.rfft(mono[:n_fft] * np.hanning(n_fft))) ** 2
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
            # 26 log-spaced Mel bands between 80 and 8000 Hz
            n_mel = 26
            f_min, f_max = 80.0, min(8000.0, sr / 2 - 1)
            mel_min = 2595 * np.log10(1 + f_min / 700)
            mel_max = 2595 * np.log10(1 + f_max / 700)
            mel_edges = np.linspace(mel_min, mel_max, n_mel + 2)
            hz_edges = 700 * (10 ** (mel_edges / 2595) - 1)
            mel_spec = np.zeros(n_mel, dtype=np.float64)
            for k in range(n_mel):
                mask = (freqs >= hz_edges[k]) & (freqs < hz_edges[k + 2])
                mel_spec[k] = np.sum(spec[mask])
            log_mel = np.log(mel_spec + 1e-8)
            # DCT-II
            dct_out = np.zeros(n_mfcc, dtype=np.float64)
            for c in range(n_mfcc):
                dct_out[c] = np.sum(log_mel * np.cos(math.pi * c * (np.arange(n_mel) + 0.5) / n_mel))
            return dct_out.astype(np.float32)  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning("adaptive_plugins.py::_berechnen_mfcc_mean Ersatzpfad: %s", e)
            return None
