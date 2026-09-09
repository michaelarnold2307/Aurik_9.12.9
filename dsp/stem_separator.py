"""
stem_separator.py - Audio Source Separation for Stem Export (GAP #43)

Separates mixed audio into individual stems:
- Vocals
- Drums
- Bass
- Other/Accompaniment

Provides two separation backends:
1. **Spectral Separation** (always available, fast, basic quality)
2. **ML-based Separation** (optional, Demucs/Banquet integration, high quality)

Author: AURIK Development Team
Version: 2.0.0 (GAP #43 Implementation)
Date: 9. Februar 2026
"""

import logging
import warnings
from typing import Any

import numpy as np
from scipy.signal import istft, stft

_logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# Check for optional ML backends
DEMUCS_AVAILABLE = False
BANQUET_AVAILABLE = False

try:
    pass

    # Note: Actual Demucs import would be here in production
    # from demucs import pretrained
    # from demucs.apply import apply_model
    # DEMUCS_AVAILABLE = True
except ImportError:
    _logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

try:
    from banquet import Banquet  # type: ignore[import-untyped]

    BANQUET_AVAILABLE = True
except ImportError:

    class Banquet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Banquet backend not available")

    BANQUET_AVAILABLE = False


class SpectralStemSeparator:
    """
    Einfaches spectral-based stem separation.

    Fast and always available, but basic quality.
    Good for preview/quick exports.
    """

    def __init__(
        self,
        vocal_freq_range: tuple[float, float] = (80, 8000),
        bass_freq_range: tuple[float, float] = (20, 250),
        drums_freq_range: tuple[float, float] = (50, 15000),
    ):
        """
        Parameters
        ----------
        vocal_freq_range : tuple
            (low_hz, high_hz) for vocal extraction
        bass_freq_range : tuple
            (low_hz, high_hz) for bass extraction
        drums_freq_range : tuple
            (low_hz, high_hz) for drums extraction
        """
        self.vocal_freq_range = vocal_freq_range
        self.bass_freq_range = bass_freq_range
        self.drums_freq_range = drums_freq_range
        self.metrics: dict[Any, Any] = {}

    def separate(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        """
        Separate audio into stems using spectral filtering.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        stems : dict
            Dictionary with keys 'vocals', 'drums', 'bass', 'other'
        """
        # Ensure stereo for processing
        is_mono = audio.ndim == 1
        if is_mono:
            audio_stereo = np.stack([audio, audio])
        else:
            # scipy.signal.stft expects shape (channels, samples)
            # Input is usually (samples, channels), so transpose
            if audio.shape[0] > audio.shape[1]:
                audio_stereo = audio.T  # (samples, 2) → (2, samples)
            else:
                audio_stereo = audio  # already (2, samples)

        # STFT
        # Adapt nperseg to signal length
        # audio_stereo now has shape (2, samples) for stereo or (samples,) for mono duplicate
        signal_length = audio_stereo.shape[-1] if audio_stereo.ndim > 1 else len(audio_stereo)
        nperseg = min(2048, max(512, signal_length // 8))
        noverlap = nperseg // 2

        f, _t, Zxx = stft(audio_stereo, sample_rate, nperseg=nperseg, noverlap=noverlap)

        # Frequency masks
        vocal_mask = self._create_freq_mask(f, self.vocal_freq_range)
        bass_mask = self._create_freq_mask(f, self.bass_freq_range)
        drums_mask = self._create_transient_mask(Zxx, f, self.drums_freq_range)

        # Extract stems via masking
        vocals_stft = Zxx * vocal_mask[:, None]
        bass_stft = Zxx * bass_mask[:, None]
        drums_stft = Zxx * drums_mask

        # Other = everything not in vocals/bass/drums
        other_stft = Zxx - vocals_stft - bass_stft - drums_stft

        # ISTFT to reconstruct
        _, vocals = istft(vocals_stft, sample_rate, nperseg=nperseg, noverlap=noverlap)
        _, bass = istft(bass_stft, sample_rate, nperseg=nperseg, noverlap=noverlap)
        _, drums = istft(drums_stft, sample_rate, nperseg=nperseg, noverlap=noverlap)
        _, other = istft(other_stft, sample_rate, nperseg=nperseg, noverlap=noverlap)

        # Convert back to original format
        if is_mono:
            vocals = vocals[0]  # Take first channel
            drums = drums[0]
            bass = bass[0]
            other = other[0]
        else:
            vocals = vocals.T
            drums = drums.T
            bass = bass.T
            other = other.T

        # Ensure correct length
        target_len = audio.shape[-1] if is_mono else audio.shape[0]
        stems = {
            "vocals": self._match_length(vocals, target_len, is_mono),
            "drums": self._match_length(drums, target_len, is_mono),
            "bass": self._match_length(bass, target_len, is_mono),
            "other": self._match_length(other, target_len, is_mono),
        }

        # Store metrics
        self.metrics = {"backend": "spectral", "processing_time_estimate": "fast", "quality": "basic"}

        orig_dtype = audio.dtype
        return {k: np.asarray(v).astype(orig_dtype) for k, v in stems.items()}

    def _create_freq_mask(self, frequencies: np.ndarray, freq_range: tuple[float, float]) -> np.ndarray:
        """Erstellt binary frequency mask."""
        low, high = freq_range
        mask = ((frequencies >= low) & (frequencies <= high)).astype(float)
        return mask

    def _create_transient_mask(
        self, Zxx: np.ndarray, frequencies: np.ndarray, freq_range: tuple[float, float]
    ) -> np.ndarray:
        """
        Erstellt transient-based mask for drums.

        Drums have strong transients (sharp onset).
        """
        # Frequency-limited mask
        freq_mask = self._create_freq_mask(frequencies, freq_range)

        # Detect transients by high-frequency energy changes
        magnitude = np.abs(Zxx)

        # Temporal derivative (high for transients)
        diff = np.diff(magnitude, axis=1, prepend=magnitude[:, :1])
        transient_strength = np.abs(diff)

        # Threshold for transient detection
        threshold = np.percentile(transient_strength, 85)
        transient_mask = (transient_strength > threshold).astype(float)

        # Combine frequency and transient masks
        combined_mask = freq_mask[:, None] * transient_mask

        return combined_mask  # type: ignore[no-any-return]

    def _match_length(self, audio: np.ndarray, target_length: int, is_mono: bool) -> np.ndarray:
        """Match audio length to target"""
        if is_mono:
            if len(audio) > target_length:
                return audio[:target_length]
            elif len(audio) < target_length:
                return np.pad(audio, (0, target_length - len(audio)))
            return audio
        else:
            # Stereo
            if audio.shape[0] > target_length:
                return audio[:target_length]
            elif audio.shape[0] < target_length:
                pad_width = ((0, target_length - audio.shape[0]), (0, 0))
                return np.pad(audio, pad_width)
            return audio

    def get_metrics(self) -> dict:
        """Gibt zurück: separation metrics."""
        return self.metrics


class BanquetStemSeparator:
    """
    ML-based stem separation using Banquet.

    High quality but requires Banquet installation.
    """

    def __init__(self, model_path: str | None = None):
        """
        Parameters
        ----------
        model_path : str, optional
            Path to custom Banquet model
        """
        if not BANQUET_AVAILABLE:
            raise RuntimeError("Banquet not available. Install with: pip install banquet")

        self.model = Banquet(model_path) if model_path else Banquet()
        self.metrics: dict[Any, Any] = {}

    def separate(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        """
        Separate audio using Banquet.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        stems : dict
            Dictionary with keys 'vocals', 'drums', 'bass', 'other'
        """
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        orig_dtype = audio.dtype
        result = self.model.separate(audio, sample_rate)

        # Normalize keys  (Banquet may return different key names)
        stems = {
            "vocals": np.asarray(result.get("vocals", result.get("vocal", audio))),
            "drums": np.asarray(result.get("drums", result.get("drum", np.zeros_like(audio)))),
            "bass": np.asarray(result.get("bass", np.zeros_like(audio))),
            "other": np.asarray(result.get("other", result.get("accompaniment", np.zeros_like(audio)))),
        }

        # Store metrics
        self.metrics = {"backend": "banquet", "quality": "high", "model_type": "ml"}

        return {k: v.astype(orig_dtype) for k, v in stems.items()}

    def get_metrics(self) -> dict:
        """Gibt zurück: separation metrics."""
        return self.metrics


class MLStemSeparator:
    """SOTA ML-basierte Stem-Separation.

    Kaskade nach §4.1 DSP-Entscheidungsmatrix:
      Tier-1: MelBandRoformer (BSRoformer) — Vocals-SOTA, 860 MB ONNX
      Tier-2: MDX23C (Kim_Vocal_2 + Kim_Inst) — robuster Fallback
      Tier-3: HTDemucs-6s — Legacy-Fallback
      Tier-4: HPSS-Wiener (scipy, DSP-only) — immer verfügbar

    Die Fallback-Kette ist vollständig ML-Failure-safe: kein OOM-Kill kann
    die Pipeline stoppen, weil Tier-4 ohne ML-Modelle auskommt.

    Anmerkung: try_allocate() für ML-Modelle wird in den jeweiligen Plugins
    gehandhabt. Diese Klasse delegiert nur ohne Lazy-Load.
    """

    def __init__(self) -> None:
        self.metrics: dict = {}

    def separate(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        """Trenne Audio in Stems (vocals, drums, bass, other).

        Parameters
        ----------
        audio       : float32 ndarray, mono [samples] oder stereo [samples, 2]
        sample_rate : int — muss 48000 sein (Pflicht per §2.0)

        Returns
        -------
        dict mit keys 'vocals', 'drums', 'bass', 'other' (alle float32)
        """
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        orig_dtype = audio.dtype

        # ── Tier-1: MelBandRoformer ───────────────────────────────────────
        try:
            from plugins.bs_roformer_plugin import get_bs_roformer

            plugin = get_bs_roformer()
            # separate() gibt StemSeparationResult zurück; stems = dict[name → array]
            result = plugin.separate(audio, sample_rate, stems=["vocals", "drums", "bass", "other"])
            stems = result.stems
            if stems and "vocals" in stems:
                out = {
                    "vocals": np.asarray(stems.get("vocals", np.zeros_like(audio)), dtype=orig_dtype),
                    "drums": np.asarray(stems.get("drums", np.zeros_like(audio)), dtype=orig_dtype),
                    "bass": np.asarray(stems.get("bass", np.zeros_like(audio)), dtype=orig_dtype),
                    "other": np.asarray(stems.get("other", np.zeros_like(audio)), dtype=orig_dtype),
                }
                self.metrics = {"backend": "BSRoformer", "quality": "SOTA", "tier": 1}
                _logger.info("MLStemSeparator: Tier-1 BSRoformer OK")
                return {k: np.clip(v, -1.0, 1.0).astype(orig_dtype) for k, v in out.items()}
        except Exception as _err:
            _logger.warning("MLStemSeparator: BSRoformer fehlgeschlagen (%s) → Tier-2", _err)

        # ── Tier-2: MDX23C ────────────────────────────────────────────────
        try:
            from plugins.mdx23c_plugin import get_loaded_mdx23c_plugin, get_mdx23c_plugin

            mdx = get_loaded_mdx23c_plugin()
            if mdx is None:
                mdx = get_mdx23c_plugin()
            all_stems = mdx.separate_all_stems(audio, sample_rate, stems=["vocals", "inst"])
            vocals = np.asarray(all_stems.get("vocals", np.zeros_like(audio)), dtype=orig_dtype)
            inst = np.asarray(all_stems.get("inst", np.zeros_like(audio)), dtype=orig_dtype)
            # MDX23C liefert nur vocals + inst; Drums/Bass aus Instrumental via HPSS
            _length = min(len(inst) if inst.ndim == 1 else inst.shape[0], 48000 * 10)
            inst_mono = (inst[:_length] if inst.ndim == 1 else inst[:_length].mean(axis=1)).astype(np.float64)
            from scipy.ndimage import median_filter
            from scipy.signal import istft as _istft
            from scipy.signal import stft as _stft

            _np, _nov = 2048, 1536
            _, _, Zxx = _stft(inst_mono, fs=sample_rate, nperseg=_np, noverlap=_nov)
            _m = np.abs(Zxx)
            _h = median_filter(_m, size=(1, 31))
            _p = median_filter(_m, size=(31, 1))
            _tot = _h + _p + 1e-12
            _, _harm = _istft(Zxx * (_h / _tot), fs=sample_rate, nperseg=_np, noverlap=_nov)
            _, _perc = _istft(Zxx * (_p / _tot), fs=sample_rate, nperseg=_np, noverlap=_nov)
            n = inst.shape[0] if inst.ndim == 2 else len(inst)

            def _fit(a: np.ndarray, n: int) -> np.ndarray:
                return a[:n] if len(a) >= n else np.pad(a, (0, n - len(a)))

            harm_a = _fit(_harm, n).astype(orig_dtype)
            perc_a = _fit(_perc, n).astype(orig_dtype)
            from scipy.signal import butter, sosfilt

            sos_bass = butter(4, 250.0 / (sample_rate / 2.0), btype="low", output="sos")
            bass_a = sosfilt(sos_bass, harm_a).astype(orig_dtype)
            self.metrics = {"backend": "MDX23C+HPSS", "quality": "high", "tier": 2}
            _logger.info("MLStemSeparator: Tier-2 MDX23C OK")
            return {
                "vocals": np.clip(vocals, -1.0, 1.0).astype(orig_dtype),
                "drums": np.clip(perc_a, -1.0, 1.0).astype(orig_dtype),
                "bass": np.clip(bass_a, -1.0, 1.0).astype(orig_dtype),
                "other": np.clip(harm_a - bass_a, -1.0, 1.0).astype(orig_dtype),
            }
        except Exception as _err:
            _logger.warning("MLStemSeparator: MDX23C fehlgeschlagen (%s) → Tier-3", _err)

        # ── Tier-3: HTDemucs ─────────────────────────────────────────────
        try:
            from plugins.htdemucs_plugin import get_htdemucs_plugin

            ht = get_htdemucs_plugin()
            if ht is None:
                raise RuntimeError("HTDemucs plugin nicht verfügbar")
            ht_stems = ht.separate(audio, sample_rate)
            # §Fix 2026-09-08: htdemucs_plugin.separate() liefert ein
            # SeparationResult-Dataclass, KEIN Dict — "vocals" in ht_stems
            # und .get() warfen TypeError → stiller Tier-4-Fallback (§V6).
            if ht_stems is not None and getattr(ht_stems, "vocals", None) is not None:
                self.metrics = {"backend": "HTDemucs", "quality": "high", "tier": 3}
                _logger.info("MLStemSeparator: Tier-3 HTDemucs OK")
                out = {
                    "vocals": np.asarray(getattr(ht_stems, "vocals", np.zeros_like(audio)), dtype=orig_dtype),
                    "drums": np.asarray(getattr(ht_stems, "drums", np.zeros_like(audio)), dtype=orig_dtype),
                    "bass": np.asarray(getattr(ht_stems, "bass", np.zeros_like(audio)), dtype=orig_dtype),
                    "other": np.asarray(getattr(ht_stems, "other", np.zeros_like(audio)), dtype=orig_dtype),
                }
                return {k: np.clip(v, -1.0, 1.0).astype(orig_dtype) for k, v in out.items()}
        except Exception as _err:
            _logger.warning("MLStemSeparator: HTDemucs fehlgeschlagen (%s) → Tier-4 HPSS", _err)

        # ── Tier-4: NMF-β-Separation (sdB ≥ 5 dB guard) ─────────────────
        # §2.47 VERBOTEN: MDX23C-Fallback darf nicht direkt auf HPSS fallen —
        # NMF-β-Separation muss als Zwischenstufe versucht werden.
        try:
            _nmf_result = self._nmf_beta_separate(audio, sample_rate, orig_dtype)
            if _nmf_result is not None:
                _logger.info("MLStemSeparator: Tier-4 NMF-β DSP-Fallback OK (sdB ≥ 5 dB)")
                self.metrics = {"backend": "NMF-beta", "quality": "basic", "tier": 4}
                return _nmf_result
            _logger.info("MLStemSeparator: NMF-β sdB < 5 dB → Tier-5 HPSS")
        except Exception as _err:
            _logger.warning("MLStemSeparator: NMF-β fehlgeschlagen (%s) → Tier-5 HPSS", _err)

        # ── Tier-5: HPSS-Wiener (tertiärer Fallback, kein ML) ────────────
        _logger.info("MLStemSeparator: Tier-5 HPSS-Wiener DSP-Fallback")
        self.metrics = {"backend": "HPSS-Wiener", "quality": "basic", "tier": 5}
        return self._hpss_separate(audio, sample_rate, orig_dtype)

    def _nmf_beta_separate(
        self,
        audio: np.ndarray,
        sr: int,
        dtype: np.dtype,
    ) -> dict[str, np.ndarray] | None:
        """NMF-β-Separation (Kullback-Leibler, n_components=8) — Tier-4.

        Quality guard: if signal-to-distortion ratio (sdB) < 5 dB, returns None
        so the caller falls through to HPSS (Tier-5).

        Implements §2.47: MDX23C-Fallback must traverse NMF-β before HPSS.
        """
        from scipy.signal import istft as _istft
        from scipy.signal import stft as _stft

        try:
            from sklearn.decomposition import NMF as _NMF
        except ImportError:
            return None  # sklearn missing → caller falls to HPSS

        mono = (audio.mean(axis=1) if audio.ndim == 2 else audio).astype(np.float64)
        n = len(mono)
        nperseg, noverlap = 2048, 1536
        _, _, Zxx = _stft(mono, fs=sr, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(Zxx).astype(np.float32)

        n_components = 8
        model = _NMF(n_components=n_components, beta_loss="kullback-leibler", solver="mu", max_iter=60, random_state=0)
        W = model.fit_transform(mag)  # (n_freq, n_comp)
        H = model.components_  # (n_comp, n_time)

        # Assign each component: high temporal variance → percussive; else harmonic
        temporal_var = np.var(H, axis=1)
        is_perc = temporal_var > np.median(temporal_var)
        W_harm = W[:, ~is_perc].sum(axis=1, keepdims=True) if (~is_perc).any() else W.mean(axis=1, keepdims=True)
        W_perc = W[:, is_perc].sum(axis=1, keepdims=True) if is_perc.any() else np.zeros_like(W_harm)

        # Build Wiener-style masks over full spectrogram
        mag_harm = W_harm * H[~is_perc].sum(axis=0, keepdims=True) if (~is_perc).any() else np.zeros_like(mag)
        mag_perc = W_perc * H[is_perc].sum(axis=0, keepdims=True) if is_perc.any() else np.zeros_like(mag)
        total = mag_harm + mag_perc + 1e-10
        mask_h = mag_harm / total
        mask_p = mag_perc / total

        _, harm_a = _istft(Zxx * mask_h, fs=sr, nperseg=nperseg, noverlap=noverlap)
        _, perc_a = _istft(Zxx * mask_p, fs=sr, nperseg=nperseg, noverlap=noverlap)

        def _fit(a: np.ndarray) -> np.ndarray:
            return (a[:n] if len(a) >= n else np.pad(a, (0, n - len(a)))).astype(dtype)

        harm_a = _fit(harm_a)
        perc_a = _fit(perc_a)

        # sdB quality guard: reconstruction sdB ≥ 5 dB
        recon = (harm_a + perc_a).astype(np.float64)
        orig_f = mono.astype(np.float64)
        err = orig_f - recon
        signal_power = float(np.mean(orig_f**2))
        noise_power = float(np.mean(err**2)) + 1e-15
        sdb = 10.0 * np.log10(signal_power / noise_power) if signal_power > 0 else 0.0
        if sdb < 5.0:
            _logger.debug("NMF-β sdB=%.1f dB < 5 dB — quality guard failed", sdb)
            return None

        from scipy.signal import butter, sosfilt

        sos_bass = butter(4, 250.0 / (sr / 2.0), btype="low", output="sos")
        bass_a = sosfilt(sos_bass, harm_a).astype(dtype)
        vocals_a = (harm_a - bass_a).astype(dtype)

        if audio.ndim == 2:

            def _stereo(m: np.ndarray) -> np.ndarray:
                return np.stack([m, m], axis=1)

            return {
                "vocals": np.clip(_stereo(vocals_a), -1.0, 1.0).astype(dtype),
                "drums": np.clip(_stereo(perc_a), -1.0, 1.0).astype(dtype),
                "bass": np.clip(_stereo(bass_a), -1.0, 1.0).astype(dtype),
                "other": np.clip(_stereo(harm_a), -1.0, 1.0).astype(dtype),
            }
        return {
            "vocals": np.clip(vocals_a, -1.0, 1.0).astype(dtype),
            "drums": np.clip(perc_a, -1.0, 1.0).astype(dtype),
            "bass": np.clip(bass_a, -1.0, 1.0).astype(dtype),
            "other": np.clip(harm_a, -1.0, 1.0).astype(dtype),
        }

    def _hpss_separate(
        self,
        audio: np.ndarray,
        sr: int,
        dtype: np.dtype,
    ) -> dict[str, np.ndarray]:
        """HPSS-Wiener-Masken Separation (Fitzgerald 2010) — Tier-4 Fallback."""
        from scipy.ndimage import median_filter
        from scipy.signal import butter, sosfilt
        from scipy.signal import istft as _istft
        from scipy.signal import stft as _stft

        mono = (audio.mean(axis=1) if audio.ndim == 2 else audio).astype(np.float64)
        n = len(mono)
        nperseg, noverlap = 2048, 1536
        _, _, Zxx = _stft(mono, fs=sr, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(Zxx)
        harm_mag = median_filter(mag, size=(1, 31))
        perc_mag = median_filter(mag, size=(31, 1))
        total = harm_mag + perc_mag + 1e-12
        mask_h, mask_p = harm_mag / total, perc_mag / total
        _, harm_a = _istft(Zxx * mask_h, fs=sr, nperseg=nperseg, noverlap=noverlap)
        _, perc_a = _istft(Zxx * mask_p, fs=sr, nperseg=nperseg, noverlap=noverlap)

        def _fit(a: np.ndarray) -> np.ndarray:
            return (a[:n] if len(a) >= n else np.pad(a, (0, n - len(a)))).astype(dtype)

        harm_a = _fit(harm_a)
        perc_a = _fit(perc_a)
        sos_bass = butter(4, 250.0 / (sr / 2.0), btype="low", output="sos")
        bass_a = sosfilt(sos_bass, harm_a).astype(dtype)
        vocals_a = (harm_a - bass_a).astype(dtype)

        if audio.ndim == 2:  # zurück auf stereo duplizieren

            def _stereo(m: np.ndarray) -> np.ndarray:
                return np.stack([m, m], axis=1)

            return {
                "vocals": np.clip(_stereo(vocals_a), -1.0, 1.0).astype(dtype),
                "drums": np.clip(_stereo(perc_a), -1.0, 1.0).astype(dtype),
                "bass": np.clip(_stereo(bass_a), -1.0, 1.0).astype(dtype),
                "other": np.clip(_stereo(harm_a), -1.0, 1.0).astype(dtype),
            }
        return {
            "vocals": np.clip(vocals_a, -1.0, 1.0).astype(dtype),
            "drums": np.clip(perc_a, -1.0, 1.0).astype(dtype),
            "bass": np.clip(bass_a, -1.0, 1.0).astype(dtype),
            "other": np.clip(harm_a, -1.0, 1.0).astype(dtype),
        }

    def get_metrics(self) -> dict:
        return self.metrics


class StemSeparator:
    """
    Einheitliche API für Stem-Separation.

    Automatically selects best available backend:
    - Banquet (if available) for best quality
    - Spectral separation (fallback) for speed
    """

    def __init__(self, backend: str = "auto", **backend_kwargs):
        """
        Parameters
        ----------
        backend : str
            'auto', 'spectral', 'banquet', or 'demucs'
        backend_kwargs : dict
            Parameters passed to backend
        """
        if backend == "auto":
            # Keep auto-mode deterministic and lightweight in default/test setups.
            # High-cost ML backend remains available via explicit backend="demucs".
            backend = "banquet" if BANQUET_AVAILABLE else "spectral"

        if backend == "banquet":
            if not BANQUET_AVAILABLE:
                _logger.warning("Banquet not available, falling back to spectral")
                backend = "spectral"
            else:
                self.backend = BanquetStemSeparator(**backend_kwargs)

        if backend == "spectral":
            self.backend = SpectralStemSeparator(**backend_kwargs)  # type: ignore[assignment]

        if backend == "demucs":
            try:
                self.backend = MLStemSeparator(**backend_kwargs)  # type: ignore[assignment]
            except Exception as _exc:
                _logger.warning("MLStemSeparator nicht verfügbar (%s), Fallback auf spectral", _exc)
                backend = "spectral"
                self.backend = SpectralStemSeparator(**backend_kwargs)  # type: ignore[assignment]

        # Persistente Zuweisung — unabhängig vom Backend-Pfad immer gesetzt
        self.backend_name = backend

    def separate(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        """
        Separate audio into stems.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        stems : dict
            Dictionary with keys:
            - 'vocals': Vocal stem
            - 'drums': Drum stem
            - 'bass': Bass stem
            - 'other': Other/accompaniment stem
        """
        return self.backend.separate(audio, sample_rate)

    def get_backend_info(self) -> dict[str, str]:
        """Gibt zurück: information about current backend."""
        if self.backend_name == "banquet":
            return {
                "backend": "banquet",
                "quality": "high",
                "speed": "medium",
                "description": "ML-based separation (Banquet)",
            }
        elif self.backend_name == "spectral":
            return {
                "backend": "spectral",
                "quality": "basic",
                "speed": "fast",
                "description": "Spectral frequency-based separation",
            }
        elif self.backend_name == "demucs":
            tier = getattr(self.backend, "metrics", {}).get("tier", "?")
            quality = getattr(self.backend, "metrics", {}).get("quality", "SOTA")
            backend_name = getattr(self.backend, "metrics", {}).get("backend", "ML")
            return {
                "backend": backend_name,
                "quality": quality,
                "speed": "medium",
                "description": f"ML Stem-Sep Tier-{tier}: BSRoformer→MDX23C→HTDemucs→HPSS",
            }
        else:
            return {
                "backend": self.backend_name,
                "quality": "unknown",
                "speed": "unknown",
                "description": "Unknown backend",
            }

    def get_metrics(self) -> dict:
        """Gibt zurück: separation metrics from backend."""
        return self.backend.get_metrics()


# Convenience function
def separate_stems(audio: np.ndarray, sample_rate: int, backend: str = "auto") -> dict[str, np.ndarray]:
    """
    Separate audio into stems.

    Parameters
    ----------
    audio : np.ndarray
        Input audio (mono or stereo)
    sample_rate : int
        Sample rate in Hz
    backend : str
        'auto', 'spectral', 'banquet', or 'demucs'

    Returns
    -------
    stems : dict
        Dictionary with 'vocals', 'drums', 'bass', 'other'

    Examples
    --------
    >>> stems = separate_stems(audio, 48000)
    >>> vocals = stems['vocals']
    >>> drums = stems['drums']
    """
    separator = StemSeparator(backend=backend)
    return separator.separate(audio, sample_rate)


# Legacy compatibility
class AiStemSeparator:
    """Legacy wrapper for backward compatibility"""

    def __init__(self, model_path: str | None = None):
        self.separator = StemSeparator(backend="auto")

    def separate(self, audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
        stems = self.separator.separate(audio, sr)
        # Add 'piano' key for backward compatibility
        stems["piano"] = stems["other"]
        return stems


# CLI interface
if __name__ == "__main__":
    import argparse
    import os

    import soundfile as sf

    parser = argparse.ArgumentParser(description="AURIK Stem Separator (GAP #43)")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--output-dir", "-o", default="stems", help="Output directory")
    parser.add_argument(
        "--backend", "-b", default="auto", choices=["auto", "spectral", "banquet"], help="Separation backend"
    )
    parser.add_argument("--format", "-f", default="wav", help="Output format (wav, flac, etc.)")

    args = parser.parse_args()

    # Load audio
    _logger.info("Loading: %s", args.input)
    from backend.file_import import load_audio_file

    _res = load_audio_file(args.input)
    if _res is None:
        raise RuntimeError(f"Audiodatei konnte nicht geladen werden: {args.input}")
    audio, sr = _res["audio"], int(_res["sr"])
    _logger.info("Loaded %d Hz, shape %s", sr, audio.shape)

    # Separate stems
    separator = StemSeparator(backend=args.backend)
    backend_info = separator.get_backend_info()
    _logger.info(
        "Backend: %s (%s quality, %s speed)", backend_info["backend"], backend_info["quality"], backend_info["speed"]
    )

    _logger.info("Separating stems...")
    stems = separator.separate(audio, sr)

    # Export stems
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input))[0]

    for stem_name, stem_audio in stems.items():
        ext = "." + args.format
        output_path = os.path.join(args.output_dir, f"{base_name}_{stem_name}{ext}")
        sf.write(output_path, stem_audio, sr)
        _logger.info("%-8s -> %s", stem_name, output_path)

    # Print metrics
    metrics = separator.get_metrics()
    _logger.info(
        "Separation complete! Backend: %s  Quality: %s",
        metrics.get("backend", "unknown"),
        metrics.get("quality", "unknown"),
    )
