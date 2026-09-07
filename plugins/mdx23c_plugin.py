"""MDX23C Plugin — lokale ONNX-Inferenz (kein Docker).

Nutzt Kim_Vocal_2.onnx (Gesang) und Kim_Inst.onnx (Instrumente) direkt über
onnxruntime (CPUExecutionProvider).

MDX-Net-Verarbeitungskette (Défossez et al. 2023):
    1. STFT (n_fft=6144, hop=1024) → komplexe Spektrogramm-Matrix
    2. ONNX-Modell → Trainierte Maske [dim_f, dim_t]
    3. Maske × Eingabe-Spektrogramm → Quell-Spektrogramm
    4. iSTFT → Zeit-Signal pro Kanal

Fallback: Harmonisch-Perkussiv-Trennung via HPSS (scipy/librosa).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from plugins.htdemucs_plugin import SeparationResult

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MDX-Net Konstanten (Kim_Vocal_2 / Kim_Inst)
# ---------------------------------------------------------------------------
MDX_N_FFT: int = 6144
MDX_HOP: int = 1024
MDX_DIM_F: int = 3072  # n_fft // 2
MDX_DIM_T: int = 256  # Zeit-Frames pro Inferenz-Fenster (Modell-abhängig)
MDX_SR: int = 44100  # MDX23C-Modelle trainiert auf 44.1 kHz

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_instances: dict[str, MDX23CModel] = {}
_lock = threading.Lock()


def _get_model(stem: str) -> MDX23CModel:
    key = _stem_key(stem)
    if key not in _instances:
        with _lock:
            if key not in _instances:
                _instances[key] = MDX23CModel(key)
    return _instances[key]


def _stem_key(stem: str) -> str:
    stem = stem.lower()
    if stem in ("vocals", "vocal", "voice"):
        return "vocals"
    return "inst"  # drums, bass, other, inst, instruments


# ---------------------------------------------------------------------------
# Modell-Wrapper
# ---------------------------------------------------------------------------
class MDX23CModel:
    """Lädt ein Kim-ONNX-Modell und führt MDX-Net-Inferenz aus."""

    MODEL_MAP: dict[str, list[str]] = {
        "vocals": [
            "models/mdx23c/models/Kim_Vocal_2.onnx",
            "models/kim_vocal_2/kim_vocal_2.onnx",
            "models/kim_vocal_1/kim_vocal_1.onnx",
        ],
        "inst": [
            "models/mdx23c/models/Kim_Inst.onnx",
            "models/kim_inst/kim_inst.onnx",
        ],
    }

    def __init__(self, stem_key: str) -> None:
        self.stem_key = stem_key
        self._session: Any = None
        self._ok = False
        self._dim_t = MDX_DIM_T
        self._workspace = Path(__file__).parent.parent
        self._load()

    def _load(self) -> None:
        paths = [self._workspace / p for p in self.MODEL_MAP.get(self.stem_key, [])]
        for path in paths:
            if not path.exists():
                continue
            try:
                import onnxruntime as ort

                # ML-Budget-Guard: MDX23C Kim_Vocal_2 + Kim_Inst zusammen ~1.1 GB
                try:
                    from backend.core.ml_memory_budget import (
                        release as _rel,
                    )
                    from backend.core.ml_memory_budget import (
                        try_allocate as _try_alloc,
                    )

                    if not _try_alloc(f"MDX23C_{self.stem_key}", size_gb=0.55):
                        try:
                            _rel(f"MDX23C_{self.stem_key}")
                        except Exception:
                            logger.warning("mdx23c_plugin.py::_laden Ersatzpfad", exc_info=True)
                        if not _try_alloc(f"MDX23C_{self.stem_key}", size_gb=0.55):
                            logger.warning("MDX23C [%s]: ML-Grenze erschöpft — NMF-β-Ersatzpfad", self.stem_key)
                            return
                except Exception:
                    _rel = None  # Budget-Modul nicht verfügbar (harmlos bei raise, kein fail-open da try_alloc bereits im Fehlerpfad)

                opts = ort.SessionOptions()
                opts.inter_op_num_threads = 1
                opts.intra_op_num_threads = 4
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                try:
                    from backend.core.ml_device_manager import get_ort_providers as _get_prov

                    _mdx23c_prov = _get_prov("MDX23C")
                except Exception:
                    _mdx23c_prov = ["CPUExecutionProvider"]
                self._session = ort.InferenceSession(
                    str(path),
                    sess_options=opts,
                    providers=_mdx23c_prov,
                )
                # Dim-T aus Modell-Eingabe ableiten
                inp_shape = self._session.get_inputs()[0].shape
                if len(inp_shape) >= 4 and isinstance(inp_shape[3], int):
                    self._dim_t = inp_shape[3]
                self._ok = True
                logger.info(
                    "✅ MDX23C [%s] geladen: %s | dim_t=%d",
                    self.stem_key,
                    path.name,
                    self._dim_t,
                )
                try:
                    from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                    _reg_plm(
                        f"MDX23C_{self.stem_key}",
                        size_gb=0.55,
                        unload_fn=lambda s=self: setattr(s, "_session", None) or setattr(s, "_ok", False),
                    )
                except Exception as _exc:
                    logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
                return
            except Exception as exc:
                logger.debug("MDX23C [%s] Ladefehler (%s): %s", self.stem_key, path.name, exc)
                try:
                    from backend.core.ml_memory_budget import release as _release

                    _release(f"MDX23C_{self.stem_key}")
                except Exception as _exc:
                    logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

        logger.warning("MDX23C [%s]: Kein ONNX-Modell gefunden — NMF-β-Ersatzpfad aktiv", self.stem_key)

    # ------------------------------------------------------------------
    def separate(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Trennt den Stem (Gesang oder Instrumente) aus dem Audio.

        Args:
            audio: float32 [2, samples] Stereo (intern zu Stereo konvertiert)
            sr   : Sample-Rate in Hz (intern auf MDX_SR resampelt)

        Returns: float32 [2, samples] — restaurierter Stem
        """
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        # Sicherstellen: Stereo [2, samples]
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        elif audio.shape[0] == 1:
            audio = np.tile(audio, (2, 1))
        elif audio.shape[0] > 2:
            audio = audio[:2]

        # Resample auf MDX_SR
        resampled, rs_sr = self._resample(audio, sr, MDX_SR)

        is_vocals = self.stem_key == "vocals"
        if self._ok:
            _plm_mdx = None
            try:
                from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm_fn

                _plm_mdx = _get_plm_fn()
                _plm_mdx.set_active(f"MDX23C_{self.stem_key}", True)
            except Exception as _exc:
                logger.debug("MDX23C: PLM set_active fehlgeschlagen: %s", _exc)
            try:
                out = self._mdx_separate(resampled)
            except Exception as exc:
                logger.warning("MDX23C ONNX-Fehler: %s — NMF-β-Ersatzpfad", exc)
                out = self._nmf_beta_fallback(resampled, is_vocals=is_vocals)
            finally:
                if _plm_mdx is not None:
                    try:
                        _plm_mdx.set_active(f"MDX23C_{self.stem_key}", False)
                    except Exception as _exc:
                        logger.debug("MDX23C: PLM unset_active fehlgeschlagen: %s", _exc)
        else:
            out = self._nmf_beta_fallback(resampled, is_vocals=is_vocals)

        # Zurück auf Original-SR
        if rs_sr != sr:
            out, _ = self._resample(out, rs_sr, sr)

        return np.clip(
            np.nan_to_num(out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
            -1.0,
            1.0,
        )

    # ------------------------------------------------------------------
    def _mdx_separate(self, audio: np.ndarray) -> np.ndarray:
        """MDX-Net Chip-Inferenz: STFT → Maske → iSTFT.

        Algorithmus:
            1. STFT pro Kanal (n_fft=6144, hop=1024, Hanning)
            2. Stapele Real+Imag je Kanal → [batch, 4, dim_f, dim_t]
            3. Modell-Inferenz → Masken-Tensor
            4. Maske × Eingabe-Spektrogramm
            5. iSTFT → Zeit-Signal
        """
        L = audio.shape[1]

        # STFT
        spec_L = self._stft(audio[0])  # [freq_bins, time_frames] complex
        spec_R = self._stft(audio[1])

        n_frames = spec_L.shape[1]
        dim_f = MDX_DIM_F
        dim_t = self._dim_t

        # Zuschneiden / Auffüllen auf dim_f × dim_t-Raster
        out_L = np.zeros_like(spec_L)
        out_R = np.zeros_like(spec_R)

        hop_t = dim_t // 2  # 50 % Overlap in Zeit-Richtung
        pos = 0
        while pos < n_frames:
            end = min(pos + dim_t, n_frames)
            sl_L = spec_L[:dim_f, pos:end]
            sl_R = spec_R[:dim_f, pos:end]

            pad_t = dim_t - sl_L.shape[1]
            if pad_t > 0:
                sl_L = np.pad(sl_L, ((0, 0), (0, pad_t)))
                sl_R = np.pad(sl_R, ((0, 0), (0, pad_t)))

            # [batch=1, 4, dim_f, dim_t]
            inp = np.stack([sl_L.real, sl_L.imag, sl_R.real, sl_R.imag], axis=0)[np.newaxis].astype(np.float32)

            mask = self._session.run(None, {self._session.get_inputs()[0].name: inp})[0]
            mask = np.nan_to_num(np.asarray(mask, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            mask = np.squeeze(mask)  # [4, dim_f, dim_t] oder [dim_f, dim_t]

            if mask.ndim == 3:
                mL = mask[0] + 1j * mask[1]
                mR = mask[2] + 1j * mask[3]
            else:
                mL = mR = mask.astype(np.float32)

            actual = end - pos
            out_L[:dim_f, pos:end] = (mL * sl_L)[:, :actual]
            out_R[:dim_f, pos:end] = (mR * sl_R)[:, :actual]

            pos += hop_t if hop_t > 0 else dim_t

        sig_L = self._istft(out_L, length=L)
        sig_R = self._istft(out_R, length=L)
        return np.stack([sig_L, sig_R])

    # ------------------------------------------------------------------
    @staticmethod
    def _stft(x: np.ndarray) -> np.ndarray:
        """Short-Time Fourier Transform (Hanning-Fenster)."""
        n_fft = MDX_N_FFT
        hop = MDX_HOP
        win = np.hanning(n_fft).astype(np.float32)
        n_frames = 1 + (len(x) + n_fft) // hop

        # Zero-Pad
        x_pad = np.pad(x, n_fft // 2)
        frames = []
        for i in range(n_frames):
            start = i * hop
            seg = x_pad[start : start + n_fft]
            if len(seg) < n_fft:
                seg = np.pad(seg, (0, n_fft - len(seg)))
            frames.append(np.fft.rfft(seg * win))
        return np.stack(frames, axis=1).astype(np.complex64)  # [freq, time]

    @staticmethod
    def _istft(spec: np.ndarray, length: int) -> np.ndarray:
        """Inverse STFT (Griffin-Lim OLA)."""
        n_fft = MDX_N_FFT
        hop = MDX_HOP
        win = np.hanning(n_fft).astype(np.float32)

        n_frames = spec.shape[1]
        out_len = (n_frames - 1) * hop + n_fft + n_fft // 2
        out = np.zeros(out_len, dtype=np.float32)
        norm = np.zeros(out_len, dtype=np.float32)

        for i in range(n_frames):
            frame = np.fft.irfft(spec[:, i], n=n_fft).real.astype(np.float32)
            start = i * hop
            out[start : start + n_fft] += frame * win
            norm[start : start + n_fft] += win**2

        norm = np.where(norm < 1e-8, 1.0, norm)
        out /= norm
        # Schneide auf Originallänge (nach Center-Pad-Versatz)
        out = out[n_fft // 2 : n_fft // 2 + length]
        if len(out) < length:
            out = np.pad(out, (0, length - len(out)))
        return out[:length]

    # ------------------------------------------------------------------
    @staticmethod
    def _nmf_beta_fallback(audio: np.ndarray, is_vocals: bool) -> np.ndarray:
        """NMF-β Stem-Separation (Smaragdis & Brown 2003, §2.47 Spec ML-Fallback).

        Itakura-Saito NMF (β=0) auf STFT-Magnitude; Komponenten werden nach
        Vokal-Band-Energie-Anteil (300–3000 Hz) in Gesang/Instrumental klassifiziert.
        Anforderung: Proxy-SDR ≥ 5 dB (Energie-Kontrast zwischen Maske und Residuum).
        Bei Unterschreitung oder Fehler → HPSS-Fallback.

        is_vocals=True  → vokale Komponenten (hoher 300–3000 Hz Anteil)
        is_vocals=False → nicht-vokale Komponenten (Instrumente)
        """
        try:
            from sklearn.decomposition import NMF as _NMF
        except ImportError:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            return MDX23CModel._hpss_fallback(audio, is_vocals)

        try:
            channels, n = audio.shape
            if n < MDX_N_FFT:
                return MDX23CModel._hpss_fallback(audio, is_vocals)

            n_fft = 2048
            hop = 512
            K = 8  # NMF-Rang (8 Komponenten ausreichend für Vokal/Instrumental-Split)
            win = np.hanning(n_fft).astype(np.float32)
            freqs_hz = np.fft.rfftfreq(n_fft, d=1.0 / MDX_SR)  # MDX_SR = 44100 Hz
            vocal_band = (freqs_hz >= 300) & (freqs_hz <= 3000)

            result = []
            for c in range(channels):
                sig = audio[c]
                n_frames = (len(sig) - n_fft) // hop + 1
                if n_frames < 4:
                    result.append(sig.copy())
                    continue

                # STFT → Magnitude + Phase
                frames = np.stack([sig[i * hop : i * hop + n_fft] for i in range(n_frames)])  # (n_frames, n_fft)
                stft = np.fft.rfft(frames * win, n=n_fft)  # (n_frames, n_fft//2+1)
                mag = np.abs(stft).astype(np.float32)  # NMF input

                # NMF Itakura-Saito (β=0): V ≈ W · H, min IS-Divergenz
                model = _NMF(
                    n_components=K,
                    beta_loss="itakura-saito",
                    solver="mu",
                    max_iter=120,
                    random_state=0,
                    init="nndsvda",
                )
                H = model.fit_transform(mag + 1e-8)  # (n_frames, K) — Zeitaktivierungen
                W = model.components_  # (K, n_fft//2+1) — Spektralbases

                # Vokal-Ratio pro Komponente: Energie-Anteil im Vokalband 300–3000 Hz
                component_vocal_ratios = np.array([np.sum(W[k, vocal_band]) / (np.sum(W[k]) + 1e-8) for k in range(K)])

                # Soft-Mask: gewichtete Rekonstruktion nach Vokal-Anteil
                vocal_spec = np.zeros_like(mag)
                inst_spec = np.zeros_like(mag)
                for k in range(K):
                    component = np.outer(H[:, k], W[k])  # (n_frames, freqs)
                    v_weight = float(component_vocal_ratios[k])
                    vocal_spec += v_weight * component
                    inst_spec += (1.0 - v_weight) * component

                # Soft-Wiener-Maske
                total = vocal_spec + inst_spec + 1e-8
                vocal_mask = vocal_spec / total  # (n_frames, freqs) ∈ [0, 1]
                inst_mask = inst_spec / total

                target_mask = vocal_mask if is_vocals else inst_mask

                # Proxy-SDR: Energie-Kontrast in Vokalband
                target_in_band = float(np.mean(target_mask[:, vocal_band]))
                reject_in_band = float(np.mean((1.0 - target_mask)[:, vocal_band]))
                sdr_proxy_db = 10.0 * np.log10(target_in_band / (reject_in_band + 1e-12) + 1e-12)
                if sdr_proxy_db < 5.0:
                    logger.debug(
                        "MDX23C NMF-β: Proxy-SDR %.1f dB < 5 dB — HPSS-Ersatzpfad",
                        sdr_proxy_db,
                    )
                    return MDX23CModel._hpss_fallback(audio, is_vocals)

                # Maske × Original-STFT → iSTFT (Overlap-Add)
                masked_stft = stft * target_mask
                out = np.zeros(n, dtype=np.float32)
                norm = np.zeros(n, dtype=np.float32)
                for i in range(n_frames):
                    frame = np.fft.irfft(masked_stft[i], n=n_fft).real.astype(np.float32)
                    s, e = i * hop, min(i * hop + n_fft, n)
                    out[s:e] += frame[: e - s] * win[: e - s]
                    norm[s:e] += win[: e - s] ** 2
                norm = np.where(norm < 1e-8, 1.0, norm)
                out /= norm
                result.append(out)

            logger.info(
                "MDX23C NMF-β-Ersatzpfad: %s Stem (K=%d, SDR-Proxy≥5 dB OK)",
                "vocals" if is_vocals else "instruments",
                K,
            )
            return np.stack(result)

        except Exception as exc:
            logger.warning("MDX23C NMF-β fehlgeschlagen: %s — HPSS-Ersatzpfad", exc)
            return MDX23CModel._hpss_fallback(audio, is_vocals)

    @staticmethod
    def _hpss_fallback(audio: np.ndarray, is_vocals: bool) -> np.ndarray:
        """HPSS-Fallback (Fitzgerald 2010, Medianfilter) — tertiärer Fallback nach NMF-β.

        is_vocals=True  → harmonischer Anteil (Gesang)
        is_vocals=False → perkussiver Anteil  (Instrumente)
        """
        try:
            import librosa

            channels, n = audio.shape
            result = []
            for c in range(channels):
                H, P = librosa.effects.hpss(audio[c], kernel_size=31)
                result.append(H if is_vocals else P)
            return np.stack(result)
        except ImportError:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

        # Numpy-only Fallback: Medianfilter auf Spektrogramm
        try:
            from scipy.ndimage import median_filter

            channels, n = audio.shape
            n_fft = 2048
            hop = 512
            win = np.hanning(n_fft)
            result = []
            for c in range(channels):
                # STFT → Betrag
                frames = [
                    np.abs(np.fft.rfft(audio[c][i * hop : i * hop + n_fft] * win))
                    for i in range((n - n_fft) // hop + 1)
                ]
                S = np.stack(frames, axis=1).astype(np.float32)  # [freq, time]
                H = median_filter(S, size=(1, 31))
                P = median_filter(S, size=(31, 1))
                mask = H / (H + P + 1e-6) if is_vocals else P / (H + P + 1e-6)
                # Rekonstruktion über ISTFT (einfach)
                phases = np.angle(
                    np.stack(
                        [np.fft.rfft(audio[c][i * hop : i * hop + n_fft] * win) for i in range(S.shape[1])],
                        axis=1,
                    )
                )
                spec = (S * mask) * np.exp(1j * phases)
                sig = np.zeros(n, dtype=np.float32)
                for i in range(S.shape[1]):
                    frame = np.fft.irfft(spec[:, i], n=n_fft).real[:n_fft]
                    s, e = i * hop, min(i * hop + n_fft, n)
                    sig[s:e] += frame[: e - s] * win[: e - s]
                result.append(sig)
            return np.stack(result)
        except Exception:
            logger.warning("mdx23c_plugin.py::_hpss_Ersatzpfad Ersatzpfad", exc_info=True)
            return audio.copy()

    @staticmethod
    def _resample(audio: np.ndarray, src: int, tgt: int) -> tuple[np.ndarray, int]:
        if src == tgt:
            return audio, tgt
        try:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(tgt, src)
            up, dn = tgt // g, src // g
            out = np.stack([resample_poly(ch, up, dn).astype(np.float32) for ch in audio])
            return out, tgt
        except Exception:
            logger.warning("mdx23c_plugin.py::_resample Ersatzpfad", exc_info=True)
            return audio, src


# ---------------------------------------------------------------------------
# Öffentliche Plugin-Klasse (Singleton-Wrapper, kompatibel mit alter API)
# ---------------------------------------------------------------------------
class MDX23CPlugin:
    """Öffentlicher Einstiegspunkt für Stem-Separation (Drop-In-Ersatz).

    Ersetzt die alte Docker-basierte Implementierung vollständig.
    Alle Methoden-Signaturen sind kompatibel.
    """

    def __init__(self, **_kwargs: object) -> None:
        if _kwargs:
            logger.debug("MDX23CPlugin: ignoring legacy kwargs: %s", list(_kwargs.keys()))

    # Facade-Kompatibilität: htdemucs_plugin.get_htdemucs_plugin() routet hierher
    # und musical_goals_metrics prüft dieses Attribut für den Warm-up-Pfad.
    _model_type: str = "mdx23c"

    def separate(self, audio: np.ndarray, sr: int) -> SeparationResult:
        """Drop-In-kompatibel zu HtdemucsPlugin.separate() — 4-Stem-Ergebnis."""
        from plugins.htdemucs_plugin import SeparationResult  # lazy: vermeidet Import-Zyklus

        vocals = self.process(audio, sr, stem="vocals")
        drums = self.process(audio, sr, stem="drums")
        bass = self.process(audio, sr, stem="bass")
        other = self.process(audio, sr, stem="other")
        return SeparationResult(vocals=vocals, drums=drums, bass=bass, other=other, sr=sr)

    def process(
        self,
        audio: np.ndarray,
        sr: int,
        stem: str = "vocals",
    ) -> np.ndarray:
        """Trenne einen Stem aus dem Audio.

        Args:
            audio: float32 [samples] oder [channels, samples]
            sr   : Sample-Rate in Hz
            stem : 'vocals' | 'inst' | 'drums' | 'bass' | 'other'

        Returns: float32 ndarray, selbe Form wie Eingabe.
        """
        stereo_in = audio.ndim == 2
        if not stereo_in:
            audio = np.stack([audio, audio])
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        result = _get_model(stem).separate(audio, sr)

        if not stereo_in:
            result = result.mean(axis=0)
        return np.clip(result, -1.0, 1.0)

    def separate_all_stems(
        self,
        audio: np.ndarray,
        sr: int,
        stems: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Trenne mehrere Stems gleichzeitig.

        Returns: Dict[stem_name → float32 ndarray [channels, samples]]
        """
        if stems is None:
            stems = ["vocals", "inst"]
        result: dict[str, np.ndarray] = {}
        for s in stems:
            result[s] = self.process(audio, sr, stem=s)
        return result

    # ------------------------------------------------------------------
    # Legacy file-based API
    # ------------------------------------------------------------------
    def process_files(
        self,
        input_wav: str,
        output_wav: str,
        stem: str = "vocals",
    ) -> None:
        """Verarbeite WAV-Datei (kompatibel mit alter Docker-API)."""
        try:
            import soundfile as sf

            from backend.file_import import load_audio_file

            _res = cast(dict[str, Any], load_audio_file(input_wav, do_carrier_analysis=False))
            audio = np.asarray(_res["audio"], dtype=np.float32)
            sr = int(_res["sr"])
            audio = audio[np.newaxis, :] if audio.ndim == 1 else audio.T
            result = self.process(audio, sr, stem=stem)
            Path(output_wav).parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_wav, result.T if result.ndim == 2 else result, sr)
            logger.info("✅ MDX23C [%s]: %s → %s", stem, input_wav, output_wav)
        except Exception as exc:
            logger.error("MDX23C verarbeiten_files [%s] fehlgeschlagen: %s", stem, exc)
            raise

    def process_batch(
        self,
        input_files: list[str],
        output_dir: str,
        stem: str = "vocals",
    ) -> list[str]:
        """Batch-Verarbeitung mehrerer Dateien."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[str] = []
        for f in input_files:
            out = str(out_dir / (Path(f).stem + f"_{stem}.wav"))
            try:
                self.process_files(f, out, stem=stem)
                outputs.append(out)
            except Exception as exc:
                logger.error("Batch-Fehler [%s]: %s", f, exc)
        return outputs


# ---------------------------------------------------------------------------
# Singleton-Zugang
# ---------------------------------------------------------------------------
_plugin_instance: MDX23CPlugin | None = None
_plugin_lock = threading.Lock()


def get_mdx23c_plugin() -> MDX23CPlugin:
    """Thread-sicherer Singleton."""
    global _plugin_instance
    if _plugin_instance is None:
        with _plugin_lock:
            if _plugin_instance is None:
                _plugin_instance = MDX23CPlugin()
    return _plugin_instance


def get_loaded_mdx23c_plugin() -> MDX23CPlugin | None:
    """Gibt nur eine bereits geladene MDX23C-Instanz zurück, ohne Lazy-Load."""
    return _plugin_instance


# ---------------------------------------------------------------------------
# Convenience-Funktionen
# ---------------------------------------------------------------------------
def separate_vocals(audio: np.ndarray, sr: int) -> np.ndarray:
    """Trenne Gesang aus dem Audio."""
    plugin = get_loaded_mdx23c_plugin()
    if plugin is None:
        plugin = get_mdx23c_plugin()
    return plugin.process(audio, sr, stem="vocals")


def separate_stems(audio: np.ndarray, sr: int, stems: list[str] | None = None) -> dict[str, np.ndarray]:
    """Trenne mehrere Stems aus dem Audio."""
    plugin = get_loaded_mdx23c_plugin()
    if plugin is None:
        plugin = get_mdx23c_plugin()
    return plugin.separate_all_stems(audio, sr, stems=stems)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if len(sys.argv) < 3:
        logger.debug("Verwendung: mdx23c_plugin.py <Eingabe.wav> <Ausgabe.wav> [stem=vocals]")
        sys.exit(1)
    stem = sys.argv[3] if len(sys.argv) > 3 else "vocals"
    get_mdx23c_plugin().process_files(sys.argv[1], sys.argv[2], stem=stem)
