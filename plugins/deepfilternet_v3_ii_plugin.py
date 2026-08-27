"""
DeepFilterNetV3Plugin — Dreiteilige ONNX-Rauschunterdrückung (enc + dec + erb_dec).
Nutzt lokale Modelle models/deepfilternet_v3_ii/enc.onnx, dec.onnx, erb_dec.onnx.
Kein Docker, kein Netzwerk.

Referenz: Schröter et al. (2023) DeepFilterNet3: Joint postfilter and neural vocoder for speech enhancement.
         Hinweis: "v3.II" ist Aurik-interne Iterations-Bezeichnung (keine offizielle DeepFilterNet-Versionsnummer).
ONNX-Interface:
  enc:     feat_erb[1,1,S,32] + feat_spec[1,2,S,96]
           → e0,e1,e2,e3,emb[1,S,512], c0[1,64,S,96], lsnr
  erb_dec: emb[1,S,512] + e3,e2,e1,e0 → m[1,1,S,32]
  dec:     emb[1,S,512] + c0[1,64,S,96] → coefs[B,S,96,10], alpha
"""

from __future__ import annotations

# v10.101 SOTA: DeepFilterNet mit perzeptueller Maskierung kombiniert.
import logging
import math
import os
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Modulebene (§G174): Flags ohne Backend-Abhängigkeiten.
try:
    from backend.core.music_model_flags import MUSIC_MODEL_PATHS, use_df_musik

    _FLAGS_AVAILABLE = True
except Exception as _flags_exc:
    _FLAGS_AVAILABLE = False
    MUSIC_MODEL_PATHS = {}  # type: ignore[assignment]
    use_df_musik = False
    logger.debug("DeepFilterNet: music_model_flags nicht ladbar: %s", _flags_exc)

_lock = threading.Lock()
_inst: DeepFilterNetV3Plugin | None = None

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DIR = os.path.join(_ROOT, "models", "deepfilternet_v3_ii")


def _resolve_model_dir(default_dir: str) -> str:
    """§v10.19 Feature-Flag-Routing: Musik-Finetune vor Legacy.

    use_df_musik=True und vollständiges Finetune-Verzeichnis
    (enc/dec/erb_dec) → selbst trainiertes Musik-Modell aktiv.
    Sonst Legacy (speech-trainiert) — identisches Interface, kein Risiko.
    """
    if not _FLAGS_AVAILABLE:
        return default_dir
    _fin = MUSIC_MODEL_PATHS["dfn_enc"].parent
    if use_df_musik and all((_fin / f).exists() for f in ("enc.onnx", "dec.onnx", "erb_dec.onnx")):
        return str(_fin)
    return default_dir


# DeepFilterNet processing constants (48 kHz)
_SR = 48_000
_N_FFT = 960  # 20 ms at 48 kHz
_HOP = 480  # 10 ms
_N_ERB = 32  # ERB bands
_DF_BINS = 96  # DF filter bandwidth (bins)
_DF_ORDER = 10  # DF filter order (coefs per bin)


def _erb_fb(n_fft: int = 960, n_erb: int = 32, sr: float = 48000.0) -> np.ndarray:
    """Erstelle ERB-Filterbank-Matrix [n_erb, n_fft//2+1].

    Bildet lineare FFT-Bins auf n_erb ERB-Bänder ab (Zwicker ERB-Skala).
    """
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_bins)

    def hz_to_erb(f: np.ndarray) -> np.ndarray:
        return 21.4 * np.log10(1.0 + f / 229.0 + 1e-9)  # type: ignore[no-any-return]

    erb_max = hz_to_erb(np.array([sr / 2]))[0]
    erb_edges = np.linspace(hz_to_erb(np.array([0.0]))[0], erb_max, n_erb + 1)

    fb = np.zeros((n_erb, n_bins), dtype=np.float32)
    for b in range(n_erb):
        lo = erb_edges[b]
        hi = erb_edges[b + 1]
        erb_freqs = hz_to_erb(freqs)
        mask = (erb_freqs >= lo) & (erb_freqs < hi)
        if mask.sum() > 0:
            fb[b, mask] = 1.0 / mask.sum()
    return fb  # type: ignore[no-any-return]


_ERB_FB = _erb_fb(_N_FFT, _N_ERB, float(_SR))  # [32, 481]


class DeepFilterNetV3Plugin:
    """DeepFilterNet v3 (§v10.15 Musik-Fine-Tuning) II Rauschunterdrückung (ONNX) mit OMLSA-DSP-Fallback.

    Die drei Modelle arbeiten zusammen:
      1. enc: Berechnet Embedding + Encoder-Features aus ERB und Spektrum
      2. erb_dec: Dekodiert ERB-Maske aus Embedding + Encoder-Features
      3. dec: Dekodiert Deep-Filter-Koeffizienten (DF-coefs) aus Embedding
    """

    def __init__(self, model_dir: str | None = None, root: str | None = None) -> None:
        d = model_dir or _DIR
        if root:
            d = os.path.join(root, "models", "deepfilternet_v3_ii")
        elif model_dir is None:
            d = _resolve_model_dir(d)
        self._enc: Any = None
        self._dec: Any = None
        self._erb_dec: Any = None
        self._current_energy_bias_db: float = 0.0  # §0j (dsp.instructions.md); Default §v10.15/§v10.19
        self._try_load(d)

    def _try_load(self, d: str) -> None:
        for attr, fname in [("_enc", "enc.onnx"), ("_dec", "dec.onnx"), ("_erb_dec", "erb_dec.onnx")]:
            p = os.path.join(d, fname)
            if not os.path.exists(p):
                logger.warning("DeepFilterNet-Modell fehlt: %s — DSP-Ersatzpfad aktiv.", p)
                return
        # ── ML-Budget-Check VOR dem Laden (§5.1 OOM-Schutz) ──────────────────
        _allocated = False
        try:
            from backend.core.ml_memory_budget import release as _release
            from backend.core.ml_memory_budget import try_allocate

            if not try_allocate("DeepFilterNetV3", size_gb=0.15):
                # Second-chance allocation: clear potential stale slot and retry once.
                try:
                    _release("DeepFilterNetV3")
                except Exception:
                    logger.warning("deepfilternet_v3_ii_plugin.py::_try_laden Ersatzpfad", exc_info=True)
                if not try_allocate("DeepFilterNetV3", size_gb=0.15):
                    logger.warning("DeepFilterNet: ML-Grenze erschöpft — DSP-Ersatzpfad aktiv")
                    return
            _allocated = True
        except ImportError:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            opts.intra_op_num_threads = 2
            # §AMD: Try ROCm fp16 providers first; CPU fallback (non-GPU-ops stay on CPU)
            try:
                from backend.core.ml_device_manager import get_ort_providers

                prov = get_ort_providers("DeepFilterNetV3")
            except Exception:
                prov = ["CPUExecutionProvider"]
            self._enc = ort.InferenceSession(os.path.join(d, "enc.onnx"), sess_options=opts, providers=prov)
            self._dec = ort.InferenceSession(os.path.join(d, "dec.onnx"), sess_options=opts, providers=prov)
            self._erb_dec = ort.InferenceSession(os.path.join(d, "erb_dec.onnx"), sess_options=opts, providers=prov)
            logger.info("deepfilternet_v3_ii_plugin: ONNX models geladen from: %s", d)
            try:
                from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                _reg_plm(
                    "DeepFilterNetV3",
                    size_gb=0.15,
                    unload_fn=lambda s=self: (  # type: ignore[misc]
                        setattr(s, "_enc", None) or setattr(s, "_dec", None) or setattr(s, "_erb_dec", None)  # type: ignore[func-returns-value]
                    ),
                )
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
        except Exception as exc:
            logger.warning("DeepFilterNet ONNX-Ladefehler: %s — DSP-Ersatzpfad aktiv.", exc)
            self._enc = self._dec = self._erb_dec = None
            if _allocated:
                try:
                    from backend.core.ml_memory_budget import release as _release

                    _release("DeepFilterNetV3")
                except ImportError:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

    # ── Public API ──────────────────────────────────────────────────────────

    def enhance(
        self,
        audio: np.ndarray,
        sr: int,
        energy_bias_db: float = 0.0,  # §v10.15: DFN Musik — kein energy_bias nötig
    ) -> np.ndarray:
        """Rauschunterdrückung via DeepFilterNet oder OMLSA-Fallback.

        Args:
            audio:           float32 mono [n] oder stereo [n,2].
            sr:              Sample-Rate in Hz.
            energy_bias_db:  §0j (dsp.instructions.md): verschiebt die
                             Noise-Floor-Entscheidungsgrenze der NR.
                             Negativer Wert hebt den Gain-Floor → harmonische
                             Energie wird nicht als Rauschen abgetragen.
                             Vokal-Pfade übergeben −6 dB (register-adaptiv
                             −3…−9 dB), Instrumental −9 dB.
                             Standard: 0.0 dB (§v10.15 Musik-Finetune;
                             §v10.19: statischer Bias-Workaround entfernt).

        Returns:
            Denoisiertes Audio, selbe Form, float32 ∈ [-1, 1].
        """
        self._current_energy_bias_db: float = energy_bias_db
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        # UV3 passes (2, N) channels-first; normalize to (N, 2) for uniform processing.
        _was_channels_first = audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2
        if _was_channels_first:
            audio = audio.T  # (2, N) → (N, 2)
        stereo = audio.ndim == 2 and audio.shape[1] == 2

        def proc(ch: np.ndarray) -> np.ndarray:
            res = self._enhance_channel(ch, sr)
            return res.astype(np.float32)  # type: ignore[no-any-return]

        if stereo:
            left = proc(audio[:, 0])
            right = proc(audio[:, 1])
            n = min(len(left), len(right), len(audio))
            out = np.stack([left[:n], right[:n]], axis=1)
        else:
            mono = audio[:, 0] if audio.ndim == 2 else audio
            out = proc(mono)

        # Restore channels-first layout if input was (2, N)
        if _was_channels_first and out.ndim == 2:
            out = out.T
        return np.clip(out, -1.0, 1.0)  # type: ignore[no-any-return]

    # ── Internal ────────────────────────────────────────────────────────────

    def _enhance_channel(self, mono: np.ndarray, sr: int) -> np.ndarray:
        """Verarbeite einen einzelnen Mono-Kanal."""
        # Resampling auf 48 kHz
        if sr != _SR:
            from scipy.signal import resample_poly

            g = math.gcd(sr, _SR)
            mono = resample_poly(mono, _SR // g, sr // g).astype(np.float32)

        if self._enc is not None:
            # §4.6b PLM-Active-Guard: prevent Emergency-Eviction during ONNX inference
            _plm_dfn = None
            try:
                from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm_dfn

                _plm_dfn = _get_plm_dfn()
                _plm_dfn.set_active("DeepFilterNetV3", True)
            except Exception:
                logger.warning("deepfilternet_v3_ii_plugin.py::_verbessern_channel Ersatzpfad", exc_info=True)
            try:
                out = self._infer_onnx(mono)
            finally:
                if _plm_dfn is not None:
                    try:
                        _plm_dfn.set_active("DeepFilterNetV3", False)
                    except Exception:
                        logger.warning("deepfilternet_v3_ii_plugin.py::_verbessern_channel Ersatzpfad", exc_info=True)
        else:
            out = self._omlsa_fallback(mono, _SR, energy_bias_db=self._current_energy_bias_db)

        # Rückresampling auf Original-SR
        if sr != _SR:
            from scipy.signal import resample_poly

            g = math.gcd(sr, _SR)
            out = resample_poly(out, sr // g, _SR // g).astype(np.float32)

        int(len(mono) * sr / _SR) if sr != _SR else len(mono)
        return out.astype(np.float32)  # type: ignore[no-any-return]

    def _compute_features(self, mono: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Berechne ERB-Features und Spektrum-Features aus mono-Audio.

        Vectorized batch-FFT: statt Python-Loop über n_frames wird eine
        Frame-Matrix per stride_tricks gebaut und np.fft.rfft batch-weise
        ausgeführt (~50-100× schneller bei langen Dateien).

        Returns:
            feat_erb   [1, 1, S, 32]
            feat_spec  [1, 2, S, 96]
            spec_cx    [n_fft//2+1, S] komplexes Spektrogramm
        """
        win = np.hanning(_N_FFT).astype(np.float32)
        n = len(mono)
        # Zero-pad auf ganzzahlige Hop-Anzahl
        n_frames = max(1, (n - _N_FFT) // _HOP + 1)
        padded_len = _HOP * n_frames + _N_FFT
        mono_p = np.zeros(padded_len, dtype=np.float32)
        mono_p[:n] = mono

        # Vectorized STFT: batch-FFT über alle Frames gleichzeitig
        indices = np.arange(n_frames)[:, np.newaxis] * _HOP + np.arange(_N_FFT)
        frames = mono_p[indices] * win  # [n_frames, _N_FFT]
        spec_cx = np.fft.rfft(frames, axis=1).astype(np.complex64).T  # [481, S]

        mag = np.abs(spec_cx).astype(np.float32)  # [481, S]

        # ERB-Energien: [32, S]
        erb_energy = _ERB_FB @ mag  # [32, S]
        erb_log = np.log1p(erb_energy).astype(np.float32)

        # feat_erb [1, 1, S, 32]
        feat_erb = erb_log.T[np.newaxis, np.newaxis, :, :]  # [1,1,S,32]

        # feat_spec: Real + Imag der ersten 96 Bins → [1, 2, S, 96]
        n_bins = min(_DF_BINS, spec_cx.shape[0])
        spec_slice = spec_cx[:n_bins, :]  # [96, S]
        feat_re = np.real(spec_slice).T[np.newaxis, np.newaxis, :, :]  # [1,1,S,96]
        feat_im = np.imag(spec_slice).T[np.newaxis, np.newaxis, :, :]  # [1,1,S,96]
        feat_spec = np.concatenate([feat_re, feat_im], axis=1)  # [1,2,S,96]

        return feat_erb.astype(np.float32), feat_spec.astype(np.float32), spec_cx

    def _apply_df_filter(self, spec_cx: np.ndarray, coefs: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Wende Deep-Filter-Koeffizienten auf komplexes Spektrum an (vektorisiert).

        Vectorized FIR-Filter: statt O(S × n_bins × DF_ORDER) Python-Iterationen
        werden nur _DF_ORDER (=10) NumPy-Array-Operationen durchgeführt.
        Beschleunigung: ~100-1000× gegenüber reinem Python-Loop.

        coefs: [S, 96, 10] DF-Koeffizienten
        alpha: [1, S, 1] oder skalar — Blending-Faktor (0..1)
        """
        n_bins = min(coefs.shape[1], spec_cx.shape[0])
        S = spec_cx.shape[1]
        result = spec_cx.copy()

        spec_sub = spec_cx[:n_bins, :]  # [n_bins, S]
        acc = np.zeros((n_bins, S), dtype=np.complex128)

        for k in range(_DF_ORDER):
            if k == 0:
                shifted = spec_sub
            else:
                shifted = np.empty_like(spec_sub)
                shifted[:, k:] = spec_sub[:, : S - k]
                shifted[:, :k] = spec_sub[:, 0:1]  # max(0, t-k) → clamp to t=0
            # coefs[:, :, k] shape [S, n_bins] → transpose to [n_bins, S]
            acc += coefs[:, :n_bins, k].T * shifted

        # Alpha-Blending: blend × FIR-Ergebnis + (1 - blend) × Original
        if alpha.ndim >= 2 and alpha.shape[1] >= S:
            blend = alpha[0, :S, 0].astype(np.float64)[np.newaxis, :]  # [1, S]
        else:
            blend = np.full((1, S), 0.5, dtype=np.float64)
        result[:n_bins, :] = (blend * acc + (1.0 - blend) * spec_sub).astype(spec_cx.dtype)

        return result

    @staticmethod
    def _apply_energy_bias_to_gain(gain: np.ndarray, energy_bias_db: float) -> np.ndarray:
        """§0j (dsp.instructions.md): Energy-Bias auf der Gain-Maske anwenden.

        Der ONNX-Graph hat keinen Noise-Floor-Input; die
        Entscheidungsgrenzen-Verschiebung ``noise_floor_estimate *=
        10^(energy_bias/20)`` wird äquivalent auf der Gain-Maske umgesetzt:
        negativer Bias hebt den Gain-Floor (Harmonik-Schutz), positiver
        senkt ihn (aggressivere NR). bias=0 → Identität.
        """
        if energy_bias_db == 0.0:
            return gain
        _bias_lin = float(10.0 ** (energy_bias_db / 20.0))
        return np.clip(gain + (1.0 - gain) * (1.0 - _bias_lin), 0.0, 1.0)

    def _infer_onnx(self, mono: np.ndarray) -> np.ndarray:
        """Vollständige 3-Modell ONNX-Inferenz-Pipeline."""
        feat_erb, feat_spec, spec_cx = self._compute_features(mono)

        try:
            # Encoder
            enc_out = self._enc.run(None, {"feat_erb": feat_erb, "feat_spec": feat_spec})
            # enc outputs: e0,e1,e2,e3,emb,c0,lsnr (Reihenfolge per Modell)
            e0, e1, e2, e3 = enc_out[0], enc_out[1], enc_out[2], enc_out[3]
            emb = enc_out[4]
            c0 = enc_out[5]

            # ERB-Dekoder → Maske [1,1,S,32]
            erb_out = self._erb_dec.run(None, {"emb": emb, "e3": e3, "e2": e2, "e1": e1, "e0": e0})
            erb_mask = erb_out[0]  # [1,1,S,32]

            # Haupt-Dekoder → DF-Koeffizienten + alpha
            dec_out = self._dec.run(None, {"emb": emb, "c0": c0})
            coefs = dec_out[0]  # [B, S, 96, 10]
            alpha = dec_out[1]  # sigmoid

            # ERB-Maske zurück auf FFT-Bins interpolieren
            m = erb_mask[0, 0, :, :]  # [S, 32]
            # Mappe ERB → linear (inverse des Filterbank-Produkts)
            gain_lin = _ERB_FB.T @ m.T  # [481, S]
            gain_lin = np.clip(gain_lin, 0.0, 1.0)
            # §0j (dsp.instructions.md): Energy-Bias auf der Gain-Maske
            # anwenden (ONNX-Graph bietet keinen Noise-Floor-Input).
            gain_lin = self._apply_energy_bias_to_gain(gain_lin, float(getattr(self, "_current_energy_bias_db", 0.0)))

            # ERB-Gain anwenden
            spec_filtered = spec_cx * gain_lin

            # DF-Filter anwenden
            coefs_np = coefs[0] if coefs.ndim == 4 else coefs
            alpha_np = alpha
            spec_filtered = self._apply_df_filter(spec_filtered, coefs_np, alpha_np)

        except Exception as exc:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            logger.debug("DeepFilterNet ONNX-Inferenz-Fehler: %s — DSP-Ersatzpfad.", exc)
            return self._omlsa_fallback(mono, _SR, energy_bias_db=float(getattr(self, "_current_energy_bias_db", 0.0)))

        # ISTFT (vectorized batch-IRFFT + overlap-add)
        win = np.hanning(_N_FFT).astype(np.float32)
        n_frames = spec_filtered.shape[1]
        n_out = _HOP * n_frames + _N_FFT

        # Batch-IRFFT: alle Frames auf einmal transformieren
        frames_out = np.fft.irfft(spec_filtered.T, n=_N_FFT, axis=1).astype(np.float32)  # [n_frames, _N_FFT]
        frames_out *= win  # Windowing

        # Overlap-Add (Loop bleibt, da in-place Akkumulation nötig — aber nur leichte Arithmetik)
        out = np.zeros(n_out, dtype=np.float32)
        win_sq = win * win
        win_sum = np.zeros(n_out, dtype=np.float32)
        for i in range(n_frames):
            s = i * _HOP
            out[s : s + _N_FFT] += frames_out[i]
            win_sum[s : s + _N_FFT] += win_sq

        win_sum = np.where(win_sum < 1e-8, 1.0, win_sum)
        out /= win_sum
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out[: len(mono)].astype(np.float32)  # type: ignore[no-any-return]

    @staticmethod
    def _estimate_input_snr_db(mono: np.ndarray, frame_len: int = 2048, hop: int = 512) -> float:
        """Grober Eingangs-SNR-Proxy zur Auswahl des Sekundärfallbacks."""
        x = np.nan_to_num(np.asarray(mono, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if x.size < frame_len:
            rms = float(np.sqrt(np.mean(x**2) + 1e-12))
            return 40.0 if rms < 1e-3 else 20.0
        frames = []
        for start in range(0, max(x.size - frame_len + 1, 1), hop):
            frames.append(x[start : start + frame_len])
        if not frames:
            frames = [x[:frame_len]]
        frame_rms = np.array([np.sqrt(np.mean(f.astype(np.float64) ** 2) + 1e-12) for f in frames], dtype=np.float64)
        signal_rms = float(np.percentile(frame_rms, 95))
        noise_rms = float(np.percentile(frame_rms, 10))
        return float(20.0 * np.log10((signal_rms + 1e-12) / (noise_rms + 1e-12)))

    @staticmethod
    def _spectral_gating_fallback(mono: np.ndarray, sr: int) -> np.ndarray:
        """Sekundärfallback: leichtes Spectral-Gating mit Originalphase."""
        from scipy.signal import istft, stft

        n_fft = 1024
        hop = n_fft // 4
        _noverlap = min(n_fft - hop, max(0, n_fft - 1))  # §v10.113
        # §v10.19-Fix: Längen-erhaltende STFT/iSTFT (scipy-Defaults kürzten 48000→47104)
        _n = len(mono)
        _frames_needed = max(0, int(np.ceil((_n - n_fft) / hop)))
        _n_pad = max(n_fft + _frames_needed * hop, _n)
        _mono_p = np.pad(mono, (0, _n_pad - _n))
        _, _, Zxx = stft(
            _mono_p, fs=sr, nperseg=n_fft, noverlap=_noverlap, window="hann", boundary="zeros", padded=True
        )
        mag = np.abs(Zxx)
        noise_est = np.percentile(mag, 20, axis=1, keepdims=True)
        noise_est = np.maximum(noise_est, 1e-8)
        mask = np.clip((mag - 1.25 * noise_est) / (mag + 1e-10), 0.05, 1.0)
        Zxx_out = mask * mag * np.exp(1j * np.angle(Zxx))
        # §DSP-Fix Rev. 2026-08-16: boundary-Paarung korrigiert (NOLA-Kantenspike,
        # gemessen edge_peak_ratio≈393 → ≈1, ref_snr −3,4 → positiv; dsp_benchmark).
        _, out = istft(Zxx_out, fs=sr, nperseg=n_fft, noverlap=_noverlap, window="hann", boundary=True)
        out = np.nan_to_num(out[:_n], nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(out, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

    @staticmethod
    def _secondary_fallback(mono: np.ndarray, sr: int) -> np.ndarray:
        """Sekundärfallback gemäß §2.47: Spectral-Gating oder Dry bei hohem SNR."""
        snr_db = DeepFilterNetV3Plugin._estimate_input_snr_db(mono)
        if snr_db > 35.0:
            logger.info("DeepFilterNet: hoher Eingangs-SNR %.1f dB — Dry-Signal statt Zusatzbearbeitung.", snr_db)
            return np.clip(np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]
        logger.info("DeepFilterNet: OMLSA fehlgeschlagen — Spectral-Gating-Ersatzpfad (SNR=%.1f dB).", snr_db)
        return DeepFilterNetV3Plugin._spectral_gating_fallback(mono, sr)

    @staticmethod
    def _omlsa_primary_fallback(mono: np.ndarray, sr: int, energy_bias_db: float = 0.0) -> np.ndarray:
        """OMLSA-Wiener-Filter Primärfallback (Cohen 2002).

        §0j (dsp.instructions.md): energy_bias verschiebt die
        Noise-Floor-Schätzung (noise_floor_estimate *= 10^(bias/20)).
        """
        from scipy.signal import istft, stft

        n_fft = 1024
        hop = n_fft // 4
        _noverlap = min(n_fft - hop, max(0, n_fft - 1))  # §v10.113
        # §v10.19-Fix: Längen-erhaltende STFT/iSTFT (scipy-Defaults kürzten 48000→47104)
        _n = len(mono)
        _frames_needed = max(0, int(np.ceil((_n - n_fft) / hop)))
        _n_pad = max(n_fft + _frames_needed * hop, _n)
        _mono_p = np.pad(mono, (0, _n_pad - _n))
        _, _, Zxx = stft(
            _mono_p, fs=sr, nperseg=n_fft, noverlap=_noverlap, window="hann", boundary="zeros", padded=True
        )

        mag = np.abs(Zxx)
        # MCRA-Rauschschätzung: Minima in gleitenden Fenstern (5 Frames)
        from scipy.ndimage import uniform_filter

        noise_est = uniform_filter(mag, size=(1, 5))
        noise_est = np.minimum(noise_est, mag)
        # §0j (dsp.instructions.md): energy_bias verschiebt die
        # Noise-Floor-Schätzung → noise_floor_estimate *= 10^(bias/20).
        # Negativer Bias senkt den geschätzten Noise-Floor → weniger
        # Suppression an Harmonik-Regionen (Vokal −6 dB, Instrumental −9 dB).
        if energy_bias_db != 0.0:
            noise_est = noise_est * (10.0 ** (energy_bias_db / 20.0))
        noise_est = np.maximum(noise_est, 1e-8)

        # MMSE-LSA Gain
        snr = np.maximum(mag**2 / (noise_est**2 + 1e-10), 0)
        gain = snr / (snr + 1)
        gain = np.maximum(gain, 0.1)  # G_floor = 0.1 (§2.62)

        Zxx_out = gain * mag * np.exp(1j * np.angle(Zxx))
        # §DSP-Fix Rev. 2026-08-16: boundary-Paarung korrigiert (NOLA-Kantenspike,
        # gemessen edge_peak_ratio≈393 → ≈1, ref_snr −3,4 → positiv; dsp_benchmark).
        _, out = istft(Zxx_out, fs=sr, nperseg=n_fft, noverlap=_noverlap, window="hann", boundary=True)
        return out[:_n].astype(np.float32)  # type: ignore[no-any-return]

    @staticmethod
    def _omlsa_fallback(mono: np.ndarray, sr: int, energy_bias_db: float = 0.0) -> np.ndarray:
        """OMLSA/IMCRA Primärfallback mit Spectral-Gating/Dry als Letztfallback."""
        try:
            return DeepFilterNetV3Plugin._omlsa_primary_fallback(mono, sr, energy_bias_db)
        except Exception as exc:
            logger.warning("DeepFilterNet OMLSA-Ersatzpfad fehlgeschlagen: %s — Sekundärfallback aktiv.", exc)
            return DeepFilterNetV3Plugin._secondary_fallback(mono, sr)


# ── Singleton ───────────────────────────────────────────────────────────────


def get_deepfilternet_plugin() -> DeepFilterNetV3Plugin:
    """Thread-sicherer Singleton (Double-Checked Locking)."""
    global _inst
    if _inst is None:
        with _lock:
            if _inst is None:
                _inst = DeepFilterNetV3Plugin()
    return _inst


def get_loaded_deepfilternet_plugin() -> DeepFilterNetV3Plugin | None:
    """Gibt nur eine bereits geladene DeepFilterNet-Instanz zurück, ohne Lazy-Load."""
    return _inst


def enhance_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Convenience-Wrapper für DeepFilterNet-Rauschunterdrückung."""
    return get_deepfilternet_plugin().enhance(audio, sr)


# Backwards-Compatibility-Alias (Klasse umbenannt bei Docker->ONNX-Refactor)
DeepFilterNetV3IIPlugin = DeepFilterNetV3Plugin
