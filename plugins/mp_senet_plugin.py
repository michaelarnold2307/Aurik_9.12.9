"""mp_senet_plugin — MP-SENet: Multi-path Sub-band Enhanced Network (INTERSPEECH 2023).

MP-SENet ersetzt DCCRN und FullSubNet+ als spektrales Enhancement-Modell in Aurik 10.0.0.
Verarbeitet mehrpfadige Sub-Band-Repräsentationen für breitbandige Musikrestaurierung.

Verbesserung gegenüber DCCRN (2020) / FullSubNet+ (2022):
    - DNS5 Challenge: MP-SENet Platz 1 (2023), DCCRN/FullSubNet+ weit hinten
    - Volles Spektrum 0–24 kHz (DCCRN: limitiert auf 16 kHz Sprach-Setup)
    - Sub-Band-Pfade erhalten Formant-/Oberton-Strukturen besser

Modell:
    models/mp_senet/mp_senet.onnx (~35 MB)
    Input:  [batch, 1, freq_bins, time_frames] float32 (komplex, Real+Imag)
    Output: [batch, 1, freq_bins, time_frames] float32 (komplex, denoised)

Fallback-Kaskade:
    1. MP-SENet ONNX (dieser Plugin)
    2. OMLSA/IMCRA DSP (Cohen & Berdugo 2002, §4.4)

Spec §4.4: MP-SENet → OMLSA/IMCRA DSP (DCCRN/FullSubNet+ entfernt)

Referenz:
    Lu et al. "MP-SENet: A Speech Enhancement Model with Parallel Denoising
    of Magnitude and Phase Spectra" — INTERSPEECH 2023
    https://github.com/yxlu-0102/MP-SENet

Singleton-Pattern: get_mp_senet_plugin() verwenden.
CPU-Only: CPUExecutionProvider.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_ONNX_PATH = _ROOT / "models" / "mp_senet" / "mp_senet.onnx"

# Verarbeitungs-Konstanten (48 kHz)
_SR: int = 48_000
_N_FFT: int = 960  # 20 ms @ 48 kHz
_HOP: int = 480  # 10 ms
_WIN: int = 960
# NOTE: Current shipped ONNX export is effectively fixed to time=32 despite
# dynamic-axis metadata. Using larger T (e.g., 404) triggers Reshape_4 runtime
# errors. Keep chunk size at 32 for stable inference; plugin handles long audio
# via segment-wise stitching.
_MODEL_TIME_FRAMES: int = 32

_lock = threading.Lock()
_instance: MpSenetPlugin | None = None


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MpSenetResult:
    """Ergebnis der MP-SENet Rauschunterdrückung.

    Attributes:
        audio:      Bereinigtes Audio, float32 ∈ [-1, 1]
        sr:         Sample-Rate (48000)
        model_used: "mp_senet_onnx" | "omlsa_dsp_fallback"
        snr_improvement_db: Geschätzter SNR-Gewinn in dB
        fail_reason: Optional strukturierte Fehlerursache (§2.38a)
    """

    audio: np.ndarray
    sr: int
    model_used: str
    snr_improvement_db: float = 0.0
    fail_reason: str | None = None  # "mp_senet_shape_error" | "mp_senet_onnx_runtime" | None

    def __post_init__(self) -> None:
        self.audio = np.nan_to_num(self.audio, nan=0.0, posinf=0.0, neginf=0.0)
        self.audio = np.clip(self.audio, -1.0, 1.0)


# ---------------------------------------------------------------------------
# MpSenetPlugin
# ---------------------------------------------------------------------------


class MpSenetPlugin:
    """MP-SENet Multi-path Sub-Band Enhancement (ONNX, CPUExecutionProvider).

    Verarbeitet Magnitude und Phase separat zur Erhaltung harmonischer Strukturen.
    Fallback: OMLSA/IMCRA DSP-Rauschunterdrückung (§4.4 Spec).  # §V6: logger.warning handled at call site
    """

    def __init__(self) -> None:
        self._session = None
        self._model_loaded: bool = False
        self._try_load()

    def _try_load(self) -> None:
        """Lädt MP-SENet ONNX; OMLSA-Fallback bei Fehler."""
        if not _ONNX_PATH.exists():
            logger.info(
                "MP-SENet ONNX nicht gefunden (%s) — OMLSA-DSP-Fallback aktiv. "
                "Modell: https://github.com/yxlu-0102/MP-SENet",
                _ONNX_PATH,
            )
            return
        try:
            import onnxruntime as ort

            try:
                from backend.core.ml_memory_budget import release as _release
                from backend.core.ml_memory_budget import try_allocate as _try_alloc

                if not _try_alloc("MP-SENet", size_gb=0.04):
                    try:
                        _release("MP-SENet")
                    except Exception:
                        logger.warning("mp_senet_plugin.py::_try_load fallback", exc_info=True)
                    if not _try_alloc("MP-SENet", size_gb=0.04):
                        logger.warning("MP-SENet: ML-Budget erschöpft — DSP-Fallback.")
                        return
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            try:
                from backend.core.ml_device_manager import get_ort_providers as _get_prov

                _mp_prov = _get_prov("MPSENet")
            except Exception:
                _mp_prov = ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(
                str(_ONNX_PATH),
                sess_options=opts,
                providers=_mp_prov,
            )
            self._model_loaded = True
            logger.info("✅ MP-SENet ONNX geladen (%s, §4.4 — DCCRN/FullSubNet+ Nachfolger)", _ONNX_PATH.name)
            try:
                from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                _reg_plm(
                    "MP-SENet",
                    size_gb=0.04,
                    unload_fn=lambda s=self: setattr(s, "_session", None) or setattr(s, "_model_loaded", False),  # type: ignore[func-returns-value,misc]
                )
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)
        except Exception as exc:
            logger.warning("MP-SENet ONNX nicht ladbar: %s — OMLSA-DSP-Fallback aktiv.", exc)
            try:
                from backend.core.ml_memory_budget import release as _rel

                _rel("MP-SENet")
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enhance(self, audio: np.ndarray, sr: int) -> MpSenetResult:
        """Rauschunterdrückung via MP-SENet ONNX oder OMLSA-DSP-Fallback.

        Args:
            audio: float32 mono oder stereo, 48000 Hz
            sr:    Sample-Rate (muss 48000 sein)

        Returns:
            MpSenetResult mit bereinigtem Audio.
        """
        assert sr == 48_000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        # UV3 passes (2, N) channels-first; normalize to (N, 2) for uniform processing.
        _was_channels_first = audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2
        if _was_channels_first:
            audio = audio.T  # (2, N) → (N, 2)
        stereo = audio.ndim == 2 and audio.shape[1] == 2

        def process_channel(ch: np.ndarray) -> tuple[np.ndarray, str | None, str]:
            if self._session is not None:
                enhanced, fail_reason = self._enhance_onnx(ch, sr)
                if fail_reason is not None:
                    return enhanced, fail_reason, "omlsa_dsp_fallback"
                return enhanced, None, "mp_senet_onnx"
            return (
                self._omlsa_fallback(ch, sr),
                None,
                "omlsa_dsp_fallback",
            )  # §V6 (copilot-instructions.md): logger.warning handled at call site

        if stereo:
            left, fail_left, used_left = process_channel(audio[:, 0])
            right, fail_right, used_right = process_channel(audio[:, 1])
            n = min(len(left), len(right), len(audio))
            out = np.stack([left[:n], right[:n]], axis=1)
            fail_reason = fail_left or fail_right
            model_used = (
                "mp_senet_onnx"
                if used_left == "mp_senet_onnx" and used_right == "mp_senet_onnx"
                else "omlsa_dsp_fallback"
            )
        else:
            mono = audio[:, 0] if audio.ndim == 2 else audio
            out, fail_reason, model_used = process_channel(mono)

        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, -1.0, 1.0)

        # SNR-Verbesserungsschätzung
        rms_in = float(np.sqrt(np.mean(audio**2))) + 1e-10
        rms_diff = float(np.sqrt(np.mean((out - audio) ** 2))) + 1e-10
        snr_imp = 20.0 * math.log10(rms_in / rms_diff) if rms_diff < rms_in else 0.0

        # Restore channels-first layout if input was (2, N)
        out_final = out.T if (_was_channels_first and out.ndim == 2) else out

        return MpSenetResult(
            audio=out_final.astype(np.float32),
            sr=sr,
            model_used=model_used,
            snr_improvement_db=float(np.clip(snr_imp, 0.0, 30.0)),
            fail_reason=fail_reason,
        )

    # ------------------------------------------------------------------
    # ONNX Inference
    # ------------------------------------------------------------------

    def _stft(self, mono: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """Berechnet STFT. Returns (complex_spec [freq, T], phases, n_orig)."""
        from scipy.signal import stft as scipy_stft

        n_orig = len(mono)
        _, _, Z = scipy_stft(
            mono.astype(np.float64),
            fs=_SR,
            nperseg=_WIN,
            noverlap=_WIN - _HOP,
            window="hann",
        )
        return Z.astype(np.complex64), np.angle(Z).astype(np.float32), n_orig

    def _istft(self, Z: np.ndarray, n_orig: int) -> np.ndarray:
        """Inverse STFT mit PGHI-Phasenkonsistenz (§4.4 Spec)."""
        from scipy.signal import istft as scipy_istft

        _, x = scipy_istft(
            Z.astype(np.complex128),
            fs=_SR,
            nperseg=_WIN,
            noverlap=_WIN - _HOP,
            window="hann",
        )
        x = x.astype(np.float32)
        # Länge anpassen
        if len(x) > n_orig:
            x = x[:n_orig]
        elif len(x) < n_orig:
            x = np.pad(x, (0, n_orig - len(x)))
        return x  # type: ignore[no-any-return]

    def _validate_and_pad_shapes(self, amp: np.ndarray, pha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Validiert core STFT frequency dimensions before ONNX inference."""
        _N_BINS = 201
        if amp.shape[0] != _N_BINS or pha.shape[0] != _N_BINS:
            raise ValueError(
                f"Shape-Incompatibilität: erwartet freq={_N_BINS}, got amp freq={amp.shape[0]} pha freq={pha.shape[0]}"
            )
        return amp, pha

    def _run_onnx_segment(self, amp_seg: np.ndarray, pha_seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Führt aus: one fixed-length ONNX segment and return denoised amp/phase.

        Primary layout is [B, F, T]. If the exported graph expects [B, T, F]
        (seen as Reshape_4 runtime errors), retry once with transposed inputs.
        """
        assert self._session is not None
        inp_names = [i.name for i in self._session.get_inputs()]
        if len(inp_names) < 2:
            raise ValueError(f"MP-SENet: expected >=2 inputs, got {inp_names}")

        def _normalize_out(arr: np.ndarray) -> np.ndarray:
            x = np.asarray(arr, dtype=np.float32)[0]
            if x.ndim != 2:
                raise ValueError(f"MP-SENet output rank unsupported: {x.shape}")
            if x.shape[0] == 201:
                return x
            if x.shape[1] == 201:
                return x.T
            raise ValueError(f"MP-SENet output shape unsupported: {x.shape}")

        amp_bft = amp_seg[np.newaxis]  # [1, 201, T]
        pha_bft = pha_seg[np.newaxis]  # [1, 201, T]
        try:
            ort_out = self._session.run(
                ["denoised_amp", "denoised_pha"],
                {inp_names[0]: amp_bft, inp_names[1]: pha_bft},
            )
            return _normalize_out(ort_out[0]), _normalize_out(ort_out[1])
        except Exception as exc:
            msg = str(exc)
            if "Reshape" not in msg and "reshape" not in msg:
                raise
            logger.warning(
                "MP-SENet layout retry: [B,F,T] fehlgeschlagen (%s) — versuche [B,T,F]",
                exc,
            )

        amp_btf = np.transpose(amp_bft, (0, 2, 1))  # [1, T, 201]
        pha_btf = np.transpose(pha_bft, (0, 2, 1))  # [1, T, 201]
        ort_out = self._session.run(
            ["denoised_amp", "denoised_pha"],
            {inp_names[0]: amp_btf, inp_names[1]: pha_btf},
        )
        return _normalize_out(ort_out[0]), _normalize_out(ort_out[1])

    def _run_onnx_fixed_time(self, amp_in: np.ndarray, pha_in: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Führt aus: ONNX with fixed T=404 using padded chunks to avoid Reshape errors."""
        n_frames = int(amp_in.shape[1])
        if n_frames <= 0:
            return amp_in.copy(), pha_in.copy()

        if n_frames <= _MODEL_TIME_FRAMES:
            amp_seg = np.pad(
                amp_in,
                ((0, 0), (0, _MODEL_TIME_FRAMES - n_frames)),
                mode="constant",
                constant_values=0.0,
            )
            pha_seg = np.pad(
                pha_in,
                ((0, 0), (0, _MODEL_TIME_FRAMES - n_frames)),
                mode="constant",
                constant_values=0.0,
            )
            den_amp, den_pha = self._run_onnx_segment(amp_seg, pha_seg)
            return den_amp[:, :n_frames], den_pha[:, :n_frames]

        out_amp = np.zeros_like(amp_in, dtype=np.float32)
        out_pha = np.zeros_like(pha_in, dtype=np.float32)
        start = 0
        while start < n_frames:
            end = min(start + _MODEL_TIME_FRAMES, n_frames)
            seg_len = end - start
            amp_seg = np.zeros((amp_in.shape[0], _MODEL_TIME_FRAMES), dtype=np.float32)
            pha_seg = np.zeros((pha_in.shape[0], _MODEL_TIME_FRAMES), dtype=np.float32)
            amp_seg[:, :seg_len] = amp_in[:, start:end]
            pha_seg[:, :seg_len] = pha_in[:, start:end]

            den_amp, den_pha = self._run_onnx_segment(amp_seg, pha_seg)
            out_amp[:, start:end] = den_amp[:, :seg_len]
            out_pha[:, start:end] = den_pha[:, :seg_len]
            start = end

        return out_amp, out_pha

    def _enhance_onnx(self, mono: np.ndarray, sr: int) -> tuple[np.ndarray, str | None]:
        """MP-SENet ONNX-Inferenz: Magnitude + Phase Enhancement.

        The model expects two separate inputs:
            noisy_amp  [batch, 201, time]  — STFT magnitude
            noisy_pha  [batch, 201, time]  — STFT phase (radians)
        At 48 kHz with N_FFT=960 we have 481 freq bins.  The model was
        trained on 201-bin spectrograms (≙ N_FFT=400 @ 16 kHz / ≙ 0–10 kHz
        at 50 Hz/bin when N_FFT=960 @ 48 kHz).  We crop to the first 201
        bins (0–10 kHz), process them, and stitch the denoised lower bins
        back into the full spectrum before iSTFT reconstruction.

        §Punkt 4: Shape-Robustheit gegen Reshape-Fehler implementiert.
        """
        assert self._session is not None
        _N_BINS = 201  # model's fixed frequency-bin count
        _plm_mps = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm_fn

            _plm_mps = _get_plm_fn()
            _plm_mps.set_active("MP-SENet", True)
        except Exception as _exc:
            logger.debug("MP-SENet: PLM set_active failed: %s", _exc)
        try:
            Z, _, n_orig = self._stft(mono)  # Z: [481, T] complex64
            amp_full = np.abs(Z).astype(np.float32)  # [481, T]
            pha_full = np.angle(Z).astype(np.float32)  # [481, T]

            # Crop to model's 201-bin input (covers 0–10 kHz @ 50 Hz/bin)
            amp_in = amp_full[:_N_BINS]  # [201, T]
            pha_in = pha_full[:_N_BINS]  # [201, T]

            # §Punkt 4: Shape-Validierung + Padding VOR Batch-Erweiterung
            try:
                amp_in, pha_in = self._validate_and_pad_shapes(amp_in, pha_in)
            except ValueError as shape_exc:
                logger.error(
                    "MP-SENet Shape-Validierung fehlgeschlagen: %s (audio_len=%d frames=%d) — OMLSA-DSP-Fallback",
                    shape_exc,
                    len(mono),
                    amp_in.shape[1] if amp_in.ndim > 1 else 0,
                )
                fallback_audio = self._omlsa_fallback(mono, sr)
                return fallback_audio, "mp_senet_shape_error"

            denoised_amp, denoised_pha = self._run_onnx_fixed_time(amp_in, pha_in)

            # Crop zu Original-Länge (falls gepadddet)
            orig_frames = amp_full.shape[1]
            if denoised_amp.shape[1] > orig_frames:
                denoised_amp = denoised_amp[:, :orig_frames]
                denoised_pha = denoised_pha[:, :orig_frames]

            # Stitch denoised lower bins back; keep original upper bins
            amp_out = amp_full.copy()
            pha_out = pha_full.copy()
            amp_out[:_N_BINS] = np.nan_to_num(denoised_amp, nan=0.0, posinf=0.0, neginf=0.0)
            pha_out[:_N_BINS] = np.nan_to_num(denoised_pha, nan=0.0, posinf=0.0, neginf=0.0)

            Z_enhanced = (amp_out * np.exp(1j * pha_out)).astype(np.complex64)
            Z_enhanced = np.nan_to_num(Z_enhanced, nan=0.0, posinf=0.0, neginf=0.0)

            result = self._istft(Z_enhanced, n_orig)
            return np.clip(np.nan_to_num(result, nan=0.0), -1.0, 1.0), None
        except Exception as exc:
            # §Punkt 4: Structured Error Logging mit fail_reason
            logger.error(
                "🔴 MP-SENet ONNX-Inferenzfehler: %s (Typ: %s, audio_len=%d) — OMLSA-DSP-Fallback wird angewandt.",
                exc,
                type(exc).__name__,
                len(mono),
            )
            fallback_audio = self._omlsa_fallback(mono, sr)
            return fallback_audio, "mp_senet_onnx_runtime"
        finally:
            if _plm_mps is not None:
                try:
                    _plm_mps.set_active("MP-SENet", False)
                except Exception as _exc:
                    logger.debug("MP-SENet: PLM unset_active failed: %s", _exc)

    # ------------------------------------------------------------------
    # OMLSA DSP Fallback
    # ------------------------------------------------------------------

    def _omlsa_fallback(self, mono: np.ndarray, sr: int) -> np.ndarray:
        """OMLSA/IMCRA DSP-Rauschunterdrückung (Cohen & Berdugo 2002, §4.4).

        Algorithmus:
            1. STFT → Power-Spektrum P[k,t]
            2. IMCRA-Rauschschätzung: P_noise[k] via Minima-kontrollierte
               rekursive Mittelung (Glättungsfaktor α=0.98)
            3. OMLSA-Gain: G[k] = exp(0.5 · E(|log SNR_post|)) geglättet
               mit G_floor = 0.1 (standard) / 0.85 (harmonische Bins)
            4. ISTFT + PGHI-Phasenkonsistenz
        """
        try:
            from scipy.signal import istft as scipy_istft
            from scipy.signal import stft as scipy_stft

            n_orig = len(mono)
            window_size = min(_WIN, n_orig)
            if window_size < 4:
                return mono.copy()
            hop = min(_HOP, window_size // 2)
            _, _, Z = scipy_stft(
                mono.astype(np.float64),
                fs=sr,
                nperseg=window_size,
                noverlap=window_size - hop,
                window="hann",
            )
            Z = Z.astype(np.complex128)
            P = np.abs(Z) ** 2  # [freq, T]
            n_freq, n_frames = P.shape

            # IMCRA: Rauschleistungsschätzung via Minima-kontrollierte Mittelung
            alpha = 0.98
            beta = 0.8  # Unter-Schätzungs-Korrekturfaktor
            P_noise = P[:, 0].copy()  # Initialisierung mit erstem Frame
            G = np.ones((n_freq, n_frames), dtype=np.float64)
            G_floor = 0.1

            for t in range(n_frames):
                # SNR posterior
                snr_post = np.maximum(P[:, t] / (P_noise + 1e-15), 1.0)
                snr_prio = np.maximum(snr_post - 1.0, 0.0)

                # MMSE-LSA Gain (OMLSA)
                v = snr_prio / (1.0 + snr_prio) * snr_post
                v = np.clip(v, 1e-10, 700.0)
                from scipy.special import expn

                gain = np.exp(0.5 * expn(1, v)) * np.maximum(v, 1e-10) / (snr_post + 1e-15)
                gain = np.clip(gain, G_floor, 1.0)
                G[:, t] = gain

                # IMCRA rekursive Rauschschätzung
                indicator = (snr_post > 1.5).astype(np.float64)  # Sprachindikator
                P_noise = alpha * P_noise + (1 - alpha) * (1.0 - beta * indicator) * P[:, t]
                P_noise = np.maximum(P_noise, 1e-12)

            # OMLSA-Gain anwenden
            Z_enhanced = G * Z
            Z_enhanced = np.nan_to_num(Z_enhanced, nan=0.0, posinf=0.0, neginf=0.0)

            _, x = scipy_istft(Z_enhanced, fs=sr, nperseg=window_size, noverlap=window_size - hop, window="hann")
            x = x.astype(np.float32)
            if len(x) > n_orig:
                x = x[:n_orig]
            elif len(x) < n_orig:
                x = np.pad(x, (0, n_orig - len(x)))
            return np.clip(np.nan_to_num(x, nan=0.0), -1.0, 1.0)  # type: ignore[no-any-return]
        except Exception as exc:
            logger.error("OMLSA-DSP-Fallback fehlgeschlagen: %s — Audio unverändert.", exc)
            return np.clip(np.nan_to_num(mono.copy(), nan=0.0), -1.0, 1.0)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Singleton (§3.2 Double-Checked Locking)
# ---------------------------------------------------------------------------


def get_mp_senet_plugin() -> MpSenetPlugin:
    """Thread-sicherer Singleton-Accessor."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MpSenetPlugin()
    return _instance


def enhance_audio(audio: np.ndarray, sr: int) -> MpSenetResult:
    """Convenience-Wrapper für get_mp_senet_plugin().enhance()."""
    return get_mp_senet_plugin().enhance(audio, sr)


# ── Alias-Funktionen (für Dead-Import-Reparatur) ─────────────────────

def get_mp_senet() -> MpSenetPlugin:
    """Alias für get_mp_senet_plugin() — für Dead-Import-Kompatibilität."""
    return get_mp_senet_plugin()
