"""SOTA Spectral Denoiser — OMLSA/IMCRA (§4.1/§4.2, Cohen 2002/2003).

Aurik-konforme Neuimplementierung des spektralen Denoisers:

- **IMCRA-Rauschschätzung** (Improved Minima-Controlled Recursive Averaging,
  Cohen 2003) mit Zwei-Halb-Fenster-Minimum (O(1) pro Frame) und
  Sprach-Präsenz-Wahrscheinlichkeit über lokale Minimum-Statistik.
- **OMLSA-Gain** (Optimal Modified Log-Spectral Amplitude, Cohen & Berdugo
  2002): G = xi/(1+xi) * exp(0.5*E1(v)) mit decision-directed a-priori-SNR.
  §4.2: Klassischer Wiener als Primärverarbeitung ist VERBOTEN — OMLSA/IMCRA
  ist der normative Ersatz (Spec 04_dsp_standards §4.1).
- **Tonal-Protection**: Stationäre Sinusanteile (Musik) dürfen nicht als
  Rauschen klassifiziert werden — Bins > 25 dB über dem lokalen Median bzw.
  Rauschboden bleiben unangetastet (G=1).
- **Konsistentes STFT/ISTFT-Paar**: boundary="zeros" wird beidseitig explizit
  gesetzt. Damit ist die Rekonstruktion sowohl mit vanilla scipy als auch
  unter dem backend-eigenen signal.stft-Wrapper (§v10.115) frame-korrekt —
  kein Lag/Versatz, Ausgabelänge == Eingabelänge.
- **Stereo (N, C)**: Gain wird auf dem Mono-Mix geschätzt und identisch auf
  beide Kanäle angewendet (verlinkte Kanäle, erhält das Stereo-Bild).
- **ML-Hook**: optionaler ML-Ausgang (Deep Spectral Masking / DeepFilterNet /
  SGMSE) wird über backend.core.dsp.hybrid_ml_blend.hybrid_ml_apply
  (§G104 JND-Gate, §G101 Perceptual-Blend, §8.2 Energie-Guard) eingeblendet.
  Ohne backend (Standalone/DSP-only) deterministischer skalarer Fallback.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.signal import istft, stft
from scipy.special import exp1

logger = logging.getLogger(__name__)

# IMCRA/OMLSA-Konstanten (Cohen 2003 / Cohen & Berdugo 2002)
_ALPHA_SMOOTH = 0.85  # Zeitglättung der Periodogramm-Schätzung
_ALPHA_DD = 0.98  # Decision-Directed Gewicht (a-priori-SNR)
_XI_MIN = 10 ** (-2.5)  # Floor a-priori-SNR (~-25 dB)
_GAMMA_FLOOR = 1e-3
_SPEECH_DELTA = 5.0  # Lokale-Minimum-Ratio-Schwelle für Sprach-Präsenz (≈7 dB)
_TONAL_PROTECT_DB = 25.0  # Bins > 25 dB über Median/Rauschboden gelten als tonal


class SpectralDenoiser:
    """OMLSA/IMCRA-Rauschunterdrückung nach Spec 04 (§4.1), §4.2-konform.

    Args:
        n_fft: FFT-Größe (Standard 1024 @ 48 kHz, 512 @ 16 kHz sinnvoll).
        hop_length: Hop-Size (n_fft // 4 als Standard).
        noise_profile_frames: IMCRA-Minimum-Fenster in Frames (W).
        reduction_db: Maximale Dämpfung in dB → OMLSA-Gain-Floor.
        alpha_smooth: IMCRA-Recursive-Averaging-Faktor.
        speech_delta: Schwelle für Sprach-Präsenz (lokale Minimum-Ratio).
        tonal_protect_db: Tonal-Schutzschwelle in dB (0 deaktiviert).
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        noise_profile_frames: int = 10,
        reduction_db: float = 18.0,
        alpha_smooth: float = _ALPHA_SMOOTH,
        speech_delta: float = _SPEECH_DELTA,
        tonal_protect_db: float = _TONAL_PROTECT_DB,
    ) -> None:
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.noise_profile_frames = int(noise_profile_frames)
        self.reduction_db = float(reduction_db)
        self.alpha_smooth = float(alpha_smooth)
        self.speech_delta = float(speech_delta)
        self.tonal_protect_db = float(tonal_protect_db)

    # ------------------------------------------------------------------ #
    #  Öffentliche API                                                    #
    # ------------------------------------------------------------------ #

    def process(
        self,
        audio: npt.NDArray[np.floating],
        sr: int,
        *,
        ml_output: npt.NDArray[np.floating] | None = None,
        ml_wet: float = 1.0,
        material_type: str = "unknown",
        genre: str = "unknown",
    ) -> npt.NDArray[np.floating]:
        """Denoist `audio` (mono [T] oder stereo [T, C]/[2, T]).

        Returns:
            Denoisiertes Signal in exakt der Eingabe-Shape und -Länge
            (Lag-Integrität). dtype float32 bei float32-Eingabe.

        Kwargs:
            ml_output: Optionaler ML-verarbeiteter Ausgang (gleiche Shape).
                Wird per hybrid_ml_apply (§G104/§G101/§8.2) eingeblendet.
            ml_wet: Maximaler Wet-Anteil für den ML-Hook [0, 1].
            material_type/genre: Adaptivität für den Blend (JND-Faktoren).
        """
        audio = np.asarray(audio)
        orig_dtype = audio.dtype
        in_dtype = np.float64 if orig_dtype == np.float64 else np.float32
        x = np.nan_to_num(audio.astype(in_dtype), nan=0.0, posinf=0.0, neginf=0.0)

        if x.ndim not in (1, 2) or x.size == 0:
            return audio.copy()

        # §23-TONALITY (Vorschlag 01): tonales/sauberes Signal ⇒ DSP-NR überspringen
        # (Passthrough). Mit ml_output greift das Gate nicht — der ML-Ausgang wird
        # dann bewusst über die Hybrid-Naht geblendet.
        if ml_output is None:
            try:
                from backend.core.dsp.tonality_gate import is_tonal_clean  # pylint: disable=import-outside-toplevel

                if is_tonal_clean(x, sr):
                    logger.info("spectral_denoiser: tonality_gate → Passthrough (tonales/sauberes Signal)")
                    return audio.copy()
            except Exception as _tg_exc:  # nicht blockierend
                logger.debug("spectral_denoiser: tonality_gate nicht verfügbar (%s)", _tg_exc)

        # Channels-last-Normalisierung: (T,) oder (T, C)
        ch_first = x.ndim == 2 and x.shape[0] == 2 and x.shape[1] > 2
        if ch_first:
            x = x.T
        n_orig = x.shape[0]

        n_fft = max(16, int(self.n_fft))
        hop = max(1, int(self.hop_length))
        if n_fft <= hop:
            n_fft = hop + 1
        if n_fft > n_orig:
            n_fft = max(4, n_orig // 2)
            hop = max(1, n_fft // 4)
        # §v10.115/§v10.119: explizit boundary="zeros" auf BEIDEN Seiten —
        # frame-konsistent unter vanilla scipy UND backend-signal-Wrapper.
        _stft_kw = {"fs": sr, "nperseg": n_fft, "noverlap": n_fft - hop, "window": "hann", "boundary": "zeros"}

        channels = [x] if x.ndim == 1 else [x[:, c] for c in range(x.shape[1])]
        specs = [stft(ch, **_stft_kw)[2] for ch in channels]

        # Gain aus Mono-Mix (Stereo: verlinkte Kanäle, kein unabhängiges Gating)
        mix = channels[0] if len(channels) == 1 else np.mean(cast(npt.NDArray[np.floating], np.stack(channels, axis=0)), axis=0)
        _, _, Z_mix = stft(mix, **_stft_kw)
        gain = self._compute_omlsa_gain(np.abs(Z_mix) ** 2)

        outs: list[npt.NDArray[np.floating]] = []
        for Z_ch in specs:
            _, out_ch = istft(Z_ch * gain, **_stft_kw)
            out_ch = np.real(out_ch)
            if len(out_ch) >= n_orig:
                out_ch = out_ch[:n_orig]
            else:
                out_ch = np.pad(out_ch, (0, n_orig - len(out_ch)))
            outs.append(out_ch)

        out = outs[0] if len(outs) == 1 else np.column_stack(outs)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, -1.0, 1.0)

        # §ML-Hook: Optionaler ML-Ausgang über die kanonische Hybrid-Naht.
        if ml_output is not None:
            out = self._blend_ml(out, ml_output, sr, ml_wet, material_type, genre, ch_first)

        if ch_first:
            out = out.T
        return out.astype(orig_dtype)

    # ------------------------------------------------------------------ #
    #  OMLSA/IMCRA-Kern                                                   #
    # ------------------------------------------------------------------ #

    def _compute_omlsa_gain(self, periodogram: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """IMCRA-Rauschschätzung + OMLSA-Gain (Cohen 2002/2003).

        Returns gain matrix (n_freqs, n_frames) in [g_min, 1.0] mit
        Tonal-Protection (G=1 für stationäre Sinusanteile).
        """
        n_freqs, n_frames = periodogram.shape
        P = np.maximum(periodogram, 1e-12)

        W = max(int(self.noise_profile_frames), 15)
        half_w = max(2, W // 2)

        g_min = float(10 ** (-max(0.0, self.reduction_db) / 20.0))
        alpha = float(np.clip(self.alpha_smooth, 0.5, 0.98))
        delta = max(1.0, float(self.speech_delta))

        lam_d = P[:, 0].copy()  # Rauschschätzung, init mit erstem Frame
        P_smooth = P[:, 0].copy()
        gain_prev = np.ones(n_freqs)
        gamma_prev = np.maximum(P[:, 0] / (lam_d + 1e-12), _GAMMA_FLOOR)

        min_prev_half = np.full(n_freqs, np.inf)
        min_curr_half = np.full(n_freqs, np.inf)
        gains = np.empty_like(P)

        for t in range(n_frames):
            P_smooth = alpha * P_smooth + (1.0 - alpha) * P[:, t]

            # Zwei-Halb-Fenster-Minimum (O(1) amortisiert, Cohen-2003-Kernidee)
            if t % half_w == 0:
                min_prev_half = min_curr_half
                min_curr_half = np.full(n_freqs, np.inf)
            min_curr_half = np.minimum(min_curr_half, P_smooth)
            s_min = np.minimum(min_prev_half, min_curr_half)

            # Sprach-Präsenz: Ratio lokale Energie / lokales Minimum
            ratio = P_smooth / np.maximum(s_min, 1e-12)
            p_presence = np.clip((ratio - 1.0) / (delta - 1.0), 0.0, 1.0)
            alpha_d = alpha + (1.0 - alpha) * p_presence
            lam_d = alpha_d * lam_d + (1.0 - alpha_d) * P[:, t]
            lam_d = np.maximum(lam_d, 1e-12)

            # A-posteriori- und decision-directed a-priori-SNR
            gamma = np.maximum(P[:, t] / lam_d, _GAMMA_FLOOR)
            xi = _ALPHA_DD * (gain_prev**2) * gamma_prev + (1.0 - _ALPHA_DD) * np.maximum(gamma - 1.0, 0.0)
            xi = np.maximum(xi, _XI_MIN)

            # OMLSA-Gain: G = xi/(1+xi) * exp(0.5 * E1(v)), v = xi*gamma/(1+xi)
            v = xi * gamma / (1.0 + xi)
            with np.errstate(over="ignore", invalid="ignore"):
                g_omlsa = (xi / (1.0 + xi)) * np.exp(0.5 * exp1(np.maximum(v, 0.0)))
            g_omlsa = np.nan_to_num(g_omlsa, nan=g_min, posinf=1.0, neginf=g_min)
            g = np.clip(g_omlsa, g_min, 1.0)

            # Tonal-Protection: stationäre Sinusanteile nicht als Rauschen dämpfen
            if self.tonal_protect_db > 0.0:
                ref = np.maximum(np.median(P_smooth), np.median(lam_d)) * (10.0 ** (self.tonal_protect_db / 10.0))
                g = np.where(P_smooth > ref, 1.0, g)

            gains[:, t] = g
            gain_prev = g
            gamma_prev = gamma

        return gains

    # ------------------------------------------------------------------ #
    #  ML-Hook                                                            #
    # ------------------------------------------------------------------ #

    def _blend_ml(
        self,
        dsp_out: npt.NDArray[np.floating],
        ml_output: npt.NDArray[np.floating],
        sr: int,
        ml_wet: float,
        material_type: str,
        genre: str,
        ch_first: bool,
    ) -> npt.NDArray[np.floating]:
        """Blendet ML-Ausgang deterministisch auf den DSP-Ausgang ein."""
        ml = np.asarray(ml_output)
        if ml.ndim == 2 and ch_first:
            ml = ml.T
        ml = np.nan_to_num(ml.astype(dsp_out.dtype), nan=0.0, posinf=0.0, neginf=0.0)
        if ml.shape != dsp_out.shape:
            logger.debug("spectral_denoiser: ml_output Shape-Mismatch %s vs %s — DSP-only", ml.shape, dsp_out.shape)
            return dsp_out
        wet = float(np.clip(ml_wet, 0.0, 1.0))
        if wet <= 0.0:
            return dsp_out
        try:
            from backend.core.dsp.hybrid_ml_blend import hybrid_ml_apply  # pylint: disable=import-outside-toplevel

            out = hybrid_ml_apply(
                dsp_out,
                ml,
                sr,
                scalar_wet=wet,
                material_type=material_type,
                genre=genre,
            )
        except Exception as _hml_exc:  # backend nicht verfügbar → deterministischer Fallback
            logger.debug("spectral_denoiser: hybrid_ml_apply nicht verfügbar (%s) — skalarer Blend", _hml_exc)
            out = dsp_out + wet * (ml - dsp_out)
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(dsp_out.dtype)
