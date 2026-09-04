"""
Phase 55: Diffusion-Inpainting v1.0 — Masked Spectral Reconstruction
======================================================================

Weltklasse-Inpainting für Audio-Lücken und Dropouts >20ms, basierend auf
iterativer Diffusion im Spektralbereich (DDPM-inspiriert, pure DSP-Fallback +
optionaler DiffWave/AudioLDM2-Plugin-Pfad).

ALGORITHMUS — Dreistufige Masked Diffusion:
  1. **Lücken-Detektion** (Defect Mask):
     - RMS-Energie <  -60 dBFS für ≥ min_gap_ms Millisekunden → Dropout
     - Phase-Diskontinuität > π/2 zwischen Frames → Phasenbruch
     - Spectral Centroid-Sprung > 2 Oktaven → Spektralsprung
     - Masken-Dilation: ±5ms Rand-Padding

  2. **Kontextuelles Prior-Modell** (DSP-Diffusion):
     - Forward-Process: Maskierte Bins mit gaußschem Rauschen auffüllen
     - Reverse-Process (T=50 Denoising-Steps):
         Step t: x_{t-1} = f(x_t, context_left, context_right, t/T)
         Gewichtung: cos²-Interpolation von linkem/rechtem Kontext
         Rauschanteil: σ_t = σ_max * (t/T)^2 (Cosine-Schedule)
     - Harmonisches Prior: AR-Modell aus Vorfeld-Segment (Burg-Schätzung Ordnung 64)
     - Envelopen-Continuity: Attack/Release-Matching am Rand

  3. **Plugin-Pfad** (optional, wenn DiffWave-Gewichte vorhanden):
     - Lädt `plugins/diffwave_plugin.py` dynamisch
     - Konditioniert mit linkem+rechtem Kontext-Mel-Spektrogramm
     - Fallback auf DSP-Diffusion bei fehlenden Gewichten  # §V6 (copilot-instructions.md): logger.warning handled at call site

METRIKEN:
  - n_gaps_detected: Anzahl erkannter Lücken
  - total_gap_ms: Summe rekonstruierter Millisekunden
  - max_gap_ms: Größte Einzellücke
  - plugin_used: Ob DiffWave-Plugin genutzt wurde
  - reconstruction_quality: Geschätzter PESQ-Proxy (Energie-Kontinuität-Score)

Author: Aurik Development Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import os  # §v10.105 module-level (prevents UnboundLocalError)
import time
from typing import Any

import numpy as np

from backend.core.ml_model_readiness import check_ml_model_ready
from backend.core.restoration_policy import get_effective_song_goal_weights

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)

# ─── Algorithmus-Konstanten ─────────────────────────────────────────────
_FFT_SIZE = 2048
_HOP = 512
_WIN = "hann"
_MIN_GAP_MS_DEFAULT = 20.0  # Minimale Lückenlänge für Inpainting
_ENERGY_THRESH_DBFS = -60.0  # Energie-Schwelle für Dropout-Erkennung
_DIFFUSION_STEPS = 50  # Basis-Schrittanzahl (Short Gaps < 50 ms)
_DIFFUSION_STEPS_MED = 100  # Mittlere Lücken 50–100 ms
_DIFFUSION_STEPS_LONG = 150  # Lange Lücken > 100 ms (höchste Qualität)
_PHASE55_WALL_BUDGET_S = 120.0  # bounded runtime; remaining gaps stay passthrough-safe


def _adaptive_steps(gap_ms: float) -> int:
    """Wählt Diffusionsschritt-Anzahl adaptiv nach Lückengröße.

    Kurze Lücken (<50 ms) brauchen weniger Iterationen (Kontext dominant),
    lange Lücken (>100 ms) profitieren von mehr Denoising-Schritten,
    da der AR-Prior weniger verlässlich wird.
    """
    if gap_ms < 50.0:
        return _DIFFUSION_STEPS  # 50
    if gap_ms < 100.0:
        return _DIFFUSION_STEPS_MED  # 100
    return _DIFFUSION_STEPS_LONG  # 150


_AR_ORDER = 64  # AR-Modell-Ordnung für harmonisches Prior
_CONTEXT_FRAMES = 20  # Kontext-Frames links/rechts
_MASK_DILATION_FRAMES = 3  # Dilations-Padding um Maske
_SIGMA_MAX = 0.3  # Maximale Rausch-Standardabweichung
_MAX_DSP_INPAINT_GAP_MS = 800.0  # Longer regions are not true dropouts; avoid O(n*steps) stalls
_MAX_THRASH_INPAINT_GAP_MS = 300.0  # In thrashing mode keep per-gap work strictly bounded
_MAX_DETECTED_GAP_MS = 3000.0  # Safety cap for detector drift on sustained quiet passages

# §0 Primum-non-nocere: Material-Bandbreiten-Cap verhindert HF-Halluzination
# beim Inpainting historischer Träger (wax_cylinder BW ≤ 5 kHz, wire_rec. ≤ 6 kHz).
# Ohne Cap generiert AR/Diffusion synthetische Obertöne, die nie vorhanden waren.
# Literature anchors for the caps:
#   - Wax cylinders: typically ~2.5–4.5 kHz useful bandwidth; 5 kHz used here as
#     conservative upper safety cap (Casey, Sound Directions 2007; IASA TC-04).
#   - Wire recording: typically ~4–6 kHz depending on transport speed and head design;
#     6 kHz used as the hard upper bound (Morton 2006; ARSC preservation notes).
#   - Shellac: archival transfer practice usually preserves content below ~7 kHz.
_MATERIAL_BW_CAP_HZ: dict[str, float] = {
    "wax_cylinder": 5000.0,  # Mechanische Aufzeichnung 1900–1930
    "wire_recording": 6000.0,  # Stahlbandfone 1940–1955
    "shellac": 7000.0,  # Schellackplatten 1898–1960, §0 Vintage Aesthetics (≤ 7 kHz)
    "lacquer_disc": 8000.0,  # Acetat-Lackfolien 1930–1950 (konservativ)
}


def _apply_bw_cap(segment: np.ndarray, sample_rate: int, cap_hz: float) -> np.ndarray:
    """Low-pass filter reconstructed segment to material bandwidth cap.

    Prevents AR/diffusion hallucinating HF content never present in the
    source material (§0 Primum-non-nocere, §2.55 BW-Cap Invariante).
    Uses Butterworth 8th-order to avoid ringing in short segments.
    """
    if cap_hz >= sample_rate / 2 - 100:
        return segment
    nyq = sample_rate / 2.0
    norm_cut = min(cap_hz / nyq, 0.99)
    from scipy import signal as _sps  # pylint: disable=import-outside-toplevel

    sos = _sps.butter(4, norm_cut, btype="low", output="sos")
    # filtfilt for zero-phase; fall back to sosfilt for very short segments
    if len(segment) > 20:
        filtered = _sps.sosfiltfilt(sos, segment)
    else:
        filtered = _sps.sosfilt(sos, segment)
    return np.clip(filtered.astype(segment.dtype), -1.0, 1.0)  # type: ignore[no-any-return]


def _cosine_schedule(t: int, T: int) -> float:
    """Cosine Noise-Schedule: σ_t = σ_max * (t/T)²"""
    return _SIGMA_MAX * (t / max(T, 1)) ** 2


def _burg_ar_predict(context: np.ndarray, order: int, n_samples: int) -> np.ndarray:
    """
    AR-Prädiktor via Levinson-Durbin (Toeplitz-Normalgleichungen, AR-Ordnung 64).
    Extrapoliert n_samples Samples aus dem Kontext.

    Algorithmus:
        1. Autokorrelationsschätzung (positive Lags 0…order)
        2. Aufstellen der Toeplitz-Normalgleichungen R·a = r
        3. Numerische Lösung via np.linalg.solve (= Levinson-Durbin-Annäherung)
        4. Rekursive Vorwärtsvorhersage mit gespeicherten Kontextwerten

    Referenz:
        Levinson (1947), Durbin (1960) — Toeplitz-Rekurrenz für AR-Schätzung
    """
    if len(context) < order + 1:
        return np.zeros(n_samples)  # type: ignore[no-any-return]

    # Autokorrelation schätzen — FFT-based O(N log N)
    from backend.core.core_utils import fft_autocorr  # pylint: disable=import-outside-toplevel

    ac = fft_autocorr(context, max_lag=order)
    if ac[0] < 1e-10:
        return np.zeros(n_samples)  # type: ignore[no-any-return]

    # Toeplitz-System lösen (Levinson-Durbin approx)
    try:
        R = np.array([ac[abs(i)] for i in range(order)])
        Rmat = np.array([[ac[abs(i - j)] for j in range(order)] for i in range(order)])
        if np.linalg.matrix_rank(Rmat) < order:
            return np.zeros(n_samples)  # type: ignore[no-any-return]
        ar_coeff = np.linalg.solve(Rmat, R)
    except np.linalg.LinAlgError as exc:
        logger.debug("§V6 AR-Toeplitz-Solve fehlgeschlagen — Nullen zurückgegeben (LinAlgError): %s", exc)
        return np.zeros(n_samples)  # type: ignore[no-any-return]

    # Vorhersage iterativ berechnen
    buf = list(context[-order:])
    predicted = []
    for _ in range(n_samples):
        val = np.dot(ar_coeff, buf[-order:][::-1])
        val = np.clip(val, -1.0, 1.0)
        predicted.append(val)
        buf.append(val)

    return np.array(predicted)  # type: ignore[no-any-return]


def _detect_gaps(audio: np.ndarray, sample_rate: int, min_gap_ms: float) -> list[tuple[int, int]]:
    """
    Erkennt Dropout-Lücken im Audio-Signal.
    Returns list of (start_sample, end_sample) tuples.
    """
    min_gap_samples = max(1, int(min_gap_ms * sample_rate / 1000))

    # Frame-weise RMS
    frame_size = _HOP
    n_frames = len(audio) // frame_size
    if n_frames < 3:
        return []

    frame_rms = np.array(
        [np.sqrt(np.mean(audio[i * frame_size : (i + 1) * frame_size] ** 2) + 1e-12) for i in range(n_frames)],
        dtype=np.float64,
    )
    frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-10))

    # Kontext-adaptive Dropout-Maske:
    # - absolute floor (historical dropout floor)
    # - AND local context dip (>=10 dB below 250ms neighborhood median)
    # So musical quiet passages won't be flagged as transport dropouts.
    from scipy.ndimage import median_filter as _medfilt  # pylint: disable=import-outside-toplevel

    _ctx = max(5, int(0.25 * sample_rate / frame_size))
    if _ctx % 2 == 0:
        _ctx += 1
    local_ref_db = _medfilt(frame_db, size=_ctx)
    is_dropout = (frame_db < _ENERGY_THRESH_DBFS) & (frame_db < (local_ref_db - 10.0))

    # Zusammenhängende Regionen finden
    gaps = []
    in_gap = False
    gap_start = 0
    for i, dropout in enumerate(is_dropout):
        if dropout and not in_gap:
            gap_start = i * frame_size
            in_gap = True
        elif not dropout and in_gap:
            gap_end = i * frame_size
            gap_len = gap_end - gap_start
            if gap_len >= min_gap_samples:
                # Boundary contrast validation: real dropouts are much lower than
                # immediate pre/post context. Reject low-energy musical sections.
                _s_fr = max(0, gap_start // frame_size)
                _e_fr = min(n_frames, max(_s_fr + 1, gap_end // frame_size))
                _pre_s = max(0, _s_fr - 5)
                _pre_e = _s_fr
                _post_s = _e_fr
                _post_e = min(n_frames, _e_fr + 5)
                _gap_db = float(np.median(frame_db[_s_fr:_e_fr]))
                _pre_db = float(np.median(frame_db[_pre_s:_pre_e])) if _pre_e > _pre_s else _gap_db
                _post_db = float(np.median(frame_db[_post_s:_post_e])) if _post_e > _post_s else _gap_db
                _contrast_ok = (_pre_db > _gap_db + 8.0) or (_post_db > _gap_db + 8.0)
                _within_cap = gap_len <= int(_MAX_DETECTED_GAP_MS * sample_rate / 1000.0)
                if _contrast_ok and _within_cap:
                    gaps.append((gap_start, gap_end))
            in_gap = False

    if in_gap:
        gap_end = len(audio)
        gap_len = gap_end - gap_start
        if gap_len >= min_gap_samples and gap_len <= int(_MAX_DETECTED_GAP_MS * sample_rate / 1000.0):
            _s_fr = max(0, gap_start // frame_size)
            _e_fr = min(n_frames, max(_s_fr + 1, gap_end // frame_size))
            _pre_s = max(0, _s_fr - 5)
            _pre_e = _s_fr
            _gap_db = float(np.median(frame_db[_s_fr:_e_fr]))
            _pre_db = float(np.median(frame_db[_pre_s:_pre_e])) if _pre_e > _pre_s else _gap_db
            # Fadeout guard (trailing gap only): check energy slope before gap_start.
            # A musical fadeout has a sustained declining slope — NOT a sudden transport dropout.
            # Frame hop = _HOP / sample_rate ≈ 10.7 ms; slope unit = dB/frame.
            # Threshold: -0.5 dB/frame ≈ -47 dB/s — typical fadeout rate. True dropouts
            # are instantaneous (slope ≈ -40 dB in 1 frame = -3733 dB/s).
            _fade_win_start = max(0, _s_fr - max(4, int(0.3 * sample_rate / frame_size)))
            _fade_frames_db = frame_db[_fade_win_start:_s_fr]
            _is_fadeout = False
            if len(_fade_frames_db) >= 4:
                try:
                    _slope = float(np.polyfit(np.arange(len(_fade_frames_db)), _fade_frames_db, 1)[0])
                    _is_fadeout = _slope < -0.5  # energy was already declining → fadeout, not dropout
                except Exception as e:
                    logger.warning("Verarbeitungsschritt_55_diffusion_inpainting.py::unbekannter Ersatzpfad: %s", e)
            if _pre_db > _gap_db + 8.0 and not _is_fadeout:
                gaps.append((gap_start, gap_end))

    return gaps


def _inpaint_gap_dsp(
    audio: np.ndarray,
    start: int,
    end: int,
    _sample_rate: int = 48000,
    n_steps: int = _DIFFUSION_STEPS,
) -> np.ndarray:
    """
    DSP-basierende Diffusions-Inpainting für eine einzelne Lücke.
    Kombiniert AR-Prior mit iterativem Denoising.
    """
    gap_len = end - start
    context_samples = _CONTEXT_FRAMES * _HOP

    # Kontext-Puffer links und rechts
    left_ctx = audio[max(0, start - context_samples) : start].copy()
    right_ctx = audio[end : min(len(audio), end + context_samples)].copy()
    right_ctx_rev = right_ctx[::-1]

    # §v10.0.0: Adaptive AR order — order 64 diverges for gaps > 50 ms (2 400 samples).
    # AR(192) covers 3× more spectral modes; safe as long as context length > order.
    _AR_ORDER_ADAPTIVE = min(192, max(16, len(left_ctx) - 1)) if gap_len > 2400 else _AR_ORDER

    # AR-Vorhersage von links und von rechts (gespiegelt)
    ar_left = _burg_ar_predict(left_ctx, _AR_ORDER_ADAPTIVE, gap_len)
    ar_right = _burg_ar_predict(right_ctx_rev, _AR_ORDER_ADAPTIVE, gap_len)[::-1]

    # Cosine-Gewichtung: links → rechts
    t_vec = np.linspace(0, np.pi / 2, gap_len)
    w_left = np.cos(t_vec) ** 2
    w_right = np.sin(t_vec) ** 2

    # Kombinierter AR-Prior
    x = w_left * ar_left + w_right * ar_right

    # Envelopen-Kontinuität erzwingen
    if len(left_ctx) > 0 and len(right_ctx) > 0:
        env_left = np.abs(left_ctx[-1]) if len(left_ctx) > 0 else 0.0
        env_right = np.abs(right_ctx[0]) if len(right_ctx) > 0 else 0.0
        env_target = w_left * env_left + w_right * env_right
        x_env = np.abs(x) + 1e-10
        x = x * (env_target / x_env).clip(0.0, 2.0)

    # §2.40 Determinismus: Seeded RNG, derived from left-context fingerprint + gap position.
    # Ensures bit-exact reproducibility for identical inputs (same audio, same position).
    _ctx_seed = int(abs(float(np.sum(np.abs(left_ctx[: min(len(left_ctx), 64)])))) * 1e5 + start) % (2**31)
    _rng55 = np.random.default_rng(seed=_ctx_seed)

    # Reverse Diffusion (iteratives Denoising, T=n_steps Steps, adaptiv)
    for step in range(n_steps, 0, -1):
        sigma = _cosine_schedule(step, n_steps)
        if sigma > 0:
            noise = _rng55.standard_normal(gap_len) * sigma
            # Denoising: Projektionsschritt zurück zum Prior
            x = x + noise * 0.2
            # Regularisierung: Low-pass smoothing bei hohem Rauschen
            if sigma > 0.1:
                kernel_size = max(3, int(sigma * 20) | 1)
                x = np.convolve(x, np.ones(kernel_size) / kernel_size, mode="same")

    # Normierung auf Kontext-Energie-Level
    if len(left_ctx) > 10:
        ctx_rms = np.sqrt(np.mean(left_ctx[-100:] ** 2)) if np.any(left_ctx[-100:]) else 1e-4
        rec_rms = np.sqrt(np.mean(x**2)) + 1e-10
        x = x * (ctx_rms / rec_rms)

    return np.clip(x, -1.0, 1.0)  # type: ignore[no-any-return]


def _nmf_gap_fallback(
    channel: np.ndarray,
    start: int,
    end: int,
    sr: int,
) -> np.ndarray:
    """NMF-\u03b2 (IS-Divergenz, \u03b2=0) per-Gap-Inpainting — letzter DSP-Fallback (§2.47).

    Wenn _inpaint_gap_dsp scheitert, rekonstruiert dieser Fallback das Lücken-Segment
    durch NMF-Komponentenlernen auf dem umgebenden Kontext und überträgt die spektrale
    Struktur auf die Lücke. Reproduzierbar via deterministischem RNG-Seed.

    Algorithmus:
        1. Kontext-Fenster (±context_samples) via STFT analysieren
        2. NMF-Komponentenzerlegung (Rang 8, 10 multiplikative IS-Divergenz-Schritte)
        3. Durchschnittliche Aktivierung → Lücken-Magnitude
        4. Phasenkontinuität via Velocity-Fortsetzung (δφ aus Kontext)
        5. Phasen-kohärente Rekonstruktion via PGHI (Fallback: ISTFT)

    Reference: Févotte & Idier (2011) — NMF mit β-Divergenz (β=0 ≡ IS-Divergenz).
    """
    gap_len = end - start
    if gap_len <= 0:
        return channel[start:end]

    _n_fft = 1024
    _hop = 256
    ctx_samples = min(max(gap_len * 4, _n_fft * 2), len(channel) // 4)

    ctx_start = max(0, start - ctx_samples)
    ctx_end = min(len(channel), end + ctx_samples)
    context = channel[ctx_start:ctx_end].astype(np.float64)

    from scipy import signal as _sps  # pylint: disable=import-outside-toplevel

    _, _, Z_ctx = _sps.stft(context, fs=sr, window="hann", nperseg=_n_fft, noverlap=_n_fft - _hop)
    mag_ctx = np.abs(Z_ctx)  # (n_bins, n_frames)
    n_bins, n_frames = mag_ctx.shape
    if n_frames < 4 or n_bins < 4:
        # Fallback: zeros
        return np.zeros(gap_len, dtype=np.float32)  # type: ignore[no-any-return]

    # NMF-β (IS-Divergenz, β=0): multiplicative update rules (Févotte 2011)
    _rank = min(8, n_frames // 2)
    _eps = 1e-10
    _rng = np.random.default_rng(seed=int(np.abs(np.sum(context[: min(64, len(context))])) * 1e4 + start) % (2**31))
    _W = _rng.random((n_bins, _rank)) + 0.1  # basis spectra (n_bins, rank)
    _H = _rng.random((_rank, n_frames)) + 0.1  # activations (rank, n_frames)

    for _ in range(10):
        _WH = _W @ _H + _eps
        # IS-divergence updates (β=0)
        _H *= (_W.T @ (mag_ctx / _WH**2)) / (_W.T @ (1.0 / _WH) + _eps)
        _WH = _W @ _H + _eps
        _W *= ((mag_ctx / _WH**2) @ _H.T) / ((1.0 / _WH) @ _H.T + _eps)

    # Estimate gap magnitude from average activations
    gap_start_ctx = start - ctx_start
    gap_duration = gap_len / sr
    n_gap_frames = max(1, int(np.round(gap_duration * sr / _hop)))
    avg_H = np.mean(_H, axis=1, keepdims=True)  # (rank, 1)
    gap_mag = (_W @ (avg_H * np.ones((1, n_gap_frames)))).clip(0.0)  # (n_bins, n_gap_frames)

    # Phase velocity continuation from left context edge
    left_frame = max(0, gap_start_ctx // _hop - 1)
    if left_frame >= 1:
        phi_t1 = np.angle(Z_ctx[:, left_frame])
        phi_t0 = np.angle(Z_ctx[:, left_frame - 1])
        delta_phi = phi_t1 - phi_t0
    else:
        phi_t1 = np.zeros(n_bins)
        delta_phi = np.zeros(n_bins)

    gap_phase = np.zeros((n_bins, n_gap_frames))
    for i in range(n_gap_frames):
        gap_phase[:, i] = phi_t1 + delta_phi * (i + 1)

    Z_gap = gap_mag * np.exp(1j * gap_phase)

    # Phase-coherent reconstruction via PGHI (§2.47 VERBOTEN: direktes ISTFT)
    try:
        from dsp.pghi import pghi_reconstruct as _pghi_rec  # pylint: disable=import-outside-toplevel

        _initial_phase_gap = gap_phase.astype(np.float32)
        _gap_audio = _pghi_rec(
            gap_mag.astype(np.float32),
            sr=sr,
            win_size=_n_fft,
            hop=_hop,
            initial_phase=_initial_phase_gap,
        )
    except (ImportError, Exception) as _pghi_err:
        logger.debug("Verarbeitungsschritt_55 NMF-gap PGHI-Ersatzpfad: %s", _pghi_err)
        _, _gap_audio = _sps.istft(Z_gap, fs=sr, window="hann", nperseg=_n_fft, noverlap=_n_fft - _hop)

    _gap_audio = np.asarray(_gap_audio, dtype=np.float64)

    # Trim/pad to exact gap length
    if len(_gap_audio) >= gap_len:
        _gap_audio = _gap_audio[:gap_len]
    else:
        _gap_audio = np.pad(_gap_audio, (0, gap_len - len(_gap_audio)))

    # Energy normalisation to match context boundary
    ctx_border_rms = float(np.sqrt(np.mean(channel[max(0, start - 64) : start] ** 2)) + 1e-10)
    rec_rms = float(np.sqrt(np.mean(_gap_audio**2)) + 1e-10)
    _gap_audio = _gap_audio * (ctx_border_rms / rec_rms)

    _gap_audio = np.nan_to_num(_gap_audio, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(_gap_audio, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]


def _try_cqtdiff_plus_plugin(audio: np.ndarray, start: int, end: int, sample_rate: int) -> np.ndarray | None:
    """
    CQTdiff Inpainting für Lücken ≥ 50 ms (Moliner & Välimäki 2022, §4.5 Aurik-Spec).

    CQTdiff konditioniert Score-basierte Diffusion im CQT-Domäne:
    - Logarithmische Frequenzauflösung ≡ musikalische Tonleiter
    - Harmonisch kohärente Füll-Lösung (keine Phasen-Inkoharenz)
    - Mindest-Lückengröße: 50 ms (CQTdiffPlusPlugin.MIN_GAP_MS)

    Gibt None zurück wenn:
    - Lücke < 50 ms (NMF-β in Phase 24 übernimmt)
    - Plugin oder ONNX-Modell nicht verfügbar
    """
    gap_ms = (end - start) / sample_rate * 1000.0
    if gap_ms < 50.0:
        return None  # Kurze Lücken → DSP-Diffusion (NMF-β-Äquivalent)
    try:
        import os as _os  # pylint: disable=import-outside-toplevel
        import sys  # pylint: disable=import-outside-toplevel

        _plugins_dir = _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "plugins")
        if _plugins_dir not in sys.path:
            sys.path.insert(0, _os.path.abspath(_plugins_dir))

        from backend.core.plugin_lifecycle_manager import (  # pylint: disable=import-outside-toplevel
            get_plugin_lifecycle_manager as _get_plm55a,
        )
        from plugins.cqtdiff_plus_plugin import get_cqtdiff_plus  # pylint: disable=import-outside-toplevel

        plugin = get_cqtdiff_plus()
        _plm55a = _get_plm55a()
        _plm55a.set_active("CQTdiffPlus", True)
        try:
            result = plugin.inpaint(audio=audio, sr=sample_rate, gap_start_sample=start, gap_end_sample=end)
        finally:
            _plm55a.set_active("CQTdiffPlus", False)
        # InpaintingResult.audio = volles Audio-Signal mit gefüllter Lücke
        repaired_segment = result.audio[start:end]
        if repaired_segment is not None and np.isfinite(repaired_segment).all():
            return np.clip(repaired_segment.astype(np.float32), -1.0, 1.0)  # type: ignore[no-any-return]
        return None
    except Exception as _e:
        logger.debug("CQTdiff-Plugin nicht verfügbar: %s", _e)
        return None


def _try_flow_matching_plugin(
    audio: np.ndarray,
    start: int,
    end: int,
    sample_rate: int,
    goal_weights: dict[str, float] | None = None,
    restorability_score: float = 65.0,
) -> np.ndarray | None:
    """
    Versucht FlowMatchingPlugin (Primär-Inpainting, §4.5) für Lücken aller Größen.

    FlowMatchingPlugin (Lipman et al. 2023) verwendet 4–16 Flow-Schritte statt
    1000 DDPM-Schritte — deutlich schneller und qualitativ gleichwertig oder besser.
    Aktiviert für Lücken aller Größen (20 ms – 30 s).
    """
    try:
        import os as _os  # pylint: disable=import-outside-toplevel
        import sys  # pylint: disable=import-outside-toplevel

        _plugins_dir = _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "plugins")
        if _plugins_dir not in sys.path:
            sys.path.insert(0, _os.path.abspath(_plugins_dir))

        from flow_matching_plugin import inpaint_flow  # pylint: disable=import-outside-toplevel

        from backend.core.plugin_lifecycle_manager import (  # pylint: disable=import-outside-toplevel
            get_plugin_lifecycle_manager as _get_plm55b,
        )

        _plm55b = _get_plm55b()
        _plm55b.set_active("FlowMatching", True)
        try:
            result = inpaint_flow(
                audio,
                start,
                end,
                sample_rate,
                goal_weights=goal_weights,
                restorability_score=restorability_score,
            )
        finally:
            _plm55b.set_active("FlowMatching", False)
        if result is not None and result.success:
            repaired_segment = result.audio[start:end]
            if repaired_segment is not None and np.isfinite(repaired_segment).all():
                return np.clip(repaired_segment.astype(np.float32), -1.0, 1.0)  # type: ignore[no-any-return]
        return None
    except Exception as _e:
        logger.debug("FlowMatchingPlugin nicht verfügbar: %s", _e)
        return None


def _try_diffwave_plugin(audio: np.ndarray, start: int, end: int, sample_rate: int) -> np.ndarray | None:
    """
    Versucht DiffWave-Plugin für Inpainting zu laden. Gibt None zurück wenn nicht verfügbar.
    Fallback-Priorität 2 nach FlowMatchingPlugin.
    """
    try:
        import importlib  # pylint: disable=import-outside-toplevel
        import os  # pylint: disable=import-outside-toplevel
        import sys  # pylint: disable=import-outside-toplevel

        plugins_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "plugins")
        if plugins_dir not in sys.path:
            sys.path.insert(0, os.path.abspath(plugins_dir))

        dw = importlib.import_module("diffwave_plugin")
        if not hasattr(dw, "inpaint"):
            return None

        from backend.core.plugin_lifecycle_manager import (  # pylint: disable=import-outside-toplevel
            get_plugin_lifecycle_manager as _get_plm55c,
        )

        _plm55c = _get_plm55c()
        _plm55c.set_active("DiffWave", True)
        try:
            return dw.inpaint(audio, start, end, sample_rate)  # type: ignore[no-any-return]
        finally:
            _plm55c.set_active("DiffWave", False)
    except Exception as e:
        logger.debug("DiffWave-Plugin nicht verfügbar: %s", e)
        return None


def _try_consistency_model_inpainting(channel: np.ndarray, start: int, end: int, sample_rate: int) -> np.ndarray | None:
    """Priority 0.8: Consistency Model inpainting (Song et al. 2023, ICML).

    Single-step diffusion via consistency distillation — 10–100× faster than DDPM,
    comparable quality for audio gaps of any size. Runs even during ml_thrashing since
    the model requires less GPU/CPU than multi-step diffusion.

    Returns filled gap segment or None if plugin unavailable.
    """
    assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
    gap_len = end - start
    if gap_len <= 0:
        return None
    try:
        import os as _os  # pylint: disable=import-outside-toplevel
        import sys  # pylint: disable=import-outside-toplevel

        _plugins_dir = _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "plugins")
        if _os.path.abspath(_plugins_dir) not in sys.path:
            sys.path.insert(0, _os.path.abspath(_plugins_dir))

        try:
            from plugins.consistency_inpaint_plugin import (  # type: ignore[import]  # pylint: disable=import-outside-toplevel
                get_consistency_inpaint_plugin,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            logger.debug("§V6 consistency_inpaint_plugin nicht verfügbar — None zurückgegeben (phase_55): %s", exc)
            return None

        from backend.core.plugin_lifecycle_manager import (  # pylint: disable=import-outside-toplevel
            get_plugin_lifecycle_manager as _get_plm55d,
        )

        cm = get_consistency_inpaint_plugin()
        if cm is None:
            return None

        # Context window: 300 ms before and after gap (enough for musical phrase context)
        ctx_samples = min(int(0.30 * sample_rate), start)
        ctx_l = channel[max(0, start - ctx_samples) : start]
        ctx_r = channel[end : min(len(channel), end + ctx_samples)]

        _plm55d = _get_plm55d()
        _plm55d.set_active("ConsistencyInpaint", True)
        try:
            result = cm.inpaint(ctx_l, ctx_r, gap_len, sample_rate)
        finally:
            _plm55d.set_active("ConsistencyInpaint", False)
        if result is None or len(result) == 0:
            return None

        result = np.nan_to_num(np.asarray(result, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        _out_cim: np.ndarray = np.clip(result[:gap_len], -1.0, 1.0).astype(np.float32)
        return _out_cim
    except Exception as _exc:
        logger.debug("_try_consistency_model_inpainting fehlgeschlagen (unkritisch): %s", _exc)
        return None


def _try_dac_token_inpainting(channel: np.ndarray, start: int, end: int, sample_rate: int) -> np.ndarray | None:
    """Priority 1.5: DAC discrete audio codec token inpainting (Kumar et al. 2024).

    Encodes context into discrete audio tokens, inpaints the missing token sequence
    via causal AR / masked prediction, decodes back to waveform. Produces harmonically
    coherent fills for long gaps (≥ 50 ms) where CQTdiff+ is unavailable.

    Returns filled gap segment or None if plugin unavailable.
    """
    assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
    gap_len = end - start
    if gap_len <= 0:
        return None
    try:
        import os as _os  # pylint: disable=import-outside-toplevel
        import sys  # pylint: disable=import-outside-toplevel

        _plugins_dir = _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "plugins")
        if _os.path.abspath(_plugins_dir) not in sys.path:
            sys.path.insert(0, _os.path.abspath(_plugins_dir))

        from backend.core.plugin_lifecycle_manager import (  # pylint: disable=import-outside-toplevel
            get_plugin_lifecycle_manager as _get_plm55e,
        )
        from plugins.dac_plugin import get_dac_plugin  # pylint: disable=import-outside-toplevel

        dac = get_dac_plugin()
        if dac is None:
            return None

        # Context window: 500 ms (larger for token-based model)
        ctx_samples = min(int(0.50 * sample_rate), start)
        ctx_l = channel[max(0, start - ctx_samples) : start]
        ctx_r = channel[end : min(len(channel), end + ctx_samples)]

        _plm55e = _get_plm55e()
        _plm55e.set_active("DACInpaint", True)
        try:
            result = dac.inpaint(ctx_l, ctx_r, gap_len, sample_rate)  # type: ignore[attr-defined]
        finally:
            _plm55e.set_active("DACInpaint", False)
        if result is None or len(result) == 0:
            return None

        result = np.nan_to_num(np.asarray(result, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        _out_dac: np.ndarray = np.clip(result[:gap_len], -1.0, 1.0).astype(np.float32)
        return _out_dac
    except Exception as _exc:
        logger.debug("_try_dac_token_inpainting fehlgeschlagen (unkritisch): %s", _exc)
        return None


def _is_ml_thrashing() -> bool:
    """Gibt True when ML paths should be avoided due to active swap thrashing zurück.

    Result is cached for 30 s to avoid log-spam from per-channel calls (BUG H).
    """
    import threading as _th  # pylint: disable=import-outside-toplevel

    _cache = getattr(_is_ml_thrashing, "_cache", None)
    _lock = getattr(_is_ml_thrashing, "_lock", None)
    if _lock is None:
        _is_ml_thrashing._lock = _th.Lock()  # type: ignore[attr-defined]  # pylint: disable=protected-access
        _is_ml_thrashing._cache = (False, 0.0)  # type: ignore[attr-defined]  # pylint: disable=protected-access
        _lock = _is_ml_thrashing._lock  # type: ignore[attr-defined]  # pylint: disable=protected-access
        _cache = _is_ml_thrashing._cache  # type: ignore[attr-defined]  # pylint: disable=protected-access

    now = time.monotonic()
    if _cache is None:
        _cache = (False, 0.0)
    _result, _ts = _cache
    if now - _ts < 30.0:
        return _result
    with _lock:
        _result, _ts = _is_ml_thrashing._cache  # type: ignore[attr-defined]  # pylint: disable=protected-access
        if now - _ts < 30.0:
            return _result  # type: ignore[no-any-return]
        try:
            from backend.core.ml_memory_budget import is_system_thrashing  # pylint: disable=import-outside-toplevel

            _result = bool(is_system_thrashing())
        except Exception:
            _result = False
        _is_ml_thrashing._cache = (_result, now)  # type: ignore[attr-defined]  # pylint: disable=protected-access
    _final_thr: bool = bool(_result)
    return _final_thr


def _conservative_boundary_fill(channel: np.ndarray, start: int, end: int) -> np.ndarray:
    """Fill gap with a boundary-constrained cosine interpolation.

    For trailing gaps (end == len(channel)): fade from left boundary to 0.0 (silence).
    Without this, the fill level stays at the pre-silence sample value, creating
    a constant tone/hiss in what should be the fadeout silence ('explodiert').
    """
    gap_len = max(0, end - start)
    if gap_len <= 0:
        return np.zeros(0, dtype=np.float32)  # type: ignore[no-any-return]

    left = float(channel[start - 1]) if start > 0 else 0.0
    # Trailing gap (end of audio): right boundary is silence (0.0), not left.
    # This prevents the fadeout silence from being filled with non-zero content.
    right = float(channel[end]) if end < len(channel) else 0.0
    t = np.linspace(0.0, 1.0, gap_len, dtype=np.float32)
    fade = 0.5 - 0.5 * np.cos(np.pi * t)
    seg = (1.0 - fade) * left + fade * right
    _seg_clean = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(_seg_clean, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]


def _gap_candidate_is_damaging(candidate: np.ndarray, channel: np.ndarray, start: int, end: int) -> bool:
    """Erkennt boundary/energy anomalies that indicate risky inpainting output."""
    gap_len = max(0, end - start)
    if gap_len <= 2 or len(candidate) == 0:
        return False

    seg = np.asarray(candidate, dtype=np.float32)[:gap_len]
    if not np.isfinite(seg).all():
        return True

    left_ctx = channel[max(0, start - 256) : start]
    right_ctx = channel[end : min(len(channel), end + 256)]
    if len(left_ctx) == 0 and len(right_ctx) == 0:
        return False
    ctx = np.concatenate((left_ctx[-128:], right_ctx[:128]))

    left_edge = float(channel[start - 1]) if start > 0 else float(seg[0])
    right_edge = float(channel[end]) if end < len(channel) else float(seg[-1])

    jump_l = abs(float(seg[0]) - left_edge)
    jump_r = abs(float(seg[-1]) - right_edge)
    ctx_delta = np.diff(ctx.astype(np.float32)) if len(ctx) > 4 else np.array([0.0], dtype=np.float32)
    median_ctx_step = float(np.median(np.abs(ctx_delta))) + 1e-6
    jump_limit = max(0.18, 10.0 * median_ctx_step)
    if jump_l > jump_limit or jump_r > jump_limit:
        return True

    ctx_p99 = float(np.percentile(np.abs(ctx), 99.0)) + 1e-6 if len(ctx) > 8 else 1e-3
    seg_p99 = float(np.percentile(np.abs(seg), 99.0))
    return bool(seg_p99 > ctx_p99 * 2.5)


def _apply_shared_stereo_ratio(
    audio_stereo: np.ndarray,
    mono_reference: np.ndarray,
    mono_repaired: np.ndarray,
) -> np.ndarray:
    """Wendet eine gemeinsame signed Mono-Ratio konsistent auf beide Stereokanaele an."""
    _den = np.where(np.abs(mono_reference) > 1e-10, mono_reference, np.sign(mono_reference + 1e-30) * 1e-10)
    _ratio = mono_repaired / _den
    _ratio = np.clip(_ratio, -10.0, 10.0)
    _out = np.column_stack([audio_stereo[:, 0] * _ratio, audio_stereo[:, 1] * _ratio])
    return np.clip(np.nan_to_num(_out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]


def _process_channel(
    channel: np.ndarray,
    sample_rate: int,
    min_gap_ms: float,
    repaired_gap_samples: list[tuple[int, int]] | None = None,
    bw_cap_hz: float | None = None,
    precomputed_gaps: list[tuple[int, int]] | None = None,
    wall_budget_s: float = _PHASE55_WALL_BUDGET_S,
    goal_weights: dict[str, float] | None = None,
    restorability_score: float = 65.0,
    protected_zones: list[tuple[float, float, float]] | None = None,
    base_strength: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Inpainting für einen Mono-Kanal. Returns (repaired, stats)."""
    result = channel.copy()
    gaps = list(precomputed_gaps) if precomputed_gaps is not None else _detect_gaps(channel, sample_rate, min_gap_ms)
    _t_channel_start = time.perf_counter()

    # §11.7a: Bereits von RekonstruktionsDenker reparierte Gaps filtern
    if repaired_gap_samples:
        filtered = []
        for gs, ge in gaps:
            overlap = any(rs < ge and re > gs for rs, re in repaired_gap_samples)
            if not overlap:
                filtered.append((gs, ge))
        n_skipped = len(gaps) - len(filtered)
        gaps = filtered
    else:
        n_skipped = 0

    stats = {
        "n_gaps": len(gaps),
        "total_gap_ms": 0.0,
        "max_gap_ms": 0.0,
        "plugin_used": False,
        "pre_repaired_skipped": n_skipped,
        "damage_guard_activations": 0,
        "ml_thrashing_guard": False,
        "wall_budget_hits": 0,
    }

    ml_thrashing = _is_ml_thrashing()
    if ml_thrashing:
        stats["ml_thrashing_guard"] = True
        # Rate-limit: log at most once per 60 s to prevent log-spam (BUG H extended).
        # _is_ml_thrashing() caches the RESULT for 30 s, but NOT whether the warning
        # was emitted — without this guard, _process_channel calls in a loop (e.g.
        # axis-orientation bug with (2,N) audio) produce 94 K+ identical warnings.
        _now_warn = time.monotonic()
        _last_warn_ts = getattr(_is_ml_thrashing, "_last_warn_ts", 0.0)
        if _now_warn - _last_warn_ts >= 60.0:
            logger.warning(
                "Verarbeitungsschritt_55: ML-Thrashing erkannt — konservativer DSP-Pfad zum Schutz von Musik/Sprache aktiv"
            )
            _is_ml_thrashing._last_warn_ts = _now_warn  # type: ignore[attr-defined]  # pylint: disable=protected-access

    for start, end in gaps:
        if wall_budget_s > 0.0 and (time.perf_counter() - _t_channel_start) > wall_budget_s:
            stats["wall_budget_hits"] += 1
            logger.warning(
                "Verarbeitungsschritt_55: Wall-Grenze %.1fs erreicht — verbleibende Gaps bleiben im sicheren Passthrough",
                wall_budget_s,
            )
            break

        gap_ms = (end - start) / sample_rate * 1000
        local_strength = DiffusionInpaintingPhase._compute_inpainting_local_strength(
            channel,
            start,
            end,
            sample_rate,
            base_strength,
            protected_zones,
        )
        local_ratio = (
            float(np.clip(local_strength / max(base_strength, 1e-6), 0.0, 1.0)) if base_strength > 1e-6 else 0.0
        )

        # §V38 VFA-Schutzzonen: ML-Inpainting in Vibrato/Frisson/Flüster/Passaggio-Zonen
        # → konservativer Boundary-Fill (ML-Synthese würde falschen Pitch-Verlauf erzeugen → VQI-Degradation)
        _p55_vfa_boundary = False
        if protected_zones:
            _p55_gs = start / sample_rate
            _p55_ge = end / sample_rate
            for _pz55_s, _pz55_e, _pz55_cap in protected_zones:
                if _p55_gs < _pz55_e and _p55_ge > _pz55_s:
                    _p55_vfa_boundary = True
                    logger.debug(
                        "§V38 Verarbeitungsschritt_55: Gap [%.3f\u2013%.3f s] in VFA-Schutzzone"
                        " [%.3f\u2013%.3f s, cap=%.2f] \u2192 Boundary-Fill",
                        _p55_gs,
                        _p55_ge,
                        _pz55_s,
                        _pz55_e,
                        _pz55_cap,
                    )
                    break
        if _p55_vfa_boundary:
            candidate = _conservative_boundary_fill(channel, start, end)
            result[start:end] = np.clip(
                np.nan_to_num(candidate[: end - start], nan=0.0, posinf=0.0, neginf=0.0),
                -1.0,
                1.0,
            )
            if bw_cap_hz is not None:
                gap_segment = result[start:end]
                result[start:end] = _apply_bw_cap(gap_segment, sample_rate, bw_cap_hz)
            stats["total_gap_ms"] += gap_ms
            stats["max_gap_ms"] = max(stats["max_gap_ms"], gap_ms)
            continue

        # Long "gaps" are usually musical low-energy passages or detection drift,
        # not true transport dropouts. Full AR/diffusion on these regions can stall
        # phase_55 for many minutes with no fidelity gain. Use safe conservative fill.
        if gap_ms > _MAX_DSP_INPAINT_GAP_MS or (ml_thrashing and gap_ms > _MAX_THRASH_INPAINT_GAP_MS):
            stats["damage_guard_activations"] += 1
            logger.warning(
                "Verarbeitungsschritt_55: Gap %.1f ms über Inpainting-Limit (thrashing=%s) — konservativer Boundary-Fill",
                gap_ms,
                ml_thrashing,
            )
            candidate = _conservative_boundary_fill(channel, start, end)
            result[start:end] = np.clip(
                np.nan_to_num(candidate[: end - start], nan=0.0, posinf=0.0, neginf=0.0),
                -1.0,
                1.0,
            )
            if bw_cap_hz is not None:
                gap_segment = result[start:end]
                result[start:end] = _apply_bw_cap(gap_segment, sample_rate, bw_cap_hz)
            stats["total_gap_ms"] += gap_ms
            stats["max_gap_ms"] = max(stats["max_gap_ms"], gap_ms)
            continue

        # Adaptive Schrittzahl je nach Lückengröße
        n_steps = _adaptive_steps(gap_ms)
        if ml_thrashing:
            n_steps = min(n_steps, 32)

        plugin_result = None
        if ml_thrashing:
            # Hard-thrashing mode: never attempt ML plugin loads per-gap.
            # Repeated allocation attempts under swap=100% create minute-long loops
            # without quality gain; deterministic DSP/NMF fallback is safer/faster.
            try:
                candidate = _inpaint_gap_dsp(channel, start, end, n_steps)
            except Exception as _dsp_exc:
                logger.debug(
                    "DSP-AR-Diffusion fehlgeschlagen — NMF-\u03b2-Ersatzpfad (gap %d:%d): %s", start, end, _dsp_exc
                )
                candidate = _nmf_gap_fallback(channel, start, end, sample_rate)
        else:
            # Priorität 0 [TIER-0]: FlowMatchingPlugin (Lipman et al. 2023, §4.4 SOTA-Matrix primär)
            # Flow Matching: 4–16 Schritte statt 50–200 DDPM-Schritte → 10–50× schneller,
            # gleichwertige oder bessere Qualität. Aktiviert für Lücken aller Größen (20 ms – 30 s).
            plugin_result = _try_flow_matching_plugin(
                channel,
                start,
                end,
                sample_rate,
                goal_weights=goal_weights,
                restorability_score=restorability_score,
            )
            if plugin_result is not None:
                candidate = plugin_result[: end - start]
                stats["plugin_used"] = True
                stats["flow_matching_tier0_used"] = stats.get("flow_matching_tier0_used", 0) + 1
            else:
                # Priorität 0.8: Consistency Model (Song et al. 2023) — 1-Schritt-Diffusions-Inpainting.
                plugin_result = _try_consistency_model_inpainting(channel, start, end, sample_rate)
                if plugin_result is not None:
                    logger.debug("Verarbeitungsschritt_55: Consistency Model Inpainting OK (gap=%.1f ms)", gap_ms)
                    candidate = plugin_result[: end - start]
                    stats["plugin_used"] = True
                    stats["consistency_model_used"] = stats.get("consistency_model_used", 0) + 1
                else:
                    # Priorität 1: CQTdiff+ (≥ 50 ms Lücken, harmonisch kohärent, §4.5 Spec)
                    plugin_result = _try_cqtdiff_plus_plugin(channel, start, end, sample_rate)
                    if plugin_result is not None:
                        candidate = plugin_result[: end - start]
                        stats["plugin_used"] = True
                    else:
                        # Priorität 1.5: DAC Token Inpainting (Kumar et al. 2024) — diskrete Codec-Token.
                        # Musikalisch kohärente Fills für Lücken ≥ 50 ms, wenn CQTdiff+ nicht verfügbar.
                        if gap_ms >= 50.0:
                            plugin_result = _try_dac_token_inpainting(channel, start, end, sample_rate)
                            if plugin_result is not None:
                                logger.debug("Verarbeitungsschritt_55: DAC Token Inpainting OK (gap=%.1f ms)", gap_ms)
                                candidate = plugin_result[: end - start]
                                stats["plugin_used"] = True
                                stats["dac_token_used"] = stats.get("dac_token_used", 0) + 1

                        if plugin_result is None:
                            # Priorität 2+: DSP AR-Diffusion → NMF-β IS-Divergenz (§2.47 Fallback-Pflicht)
                            try:
                                candidate = _inpaint_gap_dsp(channel, start, end, n_steps)
                            except Exception as _dsp_exc:
                                logger.debug(
                                    "DSP-AR-Diffusion fehlgeschlagen — NMF-\u03b2-Ersatzpfad (gap %d:%d): %s",
                                    start,
                                    end,
                                    _dsp_exc,
                                )
                                candidate = _nmf_gap_fallback(channel, start, end, sample_rate)

        if _gap_candidate_is_damaging(candidate, channel, start, end):
            stats["damage_guard_activations"] += 1
            logger.warning(
                "Verarbeitungsschritt_55: Damage-Guard für Gap [%d:%d] (%.1f ms) aktiviert — ersetze riskante Rekonstruktion",
                start,
                end,
                gap_ms,
            )
            candidate = _conservative_boundary_fill(channel, start, end)

        if local_ratio < 0.999:
            source_segment = channel[start:end]
            candidate = source_segment + local_ratio * (candidate[: end - start] - source_segment)

        result[start:end] = np.clip(
            np.nan_to_num(candidate[: end - start], nan=0.0, posinf=0.0, neginf=0.0),
            -1.0,
            1.0,
        )

        # §0 BW-Cap: historische Träger dürfen keine HF-Details halluzinieren
        if bw_cap_hz is not None:
            gap_segment = result[start:end]
            result[start:end] = _apply_bw_cap(gap_segment, sample_rate, bw_cap_hz)

        stats["total_gap_ms"] += gap_ms
        stats["max_gap_ms"] = max(stats["max_gap_ms"], gap_ms)

    return result, stats


# ─── Energie-Kontinuitäts-Score (PESQ-Proxy) ────────────────────────────
def _reconstruction_quality_score(_original: np.ndarray, repaired: np.ndarray, gaps: list[tuple[int, int]]) -> float:
    """Schätzt die Rekonstruktionsqualität (0–1) durch Energie-Kontinuität."""
    if not gaps:
        return 1.0

    scores = []
    for start, end in gaps:
        border = max(1, (end - start) // 4)
        left = repaired[max(0, start - border) : start]
        center = repaired[start:end]
        right = repaired[end : end + border]

        if len(left) < 2 or len(center) < 2 or len(right) < 2:
            scores.append(0.8)
            continue

        rms_l = np.sqrt(np.mean(left**2))
        rms_c = np.sqrt(np.mean(center**2))
        rms_r = np.sqrt(np.mean(right**2))

        # Ideale Energie sollte kontinuierlich sein
        expected = (rms_l + rms_r) / 2
        deviation = abs(rms_c - expected) / (expected + 1e-10)
        score = max(0.0, 1.0 - deviation)
        scores.append(score)

    return float(np.mean(scores))


# ─── Phase-Klasse ───────────────────────────────────────────────────────
class DiffusionInpaintingPhase(PhaseInterface):
    """
    Phase 55: Diffusions-basiertes Audio-Inpainting für Lücken und Dropouts.

    Ersetzt einfache Null-Interpolation durch iterative Diffusionsrekonstruktion
    mit harmonischem AR-Prior. Optional: DiffWave-Plugin-Pfad für ML-gestützte
    Rekonstruktion wenn Modellgewichte vorhanden.
    """

    @staticmethod
    def _derive_safe_inpainting_strength(
        effective_strength: float,
        material_key: str,
        vocals_confidence: float,
    ) -> float:
        """Reduce wet blend for content that is prone to synthetic overfill artifacts."""
        strength = float(effective_strength)
        if vocals_confidence >= 0.40:
            strength *= 0.78
        _is_analog_sensitive = any(
            token in material_key for token in ("vinyl", "shellac", "wax_cylinder", "wire_recording", "lacquer_disc")
        )
        if _is_analog_sensitive:
            strength *= 0.85
        # Prevent tonal-center drift spikes from aggressive diffuse fill on vocal analog material.
        if _is_analog_sensitive and vocals_confidence >= 0.40:
            strength = min(strength, 0.58)
        return float(np.clip(strength, 0.0, 1.0))

    @staticmethod
    def _compute_inpainting_local_strength(
        mono_ref: np.ndarray,
        start: int,
        end: int,
        sr: int,
        base_strength: float,
        protected_zones: list[tuple[float, float, float]] | None,
    ) -> float:
        """Per-Gap-Strength-Orakel für Diffusions-Inpainting (§V38)."""
        if base_strength < 1e-6:
            return 0.0

        context_samples = int(0.250 * sr)
        left_context = mono_ref[max(0, start - context_samples) : start]
        right_context = mono_ref[end : min(len(mono_ref), end + context_samples)]
        gap_segment = mono_ref[start:end]

        seg_rms = float(np.sqrt(np.mean(gap_segment * gap_segment) + 1e-12)) if len(gap_segment) > 0 else 0.0
        ref_parts = [part for part in (left_context, right_context) if len(part) > 0]
        if ref_parts:
            reference = np.concatenate(ref_parts)
            ref_rms = float(np.sqrt(np.mean(reference * reference) + 1e-12))
        else:
            ref_rms = seg_rms + 1e-9

        dropout_severity = float(np.clip(1.0 - seg_rms / max(ref_rms, 1e-9), 0.0, 1.0))
        duration_ms = float((end - start) / max(sr, 1) * 1000.0)
        severity_factor = float(np.clip(0.45 + 0.55 * dropout_severity, 0.45, 1.0))
        duration_factor = float(np.clip(0.60 + 0.40 * min(duration_ms / 120.0, 1.0), 0.60, 1.0))
        local_strength = base_strength * severity_factor * duration_factor

        if protected_zones:
            start_s = start / sr
            end_s = end / sr
            for zone_start, zone_end, zone_cap in protected_zones:
                if start_s < zone_end and end_s > zone_start:
                    local_strength = min(local_strength, float(zone_cap))

        return float(np.clip(local_strength, 0.0, base_strength))

    @staticmethod
    def _compute_inpainting_profile(
        material_key: str,
        quality_mode: str,
        restorability_score: float,
    ) -> dict[str, float]:
        """§2.54 Runtime profile for diffusion inpainting."""
        _material = str(material_key or "unknown").strip().lower()
        _aliases = {"restoration": "balanced", "studio_2026": "maximum"}
        _mode = _aliases.get(
            str(quality_mode or "balanced").strip().lower(), str(quality_mode or "balanced").strip().lower()
        )

        if any(token in _material for token in ("wax_cylinder", "wire_recording", "shellac", "lacquer_disc")):
            min_gap_ms = 28.0
            wall_budget_seconds = 120.0
        elif any(token in _material for token in ("vinyl", "tape", "reel_tape", "cassette")):
            min_gap_ms = 22.0
            wall_budget_seconds = 110.0
        elif any(token in _material for token in ("cd_digital", "dat", "flac", "streaming")):
            min_gap_ms = 14.0
            wall_budget_seconds = 90.0
        else:
            min_gap_ms = 18.0
            wall_budget_seconds = 100.0

        _rest = float(np.clip(float(restorability_score or 50.0), 0.0, 100.0))
        _rest_norm = _rest / 100.0
        min_gap_ms += (_rest_norm - 0.5) * 12.0
        wall_budget_seconds += (0.5 - _rest_norm) * 70.0

        _mode_offsets = {
            "fast": (8.0, -35.0),
            "balanced": (0.0, 0.0),
            "quality": (-4.0, 25.0),
            "maximum": (-7.0, 55.0),
        }
        _gap_off, _budget_off = _mode_offsets.get(_mode, (0.0, 0.0))
        min_gap_ms += _gap_off
        wall_budget_seconds += _budget_off

        return {
            "min_gap_ms": float(np.clip(min_gap_ms, 8.0, 80.0)),
            "wall_budget_seconds": float(np.clip(wall_budget_seconds, 40.0, 240.0)),
        }

    def get_metadata(self) -> PhaseMetadata:
        """Implementiert PhaseInterface.get_metadata()."""
        return PhaseMetadata(
            phase_id="phase_55_diffusion_inpainting",
            name="Diffusion Inpainting",
            category=PhaseCategory.RESTORATION,
            priority=9,  # CRITICAL
            dependencies=["phase_24_dropout_repair", "phase_50_spectral_repair"],
            estimated_time_factor=0.08,
            version="1.0.0",
            memory_requirement_mb=128,
            is_cpu_intensive=True,
            quality_impact=0.85,
            description="Masked Diffusion Inpainting für Audio-Lücken und Dropouts >20ms",
        )

    def _nmf_spectral_inpainting_fallback(self, audio: np.ndarray, sr: int, strength: float = 0.5) -> np.ndarray:
        """NMF-\u03b2 (IS-Divergenz, \u03b2=0) Ganzsignal-Inpainting — absoluter letzter Fallback (§2.47).

        Wird aufgerufen wenn alle Kanal-Verarbeitungspfade (_process_channel) fehlschlagen.
        Zerlegt das gesamte Spektrum in NMF-Komponenten und rekonstruiert Niedrigenergie-Regionen
        (\u22641. Quartil), die wahrscheinlich beschädigt sind, via NMF-Schätzung.

        Args:
            audio:    Input audio (mono 1-D oder Stereo N×2).
            sr:       Sample-Rate (muss 48000 Hz sein).
            strength: Blend-Faktor für NMF-Reparatur in Niedrigenergie-Bins (0–1).

        Returns:
            Inpainted audio, identical shape to input, NaN/Inf-frei, ∈ [-1, 1].
        """
        from scipy import signal as _sps  # pylint: disable=import-outside-toplevel

        _n_fft = 2048
        _hop = 512
        _is_stereo = audio.ndim == 2
        _mono = np.mean(audio, axis=1) if _is_stereo else audio.copy()
        _mono = _mono.astype(np.float64)

        _, _, _Z = _sps.stft(_mono, fs=sr, nperseg=_n_fft, noverlap=_n_fft - _hop, window="hann")
        _mag = np.abs(_Z)
        _phase = np.angle(_Z)

        _n_freq, _n_time = _mag.shape
        _rank = min(16, max(2, _n_time // 4))
        if _n_time < 4:
            return audio.copy()

        _eps = 1e-10
        _rng = np.random.default_rng(seed=42)  # deterministic — reproducible for same input
        _W = _rng.random((_n_freq, _rank)) + 0.1  # basis spectra
        _H = _rng.random((_rank, _n_time)) + 0.1  # activations

        for _ in range(10):  # IS-divergence multiplicative updates (\u03b2=0)
            _WH = _W @ _H + _eps
            _H *= (_W.T @ (_mag / _WH**2)) / (_W.T @ (1.0 / _WH) + _eps)
            _WH = _W @ _H + _eps
            _W *= ((_mag / _WH**2) @ _H.T) / ((1.0 / _WH) @ _H.T + _eps)

        _reconstructed_mag = _W @ _H

        # Apply NMF reconstruction only to lowest-quartile energy bins (likely damage)
        _energy_mask = _mag < np.percentile(_mag, 25)
        _mag_repaired = np.where(
            _energy_mask,
            (1.0 - strength) * _mag + strength * _reconstructed_mag,
            _mag,
        )

        # Phase-coherent ISTFT via PGHI (§2.47)
        try:
            from dsp.pghi import pghi_reconstruct as _pghi_fb  # pylint: disable=import-outside-toplevel

            _repaired_mono = _pghi_fb(
                _mag_repaired.astype(np.float32),
                sr=sr,
                win_size=_n_fft,
                hop=_hop,
                initial_phase=_phase.astype(np.float32),
            )
            _repaired_mono = np.asarray(_repaired_mono, dtype=np.float64)
        except (ImportError, Exception) as _pghi_fb_err:
            logger.debug("Verarbeitungsschritt_55 NMF-Ersatzpfad PGHI-Ersatzpfad: %s", _pghi_fb_err)
            _Z_repaired = _mag_repaired * np.exp(1j * _phase)
            _, _repaired_mono = _sps.istft(_Z_repaired, fs=sr, nperseg=_n_fft, noverlap=_n_fft - _hop, window="hann")

        _repaired_mono = np.asarray(_repaired_mono, dtype=np.float64)
        if len(_repaired_mono) > len(_mono):
            _repaired_mono = _repaired_mono[: len(_mono)]
        elif len(_repaired_mono) < len(_mono):
            _repaired_mono = np.pad(_repaired_mono, (0, len(_mono) - len(_repaired_mono)))

        _repaired_mono = np.nan_to_num(_repaired_mono, nan=0.0, posinf=0.0, neginf=0.0)
        _repaired_mono = np.clip(_repaired_mono, -1.0, 1.0)

        if _is_stereo:
            return _apply_shared_stereo_ratio(audio, _mono, _repaired_mono)

        return _repaired_mono.astype(np.float32)  # type: ignore[no-any-return]

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        **kwargs: Any,
    ) -> PhaseResult:
        check_ml_model_ready("AudioLDM2", phase_name="55")
        check_ml_model_ready("PANNs", phase_name="55")
        check_ml_model_ready("Whisper", phase_name="55")
        sample_rate = int(kwargs.get("sample_rate", sample_rate))
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        t0 = time.perf_counter()

        # §4.6b: Pre-phase eviction — free previous phase models to prevent OOM
        try:
            from backend.core.plugin_lifecycle_manager import (  # pylint: disable=import-outside-toplevel
                get_plugin_lifecycle_manager as _get_plm_evict55,
            )

            _get_plm_evict55().evict_for_phase("phase_55_diffusion_inpainting")
        except Exception as e:
            logger.warning("Verarbeitungsschritt_55_diffusion_inpainting.py::verarbeiten Ersatzpfad: %s", e)

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        effective_strength = float(kwargs.get("strength", 1.0)) * phase_locality_factor
        effective_strength = float(np.clip(effective_strength, 0.0, 1.0))

        # §0 BW-Cap: prevent hallucination of HF content on bandwidth-limited carriers
        _material = kwargs.get("material_type", material_type) or material_type
        _mat_key = str(_material).lower() if _material is not None else ""
        _inpainting_profile = self._compute_inpainting_profile(
            _mat_key,
            str(kwargs.get("quality_mode", "balanced")),
            float(kwargs.get("restorability_score", 50.0)),
        )
        _vocals_conf = float(kwargs.get("panns_vocals_confidence", 0.0))
        if _vocals_conf == 0.0:  # Fallback: direct callers may use panns_singing key
            _vocals_conf = float(kwargs.get("panns_singing", 0.0))
        safe_strength = self._derive_safe_inpainting_strength(effective_strength, _mat_key, _vocals_conf)

        # §V41 ForwardMaskingGuard — Enhancement-Stärke in post-transienten Masking-Zonen erhöhen
        if _vocals_conf >= 0.25 and effective_strength > 0.0:
            try:
                from backend.core.dsp.temporal_masking import (
                    get_forward_masking_guard as _fmg_fn_55,
                )

                _fmz_55 = kwargs.get("forward_masking_zones") or _fmg_fn_55().compute_zones(audio, sample_rate)
                if _fmz_55:
                    _n_s_55 = audio.shape[-1] if audio.ndim > 1 else len(audio)
                    _zone_s_55 = sum(z.end_sample - z.start_sample for z in _fmz_55)
                    _zone_frac_55 = float(np.clip(_zone_s_55 / max(1, _n_s_55), 0.0, 1.0))
                    effective_strength = float(np.clip(effective_strength + _zone_frac_55 * 0.15, 0.0, 1.0))
            except Exception as _fmg_exc_55:
                logger.debug("Verarbeitungsschritt55 §V41 ForwardMaskingGuard nicht blockierend: %s", _fmg_exc_55)
        _goal_weights = get_effective_song_goal_weights(kwargs)
        if not isinstance(_goal_weights, dict):
            _goal_weights = None
        _restorability_score = float(kwargs.get("restorability_score", 50.0))
        # Accept both enum value strings like "MaterialType.WAX_CYLINDER" and plain keys
        for _mk, _cap in _MATERIAL_BW_CAP_HZ.items():
            if _mk in _mat_key:
                _bw_cap_hz: float | None = _cap
                break
        else:
            _bw_cap_hz = None

        if effective_strength <= 1e-6:
            dry = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            dry = np.clip(dry, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=dry,
                execution_time_seconds=time.perf_counter() - t0,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
            )

        min_gap_ms = float(_inpainting_profile["min_gap_ms"])
        wall_budget_s = float(_inpainting_profile["wall_budget_seconds"])
        min_gap_ms_eff = float(min_gap_ms) / max(safe_strength, 0.1)
        source_audio = audio

        # §11.7a: Bereits von RekonstruktionsDenker reparierte Gap-Regionen
        _repaired_gaps: list[tuple[int, int]] = kwargs.get("repaired_gap_samples", [])

        # §2.68d [RELEASE_MUST] SSIP Null-Propagation-Guard: Stille-Zonen aus ORIGINAL-Audio.
        # _get_structural_silence_zones() liefert immer eine Liste — niemals None.
        _ssip_zones_p55: list[tuple[int, int]] = []
        try:
            from backend.core.dsp.structural_silence_isolation import (  # pylint: disable=import-outside-toplevel
                _get_structural_silence_zones as _ssip_get_zones,
            )

            _mat_key_p55 = str(kwargs.get("material_type", material_type) or material_type or "unknown").lower()
            _ssip_zones_p55 = _ssip_get_zones(kwargs, audio, sample_rate, _mat_key_p55)
            if _ssip_zones_p55:
                _repaired_gaps = list(_repaired_gaps) + _ssip_zones_p55
                logger.debug(
                    "§2.68 SSIP Verarbeitungsschritt_55: %d strukturelle Stille-Zone(n) als pre-repaired markiert",
                    len(_ssip_zones_p55),
                )
        except Exception as _ssip_exc_p55:
            logger.debug("SSIP Verarbeitungsschritt_55 nicht blockierend: %s", _ssip_exc_p55)

        # §silence-guarantee: gewollte Stille sind KEINE Gaps — Stille-Zonen zur
        # pre-repaired-Liste hinzufügen, damit Inpainting sie vollständig überspringt.
        _silence_mask_p55: np.ndarray | None = kwargs.get("silence_mask")
        if _silence_mask_p55 is None:
            _ctx_p55 = kwargs.get("restoration_context")
            if isinstance(_ctx_p55, dict):
                _silence_mask_p55 = _ctx_p55.get("silence_mask")
        if isinstance(_silence_mask_p55, np.ndarray) and _silence_mask_p55.size > 1:
            try:
                _n_samp_p55 = int(audio.shape[-1] if audio.ndim == 2 else audio.shape[0])
                _mask_mono_p55 = _silence_mask_p55.ravel()[:_n_samp_p55]
                _is_sil_p55 = _mask_mono_p55 < 0.5
                if np.any(_is_sil_p55):
                    _padded_p55 = np.concatenate(([False], _is_sil_p55, [False])).astype(np.int8)  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
                    _changes_p55 = np.diff(_padded_p55)
                    _sil_starts_p55 = np.where(_changes_p55 == 1)[0].tolist()
                    _sil_ends_p55 = np.where(_changes_p55 == -1)[0].tolist()
                    _silence_zones_p55 = list(zip(_sil_starts_p55, _sil_ends_p55))
                    _repaired_gaps = list(_repaired_gaps) + _silence_zones_p55
                    logger.debug(
                        "§silence-guarantee Verarbeitungsschritt_55: %d Stille-Zone(n) als pre-repaired markiert"
                        " — Diffusions-Inpainting überspringt sie",
                        len(_silence_zones_p55),
                    )
            except Exception as _sil_exc_p55:
                logger.debug("§silence-guarantee Verarbeitungsschritt_55: nicht blockierend error: %s", _sil_exc_p55)

        # §V38 VFA-Schutzzonen für _process_channel sammeln (§0p Vocal-Supremacy)
        _p55_protected_zones: list[tuple[float, float, float]] = []
        for _z in kwargs.get("vibrato_zones") or []:
            try:
                _p55_protected_zones.append((float(_z[0]), float(_z[1]), 0.20))  # §0p Vibrato
            except Exception as e:
                logger.warning("Verarbeitungsschritt_55_diffusion_inpainting.py::unbekannter Ersatzpfad: %s", e)
        for _z in kwargs.get("frisson_zones") or []:
            try:
                _fz_s = float(getattr(_z, "start_s", None) or _z[0])
                _fz_e = float(getattr(_z, "end_s", None) or _z[1])
                _p55_protected_zones.append((_fz_s, _fz_e, 0.30))  # Frisson sakrosankt
            except Exception as e:
                logger.warning("Verarbeitungsschritt_55_diffusion_inpainting.py::unbekannter Ersatzpfad: %s", e)
        for _z in kwargs.get("whisper_zones") or []:
            try:
                _p55_protected_zones.append((float(_z[0]), float(_z[1]), 0.25))  # Flüsterpassagen
            except Exception as e:
                logger.warning("Verarbeitungsschritt_55_diffusion_inpainting.py::unbekannter Ersatzpfad: %s", e)
        for _z in kwargs.get("passaggio_zones") or []:
            try:
                _p55_protected_zones.append((float(_z[0]), float(_z[1]), 0.35))  # Passaggio-Übergänge
            except Exception as e:
                logger.warning("Verarbeitungsschritt_55_diffusion_inpainting.py::unbekannter Ersatzpfad: %s", e)
        if _p55_protected_zones:
            logger.debug(
                "§V38 Verarbeitungsschritt_55: %d VFA-Schutzzone(n) aktiv (Vibrato/Frisson/Flüster/Passaggio)",
                len(_p55_protected_zones),
            )
        _p55_pz = _p55_protected_zones or None

        _n_repaired_skipped = 0
        _damage_guard_hits = 0
        _thrash_guard_active = False
        _channel_failures = 0
        _channel_total = 1
        _full_nmf_fallback_used = False

        if audio.ndim == 1:
            # Mono
            try:
                repaired, stats = _process_channel(
                    audio,
                    sample_rate,
                    min_gap_ms_eff,
                    _repaired_gaps,
                    _bw_cap_hz,
                    wall_budget_s=wall_budget_s,
                    goal_weights=_goal_weights,
                    restorability_score=_restorability_score,
                    protected_zones=_p55_pz,
                    base_strength=safe_strength,
                )
            except Exception as _ch55_exc:
                logger.warning(
                    "Verarbeitungsschritt_55 mono channel processing fehlgeschlagen, using passthrough candidate: %s",
                    _ch55_exc,
                )
                repaired = audio.copy()
                stats = {
                    "n_gaps": 0,
                    "total_gap_ms": 0.0,
                    "max_gap_ms": 0.0,
                    "plugin_used": False,
                    "pre_repaired_skipped": 0,
                    "damage_guard_activations": 0,
                    "ml_thrashing_guard": False,
                }
                _channel_failures = 1
            gaps = _detect_gaps(audio, sample_rate, min_gap_ms_eff)
            quality = _reconstruction_quality_score(audio, repaired, gaps)
            n_gaps = stats["n_gaps"]
            total_gap_ms = stats["total_gap_ms"]
            max_gap_ms = stats["max_gap_ms"]
            plugin_used = stats["plugin_used"]
            _n_repaired_skipped = stats.get("pre_repaired_skipped", 0)
            _damage_guard_hits = int(stats.get("damage_guard_activations", 0))
            _thrash_guard_active = bool(stats.get("ml_thrashing_guard", False))
        else:
            # §2.51 Linked-Stereo: Gap-Detektion auf Mono-Mix, kohärente Reparatur
            # Detect gaps on mono downmix to ensure identical gap regions for L+R.
            #
            # §VERBOTEN Axis-Orientierungs-Guard: UV3 internes Format ist channel-first (2, N).
            # Phase_55 verarbeitet sample-first (N, 2). Falsche Orientation führt dazu, dass
            # `for ch in range(audio.shape[1])` N-mal statt 2-mal iteriert → 94K+ Warnungen
            # und keine Gap-Detektion (2-Sample-Arrays haben keine Gaps).
            _was_channel_first = audio.ndim == 2 and audio.shape[0] in (1, 2) and audio.shape[1] > audio.shape[0]
            if _was_channel_first:
                audio = audio.T  # (2, N) → (N, 2) für korrekte sample-first-Verarbeitung
                logger.debug("Verarbeitungsschritt_55: Axis-Orientierungs-Korrektur (2,N) → (N,2) angewendet")

            n_channels = audio.shape[1]
            _channel_total = n_channels
            # §2.51 Mono-Mix für verknüpfte Gap-Detektion
            mono_mix = np.mean(audio, axis=1)  # (N, channels) → (N,) korrekt
            mono_gaps = _detect_gaps(mono_mix, sample_rate, min_gap_ms_eff)

            channels_repaired = []
            n_gaps = 0
            total_gap_ms = 0.0
            max_gap_ms = 0.0
            plugin_used = False
            quality_scores = []

            for ch in range(n_channels):
                try:
                    ch_rep, stats = _process_channel(
                        audio[:, ch],
                        sample_rate,
                        min_gap_ms_eff,
                        _repaired_gaps,
                        _bw_cap_hz,
                        precomputed_gaps=mono_gaps,
                        wall_budget_s=wall_budget_s,
                        goal_weights=_goal_weights,
                        restorability_score=_restorability_score,
                        protected_zones=_p55_pz,
                        base_strength=safe_strength,
                    )
                except Exception as _ch55_exc:
                    logger.warning(
                        "Verarbeitungsschritt_55 stereo channel %d processing fehlgeschlagen, using passthrough candidate: %s",
                        ch,
                        _ch55_exc,
                    )
                    ch_rep = audio[:, ch].copy()
                    stats = {
                        "n_gaps": 0,
                        "total_gap_ms": 0.0,
                        "max_gap_ms": 0.0,
                        "plugin_used": False,
                        "pre_repaired_skipped": 0,
                        "damage_guard_activations": 0,
                        "ml_thrashing_guard": False,
                    }
                    _channel_failures += 1
                channels_repaired.append(ch_rep)
                n_gaps = max(n_gaps, stats["n_gaps"])
                total_gap_ms += stats["total_gap_ms"]
                max_gap_ms = max(max_gap_ms, stats["max_gap_ms"])
                plugin_used = plugin_used or stats["plugin_used"]
                _n_repaired_skipped += stats.get("pre_repaired_skipped", 0)
                _damage_guard_hits += int(stats.get("damage_guard_activations", 0))
                _thrash_guard_active = _thrash_guard_active or bool(stats.get("ml_thrashing_guard", False))

                quality_scores.append(_reconstruction_quality_score(audio[:, ch], ch_rep, mono_gaps))

            repaired = np.column_stack(channels_repaired)  # list[(N,)] → (N, channels)
            if _was_channel_first:
                repaired = repaired.T  # (N, 2) → (2, N) zurück in UV3-Format
            quality = float(np.mean(quality_scores)) if quality_scores else 1.0

        if _channel_failures >= _channel_total:
            logger.warning(
                "Verarbeitungsschritt_55: all channel paths fehlgeschlagen (%d/%d) — invoking full-spectrum NMF Ersatzpfad",
                _channel_failures,
                _channel_total,
            )
            _fallback_input = audio
            repaired = self._nmf_spectral_inpainting_fallback(_fallback_input, sample_rate, strength=safe_strength)
            if audio.ndim == 2 and "_was_channel_first" in locals() and _was_channel_first:
                repaired = repaired.T
            _full_nmf_fallback_used = True

        if 0.0 < safe_strength < 1.0:
            repaired = source_audio + safe_strength * (repaired - source_audio)

        elapsed = time.perf_counter() - t0

        repaired = np.nan_to_num(repaired, nan=0.0, posinf=0.0, neginf=0.0)
        repaired = np.clip(repaired, -1.0, 1.0)

        # §2.68 SSIP Post-Inpainting-Audit: Hard-Reset in Stille-Zonen (letzter Layer, V17).
        # VERBOTEN: np.clip als Ersatz — Hard-Reset reproduziert Original exakt.
        if _ssip_zones_p55:
            try:
                from backend.core.dsp.structural_silence_isolation import (  # pylint: disable=import-outside-toplevel
                    get_structural_silence_isolator as _get_ssip_audit55,
                )

                repaired = _get_ssip_audit55().post_inpainting_silence_audit(
                    audio_before_inpainting=source_audio,
                    audio_after_inpainting=repaired,
                    silence_zones=_ssip_zones_p55,
                    sr=sample_rate,
                )
            except Exception as _ssip_audit_exc:
                logger.debug(
                    "SSIP post_inpainting_silence_audit Verarbeitungsschritt_55 (nicht blockierend): %s",
                    _ssip_audit_exc,
                )

        # §2.46f NPA-Guard: Atemgeräusche/Vibrato in Lücken-Rändern nicht überschreiben.
        # §2.46e Hallucination-Guard: ML-Inpainting darf kein neues spektrales Material einbringen.
        try:
            from backend.core.dsp.hallucination_guard import (  # pylint: disable=import-outside-toplevel
                check_hallucination as _check_hg55,
            )
            from backend.core.natural_performance_detector import (  # pylint: disable=import-outside-toplevel
                get_natural_performance_detector,
            )

            _mono55 = source_audio.mean(axis=0) if source_audio.ndim == 2 else source_audio
            n_samples55 = _mono55.shape[0]
            # §2.46f NPA-Guard
            try:
                _npa_mask55 = (
                    get_natural_performance_detector()
                    .detect(_mono55, sample_rate)
                    .get_protected_mask(n_samples55, sample_rate)
                )
                if _npa_mask55 is not None and _npa_mask55.any():
                    if repaired.ndim == 2:
                        repaired[:, _npa_mask55] = source_audio[:, _npa_mask55]
                    else:
                        repaired[_npa_mask55] = source_audio[_npa_mask55]
            except Exception as _npa55_exc:
                logger.debug("§2.46f Verarbeitungsschritt55 NPA-Guard (nicht blockierend): %s", _npa55_exc)
            # §2.36 Phonem-Schutz: Konsonanten-Bursts (/p/,/t/,/k/) die als Dropout
            # klassifiziert wurden dürfen nicht durch ML-Inpainting ersetzt werden —
            # Artikulation schlägt Lücken-Filling (§2.36 RELEASE_MUST).
            try:
                from backend.core.lyrics_guided_enhancement import (  # pylint: disable=import-outside-toplevel
                    get_lyrics_guided_enhancement,
                )

                _lge55 = get_lyrics_guided_enhancement()
                _phon_mask55 = _lge55.get_phoneme_mask(_mono55, sample_rate, hop_length=512)
                if _phon_mask55 is not None and len(_phon_mask55) > 0:
                    hop55 = 512
                    for _fi55, _is_burst55 in enumerate(_phon_mask55):
                        if _is_burst55:
                            _s55 = min(_fi55 * hop55, n_samples55)
                            _e55 = min(_s55 + hop55, n_samples55)
                            if repaired.ndim == 2:
                                repaired[:, _s55:_e55] = source_audio[:, _s55:_e55]
                            else:
                                repaired[_s55:_e55] = source_audio[_s55:_e55]
            except Exception as _ph55_exc:
                logger.debug("§2.36 Verarbeitungsschritt55 Phonem-Guard (nicht blockierend): %s", _ph55_exc)
            # §2.46e Hallucination-Guard: nur im Restoration-Modus (nicht Studio 2026)
            try:
                _mode55 = str(kwargs.get("mode", "restoration")).lower()
                if "studio" not in _mode55:
                    _bw_cap55 = float(_inpainting_profile.get("bw_cap_hz", 22050.0))
                    _mono_rep55 = repaired.mean(axis=0) if repaired.ndim == 2 else repaired
                    _mono_src55 = source_audio.mean(axis=0) if source_audio.ndim == 2 else source_audio
                    _hg_result55 = _check_hg55(
                        _mono_src55,
                        _mono_rep55,
                        sr=sample_rate,
                        mode=_mode55,
                        material_bw_ceiling_hz=_bw_cap55,
                        bw_extension_context=True,
                    )
                    if _hg_result55.requires_rollback:
                        logger.debug(
                            "§2.46e Verarbeitungsschritt55 Hallucination rollback: spectral_novelty=%.3f",
                            _hg_result55.spectral_novelty,
                        )
                        repaired = source_audio
                    if _hg_result55.score_penalty > 0:
                        logger.info(
                            "§2.46e Verarbeitungsschritt55 Wert_penalty=%.1f (spectral_novelty=%.3f)",
                            _hg_result55.score_penalty,
                            _hg_result55.spectral_novelty,
                        )
            except Exception as _hg55_exc:
                logger.debug("§2.46e Verarbeitungsschritt55 Hallucination-Guard (nicht blockierend): %s", _hg55_exc)
        except Exception as _guard55_exc:
            logger.debug("§2.46f/§2.46e Verarbeitungsschritt55 guards (nicht blockierend): %s", _guard55_exc)

        # §V19 (Spec-Vintage-Guard) Noise-Textur-Invariante (§NTI): Residual-Rauschen darf Material-Profil nicht ändern (non-blocking)
        try:
            from backend.core.dsp.noise_texture_guard import (  # pylint: disable=import-outside-toplevel
                compute_noise_texture_distance as _nt55_fn,
            )

            _nt55_thr = 0.18 if _vocals_conf >= 0.35 else 0.25
            _nt55_d = _nt55_fn(source_audio - repaired, _mat_key_p55, sr=sample_rate)
            if _nt55_d > _nt55_thr:
                repaired = (0.5 * repaired + 0.5 * source_audio).astype(np.float32)
                logger.warning(
                    "§V19 (Spec-Vintage-Guard) Verarbeitungsschritt_55 noise_texture_distance=%.3f > %.2f → 50%% Dry-Blend",
                    _nt55_d,
                    _nt55_thr,
                )
        except Exception as _nt55_exc:
            logger.debug(
                "§V19 (Spec-Vintage-Guard) Verarbeitungsschritt_55 noise_texture (nicht blockierend): %s", _nt55_exc
            )

        # §V24 (Spec-Vintage-Guard) Spektralfarbe-Prüfung (§2.74, non-blocking): Inpainting darf Spektralfarbe nicht verändern
        try:
            from backend.core.dsp.spectral_color_guard import (  # pylint: disable=import-outside-toplevel
                check_spectral_color_preservation as _scg55,
            )

            _sc55 = _scg55(source_audio, repaired, sample_rate)
            if not _sc55.ok:
                _sc55_wet = 0.70
                repaired = (_sc55_wet * repaired + (1.0 - _sc55_wet) * source_audio).astype(np.float32)
                logger.warning(
                    "§V24 (Spec-Vintage-Guard) Verarbeitungsschritt_55 spectral_color non-ok → strength −30%%"
                )
        except Exception as _sc55_exc:
            logger.debug(
                "§V24 (Spec-Vintage-Guard) Verarbeitungsschritt_55 spectral_color (nicht blockierend): %s", _sc55_exc
            )

        return PhaseResult(
            success=True,
            audio=repaired,
            execution_time_seconds=elapsed,
            metadata={
                "n_gaps_detected": n_gaps,
                "total_gap_ms": round(total_gap_ms, 2),
                "max_gap_ms": round(max_gap_ms, 2),
                "plugin_used": plugin_used,
                "damage_guard_activations": int(_damage_guard_hits),
                "ml_thrashing_guard": bool(_thrash_guard_active),
                "wall_budget_seconds": wall_budget_s,
                "inpainting_profile": dict(_inpainting_profile),
                "reconstruction_quality": round(quality, 4),
                "channel_failures": int(_channel_failures),
                "full_nmf_fallback_used": bool(_full_nmf_fallback_used),
                "diffusion_steps": f"{_DIFFUSION_STEPS}/{_DIFFUSION_STEPS_MED}/{_DIFFUSION_STEPS_LONG} (adaptive)",
                "min_gap_ms": min_gap_ms,
                "min_gap_ms_effective": round(min_gap_ms_eff, 2),
                "ar_order": _AR_ORDER,
                "primary_ml": "cqtdiff",
                "pre_repaired_gaps_skipped": _n_repaired_skipped,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": effective_strength,
                "safe_strength": safe_strength,
                "per_gap_local_strength_oracle": True,
                "panns_vocals_confidence": _vocals_conf,
                "bw_cap_hz": _bw_cap_hz,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
        )
