"""
core/phase_defect_verifier.py
Phase Defect Verifier (PDV) — Ursache 7: Post-Phase Defekt-Verifikation
========================================================================

After each restorative phase runs, verifies that the phase did not measurably
WORSEN the defect type(s) it was selected to fix.

Rationale (§2.45, §0 Primum non nocere):
  PMGG guards musical goals (natürlichkeit, timbre, etc.) and AFG guards
  artifact freedom, but neither checks whether the targeted physical defect
  (e.g. crackle, hum, DC offset) actually improved.  A phase can pass both
  gates while its targeted defect score increases — this is a silent
  quality regression.

Design:
  - Uses lightweight DSP proxies only (< 5 ms per defect type at 48 kHz).
  - No ML, no full DefectScanner.scan().
  - Advisory + conditional rollback: rolls back ONLY when a proxy worsens
    by more than _ROLLBACK_THRESHOLD (25 %) relative to the pre-phase value.
  - All results stored in a per-session telemetry dict for metadata export.

Proxies implemented (lower = better defect level, unless noted):
  CLICKS / CRACKLE        → impulse-ratio: p99.9(|audio|) / RMS
  HIGH_FREQ_NOISE         → HF noise floor: 5th-pct short-time energy 4–16 kHz
  HUM                     → hum energy at 50/60 Hz harmonics (up to 7th)
  LOW_FREQ_RUMBLE         → energy below 80 Hz
  DC_OFFSET               → abs(mean(audio))
  DROPOUTS                → fraction of 10-ms frames with RMS < –60 dBFS
  PHASE_ISSUES            → mid/side imbalance as mono-compat ratio (higher = better)

DefectTypes without a fast proxy (WOW, FLUTTER, PITCH_DRIFT, CLIPPING,
BANDWIDTH_LOSS, COMPRESSION_ARTIFACTS, …) are silently skipped.

Thread-safety: per-session telemetry is protected by a reentrant lock;
    the verifier singleton is double-checked-locking.

Author: Aurik Development Team
Version: 1.0.0 (v10.0.0 — Ursache 7)
"""

from __future__ import annotations

import importlib
import logging
import sys
import threading
from types import ModuleType
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_reverse_phase_map_module: ModuleType | None
try:
    _reverse_phase_map_module = importlib.import_module("backend.core.defect_phase_mapper")
except Exception:
    _reverse_phase_map_module = None

# ---------------------------------------------------------------------------
# Rollback threshold: if proxy_after > proxy_before * (1 + threshold), rollback.
# 25 % relative increase = the phase measurably WORSENED the targeted defect.
# ---------------------------------------------------------------------------
_ROLLBACK_THRESHOLD: float = 0.25

# For PHASE_ISSUES proxy (mono_compat): rollback if ratio DROPS by > threshold.
# (mono_compat: higher = better, so worsening = decrease)
_ROLLBACK_THRESHOLD_COMPAT: float = 0.20

# Absolute minimum delta to avoid rollback on tiny proxy jitter near floor.
_MIN_ABS_DELTA_BY_PROXY: dict[str, float] = {
    "hf_noise_floor": 0.01,
    "impulse_ratio": 0.10,
    "hum_energy": 0.01,
    "low_freq_energy": 0.01,
    "dc_offset": 0.002,
    "dropout_ratio": 0.01,
    "mono_compat": 0.02,
}

# Goal-aware adaptive threshold scaling (song-specific targets).
_GOAL_ADAPTIVE_THRESHOLD_CAP_MIN: float = 0.60
_GOAL_ADAPTIVE_THRESHOLD_CAP_MAX: float = 1.80
_NATURALNESS_PRESERVE_DELTA: float = 0.01
_PDV_REWEIGHT_ALPHAS: tuple[float, ...] = (0.92, 0.85, 0.78)

# Psychoakustische Audibility-Guards fuer Defect-Drift-Entscheidungen.
_PSYCHO_STRICTEN_FACTOR: float = 0.65
_PSYCHO_RELAX_FACTOR: float = 1.10
_PSYCHO_AUDIBILITY_DELTA_TRIGGER: float = 0.05
_PSYCHO_HARSHNESS_DELTA_TRIGGER: float = 0.05
_PSYCHO_BURSTINESS_DELTA_TRIGGER: float = 0.04
_PSYCHO_MODULATION_DELTA_TRIGGER: float = 0.04


def _compat_helper(name: str, default: Any) -> Any:
    """Resolve monkeypatch-compatible helpers from cassette_defect_verifier when present."""
    cassette_module = sys.modules.get("backend.core.cassette_defect_verifier")
    candidate = getattr(cassette_module, name, None) if cassette_module is not None else None
    return candidate if callable(candidate) else default


def _norm_goal_key(name: str) -> str:
    n = str(name).strip().lower()
    aliases = {
        "naturalness": "natuerlichkeit",
        "warmth": "waerme",
        "brightness": "brillanz",
        "clarity": "transparenz",
        "spatialdepth": "spatial_depth",
    }
    return aliases.get(n, n)


def _goal_val(source: dict[str, float] | None, key: str, default: float = 0.0) -> float:
    if not isinstance(source, dict):
        return float(default)
    if key in source:
        return float(source[key])
    k_norm = _norm_goal_key(key)
    for k, v in source.items():
        if _norm_goal_key(k) == k_norm:
            return float(v)
    return float(default)


def _is_legacy_material(material_type: str | None) -> bool:
    mat = str(material_type or "").lower()
    return any(k in mat for k in ("shellac", "wax", "wire", "vinyl", "tape", "cassette"))


def _compute_goal_adaptive_threshold_scale(
    proxy_name: str,
    goal_before: dict[str, float] | None,
    goal_after: dict[str, float] | None,
    goal_targets: dict[str, float] | None,
    goal_weights: dict[str, float] | None,
    material_type: str | None,
) -> float:
    """Compute proxy-threshold multiplier from song-specific goal targets.

    > 1.0 => more tolerant, < 1.0 => stricter.
    """
    scale = 1.0

    # Weighted clarity-vs-preserve balance from SongGoalImportance.
    w_clarity = (_goal_val(goal_weights, "transparenz", 1.0) + _goal_val(goal_weights, "brillanz", 1.0)) / 2.0
    w_preserve = (_goal_val(goal_weights, "natuerlichkeit", 1.0) + _goal_val(goal_weights, "waerme", 1.0)) / 2.0
    if w_preserve > 1e-6:
        scale *= float(np.clip(np.sqrt(w_clarity / w_preserve), 0.85, 1.25))

    # Song-target deficits/surplus (dynamic per song, not global fixed).
    t_trans = _goal_val(goal_targets, "transparenz", 0.0)
    t_bril = _goal_val(goal_targets, "brillanz", 0.0)
    t_waer = _goal_val(goal_targets, "waerme", 0.0)
    b_trans = _goal_val(goal_before, "transparenz", 0.0)
    b_bril = _goal_val(goal_before, "brillanz", 0.0)
    a_trans = _goal_val(goal_after, "transparenz", b_trans)
    a_bril = _goal_val(goal_after, "brillanz", b_bril)
    b_waer = _goal_val(goal_before, "waerme", 0.0)

    clarity_def_before = max(0.0, t_trans - b_trans) + max(0.0, t_bril - b_bril)
    clarity_def_after = max(0.0, t_trans - a_trans) + max(0.0, t_bril - a_bril)
    clarity_improved = clarity_def_after < clarity_def_before - 1e-6
    warmth_surplus = max(0.0, b_waer - t_waer)

    if proxy_name == "hf_noise_floor":
        if clarity_def_before > 0.05:
            scale *= 1.15
        if clarity_improved:
            scale *= 1.10
        if warmth_surplus > 0.05 and _is_legacy_material(material_type):
            scale *= 1.10

    return float(np.clip(scale, _GOAL_ADAPTIVE_THRESHOLD_CAP_MIN, _GOAL_ADAPTIVE_THRESHOLD_CAP_MAX))


# ---------------------------------------------------------------------------
# Fast proxy implementations (all operate on mono mix if stereo)
# ---------------------------------------------------------------------------


def _as_mono(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:
        return np.asarray(np.mean(arr, axis=0, dtype=np.float32), dtype=np.float32)  # type: ignore[no-any-return]
    return arr  # type: ignore[no-any-return]


def _proxy_impulse_ratio(audio: np.ndarray) -> float:
    """99.9th-pct amplitude / RMS — click/crackle indicator. Lower = better."""
    try:
        mono = _as_mono(audio)
        rms = float(np.sqrt(np.mean(mono**2)) + 1e-9)
        p999 = float(np.percentile(np.abs(mono), 99.9))
        return float(p999 / rms)
    except Exception as e:
        logger.warning("Verarbeitungsschritt_defect_verifier.py::_proxy_impulse_Verhaeltnis Ersatzpfad: %s", e)
        return 1.0


def _proxy_hf_noise_floor(audio: np.ndarray, sr: int) -> float:
    """5th-percentile short-time energy in the 4–16 kHz band. Lower = better."""
    try:
        mono = _as_mono(audio)
        n_fft: int = 2048
        hop: int = 1024
        bin_4k = max(int(4000 * n_fft / sr), 1)
        bin_16k = min(int(16000 * n_fft / sr), n_fft // 2 + 1)
        if bin_16k <= bin_4k:
            return 0.0
        frame_energies: list[float] = []
        frame_rms: list[float] = []
        for i in range(0, max(0, len(mono) - n_fft), hop):
            chunk = mono[i : i + n_fft]
            if len(chunk) < n_fft:
                break
            mag = np.abs(np.fft.rfft(chunk * np.hanning(n_fft)))
            frame_energies.append(float(np.mean(mag[bin_4k:bin_16k])))
            frame_rms.append(float(np.sqrt(np.mean(chunk**2) + 1e-12)))
        if frame_energies:
            # Noise-floor proxy should reflect quiet/near-silence frames, not voiced/music peaks.
            quiet_threshold = float(np.percentile(frame_rms, 35)) if frame_rms else 0.0
            quiet_energies = [e for e, r in zip(frame_energies, frame_rms, strict=False) if r <= quiet_threshold]
            if quiet_energies:
                return float(np.percentile(quiet_energies, 5))
            return float(np.percentile(frame_energies, 5))
    except Exception as e:
        logger.warning("Verarbeitungsschritt_defect_verifier.py::_proxy_hf_noise_floor Ersatzpfad: %s", e)
    return 0.0


def _compute_hf_noise_audibility(audio: np.ndarray, sr: int) -> float:
    """Psychoakustische Schaetzung der HF-Rausch-Hoerbarkeit (0..1).

    Modell: HF-Energie (4-12 kHz) wird gegen eine einfache simultane
    Maskierung durch 300-3000 Hz verglichen. Hoher Wert = eher hoerbar.
    """
    try:
        mono = _as_mono(audio)
        n_fft = 2048
        hop = 1024
        if len(mono) < n_fft:
            return 0.0

        hf_lo = max(1, int(4000 * n_fft / sr))
        hf_hi = min(int(12000 * n_fft / sr), n_fft // 2 + 1)
        mask_lo = max(1, int(300 * n_fft / sr))
        mask_hi = min(int(3000 * n_fft / sr), n_fft // 2 + 1)
        if hf_hi <= hf_lo or mask_hi <= mask_lo:
            return 0.0

        audible_scores: list[float] = []
        win = np.hanning(n_fft).astype(np.float32)
        eps = 1e-12
        for i in range(0, len(mono) - n_fft + 1, hop):
            frame = mono[i : i + n_fft] * win
            mag = np.abs(np.fft.rfft(frame)).astype(np.float32)
            hf_e = float(np.mean(mag[hf_lo:hf_hi]) + eps)
            mask_e = float(np.mean(mag[mask_lo:mask_hi]) + eps)
            hf_db = 20.0 * np.log10(hf_e)
            mask_db = 20.0 * np.log10(mask_e)
            # Vereinfachte simultane Maskierung: Signal wird ab ~18 dB unter
            # dominanter Midband-Energie zunehmend unhoerbar.
            audible_db = hf_db - (mask_db - 18.0)
            audible_scores.append(float(np.clip((audible_db + 18.0) / 36.0, 0.0, 1.0)))

        if not audible_scores:
            return 0.0
        return float(np.clip(np.percentile(audible_scores, 80), 0.0, 1.0))
    except Exception as e:
        logger.warning("Verarbeitungsschritt_defect_verifier.py::_berechnen_hf_noise_audibility Ersatzpfad: %s", e)
        return 0.0


def _compute_transient_harshness(audio: np.ndarray, sr: int) -> float:
    """Schaetzt psychoakustisch harte HF-Transienten (0..1).

    Hohe Werte korrelieren mit klickender/kratziger Wahrnehmung im Praesenzband.
    """
    try:
        mono = _as_mono(audio)
        n_fft = 1024
        hop = 256
        if len(mono) < n_fft:
            return 0.0

        lo = max(1, int(3000 * n_fft / sr))
        hi = min(int(10000 * n_fft / sr), n_fft // 2 + 1)
        if hi <= lo:
            return 0.0

        win = np.hanning(n_fft).astype(np.float32)
        hf_env: list[float] = []
        for i in range(0, len(mono) - n_fft + 1, hop):
            frame = mono[i : i + n_fft] * win
            mag = np.abs(np.fft.rfft(frame)).astype(np.float32)
            hf_env.append(float(np.mean(mag[lo:hi])))

        if len(hf_env) < 3:
            return 0.0
        env = np.asarray(hf_env, dtype=np.float32)
        d = np.diff(env)
        jumps = d[d > 0.0]
        if jumps.size == 0:
            return 0.0

        ref = float(np.percentile(env, 50) + 1e-9)
        burst = float(np.percentile(jumps, 95))
        ratio = burst / ref
        return float(np.clip(ratio / 6.0, 0.0, 1.0))
    except Exception as e:
        logger.warning("Verarbeitungsschritt_defect_verifier.py::_berechnen_transient_harshness Ersatzpfad: %s", e)
        return 0.0


def _compute_quasi_peak_burstiness(audio: np.ndarray, sr: int) -> float:
    """Approximation eines 468-aehnlichen Quasi-Peak-Burst-Indikators (0..1).

    Ziel: kurze, stoerende Impuls-Bursts staerker gewichten als stationaere Energie.
    """
    try:
        mono = _as_mono(audio)
        if mono.size < 16:
            return 0.0

        # Leichtes Praesenzband-Weighting (~4-8 kHz) ueber 1. Ordnung Diffs.
        hp = np.diff(mono, prepend=mono[0]).astype(np.float32)
        env = np.abs(hp)

        # Quasi-Peak-Huelle: schneller Attack, langsamer Release.
        attack = float(np.exp(-1.0 / max(1.0, sr * 0.0015)))
        release = float(np.exp(-1.0 / max(1.0, sr * 0.0250)))
        qp = np.empty_like(env, dtype=np.float32)
        state = 0.0
        for i, x in enumerate(env):
            if x > state:
                state = attack * state + (1.0 - attack) * float(x)
            else:
                state = release * state + (1.0 - release) * float(x)
            qp[i] = state

        p95 = float(np.percentile(qp, 95))
        med = float(np.percentile(qp, 50) + 1e-9)
        ratio = p95 / med
        return float(np.clip((ratio - 1.0) / 8.0, 0.0, 1.0))
    except Exception as e:
        logger.warning("Verarbeitungsschritt_defect_verifier.py::_berechnen_quasi_peak_burstiness Ersatzpfad: %s", e)
        return 0.0


def _compute_modulation_roughness(audio: np.ndarray, sr: int) -> float:
    """Schaetzt stoerende Modulationsrauhigkeit im Kurzzeitverlauf (0..1).

    Hohe Werte deuten auf unruhige, pumpende Pegelmodulationen hin, die
    psychoakustisch schnell als kuenstlich wahrgenommen werden.
    """
    try:
        mono = _as_mono(audio)
        frame = max(64, int(0.010 * sr))
        hop = max(32, int(0.005 * sr))
        if len(mono) < frame:
            return 0.0

        env: list[float] = []
        for i in range(0, len(mono) - frame + 1, hop):
            chunk = mono[i : i + frame]
            env.append(float(np.sqrt(np.mean(chunk**2) + 1e-12)))

        if len(env) < 4:
            return 0.0

        e = np.asarray(env, dtype=np.float32)
        d = np.abs(np.diff(e))
        if d.size == 0:
            return 0.0

        ref = float(np.percentile(e, 50) + 1e-9)
        jitter = float(np.percentile(d, 90))
        ratio = jitter / ref
        return float(np.clip(ratio / 0.35, 0.0, 1.0))
    except Exception as e:
        logger.warning("Verarbeitungsschritt_defect_verifier.py::_berechnen_modulation_roughness Ersatzpfad: %s", e)
        return 0.0


def _select_reweight_alphas_for_defect(worst_defect: str) -> tuple[float, ...]:
    """Defektspezifische Reweight-Staffel fuer intelligentere Recovery-Pfade."""
    d = str(worst_defect or "").upper()
    if d in {"CLICKS", "CRACKLE"}:
        return (0.90, 0.82, 0.74)
    if d in {"HUM", "LOW_FREQ_RUMBLE"}:
        return (0.95, 0.90, 0.85)
    if d == "DC_OFFSET":
        return (0.97, 0.92, 0.88)
    return _PDV_REWEIGHT_ALPHAS


def _frequency_selective_blend(
    audio_before: np.ndarray,
    audio_after: np.ndarray,
    alpha: float,
    worst_defect: str,
    sr: int,
) -> np.ndarray:
    """Defektspezifisches Reweighting im Frequenzraum.

    Ziel: problematische Defektbaender konservativer aus dem Pre-Phase-Audio
    uebernehmen, waehrend der Rest des Spektrums vom Post-Phase-Audio profitiert.
    """
    d = str(worst_defect or "").upper()
    if d not in {"HUM", "LOW_FREQ_RUMBLE", "DC_OFFSET", "CLICKS", "CRACKLE"}:
        return np.clip(alpha * audio_after + (1.0 - alpha) * audio_before, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

    arr_b = np.asarray(audio_before, dtype=np.float32)
    arr_a = np.asarray(audio_after, dtype=np.float32)
    if arr_b.shape != arr_a.shape:
        return np.clip(alpha * arr_a + (1.0 - alpha) * arr_b, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

    was_mono = arr_b.ndim == 1
    if was_mono:
        arr_b = arr_b[np.newaxis, :]
        arr_a = arr_a[np.newaxis, :]

    n = int(arr_b.shape[-1])
    if n < 16:
        out = np.clip(alpha * arr_a + (1.0 - alpha) * arr_b, -1.0, 1.0).astype(np.float32)
        return out[0] if was_mono else out  # type: ignore[no-any-return]

    local_alpha = float(np.clip(alpha - 0.28, 0.05, 0.98))

    if d in {"CLICKS", "CRACKLE"}:
        # Quasi-peak-gefuehrte, zeitlokale Reweight-Maske fuer Impulsartefakte.
        out_td = np.empty_like(arr_b, dtype=np.float32)
        for ch in range(arr_b.shape[0]):
            hp = np.diff(arr_a[ch], prepend=arr_a[ch][0]).astype(np.float32)
            env = np.abs(hp)
            thr = float(np.percentile(env, 92))
            burst_mask = (env >= thr).astype(np.float32)
            # leichte Ausdehnung in der Zeit, damit kurze Bursts voll erfasst werden
            k = max(3, int(0.0015 * sr))
            kernel = np.ones(k, dtype=np.float32) / float(k)
            burst_mask = np.convolve(burst_mask, kernel, mode="same")
            burst_mask = np.clip(burst_mask, 0.0, 1.0)

            alpha_curve = (float(alpha) - (float(alpha) - local_alpha) * burst_mask).astype(np.float32)
            out_td[ch] = alpha_curve * arr_a[ch] + (1.0 - alpha_curve) * arr_b[ch]

        out_td = np.clip(out_td, -1.0, 1.0).astype(np.float32)
        return out_td[0] if was_mono else out_td  # type: ignore[no-any-return]

    freqs = np.fft.rfftfreq(n, d=1.0 / float(max(1, sr)))
    alpha_map = np.full(freqs.shape, float(alpha), dtype=np.float32)

    if d == "DC_OFFSET":
        band_mask = freqs <= 15.0
    elif d == "LOW_FREQ_RUMBLE":
        band_mask = freqs <= 120.0
    else:
        # HUM: 50/60 Hz + Harmonische konservativer behandeln.
        band_mask = np.zeros(freqs.shape, dtype=bool)
        for f0 in (50.0, 60.0):
            for k in range(1, 7):
                fk = f0 * float(k)
                band_mask |= np.abs(freqs - fk) <= 8.0

    alpha_map[band_mask] = local_alpha

    out = np.empty_like(arr_b, dtype=np.float32)
    for ch in range(arr_b.shape[0]):
        spec_b = np.fft.rfft(arr_b[ch])
        spec_a = np.fft.rfft(arr_a[ch])
        spec_mix = alpha_map * spec_a + (1.0 - alpha_map) * spec_b
        out[ch] = np.fft.irfft(spec_mix, n=n).astype(np.float32)

    out = np.clip(out, -1.0, 1.0).astype(np.float32)
    return out[0] if was_mono else out  # type: ignore[no-any-return]


def _proxy_hum_energy(audio: np.ndarray, sr: int) -> float:
    """Sum of energy at 50 Hz and 60 Hz harmonic series (up to 7th). Lower = better."""
    try:
        mono = _as_mono(audio)
        n = len(mono)
        if n < 2:
            return 0.0
        mag = np.abs(np.fft.rfft(mono.astype(np.float32)))
        energy = 0.0
        for f0 in (50.0, 60.0):
            for harmonic in range(1, 8):
                freq = f0 * harmonic
                if freq >= sr / 2:
                    break
                b_center = int(round(freq * n / sr))
                for b in range(max(0, b_center - 1), min(b_center + 2, len(mag))):
                    energy += float(mag[b])
        return float(energy)
    except Exception as e:
        logger.warning("Verarbeitungsschritt_defect_verifier.py::_proxy_hum_energy Ersatzpfad: %s", e)
        return 0.0


def _proxy_low_freq_energy(audio: np.ndarray, sr: int) -> float:
    """Mean spectral energy below 80 Hz. Lower = better."""
    try:
        mono = _as_mono(audio)
        n = len(mono)
        mag = np.abs(np.fft.rfft(mono.astype(np.float32)))
        bin_80 = int(80 * n / sr)
        if bin_80 > 0 and bin_80 < len(mag):
            return float(np.mean(mag[:bin_80]))
    except Exception as e:
        logger.warning("Verarbeitungsschritt_defect_verifier.py::_proxy_low_freq_energy Ersatzpfad: %s", e)
    return 0.0


def _proxy_dc_offset(audio: np.ndarray) -> float:
    """Absolute mean — DC offset indicator. Lower = better."""
    try:
        mono = _as_mono(audio)
        return float(abs(np.mean(mono)))
    except Exception as e:
        logger.warning("Verarbeitungsschritt_defect_verifier.py::_proxy_dc_offset Ersatzpfad: %s", e)
        return 0.0


def _proxy_dropout_ratio(audio: np.ndarray, sr: int) -> float:
    """Fraction of 10-ms frames whose RMS < –60 dBFS. Lower = better."""
    try:
        mono = _as_mono(audio)
        frame_len = max(1, int(0.010 * sr))
        n_frames = len(mono) // frame_len
        if n_frames == 0:
            return 0.0
        threshold = 10.0 ** (-60.0 / 20.0)  # ≈ 0.001
        silent = 0
        for i in range(n_frames):
            chunk = mono[i * frame_len : (i + 1) * frame_len]
            rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))
            if rms < threshold:
                silent += 1
        return float(silent / n_frames)
    except Exception as e:
        logger.warning("Verarbeitungsschritt_defect_verifier.py::_proxy_dropout_Verhaeltnis Ersatzpfad: %s", e)
        return 0.0


def _proxy_mono_compat(audio: np.ndarray) -> float:
    """Mid energy / (Mid + Side energy) ratio ∈ [0, 1]. Higher = better mono compat."""
    try:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2:
            return 1.0
        mid = 0.5 * (arr[0] + arr[1])
        side = 0.5 * (arr[0] - arr[1])
        mid_rms = float(np.sqrt(np.mean(mid**2)) + 1e-12)
        side_rms = float(np.sqrt(np.mean(side**2)) + 1e-12)
        return float(np.clip(mid_rms / (mid_rms + side_rms + 1e-12), 0.0, 1.0))
    except Exception as e:
        logger.warning("Verarbeitungsschritt_defect_verifier.py::_proxy_mono_compat Ersatzpfad: %s", e)
        return 1.0


# ---------------------------------------------------------------------------
# DefectType name → proxy function dispatch
# (uses string names to avoid circular imports when DefectType changes)
# ---------------------------------------------------------------------------

_PROXY_DISPATCH: dict[str, str] = {
    "CLICKS": "impulse_ratio",
    "CRACKLE": "impulse_ratio",
    "HIGH_FREQ_NOISE": "hf_noise_floor",
    "HUM": "hum_energy",
    "LOW_FREQ_RUMBLE": "low_freq_energy",
    "DC_OFFSET": "dc_offset",
    "DROPOUTS": "dropout_ratio",
    "PHASE_ISSUES": "mono_compat",
}

# Proxies where HIGHER value = better (i.e., worsening = decrease)
_HIGHER_IS_BETTER: frozenset[str] = frozenset({"mono_compat"})


def _compute_proxy(proxy_name: str, audio: np.ndarray, sr: int) -> float:
    if proxy_name == "impulse_ratio":
        return _proxy_impulse_ratio(audio)
    if proxy_name == "hf_noise_floor":
        return _proxy_hf_noise_floor(audio, sr)
    if proxy_name == "hum_energy":
        return _proxy_hum_energy(audio, sr)
    if proxy_name == "low_freq_energy":
        return _proxy_low_freq_energy(audio, sr)
    if proxy_name == "dc_offset":
        return _proxy_dc_offset(audio)
    if proxy_name == "dropout_ratio":
        return _proxy_dropout_ratio(audio, sr)
    if proxy_name == "mono_compat":
        return _proxy_mono_compat(audio)
    return 0.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


# canonical import — keep for backward compatibility
from backend.core.cassette_defect_verifier import DefectVerificationResult

# ---------------------------------------------------------------------------
# PhaseDefectVerifier singleton
# ---------------------------------------------------------------------------

_PDV_INSTANCE_HOLDER: dict[str, PhaseDefectVerifier | None] = {"instance": None}
_pdv_lock = threading.Lock()


def get_phase_defect_verifier() -> PhaseDefectVerifier:
    """Thread-safe singleton accessor."""
    if _PDV_INSTANCE_HOLDER["instance"] is None:
        with _pdv_lock:
            if _PDV_INSTANCE_HOLDER["instance"] is None:
                _PDV_INSTANCE_HOLDER["instance"] = PhaseDefectVerifier()
    instance = _PDV_INSTANCE_HOLDER["instance"]
    assert instance is not None
    return instance


class PhaseDefectVerifier:
    """Verifies that each restorative phase does not worsen its targeted defects.

    Usage in _execute_pipeline():
        # Before phase:
        _pdv_ref = current_audio  # already available as _afg_phase_input
        # After phase + quality intervention:
        current_audio = get_phase_defect_verifier().check(
            phase_id, _pdv_ref, current_audio, sample_rate, metadata_dict
        )
    """

    def __init__(self) -> None:
        self._session_telemetry: list[dict] = []
        self._telem_lock = threading.RLock()
        self._reverse_map: dict[str, list[str]] | None = None
        self._rmap_lock = threading.Lock()

    def _get_reverse_map(self) -> dict[str, list[str]]:
        """Lazy-load reverse phase→defect-type map (string names, no circular import)."""
        if self._reverse_map is None:
            with self._rmap_lock:
                if self._reverse_map is None:
                    try:
                        module = _reverse_phase_map_module
                        if module is None:
                            module = importlib.import_module("backend.core.defect_phase_mapper")
                        raw = module.get_reverse_phase_map()
                        # Convert DefectType enum values to string names
                        self._reverse_map = {pid: [dt.name for dt in dtypes] for pid, dtypes in raw.items()}
                    except Exception as exc:
                        logger.debug("PDV: reverse map laden fehlgeschlagen: %s", exc)
                        self._reverse_map = {}
        return self._reverse_map or {}

    def measure_proxies(self, phase_id: str, audio: np.ndarray, sr: int) -> dict[str, float]:
        """Berechnet defect proxies for all targeted defect types of phase_id.

        Returns an empty dict if phase_id is not a restorative phase or if
        audio/sr are invalid.  Never raises.
        """
        try:
            reverse_map = self._get_reverse_map()
            defect_names = reverse_map.get(phase_id, [])
            if not defect_names:
                return {}
            result: dict[str, float] = {}
            for dname in defect_names:
                pname = _PROXY_DISPATCH.get(dname)
                if pname is None:
                    continue
                result[dname] = _compute_proxy(pname, audio, sr)
            return result
        except Exception as exc:
            logger.debug("PDV.measure_proxies(%s) fehlgeschlagen: %s", phase_id, exc)
            return {}

    def check(
        self,
        phase_id: str,
        audio_before: np.ndarray,
        audio_after: np.ndarray,
        sr: int,
        metadata_store: dict | None = None,
        goal_before: dict[str, float] | None = None,
        goal_after: dict[str, float] | None = None,
        goal_targets: dict[str, float] | None = None,
        goal_weights: dict[str, float] | None = None,
        material_type: str | None = None,
    ) -> np.ndarray:
        """Prüft whether the phase worsened its targeted defects.

        Args:
            phase_id:       Full phase ID (e.g. 'phase_09_crackle_removal').
            audio_before:   Audio BEFORE the phase ran (pre-phase reference).
            audio_after:    Audio AFTER the phase + quality-intervention.
            sr:             Sample rate (must be 48000 in processing context).
            metadata_store: If provided, appended with a 'phase_defect_verification'
                            entry for this phase (non-blocking, best-effort).

        Returns:
            audio_after normally, or audio_before if a rollback was triggered.
            Never raises.
        """
        try:
            proxies_before = self.measure_proxies(phase_id, audio_before, sr)
            if not proxies_before:
                # No proxies available for this phase — skip silently.
                return audio_after

            reverse_map = self._get_reverse_map()
            all_defects = reverse_map.get(phase_id, [])
            skipped: list[str] = [d for d in all_defects if d not in proxies_before]

            # Naturalness is a hard guard: bei Verletzung kein Reweighting.
            _nat_before = _goal_val(goal_before, "natuerlichkeit", 0.0)
            _nat_after = _goal_val(goal_after, "natuerlichkeit", _nat_before)
            _hard_naturalness_violation = _nat_after + _NATURALNESS_PRESERVE_DELTA < _nat_before

            def _evaluate_candidate(candidate_audio: np.ndarray) -> tuple[dict[str, float], str, float, bool]:
                _proxies_after = self.measure_proxies(phase_id, candidate_audio, sr)
                _worst_defect = ""
                _worst_change = 0.0
                _rollback = False
                _hf_fn = _compat_helper("_compute_hf_noise_audibility", _compute_hf_noise_audibility)
                _harsh_fn = _compat_helper("_compute_transient_harshness", _compute_transient_harshness)
                _burst_fn = _compat_helper("_compute_quasi_peak_burstiness", _compute_quasi_peak_burstiness)
                _mod_fn = _compat_helper("_compute_modulation_roughness", _compute_modulation_roughness)
                _hf_aud_before = _hf_fn(audio_before, sr)
                _hf_aud_after = _hf_fn(candidate_audio, sr)
                _harsh_before = _harsh_fn(audio_before, sr)
                _harsh_after = _harsh_fn(candidate_audio, sr)
                _burst_before = _burst_fn(audio_before, sr)
                _burst_after = _burst_fn(candidate_audio, sr)
                _mod_before = _mod_fn(audio_before, sr)
                _mod_after = _mod_fn(candidate_audio, sr)

                for dname, val_before in proxies_before.items():
                    val_after = _proxies_after.get(dname, val_before)
                    pname = _PROXY_DISPATCH.get(dname, "")

                    if pname in _HIGHER_IS_BETTER:
                        if val_before > 1e-9:
                            rel_change = (val_before - val_after) / val_before
                        else:
                            rel_change = 0.0
                        threshold = _ROLLBACK_THRESHOLD_COMPAT
                    else:
                        if val_before > 1e-9:
                            rel_change = (val_after - val_before) / val_before
                        else:
                            rel_change = 0.0
                        threshold = _ROLLBACK_THRESHOLD

                    _threshold_scale = _compute_goal_adaptive_threshold_scale(
                        pname,
                        goal_before=goal_before,
                        goal_after=goal_after,
                        goal_targets=goal_targets,
                        goal_weights=goal_weights,
                        material_type=material_type,
                    )
                    threshold *= _threshold_scale

                    if pname == "hf_noise_floor":
                        if (_hf_aud_after - _hf_aud_before) > _PSYCHO_AUDIBILITY_DELTA_TRIGGER:
                            threshold *= _PSYCHO_STRICTEN_FACTOR
                        elif (_hf_aud_before - _hf_aud_after) > _PSYCHO_AUDIBILITY_DELTA_TRIGGER:
                            threshold *= _PSYCHO_RELAX_FACTOR

                    if pname == "impulse_ratio":
                        if (_harsh_after - _harsh_before) > _PSYCHO_HARSHNESS_DELTA_TRIGGER:
                            threshold *= _PSYCHO_STRICTEN_FACTOR
                        elif (_harsh_before - _harsh_after) > _PSYCHO_HARSHNESS_DELTA_TRIGGER:
                            threshold *= _PSYCHO_RELAX_FACTOR
                        if (_burst_after - _burst_before) > _PSYCHO_BURSTINESS_DELTA_TRIGGER:
                            threshold *= _PSYCHO_STRICTEN_FACTOR
                        elif (_burst_before - _burst_after) > _PSYCHO_BURSTINESS_DELTA_TRIGGER:
                            threshold *= _PSYCHO_RELAX_FACTOR

                    if pname in {"impulse_ratio", "hf_noise_floor"}:
                        if (_mod_after - _mod_before) > _PSYCHO_MODULATION_DELTA_TRIGGER:
                            threshold *= _PSYCHO_STRICTEN_FACTOR
                        elif (_mod_before - _mod_after) > _PSYCHO_MODULATION_DELTA_TRIGGER:
                            threshold *= _PSYCHO_RELAX_FACTOR

                    abs_delta = abs(float(val_after) - float(val_before))
                    min_abs_delta = _MIN_ABS_DELTA_BY_PROXY.get(pname, 0.0)
                    if abs_delta < min_abs_delta:
                        rel_change = 0.0

                    if rel_change > _worst_change:
                        _worst_change = rel_change
                        _worst_defect = dname

                    if rel_change > threshold:
                        _rollback = True

                return _proxies_after, _worst_defect, _worst_change, _rollback

            proxies_after, worst_defect, worst_change, rollback = _evaluate_candidate(audio_after)

            if _hard_naturalness_violation:
                rollback = True
                worst_defect = "NATURALNESS_PRESERVE_GUARD"
                worst_change = max(worst_change, _nat_before - _nat_after)

            _reweight_applied = False
            _reweight_alpha = 0.0
            _reweight_strategy = "none"
            if rollback and not _hard_naturalness_violation:
                _alphas = _select_reweight_alphas_for_defect(worst_defect)
                for alpha in _alphas:
                    try:
                        _wd_key = str(worst_defect or "").upper()
                        if _wd_key in {"HUM", "LOW_FREQ_RUMBLE", "DC_OFFSET", "CLICKS", "CRACKLE"}:
                            _blend_fn = _compat_helper("_frequency_selective_blend", _frequency_selective_blend)
                            blended = _blend_fn(
                                audio_before=audio_before,
                                audio_after=audio_after,
                                alpha=float(alpha),
                                worst_defect=worst_defect,
                                sr=sr,
                            )
                            _candidate_strategy = (
                                "burst_selective" if _wd_key in {"CLICKS", "CRACKLE"} else "frequency_selective"
                            )
                        else:
                            blended = np.clip(alpha * audio_after + (1.0 - alpha) * audio_before, -1.0, 1.0).astype(
                                np.float32
                            )
                            _candidate_strategy = "global"
                        _pa, _wd, _wc, _rb = _evaluate_candidate(blended)
                        if not _rb:
                            audio_after = blended
                            proxies_after = _pa
                            worst_defect = _wd
                            worst_change = _wc
                            rollback = False
                            _reweight_applied = True
                            _reweight_alpha = float(alpha)
                            _reweight_strategy = _candidate_strategy
                            logger.info(
                                "§PDV reweight: %s stabilized with alpha=%.2f (%s) — rollback avoided",
                                phase_id,
                                alpha,
                                _candidate_strategy,
                            )
                            break
                    except Exception as _rw_exc:
                        logger.debug("PDV reweight alpha=%.2f fehlgeschlagen for %s: %s", alpha, phase_id, _rw_exc)

            result = DefectVerificationResult(
                phase_id=phase_id,
                targeted_defects=list(proxies_before.keys()),
                proxies_before=proxies_before,
                proxies_after=proxies_after,
                worst_defect=worst_defect,
                worst_relative_change=round(worst_change, 4),
                rollback_triggered=rollback,
                skipped_defects=skipped,
            )

            if rollback:
                logger.warning(
                    "§PDV rollback: %s worsened %s by %.1f%% (pre=%.4f, post=%.4f) — reverting to pre-Verarbeitungsschritt audio",
                    phase_id,
                    worst_defect,
                    worst_change * 100.0,
                    proxies_before.get(worst_defect, 0.0),
                    proxies_after.get(worst_defect, 0.0),
                )
            elif worst_change > 0.0:
                logger.debug(
                    "§PDV: %s minor drift on %s (+%.1f%%) — below rollback Schwelle, keeping",
                    phase_id,
                    worst_defect,
                    worst_change * 100.0,
                )

            self._record_telemetry(result)

            if metadata_store is not None:
                try:
                    pdv_list = metadata_store.setdefault("phase_defect_verification", [])
                    psycho_before = {
                        "hf_noise_audibility": _compat_helper(
                            "_compute_hf_noise_audibility", _compute_hf_noise_audibility
                        )(audio_before, sr),
                        "transient_harshness": _compat_helper(
                            "_compute_transient_harshness", _compute_transient_harshness
                        )(audio_before, sr),
                        "quasi_peak_burstiness": _compat_helper(
                            "_compute_quasi_peak_burstiness", _compute_quasi_peak_burstiness
                        )(audio_before, sr),
                    }
                    psycho_after = {
                        "hf_noise_audibility": _compat_helper(
                            "_compute_hf_noise_audibility", _compute_hf_noise_audibility
                        )(audio_after, sr),
                        "transient_harshness": _compat_helper(
                            "_compute_transient_harshness", _compute_transient_harshness
                        )(audio_after, sr),
                        "quasi_peak_burstiness": _compat_helper(
                            "_compute_quasi_peak_burstiness", _compute_quasi_peak_burstiness
                        )(audio_after, sr),
                    }
                    pdv_list.append(
                        {
                            "phase_id": phase_id,
                            "targeted_defects": result.targeted_defects,  # type: ignore[attr-defined]
                            "worst_defect": result.worst_defect,  # type: ignore[attr-defined]
                            "worst_relative_change": result.worst_relative_change,  # type: ignore[attr-defined]
                            "rollback": result.rollback_triggered,  # type: ignore[attr-defined]
                            "reweight_applied": _reweight_applied,
                            "reweight_alpha": _reweight_alpha,
                            "reweight_strategy": _reweight_strategy,
                            "goal_aware": True,
                            "psychoacoustic_guard": True,
                            "psychoacoustic_before": psycho_before,
                            "psychoacoustic_after": psycho_after,
                            "goal_target_keys": sorted(goal_targets.keys()) if isinstance(goal_targets, dict) else [],
                        }
                    )
                except Exception as e:
                    logger.warning("Verarbeitungsschritt_defect_verifier.py::unbekannter Ersatzpfad: %s", e)

            return audio_before if rollback else audio_after

        except Exception as exc:
            logger.debug("PDV.Pruefung(%s) fehlgeschlagen: %s — returning audio_after unchanged", phase_id, exc)
            return audio_after

    def _record_telemetry(self, result: DefectVerificationResult) -> None:
        try:
            with self._telem_lock:
                self._session_telemetry.append(
                    {
                        "phase_id": result.phase_id,
                        "worst_defect": result.worst_defect,  # type: ignore[attr-defined]
                        "worst_relative_change": result.worst_relative_change,  # type: ignore[attr-defined]
                        "rollback": result.rollback_triggered,  # type: ignore[attr-defined]
                    }
                )
        except Exception as e:
            logger.warning("Verarbeitungsschritt_defect_verifier.py::_aufzeichnen_telemetry Ersatzpfad: %s", e)

    def get_session_summary(self) -> dict:
        """Gibt a summary of all PDV checks in this session zurück."""
        try:
            with self._telem_lock:
                entries = list(self._session_telemetry)
            rollbacks = [e for e in entries if e.get("rollback")]
            misses = [e for e in entries if e.get("worst_relative_change", 0.0) > 0.0]
            return {
                "total_checked": len(entries),
                "rollback_count": len(rollbacks),
                "miss_count": len(misses),
                "rollback_phases": [e["phase_id"] for e in rollbacks],
                "miss_phases": [e["phase_id"] for e in misses],
            }
        except Exception as e:
            logger.warning("Verarbeitungsschritt_defect_verifier.py::get_Sitzung_summary Ersatzpfad: %s", e)
            return {}

    def reset_session(self) -> None:
        """Setzt zurück: session telemetry (call before each new restoration run)."""
        with self._telem_lock:
            self._session_telemetry.clear()
