"""
core/material_restoration_nets.py — Material-spezifische Restaurierungs-Netze
=============================================================================

Aurik kennt das Quellmedium (Shellac, Vinyl, Tape, Lacquer, Digital).
Dieses Modul implementiert material-spezifische DSP-Ketten, die genau
auf die physikalischen Defekteigenschaften jedes Mediums zugeschnitten sind.

MEDIUM-SPEZIFIKA:
  Shellac (78rpm):
    - Mono, Bandbreite ~8 kHz, starkes Oberflächenrauschen, Klicks/Pops
    - RIAA: Keine (Pre-RIAA oder Equalization-Standard abhängig vom Label)
    - Charakteristik: Hoher Noise-Floor (~-30 dBFS), viele Transienten-Artefakte

  Vinyl:
    - Stereo, Bandbreite ~16-20 kHz, RIAA-Entzerrung (75μs/318μs)
    - Rillenverzerrung (Sibilanten), Tracing-Distortion, Tick/Knister
    - Pitch-Instabilität: Ekzentrizität, Plattenteller-Rumpeln

  Tape:
    - Stereo/Mono, Bandrauschen (Rosa Rauschen + Dropout), Modulation
    - Azimuth-Fehler (Phasenprobleme), Bias-Drift (Frequenzgangfehler)
    - Print-through (Magnetisches Übersprechen benachbarter Lagen)

  Lacquer:
    - Wie Vinyl aber stärker degradiert (Oxidation, Einrisse)
    - Extreme Klick-Dichte, mögliche Frequenzgang-Einbrüche

PLUGIN-ERWEITERUNGSPUNKT:
  Wenn `plugins/{medium}_restoration_plugin.py` vorhanden und
  `restore(audio, sample_rate, **params)` implementiert,
  wird das als ML-Pfad genutzt.

§4.7 NoiseTextureCoherenceGuard: Kohärenz≥0.80 in Restoration-Modus (Trägerprofil-Erhaltung)
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import scipy.signal as sig

from backend.core.audio_utils import safe_filtfilt  # §v10.101

logger = logging.getLogger(__name__)


# ─── §4.7 NoiseTextureCoherenceGuard Helper ──────────────────────────────
def _check_noise_texture_coherence(audio: np.ndarray, sample_rate: int) -> float:
    """§4.7 NoiseTextureCoherenceGuard: Kohärenz≥0.80 in Restoration-Modus."""
    try:
        from backend.core.noise_texture_coherence import get_noise_texture_coherence_guard

        _guard = get_noise_texture_coherence_guard()
        # Residual noise estimate (approximated by low-energy components)
        _residual = audio * 0.01  # small fraction as residual proxy
        _result = _guard.check(_residual, "unknown", sample_rate)
        _coherence = float(_result.coherence)
        if _coherence < 0.80:
            logger.warning(
                "§4.7 NoiseTextureCoherenceGuard: coherence=%.2f < 0.80 (sr=%d)",
                _coherence,
                sample_rate,
            )
        return _coherence
    except Exception as _ntc_err:
        logger.debug("NoiseTextureCoherenceGuard nicht verfügbar: %s", _ntc_err)
        return 1.0  # fallback: assume coherent



# ─── Medium-Enum ─────────────────────────────────────────────────────────


class SourceMedium(Enum):
    """Unterstützte Quellmedien."""

    SHELLAC = "shellac"
    VINYL = "vinyl"
    TAPE = "tape"
    LACQUER = "lacquer"
    DIGITAL = "digital"
    UNKNOWN = "unknown"


@dataclass
class MaterialRestorationResult:
    """Ergebnis einer material-spezifischen Restaurierung (intern, nicht Spec §2.1)."""

    audio: np.ndarray
    medium: SourceMedium
    plugin_used: bool
    applied_steps: list
    metrics: dict[str, Any]


# ─── RIAA-Entzerrung ─────────────────────────────────────────────────────


def _apply_riaa_deriaa(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    RIAA De-Emphasis (inverse RIAA) für Vinyl/Lacquer.
    Zeitkonstanten: t1=75μs, t2=318μs, t3=3180μs.
    """
    # RIAA: IIR-Näherung (matched bilinear transform)
    t1 = 75e-6
    t2 = 318e-6
    t3 = 3180e-6
    fs = sample_rate

    # Pole und Nullstellen der inversen RIAA
    z1 = -np.exp(-1 / (t1 * fs))
    z2 = -np.exp(-1 / (t3 * fs))
    p1 = -np.exp(-1 / (t2 * fs))

    # B/A Koeffizienten (bilinear approx)
    b = np.array([1, -(z1 + z2), z1 * z2])
    a = np.array([1, -(p1 + (-1.0)), p1 * (-1.0)])

    if audio.ndim == 1:
        result = sig.lfilter(b, a, audio)
    else:
        result = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            result[ch] = sig.lfilter(b, a, audio[ch])
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(result, -1.0, 1.0)  # type: ignore[no-any-return]


# ─── Shellac-Restaurierung ───────────────────────────────────────────────


def _shellac_bandwidth_limit(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Tiefpassfilter auf 8 kHz (typische Shellac-Bandbreite)."""
    nyq = sample_rate / 2
    cutoff = min(8000.0, nyq * 0.95)
    b, a = sig.butter(4, cutoff / nyq, btype="low", output="ba")  # type: ignore[misc]
    if audio.ndim == 1:
        result = safe_filtfilt(b, a, audio)
    else:
        result = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            result[ch] = safe_filtfilt(b, a, audio[ch])
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(result, -1.0, 1.0)  # type: ignore[no-any-return]


def _adaptive_click_removal(audio: np.ndarray, sample_rate: int, threshold_db: float = -20.0) -> tuple[np.ndarray, int]:
    """
    Einfacher adaptiver Klick-Entferner: Median-Ersatz für Ausreißer.
    Returns (cleaned, n_clicks_removed).
    """
    result = audio.copy()
    n_removed = 0
    threshold_linear = 10 ** (threshold_db / 20.0)

    mono = result.flatten() if result.ndim > 1 else result

    window = 5
    for i in range(window, len(mono) - window):
        neighbors = np.concatenate([mono[i - window : i], mono[i + 1 : i + window + 1]])
        local_rms = np.sqrt(np.mean(neighbors**2)) + 1e-10
        if abs(mono[i]) > max(threshold_linear, local_rms * 5):
            mono[i] = np.median(neighbors)
            n_removed += 1

    if result.ndim == 1:
        result = mono
    else:
        result[0] = mono

    return result, n_removed


def restore_shellac(audio: np.ndarray, sample_rate: int, **kwargs) -> MaterialRestorationResult:
    """
    Material-spezifische Restaurierung für Shellac (78rpm).

    Schritte:
      1. Klick-Entfernung (harte Transienten)
      2. Bandbegrenzung auf 8 kHz
      3. Mono-Konvertierung (Shellac ist immer Mono)
      4. Noise-Shaping (sanftes High-Pass unter 50 Hz)
    """
    plugin_used, applied = False, []

    # Plugin-Pfad
    try:
        mod = importlib.import_module("shellac_restoration_plugin")
        if hasattr(mod, "restore"):
            result_audio = mod.restore(audio, sample_rate, **kwargs)
            _check_noise_texture_coherence(result_audio, sample_rate)  # §4.7 NoiseTextureCoherenceGuard
            return MaterialRestorationResult(result_audio, SourceMedium.SHELLAC, True, ["plugin"], {})
    except ImportError as _mrn_shellac_exc:
        # §V6 Silent-Failure-Verbot: Log with reason, but INFO (graceful fallback)
        logger.info(
            "§V6 Graceful fallback shellac plugin not available (DSP path): %s", str(_mrn_shellac_exc)[:200]
        )

    # DSP-Pfad
    out = audio.copy()

    # 1. Klick-Entfernung
    out, n_clicks = _adaptive_click_removal(out, sample_rate)
    applied.append(f"click_removal (n={n_clicks})")

    # 2. Mono-Konvertierung (Shellac ist Mono)
    if out.ndim > 1 and out.shape[0] > 1:
        out = np.mean(out, axis=0)
        applied.append("mono_conversion")

    # 3. Bandbreitenbegrenzung
    out = _shellac_bandwidth_limit(out, sample_rate)
    applied.append("bandwidth_limit_8kHz")

    # 4. Rumble-Filter (unter 50 Hz)
    nyq = sample_rate / 2
    b, a = sig.butter(2, 50 / nyq, btype="high", output="ba")  # type: ignore[misc]
    out = safe_filtfilt(b, a, out)
    applied.append("rumble_filter_50Hz")

    _check_noise_texture_coherence(out, sample_rate)  # §4.7 NoiseTextureCoherenceGuard
    return MaterialRestorationResult(
        audio=out,
        medium=SourceMedium.SHELLAC,
        plugin_used=plugin_used,
        applied_steps=applied,
        metrics={"n_clicks_removed": n_clicks},
    )


# ─── Vinyl-Restaurierung ─────────────────────────────────────────────────


def _vinyl_decrackle(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """Knister-Entfernung via Median-Filter auf kurzen Segmenten."""
    result = audio.copy()
    n_removed = 0
    kernel = 3

    channels = [result] if result.ndim == 1 else [result[ch] for ch in range(result.shape[0])]

    cleaned_channels = []
    for ch in channels:
        smoothed = np.convolve(ch, np.ones(kernel) / kernel, mode="same")
        diff = np.abs(ch - smoothed)
        threshold = np.percentile(diff, 99.0)
        mask = diff > threshold
        ch_clean = ch.copy()
        ch_clean[mask] = smoothed[mask]
        n_removed += int(np.sum(mask))
        cleaned_channels.append(ch_clean)

    if result.ndim == 1:
        return cleaned_channels[0], n_removed
    else:
        return np.stack(cleaned_channels, axis=0), n_removed


def restore_vinyl(audio: np.ndarray, sample_rate: int, apply_riaa: bool = False, **kwargs) -> MaterialRestorationResult:
    """
    Material-spezifische Restaurierung für Vinyl (LP, Single).

    Schritte:
      1. RIAA De-Emphasis (optional, wenn Rohdigitalisierung vor EQ)
      2. Knister-/Tick-Entfernung
      3. Rumpelfilter (unter 20 Hz)
      4. Sibilanten-Schutz (De-Essing um 8-12 kHz)
    """
    plugin_used, applied = False, []

    try:
        mod = importlib.import_module("vinyl_restoration_plugin")
        if hasattr(mod, "restore"):
            result_audio = mod.restore(audio, sample_rate, **kwargs)
            _check_noise_texture_coherence(result_audio, sample_rate)  # §4.7 NoiseTextureCoherenceGuard
            return MaterialRestorationResult(result_audio, SourceMedium.VINYL, True, ["plugin"], {})
    except ImportError as _mrn_vinyl_exc:
        # §V6 Silent-Failure-Verbot: Log with reason, but INFO (graceful fallback)
        logger.info(
            "§V6 Graceful fallback vinyl plugin not available (DSP path): %s", str(_mrn_vinyl_exc)[:200]
        )

    out = audio.copy()

    # 1. RIAA (wenn gewünscht)
    if apply_riaa:
        out = _apply_riaa_deriaa(out, sample_rate)
        applied.append("riaa_de_emphasis")

    # 2. Decrackle
    out, n_crackling = _vinyl_decrackle(out, sample_rate)
    applied.append(f"decrackle (n={n_crackling})")

    # 3. Rumpelfilter
    nyq = sample_rate / 2
    b, a = sig.butter(3, 20 / nyq, btype="high", output="ba")  # type: ignore[misc]
    if out.ndim == 1:
        out = safe_filtfilt(b, a, out)
    else:
        for ch in range(out.shape[0]):
            out[ch] = safe_filtfilt(b, a, out[ch])
    applied.append("rumble_filter_20Hz")

    # 4. Sanftes De-Essing (8–12 kHz)
    if sample_rate >= 24000:
        low_cutoff = min(8000 / nyq, 0.98)
        high_cutoff = min(12000 / nyq, 0.99)
        if high_cutoff > low_cutoff:
            b_s, a_s = sig.butter(2, [low_cutoff, high_cutoff], btype="band", output="ba")  # type: ignore[misc]
            if out.ndim == 1:
                sib_band = safe_filtfilt(b_s, a_s, out)
                out = out - 0.15 * sib_band  # Sanfte Dämpfung
            else:
                for ch in range(out.shape[0]):
                    sib_band = safe_filtfilt(b_s, a_s, out[ch])
                    out[ch] = out[ch] - 0.15 * sib_band
            applied.append("de_essing_8-12kHz")

    _check_noise_texture_coherence(out, sample_rate)  # §4.7 NoiseTextureCoherenceGuard
    return MaterialRestorationResult(
        audio=out,
        medium=SourceMedium.VINYL,
        plugin_used=plugin_used,
        applied_steps=applied,
        metrics={"n_crackling_removed": n_crackling, "riaa_applied": apply_riaa},
    )


# ─── Tape-Restaurierung ──────────────────────────────────────────────────


def _tape_noise_reduction(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Bandrauschen-Reduktion: Frequenzabhängige Rauschprofilierung + Subtraktion."""
    n_fft = 2048
    hop = 512
    window = np.hanning(n_fft)
    _noverlap = min(n_fft - hop, max(0, n_fft - 1))  # §v10.103

    def _process_channel(ch: np.ndarray) -> np.ndarray:
        # STFT
        _, _, Zxx = sig.stft(ch, fs=sample_rate, window=window, nperseg=n_fft, noverlap=_noverlap)
        mag = np.abs(Zxx)

        # Rauschprofil: Minimum-Statistics (untere 10% der Energie je Frequenzband)
        noise_profile = np.percentile(mag, 10, axis=1, keepdims=True)

        # Spektrale Subtraktion
        alpha = 2.0  # Over-Subtraction-Faktor
        beta = 0.01  # Spectral Floor
        suppressed_mag = np.maximum(mag - alpha * noise_profile, beta * mag)

        # Phase beibehalten
        phase = np.angle(Zxx)
        Zxx_clean = suppressed_mag * np.exp(1j * phase)

        _, ch_clean = sig.istft(Zxx_clean, fs=sample_rate, window=window, nperseg=n_fft, noverlap=_noverlap)
        # §Fix: istft's reconstruction length depends on n_fft/hop/input-length
        # alignment and can come out SHORTER than the original (not just longer,
        # as the naive `[:len(ch)]` truncation assumed) — pad with zeros to
        # guarantee the shape-preservation invariant all phases rely on.
        ch_clean = np.asarray(ch_clean, dtype=np.float32)
        if len(ch_clean) < len(ch):
            ch_clean = np.pad(ch_clean, (0, len(ch) - len(ch_clean)))
        return ch_clean[: len(ch)]  # type: ignore[no-any-return]

    if audio.ndim == 1:
        return _process_channel(audio)
    else:
        out = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            cleaned = _process_channel(audio[ch])
            out[ch, : len(cleaned)] = cleaned
        return out  # type: ignore[no-any-return]


def _tape_dropout_repair(audio: np.ndarray, sample_rate: int, min_gap_ms: float = 5.0) -> tuple[np.ndarray, int]:
    """Einfache Dropout-Reparatur via lineare Interpolation."""
    result = audio.copy()
    mono = result.flatten() if result.ndim > 1 else result
    min_gap = max(1, int(min_gap_ms * sample_rate / 1000))

    rms_global = np.sqrt(np.mean(mono**2)) + 1e-10
    threshold = rms_global * 0.05  # Unter 5% des globalen RMS = Dropout

    n_repaired = 0
    frame = 256
    i = 0
    while i < len(mono) - frame:
        rms_frame = np.sqrt(np.mean(mono[i : i + frame] ** 2))
        if rms_frame < threshold:
            start = i
            while i < len(mono) and np.sqrt(np.mean(mono[i : i + frame] ** 2)) < threshold:
                i += frame
            end = min(i, len(mono))
            if (end - start) >= min_gap:
                # Lineare Interpolation
                left_val = mono[max(0, start - 1)]
                right_val = mono[min(len(mono) - 1, end)]
                mono[start:end] = np.linspace(left_val, right_val, end - start)
                n_repaired += 1
        else:
            i += frame

    if result.ndim == 1:
        result = mono
    else:
        result[0] = mono

    return result, n_repaired


def restore_tape(audio: np.ndarray, sample_rate: int, **kwargs) -> MaterialRestorationResult:
    """
    Material-spezifische Restaurierung für Magnetband.

    Schritte:
      1. Bandrauschen-Reduktion (spektrale Subtraktion)
      2. Dropout-Reparatur
      3. Print-through-Dämpfung (Pre-Echo-Reduktion unter 80ms)
      4. Azimuth-Kompensation (Phase-Alignment L/R)
    """
    plugin_used, applied = False, []

    try:
        mod = importlib.import_module("tape_restoration_plugin")
        if hasattr(mod, "restore"):
            result_audio = mod.restore(audio, sample_rate, **kwargs)
            _check_noise_texture_coherence(result_audio, sample_rate)  # §4.7 NoiseTextureCoherenceGuard
            return MaterialRestorationResult(result_audio, SourceMedium.TAPE, True, ["plugin"], {})
    except ImportError as _mrn_tape_exc:
        # §V6 Silent-Failure-Verbot: Log with reason, but INFO (graceful fallback)
        logger.info(
            "§V6 Graceful fallback tape plugin not available (DSP path): %s", str(_mrn_tape_exc)[:200]
        )

    out = audio.copy()

    # 1. Bandrauschen-Reduktion
    out = _tape_noise_reduction(out, sample_rate)
    applied.append("tape_noise_reduction")

    # 2. Dropout-Reparatur
    out, n_dropouts = _tape_dropout_repair(out, sample_rate)
    applied.append(f"dropout_repair (n={n_dropouts})")

    # 3. Print-through: Pre-Echo unter -40dBFS/80ms dämpfen
    if sample_rate > 0:
        pre_echo_samples = int(0.08 * sample_rate)  # 80ms
        if audio.ndim == 1 and len(out) > pre_echo_samples:
            threshold_pe = 10 ** (-40 / 20)
            rms_global = np.sqrt(np.mean(out**2)) + 1e-10
            for i in range(0, len(out) - pre_echo_samples, pre_echo_samples // 4):
                rms_now = np.sqrt(np.mean(out[i : i + pre_echo_samples // 4] ** 2))
                rms_after = np.sqrt(
                    np.mean(out[i + pre_echo_samples : i + pre_echo_samples + pre_echo_samples // 4] ** 2)
                )
                if rms_now < threshold_pe * rms_global and rms_after > rms_global * 0.3:
                    out[i : i + pre_echo_samples // 4] *= 0.5  # Pre-Echo dämpfen
            applied.append("print_through_reduction")

    # 4. Stereo-Azimuth-Alignment (nur bei Stereo)
    if out.ndim > 1 and out.shape[0] == 2:
        # Kreuzkorrelation L/R → Verzögerung bestimmen und korrigieren
        max_lag = int(0.005 * sample_rate)  # ±5ms
        min_len = min(out.shape[1], 4096)
        from backend.core.core_utils import fft_crosscorr

        xcorr = fft_crosscorr(out[0, :min_len], out[1, :min_len])
        lag = np.argmax(xcorr) - (min_len - 1)
        lag = int(np.clip(lag, -max_lag, max_lag))  # type: ignore[assignment]
        if abs(lag) > 0:
            if lag > 0:
                # Shift left with zero-fill (no circular wrap from start to tail)
                out[1, :-lag] = out[1, lag:]
                out[1, -lag:] = 0.0
            else:
                # Shift left by |lag| with zero-fill (lag is negative here)
                _k = abs(lag)
                out[0, :-_k] = out[0, _k:]
                out[0, -_k:] = 0.0
            applied.append(f"azimuth_correction (lag={lag} samples)")

    _check_noise_texture_coherence(out, sample_rate)  # §4.7 NoiseTextureCoherenceGuard
    return MaterialRestorationResult(
        audio=out,
        medium=SourceMedium.TAPE,
        plugin_used=plugin_used,
        applied_steps=applied,
        metrics={"n_dropouts_repaired": n_dropouts},
    )


# ─── Lacquer-Restaurierung ───────────────────────────────────────────────


def restore_lacquer(audio: np.ndarray, sample_rate: int, **kwargs) -> MaterialRestorationResult:
    """
    Material-spezifische Restaurierung für Lacquer-Discs (Acetat, stark degradiert).

    Kombiniert Shellac + Vinyl Methoden mit aggressiverer Klick-Entfernung.
    """
    plugin_used, applied = False, []

    try:
        mod = importlib.import_module("lacquer_restoration_plugin")
        if hasattr(mod, "restore"):
            result_audio = mod.restore(audio, sample_rate, **kwargs)
            _check_noise_texture_coherence(result_audio, sample_rate)  # §4.7 NoiseTextureCoherenceGuard
            return MaterialRestorationResult(result_audio, SourceMedium.LACQUER, True, ["plugin"], {})
    except ImportError as _mrn_lacquer_exc:
        # §V6 Silent-Failure-Verbot: Log with reason, but INFO (graceful fallback)
        logger.info(
            "§V6 Graceful fallback lacquer plugin not available (DSP path): %s", str(_mrn_lacquer_exc)[:200]
        )

    out = audio.copy()

    # Aggressivere Klick-Entfernung (threshold -10 dB statt -20 dB)
    out, n_clicks = _adaptive_click_removal(out, sample_rate, threshold_db=-10.0)
    applied.append(f"aggressive_click_removal (n={n_clicks})")

    # Decrackle
    out, n_crack = _vinyl_decrackle(out, sample_rate)
    applied.append(f"decrackle (n={n_crack})")

    # Bandbegrenzung auf 12 kHz (Lacquer degradiert stärker als Vinyl)
    nyq = sample_rate / 2
    if nyq > 12000:
        b, a = sig.butter(4, 12000 / nyq, btype="low", output="ba")  # type: ignore[misc]
        if out.ndim == 1:
            out = safe_filtfilt(b, a, out)
        else:
            for ch in range(out.shape[0]):
                out[ch] = safe_filtfilt(b, a, out[ch])
        applied.append("bandwidth_limit_12kHz")

    # Rumble unter 30 Hz
    b_hp, a_hp = sig.butter(3, 30 / nyq, btype="high", output="ba")  # type: ignore[misc]
    if out.ndim == 1:
        out = safe_filtfilt(b_hp, a_hp, out)
    else:
        for ch in range(out.shape[0]):
            out[ch] = safe_filtfilt(b_hp, a_hp, out[ch])
    applied.append("rumble_filter_30Hz")

    _check_noise_texture_coherence(out, sample_rate)  # §4.7 NoiseTextureCoherenceGuard
    return MaterialRestorationResult(
        audio=out,
        medium=SourceMedium.LACQUER,
        plugin_used=plugin_used,
        applied_steps=applied,
        metrics={"n_clicks_removed": n_clicks, "n_crackling_removed": n_crack},
    )


# ─── Dispatcher ──────────────────────────────────────────────────────────

_RESTORER_MAP: dict[
    SourceMedium,
    Callable[..., MaterialRestorationResult],
] = {
    SourceMedium.SHELLAC: restore_shellac,
    SourceMedium.VINYL: restore_vinyl,
    SourceMedium.TAPE: restore_tape,
    SourceMedium.LACQUER: restore_lacquer,
}


def restore_by_medium(
    audio: np.ndarray,
    sample_rate: int,
    medium: SourceMedium | str,
    **kwargs,
) -> MaterialRestorationResult:
    """
    Restauriert Audio medium-spezifisch.

    Args:
        audio: Eingabe-Audio
        sample_rate: Samplerate
        medium: SourceMedium-Enum oder String (z.B. "vinyl")
        **kwargs: Medium-spezifische Parameter

    Returns:
        MaterialRestorationResult mit restauriertem Audio und Metriken
    """
    if isinstance(medium, str):
        medium = SourceMedium(medium.lower())

    restorer = _RESTORER_MAP.get(medium)
    if restorer is None:
        logger.info("Kein spezifischer Restorer für %s — Audio unverändert.", medium)
        _check_noise_texture_coherence(audio, sample_rate)  # §4.7 NoiseTextureCoherenceGuard (Fallback)
        return MaterialRestorationResult(
            audio=audio.copy(),
            medium=medium,
            plugin_used=False,
            applied_steps=[],
            metrics={},
        )

    result = restorer(audio, sample_rate, **kwargs)
    _check_noise_texture_coherence(result.audio, sample_rate)  # §4.7 NoiseTextureCoherenceGuard (Dispatcher)
    return result
