"""
§2.35c [RELEASE_MUST] LPC-Formant-Tracker für Shellac-Material — Aurik 10.0.0

Burg-LPC-basierte Formant-Schätzung für schmalbandiges Material (BW ≤ 8 kHz).
Wenn MelBandRoformer / MDX23C / NMF / HPSS fehlschlagen, ist dies der finale
DSP-Fallback für Formant-Enhancement auf Shellac/WaxCylinder.

Ziel: F1–F3 schätzen + leichter Formant-Boosting (max +3 dB) → Vokalklarheit
verbessern ohne Artefakte einzuführen.

§0 Primum non nocere: Kein Eingriff wenn Formant-Schätzung unsicher (< 3 stabile Frames).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

logger = logging.getLogger(__name__)

# Maximaler Boost pro Formant (§2.35c — kein Over-Processing)
_MAX_FORMANT_BOOST_DB = 3.0
# LPC-Ordnung: §4.4 VERBOTEN: LPC < 12 as primary.
# Strategie: Downsampling auf 16 kHz → LPC Ordnung 16 (§4.4 "alternativ: Downsampling
# auf 16 kHz → LPC Ord. 16 → Upsampling"). Bei 16 kHz SR sind F1–F4 (< 8 kHz) sicher
# erfasst; bei 48 kHz nativ bräuchte man Ord. 30–40 — hier nutzen wir den sicheren
# Downsampling-Pfad, da Shellac BW ≤ 8 kHz und Ordnung 16 @ 16 kHz ausreicht.
_LPC_ANALYSIS_SR = 16_000  # Analyse-SR nach Downsampling
_LPC_ORDER = 16  # Ordnung 16 bei 16 kHz (entspricht ~30 bei 48 kHz)
# Analyse-Bandbreite für Shellac (BW ≤ 8 kHz Material-Ceiling)
_SHELLAC_BW_HZ = 7000.0

# §Lücke3 WLPC: SNR-Schwelle — unterhalb davon wird noise-robuster Pfad aktiviert
_WLPC_SNR_THRESHOLD_DB = 15.0
# §Lücke3 WLPC: Ära-Schwelle für historisches Vokal-Material
_WLPC_ERA_THRESHOLD = 1960
# §Lücke3 WLPC: Wiener-Gain-Floor (verhindert Übersubtraktions-Artefakte)
_WLPC_GAIN_FLOOR = 0.10


def _snr_estimate_db(frame_rms_list: list[float]) -> float:
    """§Lücke3 WLPC: Schätzt Signal-Rausch-Abstand aus der Energie-Verteilung der Frames.

    Nutzt den Abstand zwischen Signalpegel (75. Perzentile) und Rauschboden
    (10. Perzentile) als SNR-Proxy — robust gegen kurze Stille-Lücken.

    Returns:
        Geschätzter SNR in dB (0 wenn zu wenig Frames vorhanden).
    """
    if len(frame_rms_list) < 4:
        return 40.0  # Unbekannter SNR → Standard-Pfad annehmen
    arr = np.asarray(frame_rms_list, dtype=np.float64)
    arr = arr[arr > 1e-8]  # Stille-Frames herausfiltern
    if len(arr) < 2:
        return 40.0
    signal_level = float(np.percentile(arr, 75))
    noise_floor = float(np.percentile(arr, 10))
    if noise_floor < 1e-10 or signal_level < 1e-10:
        return 40.0
    return float(20.0 * np.log10(max(signal_level / noise_floor, 1.0)))


def _wlpc_prewhiten_frame(frame: np.ndarray, noise_psd: np.ndarray) -> np.ndarray:
    """§Lücke3 WLPC: Spektrale Vorweißung eines Analyse-Frames für noise-robuste LPC-Schätzung.

    Wiener-Filter im Frequenzbereich: W(f) = max(S(f) - N(f), floor) / (S(f) + eps)
    Die Vorweißung wird NUR für die LPC-Polynomschätzung verwendet —
    das Ausgabe-Audio bleibt unverändert (§0 Primum non nocere).

    Args:
        frame:     Analyse-Frame (float64, Länge NFFT)
        noise_psd: Geschätztes Rausch-PSD (gleiche FFT-Länge wie Hälfte + 1)

    Returns:
        Vorgeweißter Frame (float64, gleiche Länge wie Input)
    """
    n = len(frame)
    nfft = max(64, int(2 ** np.ceil(np.log2(n))))
    # FFT des Frames (mit Hanning-Fenster bereits angewendet vom Aufrufer)
    X = np.fft.rfft(frame, n=nfft)
    signal_psd = np.abs(X) ** 2

    # Längenanpassung noise_psd ↔ signal_psd (durch verschiedene NFFT möglich)
    n_bins = len(signal_psd)
    if len(noise_psd) != n_bins:
        # Lineare Interpolation
        old_bins = np.linspace(0, 1, len(noise_psd))
        new_bins = np.linspace(0, 1, n_bins)
        noise_interp = np.interp(new_bins, old_bins, noise_psd)
    else:
        noise_interp = noise_psd

    # Wiener-Gain: spektrale Übersubtraktions-Methode mit Floor
    gain = np.maximum(signal_psd - noise_interp, _WLPC_GAIN_FLOOR * signal_psd)
    gain = gain / (signal_psd + 1e-12)
    gain = np.clip(gain, _WLPC_GAIN_FLOOR, 1.0)

    # Vorgeweißter Frame: nur Real-Anteil, gleiche Länge wie Input
    X_white = X * gain
    frame_white = np.fft.irfft(X_white, n=nfft)
    return np.asarray(frame_white[:n], dtype=np.float64)  # type: ignore[no-any-return]


def _burg_lpc(x: np.ndarray, order: int) -> np.ndarray:
    """
    Burg-Algorithmus für LPC-Koeffizienten (schnell, numerisch stabil).

    Returns:
        a: LPC-Koeffizienten [1, a1, a2, ..., a_order]
    """
    n = len(x)
    f = x.copy().astype(np.float64)
    b = x.copy().astype(np.float64)

    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    k_coeffs = np.zeros(order, dtype=np.float64)

    for m in range(1, order + 1):
        num = -2.0 * float(np.dot(f[m:], b[: n - m]))
        denom = float(np.dot(f[m:], f[m:]) + np.dot(b[: n - m], b[: n - m])) + 1e-10
        k = num / denom
        k_coeffs[m - 1] = k

        a_new = a.copy()
        for i in range(1, m + 1):
            a_new[i] = a[i] + k * a[m - i]
        a = a_new

        f_new = f[m:] + k * b[: n - m]
        b_new = b[: n - m] + k * f[m:]
        f = f_new
        b = b_new

    return a  # type: ignore[no-any-return]


def _lpc_to_formants(a: np.ndarray, sr: int, max_formants: int = 4) -> list[float]:
    """
    Extrahiert Formantfrequenzen aus LPC-Koeffizienten via Polstellen-Analyse.

    Returns:
        Sortierte Liste von Formantfrequenzen in Hz (nur stimmhafte, F0–max_formants)
    """
    roots = np.roots(a)
    # Nur Wurzeln mit positivem Imaginärteil (eine pro konjugiertem Paar)
    roots = roots[np.imag(roots) > 0.01]
    if len(roots) == 0:
        return []

    angles = np.angle(roots)
    formants_hz = sorted([float(ang * sr / (2.0 * np.pi)) for ang in angles if ang > 0])
    # Filter: nur sinnvolle Vokalformanten (50 Hz – 4500 Hz für Shellac)
    formants_hz = [f for f in formants_hz if 50.0 <= f <= 4500.0]
    return formants_hz[:max_formants]


def _formant_boost_eq(
    audio: np.ndarray,
    sr: int,
    formants_hz: list[float],
    boost_db: float = 2.0,
) -> np.ndarray:
    """
    Sanfter Formant-Boost via Biquad-Peaking-Filter (Bell-EQ) um F1–F3.

    §0 Primum non nocere: boost_db ≤ _MAX_FORMANT_BOOST_DB (3 dB).
    """
    from scipy.signal import sosfiltfilt  # pylint: disable=import-outside-toplevel

    boost_db = float(np.clip(boost_db, 0.0, _MAX_FORMANT_BOOST_DB))
    if boost_db < 0.1 or not formants_hz:
        return audio

    out = audio.copy().astype(np.float64)
    for f_hz in formants_hz[:4]:  # F1–F4 (inkl. Singer's Formant 3–4 kHz, §0p)
        if f_hz <= 0 or f_hz >= sr / 2.0:
            continue
        # Q-Wert: breit genug für Formant-Envelope, eng genug um Nachbar nicht zu berühren
        q = max(1.5, f_hz / 250.0)
        # Biquad Peaking EQ (cookbook: Robert Bristow-Johnson)
        A = 10.0 ** (boost_db / 40.0)
        w0 = 2.0 * np.pi * f_hz / sr
        alpha = np.sin(w0) / (2.0 * q)
        b0 = 1.0 + alpha * A
        b1 = -2.0 * np.cos(w0)
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha / A
        sos_eq = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])
        if out.ndim == 2:
            for ch in range(out.shape[1] if out.shape[0] > 2 else out.shape[0]):
                if out.shape[0] > 2:  # (N, 2)
                    out[:, ch] = sosfiltfilt(sos_eq, out[:, ch])
                else:  # (2, N)
                    out[ch, :] = sosfiltfilt(sos_eq, out[ch, :])
        else:
            out = sosfiltfilt(sos_eq, out)

    return np.clip(out, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]


def lpc_formant_enhance(
    audio: np.ndarray,
    sr: int,
    max_boost_db: float = 2.5,
    frame_len_ms: float = 30.0,
    hop_len_ms: float = 10.0,
    min_voiced_frames: int = 3,
    era_decade: int | None = None,
    snr_hint_db: float | None = None,
) -> np.ndarray[Any, Any]:
    """
    §2.35c LPC-Formant-Enhancement für Shellac (DSP-Fallback).

    Schätzt F1–F3 über mehrere Frames (Burg-LPC Ordnung 12/16), mittelt stabile
    Schätzungen, und boosted diese Frequenzbereiche sanft (max 3 dB).

    §Lücke3 WLPC: Bei era_decade < 1960 oder snr_hint_db < 15 dB wird automatisch
    der noise-robuste WLPC-Pfad (spektrale Vorweißung vor Burg-LPC) aktiviert.
    Die Vorweißung betrifft nur die LPC-Polynomschätzung — das Ausgabe-Audio
    wird unverändert durch _formant_boost_eq bearbeitet (§0 Primum non nocere).

    Args:
        audio:            Mono oder Stereo (float32)
        sr:               Abtastrate (48000 oder downsampled für Analyse)
        max_boost_db:     Maximaler Boost (Default: 2.5 dB, nie > 3.0 dB)
        frame_len_ms:     Framelänge für LPC-Analyse (Default: 30 ms)
        hop_len_ms:       Hop-Länge (Default: 10 ms)
        min_voiced_frames: Mindest-Frames für stabile Formant-Schätzung
        era_decade:       Ära des Materials in Jahrzehnten (z.B. 1940). None = unbekannt.
        snr_hint_db:      Vorgeschätzter SNR in dB. None = automatische Schätzung.

    Returns:
        audio mit Formant-Enhancement (gleiche Form/Länge wie Input)
    """
    audio_in: np.ndarray[Any, Any] = np.asarray(audio, dtype=np.float32)
    if audio_in.ndim == 2:
        mono = np.mean(audio_in, axis=1 if audio_in.shape[0] <= 8 else 0).astype(np.float64)
    else:
        mono = audio_in.astype(np.float64)

    n = len(mono)
    if n < int(0.1 * sr):
        logger.debug("LPC-Formant-Tracker: zu kurz (%.1f s) — übersprungen", n / sr)
        return audio_in

    frame_len = int(frame_len_ms / 1000.0 * sr)
    # hop_len at native SR is not used after downsampling — analysis uses _hop_len_16k

    if frame_len < 64 or n < frame_len:
        return audio_in

    # §4.4 Downsampling auf 16 kHz für LPC-Analyse (Ordnung 16 bei 16 kHz).
    # Shellac BW ≤ 8 kHz → Nyquist bei 16 kHz reicht vollständig aus.
    # Rücktransformation nur für EQ-Anwendung, nicht für das Ausgangssignal.
    try:
        import resampy  # type: ignore[import]  # pylint: disable=import-outside-toplevel

        mono_16k: np.ndarray = resampy.resample(mono, sr, _LPC_ANALYSIS_SR).astype(np.float64)
        _analysis_sr = _LPC_ANALYSIS_SR
    except Exception:
        # Fallback: scipy resample wenn resampy nicht verfügbar
        try:
            from scipy.signal import resample_poly as _rspoly  # pylint: disable=import-outside-toplevel

            _ratio_num = _LPC_ANALYSIS_SR
            _ratio_den = sr
            from math import gcd as _gcd  # pylint: disable=import-outside-toplevel

            _g = _gcd(_ratio_num, _ratio_den)
            mono_16k = _rspoly(mono, _ratio_num // _g, _ratio_den // _g).astype(np.float64)
            _analysis_sr = _LPC_ANALYSIS_SR
        except Exception:
            mono_16k = mono.copy()
            _analysis_sr = sr

    # Frame-Parameter für Analyse-SR
    frame_len_16k = int(frame_len_ms / 1000.0 * _analysis_sr)
    _hop_len_16k = int(hop_len_ms / 1000.0 * _analysis_sr)
    n_16k = len(mono_16k)

    if frame_len_16k < 32 or n_16k < frame_len_16k:
        return audio_in

    # Analyse auf tiefpassgefiltertem Signal (Shellac-BW ≤ 8 kHz) — zero-phase für präzise Formant-Erkennung (§III)
    try:
        from scipy.signal import butter as _butter_lpc

        nyq = min(_SHELLAC_BW_HZ, _analysis_sr / 2.0 - 100.0)
        if nyq > 100.0:
            sos_lp = _butter_lpc(4, nyq, btype="low", fs=_analysis_sr, output="sos")
            mono_lp = sosfiltfilt(sos_lp, mono_16k)
        else:
            mono_lp = mono_16k.copy()
    except Exception:
        mono_lp = mono_16k.copy()

    # Frame-weise LPC + Formant-Extraktion
    # §Lücke3 WLPC: Noise-robuster Pfad für historisches Material oder niedriges SNR
    _frame_rms_list: list[float] = []
    for fi in range(0, n_16k - frame_len_16k, _hop_len_16k):
        _frame_rms_list.append(float(np.sqrt(np.mean(mono_lp[fi : fi + frame_len_16k] ** 2))))

    _use_wlpc = False
    _effective_snr = snr_hint_db if snr_hint_db is not None else _snr_estimate_db(_frame_rms_list)
    if (era_decade is not None and int(era_decade) < _WLPC_ERA_THRESHOLD) or _effective_snr < _WLPC_SNR_THRESHOLD_DB:
        _use_wlpc = True

    # §Lücke3 WLPC: Rausch-PSD aus den 20% leisen Frames schätzen
    _noise_psd: np.ndarray | None = None
    if _use_wlpc and len(_frame_rms_list) >= 4:
        try:
            _rms_arr = np.asarray(_frame_rms_list, dtype=np.float64)
            _noise_threshold_rms = float(np.percentile(_rms_arr[_rms_arr > 1e-8], 20))
            _noise_psds: list[np.ndarray] = []
            for fi in range(0, n_16k - frame_len_16k, _hop_len_16k):
                _nf = mono_lp[fi : fi + frame_len_16k]
                if float(np.sqrt(np.mean(_nf**2))) <= _noise_threshold_rms * 1.5:
                    _nfft_n = max(64, int(2 ** np.ceil(np.log2(frame_len_16k))))
                    _noise_psds.append(np.abs(np.fft.rfft(_nf, n=_nfft_n)) ** 2)
            if _noise_psds:
                _noise_psd = np.mean(np.stack(_noise_psds, axis=0), axis=0)
            logger.debug(
                "§Lücke3 WLPC: era_decade=%s snr=%.1f dB noise_frames=%d",
                era_decade,
                _effective_snr,
                len(_noise_psds),
            )
        except Exception as _wlpc_est_exc:
            logger.debug("§Lücke3 WLPC Rausch-PSD-Schätzung fehlgeschlagen: %s", _wlpc_est_exc)
            _use_wlpc = False

    all_formants: list[list[float]] = []
    for fi in range(0, n_16k - frame_len_16k, _hop_len_16k):
        frame = mono_lp[fi : fi + frame_len_16k]
        # Voiced-Gate: Rahmen mit genug Energie
        rms = float(np.sqrt(np.mean(frame**2)))
        if rms < 1e-4:
            continue
        try:
            windowed = frame * np.hanning(len(frame))
            # §Lücke3 WLPC: Vorgeweißter Frame für LPC-Schätzung bei verrauschtem Material
            if _use_wlpc and _noise_psd is not None:
                lpc_frame = _wlpc_prewhiten_frame(windowed, _noise_psd)
            else:
                lpc_frame = windowed
            a = _burg_lpc(lpc_frame, _LPC_ORDER)
            frms = _lpc_to_formants(a, _analysis_sr)
            if len(frms) >= 2:
                all_formants.append(frms)
        except Exception:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            continue

    if len(all_formants) < min_voiced_frames:
        logger.debug(
            "LPC-Formant-Tracker: zu wenige stabile Frames (%d < %d) — kein Boost",
            len(all_formants),
            min_voiced_frames,
        )
        return audio_in

    # Robuste Mittelung (Median pro Formant-Index)
    max_n_formants = max(len(fs) for fs in all_formants)
    stable_formants: list[float] = []
    for k in range(min(3, max_n_formants)):
        vals = [fs[k] for fs in all_formants if len(fs) > k]
        if len(vals) >= min_voiced_frames:
            stable_formants.append(float(np.median(vals)))

    if not stable_formants:
        return audio_in

    logger.debug(
        "§2.35c LPC-Formant-Tracker: F1-F3 = %s (max_boost=%.1f dB, wlpc=%s)",
        [f"{f:.0f} Hz" for f in stable_formants],
        max_boost_db,
        _use_wlpc,
    )

    return _formant_boost_eq(audio_in, sr, stable_formants, boost_db=min(max_boost_db, _MAX_FORMANT_BOOST_DB))


# ---------------------------------------------------------------------------
# Thread-safe Singleton
# ---------------------------------------------------------------------------


def check_formant_shift_db(
    audio_pre: np.ndarray,
    audio_post: np.ndarray,
    sr: int,
    threshold_db: float = 2.0,
    max_formants: int = 4,
) -> tuple[bool, float]:
    """§G1 (GEBOTE.md)/§G5 (GEBOTE.md) Formant ±2 dB Guard (§0p RELEASE_MUST).

    Compares spectral energy at F1–F4 formant frequencies between pre-NR and
    post-NR audio. Returns (rollback_needed, max_shift_db).

    Uses a ±semitone band (±5.9%) around each detected formant frequency to
    measure energy shift. G_floor for LPC analysis uses a 2 s mid-segment window.
    """
    try:
        # Mono conversion
        def _to_mono(a: np.ndarray) -> np.ndarray:
            a = np.asarray(a, dtype=np.float32)
            if a.ndim == 2:
                result = np.mean(a, axis=0) if a.shape[0] == 2 and a.shape[1] > 2 else np.mean(a, axis=1)
                return np.asarray(result, dtype=np.float32)  # type: ignore[no-any-return]
            return a

        pre_m = _to_mono(audio_pre)
        post_m = _to_mono(audio_post)
        n = min(len(pre_m), len(post_m))
        # §DSP: trim to same length to prevent shape mismatch
        pre_m = pre_m[:n]
        post_m = post_m[:n]
        if n < int(0.1 * sr):
            return False, 0.0  # too short — skip

        # Use 2 s mid-segment for analysis
        win = min(int(2.0 * sr), n)
        mid = n // 2
        pre_seg = pre_m[mid - win // 2 : mid - win // 2 + win].astype(np.float64)
        post_seg = post_m[mid - win // 2 : mid - win // 2 + win].astype(np.float64)

        # Downsample to 16 kHz for LPC formant analysis — Anti-Aliasing-Filter
        # VOR Dezimation verhindert Aliasing von 16–24 kHz auf 0–8 kHz.
        # (resampy in lpc_formant_analyze enthält AA; einfache Stride-Dezimation hier nicht.)
        ds = max(1, sr // 16000)
        _pre_seg_lpc = pre_seg  # Referenz für FFT-Spektralmessung unverändert
        # §Bugfix: ensure equal lengths for anti-aliasing filter
        _min_len = min(len(pre_seg), len(post_seg))
        pre_seg = pre_seg[:_min_len]
        post_seg = post_seg[:_min_len]
        if ds > 1:
            _aa_nyq = sr / (2.0 * ds) * 0.90  # 90 % der Nyquist-Frequenz nach Dezimation
            _aa_sos = butter(4, _aa_nyq, btype="low", fs=sr, output="sos")
            _pre_seg_lpc = sosfiltfilt(_aa_sos, pre_seg.copy())
            sosfiltfilt(_aa_sos, post_seg.copy())
        pre_ds = _pre_seg_lpc[::ds]
        sr_ds = sr // ds

        # Estimate LPC coefficients on the pre-segment and extract formants
        try:
            a = _burg_lpc(pre_ds * np.hanning(len(pre_ds)), _LPC_ORDER)
            formants_hz = _lpc_to_formants(a, sr_ds, max_formants=max_formants)
        except Exception as e:
            logger.debug("lpc_formant_tracker.py::_to_mono lpc: %s", e)
            return False, 0.0
        if not formants_hz:
            return False, 0.0

        # FFT for spectral energy measurement (native sr, 4096 bins)
        n_fft = 4096
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        _pre_buf = pre_seg[:n_fft] if len(pre_seg) >= n_fft else np.pad(pre_seg, (0, n_fft - len(pre_seg)))
        spec_pre = np.abs(np.fft.rfft(_pre_buf)) + 1e-12
        _post_buf = post_seg[:n_fft] if len(post_seg) >= n_fft else np.pad(post_seg, (0, n_fft - len(post_seg)))
        spec_post = np.abs(np.fft.rfft(_post_buf)) + 1e-12

        # Measure energy at each formant band (±semitone ≈ ±5.9%)
        # §V43: Pro-Formant JND-Toleranz (resolve_jnd_tolerance_db) statt uniformem threshold_db.
        # F1 (~600 Hz) → ~1.8 dB; F2 (~1.5 kHz) → ~1.1 dB; F3/F4 (~3 kHz) → ~0.8 dB.
        # min(threshold_db, jnd_tol) → immer das Strengere (Primum non nocere).
        max_shift = 0.0
        rollback = False
        for f_hz in formants_hz:
            if f_hz <= 0 or f_hz >= sr / 2.0:
                continue
            f_lo = f_hz * 0.941  # -1 Halbton
            f_hi = f_hz * 1.059  # +1 Halbton
            band_mask = (freqs >= f_lo) & (freqs <= f_hi)
            if not np.any(band_mask):
                continue
            e_pre = float(np.mean(spec_pre[band_mask] ** 2))
            e_post = float(np.mean(spec_post[band_mask] ** 2))
            if e_pre > 1e-20:
                shift_db = abs(10.0 * np.log10(max(e_post, 1e-20) / e_pre))
                max_shift = max(max_shift, shift_db)
                # §V43: Strengere der beiden Toleranzen — verhindert hörbare F3/F4-Verschiebungen
                _jnd_tol = min(threshold_db, resolve_jnd_tolerance_db(f_hz))
                if shift_db > _jnd_tol:
                    rollback = True

        return rollback, max_shift
    except Exception as e:
        logger.debug("lpc_formant_tracker.py::unknown: %s", e)
        return False, 0.0


class _LPCFormantTracker:
    """Singleton-Wrapper für lpc_formant_enhance."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def enhance(
        self,
        audio: np.ndarray,
        sr: int,
        max_boost_db: float = 2.5,
        era_decade: int | None = None,
        snr_hint_db: float | None = None,
    ) -> np.ndarray:
        """Wendet LPC Formant Enhancement auf audio an (thread-safe).

        §Lücke3 WLPC: era_decade < 1960 oder snr_hint_db < 15 dB aktiviert
        den noise-robusten WLPC-Pfad automatisch.
        """
        with self._lock:
            return lpc_formant_enhance(
                audio,
                sr,
                max_boost_db=max_boost_db,
                era_decade=era_decade,
                snr_hint_db=snr_hint_db,
            )

    def track(self, audio: np.ndarray, sr: int) -> dict:
        """Gibt formant frequencies from LPC analysis zurück.

        Returns dict with keys ``f1_mean``, ``f2_mean``, ``f3_mean``, ``f4_mean``
        (Hz, 0.0 if not detected). Used by phase_03/20/29 formant integrity checks.
        """
        try:
            mono = np.asarray(audio, dtype=np.float32)
            if mono.ndim == 2:
                mono = np.mean(mono, axis=0) if mono.shape[0] == 2 and mono.shape[1] > 2 else np.mean(mono, axis=1)
            if mono.size < 64:
                return {"f1_mean": 0.0, "f2_mean": 0.0, "f3_mean": 0.0, "f4_mean": 0.0}

            # Bound analysis cost: use a centered max-2s window before LPC estimation.
            max_win = min(mono.size, int(max(sr, 1) * 2.0))
            mid = mono.size // 2
            start = max(0, mid - max_win // 2)
            start = min(start, mono.size - max_win)  # §Bugfix: verhindert 1-Sample-Off-by-One
            mono_win = mono[start : start + max_win]

            ds = max(1, sr // 16000)
            _mono_aa = mono_win
            if ds > 1:
                _aa_sos_t = butter(4, (sr / (2.0 * ds)) * 0.90, btype="low", fs=sr, output="sos")
                _mono_aa = sosfiltfilt(_aa_sos_t, mono_win.astype(np.float64))
            mono_ds = _mono_aa[::ds].astype(np.float64)
            sr_ds = max(1, sr // ds)
            if mono_ds.size <= (_LPC_ORDER + 1):
                return {"f1_mean": 0.0, "f2_mean": 0.0, "f3_mean": 0.0, "f4_mean": 0.0}

            windowed = mono_ds * np.hanning(len(mono_ds))
            lpc_a = _burg_lpc(windowed, _LPC_ORDER)
            formants = _lpc_to_formants(lpc_a, sr_ds, max_formants=4)
            keys = ["f1_mean", "f2_mean", "f3_mean", "f4_mean"]
            result = dict.fromkeys(keys, 0.0)
            for i, f in enumerate(formants[:4]):
                result[keys[i]] = float(f)
            return result
        except Exception as e:
            logger.debug("lpc_formant_tracker.py::track AA-filter: %s", e)
            return {"f1_mean": 0.0, "f2_mean": 0.0, "f3_mean": 0.0, "f4_mean": 0.0}

    # ── §2.8 Gender-Classification via Formants ──────────────────────────
    # Wird von phase_19_de_esser._detect_gender_robust() als Fallback
    # aufgerufen, wenn GenderDetector aus vocal_ai_enhancement nicht
    # verfügbar ist oder UNKNOWN liefert.

    # Formant- und F0-Bereiche (identisch mit GenderDetector.formant_ranges,
    # zentral definiert in vocal_ai_enhancement.py §2.8).
    _GENDER_RANGES: dict[str, dict[str, tuple[float, float]]] = {
        "male": {
            "f0": (85, 180),
            "f1": (270, 730),
            "f2": (840, 2290),
            "f3": (1690, 3010),
        },
        "female": {
            "f0": (165, 700),
            "f1": (310, 860),
            "f2": (920, 2790),
            "f3": (1890, 3310),
        },
        "child": {
            "f0": (250, 600),
            "f1": (370, 1030),
            "f2": (1170, 3330),
            "f3": (2590, 4990),
        },
    }

    @staticmethod
    def _scan_f0_voiced(audio_mono: np.ndarray, sr: int) -> float:
        """Scannt durch das Audio (100ms-Fenster, 50ms Hop) bis ein voiced
        Segment mit zuverlässigem F0 gefunden wird.  Vermeidet den
        Fehler, dass ein instrumentales Intro die F0-Detektion blockiert.

        Returns:
            F0 in Hz, oder 0.0 wenn kein voiced segment gefunden wurde.
        """
        chunk_samples = max(int(sr * 0.1), 512)  # 100 ms
        hop_samples = chunk_samples // 2  # 50 ms Überlappung
        max_chunks = min(60, max(1, (len(audio_mono) - chunk_samples) // hop_samples + 1))
        best_f0 = 0.0
        best_peak_height = 0.0
        for chunk_idx in range(max_chunks):
            start = chunk_idx * hop_samples
            if start + chunk_samples > len(audio_mono):
                break
            segment = audio_mono[start : start + chunk_samples].astype(np.float64)
            rms = float(np.sqrt(np.mean(segment**2) + 1e-12))
            if rms < 1e-6:
                continue  # silence
            # FFT-based autocorrelation
            n = len(segment)
            fft = np.fft.rfft(segment, n=2 * n)
            autocorr = np.fft.irfft(fft * np.conj(fft))[:n]
            autocorr = autocorr / (autocorr[0] + 1e-10)
            min_period = int(sr / 500)
            max_period = int(sr / 50)
            if max_period <= min_period or max_period > len(autocorr):
                continue
            autocorr_search = autocorr[min_period:max_period]
            if len(autocorr_search) < 2:
                continue
            peaks, props = find_peaks(autocorr_search, height=0.12)
            if len(peaks) == 0:
                continue
            best_peak = peaks[np.argmax(autocorr_search[peaks])]
            peak_height = float(autocorr_search[best_peak])
            period = best_peak + min_period
            f0 = float(sr) / float(period)
            # Plausibilitätscheck: F0 im Stimmumfang
            if f0 < 70.0 or f0 > 800.0:
                continue
            if peak_height > best_peak_height:
                best_peak_height = peak_height
                best_f0 = f0
        return best_f0

    @staticmethod
    def _estimate_formants_from_voiced(audio_mono: np.ndarray, sr: int, max_frames: int = 40) -> list[float]:
        """Extrahiert mittlere Formanten (F1–F3) aus voiced Frames via
        Burg-LPC.  Scannt das Audio, sammelt voiced Frames und mittelt.
        """
        frame_samples = max(int(sr * 0.025), 64)  # 25 ms
        hop_samples = frame_samples // 2  # 50 % Überlappung
        ds = max(1, sr // _LPC_ANALYSIS_SR)
        sr_ds = max(1, sr // ds)
        all_formants: list[list[float]] = []
        max_iter = min(max_frames * 4, max(1, (len(audio_mono) - frame_samples) // hop_samples + 1))
        collected = 0
        for i in range(max_iter):
            start = i * hop_samples
            if start + frame_samples > len(audio_mono):
                break
            frame = audio_mono[start : start + frame_samples].astype(np.float64)
            rms = float(np.sqrt(np.mean(frame**2) + 1e-12))
            if rms < 1e-6:
                continue
            windowed = frame * np.hanning(len(frame))
            # Downsample für LPC
            _mono_aa = windowed
            if ds > 1:
                _aa_sos = butter(4, (sr / (2.0 * ds)) * 0.90, btype="low", fs=sr, output="sos")
                _mono_aa = sosfiltfilt(_aa_sos, windowed)
            mono_ds = _mono_aa[::ds]
            if mono_ds.size <= (_LPC_ORDER + 1):
                continue
            try:
                lpc_a = _burg_lpc(mono_ds * np.hanning(len(mono_ds)), _LPC_ORDER)
                fmts = _lpc_to_formants(lpc_a, sr_ds, max_formants=3)
                if len(fmts) >= 2 and all(80.0 < f < 5500.0 for f in fmts[:3]):
                    all_formants.append(fmts[:3])
                    collected += 1
                    if collected >= max_frames:
                        break
            except Exception:
                logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                continue
        if not all_formants:
            return []
        max_len = max(len(f) for f in all_formants)
        padded = [np.pad(np.asarray(f), (0, max_len - len(f)), constant_values=0.0) for f in all_formants]
        avg = np.mean(padded, axis=0)
        return [float(v) for v in avg if v > 0.0]

    def classify_gender_via_formants(self, audio: np.ndarray, sr: int) -> str:
        """Klassifiziert das Stimmgeschlecht anhand von F0 + Formanten (LPC).

        Scannt durch das Audio nach voiced Segmenten, extrahiert F0 und
        Formanten (F1–F3) und klassifiziert nach den gleichen Bereichen
        wie GenderDetector in vocal_ai_enhancement.py.

        Args:
            audio: Mono- oder Stereo-Audio (wird automatisch zu Mono gemittelt).
            sr: Sample-Rate in Hz.

        Returns:
            Eines von ``"male"``, ``"female"``, ``"child"``, ``"unknown"``.
        """
        try:
            mono = np.asarray(audio, dtype=np.float64)
            if mono.ndim == 2:
                mono = np.mean(mono, axis=0) if mono.shape[0] == 2 and mono.shape[1] > 2 else np.mean(mono, axis=1)
            if mono.size < int(sr * 0.05):
                return "unknown"

            f0 = self._scan_f0_voiced(mono, sr)
            formants = self._estimate_formants_from_voiced(mono, sr)

            if f0 <= 0.0 and len(formants) < 2:
                return "unknown"

            # Score each gender
            scores: dict[str, float] = {}
            for gender, ranges in self._GENDER_RANGES.items():
                score = 0.0
                count = 0
                # F0 score
                if f0 > 0.0:
                    f0_lo, f0_hi = ranges["f0"]
                    if f0_lo <= f0 <= f0_hi:
                        score += 1.0
                    else:
                        dist = (f0_lo - f0) / f0_lo if f0 < f0_lo else (f0 - f0_hi) / f0_hi
                        score += max(0.0, 1.0 - dist)
                    count += 1
                # Formant scores (F1, F2, F3)
                for fi, fmt_val in enumerate(formants[:3], 1):
                    fk = f"f{fi}"
                    if fk not in ranges:
                        continue
                    flo, fhi = ranges[fk]
                    if flo <= fmt_val <= fhi:
                        score += 1.0
                    else:
                        dist = (flo - fmt_val) / flo if fmt_val < flo else (fmt_val - fhi) / fhi
                        score += max(0.0, 1.0 - dist * 0.5)
                    count += 1
                scores[gender] = score / count if count > 0 else 0.0

            if not scores:
                return "unknown"
            best = max(scores.items(), key=lambda x: x[1])
            if best[1] < 0.4:
                return "unknown"

            # Tie-breaking: CHILD vs FEMALE bei knappem Score und F0 < 350 Hz
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if (
                len(sorted_scores) >= 2
                and sorted_scores[0][0] == "child"
                and sorted_scores[1][0] == "female"
                and (sorted_scores[0][1] - sorted_scores[1][1]) < 0.05
                and f0 < 350.0
            ):
                return "female"
            return best[0]
        except Exception as e:
            logger.debug("lpc_formant_tracker::classify_gender_via_formants: %s", e)
            return "unknown"


_tracker_instance: _LPCFormantTracker | None = None
_tracker_lock = threading.Lock()


def get_lpc_formant_tracker() -> _LPCFormantTracker:
    """Singleton-Zugriff auf den LPC-Formant-Tracker."""
    global _tracker_instance  # pylint: disable=global-statement
    if _tracker_instance is None:
        with _tracker_lock:
            if _tracker_instance is None:
                _tracker_instance = _LPCFormantTracker()
    return _tracker_instance


# ─────────────────────────────────────────────────────────────────────────────
# §JND Frequenzabhängige JND-Toleranz (Gap 4 — §V43)
# ─────────────────────────────────────────────────────────────────────────────
# Stützstellen: Zwicker & Fastl 1999 §4.2 + Moore 2012 §2.4 (Spectral Masking)
# Die JND für Formant-Energie ist NICHT identisch mit der ISO-226-Lautstärke-JND.
# Sie beschreibt die minimale Pegeländerung eines Formant-Peaks, die von einem
# trainierten Hörer bei kurzen Vokal-Segmenten noch detektierbar ist.
# Für F1/F2 ist die Detektion wegen der Vokaldistinktion sensitiver (1.0–1.5 dB)
# als für F3/F4 (0.7–1.0 dB bei 2–4 kHz, höhere Frequenzauflösung des Ohrs).
_JND_BREAKPOINTS_HZ: list[float] = [200.0, 500.0, 1000.0, 2000.0, 4000.0, 6000.0]
_JND_VALUES_DB: list[float] = [3.0, 2.0, 1.5, 1.0, 0.8, 0.7, 0.6]
# Interpolation: unter 200 Hz → 3.0 dB; über 6000 Hz → 0.6 dB


def resolve_jnd_tolerance_db(freq_hz: float) -> float:
    """Frequenzabhängige JND-Toleranz (ISO 226-basiert) für Formant-Energie.

    Gibt den psychoakustisch motivierten Schwellwert zurück, ab dem eine
    Energieänderung eines Formant-Peaks wahrnehmbar ist. Ersetzt den uniformen
    ±1 dB-Wert aus §0p für präzisere Formant-Guards (VERBOTEN V43).

    Stützstellen (Zwicker & Fastl 1999, Moore 2012):
      < 200 Hz:     3.0 dB  (Basalbereich: schlechtere Freq.-Auflösung)
      200–500 Hz:   2.0 dB  (F1-Nähe: Vokal-Identität, erhöhte Sensitivität)
      500–1000 Hz:  1.5 dB  (F1 Hauptregion)
      1000–2000 Hz: 1.0 dB  (F2 Hauptregion: optimale Gehör-Sensitivität)
      2000–4000 Hz: 0.8 dB  (F2–F3: engere kritische Bänder)
      4000–6000 Hz: 0.7 dB  (F3–F4: hohe Frequenzauflösung des Ohrs)
      > 6000 Hz:    0.6 dB  (F4+: sehr enge kritische Bänder)

    Args:
        freq_hz: Frequenz in Hz (z.B. F1-Mittelwert aus LPC-Analyse).

    Returns:
        JND-Toleranz in dB. ±value = Energieänderung < value ist nicht wahrnehmbar.
    """
    freq_hz = float(np.clip(freq_hz, 20.0, 24000.0))

    if freq_hz < _JND_BREAKPOINTS_HZ[0]:
        return float(_JND_VALUES_DB[0])
    if freq_hz >= _JND_BREAKPOINTS_HZ[-1]:
        return float(_JND_VALUES_DB[-1])

    # Lineare Interpolation zwischen Stützstellen
    for i in range(len(_JND_BREAKPOINTS_HZ) - 1):
        f_low = _JND_BREAKPOINTS_HZ[i]
        f_high = _JND_BREAKPOINTS_HZ[i + 1]
        if f_low <= freq_hz < f_high:
            t = (freq_hz - f_low) / (f_high - f_low)
            return float(_JND_VALUES_DB[i] + t * (_JND_VALUES_DB[i + 1] - _JND_VALUES_DB[i]))

    return float(_JND_VALUES_DB[-1])  # Fallback
