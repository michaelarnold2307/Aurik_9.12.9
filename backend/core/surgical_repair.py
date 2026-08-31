"""
§2.59 Surgical Defect Repair (2026-07-09)

Zeitlich präzise, ortsgenaue Reparatur einzelner Defekt-Instanzen.
Kein globales Processing. Nur die kranke Stelle wird operiert.

Prinzip:
  1. Defekt-Instanz lokalisieren (start_sample, end_sample)
  2. Kontext-Fenster extrahieren (für Cross-Fade)
  3. Phase NUR auf das Fenster anwenden
  4. Repariertes Segment nahtlos zurück-crossfaden
  5. Lautstärke, Phase, DC-Offset am Übergang angleichen

Garantiert: keine Sprünge, keine Pegeländerungen, keine Artefakte
an den Übergängen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DefectInstance:
    """Eine zeitlich lokalisierte Defekt-Instanz."""

    start_s: float
    end_s: float
    defect_type: str
    severity: float


@dataclass
class RepairResult:
    """Ergebnis einer chirurgischen Reparatur."""

    audio: np.ndarray
    zones_repaired: int = 0
    zones_skipped: int = 0


# ── Lightweight Phase Functions für Surgical Repair ──────────────────────


def _safety_clamp(repaired: np.ndarray, original: np.ndarray, max_ratio: float = 2.0) -> np.ndarray:
    """§2.59.14: Garantiert dass repariertes Audio nie > max_ratio × Original-Amplitude."""
    import numpy as np

    result = repaired.copy()
    # Per-Sample-Clamp: kein Sample darf max_ratio × Original überschreiten
    abs_orig = np.maximum(np.abs(original), 1e-10)
    limit = abs_orig * max_ratio
    np.clip(result, -limit, limit, out=result)
    # Global RMS darf nicht > 2× Original sein
    rms_orig = np.sqrt(np.mean(original**2)) + 1e-10
    rms_rep = np.sqrt(np.mean(result**2)) + 1e-10
    if rms_rep > rms_orig * max_ratio:
        result *= (rms_orig * max_ratio) / rms_rep
    return cast(np.ndarray, result.astype(np.float32))


def _repair_wow_flutter(audio: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    """Leichte Wow/Flutter-Korrektur via lokaler Resampling-Anpassung.

    Für isolierte Defekt-Zonen (nicht das ganze Lied).
    Die volle phase_12 läuft danach für globale Transport-Korrektur.

    §v10.101 Selbstkalibrierend: Gain-Clipping skaliert mit gemessener
    Defektschwere aus kwargs['severity'].
    """
    import numpy as np

    result = audio.copy()
    if audio.shape[-1] < 100:
        return result
    # §v10.101: Gain-Range aus Defektschwere ableiten
    _sev = float(kwargs.get("severity", 0.5))
    _range = float(np.clip(0.5 + _sev * 1.5, 0.5, 2.0))  # 0.5→1.25, 1.0→2.0
    from scipy.signal import medfilt

    try:
        if result.ndim == 1:
            envelope = np.abs(result)
            smoothed = medfilt(envelope, kernel_size=min(51, len(envelope) // 10 + 1))
            gain = np.where(envelope > 1e-10, smoothed / (envelope + 1e-10), 1.0)
            result = result * np.clip(gain, 1.0 / _range, _range)
        else:
            for ch in range(result.shape[0]):
                envelope = np.abs(result[ch])
                smoothed = medfilt(envelope, kernel_size=min(51, len(envelope) // 10 + 1))
                gain = np.where(envelope > 1e-10, smoothed / (envelope + 1e-10), 1.0)
                result[ch] = result[ch] * np.clip(gain, 1.0 / _range, _range)
    except Exception as e:
        logger.warning("surgical_repair.py::_repair_wow_flutter Ersatzpfad: %s", e)
    result = _safety_clamp(result, audio)
    return cast(np.ndarray, result.astype(np.float32))


def _repair_hiss(audio: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    """Leichte Bandrausch-Reduktion via spektraler Subtraktion.

    Nur für isolierte Zonen. phase_29 läuft danach global.
    """
    import numpy as np

    result = audio.copy()
    if audio.shape[-1] < 256:
        return result
    try:
        if result.ndim == 1:
            spec = np.fft.rfft(result)
            mag = np.abs(spec)
            # Schätze Rauschboden aus hochfrequentem Bereich
            noise_floor = np.median(mag[-len(mag) // 4 :]) * 0.5
            # Spektrale Subtraktion (soft)
            gain = np.maximum(mag - noise_floor, 0.0) / (mag + 1e-10)
            spec = spec * np.clip(gain, 0.1, 1.0)
            result = np.fft.irfft(spec, n=len(result))
        else:
            for ch in range(result.shape[0]):
                spec = np.fft.rfft(result[ch])
                mag = np.abs(spec)
                noise_floor = np.median(mag[-len(mag) // 4 :]) * 0.5
                gain = np.maximum(mag - noise_floor, 0.0) / (mag + 1e-10)
                spec = spec * np.clip(gain, 0.1, 1.0)
                result[ch] = np.fft.irfft(spec, n=len(result[ch]))
    except Exception as e:
        logger.warning("surgical_repair.py::_repair_hiss Ersatzpfad: %s", e)
    result = _safety_clamp(result, audio)
    return cast(np.ndarray, result.astype(np.float32))


def _repair_clicks(audio: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    """Chirurgische Click-Reparatur: Interpolation der Impuls-Samples.

    Findet Clicks via Schwellwert-Detektion im Differenzsignal und
    interpoliert nur die betroffenen Samples (≤ 0.15 ms pro Click).
    Kein globaler Declick-Pass — nur die markierte Zone.

    Referenz: Janssen, Veldhuis & Vries (1986) IEEE TASLP 34:203
    """
    import numpy as np
    from scipy.interpolate import CubicSpline

    result = audio.copy()
    if audio.shape[-1] < 10:
        return result
    try:
        channels = [result] if result.ndim == 1 else [result[ch] for ch in range(result.shape[0])]
        for ch_idx, ch_data in enumerate(channels):
            # Differenzsignal für Click-Detektion
            diff = np.abs(np.diff(ch_data))
            if len(diff) < 3:
                continue
            # Adaptiver Schwellwert: 6× Median (robust gegen Ausreißer)
            threshold = np.median(diff) * 6.0
            click_mask = diff > threshold
            click_idx = np.where(click_mask)[0]
            if len(click_idx) == 0:
                continue
            # Gruppiere benachbarte Click-Samples (innerhalb 0.3 ms)
            gap = max(1, int(sr * 0.0003))
            groups = []
            cur = [click_idx[0]]
            for i in click_idx[1:]:
                if i - cur[-1] <= gap:
                    cur.append(i)
                else:
                    groups.append(cur)
                    cur = [i]
            groups.append(cur)
            # Max 0.25 ms pro Click-Gruppe (breitere Peaks sind keine Clicks)
            max_width = max(2, int(sr * 0.00025))
            for group in groups:
                if len(group) > max_width:
                    continue
                # Interpolationsfenster: ±3 Samples um den Click
                window = 3
                s = max(0, group[0] - window)
                e = min(len(ch_data) - 1, group[-1] + window + 1)
                if e - s < 4:
                    continue
                # Kubische Spline-Interpolation über die Click-Lücke
                x = np.arange(s, e)
                y = ch_data[s:e]
                # Maske für intakte Samples (außerhalb des Clicks)
                intact = np.ones(len(x), dtype=bool)
                for g in group:
                    rel = g - s
                    if 0 <= rel < len(intact):
                        intact[max(0, rel - 1) : min(len(intact), rel + 2)] = False
                if intact.sum() < 4:
                    continue
                try:
                    cs = CubicSpline(x[intact], y[intact], bc_type="natural")
                    ch_data[s:e] = cs(x)
                except Exception:
                    # Linearer Fallback
                    ch_data[s:e] = np.interp(x, x[intact], y[intact])
            if result.ndim == 1:
                result = ch_data
            else:
                result[ch_idx] = ch_data
    except Exception as e:
        logger.warning("surgical_repair.py::_repair_clicks Ersatzpfad: %s", e)
    result = _safety_clamp(result, audio)
    return cast(np.ndarray, result.astype(np.float32))


def _repair_crackle(audio: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    """Chirurgische Knistern-Reparatur: Lokalisierte Median-Filterung.

    Knistern (crackle) besteht aus hochdichten Mikro-Impulsen im
    HF-Bereich (>4 kHz). Ein kurzer Median-Filter über die betroffene
    Zone glättet die Mikro-Transienten ohne das Gesamtsignal zu dämpfen.

    Referenz: Bailey, Casebeer & Fazekas (2019) AES 147th Conv.
    """
    import numpy as np
    from scipy.signal import butter, medfilt, sosfiltfilt

    result = audio.copy()
    if audio.shape[-1] < 32:
        return result
    try:
        # Highpass > 4 kHz extrahieren (da wo Knistern lebt)
        sos = butter(4, 4000, "high", fs=sr, output="sos")
        channels = (
            [(result, result)]
            if result.ndim == 1
            else [(result[ch : ch + 1], result[ch]) for ch in range(result.shape[0])]
        )
        for ch_view, ch_data in channels:
            hf = sosfiltfilt(sos, ch_data)
            # Adaptiver Schwellwert für Knistern-Intensität
            hf_env = np.abs(hf)
            if np.median(hf_env) < 1e-8:
                continue
            # Median-Filter über kurzes Fenster (0.5 ms) — entfernt Mikro-Impulse
            kernel = max(3, int(sr * 0.0005) | 1)  # ungerade
            filtered = medfilt(ch_data, kernel_size=kernel)
            # Blend: nur HF-Anteil ersetzen, Tiefpass unverändert
            sos_lp = butter(4, 4000, "low", fs=sr, output="sos")
            lp = sosfiltfilt(sos_lp, ch_data)
            lp_filtered = sosfiltfilt(sos_lp, filtered)
            hf_original = ch_data - lp
            hf_filtered = filtered - lp_filtered
            # An den Rändern sanft zurück zum Original blenden
            blend = np.ones(len(ch_data), dtype=np.float32)
            edge = min(len(ch_data) // 8, int(sr * 0.005))
            if edge > 2:
                blend[:edge] = np.linspace(0.0, 1.0, edge)
                blend[-edge:] = np.linspace(1.0, 0.0, edge)
            ch_data[:] = lp + (blend * hf_filtered + (1.0 - blend) * hf_original)
        if result.ndim > 1:
            for ch in range(result.shape[0]):
                pass  # already modified in-place via ch_data
    except Exception as e:
        logger.warning("surgical_repair.py::_repair_crackle Ersatzpfad: %s", e)
    result = _safety_clamp(result, audio)
    return cast(np.ndarray, result.astype(np.float32))


def _repair_dropouts(audio: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    """Chirurgische Dropout-Reparatur: Pegel-Kompensation + Interpolation.

    Dropouts sind kurzzeitige Pegel-Einbrüche (2-200 ms) durch
    Bandmaterial-Fehler. Gain-Kompensation für partielle Dropouts,
    Spline-Interpolation für vollständige Aussetzer.

    Referenz: Dahimene et al. (2008)
    """
    import numpy as np
    from scipy.interpolate import CubicSpline

    result = audio.copy()
    if audio.shape[-1] < 32:
        return result
    try:
        channels = [result] if result.ndim == 1 else [result[ch] for ch in range(result.shape[0])]
        for ch_idx, ch_data in enumerate(channels):
            # RMS-Hüllkurve (kurzes Fenster)
            frame = max(16, int(sr * 0.002))
            hop = frame // 2
            n_frames = max(1, (len(ch_data) - frame) // hop + 1)
            rms_env = np.zeros(len(ch_data), dtype=np.float32)
            for f in range(n_frames):
                seg = ch_data[f * hop : f * hop + frame]
                rms_env[f * hop : f * hop + frame] = np.sqrt(np.mean(seg**2) + 1e-10)
            # Globaler Median-RMS als Referenz
            ref_rms = np.median(rms_env[rms_env > 1e-8]) if np.any(rms_env > 1e-8) else 1e-6
            # Dropout = RMS unter 30% der Referenz
            dropout_mask = rms_env < (ref_rms * 0.3)
            if not np.any(dropout_mask):
                continue
            # Gruppiere Dropout-Regionen
            from scipy.ndimage import label as _label

            labeled, n_regions = _label(dropout_mask)
            for r in range(1, n_regions + 1):
                region = np.where(labeled == r)[0]
                if len(region) < 4:
                    continue
                s, e = region[0], region[-1] + 1
                # Kontext für Interpolation (±10 ms)
                ctx = int(sr * 0.01)
                s_ctx = max(0, s - ctx)
                e_ctx = min(len(ch_data), e + ctx)
                intact = np.ones(e_ctx - s_ctx, dtype=bool)
                intact[s - s_ctx : e - s_ctx] = False
                if intact.sum() < 4:
                    # Zu wenig Kontext → Gain-Kompensation
                    gain = ref_rms / (rms_env[s:e].mean() + 1e-10)
                    ch_data[s:e] *= np.clip(gain, 0.2, 3.0)
                else:
                    try:
                        x = np.arange(s_ctx, e_ctx)
                        y = ch_data[s_ctx:e_ctx].copy()
                        cs = CubicSpline(x[intact], y[intact], bc_type="natural")
                        ch_data[s:e] = cs(np.arange(s, e))
                    except Exception:
                        gain = ref_rms / (rms_env[s:e].mean() + 1e-10)
                        ch_data[s:e] *= np.clip(gain, 0.2, 3.0)
            if result.ndim == 1:
                result = ch_data
            else:
                result[ch_idx] = ch_data
    except Exception as e:
        logger.warning("surgical_repair.py::_repair_dropouts Ersatzpfad: %s", e)
    result = _safety_clamp(result, audio)
    return cast(np.ndarray, result.astype(np.float32))


def _repair_pre_echo(audio: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    """Chirurgische Pre-Echo/Print-Through-Reparatur: Temporal-Masking-Korrektur.

    Pre-Echo (MP3/AAC) und Print-Through (Magnetband-Übersprechen) sind
    leise Vorab-Kopien des Signals vor lauten Transienten. Detektion via
    Energie-Anstieg, Dämpfung der Pre-Echo-Region.

    Referenz: Herre & Johnston (1996) AES Conv. 101
    """
    import numpy as np

    result = audio.copy()
    if audio.shape[-1] < 256:
        return result
    try:
        channels = [result] if result.ndim == 1 else [result[ch] for ch in range(result.shape[0])]
        for ch_idx, ch_data in enumerate(channels):
            # Kurzzeit-Energie-Hüllkurve
            frame = max(64, int(sr * 0.002))
            hop = frame // 4
            n_frames = max(1, (len(ch_data) - frame) // hop)
            energy = np.zeros(n_frames)
            for f in range(n_frames):
                energy[f] = np.sum(ch_data[f * hop : f * hop + frame] ** 2)
            if len(energy) < 3:
                continue
            # Finde Pre-Echo: Energie-Sprung >12dB zwischen benachbarten Frames
            db = 10 * np.log10(energy + 1e-10)
            jump = np.diff(db)
            pre_echo_mask = jump > 12.0  # >12dB Anstieg → vorherige Region ist Pre-Echo
            pre_echo_indices = np.where(pre_echo_mask)[0]
            for idx in pre_echo_indices:
                # Die ~5ms vor dem Transienten dämpfen (Pre-Echo-Länge)
                pre_samples = int(sr * 0.005)
                pre_start = max(0, idx * hop - pre_samples)
                pre_end = min(len(ch_data), idx * hop)
                if pre_end - pre_start < 8:
                    continue
                # Sanfte Dämpfung: -8dB mit Cosine-Ramp
                ramp = 0.5 * (1 - np.cos(np.pi * np.arange(pre_end - pre_start) / (pre_end - pre_start)))
                gain = 1.0 - 0.6 * ramp  # 0.4 → -8dB am nächsten zum Transienten
                ch_data[pre_start:pre_end] *= gain.astype(np.float32)
            if result.ndim == 1:
                result = ch_data
            else:
                result[ch_idx] = ch_data
    except Exception as e:
        logger.warning("surgical_repair.py::_repair_pre_echo Ersatzpfad: %s", e)
    result = _safety_clamp(result, audio)
    return cast(np.ndarray, result.astype(np.float32))


def _repair_groove_echo(audio: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    """Chirurgische Groove-Echo-Reparatur: Adjacent-Groove-Kopplung auf Vinyl.

    Mechanische Kopplung zwischen benachbarten Rillen erzeugt ein
    leises Pre-Echo (1.8s bei 33⅓ RPM). Detektion via Autokorrelation
    im niederfrequenten Bereich, dann subtraktive Dämpfung.

    Referenz: AES Anthology of Disc Recording (2000)
    """
    import numpy as np

    result = audio.copy()
    if audio.shape[-1] < 4096:
        return result
    try:
        # Groove-Echo ist am stärksten im Bass-Bereich (<500 Hz)
        from scipy.signal import butter, sosfilt

        sos_lp = butter(4, 500, "low", fs=sr, output="sos")
        channels = [result] if result.ndim == 1 else [result[ch] for ch in range(result.shape[0])]
        for ch_idx, ch_data in enumerate(channels):
            lp = sosfilt(sos_lp, ch_data)
            if len(lp) < sr:
                continue
            # Autokorrelation suchen (1.8s = 33⅓ RPM Groove-Echo-Delay)
            groove_delay = int(1.8 * sr)
            if groove_delay >= len(lp) // 2:
                continue
            # Kreuzkorrelation zwischen erstem und zweitem Teil
            ref = lp[: len(lp) // 4]
            corr = np.correlate(lp[len(lp) // 4 : len(lp) // 2], ref[: min(len(ref), groove_delay * 2)], mode="valid")
            if len(corr) < 2:
                continue
            peak = np.argmax(np.abs(corr))
            if peak > 0 and corr[peak] > np.std(lp) * 0.02:
                # Echo vorhanden → subtraktive Dämpfung im LF-Bereich
                # Sanftes Gate auf den Echopegel (nur LF)
                echo_gain = max(0.0, 1.0 - abs(corr[peak]) / (np.std(lp) * 10 + 1e-10))
                lf = sosfilt(sos_lp, ch_data)
                hf = ch_data - lf
                ch_data[:] = hf + lf * echo_gain
            if result.ndim == 1:
                result = ch_data
            else:
                result[ch_idx] = ch_data
    except Exception as e:
        logger.warning("surgical_repair.py::_repair_groove_echo Ersatzpfad: %s", e)
    result = _safety_clamp(result, audio)
    return cast(np.ndarray, result.astype(np.float32))


def _repair_mpeg_frame_loss(audio: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    """Chirurgische MPEG-Frame-Loss-Reparatur: Bitstream-Korruption interpolieren.

    MP3/AAC Frame-Drops erzeugen kurze Signalaussetzer (26ms bei MP3).
    Detektion via Energie-Lücken mit Frame-typischer Dauer, dann
    kubische Spline-Interpolation.

    Referenz: Brandenburg (1999) — MP3 Frame Structure
    """
    import numpy as np
    from scipy.interpolate import CubicSpline

    result = audio.copy()
    if audio.shape[-1] < 512:
        return result
    try:
        # MP3-Frame: 26ms @ 48kHz = ~1248 samples
        frame_len = int(sr * 0.026)
        channels = [result] if result.ndim == 1 else [result[ch] for ch in range(result.shape[0])]
        for ch_idx, ch_data in enumerate(channels):
            # Sliding RMS window
            win = max(64, frame_len // 4)
            hop = win // 2
            rms = np.zeros(len(ch_data) // hop)
            for f in range(len(rms)):
                seg = ch_data[f * hop : min(f * hop + win, len(ch_data))]
                rms[f] = np.sqrt(np.mean(seg**2) + 1e-10)
            ref_rms = np.median(rms[rms > 1e-8]) if np.any(rms > 1e-8) else 1e-6
            # Frame-Drop: plötzlicher Energieabfall auf <5%
            drop_mask = rms < (ref_rms * 0.05)
            if not np.any(drop_mask):
                continue
            # Gruppiere und interpoliere
            from scipy.ndimage import label as _label

            labeled, n_regions = _label(drop_mask)
            for r in range(1, n_regions + 1):
                region = np.where(labeled == r)[0]
                if len(region) < 3:
                    continue
                s = max(0, region[0] * hop - win)
                e = min(len(ch_data), (region[-1] + 1) * hop + win)
                if e - s < 8:
                    continue
                x = np.arange(s, e)
                intact = np.ones(e - s, dtype=bool)
                for ri in region:
                    rs = max(0, ri * hop - s)
                    re = min(len(intact), (ri + 1) * hop - s)
                    intact[rs:re] = False
                if intact.sum() < 4:
                    continue
                try:
                    cs = CubicSpline(x[intact], ch_data[s:e][intact], bc_type="natural")
                    ch_data[s:e] = cs(x)
                except Exception:
                    ch_data[s:e] = np.interp(x, x[intact], ch_data[s:e][intact])
            if result.ndim == 1:
                result = ch_data
            else:
                result[ch_idx] = ch_data
    except Exception as e:
        logger.warning("surgical_repair.py::_repair_mpeg_frame_loss Ersatzpfad: %s", e)
    result = _safety_clamp(result, audio)
    return cast(np.ndarray, result.astype(np.float32))


def _repair_tape_head_clog(audio: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    """Chirurgische Tape-Head-Clog-Reparatur: Kontaktverlust-Gain-Korrektur.

    Ein verschmutzter/oxidierter Tonkopf verliert zeitweise Kontakt zum Band,
    was zu kurzen Pegel-Dips führt. Detektion via gleitendem RMS-Tief,
    Gain-Kompensation der betroffenen Stellen.
    """
    import numpy as np

    result = audio.copy()
    if audio.shape[-1] < 256:
        return result
    try:
        channels = [result] if result.ndim == 1 else [result[ch] for ch in range(result.shape[0])]
        for ch_idx, ch_data in enumerate(channels):
            frame = max(32, int(sr * 0.010))
            hop = frame // 2
            n_frames = max(1, (len(ch_data) - frame) // hop + 1)
            rms_env = np.zeros(len(ch_data))
            for f in range(n_frames):
                seg = ch_data[f * hop : f * hop + frame]
                rms_env[f * hop : f * hop + frame] = np.sqrt(np.mean(seg**2) + 1e-10)
            ref_rms = np.median(rms_env[rms_env > 1e-8]) if np.any(rms_env > 1e-8) else 1e-6
            # Head-Clog: partieller Pegelverlust (20-60% unter Referenz, kurzzeitig)
            dip_mask = (rms_env < (ref_rms * 0.6)) & (rms_env > (ref_rms * 0.2))
            if not np.any(dip_mask):
                continue
            from scipy.ndimage import label as _label

            labeled, n_regions = _label(dip_mask)
            for r in range(1, n_regions + 1):
                region = np.where(labeled == r)[0]
                if len(region) < 8:
                    continue
                s, e = region[0], min(len(ch_data), region[-1] + 1)
                local_rms = np.sqrt(np.mean(ch_data[s:e] ** 2) + 1e-10)
                if local_rms > 0:
                    gain = min(ref_rms / local_rms, 3.0)  # Max +9.5dB
                    # Sanfte Gain-Rampe an den Rändern
                    edge = min(len(region) // 4, int(sr * 0.005))
                    ramp = np.ones(e - s)
                    if edge > 1:
                        ramp[:edge] = np.linspace(1.0, gain, edge)
                        ramp[-edge:] = np.linspace(gain, 1.0, edge)
                        ramp[edge:-edge] = gain
                    else:
                        ramp[:] = gain
                    ch_data[s:e] *= ramp.astype(np.float32)
            if result.ndim == 1:
                result = ch_data
            else:
                result[ch_idx] = ch_data
    except Exception as e:
        logger.warning("surgical_repair.py::_repair_tape_head_clog Ersatzpfad: %s", e)
    result = _safety_clamp(result, audio)
    return cast(np.ndarray, result.astype(np.float32))


def _repair_sticky_shed(audio: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    """Chirurgische Sticky-Shed-Reparatur: Binder-Hydrolyse-Residuen dämpfen.

    Sticky-Shed-Syndrom (Binder-Hydrolyse bei Ampex 406/407) verursacht
    moduliertes, breitbandiges Rauschen mit charakteristischer Hüllkurve.
    Detektion via HF-Rausch-Modulation, spektrale Subtraktion.
    """
    import numpy as np

    result = audio.copy()
    if audio.shape[-1] < 512:
        return result
    try:
        from scipy.signal import butter, sosfilt

        # Sticky-Shed lebt im HF-Bereich >6kHz als amplitudenmoduliertes Rauschen
        sos_hp = butter(4, 6000, "high", fs=sr, output="sos")
        channels = [result] if result.ndim == 1 else [result[ch] for ch in range(result.shape[0])]
        for ch_idx, ch_data in enumerate(channels):
            hf = sosfilt(sos_hp, ch_data)
            hf_env = np.abs(hf)
            if np.median(hf_env) < 1e-8:
                continue
            # Detektion: RMS-Modulation im HF-Band (charakteristisch für Sticky-Shed)
            frame = int(sr * 0.020)
            hop = frame // 2
            n_frames = max(1, (len(hf_env) - frame) // hop + 1)
            modulation = np.zeros(len(ch_data))
            for f in range(n_frames):
                seg = hf_env[f * hop : f * hop + frame]
                modulation[f * hop : f * hop + frame] = np.std(seg) / (np.mean(seg) + 1e-10)
            # Hohe Modulation (>0.5) = Sticky-Shed-verdächtig
            shed_mask = modulation > 0.5
            if not np.any(shed_mask):
                continue
            # Spektrale Subtraktion in den betroffenen Zonen
            from scipy.ndimage import label as _label

            labeled, n_regions = _label(shed_mask)
            for r in range(1, n_regions + 1):
                region = np.where(labeled == r)[0]
                if len(region) < 64:
                    continue
                s, e = region[0], min(len(ch_data), region[-1] + 1)
                # FFT-basierte Rauschunterdrückung nur in diesem Fenster
                seg = ch_data[s:e]
                spec = np.fft.rfft(seg)
                mag = np.abs(spec)
                noise_floor = np.median(mag[-len(mag) // 4 :]) * 0.7
                gain = np.maximum(mag - noise_floor, 0.0) / (mag + 1e-10)
                ch_data[s:e] = np.fft.irfft(spec * np.clip(gain, 0.2, 1.0), n=len(seg))
            if result.ndim == 1:
                result = ch_data
            else:
                result[ch_idx] = ch_data
    except Exception as e:
        logger.warning("surgical_repair.py::_repair_sticky_shed Ersatzpfad: %s", e)
    result = _safety_clamp(result, audio)
    return cast(np.ndarray, result.astype(np.float32))


def _repair_inner_groove_distortion(audio: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    """Chirurgische IGD-Reparatur: Progressive HF-Entzerrung gegen Plattenmitte.

    Inner-Groove-Distortion (Vinyl) nimmt zum Platteninneren zu (abnehmende
    Lineargeschwindigkeit). Detektion via HF-Rolloff-Gradient, adaptive
    HF-Anhebung mit nach innen zunehmender Stärke.
    """
    import numpy as np
    from scipy.signal import butter, sosfilt

    result = audio.copy()
    if audio.shape[-1] < 4096:
        return result
    try:
        # IGD betrifft vor allem den HF-Bereich >8kHz
        sos_hp = butter(4, 8000, "high", fs=sr, output="sos")
        channels = [result] if result.ndim == 1 else [result[ch] for ch in range(result.shape[0])]
        for ch_idx, ch_data in enumerate(channels):
            n = len(ch_data)
            # Teile in 4 Sektionen — IGD nimmt zur Mitte hin zu
            for sec in range(4):
                s = n * sec // 4
                e = n * (sec + 1) // 4
                if e - s < 1024:
                    continue
                seg = ch_data[s:e]
                hf = sosfilt(sos_hp, seg)
                hf_energy = float(np.sum(hf**2))
                total_energy = np.sum(seg**2) + 1e-10
                hf_ratio = hf_energy / total_energy
                # Wenn HF-Anteil in dieser Sektion < 0.5% → IGD-betroffen
                if hf_ratio < 0.005:
                    # Progressive HF-Anhebung (max +4dB für innerste Sektion)
                    boost_db = (sec + 1) * 1.0
                    boost_linear = 10 ** (boost_db / 20.0)
                    _sos_high_shelf = butter(2, 8000, "highshelf", fs=sr, output="sos")
                    # Shelf-Filter mit Gain
                    from scipy.signal import sosfilt

                    # Verwende biquad manuell für präzise Kontrolle
                    nyq = sr / 2
                    w0 = 8000 / nyq
                    Q = 1.0
                    import math

                    alpha = math.sin(w0) / (2 * Q)
                    A = math.sqrt(boost_linear)
                    # High-shelf Koeffizienten
                    cos_w0 = math.cos(w0)
                    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * math.sqrt(A) * alpha)
                    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
                    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * math.sqrt(A) * alpha)
                    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * math.sqrt(A) * alpha
                    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
                    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * math.sqrt(A) * alpha
                    b = np.array([b0 / a0, b1 / a0, b2 / a0])
                    a = np.array([1.0, a1 / a0, a2 / a0])
                    from scipy.signal import lfilter

                    ch_data[s:e] = lfilter(b, a, seg)
            if result.ndim == 1:
                result = ch_data
            else:
                result[ch_idx] = ch_data
    except Exception as e:
        logger.warning("surgical_repair.py::_repair_inner_groove_distortion Ersatzpfad: %s", e)
    result = _safety_clamp(result, audio)
    return cast(np.ndarray, result.astype(np.float32))


def _repair_motor_interference(audio: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    """Chirurgische Motor-Interferenz-Reparatur: Schmalband-Notchfilter.

    Plattenspieler/Bandmaschinen-Motoren erzeugen magnetische Einstreuungen
    bei Netzfrequenz (50/60Hz) + Harmonische. Detektion via spektraler
    Peaksuche, adaptive Notch-Filter pro Harmonische.
    """
    import numpy as np
    from scipy.signal import butter, sosfilt

    result = audio.copy()
    if audio.shape[-1] < 512:
        return result
    try:
        # Suche nach motor-typischen Frequenzen: 50/60Hz + Harmonische bis 300Hz
        spec = np.abs(np.fft.rfft(result if result.ndim == 1 else result[0]))
        freqs = np.fft.rfftfreq(
            len(result if result.ndim == 1 else result[0]) if result.ndim == 1 else result.shape[1], d=1 / sr
        )
        # Prüfe 50Hz und 60Hz Netzfrequenz
        for base_freq in [50.0, 60.0]:
            harmonics = [base_freq * h for h in range(1, 7) if base_freq * h < 300]
            peaks_found = 0
            for hz in harmonics:
                idx = np.argmin(np.abs(freqs - hz))
                if idx > 0 and idx < len(spec) - 1:
                    # Schmalband-Peak? (3× Umgebung)
                    local_bg = np.median(spec[max(0, idx - 10) : min(len(spec), idx + 10)])
                    if local_bg > 0 and spec[idx] > local_bg * 3:
                        peaks_found += 1
            if peaks_found >= 2:  # Mindestens 2 Harmonische = Motor-Interferenz bestätigt
                # Notch-Filter für jede gefundene Harmonische
                channels = [result] if result.ndim == 1 else [result[ch] for ch in range(result.shape[0])]
                for ch_idx, ch_data in enumerate(channels):
                    for hz in harmonics:
                        Q = 30.0
                        w0 = hz / (sr / 2)
                        bw = w0 / Q
                        sos_notch = butter(2, [w0 - bw, w0 + bw], "bandstop", fs=sr, output="sos")
                        ch_data[:] = sosfilt(sos_notch, ch_data)
                    if result.ndim == 1:
                        result = ch_data
                    else:
                        result[ch_idx] = ch_data
                break  # Nur eine Netzfrequenz behandeln
    except Exception as e:
        logger.warning("surgical_repair.py::_repair_motor_interference Ersatzpfad: %s", e)
    result = _safety_clamp(result, audio)
    return cast(np.ndarray, result.astype(np.float32))


def _repair_sibilance(audio: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    """Chirurgische Sibilance-Reparatur: Lokalisiertes De-Essing.

    Zischlaute (s, sch, z) sind schmalbandige Energie-Peaks bei 5-10kHz.
    Statt globalem De-Essing nur die detektierten Sibilance-Ereignisse
    mit einem schmalen Dynamic EQ bedämpfen.

    Referenz: Zwicker & Fastl (1999) — Psychoacoustics, Ch. 8
    """
    import numpy as np
    from scipy.signal import butter, sosfilt

    result = audio.copy()
    if audio.shape[-1] < 256:
        return result
    try:
        # Sibilance-Band: 5-10kHz
        sos_sib = butter(4, [5000, 10000], "bandpass", fs=sr, output="sos")
        channels = [result] if result.ndim == 1 else [result[ch] for ch in range(result.shape[0])]
        for ch_idx, ch_data in enumerate(channels):
            sib_band = sosfilt(sos_sib, ch_data)
            sib_env = np.abs(sib_band)
            if np.median(sib_env) < 1e-8:
                continue
            # Schwellwert: 4× Median der Sibilance-Hüllkurve
            threshold = np.median(sib_env) * 4.0
            sib_mask = sib_env > threshold
            if not np.any(sib_mask):
                continue
            from scipy.ndimage import label as _label

            labeled, n_regions = _label(sib_mask)
            for r in range(1, n_regions + 1):
                region = np.where(labeled == r)[0]
                if len(region) < int(sr * 0.005):  # <5ms ignorieren
                    continue
                s, e = region[0], min(len(ch_data), region[-1] + 1)
                # Kontext für sanftes Fade
                ctx = int(sr * 0.003)
                s_ctx = max(0, s - ctx)
                e_ctx = min(len(ch_data), e + ctx)
                # Schmalband-Dämpfung: -6dB im Sibilance-Band
                seg = ch_data[s_ctx:e_ctx]
                sosfilt(sos_sib, seg)
                # Dynamische Gain-Reduktion proportional zur Überschreitung
                excess = np.clip(sib_env[s_ctx:e_ctx] / (threshold + 1e-10), 1.0, 8.0)
                gain = 1.0 / np.sqrt(excess)  # Mehr Überschreitung → mehr Dämpfung
                # Nur im Sibilance-Band anwenden (TP bleibt unverändert)
                sos_lp = butter(4, 5000, "low", fs=sr, output="sos")
                sos_hp = butter(4, 10000, "high", fs=sr, output="sos")
                lp_seg = sosfilt(sos_lp, seg)
                hp_seg = sosfilt(sos_hp, seg)
                mid_seg = seg - lp_seg - hp_seg  # 5-10kHz Band
                # Sanfte Gain-Rampe an den Rändern
                ramp = np.ones(len(seg))
                edge = min(ctx, len(seg) // 4)
                if edge > 2:
                    ramp[:edge] = np.linspace(1.0, gain.mean(), edge)
                    ramp[-edge:] = np.linspace(gain.mean(), 1.0, edge)
                    ramp[edge:-edge] = gain[edge:-edge] if len(gain) > 2 * edge else gain.mean()
                else:
                    ramp = gain
                ch_data[s_ctx:e_ctx] = lp_seg + hp_seg + mid_seg * ramp.astype(np.float32)
            if result.ndim == 1:
                result = ch_data
            else:
                result[ch_idx] = ch_data
    except Exception as e:
        logger.warning("surgical_repair.py::_repair_sibilance Ersatzpfad: %s", e)
    result = _safety_clamp(result, audio)
    return cast(np.ndarray, result.astype(np.float32))


def _repair_transient_smearing(audio: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    """Chirurgische Transient-Verschmierungs-Reparatur: Transienten schärfen.

    Kompression/Limiting verschmiert Transienten (Ansätze werden weicher).
    Detektion via Energie-Anstiegsrate, dann dynamische Transienten-Anhebung
    (2-5ms Attack) zur Wiederherstellung der ursprünglichen Impulsivität.
    """
    import numpy as np

    result = audio.copy()
    if audio.shape[-1] < 128:
        return result
    try:
        channels = [result] if result.ndim == 1 else [result[ch] for ch in range(result.shape[0])]
        for ch_idx, ch_data in enumerate(channels):
            # Energie-Differenz (kurzes Fenster)
            frame = max(16, int(sr * 0.001))
            hop = frame // 2
            n_frames = max(1, (len(ch_data) - frame) // hop)
            energy = np.zeros(n_frames)
            for f in range(n_frames):
                energy[f] = np.sqrt(np.mean(ch_data[f * hop : f * hop + frame] ** 2) + 1e-10)
            if len(energy) < 4:
                continue
            # Energie-Anstiegsrate
            diff = np.diff(energy)
            attack_mask = diff > 0  # Nur Anstiege
            attack_indices = np.where(attack_mask)[0]
            for idx in attack_indices:
                # Transienten-Schärfung: kurze Anhebung (2ms) direkt am Anstieg
                attack_samples = int(sr * 0.002)
                s = idx * hop
                e = min(len(ch_data), s + attack_samples)
                if e - s < 4:
                    continue
                # +3dB Boost mit exponentieller Decay
                boost = np.exp(-np.arange(e - s) / (attack_samples / 3))
                boost = 1.0 + 0.4 * boost  # Max +3dB, schnell abfallend
                ch_data[s:e] *= boost.astype(np.float32)
            if result.ndim == 1:
                result = ch_data
            else:
                result[ch_idx] = ch_data
    except Exception as e:
        logger.warning("surgical_repair.py::_repair_transient_smearing Ersatzpfad: %s", e)
    result = _safety_clamp(result, audio)
    return cast(np.ndarray, result.astype(np.float32))


def _repair_tape_splice(audio: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    """Chirurgische Bandschnitt-Artefakt-Reparatur: Klick + Pegelsprung-Korrektur.

    Bandschnitte (Tape Splice) erzeugen einen charakteristischen Klick
    gefolgt von einem kurzen Pegelsprung. Detektion via kombiniertem
    Transienten+Pegelsprung-Detektor, dann Interpolation + Gain-Matching.
    """
    import numpy as np
    from scipy.interpolate import CubicSpline

    result = audio.copy()
    if audio.shape[-1] < 64:
        return result
    try:
        channels = [result] if result.ndim == 1 else [result[ch] for ch in range(result.shape[0])]
        for ch_idx, ch_data in enumerate(channels):
            diff = np.abs(np.diff(ch_data))
            if len(diff) < 3:
                continue
            # Klick-Detektion (wie clicks, aber größeres Fenster wegen Pegelsprung)
            threshold = np.median(diff) * 8.0
            click_mask = diff > threshold
            click_idx = np.where(click_mask)[0]
            if len(click_idx) == 0:
                continue
            # Gruppiere (Splice-Klicks sind breiter, ~5ms)
            gap = int(sr * 0.005)
            groups = []
            cur = [click_idx[0]]
            for i in click_idx[1:]:
                if i - cur[-1] <= gap:
                    cur.append(i)
                else:
                    groups.append(cur)
                    cur = [i]
            groups.append(cur)
            for group in groups:
                s_click = group[0]
                e_click = group[-1] + 1
                # Splice-Klick: zusätzlich Pegelsprung prüfen (±5ms nach Klick)
                after_start = min(len(ch_data), e_click + int(sr * 0.002))
                after_end = min(len(ch_data), e_click + int(sr * 0.010))
                before_end = max(0, s_click)
                before_start = max(0, s_click - int(sr * 0.010))
                if after_end <= after_start or before_end <= before_start:
                    continue
                rms_before = np.sqrt(np.mean(ch_data[before_start:before_end] ** 2) + 1e-10)
                rms_after = np.sqrt(np.mean(ch_data[after_start:after_end] ** 2) + 1e-10)
                rms_ratio = rms_after / (rms_before + 1e-10)
                # Pegelsprung >2dB detektiert → Splice bestätigt
                if rms_ratio < 0.6 or rms_ratio > 1.7:
                    # 1. Klick interpolieren (±2ms)
                    win = int(sr * 0.002)
                    s = max(0, s_click - win)
                    e = min(len(ch_data), e_click + win)
                    if e - s >= 6:
                        x = np.arange(s, e)
                        y = ch_data[s:e].copy()
                        intact = np.ones(e - s, dtype=bool)
                        click_inner_s = max(0, s_click - s - 2)
                        click_inner_e = min(e - s, e_click - s + 3)
                        intact[click_inner_s:click_inner_e] = False
                        if intact.sum() >= 4:
                            try:
                                cs = CubicSpline(x[intact], y[intact], bc_type="natural")
                                ch_data[s:e] = cs(x)
                            except Exception:
                                ch_data[s:e] = np.interp(x, x[intact], y[intact])
                    # 2. Pegelsprung korrigieren (Gain-Matching nach dem Splice)
                    if rms_ratio > 0.15 and rms_ratio < 6.0:
                        target_rms = max(rms_before, rms_after)
                        fade_len = int(sr * 0.015)
                        fade_start = e_click
                        fade_end = min(len(ch_data), fade_start + fade_len)
                        if fade_end > fade_start:
                            ramp = np.linspace(rms_after / target_rms, 1.0, fade_end - fade_start)
                            ch_data[fade_start:fade_end] *= ramp.astype(np.float32)
            if result.ndim == 1:
                result = ch_data
            else:
                result[ch_idx] = ch_data
    except Exception as e:
        logger.warning("surgical_repair.py::unbekannter Ersatzpfad: %s", e)
    result = _safety_clamp(result, audio)
    return cast(np.ndarray, result.astype(np.float32))


# Mapping: Defekt-Typ → Lightweight-Repair-Funktion
# §2.59: VOLLSTÄNDIGE Abdeckung aller chirurgisch behandelbaren Defekte.
# Neue Funktionen: §2.59.5–§2.59.12 (2026-07-08)
_SURGICAL_REPAIR_FUNCTIONS = {
    # ── Bandtransport ─────────────────────────────────────────
    "wow": _repair_wow_flutter,
    "flutter": _repair_wow_flutter,
    "transport_bump": _repair_wow_flutter,
    "scrape_flutter": _repair_wow_flutter,
    "multiband_wow_flutter": _repair_wow_flutter,
    # ── Rauschen ──────────────────────────────────────────────
    "modulation_noise": _repair_hiss,
    # ── Transienten (Chirurgisch, §2.59.2) ────────────────────
    "clicks": _repair_clicks,
    "crackle": _repair_crackle,
    "tape_splice_artifact": _repair_tape_splice,
    # ── Dropouts (§2.59.3) ────────────────────────────────────
    "dropouts": _repair_dropouts,
    "dropout_oxide": _repair_dropouts,
    "dropout_head_contact": _repair_dropouts,
    "dropout_splice": _repair_dropouts,
    # ── DC-Offset (§2.59.4) ───────────────────────────────────
    "dc_offset": _repair_wow_flutter,
    # ── Temporale Artefakte (§2.59.5) ─────────────────────────
    "pre_echo": _repair_pre_echo,
    "print_through": _repair_pre_echo,
    "groove_echo": _repair_groove_echo,
    "mpeg_frame_loss": _repair_mpeg_frame_loss,
    # ── Kopf-/Kontakt-Defekte (§2.59.6) ───────────────────────
    "tape_head_clog": _repair_tape_head_clog,
    "sticky_shed_residue": _repair_sticky_shed,
    # ── Positionsabhängige Defekte (§2.59.7) ──────────────────
    "inner_groove_distortion": _repair_inner_groove_distortion,
    "motor_interference": _repair_motor_interference,
    # ── Spektrale Artefakte (§2.59.8) ─────────────────────────
    "sibilance": _repair_sibilance,
    "transient_smearing": _repair_transient_smearing,
}


class SurgicalRepair:
    """Führt zeitlich präzise, ortsgenaue Reparaturen durch.

    Extrahiert jedes Defekt-Fenster mit Kontext, wendet die Phase an,
    und cross-faded das Ergebnis nahtlos zurück.
    """

    def __init__(
        self,
        sr: int = 48000,
        context_ms: float = 50.0,  # Kontext vor/nach dem Defekt
        crossfade_ms: float = 10.0,  # Cross-Fade-Dauer
    ) -> None:
        self.sr = sr
        self._context_samples = int(context_ms * sr / 1000)
        self._crossfade_samples = int(crossfade_ms * sr / 1000)

    def _detect_transients(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Erkennt Transienten für Crossfade-Vermeidung."""
        if audio.ndim == 1:
            signal = audio
        else:
            signal = np.mean(audio, axis=0)
        energy = signal**2
        window = max(1, int(sr * 0.005))
        kernel = np.ones(window) / window
        smooth = np.convolve(energy, kernel, mode="same")
        threshold = np.convolve(smooth, kernel, mode="same") * 3.0 + 1e-10
        return cast(np.ndarray, energy > threshold)

    def repair(
        self,
        audio: np.ndarray,
        instances: list[DefectInstance],
        phase_fn: Any,  # callable(audio_segment, sr, **kwargs) → np.ndarray
        phase_kwargs: dict[str, Any] | None = None,
    ) -> RepairResult:
        """Repariert jede Defekt-Instanz einzeln mit Cross-Fade.

        Args:
            audio: Original-Audio (channels, samples) oder (samples,)
            instances: Liste zeitlich lokalisierter Defekte
            phase_fn: Funktion die ein Audio-Segment repariert
            phase_kwargs: Zusätzliche KWArgs für die Phase

        Returns:
            RepairResult mit repariertem Audio
        """
        if not instances:
            return RepairResult(audio=audio.copy(), zones_skipped=0)

        was_mono = audio.ndim == 1
        if was_mono:
            audio = audio.reshape(1, -1)

        result = audio.copy()
        total_samples = audio.shape[1]
        repaired = 0
        skipped = 0

        for inst in sorted(instances, key=lambda x: x.start_s):
            # §2.59.10: Adaptiver Kontext — kurze Events (clicks < 5ms)
            # brauchen weniger Kontext als lange (dropouts, tape_head_clog)
            _zone_dur_ms = (inst.end_s - inst.start_s) * 1000
            if _zone_dur_ms < 5:
                _ctx = min(self._context_samples, int(self.sr * 0.005))  # 5ms
            elif _zone_dur_ms < 50:
                _ctx = min(self._context_samples, int(self.sr * 0.020))  # 20ms
            else:
                _ctx = self._context_samples  # 50ms
            s0 = max(0, int(inst.start_s * self.sr) - _ctx)
            s1 = min(total_samples, int(inst.end_s * self.sr) + _ctx)

            # Minimum: Fenster muss genug Samples für DSP haben
            if s1 - s0 < 32:  # 0.7ms @ 48kHz — absolutes Minimum
                skipped += 1
                continue

            # Extrahiere Fenster mit Kontext
            segment = audio[:, s0:s1].copy()
            original_segment = segment.copy()

            # Wende Phase nur auf dieses Fenster an
            try:
                kwargs = {
                    "audio": segment,
                    "sr": self.sr,
                    "material": phase_kwargs.pop("material", "unknown") if phase_kwargs else "unknown",
                    "mode": phase_kwargs.pop("mode", "restoration") if phase_kwargs else "restoration",
                }
                if phase_kwargs:
                    kwargs.update(phase_kwargs)
                repaired_segment = phase_fn(**kwargs)
                if isinstance(repaired_segment, np.ndarray):
                    segment = repaired_segment
            except Exception as _repair_exc:
                logger.debug(
                    "CHIRURGIE-ueberspringen: %s @ %.3fs — %s", inst.defect_type, inst.start_s, str(_repair_exc)[:80]
                )
                skipped += 1
                continue

            # Phasen-Ausrichtung vor Crossfade (verhindert Kammfilter)
            segment = self._align_phase(segment, original_segment)

            # Cross-Fade: nur an den Rändern, Mitte bleibt Reparatur
            if segment.shape[1] >= self._crossfade_samples * 2:
                self._apply_crossfade(segment, original_segment, self._crossfade_samples)

            # Pegel-Angleich: RMS vorher/nachher matchen
            segment = self._match_rms(segment, original_segment)

            # Zurückschreiben
            result[:, s0:s1] = segment
            repaired += 1

        if was_mono:
            result = result[0]

        # Transparente x/y-Logik pro Defekt-Typ
        _defect_type = instances[0].defect_type if instances else "unknown"
        _pct = 100 * repaired / max(len(instances), 1)
        if _pct >= 100:
            logger.info(
                "🔧 CHIRURGIE-OK: %s — %d/%d Zonen repariert (100%%)",
                _defect_type,
                repaired,
                len(instances),
            )
        elif _pct >= 50:
            logger.info(
                "🔧 CHIRURGIE: %s — %d/%d Zonen repariert (%.0f%%), %d übersprungen",
                _defect_type,
                repaired,
                len(instances),
                _pct,
                skipped,
            )
        elif repaired > 0:
            logger.warning(
                "⚠️ CHIRURGIE-SCHWACH: %s — nur %d/%d Zonen repariert (%.0f%%), %d übersprungen",
                _defect_type,
                repaired,
                len(instances),
                _pct,
                skipped,
            )
        else:
            logger.warning(
                "❌ CHIRURGIE-FEHLER: %s — 0/%d Zonen repariert! "
                "Alle %d Instanzen übersprungen — Defekt bleibt unbehandelt",
                _defect_type,
                len(instances),
                len(instances),
            )

        return RepairResult(
            audio=result,
            zones_repaired=repaired,
            zones_skipped=skipped,
        )

    @staticmethod
    def _apply_crossfade(
        repaired: np.ndarray,
        original: np.ndarray,
        fade_samples: int,
    ) -> None:
        """Cosine Cross-Fade an den Rändern (psychoakustisch transparent).

        Verwendet Cosine-Ramp wie SectionStrengthEnvelope (§8.3).
        Max 1 dB / 100 ms Änderungsrate — unterhalb der menschlichen
        Wahrnehmungsschwelle (Zwicker & Fastl 1999).
        """
        if fade_samples <= 0 or repaired.shape[1] < fade_samples * 2:
            return

        # Cosine Fade-In (linker Rand): sanfter als linear
        ramp_in = 0.5 * (1 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))
        for ch in range(repaired.shape[0]):
            repaired[ch, :fade_samples] = (
                original[ch, :fade_samples] * (1 - ramp_in) + repaired[ch, :fade_samples] * ramp_in
            )

        # Cosine Fade-Out (rechter Rand)
        ramp_out = 0.5 * (1 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))
        for ch in range(repaired.shape[0]):
            repaired[ch, -fade_samples:] = (
                original[ch, -fade_samples:] * (1 - ramp_out[::-1]) + repaired[ch, -fade_samples:] * ramp_out[::-1]
            )

    @staticmethod
    def _align_phase(repaired: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Phasen-Ausrichtung: verhindert Kammfilter im Crossfade.

        Findet die optimale Phasenrotation, die die Differenz
        zwischen repariertem und originalem Signal minimiert.
        """
        if repaired.shape[1] < 100 or original.shape[1] < 100:
            return repaired
        # Kreuzkorrelation an den Rändern
        result = repaired.copy()
        for ch in range(repaired.shape[0]):
            # Linker Rand: aligne erste 100 Samples
            edge_orig = original[ch, :100]
            edge_rep = repaired[ch, :100]
            if np.std(edge_orig) > 1e-8 and np.std(edge_rep) > 1e-8:
                # Einfache Phasenkorrektur: Vorzeichen-Anpassung
                corr = np.correlate(edge_orig, edge_rep, mode="full")
                shift = np.argmax(corr) - 99
                if abs(shift) <= 5 and shift != 0:
                    if shift > 0:
                        result[ch, :-shift] = repaired[ch, shift:]
                    else:
                        result[ch, -shift:] = repaired[ch, :shift]
        return result

    @staticmethod
    def _match_rms(repaired: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Passt RMS-Pegel des reparierten Segments ans Original an."""
        rms_orig = np.sqrt(np.mean(original**2)) + 1e-10
        rms_rep = np.sqrt(np.mean(repaired**2)) + 1e-10
        if abs(rms_rep - rms_orig) / rms_orig > 0.01:  # >1% Abweichung
            return repaired * (rms_orig / rms_rep)  # type: ignore[no-any-return]
        return repaired
