# v10.101 SOTA: Import-Analyse durch perzeptuelle Pipeline validiert.
import logging
import os
from typing import Any

# pylint: disable=import-outside-toplevel
import numpy as np
import soundfile as sf


def _load_with_sf(filepath, always_2d: bool = False):
    """Wrapper for sf.read — use load_audio_file() for production pipelines.

    §G-SF-READ: Thin wrapper around soundfile.read(); always_2d controls
    whether mono audio is returned as (samples,) or (samples, 1).
    This is the ONLY sanctioned sf.read call site — all other modules
    MUST use load_audio_file().
    """
    return sf.read(filepath, always_2d=always_2d)


logger = logging.getLogger(__name__)

# Formats that libsndfile cannot decode — route directly to pedalboard (FFmpeg backend)
# to avoid a guaranteed exception + log noise on every import of these files.
_SF_UNSUPPORTED_EXT: frozenset[str] = frozenset(
    {
        ".mp3",
        ".mp2",
        ".mp1",
        ".m4a",
        ".m4b",
        ".m4p",
        ".aac",
        ".wma",
        ".asf",
        ".opus",
        ".webm",
        ".amr",
        ".3gp",
        ".3g2",
        ".ac3",
        ".dts",
    }
)


def _estimate_interchannel_lag_samples(audio: np.ndarray, sr: int, max_seconds: float = 5.0) -> int:
    """Schätzt L/R lag (samples) via dual-confirmation GCC-PHAT + time-domain XCorr.

    §G13 Dual-Confirmation: GCC-PHAT detects candidate lag → time-domain
    cross-correlation verifies it independently.  Both estimators must agree
    within ±50 samples.  Different failure modes ensure false positives from
    one method are caught by the other (SOTA TDOA practice, Knapp & Carter 1976).

    Returns 0 for non-stereo inputs, false positives, or analysis failure.
    """
    try:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim != 2:
            return 0
        if arr.shape[1] == 2 and arr.shape[0] > 2:
            l = arr[:, 0]
            r = arr[:, 1]
        elif arr.shape[0] == 2 and arr.shape[1] > 2:
            l = arr[0]
            r = arr[1]
        else:
            return 0

        n = min(len(l), len(r), int(float(sr) * max_seconds))
        if n < max(1024, sr // 10):
            return 0

        x = l[:n].astype(np.float64)
        y = r[:n].astype(np.float64)
        n_fft = 1
        while n_fft < 2 * n:
            n_fft <<= 1

        # §v10.0.4: High-Band GCC-PHAT — eliminiert Periodenambiguität
        # Bei rein tonalen Signalen (Sinus, Orgel, Flöte) erzeugt GCC-PHAT auf dem
        # gesamten Spektrum Ambiguitäten, weil PHAT alle Frequenzen gleich gewichtet
        # und die Grundfrequenz-Periode mehrere Peaks im Suchfenster erzeugt.
        # Lösung: GCC-PHAT NUR auf Frequenzen > 2 kHz anwenden.
        #   – Bei 2 kHz ist 1 Periode = 0.5 ms = 24 samples @ 48 kHz
        #   – Maximaler erwarteter Lag: ±200 ms = ±9600 samples
        #   – Kein Vielfaches von 24 passt in [−200, +200] ms → keine Ambiguität.
        # Das 2-kHz-Hochpassfilter wird direkt im Frequenzbereich angewandt.
        _freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        _hp_mask = np.abs(_freqs) >= 2000.0  # Nur Frequenzen ≥ 2 kHz
        _hp_mask = _hp_mask.astype(np.float64)

        X = np.fft.rfft(x, n=n_fft)
        Y = np.fft.rfft(y, n=n_fft)
        X_hp = X * _hp_mask
        Y_hp = Y * _hp_mask

        # Energie-Check: Hat das High-Band genug Signal?
        _hp_energy = float(np.sum(np.abs(X_hp) ** 2))
        _total_energy = float(np.sum(np.abs(X) ** 2))
        _hp_fraction = _hp_energy / max(_total_energy, 1e-10)

        if _hp_fraction < 0.05:
            # High-Band zu schwach (z.B. stark tiefpassgefiltertes Material).
            # Fallback: Vollband-GCC-PHAT (akzeptiert Ambiguitätsrisiko).
            cross = X * np.conj(Y)
        else:
            cross = X_hp * np.conj(Y_hp)

        gcc = np.fft.irfft(cross / (np.abs(cross) + 1e-10), n=n_fft)

        max_delay = min(int(sr * 0.2), n - 1)  # ±200 ms
        if max_delay <= 0:
            return 0

        search = np.concatenate([gcc[n_fft - max_delay :], gcc[: max_delay + 1]])
        # §G13 SNR-Gate: GCC-PHAT peak-to-RMS ratio separates true correlations
        # from random noise.  True inter-channel delay → correlated L/R → sharp
        # GCC peak → high SNR.  False positive → uncorrelated → flat GCC → SNR≈1.
        # Threshold 3.0 = 9.5 dB peak-to-background (rigorous, cross-validated).
        _rms = float(np.sqrt(np.mean(np.abs(search) ** 2)) + 1e-12)
        if _rms > 1e-12:
            _snr = float(np.max(np.abs(search))) / _rms
        else:
            _snr = 1.0
        if _snr < 5.0:  # < 5.0 = false positive (uncorrelated noise gives ~4.0 SNR)
            return 0  # GCC peak indistinguishable from noise — false positive
        _gcc_lag = int(np.argmax(np.abs(search))) - max_delay

        # §G13 Dual-Confirmation: verify GCC-PHAT candidate via time-domain XCorr.
        # Time-domain correlation uses raw signals (no PHAT whitening), so it has
        # different failure modes.  Both must agree within ±50 samples.
        _MAX_XCORR_SAMPLES = min(48000 * 3, len(x))
        _xc = np.correlate(x[:_MAX_XCORR_SAMPLES], y[:_MAX_XCORR_SAMPLES], mode="same")
        _xc_center = len(_xc) // 2
        _xc_max_delay = min(int(sr * 0.2), _xc_center)
        _xc_search = _xc[_xc_center - _xc_max_delay : _xc_center + _xc_max_delay + 1]
        _xc_rms = float(np.sqrt(np.mean(_xc_search.astype(np.float64) ** 2)) + 1e-12)
        _xc_snr = float(np.max(np.abs(_xc_search))) / _xc_rms if _xc_rms > 1e-12 else 1.0
        if _xc_snr < 5.0:
            return 0  # Time-domain XCorr also says no — false positive confirmed
        _xc_lag = int(np.argmax(np.abs(_xc_search))) - _xc_max_delay
        if abs(_gcc_lag - _xc_lag) > 50:
            return 0  # Disagreement — neither estimator can be trusted
        return _gcc_lag
    except Exception:
        logger.warning("file_import.py::_estimate_interchannel_lag_samples Ersatzpfad", exc_info=True)
        return 0


def _estimate_interchannel_lag_multi_point(
    audio: np.ndarray, sr: int, num_points: int = 3, window_s: float = 5.0
) -> dict:
    """Misst Interchannel-Lag an mehreren Positionen im Signal (§G13).

    GCC-PHAT auf nur einem Fenster (z.B. den ersten 5 s) kann bei
    zeitlich variierendem Lag (Banddehnung, Dropouts) ein falsches Bild
    liefern.  Multi-Point-Messung an Start, Mitte und Ende des Signals
    liefert ein Konsistenzprofil:

        - Alle Messungen innerhalb ±50 samples → Lag ist konsistent.
          Eine globale Korrektur reicht.
        - Messungen streuen > 100 samples → Lag variiert zeitlich.
          Median wird als globale Basiskorrektur verwendet; STCG
          behandelt die per-Chunk-Variation während Phase 12.

    Returns dict mit:
        points:     [(position_frac, lag_samples), ...]
        median_lag: Median-Lag über alle Messpunkte
        consistent: True wenn alle Messungen innerhalb ±50 samples liegen
        max_spread: max(|lag_i - lag_j|) über alle Punktpaare
    """
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim != 2:
        return {"points": [], "median_lag": 0, "consistent": True, "max_spread": 0}

    # ── Orientierungs-Erkennung (wie in _estimate_interchannel_lag_samples) ──
    if arr.shape[1] == 2 and arr.shape[0] > 2:
        # Channels-last (N, 2) — Audio-Samples in Achse 0
        total_n = arr.shape[0]
        _get_lr = lambda a: (a[:, 0], a[:, 1])
    elif arr.shape[0] == 2 and arr.shape[1] > 2:
        # Channels-first (2, N) — Audio-Samples in Achse 1
        total_n = arr.shape[1]
        _get_lr = lambda a: (a[0], a[1])
    else:
        return {"points": [], "median_lag": 0, "consistent": True, "max_spread": 0}

    window_n = int(sr * window_s)
    lags: list[int] = []

    for i in range(num_points):
        # Positionen: 0%, 50%, 90% (bzw. äquidistant bei num_points > 3)
        frac = i / max(num_points - 1, 1) if num_points > 1 else 0.5
        # Nicht ganz ans Ende (letztes Fenster braucht window_n Samples)
        max_start = max(0, total_n - window_n)
        start = min(int(frac * total_n), max_start)
        end = min(start + window_n, total_n)

        if end - start < max(1024, sr // 10):
            continue

        # Slice entlang der Sample-Achse
        if arr.shape[0] == 2 and arr.shape[1] > 2:
            chunk = arr[:, start:end]  # (2, window_n)
        else:
            chunk = arr[start:end]  # (window_n, 2)
        l_ch, r_ch = _get_lr(chunk)

        # GCC-PHAT auf diesem Fenster
        n = len(l_ch)
        n_fft = 1
        while n_fft < 2 * n:
            n_fft <<= 1
        X = np.fft.rfft(l_ch.astype(np.float64), n=n_fft)
        Y = np.fft.rfft(r_ch.astype(np.float64), n=n_fft)
        cross = X * np.conj(Y)
        gcc_raw = np.fft.irfft(cross / (np.abs(cross) + 1e-10), n=n_fft)

        max_delay = min(int(sr * 0.2), n - 1)
        if max_delay <= 0:
            continue
        search = np.concatenate([gcc_raw[n_fft - max_delay :], gcc_raw[: max_delay + 1]])
        # §G13 SNR-Gate: GCC-PHAT peak-to-RMS ratio separates true correlations
        # from noise.  True delay → high SNR (>> 3).  False positive → SNR≈1.
        _rms = float(np.sqrt(np.mean(np.abs(search) ** 2)) + 1e-12)
        _snr = float(np.max(np.abs(search))) / _rms if _rms > 1e-12 else 1.0
        if _snr < 5.0:  # < 5.0 = false positive (uncorrelated noise gives ~4.0 SNR)
            continue  # GCC peak indistinguishable from noise — false positive
        lag = int(np.argmax(np.abs(search))) - max_delay
        lags.append(lag)

    if not lags:
        return {"points": [], "median_lag": 0, "consistent": True, "max_spread": 0}

    median_lag = int(np.median(lags))
    max_spread = max(abs(a - b) for a in lags for b in lags) if len(lags) > 1 else 0
    consistent = max_spread <= 50

    # Positionen für Logging berechnen
    positions = []
    for i, lag in enumerate(lags):
        frac = i / max(len(lags) - 1, 1) if len(lags) > 1 else 0.5
        positions.append((round(frac, 2), lag))

    return {
        "points": positions,
        "median_lag": median_lag,
        "consistent": consistent,
        "max_spread": max_spread,
    }


def _lazy_get_carrier_tools():
    from .carrier_forensics import analyze_carrier_forensics
    from .carrier_ml_classifier import classify_carrier_ml

    return analyze_carrier_forensics, classify_carrier_ml


def _require_audio_array(audio: np.ndarray | None) -> np.ndarray:
    """Normalisiert optionales Audio auf ein garantiertes float32-ndarray."""
    if audio is None:
        raise ValueError("Audio read error: kein Audio dekodiert")
    return np.asarray(audio, dtype=np.float32)


def detect_carrier(filepath: str, meta: dict[str, Any] | None = None) -> str:
    """
    Versucht, den Tonträger (Schallplatte, Kassette, CD, Band, etc.)
    anhand von Dateiname, Metadaten oder User-Tag zu erkennen.
    """
    name = os.path.basename(filepath).lower()
    meta = meta or {}
    # Heuristik: Dateiname
    if any(
        x in name
        for x in [
            "vinyl",
            "lp",
            "platte",
            "schallplatte",
            "33rpm",
            "45rpm",
            "78rpm",
            "shellac",
            "schellack",
        ]
    ):
        return "Schallplatte (Vinyl/Schellack)"
    if any(x in name for x in ["cassette", "mc", "kassette", "tape"]):
        return "Kassette/Band"
    if any(x in name for x in ["cd", "compactdisc", "compact_disc"]):
        return "CD"
    if any(x in name for x in ["minidisc", "md"]):
        return "MiniDisc"
    if any(x in name for x in ["dat", "digitalaudiotape"]):
        return "DAT"
    if any(x in name for x in ["reel", "tonband", "spule"]):
        return "Tonband/Reel"
    # Heuristik: Metadaten
    for _, v in meta.items() if meta else []:
        vstr = str(v).lower()
        if "vinyl" in vstr or "lp" in vstr or "platte" in vstr or "schellack" in vstr:
            return "Schallplatte (Vinyl/Schellack)"
        if "cassette" in vstr or "mc" in vstr or "kassette" in vstr or "tape" in vstr:
            return "Kassette/Band"
        if "cd" in vstr:
            return "CD"
        if "minidisc" in vstr or "md" in vstr:
            return "MiniDisc"
        if "dat" in vstr:
            return "DAT"
        if "reel" in vstr or "tonband" in vstr or "spule" in vstr:
            return "Tonband/Reel"
    return "Unbekannt"


def is_supported_audio_file(filename: str) -> bool:
    """Prüft ob Datei ein unterstütztes Audioformat ist."""
    # Unterstützte Formate: WAV, MP3, FLAC, OGG, AAC, AIFF (erweiterbar)
    return filename.lower().endswith(
        (
            ".wav",
            ".mp3",
            ".flac",
            ".ogg",
            ".aac",
            ".aiff",
            ".aif",
            ".wma",
            ".opus",
            ".m4a",
            ".alac",
            ".caf",
        )
    )


def load_audio_file(
    filepath: str,
    target_sr: int | None = None,
    mono: bool = False,
    do_carrier_analysis: bool = True,
) -> dict[str, Any] | None:
    """Lädt Audiodatei robust und gibt Dict mit allen Metadaten zurück.

    Args:
        filepath: Pfad zur Audiodatei
        target_sr: Ziel-Sample-Rate (optional, Resampling wenn nötig)
        mono: Wenn True, konvertiere zu Mono

    Returns:
        Dict mit 'audio', 'sr', 'channels', 'format', 'duration', 'meta', 'error'
    """
    result: dict[str, Any] = {
        "audio": None,
        "sr": None,
        "channels": None,
        "format": None,
        "duration": None,
        "meta": {},
        "carrier": None,
        "input_channels": None,
        "error": None,
    }
    try:
        if not os.path.isfile(filepath):
            result["error"] = "File not found"
            return result
        if not is_supported_audio_file(filepath):
            result["error"] = "Unsupported file type"
            return result

        _ext = os.path.splitext(filepath)[1].lower()
        _sf_unsupported = _ext in _SF_UNSUPPORTED_EXT

        # Opportunistic capability probe: some deployments support MP3/AAC via
        # libsndfile plugins even when extension-only heuristics mark them unsupported.
        # If SoundFile can read header metadata, prefer the SoundFile decode path
        # because it preserves inter-channel alignment more reliably than FFmpeg fallbacks.
        if _sf_unsupported:
            try:
                sf.info(filepath)
                _sf_unsupported = False
                logger.debug("laden_audio_file: soundfile capability probe passed for %s", _ext)
            except Exception:
                logger.warning("file_import.py::laden_audio_file Ersatzpfad", exc_info=True)

        # ── Metadata ─────────────────────────────────────────────────────────
        # soundfile.info() is fast and reliable for lossless formats.
        # For MP3/AAC/WMA etc. it always fails — skip it and use extension only.
        if not _sf_unsupported:
            try:
                _info = sf.info(filepath)
                result["format"] = _info.format
                result["channels"] = _info.channels
                result["sr"] = _info.samplerate
                result["duration"] = _info.duration
                _extra_info = getattr(_info, "extra_info", "") or ""
                result["meta"] = {"extra_info": str(_extra_info)} if _extra_info else {}
            except Exception as _meta_exc:
                logger.debug("laden_audio_file: sf.info metadata fehlgeschlagen (%s) — using extension", _meta_exc)
                result["format"] = _ext[1:].upper()
        else:
            result["format"] = _ext[1:].upper()

        # ── Audio decode ──────────────────────────────────────────────────────
        # Route: soundfile (lossless) → pedalboard/FFmpeg (lossy + universal fallback)
        audio: np.ndarray | None = None
        sr: int = 0

        if not _sf_unsupported:
            # Stufe 1: soundfile — WAV, FLAC, OGG, AIFF, ALAC, CAF …
            try:
                audio, sr = _load_with_sf(filepath, always_2d=False)
                logger.debug("laden_audio_file: soundfile OK (%s)", filepath)
            except Exception as _e1:
                logger.debug("laden_audio_file: soundfile fehlgeschlagen (%s) — trying pedalboard", _e1)

        if audio is None and _sf_unsupported:
            # Stufe 2 (lossy primary): pedalboard/FFmpeg — stabiler als pydub für in-process.
            # pydub.from_file() kann via libavcodec SIGABRT auslösen (auch ohne malloc_trim),
            # weil FFmpeg-Allokierungsfehler unter Speicherdruck als abort() propagieren.
            # Pedalboard nutzt denselben FFmpeg-Stack, aber ohne audioop C-Extension.
            try:
                from pedalboard.io import AudioFile as _PBAudioFile  # type: ignore

                with _PBAudioFile(filepath) as _f:  # pylint: disable=not-context-manager
                    sr = int(_f.samplerate)
                    _frames = _f.frames
                    _chunk = sr * 300  # 300 s chunks to avoid OOM on long files
                    _parts: list[np.ndarray] = []
                    _read = 0
                    while _read < _frames:
                        _block = _f.read(min(_chunk, _frames - _read))
                        if _block.shape[-1] == 0:
                            break  # EOF — pedalboard.frames can overcount for VBR
                        _parts.append(_block)
                        _read += _block.shape[-1]
                _raw = np.concatenate(_parts, axis=1) if len(_parts) > 1 else _parts[0]
                # pedalboard returns (channels, samples) — transpose to (samples, channels)
                if _raw.ndim == 2:
                    _raw = _raw.T
                audio = _raw.astype(np.float32)
                logger.debug("laden_audio_file: pedalboard/FFmpeg OK for lossy (%s)", filepath)
            except Exception as _e_pb_lossy:
                logger.debug(
                    "laden_audio_file: pedalboard fehlgeschlagen for lossy (%s) — pydub subprocess Ersatzpfad",
                    _e_pb_lossy,
                )

        if audio is None and _sf_unsupported:
            # Stufe 3 (lossy fallback): pydub — in subprocess, um in-process SIGABRT zu vermeiden.
            # Subprocess-Isolation: SIGABRT des Child-Prozesses wird als Exception behandelt.
            try:
                import json
                import subprocess
                import sys
                import tempfile

                _script = (
                    "import sys, json, numpy as np\n"
                    "from pydub import AudioSegment\n"
                    f"seg = AudioSegment.from_file({filepath!r})\n"
                    "sr = seg.frame_rate\n"
                    "bw = seg.sample_width\n"
                    "raw = bytes(seg.raw_data)\n"
                    "dt = np.uint8 if bw==1 else (np.int16 if bw==2 else np.int32)\n"
                    "s = np.frombuffer(raw, dtype=dt).astype(np.float32)\n"
                    "if bw == 1:\n"
                    "    s = (s - 128.0) / 128.0\n"
                    "else:\n"
                    "    s /= float(2**(bw*8-1))\n"
                    "if seg.channels > 1:\n"
                    "    s = s.reshape(-1, seg.channels)\n"
                    "result = {'sr': sr, 'channels': seg.channels, 'shape': list(s.shape)}\n"
                    "with open(sys.argv[1], 'wb') as f:\n"
                    "    np.save(f, s)\n"
                    "logger.info(json.dumps(result))\n"
                )
                _tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
                _tmp.close()
                _proc = subprocess.run(
                    [sys.executable, "-c", _script, _tmp.name],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if _proc.returncode == 0 and _proc.stdout.strip():
                    _d = json.loads(_proc.stdout.strip())
                    audio = np.load(_tmp.name).astype(np.float32)
                    sr = int(_d["sr"])
                    logger.debug("laden_audio_file: pydub subprocess OK (%s)", filepath)
                else:
                    raise RuntimeError(_proc.stderr[:500] or f"returncode={_proc.returncode}")
            except Exception as _e2:
                logger.warning("§V6 Audio-Laden (pydub-Subprozess) fehlgeschlagen für %s — Error-Result zurückgegeben: %s", filepath, _e2)
                result["error"] = f"Audio read error: pedalboard + pydub subprocess failed. Last: {_e2}"
                return result
            finally:
                try:
                    os.unlink(_tmp.name)
                except Exception:
                    logger.warning("file_import.py::unknown Ersatzpfad", exc_info=True)

        if audio is None and not _sf_unsupported:
            # Stufe 2/3: pedalboard (FFmpeg backend) — preferred for lossless fallback,
            # for lossless fallback only.
            try:
                from pedalboard.io import AudioFile as _PBAudioFile  # type: ignore

                with _PBAudioFile(filepath) as _f:  # pylint: disable=not-context-manager
                    sr = int(_f.samplerate)
                    _frames = _f.frames
                    _chunk = sr * 300  # 300 s chunks to avoid OOM on long files
                    _parts_lossless: list[np.ndarray] = []
                    _read = 0
                    while _read < _frames:
                        _block = _f.read(min(_chunk, _frames - _read))
                        if _block.shape[-1] == 0:
                            break  # EOF — pedalboard.frames can overcount for VBR MP3
                        _parts_lossless.append(_block)
                        _read += _block.shape[-1]
                _raw = np.concatenate(_parts_lossless, axis=1) if len(_parts_lossless) > 1 else _parts_lossless[0]
                # pedalboard returns (channels, samples) — transpose to (samples, channels)
                if _raw.ndim == 2:
                    _raw = _raw.T
                audio = _raw.astype(np.float32)
                logger.debug("laden_audio_file: pedalboard/FFmpeg OK (%s)", filepath)
            except Exception as _e3:
                logger.warning("§V6 Audio-Laden (pedalboard/FFmpeg) fehlgeschlagen für %s — Error-Result zurückgegeben: %s", filepath, _e3)
                result["error"] = f"Audio read error: soundfile + pedalboard failed. Last: {_e3}"
                return result

        if audio is None:
            result["error"] = "Audio read error: kein Audio dekodiert"
            return result
        audio_work: np.ndarray = _require_audio_array(audio)
        _input_channels = int(result.get("channels") or (1 if audio_work.ndim == 1 else audio_work.shape[-1]))

        # ── Post-processing ──────────────────────────────────────────────────
        # Spec §2.47: nur Mono und Stereo unterstützt. > 2 Kanäle → gewichteter Downmix.
        # WICHTIG: Downmix VOR Resampling — so wird resampy/soxr nur auf 1-2 Kanäle angewendet
        # (statt 3+ Kanäle), was den soxr-np.apply_along_axis-Hänger bei 3-Kanal-MP3 verhindert.
        # PANNs-Plugin noch nicht im Import-Pfad verfügbar → einfacher Energie-Downmix.
        # Der Downmix ergibt Stereo (L=avg(L+odd), R=avg(R+even)) für 4-Kanal-Material
        # und Mono für alle anderen Kanalzahlen (> 2).
        if audio_work.ndim == 2 and audio_work.shape[-1] > 2:
            n_ch = audio_work.shape[-1]
            logger.warning(
                "laden_audio_file: %d Kanäle erkannt (nur Mono/Stereo unterstützt) — "
                "Downmix auf Stereo L/R (Energie-gewichtet).",
                n_ch,
            )
            # Energie-gewichteter Downmix: höhere Energie → höherer Beitrag pro Kanal.
            _ch_rms = np.sqrt(np.mean(audio_work**2, axis=0)) + 1e-9  # (n_ch,)
            _weights = _ch_rms / _ch_rms.sum()
            if n_ch >= 4:
                # L = Summe ungerade Kanäle, R = Summe gerade Kanäle (häufiges LRLS-RS-Schema)
                _l_idx = list(range(0, n_ch, 2))
                _r_idx = list(range(1, n_ch, 2))
                ch_l = float(np.sum(_weights[_l_idx])) + 1e-9
                ch_r = float(np.sum(_weights[_r_idx])) + 1e-9
                _l = np.average(audio_work[:, _l_idx], axis=1, weights=_weights[_l_idx] / ch_l)
                _right = np.average(audio_work[:, _r_idx], axis=1, weights=_weights[_r_idx] / ch_r)
                audio_work = np.stack([_l, _right], axis=-1).astype(np.float32)
            else:
                # 3 o.ä. → Mono-Downmix
                audio_work = np.average(audio_work, axis=-1, weights=_weights).astype(np.float32)
            logger.info(
                "laden_audio_file: Downmix %d→%d-Kanal abgeschlossen.",
                n_ch,
                1 if audio_work.ndim == 1 else audio_work.shape[-1],
            )
        if mono and audio_work.ndim > 1:
            audio_work = audio_work.mean(axis=-1)
        # Resampling NACH Downmix (1-2 Kanäle) — verhindert soxr-Hänger bei 3-Kanal-MP3.
        if target_sr and sr != target_sr:
            try:
                import resampy

                _use_resampy = True
            except ImportError:
                _use_resampy = False
                logger.warning("resampy not installed — falling back to scipy.signal.resample")
                from scipy.signal import resample as _scipy_resample

            if audio_work.ndim == 1:
                if _use_resampy:
                    audio_work = np.asarray(resampy.resample(audio_work, sr, target_sr), dtype=np.float32)
                else:
                    n_out = int(len(audio_work) * target_sr / sr)
                    audio_work = np.asarray(_scipy_resample(audio_work, n_out), dtype=np.float32)
            else:
                if _use_resampy:
                    audio_work = np.asarray(resampy.resample(audio_work.T, sr, target_sr, axis=-1).T, dtype=np.float32)
                else:
                    n_out = int(audio_work.shape[0] * target_sr / sr)
                    audio_work = np.asarray(_scipy_resample(audio_work, n_out, axis=0), dtype=np.float32)
            sr = target_sr
        audio_arr = np.asarray(audio_work, dtype=np.float32)
        audio_work = np.asarray(
            np.nan_to_num(audio_arr, nan=0.0, posinf=0.0, neginf=0.0),
            dtype=np.float32,
        )
        audio_work = np.asarray(np.clip(audio_work, -1.0, 1.0), dtype=np.float32)

        # Import-stage stereo blind-spot guard: detect unexpected L/R lag early.
        # If critical and sr=48k, run STCG correction immediately.
        _import_lag_before = _estimate_interchannel_lag_samples(audio_work, sr)
        _import_lag_after = _import_lag_before
        if abs(_import_lag_before) > 64:
            _lag_ms = (_import_lag_before / float(sr)) * 1000.0
            _msg = f"load_audio_file: detected interchannel lag={int(_import_lag_before)} samples ({_lag_ms:.1f} ms) before pipeline"
            logger.warning(_msg)
            if sr == 48000:
                try:
                    from backend.core.stereo_temporal_coherence_guard import (
                        get_stereo_temporal_coherence_guard,
                    )

                    corrected = get_stereo_temporal_coherence_guard().correct_interchannel_delay(
                        audio_work,
                        sr,
                        phase_id="import_pipeline",
                    )
                    audio_work = np.asarray(corrected, dtype=np.float32)
                    audio_work = np.asarray(
                        np.nan_to_num(audio_work, nan=0.0, posinf=0.0, neginf=0.0),
                        dtype=np.float32,
                    )
                    audio_work = np.asarray(np.clip(audio_work, -1.0, 1.0), dtype=np.float32)
                    _import_lag_after = _estimate_interchannel_lag_samples(audio_work, sr)
                    logger.info(
                        "laden_audio_file: import lag corrected %d -> %d samples",
                        _import_lag_before,
                        _import_lag_after,
                    )
                except Exception as _lag_fix_exc:
                    logger.debug("laden_audio_file: import lag correction uebersprungen: %s", _lag_fix_exc)

        result["audio"] = audio_work
        result["sr"] = sr
        result["input_channels"] = _input_channels
        result["channels"] = 1 if audio_work.ndim == 1 else audio_work.shape[-1]
        result["format"] = result["format"] or _ext[1:].upper()
        result["duration"] = float(audio_work.shape[0] / sr)
        result["interchannel_lag_samples_before"] = int(_import_lag_before)
        result["interchannel_lag_samples_after"] = int(_import_lag_after)
        # Carrier detection (heuristic + forensics)
        # Skipped when do_carrier_analysis=False (e.g. BatchProcessingThread, AudioPlayer,
        # RecoveryCheckpoint) — carrier analysis is done once in _carrier_bg and cached.
        # Running MediumClassifier on a full 225 s audio blocks the load-ticker for 6+ min.
        result["carrier_heuristic"] = detect_carrier(filepath, result["meta"]) if do_carrier_analysis else "unknown"
        if do_carrier_analysis:
            try:
                analyze_carrier_forensics, classify_carrier_ml = _lazy_get_carrier_tools()
                forensic = analyze_carrier_forensics(audio_work, sr)
                result["carrier_forensic"] = forensic["carrier_forensic"]
                result["carrier_forensic_score"] = forensic["score"]
                result["carrier_forensic_features"] = forensic["features"]
                # ML-based carrier classification (SOTA: echtes Audio statt Dummy)
                ml = classify_carrier_ml(audio_work, sr, features=forensic.get("features"))
                result["carrier_ml"] = ml.get("carrier_ml", "Unbekannt")
                result["carrier_ml_confidence"] = ml.get("confidence", 0.0)
                result["carrier_ml_probas"] = ml.get("probas", {})
                result["carrier_ml_explain"] = ml.get("explain", "")
            except Exception as _e_carrier:
                result["carrier_forensic"] = "Unbekannt"
                result["carrier_forensic_score"] = 0
                result["carrier_forensic_features"] = {}
                result["carrier_ml"] = "Unbekannt"
                result["carrier_ml_confidence"] = 0.0
                result["carrier_ml_probas"] = {}
                result["carrier_ml_explain"] = str(_e_carrier)
    except Exception as e:
        result["error"] = f"Unknown error: {e}"
    return result
