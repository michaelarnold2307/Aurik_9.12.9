"""
§silence-guarantee [RELEASE_MUST] SilenceMask — backend/core/dsp/silence_mask.py

Berechnet eine per-Song-Silence-Maske aus dem Original-Audio (vor jeder Restaurierung).
Die Maske klassifiziert jeden Sample als "aktiv" (1.0) oder "Stille" (0.0) mit sanften
Übergangsrampen (20 ms), sodass Stilleabschnitte beim Blending exakt das Original
behalten und Übergänge artefaktfrei klingen.

Zweck:
    Verhindert, dass Content-Injection-Phasen (Inpainting, Dropout-Repair,
    AR-Interpolation) in gewollten Stilleregionen Pegelexplosionen erzeugen.
    Eine einzige Berechnung pre-Pipeline genügt; die Maske wird in
    _restoration_context["silence_mask"] hinterlegt und von UV3 post-Phase
    via apply_silence_preservation() angewandt.

Breath-Schutz (§2.46f):
    Atemgeräusche (-55 bis -38 dBFS, spectral_flatness > 0.4) sind keine Stille
    und werden NICHT maskiert — sie sind ein Natürlichkeitsmarker.

Singleton:
    get_silence_mask_computer() liefert eine thread-sichere Instanz.

Author: Aurik 10.0.0 Engineering
Version: 1.0.0 (§silence-guarantee RELEASE_MUST, v10.0.0)
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Material-adaptive Silence-Schwellen (dBFS RMS per 10ms-Frame) ──────────────
# Unterhalb dieser Grenze gilt ein Frame als "Stille".
# Werte = typischer Rauschboden des Trägers + 4–5 dB Sicherheitsabstand.
_MATERIAL_SILENCE_DB: dict[str, float] = {
    "shellac": -22.0,  # Shellac-Rauschen ~-20 dBFS → Stille < -22
    "wax_cylinder": -20.0,  # Wachswalze: sehr hohes Grundrauschen
    "acoustic_78": -22.0,  # Akustische 78rpm-Aufnahme
    "wire_recording": -22.0,  # Drahtaufnahme
    "lacquer_disc": -28.0,  # Lackfolie
    "vinyl": -33.0,  # Vinyl-Rauschen ~-30 dBFS
    "reel_tape": -40.0,  # Bandmaschine ~-36 dBFS
    "tape": -38.0,  # Tonband allgemein
    "cassette": -43.0,  # Kassette ~-38 dBFS
    "cd_digital": -57.0,  # CD: quantisierungsrausch bei -93 dBFS
    "dat": -57.0,  # DAT ähnlich CD
    "mp3_high": -52.0,  # MP3 320 kbps
    "mp3_medium": -52.0,
    "mp3_low": -48.0,  # MP3 64–128 kbps: pre-echo floor erhöht Stille
    "aac": -52.0,
    "minidisc": -48.0,
    "streaming": -54.0,
    "unknown": -45.0,
}

# Default für unbekannte Materialien
_DEFAULT_SILENCE_DB = -45.0

# Mindestdauer einer Stilleregion (Sekunden) — verhindert, dass kurze Pausen maskiert werden
_MIN_SILENCE_DURATION_S = 0.20  # 200 ms

# Übergangsrampe an Stille-Grenzen (Sekunden) — artefaktfreier Blend
_CROSSFADE_DURATION_S = 0.020  # 20 ms

# Breath-Schutz §2.46f: Spectral Flatness über diesem Wert + RMS in [_BREATH_FLOOR, _BREATH_CEIL]
# → Frame ist Atemgeräusch, KEINE Stille
_BREATH_FLATNESS_MIN = 0.40
_BREATH_RMS_FLOOR_DB = -57.0  # 2 dB unter Stille-Minimum für digitale Träger
_BREATH_RMS_CEIL_DB = -38.0  # Atemgeräusche selten lauter als -38 dBFS


def compute_silence_mask(
    audio: np.ndarray,
    sr: int,
    *,
    material_key: str | None = None,
    min_duration_s: float = _MIN_SILENCE_DURATION_S,
    crossfade_s: float = _CROSSFADE_DURATION_S,
    protect_breath: bool = True,
) -> np.ndarray:
    """Berechnet eine Sample-genaue Silence-Maske aus dem Original-Audio.

    Returns:
        float32 ndarray mit Shape (N,), Werte 0.0 (Stille) … 1.0 (aktiv).
        Übergänge: sanfte lineare Rampe über crossfade_s an Stille-Grenzen.

    Semantik der Maske:
        0.0 → Stilleregion: Original-Audio exakt bewahren
        1.0 → Aktive Region: prozessiertes Audio verwenden
        0.0…1.0 → Übergangszone: Blend zwischen beiden

    Args:
        audio:         Original-Audio (mono oder stereo, beliebiges Layout).
        sr:            Abtastrate in Hz.
        material_key:  Materialtyp für adaptive Schwelle (z. B. "vinyl", "cassette").
        min_duration_s: Mindestlänge einer Stilleregion in Sekunden.
        crossfade_s:   Rampenlänge an Stille-Übergängen in Sekunden.
        protect_breath: §2.46f — Breath-Frames aus Stille-Maske ausschließen.
    """
    try:
        _mat = (material_key or "unknown").lower().replace("-", "_").replace(" ", "_")
        _threshold_db = float(_MATERIAL_SILENCE_DB.get(_mat, _DEFAULT_SILENCE_DB))

        # Mono-Referenz für Energie-Analyse
        _ref = np.asarray(audio, dtype=np.float32)
        if _ref.ndim == 2:
            # Channels-first (2, N) oder channels-last (N, 2)
            if _ref.shape[0] == 2 and _ref.shape[1] > 2:
                _mono = np.mean(_ref, axis=0)
            elif _ref.shape[1] == 2:
                _mono = np.mean(_ref, axis=1)
            else:
                _mono = np.mean(_ref, axis=0)
        else:
            _mono = _ref.copy()

        _n_total = int(_mono.shape[0])
        if _n_total < 1:
            return np.ones(0, dtype=np.float32)  # type: ignore[no-any-return]

        # Frame-Einstellungen (10 ms)
        _frame_n = max(240, int(sr * 0.010))

        # Threshold in linearer Leistung
        float(10.0 ** (_threshold_db / 10.0))

        # ── Frame-weise RMS berechnen ──────────────────────────────────────────
        _n_frames = max(1, _n_total // _frame_n)
        _frame_rms_db = np.full(_n_frames, -120.0, dtype=np.float64)
        _frame_flatness = np.zeros(_n_frames, dtype=np.float64)  # für Breath-Schutz

        for _k in range(_n_frames):
            _s = _k * _frame_n
            _e = min(_s + _frame_n, _n_total)
            _f = _mono[_s:_e].astype(np.float64)
            _power = float(np.mean(_f**2))
            _frame_rms_db[_k] = 10.0 * np.log10(_power + 1e-24)

            if protect_breath and _f.size >= 16:
                # Spectral Flatness via Fourier-Spektrum (Wiener-Entropie)
                _spec = np.abs(np.fft.rfft(_f * np.hanning(len(_f))))[1:]  # DC entfernen
                _spec = np.maximum(_spec, 1e-24)
                _geo_mean = float(np.exp(np.mean(np.log(_spec))))
                _arith_mean = float(np.mean(_spec))
                if _arith_mean > 0:
                    _frame_flatness[_k] = _geo_mean / _arith_mean

        # ── Stille-Klassifikation pro Frame ────────────────────────────────────
        _is_silence = _frame_rms_db < _threshold_db  # True = Stille

        if protect_breath:
            # §2.46f: Frames mit Atem-Signatur von Stille ausnehmen.
            # WICHTIG: Breath-Schutz nur bei DIGITALEN Trägern aktiv — auf analogen
            # Trägern (Vinyl, Shellac, Tape, Kassette) ist breitbandiges Rauschen auf
            # Atemgeräusch-Pegel einfach Trägerrauschen, KEIN Atemgeräusch.
            # Auf analogen Trägern würde die Flatness-Prüfung das Trägerrauschen
            # fälschlicherweise als Atem klassifizieren und korrekte Stille-Erkennung
            # verhindern.
            _DIGITAL_MATERIALS = frozenset(
                {
                    "cd_digital",
                    "dat",
                    "mp3_high",
                    "mp3_medium",
                    "mp3_low",
                    "aac",
                    "minidisc",
                    "streaming",
                }
            )
            _is_digital = _mat in _DIGITAL_MATERIALS
            if _is_digital and np.any(_frame_flatness > 0):
                _is_breath = (
                    (_frame_flatness >= _BREATH_FLATNESS_MIN)
                    & (_frame_rms_db >= _BREATH_RMS_FLOOR_DB)
                    & (_frame_rms_db < _BREATH_RMS_CEIL_DB)
                )
                _is_silence = _is_silence & ~_is_breath

        # ── Mindestlänge: nur zusammenhängende Stilleregionen ≥ min_duration_s ─
        _min_frames = max(1, int(min_duration_s * sr / _frame_n))
        _filtered_silence = _is_silence.copy()

        # RLE-Scan: Stille-Regionen kürzer als _min_frames werden rückgesetzt
        _k = 0
        while _k < _n_frames:
            if _is_silence[_k]:
                _start = _k
                while _k < _n_frames and _is_silence[_k]:
                    _k += 1
                _run = _k - _start
                if _run < _min_frames:
                    _filtered_silence[_start:_k] = False
            else:
                _k += 1

        # ── Sample-genaue Maske aus Frame-Klassifikation ────────────────────────
        # Stille = 0.0, aktiv = 1.0 (in Ganzzahl-Auflösung der Frame-Raster)
        _mask_samples = np.ones(_n_total, dtype=np.float32)
        for _k in range(_n_frames):
            if _filtered_silence[_k]:
                _s = _k * _frame_n
                _e = min(_s + _frame_n, _n_total)
                _mask_samples[_s:_e] = 0.0

        # Restliche Samples nach dem letzten Frame (ggf. kürzer) erben den
        # Klassifikationsstatus des letzten Frames
        _last_frame_start = (_n_frames - 1) * _frame_n
        if _n_frames > 0 and _filtered_silence[_n_frames - 1]:
            _mask_samples[_last_frame_start:] = 0.0

        # ── Crossfade-Rampen an Stille-Grenzen ─────────────────────────────────
        _xfade_n = max(2, int(crossfade_s * sr))
        _mask_smooth = _mask_samples.copy()

        # Übergänge finden: 0→1 (Stille→Aktiv) und 1→0 (Aktiv→Stille)
        _diff = np.diff(_mask_samples.astype(np.int8))  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
        _rise = np.where(_diff > 0)[0] + 1  # Stille→Aktiv: Index des ersten aktiven Samples
        _fall = np.where(_diff < 0)[0] + 1  # Aktiv→Stille: Index des ersten stillen Samples

        # RISE: letzte _xfade_n Stille-Samples VOR aktiver Zone → Einblende-Rampe 0→1
        for _idx in _rise:
            _s = max(0, _idx - _xfade_n)
            _e = _idx  # exclusive — all these samples are currently 0 (silence)
            _fade_len = _e - _s
            if _fade_len > 0:
                # linspace(0, 1, fade_len+1)[:-1] → aufsteigend, endet knapp unter 1.0
                _mask_smooth[_s:_e] = np.linspace(0.0, 1.0, _fade_len + 1, dtype=np.float32)[:_fade_len]

        # FALL: erste _xfade_n Stille-Samples NACH aktiver Zone → Ausblende-Rampe 1→0
        for _idx in _fall:
            _s = _idx
            _e = min(_n_total, _idx + _xfade_n)
            _fade_len = _e - _s
            if _fade_len > 0:
                # linspace(1, 0, fade_len+1)[1:] → absteigend, startet knapp unter 1.0
                _mask_smooth[_s:_e] = np.linspace(1.0, 0.0, _fade_len + 1, dtype=np.float32)[1 : _fade_len + 1]

        # Numerische Sicherheit
        _mask_smooth = np.clip(np.nan_to_num(_mask_smooth, nan=1.0), 0.0, 1.0)

        _n_silence_frames = int(np.sum(_filtered_silence))
        _total_dur_s = _n_total / float(sr)
        _silence_dur_s = _n_silence_frames * (_frame_n / float(sr))
        logger.info(
            "silence_mask: mat=%s thr=%.1f dBFS silence=%.2f s / %.2f s (%.0f%%)",
            _mat,
            _threshold_db,
            _silence_dur_s,
            _total_dur_s,
            100.0 * _silence_dur_s / max(_total_dur_s, 1e-3),
        )
        return _mask_smooth.astype(np.float32)  # type: ignore[no-any-return]

    except Exception:  # pragma: no cover
        logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
        logger.exception("silence_mask: Fehler bei compute_silence_mask — Fallback: alle aktiv")
        _n_fb = int(np.asarray(audio).shape[-1]) if np.asarray(audio).ndim >= 1 else 0
        return np.ones(max(0, _n_fb), dtype=np.float32)  # type: ignore[no-any-return]


def apply_silence_preservation(
    original: np.ndarray,
    processed: np.ndarray,
    silence_mask: np.ndarray,
) -> np.ndarray:
    """Stellt Stilleregionen exakt aus dem Original wieder her.

    Blended processed und original gemäß der Silence-Maske:
        result = (1 - mask) * original + mask * processed

    Dadurch gilt:
        - Stille (mask=0.0): Ergebnis = Original (keine Pegelexplosion möglich)
        - Aktiv (mask=1.0): Ergebnis = Prozessiert
        - Übergang (0…1): kontinuierlicher Blend

    Das Ergebnis klingt "als wäre nie eingegriffen worden" in Stilleregionen.

    Args:
        original:      Unbearbeitetes Referenz-Audio (vor Pipeline).
        processed:     Prozessiertes Audio nach Phase.
        silence_mask:  Maske aus compute_silence_mask() (Sample-genau, 1D float32).

    Returns:
        float32 ndarray, gleiche Shape wie processed.
    """
    try:
        _orig = np.asarray(original, dtype=np.float32)
        _proc = np.asarray(processed, dtype=np.float32)
        _mask = np.asarray(silence_mask, dtype=np.float32)

        # §v10.30: Normalisiere auf gleiches Channel-Layout bevor shape[-1]
        # verwendet wird. shape[-1] ist die Kanal-Anzahl bei channels-last (N,2)
        # aber die Sample-Anzahl bei channels-first (2,N). Ohne Normalisierung
        # wird _n = min(orig.shape[-1], proc.shape[-1]) = min(2, 10815948) = 2,
        # was zu broadcasting-Fehlern und Datenkorruption führt.
        def _to_channels_first(arr: np.ndarray) -> np.ndarray:
            if arr.ndim == 2 and arr.shape[-1] == 2 and arr.shape[0] > 2:
                return arr.T.copy()  # (N, 2) → (2, N)
            return arr

        _orig = _to_channels_first(_orig)
        _proc = _to_channels_first(_proc)

        # Längenabgleich — §v10.14.1: Maske kann 1D (N,) oder 2D (2,N) sein.
        # Bei 2D-Maske beide Kanäle gleich behandeln → erste Zeile nehmen.
        # §Bugfix: Validiere Mask-Shape vor Extraktion — fehlerhafte Upstream-Masken
        # können 0-d oder unerwartete Shapes haben (§v10.15).
        if _mask.ndim == 0 or _mask.size == 0:
            logger.warning("silence_mask: empty/scalar mask received, returning proc unchanged")
            return _proc
        if _mask.ndim == 1:
            _mask_1d = _mask
        elif _mask.ndim == 2:
            if _mask.shape[0] <= 8:
                _mask_1d = _mask[0, :]
            else:
                _mask_1d = _mask[:, 0]
        else:
            logger.warning("silence_mask: mask ndim=%d unexpected, returning proc unchanged", _mask.ndim)
            return _proc
        if _mask_1d.ndim != 1:
            logger.warning("silence_mask: mask_1d ndim=%d after extraction, returning proc unchanged", _mask_1d.ndim)
            return _proc
        _n = min(_orig.shape[-1], _proc.shape[-1], _mask_1d.shape[0])
        if _n <= 0:
            return _proc  # type: ignore[no-any-return]

        # Maske auf Audio-Shape erweitern (Stereo-Support)
        if _proc.ndim == 2:
            # Stereo: Maske als Zeile (1, N) oder Spalte (N, 1)
            if _proc.shape[0] == 2 and _proc.shape[1] > 2:
                # (2, N) channels-first
                _m = _mask_1d[:_n][np.newaxis, :]
                _o = _orig[..., :_n]
                _p = _proc[..., :_n]
            else:
                # (N, 2) channels-last
                _m = _mask_1d[:_n][:, np.newaxis]
                _o = _orig[:_n]
                _p = _proc[:_n]
        else:
            _m = _mask_1d[:_n]
            _o = _orig[:_n]
            _p = _proc[:_n]

        # Blend: original in Stille, processed in aktiven Regionen
        _result = (1.0 - _m) * _o + _m * _p
        _result = np.clip(np.nan_to_num(_result, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)

        # Sicherstellen, dass die ursprüngliche Shape erhalten bleibt
        if _proc.ndim == 2:
            if _proc.shape[0] == 2 and _proc.shape[1] > 2:
                _out = _proc.copy()
                _out[..., :_n] = _result
            else:
                _out = _proc.copy()
                _out[:_n] = _result
        else:
            _out = _proc.copy()
            _out[:_n] = _result

        return _out.astype(np.float32)  # type: ignore[no-any-return]
    except Exception:  # pragma: no cover
        logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
        logger.exception("silence_mask: Fehler bei apply_silence_preservation — Fallback: unverändert")
        return np.asarray(processed, dtype=np.float32)  # type: ignore[no-any-return]


# ── Singleton ──────────────────────────────────────────────────────────────────


class SilenceMaskComputer:
    """Thread-sichere Klasse für Silence-Mask-Berechnungen."""

    def compute(
        self,
        audio: np.ndarray,
        sr: int,
        material_key: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Wrapper für compute_silence_mask()."""
        return compute_silence_mask(audio, sr, material_key=material_key, **kwargs)

    def apply(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        silence_mask: np.ndarray,
    ) -> np.ndarray:
        """Wrapper für apply_silence_preservation()."""
        return apply_silence_preservation(original, processed, silence_mask)


_instance: SilenceMaskComputer | None = None
_lock = threading.Lock()


def get_silence_mask_computer() -> SilenceMaskComputer:
    """Gibt die thread-sichere Singleton-Instanz zurück."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = SilenceMaskComputer()
    return _instance
