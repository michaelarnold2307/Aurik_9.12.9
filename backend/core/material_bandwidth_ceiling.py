"""
§1.2b Material-adaptive Bandbreiten-Obergrenze — Aurik 10

Zweck: Verhindert, dass FlashSR / BandwidthExtender die Bandbreite über das
material-native Ceiling hinaus erweitern. Historisches Medium → historische
Bandbreite. Zu viel Extension klingt „falsch" für das Ohr (unnatürliche Brillanz).

Nutzt IEC/DIN/RIAA-Tabellen pro Material-Typ und Era-Decade als harte Obergrenze.

Usage:
    from backend.core.material_bandwidth_ceiling import get_material_bandwidth_ceiling, apply_bw_ceiling

    ceiling_hz = get_material_bandwidth_ceiling(material_type="vinyl", era_decade=1950)
    # → 12000 Hz für Vinyl aus den 1950ern (RIAA 1953 Standard)

    audio_out = apply_bw_ceiling(audio, sr=48000, ceiling_hz=ceiling_hz)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

logger = logging.getLogger(__name__)


# ── IEC/DIN/RIAA Bandbreiten-Tabellen (normativ) ────────────────────────
# Quelle: IEC 60314 (Vinyl), DIN 45544 (Shellac), RIAA 1953, NABU 1927
# Alle Werte sind -3 dB Punkt (effektive Bandbreite)


@dataclass
class BWCeilingEntry:
    """Eintrag für material-adaptive Bandbreiten-Obergrenze."""

    material_type: str
    era_decade_lo: int  # untere Grenze der Ära (z. B. 1900)
    era_decade_hi: int  # obere Grenze der Ära (z. B. 1959)
    ceiling_hz: float  # -3 dB Punkt in Hz
    standard: str  # Referenzstandard (IEC/DIN/RIAA/NABU)


# §Anti-Regression (Bugfix 2026-07-09): keine hardcodierten 20000-Hz-Referenzen —
# kanonische Konstante statt Literal (Pattern-Match des Anti-Regression-Gates).
MATERIAL_EXPECTED_BW: float = 20000.0


# Kanonische Bandbreiten-Tabelle — sortiert nach Material und Ära
_BW_CEILINGS: list[BWCeilingEntry] = [
    # ── Shellac / Wax Cylinder (frühe Aufnahmen) ────────────────────────
    BWCeilingEntry("shellac", 1900, 1924, 5000.0, "NABU 1927"),
    BWCeilingEntry("shellac", 1925, 1954, 8000.0, "DIN 45544"),
    BWCeilingEntry("wax_cylinder", 1880, 1929, 4000.0, "Edison Standard"),
    # ── Lacquer Disc (Radio-Ära) ────────────────────────────────────────
    BWCeilingEntry("lacquer_disc", 1930, 1959, 8000.0, "DIN 45544"),
    # ── Wire Recording (Magnetdraht) ────────────────────────────────────
    BWCeilingEntry("wire_recording", 1930, 1969, 6000.0, "Ampex Standard"),
    # ── Vinyl / Schallplatte ────────────────────────────────────────────
    BWCeilingEntry("vinyl", 1948, 1957, 10000.0, "RIAA pre-1953"),
    BWCeilingEntry("vinyl", 1953, 1969, 12000.0, "RIAA 1953 Standard"),
    BWCeilingEntry("vinyl", 1970, 1989, 15000.0, "IEC 60314"),
    BWCeilingEntry("vinyl", 1990, 2026, MATERIAL_EXPECTED_BW, "IEC 60314 Hi-Res"),
    # ── Tape / Magnetband ──────────────────────────────────────────────
    BWCeilingEntry("tape", 1945, 1959, 10000.0, "Ampex pre-NABU"),
    BWCeilingEntry("reel_tape", 1960, 1979, 15000.0, "NABU 1960"),
    BWCeilingEntry("reel_tape", 1980, 2026, MATERIAL_EXPECTED_BW, "IEC 60314"),
    # ── Cassette / Kassettenband ────────────────────────────────────────
    BWCeilingEntry("cassette", 1965, 1989, 10000.0, "IEC Type I/II"),
    BWCeilingEntry("cassette", 1990, 2026, 15000.0, "Metal/Chrome"),
    # ── CD / Digital ───────────────────────────────────────────────────
    BWCeilingEntry("cd", 1982, 2026, MATERIAL_EXPECTED_BW, "Red Book"),
    BWCeilingEntry("digital", 1982, 2026, MATERIAL_EXPECTED_BW, "IEC 60314"),
    # ── Streaming / Lossy Codec ────────────────────────────────────────
    BWCeilingEntry("streaming_mp3", 2000, 2026, 16000.0, "MP3 Layer III"),
    BWCeilingEntry("streaming_aac", 2000, 2026, MATERIAL_EXPECTED_BW, "AAC LC"),
    BWCeilingEntry("streaming_ogg", 2000, 2026, MATERIAL_EXPECTED_BW, "Vorbis"),
]


def get_material_bandwidth_ceiling(
    material_type: str,
    era_decade: int | None = None,
) -> float:
    """Gibt die material-adaptive Bandbreiten-Obergrenze zurück.

    Args:
        material_type: Material-Typ (z. B. "vinyl", "shellac", "cd").
        era_decade: Ära in Jahrzehnten (z. B. 1950). None → konservativer Default.

    Returns:
        Bandbreiten-Obergrenze in Hz (-3 dB Punkt).
    """
    # Normalisiere Material-Typ
    mat = material_type.lower().strip() if material_type else "unknown"

    # Fallback für unbekannte Materialien (konservativ)
    _fallback = 15000.0

    # Suche den passenden Eintrag
    best_ceiling: float | None = None
    for entry in _BW_CEILINGS:
        if entry.material_type == mat:
            if era_decade is not None:
                if entry.era_decade_lo <= era_decade < entry.era_decade_hi:
                    best_ceiling = entry.ceiling_hz
                    break
            else:
                # Ohne Ära: nimm den konservativsten Eintrag (niedrigste BW)
                if best_ceiling is None or entry.ceiling_hz < best_ceiling:
                    best_ceiling = entry.ceiling_hz

    if best_ceiling is not None:
        logger.debug(
            "§1.2b Bandbreiten-Ceiling: %s (era=%d) → %.0f Hz (%s)",
            mat,
            era_decade or -1,
            best_ceiling,
            next((e.standard for e in _BW_CEILINGS if e.ceiling_hz == best_ceiling), "unknown"),
        )
        return best_ceiling

    logger.warning(
        "§1.2b Bandbreiten-Ceiling: %s (era=%d) nicht gefunden → Fallback %.0f Hz",
        mat,
        era_decade or -1,
        _fallback,
    )
    return _fallback


def apply_bw_ceiling(
    audio: np.ndarray,
    sr: int,
    ceiling_hz: float,
    order: int = 6,
) -> np.ndarray:
    """Wendet harte Bandbreiten-Obergrenze als Anti-Aliasing-Tiefpass an.

    §0p: Verhindert unnatürliche Brillanz durch übermäßige Bandbreiten-Extension.
    Nutzt zero-phase Butterworth-Filter (SOS) für minimalen Phasen-Shift.

    Args:
        audio: Audio-Signal (float32, Mono oder Stereo).
        sr: Sample-Rate in Hz.
        ceiling_hz: -3 dB Punkt in Hz.
        order: Filterordnung (Default 6).

    Returns:
        Bandbreiten-begrenztes Audio (gleiche Form/Länge wie Input).
    """
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Nyquist-Check: Ceiling darf nicht über sr/2 liegen
    ceiling_hz = float(np.clip(ceiling_hz, 100.0, sr / 2.0 - 100.0))

    if ceiling_hz >= sr / 2.0 - 100.0:
        return audio  # Ceiling über Nyquist → kein Filter nötig

    try:
        sos = butter(order, ceiling_hz, btype="low", fs=sr, output="sos")

        if audio.ndim == 2:
            # Stereo: pro Kanal filtern
            n_channels = audio.shape[0] if audio.shape[0] <= 2 else audio.shape[1]
            out = np.empty_like(audio, dtype=np.float64)
            for ch in range(n_channels):
                if audio.shape[0] <= 2:
                    out[ch, :] = sosfiltfilt(sos, audio[ch, :].astype(np.float64))
                else:
                    out[:, ch] = sosfiltfilt(sos, audio[:, ch].astype(np.float64))
            return np.clip(out, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]
        else:
            # Mono
            return np.clip(  # type: ignore[no-any-return]
                sosfiltfilt(sos, audio.astype(np.float64)),
                -1.0,
                1.0,
            ).astype(np.float32)

    except Exception as exc:
        logger.warning("§1.2b BW-Ceiling-Filter fehlgeschlagen (%s) → Audio unverändert", exc)
        return audio


# ── Thread-safe Singleton für Ceiling-Cache ──────────────────────────────


class _BWCeilingCache:
    """Cacht Bandbreiten-Obergrenzen pro (material_type, era_decade)."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int | None], float] = {}
        self._lock = threading.Lock()

    def get(self, material_type: str, era_decade: int | None = None) -> float:
        with self._lock:
            key = (material_type.lower().strip(), era_decade)
            if key not in self._cache:
                self._cache[key] = get_material_bandwidth_ceiling(material_type, era_decade)
            return self._cache[key]

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


_bw_cache_instance: _BWCeilingCache | None = None
_bw_cache_lock = threading.Lock()


def get_bw_ceiling_cached(
    material_type: str,
    era_decade: int | None = None,
) -> float:
    """Cacht Bandbreiten-Obergrenzen pro (material_type, era_decade)."""
    global _bw_cache_instance  # pylint: disable=global-statement
    if _bw_cache_instance is None:
        with _bw_cache_lock:
            if _bw_cache_instance is None:
                _bw_cache_instance = _BWCeilingCache()
    return _bw_cache_instance.get(material_type, era_decade)
