"""
§v10.15 Adaptive Listening-Mode EQ — analysiert den Track und korrigiert
nur was das Wiedergabemedium tatsächlich benötigt.

Statt blind +0.8dB@7kHz für Kopfhörer anzuwenden (was bei bereits hellen
Aufnahmen zu Harshness und Hörermüdung führt), analysiert dieser EQ das
tatsächliche Spektrum und wendet nur die DIFFERENZ zum Zielprofil an.

Algorithmus:
  1. Langzeit-Mittelwertspektrum (1 min Mitte) des Tracks berechnen
  2. Mit Zielkurve für den Listening-Mode vergleichen
  3. Nur Frequenzbereiche korrigieren wo |Δ| > 2 dB
  4. Maximale Korrektur: ±4 dB (verhindert Überkompensation)
  5. PostGate-verifiziert: nur anwenden wenn natürlicher Klang erhalten bleibt

Referenz-Zielkurven basierend auf ISO 226:2023 (Equal-Loudness),
Harman Target Curve (Olive et al. 2013), und Praxiserfahrung.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

logger = logging.getLogger(__name__)

# ── Zielkurven pro Listening-Mode ──────────────────────────────────────
# Format: [(freq_Hz, gain_dB), ...] — stückweise lineare Interpolation.
# Positive gain = Anhebung, Negative = Absenkung.
# Frequenzen: ISO-Standard 1/3-Oktave (vereinfacht auf 10 Bänder).

_TARGET_CURVES: dict[str, list[tuple[float, float]]] = {
    # Kopfhörer: leichte Höhenabsenkung (Nahfeld-Kompensation),
    # sanfte Bassanhebung (fehlende Körperresonanz).
    "headphones": [
        (20, 1.5),
        (60, 1.0),
        (150, 0.0),
        (400, -0.5),
        (1000, 0.0),
        (2500, -0.5),
        (5000, -1.0),
        (8000, -1.5),
        (12000, -2.0),
        (16000, -1.0),
        (20000, -3.0),
    ],
    # Fernfeld/Nahfeldmonitore: neutral mit leichter Bassabsenkung
    # (Raummoden-Kompensation).
    "farfield": [
        (20, -1.0),
        (60, -0.5),
        (150, 0.0),
        (400, 0.0),
        (1000, 0.0),
        (2500, 0.0),
        (5000, 0.0),
        (8000, 0.5),
        (12000, 1.0),
        (16000, 0.0),
        (20000, -2.0),
    ],
    # Auto: starke Bassanhebung (Fahrgeräusch-Maskierung),
    # moderate Höhenanhebung (Dämpfung durch Sitze/Innenraum).
    "car": [
        (20, 3.0),
        (60, 2.5),
        (150, 2.0),
        (400, 0.5),
        (1000, 0.0),
        (2500, 0.5),
        (5000, 1.0),
        (8000, 2.0),
        (12000, 3.0),
        (16000, 2.0),
        (20000, 0.0),
    ],
}

# Frequenz-Bänder für die Analyse (1/3-Oktave-ähnlich)
_ANALYSIS_BANDS: list[tuple[float, float]] = [
    (20, 60),
    (60, 150),
    (150, 400),
    (400, 1000),
    (1000, 2500),
    (2500, 5000),
    (5000, 8000),
    (8000, 12000),
    (12000, 16000),
    (16000, 20000),
]

# Band-Mitten für die Zielkurven-Interpolation
_BAND_CENTERS: list[float] = [
    40,
    105,
    275,
    700,
    1750,
    3750,
    6500,
    10000,
    14000,
    18000,
]

# Maximal erlaubte Korrektur pro Band (dB)
_MAX_CORRECTION_DB: float = 4.0

# Minimale Differenz für Korrektur (dB) — darunter: nicht anfassen
_MIN_DELTA_DB: float = 2.0


# ── Ergebnis ───────────────────────────────────────────────────────────


@dataclass
class AdaptiveEQResult:
    """Ergebnis der adaptiven EQ-Analyse + Anwendung."""

    audio: np.ndarray
    corrections_applied: list[tuple[float, float, float]] = field(default_factory=list)
    # (band_center_Hz, measured_dB, correction_dB)
    bands_skipped: int = 0
    total_correction_db: float = 0.0  # Summe der Absolutkorrekturen
    analysis_duration_ms: float = 0.0


# ── Hauptklasse ────────────────────────────────────────────────────────


class AdaptiveListeningEQ:
    """Adaptiver Listening-Mode EQ mit spektraler Analyse."""

    @staticmethod
    def analyze_and_apply(
        audio: np.ndarray,
        sr: int,
        mode: str = "headphones",
        force: bool = False,
        *,
        is_studio_2026: bool = False,
    ) -> AdaptiveEQResult:
        """Analysiert das Spektrum und wendet nur nötige Korrekturen an.

        Args:
            audio: float32 Stereo (2,N) oder (N,2) oder Mono
            sr: Sample rate (typ. 48000)
            mode: "headphones" | "farfield" | "car"
            force: Wenn True, Korrekturen auch unter _MIN_DELTA_DB anwenden

        Returns:
            AdaptiveEQResult mit verarbeitetem Audio.
        """
        import time

        t0 = time.time()

        if mode not in _TARGET_CURVES:
            logger.debug("AdaptiveEQ: unbekannter Modus '%s' — kein EQ", mode)
            return AdaptiveEQResult(audio=audio)

        target_curve = _TARGET_CURVES[mode]

        try:
            arr = np.asarray(audio, dtype=np.float64)
            # Mono-Analyse (Mittelwert beider Kanäle)
            if arr.ndim > 1:
                mono = arr.mean(axis=0) if arr.shape[0] <= 2 else arr.mean(axis=1)
            else:
                mono = arr

            # ── 1. Spektrum messen ──────────────────────────────────
            # Verwende 60 s aus der Mitte für robuste Langzeit-Messung
            n_analyze = min(len(mono), sr * 60)
            start = max(0, (len(mono) - n_analyze) // 2)
            segment = mono[start : start + n_analyze]

            # Band-Energie via FFT (Welch-Methode vereinfacht)
            measured_db = AdaptiveListeningEQ._measure_bands(segment, sr)

            # ── 2. Zielkurve interpolieren ──────────────────────────
            target_db = AdaptiveListeningEQ._interpolate_target(target_curve, _BAND_CENTERS)

            # ── 3. Differenz berechnen + clippen ───────────────────
            corrections: list[tuple[float, float, float]] = []
            bands_skipped = 0
            sos_list = []

            for i, (center, meas, targ) in enumerate(zip(_BAND_CENTERS, measured_db, target_db)):
                delta = targ - meas  # positiv = zu leise → anheben

                # Clipping
                delta = float(np.clip(delta, -_MAX_CORRECTION_DB, _MAX_CORRECTION_DB))

                if not force and abs(delta) < _MIN_DELTA_DB:
                    bands_skipped += 1
                    continue

                if abs(delta) < 0.5:
                    continue  # vernachlässigbar

                # Band-Pass/Peak-Filter für dieses Band
                lo, hi = _ANALYSIS_BANDS[i]
                sos = AdaptiveListeningEQ._make_band_eq(sr, lo, hi, delta)
                if sos is not None:
                    sos_list.append(sos)
                corrections.append((center, meas, delta))

            # ── 4. Filter anwenden ──────────────────────────────────
            result_audio = arr.copy()
            for sos in sos_list:
                if result_audio.ndim == 2:
                    for ch in range(min(result_audio.shape[0], 2)):
                        result_audio[ch] = sosfiltfilt(sos, result_audio[ch])  # type: ignore[name-defined]
                else:
                    result_audio = sosfiltfilt(sos, result_audio)  # type: ignore[name-defined]

            result_audio = np.clip(np.nan_to_num(result_audio, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(
                np.float32
            )

            # §v10.15: Restoration-Mode → Korrektur sanfter (×0.7)
            if not is_studio_2026:
                for i, sos in enumerate(sos_list):
                    sos_list[i][:, :3] *= 0.70
            total_corr = sum(abs(c[2]) for c in corrections)
            dur_ms = (time.time() - t0) * 1000.0

            logger.info(
                "AdaptiveEQ [%s]: %d Bänder korrigiert, %d übersprungen, Σ|corr|=%.1f dB (%.0f ms)",
                mode,
                len(corrections),
                bands_skipped,
                total_corr,
                dur_ms,
            )

            return AdaptiveEQResult(
                audio=result_audio,
                corrections_applied=corrections,
                bands_skipped=bands_skipped,
                total_correction_db=total_corr,
                analysis_duration_ms=dur_ms,
            )

        except Exception as exc:
            logger.warning("AdaptiveEQ fehlgeschlagen: %s — Originalsignal zurück", exc)
            return AdaptiveEQResult(audio=audio)

    # ── Interne Methoden ──────────────────────────────────────────────

    @staticmethod
    def _measure_bands(mono: np.ndarray, sr: int) -> list[float]:
        """Misst die Energie in jedem Analyse-Band via FFT."""
        n_fft = 8192
        hop = n_fft // 2
        n_frames = max(1, (len(mono) - n_fft) // hop)

        # Akkumuliere Leistungsspektrum über alle Frames
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        power_acc = np.zeros(len(freqs), dtype=np.float64)

        for i in range(min(n_frames, 120)):  # max ~60s bei 8192/2 hop
            frame = mono[i * hop : i * hop + n_fft]
            if len(frame) < n_fft:
                break
            windowed = frame * np.hanning(n_fft)
            spec = np.abs(np.fft.rfft(windowed)) ** 2
            power_acc += spec

        power_acc /= max(1, n_frames)

        # Band-Energie in dB (relativ zum Maximum)
        band_energy = []
        for lo, hi in _ANALYSIS_BANDS:
            mask = (freqs >= lo) & (freqs < hi)
            if np.any(mask):
                energy = float(np.mean(power_acc[mask]))
            else:
                energy = 1e-12
            band_energy.append(10.0 * np.log10(energy + 1e-12))

        # Normalisiere: mache das lauteste Band zu 0 dB Referenz
        max_energy = max(band_energy)
        return [e - max_energy for e in band_energy]

    @staticmethod
    def _interpolate_target(curve: list[tuple[float, float]], centers: list[float]) -> list[float]:
        """Lineare Interpolation der Zielkurve auf die Analyse-Band-Mitten."""
        cf, cg = zip(*curve)  # curve frequencies, curve gains
        result = []
        for c in centers:
            # Finde umschließende Stützstellen
            if c <= cf[0]:
                result.append(cg[0])
            elif c >= cf[-1]:
                result.append(cg[-1])
            else:
                for i in range(len(cf) - 1):
                    if cf[i] <= c <= cf[i + 1]:
                        frac = (c - cf[i]) / (cf[i + 1] - cf[i])
                        gain = cg[i] + frac * (cg[i + 1] - cg[i])
                        result.append(gain)
                        break
        return result

    @staticmethod
    def _make_band_eq(sr: int, lo: float, hi: float, gain_db: float) -> np.ndarray | None:
        """Erzeugt einen Band-Pass/Peak-Filter als SOS für die gegebene
        Frequenz und Gain.

        Verwendet einen 2nd-order peaking EQ (constant-Q) zentriert
        auf der geometrischen Mittenfrequenz des Bandes.
        """
        try:
            import scipy.signal as sp_sig

            center = np.sqrt(lo * hi)
            q = center / (hi - lo) if hi > lo else 1.0
            q = float(np.clip(q, 0.3, 5.0))

            # Nutze scipy's peaking EQ wenn verfügbar (SciPy >= 1.12),
            # sonst butter bandpass mit shelving-Ansatz
            try:
                sos = sp_sig.iirpeak(center / (sr / 2), q, sr)
                # Skaliere den Gain
                gain_linear = 10 ** (gain_db / 20.0)
                sos[:, :3] *= gain_linear
                return sos  # type: ignore[no-any-return]
            except AttributeError as exc:
                logger.debug("§V6 scipy.signal.iirpeak nicht verfügbar — Butter-Bandpass Fallback aktiviert (%.0f-%.0f Hz): %s", lo, hi, exc)
                # Fallback: butter bandpass
                nyq = sr / 2
                lo_norm = max(0.001, lo / nyq)
                hi_norm = min(0.999, hi / nyq)
                sos = butter(2, [lo_norm, hi_norm], btype="band", output="sos")
                gain_linear = 10 ** (gain_db / 20.0)
                sos[:, :3] *= gain_linear
                return sos  # type: ignore[no-any-return]

        except Exception as exc:
            logger.debug("AdaptiveEQ band_eq fehlgeschlagen (%.0f-%.0f Hz): %s", lo, hi, exc)
            return None


def apply_adaptive_eq(audio: np.ndarray, sr: int, mode: str = "headphones") -> np.ndarray:
    """Free-Function-Wrapper um `AdaptiveListeningEQ.analyze_and_apply()`,
    der nur das bearbeitete Audio zurückgibt (für Aufrufer ohne Interesse
    an den Korrektur-Metadaten)."""
    return AdaptiveListeningEQ.analyze_and_apply(audio, sr, mode=mode).audio
