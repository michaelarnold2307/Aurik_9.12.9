"""
Vocal Harmonic Decomposition — Lücke 2 (v10.0.0.x)
=================================================

Dekomponiert das Audio VOR ML-NR in Vokalharmonische und Nicht-Harmonische.
Erlaubt separat skalierte NR-Gain-Floors auf beiden Schichten:

    Harmonische Bins   → stark geschützter G_floor (0.25–0.45)
    Nicht-Harmonische  → Standard G_floor (0.10–0.15)

ALGORITHMUS:
    1. F0-Extraktion: FCPE-Plugin (Spec 04: primär) oder ZCPA-DSP-Fallback
       (CREPE ist als eigenständiger Produktions-Tier VERBOTEN — Spec 04, Z. 1129)
    2. Harmonische Bin-Maske im STFT: für jeden Frame alle Partials
       (F0, 2F0, 3F0 ... 16F0) mit Gaußscher Breite σ = ±50 Hz
    3. Maske → soft mask (Hann-gewichtet) zur sanften Trennung
    4. Harmonische-Maske und Nicht-Harmonische-Maske getrennt ausgeben

Verwendung in phase_03:
    from backend.core.dsp.vocal_harmonic_decomp import VocalHarmonicMask
    vmask = VocalHarmonicMask(audio, sr)
    harm_mask = vmask.harmonic_mask()    # shape: (n_freq, n_frames)
    phase_03 setzt G_floor in harm_mask-Bins auf harm_g_floor (höher)

Author: Aurik Development Team
Version: 1.0.0 (v10.0.0.x — Lücke 2)
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import correlate as _scipy_correlate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

_N_FFT = 2048
_HOP = 512
_MAX_PARTIALS = 16  # Harmonische 1–16 × F0
_PARTIAL_WIDTH_HZ = 50.0  # Gaußsche Halbwertsbreite ±50 Hz je Partial
_F0_MIN_HZ = 60.0  # Tiefste Vokalgrundfrequenz (Bass-Sänger)
_F0_MAX_HZ = 1100.0  # Höchste Vokalgrundfrequenz (Sopran-Oktave)

# G_floor-Empfehlungen für NR-Integration
HARMONIC_G_FLOOR_DEFAULT = 0.35  # Vokalharmonische stark schützen
NONHARMONIC_G_FLOOR_DEFAULT = 0.10  # Rest: Standard-Floor


# ---------------------------------------------------------------------------
# F0-Schätzung (ZCPA-DSP-Fallback, CREPE via optionalem Import)
# ---------------------------------------------------------------------------


def _estimate_f0_zcpa(mono: np.ndarray, sr: int, hop: int) -> np.ndarray:
    """
    Zero-Crossing-Peak-Amplitude F0-Schätzung (schnell, robust).
    Liefert F0 in Hz pro Frame (0.0 = unvoiced/Pause).
    """
    n_frames = max(1, (len(mono) - _N_FFT) // hop + 1)
    f0_frames = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        frame = mono[i * hop : i * hop + _N_FFT]
        if len(frame) < _N_FFT:
            break
        # Autokorrelation für F0-Schätzung
        corr = _scipy_correlate(frame, frame, mode="full", method="fft")
        corr = corr[len(corr) // 2 :]

        # Suche erstes lokales Maximum nach Minimumsuche (τ_min entspricht F0_max)
        tau_min = int(sr / _F0_MAX_HZ)
        tau_max = int(sr / _F0_MIN_HZ)
        tau_min = max(1, tau_min)
        tau_max = min(len(corr) - 1, tau_max)

        if tau_max <= tau_min:
            continue

        search = corr[tau_min:tau_max]
        if len(search) < 2:
            continue

        # Peak-Detektion
        peak_rel = int(np.argmax(search))
        tau = tau_min + peak_rel
        f0 = sr / tau if tau > 0 else 0.0

        # Voiced-Gate: Autokorrelation-Konfidenz
        conf = float(corr[tau] / (corr[0] + 1e-12))
        if conf < 0.25 or not (_F0_MIN_HZ <= f0 <= _F0_MAX_HZ):
            f0 = 0.0

        f0_frames[i] = float(f0)

    return f0_frames  # type: ignore[no-any-return]


def _estimate_f0_fcpe(mono: np.ndarray, sr: int) -> np.ndarray | None:
    """FCPE-F0-Schätzung via Plugin (§4.4 Primär; CREPE nur als FCPE-interner Delegate)."""
    try:
        from plugins.fcpe_plugin import get_fcpe_plugin  # pylint: disable=import-outside-toplevel

        result = get_fcpe_plugin().analyze(mono, sr)
        if result is None:
            return None

        f0_raw: np.ndarray = np.asarray(result.f0_hz, dtype=np.float32)
        confidence: np.ndarray = np.asarray(result.voiced_prob, dtype=np.float32)
        # Unvoiced-Gate: Konfidenz < 0.5 → 0 Hz
        f0_raw[confidence < 0.50] = 0.0
        return f0_raw

    except Exception as exc:
        logger.debug("VocalHarmonicMask: FCPE nicht verfuegbar, using ZCPA — %s", exc)
        return None


# ---------------------------------------------------------------------------
# Maske
# ---------------------------------------------------------------------------


class VocalHarmonicMask:
    """
    Berechnet die Harmonische-Maske eines Vokalsignals im STFT-Bereich.

    Singleton-freie Klasse — wird per Aufruf instanziiert, GC'et nach Benutzung.

    Beispiel::

        mask = VocalHarmonicMask(audio, sr)
        H = mask.harmonic_mask()   # shape (n_freq, n_frames), Werte [0, 1]
        # 1 = Harmonischer Bin, 0 = Rauschen/Nicht-Harmonisch
    """

    def __init__(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        n_fft: int = _N_FFT,
        hop: int = _HOP,
        max_partials: int = _MAX_PARTIALS,
        partial_width_hz: float = _PARTIAL_WIDTH_HZ,
        use_fcpe: bool = True,
    ) -> None:
        self._sr = sr
        self._n_fft = n_fft
        self._hop = hop
        self._max_partials = max_partials
        self._partial_width_hz = partial_width_hz

        # Mono
        if audio.ndim == 2:
            mono = np.nan_to_num(audio.mean(axis=0), nan=0.0).astype(np.float64)
        else:
            mono = np.nan_to_num(audio, nan=0.0).astype(np.float64)
        self._mono = mono

        # F0-Schätzung
        f0_raw: np.ndarray | None = None
        if use_fcpe:
            f0_raw = _estimate_f0_fcpe(mono, sr)
        self._f0: np.ndarray = f0_raw if f0_raw is not None else _estimate_f0_zcpa(mono, sr, hop)

        # Frequenzachse
        self._freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)  # (n_freq,)
        self._n_freq = len(self._freqs)

    # ------------------------------------------------------------------

    def harmonic_mask(
        self,
        *,
        soft: bool = True,
    ) -> np.ndarray:
        """
        Erstellt die Harmonische-Maske.

        Args:
            soft: Wenn True, Gaußsche Soft-Mask (glatter Übergang).
                  Wenn False, Hard-Mask (Rechteck ±partial_width_hz).

        Returns:
            np.ndarray, shape (n_freq, n_frames), Werte ∈ [0, 1].
            1 = Vokal-Harmonisch, 0 = Nicht-Harmonisch.
        """
        n_frames = len(self._f0)
        mask = np.zeros((self._n_freq, n_frames), dtype=np.float32)

        sigma_hz = self._partial_width_hz / 2.355  # Halbwertsbreite → Standardabweichung

        for t, f0 in enumerate(self._f0):
            if t >= n_frames:
                break
            if f0 < _F0_MIN_HZ:
                continue  # Unvoiced/Pause → keine Maske für diesen Frame

            for k in range(1, self._max_partials + 1):
                center = f0 * k
                if center > self._freqs[-1]:
                    break  # Oberhalb Nyquist

                if soft:
                    # Gaußsche Gewichtung um Partial-Mittelpunkt
                    gaussian = np.exp(-0.5 * ((self._freqs - center) / sigma_hz) ** 2)
                    mask[:, t] = np.minimum(1.0, mask[:, t] + gaussian.astype(np.float32))
                else:
                    # Hard-Mask: alle Bins ±partial_width_hz um Partial
                    hard = (np.abs(self._freqs - center) <= self._partial_width_hz).astype(np.float32)
                    mask[:, t] = np.minimum(1.0, mask[:, t] + hard)

        return mask  # type: ignore[no-any-return]

    def nonharmonic_mask(self, *, soft: bool = True) -> np.ndarray:
        """
        Inverse der Harmonischen-Maske: 1 = Nicht-Harmonisch.

        Returns:
            np.ndarray, shape (n_freq, n_frames), Werte ∈ [0, 1].
        """
        return 1.0 - self.harmonic_mask(soft=soft)  # type: ignore[no-any-return]

    def apply_g_floor_adjustment(
        self,
        g_floor_map: np.ndarray,
        *,
        harm_g_floor: float = HARMONIC_G_FLOOR_DEFAULT,
        nonharm_g_floor: float = NONHARMONIC_G_FLOOR_DEFAULT,
    ) -> np.ndarray:
        """
        Justiert eine NR-G_floor-Map (n_freq × n_frames) basierend auf der Harmonik-Maske.

        In Vokalharmonischen-Bins wird g_floor auf harm_g_floor angehoben
        (schützt Stimmtimbre), außerhalb bleibt nonharm_g_floor.

        Args:
            g_floor_map:    Eingangs-G_floor-Map (n_freq × n_frames), float
            harm_g_floor:   Ziel-G_floor für harmonische Bins
            nonharm_g_floor: Ziel-G_floor für nicht-harmonische Bins

        Returns:
            Adjustierte G_floor-Map, gleiche shape.
        """
        h_mask = self.harmonic_mask(soft=True)

        # Wenn keine externe g_floor_map übergeben: eigene Maske als Basis verwenden
        if g_floor_map is None:
            adjusted = h_mask * harm_g_floor + (1.0 - h_mask) * nonharm_g_floor
            return adjusted.astype(np.float32)

        # Maske auf gleiche shape bringen falls nötig
        if h_mask.shape != g_floor_map.shape:
            # Zeitachse angleichen per Wiederholung des letzten Frames
            n_frames_needed = g_floor_map.shape[1] if g_floor_map.ndim > 1 else 1
            if h_mask.shape[1] < n_frames_needed:
                pad = np.tile(h_mask[:, -1:], (1, n_frames_needed - h_mask.shape[1]))
                h_mask = np.concatenate([h_mask, pad], axis=1)
            h_mask = h_mask[:, :n_frames_needed]

        # Blende: harm_g_floor × H_maske + nonharm_g_floor × (1 − H_maske)
        adjusted = h_mask * harm_g_floor + (1.0 - h_mask) * nonharm_g_floor
        # Nie unter dem übergebenen g_floor_map (konservativ — nimm das Maximum)
        adjusted = np.maximum(adjusted, g_floor_map).astype(np.float32)
        return adjusted  # type: ignore[no-any-return]

    @property
    def voiced_fraction(self) -> float:
        """Anteil stimmhafter Frames (F0 > 0) im Signal."""
        return float(np.mean(self._f0 > _F0_MIN_HZ))

    @property
    def f0_contour(self) -> np.ndarray:
        """F0-Kontur in Hz pro Frame (0 = unvoiced)."""
        return self._f0.copy()


# ---------------------------------------------------------------------------
# Schnelle Top-Level-Funktion
# ---------------------------------------------------------------------------


def build_vocal_harmonic_mask(
    audio: np.ndarray,
    sr: int,
    *,
    use_fcpe: bool = True,
) -> VocalHarmonicMask | None:
    """
    Erstellt VocalHarmonicMask. Non-blocking: Exception → None.

    Verwendung in Phase-03::

        vmask = build_vocal_harmonic_mask(audio, sr)
        if vmask is not None and vmask.voiced_fraction > 0.15:
            g_floor_map = vmask.apply_g_floor_adjustment(g_floor_map)
    """
    try:
        return VocalHarmonicMask(audio, sr, use_fcpe=use_fcpe)
    except Exception as exc:
        logger.warning("§G23 build_vocal_harmonic_mask: DSP-Ersatzpfad — %s", exc, exc_info=True)
        return None
