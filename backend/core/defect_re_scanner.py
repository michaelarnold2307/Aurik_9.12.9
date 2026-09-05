"""
§v10.28 Bidirectional Re-Scanner — erkennt neu enthüllte Defekte nach subtractiven Phasen.

Wenn Phase 01 Klicks entfernt oder Phase 03 entrauscht, können darunter liegende,
leisere Defekte für das menschliche Ohr hörbar werden. Dieser Re-Scanner prüft
nach den drei großen subtractiven Phasen (01, 03, 07), ob solche Defekte jetzt
sichtbar sind und meldet deren geschätzte Severity.

Anders als der DefectScanner (62 Defekte, teuer) ist dieser Scan leichtgewichtig:
- FFT-basiert, O(N log N)
- Nur 3–4 Defekte pro Phase
- ~50ms Laufzeit bei 48kHz/10s Audio
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

# ── Phasen, die einen Re-Scan triggern ────────────────────────────────
_RESCAN_TRIGGER_PHASES: frozenset[str] = frozenset({"phase_01_click_removal", "phase_03_denoise", "phase_07_declipper"})

# ── Phase-spezifische Defekt-Checks ────────────────────────────────────
# Jede subtractive Phase enthüllt bestimmte Defekte:
#   Phase 01 (Click Removal):  darunter liegendes Knistern/Transienten
#   Phase 03 (Denoise):        subtiles Rauschen unterhalb des alten Noise-Floors
#   Phase 07 (Declipper):      harmonische Verzerrung, die vom Clipping maskiert war

_PHASE_DEFECT_CHECKS: dict[str, list[tuple[str, tuple[float, float]]]] = {
    "phase_01_click_removal": [
        ("CRACKLE", (2000, 16000)),
    ],
    "phase_03_denoise": [
        ("HIGH_FREQ_NOISE", (8000, 20000)),
        ("HISS", (6000, 20000)),
        ("MODULATION_NOISE", (2000, 8000)),
    ],
    "phase_07_declipper": [
        ("OVERLOAD_DISTORTION", (1000, 8000)),
        ("INTERMODULATION_DISTORTION", (500, 10000)),
    ],
}

# ── Severity-Multiplikatoren pro Defekttyp ─────────────────────────────
_SEVERITY_MULTIPLIER: dict[str, float] = {
    "CRACKLE": 3.0,
    "HIGH_FREQ_NOISE": 2.0,
    "HISS": 2.5,
    "MODULATION_NOISE": 1.5,
    "OVERLOAD_DISTORTION": 2.0,
    "INTERMODULATION_DISTORTION": 1.5,
}


class DefectReScanner:
    """Leichter FFT-basierter Re-Scanner für enthüllte Defekte."""

    def __init__(self, n_fft: int = 4096):
        self.n_fft = n_fft

    def scan(
        self,
        audio: np.ndarray,
        sample_rate: int,
        phase_id: str = "",
    ) -> dict[str, float]:
        """Analysiert Audio auf Defekte, die nach einer subtractiven Phase
        sichtbar geworden sein könnten.

        Args:
            audio:       Aktuelles Audio-Signal (NACH der Phase, mono oder stereo)
            sample_rate: Sample-Rate in Hz
            phase_id:    ID der gerade ausgeführten Phase

        Returns:
            {defect_type: severity} — nur Defekte mit severity > 0.05.
            Leeres Dict wenn phase_id kein Re-Scan-Trigger ist.
        """
        if phase_id and phase_id not in _RESCAN_TRIGGER_PHASES:
            return {}

        # Mono, float32
        a = np.asarray(audio, dtype=np.float32)
        if a.ndim > 1:
            a = np.mean(a, axis=0)

        # Leises/Silence-Signal → keine Defekte
        rms = float(np.sqrt(np.mean(a * a) + 1e-12))
        if rms < 0.001:
            return {}

        # FFT-Frequenz-Achse
        freqs = np.fft.rfftfreq(self.n_fft, 1.0 / max(sample_rate, 1))

        # Magnituden-Spektrum
        spec = self._compute_magnitude_spectrum(a, freqs)
        total_energy = float(np.sum(spec**2)) + 1e-12

        revealed: dict[str, float] = {}

        checks = _PHASE_DEFECT_CHECKS.get(phase_id, [])

        for defect_type, (low_hz, high_hz) in checks:
            band_energy = self._band_energy(spec, freqs, low_hz, high_hz)
            if band_energy <= 0:
                continue

            energy_ratio = band_energy / total_energy

            # Mindestschwelle: 0.05% der Gesamtenergie im Defektband
            if energy_ratio >= 0.0005:
                severity = self._estimate_severity(defect_type, energy_ratio)
                if severity > 0.05:
                    revealed[defect_type] = severity
                    logger.debug(
                        "§v10.28 Re-Scan: %s enthüllt nach %s — Band [%d–%d Hz] Verhaeltnis=%.4f → severity=%.3f",
                        defect_type,
                        phase_id,
                        int(low_hz),
                        int(high_hz),
                        energy_ratio,
                        severity,
                    )

        return revealed

    # ── Private helpers ────────────────────────────────────────────────

    def _compute_magnitude_spectrum(self, audio: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Gemitteltes Magnituden-Spektrum über Frames."""
        n_samples = len(audio)
        hop = self.n_fft // 2
        n_frames = max(1, (n_samples - self.n_fft) // hop + 1)

        if n_frames <= 0:
            return cast(np.ndarray, (np.zeros(self.n_fft // 2 + 1, dtype=np.float64)))

        window = np.hanning(self.n_fft)
        spec_sum = np.zeros(self.n_fft // 2 + 1, dtype=np.float64)

        max_frames = min(n_frames, 500)
        for i in range(max_frames):
            start = i * hop
            if start + self.n_fft > n_samples:
                break
            frame = audio[start : start + self.n_fft] * window
            spec = np.abs(np.fft.rfft(frame))
            spec_sum += spec

        return cast(np.ndarray, (spec_sum / max(max_frames, 1)))

    @staticmethod
    def _band_energy(
        spectrum: np.ndarray,
        freqs: np.ndarray,
        low_hz: float,
        high_hz: float,
    ) -> float:
        """Spektrale Energie im Frequenzband [low_hz, high_hz]."""
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        if not np.any(mask):
            return 0.0
        return float(np.sum(spectrum[mask] ** 2))

    @staticmethod
    def _estimate_severity(defect_type: str, energy_ratio: float) -> float:
        """Severity aus Energie-Anteil schätzen."""
        multiplier = _SEVERITY_MULTIPLIER.get(defect_type, 2.0)
        return float(np.clip(energy_ratio * multiplier, 0.0, 1.0))

    @staticmethod
    def _get_checks_for_phase(phase_id: str) -> list[tuple[str, tuple[float, float]]]:
        """Phase-spezifische Defekt-Checks (für Tests)."""
        return _PHASE_DEFECT_CHECKS.get(phase_id, [])


# ── Singleton-Funktion (für Dead-Import-Reparatur) ───────────────────

def get_defect_re_scanner() -> DefectReScanner:
    """Gibt eine DefectReScanner-Instanz zurück."""
    return DefectReScanner()
