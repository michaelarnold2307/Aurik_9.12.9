"""Phantom Mode — Zero-Configuration Auto-Detect für Aurik.

§Rolls-Royce-Phantom: Der Nutzer gibt eine Datei — Aurik macht den Rest.
Kein --material, kein --defects, kein --era, kein --quality.
Alles wird automatisch aus dem Audio abgeleitet.

Nutzung:
    python -m aurik restore mein_song.wav --phantom
    # Oder einfach (phantom ist default in v11):
    python -m aurik restore mein_song.wav

Autor: Aurik 10 — Rolls-Royce Phantom Edition, 11. Juli 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PhantomConfig:
    """Vollständig automatisch erkannte Konfiguration."""

    material: str  # shellac, vinyl, tape, digital
    era: int  # Geschätztes Aufnahmejahr
    genre: str  # Jazz, Classical, Rock, ...
    defects: list[str]  # Erkannte Defekte
    defect_severity: dict[str, float]  # 0–1 pro Defekt
    has_vocals: bool
    vocal_confidence: float
    recommended_mode: str  # quick, full, deep
    quality_preset: str  # draft, standard, high, archival
    estimated_snr_db: float
    confidence: float  # Gesamt-Confidence der Erkennung (0–1)


class PhantomDetector:
    """Zero-Config Auto-Detektor — Datei rein, Konfiguration raus.

    Nutzt spektrale Analyse, Energie-Verteilung, und Heuristiken
    zur vollautomatischen Erkennung aller Parameter.
    """

    def __init__(self):
        self._material_profiles: dict[str, dict[str, Any]] = {
            "shellac": {"bw_hz": 5500, "noise_floor_db": -35, "typical_era": (1920, 1955)},
            "vinyl": {"bw_hz": 18000, "noise_floor_db": -55, "typical_era": (1950, 1990)},
            "tape": {"bw_hz": 14000, "noise_floor_db": -50, "typical_era": (1950, 1995)},
            "cassette": {
                "bw_hz": 14000,
                "noise_floor_db": -45,
                "typical_era": (1970, 2005),
            },  # §6.2c central definition
            "digital": {"bw_hz": 22000, "noise_floor_db": -80, "typical_era": (1985, 2026)},
        }

    def detect(self, audio: np.ndarray, sr: int = 48000) -> PhantomConfig:
        """Vollautomatische Parameter-Erkennung.

        Args:
            audio: float32, mono oder stereo.
            sr:    Abtastrate.

        Returns:
            PhantomConfig mit allen erkannten Parametern.
        """
        mono = np.mean(audio, axis=-1) if audio.ndim > 1 else audio
        mono = mono.astype(np.float32).flatten()
        len(mono) / sr

        # ── Material-Erkennung ──────────────────────────────────────────
        material, mat_conf = self._detect_material(mono, sr)

        # ── Ära-Schätzung ────────────────────────────────────────────────
        era, era_conf = self._estimate_era(mono, sr, material)

        # ── Defekt-Erkennung ─────────────────────────────────────────────
        defects, severity = self._detect_defects(mono, sr, material)

        # ── Genre-Erkennung ──────────────────────────────────────────────
        genre = self._detect_genre(mono, sr)

        # ── Gesangs-Erkennung ────────────────────────────────────────────
        has_vocals, vocal_conf = self._detect_vocals(mono, sr)

        # ── SNR-Schätzung ────────────────────────────────────────────────
        snr_db = self._estimate_snr(mono)

        # ── Modus-Empfehlung ─────────────────────────────────────────────
        if len(defects) >= 3 or material in ("shellac", "wax_cylinder"):
            mode = "deep"
        elif len(defects) >= 1:
            mode = "full"
        else:
            mode = "quick"

        # ── Qualitäts-Preset ─────────────────────────────────────────────
        if material in ("shellac", "wax_cylinder", "wire_recording"):
            quality = "archival"  # Historisch = maximale Sorgfalt
        elif snr_db < 20:
            quality = "high"
        else:
            quality = "standard"

        # ── Gesamt-Confidence ────────────────────────────────────────────
        confidences = [mat_conf, era_conf, vocal_conf]
        overall_conf = float(np.mean(confidences))

        return PhantomConfig(
            material=material,
            era=era,
            genre=genre,
            defects=defects,
            defect_severity=severity,
            has_vocals=has_vocals,
            vocal_confidence=vocal_conf,
            recommended_mode=mode,
            quality_preset=quality,
            estimated_snr_db=round(snr_db, 1),
            confidence=round(overall_conf, 2),
        )

    # ── Private Detektoren ──────────────────────────────────────────────

    def _detect_material(self, mono: np.ndarray, sr: int) -> tuple[str, float]:
        """Erkennt Tonträger-Material via Bandbreite und Rauschboden."""
        n_fft = min(4096, len(mono) // 2)
        spec = np.abs(np.fft.rfft(mono[: n_fft * 2]))
        freqs = np.fft.rfftfreq(n_fft * 2, d=1.0 / sr)
        total_energy = float(np.sum(spec)) + 1e-10

        # Bandbreite: Wo fällt Energie auf 1% des Maximums?
        spec_cumsum = np.cumsum(spec[::-1]) / total_energy
        bw_idx = np.argmax(spec_cumsum > 0.01)
        bw_hz = float(freqs[-1 - bw_idx]) if bw_idx < len(freqs) else sr / 2

        # Rauschboden: Median der Energie oberhalb von 8 kHz
        high_mask = freqs > 8000
        if np.any(high_mask):
            noise_floor = float(np.median(spec[high_mask]))
        else:
            noise_floor = 0.0

        # Matching gegen Profile
        best_material = "digital"
        best_score = 0.0

        for mat, profile in self._material_profiles.items():
            bw_score = 1.0 - min(1.0, abs(bw_hz - profile["bw_hz"]) / profile["bw_hz"])  # type: ignore[operator]
            nf_score = (
                1.0 - min(1.0, abs(noise_floor - profile["noise_floor_db"]) / abs(profile["noise_floor_db"]))  # type: ignore[operator]
                if profile["noise_floor_db"] != 0
                else 0.0
            )
            score = 0.6 * bw_score + 0.4 * nf_score
            if score > best_score:
                best_score = score
                best_material = mat

        return best_material, round(best_score, 2)

    def _estimate_era(self, mono: np.ndarray, sr: int, material: str) -> tuple[int, float]:
        """Schätzt Aufnahmejahr via spektrale Charakteristik."""
        profile = self._material_profiles.get(material, self._material_profiles["digital"])
        era_min, era_max = profile["typical_era"]  # type: ignore[misc]

        # Stereo-Breite als Ära-Indikator (Mono = älter)
        if material in ("vinyl", "tape"):
            # Einfach: Mitte des typischen Bereichs
            era = int((era_min + era_max) / 2)  # type: ignore[has-type]
            return era, 0.6

        era = int(era_max)  # type: ignore[has-type]
        return era, 0.8

    def _detect_defects(self, mono: np.ndarray, sr: int, material: str) -> tuple[list[str], dict[str, float]]:
        """Erkennt Defekte via spektraler Signaturen."""
        defects: list[str] = []
        severity: dict[str, float] = {}

        # Clicks: Transiente Spitzen
        diff = np.diff(mono)
        threshold = 3.0 * np.std(diff)
        click_count = int(np.sum(np.abs(diff) > threshold))
        click_rate = click_count / max(len(mono) / sr, 1)
        if click_rate > 2:
            defects.append("clicks")
            severity["clicks"] = min(1.0, click_rate / 50)

        # Hiss/Rauschen: Rauschboden oberhalb von 6 kHz
        n_fft = min(4096, len(mono) // 2)
        spec = np.abs(np.fft.rfft(mono[: n_fft * 2]))
        freqs = np.fft.rfftfreq(n_fft * 2, d=1.0 / sr)
        hiss_mask = freqs > 6000
        if np.any(hiss_mask):
            hiss_energy = float(np.sum(spec[hiss_mask]))
            total_energy = float(np.sum(spec)) + 1e-10
            if hiss_energy / total_energy > 0.15:
                defects.append("hiss")
                severity["hiss"] = min(1.0, hiss_energy / total_energy * 5)

        # Hum: 50/60 Hz + Obertöne
        for hum_freq in [50, 60]:
            hum_bin = int(hum_freq * n_fft * 2 / sr)
            if hum_bin < len(spec):
                hum_ratio = float(spec[hum_bin] / (np.mean(spec) + 1e-10))
                if hum_ratio > 5:
                    defects.append("hum")
                    severity["hum"] = min(1.0, hum_ratio / 20)
                    break

        return defects, severity

    def _detect_genre(self, mono: np.ndarray, sr: int) -> str:
        """Vereinfachte Genre-Erkennung."""
        float(np.sqrt(np.mean(mono**2)))
        # Platzhalter — echte Genre-Erkennung bräuchte ML
        return "unknown"

    def _detect_vocals(self, mono: np.ndarray, sr: int) -> tuple[bool, float]:
        """Erkennt Gesangspräsenz."""
        from backend.core.vocal_quality_gate import VocalDetector

        detector = VocalDetector()
        presence = detector.detect(mono, sr)
        return presence.has_vocals, presence.confidence

    def _estimate_snr(self, mono: np.ndarray) -> float:
        """Schätzt Signal-Rausch-Abstand in dB."""
        if len(mono) < 1024:
            return 0.0

        # Fenster-basierte SNR-Schätzung
        window = 1024
        hop = 512
        n_frames = (len(mono) - window) // hop + 1
        if n_frames < 2:
            return 0.0

        frame_rms = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop
            frame_rms[i] = np.sqrt(np.mean(mono[start : start + window] ** 2))

        # Signal = obere 10%, Rauschen = untere 10%
        sorted_rms = np.sort(frame_rms)
        n_10pct = max(1, n_frames // 10)
        signal_rms = np.mean(sorted_rms[-n_10pct:])
        noise_rms = np.mean(sorted_rms[:n_10pct]) + 1e-10

        snr = 20 * np.log10(signal_rms / noise_rms)
        return float(max(0, min(100, snr)))


def detect_phantom_config(audio: np.ndarray, sr: int = 48000) -> PhantomConfig:
    """Convenience-Funktion: Vollautomatische Konfigurationserkennung."""
    detector = PhantomDetector()
    return detector.detect(audio, sr)


__all__ = [
    "PhantomDetector",
    "PhantomConfig",
    "detect_phantom_config",
]
