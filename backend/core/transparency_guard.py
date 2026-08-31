"""
§v10 Transparency Guard — „Klingt das natürlich oder bearbeitet?"

Der ultimative Test für jede Restaurierung: Hört man, DASS etwas
verändert wurde? Wenn ja — ist die Restaurierung gescheitert, egal
wie gut die technischen Metriken sind.

Das menschliche Ohr erkennt Bearbeitung an vier Merkmalen:

1. SPECTRAL WATER („wässrig"):  Schmalband-Lücken im Spektrum
   durch zu aggressive Rauschunterdrückung. Klingt „metallisch".

2. TRANSIENT SMEAR („verschmiert"):  Attack-Phasen von Instrumenten
   werden durch zu lange Filter weichgezeichnet. Klingt „matt".

3. NOISE BREATHING („Atmen"):  Rauschboden moduliert im Takt der
   Musik. Das Ohr hört „da wurde ent rauscht".

4. PHASE WARP („Phantom-Kanal"):  Phasenverschiebungen erzeugen
   räumliche Artefakte. Klingt „hohl" oder „komisch breit".

Composite TS (Transparency Score) ∈ [0,1]:
- TS ≥ 0.85 = Transparent — niemand hört, dass bearbeitet wurde
- TS ≥ 0.60 = Akzeptabel — nur geschulte Ohren erkennen es
- TS < 0.40 = Hörbar bearbeitet — künstlich, wässrig, unnatürlich
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TransparencyResult:
    score: float
    spectral_water: float  # 0=stark wässrig, 1=keine Artefakte
    transient_smear: float  # 0=stark verschmiert, 1=kristallklar
    noise_breathing: float  # 0=starkes Atmen, 1=konstanter Rauschboden
    phase_warp: float  # 0=stark verzerrt, 1=natürliche Phase
    sterility: float = 1.0  # §v10: 0=steril/leblos, 1=natürlich lebendig

    label: str = ""
    recommendation: str = ""
    artifacts: list[str] = field(default_factory=list)


def check_transparency(audio: np.ndarray, sr: int, reference: np.ndarray | None = None) -> TransparencyResult:
    """Prüft ob die Bearbeitung hörbar ist.

    Args:
        audio:     Restauriertes Audio
        sr:        Sample-Rate
        reference: Optional: Original zum Vergleich

    Returns:
        TransparencyResult
    """
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 2:
        mono = arr.mean(axis=1) if arr.shape[1] <= 2 else arr.mean(axis=0)
    else:
        mono = arr
    mono = np.atleast_1d(mono).ravel()

    water = _check_spectral_water(mono, sr)
    smear = _check_transient_smear(mono, sr)
    breath = _check_noise_breathing(mono, sr)
    warp = _check_phase_warp(arr, sr) if arr.ndim == 2 else 1.0
    sterile = _check_sterility(mono, sr, reference)

    score = float(np.clip(water * 0.25 + smear * 0.20 + breath * 0.20 + warp * 0.15 + sterile * 0.20, 0.0, 1.0))

    artifacts = []
    if water < 0.5:
        artifacts.append("wässrig/metallisch")
    if smear < 0.5:
        artifacts.append("verschmiert/matt")
    if breath < 0.4:
        artifacts.append("atmender Rauschboden")
    if warp < 0.5:
        artifacts.append("Phasen-Artefakte")
    if sterile < 0.3:
        artifacts.append("steril/leblos")

    if score >= 0.85:
        label = "Transparent"
        rec = "Niemand hört, dass bearbeitet wurde — natürlich und unverfälscht."
    elif score >= 0.60:
        label = "Akzeptabel"
        rec = "Nur geschulte Ohren erkennen leichte Bearbeitungsspuren. " + (artifacts[0] if artifacts else "")
    elif score >= 0.40:
        label = "Hörbar"
        rec = "Bearbeitung ist erkennbar — " + (artifacts[0] if artifacts else "unnatürlicher Klang.")
    else:
        label = "Künstlich"
        rec = "Deutlich bearbeitet — " + ", ".join(artifacts[:3]) if artifacts else "wirkt unnatürlich."

    return TransparencyResult(score, water, smear, breath, warp, label, rec, artifacts)  # type: ignore[arg-type]


def _check_spectral_water(mono: np.ndarray, sr: int) -> float:
    """Erkennt wässrige/metallische Artefakte via Spektral-UNNATÜRLICHKEIT.

    Ein natürliches Spektrum hat sanfte Übergänge. Wässrige Artefakte
    entstehen durch spektrale Subtraktion und erzeugen scharfe Einbrüche
    oder isolierte Spitzen — das Spektrum wirkt „gekämmt".
    """
    n_fft = 2048
    if len(mono) < 2 * n_fft:
        return 0.7

    # Mehrere Frames analysieren
    hop = n_fft // 2
    frame_scores = []
    for i in range(0, len(mono) - n_fft, hop):
        frame = np.nan_to_num(mono[i : i + n_fft], nan=0.0, posinf=0.0, neginf=0.0) * np.hanning(n_fft)
        spec = np.abs(np.fft.rfft(frame))
        spec_db = 20.0 * np.log10(np.maximum(spec, 1e-12))

        # Glätte und berechne Residual-Varianz
        kernel = np.ones(15) / 15.0
        smooth = np.convolve(spec_db, kernel, mode="same")
        residual = spec_db - smooth
        mid = residual[len(kernel) // 2 : -len(kernel) // 2] if len(residual) > len(kernel) else residual
        if len(mid) > 10:
            frame_scores.append(float(np.var(mid)))

    if not frame_scores:
        return 0.7

    avg_var = float(np.mean(frame_scores))

    if avg_var < 15:
        return 0.90
    elif avg_var < 30:
        return 0.65
    elif avg_var < 50:
        return 0.40
    elif avg_var < 80:
        return 0.20
    return 0.10


def _check_transient_smear(mono: np.ndarray, sr: int) -> float:
    """Prüft ob Transienten (Attack-Phasen) verschmiert wurden."""
    win = int(0.005 * sr)  # 5ms
    if len(mono) < 10 * win:
        return 0.7

    # Finde Transienten via Energie-Sprung
    energy = []
    for i in range(0, len(mono) - win, win // 2):
        energy.append(float(np.sum(mono[i : i + win] ** 2)))
    energy = np.array(energy)  # type: ignore[assignment]
    energy_db = 10.0 * np.log10(energy + 1e-12)  # type: ignore[operator]
    jumps = np.diff(energy_db)

    # Zähle scharfe Transienten (>10dB Anstieg)
    transients = jumps > 10.0
    n_transients = int(np.sum(transients))

    if n_transients < 2:
        return 0.5  # Keine Transienten zum Beurteilen

    # Für jeden Transient: prüfe Anstiegszeit
    rise_times = []
    for idx in np.where(transients)[0]:
        if idx < 2:
            continue
        # Wie viele Samples bis zum Peak?
        rise = 0
        for k in range(idx - 2, min(idx + 3, len(energy_db))):
            if energy_db[k] - energy_db[idx - 1] > 3:
                rise += 1
        rise_times.append(rise)

    if not rise_times:
        return 0.5

    mean_rise = np.mean(rise_times)
    # Kurze Anstiegszeit (1-2 Frames = 2-5ms) = scharfe Transienten
    # Lange Anstiegszeit (>5 Frames = >12ms) = verschmierte Transienten
    if mean_rise < 3:
        return 0.90
    elif mean_rise < 6:
        return 0.60
    elif mean_rise < 10:
        return 0.35
    return 0.15


def _check_noise_breathing(mono: np.ndarray, sr: int) -> float:
    """Erkennt atmenden Rauschboden (Gate-Pumping)."""
    win = int(0.1 * sr)  # 100ms
    if len(mono) < 20 * win:
        return 0.7

    # RMS in 100ms-Blöcken
    rms_blocks = []
    for i in range(0, len(mono) - win, win):
        chunk = mono[i : i + win]
        rms_blocks.append(float(np.sqrt(np.mean(chunk**2))))

    rms_blocks = np.array(rms_blocks)  # type: ignore[assignment]
    rms_db = 20.0 * np.log10(rms_blocks + 1e-12)  # type: ignore[operator]

    # Finde leiseste Passagen (Noise Floor)
    noise_floor_percentile = 15
    noise_idx = rms_db <= np.percentile(rms_db, noise_floor_percentile)
    noise_levels = rms_db[noise_idx]

    if len(noise_levels) < 3:
        return 0.7

    # Variation des Noise Floors = Atmen
    noise_variation = float(np.std(noise_levels))

    if noise_variation < 3.0:
        return 0.95  # Kaum Variation
    elif noise_variation < 6.0:
        return 0.65
    elif noise_variation < 12.0:
        return 0.35
    return 0.15


def _check_phase_warp(arr: np.ndarray, sr: int) -> float:
    """Prüft Phasen-Kohärenz (Stereo)."""
    if arr.ndim != 2 or arr.shape[1] != 2:
        return 0.8

    L = arr[:, 0].astype(np.float64)
    R = arr[:, 1].astype(np.float64)

    n_fft = 2048
    if len(L) < n_fft:
        return 0.8

    # Phase-Differenz zwischen L und R über mehrere Frames
    hop = n_fft // 4
    phase_variances = []
    for i in range(0, len(L) - n_fft, hop):
        spec_L = np.fft.rfft(L[i : i + n_fft] * np.hanning(n_fft))
        spec_R = np.fft.rfft(R[i : i + n_fft] * np.hanning(n_fft))
        phase_diff = np.angle(spec_L / (spec_R + 1e-12))
        phase_variances.append(float(np.var(phase_diff)))

    if not phase_variances:
        return 0.8

    mean_phase_var = float(np.mean(phase_variances))
    # Niedrige Varianz = konsistente Phase = natürlich
    # Hohe Varianz = Phasen-Artefakte
    if mean_phase_var < 0.5:
        return 0.90
    elif mean_phase_var < 1.5:
        return 0.60
    elif mean_phase_var < 3.0:
        return 0.35
    return 0.15


def _check_sterility(mono: np.ndarray, sr: int, reference: np.ndarray | None = None) -> float:
    """§v10: Erkennt sterilen/leblosen Klang — wenn zu viel bereinigt wurde."""
    win = int(0.05 * sr)
    if len(mono) < 20 * win:
        return 0.7

    rms_vals = [float(np.sqrt(np.mean(mono[i : i + win] ** 2))) for i in range(0, len(mono) - win, win)]
    rms_db = 20.0 * np.log10(np.array(rms_vals) + 1e-12)
    noise_floor_db = float(np.percentile(rms_db, 15))
    peak_db = float(np.max(rms_db))
    dr = peak_db - noise_floor_db

    score_dr = 0.15 if dr < 10 else 0.40 if dr < 20 else 0.70 if dr < 35 else 0.90
    score_nf = (
        0.10 if noise_floor_db < -90 else 0.35 if noise_floor_db < -75 else 0.60 if noise_floor_db < -60 else 0.85
    )
    diffs = np.abs(np.diff(rms_db))
    md = float(np.median(diffs[diffs > 0.5])) if np.any(diffs > 0.5) else 0.0
    score_micro = 0.10 if md < 0.5 else 0.40 if md < 1.5 else 0.70 if md < 3.0 else 0.95
    return float(np.clip(score_dr * 0.35 + score_nf * 0.40 + score_micro * 0.25, 0.0, 1.0))
