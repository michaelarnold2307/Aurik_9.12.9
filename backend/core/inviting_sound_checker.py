"""
§v10 Einladender-Klang-Checker
===============================
„Legt sich das Ohr hinein oder wird es zurückgewiesen?"

HPE misst ANGENEHMHEIT, aber das ist nur die halbe Wahrheit.
Ein Klang kann „angenehm" sein (HPE=0.80) und trotzdem das Ohr
zurückweisen — weil er zu scharf anspringt, zu hart komprimiert ist,
oder den Hörer durch plötzliche Artefakte erschreckt.

Dieser Checker prüft spezifisch auf ZURÜCKWEISUNGS-Faktoren:

1. EAR-FATIGUE:     Energie >8kHz über 30s > Schwellwert → Ohr ermüdet
2. STARTLE-ATTACK:  Transienten-Anstieg >18dB/5ms → Ohr erschrickt
3. STEREO-STRESS:   Phasen-Differenz L/R >45° über 1kHz → räumliche Desorientierung
4. GATE-PUMPING:    Rauschboden-Modulation >6dB/100ms → „Atmen" des Rauschens
5. FREQUENCY-PEAKS: Schmalband-Resonanzen >12dB über Nachbar-Bark-Band → Frequenz-Stress

Composite IS (Inviting Score) ∈ [0,1]:
- IS ≥ 0.80 = Einladend — das Ohr legt sich gern hinein
- IS ≥ 0.50 = Neutral — bleibt, aber nicht begeistert
- IS < 0.30 = Zurückweisend — das Ohr will weg
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InvitingResult:
    score: float
    ear_fatigue: float
    startle_attack: float
    stereo_stress: float
    gate_pumping: float
    frequency_peaks: float
    label: str = ""
    recommendation: str = ""
    rejection_factors: list[str] = field(default_factory=list)


def check_inviting_sound(audio: np.ndarray, sr: int) -> InvitingResult:
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 2:
        mono = arr.mean(axis=1) if arr.shape[1] <= 2 else arr.mean(axis=0)
    else:
        mono = arr
    mono = np.atleast_1d(mono).ravel()

    fatigue = _check_ear_fatigue(mono, sr)
    startle = _check_startle_attack(mono, sr)
    stress = _check_stereo_stress(arr, sr) if arr.ndim == 2 else 1.0
    pumping = _check_gate_pumping(mono, sr)
    peaks = _check_frequency_peaks(mono, sr)

    score = float(np.clip(fatigue * 0.25 + startle * 0.20 + stress * 0.15 + pumping * 0.20 + peaks * 0.20, 0.0, 1.0))

    rejects = []
    if fatigue < 0.4:
        rejects.append("Hohe Frequenzen ermüden das Ohr")
    if startle < 0.4:
        rejects.append("Zu scharfe Transienten erschrecken")
    if stress < 0.4:
        rejects.append("Stereofeld erzeugt räumlichen Stress")
    if pumping < 0.3:
        rejects.append("Rauschboden-Atmung lenkt ab")
    if peaks < 0.3:
        rejects.append("Schmalband-Resonanzen stechen heraus")

    if score >= 0.80:
        label = "Einladend"
        rec = "Der Klang lädt das Ohr ein — nichts weist zurück."
    elif score >= 0.50:
        label = "Neutral"
        rec = "Bleibt angenehm. " + (rejects[0] if rejects else "")
    elif score >= 0.30:
        label = "Leicht zurückweisend"
        rec = rejects[0] if rejects else "Einige Faktoren stören."
    else:
        label = "Zurückweisend"
        rec = "Mehrere Faktoren weisen das Ohr zurück: " + ", ".join(rejects[:3]) if rejects else ""

    return InvitingResult(score, fatigue, startle, stress, pumping, peaks, label, rec, rejects)


def _check_ear_fatigue(mono: np.ndarray, sr: int) -> float:
    n_fft = 4096
    if len(mono) < n_fft:
        return 0.7
    spec = np.abs(np.fft.rfft(mono[:n_fft] * np.hanning(n_fft)))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    high = np.sum(spec[(freqs >= 8000) & (freqs <= 16000)] ** 2) if np.any(freqs >= 8000) else 0.0
    mid = np.sum(spec[(freqs >= 500) & (freqs <= 4000)] ** 2) if np.any(freqs >= 500) else 1.0
    ratio = high / (mid + 1e-12)

    if ratio < 0.005:
        return 0.9
    elif ratio < 0.02:
        return 0.7
    elif ratio < 0.05:
        return 0.5
    elif ratio < 0.15:
        return 0.3
    return 0.1


def _check_startle_attack(mono: np.ndarray, sr: int) -> float:
    env = np.abs(mono)
    win = int(0.005 * sr)
    if len(mono) < 2 * win:
        return 0.7

    peaks = []
    for i in range(win, len(mono) - win, win // 4):
        chunk = env[i : i + win]
        peaks.append(float(np.max(chunk)))
    peaks = np.array(peaks)  # type: ignore[assignment]
    peaks_db = 20.0 * np.log10(peaks + 1e-12)  # type: ignore[operator]

    jumps = np.diff(peaks_db)
    startling = int(np.sum(jumps > 18.0))
    ratio = startling / max(len(jumps), 1)

    if ratio < 0.001:
        return 0.95
    elif ratio < 0.005:
        return 0.7
    elif ratio < 0.02:
        return 0.4
    return 0.15


def _check_stereo_stress(arr: np.ndarray, sr: int) -> float:
    if arr.shape[1] != 2:
        return 0.8
    L = arr[:, 0].astype(np.float64)
    R = arr[:, 1].astype(np.float64)

    n_fft = 2048
    if len(L) < n_fft:
        return 0.8
    spec_L = np.fft.rfft(L[:n_fft] * np.hanning(n_fft))
    spec_R = np.fft.rfft(R[:n_fft] * np.hanning(n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    phase_diff = np.angle(spec_L / (spec_R + 1e-12))
    high_mask = freqs > 1000
    if np.any(high_mask):
        high_phase = np.abs(phase_diff[high_mask])
        mean_phase = float(np.mean(high_phase))
        if mean_phase > np.pi / 4:
            return 0.3
        elif mean_phase > np.pi / 8:
            return 0.6
        return 0.9
    return 0.8


def _check_gate_pumping(mono: np.ndarray, sr: int) -> float:
    win = int(0.1 * sr)
    if len(mono) < 10 * win:
        return 0.7

    rms_timeline = []
    for i in range(0, len(mono) - win, win // 4):
        chunk = mono[i : i + win]
        rms = float(np.sqrt(np.mean(chunk**2)))
        rms_timeline.append(rms)
    rms = np.array(rms_timeline)  # type: ignore[assignment]
    rms_db = 20.0 * np.log10(rms + 1e-12)

    diffs = np.abs(np.diff(rms_db))
    pumping_events = int(np.sum((diffs > 6.0) & (diffs < 20.0)))
    ratio = pumping_events / max(len(diffs), 1)

    if ratio < 0.005:
        return 0.95
    elif ratio < 0.02:
        return 0.65
    elif ratio < 0.08:
        return 0.35
    return 0.15


def _check_frequency_peaks(mono: np.ndarray, sr: int) -> float:
    n_fft = 4096
    if len(mono) < n_fft:
        return 0.7

    spec = np.abs(np.fft.rfft(mono[:n_fft] * np.hanning(n_fft)))
    spec_db = 20.0 * np.log10(spec + 1e-12)

    bark_edges = [
        0,
        100,
        200,
        300,
        400,
        510,
        630,
        770,
        920,
        1080,
        1270,
        1480,
        1720,
        2000,
        2320,
        2700,
        3150,
        3700,
        4400,
        5300,
        6400,
        7700,
        9500,
        12000,
        15500,
    ]
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    peak_bands = 0
    for i in range(len(bark_edges) - 1):
        mask = (freqs >= bark_edges[i]) & (freqs < bark_edges[i + 1])
        if mask.sum() > 0:
            band_max = float(np.max(spec_db[mask]))
            band_mean = float(np.mean(spec_db[mask]) + 1e-12)
            if band_max - band_mean > 12.0:
                peak_bands += 1

    if peak_bands == 0:
        return 0.95
    elif peak_bands <= 2:
        return 0.7
    elif peak_bands <= 5:
        return 0.4
    return 0.15


def check_inviting_sound_per_band(audio: np.ndarray, sr: int) -> dict:
    """§v10 Frequenzspezifischer Inviting-Check pro Bark-Band.

    Zeigt WELCHE Frequenzbereiche das Ohr zurückweisen.
    Returns dict mit Band-Name -> (Score, Problem).
    """
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 2:
        mono = arr.mean(axis=1) if arr.shape[1] <= 2 else arr.mean(axis=0)
    else:
        mono = arr
    mono = np.atleast_1d(mono).ravel()
    n_fft = 4096
    if len(mono) < n_fft:
        return {"fullband": (0.5, "zu kurz")}
    spec = np.abs(np.fft.rfft(mono[:n_fft] * np.hanning(n_fft)))
    spec_db = 20.0 * np.log10(spec + 1e-12)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    bands = {
        "sub_bass": (20, 60),
        "bass": (60, 250),
        "low_mid": (250, 500),
        "mid": (500, 2000),
        "high_mid": (2000, 4000),
        "presence": (4000, 8000),
        "brilliance": (8000, 16000),
    }
    results = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        if mask.sum() == 0:
            results[name] = (0.7, "keine Energie")
            continue
        band_energy = float(np.sum(spec[mask] ** 2))
        total_energy = float(np.sum(spec**2)) + 1e-12
        share = band_energy / total_energy
        band_max = float(np.max(spec_db[mask]))
        band_mean = float(np.mean(spec_db[mask]))
        score = 1.0
        issue = ""
        if name == "sub_bass" and share > 0.25:
            issue = "dröhnend"
            score = 0.3
        elif name == "bass":
            if share < 0.08:
                issue = "dünn"
                score = 0.3
            elif share > 0.40:
                issue = "mulmig"
                score = 0.4
        elif name == "low_mid" and share > 0.35:
            issue = "kastig"
            score = 0.4
        elif name == "presence" and share > 0.30:
            issue = "bissig"
            score = 0.35
        elif name == "brilliance":
            if share < 0.005:
                issue = "dumpf"
                score = 0.25
            elif share > 0.20:
                issue = "scharf"
                score = 0.35
        if band_max - band_mean > 15.0 and score > 0.5:
            issue = "Resonanz"
            score = max(0.3, score - 0.3)
        if not issue:
            issue = ""
        results[name] = (score, issue)
    return results
