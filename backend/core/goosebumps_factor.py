"""
§v10 Gänsehaut-Faktor (Emotional Pleasantness) — „Bewegt es?"

Während der HPE die psychoakustische ANGENEHMHEIT misst, misst der
Gänsehaut-Faktor die EMOTIONALE WIRKUNG — das, was den Unterschied
zwischen „klingt sauber" und „Gänsehaut" ausmacht.

Wissenschaftliche Basis:
- Sloboda (1991): „Music Structure and Emotional Response" — Gänsehaut
  korreliert mit unerwarteten harmonischen Wendungen, dynamischen
  Kontrasten und Frequenz-Modulationen
- Panksepp (1995): „The Emotional Sources of Chills" — Trennung von
  „thrill" (Aufregung) und „chill" (Trauer/Ergriffenheit)
- Blood & Zatorre (2001): fMRI-Studien zeigen, dass musikalische
  Gänsehaut das Belohnungszentrum (Nucleus Accumbens) aktiviert
- Grewe et al. (2007): Physiologische Korrelate: Herzfrequenz-Änderung,
  Hautleitfähigkeit, Atemmuster — alle durch akustische Merkmale
  vorhersagbar

Die fünf Dimensionen des Gänsehaut-Faktors:

1. DYNAMIC CONTRAST (0-1): Plötzliche Lautstärke-Wechsel (>6 dB/500ms)
   zu laut = Schock, zu leise = unsichtbar. Optimal: moderate Kontraste.

2. HARMONIC SURPRISE (0-1): Unerwartete Akkordwechsel (via Chroma-Differenz)
   Zu vorhersagbar = langweilig. Zu chaotisch = verwirrend. Optimal: Goldlöckchen.

3. SPECTRAL SHIMMER (0-1): Hochfrequente Mikro-Variation (8-14 kHz)
   Fehlt komplett = steril. Zu viel = schrill. Optimal: subtiler Glanz.

4. TEMPORAL BREATH (0-1): Natürliche Mikro-Timing-Variationen (Rubato)
   Zu perfekt = maschinell. Zu unregelmäßig = stolpernd. Optimal: menschlich.

5. FREQUENCY WARMTH (0-1): Sub-Bass bis Low-Mid-Präsenz (30-300 Hz)
   Fehlt = kalt/dünn. Zu viel = mulmig. Optimal: warm, nicht dröhnend.

Composite Goosebumps Score G ∈ [0,1]:
- G ≥ 0.70 = Hohe Gänsehaut-Wahrscheinlichkeit
- G ≥ 0.45 = Moderate emotionale Wirkung
- G < 0.25 = Emotional flach
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GoosebumpsResult:
    """Gänsehaut-Faktor Ergebnis."""

    score: float = 0.0
    dynamic_contrast: float = 0.0
    harmonic_surprise: float = 0.0
    spectral_shimmer: float = 0.0
    temporal_breath: float = 0.0
    frequency_warmth: float = 0.0
    label: str = "neutral"
    recommendation: str = ""
    issues: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


# §v10.0.5 Genre-adaptive Goosebumps-Gewichtung
# Jedes Genre hat unterschiedliche emotionale Charakteristiken:
# Ambient = natürlicherweise wenig Dynamik → niedrigere dynamic-Erwartung.
# Klassik/Oper = hohe harmonische Komplexität → höheres harmonic-Gewicht.
# Metal/Hip-Hop = basslastig → höheres warmth-Gewicht.
_GENRE_WEIGHTS: dict[str, dict[str, float]] = {
    "ambient": {"dynamic": 0.10, "harmonic": 0.20, "shimmer": 0.30, "breath": 0.20, "warmth": 0.20},
    "classical": {"dynamic": 0.25, "harmonic": 0.30, "shimmer": 0.15, "breath": 0.15, "warmth": 0.15},
    "klassik": {"dynamic": 0.25, "harmonic": 0.30, "shimmer": 0.15, "breath": 0.15, "warmth": 0.15},
    "oper": {"dynamic": 0.25, "harmonic": 0.30, "shimmer": 0.15, "breath": 0.15, "warmth": 0.15},
    "jazz": {"dynamic": 0.20, "harmonic": 0.25, "shimmer": 0.20, "breath": 0.20, "warmth": 0.15},
    "metal": {"dynamic": 0.20, "harmonic": 0.15, "shimmer": 0.25, "breath": 0.10, "warmth": 0.30},
    "hiphop": {"dynamic": 0.15, "harmonic": 0.10, "shimmer": 0.25, "breath": 0.20, "warmth": 0.30},
    "hip-hop": {"dynamic": 0.15, "harmonic": 0.10, "shimmer": 0.25, "breath": 0.20, "warmth": 0.30},
    "electronic": {"dynamic": 0.20, "harmonic": 0.15, "shimmer": 0.30, "breath": 0.15, "warmth": 0.20},
    "reggae": {"dynamic": 0.15, "harmonic": 0.15, "shimmer": 0.20, "breath": 0.25, "warmth": 0.25},
    "gospel": {"dynamic": 0.25, "harmonic": 0.20, "shimmer": 0.20, "breath": 0.15, "warmth": 0.20},
}

_GENRE_THRESHOLDS: dict[str, dict[str, float]] = {
    "ambient": {
        "dynamic_low": 0.10,
        "dynamic_high": 0.70,
        "shimmer_low": 0.15,
        "shimmer_high": 0.90,
        "breath_low": 0.10,
        "warmth_low": 0.20,
    },
    "classical": {
        "dynamic_low": 0.25,
        "dynamic_high": 0.90,
        "shimmer_low": 0.20,
        "shimmer_high": 0.85,
        "breath_low": 0.25,
        "warmth_low": 0.25,
    },
    "klassik": {
        "dynamic_low": 0.25,
        "dynamic_high": 0.90,
        "shimmer_low": 0.20,
        "shimmer_high": 0.85,
        "breath_low": 0.25,
        "warmth_low": 0.25,
    },
    "oper": {
        "dynamic_low": 0.25,
        "dynamic_high": 0.90,
        "shimmer_low": 0.20,
        "shimmer_high": 0.85,
        "breath_low": 0.25,
        "warmth_low": 0.25,
    },
    "jazz": {
        "dynamic_low": 0.20,
        "dynamic_high": 0.85,
        "shimmer_low": 0.20,
        "shimmer_high": 0.85,
        "breath_low": 0.20,
        "warmth_low": 0.25,
    },
    "metal": {
        "dynamic_low": 0.20,
        "dynamic_high": 0.95,
        "shimmer_low": 0.15,
        "shimmer_high": 0.90,
        "breath_low": 0.10,
        "warmth_low": 0.20,
    },
    "hiphop": {
        "dynamic_low": 0.15,
        "dynamic_high": 0.90,
        "shimmer_low": 0.15,
        "shimmer_high": 0.90,
        "breath_low": 0.15,
        "warmth_low": 0.20,
    },
    "hip-hop": {
        "dynamic_low": 0.15,
        "dynamic_high": 0.90,
        "shimmer_low": 0.15,
        "shimmer_high": 0.90,
        "breath_low": 0.15,
        "warmth_low": 0.20,
    },
    "electronic": {
        "dynamic_low": 0.20,
        "dynamic_high": 0.90,
        "shimmer_low": 0.20,
        "shimmer_high": 0.95,
        "breath_low": 0.10,
        "warmth_low": 0.25,
    },
    "reggae": {
        "dynamic_low": 0.15,
        "dynamic_high": 0.80,
        "shimmer_low": 0.20,
        "shimmer_high": 0.85,
        "breath_low": 0.20,
        "warmth_low": 0.20,
    },
    "gospel": {
        "dynamic_low": 0.25,
        "dynamic_high": 0.90,
        "shimmer_low": 0.20,
        "shimmer_high": 0.85,
        "breath_low": 0.15,
        "warmth_low": 0.25,
    },
}


def _get_genre_weights(genre: str) -> dict[str, float]:
    return _GENRE_WEIGHTS.get(
        genre.lower(), {"dynamic": 0.25, "harmonic": 0.20, "shimmer": 0.20, "breath": 0.15, "warmth": 0.20}
    )


def _get_genre_thresholds(genre: str) -> dict[str, float]:
    return _GENRE_THRESHOLDS.get(
        genre.lower(),
        {
            "dynamic_low": 0.30,
            "dynamic_high": 0.85,
            "shimmer_low": 0.25,
            "shimmer_high": 0.85,
            "breath_low": 0.20,
            "warmth_low": 0.30,
        },
    )


def compute_goosebumps(
    audio: np.ndarray,
    sr: int,
    *,
    genre: str = "unknown",
) -> GoosebumpsResult:
    """Berechnet den Gänsehaut-Faktor.

    Args:
        audio:  Mono oder Stereo Audio
        sr:     Sample-Rate
        genre:  Optionales Genre für kontextsensitive Bewertung

    Returns:
        GoosebumpsResult mit Score und Detail-Werten
    """
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 2:
        mono = arr.mean(axis=1) if arr.shape[1] <= 2 else arr.mean(axis=0)
    else:
        mono = arr
    mono = np.atleast_1d(mono).ravel()

    # ── 1. Dynamic Contrast ──
    dynamic = _measure_dynamic_contrast(mono, sr)

    # ── 2. Harmonic Surprise ──
    harmonic = _measure_harmonic_surprise(mono, sr)

    # ── 3. Spectral Shimmer ──
    shimmer = _measure_spectral_shimmer(mono, sr)

    # ── 4. Temporal Breath ──
    breath = _measure_temporal_breath(mono, sr)

    # ── 5. Frequency Warmth ──
    warmth = _measure_frequency_warmth(mono, sr)

    # ── Composite Score ──
    # §v10.0.5 Genre-adaptive Gewichtung: jede Musikrichtung hat
    # unterschiedliche emotionale Charakteristiken.
    _genre_weights = _get_genre_weights(genre)
    score = float(
        np.clip(
            dynamic * _genre_weights.get("dynamic", 0.25)
            + harmonic * _genre_weights.get("harmonic", 0.20)
            + shimmer * _genre_weights.get("shimmer", 0.20)
            + breath * _genre_weights.get("breath", 0.15)
            + warmth * _genre_weights.get("warmth", 0.20),
            0.0,
            1.0,
        )
    )

    # ── Genre-adaptive Label & Recommendation ──
    _genre_thresholds = _get_genre_thresholds(genre)
    issues = []
    if dynamic < _genre_thresholds.get("dynamic_low", 0.3):
        issues.append("Zu wenig dynamische Kontraste — wirkt flach")
    elif dynamic > _genre_thresholds.get("dynamic_high", 0.85):
        issues.append("Zu extreme Lautstärke-Wechsel — wirkt unruhig")
    if shimmer < _genre_thresholds.get("shimmer_low", 0.25):
        issues.append("Fehlender Hochfrequenz-Glanz — wirkt dumpf")
    elif shimmer > _genre_thresholds.get("shimmer_high", 0.85):
        issues.append("Übermäßige Höhen — wirkt schrill")
    if breath < _genre_thresholds.get("breath_low", 0.2):
        issues.append("Zu perfektes Timing — wirkt maschinell")
    if warmth < _genre_thresholds.get("warmth_low", 0.3):
        issues.append("Fehlende Bass-Wärme — wirkt kalt/dünn")

    if score >= 0.70:
        label = "Gänsehaut"
        rec = "Emotional sehr wirkungsvoll — für ungestörten Musikgenuss optimiert."
    elif score >= 0.45:
        label = "Bewegend"
        rec = "Gute emotionale Wirkung. " + (issues[0] if issues else "Keine Optimierung nötig.")
    elif score >= 0.25:
        label = "Neutral"
        rec = "Emotionale Wirkung könnte stärker sein. " + (issues[0] if issues else "")
    else:
        label = "Flach"
        rec = "Emotional wenig wirkungsvoll — " + (issues[0] if issues else "alle Dimensionen prüfen.")

    return GoosebumpsResult(
        score=score,
        dynamic_contrast=dynamic,
        harmonic_surprise=harmonic,
        spectral_shimmer=shimmer,
        temporal_breath=breath,
        frequency_warmth=warmth,
        label=label,
        recommendation=rec,
        issues=issues,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Messfunktionen
# ═══════════════════════════════════════════════════════════════════════════


def _measure_dynamic_contrast(mono: np.ndarray, sr: int) -> float:
    """Dynamic Contrast: Plötzliche Lautstärke-Wechsel (6-15 dB/500ms).

    Zu wenig = langweilig. Zu viel = anstrengend. Sweet spot: 0.4-0.7.
    """
    win = int(0.5 * sr)  # 500ms
    if len(mono) < 2 * win:
        return 0.5

    rms_timeline = []
    for i in range(0, len(mono) - win, win // 2):
        chunk = mono[i : i + win]
        rms_timeline.append(float(np.sqrt(np.mean(chunk**2))))

    if len(rms_timeline) < 3:
        return 0.5

    rms_db = 20.0 * np.log10(np.array(rms_timeline) + 1e-12)
    diffs = np.abs(np.diff(rms_db))

    # Zähle „bedeutsame" Kontraste (6-15 dB)
    meaningful = int(np.sum((diffs >= 6.0) & (diffs <= 15.0)))
    extreme = int(np.sum(diffs > 15.0))

    contrast_ratio = meaningful / max(len(diffs), 1)
    extreme_penalty = extreme / max(len(diffs), 1) * 0.5

    return float(np.clip(contrast_ratio * 3.0 - extreme_penalty, 0.0, 1.0))


def _measure_harmonic_surprise(mono: np.ndarray, sr: int) -> float:
    """Harmonic Surprise: Via Chroma-Vektor-Differenz über Zeit.

    Misst die Rate harmonischer Wechsel. Zu langsam = vorhersagbar.
    Zu schnell = chaotisch. Sweet spot: moderate Wechselrate.
    """
    n_fft = 4096
    hop = n_fft // 4
    if len(mono) < 2 * n_fft:
        return 0.5

    # Vereinfachte Chroma-Extraktion: 12 Halbtöne, gemittelt über Oktaven
    chroma_sequence = []
    for i in range(0, len(mono) - n_fft, hop):
        frame = mono[i : i + n_fft] * np.hanning(n_fft)
        spec = np.abs(np.fft.rfft(frame))
        chroma = np.zeros(12)
        for c in range(12):
            # Summiere alle Frequenzen, die auf dieses Chroma fallen
            for k in range(c, len(spec), 12):
                if k < len(spec):
                    chroma[c] += spec[k]
        chroma /= np.sum(chroma) + 1e-12
        chroma_sequence.append(chroma)

    if len(chroma_sequence) < 3:
        return 0.5

    chroma_seq = np.array(chroma_sequence)
    # Differenz zwischen aufeinanderfolgenden Chroma-Vektoren
    diffs = np.array([np.sum(np.abs(chroma_seq[j + 1] - chroma_seq[j])) for j in range(len(chroma_seq) - 1)])

    mean_change = float(np.mean(diffs))
    variance = float(np.var(diffs))

    # Sweet spot: moderate mean change (0.3-0.7) mit etwas Varianz
    mean_score = 1.0 - abs(mean_change - 0.5) * 2.0
    var_score = min(variance * 5.0, 1.0)

    return float(np.clip(mean_score * 0.6 + var_score * 0.4, 0.0, 1.0))


def _measure_spectral_shimmer(mono: np.ndarray, sr: int) -> float:
    """Spectral Shimmer: Hochfrequente Mikro-Variation (8-14 kHz).

    Das ist der „Glanz", der Sterilität verhindert und Leben einhaucht.
    Zu wenig = dumpf/steril. Zu viel = schrill.
    """
    n_fft = 2048
    if len(mono) < n_fft:
        return 0.5

    spec = np.abs(np.fft.rfft(mono[:n_fft] * np.hanning(n_fft)))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # Energie in 8-14 kHz
    high_mask = (freqs >= 8000) & (freqs <= 14000)
    mid_mask = (freqs >= 500) & (freqs <= 4000)

    high_energy = np.sum(spec[high_mask] ** 2) if np.any(high_mask) else 0.0
    mid_energy = np.sum(spec[mid_mask] ** 2) if np.any(mid_mask) else 1.0

    shimmer_ratio = high_energy / (mid_energy + 1e-12)

    # Sweet spot: 0.01-0.05 (sehr subtiler Hochfrequenz-Anteil)
    if shimmer_ratio < 0.003:
        return 0.1  # Praktisch kein Glanz → steril
    elif shimmer_ratio < 0.01:
        return 0.3  # Sehr wenig Glanz
    elif shimmer_ratio < 0.05:
        return float(0.5 + (shimmer_ratio - 0.01) / 0.04 * 0.4)  # Sweet spot
    elif shimmer_ratio < 0.10:
        return float(0.9 - (shimmer_ratio - 0.05) / 0.05 * 0.4)  # Etwas zu viel
    else:
        return 0.3  # Zu viel → schrill


def _measure_temporal_breath(mono: np.ndarray, sr: int) -> float:
    """Temporal Breath: Natürliche Mikro-Timing-Variationen.

    Misst die Abweichung von perfekt regelmäßigem Timing.
    Perfekte Regelmäßigkeit = maschinell (Score niedrig).
    Moderate Unregelmäßigkeit = menschlich (Score hoch).
    """
    # Onset-Detektion via Energie-Anstieg
    win = int(0.010 * sr)  # 10ms
    if len(mono) < 20 * win:
        return 0.5

    energy = []
    for i in range(0, len(mono) - win, win // 2):
        chunk = mono[i : i + win]
        energy.append(float(np.sum(chunk**2)))

    energy = np.array(energy)  # type: ignore[assignment]
    energy_db = 10.0 * np.log10(energy + 1e-12)  # type: ignore[operator]

    # Onsets = positive Energie-Sprünge > 3 dB
    onsets = []
    for i in range(1, len(energy_db)):
        if energy_db[i] - energy_db[i - 1] > 3.0:
            onsets.append(i)

    if len(onsets) < 4:
        return 0.5

    # Inter-Onset-Intervalle
    iois = np.diff(onsets).astype(float)

    # Coefficient of Variation (CV) der IOIs
    cv = float(np.std(iois) / (np.mean(iois) + 1e-12))

    # Sweet spot: CV ≈ 0.05-0.15 (natürlich, nicht maschinell)
    # CV < 0.02 → zu perfekt (maschinell)
    # CV > 0.30 → zu chaotisch
    if cv < 0.02:
        return 0.15  # Maschinell
    elif cv < 0.05:
        return float(0.15 + (cv - 0.02) / 0.03 * 0.45)  # Übergang
    elif cv < 0.15:
        return float(0.6 + (cv - 0.05) / 0.10 * 0.35)  # Sweet spot
    elif cv < 0.30:
        return float(0.95 - (cv - 0.15) / 0.15 * 0.45)  # Zunehmend chaotisch
    else:
        return 0.3  # Zu chaotisch


def _measure_frequency_warmth(mono: np.ndarray, sr: int) -> float:
    """Frequency Warmth: Sub-Bass bis Low-Mid-Präsenz (30-300 Hz).

    Fehlt = kalt/dünn. Zu viel = mulmig. Sweet spot: präsent aber nicht dominant.
    """
    n_fft = 4096
    if len(mono) < n_fft:
        return 0.5

    spec = np.abs(np.fft.rfft(mono[:n_fft] * np.hanning(n_fft)))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    bass_mask = (freqs >= 30) & (freqs <= 300)
    full_mask = (freqs >= 30) & (freqs <= 8000)

    bass_energy = np.sum(spec[bass_mask] ** 2) if np.any(bass_mask) else 0.0
    full_energy = np.sum(spec[full_mask] ** 2) if np.any(full_mask) else 1.0

    warmth_ratio = bass_energy / (full_energy + 1e-12)

    # Sweet spot: 15-35% der Energie im Bassbereich
    if warmth_ratio < 0.08:
        return 0.15  # Praktisch kein Bass → kalt/dünn
    elif warmth_ratio < 0.15:
        return float(0.15 + (warmth_ratio - 0.08) / 0.07 * 0.45)
    elif warmth_ratio < 0.35:
        return float(0.6 + (warmth_ratio - 0.15) / 0.20 * 0.35)  # Sweet spot
    elif warmth_ratio < 0.50:
        return float(0.95 - (warmth_ratio - 0.35) / 0.15 * 0.45)
    else:
        return 0.2  # Zu viel Bass → mulmig


# ═══════════════════════════════════════════════════════════════════════════
# Integration: Gesamt-Emotional-Score (HPE + Gänsehaut)
# ═══════════════════════════════════════════════════════════════════════════


def compute_emotional_impact(
    audio: np.ndarray,
    sr: int,
    *,
    genre: str = "unknown",
) -> dict:
    """Kombiniert HPE und Gänsehaut-Faktor zu einem Gesamt-Emotional-Score.

    E ∈ [0, 1] — der ultimative Qualitäts-Indikator für Aurik:
    - E ≥ 0.75 = Weltklasse (Gänsehaut + angenehm)
    - E ≥ 0.55 = Sehr gut
    - E < 0.30 = Überarbeitung nötig
    """
    from backend.core.human_pleasantness_estimator import compute_pleasantness

    hpe = compute_pleasantness(audio, sr)
    gb = compute_goosebumps(audio, sr, genre=genre)

    # Kombination: 60% Angenehmheit + 40% emotionale Wirkung
    emotional_score = float(np.clip(hpe.score * 0.60 + gb.score * 0.40, 0.0, 1.0))

    if emotional_score >= 0.75:
        combined_label = "Weltklasse"
    elif emotional_score >= 0.55:
        combined_label = "Sehr gut"
    elif emotional_score >= 0.35:
        combined_label = "Gut"
    else:
        combined_label = "Optimierungsbedarf"

    return {
        "emotional_score": emotional_score,
        "label": combined_label,
        "pleasantness": hpe.score,
        "pleasantness_label": hpe.label,
        "goosebumps": gb.score,
        "goosebumps_label": gb.label,
        "recommendation": (
            f"{hpe.recommendation} {gb.recommendation}"
            if hpe.score < 0.7 or gb.score < 0.45
            else "Weltklasse-Ergebnis — bereit für ungestörten Musikgenuss mit Gänsehaut."
        ),
    }
