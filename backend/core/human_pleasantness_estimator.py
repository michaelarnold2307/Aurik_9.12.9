"""
§v10 Human Pleasantness Estimator (HPE) — „Klingt das angenehm für menschliche Ohren?"

Der HPE ist die fehlende Brücke zwischen Auriks technischen Metriken und
dem, worauf es wirklich ankommt: menschlichem Wohlklang.

Während PMGG auf „Ähnlichkeit zum Original" optimiert (preservation),
optimiert der HPE auf PSYCHOAKUSTISCHE ANGENEHMHEIT — unabhängig davon,
ob das Ergebnis vom Original abweicht. Eine sauberere, klarere Aufnahme
DARF anders klingen als das verrauschte Original.

Wissenschaftliche Basis (ISO 532, Zwicker/Fastl, ANSI S3.4):
- Sharpness (Zwicker): acum/Zwicker — zu scharf = unangenehm
- Roughness (Fastl): asper/Zwicker — Modulation 15-300Hz = rau
- Tonalness: Verhältnis tonale/rauschartige Komponenten
- Loudness (Moore/Glasberg): sone/ISO 532 — zu laut = unangenehm
- Fluctuation Strength: vacil/Zwicker — Modulation <20Hz = schwankend

Composite Score P ∈ [0.0, 1.0] wo 1.0 = maximal angenehm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PleasantnessResult:
    """Ergebnis der psychoakustischen Angenehmheits-Prüfung."""

    score: float  # 0.0 = sehr unangenehm, 1.0 = sehr angenehm
    sharpness_zwicker: float  # acum (≤ 2 = angenehm, > 3 = scharf)
    roughness_asper: float  # asper (≤ 1 = glatt, > 2 = rau)
    loudness_sone: float  # sone (10-25 = angenehm für Musik)
    tonalness: float  # 0-1 (höher = tonaler = angenehmer)
    fluctuation_vacil: float  # vacil (≤ 1 = stabil, > 2 = schwankend)

    # Interpretation
    label: str = ""  # "Sehr angenehm", "Angenehm", "Neutral", "Anstrengend"
    issues: list[str] = field(default_factory=list)
    recommendation: str = ""


def compute_pleasantness(
    audio: np.ndarray,
    sr: int,
    *,
    original_audio: np.ndarray | None = None,
) -> PleasantnessResult:
    """Berechnet den psychoakustischen Angenehmheits-Score.

    Args:
        audio:           Mono oder Stereo Audio
        sr:              Sample-Rate (muss ≥ 44100 sein)
        original_audio:  Optionales Original zum Vergleich

    Returns:
        PleasantnessResult mit Score und Detail-Werten
    """
    arr = np.asarray(audio, dtype=np.float64)
    # Mono für Analyse
    if arr.ndim == 2:
        mono = arr.mean(axis=1) if arr.shape[1] <= 2 else arr.mean(axis=0)
    else:
        mono = arr
    mono = np.atleast_1d(mono).ravel()

    # ── 1. Sharpness (Zwicker, acum) ──
    sharpness = _compute_zwicker_sharpness(mono, sr)

    # ── 2. Roughness (Fastl, asper) ──
    roughness = _compute_roughness(mono, sr)

    # ── 3. Loudness (vereinfachtes Moore/Glasberg via RMS + Gewichtung) ──
    loudness_sone = _estimate_loudness_sone(mono, sr)

    # ── 4. Tonalness ──
    tonalness = _compute_tonalness(mono, sr)

    # ── 5. Fluctuation Strength ──
    fluctuation = _compute_fluctuation_strength(mono, sr)

    # ── Composite Score ──
    # Gewichte: Sharpness und Roughness sind die dominanten Faktoren
    # für „angenehm/unangenehm" (Zwicker & Fastl 2007, Kap. 11)
    s_sharp = max(0.0, 1.0 - (sharpness - 1.5) / 2.5)  # sharpness < 1.5 → score=1, > 4 → score=0
    s_rough = max(0.0, 1.0 - roughness / 2.0)  # roughness < 0.5 → score=1, > 2 → score=0
    s_loud = max(0.0, 1.0 - abs(loudness_sone - 18.0) / 15.0)  # optimal bei ~18 sone
    s_tonal = tonalness  # tonalness direkt als Score
    s_fluct = max(0.0, 1.0 - fluctuation / 2.0)

    score = float(np.clip(s_sharp * 0.30 + s_rough * 0.25 + s_loud * 0.15 + s_tonal * 0.20 + s_fluct * 0.10, 0.0, 1.0))

    # ── Label & Issues ──
    issues = []
    if sharpness > 3.5:
        issues.append(f"Zu scharf ({sharpness:.1f} acum) — Höhen abdämpfen")
    elif sharpness < 0.8:
        issues.append(f"Zu dumpf ({sharpness:.1f} acum) — Höhen anheben")
    if roughness > 2.0:
        issues.append(f"Zu rau ({roughness:.1f} asper) — Modulation glätten")
    if loudness_sone > 30:
        issues.append(f"Zu laut ({loudness_sone:.0f} sone) — Pegel senken")
    elif loudness_sone < 5:
        issues.append(f"Zu leise ({loudness_sone:.0f} sone) — Pegel anheben")
    if tonalness < 0.25:
        issues.append(f"Zu rauschhaft ({tonalness:.2f}) — mehr tonale Anteile erwünscht")

    if score >= 0.70:
        label = "Sehr angenehm"
        rec = "Keine Änderungen nötig — für menschliche Ohren optimiert."
    elif score >= 0.50:
        label = "Angenehm"
        rec = "Leichte Optimierungen möglich — " + (issues[0] if issues else "insgesamt gut.")
    elif score >= 0.35:
        label = "Neutral"
        rec = "Spürbare Verbesserung möglich — " + (issues[0] if issues else "Details prüfen.")
    else:
        label = "Anstrengend"
        rec = "Deutliche Überarbeitung nötig — " + (issues[0] if issues else "alle Dimensionen prüfen.")

    return PleasantnessResult(
        score=score,
        sharpness_zwicker=sharpness,
        roughness_asper=roughness,
        loudness_sone=loudness_sone,
        tonalness=tonalness,
        fluctuation_vacil=fluctuation,
        label=label,
        issues=issues,
        recommendation=rec,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Psychoakustische Messfunktionen
# ═══════════════════════════════════════════════════════════════════════════


def _compute_zwicker_sharpness(mono: np.ndarray, sr: int) -> float:
    """Zwicker Sharpness (acum) nach ISO 532-1 / DIN 45692.

    Korrekte Implementierung mit:
    - Bark-Skala via z(f) = 13*arctan(0.00076*f) + 3.5*arctan((f/7500)^2)
    - 24 aequidistante Bark-Baender (0-24 Bark, je 1 Bark breit)
    - g(z) = 1 fuer z <= 15.8, g(z) = 0.066*exp(0.171*z) fuer z > 15.8
    - Sharpness S = 0.11 * sum N'(z)*g(z)*z / sum N'(z)

    Typische Werte: dumpf=0.8-1.2, normal=1.5-2.0, hell=2.5-3.5, schrill=>4.0
    """
    n_fft = 4096
    if len(mono) < n_fft:
        return 1.5

    spec = np.abs(np.fft.rfft(mono[:n_fft] * np.hanning(n_fft)))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # Bark-Rate: z(f) = 13*arctan(0.076f) + 3.5*arctan(f^2/56.25e6)
    z_f = 13.0 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500.0) ** 2)

    # 24 Bark-Baender, je 1 Bark breit
    n_bands = 24
    band_energy = np.zeros(n_bands)
    for i in range(n_bands):
        z_low = i
        z_high = i + 1
        mask = (z_f >= z_low) & (z_f < z_high)
        if mask.sum() > 0:
            band_energy[i] = np.mean(spec[mask])

    # Spezifische Loudness N' proportional zu E^0.23
    eps = 1e-12
    specific_loudness = (band_energy + eps) ** 0.23
    specific_loudness /= np.sum(specific_loudness) + eps

    # Bark-Mitten: 0.5, 1.5, ..., 23.5
    z_centers = np.arange(0.5, 24.0, 1.0)

    # g(z) nach ISO 532-1
    g_z = np.ones(n_bands)
    high = z_centers > 15.8
    g_z[high] = 0.066 * np.exp(0.171 * z_centers[high])

    numerator = float(np.sum(specific_loudness * g_z * z_centers))
    denominator = np.sum(specific_loudness) + eps

    sharpness = 0.11 * numerator / denominator
    return float(np.clip(sharpness, 0.5, 5.0))


def _compute_roughness(mono: np.ndarray, sr: int) -> float:
    """Roughness (asper): RMS-Varianz in 50ms-Fenstern.

    Wahrgenommene Rauigkeit korreliert mit der Variation des Signalpegels
    im Bereich 15-300 Hz (Zwicker & Fastl 2007). Einfacher, robuster Proxy.
    """
    win = int(0.05 * sr)  # 50ms
    if len(mono) < 4 * win:
        # §v10.93: < 200ms — RMS-basierter Modulation-Proxy
        _r = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + 1e-12))
        return float(np.clip(_r * 5.0, 0.30, 0.70))

    rms_vals = []
    for i in range(0, len(mono) - win, win // 2):
        chunk = mono[i : i + win]
        rms_vals.append(float(np.sqrt(np.mean(chunk**2))))

    rms_vals = np.array(rms_vals)  # type: ignore[assignment]
    rms_db = 20.0 * np.log10(rms_vals + 1e-12)  # type: ignore[operator]
    rms_range = float(np.max(rms_db) - np.min(rms_db))

    # Dynamik > 20dB = sehr lebendig, > 10dB = normal, < 3dB = flach
    # Flache Signale mit schneller Variation = rau
    # Berechne Mikro-Variation: RMS der Differenzen
    diffs = np.abs(np.diff(rms_db))
    micro_var = float(np.mean(diffs))

    # Kombiniere: hohe Mikro-Variation + flaches Signal = rau
    #              niedrige Mikro-Variation + dynamisch = glatt
    roughness = float(np.clip(micro_var * (1.0 + max(0, 15 - rms_range) / 15.0) * 0.25, 0.1, 5.0))
    return roughness


def _estimate_loudness_sone(mono: np.ndarray, sr: int) -> float:
    """Vereinfachte Loudness in Sone (ISO 532 via RMS + Frequenzgewichtung).

    Genauere Implementierung in dsp/psychoacoustics.py (Zwicker/ISO 532-1).
    """
    rms = float(np.sqrt(np.mean(mono**2)) + 1e-12)
    rms_db = 20.0 * np.log10(rms)

    # Pegel → Sone (vereinfacht: 1 sone = 40 phon bei 1 kHz)
    # 0 dBFS ≈ 100 dB SPL → pegeldB = dBFS + 100
    spl_estimate = rms_db + 100.0

    # ISO 226 Näherung: phon → sone
    if spl_estimate < 40:
        sone = 0.0
    else:
        sone = 2.0 ** ((spl_estimate - 40.0) / 10.0)

    return float(np.clip(sone, 0.0, 100.0))


def _compute_tonalness(mono: np.ndarray, sr: int) -> float:
    """Tonalness: Verhaeltnis tonaler zu rauschartigen Komponenten.

    Kalibriert mit echter Musik: Popsong mit Gesang ~0.4-0.6,
    Sprache ~0.3-0.5, reines Rauschen ~0.0-0.2, Sinus ~0.9-1.0.
    """
    n_fft = 4096
    if len(mono) < n_fft:
        # §v10.93: < 93ms — simplified flatness proxy from RMS
        _r = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + 1e-12))
        return float(np.clip(1.0 - _r * 3.0, 0.30, 0.70))

    spec = np.abs(np.fft.rfft(mono[:n_fft] * np.hanning(n_fft)))

    # Spektrale Flachheit (Spectral Flatness):
    # tonal = Energie in wenigen Bins konzentriert -> niedrige Flachheit
    # rauschen = Energie gleichmaessig verteilt -> hohe Flachheit
    # Formel: SF = exp(mean(log(spec))) / mean(spec)
    # tonalness = 1 - SF (normiert)
    spec_linear = spec[1:] + 1e-12  # ignoriere DC
    log_spec = np.log(spec_linear)
    spectral_flatness = float(np.exp(np.mean(log_spec)) / (np.mean(spec_linear) + 1e-12))

    # Kalibriert: Sinus SF~1e-5, Musik SF~0.02, Rauschen SF~0.3
    # tonalness = 1 - SF/0.5 (clamped)
    tonalness = float(np.clip(1.0 - spectral_flatness / 0.5, 0.0, 1.0))
    return tonalness


def _compute_fluctuation_strength(mono: np.ndarray, sr: int) -> float:
    """Fluctuation Strength (vacil): Modulation < 20 Hz.

    Hohe Fluktuation = unangenehmes „Wabern".
    """
    win = int(1.0 * sr)  # 1s Fenster
    if len(mono) < 2 * win:
        # §v10.93: < 2s — Peak/RMS-Crest als Fluktuations-Proxy
        peak = float(np.max(np.abs(mono)) + 1e-12)
        rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + 1e-12))
        crest = peak / rms if rms > 0 else 1.0
        return float(np.clip(crest / 15.0, 0.30, 0.70))

    # RMS-Verlauf in 1s-Schritten
    rms_timeline = []
    for i in range(0, len(mono) - win, win):
        chunk = mono[i : i + win]
        rms_timeline.append(float(np.sqrt(np.mean(chunk**2))))

    if len(rms_timeline) < 3:
        # §v10.93: < 3 RMS-Blöcke — Peak/RMS-Crest als Proxy
        peak = float(np.max(np.abs(mono)) + 1e-12)
        rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + 1e-12))
        crest = peak / rms if rms > 0 else 1.0
        return float(np.clip(crest / 15.0, 0.30, 0.70))

    rms_timeline = np.array(rms_timeline)  # type: ignore[assignment]
    rms_db = 20.0 * np.log10(rms_timeline + 1e-12)  # type: ignore[operator]

    # Fluktuation = Varianz des RMS über 1-Sekunden-Blöcke
    fluctuation = float(np.clip(np.std(rms_db) / 3.0, 0.1, 5.0))
    return fluctuation


# ═══════════════════════════════════════════════════════════════════════════
# Vergleich: Original vs. Restauriert
# ═══════════════════════════════════════════════════════════════════════════


def compare_pleasantness(
    original: np.ndarray,
    restored: np.ndarray,
    sr: int,
) -> dict:
    """Vergleicht die Angenehmheit von Original und restauriertem Audio.

    Gibt zurück, OB und WARUM das restaurierte Audio angenehmer (oder weniger
    angenehm) als das Original ist.

    Returns:
        dict mit 'improved', 'delta_score', 'original', 'restored', 'verdict'
    """
    orig = compute_pleasantness(original, sr)
    rest = compute_pleasantness(restored, sr)

    delta = rest.score - orig.score
    improved = delta > 0.03  # Mindestschwelle für „hörbare Verbesserung"

    if improved and rest.score >= 0.70:
        verdict = "Deutlich angenehmer — die Restaurierung verbessert den Höreindruck spürbar."
    elif improved:
        verdict = "Etwas angenehmer — leichte Verbesserung des Höreindrucks."
    elif delta > -0.03:
        verdict = "Kein signifikanter Unterschied in der Angenehmheit."
    else:
        verdict = f"Weniger angenehm — die Restaurierung könnte überbearbeitet sein. {rest.recommendation}"

    return {
        "improved": improved,
        "delta_score": float(delta),
        "original": {"score": orig.score, "label": orig.label, "issues": orig.issues},
        "restored": {"score": rest.score, "label": rest.label, "issues": rest.issues},
        "verdict": verdict,
    }
