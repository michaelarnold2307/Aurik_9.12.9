"""
§v10 Sweet Spot Optimizer — Aurik findet autonom den optimalen Klang.

Das Ohr soll sich in den Klang „hineinlegen" können — mühelos, ohne
durch Störfaktoren zurückgewiesen zu werden. Dafür muss Aurik bei
JEDEM Song eigenständig den Sweetpoint finden — ohne menschliches Tuning.

Architektur:
  1. SweetSpot definiert 20-dimensionale „Grünzone"
  2. Optimizer tastet Parameterraum via Boundary-Optimizer ab
  3. Jeder Testpunkt wird auf ALLEN 20 Metriken bewertet
  4. Sweet Spot = der Punkt, wo die meisten Metriken gleichzeitig grün sind
  5. „Genug" = wenn weitere Optimierung keine Metrik mehr verbessert

Grünzone-Schwellwerte (empirisch kalibriert für „Hineinlegen"):
  HPE ≥ 0.65 | Inviting ≥ 0.70 | Transparency ≥ 0.65 | Goosebumps ≥ 0.40

Die 3 Lücken-Schließer:
  COMB_FILTER: Erkennt periodische Notch-Muster via Autokorrelation des Spektrums
  MUSICAL_COMPRESSION: Unterscheidet „gute" von „böser" Kompression via Krestfaktor
  MASKING: Erkennt Frequenz-Überdeckung via Bark-Band-Energie-Überlappung
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SweetSpotResult:
    """Das Ergebnis der Sweet-Spot-Optimierung."""

    score: float  # 0-1, wie nah am Sweet Spot
    all_green: bool  # True = ALLE Metriken in Grünzone

    hpe_score: float = 0.5
    inviting_score: float = 0.5
    transparency_score: float = 0.5
    goosebumps_score: float = 0.5

    comb_filter: float = 1.0  # 0=starker Kammfilter, 1=kein Kammfilter
    musical_compression: float = 1.0  # 0=bösartig komprimiert, 1=musikalisch
    masking_health: float = 1.0  # 0=starke Maskierung, 1=klare Trennung
    spectral_color: float = 1.0  # 0=Klangfarbe zerstoert, 1=perfekt erhalten
    microdynamics: float = 1.0  # 0=Dynamik platt, 1=natuerlich

    green_count: int = 0
    total_metrics: int = 20
    label: str = ""
    recommendation: str = ""
    warnings: list[str] = field(default_factory=list)


# Grünzone: Schwellwerte für „das Ohr legt sich hinein"
GREEN_ZONE = {
    "hpe": 0.55,
    "inviting": 0.65,
    "transparency": 0.60,
    "goosebumps": 0.20,
    "comb_filter": 0.80,
    "musical_compression": 0.60,
    "masking_health": 0.70,
    "spectral_color": 0.85,
    "microdynamics": 0.80,
}
GREEN_ZONE_RESTORATION = {
    "hpe": 0.55,
    "inviting": 0.70,
    "transparency": 0.70,
    "goosebumps": 0.15,
    "comb_filter": 0.85,
    "musical_compression": 0.55,
    "masking_health": 0.75,
    "spectral_color": 0.90,
    "microdynamics": 0.85,
}
GREEN_ZONE_STUDIO2026 = {
    "hpe": 0.65,
    "inviting": 0.60,
    "transparency": 0.50,
    "goosebumps": 0.30,
    "comb_filter": 0.75,
    "musical_compression": 0.40,
    "masking_health": 0.60,
    "spectral_color": 0.75,
    "microdynamics": 0.65,
}


def get_mode_green_zone(mode=""):
    m = str(mode or "restoration").lower()
    return dict(GREEN_ZONE_STUDIO2026) if "studio" in m or "2026" in m else dict(GREEN_ZONE_RESTORATION)


def find_sweet_spot(
    audio: np.ndarray,
    sr: int,
    *,
    genre: str = "unknown",
    reference: np.ndarray | None = None,
) -> SweetSpotResult:
    """Findet den Sweet Spot — alle Metriken gleichzeitig bewerten.

    Args:
        audio:  Mono oder Stereo Audio
        sr:     Sample-Rate
        genre:  Optionales Genre
        reference: Optionales Original-Signal (Pre-Restaurierung). Ist es
            gegeben, werden die drei vererbbaren Metriken (Kammfilter,
            Kompression, Maskierung) BASELINE-RELATIV bewertet: geerbte
            Original-Artefakte (Vinyl-Rotation, mp3_low-Quelle) werden der
            Pipeline nicht als Verschlechterung angelastet.

    Returns:
        SweetSpotResult mit Gesamtbewertung und allen Detail-Metriken
    """
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 2:
        mono = arr.mean(axis=1) if arr.shape[1] <= 2 else arr.mean(axis=0)
    else:
        mono = arr
    mono = np.atleast_1d(mono).ravel()

    # ── 4 Haupt-Metriken (existierende Module) ──
    hpe = _get_hpe(mono, sr)
    inviting = _get_inviting(mono, sr)
    transparency = _get_transparency(mono, sr, arr)
    goosebumps = _get_goosebumps(mono, sr)
    spectral_color = _check_spectral_color(mono, sr)
    microdynamics = _check_microdynamics(mono, sr)

    # ── 3 Lücken-Schließer ──
    comb_filter = _check_comb_filter(mono, sr)
    musical_comp = _check_musical_compression(mono, sr)
    masking = _check_frequency_masking(mono, sr)

    # §Spec 24 (Root-Fix 2026-08-16): Baseline-relative Bewertung der drei
    # vererbbaren Metriken. Vinyl-Rotation/mp3_low-Quellen HABEN inhärente
    # Kammfilter-/Kompressions-/Maskierungs-Artefakte — ohne Baseline bestraft
    # der Score die Pipeline für geerbtes Material und der SweetSpot-Loop
    # dreht leer (Befund: comb=0.10, 3 Iterationen ohne Fortschritt).
    _inherited: dict[str, bool] = {}
    if reference is not None:
        _ref_arr = np.asarray(reference, dtype=np.float64)
        if _ref_arr.ndim == 2:
            _ref_mono = _ref_arr.mean(axis=1) if _ref_arr.shape[1] <= 2 else _ref_arr.mean(axis=0)
        else:
            _ref_mono = _ref_arr
        _ref_mono = np.atleast_1d(_ref_mono).ravel()
        if _ref_mono.size >= 4096:
            _ref_comb = _check_comb_filter(_ref_mono, sr)
            _ref_comp = _check_musical_compression(_ref_mono, sr)
            _ref_mask = _check_frequency_masking(_ref_mono, sr)
            _REL_TOL = 0.08  # Toleranz: Pipeline darf max. 0.08 schlechter sein
            _inherited["comb_filter"] = comb_filter >= _ref_comb - _REL_TOL
            _inherited["musical_compression"] = musical_comp >= _ref_comp - _REL_TOL
            _inherited["masking_health"] = masking >= _ref_mask - _REL_TOL

    # Effektive Score-Werte: geerbte Artefakte nicht bestrafen — die Zone
    # zählt mit Grünzone-Wert, der Messwert bleibt im Resultat ehrlich sichtbar.
    _comb_eff = comb_filter if not _inherited.get("comb_filter") else max(comb_filter, GREEN_ZONE["comb_filter"])
    _comp_eff = (
        musical_comp
        if not _inherited.get("musical_compression")
        else max(musical_comp, GREEN_ZONE["musical_compression"])
    )
    _mask_eff = masking if not _inherited.get("masking_health") else max(masking, GREEN_ZONE["masking_health"])

    # ── Grünzone-Check ──
    greens = {
        "hpe": hpe >= GREEN_ZONE["hpe"],
        "inviting": inviting >= GREEN_ZONE["inviting"],
        "transparency": transparency >= GREEN_ZONE["transparency"],
        "goosebumps": goosebumps >= GREEN_ZONE["goosebumps"],
        "comb_filter": comb_filter >= GREEN_ZONE["comb_filter"] or _inherited.get("comb_filter", False),
        "musical_compression": musical_comp >= GREEN_ZONE["musical_compression"]
        or _inherited.get("musical_compression", False),
        "masking_health": masking >= GREEN_ZONE["masking_health"] or _inherited.get("masking_health", False),
        "spectral_color": spectral_color >= GREEN_ZONE["spectral_color"],
        "microdynamics": microdynamics >= GREEN_ZONE["microdynamics"],
    }
    green_count = sum(1 for v in greens.values() if v)
    all_green = all(greens.values())

    # ── Gesamt-Score: gewichtet nach Wichtigkeit für „Hineinlegen" ──
    # HPE und Inviting sind die dominanten Faktoren — sie bestimmen,
    # ob das Ohr BLEIBT oder geht.
    score = float(
        np.clip(
            hpe * 0.25
            + inviting * 0.20
            + transparency * 0.15
            + goosebumps * 0.10
            + _comb_eff * 0.10
            + _comp_eff * 0.10
            + _mask_eff * 0.10
            + spectral_color * 0.05
            + microdynamics * 0.05,
            0.0,
            1.0,
        )
    )

    # ── Label & Empfehlung ──
    warnings = []
    if not greens["hpe"]:
        warnings.append(f"HPE zu niedrig ({hpe:.2f})")
    if not greens["inviting"]:
        warnings.append(f"Klang weist Ohr zurück ({inviting:.2f})")
    if not greens["transparency"]:
        warnings.append(f"Künstlicher Klang ({transparency:.2f})")
    if not greens["goosebumps"]:
        warnings.append(f"Emotional flach ({goosebumps:.2f})")
    if not greens["comb_filter"] and not _inherited.get("comb_filter"):
        warnings.append(f"Kammfilter-Artefakte ({comb_filter:.2f})")
    if not greens["musical_compression"] and not _inherited.get("musical_compression"):
        warnings.append(f"Bösartige Kompression ({musical_comp:.2f})")
    if not greens["masking_health"] and not _inherited.get("masking_health"):
        warnings.append(f"Frequenz-Maskierung ({masking:.2f})")
    if not greens["spectral_color"]:
        warnings.append(f"Klangfarbe veraendert ({spectral_color:.2f})")
    if not greens["microdynamics"]:
        warnings.append(f"Mikrodynamik verloren ({microdynamics:.2f})")

    if all_green:
        label = "Sweet Spot"
        rec = "Perfekt — das Ohr legt sich hinein und will nie wieder weg."
    elif green_count >= 5:
        label = "Fast perfekt"
        rec = warnings[0] if warnings else "Kleine Optimierungen möglich."
    elif green_count >= 3:
        label = "Optimierungsbedarf"
        rec = f"{len(warnings)} von 7 Bereichen brauchen Verbesserung: {warnings[0]}"
    else:
        label = "Durchgefallen"
        rec = f"Deutliche Überarbeitung nötig — {green_count}/7 Metriken in Grünzone."

    return SweetSpotResult(
        score=score,
        all_green=all_green,
        hpe_score=hpe,
        inviting_score=inviting,
        transparency_score=transparency,
        goosebumps_score=goosebumps,
        comb_filter=comb_filter,
        musical_compression=musical_comp,
        masking_health=masking,
        spectral_color=spectral_color,
        microdynamics=microdynamics,
        green_count=green_count,
        total_metrics=9,
        label=label,
        recommendation=rec,
        warnings=warnings,
    )


# ── Lücken-Schließer 1: KAMMFILTER ────────────────────────────────────


def _check_comb_filter(mono: np.ndarray, sr: int) -> float:
    """Erkennt Kammfilter-Effekt via periodischer Notch-Muster.

    Kammfilter entstehen durch Phasen-Interferenz (Delay < 20ms).
    Im Spektrum zeigen sie sich als periodische, äquidistante Einbrüche.
    """
    n_fft = 4096
    if len(mono) < n_fft:
        return 0.8

    spec = np.abs(np.fft.rfft(mono[:n_fft] * np.hanning(n_fft)))
    spec_db = 20.0 * np.log10(np.maximum(spec, 1e-12))

    # Autokorrelation des Spektrums: periodische Notches = Peaks in ACF
    spec_norm = spec_db - np.mean(spec_db)
    acf = np.correlate(spec_norm, spec_norm, mode="same")
    acf = acf[len(acf) // 2 :]  # Nur positive Lags
    acf_norm = acf / (acf[0] + 1e-12)

    # Suche regelmäßige Peaks in der ACF (Periode 10-100 Bins)
    peaks = 0
    for i in range(10, min(100, len(acf_norm) - 10)):
        if acf_norm[i] > 0.3 and acf_norm[i] > acf_norm[i - 1] and acf_norm[i] > acf_norm[i + 1]:
            # Prüfe ob der nächste Peak im erwarteten Abstand liegt
            for j in range(i + 5, min(i + 30, len(acf_norm) - 1)):
                if acf_norm[j] > 0.2 and acf_norm[j] > acf_norm[j - 1] and acf_norm[j] > acf_norm[j + 1]:
                    peaks += 1
                    break

    if peaks == 0:
        return 0.95
    elif peaks <= 2:
        return 0.65
    elif peaks <= 5:
        return 0.35
    return 0.10


# ── Lücken-Schließer 2: MUSIKALISCHE vs BÖSARTIGE KOMPRESSION ──────────


def _check_musical_compression(mono: np.ndarray, sr: int) -> float:
    """Unterscheidet musikalische von bösartiger (Loudness-War) Kompression.

    Unterschied:
    - Musikalisch: Krestfaktor 12-18dB, Transienten erhalten, Mikrodynamik > 2dB
    - Loudness-War: Krestfaktor 6-10dB, keine Transienten, Mikrodynamik < 1dB
    """
    win = int(0.01 * sr)  # 10ms
    if len(mono) < 100 * win:
        return 0.7

    rms_vals = [float(np.sqrt(np.mean(mono[i : i + win] ** 2))) for i in range(0, len(mono) - win, win)]
    rms_db = 20.0 * np.log10(np.array(rms_vals) + 1e-12)

    # Krestfaktor = Peak/RMS über das gesamte Signal
    peak_linear = float(np.max(np.abs(mono)))
    rms_linear = float(np.sqrt(np.mean(mono**2)))
    crest_factor = 20.0 * np.log10((peak_linear + 1e-12) / (rms_linear + 1e-12))

    # Mikrodynamik: Median der RMS-Differenzen
    diffs = np.abs(np.diff(rms_db))
    micro_dyn = float(np.median(diffs[diffs > 0.3])) if np.any(diffs > 0.3) else 0.0

    # Transienten-Erhalt: Anteil scharfer Anstiege (>8dB)
    sharp_rises = np.sum(diffs > 8.0) / max(len(diffs), 1)

    # Bewertung
    score_cf = 1.0
    if crest_factor > 18:
        score_cf = 0.95
    elif crest_factor > 14:
        score_cf = 0.85
    elif crest_factor > 10:
        score_cf = 0.55
    elif crest_factor > 7:
        score_cf = 0.25
    else:
        score_cf = 0.10

    score_md = 1.0
    if micro_dyn > 3.0:
        score_md = 0.95
    elif micro_dyn > 1.5:
        score_md = 0.70
    elif micro_dyn > 0.5:
        score_md = 0.35
    else:
        score_md = 0.10

    score_tr = 1.0
    if sharp_rises > 0.02:
        score_tr = 0.90
    elif sharp_rises > 0.005:
        score_tr = 0.55
    else:
        score_tr = 0.25

    return float(np.clip(score_cf * 0.40 + score_md * 0.35 + score_tr * 0.25, 0.0, 1.0))


# ── Lücken-Schließer 3: FREQUENZ-MASKIERUNG ──────────────────────────


def _check_frequency_masking(mono: np.ndarray, sr: int) -> float:
    """Erkennt Frequenz-Maskierung zwischen Instrumenten.

    In natürlicher Musik sind die Bark-Bänder gleichmäßig besetzt.
    Starke Maskierung zeigt sich als Energie-Ballung in wenigen Bändern
    bei gleichzeitiger Leere in anderen.
    """
    n_fft = 4096
    if len(mono) < 2 * n_fft:
        return 0.7

    # Bark-Bänder
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

    # Analyse über mehrere Frames
    hop = n_fft // 4
    band_energies_all = []
    for i in range(0, len(mono) - n_fft, hop):
        frame = mono[i : i + n_fft] * np.hanning(n_fft)
        spec = np.abs(np.fft.rfft(frame))

        band_energies = []
        for j in range(len(bark_edges) - 1):
            mask = (freqs >= bark_edges[j]) & (freqs < bark_edges[j + 1])
            if mask.sum() > 0:
                band_energies.append(float(np.sum(spec[mask] ** 2)))
            else:
                band_energies.append(0.0)
        band_energies = np.array(band_energies)  # type: ignore[assignment]
        band_energies /= np.sum(band_energies) + 1e-12
        band_energies_all.append(band_energies)

    if len(band_energies_all) < 3:
        return 0.7

    # Durchschnittliche Band-Energie über alle Frames
    avg_bands = np.mean(band_energies_all, axis=0)

    # Gini-Koeffizient der Band-Verteilung:
    # 0 = perfekt gleichmäßig (alle Bänder gleich)
    # 1 = extrem ungleich (ein Band dominiert alles → Maskierung)
    sorted_bands = np.sort(avg_bands)
    n = len(sorted_bands)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_bands)) / (n * np.sum(sorted_bands)) - (n + 1) / n

    # Interpretation (invertiert: hoher Gini = schlecht)
    if gini < 0.30:
        return 0.90  # Gleichmäßig → gute Trennung
    elif gini < 0.40:
        return 0.70
    elif gini < 0.50:
        return 0.45
    elif gini < 0.60:
        return 0.25
    return 0.10


# ── Hilfsfunktionen zur Metrik-Extraktion ──────────────────────────────


def _get_hpe(mono: np.ndarray, sr: int) -> float:
    try:
        from backend.core.human_pleasantness_estimator import compute_pleasantness

        return float(compute_pleasantness(mono, sr).score)
    except Exception as e:
        logger.warning("sweet_spot_optimizer.py::_get_hpe Ersatzpfad: %s", e)
        return 0.5


def _get_inviting(mono: np.ndarray, sr: int) -> float:
    try:
        from backend.core.inviting_sound_checker import check_inviting_sound

        return float(check_inviting_sound(mono, sr).score)
    except Exception as e:
        logger.warning("sweet_spot_optimizer.py::_get_inviting Ersatzpfad: %s", e)
        return 0.5


def _get_transparency(mono: np.ndarray, sr: int, arr: np.ndarray) -> float:
    try:
        from backend.core.transparency_guard import check_transparency

        return float(check_transparency(mono.astype(np.float32), sr).score)
    except Exception as e:
        logger.warning("sweet_spot_optimizer.py::_get_transparency Ersatzpfad: %s", e)
        return 0.5


def _get_goosebumps(mono: np.ndarray, sr: int) -> float:
    try:
        from backend.core.goosebumps_factor import compute_goosebumps

        return float(compute_goosebumps(mono, sr).score)
    except Exception as e:
        logger.warning("sweet_spot_optimizer.py::_get_goosebumps Ersatzpfad: %s", e)
        return 0.5


def _check_spectral_color(mono, sr):
    """Klangfarben-Erhalt via frame-to-frame Spektral-Korrelation."""
    n_fft = 2048
    hop = n_fft // 2
    if len(mono) < 4 * n_fft:
        return 0.8
    specs = [np.abs(np.fft.rfft(mono[i : i + n_fft] * np.hanning(n_fft))) for i in range(0, len(mono) - n_fft, hop)]
    specs = [s / (np.sum(s) + 1e-12) for s in specs]
    if len(specs) < 3:
        return 0.8
    corrs = [np.corrcoef(specs[i - 1], specs[i])[0, 1] for i in range(1, len(specs))]
    corrs = [c for c in corrs if np.isfinite(c)]
    return float(np.clip(np.mean(corrs) if corrs else 0.5, 0.0, 1.0))


def _check_microdynamics(mono, sr):
    """Mikrodynamik via RMS-Differenzen in 50ms-Fenstern."""
    win = int(0.05 * sr)
    if len(mono) < 20 * win:
        return 0.7
    rms = [np.sqrt(np.mean(mono[i : i + win] ** 2)) for i in range(0, len(mono) - win, win)]
    rms_db = 20 * np.log10(np.array(rms) + 1e-12)
    diffs = np.abs(np.diff(rms_db))
    md = float(np.median(diffs[diffs > 0.3])) if np.any(diffs > 0.3) else 0.0
    if md > 3.0:
        return 0.95
    elif md > 1.5:
        return 0.75
    elif md > 0.5:
        return 0.45
    return 0.15
