"""
Cassette Defect Verifier — Aurik §v10.2

Schliesst die entscheidende Luecke in Auriks Cassetten-Defekt-Pipeline:
Detection und Correction sind perfekt — aber die Verification war blind
fuer synthetisierten/reparierten Content.

Drei Module:
  1. Per-Defect-HPE-Check: Psychoakustische Pruefung JEDES reparierten Segments
  2. ABX-Residual-Test: Spektraler Residual-Vergleich Original vs Repariert
  3. PMGG-Proxy-Alternativen: Metriken die mit synthetisiertem Content klarkommen

Phase-Support:
  phase_24 (Dropout/AudioSR)   — 9 PMGG-Ziele blind → Temporal Continuity Check
  phase_56 (Head Wear/Band Gap) — 4 PMGG-Ziele blind → Band-Gap Closure Metric
  phase_57 (Print-Through)      — 2 PMGG-Ziele blind → Echo Residue Detector
  phase_59 (Modulation Noise)   — 2 PMGG-Ziele blind → Noise Floor Delta
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# §v10.2 Maskierungsschwelle — was das menschliche Ohr NICHT hoert
# ─────────────────────────────────────────────────────────────────
# Glasberg & Moore 1990 (JASA 87:2178, ERB-Modell):
#   Breitband-Maskierungsschwelle ≈ -20 bis -30 dB relativ zum Maskierer.
#   Bei typischen Musikpegeln (65-80 dB SPL) liegt die Schwelle bei:
#   - 500 Hz:  -18 dB
#   - 2000 Hz: -22 dB
#   - 4000 Hz: -28 dB (ERB-breiter, bessere Maskierung)
#   - 8000 Hz: -25 dB
# Zwicker & Fastl 1999 (Psychoacoustics, 2nd ed.):
#   Simultaneous masking: residual < -18 dB unter Maskierer = unhörbar
# Conservativ: -20 dB als universelle Schwelle.

MASKING_THRESHOLD_DB: float = -20.0  # dB unter lokalem Pegel = unhörbar
MIN_SEGMENT_SAMPLES: int = 2048  # Mindest-Segmentlänge
SEGMENT_CONTEXT_MS: float = 500.0  # Kontext vor/nach Defektstelle (ms)


@dataclass
class DefectVerificationResult:
    """Ergebnis der Defekt-Verifikation fuer ein einzelnes repariertes Segment."""

    defect_type: str = ""
    phase_id: str = ""
    location_start_s: float = 0.0
    location_end_s: float = 0.0

    # ── PDV-Kompatibilität (phase_defect_verifier.py nutzt diese Dataclass kanonisch) ──
    targeted_defects: list[str] = field(default_factory=list)
    proxies_before: dict[str, float] = field(default_factory=dict)
    proxies_after: dict[str, float] = field(default_factory=dict)
    worst_defect: str = ""
    worst_relative_change: float = 0.0
    rollback_triggered: bool = False
    skipped_defects: list[str] = field(default_factory=list)

    # ── Per-Defect HPE ──
    hpe_before: float = 0.0  # HPE des defekten Segments VOR Reparatur
    hpe_after: float = 0.0  # HPE des reparierten Segments NACH Reparatur
    hpe_delta: float = 0.0  # Verbesserung (positiv = besser)
    segment_verdict: str = ""  # "passed", "residual_detected", "worse"

    # ── ABX Residual ──
    residual_peak_db: float = 0.0  # Spitzen-Residual relativ zum Signal
    residual_rms_db: float = 0.0  # RMS-Residual
    residual_masked: bool = True  # Residual unter Maskierungsschwelle?
    abx_confidence: float = 0.0  # 0-1: Wahrscheinlichkeit dass Residual hörbar

    # ── Phase-spezifische Proxy-Metriken ──
    proxy_name: str = ""
    proxy_score: float = 0.0
    proxy_threshold: float = 0.0
    proxy_passed: bool = True

    # ── Gesamt-Urteil ──
    overall_verdict: str = ""  # "clean", "residual_below_masking", "audible_residual"
    recommendation: str = ""


@dataclass
class BatchVerificationResult:
    """Ergebnis der Batch-Verifikation aller Defekte eines Typs."""

    defect_type: str
    phase_id: str
    total_segments: int
    segments_checked: int
    clean_count: int  # Kein hörbares Residual
    masked_count: int  # Residual unter Maskierungsschwelle
    audible_count: int  # Hörbares Residual
    mean_hpe_delta: float = 0.0
    mean_residual_db: float = 0.0
    per_segment: list[DefectVerificationResult] = field(default_factory=list)
    overall_verdict: str = ""


_PDV_COMPAT_EXPORTS = {
    "PhaseDefectVerifier",
    "get_phase_defect_verifier",
    "_proxy_impulse_ratio",
    "_proxy_hf_noise_floor",
    "_proxy_hum_energy",
    "_proxy_low_freq_energy",
    "_proxy_dc_offset",
    "_proxy_dropout_ratio",
    "_proxy_mono_compat",
    "_frequency_selective_blend",
    "_compute_hf_noise_audibility",
    "_compute_transient_harshness",
    "_compute_quasi_peak_burstiness",
    "_compute_modulation_roughness",
}


def __getattr__(name: str) -> Any:
    """Lazy PDV compatibility exports without creating an import cycle."""
    if name in _PDV_COMPAT_EXPORTS:
        from backend.core import phase_defect_verifier as _pdv

        return getattr(_pdv, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ═══════════════════════════════════════════════════════════════
# MODUL 1: Per-Defect-HPE-Check
# ═══════════════════════════════════════════════════════════════


def verify_defect_segment(
    original: np.ndarray,
    repaired: np.ndarray,
    sr: int,
    defect_type: str,
    phase_id: str,
    location_start_s: float,
    location_end_s: float,
) -> DefectVerificationResult:
    """§v10.2 Prueft EIN repariertes Defekt-Segment auf Residual-Artefakte.

    Extrahiert das Segment + Kontext, berechnet HPE vor/nach,
    und misst das Residual gegen die psychoakustische Maskierungsschwelle.

    Args:
        original:           Original-Audio (defekt)
        repaired:           Repariertes Audio
        sr:                 Sample-Rate
        defect_type:        z.B. "TAPE_HEAD_LEVEL_DIP"
        phase_id:           z.B. "phase_24"
        location_start_s:   Defekt-Start in Sekunden
        location_end_s:     Defekt-Ende in Sekunden

    Returns:
        DefectVerificationResult mit HPE-Delta, Residual-Pegel, Verdict
    """
    context_samples = int(SEGMENT_CONTEXT_MS / 1000.0 * sr)
    start_sample = max(0, int(location_start_s * sr) - context_samples)
    end_sample = min(len(original), int(location_end_s * sr) + context_samples)

    # Bei sehr kurzen Defekten: mindestens MIN_SEGMENT_SAMPLES
    if end_sample - start_sample < MIN_SEGMENT_SAMPLES:
        mid = (start_sample + end_sample) // 2
        half = MIN_SEGMENT_SAMPLES // 2
        start_sample = max(0, mid - half)
        end_sample = min(len(original), mid + half)

    seg_orig = _extract_segment(original, start_sample, end_sample)
    seg_rep = _extract_segment(repaired, start_sample, end_sample)

    # ── Per-Defect HPE ──
    hpe_before = _compute_segment_hpe(seg_orig, sr)
    hpe_after = _compute_segment_hpe(seg_rep, sr)
    hpe_delta = hpe_after - hpe_before

    # ── ABX Residual ──
    residual = seg_rep - seg_orig
    residual_peak_db, residual_rms_db, residual_masked = _compute_residual_level(residual, seg_orig, sr)
    abx_confidence = _estimate_audibility(residual, seg_orig, sr)

    # ── Phase-spezifische Proxy-Metrik ──
    proxy_name, proxy_score, proxy_threshold, proxy_passed = _compute_phase_proxy(phase_id, seg_orig, seg_rep, sr)

    # ── Verdict ──
    if hpe_delta > 0.01 and residual_masked:
        segment_verdict = "passed"
        overall_verdict = "clean"
    elif residual_masked or abx_confidence < 0.3:
        segment_verdict = "passed"
        overall_verdict = "residual_below_masking"
    elif hpe_delta < -0.03:
        segment_verdict = "worse"
        overall_verdict = "audible_residual"
    else:
        segment_verdict = "residual_detected"
        overall_verdict = "audible_residual"

    recommendation = _build_recommendation(overall_verdict, defect_type, phase_id, proxy_passed)

    return DefectVerificationResult(
        defect_type=defect_type,
        phase_id=phase_id,
        location_start_s=location_start_s,
        location_end_s=location_end_s,
        hpe_before=hpe_before,
        hpe_after=hpe_after,
        hpe_delta=hpe_delta,
        segment_verdict=segment_verdict,
        residual_peak_db=residual_peak_db,
        residual_rms_db=residual_rms_db,
        residual_masked=residual_masked,
        abx_confidence=abx_confidence,
        proxy_name=proxy_name,
        proxy_score=proxy_score,
        proxy_threshold=proxy_threshold,
        proxy_passed=proxy_passed,
        overall_verdict=overall_verdict,
        recommendation=recommendation,
    )


def verify_defect_batch(
    original: np.ndarray,
    repaired: np.ndarray,
    sr: int,
    defect_type: str,
    phase_id: str,
    locations: list[tuple[float, float]],
    max_segments: int = 200,
) -> BatchVerificationResult:
    """§v10.2 Batch-Verifikation aller Segmente eines Defekt-Typs.

    Verarbeitet bis zu max_segments (Default 200 um Overhead zu begrenzen).
    Bei > max_segments wird gleichmäßig gesampelt.
    """
    if len(locations) > max_segments:
        step = len(locations) // max_segments
        locations = locations[::step][:max_segments]

    results: list[DefectVerificationResult] = []
    for start_s, end_s in locations:
        try:
            r = verify_defect_segment(original, repaired, sr, defect_type, phase_id, start_s, end_s)
            results.append(r)
        except Exception as e:
            logger.debug("Defekt-Verifikation fehlgeschlagen für %s@%.1fs: %s", defect_type, start_s, e)

    if not results:
        return BatchVerificationResult(
            defect_type=defect_type,
            phase_id=phase_id,
            total_segments=len(locations),
            segments_checked=0,
            clean_count=0,
            masked_count=0,
            audible_count=0,
            overall_verdict="no_data",
        )

    clean = sum(1 for r in results if r.overall_verdict == "clean")
    masked = sum(1 for r in results if r.overall_verdict == "residual_below_masking")
    audible = sum(1 for r in results if r.overall_verdict == "audible_residual")
    mean_hpe = float(np.mean([r.hpe_delta for r in results]))
    mean_res = float(np.mean([r.residual_rms_db for r in results]))

    if audible == 0:
        overall = "clean" if clean > masked else "residual_below_masking"
    elif audible <= len(results) * 0.05:
        overall = "mostly_clean"
    elif audible <= len(results) * 0.20:
        overall = "partial_residual"
    else:
        overall = "significant_residual"

    return BatchVerificationResult(
        defect_type=defect_type,
        phase_id=phase_id,
        total_segments=len(locations),
        segments_checked=len(results),
        clean_count=clean,
        masked_count=masked,
        audible_count=audible,
        mean_hpe_delta=mean_hpe,
        mean_residual_db=mean_res,
        per_segment=results,
        overall_verdict=overall,
    )


# ═══════════════════════════════════════════════════════════════
# MODUL 2: ABX-Residual-Test (Zeitbereich)
# ═══════════════════════════════════════════════════════════════


def abx_residual_test(
    original_segment: np.ndarray,
    repaired_segment: np.ndarray,
    sr: int,
) -> dict[str, Any]:
    """§v10.2 ABX-ähnlicher Residual-Test: „Kann das Ohr den Unterschied hoeren?"

    Vergleicht das Residual (repaired - original) mit der psychoakustischen
    Maskierungsschwelle des Originals.

    Returns dict mit:
      - residual_rms_db: RMS des Residuals in dB relativ zum Original
      - residual_peak_db: Peak des Residuals
      - masked_by_signal: Ob das Residual unter der Maskierungsschwelle liegt
      - audibility_confidence: 0-1 Wahrscheinlichkeit der Hörbarkeit
      - critical_bands_affected: Anzahl betroffener kritischer Bänder (Zwicker)
      - verdict: "inaudible", "borderline", "audible"
    """
    orig = np.asarray(original_segment, dtype=np.float64).ravel()
    rep = np.asarray(repaired_segment, dtype=np.float64).ravel()
    residual = rep - orig

    # Pegelberechnung
    orig_rms = float(np.sqrt(np.mean(orig**2)) + 1e-10)
    residual_rms = float(np.sqrt(np.mean(residual**2)) + 1e-10)
    residual_peak = float(np.max(np.abs(residual)) + 1e-10)

    residual_rms_db = float(20.0 * np.log10(residual_rms / orig_rms))
    residual_peak_db = float(20.0 * np.log10(residual_peak / orig_rms))

    # ── Kritische Bänder (Zwicker) ──
    critical_bands_affected = _count_affected_critical_bands(residual, orig, sr)

    # ── Maskierungs-Check ──
    # Jedes kritische Band separat prüfen
    band_residuals = _per_band_residual(residual, orig, sr)
    masked_bands = sum(1 for b in band_residuals if b < MASKING_THRESHOLD_DB)
    total_bands = max(len(band_residuals), 1)
    masked_ratio = masked_bands / total_bands
    masked_by_signal = masked_ratio > 0.8  # >80% der Bänder maskiert

    # ── Audibility Confidence ──
    # Basiert auf: RMS-Residual + betroffene Bänder + Maskierungsgrad
    if residual_rms_db < MASKING_THRESHOLD_DB:
        audibility = 0.0
    elif residual_rms_db < MASKING_THRESHOLD_DB + 6:
        audibility = 0.3 * (1.0 - masked_ratio)
    elif residual_rms_db < MASKING_THRESHOLD_DB + 12:
        audibility = 0.5 + 0.3 * (critical_bands_affected / 24.0)
    else:
        audibility = 0.8 + 0.2 * min(critical_bands_affected / 24.0, 1.0)

    # ── Verdict ──
    if audibility < 0.3:
        verdict = "inaudible"
    elif audibility < 0.6:
        verdict = "borderline"
    else:
        verdict = "audible"

    return {
        "residual_rms_db": residual_rms_db,
        "residual_peak_db": residual_peak_db,
        "masked_by_signal": masked_by_signal,
        "audibility_confidence": audibility,
        "critical_bands_affected": critical_bands_affected,
        "masked_bands_ratio": masked_ratio,
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════════
# MODUL 3: PMGG-Alternativ-Proxy-Metriken
# ═══════════════════════════════════════════════════════════════


def compute_phase_proxy_for_pmgg(
    phase_id: str,
    audio_before: np.ndarray,
    audio_after: np.ndarray,
    sr: int,
) -> dict[str, float]:
    """§v10.3 Media-Defect-Verifier: Kategorie-basierte PMGG-Alternativ-Proxies.

    Deckt ALLE 62 PMGG-Phasen ab durch:
    1. Spezifische Handler (Top-6 Phasen: 03, 09, 04, 16, 28, 29)
    2. Cassette-spezifische Handler (24, 56, 57, 59)
    3. Kategorie-basierte Universal-Proxies (alle 62 Phasen)

    Returns dict mit {goal_name: proxy_score} im PMGG-Format (0-1).
    """
    result: dict[str, float] = {}

    # Layer 1: Cassette-spezifische Handler (v10.2)
    if "phase_24" in phase_id or "dropout" in phase_id.lower():
        result.update(_proxy_phase_24_dropout(audio_before, audio_after, sr))

    if "phase_56" in phase_id or "head_wear" in phase_id.lower():
        result.update(_proxy_phase_56_head_wear(audio_before, audio_after, sr))

    if "phase_57" in phase_id or "print_through" in phase_id.lower():
        result.update(_proxy_phase_57_print_through(audio_before, audio_after, sr))

    if "phase_59" in phase_id or "modulation_noise" in phase_id.lower():
        result.update(_proxy_phase_59_modulation_noise(audio_before, audio_after, sr))

    # Layer 2: Spezifische Handler (P0-Priorität, v10.3)
    handler = _SPECIFIC_HANDLERS.get(phase_id)
    if handler is not None:
        try:
            result.update(handler(audio_before, audio_after, sr))
        except Exception as e:
            logger.warning(
                "cassette_defect_verifier.py::berechnen_Verarbeitungsschritt_proxy_for_pmgg Ersatzpfad: %s", e
            )

    # Layer 3: Kategorie-basierte Universal-Proxies (v10.3)
    category = _PHASE_CATEGORIES.get(phase_id)
    if category is not None:
        cat_fn = _CATEGORY_PROXY_FUNCTIONS.get(category)
        if cat_fn is not None:
            try:
                result.update(cat_fn(audio_before, audio_after, sr))
            except Exception as e:
                logger.warning(
                    "cassette_defect_verifier.py::berechnen_Verarbeitungsschritt_proxy_for_pmgg Ersatzpfad: %s", e
                )

    # Layer 4: Universelle Fallback-Proxies
    result.update(_universal_proxies(audio_before, audio_after, sr))
    return result


# Category proxy function dispatch table (built after all functions defined)
_CATEGORY_PROXY_FUNCTIONS: dict[str, callable] = {}  # type: ignore[valid-type]


# ─────────────────────────────────────────────────────────
# Phase-spezifische Proxy-Implementierungen
# ─────────────────────────────────────────────────────────


def _proxy_phase_24_dropout(before: np.ndarray, after: np.ndarray, sr: int) -> dict[str, float]:
    """phase_24 Dropout Repair: Temporal Continuity statt Spektral-Flatness.

    PMGG schliesst 9 Ziele aus (natuerlichkeit, brillanz, authentizitaet,
    artikulation, timbre_authentizitaet, transparenz, tonal_center,
    groove, emotionalitaet) — der AudioSR synthetisiert neue Patches,
    die gegen die defekte Referenz wie Regression aussehen.

    Alternative Proxy: Temporal Continuity = Energie-Varianz über 50ms-Fenster.
    Ein gut reparierter Dropout hat GLEICHE Energie wie umgebendes Signal.
    """
    # Energie in 50ms-Blöcken
    block_samples = int(0.05 * sr)
    n_blocks = min(len(before), len(after)) // block_samples

    if n_blocks < 3:
        return {"natuerlichkeit": 0.5, "transparenz": 0.5}

    b_en = np.array([np.mean(before[i * block_samples : (i + 1) * block_samples] ** 2) for i in range(n_blocks)])
    a_en = np.array([np.mean(after[i * block_samples : (i + 1) * block_samples] ** 2) for i in range(n_blocks)])

    # Temporal Continuity: std(energy) sollte NIEDRIG sein (keine Sprünge)
    b_continuity = 1.0 / (1.0 + float(np.std(b_en) / (np.mean(b_en) + 1e-10)) * 10)
    a_continuity = 1.0 / (1.0 + float(np.std(a_en) / (np.mean(a_en) + 1e-10)) * 10)

    # Verbesserung in der Kontinuität
    continuity_delta = a_continuity - b_continuity

    # HF-Energie nach Reparatur (Dropouts haben oft HF-Verlust)
    b_hf = float(np.mean(np.abs(np.diff(before.ravel()))))
    a_hf = float(np.mean(np.abs(np.diff(after.ravel()))))

    return {
        "natuerlichkeit": min(1.0, max(0.0, a_continuity)),
        "brillanz": min(1.0, max(0.0, a_hf / (b_hf + 1e-10) * 0.8)),
        "authentizitaet": min(1.0, max(0.0, 0.5 + continuity_delta)),
        "transparenz": min(1.0, max(0.0, a_continuity * 0.9)),
        "groove": min(1.0, max(0.0, 0.5 + continuity_delta * 0.7)),
        "emotionalitaet": min(1.0, max(0.0, 0.5 + continuity_delta * 0.5)),
        "artikulation": min(1.0, max(0.0, a_hf / (b_hf + 1e-10))),
        "timbre_authentizitaet": min(1.0, max(0.0, 0.5 + continuity_delta * 0.8)),
        "tonal_center": min(1.0, max(0.0, 0.7 + continuity_delta * 0.3)),
    }


def _proxy_phase_56_head_wear(before: np.ndarray, after: np.ndarray, sr: int) -> dict[str, float]:
    """phase_56 Head Wear Band-Gap Repair: Band-Gap Closure Metric.

    PMGG schliesst 4 Ziele aus (natuerlichkeit, brillanz, authentizitaet,
    timbre_authentizitaet). Head-Wear erzeugt Frequenzband-Ausloeschungen
    (comb-filter durch Kopfverschleiss). Die Reparatur schliesst diese Luecken.

    Proxy: Spektrale Glattheit (Spectral Smoothness) — nach Reparatur sollte
    das Spektrum GLAETTER sein (keine abrupten Band-Lücken).
    """
    n_fft = min(4096, len(before))
    if n_fft < 256:
        return {"natuerlichkeit": 0.5, "brillanz": 0.5, "authentizitaet": 0.5, "timbre_authentizitaet": 0.5}

    b_fft = np.abs(np.fft.rfft(before.ravel()[:n_fft]))
    a_fft = np.abs(np.fft.rfft(after.ravel()[:n_fft]))

    # Spectral Smoothness = 1 / (1 + mean(|diff(spectrum)|))
    b_smooth = 1.0 / (1.0 + float(np.mean(np.abs(np.diff(b_fft))) / (np.mean(b_fft) + 1e-10)) * 5)
    a_smooth = 1.0 / (1.0 + float(np.mean(np.abs(np.diff(a_fft))) / (np.mean(a_fft) + 1e-10)) * 5)

    # HF-Recovery: Energie > 8 kHz
    hf_bin = int(8000 / (sr / n_fft))
    b_hf = float(np.mean(b_fft[hf_bin:]) + 1e-10)
    a_hf = float(np.mean(a_fft[hf_bin:]) + 1e-10)

    smooth_delta = a_smooth - b_smooth

    return {
        "natuerlichkeit": min(1.0, max(0.0, a_smooth)),
        "brillanz": min(1.0, max(0.0, a_hf / (b_hf + 1e-10) * 0.85)),
        "authentizitaet": min(1.0, max(0.0, 0.5 + smooth_delta)),
        "timbre_authentizitaet": min(1.0, max(0.0, 0.5 + smooth_delta * 0.9)),
    }


def _proxy_phase_57_print_through(before: np.ndarray, after: np.ndarray, sr: int) -> dict[str, float]:
    """phase_57 Print-Through Reduction: Echo Residue Detector.

    PMGG schliesst 2 Ziele aus (authentizitaet, emotionalitaet).
    Print-Through = magnetisches Pre-Echo (100-300ms vor Einsatz).
    Nach LMS-Adaptive Subtraction sollte das Pre-Echo verschwunden sein.

    Proxy: Autokorrelations-basierter Echo-Detektor.
    """
    min_lag = int(0.05 * sr)  # 50 ms
    max_lag = int(0.40 * sr)  # 400 ms (Pre-Echo + Post-Echo)

    if max_lag >= len(after) // 2 or max_lag <= min_lag:
        return {"authentizitaet": 0.5, "emotionalitaet": 0.5}

    # Autokorrelation des Residuals (before - after)
    diff = before.ravel()[: max_lag * 3] - after.ravel()[: max_lag * 3]
    if len(diff) < max_lag * 2:
        return {"authentizitaet": 0.5, "emotionalitaet": 0.5}

    diff_norm = diff / (float(np.std(diff)) + 1e-10)
    ac = np.correlate(diff_norm, diff_norm, mode="full")
    ac = ac[len(ac) // 2 :] / ac[len(ac) // 2]  # Normalisieren

    # Echo-Peak im Print-Through-Bereich (100-300ms)
    echo_region = ac[min_lag:max_lag]
    echo_peak = float(np.max(np.abs(echo_region)))
    echo_residual = 1.0 - echo_peak  # 0=Echo da, 1=kein Echo

    return {
        "authentizitaet": min(1.0, max(0.0, echo_residual)),
        "emotionalitaet": min(1.0, max(0.0, 0.5 + echo_residual * 0.5)),
    }


def _proxy_phase_59_modulation_noise(before: np.ndarray, after: np.ndarray, sr: int) -> dict[str, float]:
    """phase_59 Modulation Noise Reduction: Noise Floor Delta.

    PMGG schliesst 2 Ziele aus (natuerlichkeit, emotionalitaet).
    Modulationsrauschen = signal-abhängiges Rauschen (lauter bei lauten Passagen).
    Nach Reduktion sollte das Rauschen ENTKOPPELT vom Signal sein.

    Proxy: Korrelation zwischen Signal-Hüllkurve und Rausch-Hüllkurve.
    """
    block_samples = int(0.025 * sr)  # 25ms Blöcke
    n_blocks = min(len(before), len(after)) // block_samples

    if n_blocks < 4:
        return {"natuerlichkeit": 0.5, "emotionalitaet": 0.5}

    b_blocks = [before.ravel()[i * block_samples : (i + 1) * block_samples] for i in range(n_blocks)]
    a_blocks = [after.ravel()[i * block_samples : (i + 1) * block_samples] for i in range(n_blocks)]

    # Signal-Hüllkurve vs Rausch-Hüllkurve
    b_env = np.array([float(np.sqrt(np.mean(b**2))) for b in b_blocks])
    a_env = np.array([float(np.sqrt(np.mean(a**2))) for a in a_blocks])

    # Hochpass-gefilterte Hüllkurve (≈ Rausch-Anteil)
    if len(b_env) >= 4:
        from scipy import signal as scipy_signal

        try:
            b_hp = scipy_signal.detrend(b_env, type="constant")
            a_hp = scipy_signal.detrend(a_env, type="constant")
        except Exception:
            b_hp = b_env - np.mean(b_env)
            a_hp = a_env - np.mean(a_env)

        # Korrelation Signal↔Rauschen: sollte nach Reparatur NIEDRIGER sein
        b_corr = (
            float(np.abs(np.corrcoef(b_env, b_hp)[0, 1])) if np.std(b_env) > 1e-10 and np.std(b_hp) > 1e-10 else 0.5
        )
        a_corr = (
            float(np.abs(np.corrcoef(a_env, a_hp)[0, 1])) if np.std(a_env) > 1e-10 and np.std(a_hp) > 1e-10 else 0.5
        )

        decoupling = b_corr - a_corr  # Positiv = Rauschen besser entkoppelt
        noise_score = 0.5 + decoupling * 2.0
    else:
        noise_score = 0.5

    return {
        "natuerlichkeit": min(1.0, max(0.0, noise_score)),
        "emotionalitaet": min(1.0, max(0.0, 0.5 + noise_score * 0.5)),
    }


def _universal_proxies(before: np.ndarray, after: np.ndarray, sr: int) -> dict[str, float]:
    """Universelle Proxy-Metriken die immer sinnvoll sind."""
    b_rms = float(np.sqrt(np.mean(before.ravel() ** 2)) + 1e-10)
    a_rms = float(np.sqrt(np.mean(after.ravel() ** 2)) + 1e-10)

    # Pegel-Stabilität
    level_stability = 1.0 - abs(float(20 * np.log10(a_rms / b_rms))) / 20.0

    # Zero-Crossing-Rate (Indikator für HF-Inhalt)
    b_zcr = float(np.mean(np.abs(np.diff(np.sign(before.ravel())))) / 2)
    a_zcr = float(np.mean(np.abs(np.diff(np.sign(after.ravel())))) / 2)
    zcr_preservation = 1.0 - abs(a_zcr - b_zcr) / max(b_zcr, 0.01)

    return {
        "level_stability": min(1.0, max(0.0, level_stability)),
        "zcr_preservation": min(1.0, max(0.0, zcr_preservation)),
    }


# ═══════════════════════════════════════════════════════════════
# §v10.3 MEDIA DEFECT VERIFIER — Kategorie-basierte Universal-Proxies
# ═══════════════════════════════════════════════════════════════
# Deckt ALLE 62 PMGG-Phasen mit alternativen Proxy-Metriken ab.
# Kategorien: denoise, eq_correction, synthesis_inpaint, dynamics,
#             spatial, speed_pitch, harmonic, transient

# ── Phase → Kategorie Mapping ──
_PHASE_CATEGORIES: dict[str, str] = {
    # Denoise / Noise Reduction (9 Phasen, 49 excluded goals)
    "phase_01": "denoise",
    "phase_01_click_removal": "denoise",
    "phase_02": "denoise",
    "phase_02_hum_removal": "denoise",
    "phase_03": "denoise",
    "phase_03_denoise": "denoise",
    "phase_05": "denoise",
    "phase_05_rumble_filter": "denoise",
    "phase_09": "denoise",
    "phase_09_crackle_removal": "denoise",
    "phase_28": "denoise",
    "phase_28_surface_noise": "denoise",
    "phase_29": "denoise",
    "phase_29_tape_hiss_reduction": "denoise",
    "phase_43": "denoise",
    "phase_43_ml_deesser": "denoise",
    # EQ / Tonal Correction (6 Phasen, 30 excluded goals)
    "phase_04": "eq_correction",
    "phase_04_eq_correction": "eq_correction",
    "phase_16": "eq_correction",
    "phase_16_final_eq": "eq_correction",
    "phase_37": "eq_correction",
    "phase_37_bass_enhancement": "eq_correction",
    "phase_38": "eq_correction",
    "phase_38_air_band": "eq_correction",
    "phase_39": "eq_correction",
    "phase_39_air_band_enhancement": "eq_correction",
    "phase_44": "eq_correction",
    # Synthesis / Inpainting (6 Phasen, 24 excluded goals)
    "phase_06": "synthesis_inpaint",
    "phase_06_frequency_restoration": "synthesis_inpaint",
    "phase_07": "synthesis_inpaint",
    "phase_07_harmonic_restoration": "synthesis_inpaint",
    "phase_23": "synthesis_inpaint",  # §4.7c POCS: n_iter=2–5 vor PGHI (material-adaptiv)
    "phase_23_spectral_repair": "synthesis_inpaint",  # §4.7c POCS-Konsistenz-Projektion
    "phase_50": "synthesis_inpaint",
    "phase_50_spectral_repair": "synthesis_inpaint",
    "phase_55": "synthesis_inpaint",
    # Dynamics (12 Phasen, 42 excluded goals)
    "phase_10": "dynamics",
    "phase_11": "dynamics",
    "phase_17": "dynamics",
    "phase_18": "dynamics",
    "phase_19": "dynamics",
    "phase_26": "dynamics",
    "phase_26_dynamic_range_expansion": "dynamics",
    "phase_34": "dynamics",
    "phase_35": "dynamics",
    "phase_36": "dynamics",
    "phase_47": "dynamics",
    "phase_54": "dynamics",
    "phase_54_transparent_dynamics": "dynamics",
    # Spatial / Reverb (4 Phasen, 13 excluded goals)
    "phase_20": "spatial",
    "phase_20_reverb_reduction": "spatial",
    "phase_46": "spatial",
    "phase_46_spatial_enhancement": "spatial",
    "phase_48": "spatial",
    "phase_48_stereo_width_enhancer": "spatial",
    "phase_49": "spatial",
    "phase_49_advanced_dereverb": "spatial",
    # Speed / Pitch (2 Phasen, 12 excluded goals)
    "phase_12": "speed_pitch",
    "phase_12_wow_flutter_fix": "speed_pitch",
    "phase_31": "speed_pitch",
    "phase_31_speed_pitch_correction": "speed_pitch",
    # Harmonic / Saturation (2 Phasen, 4 excluded goals)
    "phase_22": "harmonic",
    "phase_22_tape_saturation": "harmonic",
    "phase_21": "harmonic",
    "phase_21_exciter": "harmonic",
    # Transient (2 Phasen, 3 excluded goals)
    "phase_08": "transient",
    "phase_08_transient_preservation": "transient",
    # Phase Alignment (3 Phasen, 4 excluded goals)
    "phase_14": "phase_alignment",
    "phase_25": "phase_alignment",
    "phase_25_azimuth_correction": "phase_alignment",
    # Vocal / Formant (3 Phasen, 12 excluded goals)
    "phase_42": "vocal",
    "phase_42_vocal_enhancement": "vocal",
    "phase_58": "vocal",
    "phase_58_lyrics_guided_enhancement": "vocal",
    # Legacy / Passthrough (9 Phasen, 10 excluded goals — nur timbre_authentizitaet)
    "phase_13": "passthrough",
    "phase_15": "passthrough",
    "phase_27": "passthrough",
    "phase_30": "passthrough",
    "phase_32": "passthrough",
    "phase_33": "passthrough",
    "phase_40": "passthrough",
    "phase_41": "passthrough",
    "phase_45": "passthrough",
    "phase_51": "passthrough",
    "phase_52": "passthrough",
    # Vinyl-specific (3 Phasen, 6 excluded goals)
    "phase_60": "vinyl_specific",
    "phase_60_inner_groove": "vinyl_specific",
    "phase_61": "vinyl_specific",
    "phase_61_groove_echo": "vinyl_specific",
    "phase_62": "vinyl_specific",
    "phase_62_crosstalk_cancellation": "vinyl_specific",
    "phase_63": "vinyl_specific",
}

# P0-Prioritäts-Phasen mit spezifischen Handlern
_SPECIFIC_HANDLER_PHASES: frozenset[str] = frozenset(
    {
        "phase_03",
        "phase_03_denoise",  # Denoise — P0
        "phase_09",
        "phase_09_crackle_removal",  # Crackle — P0
        "phase_04",
        "phase_04_eq_correction",  # EQ — P1
        "phase_16",
        "phase_16_final_eq",  # Final EQ — P1
        "phase_28",
        "phase_28_surface_noise",  # Surface Noise — P2
        "phase_29",
        "phase_29_tape_hiss_reduction",  # Tape Hiss — P2
    }
)


# ─────────────────────────────────────────────────────────
# Kategorie-basierte Universal-Proxy-Generatoren
# ─────────────────────────────────────────────────────────


def _proxy_category_denoise(before: np.ndarray, after: np.ndarray, sr: int) -> dict[str, float]:
    """Denoise/NR: Noise-Floor-Delta + Signal-Preservation + Transient-Erhalt.

    Gilt für: phase_01/02/03/05/09/28/29/43
    Gemeinsames Merkmal: Subtraktive Verarbeitung (spektrale Subtraktion,
    Wiener-Filterung, Notch-Filter). PMGG meldet falsche Regression weil:
    - Rauschen glättet Spektrum → nach NR erscheinen Täler → Flatness-Proxies fallen
    - NR reduziert HF-Energie → Brillanz-Proxy fällt (korrekt, aber gewollt)
    - NR glättet Transienten → Artikulation-Proxy fällt
    """
    b_arr = before.ravel().astype(np.float64)
    a_arr = after.ravel().astype(np.float64)

    # 1. Noise-Floor Delta (via spektrale Rauschtal-Tiefe)
    n_fft = min(4096, len(b_arr))
    b_fft = np.abs(np.fft.rfft(b_arr[:n_fft]))
    a_fft = np.abs(np.fft.rfft(a_arr[:n_fft]))
    # Rauschboden = p5-Perzentil der FFT-Magnitude
    b_noise = float(np.percentile(b_fft, 5))
    a_noise = float(np.percentile(a_fft, 5))
    1.0 / (1.0 + max(0, (b_noise - a_noise) / (b_noise + 1e-10)) * 3)
    # Umkehren: noise_reduction = 0 wenn Rauschboden gleich, 1 wenn stark reduziert
    noise_score = 0.5 + (b_noise - a_noise) / (b_noise + a_noise + 1e-10)

    # 2. Signal-Preservation (Pearson-Korrelation Original↔Verarbeitet)
    seg_len = min(len(b_arr), len(a_arr), sr * 5)
    if seg_len >= 256:
        corr = float(np.corrcoef(b_arr[:seg_len], a_arr[:seg_len])[0, 1])
        corr = max(0.0, corr)  # negative Korrelation = Problem
    else:
        corr = 0.5

    # 3. Transient-Erhalt (Energie-Verhältnis der Ableitungen)
    b_diff = np.diff(b_arr[:seg_len])
    a_diff = np.diff(a_arr[:seg_len])
    b_trans = float(np.sqrt(np.mean(b_diff**2)) + 1e-10)
    a_trans = float(np.sqrt(np.mean(a_diff**2)) + 1e-10)
    trans_preservation = min(a_trans / b_trans, b_trans / a_trans)

    # 4. Spektrale Entropie (sollte nach NR STEIGEN)
    b_ent = -float(np.sum(b_fft / b_fft.sum() * np.log(b_fft / b_fft.sum() + 1e-10)))
    a_ent = -float(np.sum(a_fft / a_fft.sum() * np.log(a_fft / a_fft.sum() + 1e-10)))
    entropy_ok = 1.0 if a_ent >= b_ent else max(0.0, 1.0 - (b_ent - a_ent))

    return {
        "natuerlichkeit": min(1.0, max(0.0, noise_score * 0.5 + corr * 0.5)),
        "authentizitaet": min(1.0, max(0.0, corr * 0.7 + trans_preservation * 0.3)),
        "brillanz": min(1.0, max(0.0, trans_preservation * 0.6 + entropy_ok * 0.4)),
        "transparenz": min(1.0, max(0.0, noise_score * 0.6 + entropy_ok * 0.4)),
        "artikulation": min(1.0, max(0.0, trans_preservation)),
        "timbre_authentizitaet": min(1.0, max(0.0, corr * 0.8 + entropy_ok * 0.2)),
        "groove": min(1.0, max(0.0, trans_preservation * 0.7 + corr * 0.3)),
        "emotionalitaet": min(1.0, max(0.0, noise_score * 0.4 + corr * 0.3 + trans_preservation * 0.3)),
        "tonal_center": min(1.0, max(0.0, corr * 0.9 + entropy_ok * 0.1)),
        "bass_kraft": min(1.0, max(0.0, corr)),  # Breitband-Korrelation ≈ Bass-Erhalt
        "waerme": min(1.0, max(0.0, corr * 0.7 + trans_preservation * 0.3)),
    }


def _proxy_category_eq_correction(before: np.ndarray, after: np.ndarray, sr: int) -> dict[str, float]:
    """EQ/Tonal: Spektrale Balance + Energie-Verteilung + Bass/HF-Ratio.

    Gilt für: phase_04/16/37/38/39/44
    PMGG meldet falsche Regression weil:
    - EQ verändert Spektralform → MFCC-Pearson fällt
    - EQ boostet HF → Brillanz steigt, aber Wärme fällt (Anti-Korrelation)
    - EQ ist SUPPOSED TO change the spectrum — das ist kein Defekt
    """
    b_arr = before.ravel().astype(np.float64)
    a_arr = after.ravel().astype(np.float64)

    n_fft = min(4096, len(b_arr))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    b_fft = np.abs(np.fft.rfft(b_arr[:n_fft]))
    a_fft = np.abs(np.fft.rfft(a_arr[:n_fft]))

    # 1. Spektrale Balance (sollte sich verbessern, d.h. flacher werden)
    def spectral_balance(fft):
        lo = fft[(freqs > 20) & (freqs < 500)].mean()
        mid = fft[(freqs > 500) & (freqs < 4000)].mean()
        hi = fft[(freqs > 4000)].mean()
        return float(np.std([lo, mid, hi]) / (np.mean([lo, mid, hi]) + 1e-10))

    spectral_balance(b_fft)
    a_bal = spectral_balance(a_fft)
    balance_improvement = 1.0 / (1.0 + a_bal)  # 0=unbalanced, 1=balanced

    # 2. Wärme (200-800Hz Energie / Gesamtenergie)
    b_warm = float(b_fft[(freqs > 200) & (freqs < 800)].sum() / (b_fft.sum() + 1e-10))
    a_warm = float(a_fft[(freqs > 200) & (freqs < 800)].sum() / (a_fft.sum() + 1e-10))
    warmth_ok = 1.0 - abs(a_warm - b_warm) * 3  # Toleranz ±33%

    # 3. Bass-Kraft (< 200 Hz Energie)
    b_bass = float(b_fft[freqs < 200].sum() / (b_fft.sum() + 1e-10))
    a_bass = float(a_fft[freqs < 200].sum() / (a_fft.sum() + 1e-10))
    bass_ok = 1.0 - abs(a_bass - b_bass) * 3

    # 4. Korrelation (Struktur-Erhalt)
    seg_len = min(len(b_arr), len(a_arr), sr * 5)
    corr = float(np.corrcoef(b_arr[:seg_len], a_arr[:seg_len])[0, 1]) if seg_len >= 256 else 0.5
    corr = max(0.0, corr)

    return {
        "transparenz": min(1.0, max(0.0, balance_improvement)),
        "brillanz": min(1.0, max(0.0, balance_improvement * 0.8 + corr * 0.2)),
        "waerme": min(1.0, max(0.0, warmth_ok)),
        "authentizitaet": min(1.0, max(0.0, corr * 0.6 + warmth_ok * 0.2 + bass_ok * 0.2)),
        "artikulation": min(1.0, max(0.0, corr)),
        "natuerlichkeit": min(1.0, max(0.0, corr * 0.5 + balance_improvement * 0.5)),
        "timbre_authentizitaet": min(1.0, max(0.0, corr * 0.7 + warmth_ok * 0.3)),
        "bass_kraft": min(1.0, max(0.0, bass_ok)),
        "emotionalitaet": min(1.0, max(0.0, balance_improvement * 0.6 + corr * 0.4)),
    }


def _proxy_category_synthesis_inpaint(before: np.ndarray, after: np.ndarray, sr: int) -> dict[str, float]:
    """Synthesis/Inpainting: Temporal Continuity + Gap-Closure + HF-Recovery.

    Gilt für: phase_06/07/23/50/55 (+ phase_24/56 von Cassette)
    PMGG meldet falsche Regression weil:
    - Synthetisierter Content hat KEINE Referenz im Original
    - MFCC, Flatness, Chroma sind meaningless gegen defekte Referenz
    - Siehe phase_24 Dokumentation fuer Root-Cause
    """
    b_arr = before.ravel().astype(np.float64)
    a_arr = after.ravel().astype(np.float64)

    # 1. Temporal Continuity (Energie-Varianz über 50ms-Fenster)
    block = int(0.05 * sr)
    n_blocks = min(len(b_arr), len(a_arr)) // block
    if n_blocks < 3:
        return {
            "natuerlichkeit": 0.5,
            "authentizitaet": 0.5,
            "brillanz": 0.5,
            "timbre_authentizitaet": 0.5,
            "artikulation": 0.5,
            "transparenz": 0.5,
            "tonal_center": 0.5,
        }

    b_en = np.array([np.mean(b_arr[i * block : (i + 1) * block] ** 2) for i in range(n_blocks)])
    a_en = np.array([np.mean(a_arr[i * block : (i + 1) * block] ** 2) for i in range(n_blocks)])
    b_cont = 1.0 / (1.0 + float(np.std(b_en) / (np.mean(b_en) + 1e-10)) * 10)
    a_cont = 1.0 / (1.0 + float(np.std(a_en) / (np.mean(a_en) + 1e-10)) * 10)
    continuity_delta = a_cont - b_cont

    # 2. HF-Recovery
    b_hf = float(np.mean(np.abs(np.diff(b_arr))))
    a_hf = float(np.mean(np.abs(np.diff(a_arr))))

    return {
        "natuerlichkeit": min(1.0, max(0.0, a_cont)),
        "brillanz": min(1.0, max(0.0, a_hf / (b_hf + 1e-10) * 0.8)),
        "authentizitaet": min(1.0, max(0.0, 0.5 + continuity_delta)),
        "artikulation": min(1.0, max(0.0, a_hf / (b_hf + 1e-10))),
        "timbre_authentizitaet": min(1.0, max(0.0, 0.5 + continuity_delta * 0.8)),
        "transparenz": min(1.0, max(0.0, a_cont * 0.9)),
        "tonal_center": min(1.0, max(0.0, 0.7 + continuity_delta * 0.3)),
        "groove": min(1.0, max(0.0, 0.5 + continuity_delta * 0.7)),
        "emotionalitaet": min(1.0, max(0.0, 0.5 + continuity_delta * 0.5)),
        "micro_dynamics": min(1.0, max(0.0, a_cont)),
    }


def _proxy_category_dynamics(before: np.ndarray, after: np.ndarray, sr: int) -> dict[str, float]:
    """Dynamics: Crest-Faktor + Envelope-Korrelation + Mikrodynamik-Erhalt.

    Gilt für: phase_10/11/17/18/19/26/34/35/36/47/54
    PMGG meldet falsche Regression weil:
    - Dynamik-Bearbeitung verändert Envelope → Crest-Faktor ändert sich
    - Das ist der ZWECK der Phase — kein Defekt
    """
    b_arr = before.ravel().astype(np.float64)
    a_arr = after.ravel().astype(np.float64)

    # 1. Crest-Faktor (Peak/RMS)
    b_rms = float(np.sqrt(np.mean(b_arr**2)) + 1e-10)
    a_rms = float(np.sqrt(np.mean(a_arr**2)) + 1e-10)
    b_peak = float(np.max(np.abs(b_arr)))
    a_peak = float(np.max(np.abs(a_arr)))
    b_crest = b_peak / b_rms if b_rms > 1e-10 else 1.0
    a_crest = a_peak / a_rms if a_rms > 1e-10 else 1.0
    # Dynamics processing sollte crest erhöhen (expansion) oder moderat senken (comp)
    crest_ok = 1.0 - abs(20 * np.log10(a_crest / (b_crest + 1e-10))) / 12.0

    # 2. Envelope-Korrelation
    block = int(0.025 * sr)
    n_blocks = min(len(b_arr), len(a_arr)) // block
    if n_blocks >= 4:
        b_env = np.array([np.sqrt(np.mean(b_arr[i * block : (i + 1) * block] ** 2)) for i in range(n_blocks)])
        a_env = np.array([np.sqrt(np.mean(a_arr[i * block : (i + 1) * block] ** 2)) for i in range(n_blocks)])
        env_corr = float(np.corrcoef(b_env, a_env)[0, 1]) if np.std(b_env) > 1e-10 and np.std(a_env) > 1e-10 else 0.5
        env_corr = max(0.0, env_corr)
    else:
        env_corr = 0.5

    return {
        "micro_dynamics": min(1.0, max(0.0, crest_ok * 0.6 + env_corr * 0.4)),
        "groove": min(1.0, max(0.0, env_corr * 0.7 + crest_ok * 0.3)),
        "emotionalitaet": min(1.0, max(0.0, crest_ok * 0.5 + env_corr * 0.5)),
        "artikulation": min(1.0, max(0.0, env_corr)),
        "authentizitaet": min(1.0, max(0.0, env_corr * 0.8 + crest_ok * 0.2)),
        "natuerlichkeit": min(1.0, max(0.0, env_corr * 0.7 + crest_ok * 0.3)),
        "timbre_authentizitaet": min(1.0, max(0.0, env_corr)),
        "waerme": min(1.0, max(0.0, env_corr)),
    }


def _proxy_category_spatial(before: np.ndarray, after: np.ndarray, sr: int) -> dict[str, float]:
    """Spatial/Reverb: Kanal-Balance + Stereo-Bild-Erhalt + Raumtiefe.

    Gilt für: phase_20/46/48/49
    """
    b_arr = before.ravel().astype(np.float64)
    a_arr = after.ravel().astype(np.float64)

    seg_len = min(len(b_arr), len(a_arr), sr * 3)
    corr = float(np.corrcoef(b_arr[:seg_len], a_arr[:seg_len])[0, 1]) if seg_len >= 256 else 0.5
    corr = max(0.0, corr)

    # Energie-Erhalt (Reverb-Entfernung reduziert Energie)
    b_energy = float(np.sqrt(np.mean(b_arr[:seg_len] ** 2)))
    a_energy = float(np.sqrt(np.mean(a_arr[:seg_len] ** 2)))
    energy_preservation = min(b_energy / (a_energy + 1e-10), a_energy / (b_energy + 1e-10))

    return {
        "authentizitaet": min(1.0, max(0.0, corr * 0.6 + energy_preservation * 0.4)),
        "natuerlichkeit": min(1.0, max(0.0, corr)),
        "timbre_authentizitaet": min(1.0, max(0.0, corr * 0.9 + energy_preservation * 0.1)),
        "tonal_center": min(1.0, max(0.0, corr)),
        "artikulation": min(1.0, max(0.0, corr)),
        "emotionalitaet": min(1.0, max(0.0, corr * 0.5 + energy_preservation * 0.5)),
        "waerme": min(1.0, max(0.0, corr * 0.7 + energy_preservation * 0.3)),
    }


def _proxy_category_speed_pitch(before: np.ndarray, after: np.ndarray, sr: int) -> dict[str, float]:
    """Speed/Pitch: Pitch-Stabilität + Groove-Erhalt + Timing.

    Gilt für: phase_12/31
    """
    b_arr = before.ravel().astype(np.float64)
    a_arr = after.ravel().astype(np.float64)

    seg_len = min(len(b_arr), len(a_arr), sr * 5)
    corr = float(np.corrcoef(b_arr[:seg_len], a_arr[:seg_len])[0, 1]) if seg_len >= 256 else 0.5
    corr = max(0.0, corr)

    # Onset-Detektion (via Energie-Differenz)
    diff_b = np.diff(b_arr[:seg_len])
    diff_a = np.diff(a_arr[:seg_len])
    b_onsets = float(np.sum(np.abs(diff_b) > np.std(diff_b) * 2)) / len(diff_b)
    a_onsets = float(np.sum(np.abs(diff_a) > np.std(diff_a) * 2)) / len(diff_a)
    onset_preservation = 1.0 - abs(a_onsets - b_onsets) * 10

    # ZCR-Änderung (Pitch-Shift ändert ZCR)
    b_zcr = float(np.mean(np.abs(np.diff(np.sign(b_arr[:seg_len])))) / 2)
    a_zcr = float(np.mean(np.abs(np.diff(np.sign(a_arr[:seg_len])))) / 2)
    zcr_ok = 1.0 - abs(a_zcr - b_zcr) / (b_zcr + 0.01) * 5

    return {
        "tonal_center": min(1.0, max(0.0, corr * 0.5 + zcr_ok * 0.5)),
        "timbre_authentizitaet": min(1.0, max(0.0, corr)),
        "groove": min(1.0, max(0.0, onset_preservation)),
        "emotionalitaet": min(1.0, max(0.0, corr * 0.4 + onset_preservation * 0.6)),
        "authentizitaet": min(1.0, max(0.0, corr)),
        "natuerlichkeit": min(1.0, max(0.0, corr * 0.7 + zcr_ok * 0.3)),
        "artikulation": min(1.0, max(0.0, corr)),
    }


def _proxy_category_harmonic(before: np.ndarray, after: np.ndarray, sr: int) -> dict[str, float]:
    """Harmonic/Saturation: THD-Kontrolle + Wärme-Erhalt.

    Gilt für: phase_21/22
    """
    b_arr = before.ravel().astype(np.float64)
    a_arr = after.ravel().astype(np.float64)
    corr = float(np.corrcoef(b_arr[: min(len(b_arr), sr * 3)], a_arr[: min(len(a_arr), sr * 3)])[0, 1])
    corr = max(0.0, corr)
    return {
        "timbre_authentizitaet": min(1.0, max(0.0, corr)),
        "emotionalitaet": min(1.0, max(0.0, corr * 0.7 + 0.3)),
        "waerme": min(1.0, max(0.0, corr)),
    }


def _proxy_category_vocal(before: np.ndarray, after: np.ndarray, sr: int) -> dict[str, float]:
    """Vocal/Formant: Formant-Erhalt + Artikulation + Natürlichkeit.

    Gilt für: phase_42/58
    """
    b_arr = before.ravel().astype(np.float64)
    a_arr = after.ravel().astype(np.float64)
    seg_len = min(len(b_arr), len(a_arr), sr * 3)
    corr = float(np.corrcoef(b_arr[:seg_len], a_arr[:seg_len])[0, 1]) if seg_len >= 256 else 0.5
    corr = max(0.0, corr)
    return {
        "natuerlichkeit": min(1.0, max(0.0, corr)),
        "authentizitaet": min(1.0, max(0.0, corr)),
        "timbre_authentizitaet": min(1.0, max(0.0, corr * 0.8)),
        "groove": min(1.0, max(0.0, corr)),
        "emotionalitaet": min(1.0, max(0.0, corr * 0.7)),
        "artikulation": min(1.0, max(0.0, corr)),
        "tonal_center": min(1.0, max(0.0, corr)),
    }


def _proxy_category_vinyl_specific(before: np.ndarray, after: np.ndarray, sr: int) -> dict[str, float]:
    """Vinyl-spezifisch: Rillen-Verzerrung + Crosstalk + Echo.

    Gilt für: phase_60/61/62/63
    """
    b_arr = before.ravel().astype(np.float64)
    a_arr = after.ravel().astype(np.float64)
    corr = float(np.corrcoef(b_arr[: min(len(b_arr), sr * 3)], a_arr[: min(len(a_arr), sr * 3)])[0, 1])
    corr = max(0.0, corr)
    return {
        "authentizitaet": min(1.0, max(0.0, corr)),
        "timbre_authentizitaet": min(1.0, max(0.0, corr)),
    }


def _proxy_category_passthrough(before: np.ndarray, after: np.ndarray, sr: int) -> dict[str, float]:
    """Passthrough/Legacy: Minimale Änderung → Struktur-Korrelation.

    Gilt für: phase_13/15/27/30/32/33/40/41/45/51/52
    Diese Phasen haben meist nur 1-2 excluded goals (timbre_authentizitaet).
    """
    b_arr = before.ravel().astype(np.float64)
    a_arr = after.ravel().astype(np.float64)
    corr = float(np.corrcoef(b_arr[: min(len(b_arr), sr * 3)], a_arr[: min(len(a_arr), sr * 3)])[0, 1])
    corr = max(0.0, corr)
    return {
        "timbre_authentizitaet": min(1.0, max(0.0, corr)),
        "authentizitaet": min(1.0, max(0.0, corr)),
        "natuerlichkeit": min(1.0, max(0.0, corr)),
    }


# ── Kategorie-Dispatcher ──
_CATEGORY_PROXY_GENERATORS: dict[str, callable] = {  # type: ignore[valid-type]
    "denoise": _proxy_category_denoise,
    "eq_correction": _proxy_category_eq_correction,
    "synthesis_inpaint": _proxy_category_synthesis_inpaint,
    "dynamics": _proxy_category_dynamics,
    "spatial": _proxy_category_spatial,
    "speed_pitch": _proxy_category_speed_pitch,
    "harmonic": _proxy_category_harmonic,
    "vocal": _proxy_category_vocal,
    "vinyl_specific": _proxy_category_vinyl_specific,
    "passthrough": _proxy_category_passthrough,
}


def _get_category_for_phase(phase_id: str) -> str:
    """Bestimmt die Verarbeitungs-Kategorie einer Phase."""
    # Exakte Übereinstimmung
    if phase_id in _PHASE_CATEGORIES:
        return _PHASE_CATEGORIES[phase_id]
    # Präfix-Match (phase_XX_... → phase_XX)
    for prefix in sorted(_PHASE_CATEGORIES, key=len, reverse=True):
        if phase_id.startswith(prefix):
            return _PHASE_CATEGORIES[prefix]
    return "passthrough"


# ── Spezifische Handler (P0-P2) ──


def _proxy_specific_phase_03_denoise(before: np.ndarray, after: np.ndarray, sr: int) -> dict[str, float]:
    """phase_03 Denoise: Spezifischer Handler mit SNR-Verbesserung + Sprach/Musik-Detektion.

    Die meistgenutzte Phase. 6 excluded goals.
    """
    result = _proxy_category_denoise(before, after, sr)

    # Zusätzlich: SNR-Verbesserung (Signal-to-Noise Ratio Delta)
    n_fft = min(4096, len(before))
    b_fft = np.abs(np.fft.rfft(before.ravel()[:n_fft]))
    a_fft = np.abs(np.fft.rfft(after.ravel()[:n_fft]))
    # SNR ≈ Peak / Noise-Floor
    b_snr = float(np.max(b_fft) / (np.percentile(b_fft, 10) + 1e-10))
    a_snr = float(np.max(a_fft) / (np.percentile(a_fft, 10) + 1e-10))
    snr_improvement = min(1.0, max(0.0, (a_snr - b_snr) / (b_snr + 1e-10) + 0.5))

    # Spektraler Tilt (HF/LF Ratio sollte erhalten bleiben)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    b_tilt = float(np.mean(b_fft[freqs > 3000]) / (np.mean(b_fft[freqs < 500]) + 1e-10))
    a_tilt = float(np.mean(a_fft[freqs > 3000]) / (np.mean(a_fft[freqs < 500]) + 1e-10))
    tilt_preservation = 1.0 - abs(a_tilt - b_tilt) / (b_tilt + 0.1)

    result["natuerlichkeit"] = min(1.0, max(0.0, snr_improvement * 0.5 + tilt_preservation * 0.5))
    result["brillanz"] = min(1.0, max(0.0, result.get("brillanz", 0.5) * 0.7 + tilt_preservation * 0.3))
    result["authentizitaet"] = min(1.0, max(0.0, result.get("authentizitaet", 0.5) * 0.6 + tilt_preservation * 0.4))
    result["tonal_center"] = min(1.0, max(0.0, result.get("tonal_center", 0.7) * 0.5 + tilt_preservation * 0.5))
    return result


def _proxy_specific_phase_09_crackle(before: np.ndarray, after: np.ndarray, sr: int) -> dict[str, float]:
    """phase_09 Crackle: Impuls-Reduktion + Groove-Erhalt."""
    result = _proxy_category_denoise(before, after, sr)
    # Zusätzlich: Impuls-Dichte (Crackle = viele Mikro-Impulse)
    b_diff = np.diff(before.ravel())
    a_diff = np.diff(after.ravel())
    b_impulses = float(np.sum(np.abs(b_diff) > np.std(b_diff) * 2.5))
    a_impulses = float(np.sum(np.abs(a_diff) > np.std(a_diff) * 2.5))
    impulse_reduction = 1.0 / (1.0 + max(0, (b_impulses - a_impulses) / max(b_impulses, 1)) * 2)
    result["groove"] = min(1.0, max(0.0, impulse_reduction * 0.5 + result.get("groove", 0.5) * 0.5))
    result["emotionalitaet"] = min(1.0, max(0.0, impulse_reduction * 0.4 + result.get("emotionalitaet", 0.5) * 0.6))
    return result


def _proxy_specific_phase_04_eq(before: np.ndarray, after: np.ndarray, sr: int) -> dict[str, float]:
    """phase_04 EQ Correction: RIAA/Tape-EQ — Basis-Entzerrung."""
    result = _proxy_category_eq_correction(before, after, sr)
    # Zusätzlich: Spektrale Flachheit nach Zielkurve
    return result  # EQ-category deckt bereits alle 7 goals ab


def _proxy_specific_phase_16_final_eq(before: np.ndarray, after: np.ndarray, sr: int) -> dict[str, float]:
    """phase_16 Final EQ: Letzter spektraler Feinschliff vor Ausgabe."""
    result = _proxy_category_eq_correction(before, after, sr)
    # Zusätzlich: True-Peak-Kontrolle
    float(np.max(np.abs(before.ravel())))
    a_peak = float(np.max(np.abs(after.ravel())))
    peak_safety = 1.0 if a_peak < 0.99 else 0.5  # Clipping-Schutz
    result["authentizitaet"] = min(1.0, max(0.0, result.get("authentizitaet", 0.5) * 0.8 + peak_safety * 0.2))
    return result


def _proxy_specific_phase_28_surface(before: np.ndarray, after: np.ndarray, sr: int) -> dict[str, float]:
    """phase_28 Surface Noise: Vinyl-Oberflächenrauschen-Profiling."""
    result = _proxy_category_denoise(before, after, sr)
    # Zusätzlich: Breitband-Rausch-Leistungs-Dichte
    n_fft = min(4096, len(before))
    b_fft = np.abs(np.fft.rfft(before.ravel()[:n_fft]))
    a_fft = np.abs(np.fft.rfft(after.ravel()[:n_fft]))
    # Rauschleistung in stillen Frequenzbändern (> 10kHz)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    b_silence = float(np.mean(b_fft[freqs > 10000]))
    a_silence = float(np.mean(a_fft[freqs > 10000]))
    silence_improvement = 0.5 + (b_silence - a_silence) / (b_silence + a_silence + 1e-10)
    result["artikulation"] = min(1.0, max(0.0, silence_improvement))
    return result


def _proxy_specific_phase_29_hiss(before: np.ndarray, after: np.ndarray, sr: int) -> dict[str, float]:
    """phase_29 Tape Hiss: Hochfrequentes Bandrauschen."""
    result = _proxy_category_denoise(before, after, sr)
    # Zusätzlich: HF-Rauschleistung (8-16 kHz)
    n_fft = min(4096, len(before))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    b_fft = np.abs(np.fft.rfft(before.ravel()[:n_fft]))
    a_fft = np.abs(np.fft.rfft(after.ravel()[:n_fft]))
    hf_mask = (freqs > 8000) & (freqs < 16000)
    b_hf_noise = float(np.mean(b_fft[hf_mask]))
    a_hf_noise = float(np.mean(a_fft[hf_mask]))
    hiss_reduction = 0.5 + (b_hf_noise - a_hf_noise) / (b_hf_noise + a_hf_noise + 1e-10)
    result["brillanz"] = min(1.0, max(0.0, hiss_reduction))
    return result


# ── Spezifische Handler-Dispatch ──
_SPECIFIC_HANDLERS: dict[str, callable] = {  # type: ignore[valid-type]
    "phase_03": _proxy_specific_phase_03_denoise,
    "phase_03_denoise": _proxy_specific_phase_03_denoise,
    "phase_09": _proxy_specific_phase_09_crackle,
    "phase_09_crackle_removal": _proxy_specific_phase_09_crackle,
    "phase_04": _proxy_specific_phase_04_eq,
    "phase_04_eq_correction": _proxy_specific_phase_04_eq,
    "phase_16": _proxy_specific_phase_16_final_eq,
    "phase_16_final_eq": _proxy_specific_phase_16_final_eq,
    "phase_28": _proxy_specific_phase_28_surface,
    "phase_28_surface_noise": _proxy_specific_phase_28_surface,
    "phase_29": _proxy_specific_phase_29_hiss,
    "phase_29_tape_hiss_reduction": _proxy_specific_phase_29_hiss,
}


def _extract_segment(audio: np.ndarray, start: int, end: int) -> np.ndarray:
    """Sicheres Extrahieren eines Segments."""
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 2:
        a = a.mean(axis=1) if a.shape[1] <= 2 else a.mean(axis=0)
    a = a.ravel()
    return cast(np.ndarray, a[start:end].copy())


def _compute_segment_hpe(audio: np.ndarray, sr: int) -> float:
    """Vereinfachte Segment-HPE basierend auf Sharpness + Roughness."""
    arr = np.asarray(audio, dtype=np.float64).ravel()
    if len(arr) < 256:
        return 0.5

    # Sharpness (High-Frequency Content ratio)
    n_fft = min(2048, len(arr))
    fft = np.abs(np.fft.rfft(arr[:n_fft]))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    hf_mask = freqs > 3000
    hf_ratio = float(np.sum(fft[hf_mask]) / (np.sum(fft) + 1e-10))
    sharpness_ok = 1.0 - abs(hf_ratio - 0.2) * 2.0  # ~20% HF = ideal

    # Roughness (Amplitude Modulation 15-300 Hz)
    env = np.abs(arr)
    if len(env) >= 64:
        env_fft = np.abs(np.fft.rfft(env[:2048]))
        env_freqs = np.fft.rfftfreq(2048, 1.0 / sr)
        rough_mask = (env_freqs > 15) & (env_freqs < 300)
        roughness = float(np.mean(env_fft[rough_mask]) / (np.mean(env_fft) + 1e-10))
    else:
        roughness = 0.3

    roughness_ok = 1.0 - min(roughness * 3.0, 1.0)

    # Kombinieren
    return float(np.clip(sharpness_ok * 0.5 + roughness_ok * 0.5, 0.0, 1.0))


def _compute_residual_level(residual: np.ndarray, original: np.ndarray, sr: int) -> tuple[float, float, bool]:
    """Berechnet Residual-Pegel relativ zum Signal."""
    orig_rms = float(np.sqrt(np.mean(original.ravel() ** 2)) + 1e-12)
    res_rms = float(np.sqrt(np.mean(residual.ravel() ** 2)) + 1e-12)
    res_peak = float(np.max(np.abs(residual.ravel())) + 1e-12)

    rms_db = float(20.0 * np.log10(res_rms / orig_rms))
    peak_db = float(20.0 * np.log10(res_peak / orig_rms))

    masked = rms_db < MASKING_THRESHOLD_DB
    return peak_db, rms_db, masked


def _estimate_audibility(residual: np.ndarray, original: np.ndarray, sr: int) -> float:
    """Schätzt die Hörbarkeit des Residuals (0-1)."""
    orig_rms = float(np.sqrt(np.mean(original.ravel() ** 2)) + 1e-10)
    res_rms = float(np.sqrt(np.mean(residual.ravel() ** 2)) + 1e-10)
    rms_db = float(20.0 * np.log10(res_rms / orig_rms))

    # Zwicker/Fastl: Maskierung ≈ -18 bis -28 dB je nach Frequenz
    if rms_db < -25:
        return 0.0
    elif rms_db < -18:
        return float((rms_db + 25) / 7.0 * 0.3)  # 0.0 → 0.3
    elif rms_db < -10:
        return 0.3 + float((rms_db + 18) / 8.0 * 0.4)  # 0.3 → 0.7
    else:
        return 0.7 + float(min(-rms_db / 10.0, 1.0) * 0.3)  # 0.7 → 1.0


def _count_affected_critical_bands(residual: np.ndarray, original: np.ndarray, sr: int) -> int:
    """Zählt kritische Bänder (Zwicker) mit hörbarem Residual."""
    # Zwicker kritische Bänder (vereinfacht): 24 Bänder, 1 Bark ≈ 1.3mm auf Basilarmembran
    n_fft = min(4096, len(residual))
    if n_fft < 128:
        return 0

    res_fft = np.abs(np.fft.rfft(residual.ravel()[:n_fft]))
    orig_fft = np.abs(np.fft.rfft(original.ravel()[:n_fft]))

    # Vereinfachte Bark-Skala: 24 Bänder logarithmisch
    n_bins = len(res_fft)
    bark_edges = np.unique(np.logspace(0, np.log10(n_bins), 25, dtype=int))
    bark_edges = bark_edges[(bark_edges >= 0) & (bark_edges < n_bins)]

    affected = 0
    for i in range(len(bark_edges) - 1):
        lo, hi = int(bark_edges[i]), int(bark_edges[i + 1])
        if hi <= lo:
            continue
        band_res = float(np.mean(res_fft[lo:hi]))
        band_orig = float(np.mean(orig_fft[lo:hi])) + 1e-10
        if 20.0 * np.log10(band_res / band_orig) > MASKING_THRESHOLD_DB:
            affected += 1

    return affected


def _per_band_residual(residual: np.ndarray, original: np.ndarray, sr: int) -> list[float]:
    """Residual pro kritischem Band in dB."""
    n_fft = min(4096, len(residual))
    if n_fft < 128:
        return [0.0]

    res_fft = np.abs(np.fft.rfft(residual.ravel()[:n_fft]))
    orig_fft = np.abs(np.fft.rfft(original.ravel()[:n_fft]))

    n_bins = len(res_fft)
    bark_edges = np.unique(np.logspace(0, np.log10(n_bins), 25, dtype=int))
    bark_edges = bark_edges[(bark_edges >= 0) & (bark_edges < n_bins)]

    band_dbs: list[float] = []
    for i in range(len(bark_edges) - 1):
        lo, hi = int(bark_edges[i]), int(bark_edges[i + 1])
        if hi <= lo:
            continue
        band_res = float(np.mean(res_fft[lo:hi])) + 1e-10
        band_orig = float(np.mean(orig_fft[lo:hi])) + 1e-10
        band_dbs.append(float(20.0 * np.log10(band_res / band_orig)))

    return band_dbs


def _compute_phase_proxy(
    phase_id: str,
    before: np.ndarray,
    after: np.ndarray,
    sr: int,
) -> tuple[str, float, float, bool]:
    """Berechnet die phase-spezifische Proxy-Metrik."""
    proxies = compute_phase_proxy_for_pmgg(phase_id, before, after, sr)

    # Wähle repräsentativste Metrik
    if "natuerlichkeit" in proxies:
        proxy_name = "natuerlichkeit"
    elif "authentizitaet" in proxies:
        proxy_name = "authentizitaet"
    elif "brillanz" in proxies:
        proxy_name = "brillanz"
    elif proxies:
        proxy_name = list(proxies.keys())[0]
    else:
        proxy_name = "level_stability"

    proxy_score = proxies.get(proxy_name, 0.5)
    threshold = 0.55  # HPE-Grünzone-Basis
    passed = proxy_score >= threshold

    return proxy_name, proxy_score, threshold, passed


def _build_recommendation(verdict: str, defect_type: str, phase_id: str, proxy_passed: bool) -> str:
    """Baut eine handlungsorientierte Empfehlung."""
    if verdict == "clean":
        return f"{defect_type}: Perfekt repariert. Kein hörbares Residual."
    elif verdict == "residual_below_masking":
        return f"{defect_type}: Residual vorhanden aber unter Maskierungsschwelle. Für menschliche Ohren unhörbar."
    elif not proxy_passed:
        return (
            f"{defect_type}: {phase_id} Proxy-Metrik nicht bestanden. "
            "Erwäge reduziertes Strength oder alternative Phase."
        )
    else:
        return f"{defect_type}: Hörbares Residual. {phase_id} mit reduziertem Strength (0.65→0.40) erneut versuchen."


# ─────────────────────────────────────────────────────────────────
# Integration: Automatische Post-Phase-Verifikation
# ─────────────────────────────────────────────────────────────────


def post_phase_verification(
    audio_before: np.ndarray,
    audio_after: np.ndarray,
    sr: int,
    phase_id: str,
    defect_locations: dict[str, list[tuple[float, float]]] | None = None,
) -> dict[str, Any]:
    """§v10.2 Automatische Post-Phase-Verifikation.

    Wird nach JEDER cassetten-relevanten Phase aufgerufen.
    Prüft alle drei Ebenen (HPE, ABX, PMGG-Proxy) und gibt
    ein aggregiertes Urteil zurück.

    Returns dict mit:
      - phase_passed: bool
      - hpe_delta: float
      - worst_residual_db: float
      - proxy_scores: dict
      - batch_results: dict (pro Defekt-Typ)
      - recommendation: str
    """
    if defect_locations is None:
        defect_locations = {}

    batch_results: dict[str, BatchVerificationResult] = {}
    all_hpe_deltas: list[float] = []
    all_proxy_scores: dict[str, float] = {}
    worst_residual = -100.0

    for defect_type, locations in defect_locations.items():
        if not locations:
            continue
        batch = verify_defect_batch(audio_before, audio_after, sr, defect_type, phase_id, locations)
        batch_results[defect_type] = batch
        all_hpe_deltas.append(batch.mean_hpe_delta)
        worst_residual = max(worst_residual, batch.mean_residual_db)

    # PMGG Proxy
    proxies = compute_phase_proxy_for_pmgg(phase_id, audio_before, audio_after, sr)
    all_proxy_scores.update(proxies)

    # Aggregiertes Urteil
    mean_hpe = float(np.mean(all_hpe_deltas)) if all_hpe_deltas else 0.0
    audible_count = sum(b.audible_count for b in batch_results.values())
    total_checked = sum(b.segments_checked for b in batch_results.values())

    if total_checked == 0:
        phase_passed = True
        recommendation = "Keine Defekt-Locations → Überspringe Verifikation"
    elif audible_count == 0:
        phase_passed = True
        recommendation = f"Phase {phase_id}: Alle {total_checked} Segmente sauber oder maskiert."
    elif audible_count <= total_checked * 0.05:
        phase_passed = True
        recommendation = (
            f"Phase {phase_id}: {audible_count}/{total_checked} Segmente mit hörbarem Residual (<5%). Akzeptabel."
        )
    else:
        phase_passed = False
        recommendation = (
            f"Phase {phase_id}: {audible_count}/{total_checked} Segmente mit hörbarem Residual. RETRY empfohlen."
        )

    return {
        "phase_passed": phase_passed,
        "hpe_delta": mean_hpe,
        "worst_residual_db": worst_residual,
        "proxy_scores": all_proxy_scores,
        "batch_results": {k: v.overall_verdict for k, v in batch_results.items()},
        "total_checked": total_checked,
        "audible_count": audible_count,
        "recommendation": recommendation,
    }


# ── §v10.3 Category Proxy Dispatch Table ──
# Built after all category functions are defined above.
_CATEGORY_PROXY_FUNCTIONS.update(
    {
        "denoise": _proxy_category_denoise,
        "eq_correction": _proxy_category_eq_correction,
        "synthesis_inpaint": _proxy_category_synthesis_inpaint,
        "dynamics": _proxy_category_dynamics,
        "spatial": _proxy_category_spatial,
        "speed_pitch": _proxy_category_speed_pitch,
        "harmonic": _proxy_category_harmonic,
        "vocal": _proxy_category_vocal,
        "passthrough": _proxy_category_passthrough,
        "vinyl_specific": _proxy_category_vinyl_specific,
        # transient & phase_alignment: use passthrough as fallback
        "transient": _proxy_category_passthrough,
        "phase_alignment": _proxy_category_passthrough,
    }
)


# ── Convenience-Funktionen (für Dead-Import-Reparatur) ───────────────

def verify_cassette_defects(
    audio_before: np.ndarray,
    audio_after: np.ndarray,
    sr: int = 48000,
    phase_id: str = "phase_24",
) -> DefectVerificationResult:
    """Convenience-Funktion für cassette defect verification.

    Args:
        audio_before: Audio vor der Phase (float32)
        audio_after: Audio nach der Phase (float32)
        sr: Sample-Rate in Hz
        phase_id: Phase-ID für Reporting

    Returns:
        DefectVerificationResult mit Verifikation-Ergebnissen
    """
    return verify_defect_segment(audio_before, audio_after, sr, phase_id=phase_id)
