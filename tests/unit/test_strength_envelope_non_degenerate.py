"""Strength-Envelope-Nichtdegenerations-Regressionstest (Produktionsbefund 2026-09-07).

Befund: Der B3-Phase-2 Early-Merge überschrieb ENUM-Scores (mit Locations)
durch 0.06-Stubs → Envelope degenerierte zu μ=0.060 σ=0.000 → alle Phasen
liefen mit Floor-Stärke (No-Op-Kaskade). Dieser Test sichert die Kette
Merge → Locations-Extraktion → compute_strength_envelope gegen σ=0 ab.
"""

from __future__ import annotations

import numpy as np

from backend.core.defect_scanner import (
    DefectAnalysisResult,
    DefectScore,
    DefectType,
    MaterialType,
)
from backend.core.strength_envelope import compute_strength_envelope
from backend.core.unified_restorer_v3 import _b3_merge_full_song_defect_types


def _extract_locations_like_execute_pipeline(result: DefectAnalysisResult) -> tuple[dict, dict]:
    """Extraktion exakt wie in _execute_pipeline (§2.71)."""
    _STATIONARY = {
        "hum", "hiss", "noise_level", "low_freq_rumble", "high_freq_noise", "bias_error",
        "dolby_nr_mismatch", "riaa_curve_error", "generation_loss", "modulation_noise",
        "speed_calibration_error", "hf_remanence_loss",
    }
    _dur = float(getattr(result, "duration_seconds", 224.3))
    dloc: dict[str, list] = {}
    dsev: dict[str, float] = {}
    for _dt, _ds in result.scores.items():
        _key = _dt.value if hasattr(_dt, "value") else str(_dt)
        _has_sev = float(getattr(_ds, "severity", 0.0)) > 0.01
        if getattr(_ds, "locations", None):
            dloc[_key] = list(_ds.locations)
        elif _has_sev and _key in _STATIONARY:
            dloc[_key] = [(0.0, _dur)]
        dsev[_key] = float(getattr(_ds, "severity", 0.0))
    return dloc, dsev


def test_b3_merge_preserves_locations_and_envelope_stays_strong() -> None:
    """Die komplette Produktionskette darf nie wieder σ=0.000 liefern."""
    rng = np.random.default_rng(42)
    _locs = lambda n: [(float(a), float(a + 0.03)) for a in np.sort(rng.uniform(0, 224, n))]

    result = DefectAnalysisResult(
        material_type=MaterialType.VINYL,
        scores={
            DefectType.CLICKS: DefectScore(DefectType.CLICKS, 0.708, 0.99, _locs(500)),
            DefectType.CRACKLE: DefectScore(DefectType.CRACKLE, 0.839, 0.90, _locs(80)),
            DefectType.TRANSPORT_BUMP: DefectScore(DefectType.TRANSPORT_BUMP, 0.638, 0.99, _locs(40)),
            DefectType.WOW: DefectScore(DefectType.WOW, 1.0, 0.63, _locs(10)),
            DefectType.HUM: DefectScore(DefectType.HUM, 0.497, 0.45, _locs(3)),
        },
        analysis_time_seconds=107.0,
        sample_rate=48_000,
        duration_seconds=224.3,
    )

    # 1) B3-Phase-2 Early-Merge mit String-Set (der Produktions-Trigger)
    full_dt = {
        "bandwidth_loss", "bias_error", "clicks", "compression_artifacts", "crackle",
        "crosstalk", "digital_artifacts", "dropout_head_contact", "dropout_oxide",
        "dropouts", "flutter", "flutter_spectral_sidebands", "generation_loss",
        "groove_echo", "hf_remanence_loss", "high_freq_noise", "hum",
        "inner_groove_distortion", "jitter_artifacts", "low_freq_rumble",
        "modulation_noise", "motor_interference", "mpeg_frame_loss",
        "multiband_wow_flutter", "nr_breathing_artifact", "overload_distortion",
        "phase_rotation", "pitch_drift", "proximity_effect_excess", "reverb_excess",
        "room_mode_resonance", "sibilance", "soft_saturation", "stereo_field_collapse",
        "stereo_imbalance", "sticky_shed_residue", "stylus_damage", "tape_head_clog",
        "tape_head_level_dip", "transient_smearing", "transport_bump",
        "vocal_harshness", "wow",
    }
    _b3_merge_full_song_defect_types(result, full_dt)
    assert len(result.scores[DefectType.CLICKS].locations) == 500

    # 2) Extraktion wie _execute_pipeline
    dloc, dsev = _extract_locations_like_execute_pipeline(result)
    n_locs = sum(len(v) for v in dloc.values())
    assert n_locs > 600, f"Locations verloren: {n_locs}"

    # 3) Envelope auf Chunk 0 (30 s)
    chunk = np.zeros((2, 30 * 48_000), dtype=np.float32)
    env = compute_strength_envelope(
        defect_locations=dloc,
        defect_severity_map=dsev,
        defect_saliency_map={},
        audio_duration_s=30.0,
        sample_rate=48_000,
        audio=chunk[0],
    )
    mu, sd = float(np.mean(env)), float(np.std(env))
    assert sd > 0.001, f"Envelope degeneriert: μ={mu:.3f} σ={sd:.3f}"
    assert mu > 0.2, f"Envelope am Floor: μ={mu:.3f} (Produktionsbefund war 0.060)"


def test_envelope_without_locations_still_has_floor_semantics() -> None:
    """Kontrolltest: Ohne Locations darf der Envelope nur den dokumentierten Floor tragen."""
    dloc: dict[str, list] = {}
    dsev = {"hum": 0.4}
    chunk = np.zeros(30 * 48_000, dtype=np.float32)
    env = compute_strength_envelope(
        defect_locations=dloc,
        defect_severity_map=dsev,
        defect_saliency_map={},
        audio_duration_s=30.0,
        sample_rate=48_000,
        audio=chunk,
    )
    assert np.std(env) < 1e-6, "Ohne Locations/Stationary-Einträge muss der Envelope flach sein"
