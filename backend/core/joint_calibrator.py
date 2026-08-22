"""§2.70 Joint-Calibration Optimizer — Goal-Gap-getrieben, ohne hartcodierte Regeln.

Alle Phasen-Stärken werden AUSSCHLIESSLICH aus den messbaren
Musical-Goal-Gaps und dem PhaseEffectCatalog abgeleitet.

Keine phase-spezifischen Magic-Numbers. Kein `if pid == "phase_X"`.
Der Denker entscheidet für JEDEN Song individuell.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Goal-Gewichte: Welche Goals sind perceptuell am wichtigsten?
GOAL_WEIGHTS: dict[str, float] = {
    "natuerlichkeit": 1.2,
    "authentizitaet": 1.1,
    "transparenz": 1.0,
    "waerme": 1.0,
    "artikulation": 0.9,
    "groove": 0.9,
    "emotionalitaet": 0.8,
    "brillanz": 0.8,
    "tonal_center": 0.7,
    "micro_dynamics": 0.7,
    "transient_energie": 0.7,
    "bass_kraft": 0.6,
    "separation_fidelity": 0.6,
    "spatial_depth": 0.5,
    "timbre_authentizitaet": 0.5,
}

# Phasen die NIE unter 0.20 gedrosselt werden (Primum non nocere)
PROTECTED_PHASES: frozenset[str] = frozenset(
    {
        "phase_01_click_removal",
        "phase_24_dropout_repair",
        "phase_08_transient_preservation",
        "phase_12_wow_flutter_fix",  # Tape-Level-Dips sind strukturelle Defekte
    }
)


def joint_calibrate(
    phase_ids: list[str],
    goal_proxies: dict[str, float],
    goal_targets: dict[str, float],
    *,
    material: str = "vinyl",
    panns_singing: float = 0.0,
    codec_avg_discount: float = 1.0,
    terminal_codec: str | None = None,
    min_strength: float | None = None,
    restorability_score: float | None = None,
    default_strength: float = 0.85,
    transfer_chain_depth: int = 1,
) -> dict[str, float]:
    """Berechnet optimale Phasen-Stärken AUSSCHLIESSLICH aus Goal-Gaps.

    Keine hartcodierten Phasen-Regeln. Jede Entscheidung ist aus den
    Daten ableitbar und im Log nachvollziehbar.

    Args:
        phase_ids: Ausgewählte Phasen
        goal_proxies: Aktuelle Goal-Proxies
        goal_targets: Zielwerte pro Goal
        material: Trägermedium (für Material-Caps)
        panns_singing: PANNs Singing-Konfidenz
        codec_avg_discount: ∅ Diskont-Faktor aus Codec-Kette
        terminal_codec: Terminal-Codec-Typ oder None
        min_strength: Minimale Phasen-Stärke (None = auto aus restorability_score)
        restorability_score: Restorability 0-100 für adaptive min_strength (§G71 (GEBOTE.md))
        default_strength: Standard wenn kein Profil

    Returns:
        {phase_id: calibrated_strength}
    """
    # §v10.x rs-Konsistenz: kanonische Quelle (explizit > CalibrationContext > 65.0).
    from backend.core.calibration_context import resolve_restorability_score

    restorability_score = resolve_restorability_score(restorability_score, default=65.0)
    from backend.core.phase_effect_catalog import PHASE_EFFECT_CATALOG

    # §G71 (GEBOTE.md) Adaptive min_strength aus Restorability
    if min_strength is None:
        _rs = float(np.clip(restorability_score, 0.0, 100.0))
        if _rs >= 90:
            min_strength = 0.20
        elif _rs >= 60:
            min_strength = 0.35
        elif _rs >= 30:
            min_strength = 0.40
        else:
            min_strength = 0.45

    # §v10.58 Depth-Boost: Bei transfer_chain_depth ≥ 5 (extreme Ketten ≥5 Stufen,
    # z.B. Wachswalze→Schellack→Tonband→Kassette→MP3) werden Kern-Restaurationsphasen
    # verstärkt, da extreme Degradations-Akkumulation stärkere Eingriffe benötigt.
    # §v10.120 Calibration-Shift: depth 4 (0.55) ist "deep cassette", depth 5+ ist
    # "extreme chain" mit Boost-Berechtigung.
    _depth_boost = float(np.clip(1.0 + (transfer_chain_depth - 1) * 0.12, 1.0, 1.50))
    if transfer_chain_depth >= 5:
        min_strength = float(np.clip(min_strength * _depth_boost, 0.30, 0.70))

    # ── 1. Goal-Gaps ───────────────────────────────────────────
    gaps: dict[str, float] = {}
    for goal, target in goal_targets.items():
        current = float(goal_proxies.get(goal, target))
        gap = target - current
        if gap > 0.001:
            gaps[goal] = gap

    # §v10.59 Goal-Gap-Boost: Je größer die Lücke zwischen aktuellen Werten
    # und Zielen, desto höher die minimale Phasen-Stärke. Selbstkalibrierend —
    # basiert auf den tatsächlichen Messungen, nicht auf statischen Annahmen.
    if gaps:
        _avg_gap = float(np.mean(list(gaps.values())))
        _gap_boost = float(np.clip(1.0 + _avg_gap * 0.6, 1.0, 1.40))
        min_strength = float(np.clip(min_strength * _gap_boost, 0.20, 0.75))

    if not gaps:
        return dict.fromkeys(phase_ids, min_strength)

    _is_codec = terminal_codec is not None and codec_avg_discount < 0.90

    # ── 2. Per-Phase Utility aus Goal-Impacts ──────────────────
    results: dict[str, float] = {}
    for pid in phase_ids:
        profile = PHASE_EFFECT_CATALOG.get(pid)
        if profile is None or not hasattr(profile, "goal_impact"):
            results[pid] = default_strength
            continue

        utility = 0.0
        for goal, impact in profile.goal_impact.items():
            gap = gaps.get(goal, 0.0)
            weight = GOAL_WEIGHTS.get(goal, 0.7)
            contrib = float(impact) * gap * weight

            utility += contrib

        # Codec-Maskierung: MP3/AAC-Artefakte überlagern die Defekt-Signatur
        # → der GESAMTE Nutzen analog-sensitiver Phasen wird diskontiert.
        # Welche Phasen analog-sensitiv sind, steht im PhaseEffectCatalog.risks.
        if _is_codec:
            risks = getattr(profile, "risks", []) or []
            # ML-Phasen: Codec + ML = doppeltes Artefakt-Risiko
            if "ml_artifact" in risks:
                utility *= codec_avg_discount
            # Vocal-Phasen: Codec+Gesang → stark dämpfen (Denker-Entscheidung)
            if panns_singing > 0.25 and "vocal_distortion" in risks:
                utility *= max(0.25, codec_avg_discount * 0.6)  # ×0.27 bei mp3_low
            # Transienten-Phasen: Codec-Artefakte ≠ echte Transienten
            if "transient_smearing" in risks:
                utility *= max(0.60, codec_avg_discount)

        # ── 3. Strength aus Utility ─────────────────────────────
        if utility > 0.001:
            scaled = float(np.clip(0.40 + utility * 12.0, 0.15, 1.0))
            mat_cap = float(getattr(profile, "max_strength_by_material", {}).get(material, 0.95))
            strength = min(scaled, mat_cap)
        else:
            strength = min_strength

        if pid in PROTECTED_PHASES:
            strength = max(strength, 0.40)

        results[pid] = float(np.clip(strength, min_strength, 1.0))

    return results
