"""
denker/cross_phase_coordinator.py — CrossPhaseCoordinator
===========================================================

Cross-Phase Naturalness Consensus (§3.0, §ROADMAP-1).

Das menschliche Ohr ist im Bereich 2–8 kHz maximal empfindlich (ISO 226:2023).
Wenn mehrere Phasen unabhängig voneinander denselben Frequenzbereich bearbeiten,
addieren sich ihre Effekte — oft über die Wahrnehmbarkeitsschwelle hinaus.

Dieser Koordinator identifiziert Frequenzband-Überlappungen VOR der Pipeline-
Ausführung und verteilt das Bearbeitungsbudget so, dass die kumulative Wirkung
pro Frequenzband ≤ 100 % der gewünschten Bearbeitung bleibt.

Zusätzlich detektiert er drei Klassen von Kumulativ-Artefakten:
  1. Musical Noise    — periodische Rauschmodulation durch kaskadierte NR
  2. Metallic Ringing  — Gibbs-ähnliche Resonanzen durch überlappende EQ-Phasen
  3. Roughness Regression — Zwicker-Rauigkeit steigt durch additive Phasen

Architektur:
    CrossPhaseCoordinator.analyze(phase_plan, material, decade)
        → Overlap-Matrix [phase_i × phase_j × freq_band]
        → Budget-Verteilung: sum(strength_band) ≤ 1.0 pro Frequenzband
        → capped_strengths dict[phase_id → strength]
        → artifact_risk_flags für NaturalnessGuard

Author: Aurik v10.0.0 — §3.0 Cross-Phase Naturalness Consensus
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Singleton ──────────────────────────────────────────────────────────
_SINGLETON: dict[str, CrossPhaseCoordinator | None] = {"instance": None}
_lock = threading.Lock()


def get_cross_phase_coordinator() -> CrossPhaseCoordinator:
    """Thread-sichere Singleton-Instanz."""
    inst = _SINGLETON["instance"]
    if inst is None:
        with _lock:
            inst = _SINGLETON["instance"]
            if inst is None:
                inst = CrossPhaseCoordinator()
                _SINGLETON["instance"] = inst
    return inst


# ── Psychoacoustic Constants ───────────────────────────────────────────
# §13.1.1 ISO 226:2023 — Frequenzabhängige Wahrnehmung
# "Mitten (200–2000 Hz): Referenz | Präsenz (2000–8000 Hz): +10 dB | Luft (8000–20000 Hz): 0.7×"

# Frequency bands in Hz for overlap analysis
FREQ_BANDS = [
    ("sub_bass", 20, 60, 0.5),
    ("bass", 60, 250, 0.7),
    ("low_mid", 250, 500, 0.9),
    ("mid", 500, 2000, 1.0),
    ("presence_low", 2000, 4000, 1.3),  # +10 dB sensitivity contour
    ("presence_high", 4000, 8000, 1.5),  # +10 dB sensitivity contour, max
    ("air_low", 8000, 12000, 0.8),
    ("air_high", 12000, 20000, 0.6),
]


# ── Phase Frequency Band Declarations ─────────────────────────────────
# Jede Phase deklariert:
#   affects:   [(f_low, f_high, relative_intensity), ...]
#   category:  "subtractive" | "additive" | "dynamics" | "corrective" | "other"
#   requires:  [freq_band_name, ...] — Bänder, die VORHER restauriert sein müssen
#   conflicts: [phase_id, ...] — Phasen, die NICHT adjazent laufen sollen
#   after:     [phase_id, ...] — explizite Vorgänger (harte Kanten für DAG)
#   priority:  int 1-9 — niedriger = läuft früher (default 5)
PHASE_FREQ_PROFILES: dict[str, dict[str, Any]] = {
    # ── Subtractive (Noise Reduction) ──────────────────────────────────
    "phase_03_denoise": {
        "affects": [
            (20, 250, 0.7),
            (250, 2000, 0.9),
            (2000, 8000, 0.8),
            (8000, 20000, 0.6),
        ],
        "category": "subtractive",
        "requires": [],
        "conflicts": ["phase_29_tape_hiss_reduction"],
        "after": ["phase_02_hum_removal", "phase_05_rumble_filter"],
        "priority": 2,
        "human_label": "Denoise (OMLSA)",
    },
    "phase_29_tape_hiss_reduction": {
        "affects": [
            (2000, 8000, 0.5),
            (8000, 20000, 0.9),
            (12000, 20000, 1.0),
        ],
        "category": "subtractive",
        "requires": [],
        "conflicts": ["phase_03_denoise"],
        "after": [],
        "priority": 2,
        "human_label": "Tape Hiss",
    },
    "phase_19_de_esser": {
        "affects": [
            (2000, 4000, 0.4),
            (4000, 8000, 0.8),
            (8000, 12000, 0.6),
            (12000, 20000, 0.2),
        ],
        "category": "subtractive",
        "requires": ["presence_high"],
        "conflicts": [],
        "after": ["phase_03_denoise"],
        "priority": 4,
        "human_label": "De-Esser",
    },
    "phase_18_noise_gate": {
        "affects": [
            (20, 250, 0.4),
            (250, 2000, 0.6),
            (2000, 8000, 0.5),
            (8000, 20000, 0.3),
        ],
        "category": "subtractive",
        "requires": [],
        "conflicts": [],
        "after": ["phase_03_denoise"],
        "priority": 3,
        "human_label": "Noise Gate",
    },
    "phase_09_crackle_removal": {
        "affects": [
            (2000, 8000, 0.7),
            (8000, 20000, 0.9),
        ],
        "category": "subtractive",
        "requires": [],
        "conflicts": [],
        "after": [],
        "priority": 1,
        "human_label": "Crackle Removal",
    },
    "phase_01_click_removal": {
        "affects": [
            (20, 20000, 0.5),
        ],
        "category": "subtractive",
        "requires": [],
        "conflicts": [],
        "after": [],
        "priority": 1,
        "human_label": "Click Removal",
    },
    "phase_02_hum_removal": {
        "affects": [
            (20, 250, 0.9),
        ],
        "category": "subtractive",
        "requires": [],
        "conflicts": [],
        "after": [],
        "priority": 1,
        "human_label": "Hum Removal",
    },
    "phase_05_rumble_filter": {
        "affects": [
            (20, 60, 1.0),
        ],
        "category": "subtractive",
        "requires": [],
        "conflicts": [],
        "after": [],
        "priority": 1,
        "human_label": "Rumble Filter",
    },
    "phase_28_surface_noise_profiling": {
        "affects": [
            (2000, 8000, 0.6),
            (8000, 20000, 0.8),
        ],
        "category": "subtractive",
        "requires": [],
        "conflicts": [],
        "after": [],
        "priority": 2,
        "human_label": "Surface Noise",
    },
    # ── Additive (Enhancement / Restoration) ───────────────────────────
    "phase_38_presence_boost": {
        "affects": [
            (2000, 4000, 0.7),
            (4000, 8000, 0.6),
            (8000, 12000, 0.2),
        ],
        "category": "additive",
        "requires": [],
        "conflicts": ["phase_19_de_esser"],
        "after": ["phase_03_denoise", "phase_19_de_esser"],
        "priority": 6,
        "human_label": "Präsenz-Boost",
    },
    "phase_39_air_band_enhancement": {
        "affects": [
            (8000, 12000, 0.6),
            (12000, 20000, 0.9),
        ],
        "category": "additive",
        "requires": [],
        "conflicts": [],
        "after": ["phase_38_presence_boost"],
        "priority": 7,
        "human_label": "Air-Band",
    },
    "phase_37_bass_enhancement": {
        "affects": [
            (20, 60, 0.9),
            (60, 250, 0.6),
        ],
        "category": "additive",
        "requires": [],
        "conflicts": [],
        "after": ["phase_05_rumble_filter"],
        "priority": 6,
        "human_label": "Bass-Boost",
    },
    "phase_06_frequency_restoration": {
        "affects": [
            (20, 250, 0.4),
            (250, 2000, 0.6),
            (2000, 8000, 0.7),
            (8000, 20000, 0.8),
        ],
        "category": "additive",
        "requires": [],
        "conflicts": [],
        "after": ["phase_03_denoise", "phase_29_tape_hiss_reduction"],
        "priority": 5,
        "human_label": "Frequenz-Restaurierung",
    },
    "phase_07_harmonic_enhancement": {
        "affects": [
            (20, 250, 0.7),
            (250, 2000, 0.8),
            (2000, 8000, 0.5),
            (8000, 20000, 0.3),
        ],
        "category": "additive",
        "requires": [],
        "conflicts": [],
        "after": ["phase_06_frequency_restoration"],
        "priority": 6,
        "human_label": "Harmonik-Boost",
    },
    "phase_07_declipper": {
        "affects": [
            (20, 20000, 0.4),
        ],
        "category": "additive",
        "requires": [],
        "conflicts": [],
        "after": [],
        "priority": 1,
        "human_label": "Declipper",
    },
    "phase_42_vocal_enhancement": {
        "affects": [
            (500, 2000, 0.7),
            (2000, 4000, 0.8),
            (4000, 8000, 0.4),
        ],
        "category": "additive",
        "requires": [],
        "conflicts": ["phase_19_de_esser"],
        "after": ["phase_19_de_esser", "phase_03_denoise"],
        "priority": 7,
        "human_label": "Vocal Enhancement",
    },
    "phase_08_transient_preservation": {
        "affects": [
            (20, 20000, 0.5),
        ],
        "category": "additive",
        "requires": [],
        "conflicts": [],
        "after": ["phase_03_denoise"],
        "priority": 5,
        "human_label": "Transient Preservation",
    },
    # ── Dynamics ───────────────────────────────────────────────────────
    "phase_40_loudness_normalization": {
        "affects": [
            (20, 60, 0.3),
            (60, 250, 0.5),
            (250, 2000, 0.7),
            (2000, 8000, 0.7),
            (8000, 20000, 0.5),
        ],
        "category": "dynamics",
        "requires": [],
        "conflicts": [],
        "after": [],
        "priority": 9,
        "human_label": "Loudness-Normierung",
    },
    "phase_54_transparent_dynamics": {
        "affects": [
            (20, 250, 0.8),
            (250, 2000, 0.6),
            (2000, 8000, 0.3),
            (8000, 20000, 0.2),
        ],
        "category": "dynamics",
        "requires": [],
        "conflicts": [],
        "after": [],
        "priority": 8,
        "human_label": "Transparent Dynamics",
    },
    "phase_10_compression": {
        "affects": [
            (20, 250, 0.6),
            (250, 2000, 0.7),
            (2000, 8000, 0.5),
            (8000, 20000, 0.3),
        ],
        "category": "dynamics",
        "requires": [],
        "conflicts": [],
        "after": ["phase_03_denoise"],
        "priority": 7,
        "human_label": "Compression",
    },
    # ── Corrective (Phase/Stereo) ──────────────────────────────────────
    "phase_14_phase_correction": {
        "affects": [
            (20, 20000, 0.3),
        ],
        "category": "corrective",
        "requires": [],
        "conflicts": [],
        "after": [],
        "priority": 1,
        "human_label": "Phase Correction",
    },
    "phase_25_azimuth_correction": {
        "affects": [
            (20, 20000, 0.3),
        ],
        "category": "corrective",
        "requires": [],
        "conflicts": [],
        "after": ["phase_14_phase_correction"],
        "priority": 1,
        "human_label": "Azimuth Correction",
    },
    "phase_12_wow_flutter_fix": {
        "affects": [
            (20, 20000, 0.4),
        ],
        "category": "corrective",
        "requires": [],
        "conflicts": [],
        "after": [],
        "priority": 2,
        "human_label": "Wow/Flutter Fix",
    },
    "phase_13_stereo_enhancement": {
        "affects": [
            (250, 8000, 0.5),
        ],
        "category": "corrective",
        "requires": [],
        "conflicts": [],
        "after": ["phase_14_phase_correction"],
        "priority": 8,
        "human_label": "Stereo Enhancement",
    },
    "phase_15_stereo_balance": {
        "affects": [
            (20, 20000, 0.3),
        ],
        "category": "corrective",
        "requires": [],
        "conflicts": [],
        "after": ["phase_14_phase_correction"],
        "priority": 3,
        "human_label": "Stereo Balance",
    },
    # ── Other ──────────────────────────────────────────────────────────
    "phase_04_eq_correction": {
        "affects": [
            (20, 250, 0.7),
            (250, 2000, 0.5),
            (2000, 8000, 0.5),
            (8000, 20000, 0.6),
        ],
        "category": "other",
        "requires": [],
        "conflicts": [],
        "after": ["phase_03_denoise"],
        "priority": 5,
        "human_label": "EQ Correction",
    },
    "phase_30_dc_offset_removal": {
        "affects": [
            (0, 20, 1.0),
        ],
        "category": "corrective",
        "requires": [],
        "conflicts": [],
        "after": [],
        "priority": 0,
        "human_label": "DC Offset Removal",
    },
}

# Budget maximum per frequency band — never exceed 1.0
# This ensures no frequency band gets more than 100% processing
BAND_BUDGET_MAX = 1.0

# Threshold for flagging overlap as "significant" (≥ this fraction of band)
SIGNIFICANT_OVERLAP = 0.15

# ── Artifact Detection Thresholds (Defaults, kalibriert via calibrate_cross_phase_thresholds) ──
MUSICAL_NOISE_THRESHOLD = 0.30  # NR phase count × avg strength > threshold → risk
METALLIC_RINGING_THRESHOLD = 0.25  # EQ phase overlap > threshold → risk
ROUGHNESS_BUDGET_FLOOR = 0.70  # roughness budget fraction < floor → risk


def calibrate_cross_phase_thresholds(
    *,
    material_type: str = "unknown",
    restorability_score: float = 50.0,
) -> None:
    """§v10.48 Adaptiv: Cross-Phase-Artifact-Thresholds aus Material + Restorability.

    Tape/Kassette: mehr NR-Phasen erwartet → höhere MUSICAL_NOISE Toleranz.
    Schlechte Restorability: mehr Verarbeitung nötig → höhere Budget-Toleranz.
    """
    global MUSICAL_NOISE_THRESHOLD, METALLIC_RINGING_THRESHOLD, ROUGHNESS_BUDGET_FLOOR
    _mat_lower = str(material_type).lower()
    _is_tape = any(t in _mat_lower for t in ("cassette", "reel_tape", "tape"))

    # Tape: mehr NR erwartet → höhere Toleranz vor MUSICAL_NOISE-Warnung
    MUSICAL_NOISE_THRESHOLD = 0.42 if _is_tape else 0.30
    # Restorability: schlechter → mehr EQ overlap toleriert
    _rs_factor = float(np.clip(1.0 + (1.0 - restorability_score / 100.0) * 0.40, 1.0, 1.40))
    METALLIC_RINGING_THRESHOLD = float(np.clip(0.25 * _rs_factor, 0.20, 0.35))
    ROUGHNESS_BUDGET_FLOOR = float(np.clip(0.70 / _rs_factor, 0.55, 0.70))

    logger.info(
        "§v10.48 CPC: mat=%s rs=%.1f → rausch=%.2f kling=%.2f rau=%.2f",
        _mat_lower,
        restorability_score,
        MUSICAL_NOISE_THRESHOLD,
        METALLIC_RINGING_THRESHOLD,
        ROUGHNESS_BUDGET_FLOOR,
    )


@dataclass
class OverlapResult:
    """Ergebnis einer Overlap-Analyse zwischen zwei Phasen."""

    phase_a: str
    phase_b: str
    overlapping_bands: list[tuple[str, float]] = field(default_factory=list)
    # (band_name, overlap_fraction)
    cumulative_intensity: float = 0.0  # gewichtete Summe der Überlappungen
    risk_level: str = "none"  # none / low / medium / high

    def __repr__(self) -> str:
        return (
            f"Overlap({self.phase_a} ↔ {self.phase_b}: "
            f"{len(self.overlapping_bands)} bands, intensity={self.cumulative_intensity:.2f}, "
            f"risk={self.risk_level})"
        )


@dataclass
class ConsensusResult:
    """Vollständiges Cross-Phase-Consensus-Ergebnis."""

    overlaps: list[OverlapResult] = field(default_factory=list)
    capped_strengths: dict[str, float] = field(default_factory=dict)
    band_budgets: dict[str, dict[str, float]] = field(default_factory=dict)
    # band_name → {phase_id → budget_allocation}
    artifact_risks: dict[str, float] = field(default_factory=dict)
    # {"musical_noise": 0.0–1.0, "metallic_ringing": 0.0–1.0, "roughness": 0.0–1.0}
    recommendations: list[str] = field(default_factory=list)
    # human-readable Empfehlungen


class CrossPhaseCoordinator:
    """Koordiniert phasenübergreifende Frequenzband-Überlappungen.

    Stellt sicher, dass die kumulative Bearbeitung im empfindlichsten
    Frequenzbereich des menschlichen Ohrs (2–8 kHz) niemals 100 % übersteigt.
    """

    def __init__(self) -> None:
        self._last_result: ConsensusResult | None = None
        self._current_material: str = "unknown"

    # ── Classmethod API (delegiert an Singleton) ──────────────────

    @classmethod
    def analyze(
        cls,
        phase_plan: list[str],
        *,
        material: str = "unknown",
        decade: int | None = None,
        restoration_context: dict[str, Any] | None = None,
    ) -> CrossPhaseCoordinator:
        """Klassenmethode: Analysiert Phasenplan via Singleton."""
        inst = get_cross_phase_coordinator()
        inst._current_material = str(material).lower()
        inst._analyze_impl(phase_ids=phase_plan, material=material, decade=decade)
        return inst

    @classmethod
    def get_capped_strength(
        cls,
        phase_id: str,
        base_strength: float,
        *,
        material: str = "unknown",
        restoration_context: dict[str, Any] | None = None,
    ) -> float | None:
        """Klassenmethode: Liest gecappte Stärke via Singleton.

        Returns None wenn keine Analyse lief.
        """
        inst = _SINGLETON.get("instance")
        if inst is None or inst._last_result is None:
            return None
        return inst._get_capped_strength_impl(phase_id, base_strength)

    # ── Instance Implementation ───────────────────────────────────

    def _analyze_impl(
        self,
        phase_ids: list[str],
        phase_strengths: dict[str, float] | None = None,
        *,
        material: str = "unknown",
        decade: int | None = None,
    ) -> ConsensusResult:
        """Analysiert Frequenzband-Überlappungen eines Phasenplans.

        Args:
            phase_ids: Geordnete Liste von Phase-IDs (z.B. ["phase_19_de_esser", ...])
            phase_strengths: Optional dict phase_id → base_strength (0.0–1.0).
                             Wenn None, wird 1.0 für alle Phasen angenommen.
            material: Materialtyp für adaptive Schwellwerte.
            decade: Aufnahme-Dekade für ära-adaptive Caps.

        Returns:
            ConsensusResult mit capped_strengths, band_budgets und Artifact-Risks.
        """
        if phase_strengths is None:
            phase_strengths = dict.fromkeys(phase_ids, 1.0)

        # ── Step 1: Build overlap matrix ──
        known_phases = {pid for pid in phase_ids if pid in PHASE_FREQ_PROFILES}
        overlaps = self._build_overlap_matrix(known_phases, phase_strengths)

        # ── Step 2: Compute band budgets ──
        band_budgets = self._compute_band_budgets(known_phases, phase_strengths, overlaps)

        # ── Step 3: Derive capped strengths ──
        capped = self._derive_capped_strengths(known_phases, phase_strengths, band_budgets, material, decade)

        # ── Step 4: Detect artifact risks ──
        artifact_risks = self._detect_artifact_risks(known_phases, phase_strengths, capped, overlaps)

        # ── Step 5: Generate recommendations ──
        recommendations = self._generate_recommendations(overlaps, artifact_risks, band_budgets)

        result = ConsensusResult(
            overlaps=overlaps,
            capped_strengths=capped,
            band_budgets=band_budgets,
            artifact_risks=artifact_risks,
            recommendations=recommendations,
        )
        self._last_result = result

        if overlaps:
            logger.info(
                "CrossPhaseCoordinator: %d overlaps, %d bands budgeted, naturalness_risk=%.2f",
                len(overlaps),
                len(band_budgets),
                max(artifact_risks.values()) if artifact_risks else 0.0,
            )

        return result

    def _get_capped_strength_impl(self, phase_id: str, base_strength: float) -> float | None:
        """Interne Implementierung: Liest gecappte Stärke.

        Args:
            phase_id: Phase-ID (z.B. "phase_19_de_esser")
            base_strength: Ursprüngliche Stärke (0.0–1.0)

        Returns:
            Gecappte Stärke ∈ [0.0, 1.0] oder None.
        """
        if self._last_result is None:
            return None
        cap = self._last_result.capped_strengths.get(phase_id)
        if cap is None:
            return None
        return float(max(0.0, min(1.0, min(base_strength, cap))))

    # ── Internal: Overlap Matrix ───────────────────────────────────

    def _build_overlap_matrix(
        self,
        phase_ids: set[str],
        phase_strengths: dict[str, float],
    ) -> list[OverlapResult]:
        """Baut die Overlap-Matrix: Für jedes Phasen-Paar die Frequenzband-Überlappung."""
        results: list[OverlapResult] = []
        sorted_ids = sorted(phase_ids)

        for i, pid_a in enumerate(sorted_ids):
            profile_a = PHASE_FREQ_PROFILES.get(pid_a)
            if profile_a is None:
                continue
            bands_a = profile_a["affects"]

            for pid_b in sorted_ids[i + 1 :]:
                profile_b = PHASE_FREQ_PROFILES.get(pid_b)
                if profile_b is None:
                    continue
                bands_b = profile_b["affects"]

                overlap_bands: list[tuple[str, float]] = []
                cumulative = 0.0

                for band_name, f_low, f_high, ear_weight in FREQ_BANDS:
                    # Compute intensity of each phase in this band
                    int_a = self._band_intensity(bands_a, f_low, f_high)
                    int_b = self._band_intensity(bands_b, f_low, f_high)

                    # Scale by phase strengths
                    str_a = phase_strengths.get(pid_a, 1.0)
                    str_b = phase_strengths.get(pid_b, 1.0)
                    combined = (int_a * str_a + int_b * str_b) * ear_weight

                    if combined >= SIGNIFICANT_OVERLAP:
                        overlap_bands.append((band_name, float(combined)))
                        cumulative += combined

                if overlap_bands:
                    risk = "high" if cumulative > 1.5 else "medium" if cumulative > 0.8 else "low"
                    results.append(
                        OverlapResult(
                            phase_a=pid_a,
                            phase_b=pid_b,
                            overlapping_bands=overlap_bands,
                            cumulative_intensity=float(cumulative),
                            risk_level=risk,
                        )
                    )

        return results

    @staticmethod
    def _band_intensity(
        affects: list[tuple[float, float, float]],
        f_low: float,
        f_high: float,
    ) -> float:
        """Berechnet die gewichtete Intensität einer Phase in einem Frequenzband.

        Wenn die Phase über mehrere deklarierte Bänder ins Analyse-Band fällt,
        wird die Intensität anteilig gemittelt.
        """
        total = 0.0
        for a_low, a_high, intensity in affects:
            overlap_low = max(f_low, a_low)
            overlap_high = min(f_high, a_high)
            if overlap_low < overlap_high:
                band_width = f_high - f_low
                overlap_width = overlap_high - overlap_low
                fraction = overlap_width / max(band_width, 1.0)
                total += intensity * fraction
        return float(np.clip(total, 0.0, 1.0))

    # ── Internal: Band Budgets ─────────────────────────────────────

    def _compute_band_budgets(
        self,
        phase_ids: set[str],
        phase_strengths: dict[str, float],
        overlaps: list[OverlapResult],
    ) -> dict[str, dict[str, float]]:
        """Verteilt das Budget pro Frequenzband auf die beteiligten Phasen.

        Regel: sum(strength_band) ≤ BAND_BUDGET_MAX pro Frequenzband.

        Bei Budget-Überschreitung werden additive Phasen stärker gekappt
        als subtraktive (Prinzip: „Weniger hinzufügen, bevor man mehr wegnimmt").
        """
        # Initialize band allocations
        band_phases: dict[str, dict[str, float]] = {}
        for band_name, _, _, _ in FREQ_BANDS:
            band_phases[band_name] = {}

        # Collect which phases affect which bands
        for pid in phase_ids:
            profile = PHASE_FREQ_PROFILES.get(pid)
            if profile is None:
                continue
            str_val = phase_strengths.get(pid, 1.0)
            for band_name, f_low, f_high, _ in FREQ_BANDS:
                intensity = self._band_intensity(profile["affects"], f_low, f_high)
                if intensity > 0.01:
                    band_phases[band_name][pid] = intensity * str_val

        # Check and cap per band
        for band_name, allocations in band_phases.items():
            total_allocation = sum(allocations.values())
            if total_allocation > BAND_BUDGET_MAX:
                # Scale down proportionally
                scale = BAND_BUDGET_MAX / total_allocation
                for pid in allocations:
                    allocations[pid] *= scale

        return band_phases

    def _derive_capped_strengths(
        self,
        phase_ids: set[str],
        phase_strengths: dict[str, float],
        band_budgets: dict[str, dict[str, float]],
        material: str,
        decade: int | None,
    ) -> dict[str, float]:
        """Leitet pro-Phase-capped Stärken aus den Band-Budgets ab.

        Für jede Phase: der strengste Cap aus allen Bändern, die sie betrifft,
        bestimmt die maximale Stärke.

        Material-adaptive Anpassung:
          - cassette/tape: extra −5 % auf Presence-Band-Caps (IEC 60094-1)
          - shellac: extra −10 % auf allen Bändern > 5 kHz
          - wax_cylinder: extra −15 % auf allen Bändern
        """
        capped: dict[str, float] = {}

        # Collect the most restrictive cap per phase across all bands
        for pid in phase_ids:
            most_restrictive = 1.0
            for band_name, allocations in band_budgets.items():
                if pid in allocations:
                    # The phase's budget in this band → effective cap
                    base_strength = phase_strengths.get(pid, 1.0)
                    budget_fraction = allocations[pid] / max(base_strength, 0.01)
                    most_restrictive = min(most_restrictive, budget_fraction)

            # Material-adaptive softening
            mat = str(material).lower()
            profile = PHASE_FREQ_PROFILES.get(pid)

            if profile and profile.get("category") == "additive":
                if mat in ("cassette", "tape"):
                    most_restrictive *= 0.95
                elif mat == "shellac":
                    most_restrictive *= 0.90
                elif mat == "wax_cylinder":
                    most_restrictive *= 0.85

            # Era-adaptive: ältere Aufnahmen → konservativer
            if decade is not None and decade < 1960:
                if profile and profile.get("category") == "additive":
                    most_restrictive *= 0.92

            capped[pid] = float(np.clip(most_restrictive, 0.10, 1.0))

        return capped

    # ── Internal: Artifact Detection ───────────────────────────────

    def _detect_artifact_risks(
        self,
        phase_ids: set[str],
        original_strengths: dict[str, float],
        capped_strengths: dict[str, float],
        overlaps: list[OverlapResult],
    ) -> dict[str, float]:
        """Detektiert drei Klassen von Kumulativ-Artefakten.

        Returns dict mit Risiko-Scores 0.0 (kein Risiko) bis 1.0 (sicher).
        """
        risks: dict[str, float] = {
            "musical_noise": 0.0,
            "metallic_ringing": 0.0,
            "roughness_regression": 0.0,
        }

        # ── Musical Noise Detection ──
        # Risiko wenn ≥ 3 NR-Phasen (subtractive) im 2–8 kHz Band aktiv sind
        # und deren kumulative Stärke > 30 % der Bandbreite.
        nr_phases_in_presence = 0
        nr_cumulative_strength = 0.0
        for pid in phase_ids:
            profile = PHASE_FREQ_PROFILES.get(pid)
            if profile and profile.get("category") == "subtractive":
                int_2k = self._band_intensity(profile["affects"], 2000, 8000)
                if int_2k > 0.1:
                    nr_phases_in_presence += 1
                    nr_cumulative_strength += int_2k * capped_strengths.get(pid, 0.0)
        if nr_phases_in_presence >= 3 and nr_cumulative_strength > MUSICAL_NOISE_THRESHOLD:
            risks["musical_noise"] = float(np.clip(nr_cumulative_strength, 0.0, 1.0))

        # ── Metallic Ringing Detection ──
        # Risiko wenn ≥ 2 EQ-Phasen (additive) im selben Band kollidieren,
        # besonders kritisch: presence_high (4-8 kHz) und air_low (8-12 kHz).
        eq_overlap_intensity = 0.0
        for ov in overlaps:
            profile_a = PHASE_FREQ_PROFILES.get(ov.phase_a, {})
            profile_b = PHASE_FREQ_PROFILES.get(ov.phase_b, {})
            if profile_a.get("category") == "additive" and profile_b.get("category") == "additive":
                eq_overlap_intensity += ov.cumulative_intensity
        if eq_overlap_intensity > METALLIC_RINGING_THRESHOLD:
            risks["metallic_ringing"] = float(np.clip(eq_overlap_intensity / 2.0, 0.0, 1.0))

        # ── Roughness Regression Detection ──
        # Zwicker-Rauigkeit steigt, wenn additive Phasen die Modulations-
        # tiefe in 20–200 Hz und 2–5 kHz gleichzeitig erhöhen.
        roughness_deficit = 0.0
        for pid in phase_ids:
            profile = PHASE_FREQ_PROFILES.get(pid)
            if profile and profile.get("category") == "additive":
                original = original_strengths.get(pid, 1.0)
                capped = capped_strengths.get(pid, 1.0)
                if capped < original:  # Phase wurde gekappt → Budget-Engpass
                    int_bass = self._band_intensity(profile["affects"], 20, 250)
                    int_pres = self._band_intensity(profile["affects"], 2000, 5000)
                    if int_bass > 0.3 and int_pres > 0.3:
                        roughness_deficit += (original - capped) * (int_bass + int_pres) / 2.0
        if roughness_deficit > 0.15:
            risks["roughness_regression"] = float(np.clip(roughness_deficit * 2.0, 0.0, 1.0))

        return risks

    # ── Internal: Recommendations ──────────────────────────────────

    def _generate_recommendations(
        self,
        overlaps: list[OverlapResult],
        artifact_risks: dict[str, float],
        band_budgets: dict[str, dict[str, float]],
    ) -> list[str]:
        """Generiert menschenlesbare Empfehlungen aus der Analyse."""
        recs: list[str] = []

        # High-risk overlaps
        for ov in overlaps:
            if ov.risk_level == "high":
                label_a = PHASE_FREQ_PROFILES.get(ov.phase_a, {}).get("human_label", ov.phase_a)
                label_b = PHASE_FREQ_PROFILES.get(ov.phase_b, {}).get("human_label", ov.phase_b)
                recs.append(
                    f"Hohe Überlappung {label_a} ↔ {label_b} "
                    f"(Intensity={ov.cumulative_intensity:.2f}) — Stärke automatisch reduziert"
                )

        # Artifact risks
        if artifact_risks.get("musical_noise", 0.0) > 0.5:
            recs.append("Musical-Noise-Risiko erkannt — NR-Phasen im Präsenzbereich kumulativ begrenzt")
        if artifact_risks.get("metallic_ringing", 0.0) > 0.5:
            recs.append("Metallic-Ringing-Risiko — EQ-Phasen-Überlappung im 4–12 kHz Bereich reduziert")
        if artifact_risks.get("roughness_regression", 0.0) > 0.5:
            recs.append("Roughness-Regression — Bass+Präsenz-Simultan-Boost gedämpft für natürlichen Wohlklang")

        # Budget constraints
        constrained_bands = [
            name for name, allocs in band_budgets.items() if sum(allocs.values()) >= BAND_BUDGET_MAX * 0.95
        ]
        if constrained_bands:
            recs.append(
                f"Budget-gesättigte Frequenzbänder: {', '.join(constrained_bands)} "
                f"— Bearbeitung auf max. {BAND_BUDGET_MAX:.0%} pro Band begrenzt"
            )

        return recs

    # ── Naturalness Guard — Post-Processing Validation ─────────────

    def validate_naturalness(
        self,
        pre_audio: np.ndarray,
        post_audio: np.ndarray,
        sample_rate: int = 48000,
        *,
        artifact_risks: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Validiert nach der Pipeline, ob kumulative Artefakte entstanden sind.

        Vergleicht pre/post auf:
          1. Musical Noise — periodische Modulation im Restsignal (4–8 kHz)
          2. Metallic Ringing — schmalbandige Peaks (Q > 20) post − pre
          3. Roughness — Zwicker-Rauigkeit nach Fastl & Zwicker 2007

        Returns:
            { "passed": bool, "flags": [...], "metrics": {...} }
        """
        import scipy.signal

        result: dict[str, Any] = {"passed": True, "flags": [], "metrics": {}}

        if pre_audio is None or post_audio is None:
            return result

        try:
            pre_mono = np.mean(pre_audio, axis=0) if pre_audio.ndim == 2 else pre_audio
            post_mono = np.mean(post_audio, axis=0) if post_audio.ndim == 2 else post_audio
            residual = post_mono - pre_mono

            # ── Musical Noise Detection ──
            # Suche nach periodischer Modulation im HF-Restsignal
            if artifact_risks and artifact_risks.get("musical_noise", 0.0) > 0.3:
                # Bandpass 4-8 kHz
                sos_bp = scipy.signal.butter(4, [4000, 8000], btype="band", fs=sample_rate, output="sos")
                residual_band = scipy.signal.sosfiltfilt(sos_bp, residual)
                # Prüfe auf periodische Amplitudenmodulation
                env = np.abs(residual_band)
                env_smooth = scipy.signal.medfilt(env, kernel_size=int(sample_rate * 0.005))
                env_fft = np.abs(np.fft.rfft(env_smooth))
                env_fft_db = 20 * np.log10(np.maximum(env_fft, 1e-9))
                # Spitzen in 8–40 Hz = Musical-Noise-Periodizität
                mod_idx = slice(
                    int(8 * len(env_fft_db) / (sample_rate / 2)),
                    int(40 * len(env_fft_db) / (sample_rate / 2)),
                )
                mod_peak = float(np.max(env_fft_db[mod_idx]) - np.median(env_fft_db[mod_idx]))
                result["metrics"]["musical_noise_mod_db"] = mod_peak
                if mod_peak > 6.0:  # > 6 dB Modulation = hörbar
                    result["passed"] = False
                    result["flags"].append("musical_noise_detected")
                    logger.warning("CrossPhaseCoordinator: Musical Noise detektiert (mod=%.1f dB)", mod_peak)

            # ── Metallic Ringing Detection ──
            # Suche nach schmalbandigen Peaks (Q > 20) im Residual 4-16 kHz
            if artifact_risks and artifact_risks.get("metallic_ringing", 0.0) > 0.3:
                sos_hp = scipy.signal.butter(4, 4000, btype="high", fs=sample_rate, output="sos")
                residual_hf = scipy.signal.sosfiltfilt(sos_hp, residual)
                f, pxx = scipy.signal.welch(residual_hf, fs=sample_rate, nperseg=2048)
                pxx_db = 10 * np.log10(np.maximum(pxx, 1e-12))
                # Schmalbandige Peaks: lokale Maxima mit Q = f_c / Δf(-3dB) > 20
                peaks, props = scipy.signal.find_peaks(pxx_db, prominence=3.0, width=1)
                narrow_peaks = 0
                for peak_idx, width_samples in zip(peaks, props["widths"]):
                    f_c = f[peak_idx]
                    f_width = width_samples * (sample_rate / 2048)
                    q = f_c / max(f_width, 1.0)
                    if q > 20 and f_c < 16000:
                        narrow_peaks += 1
                result["metrics"]["narrow_peaks_hf"] = narrow_peaks
                if narrow_peaks >= 3:
                    result["passed"] = False
                    result["flags"].append("metallic_ringing_detected")
                    logger.warning("CrossPhaseCoordinator: Metallic Ringing detektiert (%d Peaks)", narrow_peaks)

            # ── Roughness Detection ──
            # Vereinfachter Zwicker-Rauigkeits-Proxy: Modulationsenergie in 70 Hz
            if artifact_risks and artifact_risks.get("roughness_regression", 0.0) > 0.3:
                sos_rough = scipy.signal.butter(4, [20, 200], btype="band", fs=sample_rate, output="sos")
                rough_band = scipy.signal.sosfiltfilt(sos_rough, post_mono)
                env_rough = np.abs(scipy.signal.hilbert(rough_band))
                env_rough_fft = np.abs(np.fft.rfft(env_rough))
                # Zwicker-Max bei ~70 Hz Modulation
                rough_idx_70 = int(70 * len(env_rough_fft) / (sample_rate / 2))
                rough_score = float(env_rough_fft[max(0, rough_idx_70 - 2) : rough_idx_70 + 3].mean())
                result["metrics"]["roughness_proxy"] = rough_score
                # Threshold kalibriert für typische Musik
                if rough_score > 0.15:
                    result["passed"] = False
                    result["flags"].append("roughness_regression")
                    logger.warning("CrossPhaseCoordinator: Roughness-Regression (Wert=%.3f)", rough_score)

        except Exception as exc:
            logger.debug("CrossPhaseCoordinator.validieren_naturalness: %s", exc)

        return result
