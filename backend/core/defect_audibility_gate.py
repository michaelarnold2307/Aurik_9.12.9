"""defect_audibility_gate - Hörbarkeits-Gate für Restdefekte (Hörordnung Ebene 2).

hoerordnung.instructions.md §4: "Reparatur gilt als abgeschlossen, wenn ein
Defekt unter der Maskierungsschwelle liegt - nicht wenn sein Messwert Null
ist."  Dieser Guard übersetzt das in ein Entscheidungs-Gate am Lauf-Ende:

  * Eingabe: Per-Defekt-Reduktion aus dem §v10.702/§v10.703 Post-Scan
    (self._defect_reduction_per_type): pre/post-Severity, reduction,
    masked_events (ERB-maskierte Events laut DefectScanner).
  * Schwelle: material-/ketten-adaptive JND-Schwelle (Severity-Proxy,
    §v10.704 S3) - kanonische Quelle; der Inline-Block in
    unified_restorer_v3.py importiert MATERIAL_JND_OFFSET hierher.
  * Status je Defekttyp: resolved | never_audible | masked | audible |
    physical_cap.  "Maskiert" (ERB) und "physikalische Obergrenze"
    (z. B. bandwidth_loss am Ketten-Ende einer mp3-Quelle) gelten als
    hörbarkeits-erfüllt; nur *unmaskierte* Restdefekte über Schwelle
    lassen das Gate kippen (gate_passed=False) und werden als
    "nachbehandlungswürdig" (improvable_types) ausgewiesen.

Stateless, numpy-frei, rein deklarativ über die Post-Scan-Zahlen -
kein zweiter Audio-Scan (der lief bereits im §B2-Block).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# --- Kanonische Hörbarkeits-Schwellen (§v10.704 S3) --------------------------
# Basis 0.08 (Severity-Skala 0..1); Material-Offset (Vinyl: breiterer
# Frequenzgang → niedrigere Schwelle = mehr hörbar; Cassette: höherer
# Noise-Floor → mehr Maskierung); tiefere Transfer-Kette → +0.01/Ebene.
AUDIBLE_BASE = 0.08
MATERIAL_JND_OFFSET: dict[str, float] = {
    "cassette": 0.04,
    "cassette_tape": 0.04,
    "vinyl": -0.02,
    "lp": -0.02,
    "shellac": 0.02,
    "reel_tape": 0.00,
    "cd_digital": -0.03,
    "mp3_low": 0.05,
    "mp3_high": 0.03,
    "aac": 0.03,
    "streaming": 0.02,
}
AUDIBLE_CLIP_MIN = 0.03
AUDIBLE_CLIP_MAX = 0.15
DEPTH_OFFSET_PER_LEVEL = 0.01

# Defekttypen, deren Rest an der physikalischen Obergrenze der Quelle liegt
# (z. B. durch Codec-Kette verlorenes Band): kein hörbarkeits-relevanter
# Nachbehandlungsspielraum - als "erfüllt mit Dokumentation" gewertet.
PHYSICAL_CAP_DEFECT_TYPES: frozenset[str] = frozenset({"bandwidth_loss"})

# Maskierte Events gelten nur bis zu dieser post-Severity als „maskiert“;
# darüber ist der Restdefekt sicher exponiert (konservativ).
_MASKED_EVENTS_MAX_POST = 0.35

# Defekttyp → zuständige Nachbehandlungs-Phase (m1b, gezielte Stufe-2-Queue).
# Nur Typen mit klarer, sicherer Phasen-Zuordnung; ohne Eintrag kein Deferral
# (kein blindes „mehr von allem“). Phasen 21/35/42 sind Restoration-verboten.
DEFECT_RETRY_PHASE_MAP: dict[str, str] = {
    "hum": "phase_02_hum_removal",
    "hum_buzz": "phase_02_hum_removal",
    "clicks": "phase_01_click_removal",
    "click": "phase_01_click_removal",
    "click_pop": "phase_27_click_pop_removal",
    "crackle": "phase_09_crackle_removal",
    "wow": "phase_12_wow_flutter_fix",
    "flutter": "phase_12_wow_flutter_fix",
    "jitter_artifacts": "phase_12_wow_flutter_fix",
    "motor_interference": "phase_12_wow_flutter_fix",
    "speed_variation": "phase_12_wow_flutter_fix",
    "hiss": "phase_29_tape_hiss_reduction",
    "high_frequency_hiss": "phase_29_tape_hiss_reduction",
    "tape_hiss": "phase_29_tape_hiss_reduction",
    "reverb_excess": "phase_49_advanced_dereverb",
    "echo": "phase_61_groove_echo_cancellation",
    "compression_artifacts": "phase_10_compression",
}


def retry_phases_for_types(improvable_types: list[str]) -> list[str]:
    """Mappt hörbar gebliebene Defekttypen auf ihre Nachbehandlungs-Phase(n)."""
    seen: set[str] = set()
    out: list[str] = []
    for dt in improvable_types or []:
        ph = DEFECT_RETRY_PHASE_MAP.get(str(dt).lower())
        if ph and ph not in seen:
            seen.add(ph)
            out.append(ph)
    return out


def audible_threshold(material_key: str, chain_depth: int = 1) -> float:
    """Material-/ketten-adaptive JND-Hörbarkeitsschwelle (Severity-Skala)."""
    mat = str(material_key or "").lower()
    depth = max(1, int(chain_depth or 1))
    thr = AUDIBLE_BASE + MATERIAL_JND_OFFSET.get(mat, 0.0) + (depth - 1) * DEPTH_OFFSET_PER_LEVEL
    return float(max(AUDIBLE_CLIP_MIN, min(AUDIBLE_CLIP_MAX, thr)))


@dataclass
class DefectAudibilityReport:
    """Ergebnis des Hörbarkeits-Gates am Lauf-Ende."""

    threshold: float
    material_key: str
    chain_depth: int
    per_type: dict[str, dict[str, Any]] = field(default_factory=dict)
    n_total: int = 0
    n_audible_pre: int = 0
    n_audible_post_raw: int = 0  # post >= Schwelle (unabhängig von Maskierung)
    n_masked: int = 0  # post >= Schwelle, aber ERB-maskierte Events vorhanden
    n_audible_unmasked: int = 0  # hörbar geblieben → Gate-relevant
    n_resolved: int = 0  # pre >= Schwelle → post < Schwelle
    n_never_audible: int = 0
    n_physical_cap: int = 0
    improvable_types: list[str] = field(default_factory=list)
    gate_passed: bool = True

    def to_metadata(self) -> dict[str, Any]:
        return {
            "gate_passed": bool(self.gate_passed),
            "threshold": round(float(self.threshold), 4),
            "material": str(self.material_key),
            "chain_depth": int(self.chain_depth),
            "n_total": int(self.n_total),
            "n_audible_pre": int(self.n_audible_pre),
            "n_audible_post_raw": int(self.n_audible_post_raw),
            "n_masked": int(self.n_masked),
            "n_audible_unmasked": int(self.n_audible_unmasked),
            "n_resolved": int(self.n_resolved),
            "n_physical_cap": int(self.n_physical_cap),
            "improvable_types": list(self.improvable_types),
            "per_type": {
                k: {kk: vv for kk, vv in v.items() if kk != "phase_hint"}
                for k, v in self.per_type.items()
            },
        }


def _sev(value: Any) -> float:
    try:
        v = float(value or 0.0)
    except (TypeError, ValueError):
        v = 0.0
    if v != v:  # NaN-Schutz
        return 0.0
    return max(0.0, min(1.0, v))


def evaluate_defect_audibility(
    defect_reduction_per_type: dict[str, dict[str, Any]] | None,
    *,
    material_key: str = "vinyl",
    chain_depth: int = 1,
    physical_cap_types: set[str] | None = None,
) -> DefectAudibilityReport:
    """Bewertet die Restdefekte gegen die Hörbarkeitsschwelle (reine Funktion).

    Args:
        defect_reduction_per_type: §B2-Post-Scan-Daten {type: {pre, post,
            reduction, masked_events, ...}}. Fehlt der Eintrag (kein Post-Scan),
            gilt das Gate als nicht bewertbar → passed=True (kein Block).
        material_key: Material (z. B. "vinyl", "mp3_low").
        chain_depth: Tiefe der Transfer-Kette (1 = keine Zwischenstufen).
        physical_cap_types: zusätzliche Typen ohne Nachbesserungsspielraum.
    """
    thr = audible_threshold(material_key, chain_depth)
    caps = set(PHYSICAL_CAP_DEFECT_TYPES) | set(physical_cap_types or set())
    report = DefectAudibilityReport(
        threshold=thr,
        material_key=str(material_key),
        chain_depth=max(1, int(chain_depth or 1)),
    )
    data = defect_reduction_per_type or {}
    report.n_total = len(data)
    for dt_name, entry in data.items():
        if not isinstance(entry, dict):
            continue
        pre = _sev(entry.get("pre"))
        post = _sev(entry.get("post"))
        masked = 0
        try:
            masked = int(entry.get("masked_events", 0) or 0)
        except (TypeError, ValueError):
            masked = 0
        aud_pre = pre >= thr
        aud_post = post >= thr
        status: str
        if not aud_pre and not aud_post:
            status = "never_audible"
        elif aud_pre and not aud_post:
            status = "resolved"
        elif aud_post and masked > 0 and post <= _MASKED_EVENTS_MAX_POST:
            status = "masked"
        elif aud_post and dt_name in caps:
            status = "physical_cap"
        elif aud_post:
            status = "audible"
        else:  # pre < thr <= post ist durch obige Zweige abgedeckt
            status = "audible"
        report.per_type[dt_name] = {
            "pre": round(pre, 4),
            "post": round(post, 4),
            "reduction": round(max(0.0, pre - post), 4),
            "masked_events": masked,
            "audible_pre": bool(aud_pre),
            "audible_post": bool(aud_post),
            "status": status,
        }
        if aud_pre:
            report.n_audible_pre += 1
        if aud_post:
            report.n_audible_post_raw += 1
        if status == "resolved":
            report.n_resolved += 1
        elif status == "masked":
            report.n_masked += 1
        elif status == "physical_cap":
            report.n_physical_cap += 1
        elif status == "audible":
            report.n_audible_unmasked += 1
            if pre - post > 0.005:  # Reduktion fand statt → Spielraum für mehr
                report.improvable_types.append(dt_name)
        elif status == "never_audible":
            report.n_never_audible += 1
    report.gate_passed = report.n_audible_unmasked == 0
    return report


def log_audibility_report(report: DefectAudibilityReport) -> None:
    """Einheitliche Log-Ausgabe (INFO bei bestanden, WARNING bei Resthörbarem)."""
    if report.gate_passed:
        logger.info(
            "§Hörbarkeits-Gate BESTANDEN (thr=%.3f, material=%s): total=%d "
            "audible_pre=%d resolved=%d masked=%d physical_cap=%d",
            report.threshold,
            report.material_key,
            report.n_total,
            report.n_audible_pre,
            report.n_resolved,
            report.n_masked,
            report.n_physical_cap,
        )
    else:
        logger.warning(
            "§Hörbarkeits-Gate NICHT bestanden (thr=%.3f, material=%s): "
            "%d unmaskierte Restdefekt-Typen über Schwelle - nachbehandlungswürdig: %s",
            report.threshold,
            report.material_key,
            report.n_audible_unmasked,
            ", ".join(report.improvable_types) or "(keine Reduktion erzielt)",
        )
