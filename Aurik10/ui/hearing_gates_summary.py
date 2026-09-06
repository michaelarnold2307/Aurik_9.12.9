"""hearing_gates_summary — GUI-Aufbereitung der Hör-Gates (T1, headless-sicher).

Reine Funktionen über result.metadata (kein PySide-Import → unit-testbar).
Konsumierte Keys (aus unified_restorer_v3.py):
  * metadata["audibility_gate"] = {gate_passed, threshold, n_audible_unmasked,
    n_masked, n_resolved, improvable_types, …}
  * metadata["einladungs_gate_passed"/"_corrected"/"_sharpness_jump"/…]
  * metadata["vocal_drive_*"] (vocal_drive_telemetry-Felder)
"""

from __future__ import annotations

from typing import Any


def _ag(meta: dict) -> dict:
    v = meta.get("audibility_gate") or {}
    return v if isinstance(v, dict) else {}


def _num(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def hearing_gate_status(meta: dict) -> str:
    """green | yellow | red — Ampel über alle Hör-Gates."""
    ag = _ag(meta)
    if ag.get("gate_passed") is False:
        return "red"
    if meta.get("einladungs_gate_passed") is False and not meta.get("einladungs_gate_corrected"):
        return "red"
    if meta.get("vocal_drive_hard_revert") is True:
        return "red"
    # gelb: korrigiert, maskiert oder Nachbehandlung gequeued
    if (
        meta.get("einladungs_gate_corrected") is True
        or _num(ag.get("n_masked")) > 0
        or bool(ag.get("improvable_types"))
        or meta.get("vocal_drive_blend", 1.0) < 1.0
    ):
        return "yellow"
    return "green"


def hearing_gates_line(meta: dict) -> str:
    """Kurzzeile für den Qualitäts-Score-Text."""
    icon = {"green": "\U0001F7E2", "yellow": "\U0001F7E1", "red": "\U0001F534"}[hearing_gate_status(meta)]
    return f"Hör-Gates: {icon} {hearing_gate_status(meta).upper()}"


def hearing_gates_details(meta: dict) -> list[str]:
    """Detailzeilen für den Ergebnis-Banner."""
    out: list[str] = []
    ag = _ag(meta)
    if ag:
        out.append(
            "Audibility: bestanden" if ag.get("gate_passed") is not False else "Audibility: Restdefekte über Schwelle"
        )
        if _num(ag.get("n_audible_unmasked")) > 0:
            out.append(f"  hörbar: {int(_num(ag.get('n_audible_unmasked')))} Typ(en)")
        if ag.get("improvable_types"):
            out.append("  Stufe-2-Queue: " + ", ".join(str(x) for x in ag["improvable_types"][:6]))
        if _num(ag.get("n_masked")) > 0:
            out.append(f"  maskiert: {int(_num(ag.get('n_masked')))} Typ(en)")
    eg_passed = meta.get("einladungs_gate_passed")
    if eg_passed is False:
        if meta.get("einladungs_gate_corrected"):
            out.append("Wohlklang: korrigiert (Blend)")
        else:
            out.append("Wohlklang: verletzt")
    elif eg_passed is True:
        out.append("Wohlklang: erfüllt")
    if meta.get("vocal_drive_hard_revert") is True:
        out.append("Vocal-Drive: Phasen-Rücknahme aktiv")
    elif meta.get("vocal_drive_blend", 1.0) < 1.0:
        out.append(f"Vocal-Drive: Blend {_num(meta.get('vocal_drive_blend')):.2f}")
    return out or ["Hör-Gates: keine Gate-Metadaten (n/a)"]


def apply_resolved_defects(
    counts: dict[str, int], resolved: list[str] | None
) -> tuple[dict[str, int], int, list[str]]:
    """Defekt-Chip-Subtraktion (Echtzeit, pure Logik).

    Args:
        counts: Chip-Zähler je Defekttyp (aktuelle Restanzahl).
        resolved: Defekttypen, die seit dem letzten Callback behoben wurden
            (Engine-Payload live_metrics["resolved"]).

    Returns:
        (neue Zähler, verbleibende Gesamtzahl, Liste der nun vollständig
        behobenen Typen) — nie negativ, unbekannte Typen werden ignoriert.
    """
    out = dict(counts or {})
    done: list[str] = []
    for dt in resolved or []:
        key = str(dt)
        if key not in out or out[key] <= 0:
            continue
        out[key] = out[key] - 1
        if out[key] == 0:
            done.append(key)
    return out, int(sum(out.values())), done
