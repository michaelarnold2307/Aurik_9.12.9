"""§Anti-Fatigue-Pass — komponenten-getriebene Hörermüdungs-Prävention.

Hörordnung §6 (Ermüdungs-Abbruch, hoerordnung.instructions.md) + §V7
(Ursache statt Symptom, copilot-instructions.md): Die Listening-Fatigue-Metrik
(`backend/core/listening_fatigue_metric.py`) misst drei Komponenten —
Spektralbalance (HF-Anteil), Crest-Faktor (Kompression) und Mikrodynamik.
OneTakeExport korrigierte bisher NUR die HF-Komponente (blindes High-Shelf);
Crest-/Mikrodynamik-Anteile blieben unbehandelt → „BEST-EFFORT (no corrections
possible)“. Zusätzlich trieb die eigene Export-Korrekturkette die Fatigue
(Gain → Limiter → Kompression).

Dieses Modul liefert:

1. `fatigue_correction_plan()` — komponenten-justierter Korrektur-Plan
   (High-Shelf nur wenn `hf_dev` es rechtfertigt; Mikrodynamik-Expansion nur
   wenn Crest/Mikro es rechtfertigen). Keine diskreten Stufen (§V26),
   Stärke kontinuierlich aus den Messwerten abgeleitet.
2. `anti_fatigue_pass()` — wendet den Plan mit Do-No-Harm an (Hörordnung §7:
   übernehmen nur, wenn die Fatigue nachweislich sinkt und kein Peak-Schaden
   entsteht) und liefert Vorher/Nachher-Werte für Telemetrie.

Deterministisch (§G5), numpy/scipy-only, keine externen Modelle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ── Konstanten (psychoakustisch begründet) ──────────────────────────────

_FATIGUE_THRESHOLD: float = 0.40  # synchron zu export_quality_gate._FATIGUE_WARN
_HF_CUT_FREQ: float = 4000.0  # Bark 17+: Ohr reagiert hier empfindlich auf Überbetonung
_HF_CUT_MAX_DB: float = 3.0  # Obergrenze — Brillanz-Boundary (Wohlklang-Ordnung Ebene 3)
_MICRO_EXPAND_MAX_DB: float = 2.0  # sanfte Obergrenze — Dynamics-Restauration, kein Pumping
_COMPONENT_MIN: float = 0.05  # unterhalb gilt eine Komponente als unkritisch


@dataclass
class FatigueCorrectionPlan:
    """Korrektur-Plan aus den Fatigue-Komponenten."""

    hf_cut_db: float = 0.0  # ≤ 0: High-Shelf-Absenkung ab 4 kHz
    micro_expand_db: float = 0.0  # ≥ 0: sanfte Mikrodynamik-Expansion
    reason: str = ""

    @property
    def is_empty(self) -> bool:
        return self.hf_cut_db == 0.0 and self.micro_expand_db == 0.0


@dataclass
class AntiFatigueResult:
    """Ergebnis eines Anti-Fatigue-Passes."""

    audio: np.ndarray
    before: float = 0.0
    after: float = 0.0
    plan: FatigueCorrectionPlan | None = None
    applied: bool = False
    reason: str = ""


def fatigue_correction_plan(
    components: dict[str, float],
    *,
    fatigue: float | None = None,
    threshold: float = _FATIGUE_THRESHOLD,
) -> FatigueCorrectionPlan:
    """Leitet den Korrektur-Plan aus den Fatigue-Komponenten ab.

    Args:
        components: dict aus ``measure_fatigue(..., return_components=True)``
            (hf_dev, crest_dev, micro_dev, …).
        fatigue: Gesamt-Fatigue (wenn None, wird nur komponenten-basiert geplant).
        threshold: Schwelle, ab der überhaupt korrigiert wird.

    Returns:
        FatigueCorrectionPlan — leer, wenn nichts zu tun ist.
    """
    if fatigue is not None and float(fatigue) <= threshold:
        return FatigueCorrectionPlan(reason=f"fatigue {float(fatigue):.2f} ≤ {threshold:.2f} — kein Eingriff")

    hf_dev = float(components.get("hf_dev", 0.0) or 0.0)
    crest_dev = float(components.get("crest_dev", 0.0) or 0.0)
    micro_dev = float(components.get("micro_dev", 0.0) or 0.0)

    hf_cut_db = 0.0
    micro_expand_db = 0.0
    if hf_dev > _COMPONENT_MIN:
        # §V26: kontinuierliche Stärke aus der Abweichung, gedeckelt.
        hf_cut_db = float(np.clip(-hf_dev * 6.0, -_HF_CUT_MAX_DB, 0.0))
    if micro_dev > _COMPONENT_MIN or crest_dev > _COMPONENT_MIN:
        micro_expand_db = float(
            np.clip(1.2 * micro_dev + 0.8 * crest_dev, 0.0, _MICRO_EXPAND_MAX_DB)
        )

    if hf_cut_db == 0.0 and micro_expand_db == 0.0:
        return FatigueCorrectionPlan(
            reason=f"Komponenten unkritisch (hf={hf_dev:.2f} crest={crest_dev:.2f} micro={micro_dev:.2f})"
        )

    return FatigueCorrectionPlan(
        hf_cut_db=round(hf_cut_db, 3),
        micro_expand_db=round(micro_expand_db, 3),
        reason=(
            f"hf_cut={hf_cut_db:+.2f}dB@4kHz, micro_expand={micro_expand_db:+.2f}dB "
            f"(hf_dev={hf_dev:.2f} crest_dev={crest_dev:.2f} micro_dev={micro_dev:.2f})"
        ),
    )


def apply_hf_shelf(audio: np.ndarray, sr: int, cut_db: float) -> np.ndarray:
    """High-Shelf-Absenkung ab 4 kHz (SOS, phasenlinear via sosfiltfilt)."""
    if cut_db >= 0.0:
        _out: np.ndarray = np.asarray(audio, dtype=np.float64)
        return _out
    try:
        from scipy.signal import butter, sosfiltfilt

        sos = butter(2, _HF_CUT_FREQ / (sr / 2), btype="highshelf", output="sos")
        sos[:, :3] *= 10.0 ** (cut_db / 40.0)
        arr = np.asarray(audio, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] <= 2:
            out = np.stack([sosfiltfilt(sos, arr[ch]) for ch in range(arr.shape[0])], axis=0)
        elif arr.ndim == 2:
            out = np.stack([sosfiltfilt(sos, arr[:, ch]) for ch in range(arr.shape[1])], axis=1)
        else:
            out = np.asarray(sosfiltfilt(sos, arr))
        _out = np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        return _out
    except Exception as exc:
        logger.debug("§Anti-Fatigue hf_shelf nicht verfügbar: %s", exc)
        _out = np.asarray(audio, dtype=np.float64)
        return _out


def apply_microdynamics_expansion(audio: np.ndarray, sr: int, expand_db: float) -> np.ndarray:
    """Sanfte Mikrodynamik-Expansion (nur leise Frames anheben, Peaks unberührt).

    Hebt Frames unterhalb des geometrischen Mittels der Frame-Energie an
    (max. ``expand_db``) — erhöht Crest/Mikrodynamik, ohne den Spitzenpegel
    zu verändern (kein Limiter-Oszillationsrisiko in der Export-Schleife).
    """
    if expand_db <= 0.0:
        _out: np.ndarray = np.asarray(audio, dtype=np.float64)
        return _out

    arr = np.asarray(audio, dtype=np.float64)
    mono = arr.mean(axis=0) if arr.ndim == 2 else arr

    frame = max(1, int(sr * 0.05))  # 50 ms
    hop = max(1, frame // 2)
    n_frames = max(1, (len(mono) - frame) // hop + 1)
    env = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        seg = mono[i * hop : i * hop + frame]
        env[i] = float(np.sqrt(np.mean(seg**2)) + 1e-12)

    # Referenz: geometrisches Mittel der Frame-Energie (leise Frames darunter).
    ref = float(np.exp(np.mean(np.log(env + 1e-12))) + 1e-12)
    max_gain = float(10.0 ** (expand_db / 20.0))
    gains = np.clip((ref / (env + 1e-12)) ** 0.5, 1.0, max_gain)

    # Sample-genaue Interpolation der Frame-Gains.
    times = np.arange(n_frames, dtype=np.float64) * hop + frame / 2.0
    sample_axis = np.arange(len(mono), dtype=np.float64)
    gain_curve = np.interp(sample_axis, times, gains)

    if arr.ndim == 2:
        out = arr * gain_curve[np.newaxis, :] if arr.shape[0] <= 2 else arr * gain_curve[:, np.newaxis]
    else:
        out = arr * gain_curve
    _out = np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
    return _out


def anti_fatigue_pass(audio: np.ndarray, sr: int) -> AntiFatigueResult:
    """Komponenten-getriebene Hörermüdungs-Prävention mit Do-No-Harm.

    Misst die Fatigue-Komponenten, wendet den Plan an und übernimmt das
    Ergebnis nur, wenn die Fatigue nachweislich sinkt (Hörordnung §7:
    keine Verschlechterung für einen Metrik-Punkt).
    """
    arr = np.asarray(audio, dtype=np.float64)
    try:
        from backend.core.listening_fatigue_metric import measure_fatigue

        _components = measure_fatigue(arr.astype(np.float32), sr, return_components=True)
        if not isinstance(_components, dict):
            return AntiFatigueResult(audio=arr, reason="Fatigue-Messung nicht verfügbar")
        before = float(_components["fatigue"])
        plan = fatigue_correction_plan(_components, fatigue=before)
        if plan.is_empty:
            return AntiFatigueResult(audio=arr, before=before, after=before, plan=plan, applied=False, reason=plan.reason)

        candidate = arr
        if plan.hf_cut_db < 0.0:
            candidate = apply_hf_shelf(candidate, sr, plan.hf_cut_db)
        if plan.micro_expand_db > 0.0:
            candidate = apply_microdynamics_expansion(candidate, sr, plan.micro_expand_db)

        _after_components = measure_fatigue(candidate.astype(np.float32), sr, return_components=True)
        after = float(_after_components["fatigue"]) if isinstance(_after_components, dict) else before

        # Do-No-Harm (Hörordnung §7): nur übernehmen, wenn Fatigue sinkt.
        if np.isfinite(after) and after < before:
            return AntiFatigueResult(
                audio=candidate,
                before=round(before, 4),
                after=round(after, 4),
                plan=plan,
                applied=True,
                reason=plan.reason,
            )
        return AntiFatigueResult(
            audio=arr,
            before=round(before, 4),
            after=round(after, 4),
            plan=plan,
            applied=False,
            reason=f"Do-No-Harm: Fatigue {before:.3f}→{after:.3f} nicht verbessert — Original bleibt",
        )
    except Exception as exc:
        logger.debug("§Anti-Fatigue nicht blockierend: %s", exc)
        return AntiFatigueResult(audio=arr, reason=f"nicht verfügbar: {exc}")


__all__ = [
    "AntiFatigueResult",
    "FatigueCorrectionPlan",
    "anti_fatigue_pass",
    "apply_hf_shelf",
    "apply_microdynamics_expansion",
    "fatigue_correction_plan",
]
