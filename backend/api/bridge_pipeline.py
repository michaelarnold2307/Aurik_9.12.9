"""Aurik 10 — Bridge: Pipeline Health State + Trace (§v10)
===========================================================
Pipeline-Telemetrie und -Überwachung für Frontend/CLI → Backend-Core.

Enthält:
  - get_pipeline_trace (vollständiger Pipeline-Trace mit Goal-Timeline)
  - get_pipeline_ab_snapshots (A/B-Vergleichs-Snippets pro Phase als Base64-WAV)
  - run_album_consistency_pass (post-batch LUFS/Spectral-Tilt Alignment)
  - Crash Report Visibility (§v10.993: Live-Fehler erreichen den Nutzer)
  - Guard Report Telemetrie (§v10.990: 4-Schicht + UTMOS-Loop)
  - Restoration Bericht (§v10.996: Plan → Ausführung → Beweis)

Referenz: AGENTS.md §1 (Normative Kette), .github/copilot-instructions.md §V4 Bridge-Verbot.
"""

# pylint: disable=import-outside-toplevel
# cspell:disable

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _coerce_dict_str_any(raw: Any) -> dict[str, Any]:
    """Normalisiert optionale Metadaten auf ein dict[str, Any]."""
    return dict(raw) if isinstance(raw, dict) else {}


# ---------------------------------------------------------------------------
# Pipeline Trace — vollständiger Trace mit Goal-Timeline
# ---------------------------------------------------------------------------


def get_pipeline_trace(result: Any) -> dict[str, Any]:
    """Gibt vollständigen Pipeline-Trace als Dict zurück (für Frontend/CLI/Debug).

    Delegiert an backend.api.debug_api.get_debug_summary() und ergänzt Goal-Timeline.
    Benötigt enable_debug_trace=True beim restore()-Aufruf für vollständige Goal-Daten.
    """
    try:
        from backend.api.debug_api import get_debug_summary, get_goal_fails, get_goals_timeline, get_worst_phases

        summary = _coerce_dict_str_any(get_debug_summary(result))
        summary["goal_timeline"] = get_goals_timeline(result)
        summary["worst_phases"] = get_worst_phases(result, n=5)
        summary["goal_fails"] = get_goal_fails(result)
        return summary
    except Exception as e:
        logger.debug("get_pipeline_trace fehlgeschlagen: %s", e)
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# A/B-Vergleichs-Snapshots pro Phase als Base64-WAV (§v10)
# ---------------------------------------------------------------------------


def get_pipeline_ab_snapshots(*, include_audio: bool = True, max_duration_s: float = 5.0) -> list[dict]:
    """§v10 A/B-Vergleichs-Snapshots für den GUI-Player.

    Liefert Vorher/Nachher-Audio-Snippets pro Phase als Base64-kodiertes WAV.
    Der GUI-Player kann diese direkt dekodieren und abspielen.

    Args:
        include_audio: Wenn True, Base64-WAV-Audio einbetten (größer aber direkt abspielbar)
        max_duration_s: Maximale Dauer pro Snippet in Sekunden (Default 5s)

    Returns:
        Liste von dicts mit phase, pre_audio_b64, post_audio_b64, sample_rate, duration_s
    """
    try:
        import base64
        import io

        import numpy as np

        from backend.core.sota_improvements import get_ab_comparison_state

        ab = get_ab_comparison_state()
        if not ab.ab_snippets:
            return []

        snippets = []
        for s in ab.ab_snippets[-10:]:
            pre = np.asarray(s.get("pre", s.get("pre_phase_audio", np.zeros(1))), dtype=np.float32)
            post = np.asarray(s.get("post", s.get("post_phase_audio", np.zeros(1))), dtype=np.float32)
            phase = str(s.get("phase", "unknown"))

            # Limit duration
            sr = 48000
            max_samples = int(max_duration_s * sr)
            if pre.ndim >= 1 and len(pre) > max_samples:
                mid = len(pre) // 2
                pre = pre[mid - max_samples // 2 : mid + max_samples // 2]
            if post.ndim >= 1 and len(post) > max_samples:
                mid = len(post) // 2
                post = post[mid - max_samples // 2 : mid + max_samples // 2]

            # Ensure mono for smaller payload
            if pre.ndim > 1 and pre.shape[-1] <= 2:
                pre = pre.mean(axis=-1) if pre.shape[-1] == 2 else pre
            if post.ndim > 1 and post.shape[-1] <= 2:
                post = post.mean(axis=-1) if post.shape[-1] == 2 else post

            entry = {
                "phase": phase,
                "sample_rate": sr,
                "duration_s": float(min(len(pre), len(post))) / sr if len(pre) > 0 and len(post) > 0 else 0.0,
            }

            if include_audio:
                # Encode as 16-bit PCM WAV → Base64
                import wave as _wave

                for key, arr in [("pre_audio_b64", pre), ("post_audio_b64", post)]:
                    if len(arr) == 0:
                        entry[key] = ""
                        continue
                    arr_16 = np.clip(arr * 32767, -32768, 32767).astype(np.int16)  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
                    buf = io.BytesIO()
                    with _wave.open(buf, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(sr)
                        wf.writeframes(arr_16.tobytes())
                    entry[key] = base64.b64encode(buf.getvalue()).decode("ascii")
            else:
                entry["pre_shape"] = list(pre.shape) if hasattr(pre, "shape") else [len(pre)]
                entry["post_shape"] = list(post.shape) if hasattr(post, "shape") else [len(post)]

            snippets.append(entry)

        return snippets
    except Exception as _snap_exc:
        logger.warning(
            "§G93 bridge: get_pipeline_ab_snapshots DSP-Ersatzpfad → returning []: %s", _snap_exc, exc_info=True
        )
        return []


# ---------------------------------------------------------------------------
# Album Consistency Pass (§1.4) — post-batch LUFS/Spectral-Tilt Alignment
# ---------------------------------------------------------------------------


def run_album_consistency_pass(
    output_files: list[str],
    sr: int = 48000,
    dry_run: bool = False,
) -> dict:
    """Führt aus: post-batch album consistency pass over a list of restored output files.

    Aligns LUFS (±3 dB max) and spectral tilt (±1.5 dB/oct max shelf) across
    songs that deviate more than the outlier threshold from the album median.
    Songs already within the median ± threshold are NOT touched (§0).

    Args:
        output_files:  Paths to fully-restored WAV/FLAC files.
        sr:            Sample rate to assume (default 48000).
        dry_run:       Analyze only — do not rewrite any files.

    Returns:
        Serializable dict with album-level stats and per-song correction report.
    """
    import time as _time

    from backend.core.album_consistency import get_album_consistency_pass as _get

    _pass = _get()

    # Lade Audio-Daten (defensiv: leere Arrays bei fehlenden Dateien)
    audios: list[np.ndarray] = []
    srs: list[int] = []
    paths: list[str] = []
    for fp in output_files:
        try:
            import soundfile as _sf

            data, file_sr = _sf.read(fp)
            audios.append(np.asarray(data, dtype=np.float32))
            srs.append(file_sr)
            paths.append(fp)
        except Exception:
            # Defensiv: leeres Array statt Crash
            audios.append(np.zeros(48000, dtype=np.float32))
            srs.append(sr)
            paths.append(fp)

    _start = _time.time()
    _report = _pass.analyze(audios=audios, srs=srs, file_paths=paths)  # type: ignore[attr-defined]
    _elapsed = float(_time.time() - _start)

    songs_out = []
    for _sp in _report.songs or []:
        songs_out.append(
            {
                "file": getattr(_sp, "file_path", ""),
                "lufs": float(getattr(_sp, "lufs", 0.0)),
                "spectral_tilt": float(getattr(_sp, "spectral_tilt", 0.0)),
                "dynamic_range_db": float(getattr(_sp, "dynamic_range_db", 12.0)),
                "lufs_correction_db": float(getattr(_sp, "lufs_correction_db", 0.0)),
                "tilt_correction_db": float(getattr(_sp, "tilt_correction_db", 0.0)),
            }
        )

    return {
        "n_songs": int(getattr(_report, "n_songs", 0)),
        "album_lufs_median": float(getattr(_report, "album_lufs_median", -23.0))
        if getattr(_report, "album_lufs_median", -23.0) == getattr(_report, "album_lufs_median", -23.0)
        else None,
        "album_tilt_median": float(getattr(_report, "album_tilt_median", 0.0))
        if getattr(_report, "album_tilt_median", 0.0) == getattr(_report, "album_tilt_median", 0.0)
        else None,
        "corrections_applied": int(getattr(_report, "corrections_applied", 0)),
        "skipped_insufficient_songs": bool(getattr(_report, "skipped_insufficient_songs", False)),
        "elapsed_seconds": _elapsed,
        "dry_run": dry_run,
        "songs": songs_out,
    }


# ---------------------------------------------------------------------------
# §v10.993: Crash-Report-Sichtbarkeit — Live-Fehler erreichen den Nutzer
# ---------------------------------------------------------------------------


def get_new_crash_reports() -> list[dict]:
    """Unbehandelte Fehler der letzten Sitzung(en) — für die GUI-Anzeige beim Start."""
    try:
        from backend.core.crash_reporter import get_new_reports as _new

        return list(_new() or [])
    except Exception as exc:
        logger.debug("§V6 crash_reporter.get_new_reports fehlgeschlagen — leere Liste zurückgegeben: %s", exc)
        return []


def mark_crash_reports_seen() -> None:
    """Setzt die Basislinie: Reports gelten als gesehen (keine erneute Anzeige)."""
    try:
        from backend.core.crash_reporter import mark_reports_seen as _mark

        _mark()
    except Exception as _mark_exc:
        logger.debug("bridge: Markierung nicht möglich: %s", _mark_exc)


def install_crash_handler() -> None:
    """Installiert den globalen Exception-Hook (Spec 08 §11: UI via Bridge)."""
    try:
        from backend.core.crash_reporter import install_crash_handler as _install

        _install()
    except Exception as _inst_exc:
        logger.debug("bridge: Installation nicht möglich: %s", _inst_exc)


# ---------------------------------------------------------------------------
# §v10.990 Guard Report Telemetrie — 4-Schicht + UTMOS-Loop
# ---------------------------------------------------------------------------


def get_guard_report(result: object) -> dict:
    """Guard-Telemetrie (4-Schicht) + UTMOS-Loop aus einem Restorations-Ergebnis.

    Liest §v10.990 RepairReport-Felder (guard_violations, utmos_*) — defensiv:
    jedes fehlende Feld ergibt 0/leer.
    """
    if result is None:
        return {}
    try:
        report = getattr(result, "repair_report", None) or result
        violations = getattr(report, "guard_violations", None) or {}
        meta = getattr(result, "metadata", None) or {}

        def _int(d: dict, key: str) -> int:
            return int(d.get(key, 0) or 0)

        return {
            "guards": {
                "truepeak": int(violations.get("truepeak", 0) or 0),
                "pumping": int(violations.get("pumping", 0) or 0),
                "formant": int(violations.get("formant", 0) or 0),
                "spectral": int(violations.get("spectral", 0) or 0),
                "peak_delta_db": float(getattr(report, "guard_peak_delta_db", 0.0) or 0.0),
            },
            "utmos_loop": {
                "iterations": int(getattr(report, "utmos_iterations", 0) or 0),
                "mos_before": float(getattr(report, "utmos_mos_before", 0.0) or 0.0),
                "mos_after": float(getattr(report, "utmos_mos_after", 0.0) or 0.0),
                "blend_back": int(getattr(report, "utmos_blend_count", 0) or 0) > 0,
            },
            "legacy_meta": {
                "hf_hallucination_fired": _int(meta, "hf_hallucination_guard_fired"),
                "spectral_tilt_fired": _int(meta, "spectral_tilt_guard_fired"),
            },
        }
    except Exception as exc:
        logger.debug("§V6 get_guard_report fehlgeschlagen — leeres Dict zurückgegeben: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# §v10.996: Konsolidierter Restaurierungs-Bericht — der Kreis schließt sich
# ---------------------------------------------------------------------------


def get_restoration_bericht(result: object, defect_result: object = None) -> dict:
    """§v10.996: Der EINE Abschluss-Bericht: Plan → Ausführung → Beweis.

    Verbindet die Einwilligungs-Ansicht (§v10.992, Plan) mit der Ausführung
    (Ergebnis-Metadaten) und dem Sicherheitsnetz (§v10.990 Guards):

      found          — was wurde erkannt (laienverständlich, nach Schwere)
      planned        — was sollte Aurik tun (Handlungssätze in Reihenfolge)
      done_count     — wie viele Schritte tatsächlich ausgeführt wurden
      skipped_count  — übersprungen (kein Effekt / nicht nötig)
      deferred_count — verschoben (ML-Veredelung)
      no_effect_count— liefen, änderten aber nichts
      guards         — Guard-Eingriffe + UTMOS-Kontrolle (get_guard_report)
      proof          — Qualität vorher/nachher, MUSHRA, HPI, Narrator-Verdict
      was_reverted   — Do-No-Harm hat zurückgerollt
    """
    if result is None:
        return {}
    try:
        # Kein Restorations-Ergebnis (kein metadata, kein quality_estimate) → {}
        if not hasattr(result, "metadata") and not hasattr(result, "quality_estimate"):
            return {}
        consent = get_repair_plan_consent(defect_result) if defect_result is not None else {}
        meta = getattr(result, "metadata", None) or {}
        _q_raw = float(getattr(result, "quality_estimate", 0.0) or 0.0)
        _mushra = float((meta.get("mushra") or {}).get("mushra_score", 0.0) or 0.0)
        _hpi = float(meta.get("hpi_score", 0.0) or 0.0)
        _narrator = meta.get("narrator", {}) or {}
        return {
            "found": list((consent.get("found") or [])[:4]),
            "planned": list((consent.get("will_do") or [])[:8]),
            "done_count": int(meta.get("phases_total", 0) or 0),
            "skipped_count": len(getattr(result, "phases_skipped", None) or []),
            "deferred_count": len(getattr(result, "deferred_phases", None) or []),
            "no_effect_count": int(meta.get("no_effect_phase_count", 0) or 0),
            "guards": get_guard_report(result),
            "proof": {
                "quality_before": float(meta.get("restorability_score", 0.0) or 0.0) or None,
                "quality_after": round(_q_raw * 100, 1) if _q_raw > 0 else None,
                "mushra": round(_mushra, 1) if _mushra > 0 else 0.0,
                "hpi": round(_hpi, 4) if _hpi > 0 else 0.0,
                "verdict": str(_narrator.get("verdict", "") or ""),
                "emotional": str(_narrator.get("emotional_summary", "") or ""),
            },
            "was_reverted": bool((meta.get("do_no_harm") or {}).get("reverted", False)),
        }
    except Exception as exc:
        logger.debug("§V6 get_restoration_bericht fehlgeschlagen — leeres Dict zurückgegeben: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# §v10.992: Laienverständliche Einwilligungs-Ansicht — Defekt-Kategorien in Alltagssprache
# (Helper für get_restoration_bericht)
# ---------------------------------------------------------------------------

_DEFECT_LAYMAN_LABELS: dict[str, str] = {
    "click": "Knackser & Klicks",
    "crackle": "Knistern",
    "pop": "Ploppen",
    "hum": "Brummen",
    "hiss": "Rauschen",
    "tape_hiss": "Bandrauschen",
    "vinyl_noise": "Plattenrauschen",
    "wow_flutter": "Tonhöhenschwankungen",
    "clipping": "Übersteuerung",
    "dropout": "Aussetzer",
    "pre_echo": "Echo-Artefakte",
    "print_through": "Kopiereffekte",
    "sibilance": "Zischlaute",
    "breath": "Atemgeräusche",
    "de_essing": "verfärbte Zischlaute",
    "phase_error": "Phasenfehler",
    "distortion": "Verzerrung",
    "gate_chatter": "Gate-Flattern",
    "reverb_tail": "störender Nachhall",
    "unknown": "unklare Störungen",
}

# Plan-Phasen → Handlungs-Sätze in Alltagssprache („Aurik wird …")
_PHASE_ACTIONS: dict[str, str] = {
    "phase_01_click_removal": "Knackser & Klicks entfernen",
    "phase_02_hum_removal": "Brummen entfernen",
    "phase_03_denoise": "Rauschen reduzieren",
    "phase_06_frequency_restoration": "fehlende Höhen rekonstruieren",
    "phase_07_declipper": "Übersteuerung reparieren",
    "phase_09_crackle_removal": "Knistern entfernen",
    "phase_12_wow_flutter_fix": "Tonhöhenschwankungen begradigen",
    "phase_14_phase_correction": "Stereo-Phasenlage korrigieren",
    "phase_19_de_esser": "Zischlaute entschärfen",
    "phase_20_reverb_reduction": "störenden Hall reduzieren",
    "phase_24_dropout_repair": "Aussetzer reparieren",
    "phase_28_surface_noise_profiling": "Plattenrauschen mindern",
    "phase_29_tape_hiss_reduction": "Bandrauschen mindern",
    "phase_55_diffusion_inpainting": "fehlende Stellen rekonstruieren",
    "phase_57_print_through_reduction": "Kopiereffekte entfernen",
}


def _severity_word(value: float) -> str:
    """Schwere in Alltagssprache — identisch mit DefectCounterWidget._severity_word."""
    if value >= 0.6:
        return "Kritisch"
    if value >= 0.3:
        return "Stark"
    if value >= 0.1:
        return "Mittel"
    return "Leicht"


def get_repair_plan_consent(defect_result: object) -> dict:
    """§v10.992: Laienverständliche Einwilligungs-Ansicht des automatischen Plans.

    Aus DefektErgebnis (_consensus_manifest, repair_plan) entstehen:
      found:   [{"label": "Knistern", "severity": "Stark"}, …]  — sortiert nach Schwere
      will_do: ["Knistern entfernen", "Rauschen reduzieren", …]  — Plan-Reihenfolge

    KEINE Entscheidungs-Oberfläche: reine Transparenz, was Aurik tun wird.
    """
    if defect_result is None:
        return {}
    try:
        found: list[dict[str, str]] = []
        manifest = getattr(defect_result, "_consensus_manifest", None)
        if manifest is not None:
            by_cat: dict[str, float] = {}
            for d in list(getattr(manifest, "defects", []) or []):
                _cat = getattr(d, "category", "unknown")
                # Enum-Member (DefectCategory.CLICK) → reiner Wert ("click")
                cat = str(getattr(_cat, "value", None) or _cat or "unknown")
                sev = float(getattr(d, "severity", 0.0) or 0.0)
                by_cat[cat] = max(by_cat.get(cat, 0.0), sev)
            for cat, sev in sorted(by_cat.items(), key=lambda kv: -kv[1]):
                found.append(
                    {
                        "label": _DEFECT_LAYMAN_LABELS.get(cat, cat.replace("_", " ")),
                        "severity": _severity_word(sev),
                    }
                )
        elif hasattr(defect_result, "defect_scores"):
            scores = dict(getattr(defect_result, "defect_scores", {}) or {})
            for key, val in sorted(scores.items(), key=lambda kv: -float(kv[1] or 0.0)):
                found.append(
                    {
                        "label": _DEFECT_LAYMAN_LABELS.get(str(key), str(key).replace("_", " ")),
                        "severity": _severity_word(float(val or 0.0)),
                    }
                )

        will_do: list[str] = []
        plan = getattr(defect_result, "repair_plan", None)
        if plan is not None:
            from backend.core.phase_display_formatter import get_phase_display

            for pid in list(getattr(plan, "phase_order", []) or []):
                pid_str = str(pid)
                action = _PHASE_ACTIONS.get(pid_str)
                if not action:
                    action = str(get_phase_display(pid_str) or pid_str)
                    parts = action.split(" ", 1)
                    if len(parts) == 2 and any(ord(c) > 127 for c in parts[0]):
                        action = parts[1]  # Emoji-Präfix entfernen
                will_do.append(action)
        if not found and not will_do:
            return {}  # Kein Analyse-Material → Frontend blendet die Zeile aus
        return {"found": found, "will_do": will_do}
    except Exception as exc:
        logger.debug("§V6 get_repair_plan_consent fehlgeschlagen — leeres Dict zurückgegeben: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Public API — explizite Export-Liste
# ---------------------------------------------------------------------------

__all__ = [
    # Pipeline Trace — vollständiger Trace mit Goal-Timeline
    "get_pipeline_trace",
    # A/B-Vergleichs-Snapshots pro Phase als Base64-WAV (§v10)
    "get_pipeline_ab_snapshots",
    # Album Consistency Pass (§1.4) — post-batch LUFS/Spectral-Tilt Alignment
    "run_album_consistency_pass",
    # §v10.993: Crash-Report-Sichtbarkeit — Live-Fehler erreichen den Nutzer
    "get_new_crash_reports",
    "mark_crash_reports_seen",
    "install_crash_handler",
    # §v10.990 Guard Report Telemetrie — 4-Schicht + UTMOS-Loop
    "get_guard_report",
    # §v10.996: Konsolidierter Restaurierungs-Bericht — der Kreis schließt sich
    "get_restoration_bericht",
    # §v10.992: Laienverständliche Einwilligungs-Ansicht
    "get_repair_plan_consent",
]
