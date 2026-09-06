#!/usr/bin/env python3
"""§v10.8xx benchmark_effizienz_matrix — Effizienz-Profil der UV3-Restaurierung.

Misst Wandzeit, RT-Faktor und Qualitäts-Kennzahlen derselben Audio-Sequenz
über eine Matrix aus Qualitätsmodi und Ebenen-Schaltern:

  Zellen (Defaults):
    fast       — QualityMode.FAST (8×-RT-Pfad)
    balanced   — QualityMode.BALANCED (32×-RT-Pfad, GUI-Default)
    maximum    — QualityMode.MAXIMUM (32×-RT-Pfad, volle Phasenmenge)
  Ebenen-Abschaltungen (nur Messung, kein Produktions-Eingriff):
    balanced_nopmg    — BALANCED ohne PMGG (§2.29, enable_phase_gate=False)
    balanced_adaptive — BALANCED mit RT-basiertem adaptiven Skipping
                        (enable_adaptive_skipping=True, opt-in)

Pro Zelle werden aufgezeichnet:
  * Wandzeit je Stufe via progress_callback (phase, pct) + Log-Zeitstempel
  * RSS-Verlauf (psutil oder /proc-Fallback, 2-s-Raster)
  * Ergebnis-Kennzahlen: rt_factor, quality_estimate, phases_executed/-skipped,
    PQS-MOS, HPI, Detected-Material
  * restauriertes Audio als WAV (PCM_24) für spätere Hör-/Qualitätsprüfung

Nutzung (aus dem Repo-Root, ROCm-Venv):
  python scripts/benchmark_effizienz_matrix.py --cells fast
  python scripts/benchmark_effizienz_matrix.py            # alle Zellen

SOTA-CI-Erweiterungen (ausschließlich diese Datei):
  * --enforce-budget / --ci : Budget-Enforcement gegen die Performance-Budget-
    Tabelle (§Performance-Budget, copilot-instructions.md, synchron Spec 07 §9).
    Nur Timing-Daten, die die Pipeline tatsächlich liefert, werden geprüft;
    alles andere wird als null + logger.warning (Begründung) ins JSON geschrieben
    — niemals geschätzt. Verletzungen landen unter ``budget_violations`` und
    führen im --ci-Modus zu Exit-Code 1.
  * --bootstrap-ci : 95%-Konfidenzintervalle der Qualitäts-/MUSHRA-Werte je Zelle
    via Percentile-Bootstrap (deterministischer Seed, §G5). Wiederverwendet das
    Bootstrap-Muster aus scripts/non_inferiority_gate.py (RandomState(seed),
    n_boot) und die Kalibrier-Konvention aus scripts/calibrate_mushra_bootstrap.py.
  * --profile-top-phases N : die N langsamsten Phasen je Zelle (Wall-Zeit aus den
    realen progress_callback-Zeitstempeln) landen unter ``top_phases``.
  * --repeats N : N Wiederholungen je Zelle mit deterministischem Seed-Offset
    (AURIK_MASTER_SEED = 42+i, §G5); liefert echte Stichproben für --bootstrap-ci
    (mit einer Beobachtung degeneriert das CI zu null).

Bestehende Aufrufe ohne Flags verhalten sich identisch (deterministisch,
gleicher Output-Aufbau); die additiven JSON-Schlüssel sind an ihre Flags gebunden.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

OUT_ROOT = ROOT / "output_audio" / "benchmark_effizienz"

# --------------------------------------------------------------------------
# Performance-Budget (pro Minute Audio) — §Performance-Budget (copilot-instructions.md),
# synchron zu Spec 07 §9. Einheit: Sekunden Verarbeitung pro Minute Audio.
# --------------------------------------------------------------------------
BUDGETS_S_PER_MIN: dict[str, float] = {
    "defect_scanner": 4.0,
    "phase_pipeline_total": 240.0,
    "feedback_chain": 120.0,
    "excellence_optimizer": 60.0,
    "restorability_estimator": 5.0,
    "export_flac": 10.0,
}

# Deterministischer Bootstrap-Seed (§G5): gleicher Input + gleiche Version
# ⇒ bit-identischer Output. Konvention aus calibrate_mushra_bootstrap.py
# (RandomState(42)) und non_inferiority_gate.py (seed 42, n_boot 5000).
_BOOTSTRAP_SEED = 42
_N_BOOT = 5000
_BOOTSTRAP_ALPHA = 0.05  # 95%-CI

# Basis für deterministische Wiederholungs-Seeds bei --repeats N (§G5).
_REPEAT_BASE_SEED = 42


def _repeat_seed_schedule(base_seed: int, n: int) -> list[int]:
    """Deterministische Seed-Folge für --repeats N (§G5).

    Wiederholung i nutzt AURIK_MASTER_SEED = base_seed + i (0-basiert) —
    reproduzierbar über Läufe hinweg, dokumentiert im Ergebnis-JSON unter
    ``repeat_seeds``.
    """
    return [int(base_seed) + i for i in range(max(1, int(n)))]


# --------------------------------------------------------------------------
# Zellendefinitionen
# --------------------------------------------------------------------------
@dataclass
class Cell:
    id: str
    quality_mode: str
    enable_phase_gate: bool = True
    enable_adaptive_skipping: bool = False


CELLS: list[Cell] = [
    Cell(id="fast", quality_mode="fast"),
    Cell(id="balanced", quality_mode="balanced"),
    Cell(id="maximum", quality_mode="maximum"),
    Cell(id="balanced_nopmg", quality_mode="balanced", enable_phase_gate=False),
    Cell(id="balanced_adaptive", quality_mode="balanced", enable_adaptive_skipping=True),
]

CELL_BY_ID = {c.id: c for c in CELLS}


# --------------------------------------------------------------------------
# RSS-Sampler (eigener Thread, 2-s-Raster)
# --------------------------------------------------------------------------
def _read_rss_mb() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


class RssSampler:
    def __init__(self, interval_s: float = 2.0) -> None:
        self._interval = interval_s
        self.samples: list[tuple[float, float | None]] = []
        self._stop = False

    def start(self) -> None:
        import threading

        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        t0 = time.monotonic()
        while not self._stop:
            self.samples.append((time.monotonic() - t0, _read_rss_mb()))
            time.sleep(self._interval)

    def stop(self) -> None:
        self._stop = True


# --------------------------------------------------------------------------
# Eine Zelle ausführen
# --------------------------------------------------------------------------
def run_cell(
    cell: Cell,
    audio: np.ndarray,
    sr: int,
    log_path: Path,
    out_root: Path,
    save_wav: bool,
    master_seed: int | None = None,
) -> dict[str, Any]:
    from backend.core.unified_restorer_v3 import QualityMode, RestorationConfig, UnifiedRestorerV3

    # Zellen-Log: INFO des Restorers wird hierhin gespiegelt (spätere
    # Phasen-Zeitanalyse aus den ▶/✅-Zeilen möglich).
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    try:
        cfg = RestorationConfig(
            mode=QualityMode(cell.quality_mode),
            enable_phase_gate=cell.enable_phase_gate,
            enable_adaptive_skipping=cell.enable_adaptive_skipping,
        )
        engine = UnifiedRestorerV3(cfg)

        # §G5: deterministische Wiederholungs-Seeds (--repeats N) — die Pipeline
        # liest AURIK_MASTER_SEED via seed_manager.start_session(master_seed=...).
        if master_seed is not None:
            os.environ["AURIK_MASTER_SEED"] = str(int(master_seed))

        progress: list[dict[str, float | str]] = []
        last_pct: dict[str, Any] = {"pct": -1}

        def _cb(pct: float, phase: str, elapsed_s: float) -> None:
            # Nur relevante Übergänge speichern (kein Spam bei identischem pct).
            if int(pct) != int(last_pct["pct"]) or phase != last_pct.get("phase"):
                progress.append(
                    {
                        "t": round(time.monotonic(), 3),
                        "pct": float(pct),
                        "phase": str(phase),
                        "elapsed_s": round(float(elapsed_s), 3),
                    }
                )
                last_pct["pct"] = pct
                last_pct["phase"] = phase

        sampler = RssSampler()
        sampler.start()
        t_wall0 = time.monotonic()
        try:
            result = engine.restore(
                audio,
                sample_rate=sr,
                progress_callback=_cb,
            )
        finally:
            sampler.stop()
        t_wall = time.monotonic() - t_wall0

        wav_path: str | None = None
        if save_wav:
            try:
                import soundfile as sf

                _out_wav = out_root / f"restored_{cell.id}.wav"
                _arr_out = np.asarray(result.audio, dtype=np.float32)
                if _arr_out.ndim == 2 and _arr_out.shape[0] == 2 and _arr_out.shape[1] != 2:
                    _arr_out = _arr_out.T  # (2,N)-intern → (N,2) für soundfile
                # Benchmark-Artefakt (kein Produkt-Export): PCM_24 ohne Dither-Pflicht,
                # da kein bit_depth < 32-Quantisierungsziel fürs menschliche Ohr.
                sf.write(str(_out_wav), _arr_out, sr, format="WAV", subtype="PCM_24")
                wav_path = str(_out_wav)
            except Exception as _wav_exc:
                logger.warning("WAV-Save für Zelle %s fehlgeschlagen: %s", cell.id, _wav_exc)

        pqs_mos: float | None = None
        try:
            pqs_mos = float(getattr(result.pqs_result, "mos", None) or 0.0) or None
        except Exception:
            pqs_mos = None

        meta = dict(result.metadata or {})
        # Vollständige Vocal-/Hör-Invarianten-Telemetrie sichern (nicht nur 60 Keys).
        _vocal_meta: dict[str, str] = {}
        for _mk, _mv in meta.items():
            _mk_l = _mk.lower()
            if any(
                _s in _mk_l
                for _s in ("vocal", "drive", "level_1", "einladung", "vqi", "sing", "breath")
            ):
                try:
                    _vocal_meta[_mk] = json.dumps(_mv, ensure_ascii=False, default=str)[:400]
                except Exception:
                    _vocal_meta[_mk] = str(_mv)[:400]
        return {
            "cell": cell.id,
            "quality_mode": cell.quality_mode,
            "enable_phase_gate": cell.enable_phase_gate,
            "enable_adaptive_skipping": cell.enable_adaptive_skipping,
            "wall_s": round(t_wall, 2),
            "audio_duration_s": round(float(len(audio) / sr), 3),
            "rt_factor": round(t_wall / max(float(len(audio) / sr), 1e-9), 2),
            "engine_total_s": round(float(getattr(result, "total_time_seconds", 0.0) or 0.0), 2),
            "engine_rt_factor": round(float(getattr(result, "rt_factor", 0.0) or 0.0), 2),
            "quality_estimate": round(float(result.quality_estimate or 0.0), 4),
            "material_detected": str(getattr(result, "material_type", "")),
            "phases_executed": int(len(result.phases_executed or [])),
            "phases_skipped": int(len(result.phases_skipped or [])),
            "deferred_phases": list(result.deferred_phases or []),
            "pqs_mos": pqs_mos,
            "hpi": meta.get("hpi", None),
            "artifact_freedom": meta.get("artifact_freedom", None),
            "pipeline_budget_timings": meta.get("pipeline_budget_timings", None),
            "master_seed": int(master_seed) if master_seed is not None else None,
            "n_progress_events": len(progress),
            "progress": progress,
            "rss_samples": sampler.samples,
            "meta_keys": sorted(meta.keys())[:60],
            "vocal_meta": _vocal_meta,
            "warnings": list(getattr(result, "warnings", []) or [])[:10],
            "error": None,
            "wav_path": wav_path,
        }
    except Exception as exc:  # Zellenfehler protokollieren statt Matrix abzubrechen
        logger.exception("Zelle %s fehlgeschlagen", cell.id)
        return {"cell": cell.id, "quality_mode": cell.quality_mode, "error": repr(exc)}
    finally:
        if master_seed is not None:
            os.environ.pop("AURIK_MASTER_SEED", None)
        root_logger.removeHandler(file_handler)
        file_handler.close()


def load_clip(path: Path, seconds: float) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2 and audio.shape[1] == 2:
        pass  # Stereo belassen
    elif audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    # Kein Resample nötig: die Engine konvertiert intern auf 48 kHz.
    n = int(seconds * sr)
    if len(audio) > n:
        audio = audio[:n]
    return audio, sr


# --------------------------------------------------------------------------
# SOTA-CI-Helfer
# --------------------------------------------------------------------------
def _bootstrap_percentile_ci(
    values: list[float],
    n_boot: int = _N_BOOT,
    seed: int = _BOOTSTRAP_SEED,
    alpha: float = _BOOTSTRAP_ALPHA,
) -> tuple[float, float] | None:
    """Percentile-Bootstrap-95%-CI des Mittelwerts (deterministischer Seed, §G5).

    Wiederverwendet das Bootstrap-Muster aus scripts/non_inferiority_gate.py
    (RandomState(seed), n_boot Resampling, np.percentile). Benötigt >= 2
    Beobachtungen; sonst None (kein CI schätzbar — niemals erfinden).
    """
    arr = np.asarray([float(v) for v in values if v is not None], dtype=np.float64)
    if arr.size < 2 or not np.all(np.isfinite(arr)):
        return None
    rng = np.random.RandomState(seed)  # deterministisch (§G5)
    n = int(arr.size)
    means = np.empty(n_boot, dtype=np.float64)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        means[_] = float(arr[idx].mean())
    lo, hi = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)


def _collect_quality_observations(entry: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Sammelt die real verfügbaren Qualitäts-/MUSHRA-Werte einer Zelle.

    Rückgabe: (quality_obs [0..1], mushra_obs [MOS-artig]).
    quality-Skala [0,1]: quality_estimate sowie die 0..1-Kennzahlen hpi/
    artifact_freedom, die im Ergebnis-Dict als 0..1-Anteile geführt werden.
    mushra-Skala (MOS-artig): pqs_mos.
    Es wird ausschließlich auf die bereits im Zellen-Dict gesicherten Felder
    zugegriffen — nichts umskaliert oder neu erfunden.
    """
    quality_obs: list[float] = []
    mushra_obs: list[float] = []

    # Primär-Eintrag + ggf. Wiederholungen (--repeats N): jede Wiederholung
    # liefert eigene Beobachtungen, damit das Bootstrap-CI auf echten
    # Stichproben statt Einzelwerten aufsetzt.
    for _src in [entry, *list(entry.get("repeats") or [])]:
        qe = _src.get("quality_estimate")
        if isinstance(qe, (int, float)) and qe is not None:
            quality_obs.append(float(qe))

        mos = _src.get("pqs_mos")
        if isinstance(mos, (int, float)) and mos is not None:
            mushra_obs.append(float(mos))

        for _key in ("hpi", "artifact_freedom"):
            _v = _src.get(_key)
            if isinstance(_v, (int, float)) and _v is not None:
                quality_obs.append(float(_v))

    return quality_obs, mushra_obs


def _derive_phase_wall_times(progress: list[dict[str, Any]]) -> dict[str, float]:
    """Leitet Wall-Zeit je Phase aus den realen progress_callback-Zeitstempeln ab.

    Jedes progress-Ereignis trägt ``t`` (monotonic) und ``phase``. Das Intervall
    bis zum nächsten Ereignis wird der Phase des aktuellen Ereignisses zugerechnet;
    kontinuierliche Runs derselben Phase akkumulieren. Liefert {} wenn <2 Ereignisse.
    """
    if not progress or len(progress) < 2:
        return {}
    acc: dict[str, float] = {}
    for i in range(len(progress) - 1):
        _t0 = progress[i].get("t")
        _t1 = progress[i + 1].get("t")
        _ph = progress[i].get("phase")
        if _t0 is None or _t1 is None or _ph is None:
            continue
        try:
            dt = float(_t1) - float(_t0)
        except (TypeError, ValueError):
            continue
        if dt < 0:
            continue  # monotonic-Verletzung absichern
        key = str(_ph)
        acc[key] = acc.get(key, 0.0) + dt
    return acc


def _enforce_budget(entry: dict[str, Any], audio_minutes: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Prüft gemessene Zeiten gegen die Performance-Budget-Tabelle.

    Rückgabe: (budget_violations, budget_checks). budget_checks dokumentiert je
    Operation limit, gemessen (sec pro Minute Audio) und ob ein Verstoß vorliegt.
    Primärquelle ist metadata["pipeline_budget_timings"] aus der Pipeline
    (reale Per-Operation-Timings, Spec 07 §9.1d); nur wo diese fehlen, greift
    der Legacy-Fallback engine_total_s/wall_s für phase_pipeline_total. Alles
    andere wird als null + logger.warning (Begründung) geführt — NIEMALS geschätzt.
    """
    checks: dict[str, Any] = {}
    violations: list[dict[str, Any]] = []

    _pt = entry.get("pipeline_budget_timings")
    if not isinstance(_pt, dict):
        _pt = {}
        logger.warning(
            "Budget-Check: keine pipeline_budget_timings im Ergebnis (Zelle %s) — Fallback auf Gesamtzeit",
            entry.get("cell"),
        )

    def _check_op(op: str, measured_s: float | None) -> None:
        if measured_s is None or not isinstance(measured_s, (int, float)) or audio_minutes <= 0:
            checks[op] = None
            logger.warning(
                "Budget-Check '%s' nicht verfügbar (kein Per-Operation-Timing) — als null geführt, nicht geschätzt (Zelle %s)",
                op,
                entry.get("cell"),
            )
            return
        per_min = float(measured_s) / audio_minutes
        limit = BUDGETS_S_PER_MIN[op]
        ok = per_min <= limit
        checks[op] = {
            "operation": op,
            "limit_s_per_min": limit,
            "measured_s_per_min": round(per_min, 2),
            "measured_total_s": round(float(measured_s), 2),
            "violation": not ok,
        }
        if not ok:
            violations.append(
                {
                    "operation": op,
                    "limit_s_per_min": limit,
                    "measured_s_per_min": round(per_min, 2),
                    "cell": entry.get("cell"),
                }
            )

    # 1) Phase-Pipeline gesamt: pipeline_budget_timings.phase_pipeline_s
    #    (Summe der realen per-Phase-Durationen) bevorzugt; sonst engine_total_s/wall_s.
    _pps = _pt.get("phase_pipeline_s")
    if isinstance(_pps, (int, float)) and _pps is not None and float(_pps) > 0:
        _check_op("phase_pipeline_total", float(_pps))
    else:
        _ett = entry.get("engine_total_s")
        _fallback = None
        if isinstance(_ett, (int, float)) and _ett is not None and float(_ett) > 0:
            _fallback = float(_ett)
        elif isinstance(entry.get("wall_s"), (int, float)):
            _fallback = float(entry["wall_s"])
        _check_op("phase_pipeline_total", _fallback)

    # 2) Übrige Operationen — aus pipeline_budget_timings (seit v10.0.20 real
    #    gemessen); export_flac läuft außerhalb des Restorers und bleibt null.
    for _op in ("defect_scanner", "feedback_chain", "excellence_optimizer", "restorability_estimator"):
        _check_op(_op, _pt.get(f"{_op}_s"))
    checks["export_flac"] = None
    logger.warning(
        "Budget-Check 'export_flac' nicht verfügbar (Export läuft außerhalb des Restorers) — als null geführt (Zelle %s)",
        entry.get("cell"),
    )

    return violations, checks


def _top_slowest_phases(progress: list[dict[str, Any]], n: int) -> list[dict[str, Any]] | None:
    """Die n langsamsten Phasen (Wall-Zeit) aus den realen progress-Zeitstempeln.

    None wenn keine Phasen-Timing-Daten vorliegen (liefert der Aufrufer als
    null in den Output + Warning).
    """
    wall = _derive_phase_wall_times(progress)
    if not wall:
        return None
    ordered = sorted(wall.items(), key=lambda kv: -kv[1])[: max(0, int(n))]
    return [{"phase": ph, "wall_s": round(d, 3)} for ph, d in ordered if d >= 0]


def main() -> None:
    ap = argparse.ArgumentParser(description="UV3-Effizienz-Matrix-Benchmark")
    ap.add_argument(
        "--input",
        default=str(ROOT / "test_audio" / "_elke_60s_excerpt.wav"),
        help="Eingabe-WAV (degradiert, real)",
    )
    ap.add_argument("--seconds", type=float, default=30.0, help="zu verarbeitende Länge in s")
    ap.add_argument(
        "--cells",
        nargs="+",
        default=[c.id for c in CELLS],
        help="Zellen: " + ", ".join(CELL_BY_ID),
    )
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--no-wav", action="store_true", help="kein Ergebnis-WAV speichern")
    ap.add_argument(
        "--enforce-budget",
        action="store_true",
        help="Budget-Enforcement gegen die Performance-Budget-Tabelle (copilot-instructions.md, Spec 07 §9)",
    )
    ap.add_argument(
        "--bootstrap-ci",
        action="store_true",
        help="95%%-Konfidenzintervalle der Qualitäts-/MUSHRA-Werte je Zelle (Percentile-Bootstrap, deterministisch)",
    )
    ap.add_argument(
        "--ci",
        action="store_true",
        help="CI-Modus: impliziert --enforce-budget, --bootstrap-ci und Exit-Code 1 bei Budget-Verletzungen",
    )
    ap.add_argument(
        "--profile-top-phases",
        type=int,
        default=None,
        help="Anzahl der langsamsten Phasen je Zelle (Wall-Zeit) in top_phases; nur aktiv wenn gesetzt (--ci setzt 3)",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Wiederholungen je Zelle mit deterministischem Seed-Offset (AURIK_MASTER_SEED = 42+i, §G5); liefert echte Stichproben für --bootstrap-ci",
    )
    args = ap.parse_args()

    # --ci impliziert Enforcement + Bootstrap-CI + Phase-Profiling (top 3) + Exit-Code-Verhalten.
    enforce_budget = bool(args.enforce_budget or args.ci)
    bootstrap_ci = bool(args.bootstrap_ci or args.ci)
    profile_top_phases = 3 if (args.ci and args.profile_top_phases is None) else args.profile_top_phases

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cells = [CELL_BY_ID[cid] for cid in args.cells if cid in CELL_BY_ID]
    if not cells:
        sys.exit(f"Keine gültige Zelle. Verfügbar: {', '.join(CELL_BY_ID)}")

    clip_path = Path(args.input)
    if not clip_path.exists():
        sys.exit(f"Input existiert nicht: {clip_path}")

    audio, sr = load_clip(clip_path, args.seconds)
    logger.warning("Clip: %s | %.1f s | %d Hz | shape=%s", clip_path.name, len(audio) / sr, sr, audio.shape)

    results: list[dict[str, Any]] = []
    any_budget_violation = False
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    _repeat_n = max(1, int(args.repeats))
    for cell in cells:
        logger.warning("=== Zelle %s (%s) gestartet: %s ===", cell.id, cell.quality_mode, time.strftime("%H:%M:%S"))
        # §G5: deterministische Wiederholungs-Seeds (AURIK_MASTER_SEED = 42+i);
        # jede Wiederholung liefert eigene Beobachtungen für --bootstrap-ci.
        _repeat_seeds = _repeat_seed_schedule(_REPEAT_BASE_SEED, _repeat_n)
        _rep_entries: list[dict[str, Any]] = []
        for _rep_i in range(_repeat_n):
            _log_path = out_root / (
                f"cell_{cell.id}_{run_tag}.log"
                if _repeat_n == 1
                else f"cell_{cell.id}_{run_tag}_r{_rep_i}.log"
            )
            _save_wav = (not args.no_wav) and _rep_i == 0
            _rep_entries.append(
                run_cell(cell, audio, sr, _log_path, out_root, _save_wav, master_seed=_repeat_seeds[_rep_i])
            )
        entry = dict(_rep_entries[0])
        entry["clip"] = clip_path.name
        entry["seconds"] = float(args.seconds)
        entry["run_tag"] = run_tag
        if _repeat_n > 1:
            entry["repeat_count"] = _repeat_n
            entry["repeat_seeds"] = list(_repeat_seeds)
            entry["repeats"] = _rep_entries

        # ── Feature 3: Phase-Profiling (top_phases, nur auf Opt-in) ──
        if profile_top_phases is not None:
            _top = _top_slowest_phases(list(entry.get("progress", []) or []), profile_top_phases)
            if _top is None:
                entry["top_phases"] = None
                logger.warning(
                    "Phase-Profiling: keine Phasen-Timing-Daten verfügbar (top_phases=null, Zelle %s)",
                    cell.id,
                )
            else:
                entry["top_phases"] = _top

        # ── Feature 2: Bootstrap-95%-CI (quality_ci95 / mushra_ci95) ──
        if bootstrap_ci:
            quality_obs, mushra_obs = _collect_quality_observations(entry)
            entry["bootstrap_seed"] = _BOOTSTRAP_SEED
            entry["bootstrap_n"] = _N_BOOT
            entry["bootstrap_alpha"] = _BOOTSTRAP_ALPHA
            _qci = _bootstrap_percentile_ci(quality_obs)
            if _qci is None:
                entry["quality_ci95"] = None
                logger.warning(
                    "Bootstrap-CI 'quality_ci95' nicht schätzbar (<2 Qualitäts-Beobachtungen) — null, Zelle %s",
                    cell.id,
                )
            else:
                entry["quality_ci95"] = {"low": round(_qci[0], 4), "high": round(_qci[1], 4)}
            _mci = _bootstrap_percentile_ci(mushra_obs)
            if _mci is None:
                entry["mushra_ci95"] = None
                logger.warning(
                    "Bootstrap-CI 'mushra_ci95' nicht schätzbar (<2 MUSHRA-Beobachtungen) — null, Zelle %s",
                    cell.id,
                )
            else:
                entry["mushra_ci95"] = {"low": round(_mci[0], 4), "high": round(_mci[1], 4)}

        # ── Feature 1: Budget-Enforcement ──
        if enforce_budget:
            _audio_minutes = float(entry.get("seconds", args.seconds)) / 60.0
            _all_violations: list[dict[str, Any]] = []
            _all_checks: list[dict[str, Any]] = []
            for _rep_i, _rep_entry in enumerate(_rep_entries):
                _violations, _checks = _enforce_budget(_rep_entry, _audio_minutes)
                if _repeat_n > 1:
                    for _x in _violations:
                        _x = dict(_x)
                        _x["repeat"] = _rep_i
                        _all_violations.append(_x)
                else:
                    _all_violations.extend(_violations)
                _all_checks.append(_checks)
            entry["budget_checks"] = _all_checks[0] if _repeat_n == 1 else _all_checks
            entry["budget_violations"] = _all_violations
            if _all_violations:
                any_budget_violation = True

        results.append(entry)
        if entry.get("error"):
            logger.warning("Zelle %s FEHLER: %s", cell.id, entry["error"])
        else:
            logger.warning(
                "Zelle %s fertig: wall=%.1f s | RT=%.1f× | quality=%.3f | MOS=%s | Phasen=%d(+%d skip)",
                cell.id,
                entry["wall_s"],
                entry["rt_factor"],
                entry["quality_estimate"],
                entry.get("pqs_mos"),
                entry.get("phases_executed", 0),
                entry.get("phases_skipped", 0),
            )
    out_file = out_root / f"results_{run_tag}.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.warning("Ergebnisse: %s", out_file)
    # Kurzfassung auf stdout
    for r in results:
        if r.get("error"):
            print(f"{r['cell']:22s} FEHLER {r['error'][:80]}")
        else:
            print(
                f"{r['cell']:22s} wall={r['wall_s']:8.1f}s  RT={r['rt_factor']:6.1f}×  "
                f"qual={r['quality_estimate']:.3f}  MOS={r.get('pqs_mos')}  "
                f"phases={r['phases_executed']}(skip {r['phases_skipped']})"
            )

    # Exit-Code 1 bei Budget-Verletzungen im --ci-Modus.
    if args.ci and any_budget_violation:
        logger.warning("Budget-Verletzungen aufgetreten — Exit-Code 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
