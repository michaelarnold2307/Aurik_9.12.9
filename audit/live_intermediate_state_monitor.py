from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Logger-Pflicht (§III DSP, AGENTS.md §3): Diagnostik zusätzlich zum
# stdout-Protokoll ([intermediate-audit] …) über den Standard-Logging-Kanal
# ausgeben, damit Fehler in der zentralen Log-Kette sichtbar bleiben.
logger = logging.getLogger(__name__)

try:  # Package-Kontext (Tests/CI): audit.code_weakness_scanner
    from audit.code_weakness_scanner import (
        scan_workspace,
        summarize_findings,
        write_json_report,
        write_markdown_report,
    )
except ImportError:  # Direktskript-Start: python audit/live_intermediate_state_monitor.py
    from code_weakness_scanner import (
        scan_workspace,
        summarize_findings,
        write_json_report,
        write_markdown_report,
    )

EVENT_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("run_start", re.compile(r"AurikDenker\.denke\(\) gestartet", re.IGNORECASE), "info"),
    ("run_step", re.compile(r"AurikDenker \[[0-9]+/[0-9]+\]", re.IGNORECASE), "info"),
    ("phase_start", re.compile(r"▶\s*phase_[0-9]+_[a-z0-9_]+\s+startet", re.IGNORECASE), "info"),
    ("phase_ok", re.compile(r"✅\s*phase_[0-9]+_[a-z0-9_]+:\s*PMGG action=", re.IGNORECASE), "info"),
    ("hpi", re.compile(r"§2\.44 HPI\(", re.IGNORECASE), "info"),
    (
        "artifact_freedom_final",
        re.compile(r"§2\.49 Final artifact_freedom=([0-9]+\.[0-9]+)", re.IGNORECASE),
        "warning",
    ),
    ("vocal_rollback", re.compile(r"VocalNoHarmGate rollback", re.IGNORECASE), "warning"),
    ("temporal_guard", re.compile(r"TemporalContinuityGuard §2\.69", re.IGNORECASE), "warning"),
    ("goal_regression", re.compile(r"ExcellenceOptimizer: Rollback", re.IGNORECASE), "warning"),
    ("runtime_error", re.compile(r"^\[[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:,]+\]\s+ERROR\b", re.IGNORECASE), "critical"),
    ("export_gate_fail", re.compile(r"Export-Quality-Gate FAILED", re.IGNORECASE), "critical"),
    ("batch_error", re.compile(r"BatchProcessingThread: Fehler", re.IGNORECASE), "critical"),
    ("run_end", re.compile(r"AurikDenker\.denke\(\) abgeschlossen", re.IGNORECASE), "info"),
]

PHASE_START_RX = re.compile(r"▶\s*(phase_[0-9]+_[a-z0-9_]+)\s+startet", re.IGNORECASE)
PHASE_OK_RX = re.compile(r"✅\s*(phase_[0-9]+_[a-z0-9_]+):\s*PMGG action=([a-z_]+)", re.IGNORECASE)
PMGG_RX = re.compile(
    r"PMGG\s+([a-z_]+).*?phase=(phase_[0-9]+_[a-z0-9_]+).*?delta=([-+]?[0-9]*\.?[0-9]+).*?action=([a-z_]+)",
    re.IGNORECASE,
)
ACTIVE_INTERVENTION_REJECTED_RX = re.compile(
    r"ActiveIntervention\s+(phase_[0-9]+_[a-z0-9_]+)\s+REJECTED",
    re.IGNORECASE,
)
ACTIVE_INTERVENTION_MICRO_FALLBACK_RX = re.compile(
    r"ActiveIntervention\s+(phase_[0-9]+_[a-z0-9_]+)\s+MICRO-FALLBACK\s+applied",
    re.IGNORECASE,
)

TERMINAL_EVENTS = {"run_end", "export_gate_fail", "batch_error"}

SPEC_GAP_PATTERNS: list[tuple[str, re.Pattern[str], str, str]] = [
    (
        "ui_recursion",
        re.compile(r"RecursionError", re.IGNORECASE),
        "critical",
        "UI-Reentrancy/Signal-Kaskade stabilisieren (modern_window).",
    ),
    (
        "material_floor_call_contract",
        re.compile(r"get_effective_material_floor\(\) got an unexpected keyword argument 'goal'", re.IGNORECASE),
        "critical",
        "Call-Contract get_effective_material_floor angleichen (goal/goal_name Alias).",
    ),
    (
        "temporal_continuity_violation",
        re.compile(r"TemporalContinuityGuard §2\.69 gain_step_db=([0-9]+\.[0-9]+)", re.IGNORECASE),
        "high",
        "Gain-Step-Übergänge glätten, Mikro-Klick-Risiko reduzieren (§2.69).",
    ),
    (
        "vocal_no_harm_rollback",
        re.compile(r"VocalNoHarmGate rollback", re.IGNORECASE),
        "high",
        "Vocal-Pfad stärken, Rollback-Ursache in betroffener Phase beheben (§0p).",
    ),
    (
        "goal_regression",
        re.compile(r"ExcellenceOptimizer: Rollback", re.IGNORECASE),
        "high",
        "Goal-Regression pro Phase minimieren (PMGG/CIG-konform).",
    ),
    (
        "export_gate_failed",
        re.compile(r"Export-Quality-Gate FAILED|quality_estimate\s+0\.000\s+<\s+0\.55", re.IGNORECASE),
        "critical",
        "Pipeline-Fehlerursache vor Export-Gate beheben; fail-closed darf nicht final enden.",
    ),
    (
        "active_intervention_rejected",
        re.compile(r"ActiveIntervention\s+phase_[0-9]+_[a-z0-9_]+\s+REJECTED", re.IGNORECASE),
        "high",
        "REJECTED-Phase mit konservativem Fallback weiterführen statt ohne wirksamen Delta-Eingriff.",
    ),
    (
        "micro_delta_warning",
        re.compile(r"^\[[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:,]+\]\s+WARNING\b", re.IGNORECASE),
        "high",
        "Mikro-Delta sofort analysieren und run-lokal korrigieren (keine Warnung unbehandelt lassen).",
    ),
]

_RUN_WARNING_IGNORE_SUBSTRINGS: tuple[str, ...] = (
    "OffTrack-Guard: Graceful stop angefordert",
    "MICRO-FALLBACK applied",
    "kritisches Stereo-Feldproblem",
    "Stereo-Notfall-Remediation",
)

_SEVERITY_RANK: dict[str, int] = {"critical": 3, "high": 2, "warning": 2, "info": 1, "low": 0}

# Wiederholte identische Warnungen (z. B. PLM-Swapdruck im 10s-Takt) sollen
# das Live-Audit nicht fluten. Deduplizierung erfolgt über normalisierten
# Fingerprint und ein Event-basiertes Cooldown-Fenster.
_RUN_WARNING_DEDUP_COOLDOWN_EVENTS = 24


def _warning_fingerprint(line: str) -> str:
    """Erzeugt einen stabilen Fingerprint aus einer Warnungszeile."""
    payload = line
    if ": " in payload:
        parts = payload.split(": ", 2)
        payload = parts[-1] if len(parts) >= 3 else payload
    payload = re.sub(r"\d+\.\d+", "<num>", payload)
    payload = re.sub(r"\d+", "<int>", payload)
    payload = re.sub(r"\s+", " ", payload).strip().lower()
    return payload


def _run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _evaluate_runtime(workspace: Path) -> dict[str, Any]:
    runtime_report = workspace / "audit/runtime_spec_report_live.json"
    cmd = [
        str(workspace / ".venv_aurik/bin/python"),
        "audit/runtime_spec_check.py",
        "--backend-log",
        "logs/aurik_backend.log",
        "--frontend-log",
        "logs/aurik_frontend.out",
        "--output",
        str(runtime_report),
    ]
    rc, out, err = _run_cmd(cmd, cwd=workspace)
    data = _read_json(runtime_report)
    return {
        "exit_code": rc,
        "stdout": out.strip(),
        "stderr": err.strip(),
        "required_passed": data.get("required_passed"),
        "required_total": data.get("required_total"),
        "compliance_ok": data.get("compliance_ok"),
    }


def _evaluate_final(workspace: Path) -> dict[str, Any]:
    release_report = workspace / "audit/release_report_live.json"
    consolidated_report = workspace / "audit/consolidated_release_status_live.json"

    release_cmd = [
        str(workspace / ".venv_aurik/bin/python"),
        "audit/release_check.py",
        "--audit-path",
        "audit/audit_trail.json",
        "--output",
        str(release_report),
    ]
    cons_cmd = [
        str(workspace / ".venv_aurik/bin/python"),
        "audit/release_runtime_consistency.py",
        "--release-report",
        str(release_report),
        "--runtime-report",
        "audit/runtime_spec_report_live.json",
        "--output",
        str(consolidated_report),
    ]

    rel_rc, rel_out, rel_err = _run_cmd(release_cmd, cwd=workspace)
    cons_rc, cons_out, cons_err = _run_cmd(cons_cmd, cwd=workspace)

    consolidated = _read_json(consolidated_report)
    return {
        "release_exit_code": rel_rc,
        "release_stdout": rel_out.strip(),
        "release_stderr": rel_err.strip(),
        "consolidated_exit_code": cons_rc,
        "consolidated_stdout": cons_out.strip(),
        "consolidated_stderr": cons_err.strip(),
        "final_ready": consolidated.get("final_ready"),
        "runtime_compliance_ok": consolidated.get("runtime_compliance_ok"),
        "required_passed": consolidated.get("required_passed"),
        "required_total": consolidated.get("required_total"),
        "reasons": consolidated.get("reasons"),
    }


def _scan_code_weaknesses(workspace: Path, json_out: Path, md_out: Path) -> dict[str, Any]:
    """Führt den statischen Code-Schwachstellen-Scan aus und schreibt die Reports.

    Der Watchdog darf den Monitor nie stoppen (§V6): Fehler werden als
    Warnung gemeldet und mit einer leeren Zusammenfassung weitergefahren.
    """
    try:
        result = scan_workspace(workspace)
    except Exception as exc:
        logger.warning("code-weakness-scan fehlgeschlagen: %s", exc)
        print(f"[intermediate-audit] WARNUNG code-weakness-scan fehlgeschlagen: {exc}")
        return {"error": str(exc), "total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}

    try:
        write_json_report(json_out, result)
        write_markdown_report(md_out, result)
    except OSError as exc:
        logger.warning("code-weakness-report nicht schreibbar: %s", exc)
        print(f"[intermediate-audit] WARNUNG code-weakness-report nicht schreibbar: {exc}")

    summary = summarize_findings(result)
    summary["report_json"] = str(json_out)
    summary["report_md"] = str(md_out)
    return summary


def _classify(line: str) -> tuple[str, str] | None:
    for event_id, pattern, severity in EVENT_PATTERNS:
        if pattern.search(line):
            return event_id, severity
    return None


def _detect_spec_gap_from_line(line: str) -> dict[str, str] | None:
    # Self-feedback aus dem Guard selbst darf keine neuen Gaps erzeugen.
    # Sonst entsteht eine Warnungs-Schleife im Live-Audit.
    if any(token in line for token in _RUN_WARNING_IGNORE_SUBSTRINGS):
        return None

    for gap_id, pattern, severity, action in SPEC_GAP_PATTERNS:
        if pattern.search(line):
            return {
                "gap_id": gap_id,
                "severity": severity,
                "recommended_action": action,
                "evidence": line.strip(),
            }
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_intervention_queue(path: Path, spec_gaps: list[dict[str, str]], event_payload: dict[str, Any]) -> None:
    items: list[dict[str, Any]] = []
    for gap in spec_gaps:
        gid = str(gap.get("gap_id", "")).strip()
        if not gid:
            continue
        sev = str(gap.get("severity", "high")).strip().lower()
        items.append(
            {
                "gap_id": gid,
                "severity": sev,
                "severity_rank": _SEVERITY_RANK.get(sev, 0),
                "recommended_action": str(gap.get("recommended_action", "")).strip(),
                "evidence": str(gap.get("evidence", "")).strip(),
                "status": "open",
            }
        )

    items.sort(key=lambda x: int(x.get("severity_rank", 0)), reverse=True)
    payload = {
        "timestamp": event_payload.get("timestamp"),
        "event_index": event_payload.get("event_index"),
        "event_id": event_payload.get("event_id"),
        "critical_open": sum(1 for i in items if i.get("severity") == "critical"),
        "high_open": sum(1 for i in items if i.get("severity") == "high"),
        "items": items,
    }
    _write_json(path, payload)


def _write_offtrack_stop_request(path: Path, event_payload: dict[str, Any], active: bool) -> None:
    token = f"{event_payload.get('event_index', 0)}:{event_payload.get('event_id', 'event')}"
    payload = {
        "timestamp": event_payload.get("timestamp"),
        "token": token,
        "active": bool(active),
        "reason": "off_track_trigger" if active else "cleared",
        "event_id": event_payload.get("event_id"),
        "severity": event_payload.get("severity"),
        "worldclass_trend": event_payload.get("worldclass_trend", {}),
        "spec_gaps": event_payload.get("spec_gaps", []),
    }
    _write_json(path, payload)


def _clear_offtrack_stop_request(path: Path, reason: str) -> None:
    """Schreibt einen inaktiven OffTrack-Request (Fail-Safe bei Monitor-Stop)."""
    payload = {
        "timestamp": datetime.now().isoformat(),
        "token": f"clear:{reason}",
        "active": False,
        "reason": reason,
        "event_id": "monitor_shutdown",
        "severity": "info",
        "worldclass_trend": {},
        "spec_gaps": [],
    }
    _write_json(path, payload)


def _phase_from_line(line: str) -> str:
    m = PHASE_START_RX.search(line)
    if m:
        return str(m.group(1)).lower()
    m = PHASE_OK_RX.search(line)
    if m:
        return str(m.group(1)).lower()
    m = PMGG_RX.search(line)
    if m:
        return str(m.group(2)).lower()
    m = ACTIVE_INTERVENTION_REJECTED_RX.search(line)
    if m:
        return str(m.group(1)).lower()
    return ""


def _compute_trajectory(phase_state: dict[str, dict[str, Any]], runtime_eval: dict[str, Any]) -> dict[str, Any]:
    started = sum(1 for v in phase_state.values() if bool(v.get("started")))
    ok = sum(1 for v in phase_state.values() if bool(v.get("ok")))
    pmmg_regressive = sum(int(v.get("pmmg_regressive", 0) or 0) for v in phase_state.values())
    pmmg_total = sum(int(v.get("pmmg_total", 0) or 0) for v in phase_state.values())
    intervention_rejected = sum(int(v.get("intervention_rejected", 0) or 0) for v in phase_state.values())

    reg_rate = (pmmg_regressive / max(1, pmmg_total)) if pmmg_total > 0 else 0.0
    reject_rate = (intervention_rejected / max(1, started)) if started > 0 else 0.0

    # Zwischenzeiten-Logik: schon waehrend des Laufs eng fuehren.
    phase_on_track = reg_rate <= 0.20 and reject_rate <= 0.25
    runtime_ok = bool(runtime_eval.get("compliance_ok", False))

    status = "on_track" if phase_on_track else "off_track"
    if not runtime_ok and started >= 1:
        status = "at_risk"

    return {
        "status": status,
        "phases_started": started,
        "phases_ok": ok,
        "pmmg_regressive_count": pmmg_regressive,
        "pmmg_total_count": pmmg_total,
        "pmmg_regressive_rate": round(reg_rate, 4),
        "intervention_rejected_count": intervention_rejected,
        "intervention_rejected_rate": round(reject_rate, 4),
    }


def _has_offtrack_trigger(spec_gaps: list[dict[str, str]]) -> bool:
    hard_ids = {
        "export_gate_failed",
        "material_floor_call_contract",
        "ui_recursion",
    }
    for gap in spec_gaps:
        gid = str(gap.get("gap_id", "")).strip()
        sev = str(gap.get("severity", "")).strip().lower()
        if gid in hard_ids:
            return True
        if sev == "critical":
            return True
    return False


def main() -> int:
    """Liest Aurik-Run-Log von stdin und schreibt Zwischenstands-Snapshots in JSON/JSONL."""
    parser = argparse.ArgumentParser(description="Live Zwischenstands-Audit fuer Aurik-Runs")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--snapshot-jsonl", default="audit/intermediate_runtime_snapshots.jsonl")
    parser.add_argument("--latest-json", default="audit/intermediate_runtime_latest.json")
    parser.add_argument("--intervention-queue-json", default="audit/intervention_queue_live.json")
    parser.add_argument("--offtrack-stop-request-json", default="audit/offtrack_stop_request.json")
    parser.add_argument(
        "--no-scan-code-weaknesses",
        action="store_true",
        help="Statischen Code-Schwachstellen-Scan am Run-Start deaktivieren.",
    )
    parser.add_argument("--code-weakness-report-json", default="audit/code_weakness_report.json")
    parser.add_argument("--code-weakness-report-md", default="audit/code_weakness_report.md")
    parser.add_argument("--max-events", type=int, default=1000)
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    snapshot_jsonl = workspace / args.snapshot_jsonl
    latest_json = workspace / args.latest_json
    intervention_queue_json = workspace / args.intervention_queue_json
    offtrack_stop_request_json = workspace / args.offtrack_stop_request_json

    def _workspace_path(raw: str) -> Path:
        p = Path(raw)
        return p if p.is_absolute() else workspace / p

    cw_report_json = _workspace_path(args.code_weakness_report_json)
    cw_report_md = _workspace_path(args.code_weakness_report_md)

    event_count = 0
    observed_spec_gaps: dict[str, dict[str, str]] = {}
    run_started = False
    phase_state: dict[str, dict[str, Any]] = {}
    offtrack_latched = False
    warning_last_seen: dict[str, int] = {}
    code_weakness_summary: dict[str, Any] | None = None

    for raw_line in sys.stdin:
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue

        classified = _classify(line)
        if classified is None:
            if not run_started:
                continue
            # Nach Run-Start wird jede Warnung als Mikro-Delta-Intervention behandelt.
            if re.search(r"^\[[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:,]+\]\s+WARNING\b", line, flags=re.IGNORECASE):
                if any(token in line for token in _RUN_WARNING_IGNORE_SUBSTRINGS):
                    continue
                _fp = _warning_fingerprint(line)
                _last_idx = warning_last_seen.get(_fp)
                if _last_idx is not None and (event_count - _last_idx) < _RUN_WARNING_DEDUP_COOLDOWN_EVENTS:
                    continue
                warning_last_seen[_fp] = event_count
                classified = ("run_warning", "warning")
            else:
                continue

        if classified is None:
            continue

        event_id, severity = classified

        # Vor dem Start des aktiven Run-Fensters nur run_start akzeptieren.
        if not run_started and event_id != "run_start":
            continue

        event_count += 1

        if event_id == "run_start":
            run_started = True
            observed_spec_gaps.clear()
            phase_state.clear()
            # Statische Schwachstellen-Prüfung: pro Run einmal neu scannen, damit
            # Codeänderungen zwischen den Runs sofort ausgewiesen werden.
            if not args.no_scan_code_weaknesses:
                code_weakness_summary = _scan_code_weaknesses(workspace, cw_report_json, cw_report_md)

        phase_id = _phase_from_line(line)
        if phase_id:
            st = phase_state.setdefault(
                phase_id,
                {
                    "started": False,
                    "ok": False,
                    "pmmg_total": 0,
                    "pmmg_regressive": 0,
                    "intervention_rejected": 0,
                },
            )
            if PHASE_START_RX.search(line):
                st["started"] = True
            m_ok = PHASE_OK_RX.search(line)
            if m_ok:
                st["ok"] = True
            m_pmmg = PMGG_RX.search(line)
            if m_pmmg:
                st["pmmg_total"] = int(st.get("pmmg_total", 0) or 0) + 1
                try:
                    delta = float(m_pmmg.group(3))
                except (ValueError, TypeError):
                    delta = 0.0
                action = str(m_pmmg.group(4) or "").lower()
                if delta < -0.02 or action in {"rollback", "regressive", "failed"}:
                    st["pmmg_regressive"] = int(st.get("pmmg_regressive", 0) or 0) + 1
            if ACTIVE_INTERVENTION_REJECTED_RX.search(line):
                st["intervention_rejected"] = int(st.get("intervention_rejected", 0) or 0) + 1

        line_gap = _detect_spec_gap_from_line(line)
        if line_gap is not None:
            observed_spec_gaps[str(line_gap["gap_id"])] = line_gap

        # Wenn eine vorher abgelehnte Intervention durch MICRO-FALLBACK
        # abgesichert wurde, gilt der harte REJECTED-Gap als entschärft.
        if ACTIVE_INTERVENTION_MICRO_FALLBACK_RX.search(line):
            observed_spec_gaps.pop("active_intervention_rejected", None)

        runtime_eval = _evaluate_runtime(workspace)
        spec_gaps = list(observed_spec_gaps.values())
        crit_count = sum(1 for g in spec_gaps if g.get("severity") == "critical")
        high_count = sum(1 for g in spec_gaps if g.get("severity") == "high")
        trajectory = _compute_trajectory(phase_state, runtime_eval)
        if _has_offtrack_trigger(spec_gaps):
            trajectory["status"] = "off_track"

        event_payload: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "event_index": event_count,
            "event_id": event_id,
            "severity": severity,
            "log_line": line,
            "runtime": runtime_eval,
            "spec_gaps": spec_gaps,
            "worldclass_trend": {
                "critical_gap_count": crit_count,
                "high_gap_count": high_count,
                "runtime_compliance_ok": bool(runtime_eval.get("compliance_ok", False)),
                "on_worldclass_path": bool(runtime_eval.get("compliance_ok", False)) and crit_count == 0,
                "trajectory": trajectory,
            },
        }
        if code_weakness_summary is not None:
            event_payload["code_weaknesses"] = code_weakness_summary

        if event_id in TERMINAL_EVENTS:
            event_payload["final"] = _evaluate_final(workspace)

        _append_jsonl(snapshot_jsonl, event_payload)
        _write_json(latest_json, event_payload)
        _write_intervention_queue(intervention_queue_json, spec_gaps, event_payload)

        # Laufzeit-Interventionssignal mit Latch:
        # pro zusammenhaengender Off-Track-Phase genau EIN aktiver Stop-Request.
        # Verhindert Token-Sturm (event_index-wechsel) und UI-Logflut.
        _offtrack_now = str(trajectory.get("status", "")) == "off_track" or _has_offtrack_trigger(spec_gaps)
        if _offtrack_now and not offtrack_latched:
            _write_offtrack_stop_request(offtrack_stop_request_json, event_payload, active=True)
            offtrack_latched = True
        elif (not _offtrack_now) and offtrack_latched:
            _write_offtrack_stop_request(offtrack_stop_request_json, event_payload, active=False)
            offtrack_latched = False
        weakness_note = ""
        if code_weakness_summary is not None and "error" not in code_weakness_summary:
            weakness_note = (
                f" weaknesses_total={code_weakness_summary.get('total')} "
                f"weakness_critical={code_weakness_summary.get('critical')} "
                f"weakness_high={code_weakness_summary.get('high')}"
            )
        print(
            "[intermediate-audit] "
            f"event={event_id} severity={severity} "
            f"required={runtime_eval.get('required_passed')}/{runtime_eval.get('required_total')} "
            f"compliance={runtime_eval.get('compliance_ok')}"
            f"{weakness_note}"
        )
        sys.stdout.flush()

        if event_count >= args.max_events:
            print("[intermediate-audit] max-events erreicht, Monitor beendet.")
            return 0

        if event_id in TERMINAL_EVENTS:
            _write_offtrack_stop_request(offtrack_stop_request_json, event_payload, active=False)
            print("[intermediate-audit] terminales Ereignis erkannt, Monitor beendet.")
            return 0

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt as exc:
        # Ctrl+C soll den Monitor sauber beenden (ohne Traceback) und
        # einen evtl. aktiven OffTrack-Stop-Request zurücksetzen.
        _clear_offtrack_stop_request(Path("audit/offtrack_stop_request.json"), "keyboard_interrupt")
        print("[intermediate-audit] monitor durch Benutzerabbruch beendet.")
        raise SystemExit(130) from exc
