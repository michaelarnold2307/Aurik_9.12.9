#!/usr/bin/env python3
# pylint: disable=import-outside-toplevel
"""
aurik_debug.py — Standalone Debug-CLI für die Aurik-Pipeline.

LEGACY_NON_RELEASE: Dieser Debug-CLI nutzt absichtlich einen direkten UV3-Bypass
(`UnifiedRestorerV3.restore(...)`) für Telemetrie-Diagnosen. Er ist kein
Desktop-Release-Einstieg und darf den Canonical Contract der Release-Pfade
nicht ersetzen.

Läuft die vollständige Restaurierungs-Pipeline mit aktiviertem Debug-Trace
und gibt einen strukturierten Bericht aus — ohne Raten, ohne Suchen in Logs.

Usage:
    python -m cli.aurik_debug <audio_datei> [Optionen]

    python -m cli.aurik_debug song.mp3
    python -m cli.aurik_debug song.mp3 --mode studio_2026
    python -m cli.aurik_debug song.mp3 --trace-json trace.json
    python -m cli.aurik_debug song.mp3 --goals-only
    python -m cli.aurik_debug song.mp3 --decisions-only
    python -m cli.aurik_debug song.mp3 --summary-json

Ausgaben:
    - Vollständiger Debug-Bericht (stdout)
    - Optionales JSON-Trace (--trace-json)
    - Optionale restaurierte Audio-Datei (--out)

Exit-Codes:
    0 = Erfolg (alle Goals ≥ Schwellwert oder kein Gate-Fail)
    1 = Goal-Fails vorhanden
    2 = Pipeline-Fehler
    3 = Import-Fehler / Konfigurationsfehler
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Logging frühzeitig einrichten
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("aurik_debug")


def _setup_workspace() -> bool:
    """Fügt Workspace-Root zum sys.path hinzu falls nötig."""
    script_dir = Path(__file__).resolve().parent
    workspace = script_dir.parent
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    return True


def _load_audio(path: str) -> tuple[Any, int]:
    """Lädt Audio mit Aurik-konformem load_audio_file()."""
    try:
        from backend.file_import import load_audio_file

        payload = load_audio_file(path, do_carrier_analysis=False)
        if payload is None:
            raise RuntimeError("Audio-Import lieferte kein Ergebnisobjekt")
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unerwartetes Import-Ergebnis: {type(payload)!r}")

        err = payload.get("error")
        if err:
            raise RuntimeError(str(err))

        audio = payload.get("audio")
        sr = payload.get("sr")
        if audio is None or sr is None:
            raise RuntimeError("Audio-Import unvollständig: 'audio' oder 'sr' fehlt")

        return audio, int(sr)
    except Exception as e:
        logger.error("Audio-Import fehlgeschlagen (%s): %s", path, e)
        raise


def _run_restore(audio: Any, sr: int, mode: str, verbose: bool) -> Any:
    """Führt Pipeline mit aktiviertem Debug-Trace aus."""
    try:
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3
    except ImportError as e:
        logger.error("UV3 Import fehlgeschlagen: %s", e)
        raise

    if verbose:
        logging.getLogger("backend").setLevel(logging.INFO)
        logging.getLogger("backend.core").setLevel(logging.DEBUG)

    restorer = UnifiedRestorerV3()

    _progress_log: list[str] = []

    def _progress(pct: int, phase: str, elapsed: float) -> None:
        if verbose:
            print(f"  [{pct:3d}%] {phase} ({elapsed:.1f}s)", file=sys.stderr, flush=True)
        _progress_log.append(f"{pct}% {phase}")

    print(f"\n⚙ Pipeline läuft ({mode})…", file=sys.stderr, flush=True)
    t0 = time.perf_counter()

    result = restorer.restore(
        audio,
        sample_rate=sr,
        mode=mode,
        progress_callback=_progress,
        enable_debug_trace=True,  # §DEBUG: Goal-Daten pro Phase aktivieren
    )

    elapsed = time.perf_counter() - t0
    print(f"✓ Fertig in {elapsed:.1f}s", file=sys.stderr, flush=True)
    return result


def _print_header(audio_path: str, mode: str) -> None:
    print(f"\n{'═' * 80}")
    print(f"  AURIK DEBUG — {Path(audio_path).name}  |  Modus: {mode}")
    print(f"{'═' * 80}")


def _print_summary(result: Any) -> None:
    """Kurzübersicht ohne Trace-Import."""
    mat = getattr(result, "material_type", None)
    cfg = getattr(result, "config", None)
    phases_ex = len(getattr(result, "phases_executed", []) or [])
    phases_sk = len(getattr(result, "phases_skipped", []) or [])
    rt = getattr(result, "rt_factor", 0) or 0
    meta = getattr(result, "metadata", {}) or {}
    fail_reasons = meta.get("fail_reasons", [])

    print(f"\n  Material    : {mat.value if hasattr(mat, 'value') else mat or '?'}")  # type: ignore[union-attr]
    print(f"  Mode        : {cfg.mode.value if cfg and hasattr(cfg.mode, 'value') else '?'}")
    print(f"  Zeit        : {getattr(result, 'total_time_seconds', 0):.1f}s  RT-Faktor: {rt:.2f}×")
    print(f"  Phasen      : {phases_ex} ausgeführt, {phases_sk} übersprungen")
    if fail_reasons:
        print(f"\n  ✗ FAIL-REASONS ({len(fail_reasons)}):")
        for fr in (fail_reasons if isinstance(fail_reasons, list) else [str(fail_reasons)])[:5]:
            print(f"    • {fr}")


def _resolve_debug_modes(raw_mode: str | None) -> tuple[str, str, str]:
    """Normalisiert Legacy-CLI-Modi auf Denker- und Debug-Gate-Modi.

    Returns:
        tuple aus (denker_mode, goal_gate_mode, display_mode)
    """
    raw_norm = str(raw_mode or "restoration").strip().lower().replace("_", "").replace(" ", "")

    # Legacy-Debug-Aliase (keine Release-Oberfläche) auf die zwei kanonischen Modi abbilden.
    legacy_aliases = {
        "fast": "restoration",
        "balanced": "restoration",
        "maximum": "studio2026",
    }
    if raw_norm in legacy_aliases:
        denker_mode = legacy_aliases[raw_norm]
    else:
        try:
            from backend.api.bridge import normalize_user_mode

            canonical = normalize_user_mode(raw_mode)
            denker_mode = "studio2026" if canonical == "Studio 2026" else "restoration"
        except Exception:
            denker_mode = "studio2026" if raw_norm in {"studio", "studio2026"} else "restoration"

    goal_gate_mode = "studio_2026" if denker_mode == "studio2026" else "restoration"
    display_mode = "Studio 2026" if denker_mode == "studio2026" else "Restoration"
    return denker_mode, goal_gate_mode, display_mode


def main(argv: list[str] | None = None) -> int:
    """CLI-Einstieg für den Legacy-Debug-Bypass mit strukturierter Telemetrie."""
    parser = argparse.ArgumentParser(
        prog="aurik_debug",
        description="Aurik Pipeline Debug-CLI — vollständige Telemetrie ohne Raten",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  aurik_debug song.mp3
  aurik_debug song.mp3 --mode studio_2026 --trace-json trace.json
  aurik_debug song.mp3 --goals-only
  aurik_debug song.mp3 --decisions-only
  aurik_debug song.mp3 --out restauriert.wav

Exit-Codes: 0=OK, 1=Goal-Fails, 2=Pipeline-Fehler, 3=Import-Fehler
""",
    )
    parser.add_argument("audio", help="Eingabe-Audio-Datei (mp3, wav, flac, …)")
    parser.add_argument(
        "--mode",
        default="restoration",
        choices=["restoration", "studio_2026", "fast", "balanced", "quality", "maximum"],
        help="Restaurierungs-Modus (Standard: restoration)",
    )
    parser.add_argument(
        "--out",
        metavar="WAV_DATEI",
        default=None,
        help="Restauriertes Audio als WAV speichern",
    )
    parser.add_argument(
        "--trace-json",
        metavar="JSON_DATEI",
        default=None,
        help="Vollständigen Trace als JSON speichern",
    )
    parser.add_argument(
        "--goals-only",
        action="store_true",
        help="Nur Goal-Matrix ausgeben (kein vollständiger Bericht)",
    )
    parser.add_argument(
        "--decisions-only",
        action="store_true",
        help="Nur Phasen-Entscheidungen ausgeben",
    )
    parser.add_argument(
        "--deltas-only",
        action="store_true",
        help="Nur Goal-Deltas (Anfang→Ende) ausgeben",
    )
    parser.add_argument(
        "--summary-json",
        action="store_true",
        help="Kurzübersicht als JSON auf stdout (für Scripting)",
    )
    parser.add_argument(
        "--worst-phases",
        type=int,
        default=0,
        metavar="N",
        help="N schlechteste Phasen (größte Goal-Regressionen) anzeigen",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Ausführliches Logging (inkl. Phase-Fortschritt)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Kein restauriertes Audio speichern (auch wenn --out angegeben)",
    )

    args = parser.parse_args(argv)

    _setup_workspace()
    denker_mode, goal_gate_mode, display_mode = _resolve_debug_modes(args.mode)

    # --- Audio laden ---
    try:
        audio, sr = _load_audio(args.audio)
    except Exception as e:
        print(f"\n✗ Audio-Fehler: {e}", file=sys.stderr)
        return 3

    # --- Pipeline ---
    try:
        _print_header(args.audio, display_mode)
        result = _run_restore(audio, sr, denker_mode, args.verbose)
    except Exception as e:
        print(f"\n✗ Pipeline-Fehler: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc(file=sys.stderr)
        return 2

    # --- Optionale Audio-Ausgabe ---
    if args.out and not args.no_audio:
        try:
            import soundfile as sf

            from backend.api.bridge import export_guard

            _audio_out = getattr(result, "audio", None)
            if _audio_out is not None:
                out_path = Path(args.out)
                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                try:
                    sf.write(tmp_path, export_guard(_audio_out), 48000)
                    tmp_path.replace(out_path)
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink(missing_ok=True)
                print(f"\n✓ Audio gespeichert: {args.out}", file=sys.stderr)
            else:
                print("\n⚠ Kein Audio in Result — --out ignoriert", file=sys.stderr)
        except Exception as e:
            print(f"\n⚠ Audio-Speichern fehlgeschlagen: {e}", file=sys.stderr)

    # --- Debug-Importe ---
    try:
        from backend.api.debug_api import (
            get_debug_summary,
            get_goal_fails,
            get_worst_phases,
        )
        from backend.core.pipeline_trace import (
            build_from_result,
            format_goal_deltas,
            format_phase_decisions,
        )
        from backend.core.pipeline_trace import (
            format_full_report as _fmt_full,
        )
        from backend.core.pipeline_trace import (
            format_goals_table as _fmt_goals,
        )
    except ImportError as e:
        print(f"\n✗ Debug-API Import fehlgeschlagen: {e}", file=sys.stderr)
        _print_summary(result)
        return 3

    # --- JSON-Summary-Modus ---
    if args.summary_json:
        summary = get_debug_summary(result)
        summary["goal_fails"] = get_goal_fails(result, goal_gate_mode)
        summary["worst_phases"] = (
            get_worst_phases(result, n=5) if args.worst_phases == 0 else get_worst_phases(result, n=args.worst_phases)
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1 if summary.get("goal_fails") else 0

    # --- Trace aufbauen ---
    trace = build_from_result(result)

    # --- Trace-JSON speichern ---
    if args.trace_json:
        try:
            with open(args.trace_json, "w", encoding="utf-8") as f:
                f.write(trace.to_json())
            print(f"\n✓ Trace-JSON gespeichert: {args.trace_json}", file=sys.stderr)
        except Exception as e:
            print(f"\n⚠ Trace-JSON speichern fehlgeschlagen: {e}", file=sys.stderr)

    # --- Ausgabe wählen ---
    if args.goals_only:
        print(_fmt_goals(trace))
    elif args.decisions_only:
        print(format_phase_decisions(trace))
    elif args.deltas_only:
        print(format_goal_deltas(trace))
    elif args.worst_phases > 0:
        worst = get_worst_phases(result, n=args.worst_phases)
        print(f"\nDie {args.worst_phases} Phasen mit den größten Goal-Regressionen:")
        print("-" * 60)
        for i, ph in enumerate(worst, 1):
            print(f"  {i}. {ph['phase_id']} — Gesamt-Regression: {ph['total_regression']:.4f}")
            for g, v in ph["regressions"].items():
                print(f"       {g}: {v:+.4f}")
    else:
        # Vollständiger Bericht
        print(_fmt_full(trace))

    # --- Goal-Fails prüfen (Exit-Code) ---
    goal_fails = get_goal_fails(result, goal_gate_mode)
    if goal_fails:
        print(f"\n⚠ {len(goal_fails)} Goal(s) unter Schwellwert:")
        for gf in goal_fails:
            print(f"  ✗ {gf['goal']}: {gf['value']:.3f} (Schwelle: {gf['threshold']:.2f}, Δ: {gf['delta']:+.3f})")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
