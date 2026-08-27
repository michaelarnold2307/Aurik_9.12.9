#!/usr/bin/env python3
"""Challenger-Round-Harness (Punkt 3): Incumbent vs. Kandidat auf dem goldenen Set.

prepare:  Validiert die Paarung (für jedes Golden-Set-Item müssen Incumbent- und
          Challenger-Ausgabe existieren) und baut ein reproduzierbares
          (seed-fixiertes) MUSHRA-Trial-Paket inkl. 3.5-kHz-Anchor
          (ITU-R BS.1534-3). Hidden Reference = degradierte Quelle des Items.
          Fehlender Anchor (z. B. soundfile nicht verfügbar) ist nicht-fatal
          und wird im Paket als Problem vermerkt.

decide:   Entscheidungsregel — der Kandidat wird übernommen (ADOPT), wenn
          (a) die Bootstrap-CI-Untergrenze von challenger - incumbent > 0 ist
              (signifikant besser am Ohr) und
          (b) der Kandidat die Non-Inferiority gegen den Anchor besteht
              (Punkt-1-Gate, scripts/non_inferiority_gate.py).
          Fehlende/unvollständige Urteile → BLOCKED (fail-closed).

Usage:
    python scripts/challenger_round.py prepare --golden audit/golden_listening_set.json \
        --incumbent-dir <dir> --challenger-dir <dir> --output <round-dir> [--seed 42]
    python scripts/challenger_round.py decide --verdicts <json> [--margin 5.0] [--min-listeners 10]

Exit-Codes (decide): 0 = ADOPT, 1 = REJECT, 2 = BLOCKED
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent


def _import_scripts_module(name: str) -> Any:
    """Importiert ein Schwester-Modul aus scripts/ (robust für direkte Tests)."""
    sys.path.insert(0, str(_ROOT))
    return __import__(name)


def _find(candidate_dir: Path, item_id: str) -> Path | None:
    if not Path(candidate_dir).is_dir():
        return None
    exact = Path(candidate_dir) / f"{item_id}.wav"
    if exact.exists():
        return exact
    matches = sorted(Path(candidate_dir).glob(f"*{item_id}*"))
    for m in matches:
        if m.suffix.lower() in (".wav", ".flac"):
            return m
    return None


def _compute_anchor_or_none(source: Path, output_dir: Path) -> Path | None:
    try:
        ps = _import_scripts_module("prepare_listening_study")
        return Path(ps._compute_anchor(source, output_dir))
    except Exception:
        return None


def prepare(
    golden: dict[str, Any],
    incumbent_dir: Path,
    challenger_dir: Path,
    output_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Baut das Trial-Paket; fehlende Dateien landen in problems (nicht-fatal)."""
    rng = random.Random(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials: list[dict[str, Any]] = []
    problems: list[str] = []
    for it in golden.get("items", []):
        iid = str(it.get("id", "?"))
        src = Path(str(it.get("path", "")))
        if not src.exists():
            problems.append(f"{iid}: Quelle fehlt ({src})")
            continue
        inc = _find(incumbent_dir, iid)
        cha = _find(challenger_dir, iid)
        if inc is None:
            problems.append(f"{iid}: Incumbent-Ausgabe fehlt in {incumbent_dir}")
        if cha is None:
            problems.append(f"{iid}: Challenger-Ausgabe fehlt in {challenger_dir}")
        if inc is None or cha is None:
            continue
        anchor = _compute_anchor_or_none(src, output_dir)
        if anchor is None:
            problems.append(f"{iid}: 3.5-kHz-Anchor konnte nicht erzeugt werden")
        conditions: dict[str, str] = {
            "hidden_ref": str(src),
            "incumbent": str(inc),
            "challenger": str(cha),
        }
        keys = list(conditions.keys())
        if anchor is not None:
            conditions["anchor"] = str(anchor)
            keys.append("anchor")
        order = keys[:]
        rng.shuffle(order)
        trials.append(
            {
                "trial_id": f"round_{seed}_{iid}",
                "item_id": iid,
                "material": it.get("material"),
                "depth": it.get("depth"),
                "conditions": conditions,
                "hidden_ref_key": "hidden_ref",
                "anchor_key": "anchor" if anchor is not None else None,
                "display_order": order,
            }
        )
    package = {
        "round_seed": seed,
        "items": len(trials),
        "problems": problems,
        "trials": trials,
    }
    (output_dir / "challenger_package.json").write_text(
        json.dumps(package, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return package


def decide(
    verdicts: dict[str, Any],
    margin: float = 5.0,
    min_listeners: int = 10,
) -> dict[str, Any]:
    """Entscheidungsregel: ADOPT nur bei signifikanter Verbesserung + Anchor-Non-Inferiority."""
    gate = _import_scripts_module("non_inferiority_gate")
    items = verdicts.get("items", [])
    report: dict[str, Any] = {"decision": "BLOCKED", "items": [], "reasons": []}
    if not items:
        report["reasons"].append("keine Items in den Verdicts")
        return 2, report
    for it in items:
        iid = str(it.get("item_id", "?"))
        scores = list(it.get("scores", []))
        item_report: dict[str, Any] = {"item_id": iid, "listeners": len(scores)}
        if len(scores) < min_listeners:
            item_report["decision"] = "BLOCKED"
            report["reasons"].append(f"{iid}: nur {len(scores)} Hörer (< {min_listeners})")
            report["items"].append(item_report)
            continue
        inc = np.array([float(s["incumbent"]) for s in scores])
        cha = np.array([float(s["challenger"]) for s in scores])
        lo, hi = gate.bootstrap_ci(cha - inc)
        item_report["mean_diff_vs_incumbent"] = float((cha - inc).mean())
        item_report["ci95_low"] = lo
        item_report["ci95_high"] = hi
        better = lo > 0.0
        # Non-Inferiority gegen den Anchor (Punkt-1-Gate)
        anchor_verdicts = {
            "items": [
                {
                    "item_id": iid,
                    "scores": [
                        {
                            "listener": s["listener"],
                            "anchor": s["anchor"],
                            "candidate": s["challenger"],
                        }
                        for s in scores
                    ],
                }
            ]
        }
        anchor_report = gate.evaluate(anchor_verdicts, margin=margin, min_listeners=min_listeners)
        anchor_ok = anchor_report["decision"] == "PASS"
        item_report["anchor_non_inferiority"] = anchor_report["decision"]
        if not better:
            item_report["decision"] = "REJECT"
            report["reasons"].append(f"{iid}: CI-Untergrenze {lo:.2f} <= 0 (keine signifikante Verbesserung)")
        elif not anchor_ok:
            item_report["decision"] = "REJECT"
            report["reasons"].append(f"{iid}: Non-Inferiority gegen Anchor nicht bestanden")
        else:
            item_report["decision"] = "ADOPT"
        report["items"].append(item_report)
    decisions = {r["decision"] for r in report["items"]}
    if "BLOCKED" in decisions:
        report["decision"] = "BLOCKED"
    elif "REJECT" in decisions:
        report["decision"] = "REJECT"
    else:
        report["decision"] = "ADOPT"
    return {"ADOPT": 0, "REJECT": 1, "BLOCKED": 2}[report["decision"]], report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Challenger-Round-Harness")
    sub = parser.add_subparsers(dest="command", required=True)
    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--golden", type=Path, required=True)
    p_prep.add_argument("--incumbent-dir", type=Path, required=True)
    p_prep.add_argument("--challenger-dir", type=Path, required=True)
    p_prep.add_argument("--output", type=Path, required=True)
    p_prep.add_argument("--seed", type=int, default=42)
    p_dec = sub.add_parser("decide")
    p_dec.add_argument("--verdicts", type=Path, required=True)
    p_dec.add_argument("--margin", type=float, default=5.0)
    p_dec.add_argument("--min-listeners", type=int, default=10)
    args = parser.parse_args(argv)

    if args.command == "prepare":
        golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
        package = prepare(golden, args.incumbent_dir, args.challenger_dir, args.output, args.seed)
        print(json.dumps(package, indent=2, ensure_ascii=False))
        return 0 if not package["problems"] else 1
    verdicts = json.loads(Path(args.verdicts).read_text(encoding="utf-8"))
    code, report = decide(verdicts, margin=args.margin, min_listeners=args.min_listeners)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
