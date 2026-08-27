#!/usr/bin/env python3
"""MUSHRA-Studienpaket für das goldene Hör-Set (Empfehlung 10: die Ohr-Schleife).

Erzeugt pro Golden-Set-Item:
    - hidden_ref  = degradierte Quelle des Items
    - anchor      = 3.5-kHz-Tiefpass (ITU-R BS.1534-3, via prepare_listening_study)
    - candidate   = Platzhalter-Pfad, den die Hörrunde mit der zu bewertenden
                    Restaurierung füllt (z. B. Release-Kandidat oder Challenger)
und schreibt ein deterministisch gemischtes Trial-Paket plus eine
Verdicts-Vorlage im Schema von scripts/non_inferiority_gate.py.

Usage:
    python scripts/package_golden_study.py [--out audit/listening_study/round_<datum>]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_OUT = ROOT / "audit" / "listening_study" / f"round_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
SEED = 42


def _compute_anchor(source: Path, out_dir: Path) -> Path:
    from prepare_listening_study import _compute_anchor

    return Path(_compute_anchor(source, out_dir))


def build_package(golden: dict[str, Any], out_dir: Path, seed: int = SEED) -> dict[str, Any]:
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    anchors_dir = out_dir / "anchors"
    anchors_dir.mkdir(parents=True, exist_ok=True)
    trials: list[dict[str, Any]] = []
    problems: list[str] = []
    for it in golden.get("items", []):
        iid = str(it.get("id"))
        src = Path(str(it.get("path", "")))
        if not src.exists():
            problems.append(f"{iid}: Quelle fehlt")
            continue
        try:
            anchor = _compute_anchor(src, anchors_dir)
        except Exception:
            anchor = None
            problems.append(f"{iid}: Anchor-Erzeugung fehlgeschlagen")
        conditions: dict[str, str] = {
            "hidden_ref": str(src),
            "candidate": str(out_dir / "candidates" / f"{iid}_candidate.wav"),
        }
        keys = list(conditions)
        if anchor is not None:
            conditions["anchor"] = str(anchor)
            keys.append("anchor")
        order = keys[:]
        rng.shuffle(order)
        trials.append(
            {
                "trial_id": f"golden_{iid}",
                "item_id": iid,
                "material": it.get("material"),
                "depth": it.get("depth"),
                "conditions": conditions,
                "hidden_ref_key": "hidden_ref",
                "anchor_key": "anchor" if anchor is not None else None,
                "display_order": order,
            }
        )
    package = {"round_seed": seed, "items": len(trials), "problems": problems, "trials": trials}
    (out_dir / "study_package.json").write_text(
        json.dumps(package, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # Verdicts-Vorlage im Schema des Non-Inferiority-Gates (auszufüllen nach Hörrunde)
    template = {
        "protocol": "MUSHRA ITU-R BS.1534-3",
        "margin_points": 5.0,
        "min_listeners": 10,
        "items": [
            {
                "item_id": t["item_id"],
                "scores": [{"listener": "P01", "anchor": 0.0, "candidate": 0.0}],
            }
            for t in trials
        ],
    }
    (out_dir / "verdict_template.json").write_text(
        json.dumps(template, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return package


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MUSHRA-Studienpaket für das goldene Hör-Set")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    golden = json.loads((ROOT / "audit" / "golden_listening_set.json").read_text(encoding="utf-8"))
    package = build_package(golden, args.out)
    print(f"Studienpaket: {args.out}")
    print(f"Trials: {package['items']}, Probleme: {len(package['problems'])}")
    for p in package["problems"]:
        print(f"  {p}")
    return 0 if not package["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
