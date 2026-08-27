#!/usr/bin/env python3
"""Non-Inferiority-Gate auf MUSHRA-Hörurteilen (goldenes Hör-Set, Punkt 1).

Fail-closed: Ohne vollständige Hörurteile (>= min_listeners pro Item) ist der
Status BLOCKED. Mit Urteilen: gepaarte Bootstrap-Konfidenzintervalle (fixer
Seed → deterministisch, §G5 (copilot-instructions.md) der Differenz candidate - anchor pro Item.
Ein Item besteht, wenn die untere 95%-CI-Grenze über -margin liegt.
Der Gate besteht nur, wenn alle Items bestehen.

Verbschema (JSON):
{
  "protocol": "MUSHRA ITU-R BS.1534-3",
  "items": [
    {"item_id": "vinyl_01", "scores": [
        {"listener": "P01", "anchor": 62.0, "candidate": 70.5}, ...]}
  ]
}

Usage:
    python scripts/non_inferiority_gate.py --verdicts <json> [--margin 5.0] [--min-listeners 10]

Exit-Codes: 0 = PASS, 1 = FAIL, 2 = BLOCKED
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SEED = 42
_N_BOOT = 5000


def bootstrap_ci(
    diffs: np.ndarray,
    n_boot: int = _N_BOOT,
    seed: int = _SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile-Bootstrap-CI der Mittelwert-Differenz (fixed seed)."""
    rng = np.random.RandomState(seed)
    n = int(diffs.size)
    if n < 2:
        raise ValueError("mindestens 2 Beobachtungen für Bootstrap nötig")
    means = np.empty(n_boot, dtype=np.float64)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        means[_] = float(diffs[idx].mean())
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def evaluate(
    verdicts: dict[str, Any],
    margin: float = 5.0,
    min_listeners: int = 10,
) -> dict[str, Any]:
    """Non-Inferiority: candidate darf höchstens margin MUSHRA-Punkte schlechter sein.

    Returns report with decision in {"PASS", "FAIL", "BLOCKED"}.
    """
    items = verdicts.get("items", [])
    report: dict[str, Any] = {"decision": "BLOCKED", "items": [], "reasons": []}
    if not items:
        report["reasons"].append("keine Items in den Verdicts")
        return report
    for it in items:
        iid = str(it.get("item_id", "?"))
        scores = list(it.get("scores", []))
        item_report: dict[str, Any] = {"item_id": iid, "listeners": len(scores)}
        if len(scores) < min_listeners:
            item_report["decision"] = "BLOCKED"
            report["reasons"].append(f"{iid}: nur {len(scores)} Hörer (< {min_listeners})")
            report["items"].append(item_report)
            continue
        anchor = np.array([float(s["anchor"]) for s in scores])
        candidate = np.array([float(s["candidate"]) for s in scores])
        diffs = candidate - anchor
        lo, hi = bootstrap_ci(diffs)
        item_report["mean_diff"] = float(diffs.mean())
        item_report["ci95_low"] = lo
        item_report["ci95_high"] = hi
        item_report["decision"] = "PASS" if lo > -margin else "FAIL"
        if item_report["decision"] == "FAIL":
            report["reasons"].append(f"{iid}: CI-Untergrenze {lo:.2f} <= -{margin}")
        report["items"].append(item_report)
    decisions = {r["decision"] for r in report["items"]}
    if "BLOCKED" in decisions:
        report["decision"] = "BLOCKED"
    elif "FAIL" in decisions:
        report["decision"] = "FAIL"
    else:
        report["decision"] = "PASS"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Non-Inferiority-Gate (MUSHRA, fail-closed)")
    parser.add_argument("--verdicts", type=Path, required=True, help="Hörurteile-JSON")
    parser.add_argument("--margin", type=float, default=5.0, help="Non-Inferiority-Marge (Punkte)")
    parser.add_argument("--min-listeners", type=int, default=10)
    args = parser.parse_args(argv)

    verdicts = json.loads(Path(args.verdicts).read_text(encoding="utf-8"))
    report = evaluate(verdicts, margin=args.margin, min_listeners=args.min_listeners)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return {"PASS": 0, "FAIL": 1, "BLOCKED": 2}[report["decision"]]


if __name__ == "__main__":
    sys.exit(main())
