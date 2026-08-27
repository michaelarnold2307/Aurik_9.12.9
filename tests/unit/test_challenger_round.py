"""Tests für scripts/challenger_round.py — Paarung, Paketbau, Entscheidungsregel."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

import challenger_round as cr


def _golden(tmp_path: Path, ids: list[str]) -> dict:
    items = []
    for i in ids:
        src = tmp_path / f"{i}.wav"
        src.write_bytes(b"x")
        items.append({"id": i, "path": str(src), "material": "vinyl", "depth": "2"})
    return {"version": 1, "items": items}


def test_prepare_builds_trials_for_all_items(tmp_path: Path) -> None:
    golden = _golden(tmp_path, ["v1", "v2"])
    inc = tmp_path / "inc"
    cha = tmp_path / "cha"
    inc.mkdir()
    cha.mkdir()
    (inc / "v1.wav").write_bytes(b"i")
    (inc / "v2.wav").write_bytes(b"i")
    (cha / "v1.wav").write_bytes(b"c")
    (cha / "v2.wav").write_bytes(b"c")
    pkg = cr.prepare(golden, inc, cha, tmp_path / "round", seed=42)
    assert len(pkg["trials"]) == 2
    for t in pkg["trials"]:
        assert set(t["conditions"]) >= {"hidden_ref", "incumbent", "challenger"}
        assert len(t["display_order"]) == len(t["conditions"])
    # seed-fixiert reproduzierbar
    pkg2 = cr.prepare(golden, inc, cha, tmp_path / "round2", seed=42)
    assert pkg["trials"][0]["display_order"] == pkg2["trials"][0]["display_order"]


def test_prepare_reports_missing_challenger(tmp_path: Path) -> None:
    golden = _golden(tmp_path, ["v1"])
    inc = tmp_path / "inc"
    cha = tmp_path / "cha"
    inc.mkdir()
    cha.mkdir()
    (inc / "v1.wav").write_bytes(b"i")
    pkg = cr.prepare(golden, inc, cha, tmp_path / "round", seed=42)
    assert pkg["trials"] == []
    assert any("challenger" in p.lower() for p in pkg["problems"])


def _round_verdicts(win: float, n: int = 10) -> dict:
    rng = np.random.RandomState(11)
    return {
        "items": [
            {
                "item_id": "v1",
                "scores": [
                    {
                        "listener": f"P{i:02d}",
                        "incumbent": float(60 + rng.randn() * 3),
                        "challenger": float(60 + win + rng.randn() * 3),
                        "anchor": float(55 + rng.randn() * 3),
                    }
                    for i in range(n)
                ],
            }
        ]
    }


def test_decide_adopts_on_clear_win() -> None:
    code, report = cr.decide(_round_verdicts(6.0), margin=5.0, min_listeners=10)
    assert code == 0
    assert report["decision"] == "ADOPT"


def test_decide_rejects_on_loss() -> None:
    code, report = cr.decide(_round_verdicts(-6.0), margin=5.0, min_listeners=10)
    assert code == 1
    assert report["decision"] == "REJECT"


def test_decide_blocked_without_enough_listeners() -> None:
    code, report = cr.decide(_round_verdicts(6.0, n=9), margin=5.0, min_listeners=10)
    assert code == 2
    assert report["decision"] == "BLOCKED"
