"""Tests für scripts/non_inferiority_gate.py — Bootstrap-CI & fail-closed."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

import non_inferiority_gate as nig


def _verdicts(offset: float, n: int = 10) -> dict:
    rng = np.random.RandomState(7)
    return {
        "items": [
            {
                "item_id": "t1",
                "scores": [
                    {
                        "listener": f"P{i:02d}",
                        "anchor": float(60 + rng.randn() * 3),
                        "candidate": float(60 + offset + rng.randn() * 3),
                    }
                    for i in range(n)
                ],
            }
        ]
    }


def test_blocked_on_empty_verdicts() -> None:
    assert nig.evaluate({})["decision"] == "BLOCKED"


def test_blocked_on_too_few_listeners() -> None:
    report = nig.evaluate(_verdicts(8.0, n=9), margin=5.0, min_listeners=10)
    assert report["decision"] == "BLOCKED"


def test_pass_with_clear_margin() -> None:
    report = nig.evaluate(_verdicts(8.0), margin=5.0, min_listeners=10)
    assert report["decision"] == "PASS"
    assert report["items"][0]["ci95_low"] > -5.0


def test_fail_when_worse_than_margin() -> None:
    report = nig.evaluate(_verdicts(-10.0), margin=5.0, min_listeners=10)
    assert report["decision"] == "FAIL"
    assert report["items"][0]["ci95_low"] <= -5.0


def test_deterministic() -> None:
    v = _verdicts(3.0)
    assert nig.evaluate(v) == nig.evaluate(v)


def test_bootstrap_ci_needs_two_observations() -> None:
    with pytest.raises(ValueError):
        nig.bootstrap_ci(np.array([1.0]))
