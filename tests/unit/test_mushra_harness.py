"""Tests für scripts/mushra_harness.py — Auswertung (ITU-R BS.1534)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.mushra_harness import StimulusScore, analyze_answers


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "answers.csv"
    with open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["listener", "kind", "rating", "trial_label"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def _args(**kw) -> argparse.Namespace:
    base = {"gap": 8.0, "margin": 15.0}
    base.update(kw)
    return argparse.Namespace(**base)


class TestAnalyzeAnswers:
    def test_go_case(self, tmp_path: Path) -> None:
        rows = [
            {"listener": "L1", "kind": "hidden_ref", "rating": "95", "trial_label": "p1"},
            {"listener": "L1", "kind": "aurik", "rating": "88", "trial_label": "p1"},
            {"listener": "L1", "kind": "anchor", "rating": "40", "trial_label": "p1"},
        ]
        p = _write(tmp_path, rows)
        rc = analyze_answers(p, tmp_path, _args())
        assert rc == 0
        data = json.loads((tmp_path / "analysis_answers.json").read_text(encoding="utf-8"))
        assert data["go"] is True
        assert data["scores"]["p1"]["aurik"]["mean"] == 88.0
        assert data["listeners_valid"] == 1

    def test_no_go_and_listener_exclusion(self, tmp_path: Path) -> None:
        rows = [
            {"listener": "L1", "kind": "hidden_ref", "rating": "40", "trial_label": "p1"},
            {"listener": "L1", "kind": "aurik", "rating": "90", "trial_label": "p1"},
            {"listener": "L2", "kind": "hidden_ref", "rating": "92", "trial_label": "p1"},
            {"listener": "L2", "kind": "aurik", "rating": "75", "trial_label": "p1"},
            {"listener": "L2", "kind": "anchor", "rating": "35", "trial_label": "p1"},
        ]
        p = _write(tmp_path, rows)
        rc = analyze_answers(p, tmp_path, _args())
        assert rc == 1  # Aurik 75 < hidden_ref 92 - gap 8
        data = json.loads((tmp_path / "analysis_answers.json").read_text(encoding="utf-8"))
        assert "L1" in data["excluded"]
        assert data["scores"]["p1"]["aurik"]["mean"] == 75.0
        assert data["go"] is False

    def test_empty_input(self, tmp_path: Path) -> None:
        p = _write(tmp_path, [])
        assert analyze_answers(p, tmp_path, _args()) == 2


class TestStimulusScore:
    def test_fields(self) -> None:
        s = StimulusScore(kind="aurik", mean=25.0, ci95=3.1, n=4)
        assert s.kind == "aurik" and s.mean == 25.0 and s.n == 4
