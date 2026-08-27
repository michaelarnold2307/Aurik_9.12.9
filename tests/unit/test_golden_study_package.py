"""Tests für scripts/package_golden_study.py — MUSHRA-Paket für das goldene Set."""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

import package_golden_study as pgs


def _make_wav(path: Path, seconds: float = 1.0, sr: int = 48000) -> None:
    t = np.arange(int(seconds * sr)) / sr
    audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _golden(tmp_path: Path, ids: list[str]) -> dict:
    items = []
    for i in ids:
        src = tmp_path / f"{i}.wav"
        _make_wav(src)
        items.append({"id": i, "path": str(src), "material": "vinyl", "depth": "1"})
    return {"items": items}


def test_build_package_covers_all_items(tmp_path: Path) -> None:
    golden = _golden(tmp_path, ["a", "b"])
    pkg = pgs.build_package(golden, tmp_path / "round", seed=42)
    assert len(pkg["trials"]) == 2
    for t in pkg["trials"]:
        assert set(t["conditions"]) >= {"hidden_ref", "candidate", "anchor"}
        assert t["hidden_ref_key"] == "hidden_ref"
        assert len(t["display_order"]) == len(t["conditions"])
    assert (tmp_path / "round" / "study_package.json").exists()


def test_build_package_deterministic_order(tmp_path: Path) -> None:
    golden = _golden(tmp_path, ["a", "b"])
    p1 = pgs.build_package(golden, tmp_path / "r1", seed=42)
    p2 = pgs.build_package(golden, tmp_path / "r2", seed=42)
    assert p1["trials"][0]["display_order"] == p2["trials"][0]["display_order"]


def test_verdict_template_matches_gate_schema(tmp_path: Path) -> None:
    golden = _golden(tmp_path, ["a"])
    pgs.build_package(golden, tmp_path / "round", seed=42)
    tpl = json.loads((tmp_path / "round" / "verdict_template.json").read_text(encoding="utf-8"))
    assert tpl["min_listeners"] == 10
    assert tpl["items"][0]["item_id"] == "a"
    assert {"listener", "anchor", "candidate"} <= set(tpl["items"][0]["scores"][0])
