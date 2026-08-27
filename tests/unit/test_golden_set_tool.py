"""Tests für scripts/golden_set_tool.py — Coverage-Quoten & fail-closed-Verhalten."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

import golden_set_tool as gst


def _make_item(tmp_path: Path, mid: str, material: str, depth: str) -> dict:
    f = tmp_path / f"{mid}.wav"
    f.write_bytes(b"RIFF-platzhalter")  # Inhalt egal — Check prüft nur Existenz
    return {
        "id": mid,
        "path": str(f),
        "material": material,
        "depth": depth,
        "restorability_score": 60,
        "classification_verified": True,
    }


def _full_items(tmp_path: Path) -> list[dict]:
    return [_make_item(tmp_path, f"{m}_{d}", m, d) for m in gst.MATERIALS for d in gst.DEPTH_CLASSES]


def test_check_fails_on_insufficient_coverage(tmp_path: Path) -> None:
    manifest = {"version": 1, "items": _full_items(tmp_path)[:2]}
    code, report = gst.check_manifest(manifest)
    assert code == 1
    assert report["decision"] == "FAIL"
    assert any("Material" in p for p in report["coverage"]["problems"])


def test_check_blocked_without_verdicts(tmp_path: Path) -> None:
    manifest = {"version": 1, "items": _full_items(tmp_path)}
    code, report = gst.check_manifest(manifest)
    assert code == 2
    assert report["decision"] == "BLOCKED"
    assert any("keine Hörurteile" in p for p in report["verdicts"]["problems"])


def test_check_blocked_with_too_few_listeners(tmp_path: Path) -> None:
    items = _full_items(tmp_path)
    manifest = {
        "version": 1,
        "items": items,
        "last_verdicts": {
            "listeners": 9,
            "items_covered": [i["id"] for i in items],
            "date": "2026-08-03",
        },
    }
    code, report = gst.check_manifest(manifest)
    assert code == 2
    assert any("nur 9 Hörer" in p for p in report["verdicts"]["problems"])


def test_check_passes_with_complete_verdicts(tmp_path: Path) -> None:
    items = _full_items(tmp_path)
    manifest = {
        "version": 1,
        "items": items,
        "last_verdicts": {
            "listeners": 10,
            "items_covered": [i["id"] for i in items],
            "date": "2026-08-03",
        },
    }
    code, report = gst.check_manifest(manifest)
    assert code == 0
    assert report["decision"] == "PASS"
    assert report["coverage"]["problems"] == []


def test_init_scan_detects_material_from_subdir(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "vinyl").mkdir(parents=True)
    (corpus / "digital").mkdir()
    (corpus / "vinyl" / "a.wav").write_bytes(b"x")
    (corpus / "digital" / "b.wav").write_bytes(b"x")
    items = gst.scan_corpus(corpus)
    mats = {i["material"] for i in items}
    assert mats == {"vinyl", "digital"}
    assert all(i["depth"] is None for i in items)


def test_scan_corpus_subdir_filter_excludes_clean_and_nonmaterials(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "vinyl" / "damaged").mkdir(parents=True)
    (corpus / "vinyl" / "clean").mkdir(parents=True)
    (corpus / "vocals").mkdir(parents=True)
    (corpus / "vinyl" / "damaged" / "a.wav").write_bytes(b"x")
    (corpus / "vinyl" / "clean" / "b.wav").write_bytes(b"x")
    (corpus / "vocals" / "c.wav").write_bytes(b"x")
    items = gst.scan_corpus(corpus, subdirs=("damaged",))
    assert [i["id"] for i in items] == ["vinyl_01"]
    assert items[0]["material"] == "vinyl"


def test_crosscheck_single_item(tmp_path: Path) -> None:
    """crosscheck misst beide Schätzer gegen die kuratierten Labels (Empfehlung 9)."""
    import json

    pytest.importorskip("joblib")
    art = _ROOT / "models" / "medium_shallow_v1.joblib"
    if not art.exists():
        pytest.skip("Artefakt fehlt — erst scripts/train_medium_classifier.py ausführen")
    golden = json.loads((_ROOT / "audit" / "golden_listening_set.json").read_text(encoding="utf-8"))
    it = golden["items"][0]
    manifest = {
        "items": [
            {k: it[k] for k in ("id", "path", "material", "depth", "era_year", "detected_material", "detected_depth")}
            | {"classification_verified": True}
        ]
    }
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(manifest), encoding="utf-8")
    report = gst.crosscheck(mp)
    assert report["n_verified"] == 1
    assert set(report["medium_detector_agreement"]) == {"material", "depth"}
    assert set(report["shallow_cv_accuracy"]) == {"material", "depth"}
    assert "shallow_train_agreement" in report


def test_scan_corpus_uses_manifest_truth_and_declared_chain(tmp_path: Path) -> None:
    """§15.2: manifest.yaml ist die kuratierte Wahrheit (Material, era, chain)."""
    corpus = tmp_path / "corpus"
    (corpus / "vinyl" / "damaged").mkdir(parents=True)
    (corpus / "vinyl" / "damaged" / "a.wav").write_bytes(b"x")
    (corpus / "vinyl" / "manifest.yaml").write_text(
        "corpus_version: 1.0.0\nmaterial: vinyl\nentries:\n"
        "- file: damaged/a.wav\n  material: vinyl\n  era_year: 1965\n"
        "  genre: rock\n  license: CC0\n  chain: [vinyl, cassette, mp3_low]\n",
        encoding="utf-8",
    )
    items = gst.scan_corpus(corpus, subdirs=("damaged",))
    assert len(items) == 1
    it = items[0]
    assert it["id"] == "a"
    assert it["material"] == "vinyl"
    assert it["era_year"] == 1965
    assert it["genre"] == "rock"
    assert it["declared_chain"] == ["vinyl", "cassette", "mp3_low"]
    # Deklarierte Kette setzt die Depth autoritativ — ohne Detektor-Votum.
    probs = gst.verify_item(it, verified_by="tester", restorability=60)
    assert probs == []
    assert it["depth"] == "3"
    assert it["classification_verified"] is True
    assert it["verified_by"] == "tester"
