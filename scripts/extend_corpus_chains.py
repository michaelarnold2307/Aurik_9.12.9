#!/usr/bin/env python3
"""Corpus-Ketten-Erweiterung — deklarierte Tonträgerketten (§15.2, goldenes Hör-Set).

Zwei Schritte, beide deterministisch (§G5 (copilot-instructions.md)) und idempotent:

1. Deklaration: Jeder bestehende `damaged/`-Eintrag erhält `chain: [<material>]`
   — die dokumentierte Corpus-Semantik: „damaged/ = Defekte Original-Aufnahmen“
   (corpus/README.md), d. h. EINE Generation. Keine Audio-Datei wird verändert;
   bestehende checksum_sha256-Felder bleiben unangetastet.

2. Synthese: Mehr-Generationen-Items für die Depth-Klassen 2–4 werden aus
   bestehenden damaged-Quellen erzeugt (Kettenglieder als dokumentierte,
   synthetische DSP-Proxys — dieselbe Konvention wie scripts/generate_corpus.py,
   das z. B. „mp3_artifacts“ als Hiss-Proxy erzeugt). Jedes Item bekommt einen
   Manifest-Eintrag mit deklarierter `chain`, CC0-Lizenz und Checksumme.

Usage:
    python scripts/extend_corpus_chains.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
MATERIALS = ("shellac", "vinyl", "tape", "reel_tape", "cassette", "digital")

# Geplante Ketten für die Depth-Quoten (>=2 pro Klasse 2/3/4+):
# (Material, Kette ab Quelle, Seed)
_CHAINS: list[tuple[str, list[str], int]] = [
    ("vinyl", ["vinyl", "cassette"], 101),
    ("tape", ["tape", "reel_tape"], 102),
    ("digital", ["digital", "mp3_low"], 103),
    ("vinyl", ["vinyl", "reel_tape", "mp3_low"], 201),
    ("tape", ["tape", "cassette", "mp3_low"], 202),
    ("shellac", ["shellac", "vinyl", "reel_tape"], 203),
    ("vinyl", ["vinyl", "reel_tape", "cassette", "mp3_low"], 301),
    ("shellac", ["shellac", "vinyl", "reel_tape", "mp3_low"], 302),
]


def _lowpass(audio: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, cutoff, btype="low", fs=sr, output="sos")
    return sosfiltfilt(sos, audio, axis=0).astype(np.float32)


def _stage(role: str, audio: np.ndarray, sr: int, seed: int) -> np.ndarray:
    """Dokumentierter synthetischer Proxy für ein Kettenglied (deterministisch)."""
    rng = np.random.RandomState(seed)
    a = audio.astype(np.float32)
    if role == "reel_tape":
        a = _lowpass(a, sr, 12000.0)
        a = a + rng.normal(0.0, 0.004, a.shape).astype(np.float32)  # Hiss
        a = a * (1.0 + 0.004 * np.sin(2 * np.pi * 0.4 * np.arange(a.shape[0]) / sr)[:, None])  # Wow-Prox
    elif role == "cassette":
        a = _lowpass(a, sr, 10000.0)
        a = a + rng.normal(0.0, 0.006, a.shape).astype(np.float32)  # Hiss
        for _ in range(6):  # Dropouts
            start = int(rng.randint(0, max(1, a.shape[0] - sr // 40)))
            length = int(rng.randint(sr // 400, sr // 200))
            a[start : start + length] *= 0.1
    elif role == "mp3_low":
        a = _lowpass(a, sr, 11000.0)
        a = a + rng.normal(0.0, 0.003, a.shape).astype(np.float32)  # Artefakt-Prox
    elif role == "vinyl":
        a = a + rng.uniform(0.0, 0.004, a.shape).astype(np.float32)  # Surface-Noise
        for _ in range(int(0.03 * a.shape[0] / sr * 100)):  # Crackles
            pos = int(rng.randint(0, max(1, a.shape[0] - 1)))
            a[pos] += float(rng.uniform(0.01, 0.03))
    return np.clip(a, -1.0, 1.0).astype(np.float32)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _load_manifest(material: str) -> dict[str, Any]:
    import yaml

    mf = CORPUS / material / "manifest.yaml"
    if not mf.exists():
        return {"corpus_version": "1.0.0", "material": material, "entries": []}
    return yaml.safe_load(mf.read_text(encoding="utf-8")) or {}


def _save_manifest(material: str, data: dict[str, Any]) -> None:
    import yaml

    mf = CORPUS / material / "manifest.yaml"
    mf.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )


def _pick_source(material: str) -> tuple[Path, dict[str, Any]]:
    """Kleinsten beschädigten Eintrag als deterministische Quelle wählen."""
    entries = [
        e
        for e in _load_manifest(material).get("entries", [])
        if str(e.get("file", "")).startswith("damaged/") and not e.get("chain")
    ]
    if not entries:
        entries = [
            e for e in _load_manifest(material).get("entries", []) if str(e.get("file", "")).startswith("damaged/")
        ]
    if not entries:
        raise FileNotFoundError(f"kein damaged-Eintrag für Material {material}")
    entry = sorted(entries, key=lambda e: str(e.get("file")))[0]
    return CORPUS / material / str(entry["file"]), entry


def main() -> int:
    import soundfile as sf

    # ── Schritt 1: Deklarierte Ketten für bestehende damaged-Einträge ──
    declared_existing = 0
    for mat in MATERIALS:
        data = _load_manifest(mat)
        changed = False
        for e in data.get("entries", []):
            if str(e.get("file", "")).startswith("damaged/") and not e.get("chain"):
                e["chain"] = [mat]  # „Defekte Original-Aufnahmen“ = eine Generation
                changed = True
                declared_existing += 1
        if changed:
            _save_manifest(mat, data)
    print(f"Schritt 1: {declared_existing} bestehende damaged-Einträge mit chain deklariert")

    # ── Schritt 2: Mehr-Generationen-Items synthetisieren ──
    generated = 0
    for material, chain, seed in _CHAINS:
        try:
            src_path, src_entry = _pick_source(chain[0])
        except FileNotFoundError as exc:
            print(f"übersprungen {chain}: {exc}")
            continue
        audio, sr = sf.read(str(src_path), dtype="float32")
        if audio.ndim == 1:
            audio = audio[:, None]
        staged = audio
        for role in chain[1:]:
            staged = _stage(role, staged, sr, seed=seed + 10 * chain.index(role))
        out_name = f"{src_path.stem}_chain_{'_'.join(chain)}.wav"
        out_path = src_path.parent / out_name
        if out_path.exists():
            print(f"existiert bereits: {out_path.name}")
            continue
        sf.write(str(out_path), staged, sr)
        duration = float(len(staged) / sr)
        entry = {
            "file": f"damaged/{out_name}",
            "duration_s": round(duration, 2),
            "sample_rate": int(sr),
            "material": chain[0],
            "era_year": int(src_entry.get("era_year", 1965)),
            "genre": str(src_entry.get("genre", "unknown")),
            "defect_types": ["chain_artifacts"],
            "license": "CC0 (generated)",
            "source": "Synthetic (Aurik Corpus Generator) — chain synthesis (deterministic)",
            "source_attribution": "Synthetic Corpus Generator (CC0)",
            "checksum_sha256": _sha256(out_path),
            "chain": chain,
        }
        data = _load_manifest(chain[0])
        data.setdefault("entries", []).append(entry)
        _save_manifest(chain[0], data)
        generated += 1
        print(f"generiert: {out_name} (chain={chain})")
    print(f"Schritt 2: {generated} Mehr-Generationen-Items")
    return 0


if __name__ == "__main__":
    sys.exit(main())
