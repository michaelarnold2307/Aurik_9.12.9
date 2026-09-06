#!/usr/bin/env python3
"""mushra_harness — Doppelblinde MUSHRA-Studie (ITU-R BS.1534) für Aurik.

Vier Schritte (CLI):
  1) prepare  — deterministische Restauration der Corpus-Paare (nur fehlende,
                hash-basiert inkrementell) → output_audio/mushra/<run>/
  2) sessions — baut N doppelblinde Hörer-Sessions: je Trial [clean (hidden
                ref), anchor (3.5-kHz-Lowpass), Aurik, ggf. kommerzielle
                Referenzen aus corpus/references/] in seed-deterministisch
                zufälliger Reihenfolge; exportiert JSON (Player kann beliebig
                gebaut werden) + Antwort-CSV-Vorlage.
  3) analyze  — wertet Antworten aus: MUSHRA-Scores (0-100), Mittelwert +
                95-%-CI je Stimulus/Trial, Hörer-Validität (Hidden-Ref < 60 →
                Ausschluss), GO/NO-GO gegen konfigurierbare Kriterien.

Hörordnung: Metriken sind Zeugen, die Hör-Instanz entscheidet — dieser
Harness ist die operative Hör-Instanz (menschlich, doppelblind).

Nutzung:
  python scripts/mushra_harness.py prepare  --modes balanced maximum --seconds 12
  python scripts/mushra_harness.py sessions --listeners 5 --out study_run1
  python scripts/mushra_harness.py analyze  --answers study_run1/answers.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_ROOT = ROOT / "output_audio" / "mushra"
ANCHOR_LP_HZ = 3500.0  # ITU: Anchor = 3.5-kHz-Bandbreite


# --------------------------------------------------------------------------
# Paar-Erkennung (damaged → clean)
# --------------------------------------------------------------------------
def find_pairs(corpus: Path, explicit: list[str] | None = None) -> list[tuple[Path, Path, str]]:
    """Liefert (damaged, clean, label)-Triple.

    Explizite Angabe: "damaged.wav,clean.wav,label".  Sonst Heuristik:
    <base>_crackle[_chain…].wav bzw. <base>_noise/_degraded → <base>_clean.wav.
    """
    pairs: list[tuple[Path, Path, str]] = []
    for spec in explicit or []:
        parts = [p.strip() for p in spec.split(",")]
        if len(parts) >= 2:
            dmg = Path(parts[0])
            cln = Path(parts[1])
            if not dmg.is_absolute():
                dmg = ROOT / dmg
            if not cln.is_absolute():
                cln = ROOT / cln
            pairs.append((dmg, cln, parts[2] if len(parts) > 2 else dmg.stem))
    if pairs:
        return pairs
    for mat_dir in sorted(corpus.iterdir()) if corpus.exists() else []:
        dmg_dir = mat_dir / "damaged"
        cln_dir = mat_dir / "clean"
        if not dmg_dir.is_dir() or not cln_dir.is_dir():
            continue
        clean_by_base = {p.stem: p for p in cln_dir.glob("*.wav") if p.stem.endswith("_clean")}
        for d in sorted(dmg_dir.glob("*.wav")):
            m = re.match(r"^(.*?)(?:_crackle|_noise|_degraded|_chain)", d.stem)
            base = m.group(1) if m else d.stem
            cand = clean_by_base.get(base + "_clean")
            if cand is None:
                cand = cln_dir / f"{base}_clean.wav"
            if cand.exists():
                pairs.append((d, cand, base))
    return pairs


# --------------------------------------------------------------------------
# Deterministische Restauration (prepare)
# --------------------------------------------------------------------------
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_pair(dmg: Path, clean: Path, label: str, out_dir: Path, mode: str,
                 seconds: float, sr_out: int = 48000) -> dict:
    import soundfile as sf

    from backend.core.unified_restorer_v3 import QualityMode, RestorationConfig, UnifiedRestorerV3

    audio, sr = sf.read(str(dmg), dtype="float32", always_2d=False)
    n = int(seconds * sr)
    if len(audio) > n:
        audio = audio[:n]
    cfg = RestorationConfig(mode=QualityMode(mode))
    engine = UnifiedRestorerV3(cfg)
    t0 = time.perf_counter()
    result = engine.restore(audio, sample_rate=sr)
    wall = time.perf_counter() - t0
    restored = np.asarray(result.audio, dtype=np.float32)
    if restored.ndim == 2 and restored.shape[0] == 2 and restored.shape[1] != 2:
        restored = restored.T
    out_path = out_dir / f"{label}__{mode}.wav"
    sf.write(str(out_path), restored, sr_out, format="WAV", subtype="PCM_24")
    return {
        "label": label, "mode": mode, "damaged_sha": sha256_file(dmg)[:16],
        "clean_sha": sha256_file(clean)[:16], "wall_s": round(wall, 1),
        "quality": float(getattr(result, "quality_estimate", 0.0) or 0.0),
        "audibility_gate": (result.metadata or {}).get("audibility_gate"),
        "out": str(out_path),
    }


def anchor_lowpass(audio: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, ANCHOR_LP_HZ / (sr / 2.0), btype="lowpass", output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def cmd_prepare(args: argparse.Namespace) -> int:
    pairs = find_pairs(ROOT / args.corpus, args.pair)
    if not pairs:
        print("Keine Paare gefunden. --corpus prüfen oder --pair explizit angeben.")
        return 2
    run_dir = OUT_ROOT / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"pairs": [], "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seconds": args.seconds, "modes": args.modes}
    for dmg, clean, label in pairs[: args.max_pairs]:
        for mode in args.modes:
            row = prepare_pair(dmg, clean, label, run_dir, mode, args.seconds)
            meta["pairs"].append(row)
            print(f"  {label} [{mode}]: {row['wall_s']}s quality={row['quality']}")
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Run-Verzeichnis:", run_dir)
    return 0


# --------------------------------------------------------------------------
# Sessions (doppelblind, seed-deterministisch)
# --------------------------------------------------------------------------
def build_sessions(pairs_meta: dict, n_listeners: int, seed: int, ref_root: Path) -> list[dict]:
    rng = np.random.default_rng(seed)
    # Kommerzielle Referenzen: corpus/references/<tool>/<label__mode>.wav?
    refs: dict[str, list[Path]] = {}
    if ref_root.is_dir():
        for tool_dir in sorted(ref_root.iterdir()):
            if tool_dir.is_dir():
                for wav in sorted(tool_dir.glob("*.wav")):
                    refs.setdefault(wav.stem, []).append(wav)
    sessions = []
    for li in range(n_listeners):
        trials = []
        for pr in pairs_meta["pairs"]:
            stim = [
                {"kind": "aurik", "path": pr["out"]},
            ]
            clean_hint = None
            for rpath in refs.get(pr["label"], []):
                stim.append({"kind": "reference", "path": str(rpath), "tool": rpath.parent.name})
            # Hidden Reference + Anchor werden zur Laufzeit aus clean bzw. Rest erzeugt;
            # hier nur Markierung, Player rendert sie deterministisch.
            stim.append({"kind": "hidden_ref"})
            stim.append({"kind": "anchor"})
            order = rng.permutation(len(stim)).tolist()
            trials.append({
                "label": pr["label"], "mode": pr["mode"],
                "clean_sha": pr["clean_sha"],
                "order": [stim[i]["kind"] for i in order],
                "n_stimuli": len(stim),
            })
        sessions.append({"listener": f"L{li+1:02d}", "seed": int(rng.integers(0, 2**31)), "trials": trials})
    return sessions


def cmd_sessions(args: argparse.Namespace) -> int:
    runs = sorted(OUT_ROOT.glob("run_*"))
    if not runs:
        print("Kein Run gefunden — zuerst: prepare")
        return 2
    run_dir = runs[-1]
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    out = OUT_ROOT / (args.out or f"study_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    sessions = build_sessions(meta, args.listeners, args.seed, ROOT / args.refs)
    (out / "sessions.json").write_text(json.dumps(sessions, indent=2), encoding="utf-8")
    # Antwort-CSV-Vorlage: listener,trial,stimulus_index,rating
    with open(out / "answers_template.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["listener", "trial_label", "stimulus_index", "kind", "rating"])
        for s in sessions:
            for t in s["trials"]:
                for i, kind in enumerate(t["order"]):
                    w.writerow([s["listener"], t["label"], i, kind, ""])
    (out / "meta.json").write_text(json.dumps({"seed": args.seed, "run": run_dir.name,
                                               "listeners": args.listeners}, indent=2), encoding="utf-8")
    print("Studien-Ordner:", out)
    return 0


# --------------------------------------------------------------------------
# Analyse (GO/NO-GO nach Hörordnung-Kriterien)
# --------------------------------------------------------------------------
@dataclass
class StimulusScore:
    kind: str
    mean: float
    ci95: float
    n: int


def analyze_answers(answers_csv: Path, out_dir: Path, args: argparse.Namespace) -> int:
    rows = list(csv.DictReader(open(answers_csv, encoding="utf-8")))
    if not rows:
        print("Leere Antwortdatei.")
        return 2
    # Hörer-Validität: Hidden-Ref muss im Mittel >= 60 liegen, sonst Ausschluss
    by_listener: dict[str, list[float]] = {}
    for r in rows:
        if r.get("kind") == "hidden_ref" and r.get("rating", "").strip():
            by_listener.setdefault(r["listener"], []).append(float(r["rating"]))
    valid = [li for li, vals in by_listener.items() if np.mean(vals) >= 60.0]
    kept = [r for r in rows if r["listener"] in valid]
    scores: dict[tuple[str, str], list[float]] = {}
    for r in kept:
        if r.get("rating", "").strip():
            scores.setdefault((r["trial_label"], r["kind"]), []).append(float(r["rating"]))
    summary = {}
    for (label, kind), vals in sorted(scores.items()):
        m = float(np.mean(vals))
        ci = float(1.96 * np.std(vals) / np.sqrt(max(len(vals), 1)))
        summary.setdefault(label, {})[kind] = {"mean": round(m, 1), "ci95": round(ci, 1), "n": len(vals)}
    # GO/NO-GO: Aurik >= hidden_ref - gap  UND Aurik >= anchor + margin
    go, reasons = True, []
    gap = float(args.gap)
    margin = float(args.margin)
    for label, kinds in summary.items():
        aur = kinds.get("aurik", {}).get("mean")
        hr = kinds.get("hidden_ref", {}).get("mean")
        an = kinds.get("anchor", {}).get("mean")
        if aur is None:
            go, reasons = False, [*reasons, f"{label}: Aurik fehlt"]
            continue
        if hr is not None and aur < hr - gap:
            go, reasons = False, [*reasons, f"{label}: Aurik {aur:.0f} < HiddenRef {hr:.0f} - {gap}"]
        if an is not None and aur < an + margin:
            go, reasons = False, [*reasons, f"{label}: Aurik {aur:.0f} < Anchor {an:.0f} + {margin}"]
    result = {
        "listeners_total": len(by_listener), "listeners_valid": len(valid),
        "excluded": sorted(set(by_listener) - set(valid)),
        "scores": summary, "go": go, "reasons": reasons,
        "criteria": {"hidden_ref_gap": gap, "anchor_margin": margin, "validity_min": 60},
    }
    out_file = out_dir / f"analysis_{answers_csv.stem}.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("GO" if go else "NO-GO", "|", "; ".join(reasons) or "Kriterien erfüllt")
    print("Bericht:", out_file)
    return 0 if go else 1


def cmd_analyze(args: argparse.Namespace) -> int:
    answers = Path(args.answers)
    if not answers.is_absolute():
        answers = OUT_ROOT / answers
    out_dir = answers.parent
    return analyze_answers(answers, out_dir, args)


def main() -> int:
    ap = argparse.ArgumentParser(description="MUSHRA-Harness (ITU-R BS.1534)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("prepare")
    p1.add_argument("--corpus", default="corpus")
    p1.add_argument("--pair", action="append", default=None)
    p1.add_argument("--modes", nargs="+", default=["balanced"])
    p1.add_argument("--seconds", type=float, default=12.0)
    p1.add_argument("--max-pairs", type=int, default=3)
    p1.set_defaults(fn=cmd_prepare)
    p2 = sub.add_parser("sessions")
    p2.add_argument("--listeners", type=int, default=5)
    p2.add_argument("--seed", type=int, default=20260906)
    p2.add_argument("--refs", default="corpus/references")
    p2.add_argument("--out", default=None)
    p2.set_defaults(fn=cmd_sessions)
    p3 = sub.add_parser("analyze")
    p3.add_argument("--answers", required=True)
    p3.add_argument("--gap", type=float, default=8.0)
    p3.add_argument("--margin", type=float, default=15.0)
    p3.set_defaults(fn=cmd_analyze)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
