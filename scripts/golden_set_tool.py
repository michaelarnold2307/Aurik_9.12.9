#!/usr/bin/env python3
"""Golden Listening Set — Manifest- und Coverage-Werkzeug (Ohr-Messapparatur).

Das goldene Hör-Set ist der fixe Corpus, an dem jede Release-Entscheidung am
menschlichen Ohr validiert wird (docs/guides/GOLDEN_LISTENING_SET.md).
Fail-closed: Ohne vollständige MUSHRA-Hörurteile ist der Gate-Status BLOCKED —
Proxy-Metriken (PMGG, MERT-MUSHRA) können den Gate niemals öffnen.

Usage:
    python scripts/golden_set_tool.py init --corpus <dir> [--classify] [--subdir damaged]
    python scripts/golden_set_tool.py check [--manifest <path>]
    python scripts/golden_set_tool.py verify --item <id>|--all --verified-by NAME [--accept-detected]
    python scripts/golden_set_tool.py status [--manifest <path>]

Exit-Codes: 0 = PASS, 1 = FAIL (Format/Coverage), 2 = BLOCKED (keine Hörurteile)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))  # backend-Imports bei direktem Script-Lauf
DEFAULT_MANIFEST = _ROOT / "audit" / "golden_listening_set.json"

MATERIALS = ("vinyl", "tape", "shellac", "digital", "cassette", "reel_tape")
DEPTH_CLASSES = ("1", "2", "3", "4+")
MIN_ITEMS_TOTAL = 12
MIN_PER_MATERIAL = 2
MIN_PER_DEPTH = 2
MIN_LISTENERS = 10
AUDIO_SUFFIXES = (".wav", ".flac")


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "items": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


_EXCLUDED_DIRS = ("aurik_processed", "processed", "restored", "reference")


def _load_material_manifests(corpus: Path) -> dict[str, dict[str, Any]]:
    """Lädt manifest.yaml je Material-Verzeichnis — die kuratierte Corpus-Wahrheit.

    §15.2 (corpus/README.md): Jedes Material-Verzeichnis führt eine
    manifest.yaml mit file, material, era_year, genre, defect_types, license.
    Index nach absolutem Datei-Pfad für O(1)-Lookup im Scan.
    """
    index: dict[str, dict[str, Any]] = {}
    try:
        import yaml
    except ImportError:
        return index
    for mat in MATERIALS:
        mdir = corpus / mat
        mf = mdir / "manifest.yaml"
        if not mf.exists():
            continue
        try:
            data = yaml.safe_load(mf.read_text(encoding="utf-8")) or {}
        except Exception:
            logger.debug("golden_set_tool: manifest.yaml unlesbar: %s", mf, exc_info=True)
            continue
        for entry in data.get("entries", []) or []:
            rel = str(entry.get("file", "")).strip()
            if not rel:
                continue
            index[str((mdir / rel).resolve())] = entry
    return index


def scan_corpus(corpus: Path, subdirs: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    """Sammelt Audio-Dateien; Material/Ära/Genre aus manifest.yaml (kuratierte Wahrheit).

    Regeln (Zirkularitätsschutz):
    - Nur Transfer-Materialien (rel.parts[0] in MATERIALS) — vocals/reverb
      etc. gehören nicht ins goldene Set.
    - Verarbeitete/Restaurierte Ausgaben (aurik_processed, restored/,
      ...restored.wav) werden ausgeschlossen.
    - subdirs begrenzt den Scan auf bestimmte Unterverzeichnisse
      (z. B. ("damaged",)) — degradierte Quellen, nie clean/restored.
    - Ohne manifest.yaml fällt der Scan auf Verzeichnis-Inferenz zurück.
    """
    manifests = _load_material_manifests(corpus)
    files: set[str] = set()
    for suf in AUDIO_SUFFIXES:
        for p in corpus.rglob(f"*{suf}"):
            rel = p.relative_to(corpus)
            if not rel.parts or rel.parts[0].lower() not in MATERIALS:
                continue
            if any(part in _EXCLUDED_DIRS for part in rel.parts):
                continue
            if "restored" in p.stem.lower():
                continue
            if subdirs and not any(part in subdirs for part in rel.parts):
                continue
            files.add(str(p))
    items: list[dict[str, Any]] = []
    for idx, f in enumerate(sorted(files)):
        rel = Path(f).relative_to(corpus)
        entry = manifests.get(str(Path(f).resolve()))
        material = str((entry or {}).get("material") or rel.parts[0]).lower()
        items.append(
            {
                # Manifest-Einträge bekommen stabile Stem-IDs (Kuration!),
                # manifest-lose Dateien den Index (best effort).
                "id": str(entry.get("file", Path(f).stem)).split("/")[-1].rsplit(".", 1)[0]
                if entry
                else f"{material}_{idx + 1:02d}",
                "path": str(f),
                "material": material if material in MATERIALS else "unknown",
                "era_year": int(entry["era_year"]) if entry and entry.get("era_year") else None,
                "genre": str(entry.get("genre") or "") if entry else None,
                "defect_types": list(entry.get("defect_types") or []) if entry else None,
                "license": str(entry.get("license") or "") if entry else None,
                # Deklarierte Tonträgerkette (kuratierte Wahrheit, falls im Manifest
                # gepflegt) — der autoritative Durchreiche-Kanal für die Depth.
                "declared_chain": list(entry.get("chain") or []) if entry else None,
                "depth": None,
                "restorability_score": None,
                "detected_depth": None,
                "detected_restorability_score": None,
                "detected_material": None,
                "detected_confidence": None,
                "classification_issues": [],
                "classification_verified": False,
                "verified_by": None,
                "verified_at": None,
            }
        )
    return items


def _load_audio(path: str) -> tuple[np.ndarray, int] | None:
    """Lädt Audio als float32 (soundfile, sonst scipy-WAV). None bei Fehler."""
    try:
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32")
        return np.asarray(audio, dtype=np.float32), int(sr)
    except Exception:
        pass
    try:
        from scipy.io import wavfile

        sr, audio = wavfile.read(path)
        audio = np.asarray(audio, dtype=np.float32)
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / 32768.0
        return audio, int(sr)
    except Exception:
        return None


def classify_item(item: dict[str, Any]) -> None:
    """Detektor-Lauf — NUR provisorisch (detected_*); authoritativ erst nach Kuration.

    Begründung (§G5 (copilot-instructions.md), kein stilles Vertrauen in Schätzer): Der MediumDetector
    fehllabelt Materialien (empirisch belegt: Vinyl→Shellac bei conf 0.88).
    Deshalb:
    - Detektor-Werte landen ausschließlich in detected_*-Feldern.
    - Material-Cross-Check gegen die Corpus-Struktur (kuratierte Ground Truth
      des Korpus-Layouts). Mismatch → classification_issues.
    - Die authoritativen Felder depth/restorability_score bleiben None, bis
      verify_item sie nach Kurations-Entscheidung setzt.
    """
    loaded = _load_audio(str(item["path"]))
    if loaded is None:
        logger.debug("golden_set_tool: Audio nicht lesbar — %s", item.get("id"))
        item["classification_issues"] = ["Audio nicht lesbar"]
        item["classification_verified"] = False
        return
    audio, sr = loaded
    try:
        from backend.core.restorability_estimator import estimate_restorability

        _res = estimate_restorability(audio, sr, material=str(item.get("material") or "unknown"))
        _rs = float(getattr(_res, "restorability_score", float("nan")))
        item["detected_restorability_score"] = int(round(_rs)) if np.isfinite(_rs) else None
    except Exception:
        logger.debug("golden_set_tool: Restorability fehlgeschlagen für %s", item.get("id"), exc_info=True)
        item["detected_restorability_score"] = None
    try:
        # §9.4 Anti-Parallelwelten: die kanonische Implementierung liegt im
        # Top-Level-Paket forensics/medium_detector.py — exakt die Institution,
        # die auch die Pre-Analysis nutzt (pre_analysis.py: get_medium_detector()).
        # Der Singleton (§2.47a) vermeidet Mehrfach-Instanzen und nutzt ggf.
        # bereits vorhandenen Zustand.
        from forensics.medium_detector import get_medium_detector

        _file_ext = Path(str(item["path"])).suffix
        # §v10.14.1 Ära-Prior: Die Corpus-Manifeste liefern era_year als
        # kuratierte Wahrheit → era_confidence=0.9 aktiviert den Prior (>=0.40)
        # und löst die audio-inhärente Vinyl/Shellac-Ambiguität auf.
        _era = item.get("era_year")
        _era_decade = int(_era) if _era else None
        _era_conf = 0.9 if _era_decade else 0.0
        _chain = get_medium_detector().detect(
            audio, sr, file_ext=_file_ext, era_decade=_era_decade, era_confidence=_era_conf
        )
        item["detected_material"] = str(getattr(_chain, "primary_material", "") or "").lower()
        item["detected_confidence"] = float(getattr(_chain, "confidence", 0.0) or 0.0)
        _depth = int(len(getattr(_chain, "transfer_chain", []) or []))
        item["detected_depth"] = "4+" if _depth >= 4 else str(_depth)
    except Exception:
        logger.debug("golden_set_tool: MediumDetector fehlgeschlagen für %s", item.get("id"), exc_info=True)
        item["detected_material"] = None
        item["detected_confidence"] = None
        item["detected_depth"] = None
    issues: list[str] = []
    corpus_mat = str(item.get("material") or "").lower()
    det_mat = str(item.get("detected_material") or "").lower()
    if det_mat and corpus_mat and det_mat != corpus_mat:
        issues.append(
            f"Material-Mismatch: Corpus={corpus_mat}, Detektor={det_mat} (conf={item.get('detected_confidence')})"
        )
    item["classification_issues"] = issues
    item["classification_verified"] = False
    item["verified_by"] = None
    item["verified_at"] = None
    # Authoritativ bleibt None — Detektor-Werte dürfen ohne Kuration NIE zählen.
    item["depth"] = None
    item["restorability_score"] = None


def verify_item(
    item: dict[str, Any],
    *,
    depth: str | None = None,
    restorability: int | None = None,
    verified_by: str,
    accept_detected: bool = False,
    force: bool = False,
) -> list[str]:
    """Kuration: Setzt die authoritativen Werte — der einzige Weg, sie zu füllen.

    Regeln:
    - verified_by ist Pflicht (Audit-Trail; ohne Kurations-Entscheidung keine Werte).
    - accept_detected übernimmt Detektor-Werte, ABER verweigert bei
      Material-Mismatch (force=False) und fehlenden Detektor-Werten.
    - Explizite depth/restorability überschreiben — der Kurator ist die Wahrheit.
    """
    problems: list[str] = []
    if not str(verified_by or "").strip():
        raise ValueError("verified_by ist Pflicht (Audit-Trail)")
    mismatch = any("Material-Mismatch" in str(i) for i in item.get("classification_issues", []))
    declared = list(item.get("declared_chain") or [])
    new_depth = depth
    new_rs = restorability
    if new_depth is None and declared:
        # Kuratierte Kette = Wahrheit: Depth aus der deklarierten Kette,
        # unabhängig vom Detektor-Votum.
        _d = len(declared)
        new_depth = "4+" if _d >= 4 else str(_d)
    if new_depth is None and accept_detected:
        det = item.get("detected_depth")
        if det is None:
            problems.append("kein Detektor-Depth vorhanden")
        elif (mismatch and not declared) and not force:
            problems.append("Material-Mismatch — accept_detected verweigert (einzeln kurieren)")
        else:
            new_depth = str(det)
    if new_rs is None and accept_detected:
        det = item.get("detected_restorability_score")
        if det is None:
            problems.append("kein Detektor-Restorability vorhanden")
        elif (mismatch and not declared) and not force:
            problems.append("Material-Mismatch — accept_detected verweigert (einzeln kurieren)")
        else:
            new_rs = int(det)
    if problems:
        return problems
    if new_depth is None or new_rs is None:
        return ["depth und restorability_score müssen gesetzt sein (explizit oder detected)"]
    item["depth"] = str(new_depth)
    item["restorability_score"] = int(new_rs)
    item["classification_verified"] = True
    item["verified_by"] = str(verified_by).strip()
    item["verified_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return []


def verify_all(
    items: list[dict[str, Any]],
    *,
    verified_by: str,
    accept_detected: bool = True,
    force: bool = False,
) -> tuple[list[dict[str, Any]], list[tuple[str, list[str]]]]:
    """Kuriert alle Items; verweigerte Items landen in skipped (mit Gründen)."""
    done: list[dict[str, Any]] = []
    skipped: list[tuple[str, list[str]]] = []
    for it in items:
        problems = verify_item(it, verified_by=verified_by, accept_detected=accept_detected, force=force)
        if problems:
            skipped.append((str(it.get("id")), problems))
        else:
            done.append(it)
    return done, skipped


def curation_report(manifest: dict[str, Any]) -> dict[str, Any]:
    items = manifest.get("items", [])
    verified = [str(it.get("id")) for it in items if it.get("classification_verified") is True]
    unverified = [str(it.get("id")) for it in items if it.get("classification_verified") is not True]
    mismatches = {
        str(it.get("id")): list(it.get("classification_issues", []))
        for it in items
        if any("Material-Mismatch" in str(i) for i in it.get("classification_issues", []))
    }
    return {"verified": len(verified), "unverified": unverified, "mismatches": mismatches}


def coverage_report(manifest: dict[str, Any]) -> dict[str, Any]:
    items = manifest.get("items", [])
    mat_counts = dict.fromkeys(MATERIALS, 0)
    depth_counts = dict.fromkeys(DEPTH_CLASSES, 0)
    problems: list[str] = []
    for it in items:
        mid = str(it.get("id", "?"))
        if not Path(str(it.get("path", ""))).exists():
            problems.append(f"{mid}: Audio-Datei fehlt ({it.get('path')})")
        if it.get("classification_verified") is not True:
            # Detektor-Werte dürfen ohne Kurations-Entscheidung NIE in den
            # Gate einfließen (beobachtet: Vinyl→Shellac bei conf 0.88).
            problems.append(f"{mid}: Klassifikation nicht kuriert (verified fehlt)")
        mat = str(it.get("material", "")).lower()
        if mat in mat_counts:
            mat_counts[mat] += 1
        else:
            problems.append(f"{mid}: unbekanntes Material {mat!r} (erlaubt: {MATERIALS})")
        dep = str(it.get("depth", "")).lower()
        if dep in depth_counts:
            depth_counts[dep] += 1
        else:
            problems.append(f"{mid}: fehlende/unbekannte Depth-Klasse {dep!r} (erlaubt: {DEPTH_CLASSES})")
    if len(items) < MIN_ITEMS_TOTAL:
        problems.append(f"nur {len(items)} Items (< {MIN_ITEMS_TOTAL})")
    for m in MATERIALS:
        if mat_counts[m] < MIN_PER_MATERIAL:
            problems.append(f"Material {m}: {mat_counts[m]} Items (< {MIN_PER_MATERIAL})")
    for d in DEPTH_CLASSES:
        if depth_counts[d] < MIN_PER_DEPTH:
            problems.append(f"Depth {d}: {depth_counts[d]} Items (< {MIN_PER_DEPTH})")
    return {
        "items_total": len(items),
        "material_counts": mat_counts,
        "depth_counts": depth_counts,
        "problems": problems,
    }


def verdict_status(manifest: dict[str, Any]) -> dict[str, Any]:
    verdicts = manifest.get("last_verdicts") or {}
    listeners = int(verdicts.get("listeners", 0) or 0)
    covered = {str(i) for i in (verdicts.get("items_covered") or [])}
    item_ids = {str(it.get("id")) for it in manifest.get("items", [])}
    missing = sorted(item_ids - covered)
    problems: list[str] = []
    if not verdicts:
        problems.append("keine Hörurteile hinterlegt (fail-closed)")
    if listeners < MIN_LISTENERS:
        problems.append(f"nur {listeners} Hörer (< {MIN_LISTENERS})")
    if missing:
        problems.append(f"ohne Urteile: {missing}")
    return {"listeners": listeners, "missing": missing, "problems": problems}


def check_manifest(manifest: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Coverage + Hörurteil-Status eines geladenen Manifests. 0=PASS, 1=FAIL, 2=BLOCKED."""
    cov = coverage_report(manifest)
    ver = verdict_status(manifest)
    report: dict[str, Any] = {"coverage": cov, "verdicts": ver, "curation": curation_report(manifest)}
    if cov["problems"]:
        report["decision"] = "FAIL"
        return 1, report
    if ver["problems"]:
        report["decision"] = "BLOCKED"
        return 2, report
    report["decision"] = "PASS"
    return 0, report


def check(path: Path) -> tuple[int, dict[str, Any]]:
    """Lädt das Manifest von Disk und prüft es (siehe check_manifest)."""
    code, report = check_manifest(load_manifest(path))
    report["manifest"] = str(path)
    return code, report


def crosscheck(path: Path) -> dict[str, Any]:
    """Empfehlung 9: Flacher Klassifikator vs. MediumDetector vs. kuratierte Labels.

    Misst auf den verifizierten Manifest-Items die Übereinstimmung beider
    Schätzer mit der kuratierten Wahrheit — deterministisch, ohne Modelle zu
    verändern. Ohne Artefakt (models/medium_shallow_v1.joblib) → Fehlertext.
    """
    import joblib

    manifest = load_manifest(path)
    art_path = _ROOT / "models" / "medium_shallow_v1.joblib"
    if not art_path.exists():
        return {"error": f"Artefakt fehlt: {art_path} (erst scripts/train_medium_classifier.py ausführen)"}
    sys.path.insert(0, str(_ROOT / "scripts"))
    import train_medium_classifier as tm

    art = joblib.load(str(art_path))
    n = det_mat_ok = shallow_mat_ok = det_dep_ok = shallow_dep_ok = 0
    for it in manifest.get("items", []):
        if it.get("classification_verified") is not True:
            continue
        loaded = _load_audio(str(it["path"]))
        if loaded is None:
            continue
        audio = loaded[0]
        feats = tm.extract_features(audio)
        era = float(it.get("era_year") or 0.0)
        feats = np.concatenate([feats, np.asarray([era, era // 10.0 * 10.0], dtype=np.float32)]).reshape(1, -1)
        n += 1
        if it.get("detected_material") == it.get("material"):
            det_mat_ok += 1
        if it.get("detected_depth") == it.get("depth"):
            det_dep_ok += 1
        if str(art["material"]["model"].predict(feats)[0]) == str(it.get("material")):
            shallow_mat_ok += 1
        if str(art["depth"]["model"].predict(feats)[0]) == str(it.get("depth")):
            shallow_dep_ok += 1
    return {
        "n_verified": n,
        "medium_detector_agreement": {
            "material": round(det_mat_ok / n, 4) if n else None,
            "depth": round(det_dep_ok / n, 4) if n else None,
        },
        "shallow_train_agreement": {
            # Achtung: Trainings-Set-Leakage — der Klassifikator wurde auf
            # diesen Items gefittet. Die ehrlichen Werte sind cv_accuracy.
            "material": round(shallow_mat_ok / n, 4) if n else None,
            "depth": round(shallow_dep_ok / n, 4) if n else None,
        },
        "shallow_cv_accuracy": {
            "material": round(float(art["material"]["cv"]["accuracy"]), 4),
            "depth": round(float(art["depth"]["cv"]["accuracy"]), 4),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Golden Listening Set — Manifest & Coverage")
    sub = parser.add_subparsers(dest="command", required=True)
    p_init = sub.add_parser("init", help="Manifest aus Corpus-Verzeichnis erzeugen")
    p_init.add_argument("--corpus", type=Path, required=True)
    p_init.add_argument("--classify", action="store_true", help="Depth/Restorability detektieren")
    p_init.add_argument(
        "--subdir", nargs="+", default=None, help="nur diese Unterverzeichnisse scannen (z. B. damaged)"
    )
    p_init.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p_check = sub.add_parser("check", help="Coverage + Hörurteile prüfen (fail-closed)")
    p_check.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p_ver = sub.add_parser("verify", help="Klassifikation kurieren (einziger Weg zu authoritativen Werten)")
    p_ver.add_argument("--item", default=None, help="Item-ID (oder --all)")
    p_ver.add_argument("--all", action="store_true", help="alle Items kurieren (erfordert --accept-detected)")
    p_ver.add_argument("--depth", default=None, help="explizite Depth-Klasse (1|2|3|4+)")
    p_ver.add_argument("--restorability", type=int, default=None, help="expliziter Restorability-Score (0-100)")
    p_ver.add_argument("--accept-detected", action="store_true", help="Detektor-Werte übernehmen (ohne Mismatch)")
    p_ver.add_argument("--force", action="store_true", help="accept-detected trotz Material-Mismatch")
    p_ver.add_argument("--verified-by", required=True, help="Name des Kurators (Audit-Trail)")
    p_ver.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p_cross = sub.add_parser("crosscheck", help="Flacher Klassifikator vs. MediumDetector vs. kuratierte Labels")
    p_cross.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p_status = sub.add_parser("status", help="Kurzstatus ausgeben")
    p_status.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    if args.command == "init":
        subdirs = tuple(args.subdir) if args.subdir else None
        items = scan_corpus(args.corpus, subdirs=subdirs)
        if args.classify:
            for it in items:
                classify_item(it)
        manifest = {
            "version": 1,
            "description": "Goldenes Hör-Set (docs/guides/GOLDEN_LISTENING_SET.md)",
            "items": items,
            "last_verdicts": None,
        }
        save_manifest(args.manifest, manifest)
        print(f"Manifest geschrieben: {args.manifest} ({len(items)} Items)")
        return 0

    if args.command == "check":
        code, report = check(args.manifest)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return code

    if args.command == "crosscheck":
        print(json.dumps(crosscheck(args.manifest), indent=2, ensure_ascii=False))
        return 0

    if args.command == "verify":
        manifest = load_manifest(args.manifest)
        items = manifest.setdefault("items", [])
        if args.all:
            if not args.accept_detected:
                print("--all erfordert --accept-detected (sonst einzeln kurieren)", file=sys.stderr)
                return 1
            done, skipped = verify_all(items, verified_by=args.verified_by, accept_detected=True, force=args.force)
            print(f"kuriert: {len(done)}, verweigert: {len(skipped)}")
            for iid, probs in skipped:
                print(f"  {iid}: {probs}")
            if skipped:
                return 1
        else:
            if not args.item:
                print("--item <id> oder --all erforderlich", file=sys.stderr)
                return 1
            target = next((i for i in items if i.get("id") == args.item), None)
            if target is None:
                print(f"Item {args.item} nicht gefunden", file=sys.stderr)
                return 1
            problems = verify_item(
                target,
                depth=args.depth,
                restorability=args.restorability,
                verified_by=args.verified_by,
                accept_detected=args.accept_detected,
                force=args.force,
            )
            if problems:
                for p in problems:
                    print(f"  {p}", file=sys.stderr)
                return 1
            print(
                f"kuriert: {target['id']} depth={target['depth']} "
                f"rs={target['restorability_score']} by={target['verified_by']}"
            )
        save_manifest(args.manifest, manifest)
        return 0

    report = check(args.manifest)[1]
    print(
        f"decision={report['decision']} "
        f"items={report['coverage']['items_total']} "
        f"problems={len(report['coverage']['problems']) + len(report['verdicts']['problems'])}"
    )
    return {"PASS": 0, "FAIL": 1, "BLOCKED": 2}[report["decision"]]


if __name__ == "__main__":
    sys.exit(main())
