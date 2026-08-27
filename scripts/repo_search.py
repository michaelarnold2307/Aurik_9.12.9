#!/usr/bin/env python3
"""Repo-Suche — BM25 über Modul-Docstrings und Symbole mit Status-Gewichtung.

Embedding-freies Retrieval (nur Standardbibliothek). Der BM25-Score folgt
dem Inhalt; die Registry-Gewichtung setzt „Status schlägt Score“ um:
ACTIVE gewinnt gegen semantisch ähnliche DEPRECATED-Dateien, FORBIDDEN wird
nie vorgeschlagen. Unregistrierte Bestandsdateien zählen als aktiv-ähnlich
(Rollout der FILE_REGISTRY läuft noch).

Betriebsarten:
  python scripts/repo_search.py "defekt quantisierung"               # Top 10
  python scripts/repo_search.py --json "query"                       # maschinenlesbar
  python scripts/repo_search.py --before-create pfad/datei.py        # Kanon-Check
  python scripts/repo_search.py --before-create pfad/datei.py --query "..."

Exit-Codes:
  0 = fertig; keine kanonische Alternative gefunden
  2 = Nutzungsfehler
  3 = kanonische ACTIVE-Alternative existiert (advisory: vor Anlage prüfen)
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STATUS_WEIGHT = {
    "ACTIVE": 1.0,
    "MIGRATING": 0.7,
    "DEPRECATED": 0.6,
    "TEST_ONLY": 0.5,
    "GENERATED": 0.4,
    "ARCHIVED": 0.2,
    "FORBIDDEN": 0.0,
}
DEFAULT_WEIGHT = 0.9
TOP_N = 10
_BEFORE_CREATE_SCORE = 4.0
_TOKEN_RE = re.compile(r"[a-zäöüß][a-zäöüß0-9_]{1,}")
_SUFFIX_RE = re.compile(r"^(?P<base>.+?)(?:[-_ ]?(?:old|new|legacy|backup|final|deprecated|copy|alt|v?\d+))?$")


def _load_repo_graph():
    """Lädt scripts/repo_graph.py als Modul (kein Package-Import nötig)."""
    path = Path(__file__).with_name("repo_graph.py")
    spec = importlib.util.spec_from_file_location("repo_graph", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _module_docstring(path: Path) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return ""
    return ast.get_docstring(tree) or ""


def build_index() -> dict[str, dict]:
    """Baut den Suchindex: Pfad → Tokens, Länge, Status, Domain, Gewicht."""
    rg = _load_repo_graph()
    registry = rg.parse_file_registry()
    docs: dict[str, dict] = {}
    for py in rg.collect_py_files():
        rel = str(py.relative_to(ROOT))
        reg = registry.get(rel, {})
        status = reg.get("status", "")
        _imports, classes, _functions, symbols, _is_entry = rg.parse_py(py)
        text = " ".join(
            [
                _module_docstring(py),
                Path(rel).stem,
                " ".join(classes),
                " ".join(symbols),
            ]
        )
        tokens = _tokenize(text)
        docs[rel] = {
            "tokens": Counter(tokens),
            "len": len(tokens),
            "status": status or "UNREGISTRIERT",
            "domain": reg.get("domain", ""),
            "canonical": reg.get("canonical", False),
            "weight": STATUS_WEIGHT.get(status, DEFAULT_WEIGHT),
        }
    return docs


def bm25(query: str, docs: dict[str, dict], top_n: int = TOP_N) -> list[tuple[str, float]]:
    """BM25 über alle Dokumente, multipliziert mit der Status-Gewichtung."""
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []
    n_docs = len(docs)
    avg_len = sum(d["len"] for d in docs.values()) / max(1, n_docs)
    doc_freq: Counter = Counter()
    for doc in docs.values():
        doc_freq.update(set(doc["tokens"]))
    scores: list[tuple[str, float]] = []
    k1, b = 1.5, 0.75
    for rel, doc in docs.items():
        if doc["weight"] <= 0.0:
            continue
        score = 0.0
        for token in set(q_tokens):
            df = doc_freq[token]
            if df == 0:
                continue
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            tf = doc["tokens"][token]
            denom = tf + k1 * (1.0 - b + b * doc["len"] / max(1.0, avg_len))
            score += idf * (tf * (k1 + 1.0)) / denom
        score *= doc["weight"]
        if score > 0.0:
            scores.append((rel, score))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:top_n]


def _status_tag(doc: dict) -> str:
    return " CANONICAL" if doc["status"] == "ACTIVE" and doc["canonical"] else ""


def _print_hits(hits: list[tuple[str, float]], docs: dict[str, dict]) -> None:
    if not hits:
        print("Keine Treffer.")
        return
    for i, (rel, score) in enumerate(hits, start=1):
        doc = docs[rel]
        domain = f"  domain={doc['domain']}" if doc["domain"] else ""
        print(f"{i:2d}. {rel}  score={score:.2f}  {doc['status']}{domain}{_status_tag(doc)}")


def before_create(path_arg: str, query: str | None, docs: dict[str, dict]) -> int:
    """Prüft Namens-/Semantik-Kollisionen vor einer Dateianlage. Advisory (Exit 3)."""
    target = Path(path_arg)
    if target.suffix != ".py":
        print("FEHLER: --before-create erwartet einen .py-Pfad.", file=sys.stderr)
        return 2
    stem = target.stem
    match = _SUFFIX_RE.fullmatch(stem)
    base = match.group("base") if match else stem

    hits: list[str] = []
    for rel in docs:
        other = Path(rel).stem
        other_match = _SUFFIX_RE.fullmatch(other)
        other_base = other_match.group("base") if other_match else other
        if other_base == base:
            hits.append(rel)
    found_canonical = False
    if hits:
        print(f"Namens-ähnliche Dateien zu {path_arg}:")
        for rel in sorted(hits):
            doc = docs[rel]
            print(f"  - {rel}  {doc['status']}{_status_tag(doc)}")
            if doc["status"] == "ACTIVE" and doc["canonical"]:
                found_canonical = True

    if query:
        print(f"\nSemantisch ähnlich zu '{query}':")
        for rel, score in bm25(query, docs, top_n=5):
            doc = docs[rel]
            print(f"  - {rel}  score={score:.2f}  {doc['status']}{_status_tag(doc)}")
            if doc["status"] == "ACTIVE" and doc["canonical"] and score >= _BEFORE_CREATE_SCORE:
                found_canonical = True

    if found_canonical:
        print("\nCREATE ADVISORY: kanonische ACTIVE-Alternative existiert — erweitern statt neu anlegen.")
        return 3
    print(
        "\nKeine kanonische Alternative gefunden — Anlage plausibel "
        "(Registry-Eintrag in .github/FILE_REGISTRY.md nicht vergessen)."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="BM25-Repo-Suche mit Status-Gewichtung.")
    parser.add_argument("query", nargs="?", help="Suchanfrage (optional bei --before-create)")
    parser.add_argument("--query", dest="query_opt", help="Suchanfrage für --before-create (Alias zur Position)")
    parser.add_argument("--before-create", metavar="PFAD", help="Prüfung vor Dateianlage")
    parser.add_argument("--json", action="store_true", help="maschinenlesbare Ausgabe")
    parser.add_argument("--top", type=int, default=TOP_N, help="Anzahl Treffer (Standard 10)")
    args = parser.parse_args()

    if args.before_create:
        docs = build_index()
        return before_create(args.before_create, args.query or args.query_opt, docs)

    if not args.query:
        parser.print_help()
        return 2

    docs = build_index()
    hits = bm25(args.query, docs, top_n=args.top)
    if args.json:
        out = [
            {
                "path": rel,
                "score": round(score, 4),
                "status": docs[rel]["status"],
                "domain": docs[rel]["domain"],
                "canonical": docs[rel]["canonical"],
                "weight": docs[rel]["weight"],
            }
            for rel, score in hits
        ]
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0
    _print_hits(hits, docs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
