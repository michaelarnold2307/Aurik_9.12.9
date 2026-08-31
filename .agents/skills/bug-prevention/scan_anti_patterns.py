#!/usr/bin/env python3
"""Aurik SOTA Bug-Prevention Hook (§v10.105)

Fängt die 6 Bug-Klassen aus der Exception-Forensik (Juli 2026) PROAKTIV ab,
BEVOR sie in die Pipeline gelangen.

Gefundene Anti-Patterns (aus 460 analysierten Exceptions):
  P1: shape[0] <= shape[1] — falsche Kanal-Detection (→ Broadcast-Crash)
  P2: filtfilt( ohne Längen-Guard     (→ padlen-Crash)
  P3: stft( ohne noverlap-Clamp       (→ noverlap-Crash)
  P4: os.* ohne import os            (→ UnboundLocalError)
  P5: np.asarray(Tuple) in __post_init__ (→ inhomogeneous-Crash)
  P6: MaterialType-Enum als String    (→ KeyError in Dict-Lookups)
"""

import ast
import logging
import os
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Konfiguration ──────────────────────────────────────────────────────────

EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    ".venv_aurik",
    "build",
    "dist",
    "models",
    "output_audio",
    "sessions",
    "logs",
    "data",
    "golden_samples",
    "chain_templates",
    "configs",
    ".eggs",
    "tests",  # Tests dürfen Anti-Patterns für negative Tests enthalten
}

EXCLUDE_FILES = {
    "setup.py",
    "conftest.py",
    # Fixer-Scripts beschreiben Anti-Patterns (nicht nutzen sie)
    "fix_p6_material_lookups.py",
    "fix_p6_v2.py",
}

MIN_SEVERITY = "warning"  # "error" stoppt Commit, "warning" warnt nur

# §v10.115: Continuous Analysis — Scanner lädt neue Patterns aus Exception-Forensik
_PATTERN_FEED_PATH = Path(__file__).resolve().parents[3] / "logs" / "discovered_patterns.json"

# ── P1: shape[0] <= shape[1] Anti-Pattern ─────────────────────────────────


def check_shape_anti_pattern(filepath: str, source: str) -> list[str]:
    """Findet `audio.shape[0] <= audio.shape[1]` ohne `shape[1] > 2`-Check."""
    issues = []
    # Regex: shape[0] <= shape[1] aber NICHT gefolgt von "and shape[1] > 2" auf gleicher Zeile
    pattern = re.compile(r"\.shape\[0\]\s*<=\s*\.shape\[1\]")
    lines = source.split("\n")
    for i, line in enumerate(lines, 1):
        if pattern.search(line):
            # Erlaubt wenn "shape[1] > 2" oder "shape[0] <= 2 and" auf gleicher Zeile
            if "shape[0] <= 2 and" in line or "shape[1] > 2" in line:
                continue
            # Erlaubt wenn in Kommentar
            if line.strip().startswith("#"):
                continue
            issues.append(
                f"{filepath}:{i}: P1 shape[0]<=shape[1] ohne channels-first-Guard "
                f"(→ Broadcast-Crash bei channels-last mit N≤2). "
                f"FIX: `shape[0] <= 2 and shape[1] > 2`"
            )
    return issues


# ── P2: filtfilt ohne Längen-Guard ────────────────────────────────────────


def check_filtfilt_without_guard(filepath: str, source: str) -> list[str]:
    """Findet bare `filtfilt(` oder `signal.filtfilt(` (nicht safe_filtfilt)."""
    issues = []
    lines = source.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Bare filtfilt( calls (nicht safe_filtfilt, nicht sosfiltfilt)
        if re.search(r"(?<!safe_)(?<!sos)(?<!_)(?<!\.)\bfiltfilt\(", stripped):
            # Skip spec_constitution.py — filtfilt inside ForbiddenPattern strings
            if "spec_constitution.py" in filepath:
                continue
            # Skip files that DEFINE safe_filtfilt (audio_utils.py)
            if "def safe_filtfilt" in source:
                continue
            # Prüfe ob safe_filtfilt importiert oder im File definiert ist
            if "from backend.core.audio_utils import safe_filtfilt" not in source and "safe_filtfilt" not in source:
                issues.append(
                    f"{filepath}:{i}: P2 filtfilt() ohne Längen-Guard "
                    f"(→ padlen-Crash bei kurzem Audio). "
                    f"FIX: `from backend.core.audio_utils import safe_filtfilt` + Ersetzung"
                )
    return issues


# ── P3: stft ohne noverlap-Clamp ──────────────────────────────────────────


def check_stft_without_clamp(filepath: str, source: str) -> list[str]:
    """Findet `stft(` mit `noverlap=n_fft - hop` ohne min(n_fft-1)-Clamp."""
    issues = []
    lines = source.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "stft(" in stripped and "noverlap=" in stripped:
            # Prüfe ob ein min(..., nperseg-1) Clamp existiert
            if "max(0," not in source.split("\n")[max(0, i - 3) : i + 1].__str__():
                if "min(" not in stripped:
                    # Dynamic noverlap ohne Clamp
                    if "noverlap=n_fft - hop" in stripped or "noverlap=nperseg - hop" in stripped:
                        issues.append(
                            f"{filepath}:{i}: P3 stft() noverlap ohne min(nperseg-1)-Clamp "
                            f"(→ noverlap-Crash bei kurzem Audio). "
                            f"FIX: `_noverlap = min(n_fft - hop, max(0, n_fft - 1))`"
                        )
    return issues


# ── P4: os.* ohne import os ───────────────────────────────────────────────


def check_os_without_import(filepath: str, source: str) -> list[str]:
    """Findet `os.`-Nutzung ohne `import os` auf Module-Ebene."""
    issues = []
    if "os." not in source:
        return issues
    # Prüfe ob import os existiert (nicht in Funktionen, sondern auf Module-Ebene)
    has_module_import = bool(re.search(r"^(import os|from os import)", source, re.MULTILINE))
    if not has_module_import:
        # Prüfe ob os.* in einer Funktion verwendet wird (wo import fehlen könnte)
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name) and node.value.id == "os":
                        issues.append(
                            f"{filepath}:{node.lineno}: P4 os.{node.attr} ohne `import os` "
                            f"(→ UnboundLocalError in bestimmten Umgebungen). "
                            f"FIX: `import os` am Modul-Anfang"
                        )
        except SyntaxError:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
    return issues


# ── P5: np.asarray(Tuple) in PhaseResult.__post_init__ ─────────────────────


def check_asarray_tuple(filepath: str, source: str) -> list[str]:
    """Findet `np.asarray(self.audio)` ohne Tuple-Check in __post_init__."""
    issues = []
    if "def __post_init__" not in source:
        return issues
    if "np.asarray(self.audio" not in source and "np.asarray(self.audio" not in source:
        return issues
    # Prüfe ob Tuple-Check VOR asarray existiert
    if "isinstance(self.audio, (tuple, list))" not in source:
        issues.append(
            f"{filepath}: P5 np.asarray(self.audio) ohne Tuple→ndarray-Guard "
            f"(→ inhomogeneous-Crash bei Tuple-Rückgaben). "
            f"FIX: isinstance-Check vor np.asarray()"
        )
    return issues


# ── P6: MaterialType-Enum als String in Dict-Lookup ─────────────────────────


def check_enum_as_dict_key(filepath: str, source: str) -> list[str]:
    """Findet Dict-Lookups mit material/mat wo Keys MaterialType-Enums sind."""
    issues = []
    # Nur in Dateien die MaterialType importieren
    if "MaterialType" not in source:
        return issues
    lines = source.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Skip docstring/formula lines (inside triple-quoted strings)
        if "·" in stripped or "log10" in stripped:
            continue
        # Pattern: DICT[material] oder DICT.get(material) wo MaterialType-Enum-Keys
        if re.search(r"\[material\]", stripped) or re.search(r"\.get\(material[,\)]", stripped):
            # Prüfe ob Normalisierung existiert
            context_start = max(0, i - 3)
            context = "\n".join(lines[context_start:i])
            if (
                "isinstance(material, MaterialType)" not in context
                and 'hasattr(material, "value")' not in context
                and "_mat_enum_" not in context
                and ".get(_mat_" not in context
            ):
                issues.append(
                    f"{filepath}:{i}: P6 Dict-Lookup [material] ohne Enum-Normalisierung "
                    f"(→ KeyError wenn material String statt Enum). "
                    f"FIX: isinstance(material, MaterialType) + .get()-Fallback"
                )
    return issues


# ── H-Serie: Hörordnungs-/Exportqualitäts-Anti-Patterns ─────────────────────
# Muster, die Aurik daran hindern, die hochwertigsten Exportergebnisse für das
# menschliche Gehör zu liefern. Quellen: Hörordnung (hoerordnung.instructions.md),
# dsp.instructions.md, copilot-instructions (§V5/§G5), Befunde 2026-08-23.


def check_hoerordnung_export_patterns(filepath: str, source: str) -> list[str]:
    """H-Serie: Psychoakustik-/Exportqualitäts-Schwachstellen im Code."""
    issues = []
    lines = source.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        # H01: nacktes astype(np.int16) ohne Dither (POW-r/TPDF) — §V5 (copilot-instructions.md)
        if re.search(r"astype\(np\.int16\)", stripped):
            if not re.search(r"powr|tpdf|dither|noise_shape", source, re.IGNORECASE):
                issues.append(
                    f"{filepath}:{i}: H01 nacktes astype(np.int16) ohne Dither "
                    f"(→ Quantisierungsrauschen, §V5 (copilot-instructions.md)). "
                    f"FIX: POW-r Type 3 / TPDF vor Int16-Export"
                )

        # H02: griffinlim() als Endschritt — VERBOTEN V05 (PGHI/Vocos-Pflicht)
        if re.search(r"\bgriffinlim\(", stripped):
            issues.append(
                f"{filepath}:{i}: H02 griffinlim() in Produktionscode "
                f"(→ nicht-deterministisch, V05). "
                f"FIX: PGHI (pghi_reconstruct) oder Vocos"
            )

        # H03: sosfilt() ohne zero-phase (sosfiltfilt) im Master-Audio-Pfad —
        # dsp.instructions „Bandfilter — Zero-Phase“. Nur Phasen/DSP melden:
        # dort ist Filterung Signal-Verarbeitung, wo Phase nicht zum Original
        # addiert werden darf. Analyse-/Realtime-Kontexte sind ausgenommen.
        if re.search(r"\bsosfilt\(", stripped) and (
            "/phases/" in filepath.replace("\\", "/") or "/dsp/" in filepath.replace("\\", "/")
        ):
            issues.append(
                f"{filepath}:{i}: H03 sosfilt() statt sosfiltfilt() "
                f"(→ Phase addiert zu Original, Zero-Phase-Verstoß). "
                f"FIX: sosfiltfilt(sos, audio)"
            )

        # H04: time.time() IN Entscheidungslogik (if/compare) — §G5 Determinismus.
        # Reines Profiling (Zuweisung/Subtraktion) ist zulässig und wird nicht gemeldet.
        if re.search(r"\btime\.time\(\)", stripped) and re.search(
            r"\bif\b.*time\.time\(\)|time\.time\(\).*(?:<|>|==|!=|<=|>=)", stripped
        ):
            issues.append(
                f"{filepath}:{i}: H04 time.time() in Entscheidungslogik "
                f"(→ nicht-deterministisch, §G5 (copilot-instructions.md)). "
                f"FIX: Session-Seed / monotonic statt wall-clock"
            )

        # H05: resample() ohne Längen-Guard im Master-Audio-Pfad (Zeitachsen-
        # Zerstörung — Befund 2026-08-23: 224s vs 30s → FATAL-Trim; Hörordnung:
        # Sample-Exaktheit der Zeitachse). Nur Phasen/DSP — SR-Konvertierung an
        # zentralen, bewusst guardierten Stellen ist ausgenommen.
        if re.search(r"\b(librosa\.resample|signal\.resample|resample_poly)\(", stripped) and (
            "/phases/" in filepath.replace("\\", "/") or "/dsp/" in filepath.replace("\\", "/")
        ):
            if not re.search(
                r"_len_diff|shape\[[^]]*\].*==|Längen|laenge|len_mismatch|abs\(len\(",
                source,
                re.IGNORECASE,
            ):
                issues.append(
                    f"{filepath}:{i}: H05 resample() ohne Längen-Differenz-Guard "
                    f"(→ Zeitkompression bei Längen-Mismatch, Hörordnung §Zeitachse). "
                    f"FIX: >0.1% Differenz → trim/pad statt resample"
                )

        # H06: Hard-Clamp (-1,1) in Phasen ohne Soft-Knee — §III (copilot-instructions.md)
        if "backend/core/phases" in filepath.replace("\\", "/") and re.search(
            r"np\.clip\([^)]*(?:-1\.0?,\s*1\.0?|1\.0?,\s*-1\.0?)", stripped
        ):
            if not re.search(r"soft_knee|soft-knee|knee|hanning|hann", source, re.IGNORECASE):
                issues.append(
                    f"{filepath}:{i}: H06 Hard-Clamp (-1,1) ohne Soft-Knee "
                    f"(→ hörbare Clipping-Artefakte, §III). "
                    f"FIX: Soft-Knee (6 dB, 200 ms Hanning)"
                )

        # H07: Silent-Except mit neutralem Return ohne logger — §V6 Silent-Failure-Verbot
        if re.search(r"except\s+Exception", stripped) or stripped == "except Exception:":
            _window = "\n".join(lines[i : min(i + 3, len(lines))])
            if re.search(r"return\s+[01]\.\d*", _window) and "logger" not in _window:
                issues.append(
                    f"{filepath}:{i}: H07 Silent-Except → neutraler Return ohne logger.warning "
                    f"(→ ML→DSP-Fallback unsichtbar, §V6 (copilot-instructions.md))"
                )

    return issues


# ── §v10.115 Continuous Analysis: Dynamische Pattern-Erkennung ────────────────


_H_PRIORITY = {"H02": 0, "H01": 1, "H06": 2, "H03": 3, "H05": 4, "H04": 5, "H07": 6}
_H_TITLES = {
    "H01": "H01 — Dither-Pflicht (Int16-Export)",
    "H02": "H02 — PGHI statt Griffin-Lim",
    "H03": "H03 — Zero-Phase-Filter (sosfiltfilt)",
    "H04": "H04 — Determinismus (time.time in Entscheidungen)",
    "H05": "H05 — Resample ohne Längen-Guard",
    "H06": "H06 — Soft-Knee statt Hard-Clamp",
    "H07": "H07 — Silent-Failure-Verbot (logger im Except)",
}


def _write_hoerordnung_todo(all_issues: list[str], todo_path: str) -> None:
    """Persistiert H-Serie-Funde als priorisierte, abarbeitbare To-Do-Liste.

    Format: Markdown-Checklisten mit stabilen IDs (H02-001, …), sortiert nach
    Schwere (H02 > H01 > H06 > H03 > H05 > H04 > H07). Beim Neuschreiben
    werden bereits abgehakte Einträge ([x]) übernommen — erledigte Punkte
    gehen über Sessions nicht verloren.
    """
    import hashlib as _hl
    import re as _re
    from datetime import date as _date

    _h_issues = [s for s in all_issues if _re.search(r":\s*(H0[1-7]) ", s)]
    if not _h_issues:
        _h_issues = []
    # Bereits erledigte IDs aus der bestehenden Datei laden
    _done: set[str] = set()
    try:
        with open(todo_path, encoding="utf-8") as f:
            _old = f.read()
        _done = set(_re.findall(r"^- \[x\] (H0[1-7]-[0-9a-f]{6}) ", _old, _re.MULTILINE))
    except (OSError, UnicodeDecodeError):
        pass

    # Gruppieren nach ID-Präfix + Datei (stabile ID pro Fundstelle)
    _entries: list[tuple[int, str, str, str]] = []
    _seen: dict[str, int] = {}
    for _issue in sorted(_h_issues):
        _m = _re.match(r"^(.+?):(\d+): (H0[1-7]) (.*)$", _issue)
        if not _m:
            continue
        _path, _line, _hid, _desc = _m.group(1), int(_m.group(2)), _m.group(3), _m.group(4)
        _key = f"{_path}:{_line}:{_hid}"
        if _key in _seen:
            continue
        _seen[_key] = 1
        # Stabile ID pro Fundstelle (Hash aus Pfad:Zeile:Muster) — unabhängig
        # von der Gesamtmenge, damit der Status-Erhalt über Sessions hält.
        _eid = f"{_hid}-{_hl.sha256(_key.encode('utf-8')).hexdigest()[:6]}"
        _prio = _H_PRIORITY.get(_hid, 9)
        _entries.append((_prio, _eid, f"{_path}:{_line}", _desc))

    _entries.sort(key=lambda e: (e[0], e[2]))

    _out_lines = [
        "# Hörordnungs-Schwachstellen — To-Do für die nächste Programmier-Session",
        "",
        f"> Generiert von `scan_anti_patterns.py --write-todo` ({_date.today().isoformat()}).",
        "> Abgehakte Einträge bleiben erledigt; neue Funde werden angehängt.",
        "> Priorität: H02 > H01 > H06 > H03 > H05 > H04 > H07.",
        "",
    ]
    _prev_prio: int | None = None
    for _prio, _eid, _loc, _desc in _entries:
        _hid = _eid[:3]
        if _prev_prio != _prio:
            _out_lines.append(f"## {_H_TITLES.get(_hid, _hid)}")
            _prev_prio = _prio
        _mark = "x" if _eid in _done else " "
        _out_lines.append(f"- [{_mark}] {_eid} | {_loc} | {_desc}")
    _out_lines.append("")

    _dest = Path(todo_path)
    _dest.parent.mkdir(parents=True, exist_ok=True)
    _dest.write_text("\n".join(_out_lines), encoding="utf-8")
    print(f"📋 Hörordnungs-To-Do geschrieben: {_dest} ({len(_entries)} Einträge, {len(_done)} bereits erledigt)")


def _load_discovered_patterns() -> list[str]:
    """Lädt vom Pattern-Miner entdeckte Patterns aus logs/discovered_patterns.json."""
    import json

    if not _PATTERN_FEED_PATH.exists():
        return []
    try:
        with open(_PATTERN_FEED_PATH) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    issues = []
    for pattern in data.get("patterns", []):
        if pattern.get("status") != "active":
            continue
        regex = pattern.get("regex")
        if not regex:
            continue
        message = pattern.get("message", "P7 Dynamisch entdecktes Anti-Pattern")
        for root in data.get("scan_roots", ["backend/core"]):
            repo_root = _PATTERN_FEED_PATH.parents[1]
            scan_dir = repo_root / root
            if not scan_dir.exists():
                continue
            for dirpath, _dirnames, filenames in os.walk(scan_dir):
                for fn in filenames:
                    if not fn.endswith(".py"):
                        continue
                    fp = os.path.join(dirpath, fn)
                    try:
                        with open(fp, encoding="utf-8") as fh:
                            src = fh.read()
                    except (UnicodeDecodeError, IsADirectoryError):
                        continue
                    for i, line in enumerate(src.split("\n"), 1):
                        s = line.strip()
                        if s.startswith("#"):
                            continue
                        if re.search(regex, s):
                            issues.append(f"{fp}:{i}: {message}")
    return issues


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    """Scannt alle Python-Dateien auf bekannte Bug-Patterns.

    §v10.114: Scanner auf alle Layer ausgeweitet (backend/core, plugins,
    Aurik10, denker, scripts).

    --write-todo: persistiert die H-Serie-Funde als priorisierte To-Do-Liste
    (reports/hoerordnung_schwachstellen.md) für die nächste Programmier-Session.
    Status-Erhalt: bereits abgehakte Einträge bleiben erledigt.
    """
    import argparse

    _parser = argparse.ArgumentParser()
    _parser.add_argument("--write-todo", action="store_true", help="H-Serie als To-Do-Liste persistieren")
    _parser.add_argument(
        "--todo-path",
        default=str((Path(__file__).resolve().parents[3]) / "reports" / "hoerordnung_schwachstellen.md"),
        help="Zielpfad der To-Do-Liste",
    )
    _args, _ = _parser.parse_known_args()

    root = Path(__file__).resolve().parents[3]  # .agents/skills/bug-prevention/ → repo root

    SCAN_ROOTS = [
        root / "backend" / "core",
        root / "plugins",
        root / "Aurik10",
        root / "denker",
        root / "scripts",
    ]

    all_issues: list[str] = []
    files_scanned = 0

    for scan_root in SCAN_ROOTS:
        if not scan_root.exists():
            continue

        for dirpath, dirnames, filenames in os.walk(scan_root):
            # Filtere Verzeichnisse
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                if filename in EXCLUDE_FILES:
                    continue

                filepath = os.path.join(dirpath, filename)
                files_scanned += 1

                try:
                    with open(filepath, encoding="utf-8") as f:
                        source = f.read()
                except (UnicodeDecodeError, IsADirectoryError):
                    continue

                # Alle Checks
                all_issues.extend(check_shape_anti_pattern(filepath, source))
                all_issues.extend(check_filtfilt_without_guard(filepath, source))
                all_issues.extend(check_stft_without_clamp(filepath, source))
                all_issues.extend(check_os_without_import(filepath, source))
                all_issues.extend(check_asarray_tuple(filepath, source))
                all_issues.extend(check_enum_as_dict_key(filepath, source))
                all_issues.extend(check_hoerordnung_export_patterns(filepath, source))

    # §v10.115: Lade dynamisch entdeckte Patterns aus Exception-Forensik
    discovered = _load_discovered_patterns()
    all_issues.extend(discovered)

    # ── H-Serie-To-Do persistieren (Session-Übergabe) ──
    if _args.write_todo:
        _write_hoerordnung_todo(all_issues, _args.todo_path)

    # Ausgabe
    if all_issues:
        print(
            f"\n🔍 Aurik SOTA Bug-Scan: {len(all_issues)} potentielle Bugs gefunden "
            f"({files_scanned} Dateien gescannt)\n"
        )
        for issue in sorted(all_issues):
            print(f"  {issue}")

        if MIN_SEVERITY == "error":
            print(f"\n❌ {len(all_issues)} Fehler — Commit blockiert.")
            print(
                "   Behebe die oben genannten Anti-Patterns oder füge    begründete Ausnahmen in EXCLUDE_FILES hinzu."
            )
            return 1
        else:
            print(f"\n⚠️  {len(all_issues)} Warnungen — bitte vor Commit prüfen.")
            return 0
    else:
        print(f"✅ Aurik SOTA Bug-Scan: Keine Anti-Patterns gefunden ({files_scanned} Dateien gescannt)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
