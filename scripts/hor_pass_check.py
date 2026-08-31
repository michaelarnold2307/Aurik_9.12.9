#!/usr/bin/env python3
"""Hör-Pass-Log-Check — automatische Marker-Prüfung für den GO/NO-GO-Hör-Pass.

Ergänzt den menschlichen Hör-Entscheid nach
`docs/guides/GO_NO_GO_DECISION_PROTOCOL.md`: Das Ohr entscheidet GO/NO-GO,
dieses Skript bestätigt maschinell, dass die Wohlklang-Fixes der Session
2026-08-22 im jeweiligen Lauf tatsächlich gefeuert haben — und dass keine
bekannte Regressions-Signatur wieder aufgetaucht ist.

Betriebsarten:
  python scripts/hor_pass_check.py --log pfad/zum/lauf.log

Exit-Codes:
  0 = keine Regressions-Signatur gefunden
  1 = mindestens eine Regressions-Signatur gefunden
  2 = Nutzungsfehler (Log fehlt/unlesbar)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bekannte Regressions-Signaturen der Session 2026-08-22 — dürfen NICHT erscheinen.
REGRESSION_MARKERS: list[tuple[str, str]] = [
    ("EraClassifier dominiert MC", "Material-Flip-Flop (Konsens-Write-back inaktiv)"),
    ("Audio zu kurz (2 Samples)", "2-Sample-Scan (Kanal-Mix-Bug in Consensus)"),
    ("RLP-last fehlgeschlagen", "RLP-padlen-Crash (Längenschutz inaktiv)"),
    ("HPE/Goosebumps übersprungen", "HPE-Skip bei langen Songs (Excerpt-Fix inaktiv)"),
    ("initial Wert call took", "FeedbackChain ohne Längen-Cap"),
    ("but formants are female-typical", "Contralto-Fehlmeldung (F2-Konsistenz-Fix inaktiv)"),
    ("Verarbeitungsschritt_dag.py::_Verarbeitungsschritt_num Ersatzpfad", "Glue-Exception-Pfad (Werkzeugsverbot)"),
    ("CLAP-Modell nicht geladen", "Era-Tier-1 deterministisch übersprungen"),
    ("(48 entfernt", "Surgery-Doppelzählung (Zuweisung statt Addition)"),
]

# Erwartete Fix-Marker eines vollständigen Laufes.
POSITIVE_MARKERS: list[tuple[str, str]] = [
    ("pre_Analyse: vollstaendig", "Pre-Analyse abgeschlossen"),
    ("PIM-first: Intensitäts-Map", "PIM-first aktiv"),
    ("Material-Konsens final — primary=", "Konsens-Write-back aktiv"),
    # Hörordnung (2026-08-23): erwartbare Marker der verdrahteten Ebenen
    ("Hörordnung Ebene 2: PerceptualSalience wirkt als Pass-Through", "Salience-Pass-Through erkannt (Ebene 2)"),
    ("Einladungs-Gate:", "Einladungs-Gate gelaufen (Ebene 4)"),
    ("maskiert (ERB)", "ERB-Maskierungs-Skip möglich (Ebene 2, Stufe B)"),
    ("Messartefakt-Verdacht", "Konfliktregel-Kennzeichnung aktiv (Hörordnung §7)"),
    ("Restoration vollstaendig", "Restoration abgeschlossen"),
]

# Mengen-Prüfungen (Warnung, kein Fehlschlag — Fallback-Pfade sind spezifiziert).
WARN_COUNTS: list[tuple[str, str, int]] = [
    ("DefectScan abgeschlossen", "Mehr als ein Full-Song-DefectScan", 1),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Hör-Pass-Log-Check (Marker vs. Regressionen).")
    parser.add_argument("--log", required=True, help="Pfad zum Lauf-Log")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.is_file():
        print(f"FEHLER: Log nicht gefunden: {log_path}", file=sys.stderr)
        return 2
    text = log_path.read_text(encoding="utf-8", errors="replace")

    regressions = 0

    print(f"== Regressionen (dürfen NICHT erscheinen) — {log_path.name} ==")
    for marker, desc in REGRESSION_MARKERS:
        if marker in text:
            print(f"  ❌ REGRESSION: {desc}  (Marker: {marker!r})")
            regressions += 1
    if regressions == 0:
        print("  ✅ keine Regressions-Signatur gefunden")

    print("== Erwartete Fix-Marker ==")
    for marker, desc in POSITIVE_MARKERS:
        if marker in text:
            print(f"  ✅ {desc}")
        else:
            print(f"  ◐ FEHLT: {desc} (Marker: {marker!r})")

    print("== Mengen-Prüfungen (Warnung bei Überschreitung) ==")
    for marker, desc, limit in WARN_COUNTS:
        count = text.count(marker)
        status = "✅" if count <= limit else "⚠️"
        print(f"  {status} {desc}: {count}× (Limit {limit})")

    if regressions:
        print("\nErgebnis: REGRESSIONEN GEFUNDEN — Hör-Pass blockiert bis behoben.")
        return 1
    print("\nErgebnis: SAUBER — Hör-Entscheid jetzt nach GO/NO-GO-Protokoll treffen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
