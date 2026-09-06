# GUI-Reife-Audit (2026-09-06) — Aurik10 Gesamtpaket

Umfang: Aurik10 = 37.017 Zeilen / 20+ Module; modern_window.py = 26.611 Zeilen.

## Evidenz (Kennzahlen, Zähler über Aurik10)

| Dimension | Messwert | Befund |
|---|---|---|
| Code-Umfang | modern_window.py 26.611 Z. | Monolith; Aufteilung in Widgets begonnen (ui/*, 17 Module) |
| Threading | 3 QThread-Subklassen (ml_refinement_thread 2, modern_window 1); IPC ipc/pipeline_process.py | KMV-Stufe-2 als Thread vorhanden (Konsument deferred_phases) |
| Abbruch | 15 Cancel-/Abort-Flags in modern_window.py | Pfade existieren; Abbruch-Telemetrie/Recovery ungeprüft |
| Fortschritt | 50× setValue (Fortschrittsbalken) + restoration_status_panel.py | Mechanik vorhanden; Phasen-/Tail-Fortschritt („QA-Phase") fehlt |
| Fehlerbehandlung | 195 breite `except:`-Zeilen ohne Typ; 9 QMessageBox-critical/warning | Risiko stiller Fehler; Rückmeldung ungleichmäßig |
| i18n | nur 26 `t('`-Aufrufe; ≥629 Zeilen mit Umlaut-Rohstrings | i18n-Lücke (t()-Wrapper vorhanden, Abdeckung gering) |
| Tests | 7 GUI-nahe Dateien in tests/unit/ | Headless-testbare Helfer ausbaufähig |
| Neue Hör-Gates in GUI | 0 Treffer für audibility/einladung/vocal_drive | Anzeige fehlt (Anker: _build_quality_score_text:22532, _build_quality_banner_sections:22610) |
| Export | GUI-Schreiber modern_window.py:3961 (sf.write-Fallback) + AudioExporter; Suffix „_restauriert" (:11567) | Pfad vorhanden; Gate-Hürden (audibility/vocal) vor Schreibpunkt ungeprüft |

## Tranchen (priorisiert, je einzeln validiert)

- **T1 Hör-Gates sichtbar machen** (Nutzwert sofort): Helper `_summarize_hearing_gates(meta)` → Zeile in Score-Text + Detail im Banner; Caller-Durchreichung. GO: Unit-Test Helper + Smoke-Import.
- **T2 Export-Härtung**: vor sf.write/AudioExporter: audibility_gate + vocal_drive + einladung als letzte Hürde (bei NO-GO: Warnung im Dialog + „trotzdem exportieren"-Pfad). GO: Pfad-Test mit künstlicher Metadata.
- **T3 Tail-Fortschritt**: Phasen-Ende → „Qualitäts-Prüfung x/y" im Status-Panel (beseitigt „hängt"-Wahrnehmung). GO: manueller Lauf.
- **T4 Fehler-Sichtbarkeit**: stille excepts → logger.warning + einmaliger UI-Hinweis je Kategorie. GO: Zähler sinkt, kein Dialog-Spam.
- **T5 i18n-Abdeckung**: Umlaut-Rohstrings → t(); GO: ≥90 % der sichtbaren Strings via t().
- **T6 GUI-Tests**: Helfer aus modern_window extrahieren & testen (Gate-Summary, Export-Guard, Fortschritt). GO: +5 Unit-Tests grün.
- **T7 Langzeit/Stabilität (Folge)**: Session-Recovery nach Abbruch, Ressourcen-Freigabe, Regression-Suite GUI-Smoke (headless).

## Verknüpfung Kern
Alle neuen Lauf-Gates liefern result.metadata (audibility_gate, vocal_drive_*, einladung_gate_*) — GUI muss sie nur anzeigen (T1) und beim Export als letzte Hürde nutzen (T2).
