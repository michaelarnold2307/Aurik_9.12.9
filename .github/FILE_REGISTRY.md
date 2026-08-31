# FILE_REGISTRY — Kanonische Datei-Identität, Lifecycle und Replacements

> **Status: Aktiv — CI-enforced.** Kanonische Quelle dafür, welche Datei
> aktuell (Canonical), welche ersetzt ist und welcher Status gilt.
> Pre-Commit-Hook `aurik-file-lifecycle` erzwingt fail-closed: neue
> Code-Dateien (backend/ plugins/ denker/ Aurik10/ cli/ scripts/) brauchen
> einen Eintrag (Write-Gate), DEPRECATED/MIGRATING braucht ein
> „Ersetzt“-Ziel, FORBIDDEN/ARCHIVED dürfen nicht importiert werden.
> Validator: `scripts/file_registry_check.py`. Duplikat-Erkennung:
> `scripts/repo_graph.py --duplicates` (Hook `aurik-symbol-duplicates`).
> Suche vor Dateianlage: `scripts/repo_search.py --before-create`.
> Drift-Schutz: `scripts/spec_drift_check.py` (WATCHED_FILES).

## Status-Enums

| Status | Bedeutung |
|---|---|
| ACTIVE | kanonisch oder aktiv in Verwendung |
| DEPRECATED | ersetzt; nur noch für Migration; braucht „Ersetzt“-Ziel |
| MIGRATING | Übergang läuft; Ziel steht in „Ersetzt“ |
| GENERATED | maschinell erzeugt; nicht von Hand ändern |
| TEST_ONLY | nur Testnutzung; nicht aus Produktion importieren |
| ARCHIVED | stillgelegt; kein Import erlaubt |
| FORBIDDEN | Import verboten (fail-closed) |

## Dateien

| Pfad | Status | Domain | Canonical | Ersetzt | Grund |
|---|---|---|---|---|---|
| scripts/repo_graph.py | ACTIVE | tooling/ci | ja | — | Import-Graph, Symbole, Duplikat-Check; konsolidiert audit_silent_dead_imports.py + audit_bridge_coverage.py |
| scripts/file_registry_check.py | ACTIVE | tooling/ci | ja | — | Write-Gate `aurik-file-lifecycle`; validiert diese Registry (R1–R7) |
| scripts/change_ledger.py | ACTIVE | tooling/ci | ja | — | TASK_CHANGES.md (snapshot/check); CI-Abdeckungs-Gate |
| scripts/repo_search.py | ACTIVE | tooling/agents | ja | — | BM25-Suche mit Status-Gewichtung; `--before-create` vor Dateianlage |
| scripts/hor_pass_check.py | ACTIVE | tooling/ci | ja | — | Hör-Pass-Log-Check: prüft Lauf-Logs gegen Fix-Marker und Regressions-Signaturen (GO/NO-GO-Ergänzung) |
| backend/core/inviting_sound_gate.py | ACTIVE | backend/core | ja | — | Hörordnung Ebene 4: Einladungs-Gate (Roughness/Sharpness-Fenster + Ermüdungs-Abbruch); nutzt zwicker_metrics; Sharpness (Bismarck-Näherung) hier implementiert (2026-08-23) |
| backend/core/residuum_masking.py | ACTIVE | backend/core | ja | — | Hörordnung Ebene 2: Residuum-basiertes Bark-Masking (Defekt-Anteil vs. maskierender Inhalt, ISO 11172-3 Spread); 3. Term im Salience-Blend (2026-08-23) |
| scripts/horordnung_calibration.py | ACTIVE | tooling/agents | ja | — | Kalibrierungs-Harness: psychoakustische Invarianten der Hörordnungs-Module gegen synthetische Referenz-Signale (Vorstufe Panel-Tests, 2026-08-23) |
| backend/core/librosa_bootstrap.py | ACTIVE | backend/core | nein | — | Vor-Gate-Bestand: bei Write-Gate-Einführung bereits gestaged (2026-08-22) |
| plugins/aero_plugin.py | ACTIVE | plugins/aero | nein | — | Vor-Gate-Bestand (2026-08-22) |
| scripts/corpus_fetcher.py | ACTIVE | tooling/corpus | nein | — | Vor-Gate-Bestand (2026-08-22) |
| scripts/dsp_benchmark.py | ACTIVE | tooling/benchmark | nein | — | Vor-Gate-Bestand (2026-08-22) |
| scripts/external_benchmark_ffmpeg.py | ACTIVE | tooling/benchmark | nein | — | Vor-Gate-Bestand (2026-08-22) |
| scripts/pitch_tracker_benchmark.py | ACTIVE | tooling/benchmark | nein | — | Vor-Gate-Bestand (2026-08-22) |
| cli/aurik_debug.py | ACTIVE | tooling/cli | ja | — | Standalone Debug-CLI (LEGACY_NON_RELEASE); Importe 2026-08-22 auf core.unified_restorer_v3 + pipeline_trace repariert |
| backend/core/regulator/regulator.py | ACTIVE | backend/core/regulator | ja | — | Kanonischer Regulator (importiert von pipeline.py + UV3) |
| backend/core/regulator/regulator_v8.py | DEPRECATED | backend/core/regulator | nein | backend/core/regulator/regulator.py | Suffix-Variante ohne Importe (repo_graph --duplicates Fund 2026-08-22); canonical = regulator.py |
| scripts/prepare_vocal_snr_round.py | ACTIVE | tooling/audit | nein | — | Vor-Gate-Bestand (2026-08-22) |
| scripts/venv_sitecustomize.py | ACTIVE | tooling/venv | nein | — | Vor-Gate-Bestand (2026-08-22) |

## Pflege-Regeln

- **Neue Datei anlegen:** zuerst `scripts/repo_search.py --before-create <pfad>`
  (kanonische Alternative? Namens-/Symbol-Ähnlichkeit?), dann hier eintragen
  (Status, Domain, Canonical, Ersetzt, Grund). Ohne Eintrag blockt
  `aurik-file-lifecycle` den Commit.
- **Ersetzen statt parallel bauen:** bestehende Implementierung auf
  DEPRECATED setzen mit „Ersetzt“ = neue Datei; die neue Datei als ACTIVE
  mit „Ersetzt“ = alte Datei eintragen. Git ist das Archiv — keine
  `-old`/`-v2`-Dateien neu anlegen.
- **Status-Änderung:** DEPRECATED/MIGRATING nur mit „Ersetzt“-Ziel;
  FORBIDDEN/ARCHIVED erst nachdem kein Importer mehr darauf zeigt
  (`scripts/repo_graph.py --check` meldet Verstöße).
- **Entfernen:** erst wenn die Datei im Repo gelöscht ist; der Eintrag kann
  als ARCHIVED (Ersetzt = Nachfolger) erhalten bleiben oder entfernt werden.
