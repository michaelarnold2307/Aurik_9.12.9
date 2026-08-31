# AGENTS.md — Universeller Einstiegspunkt für alle Agenten

Projekt: **Aurik 10 — Weltklasse-Audio-Restaurierung** — psychoakustisch präzise
Musikwiederherstellung mit deterministischer Reproduzierbarkeit und natürlichem
Wohlklang für das menschliche Ohr.

> Diese Datei ist der verbindliche Einstieg für jeden Agenten (Kun, Claude Code,
> Codex, GitHub Copilot, Cursor, …). Sie definiert keine eigenen Regeln, sondern
> routet auf die normativen Dokumente und legt die Konflikt-Auflösung fest.
> Arbeitssprache: Deutsch.

## 1. Normative Kette — in dieser Reihenfolge maßgeblich

1. **`.github/copilot-instructions.md`** — Eiserne Regeln: §I GEBOTE (G1–G9),
   §II VERBOTE (V1–V9), §III DSP-Spezialregeln, §IV Export-Reihenfolge,
   §V CD-Rauschprofil-Modell, §VI Startup-Vertrag (§v10.305),
   `[RELEASE_MUST]`-Anforderungen, §0a verbotene Phasen, Performance-Budget,
   Bug-Klassen. Dokumentierte Vorrangregel: **Spec > Code > Kommentar**.
   Wird von CI und Tests geparst.
2. **`.github/VERBOTEN.md`** — Normative Quelle der Linter-Regeln **V01–V52**
   (Teil A: Grundregeln, Teil B: Anti-Patterns mit Produktions-Evidenz).
3. **`.github/instructions/hoerordnung.instructions.md`** — Psychoakustische
   Wahrheits-Ordnung. Normative Spitze für **Hör-Entscheidungen**: Hör-Invarianten
   → Audibility (Maskierungsschwelle statt Mess-Null) → lexikografische
   Wohlklang-Ordnung → Einladungs-Gate, plus Konfliktregel (Metriken sind Zeugen,
   die Hör-Instanz entscheidet — aber nie gegen Ebene 1). Regelt nur den
   Entscheidungsfluss; die Berechnung der Messgrößen bleibt bei den unten
   genannten Domain-Regeln.
4. **`.github/instructions/`** — Domain-Regeln: `pipeline.instructions.md` (UV3,
   größte Datei), `phases.instructions.md`, `dsp.instructions.md`,
   `musical_goals.instructions.md`, `tests.instructions.md`.
5. **`.github/specs/`** — Nummerierte Specs 01–22 plus versionierte v10.xx
   (92 Dateien). Änderungen hier lösen den CI-Evidenzblock-Zwang aus (§4).
6. **`.github/GEBOTE.md`** — Katalog §G1–§G187 (Kategorien I–XXIV und XI-b). Hat
   Referenzcharakter: Der Pre-Commit-Verifier prüft nur eine hartkodierte
   Teilmenge. Der Dateikopf beansprucht Vorrang vor Specs — das erzwingt CI
   aber nicht. Bei Konflikten: nicht stillschweigend entscheiden, sondern im
   PR-Evidenzblock dokumentieren.

## 2. Maschinen-Wahrheit — was tatsächlich enforced wird

Die Markdown-Dokumente beschreiben die Regeln; die Gates sind:

- **Pre-Commit** (`.pre-commit-config.yaml`): `aurik-compliance`,
  `aurik-verboten-linter` (fail-closed), `gebote-verifier`,
  `aurik-bug-prevention` (`.agents/skills/bug-prevention/scan_anti_patterns.py`),
  `aurik-ruff-critical-static-gate` (F821/F601/B009/I001), `aurik-unit-smoke`
  (tests/unit, maxfail=3), `aurik-id-registry` (fail-closed: R1 unbekannte
  IDs, R2 nackte Ambiguitäts-Zitate), `aurik-file-lifecycle` (Write-Gate:
  neue Code-Dateien ⇒ Eintrag in `.github/FILE_REGISTRY.md`),
  `aurik-symbol-duplicates` (`scripts/repo_graph.py --duplicates`),
  `aurik-horordnung-calibration` (psychoakustische Invarianten,
  Hörordnung §8a) und rund 25 weitere `aurik-*`-Guards.
- **CI** (`.github/workflows/ci-lite.yml`, `nightly-quality.yml`,
  `solo-release-gate.yml`): `scripts/compliance_check.py` (R01–R18),
  `scripts/release_must_coverage_check.py` (jeder `[RELEASE_MUST]`-Header in
  copilot-instructions.md braucht einen Test), `scripts/spec_drift_check.py`
  (Hash-Drift für copilot-instructions.md, Specs 01–08, ID-Registry,
  FILE_REGISTRY, Kollisions-Karte, AGENTS.md).
- **Regeländerung ⇒ Skript nachziehen**: `compliance_check.py`,
  `aurik_verboten_linter.py` und `gebote_verifier.py` haben ihre Regeln
  **hartkodiert**; sie parsen die MD-Dateien nicht. Änderst du ein Gebot oder
  Verbot im Markdown, ändere parallel das zugehörige Skript — sonst prüft CI
  eine andere Regel als dokumentiert.

## 3. Nicht verhandelbar — Schnell-Referenz

Details immer in der normativen Kette (§1) nachlesen.

- **Bridge-Verbot (§V4)**: UI/Frontend (Aurik10, CLI) importiert `backend/core/`
  nie direkt — nur über `backend/api/bridge.py`. Denker-Schicht (`denker/`)
  ausgenommen.
- **Neue Datei anlegen (Write-Gate)**: vor dem Anlegen
  `python scripts/repo_search.py --before-create <pfad>` prüfen (kanonische
  Alternative? Namens-/Symbol-Ähnlichkeit?); danach Eintrag in
  `.github/FILE_REGISTRY.md` (Status, Domain, Canonical, Ersetzt, Grund).
  Ohne Eintrag blockt `aurik-file-lifecycle` den Commit. Task-Ledger:
  `python scripts/change_ledger.py snapshot` → `TASK_CHANGES.md` (CI
  erzwingt Abdeckung im PR).
- **Determinismus (§G5)**: gleicher Input + gleiche Version ⇒ bit-identischer
  Output. Seeds pro Session; kein `time.time()` in Entscheidungslogik.
- **Dither (§V5)**: bei bit_depth < 32 immer POW-r Type 3 (primär) oder TPDF
  (Fallback); kein nacktes `astype(np.int16)`.
- **Silent-Failure-Verbot (§V6)**: jeder ML→DSP-Fallback mit
  `logger.warning()` + Begründung.
- **Workaround-Verbot (§V7)**: Ursache statt Symptom; keine phasen-individuellen
  Schwellwerte; Stärke-Entscheidungen zentral über `global_scalar`.
- **Song-Isolation (§V8/§G1)**: alle Stateful-Module (Circuit-Breaker, Caches,
  Lernparameter) pro Song zurücksetzen.
- **DSP (§III)**: Soft-Knee (6 dB, 200 ms Hanning) statt Hard-Clamp;
  PIM-first, RLP-last; Glue Stage immer als vorletzte Phase; 62 DefectTypes
  (keine ad-hoc-Neuen); NaN/Inf-Schutz in jeder Phase (§0a); Logger-Pflicht
  (`logging.getLogger(__name__)`) in jedem Modul.
- **Verbotene Phasen (§0a)**: `phase_21_exciter`,
  `phase_35_multiband_compression`, `phase_42_vocal_enhancement` sind im
  Restoration-Modus verboten.
- **GUI/Startup (§VI)**: `t()` für alle benutzersichtbaren Strings;
  Launch-Skripte mit `python3 -B`; GPU-Detection im Hauptthread vor
  `ModernMainWindow`.
- **Bekannter Ist-Zustand (Rev. 2026-08-16)**: V01/V08-ERROR-Verstöße
  existieren im Production-Code nicht mehr (gemessen: Linter `--ci
  --errors-only` auf `backend/` → clean). Der ERROR-Gate-Test
  (`test_backend_no_error_violations`) läuft wieder unskipped. Vendored-
  Drittanbieter-Code (`plugins/_vendor_*`) ist im Linter via `SKIP_DIRS`
  ausgenommen (unverändert kopiert, MIT, LICENSE beiliegend).

## 4. PR-/CI-Vertrag — ohne den wird nichts gemerged

- Änderungen an `.github/specs/` ⇒ PR-Body MUSS die Abschnitte
  `## Evidenzblock`, `## Seed`, `## 95 %-CI`, `## Maintainer Sign-off`
  enthalten (Vorlage: `.github/pull_request_template.md`).
- Conventional Commits (Pre-Commit-erzwungen): feat/fix/docs/style/refactor/
  perf/test/chore/spec/build/ci.
- Spec-Referenzen (§G1, §V1, …) sind in Code-Kommentaren zu verwenden (§G9).

## 5. ID-Zitierdisziplin — Pflicht

Die §G-/§V-IDs sind über die Dokumente hinweg **nicht eindeutig**. Beispiel:
§G4 ist in copilot-instructions.md die CD-Rauschprofil-Pflicht, in GEBOTE.md
die Ghost-Echo-Freiheit; GEBOTE.md hatte doppelte Kategorie-Nummern
(XVIII, XXII) und Notlösungs-IDs (§SC-G71) — seit der Phase-1-Bereinigung
sind die Startup-Regeln §G173–§G182 (Kategorie XXIV) und XI-b §G183–§G187;
die Aliasse §SC-G71–§SC-G80 bleiben auflösbar. Zitiere deshalb immer mit Quelle:
**„§G4 (copilot-instructions.md)“** oder **„§G71 (GEBOTE.md, Kategorie IX)“** —
nie nackte IDs. Maschinelle Unterstützung: Kanonische Registry
`.github/ID_REGISTRY.md`, Validator `scripts/id_registry_check.py`
(fail-closed im Pre-Commit; `--fix` für mechanische Qualifikation),
Kollisions-Karte und Bereinigungsplan `docs/ID_COLLISION_MAP.md`.

## 6. Veraltet oder doppelt — nicht als normativ behandeln

- `.github/VERBOTE.md` — trägt einen Veraltet-Banner; normativ ist nur
  `VERBOTEN.md` (V01–V52). Die §V-IDs hier kollidieren mit anderen Dokumenten.
- `.github/GEBOTEN.md` — trägt einen Referenz-Banner; von keinem Tool enforced.
- `.agents/skills/*` — überwiegend Stubs oder veraltete Kopien von
  Root-Dokumenten; `claude/SKILL.md` und `spec/SKILL.md` verweisen jetzt auf
  `CLAUDE.md` bzw. die normative Kette. Funktionale Ausnahme:
  `bug-prevention/scan_anti_patterns.py` (im Pre-Commit verdrahtet).
- `.github/agents/*.agent.md` — Rollen-Hinweise (Copilot Custom Agents);
  Kennzahlen aktualisiert, aber von keinem Tool referenziert.
- `.github/GOVERNANCE.md` — Prozess-Hintergrund; veralteter §5.1-Verweis
  entfernt.

## 7. Aufgabenbezogene Lektüre

- Hör-Entscheidungen & Zielkonflikte: `.github/instructions/hoerordnung.instructions.md`
  (Rollen-Spitze über den Goal-Regeln)
- UV3-Pipeline: `.github/instructions/pipeline.instructions.md` +
  `.github/specs/02_pipeline_architecture.md`
- Phasen: `phases.instructions.md` + `06_phases_system.md`
- DSP: `dsp.instructions.md` + `04_dsp_standards.md`
- Musical Goals: `musical_goals.instructions.md` + `01_musical_goals.md`
- Tests: `tests.instructions.md` + `07_quality_and_tests.md`
- Startup/GUI/Threading: `.github/specs/v10.305_startup_integration_contract.md`
  - §VI der copilot-instructions
- Neue Spec anlegen: `.github/specs/XX_measure_template.md`
- Hör-Tests / GO-NO-GO: `docs/guides/GO_NO_GO_DECISION_PROTOCOL.md` (beratend)
- Repo-Karte & Lifecycle: `.github/FILE_REGISTRY.md` +
  `scripts/repo_graph.py` (`--write-json` → `.github/repo_graph.json`,
  `--check`, `--duplicates`)
- Suche vor Dateianlage: `scripts/repo_search.py --before-create <pfad>`
- Task-Ledger: `scripts/change_ledger.py snapshot` → `TASK_CHANGES.md`

## 8. Agent-spezifische Hinweise

- **Claude Code**: liest zusätzlich `CLAUDE.md` (Projekt-Kontext +
  v10-Invarianten). Bei Widerspruch gilt die normative Kette aus §1.
- **GitHub Copilot**: nutzt `.github/copilot-instructions.md` automatisch
  (GitHub-Konvention) — dieselbe normative Spitze wie §1.
- **Codex, Cursor und andere**: diese Datei ist der Einstieg; dann wie §1
  routen.
