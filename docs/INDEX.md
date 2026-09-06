# Aurik 10 — Dokumentationsindex

Offizielle Dokumentation für Aurik 10 (Version 10.0.20, Stand 2026-09-06).

## Normativer Vorrang

Bei Abweichungen zwischen Einzel-Dokumenten und Spezifikation gilt die
normative Kette aus `AGENTS.md` §1 — in dieser Reihenfolge maßgeblich:

1. `.github/copilot-instructions.md` (Gebote/Verbote, DSP-Regeln, Startup-Vertrag)
2. `.github/VERBOTEN.md` (V01–V52, Linter-Quelle)
3. `.github/instructions/` — `hoerordnung.instructions.md` (Hör-Entscheidungen)
   und Domain-Regeln (pipeline, phases, dsp, musical_goals, tests)
4. `.github/specs/` (01–22 + versionierte v10.xx; kanonischer Index: `00_SPEC_INDEX.md`)
5. `.github/GEBOTE.md` (Referenzkatalog; keine Tool-Enforcement)

## Kernfakten

- Phasen: 69 Phasen-Dateien (Phase 01–66, Glue Stage, Interface)
- Musical Goals: 15 (Spec 01)
- DetectionTypes: 62
- Kausal-Ursachen: 62
- Tests: ~18.400

## Release-Must-Leitplanken

- Desktop-only (Linux AppImage, Windows 10/11)
- 100 % offline nach Installation
- Endnutzer-Workflow: One-Button mit Moduswahl `Restoration` oder `Studio 2026`
- Kanonischer Vertrag: Bridge -> AurikDenker.denke -> export_guard

## Startpunkte

### Fuer Anwender

- [Installations-Guide](guides/INSTALLATION.md)
- [Benutzerhandbuch](guides/USER_GUIDE.md)
- [Konfigurations-Guide](guides/CONFIGURATION.md)
- [Troubleshooting](guides/TROUBLESHOOTING.md)

### Fuer Entwicklung und Audit

- [KI-Agent Integration Guide](KI-AGENT-INTEGRATION-GUIDE.md)
- [Python API](api/PYTHON_API.md)
- [Architektur-Ueberblick](architecture/ARCHITECTURE.md)
- [Phasen-Ueberblick](architecture/PHASES_OVERVIEW.md)
- [Pipeline-Analyse](architecture/PIPELINE_FLOW_ANALYSIS.md)
- [CI/CD](CI_CD.md)
- [Testing Guide (inkl. Gate-Runbook)](development/TESTING.md)
- [Worldclass Change Acceptance Protocol](archive/WORLDCLASS_CHANGE_ACCEPTANCE_PROTOCOL.md)
- [Spec-Evidenzberichte](reports/spec_evidence/README.md)
- [Historisches Archiv](archive/README.md) — UAT-Reports, Playbooks, alte Roadmaps

## Kanonischer Vertragsfluss (Kurz)

```text
Import (Bridge) -> Voranalyse -> AurikDenker.denke -> Holistic Gates -> export_guard
```

## Legacy-Regel

Historische Dokumente mit v2-/Server-/Docker-Produktpfaden sind nur als
`LEGACY_NON_RELEASE` zu betrachten, sofern sie nicht auf den kanonischen Vertrag
aktualisiert wurden.
