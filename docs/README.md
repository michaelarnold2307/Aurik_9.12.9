# Aurik 10 — Dokumentations-Übersicht

**Version:** 10.0.20 (version.py) | **Status:** ✅ Produktionsbereit | **Stand:** 2026-09-06

> Normativer Ist-Stand: `.github/copilot-instructions.md`, `.github/VERBOTEN.md`,
> `.github/instructions/` (hoerordnung + Domain-Regeln), `.github/specs/`
> (Index: `00_SPEC_INDEX.md`), `CHANGELOG.md`, `denker/README.md`.
> Konflikt-Regel und ID-Zitierdisziplin: siehe `AGENTS.md` (§1, §5).

---

## 📖 Quick Links

### Für Anwender

- **[Installations-Guide](guides/INSTALLATION.md)** — AppImage (Linux) / Installer (Windows)
- **[Konfigurations-Guide](guides/CONFIGURATION.md)** — Modi & Parameter
- **[Troubleshooting](guides/TROUBLESHOOTING.md)** — Problemlösung

### Für Entwickler

- **[KI-Agent Integration Guide](KI-AGENT-INTEGRATION-GUIDE.md)** — Regeln für KI-Agenten **(Pflicht!)**
- **[Python API Reference](api/PYTHON_API.md)** — API-Dokumentation
- **[Architecture Overview](architecture/ARCHITECTURE.md)** — Systemarchitektur (4 Schichten)
- **[Phases Overview](architecture/PHASES_OVERVIEW.md)** — 69 Phasen-Dateien (Phase 01–66 + Glue + Interface)
- **[Contributing Guide](development/CONTRIBUTING.md)** — Beitrag leisten
- **[Testing Guide](development/TESTING.md)** — Teststrategie und Qualitätssicherung

### Status & Fortschritt

- **[Project Status Report](PROJECT_STATUS.md)** — Projektstatus (Living Document)
- **[Aurik-10-Roadmap](AURIK_10_ROADMAP.md)** — Fortschritt & Meilensteine (historisch + aktuell)

📚 **[Vollständiger Dokumentations-Index](INDEX.md)**

---

## 🎯 Aurik 10 — Kennzahlen

| Metrik | Wert |
|---|---|
| **Version** | 10.0.20 (version.py, single source of truth) |
| **Tests** | 285+ Denker + 18.400+ gesamt |
| **Defekttypen** | 62/62 erkannt & gemappt |
| **Phasen** | 69 Phasen-Dateien (Phase 01–66, Glue Stage, Interface) |
| **Materialien** | 16 Typen auto-erkannt (+ `UNKNOWN`) |
| **SourceMediums** | 17 Trägermedien (Wax Cylinder … DAW-Limiter) |
| **Genres** | 19 Profile |
| **Musical Goals** | 15 (Spec 01) — u. a. Brillanz, Wärme, Natürlichkeit, Authentizität, Emotionalität, Transparenz, Bass-Kraft, Groove, Raumtiefe, Timbre-Authentizität, Tonales Zentrum, Mikro-Dynamik, Separation-Treue, Artikulation, Transienten-Energie |
| **Post-Processing** | 8-stufige wissenschaftliche Pipeline |
| **Hardware** | CPU + optionale AMD-GPU (ROCM/DirectML) |
| **Betrieb** | 100 % offline nach Installation |

## 🎛️ Zwei Restaurierungs-Modi

| Modus | Ziel | Strength |
| --- | --- | --- |
| **Restoration** | Originalgetreu, minimal-invasiv, historisch authentisch | Konservativ |
| **Studio 2026** | Modern, klar, kräftig — heutiger Referenzstandard | Aggressiv |

## 📦 Distribution

| Plattform | Format | Status |
| --- | --- | --- |
| **Linux** | AppImage (`.AppImage`) | ✅ |
| **Windows 10/11** | NSIS-Installer (`.exe`) | ✅ |

## 🖥️ GPU-Unterstützung

Aurik beschleunigt rechenintensive Phasen optional über GPU:

| Plattform | GPU-API | Treiber |
| --- | --- | --- |
| **Linux** | AMD ROCm 7.2.4 (PyTorch ROCm) | `amdgpu` + ROCm Runtime |
| **Windows 10/11** | AMD DirectML (ONNX Runtime) | Aktueller Adrenalin-Treiber |

GPU-Beschleunigung ist **optional** — CPU-Betrieb jederzeit möglich. Speedup: 2×–8× (phase-abhängig).
