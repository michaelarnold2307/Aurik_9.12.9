# Aurik 10.0.8 — Dokumentations-Übersicht

**Version:** 10.0.8 | **Status:** ✅ Produktionsbereit | **Stand:** Juli 2026

> Normativer Ist-Stand: `SPEC.md`, `.github/specs/`, `CHANGELOG.md`, `denker/README.md`.

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
- **[Phases Overview](architecture/PHASES_OVERVIEW.md)** — 68-Phasen-Pipeline
- **[Contributing Guide](development/CONTRIBUTING.md)** — Beitrag leisten
- **[Testing Guide](development/TESTING.md)** — Teststrategie und Qualitätssicherung

### Status & Fortschritt

- **[Project Status Report](PROJECT_STATUS.md)** — Projektstatus (Living Document)
- **[Roadmap](aurik9_roadmap.md)** — Zukunftspläne

📚 **[Vollständiger Dokumentations-Index](INDEX.md)**

---

## 🎯 Aurik 10.0.8 — Kennzahlen

| Metrik | Wert |
|---|---|
| **Version** | 10.0.8 (Weltspitze) |
| **Tests** | 285+ Denker + 18.400+ gesamt |
| **Defekttypen** | 62/62 erkannt & gemappt |
| **Phasen** | 68 (Phase 01–66 + Vocal Repair + Glue Stage) |
| **Materialien** | 16 Typen auto-erkannt |
| **SourceMediums** | 17 Trägermedien (Wax Cylinder … DAW-Limiter) |
| **Genres** | 19 Profile |
| **Musical Goals** | 14 (Brillanz, Wärme, Natürlichkeit, Authentizität, Emotionalität, Transparenz, Bass-Kraft, Groove, Raumtiefe, Timbre-Authentizität, Tonales Zentrum, Mikro-Dynamik, Separation-Treue, Artikulation) |
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
