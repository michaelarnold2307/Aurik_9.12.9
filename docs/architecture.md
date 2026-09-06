# Aurik Architektur

> §15.7: C4-Architekturdiagramme (Context, Container, Component).
> ⚠️ **Ergänzungsdokument**: Kanonische Architektur-Übersicht ist `architecture/ARCHITECTURE.md`
> (verlinkt über `README.md`/`INDEX.md`); bei Abweichung gilt diese.

## C4 Level 1: System Context

```mermaid
C4Context
    title System Context — Aurik 10

    Person(user, "Audio Engineer", "Restauriert historische Musikaufnahmen")
    System(aurik, "Aurik", "Intelligente Musik-Restaurierung")

    System_Ext(archive, "Internet Archive", "Public-Domain-Quellen")
    System_Ext(storage, "Dateisystem", "WAV/FLAC/MP3")

    Rel(user, aurik, "Restauriert Audio via", "CLI / Python API / REST")
    Rel(aurik, storage, "Liest/Schreibt", "WAV, FLAC, MP3")
    Rel(aurik, archive, "Optional: Lädt Referenz-Material", "HTTP")
```

## C4 Level 2: Container

```mermaid
C4Container
    title Container — Aurik 10

    Person(user, "Audio Engineer", "")

    Container_Boundary(aurik, "Aurik") {
        Container(cli, "CLI", "Python", "Kommandozeilen-Interface")
        Container(api, "REST API", "Python/Flask", "HTTP-API für GUI-Integration")
        Container(denker, "Denker", "Python", "Kognitive Orchestrierungsschicht")
        Container(backend, "Backend Core", "Python", "Phasen-Pipeline (69 Dateien), DSP, ML")
        Container(plugins, "Plugin System", "Python", "Plugin-SDK + 58 Plugins")
    }

    ContainerDb(models, "ONNX Models", "Dateisystem", "PANNS, Whisper, wav2vec2, RMVPE")

    Rel(user, cli, "Nutzt", "CLI")
    Rel(user, api, "Nutzt", "HTTP")
    Rel(cli, denker, "startet", "Python API")
    Rel(api, denker, "startet", "Python API")
    Rel(denker, backend, "orchestriert", "Python API")
    Rel(backend, models, "lädt", "ONNX Runtime")
    Rel(backend, plugins, "erweitert via", "Plugin API")
```

## C4 Level 3: Pipeline Component

```mermaid
C4Component
    title Component — Restaurierungspipeline (69 Phasen-Dateien)

    Container_Boundary(pipeline, "Pipeline") {
        Component(load, "Phase 01: Load", "Import & Validierung")
        Component(restore, "Phases 02-66", "Restaurierung (65 DSP/ML-Phasen)")
        Component(glue, "Phase 67: Glue", "Zusammenführung")
        Component(export, "Phase 68: Export", "Format-Optimierung & Ausgabe")

        Component(guard, "ErrorGuard", "Graceful Degradation")
        Component(goals, "Musical Goals", "Qualitäts-Gates (14 Ziele)")
        Component(mushra, "MUSHRA Estimator", "Objektive Qualitäts-Approximation")
    }

    ComponentDb(session_mgr, "Session Manager", "ONNX-Lifecycle")

    Rel(load, restore, "→", "np.ndarray")
    Rel(restore, glue, "→", "mit Metadaten")
    Rel(glue, export, "→", "finales Audio")

    Rel(guard, restore, "schützt", "Decorator")
    Rel(goals, restore, "prüft", "per Phase")
    Rel(mushra, export, "bewertet", "OQS")
    Rel(restore, session_mgr, "nutzt", "ONNX-Sessions")
```

## Schlüsselkonzepte

| Konzept | Beschreibung | Spec |
|---------|-------------|------|
| **Denker** | Kognitive Orchestrierung: Material-Erkennung → Pipeline-Auswahl → Phasen-Steuerung | §2.1 |
| **Phasen-Pipeline (69 Dateien)** | Sequenzielle Audio-Verarbeitung: Click Removal → Denoise → EQ → ... → Export | §2.2 |
| **Musical Goals** | 15 Qualitäts-Ziele (Spec 01) | §1.2 |
| **Bridge** | API-Schicht: Trennt Frontend von Backend Core (V01-Bypass-Verbot) | §8.1 |
| **ErrorGuard** | Graceful Degradation bei Phasenfehlern | §15.8 |
| **Plugin SDK** | ABC-basierte Plugin-Architektur für Drittentwickler | §15.6 |
