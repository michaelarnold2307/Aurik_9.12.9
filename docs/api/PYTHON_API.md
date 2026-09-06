# Aurik 10 - Python API (Kanonischer Vertrag)

**Version:** 10.0.20  
**Stand:** 2026-09-06  
**Status:** RELEASE_MUST-konform

Diese Referenz beschreibt ausschliesslich den produktiven Desktop-Release-Pfad.

## Geltungsbereich

- Desktop-only (Linux AppImage, Windows 10/11)
- 100 % offline nach Installation
- Nur Mono/Stereo (mehr als 2 Kanaele werden nicht als eigener Produktpfad verarbeitet)
- Einziger produktiver Einstieg: Bridge + Denker

## Kanonische Aufrufkette

```text
Audio-Import  -> backend.api.bridge.get_load_audio_fn()
Voranalyse    -> backend.api.bridge.run_pre_analysis() genau einmal
Pipeline      -> get_aurik_denker_instance().denke(...)
Modus         -> exakt restoration oder studio2026
Export        -> export_guard() + validate_export_quality() + AudioExporter
Telemetry     -> metadata inkl. fail_reason/degradation_status/quality_gate_payload
```

## Minimalbeispiel (Release-konform)

```python
from backend.api.bridge import (
    get_load_audio_fn,
    run_pre_analysis,
    get_aurik_denker_instance,
)

# 1) Import ueber Bridge
load_audio = get_load_audio_fn()
audio, sr = load_audio("input.wav")

# 2) Voranalyse genau einmal
pre = run_pre_analysis(audio, sr)

# 3) Denker-Aufruf mit exakt zwei Modi
denker = get_aurik_denker_instance()
result = denker.denke(audio, sr, mode="restoration")

# 4) Export- und Qualitaetsgate laufen im Releasepfad verpflichtend
# (export_guard + validate_export_quality + AudioExporter)
print(result.metadata.get("quality_gate_payload", {}))
```

## Verfuegbare Modi

- `restoration`
- `studio2026`

Andere Modusnamen sind im Releasepfad nicht zulaessig.

## Qualitaets- und Sicherheitsgates

- `artifact_freedom < 0.95` => Veto
- Vokalpfad (`panns_singing >= 0.35`): VQI als Recovery-Trigger
- `export_guard()` ist vor Ausgabe verpflichtend

## Verbotene Release-Pfade

- Direkter `sf.read(path)` oder `librosa.load(path)` als Importvertrag
- Direkter `UnifiedRestorerV3.restore()`-Bypass ohne Bridge/Denker
- Eigene Exportpfade ohne `export_guard()`
- Dokumentierte manuelle Parameterpflicht im Endnutzer-Standardpfad

## Legacy-Hinweis

Historische v2-Beispiele sind kein Release-Vertrag. Falls noch in Alt-Dokumenten referenziert,
gelten sie als `LEGACY_NON_RELEASE`.
