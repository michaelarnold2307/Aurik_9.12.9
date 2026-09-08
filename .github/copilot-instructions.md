# Aurik 10 — Copilot Instructions & Eiserne Regeln

> Normativer Ist-Stand. Jede Code-Änderung MUSS diese Regeln einhalten.
> Bei Widerspruch zwischen Specs und Code gilt: Spec > Code > Kommentar.

---

## §I  GEBOTE (Muss-Vorschriften)

Jedes GEBOT ist eine nicht verhandelbare Vorschrift. Verstöße sind Build-Fehler.

### §G1 — Kategorie I: Individuelle Song-Maximierung
**Jeder importierte Song wird individuell maximal für das menschliche Ohr verbessert.**
- Kein Song beeinflusst die Verarbeitung eines anderen Songs.
- Alle Stateful-Module (Circuit-Breaker, Caches, Learned Parameters) werden pro Song zurückgesetzt.
- Die Verarbeitungsintensität wird aus den spezifischen Defekten DIESES Songs abgeleitet, nicht aus globalen Statistiken.

### §G2 — Kategorie I: Vollständige Defektbehebung
**Defekte werden über den gesamten Song präzise und vollständig behoben.**
- Keine "Checkpoint"-Strategie – jeder Frame, jedes Sample wird geprüft.
- Defekterkennung läuft auf dem gesamten Signal, nicht nur an Stichproben.
- Per-Chunk-Verarbeitung mit Overlap für nahtlose Übergänge.

### §G3 — Kategorie I: Natürlicher Wohlklang
**Die bearbeitete Aufnahme klingt für das menschliche Ohr natürlich und unverfälscht.**
- Keine Verzerrungen des Gesangs (Formanten, Vibrato, Phrasierung bleiben erhalten).
- Keine "Ghost-Echos" durch Phasenverschiebungen oder harte Schnittkanten.
- Crossfades: 200 ms Hanning (Minimum), Cosine-Interpolation für Parameter.

### §G4 — CD-Rauschprofil-Pflicht
**Jeder Export (Restoration + Studio 2026) MUSS ein CD-charakteristisches Rauschprofil enthalten, unabhängig vom Quellmaterial.**
- Das Rauschprofil wird psychoakustisch maskiert: Nur dort appliziert, wo das menschliche Ohr es wahrnimmt.
- In lauten Passagen (Signal > Maskierungsschwelle) wird KEIN Rauschprofil hinzugefügt.
- In stillen Passagen, Pausen und Ausklängen wird das CD-Rauschprofil mit dem natürlichen CD-Rauschspektrum eingebracht.
- Das Rauschprofil simuliert das thermische Rauschen eines CD-Wandlers (~−96 dBFS, spektral flach bis 20 kHz).

### §G5 — Kategorie II: Deterministische Reproduzierbarkeit
**Derselbe Input + dieselbe Aurik-Version = derselbe Output (Bit-identisch).**
- Alle Zufallsgeneratoren mit fixem Seed (pro Session, nicht global).
- Keine zeitabhängigen Entscheidungen (kein `time.time()` in Entscheidungslogik).
- Export-Dither verwendet deterministischen Seed pro Export.

### §G6 — Kategorie II: Psychoakustische Präzision
**Alle Signaländerungen werden gegen das menschliche Gehör validiert.**
- Frequenzgang-Änderungen >0.5 dB nur mit ERB-bandbewerteter Metrik.
- Dynamik-Änderungen >1 dB nur mit Zwicker-Lautheitsmodell (ISO 532B).
- Rauschprofil-Injektion folgt dem ISO 389-7 Hörschwellenmodell.

### §G7 — Kategorie III: Chirurgische Defektbehandlung
**Jeder Defekt wird mit dem exakt richtigen Werkzeug in der exakt richtigen Intensität behandelt.**
- 62 DefectTypes mit individuellem Phase-Mapping.
- Material-Sensitivity pro Defekttyp.
- Kein "One-Size-Fits-All" — die Stärke wird pro Defektinstanz kalibriert.

### §G8 — Kategorie III: Transparenz
**Jede Entscheidung ist nachvollziehbar.**
- Alle ML→DSP-Fallbacks werden mit `logger.warning()` protokolliert.
- Jede Parameteränderung wird im Audit-Log vermerkt.
- Export-Metadaten enthalten Aurik-Version + Verarbeitungskette.

### §G9 — Projektweite Konsistenz
**Alle Maßnahmen müssen im gesamten Projekt konsistent sein.**
- Änderungen an Shared Contracts (Bridge-API, Enums) → alle Consumer aktualisieren.
- Spec-Referenzen (§G1, §V1 etc.) sind in Code-Kommentaren zu verwenden.
- Keine Ad-hoc-Parameter — alle Konstanten über Config oder Enum.
- **Canonical Contract Drift Gate**: GUI, CLI, REST und Batch MÜSSEN denselben
  Pfad durchlaufen (Import → Pre-Analysis → `AurikDenker.denke()` → Bridge-/
  Exporter-Export) und dieselben Bridge-Resolver (z. B. `normalize_user_mode()`)
  verwenden. Kein Parallelpfad darf eigene Mode-/Alias-Mappings definieren, die
  vom kanonischen Bridge-Vertrag abweichen (vgl. `tests/normative/test_canonical_contract_drift_gate.py`).

---

## §II VERBOTE (Muss-Nicht-Vorschriften)

Jedes VERBOT definiert eine unzulässige Handlung. Verstöße sind Build-Fehler.

### §V1 — Vocal-Distortion-Verbot
**VERBOTEN: Jegliche Verzerrung des Gesangs.**
- Keine Formanten-Verschiebung >5%.
- Keine Vibrato-Unterdrückung oder -Modifikation.
- Keine hörbare Änderung der Stimmcharakteristik.
- Phase 42 (Vocal Enhancement) arbeitet ausschließlich additiv (keine subtraktiven Eingriffe in existierende Frequenzanteile).

### §V2 — Ghost-Echo-Verbot
**VERBOTEN: Künstliche Echos oder Nachhall-Fahnen.**
- Keine harten Schnittkanten ohne Crossfade (Minimum: 200 ms Hanning).
- STCG (Stereo Temporal Coherence Guard) verhindert L/R-Phasenverschiebungen.
- Feedback-Chain Phasen laufen mit identischem Crossfade wie Haupt-Pipeline.

### §V3 — Rauschprofil-Full-Song-Verbot
**VERBOTEN: Das CD-Rauschprofil über den kompletten Song zu legen.**
- Das Rauschprofil wird psychoakustisch maskiert appliziert (§G4).
- In hörbaren Passagen (Musik, Gesang) wird das Rauschprofil NICHT hinzugefügt.
- Nur in Lücken, Pausen, Ausklängen und unterhalb der Maskierungsschwelle.
- Verstoß führt zu hörbarem "Rauschteppich" über dem gesamten Signal — das ist unzulässig.

### §V4 — Bridge-Bypass-Verbot
**VERBOTEN: Direkter Import von `backend/core/` aus UI-/Frontend-Code.**
- Aurik10, CLI, GUI: nur über `backend/api/bridge.py`.
- Denker-Schicht (`denker/`) ist von diesem Verbot ausgenommen.

### §V5 — Truncation-ohne-Dither-Verbot
**VERBOTEN: Integer-Quantisierung ohne Dithering.**
- Bei bit_depth < 32 MUSS POW-r Type 3 Dither (primär) oder TPDF (Fallback) angewandt werden.
- Kein einfaches `astype(np.int16)` ohne vorheriges Dither.

### §V6 — Silent-Failure-Verbot
**VERBOTEN: ML→DSP-Fallbacks ohne `logger.warning()`.**
- Jeder Fallback MUSS mit Begründung protokolliert werden.
- Kein stilles Degradieren der Qualität.

### §V7 — Workaround-Verbot
**VERBOTEN: Symptombehandlung statt Ursachenbehebung.**
- Keine phasen-individuellen Schwellwerte als Workaround.
- Alle Stärke-Entscheidungen fließen zentral über `global_scalar`.

### §V8 — Song-Cross-Contamination-Verbot
**VERBOTEN: Zustand eines Songs beeinflusst die Verarbeitung eines anderen.**
- Alle global/module-level Zustände MÜSSEN pro Song zurückgesetzt werden.
- Circuit-Breaker, Caches, Lernparameter: Reset bei neuem Song (§G1).

### §V9 — Rauschprofil-Quellmaterial-Kopie-Verbot
**VERBOTEN: Das Rauschprofil aus dem Quellmaterial zu extrahieren und wieder einzufügen.**
- Das CD-Rauschprofil MUSS generiert werden, nicht kopiert.
- Das Quellrauschen (z.B. Bandrauschen, Vinyl-Knistern) ist ein DEFEKT und wird entfernt.
- Nach der Defektentfernung wird das saubere CD-Rauschprofil frisch generiert und psychoakustisch maskiert appliziert.

---

## §III DSP-Spezialregeln

1. **Soft-Knee-Gate**: `apply_musical_gain_envelope()` mit Sigmoid-Soft-Knee (6 dB), 200 ms Hanning-Crossfade. KEIN Hard-Clamp.
2. **PIM-first**: Vor jedem Phasen-Loop PIM-Intensitäts-Map berechnen und in `restoration_context` speichern.
3. **RLP-last**: Nach jedem Phasen-Loop RLP ausführen. Korrekturen nur bei objektiver Verbesserung übernehmen.
4. **Glue Stage**: Läuft in ALLEN Modi als vorletzte Phase.
5. **62 DefectTypes**: Keine willkürlichen neuen DefectTypes ohne Phase-Mapping und Material-Sensitivity.
6. **NaN/Inf-Schutz**: Jede Phase MUSS `np.nan_to_num()` oder `np.isfinite()` auf Ausgabe-Audio anwenden (§0a).
7. **Logger-Pflicht**: Jede Python-Datei mit `logger` MUSS `import logging` und `logger = logging.getLogger(__name__)` definieren.
8. **POW-r Type 3 Dither**: Primäres Dithering-Verfahren. Psychoakustisch optimiert für 48 kHz / 16-bit. 24-bit: reduziert wahrgenommenen Noise Floor ≥ 14 dB unter TPDF.

---

## §IV Export-Pipeline-Reihenfolge

```
1. _export_guard()          — NaN/Inf-Bereinigung + True-Peak-Schutz
2. _export_nuance_guard()   — Subtile perzeptuelle Politur (non-destructive)
3. cd_noise_profile_inject() — CD-Rauschprofil psychoakustisch maskiert (§G4, §V3, §V9)
4. apply_dither()           — POW-r Type 3 / TPDF (§V5)
5. Atomic write             — .tmp → os.replace
6. Metadata transfer        — Aurik Provenance + Carrier-Chain
```

---

## §V Modellierung des CD-Rauschprofils

Das CD-Rauschprofil simuliert das thermische Rauschen eines 16-bit CD-Wandlers:

- **Spektrum:** Flach von 20 Hz bis 20 kHz (weißes Rauschen)
- **Pegel:** −96 dBFS (16-bit theoretischer Noise Floor) mit Dither-Shaping Anhebung auf ~−90 dBFS oberhalb 10 kHz
- **Charakteristik:** Additives, unkorreliertes Rauschen pro Kanal
- **Psychoakustische Maske:** ERB-bandweise Berechnung der Maskierungsschwelle nach ISO 389-7. Nur Zeit-Frequenz-Zellen unterhalb der Schwelle erhalten Rauschen.
- **Einblendung:** 500 ms Cosine-Fade-In/Out an Übergängen zwischen maskierten und unmaskierten Regionen.

---

## §VI Referenzen

- Spezifikationen: `.github/specs/`
- Instruktionen: `.github/instructions/`
- GEBOTE/VERBOTE: dieses Dokument (§I, §II)
- DSP-Spezialregeln: §III
- Export-Pipeline: §IV
- CD-Rauschprofil: §V
- **Startup & Kommunikation: §VI — §v10.305 Startup-Integrations-Vertrag**

### §VI — Startup-Integrations-Vertrag (§v10.305)

1. **GPU-Detection im Hauptthread**: `get_ml_device_manager()` VOR `ModernMainWindow` aufrufen. Kein `torch.zeros("cuda")`.
2. **Warmup ohne GPU**: `warmup_models_background()` darf `get_ml_device_manager()` NICHT aufrufen.
3. **Unified Progress**: `_sync_unified_progress()` ist die EINZIGE Quelle für beide Fortschrittsbalken.
4. **Kontextbewusste Kommunikation**: Jeder Song bekommt individuelle Statusmeldungen via `_build_context_status()`.
5. **i18n-Pflicht**: Jeder benutzersichtbare String MUSS `t()` verwenden.
6. **Cache-Safety**: `python3 -B` in allen Launch-Skripten.
7. **Lock-Disziplin**: Kein `import` während `with lock:`.
8. **Event-Garantie**: `threading.Event.set()` in `finally`.
9. **Plugin-Namen-Validierung**: `_failed > 0` nach Warmup → Warning.
10. **Startup-Smoke-Test**: `tests/test_startup_smoke.py` muss bestehen.

---

*Letzte Änderung: 2026-07-30 — v10.0.14: §v10.305 Startup-Integration, Unified Progress, Context-Aware Communication*

---

## Linter-Codes Referenz (normativ, aus VERBOTEN.md)

Jede Session MUSS diese Codes kennen (vollständige Liste der Linter-Referenz-Tabelle):

| V01 | V02 | V03 | V04 | V05 | V08 | V09 | V11 | V12 | V13 | V27 | V29 | V31 | V32 | V33 | V39 | V40 | V41 | V42 | V43 | V44 | V45 | V46 | V47 | V48 | V49 | V50 | V51 | V52 | V53 | V54 | V55 | V56 | V57 | V58 | V70 | V71 | V72 | V73 | V74 | V75 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## [RELEASE_MUST] Autonomer Magic-Button-Betrieb

Der Magic-Button (One-Button-Restauration) MUSS im autonomen Betrieb folgende
Kette ohne Benutzer-Interaktion durchlaufen: Preflight → DefectScan → Phasen-Plan →
Restauration → MQA-Gate → Export-Gate → GUI-Narrativ. Jede Gate-Entscheidung MUSS
im Narrativ begründet werden (§G155).

## [RELEASE_MUST] Strength-Envelope-Nichtdegeneration (v10.0.x)

Der Strength-Envelope (§2.71) MUSS bei vorhandenen Defect-Locations eine reale
Ortsvarianz aufweisen (σ > 0 und μ deutlich über dem Floor). Ein degenerierter
Envelope (μ=0.060 σ=0.000 — Produktionsbefund 2026-09-07: B3-Phase-2 Early-Merge
überschrieb ENUM-Scores durch 0.06-Stubs und löschte alle Locations) erzeugt eine
No-Op-Kaskade über alle Phasen und ist ein RELEASE-BLOCKER. Die komplette Kette
Merge → Locations-Extraktion → compute_strength_envelope MUSS als Regressionstest
abgesichert sein; Phasen-Stärken unterhalb des Floors bei vorhandenem
Reparatur-Kontext sind ebenso zu behandeln.

## §0a Normativ verbotene Phasen (Restoration-Guard)

Folgende Phasen sind im Restoration-Modus normativ VERBOTEN (§0a/UV3-Forbidden-Set):

- `phase_21_exciter`
- `phase_35_multiband_compression`
- `phase_42_vocal_enhancement`

## Performance-Budget (pro Minute Audio, synchron zu Spec 07 §9)

| Operation | Limit / Minute Audio |
| --- | --- |
| DefectScanner | <= **4 s** |
| Phase-Pipeline gesamt | <= **240 s** |
| FeedbackChain (alle Iterationen) | <= **120 s** |
| ExcellenceOptimizer | <= **60 s** |
| RestorabilityEstimator | <= **5 s** |
| Export (FLAC 24-bit) | <= 10 s |

Die Budget-Tabelle wird maschinell je Zelle von
`scripts/benchmark_effizienz_matrix.py --ci --enforce-budget` geprüft;
Verletzungen werden im Ergebnis-JSON unter dem Schlüssel `budget_violations`
gemeldet und führen im CI-Modus zu Exit 1. Die Pipeline reicht dazu seit
v10.0.20 reale Per-Operation-Timings als `metadata["pipeline_budget_timings"]`
nach außen; fehlende Timings werden als `null` dokumentiert (nicht geschätzt).
Mit `--repeats N` (deterministische Seed-Folge `AURIK_MASTER_SEED = 42+i`,
§G5) liefert der Harness echte Stichproben für `--bootstrap-ci`.

## Bug-Klassen (normativ, synchron zu Spec 10)

| Klasse | Bedeutung | Priorität |
| --- | --- | --- |
| R-BLOCKER | Korrektheit-kritisch: Export-Fehler, Crash, Daten-Verlust, Clipping-Artefakt | Sofort (P0) |
| AUDIO-QUALITY | Hörbar schlechteres Ergebnis als möglich | Nächster Commit (P1) |
| SPEC-GAP | Spec-Anforderung implementiert aber nicht getestet oder nur partiell | P2 |
| TYPE-SAFETY | mypy-Fehler (no-any-return, override, etc.) | P3 |
| TEST-DESIGN | Test-Assertion bricht durch nichtlineare Guards (deterministisch) | P2 |
