# Spec 24: ML-Fallback-Architektur — Kein stiller Ausfall mehr

> **Version:** Aurik 10.0.24 · **Scope:** Systemische Stabilität
> **Status:** In Umsetzung
> **Erstellt:** 2026-08-16 · **Abgeschlossen:** —

## Prämisse

ROCm/numba-Defekte ließen ML-Pfade im Betrieb still ausfallen: PANNs-Tags leer
(`PANNs tags: {}`), Genre-CLAP auf Fallback-Konstante 0.350, Readiness-Selbsttest
CRITICAL („NIEMALS als bereit erkannt"), EraClassifier-Traceback-Spam. Zusätzlich
verletzte der §3.0-CrossPhase-Consensus-Cap die §v10.53-Invariante (explizite
Stärke wurde überschrieben). Ziel: EINE kanonische Fallback-Architektur, die
jeden ML→CPU/DSP-Ausfall sichtbar (§V6) und hörqualitäts-neutral macht.

## Maßnahme

Sechs kanonische Muster, je an EINEM zentralen Ort — Plugins nutzen die Helfer,
statt eigene Fallback-Kopien zu pflegen. Keine Komponente wird deaktiviert;
jeder Fallback ist additiv und loggt Warnung + Begründung.

### Implementierung

1. `backend/core/ml_device_manager.py`: `ort_run_with_cpu_fallback(session, feeds,
   rebuild_cpu_factory, label)` — MIOpen-Kernel-Fehler („Code object build failed“)
   → einmaliger CPU-Session-Rebuild. Verdrahtet in LAION-CLAP (`embed_audio`),
   MelBandRoformer (`separate`), PANNs (`get_tags`).
2. `backend/core/resampling_utils.py`: `resample_audio(audio, orig_sr, target_sr)` —
   librosa zuerst; bei `get_call_template`-AttributeError scipy.signal.resample_poly.
   **Never-Pass-Through bei falscher Samplerate** (korrumpiert ML-Embeddings).
   `genre_classifier._resample` darauf umgestellt. Ebenso: DSP-Ersatzpfade für
   `_onset_rate` (Energie-Flux) und `_estimate_key` (FFT-Pitch-Class-Profile) —
   echte Messwerte statt Konstanten-Fallbacks (2.0 / „Unbekannt“).
3. `backend/core/ml_model_readiness.py`: Readiness-Checks werfen nie — Import-Ketten-
   Fehler melden „nicht bereit“ mit §V6-Warnung statt CRITICAL-Selbsttest-Abbruch.
4. `backend/core/era_classifier.py`: Tier-1 ruft CLAP nur bei geladenem Modell
   (eine Warnung, kein Traceback-Spam); `EraResult`-Label/Decade-Invariante
   (Snap auf VALID_DECADES zieht das Label nach).
5. `backend/core/genre_classifier.py`: Stille CLAP-Tag-Fallbacks loggen §V6-
   Warnungen (positiv UND negativ) — Degradation ist nie wieder unsichtbar.
6. `backend/core/unified_restorer_v3.py`: §v10.53-Invariante — §DENKER- und
   §3.0-CrossPhase-Modulation sind mit `not _strength_explicit` geschützt;
   explizite Stärke bleibt autoritativ (Spec 23-Nachtrag).
7. `backend/core/pre_analysis.py`: Chain-Depth-Cap nach wörtlicher
   v10.19-Regel — `_md_confidence < 0.50 → max_chain_depth=2` (obere Stufen
   nutzen die geboostete Konfidenz). Verhindert kettenadaptive Tape-Detektoren
   auf ungeklärten Ketten (Befund: 658 Head-Dip-False-Positives bei
   md_conf=0.31).
8. `backend/core/pre_analysis.py`: Toter Code-Block entfernt — der
   §v10.220-DefectConsensusPipeline-Aufruf war in den Dataclass-Body gerückt
   (NameError still geschluckt → lief nie). Integration als Roadmap dokumentiert
   (Manifest→DefectAnalysisResult-Adapter nötig).
9. `backend/core/defect_scanner.py` + `forensics/medium_detector.py`:
   Zero-Length-Guards — bei Audio < 0.05 s liefern scan()/detect() ehrliche
   Leer-Ergebnisse statt Unsinn (Befund: „0.0s Audio“ → vinyl=1.00 aus Stille,
   8 Consensus-Defekte).
10. `backend/core/unified_restorer_v3.py`: §v10.705-B6-Warnpflicht umgesetzt —
    Chain-Injection fremder Material-Pflichtphasen loggt jetzt sichtbar
    („Material-Fremdlauf“); zuvor still (Befund: phase_64_tape_splice_repair
    auf vinyl-Primary ohne Spur).

### Erfolgskriterium

- Kein CRITICAL im Readiness-Selbsttest bei numba/librosa-defekter Umgebung.
- `PANNs tags` nicht leer bei MIOpen-Kernel-Fehler (CPU-Retry greift).
- Genre-CLAP rechnet mit korrekt resampledtem Audio (kein Pass-through).
- Kein Traceback-Spam im EraClassifier-Log; eine §V6-Warnung pro Fallback.
- Alle bestehenden Tests grün (Repro-Ketten: 101/101 Genre+Resample,
  103/103 Era, 11/11 ROCm-Fallback, 50/50 FeedbackChain).

### Aufwand

3h | **Wohlklang-Wirkung:** Indirekt

### Risiken & Gegenmaßnahmen

| Risiko | Eintrittswkt. | Gegenmaßnahme |
|--------|---------------|---------------|
| CPU-Retry verlängert Inferenz-Latenz | Hoch (bei jedem MIOpen-Defekt) | Einmaliger Rebuild; Session bleibt dauerhaft CPU — Folge-Inferenzen normal |
| scipy.resample_poly weicht minimal von librosa ab | Niedrig | Phasenlineare Polyphasen-Semantik identisch; Längen-Konvention ceil() getestet |
| „Nie werfen“ verdeckt echte Registrierungsfehler | Niedrig | Selftest prüft weiterhin ALLE Checks und loggt Attribut-Fehler; nur der AST-Perceptual-Check degradiert explizit |

---

## Ziel-Matrix

| Ziel | Betroffen? | Wie? |
|------|-----------|------|
| Hörbarer Wohlklang | Nein | Indirekt: ML-Pfade fallen nicht mehr still aus |
| Systemische Stabilität | Ja | Ein kanonischer Fallback-Ort; Ausfälle sichtbar statt still |
| Nachhaltige Wartbarkeit | Nein | Sekundär: fünf Flake-Muster in tests.instructions.md verankert |

> **Regel:** Eine Maßnahme adressiert GENAU EIN Ziel als primäres Ziel.
> Die anderen beiden dürfen als sekundäre Ziele profitieren, aber nicht
> im Fokus stehen. Keine Maßnahme adressiert alle drei gleichzeitig.
