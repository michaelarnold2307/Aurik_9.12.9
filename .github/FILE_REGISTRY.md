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
| scripts/gen_integration_matrix.py | ACTIVE | tooling/ci | ja | — | Regeneriert .github/GEBOTE_INTEGRATION_MATRIX.md (GEBOTE-/VERBOTEN-Integrations-Status); maschinell geprüft durch audit/spec_integration_scanner.py |
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
| tests/integration/test_phase_cascade_integration.py | ACTIVE | testing/integration | ja | — | Phase-Kaskaden-E2E: 15+ Tests für NR-/Dynamik-/Gesang-Kette, NaN/Inf-Schutz, Shape-Erhalt, Determinismus, Peak-Bounds, Edge-Cases, Performance-Grenzen (SOTA 2026-09) |
| backend/core/phases/_denoise_helpers.py | ACTIVE | backend/core/dsp | ja | — | Module-level helper functions & constants für phase_03_denoise: Era-adaptive NR-Routing (§4.4), Decade-Stärke-Multiplikator (§2.14+); stateless, keine Phase-Instanz nötig (Split 2026-09) |
| backend/core/phases/_denoise_algorithms.py | ACTIVE | backend/core/dsp | ja | — | Core Denoising-Algorithmen: IMCRA Noise Estimation (Cohen 2002), OMLSA Gain (Cohen 2003), Salience G_floor, Adaptive Guard Profile, Phase Correction (Prusa 2017), ERB-Bands (Glasberg 1990), Multi-/Masking-Gate, Musical Noise Suppression, Transient Preservation; stateless Funktionen (Split 2026-09) |
| backend/api/bridge_cache.py | ACTIVE | api/bridge | ja | — | Thread-safe LRU caches für Analyse-Ergebnisse (Defect, Era/Genre, Medium, Restorability); content-addressed Keys verhindern redundante Re-Analyse bei Datei-Umbenennung; extrahiert aus bridge.py Zeilen 238-640 (Split 2026-09) |
| backend/api/bridge_core.py | ACTIVE | api/bridge | ja | — | Lazy-import wrappers für Enums, Restorer-Klassen, Denker, DefectScanner, MediumClassifier, Era/Genre-Classifiers und RestorabilityEstimator; extrahiert aus bridge.py (Split 2026-09) |
| tests/unit/test_ml_dsp_fallback_paths.py | ACTIVE | testing/unit | ja | — | §V6 ML→DSP-Fallback-Pfade: Pre-Echo-Detector, Noise-Texture-Guard, Vocal-Harmonic-Decomp (ZCPA), SOTA-Vocal-Pipeline, Phoneme-Boundary-Detector, Hallucination-Guard; 15 Tests für Fallback-Kette ohne ONNX/PyTorch (2026-09-04) |
| tests/unit/test_mushra_corpus.py | ACTIVE | testing/unit | ja | — | Real-Audio-Korpus MUSHRA-Tests: Korpus-Integrität, MUSHRA-Score-Berechnung für damaged/clean-Paare, Regression-Detection, JSON-Bericht; 7 Tests (2026-09-04) |
| tests/unit/test_gpu_detection_failsafe.py | ACTIVE | testing/unit | ja | — | GPU-Detection Failsafe: CPU-only Modus, Detection-Timeout-Fallback, ONNX-CPU-Provider, ML-Inferenz-auf-CPU, Memory-Budget; 7 Tests (2026-09-04) |
| tests/unit/test_multimodal_restoration.py | ACTIVE | testing/unit | ja | — | Multi-Modal-Restaurierung: MultimodalDecisionEngine, Genre/Era-Ketten, Prompt-NLP, Fallback-Chains, Metadata; 10 Tests (2026-09-04) |
| backend/core/presence_embedding.py | ACTIVE | backend/core | ja | — | §G90 PresenceEmbedding: perzeptuelle Metrik für menschliche Präsenz (VFC/TI/RTC/MDL/SAA); löst 43→43-Paradox; Singleton get_presence_embedding() (2026-09-04) |
| tests/unit/test_presence_embedding.py | ACTIVE | testing/unit | ja | — | PresenceEmbedding Unit-Tests: clean Audio Score, delta-Vergleich, Threshold-Passing, Sub-Score-Bounds, Mono/Stereo; 8 Tests (2026-09-04) |
| backend/core/era_authentic_completion.py | ACTIVE | backend/core | ja | — | §G90 EraAuthenticPerceptualCompletion: Ära-authentische HF-Ergänzung bei BW < 10 kHz; era-spezifische Parameter (1890–1990); BandwidthExtender + spectral shaping + validation; Singleton get_era_completion() (2026-09-04) |
| tests/unit/test_era_authentic_perceptual_completion.py | ACTIVE | testing/unit | ja | — | EraAuthenticCompletion Unit-Tests: BrillanzCeiling, Activation, EraCeiling, NaN-Safety, Stereo, AnchorGuidance, Singleton; 36 Tests (2026-09-04) |
| backend/core/dynamic_preservation_guard.py | ACTIVE | backend/core | ja | — | §0p Dynamik-Erhaltungs-Guard: verhindert Überkompression (> 3 dB RMS/Peak-Reduktion → Rollback); strength_scalar ∈ [0,1]; Singleton get_dynamic_preservation_guard() (2026-09-05) |
| backend/core/seed_manager.py | ACTIVE | backend/core | ja | — | §G5 Deterministischer Seed-Manager: Master-Seed pro Session + phasenspezifische Seeds; kein time.time(); reproduzierbar für A/B-Vergleiche; Singleton get_seed_manager() (2026-09-05) |
| backend/core/segment_quality_scorer.py | ACTIVE | backend/core | ja | — | §v10.101 Segment-weise Qualitätsbewertung: 5s gleitende Fenster, OQS/HPI Proxy; Segmente < Threshold isoliert neu restaurieren; Singleton get_segment_quality_scorer() (2026-09-05) |
| backend/core/vibrato_detector.py | ACTIVE | backend/core | ja | — | Adaptiver Vibrato-Detektor: Era-spezifische Raten (3–7 Hz); F0-temporal Autokorrelation + spektrale Analyse; stateless detect_vibrato_rate() (2026-09-05) |
| backend/core/phase_interaction_denker.py | ACTIVE | backend/core | ja | — | §11 Cross-Phase Consensus: detektiert Interferenzen zwischen aufeinanderfolgenden Phasen; neue Peaks > -60 dBFS → Interferenz-Flag; Singleton get_phase_interaction_denker() (2026-09-05) |
| backend/core/listener_feedback_loop.py | ACTIVE | backend/core | ja | — | Listener-in-the-Loop: segmentweise A/B-Bewertung durch Hörer (< 6 → neu restaurieren); ReviewSegment + FeedbackResult; Singleton get_listener_feedback_loop() (2026-09-05) |
| backend/core/logging_utils.py | ACTIVE | backend/core | ja | — | Logging-Hilfsfunktionen: standardisierte Logger-Konfiguration, Formatvorlagen für DSP/ML-Phasen (2026-09-05) |
| backend/core/material_bandwidth_ceiling.py | ACTIVE | backend/core | ja | — | Material-spezifische Bandbreiten-Obergrenze: Medium/Era-adaptive HF-Ceiling; verhindert Überextrapolation (2026-09-05) |
| backend/core/perceptual_phase_gate.py | ACTIVE | backend/core | ja | — | Perzeptuelles Gate für Phasen-Entscheidungen: Roughness/Sharpness-Fenster + Ermüdungs-Abbruch; nutzt zwicker_metrics (2026-09-05) |
| backend/core/reconstruction_context.py | ACTIVE | backend/core | ja | — | Rekonstruktions-Kontext-Management: Song-isolierte Stateful-Module, Circuit-Breaker + Caches pro Song (§V8/§G1) (2026-09-05) |
| backend/core/vocal_supremacy_gate.py | ACTIVE | backend/core | ja | — | Vokal-Suprematie-Gate: Vokal hat Priorität über alle anderen Spuren; verhindert Überkompression von Gesang (2026-09-05) |
| backend/core/logging_utils.py | ACTIVE | backend/core | ja | — | Logging-Hilfsfunktionen: standardisierte Logger-Konfiguration, Formatvorlagen für DSP/ML-Phasen (2026-09-05) |
| backend/core/material_bandwidth_ceiling.py | ACTIVE | backend/core | ja | — | Material-spezifische Bandbreiten-Obergrenze: Medium/Era-adaptive HF-Ceiling; verhindert Überextrapolation (2026-09-05) |
| backend/core/perceptual_phase_gate.py | ACTIVE | backend/core | ja | — | Perzeptuelles Gate für Phasen-Entscheidungen: Roughness/Sharpness-Fenster + Ermüdungs-Abbruch; nutzt zwicker_metrics (2026-09-05) |
| backend/core/reconstruction_context.py | ACTIVE | backend/core | ja | — | Rekonstruktions-Kontext-Management: Song-isolierte Stateful-Module, Circuit-Breaker + Caches pro Song (§V8/§G1) (2026-09-05) |

## Pflege-Regeln

| backend/core/cassette_defect_verifier.py | ACTIVE | backend/core | ja | — | Kassetten-Defekt-Verifizierung: validiert Defekt-Erkennung für Kassetten-Material; verhindert False-Negatives bei Bandgeräuschen (2026-09-05) |
| backend/core/defect_detection_quality_gate.py | ACTIVE | backend/core | ja | — | Qualitätsgate für Defekt-Detection: prüft Erkennungsrate, False-Positive-Rate, Timing-Limits; blockt Pipeline bei Degradation (2026-09-05) |
| backend/core/defect_re_scanner.py | ACTIVE | backend/core | ja | — | Sekundärer Defekt-Scanner: validiert primäre Erkennung; multi-pass Konsens für kritische Segmente (2026-09-05) |
| backend/core/dsp/cumulative_hallucination_tracker.py | ACTIVE | backend/core/dsp | ja | — | Kumulativer Halluzinations-Tracker: misst kumulative Artefaktbildung über Phasen; Abort bei Threshold-Überschreitung (2026-09-05) |
| backend/core/intentional_artifact_classifier.py | ACTIVE | backend/core | ja | — | Klassifiziert intentionale vs. unbeabsichtigte Artefakte; schützt musikalisch gewollte Effekte vor Überkorrektur (2026-09-05) |

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
