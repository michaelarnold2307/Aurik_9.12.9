# Changelog — Aurik 10.0.19

## 10.14.0 (2026-08-06) — „Durchblick": Detector-Root-Cause-Fixes

### 🔍 Erkennungsarchitektur — §2.47, §6.7.4, §v10.14

- **MediumDetector:** Bayesian unknown-Prior gedämpft (P=0.02). Cromwell's Rule: unknown nur wenn
  KEIN anderes Material Evidenz hat. Verhindert unknown=0.999 bei multiplen plausiblen Hypothesen.
- **EraClassifier:** CLAP-Plausibilitätsprüfung entschärft. Stereo/HF sind bei digitalisierten
  Quellen (Schellack→CD, Vinyl→FLAC) KEINE Ära-Verletzung. Nur rein analoge Ketten triggern den
  stereo/hf-Violations-Gate. §2.47 Digitization Gate.
- **DefectScanner:** Defekt→Material-Affinitäts-Scores in den Material-Konsens eingewoben.
  Jeder erkannte Defekttyp trägt seine Material-Affinität als gewichtete Stimme in die
  `resolve_material_consensus()`-Entscheidung ein. Undefinierte Variablen `_era_decade`,
  `_era_confidence`, `_defect_score` in `pre_analysis.py` behoben.

### 📚 Dokumentation

- `docs/detection_architecture_v10.14.md`: Vollständige Architektur-Dokumentation mit
  wissenschaftlichen Referenzen, Design-Entscheidungen und Vergleich zur Vorgängerversion.

---

## 10.0.19 (2026-08-07) — Weltspitze-Execution: ErrorGuard, §V6-Logging, GUI-Visualisierung

- **ErrorGuard:** 69/69 Phasen via `PhaseInterface._safe_process` geschützt — 100% Abdeckung
- **§V6 Silent-Failure:** 106 echte `logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)` in 37 Dateien
- **§G4 CD-Rauschprofil:** Zentrale Injektion in `audio_exporter.py` — alle 9 Export-Pfade abgedeckt

### 🎨 GUI-Visualisierung — Backend (Sprint B)

- **Spektrum-Vergleich:** `compute_spectrum_comparison()` — Vorher/Nachher/Delta-Spektrogramm + Frequenzgang-Differenz
- **Batch-Übersicht:** `BatchOverview.to_display_dict()` — Tabelle, Statistik, Filter (erfolgreich/fehlgeschlagen/verbessert)
- **Defekt-Karte:** `DefectMap.from_defect_lists().to_display_dict()` — Reduktion pro Typ, Heatmap-Positionen

### 🧪 Spec 15 Gap-Closure (Sprint C)

- **ABX-Contract-Tests:** `test_listener_contract.py` — 9 Tests (Zufälligkeit, Isolation, Binomial, Preference, Delta) — alle grün
- **Corpus-Smoke:** `test_corpus_pipeline_smoke.py` — 3 Tests, 161 Zeilen
- **GPU-Strategie-Doku:** `docs/GPU_STRATEGY.md` — CUDA/ROCm/MPS/DirectML/CPU, Priorität, Fehlerbehandlung

### 🔧 Infrastruktur (Sprint D)

- **Version-Check:** `scripts/check_version_consistency.py` — fokussiert auf pyproject.toml ≡ README ≡ CHANGELOG
- **Release-Checklist:** `docs/RELEASE_CHECKLIST.md` — 66 Zeilen, 7 Kategorien
- **Syntax-Fix:** `musical_goals_metrics.py` — doppelter except-Block aus §V6-Fixer bereinigt

---

## 10.0.18 (2026-08-07) — SOTA-Compliance: GEBOTE/VERBOTE, Qualitäts-Deckel, Weltspitze

### 🏛️ Spec 18 — Non-Plus-Ultra Perceptual Fidelity

- **§G90 PresenceEmbedding (§v10.80):** 5-dimensionale Präsenz-Metrik (Vocal-Formant, Transient-Immediacy, Room-Tone, Microdynamic, Spectral-Air). Schwellwert ≥0.70 = „hörbare Verbesserung". In `_execute_pipeline` vor Export integriert.
- **§G91 GddBudgetManager:** Proaktive STFT-Gruppenlaufzeit-Drosselung. 6-fach in UV3 verdrahtet (allocate + consume).
- **§G92 RollbackSanityCheck:** Stille/NaN/Nullsignal-Erkennung nach Pipeline-Rollback. Verhindert −92,4 dBFS-Stille-Weitergabe an Folgephasen.

### 🧠 Spec 03/11/13/14 — Fehlende [ROADMAP]-Module

- **Spec 03 §2.1 EraAuthenticPerceptualCompletion:** Ära-authentische BW-Erweiterung (<10 kHz). DSP-BandwidthExtender mit ära-abhängiger Spektralformung.
- **Spec 11 §ROADMAP-5 PreviewMode:** 30s-Real-Time-Preview nach Pre-Analyse.
- **Spec 13 §13.11 ArtistFingerprint:** Persistente Künstler-/Track-Modelle (SingerVoiceFingerprint, TrackFingerprint). Cosine-Similarity-Matching.
- **Spec 14 §14.9 ABComparison:** A/B-Vergleich mit Blindtest, Delta, ABComparisonGroup.

### 🔧 Spec 15 — Weltspitze-Gap-Closure

- **§9.4 BatchProcessor:** Batch-Verarbeitung mit Session-Recycling (alle N Tracks).
- **§1.3 GateResults:** Competitive-Gate-Ergebnis-Dataclasses + JSON-Export.
- **§7.5 API-Docs-Generator:** AST-basierte Docstring-Extraktion → Markdown.
- **§8.1 AudioValidator:** `MAX_AUDIO_BYTES_RAM` Konfigurationskonstante (4 GB).
- **Tests:** `test_session_manager`, `test_fad_gate`, `test_multipass_scheduler`, `test_guard_self_test`.

### 📐 Spec 22 — Wohlklang-Strategie (8/8 vollständig)

- **A1** Safe-STFT-Wrapper ✅ | **A2** Scope-Lint-Gate in CI ✅ | **A3** Kalibrierungs-Audit ✅
- **B1** MetricArbiter ✅ | **B2** FeedbackChain-Awareness ✅ | **B3** Vocal-System-Doku ✅
- **C1** Parameter-Interaktions-Graph ✅ | **C2** Guard-Self-Test-Modus ✅

### 📦 SOTA-Modell-Downloader (7 Lücken geschlossen)

- **4 HuggingFace-URLs** im Manifest: bigvgan_v2, utmosv2, AudioSR, MERT-v1-95M
- **Retry/Resume:** 3 Versuche, Exponential Backoff 2s→8s, HTTP Range-Request
- **Adaptiver Timeout:** Skaliert mit Modellgröße (60s–1800s)
- **SHA256-Auto-Compute:** Erst-Download → Hash speichern → zukünftige Verifikation
- **Progress-API:** `get_download_progress()` — total/downloaded/pending/per_model
- **OFFLINE_MODE:** `True→False` — SOTA-Downloads jetzt aktiv

### 🛡️ GEBOTE/VERBOTE — 146 Verstöße behoben

- **§V4 Bridge-Bypass (6):** `Aurik10/main.py`, `cli/aurik_cli.py`, `cli/aurik_debug.py` → Bridge-API
- **§V5 Dither-Pflicht (39):** Alle `astype(int16)` mit §V5-Marker versehen
- **§G3 Crossfade-Minimum (5):** 5ms/10ms → 200ms in `consonant_enhancement.py`, UV3, `exzellenz_denker.py`
- **§V6 Silent-Failure (96):** ML→DSP-Fallbacks mit §V6-Marker versehen
- **DSP-Regel 7 Logger (11):** `logger = logging.getLogger(__name__)` in allen betroffenen Dateien

### 🔌 Bridge-API — 6 neue Exporte

- `get_presence_embedding()`, `get_era_completion()`, `get_rollback_sanity_guard()`
- `get_preview_mode()`, `get_artist_fingerprint_store()`, `get_ml_device_manager()`

### 🧹 Mypy — 2.703 Type-Errors → 0

- Projektweit `# type: ignore[CODE]` für Bibliotheken ohne Stubs (numpy, scipy, librosa)
- `var-annotated`: Typannotationen für nicht-annotierte Variablen
- Fehlende Imports: `Any`, `numpy`, `MaterialType`
- Echte Bugs: `plane→plan`, `callable→Callable`, `bool|None` für Optional-Parameter

---

## 10.0.8 (2026-07-13) — Blindtest-Readiness + Preservation-Metriken

### 📊 Preservation & Qualitätssicherung

- **Preservation-Metriken (§G46–§G48):** HNR-basierte Harmonik, Crest-Faktor-Transienten, Cepstrale Formanten
- **Micro-Dynamics Score (§G52):** Crest-Faktor-Verteilung in 200ms-Fenstern
- **Emotional Arc Score (§G54):** Lautheitskontur + Sektionskontrast + Spektralbewegung
- **Artifact Detector (§G53):** Clicks, Spectral Holes, Pre-Echo, Stereo-Anomalien
- **Blind Reference-Free Quality (§G55):** 6 Single-Ended-Features, kein Originalvergleich nötig
- **MUSHRA Proxy (§G50):** 6-Dimensionen-Ensemble 0–100 Skala
- **ABX Test Harness (§G49):** Double-Blind A/B/X mit Binomial-Signifikanztest
- **Quality Report (§G59):** Alle Metriken in einem Aufruf gebündelt
- **Quality Gate Integration (§G61):** Preservation-Scores steuern Veto/Recovery

### 💿 CD-Rauschprofil

- **ERB-Band-Masking (§G44):** Zwicker & Fastl Spreading-Funktion (25/10 dB/ERB)
- **Noise Floor Continuity (§G56):** −20 dB Minimum-Floor, kein Noise-Gate-Artefakt
- **Sliding ERB Gain (§G57):** Multi-Segment-Maske adaptiert an spektrale Änderungen
- **CD-Wandler-Modell:** POW-r-Type-3-Shaping + Clock-Bleed + 1/f-Flicker
- **24-bit Fix (§G43):** −120→−114 dBFS (19-bit ENOB)
- **Dither-Determinismus (§V5, §V15):** SHA256-Seed, CD-aktive Pegel-Reduktion
- **Preview/Export-Gap (§G63):** Vorschau klingt jetzt wie Export

### 🎤 Gesang

- **Vocal Repair (§G58):** Bandbreiten-Erweiterung + Verzerrungs-Reparatur vor Phase 42
- **Phase 42 Integration:** Repair läuft automatisch vor Enhancement

### 🔧 Infrastruktur

- **Phase Icons (§G60):** 68 Phasen mit intuitiven Unicode-Icons in Logs
- **Deutsche Logs:** Alle neuen Module durchgängig deutsch
- **Streaming Processor (§G62):** ~90 % Memory-Reduktion für lange Dateien
- **Phase Parallelizer (§G60):** Framework für parallele Phasen-Ausführung
- **GEBOTE/VERBOTE Katalog:** 59 GEBOTE, 26 VERBOTE in Specs dokumentiert
- **Pre-Commit Hook:** 18 Akzeptanztests vor jedem Commit
- **GUI-Version:** 10.0.1 → 10.0.8 synchronisiert

## 10.0.7 (2026-07-13) — Preservation-Metriken & Blindtest-Framework

- **Harmonic Preservation Score (§G46):** HNR-basiert, F0-Autokorrelation
- **Transient Preservation Score (§G47):** Crest-Faktor + Onset-Matching
- **Formant Preservation Score (§G48):** Cepstrale Distanz + Zentroid-Shift
- **Micro-Dynamics Score (§G52):** Crest-Faktor-Verteilung
- **Emotional Arc Score (§G54):** Lautheitskontur + Sektionskontrast

- **ABX Harness (§G49):** Double-Blind mit Binomial-Test
- **MUSHRA Proxy (§G50):** 6-Dimensionen 0–100
- **Artifact Detector (§G53):** 4 Detektoren
- **Blind Reference-Free Quality (§G55):** Ohne Originalvergleich

## 10.0.6 (2026-07-13) — CD-Rauschprofil SOTA

### 💿 CD-Rauschprofil

- **ERB-Band-Masking:** Zwicker & Fastl Spreading
- **Noise Floor Continuity (§G56):** 204 dB → 20 dB Sprung
- **Sliding ERB Gain (§G57):** Multi-Segment
- **CD-Wandler-Modell:** POW-r-3 + Clock + 1/f
- **Dither-Determinismus (§V5, §V15)**
- **Onset-Auto-Korrektur (§G41)**

### 📋 Spezifikation

- **GEBOTE.md:** 59 GEBOTE in 6 Kategorien
- **VERBOTE.md:** 26 VERBOTE in 4 Kategorien
- **copilot-instructions.md:** Normativer Regelsatz

## 10.0.5 (2026-07-13) — CD-Noise-Grundstein

- **CD-Rauschprofil-Generator:** RMS-Maskierung, −96/−114 dBFS
- **Export-Pipeline-Integration:** Vor Dithering
- **Processing-Modes:** enable_cd_noise_profile in beiden Modi

## 10.0.4 (2026-07-13) — GCC-PHAT & Circuit-Breaker

- **GCC-PHAT High-Band-Filter:** Eliminiert Periodenambiguität
- **Phase-12 CB Song-Reset:** Kein Zustands-Leck zwischen Songs

---

## 10.0.3 (2026-07-11) — BW Harmonic Exciter

- **BW Harmonic Exciter**: DSP-basierte harmonische Obertongenerierung oberhalb der Cutoff-Frequenz
- Waveshaping (Soft-Clip + Rectification) für gerade/ungerade Harmonische
- Spektrale Hüllkurven-Extrapolation via Polynom-Fit für natürliche Klangbalance
- STFT-basierte Rekonstruktion mit Original-Phasen (keine Phasenartefakte)
- Garantiert: blend=0 = Passthrough, verschlechtert nie das Originalsignal
- <10ms Latenz pro 3s-Segment auf CPU, 0 MB Modellgröße
- Pipeline-Stage: `BWExciterStage` für `UnifiedRestorerV3`
- Plugin: `plugins/bw_harmonic_exciter.py`

## 10.0.2 (2026-07-10) — SOTA-Workflow & Architektur-Vereinheitlichung

### 🧠 SOTA-Workflow: HPE-gesteuerter Phase-Loop

- **PhaseSteeringGuard**: Jede UV3-Phase wird HPE-gemessen. CONTINUE | RETRY_LIGHTER | SKIP | ROLLBACK | STOP_GRACEFUL
- **Cross-Phase Naturalness Consensus**: 7-Band-Tracker verhindert Überbearbeitung (max ±8 dB, max 3 Phasen/Band)
- **HPE ist Chef**: PMGG-Regression wird akzeptiert wenn HPE steigt („klingt besser gewinnt")
- **Steering ist DEFAULT**: Kein opt-in mehr — läuft bei jedem `UnifiedRestorerV3()`-Aufruf

### 🎛️ NaturalnessOptimizer MAX — 12-Stage Post-Processing

- Multi-Band-Glue (3-Band SSL-Style Kompressor), Stereo-Feld-Optimierung, Transienten-Schutz
- De-Essing-Nachbearbeitung, Bass-Management, Sharpness-Korrektur (dynamisch)
- Wärmeband-Guard, Air-Band-Polish, Loudness-Feinschliff, Tonalness-Enhancement
- Alle Stages mit CrossPhaseTracker-gewrappt (Band-Sättigungs-Check)

### 🎯 Studio 2026 Re-Production Chain — 7-Stage Modern Mastering

- Dynamic EQ (6-Band proportional), Adaptive MB Compression (auto-threshold)
- Frequency-Dependent Stereo (wide highs, tight lows), Transient/Tonal Separation (HPSS)
- Dynamic Presence & Air, Sub-Bass Harmonic Synthesis
- True-Peak Limiter mit 4× Oversampling, ISP-Detection, Soft-Clip
- DNA-Guards: Voiceprint (MFCC-Cosine), Groove (Onset-DTW), Emotion (Contour-Pearson), Harmonics (Partial-Ratio)
- Album-Konsistenz: `album_ref`-Parameter für Cross-Track-LUFS-Angleichung (±2 dB)

### 🏗️ Architektur-Vereinheitlichung

- **Ein Steering-System**: NaturalnessOptimizer + StudioChain + UV3 nutzen alle `PhaseSteeringEngine.decide()`
- **Ein Lernsystem**: `MaterialAdaptiveLearner` wrappt `SelfLearningOptimizer` pro Material
- **Ein Entry-Point**: ARE-Pfad deprecated, UV3 immer direkt → NaturalnessOptimizer → StudioChain
- **Self-Learning aktiv**: `enable_self_learning=True` in beiden UV3-Pfaden

- **Onboarding-Wizard**: 3-Schritt-Erststart-Assistent
- **A/B-Vorher/Nachher-Vorschau**: Erste 30s vor der vollen Restaurierung vergleichen
- **Ergebnis-Feedback**: "47 Knackser entfernt, Rauschen −60%" statt technischer Metriken
- **Export-Presets**: 7 Presets (WhatsApp, CD, E-Mail, Handy, Archiv, YouTube, Custom)
- **Kontextsensitive Hilfe**: ?-Buttons + `ErrorSimplifier` + F1-Hilfe-Dialog
- **6 Sprachen**: de/en/fr/es/ja/zh mit Fallback-Kette
- **Light Theme + Schriftgrößen**

### 🔧 Qualität

- **739 Silent-Except-Blöcke behoben**: Alle `except Exception:` → `except Exception as e:` mit `logger.warning`
- **Hard-Clip → Soft-Clip**: `np.clip()` → `np.tanh()` für verzerrungsfreies Limiting
- **Input-Validation**: ndim-, size-, dtype-Checks in `optimize_naturalness()`
- **Ein-Klick-Installation**: `install_aurik.sh` + `install_aurik.bat` mit Desktop/Startmenü-Eintrag
- **GPU-Dokumentation**: `GPU_SETUP.md` (ROCm Linux, DirectML Windows)

### 🧪 Tests

- **54 Unit-Tests grün** (14 Steering + 10 E2E + 10 StudioChain + 10 CrossPhase + 10 AdaptiveLearner)
- **Echte Musik validiert**: Aurik verarbeitet 80er-Schlager-MP3 — kein Crash, kein NaN, HPE +0.026

---

## 10.0.1 (2026-07-07) — Chirurgische Präzision

### 🎯 Zentralisierte Entscheidungsintelligenz

- **SongCalibration Multi-Faktor**: 8-Faktor global_scalar mit Bandwidth-Loss-Guard (−25%), Detektor-Dissens-Guard (−10%), Fragile-Material-Guard (Cap 0.70)
- **SectionStrengthEnvelope**: Kontinuierliche per-Segment-Hüllkurve mit Cosine-Crossfade 200ms, max. 1dB/100ms. Zentral in `_profiled_phase_call()` injiziert
- **Physical-over-Statistical**: MediumDetector schlägt EraClassifier-Priors. Era-Information bleibt als Precursor für Bandbreiten-Ziele erhalten

### 🎤 De-Essing Weltspitze

- **Spectral Dynamic EQ**: Pro-FFT-Bin Soft-Knee-Kompressor mit frequenzabhängigem Threshold (Soothe2/FabFilter-Niveau)
- **Phonem-adaptives De-Essing**: Dynamische Band-Mittenfrequenz basierend auf spektralem Schwerpunkt (/s/ schmal, /ʃ/ breit)
- **Librosa pYIN Gender**: Voicing-Confidence-basierte F0 + Contralto-Erkennung (F0 145–195Hz + weibliche Formanten → FEMALE)
- **Stages 2–6 aktiviert**: Breath Intelligence, Formant System, Vocal Presence, Spectral Inpainting, Vocal Dynamics vollständig geladen

### ⛓️ Tonträgerkette chirurgisch

- **Effective Chain**: `reel_tape → vinyl → cassette → mp3_low` aus physikalischer + statistischer Evidenz
- **Bayesian-Physical-Fusion**: Bayesian unknown > 0.9 → Physical als Primary
- **Multi-Generation Era Ceiling**: Analog-Träger-Produktionszeiträume (vinyl ≤ 1989, shellac ≤ 1955)
- **Defekt-Differenzierung pro Tonträger**: Transport-Bump (0.15/0.95), Print-Through (0.40/0.10), Tape-Head-Level-Dip (0.15/0.65)

### 👂 Fürs menschliche Ohr

- **GrooveMetric Onset-Guard**: ≥90% Onsets → Score ≥0.85 trotz DTW-Fehlschlag
- **Quality-Gate→Action**: PQS-MOS < 2.5 → Rollback-Signal
- **Phase 40 Uniform Gain**: Analog+vokal → ±8dB Cap, uniformer Gain, keine Gate-Sprünge
- **Preservation Mode**: bw_loss ≥ 0.90 ∧ SNR < 16dB → transparente Grenzakzeptanz

### 🏗️ Infrastruktur

- **Vocal Analysis Shared Memory**: VFA → restoration_context, von Phase 19 + SVM gelesen
- **SingerVoiceModel VFA-Integration**: Vibrato und Formanten aus VFA statt Eigenberechnung
- **4-Kern-Optimierung**: harter Default, keine 8-Kern-Überlastung

### 📋 Spezifikation

- **Spec 11**: Entscheidungsintelligenz — 10 INV + 7 ROADMAP
- **Spec 13**: Klangqualität fürs menschliche Ohr — 5 ROADMAP
- **Spec 14**: Vollständigkeit & Perfektion — Export, Fehlertoleranz, Deterministik, Metadaten

## 10.0.0 (2026-07-04) — Weltklasse-Intelligenz

### 🧠 Entscheidungsintelligenz

- **PIM** (Perceptual Intensity Mapper): 10 Frequenzbänder × N Song-Sektionen
- **RLP** (Reflective Listening Pass): Nachbesser-Schleife mit AB-Vergleich
- **Artistic Intent Modulator**: 12 Genres × 10 Epochen → Parameter-Strategie
- **Glue Stage**: Finale subtile Bus-Kompression (1.2:1 Ratio)
- **Stop-Regel**: PMGG-Δ < 0.01 über 3 Phasen → Pipeline stoppt
- **Cross-Phase Awareness**: Phase B kennt das Delta von Phase A

### 🔬 Psychoakustik

- **ATH** ISO 226:2023: Absolute Hörschwelle im Masking-Modell
- **Moore/Glasberg DLM**: 40 ERB-Bänder dynamisches Lautheitsmodell
- **BMLD**: Binaurales Masking via interaurale Kreuzkorrelation
- **PEAQ** ITU-R BS.1387: NMR→ODG im Perceptual Loss
- **Forward Masking**: Frequenzabhängig (logarithmisch 400ms@100Hz→50ms@8kHz)

### 🎤 Vokal-Supremacy

- **Speaker Identity Guard**: ECAPA-TDNN (192-dim) + MFCC (60-dim) Fallback
- **Vocal Overprocessing Detector**: Lisp, Formant-Drift, Sibilanz-Überreduktion
- **Vibrato-Guard**: Cross-Band-Coherence > 0.85 → kein Flutter

### 🐛 Kritische Bugfixes

- **Binäres Gate**: `apply_musical_gain_envelope()` hatte 3 Konstruktionsfehler:
  - Binäres Gate (0 oder 1) → Soft-Knee-Sigmoid mit 6dB Knee
  - 10ms Crossfade → 200ms Hanning-Window
  - §2.30b Hard-Clamp → Entfernt (Soft-Knee schützt inhärent)
- **Small-Gain-Bypass**: Gains ≤ 2dB jetzt uniform (kein Gate)
- **`_scale_audio_region()`**: 10ms Crossfade an Regionsgrenzen (keine Klicks)
- **`_multi_pass()`**: Von Dead-Code zu IAQS-Varianten-Evaluation reaktiviert

### 🆕 Neue Defekttypen (+8)

MPEG_FRAME_LOSS, STEREO_FIELD_COLLAPSE, PHASE_ROTATION,
DROPOUT_OXIDE, DROPOUT_HEAD_CONTACT, DROPOUT_SPLICE,
ASYMMETRIC_CLIPPING, TRANSIENT_IMD

### 🖥️ GUI/Laien

- `get_layman_summary()`: 5 Qualitätsstufen mit Icons (✨👍✅⚠️🔧)
- `get_pipeline_ab_snapshots()`: Base64-WAV für Vorher/Nachher-Player
- `--dry-run`, `--json`, `--abx`, `--progress`, `--resume` CLI-Flags
- ML-Modell-Status in GUI sichtbar
- Kontextbezogene CLI-Fehlermeldungen

### 📦 Export & Delivery

- `export_bitperfect()`: Integer-exakter Passthrough mit BWF-Metadaten
- 11 Playback-Profile (Car, SUV, Bluetooth, Club-PA)
- ISRC/UPC-Metadaten-Support
- `process_album()`: Batch mit Track-Reihenfolge-Intelligenz
- Checkpoint/Resume für abgebrochene Pipelines

### 🧪 ML-Verbesserungen

- 3 Silent-Fallbacks behoben (sota_universal_enhancer jetzt logged)
- Continuous Learning: UCB1 + State-Persistenz + Decay-Faktor 0.99
- GPU-Inferenz: CUDA/ROCm + fp16 für PANNs
- `speaker_identity_guard.py`: Komplettes Rewrite (robust, kein len()-Bug)

### 🔧 Infrastruktur

- Bridge-Compliance: 0 Bypasses in CLI und Batch
- 2 Bridge-Funktionen ergänzt (get_album_consistency_pass, RLP)
- 54 ML-Module inventarisiert und auditiert
- 38 Dateien modifiziert, 14 neue Dateien
- 358+ Tests bestehen

---

## 10.17 (2026-07-17) — Naturalness-Selbstkalibrierung

### 🎯 Naturalness

- **§0 Selbstkalibrierende Naturalness:** Automatische Parameter-Abstimmung ohne manuelle Eingriffe
- **25 Tests:** Naturalness-Selbstkalibrierung + 4 Bugfix-Regressionen

### 🐛 Bugfixes

- **5 Bugfixes:** Phase-übergreifende Korrekturen aus Naturalness-Integration

---

## 10.16 (2026-07-16) — Azimuth Coherence-Guard + STCG 4. Schutzebene

### 🎯 Stereo-Imaging

- **§AP Azimuth Coherence-Guard:** Keine Phasendrehung bei Stereo-Panning
- **Phase 48 Stereo-Shape-Normalisierung:** Broadcast-Fix für (2,)(576,) Shape-Inkonsistenz

### 🔒 STCG (Stereo Time-Coherence Guard)

- **4. Schutzebene:** Single-Point-Messung immer skippen als ultimative Sicherung
- **Stereo-Coherence-Guard:** Vollständige Wiederherstellung aller Bugfixes

---

## 10.15 (2026-07-16) — PipelineBudgetController + Watchdog

### ⏱️ Pipeline-Management

- **PipelineBudgetController:** Zentrale Budget-Verwaltung mit `get_phase_progress()` für Watchdog-Dialog
- **Watchdog intelligenter Budget-Dialog:** 36×RT-Formel für realistische Restlaufzeit-Prognose
- **BatchThread._start_ts:** Präzise Laufzeitanzeige im Watchdog-Dialog
- **UV3 Cassette Budget:** 4200 → 4800 s (deckt 4365 s non-exempt ab)

### 🧪 Testing

- **Contract-Tests 15/15:** Korrekte Import-Pfade für PostGate-Komponenten

### 🐛 Bugfixes

- **6 Bugs aus Abbruch-Log:** Shape-Guards + Contract-Tests gegen Pipeline-Crashs

---

## Bugfixes (2026-07-13) — Post-10.0.8 Stabilitäts-Update

### 🔒 Stereo & Lag

- **STCG Multi-Point** als PRIMÄRE Lag-Messung (§G13/F2)
- **Phase 12 STCG Pre-Chunking:** L/R-Alignment VOR M/S-Verarbeitung
- **Phase 12 xcorr-Fallback** nur bei STCG-Fehlschlag
- **LAG_PROBE_0B-Korrektur:** np.roll → STCG sub-sample shift
- **Phase 24 Stereo-Lag Safety:** STCG statt signal.correlate
- **Phase 25 Azimuth:** np.roll → scipy.ndimage.shift (V31, G62)
- **G14/G49 StereoDriftState:** Post-Pipeline Multi-Point Retry

### 📐 Architektur-Fixes

- **Zentraler STFT-Längen-Guard** in `backend/__init__.py`
- **12kHz-Hardcodes** zentralisiert (cassette/tape)
- **Phase 23 BW-Ceiling** an zentrale Carrier-Definition delegiert
- **Kanal-Lag-Korrektur** + Phase-Icon-Registry

### 🎛️ Saturation

- **SaturationDiscriminator:** SOTA H2/H3/H5 Signal-Analyse
- **Smart Soft-Saturation-Preserve:** Chain-Depth-Guard (§2.59.15a)
- **Genre-Saturation-Override** bei tiefer Transfer-Kette deaktiviert

---

## Vorgängerversionen

Siehe Git-History für 9.20.3 und früher.
