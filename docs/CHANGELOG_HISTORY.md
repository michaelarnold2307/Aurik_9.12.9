# Aurik 10.0.10 — Spec-Änderungshistorie

> Stand: 16. August 2026

## v10.0.x (16. August 2026) — ML-Fallback-Architektur & Suite-Stabilisierung

- **§v10.53-Invariante**: Explizite Stärke bleibt autoritativ — §DENKER- und
  §3.0-CrossPhase-Modulation in `_profiled_phase_call` mit `not _strength_explicit`
  geschützt (vorher überschrieb der Consensus-Cap explizite 0.99; Spec 23-Nachtrag).
- **ROCm/MIOpen-CPU-Fallback**: LAION-CLAP, MelBandRoformer, PANNs — bei
  Nicht-Memory-ONNX-Fehlern einmaliger CPU-Session-Rebuild statt Ausfall.
  Zentrale Helper `ort_run_with_cpu_fallback()` in `ml_device_manager.py`.
- **numba-Resample-Guard**: `resampling_utils.resample_audio()` — librosa zuerst,
  bei `get_call_template`-Defekt scipy.signal.resample_poly. Never-Pass-Through.
  Genre-Classifier-CLAP-Pfad darauf umgestellt (§V6-Warnungen statt stiller 0.35);
  DSP-Ersatzpfade für `_onset_rate` (Energie-Flux) und `_estimate_key` (FFT-PCP).
- **Readiness-Vertrag**: Readiness-Checks werfen nie — Import-Ketten-Fehler
  (numba/librosa) melden „nicht bereit“ statt CRITICAL-Selbsttest-Abbruch.
- **EraClassifier**: Tier-1 ruft CLAP nur bei geladenem Modell (eine §V6-Warnung,
  kein Traceback-Spam); EraResult-Label/Decade-Invariante (Snap → Label).
- **FeedbackChain-Budget**: Formel auf Normtabelle korrigiert
  (≤120 s/min Audio, Floor 60 s — vorher min 120 s).
- **Suite-Stabilisierung**: 5 Flake-Klassen kanonisch gefixt (siehe
  `.github/instructions/tests.instructions.md`, Abschnitt Test-Isolation).

## v10.10.0 (19. Juli 2026) — Preset-Learning × Selbstkalibrierung

- **Mid-Pipeline HPE Quality Gate**: Alle 8 Phasen HPE-Score-Prüfung mit automatischer Selbstkalibrierung. `mid_pipeline_quality_gate.py` (181 LOC).
- **Reference-Track Auto-Calibration**: Referenz-Track → automatische Song-Goal-Kalibrierung. Preset ∩ Material-Floor. `reference_track_calibrator.py` (243 LOC).
- **Magic Restore Preset**: Ein-Klick-Restauration. 6 Built-in-Presets + User-Learning. `magic_restore_preset.py` (297 LOC).
- **Spektrogramm-Snapshot**: 256×256 Graustufen-PNG pro Phase für Debugging. `spectrogram_snapshot.py` (106 LOC).
- **Parallel Stereo Executor**: 12 DSP-Phasen left/right parallel via ThreadPool (~40% Speedup). `parallel_stereo_executor.py` (103 LOC).
- **PhaseFingerprint**: Inkrementelles Re-Processing. Hash-basierter Parameter-Vergleich → nur geänderte Phasen neu. `phase_fingerprint.py` (132 LOC).
- **Model Warm-Up Pool**: 5 häufigste Modelle parallel laden während Pre-Analyse. `model_warmup_pool.py` (123 LOC).
- **Album-Konsistenz-Gate**: Track 1 = Referenz, Tracks 2-N = LUFS/Tilt/Width-kalibriert. `album_consistency_gate.py` (62 LOC).

## v10.9.0 (19. Juli 2026) — SOTA-Kalibrierung & ML-Orchestrierung

- **Model-Chain-Orchestrator**: Shared ML-Modelle (DeepFilterNet, Banquet, SGMSE+, BW). RAM-Budget 6GB, Auto-Eviction. `model_chain_orchestrator.py` (170 LOC).
- **Pre-Analyse Era-Fallback**: Material→Decade-Map wenn Era-Classifier None liefert. 13 Material-Typen abgedeckt.
- **Phase-07 Drive-Dämpfung**: Drive 2.5→1.8, 0.6–1.8 Range. h2-Überproduktion ~3× reduziert.
- **Adaptive Phase-Reihenfolge**: Material-spezifische Order (Kassette: Hiss vor Harmonics). `adaptive_phase_order.py` (120 LOC).
- **OneTakeExport Intelligent Mastering**: Adaptiver Fatigue-Cut (-1/-2/-3dB). TP-Limiter-Schwelle -0.3 dBTP.
- **Live A/B Preview**: Pre/Post-Audio-Ring für GUI-Playback. `live_ab_preview.py` (95 LOC).
- **Restoration Report HTML**: Vollständiger HTML-Report mit Phasen/Defekten/Joy/Fatigue. `restoration_report.py` (145 LOC).
- **Early-Silence-Gate**: RMS < -60 dBFS → Phase skipped + WARNING.
- **Wet/Dry-Kohärenz**: Strength proportional zu wet_dry bei QOL-Phasen.

## v10.35.0 (18. Juli 2026) — SOTA Logging & Dead-Feature-Aktivierung

- **SOTA Logging**: 15 Features debug→warning (PGHI, MRN, EAPC, HHC (4×), MaskingClamp, Fletcher-Munson, PhaseSkipper, SteeringGuard, librosa/sklearn, ConsonantEnhancement, ContractValidator (5×), GlobalGainBudget, DonationReminder).
- **MRN Plugins aktiviert**: 4 ML-Chain-Plugins (Shellac: BW+DFN, Vinyl: Banquet+DFN, Tape: SGMSE++DFN, Lacquer: BW+DFN 40%Wet).
- **Phase 07 Kalibrierung**: Pre-Echo-Schwelle 2→3.5ms, Blend-Cap 0.60, h2 material-adaptiv.
- **GUI-Kommunikation**: Denker→GUI Live-Toasts, ErrorSimplifier verdrahtet, Experience Insights in Results Summary.
- **SFT ArtifactRescue**: Min-Wet 0.05, WARNING bei aggressivem Rollback.
- **Bugfixes**: OneTakeExport TP 0.0→-0.3, build_results_data restoration_result, Pre-commit Path.resolve().
- **Pakete**: transformers installiert.

---

## v10.0.0 (2. Juli 2026) — UI-Livezaehler und Fortschrittsreserve

- **Frontend-Version synchronisiert**: Paket- und sichtbare App-Version auf `10.0.8` angehoben;
  Frontend-Titel/Splash/About beziehen diese Version weiterhin aus `Aurik10.__version__`.
- **Tonaussetzer-Livezaehler**: Dropout-Aliase (`dropouts`, `DROPOUTS`, `gap`, `gaps`,
  `tape_dropout`) werden auf `dropout` normalisiert. Lokalisierte Dropout-Marker werden waehrend
  echter Dropout-/Inpainting-Phasen anhand des Timeline-Cursors als behoben markiert; erledigte
  Marker koennen aus alten `_locations` nicht wieder aktiv werden.
- **Fortschrittsreserve**: UV3-Post-Processing belegt in der GUI nur noch 83-90 %. 90-100 % bleibt
  fuer Defektabschluss, Export-Quality-Gate, Datei-Export und UI-Finalisierung reserviert, damit
  lange Endphasen nicht als eingefrorene 97-%-Anzeige erscheinen.
- **Real-Audio-Gate-Typisierung**: `test_real_audio_edge_lag_gate.py` verengt Fixture-`sr` statisch
  per `typing.cast`, wodurch Pylance/mypy `call-overload` auf `int(object)` blockiert wird.
- **Normative Gates**: `test_modern_window_gui_contract.py` sichert Dropout-Aliase,
  timelinebasierte Chip-Zaehlung und Fortschrittsreserve ab.

## v10.0.0 (2. Juli 2026) — CD-Rauschboden für analoge Exportziele

- **Rauschboden-Doktrin aktualisiert**: Analoge Tonträger (`shellac`, `wax_cylinder`,
  `lacquer_disc`, `wire_recording`, `vinyl`, `tape`, `reel_tape`, `cassette`) bekommen
  im Export keinen analogen Mindest-Rauschboden mehr zurück. Hiss, Oberflächenrauschen
  und Schellack-Rauschbett gelten als reparierbare Trägerdefekte, solange Vocal-/Musik-
  Schutzgates eingehalten werden.
- **`noise_floor_guard`**: Materialspezifische Analog-Mindestböden auf `None` gesetzt;
  der Guard injiziert nur noch bei explizitem Floor-Override.
- **`noise_texture_resynth`**: Analoge Materialien zielen bei Over-NR-Korrektur auf
  `cd_digital`-Textur mit CD-ähnlichem Floor (ca. −74 dBFS), statt Vinyl-/Shellac-/Tape-
  Rauschteppich wiederherzustellen.
- **Specs synchronisiert**: §0a/V19/V21 und Spec 07 §8.2 beschreiben CD-ähnlichen
  Export-Rauschboden als Ziel für alle analogen Tonträger.

## v10.0.0 (4. Juni 2026) — InnovationSuperiorityOrchestrator (Disziplin-Parallelisierung)

- **Neues Kernmodul**: `backend/core/innovation_superiority_orchestrator.py` fuehrt
  einen disziplinuebergreifenden Runtime-Orchestrator ein (Defektintelligenz,
  Vokalintegritaet, timbrale Wahrhaftigkeit, zeitliche Praezision,
  raeumliche Kohaerenz, Zielkonvergenz).
- **UV3-Closed-Loop Integration**: Der Orchestrator laeuft parallel zum
  `MetricReliabilityGraph` im §2.78-Pfad und erzeugt pro Phase
  `priority_goals`, `recovery_phase_hints`, `goal_confidence_uplift`,
  `discipline_scores`, `innovation_intensity`.
- **Safety-Design**: Rein advisory, keine Gate-Ueberschreibung, gebundene
  Confidence-Uplifts (`<= 0.05`) bei weiterhin unveraenderten Bounds
  (`[0.20, 0.98]`) fuer die finalen Goal-Konfidenzen.
- **Tests**: Neue Unit-Suite
  `tests/unit/test_innovation_superiority_orchestrator.py`; UV3- und
  Reliability-Slices weiterhin gruen.

## v10.0.0 (4. Juni 2026) — PMGG Telemetry-Alignment + Coalition-Aware Rescheduler

- **PMGG-Rekonstruktions-Telemetrie normiert**: Die Spezifikation dokumentiert jetzt
  `pmgg_reconstruction_threshold_multiplier` und die zugehörigen Begleitfelder als
  kanonische Metadaten für lokal gestützte Rekonstruktionsentscheidungen.
- **Rekonstruktive Entscheidungen bleiben lokal gestützt**: `passed_reconstruction_localized`
  ist als kontextgewichtete Entscheidung beschrieben; der Threshold-Multiplikator ist Teil
  der Telemetrie und kein freistehender UI-Proxy.
- **AdaptivePhaseRescheduler koalitionssensitiv**: §2.78 priorisiert jetzt nicht nur Gap,
  Confidence, Chain und Material, sondern auch Recovery-Phasen, die mehrere offene Goals
  gemeinsam bedienen können.
- **Pipeline-Regelwerk nachgezogen**: Die UV3-/§2.78-Instruktion beschreibt nun den
  Koalitionsbonus ausdrücklich als Reihenfolge-Hint bei unverändert harten Minimal-
  Intervention-/Session-Guards.
- **Tests**: Rescheduler-Unit-Suite und PMGG-Rekonstruktionssuiten decken die neuen
  Metadaten- und Priorisierungsregeln ab.

## v10.0.0 (4. Juni 2026) — Coalition-Aware RestorationMemory Feedback Loop

- **UV3 prior propagation**: Geladene RestorationMemory-Priors werden zusätzlich in
  `_restoration_context["restoration_memory_prior"]` gespiegelt, damit Downstream-Module und
  Telemetrie denselben Prior-Kontext nutzen.
- **Coalition-aware prior learning**: Erfolgreiche Runs persistieren nun Koalitions-Lernsignale
  (`_coalition_learning_factor`, `_dominant_phase_coalition_ratio`, `_phase_coalition_count`)
  als advisory Metadaten in `phase_params`.
- **GPOptimizer context blend**: `_apply_context_priors(...)` nutzt den
  `_coalition_learning_factor` als kleinen Zusatzhebel für das Prior-Weighting (begrenzt,
  hard-gate-safe, ohne Überschreibung normativer Gates).
- **Tests**: `tests/unit/test_gp_parameter_optimizer.py` enthält den neuen Regressionstest
  `test_85_coalition_learning_factor_increases_memory_blend`; UV3- und GP-Slices grün.

## v10.0.0 (4. Juni 2026) — Runtime-Metric-Reliability Graph im Closed Loop

- **Neues Kernmodul**: `backend/core/metric_reliability_graph.py` führt einen
  kontextsensitiven Reliability-Graph für Goal-Proxys ein (Modus/Material/Era/TCCI-Bucket,
  EWMA-Update, lokale atomare Persistenz unter `~/.aurik/metric_reliability_graph.json`).
- **UV3-Integration**: Vor jedem §2.78-Rescheduler-Aufruf werden aktuelle
  Phase-Deltas + PMGG-Metadaten in den Graph geschrieben; die resultierende
  Runtime-Reliability wird mit der bestehenden Goal-Confidence-Map geblendet
  (`0.60 base + 0.40 runtime`).
- **Zielwirkung**: Proxy-Verlässlichkeit wird jetzt pro Lauf und Kontext
  nachkalibriert, wodurch Grenzfall-Entscheidungen stabiler und weniger
  over-/under-conservative werden, ohne harte Gates zu überschreiben.
- **Tests**: Neue Unit-Suite `tests/unit/test_metric_reliability_graph.py`.

## v10.0.0 (4. Juni 2026) — Adaptive Cross-Run Blend Weights

- **Gatekeeper-Ausbaustufe**: Runtime-Reliability wird nicht mehr mit fixen Gewichten
  gemischt. `MetricReliabilityGraph.get_blend_weights(...)` liefert jetzt
  evidenzadaptive Gewichte (`base`, `runtime`) pro Kontext.
- **UV3-Integration**: §2.78 nutzt die adaptiven Gewichte direkt beim
  Goal-Confidence-Blending vor dem Rescheduler-Entscheid.
- **Safety-Bounds**: Blend-Gewichte bleiben normiert und gebunden
  (`base ∈ [0.40, 0.75]`, `runtime ∈ [0.25, 0.60]`), advisory-only.
- **Tests**: `tests/unit/test_metric_reliability_graph.py` enthält zusätzliche
  Validierung für die support-abhängige Gewichtsverschiebung.

## v10.0.0 (4. Juni 2026) — PMGG Localized-Counterfactual Confidence Gate

- **PMGG localized reconstruction confidence**: `backend/core/per_phase_musical_goals_gate.py`
  ergänzt `_assess_reconstruction_localized_confidence(...)` für rekonstruktive
  Phasen (`phase_24`, `phase_55`).
- **Counterfactual-Freigaben jetzt kontextgewichtet**:
  `passed_reconstruction_localized` erfordert nicht mehr nur
  `control_regression <= threshold`, sondern zusätzlich ausreichende
  Entscheidungskonfidenz aus Zielabdeckung, Kontrollfenster-Leckage und
  Threshold-Sicherheitsmarge.
- **Scope-Erweiterung**: Counterfactual-Logik gilt jetzt konsistent für
  rekonstruktive Prefixe `phase_23`, `phase_24`, `phase_50`, `phase_55`
  (inkl. `SPECTRAL_HOLES`-/`DROPOUTS`-Key-Mapping).
- **Neue PMGG-Telemetrie** in `PhaseGateLogEntry.metadata`:
  `pmgg_reconstruction_localized`, `pmgg_reconstruction_confidence`,
  `pmgg_reconstruction_reason`, `pmgg_reconstruction_target_coverage`,
  `pmgg_reconstruction_control_coverage`, `pmgg_reconstruction_control_regression`.
- **Unsicherheitsgründe explizit**: Bei fehlendem Counterfactual-Fenster oder
  aktivem Content-Integrity-Guard werden die Gründe
  `counterfactual_window_unavailable` bzw. `content_integrity_guard_active`
  in der PMGG-Metadatenebene dokumentiert.
- **Tests**: neue Unit-Suite
  `tests/unit/test_pmgg_reconstruction_confidence.py` (Konfidenz-Logik +
  Metadatenpropagation).

## v10.0.0 (14. Mai 2026) — §0p Vocal-Supremacy-Invarianten vollständig implementiert

## v10.0.0 (4. Juni 2026) — PMGG Reconstruction-Collateral-Guard + Phase-01 Stereo-Lokalität

- **PMGG rekonstruktive Kollateralprüfung**: `backend/core/per_phase_musical_goals_gate.py` bewertet
  `phase_24_dropout_repair` und `phase_55_diffusion_inpainting` nicht mehr ausschließlich im defektzentrierten
  Stichprobenfenster. Ein zusätzliches defektfernes Kontrollfenster erlaubt `passed_reconstruction_localized`,
  wenn außerhalb der Rekonstruktionszone keine Regression vorliegt und der Content-Integrity-Guard sauber bleibt.
- **Phase 55 Per-Gap-Orakel**: `backend/core/phases/phase_55_diffusion_inpainting.py` nutzt nun ein echtes
  lokales Strength-Orakel pro Gap (250-ms-Kontext, Gap-Schwere, Gap-Dauer, VFA-Schutzzonen) statt rein globalem
  `safe_strength`-Blend.
- **Phase 01 Linked-Stereo Sparse Patch Transfer**: `backend/core/phases/phase_01_click_removal.py` ersetzt den
  globalen Mono-Gain-Transfer durch mono-gekoppelte Ereignisdetektion und kanalweise lokale Patch-Reparatur.
  Saubere Gegenkanäle bleiben außerhalb echter Click-Evidenz unverändert; starke Einzelereignisse nutzen einen
  lokalen DeepFilterNet-Patchpfad mit kurzem Crossfade.
- **Testabdeckung**: neue PMGG-/Phase-55-Unit-Tests plus Phase-01-Stereo-Regressionstest; fokussierte und schmale
  Integrationsblöcke grün.

- **§0p Vibrato-Schutz**: `VFAResult.vibrato_zones` + `VocalFocusAnalyzer._detect_vibrato()` (via
  `natural_performance_detector`); UV3 propagiert `vibrato_zones` als Phase-kwarg + cappt Strength
  auf 0.20 in Vibrato-Zonen.
- **§0p Formant-Integritäts-Wächter**: Post-Phase Formant-Guard für `_FORMANT_GUARD_PHASES`
  (phase_03/20/29/42/49): LPC F1/F2-Verifikation pre/post; Überschreitung > 2 dB → sofortiger
  Rollback auf Phase-Input.
- **phase_55 HNR-Blend (§0p v10.0.0)**: `phase_55_diffusion_inpainting` zu `_NR_PHASES_HNR`
  hinzugefügt — Diffusions-Inpainting kann Stimmharmonik nicht mehr ohne ΔHNR-Guard halluzinieren.

## v10.0.0 (21. Mai 2026) — Strict-Conflict: Psychoakustische Severity-Orchestrierung + Telemetrie-Kohärenz

- **Tape-Spezialdefekte produktiv verdrahtet (SOTA-Lückenschluss)**:
  - Neuer DefectType `SCRAPE_FLUTTER` inkl. materialadaptiver Scanner-Thresholds, Full-Audio-Detektion,
    Cross-Material-Fallback-Gate und End-to-End-Routing über CausalDefectReasoner + DefectPhaseMapper.
  - Neuer DefectType `TAPE_HEAD_CLOG` inkl. lokaler HF-Bandlücken-Detektion, Randtransienten-Guard,
    Cross-Material-Fallback-Gate und End-to-End-Routing über CausalDefectReasoner + DefectPhaseMapper.
  - Primäre Phasenrouten:
    - `scrape_flutter` → `phase_12_wow_flutter_fix` (+ `phase_31_speed_pitch_correction`)
    - `tape_head_clog` → `phase_56_spectral_band_gap_repair` (+ `phase_25_azimuth_correction`)
  - Neue Regressionstests: `tests/unit/test_tape_specialist_defects.py` sowie Mapper-Absicherung
    in `tests/test_autonomous_restoration_engine.py`.

- **Strict-Conflict-Relevanzfilter (UV3 Report)**:
  - Pass-äquivalente Entscheidungen werden konsistent ignoriert (`passed`, `passthrough`,
    `sub_threshold`, `passed_p4p5_tolerated`, `best_effort_accepted` sowie
    `pmgg_decision_reason`-basierte Pass-Gründe).
  - `best_effort_phases` berücksichtigt nur konfliktrelevante Aktionen.
- **Reason-gewichtete Runtime-Adaption** (`_register_phase_goal_conflict_event`):
  - `reason_decay_weight` und `reason_cap_tighten_weight` steuern Family-Decay und Cap-Tightening.
  - `pmgg_best_effort` gezielt milder; `best_effort_emergency` gezielt strenger.
  - `artifact_freedom_rollback` skaliert streng nach Defizit (`0.95 - artifact_freedom`).
  - `vocal_no_harm_rollback` mit `singer_identity=False` bekommt Zusatzgewicht.
- **Goal-Prioritätskopplung**:
  - P1/P2-Regr. werden bei gleicher Magnitude strenger gewichtet als P4/P5-Regr.
  - optionaler Song-Weight-Einfluss (`_song_goal_weights`) in der Severity-Bildung.
- **Vocal-Supremacy-Kopplung**:
  - bei `panns_singing >= 0.35` zusätzliche Schärfung für vokalkritische Goals
    (`natuerlichkeit`, `authentizitaet`, `artikulation`, `timbre_authentizitaet`,
    `vocal_quality`, `formant_fidelity`).
- **Strict-Report-Upgrade** (`_build_strict_phase_goal_conflict_report`):
  - `regressive_phase_events[*].severity` neu.
  - `regressive_weight_sum` neu; `conflict_score` nutzt gewichtete statt reine Event-Anzahl.
  - `runtime_severity_sum` und `runtime_severity_max` neu.
- **Severity-Fingerprint-Telemetrie** pro Runtime-Event:
  - `severity_score`, `severity_bucket`, `severity_fingerprint`
  - Fingerprint-Felder: `reason_weight_base`, `dynamic_weight`, `vocal_presence`,
    `goal_priority_peak`, `artifact_deficit`, `emergency_action`.
- **Unkonventioneller Disagreement-Guard (neu)**:
  - Bei hoher Reason-Diversität im jüngsten Family-Verlauf (`>=3`) wird
    `pmgg_best_effort` adaptiv gedämpft (`disagreement_brake_applied=True`),
    um Fehlübersteuerung bei unsicheren Konfliktsignalen zu vermeiden.
  - Guard wird bei jüngstem harten Reason automatisch deaktiviert
    (`artifact_freedom_rollback`, `noise_texture_rollback`,
    `vocal_no_harm_rollback`, `hf_hallucination_rescue`).
- **Test-/Security-Status für den gesamten Maßnahmenblock**:
  - `tests/unit/test_strict_conflict_artifact_feedback.py` sukzessive erweitert (bis 24 Tests grün)
  - PMGG/UV3-Cluster zuletzt: **556 passed**
  - Pyright auf Kernpfaden: **0 Fehler**
  - Snyk (`backend/core`): **issueCount=0**

## v10.0.0 (14. Mai 2026) — Musical-Goals-Metriken: 6 Metadaten-Propagierungslücken + Snyk-Sicherheitsfixes

- **BrillanzMetric**: Material-adaptive HF-Crest-Formel (tape/cassette offset=0.10/divisor=1.20;
  reel_tape offset=0.05/divisor=1.40); Ceilings rekalibriert (tape 0.42→0.78, reel_tape 0.52→0.85).
- **HolisticPerceptualGate**: `logger.debug → logger.warning(exc_info=True)` — HPG-Exceptions
  nicht mehr unsichtbar — §2.44-Gate chronisch aktiv.
- **ExcellenceOptimizer + §2.48**: `material_type`-Propagation in alle `measure_all()`-Aufrufe.
- **EraAuthenticPerceptualCompletion**: Material-Ceiling-Grenze für BW-Synthese (§2.46e).
- **`_fast_goal_snapshot()`**: Material-adaptiver Natürlichkeits-Proxy-Boden (tape/cassette: 0.10).
- **Snyk-Sicherheitsfixes (OWASP)**: torch 2.2.2→2.7.0 (CVE-2025-32434), setuptools≥70.3.0
  (CVE-2024-6345), pillow≥11.0.0, sympy≥1.13.1,<1.14.0, recharts ^3.8.0.

## v10.0.0 (Mai 2026) — Musical-Goals-Metriken: Material-adaptive Böden + Tape-Ceiling-Korrektur

- 4 systematische Kalibrierungsfehler in Musical-Goals-Metriken behoben.
- Material-adaptive Böden für Shellac (~0.62), Vinyl (~0.72), CD (~0.82/0.90).
- Tape-Ceiling-Korrektur: korrekte Formel + Boden-Kalibrierung für reel_tape und cassette.

## v10.0.0 (Mai 2026) — Perceptual-Quality-Bugfixes: Echo + Kratzig + Pegelexplosion

- 3 perceptual-quality Bugfixes: Echo-Artefakte, Kratzig-Detektor-False-Positives,
  Pegelexplosion in leisen Zonen behoben.

## v10.0.0 (Mai 2026) — BrillanzMetric-Offset-Rekalibrierung + Noise-Floor-Test-Fix

- BrillanzMetric-Offset-Rekalibrierung für konsistente Scores über Materialtypen.
- Noise-Floor-Test-Fix für deterministisches Testergebnis.

## v10.0.0 (Mai 2026) — BrillanzMetric + TransparenzMetric Sparse-Signal-Fixes

- Sparse-Signal-Fixes: verhindern Score-Kollaps bei armütigem Spektrum.

## v10.0.0 (Mai 2026) — HPSS-Kernel-Swap-Fix + AMRB-LUFS-Root-Cause

- HPSS-Kernel-Swap rückkorrigiert; AMRB LUFS Root Cause identifiziert und behoben.

## v10.0.0 (Mai 2026) — Vocal-Supremacy + VocalFocusAnalyzer + §0p

- **Neues Kernmodul `VocalFocusAnalyzer`**: `VocalRegisterDetector`, `FrissonCandidateDetector`,
  LPC-FormantTracker, Passaggio-Detektion — injiziert `vfa_result` in `_restoration_context`.
- **§0p Vocal-Supremacy-Doktrin** in copilot-instructions.md: 4-stufige Hierarchie,
  VQI-Gate als zweiter Recovery-Trigger, Vocal-DSP-Invarianten, HPI-4-Varianten.
- **phase_29 Stereo-Lag-Fix**: channels-last-Normalisierung + Loudness-Rescue-Gate +
  inter-channel Lag-Korrekturteiemetrie.
- **Spec-Updates**: pipeline.instructions.md, dsp.instructions.md, phases.instructions.md,
  musical_goals.instructions.md — alle §0p-Regeln normativ ergänzt.

## v10.0.0 (Apr 2026) — §2.51a Stereo-Hörsicherheitsprofil

- Dreistufiges Stereo-Guardrail-Profil: Hard-Fail (Delay > 1.0 ms, Imbalance > 6 dB,
  True-Peak > -1.0 dBTP), Warnstufe, Zielwerte.

## v10.0.0 (Apr 2026) — §2.45a Cumulative-Guard Decoupling + §2.30b ADMM Wall-Time

- `_cum_rms_reference_audio` immer initialisiert (unabhängig von AFG-Singleton) —
  verhindert unsichtbar deaktivierten Pegelschutz bei AFG-Init-Fehler.
- ADMM Wall-Time Fixes in phase_23.

## v10.0.0 (Apr 2026) — §09.2 Adaptive Goal Thresholds → PMGG Propagation

- Ära-adaptive Schwellwerte aus `calibration_matrix` propagieren korrekt in PMGG.

## v10.0.0 (Apr 2026) — AMD GPU: RDNA4 + vollständige APU-Abdeckung

- AMD GPU Mixed-Mode: RDNA4-Support, APU-Tier-Erkennung, DirectML (Windows) + ROCm (Linux).

## v10.0.0 (Apr 2026) — Globaler Quality-First-Schalter + 68-Phasen-Audit

- `UV3._profiled_phase_call()` injiziert `quality_first_unleashed=True` global in
  `quality`/`maximum`/studio-Modi — alle 68 Phasen konsistent auf Quality-First-Policy.
- Neuer 68-Phasen-Policy-Test: `test_quality_first_policy_68_phase_audit.py`.

## v10.0.0–19 (Apr 2026) — Quality-First, PLM-Guards, Peak-Guard, Stereo-Fixes

- v10.0.0: Quality-First Zeitgates über phase_06/12/19/42/49 gehärtet.
- v10.0.0: AudioSR-Watchdog in phase_06.
- v10.0.0–24: Peak-Guard Conformity + Wide-Stereo-Guard phase_13/14 (R11 UAT Fix).
- v10.0.0–39: PLM set_active Guards für alle schweren ML-Plugins (CREPE, SGMSE+, AudioSR,
  DiffWave, MIIPHER, FCPE, Vocos, BigVGAN, ResembleEnhance, HiFiGAN, BanquetVinyl u.a.).
- v10.0.0: Stereo-Slicing-Bug + Ketten-Pflichtphasen + corrcoef NaN-Guard.

> Vollständige Details zu v10.0.0–19 und allen Patch-Releases: `CHANGELOG.md`

---

## v10.0.0 (7. April 2026) — Deep-Transfer-Chain-Nachschärfung (3+/4+)

- **[RELEASE_MUST] §2.46a Deep-Transfer-Chain-Pflicht — NEU** in Slim Core:
  Importsongs mit 3+ Tonträgerstufen müssen vollständig modelliert werden; keine
  Verkürzung auf Primärträger + eine Sekundärstufe.
- **Zwischenstufen-Pflicht**: Digitale Zwischenstufen (`cd_digital`, `dat`) sind bei
  Evidenz verpflichtend zu führen, bevor ein lossy Codec-Layer (`mp3_low`, `aac`, ...)
  angehängt wird.
- **Kausalitäts-Invariante**: Kettenreihenfolge bleibt gemäß `_MEDIUM_ORDER` monoton;
  keine Rückwärtssprünge in der Transferkette.
- **Normalisierungs-Invariante**: Nach Material-Key-Normalisierung müssen Duplikate
  konsolidiert werden (Konfidenz = `max`), um künstlich erhöhte
  `source_fidelity_generation_count` zu vermeiden.
- **Testpflicht bindend erweitert**: Mindestens ein Unit-Test für 4-stufige Kette mit
  digitaler Zwischenstufe und ein Unit-Test für `file_ext=.mp3` mit physikalischer
  Inferenz + 4-stufigem Ergebnis; zusätzlich Integrationstest für unveränderte
  Durchreichung bis `metadata["song_calibration"]`.
- **Spec 05 §6.7 aktualisiert**: Phase-2-Transferkettenaufbau beschreibt jetzt
  explizit Deep-Chains (3+/4+/5+), Zwischenstufen-Regeln und Testpflichten.
- **Betroffene Dateien**: `.github/copilot-instructions.md`,
  `.github/specs/05_material_system.md`.

### v10.0.0 Nachtrag — Teststabilisierung Audit-Trail (7. April 2026)

- Integrationstest für Source-Fidelity-Audit-Trail auf deterministischen,
  zeitsicheren Datenfluss fokusiert (heavy Runtime-Pfade im Testkontext gestubbt),
  ohne normative Export-Anforderungen zu lockern.
- Ziel: Verlässliche CI-Aussage für die Pflichtfelder
  `source_fidelity_transfer_chain`, `source_fidelity_generation_count`,
  `source_fidelity_hf_loss_db`.

## v10.0.0 — instructions_version 6.2 (7. April 2026) — Klangwahrheits-Tiefenrevision

- **instructions_version**: 6.1 → **6.2** — Klangwahrheits-Tiefenrevision
- **§0 Oberstes Prinzip — reformuliert**: „Originale Performance hören" statt „besser klingen
  als der Input." Für Restoration ist das Ziel Ununterscheidbarkeit vom Studio-Original,
  nicht bloße Verbesserung.
- **§0a Modus-Differenzierung — erweitert**: Neue Zeilen Rauschboden (Restoration: material-adaptiv;
  Studio 2026: ≤ −72 dBFS), Qualitätsmaß (Restoration: Nähe zum Original; Studio 2026: PQS-Improvement),
  Authentizität (akustisch nicht unterscheidbar vs. musikalische Intention modernisieren).
  Klangziel Restoration: „Tonträgerkette invertieren".
- **§2.44 HPI-Gate — Restoration-Formel korrigiert**: `PQS_improvement` ersetzt durch
  `timbral_fidelity(input, output)` — misst akustische Nähe zum Original statt „Verbesserung."
  timbral_fidelity = MFCC-Distanz + Spectral-Envelope-Korrelation + Crest-Factor-Erhalt.
- **§2.46 Carrier-Chain-Inversion — NEU**: Explizites Prinzip für Restoration — nicht Einzel-Defekte
  reparieren, sondern die **gesamte Tonträgerkette invertieren** (ADC-Artefakte → Playback →
  Alterung → Carrier-Encoding, dabei Mixer/Preamp-Charakter + Studio-Raumklang bewahren).
  Studio 2026: + Enhancement-Kette, Mixer/Preamp darf modernisiert werden.
- **Rauschboden — modus-differenziert**: Restoration: material-adaptiv (Shellac ≤ −45, Vinyl ≤ −55,
  Tape ≤ −60, Digital ≤ −72 dBFS) — kein aggressiveres Denoising als das Original-Aufnahmemedium.
  Studio 2026: ≤ −72 dBFS (unverändert). Aktualisiert in: Spec 01 §8.2, Spec 02 §1.4, fix-metric,
  quality-benchmark.
- **WärmeMetric §9.7.14 — reformuliert**: Primär-Proxy: Even-Harmonic-Ratio (H2/H4 THD_even/THD_total,
  ISO 226:2003 gewichtet) — misst wahrgenommene Wärme von Röhren-/Bandsignalketten. Sekundär:
  Spektral-Band-Ratio E(200–800)/E(800–3000) als Tilt-Proxy.
- **Studio-2026-Schwellwerte synchronisiert**: Brillanz ≥ 0.90, Bass-Kraft ≥ 0.88, Raumtiefe ≥ 0.78,
  Separation ≥ 0.85 — konsistent zwischen Spec 01, Spec 02 und fix-metric SKILL.
- **Spec 02 ergänzt**: §2.44/§2.45/§2.46 als vollständige Abschnitte, §1.4 Modi mit Rauschboden-
  und HPI-Gate-Referenzen.
- **Betroffene Dateien**: copilot-instructions.md (Slim Core), Spec 01, Spec 02, fix-metric SKILL,
  pipeline-debug SKILL, quality-benchmark SKILL.
- Keine Code-Änderungen. Alle Tests unverändert.

### v10.0.0 Nachtrag 2 — Referenz-Paradoxon, Interaktions-Guard, Artefakt-Gate (6. April 2026)

- **§0a Rauschboden**: Nicht nur **Niveau** sondern auch **Textur** des Restrauschens muss dem
  originalen Trägerprofil entsprechen. „Kein weißes Rauschen nach Vinyl-Denoising."
- **§2.44 HPI-Gate — Referenz-Paradoxon explizit gelöst**: `timbral_fidelity` misst jetzt
  **strukturelle Klangkohärenz** (Spectral-Envelope-Kontinuität, Crest-Factor-Konsistenz,
  MFCC-Stabilität), nicht bloße Ähnlichkeit zum degradierten Input. Restorability-abhängiger
  Referenz-Anker: > 70 → Input-Referenz; ≤ 50 → MERT-Referenz-Vektor aus GP-Memory
  (genre × material × ära). `artifact_freedom` (§2.49) als neuer Multiplikator in beiden
  HPI-Formeln.
- **§2.44 MERT-Referenz-Embedding-Aufbau**: 36 Bootstrap-Prototypen (12 Genres × 3 Ära-Bins)
  im Bundle; inkrementell verfeinert per EMA (α = 0.15) nach jeder erfolgreichen Restaurierung
  (HPI > 0.5 + artifact_freedom ≥ 0.95). 5-stufige Fallback-Kaskade bis hin zu rein gegen Input.
  Qualitäts-Gate: nur Outputs mit HPI > 0.5 fließen in Referenz-Aufbau ein.
- **§2.44 emotional_arc_preservation erweitert**: Jetzt Arousal/Valence-Bogen + **Makrodynamik**
  (Vers-/Refrain-/Bridge-Pegelrelationen) + **Lyrics-Salienz** (§2.36 Phonem-Boost-Konsistenz).
- **§2.48 Kumulative-Phasen-Interaktions-Guard — NEU** (Slim Core + Spec 02 + pipeline-debug):
  Kumulative P1/P2-Drift-Messung nach jeder Phase. Drift < −0.05 → Rollback auf best_checkpoint.
  5 kritische Interaktions-Paare definiert. **Kumulative STFT-Phasenkohärenz**: Nach ≥ 3
  STFT-Phasen: Gruppenlaufzeit-Deviation ≤ 2 ms, sonst Rollback.
- **§2.49 Artefakt-Freiheits-Gate — NEU** (Slim Core + Spec 02 + pipeline-debug + new-phase):
  5 Artefakttypen + **Rauschtextur-Kohärenz** (Spectral-Tilt-Differenz: ≤ 3 dB/Oktave OK,
  > 6 dB/Oktave Rollback). **Material-adaptive Schwellwerte**: Shellac toleranter
  (Musical Noise > 22 dB), Digital streng (> 12 dB). Selbstkalibrierung nach 3 Verarbeitungen.
- **Material-Ähnlichkeitsmatrix** (Spec 02 §2.47): Explizite 9×9-Matrix für GP-Cross-Material-
  Transfer. Transferierbarkeits-Regeln mit Lengthscale/Varianz-Skalierung, Mindest-Ähnlichkeit 0.3.
- **Betroffene Dateien**: copilot-instructions.md, Spec 02, fix-metric SKILL, pipeline-debug
  SKILL, new-phase SKILL.
- Keine Code-Änderungen. Alle Tests unverändert.

### v10.0.0 Nachtrag — Adaptive-Intelligence-Erweiterung (6. April 2026)

- **§2.47 Adaptive-Intelligence-Prinzip — NEU** (Slim Core + Spec 02): Übergeordnetes Prinzip,
  das die Einzelmechanismen (Material-Erkennung, Ära, Genre, Restorability, Defektanalyse,
  Song-Kalibrierung, GPOptimizer) als kohärente Adaptions-Kaskade definiert. Dieselbe Pipeline
  verarbeitet Schellack 1928 fundamental anders als CD 2005 — ohne manuellen Eingriff.
- **§2.31d Kombinierte Extrembedingungen** (Spec 01): Kaskadierung bei gleichzeitigen
  Extremfaktoren (Restorability < 20 + Shellac, Era ≤ 1940 + BW < 5 kHz, Dateilänge < 10 s
  oder > 60 min). Scale-Factor 0.65 für schwerstbeschädigtes Material.
- **§2.31e Prior-Konflikt-Auflösung** (Spec 01): Material-Prior = Vorrang bei physikalischen
  Grenzen; Ära-Prior = Vorrang bei ästhetischen Entscheidungen.
- **ML-Failure-Degradations-Kaskade** (Spec 02 §2.47 + ml-plugin SKILL): Vollständige
  Fallback-Tabelle (9 ML-Kaskaden: DeepFilterNet→OMLSA→Spectral-Gating, MDX23C→NMF→Bypass,
  AudioSR→NVSR→SBR, CREPE→pYIN→YIN, MERT→MFCC→Bypass etc.). Invariante: Kein ML-Failure
  darf die Pipeline abbrechen.
- **GP-Wissenstransfer** (Spec 02 §2.47): Cross-Material-Generalisierung bei < 10 Beobachtungen;
  Batch-Konvergenz bei sequenzieller Verarbeitung gleichen Materials.
- **DEFAULT_RESTORATION_PROFILE** (Spec 03): Explizites neutrales Profil für 12 Genres ohne
  eigenes Restaurierungsprofil (Pop, Blues, Soul, Country, Folk, Funk, Electronic, Hip-Hop,
  Metal, Latin, Gospel, Reggae). GP-Memory pro Genre aktiv → lernt genre-spezifisch.
- **Edge-Cases** (Slim Core + Spec 01): Dateilänge < 10 s → Goal-Deaktivierung + FeedbackChain
  max 2 Iter; > 60 min → Segment-Verarbeitung; Restorability < 20 + Shellac/Wax → „Hörbar
  machen" statt „Originalgetreu restaurieren".
- **Betroffene Dateien**: copilot-instructions.md, Spec 01, Spec 02, Spec 03, pipeline-debug
  SKILL, ml-plugin SKILL.
- Keine Code-Änderungen. Alle Tests unverändert.

## v10.0.0 — instructions_version 6.1 (6. April 2026) — Klangprinzipien-Revision

- **instructions_version**: 6.0 → **6.1** — Klangprinzipien-Revision
- **§0 Oberstes Prinzip — Klangwahrheit**: Neu in Slim Core. Drei hierarchische Leitprinzipien
  (Primum non nocere > Minimal-Intervention > Perceptuelle Verbesserung). §0 ist normativ
  übergeordnet — Klang schlägt Metrik. §0a Modus-Differenzierung: Restoration (Original-Charakter)
  vs. Studio 2026 (Studio-Klang auf internem Spitzenniveau, maximal-zielgerichtete Intervention).
- **§2.29d Differenziertes Regressions-Regime**: P1/P2 bleiben hart (keine Regression erlaubt).
  P3–P5 neu: Pipeline-Netto-Budget — Einzelphasen dürfen vorübergehend verschlechtern, wenn am
  Kettenende alle Goals ≥ Schwellwert. Verhindert übervorsichtiges Wet/Dry bei De-Hiss/EQ.
- **§2.44 Holistic Perceptual Gate (HPI)**: Neues letztes Gate vor Export. Misst Gesamt-Hörverbesserung
  statt nur Einzel-Goals. Modus-differenziert: Restoration (MERT-dominant), Studio 2026 (PQS-dominant +
  studio_quality_gain).
- **§2.45 Minimal-Intervention-Prinzip**: Phasen ohne perceptual_delta > 0 werden übersprungen.
  Modus-differenziert: Restoration (minimal), Studio 2026 (volle Kette, aber jede Phase muss Klanggewinn nachweisen).
- **§9.7.16 NatuerlichkeitMetric-Reform [TARGET_2026]**: Modus-differenzierte Reformulierung.
  Restoration: MERT-Distanz zum Input. Studio 2026: MERT-Distanz zu Studio-Referenzen.
  Signal-Statistiken (Flatness, ZCR) durch Wahrnehmungs-Features ersetzen.
- Skills aktualisiert: `new-phase`, `fix-metric`, `pipeline-debug`, `quality-benchmark` — alle
  mit Modus-Differenzierung für Restoration vs. Studio 2026.
- Keine Code-Änderungen. Alle Tests unverändert.

## v10.0.0 — instructions_version 6.0 (6. April 2026) — Skills-Architektur

- **instructions_version**: 5.0 → **6.0** — Skills-Architektur
- **copilot-instructions.md**: Monolithische 890-Zeilen-Datei → **171-Zeilen Slim Core**.
  Detailwissen ausgelagert in 9 aufgabenspezifische Skills unter `.github/skills/*/SKILL.md`:
  - `new-phase` — Phasen-Interface, PMGG-Exclusions, Vocal-Kette, LyricsGuided §2.36
  - `fix-metric` — 14 Musical Goals, Sub-Metriken, Recalibration §9.7.15, Stable-Metric §2.29b
  - `aurik-dsp-decision` — SOTA-Matrix, DSP-Pflichtregeln, MRSA-Zonen, Vintage, Dithering
  - `ml-plugin` — Memory-Budget, Headroom-Guard, Fallback-Kaskaden, ONNX/Torch-Config
  - `pipeline-debug` — Denker, UV3-Reihenfolge, SongCalibration, KMV §2.38, OOM §2.39, Determinismus §2.40
  - `ui-feature` — Thread-Safety, Signale, Progress, Shortcuts, Bridge, Preanalysis-Gate
  - `test-writing` — Marker, Heavy-Isolation, CI-Gates, Test-Pattern, Task-Runner
  - `quality-benchmark` — OQS/AMRB, PQS-MOS, quality_estimate, MUSHRA §8.4, Modi
  - `aurik-architecture-diagram` — Mermaid-Diagramme (bereits vorhanden)
- **Motivation**: ~80 % der always-loaded Instructions waren für jede einzelne Aufgabe irrelevant.
  Skills-Architektur reduziert auf ~600 Tokens always-loaded + ~2000 Tokens task-relevant.
- **Backup**: `copilot-instructions-v5-full.md` enthält die vollständige v5.0-Version.
- Keine Code-Änderungen. Alle 8 Specs unverändert. Tests unverändert.

## v10.0.0 (6. April 2026) — Spec-Sync: instructions_version 4.2 → 5.0

- **instructions_version**: 4.2 → **5.0**
- **copilot-instructions.md**: Header auf v10.0.0 aktualisiert. Neue §-Nummern:
  - §2.41 Denker-Vollkontext — Material-adaptive DSP-Reparatur (v10.0.0)
  - §2.42 SourceFidelityReconstructor — Generationsverlust-Kompensation (v10.0.0–116)
  - §2.43 Phase-Preserved Wet/Dry-Blend (v10.0.0)
  - §9.7.15 Musical-Goals-Metriken-Recalibration (v10.0.0)
  - Weitere Fixes v10.0.0–121 (Phase 09/29/40/42/55, HPSS, ExcellenceOptimizer, MDEM, EmotionalArc, OMLSA, De-Esser, Phase 12, load_audio_file)
- **Spec 03**: `SourceFidelityReconstructor`, `PerceptualSalienceEstimator`, `MediumDetector` in §2.1 Pflicht-Kernmodule aufgenommen. HPSS Kernel 31→17/13 aktualisiert.
- **Spec 06**: Phase 06/38/39 mit SourceFidelity-Integration, Phase 55 adaptiver AR-Order.
- **Spec 08**: A/B-Sync-Loop + Queue-Drag-&-Drop, Plugin-Anzahl 51. Keyboard-Shortcuts erweitert.
- **pyproject.toml**: Version 9.10.103 → 9.10.121 synchronisiert.
- Testzahl: ~9990+ `def test_`-Funktionen in 375 Testdateien.

## v10.0.0 (4. April 2026) — Defect-Locations-Completeness (Core uncapped)

- **instructions_version**: 4.1 -> **4.2**
- **copilot-instructions.md**: Defect-Locations-Flow um normative Completeness-Invariante ergänzt: harte Caps auf `defect_locations` im Analyse-/Reparaturpfad verboten; UI-Dichte-Reduktion nur als Anzeige erlaubt.
- **Spec 06** (`.github/specs/06_phases_system.md`): `[RELEASE_MUST] Location-Completeness-Invariante` ergänzt (Core uncapped, vollständige Eventlisten auch bei hoher Dichte, keine UI-Rückwirkung auf Routingdaten).
- **Spec 07** (`.github/specs/07_quality_and_tests.md`): Pflicht-Testfall ergänzt: synthetische Signale mit >50 non-stationary Events müssen >50 `locations` liefern; feste Core-Caps (50/100/256) gelten als Regression.

## v10.0.0 (4. April 2026) — Genre-Phase-2: 17-Genre-Härtung + Disambiguation-Gates

- **instructions_version**: 4.0 -> **4.1**
- **copilot-instructions.md**: Neuer Abschnitt `§2.19 Genre-Classifier-Härtung (17 Genres, [RELEASE_MUST])`.
- **Spec 03** (`.github/specs/03_cognitive_modules.md`): Normative Erweiterung der Non-Schlager-Open-Set-Logik und Anti-Falsifikations-Matrix bestätigt.
- **Bindende Disambiguation-Gates**: `Funk` (warmes Centroid-Fenster), `Latin` (BPM-kontextierter Centroid-Bonus + Mindest-Centroid), `Electronic` (Synth-Centroid-Gate), `Hip-Hop` (Vokal/Sample-Centroid-Gate), `Reggae` (Tempo-Gate), `Folk` (DR-Guard), `Jazz` (HSI-Guard).
- **Testhärtung**: Unit-Test-Isolierung dokumentiert (nicht getestete neue `_score_*`-Methoden auf Neutralwert patchen), um Open-Set-Margin-Kollapse durch Feature-Artefakte zu verhindern.

## v10.0.0 (3. April 2026) — Genre-Phase-1: Family-Stage + Top-k + Open-Set + Lyrics-Fusion

- **Code** (`backend/core/genre_classifier.py`):
  - `SchlagerClassificationResult` erweitert: `genre_family`, `genre_family_confidence`, `top_genres`, `open_set_unknown`
  - Neue interne Stufen: `_compute_non_schlager_scores()`, `_infer_genre_family()`, `_build_top_genres()`, `_is_open_set_unknown()`
  - Open-Set-Regel: zu niedriger Top-Score oder zu geringe Top1-Top2-Margin → `genre_label="Unbekannt"`
- **Lyrics-Fusion**: DSP-Sprachscore + §2.36-Lyrics-Hinweis (max-basierter Merge) → reduziert Jazz-Fehlklassifikation bei deutschsprachigen Schlager-Aufnahmen
- **UI** (`Aurik10/ui/modern_window.py`): Tooltip zeigt Genre-Familie, Top-k, Open-Set-Status; Genre-Badge mit Ampelpunkt (Grün ≥0.70 / Gelb 0.50–0.69 / Rot < 0.50)
- **Tests**: 3 neue Tests für Family/Top-k/Open-Set

## v10.0.0 (3. April 2026) — Dokumentations-Sync: Phasen 01–64 + Kausal-Mapping

- **Spec 06** (`.github/specs/06_phases_system.md`): Phasenliste auf **01–64** aktualisiert; `phase_57` (Print-Through) / `phase_58` (Lyrics) korrekt zugeordnet; Phase 59–64 ergänzt; `CAUSE_TO_PHASES` um neue Defektursachen (modulation_noise, inner_groove_distortion, groove_echo, crosstalk, intermodulation_distortion, tape_splice_artifact) erweitert
- **Spec 02**: `processing_sr=48000` auf Phasen 01–64 aktualisiert
- **Spec 08**: `phase_output_guard`-Scope auf 01–64 aktualisiert
- **copilot-instructions.md**: UV3-Kernreihenfolge und SR-Regeln auf 64 Phasen vereinheitlicht

## v10.0.0 (3. April 2026) — Normative Nachschärfung: Tonträgerkette + Lyrics-Produktivpfad

- **copilot-instructions.md**: Autoritatives Produktionsmodul auf `backend/core/lyrics_guided_enhancement.py` festgelegt; Docker-/MFA-Altpfade aus `backend/lyrics_guided/` als nicht-normativ markiert
- **Datenschutz-Guard**: Unzulässige Persistenz von Worttext/Transkript in Logs, metadata, Checkpoints explizit verboten
- `_carrier_bg`-Pflicht-Invariante: `get_medium_detector().detect()` statt `medium_classifier.classify_medium()` (file_ext-Kontext)

## v10.0.0 (3. April 2026) — EmotionalitaetMetric MERT-Blend + WaermeMetric-Guard + AMRB-Codec-Kalibrierung

- **Code** (`backend/core/musical_goals/musical_goals_metrics.py`): `EmotionalitaetMetric` erhält MERT-Arousal-Blend; `WaermeMetric` mit reverb-invariantem Sub-Band-Verhältnis (§9.7.14 Nachfolge)
- **AMRB-05 Pre-Echo**: Spec 01 Baselines für CODEC-Szenarien kalibriert

## v10.0.0 (3. April 2026) — Codec-Reparatur: Apollo DSP-Fallback + Phase-23-Integration + AMRB-05-Pre-Echo

- **Phase 23**: AudioSR-Inpainting erhält expliziten Fallback-Pfad über Apollo-Plugin für Codec-Artefakte
- **Apollo DSP-Fallback**: Strukturierter Fallback in `plugins/apollo_plugin.py` für OOM-Szenarien
- **AMRB-05**: Pre-Echo-Szenario Baseline-Kalibrierung für `mp3_low`/`aac`

## v10.0.0 (3. April 2026) — AMRB-Kalibrierung SHELLAC/CODEC/VOCAL + P4-AudioLDM2-Cascade

- **AMRB**: Shellac/Codec/Vocal-Szenarien (AMRB-01, 05, 09) mit aktualisierten Baselines kalibriert
- **AudioLDM2-Kaskade**: P4-Inpainting-Fallback für AMRB-05-Pre-Echo-Reparatur dokumentiert

## v10.0.0 (30. März 2026) — §2.29c Restorative-Phase-Baseline-Capping + PMGG Exclusion-Fixes

- **instructions_version**: 4.0
- **§2.29c** (neu): Defekt-inflationierte Baselines werden gedeckelt (`_RESTORATIVE_PHASES` + `_CANONICAL_THRESHOLDS` + `effective_scores_before`)
- **§2.31b Material-adaptive PHASE_GOAL_EXCLUSIONS**: `cd_digital`/`dat` → phase_03/phase_29 auf `{"natuerlichkeit", "artikulation"}` reduziert; brillanz/transparenz/waerme (§9.7.12/13/14 SNR-robust) aus allen Materialausschlüssen entfernt
- **Tests**: 122 PMGG-Tests in `test_per_phase_musical_goals_gate.py`

## v10.0.0 (30. März 2026) — §9.7.11 ext: tonal_center in phase_03/phase_29 PMGG-Exclusions

- `tonal_center` und `timbre_authentizitaet` zu phase_03/phase_29 PHASE_GOAL_EXCLUSIONS hinzugefügt (shaped NR → K-S volatile + Centroid-CV-Disturbance)

## v10.0.0 (30. März 2026) — §2.31a Iterative Mid-Pipeline-Kalibrierung

- `_build_song_calibration_profile()` kann während der Phasenkette iterativ aktualisiert werden; Kalib.-Profil-Invalidierung bei starken Defektänderungen

## v10.0.0 (30. März 2026) — §9.7.11 K-S + TonalCenterMetric aus _PRECISE_METRICS + K-S Hanning-Fix

- `TonalCenterMetric` aus `_PRECISE_METRICS` entfernt; Krumhansl-Schmuckler-KDE mit korrektem Hanning-Window implementiert

## v10.0.0 (30. März 2026) — PMGG tonal_center §9.7.11 Krumhansl-Schmuckler-Proxy

- **instructions_version**: 3.9 → **4.0**
- `tonal_center`-Proxy auf Krumhansl-Schmuckler-Key-Detection umgestellt (SNR-invariant); alle früheren tonal_center-Exclusions für phase_02/03/04/08/18/29/49 entfernt

## v10.0.0 (30. März 2026) — PMGG phase_20/phase_23 Exclusions + phase_29 analog timbre

- phase_20: `{"authentizitaet", "natuerlichkeit"}` Exclusions (SGMSE+ Reverb-Reduction)
- phase_23/24: `{"natuerlichkeit", "brillanz", "authentizitaet", "artikulation", "timbre_authentizitaet"}` (AudioSR Inpainting: synthetisierter Inhalt)
- phase_29: `timbre_authentizitaet` Exclusion für analoge Materialien ergänzt

## v10.0.0 (30. März 2026) — PMGG phase_02 Exclusions erweitert

- phase_02: Exclusion-Set auf `{"bass_kraft", "authentizitaet", "natuerlichkeit", "transparenz", "groove", "timbre_authentizitaet"}` erweitert (Kammfilter Hum-Removal Root-Causes)

## v10.0.0 (30. März 2026) — Dual-SR-Vertrag + 48-kHz-Fail-fast-Härtung

- **instructions_version**: 3.7 → **3.8**
- **copilot-instructions.md**:
  - Performance-Budget um harte Dual-SR-Invarianten erweitert (`analysis_audio/analysis_sr` getrennt von `processing_audio/processing_sr=48000`).
  - Fail-fast-Vertrag ergänzt: Kein Weiterlauf auf Nicht-48k, wenn Normierung auf 48 kHz nicht möglich ist.
  - Resampling-Scope-Invariante ergänzt: Resampling darf nur den Verarbeitungspfad beeinflussen, Analysepfad bleibt native SR.
- **Spec 02** (`.github/specs/02_pipeline_architecture.md`):
  - Neuer Abschnitt **§2.2.0 Sample-Rate-Vertrag (Dual-SR, RELEASE_MUST)**.
  - Kanonischer Ablauf um expliziten `Dual-SR-Split` vor der Pipeline erweitert.
- **Spec 04** (`.github/specs/04_dsp_standards.md`):
  - Neuer Abschnitt **§4.1a Sample-Rate-Vertrag (Dual-SR, RELEASE_MUST)**.
  - Klargestellt: Komponenten mit abweichender Modell-SR dürfen intern resamplen, müssen aber wieder `processing_sr=48000` zurückgeben.
- **Code-Umsetzung**:
  - `backend/core/unified_restorer_v3.py`: native-SR-Routing für RestorabilityEstimator, Medium/Era/Genre-Classifier und DefectScanner; harter Abbruch bei fehlender 48-kHz-Normierung.
  - `cli/aurik_cli.py`: `_resample_to_48k()` wirft RuntimeError statt still auf Original-SR weiterzulaufen.

---

## v10.0.0 (29. März 2026) — Ganzheitliche Song-Selbstkalibrierung (psychoakustisch priorisiert)

- **instructions_version**: 3.6 → **3.7**
- **copilot-instructions.md**:
  - Neuer `[RELEASE_MUST]`-Abschnitt **§2.31a Ganzheitliche Song-Selbstkalibrierung**.
  - Pflichtprofil `song_calibration_profile` (inkl. `global_scalar` + `family_scalars`) als phasenübergreifende Vorgabe.
  - Klargestellt: bounded Skalare (anti-overfitting), deterministische Reproduzierbarkeit, psychoakustische Priorität (P1/P2, Maskierung, Transienten-Integrität).
- **Spec 02** (`.github/specs/02_pipeline_architecture.md`):
  - Pipeline um expliziten Schritt `SongCalibrationProfile` ergänzt (vor Klassifikations-/Phasenkette, mit Familien-Skalierung).
  - Phasen-Ausführung ergänzt um family-basierte strength/wet-dry-Skalierung mit psychoakustischer Priorisierung.
- **Spec 07** (`.github/specs/07_quality_and_tests.md`):
  - Neuer Abschnitt **§8.3.1 Song-Selbstkalibrierung** mit Qualitäts- und Testpflichten (`metadata["song_calibration"]`).

---

## v10.0.0 (29. März 2026) — Quality-First Standardpfad + MP-SENet Runtime-Vertrag

- **instructions_version**: 3.5 → **3.6**
- **copilot-instructions.md**:
  - RELEASE_MUST-Abschnitt **Quality-First Hauptlauf (v10.0.0)** als nutzerseitiger Standardpfad präzisiert: GUI/CLI/Batch müssen `AurikDenker.denke(..., no_rt_limit=True)` nutzen.
  - PerformanceGuard- und KMV-Kontrakt explizit als Schutz-/Telemetrieschicht bei gleichzeitiger Qualitätspriorisierung dokumentiert.
- **Spec 02** (`.github/specs/02_pipeline_architecture.md`):
  - Kanonischer Ablauf um Quality-First-Hauptlauf ergänzt (Stage-1 ohne RT-bedingtes Qualitätsopfer in Standardpfaden).
  - RT-limitierte Pfade als explizite Nicht-Standardpfade klar abgegrenzt.
- **Spec 04** (`.github/specs/04_dsp_standards.md`):
  - MP-SENet ONNX Runtime-Vertrag ergänzt: segmentierte Inferenz mit fixem Zeitfenster und robustem Layout-Handling; Fehlerfall führt zu DSP-Fallback statt Hard-Fail.
- **Spec 08** (`.github/specs/08_architecture_and_distribution.md`):
  - Architektur- und Entry-Point-Vertrag auf Quality-First-Standard (`no_rt_limit=True`) harmonisiert.
  - Plugin-Matrix-Hinweis zum MP-SENet-Runtime-Verhalten ergänzt.

---

## v10.0.0 (28. März 2026) — §3.9 Stabilitäts-Invarianten (Crash/OOM/Deadlock-Härtung)

- **instructions_version**: 3.4 → **3.5**
- **Tiefenanalyse**: Systematische Analyse aller Absturz-, OOM-, Deadlock- und Freeze-Szenarien im vollständigen Stack (UV3, ARM, PLM, ml_memory_budget, modern_window, BatchProcessingThread, MLRefinementThread)
- **Neue normative §§**:
  - `spec 08` **§3.9.1** Per-Phase-Inference-Timeout — `concurrent.futures.wait(timeout=300s)` für schwere ML-Inferenz; Timeout → DSP-Fallback + `deferred_phases`
  - `spec 08` **§3.9.2** SIGTERM-Handler in `main.py` — Emergency-Checkpoint bei gracefulem OS-Shutdown; SIGKILL-Limitation dokumentiert
  - `spec 08` **§3.9.3** Phase-Output-Guard — `@phase_output_guard`-Decorator-Kontrakt: `nan_to_num` + `clip` + `assert isfinite` strukturell erzwungen (keine reine Konvention mehr)
  - `spec 08` **§3.9.4** ThreadPoolExecutor-Lifecycle — `shutdown(wait=True, cancel_futures=True)` Pflicht in allen Cleanup-Pfaden; Kontext-Manager bevorzugt
  - `spec 08` **§3.9.5** ml_memory_budget Startup-Reconciliation — Budget-Reset auf 0 beim Prozessstart; verhindert stale Allokation nach SIGKILL
  - `spec 08` **§3.9.6** Structured Exception Logging — VERBOTEN: `except Exception: pass`; Pflicht: `fail_reasons`-Eintrag + `logger.error(..., exc_info=True)`
  - `spec 08` **§3.9.7** Audio-Buffer-RAM-Guard — `_check_audio_buffer_size(audio)` nach `soundfile.read()` vor Pipeline; `MAX_AUDIO_BYTES_RAM = 2 GB`
  - `spec 08` **§3.9.8** Lock-Acquisition-Order — Bindende Prioritätsreihenfolge: `MLMemoryBudget (P1) → PLM (P2) → ARM (P3)`; zirkuläres Locking verboten
  - `spec 08` **§3.9.9** MLRefinementThread Buffer-Release — `DeferredRefinementJob.release_buffer()` im `finally`-Block; `audio_original` nach Release auf `None`
  - `spec 02` **§2.42** Pipeline-Stabilitäts-Kontrakt — Referenztabelle aller 15 Stabilitäts-Invarianten (S-01 bis S-15); Checkliste für neue Module
- **Gate-Tabelle**: §3.9-Zeile ergänzt (`tests/normative/test_stability_invariants.py` — je Invariante ≥ 3 Tests)
- **copilot-instructions.md**: §3.9-Kurzbeschreibung in Präambel ergänzt

---

## v10.0.0 (28. März 2026) — PMGG Stable-Metric-Invariante + Tiefen-Immersions-Prinzip

- **instructions_version**: 3.3 → **3.4**
- **Root-Cause-Fix**: `NatuerlichkeitMetric` aus `_PRECISE_METRICS` entfernt — CREPE Load-State ändert Gewichte (w_crepe 0.0 → 0.18) zwischen `scores_before`/`scores_after` → Pseudo-Regression Δ≈0.15–0.28 auf unverändertem Audio → false P1-Kaskade → phase_03 best-effort @ 5.6 % Wet → Noise Floor −55 dBFS statt −72 dBFS → Tiefen-Immersion zerstört
- **Audio-Cap**: `_apply_precise_metric_overrides` kürzt auf max. **2.5 s** (NMF/Onset-Runtime-Schutz auf Langaudio)
- **`PHASE_GOAL_EXCLUSIONS`**: `phase_03`, `phase_02`, `phase_24` schließen `natuerlichkeit` aus
- **`_PRECISE_OVERRIDE_WARN_MS`**: 120 ms → **200 ms** (7 DSP-Only-Metriken ohne CREPE)
- **Neue normative §§**:
  - `copilot-instructions.md` **§2.29b** PMGG Stable-Metric-Invariante (7 Invarianten, `PHASE_GOAL_EXCLUSIONS`-Dokumentation, CREPE-Kausalkette)
  - `copilot-instructions.md` **Tiefen-Immersions-Prinzip** §8.3-Ergänzung (5-Schichten-Tabelle, Phase_03→Noise-Floor→Immersion-Kausalkette)
  - `spec 02` **§9.7.7** PMGG Stable-Metric-Invariante + **§9.7.8** Precise-Metric Audio-Cap
  - `spec 07` **§8.3.1** Tiefen-Immersions-Prinzip (Kausaldiagramm, Schichtenmodell)
  - `spec 08` §9.7 Code-Block: §9.7.5–§9.7.8 ergänzt
  - `spec 04` §4.2: `griffin_lim()` VERBOTEN als Phasengenerator-Endschritt (IPD-Kollaps, Raumtiefe)
- **Gate-Tabelle**: §2.29b-Zeile ergänzt (`tests/unit/test_per_phase_musical_goals_gate.py`)
- **Tests**: 35 Unit-Tests `test_per_phase_musical_goals_gate.py` — alle grün

## v10.0.0 (26. März 2026) — Mode-differenzierte Musical Goals + Priority-Aware PMGG

- **instructions_version**: 3.0 → **3.1** (§2.29 Priority-Aware Retries, mode-differenzierte Schwellwerte)
- **Spec 01 §1.2**: Schwellwert-Tabelle jetzt mit Spalten Restoration / Studio 2026 (Pareto-Differenzierung: P3–P5 gesenkt für Restoration)
- **Spec 01 §2.29** (neu): Priority-Aware PMGG Retry-Budget — P1/P2 volle Kaskade, P3 max 2, P4/P5 kein Retry
- **Spec 02 §2.29**: `_RETRY_STRENGTHS` 4 Stufen (Floor 0.25), `_PRIORITY_MAX_RETRIES`, `_PRIORITY_THRESHOLD_FACTOR`, Action `passed_p4p5_tolerated`
- **Code** (`musical_goals_metrics.py`): `MusicalGoalsChecker(mode=)`, `get_mode_thresholds()`, `_THRESHOLDS_RESTORATION`, `_THRESHOLDS_STUDIO_2026`
- **Code** (`per_phase_musical_goals_gate.py`): `_max_regression_priority_aware()`, Priority-Budget-Konstanten, Emergency nur P1/P2
- **Code** (`unified_restorer_v3.py`, `aurik_denker.py`): Mode-Parameter an MusicalGoalsChecker durchgereicht

## v10.0.0 (28. März 2026) — ML-Headroom-Guard + Structured Fallback-Normierung

- **instructions_version**: 3.1 → **3.2**
- **copilot-instructions.md**: neuer `[RELEASE_MUST]`-Abschnitt **§2.38a ML-Headroom-Guard + Structured Fallback**
  - Heavy-ML-Load nur nach RAM-Headroom-Check
  - Pflichtfelder fuer `metadata["ml_guard_events"]`
  - Guard-Trigger fuehrt zu DSP-Fallback innerhalb derselben Phase (kein Original-Rollback)
  - Guard-betroffene Phasen muessen in `deferred_phases` fuer KMV Stufe 2 eingetragen werden
- **Spec 02 §2.38a**: `RestorationResult`-Kontrakt fuer strukturierte ML-Guard-Events ergaenzt
- **Spec 07 §5.4**: normative Testfaelle fuer Low-RAM-Completion, Guard-Event-Contract, Deferred-Phase-Contract und KMV-Qualitaetsrueckgewinnung ergaenzt
- **Spec 08 §3.5a**: Architekturkontrakt fuer Headroom-Guard vor Modell-Load/Inferenz inkl. Cleanup-Reihenfolge (`evict_stale_plugins` + `gc.collect` + `malloc_trim`) ergaenzt

## v10.0.0 (28. März 2026) — Maximum-Qualitaets-Gates (Determinismus, Stratified Competition, Mini-MUSHRA)

- **instructions_version**: 3.2 → **3.3**
- **copilot-instructions.md**:
  - Gate-Tabelle erweitert um Vollpipeline-Determinismus, stratifiziertes Konkurrenz-Gate und externes Mini-MUSHRA-Artefakt
  - neue `[RELEASE_MUST]` Abschnitte §2.40 (Determinismus + Stratified Competition) und §8.4 (Mini-MUSHRA-Protokoll)
- **Spec 01**:
  - §2.35 Vocal-Exzellenz-Zusatzmetriken (Formant-Stabilitaet, Sibilance-Natuerlichkeit, Konsonanten-Klarheit)
  - §2.36 Pareto-Tie-Break nach Hoerprioritaet
- **Spec 02**:
  - §2.40 Vollpipeline-Determinismus-Vertrag (bitnahe Reproduktion)
  - §2.41 Structured Fail-Reason Taxonomie (`metadata["fail_reasons"]`)
- **Spec 07**:
  - §5.5 Determinismus-Gate
  - §5.6 Stratifiziertes Konkurrenz-Gate
  - §5.7 Externes Mini-MUSHRA-Artefakt als Release-Pflicht bei Kern-aenderungen

## v10.0.0 (26. März 2026) — OOM-Recovery-Checkpoint-System

- **Spec 02 §2.39** (neu): OOM-Recovery-Checkpoint-System — Checkpoint-Save + Startup-Resume
- **Neues Modul**: `backend/core/recovery_checkpoint.py`
- **Code**: UV3 MemoryError-Handler, `restore_from_checkpoint()`, Frontend Startup-Recovery

## v10.0.0 (25. März 2026) — Perceptual Salience + Denker-Differenzierung

- **instructions_version**: 2.7 → **2.8** (neue RELEASE_MUST-Regeln §9.1c, §11.7a)
- **Spec 07 §9.1c** (neu): Perceptual-Salience-Annotation — psychoakustische Maskierungsmodelle (Fastl & Zwicker 2007) für Defekt-Salienz
- **Spec 03 §11.7a** (neu): Denker-Rollendifferenzierung — disjunkte Verantwortungen für ReparaturDenker, RekonstruktionsDenker, RestaurierDenker
- **Neues Modul**: `backend/core/perceptual_salience.py` — `PerceptualSalienceEstimator` (Singleton)
  - Simultane Maskierung (12 dB Schwelle), temporale Vorwärts-Maskierung (200 ms, 8 dB), Rückwärts-Maskierung (20 ms, 6 dB)
  - Severity-Skalierung: `severity * (0.3 + 0.7 * mean_salience)`
- **Code**: `DefectScanner.scan()` — Integration nach §9.1b Intro-Boost, vor Location-Offset
- **Code**: `RekonstruktionsDenker.rekonstruiere()` — akzeptiert `defect_result`, extrahiert BANDWIDTH_LOSS, generiert ReconstructionContext-Felder
- **Code**: `RestaurierDenker.restauriere()` — akzeptiert `reconstruction_context`, reicht an UV3 weiter
- **Code**: `AurikDenker._run_rest()` — Kontextfluss: defect_result → RekonstruktionsDenker, rek → RestaurierDenker
- **Tests**: 35 neue Tests (`tests/unit/test_perceptual_salience.py`)

## v10.0.0 (24. März 2026) — Dropout-Erkennung: 3 neue Spec-Paragraphen + Code-Fixes

- **instructions_version**: 2.6 → **2.7** (3 neue RELEASE_MUST-Regeln)
- **Spec 05 §6.2a** (neu): Material-Prioritäts-Phasen MÜSSEN unbedingt aktiviert werden, unabhängig vom Severity-Score
- **Spec 05 §6.4a** (neu): Material-adaptive Erkennungsschwellen im DefectScanner (Analog 20 % vs. Digital 10 %)
- **Spec 07 §9.1a** (neu): Nicht-stationäre Defekttypen (DROPOUTS, TRANSPORT_BUMP) auf vollständigem Audio — kein 60 s Center-Crop
- **Code**: `DefectScanner._detect_dropouts()` — 5 ms Fenster (war 10 ms), material-adaptive Schwelle, duration-basierte Severity, NaN-Guard
- **Code**: `DefectScanner.scan()` — `_audio_mono_full` vor Center-Crop gesichert; `self.material_type` aus resolved type gesetzt
- **Code**: `UnifiedRestorerV3._select_phases()` — Phase 24 unbedingt für dropout-prone Materials (inkl. DAT); dedupliziertes `_DROPOUT_PRONE_MATERIALS` Set
- **Ursache**: Tape-Dropouts im Intro (Sec 0–5) waren unhörbar für die Pipeline: 60 s Center-Crop → Intro ausgeschlossen; 10 % statische Schwelle zu hoch für graduellen Tape-Pegelverlust; Phase 24 durch `sev > 0.10` Gate blockiert obwohl sie eigene Multi-Modal-Detektion besitzt

## v10.0.0d (21. März 2026) — Denker-Härtung: Pipeline-Zuverlässigkeit

- **Fix 1**: `AurikDenker._recommend_autopilot_mode()` in try/except gewrappt — verhindert Gesamtpipeline-Abbruch bei Autopilot-Fehler (Fallback: requested mode)
- **Fix 2**: 5 fehlende Tier-1-Severity-Checks in `_select_phases()` ergänzt:
  - `HEAD_WEAR` (>0.15) → phase_56 + phase_14 + phase_06
  - `AZIMUTH_ERROR` (>0.12) → phase_25 + phase_14 + phase_06
  - `TRANSIENT_SMEARING` (>0.15) → phase_08 + phase_36
  - `PRE_ECHO` (>0.15) → phase_23 + phase_50 + phase_08
  - `SIBILANCE` (>0.15) → phase_19 + phase_43
  - Bisher wurden diese 5 DefectTypes nur indirekt (CausalReasoner Tier 1.5 / PANNs) oder gar nicht in Phasen übersetzt
- **Fix 3**: `DefectScanner.scan()` None-Guard in UV3 — erzeugt Fallback-`DefectAnalysisResult` statt AttributeError-Crash
- **Fix 4**: NaN-Merge in paralleler Phase-Ausführung: `logger.warning()` statt stiller `pass` — Traceability bei NaN-Revert

## v10.0.0c (21. März 2026) — Spec-Konsistenz-Audit Mittel-Prio: instructions_version 2.3

- **instructions_version**: 2.2 → **2.3** (Bump für Mittel-Prio Spec-Korrekturen)
- **B-2**: `streaming`-Material in Spec 05 §6.2 ergänzt (MOS ≥ 4.1, Dropouts/Codec-Artefakte/Bitrate-Varianz)
- **B-3**: `MusikalischerGlobalplanDienst` (v10.0.0) in Spec 02 §2.2 Pipeline + Spec 03 §2.1 Kernmodule dokumentiert
- **B-5**: `BigVGAN-v2` Plugin in Spec 08 §11.3 Plugin-Policy ergänzt (0,4 GB, SEKUNDÄRER Vocoder)
- **D-5**: Vocoder-Kaskade in Spec 04 §4.5 explizit 4-stufig dokumentiert (Vocos → BigVGAN-v2 → HiFi-GAN → PGHI-ISTFT)
- **D-6**: Hardcodierte Testzähler in Specs 07/08 durch dynamische CI-Referenz ersetzt (`pytest --collect-only`)
- **D-4**: wow/flutter — bereits korrekt in Spec 05/06 dokumentiert, kein Fix nötig

---

## v10.0.0b (21. März 2026) — Spec-Konsistenz-Audit: instructions_version 2.2

- **instructions_version**: 2.1 → **2.2** (Bump für SIBILANCE-Mapping + Zahlen-Korrekturen)
- **B-1**: `SIBILANCE` CAUSE_TO_PHASES-Mapping in `causal_defect_reasoner.py` ergänzt
  (`phase_19_de_esser`, `phase_43_ml_deesser`, `phase_42_vocal_enhancement`)
- **A-1**: DefectScanner-Zählung: 29/30/27 → einheitlich **28** (Instructions + Specs 02/03/05)
- **A-3**: SGMSE+ Modellgröße: Spec 08 „120 MB ONNX" → „251 MB TorchScript" (Realwert)
- **A-4**: Vocos Primär-Modell: Spec 08 „24 kHz ONNX" → „48 kHz nativ (bevorzugt)"
- **A-5**: Testzahl: Specs 07/08 „6312" → „~7750+" (aktueller Stand v10.0.0)
- **A-6**: Materialien-Zählung: Spec 03/05 „17" → „15 + 2 Multichannel" (SUPPORTED_MATERIALS = 15)
- **A-7**: `DefectType` Docstring: „30 Defekttypen" → „28 Defekttypen"
- **A-8**: `utmos_plugin.py` Docstring: CDPAM-Fallback entfernt (VERBOTEN laut §4.4)
- **B-1b**: SIBILANCE + CAUSE_TO_PHASES in Spec 05 §6.3 + Spec 06 §7.2 ergänzt

---

## v10.0.0 (14. März 2026) — §SR-Invariante lückenlos

- `backend/core/genre_classifier.py`: `GermanSchlagerClassifier.classify()` — `assert sr == 48000`
- `backend/core/feedback_chain.py`: `FeedbackChain.run()` — `assert _sr == 48000`
- `backend/core/causal_defect_reasoner.py`: Default `44100` → `48000` korrigiert; bedingter Assert
- `backend/core/perceptual_embedder.py`: `PerceptualEmbedder.embed()` — `assert sample_rate == 48000`
- `backend/core/excellence_optimizer.py`: `assert sample_rate == 48000` als erste Zeile
- `backend/core/unified_restorer_v3.py`: Musical-Goals-Re-Pass-Zweig `logger.info` → `logger.warning`

---

## v10.0.0 (14. März 2026) — §Dach: MusikalischerGlobalplan

- `backend/core/musikalischer_globalplan.py` (neu): `MusikalischerGlobalplanDienst` (Singleton,
  Double-Checked Locking); 13 Ära-Profile (1890–2020); 7 Genre-Modifikatoren; 17 Per-Phase-Adjustments
- `backend/core/unified_restorer_v3.py`: `RestorationConfig.global_plan`-Feld; `_profiled_phase_call()`
- `denker/restaurier_denker.py`: `global_plan`-Parameter-Weitergabe
- `denker/aurik_denker.py`: Stufe 4 (DSP-only Globalplan); `AurikErgebnis.global_plan`-Feld
- 60 neue Tests `test_musikalischer_globalplan.py`

---

## v10.0.0 (12. März 2026) — §9.7 Performance-Optimierungen

- `backend/core/defect_scanner.py`: SHA256-Cache (`_scan_cache`, max. 128, FIFO, Thread-sicher)
- `plugins/panns_plugin.py`: SHA256-Cache (`_tags_cache`, max. 128, FIFO, Thread-sicher)
- `backend/core/unified_restorer_v3.py`: Parallele Eingangs-Analyse via `ThreadPoolExecutor(3)`
- `backend/core/per_phase_musical_goals_gate.py`: `PHASE_SAMPLE_DURATIONS` für 6 triviale Phasen
- `Aurik10/main.py`: Hintergrund-Warmup-Thread (daemon=True, 2 s Verzögerung)

---

## v10.0.0 (9. März 2026) — Infrastruktur: SBOM, GP-Backup, i18n-Tests

- `scripts/generate_sbom.py`: SPDX-SBOM-Generator mit SHA256-Modell-Verifikation
- `scripts/backup_gp_memory.py`: Backup/Restore für GP-Speicher (tar.gz)
- `scripts/verify_requirements.py` + `verify_requirements.sh`: pip dry-run CI-Check
- `tests/unit/test_export_roundtrip.py`: 20 Tests FLAC/WAV, Mono+Stereo
- `tests/unit/test_i18n.py`: 20 Tests DE↔EN, Thread-Sicherheit
- `tests/unit/test_gp_memory_migration.py`: 25 Tests v1→v2-Migration, MAX_OBSERVATIONS

---

## v10.0.0 (7. März 2026) — Spec-Konsistenz-Audit: 6 Korrekturen

- **S-1**: `EraResult.is_remaster_suspected: bool = False` in Spec ergsänzt (war seit v10.0.0 implementiert)
- **S-2**: `wrap_phase(restorability_score)` Default-Kommentar verschsärft (nur Testfallback)
- **S-3**: `MaterialQuality`-Enum + `MaterialQualityAssessment`-Dataclass vollständig in §2.31 definiert
- **S-4**: GP-Gedächtnis-Verzeichnis um Genre-Keys erweitert (schlager.json, jazz.json, etc.)
- **S-5**: Manifest-Beispiel: `"bs_roformer"` → `"mdx23c_kim_vocal_2"`
- **S-6**: README.md: Materialanzahl 17 → **15** (3 Stellen); quadrophony/ambisonic entfernt

---

- §2.1: `CausalDefectReasoner` → **27 DefectTypes → 14 Ursachen (historischer Zwischenstand)** (war: 24/11)
- §2.2: DefectScanner-Zeile auf **27 DefectTypes** aktualisiert
- §2.4: Ursachen-Liste um `riaa_curve_error`, `aliasing`, `bias_error` ergänzt
- §6.3: DefectType-Vollkatalog auf **27 Defekte** erweitert; neuer Abschnitt
  „Entzerrungs- & Digitalisierungsfehler" mit `RIAA_CURVE_ERROR`, `ALIASING`, `BIAS_ERROR`

---

## v10.0.0b (März 2026) — §2.36 Lyrics-Guided Enhancement (v10.0-Spec)

- `LyricsTranscriber`: Whisper-Tiny ONNX lokal (39 MB), CPUExecutionProvider, stiller DSP-Fallback
- `ContentAwareProcessor`: Phonem-Typ × Betonung → Salienz-Boost 0.5–2.0,
  G_floor 0.90 an fricative+stressed-Bins, PAM-Integration §2.22
- `LyricsGuidedTimeline`: WaveformWidget-Farboverlay, Shortcut `L`,
  Datenschutz: kein Lyrics-Text geloggt
- Pipeline-Position (v10.0): LyricsTranscriber → ContentAwareProcessor → PAM.apply_to_gain()
- Manifest-Eintrag `whisper_tiny` (bundled:true, 39 MB, Fallback: energy_segmentation_dsp)
- §12: Radford et al. (2022) Whisper aufgenommen
- Roadmap: Tier 2+3 abgeschlossen, Tier 4 als v10.0-Ziel

---

## v10.0.0 (März 2026) — Spec-Konsistenz-Audit (14 Lücken)

- §2.2: RestorationResult JSON-Serialisierungsschema ergänzt (audio nicht in JSON,
  NaN/Inf → null, genealogy als Sidecar)
- §2.20: `JAZZ_RESTORATION_PROFILE`, `KLASSIK_RESTORATION_PROFILE`,
  `OPER_RESTORATION_PROFILE`, `ROCK_RESTORATION_PROFILE` mit allen Parametern
- §4.1: DDSP → NumPy/SciPy-Eigenimplementierung `dsp/ddsp_synth.py` (kein TensorFlow)
- §11.4: A/B-Vergleich (A/B-Shortcuts), vollständige Keyboard-Shortcut-Tabelle,
  Preset-Browser & Queue-Widget spezifiziert
- §13.3: Out-of-the-Box-Garantie verschärft — 100 % offline, SOTA-Upgrades lokal gebündelt,
  `sota_upgrade`-Feld nur Entwickler-Metadaten (kein Laufzeit-Download)
- §13.5: Setup-Wizard: SOTA-Upgrade-Checkbox entfernt
- §13.8 (neu): Manuelles Update-Verfahren dokumentiert

---

## v10.0.0 (Feb 2026) — RemasterDetector + temporale Defektverortung

- `core/remaster_detector.py` (neu): `RemasterDetector`-Singleton; analysiert Rauschboden
  (< −80 dBFS → `_floor_score`) und HF-Rolloff (> 18 kHz → `_bw_score`);
  `confidence = 0.55·floor_score + 0.45·bw_score`; `is_remaster=True` wenn ≥ 0.35
- `plugins/era_classifier_plugin.py`: `EraResult.is_remaster_suspected`-Feld ergänzt
- `core/defect_scanner.py`: `_detect_print_through()` → `locations` mit 20-ms-Dedup,
  50-Einträge-Cap und Zeitstempel
- Tests: `test_remaster_detector.py` (18 Tests), `test_defect_scanner_temporal.py` (17 Tests)

---

## v10.0.0 (Feb 2026) — SGMSE+ entfernt, WPE als kanonisches Dereverb-Plugin

- `plugins/wpe_plugin.py` (neu): `WpePlugin`, 3-Tier-WPE (nara_wpe → NumPy-WPE → OMLSA),
  kein Checkpoint, kein Großmodell-Speicher
- `plugins/sgmse_plugin.py`: Thin-Shim → `wpe_plugin` (Backward-Compat)
- `models/sgmse_plus/` gelöscht
- §4.4: SGMSE+ → WPE (Nakatani 2010), 3-Tier-Kaskade
- §9.5: SGMSE+-Eintrag → WPE-DSP-Hinweis (kein RAM-Budget)
- §11.3: sgmse_plugin → wpe_plugin (kanonisch) + sgmse_plugin (Shim)

---

## v10.0.0 (Feb 2026) — SCHRITTE_ZUR_MUSIKALISCHEN_EXZELLENZ abgeschlossen

Testzahl: 6394 → **6312** (nach v2-Cleanup).

- **K-1**: TIER-1/TIER-6-Assertions in `_validate_restoration_result()`
- **K-2**: `quality_estimate`-Formel: `0.40·(1−sev) + 0.60·(pqs_mos−1)/4`; `× 1.15`-Bonus entfernt
- **M-1**: `_SEQUENTIAL_TIER_PHASES`-Frozenset; TIER-0/TIER-1 immer sequenziell (§2.2.1)
- **M-2**: `self._warnings`-Liste; sicherer `_get_phase()`-except-Block
- **M-3**: `_era_for_stereo`-Fallback auf `SimpleNamespace(decade=1960)`
- **I-3**: `scores["artikulation"] = scores["articulation"]`-Alias
- **W-4a**: `VocalAIEnhancement = UnifiedVocalAIEnhancer`-Alias
- **W-4b**: 11 Vokalketteninvarianten-Tests (`test_vocal_chain_invariants.py`)
- **V-5**: CI-Stub-Guard `tests/normative/test_no_production_stubs.py`
- v2-Cleanup: `unified_restorer_v2.py`, `context_aware_deesser_v2.py`, 17 v2-Tests entfernt

---

## v10.0.0 (Feb 2026) — 4 neue Qualitätsmechanismen

### §2.27 TransientDecoupledProcessing (TDP)

- HPSS-Trennung (Medianfilter-Kernel 31) am allerersten Pipeline-Schritt
- `audio_percussive` → NUR phase_01/phase_27; `audio_harmonic` → volle Pipeline
- OLA-Crossfade Hanning 10 ms; DTW-Sicherheitsnetz (> 8 ms → Original-Percussive)
- Effekt: GrooveMetric +0.03–0.06, Timbre-Authentizität +0.02–0.04
- Referenz: Fitzgerald (2010)

### §2.28 HarmonicPreservationGuard (HPG)

- CREPE (full, Fallback pYIN) → f₀(t), Voicing ≥ 0.6
- Harmonisches Gitter fₙ = n·f₀·√(1+B·n²), Fletcher-B aus INHARMONICITY_PRIORS
- G_floor-Override: 0.85 an protected_bins, 0.10 sonst
- Energie-Korrektur nach NR: gain ∈ [1.0, 2.0] + PGHI
- Effekt: Natürlichkeit +0.03–0.07, Authentizität +0.03–0.06

### §2.29 PerPhaseMusicalGoalsGate (PMGG)

- 5-s-Stichprobe nach jeder Phase → measure_quick() auf 6 Schnell-Ziele (≤ 200 ms)
- Δ < −REGRESSION_THRESHOLD → Retry-1 (×0.65) → … → Retry-5 (×0.10) → Rollback
- Adaptiver Schwellwert: 0.012 (restorability ≥ 70) / 0.040 / 0.060 (< 40)
- Max. Retries: 5 (v10.0.0-B3)

### §2.30 MicroDynamicsEnvelopeMorphing (MDEM)

- 400-ms-LUFS-Profile; G[k] ±3.0 LU; Savitzky-Golay-Glättung; lineare Interpolation
- Stille-Segmente (< −60 LUFS) → G[k] = 0; True-Peak nach Morphing
- Effekt: MicroDynamicsMetric Pearson 0.88 → 0.93–0.96

---

## v10.0.0 (Feb 2026) — Spec-Konsistenz-Audit (7 Inkonsistenzen)

- §2.1: MusicalGoalsChecker → **14** Ziele; CausalDefectReasoner → **23** DefectTypes
- §2.2: 9 neue Module ins Pipeline-Diagramm: RestorabilityEstimator, EraClassifier,
  GermanSchlagerClassifier, UncertaintyQuantifier, IAD, TemporalQualityCoherenceMetric,
  EmotionalArcPreservationMetric; RestorationResult-Felder vervollständigt
- §7.1: `phase_56_spectral_band_gap_repair.py` ergänzt
- §7.2: `compression_artifacts` und `head_wear` in CAUSE_TO_PHASES ergänzt
- §9.1: Checkliste „Musical Goals (alle 8)" → **14**
- §11.3: `flow_matching_plugin.py`, `era_classifier_plugin.py`, `core/genre_classifier.py` ergänzt

---

## v10.0.0 (Feb 2026) — 11 Architektur-Lücken geschlossen

- §1.4 **StemRemixBalancer**: g_voc/g_inst Gain-Korrektur; |LUFS(mix) − L_orig| ≤ 0.3 LU
- §2.21 **EnsembleProcessor**: 3 Ketten (×0.6/×1.0/×1.4), frame-by-frame Goals-Voting
- §2.22 **PerceptualAttentionModel**: PANNs+MERT Salienz-Karte [n_frames × 24] ∈ [0.3, 2.0]
- §2.23 **IntroducedArtifactDetector**: ML_HALLUCINATION / NMF_RESIDUAL_CLICK / SMEARING / MUSICAL_NOISE
- §2.24 **BatchSessionLearner**: GP-Warm-Start zwischen Dateien (SHA256-Session-ID)
- §2.25 **ReferenceAnchorSynthesizer**: 270 Anker (10 Dek × 9 Genres × 3 Mat), k=3 k-NN
- §2.26 **RestorabilityEstimator**: < 5-s-Assessment, Score 0–100, predicted MOS
- §4.5 **SpectralBandGapRepair**: HEAD_WEAR, 56. Phase; Harmonische Interpolation + NMF-β + PGHI
- §6.1: `quadrophony`/`ambisonic` formal aus SUPPORTED_MATERIALS entfernt (Nur Mono/Stereo)
- §8.2: **EmotionalArcPreservationMetric** (Arousal/Valence Pearson, Klimax-Peak-Abweichung)
- §9.6: **Progressive-Quality-Mode**: Stage-1 (5-s, ≤ 8 s) + Stage-2 (volle Pipeline)

---

## v10.0.0 (Feb 2026) — Zero-Shot-Genre-Klassifikation & Schlager-Erkennung

### §2.19 GermanSchlagerClassifier (6-Schicht-Ensemble)

- Tier-1: LAION-CLAP Zero-Shot (7 positive + 5 negative Prompts, 30 % Gewicht)
- Tier-2: Akkordeon Reed-Beating DSP (Hilbert, [5–15] Hz + [4–8] Hz)
- Tier-3: Harmonischer Simplizitäts-Index (CQT-Chroma, Quintenkreis ≤ 2)
- Tier-4: Rhythmus-Muster (madmom → Schunkel/Walzer/Marsch/Disco)
- Tier-5: Deutsch-Vokal-Formant-Prior (LPC-Burg, SAMPA-Polygone, ±0.08 max.)
- Tier-6: Melodische Wiederholungsrate (MFCC-SSM, Schwelle 0.85, ≥ 8 s Abstand)
- Voting: ≥ 3 von 5 DSP-Schichten + Konfidenz ≥ 0.52 → is_schlager=True
- `SCHLAGER_RESTORATION_PROFILE`: TonalCenter 0.97, Wärme 0.88, Brillanz 0.82

### §2.20 Genre-Klassifikations-Matrix

- 9 Genres mit Erkennungsmethode und Pipeline-Anpassung dokumentiert

---

## v10.0.0 (Feb 2026) — Weltführungsanspruch (14 Spec-Lücken)

- Musical Goals 9 → **14**: TonalCenterMetric (Chroma ≥ 0.95) + MicroDynamicsMetric (LUFS ≥ 0.92)
- §2.14 EraClassifier: 1890–2025, CLAP-Tier-1 + DSP-Tier-2 + Mikrofon-Tier-3
- §2.15 Uncertainty Quantification: Konfidenz-Schwellen 0.80/0.50/0.00
- §2.16 TemporalQualityCoherenceMetric: MOS-Spanne ≤ 0.30, σ ≤ 0.15
- §2.17 MusicalStructureAnalyzer: SSM + Novelty-Kurve (Foote 2000), Chorus-Prior
- §2.18 StereoAuthenticityInvariant: Mono ≥ 0.97, Decca ∈ [0.25, 0.65], Abbey Road ≤ ±3°
- §4.4: Flow Matching, EnCodec/DAC, EraClassifier, MusicalStructureAnalyzer ergänzt
- §6.1/6.2: `wax_cylinder`, `wire_recording`, `lacquer_disc` hinzugefügt (MOS ≥ 3.5/3.6/3.7)
- §8.1: Rauschboden ≤ −72 dBFS / −75 dB(A), HF-Limit ≤ +4 dB kumulativ
- §8.2: EmotionalArcPreservationMetric spezifiziert; Kompetitiver Benchmark (≥ iZotope RX 11)

---

## v10.0.0 (Feb 2026) — SHA256-Größenabgleich (15 Modelle)

- MDX23C: 2× 64 MB; Apollo: 65 MB; CREPE: 85 MB; DeepFilterNet: 37 MB
- Vocos: 52 MB; Banquet Vinyl: 92 MB; Resemble-Enhance: 41 MB
- UVR HQ: 56–64 MB; HTDemucs 6s: 2,5 MB

---

## v10.0.0 (Feb 2026) — Manifest-Verifizierung & Offline-Garantie

- Apollo SHA256 `440c48b1…` / 67,7 MB als primäres Manifest-Modell korrigiert
- SGMSE+, MERT (3,9 GB), AudioSR (5,9 GB) als Lazy-Load dokumentiert
- §11.3: ✅-Markierungen für lokal gebündelte vs. SOTA-Upgrade-Modelle

---

## v10.0.0 (Feb 2026) — Manifest-Abgleich

- §4.4: Apollo primär (bundled) / Resemble-Enhance Fallback (nicht umgekehrt)
- §11.3: 18 Manifest-Modelle und Plugin-Dateien vollständig dokumentiert

---

## v10.0.0 (Feb 2026) — SOTA-Audit & Musik-Ausrichtung

- Musical Goal 9: **SpatialDepthMetric** (≥ 0.75)
- OQS-Evaluator (algorithmische PEAQ-Approximation, kein BS.1534-3-Hörertest)
- AMRB v1.0 (10 Szenarien, interne Führungs-Schwelle ≥ 84.0)
- §4.4: BS-RoFormer, CQTdiff+, Apollo, LAION-CLAP, UTMOS/VERSA als neue Primär-Algorithmen
- BigVGAN-v2 (NVIDIA 2024, Apache-2.0) als primärer Vocoder-Endschritt (MOS < 4.3)
- PESQ/DNSMOS/NISQA/STOI/POLQA explizit verboten (§4.4, §10.2, §11.3)
- CDPAM als primäre Musik-Wahrnehmungsmetrik; ViSQOL v3 --audio Mode erzwungen

---

## v10.0.0 (Feb 2026) — Über-SOTA-DSP-Erweiterungen

- Musical Goal 8: **GrooveMetric** (DTW ≤ 8 ms RMS)
- **Multi-Resolution STFT MRSA**: 128–65536 Samples pro Frequenzzone + PGHI
- **Psychoakustisches Masking-Modell** (ISO 11172-3, OMLSA-Gain-Modifier)
- §2.11 **Harmonic Lattice Coherence** (Fletcher-Modell, B-Koeff., ±3 Cent)
- §2.12 **Musikalische Phrasenkontextfenster** (madmom Beat-Tracking, ≤ 30 s Kontext)
- §2.13 **Künstler-Signaturmodell** (Formant/Vibrato/Breathiness, artist_signatures/)
- `SOFT_SATURATION` als 22. DefectType (Tube-Sättigung BEWAHREN)
- **Noise-Shaped Dithering POW-r Typ 3** beim 24→16 bit Export

---

## v10.0.0 (Feb 2026) — Architektur-Fundament

- Thread-safe Singletons (Double-Checked Locking, §3.2)
- PEP 484 Type-Annotation-Pflicht + mypy strict (§3.7)
- SHA256-Ergebnis-Cache für teure Operationen (§3.8)
- MOO-GP-Optimizer (Pareto-Front, 14 Objectives)
- **SegmentAdaptiveProcessor** (Content-Aware, §2.10)
- Consonant Enhancement in Vocal-Pipeline
- Print-Through: Adaptive Temporal Subtraction + CAUSE_TO_PHASES
- PEAQ/FAD als optionale Parallel-Metriken
- Reference Mastering (Optimal Transport)
- Defektdichte-adaptive Chunk-Größe (5/15/60/120 s)
- Restaurierungs-Genealogie / Sample-Audit-Trail

---

## v10.0.0 (Basis)

- Initiale Systemspezifikation: 14 Musical Goals, 27 DefectTypes
- Pipeline-Grundstruktur: TDP → RestorabilityEstimator → EraClassifier → … → MDEM
- Kernmodule: PerceptualEmbedder, CausalDefectReasoner, GPParameterOptimizer,
  PerceptualQualityScorer, MediumClassifier, DefectScanner, UnifiedRestorerV3
- Out-of-the-Box-Pflicht (bindend): AppImage / NSIS-Installer, alle Modelle gebündelt
- Restaurierungs-Modi: Restoration + Studio 2026
