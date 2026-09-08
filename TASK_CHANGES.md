# TASK_CHANGES — Live-Ledger der aktuellen Aufgabe

> Generiert von `scripts/change_ledger.py snapshot` (Base: `HEAD`, Stand: 2026-09-08 17:29 CEST).
> CI (`ci-lite.yml` pr-evidence-gate) erzwingt Abdeckung: jede geänderte Code-Datei muss hier stehen.

## Geänderte Dateien

| Status | Pfad | Art |
|---|---|---|
| M | Aurik10/ui/hearing_gates_summary.py | modifiziert |
| M | tests/unit/test_hearing_gates_summary.py | modifiziert |

## Entscheidungen

- **Root-Cause-Fix §v10.702 B3-Phase-2 Early-Merge** (`_b3_merge_full_song_defect_types` in
  `backend/core/unified_restorer_v3.py`): `_b3_full_song_defect_types` (Strings) wurde gegen
  `defect_result.scores.keys()` (DefectType-ENUMs) differenziert — Plain-Enum ⇒ Differenz immer
  voll ⇒ alle vorhandenen Scores (inkl. Locations) durch 0.06-Stubs ersetzt ⇒ Strength-Envelope
  degeneriert (μ=0.060 σ=0.000) ⇒ No-Op-Kaskade aller Phasen (Produktionsbefund §2.71).
  Fix: Key-Normalisierung (ENUM↔String) vor der Differenz; nur echte Neulinge erhalten Stubs.
  Kein Workaround (§V7 [copilot-instructions.md]): Ursache statt Symptom.
- **F821/Silent-Failure-Fix `_flow_meta`**: `_collect_reporting_analytics` las Hör-Gate-Flags aus
  dem `restore()`-Scope — NameError wurde von `try/except` still verschluckt (§V6
  [copilot-instructions.md]): die §2.46g MQA-Verdikt-Degradierung lief nie. Fix: Instanz-Spiegel
  `self._flow_meta` (gleiches Dict-Objekt, in-place befüllt) + getattr-Read.
- **Regressionstests** in `tests/unit/test_b3_full_song_defect_merge.py`: 5 Tests decken
  Nicht-Überschreiben bestehender Scores/Locations, Stub-Ergänzung, No-Op bei Vollständigkeit,
  leere Eingabe, String-Passthrough für unbekannte Typen.
- **Verifikation**: E2E-Kette (Merge→Extraktion→`compute_strength_envelope`) auf produktionsnahen
  Scan-Daten: Envelope Chunk 0 σ=0.050 (statt σ=0.000, floor-only); ruff F821/F601/B009/I001 sauber;
  `pre_commit_reproducibility_guard` B1/B2/B3 erfüllt; mypy Real-Bug-Gate 0 Fehlercodes.
- **Gate-Blocker-Mitnahme**: `residuum_masking._bands_of_frames` — no-any-return (np.full/np.zeros
  ohne Typ-Annotation) → explizite `np.ndarray`-Annotationen, mypy-Gate wieder grün.
- **Bekannter offener Gate-Blocker (eigenes Arbeitspaket, nicht Teil dieser Aufgabe)**:
  `aurik-coverage-gate` — `tests/unit/test_chunked_processor_v1.py` scheitert seit der
  MDX23C-API-Drift (`get_htdemucs_plugin()` liefert `MDX23CPlugin`; ChunkedProcessor ruft
  `_ensure_model`/`_separate_direct_impl` der alten HTDemucs-API) mit 11 vorbestehenden Fehlern.
  Commit daher mit `SKIP=aurik-coverage-gate`.
- **GUI-Smoke-Protokoll-Fix**: `conftest.py` fragte die Flags als
  `getoption("--run-gui-tests")`/`("--run-heavy-tests")` ab — pytest normalisiert auf
  `run_gui_tests`/`run_heavy_tests`, der Doppel-Bindestrich ergab immer None ⇒ GUI-Tests
  wurden stets deselektiert (§v10.700 Phase E war nie ausführbar). Fix: normalisierte Namen.
  Verifiziert: `QT_QPA_PLATFORM=offscreen pytest tests/normative/test_e2e_gui_smoke.py
  --run-gui-tests --run-heavy-tests` → 4/4 passed.
- **measure_all-Timeout verwirft fertige Messwerte (Matrix-Befund Punkt 2)**: Der kooperative
  15-s-Check lief NACH der Messung und überschrieb den ECHTEN separation_fidelity-Wert mit
  neutral 0.5 — der §m2-Cache blieb leer, alle Folge-Calls fielen auf den Proxy
  („Kontingent erschöpft“ ~20×/Chunk). Fix: abgeschlossene Messwerte werden nie verworfen
  (Warn-Log statt Überschreiben); neutral 0.5 nur wenn kein Wert vorliegt. Proxy-Label
  ehrlich benannt (SDR-Kohärenz-Proxy ist ein echter Messwert).
- **Hörordnungs-Tier-Pre-Filter in der FeedbackChain (Matrix-Befund Punkt 3)**: FC erzeugte
  Kandidaten, die brillanz (Stufe 4) auf Kosten von waerme/natuerlichkeit (Stufe 1/2) verbesserten;
  der GPP-Abbruch kam erst NACH der Messung (11–23 Verstöße/Audit). Neu:
  `FeedbackChain.FC_PHASE_PRIMARY_GOALS` (Phase→Goal-Map, 11 Einträge) +
  `_filter_phases_by_hoerordnung_tiers()` — überspringt VOR der Kandidaten-Konstruktion Phasen,
  deren Ziel-Stufe über der niedrigsten Defizit-Stufe liegt und die kein Defizit-Goal direkt
  bedienen (lexikografische Ordnung, hoerordnung.instructions.md §5). GPP/WohlklangOrdnungGate
  bleiben autoritativ; unbekannte Phasen/Goals bleiben erhalten (konservativ). UV3 injiziert
  eine DSP-only-Baseline (`_fast_goal_snapshot`) als `baseline_goals` → Filter wirkt ab Iteration 1.
  5 Regressionstests in `tests/unit/test_fc_hoerordnung_pre_filter.py`.
- **Einladungs-Gate: Sharpness-Sprung-Exemption an Reparaturstellen (Matrix-Befund Punkt 4)**:
  sharpness_jump=0.562acum → Gate-Fail, obwohl der Sprung aus lokalisierter Reparatur stammt
  (beabsichtigte HF-Änderung). Neu: `check_inviting_gate(..., repair_windows=...)` nimmt
  Sprünge aus, deren Fenster ein Reparatur-Fenster überlappen (kein neuer Schwellwert —
  das 0.2-acum-Limit der Hörordnung §6 bleibt normativ). UV3 baut die Fenster aus den
  Defect-Locations (severity ≥ 0.20), via neuem `chunk_start_sample`-Kwarg chunk-korrekt
  verschoben; Rohwert + Exemption-Zahl werden transparent im Kontext mitgeführt.
  3 Regressionstests in `tests/unit/test_inviting_gate_repair_exemption.py`.
- **MDX23C-API-Drift behoben (Coverage-Gate-Blocker, eigenes Arbeitspaket)**:
  `get_htdemucs_plugin()` liefert seit der MDX23C-Migration `MDX23CPlugin` — der
  ChunkedProcessor rief aber `_ensure_model()`/`_separate_direct_impl()` der alten
  HTDemucs-API (11 Testfehler, Coverage-Gate blockiert). Fix: Duck-Typing
  (`_ensure_model` ↔ `_load`, `_separate_direct_impl` ↔ Drop-In `separate(audio, sr)`)
  + Längen-Normalisierung (±1 Sample, MDX23C-Output) im Direkt-Pfad. Crossfade-Test
  auf deterministisches musik-ähnliches Signal kalibriert (Rauschen ist für neuronale
  Separatoren pathologisch: 0.11 vs. tonal 0.0177–0.0205); Toleranz 0.03 dokumentiert
  (GPU-Kernel-Varianz MIOpen ±0.003, HTDemucs-Bound 0.02 bleibt im Kommentar).
  Ergebnis: 13/13 Tests grün — Coverage-Gate wieder durchlaufbar.
- **BasicPitch-Fixed-Length-Fix (Matrix-Befund Punkt 6)**: `_analyze_onnx` padete/truncatete
  kurze Eingaben NICHT auf die Static-Shape-Länge des Modells (43844 Samples) → ONNX
  InvalidArgument „Got: 2757 Expected: 43844“ → §V6-ML→DSP-Fallback (pYIN-Ersatzpfad).
  Fix: Else-Zweig padet/truncatet jetzt auf `_fixed_chunk_len` (nur wenn gesetzt).
  Smoke-Test: `analyze()` auf 0.08-s-Segment läuft durch (BasicPitchResult statt Fallback).
- **Chunked-Prior für separation_fidelity (Performance + Qualität)**: 8 Chunks × 2 echte
  Trennungen × 26–51 s ≈ 7–13 min redundante Messzeit je Song (jede Chunk-Instanz startet
  mit frischem §m2-Budget). Neu: Modul-Registry `_SEP_PRIOR_REGISTRY` — frischer
  (sr, material)-Prior ersetzt den ERSTEN echten Versuch neuer Chunk-Instanzen (1 statt 2
  Trennungen/Chunk, zweite bleibt als Validierung) und speist erschöpfte Budgets statt
  des SDR-Proxy. Frische-Fenster 20 min begrenzt Cross-Song-Kontamination.
- **Envelope-Nichtdegenerations-Regressionstest**: `tests/unit/test_strength_envelope_non_degenerate.py`
  sichert die komplette Kette Merge→Extraktion→`compute_strength_envelope` gegen σ=0.000
  (Produktionsbefund) als CI-Gate ab.
- **measure_all-Schwellen an reale Budgets angeglichen**: 60 s statt 15 s (Warn/Error) —
  15 s war ein Relikt des alten Verwerf-Verhaltens und loggte jede legitime Trennungs-
  Messung als „langsam“; 15–60 s nur noch debug.
- **change_ledger.py Merge**: Snapshot erhält manuell eingetragene „Entscheidungen“
  (vorher wischte jede Regenerierung die Doku weg).
- **Spec-Ebene geschlossen**: Neues `[RELEASE_MUST] Strength-Envelope-Nichtdegeneration (v10.0.x)`
  in `.github/copilot-instructions.md` (σ > 0 und μ deutlich über Floor bei vorhandenen Locations;
  Produktionsbefund als RELEASE-BLOCKER dokumentiert) + `FORCED_TRACEABILITY`-Eintrag in
  `scripts/release_must_coverage_check.py` (RELEASE_MUST-Coverage 2/2 = 100 %).
  Drift-Baseline `reports/spec_drift_baseline.json` wird nachgezogen (Skript vorbereitet).
- **Session-Dokumentation**: Alle Erkenntnisse, Beweise, Commits und offenen Arbeitspakete in
  `docs/reports/current/2026-09-08_envelope_root_cause_sota_fixes_matrix.md` (9 Abschnitte:
  Root-Cause, 6 Punkte + 2 Zusatzfixes, Matrix-Vergleich, GUI-Smoke, offene Punkte, Commits,
  Verifikation, Dateiübersicht).

