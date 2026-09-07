# TASK_CHANGES — Live-Ledger der aktuellen Aufgabe

> Generiert von `scripts/change_ledger.py snapshot` (Base: `HEAD`, Stand: 2026-09-07 18:33 CEST).
> CI (`ci-lite.yml` pr-evidence-gate) erzwingt Abdeckung: jede geänderte Code-Datei muss hier stehen.

## Geänderte Dateien

| Status | Pfad | Art |
|---|---|---|
| M | .gitignore | modifiziert |
| M | TASK_CHANGES.md | modifiziert |
| M | backend/core/unified_restorer_v3.py | modifiziert |
| M | backend/core/residuum_masking.py | modifiziert |
| M | conftest.py | modifiziert |
| ?? | tests/unit/test_b3_full_song_defect_merge.py | ungetrackt |

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

