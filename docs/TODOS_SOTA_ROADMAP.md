# TODOS — SOTA-Roadmap & Lücken-Schluss (nächste Sessions)

> **Stand:** 2026-09-08 · Quelle: Matrix-Endlauf (3 Zellen, Elke-Best-Vinyl) + unabhängige
> Spec-vs-SOTA-Tiefenanalyse (`docs/reports/current/2026-09-08_envelope_root_cause_sota_fixes_matrix.md`).
> **Wie erkannt wird:** Jede Aufgabe hat eine TODO-ID (`TODO-P0-1` …), Ziel, Wirkung,
> Beleg (Pfad:Zeile) und Akzeptanzkriterium. Agenten: Aufgabe mit ID greppen, Beleg lesen, umsetzen,
> Akzeptanzkriterium als Test/Gate nachziehen, `change_ledger.py snapshot` + Commit.
> **Reihenfolge:** P0-1 → P0-2 → P0-3 → P1-1 → P1-2 → P1-3 → P1-4 → P2-1.

---

## TODO-P0-1 · Analytik + End-Gate von per-Chunk auf Song-Ebene heben (größter Hebel)

- **Ziel:** Chunks restau­rieren → assemblieren → **einmal** song-weit validieren (GOAL_SCORECARD,
  End-Gate-Wiederherstellung, HPI, EmotionalArc, VQI, Einladungs-Gate). Nur lokal-stationäre
  DSP/ML-Phasen bleiben je Chunk.
- **Wirkung:** Performance ~53× → 15–25× RT (8–9 End-Gate-Runden × measure_all je Chunk entfallen);
  **Qualität steigt**, weil HPI/Sänger-Identität/EmotionalArc/VQI per Spec song-globale Größen sind
  und heute auf 30-s-Ausschnitten semantisch verfälscht laufen.
- **Beleg:** `backend/core/unified_restorer_v3.py` `_restore_chunked` (GOAL_SCORECARD/Recovery je
  `restore()`-Aufruf); `pipeline.instructions.md` §2.45b (EmotionalArc), §2.44 (HPI als Export-Gate);
  Session-Befund: 8–9 Runden je Chunk, 3 h 19–38 min/Lauf.
- **Extraktionsgrenzen (aus Session-Analyse 2026-09-08, restore()-Tail in unified_restorer_v3.py):**
  Song-globale Blöcke, die nach der Chunk-Assembly EINMAL laufen sollen:
  (a) GOAL_SCORECARD + End-Gate-Recovery-Kaskade (~Z. 16600–17200),
  (b) Einladungs-Gate (~Z. 17972), (c) MQA/_collect_reporting_analytics (~Z. 18667),
  (d) Audibility-Gate + m1b-Queue (~Z. 23260–23340). Vorschlag: neuer
  `_run_song_level_tail(assembled_audio, …)`-Aufruf am Ende von `_restore_chunked`;
  per-Chunk-restore() erhält einen Flag, der diese Blöcke überspringt. Chunk-lokal
  bleiben: Strength-Envelope, Phasen-Loop, FC-Iterationen, PMGG.
- **Akzeptanz:** 224-s-Referenzlauf ≤ 40 min Gesamtlaufzeit; 3-Zellen-Output bleibt bit-identisch
  (Determinismus §G5); alle song-globalen Gates laufen nachweislich auf dem assemblierten Song.

## TODO-P0-2 · Tier-Map-Synchronisation + Pre-Commit-Gate

- **Ziel:** `PRIORITY_MAP` (brillanz=5, spatial_depth=5) und `HEARING_TIER_MAP` (brillanz=4,
  spatial_depth=4) in `goal_priority_protocol.py` vereinheitlichen; Goal-Namens-Aliase
  (`timbre`/`timbre_authentizitaet`, `raumtiefe`/`spatial_depth`, `sep_fidelity`/`separation_fidelity`)
  kanonisieren; deterministischen Sync-Test (analog `test_pmgg_cig_sync.py`) als Pre-Commit-Gate.
- **Wirkung:** Verhindert divergente Gate-Entscheidungen (WohlklangOrdnungGate nutzt `hearing_tier()`,
  FeedbackChain-Abort `priority_of()` — zwei Gewinner bei identischem Konflikt).
- **Beleg:** `backend/core/goal_priority_protocol.py:39–84`; `backend/core/wohlklang_ordnung_gate.py`;
  Tiefenanalyse Abschnitt A.2.
- **Akzeptanz:** Neuer Test schlägt bei jeder Divergenz fehl; CI grün; keine stillen Default-Tier-3-Fälle
  für bekannte Goals.
- **Status 2026-09-08: UMGESETZT** — `GOAL_ALIASES` + `canonical_goal()` + `verify_map_consistency()`
  in `goal_priority_protocol.py`; `hearing_tier()`/`priority_of()` nutzen die Kanonisierung;
  Sync-Test `tests/unit/test_goal_tier_map_sync.py` (5 Tests) läuft in unit-smoke/coverage.

## TODO-P0-3 · Budget-Wahrheit: drei Zahlen in eine konvergieren

- **Ziel:** Performance-Budget-Tabelle (240 s/min), PerformanceGuard-Limits (32× RT für alle Modi,
  obwohl Balanced-Doku „3× RT“ sagt) und Realität (53× RT) in **eine** normative Quelle bringen;
  `add_analytics_overhead()`-Verschleierung durch ehrliches Messzeit-Reporting ersetzen.
- **Wirkung:** Kein Schein-Soll mehr; Budget-Entscheidungen (Phase-Deferrals) werden korrekt kalibriert.
- **Beleg:** `.github/copilot-instructions.md` Performance-Budget; `backend/core/performance_guard.py:46,53,120–130,293`.
- **Akzeptanz:** Eine Budget-Norm mit Querverweisen; Benchmark-Matrix meldet Verletzungen ehrlich;
  nach TODO-P0-1 neu kalibriert (Ziel: 32× wieder erreichbar).

## TODO-P1-1 · Modell-Residency & Warm-up-Policy

- **Ziel:** Spezifizieren und umsetzen, welche Modelle warmgehalten werden (Residency/LRU je Session)
  und wie Warm-up einmalig je Modell amortisiert wird; deterministisches Multi-Song-Batching desselben
  Modells unter Wahrung von §G1 (Seed-Isolation pro Song).
- **Wirkung:** ~5 min Modell-Ladezeit je Lauf entfällt; batchfähige GPU-Nutzung senkt 53× weiter.
- **Beleg:** Session-Befund (jede Matrix-Zelle lädt Modelle neu); `spec 15 §15.9` (InferenceSessionManager
  Roadmap); `copilot-instructions.md` §G1.
- **Akzeptanz:** Zweiter Lauf in derselben Session ohne Modell-Nachladen; Determinismus-Nachweis je Song.

## TODO-P1-2 · Separation auf VS-1/GSEP + Demucs v5 heben

- **Ziel:** Vocal-Router um VS-1 (SongEval-2025-Gewinner) als Top-Stufe erweitern; Demucs v5 als
  zusätzliche Stufe prüfen. Hörordnungs-Invarianten (Sänger-Identität, Ebene 1) decken das Risiko ab.
- **Wirkung:** Größter Hörgewinn je Aufwand (Stem-Ersetzung, `separation_fidelity`, Ebene 3).
- **Beleg:** `backend/core/sota_vocal_model_router.py:4,57–150` (aktuelle Kette BS-RoFormer → Demucs v4 → MDX23C);
  Tiefenanalyse A.1.
- **Akzeptanz:** A/B-Metrik `separation_fidelity` + `singer_identity_cosine` ≥ MDX23C-Stand; Hörstichprobe.

## TODO-P1-3 · Audibility (JND/Masking) auf alle Schwellwert-Guards

- **Ziel:** Formant-, Wärme-, Onset-, Spektralfarben-, Gain-Step-Toleranzen von fixen dB/Korrelations-Werten
  auf `max(fixed, lokale_Maskierungs-JND)` umstellen (wie Hörordnung Ebene 2 es als Prinzip fordert).
- **Wirkung:** Weniger falsche Rollbacks (Qualität) und weniger unnötige End-Gate-Recovery-Runden (Performance).
- **Beleg:** `dsp.instructions.md` §0p/§WBG/§SCK/§ATI; `hoerordnung.instructions.md` §4;
  `backend/core/residuum_masking.py` (bereits maskierungsbasiert — als Muster).
- **Akzeptanz:** Guard-Entscheidungen mit Maskierungskontext nachweisbar (Logs); Regressionstests für
  maskierte vs. unmaskierte Fälle.

## TODO-P1-4 · Externe Blind-Hörstudie + GPU-A/B-Kalibration

- **Ziel:** MUSHRA-Blindstudie nach `docs/guides/MUSHRA_STUDIENPROTOKOLL.md` mit menschlichen Hörern;
  CPU-vs-ROCm-Gleichwertigkeitslauf erzwingen (löst „CPU ist Referenz“ vs. GPU-Produktion auf).
- **Wirkung:** Der zentrale „Ohr entscheidet“-Anspruch wird validiert; GPU-Betrieb bekommt Referenz-Status.
- **Beleg:** `spec 15_world_class_gap_closure.md §15.3/§15.10` („null menschliche Hörer“);
  `v10.900 §9.3` (CPU-Referenz).
- **Akzeptanz:** Studienbericht + statistische Auswertung im Repo; GPU-A/B-Bit-Identität oder dokumentierte
  tolerierte Abweichung.

## TODO-P2-1 · Hygiene: UTF-16-Bereinigung + Monolith-Hinweis

- **Ziel:** UTF-16/UTF-16LE-kodierte Code-Teile in `backend/core/` nach UTF-8 konvertieren (brechen
  grep-basierte CI-Scans); Refactor-Plan für das 45.309-Zeilen-God-Object `unified_restorer_v3.py`
  (Budget-, Gate-, Recovery-, Chunk-Logik getrennt) skizzieren.
- **Wirkung:** CI-Scans greifen wieder; Konsistenz-Lücken (Tiefenanalyse Abschnitt C) werden strukturell
  seltener.
- **Beleg:** Tiefenanalyse Methodik-Absatz; `unified_restorer_v3.py`.
- **Akzeptanz:** `file` meldet UTF-8 für alle .py; Refactor-Plan als Doc, keine Verhaltensänderung.

---

## Hintergrund (damit die nächste Session sofort einsteigt)

- **Session-Report:** `docs/reports/current/2026-09-08_envelope_root_cause_sota_fixes_matrix.md`
  (Abschnitt 10 = Tiefenanalyse; Abschnitte 1–9 = Root-Cause + Fixes + Matrix).
- **Matrix-Ergebnis (3 Zellen, gleicher Clip) — ABGESCHLOSSEN 2026-09-08:**
  3× EXIT=0, 3× bit-identischer Output (MD5 765c3f544c279f205d32288eef5db95c),
  Envelope μ=0.815 σ=0.081 (7 Chunks, nie mehr degeneriert), Einladungs-Gate:
  Zelle 1 (Baseline) 8× NICHT BESTANDEN → Zellen 2+3 (Fix-Stand) 8× BESTANDEN
  (8 Exemptions an Reparaturstellen, raw 0.562 → effektiv 0.193), Laufzeit
  3 h 19–38 min je Zelle (P0-1 zielt auf ≤40 min). bit-identischer Output (MD5 `765c3f54…`),
  Envelope μ=0.815 σ=0.081 (vor Fix μ=0.060 σ=0.000), Einladungs-Gate BESTANDEN nach
  Reparatur-Exemption (raw 0.562 → effektiv 0.193), Laufzeit 3 h 19–38 min je Zelle.
- **Bereits umgesetzt (nicht neu machen):** B3-Early-Merge-Fix, `_flow_meta`-Spiegel,
  measure_all-Verwerf-Fix, FC-Hörordnungs-Pre-Filter, Einladungs-Gate-Exemption,
  MDX23C-API-Drift, BasicPitch-Fixed-Length, Chunked-Prior (§m2), Envelope-Regressionstest,
  RELEASE_MUST Strength-Envelope-Nichtdegeneration, Ledger-Merge, GUI-Smoke-Flag-Fix.
