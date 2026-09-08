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
- **Status 2026-09-08: TEILWEISE UMGESETZT (Slice A + a1 + a2).**
  `_should_run_end_gate_cascade()` + Kwargs `_chunked_tail_skip`/`_chunked_last`;
  im Chunked-Pfad läuft die End-Gate-Recovery-Kaskade nur noch auf dem letzten Chunk.
  Block a1: `_measure_goals_for_tail()` überspringt `measure_all()` auf
  Nicht-letzten-Chunks; `_restore_chunked` misst die GOAL_SCORECARD einmal auf dem
  assemblierten Song. Block a2: `_run_song_level_end_gate()` — die 654-Zeilen-Kaskade
  als Methode extrahiert und läuft nach der Assembly EINMAL song-global (kompensiert
  Slice A qualitativ; metadata `p0_1_song_end_gate_applied`). Tests: 11 Fälle.
  Verbleibende Blöcke: (b) Einladungs-Gate, (c) MQA/_collect_reporting_analytics,
  (d) Audibility-Gate.
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

## TODO-P1-5 · §v10.709 authentizitaet-Erhalt nach phase_12_wow_flutter_fix

- **Befund (2026-09-08, Verifikationslauf P0-1):**
  `WARNING §v10.709 Quality-Degradation #1 nach phase_12_wow_flutter_fix: ['authentizitaet']` —
  die Flutter-Korrektur (4–100-Hz-Band, 45 % des Wow/Flutter-Blends) flacht auch
  Vibrato/Intonations-Bends der Performance ab (authentizitaet = versa_similarity fällt).
- **SOTA-Lösung (UMGESETZT 2026-09-08, §AUTH-P12):** `_preserve_musical_modulation()` in
  `backend/core/phases/phase_12_wow_flutter_fix.py` — Root-Cause statt Workaround (§V7):
  Wo die musikalische Modulationstiefe (Vibrato-Band, Vokal-Frames) die Flutter-Korrektur
  dominiert, wird die Korrektur proportional Richtung Identität zurückgenommen (max. 85 %);
  mechanischer Wow/Flutter bleibt voll korrigiert. Deterministisch, NaN/Inf-geschützt (§0a).
- **Beleg:** `tests/unit/test_phase_12_musical_modulation_preservation.py` (6 Fälle, grün):
  Vibrato → ≥60 % zurückgenommen, Wow-only → unverändert, Passthrough/NaN/Short-Guards.
- **Akzeptanz:** Nächster Referenzlauf ohne §v10.709-authentizitaet-Warnung nach phase_12;
  bit-identischer 3-Zellen-Output (Determinismus §G5); Wow/Flutter-Reduktion unverändert.

## TODO-P1-6 · DeepFilterNet ML→DSP-Fallback (dec.onnx ohne Alpha-Head) — UMGESETZT 2026-09-08

- **Befund (Live-Log 07:35:55, 2026-09-08):** `IndexError: list index out of range` in
  `_infer_spectral_chunk` (`alpha = dec_out[1]`) → stiller OMLSA-Fallback statt trainiertem DFN.
- **Root-Cause (gemessen):** Der DFN3-Export `models/deepfilternet_v3_ii/{,finetuned/}dec.onnx`
  hat NUR einen Output (`coefs`) — `df_fc_a` (Alpha-Head) ist im trainierten DeepFilterNet3-
  Forward unbenutzt (df/deepfilternet3.py:321 definiert, Forward wendet `df_op(coefs)` ohne
  Alpha-Blend an). `dec.onnx.orig` (DFN2-Ära, emb=512) ist nicht kompatibel. Das Export-Skript
  hatte einen toten `alpha`-Verweis in `dynamic_axes`.
- **Lösung (§P1-6):** (1) Plugin: alpha optional (`dec_out[1] if len>1 else None`),
  `alpha=None` → pure DF wie trainierter Forward (blend=1.0); Load-Time-Warnung bei fehlendem
  Alpha-Head. (2) Gepolsterte Rand-Chunks (T=100-Modell, l<100): nur die ersten l
  Output-Frames von Maske/Koeffizienten verwenden (Broadcast-Fix). (3) Kurze Signale
  (S<T): Chunk-Pfad mit Pad+Trim statt Ganzsignal (enc erwartet exakt T=100).
  (4) `export_df_musik_onnx.py`: toten Alpha-Verweis entfernt, DFN3-Realität dokumentiert.
- **Beleg:** `tests/unit/test_deepfilternet_plugin_alpha.py` (4 Fälle, grün);
  Echt-Modell-Probe: 1 s + 3 s Audio vollständig durch den ONNX-Pfad (0.04/0.19 s),
  kein Fallback mehr.
- **Akzeptanz:** Nächster Lauf ohne „ML→DSP-Fallback“-Traceback für DeepFilterNet;
  Rauschunterdrückung über trainiertes DFN statt OMLSA.

## TODO-P1-7 · Keine hörbaren Restdefekte: m1b intern ausführen (Stufe-2-Nachbehandlung)

- **Ziel (User-Anforderung 2026-09-08):** In allen Importfiles sollen keine hörbaren
  Restdefekte übrig bleiben — bei vollem Erhalt von Musikalität und Klang, soweit möglich.
- **Befund (§v10.703 Defekt-Countdown, Lauf 2026-09-08):** 46 gefunden → 42 über
  Hörbarkeits-Schwelle → 3 behoben → **42 über Schwelle verbleibend**. Die m1b-Queue
  (Hörbarkeits-Gate → `deferred_phases`) wurde bisher NUR in die GUI-KMV-Queue gestellt;
  im Headless-/CLI-Flow konsumierte niemand sie → Restdefekte blieben unangetastet.
- **Lösung (§P1-7):** `_run_m1b_targeted_retry()` in `unified_restorer_v3.py` führt die
  sicher zugeordneten Retry-Phasen (`DEFECT_RETRY_PHASE_MAP`: hum/clicks/crackle/wow-flutter/
  hiss/reverb/echo/compression) intern EINMAL aus: nur Phasen mit klarer Typ-Zuordnung
  (§V7 kein „mehr von allem“), verbotene Phasen (§0a) ausgeschlossen, Re-Entry-Guard
  `_m1b_pass_active`, deterministisch, bei Fehler/keiner Ausführung bleibt das Original
  (kein Audio-Ersatz). Verkabelt: (1) restore()-Tail nach dem Hörbarkeits-Gate
  (`m1b_retry_applied`/`m1b_retry_types` im Ergebnis-Metadata), (2) `_restore_chunked`
  nach der Song-Assembly (song-global, letzter Chunk-Scan bestimmt die Typen).
- **Beleg:** `tests/unit/test_m1b_targeted_retry.py` (6 Fälle, grün: nur gemappte Phasen,
  §0a-Ausschluss, Re-Entry-Guard, no-exec → None, Chunk-Shift-Restore);
  Restorer-Suiten 289 passed.
- **Akzeptanz:** Nächster Referenzlauf: `n_audible_unmasked` deutlich reduziert (Ziel: 0,
  soweit physisch möglich — `physical_cap`-Typen dokumentiert ausgenommen);
  `m1b_retry_applied=True` im Metadata; keine Verschlechterung von
  authentizitaet/natuerlichkeit (GOAL_SCORECARD ≥ Vorlauf).

## TODO-P1-8 · Export NACH dem 2. Durchgang — FinalPolish/OneTakeExport ans Tail-Ende — UMGESETZT 2026-09-08

- **Befund (User-Frage 2026-09-08):** Lief der Export nach dem 2. Durchgang? Nein —
  `apply_final_polish` (Era-EQ + Noise-Shaped Dither) und `OneTakeExport.prepare`
  (LUFS/True-Peak) liefen bei Z. 14059/14105 MITTEN im Tail (STUFE-8), die
  m1b-Nachbehandlung erst bei Z. 22730 — der m1b-Output wurde nie neu exportiert:
  Dither/LUFS/TP galten für einen Zwischenstand, Humanization/PEO/MDEM/Goosebumps/m1b
  liefen teils auf bereits gedithertem Audio.
- **Lösung (§P1-8):** Export-Finalisierung (FinalPolish → OneTakeExport) ans TAIL-ENDE
  verschoben — NACH m1b. Dither ist damit der letzte Quantisierungsschritt (§V5),
  LUFS/True-Peak gelten für das FINALE Audio, alle DSP/ML-Schritte laufen auf voller
  Float-Präzision. `result.audio` wird nach der Finalisierung aktualisiert.
  Chunked-Pfad: nach song-globaler m1b nur OneTakeExport (idempotente Zielkorrektur);
  FinalPolish lief je Chunk und wird nicht doppelt angewendet (kein Doppel-EQ).
- **Beleg:** Restorer-Suiten 296 passed (inkl. Alignment); Linter/GEBOTE clean.
- **Akzeptanz:** Referenzlauf: Log zeigt „§P1-8 FinalPolish (nach m1b)“ und
  „§P1-8 OneTakeExport (nach m1b)“ AM ENDE; LUFS/TP im Zielband; bit-identischer
  3-Zellen-Output (Determinismus §G5 innerhalb der Version).

## TODO-P2-1 · Hygiene: UTF-16-Bereinigung + Monolith-Hinweis — ERLEDIGT 2026-09-08 (Guard-Teil)

- **Befund (gemessen 2026-09-08):** Alle 3175 getrackten Textdateien sind valides UTF-8;
  0 UTF-16-Dateien, 0 BOMs, 0 invalide Sequenzen. Das früher beobachtete „UTF-16-Garble“
  war ein Anzeige-Artefakt des Tool-Kanals (UTF-8-Bytes werden in manchen Ausgaben als
  UTF-16LE fehlinterpretiert — per Hex-Analyse belegt), kein Repo-Zustand.
- **Umgesetzt:** `scripts/utf8_hygiene_check.py` (fail-closed: R1 UTF-16/32-BOM,
  R2 invalide UTF-8-Sequenzen, R3 NUL-Byte-Fenster ≥10 % je 2-KiB-Fenster, auch in
  Dateien < 2 KiB) + Pre-Commit-Hook `aurik-utf8-hygiene` + FILE_REGISTRY-Eintrag +
  Drift-Baseline nachgezogen. Negativtest: BOM-Datei und BOM-lose UTF-16LE-Datei → EXIT 1.
- **Offen (Folge-Session):** Refactor-Plan für das 45.309-Zeilen-God-Object
  `unified_restorer_v3.py` (Budget-, Gate-, Recovery-, Chunk-Logik getrennt) als Doc skizzieren.
- **Akzeptanz (erfüllt):** `file` meldet UTF-8 für alle .py; Guard verhindert Regressionen.

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
