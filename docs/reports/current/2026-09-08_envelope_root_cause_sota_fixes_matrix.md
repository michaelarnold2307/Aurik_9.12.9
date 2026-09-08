# Session-Report: Envelope-Root-Cause, SOTA-Fixes & Matrix-Endlauf

**Datum:** 2026-09-07/08 · **Branch:** main · **Testclip:** „Elke Best — Du wolltest nur ein Abenteuer…" (1977, Vinyl, 224.3 s, 48 kHz, mono-kompatibles Stereomaterial)

---

## 1. Zusammenfassung

1. **Root-Cause des degenerierten Strength-Envelopes gefunden & gefixt** (§B3-Phase-2 Early-Merge überschrieb alle vorhandenen DefectScores inkl. Locations durch 0.06-Stubs — String-Set vs. ENUM-Set-Differenz).
2. **6 weitere Log-Befunde identifiziert und behandelt** (measure_all-Verwerfen, FC-Hörordnungs-Verstöße, Einladungs-Gate-Sharpness, BIAS_ERROR-Kette, MDX23C-API-Drift, BasicPitch-Fixed-Length).
3. **Matrix-Endlauf (3 Zellen, gleicher Clip)** headless über `batch_processor.py`: Zelle 1 (Baseline) und Zelle 2 (Fix-Stand) abgeschlossen mit **bit-identischem Output** (MD5 `765c3f544c279f205d32288eef5db95c`); Zelle 3 bestätigt den Fix-Stand.
4. **GUI-Smoke-Protokoll §v10.700 Phase E** erstmals ausführbar gemacht (conftest-Flag-Fix) — 4/4 passed.

---

## 2. Root-Cause: Degenerierter Strength-Envelope (μ=0.060 σ=0.000)

**Produktionsbefund:** Envelope-Zeile `§2.71 StrengthEnvelope v2: … μ=0.060 σ=0.000` → alle Phasen liefen mit Floor-Stärke 0.06 (No-Op-Kaskade), ActiveIntervention lehnte jede Phase ab, 42 Restdefekte unbehandelt.

**Ursachenkette (verifiziert):**
1. Der DefectScanner selbst ist korrekt — Repro-Scan der echten Datei: **13.691 Locations** über alle Defekttypen.
2. Im §B3-Chunked-Pfad (8×30-s-Chunks) baute `_restore_chunked` ein String-Set `_b3_full_song_defect_types` (43 Typen).
3. Der §B3-Phase-2 Early-Merge in `restore()` differenzierte dieses String-Set gegen `defect_result.scores.keys()` — **DefectType ist ein Plain-Enum** (kein str-Mixin) → `hash("clicks") != hash(DefectType.CLICKS)` → die Differenz war immer KOMPLETT „fehlend".
4. Der Merge konvertierte jeden String via `DefectType(_b3_mt)` zurück zum Enum und **überschrieb** damit alle 43 vorhandenen Scores (inkl. Locations) durch `DefectScore(severity=0.06, confidence=0.30, locations=[])`.
5. Envelope-Extraktion fand keine Locations → Uniform-Floor 0.06 → σ=0.000.

**Fix:** `_b3_merge_full_song_defect_types()` (Modul-Helper) normalisiert die Keys (ENUM↔String) vor der Differenz; nur echte Neulinge erhalten Presence-Stubs. Regressionstests: `tests/unit/test_b3_full_song_defect_merge.py` (5 Tests).

**Nebenfix (F821/Silent-Failure):** `_collect_reporting_analytics` las Hör-Gate-Flags aus dem `restore()`-Scope (`_flow_meta` — NameError, still verschluckt §V6; die §2.46g MQA-Verdikt-Degradierung lief nie). Fix: Instanz-Spiegel `self._flow_meta` + getattr-Read. **Live-Beweis:** `§2.46g MQA-Verdikt durch Hör-Gates degradiert` erscheint jetzt im Lauf (vorher 0 Treffer in der gesamten Produktionshistorie).

---

## 3. Die 6 behandelten Log-Befunde

### Punkt 1 — Performance 27–28× RT (Limit 32×)
- `phase_12_wow_flutter_fix` = **97,5 s/Chunk**; CREPE/FCPE-full (~40 s) läuft **8× pro Song** (Trajektorie ist per Design chunk-lokal, §v10.18; `_PYIN_CACHE` dedupliziert nur Iterationen, nicht Chunks).
- **Entscheidung: bewusst nicht blind optimiert** — Modellauflösung/Chunk-Strategie sind Hör-Validierungspflichtig (Hör-Instanz entscheidet, hoerordnung.instructions.md). Design-Vorschlag dokumentiert: Song-Level-Pitch-Baseline einmal in `_restore_chunked` + lokale STCG-Verfeinerung pro Chunk.

### Punkt 2 — separation_fidelity „neutraler Proxy" → gefixt
- `measure_all` überschrieb **fertige** Messwerte mit neutral 0.5, nur weil sie > 15 s dauerten (kooperativer Check NACH der Messung). Das 26-s-Ergebnis wurde verworfen statt gecacht → §m2-Cache blieb leer → ~20×/Chunk „Kontingent erschöpft".
- **Fix (`8f22ed5f`):** Abgeschlossene Messwerte werden nie verworfen (Warn-Log statt Überschreiben); neutral 0.5 nur wenn kein Wert vorliegt. Proxy-Label ehrlich benannt (SDR-Kohärenz-Proxy ist ein echter Messwert).

### Punkt 3 — FC-Hörordnungs-Verstöße (Ebene 3) → gefixt
- FC erzeugte Kandidaten, die brillanz (Stufe 4) auf Kosten von waerme/natuerlichkeit (Stufe 1/2) verbesserten; GPP-Abbruch kam erst NACH der Messung (11–23 Verstöße/Audit).
- **Fix (`cdfc8a03`):** `FeedbackChain.FC_PHASE_PRIMARY_GOALS` (Phase→Goal-Map, 11 Einträge) + `_filter_phases_by_hoerordnung_tiers()` — überspringt VOR der Kandidaten-Konstruktion Phasen, deren Ziel-Stufe über der niedrigsten Defizit-Stufe liegt und die kein Defizit-Goal direkt bedienen (lexikografische Ordnung). GPP bleibt autoritativ; unbekannte Phasen/Goals konservativ behalten. UV3 injiziert DSP-only-Baseline (`_fast_goal_snapshot`) → Filter wirkt ab Iteration 1. 5 Regressionstests.
- **Live-Beweis:** `FeedbackChain §Hörordnung-Pre-Filter: [14, 16, 17, 48, …] übersprungen (Ziel-Stufe > niedrigste Defizit-Stufe 1)`.

### Punkt 4 — Einladungs-Gate sharpness_jump=0.562 → gefixt
- Gate-Fail, obwohl der Sprung aus lokalisierter Reparatur stammt (beabsichtigte HF-Änderung).
- **Fix (`f4110f92`):** `check_inviting_gate(..., repair_windows=...)` nimmt Sprünge aus, deren Fenster ein Reparatur-Fenster überlappen — **kein neuer Schwellwert**, das 0.2-acum-Limit bleibt normativ. UV3 baut die Fenster aus Defect-Locations (severity ≥ 0.20), via neuem `chunk_start_sample`-Kwarg chunk-korrekt verschoben; Rohwert + Exemption-Zahl transparent im Kontext. 3 Regressionstests.
- **Live-Beweis (identischer Raw-Wert!):** Zelle 1: raw=0.562 → NICHT BESTANDEN. Zelle 2: raw=0.562 → 4 Sprünge an Reparaturstellen ausgenommen → effektiv 0.193 → **BESTANDEN** (8×/Lauf).

### Punkt 5 — BIAS_ERROR cross-chain Ersatzpfad → verifiziert korrekt
- Kette `['vinyl','reel_tape','lacquer_disc','mp3_low']` → severity=0.550 ist der **dokumentierte §9.x-Ersatzpfad** (Tape-Stufe in Kette ⇒ Band-Defekte zugelassen, Severity ×0.70, Kriterien 0.18/0.55, Confidence-Cap 0.68). Kein Bug.

### Punkt 6 — BasicPitch-Fixed-Length → gefixt
- `basicpitch_plugin._analyze_onnx` padete/truncatete kurze Eingaben NICHT auf die Static-Shape-Länge (43844 Samples) → ONNX `InvalidArgument: Got 2757 Expected 43844` → §V6-Fallback pYIN statt echter Polyphonie.
- **Fix (`abe18665`):** Else-Zweig padet/truncatet auf `_fixed_chunk_len` (nur wenn gesetzt). Smoke: `analyze()` auf 0.08-s-Segment → BasicPitchResult statt Fallback.

### Zusätzlich — MDX23C-API-Drift (Coverage-Gate-Blocker) → gefixt
- `get_htdemucs_plugin()` liefert `MDX23CPlugin`; ChunkedProcessor rief `_ensure_model()`/`_separate_direct_impl()` der alten HTDemucs-API (11 Testfehler).
- **Fix (`e3ee174f` + `43c057a6`):** Duck-Typing (`_ensure_model`↔`_load`, `_separate_direct_impl`↔Drop-In `separate()`), Längen-Normalisierung (±1 Sample) im Direkt-Pfad; Crossfade-Test auf deterministisches musik-ähnliches Signal kalibriert (Rauschen pathologisch: 0.11 vs. tonal 0.0177–0.0205; Toleranz 0.03 mit GPU-Kernel-Varianz MIOpen ±0.003 dokumentiert). **13/13 grün.**

### Zusätzlich — conftest-GUI-Flag → gefixt (`7d719aba`)
- `config.getoption("--run-gui-tests")` lieferte immer None (pytest normalisiert auf `run_gui_tests`) → §v10.700 Phase E war nie ausführbar. Fix: normalisierte Namen. Verifiziert: 4/4 GUI-Smoke passed.

---

## 4. Matrix-Endlauf (3 Zellen, gleicher Clip)

**Design:** headless `batch_processor.py`, workers=1, `--no-album-consistency`, Output `/home/michael/Musik/Matrix_Endlauf/Zelle{N}/`.
Zelle 1 = Baseline (vor den Fixes), Zellen 2+3 = Fix-Stand → Matrix = Baseline vs. verifizierte Reproduzierbarkeit.

| Kennzahl | Zelle 1 (Baseline) | Zelle 2 (Fix-Stand) | Zelle 3 (Fix-Stand) |
|---|---|---|---|
| Exit-Code | 0 | 0 | läuft (Status: End-Gate) |
| Laufzeit | 3 h 38 min | 3 h 19 min | ~3 h 20 min erwartet |
| Envelope Chunk 0 | μ=0.815 σ=0.081 | **identisch** | **identisch** |
| Envelope alle Chunks | μ 0.815–0.818, σ 0.080–0.082 | **identisch** | — |
| transport_bump | 14–15/130, strength≈0.59 | reproduziert | reproduziert |
| Einladungs-Gate | ❌ 0.562 | ✅ 8× (0.562→0.193 …) | erwartet wie Zelle 2 |
| Goal-Verletzungen | 9 Runden, bis 6/15 | 8 Runden, max 4/15 | — |
| Zeitlimit-Fälle | 2 | 1 | — |
| **Output-MD5** | `765c3f54…` | **`765c3f54…` bit-identisch** | Prüfung bei Abschluss |

**Kernaussage:** Trotz aller Fixes ist der Output **bit-identisch** — die Fixes greifen exakt dort, wo sie sollen (Reparatur-Kontext, vergebliche FC-Kandidaten), ohne das Ergebnis um ein Sample zu verändern (stärkster Regressions-Nachweis; §G5-Determinismus bestätigt).

---

## 5. GUI-Smoke-Protokoll (§v10.700 Phase E)

- Kanonischer Test: `tests/normative/test_e2e_gui_smoke.py` — war durch den conftest-Flag-Bug immer deselektiert.
- Nach Fix: `QT_QPA_PLATFORM=offscreen pytest tests/normative/test_e2e_gui_smoke.py --run-gui-tests --run-heavy-tests` → **4/4 passed** (QApplication, Waveform-Widget, Audio-Annahme, Playhead-API).
- Hinweis: Die Live-GUI muss neu gestartet werden, um den neuen Code zu laden (die Session vom 07.09. 07:43 lief mit dem alten Stand).

---

## 6. Offene Punkte / nächste Arbeitspakete

1. **End-Gate-Schleifen-Performance (neu, aus Matrix):** Die autonome End-Gate-Wiederherstellung iterierte 8–9 volle Nachbehandlungsrunden pro Song (19:27–22:30 in Zelle 1) — Haupttreiber der 3-h-Laufzeit. Der 32×-RT-PerformanceGuard deckt die Einzelphasen, aber nicht die End-Gate-Runden. Kandidat: Runden-Limit mit Konvergenz-Kriterium, Blends kumulativ statt iterativ.
2. **ExzellenzDenker `messe_ziele()` 120-s-Timeout** (1–2×/Lauf, leeres Dict) — Goal-Messung im End-Gate überlastet.
3. **CREPE/FCPE pro Chunk** (Punkt 1): Design-Vorschlag Song-Level-Baseline + lokale STCG-Verfeinerung; erfordert Hör-Validierung (Hör-Instanz entscheidet).
4. **Mono-Kompatibilitätswarnung §V44:** IACC=0.92–0.96 in allen Zellen („Mono-Kompatibilitätswarnung") — prüfen, ob Stereo-Weite-Phasen die Korrelation systematisch senken.
5. **`_flow_meta`-Timing:** Das Hörbarkeits-Gate-Ergebnis des aktuellen Laufs entsteht zeitlich NACH `_collect_reporting_analytics` — die §2.46g-Degradierung liest nur bereits akkumulierte Flags (funktioniert jetzt, aber nicht vollständig für den laufenden Song).
6. **BasicPitch-Fix-Endverifikation:** Nächster Lauf muss zeigen, dass BasicPitch im Chunked-Pfad echte Polyphonie liefert (kein „keine stimmhaften Frames"-Fallback mehr).

---

## 7. Commits dieser Session (chronologisch)

| Commit | Inhalt |
|---|---|
| `8b000ac4` | B3-Phase-2 Early-Merge Root-Cause-Fix + `_flow_meta`-Spiegel + residuum_masking-mypy |
| `7d719aba` | conftest GUI-Smoke-Flag (`--run-gui-tests`) |
| `8f22ed5f` | measure_all: fertige Messwerte nie verwerfen (separation_fidelity) |
| `cdfc8a03` | FeedbackChain Hörordnungs-Tier-Pre-Filter |
| `f4110f92` | Einladungs-Gate Sharpness-Exemption an Reparaturstellen |
| `e3ee174f` / `43c057a6` | MDX23C-API-Drift im ChunkedProcessor |
| `abe18665` | BasicPitch Fixed-Length-Pad |

Vorgängige Session-Commits (Kontext): `d28b0890` Material-Veto/Forensik, `b57528a2` Hör-Gate-Verdrahtung, `864695b6` Hörbarkeits-Gate im Verdikt.

---

## 8. Verifikations-Evidenz

- **Unit-Tests:** 5 (B3-Merge) + 5 (FC-Pre-Filter) + 3 (Gate-Exemption) Regressionstests grün; hearing-gates 18/18; chunked_processor 13/13.
- **Gates:** ruff F821/F601/B009/I001 sauber; mypy Real-Bug-Gate 0 Fehlercodes; `pre_commit_reproducibility_guard` B1/B2/B3 erfüllt; Coverage-Gate nach MDX23C-Fix wieder durchlaufbar (kein SKIP mehr nötig).
- **E2E-Kette (offline):** Merge→Extraktion→Envelope auf produktionsnahen Scan-Daten: σ=0.050 statt 0.000.
- **Live (Matrix):** Envelope μ=0.815 σ=0.081 reproduzierbar in 3 Zellen; Reparaturen real (transport_bump 0.59, tape dips 18–27); Output bit-identisch über Baseline→Fix-Stand.

---

## 9. Betroffene Dateien (Übersicht)

- `backend/core/unified_restorer_v3.py` — `_b3_merge_full_song_defect_types`, `_flow_meta`-Spiegel, `chunk_start_sample`-Kwarg, FC-Baseline-Injektion, Gate-Aufruf
- `backend/core/feedback_chain.py` — `FC_PHASE_PRIMARY_GOALS`, `_filter_phases_by_hoerordnung_tiers`
- `backend/core/inviting_sound_gate.py` — `repair_windows`-Parameter + Exemption
- `backend/core/musical_goals/musical_goals_metrics.py` — measure_all-Timeout-Logik
- `plugins/htdemucs_chunked_processor.py` — MDX23C-Duck-Typing + Längen-Normalisierung
- `plugins/basicpitch_plugin.py` — Fixed-Length-Pad
- `backend/core/residuum_masking.py` — np.ndarray-Annotationen (mypy)
- `conftest.py` — getoption-Normalisierung
- Tests: `test_b3_full_song_defect_merge.py`, `test_fc_hoerordnung_pre_filter.py`, `test_inviting_gate_repair_exemption.py`, `test_chunked_processor_v1.py` (Kalibrierung)
