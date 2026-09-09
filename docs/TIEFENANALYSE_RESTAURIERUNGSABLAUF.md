# Tiefenanalyse: Gesamter Restaurierungs-Ablauf (2026-09-08)

> **Auftrag**: Prüfen, ob der Ablauf vom Import bis zum Export sinnvoll ist oder
> der Tail nach den Phasen kontraproduktiv/pure Zeitverschwendung ist; ob 30-s-Chunks
> den GESAMTEN Prozess durchlaufen statt der ganze Song; und ob nach der Restaurierung
> eine alternative Parallelversion läuft. Ziel: nur EINE Version, die das Beste vereint;
> Parallelversion komplett entfernen.

---

## 1. Gesamtbild des Ablaufs (Ist-Zustand, mit Belegen)

```
Import/Audio laden
  └─ AurikDenker.denke()                        (denker/aurik_denker.py, ~2103)
       ├─ Analysen: Era, Genre, Medium, DefectScan, Restorability  (Caches)
       ├─ ReparaturDenker / RekonstruktionsDenker (Gaps, Bandbreite)
       └─ RestaurierDenker.restauriere()          (denker/restaurier_denker.py:356)
            └─ UnifiedRestorerV3.restore()        (backend/core/unified_restorer_v3.py:7833)
                 ├─ wenn Audio > 120 s  → _restore_chunked()  (Zeile 45042)
                 │    └─ Chunks à 30 s (≤300 s Songs) / 60 s (>300 s), Zeile 45075
                 │         └─ PRO CHUNK: self.restore() = KOMPLETTE Pipeline
                 │              (Vor-Analyse nur Chunk 0; Zustand eingefroren, §B3)
                 └─ sonst: Pipeline direkt auf dem ganzen Audio
                 ├─ Phasen (_execute_pipeline, Zeile 35736)
                 ├─ pct 86–98 Tail: FeedbackChain → Audio-Nachbearbeitung →
                 │   Qualitätsprüfung → Musical Goals → Finalisierung
                 └─ Chunked-Tail (nur letzter Chunk: _chunked_tail_skip, Zeile 8105):
                      End-Gate-Kaskade (16727), m1b-Gate (18041/18115, 22674)
  └─ _restore_chunked: Assembly → Song-SCORECARD (Analytik) →
       m1b Targeted Retry auf dem ASSEMBLIERTEN Song (45465) → Rückgabe
GUI (modern_window.py): BatchThread → Ergebnis → Export (Datei schreiben)
```

## 2. Frage 1: Laufen 30-s-Chunks durch den GESAMTEN Prozess? — JA

**Beleg**: `_restore_chunked` (Zeile 45042) ruft für jeden Chunk `self.restore(...)`
vollständig auf (erster Chunk ~45198, Folge-Chunks ~45360; per-Chunk-Progress-Wrapper
~45272–45309). D. h. **jeder 30/60-s-Chunk durchläuft die komplette Phase-Liste plus
Vor-/Nachschritte** — nicht der ganze Song.

**Bewertung**:
- Der Entwurf war bewusst **RAM O(1)** (§v10.452: „wird IMMER gechunked. RAM O(1),
  Modelle bleiben geladen“) und ist deterministisch abgesichert (B3-Chunk-Determinismus).
- **Nachteil (fundamental)**: Alle Entscheidungen, die Song-Globalität brauchen —
  Loudness/LUFS, FeedbackChain, Musical-Goals, Defekt-Hörbarkeit, Formant/Gender,
  Stereo-Phase — laufen auf 30-s-Fenstern und können Song-Kontext nicht sehen.
  Chunk-Grenzen erzeugen zusätzlich Nahtrisiko (Crossfade-Handling nötig).
- **Die Separation läuft aktuell NICHT chunk-übergreifend geplant**:
  - BS-RoFormer-Separation (Studio-Modus) läuft pro restore()-Aufruf = pro Chunk
    (Zeile 15179 ff.).
  - Demucs läuft nur in phase_42 (im Restoration-Modus gemäß §0a verboten) bzw.
    im experimentalen `source_aware_restorer` (Env-Opt-in `AURIK_SOURCE_SEPARATION=1`).
  - MelBandRoFormer-Top-Stufe (Router) verarbeitet den Input ihres restore()-Aufrufs
    (im Chunked-Pfad also je Chunk).

**Empfehlung (geplant, siehe §6)**: Pipeline EINMAL auf dem ganzen Song; ausschließlich
die Separation intern chunkweise (BS-RoFormer/Demucs haben interne Chunked-Inferenz).
Das Separationsmodell auf Ganz-Song umzustellen ist NICHT nötig und wegen Modell-Limits
auch nicht sinnvoll — die interne Chunked-Inferenz liefert ohne Qualitätseinbußen dasselbe
Ergebnis (Overlap-Add), daher bleibt Separation chunkweise.

## 3. Frage 2: Ist der Tail (pct 86–98) sinnvoll oder Zeitverschwendung?

| Schritt | pct | Audio-Einfluss? | Urteil |
|---|---|---|---|
| FeedbackChain (Korrektur-Schleife) | 86–89 | JA — passt Phasen-Stärken an Messwerte an | **sinnvoll** (Qualitätskern) |
| Audio-Nachbearbeitung | 91 | JA — DSP-Nachbearbeitung | **sinnvoll** |
| Qualitätsprüfung | 93 | nein — Analytik | sinnvoll, billig (Gates) |
| Musical Goals geprüft | 96 | nein — Analytik | sinnvoll (GO/NO-GO-Gate) |
| Ergebnis finalisiert + Defekt-Countdown | 98 | nein — Analytik/Status | sinnvoll (GUI §GUI-T7) |
| End-Gate-Kaskade (nur letzter Chunk, 16727) | — | ja (Nachschärfung falls Gates reißen) | **sinnvoll**, aber nur EINMAL nötig |
| m1b Targeted Retry auf assembliertem Song (45465) | — | JA — zweiter Pipeline-Pass mit Retry-Phasen | **sinnvoll** (behebt hörbare Restdefekte, max. 1 Pass, §V7-konform) |

**Fazit Tail**: Kein Schritt ist „pure Zeitverschwendung“. Die Analytik-Schritte sind
billig und speisen die Gates; die hörbaren Schritte (FeedbackChain, m1b-Retry) sind die
Qualitätsdifferenzierung. **Kontraproduktiv ist nicht der Tail, sondern dass er im
Chunked-Pfad auf 30-s-Fenstern statt auf dem Song läuft** (nur m1b-Retry + SCORECARD
laufen song-global).

## 4. Frage 3: Gibt es eine alternative Parallelversion? — JA, und sie ist jetzt ENTFERNT

**Befund (belegt)**:
1. **ARE-Legacy-Pfad** in `denker/restaurier_denker.py`: Bei fehlenden Caches lief
   `AurikAutonomousPipeline.process()` = **Multi-Varianten-Restaurierung**
   (`autonomous_restoration_engine.py`: `passes_executed=len(variants)`,
   `winning_variant`, `variant_scores`) und **danach ein ZWEITER voller UV3-Pass**
   auf dem ARE-Audio (`restorer.restore(_are_audio, ...)`) — zwei konkurrierende
   Versionen hintereinander, von denen nur die zweite überlebt.
2. **Direkt-UV3-Pfad** (§v10.5, Zeile ~809): Produktion (GUI via Denker) übergibt
   immer Caches → läuft bereits direkt; der Legacy-Pfad war nur noch für Cache-lose
   Aufrufer aktiv und laut eigenem Kommentar „deprecated“.
3. **Sweet-Spot-/Retry-Schleifen** (`_optimize_to_sweet_spot`, `_retry_lighter`):
   KEINE Parallelversion, sondern bedingte Rettungsversuche nach Gate-Versagen —
   bleiben (max. 3 Iterationen, deterministisch).
4. **Source-Aware-Restore** (Demucs → Per-Stem-UV3 → Remix, `source_aware_restorer.py`):
   Env-Opt-in (`AURIK_SOURCE_SEPARATION=1`), Standard AUS — kein Lauf „nach“ der
   Restaurierung, sondern ein alternativer Modus. Bleibt opt-in (Tests hängen daran).

**Umgesetzt (dieser Commit)**: ARE-Legacy-Block, `_get_are_pipeline`/`_build_are_pipeline`,
der `AurikAutonomousPipeline`-Import und das `_pipeline`-Attribut wurden aus
`denker/restaurier_denker.py` **komplett gelöscht**. Es existiert jetzt genau EINE
Restaurierungsversion: der direkte UV3-Pfad. 285/285 Denker-Tests grün.

## 5. Bewertung des Gesamt-Ablaufs (Verdikt)

| Abschnitt | Urteil |
|---|---|
| Import → Analysen (Caches) | sinnvoll (Wiederverwendung statt Doppelarbeit) |
| Reparatur-/RekonstruktionsDenker | sinnvoll (Gap-/Bandbreiten-Vorarbeit für UV3) |
| Pipeline je Chunk (30/60 s) | **fundamental falsch skaliert** — Pipeline gehört auf den Song |
| Phasen selbst | sinnvoll (deterministisch, guards) |
| Tail 86–98 + Gates + m1b-Retry | sinnvoll — KEINE Zeitverschwendung; Retry ist die Qualitätsdifferenzierung |
| ARE-Parallelversion | **kontraproduktiv** — ENTFERNT |
| Export | sinnvoll |

## 6. Umsetzungsplan Ganz-Song-Refactor (nächste Sessions, mit Referenzlauf)

1. `_restore_chunked`: neuen Modus einführen, in dem `chunks == [ganzer Song]` ist
   (Pipeline läuft EINMAL auf dem Song); Chunked-Progress-Mapping entfällt dort.
2. Separation bleibt chunkweise: BS-RoFormer/Demucs rufen intern Chunked-Inferenz
   (Overlap-Add) auf — keine Qualitätseinbußen, Modell-Limit bleibt respektiert.
3. Tail/Gates/m1b laufen dann ohnehin song-global — Chunk-Sonderlogik
   (`_chunked_tail_skip`, `_chunked_last`, B3-State-Freeze) kann schrittweise abgebaut werden.
4. Verifikation: **Referenzlauf (224 s)** — Bit-Determinismus-Referenz neu erzeugen,
   ≤ 40 min Budget, `n_audible → 0`, RAM-Messung (Ziel < 8 GB).
5. Rollback-Plan: der Modus bleibt feature-flagged (Env/Kwarg), bis der Referenzlauf grün ist.

**Bewusst NICHT umgesetzt wurde**: das Separationsmodell auf Ganz-Song umzubauen
(internes Chunking existiert und ist qualitätsgleich) und der Ganz-Song-Refactor selbst —
beides verlangt den neuen Referenzlauf, der Nutzer-Audio voraussetzt (§V7: keine
ungeprüften Strukturänderungen am Qualitätspfad).
