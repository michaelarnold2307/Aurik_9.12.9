# Spec 19: SOTA Vocal Gender Detection — §2.8 Perfection

> **Version:** Aurik 10.0.0 · **Scope:** Gender-Erkennung für De-Esser, Vocal-Enhancement, Formant-Preservation
> **Status:** Implementiert · **Audit-Datum:** 2026-08-04
> **Behebt:** 5 strukturelle Bugs in der Gender-Detection-Chain

## Inhaltsverzeichnis

1. [§19.1 Root-Cause-Analyse](#191-root-cause-analyse)
2. [§19.2 SOTA-Architektur](#192-sota-architektur)
3. [§19.3 Änderungen im Detail](#193-änderungen-im-detail)
4. [§19.4 Invarianten & Contracts](#194-invarianten--contracts)
5. [§19.5 Test-Strategie](#195-test-strategie)

---

## §19.1 Root-Cause-Analyse

### Bug 1: `classify_gender_via_formants` existierte nicht (`lpc_formant_tracker.py`)

**Symptom:** `phase_19_de_esser._detect_gender_robust()` (ursprüngliche Version, später
überschrieben) rief `get_lpc_formant_tracker().classify_gender_via_formants()` auf —
eine Methode, die in `_LPCFormantTracker` nicht definiert war.

**Root Cause:** Die Methode war in Phase 19 referenziert, aber nie im LPC-Tracker
implementiert. Der `except Exception: return "female"`-Fallback verschleierte den Bug
komplett — jede Stimme wurde als weiblich klassifiziert.

**Fix:** `classify_gender_via_formants(audio, sr) → str` implementiert mit:

- `_scan_f0_voiced()` — scanned durch Audio in 100ms-Fenstern
- `_estimate_formants_from_voiced()` — Burg-LPC Formanten (F1–F3) aus voiced Frames
- Gleiche `_GENDER_RANGES` wie `GenderDetector.formant_ranges`

### Bug 2: `GenderDetector._detect_f0()` prüfte nur erste 100ms (`vocal_ai_enhancement.py`)

**Symptom:** Bei Tracks mit instrumentalem Intro (≥100ms ohne Gesang) lieferte
`_detect_f0()` 0.0 Hz. Folge: `_classify_gender()` gibt UNKNOWN zurück, die gesamte
`GenderDetector.detect()`-Kette scheitert noch bevor sie beginnt.

**Root Cause:** `max_samples = min(len(audio), int(self.sr * 0.1))` → `segment = audio[:max_samples]`
betrachtet ausschließlich den Track-Anfang. Die ursprüngliche Begründung "prevents O(N²)"
ignorierte, dass ein nicht-voicedes Intro die Detektion komplett blockiert.

**Fix:** Scanning-Ansatz: 60×100ms Fenster mit 50ms Hop. Jedes Fenster wird auf RMS und
F0 geprüft. Das Fenster mit dem höchsten Autokorrelations-Peak gewinnt. Kosten:
O(chunks × N log N) ≈ O(60 × 4800 log 4800) ≈ 3.4M OPs — vernachlässigbar.

### Bug 3: Dead-Code-Katastrophe in `phase_19_de_esser.py`

**Symptom:** Vier Stub-Methoden (`_detect_gender_robust` v1, `_detect_gender_timeline` v1,
`_process_per_gender_segments` v1, `_apply_formant_preservation` v1) waren als Dead Code
vorhanden und wurden von den echten Implementierungen überschrieben.

**Schwerwiegender:** Die echten Methoden (560 Zeilen) waren in `_build_union_vocal_profile()`
gefangen — NACH einem `return`-Statement. Sie wurden als lokale, unerreichbare Funktionen
definiert und nie ausgeführt.

**Root Cause:** Indentationsfehler. `_build_union_vocal_profile` (Modul-Level, indent=0)
endete nicht nach seinem `return {…}`, sondern absorbierte alle folgenden indent=4-Methoden
als Funktionskörper. Der `return` verhinderte die Ausführung, aber nicht die Definition.

**Fix:**

1. 4 Stub-Methoden entfernt
2. 560 Zeilen echter Methoden aus `_build_union_vocal_profile` extrahiert
3. Methoden korrekt in `DeEsserPhase`-Klasse platziert

### Bug 4: `_detect_gender_simple()` nahm nur erste 5 Sekunden

**Symptom:** Gleiches Intro-Problem wie Bug 2. Der Ultimate-Fallback für die
Gender-Erkennung war genauso blind für Intros.

**Root Cause:** `max_samples = sample_rate * 5` → `audio = audio[:max_samples]`.
5s Autokorrelation auf 240k Samples ist zudem teuer (~240k² Op ohne FFT).

**Fix:** Scanning-Ansatz: 12×2s Fenster mit 1s Hop. FFT-basierte Autokorrelation
pro Fenster. Robust und schnell.

### Bug 5: Kein LPC-Fallback in der lebenden `_detect_gender_robust`

**Symptom:** Wenn `GenderDetector` (vocal_ai_enhancement) fehlschlug ODER `_HAS_ROBUST_GENDER`
False war, fiel die Kette direkt auf `_detect_gender_simple` zurück — ohne den
LPC-Formant-Tracker als zweite Meinung zu konsultieren.

**Fix:** Dreistufige Fallback-Kette:

1. GenderDetector + pYIN + Contralto
2. LPC Formant Tracker (Burg-LPC + scanning F0) ← **NEU**
3. `_detect_gender_simple` (scanning)

---

## §19.2 SOTA-Architektur

```
Phase 19 DeEsserPhase.process()
  └─ _detect_gender_robust(audio, sr)           ← SOTA-Hauptdetektor
       ├─ 1. GenderDetector (vocal_ai_enhancement)
       │      ├─ _detect_f0() → scanning 60×100ms ← Bug 2 fix
       │      ├─ _detect_formants() → spectral peaks + WORLD
       │      ├─ _classify_gender(F0, formants)
       │      └─ pYIN F0 (librosa) → voiced-frame median
       │      └─ Contralto-Erkennung (F0=male, Formanten=female)
       │
       ├─ 2. LPC Formant Tracker (lpc_formant_tracker) ← Bug 1 + 5 fix
       │      ├─ _scan_f0_voiced() → scanning 60×100ms
       │      ├─ _estimate_formants_from_voiced() → Burg-LPC 40 frames
       │      └─ classify_gender_via_formants(F0, F1–F3)
       │
       └─ 3. _detect_gender_simple() → scanning 12×2s ← Bug 4 fix

Phase 19 DeEsserPhase.process()
  └─ _detect_gender_timeline(audio, sr)         ← Echt-Implementierung
       ├─ pYIN F0 pro Frame
       ├─ Voiced-Segment-Extraktion
       ├─ Pro-Segment: F0-Median, Vibrato, Gender
       └─ Merge benachbarter gleicher Gender

Pipeline (unified_restorer_v3):
  └─ _select_phases()
       └─ GenderDetector(sr).detect(audio_mono)  ← Bug 2 fix wirkt hier
            └─ self._detected_vocal_gender → restoration_context["vocal_gender"]
```

### Gender-Bereiche (zentral, identisch in allen Detektoren)

| Merkmal | Male | Female | Child |
|---------|------|--------|-------|
| **F0** | 85–180 Hz | 165–700 Hz | 250–600 Hz |
| **F1** | 270–730 Hz | 310–860 Hz | 370–1030 Hz |
| **F2** | 840–2290 Hz | 920–2790 Hz | 1170–3330 Hz |
| **F3** | 1690–3010 Hz | 1890–3310 Hz | 2590–4990 Hz |

Quellen: Titze 1994 (singing ranges), Klatt & Klatt 1990 (speech formants).

### Tie-Breaking-Regeln

1. **CHILD vs FEMALE**: Wenn Score-Delta < 0.05 UND F0 < 350 Hz → FEMALE
   (Kinderstimmen mit F0 < 350 Hz in professionellen Aufnahmen extrem selten)
2. **Contralto**: Wenn GenderDetector "male" sagt, ABER F0 140–220 Hz UND
   F1+F2 im weiblichen Bereich → Override auf FEMALE

### Contralto-Zonen-Erweiterung (2026-08-22)

Die Contralto-Zone ist in `phase_19_de_esser.py` auf **120–240 Hz** erweitert
(§13.7 nennt 145–195 Hz, §19 nennt 140–220 Hz als Kern), um **Oktavfehler der
F0-Detektion** abzudecken (94 Hz gemessen statt 188 Hz bei sehr tiefen
Frauenstimmen). Regeln:

- **Oktavkandidat**: F0 < 120 Hz UND 2×F0 in [120, 240] → effektives F0 = 2×F0.
- **Notfall-Regel §v10.303.11** (F1 > 300 Hz + F0 < 120 Hz → Override) greift
  **nur bei degradiertem F2** (F2-Messung fehlt < 50 Hz oder
  `bandwidth_loss > 0.5`). Eine **gesunde** F2-Messung unterhalb des weiblichen
  Bereichs (z. B. 719 Hz = männliches /u/-Profil) ist Evidenz für MALE und wird
  nicht überstimmt (Befund 2026-08-22: Override trotz F2=719 Hz + Meldung
  behauptete unmöglich „F2=719 Hz in [920–2790]“).
- **Confidence**: Der Override setzt confidence auf **0.65** (Contralto-Floor)
  und erbt nicht die Confidence des widersprochenen 'male'-Urteils.

### §19.2b Umsetzung im Pipeline-Level GenderDetector — Befund Elke Best (2026-09-08)

**Symptom:** Gender wird bei Elke Best (Mezzosopran/Alt, volles Pop-Arrangement)
als `UNKNOWN` bzw. `MALE` erkannt — obwohl die älteren Logs (`logs/elke_best_*.log`)
„Auto-detected gender: female" zeigten. Die §19.2-Architektur war bis dahin nur in
`phase_19_de_esser._detect_gender_robust` umgesetzt — NICHT im Pipeline-Level
Detektor `vocal_ai_enhancement.GenderDetector`, den `unified_restorer_v3`
(`_select_phases`, Zeile ~27504) für `restoration_context["vocal_gender"]` nutzt.

**Root Cause (gemessen an `test_audio/_elke_60s_excerpt.wav`):**

1. `_detect_f0()` (Scanning) betrachtet nur die ersten ~3 s (60×100 ms) → greift
   die Basslinie (111 Hz) statt der Stimme; bei instrumentalem Intro > 3 s → 0 Hz → UNKNOWN.
2. `_detect_formants()` mittelte Spektral-Peaks über ALLE Frames (inkl. Instrumente)
   → F2=704 Hz statt weiblich (>920 Hz).
3. Kein pYIN, kein Contralto-Override im Detektor → F0=111 Hz + kontaminierte
   Formanten → MALE (confidence 0.93).

**Umsetzung (deterministisch, §G5):**

- **`_detect_pyin_f0()`**: pYIN (Mauch & Dixon 2014) mit gestufter Schwelle
  (voiced_prob > 0.4 → 0.25; gemessen: bei vollem Pop-Arrangement liegt p90 ≈ 0.2,
  max ≈ 0.85 — die frühere Schwelle 0.8 fand nichts), NaN-Filter, Median über
  voiced Frames, Fenster-Leiter 0–30 s → Track-Mitte → Track-Ende (I-19.2).
- **pYIN-Override** in `detect()`: pYIN gewinnt bei F0=0 oder Abweichung > 15 %
  (Oktavfehler, Bass-Masking, Vibrato) — Regel identisch mit phase_19 (§2.11).
- **`_detect_formants(audio, voiced_times)`**: Formanten NUR aus Frames mit
  stimmhaftem Zentrum (Gate ±12 ms); leeres Gate → ungegateter Fallback.
- **`_apply_contralto_override()`**: Spec-19-Contralto-Regel (Zone 120–240 Hz
  inkl. Oktavkandidat 2×F0, F1 UND F2 weiblich → FEMALE, Confidence-Floor 0.65).

**Evidenz:** `_elke_60s_excerpt.wav`: vorher F0=111.1 Hz, Formanten [325, 704, 1083]
→ MALE 0.93; nachher pYIN-F0=323.2 Hz, voiced-Formanten [353, 741, 1117] → FEMALE 0.92.
`Elke Best - 30 Sekunden.mp3`: 94.5 Hz → 358.6 Hz → FEMALE 0.92. Regressionstests G09a–G09d
in `tests/normative/test_gender_detection_sota_gate.py` (G09a echt, Modell-unabhängig,
skip ohne Testaudio).

---

## §19.3 Änderungen im Detail

### Datei 1: `backend/core/vocal_ai_enhancement.py`

| Zeilen | Änderung |
|--------|----------|
| 202–250 | `_detect_f0()`: Von `audio[:100ms]` auf Scanning (60×100ms Fenster, 50ms Hop). RMS-Gate, F0-Plausibilität 70–800 Hz, stärkster Peak gewinnt. |

### Datei 2: `backend/core/dsp/lpc_formant_tracker.py`

| Zeilen | Änderung |
|--------|----------|
| 568–594 | `_GENDER_RANGES`: Klassenkonstante mit F0/F1/F2/F3-Bereichen (identisch mit `GenderDetector`) |
| 596–643 | `_scan_f0_voiced()`: Static method. Scannt 60×100ms Fenster, FFT-Autokorrelation, bester Peak gewinnt |
| 645–692 | `_estimate_formants_from_voiced()`: Static method. 40 Frames via Burg-LPC @ 16kHz, Downsampling+AA-Filter, Median-Mittelung |
| 694–768 | `classify_gender_via_formants(audio, sr) → str`: Hauptmethode. Mono-Konversion, F0+Formanten-Extraktion, Scoring, Tie-Breaking |

### Datei 3: `backend/core/phases/phase_19_de_esser.py`

| Zeilen | Änderung |
|--------|----------|
| 2469 (gelöscht) | Dead `_detect_gender_robust` v1 (überschrieben) |
| 2483–2493 (gelöscht) | 3 Stub-Methoden (`_detect_gender_timeline`, `_process_per_gender_segments`, `_apply_formant_preservation`) |
| ~2500–2700 (verschoben) | 560 Zeilen Methoden aus `_build_union_vocal_profile`-Gefängnis befreit → jetzt korrekt in `DeEsserPhase` |
| 2949–2965 | LPC-Fallback in `_detect_gender_robust`-Kette: nach GenderDetector-Failure → `classify_gender_via_formants` |
| 2968–3038 | `_detect_gender_simple()`: Von `audio[:5s]` auf Scanning (12×2s Fenster, 1s Hop) |

---

## §19.4 Invarianten & Contracts

### I-19.1: classify_gender_via_formants existiert

`get_lpc_formant_tracker().classify_gender_via_formants(audio, sr)` MUSS ohne
`AttributeError` aufrufbar sein und einen der Strings `"male"`, `"female"`,
`"child"`, `"unknown"` zurückgeben.

### I-19.2: Scanning-F0 überlebt Intros

`GenderDetector._detect_f0(audio_with_intro)` MUSS einen F0-Wert > 0 liefern,
wenn IRGENDWO im Audio ein voiced Segment existiert. Getestet mit 1.5s Stille
vor einem 220Hz-Ton.

### I-19.3: Alle 5 Methoden auf DeEsserPhase

`DeEsserPhase` MUSS folgende Methoden besitzen:

- `_detect_gender_robust`
- `_detect_gender_simple`
- `_detect_gender_timeline`
- `_process_per_gender_segments`
- `_apply_formant_preservation`

### I-19.4: Keine toten Stubs

Keine der o.g. Methoden darf `return []` oder `return audio` als einzige
Implementierung haben (ausser wenn sinnvoll — z.B. `_apply_formant_preservation`
darf `return processed` als Fallback, aber nicht als permanente Stub-Implementierung).

### I-19.5: _build_union_vocal_profile ist clean

`_build_union_vocal_profile` MUSS nach seinem `return {…}` enden und darf KEINE
nachfolgenden Methodendefinitionen absorbieren.

### I-19.6: Dreistufige Fallback-Kette

`_detect_gender_robust` MUSS nach GenderDetector-Failure den LPC-Formant-Tracker
konsultieren, bevor es auf `_detect_gender_simple` zurückfällt.

---

## §19.5 Test-Strategie

### Normative Tests (`tests/normative/test_gender_detection_sota_gate.py`)

| Test | Invariante | Was geprüft wird |
|------|-----------|-----------------|
| `test_lpc_classify_gender_exists` | I-19.1 | `classify_gender_via_formants` aufrufbar, gibt validen String zurück |
| `test_lpc_classify_gender_synthetic` | I-19.1 | Synthetische Töne (120Hz+500Hz+1500Hz → male, 220Hz+700Hz+2000Hz → female) |
| `test_lpc_classify_gender_with_intro` | I-19.2 | 1.5s Stille + Ton → erkennt Gender trotz Intro |
| `test_gender_detector_f0_scanning` | I-19.2 | `_detect_f0` mit Intro liefert korrekten F0 |
| `test_deesser_phase_methods_present` | I-19.3 | Alle 5 Methoden auf `DeEsserPhase` |
| `test_deesser_no_dead_stubs` | I-19.4 | Keine `return []`- oder `return audio`-Stubs |
| `test_build_union_vocal_profile_clean` | I-19.5 | `_build_union_vocal_profile` enthält keine `def` nach `return` |
| `test_robust_fallback_has_lpc` | I-19.6 | `_detect_gender_robust` Quellcode enthält LPC-Fallback |
| `test_gender_detector_detect_chain` | I-19.2 | `GenderDetector.detect()` vollständige Kette mit Intro |
| `test_deesser_gender_simple_scanning` | I-19.2 | `_detect_gender_simple` erkennt Gender mit Intro |

### Integrationstest (`tests/integration/test_gender_detection_integration.py`)

- E2E: `DeEsserPhase.process()` mit synthetischem Audio → Metadata enthält korrektes Gender
- E2E: Gender-Detection via unified_restorer_v3 `_select_phases` → `_detected_vocal_gender` gesetzt
- Cross-detector Konsistenz: GenderDetector vs LPC vs Simple müssen bei klarem Signal übereinstimmen
