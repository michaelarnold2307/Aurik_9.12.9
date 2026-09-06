> **⚠️ ARCHIVIERT — siehe `docs/architecture/ARCHITECTURE.md` und `.github/specs/` (Index: `00_SPEC_INDEX.md`) für die aktuelle Architektur (Version 10.0.20).**
>
> Diese Datei beschreibt die historische Aurik-9.x-Architektur und ist nicht mehr aktuell.

# Aurik-9-Ära — Implementierte Architektur (ARCHIVIERT)

**Stand:** 2026-09-06 (archiviert)  
**Version:** — (archiviert; aktuell: 10.0.20)  
**Status:** ⚠️ ARCHIVIERT (nicht mehr aktualisiert)

> Hinweis: Diese Seite ist eine historische Architekturübersicht. Normative Details stehen in der
> normativen Kette (`AGENTS.md` §1) mit `.github/specs/` (Index: `00_SPEC_INDEX.md`).

---

## Design-Prinzipien

### 1. Defect-First, Cognitive Pipeline

Aurik 10.0.0 verarbeitet Audio über eine streng geordnete kognitive Pipeline.
Vor jeder Restaurierung wird das Material analysiert, die Ursache ermittelt
(Bayesianische Kausalinferenz) und die Verarbeitungsparameter optimiert (GP-Optimizer).

```text
TransientDecoupledProcessing (TDP)           ← Schritt 0: Trennung
  -> RestorabilityEstimator                   ← < 5 s Vor-Assessment
  -> EraClassifier (1890–2025)                ← Dekaden-Prior
  -> GermanSchlagerClassifier                 ← Zero-Shot Genre
    -> MediumDetector (transfer-chain-aware)    ← Träger-Erkennung
    -> DefectScanner (62 DefectTypes)           ← Defekt-Erkennung
    -> CausalDefectReasoner (62 Kausal-Ursachen)← Ursachen-Inferenz
  -> UncertaintyQuantifier                    ← Konfidenz
  -> GPParameterOptimizer (MOO-Pareto)        ← Parameter-Vorschlag
  -> HarmonicPreservationGuard                ← Partial-Masken
  -> Phasen 01–68 (je via PMGG-Gate)         ← Verarbeitung
  -> EraAuthenticPerceptualCompletion         ← BW < 10 kHz
  -> IntroducedArtifactDetector               ← Artefakt-Check
  -> FeedbackChain (max. 5 Iter.)             ← Iterative Optimierung
  -> TemporalQualityCoherenceMetric           ← Zeitliche Konsistenz
  -> PerceptualQualityScorer                  ← PQS-MOS
  -> ExcellenceOptimizer                      ← GP-Fine-Tuning
  -> MusicalGoalsChecker (14 Ziele)           ← Qualitäts-Gate
  -> EmotionalArcPreservationMetric           ← Emotionaler Bogen
  -> MicroDynamicsEnvelopeMorphing            ← Mikro-Dynamik
  -> GPParameterOptimizer.update()            ← Persistenz
  -> RestorationResult                        ← Ausgabe
```

### 2. 14 Musical Goals als Wahrheitskriterium

Kein technischer Score (SNR, THD) — ausschließlich psychoakustische Musical Goals
entscheiden über Qualität. Alle 14 Ziele müssen nach jeder Restaurierung erfüllt sein.

GoalApplicabilityFilter deaktiviert physikalisch irrelevante Ziele (z. B. SpatialDepth
bei Mono-Aufnahmen). GoalPriorityProtocol (Stufe 1–5) steuert Pareto-Kompromisse.

### 3. Numerische Robustheit (Pflicht)

Jede Funktion die Audio oder Scores zurückgibt ist NaN/Inf-frei:

```python
result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
audio = np.clip(audio, -1.0, 1.0)
if not math.isfinite(score): return  # Score-Update überspringen
```

### 4. Singleton + Thread-Safety (Pflicht)

```python
_instance = None
_lock = threading.Lock()

def get_my_module():
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MyModule()
    return _instance
```

### 5. SR-Invariante (Pflicht)

Interne Verarbeitung immer auf 48 000 Hz:

```python
assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
```

---

## Wichtige neue Module (v10.0.0–v10.0.0)

| Modul | Zweck | Position in Pipeline |
| --- | --- | --- |
| `TransientDecoupledProcessing` | HPSS vor NR — Groove-Erhalt | Allererster Schritt |
| `HarmonicPreservationGuard` | G_floor=0.85 an Partials | Vor phase_03 + phase_29 |
| `PerPhaseMusicalGoalsGate` | Rollback bei Regression nach jeder Phase | Umhüllt alle Phasen |
| `MicroDynamicsEnvelopeMorphing` | LUFS-Profil-Korrektur 400 ms | Letzter Schritt vor Export |
| `StemRemixBalancer` | LUFS-korrekter Re-Mix nach Stem-Separation | Studio 2026 |
| `RestorabilityEstimator` | < 5 s Vor-Assessment, Score 0–100 | Vor Verarbeitung |
| `RemasterDetector` | Erkennt bereits gemasterte Quellen | Teil von EraClassifier |
| `EraAuthenticPerceptualCompletion` | DDSP-Synthese fehlender Partials | Nach phase_56 |
| `MusikalischerGlobalplanDienst` | Cross-Phase-Globalplan, 13 Ära-Profile | AurikDenker Stufe 4 |
| `LyricsGuidedEnhancement` | Phonem-Alignment + Phonemklassen-DSP (§2.36) | Phase 58 |
| `PhonemeTimeline` | Segment-Timeline: vowel\_stressed/fricative/plosive/silence | Phase 58 |
| `GenreClassifier` | Genre-Phase-1: Family+Top-k+Open-Set + SongCal-Fusion | Nach EraClassifier |
| `RecoveryCheckpoint` | OOM-Recovery-Checkpoint (atomisch, 7 Tage TTL) | §2.39, MemoryError-Handler |
| `PerceptualSalienceEstimator` | Psychoakust. Salienz-Annotation je Defekt (§9.1c) | DefectScanner-Post |

---

## Softwareschichten-Architektur

```text
┌──────────────────────────────────────────┐
│  Frontend (frontend/)   PyQt5 Dark Theme │
├──────────────────────────────────────────┤
│  CLI        aurik_cli.py                 │
├──────────────────────────────────────────┤
│  API-Schicht  backend/api/rest/          │
├──────────────────────────────────────────┤
│  Backend-Core  core/ · plugins/ · dsp/  │
└──────────────────────────────────────────┘
```

Keine Direktverbindungen zwischen nicht-benachbarten Schichten.
Frontend kommuniziert ausschließlich über API-Schicht oder Qt-Signals/Slots.

---

## Kognitive Orchestrierungsschicht (denker/)

`denker/` enthält 10 Sub-Denker-Module die alle Kernmodule orchestrieren:

| Modul | Zweck |
| --- | --- |
| `aurik_denker.py` | Haupt-Orchestrator |
| `tontraeger_denker.py` | MediumDetector (MediumClassifier nur Legacy-Kompat) |
| `tontraegerkette_denker.py` | Tonträgerketten-Erkennung (§6.7) |
| `defekt_denker.py` | DefectScanner + CausalDefectReasoner |
| `strategie_denker.py` | PerformanceGuard, RT-Limit |
| `restaurier_denker.py` | UnifiedRestorerV3 |
| `reparatur_denker.py` | scipy-Direktreparatur |
| `rekonstruktions_denker.py` | GapReconstructor / Inpainting |
| `exzellenz_denker.py` | 14 Musical Goals + ExcellenceOptimizer |

---

**Aurik 10.0.8 — Juli 2026**
