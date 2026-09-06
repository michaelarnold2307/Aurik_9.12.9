# 🎵 Aurik 10 — Intelligentes Musik-Restaurierungs- und Rekonstruktionssystem

**Version:** 10.0.20 | **Status:** ✅ Weltspitze-Execution | **Stand:** v10.0.20 — Era-/Material-Kalibrierung + Hör-Gates Ebenen 1/2/4

> Normativer Ist-Stand: `.github/specs/`, `.github/copilot-instructions.md`, `CHANGELOG.md`, `denker/README.md`.

![Tests](https://img.shields.io/badge/tests-285%2B%20Denker%20%2B%2018.400%2B%20gesamt-brightgreen)
![DefectTypes](https://img.shields.io/badge/Defekttypen-62%2F62%20erkannt%20%26%20gemappt-brightgreen)
![Materials](https://img.shields.io/badge/Materialien-16%20Typen-blue)
![Genres](https://img.shields.io/badge/Genres-19%20Profile-blue)
![Denker](https://img.shields.io/badge/Denker-Intelligenz-Material%20%2B%20Vocal%20adaptiv-orange)
![PostProc](https://img.shields.io/badge/Post--Processing-8%20Stufen%20wissenschaftlich-orange)
![Hardware](https://img.shields.io/badge/Hardware-CPU%20%2B%20AMD--GPU%20optional-orange)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

---

## 🎯 Was ist Aurik 10?

Aurik 10 ist ein **intelligentes, kontextbewusstes Musik- und
Gesangs-Restaurations-, Reparatur- und Rekonstruktions-Denkersystem**.

Es kombiniert psychoakustisch fundierte DSP, Bayesianische Kausalinferenz,
Gaussianische Prozess-Optimierung und perceptuelle Qualitätsbewertung zu einer
kognitiven Restaurierungs-Intelligenz — für Desktop (Linux & Windows 10/11),
vollständig offline, ohne Cloud- oder Netzwerkabhängigkeiten.

**Produktvertrag (Release-Must):**

- Desktop-only: Linux AppImage, Windows 10/11 Installer
- 100 % offline nach Installation
- Endnutzer-Workflow: genau eine Entscheidung pro Datei, `Restoration` oder `Studio 2026`
- Kanonischer Laufzeitpfad: Bridge -> `AurikDenker.denke(...)` -> `export_guard()`

**Aktuelle Ergebnisse (v10.0.8):**

- ✅ **285+ Denker-Tests** — grün (zzgl. ~15.000 weitere Test-Suites)
- ✅ **62/62 Defekttypen** — vollständig erkannt und auf Phasen gemappt
- ✅ **16 Materialien** — auto-erkannt mit Transfer-Chain-Analyse
- ✅ **19 Genre-Profile** — inkl. Reggae, Latin, Gospel, Country, Funk, Ambient, World
- ✅ **17 SourceMediums** — Wax Cylinder, Wire Recording, Lacquer Disc u.v.m.
- ✅ **Denker-Intelligenz** — Material-adaptive Phasen-Erzwingung + Mindest-Stärke
- ✅ **8-stufige Post-Processing-Pipeline** — wissenschaftliche Restaurierungs-Reihenfolge
- ✅ **ML-Hybrid-Export** — DeepFilterNet, Demucs, AudioSR (RAM-abhängig)
- ✅ **MAD-Dropout-Repair** — statistische Ausreißer, keine False-Positives
- ✅ **CPU + optionale AMD-GPU-Beschleunigung** — keine GPU-Pflicht

**Hinweis zur Evidenz:** Interne Qualitätsangaben wie PQS-MOS und OQS dienen als
technische Steuerungs- und Freigabemetriken. Externe Superlative wie
"weltbeste" oder formale Hörtest-Äquivalenz werden erst durch unabhängige,
verblindete Hörtests und reproduzierbare Wettbewerbsvergleiche belastbar.

**Über-SOTA DSP-Algorithmen (v9.x.x — vollständig implementiert):**

| Phase | Legacy (verboten) | Über-SOTA (aktiv) | Referenz |
| --- | --- | --- | --- |
| Phase 03 Denoise | ~~Wiener 1984~~ | **OMLSA + IMCRA** + HarmonicPreservationGuard | Cohen 2002/2003 |
| Phase 09 Crackle | ~~Medianfilter~~ | **RBME + Sparse Bayes** | Cemgil 2006, Bando 2019 |
| Phase 12 Wow/Flutter | ~~YIN~~ | **pYIN probabilistisch** + DTW | Mauch & Dixon 2014 |
| Phase 24 Dropout | ~~AR-Spline~~ | **CQTdiff+ / NMF-β + PGHI** | Moliner 2023, Févotte 2011 |
| Phase 55 Inpainting | ~~Griffin-Lim~~ | **Flow Matching / DiffWave** | Lipman 2023, Bai 2024 |
| Phase 56 BandGap | — | **SpectralBandGapRepair** (HEAD_WEAR) | Roebel 2010 |
| Phase 57 Limiter | — | **DAW-Limiter-Erkennung + Dekompressor** | Giannoulis 2012 |
| Phase 58 Hallucination | — | **§2.46e HallucinationGuard + IntroducedArtifactDetector** | Aurik intern |
| Phase 66 Comfort | — | **Human-Hearing-Comfort-Policy** (zentrale Hörkomfort-Steuerung) | Aurik §2.44 |

## 🧠 Denker-Intelligenz — Autonome Defekt→Reparatur-Entscheidungskette

**Stand: 6. Juli 2026** | **Dateien: `denker/phase_interaction_denker.py`, `backend/core/vocal_no_harm_gate.py`**

### Entscheidungskette pro Defekt

```
DefectScanner (62 Typen)
  → DefectPhaseMapper (Primary + Secondary Phases)
  → PhaseInteractionDenker (Material-Kritische-Phasen-Injektion §2.5a)
  → VocalNoHarmGate (Material-adaptive PANNS-Schwelle)
  → _profiled_phase_call (Material-adaptive Mindest-Stärke 30-40%)
  → Post-Processing (8-stufige wissenschaftliche Pipeline)
  → EXPORT
```

### Material-adaptive Parameter

| Material | PANNS-Schwelle | Mindest-Stärke | Erzwungene Phasen |
| --- | --- | --- | --- |
| cassette/tape | **0.55** (+57%) | **40%** | phase_14, 25, 56, 24 |
| reel_tape | **0.50** (+43%) | **40%** | phase_14, 25, 56 |
| vinyl/shellac | **0.45** (+29%) | **35%** | phase_09, 28 |
| digital/andere | 0.35 (default) | 30% | — |

### 8-stufige wissenschaftliche Post-Processing-Pipeline

| Stufe | Kategorie | Module |
| --- | --- | --- |
| 1 | Breitband | (UV3: Hum, Rumpel, DC) |
| 2 | Impulsiv | PrecisionDropout, VocalScratch, TapeHead |
| 3 | Rauschen | (UV3: phase_03, 29) |
| 4 | Spektral | AntiMuffling |
| 5 | Räumlich | SmartTapeRepair (Azimuth), EchoRemoval |
| 6 | Dynamik | (UV3: phase_10, 26, 54) |
| 7 | Enhancement | SibilanceMax, VocalClarity, Specialized |
| 8 | Ausgabe | Humanization, PerceptualOptimizer (ML-Hybrid), Listening-EQ |

### Prinzip

**Niemals einen erkannten Defekt unbehandelt lassen.** Wenn ein Defekt erkannt wird, MUSS mindestens eine Reparatur-Phase laufen. Material-Confidence beeinflusst die Stärke, nicht die Selektion.

---

## 🚀 Aurik 10.0.8 — Weltklasse-Intelligenz

**Stand: 4. Juli 2026** | **38 Dateien modifiziert, 14 neue Dateien** | **358+ Tests**

### Neue psychoakustische Modelle

| Modell | Standard | Zweck |
|--------|----------|-------|
| **ATH** | ISO 226:2023 | Absolute Hörschwelle — Defekte unterhalb der Hörbarkeit werden ignoriert |
| **Moore/Glasberg DLM** | Moore & Glasberg 2007 | Dynamisches Lautheitsmodell mit 40 ERB-Bändern |
| **BMLD** | Binaural Masking | Interaurale Kreuzkorrelation für räumliches Hören |
| **PEAQ** | ITU-R BS.1387 | Standardisierte perzeptuelle Audioqualitätsmetrik |
| **Forward Masking** | Fastl & Zwicker 2007 | Frequenzabhängiges zeitliches Masking (logarithmisch) |

### Neue Entscheidungsintelligenz

| Komponente | Funktion |
|------------|----------|
| **PIM** (Perceptual Intensity Mapper) | 10 Frequenzbänder × N Song-Sektionen → kalibrierte Intensitäts-Map |
| **RLP** (Reflective Listening Pass) | „Nochmal hinhören" — diagnostiziert Restprobleme und bessert nach |
| **Artistic Intent Modulator** | 12 Genres × 10 Epochen → konservativ/aggressiv-Strategie |
| **Glue Stage** | Finale subtile Bus-Kompression (1.2:1, <1.5dB GR) |
| **Stop-Regel** | PMGG-Δ < 0.01 über 3 Phasen → Pipeline stoppt |
| **Cross-Phase Awareness** | Phase B kennt Δ von Phase A |

### Neue Defekttypen (+8)

`MPEG_FRAME_LOSS`, `STEREO_FIELD_COLLAPSE`, `PHASE_ROTATION`, `DROPOUT_OXIDE`, `DROPOUT_HEAD_CONTACT`, `DROPOUT_SPLICE`, `ASYMMETRIC_CLIPPING`, `TRANSIENT_IMD`

### Vokal-Supremacy

- **Speaker Identity Guard**: ECAPA-TDNN (192-dim) + MFCC-Fallback (60/80-dim)
- **Vocal Overprocessing Detector**: Lisp-Erkennung, Formant-Drift, Sibilanz-Überreduktion
- **Vibrato-Guard**: Cross-Band-Coherence schützt Vibrato vor Flutter-Fehlklassifikation

### GUI/Laien-Verbesserungen

- `get_layman_summary()`: „Deine Musik erstrahlt in neuem Glanz!"-Kommunikation
- `get_pipeline_ab_snapshots()`: Base64-Audio für Vorher/Nachher-Player
- `--dry-run`, `--json`, `--abx`, `--progress`, `--resume` CLI-Flags
- ML-Modell-Status in der GUI sichtbar

### Export & Delivery

- **Bit-Perfect-Archiv-Pfad**: `export_bitperfect()` mit BWF-Metadaten
- **11 Playback-Profile** (inkl. Car-Sedan, SUV, Bluetooth-Speaker, Club-PA)
- **ISRC/UPC-Metadaten**, **Multi-Format-Export**
- **Continuous Learning**: UCB1 + State-Persistenz + Decay-Faktor 0.99

### §v10 Pleasantness-First: Jeder Song individuell (Juli 2026)

- **SNR-Adaptive Defekterkennung**: Kein blinder Material-Glaube — jeder Song wird gemessen.
  Click-Thresholds, Tape-Splice, alle 8 Detektoren passen sich dem gemessenen SNR an.
- **Spectrum-Aware EQ**: Final EQ (Phase 16) + Mastering Polish (Phase 17) messen
  das IST-Spektrum und skalieren Gains adaptiv. Kein Song bekommt dieselbe EQ.
- **Harmonic-Aware Saturation**: Phase 17 misst Even/Odd-Harmonic-Ratio —
  bereits gesättigte Songs bekommen weniger Enhancement.
- **No-Blind-Trust-Prinzip**: `_estimate_local_snr()`, `_measure_spectral_deviation()`,
  `_measure_harmonic_density()` — 3 neue Messfunktionen für individuelle Song-Optimierung.

### Behobene Bugs

| Bug | Fix |
|-----|-----|
| Binäres Gate (Lautstärkesprünge 3-18dB) | Soft-Knee-Sigmoid + 200ms Hanning-Crossfade |
| Hard-Clamp erzeugte Klicks | Entfernt, Soft-Knee schützt inhärent |
| `_multi_pass()` Dead Code | Reaktiviert mit IAQS-Varianten-Evaluation |
| 3 Silent ML-Fallbacks | Alle mit `logger.warning()` versehen |
| Bridge-Bypasses (CLI + Batch) | 2 Bypasses → Bridge-Funktionen |

**Kognitive Module (v9.x.x — 41 Kernmodule):**

| Modul | Zweck |
| --- | --- |
| `PerceptualEmbedder` | 256-dim psychoakustischer Einbettungsraum (L2-normalisiert) |
| `CausalDefectReasoner` | Bayesianische Kausalinferenz, **62 Kausal-Ursachen** |
| `GPParameterOptimizer` | RBF-GP + UCB + **MOO Pareto-Front** (14 Objectives) |
| `PerceptualQualityScorer` | Gammatone-NSIM + MCD + LUFS + MOS |
| `MusicalGoalsChecker` | **14 musikalische Qualitätsziele** |
| `MediumDetector` | File-ext-aware Tonträgerketten-Erkennung, autoritatives Materialsystem |
| `DefectScanner` | 56 DefectTypes, material-adaptive Material-Priors |
| `TransientDecoupledProcessing` | HPSS-Trennung — Groove-Schutz vor jeder NR |
| `HarmonicPreservationGuard` | CREPE/pYIN → G_floor 0.85 an Harmonik-Bins |
| `PerPhaseMusicalGoalsGate` | Rollback bei kumulativer Degradation (66 Phasen) |
| `EraClassifier` | Ära-Erkennung 1890–2025, GP-Warmstart pro Dekade |
| `GermanSchlagerClassifier` | Zero-Shot 6-Schicht-Ensemble (kein Schlager-Training nötig) |
| `ArtistSignatureStore` | Longitudinaler Klang-Fingerabdruck pro Künstler/Session |
| `MusicalStructureAnalyzer` | SSM-Novelty, Chorus als Inpainting-Referenz |
| `MusicalPhraseContextExtractor` | Beat-Tracking → Phrasen-Kontext für Dropout-Inpainting |
| `UnifiedRestorerV3` | **66-Phasen-Orchestrator** (Defect-First + Vocal-Supremacy §0p) |
| `FeedbackChain` | Iterative PQS-Qualitätsschleife, max. 5 Iter. |
| `ExcellenceOptimizer` | GP-Pareto-Optimierung, `ExcellenceResult` |
| `EnsembleProcessor` | 3 parallele Ketten (CONSERVATIVE/BALANCED/AGGRESSIVE) |
| `RestorabilityEstimator` | < 5 s Vor-Assessment, Predicted MOS + Score 0–100 |
| `UncertaintyQuantifier` | Konfidenz-Schwellen (0.80/0.50), GP-Rückhaltung |
| `TemporalQualityCoherenceMetric` | MOS-Spanne ≤ 0.30, σ ≤ 0.15 über Zeitachse |
| `AdaptiveGoalThresholds` | Material- und ära-adaptive Schwellwerte pro Restaurierung |
| `GoalApplicabilityFilter` | Deaktiviert physikalisch unmessbare Goals (Mono/Bandbreite) |
| `PhysicalCeilingEstimator` | Shannon-Grenze pro Goal, frühe Terminierung |
| `GoalPriorityProtocol` | 5-stufige Vorranghierarchie bei Pareto-Konflikten |
| `MicroDynamicsEnvelopeMorphing` | 400 ms LUFS-Profil-Korrektur, Savitzky-Golay |
| `EmotionalArcPreservationMetric` | Arousal/Valence Pearson ≥ 0.85/0.80, Klimax-Erhalt |
| `IntroducedArtifactDetector` | ML_HALLUCINATION / NMF_CLICK / SMEARING-Detektion |
| `StemRemixBalancer` | LUFS-korrekter Re-Mix nach getrennter Stem-Verarbeitung |
| `MusikalischerGlobalplanDienst` | Cross-Phase-Globalplan: 13 Ära-Profile × Genre-Modifikatoren, 17 Phase-Adjustments (v10.0.0) |
| `PerceptualAttentionModel` | Salienz-Karte [n_frames × 24 Bark-Bänder] ∈ [0.3, 2.0] |
| `BatchSessionLearner` | GP-Warm-Start von Datei zu Datei (SHA256-Session-ID) |
| `ReferenceAnchorSynthesizer` | 270 MUSDB18-HQ-Ankerpunkte (Ära × Genre × Material) |
| `VocalAIEnhancement` | Stimmtyp-adaptiv (MALE/FEMALE/CHILD/ANDROGYNOUS) |
| `HarmonicLatticeAnalyzer` | Fletcher-Modell, B-Koeff., Partial-Abw. ≤ 3 Cent |
| `StereoAuthenticityInvariant` | Mono-Ära M/S ≥ 0.97, Decca-Wide ∈ [0.25, 0.65] |
| `LyricsGuidedEnhancement` | Wort-zeitgenaue Klangverbesserung via Transkription (§2.36, Pflicht ab v10.0.0.x); Stimmtyp- und Phonem-adaptiv |
| `HumanHearingComfortPolicy` | Zentrale Hörkomfort-Steuerung: Peak-/HF-Caps, Eingriffsbudget pro Phase (§2.44) |
| `LiveDefectCounter` | GUI-Live-Defektzähler: Dropout-Chips via Timeline-Cursor (v10.0.0) |
| `RecordingChainProfiler` | Aufnahmeketten-Profiler: DAW-Limiter, Kassetten-Charakteristik (§2.66) |

---

## 🧠 Kognitive Orchestrierungsschicht (`denker/`)

`denker/` koordiniert alle Kernmodule als Hochsprachen-Orchestrierungsschicht
und produziert das vollständige `AurikErgebnis` (17 Felder, `@dataclass`).

| Denker | Zuständigkeit |
| --- | --- |
| `TontraegerDenker` | Trägermedium-Erkennung (Vinyl / Tape / CD / Digital) |
| `TontraegerketteDenker` | §6.6-Ketten-Erkennung (bindend ab v10.0.0) |
| `DefektDenker` | Defektanalyse via `CausalDefectReasoner` |
| `MusikalischerGlobalplanDienst` | Cross-Phase-Globalplan: 13 Ära-Profile × 17 Phase-Adjustments (§Dach) |
| `StrategieDenker` | Phasenstrategie + RT-Guard (`_3X_RT_LIMIT = 8.0`) + Human-Hearing-Comfort-Profil |
| `RestaurierDenker` | Vollrestaurierung via `UnifiedRestorerV3` |
| `ReparaturDenker` | Self-contained scipy-Direktreparatur |
| `RekonstruktionsDenker` | Lückenfüllung / Inpainting via `GapReconstructor` |
| `ExzellenzDenker` | 14 Musical Goals + `ExcellenceOptimizer` |
| `PhaseInteractionDenker` | Phasenübergreifende Interaktionsanalyse + Koalitions-Evaluation (§2.67) |

**Entry-Point:** `from denker import restauriere` ·
**Tests:** `tests/unit/test_denker/` (10 Dateien) ·
**Doku:** [`denker/README.md`](denker/README.md)

> Hinweis: `denker/` ist die interne Orchestrierungsschicht. Release-faehige Oberflaechen
> (GUI, CLI, Desktop-Pfade) laufen normativ ueber die Bridge und nachgelagerte Exportgates.

---

## 🚀 Quick Start

### 🎵 Für Einsteiger — Aurik in 3 Schritten starten

> **Kein Python, kein Terminal, keine Cloud nötig.** Aurik läuft direkt auf Ihrem Desktop.

| Schritt | Aktion | Was passiert |
| --- | --- | --- |
| **1** | **Datei öffnen** — Doppelklick auf `AURIK910.AppImage` (Linux) oder `AURIK910.exe` (Windows) | Das Programm startet. Alle KI-Modelle sind bereits enthalten — keine Internetverbindung nötig. |
| **2** | **Aufnahme laden** — Klick auf **📂 Datei öffnen** oder die Audiodatei ins Fenster ziehen | Aurik erkennt automatisch den Tonträger (Vinyl, Kassette, Shellac …) und analysiert alle Defekte. |
| **3** | **Modus wählen und starten** — Klick auf **📀 Restoration** oder **🎯 Studio 2026** | Die bearbeitete Datei wird im Ordner `output/` gespeichert; Qualitäts- und Exportgates laufen automatisch. |

**Unterstützte Formate:** WAV, FLAC, MP3, AIFF, OGG, M4A, WMA, AAC — Mono & Stereo

**Tastenkurzbefehle:** `A` = Original anhören, `B` = Restauriert anhören, `Leertaste` = Play/Pause

### Dokumentation

- Anwender- und Entwicklerdokumentation: [docs/INDEX.md](docs/INDEX.md)
- Benutzerpfad: [docs/guides/USER_GUIDE.md](docs/guides/USER_GUIDE.md)
- Kanonische Entwickler-API: [docs/api/PYTHON_API.md](docs/api/PYTHON_API.md)

---

### Fuer Entwicklung im Repository

```bash
# Clone Repository
git clone https://github.com/aurik-audio/Aurik_Standalone.git
cd Aurik_Standalone

# Setup Virtual Environment
python3 -m venv .venv_aurik
source .venv_aurik/bin/activate  # Linux/macOS
# .venv_aurik\Scripts\activate  # Windows

# Install Dependencies fuer Entwicklung/Tests
pip install -r requirements/requirements.txt
```

> Hinweis: Diese Schritte sind nur fuer Entwicklung im Repository. Endnutzer verwenden die gebuendelte Desktop-App.

### Präventiv-Quote (§v10.118)

**Mindestens 40% jeder Entwicklungssession müssen präventive Maßnahmen sein**
(Stabilität, Wartbarkeit, Linting, Audits, Scope-Bug-Detection).
Die restlichen 60% dürfen hörbare Wohlklang-Verbesserungen sein.

Diese Quote verhindert die "Feature-Falle" — immer neue Funktionen, nie stabiler.
Sie ist in `.github/specs/22_wohlklang_strategie.md` dokumentiert und gilt
für alle Contributors ab v10.0.16.

### GUI starten

```bash
./run_aurik.sh
# alternativ (Legacy-Kompatibilitaet):
python start_aurik_90.py
```

Datei laden → **Magic Button** wählen:

- **💿 Restoration** — originalgetreue Restaurierung
- **🎯 Studio 2026** — Highend-Studio-Sound

### CLI-Nutzung

```bash
# Restoration-Modus
PYTHONPATH=. ./.venv_aurik/bin/python cli/aurik_cli.py \
  --input aufnahme.wav --output restauriert.wav --mode Restoration

# Studio 2026-Modus
PYTHONPATH=. ./.venv_aurik/bin/python cli/aurik_cli.py \
  --input aufnahme.wav --output studio.wav --mode "Studio 2026"

# Optionale Parameter: -q/--quiet, --bit-depth 16|24|32, --output-sr 44100|48000
```

> Der Endnutzervertrag bleibt identisch: Modus waehlen, starten, Export automatisch pruefen.
> Intern mappt die Laufzeit den Modus auf den kanonischen Verarbeitungspfad.

**Exit-Codes (CLI):**
0 = Erfolg · 1 = Argumentfehler · 2 = Input fehlt · 3 = Importfehler · 4 = Pipelinefehler ·
5 = Exportfehler · 6 = Resamplingfehler · 10 = Pre-Analysis-Fehler. Quality-Gate-,
P1/P2- und Pegelabweichungen werden spec-konform als `degraded` exportiert, nicht hart abgebrochen.

### Python API

```python
from backend.api.bridge import (
    get_aurik_denker_instance,
    get_load_audio_fn,
    run_pre_analysis,
)

load_audio = get_load_audio_fn()
audio, sr = load_audio("input.wav")

# Voranalyse genau einmal pro Datei
pre_analysis = run_pre_analysis(audio, sr)

denker = get_aurik_denker_instance()
result = denker.denke(audio, sr, mode="restoration")

print(pre_analysis)
print(f"Quality: {result.quality_estimate:.2f}")
print(f"RT Factor: {result.rt_factor:.2f}×")
print(result.metadata.get("quality_gate_payload", {}))
```

> Fuer produktive Exporte ueber GUI/CLI laufen zusaetzlich `export_guard()` und
> `validate_export_quality()` im kanonischen Releasepfad.

---

## 📋 Features

### 🎼 Restaurierungs-Pipeline (66 Phasen)

**Pipeline-Reihenfolge (v10.0.0 — kanonisch):**

```text
TransientDecoupledProcessing → RestorabilityEstimator → EraClassifier
→ GermanSchlagerClassifier → MediumDetector → TontraegerketteDenker
→ DefectScanner → CausalDefectReasoner → UncertaintyQuantifier
→ MusikalischerGlobalplanDienst → GPParameterOptimizer
→ HarmonicPreservationGuard → Phase 01–66 (mit PerPhaseMusicalGoalsGate)
→ IntroducedArtifactDetector → HallucinationGuard → FeedbackChain
→ TemporalQualityCoherenceMetric → PerceptualQualityScorer
→ HumanHearingComfortPolicy → ExcellenceOptimizer → MusicalGoalsChecker
→ EmotionalArcPreservationMetric → MicroDynamicsEnvelopeMorphing
→ GPParameterOptimizer.update() → RestorationResult
```

**Defektkorrektur (Phase 01–30):**

- Phase 01: Click Removal · Phase 02: Hum Removal · Phase 03: Denoise (OMLSA)
- Phase 09: Crackle Removal (RBME) · Phase 12: Wow/Flutter Fix (pYIN)
- Phase 24: Dropout Repair (NMF-β+PGHI) · Phase 29: Tape Hiss (OMLSA)
- Phase 30: DC-Offset Removal · + weitere 22 Defekt-Phasen

**Enhancement & Mastering (Phase 31–55):**

- Phase 38: Presence Boost · Phase 39: Air-Band Enhancement (> 12 kHz)
- Phase 40: Loudness-Normierung (EBU R128) · Phase 47: True-Peak-Limiter (−1 dBTP)
- Phase 48: Stereo-Width · Phase 49: Advanced Dereverb (Blind-RIR)
- Phase 55: DiffWave/Flow-Matching-Inpainting · + Instrumental- und Vocal-Phasen

**Guard- & Policy-Phasen (Phase 56–66):**

- Phase 56: SpectralBandGapRepair (HEAD_WEAR) · Phase 57: DAW-Limiter-Erkennung + Dekompressor
- Phase 58: HallucinationGuard (§2.46e) · Phase 62: TemporalContinuityGuard
- Phase 63: Real-Audio-Comfort-Gate · Phase 66: Human-Hearing-Comfort-Policy (§2.44)

**Aktuelle Erweiterungen (v10.0.0):**

- §2.44 HPG Reference Memory Bootstrap + Human-Hearing-Comfort-Policy
- §2.46e HallucinationGuard
- §2.66 RecordingChainProfiler (DAW-Limiter, Kassetten-Charakteristik)
- §2.67 Phase-Koalitions-Evaluation
- §2.69 TemporalContinuityGuard
- §2.70 RestorationMemory

**Instrument-adaptive Phasen (PANNs-aktiviert):**

- Guitar → Phase 44 · Brass → Phase 45 · Drums → Phase 51 · Piano → Phase 52 · Vocals → Phase 42

### 🤖 ML-Plugin-Architektur

**Prinzip:** DSP als Fundament, ML als Erweiterung — immer mit DSP-Fallback:

| Situation | ML-Plugin (primär) | Fallback |
| --- | --- | --- |
| Breites Rauschen | DeepFilterNet v3.II (ONNX, 37 MB) | OMLSA+IMCRA (DSP) |
| Raumrauschen / Reverb | WPE (Nakatani 2010) | nara_wpe → OMLSA (DSP) |
| Stem-Separation Vocals | MDX23C Kim_Vocal_2 (64 MB) | NMF-β |
| Stem-Separation Instrumente | MDX23C Kim_Inst (64 MB) | Energy-Masking |
| Codec-Artefakte | **Apollo** (65 MB ONNX) | Resemble-Enhance |
| Dropout < 50 ms | NMF-β + Sinusoidal (DSP) | Consistent Wiener |
| Dropout 50–999 ms | CQTdiff+ / **Flow Matching** | DiffWave ONNX |
| Pitch-Tracking mono | CREPE full (85 MB) | pYIN (DSP) |
| Pitch-Tracking polyphon | BasicPitch (ONNX) | pYIN Multi-Pitch |
| Audio-Tagging / Genre | PANNs CNN14 (81 KB) | DSP Spectral Fingerprint |
| Bandbreiten-Erweiterung | **BW Harmonic Exciter** (DSP, 0 MB) | AudioSR (5,9 GB, lazy) |
| Vocos-Vocoder (Synthese) | **Vocos 24 kHz** (52 MB) | HiFi-GAN → PGHI-ISTFT |
| MOS Musik (ohne Referenz) | **CDPAM** (102 MB) | PQS-DSP (Gammatone) |
| MOS Musik (mit Referenz) | **ViSQOL v3 `--audio`** | PQS-DSP |
| Music Understanding | MERT-v1-330M (3,9 GB, lazy) | Harmonicity+Chroma DSP |
| ~~MOS-Schätzung~~ | ~~DNSMOS / NISQA~~ | **⛔ VERBOTEN** für Musik |

**Alle ML-Plugins:** `plugins/` — jedes mit `try/except ImportError` DSP-Fallback.  
**Hardware-Modus:** CPU verpflichtend, optionale AMD-GPU-Beschleunigung (ROCm/DirectML) fuer Heavy-Modelle mit transparentem CPU-Fallback.  
**Bundled:** Alle primären Modelle lokal gebündelt — kein Download beim ersten Start.

### 🎯 Material-Adaptive Verarbeitung (15 Typen)

**Auto-Detection** via `MediumDetector` (file-ext-aware, DSP + forensische Kettenlogik) — **15 Material-Typen:**

| Material | Hauptdefekte | PQS-Ziel |
| --- | --- | --- |
| `tape` | Dropout, Hiss, Wow/Flutter | MOS ≥ 4.2 |
| `reel_tape` | Print-Through, Hiss, Dropout | MOS ≥ 4.3 |
| `vinyl` | Crackle, Warp, Rille-Distortion | MOS ≥ 4.0 |
| `shellac` | Breites Rauschen, BW ≤ 8 kHz | MOS ≥ 3.8 |
| `wax_cylinder` | Extremrauschen, HF ≤ 5 kHz, Zylinderverzerrung | MOS ≥ 3.5 |
| `wire_recording` | Magnetdraht-Jitter, Frequenz-Dropout | MOS ≥ 3.6 |
| `lacquer_disc` | Riss-Klicken, Rille-Ermüdung, Substrat-Rauschen | MOS ≥ 3.7 |
| `dat` | Jitter, Dropout, ATRAC-Artefakte | MOS ≥ 4.4 |
| `cd_digital` | Clipping, Quantisierungsrauschen | MOS ≥ 4.5 |
| `mp3_low` | Schwere Codec-Artefakte (< 128 kbps) | MOS ≥ 3.9 |
| `mp3_high` | Moderate Codec-Artefakte (≥ 128 kbps) | MOS ≥ 4.2 |
| `aac` | Präsenz-Verlust, Apple-Kompression | MOS ≥ 4.2 |
| `minidisc` | ATRAC-Stufigkeit, HF-Verlust | MOS ≥ 4.0 |
| `streaming` | Variables Bitrate-Profil | MOS ≥ 4.1 |
| `unknown` | Konservative Prior, alle Tier-1 Phasen | MOS ≥ 3.8 |

### 📊 Die 14 Musikalischen Qualitätsziele

Nach jeder Restaurierung werden alle 14 Ziele geprüft (adaptiv via `AdaptiveGoalThresholds`
und `GoalApplicabilityFilter`). Regression in einem Ziel macht das Feature ungültig:

| # | Ziel | Frequenzbereich / Messgröße | Schwellwert |
| --- | --- | --- | --- |
| 1 | **Brillanz** | HF-Klarheit 8–20 kHz | ≥ **0.85** |
| 2 | **Wärme** | Mitten 200–2000 Hz | ≥ **0.80** |
| 3 | **Natürlichkeit** | Artefaktfreiheit | ≥ **0.90** |
| 4 | **Authentizität** | Spektraler Fingerabdruck | ≥ **0.88** |
| 5 | **Emotionalität** | Dynamik, Modulationstiefe | ≥ **0.87** |
| 6 | **Transparenz** | Klangbildtrennung | ≥ **0.89** |
| 7 | **Bass-Kraft** | 20–250 Hz + Virtual Pitch | ≥ **0.85** |
| 8 | **Groove** | Mikro-Timing, DTW ≤ 8 ms RMS | ≥ **0.88** |
| 9 | **Raumtiefe** | Stereobreite, Phantom-Center | ≥ **0.75** |
| 10 | **Timbre-Authentizität** | MFCC-Pearson ≥ 0.95 | ≥ **0.87** |
| 11 | **Tonales Zentrum** | Chroma-Korrelation, kein Key-Shift | ≥ **0.95** |
| 12 | **Mikro-Dynamik** | LUFS-Profil 400 ms, Crest-Faktor | ≥ **0.92** |
| 13 | **Separation-Treue** | SDR ≥ 8 dB / SIR ≥ 12 dB | ≥ **0.82** |
| 14 | **Artikulation** | Attack-Charakter, Transient-Shape | ≥ **0.85** |

**PQS-Metriken** (`PerceptualQualityScorer`):

| Metrik | Minimum | Internes Spitzenziel |
| --- | --- | --- |
| MOS | ≥ 3.8 | ≥ 4.5 |
| NSIM | ≥ 0.70 | ≥ 0.90 |
| MCD | ≤ 8.0 dB | ≤ 3.0 dB |
| Spectral Coherence | ≥ 0.60 | ≥ 0.85 |

### ⚡ Verarbeitungs-Modi

**💿 Restoration-Modus** (auf Originaltreue optimiert):

- Chroma-Korrelation ≥ 0.95 · LUFS-Differenz ≤ 1 LU
- Kein Harmonic-Exciter-Material · Authentizität über alles
- `ExcellenceOptimizer(mode="restoration")`: konservative GP-Params
- `MicroDynamicsEnvelopeMorphing` MAX_GAIN = 2.0 LU

**🎯 Studio 2026-Modus** (auf modernen Studiosound optimiert):

- PQS MOS ≥ 4.5 · Brillanz ≥ 0.90 · Bass-Kraft ≥ 0.88
- Stem-Separation (MDX23C/BS-RoFormer) → `StemRemixBalancer` → Re-Mix
- `ExcellenceOptimizer(mode="studio2026")`: aggressive Pareto-GP-Params
- 11-stufige Verarbeitungskette bis zum finalen True-Peak-Limiter

**🎵 Genre-Restore-Profile:**

- Schlager: Akkordeon-Charakter erhalten, DeEsser ≤ 45 %, Wärme 0.88
- Klassik: Dereverb deaktiviert, Transienten-Erhalt maximiert
- Jazz: Groove-DTW ≤ 4 ms (Timing heilig), HSI bewahren

---

## 🧪 Testing & Validation

### Test-Suite (~18.400 Tests, 511 mit Markern)

```bash
# Alle Tests
pytest tests/ --disable-warnings --tb=short

# Unit-Tests (schnell, alle mit @pytest.mark.unit)
pytest tests/unit -m unit --timeout=30 --tb=short -q

# Pleasantness-First Tests
pytest -m pleasantness

# Goal-Achievement Tests (beweisen Weltklasse-Klang)
pytest -m goal_achievement

# No-Blind-Trust Verifikation
pytest tests/unit/test_no_blind_trust.py

# Musical Goals
pytest tests/musical_goals tests/unit -q

# Schlager-Klassifikation (≥ 35 Tests)
pytest tests/unit/test_v99_genre_schlager.py -v

# Normative Contract-Gates
pytest tests/normative -q

# Neue v10.0.0.x Module
pytest tests/unit/test_human_hearing_comfort.py -v
pytest tests/unit/test_adaptive_pipeline_canonical_policy_guard.py -v
pytest tests/normative/test_real_audio_edge_lag_gate.py -v
pytest tests/normative/test_modern_window_gui_contract.py -v

# Phase Contract Tests (§v10.0.5 — PostGate Signatur + Genre-Propagation)
pytest backend/tests/test_phase_contracts.py -v
```

## ⏱️ Performance & Phase-Budget

Aurik verwendet ein **Wall-Time-Budget** pro Song, um zu verhindern dass die
Pipeline auf schwacher Hardware unbegrenzt läuft. Das Kernprinzip:

| Budget-Typ | Beschreibung |
|------------|-------------|
| **Mandatory Phases** | Denoise, Click, Crackle, Wow/Flutter, DC-Offset — laufen **immer** |
| **Enhancement Phases** | Frequency Restore, DeEsser, Dereverb, Transparent Dynamics — können **übersprungen** werden |
| **Budget Guard** | `performance_guard.py` + `_budget_pressure_skip_reason()` — überspringt Enhancement-Phasen als Passthrough wenn Wall-Time erschöpft |
| **Material-adaptiv** | Tape/Cassette bekommen höheres Budget (längere Analysen nötig) |

### Fairness-Garantie

- **Kurze Songs (< 100 s)**: Alle Enhancement-Phasen laufen vollständig
- **Lange Songs (> 200 s)**: Enhancement-Phasen können entfallen; die Kern-Reparaturkette bleibt intakt
- **`speed`-Mode**: Reduziert ML-Tiefe → alle Phasen laufen auch bei langen Songs
- **`quality`-Mode**: Maximale Qualität, aber höheres Risiko von Budget-Überspringungen

### Budget-Garantie (§v10.0.5)

Seit v10.0.5 wurde das Wall-Time-Budget **verdoppelt**:

- **Base-Budget**: Vinyl 2700s → **5400s** (90 min), Kassette 4800s → **7200s** (120 min)
- **Overhead**: 1800s → **3600s** (Modell-Laden, Kalibrierung, Setup)
- **Per-Sekunde**: 15s → **25s** (mehr Puffer für DSP-intensive Phasen)
- **RT-Limit**: 32× → **48×** (aus CRITICAL-Bereich bei 4-Kern-CPU)
- **Min-Effective-Strength-Guard**: Phasen mit strength < 0.12 werden als Passthrough übersprungen (§0)

### §v10.0.5 Änderungen (14 Dateien, ~1100 Zeilen)

| Kategorie | Highlights |
|-----------|-----------|
| **Bugfixes** | Genre-Key-Mismatch, PostGate Lambda-TypeError, Tuple-`ndim`-Crash, UVR-Div0 |
| **Genre** | ambient+world Profile (8+12 Par.), oper+schlager vervollständigt, PQC 12 Modifier |
| **Psychoakustik** | LUFS-adaptives Fletcher-Munson Phon-Level, `compute_adaptive_phon()` |
| **Quality-Gates** | Instrumental-Recovery (phase_65), Exciter-Freigabe, DeEsser-Skip für HF-loses Material |
| **Performance** | Budget ×2, RT-Limit 48×, 5 Phasen-Skip-Guards, Min-Effective-Strength |
| **Goosebumps** | Genre-adaptive Weights + Thresholds (11 Profile) |
| **OneTakeExport** | Iterative 2-Pass mit Feinkorrektur |
| **Tests** | +37 Tests (24 Contract + 13 Genre-Universalität) |

Damit laufen **alle geplanten Phasen** auch auf 4-Kern-CPU mit `quality`-Mode —
kein Budget-bedingtes Überspringen von Enhancement-Phasen mehr.

> Vor §v10.0.5: 9 Enhancement-Phasen wurden bei 225s Song auf 4 Kernen übersprungen.
> Seit §v10.0.5: Budget reicht für vollständige Pipeline-Ausführung.

**Test-Mindestanforderung pro neuem Modul:** ≥ 35 Unit-Tests,
inkl. NaN/Inf-Tests, Bounds-Tests, Mono+Stereo, Edge-Cases, Thread-Safety.

#

### 🔍 Pre-Commit Static-Value-Guard (§v10)

Verhindert blinde statische Werte ohne Song-Messung:

```bash
# Manuell ausführen
python scripts/pre_commit_static_guard.py --ci

# Als pre-commit Hook (automatisch bei jedem Commit)
# Bereits in .pre-commit-config.yaml aktiviert
```

## Ära-Klassifikation & AMRB-Benchmark

```bash
# Voranalyse läuft automatisch vor jedem CLI- und GUI-Lauf über die Bridge.

# AMRB v1.0 (10 Szenarien, interne Führungs-Schwelle ≥ 84.0)
python benchmarks/musical_restoration_benchmark.py

# Kompetitiver Benchmark (vs. iZotope RX 11; interner Führungsindikator)
python scripts/competitive_benchmark.py

# Competitive CI-Gate (schneller CI-Run)
pytest tests/normative/test_competitive_ci_gate.py -m competitive --timeout=600 -v

# Competitive Nightly (Spec-robust: n_items ≥ 5 pro Szenario)
AURIK_NIGHTLY_ITEMS=5 pytest tests/normative/test_competitive_ci_gate.py -m competitive --timeout=600 -v
```

---

## 🕰️ Ära-Klassifikation (1890–2025)

| Dekade | Material-Typ | GP-Warmstart NR | Stereo-Invariante |
| --- | --- | --- | --- |
| ≤ 1930 | wax_cylinder / shellac | NR-Stärke ∼ N(0.90, 0.05) | Mono, M/S ≥ 0.97 |
| 1930–1945 | shellac / lacquer_disc | NR-Stärke ∼ N(0.85, 0.07) | Mono |
| 1945–1960 | reel_tape / lacquer_disc | NR-Stärke ∼ N(0.75, 0.08) | Früh-Stereo (Blumlein/Decca) |
| 1960–1970 | tape / vinyl | NR-Stärke ∼ N(0.65, 0.09) | Decca-Wide [0.25, 0.65] |
| ≥ 1970 | tape / vinyl / digital | NR-Stärke ∼ N(0.50, 0.10) | Standard |

`BrillanzMetric` ceiling era-adaptiv: ≤ 1930 → 0.72 · ≤ 1950 → 0.80 · ≥ 1980 → 0.95

---

## 🎶 Genre-Klassifikation (Zero-Shot)

`GermanSchlagerClassifier` erkennt Deutschen Schlager **ohne vortrainiertes Genre-Modell**:

| Schicht | Methode | Schwellwert |
| --- | --- | --- |
| Tier-1 CLAP | 7 gewichtete Prompts (DE+EN) | ≥ 0.26 |
| Tier-2 Akkordeon | Reed-Beating AM 5–15 Hz (Hilbert) | ≥ 0.65 |
| Tier-3 HSI | Chroma Quintenkreis ≤ 2 Schritte | ≥ 0.82 |
| Tier-4 Rhythmus | Oompah/Walzer/Marsch (madmom) | ≥ 0.60 |
| Tier-5 Vokal | SAMPA-Formant-Overlap ä/ö/ü | Tie-Breaker |
| Tier-6 Repetition | SSM MFCC Kosinus ≥ 0.85 | ≥ 0.42 |

Voting: ≥ 3/5 DSP-Schichten + Gesamt ≥ 0.52 → `is_schlager=True`  
Recall ≥ 90 % (mit CLAP) · False-Positive < 5 % · ≤ 20 s/Minute Audio

---

## 🆕 Neu in v10.10 — Preset-Learning × Selbstkalibrierung

| Modul | Feature |
|-------|---------|
| `magic_restore_preset.py` | Ein-Klick-Preset: Material/Ära/Genre → optimales Preset (6 built-in + User-Learning) |
| `mid_pipeline_quality_gate.py` | HPE-Wächter alle 8 Phasen: Selbstkalibrierung bei Verschlechterung |
| `reference_track_calibrator.py` | Referenz-Track → Song-Goals automatisch (Preset ∩ Material-Floor) |
| `model_warmup_pool.py` | 5 Modelle parallel laden während Pre-Analyse → Cold-Start 5→0s |
| `phase_fingerprint.py` | Inkrementelles Re-Processing: nur geänderte Phasen neu (70% schneller) |
| `parallel_stereo_executor.py` | 12 DSP-Phasen parallel left/right (~40% Speedup) |
| `album_consistency_gate.py` | Track 1 = Referenz, Tracks 2-N = LUFS/Tilt/Width-kalibriert |
| `spectrogram_snapshot.py` | 256×256 Spektrogramm-PNG pro Phase für Debugging |

## 🆕 Neu in v10.9 — SOTA-Kalibrierung & ML-Orchestrierung

| Modul | Feature |
|-------|---------|
| `model_chain_orchestrator.py` | Shared ML-Modelle (RAM 250MB gespart), RAM-Budget 6GB |
| `adaptive_phase_order.py` | Material-adaptive Phasen-Reihenfolge (Kassette: Hiss vor Harmonics) |
| `live_ab_preview.py` | A/B-Vorher/Nachher-Ring für GUI-Playback |
| `restoration_report.py` | HTML-Report mit Phasen/Defekten/Joy/Fatigue |
| **Phase 07** | Drive 2.5→1.8, Pre-Echo 2→3.5ms, h2 material-adaptiv |
| **OneTakeExport** | Adaptiver Fatigue-Cut (-1/-2/-3dB), TP-Limiter -0.3 dBTP |
| **Pipeline** | Early-Silence-Gate, Wet/Dry-Kohärenz, Era-Material-Fallback |
| **MRN Plugins** | 4 ML-Chains: Shellac/Vinyl/Tape/Lacquer |

## 🔧 v10.35 — Logging & Dead-Feature-Aktivierung

| Kategorie | Änderung |
|-----------|----------|
| SOTA Logging | 15 Features debug→warning (PGHI, MRN, EAPC, HHC, MaskingClamp, etc.) |
| GUI-Kommunikation | Denker→GUI Toasts, ErrorSimplifier, Experience Insights |
| SFT Rescue | Min-Wet 0.05, WARNING bei aggressivem Rollback |

---

## ⚙️ Technische Details

| Aspekt | Wert |
| --- | --- |
| Interne Sample-Rate | **48 000 Hz** (alle DSP/ML/Metriken) |
| Bit-Tiefe intern | float32, Bereich [−1, 1] |
| Hardware | CPU-first mit optionalem AMD-GPU-Mixed-Mode (ROCm/DirectML), CPU-Fallback verpflichtend |
| Resampling | Lanczos-4, `scipy.signal.resample_poly`, Kaiser β=14 |
| GP-Gedächtnis | `~/.aurik/gp_memory/<material>.json` (lokal, persistent) |
| Artist-Signaturen | `~/.aurik/artist_signatures/<artist_id>.json` |
| Export-Lautheit | EBU R128: −14 LUFS (Streaming) / −18 LUFS (Archiv) |
| True-Peak-Limit | −1.0 dBTP (ITU-R BS.1770-5) |
| Dithering | POW-r Typ 3 (24→16 bit), Fallback TPDF |
| FeedbackChain | max. 5 Iterationen, Konvergenz ΔMOS < 0.02 |
| PerPhaseGoalsGate | Rollback adaptiv (0.012–0.060), max. 5 Retries |
| Chunk-Verarbeitung | 5/15/60/120 s (defektdichte-adaptiv) |

---

## 📚 Dokumentation

| Dokument | Inhalt |
| --- | --- |
| [docs/INDEX.md](docs/INDEX.md) | Vollständiger Dokumentationsindex |
| [docs/KI-AGENT-INTEGRATION-GUIDE.md](docs/KI-AGENT-INTEGRATION-GUIDE.md) | Richtlinien für KI-Agenten |
| [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) | Aktueller Projektstand |
| [docs/guides/INSTALLATION.md](docs/guides/INSTALLATION.md) | Installationsanleitung |
| [docs/guides/USER_GUIDE.md](docs/guides/USER_GUIDE.md) | Benutzerhandbuch |
| [CHANGELOG.md](CHANGELOG.md) | Versionshistorie |
| [.github/copilot-instructions.md](.github/copilot-instructions.md) | **KI-Programmierrichtlinien (bindend)** |
| [denker/README.md](denker/README.md) | Kognitive Orchestrierungsschicht — Denker-Agenten |

---

## 🙏 Dankeschön

**ML-Modelle & Algorithmen:**

- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) — Cohen/OMLSA-gestützte Rauschunterdrückung
- [Apollo](https://github.com/Qiuqiu0529/apollo) — Codec-Artefakt-Restaurierung (Zhang 2024)
- [Vocos](https://github.com/hubert-siuzdak/vocos) — Neuronaler Vocoder (MIT, 24 kHz ONNX)
- [WPE / nara_wpe](https://github.com/fgnt/nara_wpe) — Dereverb (Nakatani 2010)
- [MDX23C](https://github.com/ZFTurbo/Music-Source-Separation-Training) — Stem-Separation

**Forschungsgrundlagen:**

- iZotope RX — Kommerzieller Referenz-Standard
- Cohen (2002/2003) — OMLSA/IMCRA
- Févotte & Idier (2011) — NMF-β
- Perraudin et al. (2013) — PGHI
- Fletcher (1964) — Harmonisches Gitter / Inharmonizität

---

## 📜 Lizenz

Aurik 10.0.0 steht unter der **Apache-2.0-Lizenz** — siehe [LICENSE](LICENSE).

---

Aurik 10.0.10 — Juli 2026
