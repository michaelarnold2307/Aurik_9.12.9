# KI-Agent Integration Guide — AURIK 10.0.8

**Erstellt:** 15. Februar 2026 | **Aktualisiert:** 19. Mai 2026  
**Version:** 10.0.8  
**Zielgruppe:** KI-Agenten (GitHub Copilot, Claude, GPT) die an AURIK arbeiten  
**Status:** 🟢 AKTIV — Verbindlich für alle KI-Agenten

---

## ⚠️ PFLICHTLEKTÜRE — Lies dies zuerst

**Die bindenden KI-Programmierrichtlinien befinden sich in:**  
→ `.github/copilot-instructions.md` (verbindlich für alle KI-Agenten)

Dieses Dokument liefert **praktische Ergänzungen** zu den Richtlinien.

### Die 5 absoluten Regeln (Kurzfassung)

1. **Anti-Parallelwelten**: Vor jeder Implementierung bestehende Module in `core/`, `plugins/`, `dsp/` prüfen
2. **15 Musical Goals**: Nach jeder Restaurierung über `MusicalGoalsChecker.measure_all()` prüfen — Regression macht das Feature ungültig
3. **48 kHz überall**: `assert sample_rate == 48000` in jeder Phase und jedem Plugin
4. **CPU + optionale AMD-GPU**: Kein CUDA — `providers=["CPUExecutionProvider", "ROCMExecutionProvider"]`, `model.to("cpu")`
5. **NaN/Inf verboten**: `np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)` nach jeder numerischen Operation

---

## 📋 Systemübersicht: AURIK 10.0.8 Architektur

### Kognitive Kernmodule (Pflichtübersicht)

| Modul | Datei | Funktion |
| --- | --- | --- |
| `PerceptualEmbedder` | `core/perceptual_embedder.py` | 256-dim L2-normalisierter Einbettungsraum |
| `CausalDefectReasoner` | `core/causal_defect_reasoner.py` | Bayesianische Kausalinferenz, 62 Kausal-Ursachen |
| `GPParameterOptimizer` | `core/gp_parameter_optimizer.py` | RBF-GP + UCB, lernt dauerhaft pro Material |
| `PerceptualQualityScorer` | `core/perceptual_quality_scorer.py` | Gammatone-NSIM + MCD + LUFS + MOS |
| `MusicalGoalsChecker` | `backend/core/musical_goals/musical_goals_metrics.py` | 15 Ziele, `measure_all(audio, sr)` |
| `DefectScanner` | `core/defect_scanner.py` | 62 DefectTypes, material-adaptive Klassifikation |
| `UnifiedRestorerV3` | `core/unified_restorer_v3.py` | Phasen-Pipeline-Orchestrator (69 Phasen-Dateien) |
| `VocalAIEnhancement` | `core/vocal_ai_enhancement.py` | `VoiceGender` (MALE/FEMALE/CHILD/ANDROGYNOUS) |
| `ExcellenceOptimizer` | `core/excellence_optimizer.py` | `optimize_for_excellence()` |
| `FeedbackChain` | `core/feedback_chain.py` | Iterative PQS-Schleife, max. 5 Iterationen |
| Hör-Gates E1/2/4 | `core/dsp/level_1_invariants_guard.py`, `core/dsp/einladungs_gate.py`, `core/dsp/vocal_overdrive_guard.py`, `core/defect_audibility_gate.py` | Laufzeit-Guards der Hörordnung (Invarianten/Audibility/Wohlklang/Vocal-Schutz) |

### Kanonischer Pipeline-Ablauf

```
Eingang (beliebige SR, mono/stereo)
    │
    ▼ auf 48 kHz resampeln (Lanczos-4)
    │
    ▼ [DefectScanner.scan()] → DefectAnalysisResult (62 DefectTypes, material-adaptiv)
    │
    ▼ [CausalDefectReasoner.reason_about_defects()] → RestorationPlan
    │   .primary_cause, .recommended_phases, .phase_parameters, .reasoning
    │
    ▼ [GPParameterOptimizer.propose(material)] → ParameterProposal (aus ~/.aurik/gp_memory/)
    │
    ▼ [UnifiedRestorerV3._select_phases()] → Tier-Selektion inkl. CausalPlan
    │
    ▼ [PerceptualEmbedder.embed_audio()] → AudioEmbedding (256-dim, L2-normalisiert)
    │
    ▼ Phase 01–68 ausführen (core/phases/phase_NN_*.py)
    │
    ▼ [FeedbackChain.run()] → iteriert bis MOS konvergiert (|ΔMOS| < 0.02)
    │
    ▼ [PerceptualQualityScorer.score_audio_absolute()] → PQSResult
    │
    ▼ [ExcellenceOptimizer.optimize_for_excellence()] → ExcellenceResult
    │
    ▼ [MusicalGoalsChecker.measure_all(audio, sr)] → Dict[str, float]
    │   PFLICHT: alle 14 Ziele ≥ Schwellwert → Fehler = Rollback auf best_result
    │
    ▼ [GPParameterOptimizer.update()] → persistiert Lernerfolg
    │
    └─▶ RestorationResult (audio, defect_analysis, pqs_result, musical_goals, excellence)
```

---

## 🎯 15 Musical Goals — Pflicht-Integration

```python
from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

checker = MusicalGoalsChecker()
scores = checker.measure_all(audio, sr)  # Dict[str, float]

# Pflicht-Check nach jeder Restaurierungsoperation:
if not all(scores[g] >= checker.thresholds[g] for g in checker.thresholds):
    logger.warning("Musical Goal Regression — Rollback auf best_result")
    return best_result  # NICHT das zuletzt verarbeitete Audio!
```

**Schwellwerte + Prioritätsstufen:**

| Ziel | Restoration-Schwellwert | Studio-2026-Schwellwert | Priorität |
| --- | --- | --- | --- |
| Natürlichkeit | ≥ 0.90 | ≥ 0.90 | **P1** (Rollback bei Regression) |
| Authentizität | ≥ 0.88 | ≥ 0.88 | **P1** (Rollback bei Regression) |
| Tonales Zentrum | ≥ 0.95 | ≥ 0.97 | P2 |
| Timbre-Authentizität | ≥ 0.87 | ≥ 0.87 | P2 |
| Artikulation | ≥ 0.85 | ≥ 0.85 | P2 |
| Emotionalität | ≥ 0.82 | ≥ 0.87 | P3 |
| Mikro-Dynamik | ≥ 0.88 | ≥ 0.92 | P3 |
| Groove | ≥ 0.83 | ≥ 0.88 | P3 |
| Transparenz | ≥ 0.82 | ≥ 0.89 | P4 |
| Wärme | ≥ 0.75 | ≥ 0.80 | P4 |
| Bass-Kraft | ≥ 0.78 | ≥ 0.85 | P4 |
| Separation-Treue | ≥ 0.78 | ≥ 0.82 | P4 |
| Brillanz | ≥ 0.78 | ≥ 0.85 | P5 (best-effort) |
| Raumtiefe | ≥ 0.70 | ≥ 0.75 | P5 (best-effort) |

**Stufe 1+2** verschlechtern → sofortiger Iterations-Abbruch + Rollback auf `best_result`.  
**Stufe 5** verschlechtern → Warnung loggen, Pipeline weiterführen.

---

## 🔍 Anti-Parallelwelten-Checkliste (VOR jeder Implementierung — §9.4)

```bash
# 1. Gibt es eine bestehende Funktion?
grep -r "class MeinModul\|def meine_funktion" core/ plugins/ dsp/ backend/ denker/

# 2. Gibt es ein Plugin dafür?
ls plugins/

# 3. Gibt es eine Phase dafür?
ls core/phases/

# 4. Gibt es ein Modul mit ähnlichem Namen?
ls core/*.py | grep -i "mein_bereich"
```

**Wenn alle 4 Checks negativ** → neue Implementierung nach Thread-sicherem Singleton-Pattern (§3.2).
**Wenn ein Check positiv** → bestehende Implementierung erweitern oder einbinden. `denker/` **delegiert** — es repliziert keine Logik aus `core/`.

---

## 🗂️ Phasen-System (69 Phasen-Dateien: Phase 01–66 + Glue Stage + Interface)

Alle Phasen liegen in `core/phases/phase_NN_<beschreibung>.py` (backend/core/phases/).

**Neue Phase erstellen — Pflicht:**

1. Datei: `core/phases/phase_NN_<beschreibung>.py`
2. Implementiert `PhaseInterface.process(audio, sample_rate, **kwargs) → np.ndarray`
3. `assert sample_rate == 48000` am Anfang
4. `audio = np.clip(result, -1.0, 1.0)` am Ende (kein NaN/Inf)
5. Export in `core/phases/__init__.py`

**CausalReasoner → Phase-Mapping:**

```python
CAUSE_TO_PHASES = {
    "tape_dropout":      ["phase_24_dropout_repair", "phase_55_diffusion_inpainting"],
    "tape_hiss":         ["phase_29_tape_hiss_reduction", "phase_03_denoise"],
    "vinyl_crackle":     ["phase_09_crackle_removal", "phase_01_click_removal"],
    "vinyl_warp":        ["phase_12_wow_flutter_fix", "phase_31_speed_pitch_correction"],
    "electrical_hum":    ["phase_02_hum_removal"],
    "head_misalignment": ["phase_06_frequency_restoration", "phase_14_phase_correction"],
    "dc_offset":         ["phase_30_dc_offset_removal"],
    "digital_clip":      ["phase_23_spectral_repair", "phase_06_frequency_restoration"],
}
```

---

## 📦 Material-System (§6.1, §6.3)

**Materialien (`MaterialType`):**

```
tape · reel_tape · vinyl · shellac · wax_cylinder · wire_recording · lacquer_disc
dat · cd_digital · mp3_low · mp3_high · aac · minidisc · streaming · unknown
```

**62 DefectTypes (vollständig, Stand v10.0.8):**

```
CLICKS · CRACKLE · HUM · WOW · FLUTTER · LOW_FREQ_RUMBLE · DROPOUTS
STEREO_IMBALANCE · PHASE_ISSUES · DIGITAL_ARTIFACTS
COMPRESSION_ARTIFACTS · HIGH_FREQ_NOISE
CLIPPING · SOFT_SATURATION · DC_OFFSET · BANDWIDTH_LOSS · PITCH_DRIFT
REVERB_EXCESS · PRINT_THROUGH · QUANTIZATION_NOISE
JITTER_ARTIFACTS · DYNAMIC_COMPRESSION_EXCESS
HEAD_WEAR · AZIMUTH_ERROR · TRANSIENT_SMEARING · PRE_ECHO
RIAA_CURVE_ERROR · ALIASING · BIAS_ERROR · SIBILANCE · TRANSPORT_BUMP · VOCAL_HARSHNESS
```

⚠️ **WOW** und **FLUTTER** sind seit v10.0.0.x getrennte Defekttypen (IEC 60386-konform, nicht mehr WOW_FLUTTER).

> **Kritisch:** `SOFT_SATURATION` = Tube-/Tape-Sättigung → **BEWAHREN**, nie reparieren!  
> `CLIPPING` = Harte Amplitudenbegrenzung → **REPARIEREN** (phase_23). Diskriminierung via `classify_clipping()` (§6.3).

---

## 🎤 VoiceGender-System (VocalAIEnhancement — §2.8)

```python
from core.vocal_ai_enhancement import VoiceGender, VoiceAgeGroup, EmotionPreservationMode

# Auto-Erkennung:
detector = GenderDetector()
characteristics = detector.detect(audio)  # → VoiceCharacteristics (F₀, Formanten, Breathiness)

# Stimmtyp-adaptive Verarbeitungskette (Reihenfolge zwingend!):
# 1. CrepePlugin (f₀) → pYIN-Fallback
# 2. FormantTracker (LPC F1–F4) + WORLD-Vocoder-Quervalidierung
# 3. BreathDetector, PhonemeDetector, ConsonantDetector
# 4. ConsonantEnhancement (Frikative adaptiv: MALE 5–10 kHz, FEMALE 6–12 kHz, CHILD 7–14 kHz)
# 5. De-Esser + ML-De-Esser (stimmtyp-spezifisch)
# 6. VocalAIEnhancement.enhance(audio, characteristics)
# 7. Formant-Prüfung: Pearson(F1_before, F1_after) ≥ 0.95
enhancer = VocalAIEnhancement()
result = enhancer.enhance(audio, characteristics)
```

**Invarianten (zwingend — §2.8):**

- Formant-Korrelation Pearson ≥ 0.95 (Authentizitäts-Invariante)
- Breathiness-Ratio: Änderung ≤ ±0.05 (natürliche Stimmfärbung)
- Vibrato-Rate: Änderung ≤ ±0.3 Hz (emotionaler Ausdruck)
- Frikativ-SNR: Verbesserung ≥ +3 dB, keine HF-Energie > 14 kHz abschneiden
- PANNs-Aktivierung: Vocals confidence ≥ 0.40 (Speech: ≥ 0.35)

---

## 🧪 Test-Standards (§5.1)

```python
# Datei: tests/unit/test_v<X>_<feature_name>.py
import numpy as np
import math
import pytest

np.random.seed(42)  # Reproduzierbarkeit Pflicht

class TestMeinModul:
    def test_01_output_shape(self): ...          # Shape/Dtype korrekt
    def test_02_no_nan_output(self): ...         # np.isfinite(result).all()
    def test_03_no_clipping(self): ...           # np.max(np.abs(result)) <= 1.0
    def test_04_mono_input(self): ...            # 1D Audio
    def test_05_stereo_input(self): ...          # 2D Audio
    def test_06_silence_input(self): ...         # np.zeros(...)
    def test_07_noise_input(self): ...           # np.random.randn(...)
    def test_08_dirac_input(self): ...           # Impuls-Signal
    def test_09_bounds(self): ...                # Metriken ∈ [0,1] oder [1,5]
    def test_10_consistency(self): ...           # Selbe Eingabe → selbe Ausgabe
    def test_11_singleton_thread_safe(self): ... # concurrent.futures, 20 Threads
    def test_12_finite_all_scores(self): ...     # math.isfinite(score) für alle Felder
    # ... ≥ 35 Tests (neue Kernmodule), ≥ 20 (neue Phasen / Plugins)
```

**Mindestanforderungen:** ≥ 35 Tests (Kernmodule) / ≥ 20 (Phasen, Plugins). Alle Tests mit synthetischen Signalen — keine echten Audiodateien. `--timeout=30` via pytest.ini.

---

## ⚡ Singleton-Pattern (Pflicht — §3.2)

```python
# core/mein_modul.py
import threading
from typing import Optional

_instance: Optional["MeinModul"] = None
_lock = threading.Lock()

def get_mein_modul() -> "MeinModul":
    """Thread-sicherer Singleton (Double-Checked Locking)."""
    global _instance
    if _instance is None:          # Schnellpfad ohne Lock
        with _lock:
            if _instance is None:  # Zweiter Check unter Lock
                _instance = MeinModul()
    return _instance

def meine_convenience_funktion(audio: np.ndarray, sr: int) -> "MeinResult":
    """Convenience-Wrapper. sr MUSS 48000 sein."""
    assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
    return get_mein_modul().process(audio, sr)
```

**Kein `_cache = {}` ohne `threading.Lock()` in Produktionscode.**

---

## 🐛 Häufige Fallstricke

### Fallstrick #1: Fehlender SR-Assert

❌ Phase verarbeitet Audio mit falscher SR → subtile Qualitätsfehler
✅ `assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"`

### Fallstrick #2: NaN aus numerischen Operationen

❌ `result = np.fft.rfft(audio)` ohne NaN-Guard
✅ `result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)` nach jeder FFT

### Fallstrick #3: Kein Musical-Goals-Check (15 Ziele, nicht 7!)

❌ Neue Phase implementiert, Musical Goals nicht geprüft → Bug unbemerkt
✅ `MusicalGoalsChecker.measure_all()` nach jeder neuen Verarbeitungsoperation im Test

### Fallstrick #4: `print()` statt `logger`

❌ `print(f"Score: {score}")` in Produktionscode
✅ `logger.info("Score: %.2f", score)` mit `logger = logging.getLogger(__name__)`

### Fallstrick #5: dict statt dataclass als Rückgabe

❌ `return {"mos": 3.5, "nsim": 0.7}` aus öffentlichen Funktionen
✅ `return PQSResult(mos=3.5, nsim=0.7)` (immer `@dataclass`)

### Fallstrick #6: Hardcodierte Pfade

❌ `open("/home/user/.aurik/gp_memory/vinyl.json")`
✅ `pathlib.Path.home() / ".aurik" / "gp_memory" / f"{material}.json"`

### Fallstrick #7: Singleton ohne Lock (Race Condition bei Batch-Jobs)

❌ `_cache = {}` ohne `threading.Lock()` in Produktionscode
✅ Double-Checked Locking (§3.2) — immer `with _lock:` zweiten Guard

### Fallstrick #8: Sprach-Metriken für Musik verwenden

❌ `pesq(ref, deg, 16000, "wb")` oder `dnsmos(audio)` für Musikbewertung
✅ `PerceptualQualityScorer.score_audio_absolute(audio, sr)` (PQS-MOS)

### Fallstrick #9: SOFT_SATURATION reparieren

❌ phase_23 auf Material mit `SOFT_SATURATION`-Defekt anwenden → Wärme zerstört
✅ `classify_clipping()` zuerst — SOFT_SATURATION (gerade Obertöne) = **BEWAHREN**

### Fallstrick #10: Kein DSP-Fallback für ML-Plugin

❌ Plugin importiert ML-Modell ohne `try/except ImportError`
✅ `try: import onnxruntime ... except (ImportError, FileNotFoundError): dsp_fallback()`

---

## 📊 Architektur: Schichtentrennung (bindend — §11)

```
Frontend (PyQt5, frontend/)          ← KEIN direkter Core-Import!
    │ Qt-Signals / FastAPI HTTP
    ▼
API-Schicht (backend/api/rest/)
    │ Python-Aufruf
    ▼
Kognitive Orchestrierung (denker/)
    │ Python-Aufruf
    ▼
Backend-Core (core/ · plugins/ · dsp/)
```

**Verboten:**

- Direktaufruf von `core/`, `dsp/` oder `plugins/` aus `frontend/`-Code
- Netzwerkaufruf nach außen — Desktop-only, 100 % offline
- `torch.cuda` — CPU + optionale AMD-GPU (ROCM/DirectML)
- `print()` in Produktionscode

---

## 🔢 Logging-Konventionen (§3.5)

```python
import logging
logger = logging.getLogger(__name__)

# Log-Meldungen in ENGLISCH (nicht für Nutzer sichtbar):
logger.info("CausalReasoner: cause=%s confidence=%.2f", cause, conf)
logger.info("GP optimizer: material=%s source=%s", material, source)
logger.info("PQS score: MOS=%.2f NSIM=%.3f MCD=%.1f dB", mos, nsim, mcd)
logger.debug("Likelihood P(O|K=%s) = %.4f", cause, likelihood)

# Nutzer-Meldungen in DEUTSCH (via UI-Callback oder CLI-Output):
# "Restaurierung abgeschlossen — Qualität: Sehr gut (MOS 4.3)"
```

**Sprachkonvention:** Log-Meldungen englisch · Nutzer-UI deutsch · Code-Kommentare englisch.  
**Kein `print()` im Produktionscode — ausnahmslos.**

---

## 📐 Psychoakustische Fundierung (§4.1, §4.4)

Jede neue DSP-Funktion MUSS auf mindestens einem dieser Prinzipien basieren:

| Konzept | Anwendung |
| --- | --- |
| Bark-Skala / Critical Bands | Frequenzband-Gewichtung, spezifische Lautheit |
| ERB / Gammatone-Filterbank | PQS, PerceptualQualityScorer |
| LUFS / ITU-R BS.1770-5 | Lautstärke-Normierung (−14 LUFS EBU R128) |
| HPSS (Fitzgerald 2010) | Harmonisch/Perkussiv-Trennung (TDP) |
| OMLSA / IMCRA (Cohen 2002/2003) | Rauschunterdrückung (Primär-Algorithmus) |
| NMF mit β-Divergenz | Spektrale Dekomposition, Inpainting |
| pYIN / CREPE | Pitch-Tracking f₀ (kein YIN!) |
| Chroma / CQT | Tonale Analyse, TonalCenterMetric |
| NSIM / SSIM | Strukturelle Ähnlichkeit, Qualitätsbewertung |
| PGHI (Perraudin 2013) | Phasenkonsistenz nach Spektralmodifikation |
| GP/UCB + MOO Pareto | Parameteroptimierung (14 Objectives) |
| Bayesianische Kausalinferenz | Defektursachen-Erkennung (62 Ursachen) |
| ISO 226:2003 Equal-Loudness | BrillanzMetric + WaermeMetric-Gewichtung |
| Virtual Pitch / Missing Fundamental | BassKraftMetric (Moore et al. 2006) |

**ABSOLUT VERBOTEN für Musik-Qualitätsbewertung:** PESQ · DNSMOS · NISQA · STOI · ViSQOL `--speech`

---

---

## 📊 Neue Module seit 10.0.8 — Schnellreferenz

| Modul | Position in Pipeline | Messbarer Effekt |
| --- | --- | --- |
| `TransientDecoupledProcessing` | Allererster Schritt | GrooveMetric +0.03–0.06 |
| `HarmonicPreservationGuard` | Vor phase_03/phase_29 | Natürlichkeit +0.03–0.07 |
| `PerPhaseMusicalGoalsGate` | Wraps jede Phase | Kein kumulativer Qualitätsverlust |
| `MicroDynamicsEnvelopeMorphing` | Letzter Schritt vor Export | MicroDynamics Pearson 0.88 → 0.93+ |
| `EraClassifier` | Nach RestorabilityEstimator | GP-Warmstart per Epoche |
| `GermanSchlagerClassifier` | Nach EraClassifier | SCHLAGER_RESTORATION_PROFILE aktiv |
| `RestorabilityEstimator` | Vor EraClassifier | Score 0–100 + predicted MOS in < 5 s |
| `StemRemixBalancer` | Nach Stem-Verarbeitung | LUFS-Drift ≤ 0.3 LU garantiert |
| `IntroducedArtifactDetector` | Nach FeedbackChain | ML-Halluzinationen + NMF-Klicks erkannt |
| `AdaptiveGoalThresholds` | Nach RestorabilityEstimator | Schwellwerte = physikalisch erreichbar |
| `GoalApplicabilityFilter` | Einmal pro Restaurierung | Physikalisch irrelevante Ziele deaktiviert |
| `PhysicalCeilingEstimator` | Vor FeedbackChain | Optimierungsabbruch an Shannon-Grenze |
| `GoalPriorityProtocol` | FeedbackChain + ExcellenceOptimizer | Pareto-Konflikt Stufe 1/2 hat Vorrang |
| `EraAuthenticPerceptualCompletion` | Nach phase_55, vor IAD | Era-authentische HF-Ergänzung |

---

## ✅ Neue-Feature-Checkliste (§9.1)

```
□ Thread-sicheres Singleton: threading.Lock() + Double-Checked Locking
□ Alle öffentlichen Funktionen: vollständige PEP 484 Type-Annotations
□ NaN/Inf-Schutz in JEDER numerischen Ausgabefunktion
□ Ergebnisse als @dataclass (kein raw dict)
□ Convenience-Funktion vorhanden
□ logger = logging.getLogger(__name__)
□ ≥ 35 Unit-Tests (Kernmodule) / ≥ 20 (Phasen / Plugins)
□ Test: Shape, NaN, Bounds, Edge-Cases, Mono, Stereo, Konsistenz, Thread-Safe
□ Alle 15 Musical Goals nach neuer Funktion nicht schlechter
□ GrooveMetric: kein Timing-Flattening (DTW ≤ 8 ms RMS)
□ SOFT_SATURATION nicht als CLIPPING behandelt
□ Beide Modi (Restoration + Studio 2026) getestet
□ Eintrag in CAUSE_TO_PHASES (falls neue Phase)
□ Eintrag in models/manifest.json (falls neues ML-Modell, mit sha256)
□ DSP-Fallback für jedes Plugin (try/except ImportError)
□ assert sample_rate == 48000 in jeder Phase und jedem Plugin
□ np.clip(audio, -1.0, 1.0) vor jedem Ausgabe-Audio
□ Kein print() — nur logger.*()
□ Keine hardcodierten Pfade — pathlib.Path.home() / ".aurik" / ...
□ Alle bestehenden Tests weiterhin grün
```

---

**KI-Agent Integration Guide — Aurik 10.0.8 — Juli 2026**
**Bindend für: GitHub Copilot, Claude, GPT-Instanzen**
