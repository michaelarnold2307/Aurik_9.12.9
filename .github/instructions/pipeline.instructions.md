---
applyTo: "backend/core/unified_restorer_v3.py"
---

# UV3 — Pipeline-Regeln (normativ, Aurik 10.0.0.x)

## §2.31 Material-Phase-Initialstärken — Transfer-Chain-Aware [RELEASE_MUST v10.0.0]

**Problem**: Wenn `_restoration_context["transfer_chain"]` mehrere Stufen enthält (z.B. `["vinyl", "cassette"]`), darf nur die **strengste** Schwächungsstufe über alle Kettenglieder die Initialstärke einer Phase bestimmen.

```python
# KANONISCH — UV3 restore(), §2.31:
_mat_val = canonical_material_key(material_type)            # primäres Material
_chain_mat_vals = [canonical_material_key(s) for s in _cal_transfer_chain or []]
_material_factor_keys = list(dict.fromkeys([_mat_val] + _chain_mat_vals))

for _pid in all_phase_ids:
    _mat_s = min(_get_mat_strength(_fk, _pid) for _fk in _material_factor_keys)
    _material_phase_initial_strengths[_pid] = _mat_s

# VERBOTEN: nur _mat_val prüfen, Chain-Stufen ignorieren
# → Cassette-Material erbt dann Vinyl-Defaults → HF-Halluzination

# Logging:
logger.info("§2.31 Material-Phase-Initialstärken: %d Phasen für material=%s chain=%s",
            len(_material_phase_initial_strengths), _mat_val, _material_factor_keys)
```

**INVARIANTE**: `_MATERIAL_PHASE_FACTORS` in `defect_phase_mapper.py` MUSS für jeden möglichen Chain-Materialschlüssel einen Eintrag haben. Fehlt ein Key, fällt `_get_mat_strength()` auf Generic-Defaults zurück → zu hohe Stärke für restriktive Materialien.

## §2.44 Holistic Perceptual Index (HPI) — letztes Export-Gate

Quelle: `[SRC:S06,S07,S08,S09,S10,S11]`

Evidenzklassen-Hinweis (operativ):

- Klasse A: Loudness/TruePeak-Regeln (`[SRC:S06,S07]`) sind normgebunden.
- Klasse B: AFG-/Vocal-Detektorgrenzen (`[SRC:S14,S15,S16,S17,S18]`) erfordern peer-reviewte Evidenz + Regressionen.
- Klasse C: Kalibrierte Floors (z. B. materialadaptive Proxy-Werte) duerfen nur mit Revalidierungsnachweis angepasst werden.

```python
# KANONISCH — Restoration (Instrumental):
HPI = MERT_similarity * timbral_fidelity * artifact_freedom * emotional_arc_preservation

# KANONISCH — Restoration (Vokal, panns_singing >= 0.35) [§0p Pflicht]:
HPI = MERT_similarity * timbral_fidelity * VQI * artifact_freedom * emotional_arc_preservation

# KANONISCH — Studio 2026 (Instrumental):
HPI = studio_quality_gain * PQS_improvement * artifact_freedom * emotional_arc_preservation

# KANONISCH — Studio 2026 (Vokal, panns_singing >= 0.35) [§0p Pflicht]:
HPI = studio_quality_gain * PQS_improvement * VQI * artifact_freedom * emotional_arc_preservation

# MERT-Floor PFLICHT (Bug-Fix v10.0.0):
MERT_similarity = max(raw_mert, 0.5)  # verhindert Gesamt-Kollaps auf 0

# carrier_chain_recovery_ratio — KANONISCHE BERECHNUNG (Pflicht vor HPI):
# = Anteil der Phasen 1-4 (Carrier-Stufen), deren phase_score > 0.05:
# n_active_carrier = len([p for p in carrier_phases if metadata["phase_scores"][p] > 0.05])
# n_total_carrier = len(carrier_phases)  # phase_04/09/12/29 etc.
# carrier_chain_recovery_ratio = n_active_carrier / max(n_total_carrier, 1)
# MUSS in metadata["carrier_chain_recovery_ratio"] gesetzt werden (UV3, nach Stufe 4)

# studio_quality_gain — Studio 2026 Qualitätsfaktor:
# = VERSA(composite_score_restored) / VERSA(composite_score_input)  — normiert auf [0, 1]
# > 1.0 → geklippt auf 1.0 (Gain kann nicht > 1 sein im normierten Score)
# studio_quality_gain = min(versa_restored / max(versa_input, 0.01), 1.0)

# PQS_improvement — Perceptual Quality Score-Steigerung (Studio 2026):
# = DNSMOS(restored)["ovr"] / DNSMOS(input)["ovr"] — normiert auf [0, 1]
# PQS_improvement = min(dnsmos_ovr_restored / max(dnsmos_ovr_input, 1.0), 1.0)
# Beide Faktoren nur für Studio 2026 berechnen — nicht in Restoration!

# Primärer Veto-Faktor (artifact_freedom) + Recovery-Trigger (VQI):
if artifact_freedom < 0.95:
    return _recovery_cascade("artifact_freedom < 0.95", audio)  # KEIN Export
# material_vqi_floor: material-adaptiv aus calibration_matrix — VERBOTEN: hardcodierte Konstante
# material_vqi_floor = calibration_matrix.get_material_floor(material, "vqi")
# Shellac: 0.62 | Vinyl: 0.72 | CD/Digital: 0.82 | unknown_analog: 0.72
if panns_singing >= 0.35 and vqi < material_vqi_floor:   # VERBOTEN: vqi < 0.72 hardcoded
    return _recovery_cascade("vqi < material_floor", audio)  # Recovery-Trigger
**VERSA-Primärpflicht**: `use_versa_in_loop=True`. MERT nur Fallback → `metadata["mert_proxy_used"] = True`. `[SRC:S07]`

## §0d Vollständiges Referenz-Paradoxon-Handling (Lücke 1 — ALLE HPI-Faktoren)

```python
# Bei Carrier-Chain-Inversion klingt das restaurierte Signal INTENTIONAL anders
# als der degradierte Input → Ähnlichkeit gegen den Input ist kein Qualitätsmaß!

if carrier_chain_recovery_ratio > 0.15:
    # ALLE drei HPI-Referenzwerte auf best_carrier_checkpoint umstellen:
    # 1. timbral_fidelity → spektraler Vergleich gegen best_carrier_checkpoint
    timbral_fidelity = compute_timbral_fidelity(best_carrier_checkpoint, restored)

    # 2. MERT_similarity → MERT(best_carrier_checkpoint, restored)
    #    VERBOTEN: MERT(degraded_input, restored) — würde gute Restaurierung bestrafen
    raw_mert = mert_similarity(best_carrier_checkpoint, restored)
    MERT_similarity = max(raw_mert, 0.5)

    # 3. emotional_arc → Envelope-Korrelation gegen best_carrier_checkpoint
    emo_arc = compute_emotional_arc_preservation(
        best_carrier_checkpoint, restored, sr, frisson_zones=frisson_zones
    )

    metadata["hpi_reference"] = "best_carrier_checkpoint"
else:
    metadata["hpi_reference"] = "degraded_input"  # Standard-Pfad

# VERBOTEN: degraded_input als Referenz wenn carrier_chain_recovery_ratio > 0.15
```

## §2.45b Hochrestorabilität-Gate

```python
if restorability_score > 80 and snr_db > 40:
    metadata["high_restorability_gate"] = True
    # Phasen mit defect_severity < 0.05 → überspringen
    # Strength für _NEVER_SKIP-Phasen auf Restorability-adaptiven Minimalwert senken
```

## §2.47a PreAnalysis-Handover + MediumDetector-Konfidenz-Fallback (Lücke 4)

- `run_pre_analysis()` **genau 1×** nach Import
- `MediumDetector.detect()` **genau 1×** — nie nochmals auf restauriertem Audio
- Neuer File-Import → Cache **HARD** löschen

```python
# MediumDetector Konfidenz-Schwellen (RELEASE_MUST):
detect_result = MediumDetector.detect(audio, sr)
material_confidence = detect_result.confidence  # [0, 1]

if material_confidence >= 0.75:
    material = detect_result.material            # volles material-adaptives Processing
elif material_confidence >= 0.50:
    # Konservative Böden: Vinyl als Fallback-Material
    material = "vinyl"  # weniger aggressiv als Shellac-spezifisch
    metadata["material_fallback"] = f"confidence={material_confidence:.2f} → vinyl"
else:
    # Zu unsicher für material-adaptives Processing
    material = "unknown_analog"  # Universal-Fallback
    # Universal-Fallback-Defaults (sicher für alle analogen Träger):
    #   BW-Ceiling: 16 kHz
    #   DR-Ceiling: 70 dB
    #   VQI-Boden:  0.72 (Vinyl-Niveau)
    metadata["material_fallback"] = f"confidence={material_confidence:.2f} → unknown_analog"
    metadata["material_low_confidence_warning"] = True

# IMMER in metadata schreiben:
metadata["material_detected"] = detect_result.material
metadata["material_confidence"] = material_confidence
metadata["material_used"] = material  # kann von detected abweichen
```

- **Finaler Wert für die Planung**: Der an UV3/§v10.303 (Planer-Intelligenz,
  Low-Confidence-Gate) übergebene `material_confidence`-Wert ist die FINALE
  MediumDetector-Konfidenz NACH der bayesianischen Verfeinerung in
  `run_pre_analysis()`. Der zum Defekt-Scan-Zeitpunkt eingefrorene
  Forensic-Wert darf die Phasen-SELEKTION nicht steuern (Befund 2026-08-22:
  0.307 eingefroren vs. 0.482 final → 17 Phasen fälschlich gestrichen).
  Implementiert in `unified_restorer_v3.py` („§2.47a material_confidence-
  Aktualisierung“).
- **Material-Konsens-Vote (§v10.20, Befund 2026-08-22)**: Der
  MediumDetector-Vote in `resolve_material_consensus()` ist der PRIMÄRE
  Träger (`chain[0]`, z. B. vinyl) — nie der terminale Träger
  (`chain[-1]`, z. B. mp3_low). Der terminale Träger ist ein
  KETTEN-Attribut, kein Material-Ergebnis des Detektors. Zusätzlich ist
  `tape` Teil der chronologischen Kettenordnung (vor `cassette`), damit
  Konflikt-Korrekturen nicht terminal-first sortieren (Befund:
  „Kette KORRIGIERT: mp3_low → tape“).
- **ML-Scorer-Längen-Cap (§9 Performance-Budget, Befund 2026-08-22)**:
  Alle ML-Qualitäts-Scorer (FeedbackChain VERSA/PQS, MUSHRA-Proxy/MERT,
  HPE/Goosebumps) bewerten lange Signale (> 90 s bzw. > 60 s) auf
  deterministischen Fenstern (3×30 s Anfang/Mitte/Ende bzw. 30-s-Mittel-
  Ausschnitt) statt auf der Voll-Länge — gleicher Input ⇒ gleiche Fenster
  ⇒ bit-identische Scores (§G5 (copilot-instructions.md)). Kein Skip mehr:
  Die Wohlklang-Baseline fällt bei langen Songs nicht mehr auf Konstante
  0.5 zurück (Befund: 224 s → 37.3 s pro Aufruf, Iterations-Budget
  erschöpft).
- **Ein kanonischer Defect-Scan (§v10.702 B3-P2, Befund 2026-08-22)**:
  Liegt der pre_Analyse-Scan vor (`cached_defect_result`, vom Denker
  durchgereicht), übernimmt B3-Phase-2 dessen Defekt-Typen
  deterministisch — statt eines zweiten Full-Song-Scans mit abweichender
  Samplerate und inkonsistenten Thresholds (Befund: 95 s @48k + 77 s
  @44.1k). `scan_defect_presence` bleibt Ersatzpfad ohne Cache.
- **PIM-first erhält die Song-Struktur (§2.52b, Befund 2026-08-22)**:
  Der UV3-PIM-Aufruf reicht die `SongStructureAnalyzer`-Segmente an
  `compute_intensity_map()` durch — der Mapper lief zuvor ohne Kontext
  („10 Bänder × 1 Segmente — 0 Entscheidungen“), die per-Segment-
  Intensität war de facto inert.
- **Material-Konsens-Write-back (§v10.20, 2026-08-22)**: Der Konsens wird
  als EIGENE Felder auf dem MediumDetectionResult persistiert
  (`consensus_material`, `final_chain`) — `primary_material` bleibt der
  Medium-Primär (Kalibrierungs-Quelle). UV3 konsumiert bei vorhandenem
  Konsens dessen Kette und überspringt die Era-Dominanz-Regel — kein
  Flip-Flop vinyl→tape→vinyl mehr.
- **Session-Memory-Hygiene (§Sitzung-MEMORY, 2026-08-22)**: Nur Läufe mit
  Verdikt `improved` werden persistiert; `unchanged`/degradierte Läufe
  werden nicht als Erfahrung gespeichert — das Gedächtnis lernt nicht aus
  Fehlentscheidungen.

## §2.48 Kumulative-Phasen-Interaktions-Guard

```python
# VERBOTEN: feste Konstante
tolerance = 0.15  # FALSCH

# RICHTIG:
tolerance = compute_adaptive_drift_tolerance(
    restorability, material, severity, n_phases
)
# Carrier-Repair-Phasen (_CARRIER_REPAIR_PHASE_PREFIXES) inkrementieren
# consecutive_rollbacks NICHT

# KANONISCHE DEFINITION _CARRIER_REPAIR_PHASE_PREFIXES (alle Stufen 1–4 aus §2.46):
_CARRIER_REPAIR_PHASE_PREFIXES = {
    "phase_30",   # Stufe 1: DC-Offset
    "phase_31",   # Stufe 1: Quantisierungsrauschen
    "phase_04",   # Stufe 2: RIAA-EQ
    "phase_25",   # Stufe 2: Azimuth-Korrektur
    "phase_12",   # Stufe 2: Wow/Flutter
    "phase_09",   # Stufe 3: Crackle/Knistern
    "phase_24",   # Stufe 3: Dropout-Repair
    "phase_03",   # Stufe 4: Surface-Noise-NR
    "phase_29",   # Stufe 4: Bandrauschen/Tape-NR
}
# VERBOTEN: phase_05/06/07/23 (Stufe 5 — ADDITIV) in dieses Set — sie inkrementieren consecutive_rollbacks
```

## §2.49 Artefakt-Freiheits-Gate

Quelle: `[SRC:S03,S04,S12,S13]`

```python
# artifact_freedom = clip(1.0 - (weighted_sum / _max_tolerance) + penalties, 0, 1)
# Implementierung: backend/core/artifact_freedom_gate.py
# KEIN Pipeline-Delta — absolute Score-Berechnung gegen Original-Audio
# NICHT min()-Formel — weighted-sum ist präziser (ein schlechter Wert dominiert)
#
# SCHRITT 1 — Artefakt-Typ-Gewichte (_TYPE_WEIGHTS):
#   musical_noise:      1.0  (STFT-Bins wo restored > orig * 1.05)
#   phase_cancellation: 1.0  (Kreuz-Korrelation < -0.5 über > 10 % Frames)
#   crackle_impulse:    1.1  (Impulsnoise — besonders salient für Hörer)
#   metallic_ringing:   0.9  (Pre/Post-Transient-Energie außerhalb Grenzen)
#   pre_echo:           0.8  (zeitliches Prä-Masking-Artefakt)
#   spectral_hole:      0.6  (Frequenzlücken durch Over-Suppression)
#
# SCHRITT 2 — Salienz-Gewichtung pro Artefakt-Instanz:
#   salience_weighted_score = base_score × salienz × temporal_masking_weight
#   temporal_masking_weight ∈ [0,1]: 1.0 = kein Masking, < 1.0 = psychoakust. Maskierung
#
# SCHRITT 3 — weighted_sum = Σ(_TYPE_WEIGHTS[type] × salience_weighted_score × masking)
#
# SCHRITT 4 — _max_tolerance (material-adaptiv, §2.54):
#   digital/cd:    5.0  vinyl: 6.5  shellac/wax: 7.5  tape: 6.2  mp3_low: 6.0
#   Restorative-Bonus (§0d): _max_tolerance × (1.0 + max(0, (80-restorability)/80) × 0.8)
#   → stark degradiertes Material bekommt bis ×1.8 Toleranz
#
# SCHRITT 5 — Endformel:
#   artifact_freedom = clip(1.0 - (weighted_sum / _max_tolerance), 0, 1)
#   artifact_freedom += noise_penalty      # negativ: globales Rausch-Niveau
#   artifact_freedom += roughness_penalty  # negativ: §2.49c Rauhigkeit/Schärfe
#   artifact_freedom = clip(artifact_freedom, 0.0, 1.0)
#
# artifact_freedom < 0.95 → Gate-Fail (primärer VETO-Faktor §0h — absolut, kein Override) [SRC:S03,S04]
# metadata["weighted_artifact_sum"] und metadata["max_tolerance"] werden gepflegt
```

## §2.49a Zentrale Gate-Klassifikation (A/B/C) [RELEASE_MUST]

Die finale Endentscheidung MUSS zentral in UV3 klassifiziert werden. Einzelregeln
duerfen nicht als ungeordnete Sonderfaelle konkurrieren.

```python
# KANONISCH: _classify_quality_gate_events(...)
# Klasse A: HARD_VETO
#   - artifact_freedom < 0.95
# Klasse B: RECOVERY_TRIGGER
#   - vqi < material_vqi_floor
#   - singer_identity_cosine < 0.92 (wenn multi_singer=False UND singer_id_dsp_fallback=False)
#     §DSP-Fallback-Guard (v10.0.0): singer_id_dsp_fallback=True → Advisory only (Class C)
# Klasse C: ADVISORY
#   - vqi < mode_target (Restoration 0.82 / Studio 0.87)
#   - singer_identity_cosine < 0.92 bei singer_id_dsp_fallback=True → singer_id_advisory_cosine

quality_gate_registry = {
    "events": [...],
    "hard_veto": bool(...),
    "hard_reasons": [...],
    "recovery_triggers": [...],
    "advisories": [...],
}

metadata["quality_gate_registry"] = quality_gate_registry
```

VERBOTEN:

- Gleichwertige Behandlung von Hard-Veto und Recovery-Trigger ohne Klassenordnung.
- Verstreute End-Gate-Entscheidungen ohne zentrale Registry.

## §2.49b Recovery-Zustandsautomat [RELEASE_MUST]

Nach der Gate-Klassifikation MUSS ein deterministischer Zustandsautomat den
Exportpfad beschreiben.

```python
# KANONISCH: _resolve_recovery_state_machine(...)
# Reihenfolge:
# RUN -> SOFT_REWEIGHT -> ROLLBACK -> FALLBACK -> DEGRADED_EXPORT

recovery_state_machine = {
    "state": "RUN|SOFT_REWEIGHT|ROLLBACK|FALLBACK|DEGRADED_EXPORT",
    "trace": [...],
    "conflict_free": bool,
    "semantic_conflicts": [...],
}

metadata["recovery_state_machine"] = recovery_state_machine
```

Hinweise:

- `DEGRADED_EXPORT` ist Pflicht, sobald `degradation_status in {"degraded", "failed"}`
  oder ein `fail_reason.severity in {"degraded", "failed"}` vorliegt.
- Der Zustandsautomat ersetzt keine Guards; er standardisiert die Endentscheidung.

## §2.49c Semantik-Konfliktregel Hard vs Recovery [RELEASE_MUST]

Ein Trigger darf nicht gleichzeitig als Hard-Veto und Recovery-Trigger klassifiziert
werden.

```python
overlap = set(hard_reasons) & set(recovery_triggers)
if overlap:
    metadata["recovery_state_machine"]["conflict_free"] = False
    metadata["recovery_state_machine"]["semantic_conflicts"] = sorted(overlap)
    logger.warning("Recovery-State-Machine Konflikt: %s", ", ".join(sorted(overlap)))
```

VERBOTEN:

- Doppelte Trigger-Semantik (gleiches `event.id` in Hard und Recovery).
- Silent-Fail ohne Telemetrieeintrag zu Konflikten.

## §2.48b Umgewichtung vor Rollback [RELEASE_MUST]

Wenn eine Phase in PMGG/CIG/PDV an der Grenze scheitert, MUSS vor einem harten Rollback
ein konservativer Umgewichtungsversuch erfolgen (z. B. Dry/Wet-Reweight oder Blend zur
pre-phase Referenz), damit die Phase nicht sofort vollstaendig verwirkt wird.

Ausnahme (harte Guards):

- natuerlichkeit-Hard-Guard verletzt
- artifact_freedom < 0.95
- explizite Export-Sicherheitsverletzung

In diesen Faellen bleibt sofortiger Rollback/Veto verpflichtend.

## §8.6g-II/III Psychoakustische Anti-Klinik-Rueckkopplung [RELEASE_MUST]

```python
# End-Gate Recovery Pflicht:
# Wenn psychoacoustic_naturalness_gate FAIL, MUSS vor finaler Degradation
# ein konservativer Recovery-Versuch laufen (Blend mit sicheren Referenzen):
# - hpi_best_checkpoint
# - best_carrier_checkpoint
# - original_audio
# Uebernahme nur wenn psychoacoustic_naturalness_gate danach PASS ist.

# Phasenweise Rueckkopplung Pflicht:
# Vor phase.process() MUSS ein psychoakustischer Strength-Scalar berechnet
# werden und nur daempfend wirken (<= 1.0).

# Runtime-Delta-Loop Pflicht:
# Negative Per-Phase-Delten fuer natuerlichkeit/authentizitaet/
# emotionalitaet/micro_dynamics/transparenz akkumulieren und als
# _psycho_runtime_state in den naechsten Scalar einspeisen.

# Pflicht-Telemetrie:
# metadata["psychoacoustic_feedback_recovery"]
# phase_meta["psycho_strength_scalar"]
# phase_meta["psycho_strength_risk_score"]
# phase_meta["psycho_strength_signals"]
# _phase_metadata_accumulator["_psycho_runtime_state"]
```

VERBOTEN:

- Psycho-Scalar > 1.0 (aggressiver machen) als Teil dieses Loops.
- Recovery-Blend ohne nachgelagerten Gate-Recheck.
- Runtime-Delta-State nicht in Metadata schreiben (unsichtbare Steuerlogik).

## §1.2c Teamwork-Invariante fuer 15 Musical Goals [RELEASE_MUST]

Jede Phase traegt mit phasenspezifischer, adaptiv berechneter Staerke zur Zielerreichung bei.
Kein Einzelziel und keine Einzelphase darf die Endentscheidung dominieren.

Pflichtregeln in UV3-Endgate:

```python
# 1) Vollstaendiger Zielvektor (15 Goals) ist Pflicht:
goal_vector_keys = sorted(effective_goal_thresholds.keys())
missing = [g for g in goal_vector_keys if g not in musical_goal_scores]
if missing:
    for g in missing:
        musical_goal_scores[g] = 0.0  # fail-safe: fehlende Messung = nicht bestanden

# 2) Pass/Fail muss immer gegen den vollstaendigen 15er-Vektor laufen,
#    nicht nur gegen Keys, die zufaellig im Score-Dict auftauchen.
musical_goals_passed = {
    g: (True if g not in applicable_goals else musical_goal_scores[g] >= effective_goal_thresholds[g])
    for g in goal_vector_keys
}

# 3) Recovery/Candidate-Ranking bleibt multi-goal (weighted-gap),
#    keine Dominanz eines Einzelgoals ueber alle anderen.
```

VERBOTEN:

- Verletzungszaehler auf `len(musical_goal_scores)` basieren lassen, wenn der 15er-Vektor groesser ist.
- Ein Goal durch fehlenden Score stillschweigend aus der Gate-Entscheidung entfernen.

## VERSA — Primäre Qualitäts-Metrik (`use_versa_in_loop=True`)

```python
# VERSA (Versatile Evaluation for Speech and Audio Restoration, 2024):
# Multi-dimensionale Metrik für Audio-Restaurierung:
# - MOS-Prediction (UTMOS-basiert): subjektive Qualität [1,5]
# - Naturalness: DNSMOS-P.835 SIG-Komponente
# - Fidelität: SI-SDR und LSD (Log-Spectral Distance) gegen Referenz
# - Artifact-Freiheit: Kreuz-Spektral-Analyse auf Musical Noise
# Ausgabe: composite_score [0,1] = gewichtetes Mittel aller Komponenten
# VERSA > MERT für Restoration (MERT ist Music-Representation — kein Restaurierungsmaß)!
from backend.core.dsp.quality_predictors import get_versa_predictor
versa_result = get_versa_predictor().evaluate(audio_orig, audio_restored, sr)
versa_score = versa_result["composite_score"]  # [0, 1]

# MERT (MERT-95M Backbone, Li et al. 2023) — nur Fallback wenn VERSA OOM:
# MERT misst semantische Ähnlichkeit im Music-Representation-Space (CQT-Embeddings)
# Cosine-Similarity zwischen Original- und Restaurierungs-Embeddings
# Fallback: metadata["mert_proxy_used"] = True; MERT_floor = max(raw_mert, 0.5)
```

## §2.78 Single-Source-Orchestrierung (Anti-Parallelwelt) [RELEASE_MUST]

**Source-Traceability**: `[SRC:T1,T2]`

**Ziel**: UV3, Goal-System und Phasen dürfen niemals konkurrierende Entscheidungen treffen.
Bei Konflikt gilt eine eindeutige Prioritätskette; nur **ein** Pfad darf den finalen Export steuern.

```python
# Verbindliche Entscheidungsreihenfolge (höchste Priorität zuerst):
# 1) Safety-Hard-Gates (artifact_freedom, Stereo-Hard-Fails, Export-Safety)
# 2) Vocal-Integrität (VQI-Recovery, Formant/Vibrato-Hard-Invarianten)
# 3) FeasibilityController (physikalische Erreichbarkeit pro Goal)
# 4) PMGG/CIG/AFG lokale Phase-Entscheidungen
# 5) MusicalGoals-Optimierung (15-Goal-Teamwork)
# 6) Reporting/Analytics (nie steuernd)

# VERBOTEN:
# - Zweiter unabhängiger Exportpfad außerhalb UV3-Hauptpfad
# - Goal-Schwellen in Phasenmodulen überschreiben
# - Recovery-Entscheidung in Reporting-Modulen treffen
```

**Pflicht-Metadaten** (für Drift-Erkennung):

```python
metadata["decision_authority"] = {
    "hard_gate": "UV3",
    "vocal_gate": "UV3",
    "feasibility": "FeasibilityController",
    "phase_local": "PMGG/CIG/AFG",
    "final_export": "UV3",
}
```

## §2.79 FeasibilityController für 15 Ziele [RELEASE_MUST]

**Source-Traceability**: `[SRC:T1,T3]`

**Wissenschaftliche Begründung**: Nicht jedes degradierte Signal erlaubt vollständige Zielerreichung
(Irreversibilität von Informationsverlust; siehe ITU-R BS.1116, ITU-R BS.1534, Zwicker/Fastl).
Deshalb muss der Controller vor der Zieljagd die song-spezifische Erreichbarkeit berechnen.

```python
# Vor _execute_pipeline() (nach SongCalibration + GoalApplicability):
feasibility = estimate_goal_feasibility(
    audio=audio,
    sr=sample_rate,
    material=material_type,
    restorability=restorability_score,
    transfer_chain=_cal_transfer_chain,
)

# Ergebnisformat:
# feasibility[goal] = {
#   "reachable": bool,
#   "confidence": float,   # [0,1]
#   "max_achievable": float,
# }

# Effektive Zielschwelle (niemals höher als max_achievable):
effective_goal_thresholds[goal] = min(
    effective_goal_thresholds[goal],
    feasibility[goal]["max_achievable"],
)

# Wenn Goal physikalisch nicht erreichbar: kein Hard-Fail, aber transparentes degraded-Flag.
if not feasibility[goal]["reachable"]:
    metadata.setdefault("goal_feasibility_limits", {})[goal] = dict(feasibility[goal])
```

**VERBOTEN**:

- Unerreichbare Ziele mit aggressiven Eingriffen erzwingen (führt zu unnatürlichem Klang).
- Feasibility-Ergebnis ignorieren und trotzdem harte Zielerfüllung erzwingen.

## §2.80 Transparenz-Objektiv "kein hörbarer Eingriff" [RELEASE_MUST]

**Source-Traceability**: `[SRC:T2,T4]`

Export-Selektion muss explizit das Ziel "klingt wie ohne Eingriff" optimieren.
Dies ist ein zusammengesetztes Soft-Objektiv, **kein** neues Hard-Gate.

```python
transparency_objective = weighted_mean({
    "artifact_freedom": artifact_freedom,                # hoch
    "timbral_fidelity": timbral_fidelity,                # hoch
    "emotional_arc_preservation": emotional_arc,         # mittel-hoch
    "micro_dynamics_preservation": micro_dyn_corr,       # mittel
})

# Kandidatenauswahl (current/hpi_best/best_carrier/original):
# Primär nach Hard-Gates filtern, dann höchsten transparency_objective wählen.
```

**Normativ**:

- Goosebumps/Frisson bleibt Soft-Faktor (Recovery + Score-Penalty), kein eigenständiger Hard-Veto.
- Reporting darf `degraded` markieren, aber keine autonome Export-Blockade auslösen.

## §2.81 Spec-Upgrade-Doktrin (Upgrade vor Downgrade) [RELEASE_MUST]

Wenn produktiver Code für denselben Scope nachweisbar bessere Restaurierungs-, Reparatur-
oder Rekonstruktionsqualität liefert, MUSS die Spezifikation auf dieses höhere Niveau
angehoben werden. Ein Rückbau auf schwächere Spezifikationsstände ist unzulässig.

```python
# Zulässige Richtung bei Konflikt Code vs. Spec:
# 1) Hard-Safety-Invarianten bleiben unveränderlich (artifact_freedom, Export-Safety, Vocal-No-Harm)
# 2) Innerhalb der Safety-Hülle gewinnt die nachweisbar bessere Klanglösung
# 3) Spezifikation wird auf den besseren Zustand aktualisiert (nicht umgekehrt)

if implementation_quality > spec_quality and safety_invariants_passed:
    action = "upgrade_spec"
else:
    action = "keep_or_fix_impl"

# VERBOTEN:
# - bessere Lösung abschwächen, nur um alte Spec-Formulierung zu erfüllen
# - neue Qualitätsgewinne ohne Spec-/Test-Traceability belassen
```

Pflicht-Traceability bei jedem Upgrade:

```python
metadata["spec_upgrade"] = {
    "section": "§2.81",
    "reason": "implementation_outperforms_previous_spec",
    "evidence": {
        "artifact_freedom_delta": delta_af,
        "vqi_delta": delta_vqi,
        "goal_vector_delta": goal_delta_summary,
    },
}
```

## Quellenmatrix für §2.78–§2.80

`[SRC:T1]` Subjektive Qualitätsbewertung und kleine Beeinträchtigungen (Ground Truth):

- ITU-R BS.1116-3 (2015), Methods for the subjective assessment of small impairments in audio systems: <https://www.itu.int/rec/R-REC-BS.1116>
- ITU-R BS.1534-3 (2015), MUSHRA method for intermediate audio quality: <https://www.itu.int/rec/R-REC-BS.1534>

`[SRC:T2]` Robuste, standardisierte Sicherheits-/Loudness- und Objektivmetriken:

- ITU-R BS.1770-5 (2023), Algorithms to measure audio programme loudness and true-peak audio level: <https://www.itu.int/rec/R-REC-BS.1770>
- EBU R128 (v5.0, 2023), Loudness normalisation and permitted maximum level: <https://tech.ebu.ch/publications/r128>
- ITU-R BS.1387-2 (2023), Method for objective measurements of perceived audio quality (PEAQ): <https://www.itu.int/rec/R-REC-BS.1387>

`[SRC:T3]` Feasibility-Limitierung / keine absolute Zielerfüllungsgarantie:

- ITU-R BS.1116-3: kleine, hörbare Beeinträchtigungen müssen subjektiv validiert werden; impliziert material-/programmabhängige Erreichbarkeitsgrenzen.
- ITU-R BS.1534-3: Qualitätsurteile sind signal- und kontextabhängig; globale Einzelmetrik ersetzt keine segment-/kontextbezogene Bewertung.

`[SRC:T4]` Frisson/Goosebumps als emotionaler, interindividuell variabler Marker (Soft-Faktor):

- Blood AJ, Zatorre RJ (2001), Intensely pleasurable responses to music correlate with activity in brain regions implicated in reward and emotion, PNAS, DOI: 10.1073/pnas.191355898
- Salimpoor VN et al. (2011), Anatomically distinct dopamine release during anticipation and experience of peak emotion to music, Nat Neurosci, DOI: 10.1038/nn.2726
- Grewe O et al. (2007), Chills in response to music: psychophysiological and neurophysiological findings, Proc. R. Soc. B, DOI: 10.1098/rspb.2006.3664

## emotional_arc_preservation — Messung

```python
# Emotional Arc = Energie-/Dynamik-Verlauf über Stück-Zeitachse:
# 1. RMS-Envelope (200 ms Fenster, 50 % Overlap) für Original + Restored
# 2. Pearson-Korrelation der Envelopes r ∈ [-1, 1]
# 3. Normierung: emotional_arc = max(0, r)  (negative Korrelation = 0)
# 4. Frisson-Zonen: innerhalb Frisson-Segmente Gewicht × 2.0 (Klimax-Passagen
#    prägen Emotionswahrnehmung stärker als ruhige Strophen)
# 5. Pegelexplosion-Detektion: wenn max(restored) > max(original) * 1.5 → score = 0
from backend.core.dsp.emotional_arc import compute_emotional_arc_preservation
emo_arc = compute_emotional_arc_preservation(audio_orig, audio_restored, sr,
                                              frisson_zones=frisson_zones)
# VERBOTEN: emotional_arc aus Spektral-Merkmalen approximieren (Envelope ist Ground Truth)

# frisson_zones FORMAT (FrissonZone-Dataclass aus backend/core/frisson_candidate_detector.py):
# @dataclass
# class FrissonZone:
#     start_s: float   — Zonen-Startzeit in Sekunden
#     end_s:   float   — Zonen-Endzeit in Sekunden (≥ start_s + 0.1)
#     score:   float   — kombinierter Frisson-Score [0.0, 1.0] (höher = stärker)
#     trigger: str     — dominanter akustischer Auslöser der den Score bestimmt
#
# Bezug: frisson_zones = get_frisson_detector().detect(audio, sr)  # List[FrissonZone]
# VERBOTEN: frisson_zones=None als gültiger Zustand (V16): immer [] wenn keine Zonen
# VERBOTEN: frisson_zones ohne .start_s/.end_s/.score/.trigger — getattr-Zugriff sichert ab
# Verwendung: _restoration_context["frisson_zones"] nach VocalFocusAnalyzer setzen;
#   alle Phasen + measure_emotional_arc() + correct_emotional_arc() erhalten es via kwargs
# Mindestscore-Filter: MIN_SCORE = 0.28 (FrissonCandidateDetector intern); max_zones = 20
```

## §2.51 Stereo — Hard-Fail-Invariante

```python
# §2.51a — drei Hard-Fails, alle sofort Recovery-Kaskade:
# VERBOTEN: assert (durch python -O deaktivierbar!)
# RICHTIG: if + logger.error + _recovery_cascade()
if interchannel_delay_ms > 1.0:
    logger.error("stereo_hard_fail interchannel_delay=%.2fms", interchannel_delay_ms)
    return _recovery_cascade("stereo_interchannel_delay", audio)
if lr_imbalance_db > 6.0:
    logger.error("stereo_hard_fail lr_imbalance=%.2fdB", lr_imbalance_db)
    return _recovery_cascade("stereo_lr_imbalance", audio)
if true_peak_dbtp > -1.0:
    logger.error("stereo_hard_fail true_peak=%.2fdBTP", true_peak_dbtp)
    return _recovery_cascade("stereo_true_peak", audio)

# VERBOTEN: unabhängiges L/R-Processing
# RICHTIG: M/S-Domain oder Linked-Stereo überall
```

## §2.52 PhaseConductor — _NEVER_SKIP

```python
_NEVER_SKIP = {
    "phase_01", "phase_09", "phase_12",
    "phase_14", "phase_15",
    "phase_30",  # DC-Offset — immer
    "phase_47",  # TruePeak-Limiter — immer
}
# Diese Phasen laufen immer — auch bei hoher Restorability und bei MAS-Early-Stop
```

Die adaptive Steuerung aller PhaseConductor-Entscheidungen, inklusive Gewichtung
für Strength-/Wet-Parameter, wird ausschließlich aus `restoration_policy_profile`
abgeleitet. `song_goal_weights` ist ein abgeleiteter Kompatibilitätswert, keine
eigenständige Normquelle.

Die Denker dürfen diese zentrale Policy ausdrücklich anreichern: `StrategieDenker`
liefert Eingriffsbudget, Hörziele und Human-Hearing-Risiken; `PhaseInteractionDenker`
liefert Goal-Risiken, Phasen-Dichte und Qualitäts-Tiers; `ReparaturDenker` und
`RekonstruktionsDenker` liefern Reparatur-/Plausibilitätsrisiken. Diese Signale
fließen als `denker_policy_input` in UV3 ein und werden dort in
`restoration_policy_profile` verdichtet. Sie dürfen nie als parallele Steuerquelle
neben dem Profil genutzt werden.

`StrategieDenker` MUSS zusätzlich ein songindividuelles
`human_hearing_comfort_profile` beitragen. Dieses Profil bündelt Peak-Komfort,
HF-/Air-Toleranz, Müdigkeitsrisiko, Transientenschutz, Mikrodynamikschutz und
Overprocessing-Risiko. UV3 und finale Guards dürfen diese Werte ausschließlich
aus `restoration_policy_profile["human_hearing_comfort_profile"]` lesen; direkte
Denker-Bypässe oder phasenlokale Zweitprofile sind verboten.

## §2.53b Determinismus

```python
# precomputed_phase_plan ist Source of Truth
# UV3 überspringt _select_phases() + _optimize_phase_plan_intelligence()
# wenn precomputed_phase_plan vorhanden
```

## §2.53c Kompressionsdefekt-Routing in _select_phases

```python
# RELEASE_MUST in UV3-Selektionslogik:
# Kompressionsdefekte müssen sowohl spektrale Reparatur als auch dynamische
# Entkompression triggern.

if sev(DefectType.COMPRESSION_ARTIFACTS) > 0.25:
    selected.append("phase_23_spectral_repair")

if (
    sev(DefectType.DYNAMIC_COMPRESSION_EXCESS) > 0.30
    or sev(DefectType.COMPRESSION_ARTIFACTS) > 0.25
):
    selected.append("phase_54_transparent_dynamics")

# VERBOTEN: Nur einen der beiden Pfade zu aktivieren.
# Phase 23 behandelt spektrale Flattening-/Residue-Produkte,
# Phase 54 behandelt Pumping/Envelope-Artefakte.
```

## §2.60 Rollback-Hierarchie + `_recovery_cascade()` Vollspezifikation (Lücke 2)

Quelle: `[SRC:S06,S07,S12,S13]`

```python
# _recovery_cascade() — vollständig spezifiziert, KEINE ad-hoc-Implementierung erlaubt:
#
# Aufruf-Kontext: HolisticPerceptualGate oder Phase-Delta-Guard
# Rückgabe: np.ndarray (bestes verfügbares Audio) — NIEMALS None
# Max-Gesamtzeit: 30 s für den gesamten Kaskaden-Durchlauf (Watchdog)
# Max-Retries pro Stufe: 2 (danach zur nächsten Stufe)

def _recovery_cascade(reason: str, audio_current: np.ndarray) -> np.ndarray:
    # audio_current: Audio zum Aufrufzeitpunkt — NUR für Diagnose/Metadata gespeichert.
    # Rollback-Quellen sind die internen UV3-Checkpoints (s.u.), NICHT audio_current.
    metadata["recovery_audio_snapshot_shape"] = audio_current.shape  # Diagnose, kein Rollback-Ziel
    metadata["recovery_attempts"] = metadata.get("recovery_attempts", 0) + 1
    metadata["recovery_reasons"].append(reason)  # Liste, nie überschreiben

    # Hilfsfunktionen (UV3-private, Implementierung in unified_restorer_v3.py):
    # _validate_audio(a): bool — True wenn a nicht NaN/Inf und shape[-1] > 0
    # _pmgg_check(a): float — PMGG-Score > 0 = Qualität OK; ≤ 0 = weiteres Rollback nötig
    # _can_retry_with_reduced_strength(): bool — True wenn aktuelle Phase retry-fähig ist
    #   (Phase nicht in Blacklist, retry_count < 2, Phase hat einen strength-Parameter)
    # _retry_phase_at_half_strength(): np.ndarray — führt aktuelle Phase mit strength × 0.5 aus

    # Stufe 1: Phase-Rollback + Score negativ markieren
    if _current_phase_audio is not None:
        metadata["phase_scores"][current_phase_id] = -1.0
        audio = _current_phase_input  # auf Phase-Eingang zurück
        if _validate_audio(audio) and _pmgg_check(audio) > 0:
            metadata["recovery_level"] = 1
            metadata["status"] = "recovered"
            return audio

    # Stufe 2: Strength-Reduktion 50 % → erneutes PMGG-Check
    if _can_retry_with_reduced_strength():
        audio = _retry_phase_at_half_strength()
        if _validate_audio(audio) and _pmgg_check(audio) > 0:
            metadata["recovery_level"] = 2
            metadata["status"] = "recovered"
            return audio

    # Stufe 3: best_carrier_checkpoint (nach Carrier-Stufen 1–4, vor Enhancement)
    if _best_carrier_checkpoint is not None:
        metadata["recovery_level"] = 3
        metadata["status"] = "recovered"
        return _best_carrier_checkpoint

    # Stufe 4: Pre-Pipeline-Checkpoint (nach TDP, vor allen Phasen)
    if _pre_pipeline_checkpoint is not None:
        metadata["recovery_level"] = 4
        metadata["status"] = "recovered"
        return _pre_pipeline_checkpoint

    # Stufe 5: Original-Input-Export — IMMER besser als Artefakt
    metadata["recovery_level"] = 5
    metadata["status"] = "degraded"  # KEIN "failed" — degraded = valid output
    metadata["recovery_fail_reason"] = reason
    logger.warning("recovery_cascade: all levels exhausted → degraded export, reason=%s", reason)
    return _original_input  # nie leerer Export, nie None

# VERBOTEN: leerer Export / Abbruch ohne Ausgabe / Export mit bekanntem Artefakt
# VERBOTEN: _recovery_cascade() ohne Watchdog-Timeout (max 30 s)
# VERBOTEN: Stufe überspringen (keine "Shortcut"-Implementierungen)
# VERBOTEN: _recovery_cascade() ohne reason-Parameter aufrufen (Signatur ist fix)
```

## §2.61 Output-Length-Guard

```python
# KANONISCH — nach JEDER Phase in UV3:
# ACHTUNG: shape[-1] statt len() — len() auf 2D-Stereo (2, N) gibt 2 zurück, NICHT N!
_n_in = input_audio.shape[-1]
_n_out = output.shape[-1]
if abs(_n_out - _n_in) > 64:
    logger.error("length_mismatch phase=%s delta=%d", phase_id, _n_out - _n_in)
    output = output[..., :_n_in]  # harter Crop (funktioniert für 1D und 2D)
    metadata["length_corrections"].append(phase_id)
# VERBOTEN: Zero-Padding als primäre Längenkorrektur
```

## §2.64 Per-Phase-Score-Delta (MAS-Konvergenz)

```python
# P1P2_GOALS — KANONISCHE DEFINITION (UV3-Klassenkonstante):
# P1 + P2 Goals (universell, immer aktiv — unabhängig von panns_singing):
P1P2_GOALS = frozenset({
    "natuerlichkeit", "authentizitaet",           # P1
    "tonal_center", "timbre", "artikulation",     # P2
})
# P0-Goals (vocal_quality, formant_fidelity) werden separat via VQI-Gate überwacht (§0p)
# P3–P5-Goals dürfen vorübergehend sinken (kein sofortiger Rollback-Trigger)

# mas_gap — Abstand zum MAS-Ziel (berechnet aus estimate_song_goal_targets()):
# mas_targets = UV3._mas_targets  (gesetzt in SongCalibration, nach estimate_song_goal_targets())
# mas_gap = {g: max(0.0, mas_targets[g] - post[g]) for g in P1P2_GOALS}

# KANONISCH — jede Phase MUSS diesen Rahmen nutzen:
pre_audio = audio.copy()  # PFLICHT: vor phase.process() sichern — für Rollback unten
pre = _fast_goal_snapshot(audio, sr, material)
audio = phase.process(audio, sr)
post = _fast_goal_snapshot(audio, sr, material)
metadata["phase_deltas"][phase_id] = {g: post[g] - pre[g] for g in pre}

# Rollback wenn (AUSNAHME: Carrier-Repair-Phasen dürfen P1/P2 vorübergehend senken):
_is_carrier_repair = phase_id in _CARRIER_REPAIR_PHASE_PREFIXES
if not _is_carrier_repair and any(post[g] - pre[g] < -0.03 for g in P1P2_GOALS):
    audio = pre_audio  # Rollback — pre_audio MUSS oben gesetzt sein (s. oben)

mas_gap = {g: max(0.0, self._mas_targets[g] - post[g]) for g in P1P2_GOALS}
# MAS-Erreicht-Stop:
if all(mas_gap[g] <= 0.02 for g in P1P2_GOALS):
    metadata["mas_achieved_at_phase"] = phase_id
    self._mas_fully_achieved = True  # Flag setzen — KEIN break!
    # VERBOTEN: break hier — _NEVER_SKIP-Phasen müssen trotz MAS weiterlaufen (§0k, §2.52)
    # §2.65 prüft _mas_fully_achieved zu Beginn jeder nächsten Phase und überspringt sie
    # (außer phase_id in _NEVER_SKIP)
```

### §2.64 `_fast_goal_snapshot` — Multi-Segment-Pflicht

```python
# VERBOTEN: Single-Segment-Bias auf Audio-Mitte
spec = fft(mono[N//2: N//2 + frame_size])  # FALSCH

# RICHTIG: 3 Segmente mitteln (25%/50%/75%)
specs = [fft(seg25), fft(seg50), fft(seg75)]
spec = np.mean(specs, axis=0)

# authentizitaet-Proxy: Zentral-Drittel statt Intro
acf_segment = mono[N//3: N//3 + 8192]

# transparenz-Proxy: Vollsignal + SFM-Blend
val = 0.70 * np.log10(p95_full / p05_full + 1e-9) / 4.0 + 0.30 * (1.0 - sfm_avg)
```

## §2.65 MAS-Early-Stop

```python
# VERBOTEN: Pipeline läuft nach _mas_fully_achieved=True weiter
# RICHTIG: UV3-Loop prüft _check_mas_convergence() nach jeder Phase
if self._mas_fully_achieved and phase_id not in _NEVER_SKIP:
    logger.info("MAS erreicht bei Phase %s — Pipeline-Stop", phase_id)
    skipped.append(phase_id)
    continue  # _NEVER_SKIP-Phasen laufen trotz MAS immer durch
```

## §GOAL_BASELINE_CHECK [RELEASE_MUST] (v10.0.0) — Garantierter Goal-Recovery-Pfad

**Problem (CAUSE_TO_PHASES-Lücke)**: Wenn DefectScanner keinen Defekt-Cause für ein Musical Goal findet, trägt `CAUSE_TO_PHASES` keine Recovery-Phasen für dieses Goal ein. Der HolisticPerceptualGate-Blend kann Goals nur durch Rollback in Richtung Original bewegen — er kann ein Goal **nicht über das Original-Niveau heben**. Das bedeutet: Goals, die strukturell unter dem materialadaptiven Floor liegen, aber kein erkennbares Defekt-Signal erzeugen, würden nie erreicht.

**Lösung**: §GOAL_BASELINE_CHECK läuft im UV3 `restore()`-Pfad **nach** GPOptimizer (`selected_phases` ist fertig) und **vor** `_execute_pipeline()`. Er misst alle 15 Goal-Proxies auf dem Eingangssignal via `_fast_goal_snapshot()` (DSP-only, ≤200 ms) und fügt für jedes Goal, das unter `material_floor × 0.95` liegt, die primäre Recovery-Phase in `selected_phases` ein.

```python
# KANONISCH — UV3 restore(), nach SLR, vor _execute_pipeline():
try:
    from backend.core.calibration_matrix import (
        get_goal_recovery_phases, get_material_floor
    )
    _gbc_mat_str = str(material_type.value if material_type else "unknown").lower()
    _gbc_is_studio = self.is_studio_mode()
    _gbc_snapshot = UnifiedRestorerV3._fast_goal_snapshot(audio, sample_rate, _gbc_mat_str)
    if _gbc_snapshot and _applicable_goals:
        _selected_set: set[str] = set(selected_phases)
        for goal, proxy_score in _gbc_snapshot.items():
            if goal not in _applicable_goals:
                continue
            floor = get_material_floor(_gbc_mat_str, goal, is_studio_2026=_gbc_is_studio)
            if proxy_score < floor * 0.95:
                for phase_id in get_goal_recovery_phases(goal, is_studio_2026=_gbc_is_studio):
                    if phase_id not in _selected_set:
                        selected_phases.append(phase_id)
                        _selected_set.add(phase_id)
                        break  # §2.45: nur primäre Recovery-Phase
except Exception as _gbc_exc:
    logger.debug("§GOAL_BASELINE non-blocking: %s", _gbc_exc)
```

**Invarianten** (alle bindend, kein Override):

- **§0a**: `get_goal_recovery_phases(is_studio_2026=False)` gibt niemals `phase_21_exciter`, `phase_35_multiband_compression` oder `phase_42_vocal_enhancement` zurück.
- **§2.45**: Nur die **erste** (primäre) Recovery-Phase wird pro Goal eingefügt — kein Massen-Adding.
- **5 %-Margin**: Auslöse-Schwelle `floor × 0.95`, nicht `floor` — vermeidet False-Positives bei physikalisch limitiertem Material.
- **`_applicable_goals`-Filter**: Nur Goals, die für den aktuellen Kontext gelten (GoalApplicabilityFilter-Ergebnis), werden geprüft.
- **Non-blocking**: `try/except` — jeder Fehler lässt `selected_phases` unverändert.
- **Metadata**: Hinzugefügte Phasen werden in `metadata["goal_baseline_recovery"]` dokumentiert.
- **VERBOTEN**: `_fast_goal_snapshot()` auf restauriertem Audio ausführen — nur auf dem Eingangssignal vor der Pipeline.
- **Reihenfolge** [NORMATIV]: Die Phasen-Liste in `_GOAL_TO_RECOVERY_PHASES_RESTORATION` MUSS §2.46-Carrier-Chain-Hierarchie einhalten — subtraktive/physikalische Korrekturen (Stufen 1–4) stehen VOR additiven Eingriffen (Stufen 5–6). Mechanische Ursachen (Wow/Flutter, EQ-Fehler) MÜSSEN VOR digitalen Korrekturen stehen. **Verboten**: Phasen nach Nummernreihenfolge sortieren.
- **Kausal-Richtung** [NORMATIV]: Primärphase MUSS den Defizit-Vektor invertieren. Für `spatial_depth`-Defizit → Phase muss Raumcues **hinzufügen** (`phase_46_spatial_enhancement`), nie entfernen (`phase_49` ist Kausal-Inversion und verboten als Primärphase für `spatial_depth`). Gleiches Prinzip für alle Goals: niedriger Score = zu wenig dieser Qualität → Enhancement; hoher Score = zu viel → Reduktion.
- **Disk-Validierung** [CI-GUARD]: Alle Phase-IDs in `_GOAL_TO_RECOVERY_PHASES_RESTORATION` und `_GOAL_TO_RECOVERY_PHASES_STUDIO_EXTRAS` MÜSSEN gegen existierende `backend/core/phases/phase_*.py`-Dateien validiert sein — Test: `test_get_goal_recovery_phases_all_phase_ids_exist_on_disk()` in `tests/unit/test_calibration_matrix.py`.

**Recovery-Phase-Tabelle** (Quelle: `calibration_matrix.get_goal_recovery_phases()`):

Reihenfolge nach §2.46-Carrier-Chain: subtraktiv vor additiv, mechanisch vor digital, breiteste Wirkung zuerst.

| Goal | Primäre Restoration-Recovery-Phase | Wissenschaftliche Begründung |
|---|---|---|
| timbre_authentizitaet | phase_04_eq_correction | Stufe 2: RIAA/Carrier-EQ-Fehler — physikalische Primärursache |
| natuerlichkeit | phase_03_denoise | Breitband-NR: universell größter Einzelbeitrag zur Natuerlichkeit |
| authentizitaet | phase_09_crackle_removal | Crackle (systematisch) zerstört Authentizität stärker als Einzelklicks |
| tonal_center | phase_12_wow_flutter_fix | Stufe 2: mechanische Rotation **vor** digitalem Pitch — physikalische Ursache zuerst |
| timbre | phase_04_eq_correction | Stufe 2: Carrier-EQ-Korrektur (subtraktiv, kausal) |
| artikulation | phase_08_transient_preservation | Transienten-Hüllkurve ist primärer Träger der Artikulationsklarheit |
| emotionalitaet | phase_26_dynamic_range_expansion | Dynamikkontrast ist der primäre Emotionsträger |
| micro_dynamics | phase_26_dynamic_range_expansion | Überkompression ist Primärursache für Mikrodynamikverlust |
| groove | phase_12_wow_flutter_fix | Unregelemäßige Motorrotation = direkte physikalische Ursache für Timing-Instabilität |
| transparenz | phase_03_denoise | Breitbandrauschen maskiert Details am stärksten |
| waerme | phase_04_eq_correction | 200–600 Hz Tonalbalance (EQ zuerst) ist physikalisches Fundament der Wärme |
| bass_kraft | phase_04_eq_correction | RIAA/EQ-Fehler im Bassbereich als Primärursache |
| separation_fidelity | phase_49_advanced_dereverb | Hall-Bleed zwischen Quellen — WPE am effektivsten |
| brillanz | phase_06_frequency_restoration | BW-Erweiterung (AudioSR): primärer Pfad für verlorenen HF-Inhalt |
| spatial_depth | phase_46_spatial_enhancement | Niedrige Raumtiefe = fehlende Cues → Enhancement; **VERBOTEN phase_49** (entfernt Raumtiefe!) |

> Kanonische Quelle: `backend/core/calibration_matrix.py` → `get_goal_recovery_phases()` + `_GOAL_TO_RECOVERY_PHASES_RESTORATION`. Studio-2026-Extras in `_GOAL_TO_RECOVERY_PHASES_STUDIO_EXTRAS`. Tests: `tests/unit/test_calibration_matrix.py` (§09.10-Tests, 8 Testfunktionen).

## §2.66 RecordingChainProfiler — Träger-Ketten-Kopplung [RELEASE_MUST v10.0.0]

**Problem**: Aurik behandelt 54 Defekttypen als unabhängige Signale. Physikalisch ist eine historische Aufnahme jedoch eine **kausale Kette** (Mikrofon → Vorstufe → Band → Presswerk → Abspielkette), deren Stufen gekoppelte Degradationen erzeugen. Wenn 8 Symptome aus derselben physikalischen Quelle stammen, aktiviert der CausalDefectReasoner 8 separate Phasen-Cluster — mit Over-Processing und sich überlappenden Korrekturen.

**Lösung**: `RecordingChainProfiler` gruppiert gemeinsam auftretende Causes in **Ketten-Cluster** und liefert einen `chain_hint` an den GPOptimizer. Phasen desselben Clusters werden koordiniert aktiviert statt einzeln.

```python
# Einhängepunkt: UV3 restore(), nach CausalDefectReasoner.reason(), vor GPOptimizer
from backend.core.recording_chain_profiler import RecordingChainProfiler

chain_profile = RecordingChainProfiler().profile_chain(
    causes=restoration_plan.top_causes,   # Liste aktiver Causes (Posterior > 0.15)
    material=material_type.value,
    era=era_decade,
)
# chain_profile.dominant_cluster: str ("tape_aging" | "vinyl_wear" | "azimuth_chain" | None)
# chain_profile.cluster_weight: float [0,1] — Konfidenz der Cluster-Zuordnung
# chain_profile.suppress_causes: list[str] — redundante Causes, die nicht einzeln aktiviert werden

# GPOptimizer erhält chain_hint als Prior-Bias:
gp_result = gp_optimizer.optimize(
    defect_scores=defect_scores_norm,
    material=material,
    chain_hint=chain_profile,   # NEU: koordiniert Phasen-Stärken innerhalb Cluster
)
```

**VERBOTEN**: `RecordingChainProfiler` auf weniger als 3 aktiven Causes aufrufen — zu wenig Evidenz für Cluster-Erkennung; `chain_hint=None` zurückgeben.

## §2.56c [RELEASE_MUST] Transfer-Chain-aware Strength-Oracle-Handover

**Problem**: Material-adaptive Initialstaerken aus §2.31 wirken indirekt, reichen aber nicht als
lokale Interventionssteuerung innerhalb einer einzelnen Phase.

**Pflicht**: `_prepare_profiled_phase_runtime_context()` MUSS `transfer_chain` und
`material_confidence` direkt an `resolve_phase_strength_oracle()` uebergeben.

```python
_chain = kwargs.get("transfer_chain") \
    or getattr(kwargs.get("cached_medium_result"), "transfer_chain", None) \
    or self._restoration_context.get("transfer_chain", [])

_chain_conf = kwargs.get("material_confidence")
if not isinstance(_chain_conf, (int, float)):
    _chain_conf = getattr(kwargs.get("cached_medium_result"), "confidence", None)

oracle_profile = resolve_phase_strength_oracle(
    ...,
    transfer_chain=[str(s).lower() for s in (_chain or []) if str(s).strip()],
    chain_confidence=float(_chain_conf) if isinstance(_chain_conf, (int, float)) else None,
)
```

**Invarianten:**

- `chain_factor` muss im Oracle-Profil (`hard_caps`) persistiert werden.
- Mehrstufige Ketten (`vinyl->cassette->mp3_low`) muessen konservativer sein als Einzeltraeger. `[SRC:S03,S04]`
- Low-Confidence-Ketten duerfen die Steuerung nur abgeschwaecht beeinflussen (confidence blending). `[SRC:S03]`
- Non-blocking bleibt verpflichtend: Oracle-Ausfall darf Pipeline nicht stoppen.

**Kanonische Cluster**:

| Cluster | Enthaltene Causes | Primäre Integrierte Korrektur |
|---|---|---|
| `tape_aging` | tape_dropout, tape_hiss, hf_remanence_loss, print_through, generation_loss | phase_29 + phase_24 + phase_03 als Koalition (§2.67) |
| `vinyl_wear` | vinyl_crackle, stylus_damage, inner_groove_distortion, surface_noise | phase_01 + phase_09 + phase_05 als Koalition |
| `azimuth_chain` | head_misalignment, azimuth_error, bandwidth_loss | phase_25 + phase_04 + phase_06 als Koalition |
| `shellac_degradation` | lacquer_disc_degradation, surface_noise, bandwidth_loss, vinyl_crackle | phase_03 + phase_09 + phase_06 als Koalition |
| `mechanical_wow` | wow, flutter, flutter_spectral_sidebands, speed_calibration_error | phase_12 + phase_31 als Koalition |

> Quelle: `backend/core/recording_chain_profiler.py`. Neue Cluster über `_CHAIN_CLUSTERS`-Dict erweiterbar.

## §2.67 Phase-Koalitions-Evaluation — globale Optimierung statt lokaler Gates [RELEASE_MUST v10.0.0]

**Problem**: §2.45 (Minimal-Intervention) erzwingt `perceptual_delta > 0` **pro Phase**. Das verhindert jede Korrektursequenz, die durch ein lokales Minimum führt. Schwere Restaurierungsfälle (gekoppelte Defekte, tief-analoge Carrier) sind genau die Fälle, die mutige mehrstufige Eingriffe benötigen.

**Lösung**: Vordefinierte **Phasen-Koalitionen** werden als Gruppe evaluiert. `perceptual_delta` wird erst nach der gesamten Koalition gemessen; innerhalb der Koalition dürfen einzelne Phasen negative Deltas haben.

```python
# In UV3, Klassen-Konstante:
_PHASE_COALITIONS: dict[str, list[str]] = {
    "tape_repair":     ["phase_29_tape_hiss_reduction", "phase_12_wow_flutter_fix", "phase_24_dropout_repair"],
    "vinyl_surface":   ["phase_01_click_removal", "phase_09_crackle_removal", "phase_05_rumble_filter"],
    "carrier_invert":  ["phase_04_eq_correction", "phase_03_denoise", "phase_06_frequency_restoration"],
    "shellac_repair":  ["phase_03_denoise", "phase_09_crackle_removal", "phase_01_click_removal"],
    "mechanical_fix":  ["phase_12_wow_flutter_fix", "phase_31_speed_pitch_correction", "phase_25_azimuth_correction"],
}

# Koalitions-Ausführung in _profiled_phase_call_with_delta():
if coalition_id := _get_coalition_for_phase(phase_id):
    pre_coalition = audio.copy()
    for coalition_phase in _PHASE_COALITIONS[coalition_id]:
        audio = _run_phase(coalition_phase, audio)
    # Delta-Gate erst JETZT — nach der gesamten Koalition:
    if perceptual_delta(pre_coalition, audio) < 0:
        audio = pre_coalition  # Rollback auf Koalitions-Eingang
        metadata["coalition_rollback"] = coalition_id
    else:
        metadata["coalition_applied"] = coalition_id
    continue  # Einzelphasen der Koalition nicht nochmals einzeln ausführen
```

**Invarianten**:

- Koalitions-Phasen müssen §2.46-Carrier-Chain-Reihenfolge (subtraktiv vor additiv) einhalten.
- §0a-verbotene Phasen dürfen nie in einer Koalition für Restoration stehen.
- `_NEVER_SKIP`-Phasen laufen immer einzeln — keine Koalitions-Wrapping.
- Zeitbudget einer Koalition = Summe der Einzel-Budgets (§ Per-Phase-Zeitbudgets).
- Koalition wird nur aktiviert wenn `chain_hint.dominant_cluster` die Koalition bestätigt.

## §2.69 TemporalContinuityGuard — Zeitliche Diskontinuität [RELEASE_MUST v10.0.0]

**Problem**: Frame-by-frame-Processing erzeugt Mikro-Diskontinuitäten zwischen Verarbeitungsblöcken. Diese erscheinen in keiner Metrik und in keinem Test (synthetische Signale haben keine natürlichen Phrasengrenzen). Hörer nehmen sie als "da stimmt was nicht" wahr.

**Lösung**: Post-Phase-Hook misst Frame-RMS-Varianz-Ratio und protokolliert Überschreitungen in `metadata`.

```python
# Einhängepunkt: Ende von _profiled_phase_call_with_delta(), nach perceptual_delta-Gate
from backend.core.temporal_continuity_guard import check_temporal_continuity

tc_result = check_temporal_continuity(pre=pre_phase_audio, post=audio, phase_id=phase_id, sr=sr)
metadata.setdefault("temporal_continuity", {})[phase_id] = {
    "variance_ratio": tc_result.variance_ratio,
    "gain_step_db": tc_result.gain_step_db,
    "ok": tc_result.ok,
}
# KEIN Veto — nur Protokollierung. Warnung ab variance_ratio > 2.5:
if not tc_result.ok:
    logger.warning("temporal_continuity phase=%s variance_ratio=%.2f", phase_id, tc_result.variance_ratio)
# Zusätzlich: gain_step_db > 1.5 — abrupter Gain-Sprung an Phase-Grenze → Mikro-Klick:
if tc_result.gain_step_db > 1.5:
    logger.warning("temporal_continuity_gain phase=%s gain_step_db=%.1f dB > 1.5 → potential click",
                   phase_id, tc_result.gain_step_db)
    metadata.setdefault("temporal_continuity_gain_warnings", []).append(phase_id)
```

**`TemporalContinuityResult`-Felder**: `ok: bool`, `variance_ratio: float`, `phase_id: str`.

**Schwellwert**: `variance_ratio > 2.5` = Warnung (kein Rollback). `variance_ratio > 8.0` = zusätzlich `metadata["temporal_continuity_critical"].append(phase_id)`.

**Implementierung**:

```python
# backend/core/temporal_continuity_guard.py:
def check_temporal_continuity(pre, post, phase_id, sr):
    frame_rms_pre  = librosa.feature.rms(y=np.mean(pre, axis=0) if pre.ndim==2 else pre,
                                          frame_length=2048, hop_length=512)[0]
    frame_rms_post = librosa.feature.rms(y=np.mean(post, axis=0) if post.ndim==2 else post,
                                          frame_length=2048, hop_length=512)[0]
    variance_ratio = float(np.var(frame_rms_post) / (np.var(frame_rms_pre) + 1e-8))
    # gain_step_db: abrupter Pegel-Sprung an Phase-Grenze (Fade-out letztes Frame → Fade-in erstes)
    rms_pre_last  = float(frame_rms_pre[-1])  if len(frame_rms_pre) > 0  else 1e-8
    rms_post_first = float(frame_rms_post[0]) if len(frame_rms_post) > 0 else 1e-8
    gain_step_db = float(20 * np.log10((rms_post_first + 1e-10) / (rms_pre_last + 1e-10)))
    return TemporalContinuityResult(
        ok=variance_ratio < 2.5, variance_ratio=variance_ratio,
        phase_id=phase_id, gain_step_db=abs(gain_step_db),
    )
```

> Langfristig: `variance_ratio`-Daten aus `metadata` aggregieren → Era/Material-adaptive Schwellwerte.

## §2.70 RestorationMemory — Persistenter GPOptimizer-Prior [RELEASE_MUST v10.0.0]

**Problem**: GPOptimizer startet jeden Lauf mit uniformem Prior — kein Gedächtnis über erfolgreiche Lösungsstrategien aus vergangenen Läufen. Erfahrene Toningenieure haben dieses Gedächtnis; Aurik hat es nicht.

**Lösung**: `RestorationMemory` persistiert Bayesianische Priors in `~/.aurik/restoration_memory.json`, gehasht nach `(era, material, defect_cluster_signature)`.

```python
# Einhängepunkt: GPOptimizer.__init__() + optimize()-Rückgabe
from backend.core.restoration_memory import RestorationMemory

# Vor GPOptimizer-Aufruf: gespeicherten Prior laden
_rm = RestorationMemory()
_rm_key = (era_decade, material_type.value, _defect_cluster_hash(top_causes))
prior_data = _rm.get_prior(_rm_key)  # None wenn neu; dict{"X_init": ..., "Y_init": ...} wenn bekannt

gp_result = gp_optimizer.optimize(
    ...,
    x_init=prior_data.get("X_init") if prior_data else None,
    y_init=prior_data.get("Y_init") if prior_data else None,
)

# Nach erfolgreichem Export (HPI > 0 AND artifact_freedom >= 0.95):
if export_hpi > 0 and metadata.get("artifact_freedom", 0) >= 0.95:
    _rm.save_result(
        key=_rm_key,
        phase_params=gp_result.best_params,
        hpi_achieved=export_hpi,
    )
```

**Sicherheits-Invarianten**:

- `~/.aurik/restoration_memory.json` darf maximal 10 MB groß werden (LRU-Eviction).
- Nur Erfolge (`HPI > 0`) werden gespeichert — kein Lernen aus schlechten Läufen.
- Read ist non-blocking (`try/except`, JSON-Parsing-Fehler → `None`).
- Write ist atomic (write to `.tmp`, dann `os.replace`).
- **VERBOTEN**: `RestorationMemory` auf Cloudspeicher oder Netzwerkpfade schreiben — rein lokal.

**Coalition-aware Prior-Invariante (v10.0.0)**:

- Erfolgreiche Läufe dürfen zusätzlich Koalitions-Lernsignale in `phase_params` persistieren:
    `_coalition_learning_factor`, `_dominant_phase_coalition_ratio`, `_phase_coalition_count`.
- Diese Felder sind advisory-only für den GP-Start und dürfen harte Gates (PMGG/CIG/AFG/HPI)
    niemals überstimmen.
- Beim Laden eines Priors MUSS UV3 den geladenen Prior in `_restoration_context` spiegeln
    (`restoration_memory_prior`), damit Downstream-Module und Telemetrie dieselbe Referenz sehen.

## §2.78b Runtime-Metric-Reliability-Graph [RELEASE_MUST v10.0.0]

**Problem**: PMGG/Phase-Deltas liefern pro Phase reichhaltige Proxy-Telemetrie, aber ohne
laufzeitnahe Reliability-Kalibrierung bleiben Goal-Konfidenzen in Grenzfällen zu statisch.

**Lösung**: Ein kontextsensitiver `MetricReliabilityGraph` kalibriert Goal-Zuverlässigkeit
online aus realen Phase-Deltas und PMGG-Metadaten (Team-Net-Delta,
Reconstruction-Localized, epistemic confidence).

```python
# Vor AdaptivePhaseRescheduler.reschedule(...)
from backend.core.metric_reliability_graph import get_metric_reliability_graph

_mrg = get_metric_reliability_graph()
_mrg.update_from_phase_delta(
        phase_id=phase_id,
        goal_deltas=self._phase_deltas.get(phase_id, {}).get("delta", {}),
        phase_metadata=self._phase_metadata_accumulator.get(phase_id, {}),
        material_type=material_key,
        transfer_chain=transfer_chain,
        is_studio_2026=self.is_studio_mode(),
        era_decade=decade,
)
runtime_conf = _mrg.get_goal_reliability(...)
base_w, runtime_w = _mrg.get_blend_weights(...)
blended_conf = base_w * base_conf + runtime_w * runtime_conf
```

**Invarianten**:

- Advisory-only: Der Graph darf harte Gates (PMGG/CIG/AFG/HPI/VQI) niemals ersetzen.
- Offline-only: Persistenz ausschließlich lokal (`~/.aurik/metric_reliability_graph.json`,
    atomarer Write, non-blocking Read/Write).
- Gebundene Werte: Goal-Reliability immer in `[0.20, 0.98]`.
- Kontextbindung Pflicht: Key muss mindestens Modus, Material, Era-Bucket und
    Chain-Komplexitäts-Bucket enthalten.
- Blend-Gewichte müssen adaptiv sein (cross-run Evidenz), normiert auf 1.0 und
    in den Grenzen `base ∈ [0.40, 0.75]`, `runtime ∈ [0.25, 0.60]` liegen.

> Datei-Pfad: `~/.aurik/restoration_memory.json`. Konfigurierbar via `AURIK_MEMORY_PATH`-Env-Variable (Desktop-Offline-Pflicht beachten).

## §2.71 Formant-Toleranz-Verscharfäung [RELEASE_MUST v10.0.0]

Quelle: `[SRC:S08,S09]`

**Änderung**: Per-Formant-Toleranz statt globalem ±2 dB.

| Formant | Grenze | Begründung |
|---|---|---|
| F1 | ± 1.0 dB | Vokalität (Offenheit): hoch-perceptuell |
| F2 | ± 1.0 dB | Vokalität (Vorderzunge): hoch-perceptuell |
| F3 | ± 1.5 dB | Timbre-Qualität: mittel-perceptuell |
| F4 | ± 1.5 dB | Brillanz/Nasalität: niedriger-perceptuell |

```python
# UV3 setzt _FORMANT_TOLERANCE_DB in _restoration_context nach VocalFocusAnalyzer:
_ctx["formant_tolerance_db"] = [1.0, 1.0, 1.5, 1.5]  # F1, F2, F3, F4
# Bei era_decade < 1960: resolve_formant_tolerance_db() lockt historisch bedingt auf [1.5, 1.5, 2.0, 2.0]
```

> `resolve_formant_tolerance_db()` in `backend/core/musical_goals/era_vocal_profile.py` — gibt era-adaptierte Werte zurück; nutzt `_ctx["formant_tolerance_db"]` als Input-Maximum.

## §2.72 Vibrato-Tiefe-Schutz [RELEASE_MUST v10.0.0]

Quelle: `[SRC:S10,S11]`

**Regel**: F0-Modulationstiefe (max–min F0 in Hz innerhalb Vibrato-Zonen) darf durch NR/Kompression nicht mehr als ±10 % reduziert werden.

```python
# UV3 triggert check nach jeder NR/Dynamics-Phase auf Vokal-Material:
from backend.core.dsp.vibrato_guard import check_vibrato_depth_preservation

_vdp = check_vibrato_depth_preservation(audio_pre, audio_post, sr)
if _vdp.depth_reduction_pct > 10.0:
    # Blend in Vibrato-Segmenten: 50 % Dry
    metadata["vibrato_depth_reduction_pct"] = _vdp.depth_reduction_pct
    metadata["vibrato_depth_preservation"] = max(0.0, 1.0 - _vdp.depth_reduction_pct / 100.0)
    logger.warning("vibrato_depth: %.1f%% > 10%% → blend 50%% dry in vibrato-segments", _vdp.depth_reduction_pct)
```

**Pflicht-Telemetrie**: `vibrato_depth_preservation` MUSS zusaetzlich ueber `_update_vocal_quality_metrics(...)` in `_restoration_context["vocal_quality_check"]` und `_phase_metadata_accumulator` gespiegelt werden, damit WCS/HTEV/psychoakustische Gates denselben Wert lesen wie der Phasen-Guard.

## §2.73 Pre-Echo-Prevention [RELEASE_MUST v10.0.0]

**Regel**: Additive ML-Phasen können Transient-Onsets zeitlich verschieben (≤ 2 ms = tolerierbar; > 2 ms = Blend-Reduction).

```python
# Prüfung in _profiled_phase_call_with_delta() für ADDITIVE-Phasen:
if phase_id in {"phase_06", "phase_07", "phase_23"}:
    from backend.core.dsp.transient_guard import detect_transient_shifts
    _ts = detect_transient_shifts(pre_phase_audio, audio, sr)
    if _ts.max_shift_ms > 2.0:
        audio = pre_phase_audio * (_ts.max_shift_ms / 2.0) + audio * (1.0 - _ts.max_shift_ms / 2.0)
        metadata["onset_shift_ms"] = _ts.max_shift_ms
```

## §2.74 Spektralfarbe-Erhaltung [RELEASE_MUST v10.0.0]

**Regel**: Die charakteristische 1/3-Oktav-Kurve (200–8000 Hz) muss nach EQ/NR-Phasen zu ≥ 0.97 korreliert bleiben.

```python
# Nach EQ/NR-Phasen:
from backend.core.dsp.spectral_color_guard import check_spectral_color_preservation

_scp = check_spectral_color_preservation(pre_phase_audio, audio, sr)
if _scp.correlation < 0.97:
    audio = pre_phase_audio * 0.30 + audio * 0.70  # Strength -30 %
    metadata["spectral_color_corr"] = _scp.correlation
```

## §2.75 Mikrodynamik-Korrelation [RELEASE_MUST v10.0.0]

**Regel**: Frame-Energie-Korrelation (10 ms) auf voiced-Zonen nach NR/Dynamics-Phasen.

- `0.25 ≤ panns_singing < 0.35` → Zielwert `≥ 0.97`
- `panns_singing ≥ 0.35` → Zielwert `≥ 0.985`, Blend-Floor `0.93`

```python
from backend.core.dsp.mikrodynamik_guard import frame_energy_correlation
if panns_singing >= 0.25:
    _corr = frame_energy_correlation(pre_phase_audio, audio, sr, frame_ms=10)
    _target = 0.985 if panns_singing >= 0.35 else 0.97
    _floor = 0.93 if panns_singing >= 0.35 else 0.90
    if _corr < _target:
        _wet = float(min(1.0, max(0.0, (_corr - _floor) / max(_target - _floor, 1e-6))))
        audio = pre_phase_audio * (1.0 - _wet) + audio * _wet
        metadata["mikrodynamik_corr"] = _corr
        metadata["mikrodynamik_target_corr"] = _target
```

**Pflicht-Telemetrie**: `micro_dynamic_correlation` MUSS ueber `_update_vocal_quality_metrics(...)` in `_restoration_context["vocal_quality_check"]` und `_phase_metadata_accumulator` geschrieben werden. Finale WCS-/Psychoakustik-Gates duerfen nicht auf phasenlokale `metadata` beschraenkt bleiben.

## §2.75a Vocal-Guard-Runtime-Bridge [RELEASE_MUST v10.0.0]

**Problem**: Phase-lokale Guard-Metadaten sind fuer das End-Gate wertlos, wenn sie nur in `result.metadata` leben. UV3-Final-Gates (WCS, HTEV, Psychoakustik, Spec-Upgrade-Entscheidung) lesen aggregierte Werte aus `_restoration_context["vocal_quality_check"]` und `_phase_metadata_accumulator`.

```python
# KANONISCH nach jedem relevanten Vocal-Guard:
_update_vocal_quality_metrics(
    formant_integrity=formant_integrity,
    formant_fidelity=formant_integrity,
    vibrato_depth_preservation=vibrato_depth_preservation,
    micro_dynamic_correlation=micro_dynamic_correlation,
    noise_texture_authenticity=noise_texture_authenticity,
)
```

**Pflichtfelder**:

- `formant_integrity`
- `vibrato_depth_preservation`
- `micro_dynamic_correlation`
- `noise_texture_authenticity`

**Verboten**:

- Vocal-Guard schreibt nur in `result.metadata` oder `phase_metadata[phase_id]`
- WCS/HTEV liest Default-1.0, obwohl ein Guard bereits degradierten Wert gemessen hat

**Strenger Vocal-NTI-Pfad**: Bei `panns_singing ≥ 0.35` gilt fuer `noise_texture_authenticity` der strengere Guard-Schwellenwert `0.18` statt `0.25`.

## §2.76 Wärmeband-Guard [RELEASE_MUST v10.0.0]

**Regel**: Kumulativer 200–800 Hz Verlust über alle Phases > 2.5 dB → Blend-Faktor für alle weiteren Phasen.

```python
# In UV3._restoration_context aktuell halten:
_ctx.setdefault("warmth_band_loss_db", 0.0)
_ctx["warmth_band_loss_db"] += max(0.0, _wbd.loss_db)
if _ctx["warmth_band_loss_db"] > 2.5:
    # Blend-Faktor: bei 2.5 dB Verlust = 0.5; bei 5.0 dB = 0.0 (vollständiges Dry)
    warmth_blend = float(1.0 - _ctx["warmth_band_loss_db"] / 5.0)
    warmth_blend = max(0.0, min(1.0, warmth_blend))
    audio = pre_phase_audio * (1.0 - warmth_blend) + audio * warmth_blend
    metadata["warmth_band_loss_db"] = _ctx["warmth_band_loss_db"]
```

## §2.77 Angriffstransienten-Integrität [RELEASE_MUST v10.0.0]

**Regel**: Onset-Fenster (0–20 ms nach Transient) sind perceptuell sensitivste Frames — max. 1.5 dB Änderung.

```python
# In UV3 nach NR/EQ-Phasen:
from backend.core.dsp.onset_guard import apply_onset_protection_mask
audio = apply_onset_protection_mask(
    audio_pre=pre_phase_audio, audio_post=audio,
    onset_mask=_ctx["onset_mask"],
    max_delta_db=1.5,
)
# onset_mask wird einmalig nach SongCalibration via HPSS berechnet und in _ctx gespeichert.
```

| Material | Pflicht-Phasen (unabhängig von DefectScanner-Score) |
|---|---|
| vinyl | phase_09, phase_12, phase_05 |
| tape / cassette | phase_29, phase_24, phase_06, phase_03 |
| reel_tape | phase_29, phase_24, phase_03, phase_55 |
| shellac | phase_03, phase_06, phase_01 |
| mp3_low | phase_23, phase_03, phase_50 |

> `cassette` → intern immer als `tape` in `_MATERIAL_PRIORITY_PHASES`

## §2.78 AdaptivePhaseRescheduler — Geschlossener Regelkreis [RELEASE_MUST v10.0.0]

**Funktion**: Nach jeder Phase werden Goal-Lücken (`song_goal_targets[g] − post_snap[g]`) berechnet. Für Goals mit signifikantem Gap injiziert der Rescheduler die Primär-Recovery-Phase (aus `get_goal_recovery_phases()`) ans Ende von `selected_phases`. Python's `for`-Loop besucht neu angehängte Elemente — kein Loop-Refactoring nötig.

**Koalitionsregel**: Wenn die Primär-Recovery-Phase eines Goals auch in den Recovery-Listen
anderer offener Goals auftaucht, bekommt dieses Goal einen kleinen Prioritätsbonus. Dadurch
werden Phasen bevorzugt, die mehrere offene Defizite zusammen adressieren können, ohne dass
§2.45 Minimal-Intervention oder die Ein-Phase-pro-Goal-Regel aufgeweicht werden.

**Modul**: `backend/core/adaptive_phase_rescheduler.py` (Singleton `get_adaptive_phase_rescheduler()`)

**Invarianten**:

- §0a: `_RESTORATION_FORBIDDEN = {phase_21_exciter, phase_35_multiband_compression, phase_42_vocal_enhancement}` — nie in Restoration injiziert
- §2.45 Minimal-Intervention: nur Primär-Phase (Index 0) aus `get_goal_recovery_phases()` injiziert
- §2.52 _NEVER_SKIP: diese Phasen nie injiziert (laufen immer)
- §2.65 MAS: `_mas_fully_achieved=True` → Rescheduler nicht aufgerufen (UV3-Guard)
- §_SELF_MANAGED_GOALS: `vocal_quality`, `formant_fidelity` haben eigene Recovery-Pfade (VQI-Gate §0p) → kein Rescheduler-Override
- Max 3 Injektionen pro Song-Session (`MAX_INJECTIONS_PER_SESSION`); `reset_session()` nach Song
- Koalitionsbonus ist nur Reihenfolge-Hint; Session-Grenzen, Executed/Plan-Guards und §0a bleiben bindend

**Integration in UV3 (§Hebel-3 Block)**:

```python
# Nach _conductor.measure_state() / recommend():
_cl_post_snap = dict(self._phase_deltas.get(phase_id, {}).get("post", {}))
_cl_song_targets = getattr(self, "_song_goal_targets", None)
if _cl_post_snap and isinstance(_cl_song_targets, dict) and not getattr(self, "_mas_fully_achieved", False):
    _apr = get_adaptive_phase_rescheduler()
    _apr_result = _apr.reschedule(
        current_goal_scores=_cl_post_snap,
        song_goal_targets=_cl_song_targets,
        selected_phases=selected_phases,
        executed=set(executed) | set(skipped),
        is_studio_2026=self.is_studio_mode(),
        transfer_chain=list((self._restoration_context or {}).get("transfer_chain") or []),
        material_type=_mat_str_cond,
    )
    for _inj_pid in _apr_result.new_phases_to_append:
        if _inj_pid not in set(selected_phases):
            selected_phases.append(_inj_pid)
```

**Goal-Score-Quelle**: `self._phase_deltas[phase_id]["post"]` (kein Extra-DSP-Aufwand — Snapshot läuft ohnehin via §2.64).

**Conductor-Integration**: `_conductor.recommend()` erhält jetzt `song_goal_targets` + `current_goal_scores` → Stopp-Signal (80 % Ziele erreicht) ist jetzt aktiv (war zuvor tot).

---

## §2.29c Restorative-Baseline-Capping

```python
_RESTORATIVE_PHASES = {
    "phase_02", "phase_03", "phase_09", "phase_18",
    "phase_20", "phase_23", "phase_24", "phase_29", "phase_49"
}
# Für diese Phasen:
effective_before[g] = min(measured_before[g], canonical_threshold[g] + 0.05)
# Enhancement-Phasen: echte scores_before (kein Capping)
```

## §2.29e PMGG Team-Koordination

```python
# UV3 schreibt prior_phase_context nach jeder Phase fort
# Phase_50 nach HF-Restauration (phase_06/07/23):
#   Goal-Exclusion: brillanz, transparenz, timbre
#   Emergency-Retries unterdrückt
# CONFLICT_REGISTRY: get_conflict_phases() aus phase_ontology.py
```

## §2.29f PMGG-Fallback-Risiko + Wall-Budget-Qualitaetsguard

```python
# PMGG-Restorability-Quelle MUSS im Ergebnis sichtbar sein:
# metadata["restorability_source"] in {"cached", "estimated", "fallback"}
# Bei fallback MUSS ein Quality-Risk-Flag gesetzt werden:
# metadata["quality_risk_flags"] += [{"code": "PMGG_RESTORABILITY_FALLBACK", ...}]

# Wall-Budget-Guard darf nicht nur phase_50 betreffen.
# Qualitaetskritische Phasen mit aktivem Goal-Defizit duerfen trotz Budgetdruck laufen,
# wenn _should_protect_phase_from_wall_budget(...) == True.
# Mindest-Guard-Set in UV3:
#   - phase_50_spectral_repair (lossy chain-end + HF/Transient-Defizit)
#   - phase_39_air_band_enhancement (brillanz/transparenz/artikulation Defizit)
#   - phase_42_vocal_enhancement (nur Studio; vocal/formant/artikulation Defizit)

# Guard-Ereignisse muessen in Metadata landen:
# metadata["wall_budget_quality_guard_events"] = [
#   {"phase_id": ..., "reason": "goal_deficit_quality_guard", "transfer_chain": [...]},
# ]

# WICHTIG §0a:
# phase_42 bleibt in Restoration verboten.
# Der Guard aktiviert keine Modus-verbotenen Phasen, er verhindert nur Budget-Passthrough
# fuer bereits legal selektierte qualitaetskritische Phasen.
```

## §2.55 PMGG-CIG-Sync-Invariante

```python
# Bei neuer Phase: BEIDE Tabellen synchron aktualisieren
# CIG._PHASE_SPECIFIC_DRIFT_EXCLUSIONS[p] ∩ P1P2
# == PMGG.PHASE_GOAL_EXCLUSIONS[p] ∩ P1P2
# CI-Test: test_pmgg_cig_sync.py
```

## VQI-Gate (Gesangsmaterial)

```python
# _rollback_last_vocal_phase() — KANONISCHE DEFINITION (UV3-intern):
# UV3 pflegt _vocal_phase_inputs: List[Tuple[str, np.ndarray]] = []
# Jede Phase mit panns_singing >= 0.25: _vocal_phase_inputs.append((phase_id, pre_phase_audio))
# Maximale Stack-Tiefe: 5 (ältere Einträge werden verworfen)

def _rollback_last_vocal_phase(audio_current: np.ndarray) -> np.ndarray:
    if not _vocal_phase_inputs:
        logger.warning("singer_identity_rollback: no vocal phase inputs recorded — no-op")
        return audio_current  # kein Rollback möglich → aktuellen Stand behalten
    phase_id, pre_audio = _vocal_phase_inputs.pop()  # letzten Eintrag entfernen
    logger.info("singer_identity rollback: reverting phase=%s", phase_id)
    metadata["singer_identity_rollback_phase"] = phase_id
    return pre_audio
```

```python
# PFLICHT wenn panns_singing >= 0.35:  (≠ panns_singing_confidence — kanonischer Name: panns_singing)
if panns_singing >= 0.35:
    from backend.core.musical_goals.vocal_quality_index import compute_vqi
    result = compute_vqi(
        audio_orig=original_audio,
        audio_restored=restored_audio,
        sr=sr,
    )
    vqi = result["vqi"]  # float [0, 1]
    metadata["vqi"] = vqi
    metadata["singer_identity_cosine"] = result.get("singer_identity_cosine", 0.85)
    metadata["singer_id_dsp_fallback"] = result.get("singer_id_dsp_fallback", True)
    # singer_identity_cosine-Gate (Resemblyzer-Identitäts-Prüfung):
    # NUR aktiv wenn kein Multi-Singer-Track (Duette/Chorprojekte würden falsch-positiv)
    # UND NUR wenn kein DSP-Fallback (ZCR/Spektral-Proxy zu unzuverlässig für Rollback-Trigger)
    # §DSP-Fallback-Guard (v10.0.0): singer_id_dsp_fallback=True → Advisory only, kein Rollback
    if not metadata.get("multi_singer", False):
        sic = result.get("singer_identity_cosine", 0.85)
        _sic_dsp_fb = bool(result.get("singer_id_dsp_fallback", True))
        if sic < 0.92 and not _sic_dsp_fb:
            logger.warning("singer_identity_cosine=%.3f < 0.92 — rolling back last vocal phase", sic)
            audio_restored = _rollback_last_vocal_phase(audio_restored)
        elif sic < 0.92 and _sic_dsp_fb:
            logger.info("singer_identity_cosine=%.3f < 0.92 but singer_id_dsp_fallback=True — advisory only, no rollback", sic)
            metadata["singer_id_advisory_cosine"] = sic
    # Dreistufige Recovery-Kaskade (kein harter Veto, §0p):
    # material_vqi_floor — KANONISCHE BERECHNUNG (VERBOTEN: Konstante 0.72 hardcoden!):
    material_vqi_floor = calibration_matrix.get_material_floor(material, "vqi")
    # Fallback-Werte (falls calibration_matrix nicht verfügbar):
    # {"shellac": 0.62, "vinyl": 0.72, "cd": 0.82, "digital": 0.82, "tape": 0.72, "unknown_analog": 0.72}
    if vqi < material_vqi_floor:        # Shellac: 0.62 | Vinyl: 0.72 | CD: 0.82
        return _recovery_cascade("vqi < material_floor", audio_restored)
    elif mode == "restoration" and vqi < 0.82:
        return _recovery_cascade("vqi < restoration_target", audio_restored)
    elif mode == "studio" and vqi < 0.87:
        return _recovery_cascade("vqi < studio_target", audio_restored)
```

## Per-Phase-Zeitbudgets — Wall-Time-Watchdog (Lücke 5, RELEASE_MUST)

> Gesamtbudget Pipeline: ≤ 240 s. Jede Phase hat ein individuelles Wall-Time-Budget.
> Überschreitung → Rollback auf Phase-Eingang + DSP-Fallback, kein Pipeline-Abbruch.

```python
# Kanonischer Watchdog-Rahmen in UV3 _profiled_phase_call_with_delta():
if elapsed > _PHASE_WALL_TIME_BUDGET[phase_id]:
    logger.error("phase_timeout phase=%s elapsed=%.1fs budget=%.1fs",
                 phase_id, elapsed, _PHASE_WALL_TIME_BUDGET[phase_id])
    audio = pre_phase_audio
    metadata["phase_timeouts"].append(phase_id)
    # _NEVER_SKIP-Phasen: KEIN Blacklisting, Budget 2× (nächster Aufruf gibt mehr Zeit)
    if phase_id in _NEVER_SKIP:
        _PHASE_WALL_TIME_BUDGET[phase_id] *= 2.0  # einmalige Erhöhung, max 3× original
        logger.warning("never_skip_timeout phase=%s — budget doubled, NOT blacklisted", phase_id)
        # Kein continue — _NEVER_SKIP-Phase mit neuem Budget sofort nochmals ausführen:
        audio = phase.process(pre_phase_audio, sr, material_type=material, strength=0.3)
    else:
        # ≥ 3 Timeouts derselben Phase → Session-Blacklist (kein Retry mehr)
        if metadata["phase_timeouts"].count(phase_id) >= 3:
            metadata["phase_blacklist"].add(phase_id)
        continue
```

| Phase | Budget (s) | Begründung |
|---|---|---|
| phase_01 (DC + Clip-Repair) | 4 | leicht, immer aktiv |
| phase_03 (Surface-Noise NR) | 20 | DFN + OMLSA auf vollem Signal |
| phase_04 (RIAA-EQ) | 3 | reines DSP |
| phase_06 (BW-Erweiterung) | 45 | AudioSR-Diffusion — teuerste Phase |
| phase_07 (Harmonik-Extrapolation) | 10 | DSP |
| phase_09 (Crackle/Knistern) | 15 | ML-Transient-Detektion |
| phase_12 (Wow/Flutter) | 12 | PSOLA + Pitch-Track |
| phase_14 (Stereo-Korrektur) | 5 | DSP |
| phase_15 (Loudness-Norm) | 3 | EBUR128, trivial |
| phase_23 (Codec-BW) | 20 | NVSR oder SBR |
| phase_24 (Dropout-Repair) | 8 | ML-Inpainting |
| phase_25 (Azimuth-Korrektur) | 6 | DSP |
| phase_29 (Bandrauschen/Tape-NR) | 25 | SGMSE+ auf Tape-Material |
| phase_30 (DC-Offset) | 2 | trivial |
| phase_31 (Quantisierungs-NR) | 5 | DSP |
| phase_42 (Vokal-Enhancement) | 15 | MIIPHER-Stub + Wiener |
| phase_47 (TruePeak-Limiter) | 2 | trivial |
| phase_49 (Dereverb) | 18 | WPE oder Hybrid-Dereverb |
| phase_50 (MP3-Artefakte) | 12 | Transient-MDCT-Korrektur |
| alle anderen Phasen (Default) | 10 | konservatives Budget |

Normquelle fuer Loudness-/TruePeak-Regeln: `[SRC:S06,S07]`.

> `_PHASE_WALL_TIME_BUDGET` als Klassen-Konstante in `backend/core/unified_restorer_v3.py`.

---

## Hörordnungs-Verdrahtung (2026-08-23)

Die psychoakustische Wahrheits-Ordnung (`.github/instructions/hoerordnung.instructions.md`)
ist die Rollen-Spitze für Hör-Entscheidungen. Ihre Verdrahtungspunkte in UV3:

- **Ebene 1 (§3):** VQI-Rollback akzeptiert Checkpoints auch bei Verbesserung
  verletzter Invarianten (`consonant_clarity`/`vibrato_precision` < 0.85);
  erfolgreiche Phasen-Rücknahme unterdrückt die Phase-65-Recovery-Kaskade
  (`_vqi_retreated_ok`). **§SCK-R/§WBG-R:** Phasen-Rücknahme im zentralen
  Phasen-Call bei Spektralfarben-Korrelation < 0.60 bzw. Einzelphasen-
  Wärmeband-Verlust > 3 dB (Restoration) — statt nur Blend.
- **Ebene 2 (§4):** `residuum_masking.py` als dritter Salience-Blend-Term;
  `_should_skip_masked_phase` überspringt Phasen, deren Defekte vollständig
  ERB-maskiert sind (nur bei echter Maskierung — Pass-Through-Guard neutralisiert);
  Defekt-Countdown weist ERB-maskierte Events aus (Stufe A).
- **Ebene 3 (§5):** `HEARING_TIER_MAP`/`hearing_tier()` im GoalPriorityProtocol;
  strikte Dominanz-Guards in FeedbackChain (intern + UV3-Callback) und im
  End-Gate-Kandidaten-Ranking.
- **Ebene 4 (§6):** `inviting_sound_gate.py` (Fenster-Gate Roughness/Sharpness/
  Ermüdung), läuft nach Goosebumps vor MDEM; Ergebnis in
  `_restoration_context["inviting_gate"]`.
- **Konfliktregel (§7):** Wohlklang-Garantie-Alignment-Guard, MQA-Verdict-
  Kennzeichnung („Messartefakt-Verdacht“ bei gehaltener Hör-Instanz).
- **Wächter (§8a):** Pre-Commit-Hook `aurik-horordnung-calibration`
  (`scripts/horordnung_calibration.py`).
