# Aurik 10 — Spec 01: 15 Musikalische Ziele

> **Einzige normative Quelle** für alle Goal-Schwellwerte, Prioritäten, Adaptive Thresholds
> und Applicability-Regeln. Alle anderen Dateien **referenzieren** hierher.
>
> **Zweck**: Die 15 Ziele operationalisieren §0 — sie messen, ob Aurik bei jeder Importdatei
> unter Berücksichtigung der Quelldatei und der physikalischen Grenzen das maximal mögliche
> Ergebnis erreicht. Ihre Schwellwerte sind so kalibriert, dass sie die qualitativ hochwertigsten
> automatisierten Restaurierungsergebnisse für Musik mit Gesang weltweit ermöglichen.

### §1.10 VocalQualityGate (v10.0.0-Phantom)

6-dimensionale Gesangsqualität mit Delta-Vergleich (vor/nach Phase):
Formant-Integrität, Atem-Natürlichkeit, Sibilanz-Erhalt,
Verständlichkeit, Hörkomfort, Stimmwärme.
Rollback bei Δ < −10 oder Einzelkriterien-Verletzung.

Implementiert in `backend/core/vocal_quality_gate.py`, integriert in
`PhaseInterface._safe_process()` für Vokal-Phasen.

## §1.0a [RELEASE_MUST] Universaler Zielraum (All-Import Contract)

Die 15 Musical Goals sind **nicht** auf einen einzelnen Song oder ein Einzelgenre optimiert,
sondern auf den vollständigen Importraum von Aurik (Material × Ära × Genre × Defektschwere).

**Verbindlich:**

1. Goal-Optimierung muss über eine repräsentative Import-Matrix stabil bleiben.
2. Song-spezifische Heuristiken (Dateiname/Artist/Einzelsong-Whitelists) sind verboten.
3. Anpassung von Goal-Schwellwerten oder Retry-Politiken ist nur zulässig, wenn die
    Gesamtqualität über die Matrix nicht regressiert (kein „lokaler Song-Gewinn" bei
    globalem Qualitätsverlust).

Diese Invariante konkretisiert §0 Klangwahrheit für den gesamten Produktbetrieb.

---

## §v10 Pleasantness-First (2026-07-05)

> **HPE ist oberste Instanz.** PMGG darf Phasen ueberspringen, wenn sie
> den Klang fuer menschliche Ohren verschlechtern (§2.29 v10).
> Siehe backend/core/per_phase_musical_goals_gate.py,
> backend/core/human_pleasantness_estimator.py.
> **Kein Rollback-Verbot mehr.** CausalDefectReasoner kann irren — das Ohr nicht.

---

## §v10.101 — Perzeptuelle Architektur (2026-08-10)

> **Das menschliche Ohr ist der einzige Richter.** Aurik v10.101 wurde fundamental umgebaut:
> JEDE Verarbeitungsentscheidung fragt „Ist der Unterschied hörbar?", bevor sie handelt.

### §v10.101.1 Perzeptuelle Gewichtung der 15 Musical Goals

Die 15 Musical Goals werden ab v10.101 nach **perzeptueller Relevanz** gewichtet.
Technische Messwerte (SNR, THD) sind nur im Bereich oberhalb der Hörschwelle aussagekräftig.

| Goal | Typ | Basis | Perzeptuelle Rechtfertigung |
|------|-----|-------|---------------------------|
| groove | Timing-Präzision | DTW-Onset-Alignment (Bark-Band-constrained) | Rhythmus-Verlust sofort hörbar |
| natuerlichkeit | Klangtreue | MERT-Embedding + Wiener-Entropie | Unnatürlichkeit wird als Erstes wahrgenommen |
| waerme | Musikalische Fülle | Bark-Band-Energie (100-500 Hz) | „Kraftvoll, musikalisch" |
| artikulation | Transienten-Schärfe | Crest-Faktor + Onset-Dichte | Sprachverständlichkeit |
| authentizitaet | Quelltreue | Timbral-Fidelity (Bark) | Charakter-Erhalt |
| transparenz | Durchhörbarkeit | Maskierungs-SMR (Bark) | Details hörbar |
| emotionalitaet | Emotionale Wirkung | Emotional-Arc + Dynamik-Kontrast | Unterbewusstsein |
| brillanz | Höhenpräsenz | Bark 18-24 (3.7-15.5 kHz) | Luft, Offenheit |
| tonal_center | Harmonische Stabilität | Chroma-Vektor + Harmonic-Lattice | Musikalische Korrektheit |
| bass_kraft | Bass-Fundament | Bark 1-5 (20-400 Hz) | Physische Präsenz |
| transient_energie | Impuls-Erhalt | Onset-Preservation (JND-basiert) | Lebendigkeit |
| micro_dynamics | Feindynamik | Crest-Verteilung 200ms | Atmung, Nuance |
| spatial_depth | Räumlichkeit | IACC + Side/Mid-Ratio (Bark) | Binaurale Natürlichkeit |
| timbre_authentizitaet | Klangfarben-Treue | Cepstrale Distanz (Bark) | Wiedererkennbarkeit |
| separation_fidelity | Instrument-Trennung | HPSS-Separation (Bark) | Durchhörbarkeit |

### §v10.101.2 JND-basierte Goal-Relevanz

Ein Goal wird nur dann als „verletzt" gewertet, wenn die Abweichung vom Target
in ≥2 Bark-Bändern die Just-Noticeable-Difference (JND) überschreitet (§G104).
Unhörbare Abweichungen lösen kein Recovery aus.

## §1.2 Die 15 Musikalischen Ziele (Musical Goals) — vollständige Tabelle

Implementiert in `backend/core/musical_goals/musical_goals_metrics.py`,
aufgerufen via `MusicalGoalsChecker.measure_all(audio, sr)`.

| Ziel (Klasse) | Frequenzbereich / Messgröße | Prio | Restoration | Studio 2026 |
| --- | --- | --- | --- | --- |
| **Natürlichkeit** (`NatuerlichkeitMetric`) | Artefaktfreiheit, Rauschen, Klangbild | **1** | ≥ **0.90** | ≥ **0.92** |
| **Authentizität** (`AuthentizitaetMetric`) | Voice Identity, spektraler Fingerabdruck | **1** | ≥ **0.88** | ≥ **0.90** |
| **Tonales Zentrum** (`TonalCenterMetric`) | Chroma-Korrelation Original↔Restauriert, kein Key-Shift > 0 Cent | **2** | ≥ **0.95** | ≥ **0.96** |
| **Timbre-Authentizität** (`TimbralAuthenticityMetric`) | MFCC-Pearson ≥ 0.95, Spectral-Centroid-Korrelation ≥ 0.93, Rolloff-Abw. ≤ 5 % | **2** | ≥ **0.87** | ≥ **0.89** |
| **Artikulation** (`ArticulationMetric`) | Attack-Charakter-Erhalt (Staccato vs. Legato): Transient-Shape-Korrelation ≥ 0.90, Attack-Time-Abweichung ≤ 10 ms | **2** | ≥ **0.88** | ≥ **0.90** |
| **Transient-Energie** (`TransientEnergyMetric`) | Onset-Amplitude-Ratio: E(attack_restored, 5 ms) / E(attack_original, 5 ms) ≥ 0.85 per Onset (geometrisches Mittel aller erkannten Onsets via HPSS + librosa.onset_detect) | **2** | ≥ **0.80** | ≥ **0.83** |
| **Emotionalität** (`EmotionalitaetMetric`) | Dynamik, Ausdruck, Modulationstiefe | **3** | ≥ **0.84** | ≥ **0.87** |
| **Mikro-Dynamik** (`MicroDynamicsMetric`) | Momentane LUFS-Profil-Korrelation (400 ms-Fenster), Crest-Faktor-Erhalt ≤ 1.5 dB | **3** | ≥ **0.88** | ≥ **0.90** |
| **Groove** (`GrooveMetric`) | Mikro-Timing, Swing, Event-Onset-Präzision (DTW ≤ 8 ms RMS) | **3** | ≥ **0.83** | ≥ **0.85** |
| **Transparenz** (`TransparenzMetric`) | Klarheit, Trennung der Klangelemente | **4** | ≥ **0.82** | ≥ **0.85** |
| **Wärme** (`WaermeMetric`) | Primär: Even-Harmonic-Ratio (H2/H4 THD_even/THD_total, ISO 226:2003 gewichtet) — misst wahrgenommene Röhren-/Band-Wärme; Sekundär: Warmth Ratio E(200–800)/E(800–3000) als Spektral-Tilt-Proxy (§9.7.14) | **4** | ≥ **0.75** | ≥ **0.78** |
| **Bass-Kraft** (`BassKraftMetric`) | Bassenergie 20–250 Hz + Virtual Pitch (Missing Fundamental, Obertöne 120–500 Hz) | **4** | ≥ **0.78** | ≥ **0.80** |
| **Separation-Treue** (`SeparationFidelityMetric`) | SDR ≥ 8 dB / SIR ≥ 12 dB nach NMF-Dekomposition | **4** | ≥ **0.80** | ≥ **0.83** |
| **Brillanz** (`BrillanzMetric`) | HF-Klarheit, 8–20 kHz — Sparkle & Air | **5** | ≥ **0.78** | ≥ **0.82** |
| **Raumtiefe** (`SpatialDepthMetric`) | IACC (Interaural Cross-Correlation, Blauert 1997) + Stereobreite + Phantom-Center-Stabilität; IACC < 0.70 → wahrnehmb. Zusammenbruch | **5** | ≥ **0.70** | ≥ **0.74** |

> **v10.0.0 / Spec 09 Pareto-Differenzierung**: Restoration-Modus senkt P3–P5-Böden auf physikalisch erreichbare Werte (Pareto-Konflikte: Bass↔Transparenz [0.7], Brillanz↔Wärme [0.6]). P1/P2 bleiben identisch. Studio 2026 setzt höhere P1/P2-Böden (Enhancement braucht stärkere Naturalness-Guardrails) und niedrigere P3–P5-Böden (materialadaptiv kalibriert aus AMRB).
> **Kanonische Böden**: Die Werte in dieser Tabelle sind `CANONICAL_THRESHOLDS` — implementiert in `backend/core/calibration_matrix.py` (Single Source of Truth: Spec 09, normativ übergeordnet). Song-spezifische Ziele berechnet die adaptive Schicht §1.2b + §2.31 + §09.2 + §2.56 (Material, Ära, Genre, Tonträgerkette, Restorability, Physical Ceiling) — **garantiert wird immer der effektive Zielwert**, nicht der rohe kanonische Boden.
> **Schwellwert-Validierung**: Die Schwellwerte wurden algorithmisch aus AMRB-Benchmarkdaten (10 Szenarien, Ø OQS-Kalibrierung) abgeleitet. Ein ITU-R BS.1534-3 MUSHRA-Hörertest steht als externe Validierung aus (geplant). Bis zur Validierung gelten die Werte als „best engineering estimate“. Änderungen ausschließlich in `calibration_matrix.py` + Spec 09 (Single Source of Truth).

```python
from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

checker = MusicalGoalsChecker(mode="restoration")  # oder "studio_2026"
scores = checker.measure_all(audio, sr)  # Dict[str, float]
effective_targets = resolve_effective_goal_targets(song_context, physical_ceiling)
# Pflicht-Check nach jeder Restaurierung:
assert all(scores[g] >= effective_targets[g] for g in applicable_goals), scores
```

**Invariante (§2.54-konform)**: Am **Pipeline-Ende** müssen alle 15 anwendbaren Ziele ≥ effektivem Zielwert liegen.

### §1.2c Teamwork- statt Dominanz-Prinzip [RELEASE_MUST]

Die 15 Musical Goals sind ein kooperativer Zielvektor. Jede der 64 Phasen liefert mit
phasenspezifischem Algorithmus und adaptiver Staerke einen Teilbeitrag. Ein Einzelziel darf
die Endentscheidung nicht dominieren.

Normative Konsequenzen:

1. Pipeline-Ende MUSS gegen den vollstaendigen 15er-Zielvektor pruefen.
2. Fehlende Einzel-Goal-Messungen duerfen nicht stillschweigend ignoriert werden.
3. Goal-Recovery ist multi-goal-basiert (weighted-gap), nicht single-goal-maximierend.
Einzelphasen dürfen Proxy-Werte vorübergehend senken, wenn:

- **Carrier-Repair** (§2.44 Referenz-Paradoxon): Tonträgerschaden-Inversion verändert Chroma/Centroid intentional
- **Restorative Defektentfernung** (§2.29c Baseline-Capping): aufgeblähte scores_before durch Defekte
- **Iterative Stärke-Anpassung** (§2.54): Phase wird mit reduzierter Stärke erneut versucht, nicht sofort übersprungen

Eine Regression nach der **gesamten Kette** macht das Feature ungültig.

### §1.2a [RELEASE_MUST] Carrier-Recovery-Referenz am Pipeline-Ende (v10.0.0)

**Problem**: Am Pipeline-Ende messen referenzbasierte Goals (timbral_fidelity, MFCC, Centroid, Authentizität) den Output gegen den **degradierten Input**. Bei erfolgreicher Carrier-Chain-Inversion hat sich das Signal intentional stark verändert — die Messung bestraft korrekte Restaurierung als Regression.

**Lösung**: Wenn `metadata["carrier_chain_recovery_ratio"] > 0.15`, wird die End-Referenz für referenzbasierte Goals auf das **best_carrier_checkpoint** verschoben:

```python
# In MusicalGoalsChecker.measure_all() bzw. UV3._final_goals_check():
if metadata.get("carrier_chain_recovery_ratio", 0.0) > 0.15:
    # Referenz ist post-Carrier-Audio (nach Stufe 1–4, vor Enhancement-Phasen)
    reference_audio = metadata["best_carrier_checkpoint"]
else:
    # Standard-Referenz: degradierter Input
    reference_audio = original_audio

# Betroffene Goals (referenzbasiert):
#   timbral_authentizitaet  — MFCC-Pearson, Centroid-Korrelation
#   authentizitaet          — spektraler Fingerabdruck
#   tonal_center            — Chroma-Korrelation
# Nicht-referenzbasierte Goals (natuerlichkeit, brillanz, waerme etc.) bleiben unverändert.
```

**UV3-Pflicht**: Nach der letzten Carrier-Phase (§2.46 Stufe 4) wird `best_carrier_checkpoint = current_audio.copy()` gespeichert und `carrier_chain_recovery_ratio` berechnet:

```python
recovery_ratio = 1.0 - spectral_correlation(pre_carrier_audio, current_audio)
metadata["carrier_chain_recovery_ratio"] = recovery_ratio
metadata["best_carrier_checkpoint"] = current_audio.copy()
```

**Invariante**: Ohne §1.2a-Referenz-Shift würde jede erfolgreiche Carrier-Inversion bei Shellac/Multi-Gen-Material (recovery_ratio > 0.35) einen End-Goals-Fail erzeugen — die Pipeline wäre für genau die Materialien unbrauchbar, für die sie gebaut ist.

**§0d Dreischichtiges Carrier-Recovery-Referenzmodell — Cross-Referenz (normativ)**:

| Ebene | Mechanismus | Spec-Referenz | Beschreibung |
| --- | --- | --- | --- |
| **1. Per-Phase PMGG** | Baseline-Capping | Spec 01 §2.29c + Spec 02 §2.29c | `effective_before = min(measured, threshold + 0.05)` für restorative Phasen |
| **2. End-of-Pipeline Goals** | Carrier-Recovery-Referenz-Shift | **Spec 01 §1.2a** (hier) | Bei `recovery_ratio > 0.15` → End-Referenz = `best_carrier_checkpoint` |
| **3. HPI Export-Gate** | Referenz-Paradoxon | Spec 02 §2.44 | Restorability-abhängiger Referenz-Anker (GP-Memory MERT bei Restorability ≤ 50) |

Alle drei Ebenen MÜSSEN konsistent implementiert sein. Ein Gate (PMGG, CIG, HPI, AFG) darf den Export nie blockieren, weil sich das Signal vom **degradierten** Input entfernt hat — solange es sich dem **physikalischen Ceiling** des erkannten Materials nähert.

### §1.2b [RELEASE_MUST] Effektiver Zielwert und Erreichens-Garantie (v10.0.0)

Für jedes Importstück MUSS Aurik pro anwendbarem Musical Goal genau einen **effektiven Zielwert** berechnen. Dieser Wert ist der einzige harte Zielwert, den Pipeline, PMGG, FeedbackChain, End-Gate, UI und Report verwenden dürfen.

**Definition:**

```python
canonical_floor = CANONICAL_THRESHOLDS[goal]
material_floor = get_material_floor(material_type, goal, is_studio_2026)
song_target = estimate_song_goal_targets(
    is_studio_2026=is_studio_2026,
    goal_weights=song_goal_importance,
    restorability_score=restorability_score,
    era_decade=era_decade,
    genre_label=genre_label,
    material_type=material_type,
    transfer_chain=transfer_chain,
    production_profile=production_profile,
)[goal]
restorability_floor = get_effective_material_floor(
    material_type, goal, restorability_score, is_studio_2026
)
physical_cap = physical_ceiling.ceiling.get(goal, 0.99)
chain_cap = get_chain_end_goal_ceiling(transfer_chain, goal)

effective_target = min(
    max(restorability_floor, song_target),
    physical_cap,
    chain_cap,
)
effective_target = clip(effective_target, 0.20, 0.99)
```

**Garantie:** Nach Abschluss der Restaurierung MUSS für jedes anwendbare Ziel gelten:

```python
final_scores[goal] >= effective_target[goal]
```

Falls diese Bedingung nicht erfüllt ist, ist ein normaler Export verboten. UV3 MUSS dann in dieser Reihenfolge reagieren:

1. `§GOAL_BASELINE_CHECK`/FeedbackChain-Recovery mit `get_goal_recovery_phases(goal)` erneut anstoßen, sofern eine nicht ausgeschöpfte, §0a-konforme Phase existiert.
2. Stärke reduzieren oder auf `best_carrier_checkpoint` zurückrollen, wenn Recovery Artefakte, VQI-Abfall oder HPI-Verlust erzeugt.
3. Wenn `effective_target` trotz zulässiger Recovery physikalisch nicht erreichbar ist, MUSS `effective_target` durch `physical_ceiling`/`chain_cap` begründet abgesenkt und in `metadata["goal_target_limitations"]` dokumentiert werden.
4. Wenn auch der abgesenkte Zielwert nicht artefaktfrei erreichbar ist, MUSS der Exportstatus `degraded` sein und der beste artefaktfreie Checkpoint oder das Original exportiert werden. Ein überprozessiertes Ergebnis ist verboten (§0h).

**Wichtig:** Die Garantie bezieht sich auf den effektiv berechneten, physikalisch gedeckelten Zielwert. Sie darf niemals als Pflicht verstanden werden, ein Shellac-, Kassetten- oder MP3-low-Signal über seine Material- oder Tonträgerketten-Grenze hinaus zu erzwingen.

---

## §2.34 GoalPriorityProtocol — Hierarchie bei Ressourcen-Konflikten

```python
PRIORITY_MAP: dict[str, int] = {
    "natuerlichkeit":        1,   # Rollback bei Verschlechterung
    "authentizitaet":        1,   # Rollback bei Verschlechterung
    "tonal_center":          2,   # Rollback bei Verschlechterung
    "timbre_authentizitaet": 2,   # Rollback bei Verschlechterung
    "artikulation":          2,   # Rollback bei Verschlechterung
    "emotionalitaet":        3,
    "micro_dynamics":        3,
    "groove":                3,
    "transparenz":           4,
    "waerme":                4,
    "bass_kraft":            4,
    "separation_fidelity":   4,
    "brillanz":              5,   # Recovery-Lite: darf Endresultat nicht unkontrolliert regressieren
    "spatial_depth":         5,   # Recovery-Lite: darf Endresultat nicht unkontrolliert regressieren
}
ABORT_PRIORITY_THRESHOLD: int = 2  # Stufe 1+2 verschlechtert → Iteration sofort abbrechen
REGRESSION_EPSILON: float = 0.001

# §2.54: ABORT_PRIORITY_THRESHOLD und REGRESSION_EPSILON sind Notbremsen-Konstanten
# für die FeedbackChain. Sie greifen NUR bei katastrophalen Fällen.
# Die Routine-Steuerung läuft über den iterativen Messen→Handeln→Validieren-Zyklus
# pro Phase (PMGG + PhaseConductor + SongCalibration).
```

### §2.34b [RELEASE_MUST] Cross-Goal-Balancevertrag fuer Waerme und Raumtiefe (v10.0.0)

Bei Goal-Recovery darf `waerme` nicht isoliert maximiert werden, wenn dadurch
`spatial_depth` hoerbar kollabiert. Beide Goals sind als gekoppeltes Paar zu behandeln.

Pflichtregeln:

1. Waerme-Rettung nur bei nachweisbarer Defizitreduktion (`waerme_deficit` sinkt).
2. `spatial_depth`-Einbruch ueber Guard-Cap ist ein Hard-Negativsignal fuer Candidate-Ranking.
3. Bei Konflikt gilt Teamwork-Prinzip: kein Einzelziel darf die Endentscheidung dominieren.
4. Ranking muss konservative Kandidaten bevorzugen, wenn kritische Goals nahe Baseline bleiben.

Normativer Entscheidungsrahmen:

```python
if waerme_deficit_reduced and spatial_depth_drop <= cap:
    candidate_allowed = True
else:
    candidate_allowed = False
```

VERBOTEN:

- Waerme-Boost ohne gleichzeitige Raumtiefen-Pruefung.
- Akzeptanz eines Kandidaten nur aufgrund eines Einzelgoal-Gewinns.

**§2.29 Priority-Aware PMGG Retries (v10.0.0)**:

**Ergänzung v10.0.0 (Team-Koordination):**
PMGG-Retry/Strength-Entscheidungen sind kontextbewusst über `prior_phase_context`
zu steuern, damit Folgephasen (insb. `phase_50`) intentionale Vorphasen-Reparaturen
nicht indirekt neutralisieren. Normative Details: Spec 02 §2.29e, Spec 06 §6.9b.

PMGG-Retries werden prioritätsabhängig budgetiert:

| Priorität | Max Retries | Threshold-Faktor | Verhalten |
| --- | --- | --- | --- |
| P1 | 4 | 1.0× | Volle Retry-Kaskade + Emergency |
| P2 | 4 | 1.0× | Volle Retry-Kaskade + Emergency |
| P3 | 2 | 1.5× (mildere Erkennung) | Reduzierte Kaskade, kein Emergency |
| P4 | 1 | 2.0× (Recovery-Lite) | 1 konservativer Retry, kein Emergency |
| P5 | 1 | 2.5× (Recovery-Lite) | 1 konservativer Retry, kein Emergency |

Implementierung: `per_phase_musical_goals_gate.py` — `_PRIORITY_MAX_RETRIES`, `_PRIORITY_THRESHOLD_FACTOR`, `_max_regression_priority_aware()`.

**Normative Aufrufstellen**:

```python
# In FeedbackChain.run():
gpp = GoalPriorityProtocol()
abort_result = gpp.should_abort_iteration(scores_before, scores_after)
if abort_result.should_abort:
    best_result = previous_best
    break

# In ExcellenceOptimizer — MOO-Pareto-Konflikt:
conflict_result = gpp.resolve_conflict(goal_a, goal_b, delta_a, delta_b)
# conflict_result.winner = priorisiertes Ziel

# In UnifiedRestorerV3._profiled_phase_call() — §2.56a all-phase coupling:
# goal_weights steuern bounded harmonic_adaptation_scalar (advisory-only),
# der implizite strength/wet-dry für alle Phasen 01-64 harmonisiert.
```

## §2.35 Vocal-Exzellenz-Zusatzmetriken (PFLICHT fuer Gesangsmaterial)

Wenn PANNs/Gender/Vocal-Detektoren Gesang erkennen, werden zusaetzlich zu den 15 Musical Goals folgende Vocal-Zielwerte geprueft:

| Ziel | Messgroesse | Mindestwert |
| --- | --- | --- |
| Formant-Stabilitaet | mittlere Formant-Drift F1/F2 ueber Vokal-Segmente | <= 35 Hz |
| Sibilance-Natuerlichkeit | 5-10 kHz-Energieabweichung in Frikativen | <= 1.5 dB |
| Konsonanten-Klarheit | Plosiv/Frikativ-Onset-Praezision vs. Original | <= 6 ms |

**Invariante:** Vocal-Zusatzmetriken duerfen nie auf Kosten von P1/P2 erzwungen werden.

## §2.35b Vocal-Proximity-Score (PFLICHT bei PANNs Singing ≥ 0.35)

**Psychoakustische Motivation**: Studiogesang klingt, als befände sich der Sänger direkt
vor dem Hörer — durch klare Konsonanten, natürliche Atemgeräusche in Pausen und erhaltene
frühe Reflexionen (C80). Verlust dieser Eigenschaften durch Dereverb oder Over-Enhancement
erzeugt wahrgenommene Distanz, die unmittelbar das Immersionsgefühl (§8.3 Tiefen-Immersion)
zerstört. Dieser Score ist mit bestehenden Vocal-Metrics (§2.35) orthogonal — er misst
räumlich-zeitliche Vertrautheit, nicht spektrale Genauigkeit.

**Formel**:

```python
proximity_score = (
    konsonanten_transient_energy_ratio  # Plosiv/Frikativ-Onsets erhalten
    × breathiness_ratio                 # natürliche Atemgeräusche in Pausen
    × early_reflection_preservation     # Raumcharakter (C80) nicht zerstört
)
```

| Komponente | Messung | Schwelle |
| --- | --- | --- |
| `konsonanten_transient_energy_ratio` | Plosiv/Frikativ-Onset-Energie (Output) / Onset-Energie (Input) | ≥ 0.75 |
| `breathiness_ratio` | RMS in 100 ms-Pausen / Rauschboden (Output) vs. Input-Verhältnis | ≥ 0.60 |
| `early_reflection_preservation` | C80-Verhältnis (Output) vs. (Input): Frühenergie-Anteil ≤ 80 ms | ≥ 0.80 |

**Schwellwerte**:

- `proximity_score ≥ 0.75` → "nah" (Vokalintimität bestanden)
- `proximity_score ∈ [0.70, 0.75)` → WARNING: `metadata["vocal_proximity_warning"]` setzen
- `proximity_score < 0.70` → Dry/Wet-Rescue für Dereverb-Phase empfehlen

**Aktivierungs-Bedingungen**:

- `panns_singing_confidence ≥ 0.35` ODER `vocal_genre = True`
- Nur im finalen Gate-Check (`MusicalGoalsChecker`), nicht im PMGG-Delta

**Invarianten**:

- Vocal-Proximity-Score darf nie auf Kosten von P1/P2 erzwungen werden
- Phase-Rollback nur bei `proximity_score < 0.70` UND gleichzeitig HPI > Rollback-Schwelle
- Protokollierung immer: `RestorationResult.metadata["vocal_proximity_score"]`

> Implementierung: `backend/core/musical_goals/ki_hearing_model.py` — `compute_vocal_proximity_score(audio_orig, audio_restored, sr, vocal_segments)`
> Aufruf: `backend/core/musical_goals/musical_goals_metrics.py` — `MusicalGoalsChecker.measure_all()` wenn Vocal erkannt

---

## §2.35c [RELEASE_MUST] VocalQualityIndex (VQI) — Gesangs-Gesamtqualitäts-Gate (v10.0.0)

**Motivation**: Die bestehenden 15 Musical Goals und §2.35/§2.35b messen jeweils Teilaspekte des Gesangs. Ein zusammengesetzter `VocalQualityIndex` liefert ein einzelnes, messbares Qualitätssignal für Gesangsmaterial — und ermöglicht damit, einen extern belegbaren Spitzenanspruch für Gesangsrestaurierung überhaupt erst zu prüfen.

**Formel**:

```python
VQI = (
    0.30 × singer_identity_cosine     # Stimmidentität erhalten
  + 0.25 × formant_stability_score    # F1–F4 nicht verschoben
  + 0.20 × articulation_score         # aus §2.35 (Konsonanten-Klarheit)
  + 0.15 × proximity_score            # aus §2.35b (Intimität / Nähe)
  + 0.10 × sibilance_naturalness      # aus §2.35 (5–10 kHz)
)
```

**Schwellwerte**:

| VQI | Bedeutung | Pipeline-Reaktion |
| --- | --- | --- |
| ≥ 0.88 | Gesang auf internem Spitzenniveau | Export freigegeben |
| 0.80–0.87 | Sehr gut | Export freigegeben, `metadata["vqi_tier"] = "good"` |
| 0.72–0.79 | Akzeptabel | WARNING + Dry/Wet-Rescue empfohlen |
| < 0.72 | Unter Schwelle | Recovery-Kaskade: Rollback auf bestes Vokal-Checkpoint |

**Singer-Identity-Gate** (`singer_identity_cosine`):

- Modell: **Resemblyzer** (GE2E-Loss, 256-dim d-vector) als Primär; **X-Vector** (ECAPA-TDNN) als Fallback
- Messung: Cosinus-Ähnlichkeit des Vokal-Embeddings vor vs. nach Pipeline (nur auf Vokal-Stem)
- Pflicht-Vorab-Messung: `singer_id_pre = resemblyzer.embed(vocal_stem_pre)` nach TDP, vor Phase 03
- Post-Pipeline: `singer_identity_cosine = cos_sim(singer_id_pre, resemblyzer.embed(vocal_stem_post))`
- Schwellwert: ≥ 0.92 Pflicht; < 0.88 → Phase-Rollback der letzten Vokal-bearbeitenden Phase
- Aktivierung: `panns_singing_confidence ≥ 0.35` UND Dateilänge ≥ 10 s
- DSP-Fallback: wenn Resemblyzer nicht verfügbar → MFCC-Pearson × Centroid-Korrelation als Proxy (`singer_identity_dsp_fallback = True` in metadata)

**Formant-Stabilitäts-Score** (`formant_stability_score`):

```python
# Aus §2.35 normalisiert auf [0, 1]:
f1_drift_hz, f2_drift_hz = measure_formant_drift(vocal_pre, vocal_post, sr)
formant_stability_score = max(0.0, 1.0 - (f1_drift_hz + f2_drift_hz) / (2 × 35.0))
# 35 Hz = Schwellwert aus §2.35; bei exakt 0 Hz Drift → score = 1.0
```

**Era-adaptive Formant-Toleranz** [RELEASE_MUST]: Alle Formant-Rollback-Gates (UV3, `VocalNoHarmGate`, `phase_03`, `phase_29`, `phase_42` und jede neue Vokalphase) MÜSSEN die Toleranz über `resolve_formant_tolerance_db()` aus `backend/core/musical_goals/era_vocal_profile.py` beziehen. Ein fixer `threshold_db=2.0` ist nur noch der moderne Fallback für Material ohne Ära-Kontext. Historische Profile dürfen höhere F1/F2-F4-Toleranzen nutzen, damit historische Vokalästhetik nicht als Defekt klassifiziert wird.

**Vokalregister-adaptiver NR-Modus** (§2.35c-Ergänzung zu §4.5 NR-Routing):

DeepFilterNet `energy_bias=-6 dB` gilt pauschal — Vokalregister verfeinern dies:

| Vokalregister | Erkennung | energy_bias Anpassung |
| --- | --- | --- |
| Kopfstimme / Falsett | F0 > 600 Hz, Energie 3–8 kHz dominant | −3 dB (aggressivere NR — Falsett hat wenig Harmonik unten) |
| Bruststimme | F0 200–600 Hz, Energie < 3 kHz dominant | −6 dB (Standard) |
| Vocal Fry / Subharmonisch | F0 < 100 Hz, unregelmäßige Perioden | −9 dB (schützend — Fry-Charakter ist keine Defekt-Rauschart) |
| Flüstern | Spektrale Flachheit > 0.6, F0 nicht klar | −9 dB (Rauschen und Flüstern sind spektral ähnlich) |

Erkennung via: FCPE F0-Tracking + spektrale Flachheit (scipy.signal) im Vokal-Segment.

**Invariante**: VQI darf nie auf Kosten von P1/P2 erzwungen werden. Bei P1/P2-Konflikt haben Natürlichkeit/Authentizität Vorrang. VQI-Fail löst keine Phase-Pipeline-Unterbrechung aus — er ist post-hoc Recovery-Signal.

**Protokollierung**: `RestorationResult.metadata["vqi"]`, `metadata["singer_identity_cosine"]`, `metadata["singer_id_dsp_fallback"]`.

> Implementierung: `backend/core/musical_goals/vocal_quality_index.py`
> Aufruf: `MusicalGoalsChecker.measure_all()` nach Phase-Pipeline, wenn Vocal erkannt

## §2.35d [RELEASE_MUST] Gesangsexzellenz-Contract — Maximale Vokalqualität ohne Artefaktrisiko

Dieser Contract verdichtet die für Gesang relevanten Invarianten aus §2.35b, §2.35c,
§2.36 und §0a zu einem einzigen CI-prüfbaren Satz von Pflichten. Ziel ist nicht
aggressiveres Vocal-Processing, sondern maximale Gesangsqualität unter striktem
`Primum non nocere`.

**Vier Pflicht-Invarianten**:

1. **Frühe und konservative Aktivierung**:
    `panns_singing_confidence >= 0.35` aktiviert Gesangsmetriken und §2.36.
    Voller Vocal-Chain-Einsatz bleibt an die harte Gesangs-Evidenz `>= 0.40` gebunden;
    der Bereich `0.35–0.40` bleibt Soft-Zone mit reduzierter Stärke.
2. **Gesangsidentität vor Bearbeitungsgewinn**:
    `singer_identity_cosine >= 0.92` bleibt Pflichtziel; `VQI < 0.72` löst Recovery aus.
    Diese Gates dürfen Gesang verbessern, aber niemals die Sängeridentität oder
    Natürlichkeit opfern.
3. **Phonem-bewusste Kettenreihenfolge**:
    Die kanonische Vocal-Kette lautet
    `phase_19_de_esser -> phase_42_vocal_enhancement -> phase_43_ml_deesser -> phase_58_lyrics_guided_enhancement`.
    LyricsGuidedEnhancement kommt zuletzt, damit phonem-klassenbewusste Salienz-Steuerung
    auf einer bereits gesäuberten, aber noch nicht verfremdeten Vokalbasis arbeitet.
4. **Modus-Trennung ohne Gesangsverlust**:
    `phase_42_vocal_enhancement` bleibt in `restoration` verboten (§0a),
    `phase_58_lyrics_guided_enhancement` bleibt bei erkannter Stimme dennoch Pflicht,
    weil es keine Stem-Verfremdung ist, sondern phonem-bewusste Defektsteuerung.
5. **Export-Metadaten ohne Verlust**:
    Wenn das VQI-Gate aktiv war, MUSS `RestorationResult.metadata` die Felder
    `vqi`, `singer_identity_cosine`, `singer_id_dsp_fallback` und `vqi_tier`
    aus dem finalen UV3-Pfad enthalten. Ein bloß internes Logging oder eine nur
    phasenlokale Akkumulation genügt nicht.

**Artefaktfreiheits-Klausel**:

- Dieser Contract hebt **keine** P1/P2-Schutzregel auf.
- Gesangsverbesserung ist nur gültig, wenn Natürlichkeit/Authentizität erhalten bleiben.
- Kein Vokal-Gate darf eine aggressivere Bearbeitung erzwingen, wenn dadurch hörbare
  Artefakte, Formant-Verfärbungen oder synthetische Präsenz entstehen.

---

## §2.36 Pareto-Tie-Break nach Hoerprioritaet

Bei mehreren Pareto-aequivalenten Kandidaten gilt folgende Tie-Break-Reihenfolge:

1. kleinste P1/P2-Regression,
2. hoechster Vocal-Score (falls Gesang erkannt),
3. geringste Artefaktwahrscheinlichkeit (musical noise, chirps, metallic tails),
4. niedrigere Laufzeit nur, wenn Punkte 1-3 gleichwertig sind.

---

## §2.32 GoalApplicabilityFilter — Physikalisch irrelevante Ziele deaktivieren

```python
ALWAYS_APPLICABLE: frozenset[str] = frozenset({
    "natuerlichkeit", "authentizitaet", "emotionalitaet",
    "transparenz", "timbre_authentizitaet", "artikulation",
})
```

**Deaktivierungs-Regeln:**

| Ziel | Deaktiviert wenn |
| --- | --- |
| `SpatialDepthMetric` | EraResult.decade ≤ 1950 UND M/S-Korrelation ≥ 0.95 (Mono-Aufnahme) **ODER** Transfer-Chain-Ende ist Lossy-Codec (`mp3_low`, `mp3_high`, `aac`, `streaming`) UND L/R-Korrelation ≥ **0.83** (Near-Mono-Codec: Joint-Stereo-Kompression kollabiert Raumcues physikalisch — bestätigt cassette+mp3_low corr=0.8507, v10.0.0) |
| `SeparationFidelityMetric` | Mono-Quelle ODER PANNs < 2 Instrumente mit confidence ≥ 0.4 **ODER** Transfer-Chain-Ende ist Lossy-Codec UND L/R-Korrelation ≥ 0.83 (Joint-Stereo-Kodierung zerstört Stereo-Separation physikalisch — V52-Fix S7: parallele Regel zu `spatial_depth`; gleiches `_CODEC_JOINT_STEREO_MATS`-Set) |
| `BrillanzMetric` | Quell-Bandbreite < 8 kHz UND AudioSR nicht geladen |
| `TonalCenterMetric` | MaterialType = WAX_CYLINDER (Fix K, v10.0.0: SNR-Bedingung entfernt — K-S-Key-Detection ist SNR-invariant gemäß §9.7.11) |
| `GrooveMetric` | Dateilänge < 10 s ODER PANNs Percussion confidence < 0.15 |
| `MicroDynamicsMetric` | Dateilänge < 20 s ODER Original-LUFS-Varianz < 0.5 LU |

Filter läuft EINMAL pro Restaurierung (nach MediumClassifier + EraClassifier).
Inapplicable Goals: im UI grau ausgeblendet, in `RestorationResult.goal_applicability` gespeichert.

**Chain-End-Codec-Ausschluss** (§2.32a, v10.0.0): `evaluate_goal_applicability()` empfängt `transfer_chain: list[str] | None`. Wenn das Ketten-Ende ein Lossy-Codec ist und der primäre Träger analog ist, gelten die erweiterten Deaktivierungs-Regeln. Schwellwert für Near-Mono-Erkennung: L/R-Korrelation ≥ **0.83** (nicht 0.88 — cassette/analoges Stereo-Tape hat typisch corr ≈ 0.85–0.87; mit 0.88 würde Ausschluss nie feuern). `evaluate_goal_applicability()` übergibt dazu `transfer_chain` auch an `calibration_matrix.get_material_floor()` für korrekte Bodenberechnung (§09.13).

**Invariante (§2.32b)**: Inapplicable Goals aus UV3 (`RestorationResult.goal_applicability`) MÜSSEN über `RestaurierErgebnis.goal_applicability` (V51) vollständig an AurikDenker und weiter an `ExzellenzDenker.messe_und_repariere(inapplicable_goals=...)` propagiert werden — sonst zählen physikalisch unmögliche Scores als Violations und triggern Over-Processing (V49). `AurikErgebnis` muss ebenfalls `goal_applicability`-Feld für externe Caller besitzen (V51).

---

## §2.31 AdaptiveGoalThresholds — Material- und ära-adaptive Schwellwerte

> **§2.54 Einordnung**: AdaptiveGoalThresholds definieren die **Ziel-Schwellwerte** für Musical Goals —
> nicht die Guard-Schwellwerte für Phase-Rollback. Die Guard-Drift-Toleranzen werden separat
> über `compute_adaptive_drift_tolerance()` berechnet (§2.48/§2.54). Beide Systeme konsumieren
> Material/Era/Restorability, aber zu unterschiedlichen Zwecken:

- AdaptiveGoalThresholds → „Was ist am Pipeline-Ende für diesen Song erreichbar?"
- Adaptive Drift-Toleranz → „Wie viel darf eine Einzelphase temporär verschlechtern?"

**Adaptierungs-Algorithmus (5 Schritte):**

1. **Base-Thresholds** aus MusicalGoalsChecker (Startpunkt)
2. **Material-Prior** (physikalische Bandbreitengrenzen):
   - SHELLAC/WAX_CYLINDER: `brillanz_threshold → min(0.85, bw_hz/20000*0.85+0.20)`, `spatial_depth → 0.30` (Mono)
   - VINYL: `separation_fidelity_threshold → 0.76`
   - DAT/CD_DIGITAL: alle Schwellwerte unverändert
3. **Ära-Prior** (EraClassifier.decade):
   - decade ≤ 1940: `spatial_depth_threshold → 0.30`
   - decade ≤ 1960: `spatial_depth_threshold → 0.55`
   - decade ≥ 1970: alle Spatial-Thresholds Standard
4. **Restorability-Skalierung:**
   - restorability ≥ 70: scale_factor = 1.00
   - restorability 50–69: scale_factor = 0.93
   - restorability 30–49: scale_factor = 0.85
   - restorability < 30: scale_factor = 0.75
5. **Physical Ceiling Clamp**: `adaptive_t = min(adaptive_t, physical_ceiling[goal])`

```python
# MaterialQuality Enum (backend/core/musical_goals/adaptive_goals_system.py):
class MaterialQuality(Enum):
    PRISTINE   = "pristine"    # Studio-Qualität
    EXCELLENT  = "excellent"
    GOOD       = "good"
    FAIR       = "fair"        # MP3 192 kbps
    POOR       = "poor"        # MP3 128 kbps, Cassette
    VERY_POOR  = "very_poor"   # Stark degradiert
    EXTREME    = "extreme"     # Telefon, Walkie-Talkie

# Einstiegspunkt:
from backend.core.musical_goals.adaptive_goals_system import get_adaptive_goals_and_config
thresholds, config, quality_assessment = get_adaptive_goals_and_config(audio, sr)
```

**Invarianten:**

- Adaptierte Schwellwerte NIEMALS höher als Original-Schwellwerte
- Absolute Untergrenze: adaptive_t ≥ 0.50 (unter 0.50 → Goal deaktivieren)
- NaN in restorability_score → alle Schwellwerte auf Original-Werte

### §2.31d Kombinierte Extrembedingungen (v10.0.0)

Wenn **mehrere** erschwerende Faktoren gleichzeitig vorliegen, kaskadieren die Adaptionen:

| Kombination | Zusätzliche Anpassung |
| --- | --- |
| restorability < 20 **+** Material ∈ {SHELLAC, WAX_CYLINDER} | scale_factor = 0.65 statt 0.75; alle P3–P5 Goals → Untergrenze 0.50; Pipeline-Ziel = „Hörbar machen" |
| restorability < 30 **+** Era ≤ 1940 | Vintage-Aesthetics-Schutz verstärken: H2/H4-Preservation-Guard → G_FLOOR_HARMONIC = 0.92; kein Brillanz-Enhancement |
| Material = SHELLAC **+** Era ≤ 1930 **+** BW < 5 kHz | Brillanz-Goal deaktivieren (physikalisch unmöglich); Wärme-Goal als nicht-bindend markieren |
| Dateilänge < 10 s | GrooveMetric + MicroDynamicsMetric + EmotionalArcPreservation deaktivieren; FeedbackChain max 2 Iterationen |
| Dateilänge > 60 min | SegmentAdaptiveProcessor aktivieren; DefectScanner auf 3×60-s-Segmente (Anfang/Mitte/Ende) |

### §2.31e Prior-Konflikt-Auflösung (v10.0.0)

Wenn Ära-Prior und Material-Prior widersprüchliche Anpassungen ergeben:

- **Physikalische Grenzen** (BW, Noise-Floor, Frequenzgang): **Material-Prior hat Vorrang** — das tatsächliche physikalische Medium bestimmt, was maximal erreichbar ist
- **Ästhetische Entscheidungen** (Vintage-Wärme, Raumhall, Soft-Saturation): **Ära-Prior hat Vorrang** — der Zeitgeist der Aufnahme bestimmt, welcher Klangcharakter bewahrt wird
- **Defekt-Schwellen** (Click-Sensitivity, Hum-Threshold): **Material-Prior hat Vorrang** — analoge Medien haben andere Artefakt-Signaturen als digitale
- **Genre-Profil vs. alle anderen Priors**: Genre-Profil-`*_enabled: False`-Keys sind absolute Overrides (§2.20 Spec 03)

**Restorability-Skalierungsfaktoren — Formale Ableitung:**
Die Stufenwerte 1.00 / 0.93 / 0.85 / 0.75 sind aus dem PhysicalCeilingEstimator hergeleitet:

```python
# Formale Herleitung (normativ): scale_factor = ceiling(goal) / baseline_threshold
# Die Stufen approximieren den integralen Ø der Ceiling-Kurven über alle 15 Goals
# pro Restorability-Klasse (gemessen auf 500 AMRB-Testdateien):
# ≥ 70: ceiling_avg = 0.97 → scale = 1.00
# 50–69: ceiling_avg = 0.90 → scale = 0.93
# 30–49: ceiling_avg = 0.82 → scale = 0.85
# <  30: ceiling_avg = 0.73 → scale = 0.75
# Heuristik-Einsatz VERBOTEN — Stufen müssen aus PhysicalCeilingEstimator.ceiling_avg()
# aktualisiert werden, wenn neue AMRB-Szenarien hinzukommen.
```

---

## §2.33 PhysicalCeilingEstimator — Informationstheoretische Qualitätsdecke

**Musical-Goal-Ceiling-Mapping (empirisch aus AMRB-Daten):**

```python
HEADROOM_THRESHOLD: float = 0.03   # Verbesserung < 3 % → keine weiteren Iterationen

# Ceiling-Formeln:
natuerlichkeit_ceiling  = sigmoid((mean(SNR_b) − 5) / 5) × 0.97 + 0.03
brillanz_ceiling        = sigmoid((bw_hz − 8000) / 2000) × 0.95
spatial_depth_ceiling   = sigmoid(stereo_decorrelation × 10) × 0.92
groove_ceiling          = 1 − max(0, wow_flutter_hz − 0.5) × 0.10
tonal_center_ceiling    = sigmoid(snr_tonal_bands × 2) × 0.98
# Alle anderen Goals: 0.98 (konservative Obergrenze)

# FeedbackChain-Terminierung:
# further_optimization_worthwhile = False wenn alle Goals:
#   current_score ≥ ceiling − HEADROOM_THRESHOLD
```

Nutzer-Meldung wenn Decke erreicht (Deutsch):
> „Das Beste aus dieser Aufnahme wurde herausgeholt — die physikalischen Grenzen des Quellmaterials sind erreicht."

---

## §8.2 Perceptuelle Verpflichtungen (vollständig)

> **§2.54 Kontext**: Die untenstehenden Werte sind **Pipeline-Ende-Ziele**, nicht per-Phase-Guards.
> Einzelphasen dürfen vorübergehend davon abweichen (Carrier-Repair, Baseline-Capping, iterative Stärke-Anpassung).
> Die Werte werden durch AdaptiveGoalThresholds (§2.31) material-/era-/restorability-adaptiv skaliert.

1. **Musikalische Natürlichkeit**: MERT-Naturalness-Score ≥ 0.7
   > MERT (Li et al. ICLR 2024) ist ein Music-Understanding-Foundation-Model, kein
   > designierter MOS-Schätzer. `harmonicity` ist ein kalibrierter Proxy-Score.
   > Kalibrierung: Pearson-Korrelation MERT-harmonicity ↔ VERSA-MOS = 0.74 (n=312 Testdateien).
   > Bei VERSA-MOS verfügbar: VERSA hat Vorrang; MERT-Score dient als Schnellprüfung.
2. **Harmonische Kohärenz**: Harmonizitäts-Ratio ≥ 0.85 (via `MertPlugin.analyze().harmonicity`)
3. **Dynamik-Erhalt**: LUFS-Differenz ≤ 1 LU
4. **Transientenerhalt**: Attack-Zeiten ≤ ±2 ms Änderung
5. **Tonale Stabilität**: Chroma-Pearson ≥ 0.95
6. **Groove**: Event-Onset-DTW ≤ 8 ms RMS — kein Begradigen von Swing/Rubato
7. **Pass-Through-Invariante** (SNR > 40 dB): PQS-MOS-Verlust ≤ 0.05, alle 15 Goals ±0.02, LUFS ≤ 0.3 LU, Chroma ≥ 0.99
8. **Rauschboden** (modus-differenziert):
    - **Restoration**: Analoge Tonträger-Rauschböden werden als reparierbare Trägerdefekte behandelt. Der Export zielt für `shellac`, `wax_cylinder`, `lacquer_disc`, `wire_recording`, `vinyl`, `tape`, `reel_tape` und `cassette` auf CD-ähnlichen Rauschboden statt analogem Hiss-/Oberflächenrauschen. Bei nötiger Resttextur-Auffüllung: Zielprofil `cd_digital`, ca. −74 dBFS, Testanker ≤ −68 dBFS; keine analoge Mindestboden-Reinjektion.
    - **Studio 2026**: ≤ −72 dBFS, A-gew. ≤ −75 dB(A), 0 Musical-Noise-Events in Stille
    - **Beide Modi**: 0 Musical-Noise-Events in Stille-Segmenten (Musical Noise ist immer ein Artefakt)
9. **Mikro-Dynamik**: Pearson des 400 ms LUFS-Profils ≥ 0.92, Crest-Faktor ≤ 1.5 dB
10. **Vintage Aesthetics** (automatisch via EraClassifier):
    - 1920–1940: Rolloff ≤ 7 kHz nicht künstlich erweitern
    - 1940–1955: Röhren-Kompressions-Fingerabdruck erhalten (H2, H4 ∈ [−30, −20] dBr)
    - 1955–1965: RT60 ∈ [1.2, 2.0] s erhalten (kein aggressives Dereverb)
    - 1965–1975: Tape-Saturation-Signatur nicht entfernen
11. **Kompetitiver Benchmark**: Aurik ≥ iZotope RX 11 in ≥ 7/10 AMRB-Szenarien
12. **Emotionaler Dynamik-Bogen**: Arousal-Pearson ≥ 0.85, Valence-Pearson ≥ 0.80, Klimax-Peak-Abw. ≤ 2 Segmente

---

## §2.56 Song-Goal-Importance — Per-Song Goal Weighting (v10.0.0) [RELEASE_MUST]

Die 15 Musical Goals bilden eine Pareto-Front: nicht alle können gleichzeitig vollständig erfüllt werden,
da sie sich physikalisch teilweise ausschließen (z.B. Brillanz vs. Wärme, Transparenz vs. Raumtiefe).
Für jedes Stück muss die richtige Gewichtung aus dem musikalischen Kontext berechnet werden.

**Grundprinzip**: Jeder Song erhält vor der Phase-Pipeline ein individuelles Gewichtungsprofil
`SongGoalImportance` mit 14 Gewichten ∈ [0.3, 2.0]. Diese Gewichte bestimmen, welche Goals
bei diesem konkreten Song Vorrang genießen und welche toleranter behandelt werden.

### §2.56a 5-Stufen-Gewichtungsarchitektur (v10.0.0)

Die Berechnung erfolgt in **5 multiplikativen Stufen** + Soft-Cap + Bounds:

**Stufe 1 — Label-basiert (Genre/Era/Material/Vocal/Restorability):**

| Schritt | Beschreibung | Parameter |
| --- | --- | --- |
| 1a | **Genre-Profil** (16 Profile: Klassik, Oper, Jazz, Rock, Metal, Electronic, Hip-Hop, Pop, Schlager, Soul/R&B, Blues, Country, Folk, Reggae, Latin, Funk, Gospel) | `genre_label` → Alias-Resolution (`_GENRE_ALIASES`) → Basis-Gewichte. Unbekannt → neutral (1.0) |
| 1b | **Ära-Modifikator** (1900er–1980er) multiplikativ auf Genre | `era_decade` → z.B. 1920er: Brillanz ×0.5 (7 kHz Bandbreite) |
| 1c | **Material-Modifikator** (trägertypisch) | `material_type` → z.B. Vinyl: Bass ×1.3, Shellac: Transparenz ×0.7 |
| 1d | **Vokal-Boost** (konfidenzgewichtet) | `vocal_detected`, `vocal_confidence` → Artikulation ×1.3, Emotionalität ×1.2, Authentizität ×1.2, TransparenzVokal ×1.15, TonalCenter ×1.12, TimbreAuth ×1.1, SeparationFidelity ×1.1 (Quellen: Kreiman & Sidtis 2011; Marjieh et al. 2023; Bregman 1990; McDermott 2009 Curr Biol; London 2012; Repp & Su 2013) |
| 1e | **Restorability-Adjustment** | `restorability_score` < 40 → P3–P5-Gewichte ×0.85 |
| 1f | **Studio 2026-Modus** | `is_studio_2026` → Transparenz ×1.2, Brillanz ×1.2, Separation ×1.1 |

**Stufe 2 — Audio-abgeleitet (reale Signalanalyse):**

| Schritt | Feature | Einheit | Schwellwerte | Beispiel-Effekt |
| --- | --- | --- | --- | --- |
| 2a | SNR | dB | < 15: noisy, > 40: clean | noisy → transparenz ↑, waerme ↑ |
| 2b | Effektive Bandbreite | Hz | < 6000: BW-limited, 6k–12k: medium, > 18k: wideband | BW-limited → brillanz ↓ 0.7, waerme ↑ |
| 2c | Dynamikumfang | dB | < 20: compressed, > 50: dynamic | compressed → micro_dynamics ↑ |
| 2d | Stereo-Mono-Kompatibilität | [0, 1] | < 0.4: poor, > 0.9: good | poor → spatial_depth ↑ |
| 2e | BPM | Schläge/min | < 60: slow, > 110: fast | slow → spatial_depth ↑, groove ↓ |
| 2f | Defekt-Schwere | Dict[str, float] | max() über Familien: Rauschen, Knistern, HF-Verlust, Wow | Knistern hoch → transparenz ↑, groove ↑ |
| 2g | Spektraler Tilt | dB/Oct | < −4: dark, > −1: bright | dark → brillanz ↓, waerme ↑ |

**Stufe 2i — Tonträgerketten-Degradation (§2.46/§2.46a):**

| Feature | Einheit | Schwellwerte | Effekt |
| --- | --- | --- | --- |
| `transfer_generation_count` | int | 2: minor, 3: significant, ≥ 4: severe | Mehr Generationen → natuerlichkeit ↑, transparenz ↑; brillanz ↓ bei ≥ 4 |
| `cumulative_hf_loss_db` | dB | > 6: moderate, > 12: severe | > 12 dB → brillanz ↓ 0.85 |
| `source_fidelity_confidence` | [0, 1] | < 0.5: uncertain | Niedrig → alle Carrier-Adjustments ×confidence gedämpft |

**Stufe 3 — Psychoakustisch (Zwicker/Aures/ISO 11172-3):**

| Schritt | Feature | Einheit | Schwellwerte | Effekt |
| --- | --- | --- | --- | --- |
| 3a | Roughness (Zwicker) | asper | > 0.5: rough, < 0.2: smooth | rough → transparenz ↑, waerme ↑ |
| 3b | Sharpness (Bismarck) | acum | > 0.5: sharp, < 0.2: dull | sharp → waerme ↑ |
| 3c | Spectral Flatness | [0, 1] | > 0.3: noisy, < 0.05: tonal | noisy → transparenz ↑; tonal → tonal_center ↑ |
| 3d | Tonality | [0, 1] | > 0.5: tonal, < 0.15: atonal | tonal → tonal_center ↑; atonal → groove ↑ |
| 3e | Frequenzbalance | Dict[bass/mid/treble/air] | bass > 0.6: bass-heavy, treble < 0.1: treble-starved | bass-heavy → bass_kraft ↑; treble-starved → brillanz ↓ |
| 3f | Maskierte Komponenten | [0, 1] | > 0.3: masked | masked → transparenz ↑; **Sanity-Guard**: ratio ≥ 0.95 oder ≤ 0.01 → ignoriert (Messartefakt) |
| 3g | Perzeptueller Schwerpunkt | Bark | < 4.0: dark, > 8.0: bright | dark → waerme ↑, brillanz ↓ |

**Stufe 4 — Vokal/Harmonik/Transient (Musikerhaltung):**

| Schritt | Feature | Messverfahren | Kalibrierter Bereich | Schwellwerte | Effekt |
| --- | --- | --- | --- | --- | --- |
| 4a | HNR | Multi-Frame Pitch-Period AC (50 ms Frames, `find_peaks(height=0.1)` in [sr/2000, sr/50], Median) | Full-Mix: [−10, +20] dB | > 5 dB: clean, < 0 dB: noisy | clean → timbre ↑, artikulation ↑; noisy → transparenz ↑ |
| 4b | Harmonische Kohärenz | STFT Spectral-Peak-Consistency (`PsychoAcousticMetrics.calculate_harmonic_coherence`) | [0, 1] | > 0.7: strong, < 0.3: weak | strong → tonal_center ↑, waerme ↑; weak → groove ↑ |
| 4c | Crest Factor | `20 log10(percentile99.9 / RMS)` — robust gegen Impuls-Artefakte | 99.9-Perzentil: [6, 16] dB | > 12 dB: dynamic, < 7 dB: compressed | dynamic → micro_dynamics ↑; compressed → micro_dynamics ↑ (Repair-Priorität) |
| 4d | Transient Density | STFT Spectral-Flux + adaptive Threshold + 50 ms Min-Gap (`dsp.dtw_groove.detect_onsets`) | [0, ~20]/s | > 8/s: percussive, < 2/s: sustained | percussive → groove ↑, micro_dynamics ↑ |

> **Alle graduiert**: Schwellwerte nutzen `_f = min((val − threshold) / range, 1.0)` statt binärer Flags.

**Stufe 5 — Cross-Feature-Interaktionen (superadditive Effekte):**

| ID | Interaktion | Bedingung | Effekt |
| --- | --- | --- | --- |
| 5a | Roughness × Low SNR | rough > 0.3 AND snr < 20 dB | transparenz ↑↑, artikulation ↑ (Intelligibilitätskrise) |
| 5b | HNR × Vocal | HNR > 5 dB AND vocal_detected | timbre ↑, natuerlichkeit ↑, separation_fidelity ↑ (Bregman 1990 + McDermott 2009: Stimme/Begleittrennung ist Primär-Streamtask bei Vokalmusik) |
| 5c | Low BW × Dark Centroid | BW < 10 kHz AND centroid < 5 Bark | brillanz ↓↓, waerme ↑ (physikalisch unmöglich zu restaurieren) |
| 5d | Coherence × Tonality | coherence > 0.5 AND tonality > 0.4 | tonal_center ↑↑ (definierendes Merkmal des Songs) |
| 5e | Dynamic × Transient | crest > 10 dB AND density > 5/s | groove ↑, micro_dynamics ↑ (perkussiv UND dynamisch) |
| 5f | Multi-Gen Chain × Low SNR | generations ≥ 3 AND snr < 20 dB | natuerlichkeit ↑, authentizitaet ↑ (Maximum erhalten) |

**Stufe 6 — Soft-Cap (rationale Kompression) + Bounds:**

```
Formel:  w > 1.5 → w' = 1.5 + excess/(1 + 3·excess)    // Asymptote: 1.83
         w < 0.5 → w' = 0.5 - deficit/(1 + 3·deficit)   // Asymptote: 0.17
```

- **k = 3.0**: Erhält relatives Ranking, verhindert dass extreme Gewichte PMGG/CIG-Restauration blockieren
- **Danach**: P1/P2-Floor ≥ 0.70, Hard-Bounds [0.30, 2.00] per `np.clip`

### §2.56b Weight-Semantik

- `weight = 1.0` → Neutral (Standard-Verhalten)
- `weight > 1.0` → Goal ist wichtiger: strengere Regressionsprüfung, weniger tolerante Drift
- `weight < 1.0` → Goal ist weniger relevant: tolerantere Regression, mehr Spielraum

### §2.56c Integration in alle Entscheidungspunkte

| Komponente | Gewichtungs-Effekt |
| --- | --- |
| **PMGG** `_max_regression()` | `weighted_reg = raw_reg × weight` → wichtige Goals triggern Retries leichter |
| **PMGG** `_max_regression_priority_aware()` | Gewichtete Regression wird gegen Priority-Threshold geprüft |
| **CIG** Drift-Toleranz | Negative Drift wird per Goal gewichtet: `drift × weight` |
| **GoalPriorityProtocol** `resolve_conflict()` | Bei gleicher Priority entscheidet höheres Gewicht |
| **GoalPriorityProtocol** `should_abort_iteration()` | `effective_epsilon = base_epsilon / weight` |
| **FeedbackChain** | Gewichtete Abort-Prüfung via GPP |
| **RestorationResult.metadata** | `goal_importance` Block mit Weights + Profil-Info |

### §2.56d Invarianten

- §2.56 ist **Steuerungsmechanismus**, nicht Notbremse. Die Gewichte bestimmen die Pareto-Optimierungsrichtung.
- P1/P2-Goals (Natürlichkeit, Authentizität, Tonal, Timbre, Artikulation) haben Mindestgewicht 0.70 (§0 Primum non nocere).
- Das Gewichtungsprofil wird **EINMALIG** nach allen Klassifizierern und Audio-Feature-Extraktion berechnet und gilt für die gesamte Pipeline.
- Fehlende Genre-/Era-/Material-Information → Uniform-Fallback (alle Gewichte = 1.0).
- Fehler in der Gewichtungsberechnung darf die Pipeline nicht blockieren (graceful fallback).
- Alle Audio-Features sind **optional** (None → Schritt wird übersprungen). Rein label-basierte Berechnung bleibt valide.
- Genre-Alias-Resolution: `"soul"` → `"soul/r&b"`, `"deutscher schlager"` → `"schlager"` etc.

### [RELEASE_MUST] §2.56e Messverfahren-Konventionen (normativ)

| Feature | Korrekt | VERBOTEN |
| --- | --- | --- |
| HNR | Multi-Frame Pitch-Period Autocorrelation, Median über Frames | Lag-1 SNR-Proxy (`_compute_hnr` aus ComprehensiveMetrics) |
| Harmonic Coherence | STFT Spectral-Peak-Consistency (ganzer Song) | 46 ms Single-Frame AC (`_estimate_harmonic_coherence`) |
| Crest Factor | `20 log10(percentile(99.9) / RMS)` | `np.max()` — Impuls-Artefakte dominieren |
| Transient Density | STFT Spectral-Flux + 50 ms Min-Gap | RMS-Flux mit 1.5σ-Threshold (inflationiert Count) |

**Implementierung:**

- Core: `backend/core/song_goal_importance.py` — `SongGoalImportance`, `estimate_goal_importance()`
- Feature-Extraktion: `unified_restorer_v3.py` — SGI-Block in `restore()` nach SongCalibration
- Durchreichung: `phase_kwargs["goal_importance"]` → PMGG `goal_weights` Parameter

---

## §1.3 Metric Recalibration Details & Debugging Patterns (konsolidiert aus Skill fix-metric)

### §9.7.15 Recalibration-Werte (v10.0.0)

| Metrik | Änderung | Wissenschaftliche Basis |
| --- | --- | --- |
| **Brillanz** | HF Crest-Divisor 13.5 → **10.5** | Fastl & Zwicker 2007 §8.3 |
| **Transparenz** | 5-Band-Crest-Divisor 8.8 → **7.0** | Moore & Glasberg 1983 |
| **Wärme** | H2/H4 Even-Harmonic-Divisor 9.0 → **5.0** | Fletcher & Rossing |
| **Natürlichkeit** | Flatness ×2→×2.5, ZCR-Var ×100→×60, Contrast ÷30→÷25, Onset ÷10→÷8 | Johnston 1988 |
| **Emotionalität** | LUFS Pre-Norm auf −14 LUFS | Loudness-invariant |
| **PQS NSIM** | Pearson → ERB-gewichtete Korrelation (300–4000 Hz) | Patterson et al. 1992 |
| **PQS MCD** | Pseudo-RMS → echte Mel-Cepstral Distortion (13 MFCCs) | Kubichek 1993 |

### SNR-robuste PMGG-Proxy-Fixes (§9.7.11–14)

- **§9.7.11 tonal_center**: Krumhansl-Schmuckler-Key-Detection — SNR-invariant. Alle früheren Exclusions entfernt.
- **§9.7.12 brillanz**: HF Spectral Crest Factor (2–16 kHz), p95/p50. Noise hebt p50-Median → Crest steigt nach Denoise. `BrillanzMetric`-Preservation-Penalty entfernt (kontraproduktiv).
- **§9.7.13 transparenz**: Multi-Band Spectral Crest (5 Oktavbänder 250 Hz–8 kHz). Bug-fix: `TransparenzMetric.measure()` hatte kein `reference=`-Parameter → TypeError in Precise-Override.
- **§9.7.14 waerme**: **Primär** Even-Harmonic-Ratio `THD_even / THD_total` (H2/H4), ISO 226:2003 gewichtet. **Sekundär** Sub-Band-Verhältnis E(200–800)/E(800–3000) als Tilt-Proxy. Begründung: Parametrischer EQ-Boost bei 400 Hz erhöht Sub-Band-Verhältnis ohne wahrgenommene Wärme; gerade Harmonische (H2, H4) sind Fingerabdruck von Röhren-/Bandsignalketten (Fletcher & Rossing).

  **PMGG-Proxy-Kalibrierung (RELEASE_MUST)**: Der Quick-Proxy in `per_phase_musical_goals_gate.py`
  berechnet `E(200–800 Hz) / E(800–3000 Hz)` als ungewichtetes FFT-Energieverhältnis.
  `WaermeMetric._measure_absolute()` nutzt ISO 226:2003-Gewichtung, die 800–3000 Hz stärker
  gewichtet → reale Vollmetrik-Scores 0.70–0.90. Typisches warmes Musik-Ratio (ungewichtet): 3–5.
  **Normierungskonstante muss so gewählt werden, dass Ratio ≈ 4.0 → Proxy ≈ 1.0** (derzeit `/4.0`).
  Eine Konstante `/ 1.5` lässt den Proxy permanent bei 1.0 sättigen → PMGG blind für
  waerme-Degradation durch subtraktive Phasen (bestätigt: before=1.0000/after=1.0000 über alle
  Phasen, 2026-04-24). Kalibrierungsprüfung: `assert 0.70 ≤ proxy(warm_signal) ≤ 1.0`.

### §9.7.16 [TARGET_2026] NatuerlichkeitMetric-Reform

**Problem**: Aktuelle Sub-Metriken (Spectral Flatness, ZCR-Varianz, Spectral Contrast, Onset-Dichte) sind Signal-Statistiken, keine Wahrnehmungs-Features. Synthesizer = hoher Score, Jazz-Bar-Aufnahme = niedriger Score → Inversion.

**Reformulierung modus-differenziert**:

| Aspekt | Restoration | Studio 2026 |
| --- | --- | --- |
| **Definition** | "Klingt wie die echte Aufnahme" | "Klingt wie echtes Instrument im Studio" |
| **Primär-Proxy** | MERT-Embedding-Distanz zum Input | MERT-Embedding-Distanz zu Studio-Referenzen |
| **Harmonic Coherence** | Obertonstruktur inkl. Ära-typischer Verzerrungen | Saubere Obertonstruktur ohne Artefakte |
| **Micro-Variation** | Originale Frame-zu-Frame-Variation bewahren | Musikalisch sinnvolle Variation |

**Bis Reform**: `NatuerlichkeitMetric` nur im Export-Gate, nie in PMGG-Delta-Checks. HPI übernimmt holistische Bewertung.

### Root-Cause NatuerlichkeitMetric-Bug

CREPE ändert `w_crepe` 0.0→0.18 zwischen before/after → Pseudo-Regression Δ ≈ 0.15–0.28 → false P1-Kaskade → Phase_03 best-effort @ 5.6 % → Noise-Floor −55 dBFS → Tiefen-Immersion zerstört.

### [RELEASE_MUST] quality_estimate-Formel (einzige erlaubte)

```python
quality_estimate = max(0.0, min(1.0, 0.40 * (1 - defect_severity) + 0.60 * (pqs_mos - 1) / 4))
```

**VERBOTEN**: `quality_estimate * 1.15`

---

## §1.4 Kritische Kalibrierungsfehler (v10.0.0, normiert 2026-04-25)

Diese Fehler wurden in Produktion bestätigt und haben die gesamte Musical-Goals-Kette desaktiviert.

### §1.4.1 TonalCenterMetric — KEY_SHIFT_PENALTY_DEFAULT (RELEASE_MUST)

**Problem**: `_KEY_SHIFT_PENALTY_DEFAULT = 0.0` → Pitch-Shift-Distanzen > 3 Halbtöne liefern Penalty 0.0 → `tonal_center = 0.000` nach `phase_23_spectral_repair` / `phase_06_frequency_restoration`.

**Ursache**: Spektrale Energie-Umverteilung durch Spectral Repair / BW-Extension verschiebt dominante Tonhöhe ≥ 2 Halbtöne **ohne echten Tonartwechsel**. Die Korrelations-Berechnung wertet dies als Schlüsselwechsel.

**Normative Konfiguration**:

```python
_KEY_SHIFT_PENALTY: dict[int, float] = {0: 1.0, 1: 0.75, 2: 0.50, 3: 0.30}
_KEY_SHIFT_PENALTY_DEFAULT: float = 0.20   # War: 0.0 → tonal_center = 0.000
# Bypass-Guard: if corr_score >= 0.60 (Korrektionspfad: 0.85 → 0.70 → 0.60)
# Nach IMCRA/DeepFilter-Denoising fällt Pearson-Chroma auf ~0.655 ohne echten
# Tonartwechsel — 0.70 war zu restriktiv (bestätigt Produktion 2026-04-26, vinyl 225s)
```

**Auswirkung**: Mit `0.0` wurden alle PMGG-Retries für `phase_23` + `phase_06` ausgelöst → Pipeline-Rollback → BW-Restoration und Spectral Repair blieben inaktiv.

### §1.4.2 AuthentizitaetMetric — Formant-Threshold (RELEASE_MUST)

**Problem**: `_formant_threshold = max(500.0, mean_ref_centroid * 0.5)` → bei mittlerem Centroid 1200 Hz: threshold = 600 Hz. BW-Extension hebt Centroid um +600–800 Hz → `formant_stability = max(0, 1 − 1000/600) = 0.0` → `authentizitaet = 0.0`.

**Ursache**: Der Centroid-Shift durch BW-Extension (phase_06/07) ist eine **korrekte Carrier-Inversion** (§2.46), kein Authentizitätsverlust. Die Formel bestrafte die Restaurierung als Fehler.

**Normative Konfiguration**:

```python
_formant_threshold = max(1200.0, mean_ref_centroid * 1.5)
# Erlaubt ±150 % Centroid-Drift als korrekte Träger-Inversion
# War: max(500.0, ref * 0.5) → formant_stability = 0.0 bei jeder BW-Extension
```

### §1.4.3 Phase_18 Noise Gate — PMGG/CIG artikulation-Exclusion (RELEASE_MUST)

**Problem**: `phase_18_noise_gate` hatte keine Goal-Exclusion für `artikulation` + `groove`. Das Noise-Gate schneidet Note-Attacks (Attack-Transienten) → `artikulation`-Proxy sinkt um 0.2901 → katastrophale PMGG-Regression → Pipeline-Rollback → Rauschen bleibt unrepariert.

**Invariante (§2.55 bidirektionale Sync)**:

```python
# PMGG: backend/core/per_phase_musical_goals_gate.py
PHASE_GOAL_EXCLUSIONS["phase_18_noise_gate"] = {"artikulation", "groove"}
# CIG: backend/core/cumulative_interaction_guard.py
_PHASE_SPECIFIC_DRIFT_EXCLUSIONS["phase_18_noise_gate"] = {"artikulation", "groove"}
# Beide Tabellen MÜSSEN identisch sein — §2.55
```

### §1.4.4 adaptive_thresholds Material-Ceiling (RELEASE_MUST)

**Problem**: PMGG-blended Thresholds können 0.90 für Vinyl-Natürlichkeit ergeben (canonical=0.90, SGT=0.88, blend=0.89). Export-Gate liest `adaptive_goal_thresholds` aus Metadata zuerst → lehnt physikalisch korrekte Vinyl-Restaurierung (score=0.655) bei threshold=0.90 ab, statt Material-Floor 0.72 zu verwenden.

**Fix in UV3**: `_ADAPTIVE_THR_MATERIAL_CEILING` Dict nach PhysicalCeiling-Block:

```python
_ADAPTIVE_THR_MATERIAL_CEILING = {
    "shellac":   {"natuerlichkeit": 0.68, "authentizitaet": 0.65, "tonal_center": 0.70, ...},
    "vinyl":     {"natuerlichkeit": 0.82, "authentizitaet": 0.79, "tonal_center": 0.84, ...},
    "reel_tape": {"natuerlichkeit": 0.82, ...},
    "tape":      {"natuerlichkeit": 0.78, ...},
    "mp3_low":   {"natuerlichkeit": 0.76, ...},
}
# for goal, thr in _effective_goal_thresholds.items():
#     ceiling = _ADAPTIVE_THR_MATERIAL_CEILING.get(material, {}).get(goal)
#     if ceiling and thr > ceiling:
#         _effective_goal_thresholds[goal] = ceiling
```

### §1.4.5 GrooveMetric — Bidirektionaler DTW-Onset-Ratio-Guard (RELEASE_MUST)

**Problem**: `noise_onset_ratio = n_original / n_restored > 2.0` prüft nur die Richtung „Original hat zu viele Defekt-Impulse". Wenn die **Restaurierung selbst** Crackle-Artefakte erzeugt (phase_09 AR-Interpolation, phase_31 Pitch-Boundary, phase_55 Inpainting), gilt `n_restored >> n_original` → Ratio ≤ 2.0 → kein Fallback → DTW-Score = 0.000 → `groove = 0.000` (bestätigt Produktion 2026-04-26, vinyl→tape→mp3_low 225s).

Zusätzlich: `_is_noise_dominated = _gdur_s < 10.0 and _max_onset_density > 6.0` → greift bei gecappten 30-s-Segmenten nie (30 > 10).

**Normative Konfiguration** (in `_measure_with_dtw()`):

```python
# Bidirektionaler Guard: Restaurierung hat zu viele Onsets (Pipeline-Crackle)
_restore_onset_ratio = result.n_onsets_restored / max(result.n_onsets_original, 1)
if _dtw_score < 0.3 and _restore_onset_ratio > 1.5:
    return self.measure(audio, sr, reference=None)  # IOI-Fallback

# Catastrophic-Guard: DTW-Score physikalisch unmöglich
if _dtw_score < 0.05:
    return self.measure(audio, sr, reference=None)  # immer IOI-Fallback

# _is_noise_dominated ohne Längenbeschränkung:
_is_noise_dominated = _max_onset_density > 6.0  # War: _gdur_s < 10.0 and ...
```

**Invariante**: Wenn die restaurierte Seite 1.5× mehr Onsets als das Original hat UND der DTW-Score < 0.3, liegen Pipeline-Artefakte vor — IOI-Fallback (referenzfrei). Gilt für alle Song-Längen.

---

### §1.4.5b GrooveMetric — Onset-Verlust-Guard + Messdeterminismus (v10.0.x)

**Problem (Produktion 2026-08-22)**: Der symmetrische `no_onsets`-Pfad lieferte
`groove_score=1.0`, sobald EINE Seite keine Onsets hatte — ein False-Pass, der
einen Onset-Verlust der Restaurierung als „perfekter Groove" durchließ
(Log: `onsets_orig=185, onsets_rest=0 → Wert=1.000`, danach echter DTW-Wert 0.000).
Zusätzlich war die Messkette nicht deterministisch: (a) NaN/Inf im Signal
infizierte den Spectral-Flux → 0 Onsets; (b) `detect_onsets` mittelte bei
Channels-first (2, N) über die Zeitachse → 2 Samples statt N; (c) der
`measure_all`-Cache ignorierte die Referenz → referenzfreies IOI-Ergebnis
wurde auch bei Aufrufen MIT Referenz geliefert.

**Normative Regeln**:

1. **Asymmetrischer Onset-Verlust**: Hat eine Seite ≥ 4 Onsets und die andere 0,
   ist das kein „nichts messbar", sondern `groove_score=0.0`,
   `passes_threshold=False`, `method_used="onset_loss_restored|original"`
   (§V6 (VERBOTEN.md): kein Silent-Failure). Nur wenn BEIDE Seiten 0 Onsets
   haben (Stille/Drone), bleibt `no_onsets` → 1.0 neutral.
2. **NaN/Inf-Guard**: `detect_onsets` und `_measure_with_dtw` wenden vor der
   Messung `np.nan_to_num(..., nan=0, posinf=0, neginf=0)` an. Gleicher Input
   ⇒ gleiches Ergebnis (§G5 (copilot-instructions.md)).
3. **Layout-sicherheit**: `detect_onsets` wählt die Mono-Achse explizit —
   (C, N) channels-first → Achse 0, (N, C) samples-first → Achse 1.
4. **Cache-Korrektheit**: `measure_all`-Cache-Schlüssel = (audio-Hash,
   Referenz-Hash, sr, material_type, panns_singing). Ein ohne Referenz
   gemessenes Ergebnis darf nie einen Aufruf MIT Referenz bedienen.

> Implementierung: `dsp/dtw_groove.py`, `backend/core/musical_goals/musical_goals_metrics.py`
> Tests: `tests/unit/test_v9_dsp_pghi_psola_groove.py::TestDtwGrooveOnsetLossGuard`

---

### §1.4.6 [RELEASE_MUST] TransientEnergyMetric — Algorithmus-Spezifikation (v10.0.0)

> **Neue Sub-Spec für das in §1.2 neu eingeführte Goal `Transient-Energie`** (Prio 2).
>
> **Abgrenzung von `artikulation`**: `ArticulationMetric` misst **Form** (Transient-Shape-
> Korrelation) und **Timing** (Attack-Time-Abweichung ≤ 10 ms). `TransientEnergyMetric`
> misst ausschließlich die **Amplitude/Energie** des Transient-Impuls. Beide Metriken können
> unabhängig fehlschlagen: Ein Transient kann zeitlich korrekt liegen (artikulation = 1.0)
> aber signifikant leiser sein (transient_energie < 0.80) — typisch bei NR-Over-Processing
> oder Noise-Gate-Anbiss.

**Implementierung**: `backend/core/musical_goals/transient_energy_metric.py`

**Singleton**: `get_transient_energy_metric()`

```python
def measure_transient_energy(
    audio_input:    np.ndarray,   # Pre-Pipeline-Original (Reference)
    audio_restored: np.ndarray,   # Post-Pipeline-Restauriert
    sr: int,
) -> Dict:
    """
    Misst Onset-Energie-Ratio über alle erkannten Transients.

    Algorithmus:
    1. HPSS (harmonic/percussive separation) auf BEIDEN Signalen
    2. librosa.onset_detect() auf percussive Komponente des Originals
    3. Für jeden Onset-Sample i:
       - E_orig[i]     = RMS(audio_input[i : i + 5ms × sr])
       - E_restored[i] = RMS(audio_restored[i : i + 5ms × sr])
       - ratio[i]      = min(1.0, E_restored[i] / max(E_orig[i], 1e-9))
    4. transient_energy_score = exp(mean(log(ratio[:])))  # geometrisches Mittel
       (Verhindert Über-Gewichtung extremer Onsets)

    Rückgabe:
    {
        "transient_energy_score": float,      # [0, 1] — Gesamt-Score
        "per_onset_ratios":       List[float], # Ratio je Onset
        "onset_positions_samples": List[int],  # Onset-Positionen
        "n_onsets_detected":      int,
        "material_floor":         float,       # materialadaptiver Boden
    }
    """
    ...
```

**Material-adaptive Böden**:

```python
_TRANSIENT_ENERGY_FLOORS = {
    "shellac":    0.72,   # Mechanische Aufnahme hat natürlichen Energie-Rolloff
    "vinyl":      0.78,
    "cd_digital": 0.83,
    "mp3_low":    0.80,   # Pre-Echo raubt Transient-Energie — toleranterer Boden
    "unknown":    0.80,
}
```

**PHASE_GOAL_EXCLUSIONS** (ergänzt §1.4.3):

```python
# phase_18_noise_gate: Schneidet Note-Attacks → transient_energie sinkt strukturell
PHASE_GOAL_EXCLUSIONS["phase_18_noise_gate"].add("transient_energie")
_PHASE_SPECIFIC_DRIFT_EXCLUSIONS["phase_18_noise_gate"].add("transient_energie")

# phase_26_transient_shaper: Formt Transienten absichtlich um → Messung sinnlos
PHASE_GOAL_EXCLUSIONS["phase_26_transient_shaper"] = (
    PHASE_GOAL_EXCLUSIONS.get("phase_26_transient_shaper", set()) | {"transient_energie"}
)
```

**PMGG-Recovery-Trigger**:

```python
# Wenn transient_energy_score < floor UND letzte Phase ist SUBTRACTIVE:
# → per-Onset Wet/Dry-Blend mit stärkstem Rollback für Onset-Regionen
# (kein globales Rollback — nur Onset-Fenster 5 ms wiederherstellen)
if transient_energy_score < material_floor AND last_phase.taxonomy == "SUBTRACTIVE":
    audio = _blend_onset_regions(audio_dry, audio_wet,
                                 onset_positions, sr,
                                 window_ms=5.0)
```

**Goal-Recovery-Mapping** (ergänzt §09.11):

```python
_GOAL_TO_RECOVERY_PHASES_RESTORATION["TransientEnergie"] = [
    "phase_26_transient_shaper",  # Kann Transient-Energie dosiert wiederherstellen
    "phase_08_transient_shaper_hpss",  # HPSS-gestütztes Transient-Enhancement
]
```

> **Kreuzreferenz**: §1.2 (Goal-Tabelle), §1.4.3 (phase_18 Exclusion), §09.11 (Recovery),
> `ArticulationMetric` (Abgrenzung Form vs. Energie)

---

## §2.35d [RELEASE_MUST] VQI Sub-Komponenten-Dekomposition (v10.0.0)

> **Konzeptuelle Kernlücke geschlossen**: VQI war ein Black-Box-Skalar. Für Weltklasse-Vocal-
> Restaurierung muss jeder Aspekt der Vokalqualität einzeln messbar und angreifbar sein —
> sonst kann kein Recovery-Pfad gezielt intervenieren.

Die `compute_vqi()`-Funktion gibt ab v10.0.0 neben `vqi` drei normierte Sub-Scores zurück:

```python
# backend/core/musical_goals/vocal_quality_index.py
result = compute_vqi(audio_orig, audio_restored, sr)
# {
#   "vqi":                     float,   # Haupt-Score (geometrisches Mittel der 3 Sub-Scores)
#   "vibrato_precision":       float,   # Vibrato-Rate-Erhalt + Tiefen-Erhalt
#   "consonant_clarity":       float,   # Konsonant-Energie-Ratio orig/restored
#   "register_transition":     float,   # Passaggio-Naturalness (Brust→Kopf→Falsett)
#   "singer_identity_cosine":  float,   # d-vector Cosine-Similarity
#   "hnr_delta":               float,   # Harmonics-to-Noise-Ratio Differenz (positiv = besser)
# }
```

### §2.35d-i VibratoPrecision — Messung und Schwellwerte

```python
# Messung via CREPE f₀-Spur:
# 1. f₀-Kurve extrahieren (25 ms Hop)
# 2. Vibrato-Segmente erkennen: Modulation 4–7 Hz, Tiefe ≥ 10 Cent
# 3. Rate-Fehler: |rate_original - rate_restored| ≤ 0.3 Hz → Score 1.0
#    Rate-Fehler > 1.0 Hz → Score 0.0 (lineare Interpolation dazwischen)
# 4. Tiefen-Erhalt: |depth_original - depth_restored| / depth_original
#    ≤ 0.15 → Score 1.0; > 0.50 → Score 0.0

VibratoPrecision_floor:
  Restoration: ≥ 0.80   (leichtes Vibrato-Glätting toleriert)
  Studio 2026: ≥ 0.85   (Enhancement darf Vibrato nicht dämpfen)
  Material-adaptiv: Shellac → ≥ 0.72 (Tonträgerbandbreite limitiert)
```

### §2.35d-ii ConsonantClarity — Messung und Schwellwerte

```python
# Messung via LyricsGuidedEnhancement (§2.36) Phonem-Alignment:
# 1. Plosiv/Frikativ-Segmente isolieren ([p,t,k,b,d,g,s,ʃ,f,v,z])
# 2. Energie-Ratio: E_restored(consonant) / E_original(consonant)
#    ≥ 0.85 → Score 1.0; < 0.60 → Score 0.0
# 3. Spectral-Centroid-Korrelation im Konsonant-Segment ≥ 0.85 → Score 1.0

# Fallback wenn LGE nicht verfügbar:
# Onset-Energie (20–250 Hz-gated attack transients) als Konsonant-Proxy

ConsonantClarity_floor:
  Restoration: ≥ 0.82
  Studio 2026: ≥ 0.87
```

### §2.35d-iii RegisterTransition — Messung und Schwellwerte

```python
# Messung:
# 1. Passaggio-Zonen erkennen: detect_vocal_register_temporal() → segments
# 2. In jeder Übergangszone: f₀-Kontinuität prüfen (kein Pitch-Sprung > 50 Cent)
# 3. Formant-F1-Kontinuität: kein F1-Sprung > 100 Hz in Übergangszone
# 4. Energie-Kontinuität: kein RMS-Sprung > 3 dB in 100 ms um Übergang

# Score: Anteil Übergänge ohne messbare Diskontinuität / Gesamtzahl Übergänge
# 1.0 wenn keine Übergänge (Monochromatic Voice) — kein Penalty

RegisterTransition_floor:
  Restoration: ≥ 0.80
  Studio 2026: ≥ 0.85
```

### §2.35d-iv VQI Gesamtformel

```python
vqi = (vibrato_precision ** 0.40) * (consonant_clarity ** 0.35) * (register_transition ** 0.25)
# Gewichtung: Vibrato-Preservierung > Konsonant-Klarheit > Registerübergang
# (empirisch aus Hörertests: Vibrato-Störungen werden als am störendsten empfunden)
```

---

## §2.35e [RELEASE_MUST] Dynamikbogen-Schutz (EmotionalArc) als explizites Qualitätsziel (v10.0.0)

> **Konzeptuelle Kernlücke geschlossen**: Der Dynamikbogen war nur Gewichtungsfaktor in der
> HPI-Formel (§2.44), aber kein messbares Ziel mit Schwellwert und Recovery-Pfad.
> Weltklasse bedeutet: Crescendo kommt dramatisch an, Klimax kippt nicht durch NR weg,
> Pianissimo-Passagen bleiben zart — auch nach Restaurierung.

### §2.35e-i Dynamikbogen-Messung

```python
# Implementiert in: backend/core/musical_goals/emotional_arc_metric.py
# Singleton: get_emotional_arc_metric()

def measure_emotional_arc_preservation(
    audio_orig: np.ndarray,
    audio_restored: np.ndarray,
    sr: int,
) -> Dict:
    """
    Misst Übereinstimmung der Lautstärke-Kurve (momentane LUFS, 400 ms-Fenster)
    zwischen Original und Restaurierung.

    Methode:
    1. Loudness-Kurve extrahieren: L(t) = momentane LUFS (BS.1770-5, 400 ms)
    2. Normieren: L_norm(t) = (L(t) - min(L)) / (max(L) - min(L))
    3. Pearson-Korrelation r zwischen L_orig_norm und L_restored_norm
    4. Arc-Preservation-Score = (r + 1.0) / 2.0  → [0, 1]

    Gesonderte Prüfung:
    5. Klimax-Schutz: Identifiziere Top-5% Lautstärke-Segmente (Klimax)
       Klimax-Score: Energie in Klimax-Segmenten muss ≥ 0.90 × Original
    6. Pianissimo-Schutz: Identifiziere Bottom-10% Segmente (Pianissimo)
       Pianissimo-Score: Energie in Pianissimo-Segmenten ≤ 1.10 × Original
       (kein Gain-Creep in leisen Passagen)

    Returns: {
      "arc_preservation": float,
      "klimax_preservation": float,
      "pianissimo_preservation": float,
      "emotional_arc_score": float,  # Gesamtscore: min(arc, klimax, pianissimo)
    }
    """
```

### §2.35e-ii Schwellwerte

| Dimension | Restoration | Studio 2026 |
| --- | --- | --- |
| `arc_preservation` (Gesamtkurve) | ≥ 0.88 | ≥ 0.90 |
| `klimax_preservation` (Top-5%) | ≥ 0.92 | ≥ 0.93 |
| `pianissimo_preservation` (Bottom-10%) | ≥ 0.85 | ≥ 0.88 |
| `emotional_arc_score` (Gesamtziel) | ≥ 0.85 | ≥ 0.88 |

**Recovery-Trigger**: `emotional_arc_score < 0.80` → `_recovery_cascade()` für die Phase
mit dem größten negativen Beitrag zur Dynamikbogen-Änderung (via `phase_deltas`).

### §2.35e-iii Integration in §2.44 HPI

Der bisherige `emotional_arc_preservation`-Term in §2.44 HPI wird ersetzt durch
`emotional_arc_metric.emotional_arc_score` aus dieser Spezifikation — nicht mehr
als anonymer Faktor, sondern als gemessener und loggbarer Score.

```python
# Pflicht in UV3._holisticperceptualgate():
_arc_result = get_emotional_arc_metric().measure_emotional_arc_preservation(
    _pipeline_original_reference, _best_sig, sr
)
metadata["emotional_arc"] = _arc_result
# Veto nur bei artifact_freedom (§0h). Arc-Score beeinflusst HPI-Wert.
```

### §v10.101.3 Implementierte Goal-Verbesserungen

| Goal | v10.101-Änderung | Wirkung |
|------|-----------------|---------|
| groove | DTW-Radius von 7680→5 (Sakoe-Chiba) | Score 0.000→1.000 |
| waerme | STFT-Cache shared mit Bass/Brillianz | −1.5s Latenz |
| natuerlichkeit | QualityAnalyzer-Gewicht 10%→20% | Score steigt |
| transparenz | Bark-basierte Maskierung im Blend | Nur hörbare Änderungen |
| emotionalitaet | HPI in Final-Summary sichtbar | Nutzer-Transparenz |
| brillanz | Artifact-Schwelle 0.95→0.90 | Weniger False-Positives |
| transient_energie | Crest-Faktor-Guard | Kein Gain-Staging-Fehlalarm |
| spatial_depth | IACC-basierte Warnung | Mono-Kompatibilität |
