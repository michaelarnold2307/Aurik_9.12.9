# Aurik 10 — Spec 07: Qualitätsziele & Tests

> PQS-Metriken, AMRB-Benchmark, universelle Garantien, Test-Standards,
> E2E-Assertions, Performance-Budget.
>
> **Normative Grundlage**: Alle Schwellwerte operationalisieren §0 — sie definieren
> die untere Grenze dessen, was ein hybrider Toningenieur mit eingebettetem musikalischem
> Urteilsvermögen bei jeder Importdatei erreichen muss. Metriken unterhalb dieser
> Grenzen sind kein „ausreichendes Ergebnis" — sie zeigen an, dass das maximal
> mögliche Ergebnis noch nicht erreicht wurde.

---

## §v10 Pleasantness-First (2026-07-05)

> **HPE ist oberste Instanz.** PMGG darf Phasen ueberspringen, wenn sie
> den Klang fuer menschliche Ohren verschlechtern (§2.29 v10).
> Siehe backend/core/per_phase_musical_goals_gate.py,
> backend/core/human_pleasantness_estimator.py.
> **Kein Rollback-Verbot mehr.** CausalDefectReasoner kann irren — das Ohr nicht.

---

## §8.1 Numerische Qualitätsgrenzen

### PQS-Metriken (`backend/core/holistic_perceptual_gate.py` (v10.0.0-Phantom))

| Metrik | Hard-Fail-Minimum | Internes Spitzenziel |
| --- | --- | --- |
| PQS MOS | ≥ 3.8 | ≥ 4.5 |
| PQS NSIM | ≥ 0.70 | ≥ 0.90 |
| MCD (dB) | ≤ 8.0 | ≤ 3.0 |
| Spectral Coherence | ≥ 0.60 | ≥ 0.85 |

### quality_estimate — normative Formel

```python
quality_estimate = 0.40 * (1 - defect_severity) + 0.60 * (pqs_mos - 1) / 4
# Clip: max(0.0, min(1.0, quality_estimate))
# VERBOTEN: quality_estimate * 1.15 als fixer Bonus
# E2E-Pflicht: result.quality_estimate >= 0.55 nach erfolgreicher Restaurierung
```

### Verbotene Metriken für Musikqualitätsbewertung

```text
PESQ     # Telefonband 300–3400 Hz — strukturell ungeeignet für Vollband-Musik
DNSMOS   # 16 kHz DNS-Challenge-Sprachkorpus
NISQA    # Sprach-CNN, keine Musik-Trainingsdaten
STOI     # Sprachverständlichkeit 150–5000 Hz
ViSQOL --speech (Default) # Musikspektren systematisch falsch bewertet
```

Erlaubte Musikmetriken: **PEAQ, FAD, PQS-MOS, ViSQOL v3 (`--audio` Mode), Musical Goals**

> **CDPAM** ist als Musik-Metrik verboten (Sprachkorpus-Training, kein Vollband-Musik-Bezug).
> Ersatz: VERSA ONNX (`versa_plugin`) für blinde MOS-Bewertung. Aurik restauriert Musik und Gesang — keine Sprachmetriken.

---

## §8.1.1 OQS — Objektiver Qualitäts-Score

> ⚠ **Wichtig**: `backend/core/mushra_evaluator.py` (v10.0.0-Phantom) ist eine algorithmische Approximation
> (PEAQ-ähnlich). Es ist **kein** ITU-R BS.1534-3-konformer MUSHRA-Hörertest.
> In externen Berichten: „OQS (algorithmisch)" — niemals „MUSHRA-Score".
> Die 15 Musical-Goal-Schwellwerte sind aus AMRB-Daten hergeleitet („best engineering estimate“).
> Externe Validierung durch subjektiven Hörertest (ITU-R BS.1534-3) steht aus.
> Änderungen an Schwellwerten erfordern dokumentierten Hörertest als Präzedenz.

**Evidenzhierarchie**:

1. Interne Proxy-Metriken: OQS, PQS-MOS, Musical Goals, HPI
2. Interne Real-Audio-Gates und UAT-Matrizen
3. Reproduzierbare Competitive-Benchmarks gegen Referenzsysteme
4. Externe verblindete Hörtests

**Invariante**: Level 1 bis 3 dürfen harte Release-Entscheidungen steuern. Öffentliche Superlative,
„transparent wie das Original“ oder formale Hörtest-Äquivalenz sind erst mit Level 4 belastbar.

| OQS-Stufe | Score | Pflicht |
| --- | --- | --- |
| Excellent (A) | ≥ 91 | Exzellenz-Label — kein harter Gate-Wert |
| Good (B) | ≥ 80 | **[RELEASE_MUST] Pflicht für jede neue Phase / Plugin** |
| Fair (C) | ≥ 60 | — |

### §8.1.1a [RELEASE_MUST] Studio-2026 OQS-Gate (v10.0.0)

Für `mode="studio2026"` gilt ein dediziertes End-Gate:

- OQS ≥ **88** ist verpflichtend (vormals TARGET_2026).
- Bei OQS < 88 darf kein finaler Studio-2026-Export freigegeben werden.
- Fallback-Verhalten: kontrollierter Rollback auf bestes artefaktfreies Checkpoint-Audio
    oder Modus-Rückfall auf konservative Qualitätskette mit dokumentiertem `fail_reason`.

**Rationale:** Studio-2026 ist ein Qualitätsversprechen („modern, frisch, kräftig").
Ein optionales Roadmap-Ziel ist dafür normativ zu weich.

### §8.1.1b [RELEASE_MUST] Restoration OQS-Gate (v10.0.0)

Für `mode="restoration"` gilt ein materialadaptives End-Gate:

| Material-Klasse | OQS-Minimum | Begründung |
| --- | --- | --- |
| Digital (cd_digital, dat, streaming, aac) | ≥ **80** | Geringe Degradierung → hohe Erwartung |
| Analog modern (vinyl, tape, reel_tape, cassette, minidisc) | ≥ **72** | Moderate Degradierung |
| Analog historisch (shellac, wax_cylinder, wire_recording) | ≥ **60** | Historisches Material hat physikalische Grenzen |
| Lossy Codec (mp3_low, mp3_high) | ≥ **75** | Codec-Artefakte reparierbar |
| Unknown | ≥ **70** | Konservativer Fallback |

**Recovery-Verhalten**: Bei OQS < Minimum → §8.2b Recovery-Kaskade (kein Hardstop). Export mit Status `degraded` + `fail_reason`.

**Invariante**: Restoration-Modus darf Audio nie verschlechtern — wenn OQS(output) < OQS(input), MUSS Rollback auf Input erfolgen.

### §8.1.1c [RELEASE_MUST] MUSHRA-Referenz-Wahl bei CCR Reference-Shift (v10.0.0)

Wenn `§0d carrier_chain_recovery_ratio > 0.05` aktiv ist, wurde `original_audio_for_goals`
auf `best_carrier_checkpoint` (z. B. post-phase_23) verschoben. MUSHRA misst dann die Ähnlichkeit
des **finalen** Audios zum Carrier-Checkpoint — nicht zum degradierten Input. Enhancement-Phasen
(FeedbackChain, ExzellenzDenker) entfernen sich intentional vom Checkpoint → MUSHRA bestraft
diese korrekte Verbesserung als Degradation → OQS < anchor (bestätigt: OQS=47.9 < anchor=57.6,
2026-04-24).

**Normative Invariante**: MUSHRA misst **immer** die Qualität des restaurierten Audios relativ
zum **degradierten Input**, nie relativ zu einem Pipeline-Zwischenstand.

**Implementierung in UV3** (`_holistic_perceptual_gate`):

```python
# CCR-Referenz-Fix: Bei aktivem CCR-Shift → degradiertes audio als MUSHRA-Referenz.
_mushra_ref_src = audio if (original_audio_for_goals is not audio) else original_audio_for_goals
```

Der LUFS-Unterschied zwischen `audio` und `restored_audio` wird durch `lufs_score` ([0, 1])
in der MushraEvaluator-Gewichtungsmatrix bereits abgebildet und ist kein Ausschlusskriterium.

---

## §8.1.2 AMRB v1.0 — Aurik Musical Restoration Benchmark

> **Methodischer Hinweis**: AMRB verwendet **synthetisch degradierte Referenzaudio** (saubere Studioproduktionen
> mit kontrollierten Defekten). Die Pflicht-Scores sind auf synthetischem Material erreichbar. Sie unterscheiden
> sich bewusst von den Produktions-Gates (§8.1.1b), die für **echtes historisches Material** gelten — dieses
> hat physikalische Grenzen, die synthetische Degradation nicht vollständig modelliert. AMRB-03-SHELLAC OQS ≥ 60
> ist der Produktions-Gate für echte 78-rpm-Aufnahmen; OQS ≥ 80 ist das AMRB-Ziel auf synthetisch degradiertem
> Material mit klar definiertem SNR-Ausgangspunkt.

| Szenario | Defekt | AMRB-Pflicht-Score | Produktions-Gate (§8.1.1b) |
| --- | --- | --- | --- |
| AMRB-01-TAPE | Tape-Hiss + Dropout | OQS ≥ 80 | OQS ≥ 72 (analog modern) |
| AMRB-02-VINYL | Vinyl-Crackle + Rumble | OQS ≥ 80 | OQS ≥ 72 (analog modern) |
| AMRB-03-SHELLAC | Shellac-Breitrauschen | OQS ≥ 60 | OQS ≥ 60 (analog historisch) |
| AMRB-04-DIGITAL | Clipping + Quantisierung | OQS ≥ 80 | OQS ≥ 80 (digital) |
| AMRB-05-CODEC | Codec-Artefakte | OQS ≥ 80 | OQS ≥ 75 (lossy codec) |
| AMRB-06-VOCAL | Stimmrauschen + Pitch-Drift | OQS ≥ 80 | OQS ≥ 72 |
| AMRB-07-REVERB | Raumhall RT60=1.2s | OQS ≥ 80 | OQS ≥ 72 |
| AMRB-08-HUM | 50-Hz-Brumm + Obertöne | OQS ≥ 80 | OQS ≥ 72 |
| AMRB-09-DROPOUT | Tape-Dropout 50–200 ms | OQS ≥ 80 | OQS ≥ 72 |
| AMRB-10-COMPOSITE | Kombinierte Degradierung | OQS ≥ 80 | OQS ≥ 70 |
| AMRB-11-CASSETTE | IEC 60094-1 Typ I: BW ≤ 12 kHz + HF-Hiss + Flutter 0.15 % WRMS | OQS ≥ 72 | OQS ≥ 65 (analog Kassette) |

**[RELEASE_MUST] Fragment-Mindestlänge**: Jedes AMRB-Stimulusfragment MUSS **≥ 30 s** lang sein.
Fragmente < 30 s erzeugen OQS-Varianz von ±8 Punkten — ausreichend um einen 80-Punkt-Pass-Fail-Schwellwert
unzuverlässig zu machen. `run_amrb_baseline.py` erzwingt diesen Guard automatisch (`_MIN_AMRB_FRAGMENT_S = 30.0`)
und korrigiert kürzere `--duration`-Angaben mit einem Warn-Log. `n_items ≥ 5` bleibt Pflicht (Nightly-Config).

**Interne Führungs-Schwelle**: Gesamt-Score ≥ **84.0** UND ≥ 8/11 Szenarien bestanden.

```python
from benchmarks.musical_restoration_benchmark import run_benchmark, BenchmarkConfig
report = run_benchmark(config)
assert report.passes_os_leadership_threshold(), f"Score: {report.overall_score}"
```

### [RELEASE_MUST] AMRB Seeding-Invariante (v10.0.0 — deterministisch, MD5)

`item_seed` für AMRB-Items MUSS via `_sid_offset(sid)` berechnet werden — **niemals** `hash(sid)`.
Python's eingebautes `hash()` ist pro-Prozess randomisiert (PYTHONHASHSEED) → nicht reproduzierbar.

```python
import hashlib

def _sid_offset(sid: str) -> int:
    """Deterministisches Item-Seeding für AMRB via MD5.
    RELEASE_MUST: Kein hash(sid) — Python hash() ist PYTHONHASHSEED-abhängig.
    """
    return int(hashlib.md5(sid.encode()).hexdigest(), 16) % (2**31)

# Verwendung in MusicalRestorationBenchmark.run():
item_seed = _sid_offset(scenario_id)  # RICHTIG
# item_seed = hash(scenario_id)       # VERBOTEN — nicht reproduzierbar
```

**Nightly-Konfiguration**: `n_items ≥ 5`; Baseline-Key: `"iZotope RX 11 (commercial)"` (OQS 71.0).

---

## §8.2 Universelle Garantien

| Garantie | Messung |
| --- | --- |
| Kein NaN/Inf im Audio-Ausgang | `np.isfinite(audio).all()` |
| Kein Clipping | `np.max(np.abs(audio)) ≤ 1.0` |
| Chroma-Korrelation (Tonart) | Pearson ≥ 0.95 |
| **Pass-Through (sauberes Material)** | PQS-MOS-Verlust ≤ 0.05, Goals stabil ± 0.02 |
| **Rauschboden (Studio-2026)** | ≤ −72 dBFS, A-gew. ≤ −75 dB(A), 0 Musical-Noise-Events |
| **Rauschboden (Restoration)** | Export-Ziel für alle analogen Tonträger: CD-ähnlicher Rauschboden statt analogem Trägerboden. `shellac`, `wax_cylinder`, `lacquer_disc`, `wire_recording`, `vinyl`, `tape`, `reel_tape`, `cassette` dürfen keinen analogen Mindestboden reinjizieren; bei nötiger Resttextur-Auffüllung Ziel `cd_digital`, ca. −74 dBFS und Testanker ≤ −68 dBFS. `minidisc`, `cd_digital`, `dat`, `mp3_*` bleiben ohne analoge Floor-Injektion. |
| **Rauschtextur-Kohärenz (Restoration)** | `noise_texture_coherence ≥ 0.80` (§4.7) — analoge Trägerdefekt-Textur darf im Export nicht zurückkehren; Ziel ist CD-ähnliche Resttextur ohne Musical-Noise. |
| **Temporale Kohärenz** | MOS-Spanne über 10-s-Segmente ≤ 0.30, σ ≤ 0.15 |
| **Stereo-Authentizität** | Mono-Ären M/S-Korrelation nach Restaur. ≥ 0.97 |
| **HF-Kumulativ-Limit** | Presence + Air kumulativ ≤ +4 dB (Listening-Fatigue) |
| **BW-Material-Ceiling (Restoration)** | Output-BW ≤ `_MATERIAL_BW_CEILING_HZ[material]` (§6.2c) |
| **DR-Material-Ceiling (Restoration)** | Dynamic Range ≤ `_MATERIAL_DR_CEILING_DB[material]` (§6.2b) |
| **Musical-Goal-Erreichung** | Alle 15 anwendbaren Goals erfüllen `final_score ≥ effective_target`; nicht erreichbare Ziele müssen durch `metadata["goal_target_limitations"]` physikalisch begründet und als `degraded`/`recovered_with_limitations` ausgewiesen werden (§1.2b, §09.2b) |
| **Carrier-Recovery-Ratio vorhanden** | `metadata["carrier_chain_recovery_ratio"]` existiert (Pflichtfeld §0d) |
| Mikro-Dynamik-Erhalt | Pearson LUFS-Profil (400 ms) ≥ 0.92, Crest-Faktor ≤ 1.5 dB |
| Tests grün | Alle bestehenden Pytest-IDs (CI: `pytest --collect-only -q \| tail -1`) |

## §8.2a [RELEASE_MUST] Universal-Fidelity-Gate (All Imports)

Maximale Klangtreue ist nur erfüllt, wenn die Qualitätsziele über **alle** Importklassen
robust gelten, nicht nur für Einzelbeispiele.

**Pflicht-Gates:**

1. **Matrix-Gate**: Material × Qualitätsmodus × Kontext (Ära/Genre) muss grün sein.
2. **Shape/SR-Invarianten-Gate**: keine verdeckten Layout-/Samplerate-Verletzungen.
3. **Fallback-Konsistenz-Gate**: OOM/ML-Ausfälle dürfen Qualität degradieren, aber nie
    den Signalvertrag (Länge, Kanäle, Headroom, End-Gates) brechen.

Ein Feature gilt nur dann als produktionsreif, wenn diese Gates ohne song-spezifische
Sonderbehandlung bestehen.

## §8.2b [RELEASE_MUST] Maximal-umsetzbare Recovery bei Gate-Fail (kein Blind-Hardstop)

Ziel ist das bestmögliche reale Ergebnis pro Song, nicht ein formaler Abbruch.
Ein finaler Quality-Gate-Fail triggert deshalb eine verpflichtende Recovery-Kaskade.

**Verbindlich:**

1. Bei `quality_gate.passed=False` MUSS Aurik eine Recovery-Suche durchführen:
    adaptive Strength-Reduktion, Checkpoint-Rollback, alternative sichere Kette,
    material-/restorability-adaptive Parameter.
2. Export ist zulässig, wenn kein besseres sicheres Ergebnis mehr erreichbar ist,
    aber nur mit explizitem Status (`recovered` oder `degraded`) und vollständiger Fail-Ursache.
3. Verboten sind beide Extreme:
    a) blinder Hardstop ohne Recovery-Suche,
    b) stilles „best-effort success" ohne transparente Degradationskennzeichnung.

Diese Regel stellt sicher, dass Aurik immer das maximal umsetzbare Ergebnis liefert,
ohne Qualitätsversagen zu verschleiern.

---

## §8.3 Perceptuelle Verpflichtungen

1. **Natürlichkeit**: MERT-Naturalness-Score ≥ 0.7
   (MERT-harmonicity ist Proxy-Score, kalibriert gegen VERSA-MOS: Pearson = 0.74, n=312;
    bei verfügbarem VERSA-MOS hat dieser Vorrang — MERT nur als Schnellprüfung verwenden)
2. **Harmonische Kohärenz**: Harmonizitäts-Ratio ≥ 0.85 (`MertPlugin.analyze().harmonicity`)
3. **Dynamik-Erhalt**: LUFS-Diff Original ↔ Restauriert ≤ 1 LU
4. **Transientenerhalt**: Attack-Zeiten ≤ 2 ms Änderung
5. **Tonale Stabilität**: Chroma-Korrelation ≥ 0.95
6. **Groove**: Event-Onset-DTW ≤ 8 ms RMS — kein Begradigen von Swing/Rubato
7. **Pass-Through-Invariante** (SNR > 40 dB): PQS-Verlust ≤ 0.05, Goals ≤ ±0.02, LUFS ≤ 0.3 LU
8. **Rauschboden**: Stille-Segmente ≤ −72 dBFS / ≤ −75 dB(A), 0 Musical-Noise-Ereignisse
9. **Mikro-Dynamik**: Pearson LUFS-Profil (400 ms) ≥ 0.92, Crest-Faktor-Abw. ≤ 1.5 dB
10. **Vintage Aesthetics**: Epochen-typische Klang-Charakteristika werden bewahrt (AuthentizitätMetric ≥ vor Restaurierung)

### §8.3.1 Song-Selbstkalibrierung (phasenübergreifend, psychoakustisch priorisiert)

- Für jeden Song MUSS ein Kalibrierungsprofil (`song_calibration_profile`) erzeugt und im Ergebnis abgelegt werden.
- Profile müssen bounded sein (keine ungebremste Verstärkung), um Song-Overfitting zu verhindern.
- Phasenfamilien-Skalierung MUSS P1/P2-Hörtreue priorisieren; P3–P5 dürfen nie zu Lasten von Natürlichkeit/Authentizität erzwungen werden.
- Variationen in Tonträgerketten/Material/Defektbildern müssen zu unterschiedlichen, aber deterministischen Profilen führen.
- Pflichtartefakt für Tests/Analysen: `RestorationResult.metadata["song_calibration"]`.

### §8.3.2 [RELEASE_MUST] Experience-Propagation-Testpflicht (v10.0.0)

Zusätzlich zur Song-Selbstkalibrierung sind folgende Tests verpflichtend:

1. **Metadata-Contract Test**:
    - `RestorationResult.metadata["joy_runtime_index"]` vorhanden und finite.
    - `RestorationResult.metadata["auto_improvement_recommendations"]` schema-stabil.
    - `RestorationResult.metadata["song_calibration"]["cluster_key"]` vorhanden.
2. **Bridge-Contract Test**:
    - `backend.api.bridge.get_experience_insights()` liefert stabile Rückgabe auch bei fehlenden Feldern.
3. **Orchestrator-Propagation Test**:
    - `AurikDenker` propagiert `RestaurierErgebnis.metadata` bis `AurikErgebnis.metadata` unverändert (bis auf defensive Defaults).
4. **PMGG-CIG-Synchronisations-Test (§2.55)**:
    - `tests/unit/test_pmgg_cig_sync.py` muss bidirektional grün sein
      (PMGG→CIG und CIG→PMGG für alle P1/P2-Exclusions).

**Invariante:** Änderungen an Bridge/Denker/UI/Goal-Gates, die Experience-
Telemetrie oder PMGG/CIG-Exclusions betreffen, dürfen ohne diese Testklassen
nicht als release-fähig gelten.

### §8.3.2b [RELEASE_MUST] Canonical Contract Drift Gate — Testpflicht

Jede Änderung an GUI, CLI, Batch, REST-Legacy, Bridge, Import, Denker-Einstieg oder Export MUSS durch einen schnellen Contract-Drift-Gate abgesichert sein. Der Test darf statisch sein, muss aber folgende Klassen blockieren:

- Release-Pfade ohne `get_load_audio_fn()` / `run_pre_analysis()` / `AurikDenker.denke()`.
- Exportpfade ohne `export_guard()` + `validate_export_quality()` + `build_export_quality_gate_payload()` oder ohne `AudioExporter`/atomic-WAV-Fallback.
- Direkte `UnifiedRestorerV3.restore()`-Bypässe in GUI/CLI/Batch-Releasepfaden.
- Nicht markierte REST-/Server-Altpfade mit direktem Audio-Write.
- Neue CLI/GUI-Argumente, die mehr Nutzerentscheidungen als `Restoration` / `Studio 2026` einführen.

Kanonischer Testanker: `tests/normative/test_canonical_contract_drift_gate.py`.

**GUI-Live-Status-Zusatzpflicht:** Aenderungen an Defektchips, Waveform-Markern,
Statusmeldungen oder Hauptfortschrittsmapping MUESSEN zusaetzlich durch
`tests/normative/test_modern_window_gui_contract.py` abgesichert sein. Pflichtfaelle:
Dropout-Aliase (`dropouts`, `DROPOUTS`, `gap`, `gaps`, `tape_dropout`) zaehlen zum
sichtbaren Chip `Tonaussetzer`; lokalisierte Dropout-Events zaehlen waehrend echter
Dropout-/Inpainting-Phasen anhand des Timeline-Cursors herunter; UV3-Post-Processing darf
den Hauptbalken nicht ueber 90 % treiben, bevor Export/Finalisierung explizit gestartet sind.

### §8.3.2a Era-/VFA-/GP-Prior-Regressionspflicht [RELEASE_MUST]

Jede Änderung an Vokal-Gates, VFA-Zonen, GP-Priors oder RecordingChainProfiler-Integration MUSS fokussierte Unit-Tests enthalten:

- `EraVocalProfile`: Mapping der Ären und `resolve_formant_tolerance_db()` inklusive historisch > modern.
- `GPParameterOptimizer`: `chain_hint` skaliert Strength-/Boost-/Ratio-Parameter; `memory_prior` blendet bekannte Parameter bounds-geclampt; `propose_pareto()` akzeptiert beide Priors.
- UV3/VFA: `vocal_zone_strength_policy` wird vor `wrap_phase()` gebildet und an Phasen-kwargs weitergereicht.
- Formant-Gates: neue Vokalphasen dürfen keinen fixen `threshold_db=2.0` verwenden, wenn Era-Kontext verfügbar ist.

### §8.3.3 [RELEASE_MUST] Edge-Peak- und Stereo-Lag-Regressionstestpflicht (v10.0.0)

Änderungen an Boundary-anfälligen Phasen (insb. `phase_03`, `phase_23`, STFT/ISTFT-, Chunk- oder ML-Routen)
müssen zusätzlich folgende Nachweise liefern:

1. **Keine Intro/Outro-Pegelexplosion**:
    - Der restaurierte Output darf in Intro/Outro-Zonen keine neu eingeführten Peak-Explosionen gegenüber dem Input erzeugen.
    - Präventive Ursache-Fixes (Kontext-Padding, deterministischer Strip) sind Pflicht; reiner Post-hoc-Fade reicht nicht als alleiniger Fix.
    - Dasselbe gilt für positive Gain-Pfade ohne Boundary-Änderung: Loudness-, Export- und Mastering-Stufen müssen einen relativen Intro/Outro-Regressionscheck gegen das Eingangs-Audio nachweisen.
2. **Keine neue L/R-Zeitverschiebung**:
    - Interchannel-Delay nach Verarbeitung darf nicht über den Input hinaus regressieren.
    - §2.51a Hard-Fail (> 1 ms) bleibt bindend.
3. **Kanal-Symmetrie bei separater Verarbeitung**:
    - Für getrennte L/R-Pfade müssen Strip-Offset und Strip-Länge identisch validiert werden.
    - Für `phase_23` gilt zusätzlich: Stereo-ML-Pfade müssen M/S- oder Linked-Stereo laufen; separate L/R-ML-Inferenz ist nicht release-fähig.

**Release-Kriterium**: Ohne grüne Nachweise für Edge-Peak- und Lag-Invariante ist ein Merge in release-relevante Branches unzulässig.

**Pflicht für Gain-nahe Regressionstests:** Wenn ein Patch positiven Gain in Loudness-/Export-/Mastering-Pfaden verändert,
reicht ein globaler Peak- oder LUFS-Test nicht aus. Es MUSS zusätzlich mindestens ein Test existieren, der zeigt,
dass der Mittelteil angehoben wird, während Intro/Outro-Peaks relativ zur Referenz innerhalb des Quiet-Edge-Limits bleiben.

**CI-Contract-Test**: `tests/normative/test_edge_lag_no_regress_contract.py` muss grün sein.

**Real-Audio-Gate (heavy)**: `tests/normative/test_real_audio_edge_lag_gate.py` muss bei
`--run-heavy-tests` grün sein (Intro/Outro-Peak-Exzess + Interchannel-Delay-Delta).
Alle Fixture-Payloads in diesem Gate muessen statisch typisiert werden; `dict[str, object]`-
Werte sind vor numerischer Umwandlung per `typing.cast` oder lokaler Typpruefung zu verengen,
damit `call-overload`-Fehler nicht durch schwache Fixture-Typisierung verdeckt werden.

## §8.5 [RELEASE_MUST] Globales Parameterregister

Das Parameterregister dokumentiert normativ die zentralen Runtime-Parameter,
die im Produktionscode statisch verankert sein müssen und durch CI (R01–R07)
automatisch geprüft werden.

### §8.5A OPTIMAL

| Parameter | Zielwert | Ort |
| --- | --- | --- |
| Final TruePeak hard-guard ceiling | 0.966 (-0.3 dBFS) | unified_restorer_v3.py |
| Noise-Texture Threshold shellac | 6.0 dB/oct | _MATERIAL_NOISE_TEXTURE_ROLLBACK_THRESHOLD |
| Noise-Texture Threshold vinyl | 8.0 dB/oct | _MATERIAL_NOISE_TEXTURE_ROLLBACK_THRESHOLD |
| Noise-Texture Threshold mp3_low | 15.0 dB/oct | _MATERIAL_NOISE_TEXTURE_ROLLBACK_THRESHOLD |
| mp3_low priority phases | phase_06, phase_38, phase_39 | _MATERIAL_PRIORITY_PHASES |
| mp3_high priority phases | phase_06, phase_39 | _MATERIAL_PRIORITY_PHASES |
| Stereo-correlation guard | input-relativer delta_limit | unified_restorer_v3.py |

### §8.5B NICHT OPTIMAL

| Muster | Risiko | Status |
| --- | --- | --- |
| Hardcoded Noise-Texture-Grenze 6.0 für alle Materialien | False-Rollbacks oder Blindheit je Material | NICHT OPTIMAL |
| TruePeak-Messung via np.max statt percentile(99.9) | Impulsartefakt dominiert Headroom-Guard | NICHT OPTIMAL |
| Presence/Air-Band nur bei vocals_detected auf Lossy-Material | Codec-Artefakte bleiben unbearbeitet | NICHT OPTIMAL |
| Stereo-Guard ohne input-relative Schwelle | Falsch-positive Rollbacks bei schmalem Stereo | NICHT OPTIMAL |

---

## §9 Performance-Budget (Desktop-Hardware, GPU-Mixed-Mode optional)

> **Kanonische Quelle für RT-Limits:** copilot-instructions.md §2.37 (LIMIT_BALANCED = 32.0× RT).
> Die nachstehenden Werte gelten **pro Minute Audio** und sind mit PerformanceGuard-Toleranzen kalibriert.
> VERBOTEN: niedrigere Limits aus Vorgängerversionen (DefectScanner ≤ 2 s, Pipeline ≤ 120 s) verwenden —
> diese wurden mit v10.0.0 (Quality-First-Hauptlauf) auf die untenstehenden Werte angehoben.
> **GPU-Beschleunigung** (ROCm/DirectML) reduziert Heavy-Plugin-Inferenz erheblich; die Limits gelten für CPU-only als Worst Case.

| Operation | Limit / Minute Audio |
| --- | --- |
| DefectScanner | ≤ **4 s** |
| Phase-Pipeline gesamt | ≤ **240 s** |
| FeedbackChain (alle Iterationen) | ≤ **120 s** |
| ExcellenceOptimizer | ≤ **60 s** |
| RestorabilityEstimator | ≤ **5 s** |
| Export (FLAC 24-bit) | ≤ 10 s |

**Absolutes Zeitlimit Stufe 1:** `_MAX_TOTAL_SECONDS = 14400.0` (240 Minuten, §K 64×RT-aligned)
Nach Überschreitung: KMV Stufe 2 (`MLRefinementThread`) übernimmt automatisch.

### §9.1d Matrix-Harness-CI-Gate (Budget-Enforcement, Bootstrap-CI, Phase-Profiling, Ebene-3-Audit)

`scripts/benchmark_effizienz_matrix.py` stellt das maschinelle Gate gegen die
§9-Tabelle dieser Spec bereit:

- **Budget-Enforcement:** `--enforce-budget` (bzw. `--ci`, das es impliziert)
  prüft die gemessenen Zeiten je Zelle gegen die Tabelle in §9 dieser Spec.
  Verletzungen landen im Ergebnis-JSON unter `budget_violations` und führen im
  CI-Modus zu Exit 1. Operationen, für die die Pipeline kein Per-Operation-Timing
  liefert, werden als `null` dokumentiert (nicht geschätzt).
- **Bootstrap-95 %-CI:** `--bootstrap-ci` berechnet je Zelle Konfidenzintervalle
  der Qualitäts-/MUSHRA-Werte (Percentile-Bootstrap, deterministisch per Seed,
  §G5 (copilot-instructions.md)); Schlüssel `quality_ci95`/`mushra_ci95` plus
  `bootstrap_seed`/`bootstrap_n`/`bootstrap_alpha`.
- **Phase-Profiling:** `--profile-top-phases N` schreibt die N langsamsten
  Phasen je Zelle (Wall-Zeit) als `top_phases` ins Ergebnis-JSON
  (Opt-in; `--ci` setzt 3).
- **Ebene-3-Audit:** `backend/core/wohlklang_ordnung_gate.py`
  (`WohlklangOrdnungGate`) verifiziert die lexikografische Wohlklang-Ordnung
  (§5/§8 (hoerordnung.instructions.md)) maschinell; Ergebnis im
  Metadata-Key `wohlklang_ordnung`, GUI-Ampel via `hearing_gates_summary.py`.

**FeedbackChain-Abbruch (Fix M, v10.0.0 — MOS-Metrik präzisiert, harmonisiert v10.0.0):**

```python
MAX_ITERATIONS = 5
CONVERGENCE_DELTA = 0.02
REGRESSION_DELTA  = 0.05  # PQS-MOS-Schwelle für sofortigen Rollback

# 3-Bedingungen-Kaskade (normativ, Single Source of Truth in diesem §8.5):
#
# Bedingung 1 — Konvergenz (kein Rollback):
#   |mos_iter_n - mos_iter_n_minus_1| < CONVERGENCE_DELTA (0.02)
#   → weitere Iteration würde Qualität nicht mehr messbar verbessern → frühzeitiger Exit
#
# Bedingung 2 — Regression (sofortiger Rollback):
#   |mos_iter_n - mos_iter_n_minus_1| > REGRESSION_DELTA (0.05)
#   → sofortiger Rollback auf best_result (höchster PQS-MOS bisher)
#   → keine weiteren Iterationen
#
# Bedingung 3 — Forced Exit (Anti-Hänger):
#   n >= MAX_ITERATIONS (5)
#   → Pipeline-Timeout-Schutz; best_result exportieren
#   → metadata["feedbackchain_forced_exit"] = True
#
# NICHT: Musical Goals Ø-Score (der wird separat über GoalPriorityProtocol überwacht)
# NICHT: MERT-harmonicity-Proxy (zu niedrige Sampling-Frequenz für Iterations-Vergleich)
# NICHT: Vergleich mit Baseline (nur Iteration-zu-Iteration, um Overshoot zu detektieren)
#
# Spec 02 verweist auf diesen §8.5 als normativ übergeordnet (§2.54 ist Steuerungslogik,
# §8.5 enthält die kanonischen Schwellwerte — kein Widerspruch, unterschiedliche Ebenen).
```

### §9.1a [RELEASE_MUST] Stationäre vs. nicht-stationäre Defekt-Analyse (v10.0.0)

Der DefectScanner verwendet einen 60 s Center-Crop (`_DETECTOR_CAP_S`) für Performance-Optimierung. Nicht-stationäre Defekttypen MÜSSEN jedoch auf dem **vollständigen Audio** analysiert werden:

| Kategorie | Defekttypen | Audio-Scope | Begründung |
| --- | --- | --- | --- |
| **Nicht-stationär** | `DROPOUTS`, `TRANSPORT_BUMP` | Vollständiges Audio | Treten lokal auf (Intro, Outro, Splice-Punkte, Bandanfang) |
| **Stationär** | Alle anderen (Rauschen, Brummen, Flutter, …) | 60 s Center-Crop | Statistisch repräsentativ über kurzen Ausschnitt |

**Invariante**:

```python
_audio_mono_full = audio_mono  # MUSS vor Center-Crop gesichert werden
# ... Center-Crop ...
scores[DefectType.DROPOUTS] = self._detect_dropouts(_audio_mono_full)  # volles Audio
```

- Für synthetische Langsignale mit >50 nicht-stationären Events (z. B. Dropouts/Tape-Head-Level-Dips) MUSS `len(score.locations) > 50` gelten.
- Eine künstliche Kappung der Core-Defektliste auf feste Grenzen (z. B. 50/100/256) ist als Regression zu werten.
- Optional verdichtete Anzeige-Listen müssen in separaten UI-Tests validiert werden und dürfen die Core-Liste nicht beeinflussen.

**Location-Offset**: Nicht-stationäre Detektoren erzeugen absolute Positionen → `_FULL_AUDIO_DETECTORS`-Set in der Offset-Korrektur ausschließen.

### §9.1b [RELEASE_MUST] Intro-Salienz-Gewichtung (v10.0.0)

Die ersten 5 Sekunden eines Audiosignals bestimmen das Gesamtqualitätsurteil des Hörers (Zacharov & Koivuniemi 2001, Bech & Zacharov 2006). Defekte in dieser „Intro-Zone" MÜSSEN stärker gewichtet werden.

**Algorithmus**:

- Für jeden Defekttyp mit Locations: Anteil der Events in den ersten 5 s berechnen
- Severity-Boost: `severity *= 1.0 + 0.5 * intro_fraction` (max. 1.5×, gedeckelt auf 1.0)
- Metadata: `intro_boost_applied`, `intro_boost_factor`, `intro_events`

**Begründung**: Tape-Leader-Artefakte, Einlaufschwankungen und Splice-Dropouts treten gehäuft in den ersten Sekunden auf. Ohne Intro-Gewichtung kann die Pipeline diese als „statistisch irrelevant" einstufen, obwohl sie den ersten Höreindruck zerstören.

### §9.1c [RELEASE_MUST] Perceptual-Salience-Annotation (v10.0.0)

Jeder erkannte Defekt wird mit einem **psychoakustischen Salienz-Score** (0.0–1.0) annotiert. Defekte, die durch lautere Umgebung **maskiert** werden (Fastl & Zwicker 2007 „Psychoacoustics: Facts and Models"), erhalten einen niedrigeren Severity-Wert; exponierte (hörbare) Defekte behalten volle Severity.

**Wissenschaftliche Basis**:

- Simultane Maskierung: Kontext ≥ 12 dB über Defekt-Lautstärke → maskiert
- Temporale Vorwärts-Maskierung: Lautes Signal innerhalb 200 ms vor dem Defekt → teilmaskiert (Schwelle 8 dB)
- Temporale Rückwärts-Maskierung: Lautes Signal innerhalb 20 ms nach dem Defekt → teilmaskiert (Schwelle 6 dB)
- Loudness-Modell: ITU-R BS.1770-5 momentary loudness (400 ms Fenster, 100 ms Hop)

**Modul**: `backend/core/perceptual_salience.py` — `PerceptualSalienceEstimator` (Singleton, §3.2)

**API**: `annotate_defect_scores(audio, sr, defect_result)` → modifizierte `DefectAnalysisResult` mit:

- `metadata["perceptual_salience"]`: Mittelwert der Salienz aller Events (0.0–1.0)
- `metadata["n_salient_events"]`: Events mit Salienz ≥ 0.5
- `metadata["n_masked_events"]`: Events mit Salienz < 0.3

**Severity-Skalierung**: `severity = severity * (0.3 + 0.7 * mean_salience)` — bewahrt 30 % Basis-Severity auch für vollständig maskierte Defekte.

**Integration**: Wird im DefectScanner.scan() nach §9.1b (Intro-Boost) und vor Location-Offset aufgerufen.

**Pipeline-Nutzen**:  

1. Hörbare Defekte werden priorisiert repariert  
2. Unhörbare Defekte werden geschont (weniger Artefaktrisiko durch unnötige Reparatur)  
3. UX-Reporting: „3 hörbare Defekte behoben, 12 unterhalb der Hörschwelle"

**Invarianten**:

- Analyse-Modul → arbeitet bei nativer Import-SR (kein `assert sr == 48000`)
- NaN/Inf-Guard auf allen numerischen Ausgaben
- Thread-safe Singleton (double-checked locking)

> **Hinweis**: Die normativen Performance-Limits sind ausschließlich in §9 (oben) definiert.
> VERBOTEN: Alte Limits (DefectScanner ≤ 2 s, Pipeline ≤ 120 s, FeedbackChain ≤ 60 s) —
> diese wurden mit v10.0.0 (Quality-First-Hauptlauf, Fix I) angehoben und gelten nicht mehr.

### §8.3 [RELEASE_MUST] Psychoakustik & Gänsehaut-Prinzipien (v10.0.0)

**Oberstes Ziel**: Das Restaurierungsergebnis muss beim Hörer **emotionale Wirkung** erzeugen — nicht nur technische Korrektheit. Die Psychoakustik steht über der technischen Perfektion.

**Gänsehaut-Formel**: `(TransientIntegrity × MicroDynamik × Klarheit × Authentizität) − Artefakte`

| Komponente | Verantwortliches Modul | Anteil |
| --- | --- | --- |
| **Transient-Punch** | TDP (Transient Decoupled Processing) | ~40 % |
| **Mikro-Dynamik-Erhalt** | MDEM (400 ms LUFS-Morphing) + EmotionalArcCorrection (5 s Makro-Bogen) | ~25 % |
| **Rauschbefreiung/Klarheit** | SGMSE+ / OMLSA/IMCRA | ~20 % |
| **Vokal-Präsenz** | Phase 42 + Phase 43 + VocalAIEnhancement | ~10 % |
| **Neurale Synthese** | Vocos 48 kHz (nur Studio 2026, MOS < 4.3) | ~5 % |

**Zwei-Skalen-Dynamik-Schutz** (Pflicht — beide Ebenen im UV3 implementiert):

- **Mikro-Ebene (400 ms)**: `MicroDynamicsEnvelopeMorphing.morph()` — LUFS-Profil-Rückgewinnung (§2.30)
- **Makro-Ebene (5 s)**: `correct_emotional_arc()` — post-MDEM Gain-Korrektur bei Bogen-Abflachung.
  Algorithmus: Per-Segment RMS-Gain (orig/rest), ±6 dB max, 70 % Dämpfung, Savitzky-Golay-geglättet,
  Sicherheits-Revert wenn Arousal-Pearson sich verschlechtert.
  Modul: `backend/core/emotional_arc_preservation.py` — `correct_emotional_arc(original, restored, sr)`

**UV3-Pipeline-Integration** (nach MDEM, vor GP-Update):

1. MDEM korrigiert Mikro-Dynamik (400 ms Fenster)
2. EmotionalArc Post-MDEM-Messung: `measure_emotional_arc(original, restored, sr)`
   `arc_preserved = True` genau dann, wenn **beide** Bedingungen erfüllt:
   (a) `arousal_pearson_global ≥ 0.85` (5-s-Fenster, gesamte Datei)
   (b) kein einzelnes 5-s-Segment hat `Δarousal_local < −0.08`
       (`Δarousal_local[i] = rms_restored[i] / rms_original[i] − 1`;
        Normiert auf Original-RMS; Segment ≤ 3 s am Dateiende ignoriert).
   Rationale: Globaler Pearson ≥ 0.85 kann trotz eines eingebrochenen Segments
   (z.B. pp vor Refrain wird durch Kompression flachgelegt) bestehen. Die lokale
   Schwelle −0.08 fängt Spannungspunkte, die Schauer auslösen würden.
3. Falls `not arc_preserved`: `correct_emotional_arc()` mit Makro-Gain-Hüllkurve
4. Safety-Revert: wenn Korrektur Arousal-Pearson um > 0.02 verschlechtert → Revert

**RAM-Budget:**

- Audio-Buffer max.: 4 GB
- ML-Modelle aktiv max.: 16 GB gesamt
- Großmodelle (MERT 3,9 GB / AudioSR 5,9 GB): nur bei Bedarf (lazy load)

**Device-Policy (§GPU-Mixed-Mode, v10.0.0):**

```python
# Heavy ML Plugins (>200 MB): GPU wenn verfügbar, CPU-Fallback transparent
from backend.core.ml_device_manager import get_ort_providers, get_torch_device
providers = get_ort_providers("PluginName")  # ONNX-Runtime
device = get_torch_device("PluginName")      # PyTorch
# Leichtgewichtige Plugins (<200 MB), DSP, Analyse: immer CPU
providers = ["CPUExecutionProvider"]          # ONNX-Runtime
model = model.to("cpu")                       # PyTorch
torch.set_num_threads(os.cpu_count())         # alle CPU-Kerne nutzen
```

---

### §8.3.1 [RELEASE_MUST] Tiefen-Immersions-Prinzip — „Ohr in die Musik legen" (v10.0.0)

**Konzept**: Das Restaurierungsergebnis muss dem Hörer ermöglichen, in die Musik hineinzutauchen —
nicht nur zuzuhören, sondern sich von der Aufführung umgeben zu fühlen. Gänsehaut entsteht,
wenn alle akustischen Tiefenschichten gleichzeitig hörbar sind und die Spannungsdynamik
einer Aufführung authentisch erlebt wird.

#### Akustische Tiefenschichten (von außen nach innen)

| Schicht | Frequenzbereich | Enthält | Technische Bedingung |
| --- | --- | --- | --- |
| **Raumluft / Air** | 8–20 kHz | Saiten-Obertöne, Becken-Shimmer, Gesangs-Luft | Noise Floor < −72 dBFS; Phase_06 SBR; Phase_39 Air |
| **Vokal-Intimität** | 4–8 kHz | Frikative /s/ /f/ /ʃ/, Plosive /p/ /t/, Atem | LyricsGuidedEnhancement: `fricative ×1.55`, `plosive ×1.40` |
| **Instrument-Körper** | 200 Hz–4 kHz | Note-Sustain, Saitenresonanz, Bogen-Kratzen | TDP-Transient-Erhalt + MDEM 400 ms LUFS-Morphing |
| **Fundament** | 20–200 Hz | Kick-Punch, Bassresonanz, Raummode | BassKraftMetric + Virtual-Pitch (Missing Fundamental) |
| **Raumtiefe** | Diffus (MS) | Raumluft, Phantom-Center, Tiefenstaffelung | SpatialDepthMetric IACC ≥ 0.70, M/S-Korr. ≥ 0.97 |

#### Physikalische Kausalkette: PMGG-Stabilität → Tiefen-Immersion

```text
Phase_03_denoise bei GP-optimalem strength (§9.7.7 / §2.29b aktiv)
    → Noise Floor < −72 dBFS
    → Air-Layer (8–20 kHz) frei: Saitenobertöne, Atemgeräusch, Becken-Shimmer hörbar
    → Vokal-Intimität-Layer (4–8 kHz) frei: Frikative, Plosive, Atem wahrnehmbar
    → emotionaler Authentizitäts-Schock: „das klingt wie live im Raum"
    → Gänsehaut

Phase_03_denoise @ best-effort strength=0.056 (false P1-Regression, §9.7.7 FEHLT)
    → Noise Floor −55 dBFS (+17 dB über Ziel)
    → Mikrodetails unter Rauschteppich verdeckt
    → Studio-Distanz-Effekt; der Hörer bleibt „außen"
    → kein emotionaler Sog, keine Gänsehaut
```

**E2E-Pflicht-Assertions** (Tiefen-Immersions-Gate):

```python
assert noise_floor_silence_dbfs(restored, sr)                 <= -72.0  # Air-Layer frei
assert pearson_lufs_profile_400ms(original, restored, sr)     >= 0.92   # MDEM Mikro-Dynamik
assert emotional_arc_arousal_pearson(original, restored, sr)  >= 0.85   # Makro-Bogen
assert spatial_depth_iacc(restored, sr)                       >= 0.70   # Raumtiefe
```

#### Vokal-Intimität — Physik der akustischen Nähe

Der Hörer empfindet eine Stimme als „körperlich nah" durch drei physikalische Phänomene:

1. **Konsonanten-Transient** (Plosive /p/ /t/ /k/, Frikative /s/ /f/ /ʃ/): Der kurze Druckstoß
   simuliert das akustische Nahfeld < 50 cm. Over-Denoising verschleift diese Transienten → Stimme
   verliert physische Glaubwürdigkeit. LyricsGuidedEnhancement schützt diese Segmente mit den
   Boost-Faktoren anstatt sie zu dämpfen.

2. **Atemgeräusche** (150–800 Hz + 2–5 kHz): Das Einatmen zwischen Phrasen, das leichte Räuspern,
   das Öffnen der Lippen — diese unlyrischen Geräusche sind die stärksten Proximitäts-Signale.
   Standard-NR entfernt sie als „Rauschen". Aurik schützt sie in `silence`-Segmenten mit
   reduziertem NR-Boost (`×0.70`).

3. **Early Reflections** (1–30 ms): Die ersten Raum-Echos bestimmen die wahrgenommene
   Quellen-Distanz. Phase_20 (Reverb Reduction) darf nur Spät-Hall (> 80 ms RT60-Anteil)
   entfernen — Early Reflections müssen erhalten bleiben, sonst kollabiert die wahrgenommene
   Quellen-Distanz auf „unendlich weit".

#### [RELEASE_MUST] Spektrale Phasenkohärenz — Tiefenstaffelung durch IPD

Nach **jeder** Spektral-Modifikation: **PGHI-ISTFT** (Phase Gradient Heap Integration, Virtanen 2018).

Das menschliche Gehirn nutzt interaurale Phasendifferenzen (IPD, < 0.7 ms Laufzeitunterschied)
für horizontale Richtungslokalisierung (Blauert 1997, §3.1 Precedence Effect). Griffin-Lim
randomisiert die Phase → alle Instrumente erscheinen in derselben Tiefenebene → räumliche
Staffelung kollabiert → „flaches Stereo-Bild" ohne Tiefe.

Mit PGHI: Phasenlaufzeiten erhalten → Gitarre links-vorne, Klavier rechts-mitte, Gesang mittig-nah.
**VERBOTEN** als Studio-2026-Endschritt: `griffinlim()`.

#### Emotionaler Atemzug — Spannungsdynamik als Gänsehaut-Mechanismus

Der stärkste Gänsehaut-Auslöser ist nicht die lauteste Stelle einer Aufnahme, sondern der
Moment **kurz davor**:

- Ein `pp`, das den Hörer in die Stille zieht, bevor das `fff` trifft
- Eine Fermate, die die Zeit anhält
- Ein Diminuendo, das den nächsten Akkord unausweichlich macht

Diese Spannungsmechanik setzt voraus:

- `pearson_lufs_400ms ≥ 0.92` (MDEM): Mikro-Energiewellen auf Sub-Phrasen-Ebene erhalten
- `arousal_pearson_5s ≥ 0.85` (EmotionalArc): Intensitätsbogen über Phrasen-Ebene erhalten
- Keine künstliche Dynamik-Kompression zwischen piano und forte Stellen

Wenn alle Stellen gleich laut klingen, ist die Spannungs-Mechanik zerstört und Gänsehaut unmöglich.

---

## §8.1.3 [RELEASE_MUST] FAD (Fréchet Audio Distance) — interner Spitzenqualitäts-Indikator (v10.0.0)

**Motivation**: OQS und AMRB messen Ähnlichkeit zum Original. FAD misst **Verteilungsähnlichkeit zum Studio-Audio-Referenzset** — d.h. ob restauriertes Audio klingt wie professionelles Studio-Material, unabhängig vom konkreten Original.

**FAD-Schwellwerte** (ergänzend zu AMRB):

| Material/Modus | FAD-Ziel | FAD-Mindest (CI-Gate) |
| --- | --- | --- |
| Shellac → Restoration | ≤ 18.0 (physikalisches Limit) | ≤ 25.0 |
| Vinyl → Restoration | ≤ 8.0 | ≤ 12.0 |
| Tape → Restoration | ≤ 6.0 | ≤ 10.0 |
| Digital → Restoration | ≤ 3.0 | ≤ 5.0 |
| Studio 2026 (alle) | ≤ 2.0 | ≤ 4.0 |

**FAD < 5.0** ist der allgemeine interne Spitzenqualitäts-Indikator für moderne Studio-Klangqualität.

**Referenzset**: `benchmarks/fad_reference_set/` — 50 professionelle Studioaufnahmen (2010–2023, gemischt Genre/Ära). Referenzset darf nie durch restauriertes Audio befüllt werden.

**Messung**:

```python
# benchmarks/fad_evaluator.py
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance(model_name="vggish", sample_rate=16000)
score = fad.score("benchmarks/fad_reference_set/", "output/test_batch/")
```

**CI-Gate**: `tests/normative/test_fad_gate.py` [ROADMAP] — läuft nur mit `--run-heavy-tests`. Nicht im täglichen CI.

---

## §8.1.4 [RELEASE_MUST] VERSA-Konfidenz-Modell (v10.0.0)

**Problem**: VERSA liefert für stark degradiertes Material (SNR < 10 dB, Shellac) unzuverlässige MOS-Werte, weil es auf modernen Daten trainiert wurde. Ein niedriger VERSA-Score auf Shellac bedeutet vielleicht nur „unbekannter Klangtyp", nicht „schlechte Qualität".

**VERSA-Konfidenz-Formel**:

```python
def compute_versa_confidence(snr_estimate_db: float, material_type: str) -> float:
    """
    Gibt confidence [0, 1] zurück.
    Hohe Konfidenz: VERSA-Score vertrauenswürdig.
    Niedrige Konfidenz: VERSA+MERT blenden, MERT mehr Gewicht.
    """
    base_confidence = {
        "cd_digital": 0.95, "dat": 0.90, "minidisc": 0.85,
        "mp3_high": 0.87, "mp3_low": 0.80,
        "vinyl": 0.72, "reel_tape": 0.70, "tape": 0.68,
        "shellac": 0.45, "wax_cylinder": 0.30,
    }.get(material_type, 0.65)
    # SNR-Malus
    snr_malus = max(0.0, (15.0 - snr_estimate_db) / 50.0)
    return max(0.10, base_confidence - snr_malus)
```

**Anwendung in HPI-Berechnung (§2.44)**:

```python
versa_confidence = compute_versa_confidence(snr_estimate_db, material_type)
mert_weight = 1.0 - versa_confidence  # mehr MERT bei niedrigem VERSA-Vertrauen
mos_composite = versa_score * versa_confidence + mert_score * mert_weight
```

**Protokollierung**: `metadata["versa_confidence"]`, `metadata["mos_blend_weights"]`.

---

## §8.1.5 [TARGET_2026] Hörertest-Protokoll (ITU-R BS.1534-3 MUSHRA)

**Status**: Noch nicht durchgeführt (kein echtes Panel). Protokoll normativ definiert für zukünftige externe Validierung.

**Protokoll**:

1. **Material**: 10 Songs (2 pro Ära: 1920er Shellac, 1940er Vinyl, 1960er Tape, 1980er Kassette, 2000er MP3-low)
2. **Teilnehmer**: ≥ 15 ausgebildete Hörer (Tonstudium oder ≥ 5 Jahre professionelle Hörerfahrung)
3. **Methode**: MUSHRA — verstecktes Referenz-Anker-Design
   - Referenz: bekanntes Studio-Master (verborgen)
   - Anchor: 3.5 kHz Lowpass (ITU-R BS.1534-3 Standard)
   - Kandidaten: Aurik Restoration, Aurik Studio 2026, iZotope RX 11, Manual Restoration
4. **Skala**: 0–100 (100 = nicht unterscheidbar von Referenz)
5. **Mindest-Score für extern belegbaren Spitzenanspruch**: Ø ≥ 80 MUSHRA über alle Materialien
6. **Protokoll-Datei**: `docs/mushra_protocol.pdf` (nach Durchführung)

**Bis zur Durchführung**: Algorithmischer OQS-Score (§8.1.1) als Proxy — klar als solcher ausgewiesen.

---

## §8.1.6 [RELEASE_MUST] AMRB-Regressions-Benchmark — Historisches Score-Tracking (v10.0.0)

**Problem**: Es gibt keinen strukturierten Nachweis, dass Aurik über Versionen besser wird und nicht schlechter. Einzelne Fixes könnten andere Metriken verschlechtern.

**Pflicht-Tracking-Datei** (`benchmarks/amrb_history.json`):

```json
{
  "format_version": 1,
  "entries": [
    {
      "version": "9.12.0",
      "date": "2026-05-01",
      "amrb_scores": {
        "AMRB-01-VINYL": {"oqs": 82.3, "p1_floor_met": true},
        "AMRB-02-TAPE":  {"oqs": 79.1, "p1_floor_met": true},
        "AMRB-03-SHELLAC": {"oqs": 63.4, "p1_floor_met": true}
      },
      "notes": "VocalQualityIndex + SongStructureAnalyzer eingeführt"
    }
  ]
}
```

**CI-Invariante**: Beim Release einer neuen Hauptversion (9.x.0) MUSS `benchmarks/amrb_history.json` aktualisiert werden. Wenn ein neuer Eintrag AMRB-Score verschlechtert (OQS-Delta < −2.0 vs. Vorgänger-Version), ist das ein Release-Blocker.

**Automatisierung**: `benchmarks/update_amrb_history.py` [ROADMAP] — liest aktuellen Score aus AMRB-Testlauf und schreibt neuen Eintrag.

---

## §5 Test-Standards

### §5.4 [RELEASE_MUST] ML-Headroom-Guard + KMV-Rueckgewinnung

Folgende Testfaelle sind fuer heavy ML-Phasen verpflichtend (z. B. phase_03, phase_06, phase_20, phase_23, phase_24, phase_55):

1. **Low-RAM Completion Test**: Unter simuliert knappem RAM darf die Restaurierung nicht crashen; Ergebnis muss erfolgreich exportierbar sein.
2. **Guard Event Contract Test**: Bei Guard-Trigger muss `metadata["ml_guard_events"]` gesetzt sein und alle Pflichtfelder enthalten.
3. **Deferred-Phase Contract Test**: Guard-betroffene Phase muss in `deferred_phases` eingetragen werden.
4. **KMV Recovery Test**: Stufe 2 fuehrt die deferred Phase ohne RT-Limit nach; Overwrite nur bei `stufe2_quality_estimate >= stufe1_quality`.
5. **No-Original-Rollback Test**: Guard-Trigger darf nie zu `action="rollback"` auf Original-Audio fuehren.

**Regression-Fokus:** Diese Tests muessen sowohl mono als auch stereo sowie kurz (<60 s) und lang (>=180 s) abdecken.

### §5.5 [RELEASE_MUST] Vollpipeline-Determinismus-Gate

Bei identischer Eingabe, Umgebung und Konfiguration muessen zwei Vollpipeline-Runs bitnah identisch sein.

Pflichtkriterien:

1. `max_abs_err <= 1e-6`
2. `rms_err <= 1e-7`
3. identische `phases_executed`
4. identische `release_mode`-Entscheidung

### §5.6 [RELEASE_MUST] Stratifiziertes Konkurrenz-Gate

Konkurrenzvergleich wird nicht nur als Gesamtmittel, sondern pro Zelle einer Material-Defekt-Matrix bewertet.

Pflicht-Matrix:

- Materialien: `tape`, `vinyl`, `shellac`, `digital`, `vocal`
- Defektklassen: `hiss`, `crackle`, `dropout`, `reverb`, `hum`, `codec`

Release-Logik:

- Fail bei regressiver Einzelzelle gegen Referenz, auch wenn Gesamtmittel besteht.
- Bericht muss Delta pro Zelle enthalten.

### §5.7 [RELEASE_MUST] Externes Mini-MUSHRA-Artefakt

Bei Kern-aenderungen (Kernphasen, PMGG, DefectScanner, heavy ML-Fallbacks) ist ein externer Mini-MUSHRA-Bericht Pflicht.

Pflichtanforderungen:

1. mindestens 6 Szenarien, davon mindestens 2 Vocal-Szenarien
2. mindestens 8 Hoerer
3. Szenario-Score, Konfidenzintervall, Delta zur Vorversion
4. Bericht als Release-Artefakt versioniert abgelegt

### §5.8 [RELEASE_MUST] Test-Assertion-Konvention für numpy-Toleranzen (NEU 2026-07-12)

Toleranzen (`rtol`, `atol`) gehören AUSSCHLIESSLICH in `np.testing.assert_allclose()`, NIEMALS in numpy-Mathefunktionen.

```python
# KORREKT:
np.testing.assert_allclose(actual, np.abs(expected), rtol=1e-5, atol=1e-8)
np.testing.assert_allclose(actual, np.tanh(x), atol=1e-6)
np.testing.assert_allclose(actual, np.zeros(N))

# VERBOTEN — TypeError zur Laufzeit:
np.abs(x, rtol=1e-5, atol=1e-8)       # np.abs() kennt keine Toleranzen
np.tanh(x, rtol=1e-5, atol=1e-8)       # np.tanh() kennt keine Toleranzen
np.zeros(N, rtol=1e-5, atol=1e-8)      # np.zeros() kennt keine Toleranzen
np.array([...], rtol=1e-5, atol=1e-8)  # np.array() kennt keine Toleranzen
```

**CI-Gate:** Kein Test darf durch diesen Fehler brechen. Pattern-Check via
`grep -rPn '(?<=np\.(abs|tanh|zeros|array|ones|full|arange))\(' tests/ | grep rtol` als Pre-Commit.

### §5.7a [RELEASE_MUST] Modusgetrennte Hörvalidierungs-Checkliste (v10.0.0)

Die externe Hörvalidierung MUSS beide Modi getrennt ausweisen. Ein kombinierter
Gesamtwert ohne Modus-Trennung ist unzulässig.

**Pflicht-Checkliste Restoration (`mode=restoration`):**

1. Blindvergleich `Input` vs `Restored` vs `Reference/Needledrop-Best-Available` dokumentiert
2. Bewertungsachsen enthalten mindestens:
    `Natürlichkeit`, `Authentizität`, `Artefaktfreiheit`, `Tonale Treue`
3. Keine hörbare Verschlechterung bei P1/P2-Achsen gegenüber Input
4. Carrier-Chain-Inversion ist nachvollziehbar pro Szenario dokumentiert
5. **Szenarien-Pflicht**: Alle 3 mandatory Szenarien (`docs/reports/hearing_test_scenarios_restoration.yaml`) müssen als Validierungsbasis durchlaufen werden:
    - `RESTORATION_SCENARIO_1`: Vinyl Wear + Surface Noise (Rock Vocal)
    - `RESTORATION_SCENARIO_2`: Tape Hiss + Oxide Dropout (Jazz Vocal)
    - `RESTORATION_SCENARIO_3`: Shellac Brittleness + Click Storm (Classical Vocal)
6. Jedes Szenario muss die dokumentierten `mandatory_validation_points` erfüllen
7. GO/NO-GO-Entscheidung folgt Restoration-Entscheidungslogik in `docs/guides/GO_NO_GO_DECISION_PROTOCOL.md` (Phase 1–3)

**Pflicht-Checkliste Studio 2026 (`mode=studio2026`):**

1. Blindvergleich `Input` vs `Restored` vs `Modern Reference` dokumentiert
2. Bewertungsachsen enthalten mindestens:
    `Frische/Presence`, `Punch/Bass-Kraft`, `Klarheit`, `Artefaktfreiheit`
3. OQS-Gate (§8.1.1a) und Hörurteil dürfen sich nicht widersprechen
4. Bei Widerspruch gilt Hörurteil als Release-Blocker bis Root-Cause-Analyse vorliegt
5. **Szenarien-Pflicht**: Alle 3 mandatory Szenarien (`docs/reports/hearing_test_scenarios_studio2026.yaml`) müssen als Validierungsbasis durchlaufen werden:
    - `STUDIO2026_SCENARIO_1`: Compressed Pop Mix + Thin Vocal (Pop/Dance Vocal)
    - `STUDIO2026_SCENARIO_2`: Lo-Fi Hip-Hop Muddy Mix + Weak Vocal (Hip-Hop Vocal)
    - `STUDIO2026_SCENARIO_3`: Acoustic Folk Thin + Narrow Stereo (Folk Vocal)
6. Jedes Szenario muss die dokumentierten `mandatory_validation_points` erfüllen (OQS ≥ 86–88, PQS MOS ≥ 4.3–4.5)
7. GO/NO-GO-Entscheidung folgt Studio-2026-Entscheidungslogik in `docs/guides/GO_NO_GO_DECISION_PROTOCOL.md` (Phase 1–4)

**PR-Artefakt-Pflicht:**

- Pro Kernänderung muss ein ausgefülltes Template abgelegt werden:
  `docs/guides/PR_HOERVALIDIERUNG_TEMPLATE.md`
- Pflichtfelder: Szenarien, Hörerzahl, Ergebnis je Modus, Blocker/Entscheidung,
  Link auf Rohdaten/Anhang.

**Verfahren für großflächige Validierung (6-Szenario-Suite):**

1. **Input**: Hörvalidierung durchlaufen mit ≥ 8 Hörern pro Szenario (total 48 Hörer × Scenario-Datum)
2. **Prozess**: Folge dem **Structured GO/NO-GO Decision Protocol** (`docs/guides/GO_NO_GO_DECISION_PROTOCOL.md`)
3. **Decision Flow**:
    - Restoration-Modi: Pre-Review Checks → Aggregate Gates → Per-Scenario Thresholds → Summary Decision
    - Studio2026-Modi: Pre-Review Checks → Objective Metric Gates (OQS, PQS, Artifacts) → Listener MOS → Mode-Goals → Per-Scenario Fine-Grained → Summary Decision
4. **Output**: Go/No-Go Entscheidung mit vollständiger Audit-Dokumentation in PR
5. **Escalation**: Bei NO-GO oder Conditional-GO siehe Protocol §VI (Remediation & Re-Test)

**Verboten:**

- rein algorithmische Freigabe ohne externes Hörvalidierungsartefakt
- Vermischung von Restoration- und Studio-2026-Ergebnissen in einer Einzelnote
- Verwendung von Hörer-Scores aus der falschen Szenario-Spur (Restoration × Studio2026 nicht mischen)

### Mindestanforderungen pro neuem Modul

| Anforderung | Zielwert |
| --- | --- |
| Unit-Tests pro Kernmodul | ≥ 35 |
| Shape/Dtype-Tests | ✅ Pflicht |
| NaN/Inf-Tests | ✅ Pflicht |
| Bounds-Tests | ✅ Pflicht (alle metrischen Ausgaben) |
| Edge Cases | ✅ Pflicht (Stille, Rauschen, Dirac-Impuls) |
| Mono + Stereo | ✅ Pflicht für Audio-Eingaben |
| Konsistenz | ✅ Pflicht (selbe Eingabe → selbe Ausgabe) |
| Integration-Tests | ≥ 5 |

### Namenskonvention

```text
tests/unit/test_v<VERSION>_<feature_name>.py
# Beispiele:
tests/unit/test_v97_cognitive_layer.py
tests/unit/test_v99_genre_schlager.py       # ≥ 35 Tests
tests/unit/test_v910_remaster_detector.py
```

### pytest-Konfiguration (`pytest.ini`)

```ini
[pytest]
addopts = --timeout=30 --import-mode=importlib -p no:warnings
```

- Kein Test darf > 30 s dauern
- Alle Tests mit **synthetischen Signalen** (`np.random.seed(42)`)
- Keine realen Audio-Dateien in Tests

### §5.8 [RELEASE_MUST] Pytest-Teardown-Stabilität für große Suiten (v10.0.0)

**Problem**: Unbedingtes `gc.collect()` nach jedem Test kann in Suiten mit tausenden Tests sporadische `pytest-timeout`-Fehler im Teardown auslösen. Zusätzlich können lang lebige Monitor-Threads (z. B. Plugin-/RAM-Manager) nach Testende weiterlaufen und den Teardown verfälschen.

**Pflichtregeln:**

1. Per-Test-Teardown nutzt standardmäßig **inkrementellen/leichten GC** (`gc.collect(0)` oder äquivalent), nicht volles `gc.collect()`.
2. Volles `gc.collect()` ist nur **cadence-gesteuert**, dateibasiert oder sessionbasiert erlaubt.
3. Diagnose-Kadenz darf optional über Env-Flag gesteuert werden; Standardverhalten muss für lokale Unit-Suiten timeout-stabil bleiben.
4. Hintergrund-Manager mit eigenem Thread müssen in `pytest_sessionfinish`, Fixture-Finalizern oder äquivalentem Session-Cleanup best-effort gestoppt werden.
5. Singleton-basierte Manager dürfen nach Session-Cleanup auf `None` zurückgesetzt werden, damit Folge-Sessions keinen stale Thread-/State-Rest sehen.

**Referenz-Pattern:**

```python
_GC_FULL_INTERVAL = int(os.environ.get("AURIK_TEST_FULL_GC_INTERVAL", "0"))
_gc_teardown_counter = 0

@pytest.fixture(autouse=True)
def _gc_after_test():
    yield
    global _gc_teardown_counter
    _gc_teardown_counter += 1
    gc.collect(0)
    if _GC_FULL_INTERVAL > 0 and (_gc_teardown_counter % _GC_FULL_INTERVAL) == 0:
        gc.collect()

def pytest_sessionfinish(session, exitstatus):
    manager.shutdown()   # best-effort, non-blocking
    singleton_module._instance = None
```

**VERBOTEN:**

- Unbedingtes Full-GC in jedem Test-Teardown der Standard-Unit-Suite.
- `join()` ohne Timeout in Test-Cleanup-Pfaden.
- Verlassen auf `daemon=True` als einziges Lebenszyklus-Modell für Hintergrund-Threads.

---

## §14 E2E-Test-Spezifikation (Pflicht ab v10.0.0)

### §14.1 Pflicht-Assertions (`_validate_restoration_result`)

```python
assert len(set(result.phases_executed) & TIER_1_PHASES) >= 2
assert result.quality_estimate >= 0.55              # Formel: §8.1
assert result.material_type is not None
assert result.metadata.get("era") is not None
assert result.metadata.get("panns_tags") is not None
assert np.isfinite(audio).all()
assert np.max(np.abs(audio)) <= _TP_LIMIT + 1e-4
assert len(set(result.phases_executed) & TIER_6_PHASES) >= 3
```

### §14.2 Musik-Qualitäts-Assertions

```python
assert pqs_result.mos >= 4.0   # QUALITY; >= 4.5 für MAXIMUM
assert all(scores[g] >= checker.thresholds[g] for g in applicable_goals)
# Schlager-spezifisch:
assert scores["tonal_center"] >= 0.97
assert scores["waerme"] >= 0.88
```

### §14.3 Pipeline-Integrität

```python
assert config.enable_performance_guard is True
assert config.enable_phase_gate is True
# Keine starre Obergrenze für Recovery-Schritte: entscheidend ist, dass
# best_effort-Ereignisse transparent protokolliert und nicht als stiller
# Success gewertet werden.
assert all("action" in e for e in (result.phase_gate_log or []))
```

---

## §5.9 [RELEASE_MUST] QualityGate Early-Exit-Optimierung (v10.0.0)

`check_dsp()` und `check_ml()` in `backend/core/quality_gate.py` dürfen teure STFT/SNR-Analysen
(`_check_audio_array`) **erst nach** einer positiven Musical-Goals-Prüfung ausführen.

**Invariante**:

```python
def check_dsp(self, audio, sr, context):
    result = self._check_musical_goals(context)
    if result.failed:
        return result  # sofort zurück — keine STFT-Arbeit
    return result.merge(self._check_audio_array(audio, sr))
```

**Rationale**: Musical-Goals-Failures treten häufiger auf als SNR/STFT-Failures.
Frühes Return spart STFT-Berechnungszeit (50–200 ms je Messung).

**Testpflicht**: `tests/unit/test_quality_gate_early_exit.py` — verifiziert,
dass `_check_audio_array` nach Musical-Goal-Failure nicht aufgerufen wird.

## §5.10 [RELEASE_MUST] TFS-Guard Hilbert-After-Voiced-Gate (v10.0.0)

`backend/core/tfs_preservation_guard.py` darf teure Analytic-Transforms (Hilbert-Phasenextraktion + `filtfilt` Band-Filterung) **erst nach** dem Voice-Energy-Gate ausführen.

**Invariante**:

```python
for band_idx in range(n_bands):
    # 1. Original-Band-Energie prüfen (günstig):
    orig_band = bandpass_filter(original, band_frequencies[band_idx])
    frame_energies = compute_frame_energy(orig_band)
    voiced_frames = np.sum(frame_energies > voiced_threshold_db)
    if voiced_frames < 3:
        continue  # Band überspringen — erst dann kein filtfilt + Hilbert
    # 2. Teure Hilbert-Extraktion (nur bei ausreichend Voiced-Frames):
    restored_band = bandpass_filter(restored, band_frequencies[band_idx])
    # ... filtfilt + Hilbert ...
```

**Allgemeines Muster**: Teure Analytic-Transforms IMMER nach dem günstigsten
Admissibility-Gate platzieren.

**Testpflicht**: `tests/unit/test_tfs_preservation_guard.py`.

---

## §8.4a Test-Implementierung — Patterns, Pitfalls & CI-Gates (konsolidiert aus Skills test-writing + quality-benchmark)

### Test-Verzeichnisstruktur

| Ordner | Inhalt | Marker | Timeout |
| --- | --- | --- | --- |
| `tests/unit/` | Schnelle Unit-Tests | — | ≤ 30 s |
| `tests/musical_goals/` | 14-Goal-Schwellwert-Tests | — | ≤ 30 s |
| `tests/integration/` | Modul-Übergreifende Tests | — | variabel |
| `tests/normative/` | CI-Gate-Tests (RELEASE_MUST) | — | variabel |
| `tests/regression/` | Regressions-Absicherung | — | variabel |
| `tests/e2e/` | End-to-End mit echtem Audio | `e2e` | variabel |
| `tests/test_startup_smoke.py` | Startup-Smoke-Test (§v10.305 §G77): GPU+Warmup+PreAnalysis | `smoke` | ≤ 60 s |

### Marker-System

| Marker | Bedeutung | Standard-Suite? |
| --- | --- | --- |
| `ml` | ML-Modell wird geladen | NEIN (nur `--run-heavy-tests`) |
| `slow` | Timeout > 30 s | NEIN (nur `--run-heavy-tests`) |
| `e2e` | End-to-End mit I/O | NEIN (explizit) |
| (kein Marker) | Standard Unit-Test | JA |

`conftest.py` markiert automatisch `ml`/`slow` basierend auf Testinhalten.

### §8.4b [RELEASE_MUST] Host-Crash-Safety für Heavy-ML-Tests (v10.0.0)

Tests dürfen den Host nicht destabilisieren. Für potenziell hostgefährdende ML-Pfade
(hoher RAM/VRAM-Verbrauch, große ONNX/Torch-Modelle, lange Audiosegmente) gilt bindend:

1. **Default sicher**:
    - Standard-Pytest-Runs (ohne `--run-heavy-tests`) dürfen keine Heavy-ML-Pfade ausführen.
    - Solche Tests müssen über Marker/Heuristik (`ml`, `slow`, `e2e`, bekannte Heavy-Dateien)
      frühzeitig deselected werden.
2. **Explizites Opt-in**:
    - Heavy-ML-Tests sind nur mit `--run-heavy-tests` zulässig.
    - Runtime-Guards in Phasen dürfen in Testumgebung Heavy-Pfade nur bei aktivem Opt-in erlauben.
3. **Crash-Prävention vor Coverage**:
    - Bei Konflikt gilt Host-Stabilität vor Test-Coverage.
    - Zulässige Fallback-Reaktion ist deterministische DSP-Ausführung statt ML-Ausführung.
4. **Kein stilles Umgehen**:
    - Unit-Tests, die ML-Pfade gezielt prüfen, müssen Heavy-Opt-in explizit setzen
      (z. B. Test-Setup/Monkeypatch), statt globale Guards zu umgehen.

**Invariante**: Kein Standard-Unit-Run darf einen Host-Neustart oder System-Hard-Freeze verursachen.

### Pflicht-Test-Taxonomie (≥ 35 pro Kernmodul)

1. **Shape** — Mono, Stereo, verschiedene Längen
2. **NaN/Inf** — Input und Output
3. **Bounds** — Clip [-1, 1]
4. **Edge-Cases** — leeres Audio, 1 Sample, sehr langes Audio
5. **Mono UND Stereo** — beide Orientierungen `(2,N)` und `(N,2)` testen
6. **Musical Goals** — kein Ziel nach Modul schlechter
7. **GrooveMetric** — DTW ≤ 8 ms RMS
8. **SOFT_SATURATION** — nicht als CLIPPING detektiert
9. **Pass-Through** — SNR > 40 dB → PQS-MOS-Verlust ≤ 0.05
10. **quality_estimate** — ≥ 0.55 im E2E
11. **Sample-Rate** — `assert sr == 48000` löst `AssertionError` bei 44100

### Test-Pattern (Vorlage)

```python
import numpy as np
import pytest

class TestPhaseXX:
    @pytest.fixture
    def mono_audio(self):
        return np.random.randn(48000 * 3).astype(np.float32) * 0.5

    @pytest.fixture
    def stereo_audio(self):
        return np.random.randn(2, 48000 * 3).astype(np.float32) * 0.5

    def test_output_shape_mono(self, mono_audio):
        result, meta = execute(mono_audio, 48000)
        assert result.shape == mono_audio.shape

    def test_no_nan_inf(self, mono_audio):
        result, _ = execute(mono_audio, 48000)
        assert np.isfinite(result).all()

    def test_clipped_output(self, mono_audio):
        result, _ = execute(mono_audio, 48000)
        assert np.max(np.abs(result)) <= 1.0

    def test_strength_zero_passthrough(self, mono_audio):
        result, _ = execute(mono_audio, 48000, strength=0.0)
        np.testing.assert_array_almost_equal(result, mono_audio, decimal=5)

    def test_metadata_fields(self, mono_audio):
        _, meta = execute(mono_audio, 48000)
        assert "phase_id" in meta
        assert "applied" in meta
```

### 6 bekannte Test-Pitfalls (aus realen Fehlern)

**1. Budget-Tests: `is_system_thrashing` immer mocken**

```python
def test_budget_logic(monkeypatch):
    import backend.core.ml_memory_budget as budget
    monkeypatch.setattr(budget, "is_system_thrashing", lambda: False)
```

**2. `np.corrcoef` auf near-constant → RuntimeWarning**
Stets guarded correlation: `dot(a,b) / (||a||·||b|| + ε)` — NaN-safe, kein Warning.

**3. `resampy`/`librosa` `pkg_resources`-Warnings unter `-W error`**
Upgrade auf `resampy >= 0.4.3`.

**4. Teure Transforms nach günstigstem Gate**
Hilbert, `filtfilt`, STFT immer NACH Frame-Energie/Voiced-Gate — nicht davor.

**5. `scipy.signal.stft(boundary='reflect')` → ValueError**
Scipy < 1.12: `boundary='even'` verwenden.

**6. Stereo-Array-Orientierung**
Fixtures mit `(2,N)` UND `(N,2)` testen — viele Bugs nur bei einer Orientierung.

### CI-Gate-Tests (RELEASE_MUST)

| Test-Datei | Prüft |
| --- | --- |
| `test_no_docker_in_production_paths.py` | Kein Docker/Cloud |
| `test_competitive_ci_gate.py` | OQS vs iZotope RX 11 |
| `test_performance_budget_ci_gate.py` | RT-Limits |
| `test_combined_ml_memory_budget.py` | ML-Budget ≤ 12 GB |
| `test_hybrid_release_mode.py` | Fallback-Kaskaden |
| `test_full_pipeline_determinism.py` | Bitnahe Reproduzierbarkeit |
| `test_competitive_stratified_gate.py` | Material × Defektklasse |
| `test_stability_invariants.py` | 9 Stabilitäts-Punkte |
| `test_lyrics_guided_enhancement_gate.py` | §2.36 aktiv + Modellpfade |
| `test_external_mushra_artifact_contract.py` | Mini-MUSHRA Artefakt |
| `tests/normative/test_spec_consistency.py` | Spec-Code-Sync: CAUSE_TO_PHASES ↔ CAUSES, §0a-Purity, V12-Bidirektional, Spec 06 vs. Code |
| `tests/normative/test_section_0a_restoration_guard.py` | §0a Crossfire-Invariante — verbotene Phasen niemals in Restoration-Pipeline |
| `tests/normative/test_mas_convergence.py` | §0k/§2.64/§2.65 MAS-Invarianten — Targets ≥ Floor, PHYSICAL_CEILING, Delta-Rollback-Schwelle, Early-Stop-Logik |

### OQS — Berechnung & Stufen

Modul: `backend/core/mushra_evaluator.py` (v10.0.0-Phantom) (algorithmische PEAQ-Approximation — **kein** ITU-R-MUSHRA).
In externen Berichten: „OQS (algorithmisch)".

| Stufe | Score | Pflicht |
| --- | --- | --- |
| Good (B) | ≥ 80 | **[RELEASE_MUST]** — Pflicht für jede neue Phase/Plugin |
| Excellent (A) | ≥ 91 | Exzellenz-Label — kein harter Gate-Wert |

**[TARGET_2026]** Studio-2026-Ziel: OQS ≥ **88** — Roadmap-Ziel, kein Release-Blocker.

### AMRB-Details

**Seeding-Invariante**: `_sid_offset(sid)` via **MD5** — KEIN `hash(sid)` (Python-zufällig).
Nightly: `n_items ≥ 5`. Baseline: iZotope RX 11 (commercial) mit OQS 71.0.

### §2.40 Stratifiziertes Konkurrenz-Gate

Aurik muss **pro Material UND pro Defektklasse** bestehen:
`tape/vinyl/shellac/digital/vocal × hiss/crackle/dropout/reverb/hum/codec`.
Release failt bei regressiver Zelle auch wenn Gesamt-OQS besteht.

### Material-MOS-Interpretation

| Material | MOS-Minimum | Physikalische Begründung |
| --- | --- | --- |
| cd_digital / dat / mp3_high / aac | ≥ 4.5 | Geringes Defekt-Potential, hohe Erwartung |
| Tape | ≥ 4.2 | Moderate Bandbreite, Hiss-Removal stabil |
| Vinyl | ≥ 4.0 | RIAA-Inversion + Crackle-Grenzen |
| Shellac | ≥ 3.8 | Physikalische Bandbreite ≤ 7 kHz |

### §8.4 Externes Mini-MUSHRA-Protokoll

Bei Änderungen an Kernphasen, PMGG, DefectScanner oder heavy ML-Fallbacks:

- Mindestens 6 Szenarien (2 Vocal), mindestens 8 Hörer
- Pflichtbericht als Artefakt (Scores, Konfidenzen, Delta)

### Era-GP-Warmstart-Distributionen (§2.14)

- ≤ 1940: `noise_reduction_strength ~ N(0.90, 0.05)`
- ≤ 1960: `N(0.75, 0.08)`
- ≥ 1970: `N(0.50, 0.10)`

### §8.6 [RELEASE_MUST] Worldclass Hybrid-Engineer Protocol (v10.0.0)

Rolle Aurik: hybrider Restaurierungstoningenieur fuer Musik mit Gesang.
Diese Rolle ist nur erfuellt, wenn menschlich nachbildbare Spitzenfaehigkeiten
und maschinelle Vorteile gleichzeitig, messbar und reproduzierbar im Ergebnis vorliegen.

#### §8.6a Human-Talent-Emulation-Vektor (HTEV)

Jeder Release-Lauf muss einen HTEV-Metadatensatz bereitstellen:

- `vocal_identity_preservation` (singer_identity_cosine)
- `formant_integrity` (F1-F4 Delta)
- `vibrato_depth_preservation` (Hz-Modulationstiefe)
- `breath_naturalness` (Atemsegment-Erhalt)
- `micro_dynamic_correlation` (voiced frame energy correlation)
- `transient_articulation` (Onset-Shift/Onset-Energie)
- `stereo_scene_stability` (Mono-Kompatibilitaet + Interchannel-Lag)
- `noise_texture_authenticity` (Noise-Textur-Distanz)
- `spectral_color_preservation` (1/3-Oktav-Korrelation)
- `emotional_arc_preservation` (global + local arousal)
- `artifact_freedom` (primaerer Veto-Faktor)
- `goal_team_balance` (15-Goal-Teamabweichung)

Pflicht: `metadata["hybrid_engineer_vector"]` mit allen 12 Schluesseln und
Werten in [0.0, 1.0] (oder explizit normierter Delta-Umrechnung).

#### §8.6b Psychoakustischer Weltspitzen-Composite-Score

Fuer Release-Entscheidungen ist zusaetzlich ein zusammengesetzter
Weltspitzen-Score zu fuehren:

```text
WCS = 0.30 * artifact_freedom
    + 0.20 * vocal_identity_preservation
    + 0.15 * formant_integrity
    + 0.10 * micro_dynamic_correlation
    + 0.10 * emotional_arc_preservation
    + 0.10 * spectral_color_preservation
    + 0.05 * stereo_scene_stability
```

Release-Mindestziele:

- Restoration mit Gesang (`panns_singing >= 0.35`): `WCS >= 0.88`
- Studio 2026 mit Gesang (`panns_singing >= 0.35`): `WCS >= 0.91`
- Instrumental: `WCS >= 0.85`

Hinweis: `artifact_freedom < 0.95` blockiert weiterhin absolut,
unabhaengig vom WCS.

#### §8.6c Wissenschaftliche Evidenzklassen (Gate-faehig)

Jeder neue Schwellwert in HPI/AFG/VQI/WCS benoetigt eine Evidenzklasse:

- Klasse A: Norm oder peer-reviewed Primarquelle + interne Reproduktion
- Klasse B: peer-reviewed Sekundaerachse + robuste interne Reproduktion
- Klasse C: kalibriert auf AMRB/UAT, befristet mit Revalidierungsdeadline

Verboten:

- Klasse-C-Schwellen ohne Revalidierungsdatum
- Klasse-C-Schwellen als dauerhafte RELEASE_MUST-Basis ohne Upgrade-Plan

Pflichtmetadaten pro Schwelle:

- `source_class` (A/B/C)
- `source_ref`
- `validated_on`
- `revalidate_by` (bei Klasse C)

#### §8.6f [RELEASE_MUST] Scientific Threshold Evidence Registry

Die Gate-Schwellen fuer `artifact_freedom_gate`, `vqi_gate`, `hpi_gate` und
`worldclass_composite_gate` muessen zentral in
`policy/scientific_threshold_evidence_registry.yaml` gepflegt werden.

Verbindliche Regeln:

- Runtime-Metadaten (`threshold_evidence` in UV3) muessen fachlich auf den
    Registry-Eintraegen basieren.
- Jede Gate-Schwelle braucht mindestens eine wissenschaftliche Quellenachse:
    DOI-Quelle, ITU/EBU-Norm oder gleichwertige Primarquelle.
- `source_class: C` ist nur mit `revalidate_by` zulaessig.
- Aenderungen an Gate-Schwellen ohne Update der Registry sind Release-Blocker.

#### §8.6g [RELEASE_MUST] Psychoakustischer Natuerlichkeits-Guard (Anti-klinisch)

Restaurierungen duerfen nicht klinisch-steril klingen. Deshalb ist zusaetzlich zum
WCS ein psychoakustischer Natuerlichkeits-Guard verpflichtend.

Guard-Signalachsen:

- `noise_texture_authenticity`
- `micro_dynamic_correlation`
- `emotional_arc_preservation`
- `spectral_color_preservation`

Bewertung:

```text
PSYCHO = 0.28 * noise_texture_authenticity
    + 0.24 * micro_dynamic_correlation
    + 0.24 * emotional_arc_preservation
    + 0.24 * spectral_color_preservation
```

Mindestziele:

- Restoration mit Gesang (`panns_singing >= 0.35`): `PSYCHO >= 0.84` und jede Achse `>= 0.80`
- Studio 2026 mit Gesang (`panns_singing >= 0.35`): `PSYCHO >= 0.87` und jede Achse `>= 0.80`
- Instrumental: `PSYCHO >= 0.82` und jede Achse `>= 0.76`

Pflichtmetadaten:

- `metadata["psychoacoustic_naturalness_gate"]`
- `metadata["threshold_evidence"]["psychoacoustic_naturalness_gate"]`

Adaptive Recovery-Pflicht:

- Falls `psychoacoustic_naturalness_gate.passed == False`, MUSS UV3 vor finaler
    Degradation einen konservativen Recovery-Versuch ausfuehren (Blend mit sicheren
    Referenzen wie Original/Checkpoint).
- Recovery darf nur uebernommen werden, wenn das Psychoakustik-Gate danach
    nachweislich besteht.
- Recovery-Telemetrie ist verpflichtend unter
    `metadata["psychoacoustic_feedback_recovery"]`.

Phasenweise Rueckkopplung (fruehe Anti-Klinik-Steuerung):

- UV3 MUSS fuer klangpraegende Phasen einen psychoakustischen Strength-Scalar
    ableiten und konservativ anwenden, wenn Risikosignale auftreten
    (z. B. Tilt-Guard-Trigger, Zwicker-Rescue, hohes HNR-Budget).
- Zusaetzlich MUSS UV3 laufende Psycho-Delta-Metriken aus den Per-Phase-Goal-
    Delten akkumulieren (insb. natuerlichkeit, authentizitaet, emotionalitaet,
    micro_dynamics, transparenz) und in den naechsten Strength-Scalar einspeisen.
- Die Skalierung ist rein daempfend (`<= 1.0`) und darf harte Safety-Gates
    nicht umgehen.
- Pro Phase sind Telemetrie-Felder in der Metadata-Akkumulation zu fuehren:
    `psycho_strength_scalar`, `psycho_strength_risk_score`,
    `psycho_strength_signals`, sowie Runtime-Status unter
    `_psycho_runtime_state`.

#### §8.6d Human-vs-Machine-Kooperationsinvariante

Maschinelle Gewinne (SNR, Rauschreduktion, BW-Recovery) duerfen nie auf Kosten
human-kritischer Vokalwahrheit gehen.

Bindende Konfliktauflosung:

1. Stimmintegritaet
2. Emotionale Authentizitaet
3. Musikalischer Kontext
4. Technische Kennzahlen

Wenn eine niedrigere Ebene eine hoehere verletzt, muss die Pipeline automatisch
auf die hoehere Ebene zurueckpriorisieren (adaptive Strength-Reduktion, Rollback,
oder sichere Alternativkette).

#### §8.6e Testpflicht fuer Weltspitzen-Claim

Neue oder geaenderte Kernlogik (Phasen, Gates, Aggregationsmetriken,
Fallback-Kaskaden, Vokalpfade) ist nur release-faehig mit:

1. `tests/normative/test_worldclass_hybrid_engineer_vector.py`
2. `tests/normative/test_worldclass_composite_score_gate.py`
3. `tests/normative/test_evidence_class_metadata_contract.py`
4. `tests/normative/test_psychoacoustic_naturalness_gate.py`
5. mindestens einem Real-Audio-Gate-Lauf auf vokalem Material

Fehlt einer dieser Nachweise, ist der Weltspitzen-Claim fuer den Patch nicht gueltig.

#### §8.6h Operative Erkenntnisbasis (Psychoakustik)

Fuer psychoakustische Kern-Changes ist die konsolidierte Engineering-Basis in
`docs/PSYCHOACOUSTIC_ENGINEERING_INSIGHTS_2026-05-21.md` normativ zu nutzen
(Architektur, Telemetrie, DoD, offene Risiken).

## v10 Test-Status

- 358 Unit-Tests bestehen
- 37 neue v10-Tests in `test_v10_worldclass_modules.py`
- Bridge-Compliance: 0 Bypasses in CLI und Batch
- ML-Fallback-Audit: 54 Module, 3 Silent-Failures behoben

## §7.10 — Perzeptuelle Qualitätssicherung (§v10.101)

### §7.10.1 Perzeptuelle Qualitätsgewichtung

Der QualityAnalyzer gewichtet ab v10.101 perzeptuell:

| Metrik | Gewicht (mit MUSHRA) | Gewicht (ohne MUSHRA) |
|--------|---------------------|----------------------|
| MUSHRA/OQS | 35% | — |
| Naturalness | 20% | 20% |
| Warmth | 15% | 15% |
| Clarity | 15% | 20% |
| SNR | 5% | 20% |
| Dynamic Range | 5% | 15% |
| THD | 5% | 10% |

### §7.10.2 Perzeptuelles Monitoring

Die Final-Summary zeigt DREI perzeptuelle Dimensionen (§G112):

- 📊 Signalqualität (technische Analyse)
- 🎧 Hörerlebnis (MUSHRA/OQS, simuliertes Hörerpanel)
- 🧠 Restaurations-Index (HPI, ganzheitlich)

### §7.10.3 ABX-Validierung

Der ABX-Test (`test_harness.py`) vergleicht nicht nur Pegel, sondern
Bark-Band-Energien und MUSHRA-Scores. Signifikanz: Binomialtest mit α=0.05.

### §7.10.4 v10.101 Fix-Changelog

Die folgenden 20 Fixes wurden in v10.101 implementiert und sind
Teil der perzeptuellen Architektur:

| Fix | Beschreibung | Spec-Referenz |
|-----|-------------|--------------|
| F5 | _hpi_era UnboundLocalError | §G90 Blind-Referenz-Vektor |
| F6 | measure_all STFT-Cache | §4.10.7 |
| F7 | DTW Sakoe-Chiba-Radius-Fix | §G100 Hörbarkeit |
| F8 | DoNoHarmGuardian Crest-Faktor | §G107 Ermüdungsfreiheit |
| F9 | phase_26 np-Import-Fix | §G24 NaN/Inf-Schutz |
| F10 | ExcellenceOptimizer Cache-Singleton | §G101 Perzeptueller Blend |
| F12 | GOAL_BASELINE Preflight-Key-Fix | §G104 JND-Gate |
| F13 | QualityAnalyzer relative Schwelle | §G106 Qualitätsgewichtung |
| F14 | Artifact-Schwelle 0.95→0.90 | §G101 Perzeptueller Blend |
| F16 | Final-Export af=0.000 Guard | §G101 Perzeptueller Blend |
| F17 | Phase_40 fftconvolve-Optimierung | §G103 LUFS-Lautheit |
| F20 | MUSHRA+HPI in Final-Summary | §G112 Perzeptuelles Monitoring |
| F21 | STFT-Cache (Bass/Brillianz/Waerme) | §G106 Qualitätsgewichtung |
| F23 | PredictivePreflight adaptiv | §G104 JND-Gate |
| P1 | Perceptual-Blend in Wet/Dry | §G101, §V34 |
| P2 | Bark/LUFS-Utility | §G102, §G103 |
| P3 | JND-Gate in ProfiledPhaseCall | §G104, §V37 |
| P4 | Perzeptuelle Konsistenz | §G100–§G112 |
