 | §v10 Pleasantness-First

# Aurik 10 — Spec 04: DSP-Standards & SOTA-Algorithmen
>
> Psychoakustische Fundierung, SOTA-Entscheidungsmatrix, Pflicht-Algorithmen.
>
> **SOTA-Pflicht** gilt absolut: Nur Algorithmen auf dem aktuellen Stand der Wissenschaft
> erlauben die Klanggüte aus §0 — qualitativ hochwertigste automatisierte Restaurierung
> für Musik mit Gesang weltweit. Algorithmen ab 2018 als Minimum.
> Legacy-Algorithmen als Primärverarbeitung **VERBOTEN**.

### §4.11 ComfortGuard (v10.0.0-Phantom)

Psychoakustische Hörmüdungs-Prävention nach ISO 532-B (Zwicker Sharpness).
High-Shelf-Filter (fc=2.5 kHz, Q=0.5, max −3 dB) wird automatisch nach
JEDER Phase angewendet wenn die Sharpness im 2–5 kHz-Bereich >12% beträgt.

Implementiert in `backend/core/comfort_guard.py`, integriert in
`PhaseInterface._safe_process()`.

## §4.10 — Perzeptuelle DSP-Standards (§v10.101)

### §4.10.1 Bark-Band-Verarbeitungspflicht

Jede frequenzabhängige DSP-Operation MUSS in 24 kritischen Bark-Bändern
(Zwicker 1961, 0-15500 Hz) arbeiten. Lineare Hz-Bänder sind ab v10.101
deprecated und werden schrittweise migriert.

Referenz-Implementierung: `backend.core.dsp.bark_lufs_util.split_into_bark_bands()`

### §4.10.2 LUFS-Lautheitsmessung

Dynamik-Entscheidungen MÜSSEN ITU-R BS.1770-4 LUFS verwenden.
RMS ist als Lautheits-Proxy ab v10.101 VERBOTEN (§V36).

Referenz: `backend.core.dsp.bark_lufs_util.measure_lufs_per_bark()`

### §4.10.3 Perzeptueller Wet/Dry-Blend

Jeder Phasen-Blend MUSS `perceptual_blend()` verwenden (§G101, §V34).
Skalarer Blend (ein Wert × alle Frequenzen) ignoriert simultane Maskierung
und übernimmt unhörbare Änderungen.

Referenz: `backend.core.dsp.perceptual_blend.perceptual_blend()`

### §4.10.4 JND-Gate

Vor jeder Phase prüft `should_skip_phase()` ob die erwartete Wirkung hörbar ist.
<2 Bark-Bänder über JND → Phase überspringen (§G104, §V37).

Referenz: `backend.core.dsp.perceptual_gate.should_skip_phase()`

### §4.10.5 ISO-226-Hörschwelle

Alle Pegel-Entscheidungen konsultieren ISO 226:2003 (§G105, §V38).
Die absolute Hörschwelle variiert um >40 dB über das Spektrum.

### §4.10.6 Zeitliche Maskierung

Attack/Release orientieren sich an ISO 11172-3: Pre-Masking 20ms, Post-Masking 100ms.

## §4.1 Pflicht-Konzepte (mindestens eines pro DSP-Funktion)

| Konzept | Anwendung | Referenz |
| --- | --- | --- |
| **OMLSA / IMCRA** | Rauschunterdrückung stationär | Cohen (2002, 2003) |
| **Consistent Wiener Filter** | Spektrale Restaurierung (modernisierter Wiener) | Le Roux & Vincent (2013) |
| **PGHI** | Phasenkonsistenz nach Spektral-Modifikation | Perraudin et al. (2013) |
| **NMF mit β-Divergenz** | Spektrale Dekomposition, Inpainting | Févotte & Idier (2011) |
| **pYIN / SPICE** | Pitch-Tracking f₀ | Mauch & Dixon (2014) |
| **RBME** | Decrackle, Spectral Smoothing | Bando et al. (2019) |
| **Sinusoidal + Stochastic Modeling** | Dropout-Inpainting | Serra & Smith (1990) |
| **DTW / Optimal Transport** | Zeitausrichtung, Pitch-Korrektur | Cuturi (2013) |
| **Multi-Resolution STFT (MRSA)** | Adaptive Fenstergrößen | Bello et al. (2005) |
| **Psychoacoustic Masking** | Restaurierungs-Regler | ISO 11172-3 |
| **Harmonic Lattice (Fletcher)** | Inharmonizität-konsistente Partials | Fletcher (1964) |
| **Beat-Tracking (madmom)** | Phrasen-Kontext für Inpainting | Böck et al. (2016) |
| **DDSP (Eigenimplementierung)** | Instrument-Resonanzmodellierung | Engel et al. (ICLR 2020) |
| **ASA (Bregman)** | Common Onset, Common Fate | Bregman (1990) |
| **Virtual Pitch** | Missing Fundamental → BassKraftMetric | Moore et al. (2006) |
| **ISO 226:2003 Equal-Loudness** | BrillanzMetric/WaermeMetric-Gewichtung | ISO 226:2003 |
| **LUFS / ITU-R BS.1770-5** | Lautstärkenormalisierung (2023) | ITU-R BS.1770-5 |

> **DDSP-Implementierung**: Aurik nutzt eine leichtgewichtige NumPy/SciPy-Eigenimplementierung (`dsp/ddsp_synth.py`) — **KEIN** Google-`ddsp`-PyPI-Paket (benötigt TensorFlow). Die Eigenimplementierung deckt additive Synthese + Rauschfilter vollständig ab und ist out-of-the-box ohne TF lauffähig.

### §4.1a Sample-Rate-Vertrag (Dual-SR, [RELEASE_MUST])

- Analyse-/Klassifikations-Module arbeiten mit nativer Import-SR (`analysis_sr`).
- DSP/ML-Verarbeitungsstufen arbeiten strikt mit `processing_sr = 48000`.
- Resampling-Fehler in Richtung 48 kHz sind harte Fehler (fail-fast), kein Silent-Fallback auf Nicht-48k-Processing.
- Komponenten mit modellnativer SR duerfen intern resamplen (z. B. 48k -> 44.1k -> 48k), aber nur innerhalb der Komponente und mit Rueckgabe in `processing_sr=48000`.

---

## §4.1b [RELEASE_MUST] Psychoakustische Lautheitsmessung nach ISO 532-1 (Zwicker/Fastl)

**Warum LUFS nicht ausreicht**: LUFS (ITU-R BS.1770-5) verwendet K-Weighting (~A-Gewichtung +
Hochregal +4 dB). K-Weighting detektiert Tieftonrumpeln (200–300 Hz) nicht als Lautheitszunahme,
die das Gehör mit bis zu +6 Phon wahrnimmt (ISO 226:2003 Equal-Loudness-Contours). Ergebnis:
Rumble-Filter-Phasen werden bei LUFS-Only-Check fälschlich als lautheitsneutral eingestuft.

**MidPipeline-Guard nach subtraktiven Phasen mit großem Breitband-Impact**:

```python
# backend/core/dsp/psychoacoustics.py
def compute_specific_loudness_zwicker(audio: np.ndarray, sr: int) -> float:
    """
    ISO 532-1 stationäre Methode (Zwicker/Fastl).
    Returns total loudness N in sone.
    - Bark-Filterbank: 24 kritische Bänder (0–16 kHz)
    - Fletcher-Munson-Kurven: ISO 226:2003 Tabelle
    - Referenz: 1 sone = 40 phon bei 1 kHz
    """
```

**ΔN-Entscheidungstabelle** (Δ = output_sone - input_sone):

| ΔN (sone) | Wahrnehmung | Pipeline-Reaktion |
| --- | --- | --- |
| ≤ 0.5 | Lautheitsneutral | OK |
| 0.5 – 1.0 | Grenzbereich | INFO in `metadata["loudness_delta_sone"]` |
| 1.0 – 2.0 | Wahrnehmbar lauter | WARNING in `metadata` + im PhaseConductor-State |
| > 2.0 | Deutliche Verfälschung | FAIL → Dry/Wet-Anteil erhöhen (per-Phase trocken beimischen), kein Harter Rollback |

**Mapping sone → phon** für Diagnose: `phon = 40 + 33.2 × log10(max(N, 0.001) / 1.0)` (gültig für N ≥ 1 sone)

**Implementierung** (aktiv seit v10.0.0.x):

- Modul: `backend/core/dsp/psychoacoustics.py` (**implementiert und aktiv** — `compute_specific_loudness_zwicker()`, `evaluate_mid_pipeline_loudness_delta()`)
- Aufruf: `backend/core/unified_restorer_v3.py` — ZwickerGuard in `_profiled_phase_call` nach breitbandigen subtraktiven Phasen; Dry/Wet-Rescue: ΔN > 2.0 sone → `_rescue_wet` ∈ [0.35, 0.90]
- Methode: Stationäre ISO 532-1 (not zeitvariant) — ausreichend für Pipeline-Check, Laufzeit ≤ 50 ms / 5-s-Fenster
- Approximation: 24 Butterworth-Bandpass-Filter (Bark-Skala), nicht physikalische Cochlea-Simulation

**Rückwärtskompatibilität**: Ergänzt §2.45a (Gated-RMS-Guard) — LUFS-Check bleibt erhalten als Broadcast-Metrik.
LUFS = Distribution-Standard; Sone = psychoakustische Lästigkeitsmetrik. Beide sind Pflicht.

### §4.1c Loudness-Metrik-Hierarchie (Arbitration bei Konflikt)

Drei Loudness-Metriken sind parallel aktiv. Bei widersprüchlichen Ergebnissen gilt:

| Priorität | Metrik | Anwendung | Triggert |
| --- | --- | --- | --- |
| **1 (höchste)** | **Sone (ISO 532-1)** | Psychoakustische Wahrnehmung | Phase-Rollback / Dry-Wet-Rescue bei ΔN > 2.0 |
| **2** | **LUFS (ITU-R BS.1770-5)** | Broadcast-Normierung, Export-Gate | Makeup-Gain bei Δ > 1.0 LU |
| **3 (niedrigste)** | **Gated-RMS (dBFS)** | Per-Phase-Guard (§2.45a) | Envelope-Aware Gain bei Drift > 3.0 dB |

**Regel**: Wenn Sone-Guard „OK" (ΔN ≤ 0.5) und LUFS-Guard „FAIL" (Δ > 1.0 LU), greift LUFS-Korrektur.
Wenn Sone-Guard „FAIL" (ΔN > 2.0), dominiert Sone ungeachtet des LUFS-Status.
Gated-RMS ist Frühwarnung — triggert nur, wenn BEIDE höheren Metriken schweigen.

---

## §4.2 Verbotene Legacy-Algorithmen als Primärverarbeitung

```python
# ABSOLUT VERBOTEN als Haupt-Algorithmus:
Ephraim & Malah (1984) Wiener-Filter   # → Ersatz: OMLSA/IMCRA
Klassischer Wiener-Filter               # → Ersatz: Consistent Wiener (Le Roux 2013)
Simple Spectral Subtraction             # → Ersatz: MMSE-LSA + OMLSA
Medianfilter-Declicker (primitiv)       # → Ersatz: RBME + iterative Konsistenz
YIN Pitch-Tracker                       # → Ersatz: pYIN / CREPE
LPC Ordnung < 16                        # → Ersatz: High-Order LPC + Burg-Algorithmus
np.fft.rfft ohne PGHI nach Modifikation  # → PGHI zwingend
griffin_lim() als Phasengenerator (Endschritt)  # → PGHI / Vocos / HiFi-GAN (Griffin-Lim randomisiert Phasen → IPD-Verlust → Raumtiefe kollabiert, §8.3.1)
RMS-Normalisierung statt LUFS           # → ITU-R BS.1770-5
Peak-Normalisierung bei Restaurierung   # → LUFS + True-Peak

# VERBOTENE METRIKEN für Musikqualitätsbewertung:
PESQ    # Telefonband 300–3400 Hz
DNSMOS  # 16 kHz Sprachkorpus
NISQA   # Sprach-CNN, keine Musik-Daten
STOI    # Sprachverständlichkeit 150–5000 Hz
ViSQOL --speech  # Voice-Priors → Musik systematisch falsch bewertet
```

---

## §4.3 Frequenzband-Referenzen

```python
BARK_EDGES_HZ = [
    20, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
    1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400,
    5300, 6400, 7700, 9500, 12000, 15500
]
def hz_to_erb(f_hz: float) -> float:
    return 21.4 * math.log10(1.0 + f_hz / 229.0)
def hz_to_mel(f_hz: float) -> float:
    return 2595.0 * math.log10(1.0 + f_hz / 700.0)
```

---

## [RELEASE_MUST] §4.4 SOTA-ML-Entscheidungsmatrix (Stand 2025/2026)

| Anwendungsfall | Primär (SOTA) | DSP-Fallback (Post-2018) | VERBOTEN |
| --- | --- | --- | --- |
| Breitrauschen (Gesang/Vokal) | ML: **DeepFilterNet v3.II** (Formant-/F0-Struktur nutzt Gesangs-Harmonik; energy_bias=−6 dB Pflicht) | OMLSA/IMCRA | ~~Wiener 1984~~ |
| Breitrauschen (rein instrumental, PANNs Vocals < 0.4) | DSP: **OMLSA/IMCRA** (kein Vocal-Prior, musik-neutral) | DeepFilterNet v3.II + energy_bias=−9 dB | ~~Wiener 1984~~ |
| Nicht-stationäres Rauschen | ML: **DeepFilterNet v3.II** | MMSE-LSA + IMCRA | ~~Spectral Subtraction~~ |
| Diffuses Raumrauschen / Dereverb | ML: **SGMSE+** (ONNX) | WPE (nara_wpe) → NumPy-WPE → OMLSA | ~~einfacher Bandpass~~ |
| Stem-Separation Vocals | ML: **MelBandRoformer** (`bs_roformer_plugin`, 860 MB ONNX) | NMF-β | — |
| Stem-Separation Instrumental | ML: **MDX23C** (`mdx23c_plugin`, Kim_Vocal_2/Kim_Inst) | NMF-β | — |
| Bandbreiten-Erweiterung | ML: **FlashSR** | Sinusoidal + Stoch. Modeling | ~~Harmonics-EQ~~ |
| Dropout < 50 ms | DSP: **NMF-β + Sinusoidal** | Consistent Wiener | ~~Yule-Walker AR~~ |
| Dropout 50–999 ms | ML: **CQTdiff+** → DiffWave | Spectral Interpolation | ~~einfaches AR~~ |
| Codec-Artefakte | ML: **Apollo** (Band-Sequence Mamba) | Spectral Repair + PGHI | ~~EQ-Anhebung~~ |
| Pitch-Tracking (mono/Gesang) | ML: **FCPE** → **RMVPE** (Wei et al. ICASSP 2023, −30 % Pitch-Fehler bei Gesang) | PESTO → pYIN | ~~YIN~~, ~~CREPE als Tier-2~~ |
| Polyphoner Pitch | ML: **BasicPitch** | Spektrale Peak-Verfolgung | ~~CREPE mono~~ |
| Instrument-Resonanz | DSP: **DDSP** (Eigenimpl.) | Sinusoidal + Stoch. | ~~fixe Formant-EQ~~ |
| Formanten F1–F4 | ML: **DeepFormants CNN** (ONNX) | LPC (Burg, Ordnung **30–40 bei 48 kHz-SR**, alternativ: Downsampling auf 16 kHz → LPC Ord. 16 → Upsampling) | ~~LPC < 12~~ |
| Neuronale Synthese | ML: **Vocos 48 kHz nativ** ONNX → 44,1 kHz → BigVGAN v2 → HiFi-GAN | PGHI-ISTFT | ~~Griffin-Lim~~ |
| Generatives Inpainting | ML: **Flow Matching** | CQTdiff+ → DiffWave → NMF-β | ~~DDPM 1000 Schritte~~ |
| Audio-Tagging | ML: **BEATs** (iter3) → PANNs CNN14 | DSP Spectral Fingerprint | — |
| MOS (ohne Referenz) | ML: **VERSA** (Huang et al. 2024) → SingMOS (Gesang, PANNs Vocals ≥ 0.3–0.7, Blend-Zone) | PQS-Gammatone-DSP | ~~PESQ/DNSMOS/CDPAM~~ |
| MOS-Verifikation Gesang | ML: **UTMOS** (`utmos_plugin`, ≥18 MB PyTorch) → SingMOS | VERSA → PQS-Gammatone | ~~PESQ/NISQA~~ |
| Music/Vocal Enhancement | ML: **MP-SENet 2023** ONNX | SGMSE+ ONNX → OMLSA DSP | ~~DCCRN/FullSubNet+~~ |
| MOS (mit Referenz) | ML: **ViSQOL v3** (**`--audio` PFLICHT**) | PQS-DSP | ~~--speech Mode~~ |
| Phasen-Rekonstruktion | DSP: **PGHI** | Griffin-Lim ≥ 32 Iter. | ~~Direkte ISTFT~~ |
| Decrackle | ML: **Banquet-Vinyl** (Band-Split/SeqBand ONNX, gemessen 5,7 dB ref_snr — Rev. 2026-08-16) | DSP RBME-artig + iterative Konsistenz + Sparse Bayes | ~~Medianfilter~~ |
| Spektral-Matching | DSP: **Optimal Transport** | Multibänder-EQ | ~~fixe EQ-Kurve~~ |
| Groove / Timing | DSP: **Onset-DTW (madmom RNN)** | Beat-Tracking (librosa) | ~~fixes Raster~~ |
| Stem-Separation (Gesang, Alternativ) | ML: **HTDemucs** (Hybrid Transformer-Demucs v4, 2023) | MelBandRoformer | — |
| Stem-Separation (dichte Mixturen) | ML: **TF-GridNet** (Time-Frequency Grid Net, ICASSP 2023) | HTDemucs | — |
| Musik-NR (spezialisiert) | ML: **AERO** (Richter et al., ICASSP 2024) → **MP-SENet 2023** | OMLSA/IMCRA | ~~DeepFilterNet ohne energy_bias~~ |
| Langes Inpainting / Generativ | ML: **Consistency Models** (Song et al. 2023, < 3 s Latenz) → CQTdiff+ | DiffWave → NMF-β | ~~DDPM 1000 Schritte~~ |
| Codec-Artefakte (Streaming) | ML: **Apollo v2** (Band-Splitting Mamba v2) → Apollo v1 | Spectral Repair + PGHI | ~~EQ-Anhebung~~ |
| Stark degradierter Gesang (SNR < 10 dB) | ML: **SGMSE+ v2** (Score-Based Diffusion, Richter 2022) | DeepFilterNet v3.II + energy_bias=−6 dB | ~~VoiceFixer~~ |
| Latent-Space-Restaurierung / Codec | ML: **DAC** (Descript Audio Codec, Kumar et al. 2023) → EnCodec | CQTdiff+ → NMF-β | — |
| Singer-Identity-Erhalt | ML: **Resemblyzer** (dvector, GE2E-Loss) → X-Vector | DSP Formant-Korrelation | — |
| Vibrato-vs-Flutter-Diskriminierung | DSP: **F0-Autokorrelation** (Vibrato 4–7 Hz; Wow < 2 Hz) + FCPE | pYIN | — |

> ¹ **MIIPHER-DiT** (§v10.14): Flow-Matching-DiT-Modell (806 MB ONNX, OpSet 17). **Open-Source**, ersetzt proprietäres Google MIIPHER. **Task (Rev. 2026-08-15): Codec-Degradation** — Zielmaterialien `{mp3_low, streaming, aac, minidisc}` (Gate `should_apply`); der SNR<10-Gesang-Task wird von SGMSE+ v2 getragen, nicht von der MIIPHER-Stufe. Plugin: `plugins/miipher_dit_plugin.py`.

**HTDemucs / AERO / MIIPHER Auswahllogik (Normativ):**

- **HTDemucs** wird als alternative Vokal-Separation aktiviert wenn: `panns_singing_confidence ≥ 0.5` UND `material_type ∈ {cd_digital, mp3_low, mp3_high, dat}` UND MelBandRoformer SDR < 7 dB auf 30-s-Probe.
- **AERO** (Musik-spezialisiertes NR, ICASSP 2024) wird für `mode="studio_2026"` bevorzugt gegenüber DeepFilterNet wenn `genre_label ∈ {classical, jazz, acoustic}` — diese Genres profitieren von musikalisch-bewusstem NR stärker als Vocal-Prior-basiertem NR.
- **MIIPHER-Stufe** (Rev. 2026-08-15: **Codec-Restaurierer**, nicht SNR<10-Gesang): aktiv bei `material_type ∈ {mp3_low, streaming, aac, minidisc}` und `restorability_score < 30`; Transform in W2v-BERT-Latent-Raum — nur auf Vokal-Stem, nie auf Vollmix. Pflicht-Guard: `hallucination_guard.check_hallucination(pre, post)`. Das proprietäre Original ist nicht gebündelt (Stub → DFN → Wiener); die operative Implementierung ist der offene DiT (`plugins/miipher_dit_plugin.py`). Stark degradierter Gesang (SNR < 10 dB) läuft über SGMSE+ v2 — siehe Tabelle oben.
- **AERO/MIIPHER ONNX-Fallback**: `OMLSA/IMCRA` bei OOM oder Modell-Fehler — beide Modelle haben keinen eigenen DSP-Pass als Primär.

**DAC / Consistency-Models Laufzeitvertrag (Normativ):**

- **DAC** (Descript Audio Codec) wird ausschließlich für Latent-Space-Restaurierung eingesetzt (Phase 55 Inpainting-Extension, ≥ 200 ms Lücken) — kein generelles Encoding/Decoding des Gesamtsignals (Qualitätsverlust durch DAC-Kompression).
- **Consistency Models** ersetzen CQTdiff+ bei Laufzeitbudget < 30 s verbleibend in UV3-Wall-Time-Guard. Qualitäts-Vorrang: CQTdiff+ wenn Zeit vorhanden (bessere SSDR); Consistency wenn Zeit knapp.

**MP-SENet ONNX Laufzeitvertrag (Release-Must):**

- Eingabeformat bleibt `noisy_amp/noisy_pha: [batch, 201, time]`.
- Das aktuell gebündelte ONNX-Artefakt zeigt trotz `dynamic_axes` einen faktisch festen Zeit-Shape (`time=32`) in internen Reshape-Knoten.
- Implementierungspflicht: segmentierte Inferenz in festen 32-Frame-Chunks mit Stitching zurück auf Originallänge.
- Bei Reshape-/Layout-Fehlern: kontrollierter Fallback (Layout-Retry, dann OMLSA-DSP), niemals Hard-Crash.

---

## §4.4a [RELEASE_MUST] SOTA-Evaluations-Protokoll (v10.0.0)

**Problem**: Die SOTA-Matrix (§4.4) veraltete unbemerkt. Modelle aus 2023/2024 wurden nicht systematisch evaluiert; der Stand „2025/2026" war nicht durch einen definierten Prozess gedeckt.

**Evaluationszyklus**: Quartalsweise (Januar/April/Juli/Oktober).

**Aufnahmekriterien für neue Primärmodelle** (alle müssen erfüllt sein):

| Kriterium | Mindestanforderung |
| --- | --- |
| Qualitätsvorteil | SDR/MOS/OQS > aktuelles Primärmodell um ≥ 0.5 dB / 0.1 MOS |
| Offline-Fähigkeit | Vollständig offline, keine Cloud-API, kein Internet-Zugriff zur Laufzeit |
| ONNX-Export oder PyTorch | Stables ONNX oder PyTorch `torch.jit.trace`-Export vorhanden |
| Lizenzkompabilität | MIT / Apache 2.0 / CC BY 4.0 — kein CC NC, kein GPL ohne Ausnahme |
| Laufzeit-Budget | ≤ 60 s / Minute Audio auf Tier-2-GPU (8–15 GB VRAM) |
| DSP-Fallback | Fallback-Kette aus vorhandenem Bestand definierbar |

**Evaluations-Pipeline (Pflicht)**:

```python
# benchmarks/sota_eval.py
def evaluate_candidate(model_name, candidate_plugin, amrb_subset="all"):
    results = run_benchmark(BenchmarkConfig(
        n_items=10, min_duration_s=30,
        override_primary={use_case: candidate_plugin},
    ))
    return {
        "oqs_delta": results.overall_score - SOTA_BASELINE[use_case],
        "latency_ms": results.avg_latency_ms,
        "passes_criteria": results.overall_score > SOTA_BASELINE[use_case] + 0.5,
    }
```

**Dokumentationspflicht**: Jede Änderung an §4.4 MUSS in `docs/CHANGELOG_HISTORY.md` unter `[SOTA-Update v9.x.y]` dokumentiert werden mit: Modellname, Vorgänger, AMRB-Delta, Evaluationsdatum.

**VERBOTEN**: Manuelle Matrix-Aktualisierung ohne durchlaufenen Evaluations-Check.

## §4.4b [RELEASE_MUST] Phase-Bindung der SOTA-Matrix (v10.0.0)

Die Tabellen in §4.4 definieren die familienweite Modell-Praeferenz. Die **bindende Zuordnung
zu den einzelnen Phasen 01-64** steht in Spec 06 §7.1d.

**Konfliktregel:**

1. Spec 06 §7.1d ist autoritativ fuer die exakte Phasenbindung.
2. Spec 04 §4.4 bleibt autoritativ fuer familienweite Modellrangfolgen und Ausschlusslisten.
3. Bei Konflikt gewinnt die strengere, artefaktaermere Kombination.

**VERBOTEN:** pro Phase eigenmaechtig ein anderes Primaermodell zu setzen, ohne die Bindung in
Spec 06 §7.1d und das Evaluationsprotokoll §4.4a gemeinsam zu aktualisieren.

---

## §4.5 Pflicht-Algorithmus-Spezifikationen

### §4.5g [RELEASE_MUST] WLPC — Noise-Robuste Formant-Extraktion (v10.0.0)

**Problem**: Standard-Burg-LPC erkennt Rauschspitzen als Formanten wenn SNR < 15 dB oder
era < 1960 (Shellac, frühe elektrische Aufnahmen). Das erzeugt falsche Formant-Korrekturen
und kann die Stimme verfärben.

**Lösung — Wiener-gain Spectral Pre-Whitening** (implementiert in `lpc_formant_tracker.py`):

```python
# WLPC-Pfad: aktiv wenn era_decade < 1960 ODER effective_snr < 15 dB
# Pre-Whitening NUR für LPC-Koeffizienten-Schätzung — nie auf Output-Audio anwenden!

def _wlpc_prewhiten_frame(frame: np.ndarray, noise_psd: np.ndarray) -> np.ndarray:
    """Wiener-Gain Spektral-Pre-Whitening für rauschrobuste LPC-Schätzung."""
    fft_frame = np.fft.rfft(frame)
    signal_psd = np.abs(fft_frame) ** 2
    gain = np.maximum(1.0 - noise_psd / (signal_psd + 1e-10), _WLPC_GAIN_FLOOR)
    return np.fft.irfft(fft_frame * gain, n=len(frame))

# Noise-PSD aus den ruhigsten 20 % der Frames (Perzentil-Filter):
quiet_frames = [f for f in frames if rms(f) < np.percentile(rms_list, 20)]
noise_psd = np.mean([np.abs(np.fft.rfft(f))**2 for f in quiet_frames], axis=0)

# SNR-Schätzung: 75th/10th-Perzentil-Verhältnis der Frame-RMS-Werte
snr_db = 20 * np.log10(np.percentile(rms_list, 75) / (np.percentile(rms_list, 10) + 1e-10))
```

**Wichtige Invariante**: Pre-Whitening wird AUSSCHLIESSLICH für die LPC-Polynomial-Schätzung
verwendet. Das Output-Audio wird nicht modifiziert (§0 Primum non nocere).

**API** (`backend/core/dsp/lpc_formant_tracker.py`):

```python
# lpc_formant_enhance() und _LPCFormantTracker.enhance() — neue Parameter:
def enhance(audio, sr, era_decade: int | None = None, snr_hint_db: float | None = None):
    ...
    # WLPC-Pfad aktiv wenn:
    # - era_decade is not None and era_decade < _WLPC_ERA_THRESHOLD (1960)
    # - ODER effective_snr < _WLPC_SNR_THRESHOLD_DB (15.0 dB)
```

**VERBOTEN**: `_get_lfc().enhance(audio, sr)` ohne `era_decade` wenn der Aufrufkontext
`era_decade < 1960` kennt — Standard-Burg-LPC würde Rausch-Formanten erkennen und korrigieren.

**Pflicht-Aufrufer mit era_decade**:

- `phase_42_vocal_enhancement.py`: `_get_lfc().enhance(audio, sr, era_decade=int(_era_decade) if _era_decade else None)`
- `phase_65_vocal_naturalness_restoration.py`: `era_decade` aus `kwargs`/`_restoration_context`

### Rauschunterdrückung (Phase 03, 29)

```text
Pflicht: OMLSA (Cohen & Berdugo 2002) + IMCRA-Variante (Cohen 2003)
Gain-Glättung: MMSE-LSA
G_floor: 0.85 an HPG-protected_bins, 0.10 sonst

Stem-bewusstes NR-Routing (PANNs-konditioniert, nach TDP-Trennung):

  A) Perkussiver Stem (TDP audio_percussive):
     → KEIN NR, KEIN DeepFilterNet — nur phase_01 + phase_27 (Klick/Pop)
     Begründung: Transienten-Angriffe werden durch NR-Latenz vernichtet

  B) Harmonischer Stem MIT Gesang (PANNs Singing ≥ 0.4):
     PRIMÄR: DeepFilterNet v3.II (Formant-/F0-Struktur von Gesang nutzt
              die Harmonik-Modellierung des Netzes; NICHT weil Gesang = Sprache)
     Musik-Konfiguration:
         energy_bias = −6.0 dB (Pflicht — reduziert aggressive NR in Harmonik-Regionen)
         G_floor via HarmonicPreservationGuard (0.85 an Partial-Bins)
     FALLBACK: OMLSA/IMCRA + MMSE-LSA

  C) Harmonischer Stem REIN INSTRUMENTAL (PANNs Singing < 0.4):
     PRIMÄR: OMLSA/IMCRA (kein Vocal-Prior — musik-neutral, adaptiv)
     Begründung: DeepFilterNet würde harmonische Obertonstrukturen
                  (Streicher, Bläser, Chords) nach Vocal-Prior
                  systematisch als Rauschen fehlbewerten
     SEKUNDÄR: DeepFilterNet v3.II mit vergrößertem energy_bias = −9.0 dB
               (nur wenn OMLSA SNR-Schätzung unsicher: IMCRA-SNR < 5 dB)

Kritische Architekturnotiz (Stand April 2026):
    Es existiert kein öffentlich verfügbares, auf Musikdaten (Gesang + Instrumente)
    trainiertes neuronales Denoise-Modell in DeepFilterNet-Qualitätsstufe.
    OMLSA/IMCRA ist daher für rein instrumentales Material de-facto co-primär,
    nicht nur Fallback. DeepFilterNet wird NUR bei erkanntem Gesang (PANNs Singing
    ≥ 0.4) als Primär eingesetzt — NICHT für gesprochene Sprache (Aurik restauriert
    Musik, keine Podcasts/Hörbücher). Diese Einschätzung ist bis zum Erscheinen
    eines musik-spezifischen ML-Denoiser-Modells verbindlich.
```

### [RELEASE_MUST] Inpainting (Phase 24, 55)

```text
Kurze Lücken < 50 ms: NMF mit β-Divergenz (β=0, Itakura-Saito) + PGHI
    β-Wert-Referenz (normativ, Lücke-F-Fix v10.0.0):
        β=0 → Itakura-Saito-Divergenz (IS): minimiert relative Energiefehler,
               gewichtet kleine Energieunterschiede stark → OPTIMAL für impulsive
               Lücken (Klicks < 50 ms) und Transienten (Attack-Bins).
        β=1 → Kullback-Leibler-Divergenz (KL): bessere Approximation bei harmonischen,
               tonal-stationären Lücken — NICHT für impulsive Transienten verwenden.
        β=2 → Euklidisch: nur für breitbandige stationäre Rekonstruktion.
    Transient-Lücken (< 50 ms, Onset-Kontext): β=0 (IS) — PFLICHT (Artikel-/MicroDyn-Schutz)
    Harmonische/tonale Lücken (< 50 ms, Stationär-Kontext): β=1 (KL) erlaubt.
    VERBOTEN: β=1 für kurze Transient-Lücken (zerstört Attack-Energie-Verteilung).
Lange Lücken 50–999 ms: CQTdiff+ (Moliner & Välimäki, ICASSP 2023)
    - CQT-Domänen-Diffusion konditioniert auf Phrasen-Kontext ±30 s
    - PGHI für phasenkonsistente Rücktransformation
    Fallback: CQTdiff+ → DiffWave → NMF-β (β=0)
VERBOTEN: VoiceFixer v2 (Sprach-only, VCTK-Korpus)
VERBOTEN: VampNet (kein gebündeltes Plugin, kein stabiler ONNX-Export)
```

### Codec-Artefakte (Phase 23, 50)

```text
Pflicht: Apollo (Zhang et al. 2024)
    - Band-Splitting-RNN: Audio → 24 Sub-Bänder
    - Mamba-Backbone-Sequenzmodellierung
    - Musical Goals Check post-Rekonstruktion (Brillanz ≥ 0.85, Wärme ≥ 0.80)
Fallback: Resemble-Enhance ONNX → DSP Spectral Repair + PGHI
```

### Amplituden-Clipping (Phase 23 — CLIPPING-Typ) §4.5a [RELEASE_MUST]

```text
Diskriminierung: classify_clipping() MUSS VOR Reparatur ausgeführt werden.
    SOFT_SATURATION → Phase 23 ÜBERSPRINGEN (§5 Vintage-Guard)
    CLIPPING         → ADMM-Declipping

Primär: ADMM-basierte Sparse-Recovery (Záviška et al., EUSIPCO 2021)
    Modell: geclippte Abtastwerte y_c = clip(x, -A, +A)
    Minimierungsproblem:
        min_x  ||x||_1   s.t.   x_free = y_free   (ungesättigte Stellen exakt)
                                |x_clip| >= A       (gesättigte Stellen ≥ Clipwert)
    ADMM-Parameter:
        rho     = 0.1        (Penalty, adaptiv: rho *= 1.5 wenn Residuum > 10× Dual)
        max_iter = längenadaptiv [RELEASE_MUST]:
            max_iter = clamp(round(200 × min(1.0, 30.0 / duration_s)), 30, 200)
            → Kurze Signale (≤ 30 s): 200 Iterationen (volle Qualität)
            → Lange Signale (z. B. 225 s): 30 Iterationen (~360 s statt 2460 s)
            Begründung: Záviška 2021 zeigt Konvergenz typisch in 30–50 Iterationen;
            Iterationen 60–200 bringen Sub-Promille-Verbesserungen, blockieren aber
            den UV3 Wall-Time-Budget für alle nachfolgenden Enhancement-Phasen.
            VERBOTEN: festes max_iter=200 unabhängig von Signallänge.
        tol     = 1e-4       (Primal- und Duales Residuum unter Schwelle — bricht früh ab)
        Wall-Time-Guard: min(180 s, 1.5 × duration_s) als absoluter Timeout (Sicherheitsnetz)
    Wavelet-Prior (Daubechies db4, Level 5) als Sparsifying-Transform
    Frequenzband: 20 Hz – 20 kHz (keine Sub-Bass-Beschränkung)
    Transient-Guard: Attack-Bins (onset ±5 ms) erhalten rho × 3.0 → stärkerer Erhalt

Fallback: Consistent Wiener Filter mit Clip-Constraints
    → Spectral Repair + PGHI falls beide fehlschlagen

INVARIANTE: Nach ADMM-Declipping MUSS Transient-Shape-Korrelation
    mit pre-clipping Signal ≥ 0.88 (ArticulationMetric-Grenzwert).
    Unterschreitung → Safety-Blend mit Dry-Signal (max 30 % Wet).

VERBOTEN: Apollo für Amplituden-Clipping (Apollo = Codec-Artefakte-Training).
    Apollo NUR für digitale Codec-Komprimierungsartefakte (mp3/aac/ogg-Distortion).
```

### [RELEASE_MUST] Print-Through-Reduktion (Phase 29, reel_tape)

```text
Pflicht: Bidirektionale Adaptive Temporal Subtraction (LMS-basiert, Widrow & Stearns 1985)
Physikalisches Modell: Print-Through entsteht auf BEIDEN Seiten der Masterwicklung
    (Kopie VOR dem Original = Pre-Echo bei Vorwärtswicklung,
     Kopie NACH dem Original = Post-Echo bei Rückwärtswicklung),
     Amplitude nimmt nichtlinear mit Wicklungsabstand ab (α_pre ≠ α_post).
Schritte:
    1. Kreuzkorrelation-Peak ±600 ms → delay_pre, delay_post (beide Seiten)
    2. LMS-Adaptivfilter separat für Pre- und Post-Echo:
       alpha_pre  ∈ [0.03, 0.25]  (schwächeres Prä-Echo)
       alpha_post ∈ [0.05, 0.35]  (stärkeres Post-Echo)
    3. audio_clean[t] = audio[t]
                        − alpha_pre  · audio[t + delay_pre]
                        − alpha_post · audio[t − delay_post]
    4. Spectral Coherence vor/nach ≥ 0.90 + PGHI
Fallback: NMF-β Dekomposition (einseitig, nur Post-Echo)
VERBOTEN: Comb-Filter, einseitiges α-Modell als Pflicht-Implementierung
```

### Multi-Resolution STFT (Phase 03, 06, 07, 23, 50)

```python
# MRSA-Fenster @ SR=48000 Hz:
ZONES = {
    "sub_bass":   {"win": 65536, "hop": 16384, "hz": (20, 250)},
    "mid_low":    {"win": 16384, "hop": 4096,  "hz": (250, 800)},
    "mid":        {"win": 8192,  "hop": 2048,  "hz": (800, 2000)},
    "presence":   {"win": 1024,  "hop": 256,   "hz": (2000, 8000)},
    "air":        {"win": 128,   "hop": 32,    "hz": (8000, 24000)},
}
# PGHI per Zone; Kreuzfade Hanning 10 ms an Zonenübergängen
```

### [RELEASE_MUST] Neuronale Synthese / Vocoder-Kaskade (wenn PQS-MOS < 4.3)

```text
4-stufige Fallback-Kaskade (Studio-2026):
    1. Vocos 48 kHz nativ — models/vocos_48khz/vocos_48khz.onnx (CPUExecutionProvider)
       Mel-Bins 128; True-Peak −1.0 dBTP nach Synthese
       # VERBOTEN: vocos_mel_spec_24khz.onnx (24-kHz-Variante — SR-Mismatch zu processing_sr=48000)
    2. BigVGAN-v2 — bigvgan_v2 (0,4 GB, ONNX/PyTorch, GPU-beschleunigt via ml_device_manager)
    3. HiFi-GAN (3,6 MB ONNX) — Tertiär-Fallback
    4. PGHI-ISTFT — DSP-Endfall-Fallback
VERBOTEN: Griffin-Lim als Endschritt in Studio-2026

```

### Pit-Korrektur (Phase 12, 31)

```text
Primär: FCPE → RMVPE (Wei et al. ICASSP 2023) + DTW
Bei Gesang (PANNs Vocals ≥ 0.4):
    PSOLA (Moulines & Charpentier 1990) — formanterhaltend bei Transposition > ±2 Halbton
    Phase-Vocoder: nur für perkussive / nicht-vokale Segmente (HPSS-detektiert)
DSP-Fallback: PESTO (Riou et al. ISMIR 2023) → pYIN (Mauch & Dixon 2014)
```

### Klavier-Inharmonizität §4.5b (Phase 52) [RELEASE_MUST]

```text
Physikalisches Modell (Fletcher 1964):
    f_n = n · f_0 · sqrt(1 + B · n²)
    B = Inharmonizitätskoeffizient (abhängig von Saitenlänge, Durchmesser, Material)

B-Schätzung per Oktavlage (aus MIDI-Pitch oder f₀-Track):
    MIDI 21–35  (Kontra-Oktave,  21–62 Hz):  B ∈ [0.00005, 0.0003]
    MIDI 36–47  (Groß-Oktave,    65–247 Hz): B ∈ [0.0003,  0.001]
    MIDI 48–71  (Klein/1-Oktave, 130–987 Hz):B ∈ [0.001,   0.003]
    MIDI 72–108 (2–5-Oktave,     1047–4186 Hz):B ∈ [0.002, 0.008]

Schätzung (wenn kein MIDI-Input):
    1. f₀ per FCPE → RMVPE → nächster MIDI-Pitch
    2. Partials f_2, f_3, f_4 aus Spektrum detektieren
    3. B = mean([(f_n / (n · f_0))² - 1] / n²)  für n = 2..4
    4. B im zulässigen Bereich für die Oktavlage klemmen (s. o.)

Anwendung in phase_52:
    - Partial-Lock: Obertöne auf Fletcher-Raster statt harmonisches int-Vielfaches
    - Resynthese (DDSP additive): f_n = n · f_0 · sqrt(1 + B · n²)
    - VERBOTEN: Partial-Locking auf exakte ganzzahlige Vielfache bei Klaviermaterial
      (ergibt zu hartes/synthetisches Ergebnis)

INVARIANTE: TimbralAuthenticityMetric ≥ 0.87 nach phase_52.
    B-Schätz-Fehler > 15 % → DSP-Fallback (sinusoidales Partial-Tracking ohne Lock).
```

### Dereverb — Early-Reflection-Guard §4.5c (Phase 20, 49) [RELEASE_MUST]

```text
Physikalische Grundlage:
    Schallfeld = Direktschall + frühe Reflexionen (0–80 ms) + diffuser Nachhall (> 80 ms)
    Frühe Reflexionen (0–50 ms): definieren Raumcharakter, Wärme, Quellenlokalisierung
    (Haas-Effekt / Precedence Effect — Blauert 1997)
    Diffuser Nachhall (> 80 ms): maskierend, verschleiert Transienten, Artikulation

Pflicht-Algorithmus (SGMSE+ und WPE):
    1. RIR-Schätzung (Blind): Kreuzkorrelations-Dekonvolution oder
       ML-RT60-Schätzer (DPRNN-Head in SGMSE+ oder spectral-based RT60)
    2. Early-Reflection-Guard: C80 und D50 VOR und NACH Dereverb messen
           C80 = 10 · log10(∫₀⁸⁰ₘₛ h²(t)dt / ∫₈₀ₘₛ^∞ h²(t)dt)   [Klarheitsmaß, Musik]
           D50 = ∫₀⁵⁰ₘₛ h²(t)dt / ∫₀^∞ h²(t)dt                    [Deutlichkeitsmaß — sekundär; C80 hat Vorrang für Musik]
    3. Wet-Mix-Begrenzung: Dereverb-Intensität MUSS begrenzt werden, wenn:
           ΔC80 = C80_post − C80_pre > 6 dB  → Wet weiter reduzieren
           ΔD50 > 0.12                         → Wet weiter reduzieren
       Ziel: ΔC80 ≤ 6 dB, ΔD50 ≤ 0.12 (kein Raumcharakter-Verlust)
    4. Early-Reflection-Blend: Residual der ersten 50 ms aus Originalschall
       (vor Dereverb) mit alpha = 0.35 zurückzumischen falls ΔC80 > 4 dB:
           audio_out[t] = audio_dereverb[t] + 0.35 · audio_early[t]  (t ∈ [0, 50ms] jedes Onset)
    5. Safety-Revert: Wenn C80_post < C80_pre − 2 dB → vollständiger Rollback

Messung (einfache Approximation ohne RIR-Inversion):
    RT60-Proxy: Energie-Abfall von −5 dBFS auf −35 dBFS in Stille-Segmenten (Silero VAD)
    C80-Proxy: Energie-Ratio 80 ms / Rest in Onset-Fenstern (madmom Onset-Detektion)

INVARIANTE: NatuerlichkeitMetric darf nach phase_49 nicht sinken
    (Early-Reflection-Blend stellt Raumluft wieder her).
VERBOTEN: Aggressives Dereverb ohne Early-Reflection-Guard (klingt klinisch/steril).
```

### §4.5d [RELEASE_MUST] Psychoakustischer Masking-Guard für NR-Algorithmen (v10.0.0)

Ergänzt §4.1 (ISO 11172-3) mit einer **bindenden Gain-Floor-Invariante** für alle NR-Algorithmen.

**Grundprinzip (ISO 11172-3 Psychoacoustic Model 2)**:
Rauschen, das vom Musik-Signal maskiert wird, ist für das Gehör nicht wahrnehmbar.
Aggressives Entfernen dieser maskierten Komponenten erzeugt hörbare Stille-Artefakte
(Over-NR: tote Stille, klinisches Klangbild zwischen Phrasen).

**Bindende Invariante**:

```python
# backend/core/dsp/psychoacoustics.py
def compute_masking_threshold_iso11172(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
) -> np.ndarray:
    """
    ISO 11172-3 Psychoacoustic Model 2 (approximiert).
    Returns masking threshold per FFT bin in linear power scale.
    """
    # ... Bark-Filterbank, Spreading Function, Absolute Threshold of Hearing
    ...

# Aufruf VOR NR (DeepFilterNet, OMLSA, SGMSE+):
masking_threshold = compute_masking_threshold_iso11172(pre_nr_audio, sr)

# NR-Gain-Floor pro Frequenzband:
G_floor[band] = max(0.10, masking_threshold[band] / noise_estimate[band])

# VERBOTEN — kein NR-Gate unter G_floor in Bins mit aktiver Musik-Energie:
assert G_floor[band] >= 0.10 for band where signal_energy_db[band] > -60.0
```

**Auswirkung**: Erhält den lebendigen Rauschboden, der Musik „atmen" lässt — kein steriles, klinisches Klangbild. Rauschen unterhalb der Maskierungsschwelle bleibt erhalten, da es vom Gehör nicht als störend wahrgenommen wird.

**Integration**:

- DeepFilterNet-Wrapper: `energy_bias` bereits −6 dB (vocal) / −9 dB (instrumental) für Spec 04 §0j konform
- OMLSA: `noise_floor_softening_factor = G_floor` pro Bark-Band
- SGMSE+: `sigma_scale` per-frame so begrenzt, dass effektiver Gain-Floor ≥ 0.10

> Kreuzreferenz: §4.1 (Psychoacoustic Masking), §4.1b (ISO 532-1 Loudness), §4.5 NR-Routing; Spec 02 §2.62

### §4.5e [RELEASE_MUST] Hallucination-Guard-Referenz für additive DSP-Phasen (v10.0.0)

Jede additive DSP-Phase in Spec 04 MUSS `hallucination_guard.py` (`§2.46e`) aufrufen.
Insbesondere betroffen: Bandbreiten-Extension (phase_06), Harmonik-Synthese (phase_07), FlashSR (phase_23).

Vollständige Spec: **Spec 02 §2.46e** (Hallucination-Guard).

> Kreuzreferenz: Spec 02 §2.46e; `backend/core/dsp/hallucination_guard.py`

### [RELEASE_MUST] Phasen-Rekonstruktion (nach JEDER Spektral-Modifikation)

```text
PFLICHT: PGHI (Perraudin et al. 2013)
Fallback: Griffin-Lim ≥ 32 Iterationen
ABSOLUT VERBOTEN: Direkte ISTFT auf modifiziertem Betragsspektrum
```

**Ausnahmevertrag (eng begrenzt, RELEASE_MUST) — `phase_12_wow_flutter_fix` / Tape-Head-Level-v2:**

Direkte STFT→iSTFT-Rekonstruktion ist ausschließlich für den
`TAPE_HEAD_LEVEL_DIP`-Unterpfad in `phase_12` zulässig, wenn alle Bedingungen erfüllt sind:

1. Ziel ist frequenzabhängige Kopfkontakt-/Spacing-Loss-Kompensation (kein generatives Inpainting).
2. Linked-Stereo-Invariante: identische spektrale Gain-Maske für L und R.
3. Boundary-konsistent: SciPy-kompatibler Boundary-Modus (`even|odd|constant|zeros|None`),
   identisch zwischen STFT und iSTFT-Pfad.
4. Länge exakt konserviert (`len(out) == len(in)` via trim/pad).
5. SNR-Guard aktiv (Noise-Floor-nahe Bins nicht boosten) + `max_gain_db <= 15`.

**Verboten bleibt** direkte ISTFT für alle anderen spektralen Rekonstruktionsphasen.

**PGHI-Compliance-Status (Stand April 2026):**
Phasen mit PGHI-Rekonstruktion: `phase_03`, `phase_06`, `phase_20`, `phase_23`, `phase_24`, `phase_28`, `phase_29`, `phase_31`.
Alle anderen STFT-nutzenden Phasen verwenden kein modifiziertes Betragsspektrum (nur Analyse-STFT ohne Rücktransformation).

**[RELEASE_MUST] PGHI STFT/iSTFT-boundary-Invariante (v10.0.0):**
Jeder PGHI-Rekonstruktionsaufruf nach einer STFT-Modifikation MUSS `boundary`-konsistent sein:

```python
# PFLICHT-Regel: Externes STFT mit scipy-Standard (boundary='zeros') MUSS
# mit iSTFT boundary=True rekonstruiert werden.
# Internes STFT (PGHI-intern mit boundary=None) nutzt boundary=False in iSTFT.
#
# FALSCH (Amplitude am Rand ~1–3 dB zu niedrig, erste/letzte ~10 ms abgedämpft):
audio_out = pghi_reconstruct_from_stft(Zxx)  # ohne n_samples
# RICHTIG:
audio_out = pghi_reconstruct_from_stft(Zxx, n_samples=len(audio_in))
#   n_samples=: Exakte Längetreue durch Trim/Pad auf Originallänge.
#   dsp/pghi.py::_istft() erhält boundary=True wenn STFT extern (scipy-Standard)
#   und boundary=False nur für intern berechnete STFT-Frames.
#
# VERBOTEN: mode='edge' beim Padding von iSTFT-Kurzausgaben — klont gedämpften Endwert.
# PFLICHT: mode='constant', constant_values=0.0 (Stille statt Artefakt-Klon).
```

### [RELEASE_MUST] Dithering (24→16 bit Export)

```text
PRIMÄR: POW-r Typ 3 (Wannamaker et al. 1992) — ~+6 dB effektiver SNR
FALLBACK: TPDF-Dithering (±1 LSB)
VERBOTEN: Truncation ohne Dithering
```

### [RELEASE_MUST] MP3-Export

```text
PRIMÄR: LAME VBR V0 via FFmpeg (q:a=0) — adaptiv bis 320 kbps, ≈245 kbps Ø
VERBOTEN: CBR für Restaurierungsausgaben — CBR erzeugt Pre-Echo auf restaurierten
          Transienten (TDP §7.x, MDEM §8.3)
```

### Head-Bump Kompensation (Phase 04, Tape-Material) v10.0.0

Tape-Transport-Köpfe erzeugen eine LF-Resonanz, deren Mittenfrequenz umgekehrt proportional zur Aufnahmegeschwindigkeit ist (physik: Wellenlänge = Spaltlänge · 2 bei Resonanzfrequenz).

```text
KOMPENSATION: Parametrische Kerbfilter (IIR-Peaking, gain_db negativ) pro Bandgeschwindigkeit:
  1.875 ips (Kassette):  f_bump = 70 Hz,  cut = −2.5 dB, Q = 1.2
  3.75  ips:             f_bump = 90 Hz,  cut = −2.5 dB, Q = 1.2
  7.5   ips (Standard-Reel):  f_bump = 130 Hz, cut = −2.0 dB, Q = 1.3
  15    ips (Semi-Pro):  f_bump = 180 Hz, cut = −1.5 dB, Q = 1.4
  30    ips (Master):    f_bump = 250 Hz, cut = −1.0 dB, Q = 1.5

QUELLEN: Zar (1989) "Magnetic Recording Handbook"; Jorgensen (1996) "The Complete Handbook
         of Magnetic Recording"; White (2000) "The Recording Studio Handbook".
AKTIVIERUNG: Via kwargs["tape_speed_ips"] in phase_04.process() — optional, nur wenn Bandgeschwindigkeit bekannt.
FALLBACK: Keine Kompensation (kein crash); unbekannte Geschwindigkeit (>150% Abweichung) = bypass.
```

### Intermodulation Distortion — Bispektrum-Analyse (Phase 63) v10.0.0

Bispektrum-Kohärenz bestätigt IMD-Produkte kausal (Wishart 2013; Kim & Powers 1979):

```text
B(f1, f2) = E[X(f1) · X(f2) · conj(X(f1+f2))]

Normierte Bispektrum-Kohärenz:
  b_coherence = |B(f1,f2)| / (P(f1) · P(f2) · P(f1+f2))^(1/3)

Schwellwert: b_coherence > 0.15 → IMD-Produkt bestätigt (vs. Rauschen/Harmonik)

§2.51 STEREO-COMPLIANCE: Notch-Maske wird aus dem Mid-Signal berechnet und
symmetrisch auf Mid (100%) und Side (30%) angewendet. Kein unabhängiges L/R-Processing.
```

### Tonträgerketten-Charakteristika — BW-Grenzen (Phase 55) v10.0.0

```text
MATERIAL-SPEZIFISCHE INPAINTING-BANDBREITENBEGRENZUNG (§0 Primum non nocere):

  wax_cylinder  : ≤ 5000 Hz   (akustische Aufnahme, Horn-Resonator)
  wire_recording: ≤ 6000 Hz   (Stahldrähte 1940–1955, mechanische BW-Begrenzung)
  lacquer_disc  : ≤ 8000 Hz   (Schnitt-Spezifikation frühe Direktschnittplatten)

IMPLEMENTATION: Butterworth 4th-Order Tiefpass (sosfiltfilt, zero-phase) nach
  jedem Gap-Fill in _process_channel() — verhindert AR/Diffusion-Halluzination
  von HF-Inhalt, der im Quellmaterial nie vorhanden war.
```

---

## §4.7 [RELEASE_MUST] Noise-Texture-Coherence-Guard (v10.0.0)

**Problem**: Denoising-Phasen (phase_03, phase_29) können die spektrale Form des Restrauschens verändern. Ein Vinyl-Song, der nach Denoising weißen Rauschboden zeigt, klingt „falsch" — auch wenn der Rauschpegel niedrig ist. Die Rauschtextur ist Teil des Trägerprofils und muss kohärent bleiben (§0a).

**Trägerprofil-Referenz (normativ)**:

| Material | Rausch-Spektral-Profil | Slope (dB/oct) | Begründung |
| --- | --- | --- | --- |
| `vinyl` | Rosa/Pink | ≈ −3 | Rillengeometrie, Nadelkontakt, Motorgeräusch |
| `shellac` | Rosa mit HF-Plateau | ≈ −2.5 | Breiteres Rauschband, gröbere Rillen |
| `tape`, `reel_tape` | Brown + HF-Hiss-Buckel | ≈ −4.5 (LF) + Hiss-Peak 4–8 kHz | Magnetbandkorn + Bias-Rauschen |
| `cassette` | Brown + stärkerer HF-Hiss | ≈ −5 (LF) + Hiss-Peak 6–12 kHz | NR-Restfehler, schmaleres Band |
| `wax_cylinder` | Breitband, Pink-ähnlich | ≈ −2 | Mechanisches Rauschen, Horn-Resonanz |
| `wire_recording` | White-ish + Clicks | ≈ −1 | Drahtlauf, mechanisch |
| `cd_digital` | Weiß / Flat | ≈ 0 | Quantisierungsrauschen |
| `mp3_low`, `mp3_high` | Codec-geformt | variabel | MDCT-Quantisierung |
| `streaming`, `aac` | Codec-geformt | variabel | MDCT-Quantisierung |

**Kohärenz-Messung**:

```python
def compute_noise_texture_coherence(
    residual_noise: np.ndarray,  # restored_audio - original_dry_estimate
    material_type: str,
    sr: int = 48000,
) -> float:
    """
    Misst spektrale Kohärenz zwischen Restrauschen und materialspezifischem Trägerprofil.

    1. PSD des Restrauschens (Welch, 4096 Samples, Hann)
    2. Log-PSD normieren (Median = 0 dB)
    3. Referenz-PSD aus _PROFILE_SLOPES[material_type] generieren
    4. Pearson-Korrelation(log_residual_psd, log_reference_psd)

    Returns: Kohärenz-Score [0, 1]; ≥ 0.80 = kohärent
    """
```

**Guard-Integration**:

- **Per-Phase**: Nach jeder subtraktiven Phase (phase_03, phase_29, phase_28) messen.
  Kohärenz < 0.60 → Strength-Reduktion (−30 % Wet, `wet_mult = 0.70`), kein Rollback.
  Kohärenz ∈ [0.60, 0.80) → milde Reduktion (−15 % Wet, `wet_mult = 0.85`) **und** Telemetrie-Warning.
  Kohärenz ≥ 0.80 → OK (`wet_mult = 1.0`).
  **Rationale**: Kohärenz 0.60–0.80 ist für sensible Hörer bereits hörbar inkohärent (Vinyl klingt „digital-flach"). Eine milde Wet-Dämpfung erzwingt mehr Retention des Carrier-Profils ohne die subtraktive Wirkung zu neutralisieren. `wet_mult = 1.0` bei 0.60–0.80 ist ein normatives Defizit (v10.0.0).
- **End-of-Pipeline**: `metadata["noise_texture_coherence"]` als Pflichtfeld.
  Restoration-Modus: Kohärenz < 0.80 → Warning im Export, Empfehlung in `auto_improvement_recommendations`.
  Studio 2026: Kein Kohärenz-Zwang (Ziel ist minimaler Rauschboden).

**Invariante**: Der Guard darf Denoising nicht wirkungslos machen — er begrenzt die Texturveränderung, nicht die Rauschreduktion an sich.

**Implementierungspfad**: `backend/core/noise_texture_coherence.py` — `NoiseTextureCoherenceGuard`; Aufruf in UV3 nach subtraktiven Phasen.

---

## §4.8 [RELEASE_MUST] Generation-Loss-Kompensationsformel (v10.0.0)

**Problem**: Mehrgenerations-Übertragungsketten (z. B. Shellac → Reel-Tape → Cassette → CD → MP3) verursachen kumulative, physikalisch modellierbare Verluste. Ohne explizite Formel schätzt jede Komponente den Gesamtverlust anders, was zu inkonsistenter Restaurierungsstärke führt.

**Normative Formel** (input: `transfer_chain` aus MediumDetector):

```python
# Konsolidierte Carrier-Transfer-Charakteristik-Tabelle (normativ)
CARRIER_TRANSFER_CHARACTERISTICS = {
    # material_key: (bw_ceiling_hz, snr_floor_db, generation_loss_db_per_gen, dr_ceiling_db)
    "wax_cylinder":   ( 5000, -25, -6.0, 35),
    "shellac":        ( 8000, -30, -5.0, 45),
    "lacquer_disc":   ( 8000, -32, -4.5, 50),
    "wire_recording": ( 6000, -28, -5.5, 40),
    "vinyl":          (16000, -55, -2.0, 70),
    "tape":           (15000, -50, -3.0, 62),  # Kompaktkassette (cassette→tape normiert); Typ I ~55, Typ II ~65, Mittel 62 dB
    "reel_tape":      (18000, -60, -1.5, 72),
    "cassette":       (14000, -48, -3.5, 60),
    "dat":            (22000, -90, -0.2, 92),
    "minidisc":       (20000, -85, -0.5, 88),
    "cd_digital":     (22050, -96, -0.1, 96),
    "mp3_low":        (16000, -70, -1.5, 90),
    "mp3_high":       (20000, -80, -0.5, 93),
    "aac":            (20000, -82, -0.4, 93),
    "streaming":      (20000, -78, -0.8, 90),
    "unknown":        (20000, -50, -2.0, 70),
}

def compute_cumulative_generation_loss(transfer_chain: list[str]) -> dict:
    """
    Berechnet kumulativen Verlust über die gesamte Transferkette.

    Returns:
        {
            "generation_count": int,         # Anzahl Transfer-Stufen
            "cumulative_bw_hz": float,       # = min(bw_ceiling für jede Stufe)
            "cumulative_snr_db": float,      # ≈ 10·log10(Σ 10^(loss_i/10))  
            "cumulative_hf_loss_db": float,  # = Σ generation_loss_db_per_gen
            "cumulative_dr_ceiling_db": float,  # = min(dr_ceiling für jede Stufe)
            "source_fidelity_confidence": float,  # 1.0 / (1.0 + 0.15·gen_count)
        }

    Example:
        chain = ["shellac", "reel_tape", "cd_digital", "mp3_low"]
        → bw = min(8000, 18000, 22050, 16000) = 8000 Hz
        → hf_loss = -5.0 + -1.5 + -0.1 + -1.5 = -8.1 dB
        → dr_ceiling = min(45, 72, 96, 90) = 45 dB
        → confidence = 1.0 / (1.0 + 0.15·4) = 0.625
    """
    results = {}
    bw_values = []
    dr_values = []
    total_hf_loss = 0.0
    for material in transfer_chain:
        chars = CARRIER_TRANSFER_CHARACTERISTICS.get(material, CARRIER_TRANSFER_CHARACTERISTICS["unknown"])
        bw_values.append(chars[0])
        total_hf_loss += chars[2]
        dr_values.append(chars[3])

    results["generation_count"] = len(transfer_chain)
    results["cumulative_bw_hz"] = min(bw_values) if bw_values else 20000
    results["cumulative_hf_loss_db"] = total_hf_loss
    results["cumulative_dr_ceiling_db"] = min(dr_values) if dr_values else 70
    results["source_fidelity_confidence"] = 1.0 / (1.0 + 0.15 * len(transfer_chain))
    return results
```

**Integration**:

- `SourceFidelityReconstructor` MUSS `CARRIER_TRANSFER_CHARACTERISTICS` als Single Source of Truth verwenden (keine eigene Tabelle).
- `SongCalibrationProfile` MUSS `cumulative_hf_loss_db` und `generation_count` für Stärke-Skalierung nutzen.
- `PhysicalCeilingEstimator` MUSS `cumulative_bw_hz` und `cumulative_dr_ceiling_db` respektieren.
- `RestorationResult.metadata["carrier_chain_characteristics"]` enthält das vollständige Dict.

**Invariante**: Die Formel ist deterministisch — gleiche `transfer_chain` → gleiches Ergebnis. Keine lernbasierten oder stochastischen Elemente in der Baseline-Berechnung.

---

## §4.8a [RELEASE_MUST] DSP-PRESERVE-Taxonomie — Was niemals repariert werden darf (v10.0.0)

> **Normative Querverbindung**: §6.5 (Spec 05) definiert die vollständige Authentic Character
> Taxonomy pro Materialtyp. §4.8a spezifiziert die **DSP-seitige Durchsetzung** — wie jede
> Algorithmen-Klasse reagiert, wenn `IntentionalArtifactClassifier` ein Merkmal als PRESERVE
> klassifiziert hat.

### §4.8a-i PRESERVE-Klassen und ihre DSP-Invarianten

| PRESERVE-Klasse | Merkmale | DSP-Invariante |
| --- | --- | --- |
| **VINTAGE_SATURATION** | H2/H3/H4 < −20 dBFS; regelmäßig verteilt | NR-Gain-Floor ≥ 0.90 in Harmonic-Bins; kein THD-Remover |
| **ROOM_AMBIENCE** | frühe Reflexionen 5–50 ms; SNR > 15 dB | Dereverb-Wet ≤ 0.20; kein Gating unter −55 dBFS |
| **RECORDING_AMBIENCE** | Raumrauschen −45 bis −35 dBFS; spectral_flatness > 0.5 | Masking-Guard G_floor ≥ 0.20; kein NR darunter |
| **TAPE_SATURATION** | bandbegrenzte Obertöne 2–8 kHz; THD < 3 % | phase_07 Strength ≤ 0.10 wenn Ziel-Material gleich Tape |
| **PERFORMANCE_VIBRATO** | F0-Modulation 4–7 Hz, Tiefe ≥ 10 Cent | Pitch-Correction-Strength = 0.0 in Vibrato-Segmenten |
| **PERIOD_NOISE_FLOOR** | Spektrale Form konsistent mit Ära-Rauschprofil | OMLSA-Softening > 0.50 (kein aggressives NR) |
| **AUTHENTIC_CLICKS** | Shellac: Clicks < 5 ms in charakteristischer Dichte | phase_09 Skip wenn AUTHENTIC_CLICK_DENSITY Flag gesetzt |
| **ANALOG_WARMTH** | Tief-Mittenbetonung 200–800 Hz aus Bandbreite | kein HPF < 200 Hz für Kompensation; kein Equalizer-Flatten |

### §4.8a-ii Durchsetzung in NR-Algorithmen

```python
# Für DeepFilterNet, OMLSA, SGMSE+, Apollo:
# NACH IntentionalArtifactClassifier (Spec 03 §2.44) — VOR Phase-Ausführung:

_preserve_mask = kwargs.get("preserve_mask")  # Von UV3 injiziert
if _preserve_mask is not None:
    # preserve_mask: 1.0 = PRESERVE-Zone (kein Eingriff erlaubt)
    #               0.0 = REPAIR-Zone (normale NR-Stärke)

    # NR-Gain wird mit preserve_mask gewichtet:
    # effective_gain = preserve_mask * G_floor_preserve + (1 - preserve_mask) * G_computed
    G_PRESERVE_FLOOR = 0.90  # NR fast abgeschaltet in PRESERVE-Zonen
    effective_gain = (
        _preserve_mask * G_PRESERVE_FLOOR
        + (1.0 - _preserve_mask) * G_computed
    )
    # Nicht weniger als G_floor (§4.5d):
    effective_gain = np.maximum(effective_gain, 0.10)
```

### [RELEASE_MUST] §4.8a-iii Invarianten

- **VERBOTEN**: DSP-Algorithmus ignoriert `preserve_mask` aus `_restoration_context`.
- Alle NR-Phasen (phase_03, phase_29, phase_03_sgmse, phase_28_apollo) MÜSSEN `preserve_mask` akzeptieren.
- Laufzeit: `get_preserve_mask()` (Spec 03 §2.44) ≤ 0.5 s/min Audio.
- PRESERVE-Klassifikation ist **session-stabil**: einmal klassifiziert, gilt für gesamte Pipeline (keine Re-Klassifikation pro Phase).

---

## §4.11 [RELEASE_MUST] Pre-Echo-Detektionsalgorithmus — Temporal Masking Artifact (v10.0.0)

> **Kontext**: Pre-Echo ist das diagnostisch schwierigste Codec-Artefakt. Es entsteht, wenn
> Transform-Codecs (MP3, AAC, Opus) die zeitliche Vorwärts-Maskierung des menschlichen Gehörs
> ausnutzen: Das Gehör maskiert kurze Geräusche vor einem lauten Transient (Prä-Masking: 5–20 ms).
> Der Codec quantisiert das Block davor grob — bei niedrigen Bitraten wird das quantisierte
> Rauschen hörbar als "Vorecho" vor dem Transient (−20 bis −30 dB unter Transientpeak).
> **Konventionelle NR-Algorithmen versagen**: Pre-Echo ist kein stationäres Rauschen, kein Klick,
> keine spektrale Lücke — es ist zeitlich-energetisch lokalisiertes Vorartefakt.

### §4.11a Detektionsalgorithmus (DSP, kein ML erforderlich)

```python
# backend/core/dsp/pre_echo_detector.py
# Singleton: get_pre_echo_detector()

class PreEchoDetector:
    """
    Erkennt Pre-Echo-Artefakte durch Rückwärts-Temporal-Masking-Analyse.

    Prinzip: Ein Block VOR einem Transient-Onset sollte LEISER sein als der Transient.
    Pre-Echo = Block-Energie vor Onset überschreitet Temporal-Masking-Schwelle.
    """

    FRAME_SIZE_MS   = 23.2      # ISO 11172-3 Standard-Blockgröße (1024 Samples @ 44.1 kHz)
    HOP_SIZE_MS     = 11.6      # 50 % Overlap
    PRE_MASK_WINDOW = 3         # 3 Frames = ~34 ms vor Transient (Prä-Masking-Fenster)

    # Pre-Echo-Schwellen (dB über geschätztem Pre-Masking-Boden)
    # Kalibriert: Menschliche Hörschwelle für Vorecho (Fastl & Zwicker 2007, §7.2)
    THRESHOLDS = {
        "shellac":    +6.0,   # Shellac: hoher Rauschboden, tolerantere Schwelle
        "vinyl":      +8.0,
        "cd_digital": +12.0,  # CD/Digital: kein natürlicher Rauschboden → enge Schwelle
        "mp3_low":    +10.0,
        "mp3_high":   +11.0,
        "aac":        +11.0,
        "unknown":    +9.0,
    }

    def detect(
        self,
        audio: np.ndarray,  # Mono oder Stereo [C, T] oder [T]
        sr: int,
        material_key: str = "unknown",
    ) -> List[Dict]:
        """
        Gibt Liste erkannter Pre-Echo-Ereignisse zurück.

        Jedes Ereignis:
        {
            "onset_sample":  int,      # Sample-Position des Transients
            "pre_echo_start": int,     # Beginn des Pre-Echo-Artefakts
            "pre_echo_end":   int,     # Ende (= onset_sample)
            "severity_db":   float,   # Energie-Überschuss in dB über Masking-Floor
            "confidence":    float,   # [0, 1] Detektionssicherheit
        }

        Algorithmus:
        1. STFT (Frame 1024, Hop 512 @ 48 kHz)
        2. Onset-Detektor (Energiefluss, Kramer-Methode, Onset = +10 dB in ≤ 2 Frames)
        3. Für jeden Onset: Rückwärts-Energieprofil über PRE_MASK_WINDOW Frames
        4. Temporal-Masking-Boden = exponential decay: E_mask(t) = E_onset × 10^(-0.1 × dt_ms)
           (Prä-Masking-Zerfallsrate: −10 dB per 20 ms, Fastl & Zwicker 2007 Fig 7.3)
        5. Pre-Echo wenn: E_actual(frame) > E_mask(frame) + THRESHOLDS[material]
        """
        ...

    def repair_region(
        self,
        audio: np.ndarray,
        pre_echo_event: Dict,
        sr: int,
    ) -> np.ndarray:
        """
        Reduziert Pre-Echo durch zeitlich-selektives Spectral-Subtraction.

        NICHT: globales NR (würde Transient beschädigen)
        SONDERN: Frame-selektive Dämpfung nur im pre_echo_start:onset_sample-Bereich

        Methode (Ephraim-Malah MMSE in pre_echo_region):
        1. Noise-Estimation: E_mask(frame) als Referenz
        2. Gain-Floor: max(G_floor, masking_threshold[band] / pre_echo_energy[band])
        3. Apply gain zu spektralen Komponenten im Prä-Masking-Fenster only
        4. PGHI Phasenrekonstruktion für modifiziertes Segment
        5. 5 ms Crossfade zu benachbarten Frames (kein Click)

        VERBOTEN: Gain < G_floor = 0.10 (§2.62 Masking-Gain-Floor-Invariante)
        VERBOTEN: Eingriff außerhalb pre_echo_start:onset_sample + 2 ms Puffer
        """
        ...
```

### §4.11b Integration in Phase_50 und DefectScanner

**DefectScanner-Pflicht** (neues Defect-Cause "pre_echo" mit Algorithmus):

```python
# In DefectScanner, bei Codec-Material (mp3_*, aac, opus):
_pre_echo_detector = get_pre_echo_detector()
_pre_echo_events = _pre_echo_detector.detect(audio, sr, material_key)
if len(_pre_echo_events) > 0:
    severity = np.mean([e["severity_db"] for e in _pre_echo_events])
    defects["pre_echo"] = DefectResult(
        detected=True,
        severity=float(np.clip(severity / 20.0, 0.0, 1.0)),  # Normiert auf [0,1]
        events=_pre_echo_events,
    )
```

**Phase_50-Pflicht** (Pre-Echo-spezifische Verarbeitung):

```python
# In phase_50_spectral_repair.py, wenn pre_echo_events in kwargs:
_events = kwargs.get("pre_echo_events", [])
if _events:
    for event in _events:
        audio = _pre_echo_detector.repair_region(audio, event, sr)
    # ANSCHLIESSEND erst allgemeines spektrales Repair (HF-Spikes etc.)
    # Reihenfolge: Pre-Echo-Repair → generisches Spektral-Repair
    # Begründung: Pre-Echo-Repair ändert lokale Spektral-Energieverteilung;
    # generisches Repair würde reparierten Bereich sonst erneut als Artefakt flaggen
```

### §4.11c CAUSE_TO_PHASES-Update (normativ)

In `CAUSE_TO_PHASES` in Spec 06 §7.2:

```python
"pre_echo": [
    "phase_50_spectral_repair",      # Primär: Pre-Echo-Repair-Region + generisch
    "phase_23_spectral_repair",      # Sekundär: Spektrale Lücken nach Repair
]
# VERBOTEN: phase_03_denoise als Pre-Echo-Recovery (Pre-Echo ist kein stationäres Rauschen)
```

### §4.11d Materialspezifische Aktivierung

| Material | Aktivierung | Begründung |
| --- | --- | --- |
| `mp3_low` (< 128 kbps) | IMMER | Schwerer Pre-Echo praktisch garantiert |
| `mp3_high` (≥ 128 kbps) | Wenn DefectScanner Severity ≥ 0.25 | Seltener, aber möglich |
| `aac`, `opus` | Wenn DefectScanner Severity ≥ 0.20 | AAC hat besseres Temporal-Masking-Modell |
| `shellac`, `vinyl` | NIEMALS | Pre-Echo ist Transform-Coding-Artefakt; kein analoges Äquivalent |
| `cd_digital` | NIEMALS | CD ist linear PCM, kein lossy Codec |

> **Kreuzreferenz**: §2.49 (Pre-Echo-Detektor im AFG, Gewicht 0.8), §4.1 (ISO 11172-3
> Psychoacoustic Masking), Spec 06 §7.8 (Phase-50 HF-Spike-Schutz), §4.8a PRESERVE-Taxonomie

---

## §4.9 [RELEASE_MUST] FlashSR Wall-Time-Budget (v10.0.0)

FlashSR-Zonen-Schleifen (BWE in phase_06, phase_23, phase_24) können bei extremen Songstrukturen (>180 s, komplexe Texturen) zeitlich unbegrenzt laufen → Pipeline-Hänger.

**Normative Regel**:

```python
_AUDIOSR_WALL_BUDGET_S = 900.0  # 15 min maximal für FlashSR-Zonenschleife
```

- Vor der Zonenschleife: `wall_start = time.monotonic()`
- Jede Zone prüft: `if time.monotonic() - wall_start > _AUDIOSR_WALL_BUDGET_S: break`
- Zonen jenseits des Budgets: **Passthrough** (Original-Audio), kein Inpainting
- `metadata["flashsr_wall_budget_exceeded"]` = True bei Budget-Überschreitung
- Telemetrie: `metadata["flashsr_zones_completed"]` / `metadata["flashsr_zones_total"]`

**Invariante**: FlashSR-Timeout darf die Pipeline nicht crashen — verbleibende Zonen werden als Original-Audio beibehalten, nicht abgebrochen oder leer gelassen.

---

## §4.10 [RELEASE_MUST] Pitch-Tracking-Kaskade — Tier-Reihenfolge (v10.0.0)

Die SOTA-Matrix (§4.4) definiert die Pitch-Tracking-Kaskade als:

```
FCPE (Tier-0, primär) → RMVPE (Tier-1, Fallback) → PESTO (Tier-2) → pYIN (Tier-3, DSP-Fallback)
```

**Semantik der Pfeile**: `→` bedeutet **OOM/Fehler-Fallback**, nicht Prioritätsreihenfolge. FCPE wird immer zuerst versucht. Nur bei Fehler (OOM, ONNX-Crash, Timeout) wird der nächste Tier aktiviert.

| Tier | Modell | Typ | Anmerkung |
| --- | --- | --- | --- |
| 0 | FCPE | ML (ONNX) | Primär für alle Pitch-Pfade |
| 1 | RMVPE | ML (Torch) | −30 % Fehlerrate bei Gesang (Wei et al. ICASSP 2023) |
| 2 | PESTO | ML (light) | Schnell, weniger genau |
| 3 | pYIN | DSP | Ultimativer Fallback, kein ML nötig |

**VERBOTEN**: CREPE als Tier in der Produktionskaskade (deprecated seit v10.0.0). CREPE bleibt nur in `_PHASE_REQUIRED_MODELS` für Legacy-Kompatibilität.

**Betroffene Phasen**: phase_12 (Wow/Flutter), phase_56 (Spectral Band Gap), hybrid_wow_flutter, hybrid_speed_pitch_ml.

---

## §12 Referenzen (Auswahl — Pflicht-Algorithmen)

- Cohen & Berdugo (2002): IMCRA — _Noise Estimation by Minima Controlled Recursive Averaging_
- Cohen (2003): OMLSA — _Noise Spectrum Estimation in Adverse Environments_
- Le Roux & Vincent (2013): _Consistent Wiener Filtering for Audio Source Separation_
- Perraudin et al. (2013): PGHI — _A Non-Iterative Method for STFT Phase (Re)construction_
- Févotte & Idier (2011): _Algorithms for NMF with the β-Divergence_
- Mauch & Dixon (2014): pYIN — _A Fundamental Frequency Estimator Using Probabilistic Threshold Distributions_
- Fletcher (1964): _Normal Vibration Frequencies of a Stiff Piano String_
- Engel et al. (2020): DDSP — ICLR 2020 (Google Magenta)
- Bregman (1990): _Auditory Scene Analysis_ — MIT Press
- Moore, Glasberg & Baer (2006): Virtual Pitch — JASA
- Zhang et al. (2024): Apollo — _Band-sequence Modeling for High-Quality Music Restoration_
- Moliner & Välimäki (ICASSP 2023): CQTdiff+ — _Solving Audio Inverse Problems with a Diffusion Model_
- Lu et al. (2023): BS-RoFormer — _Music Source Separation with Band-Split RoPE Transformer_
- Siuzdak (2023): Vocos — _Resynthesizing Speech and Music with Neural Vocoders_
- Wannamaker et al. (1992): POW-r — _A Theory of Nonsubtractive Dither_
- Lipman et al. (2023): _Flow Matching for Generative Modeling_
- Radford et al. (2022): Whisper — _Robust Speech Recognition via Large-Scale Weak Supervision_
- ISO 226:2003: _Acoustics — Normal Equal-Loudness-Level Contours_
- ITU-R BS.1770-5 (2023): _Algorithms to measure audio programme loudness_
- Záviška et al. (2021): _A Proper Version of Synthesis-based Sparse Audio Declipper_ — EUSIPCO 2021
- Blauert (1997): _Spatial Hearing — The Psychophysics of Human Sound Localization_ — MIT Press
- Kuttruff (2009): _Room Acoustics_ — 5th Ed. (C80/D50 Definitionen §4.5c)

---

## §4.6 [RELEASE_MUST] Loudness-Guard DSP-Invarianten (v10.0.0)

DSP-Pflichtregeln für alle Loudness-Drift-Guards in der Pipeline (§2.45a).

### §4.6a Gated-RMS-Messung

```python
# VERBOTEN — globaler RMS misst Stille mit:
rms_db = 20.0 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-10)

# PFLICHT — Gated-RMS (nur musikalische Frames):
def _rms_dbfs_gated(audio, frame_size=2048, gate_dbfs=-50.0, min_gate_ratio=0.05):
    # Stereo → Mono downmix vor Framing
    if audio.ndim == 2:
        mono = (audio[0] + audio[1]) * 0.5
    else:
        mono = audio
    # Frame-basierte Messung
    n_frames = len(mono) // frame_size
    frames = mono[:n_frames * frame_size].reshape(n_frames, frame_size)
    frame_rms_db = 20.0 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-10)
    # Gate: nur Frames > gate_dbfs
    mask = frame_rms_db > gate_dbfs
    if mask.sum() < max(1, int(n_frames * min_gate_ratio)):
        return float(np.mean(frame_rms_db))  # Fallback: ungated
    return float(np.mean(frame_rms_db[mask]))
```

### §4.6b Envelope-Aware Gain

```python
# VERBOTEN — uniformer Gain amplifiziert Stille:
audio *= gain_factor

# PFLICHT — musik-selektiver Gain:
def _musical_gain_envelope(audio, gain, gate_dbfs=-50.0, sr=48000):
    # Gate-Envelope: 1.0 für Musik-Frames, 0.0 für Stille
    # Crossfade: 10 ms Hann-Smoothing an Gate-Übergängen
    # Gain-Formel pro Sample: out = audio * (1.0 + (gain - 1.0) * gate_envelope)
    # → Stille (gate=0): out = audio * 1.0 (unverändert)
    # → Musik (gate=1): out = audio * gain (verstärkt)
```

### §4.6c Soft-Limiter

```python
# VERBOTEN — unbedingter Soft-Limiter komprimiert Musikdynamik:
_abs = np.abs(audio)
_over = _abs > 0.92
audio = np.where(_over, np.sign(audio) * (0.92 + 0.08 * np.tanh((_abs - 0.92) / 0.08)), audio)

# PFLICHT — bedingter Soft-Limiter nur bei echtem Clipping-Risiko:
peak = float(np.max(np.abs(audio)))
if peak > 0.98:  # Nur bei echtem Clipping-Risiko
    _abs = np.abs(audio)
    _over = _abs > 0.92
    if np.any(_over):
        audio = np.where(_over, np.sign(audio) * (0.92 + 0.08 * np.tanh((_abs - 0.92) / 0.08)), audio)
audio = np.clip(audio, -1.0, 1.0)  # Finales Sicherheitsnetz
```

### Rationale

Diese drei Invarianten verhindern das Silence-Amplification-Problem: Subtraktive Phasen entfernen Rauschen aus Stille-Segmenten → globaler RMS sinkt → Guard meldet falschen Pegelkollaps → uniformer Makeup-Gain amplifiziert entrauschte Stille → Soft-Limiter komprimiert Musikpeaks. Ergebnis: Musik leiser, Stille lauter — exakt das Gegenteil des Ziels.

> Kreuzreferenz: Spec 02 §2.45a-I bis §2.45a-IV

---

## §4.7 [RELEASE_MUST] Literaturbasierte Gap-Interpolation (v10.0.0)

### §4.7a Phase-09 LPC/AR-Lücken-Interpolation

Für Crackle-Lücken ≤ 50 ms (sowie Shellac `interpolation == "hybrid"`):

```python
# Kanonischer Algorithmus (_ar_fill_channel):
# 1. Vorwärts-AR-Koeffizienten aus Pre-Gap-Kontext (Yule-Walker, lpc_order=32)
#    a_fwd = librosa.lpc(pre_context[-context_len:], order=lpc_order)
# 2. Rückwärts-AR-Koeffizienten aus Post-Gap-Kontext (gespiegeltes Signal)
#    a_bwd = librosa.lpc(post_context[:context_len][::-1], order=lpc_order)
# 3. Pol-Stabilisierung: alle Pole |z| >= 0.995 auf 0.994 spiegeln
#    (verhindert exponentielle Divergenz bei schlecht konditionierter Yule-Walker-Lösung)
# 4. Vorwärts- und Rückwärts-Vorhersage über Gap-Länge berechnen
# 5. Lineare Überblendung beider Vorhersagen: alpha = linspace(0, 1, gap_len)
#    fill = (1 - alpha) * fwd_fill + alpha * bwd_fill
# 6. 5 ms Boundary-Crossfade (Hann-Taper) an Lückenrändern
```

**Wissenschaftliche Referenzen**:

- Rabiner & Schafer (1978): _Digital Processing of Speech Signals_
- Lagrange & Marchand (2007): "Long Interpolation of Audio Signals using Linear Prediction", DAFX
- Godsill & Rayner (1998): _Digital Audio Restoration_, Springer

**VERBOTEN**: `_interpolate_hybrid()` als Stub, der intern `_interpolate_linear()` aufruft.

### §4.7b Phase-50 STFT-Konsistenz-Projektion (POCS)

Für Time-Axis-Dropout-Reparatur (Pass-2) in Phase 50:

```python
# Kanonischer Algorithmus (_fill_dropout_frames_pocs):
# 1. Initialisierung: linear interpolierte Spektren als Startwert für Dropout-Frames
# 2. ISTFT: Spektrum → Zeitbereich (setzt STFT-Konsistenz voraus)
# 3. STFT: Zeitbereich → Spektrum (projiziert auf konsistente STFT-Mannigfaltigkeit)
# 4. Projection: undamaged Frames werden auf Original-Spektraldaten zurückgesetzt
# 5. Schritte 2-4 wiederholen (n_iter=5)
#
# Konvergenz: POCS (Projection Onto Convex Sets) garantiert Annäherung an
# zulässigen Datenpunkt der Schnittmenge beider Constraints:
# - C1: Signal konsistent mit undamaged Frames
# - C2: Spektrum liegt auf korrekter STFT-Mannigfaltigkeit
```

**Wissenschaftliche Referenz**: Siedenburg & Dörfler (2013) "Audio Inpainting with Social Sparsity", JASA.

**VERBOTEN**: Einmalige lineare Interpolation als finale Time-Axis-Dropout-Reparatur.

**Testpflicht**: `tests/unit/test_literature_algorithms.py` (21 Tests, alle grün).

---

### §4.7c Phase-23 POCS — STFT-Konsistenz-Projektion vor PGHI (v10.0.0)

Bei Spectral-Repair (phase_23) Single-STFT-Fallback (`_repair_channel`): Interpolierte/Inpainting-Spektren sind STFT-inkonsistent → PGHI rekonstruiert Phase aus widersprüchlichen Magnitudes → Aliasing an Defektgrenzen.

**Kanonische Implementierung** (im `_repair_channel`-Pfad, nach `Zxx_blended`, vor PGHI):

```python
# POCS nur im Non-FAST-Modus und bei relevanter Defektabdeckung
if quality_mode not in ("FAST",) and defect_severity >= 0.005:
    n_iter = int(np.clip(round(2 + defect_severity * 15), 2, 5))
    if len(audio) / sr > 60.0:       # Wall-Time-Guard für lange Signale
        n_iter = min(n_iter, 2)
    for _ in range(n_iter):
        time_signal = librosa.istft(Zxx_blended, hop_length=..., win_length=...)
        Zxx_roundtrip = librosa.stft(time_signal, ...)
        # Re-ankern: undamaged Bins auf Original zurücksetzen
        Zxx_blended[~defect_mask] = Zxx_original[~defect_mask]
        # Defect-Bins: neue Phase aus Roundtrip, Original-Inpainting-Magnitude
        Zxx_blended[defect_mask] = np.abs(Zxx_blended[defect_mask]) * np.exp(1j * np.angle(Zxx_roundtrip[defect_mask]))
# Anschließend: PGHI-Rekonstruktion auf STFT-konsistentem Zxx_blended
```

**Parameter**:

- `n_iter`: material-adaptiv 2–5 (defect_severity linear interpoliert)
- `defect_severity >= 0.005`: Mindest-Schwelle (0.5 % Defektabdeckung)
- Wall-Time-Guard: `>60 s Signaldauer → n_iter = min(n_iter, 2)`
- FAST-Mode: POCS komplett übersprungen (Laufzeit-Priorität)
- Non-blocking: Exception im POCS-Loop lässt `Zxx_blended` unverändert

**Wissenschaftliche Referenz**: Siedenburg & Dörfler (2013) analog zu §4.7b.

**VERBOTEN**: PGHI auf direkt interpolierten/inpainteten Spektren ohne vorherige POCS-Konsistenzprojektion (erzeugt systematisches Aliasing an Defektgrenzen).

---

## §4.6a ML-Plugin-Integration — Workflow, Budget & ONNX-Chunking (konsolidiert aus Skill ml-plugin)

### Memory-Budget-Pflicht (RELEASE_MUST)

**Jeder** ML-Modell-Load MUSS diesen Ablauf einhalten:

```python
from backend.core.ml_memory_budget import get_ml_memory_budget
from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

budget = get_ml_memory_budget()
plm = get_plugin_lifecycle_manager()

# 1. Budget prüfen — VOR torch.load() / InferenceSession()
if not budget.try_allocate("my_model", size_gb=1.2):
    logger.warning("ml_budget_denied model=my_model required_gb=1.2")
    budget.release("my_model")  # safety cleanup
    return _dsp_fallback(audio, sr)  # PFLICHT: DSP-Fallback

try:
    # 2. Modell laden
    model = onnxruntime.InferenceSession(path, providers=get_ort_providers("my_model"))
    # 3. LRU-Tracking registrieren
    plm.register("my_model", size_gb=1.2, unload_fn=lambda: del_model())
except Exception:
    budget.release("my_model")  # 4. IMMER release bei Fehler
    return _dsp_fallback(audio, sr)
```

**Verboten**: `plm.try_allocate()` (existiert nicht), `torch.load(..., map_location="cuda")` ohne ml_device_manager, ML-Load ohne `try_allocate()`.

**Auto-Budget-Formel**: `max(4.0, min(12.0, RAM_GB / 3))`. Budget-Einheit: **immer GB (float)**, nie MB.

### §4.6b [RELEASE_MUST] PLM-Inferenz-Schutz — Active-Guard-Pflicht (v10.0.0)

**Problem**: Emergency-Eviction (`evict_if_needed` bei RAM > 78 %) entlädt registrierte Plugins nach LRU-Alter. Wenn ein Plugin gerade in einer Phase aktiv ist, aber `entry.active == False`, wird es evictiert → Inferenz-Crash → OOM-Eskalation → Kernel-Kill.

**Pflicht-Workflow für jede ML-Inferenz innerhalb einer Phase:**

```python
plm = get_plugin_lifecycle_manager()
plm.set_active("my_model", True)   # VOR Inferenz-Start
try:
    result = model.run(input)       # oder session.run()
finally:
    plm.set_active("my_model", False)  # IMMER nach Inferenz-Ende
```

**Invarianten:**

1. **Emergency-Eviction darf NIEMALS ein Plugin entladen, dessen `entry.active == True`** — unabhängig vom RAM-Druck. Bei 100 % aktiver Plugins und RAM-Krise → `gc.collect()` + `malloc_trim(0)`, aber kein Evict.
2. **`set_active(name, True)` MUSS vor dem ersten `model.run()`/`session.run()` der Phase aufgerufen werden** — nicht erst vor dem Ergebnis-Zugriff.
3. **`set_active(name, False)` MUSS in `finally`-Block** — auch bei Timeout, OOM oder Inferenz-Fehler.
4. **`_PHASE_REQUIRED_MODELS` MUSS alle ML-Modelle listen, die eine Phase in irgendeinem Codepfad laden kann** (primär UND Fallback). Unvollständiges Mapping = PLM evictiert benötigte Modelle bei `evict_for_phase()`.

**VERBOTEN**: ML-Inferenz ohne `set_active()`-Guard; Emergency-Eviction von `entry.active`-Plugins; `_PHASE_REQUIRED_MODELS`-Eintrag, der nur den Primärpfad listet.

### §4.6c [RELEASE_MUST] Phase-zu-Modell-Mapping — Bidirektionale Sync-Invariante (v10.0.0)

**`_PHASE_REQUIRED_MODELS`** in `backend/core/plugin_lifecycle_manager.py` ist die **Single Source of Truth** für das Phase→ML-Mapping. Es MUSS **alle** Modelle enthalten, die eine Phase laden kann:

| Phase | Primär | Fallback(s) | _PHASE_REQUIRED_MODELS |
| --- | --- | --- | --- |
| `phase_03_denoise` | SGMSE+, ResembleEnhance | DeepFilterNetV3, OMLSA (DSP) | `{"SGMSE+", "ResembleEnhance", "DeepFilterNetV3"}` |
| `phase_09_crackle_removal` | BANQUET | DSP (Median-Filter) | `{"BANQUET"}` |
| `phase_12_wow_flutter_fix` | FCPE | RMVPE, CREPE, pYIN (DSP) | `{"FCPE", "RMVPE", "CREPE"}` |
| `phase_18_noise_gate` | SileroVAD | Energy-Gate (DSP) | `{"SileroVAD"}` |
| `phase_20_reverb_reduction` | SGMSE+ | WPE (DSP) | `{"SGMSE+"}` |
| `phase_23_spectral_repair` | Apollo | FlashSR, PGHI (DSP) | `{"Apollo", "FlashSR"}` |
| `phase_24_dropout_repair` | FlashSR | AR-Interpolation (DSP) | `{"FlashSR"}` |
| `phase_29_tape_hiss_reduction` | DeepFilterNetV3 | OMLSA (DSP) | `{"DeepFilterNetV3"}` |
| `phase_42_vocal_enhancement` | BSRoFormer | NMF-β (DSP) | `{"BSRoFormer"}` |
| `phase_43_ml_deesser` | MP-SENet | OMLSA (DSP) | `{"MP-SENet"}` |
| `phase_49_advanced_dereverb` | SGMSE+ | WPE (DSP) | `{"SGMSE+"}` |
| `phase_55_diffusion_inpainting` | CQTdiff | FlowMatching, PGHI (DSP) | `{"CQTdiff", "FlowMatching"}` |
| `phase_56_spectral_band_gap` | FCPE | CREPE, pYIN (DSP) | `{"FCPE", "CREPE"}` |

**Sync-Invarianten (analog §2.55 PMGG-CIG-Sync):**

1. Jeder `try_allocate("ModelName", ...)` Aufruf innerhalb einer Phase MUSS einen korrespondierenden Eintrag in `_PHASE_REQUIRED_MODELS` haben.
2. Jeder Eintrag in `_PHASE_REQUIRED_MODELS` MUSS einem tatsächlichen `try_allocate()`-Aufruf in der Phase entsprechen.
3. **Testpflicht**: CI-Regressionstest `tests/unit/test_plm_phase_model_sync.py` scannt alle `phase_*.py`-Dateien nach `try_allocate()`-Aufrufen und gleicht mit `_PHASE_REQUIRED_MODELS` ab.

### §2.38a Headroom-Guard (RELEASE_MUST)

Für schwere ML-Pfade (SGMSE+, ResembleEnhance, FlashSR, CQTdiff/FlowMatching):

1. Vor Load: Physischer RAM-Headroom prüfen (mono/stereo, Dateilänge)
2. Bei knappem RAM: `evict_stale_plugins()` + `gc.collect()` + `malloc_trim(0)`
3. Guard triggert → DSP-Fallback innerhalb derselben Phase — **kein Phase-Skip**

**Structured Fallback-Metadaten** (Pflicht in `metadata["ml_guard_events"]`):
`phase_id`, `model`, `reason`, `required_gb`, `available_gb`, `channels`, `duration_s`, `fallback`.

### ONNX Fixed-Shape-Input — Pflicht-Chunking (RELEASE_MUST)

Vor jedem `session.run()` Input-Dimension prüfen:

```python
inp = session.get_inputs()[0]
fixed_len = None
if inp.shape and len(inp.shape) >= 2:
    dim = inp.shape[1]
    if isinstance(dim, int) and dim > 0:
        fixed_len = dim

if fixed_len is not None and len(audio) > fixed_len:
    # Chunking-Loop — 50 % Overlap, Zero-Padding für letzten Chunk
    results = []
    pos = 0
    while pos < len(audio):
        chunk = audio[pos : pos + fixed_len]
        if len(chunk) < fixed_len:
            chunk = np.pad(chunk, (0, fixed_len - len(chunk)))
        results.append(session.run(..., {inp.name: chunk[np.newaxis, :]}))
        pos += fixed_len // 2
```

**Invariante**: `INVALID_ARGUMENT`-ONNX-Fehler für falsche Eingabegröße ist immer Bug im Plugin-Code.

### ONNX-Sessions — Pflicht-Konfiguration

```python
opts = onnxruntime.SessionOptions()
opts.intra_op_num_threads = os.cpu_count()
opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
```

### Lazy-Load-Pflicht (Budget > 4 GB allein)

| Modell | Größe | Lazy-Load |
| --- | --- | --- |
| FlashSR | 5.9 GB | Pflicht |
| MERT-v1-330M | 3.9 GB | Pflicht |

### Hybrid-Release-Mode (RELEASE_MUST)

`release_mode` ∈ `primary | fallback | blocked`. Kein ML-Failure darf Pipeline vollständig abbrechen.
Jeder Fallback in `metadata["ml_fallbacks_used"]` protokollieren.

### §4.6d PLM Active-Guard-Audit — Implementierungsstatus pro ML-Phase

| Phase | Modell | `set_active()` Guard | Status |
| --- | --- | --- | --- |
| phase_03 | SGMSE+ | ✅ | v10.0.0 |
| phase_03 | ResembleEnhance | ✅ | v10.0.0.x |
| phase_03 | DeepFilterNetV3 | ✅ | v10.0.0.x |
| phase_09 | BANQUET | ✅ | v10.14 PLM-Guard |
| phase_12 | FCPE/RMVPE/CREPE | ✅ | v10.14 PLM-Guard |
| phase_18 | SileroVAD | ⬜ leicht (CPU) | nicht nötig (<100 MB) |
| phase_20 | SGMSE+ | ✅ | v10.0.0 |
| phase_23 | Apollo | ✅ | v10.0.0 |
| phase_24 | FlashSR | ✅ | v10.14 PLM-Guard |
| phase_29 | DeepFilterNetV3 | ✅ | v10.0.0.x |
| phase_42 | BSRoFormer/MDX23C | ✅ | v10.14 PLM-Guard |
| phase_43 | MP-SENet | ✅ | v10.14 PLM-Guard |
| phase_49 | SGMSE+ | ✅ | v10.0.0 |
| phase_55 | CQTdiff/FlowMatching | ✅ | v10.14 PLM-Guard |

**Invariante**: Jede Phase mit ⬜ MUSS vor Release v10.0.0 den `set_active()`-Guard implementieren.
Leichte CPU-Modelle (< 200 MB) sind ausgenommen, da Emergency-Eviction sie nicht targetiert.

### Checkliste neues ML-Plugin

```
□ plugins/<name>_plugin.py
□ ml_memory_budget.try_allocate(name, size_gb) VOR Load (Einheit: GB float)
□ ml_memory_budget.release(name) in ALLEN Fehler-Pfaden
□ plm.register(name, size_gb, unload_fn) nach erfolgreichem Load
□ plm.set_active(name, True) VOR Inferenz; plm.set_active(name, False) in finally-Block (§4.6b)
□ _PHASE_REQUIRED_MODELS-Eintrag mit ALLEN Modellen (primär + Fallback) (§4.6c)
□ DSP-Fallback für ImportError UND Budget-Überschreitung
□ Heavy: providers=get_ort_providers("Name") / Light: providers=["CPUExecutionProvider"]
□ Headroom-Guard für schwere Modelle (> 1 GB)
□ ONNX Fixed-Shape-Check: inp.shape[1] → Chunking
□ models/manifest.json: sha256 + bundled_path + size_gb + fallback
□ Tests als ml/slow markieren wenn Timeout ≥ 30 s
□ CHANGELOG.md Eintrag
```

---

## §4.6b Material×Defect DSP-Entscheidungsbaum (konsolidiert aus Skill aurik-dsp-decision)

### [RELEASE_MUST] MRSA-Zonen (5 Pflicht-Zonen)

| Zone | FFT-Size | Bereich |
| --- | --- | --- |
| sub_bass | 65536 | < 80 Hz |
| mid_low | 16384 | 80–500 Hz |
| mid | 8192 | 500 Hz–4 kHz |
| presence | 1024 | 4–8 kHz |
| air | 128 | > 8 kHz |

PGHI per Zone, Kreuzfade Hanning 10 ms. **VERBOTEN**: willkürliche FFT-Größen.

### Integrations-Checkliste neue Modelle

```
□ Lokal gebündelt (kein Download-Code in Produktion)
□ models/manifest.json v2 Eintrag mit SHA256
□ Post-2018-DSP-Fallback definiert
□ SR=48000 Konformität geprüft
□ Musik-spezifischer Benchmark (nicht PESQ/DNSMOS)
□ Material × DefectType Mapping eingetragen
□ Plugin-Policy-Konformität (§11.3)
□ Thread-safe Singleton-Integration
```

### Chunk-Verarbeitung (§7.6)

Severity ≥ 0.6 → 5 s, ≥ 0.3 → 15 s, sonst 60 s (Min 2 s / Max 120 s).
Crossfade: Hanning 10 ms. Modul: `backend/core/adaptive_chunk_processor.py`

### Versionsmatrix

| Modell | Version | Eingebunden seit |
| --- | --- | --- |
| DeepFilterNet | v3.II | Aurik 10.0 |
| MelBandRoformer | 860 MB ONNX | Aurik 10.10.x |
| MDX23C | Kim_Vocal_2 / Kim_Inst | Aurik 10.0 (Fallback) |
| Apollo | v1 TorchScript | Aurik 10.0 |
| FCPE | ONNX | Aurik 10.10.x |
| Vocos | 48 kHz nativ ONNX | Aurik 10.10.x |
| BEATs | iter3 ONNX 90 MB | Aurik 10.10.x |
| VERSA | PyTorch Checkpoint (Huang et al. 2024) | Aurik 10.10.x |
| SGMSE+ | TorchScript 251 MB | Aurik 10.10.x |
| Flow Matching | ONNX/PT | Aurik 10.10.x |
| Whisper-Tiny | ONNX 39 MB | Aurik 10.10.46b |
| Resemble-Enhance | ONNX 722 MB | Aurik 10.0 (Fallback) |

### §4.10.7 Migrationspfad Bark/LUFS (§v10.101)

**Status:** Die perzeptuellen DSP-Module sind implementiert und in der
Pipeline verdrahtet. Die Migration der Einzelphasen erfolgt schrittweise:

| Phase | Status | Nächster Schritt |
|-------|--------|-----------------|
| Wet/Dry-Blend (alle Phasen) | ✅ `perceptual_blend()` aktiv | — |
| JND-Gate (alle Phasen) | ✅ `should_skip_phase()` aktiv | — |
| Phase_35 (Multiband Compression) | ✅ Bark-Gruppen 24→4 | — |
| Phase_40 (Loudness-Normalization) | ✅ fftconvolve-Optimierung | LUFS bereits genutzt |
| QualityAnalyzer | ✅ perzeptuelle Gewichtung | — |
| DoNoHarmGuardian | ✅ Crest-Faktor | — |
| DTW-Groove | ✅ Sakoe-Chiba-Fix | — |
| STFT-Cache | ✅ measure_all | — |
| Phase_54 (Transparent Dynamics) | 🟡 lineare Bänder | Bark-Band-Migration |
| Phase_26 (DR-Expansion) | 🟡 dB-Schwellen | LUFS-Integration |
| Phase_16 (Final EQ) | 🟡 lineare Bänder | Bark-Band-Migration |
| Phase_09 (Crackle Removal) | 🟡 lineare Bänder | Bark-Band-Migration |

**Garantie:** Neue Phasen MÜSSEN Bark/LUFS verwenden (§V35, §V36).
Bestehende Phasen werden bei nächster Überarbeitung migriert.
