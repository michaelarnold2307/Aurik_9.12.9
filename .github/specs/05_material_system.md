# Aurik 10 — Spec 05: Material-System | §v10 Pleasantness-First

> Definiert alle 15 Materialtypen (+ 2 Multichannel → Downmix), defektdichte-adaptive Verarbeitungsregeln,
> GP-Gedächtnis, Export, Sample-Rate-Strategie, Tonträgerketten-Erkennung.

---

## §6.1 Unterstützte Materialien (15 Typen)

```python
# Aurik 10: ausschließlich MONO und STEREO — kein Mehrkanalformat.
# > 2 Kanäle → PANNs-gewichteter Stereo-Downmix (automatisch, kein eigenständiger Materialtyp).
SUPPORTED_MATERIALS = [
    "tape",          # Kassette: Dropout, Hiss, Wow/Flutter
    "reel_tape",     # Profi-Spulenband: Hiss, Print-Through, Dropout
    "vinyl",         # Schallplatte: Crackle, Warp, Rillenverzerrung
    "shellac",       # Schellack-78: Hochpegelrauschen, BW ≤ 8 kHz
    "wax_cylinder",  # Wachswalze (1890–1930): extrem hoher Rauschen, BW ≤ 5 kHz
    "wire_recording",# Drahtband (1940–1955): Jitter, Frequenz-Dropout
    "lacquer_disc",  # Acetat-Lackfolien (1930–1950): Riss-Klicken, Substrat-Rauschen
    "dat",           # Digital Audio Tape: Jitter, Dropout, ATRAC
    "cd_digital",    # CD/WAV: Clipping, Quantisierungsrauschen
    "mp3_low",       # MP3 < 128 kbps: starke Kompressionsartefakte
    "mp3_high",      # MP3 ≥ 128 kbps: moderate Artefakte
    "aac",           # AAC/M4A: moderne Kompression
    "minidisc",      # MiniDisc (ATRAC): 90er-Artefakte
    "streaming",     # Streaming-Kopie: variables Bitrate-Profil
    "unknown",       # Unbekannt: konservative Prior
]
# Hinweis: lacquer_disc, wax_cylinder, wire_recording → historische Materialien (v10.0.0)
# [RELEASE_MUST] Schlüssel-Mapping MediumDetector → SUPPORTED_MATERIALS (kanonisch):
# MediumDetector (forensics/medium_detector.py) verwendet intern andere Schlüsselnamen.
# MediumDetector.detect() MUSS diese Keys vor Rückgabe auf SUPPORTED_MATERIALS normieren:
#   "cassette"         → "tape"           (Compact Cassette; Typ I/II/IV).
#   "reel_wire"        → "wire_recording" (Drahtbandgerät 1940–1955)
#   "cassette_digital" → "dat"            (Digitalkassette; DAT-Verarbeitungspfad greift)
#   "vhs_audio"        → "tape"           (VHS-Tonspur; Kassetten-Restaurierungspfad)
#   "composite"        → transfer_chain[0] normiert (erster Träger bestimmt primären Pfad)
# Alle anderen MediumDetector-Schlüssel (vinyl, shellac, reel_tape, cd_digital, …)
# entsprechen direkt den SUPPORTED_MATERIALS-Werten — kein Mapping nötig.
# INVARIANTE: material_type in RestorationResult MUSS immer ein SUPPORTED_MATERIALS-Schlüssel sein.
```

---

## [RELEASE_MUST] §6.2 Material-spezifische Verarbeitungsregeln

| Material | Hauptdefekte | Prioritäts-Phasen | PQS-Erwartung |
| --- | --- | --- | --- |
| `tape` | Dropout, Hiss, Wow/Flutter | phase_24, phase_29, phase_12 | MOS ≥ 4.2 |
| `reel_tape` | Print-Through, Hiss, Dropout | phase_29, phase_03, phase_24, phase_55 | MOS ≥ 4.3 |
| `vinyl` | Crackle, Warp, Low-Freq-Rumble | phase_09, phase_12, phase_05 | MOS ≥ 4.0 |
| | | **Begründung**: Vinyl erzeugt kein DC-Offset (Tape-/ADC-Defekt); typischer LF-Defekt ist Plattenteller-Lagergeräusch (LOW_FREQ_RUMBLE, 5–20 Hz) → phase_05 (Rumble-Filter, Hochpass < 20 Hz). VERBOTEN: phase_30 als Vinyl-Prioritätsphase. | |
| `shellac` | Breites Rauschen, Bandbegr. | phase_03, phase_06, phase_01 | MOS ≥ 3.8 |
| `dat` | Jitter, Dropout, ATRAC | phase_24, phase_02, phase_23 | MOS ≥ 4.4 |
| `cd_digital` | Clipping, Quantisierung | phase_23, phase_06, phase_40 | MOS ≥ 4.5 |
| `mp3_low` | Schwere Codec-Artefakte | phase_23, phase_03, phase_50 | MOS ≥ 3.9 |
| `mp3_high` | Moderate Codec-Artefakte | phase_23, phase_50 | MOS ≥ 4.2 |
| `aac` | Präsenz-Verlust, Artefakte | phase_23, phase_38, phase_06 | MOS ≥ 4.2 |
| `minidisc` | ATRAC, HF-Verlust | phase_23, phase_06, phase_07 | MOS ≥ 4.0 |
| `wax_cylinder` | Extremrauschen, BW ≤ 5 kHz | phase_03, phase_06, phase_01, phase_29 | MOS ≥ 3.5 |
| `wire_recording` | Jitter, Freq-Dropout | phase_12, phase_24, phase_03, phase_29 | MOS ≥ 3.6 |
| `lacquer_disc` | Riss-Klicken, Substrat-Rauschen | phase_01, phase_09, phase_03, phase_29 | MOS ≥ 3.7 |
| `streaming` | Dropouts, Codec-Artefakte, Bitrate-Varianz | phase_24, phase_23, phase_50 | MOS ≥ 4.1 |
| `unknown` | Alle aktiviert | Alle Tier-1 | MOS ≥ 3.8 |

---

## §6.2a [RELEASE_MUST] Pflicht-Phasen-Aktivierung pro Material (v10.0.0)

Die in §6.2 gelisteten **Prioritäts-Phasen** eines Materials MÜSSEN **unbedingt aktiviert** werden, wenn das Material erkannt wurde — **unabhängig vom DefectScanner-Severity-Score**.

**Begründung**: Der DefectScanner arbeitet mit statistischen Schwellwerten auf begrenztem Audio-Ausschnitt. Einzelne Defekte (z. B. ein kurzer Tape-Dropout im Intro) können unter der Schwelle liegen, obwohl sie für den Hörer klar wahrnehmbar sind. Die Prioritäts-Phasen enthalten eigene, hochauflösende Detektionslogik und entscheiden selbst, ob eine Reparatur notwendig ist.

**Invariante**:

```python
# In _select_phases(): Material-Prioritäts-Phasen immer aktivieren
for phase_id in MATERIAL_PRIORITY_PHASES[material]:
    if phase_id not in selected:
        selected.append(phase_id)
```

**Ausnahme**: Phasen, die explizit durch `GoalApplicabilityFilter` für das Material deaktiviert wurden (z. B. `phase_48_stereo_imaging` bei Mono-Material).

**Fremd-Kettenglieder (§v10.705 B6 / §v10.706 B10 — ergänzt 2026-08-22)**:
Steckt ein Material nur als Zwischenstufe in der Tonträgerkette (nicht primär),
werden seine §6.2-Pflichtphasen ebenfalls aktiviert — mit einer Ausnahme:
ML-schwere Synthese-Phasen (`phase_55_diffusion_inpainting`) werden als
Fremd-Injektion nur aktiviert, wenn die Defekt-Evidenz des Kettenglieds sie
trägt (inpaint_evidence ≥ 0.35; gleiche Quelle und Schwelle wie der
Risk-Guard). Begründung: Eine „Pflicht“-Phase, die mangels Evidenz
unmittelbar wieder entfernt würde, ist Plan-Churn ohne Wohlklang-Nutzen.
Subtraktive Band-Reparaturphasen (Hiss/Dropout/…) bleiben unbedingt.

---

## §6.2b [RELEASE_MUST] Material-Dynamic-Range-Ceiling (v10.0.0)

**Problem**: `phase_26_dynamic_range_expansion` hat nur ein globales `MAX_EXPANSION_DB = 12.0`, aber kein material-spezifisches Ceiling. Ein Vinyl-Song physikalisch kann nicht mehr als ≈ 70 dB DR aufweisen — Expansion über dieses Limit erzeugt Rauschartefakte, die als „Dynamik" fehlinterpretiert werden.

**Normatives Dict (Single Source of Truth — identisch mit §4.8 `CARRIER_TRANSFER_CHARACTERISTICS`)**:

```python
_MATERIAL_DR_CEILING_DB = {
    "wax_cylinder":   35,   # Mechanische Aufnahme, Horn-Verstärkung
    "shellac":        45,   # 78 rpm, breite Rillen, hoher Rauschboden
    "lacquer_disc":   50,   # Acetat-Direktschnitt
    "wire_recording": 40,   # Stahlband, mechanische Begrenzung
    "vinyl":          70,   # LP, Best-Case-Pressung
    "tape":           62,   # Kompaktkassette (Typ I ~55 dB, Typ II ~65 dB → konservativer Mittelwert)
                            # HINWEIS: MediumDetector normiert intern "cassette" → "tape"
                            # (SUPPORTED_MATERIALS-Key). Kassetten überschreiten 65 dB nur bei
                            # Typ-IV-Metal-Band unter Best-Case-Bedingungen; 62 dB verhindert
                            # übermäßige Dynamik-Expansion, die auf echten Kassetten als
                            # Rauschartefakt hörbar wäre.
    "reel_tape":      72,   # Profi-Spulenband 15 ips (physikalisch korrekt für 30 ips: ~80 dB)
    # Interner MediumDetector-Key (vor Normalisierung cassette→tape); wird von phase_26 nicht
    # direkt verwendet, aber hier geführt für _estimate_tape_speed() und interne Ceiling-Checks.
    "cassette":       60,   # Kompaktkassette (Typ I, schlechter Transport) — MediumDetector-intern
    "dat":            92,   # Digital Audio Tape (16-bit linear)
    "minidisc":       88,   # ATRAC-Kompression
    "cd_digital":     96,   # 16-bit PCM
    "mp3_low":        90,   # Codec-bedingte theoretische Grenze
    "mp3_high":       93,
    "aac":            93,
    "streaming":      90,
    "unknown":        70,   # Konservativ: Vinyl-Niveau
}
```

**Integration in `phase_26_dynamic_range_expansion.py`**:

```python
# §6.2b DR-Ceiling (v10.0.0):
# In process():
dr_ceiling = _MATERIAL_DR_CEILING_DB.get(material_type, 70)
input_dr = compute_dynamic_range_db(audio, sr)
max_expansion = dr_ceiling - input_dr
expansion_target = min(expansion_target, max_expansion)
# Negative max_expansion → kein Bedarf für Expansion (Input bereits am Ceiling)

# §v10.61 Per-Band-Noise-Floor-Guard (v10.0.11):
# Nach DR-Ceiling-Prüfung: Downward-Expansion bekommt einen dreidimensionalen
# Noise-Floor-Guard, der die Lücke zwischen Phase_03 (Denoise) und dem finalen
# CD-Rauschprofil schließt.
#   D1: Per-Band spektrale Floor-Targets (Bass -65, Low-Mid -72, Mid-High -76, High -70 dBFS)
#   D2: Psychoakustische Maskierungs-Adaptation (+8/+5/+2/0 dB bei Band-RMS >-20/-30/-40)
#   D3: Temporale EMA-Glättung (α=0.15/0.05, asymmetrisch)
#   Asymptotischer Soft-Floor: correction = deficit × exp(-deficit/knee)
# Siehe §G87 (GEBOTE.md, Kategorie XII).
```

**Modus-Differenzierung**:

- **Restoration**: Hart-Cap bei `dr_ceiling`. Expansion über Ceiling = Artefakt (§0a).
- **Studio 2026**: Soft-Cap bei `dr_ceiling × 1.5`. Mehr Spielraum, aber nicht unbegrenzt.

**Invariante**: `_MATERIAL_DR_CEILING_DB` ist identisch mit der `dr_ceiling_db`-Spalte in `CARRIER_TRANSFER_CHARACTERISTICS` (§4.8). Änderungen müssen synchron erfolgen.

---

## §6.2c [RELEASE_MUST] Material-Bandwidth-Ceiling (v10.0.0)

**Problem**: Additive Phasen (phase_06, phase_07, phase_23, phase_39) erzeugen kumulativ Frequenzinhalt, der das physikalische BW-Limit des Quellmaterials überschreiten kann. Einzelphasen haben per-Phase-Limits, aber der kumulative Effect wird nicht zentral überwacht.

**Normatives Dict (Single Source of Truth — identisch mit §4.8 `CARRIER_TRANSFER_CHARACTERISTICS`)**:

```python
_MATERIAL_BW_CEILING_HZ = {
    "wax_cylinder":   5000,
    "shellac":        8000,
    "lacquer_disc":   8000,
    "wire_recording": 6000,
    "vinyl":         16000,
    "tape":          15000,
    "reel_tape":     18000,
    "cassette":      14000,
    "dat":           22000,
    "minidisc":      20000,
    "cd_digital":    22050,
    "mp3_low":       16000,
    "mp3_high":      20000,
    "aac":           20000,
    "streaming":     20000,
    "unknown":       20000,
}
```

**Dreistufige Enforcement-Architektur**:

1. **Per-Phase** (bestehend): Jede additive Phase hat `MATERIAL_PARAMS[material]["rolloff_hz"]` als lokales Limit.
2. **Post-Additive-Block** (NEU §2.46c): UV3 `_post_additive_bw_guard()` — Butterworth 8th-order zero-phase LPF nach dem letzten ADDITIVE-Phase-Block.
3. **Export-Gate** (bestehend): `PhysicalCeilingEstimator` → `further_optimization_worthwhile=False` wenn BW-Ceiling erreicht.

**Modus-Differenzierung**:

- **Restoration**: Hard-Cap. Output-BW darf Material-Ceiling nicht überschreiten.
- **Studio 2026**: Volle Extension erlaubt, aber OQS-äquivalenter Hörqualitätsnachweis ≥ 3.5 im Extension-Band (8 kHz–Nyquist) Pflicht; darunter → Rollback auf Material-Cap.

**Invariante**: `_MATERIAL_BW_CEILING_HZ` ist identisch mit der `bw_ceiling_hz`-Spalte in `CARRIER_TRANSFER_CHARACTERISTICS` (§4.8). Änderungen müssen synchron erfolgen.

### §6.2d [RELEASE_MUST] BW/DR-Ceiling Bidirektionale Sync-Invariante (v10.0.0)

Drei Dicts führen physikalische Materialgrenzen: `_MATERIAL_BW_CEILING_HZ` (§6.2c), `_MATERIAL_DR_CEILING_DB` (§6.2b) und `CARRIER_TRANSFER_CHARACTERISTICS` (§4.8). Diese MÜSSEN bidirektional synchron sein:

1. `_MATERIAL_BW_CEILING_HZ[m]` == `CARRIER_TRANSFER_CHARACTERISTICS[m].bw_ceiling_hz` für alle 15 Materialtypen.
2. `_MATERIAL_DR_CEILING_DB[m]` == `CARRIER_TRANSFER_CHARACTERISTICS[m].dr_ceiling_db` für alle 15 Materialtypen.
3. Änderungen an einem Dict erfordern identische Änderung im anderen.

**Testpflicht**: CI-Regressionstest `tests/unit/test_material_ceiling_sync.py` prüft die Gleichheit automatisch. Ohne diesen Test ist keine Ceiling-Änderung mergebar.

---

## §6.3 DefectType-Vollkatalog (54 DefectTypes) + §6.3b CAUSES (62 Kausal-Ursachen)

```python
# core/defect_scanner.py — DefectType (Enum, 47 Werte, Stand v10.0.0.x)
# Kanonische Kausal-Ursachen (CAUSES in causal_defect_reasoner.py): 53 Einträge

# Analoge Kerndefekte:
CLICKS, CRACKLE, HUM, LOW_FREQ_RUMBLE, DROPOUTS
WOW          # Pitch-Instabilität < 0.5 Hz (Motorgeschwindigkeit / Capstan) — IEC 60386
FLUTTER      # Pitch-Instabilität 0.5–200 Hz (Antriebsriemen / Bandführung) — IEC 60386
             # Erkennung: WOW = pYIN-Varianz über 500 ms-Fenster; FLUTTER = über 50 ms-Fenster
             # WOW → phase_12 (langsame Pitch-Korrektur); FLUTTER → phase_12 + phase_31

# Klipping, Sättigung & Gleichspannung:
CLIPPING         # Harte Amplitudenbegrenzung → REPARIEREN
SOFT_SATURATION  # Tube-/Tape-Sättigung (gerade Obertöne) → BEWAHREN!
DC_OFFSET

# Spektral:
BANDWIDTH_LOSS, HIGH_FREQ_NOISE

# Kanal/Stereo:
STEREO_IMBALANCE, PHASE_ISSUES

# Pitch:
PITCH_DRIFT

# Groove / Transienten:
TRANSIENT_SMEARING  # Ansatz-Verschmierung durch Kompression → GrooveMetric-relevant

# Hall & Magnetband:
REVERB_EXCESS, PRINT_THROUGH

# Digital/Codec:
DIGITAL_ARTIFACTS, COMPRESSION_ARTIFACTS
PRE_ECHO         # MP3/AAC Temporal-Masking-Artefakt vor Transienten
QUANTIZATION_NOISE, JITTER_ARTIFACTS, DYNAMIC_COMPRESSION_EXCESS

# Kopf-/Azimuth-Fehler:
HEAD_WEAR        # Komplette Frequenzband-Auslöschung → phase_56
AZIMUTH_ERROR    # Kammfilterung L/R durch Kopf-Fehlausrichtung → phase_14 + phase_25
                 # Signatur: frequenzabhängige L/R-Phasendifferenz, Kreuzkorrelation-Peak ≠ 0 lag
                 # Detektion: PHD(freq) = angle(STFT_L / STFT_R) → monotone HF-Drift > 20°/kHz

# Entzerrungs- & Digitalisierungsfehler (neu v10.0.0):
RIAA_CURVE_ERROR  # Falsche oder historische Disc-Entzerrungskurve → phase_04 + phase_06
                  # Kurvenvarianten (pre-RIAA 1954): NAB, Columbia, AES, Capitol, London, CCIR
                  # Erkennung: Referenzvergleich Spektral-Slope 250–8000 Hz vs. RIAA-Ideal
                  #   Abweichung > ±3 dB → RIAA_CURVE_ERROR mit erkannter Kurve als Subtyp
                  # Klassifikator liefert: curve_type ∈ {"riaa", "nab", "columbia", "aes",
                  #   "capitol", "london", "ccir", "unknown_prestandard"}
                  # phase_04 wendet Inverse-Kurve der erkannten Variante an
                  #
                  # §6.3a PRE-RIAA KURVENPARAMETER (kanonische Zeitkonstanten, bindend):
                  # Alle Werte: (τ_bass_µs, τ_mid_µs, τ_treble_µs) → Pol/Nullstellen-Triple.
                  # Inverse Korrektur: Shelving-EQ mit diesen Zeitkonstanten gespiegelt.
                  #
                  # PRE_RIAA_EQ_CURVES = {
                  #   # RIAA 1954 (Referenz — Standard ab 1954):
                  #   "riaa":           (3180, 318, 75),     # IEC 60268-4
                  #
                  #   # NAB (National Association of Broadcasters, bis 1953):
                  #   "nab":            (3180, 318, 50),     # Basswendepunkt 500 Hz, HF-Shelf 3180 µs
                  #
                  #   # Columbia 78 rpm (bis 1948):
                  #   "columbia":       (1590, 318, 0),      # Bass turnover 100 Hz, kein HF-Shelf
                  #                                          # → +6 dB Bass vs. RIAA bei 50 Hz
                  #
                  #   # AES (Audio Engineering Society, 1951–1954):
                  #   "aes":            (3180, 500, 0),      # Mittenbetonte Entzerrung
                  #
                  #   # Capitol (US, bis 1953):
                  #   "capitol":        (1590, 400, 0),      # ähnlich Columbia, flacherer HF-Abfall
                  #
                  #   # London / Decca (UK, bis 1954):
                  #   "london":         (3180, 318, 100),    # HF-Boost stärker als RIAA
                  #
                  #   # CCIR (europäischer Rundfunkstandard für Tape, sekundär für lacquers):
                  #   "ccir":           (3180, 318, 120),    # Tape-Entzerrung, 50 µs kurzfristig
                  #
                  #   # Unbekannte Vorstandardkurve — konservative Näherung Columbia:
                  #   "unknown_prestandard": (1590, 318, 0),
                  # }
                  #
                  # Erkennung Algorithmus (MediumClassifier._detect_riaa_curve_error):
                  #   1. Spectral-Slope 250–8000 Hz vs. RIAA-Ideal-LUT (±3 dB Toleranz pro Oktave)
                  #   2. Vergleich Basswendepunkt: Short-time LUFS 50–200 Hz / 200–800 Hz Ratio
                  #      → Ratio > +4 dB → columbia/nab verdächtig
                  #   3. Log-Likelihood über alle Kurven → argmax = curve_type
                  #   4. Konfidenz-Grenzwert ≥ 0.70 → RIAA_CURVE_ERROR setzen, sonst skip
                  #
                  # INVARIANTE: phase_04 MUSS bei curve_type ≠ "riaa" die exakten
                  # Zeitkonstanten aus PRE_RIAA_EQ_CURVES laden und die inverse
                  # Shelving-Kette anwenden. VERBOTEN: generische EQ-Schätzung ohne LUT.
ALIASING          # Spiegelfrequenzen durch AA-Filter-Fehler → phase_03 + phase_23
BIAS_ERROR        # Falscher Vormagnetisierungsstrom → phase_04 + phase_03 + phase_29
# --- Spec §6.3 v10.0.0: Sibilanten-Überbetonung ---
SIBILANCE         # Zischlautüberbetonung (> 6 kHz) — De-Esser-Trigger (phase_19 + phase_43)
# --- v10.0.0b: Transport-Bump ---
TRANSPORT_BUMP    # Impulsartige Mikro-Geschwindigkeitssprünge 50–300 ms (Kassette/Tape-Holpern) → phase_12
# --- v10.0.0: Vocal-Harshness ---
VOCAL_HARSHNESS   # Vokale Härte/Übersteuerung/Kratzigkeit im 2–6 kHz Band → phase_42 + phase_19
```

**CLIPPING vs. SOFT_SATURATION — kritische Unterscheidung:**

```python
def classify_clipping(audio: np.ndarray, sr: int) -> ClippingType:
    """Diskriminiert CLIPPING von SOFT_SATURATION.

    CLIPPING:        flat_tops > 0.1 % UND THD_odd > THD_even × 1.5
    SOFT_SATURATION: flat_tops < 0.1 % ODER THD_even > THD_odd
    SOFT_SATURATION → Pipeline überspringt Clipping-Reparatur (BEWAHREN!)
    """
```

---

## §6.4a [RELEASE_MUST] Material-adaptive Erkennungsschwellen im DefectScanner (v10.0.0)

DefectScanner-Erkennungsschwellen MÜSSEN **material-adaptiv** sein. Analoge Medien erfordern empfindlichere Schwellwerte als digitale Quellen.

| Defekttyp | Analog-Medien | Digital-Medien | Begründung |
| --- | --- | --- | --- |
| DROPOUTS | 20 % median-RMS | 10 % median-RMS | Tape-Dropouts: graduelle Pegelfades statt hartem Null |
| CLICKS | material-skaliert | Standard | Vinyl-Rillengeräusche vs. digitale Störimpulse |

**Analog-Medien** (empfindlichere Schwellen): `tape`, `reel_tape`, `vinyl`, `shellac`, `wax_cylinder`, `wire_recording`, `lacquer_disc`, `dat`.

**Invariante**: `_detect_dropouts()` greift auf `self.material_type` zu — dieses MUSS vor dem Aufruf aus dem resolved material_type der `scan()`-Methode gesetzt sein.

---

## §6.4b [RELEASE_MUST] `TAPE_HEAD_LEVEL_DIP` Material-Gate mit Cross-Material-Fallback (v10.0.0)

`TAPE_HEAD_LEVEL_DIP` bleibt primär material-gebunden (`tape`, `reel_tape`, `wire_recording`),
MUSS aber bei starker Morphologie auch bei Fehlklassifikation (z. B. Tape-Transfer als Vinyl markiert)
erkennbar bleiben.

### Primärregel

- Für Tape-Materialien: voller Score aus `_detect_tape_head_level_dips()`.

### Cross-Material-Fallback (nur bei starker Evidenz)

Fallback darf nur aktivieren, wenn **alle** Kriterien erfüllt sind:

- `severity >= 0.12`
- `dip_count >= 2`
- `mean_depth_db >= 6.0`
- `event_rate_per_s >= 0.15`

Bei aktivem Fallback:

- `severity := severity * 0.75`
- `confidence := min(confidence, 0.72)`
- `metadata.cross_material_fallback = True`
- `metadata.fallback_material_gate_bypassed = <material>`

### Periodizitäts-Marker (Capstan-Signatur)

Wenn mindestens 3 Events vorliegen, ist ein Confidence-Bonus zulässig, falls

- Interval-`cv < 0.35`
- `median_interval_s` in `[0.5, 3.5]`

Dann: `confidence += 0.08` (geclippt), plus

- `metadata.is_periodic_capstan`
- `metadata.median_interval_s`

### Invariante

Sauberes Nicht-Tape-Material darf durch den Fallback nicht flaggen (`severity == 0.0`).

---

## §6.4a [RELEASE_MUST] Historische Mikrofon-Response-Datenbank (v10.0.0)

**Motivation**: §2.46 Stufe 6 fordert, die Recording-Chain-Signatur zu **bewahren**. Ohne eine Referenz für die Frequenzcharakteristik des Originalaufnahme-Mikrofons ist jede EQ-Phase reine Heuristik. Diese Datenbank macht Recording-Chain-Bewahren messbar.

**Mikrofon-Response-Bibliothek** (`backend/core/microphone_response_library.py`):

```python
@dataclass
class MicrophoneProfile:
    name: str
    years_active: tuple[int, int]     # z.B. (1932, 1960)
    type: str                         # "ribbon", "condenser", "dynamic", "crystal"
    freq_response_db: dict[int, float]  # {Hz: dB} — Referenz 1 kHz = 0 dB
    genres: list[str]                 # typische Einsatzgebiete
    notes: str
```

**Kanonische Datenbank (normativ)**:

| Mikrofon | Baujahre | Typ | Charakteristika | Genre-Kontext |
| --- | --- | --- | --- | --- |
| RCA 44-BX | 1932–1955 | Ribbon | −6 dB @ 8 kHz, Hochton-Rolloff, warm | Big Band, Crooner, 1930–1950er |
| RCA 77-DX | 1950–1965 | Ribbon | +3 dB @ 3 kHz, −8 dB @ 12 kHz | Radio, Jazz, Schlager 1950er |
| Neumann CMV 3 (Ela M 1) | 1928–1940 | Condenser | stark -12 dB @ 10 kHz, Mittenbetonung | Klassik, Oper, früher Film |
| Neumann U47 | 1947–1965 | Condenser | relativ flach bis 12 kHz, sanfter Rolloff | Jazz, Klassik, Pop 1950–1965 |
| Neumann U67 | 1960–1971 | Condenser | +2 dB @ 8–12 kHz, angenehme Brillanz | Rock, Pop, Beatles-Ära |
| AKG C12 | 1953–1963 | Condenser | +3 dB @ 10 kHz, Presence-Peak | Klassik, Jazz |
| Sure SM57 | 1965–heute | Dynamic | −3 dB @ 15 kHz, +2 dB @ 5–8 kHz | Rock, Blues, Live |
| Western Electric 618-A | 1929–1940 | Dynamic | −15 dB @ 8 kHz, für 78rpm optimiert | Shellac-Ära, früher Jazz |
| Sony C37A | 1955–1965 | Condenser | +4 dB Presence 5–10 kHz | Japanischer Jazz, Pop |
| Altec 639-B | 1942–1950 | Ribbon+Dynamic | warm, kaum HF > 10 kHz | Radio, Country, Western |

**API** (`MicrophoneResponseLibrary`):

```python
def get_profile(era_decade: int, genre_label: str, material_type: str) -> MicrophoneProfile | None:
    """
    Gibt wahrscheinlichstes Mikrofon-Profil zurück.
    Reihenfolge: era_decade → genre_label → material_type (BW_CEILING).
    None wenn kein passendes Profil vorhanden (digital/moderne Ären).
    """

def get_eq_curve(era_decade: int, genre_label: str, material_type: str, target_sr: int = 48000) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Gibt Frequenz-Response als (freqs_hz: np.ndarray, gains_linear: np.ndarray) zurück.
    Frequenzen aufsteigend, gains_linear >= 0 (dB → linear konvertiert).
    Höherwertige Implementierung gegenüber {Hz: dB}-Dict: DSP-ready für np.interp.
    Kein Profil verfügbar → None. max wet_mix = 0.35 (§6.4a Hard-Cap). Keine Exception.
    """
```

**Integration mit Phase-Pipeline** (§2.46 Stufe 6):

- `phase_38` (SourceFidelityEQ / phase_06 Harmonic/BW) nutzt `get_eq_curve()` als **Zielkurve**, nicht als Korrektiv:
  - Wenn restaurierter EQ von historischem Mikrofon-EQ abweicht > 2 dB in Schlüssel-Bändern → Sanft-Anpassung Richtung Zielprofil (wet_mix ≤ 0.35)
  - **VERBOTEN**: hartes EQ-Match auf Mikrofon-Profil (Original mag mehrere Mics gehabt haben)
- Klimax/Stille-Segmente (aus §2.52b SongStructureAnalyzer): In Klimax-Phasen EQ-Anpassung deaktivieren (wet_mix = 0)
- `metadata["mic_profile_applied"]` = Mikrofon-Name oder `null`

**Erweiterbarkeit**: `backend/data/microphone_profiles.json` enthält alle Profile als JSON-Array. Neue Profile können ohne Code-Änderung ergänzt werden.

> Implementierung: `backend/core/microphone_response_library.py`
> Daten: `backend/data/microphone_profiles.json`
> Tests: `tests/unit/test_microphone_response_library.py`

---

## §6.4 GP-Gedächtnis pro Material & Genre

```text
~/.aurik/gp_memory/
    tape.json         vinyl.json      shellac.json
    digital.json      unknown.json
    schlager.json     # Genre-spezifisch (angelegt beim ersten Schlager-Job)
    jazz.json         orchestral.json
    opera.json        rock.json
```

Format:

```json
{
  "observations": [
    {"params": {"noise_reduction_strength": 0.7}, "score": 4.23, "ts": "..."}
  ],
  "version": 1
}
```

**GP-Memory-Recovery:** Korrupte Datei → `.corrupted.json` umbenennen, leer starten. Max. 500 Beobachtungen (LRU-Trim). Atomic-Write via Temp-Datei + `os.replace()`.

---

## [RELEASE_MUST] §6.5 Export-Formate & Regeln

| Format | Qualität | Anwendungsfall |
| --- | --- | --- |
| FLAC (24-bit) | Archivqualität | Standard-Export |
| WAV (24-bit, 48 kHz) | Produktionsqualität | DAW-Weiterverarbeitung |
| WAV (16-bit, 44.1 kHz) | CD-Qualität | CD-Mastering |
| MP3 CBR / VBR | 128–320 kbps / V0–V5 | Streaming, Kompatibilität |
| OGG Vorbis (q9) | Open Streaming | Plattform-unabhängig |
| AIFF (24-bit, 48 kHz) | Apple-Ökosystem | Logic Pro / Pro Tools |

**Pflicht-Regeln:**

- Bit-Tiefe: 24-bit → 24-bit (kein forced Downgrade ohne Nutzer-Wahl)
- Dithering 24→16 bit: **POW-r Typ 3** (Wannamaker 1992); Fallback: TPDF
- VERBOTEN: Truncation ohne Dithering
- Lautheit: **−14 LUFS** (EBU R128 Streaming) / **−18 LUFS** (Archiv)
- True-Peak: **−1.0 dBTP** (ITU-R BS.1770-5) — immer vor Export
- Metadaten (ID3, Vorbis Comments, BWF): vollständig übertragen + Restaurierungs-Metadaten

**MP3-Export:** LAME VBR V0 via FFmpeg (`ffmpeg -q:a 0`, adaptiv bis 320 kbps, ≈245 kbps Ø). Mono-Quellen bleiben Mono.
**VERBOTEN:** LAME CBR oder `pydub.export("out.mp3")` ohne explizite VBR-Parameter — CBR erzeugt Pre-Echo auf TDP-restaurierten Transienten (vgl. §4.5 Spec 04).

---

## §6.6 Sample-Rate- & Bit-Tiefe-Strategie

**Interne SR: immer 48 000 Hz** (vor und nach jedem DSP-Schritt).

```python
# Pflicht-Eingangsprüfung in jeder Phase und jedem Plugin:
assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"

# Resampling: Lanczos-4 (scipy.signal.resample_poly, Kaiser-Filter β=14)
# Bit-Tiefe intern: float32 in [-1, 1]
# Nach Resampling: NaN/Inf-Check Pflicht
```

---

## §6.7 Tonträgerketten-Erkennung (bindend ab v10.0.0)

**Modul**: `forensics/medium_detector.py` — `MediumDetector.detect(audio, sr, file_ext=...)` — einziges autoritatives System ab v10.0.0.

**Autoritäts- und Kettenordnung (bindend ab v10.0.0, präzisiert 2026-09-06):**

1. **Supplementär-Prinzip (Defer/Veto)**: Feature-basierte Material-Heuristiken — z. B. die DefectScanner-`_auto_detect_material`-Heuristik (§2.46f) — sind supplementär. Schwache Feature-Evidenz übernimmt den MediumDetector-Primary (Defer); ein widersprechendes Feature-Ergebnis darf den Primary nie überstimmen (Veto). Der Primary beruht auf Physical-Gates (Rotation/Infrasonic/Wow-Flutter, §6.7e) und Bayesian-Physical-Fusion (§6.8) — die stärkere Evidenzklasse. Produktionsbefund, der diese Regel erzwingt: „§2.46f Auto-detected cassette with confidence 10.42“ bei Physical-Gate-Vinyl (rotation=0.411) durch getarnte Prior-Baseline-Boni in der Heuristik.
2. **Carrier-First-Kettenordnung**: `transfer_chain[0]` ist immer der Primary (Carrier-First, §2.46a); die Folgeglieder stehen chronologisch (z. B. `vinyl → reel_tape → lacquer_disc → mp3_low`). Konsumenten (pre_analysis, Era-Ceiling §2.13, Kettenstufen-Extraktion) verlassen sich auf diese Ordnung. Der chronologische Sort ist ein Auswahlmechanismus für den Primary, nicht die finale Reihenfolge.

**Architektur**: Zweistufige Fusion aus Bayesian-Gaussian-Scoring + physikalischer Inferenz.

### Phase 1: Bayesian Gaussian-Likelihood-Scoring

16 Materialmodelle (vinyl, shellac, cassette, reel_tape, reel_wire, lacquer_disc, wax_cylinder, cd_digital, dat, minidisc, mp3_low, mp3_high, aac, cassette_digital, vhs_audio, composite) mit je 7 Feature-Dimensionen:

| Feature | Dimension |
| --- | --- |
| `bandwidth_hz` | Effektive Bandbreite (−60 dBFS HF-Rolloff) |
| `snr_db` | Spektrales SNR (Median-PSD vs. Rauschboden P5) |
| `noise_color` | Rauschfarbe-Exponent (pink=2.0, weiß=0.0) |
| `crackle_density` | Anteil Samples > 4σ (Vinyl-Knackser, events/s) |
| `wow_flutter_index` | Pitch-Varianz [Amplituden-Std] über 100-ms-Fenster |
| `infrasonic_rms` | Sub-20 Hz normierter RMS (Vinyl-Rumble, Plattentellerlagerlärm) |
| `codec_type_code` | Codec-Fingerabdruck (0.0=analog, 1.0=digital) |

Log-Likelihood: `log P(m|features) = Σ log N(f; μ_m, σ_m)` → Softmax-Posterior.

**file_ext Prior-Zeroing**: Bei digitalen Dateiendungen (`.mp3`, `.aac`, `.ogg`, `.wma`, `.opus` u. a.) werden Analog-Posteriors auf 0 gezwungen — der Bayesian-Scorer kann keine analoge Primärquelle ausgeben.

### Phase 1b: Physikalische Analogquellen-Inferenz (v10.0.0, aktualisiert v10.0.0)

Greift wenn `file_ext ∈ DIGITAL_FILE_EXTS` und Bayesian kein `best_analog` findet. Physikalische Merkmale überleben bei Kassetten/Vinyl auch nach Codec-Encoding. **Implementierung: `_infer_analog_source_from_fingerprint()` in `forensics/medium_detector.py`.**

#### Vinyl / Shellac (Disc-Quellen)

| Material | Erkennungsbedingung | Kalibrierung |
| --- | --- | --- |
| Vinyl | `infrasonic_rms > 0.030` ODER `crackle_density > 0.004 events/s` ODER `rotation_strength > 0.08` → Konfidenz berechnet; **Fallback-Gate: `rotation_strength ≥ 0.30 AND conf ≥ 0.20`** (überlebt `_pa_conf_thresh`-Dämpfung) | μ_vinyl(infrasonic)=0.08, Schwelle = μ − 1σ. Fallback deckt `rotation_strength=0.30–0.55` bei `codec_artifact=0.30–0.60` ab. |
| Shellac | `crackle_density > 0.015 AND infrasonic_rms > 0.040` (schlägt Vinyl-Erkennung) | |

**`_strong_physical_analog`-Gate** (normativ):

```python
_feature_ok = (
    fp.rotation_strength >= _pa_rot_thresh
    or fp.wow_flutter_index >= _pa_wow_thresh
    or (fp.infrasonic_rms >= _pa_infra_thresh and fp.crackle_density >= _pa_crackle_thresh)
)
# Primär: conf >= codec-adaptiver Schwellwert UND physikalisches Feature
# Fallback: rotation_strength >= 0.30 UND conf >= 0.20 (Plattenspieler-Periodizität)
_strong_physical_analog = (
    (_cand_conf >= _pa_conf_thresh and _feature_ok)
    or (_cand_conf >= 0.20 and fp.rotation_strength >= 0.30)
)
```

**VERBOTEN**: Kein Fallback-Gate → `rotation_strength=0.371` wird bei `vinyl_conf=0.250 < _pa_conf_thresh≈0.348` vollständig unterdrückt → Chain bleibt `mp3_low` (Datenmüll, §2.46b).

#### Reel-Tape (Bandmaschinen-Erkennung — zwei Pfade, IEC 60386:1987)

**Studio-Pfad** (normativ, Produktionsfall): Wenn `has_disc=True AND codec_contamination > 0.5` ist der Flutter-Bereich einer professionellen Bandmaschine (0.010–0.035 WRMS) weit unter alten Schwellwerten. Neues adaptives Gate:

```python
if has_disc and _codec_contamination > 0.5:
    _thresh = max(0.010, 0.025 * (1.0 - 0.55 * _codec_contamination))  # ≈ 0.016 bei cc=0.667
    # Rotation-Guard entfernt: Vinyl-Drehzahl ist ERWARTET wenn has_disc=True
    tape_conf = clip((wow - _thresh) / 0.10, 0.12, 0.50)
    if tape_conf >= 0.12: sources.append(("reel_tape", tape_conf))
```

**Standard-Pfad**: `wow_flutter_index > 0.20 AND rotation_strength < 0.10` (kein Disc-Source, Rotation = Tape-Motor-Artefakt unwahrscheinlich).

**Kalibrierung**: IEC 60386:1987 — Studio-Reel professionell 0.010–0.030 WRMS, Halbprofi 0.030–0.060 WRMS, Konsumerkassette 0.060–1.500 WRMS.

#### Kassette vs. reel_tape — Disambiguation (§2.46b, normativ)

Wenn beide erkannt: **wow/flutter-Schwelle trennt Studio von Consumer** (IEC 60386:1987, Pohlmann 2010):

```python
if _has_cassette and _has_reel_tape:
    if fp.wow_flutter_index < 0.06:
        sources = [(m, c) for m, c in sources if m != "cassette"]   # → reel_tape
    else:
        sources = [(m, c) for m, c in sources if m != "reel_tape"]  # → cassette
```

- `wow < 0.06 WRMS` → Studio-Bandmaschine (`reel_tape`): Präzisions-Transport, gleichmäßig
- `wow ≥ 0.06 WRMS` → Konsumerkassette (`cassette`): Capstan/Pinch-Roller-Flutter

#### Kassette (BW-basierte Erkennung)

`effective_bandwidth_hz < 15_500 AND codec_contamination < 0.30` → cassette_conf=0.55. Unterliegt Disambiguation.

#### Rückgabe

Sortierte Liste `[(material_key, confidence)]` nach Signalketten-Reihenfolge (Disc vor Band vor Codec). Konfidenzen: [0.12, 0.85].

#### Produktions-Referenzfall (Backend-Log 2026-04-21, normativ — darf nie regressieren)

| Merkmal | Messwert | Diagnose |
| --- | --- | --- |
| rotation_strength | 0.371 | Plattenspieler-Periodizität — Fallback-Gate greift (`0.371 ≥ 0.30, conf=0.250 ≥ 0.20`) |
| wow_flutter_index | 0.034 | Studio-Bandmaschine — Studio-Pfad (`0.034 > 0.016`, `cc=0.667 > 0.5`, Disambiguation: `0.034 < 0.06 → reel_tape`) |
| codec_artifact_score | 0.40 → `cc=0.667` | MP3-Encoding erkannt |
| file_ext | `.mp3` | Digital → Bayesian zeroed → Phase 1b greift |
| **Tonträgerkette** | **`vinyl → reel_tape → mp3_low`** | **Drei-stufige Restaurierung erforderlich** |

**Test-Invariante**: `tests/unit/test_vinyl_tape_mp3_chain_detection.py::TestInferAnalogSourceWithDisc::test_vinyl_reel_tape_mp3_full_chain_production_case` muss grün bleiben.

**Referenz-Fingerabdruck (Elke, Feb 2026):**

| Merkmal | Messwert | Diagnose |
| --- | --- | --- |
| infrasonic_rms | 0.065 | Vinyl-Rumble detektiert (> 0.030) |
| wow_flutter_index | 0.82 | Kassette-Flutter detektiert (≥ 0.06 WRMS → cassette, Disambiguation) |
| crackle_density | 0.006 events/s | Vinyl-Knackser detektiert (> 0.004) |
| file_ext | `.mp3` | Digital → Phase 1b physikalische Inferenz |
| Tonträgerkette | `vinyl → cassette → mp3_low` | Drei-stufige Degradation |

Rückgabe: sortierte Liste `[(material_key, confidence)]` nach Signalketten-Reihenfolge (Disc vor Band vor Codec). Konfidenzen: [0.12, 0.85].

### Phase 1c: Dolby / DBX NR-Erkennung (v10.0.0)

**Modul**: `backend/core/dolby_nr_detector.py` — `DolbyNRDetector.detect(audio, sr, material_type, era_decade)`.

Wird **automatisch** von `MediumDetector.detect()` aufgerufen wenn `primary_material ∈ {tape, reel_tape, wire_recording}`.

**Erkennungsmethode** (Frequenzband-Heuristik):

- `lf_rms` (300–1000 Hz): NR-unabhängige Referenz (Instrumenten-Grundtöne)
- `hf_rms` (800–4000 Hz + 4000–12000 Hz): Bandbereich mit stärkstem NR-Einfluss
- `hf_excess_db` = `20·log10(hf_rms / lf_rms)` − Erwartungswert_für_Material
- Slope-Analyse: DBX hat uniformen Slope (hf2 ≈ hf1), Dolby B/C wächst Richtung HF

| Typ | hf_excess Schwelle | Charakteristik |
| --- | --- | --- |
| Dolby B | ≥ 2.5 dB | +10 dB HF-Anhebung für leise Passagen; ~3-5 dB Durchschnitt |
| Dolby C | ≥ 5.0 dB | Doppelband, bis +20 dB; stark oberhalb 4 kHz |
| Dolby S | ≥ 3.5 dB | Dreifachband; ausgeprägt 2–8 kHz |
| DBX I | ≥ 4.0 dB | Breitbandig; uniformer ~9 dB/Oktave Slope |
| DBX II | ≥ 3.0 dB | Milder: ~6 dB/Oktave |

**Näherungsinversion** (statischer IIR Biquad-Kaskade):

- Dolby B: High-Shelf −4.5 dB @ 3 kHz (Q=0.6) + Peaking −2 dB @ 8 kHz (Q=0.7)
- Dolby C: High-Shelf −9 dB @ 4 kHz (Q=0.7) + Peaking −3 dB @ 10 kHz (Q=1.0)
- DBX I/II: gestaffelte High-Shelf-Kette (annähernde Steigung)

**Integration in `phase_04_eq_correction.py`**: Beide kwargs `dolby_nr_type` und `dolby_nr_confidence` werden nach RIAA/NAB-EQ angewendet. Activation via `kwargs["dolby_nr_type"] = result.dolby_nr_type`.

**`MediumDetectionResult`-Felder** (v10.0.0, erweitert v10.0.0):

- `dolby_nr_type: str = "none"` — erkannter Typ
- `dolby_nr_confidence: float = 0.0` — Konfidenz [0..1]
- `tape_speed_ips: Optional[float] = None` — erkannte Bandgeschwindigkeit (1.875 / 3.75 / 7.5 / 15 / 30 ips); None = unbekannt. Wird von Phase 1 via `wow_flutter_index` + `bandwidth_hz` + Material-Heuristik geschätzt. Aktiviert Head-Bump-Kompensation in `phase_04` (§4.5 Head-Bump).
- `riaa_curve_type: str = "riaa"` — erkannte Entzerrungskurve aus `{"riaa", "nab", "columbia", "aes", "capitol", "london", "ccir", "unknown_prestandard"}`. Default „riaa" (Standard ab 1954). Wird bei `material ∈ {vinyl, shellac, lacquer_disc}` via Spectral-Slope-Analyse gegen PRE_RIAA_EQ_CURVES (§6.3a) erkannt. Aktiviert kurvenspezifische Inverse-EQ in `phase_04`.
- `riaa_curve_confidence: float = 0.0` — Konfidenz der RIAA-Kurven-Erkennung [0..1]; ≥ 0.70 = aktiv.

**Schätz-Heuristik für `tape_speed_ips`**:

```python
# In MediumDetector._estimate_tape_speed():
if material_type not in ("tape", "reel_tape", "cassette"):
    return None
# Bandbreite → Geschwindigkeit (physikalische Beziehung: BW ∝ speed)
if bandwidth_hz < 8000:
    return 1.875   # Kompaktkassette, langsam
elif bandwidth_hz < 12000:
    return 3.75    # Standard-Kassette
elif bandwidth_hz < 16000:
    return 7.5     # Standard-Reel
elif bandwidth_hz < 20000:
    return 15.0    # Semi-Pro
else:
    return 30.0    # Profi-Master
# Zusätzliche Validierung via wow_flutter_index:
# Hoher Flutter (> 1.0) + niedrige BW → 1.875 ips mit höherer Konfidenz
```

**ACHTUNG — physikalische Limitierung**: Dies ist eine statische Näherung des level-abhängigen Dolby-Klangregelungssystems. Für Quellmaterial mit messbarer Klangverfälschung empfiehlt sich Re-Digitalisierung mit korrekt kalibriertem Wiedergabework.

### Phase 2: Transferketten-Aufbau

**Deep-Chain-Pflicht (v10.0.0)**: Kettenaufbau ist nicht auf Primärquelle + 1 Sekundärstufe begrenzt.

- Mehrere analoge Folgeglieder sind zulässig, solange die Reihenfolge kausal bleibt (`_MEDIUM_ORDER` monoton steigend).
- Digitale Zwischenstufen (`cd_digital`, `dat`) müssen eingefügt werden, wenn Posterior-Evidenz vorliegt.
- Lossy-Codec-Layer (`mp3_low`, `mp3_high`, `aac`, `streaming`) kommt am Kettenende und nur einmal.
- Nach Material-Key-Normalisierung (`cassette -> tape`, ...) werden benachbarte Duplikate konsolidiert; Link-Konfidenz bleibt der Maximalwert.

Ergebnis: `MediumDetectionResult.transfer_chain: list[str]` bildet reale 3+/4+/5+-Übertragungsketten vollständig ab.

```python
# Pflicht-Aufruf in allen Analyse-Kontexten:
from forensics.medium_detector import MediumDetector, get_medium_detector
result = get_medium_detector().detect(audio, sr, file_ext=Path(file_path).suffix)

# Kettenerkennung → MaterialType-Ableitung:
if result.transfer_chain:
    primary_material = result.transfer_chain[0]    # z. B. "vinyl"
    secondary_chain  = result.transfer_chain[1:]   # z. B. ["tape", "cd_digital", "mp3_low"]
    # → aktiviert kombinierte Phasen beider Materialien

# Kettenergebnis in RestorationResult.genealogy:
# SampleOperation(operation_type="chain_detection")
```

**VERBOTEN**: `MediumClassifier.classify_medium()` für Tonträgerketten-Erkennung. `MediumClassifier` kennt keinen Dateiendungs-Kontext und kann bei codec-enkodiertem Material "unknown" zurückgeben.

**Referenz-Fingerabdruck (Elke, Feb 2026):**

| Merkmal | Messwert | Diagnose |
| --- | --- | --- |
| infrasonic_rms | 0.065 | Vinyl-Rumble detektiert (> 0.030) |
| wow_flutter_index | 0.82 | Kassette-Flutter detektiert (> 0.30) |
| crackle_density | 0.006 events/s | Vinyl-Knackser detektiert (> 0.004) |
| file_ext | `.mp3` | Digital → Phase 1b physikalische Inferenz |
| Tonträgerkette | `vinyl → cassette → mp3_low` | Drei-stufige Degradation |

**Testpflicht (bindend)**:

- Unit-Test: 4-stufige Kette mit digitaler Zwischenstufe (`vinyl -> tape -> cd_digital -> mp3_low`).
- Unit-Test: `file_ext=.mp3` + physikalische Inferenz muss 4-stufige Kette liefern.
- Integrationstest: `source_fidelity_transfer_chain` muss unverändert bis `metadata["song_calibration"]` durchgereicht werden.

---

## §6.9 Wissenschaftliche Literaturgrundlage (normative Referenzen)

### §6.9a Tonträgerketten-Erkennung (MediumDetector, §6.7)

- **Cartwright, Pardo & Wallis (2016)** **DAFX-16** — Vinyl-Identifikation aus spektralen Merkmalen (Knisterfrequenz, Rolloff, Infrasonic-RMS): Basis der Bayesian-Feature-Kalibrierung
- **Maher (2010)** **J. Audio Eng. Soc.** 58:702 — Survey analoge Audio-Artefakt-Erkennung: methodische Grundlage für Multi-Material-Fingerabdruck
- **Declercq, De Backer & Zhu (2007)** **ICASSP** — Bayesian-Trägerklassifikation via Gaussian-Mixture: theoretische Basis des Scoring-Modells (§6.7 Phase 1)
- **IEC 60386:1987** — Wow/Flutter-Meßnorm: normative Grundlage für `wow_flutter_index`-Kalibrierung
- **Brandenburg & Bosi (1994)** **J. Audio Eng. Soc.** 42:381 — MPEG-1 Layer III (MP3): Basis der Codec-Artefakt-Erkennung (MDCT-Quantisierungs-Fingerabdruck)
- **Pan (1995)** **J. Audio Eng. Soc.** 43:529 — MPEG-2 AAC: Basis der AAC vs. MP3 Differenzierung via `codec_type_code`
- **Müller & Ewert (2011)** **IEEE Signal Proc. Mag.** 28(2):42 — Codec-Fingerprinting via MDCT-Quantisierungsartefakte: Basis des `codec_artifact_score`
- **Spijkervet & Haasdijk (2020)** **ISMIR** — ML-basierte MP3/AAC-Unterscheidung via Bitrate-Profile: Validierung der `codec_type_code`-Heuristik

### §6.9b Defekt-Erkennung (DefectScanner, §6.3/§6.4)

**Analoge Defekte:**

- **Godsill & Rayner (1998)** **Digital Audio Restoration**, Springer — Standardreferenz:
AR-Prädiktionsmodell für Clicks, Bayesian-Crackle, probabilistisches Defektmodell für alle analogen Typen
- **Janssen, Veldhuis & Vries (1986)** **IEEE TASLP** 34:203 — AR-basierte Click-Detektion via Prädiktionsfehler: Grundlage der Click/Crackle-Erkennung
- **Maher (1993)** **JASA** 93:1679 — Click- und Pop-Detektion im Zeitbereich: theoretisches Modell für Klick-Laufzeit-Diskriminator
- **Czyzewski & Kaczmarek (2003)** **AES Conv.** 115 — Parametrisches Vinyl-Wow/Flutter-Modell: Basis der Mehrband-Wow/Flutter-Detektion (`MULTIBAND_WOW_FLUTTER`) ✅
- **Bailey, Casebeer & Fazekas (2019)** **AES** 147th Conv. — Neural Network für Vinyl-Knistern-Detektion: SOTA-Validierung des Crackle-Density-Schwellwerts (4σ)
- **Esquef & Biscainho (2006)** **IEEE TASLP** 14:1207 — Modulations-Rauschen bei Bandaufnahmen: Basis von `MODULATION_NOISE` ✅
- **Dahimene, Richard & David (2008)** **IEEE TASLP** 16:757 — Dropout-Erkennung via Energietransiente: Methodik für `DROPOUTS`-Schwellwert-Kalibrierung
- **IEC 60386:1987** — Wow/Flutter-Norm (WOW < 0.5 Hz, FLUTTER 0.5–200 Hz): definiert Frequenzgrenzen von `WOW` und `FLUTTER` ✅

**Digitale Defekte:**

- **Herre & Johnston (1996)** **AES Conv.** 101 — Pre-Echo-Artefakt im MPEG-Coding durch temporales Masking-Modell: Primärquelle für `PRE_ECHO`
- **Bitto (2000)** **AES Conv.** 109 — Jitter-Messung und Perceptual Impact bei DAT/CD: Basis für `JITTER_ARTIFACTS`-Severity-Kalibrierung
- **Zölzer (2011)** **DAFX: Digital Audio Effects**, 2nd ed., Wiley — Kapitel 8 (Codec-Artefakte): Vollständige Topologie digitaler Artefakttypen; normative Basisreferenz

**Genre-/Ära-Klassifikation (GenreClassifier, EraClassifier):**

- **Pons & Serra (2022)** **ISMIR** — CNN-Spectrogramm-basierte Genre-Erkennung ohne Handfeatures: SOTA-Basis für ML-Erweiterung des GenreClassifiers
- **Won, Chun, Kim & Nam (2020)** **IEEE TASLP** — Attention-basiertes Music Tagging auf MagnaTagATune/Million Song: Basis für automatische Genre-Tag-Gewichtung
- **Li et al. (MERT, 2023/2024)** arXiv:2306.00107, ICLR 2024 — MERT: Self-supervised Music Understanding, 14 Tasks SOTA inkl. Genre: Backbone für MERT-Plugin im QualityGate ✅
- **Tsatsishvili, Pienimäki, Tervaniemi & Makkonen (2021)** **ISMIR** — Decade-Klassifikation von Popmusik via Spectrogramm-CNN: Basis der Ära-Zehnjahres-Klassifikation

**Gesangserkennung (VocalDetector, PANNs):**

- **Kong, Cao, Iqbal, Wang, Wang & Plumbley (2020)** **IEEE TASLP** — PANNs (CNN14, AudioSet): primäres Vocal-Detection-Backbone in Aurik ✅
- **Schlüter & Grill (2015)** **ISMIR** — Singing Voice Detection mit speziellem CNN: Validierung der PANNs-Konfidenz-Schwelle (0.35–0.40) für Gesangserkennung

**Instrument-Erkennung:**

- **Han, Kim & Lee (2017)** **ISMIR** — Deep-Learning Instrument Recognition (IRMAS-Datensatz): SOTA für multi-label Instrument-Tagging
- **Humphrey, Reddy & Bello (2018)** **OpenMic-2018, ISMIR** — Weakly supervised Instrument-Tagging: Basis für BEATs/PANNs Instrument-Tag-Interpretation
- **Chen, Wu & Wang (2023, BEATs)** **ICML 2023** — BEATs iter3: Audio-Tagging SOTA für Instrument-Präsenz-Detektion ✅

---

---

## §6.5 [RELEASE_MUST] Authentischer Klangcharakter vs. Defekt — Taxonomy (v10.0.0)

> **Das wichtigste Konzept für Weltklasse-Restaurierung**: Aurik muss wissen, was es BEWAHREN
> soll, nicht nur was es reparieren soll. Vintage-Charakter ist nicht dasselbe wie Schaden.
> Fehlt diese Unterscheidung, wird authentische Klangfarbe als Defekt behandelt → Zerstörung
> des originalen Klangs trotz technisch besserem Signal. §0 Primum non nocere gilt absolut.

### §6.5a Kanonische Charakter-Bewahrungsliste pro Material

Jeder Eintrag definiert: Signal-Merkmal → Klassifikation → Verarbeitungsregel.

```python
# IntentionalArtifactClassifier — normative Quelle (§2.44 Spec 03)
# Implementiert in: backend/core/intentional_artifact_classifier.py

AUTHENTIC_CHARACTER = {
    "shellac": {
        "surface_noise_texture":   "PRESERVE",  # charakteristisches 78-rpm Hintergrundrauschen;
                                                  # NR nur bis zum shellac-Boden (-45 dBFS), nicht tiefer
        "h2_h4_harmonic_saturation": "PRESERVE", # Röhren-/Kristallmikrofon-Sättigung; WärmeMetric
        "bandwidth_ceiling_8khz":  "PRESERVE",   # physikalische Grenze — KEIN BW-Extension >8 kHz
        "mono_center_image":       "PRESERVE",   # alle Shellac-Quellen sind Mono
        "soft_transients":         "PRESERVE",   # AGC-bedingte weiche Transienten (kein Transient-Shaper!)
    },
    "vinyl": {
        "groove_distortion_low":   "PRESERVE",   # leichte Rillenverzerrung < 1 % THD ist authentisch
        "interlabel_noise_texture":"PRESERVE",   # Zwillingrauschen zwischen Grooves
        "riaa_warmth_curve":       "PRESERVE",   # RIAA-Entzerrung erzeugt charakteristischen Bassanstieg
        "inner_groove_compression":"PRESERVE",   # Innenspur-Kompression ist physikalisch bedingt
        "low_level_wow_sub1hz":    "PRESERVE",   # Plattenteller-Gleichlaufschwankung < 1 Hz ist Charakter
    },
    "tape": {
        "tape_saturation_knee":    "PRESERVE",   # charakteristisches Kompressionsknie bei Übersteuerung
        "high_frequency_rolloff":  "PRESERVE",   # materialbedingt: Typ-I-Kassette rolliert ab ~12 kHz
        "bias_noise_texture":      "PRESERVE",   # Bias-Rauschen (HF-Pfeifen > 18 kHz ist KEIN Defekt)
        "dolby_breathing_slight":  "PRESERVE",   # leichtes Dolby-Atmen < -3 dB ist Epochencharakter
        "tape_compression_even_harmonics": "PRESERVE",  # Bandsättigung fügt H2 hinzu: Wärme-Merkmal
    },
    "reel_tape": {
        "studio_ambience_bleed":   "PRESERVE",   # Raumrauschboden des Aufnahmeraums ist Teil der Aufnahme
        "tape_hiss_floor_texture": "PRESERVE",   # NR nur bis zum reel_tape-Boden (-60 dBFS)
        "print_through_ghost":     "REPAIR",     # Print-Through (Vor-/Nachhall Spur auf Spur) = echter Defekt
        "tape_head_clog":          "REPAIR",     # Kopfverstopfung = echter Defekt
    },
    "wax_cylinder": {
        "trichter_bandlimit_3khz": "PRESERVE",   # akustische Aufnahme: BW ≤ 3 kHz ist die REALITÄT
        "surface_crackle_fine":    "PRESERVE",   # feines Oberflächengeräusch ist Teil der Epoche
        "mechanical_resonance":    "PRESERVE",   # Trichtereigenresonanz ≈ 300–600 Hz ist Charakteristik
        "coarse_crackle_clicks":   "REPAIR",     # grobe Klicker > 10 ms sind Defekte
    },
    "cd_digital": {
        "dithering_noise_floor":   "PRESERVE",   # 16-bit-Dithering-Rauschen ist Teil des Formats
        "pre_emphasis_curve":      "PRESERVE",   # wenn Pre-Emphasis aktiv war: originalgetreu
        "linear_phase_character":  "PRESERVE",   # CD hat nah-perfekte Linearphase — kein Phasenediting
        "quantization_artifacts_mild": "PRESERVE",  # leichte Quantisierungsartefakte < -90 dBFS: normal
    },
    "mp3_low": {
        "pre_echo_character_mild": "PRESERVE",   # leichtes Pre-Echo bei stabiler Musik ist Format-Charakter
                                                  # NICHT als Knacken klassifizieren!
        "psychoacoustic_residue":  "PRESERVE",   # wahrnehmungspsychologisches Kodierresiduum: Epoche
        "severe_pre_echo":         "REPAIR",     # starkes Pre-Echo > 40 ms ist echter Defekt (phase_50)
        "metallic_ringing":        "REPAIR",     # metallisches Klingen ist echter Defekt
    },
    "lacquer_disc": {
        "substrate_texture":       "PRESERVE",   # Acetat-Substrat-Textur ist Materialcharakter
        "light_surface_clicks":    "PRESERVE",   # leichte Oberflächenklicker ≤ 3 ms: Epoche
        "deep_crack_clicks":       "REPAIR",     # tiefe Rissklicker > 5 ms: echter Defekt
    },
}

# Klassifikations-Entscheidung:
# "PRESERVE": Kein Eingriff. Bei versehentlicher Aktivierung durch DefectScanner:
#             Strength auf max. 0.10 begrenzen.
# "REPAIR":   Voller Eingriff erlaubt (CAUSE_TO_PHASES normal aktiv).
# "AMBIGUOUS": Stärkeabhängig. Severity < 0.3 → PRESERVE. Severity ≥ 0.3 → REPAIR.
```

### §6.5b Implementierungs-Invarianten

```python
# IntentionalArtifactClassifier (Spec 03 §2.44) muss VOR DefectScanner
# in der CAUSE_TO_PHASES-Auswahl aktiv sein.
# Aufruf: classifier.classify(material_type, era_decade=None, artifact_freedom=1.0) → IntentionalArtifactResult
# Rückgabe: .preserve_features, .repair_features, .ambiguous_features, .strength_caps{feature→max_strength}
# KEIN audio/sr Parameter — Klassifikation basiert auf material_type + era_decade (DSP-frei)
# VERBOTEN: PRESERVE-Klasse durch Phase bearbeiten, die Strength > 0.10 hat
# VERBOTEN: material-agnostische NR auf Shellac/Wax-Cylinder (zerstört Epochencharakter)

# Priorität: PRESERVE-Klassifikation schlägt DefectScanner-Aktivierung.
# Ausnahme: Wenn artifact_freedom < 0.90 durch PRESERVE-Merkmal verursacht wird
#           (z.B. extreme Shellac-Sättigung maskiert Vokalqualität),
#           darf Strength bis 0.20 angehoben werden — mit HNR-Blend-Pflicht.
```

### §6.5c Era-spezifische Charakter-Profile (normative Verarbeitungsrichtlinien)

| Ära | Authentischer Charakter | Maßnahme bei Fehlklassifikation |
| --- | --- | --- |
| 1900–1925 | Trichterresonanz 300–600 Hz, BW-Ceiling ≤ 3 kHz, mono | Kein EQ unter 300 Hz; keine BW-Extension |
| 1925–1945 | Röhren-H2/H4, AGC-Drift, elektrische Brumm-Teppiche | H2/H4 PRESERVE; Brumm nur bei f ≤ 120 Hz reparieren |
| 1945–1965 | RIAA-Wärme, Raumdiffusion Aufnahmestudio, Nadelgeräusch | RIAA-Kurve NIE als Defekt klassifizieren |
| 1965–1980 | Bandkompression, Bandbreite 12–14 kHz, Analog-Hiss | Hiss-Boden PRESERVE bis reel_tape-Schwellwert |
| 1980–1995 | Dolby-B/C Rauschen, Kassettenklang, frühe Digital-Kanten | Dolby-Atmen PRESERVE wenn < -3 dB relativ zu Signal |
| 1995–2010 | MP3-Psychoakustik-Residuum, leichte Pre-Echos | Mild Pre-Echo als Format-Signatur PRESERVE |
| 2010+ | Clean Digital; Loudness-War-Clipping = echter Defekt | Clipping REPAIR; Rauschen < -90 dBFS PRESERVE |

---

## §6.6 [RELEASE_MUST] RIAA-Kurven-Klassifikation — normative Metriken (v10.0.0)

> **LÜCKE geschlossen**: §6.3a hatte Zeitkonstanten aber keine Klassifikations-Metriken.

### §6.6a Spektral-Slope-Grenzwerte pro RIAA-Kurve

Messung: Spektraler Tilt in dB/Oktave über Bereich 250–8000 Hz (Fenster: Hann, FFT 4096 bei 48 kHz).

```python
# Toleranzband: ±1.0 dB/oct (±1.5 bei SNR < 10 dB)
RIAA_SLOPE_PROFILES = {
    # name:       (slope_dB_oct, bass_boost_db_at_100hz, hf_cut_freq_hz, bass_turnover_hz)
    "riaa":       (-5.0,  +13.7,  2122, 3183),   # IEC 1994 Standard (τ: 75/318/3180 µs)
    "nab":        (-4.5,  +13.7,  1590, 3183),   # NAB Broadcast (τ: 100/318/3180 µs)
    "columbia":   (-3.5,  +16.0,  1590, 1590),   # Columbia Records 1948 (Bass-betont)
    "aes":        (-4.5,  +12.0,  3183, 3183),   # AES 1951 (weniger Bass-Boost)
    "capitol":    (-3.7,  +15.0,  1590, 2122),   # Capitol Records 1949
    "london":     (-5.2,  +13.7,  2122, 3183),   # London/Decca (mehr HF-Boost als RIAA)
    "ccir":       (-4.9,  +13.7,  3183, 3183),   # CCIR/EBU Rundfunk
    "unknown":    None,                           # Bayes-Prior: uniform über alle Kurven
}

# Klassifikations-Algorithmus (Bayes):
# P(curve | audio) ∝ P(audio | curve) × P(curve | era_decade)
# Likelihood aus: slope_match + bass_turnover_match + hf_rolloff_match
# Konfidenz-Schwelle: ≥ 0.70 → Kurve bestätigt; < 0.70 → "unknown" (konservativ)
# Bei era_decade ≤ 1950: Prior für columbia/capitol/aes erhöht (×2.5)
# Bei era_decade ≥ 1960: Prior für riaa erhöht (×3.0), columbia/capitol/aes reduziert

def classify_riaa_curve(audio, sr, era_decade):
    """Bayesianische RIAA-Kurvenklassifikation."""
    slope = _measure_spectral_slope(audio, sr, f_low=250, f_high=8000)
    bass_boost = _measure_bass_boost_at_100hz(audio, sr)
    hf_turnover = _find_hf_turnover_freq(audio, sr)

    likelihoods = {}
    for curve, (s, b, h, bt) in RIAA_SLOPE_PROFILES.items():
        if s is None:
            likelihoods[curve] = 0.1  # uniform Prior für unknown
            continue
        d_slope = abs(slope - s) / 1.0        # Normiert auf Toleranz 1 dB/oct
        d_bass  = abs(bass_boost - b) / 3.0   # Normiert auf Toleranz 3 dB
        d_hf    = abs(hf_turnover - h) / 500  # Normiert auf Toleranz 500 Hz
        likelihoods[curve] = np.exp(-0.5 * (d_slope**2 + d_bass**2 + d_hf**2))

    era_priors = _get_era_riaa_priors(era_decade)
    posteriors = {k: likelihoods[k] * era_priors.get(k, 1.0) for k in likelihoods}
    posteriors = {k: v / sum(posteriors.values()) for k, v in posteriors.items()}

    best_curve = max(posteriors, key=posteriors.get)
    return best_curve if posteriors[best_curve] >= 0.70 else "unknown"
```

---

## §6.8 Importformate (Eingang)

| Format | Erweiterungen |
| --- | --- |
| WAV / AIFF | `.wav`, `.aiff`, `.aif` |
| FLAC | `.flac` |
| MP3 | `.mp3` |
| AAC / M4A | `.aac`, `.m4a`, `.mp4` |
| OGG Vorbis | `.ogg` |
| WMA | `.wma` |
| Opus | `.opus` |
| CAF | `.caf` |

**Invarianten:**

- Alle Formate → intern float32, 48 000 Hz, Stereo oder Mono
- Maximale Dateigröße: 10 GB (darüber Chunk-Modus)
- > 2 Kanäle → PANNs-gewichteter Stereo-Downmix
- Ungültige Dateien: `AudioLoadError` + Deutsch-Meldung, kein Absturz

---

## §6.9 SourceMediumProfile-Integration (§v10.706, v10.17.0)

Die zentrale Registratur `SourceMediumProfile` definiert physikalische Limiten
PRO Trägermedium. Seit v10.17.0 werden diese Limiten in der Kalibrierung und in
mehreren Phasen durchgesetzt:

| SMP-Feld | Genutzt von | Mechanismus |
|----------|------------|-------------|
| `max_bandwidth_hz` | Phase_06, Phase_23 | Bandwidth-Cap für Frequenz-Erweiterung |
| `hiss_reduction_max_strength` | Phase_29 | Hard-Cap: `min(depth_cap, smp_cap)` |
| `harmonic_max_order` | Phase_07 | DDSP synthetisiert nur bis phys. Limit |
| `is_compressed` | Phase_26, Phase_54, SongCalibration | Dynamics-Expansion reduziert/gedeckelt |
| `has_soft_saturation` | Phase_36 | Transient-Shaper-Stärke −20% |
| `deesser_skip_saturation_conf` | SongCalibration (B11) | Vocal-Familien-Skalar reduziert |

### Drei-Schicht-Material-Intelligenz (§v10.706)

1. **Defekt-Erkennung**: DefectScanner mit chain-adaptiven Thresholds — 46+ Typen
2. **Phasen-Selektion**: Chain-Injection + PID Material-Fremdlauf-Detektion (INFO)
3. **Physikalische Kalibrierung**: SourceMediumProfile → SongCalibration → family_scalars

### Denker-IQ: Chain-Depth-adaptives Budget (§v10.706)

StrategieDenker skaliert das Zeitbudget mit der Tonträgerketten-Tiefe:

- depth=1: Budget ×1.00
- depth=4: Budget ×1.45 (Cassette-Kette)
- depth=5: Budget ×1.60
- Zusätzlich: Restorability-Faktor (rs=64 → +18%)
