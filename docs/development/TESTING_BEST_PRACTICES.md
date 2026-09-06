# Aurik 10.0.8 — Testing Best Practices

**Version:** 10.0.8  
**Datum:** Mai 2026  
**Status:** ✅ Production Ready

> **Normativer Hinweis (Release):** Produktive Pfade werden ueber Bridge + `AurikDenker.denke(...)`
> und nachgelagertes `export_guard()` abgesichert. Historische v2-Testbeispiele in diesem
> Dokument sind als `LEGACY_NON_RELEASE` einzuordnen.

---

## 📖 Inhaltsverzeichnis

- [Übersicht](#übersicht)
- [Musical Goals Testing](#musical-goals-testing)
- [Adaptive Thresholds](#adaptive-thresholds)
- [Material Quality Simulation](#material-quality-simulation)
- [Vocal Enhancement Testing](#vocal-enhancement-testing)
- [Forensic Analysis & Media Types](#forensic-analysis--media-types)
- [Test Markers](#test-markers)
- [Performance-Optimierung](#performance-optimierung)
- [Häufige Fehler](#häufige-fehler)

---

## Übersicht

Aurik-Tests müssen die **15 Musical Goals** (Spec 01) und das **Adaptive Thresholds System** berücksichtigen, um produktive und musikalisch sinnvolle Tests zu gewährleisten.

### Kernprinzipien

✅ **Musikalische Exzellenz** = Primärziel  
✅ **Adaptive Thresholds** = Material-Quality-aware  
✅ **Generation-Count** = 40% Gewichtung im Degradation Score  
✅ **Keine False Positives** bei degradiertem Material  

---

## Musical Goals Testing

### Die 14 Musical Goals (Stand v10.0.0.x)

1. **Brillanz** (HF Clarity 8–20 kHz) — Threshold: 0.85
2. **Wärme** (Mid-Range 200–2000 Hz) — Threshold: 0.80
3. **Natürlichkeit** (Gesamtklang) — Threshold: 0.90 **(Stufe 1 — Rollback)**
4. **Authentizität** (Klangidentität) — Threshold: 0.88 **(Stufe 1 — Rollback)**
5. **Emotionalität** (Dynamics & Expression) — Threshold: 0.87
6. **Transparenz** (Clarity & Separation) — Threshold: 0.89
7. **Bass-Kraft** (20–250 Hz, inkl. Virtual Pitch) — Threshold: 0.85
8. **Groove** (DTW ≤ 8 ms RMS) — Threshold: 0.88
9. **Raumtiefe** (IACC Blauert 1997) — Threshold: 0.75
10. **Timbre-Authentizität** (MFCC-Pearson ≥ 0.95) — Threshold: 0.87
11. **Tonales Zentrum** (Chroma-Korrelation ≥ 0.95) — Threshold: 0.95
12. **Mikro-Dynamik** (LUFS-Profil Pearson ≥ 0.92) — Threshold: 0.92
13. **Separation-Treue** (SDR ≥ 8 dB) — Threshold: 0.82
14. **Artikulation** (Transient-Shape-Korrelation ≥ 0.90) — Threshold: 0.85

> **Adaptive Thresholds**: Schwellwerte werden vor jeder Restaurierung material- und restorability-adaptiv skaliert (§2.31). Statische Schwellwerte allein sind verboten.

### Beispiel: Musical Goals Test

```python
@pytest.mark.integration
def test_restore_musical_goals():
    """Test: Restoration preserves all 14 Musical Goals"""
    # Generate test audio
    sr = 48000
    audio = generate_harmonic_test_signal(sr, duration=3.0)

    # Process
    denker = get_aurik_denker_instance()
    restored = denker.denke(audio, sr, mode="restoration").audio

    # Measure Musical Goals
    checker = MusicalGoalsChecker()
    goals = checker.measure_all(restored, sr)

    # Use Adaptive Thresholds (Material Quality-aware)
    thresholds = checker.thresholds

    # Validate
    for goal_name, score in goals.items():
        threshold = thresholds.get(goal_name, 0.0)
        assert score >= threshold, f"{goal_name}: {score:.3f} < {threshold:.2f}"
```

---

## Adaptive Thresholds

### Warum Adaptive Thresholds?

**Problem:** Fixe Thresholds (z.B. 0.85 für Brillanz) funktionieren nicht für alle Material-Qualitäten:

- ✅ PRISTINE Studio-Material: 0.85 ist erreichbar
- ❌ EXTREME degradiertes Material (3+ Generationen): 0.85 ist unrealistisch

**Lösung:** Adaptive Thresholds passen sich der Material-Qualität an:

- PRISTINE (degradation < 0.05): High thresholds (0.85-0.90)
- GOOD (0.15-0.25): Standard thresholds (0.80-0.85)
- POOR (0.35-0.50): Relaxed thresholds (0.50-0.65)
- EXTREME (> 0.70): Minimal thresholds (0.30-0.45)

### Material Quality Levels

```python
class MaterialQuality(Enum):
    PRISTINE = "pristine"        # Studio-Qualität, unbearbeitet
    EXCELLENT = "excellent"      # Leichte Bearbeitung
    GOOD = "good"                # Standard Digital/CD
    FAIR = "fair"                # MP3 192kbps
    POOR = "poor"                # MP3 128kbps, Cassette
    VERY_POOR = "very_poor"      # Stark degradiert
    EXTREME = "extreme"          # Telefon, Multi-Generation
```

### Generation-Count Weighted Scoring

**WELTSPITZE: 40% Gewichtung auf Generation-Count!**

```python
# Degradation Score Calculation
degradation = (
    0.12 * noise_level +
    0.12 * bandwidth_limitation +
    0.06 * artifact_density +
    0.08 * dr_penalty +
    0.40 * gen_penalty +           # 40% GEWICHTUNG!
    0.22 * defects_severity_score  # ML-basiert (98%+ Recall)
)
```

**Generation-Count Impact:**

- 1 Generation = 0.20 degradation → FAIR
- 2 Generationen = 0.40 degradation → POOR
- 3 Generationen = 0.60 degradation → VERY_POOR
- 4 Generationen = 0.80 degradation → VERY_POOR
- 5+ Generationen = 1.00 degradation → EXTREME

### Beispiel: Adaptive Thresholds Test

```python
@pytest.mark.integration
def test_adaptive_thresholds_degraded_material():
    """Test: Adaptive Thresholds relax for degraded material"""
    # Generate degraded audio (heavy noise, bandwidth limited)
    sr = 48000
    audio = generate_degraded_signal(sr, noise_level=0.15, bandwidth_limit=0.7)

    # Process
    denker = get_aurik_denker_instance()
    restored = denker.denke(audio, sr, mode="restoration").audio

    # Validate on degraded material without assuming legacy internals
    material = analyze_material(audio, sr)

    # Material should be identified as degraded
    assert material.quality_level in [
        MaterialQuality.POOR,
        MaterialQuality.VERY_POOR,
        MaterialQuality.EXTREME
    ], f"Degraded material not identified: {material.quality_level.value}"

    # Thresholds should be relaxed
    adaptive_thresholds = restorer._adaptive_thresholds
    standard_thresholds = MusicalGoalsChecker().thresholds

    relaxed_count = 0
    for goal_name, adaptive_threshold in adaptive_thresholds.items():
        standard_threshold = standard_thresholds.get(goal_name, 0.0)
        if adaptive_threshold < standard_threshold:
            relaxed_count += 1

    assert relaxed_count > 0, \
        "Adaptive Thresholds should be relaxed for degraded material"
```

---

## Material Quality Simulation

### Test-Audio für verschiedene Quality Levels

#### PRISTINE Material

```python
def generate_pristine_audio(sr=48000, duration=3.0):
    """Clean studio-quality audio"""
    t = np.linspace(0, duration, int(sr * duration))

    # Multi-frequency with harmonics
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # Fundamental
    audio += 0.1 * np.sin(2 * np.pi * 880 * t)  # 2nd harmonic
    audio += 0.05 * np.sin(2 * np.pi * 1320 * t)  # 3rd harmonic

    # Subtle dynamics
    envelope = 1.0 + 0.1 * np.sin(2 * np.pi * 2.0 * t)
    audio = audio * envelope

    return audio
```

#### DEGRADED Material

```python
def generate_degraded_audio(sr=48000, duration=3.0):
    """Heavily degraded audio (simulates cassette → MP3 → digital)"""
    t = np.linspace(0, duration, int(sr * duration))

    # Base signal
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)

    # Heavy noise (poor SNR)
    noise = np.random.randn(len(audio)) * 0.15
    audio += noise

    # Artifacts (clicks, dropouts)
    artifact_positions = np.random.choice(len(audio), size=50, replace=False)
    for pos in artifact_positions:
        audio[pos] = 0.7

    # Bandwidth limitation would be applied here
    # (spectral filtering to simulate MP3)

    return audio
```

#### EXTREME Material

```python
def generate_extreme_degraded_audio(sr=48000, duration=3.0):
    """Extremely degraded (telephone, multi-generation)"""
    t = np.linspace(0, duration, int(sr * duration))

    # Severely bandwidth-limited
    audio = 0.2 * np.sin(2 * np.pi * 440 * t)

    # Extreme noise
    noise = np.random.randn(len(audio)) * 0.25
    audio += noise

    # Heavy artifacts
    artifact_positions = np.random.choice(len(audio), size=200, replace=False)
    for pos in artifact_positions:
        audio[pos] = np.random.randn() * 0.8

    return audio
```

---

## Vocal Enhancement Testing

### Übersicht

Aurik 10.0.0 verfügt über ein **geschlechts- und alters-spezifisches Vocal Enhancement System** mit **individueller Sibilanten-Beseitigung** (De-Essing).

**Location:** `dsp/aurik_deesser_pro/music_vocal_pipeline.py`

### Die 3 Vocal Profiles

```python
VOCAL_PROFILES = {
    "female": {
        "s_band": (7000.0, 11000.0),    # Höhere Sibilanten-Frequenzen
        "max_depth_db": -3.5,           # Aggressive De-Essing
        "avg_burst_ms": 35.0,
        "allow_ml": True,
    },
    "male": {
        "s_band": (5000.0, 9000.0),     # Niedrigere Sibilanten-Frequenzen
        "max_depth_db": -2.5,           # Moderate De-Essing
        "avg_burst_ms": 45.0,
        "allow_ml": True,
    },
    "child": {
        "s_band": (9000.0, 13000.0),    # Höchste Frequenzen
        "max_depth_db": -4.0,           # Sehr aggressive De-Essing
        "avg_burst_ms": 30.0,
        "allow_ml": True,
    },
}
```

**WHY Gender-Specific Processing?**

- Weibliche Stimmen: Höhere Formanten + höhere Sibilanten-Frequenzen (7-11 kHz)
- Männliche Stimmen: Tiefere Formanten + niedrigere Sibilanten-Frequenzen (5-9 kHz)
- Kinder: Höchste Formanten + höchste Sibilanten-Frequenzen (9-13 kHz)

### 3-Pass De-Essing System

#### Pass 1: FIR-basiertes De-Essing

```python
def pass1_fir_deess(audio, events, profile, sr=48000):
    """
    Gezieltes Bandpass-Filtering der Sibilanten-Frequenzen.
    - Schnell und präzise
    - Erhält Formanten
    """
    # FIR Filter design
    taps = firwin(257, [s_band[0]/nyq, s_band[1]/nyq], pass_zero=True)
    filtered = lfilter(taps, 1.0, audio)

    # Gain reduction nur bei Sibilant-Events
    gain = 10 ** (profile.max_depth_db / 20)
    out = audio.copy()
    for ev in events:
        out[ev.start:ev.end] -= filtered[ev.start:ev.end] * (1 - gain)

    return out
```

**Musical Goal:** ✅ Brillanz (HF Clarity) - Keine Überdämpfung!

#### Pass 2: Spectral Repair

```python
def pass2_spectral_repair(audio, events, profile, sr=48000):
    """
    STFT-basierte Interpolation von Sibilant-Events.
    - Repariert Artefakte aus Pass 1
    - Erhält natürlichen Klang
    """
    stft = librosa.stft(audio, n_fft=4096, hop_length=512)

    # Interpoliere Sibilant-Frames
    for ev in events:
        t = ev.start // 512
        if 1 <= t < stft.shape[1] - 1:
            stft[band, t] = 0.5 * (stft[band, t-1] + stft[band, t+1])

    return librosa.istft(stft, hop_length=512)
```

**Musical Goal:** ✅ Natürlichkeit (Natural Sound)

#### Pass 3: ML-based HF Texture (optional)

```python
def pass3_hf_texture_ml(audio, events, profile, model, sr=48000):
    """
    Neuronales Netz für HF-Textur-Verfeinerung.
    - Nur wenn allow_ml: True
    - Verbessert Brillanz und Transparenz
    """
    # ML-Modell für HF-Textur-Prediction
    # ...
```

**Musical Goal:** ✅ Authentizität (Voice Identity >= 0.88)

### Quality Gates

#### HF Preservation Gate

```python
def adaptive_hf_gate(ratio, style="pop"):
    """
    Prüft, ob HF-Energie ausreichend erhalten wurde.

    Stil-abhängige Thresholds:
    - pop: ratio >= 0.85
    - classical: ratio >= 0.90
    - rock: ratio >= 0.80
    """
    thresholds = {"pop": 0.85, "classical": 0.90, "rock": 0.80}
    return ratio >= thresholds.get(style, 0.85)
```

#### Correlation Gate

```python
def adaptive_corr_gate(corr, min_corr=0.98):
    """
    Prüft Voice Identity Preservation.

    min_corr = 0.98: SEHR STRIKT!
    Sichert Authentizität (höchstes Musical Goal)
    """
    return corr >= min_corr
```

### Beispiel: Vocal Enhancement Test

```python
@pytest.mark.integration
@pytest.mark.musical_goals
@pytest.mark.vocal_enhancement
@pytest.mark.parametrize("gender", ["female", "male", "child"])
def test_vocal_enhancement_preserves_brillanz(gender):
    """
    Test: Vocal Enhancement preserves Brillanz (HF Clarity 8-20 kHz)

    WHY: De-Essing must reduce harsh sibilants WITHOUT destroying HF clarity

    Musical Goal: Brillanz >= 0.85 (or adaptive threshold for degraded material)
    """
    sr = 48000
    duration = 3.0

    # Generate test audio with harsh sibilants
    audio_degraded = generate_degraded_vocal_signal(sr, duration, gender)

    # Process with Vocal Enhancement
    audio_enhanced = process_vocals(audio_degraded, sr, gender=gender)

    # Measure Musical Goals
    checker = MusicalGoalsChecker()
    goals_after = checker.measure_all(audio_enhanced, sr)

    # Brillanz should be preserved
    brillanz = goals_after.get('brillanz', 0.0)

    # CRITICAL: De-Essing should NOT destroy Brillanz
    assert brillanz >= 0.70, \
        f"{gender}: Brillanz too low after de-essing: {brillanz:.3f} < 0.70"
```

### Artifact & Bias Detection

Nach jedem Pass wird automatisch auf Artefakte und Bias geprüft:

```python
# Nach Pass 1: FIR-DeEss
clipping = detect_clipping(audio1)
dc_offset = detect_dc_offset(audio1)
bias, bias_band = detect_bias(audio1, sr)

write_audit_log({
    "step": "artifact_bias_detection",
    "after": "fir_deess",
    "clipping": clipping,
    "dc_offset": dc_offset,
    "bias": bias,
    "bias_band": bias_band,
})
```

**WHY:** Bias-Detection sichert **Diskriminierungsfreiheit** und **Fairness**.

### Gender Detection

Automatische Geschlechtsbestimmung mit Fallback:

```python
# Automatische Gender-Detection
if gender is None or gender == "auto":
    if GENDER_DETECTION_AVAILABLE:
        detector = GenderDetector()
        gender = detector.detect_gender(audio)
    else:
        # Fallback: Spektrale Analyse
        gender = "unknown"
        profile = analyze_track(audio, sr, gender="unknown")
```

**Policy:** Profile-Auswahl nur nach **musikalischen Kriterien**, nicht nach Stereotypen!

### Test-Audio für Vocal Enhancement

#### Clean Vocal Signal

```python
def generate_vocal_test_signal(sr=48000, duration=3.0, gender="female"):
    """Generate realistic vocal signal with natural sibilants"""
    t = np.linspace(0, duration, int(sr * duration))

    # Fundamental frequency (gender-specific)
    f0 = 220 if gender == "female" else 110 if gender == "male" else 330

    # Vocal with harmonics
    audio = 0.4 * np.sin(2 * np.pi * f0 * t)
    audio += 0.2 * np.sin(2 * np.pi * 2 * f0 * t)  # 2nd harmonic
    audio += 0.1 * np.sin(2 * np.pi * 3 * f0 * t)  # 3rd harmonic

    # Add natural sibilants
    sibilant_freq = 8000 if gender == "female" else 6000 if gender == "male" else 10000
    # ... (add sibilant bursts)

    return audio
```

#### Degraded Vocal Signal

```python
def generate_degraded_vocal_signal(sr=48000, duration=3.0, gender="female"):
    """Generate vocal with harsh sibilants and noise"""
    audio = generate_vocal_test_signal(sr, duration, gender)

    # Add harsh sibilants (excessive HF energy)
    harsh_freq = 9000 if gender == "female" else 7000 if gender == "male" else 11000
    # ... (add harsh sibilant bursts)

    # Add noise
    noise = np.random.randn(len(audio)) * 0.05
    audio += noise

    return audio
```

### E2E: All Musical Goals Validation

```python
@pytest.mark.e2e
@pytest.mark.vocal_enhancement
def test_e2e_vocal_enhancement_all_musical_goals():
    """
    Test: E2E Vocal Enhancement validates ALL 14 Musical Goals

    Validates: Brillanz, Wärme, Natürlichkeit, Authentizität,
               Emotionalität, Transparenz, Bass-Kraft
    """
    # Process with Vocal Enhancement
    audio_enhanced = process_vocals(audio_degraded, sr, gender="female")

    # Measure ALL Musical Goals
    checker = MusicalGoalsChecker()
    goals = checker.measure_all(audio_enhanced, sr)

    # Adaptive Thresholds für degradiertes Material
    relaxed_thresholds = {
        'brillanz': 0.70,
        'natuerlichkeit': 0.70,
        'authentizitaet': 0.75,  # Höchster Threshold!
        # ...
    }

    # Validate all goals
    for goal_name, score in goals.items():
        threshold = relaxed_thresholds.get(goal_name, 0.70)
        assert score >= threshold
```

### Cross-Gender Consistency (Bias-Free Testing)

```python
@pytest.mark.e2e
@pytest.mark.vocal_enhancement
def test_e2e_vocal_enhancement_cross_gender_consistency():
    """
    Test: Equivalent quality improvement für alle Geschlechter

    POLICY: Keine Diskriminierung basierend auf Geschlecht
    """
    results = {}

    for gender in ["female", "male", "child"]:
        audio_enhanced = process_vocals(audio, sr, gender=gender)

        # Measure improvements
        improvements = calculate_improvements(audio_before, audio_enhanced)
        results[gender] = improvements

    # Improvements sollten konsistent sein
    improvement_std = np.std([r["avg"] for r in results.values()])
    assert improvement_std < 0.15, "Inconsistent results across genders"
```

---

## Forensic Analysis & Media Types

### Übersicht

Aurik 10.0.0 unterstützt **30+ analoge und digitale Tonträger** mit **Medium-spezifischen Musical Goals Thresholds** und **Forensischer Analyse**.

**Locations:**

- `forensics/unified_analyzer.py` - ML-basierte Forensic Analysis
- `forensics/signatures.py` - 30+ MediaType Definitionen
- `tests/test_comprehensive_media_types_forensics.py` - Comprehensive Tests

### Die 8 Tonträger-Hauptkategorien

1. **MECHANICAL** (1900-1930): Zylinder, Schellack
2. **VINYL** (1940-1990): LP, Singles
3. **TAPE_REEL** (1950-1990): 30/15/7.5 IPS
4. **TAPE_CASSETTE** (1960-2000): Type I/II/IV, Dolby
5. **DIGITAL_PCM** (1980-heute): CD, DAT, SACD, Hi-Res
6. **DIGITAL_LOSSY** (1995-heute): MP3, AAC, Opus
7. **BROADCAST** (1920-heute): AM, FM, DAB
8. **TELEPHONE** (1900-heute): PSTN, GSM, VoIP

### ⚠️ Aurik 10.0.0 Scope: Mono/Stereo Only

**WICHTIG:** Aurik 10.0.0 unterstützt ausschließlich **Mono- und Stereo-Formate** für **Processing**.

**❌ NICHT UNTERSTÜTZT (Processing):**

- **Surround/Multichannel:** 5.1, 7.1, Quadraphonic Surround
- **Immersive 3D Audio:** Dolby Atmos, DTS:X, 360 Reality Audio, Spatial Audio
- **Multichannel-Formate:** Jegliche Formate mit > 2 Kanälen

**✅ UNTERSTÜTZT (Processing):**

- **Mono:** 1 Kanal (historische Aufnahmen, Telefonie)
- **Stereo:** 2 Kanäle (Standard für Musik, Radio, Streaming)
- **Dual Mono:** 2 identische Kanäle

**🔍 FORENSIC DETECTION vs. PROCESSING:**

Die **Forensic Analysis** kann Multichannel-Formate **ERKENNEN** (VINYL_LP_QUAD, DTS 5.1, etc.), aber Aurik wird die **Verarbeitung ablehnen**:

```python
# Forensic Analysis: ERKENNT Quadraphonic
result = analyzer.analyze(audio_quad)
assert result.medium == MediaType.VINYL_LP_QUAD  # ✅ Detection works

# Processing: LEHNT AB (> 2 Kanäle)
try:
    processed = restorer.process(audio_quad)
except ValueError as e:
    assert "Only mono/stereo supported" in str(e)  # ✅ Expected
```

**REASON:** Aurik's Musical Goals und Adaptive Thresholds sind für stereophones Audio-Mastering optimiert. Multichannel-Formate erfordern räumliche Analysetechniken, die außerhalb des Scopes liegen.

### Medium-Spezifische Musical Goals

**WHY?** Jedes Medium hat unterschiedliche technische Limitierungen:

```python
# Mechanical Era: Extreme Einschränkungen
"CYLINDER_EDISON": {
    "brillanz": 0.30,       # < 3 kHz Bandwidth
    "waerme": 0.85,          # Mid-Range betont
    "authentizitaet": 0.90,  # Historisch wichtig!
}

# Vinyl: Wärme ist KEY
"VINYL_LP_STEREO": {
    "brillanz": 0.75,
    "waerme": 0.85,          # VINYL = WÄRME!
    "authentizitaet": 0.85,
}

# Digital Hi-Res: Höchste Qualität
"SACD_DSD": {
    "brillanz": 0.95,
    "waerme": 0.80,
    "transparenz": 0.95,
}

# Lossy: Starke Einschränkungen
"MP3_128": {
    "brillanz": 0.50,        # Starke HF-Verluste
    "natuerlichkeit": 0.55,
}
```

### Forensic Analysis System

#### 3-stufige ML-basierte Analyse

```python
class UnifiedForensicAnalyzer:
    """
    1. Medium Detection (6 Kategorien, 99%+ Ziel-Accuracy)
       → VINYL, TAPE, CASSETTE, CD, DIGITAL, LOSSY

    2. Era Detection (8 Epochen, 95%+ Ziel-Accuracy)
       → 1950s, 1960s, 1970s, 1980s, 1990s, 2000s, 2010s, 2020s

    3. Defect Detection (5 Typen, 98%+ Recall)
       → Clicks, Crackle, Hum, Wow/Flutter, Bandwidth Limit
    """
```

#### Transfer Chain Detection

```python
# Erkennt Multi-Generation Transfers
chain = [
    MediaType.VINYL_LP_STEREO,   # Original
    MediaType.CASSETTE_TYPE_I,    # 1. Generation
    MediaType.MP3_128             # 2. Generation
]

# Jeder Schritt = Quality Loss!
# Generation-Count hat 40% Gewichtung im Degradation Score
```

### Beispiel: Medium-Specific Test

```python
@pytest.mark.forensic_analysis
@pytest.mark.musical_goals
def test_medium_specific_thresholds():
    """Test: VINYL hat andere Thresholds als CD"""

    # VINYL: Wärme wichtig
    vinyl_thresholds = MEDIUM_SPECIFIC_THRESHOLDS["VINYL_LP_STEREO"]
    assert vinyl_thresholds["waerme"] == 0.85  # Hoch!

    # CD: Transparenz wichtig
    cd_thresholds = MEDIUM_SPECIFIC_THRESHOLDS["CD_STANDARD"]
    assert cd_thresholds["transparenz"] == 0.88  # Hoch!

    # VINYL > CD für Wärme
    assert vinyl_thresholds["waerme"] > cd_thresholds["waerme"]
```

### Fehlende Tonträger (TODOs)

**Moderne Lossless:**

- ❌ FLAC, ALAC, MQA

**Streaming:**

- ❌ Spotify (Ogg Vorbis 320), Apple Music (AAC 256), Tidal (MQA), Amazon Music HD

### ⛔ OUT OF SCOPE (Nicht unterstützt)

**Surround/Multichannel:**

- 🚫 5.1, 7.1 Surround
- 🚫 Quadraphonic (4.0)
- 🚫 Jegliche Multichannel-Formate (> 2 Kanäle)

**Immersive 3D Audio:**

- 🚫 Dolby Atmos
- 🚫 DTS:X
- 🚫 360 Reality Audio
- 🚫 Spatial Audio

**REASON:** Aurik 10.0.0 ist ausschließlich für Mono/Stereo optimiert.

---

## Test Markers

### Verfügbare Marker

```python
@pytest.mark.unit           # Schnelle Unit-Tests (< 1s)
@pytest.mark.integration    # Integration Tests (mehrere Komponenten)
@pytest.mark.e2e            # End-to-End Tests (vollständige Pipeline)
@pytest.mark.slow           # Langsame Tests (> 30s)
@pytest.mark.ml             # Benötigt ML-Modelle

# AURIK 9 Neue Marker (empfohlen)
@pytest.mark.musical_goals         # Tests für Musical Goals
@pytest.mark.adaptive_thresholds   # Tests für Adaptive Thresholds
@pytest.mark.material_quality      # Tests für Material Quality Assessment
@pytest.mark.vocal_enhancement     # Tests für Vocal Enhancement (De-Essing, gender-spezifisch)
@pytest.mark.forensic_analysis     # Tests für Forensic Analysis (30+ Tonträger)
@pytest.mark.transfer_chain        # Tests für Tonträgerketten-Erkennung
```

### Marker in pytest.ini

```ini
[pytest]
markers =
    unit: Schnelle Unit-Tests
    integration: Integration Tests
    e2e: End-to-End-Tests
    slow: Langsame Tests (> 30s)
    ml: Tests die ML-Modelle laden
    musical_goals: Tests für 14 Musical Goals
    adaptive_thresholds: Tests für Adaptive Threshold System
    material_quality: Tests für Material Quality Assessment
    vocal_enhancement: Tests für Vocal Enhancement
    forensic_analysis: Tests für Forensic Analysis (30+ Tonträger)
    transfer_chain: Tests für Tonträgerketten-Erkennung (De-Essing, gender-spezifisch)
```

### Test-Ausführung mit Markern

```bash
# Alle Musical Goals Tests
pytest -m musical_goals

# Alle Adaptive Thresholds Tests
pytest -m adaptive_thresholds

# Alle Vocal Enhancement Tests
pytest -m vocal_enhancement

# Alle Forensic Analysis Tests
pytest -m forensic_analysis

# Alle Transfer Chain Tests
pytest -m transfer_chain

# Vocal Enhancement mit Musical Goals
pytest -m "vocal_enhancement and musical_goals"

# Forensic Analysis mit Musical Goals
pytest -m "forensic_analysis and musical_goals"

# Schnelle Tests (ohne slow und e2e)
pytest -m "not slow and not e2e"

# Nur Integration Tests mit Musical Goals
pytest -m "integration and musical_goals"

# Vollständige Suite
pytest
```

---

## Performance-Optimierung

### Parallele Ausführung

```bash
# Parallele Ausführung (8 Workers)
pytest -n 8

# Nur schnelle Tests parallel
pytest -m "not slow" -n 8
```

### Test-Caching

```bash
# Nur fehlgeschlagene Tests re-run
pytest --lf

# Nur geänderte Tests
pytest --co
```

### Test-Auswahl

```bash
# Nur einen Test
pytest tests/test_unified_restorer.py::test_restore_musical_goals

# Pattern Matching
pytest -k "musical_goals"
```

---

## Häufige Fehler

### ❌ Fehler 1: Fixe Thresholds für degradiertes Material

```python
# FALSCH: Fixe Thresholds
def test_restore():
    restored = restorer.restore(audio, sr)
    checker = MusicalGoalsChecker()
    goals = checker.measure_all(restored, sr)

    # Fails für degradiertes Material!
    assert goals['brillanz'] >= 0.85
```

✅ **Richtig: Adaptive Thresholds**

```python
def test_restore():
    denker = get_aurik_denker_instance()
    restored = denker.denke(audio, sr, mode="restoration").audio

    # Use Adaptive Thresholds
    thresholds = MusicalGoalsChecker().thresholds

    checker = MusicalGoalsChecker()
    goals = checker.measure_all(restored, sr)

    # Passes für alle Material-Qualitäten!
    for goal, score in goals.items():
        assert score >= thresholds.get(goal, 0.0)
```

### ❌ Fehler 2: Generation-Count nicht berücksichtigt

```python
# FALSCH: Ignoriert Generation-Count
def test_material_quality():
    material = analyze_material(audio, sr)
    assert material.degradation_score < 0.3  # Fails für Multi-Generation!
```

✅ **Richtig: Generation-Count berücksichtigen**

```python
def test_material_quality():
    material = analyze_material(audio, sr)

    # Generation-Count hat 40% Gewichtung!
    if material.generation_count >= 2:
        # Multi-Generation → Höhere degradation_score erwartet
        assert material.degradation_score >= 0.30
    else:
        # Single/Pristine → Niedrigere degradation_score erwartet
        assert material.degradation_score < 0.30
```

### ❌ Fehler 3: Keine Musical Goals Validierung

```python
# FALSCH: Nur technische Validierung
def test_restore():
    restored = restorer.restore(audio, sr)
    assert restored.shape == audio.shape  # Nur Form-Check!
```

✅ **Richtig: Musical Goals Validierung**

```python
def test_restore():
    denker = get_aurik_denker_instance()
    restored = denker.denke(audio, sr, mode="restoration").audio

    # Technical validation
    assert restored.shape == audio.shape

    # Musical Goals validation
    checker = MusicalGoalsChecker()
    goals = checker.measure_all(restored, sr)

    thresholds = checker.thresholds

    for goal_name, score in goals.items():
        assert score >= thresholds.get(goal_name, 0.0), \
            f"Musical Goal '{goal_name}' violated: {score:.3f} < {thresholds.get(goal_name, 0.0):.2f}"
```

---

## Best Practice Checkliste

### Für jeden neuen Test:

- [ ] Musical Goals Validierung eingebaut?
- [ ] Adaptive Thresholds verwendet (wenn verfügbar)?
- [ ] Material Quality berücksichtigt?
- [ ] Generation-Count Impact getestet?
- [ ] Passende Marker gesetzt (`@pytest.mark.musical_goals`, etc.)?
- [ ] Test-Audio repräsentiert reale Szenarien?
- [ ] Dokumentation erklärt WHY (nicht nur HOW)?
- [ ] Assertions sind musikalisch sinnvoll?

### Für Vocal Enhancement Tests:

- [ ] Alle 3 Gender-Profile getestet (female, male, child)?
- [ ] Brillanz (HF Clarity) validiert nach De-Essing?
- [ ] Natürlichkeit (Natural Sound) verbessert?
- [ ] Authentizität (Voice Identity >= 0.88) erhalten?
- [ ] Quality Gates getestet (HF preservation, correlation >= 0.98)?
- [ ] Sibilant Detection Accuracy geprüft?
- [ ] Cross-Gender Consistency validiert (Bias-free)?
- [ ] Artifact & Bias Detection aktiviert?

### Für Forensic Analysis Tests:

- [ ] Medium-spezifische Thresholds verwendet?
- [ ] Alle 8 Hauptkategorien abgedeckt (Mechanical, Vinyl, Tape, etc.)?
- [ ] Bandwidth-Limitierungen simuliert (CYLINDER < 3 kHz, PSTN 300-3400 Hz)?
- [ ] Historische Medien: Authentizität priorisiert?
- [ ] Analog-Medien: Wärme validiert (VINYL, TAPE)?
- [ ] Digital-Medien: Transparenz validiert (CD, SACD)?
- [ ] Lossy-Medien: Bitrate-Impact getestet (128 < 192 < 320)?
- [ ] Transfer Chains: Progressive Degradation bestätigt?
- [ ] Generation-Count berücksichtigt (40% weight)?

---

## Referenzen

- **Musical Goals:** `backend/core/musical_goals/musical_goals_metrics.py`
- **Adaptive System:** `backend/core/musical_goals/adaptive_goals_system.py`
- **Unified Restorer:** `core/unified_restorer_v2.py`
- **Vocal Enhancement:** `dsp/aurik_deesser_pro/music_vocal_pipeline.py`
- **Vocal Enhancement Tests:** `tests/test_vocal_enhancement_musical_goals.py`
- **Forensic Analysis:** `forensics/unified_analyzer.py`
- **Media Types:** `forensics/signatures.py` (30+ Tonträger)
- **Comprehensive Tests:** `tests/test_comprehensive_media_types_forensics.py`
- **Testing Guide:** `docs/development/TESTING.md`

---

**Letztes Update:** 14. Februar 2026  
**Nächste Review:** Bei Major-Updates an Musical Goals System


## §v10 Pleasantness & Goal-Achievement Tests (Juli 2026)

### Neue Marker

- `@pytest.mark.pleasantness`: Tests für §v10 Pleasantness-First (HPE als oberste Instanz)
- `@pytest.mark.goal_achievement`: Tests die beweisen, dass Aurik Weltklasse-Klang liefert
- `@pytest.mark.unit`: Alle Unit-Tests (511 Dateien markiert)

### Ausführung

```bash
pytest -m pleasantness          # Pleasantness-First Verifikation
pytest -m goal_achievement      # Weltklasse-Klang Beweise
pytest -m unit                  # Alle Unit-Tests (schnell)
pytest tests/unit/test_no_blind_trust.py  # No-Blind-Trust Invariante
```

### No-Blind-Trust-Prinzip

Jeder Test, der Material-Templates prüft, MUSS auch die SNR/Spectrum/Harmonic-Messung
verifizieren. Siehe `tests/unit/test_no_blind_trust.py` als Referenz.
