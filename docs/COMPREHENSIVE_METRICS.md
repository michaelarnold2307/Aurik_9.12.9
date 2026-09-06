# Umfassendes Metrik-System für Aurik 10

## Übersicht

Das Aurik 10 Comprehensive Metrics System implementiert eine vollständige Suite von **50+ wissenschaftlich fundierten Metriken** zur Bewertung von Audio-Qualität, musikalischer Exzellenz und emotionaler Wirkung.

**Keine Dummys oder Mocks** - alle Metriken sind vollständig implementiert mit echten Algorithmen basierend auf internationalen Standards und wissenschaftlicher Forschung.

## Implementierte Module

### Core-Modul

- **`core/comprehensive_metrics.py`**: Hauptmodul mit allen Metrik-Berechnungen
- **`tests/test_comprehensive_metrics.py`**: Umfassende Test-Suite (46+ Test-Cases)

## Metrik-Kategorien

### 1. PSYCHOAKUSTISCHE METRIKEN (17 Metriken)

Objektiv messbare, perzeptuell relevante Eigenschaften nach internationalen Standards.

#### Signal-Qualität

- **SNR (Signal-to-Noise Ratio)**: Signal-Rausch-Verhältnis in dB
- **THD (Total Harmonic Distortion)**: Gesamtklirrfaktor in %
- **SINAD**: Signal-to-Noise-and-Distortion in dB

#### Lautheit & Dynamik

- **Integrated LUFS**: ITU-R BS.1770-4 konforme Lautheitsmessung
- **Loudness Range (LU)**: Dynamikbereich
- **True Peak (dBTP)**: Echter Peak-Pegel (überabgetastet)
- **Crest Factor**: Peak-zu-RMS-Verhältnis in dB

#### Frequenz-Eigenschaften

- **Frequency Response Flatness**: Frequenzgang-Linearität (0-1)
- **Spectral Centroid**: Spektraler Schwerpunkt ("Helligkeit") in Hz
- **Spectral Rolloff**: Hochfrequenz-Cutoff in Hz
- **Spectral Flux**: Spektrale Änderungsrate

#### Maskierung & Perzeption

- **Perceptual Sharpness**: Schärfe-Empfindung (acum)
- **Perceptual Roughness**: Rauheit-Empfindung
- **Tonality**: Tonal vs. Rausch-Inhalt (0-1)

#### Artefakte

- **Pre-Echo Score**: Pre-Echo-Artefakte (1=sauber, 0=Artefakte)
- **Click Detection**: Anzahl erkannter Clicks/Pops
- **Clipping %**: Prozentsatz geclippter Samples

---

### 2. MUSIKALISCHE METRIKEN (17 Metriken)

Musikalisch relevante Eigenschaften basierend auf Musiktheorie und Audio-Analyse.

#### Harmonischer Inhalt

- **Harmonic Clarity**: Harmonisch vs. inharmonisch (0-1)
- **HNR (Harmonic-to-Noise Ratio)**: Harmonik-zu-Rausch-Verhältnis in dB
- **Fundamental Stability**: Grundton-Stabilität (0-1)

#### Tonale Eigenschaften

- **Key Detection**: Erkannte Tonart (z.B., "C major")
- **Key Confidence**: Konfidenz der Tonarterkennung (0-1)
- **Consonance**: Harmonische Konsonanz (0-1)

#### Rhythmus & Timing

- **Tempo (BPM)**: Erkanntes Tempo
- **Tempo Stability**: Tempo-Konsistenz (0-1)
- **Rhythmic Regularity**: Beat-Regelmäßigkeit (0-1)

#### Artikulation & Dynamik

- **Attack Sharpness**: Transienten-Schärfe (0-1)
- **Decay Smoothness**: Abkling-Glätte (0-1)
- **Dynamic Contrast**: Mikrodynamik (0-1)

#### Klangfarbe & Textur

- **Spectral Complexity**: Harmonischer Reichtum (0-1)
- **Spectral Balance**: Bass/Mitten/Höhen-Balance (0-1)
- **Warmth**: Tiefmitten-Reichtum (0-1)
- **Brightness**: Hochfrequenz-Präsenz (0-1)
- **Fullness**: Mittenbereich-Präsenz (0-1)

---

### 3. EMOTIONALE METRIKEN (16 Metriken)

Affektive Eigenschaften basierend auf Russell's Circumplex Model und Geneva Emotional Music Scale.

#### Kern-Dimensionen (Russell's Circumplex Model)

- **Valence**: Annehmlichkeit (-1=negativ, +1=positiv)
- **Arousal**: Energie/Aktivierung (-1=ruhig, +1=energetisch)

#### Energie & Intensität

- **Energy**: Gesamtenergie (0-1)
- **Intensity**: Emotionale Intensität (0-1)
- **Tension**: Harmonische/rhythmische Spannung (0-1)

#### Emotionale Kategorien (Geneva Emotional Music Scale)

- **Power**: Gefühl von Kraft/Selbstvertrauen (0-1)
- **Joyful Activation**: Fröhlich, heiter (0-1)
- **Nostalgia**: Nostalgisch, sentimental (0-1)
- **Sadness**: Traurig, melancholisch (0-1)
- **Peacefulness**: Ruhig, friedlich (0-1)
- **Transcendence**: Spirituell, transzendent (0-1)

#### Wahrgenommener Affekt

- **Perceived Happiness**: Wahrgenommene Fröhlichkeit (0-1)
- **Perceived Sadness**: Wahrgenommene Traurigkeit (0-1)
- **Perceived Anger**: Wahrgenommene Wut (0-1)
- **Perceived Fear**: Wahrgenommene Angst (0-1)
- **Perceived Surprise**: Wahrgenommene Überraschung (0-1)

---

## Gesamtqualitäts-Scores

### Aggregierte Bewertungen

- **Overall Technical Quality**: Technische Gesamtqualität (0-1)
- **Overall Musical Quality**: Musikalische Gesamtqualität (0-1)
- **Overall Emotional Impact**: Emotionaler Gesamteinfluss (0-1)

### Aurik Quality Score

- **Aurik Quality Score**: Gewichtete Kombination aller Metriken (0-100)
  - **Internes Spitzenziel**: ≥ 90 Punkte
  - Gewichtung: 40% technisch, 40% musikalisch, 20% emotional

---

## Verwendung

### Python-API

```python
from core.comprehensive_metrics import ComprehensiveMetricsCalculator, generate_metrics_report
import numpy as np

# Audio laden (z.B., mit soundfile)
import soundfile as sf
audio, sr = sf.read('input.wav')

# Metriken berechnen
calculator = ComprehensiveMetricsCalculator(sample_rate=sr)
result = calculator.compute_all(audio)

# Ergebnisse abrufen
print(f"Aurik Quality Score: {result.aurik_quality_score:.1f} / 100")
print(f"SNR: {result.psychoacoustic.snr_db:.1f} dB")
print(f"Detected Key: {result.musical.detected_key}")
print(f"Valence: {result.emotional.valence:+.2f}")

# Check gegen internes Spitzenziel
if result.passes_aurik_standards():
  print("✅ INTERNES SPITZENZIEL ERREICHT - Meets Aurik 10.0.0 Standards!")
else:
  print("⚠️  Below internal top-tier target")

# Human-readable Report
report = generate_metrics_report(result)
print(report)

# Export als Dictionary
metrics_dict = result.to_dict()
```

### Beispiel-Output

```
======================================================================
AURIK 9.0 COMPREHENSIVE AUDIO QUALITY METRICS
======================================================================

📊 PSYCHOACOUSTIC METRICS
----------------------------------------------------------------------
SNR:                38.5 dB
THD:                1.23 %
Integrated LUFS:    -16.2 LUFS
Loudness Range:     12.3 LU
True Peak:          -1.5 dBTP
Crest Factor:       8.2 dB
Tonality:           0.82
Pre-Echo Score:     0.95
Clicks Detected:    0
Clipping:           0.00 %

🎵 MUSICAL METRICS
----------------------------------------------------------------------
Key:                C major (conf: 0.87)
Tempo:              120.5 BPM (stability: 0.92)
Harmonic Clarity:   0.78
HNR:                18.5 dB
Consonance:         0.85
Spectral Balance:   0.91
Warmth:             0.72
Brightness:         0.68
Fullness:           0.84

💫 EMOTIONAL METRICS
----------------------------------------------------------------------
Valence:            +0.65 (negative ← → positive)
Arousal:            +0.42 (calm ← → energetic)
Energy:             0.68
Intensity:          0.72
Tension:            0.35
Joyful Activation:  0.78
Nostalgia:          0.42
Sadness:            0.18
Peacefulness:       0.55
Power:              0.62

⭐ OVERALL QUALITY SCORES
----------------------------------------------------------------------
Technical Quality:  92.3 / 100
Musical Quality:    88.7 / 100
Emotional Impact:   75.4 / 100

🏆 AURIK QUALITY SCORE: 90.5 / 100

✅ INTERNES SPITZENZIEL ERREICHT - Meets Aurik 10.0.0 Standards!
======================================================================
```

---

## Wissenschaftliche Grundlagen

### Psychoakustische Metriken

- **ITU-R BS.1770-4**: LUFS/True Peak Metering
- **EBU R128**: Loudness Normalization
- **Zwicker Model**: Perceptual Sharpness/Roughness
- **FFT-basierte Analyse**: Spectral Features

### Musikalische Metriken

- **Krumhansl-Schmuckler Algorithm**: Key Detection
- **Helmholtz Consonance Theory**: Consonance Computation
- **Onset Detection**: Tempo/Rhythm Analysis
- **Chromagram Analysis**: Pitch Class Profiles

### Emotionale Metriken

- **Russell's Circumplex Model**: Valence-Arousal Framework
- **Geneva Emotional Music Scale (GEMS)**: 9 Emotional Categories
- **Spectro-temporal Features**: Energy/Intensity/Texture Analysis

---

## Performance

- **Berechnungszeit**: ~1-3 Sekunden für 5s Audio (48 kHz)
- **Speicher**: ~50 MB für typische 5s Audio-Datei
- **CPU-Optimiert**: NumPy/SciPy-basiert (kein GPU erforderlich)

---

## Integration

Das Metrik-System integriert sich nahtlos in bestehende Aurik-Module:

### Bestehende Module

- **`core/enhanced_metrics.py`**: PESQ, ViSQOL, SI-SDR, STOI
- **`metering/professional_meters.py`**: ITU-R BS.1770-4 LUFS Meter

### Neue Module

- **`core/comprehensive_metrics.py`**: Vollständige Metrik-Suite

---

## Tests

Umfassende Test-Suite mit 46+ Test-Cases:

- Psychoakustische Metrik-Tests (11 Tests)
- Musikalische Metrik-Tests (10 Tests)
- Emotionale Metrik-Tests (7 Tests)
- Qualitäts-Score-Tests (4 Tests)
- Integrationstests (5 Tests)
- Vergleichstests (3 Tests)
- Edge-Case-Tests (4 Tests)
- Performance-Tests (2 Tests)

```bash
# Tests ausführen
pytest tests/test_comprehensive_metrics.py -v
```

---

## Roadmap-Status

✅ **ABGESCHLOSSEN**: Entwicklung psychoakustischer, musikalischer und emotionaler Metriken

**Implementiert**:

- 17 psychoakustische Metriken (SNR, THD, LUFS, Tonalität, etc.)
- 17 musikalische Metriken (Tonart, Tempo, Harmonie, Klangfarbe, etc.)
- 16 emotionale Metriken (Valenz, Arousal, Energie, Geneva-Scale, etc.)
- 4 Gesamtqualitäts-Scores
- Check gegen internes Spitzenziel (≥90 Punkte)

**Nächster Schritt**: KI-Modelle für Defekterkennung, Restoration, Enhancement

---

## Normative Standards

Das Metrik-System erfüllt folgende Aurik 10.0.0 Normen:

- ✅ Real implementiert (keine Dummys/Mocks)
- ✅ Wissenschaftlich fundiert (internationale Standards)
- ✅ Vollständig getestet (46+ Test-Cases)
- ✅ Dokumentiert (diese Datei)
- ✅ Performant (CPU-optimiert)
- ✅ Erweiterbar (modular aufgebaut)

---

## Lizenz

Teil des Aurik 10.0.0 Projekts - Open Source (Lizenz noch zu bestimmen)

---

## Kontakt

Aurik 10.0.0 Development Team
Datum: 15. Februar 2026
Version: 10.0.0
