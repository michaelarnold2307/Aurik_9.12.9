# Aurik 10 — Phasen-Überblick (69 Phasen-Dateien)

**Version:** 10.0.20  
**Stand:** 2026-09-06  
**Status:** ✅ Produktionsbereit

---

## Pipeline-Flow (kanonisch)

```text
Tier 0: TransientDecoupledProcessing (HPSS — vor allem)
Tier 0: RestorabilityEstimator, EraClassifier, GermanSchlagerClassifier
Tier 1: MediumClassifier, DefectScanner, CausalDefectReasoner
Tier 1: UncertaintyQuantifier, GPParameterOptimizer
Tier 1: HarmonicPreservationGuard
Tier 2-5: Phasen 01–66 + Glue Stage + Interface = 69 Phasen-Dateien (jede via PerPhaseMusicalGoalsGate; Hör-Gates Ebenen 1/2/4 laufzeitnah)
Tier 5: EraAuthenticPerceptualCompletion (konditionell)
Tier 5: IntroducedArtifactDetector
Tier 6: FeedbackChain, TemporalQualityCoherenceMetric
Tier 6: PerceptualQualityScorer, ExcellenceOptimizer
Tier 6: MusicalGoalsChecker (15 Ziele)
Tier 6: EmotionalArcPreservationMetric
Tier 6: MicroDynamicsEnvelopeMorphing, GPParameterOptimizer.update()
```

**Parallelisierungs-Invariante:** Tier 0 und Tier 1 immer sequenziell.
Tier 2–4 dürfen parallelisieren (Merge via np.mean nur wenn gleiche Frequenzzone).
Tier 6 immer sequenziell.

---

## Vollständige Phasenliste

| Phase | Datei | Funktion |
| -------- | --------------------------- | --------------- |
| phase_01 | `phase_01_click_removal.py` | Clicks/Impulse, linked-stereo sparse patch repair |
| phase_02 | `phase_02_hum_removal.py` | Brumm 50/60 Hz |
| phase_03 | `phase_03_denoise.py` | Breitrauschen (OMLSA/IMCRA) |
| phase_04 | `phase_04_eq_correction.py` | Frequenzgang-Korrektur |
| phase_05 | `phase_05_rumble_filter.py` | Tieffrequenzrumpeln |
| phase_06 | `phase_06_frequency_restoration.py` | Bandbreitenerweiterung |
| phase_07 | `phase_07_declipper.py` | Declipper (zweites 07-Modul, SOTA-Sprint) |
| phase_07 | `phase_07_harmonic_restoration.py` | Oberton-Rekonstruktion |
| phase_08 | `phase_08_transient_preservation.py` | Transientenerhalt |
| phase_09 | `phase_09_crackle_removal.py` | Vinyl-Crackle |
| phase_10 | `phase_10_compression.py` | Dynamikkompression |
| phase_11 | `phase_11_limiting.py` | Limiting |
| phase_12 | `phase_12_wow_flutter_fix.py` | Wow/Flutter (pYIN) |
| phase_13 | `phase_13_stereo_enhancement.py` | Stereo-Erweiterung |
| phase_14 | `phase_14_phase_correction.py` | Phasen-/Azimuth-Korrektur |
| phase_15 | `phase_15_stereo_balance.py` | Stereo-Balance |
| phase_16 | `phase_16_final_eq.py` | Finales EQ |
| phase_17 | `phase_17_mastering_polish.py` | Mastering-Politur |
| phase_18 | `phase_18_noise_gate.py` | Noise-Gate |
| phase_19 | `phase_19_de_esser.py` | De-Esser |
| phase_20 | `phase_20_reverb_reduction.py` | Dereverb |
| phase_21 | `phase_21_exciter.py` | Harmonischer Exciter |
| phase_22 | `phase_22_tape_saturation.py` | Tape-Sättigungs-Emulation |
| phase_23 | `phase_23_spectral_repair.py` | Spektrale Lücken-Reparatur |
| phase_24 | `phase_24_dropout_repair.py` | Dropout-Interpolation (NMF-b) |
| phase_25 | `phase_25_azimuth_correction.py` | Azimuth-Korrektur |
| phase_26 | `phase_26_dynamic_range_expansion.py` | Dynamikbereich-Expansion |
| phase_27 | `phase_27_click_pop_removal.py` | Click/Pop (2. Pass) |
| phase_28 | `phase_28_surface_noise_profiling.py` | Oberflächenrauschen-Profil |
| phase_29 | `phase_29_tape_hiss_reduction.py` | Tape-Hiss |
| phase_30 | `phase_30_dc_offset_removal.py` | DC-Offset |
| phase_31 | `phase_31_speed_pitch_correction.py` | Geschwindigkeit/Pitch |
| phase_32 | `phase_32_mono_to_stereo.py` | Mono -> Stereo |
| phase_33 | `phase_33_stereo_width_limiter.py` | Stereo-Breiten-Begrenzer |
| phase_34 | `phase_34_mid_side_processing.py` | Mid/Side |
| phase_35 | `phase_35_multiband_compression.py` | Multibandkompression |
| phase_36 | `phase_36_transient_shaper.py` | Transient-Shaper |
| phase_37 | `phase_37_bass_enhancement.py` | Bass-Enhancement |
| phase_38 | `phase_38_presence_boost.py` | Präsenz-Boost |
| phase_39 | `phase_39_air_band_enhancement.py` | Air-Band (> 12 kHz) |
| phase_40 | `phase_40_loudness_normalization.py` | LUFS-Normierung + Pegeldrift-Korrektur |
| phase_41 | `phase_41_output_format_optimization.py` | Format-Optimierung |
| phase_42 | `phase_42_vocal_enhancement.py` | Gesangs-Enhancement |
| phase_43 | `phase_43_ml_deesser.py` | ML-De-Esser |
| phase_44 | `phase_44_guitar_enhancement.py` | Gitarren-Enhancement |
| phase_45 | `phase_45_brass_enhancement.py` | Blechbläser-Enhancement |
| phase_46 | `phase_46_spatial_enhancement.py` | Spatial-Enhancement |
| phase_47 | `phase_47_truepeak_limiter.py` | True-Peak-Limiter |
| phase_48 | `phase_48_stereo_width_enhancer.py` | Stereo-Breiten-Enhancer |
| phase_49 | `phase_49_advanced_dereverb.py` | Advanced Dereverb (WPE) |
| phase_50 | `phase_50_spectral_repair.py` | Spektrale Gesamt-Reparatur |
| phase_51 | `phase_51_drums_enhancement.py` | Schlagzeug-Enhancement |
| phase_52 | `phase_52_piano_restoration.py` | Klavier-Restaurierung |
| phase_53 | `phase_53_semantic_audio.py` | Semantische Audio-Analyse |
| phase_54 | `phase_54_transparent_dynamics.py` | Transparente Dynamik |
| phase_55 | `phase_55_diffusion_inpainting.py` | DiffWave-Inpainting |
| phase_56 | `phase_56_spectral_band_gap_repair.py` | HEAD_WEAR: Frequenzband-Lücken |
| phase_57 | `phase_57_print_through_reduction.py` | Print-Through-Reduktion (bidirektionale LMS) |
| phase_58 | `phase_58_lyrics_guided_enhancement.py` | Phonem-gef. Enhancement (§2.36 LyricsGuidedEnhancement) |
| phase_59 | `phase_59_modulation_noise_reduction.py` | Modulationsrauschen (Signal-adaptive Spektral-Gating) |
| phase_60 | `phase_60_inner_groove_distortion_repair.py` | Innenrille-Verzerrung (Positions-adaptive THD-Reduktion) |
| phase_61 | `phase_61_groove_echo_cancellation.py` | Rillen-Echo (RPM-Template-Matching + Subtraktion) |
| phase_62 | `phase_62_crosstalk_cancellation.py` | Übersprechen/Kanal-Bleed (BSS-basierte Kanaltrennung) |
| phase_63 | `phase_63_intermodulation_reduction.py` | Intermodulations-Verzerrung (Volterra-basierte IMD-Tilgung) |
| phase_64 | `phase_64_tape_splice_repair.py` | Tape-Splice-Reparatur (Splice-Click + Level-Crossfade) |
| phase_65 | `phase_65_vocal_naturalness_restoration.py` | Vocal-Naturalness (Restoration-only, Spec 06 §7.10) |
| phase_66 | `phase_66_stem_targeted_nr.py` | Stem-gezielte Rauschunterdrückung (Vocal-Stem) |

---

**Ergänzende Module:** `phase_glue_stage.py` (Glue Stage, vorletzte Stufe) und
`phase_interface.py` (Basisklasse) — zusammen mit den 67 nummerierten Modulen
(Phase 01–66; Nummer 07 doppelt belegt) ergeben sich **69 Phasen-Dateien**.
`vocal_repair.py` ist ein Zusatzmodul ohne Phasennummer (Bandbreiten-/Verzerrungs-Reparatur
vor Phase 42, §G58).

## Instrument-Aktivierungsmatrix (PANNs-gesteuert)

| PANNs-Kategorie | Aktivierte Phase(n) | Schwellwert |
| --- | --- | --- |
| Guitar / Electric Guitar | `phase_44` | >= 0.50 |
| Brass / Trumpet / Saxophone | `phase_45` | >= 0.50 |
| Drum / Percussion | `phase_51` | >= 0.50 |
| Piano / Keyboard | `phase_52` | >= 0.50 |
| Singing voice / Vocals | `phase_19 + phase_42 + phase_43 + VocalAI` | >= 0.40 |
| Speech | `phase_42` | >= 0.35 |

---

## CAUSE_TO_PHASES (Kausal-Mapping)

| Kausal-Ursache | Primäre Phasen |
| --- | --- |
| `tape_dropout` | phase_24, phase_55, phase_01, phase_03 |
| `tape_hiss` | phase_29, phase_03, phase_04, phase_40 |
| `vinyl_crackle` | phase_09, phase_01, phase_28, phase_03 |
| `vinyl_warp` | phase_12, phase_31, phase_04, phase_03 |
| `electrical_hum` | phase_02, phase_03, phase_04 |
| `head_misalignment` | phase_06, phase_04, phase_14, phase_25, phase_03 |
| `dc_offset` | phase_30, phase_40 |
| `digital_clip` | phase_23, phase_06, phase_40 |
| `compression_artifacts` | phase_23, phase_50, phase_26, phase_06, phase_54 |
| `head_wear` | phase_56, phase_14, phase_06 |
| `print_through` | phase_57, phase_29, phase_24, phase_03, phase_23 |
| `reverb_excess` | phase_20, phase_49 |
| `soft_saturation` | **(leer — BEWAHREN, kein Eingriff)** |

---

## PerPhaseMusicalGoalsGate (PMGG)

Nach jeder Phase wird eine 5-s-Stichprobe auf 6 Schnell-Ziele geprüft (< 200 ms):

- Brillanz, Wärme, Groove, TonalCenter, Natürlichkeit-MFCC, Timbre-Authentizität

Bei Regression um mehr als REGRESSION_THRESHOLD:

- Retry 1: strength x 0.65
- Retry 2: strength x 0.50
- Retry 3: strength x 0.35
- Retry 4: strength x 0.20
- Retry 5: strength x 0.10
- Rollback: Phase übersprungen, Warnung in phase_gate_log

REGRESSION_THRESHOLD ist restorability-adaptiv (§2.29 v10.0.0):

- GOOD (>= 70): 0.020
- FAIR (40–69): 0.035
- POOR (< 40):  0.055

Max. 5 Retries; P1/P2-Regression → volle Retry-Kaskade (4 Retries + Emergency); P4/P5 → nur Logging (`passed_p4p5_tolerated`).

---

**Stand: 2026-09-06 — Aurik 10.0.20**
