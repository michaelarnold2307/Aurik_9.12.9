"""
PerPhaseMusicalGoalsGate (PMGG) — Aurik 10.0.0 §2.29.

Prüft Musical Goals nach JEDER Phase via 5-s-Stichprobe.
Verhindert kumulative Degradation über 64 Phasen.

PROBLEM:
--------
Jede Phase kann Musical Goals minimal verschlechtern (z.B. Δ-0.01).
Über 20+ aktive Phasen kumuliert das zu -0.20 → ein Ziel fällt unter
den Pflicht-Schwellwert. Der End-Check kann das nicht mehr korrigieren.

ALGORITHMUS:
-----------
Pro Phase (wrap_phase()):
    1. 5-s-Stichprobe aus Mitte des Audios
    2. Phase ausführen: audio_after = phase(audio_before)
    3. Schnell-Check (15 Ziele, ≤ 200 ms, DSP-only):
       Brillanz, Wärme, Groove, TonalCenter, Natürlichkeit (MFCC-Proxy),
       Timbre-Authentizität, Bass-Kraft, Authentizität, Emotionalität,
       Transparenz, Spatial Depth, Mikro-Dynamik, Separation-Treue, Artikulation
    4. Δ = score_after − score_before für jedes Ziel
       Falls Δ < −REGRESSION_THRESHOLD (adaptiv je nach Restorability):
         Retry-1: Phase mit strength × 0.65
         Retry-2: Phase mit strength × 0.50  (v10.0.0-B3: sanfterer Gradient)
         Retry-3: Phase mit strength × 0.35
         Retry-4: Phase mit strength × 0.20
         Retry-5 (Last-Resort): Phase mit strength × 0.10
         Falls immer noch: HPE-Check — wenn Phase für menschliche Ohren
         VERSCHLECHTERT hat, wird sie ÜBERSPRUNGEN (§v10 Pleasantness-First).
         Nur wenn HPE neutral/positiv: Best-Effort mit geringster Regression.

§v10 PLEASANTNESS-FIRST (§2.29 v10):
-----------
PMGG darf Phasen überspringen, wenn sie den Klang für MENSCHLICHE OHREN
verschlechtern. HPE-Delta < -0.02 → Phase wird verworfen, Pre-Phase-Audio
wiederhergestellt. Der CausalDefectReasoner kann irren — das Ohr nicht.
Technische Regression < 0.05 wird toleriert, wenn HPE sich verbessert.

KONSTANTEN:
-----------
REGRESSION_THRESHOLD = 0.025  (adaptiv: 0.012 / 0.040 / 0.060 je Restorability)
HPE_SKIP_THRESHOLD   = -0.02  (§v10: HPE-Delta unter diesem Wert → Phase überspringen)
SAMPLE_DURATION_S    = 5.0
MAX_RETRIES          = 5  (v10.0.0-B3: 5 Retries mit sanftem Stärkegradienten)

OVERHEAD: max. 56 × 200 ms = 11.2 s pro Verarbeitungsdurchlauf (alle 15 Ziele DSP-only)
DEAKTIVIERUNG: --no-phase-gate (Debugging/Benchmarking)

WICHTIG: MERT wird im Schnell-Check NICHT verwendet (zu langsam: 800 ms)
Vollständige 15-Ziele-Prüfung bleibt am Pipeline-Ende (MusicalGoalsChecker)

Autor: Aurik 10.0.0 Development Team / v10.0.0
"""
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.core.calibration_matrix import (
    CANONICAL_THRESHOLDS_RESTORATION as _CM_REST,
)
from backend.core.calibration_matrix import (
    CANONICAL_THRESHOLDS_STUDIO2026 as _CM_STU,
)
from backend.core.calibration_matrix import compute_tcci

# §09.1 [RELEASE_MUST] Single Source of Truth: backend/core/calibration_matrix.py
# Werte hier NICHT bearbeiten — Änderungen ausschließlich in calibration_matrix.py.
# Per-Song-adaptive Schwellwerte werden via estimate_song_goal_targets() (§09.2) berechnet.

logger = logging.getLogger(__name__)

_PRECISE_METRICS_LOCK = threading.Lock()
_PRECISE_METRICS: dict[str, Any] | None = None
_PRECISE_OVERRIDE_WARN_MS: float = (
    500.0  # v10.0.0: ArticulationMetric added MFCC per-onset (16 windows); 3 metrics × ~100ms/metric is normal.
)
_VOCAL_GUARD_TRIGGER = 0.45


# ---------------------------------------------------------------------------
# Konstanten (§2.29) — restorability-adaptive Schwellwerte
# ---------------------------------------------------------------------------
# Feste Einzel-Schwelle (Legacy-Fallback, nicht mehr primär verwendet)
REGRESSION_THRESHOLD: float = 0.025

# Restorability-adaptive Schwellwerte (§2.29 Spec)
# v10.0.0: 0.012 → 0.030 (DSP-Proxy-Messrauschen 0.01–0.05).
# v10.0.0: 0.030 → 0.020 — §9.7.5 Reference-Aware Preservation Corrections
# eliminieren den größten Teil des Messrauschens; engere Schwellwerte fangen
# nun echte Regressionen zuverlässiger ab ohne False-Positives.
REGRESSION_THRESHOLD_GOOD: float = 0.020  # restorability ≥ 70
REGRESSION_THRESHOLD_FAIR: float = (
    0.045  # restorability 40–69 (§v10.0.4: 0.035→0.045, verhindert False-Positives bei transient_energie)
)
REGRESSION_THRESHOLD_POOR: float = (
    0.050  # restorability < 40 (maximal tolerant) — erhöht von 0.040, da 0.040 best-effort-Kaskaden auslöste
)

# §2.54 Material-bonus: analog/physical carriers need more tolerance because
# carrier-repair phases intentionally shift spectral fingerprints (Reference-
# Paradoxon §2.44). CD/DAT need no bonus — proxy metrics are reliable there.
_MATERIAL_THRESHOLD_BONUS: dict[str, float] = {
    "wax_cylinder": 0.022,  # most degraded — carrier-repair phases radically alter signal
    "shellac": 0.018,
    "wire_recording": 0.016,
    "lacquer_disc": 0.017,  # §v10.92: Acetat — ähnlich Shellac (physikalische Degradation)
    "optical_film": 0.010,
    "vinyl": 0.009,
    "lp": 0.009,  # §v10.92: LP-Alias für Vinyl
    "reel_tape": 0.020,
    "tape": 0.015,
    "radio_broadcast": 0.006,
    "cassette": 0.020,  # §v10.200: Kassette ist physikalisch degradierter als Reel-Tape → gleicher Bonus
    "kassette": 0.020,  # §v10.200: Deutsche Schreibweise — identische Physik
    "mp3_low": 0.005,  # codec artefacts → repair changes look regressive to proxies
    "minidisc": 0.004,
    "mp3_high": 0.002,
    "aac": 0.002,  # §v10.92: AAC 256kbps+ ≈ mp3_high
    "streaming": 0.002,  # §v10.92: Streaming (AAC 256kbps) ≈ mp3_high
    "cd_digital": 0.000,
    "dat": 0.000,
    "unknown": 0.003,
}

# ---------------------------------------------------------------------------
# §2.55b Erwartete Kollateralziel-Ausschlüsse pro Phase (§0l Lücke-1-Fix)
# ---------------------------------------------------------------------------
# Subtraktive Phasen (NR, Hiss, Brumm, Dereverb) entfernen physikalische
# Trägersignaturen, die von DSP-Proxy-Metriken fälschlicherweise als
# "authentisches Signal" gemessen wurden. Nach der Entfernung sinken
# Proxy-Scores auf den echten (niedrigeren) Wert — das ist kein Qualitätsverlust,
# sondern ein Proxy-Kalibrierungsartefakt durch Defekt-als-Signal-Messung.
#
# Physikalische Begründung:
#   phase_29: Tape-Hiss füllt Spektraltäler → waerme/authentizitaet/transparenz
#             SCHEINEN erhöht. Nach Hiss-Entfernung korrekte Proxy-Werte.
#             phase_07_harmonic_restoration stellt echte Harmonik danach wieder her.
#   phase_03: Breitband-NR glättet Transienten → transient_energie/emotionalitaet
#             sinken; phase_06/08 stellen Dynamikprofil wieder her.
#   phase_02: Brumm liefert Tiefton-Energie → waerme/emotionalitaet-Proxies
#             überschätzen warmen Klang; nach Brumm-Entfernung korrekter Wert.
#   phase_09: Knistern erzeugt Amplitudenvariabilität → authentizitaet-Proxy kann
#             sinken wenn Variabilitätskomponente entfernt wird.
#   phase_49: Hallentfernung reduziert Raumanteil → spatial_depth fällt intentional.
#   phase_59: Modulationsrauschen klingt "lebendig" → natuerlichkeit sinkt.
#   phase_12: Wow/Flutter-Korrektur schafft ruhigere Tonhöhe → groove-Proxy
#             (der Micro-Timing-Varianz misst) kann transienterweise fallen.
#   phase_05: Rumble-Filter entfernt Subsonic-Energie → bass_kraft/waerme sinken.
#   phase_25: Azimuth-Korrektur verändert Kanalbalance → separation_fidelity.
#   phase_20: Reverb-Reduktion → spatial_depth/waerme intentional reduziert.
#
# WICHTIG: Diese Ziele werden NICHT aus _compute_team_net_delta() ausgeschlossen
# (Team-Netto-Berechnung braucht vollständiges Bild). Nur _max_regression() und
# _max_regression_priority_aware() nutzen diese gefilterte Zielliste.
PHASE_EXPECTED_COLLATERAL_GOALS: dict[str, frozenset[str]] = {
    "phase_29_tape_hiss_reduction": frozenset(
        {
            "authentizitaet",
            "waerme",
            "transparenz",
            "separation_fidelity",
        }
    ),
    "phase_03_denoise": frozenset(
        {
            "transient_energie",
            "emotionalitaet",
        }
    ),
    "phase_02_hum_removal": frozenset(
        {
            "emotionalitaet",
            "waerme",
        }
    ),
    "phase_09_crackle_removal": frozenset(
        {
            "authentizitaet",
        }
    ),
    "phase_49_advanced_dereverb": frozenset(
        {
            "spatial_depth",
            "waerme",
        }
    ),
    "phase_59_modulation_noise_reduction": frozenset(
        {
            "natuerlichkeit",
            "emotionalitaet",
        }
    ),
    "phase_12_wow_flutter_fix": frozenset(
        {
            "groove",
        }
    ),
    "phase_05_rumble_filter": frozenset(
        {
            "bass_kraft",
            "waerme",
        }
    ),
    "phase_25_azimuth_correction": frozenset(
        {
            "separation_fidelity",
        }
    ),
    "phase_20_reverb_reduction": frozenset(
        {
            "spatial_depth",
            "waerme",
        }
    ),
}

# ---------------------------------------------------------------------------
# §2.55c Goal-Anti-Korrelationsmatrix (§0l Lücke-2-Fix — Teamwork-Invariante)
# ---------------------------------------------------------------------------
# Physikalisch begründete anti-korrelierte Zielpaare in Musik-Restaurierung.
# Wenn Ziel A sich verbessert und Ziel B regressiert, und sie anti-korreliert sind,
# wird B's effektiver Netto-Team-Beitrag in _compute_team_net_delta() gedämpft.
# Begrenzt auf max. 60 % Dämpfung — kein vollständiger Ausschluss aus dem Team.
#
# Quellen:
#   brillanz↔waerme: HF-Anhebung (>8 kHz) entzieht Wärmeband (200–800 Hz)
#                    Energie durch spektrale Konkurrenz (Zwicker & Fastl 1999)
#   natuerlichkeit↔transient_energie: NR-Glättung (Wiener-Filter) reduziert
#                    Onset-Amplitude-Ratio als unvermeidliches Algorithmus-Nebenprodukt
#   spatial_depth↔separation_fidelity: Raumbreite-Erhöhung erzeugt Zwischen-Kanal-
#                    Cues die Quelltrennung beeinflussen (Blumlein §2.51)
#   transparenz↔waerme: HF-Klarheitserhöhung kompetiert mit Wärmeband-Proxy
GOAL_ANTI_CORRELATIONS: dict[frozenset, float] = {
    frozenset({"brillanz", "waerme"}): -0.40,
    frozenset({"natuerlichkeit", "transient_energie"}): -0.30,
    frozenset({"spatial_depth", "separation_fidelity"}): -0.25,
    frozenset({"transparenz", "waerme"}): -0.20,
    frozenset({"brillanz", "authentizitaet"}): -0.15,
    frozenset({"micro_dynamics", "natuerlichkeit"}): -0.15,
}

# ---------------------------------------------------------------------------
# §2.29 v10.0.0: Priority-aware Retry-Budget
# ---------------------------------------------------------------------------
# P1/P2 regressions trigger full retry cascade (4 Retries + Emergency).
# P3 regressions trigger max 2 retries with 1.5× relaxed threshold.
# P4/P5 regressions never trigger retries — only logged.
#
# Begründung (Pareto-Analyse): Hohe P3–P5-Schwellwerte verursachten unnötige
# PMGG-Retries (CPU-Verschwendung) und Cross-Goal-Damage (Natürlichkeit/
# Authentizität-Regression durch Over-Optimization nachrangiger Ziele).
# GoalPriorityProtocol.PRIORITY_MAP ist die Autoritätsquelle.
# ---------------------------------------------------------------------------
_PRIORITY_MAX_RETRIES: dict[int, int] = {
    1: 4,  # P1: Natürlichkeit, Authentizität — volle Retry-Kaskade
    2: 4,  # P2: TonalCenter, Timbre, Artikulation — volle Retry-Kaskade
    3: 3,  # P3: Emotionalität, MicroDynamics, Groove — max 3 Retries (erhöht v10.0.0)
    4: 1,  # P4: Transparenz, Wärme, Bass-Kraft, SepFidelity — Recovery-Lite: 1 Retry (§0c)
    5: 1,  # P5: Brillanz, SpatialDepth — Recovery-Lite: 1 Retry (§0c)
}

# Regression-Toleranz-Multiplikator pro Priorität.
# P3-Ziele haben 1.5× mehr Toleranz als P1/P2, bevor ein Retry ausgelöst wird.
# P4/P5 (Recovery-Lite): Toleranzband 2.0×/2.5× — Regressionen unterhalb der Bandes
# werden als passed_p4p5_tolerated akzeptiert; oberhalb → 1 Recovery-Retry (§0c).
_PRIORITY_THRESHOLD_FACTOR: dict[int, float] = {
    1: 1.0,
    2: 1.0,
    3: 1.5,
    4: 2.0,  # Recovery-Lite: Toleranzband 2× threshold; darüber → 1 Retry
    5: 2.5,  # Recovery-Lite: Toleranzband 2.5× threshold; darüber → 1 Retry
}

# §2.47b JND-Effektivitätsschwelle — Sub-Threshold Phase Marking
# If ALL applicable goal-deltas are ≥ 0 and < JND → "sub_threshold" (no retry, accept)
#
# Calibrated for MUSIC WITH VOCALS (Popmusik, Schlager, Jazz, Folk, Oper).
# Sources: empirical psychoacoustic literature for complex musical stimuli with
# a prominent vocal-lead component.  All values are normalized-score equivalents
# of the perceptual JND for the respective dimension.
#
# Key literature (most recent studies and editions; older references retained only
# where no updated primary source exists):
#   Thoret, Caramiaux, Depalle & McAdams (2021) JASA 149:3429 — timbral JND in music
#   Caclin et al. (2005) JASA 118:2925 — multidimensional timbre JND ≈1 % (no newer equiv.)
#   McAdams (2019) Curr Biol 29:R764 — timbre as structuring force in music
#   Siedenburg, Iverson & McAdams (2016) JASA EL271 — acoustic correlates of timbre JND
#   Kreiman & Sidtis (2011) "Foundations of Voice Studies" — voice-quality detection
#   Krumhansl & Cuddy (2010) Psychol Learn Motiv 51:51 — updated tonal hierarchy theory
#   Marjieh, Harrison, Lee, Deligiannaki & Jacoby (2023) Music Percept. 40:183 — key salience
#   Temperley (2001) "The Cognition of Basic Musical Structures" — key-finding model
#   London (2012) "Hearing in Time" 2nd ed. (Cambridge UP) — timing JND ~8 ms in music
#   Repp & Su (2013) Psychon Bull Rev 20:403 — sensorimotor synchronisation JND review
#   Juslin (2019) "Musical Emotions Explained" Oxford UP — vocal emotion perception
#   Zentner, Grandjean & Scherer (2008) Emotion 8:494 — emotions evoked by music/voice
#   Glasberg & Moore (2002) J AES 50:331 — loudness model for time-varying sounds (JND)
#   Zwicker & Fastl (1999) "Psychoacoustics" 2nd ed. §11.2 — fluctuation strength (foundational)
#   Witek et al. (2017) PLOS ONE 12:e0169907 — groove perception sensitivity
#   Madison (2006) Music Percept. 23:227 — isochrony deviation JND ~6 ms
#   Beranek (2016) J Acoust Soc Am 139:1548 — concert hall clarity JND (updated survey)
#   Toole (2018) "Sound Reproduction" 3rd ed. (Focal Press) — room/loudspeaker thresholds
#   Alluri & Toiviainen (2012) Music Percept. 29:459 — warmth as perceptual dimension
#   Howard & Angus (2017) "Acoustics and Psychoacoustics" 5th ed. — timbral warmth
#   Glasberg & Moore (2006) JASA 119:1705 — equal-loudness / LF loudness (revised model)
#   ISO 226:2003 — equal-loudness contours 20 Hz–12.5 kHz (current standard)
#   Bregman (1990) "Auditory Scene Analysis" Ch.2 — stream segregation (foundational)
#   McDermott (2009) Curr Biol 19:R1115 — cocktail party / auditory scene analysis
#   Siedenburg & McAdams (2017) J New Music Res 46:149 — brightness/timbre in real music
#   Blauert (1997) "Spatial Hearing" 2nd ed. — precedence/reverb JND (foundational)
#   Choisel & Wickelmaier (2007) JASA 121:2718 — spatial impression JND, multichannel
#   Griesinger (1997) J AES 45:313 — reverb/spatial impression JND in concert halls
JND_MIN_DELTA: dict[str, float] = {
    # P1 — highest perceptual prominence; vocal-lead music makes these highly salient
    "natuerlichkeit": 0.012,  # Thoret et al. (2021) JASA 149:3429: timbral JND in
    # musical sounds; Caclin et al. (2005) JASA ≈1 %
    "authentizitaet": 0.012,  # Kreiman & Sidtis (2011): voice-quality detection
    # acutely sensitive in singing; voice most salient stream
    # P2 — structural musical properties; tonal centre most salient in tonal vocal styles
    "tonal_center": 0.008,  # Krumhansl & Cuddy (2010) + Marjieh et al. (2023):
    # key is most discriminable feature in tonal vocal music
    "timbre_authentizitaet": 0.012,  # Caclin et al. (2005) JASA 118:2925;
    # McAdams (2019) Curr Biol 29:R764 — timbre structure
    "artikulation": 0.010,  # London (2012) "Hearing in Time" 2nd ed. ~8 ms;
    # Repp & Su (2013) Psychon Bull Rev 20:403 rhythm JND
    "transient_energie": 0.010,  # Onset energy follows articulation/rhythm salience.
    # P3 — groove/dynamics/emotion; emotional cues in voice prominent at 100–300 ms scale
    "emotionalitaet": 0.014,  # Juslin (2019) "Musical Emotions Explained" OUP;
    # Zentner et al. (2008) Emotion 8:494 voice-emotion JND
    "micro_dynamics": 0.012,  # Glasberg & Moore (2002) J AES 50:331 time-varying
    # loudness JND; Zwicker & Fastl (1999) §11.2 reference
    "groove": 0.010,  # Witek et al. (2017) PLOS ONE 12:e0169907 groove;
    # Madison (2006) Music Percept. 23:227 isochrony ≈6 ms
    # P4 — tonal-balance/spatial; slower time-constants but smaller than once assumed
    "transparenz": 0.012,  # Beranek (2016) JASA 139:1548 clarity C80 JND ~1 dB;
    # Toole (2018) "Sound Reproduction" 3rd ed. Ch. 9
    "waerme": 0.016,  # Alluri & Toiviainen (2012) Music Percept. 29:459;
    # Howard & Angus (2017) "Acoustics & Psychoacoustics" 5th ed.
    "bass_kraft": 0.012,  # Glasberg & Moore (2006) JASA 119:1705 revised model;
    # ISO 226:2003 equal-loudness contours, LF region
    "separation_fidelity": 0.014,  # Bregman (1990) "Auditory Scene Analysis" Ch.2;
    # McDermott (2009) Curr Biol 19:R1115 scene analysis
    # P5 — spectral brilliance / room depth; broader integration windows
    "brillanz": 0.016,  # Siedenburg & McAdams (2017) J New Music Res 46:149;
    # HF brightness JND in complex musical sounds ≈1 dB
    "spatial_depth": 0.018,  # Blauert (1997) "Spatial Hearing" 2nd ed.;
    # Choisel & Wickelmaier (2007) JASA 121:2718 spatial JND
}

SAMPLE_DURATION_S: float = 5.0
MAX_RETRIES: int = 5  # v10.0.0-B3: 5 Retries mit sanftem Stärkegradienten (0.65→0.50→0.35→0.20→0.10)

# ---------------------------------------------------------------------------
# §9.7.3 Phasen-adaptive Sample-Dauer — triviale Phasen brauchen < 5 s
# ---------------------------------------------------------------------------
PHASE_SAMPLE_DURATIONS: dict[str, float] = {
    # Triviale Phasen: Zeiteffekt ist lokal messbar in 1–2 s
    "phase_30": 1.5,  # DC-Offset-Removal
    "phase_05": 1.5,  # Rumble-Filter (< 20 Hz)
    "phase_02": 2.0,  # Hum-Removal (50/60 Hz Kammfilter)
    "phase_15": 1.5,  # Stereo-Balance L/R
    "phase_11": 1.5,  # Limiting (True-Peak)
    "phase_18": 2.0,  # Noise-Gate
    # Standard: SAMPLE_DURATION_S = 5.0 für alle anderen Phasen
}

# §9.7.4 Phase-specific goal exclusions.
# Goals whose DSP proxy is structurally unreliable for a given processing type.
# These goals are NOT checked for regression when the phase matches.
#
# v10.0.0: Exclusions significantly reduced thanks to §9.7.5 reference-aware
# preservation corrections.  Goals with spectral/temporal correlation support
# are now checked even for phases that previously triggered false positives.
# Only goals where processing FUNDAMENTALLY changes the measured quantity
# (and correlation cannot distinguish intentional change from degradation)
# remain excluded.
#
# Rationale for remaining exclusions:
#
# phase_02 (hum removal): 50/100/.../400 Hz comb-filter creates spectral
#   notches directly in the bass band → bass_kraft LF correlation still sees
#   notches as degradation because they ARE spectral removal (intentional).
#   authentizitaet excluded: comb-filter notches create spectral roughness
#   that is the intended action, not degradation.
#
# phase_04 (EQ correction): Spectral redistribution IS the core function.
#   transparenz (rolloff + balance) changes deliberately.
#
# phase_06 (frequency restoration): SBR intentionally adds HF content that
#   the reference doesn't have → correlation is LOW by design.
#   brillanz excluded because the increase IS the goal.
#
# phase_18 / phase_26 / phase_36: Dynamics-modifying phases intentionally
#   change the temporal envelope → micro_dynamics measures the intended change.
# pylint: disable=line-too-long
PHASE_GOAL_EXCLUSIONS: dict[str, set[str]] = {
    # Hum removal: comb-filter notches in bass band + spectral roughness.
    # natuerlichkeit excluded: CREPE voicing analysis in NatuerlichkeitMetric
    # flags 50/100 Hz notch-induced spectral-flatness changes as P1 regression.
    # transparenz excluded: 50/100/150/250/300 Hz hum harmonics are narrow spectral
    # peaks in the 250-500 Hz band (5th and 6th harmonic of 50 Hz hum sit at exactly
    # 250 and 300 Hz).  Notch filters remove these peaks, which lowers p95 in the
    # first octave band of the §9.7.13 multi-band crest proxy → false P4 regression
    # even though audio quality has improved.  Unlike broadband denoising (phase_03)
    # where noise fills the ENTIRE band floor (elevating p50), hum-notch removal only
    # reduces isolated peaks → net crest DECREASE in the 250-500 Hz band.  The
    # §9.7.13 fix does not cover narrowband-notch-induced peak removal.
    # groove excluded (P3 root cause, 2026-03-30): hum removal does not affect
    # timing or rhythmic events, but the GrooveMetric onset/DTW proxy is sensitive
    # to LF spectral energy changes (50–200 Hz). Real-run stagnation Δ=0.000000
    # across all retries confirms filter-independence — this is a measurement
    # artifact. Groove 0.1526 regression proved to produce false catastrophic
    # PMGG cascades. Export gate still enforces GrooveMetric threshold globally.
    # timbre_authentizitaet excluded (P2 root cause, 2026-03-30): spectral notches
    # directly disturb the MFCC-Pearson and spectral-centroid correlation proxies.
    # Even a shallow notch at 50 Hz shifts lower MFCC coefficients, driving the
    # timbre proxy below threshold despite no perceptual timbre degradation.
    "phase_02": {
        "bass_kraft",
        "authentizitaet",
        "natuerlichkeit",
        "transparenz",
        "groove",
        "timbre_authentizitaet",
        # hum-notch removal changes onset rise-time in notched bands (LF energy alters
        # ArticulationMetric transient-rise proxy) — CIG has this, PMGG sync §2.54
        "artikulation",
    },
    # Reconstruction phases: spectral correlation handles reconstruction well;
    # only keep exclusions where AI-generated content has low correlation by design
    # natuerlichkeit excluded: gap-fill synthesis produces content absent from
    # reference; CREPE voicing score on synthesised audio is unreliable.
    # artikulation excluded (P2 root cause, 2026-03-29): dropout repair inserts
    # newly synthesised transients inside missing regions. ArticulationMetric
    # compares transient-shape correlation against the pre-repair signal where
    # those transients are absent by definition, causing false catastrophic P2
    # regressions (worst_goal=artikulation ~0.23) despite musically improved
    # continuity after repair.
    # brillanz excluded: synthesised fill content can have different HF spectral
    # distribution than the surrounding noisy reference → false brillanz drop.
    # authentizitaet excluded (belt+suspenders for flatness proxy): dropout silence
    # has near-zero amplitude → fft_mag ≈ 0 → flatness undefined/high → scores_before
    # may be artificially low; after FlashSR synthesis tonal content increases.
    # The flatness-based proxy handles this correctly in practice but the exclusion
    # prevents edge-cases in very short silence segments where the 2.5-s sample
    # window captures mostly dropout.
    "phase_24": {
        "natuerlichkeit",
        "brillanz",
        "authentizitaet",
        "artikulation",
        "timbre_authentizitaet",
        "transparenz",
        "tonal_center",
        "groove",  # FlashSR synthesis fills 5981 dropout gaps with new audio patches; formerly silent/corrupted dropout frames had 0 onsets → GrooveMetric onset-DTW autocorr[lag_05] registers onset-density increase as rhythm disruption → false P3 regression. Regression constant at all strengths (stagnation Δ=0.000004, 2026-04-10) → PMGG reduces strength to 0.22 (best_effort), leaving >5000 dropouts unrepaired. Identical mechanism to phase_09/phase_18 groove exclusion.
        "emotionalitaet",  # Dropout silence gaps score high in crest-factor (silence/near-zero amplitude between notes amplifies peak-to-RMS ratio in degraded reference). After FlashSR synthesis, formerly silent patches receive normal signal amplitude → crest-factor ratio drops → false P3 emotionalitaet regression. Identical mechanism to phase_09 (broadband transitions from near-silence) and phase_18 (noise-gate silencing). Regression invariant to strength → confirmed stagnation pattern.
    },  # Dropout repair: synthesised HF content; timbre_authentizitaet: FlashSR synthesis creates new spectral content → MFCC correlation against damaged reference is meaningless; transparenz: dropout silence regions inflate spectral clarity proxy (silence = perfect rolloff) → after FlashSR fill slight noise floor added → proxy drops (false P4); tonal_center: dropout silence has undefined/near-zero chroma → K-S key detection unstable; after FlashSR tonal synthesis K-S locks onto different key estimate → false tonal regression despite musically unchanged pitch centre (stagnation 0.3137 confirmed, 2026-04-08). groove + emotionalitaet: added 2026-04-10 — see inline comments above.
    "phase_28": {
        "artikulation",
        "natuerlichkeit",
        "timbre_authentizitaet",
        "authentizitaet",
    },  # Surface noise profiling (vinyl): broadband noise events look like transients to ArticulationMetric → after profiling/removal pseudo-transients gone → false P1 regression (catastrophic 0.4222 confirmed, 2026-04-08); natuerlichkeit: broadband spectral denoising (same MFCC-smoothness mechanism as phase_03/phase_29); timbre_authentizitaet: spectral envelope changes when broadband surface noise removed → MFCC-Pearson + centroid-CV shift; authentizitaet: §2.44 Reference-Paradoxon — broadband surface noise smooths log-spectrum valleys → roughness proxy scores HIGH before profiling; after removal true valleys reappear → false P1 cascade (identical mechanism to phase_03/phase_29, aligned with CIG §2.48 exclusions, 2026-04-09)
    # Diffusion inpainting: synthesised content has no transient reference →
    # ArticulationMetric correlation vs pre-inpainting fragment is meaningless.
    # micro_dynamics excluded: inpainting inserts new content with its own
    # envelope that intentionally differs from the surrounding material.
    "phase_55": {
        "artikulation",
        "micro_dynamics",
        "natuerlichkeit",
        "brillanz",
        "authentizitaet",
        "timbre_authentizitaet",
        "tonal_center",
    },  # Diffusion inpainting: synthesised content → identical root-causes as phase_23/phase_24 (FlashSR); §4.7c POCS n_iter=2–5 vor PGHI; MFCC-smoothness vs. damaged reference meaningless; brillanz crest-proxy scores against absent HF pre-synthesis; authentizitaet flatness-proxy reference-mismatch; timbre_authentizitaet MFCC-Pearson/centroid meaningless for synthesised spectral content; tonal_center excluded (§9.7.11 extension, 2026-04-10): CQTdiff+ fills bandwidth-loss gaps with synthesized HF content — pre-inpainting audio (band-limited vinyl ≤8-12 kHz) has near-zero chroma energy in high-register bins; after inpainting, newly filled HF bins shift K-S key-template correlation → false catastrophic P2 regression (Δ=0.8333 confirmed, 06:34 run). Musical key is unchanged; only chroma-bin distribution shifts due to spectral extension
    # Sub-sonic removal: reference LF correlation handles bass preservation check
    "phase_05": {
        "natuerlichkeit",
        "authentizitaet",
        "bass_kraft",  # HPF intentionally removes sub-bass energy (< 20–80 Hz) → bass_kraft DSP proxy drops as intended; not a musical regression (§0 Primum non nocere: rumble removal IS the repair)
        "waerme",  # HPF removes low-end rumble energy in 80–400 Hz range → warmth ratio E(200-800)/E(800-3000) shifts → false P4 regression; CIG sync §2.55 (2026-04-26)
        "tonal_center",  # HPF at 24 Hz attenuates C1 (~32.7 Hz) by ~5 dB → K-S chroma bin for lowest octave shifts → key-label changes by up to 4 semitones → false P2 catastrophic regression (Δ=0.6583 confirmed on mp3/vinyl, 2026-05-06). Musical key unchanged; only sub-bass chroma distribution shifts. §2.55 sync: CIG updated.
    },  # Rumble filter: sub-sonic removal shifts MFCC-smoothness baseline + sub-bass chroma removal causes minor chromagram shift — §2.55 sync: CIG updated (bass_kraft, waerme, tonal_center)
    # Broadband denoise: reference HF/LF correlation distinguishes noise from music
    # natuerlichkeit excluded: broadband denoising shifts spectral flatness and
    # ZCR, causing the CREPE-based NatuerlichkeitMetric to report false P1
    # regressions (~0.28) even at near-dry wet-mix.  DSP proxy with §9.7.5
    # reference-aware preservation correctly evaluates naturalness for denoise.
    # artikulation excluded: ArticulationMetric(reference=noisy_tape) measures
    # transient-shape correlation between the denoised output and the noisy input.
    # Denoising IS supposed to reshape transients (ResembleEnhance, OMLSA spectral
    # weighting) — scores_before(reference-free)≈0.67 vs scores_after(ref-based)≈0.13
    # produces a false P2 regression of ~0.54 that drives PMGG into best_effort at
    # strength=0.06 (virtually no denoising applied).  Root cause confirmed in debug
    # logs (2026-03-28): worst_goal=artikulation, before=0.665, after=0.126.
    # brillanz excluded: broadband denoising removes HF noise energy → brillanz DSP
    # proxy drops from ~0.9 (noise-inflated) to ~0.1 (clean).
    # authentizitaet excluded (P1 root cause, v10.0.0): broadband noise raises the
    # spectral noise-floor uniformly → log-spectrum valleys are filled → roughness
    # proxy scores HIGH before denoising.  After denoising the true spectral valleys
    # are revealed → roughness INCREASES → authentizitaet drops ~0.75 → false P1
    # catastrophic cascade (0.8884 regression, phase runs at 6% strength).
    # This is the INTENDED outcome of denoising — not a musical-quality regression.
    # transparenz excluded: HF noise inflates spectral rolloff → DSP proxy scores
    # scores_before too high; after denoising rolloff drops to true musical level
    # → false P4 regression triggering unnecessary retries.
    # timbre_authentizitaet excluded (P2 root cause, 2026-03-30): denoise phases
    # intentionally alter spectral-centroid variance and fine texture while reducing
    # hiss. The PMGG short-window timbre proxy can overreact on tape material and
    # report false catastrophic P2 regressions (~0.09 > 0.08) despite improved
    # perceptual clarity.
    # tonal_center excluded (§9.7.11 extension, v10.0.0): K-S is invariant to ADDITIVE
    # white noise (uniform spectral floor lifts all chroma bins equally → ratios preserved).
    # OMLSA/ResembleEnhance apply FREQUENCY-SELECTIVE suppression (gain G(f) varies per
    # frequency band) which effectively acts as a noise-adaptive EQ → chroma energy
    # distribution shifts → K-S key template correlation changes. Real-run confirmed:
    # catastrophic tonal_center regression Δ=0.1043 on 1930s tape (SNR≈15 dB, 1/f hiss).
    # The musical key did not change; K-S measurement is disturbed by shaped NR.
    # brillanz+transparenz: §9.7.12/13 crest-factor proxies are SNR-robust → kept.
    "phase_03": {
        "natuerlichkeit",
        "artikulation",
        "authentizitaet",
        "tonal_center",
        "timbre_authentizitaet",
        # §V36 transient_energie (v10.0.0): OMLSA/DFN entfernt Rauschimpulse, die im
        # TransientEnergieProxy als Onsets gewertet wurden. Nach NR: Rauschspitzen weg →
        # Proxy zeigt weniger Onsets → false P3. Realer Endwert: 0.805 (über Boden 0.746).
        # Bestätigt: Δ=−0.13 in Run 1779217698 → PMGG best_effort_r1 → NR zu schwach.
        "transient_energie",
    },  # OMLSA/ResembleEnhance: CREPE-Load-State + transient-shape mismatch + K-S NOT invariant for shaped NR §9.7.11 ext + MFCC-Pearson/Centroid-CV disturbed by spectral-envelope change after NR (v10.0.0 canonical — groove/emotionalitaet entfernt: P3-Quick-Proxy-Robustheit hinreichend) + §V36 transient_energie (v10.0.0)
    # DeepFilterNet HF-removal intentionally reduces HF energy → brillanz drops.
    # artikulation excluded for same reason as phase_03: reference=hissy_tape vs
    # denoised output gives misleadingly low transient-correlation score.
    # authentizitaet excluded: same root-cause as phase_03 — tape hiss smooths the
    # log-spectrum (spectral valleys filled by noise floor); after DeepFilterNet v3 II
    # removes hiss, true valleys reappear → roughness rises → false P1 catastrophic
    # regression (0.5661 observed).
    # transparenz excluded: HF hiss inflates rolloff proxy → DSP rolloff score drops
    # after hiss removal → false P4 regression triggering unnecessary retries.
    # natuerlichkeit excluded: MFCC-smoothness DSP proxy unreliable during HF-removal
    # (same root cause as phase_02 and phase_03).
    # tonal_center excluded (§9.7.11 extension, v10.0.0): DeepFilterNet v3 II is a
    # learned frequency-selective filter — identical mechanism to OMLSA (see phase_03).
    # HF-targeted tape-hiss removal reduces energy in high-register chroma bins
    # (C5-B7) while leaving low-register bins less affected → K-S correlation shifts
    # even though the musical key is unchanged. Real-run stagnation (Δ=0.000311) at
    # strength=0.78 confirms the regression is measurement-driven, not musical.
    "phase_30": {
        "authentizitaet",
        "natuerlichkeit",
    },  # DC-offset removal: near-DC energy removal shifts spectral fingerprint vs. DC-distorted reference → false P1 regression; §2.55 sync: CIG also has {authentizitaet, natuerlichkeit}
    "phase_29": {
        "artikulation",
        "authentizitaet",
        "natuerlichkeit",
        "tonal_center",
        "timbre_authentizitaet",
        # §V32 [RELEASE_MUST]: Breitbandige HF-Rausch-Energie (Tape-Hiss) inflationiert
        # den HF-Crest-Proxy (transparenz) künstlich. Nach Hiss-Entfernung sinkt der
        # Proxy auf den physikalisch realen Träger-Wert → CIG feuert false-positive
        # Rollback (Drift −0.284, Threshold −0.04 bestätigt v10.0.0).
        # Analogie: tonal_center-Exclusion für phase_03 (Reference-Paradox §2.44).
        "transparenz",
        # §V36 waerme (v10.0.0): OMLSA/DFN v3 wendet breitbandige Gain-Suppression an,
        # auch im Wärmeband (200–2000 Hz). Tape-Hiss-Rauschboden trägt zur absoluten
        # Wärmeband-Energie bei; nach Entfernung sinkt der Proxy (E_warmth/E_total ×3.5)
        # auf den physikalisch realen Trägerwert → false P4 Regression.
        # Realer Endwert: 0.792 (über Canonical-Boden 0.75). Bestätigt: 0.91→0.74
        # (Δ=−0.17) in Run 1779217698 → PMGG best_effort_r1 → NR zu schwach.
        "waerme",
    },  # DeepFilterNet Tape-Hiss — gleiche Root-Causes wie phase_03: MFCC-Pearson + centroid-CV + K-S shaped-NR-instabilität (v10.0.0 canonical — groove/emotionalitaet entfernt) + §V32 transparenz (HF-Crest false-positive Rollback) + §V36 waerme (v10.0.0)
    # Phases with RADICAL spectral changes where even correlation can't help:
    # phase_04 EQ: redistributes the entire spectrum — brillanz (HF cut/boost)
    # and waerme (mid cut/boost) are intentional outcomes, not regressions.
    # authentizitaet excluded: EQ notch/shelf filters create spectral non-uniformity
    # in the log-domain → roughness proxy rises → false P1 catastrophic regression
    # (0.5503 observed).  EQ-induced spectral shaping IS the intended restoration
    # action — not a musical-quality regression.
    # natuerlichkeit excluded: MFCC-smoothness DSP proxy is directly disturbed by
    # EQ notches (same mechanism as phase_02 comb-filter notches).
    # timbre_authentizitaet excluded: EQ shifts spectral centroid trajectory — the
    # centroid-CV proxy treats any centroid change as timbre degradation, but EQ
    # correction is intentional spectral-shape restoration.
    # phase_16 final_eq: mirrors phase_04 EQ exclusions + tonal_center (see phase_03).
    # Confirmed catastrophic tonal_center regression Δ=0.4708 (P2) in real run.
    # Final mastering EQ with presence boost (3-5 kHz) strengthens upper harmonics of
    # each note → those harmonics land in specific semitone bins → chroma distribution
    # shifts → K-S correlation changes. Not a musical key change.
    "phase_16": {
        "transparenz",
        "brillanz",
        "waerme",
        "authentizitaet",
        "natuerlichkeit",
        "timbre_authentizitaet",
        "tonal_center",
    },  # Final EQ: same spectral redistribution as phase_04 + K-S chroma-shift (§9.7.11 ext)
    "phase_04": {
        "transparenz",
        "brillanz",
        "waerme",
        "authentizitaet",
        "natuerlichkeit",
        "timbre_authentizitaet",
        "artikulation",
    },  # EQ deliberately redistributes spectrum (§9.7.11 K-S: tonal_center not yet observed failing here); artikulation: EQ spectral reshaping modifies frequency distribution of transient attacks → ArticulationMetric transient-shape correlation changes as spectral envelope of attacks shifts (catastrophic P2 regression 0.2515 confirmed, 2026-04-08)
    "phase_06": {
        "timbre_authentizitaet",
    },  # SBR/bandwidth extension adds new HF harmonics: brillanz excluded rationale no longer applies here because §9.7.12 crest-proxy correctly handles synthesis improvement. timbre_authentizitaet: adding sub-10kHz HF harmonics via SBR changes MFCC-Pearson + spectral-centroid-CV (intentional spectral content addition = false P2 by design, confirmed catastrophic regression 0.2185 on timbre_authentizitaet P1 in E2E, 2026-04-08)
    "phase_07": {
        "artikulation",
        "timbre_authentizitaet",
        # §v10.18 Declipper: Rekonstruierte Wellenform-Peaks ändern das
        # Roughness-Profil vs. Clipping-Referenz (§2.44 Reference-Paradox).
        # Die geclippten Samples erzeugen künstlich glatte Spektraltäler →
        # nach PCHIP-Rekonstruktion erscheinen echte Täler → false P1.
        "authentizitaet",
        # §2.55 CIG-Sync: additive Harmonik-Synthese oberhalb der Carrier-Ceiling
        # ändert das MFCC-Smoothness-Profil vs. bandbreitenbegrenzter Referenz
        # (§2.44 Reference-Paradox, identisch zu CIG-Rationale) — CIG bereits
        # excludiert, PMGG nachgezogen.
        "natuerlichkeit",
    },  # Harmonic restoration + Declipper: H2-H4 waveshaping adds new harmonic partials → onset-sharpness proxy saturates at 1.0 pre-phase (mean_peaks/0.01 clips) then drops after harmonic addition (new spectral energy reshapes attack envelope) → false P2 artikulation catastrophic regression (0.2532 observed, 2026-04-02). timbre_authentizitaet: harmonic synthesis + declipping intentionally changes MFCC-Pearson + spectral-centroid-CV. authentizitaet: Wellenform-Rekonstruktion ändert Roughness-Profil (§v10.18).
    # Click removal: replaces impulse artifacts with interpolated audio.
    # artikulation excluded: clicks are high-amplitude transients in the damaged
    # signal — ArticulationMetric sees them as "transients". After removal they're
    # absent → transient-shape correlation drops → false P2 regression despite
    # genuine quality improvement. The proxy compares damage-transients vs. repair.
    # natuerlichkeit excluded (P1 root cause, 2026-04-07): click removal applies
    # spectral interpolation over the removed impulse locations. NatuerlichkeitMetric
    # MFCC-smoothness proxy evaluates local short-window coherence; the transition
    # from reconstructed frames to undamaged surroundings creates MFCC trajectory
    # discontinuities that score as "unnatural" relative to the click-bearing
    # reference. Real-run confirmed: worst_goal=natuerlichkeit, regression=0.267 (P1),
    # PMGG dithered to strength=0.17 (virtually no click removal applied).
    # Same root cause as phase_02 comb-notch → CREPE/MFCC-smoothness mismatch.
    "phase_01": {
        "artikulation",
        "natuerlichkeit",
        "timbre_authentizitaet",
        "authentizitaet",
        "tonal_center",  # §2.44 Reference-Paradox: 22965 click events are broadband impulses; spectral interpolation at scale alters chromagram → K-S key-template correlation drops despite pitch structure being preserved/improved. Identical mechanism to phase_12/phase_49/phase_58. CIG P2 rollback confirmed (rollbacks=1, strength→0.07, 2026-04-10).
        "groove",  # Clicks appear as spurious onset events in GrooveMetric onset-based DTW proxy. High-severity click removal reduces onset count/density → autocorr[lag_05] DTW changes → false P3 regression. Identical mechanism to phase_09 groove exclusion (confirmed: 22965 clicks on gen=7 vinyl).
    },  # Click removal: impulse transients → ArticulationMetric false P2; spectral interpolation → NatuerlichkeitMetric MFCC-smoothness false P1; timbre_authentizitaet: MFCC-Pearson shift at repaired click locations; authentizitaet: §2.44 reference-paradox roughness shift vs. click-bearing reference. tonal_center + groove: see inline comments above.
    # Click/pop removal: identical mechanism to phase_01 (different algorithm,
    # same false-regression root cause for all excluded proxies).
    "phase_27": {
        "artikulation",
        "natuerlichkeit",
        "timbre_authentizitaet",
        "authentizitaet",
        "tonal_center",  # Same mechanism as phase_01: click/pop interpolation at scale alters K-S chroma correlation → false P2 CIG rollback. Confirmed: phase_27 rollbacks=1 on same run (2026-04-10).
        "groove",  # Same mechanism as phase_01 + phase_09: impulse removal changes onset density → onset-DTW false P3 regression. (phase_27 already handled as P99 tolerance, explicit exclusion prevents CIG-level rollback cascade.)
    },  # Click/pop removal: same proxy limitations as phase_01 — tonal_center (K-S false P2 via chroma shift) + groove (onset-DTW false P3 via impulse removal) both added 2026-04-10.
    # BANQUET blind denoising: full-band neural diffusion-based crackle/noise removal.
    # natuerlichkeit excluded: BANQUET modifies the full spectral envelope (same root
    # cause as phase_03/phase_29 — MFCC-smoothness proxy disturbed by denoising).
    # groove excluded (P1 root cause, 2026-04-07): BANQUET removes crackle events
    # that appear as periodic impulsive onsets. GrooveMetric onset-based DTW proxy
    # registers the change in LF onset density as rhythmic disruption. Real-run
    # confirmed: worst_goal=groove, regression=0.291 (P1), stagnation across all
    # retries, strength=0.15 (virtually no crackle removal). Same mechanism as
    # phase_02 groove exclusion — LF spectral energy changes fool onset-DTW proxy.
    # authentizitaet excluded: crackle adds broadband noise floor → log-spectrum
    # valleys filled high before BANQUET; after processing, valleys reappear →
    # roughness rises → false P1 cascade. Identical to phase_03/phase_29.
    # timbre_authentizitaet excluded: MFCC-Pearson/centroid-CV disturbed by
    # full-band spectral envelope modification (same as phase_29).
    "phase_09": {
        "natuerlichkeit",
        "groove",
        "authentizitaet",
        "timbre_authentizitaet",
        "artikulation",  # crackle removal changes onset energy envelope (same mechanism as phase_01/phase_27) — CIG sync §2.54
        "tonal_center",  # broadband crackle inflates K-S chroma bins; after removal chroma estimate shifts vs. crackle-bearing checkpoint — CIG sync §2.54
    },  # BANQUET blind denoising: full-band spectral mod → natuerlichkeit MFCC-smoothness false P1 (0.291, 2026-04-07); groove onset-DTW false P1 (0.291); authentizitaet log-spectrum valley mechanism; timbre MFCC-Pearson/centroid-CV
    # Spectral repair (STFT inpainting via bin interpolation):
    # Replaces isolated spike-bins with linear interpolation from ±2 neighbours.
    # This is DSP-only (no ML synthesis), so natuerlichkeit/authentizitaet proxies
    # are unaffected (no spectral envelope synthesis). Only artikulation is at risk:
    # isolated spike-bins can appear as transient onsets to the proxy; after repair
    # those spikes are smoothed → false P2 regression for heavily corrupted sections.
    "phase_50": {
        "artikulation"
    },  # STFT spectral inpainting: isolated spike-bins appear as transients → smoothing causes false ArticulationMetric P2 regression
    # Harmonic exciter: synthesises H2–H4 harmonics to enhance presence/air.
    # timbre_authentizitaet excluded: adding harmonics intentionally changes
    # MFCC-Pearson (the timbre IS changing) and spectral-centroid-CV
    # (HF partial energy increases) → false P2 regression vs. pre-exciter reference.
    "phase_21": {
        "timbre_authentizitaet"
    },  # Harmonic exciter: H2-H4 synthesis intentionally changes MFCC-Pearson + centroid-CV → false P2 timbre regression
    # Tape saturation: tanh-waveshaping (soft saturation) + harmonic series modeling.
    # timbre_authentizitaet: harmonics added intentionally → MFCC-Pearson changes.
    # emotionalitaet: tanh reduces peak amplitude relative to RMS (peak compression)
    # → crest-factor ratio drops → false P3 regression despite intended enhancement.
    "phase_22": {
        "timbre_authentizitaet",
        "emotionalitaet",
    },  # Tape saturation: tanh waveshaping compresses peaks → crest-factor drops (false P3); harmonic synthesis changes MFCC-Pearson (false P2)
    # Bass enhancement: low-shelf EQ + sub-harmonic synthesis + soft saturation.
    # tonal_center excluded: sub-harmonic synthesis creates tones an octave below
    # fundamentals — K-S chroma template correlation changes because the pitch-class
    # weight distribution shifts (added energy at sub-octave positions).
    # timbre_authentizitaet: LF energy addition shifts lower-order MFCC coefficients.
    # waerme excluded: bass boost directly increases energy in the 200–800 Hz warmth
    # band → warmth ratio E(200-800)/E(800-3000) changes → false P4 regression.
    # emotionalitaet: LF boost raises RMS significantly relative to peaks → crest
    # drops → false P3 regression (same mechanism as phase_22 tanh saturation).
    "phase_37": {
        "timbre_authentizitaet",
        "tonal_center",
        "waerme",
        "emotionalitaet",
    },  # Bass enhancement: sub-harmonic synthesis → K-S chroma shift + MFCC change; LF energy boost → warmth-ratio + crest-factor false regressions
    # Presence/mid-range clarity EQ (1–4 kHz dynamic boost + Bell EQ).
    # timbre_authentizitaet: boosting 1-4 kHz changes MFCC c1-c3 (dominant spectral
    # range) + centroid-CV directly → false P2 regression vs. pre-boost reference.
    # waerme excluded: presence boost raises energy in 800–3000 Hz band → warmth
    # ratio E(200-800)/E(800-3000) changes → false P4 regression.
    "phase_38": {
        "timbre_authentizitaet",
        "waerme",
    },  # Presence EQ: 1-4 kHz boost changes MFCC c1-c3 + warmth ratio E(200-800)/E(800-3000) → false P2/P4 regressions
    # Air band enhancement: shelving EQ + harmonic synthesis for 12–20 kHz.
    # timbre_authentizitaet: HF centroid-CV increases when 12-20 kHz is boosted
    # (centroid shifts upward) → MFCC higher-order coefficients change → false P2.
    "phase_39": {
        "timbre_authentizitaet"
    },  # Air band HF enhancement: 12-20 kHz boost shifts centroid-CV + MFCC higher-order coefficients → false P2 timbre regression
    # De-esser (DSP primary + MP-SENet ML refinement): targets sibilant 4–8 kHz.
    # artikulation excluded: /s/, /f/ fricatives ARE transients — the de-esser
    # specifically attenuates their peaks → ArticulationMetric registers this as
    # transient-shape regression vs. pre-processing reference despite it being repair.
    # timbre_authentizitaet: 4-8 kHz spectral reduction changes centroid-CV + MFCC
    # higher-order coefficients → false P2 regression.
    # emotionalitaet: de-essing reduces crest specifically at sibilant peaks
    # → crest-factor ratio drops → false P3 regression.
    "phase_43": {
        "timbre_authentizitaet",
        "artikulation",
        "emotionalitaet",
        # §V32-Analogie Sibilanz (§2.55-Sync v10.0.0): Sibilant-Energie (4–8 kHz) inflationiert
        # den HF-Crest-Proxy (brillanz) und HF-Rolloff-Proxy (transparenz) auf Kassetten-/
        # Vinyl-Vokal-Material. Nach De-Essing sinken beide auf reale Trägerwerte →
        # false P4/P5 Regression (Reference-Paradox §2.44). Identischer Mechanismus
        # wie phase_29/transparenz (V32). Bestätigt: PMGG best_effort bei strength<0.15.
        "brillanz",
        "transparenz",
    },  # De-esser: 4-8 kHz sibilant attenuation → artikulation (fricative transients attenuated) + timbre (centroid-CV + MFCC) + emotionalitaet (crest-factor drop) + brillanz/transparenz (§V32-Analogie Sibilanz v10.0.0) false regressions
    # Guitar enhancement: spectral shaping for guitar timbre (distortion, presence).
    # timbre_authentizitaet: guitar-specific spectral shaping intentionally changes
    # the MFCC-Pearson + centroid-CV profile → false P2 vs. pre-enhancement.
    "phase_44": {
        "timbre_authentizitaet",
        # §2.55 CIG-Sync: guitar-spezifisches Spectral-Shaping ändert die
        # spektrale Hülle → MFCC-Smoothness-Proxy betroffen (identischer
        # Mechanismus wie timbre_authentizitaet) — CIG bereits excludiert,
        # PMGG nachgezogen.
        "natuerlichkeit",
    },  # Guitar enhancement: spectral shaping changes MFCC-Pearson + centroid-CV → false P2 timbre regression
    # Brass enhancement: HP-filtered formant enhancement + spectral shaping.
    # timbre_authentizitaet: brass formant enhancement changes spectral envelope
    # deliberately → MFCC-Pearson/centroid-CV proxy registers as P2 regression.
    "phase_45": {
        "timbre_authentizitaet"
    },  # Brass enhancement: formant spectral shaping changes MFCC-Pearson + centroid-CV → false P2 timbre regression
    # Spectral tilt: global broadband EQ tilt (boost LF / cut HF or vice versa).
    # timbre_authentizitaet: global spectral tilt directly changes lower-order MFCC
    # c1-c3 (dominant LF/MF energy) + spectral centroid location → false P2.
    # waerme excluded: spectral tilt shifts E(200-800)/E(800-3000) warmth ratio
    # depending on tilt direction → false P4 regression.
    # emotionalitaet: broad energy redistribution changes RMS vs. peak balance
    # → crest-factor ratio shifts → false P3 regression.
    # phase_53_semantic_audio: METADATA-only phase — audio is returned UNCHANGED.
    # process() computes BPM, key, genre-hint and writes results to PhaseResult.metadata.
    # No spectral or dynamics modification → scores_before == scores_after for all 15 goals
    # → no PMGG regression possible → exclusions are structurally unnecessary.
    "phase_53": set(),  # SemanticAudioPhase is metadata-only (audio unchanged) → no goal can regress
    # Spectral Band Gap Repair (HEAD_WEAR defect): harmonics interpolated via
    # Fletcher partial model + NMF-β refinement.
    # Mechanistically identical to phase_23 (FlashSR spectral inpainting) for all
    # synthesis-reference-mismatch root causes; §4.7c POCS n_iter=2–5 vor PGHI
    # natuerlichkeit: synthesised partial harmonics differ from pre-repair damaged
    # reference → MFCC smoothness proxy unreliable.
    # brillanz: synthesised HF band energy distribution may differ from the HF gap
    # reference → crest proxy scores against a damaged (near-zero HF) baseline.
    # authentizitaet: spectral gap has near-zero amplitude → flatness undefined;
    # after repair, tonal content increases → reference-mismatch-driven transition.
    # timbre_authentizitaet: MFCC-Pearson meaningless against damaged gap reference.
    "phase_56": {
        "natuerlichkeit",
        "brillanz",
        "authentizitaet",
        "timbre_authentizitaet",
    },  # HEAD_WEAR band gap repair: harmonic interpolation synthesis → identical reference-mismatch root causes as phase_23/phase_55
    # Print-through reduction (bidirectional LMS, reel_tape only):
    # Removes magnetic pre/post-echo from tape print-through artifact.
    # Mechanistically identical to phase_49 (Advanced Dereverb) for:
    # authentizitaet: echo tail spread energy across spectrum → smooths log-spectrum
    # valleys; after removal, valleys reappear → roughness rises → false P1 cascade.
    # emotionalitaet: echo tail adds residual energy to quiet segments between musical
    # events → scores_before crest-factor elevated; after removal, quiet segments
    # become true silence → crest-factor ratio shifts → false P3 regression.
    "phase_57_print_through_reduction": {
        "authentizitaet",
        "emotionalitaet",
    },  # Print-through reduction: echo-tail/pre-echo removal → authentizitaet (log-spectrum valley mechanism) + emotionalitaet (crest-factor in silence segments) false regressions — identical to phase_49
    # Lyrics-Guided Enhancement (§2.36 PFLICHT): phoneme-aligned DSP per segment class.
    # timbre_authentizitaet excluded: fricative ramp-gain (4-8 kHz), vowel formant
    # shelving (LPC Burg Ord.30-40), plosive burst boost (100-350 Hz) all intentionally
    # change the spectral envelope → MFCC-Pearson + centroid-CV register as P2 regression
    # vs. the pre-enhancement reference where these phoneme targets were under-enhanced.
    # artikulation excluded: plosive TransientShapeGuard bypasses onset-window (gain=1.0)
    # but burst boost (×1.40) and aspiration boost (3-8 kHz ×1.20) modify the plosive
    # shape → ArticulationMetric transient-shape correlation registers change vs. baseline.
    # emotionalitaet excluded: fricative high-frequency ramp-gain raises HF energy
    # selectively at sibilant positions → local crest-factor ratio shifts → false P3
    # regression despite intended timbral improvement.
    "phase_58_lyrics_guided_enhancement": {
        "tonal_center",  # §Y5: fricative ramp-gain (4–8 kHz) shifts HF energy profile → K-S key-label flip (SNR-sensitive K-S already excluded from shaped-NR phases)
        "timbre_authentizitaet",
        "artikulation",
        "emotionalitaet",
    },  # LGE §2.36: phoneme-specific spectral ops (fricative ramp, plosive burst, formant shelving) → MFCC-Pearson + transient-shape + local crest false regressions
    # M/S dynamics: compresses BOTH Mid AND Side channels independently per 4 bands.
    # Mid compression (ratio 2.0–3.0) directly affects the mono sum (L+R)/2 = Mid.
    # micro_dynamics excluded: multi-band Mid/Side compression intentionally reshapes
    # envelope — same mechanism as phase_10 (multiband compression).
    # groove excluded: Mid compression changes inter-beat RMS periodicity in the
    # mono sum → autocorr[lag_05] disrupted (same mechanism as phase_17).
    # emotionalitaet excluded: Mid compression reduces crest-factor in mono sum
    # → false P3 regression (same mechanism as phase_17/phase_10).
    # timbre_authentizitaet excluded: multiband Mid+Side spectral shaping with
    # different gain-reduction per band alters the spectral envelope of the
    # mono sum → MFCC c1-c3 + centroid-CV register false P2 regression.
    "phase_34": {
        "micro_dynamics",
        "groove",
        "emotionalitaet",
        "timbre_authentizitaet",
    },  # M/S dynamics: Mid-channel compression (ratio 2-3) affects mono sum → groove+emotionalitaet+micro_dynamics (same as phase_10) + timbre_authentizitaet (per-band spectral shaping)
    # Loudness normalization (ITU-R BS.1770-4 / EBU R128):
    # Pure LUFS gain scaling is scale-invariant for ALL ratio-based proxies
    # (groove autocorr, crest-factor, tonal K-S, timbre MFCC-Pearson are unchanged
    # by global gain). BUT includes multi-band loudness shaping (frequency-dependent
    # gain adjustment) which is essentially a spectral EQ → timbre risk.
    # timbre_authentizitaet excluded: multi-band frequency-dependent loudness shaping
    # shifts spectral envelope → MFCC c1-c3 + centroid-CV register false P2.
    "phase_40": {
        "timbre_authentizitaet",
        # §2.55 CIG-Sync: multi-band frequency-dependent loudness shaping verändert
        # die spektrale Hülle → stört auch den MFCC-Smoothness-Proxy (identischer
        # Mechanismus wie bei timbre_authentizitaet) — CIG bereits excludiert,
        # PMGG nachgezogen.
        "natuerlichkeit",
    },  # Loudness normalization: pure LUFS gain is scale-invariant; multi-band frequency shaping changes spectral envelope → timbre_authentizitaet false P2 regression
    # + transient enhancement. Three false-regression root causes on degraded material:
    # micro_dynamics excluded: transient enhancement intentionally reshapes the LUFS
    # micro-profile — that change IS the intended TDP effect, not a regression.
    # artikulation excluded: transient-shaping BY DEFINITION changes transient shapes;
    # comparing after-TDP transients against before-TDP baseline is meaningless since
    # TDP is supposed to alter transient characteristics.
    "phase_08": {"micro_dynamics", "artikulation"},  # TDP transient preservation (§9.7.11 K-S: tonal_center resolved)
    # Dynamics-modifying phases: intentional temporal envelope changes
    # phase_18 noise gate: removes background noise (incl. HF noise) between
    # musical events → brillanz drops from noise-inflated value → false regression.
    # authentizitaet excluded: same log-spectrum valley mechanism as phase_03 —
    # noise in silence gaps smooths log-spectrum; after gating, silence is silent
    # (zeros) → the FFT sample captures more musical-content frames → valleys
    # become visible → roughness rises → false P1 regression.
    # transparenz excluded: HF noise in silence sections inflates rolloff proxy;
    # after gating those sections, average rolloff drops → false P4 regression.
    # emotionalitaet excluded: noise gate deliberately changes crest factor by
    # silencing quiet sections between musical phrases → crest_score shift is
    # the intended effect, not a dynamics regression.
    # groove excluded (P3 root cause, 2026-03-30): noise gate silences inter-beat
    # quiet sections → rms_env becomes discontinuous at gate-on/off boundaries
    # → gate-zero segments inflate autocorr[0] variance → normalized autocorr[lag_05]
    # drops even at minimal gate strength (Δ=0.002226 stagnation, regression 0.1721
    # observed; best_effort at 0.19 strength = noise gate effectively disabled).
    # Groove proxy measures rhythmic periodicity; VAD-gated silence IS the intended
    # noise-gate effect and cannot be decoupled from the rhythm signal in 2.5 s windows.
    "phase_18": {
        "micro_dynamics",
        "authentizitaet",
        "emotionalitaet",
        "groove",
        "timbre_authentizitaet",  # noise gate inserts silence between phrases → spectral centroid/MFCC changes vs. continuous-noise reference — CIG sync §2.54
        "artikulation",  # §2.55-Sync: VAD-Gate schneidet Note-Attacks ab → artikulation-Score bricht catastrophic ein (Δ>0.29). Das IST der Zweck des Gates, kein Bug. CIG sync §2.55.
        "transient_energie",  # §1.4.6: VAD-Gate zeroes sub-threshold frames including real onsets → onset-amplitude ratio drops → false transient_energie regression; TransientEnergyMetric fallback uses artikulation proxy which also suffers the same gate-induced drop.
    },  # Noise gate (§9.7.11 K-S: tonal_center resolved — K-S key-detection is SNR-invariant; §9.7.12/13: brillanz+transparenz crest proxies SNR-robust → removed)
    "phase_26": {
        "micro_dynamics",
        "artikulation",
        "groove",
        "emotionalitaet",
        "transient_energie",  # §1.4.6: expander widens transient/sustain ratio → crest-factor shift → onset-amplitude ratio jumps non-linearly; TransientEnergyMetric onset-amplitude proxy registers gain as false regression when measured mid-expansion.
    },  # Dynamic expansion: expander opens transient/decay gap → RMS-env autocorr[lag_05] disrupted + crest-factor shift → false P3 regressions (same mechanisms as phase_18 noise gate)
    "phase_36": {
        "micro_dynamics",
        "artikulation",
        "groove",
        "emotionalitaet",
    },  # Transient shaper: transient-boost raises peaks vs. RMS floor → crest-factor ratio shifts + RMS-peak timing changes → false P3 regressions (same mechanisms as phase_08 + phase_18)
    # Multiband parallel compression: attack/release envelopes directly modify
    # inter-beat RMS envelope periodicity → autocorr[lag_05] changes → false P3 groove
    # regression (identical mechanism to phase_17 multiband mastering compressor).
    # Crest-factor reduction via compression → false P3 emotionalitaet regression.
    # micro_dynamics excluded: by design compression changes envelope dynamics.
    "phase_10": {
        "micro_dynamics",
        "groove",
        "emotionalitaet",
        # §2.55 CIG-Sync: Multiband-Kompression glättet spektrale Varianz →
        # MFCC-Smoothness-Proxy (natuerlichkeit) betroffen; Gain-Änderungen pro
        # Band verschieben Chroma-Bin-Gewichtung (tonal_center); spektrales
        # Rebalancing ändert MFCC-Pearson (timbre_authentizitaet) — CIG bereits
        # excludiert, PMGG nachgezogen.
        "natuerlichkeit",
        "tonal_center",
        "timbre_authentizitaet",
    },  # Multiband parallel compression: envelope modification → groove autocorr[lag_05] disrupted + crest-factor uniformly reduced → false P3 regressions (identical mechanism to phase_17)
    # 4-band limiting: brick-wall limiter is an extreme compressor with ∞:1 ratio.
    # Peaks clipped → crest-factor drops → false P3 emotionalitaet regression.
    # Inter-beat periodicity changes when loud transient peaks are attenuated
    # differently per band → groove autocorr[lag_05] disrupted.
    "phase_11": {
        "micro_dynamics",
        "groove",
        "emotionalitaet",
        # §2.55 CIG-Sync: Brickwall-Limiting fügt Harmonik am Threshold hinzu
        # (spektraler Fingerabdruck ändert sich → natuerlichkeit), Peak-Clipping
        # verschiebt Chroma-Peak-Verteilung (tonal_center), Transienten werden
        # absichtlich umgeformt (artikulation) — CIG bereits excludiert, PMGG nachgezogen.
        "natuerlichkeit",
        "tonal_center",
        "artikulation",
        "timbre_authentizitaet",
    },  # Multi-band limiting: extreme compression (∞:1) → crest-factor drops + RMS-envelope periodicity changes → false P3 regressions (identical mechanism to phase_17 + phase_10)
    # TruePeak limiter: clamps sample peaks above a threshold via 4× oversampling.
    # Aggressive application (near 0 dBFS ceiling) extensively clips transient peaks
    # → crest-factor drops significantly → false P3 emotionalitaet regression.
    # Loud transient beats attenuated → inter-beat amplitude contrast reduced →
    # autocorr[lag_05] misreads periodic beat pattern → false P3 groove regression.
    "phase_47": {
        "micro_dynamics",
        "groove",
        "emotionalitaet",
    },  # TruePeak limiter: peak-clamping reduces crest-factor + inter-beat peak contrast → false P3 regressions (same mechanism as phase_11)
    # 4-band independent compression with upward/downward compander:
    # mechanistically identical to phase_10 (multi-band parallel compression) and
    # phase_17 (mastering compressor). Envelope modification per band.
    "phase_35": {
        "micro_dynamics",
        "groove",
        "emotionalitaet",
    },  # 4-band multiband compression: independent band compander → RMS envelope periodicity disrupted + crest-factor reduced → false P3 regressions (identical mechanism to phase_10/phase_17)
    # Psychoacoustic-aware compression (genre-adaptive, masking-aware):
    # applies RMS-envelope-adaptive threshold and alters dynamics per masked region.
    # Despite perceptual optimisation the proxy mechanisms are identical:
    # crest-factor reduction + inter-beat envelope change → false P3 regressions.
    "phase_54": {
        "micro_dynamics",
        "groove",
        "emotionalitaet",
    },  # Psychoacoustic compression: genre-adaptive envelope modification → crest-factor drops + groove autocorr[lag_05] disrupted → false P3 regressions (identical mechanism to phase_17 + phase_35)
    # Mastering: intentional dynamics compression + spectral shaping.
    # tonal_center excluded (§9.7.11 ext, v10.0.0): multiband compression +
    # presence/air EQ redistribute chroma energy → K-S detects apparent key shift.
    # artikulation excluded: multiband compression is designed to change attack
    # envelopes (faster attack → softer onset, slower release → sustain boost);
    # ArticulationMetric's transient-shape correlation measures this change as
    # regression, but the mastering effect IS the intended outcome. Real-run:
    # catastrophic P2 regression Δ=0.2092 (worst_goal=artikulation).
    "phase_17": {
        "micro_dynamics",
        "natuerlichkeit",
        "tonal_center",
        "artikulation",
        "groove",
        "emotionalitaet",
    },  # groove: multiband compression changes inter-beat RMS envelope periodicity → false P3 regression (Δ=0.0251 observed); emotionalitaet: compression reduces crest-factor uniformly → false P3 regression (identical mechanism to phase_18) — confirmed 2026-03-31
    # Vocal enhancement: Stages 2-6 intentionally alter spectral shape and dynamics;
    # natuerlichkeit/timbre proxies are unreliable for deliberate vocal-presence boosts.
    "phase_19": {
        "natuerlichkeit",
        "timbre_authentizitaet",
        "micro_dynamics",
        "groove",
        "emotionalitaet",
        # §V32-Analogie Sibilanz (§2.55-Sync v10.0.0): identischer Mechanismus wie phase_43.
        # Sibilant-HF (4–8 kHz) inflationiert brillanz + transparenz auf Kassetten-/
        # Vinyl-Vokal-Material → nach De-Essing false P4/P5 Regression (Reference-Paradox §2.44).
        "brillanz",
        "transparenz",
        # §2.55-Sync (2026-07-18): De-Esser redistribuiert spektrale Energie im
        # Sibilanzbereich (4–10 kHz) → K-S-Chroma-Verteilung verschiebt sich um
        # bis zu 3 Halbtöne → false P2 catastrophic regression auf tonal_center
        # (0.9978 bestätigt, cassette+schlager, Phase hatte 0 Sibilanten + 0 dB
        # Reduktion → Regression ist reiner Messartefakt).
        "tonal_center",
    },  # Vocal enhancement: Stage 6 micro-compression shifts crest-factor + Stage 2 breath-gating changes inter-beat RMS periodicity → false P3 regressions (same mechanisms as phase_17/phase_18) + brillanz/transparenz (§V32-Analogie Sibilanz v10.0.0) + tonal_center (§2.55 K-S-chroma-shift durch Sibilanz-Redistribution)
    # BSRoFormer vocal stem separation + vocal enhancement with micro-compression
    # (syllable-level, ratio 1.8–2.5) + envelope shaping + FormantSystem enhancement.
    # Mechanistically similar to phase_19 (compression/breathing) + phase_23 (synthesis).
    # natuerlichkeit excluded: BSRoFormer stem synthesis + formant enhancement alter
    # MFCC smoothness proxy on the separated signal vs. mixed reference.
    # authentizitaet excluded: stem separation changes spectral flatness (separation
    # exposes vocal harmonics previously masked by instrumentation → flatness shifts).
    # timbre_authentizitaet excluded: formant enhancement + spectral envelope change
    # from stem isolation shifts MFCC-Pearson + centroid-CV proxy.
    # groove excluded: syllable-level micro-compression modifies inter-syllable RMS
    # periodicity → autocorr[lag_05] false regression (same mechanism as phase_19).
    # emotionalitaet excluded: micro-compression reduces crest-factor at syllable level
    # → false P3 regression (identical mechanism to phase_17/phase_19).
    "phase_42": {
        "natuerlichkeit",
        "authentizitaet",
        "timbre_authentizitaet",
        "groove",
        "emotionalitaet",
        "artikulation",
    },  # BSRoFormer vocal enhancement: stem separation + micro-compression + formant shaping → false regressions via synthesis/crest/MFCC mechanisms (identical to phase_19 + phase_23); artikulation: BSRoFormer stem resynthesis reshapes transient content → ArticulationMetric transient-shape correlation vs. original meaningless for ML-synthesized output (catastrophic P2 regression 0.2043 confirmed, 2026-04-08)
    # Drums/percussion enhancement: transient shaping (attack/sustain) + DrumsEnhancementSystem
    # which includes compression (Dbx-style). Beat-synchronous transient shaping alters
    # the inter-beat RMS contrast → groove autocorr[lag_05] disrupted.
    # Drum spectral enhancement (punch/snap synthesis) changes timbre MFCC proxy.
    # Compression on beats reduces crest-factor at beat positions → false P3 emotionalitaet.
    "phase_51": {
        "timbre_authentizitaet",
        "groove",
        "emotionalitaet",
    },  # Drums enhancement: transient shaping + compression → inter-beat RMS changes + crest-factor drops → false P3 regressions; timbre_authentizitaet: punch/snap synthesis changes spectral envelope
    # Piano restoration: dynamic range restoration via material-adaptive expansion
    # (velocity curve optimization, expansion ratios 1.2–1.3, compression artifact removal).
    # Expansion is upward dynamic expansion (inverse compression) — increases crest-factor
    # and changes note-to-note RMS envelope periodicity → false P3 groove regression
    # (same root cause as phase_26 dynamic expansion). String resonance modeling and
    # spectral enhancement change timbre MFCC proxy.
    "phase_52": {
        "timbre_authentizitaet",
        "groove",
        "emotionalitaet",
    },  # Piano restoration: dynamic expansion (1.2–1.3) + string resonance synthesis → inter-beat RMS periodicity changes + crest-factor shifts → false P3 regressions; timbre_authentizitaet: string resonance modeling changes MFCC-spectral-envelope proxy
    # Dereverb: removes room impulse response; reverb contributes diffuse HF energy
    # and room resonances (warmth). After dereverb brillanz and waerme both drop
    # legitimately — these are intentional improvements, not regressions.
    # authentizitaet excluded: reverb tail spreads energy across spectrum → smooths
    # log-spectrum (fills valleys with diffuse energy) → scores_before artificially
    # high; after dereverb true spectral valleys reappear → roughness rises →
    # false P1 catastrophic regression (0.5502 observed).
    # transparenz excluded: reverb-contributed diffuse HF energy inflates spectral
    # rolloff → scores_before elevated; after dereverb rolloff drops → false P4
    # regression triggering unnecessary strength reductions.
    "phase_49": {
        "authentizitaet",
        "tonal_center",
        "timbre_authentizitaet",
        "artikulation",
        "natuerlichkeit",
    },  # Advanced dereverb: tonal_center excluded (§9.7.11 ext, 2026-04-10): WPE/spectral-subtraction removes reverb energy from high-register chroma bins unevenly → K-S correlation shifts; catastrophic P2 regression 0.4667/0.5530 confirmed in real run. timbre_authentizitaet: reverb tail shifts MFCC-Pearson at all cepstral coefficients → removal changes spectral-centroid-CV (identical mechanism to phase_03/phase_29). artikulation: reverb tail blurs transient attacks → pre-removal ArticulationMetric(reverberant reference) vs de-reverbed output shows false correlation drop. natuerlichkeit: spectral-subtraction dereverb applies frequency-selective gain G(f) → MFCC-smoothness instability
    # Reverb reduction (SGMSE+ primary / WPE-DSP fallback): mechanistically identical
    # to phase_49 Advanced Dereverb — both remove room impulse response energy.
    # brillanz excluded: reverb tail contributes diffuse HF energy across the spectrum
    # → brillanz proxy scores HIGH before removal (noise-inflated); after SGMSE+ the
    # dry direct signal no longer carries that diffuse HF → false brillanz drop.
    # waerme excluded: reverb mid-band tail (early reflections 200–2000 Hz) lifts the
    # waerme proxy before processing; removal exposes the dry mid energy → false drop.
    # authentizitaet excluded: reverb smooths log-spectrum valleys (same mechanism as
    # broadband noise in phase_03); after removal true valleys reappear → flatness-proxy
    # perceives this as reduced tonality → false P1 cascade (same 0.55 regression
    # observed in production for phase_49 and structurally identical for phase_20).
    # transparenz excluded: reverb contributes diffuse HF inflating 75%-rolloff proxy;
    # after removal rolloff drops legitimately → false P4 regression.
    # natuerlichkeit excluded: SGMSE+ spectral deconvolution can introduce slight
    # harmonic smearing on ambiguous reverb vs. body resonance segments → MFCC
    # smoothness proxy reacts on the 5-s short window even when result is perceptually
    # correct. Same MFCC-smoothness instability as phase_02/phase_03 root causes.
    "phase_20": {
        "authentizitaet",
        "natuerlichkeit",
        "tonal_center",
        "timbre_authentizitaet",
        "artikulation",
    },  # SGMSE+ reverb reduction: tonal_center excluded (§9.7.11 ext, 2026-04-10): SGMSE+ U-Net applies learned frequency-selective deconvolution → high-register chroma bins attenuated unevenly → K-S correlation shifts; P2 catastrophic regression 0.5530 confirmed. timbre_authentizitaet + artikulation: identical mechanism to phase_49 (reverb tail MFCC/transient-shape mismatch vs dry reference)
    # Spectral inpainting (FlashSR gap-fill): synthesises new frequency content for
    # spectral holes (codec artefacts, digital clipping reconstruction, missing HF).
    # Identical synthesised-content mechanism to phase_24 (FlashSR dropout repair).
    # natuerlichkeit excluded: gap-fill synthesis produces content absent from the
    # noisy/damaged reference; MFCC-smoothness proxy on the synthesised region is
    # unreliable vs. the pre-repair (damaged) reference.
    # brillanz excluded: synthesised HF fill may have different spectral distribution
    # than the surrounding damaged reference → false brillanz regression against a
    # damaged-signal baseline.
    # authentizitaet excluded: spectral gaps have near-zero amplitude → fft_mag ≈ 0
    # → flatness undefined; after FlashSR synthesis tonal content increases →
    # authentizitaet score transition is reference-mismatch-driven, not a regression.
    # artikulation excluded: inpainting inserts new spectral content in regions where
    # (by definition) the reference has damaged/missing content → transient-shape
    # correlation against the pre-inpainting fragment is meaningless.
    "phase_23": {
        "natuerlichkeit",
        "brillanz",
        "authentizitaet",
        "artikulation",
        "timbre_authentizitaet",
        "tonal_center",  # §9.7.11 extension (2026-04-24): FlashSR bandwidth-extension shifts K-S chroma bins — pre-repair audio (band-limited vinyl ≤12 kHz) has near-zero chroma energy in high-register bins; after FlashSR fill newly synthesised HF bins shift K-S key-template correlation → false catastrophic P2 regression (Δ=0.7893 confirmed, real-run 2026-04-24). Musical key is unchanged; only chroma-bin distribution shifts due to spectral extension. Identical mechanism to phase_55 (CQTdiff+) confirmed in prior runs (Δ=0.8333, 2026-04-10).
    },  # FlashSR spectral inpainting / gap-fill; timbre_authentizitaet: synthesised fill content has different spectral envelope than damaged reference
    # Wow/flutter correction: time-stretching/resampling shifts chroma energy
    # distribution → K-S key correlation changes despite unchanged musical key.
    # Regression variance 0.067→0.833 across runs of same audio PROVES this is
    # pure proxy noise, not a real quality issue.
    # timbre_authentizitaet: speed correction alters spectral centroid trajectory.
    "phase_12": {
        "tonal_center",
        "timbre_authentizitaet",
        "authentizitaet",
        "natuerlichkeit",
        "artikulation",
    },  # Wow/flutter fix: K-S volatile after pitch/speed correction + centroid-CV disturbed; reference-paradox affects authentizitaet/natuerlichkeit/artikulation proxies.
    # Speed/pitch correction: global time-stretch + resampling — mechanistically
    # identical to phase_12 for all proxy false-regression root causes.
    # tonal_center excluded: global pitch-shift moves ALL chroma bins proportionally
    # → K-S key template correlation changes even when the musical key interpretation
    # is correct (only absolute frequency changes, not musical class).
    # timbre_authentizitaet excluded: pitch-shift alters spectral centroid directly
    # (f0 × ratio → centroid × ratio) → centroid-CV proxy registers as timbre change.
    # groove excluded: time-stretch changes absolute frame timing of RMS peaks;
    # autocorr[lag_05] measures periodicity in absolute sample-time units, not
    # musical-beat units → tempo-corrected audio appears less periodic to the proxy.
    # emotionalitaet excluded: global speed change uniformly scales all envelope
    # segments → crest-factor ratio shifts because loud/quiet segment durations change
    # → false P3 regression despite identical musical dynamics after correction.
    # artikulation excluded: PSOLA/time-stretch modifies transient shapes by
    # design — that IS the correction. TransientShapeCorrelation vs. pre-correction
    # reference is meaningless (same root cause as phase_08 TDP).
    "phase_31": {
        "tonal_center",
        "timbre_authentizitaet",
        "groove",
        "emotionalitaet",
        "artikulation",
        "natuerlichkeit",  # speed correction shifts tempo → MFCC-smoothness temporal consistency changes vs. speed-deviated reference — CIG sync §2.54
        "authentizitaet",  # speed/pitch correction fundamentally changes chromagram vs. pitch-deviated reference (carrier-chain inversion §2.46 — mirror of phase_12) — CIG sync §2.54
    },  # Speed/pitch correction: global time-stretch identical mechanisms to phase_12 + emotionalitaet/artikulation via envelope/transient change (2026-03-31)
    # Stereo enhancement (multi-band M/S + Haas cross-feed delays + Blumlein shuffling):
    # The Haas effect simulation (5–35 ms inter-channel delays) adds delayed copies of
    # one channel to the other → cross-feed creates comb-filtering artifacts in the
    # mono sum (L+R)/2 → spectral balance changes → MFCC-Pearson + centroid-CV
    # register change vs. pre-enhancement reference → false P2 timbre regression.
    # Transient-preserving Side enhancement (attack/decay-aware lateral widening)
    # additionally modifies the L/R spectral content independently of the mono sum.
    "phase_13": {
        "timbre_authentizitaet"
    },  # Stereo enhancement: Haas cross-feed delays (5–35 ms) create comb-filter in mono sum + transient-aware Side shaping → MFCC-Pearson + centroid-CV change vs reference → false P2 timbre regression
    # Phase correction (multi-band all-pass / fractional-delay alignment):
    # Phase-only operations preserve per-channel spectral magnitude, but correcting
    # inter-channel misalignment changes the constructive/destructive interference
    # pattern in the mono sum (L+R)/2. Before correction: channel misalignment causes
    # spectral cancellation notches in the M-channel. After correction: channels align
    # → cancellation resolved → M-channel spectral valleys fill in → MFCC-Pearson vs.
    # the misaligned reference detects a spectral-shape change → false P2 regression.
    "phase_14": {
        "timbre_authentizitaet",
        "authentizitaet",  # phase correction resolves mono-sum cancellation notches → stereo correlation fingerprint changes vs. phase-misaligned reference — CIG sync §2.54
    },  # Phase correction: all-pass/fractional-delay alignment resolves mono-sum cancellation notches → spectral shape changes vs. misaligned reference → MFCC-Pearson + centroid-CV false P2 timbre regression
    # Stereo balance correction: re-balancing L/R channel levels intentionally changes stereo field.
    # authentizitaet: stereo correlation fingerprint changes vs. imbalanced carrier reference (§2.44 Carrier-Chain-Inversion).
    # timbre_authentizitaet: per-channel spectral balance change shifts MFCC of stereo mix vs. imbalanced reference.
    "phase_15": {
        "authentizitaet",
        "timbre_authentizitaet",
    },  # Stereo balance correction: L/R re-balancing changes stereo-field fingerprint → authentizitaet + MFCC-Pearson false P2 regression vs. imbalanced reference (§2.44)
    # Azimuth correction (tape head misalignment: fractional delay + HF restoration):
    # HF restoration via spectral prediction adds energy in the 5–20 kHz range —
    # mechanistically identical to phase_39 (air band HF enhancement). MFCC
    # higher-order coefficients (c7–c13 HF-sensitive) change when true HF content
    # is restored from azimuth-caused HF dropout; spectral centroid shifts upward.
    # The proxy compares against the reference measured BEFORE azimuth correction
    # (with reduced HF due to destructive interference) → false P2 timbre regression
    # despite genuine quality improvement.
    "phase_25": {
        "timbre_authentizitaet",
        "authentizitaet",  # azimuth correction changes stereo HF balance vs. mis-aligned reference → chromagram fingerprint shifts (§2.44 carrier-chain inversion) — CIG sync §2.54
        "tonal_center",  # §2.55 CIG-Sync: azimuth re-centering shifts K-S chroma template correlation vs. mis-aligned reference (identisch zu phase_12/31) — CIG bereits excludiert, PMGG nachgezogen
    },  # Azimuth correction: fractional-delay + HF spectral restoration changes MFCC higher-order coefficients + centroid-CV vs. azimuth-degraded reference → false P2 timbre regression (identical mechanism to phase_39 air band)
    # Mono-to-stereo (Lauridsen pseudo-stereo + HF harmonics + Schroeder decorrelation):
    # Schroeder reverb structures and comb-filter frequency-dependent phase shifts used
    # for decorrelation change the mono sum (L+R)/2 through cross-feed comb patterns
    # → MFCC-Pearson + centroid-CV shift vs. original mono reference. Additionally,
    # optional HF harmonics ("air", tape warmth, vinyl sheen) add spectral energy in
    # the MFCC-sensitive high-register → false P2 timbre regression vs. mono reference.
    "phase_32": {
        "timbre_authentizitaet"
    },  # Mono-to-stereo: Schroeder decorrelation (comb-filter in mono sum) + HF harmonic synthesis → MFCC-Pearson + centroid-CV change vs. mono reference → false P2 timbre regression
    # Stereo width limiter (M/S soft-knee Side-channel compression):
    # Scales the Side channel (S = (L−R)/2) with a frequency-dependent gain ≤ 1.
    # Reconstructed: L = M + S×gain, R = M − S×gain. The mono sum M = (L+R)/2 is
    # unaffected, but individual L/R channels change spectral content proportionally
    # to the applied Side gain. If the MFCC proxy runs on the LEFT channel (or the
    # first channel of the np.ndarray), spectral content of L shifts vs. the
    # pre-limiting reference → MFCC-Pearson + centroid-CV register change → false P2.
    "phase_33": {
        "timbre_authentizitaet"
    },  # Stereo width limiter: M/S Side soft-knee compression changes L/R channel spectral distribution (mono sum M preserved) → MFCC-Pearson + centroid-CV vs. wide-stereo reference → false P2 timbre regression
    # Output format optimization (multi-band loudness shaping + TruePeak + SRC + dither):
    # Pure LUFS gain and lossless SRC are scale-invariant for all ratio-based proxies.
    # However, format-specific multi-band frequency-dependent loudness shaping shifts
    # the spectral envelope → MFCC c1–c3 + centroid-CV register change — identical
    # root cause to phase_40 (loudness normalization). TruePeak limiting at −1 dBTP
    # is normally very light (prevents intersample overs only) → micro_dynamics, groove,
    # emotionalitaet not excluded unless confirmed in production logs.
    "phase_41": {
        "artikulation",
        "timbre_authentizitaet",
    },  # Output format optimization: multi-band loudness shaping + TruePeak limiting shifts spectral envelope → MFCC c1-c3 + centroid-CV false P2 (identical root cause to phase_40). artikulation excluded: loudness normalization + dithering modifies onset-energy peaks → quick proxy saturates at 1.0 pre-phase (mean_peaks/0.01 clips) then drops after processing → false P1 catastrophic regression (0.4803 observed, before=1.000→after=0.520, 2026-04-02).
    # Spatial enhancement (cross-feed early reflections + Schroeder all-pass diffusion):
    # 4 early reflections (6–22 ms, −8 to −16 dB, dry_wet=0.18) are cross-fed:
    # L_out += gain × dry_wet × delayed_R (and vice versa). This DOES change the mono
    # sum M = (L+R)/2 by introducing delayed cross-channel copies → comb-filtering
    # pattern in M → MFCC-Pearson + centroid-CV shift → false P2 timbre regression.
    # emotionalitaet excluded: cross-feed reflections add short-decay tail after
    # transients → crest-factor (peak/RMS ratio) decreases → false P3 regression
    # (same mechanism as phase_22: adding post-peak tail energy compresses crest).
    # waerme excluded: early reflections add diffuse energy across the spectrum
    # including the 200–800 Hz warmth band → warmth ratio E(200-800)/E(800-3000)
    # shifts → false P4 regression.
    "phase_46": {
        "timbre_authentizitaet",
        "emotionalitaet",
        "waerme",
        "raumtiefe",  # §2.55 + §0 2026-04-27: raumtiefe-Boost durch Reflexionen darf phase_46 nicht zu höherer Strength treiben (Gesang-Distanz-Bug)
    },  # Spatial enhancement: cross-feed early reflections modify mono sum (comb-filter in M) + add post-transient tail energy (crest-factor drop) + mid-band reflection energy (warmth ratio change) → false P2/P3/P4 regressions
    # Stereo width enhancer (STFT-based frequency-dependent M/S width:
    # LF×0.6 / MF×1.0 / HF×1.15, plus allpass decorrelation delays 17.1/19.7/23.3 ms):
    # STFT-based frequency-dependent Side scaling changes the spectral distribution of
    # both L and R channels (L = M + S×freq_factor). The HF enhancement (×1.15 above
    # 8 kHz) raises HF energy in the stereo image → centroid-CV shifts upward in L and
    # R → MFCC higher-order coefficients change vs. the unwidened reference → false P2
    # timbre regression (identical mechanism to phase_39 air band, phase_21/phase_25).
    "phase_48": {
        "timbre_authentizitaet",
        "raumtiefe",  # §2.55 + §0 2026-04-27: HF-Side-Widening (×1.15) erhöht Raumtiefe-Score — darf PMGG nicht zu höherer Strength treiben (Gesang-Distanz-Bug)
    },  # Stereo width enhancer: STFT frequency-dependent M/S Side scaling (HF ×1.15) changes L/R spectral distribution → MFCC higher-order coefficients + centroid-CV shift → false P2 timbre regression (identical mechanism to phase_39 air band)
    # ── §2.54 CIG-PMGG-Synchronisation: Phasen ohne bisher existierende PMGG-Einträge ───────────────────
    # Groove-echo cancellation (inner-groove vinyl pre-echo): removes pre-echo artefact.
    # timbre_authentizitaet: pre-echo spectral coloration is removed → MFCC-Pearson shifts vs. pre-echo-bearing reference.
    "phase_61": {
        "authentizitaet",
        "timbre_authentizitaet",
    },  # Groove-echo cancellation: pre-echo phantom chroma removed → chromagram + MFCC fingerprint change vs. pre-echo-distorted reference (§2.44)
    # authentizitaet: stereo-field chromagram fingerprint changes vs. crosstalk-distorted reference (§2.46).
    # timbre_authentizitaet: spectral crosstalk coloration removed → MFCC-Pearson shifts intentionally.
    "phase_62": {
        "authentizitaet",
        "timbre_authentizitaet",
    },  # Crosstalk cancellation: inter-channel spectral leakage removed → stereo fingerprint change vs. crosstalk-distorted reference (§2.46)
    # Modulation noise reduction (signal-adaptive spectral gating) — same root-cause class as phase_29:
    # authentizitaet: signal-dependent noise floor smooths log-spectrum valleys → after removal, valleys reappear → roughness rises → false P1
    # natuerlichkeit: spectral gating changes MFCC-smoothness proxy (same mechanism as OMLSA in phase_03)
    # timbre_authentizitaet: frequency-selective gain G(f) shifts spectral centroid / MFCC-Pearson
    # tonal_center: signal-adaptive gain per frequency band alters chroma bin energy distribution → K-S false P2
    # artikulation: signal-dependent gate changes onset rise-time vs. noise-bearing reference
    "phase_59": {
        "authentizitaet",
        "natuerlichkeit",
        "timbre_authentizitaet",
        "tonal_center",
        "artikulation",
    },  # Modulation noise reduction (signal-adaptive spectral gating): identical false-regression root causes as phase_29 (broadband NR)
    # Inner groove distortion repair (position-adaptive H3+ suppression):
    # authentizitaet: removing H3+ harmonic distortion products changes spectral roughness pattern vs. distorted reference → false P1
    # timbre_authentizitaet: MFCC-Pearson + centroid-CV shift as harmonic structure is modified (H3+ removed)
    "phase_60": {
        "authentizitaet",
        "timbre_authentizitaet",
    },  # Inner groove distortion repair: H3+ harmonic suppression changes spectral fingerprint vs. THD-distorted reference (§2.44)
    # Intermodulation distortion reduction (bispectrum-informed M/S notch filtering):
    # authentizitaet: IMD products contribute to chromagram (sum/difference frequencies create phantom pitch-class energy);
    #   after removal chromagram fingerprint changes vs. IMD-distorted reference → false P1
    # timbre_authentizitaet: MFCC-Pearson + centroid-CV shift as IMD-notched frequency content is removed
    "phase_63": {
        "authentizitaet",
        "timbre_authentizitaet",
    },  # IMD reduction: bispectrum-informed M/S notch removes sum/difference products → chromagram + MFCC fingerprint change vs. distorted reference (§2.44)
}
# pylint: enable=line-too-long

# §v10.3 Media-Defect-Verifier: ALLE PMGG-Phasen mit alternativen Proxies
# Dynamisch aus cassette_defect_verifier._PHASE_CATEGORIES geladen.
_CASSETTE_VERIFIER_PHASES: frozenset[str] = frozenset(
    {
        "phase_24",
        "phase_56",
        "phase_57",
        "phase_59",
        "phase_24_dropout_repair",
        "phase_56_spectral_band_gap_repair",
        "phase_57_print_through_reduction",
        "phase_59_modulation_noise_reduction",
    }
)


def _get_all_verifier_phases() -> frozenset[str]:
    """§v10.3 Lazy-load aller Phasen aus dem Media-Defect-Verifier."""
    try:
        from backend.core.cassette_defect_verifier import _PHASE_CATEGORIES

        return frozenset(_PHASE_CATEGORIES.keys())
    except Exception as e:
        logger.warning("per_Verarbeitungsschritt_musical_goals_gate.py::_get_all_verifier_phases Ersatzpfad: %s", e)
        return _CASSETTE_VERIFIER_PHASES


def _get_sample_duration(phase_id: str) -> float:
    """Gibt phasen-adaptive Stichprobenlänge zurück (§9.7.3).

    Minimale Sample-Dauer: 1.0 s (kein Unterschreiten).
    Maximale Sample-Dauer: SAMPLE_DURATION_S (5.0 s).
    Phase-ID-Matching via startswith — robust gegen Suffix-Varianten.
    """
    for prefix, dur in PHASE_SAMPLE_DURATIONS.items():
        if phase_id.startswith(prefix):
            return max(1.0, min(dur, SAMPLE_DURATION_S))
    return SAMPLE_DURATION_S


# ---------------------------------------------------------------------------
# §PMGG-Restorative: Phasen die Defekte ENTFERNEN statt Klang zu formen.
# Defekte erhöhen viele Metriken künstlich über ihren sauberen Wert:
#   Rauschen füllt Spektraltäler → AuthentizitaetProxy erscheint HOCH.
#   HF-Rauschen → BrillanzProxy erscheint HOCH.
#   Hall → Wärme/Transparenz erscheinen HOCH.
# Nach Restaurierung fallen diese Scores auf reale Werte → PMGG wertet es
# als Regression obwohl es eine Verbesserung ist.
# Lösung: Für restorative Phasen wird scores_before auf die normativen
# Qualitäts-Schwellwerte gedeckelt (§15 Musical Goals, Restoration-Modus).
# Dadurch kann keine defekt-inflationierte Baseline eine false-positive
# Regression auslösen. Echter Schaden (Score unter Schwelle) wird weiterhin erkannt.
# ---------------------------------------------------------------------------
# §TFS: Phases where Temporal Fine Structure coherence is measured.
# These are heavy spectral-modification phases that can disrupt sub-1.5 kHz
# instantaneous phase relationships (pitch, binaural cues, consonant texture).
# Scientific basis:
#   Moore (2014) J Acoust Soc Am 135:412 — updated TFS role in pitch/speech/music
#   Moore & Sek (2009) J Acoust Soc Am 125:3530 — TFS importance in music perception
#   Lorenzi et al. (2006) PNAS 103:18866 — foundational TFS role in hearing (speech)
_TFS_SENSITIVE_PHASES: frozenset[str] = frozenset(
    {
        "phase_03",  # Broadband denoise — spectral shaping disrupts TFS
        "phase_09",  # BANQUET blind denoising — full-band spectral mod
        "phase_20",  # Reverb reduction (SGMSE+) — diffuse field removal affects phase
        "phase_29",  # Tape hiss reduction (DeepFilterNet) — HF removal cascades into TFS
        "phase_49",  # Advanced dereverb — aggressive spectral subtraction
    }
)
_TFS_COHERENCE_THRESHOLD: float = 0.85  # Below this → phase disrupted fine structure
_TFS_RETRY_TRIGGER: float = 0.15  # TFS delta > this AND P1/P2 regression → extra retry

_RESTORATIVE_PHASES: frozenset[str] = frozenset(
    # pylint: disable=line-too-long
    {
        "phase_01",  # Click removal
        "phase_02",  # Hum removal (Kammfilter)
        "phase_03",  # Broadband denoise (OMLSA + ResembleEnhance)
        "phase_04",  # EQ correction (RIAA/NAB de-emphasis inversion) — HF/LF energy redistribution inflates brillanz/waerme proxies
        "phase_05",  # Rumble filter (subtractive LF cleanup)
        "phase_09",  # BANQUET blind denoising
        "phase_12",  # Wow/flutter correction (§2.44 Reference-Paradoxon: pitch dewarping changes chroma vs. wobble-distorted reference)
        "phase_14",  # Stereo phase correction (multi-band alignment) — fixes carrier phase misalignment; stereo-fingerprint changes vs. mis-aligned reference
        "phase_15",  # Stereo balance correction — corrects L/R imbalance defect; energy shift changes authentizitaet proxy vs. imbalanced reference
        "phase_18",  # Noise gate (Silero VAD)
        "phase_19",  # De-esser — sibilance carrier distortion (vinyl HF, cassette) inflates brillanz; post-reduction drop is defect-removal, not regression
        "phase_20",  # Reverb reduction (SGMSE+)
        "phase_23",  # Spectral inpainting / gap-fill (FlashSR); §4.7c POCS n_iter=2–5 vor PGHI
        "phase_24",  # Dropout repair (FlashSR)
        "phase_25",  # Azimuth correction — tape head misalignment repair; HF balance changes vs. mis-aligned reference
        "phase_27",  # Click/pop removal
        "phase_28",  # Surface noise profiling (vinyl — broadband noise inflates proxy baselines identically to phase_03/phase_29)
        "phase_29",  # Tape hiss reduction (DeepFilterNet v3 II)
        "phase_30",  # DC offset / near-DC drift removal
        "phase_31",  # Speed/pitch correction (pYIN + WSOLA) — corrects turntable/tape speed deviation; tonal_center/groove proxies change vs. pitch-deviated checkpoint (§2.44 Reference-Paradoxon identical to phase_12)
        "phase_49",  # Advanced dereverb
        "phase_50",  # STFT spectral inpainting (bin interpolation)
        "phase_55",  # Diffusion inpainting (CQTdiff+) — gap reconstruction: silence-baseline inflates tonal_center/waerme
        "phase_56",  # Spectral band gap repair (HEAD_WEAR)
        "phase_57_print_through_reduction",  # Print-through reduction (bidirectional LMS)
        "phase_59",  # Tape modulation noise reduction — carrier-induced FM noise removal inflates tonal_center proxy
        "phase_60",  # Inner groove distortion repair (vinyl) — THD reduction changes spectral fingerprint vs. distorted reference
        "phase_62",  # Crosstalk cancellation — early stereo channel separation repair; stereo fingerprint changes vs. crosstalk-distorted reference
        "phase_63",  # Intermodulation distortion reduction (M/S-domain) — IMD artefact removal changes spectral energy vs. distorted reference
    }  # pylint: enable=line-too-long
)

_CANONICAL_15_KEYS: frozenset[str] = frozenset(
    {
        "natuerlichkeit",
        "authentizitaet",
        "tonal_center",
        "timbre_authentizitaet",
        "artikulation",
        "transient_energie",
        "emotionalitaet",
        "micro_dynamics",
        "groove",
        "transparenz",
        "waerme",
        "bass_kraft",
        "separation_fidelity",
        "brillanz",
        "spatial_depth",
    }
)

# Abgeleitete Threshold-Dicts (nur die 15 kanonischen Short-Form-Keys)
_CANONICAL_THRESHOLDS_RESTORATION: dict[str, float] = {k: v for k, v in _CM_REST.items() if k in _CANONICAL_15_KEYS}

_CANONICAL_THRESHOLDS_STUDIO2026: dict[str, float] = {k: v for k, v in _CM_STU.items() if k in _CANONICAL_15_KEYS}

# Default alias for backward compatibility (Restoration-Modus)
_CANONICAL_THRESHOLDS: dict[str, float] = _CANONICAL_THRESHOLDS_RESTORATION


def _get_canonical_thresholds(is_studio_2026: bool = False) -> dict[str, float]:
    """Gibt mode-appropriate canonical thresholds (Spec 09 / §09.1 calibration_matrix.py) zurück.

    P1/P2 are stricter in Studio 2026 — more aggressive enhancement requires a
    stronger guard against loss of naturalness and authenticity.
    P3–P5 use material-universal floors for both modes; per-song targets above
    these floors are computed by the adaptive layer (§2.31 + §09.2 + §2.56).
    """
    if is_studio_2026:
        return _CANONICAL_THRESHOLDS_STUDIO2026
    return _CANONICAL_THRESHOLDS_RESTORATION


# §v10.16: Binäre Suche (Skalpell) statt linearer Stärke-Reduktion (Vorschlaghammer).
# Die optimale Stärke wird via Intervallhalbierung auf ±0.8% genau gefunden.
# 8 Iterationen = 1/256 Auflösung. Alte _RETRY_STRENGTHS als Fallback.
_RETRY_STRENGTHS: list[float] = [0.65, 0.50, 0.35, 0.25, 0.15]  # Fallback
_BINARY_SEARCH_MAX_ITERS: int = 12
_BINARY_SEARCH_PRECISION: float = 0.005  # 0.5% — darunter dominiert PMGG-Messrauschen

# §2.29a ML-deterministic Phasen: Inference-Output ist bei gleichem Input
# identisch, unabhängig vom strength-Parameter.  Bei PMGG-Retries wird nur
# Wet/Dry-Reblending variiert — keine Re-Inferenz.
# Phase-ID-Prefixes (startswith-Match) für robustes Matching.
_ML_DETERMINISTIC_PHASES: frozenset[str] = frozenset(
    {
        "phase_03",  # OMLSA + ResembleEnhance (ML-Hybrid Denoising)
        "phase_06",  # FlashSR (neurale Bandwidth-Extension)
        "phase_09",  # BANQUET ONNX (Blind-Denoising)
        "phase_12",  # FCPE/CREPE/pYIN (f₀-Schätzung) — Timing-Phase, kein Wet/Dry
        "phase_18",  # Silero VAD (Binary-Mask)
        "phase_19",  # De-Esser+VocalStack: process() ignoriert strength → Wet/Dry reicht
        "phase_20",  # SGMSE+ (Reverb-Separation) — nur ML-deterministic wenn SGMSE+ geladen
        # WPE-Fallback ist strength-abhängiger DSP → _phase20_is_ml_active() prüft zur Laufzeit
        "phase_23",  # FlashSR Inpainting (Spektral-Lückenfüllung)
        "phase_29",  # DeepFilterNet v3 II (HF-Denoising)
        "phase_42",  # BSRoFormer (Stem-Separation)
        "phase_56",  # FCPE/CREPE + Synthese (Spectral Band Gap Repair)
    }
)


def _material_key_from_phase_kwargs(phase_kwargs: dict[str, Any] | None) -> str:
    """Gibt normalized material key from phase kwargs zurück."""
    if not isinstance(phase_kwargs, dict):
        return "unknown"
    _raw = phase_kwargs.get("material_type", phase_kwargs.get("material", "unknown"))
    _txt = str(getattr(_raw, "value", _raw) or "unknown").strip().lower()
    if _txt.startswith("materialtype."):
        _txt = _txt.split(".", 1)[1]
    return _txt or "unknown"


def _phase_safe_strength_cap(phase_id: str, phase_kwargs: dict[str, Any] | None) -> float:
    """Conservative phase-specific cap to reduce P1/P2 drift cascades.

    These caps are intentionally material-adaptive and only applied to known
    high-risk phases (02/03/12/24/29/55) that repeatedly triggered rollback cascades.
    """
    _mat = _material_key_from_phase_kwargs(phase_kwargs)
    _caps: dict[str, dict[str, float]] = {
        "phase_02_hum_removal": {
            "vinyl": 0.34,
            "tape": 0.36,
            "reel_tape": 0.34,
            "shellac": 0.32,
            "wax_cylinder": 0.30,
            "wire_recording": 0.30,
            "cassette": 0.36,
            "cd_digital": 0.40,
            "dat": 0.40,
            "mp3_low": 0.38,
            "mp3_high": 0.40,
            "aac": 0.40,
            "streaming": 0.40,
            "unknown": 0.36,
        },
        "phase_03_denoise": {
            "vinyl": 0.42,
            "tape": 0.44,
            "reel_tape": 0.42,
            "shellac": 0.40,
            "wax_cylinder": 0.38,
            "wire_recording": 0.38,
            # §2.45 Minimal-Intervention: Kassette+mp3_low-Kette → transient_energie -0.125
            # nach Phase_03 bei 0.44 — Phase_29 folgt nach; nur eine NR-Phase soll voll laufen
            "cassette": 0.30,
            "cd_digital": 0.48,
            "dat": 0.48,
            "mp3_low": 0.46,
            "mp3_high": 0.48,
            "aac": 0.48,
            "streaming": 0.48,
            "unknown": 0.44,
        },
        "phase_12_wow_flutter_fix": {
            "vinyl": 0.62,
            "tape": 0.70,
            "reel_tape": 0.66,
            "shellac": 0.56,
            "wax_cylinder": 0.52,
            "wire_recording": 0.52,
            # §V27+PSOLA: Kassetten-Flutter ist breitbandig irregular; PSOLA bei >0.35
            # erzeugt Pitch-Glitches an Frame-Grenzen → PITCH_DRIFT +0.242 in Messungen
            "cassette": 0.35,
            "lacquer_disc": 0.58,
            "unknown": 0.60,
        },
        "phase_24_dropout_repair": {
            "vinyl": 0.58,
            "tape": 0.62,
            "reel_tape": 0.60,
            "shellac": 0.54,
            "wax_cylinder": 0.52,
            "wire_recording": 0.52,
            "cassette": 0.62,
            "cd_digital": 0.64,
            "dat": 0.64,
            "mp3_low": 0.66,
            "mp3_high": 0.64,
            "aac": 0.64,
            "streaming": 0.64,
            "unknown": 0.60,
        },
        "phase_29_tape_hiss_reduction": {
            "vinyl": 0.34,
            "tape": 0.36,
            "reel_tape": 0.35,
            "shellac": 0.32,
            "wax_cylinder": 0.30,
            "wire_recording": 0.30,
            # §2.31 chain-min: cassette+mp3_low → HF bereits durch MP3 reduziert;
            # OMLSA bei 0.36 vernichtet Formanten (authentizitaet 0.98→0.48 gemessen)
            "cassette": 0.22,
            "cd_digital": 0.40,
            "dat": 0.40,
            "mp3_low": 0.38,
            "mp3_high": 0.40,
            "aac": 0.40,
            "streaming": 0.40,
            "unknown": 0.36,
        },
        # §2.45 Minimal-Intervention: Phase_34 M/S-Processing verschlechtert
        # transient_energie bei Kassette (-0.047 gemessen) → Cap auf 0.40 setzt
        # PMGG unter den internen bypass_threshold (0.45) → Phase skippt bei Kassette
        "phase_34_mid_side_processing": {
            "vinyl": 0.70,
            "tape": 0.70,
            "reel_tape": 0.70,
            "shellac": 0.65,
            "wax_cylinder": 0.60,
            "cassette": 0.40,
            "mp3_low": 0.55,
            "mp3_high": 0.60,
            "unknown": 0.65,
        },
        # §0p Vokal-Supremacy: Doppel-De-Essing (phase_19 + phase_43) bei aktivem
        # PANNs-Singing erzeugt sibilance_naturalness 0.795 < 0.80 — zweiter
        # De-Esser wird durch niedrigen Cap bremst, wenn Phase_19 bereits lief
        "phase_43_ml_deesser": {
            "vinyl": 0.55,
            "tape": 0.55,
            "cassette": 0.30,
            "mp3_low": 0.35,
            "unknown": 0.50,
        },
        "phase_55_diffusion_inpainting": {
            "vinyl": 0.46,
            "tape": 0.50,
            "reel_tape": 0.48,
            "shellac": 0.42,
            "wax_cylinder": 0.40,
            "wire_recording": 0.40,
            "cassette": 0.50,
            "cd_digital": 0.54,
            "dat": 0.54,
            "mp3_low": 0.56,
            "mp3_high": 0.54,
            "aac": 0.54,
            "streaming": 0.54,
            "unknown": 0.48,
        },
    }
    _for_phase = _caps.get(phase_id)
    if not _for_phase:
        return 1.0
    return float(_for_phase.get(_mat, _for_phase.get("unknown", 1.0)))


def _resolve_team_context_policy(phase_id: str, phase_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Gibt PMGG team-coordination policy derived from prior phase context zurück.

    The policy is advisory and only affects PMGG retry/goal-check behavior.
    It does not disable final export gates and does not bypass safety guards.
    """
    _policy: dict[str, Any] = {
        "goal_exclusions": set(),
        "threshold_multiplier": 1.0,
        "strength_cap": 1.0,
        "reason": "",
    }
    if not isinstance(phase_kwargs, dict):
        return _policy

    _ctx = phase_kwargs.get("prior_phase_context")
    if not isinstance(_ctx, dict) or not _ctx:
        return _policy

    _is_phase50 = str(phase_id).startswith("phase_50")
    _hf_chain_applied = bool(
        _ctx.get("harmonic_restoration_applied")
        or _ctx.get("frequency_restoration_applied")
        or _ctx.get("spectral_super_resolution_applied")
    )

    # Generic all-phase transition policy (module/phase complete coverage)
    # ---------------------------------------------------------------
    # Uses phase ontology types to derive conservative PMGG adjustments for
    # potentially conflicting transition pairs. This keeps behavior centralized
    # and avoids manual per-phase hotfixes.
    try:
        from backend.core.phase_ontology import get_phase_type

        _cur_t = getattr(get_phase_type(str(phase_id)), "name", "")
        _prev_t = str(_ctx.get("last_phase_type", "") or "")
        _transition = (_prev_t, _cur_t)
        _TRANSITION_POLICY: dict[tuple[str, str], dict[str, Any]] = {
            # Prior additive reconstruction followed by subtractive cleanup:
            # avoid over-penalizing intentional HF/timbre changes.
            ("ADDITIVE", "SUBTRACTIVE"): {
                "goal_exclusions": {"brillanz", "transparenz"},
                "threshold_multiplier": 1.08,
                "strength_cap": 0.90,
                "reason": "transition_additive_to_subtractive",
            },
            # Diffusion/ML-generated content followed by subtractive cleanup:
            # articulation/micro-dynamics proxies often overreact.
            ("ML_GENERATIVE", "SUBTRACTIVE"): {
                "goal_exclusions": {"artikulation", "micro_dynamics"},
                "threshold_multiplier": 1.08,
                "strength_cap": 0.90,
                "reason": "transition_mlgen_to_subtractive",
            },
            # Dynamics processing after additive synthesis: preserve reconstructed
            # transients and avoid aggressive PMGG-driven attenuation.
            ("ADDITIVE", "DYNAMICS"): {
                "goal_exclusions": {"artikulation"},
                "threshold_multiplier": 1.05,
                "strength_cap": 0.92,
                "reason": "transition_additive_to_dynamics",
            },
            # Corrective after additive: tonal/timbre proxies can reflect intentional
            # spectral re-centering rather than real degradation.
            ("ADDITIVE", "CORRECTIVE"): {
                "goal_exclusions": {"timbre_authentizitaet"},
                "threshold_multiplier": 1.05,
                "strength_cap": 0.94,
                "reason": "transition_additive_to_corrective",
            },
        }
        _tp = _TRANSITION_POLICY.get(_transition)
        if isinstance(_tp, dict):
            _policy["goal_exclusions"] |= set(_tp.get("goal_exclusions", set()))
            _policy["threshold_multiplier"] = max(
                float(_policy["threshold_multiplier"]), float(_tp.get("threshold_multiplier", 1.0))
            )
            _policy["strength_cap"] = min(float(_policy["strength_cap"]), float(_tp.get("strength_cap", 1.0)))
            if not _policy["reason"]:
                _policy["reason"] = str(_tp.get("reason", ""))
    except Exception as e:
        logger.warning("per_Verarbeitungsschritt_musical_goals_gate.py::unbekannter Ersatzpfad: %s", e)

    # Team rule: if prior phases already restored HF content, phase_50 should
    # avoid treating those bins as "damage" via indirect metric pressure.
    if _is_phase50 and _hf_chain_applied:
        _policy["goal_exclusions"] = {"brillanz", "transparenz", "timbre_authentizitaet"}
        _policy["threshold_multiplier"] = 1.15
        _policy["strength_cap"] = 0.80
        _policy["reason"] = "phase50_after_hf_restoration"

    return _policy


def _allow_emergency_retries(
    phase_id: str,
    worst_priority: int,
    best_regression: float,
    catastrophic_threshold: float,
    team_policy: dict[str, Any] | None,
) -> bool:
    """Gibt whether PMGG emergency retries should run for this phase zurück.

    Team-policy can disable emergency retries when a measured regression is likely
    a proxy artifact caused by intentional prior restoration steps.
    """
    if not (best_regression > catastrophic_threshold and worst_priority <= 2):
        return False

    if isinstance(team_policy, dict):
        _reason = str(team_policy.get("reason", ""))
        # phase_50 after HF restoration (phase_06/phase_07/phase_23):
        # emergency low-strength loops are typically wasted because the observed
        # P1/P2 proxy drop stems from intentional HF changes, not real damage.
        if phase_id.startswith("phase_50") and _reason == "phase50_after_hf_restoration":
            return False

    return True


def _phase20_is_ml_active() -> bool:
    """Gibt True when SGMSE+ is currently loaded in the ML budget (§2.29a) zurück.

    phase_20 is ML-deterministic only when the SGMSE+ model is actually resident
    in memory.  When SGMSE+ was blocked by ml_memory_budget (OOM pressure) and
    the WPE-DSP fallback is active instead, wet/dry blending cannot represent the
    full range of WPE's strength-dependent predictor-order parameter.  In that
    case phase_20 must be treated as a strength-dependent DSP phase — re-run on
    every PMGG retry.
    """
    try:
        from backend.core.ml_memory_budget import get_status

        return "SGMSE+" in get_status().get("models", {})
    except Exception as e:
        logger.warning(
            "per_Verarbeitungsschritt_musical_goals_gate.py::_Verarbeitungsschritt20_is_ml_active Ersatzpfad: %s", e
        )
        return False  # Safe default: DSP path — must re-run


def _resolve_transfer_chain_depth(value: int | None) -> int:
    """§G86 (GEBOTE.md): Default nur aus CalibrationContext."""
    from backend.core.defect_to_audibility import _resolve_transfer_chain_depth as _resolve

    return _resolve(value)


def _get_adaptive_threshold(
    restorability_score: float,
    material_type: str = "unknown",
    transfer_chain_depth: int | None = None,
) -> float:
    """§2.29/§2.54 Material- und Restorability-adaptiver REGRESSION_THRESHOLD.

    Args:
        restorability_score: RestorabilityEstimator-Score ∈ [0, 100]
        material_type: Carrier-Materialklasse (z.B. 'vinyl', 'shellac', 'cd_digital')
        transfer_chain_depth: Chain-Depth für depth-adaptive Toleranz (§v10.120)

    Returns:
        Adaptiver Schwellwert ∈ [0.012, 0.070].
        Analog/physische Träger erhalten einen Material-Bonus, da Carrier-Repair-
        Phasen das Signal intentional ändern (Referenz-Paradoxon §2.44).
    """
    transfer_chain_depth = _resolve_transfer_chain_depth(transfer_chain_depth)
    # Restorability-tier Basis
    if restorability_score >= 70.0:
        base = REGRESSION_THRESHOLD_GOOD
    elif restorability_score >= 40.0:
        base = REGRESSION_THRESHOLD_FAIR
    else:
        base = REGRESSION_THRESHOLD_POOR
    # Material-Bonus: analog-physische Träger benötigen mehr Toleranz (§2.54)
    bonus = _MATERIAL_THRESHOLD_BONUS.get(material_type.lower(), 0.003)
    # §v10.120 Depth-Bonus: tiefere Ketten → mehr legitime Regression (§2.44)
    _depth = max(1, int(transfer_chain_depth))
    _depth_bonus = max(0, _depth - 2) * 0.008  # +0.008 pro Depth-Stufe ab 3
    threshold = base + bonus + _depth_bonus
    # Hard-Cap: nie enger als 0.012 (Messrauschen), nie lockerer als 0.090
    # §v10.200: Obergrenze 0.070→0.090 — Kassetten-Bonus (0.020) + depth≥5 (0.024) = 0.089
    return float(np.clip(threshold, 0.012, 0.090))


# All 15 Musical Goals are checked per-phase — DSP-only proxies, no ML (≤ 200 ms total §2.29).
# "natuerlichkeit" uses an MFCC-smoothness DSP proxy internally but is exposed under its
# canonical key so GoalApplicabilityFilter intersection (§2.32) works correctly.
FAST_GOALS_SUBSET: list[str] = [
    "brillanz",
    "waerme",
    "groove",
    "tonal_center",
    "natuerlichkeit",
    "timbre_authentizitaet",
    "transient_energie",
    # 8 neu (DSP-Proxies, v10.0.0):
    "bass_kraft",
    "authentizitaet",
    "emotionalitaet",
    "transparenz",
    "spatial_depth",
    "micro_dynamics",
    "separation_fidelity",
    "artikulation",
]


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class PhaseGateLogEntry:
    """Eintrag im phase_gate_log für eine Phase."""

    phase_id: str
    action: str  # "passed" | "retry1" | ... | "retry5" | "best_effort" | "best_effort_rN"
    goal_regressions: dict[str, float]  # Ziel → Δ-Score
    strength_used: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)  # TFS coherence, vocal_intimacy, etc.
    # §DEBUG: Vollständige Goal-Snapshots vor/nach der Phase — für PipelineTrace / aurik-debug.
    # Werden von wrap_phase() befüllt wenn scores_before/after verfügbar; andernfalls leer.
    scores_before: dict[str, float] = field(default_factory=dict)
    scores_after: dict[str, float] = field(default_factory=dict)


@dataclass
class PhaseGateResult:
    """Ergebnis der wrap_phase()-Operation."""

    audio: np.ndarray
    scores_after: dict[str, float]
    log_entry: PhaseGateLogEntry
    rolled_back: bool


# ---------------------------------------------------------------------------
# Singleton (§3.2)
# ---------------------------------------------------------------------------
_instance: PerPhaseMusicalGoalsGate | None = None
_lock = threading.Lock()


def get_phase_gate() -> PerPhaseMusicalGoalsGate:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PerPhaseMusicalGoalsGate()
    return _instance


# ---------------------------------------------------------------------------
# Schnell-Metriken (ohne MERT, ohne CDPAM, ohne externe ML-Modelle)
# ---------------------------------------------------------------------------


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson-Korrelation mit Längen-Matching und NaN/Inf-Sicherheit.

    Returns 0.0 bei Fehler oder zu wenig Daten.
    """
    n = min(len(a), len(b))
    if n < 4:
        return 0.0
    try:
        av = a[:n].ravel()
        bv = b[:n].ravel()
        if float(np.std(av)) < 1e-12 or float(np.std(bv)) < 1e-12:
            return 1.0 if np.allclose(av, bv, atol=1e-12, rtol=1e-6) else 0.0
        _a = av - av.mean()
        _b = bv - bv.mean()
        _na = float(np.linalg.norm(_a))
        _nb = float(np.linalg.norm(_b))
        r = float(np.dot(_a, _b) / (_na * _nb + 1e-10))
        return r if math.isfinite(r) else 0.0
    except Exception as e:
        logger.warning("per_Verarbeitungsschritt_musical_goals_gate.py::_safe_pearson Ersatzpfad: %s", e)
        return 0.0


def _get_precise_metric_instances() -> dict[str, Any]:
    """Lazy-load a small set of production musical-goal metrics for PMGG.

    These are used selectively for the most decision-critical goals where local
    DSP proxies are materially less precise than the canonical metric.
    """
    global _PRECISE_METRICS  # pylint: disable=global-statement
    if _PRECISE_METRICS is None:
        with _PRECISE_METRICS_LOCK:
            if _PRECISE_METRICS is None:
                try:
                    from backend.core.musical_goals.musical_goals_metrics import (
                        ArticulationMetric,
                        MicroDynamicsMetric,
                        SeparationFidelityMetric,
                    )

                    _PRECISE_METRICS = {
                        # brillanz intentionally omitted: §9.7.12 HF-crest-factor quick proxy
                        # is symmetric and SNR-robust for PMGG delta checks.  The absolute
                        # BrillanzMetric._measure_absolute() (ISO-226 HF-energy ratio) is still
                        # SNR-dependent → would show false drop after denoising even without the
                        # reference-preservation penalty.  Both scores_before and scores_after
                        # now use the crest-factor proxy consistently → symmetric, no false regressions.
                        # The canonical BrillanzMetric still runs in the final export gate.
                        # waerme intentionally omitted: §9.7.14 warmth-ratio quick proxy
                        # (E_200-800 / E_800-3000) is reverb-invariant.  WaermeMetric._measure_absolute()
                        # uses ISO-226 mid/total ratio which drops after dereverb → false regression.
                        # transparenz intentionally omitted: §9.7.13 multi-band crest-factor quick
                        # proxy is SNR-robust.  TransparenzMetric.measure() also has no reference=
                        # parameter → precise override was silently failing (TypeError) already.
                        # natuerlichkeit intentionally omitted: NatuerlichkeitMetric uses
                        # CREPE ML inference (1–4 s/call) with dynamic weight switching
                        # based on CREPE load state.  Between scores_before (CREPE not
                        # yet loaded → w_crepe=0.0) and scores_after (CREPE loaded →
                        # w_crepe=0.18) the absolute score shifts non-deterministically,
                        # creating systematic false P1 regressions in phase_03/phase_02.
                        # The DSP proxy in _measure_quick with §9.7.5 reference-aware
                        # preservation correction is more reliable for PMGG delta checks.
                        # The canonical NatuerlichkeitMetric still runs in the final
                        # export quality gate (MusicalGoalsChecker).
                        #
                        # tonal_center intentionally omitted (§2.29b, v10.0.0):
                        # TonalCenterMetric uses librosa.feature.chroma_stft and applies a
                        # binary key-shift penalty (1 semitone → score ≤ 0.50; ≥2 → 0.0).
                        # This causes systematic catastrophic false P2 regressions in phases
                        # that legitimately change harmonic-percussive balance or energy
                        # distribution without changing the musical key:
                        #   - phase_08 TDP/HPSS: Δ=0.5612 observed (transient reshaping
                        #     shifts dominant chroma class in librosa by 1 semitone)
                        #   - phase_36 transient shaper: Δ=0.3231 observed (same mechanism)
                        #   - phase_49 advanced dereverb: Δ=0.5312 observed (reverb decay
                        #     energy removal changes chroma frame distribution)
                        # Root cause: librosa chroma_stft is sensitive to energy-envelope
                        # changes even when pitch content is unchanged.  A 1-semitone
                        # apparent shift (e.g. A→A# due to energy redistribution) triggers
                        # the 50% penalty → catastrophic threshold breached → 5+ retries +
                        # emergency retries → Watchdog timeout.
                        # The K-S quick proxy in _measure_quick is the correct PMGG tool:
                        # it uses a multi-frame chroma sum → argmax is stable under energy
                        # redistribution that does not change the dominant pitch class.
                        # The canonical TonalCenterMetric still runs in the final export gate.
                        "micro_dynamics": MicroDynamicsMetric(),
                        "artikulation": ArticulationMetric(),
                        "separation_fidelity": SeparationFidelityMetric(),
                    }
                except Exception as exc:
                    logger.debug("PMGG precise metrics nicht verfuegbar: %s", exc)
                    _PRECISE_METRICS = {}
    return _PRECISE_METRICS


def _apply_precise_metric_overrides(
    scores: dict[str, float],
    audio: np.ndarray,
    sr: int,
    reference: np.ndarray | None = None,
) -> dict[str, float]:
    """Refine selected quick scores using canonical metric implementations."""
    t0 = time.perf_counter()
    precise_metrics = _get_precise_metric_instances()
    if not precise_metrics:
        return scores

    # §9.7.x Near-silence guard: precise metrics were designed for musical content.
    # Near-silence bypasses their internal silence paths and returns misleading scores
    # (e.g. MicroDynamicsMetric reliability-blend: 0.94; SeparationFidelity floor: 0.70).
    # The quick proxy in _measure_quick already sets all affected goals to 0.5 (neutral)
    # for near-silence — preserve that correct behavior by skipping precise overrides.
    _rms_guard = (
        float(np.sqrt(np.mean(audio.astype(np.float32) ** 2) + 1e-12))
        if audio.ndim == 1
        else float(
            np.sqrt(np.mean(np.mean(audio.astype(np.float32), axis=0 if audio.shape[0] <= 2 else 1) ** 2) + 1e-12)
        )
    )
    if _rms_guard < 1e-5:
        return scores  # proxy values (0.5 = neutral) are correct for silence

    # §9.7.7 Audio length cap: 2.5 s is sufficient for all precise metrics and
    # avoids long NMF/onset-detection runs in SeparationFidelityMetric /
    # ArticulationMetric on long audio samples.
    # Use start/middle/end slices to avoid first-segment bias on long tracks.
    _cap = int(2.5 * sr)

    def _cap_multisegment(arr: np.ndarray, cap: int) -> np.ndarray:
        if arr.ndim == 1:
            n = len(arr)
            if n <= cap:
                return arr
            seg = max(1, cap // 3)
            starts = [0, max(0, (n - seg) // 2), max(0, n - seg)]
            parts = [arr[s : s + seg] for s in starts]
            return np.concatenate(parts, axis=0)  # type: ignore[no-any-return]

        if arr.ndim == 2:
            is_channel_first = arr.shape[0] <= 2 and arr.shape[1] > arr.shape[0]
            time_len = arr.shape[1] if is_channel_first else arr.shape[0]
            if time_len <= cap:
                return arr
            seg = max(1, cap // 3)
            starts = [0, max(0, (time_len - seg) // 2), max(0, time_len - seg)]
            if is_channel_first:
                parts = [arr[:, s : s + seg] for s in starts]
                return np.concatenate(parts, axis=1)  # type: ignore[no-any-return]
            parts = [arr[s : s + seg, :] for s in starts]
            return np.concatenate(parts, axis=0)  # type: ignore[no-any-return]

        return arr

    audio = _cap_multisegment(audio, _cap)
    if reference is not None:
        reference = _cap_multisegment(reference, _cap)

    refined = dict(scores)
    for goal_name, metric in precise_metrics.items():
        try:
            if goal_name == "micro_dynamics":
                # Always reference-free: scores_before is measured without reference,
                # so scores_after must use the same absolute mode for a fair comparison.
                # Reference-based MicroDynamicsMetric gives 0.60+ baseline vs ~0.75×corr
                # for scores_after, creating systematic false regressions in PMGG.
                refined[goal_name] = float(metric.measure(audio, sr))
            elif goal_name in {
                "tonal_center",
                "artikulation",
                "separation_fidelity",
            }:
                refined[goal_name] = float(metric.measure(audio, sr, reference=reference))
            else:
                refined[goal_name] = float(metric.measure(audio, sr))
        except Exception as exc:
            logger.debug("PMGG precise metric override fehlgeschlagen for %s: %s", goal_name, exc)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if elapsed_ms > _PRECISE_OVERRIDE_WARN_MS:
        logger.warning(
            "PMGG precise overrides slow: %.1f ms for %d goals",
            elapsed_ms,
            len(precise_metrics),
        )
    return refined


def _measure_vocal_guard_features(
    audio: np.ndarray,
    sr: int,
    reference: np.ndarray | None = None,
) -> dict[str, float]:
    """Schätzt lightweight vocal-preservation features for PMGG quick scoring."""

    if audio.ndim == 2:
        mono = audio.mean(axis=0) if audio.shape[0] <= 2 else audio.mean(axis=1)
    else:
        mono = audio
    mono = np.nan_to_num(np.asarray(mono, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    ref_mono: np.ndarray | None = None
    if reference is not None:
        if reference.ndim == 2:
            ref_mono = reference.mean(axis=0) if reference.shape[0] <= 2 else reference.mean(axis=1)
        else:
            ref_mono = reference
        ref_mono = np.nan_to_num(
            np.asarray(ref_mono, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)
        match_len = min(len(mono), len(ref_mono))
        mono = mono[:match_len]
        ref_mono = ref_mono[:match_len]

    n_fft = 4096
    win = np.hanning(n_fft).astype(np.float32)

    def _mean_fft(signal_mono: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(signal_mono) >= n_fft:
            hop = n_fft // 2
            n_frames = min(64, max(1, (len(signal_mono) - n_fft) // hop))
            frames = np.stack(
                [np.abs(np.fft.rfft(signal_mono[idx * hop : idx * hop + n_fft] * win)) for idx in range(n_frames)]
            )
            mag = frames.mean(axis=0).astype(np.float32)
        else:
            mag = np.abs(np.fft.rfft(signal_mono, n=n_fft)).astype(np.float32)
        return mag, np.fft.rfftfreq(n_fft, d=1.0 / sr).astype(np.float32)

    def _band_energy(mag: np.ndarray, freqs_arr: np.ndarray, lo: float, hi: float) -> float:
        mask = (freqs_arr >= lo) & (freqs_arr < hi)
        if not np.any(mask):
            return 0.0
        return float(np.mean(mag[mask] ** 2))

    def _band_vector(signal_mono: np.ndarray, lo: float, hi: float, bands: int) -> np.ndarray:
        if len(signal_mono) < 64:
            return np.zeros(bands, dtype=np.float32)  # type: ignore[no-any-return]
        band_edges = np.exp(np.linspace(np.log(lo), np.log(hi), bands + 1)).astype(np.float32)
        mag, freqs_arr = _mean_fft(signal_mono)
        vec = np.zeros(bands, dtype=np.float32)
        for band_idx in range(bands):
            mask = (freqs_arr >= band_edges[band_idx]) & (freqs_arr < band_edges[band_idx + 1])
            if np.any(mask):
                vec[band_idx] = float(np.mean(mag[mask] ** 2))
        return vec  # type: ignore[no-any-return]

    try:
        fft_mag, freqs = _mean_fft(mono)
        total_energy = _band_energy(fft_mag, freqs, 80.0, 8000.0) + 1e-12
        voice_energy = _band_energy(fft_mag, freqs, 250.0, 4000.0)
        pitch_energy = _band_energy(fft_mag, freqs, 120.0, 1200.0)
        pitch_mask = (freqs >= 120.0) & (freqs < 1200.0)
        pitch_bins = fft_mag[pitch_mask]
        if pitch_bins.size > 8:
            crest = float(np.percentile(pitch_bins, 95) / (np.percentile(pitch_bins, 50) + 1e-12))
            harmonic_score = float(np.clip((crest - 1.5) / 8.5, 0.0, 1.0))
        else:
            harmonic_score = 0.0
        voice_ratio_score = float(np.clip((voice_energy / total_energy - 0.18) / 0.42, 0.0, 1.0))
        pitch_ratio_score = float(np.clip((pitch_energy / (voice_energy + 1e-12) - 0.20) / 0.45, 0.0, 1.0))
        periodicity_score = 0.0
        periodicity_window = mono[: min(len(mono), 4096)].astype(np.float64)
        periodicity_window = periodicity_window - np.mean(periodicity_window)
        if len(periodicity_window) > int(sr / 320):
            corr_size = 1 << int(np.ceil(np.log2(max(2, len(periodicity_window) * 2 - 1))))
            corr = np.fft.irfft(
                np.abs(np.fft.rfft(periodicity_window, n=corr_size)) ** 2,
                n=corr_size,
            )[: len(periodicity_window)]
            corr /= corr[0] + 1e-12
            lag_min = max(1, int(sr / 320))
            lag_max = min(len(corr) - 1, int(sr / 80))
            if lag_max > lag_min:
                periodicity_score = float(np.clip((float(np.max(corr[lag_min : lag_max + 1])) - 0.10) / 0.60, 0.0, 1.0))
        vocal_presence = float(
            np.clip(
                0.20 * voice_ratio_score + 0.10 * pitch_ratio_score + 0.20 * harmonic_score + 0.50 * periodicity_score,
                0.0,
                1.0,
            )
        )
    except Exception:
        vocal_presence = 0.0

    formant_stability = 0.5
    fricative_stability = 0.5
    transient_integrity = 0.5
    if ref_mono is not None and len(mono) >= 64:
        try:
            proc_formants = np.log1p(_band_vector(mono, 300.0, 3500.0, 10))
            ref_formants = np.log1p(_band_vector(ref_mono, 300.0, 3500.0, 10))
            formant_stability = float(np.clip((_safe_pearson(ref_formants, proc_formants) + 1.0) * 0.5, 0.0, 1.0))

            proc_fric = np.log1p(_band_vector(mono, 4000.0, 9000.0, 4))
            ref_fric = np.log1p(_band_vector(ref_mono, 4000.0, 9000.0, 4))
            if float(np.sum(proc_fric) + np.sum(ref_fric)) > 1e-6:
                fricative_stability = float(np.clip((_safe_pearson(ref_fric, proc_fric) + 1.0) * 0.5, 0.0, 1.0))

            proc_env = np.abs(np.diff(mono.astype(np.float64)))
            ref_env = np.abs(np.diff(ref_mono.astype(np.float64)))
            if len(proc_env) > 32 and len(ref_env) > 32:
                proc_env = proc_env / (np.max(proc_env) + 1e-12)
                ref_env = ref_env / (np.max(ref_env) + 1e-12)
                transient_integrity = float(np.clip((_safe_pearson(ref_env, proc_env) + 1.0) * 0.5, 0.0, 1.0))
        except Exception:
            formant_stability = 0.5
            fricative_stability = 0.5
            transient_integrity = 0.5

    return {
        "vocal_presence_proxy": vocal_presence,
        "vocal_formant_stability": formant_stability,
        "vocal_fricative_stability": fricative_stability,
        "vocal_transient_integrity": transient_integrity,
    }


def _measure_quick(
    audio: np.ndarray,
    sr: int,
    reference: np.ndarray | None = None,
    *,
    precise_override: bool = True,
    enable_vocal_guard: bool = True,
) -> dict[str, float]:
    """
    Misst alle 15 Musical Goals auf einer 5-s-Stichprobe in ≤ 200 ms.

    §9.7.5 (v10.0.0): Referenz-aware Preservation-Korrekturen.
    Wenn ``reference`` übergeben wird, erhalten anfällige Goals einen
    Preservation-Bonus basierend auf spektraler Korrelation.  Dies beseitigt
    False-Positive-Regressionen bei Noise-Removal, EQ, Dynamics-Phasen
    und ermöglicht breitere Goal-Prüfung mit weniger Exclusions.

    Prinzip: Wenn die Korrelation zwischen Original und Verarbeitetem hoch ist
    (musikalischer Inhalt erhalten), wird der absolute Score nach oben korrigiert.
    Bei niedriger Korrelation (echte Degradation) bleibt der absolute Score.

    Args:
        audio: Mono oder Stereo, float32, beliebige Länge
        sr: 48000 Hz
        reference: Original-Audio vor Phasen-Verarbeitung (gleiche Länge).
            None = rein absolute Messung (für scores_before).

    Returns:
        Dict mit 14 Scores ∈ [0, 1]
    """
    # §2.54 Shape-robuste Stereo-zu-Mono-Konvertierung: UV3 übergibt (2, N) channels-first,
    # aber audio[:, 0] bei (2, N) gibt 2 Samples zurück, nicht Kanal 0.
    # Fix: orientation-adaptiver mean() für beide (N, 2) und (2, N) Layouts.
    if audio.ndim == 2:
        mono = audio.mean(axis=0) if audio.shape[0] <= 2 else audio.mean(axis=1)
    else:
        mono = audio
    mono = np.nan_to_num(mono, nan=0.0).astype(np.float32)

    scores: dict[str, float] = {}

    # ── Pre-compute spectrum once — brillanz, waerme, bass_kraft, natuerlichkeit,
    #    authentizitaet, transparenz, separation_fidelity all share these arrays.
    #    If FFT fails every dependent metric gracefully falls back to 0.5 via its
    #    own try/except; the shared variables are always defined.
    #
    # §9.7.18 Fix (v10.0.0): frame-averaged STFT replaces full-signal FFT.
    # Root-cause: np.fft.rfft(mono) on a 5-min song creates 7.2M bins at 0.011 Hz
    # resolution. In each spectral band (e.g., 250–500 Hz), harmonic peaks occupy
    # 1–2 bins out of 1250+ → p95/p50 crest dominated by noise-floor distribution,
    # not harmonic peaks → brillanz and transparenz always near 0 for real music.
    # Fix: 4096-sample Hanning-windowed FFT, averaged over ≤200 frames.
    # Resolution: 11.7 Hz/bin → 21 bins per octave band → p95 correctly captures
    # spectral peaks relative to the inter-harmonic noise floor.
    _N_FFT_QK: int = 4096
    _win_qk: np.ndarray = np.hanning(_N_FFT_QK).astype(np.float32)
    _hop_qk: int = _N_FFT_QK // 2
    try:
        if len(mono) >= _N_FFT_QK:
            _n_frames_qk = min(200, max(1, (len(mono) - _N_FFT_QK) // _hop_qk))
            _fft_frames_qk = np.stack(
                [
                    np.abs(np.fft.rfft(mono[_i * _hop_qk : _i * _hop_qk + _N_FFT_QK] * _win_qk))
                    for _i in range(_n_frames_qk)
                ]
            )
            fft_mag: np.ndarray = _fft_frames_qk.mean(axis=0).astype(np.float32)
        else:
            fft_mag = np.abs(np.fft.rfft(mono, n=_N_FFT_QK)).astype(np.float32)
        freqs: np.ndarray = np.fft.rfftfreq(_N_FFT_QK, d=1.0 / sr).astype(np.float32)
        tot_energy: float = float(np.mean(fft_mag**2)) + 1e-12
    except Exception:
        fft_mag = np.zeros(_N_FFT_QK // 2 + 1, dtype=np.float32)
        freqs = np.zeros(_N_FFT_QK // 2 + 1, dtype=np.float32)
        tot_energy = 1e-12

    # §9.7.5 Pre-compute reference spectrum for preservation corrections.
    # Computed once; used by all reference-aware goal branches below.
    _ref_fft: np.ndarray | None = None
    _ref_mono: np.ndarray | None = None
    _vocal_guard: dict[str, float] = {
        "vocal_presence_proxy": 0.0,
        "vocal_formant_stability": 0.5,
        "vocal_fricative_stability": 0.5,
        "vocal_transient_integrity": 0.5,
    }
    if reference is not None:
        try:
            # §2.54 Shape-robuste Referenz-Downmix — gleiche Logik wie mono-Input
            if reference.ndim == 2:
                _rm = reference.mean(axis=0) if reference.shape[0] <= 2 else reference.mean(axis=1)
            else:
                _rm = reference
            _rm = np.nan_to_num(_rm, nan=0.0).astype(np.float32)
            _ml = min(len(mono), len(_rm))
            _ref_mono = np.asarray(_rm[:_ml])
            # §9.7.18: reference FFT uses same 4096-sample STFT basis as fft_mag
            if len(_ref_mono) >= _N_FFT_QK:
                _n_ref_qk = min(200, max(1, (len(_ref_mono) - _N_FFT_QK) // _hop_qk))
                _ref_fft = (
                    np.stack(
                        [
                            np.abs(np.fft.rfft(_ref_mono[_j * _hop_qk : _j * _hop_qk + _N_FFT_QK] * _win_qk))
                            for _j in range(_n_ref_qk)
                        ]
                    )
                    .mean(axis=0)
                    .astype(np.float32)
                )
            else:
                _ref_fft = np.abs(np.fft.rfft(_ref_mono, n=_N_FFT_QK)).astype(np.float32)
        except Exception:
            _ref_fft = None
            _ref_mono = None

    if reference is not None:
        _vocal_guard = _measure_vocal_guard_features(
            mono,
            sr,
            reference=_ref_mono if _ref_mono is not None else reference,
        )

    # ── Brillanz (§9.7.12 HF Spectral Crest Factor, 2–16 kHz) ────────
    # Root-cause of prior false regressions: the old HF-energy-ratio proxy was
    # SNR-dependent.  Broadband noise raises the HF energy floor uniformly →
    # high ratio before denoising; after denoising only musical peaks remain →
    # lower absolute energy → false drop of 0.2–0.5.
    #
    # Fix §9.7.12: Spectral crest factor = p95 / p50 within 2–16 kHz band.
    #   • Noise floor lifts the MEDIAN (p50) while leaving p95 ≈ music peaks
    #     → noisy audio: crest LOW (2–3).
    #   • After denoising, noise floor drops → p50 falls toward musical valleys
    #     → crest INCREASES (5–30) → score improves → no false regression.
    # Scientific basis: Fastl & Zwicker, "Psychoacoustics: Facts and Models",
    # 2007 §8.3 Sharpness — crest factor as perceptual brightness indicator.
    # Calibration: crest ≥ 15 → score 1.0; crest 1.5 → score 0.0.
    try:
        _rms_bril = float(np.sqrt(np.mean(mono**2) + 1e-12))
        if _rms_bril < 1e-5:
            scores["brillanz"] = 0.5  # near-silence: neutral, no HF content to score
        else:
            _hf_mask_b = (freqs >= 2000) & (freqs <= 16000)
            _mid_mask_b = (freqs >= 500) & (freqs < 2000)
            _hf_bins_b = fft_mag[_hf_mask_b]
            _mid_bins_b = fft_mag[_mid_mask_b]
            if len(_hf_bins_b) > 20 and len(_mid_bins_b) > 5:
                # §9.7.24 Fix (v10.0.0): HF-crest mit STFT-Averaging zu niedrig (Peaks geglättet).
                # Root-cause: frame-averaged STFT mittelt Transienten-Peaks → p95/p50 sinkt.
                # Fix: HF-Energie-Ratio E(2-16kHz) / E(500-16kHz) als Brillanz-Proxy.
                # Musik mit Becken/Streicher/Brillanz: ratio ~ 0.15–0.40 → score 0.5–1.0
                # Tiefes Mono-Signal (Bass only): ratio ~ 0.005 → score 0.0
                # Kalibrierung: ratio 0.05 → 0.25; 0.20 → 0.75; 0.35+ → 1.0
                _hf_energy_b = float(np.mean(_hf_bins_b**2))
                _mid_hf_energy_b = float(np.mean(np.concatenate([_mid_bins_b, _hf_bins_b]) ** 2)) + 1e-12
                _hf_ratio_b = float(_hf_energy_b / _mid_hf_energy_b)
                # score: 0 at ratio=0, 0.5 at 0.15, 1.0 at 0.30+
                scores["brillanz"] = float(np.clip((_hf_ratio_b - 0.02) / 0.28, 0.0, 1.0))
            else:
                scores["brillanz"] = 0.5
    except Exception:
        scores["brillanz"] = 0.5

    # ── Wärme (§9.7.14 Warmth Ratio: E_200-800 / E_800-3000 Hz) ──────
    # Root-cause of prior false regressions: the old mid/total-energy ratio was
    # reverb-sensitive.  Reverb tail adds diffuse energy across the mid band →
    # high ratio before dereverb; after removal dry signal has less mid energy →
    # false drop in waerme.
    #
    # Fix §9.7.14: Warmth ratio = E(200–800 Hz) / E(800–3000 Hz).
    #   • Reverb affects BOTH sub-bands proportionally (air absorption is gradual
    #     at these frequencies, early reflections span 200–3000 Hz uniformly) →
    #     the ratio stays stable during dereverb → reverb-invariant.
    #   • Only genuine spectral-balance changes (EQ, vinyl roll-off) shift the ratio
    #     in a perceptually meaningful way.
    # Scientific basis: Moore & Glasberg (1983) auditory filter bandwidths;
    # Fletcher & Rossing vocal formant structure (warmth ≈ F1/F2 energy balance).
    # Calibration: ratio 4.0 → score 1.0 (very warm); ratio 1.0 → score 0.25 (neutral);
    # ratio 0 → score 0.0 (thin).
    # §2026-04-24: Normierungskonstante 1.5 → 4.0 korrigiert:
    #   Typisches ungewichtetes E(200-800 Hz)/E(800-3000 Hz)-Verhältnis für warme Musik liegt
    #   bei 3–5 (Bass/untere Mitten dominieren). Die alte Konstante 1.5 führte dazu, dass der
    #   Proxy immer saturiert auf 1.0 war (ratio >> 1.5 → clip). WaermeMetric._measure_absolute()
    #   nutzt ISO 226:2003 Equal-Loudness-Gewichtung, die 800–3000 Hz stärker gewichtet →
    #   reale Werte 0.70–0.90. Mit 4.0 ist der Proxy sensitive für Änderungen und erkennt
    #   Warmth-Regressions durch Denoise/Dereverb-Phasen (Δ-Erkennung statt Sattigung).
    try:
        # §9.7.14 ISO-226 perceptual weighting — MUST match WaermeMetric._measure_absolute()
        # (§2.54 Calibration invariant: PMGG proxy and end-pipeline metric must use the
        # same measurement basis to avoid before=1.000/after=1.000/delta=0.000 saturation).
        # Unweighted E(200-800)/E(800-3000) at vinyl ~12–15 → clip 1.0 always (VERBOTEN).
        # With ISO-226 weighting: 800–3000 Hz band is strongly upweighted (ear more
        # sensitive there) → ratio drops to 0.6–1.2 for typical Schlager → proxy 0.15–0.30
        # → sensitive to changes, no saturation.  Lazy import to avoid circular deps.
        try:
            from backend.core.musical_goals.musical_goals_metrics import _iso226_weights as _w226

            _w = _w226(freqs)
        except Exception:
            _w = np.ones_like(freqs, dtype=np.float32)
        _wm = fft_mag * _w  # perceptually weighted magnitude
        _e_low_mid = float(np.mean(_wm[(freqs >= 200) & (freqs < 800)] ** 2)) + 1e-9
        _e_upper_mid = float(np.mean(_wm[(freqs >= 800) & (freqs < 3000)] ** 2)) + 1e-9
        # Soft-sigmoid normalization (§9.7.14 fix v10.0.0):
        # With ISO-226 weighting, warm Schlager has ratio ~6 → hard /4.0 clips to 1.0 always
        # → delta never detectable → PMGG blind for waerme regressions.
        # Soft norm: ratio/(ratio+0.70) → neutral=0.59, warm(~2.5)=0.78, Schlager(~6)=0.90,
        # delta for small change detectable (~0.005–0.01). Satisfies spec: warm→0.75–1.0.
        _ratio_w = _e_low_mid / _e_upper_mid
        scores["waerme"] = float(np.clip(_ratio_w / (_ratio_w + 0.70), 0.0, 1.0))
    except Exception:
        scores["waerme"] = 0.5

    # ── Groove (Periodizität + Microtiming-Synkopation) ────────────────
    # Two-component model:
    #   A) Rhythmic periodicity via envelope autocorrelation (existing §9.7.9)
    #   B) Microtiming syncopation complexity.
    # Scientific basis:
    #   - Witek et al. (2017), PLOS ONE 12:e0169907: groove peaks at an
    #     intermediate level of syncopation, not at perfect regularity.
    #   - Frühauf et al. (2013), Psychol. Music 41:484: moderate IOI variance
    #     increases groove; too low = mechanical, too high = unstable.
    try:
        env = np.abs(mono)
        # Hüllkurven-Autokorrelation
        hop = sr // 100  # 10 ms
        # Vectorized: non-overlapping frames via reshape (replaces Python list comprehension)
        _nf_g = (len(env) - 1) // hop
        rms_env = (
            np.mean(env[: _nf_g * hop].reshape(_nf_g, hop) ** 2, axis=1) if _nf_g > 0 else np.empty(0, dtype=np.float32)
        )
        if len(rms_env) > 10:
            # §9.7.9 LF-Robustheit: 5-Frame-Glättung der Einhüllkurve (50 ms).
            # Hintergrund: Hum (50/100 Hz) erzeugt 100/200 Hz-Modulation in |mono|.
            # Bei 10 ms hop je ~0.5–1 Perioden/Frame → frame-to-frame-Varianz.
            # Diese Varianz erhöht autocorr[0] (Gesamtenergie-Normierungsbasis)
            # ohne die 500 ms-Rhythmusperiodizität zu verändern → normiertes
            # autocorr[lag_05] wird durch LF-Spektraländerungen beeinflusst.
            # Fix: 5 × 10 ms = 50 ms Tiefpass entfernt ≥ 20 Hz Hüllkurvenkomponenten
            # (Hum-Modulation) → Normierungsbasis repräsentiert nur Rhythmusenergie.
            # Musikalischer Groove: 0.5–8 Hz (120–1920 BPM) → unverändert.
            _sw = min(5, len(rms_env) // 4)
            if _sw >= 2:
                rms_env = np.convolve(rms_env, np.ones(_sw) / float(_sw), mode="valid")
            # §9.7.9 LF-Robustheit: Mean-centering entfernt DC-Floor (Hum, Rauschen,
            # konstanter Energieboden) aus rms_env vor der FFT-Autokorrelation.
            # Ohne Mean-centering dominiert DC² den autocorr[0]-Normierungsnenner →
            # normierte Periodizität bei lag=500ms wird proportional reduziert.
            # Nach Mean-centering: nur rhythmische Variation verbleibt → DC-invariant.
            rms_env = rms_env - np.mean(rms_env)
            from backend.core.core_utils import fft_autocorr

            autocorr = fft_autocorr(rms_env)
            autocorr /= autocorr[0] + 1e-12
            # Regularität: lokaler Peak bei ~0.5 s (typisch Groove).
            # Pure autocorr[lag] can over-score aperiodic burst patterns when overall
            # envelope variance is high; use local peak contrast against neighbours.
            lag_05 = min(50, len(autocorr) - 1)  # 50 × 10 ms = 500 ms
            _p_lag = float(autocorr[lag_05])
            _ln = max(1, lag_05 - 5)
            _rn = min(len(autocorr), lag_05 + 6)
            _nb_left = autocorr[_ln:lag_05]
            _nb_right = autocorr[lag_05 + 1 : _rn]
            _nb = np.concatenate([_nb_left, _nb_right]) if (_nb_left.size + _nb_right.size) > 0 else np.empty(0)
            _nb_med = float(np.median(_nb)) if _nb.size > 0 else 0.0
            _peak_contrast = _p_lag - _nb_med
            periodicity_score = float(np.clip(0.5 + 1.2 * _peak_contrast, 0.0, 1.0))

            # Microtiming/syncopation component via inter-onset-interval variability.
            # The perceptual sweet spot is moderate variability (CV ≈ 0.20).
            _onset_thresh = float(np.mean(rms_env) + 0.5 * float(np.std(rms_env)) + 1e-9)
            _onsets = (
                np.where(
                    (rms_env[1:-1] > rms_env[:-2]) & (rms_env[1:-1] > rms_env[2:]) & (rms_env[1:-1] > _onset_thresh)
                )[0]
                + 1
            )
            if len(_onsets) >= 4:
                _ioi = np.diff(_onsets.astype(np.float32))
                _ioi_mean = float(np.mean(_ioi))
                _ioi_cv = float(np.std(_ioi)) / (_ioi_mean + 1e-9)
                _syncopation_score = float(np.clip(1.0 - 4.0 * (_ioi_cv - 0.20) ** 2, 0.5, 1.0))
            else:
                _syncopation_score = 0.5

            # Emphasize rhythmic periodicity over syncopation to keep periodic/aperiodic
            # discrimination stable on sparse click-burst test material.
            scores["groove"] = float(np.clip(0.75 * periodicity_score + 0.25 * _syncopation_score, 0.0, 1.0))
        else:
            scores["groove"] = 0.5
    except Exception:
        scores["groove"] = 0.5

    # ── Tonales Zentrum (Krumhansl-Schmuckler Key Detection, §9.7.11) ──
    # Scientific basis: Krumhansl & Schmuckler 1990, Temperley 2001,
    # Müller "Fundamentals of Music Processing" 2015 §5.3.
    #
    # WHY the previous entropy-based proxy was wrong for PMGG delta-checks:
    #   entropy = -Σ(chroma * log chroma)  measures chroma CONCENTRATION
    #   → SNR-dependent: noise spreads energy uniformly across all 12 bins
    #     → low entropy (flat chroma) BEFORE denoise; tonal signal revealed
    #     AFTER → entropy changes even though musical key is preserved.
    #   → Result: false P2 regression on EVERY noise-reducing phase at ANY
    #     strength (Δ≈0 stagnation confirmed in production logs 2026-03-30).
    #
    # K-S key detection is SNR-invariant: uniform noise raises ALL 24 major/
    # minor correlation scores equally → argmax is unchanged. Only a genuine
    # key-shift (pitch transposition) changes the dominant key label.
    #
    # Algorithm (vectorized):
    #   1. Build chroma vector from log-domain FFT magnitude (Hann window)
    #   2. Correlate against 24 Krumhansl-Schmuckler major/minor profiles
    #      (normalized to unit-variance for Pearson equivalence)
    #   3. key_before = argmax of 24 scores in _ref, key_after = argmax in proc
    #   4. Circular semitone distance d = min(|k_a − k_b| mod 12, 12 − ...) ∈ [0,6]
    #   5. tonal_center = 1 − d/6   (0 = tritone/max shift, 1 = same key)
    #   6. Fallback (no reference available): best correlation score, normalized.
    #
    # Krumhansl-Schmuckler major/minor profiles (canonical, from Krumhansl 1990
    # Table 1 + Temperley 2001 re-normalisation).
    _KS_MAJOR: np.ndarray = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        dtype=np.float32,
    )
    _KS_MINOR: np.ndarray = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
        dtype=np.float32,
    )
    # Pre-normalise profiles once (zero-mean, unit-variance)
    _ks_maj_n: np.ndarray = _KS_MAJOR - _KS_MAJOR.mean()
    _ks_maj_n /= _ks_maj_n.std() + 1e-12
    _ks_min_n: np.ndarray = _KS_MINOR - _KS_MINOR.mean()
    _ks_min_n /= _ks_min_n.std() + 1e-12

    def _ks_key(signal_mono: np.ndarray, n_fft: int = 4096, sr_inner: int = 48000) -> int:
        """Gibt dominant key label 0–23 (0–11 major, 12–23 minor, root = C) zurück.

        Uses multi-segment averaging (8 windows) for stability across
        the entire signal, rather than a single center window which is
        vulnerable to phase-specific spectral changes (e.g. TP limiting).

        Returns -1 on failure (too short / silence).
        """
        n_seg = 8
        seg_len = n_fft

        if len(signal_mono) < seg_len:
            # Short signal: single segment (original behaviour)
            segments = [signal_mono]
        else:
            # Distribute n_seg segments evenly across the signal
            step = max(1, (len(signal_mono) - seg_len) // max(1, n_seg - 1))
            segments = []
            for i in range(n_seg):
                start = min(i * step, len(signal_mono) - seg_len)
                segments.append(signal_mono[start : start + seg_len])

        # Accumulate chroma across all segments
        chroma_acc = np.zeros(12, dtype=np.float64)
        for seg in segments:
            win = np.hanning(len(seg))
            spec = np.abs(np.fft.rfft(seg * win, n=n_fft))
            freqs_k = np.fft.rfftfreq(n_fft, d=1.0 / sr_inner)
            _kb = np.where((freqs_k > 27.5) & (freqs_k < 4186.0))[0]
            if len(_kb) == 0:
                continue
            _kn = np.round(12.0 * np.log2(freqs_k[_kb] / 440.0 + 1e-12)).astype(np.int32) % 12  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
            _chroma_seg = np.zeros(12, dtype=np.float64)
            np.add.at(_chroma_seg, _kn, spec[_kb].astype(np.float64))
            seg_sum = _chroma_seg.sum()
            if seg_sum > 1e-8:
                chroma_acc += _chroma_seg / seg_sum  # normalize each segment contribution

        s = chroma_acc.sum()
        if s < 1e-8:
            return -1
        chroma_k = (chroma_acc / s).astype(np.float32)
        # Zero-mean + unit-variance normalisation of the chroma vector
        chroma_k -= chroma_k.mean()
        std_c = chroma_k.std()
        if std_c < 1e-12:
            return -1
        chroma_k /= std_c
        # Correlate against all 12 rotations of major and minor profiles
        best_score = -np.inf
        best_key = 0
        for root in range(12):
            maj_rot = np.roll(_ks_maj_n, root)
            min_rot = np.roll(_ks_min_n, root)
            r_maj = float(np.dot(chroma_k, maj_rot))
            r_min = float(np.dot(chroma_k, min_rot))
            if r_maj > best_score:
                best_score, best_key = r_maj, root  # major: 0–11
            if r_min > best_score:
                best_score, best_key = r_min, root + 12  # minor: 12–23
        return best_key

    try:
        _n_fft_ks = 4096
        _key_proc = _ks_key(mono, n_fft=_n_fft_ks, sr_inner=sr)
        if _key_proc == -1:
            scores["tonal_center"] = 0.5
        elif _ref_mono is not None:
            # Delta-mode: compare dominant key before vs. after processing.
            # Circular semitone distance on root (0–11), mode ignored for primary check
            # (mode-shift rare in restoration; penalised lightly via +6 offset if needed).
            _key_ref = _ks_key(_ref_mono, n_fft=_n_fft_ks, sr_inner=sr)
            if _key_ref == -1:
                scores["tonal_center"] = 0.5
            else:
                _root_proc = _key_proc % 12
                _root_ref = _key_ref % 12
                _d = abs(_root_proc - _root_ref)
                _d = min(_d, 12 - _d)  # circular distance ∈ [0, 6]
                # Mode mismatch (major ↔ minor): add 1 semitone equivalent penalty
                _mode_penalty = 0 if (_key_proc // 12 == _key_ref // 12) else 1
                _d = min(6, _d + _mode_penalty)
                scores["tonal_center"] = float(np.clip(1.0 - _d / 6.0, 0.0, 1.0))
        else:
            # No reference available: use normalised best K-S correlation score
            # as absolute quality indicator (0 = atonal noise, 1 = strongly tonal).
            # Re-compute the best score for absolute interpretation.
            _spec_abs = np.abs(np.fft.rfft(mono * np.hanning(len(mono)), n=_n_fft_ks))
            _freqs_abs = np.fft.rfftfreq(_n_fft_ks, d=1.0 / sr)
            _chroma_abs = np.zeros(12, dtype=np.float32)
            _kb2 = np.where((_freqs_abs > 27.5) & (_freqs_abs < 4186.0))[0]
            if len(_kb2) > 0:
                _kn2 = np.round(12.0 * np.log2(_freqs_abs[_kb2] / 440.0 + 1e-12)).astype(np.int32) % 12  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
                np.add.at(_chroma_abs, _kn2, _spec_abs[_kb2])
            _s2 = _chroma_abs.sum()
            if _s2 > 1e-8:
                _chroma_abs /= _s2
                _chroma_abs -= _chroma_abs.mean()
                _std2 = _chroma_abs.std()
                if _std2 > 1e-12:
                    _chroma_abs /= _std2
                    _best_r = float(
                        max(
                            max(float(np.dot(_chroma_abs, np.roll(_ks_maj_n, r))) for r in range(12)),
                            max(float(np.dot(_chroma_abs, np.roll(_ks_min_n, r))) for r in range(12)),
                        )
                    )
                    # K-S scores range roughly ‑1…+1; clamp to [0, 1]
                    scores["tonal_center"] = float(np.clip((_best_r + 1.0) / 2.0, 0.0, 1.0))
                else:
                    scores["tonal_center"] = 0.5
            else:
                scores["tonal_center"] = 0.5
    except Exception:
        scores["tonal_center"] = 0.5

    # ── Natürlichkeit (Log-Mel Tonal Dominance + Spectral Smoothness) ────
    # Canonical key "natuerlichkeit" — aligned with GoalApplicabilityFilter §2.32.
    # Fix §9.7.15 (v10.0.0): Krimphoff FFT-bin irregularity scored all real music
    # near-zero because harmonic peaks at FFT resolution are inherently "irregular".
    # New proxy uses 24 mel bands with two orthogonal components:
    #   1) Tonal Dominance: top-4 mel bands carry > 15–60% of energy (music vs noise)
    #      music: top4/total ≈ 0.45–0.65 | noise: ≈ 0.17 | tonal_score: 0→1
    #   2) Mel-Log Smoothness: NR artifacts (musical noise, spectral holes) create
    #      jagged log-mel transitions; clean music has smooth mel envelope.
    # Combined (0.45 smooth + 0.55 tonal) → clean music 0.85+, white noise 0.76,
    # heavy NR artifacts 0.70–0.78 — physically meaningful for all era/genre/material.
    try:
        _rms_nat = float(np.sqrt(np.mean(mono**2) + 1e-12))
        if _rms_nat < 1e-5:
            scores["natuerlichkeit"] = 0.5
        else:
            # Build 24 mel-spaced energy bands (100 Hz – 16 kHz)
            _n_mel_nat = 24
            _mel_lo_nat, _mel_hi_nat = 100.0, 16000.0
            _mel_cents_nat = np.linspace(
                2595.0 * np.log10(1.0 + _mel_lo_nat / 700.0),
                2595.0 * np.log10(1.0 + _mel_hi_nat / 700.0),
                _n_mel_nat,
            )
            _mel_fqs_nat = 700.0 * (10.0 ** (_mel_cents_nat / 2595.0) - 1.0)
            _mel_bands_nat = np.zeros(_n_mel_nat, dtype=np.float64)
            for _i_nat in range(_n_mel_nat):
                _fl_n = float(_mel_fqs_nat[_i_nat - 1]) if _i_nat > 0 else _mel_lo_nat
                _fh_n = float(_mel_fqs_nat[_i_nat + 1]) if _i_nat < _n_mel_nat - 1 else _mel_hi_nat
                _m_nat = (freqs >= _fl_n) & (freqs < _fh_n)
                if _m_nat.any():
                    _mel_bands_nat[_i_nat] = float(np.sum(fft_mag[_m_nat] ** 2))
            _mel_total_nat = float(np.sum(_mel_bands_nat)) + 1e-12
            # Component 1: Tonal dominance (top-4 bands energy fraction)
            _mel_sorted_nat = np.sort(_mel_bands_nat)[::-1]
            _top4_ratio = float(np.sum(_mel_sorted_nat[:4])) / _mel_total_nat
            _tonal_score = float(np.clip((_top4_ratio - 0.15) / 0.45, 0.0, 1.0))
            # Component 2: Mel-log smoothness (NR artifact detection)
            # §9.7.22 Fix (v10.0.0): Normierung NUR auf aktive Bänder (> 1% des Max-Bands).
            # Root-cause: Inaktive Bänder (Energie ~ 1e-12) haben log ~ -27.6,
            # während aktive Bänder log ~ 10+ haben. Die riesige Spannweite in _irr_den
            # dominiert die Normierung → fast alle echten Irregularitäten erscheinen klein.
            # Fix: Verwende nur Bänder > 1% des Maximum-Bands für Irregularitäts-Messung.
            _log_mel_nat = np.log(_mel_bands_nat + 1e-12)
            if len(_log_mel_nat) > 4:
                _active_threshold_nat = float(np.max(_mel_bands_nat)) * 0.01 + 1e-12
                _active_mask_nat = _mel_bands_nat > _active_threshold_nat
                if np.sum(_active_mask_nat) >= 4:
                    _log_active = _log_mel_nat[_active_mask_nat]
                    # Smoothness über aktive Bänder (NR-Artefakte = jagged Envelope)
                    len(_log_active)
                    _sm_act = (_log_active[:-2] + _log_active[1:-1] + _log_active[2:]) / 3.0
                    _irr_num_nat = float(np.sum(np.abs(_log_active[1:-1] - _sm_act)))
                    _irr_den_nat = float(np.sum(np.abs(_log_active))) + 1e-12
                    _irr_mel = float(np.clip(_irr_num_nat / _irr_den_nat, 0.0, 1.0))
                    _smooth_score = float(np.clip(1.0 - _irr_mel / 0.35, 0.0, 1.0))
                else:
                    _smooth_score = 0.5
            else:
                _smooth_score = 0.5
            scores["natuerlichkeit"] = 0.45 * _smooth_score + 0.55 * _tonal_score
            # §9.7.5 Preservation: log-spectral correlation with reference (belt+suspenders)
            if _ref_fft is not None:
                _fl = min(len(fft_mag), len(_ref_fft))
                if _fl > 20:
                    _log_proc = np.log(fft_mag[:_fl] + 1e-12)
                    _log_ref = np.log(_ref_fft[:_fl] + 1e-12)
                    _r = _safe_pearson(_log_ref, _log_proc)
                    if _r > 0.7:
                        scores["natuerlichkeit"] = min(1.0, scores["natuerlichkeit"] + (_r - 0.7) * 0.3)
    except Exception:
        scores["natuerlichkeit"] = 0.5

    # ── Timbre-Authentizität (MFCC-basiert: Pearson auf log-Mel) ──────
    try:
        # Proxy: Spectral Centroid-Stabilität über kurze Fenster
        hop_t = sr // 50  # 20 ms
        centroids = []
        _global_rms = float(np.sqrt(np.mean(mono**2) + 1e-12))
        _rms_gate = max(1e-5, 0.05 * _global_rms)
        for i in range(0, len(mono) - hop_t, hop_t):
            w = mono[i : i + hop_t]
            _w_rms = float(np.sqrt(np.mean(w**2) + 1e-12))
            if _w_rms < _rms_gate:
                continue
            w_fft = np.abs(np.fft.rfft(w))
            w_freqs = np.fft.rfftfreq(len(w), d=1.0 / sr)
            centroid = float(np.sum(w_freqs * w_fft) / (np.sum(w_fft) + 1e-12))
            centroids.append(centroid)
        # §9.7.23 Fix (v10.0.0): Centroid-CV-Ansatz ist invertiert für Rauschen.
        # Root-cause: Weißes Rauschen hat konstanten Spectral Centroid (~sr/4)
        # → niedrige CV → hoher Score (FALSCH). Echte Musik hat variierende
        # Centroide (Instrumental wechsel, Dynamik) → höhere CV → niedrigerer Score.
        # Fix: Mel-Band-Temporal-Stabilität = normierte RMS-Std über 24 Mel-Bänder.
        # Musik: jedes Band hat stabile relative Energie → hohe Korrelation zwischen Frames.
        # Rauschen: zufällige Band-Energie pro Frame → niedrige inter-frame Korrelation.
        if len(centroids) > 4:
            # Berechne Mel-Band-Energie pro Frame für temporale Stabilität
            _hop_tb = max(1, sr // 25)  # 40 ms frames
            _n_tb_frames = min(60, (len(mono) - 1024) // _hop_tb)
            if _n_tb_frames >= 4:
                _mel_edges_tb = np.exp(np.linspace(np.log(200.0), np.log(8000.0), 13)).astype(np.float32)
                _tb_matrix = np.zeros((_n_tb_frames, 12), dtype=np.float64)
                for _fi in range(_n_tb_frames):
                    _seg = mono[_fi * _hop_tb : _fi * _hop_tb + 1024]
                    if len(_seg) < 512:
                        continue
                    _sfft = np.abs(np.fft.rfft(_seg, n=1024))
                    _sfq = np.fft.rfftfreq(1024, d=1.0 / sr)
                    for _bi in range(12):
                        _bm = (_sfq >= _mel_edges_tb[_bi]) & (_sfq < _mel_edges_tb[_bi + 1])
                        _tb_matrix[_fi, _bi] = float(np.mean(_sfft[_bm] ** 2)) if _bm.any() else 0.0
                # Normiere jede Zeile (Frame) auf relative Verteilung
                _row_sums = _tb_matrix.sum(axis=1, keepdims=True) + 1e-12
                _tb_norm = _tb_matrix / _row_sums
                # Inter-Frame-Korrelation: stabile Timbre → hohe Korrelation
                _cors = []
                for _fi in range(1, _n_tb_frames):
                    _a, _b = _tb_norm[_fi - 1], _tb_norm[_fi]
                    _da = _a - _a.mean()
                    _db = _b - _b.mean()
                    _denom = (np.linalg.norm(_da) * np.linalg.norm(_db)) + 1e-12
                    _cors.append(float(np.dot(_da, _db) / _denom))
                _mean_cor = float(np.mean(_cors)) if _cors else 0.0
                # Musik: mean_cor ~ 0.85–0.98; Rauschen: ~ 0.0–0.15
                scores["timbre_authentizitaet"] = float(np.clip((_mean_cor + 0.1) / 1.0, 0.0, 1.0))
            else:
                scores["timbre_authentizitaet"] = 0.5
        else:
            scores["timbre_authentizitaet"] = 0.5
        # §9.7.5 Preservation: Centroid trajectory correlation with reference
        if _ref_mono is not None and len(centroids) > 2:
            _rm_ml = min(len(mono), len(_ref_mono))
            _ref_centroids = []
            _ref_global_rms = float(np.sqrt(np.mean(_ref_mono[:_rm_ml] ** 2) + 1e-12))
            _ref_rms_gate = max(1e-5, 0.05 * _ref_global_rms)
            for i in range(0, _rm_ml - hop_t, hop_t):
                _rw = _ref_mono[i : i + hop_t]
                _rw_rms = float(np.sqrt(np.mean(_rw**2) + 1e-12))
                if _rw_rms < _ref_rms_gate:
                    continue
                _rw_fft = np.abs(np.fft.rfft(_rw))
                _rw_freqs = np.fft.rfftfreq(len(_rw), d=1.0 / sr)
                _ref_centroids.append(float(np.sum(_rw_freqs * _rw_fft) / (np.sum(_rw_fft) + 1e-12)))
            if len(_ref_centroids) > 2:
                _r = _safe_pearson(np.array(_ref_centroids), np.array(centroids[: len(_ref_centroids)]))
                if _r > 0.7:
                    scores["timbre_authentizitaet"] = min(1.0, scores["timbre_authentizitaet"] + (_r - 0.7) * 0.5)
    except Exception:
        scores["timbre_authentizitaet"] = 0.5

    # ── Bass-Kraft (Bassenergie 20–250 Hz) ─────────────────────────────
    try:
        _rms_bk = float(np.sqrt(np.mean(mono**2) + 1e-12))
        if _rms_bk < 1e-5:
            scores["bass_kraft"] = 0.5  # near-silence: neutral (0/1e-12 = 0.0 is misleading)
        else:
            bass_energy = float(np.mean(fft_mag[(freqs >= 20) & (freqs <= 250)] ** 2))
            # Normierung: typische Bassenergie ~2% des Spektrums → 0.02 = Score 1.0
            scores["bass_kraft"] = float(np.clip(bass_energy / (tot_energy * 0.02 + 1e-12), 0.0, 1.0))
            # §9.7.5 Preservation: LF spectral correlation (20-500 Hz)
            if _ref_fft is not None:
                _lf = (freqs[: len(_ref_fft)] >= 20) & (freqs[: len(_ref_fft)] <= 500)
                if np.sum(_lf) > 5:
                    _r = _safe_pearson(_ref_fft[_lf], fft_mag[: len(_ref_fft)][_lf])
                    if _r > 0.7:
                        scores["bass_kraft"] = min(1.0, scores["bass_kraft"] + (_r - 0.7) * 0.5)
    except Exception:
        scores["bass_kraft"] = 0.5

    # ── Authentizität (Mel-Band-Flatness — §9.7.19 Fix v10.0.0) ──────
    # §2.29b Root-Fix (v10.0.0): Der frühere Proxy maß spektrale Rauheit — invertiert.
    # §9.7.19 Fix (v10.0.0): Full-FFT-Flatness = geom(|X|)/arith(|X|) über N//2+1 Bins.
    # Root-cause: Rausch-Bins nahe Null dominieren den Logarithmus-Mittelwert →
    # geom_mean → near-zero für Musik → flatness ≈ 0 sowohl für Musik als auch Rauschen
    # → Score = 0 für alle Eingaben (Proxy blind).
    # Fix: 24 Mel-Bänder 100–16000 Hz aggregieren je mehrere FFT-Bins → kein Band ist
    # „null". Flatness unterscheidet korrekt:
    #   Tonale Musik (wenige dominante Bänder): flatness ≈ 0.01–0.10 → score 0.71–0.97
    #   Stark verrauscht (SNR 15 dB): flatness ≈ 0.28 → score 0.20
    #   Weißes Rauschen: flatness ≈ 1.0 → score 0.0
    try:
        _rms_auth = float(np.sqrt(np.mean(mono**2) + 1e-12))
        if _rms_auth < 1e-5:
            scores["authentizitaet"] = 0.5
        else:
            _mel_auth_edges = np.exp(np.linspace(np.log(100.0), np.log(16000.0), 25)).astype(np.float32)
            _mel_auth_powers: list[float] = []
            for _k in range(24):
                _m_auth = (freqs >= _mel_auth_edges[_k]) & (freqs < _mel_auth_edges[_k + 1])
                if np.sum(_m_auth) > 0:
                    _mel_auth_powers.append(float(np.mean(fft_mag[_m_auth] ** 2)) + 1e-9)
            if len(_mel_auth_powers) >= 8:
                _mp_auth = np.array(_mel_auth_powers, dtype=np.float64)
                _geom_auth = float(np.exp(np.mean(np.log(_mp_auth))))
                _arith_auth = float(np.mean(_mp_auth))
                _flatness_auth = float(np.clip(_geom_auth / (_arith_auth + 1e-12), 0.0, 1.0))
                # Calibration (v10.0.0): flatness 0.35+ → score 0 (noisy/codec),
                # flatness 0.05 → score 0.86 (tonal Schlager), 0.01 → 0.97 (clean)
                scores["authentizitaet"] = float(np.clip(1.0 - _flatness_auth / 0.35, 0.0, 1.0))
            else:
                scores["authentizitaet"] = 0.5
            # §9.7.5 Preservation: log-spectral correlation as supplemental signal.
            if _ref_fft is not None:
                _fl = min(len(fft_mag), len(_ref_fft))
                if _fl > 20:
                    _r = _safe_pearson(
                        np.log(_ref_fft[:_fl] + 1e-12),
                        np.log(fft_mag[:_fl] + 1e-12),
                    )
                    if _r > 0.7:
                        scores["authentizitaet"] = min(1.0, scores["authentizitaet"] + (_r - 0.7) * 0.3)
    except Exception:
        scores["authentizitaet"] = 0.5

    # ── Emotionalität (Crest + RMS-Varianz + Spectral Flux) ────────────
    # Spectral flux is a classic arousal feature in MIR and improves the old
    # loudness-only proxy with a direct measure of expressive spectral change.
    # Scientific basis: Scheirer & Slaney (1997) ICASSP; Liu et al. (2003) ISMIR.
    try:
        rms_val = float(np.sqrt(np.mean(mono**2) + 1e-12))
        if rms_val < 1e-5:
            scores["emotionalitaet"] = 0.5
        else:
            # §9.7.27 Fix: Hüllkurven-Crest statt Zeitbereich-Percentile.
            # Root-cause: percentile(99.9) der absoluten Amplitude wird durch weißes
            # Rauschen stark erhöht (kurze Peaks ~3σ). Das lässt noisy > clean.
            # Fix: Hüllkurven-Crest = max(rms_frames) / mean(rms_frames) (10ms-Fenster).
            # Rauschen hat eine flache Hüllkurve (CV ~0.05) → crest_env ~1.0–1.1
            # Musik mit Dynamik: starke Onsets → crest_env 2–6
            hop_e = max(1, sr // 100)  # 10ms Frames
            rms_frames = np.array(
                [float(np.sqrt(np.mean(mono[i : i + hop_e] ** 2) + 1e-12)) for i in range(0, len(mono) - hop_e, hop_e)]
            )
            _env_peak = float(np.percentile(rms_frames, 99)) + 1e-12
            _env_mean = float(np.mean(rms_frames)) + 1e-12
            _env_crest = _env_peak / _env_mean
            # 1.0 (flat/noise) → 0.0; 3.0 (moderate dynamics) → 0.67; 5.0 (expressive) → 1.0
            crest_score = float(np.clip((_env_crest - 1.0) / 4.0, 0.0, 1.0))
            # §9.7.20 Fix (v10.0.0): dB-Domain-Bereich ersetzt lineare Varianz.
            # Root-cause: np.var(rms_frames)*1000 ≈ 0 für steady-state Musik
            # (z.B. lineare RMS-Varianz = 0.0003 → Score 0.30; bei 10ms-Frames fast
            # immer nahe 0 für Musik mit konstantem Signalpegel).
            # Perceptuelle Dynamik wird auf dB-Skala gemessen, nicht linear.
            # Fix: p90−p10 dB-Bereich der 10ms-RMS-Frames — robust und sensitiv:
            #   1 dB (over-compressed): 0.0; 6 dB: 0.29; 12 dB: 0.65; 18 dB: 1.0
            if len(rms_frames) > 4:
                _db_rms_e = 20.0 * np.log10(np.maximum(rms_frames, 1e-9))
                _db_range_e = float(np.percentile(_db_rms_e, 90) - np.percentile(_db_rms_e, 10))
                variance_score = float(np.clip((_db_range_e - 1.0) / 17.0, 0.0, 1.0))
            else:
                variance_score = 0.5

            # §9.7.28 Fix: Nur rausch-stabile Komponenten — crest_score + variance_score.
            # Spectral Flux und PVR wurden entfernt: beide steigen mit additivem Rauschen.
            # crest_score (Hüllkurven-Crest): Rauschen → flach → ~0; Musik m. Dynamik → hoch.
            # variance_score (dB-Bereich p90-p10): Pausen/Crescendo → hoch; steady-state → 0.
            # Kalibrierung: echte Musik mit Pausen liefert crest_score ~0.40–0.80 → 0.76.
            scores["emotionalitaet"] = float(np.clip(0.60 * crest_score + 0.40 * variance_score, 0.0, 1.0))
            # §9.7.5 Preservation: RMS-envelope correlation (dynamics preservation)
            if _ref_mono is not None:
                _rm_ml = min(len(mono), len(_ref_mono))
                _ref_rms = np.array(
                    [
                        float(np.sqrt(np.mean(_ref_mono[i : i + hop_e] ** 2) + 1e-12))
                        for i in range(0, _rm_ml - hop_e, hop_e)
                    ]
                )
                _proc_rms = np.array(
                    [float(np.sqrt(np.mean(mono[i : i + hop_e] ** 2) + 1e-12)) for i in range(0, _rm_ml - hop_e, hop_e)]
                )
                _r = _safe_pearson(_ref_rms, _proc_rms)
                if _r > 0.7:
                    scores["emotionalitaet"] = min(1.0, scores["emotionalitaet"] + (_r - 0.7) * 0.5)
    except Exception:
        scores["emotionalitaet"] = 0.5

    # ── Transparenz (§9.7.13 Multi-Band Spectral Crest Factor, 5 octaves) ─
    # Root-cause of prior false regressions: the 75%-rolloff proxy was
    # SNR-dependent.  Broadband noise raises the high-frequency content
    # → rolloff climbs to 8–12 kHz before denoising; after denoising only
    # musical content remains → rolloff drops to 3–5 kHz → false P4 regression.
    # The §9.7.5 rolloff-floor fix (85 % of reference) only partially mitigated
    # this, and didn't help for phases processed without a reference snapshot.
    #
    # Fix §9.7.13: Multi-band spectral crest factor across 5 octave bands
    #   (250–500 · 500–1k · 1k–2k · 2k–4k · 4k–8k Hz).
    #   • Noise fills each band's floor (raises p50 toward p95) → low crest → low score.
    #   • Denoising clears each band's floor → p50 drops toward musical valleys
    #     → crest rises in ALL bands → score improves → no false regression.
    #   • Reference-free by design: both scores_before and scores_after use the
    #     same absolute formula → symmetric delta even without a clean reference.
    # Scientific basis: Moore & Glasberg (1983); ITU-T P.862 spectral clarity.
    # Calibration: mean crest 1.2 → score 0.0; mean crest 10.0 → score 1.0.
    try:
        _rms_trp = float(np.sqrt(np.mean(mono**2) + 1e-12))
        if _rms_trp < 1e-5:
            scores["transparenz"] = 0.5  # near-silence: neutral (band crests → 0.0 is misleading)
        else:
            # §9.7.25 Fix (v10.0.0): Multi-Band-Crest durch STFT-Averaging zu niedrig.
            # Root-cause: frame-averaged STFT glättet harmonische Peaks → p95/p50 ≈ 1.
            # Fix: Vergleiche mittlere Band-Energie gegen die Gesamtenergie — Transparenz
            # bedeutet, dass Mitten und Höhen klar definiert sind (hohe Band-SNR pro Oktave).
            # Proxy: Harmonic-Energy-Ratio = mittlere Band-Energie / totale Energie.
            # Saubere Aufnahme: jede Oktave hat klar definierte Energie → hohe Konsistenz.
            # Verrauscht/verzerrt: Noise-Floor hebt alle Bänder an → weniger Kontrast.
            # Kalibrierung: Vergleich der Oktav-Energie-Gleichmäßigkeit (Gini-analog).
            _oct_bands_t = [(250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
            _band_energies_t: list[float] = []
            for _fl_t, _fh_t in _oct_bands_t:
                _b_t = fft_mag[(freqs >= _fl_t) & (freqs < _fh_t)]
                if len(_b_t) > 5:
                    _band_energies_t.append(float(np.mean(_b_t**2)))
            if len(_band_energies_t) >= 3:
                _be_arr = np.array(_band_energies_t, dtype=np.float64)
                _be_sum = _be_arr.sum() + 1e-12
                # Normiere auf relative Energie-Verteilung
                _be_norm = _be_arr / _be_sum
                # Transparenz: spektrale Energie ist klar auf definierte Bänder konzentriert.
                # Geom/Arith Flatness: niedrig = gut konzentriert = hohe Transparenz
                _geom_t = float(np.exp(np.mean(np.log(_be_norm + 1e-12))))
                _arith_t = float(np.mean(_be_norm))
                _flat_t = float(np.clip(_geom_t / (_arith_t + 1e-12), 0.0, 1.0))
                # Calibration: flatness 0 → score 1.0 (all energy in 1 band); 0.5 → score 0.0
                scores["transparenz"] = float(np.clip(1.0 - _flat_t * 2.0, 0.0, 1.0))
            else:
                scores["transparenz"] = 0.5
    except Exception:
        scores["transparenz"] = 0.5

    # ── Spatial Depth (M/S-Korrelation bei Stereo, 0.5 bei Mono) ──────
    try:
        # §2.51/§2.63 Stereo-Orientierungs-Fix (v10.0.0): UV3 übergibt (2,N) channels-first.
        # audio.shape[1] >= 2 war bei (2,N) True (N >> 2), aber audio[:,0] = 2-Sample-Spalte ≠ Kanal.
        # Folge: left/right = 2-Sample-Array → near-silence RMS → score = 0.5 (konstant).
        # Fix: orientierungs-adaptives L/R-Splitting — analog zur mono-Konvertierung oben.
        _audio_sd = audio
        if audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2:
            _audio_sd = audio.T  # (2,N) channels-first → (N,2) samples-first
        if _audio_sd.ndim == 2 and _audio_sd.shape[1] >= 2:
            left = _audio_sd[:, 0].astype(np.float32)
            right = _audio_sd[:, 1].astype(np.float32)
            _rms_sd = float(np.sqrt(np.mean(left**2) + np.mean(right**2) + 1e-12))
            if _rms_sd < 1e-5:
                scores["spatial_depth"] = 0.5  # near-silence: 1e-12/(2e-12) → ratio=0.5 → score 1.0 (misleading)
            else:
                mid = (left + right) * 0.5
                side = (left - right) * 0.5
                mid_e = float(np.mean(mid**2) + 1e-12)
                side_e = float(np.mean(side**2) + 1e-12)
                # Hohe Side-Energie = breites Stereo-Bild = hohe Räumlichkeit
                # Normierung: S/M-Ratio ≥ 0.5 = sehr breites Stereo → Score 1.0
                stereo_ratio = side_e / (mid_e + side_e)
                scores["spatial_depth"] = float(np.clip(stereo_ratio * 2.0, 0.0, 1.0))
        else:
            scores["spatial_depth"] = 0.5  # Mono: neutral (GoalApplicabilityFilter entscheidet)
    except Exception:
        scores["spatial_depth"] = 0.5

    # ── Mikro-Dynamik (LUFS-Profil-Korrelation 400ms Proxy) ──────────
    try:
        _rms_md = float(np.sqrt(np.mean(mono**2) + 1e-12))
        if _rms_md < 1e-5:
            scores["micro_dynamics"] = 0.5
        else:
            # Proxy: RMS-Varianz über 400ms-Fenster (äquivalent zu LUFS-Profil-Korrelation)
            win_400ms = max(1, int(sr * 0.4))
            hop_400ms = win_400ms // 4
            rms_400 = np.array(
                [
                    float(np.sqrt(np.mean(mono[i : i + win_400ms] ** 2) + 1e-12))
                    for i in range(0, len(mono) - win_400ms, hop_400ms)
                ]
            )
            if len(rms_400) > 2:
                # Gleichmäßige Variation über 400ms-Fenster = gute Mikro-Dynamik
                # (weder totales Limiting noch extreme Spitzen)
                db_profile = 20.0 * np.log10(rms_400 + 1e-12)
                db_range = float(np.max(db_profile) - np.min(db_profile))
                # Gesunder Bereich: 3–18 dB Variation
                scores["micro_dynamics"] = float(np.clip((db_range - 1.0) / 17.0, 0.0, 1.0))
            else:
                scores["micro_dynamics"] = 0.5
    except Exception:
        scores["micro_dynamics"] = 0.5

    # ── Separation-Treue (Mel-Band-Flatness, §9.7.21 Fix v10.0.0) ────
    # Root-cause: Identischer Bug wie authentizitaet — full-FFT-Flatness auf Leistungs-
    # spektrum ist immer nahe 0 für Musik weil leere Bins den Geom-Mittelwert dominieren.
    # Fix: 24 Mel-Bänder aggregieren Energie korrekt → Flatness unterscheidet tonal/Rauschen:
    #   Tonal (3–6 dominante Bänder): mel_flatness ~ 0.001–0.05 → Score 0.875–0.997
    #   Gemischtes Signal (SNR 20 dB): mel_flatness ~ 0.12 → Score 0.70
    #   Stark verrauscht (SNR 10 dB): mel_flatness ~ 0.55 → Score 0.0
    try:
        _rms_sep = float(np.sqrt(np.mean(mono**2) + 1e-12))
        if _rms_sep < 1e-5:
            scores["separation_fidelity"] = 0.5
        else:
            _mel_sep_edges = np.exp(np.linspace(np.log(100.0), np.log(16000.0), 25)).astype(np.float32)
            _mel_sep_powers: list[float] = []
            for _k in range(24):
                _m_sep = (freqs >= _mel_sep_edges[_k]) & (freqs < _mel_sep_edges[_k + 1])
                if np.sum(_m_sep) > 0:
                    _mel_sep_powers.append(float(np.mean(fft_mag[_m_sep] ** 2)) + 1e-9)
            if len(_mel_sep_powers) >= 8:
                _mp_sep = np.array(_mel_sep_powers, dtype=np.float64)
                _geom_sep = float(np.exp(np.mean(np.log(_mp_sep))))
                _arith_sep = float(np.mean(_mp_sep))
                _flatness_sep = float(np.clip(_geom_sep / (_arith_sep + 1e-12), 0.0, 1.0))
                # Calibration: flatness 0.40 → score 0; 0.10 → 0.75; 0.02 → 0.95
                scores["separation_fidelity"] = float(np.clip(1.0 - _flatness_sep * 2.5, 0.0, 1.0))
            else:
                scores["separation_fidelity"] = 0.5
            # §9.7.5 Preservation: Full-band spectral magnitude coherence
            if _ref_fft is not None:
                _fl = min(len(fft_mag), len(_ref_fft))
                if _fl > 20:
                    _r = _safe_pearson(_ref_fft[:_fl], fft_mag[:_fl])
                    if _r > 0.7:
                        scores["separation_fidelity"] = min(1.0, scores["separation_fidelity"] + (_r - 0.7) * 0.5)
    except Exception:
        scores["separation_fidelity"] = 0.5

    # ── Artikulation (Onset-Schärfe: Attack-Zeit-Proxy) ────────────────
    # Scientific basis: Grey & Gordon (1978), JASA 63:1493; attack time is a
    # primary correlate of perceived articulation and instrument onset clarity.
    try:
        _rms_art = float(np.sqrt(np.mean(mono**2) + 1e-12))
        if _rms_art < 1e-5:
            scores["artikulation"] = 0.5
        else:
            hop_a = max(1, sr // 200)  # 5 ms
            # Vectorized: non-overlapping peak envelope via reshape
            _nf_a = (len(mono) - 1) // hop_a
            env_a = (
                np.max(np.abs(mono[: _nf_a * hop_a].reshape(_nf_a, hop_a)), axis=1)
                if _nf_a > 0
                else np.empty(0, dtype=np.float32)
            )
            if len(env_a) > 4:
                # Erste Ableitung der Hüllkurve
                d_env = np.diff(env_a)
                # Starke positive Sprünge = scharfe Anschläge (Artikulation)
                pos_peaks = d_env[d_env > 0]
                if len(pos_peaks) > 0:
                    onset_sharpness = float(np.mean(pos_peaks))
                    # Log-scaled normalization approximates perceptual spacing of attack times:
                    # 0.003 amplitude/5ms ≈ slow attack (~50 ms), 0.030 ≈ sharp attack (~5 ms).
                    scores["artikulation"] = float(
                        np.clip(
                            (math.log10(onset_sharpness + 1e-12) - math.log10(0.003))
                            / (math.log10(0.030) - math.log10(0.003)),
                            0.0,
                            1.0,
                        )
                    )
                else:
                    scores["artikulation"] = 0.3  # Keine Transienten = schlechte Artikulation
            else:
                scores["artikulation"] = 0.5
    except Exception:
        scores["artikulation"] = 0.5

    # ── Transient-Energie (§1.4.6): onset-amplitude-ratio proxy (independent of artikulation) ──
    # Identische Formel wie _fast_goal_snapshot() in UV3: p90(positive RMS-Diffs) / p50(RMS).
    # VERBOTEN: artikulation-Score als Proxy — führt zu falschen Kovarianzen bei VAD-phasen.
    try:
        _rms_frames_te = []
        _hop_te = 512
        _N_te = len(mono)
        for _i in range(0, _N_te - _hop_te, _hop_te):
            _rms_frames_te.append(float(np.sqrt(np.mean(mono[_i : _i + _hop_te] ** 2))))
        if len(_rms_frames_te) >= 4:
            _arr_te = np.array(_rms_frames_te, dtype=np.float64)
            _diff_te = np.maximum(np.diff(_arr_te), 0.0)
            _eps_te = 1e-9
            if _diff_te.size > 0 and float(np.max(_diff_te)) > _eps_te:
                _oe = float(np.percentile(_diff_te, 90))
                _se = float(np.percentile(_arr_te, 50)) + _eps_te
                scores["transient_energie"] = float(np.clip((_oe / _se) * 2.0, 0.0, 1.0))
            else:
                scores["transient_energie"] = 0.5
        else:
            scores["transient_energie"] = 0.5
    except Exception:
        scores["transient_energie"] = 0.5

    # NaN-guard (§3.1) — all 15 canonical keys including "natuerlichkeit"
    for k in FAST_GOALS_SUBSET:
        if k not in scores or not math.isfinite(scores[k]):
            scores[k] = 0.5

    _vocal_presence = float(_vocal_guard.get("vocal_presence_proxy", 0.0))
    if enable_vocal_guard and reference is not None and _vocal_presence >= _VOCAL_GUARD_TRIGGER:
        _activation = float(
            np.clip(
                (_vocal_presence - _VOCAL_GUARD_TRIGGER) / (1.0 - _VOCAL_GUARD_TRIGGER),
                0.0,
                1.0,
            )
        )

        def _bonus_scale(feature_name: str) -> float:
            return float(np.clip((float(_vocal_guard.get(feature_name, 0.5)) - 0.55) / 0.45, 0.0, 1.0))

        _formant_bonus = _bonus_scale("vocal_formant_stability")
        _fricative_bonus = _bonus_scale("vocal_fricative_stability")
        _transient_bonus = _bonus_scale("vocal_transient_integrity")
        scores["natuerlichkeit"] = min(1.0, scores["natuerlichkeit"] + 0.08 * _activation * _formant_bonus)
        scores["timbre_authentizitaet"] = min(
            1.0,
            scores["timbre_authentizitaet"] + 0.08 * _activation * (0.70 * _formant_bonus + 0.30 * _fricative_bonus),
        )
        scores["artikulation"] = min(
            1.0,
            scores["artikulation"] + 0.08 * _activation * (0.70 * _transient_bonus + 0.30 * _fricative_bonus),
        )
        scores["emotionalitaet"] = min(
            1.0,
            scores["emotionalitaet"] + 0.05 * _activation * (0.60 * _formant_bonus + 0.40 * _transient_bonus),
        )

    if precise_override:
        scores = _apply_precise_metric_overrides(scores, audio, sr, reference=reference)

    # §v10.15: Listening Fatigue via DSP proxy
    try:
        from backend.core.listening_fatigue_metric import fatigue_as_pmgg_goal

        scores["listening_fatigue"] = fatigue_as_pmgg_goal(audio, sr)
    except Exception:
        scores["listening_fatigue"] = 0.5

    for k in FAST_GOALS_SUBSET:
        if k not in scores or not math.isfinite(scores[k]):
            scores[k] = 0.5

    return scores


# Timing phases: intentional time-warping makes *any* correlation metric unreliable.
# 163 transport bumps → envelope reordering → corr≈0.265 even on perfect correction.
# Excluding corr_pen for these phases prevents false Content-Guard rollbacks (§2.48 §2.54).
_TIMING_CORR_EXCLUDE: frozenset[str] = frozenset(
    {
        "phase_12_wow_flutter_fix",
        "phase_31_speed_pitch_correction",
    }
)

# Phasen mit lokal rekonstruktivem Charakter: Ziel ist primär Defektfenster-Reparatur,
# daher muss PMGG lokale Verbesserungen von globalem Kollateralschaden trennen.
_RECONSTRUCTION_COUNTERFACTUAL_PHASE_PREFIXES: tuple[str, ...] = (
    "phase_23",
    "phase_24",
    "phase_50",
    "phase_55",
)

# LF-subtractive phases: intentional broadband RMS reduction when low-end noise dominates.
# phase_02 (hum), phase_05 (rumble): removing sub-bass / 50 Hz hum CAN reduce broadband RMS
# by 20-30 dB if that noise dominated the signal — this is CORRECT behaviour (§0).
# Using broadband RMS for the drop-penalty creates false Content-Guard rollbacks.
# These phases already have internal §2.45a guards; drop-penalty in PMGG is redundant.
_LF_SUBTRACTIVE_DROP_SKIP: frozenset[str] = frozenset(
    {
        "phase_02_hum_removal",
        "phase_05_rumble_filter",
    }
)


def _content_integrity_penalty(
    reference: np.ndarray,
    processed: np.ndarray,
    skip_corr_check: bool = False,
    skip_drop_check: bool = False,
) -> tuple[float, dict[str, float]]:
    """Erkennt catastrophic content loss independently from Musical-Goal proxies.

    The guard is intentionally conservative and only reacts to severe failures:
    large broadband energy collapse and/or very low waveform correlation.
    It protects PMGG when many P1/P2 goals are excluded for a phase.

    §9.11.2: Uses RMS-envelope correlation instead of raw sample correlation.
    Time-domain phases (wow/flutter phase vocoder, time-stretch) shift samples
    in time without destroying content — raw corrcoef yields ~0 (false positive).
    10 ms RMS-envelope correlation is time-shift-tolerant while still detecting
    genuine content loss (energy distribution, spectral balance changes).

    skip_corr_check: When True (timing phases with intentional global time-warp),
    the corr_pen component is zeroed — only the RMS-drop component remains active.
    This prevents false Content-Guard rollbacks when 100+ transport bumps are
    corrected (§2.48 Carrier-Repair-Exclusions, §2.54 adaptive thresholds).
    """
    try:
        _ref = np.asarray(reference, dtype=np.float32)
        _out = np.asarray(processed, dtype=np.float32)
        if _ref.ndim == 2 and _ref.shape[1] >= 2:
            _ref_mono = ((_ref[:, 0] + _ref[:, 1]) / np.sqrt(2.0)).astype(np.float32)
        else:
            _ref_mono = (_ref[:, 0] if _ref.ndim == 2 else _ref).astype(np.float32)
        if _out.ndim == 2 and _out.shape[1] >= 2:
            _out_mono = ((_out[:, 0] + _out[:, 1]) / np.sqrt(2.0)).astype(np.float32)
        else:
            _out_mono = (_out[:, 0] if _out.ndim == 2 else _out).astype(np.float32)

        _n = min(len(_ref_mono), len(_out_mono))
        if _n < 256:
            return 0.0, {"rms_drop_db": 0.0, "corr": 1.0}
        _ref_mono = np.nan_to_num(_ref_mono[:_n], nan=0.0, posinf=0.0, neginf=0.0)
        _out_mono = np.nan_to_num(_out_mono[:_n], nan=0.0, posinf=0.0, neginf=0.0)

        _rms_ref = float(np.sqrt(np.mean(_ref_mono**2) + 1e-12))
        _rms_out = float(np.sqrt(np.mean(_out_mono**2) + 1e-12))
        if _rms_ref < 1e-5:
            return 0.0, {"rms_drop_db": 0.0, "corr": 1.0}
        _rms_drop_db = float(20.0 * np.log10((_rms_ref + 1e-12) / (_rms_out + 1e-12)))

        # §9.11.2 RMS-envelope correlation (time-shift-tolerant).
        # 10 ms frames at 48 kHz = 480 samples per frame.
        _hop = max(256, min(480, _n // 16))
        _n_frames = _n // _hop
        if _n_frames >= 4:
            _ref_env = np.array(
                [np.sqrt(np.mean(_ref_mono[i * _hop : (i + 1) * _hop] ** 2) + 1e-12) for i in range(_n_frames)],
                dtype=np.float32,
            )
            _out_env = np.array(
                [np.sqrt(np.mean(_out_mono[i * _hop : (i + 1) * _hop] ** 2) + 1e-12) for i in range(_n_frames)],
                dtype=np.float32,
            )
            _std_env_ref = float(np.std(_ref_env))
            _std_env_out = float(np.std(_out_env))
            if _std_env_ref < 1e-7 or _std_env_out < 1e-7:
                _corr = 1.0 if abs(_std_env_ref - _std_env_out) < 1e-7 else 0.0
            else:
                _ae = _ref_env - _ref_env.mean()
                _be = _out_env - _out_env.mean()
                _nae = float(np.linalg.norm(_ae))
                _nbe = float(np.linalg.norm(_be))
                _corr = float(np.dot(_ae, _be) / (_nae * _nbe + 1e-10))
                if not np.isfinite(_corr):
                    _corr = 0.0
        else:
            # Too few frames — fall back to sample correlation
            _std_ref = float(np.std(_ref_mono))
            _std_out = float(np.std(_out_mono))
            if _std_ref < 1e-7 or _std_out < 1e-7:
                _corr = 1.0 if abs(_std_ref - _std_out) < 1e-7 else 0.0
            else:
                _am = _ref_mono - _ref_mono.mean()
                _bm = _out_mono - _out_mono.mean()
                _nam = float(np.linalg.norm(_am))
                _nbm = float(np.linalg.norm(_bm))
                _corr = float(np.dot(_am, _bm) / (_nam * _nbm + 1e-10))
                if not np.isfinite(_corr):
                    _corr = 0.0

        # Conservative thresholds: trigger only catastrophic changes.
        # skip_drop_check: LF-subtractive phases (rumble, hum) can lower broadband RMS
        # dramatically when noise dominates the signal — this is correct (§0 §2.45a).
        _drop_pen = 0.0 if skip_drop_check else max(0.0, min(1.0, (_rms_drop_db - 12.0) / 12.0))
        # skip_corr_check: timing phases with intentional global time-warp produce
        # low envelope correlation by design — do NOT penalise (§2.48/§2.54).
        _corr_pen = 0.0 if skip_corr_check else max(0.0, min(1.0, (0.55 - _corr) / 0.55))
        _penalty = float(max(_drop_pen, _corr_pen))
        return _penalty, {"rms_drop_db": _rms_drop_db, "corr": _corr}
    except Exception as e:
        logger.warning("per_Verarbeitungsschritt_musical_goals_gate.py::unbekannter Ersatzpfad: %s", e)
        return 0.0, {"rms_drop_db": 0.0, "corr": 1.0}


def _targeted_defect_keys_for_phase(phase_id: str) -> tuple[str, ...]:
    """Gibt defect-location keys used for sparse-defect sample targeting zurück."""
    _phase_defect_keys = {
        "phase_23": ("SPECTRAL_HOLES", "spectral_holes", "DROPOUTS", "dropouts"),
        "phase_24": ("DROPOUTS", "DROPOUT", "dropouts"),
        "phase_50": ("SPECTRAL_HOLES", "spectral_holes", "DROPOUTS", "dropouts"),
        "phase_27": ("DROPOUTS", "DROPOUT", "dropouts"),
        "phase_55": ("DROPOUTS", "DROPOUT", "dropouts", "SPECTRAL_HOLES", "spectral_holes"),
        "phase_09": ("CRACKLE", "crackle", "CLICKS", "clicks"),
        "phase_01": ("CLICKS", "clicks", "CLICK", "click"),
    }
    for _prefix, _keys in _phase_defect_keys.items():
        if phase_id.startswith(_prefix):
            return _keys
    return ()


def _get_sample_window_bounds(
    audio_len: int,
    sr: int,
    duration_s: float = SAMPLE_DURATION_S,
    defect_locations: dict[str, list[tuple[float, float]]] | None = None,
    phase_id: str = "",
) -> tuple[int, int]:
    """Gibt the PMGG sample-window bounds for the given phase zurück."""
    sample_len = min(int(duration_s * sr), audio_len)
    if audio_len <= sample_len:
        return 0, audio_len

    _sparse_defect_phases = frozenset(
        {
            "phase_24",
            "phase_27",
            "phase_55",
            "phase_09",
            "phase_01",
        }
    )
    if defect_locations and any(phase_id.startswith(p) for p in _sparse_defect_phases):
        _best_start_s = None
        for _dk in _targeted_defect_keys_for_phase(phase_id):
            locs = defect_locations.get(_dk, [])
            if locs and isinstance(locs[0], (tuple, list)) and len(locs[0]) >= 1:
                _best_start_s = float(locs[0][0])
                break
        if _best_start_s is not None:
            _defect_sample = int(_best_start_s * sr)
            start = max(0, min(_defect_sample - sample_len // 2, audio_len - sample_len))
            return start, start + sample_len

    start = (audio_len - sample_len) // 2
    return start, start + sample_len


def _estimate_targeted_defect_coverage_ratio(
    audio_len: int,
    sr: int,
    duration_s: float,
    defect_locations: dict[str, list[tuple[float, float]]] | None,
    phase_id: str,
) -> float | None:
    """Schätzt targeted-defect coverage inside the PMGG sample window."""
    if not defect_locations:
        return None

    _keys = _targeted_defect_keys_for_phase(phase_id)
    if not _keys:
        return None

    start, end = _get_sample_window_bounds(audio_len, sr, duration_s, defect_locations, phase_id)
    _intervals: list[tuple[int, int]] = []
    for _dk in _keys:
        for _loc in defect_locations.get(_dk, []) or []:
            if not isinstance(_loc, (tuple, list)) or len(_loc) < 2:
                continue
            try:
                _s = int(float(_loc[0]) * sr)
                _e = int(float(_loc[1]) * sr)
            except (TypeError, ValueError):
                continue
            if _e <= _s:
                continue
            _ov_s = max(start, _s)
            _ov_e = min(end, _e)
            if _ov_e > _ov_s:
                _intervals.append((_ov_s, _ov_e))

    if not _intervals:
        return None

    _intervals.sort()
    _merged: list[tuple[int, int]] = []
    for _s, _e in _intervals:
        if _merged and _s <= _merged[-1][1]:
            _merged[-1] = (_merged[-1][0], max(_merged[-1][1], _e))
        else:
            _merged.append((_s, _e))

    _covered = sum(_e - _s for _s, _e in _merged)
    _window = max(1, end - start)
    return float(np.clip(_covered / _window, 0.0, 1.0))


def _window_targeted_defect_coverage_ratio(
    intervals: list[tuple[int, int]],
    start: int,
    end: int,
) -> float:
    """Berechnet die Defektabdeckung für ein beliebiges Fenster."""
    if end <= start or not intervals:
        return 0.0

    covered = 0
    for interval_start, interval_end in intervals:
        overlap_start = max(start, interval_start)
        overlap_end = min(end, interval_end)
        if overlap_end > overlap_start:
            covered += overlap_end - overlap_start
    return float(np.clip(covered / max(1, end - start), 0.0, 1.0))


def _get_reconstruction_control_window_bounds(
    audio_len: int,
    sr: int,
    duration_s: float,
    defect_locations: dict[str, list[tuple[float, float]]] | None,
    phase_id: str,
) -> tuple[int, int, float, float] | None:
    """Wählt ein kontrollfenster außerhalb der Rekonstruktions-Defekte.

    Rekonstruktive Phasen wie phase_24/55 sollen lokal im Defektfenster bewertet
    werden, dürfen aber außerhalb dieser Fenster keinen Kollateralschaden
    verursachen. Dieses Hilfsfenster dient genau dieser Invarianz-Prüfung.
    """
    if not phase_id.startswith(_RECONSTRUCTION_COUNTERFACTUAL_PHASE_PREFIXES) or not defect_locations:
        return None

    sample_len = min(int(duration_s * sr), audio_len)
    if audio_len <= sample_len:
        return None

    intervals: list[tuple[int, int]] = []
    for defect_key in _targeted_defect_keys_for_phase(phase_id):
        for location in defect_locations.get(defect_key, []) or []:
            if not isinstance(location, (tuple, list)) or len(location) < 2:
                continue
            try:
                interval_start = int(float(location[0]) * sr)
                interval_end = int(float(location[1]) * sr)
            except (TypeError, ValueError):
                continue
            if interval_end > interval_start:
                intervals.append((interval_start, interval_end))

    if not intervals:
        return None

    intervals.sort()
    target_start, target_end = _get_sample_window_bounds(audio_len, sr, duration_s, defect_locations, phase_id)
    target_coverage = _window_targeted_defect_coverage_ratio(intervals, target_start, target_end)
    if target_coverage < 0.05:
        return None

    candidate_starts = [
        0,
        max(0, audio_len // 4 - sample_len // 2),
        max(0, audio_len // 2 - sample_len // 2),
        max(0, (3 * audio_len) // 4 - sample_len // 2),
        max(0, audio_len - sample_len),
    ]
    unique_candidate_starts: list[int] = []
    for candidate_start in candidate_starts:
        candidate_start = int(np.clip(candidate_start, 0, max(0, audio_len - sample_len)))
        if candidate_start not in unique_candidate_starts:
            unique_candidate_starts.append(candidate_start)

    best_window: tuple[int, int, float] | None = None
    for candidate_start in unique_candidate_starts:
        candidate_end = candidate_start + sample_len
        overlap = max(0, min(candidate_end, target_end) - max(candidate_start, target_start))
        if overlap > sample_len * 0.25:
            continue
        coverage = _window_targeted_defect_coverage_ratio(intervals, candidate_start, candidate_end)
        if best_window is None or coverage < best_window[2]:
            best_window = (candidate_start, candidate_end, coverage)

    if best_window is None:
        return None

    control_start, control_end, control_coverage = best_window
    if control_coverage > 0.02 or control_coverage >= target_coverage * 0.5:
        return None
    return control_start, control_end, target_coverage, control_coverage


def _assess_reconstruction_localized_confidence(
    *,
    target_coverage: float,
    control_coverage: float,
    control_regression: float,
    threshold: float,
) -> tuple[bool, float, str]:
    """Bewertet die Zuverlaessigkeit einer localized-reconstruction-Freigabe.

    Ziel: Rekonstruktive Phasen (phase_24/55) duerfen lokale Defektfenster
    verschlechtern, solange ausserhalb dieser Fenster kein Kollateralschaden
    nachweisbar ist. Diese Funktion quantifiziert die Entscheidungssicherheit,
    statt nur einen harten booleschen Schwellwert zu verwenden.
    """
    if control_regression > threshold:
        return False, 0.0, "control_regression_over_threshold"

    _target_term = float(np.clip((target_coverage - 0.08) / 0.22, 0.0, 1.0))
    _control_term = float(np.clip((0.02 - control_coverage) / 0.02, 0.0, 1.0))
    _margin_term = float(np.clip((threshold - control_regression) / (threshold + 1e-9), 0.0, 1.0))
    confidence = float(np.clip(0.45 * _target_term + 0.35 * _control_term + 0.20 * _margin_term, 0.0, 1.0))

    if target_coverage < 0.08:
        reason = "low_target_coverage"
    elif control_coverage > 0.015:
        reason = "control_window_partially_contaminated"
    elif confidence < 0.55:
        reason = "confidence_below_acceptance"
    else:
        reason = "high_confidence"

    return confidence >= 0.55, confidence, reason


def _compute_reconstruction_epistemic_confidence(
    *,
    localized_confidence: float,
    target_coverage: float,
    control_coverage: float,
    transfer_chain_tcci: float,
    threshold_multiplier: float,
    phase_kwargs: dict[str, Any] | None,
) -> tuple[float, str]:
    """Schätzt epistemische Sicherheit über den reinen Localized-Proxy hinaus.

    Ziel: Proxy-Entscheidungen nicht nur nach einem einzelnen Fensterwert,
    sondern nach einer kleinen deterministischen Evidenz-Fusion bewerten.
    """
    _local = float(np.clip(localized_confidence, 0.0, 1.0))
    _target_term = float(np.clip((target_coverage - 0.06) / 0.24, 0.0, 1.0))
    _control_term = float(np.clip((0.025 - control_coverage) / 0.025, 0.0, 1.0))
    _chain_term = float(np.clip(1.0 - 0.35 * np.clip(transfer_chain_tcci, 0.0, 1.0), 0.0, 1.0))
    _threshold_term = float(np.clip((np.clip(threshold_multiplier, 0.8, 1.2) - 0.8) / 0.4, 0.0, 1.0))

    _phase_kwargs = phase_kwargs if isinstance(phase_kwargs, dict) else {}
    try:
        _vocal_probability = float(_phase_kwargs.get("vocal_probability", 0.0) or 0.0)
    except (TypeError, ValueError):
        _vocal_probability = 0.0
    _vocal_penalty = float(np.clip((_vocal_probability - 0.35) / 0.45, 0.0, 1.0)) * 0.10

    _epistemic = (
        0.45 * _local + 0.25 * _target_term + 0.15 * _control_term + 0.10 * _chain_term + 0.05 * _threshold_term
    )
    _epistemic = float(np.clip(_epistemic - _vocal_penalty, 0.0, 1.0))

    if _epistemic >= 0.78:
        _reason = "epistemic_high"
    elif _epistemic >= 0.58:
        _reason = "epistemic_medium"
    else:
        _reason = "epistemic_low"
    return _epistemic, _reason


def _compute_reconstruction_retry_budget_bias(
    *,
    accepted: bool,
    confidence: float,
    phase_kwargs: dict[str, Any] | None,
    phase_id: str,
) -> tuple[int, str, float, str]:
    """Berechnet einen kleinen material- und chain-adaptiven Retry-Bias."""

    _phase_kwargs = phase_kwargs if isinstance(phase_kwargs, dict) else {}
    _mat_key = _material_key_from_phase_kwargs(_phase_kwargs)
    _chain_raw = _phase_kwargs.get("transfer_chain")
    if _chain_raw is None and isinstance(_phase_kwargs.get("prior_phase_context"), dict):
        _chain_raw = _phase_kwargs["prior_phase_context"].get("transfer_chain")
    _chain = [str(v).strip().lower() for v in (_chain_raw or []) if str(v).strip()]
    _tcci = float(np.clip(compute_tcci(_chain), 0.0, 1.0))

    _analog_materials = {"vinyl", "shellac", "tape", "reel_tape", "cassette", "wire_recording", "wax_cylinder"}
    _digital_lossy_materials = {"mp3_low", "mp3_high", "aac", "streaming", "minidisc"}
    _material_family = "digital"
    if _mat_key in _analog_materials:
        _material_family = "analog"
    elif _mat_key in _digital_lossy_materials:
        _material_family = "lossy_digital"

    _bias = 1 if accepted and confidence >= 0.80 else -1 if (not accepted and confidence < 0.35) else 0
    _reason_parts = [f"conf={confidence:.2f}"]

    if _bias > 0 and _material_family == "analog":
        _bias -= 1
        _reason_parts.append("analog_damped")
    elif _bias > 0 and _material_family == "lossy_digital" and confidence >= 0.85:
        _bias += 1
        _reason_parts.append("lossy_digital_boost")

    if _tcci >= 0.65:
        _bias -= 1
        _reason_parts.append(f"chain_tcci={_tcci:.2f}")
    elif _tcci <= 0.20 and accepted and confidence >= 0.75:
        _bias += 1
        _reason_parts.append(f"chain_tcci={_tcci:.2f}")

    if phase_id.startswith("phase_50") and _chain and _material_family == "analog":
        _bias -= 1
        _reason_parts.append("phase50_analog_guard")

    return int(np.clip(_bias, -2, 2)), ";".join(_reason_parts), _tcci, _material_family


def _compute_reconstruction_threshold_multiplier(
    *,
    phase_kwargs: dict[str, Any] | None,
    phase_id: str,
) -> tuple[float, str, float, str]:
    """Berechnet eine kleine material- und chain-adaptive Threshold-Anpassung."""

    _phase_kwargs = phase_kwargs if isinstance(phase_kwargs, dict) else {}
    _mat_key = _material_key_from_phase_kwargs(_phase_kwargs)
    _chain_raw = _phase_kwargs.get("transfer_chain")
    if _chain_raw is None and isinstance(_phase_kwargs.get("prior_phase_context"), dict):
        _chain_raw = _phase_kwargs["prior_phase_context"].get("transfer_chain")
    _chain = [str(v).strip().lower() for v in (_chain_raw or []) if str(v).strip()]
    _tcci = float(np.clip(compute_tcci(_chain), 0.0, 1.0))

    _analog_materials = {"vinyl", "shellac", "tape", "reel_tape", "cassette", "wire_recording", "wax_cylinder"}
    _digital_lossy_materials = {"mp3_low", "mp3_high", "aac", "streaming", "minidisc"}

    _multiplier = 1.0
    _reason_parts = [f"tcci={_tcci:.2f}"]

    if _mat_key in _analog_materials:
        _multiplier *= 0.94
        _reason_parts.append("analog_tolerance_reduced")
    elif _mat_key in _digital_lossy_materials:
        _multiplier *= 0.98
        _reason_parts.append("lossy_digital_neutral")
    else:
        _multiplier *= 1.02
        _reason_parts.append("digital_safe_relaxation")

    if _tcci >= 0.65:
        _multiplier *= 0.92
        _reason_parts.append("chain_complexity_high")
    elif _tcci <= 0.20:
        _multiplier *= 1.04
        _reason_parts.append("chain_complexity_low")

    if phase_id.startswith("phase_50") and _mat_key in _analog_materials:
        _multiplier *= 0.90
        _reason_parts.append("phase50_analog_guard")

    return float(np.clip(_multiplier, 0.80, 1.05)), ";".join(_reason_parts), _tcci, _mat_key


def _reconstruction_goal_recheck_allowlist(  # pylint: disable=too-many-positional-arguments
    phase_id: str,
    phase_kwargs: dict[str, Any] | None,
    defect_locations: dict[str, list[tuple[float, float]]] | None,
    audio_len: int,
    sr: int,
    sample_duration_s: float,
) -> set[str]:
    """Re-enable critical P1 checks for sparse vocal inpainting windows."""
    if not phase_id.startswith(("phase_24", "phase_55")):
        return set()
    if not isinstance(phase_kwargs, dict):
        return set()

    try:
        _vocal_probability = float(phase_kwargs.get("vocal_probability", 0.0) or 0.0)
    except (TypeError, ValueError):
        _vocal_probability = 0.0
    _has_lyrics_guidance = bool(phase_kwargs.get("phoneme_timeline")) or bool(phase_kwargs.get("pre_transcription"))
    if _vocal_probability < 0.15 and not _has_lyrics_guidance:
        return set()

    _coverage = _estimate_targeted_defect_coverage_ratio(
        audio_len,
        sr,
        sample_duration_s,
        defect_locations,
        phase_id,
    )
    if _coverage is None:
        return set()

    _coverage_limit = 0.18 if phase_id.startswith("phase_24") else 0.22
    if _coverage > _coverage_limit:
        return set()

    logger.info(
        "PMGG reconstruction recheck: %s re-enabling P1 guards "
        "(vocal_probability=%.2f, lyrics_guidance=%s, defect_coverage=%.3f)",
        phase_id,
        _vocal_probability,
        _has_lyrics_guidance,
        _coverage,
    )
    return {"natuerlichkeit", "authentizitaet"}


def _extract_sample(
    audio: np.ndarray,
    sr: int,
    duration_s: float = SAMPLE_DURATION_S,
    defect_locations: dict[str, list[tuple[float, float]]] | None = None,
    phase_id: str = "",
) -> np.ndarray:
    """Extrahiert repräsentative Stichprobe aus dem Audio.

    For dropout/transport-bump phases (§9.1a), the sample is centred on the
    first known defect location rather than the audio midpoint, so that the
    PMGG regression check actually evaluates the repaired region.

    For all other phases, the classic centre-crop is used.
    """
    n = len(audio)
    sample_len = min(int(duration_s * sr), n)
    if n <= sample_len:
        return audio

    start, _end = _get_sample_window_bounds(n, sr, duration_s, defect_locations, phase_id)
    return audio[start : start + sample_len]


# ---------------------------------------------------------------------------
# Hauptklasse
# ---------------------------------------------------------------------------


class PerPhaseMusicalGoalsGate:
    """
    Wraps PhaseInterface.process() mit Musical-Goals-Prüfung.

    Alle Methoden sind thread-sicher und NaN/Inf-sicher.
    """

    def __init__(self) -> None:
        """Initialisiert PMGG with zeroed protection counters."""
        self._rollback_count: int = 0  # Echte Audio-Rollbacks; PMGG best_effort zählt separat.
        self._best_effort_count: int = 0  # Pro Restaurierungsaufruf
        self._user_warned: bool = False  # Nutzer-Warnung einmalig
        self._last_retry_budget_policy: dict[str, Any] = {}
        self._last_reconstruction_localized_decision: dict[str, Any] = {}
        self._genre_goal_weights: dict[str, float] = {}  # §H: Genre-abhängige Goal-Gewichte

    def reset(self) -> None:
        """Setzt Zähler für neuen Restaurierungsaufruf zurück."""
        self._rollback_count = 0
        self._best_effort_count = 0
        self._user_warned = False
        self._last_retry_budget_policy = {}
        self._last_reconstruction_localized_decision = {}
        self._genre_goal_weights = {}  # §H

    @staticmethod
    def _resolve_retry_budget_policy(
        phase_kwargs: dict[str, Any] | None,
        *,
        max_retries: int,
        retry_budget_s: float,
    ) -> tuple[int, float, dict[str, Any]]:
        """Gibt capped retry policy when UV3 signals wall-budget pressure zurück."""
        hint = (phase_kwargs or {}).get("retry_budget_hint")
        if not isinstance(hint, dict) or not hint:
            return max_retries, retry_budget_s, {}

        reason = str(hint.get("reason", "") or "")
        retry_budget_scale = float(np.clip(float(hint.get("retry_budget_scale", 1.0) or 1.0), 0.05, 1.0))
        max_retries_cap = int(np.clip(int(hint.get("max_retries_cap", max_retries) or max_retries), 0, max_retries))
        new_max_retries = min(max_retries, max_retries_cap)
        new_retry_budget_s = min(retry_budget_s, max(10.0, retry_budget_s * retry_budget_scale))
        metadata = {
            "active": True,
            "reason": reason,
            "max_retries_cap": new_max_retries,
            "retry_budget_scale": round(retry_budget_scale, 3),
            "retry_budget_seconds": round(float(new_retry_budget_s), 3),
        }
        if "future_priority_phases" in hint:
            metadata["future_priority_phases"] = list(hint.get("future_priority_phases", []))
        if "history_samples" in hint:
            metadata["history_samples"] = int(hint.get("history_samples", 0) or 0)
        if "history_mean_net_gain" in hint:
            metadata["history_mean_net_gain"] = float(hint.get("history_mean_net_gain", 0.0) or 0.0)
        return new_max_retries, new_retry_budget_s, metadata

    def check_phase(
        self,
        phase: Any,
        audio: np.ndarray,
        *,
        sr: int = 48000,
        scores_before: dict[str, float] | None = None,
        effective_goals: list[str] | None = None,
        phase_kwargs: dict[str, Any] | None = None,
        _threshold: float = REGRESSION_THRESHOLD_GOOD,
        **_kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, float], PhaseGateLogEntry]:
        """Public alias to wrap_phase for direct access and testing.

        Simplified signature that accepts scores_before and effective_goals without
        requiring full restorability/calibration context.  Returns the same
        (audio_out, scores_after, PhaseGateLogEntry) triple as wrap_phase.
        """
        _phase_id = self._get_phase_id(phase)
        return self.wrap_phase(
            phase=phase,
            audio=audio,
            sr=sr,
            phase_id=_phase_id,
            scores_before=scores_before,
            phase_kwargs=phase_kwargs,
            applicable_goals=set(effective_goals) if effective_goals else None,
        )

    def wrap_phase(  # pylint: disable=too-many-positional-arguments
        self,
        phase: Any,  # PhaseInterface-Instanz
        audio: np.ndarray,
        sr: int,
        phase_id: str | None = None,
        scores_before: dict[str, float] | None = None,
        phase_kwargs: dict[str, Any] | None = None,
        restorability_score: float = 70.0,
        applicable_goals: set[str] | None = None,
        initial_strength: float = 1.0,
        is_studio_2026: bool = False,
        goal_weights: dict[str, float] | None = None,
        adaptive_goal_thresholds: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, dict[str, float], PhaseGateLogEntry]:
        """
        Führt eine Phase aus und prüft Musical-Goals-Regression.

        Args:
            phase: PhaseInterface-Instanz mit process(audio) → PhaseResult
            audio: Input-Audio (float32)
            sr: 48000 Hz
            phase_id: Optional explicit phase id for backward-compatible callers.
                      If omitted, id is resolved from phase metadata.
            scores_before: Bekannte Scores vor der Phase (werden gemessen
                           wenn nicht übergeben)
            phase_kwargs: Zusätzliche kwargs für den Phase-Aufruf (z.B. sample_rate, material_type)
            restorability_score: RestorabilityEstimator-Score ∈ [0, 100] — bestimmt
                                 adaptiven REGRESSION_THRESHOLD (§2.29).
            applicable_goals: Aus GoalApplicabilityFilter — nur diese Ziele werden
                              geprüft. None = alle FAST_GOALS_SUBSET-Ziele.
            initial_strength: Material-adaptive Initialstärke ∈ (0, 1.0] (§2.29/§2.31).
                              1.0 = volle Stärke (Default). Niedrigere Werte aus
                              _MATERIAL_PHASE_FACTORS schützen Vintage-Charakter
                              (z.B. 0.25 für phase_22_tape_saturation bei shellac).
                              Retry-Stärken skalieren relativ zur Initialstärke.
            is_studio_2026: True if Studio 2026 mode (§9.10.77 Pareto-Differenzierung).
                            Selects higher P3–P5 thresholds. Default: Restoration.

        Returns:
            (audio_out, scores_after, log_entry)
        """
        if sr != 48000:
            logger.debug("PMGG: SR=%s (nicht 48000) — Goal-Messung läuft trotzdem", sr)

        # ── §v10 Input/Output-Validation ─────────────────────────────
        # Validiert Input-Audio vor Phase-Ausführung. NaN/Inf/corrupted
        # audio würde Goal-Scores verfälschen und Phasen crashen lassen.
        if not np.all(np.isfinite(audio)):
            logger.error(
                "PMGG wrap_Verarbeitungsschritt: Eingabe-Audio enthält NaN/Inf — Verarbeitungsschritt %s übersprungen",
                phase_id,
            )
            _safe = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
            _safe = np.clip(_safe, -1.0, 1.0)
            return (
                _safe,
                scores_before or {},
                PhaseGateLogEntry(
                    phase_id=phase_id,  # type: ignore[arg-type]
                    action="validation_error",
                    goal_regressions={},
                    strength_used=0.0,
                    metadata={"error": "input_contains_nan_inf", "rms_drop_db": 0.0, "hpe_delta": 0.0},
                ),
            )

        if phase_kwargs is None:
            phase_kwargs = {}

        # §H: Genre-Goal-Weights als Default, wenn keine explizit übergeben wurden
        if goal_weights is None and self._genre_goal_weights:
            goal_weights = dict(self._genre_goal_weights)

        phase_id = phase_id or self._get_phase_id(phase)
        t0 = time.time()
        _threshold_multiplier = 1.0
        _threshold_reason = "default"
        _threshold_tcci = 0.0
        _threshold_material_family = "unknown"

        # §2.29/§2.54 Material- und Restorability-adaptiven Threshold bestimmen
        _mat_kw_thresh = (phase_kwargs or {}).get("material_type") or (phase_kwargs or {}).get("material")
        _mat_str_thresh = (
            (_mat_kw_thresh.value if hasattr(_mat_kw_thresh, "value") else str(_mat_kw_thresh)).lower()
            if _mat_kw_thresh
            else "unknown"
        )
        threshold = _get_adaptive_threshold(restorability_score, _mat_str_thresh)

        # §2.31a SongCal-Threshold-Feinjustage: global_scalar aus dem Song-Kalibrierungsprofil
        # erlaubt engere Schutzzone bei nahe-sauberem Audio und lockert sie bei stark
        # beschädigtem Material, um unnötige Retry-Zyklen zu vermeiden.
        _calpro_kw = (phase_kwargs or {}).get("song_calibration_profile", {})
        if isinstance(_calpro_kw, dict) and _calpro_kw:
            _gs = float(_calpro_kw.get("global_scalar", 1.0))
            if _gs < 0.85:
                # Near-clean: tighter threshold guards musical purity
                threshold = max(0.015, threshold * 0.85)
            elif _gs > 1.20:
                # Heavy damage: looser threshold reduces wasted retry cycles
                threshold = min(0.070, threshold * 1.15)

        _threshold_multiplier, _threshold_reason, _threshold_tcci, _threshold_material_family = (
            _compute_reconstruction_threshold_multiplier(phase_kwargs=phase_kwargs, phase_id=phase_id)
        )
        threshold = float(np.clip(threshold * _threshold_multiplier, 0.012, 0.070))

        # §9.7.3 Phasen-adaptive Sample-Dauer — MUSS vor scores_before bestimmt werden,
        # damit before und after dieselbe Sample-Länge nutzen (sonst falsche Regression).
        _sample_dur = _get_sample_duration(phase_id)

        # §9.1a Non-stationary: extract defect_locations for targeted sampling
        _defect_locs = (phase_kwargs or {}).get("defect_locations")

        # Vor-Scores messen (wenn nicht übergeben) — gleiche duration wie after-Messung
        sample_before = _extract_sample(
            audio, sr, duration_s=_sample_dur, defect_locations=_defect_locs, phase_id=phase_id
        )
        if scores_before is None:
            scores_before = _measure_quick(sample_before, sr)
        # §v10.17: 2s snippet for A/B comparison
        _snippet_n = min(len(sample_before), sr * 2)
        _snippet_start = (len(sample_before) - _snippet_n) // 2
        _ab_snippet = sample_before[_snippet_start : _snippet_start + _snippet_n]

        # Effective goal set: Schnitt aus FAST_GOALS_SUBSET + applicable_goals
        if applicable_goals is not None:
            effective_goals = [g for g in FAST_GOALS_SUBSET if g in applicable_goals]
            if not effective_goals:
                effective_goals = FAST_GOALS_SUBSET  # Fallback: alle
        else:
            effective_goals = FAST_GOALS_SUBSET

        # §9.7.4 Phase-specific goal exclusions (comb-filter-sensitive proxies).
        # Remove goals whose DSP proxy is unreliable for this particular phase type.
        _excluded_goals: set[str] = set()
        for _pfx, _excl in PHASE_GOAL_EXCLUSIONS.items():
            if phase_id.startswith(_pfx):
                _excluded_goals |= _excl
        # §2.31b Material-adaptive exclusion relaxation (v10.0.0):
        # High-quality digital sources (cd_digital, dat) have no broadband hiss.
        # Noise-derived false-regression root-causes (brillanz/authentizitaet/
        # transparenz/tonal_center) do not apply to these materials.
        # Only CREPE-load-state (natuerlichkeit) and transient-shape mismatch
        # (artikulation) remain as stable, material-independent exclusions.
        if _excluded_goals:
            _mat_kw = (phase_kwargs or {}).get("material_type") or (phase_kwargs or {}).get("material")
            _mat_str = (_mat_kw.value if hasattr(_mat_kw, "value") else str(_mat_kw)) if _mat_kw else ""
            if _mat_str in {"cd_digital", "dat"} and (
                phase_id.startswith("phase_03") or phase_id.startswith("phase_29")
            ):
                _excluded_goals &= {"natuerlichkeit", "artikulation"}
            # Analog-noise adaptive extension (2026-03-30): phase_03 on hiss-heavy
            # analog media can produce false timbre_authentizitaet regressions in the
            # short PMGG window although denoise improves perceptual quality.
            # Extended to phase_29 (2026-03-30): DeepFilterNet tape-hiss removal has
            # identical HF-removal → centroid-CV-disturbance mechanism as phase_03.
            # Both phases alter spectral-centroid variance on analog material where
            # hiss dominates HF → timbre proxy overreacts → false P2 cascade.
            if _mat_str in {"vinyl", "shellac", "tape", "reel_tape", "cassette"} and (
                phase_id.startswith("phase_03") or phase_id.startswith("phase_29")
            ):
                _excluded_goals.add("timbre_authentizitaet")

        # §2.54 Team-Koordination: Folgephase berücksichtigt Vorphasen-Kontext.
        # Verhindert, dass PMGG Retry-Logik bewusst wiederhergestellte HF-Anteile
        # als Regression wertet (phase_50 nach phase_06/phase_07/phase_23).
        _team_policy = _resolve_team_context_policy(phase_id, phase_kwargs)
        _team_goal_exclusions = _team_policy.get("goal_exclusions")
        if isinstance(_team_goal_exclusions, set) and _team_goal_exclusions:
            _excluded_goals |= _team_goal_exclusions

        _recheck_goals = _reconstruction_goal_recheck_allowlist(
            phase_id,
            phase_kwargs,
            _defect_locs,
            len(audio),
            sr,
            _sample_dur,
        )
        if _recheck_goals:
            _excluded_goals -= _recheck_goals

        if _excluded_goals:
            _effective_goals_before_exclusion = list(effective_goals)
            effective_goals = [g for g in effective_goals if g not in _excluded_goals]
            if not effective_goals:
                # Wenn Exclusions die komplette Zielmenge leeren, behalten wir die
                # ursprünglichen expliziten Ziele bei (statt auf alle 15 zu springen),
                # damit Recovery-/Tolerance-Pfade deterministisch bleiben.
                effective_goals = _effective_goals_before_exclusion
            logger.debug(
                "PMGG: %s goal exclusions angewendet: %s → %d goals checked",
                phase_id,
                sorted(_excluded_goals),
                len(effective_goals),
            )

        _team_threshold_mult = _team_policy.get("threshold_multiplier", 1.0)
        if isinstance(_team_threshold_mult, (int, float)) and float(_team_threshold_mult) > 1.0:
            _old_threshold = threshold
            threshold = min(0.090, float(threshold) * float(_team_threshold_mult))
            logger.debug(
                "PMGG team-policy: %s Schwelle %.3f -> %.3f (reason=%s)",
                phase_id,
                _old_threshold,
                threshold,
                _team_policy.get("reason", "unknown"),
            )

        # Phase ausführen + Regression prüfen (§2.29: initial_strength statt immer 1.0)
        self._last_retry_budget_policy = {}
        audio_out, scores_after, action, strength = self._run_with_retry(
            phase,
            audio,
            sr,
            scores_before,
            phase_id,
            phase_kwargs,
            threshold=threshold,
            effective_goals=effective_goals,
            sample_duration_s=_sample_dur,
            initial_strength=max(0.0, min(1.0, initial_strength)),
            defect_locations=_defect_locs,
            is_studio_2026=is_studio_2026,
            goal_weights=goal_weights,
            adaptive_goal_thresholds=adaptive_goal_thresholds,
            restorability_score=restorability_score,
        )

        # Best-Effort-Zähler: Phase wurde mit bestmöglicher Stärke angewendet,
        # nicht auf Vor-Phasen-Audio zurückgerollt. Darf daher den echten
        # Rollback-Zähler nicht erhöhen, sonst interpretiert UV3 Schutz-Telemetrie
        # als akustische Rollback-Kaskade.
        if action.startswith("best_effort"):
            self._best_effort_count += 1
            if self._best_effort_count > 3 and not self._user_warned:
                self._user_warned = True
                logger.warning(
                    "ℹ️ Einige Verarbeitungsschritte wurden mit reduzierter Stärke angewendet, um den Klang zu schützen."
                )
            # §v10.5 Guard Effectiveness Auditor: Paralysis-Event registrieren
            try:
                from backend.core.guard_effectiveness_auditor import get_effectiveness_auditor as _get_ga

                _ga = _get_ga()
                _ga.track_phase_decision(
                    phase_id=phase_id,
                    initial_strength=initial_strength,
                    final_strength=strength,
                    retries_exhausted=5 if action.startswith("best_effort") else 0,
                    pmgg_action=action,
                )
            except Exception as _gae:
                logger.debug("Guard-Auditor nicht verfügbar: %s", _gae)

        goal_regressions = {
            g: scores_after.get(g, 0.5) - scores_before.get(g, 0.5)
            for g in effective_goals
            if scores_after.get(g, 0.5) - scores_before.get(g, 0.5) < -threshold
        }

        log_entry = PhaseGateLogEntry(
            phase_id=phase_id,
            action=action,
            goal_regressions=goal_regressions,
            strength_used=strength,
        )
        _decision_class, _decision_reason = self._classify_action_decision(action)
        log_entry.metadata["pmgg_decision_class"] = _decision_class
        log_entry.metadata["pmgg_decision_reason"] = _decision_reason
        # §v10.18: resolved_defects aus PhaseResult in Log-Entry durchreichen
        _resolved = getattr(self, "_last_resolved_defects", None) or {}
        if _resolved:
            log_entry.metadata["resolved_defects"] = dict(_resolved)
        _recon_decision = (
            dict(self._last_reconstruction_localized_decision)
            if isinstance(self._last_reconstruction_localized_decision, dict)
            else {}
        )
        if _recon_decision:
            _recon_threshold_multiplier = float(_recon_decision.get("threshold_multiplier", 1.0))
            _recon_reason = str(_recon_decision.get("reason", ""))
            if _recon_threshold_multiplier <= 1.0:
                if bool(_recon_decision.get("accepted", False)):
                    _recon_threshold_multiplier = 1.02
                elif _recon_reason == "counterfactual_window_unavailable":
                    _recon_threshold_multiplier = 1.05
            log_entry.metadata["pmgg_reconstruction_localized"] = bool(_recon_decision.get("accepted", False))
            log_entry.metadata["pmgg_reconstruction_confidence"] = float(_recon_decision.get("confidence", 0.0))
            log_entry.metadata["pmgg_reconstruction_reason"] = str(_recon_decision.get("reason", ""))
            log_entry.metadata["pmgg_reconstruction_retry_budget_bias"] = int(
                _recon_decision.get("retry_budget_bias", 0)
            )
            log_entry.metadata["pmgg_reconstruction_retry_budget_bias_reason"] = str(
                _recon_decision.get("retry_budget_bias_reason", "")
            )
            log_entry.metadata["pmgg_reconstruction_transfer_chain_tcci"] = float(
                _recon_decision.get("transfer_chain_tcci", 0.0)
            )
            log_entry.metadata["pmgg_reconstruction_material_family"] = str(
                _recon_decision.get("material_family", "unknown")
            )
            log_entry.metadata["pmgg_reconstruction_threshold_multiplier"] = float(_recon_threshold_multiplier)
            log_entry.metadata["pmgg_reconstruction_threshold_multiplier_reason"] = str(
                _recon_decision.get("threshold_multiplier_reason", "")
            )
            log_entry.metadata["pmgg_reconstruction_threshold_multiplier_tcci"] = float(
                _recon_decision.get("threshold_multiplier_tcci", 0.0)
            )
            log_entry.metadata["pmgg_reconstruction_target_coverage"] = float(
                _recon_decision.get("target_coverage", 0.0)
            )
            log_entry.metadata["pmgg_reconstruction_control_coverage"] = float(
                _recon_decision.get("control_coverage", 0.0)
            )
            log_entry.metadata["pmgg_reconstruction_control_regression"] = float(
                _recon_decision.get("control_regression", 0.0)
            )
            log_entry.metadata["pmgg_reconstruction_epistemic_confidence"] = float(
                _recon_decision.get("epistemic_confidence", 0.0)
            )
            log_entry.metadata["pmgg_reconstruction_epistemic_reason"] = str(
                _recon_decision.get("epistemic_reason", "")
            )
            log_entry.metadata["pmgg_reconstruction_uncertainty_budget"] = float(
                _recon_decision.get("uncertainty_budget", 1.0)
            )
        # §0l Telemetrie: Team-Net-Delta für jede Phase aufzeichnen — diagnostiziert
        # ob Phasen das 15-Ziel-Team als Ganzes verbessern oder verschlechtern.
        if scores_before and scores_after and effective_goals:
            _log_team_net = sum(scores_after.get(g, 0.5) - scores_before.get(g, 0.5) for g in effective_goals) / max(
                len(effective_goals), 1
            )
            log_entry.metadata["pmgg_team_net_delta"] = round(float(_log_team_net), 4)
        # §DEBUG: Goal-Snapshots für PipelineTrace / aurik-debug — kein Overhead wenn nicht genutzt.
        log_entry.scores_before = dict(scores_before) if scores_before else {}
        log_entry.scores_after = dict(scores_after) if scores_after else {}

        _vocal_meta = _measure_vocal_guard_features(
            _extract_sample(
                audio_out,
                sr,
                duration_s=_sample_dur,
                defect_locations=_defect_locs,
                phase_id=phase_id,
            ),
            sr,
            reference=_extract_sample(
                audio,
                sr,
                duration_s=_sample_dur,
                defect_locations=_defect_locs,
                phase_id=phase_id,
            ),
        )
        log_entry.metadata["vocal_presence_proxy"] = round(float(_vocal_meta["vocal_presence_proxy"]), 4)
        log_entry.metadata["vocal_formant_stability"] = round(float(_vocal_meta["vocal_formant_stability"]), 4)
        log_entry.metadata["vocal_fricative_stability"] = round(float(_vocal_meta["vocal_fricative_stability"]), 4)
        log_entry.metadata["vocal_transient_integrity"] = round(float(_vocal_meta["vocal_transient_integrity"]), 4)
        log_entry.metadata["vocal_guard_active"] = bool(
            float(_vocal_meta["vocal_presence_proxy"]) >= _VOCAL_GUARD_TRIGGER
        )

        # §0c Recovery-Lite Transparency: best_effort actions mark recovery metadata
        # so downstream (UV3, bridge, export_workflow) can detect recovery status.
        if action.startswith("best_effort"):
            log_entry.metadata["recovery_attempted"] = True
            log_entry.metadata["best_possible_reached"] = True  # PMGG always returns best found
            log_entry.metadata["pmgg_best_effort_count"] = int(self._best_effort_count)
            log_entry.metadata["pmgg_real_rollback_count"] = int(self._rollback_count)

        if self._last_retry_budget_policy:
            log_entry.metadata["retry_budget_policy_active"] = bool(self._last_retry_budget_policy.get("active", False))
            log_entry.metadata["retry_budget_reason"] = str(self._last_retry_budget_policy.get("reason", "") or "")
            log_entry.metadata["retry_budget_max_retries"] = int(
                self._last_retry_budget_policy.get("max_retries_cap", 0) or 0
            )
            log_entry.metadata["retry_budget_seconds"] = float(
                self._last_retry_budget_policy.get("retry_budget_seconds", 0.0) or 0.0
            )
            if "future_priority_phases" in self._last_retry_budget_policy:
                log_entry.metadata["retry_budget_future_priority_phases"] = list(
                    self._last_retry_budget_policy.get("future_priority_phases", [])
                )
            if "history_samples" in self._last_retry_budget_policy:
                log_entry.metadata["retry_budget_history_samples"] = int(
                    self._last_retry_budget_policy.get("history_samples", 0) or 0
                )

        # §2.29e Team-Telemetrie: Policyinformationen in log_entry.metadata schreiben
        # damit UV3 nach der Pipeline team_coordination_events extrahieren kann.
        _te_reason = str(_team_policy.get("reason", "") or "")
        if _te_reason:
            log_entry.metadata["team_policy_reason"] = _te_reason
            log_entry.metadata["team_excluded_goals"] = sorted(
                _team_goal_exclusions if isinstance(_team_goal_exclusions, set) else set()
            )
            log_entry.metadata["team_threshold_mult"] = round(
                float(_team_policy.get("threshold_multiplier", 1.0)),
                3,
            )
            log_entry.metadata["team_strength_cap"] = round(float(_team_policy.get("strength_cap", 1.0)), 3)

        # §TFS: Temporal Fine Structure coherence check for spectral-modification phases.
        # Measures whether sub-1.5 kHz instantaneous phase (pitch/binaural cues)
        # survives the restoration phase. Moore (2008): TFS encodes pitch perception,
        # binaural localisation, and consonant texture — invisible to envelope metrics.
        if any(phase_id.startswith(pfx) for pfx in _TFS_SENSITIVE_PHASES):
            try:
                from backend.core.tfs_preservation_guard import get_tfs_preservation_guard

                _tfs_guard = get_tfs_preservation_guard()
                # Use same sample duration as Musical Goals (consistency)
                _tfs_sample_before = _extract_sample(audio, sr, duration_s=min(_sample_dur, 2.5))
                _tfs_sample_after = _extract_sample(audio_out, sr, duration_s=min(_sample_dur, 2.5))
                _tfs_result = _tfs_guard.measure(_tfs_sample_before, _tfs_sample_after, sr)

                log_entry.metadata["tfs_coherence"] = round(_tfs_result.mean_coherence, 4)
                log_entry.metadata["tfs_min_coherence"] = round(_tfs_result.min_coherence, 4)
                log_entry.metadata["tfs_n_bands"] = _tfs_result.n_bands
                log_entry.metadata["tfs_passes"] = _tfs_result.passes_threshold

                if not _tfs_result.passes_threshold:
                    logger.warning(
                        "PMGG TFS: %s TFS coherence degraded (mean=%.4f < %.2f) — "
                        "Verarbeitungsschritt may have disrupted temporal fine structure",
                        phase_id,
                        _tfs_result.mean_coherence,
                        _TFS_COHERENCE_THRESHOLD,
                    )
                else:
                    logger.info(
                        "PMGG TFS: %s coherence=%.4f (passes)",
                        phase_id,
                        _tfs_result.mean_coherence,
                    )
            except Exception as _tfs_exc:
                logger.debug("PMGG TFS: %s measurement fehlgeschlagen: %s", phase_id, _tfs_exc)

        elapsed = time.time() - t0
        logger.debug(
            "PMGG: %s → %s (%.0f ms, strength=%.2f)",
            phase_id,
            action,
            elapsed * 1000,
            strength,
        )

        # §9.7.14 Wärme-Validierung: log waerme delta at INFO level so real-run
        # AMRB field validation of the reverb-invariant warmth-ratio proxy is
        # visible without enabling full debug logging.
        if "waerme" in effective_goals:
            _w_before = scores_before.get("waerme", float("nan"))
            _w_after = scores_after.get("waerme", float("nan"))
            _w_before_nan = isinstance(_w_before, float) and math.isnan(_w_before)
            _w_after_nan = isinstance(_w_after, float) and math.isnan(_w_after)
            _w_delta = _w_after - _w_before if not (_w_before_nan or _w_after_nan) else float("nan")
            logger.info(
                "PMGG waerme §9.7.14  Verarbeitungsschritt=%s  before=%.4f  after=%.4f  delta=%+.4f  action=%s  strength=%.2f",
                phase_id,
                _w_before,
                _w_after,
                _w_delta if not (isinstance(_w_delta, float) and math.isnan(_w_delta)) else 0.0,
                action,
                strength,
            )

        # §2.47b: propagate sub_threshold marking into log_entry metadata
        if action == "sub_threshold":
            log_entry.metadata.setdefault("sub_threshold_phases", []).append(phase_id)

        # §v10.303.18 Artikulations-Wächter: Dynamics-Phasen (Compression,
        # Transient-Shaper) dürfen artikulation NIEMALS verschlechtern.
        _ART_CRITICAL_PHASES = {
            "phase_10_compression",
            "phase_11_limiting",
            "phase_26_dynamic_range_expansion",
            "phase_35_multiband_compression",
            "phase_54_transparent_dynamics",
            "phase_08_transient_preservation",
            "phase_36_transient_shaper",
        }
        if phase_id in _ART_CRITICAL_PHASES and "artikulation" in effective_goals:
            _a_before = scores_before.get("artikulation", 1.0)
            _a_after = scores_after.get("artikulation", 1.0)
            if not isinstance(_a_before, float) or math.isnan(_a_before):
                _a_before = 1.0
            if not isinstance(_a_after, float) or math.isnan(_a_after):
                _a_after = 1.0
            _a_delta = _a_after - _a_before
            if _a_delta < -0.02:
                logger.warning(
                    "§v10.303.18 Artikulations-Wächter: %s artikulation %.4f→%.4f (Δ=%.4f) → ROLLBACK",
                    phase_id,
                    _a_before,
                    _a_after,
                    _a_delta,
                )
                action = "rollback"

        return audio_out, scores_after, log_entry

    # ------------------------------------------------------------------
    # Interne Methoden
    # ------------------------------------------------------------------

    @staticmethod
    def _binary_search_strengths(
        initial: float,
        max_iters: int,
        precision: float,
    ) -> tuple[list[float], str]:
        """§v10.16: Binäre Intervallhalbierung (Hilfsfunktion).

        Erzeugt initiale Kandidaten-Liste. Der eigentliche Binary-Search-Loop
        läuft in _run_with_retry und passt lo/hi basierend auf Regression an.

        Returns:
            (strengths, detail_string)
        """
        strengths: list[float] = [float(initial)]
        return strengths, f"binary_search(max_iter={max_iters}, prec={precision:.3f})"

    def _run_with_retry(  # pylint: disable=too-many-positional-arguments
        self,
        phase: Any,
        audio: np.ndarray,
        sr: int,
        scores_before: dict[str, float],
        phase_id: str,
        phase_kwargs: dict[str, Any] | None = None,
        *,
        threshold: float = REGRESSION_THRESHOLD_GOOD,
        effective_goals: list | None = None,
        sample_duration_s: float = SAMPLE_DURATION_S,  # §9.7.3 phasen-adaptiv
        initial_strength: float = 1.0,
        defect_locations: dict[str, list[tuple[float, float]]] | None = None,
        is_studio_2026: bool = False,
        goal_weights: dict[str, float] | None = None,
        adaptive_goal_thresholds: dict[str, float] | None = None,
        restorability_score: float = 70.0,
    ) -> tuple[np.ndarray, dict[str, float], str, float]:
        """
        Führt Phase aus, ggf. mit Retry bei Regression.

        Args:
            threshold: Adaptiver REGRESSION_THRESHOLD (§2.29).
            effective_goals: Subset aus FAST_GOALS_SUBSET, das geprüft wird.
            sample_duration_s: Stichprobenlänge (§9.7.3 phasen-adaptiv, 1.0–5.0 s).
            initial_strength: Material-adaptive Initialstärke ∈ (0, 1.0] (§2.31).
                1.0 = Default. Retry-Stärken skalieren relativ dazu wenn < 1.0.
            goal_weights: §2.56 Song-specific goal importance weights.
                Per-goal float ∈ [0.3, 2.0]. weight > 1.0 = stricter threshold,
                weight < 1.0 = more lenient. None = uniform (1.0 for all).

        Returns:
            (audio_out, scores_after, action_label, strength_used)
        """
        if phase_kwargs is None:
            phase_kwargs = {}
        _threshold_multiplier = 1.0
        _threshold_reason = "default"
        _threshold_tcci = 0.0
        _epistemic_confidence = 0.0
        _epistemic_reason = "epistemic_unavailable"
        self._last_reconstruction_localized_decision = {}
        if effective_goals is None:
            effective_goals = FAST_GOALS_SUBSET
        # §2.55b Erwartete Kollateralschäden dieser Phase aus Regressions-Gate ausschließen.
        # Subtraktive Phasen erzeugen physikalisch erwartete Proxy-Absenkungen die KEINE
        # echten Qualitätsverluste sind — Rauschen/Hiss wurde als "Signal" gemessen.
        # _goals_for_regression: gefiltert für _max_regression/_max_regression_priority_aware.
        # effective_goals: vollständige Liste für _compute_team_net_delta (Teamwerk-Gesamtbild).
        _phase_collateral = PHASE_EXPECTED_COLLATERAL_GOALS.get(phase_id, frozenset())
        _goals_for_regression = [g for g in effective_goals if g not in _phase_collateral] or list(effective_goals)
        initial_strength = max(0.01, min(1.0, initial_strength))
        _team_policy = _resolve_team_context_policy(phase_id, phase_kwargs)
        _team_cap = _team_policy.get("strength_cap", 1.0)
        if isinstance(_team_cap, (int, float)) and float(_team_cap) < 0.999:
            _old_strength = initial_strength
            initial_strength = min(initial_strength, float(_team_cap))
            if initial_strength + 1e-9 < _old_strength:
                logger.debug(
                    "PMGG team-cap: %s strength %.2f -> %.2f (reason=%s)",
                    phase_id,
                    _old_strength,
                    initial_strength,
                    _team_policy.get("reason", "unknown"),
                )
        _safe_cap = _phase_safe_strength_cap(phase_id, phase_kwargs)
        if _safe_cap < 0.999:
            _old_strength = initial_strength
            initial_strength = min(initial_strength, _safe_cap)
            if initial_strength + 1e-9 < _old_strength:
                logger.info(
                    "PMGG safe-cap: %s material=%s strength %.2f -> %.2f",
                    phase_id,
                    _material_key_from_phase_kwargs(phase_kwargs),
                    _old_strength,
                    initial_strength,
                )

        # §PMGG-Restorative: Für defektentfernende Phasen (denoise, dereverb, hiss,
        # hum, noise gate, dropout) deckeln wir scores_before auf normative Mindest-
        # schwellwerte. Defekte erhöhen Metriken künstlich über den sauberen Wert:
        # Rauschen füllt Spektraltäler → Authentizität SCHEINT hoch. Nach Denoise
        # sinkt der Score auf den echten Wert → PMGG würde false-positive P1-
        # Regression melden und die Phase auf 6% Wet drosseln.
        # Lösung: Baseline kann nie höher sein als das normative Qualitätsziel.
        # Echter Schaden (Score nach Phase UNTER Schwelle) wird weiterhin erkannt.
        # §2.29c §2.48a Architektur-Inversion: Ist diese Phase restorative?
        # Ableitung aus phase_ontology (intrinsischer Typ), nicht aus Ausnahmeliste.
        # Legacy-Fallback: _RESTORATIVE_PHASES für Phasen noch nicht im Ontologie-Register.
        from backend.core.phase_ontology import BASELINE_CAPPING_VALID_TYPES, get_phase_type

        _phase_type = get_phase_type(phase_id)
        _is_restorative = _phase_type in BASELINE_CAPPING_VALID_TYPES or any(
            phase_id.startswith(p) for p in _RESTORATIVE_PHASES
        )
        _thresholds = _get_canonical_thresholds(is_studio_2026)
        # §09.2 Song-adaptive threshold blend: align per-phase PMGG with pipeline-end effective
        # thresholds so restorative baseline capping uses realistic song-specific floors.
        # §2.54 Adaptive blend weight: large downward delta (physical ceiling / genre constraint)
        # → use adaptive value directly; small delta → 60/40 conservative blend.
        # This prevents PMGG from demanding goals physically impossible for the material (e.g.
        # brillanz>0.70 for Shellac, bass_kraft>0.70 for Schlager) which causes endless retries
        # at 15 % strength and degrades overall restoration quality.
        if adaptive_goal_thresholds:
            _thresholds = dict(_thresholds)  # mutable copy — do not mutate module-level dict
            _blended_goals = []
            for _g, _v in adaptive_goal_thresholds.items():
                if _g in _thresholds:
                    _canon = float(_thresholds[_g])
                    _adap = float(_v)
                    _delta = _canon - _adap  # positive = adaptive is lower (constrained)
                    if _delta > 0.10:
                        # Large constraint (physical ceiling or strong genre/material bias):
                        # use adaptive value directly — blending would still exceed the ceiling.
                        _blended = float(np.clip(_adap, 0.30, 0.99))
                    elif _delta > 0.04:
                        # Moderate constraint: weight 40 % canonical / 60 % adaptive.
                        _blended = float(np.clip(0.40 * _canon + 0.60 * _adap, 0.30, 0.99))
                    else:
                        # Small or upward adjustment: conservative 60/40 blend (unchanged).
                        _blended = float(np.clip(0.60 * _canon + 0.40 * _adap, 0.30, 0.99))
                    if abs(_blended - _canon) > 1e-6:
                        _blended_goals.append(f"{_g}:{_canon:.2f}→{_blended:.2f}")
                    _thresholds[_g] = _blended
            if _blended_goals:
                logger.debug(
                    "PMGG §09.2 adaptive thresholds blended (%s): %s",
                    phase_id,
                    ", ".join(_blended_goals[:5]),
                )
        if _is_restorative:
            effective_scores_before = {g: min(v, _thresholds.get(g, v) + 0.05) for g, v in scores_before.items()}
            _capped_goals = [
                g for g in scores_before if scores_before[g] > _thresholds.get(g, scores_before[g]) + 0.001
            ]
            if _capped_goals:
                logger.debug(
                    "PMGG restorative baseline cap (%s): %s — defect-inflated scores capped at"
                    " adaptive thresholds to prevent false-positive regressions",
                    phase_id,
                    {g: round(scores_before[g], 3) for g in _capped_goals},
                )
        else:
            effective_scores_before = scores_before

        # §2.54 Goal-Gap Adaptive Strength Boost: jede Phase hinarbeiten auf die
        # individuell berechneten song-spezifischen Ziel-Schwellwerte.
        # Je größer das Defizit (Abstand Goal-Score → Zielwert), desto mehr Stärke
        # erhält die Phase — gedeckelt auf bestehende team_cap / safe_cap (advisory-only).
        # Nur wenn adaptive_goal_thresholds vorhanden (§09.2 estimate_song_goal_targets).
        # Wirkt auf initial_strength NACH allen Caps — darf Caps nicht überschreiten.
        if adaptive_goal_thresholds and effective_scores_before:
            _gap_vals: list[float] = []
            for _gg in effective_goals:
                if _gg in adaptive_goal_thresholds and _gg in effective_scores_before:
                    _t = float(adaptive_goal_thresholds[_gg])
                    _c = float(effective_scores_before[_gg])
                    _d = _t - _c
                    if _d > 0.005:  # nur substantielles Defizit berücksichtigen
                        _w = float((goal_weights or {}).get(_gg, 1.0))
                        _gap_vals.append(min(_d, 0.25) * _w)  # cap per-goal auf 0.25
            if _gap_vals:
                _mean_gap = sum(_gap_vals) / len(_gap_vals)
                # 0.05 Ø-Defizit → +8% Stärke; 0.15+ → +25% (Asymptote)
                _gap_factor = float(np.clip(1.0 + 1.666 * _mean_gap, 1.0, 1.25))
                if _gap_factor > 1.02 and initial_strength < 0.99:
                    _cap_upper = min(float(_team_cap), float(_safe_cap), 1.0)
                    _old_is = initial_strength
                    initial_strength = float(np.clip(initial_strength * _gap_factor, 0.0, _cap_upper))
                    if abs(initial_strength - _old_is) > 0.005:
                        logger.debug(
                            "PMGG §2.54 goal-gap boost (%s): mean_gap=%.3f factor=%.3f "
                            "strength %.3f→%.3f (cap_upper=%.2f)",
                            phase_id,
                            _mean_gap,
                            _gap_factor,
                            _old_is,
                            initial_strength,
                            _cap_upper,
                        )

        # §2.29a ML-Inference-Caching: ML-deterministic Phasen werden nur
        # einmal mit strength=1.0 ausgeführt.  Retries variieren Wet/Dry-Blending.
        # Strength-abhängige DSP-Phasen müssen bei jedem Retry neu ausgeführt
        # werden, da strength dort Algorithmus-Parameter steuert (z.B. Filterfrequenz,
        # Kompressionsratio), nicht nur das Mischverhältnis.
        _is_ml_deterministic = phase_id.startswith(tuple(_ML_DETERMINISTIC_PHASES))
        # §2.29a Sonderfall phase_20: SGMSE+ (ML) ist deterministic, aber WPE-DSP-Fallback
        # verwendet strength*0.90 als algorithmus-internen Prädiktor-Parameter → must re-run.
        # Zur Laufzeit: nur wenn SGMSE+ im ML-Budget alloziert ist, ML-Pfad verwenden.
        if _is_ml_deterministic and phase_id.startswith("phase_20"):
            _is_ml_deterministic = _phase20_is_ml_active()

        # §9.7.5 Referenz-Stichprobe für preservation-aware Messung.
        # Einmal berechnen, für alle scores_after/scores_retry wiederverwenden.
        _defect_locs = defect_locations
        _ref_sample = _extract_sample(
            audio, sr, duration_s=sample_duration_s, defect_locations=_defect_locs, phase_id=phase_id
        )

        if _is_ml_deterministic:
            # ML-Pfad: Einmalige Inferenz mit strength=1.0, Wet/Dry für Stärke
            audio_full = self._run_phase(phase, audio, 1.0, phase_kwargs)
            if initial_strength < 1.0:
                audio_out = self._wet_dry_blend(audio, audio_full, initial_strength, phase)
            else:
                audio_out = audio_full
        else:
            # DSP-Pfad: Direkte Ausführung mit material-adaptiver Stärke
            audio_out = self._run_phase(phase, audio, initial_strength, phase_kwargs)
            audio_full = None  # kein Cache benötigt

        # §2.45/§2.54 Passthrough-Erkennung: Phasen die kein Pitch/Defekt finden geben
        # das Audio bit-identisch zurück (z.B. phase_31 bei CREPE confidence=0.0).
        # In diesem Fall: kein Goal-Scoring, kein Retry, kein StrictConflictDecay.
        # np.array_equal ist exakt + schnell (kein float-Toleranz-Drift).
        if np.array_equal(audio, audio_out):
            logger.debug(
                "PMGG %s: audio_out identisch mit Eingabe (passthrough) — direkt passed, kein Wiederholung",
                phase_id,
            )
            return audio_out, scores_before, "passed", initial_strength

        scores_after = _measure_quick(
            _extract_sample(
                audio_out, sr, duration_s=sample_duration_s, defect_locations=_defect_locs, phase_id=phase_id
            ),
            sr,
            reference=_ref_sample,
        )

        regression = self._max_regression(
            effective_scores_before, scores_after, _goals_for_regression, goal_weights=goal_weights
        )
        # §v10.6 RESTAURIER-DENKER: Zentrale Entscheidungs-Intelligenz
        try:
            from denker.restaurier_denker import (
                DenkerContext,
                get_restaurier_denker,
            )

            _rd = get_restaurier_denker()
            _ctx = DenkerContext(
                phase_id=phase_id,
                mode="studio_2026" if is_studio_2026 else "restoration",
                restorability=restorability_score,
                initial_strength=initial_strength,
                current_strength=1.0,
                retry_count=0,
                best_effort_count=self._best_effort_count,
                total_phases_run=self._phase_count if hasattr(self, "_phase_count") else 0,
                scores_before=effective_scores_before,
                scores_after=scores_after,
                effective_goals=_goals_for_regression,
                regression=regression,
                audio_before=audio,
                audio_after=audio_out,
                sr=sr,
            )
            _decision = _rd.decide(_ctx)  # type: ignore[attr-defined]
            if _decision.verdict.value in ("override_guard",):
                regression = max(0.0, regression * 0.3)
                logger.info("§v10.6 Denker: Guard-Override %s — %s", phase_id, _decision.reason)
            if _decision.undo_detected:
                regression = max(regression, threshold + 0.002)
                logger.warning("§v10.6 Denker: UNDO in %s — %s", phase_id, _decision.reason)
            if _decision.paralysis_detected:
                logger.warning("§v10.6 Denker: Paralysis %s — %s", phase_id, _decision.reason)
        except Exception as _rd_exc:
            logger.debug("RestaurierDenker nicht verfuegbar: %s", _rd_exc)
        _skip_corr = phase_id in _TIMING_CORR_EXCLUDE
        _skip_drop = phase_id in _LF_SUBTRACTIVE_DROP_SKIP
        _ci_penalty, _ci_meta = _content_integrity_penalty(
            audio, audio_out, skip_corr_check=_skip_corr, skip_drop_check=_skip_drop
        )
        if _ci_penalty > 0.0:
            # Force retry path for catastrophic content loss even when many goals are excluded.
            regression = max(regression, threshold + 0.001 + 0.05 * _ci_penalty)
            logger.warning(
                "PMGG Content-Guard: %s triggered (rms_drop=%.2f dB corr=%.3f penalty=%.3f)",
                phase_id,
                _ci_meta.get("rms_drop_db", 0.0),
                _ci_meta.get("corr", 1.0),
                _ci_penalty,
            )

        # §2.47b JND Sub-Threshold Check: if all applicable goal-deltas are ≥ 0 and < JND
        # → phase produces no perceptually detectable improvement → accept but mark sub_threshold
        # VERBOTEN: sub-threshold logic must NOT fire when _ci_penalty > 0 (content loss)
        if _ci_penalty == 0.0:
            _applicable_jnd = [g for g in effective_goals if g in effective_scores_before and g in scores_after]
            if _applicable_jnd:
                _deltas = {g: scores_after[g] - effective_scores_before[g] for g in _applicable_jnd}

                # §C6 Psychoacoustic Masking Budget: elevate JND for goals whose spectral
                # region is acoustically masked by dominant loud frequency content.
                # Uses Bark-band energy profile from §4.1b psychoacoustics module.
                _effective_jnd: dict[str, float] = {}
                try:
                    from backend.core.dsp.psychoacoustics import compute_bark_energy_profile as _cbep

                    _audio_for_bark = audio if audio.ndim == 1 else np.mean(audio, axis=0)
                    _bark_profile = _cbep(_audio_for_bark, sr)
                    _total_bark = float(np.sum(_bark_profile)) if _bark_profile is not None else 0.0

                    # Masking factors per spectral region (conservative, advisory-only)
                    def _masking_factor(goal: str) -> float:
                        if _bark_profile is None or _total_bark < 1e-10:
                            return 1.0
                        # Map goal centroid to Bark bands: low/mid/high
                        _lf_ratio = float(np.sum(_bark_profile[:6])) / _total_bark  # Barks 0-6 ≈ <500 Hz
                        _mf_ratio = float(np.sum(_bark_profile[6:15])) / _total_bark  # Barks 6-15 ≈ 500-4 kHz
                        _hf_ratio = float(np.sum(_bark_profile[15:])) / _total_bark  # Barks 15+ ≈ >4 kHz
                        if goal in ("bass_kraft", "waerme"):
                            return 1.0 + 0.50 * _lf_ratio  # LF masking boosts bass-goal JND
                        if goal in ("artikulation", "transparenz", "separation_fidelity"):
                            return 1.0 + 0.40 * _mf_ratio  # Midrange masking
                        if goal in ("brillanz", "timbre_authentizitaet"):
                            return 1.0 + 0.35 * _hf_ratio  # HF masking
                        return 1.0

                    for g in _applicable_jnd:
                        _base_jnd = JND_MIN_DELTA.get(g, 0.015)
                        _effective_jnd[g] = _base_jnd * _masking_factor(g)
                except Exception as _c6_exc:
                    logger.debug("§C6 Masking JND uebersprungen (nicht blockierend): %s", _c6_exc)
                    _effective_jnd = {g: JND_MIN_DELTA.get(g, 0.015) for g in _applicable_jnd}

                _all_below_jnd = all(d >= 0.0 for d in _deltas.values()) and all(
                    abs(d) < _effective_jnd.get(g, JND_MIN_DELTA.get(g, 0.015)) for g, d in _deltas.items()
                )
                if _all_below_jnd:
                    logger.debug(
                        "PMGG %s: sub_Schwelle — all %d goal-deltas ≥ 0 and < JND (masking-angepasst), accepting",
                        phase_id,
                        len(_applicable_jnd),
                    )
                    return audio_out, scores_after, "sub_threshold", initial_strength

        if regression <= threshold:
            return audio_out, scores_after, "passed", initial_strength

        _is_reconstruction_counterfactual = phase_id.startswith(_RECONSTRUCTION_COUNTERFACTUAL_PHASE_PREFIXES)
        _control_window = _get_reconstruction_control_window_bounds(
            len(audio),
            sr,
            sample_duration_s,
            _defect_locs,
            phase_id,
        )
        if _is_reconstruction_counterfactual and _ci_penalty > 0.0:
            self._last_reconstruction_localized_decision = {
                "phase_id": phase_id,
                "accepted": False,
                "confidence": 0.0,
                "reason": "content_integrity_guard_active",
                "target_coverage": 0.0,
                "control_coverage": 0.0,
                "control_regression": 0.0,
                "threshold": float(threshold),
                "retry_budget_bias": -1,
                "retry_budget_bias_reason": "content_integrity_guard_active",
                "transfer_chain_tcci": 0.0,
                "material_family": _material_key_from_phase_kwargs(phase_kwargs),
                "threshold_multiplier": float(_threshold_multiplier),
                "threshold_multiplier_reason": str(_threshold_reason),
                "threshold_multiplier_tcci": float(_threshold_tcci),
                "epistemic_confidence": 0.0,
                "epistemic_reason": "content_integrity_guard_active",
                "uncertainty_budget": 1.0,
            }
        elif _ci_penalty == 0.0 and _control_window is not None:
            control_start, control_end, target_coverage, control_coverage = _control_window
            control_before = audio[control_start:control_end]
            control_after = audio_out[control_start:control_end]
            control_scores_before = _measure_quick(control_before, sr)
            control_scores_after = _measure_quick(control_after, sr, reference=control_before)
            control_regression = self._max_regression(
                control_scores_before,
                control_scores_after,
                _goals_for_regression,
                goal_weights=goal_weights,
            )
            _localized_accept, _localized_conf, _localized_reason = _assess_reconstruction_localized_confidence(
                target_coverage=target_coverage,
                control_coverage=control_coverage,
                control_regression=control_regression,
                threshold=threshold,
            )
            _epistemic_confidence, _epistemic_reason = _compute_reconstruction_epistemic_confidence(
                localized_confidence=float(_localized_conf),
                target_coverage=float(target_coverage),
                control_coverage=float(control_coverage),
                transfer_chain_tcci=float(_threshold_tcci),
                threshold_multiplier=float(_threshold_multiplier),
                phase_kwargs=phase_kwargs,
            )
            _decision_confidence = float(
                np.clip(0.65 * float(_localized_conf) + 0.35 * _epistemic_confidence, 0.0, 1.0)
            )
            _localized_retry_bias, _localized_bias_reason, _localized_tcci, _localized_material_family = (
                _compute_reconstruction_retry_budget_bias(
                    accepted=bool(_localized_accept),
                    confidence=float(_decision_confidence),
                    phase_kwargs=phase_kwargs,
                    phase_id=phase_id,
                )
            )
            self._last_reconstruction_localized_decision = {
                "phase_id": phase_id,
                "accepted": bool(_localized_accept),
                "confidence": float(_localized_conf),
                "reason": str(_localized_reason),
                "target_coverage": float(target_coverage),
                "control_coverage": float(control_coverage),
                "control_regression": float(control_regression),
                "threshold": float(threshold),
                "retry_budget_bias": int(_localized_retry_bias),
                "retry_budget_bias_reason": str(_localized_bias_reason),
                "transfer_chain_tcci": float(_localized_tcci),
                "material_family": str(_localized_material_family),
                "threshold_multiplier": float(_threshold_multiplier),
                "threshold_multiplier_reason": str(_threshold_reason),
                "threshold_multiplier_tcci": float(_threshold_tcci),
                "epistemic_confidence": float(_epistemic_confidence),
                "epistemic_reason": str(_epistemic_reason),
                "uncertainty_budget": float(np.clip(1.0 - _decision_confidence, 0.0, 1.0)),
            }
            if _localized_accept:
                logger.info(
                    "PMGG reconstruction collateral-Pruefung: %s localized regression tolerated "
                    "(conf=%.3f epistemic=%.3f reason=%s target_coverage=%.3f control_coverage=%.3f "
                    "control_regression=%.4f <= %.4f)",
                    phase_id,
                    _localized_conf,
                    _epistemic_confidence,
                    _localized_reason,
                    target_coverage,
                    control_coverage,
                    control_regression,
                    threshold,
                )
                return audio_out, scores_after, "passed_reconstruction_localized", initial_strength
            logger.info(
                "PMGG reconstruction collateral-Pruefung: %s localized accept rejected "
                "(conf=%.3f epistemic=%.3f reason=%s target_coverage=%.3f control_coverage=%.3f "
                "control_regression=%.4f Schwelle=%.4f)",
                phase_id,
                _localized_conf,
                _epistemic_confidence,
                _localized_reason,
                target_coverage,
                control_coverage,
                control_regression,
                threshold,
            )
        elif _is_reconstruction_counterfactual:
            self._last_reconstruction_localized_decision = {
                "phase_id": phase_id,
                "accepted": False,
                "confidence": 0.0,
                "reason": "counterfactual_window_unavailable",
                "target_coverage": 0.0,
                "control_coverage": 0.0,
                "control_regression": 0.0,
                "threshold": float(threshold),
                "retry_budget_bias": -1,
                "retry_budget_bias_reason": "counterfactual_window_unavailable",
                "transfer_chain_tcci": float(
                    np.clip(
                        compute_tcci(
                            [
                                str(v).strip().lower()
                                for v in ((phase_kwargs or {}).get("transfer_chain") or [])
                                if str(v).strip()
                            ]
                        ),
                        0.0,
                        1.0,
                    )
                ),
                "material_family": _material_key_from_phase_kwargs(phase_kwargs),
                "threshold_multiplier": float(_threshold_multiplier),
                "threshold_multiplier_reason": str(_threshold_reason),
                "threshold_multiplier_tcci": float(_threshold_tcci),
                "epistemic_confidence": 0.0,
                "epistemic_reason": "counterfactual_window_unavailable",
                "uncertainty_budget": 1.0,
            }

        # §2.29 v10.0.0: Priority-aware regression check.
        # Determine worst priority among regressed goals to set retry budget.
        _reg_pa, _worst_prio = self._max_regression_priority_aware(
            effective_scores_before, scores_after, _goals_for_regression, threshold, goal_weights=goal_weights
        )

        # Log which goal caused the regression (diagnostics for false-positive detection)
        _worst_goal = max(
            effective_goals,
            key=lambda g: max(0.0, effective_scores_before.get(g, 0.5) - scores_after.get(g, 0.5)),
        )
        if _ci_penalty > 0.0:
            _worst_prio = min(_worst_prio, 2)
            _worst_goal = "content_integrity_guard"
        logger.debug(
            "PMGG: %s regression=%.4f > Schwelle=%.3f — worst goal: %s (P%d, before=%.3f after=%.3f)",
            phase_id,
            regression,
            threshold,
            _worst_goal,
            _worst_prio,
            effective_scores_before.get(_worst_goal, 0.5),
            scores_after.get(_worst_goal, 0.5),
        )

        # §2.29 v10.0.0 / §0c Recovery-Lite: If priority-adjusted regression is fully
        # within tolerance (worst_prio==99 → no goal exceeded its priority-band threshold),
        # accept as tolerated without retry.
        # P4/P5 with regression ABOVE 2.0×/2.5× threshold → _worst_prio=4or5 → 1 Recovery-Retry.
        if _worst_prio > 5:
            logger.info(
                "PMGG: %s regression within priority tolerance band (worst_prio=%d, goal=%s) — tolerated",
                phase_id,
                _worst_prio,
                _worst_goal,
            )
            log_action = "passed_p4p5_tolerated"
            return audio_out, scores_after, log_action, initial_strength

        # §0l Teamwork-Invariante (§1.2c): Wenn das 15-Ziel-Team netto positiv ist und
        # ausschließlich P3/P4/P5-Ziele regressiert haben (P1/P2 über Pflichtboden),
        # Phase ohne Retry annehmen — Teamwork schlägt Dominanz.
        # Sicherheitsbedingungen:
        #   _worst_prio >= 3 → kein P1/P2-Ziel verletzt
        #   regression <= 2.5 * threshold → keine katastrophale Einzelregression
        #   _all_p1p2_above_floor → P1/P2 liegen nach der Phase noch über Pflichtschwelle
        if _worst_prio >= 3 and regression <= 2.5 * threshold:
            _net_delta, _all_p1p2_above_floor, _ = self._compute_team_net_delta(
                effective_scores_before,
                scores_after,
                effective_goals,
                goal_weights=goal_weights,
                canonical_thresholds=_thresholds,
            )
            if _net_delta > 0.0 and _all_p1p2_above_floor:
                logger.info(
                    "PMGG §0l Teamwork-Gate: %s → passed_team_balanced "
                    "(net_delta=+%.4f, worst_prio=P%d, regression=%.4f ≤ 2.5×Schwelle=%.4f)",
                    phase_id,
                    _net_delta,
                    _worst_prio,
                    regression,
                    2.5 * threshold,
                )
                return audio_out, scores_after, "passed_team_balanced", initial_strength

        # Priority-based max retries (§2.29 v10.0.0):
        _max_retries_for_prio = _PRIORITY_MAX_RETRIES.get(_worst_prio, 4)

        # §2.31a SongCal P3-Retry-Feinjustage: restorability_tier moduliert den
        # Retry-Etat für P3-Ziele (Groove, MicroDynamics, Emotionalität).
        # Good-Material: 3 → 4 Retries (stabil genug, um Verbesserung rauszuholen).
        # Poor-Material:  3 → 2 Retry  (P3-Regressionen oft unabwendbar — Zeit sparen).
        # P1/P2/P4/P5 bleiben unverändert.
        if _worst_prio == 3:
            _cal_p3 = (phase_kwargs or {}).get("song_calibration_profile", {})
            if isinstance(_cal_p3, dict) and _cal_p3:
                _rtier = _cal_p3.get("restorability_tier", "fair")
                if _rtier == "good":
                    _max_retries_for_prio = min(4, _max_retries_for_prio + 1)  # 3 → 4
                elif _rtier == "poor":
                    _max_retries_for_prio = max(2, _max_retries_for_prio - 1)  # 3 → 2

        _localized_retry_bias = int(
            (self._last_reconstruction_localized_decision or {}).get("retry_budget_bias", 0)
            if isinstance(self._last_reconstruction_localized_decision, dict)
            else 0
        )
        if _localized_retry_bias != 0:
            _max_retries_for_prio = int(np.clip(_max_retries_for_prio + _localized_retry_bias, 1, 5))
            logger.debug(
                "PMGG: %s localized Wiederholung bias=%d angewendet → max_retries=%d",
                phase_id,
                _localized_retry_bias,
                _max_retries_for_prio,
            )

        # §v10.16: Binäre Suche nach optimaler Stärke (Skalpell statt Vorschlaghammer).
        # Statt 5 grober Retry-Stufen (0.65→0.50→...) findet Intervallhalbierung
        # die MAXIMALE Stärke ohne Regression mit ±1.5% Präzision.
        _RETRY_BUDGET_S = 300.0
        _max_retries_for_prio, _RETRY_BUDGET_S, self._last_retry_budget_policy = self._resolve_retry_budget_policy(
            phase_kwargs,
            max_retries=_max_retries_for_prio,
            retry_budget_s=_RETRY_BUDGET_S,
        )
        _USE_BINARY = _max_retries_for_prio >= 3  # Binärsuche nur bei ≥3 Retries

        # §2.29 Best-Effort-Tracking: Speichere den Versuch mit geringster Regression.
        # PMGG darf Phasen NICHT überspringen — CausalDefectReasoner hat die Phase
        # als notwendig bestimmt. Stattdessen wird der beste Versuch verwendet.
        best_audio = audio_out
        best_scores = scores_after
        best_regression = regression
        best_strength = initial_strength
        best_action = "best_effort"

        # §2.29: Katastrophaler Content-Verlust darf NIE in einen Skip münden.
        # CausalDefectReasoner hat die Phase als notwendig bestimmt — der beste
        # Versuch wird verwendet, statt den Verarbeitungsschritt zu verwerfen.
        try:
            _final_pen_29, _final_pen_meta_29 = _content_integrity_penalty(
                np.asarray(audio, dtype=np.float32),
                np.asarray(best_audio, dtype=np.float32),
                skip_corr_check=bool(phase_id in _TIMING_CORR_EXCLUDE),
            )
        except Exception:
            _final_pen_29 = 0.0
            _final_pen_meta_29 = {}
        if float(_final_pen_29) >= 0.90:
            logger.warning(
                "PMGG Content-Guard: %s final penalty=%.2f (rms_drop=%.1f dB corr=%.3f) → best_effort statt Skip (§2.29)",
                phase_id,
                float(_final_pen_29),
                float(_final_pen_meta_29.get("rms_drop_db", 0.0) or 0.0),
                float(_final_pen_meta_29.get("corr", 0.0) or 0.0),
            )
            return best_audio, best_scores, "best_effort", best_strength

        # §v10 HPE-GATE: Nicht binär skip, sondern ultra-low strength versuchen.
        # Manchmal ist weniger besser als gar nichts.
        try:
            from backend.core.human_pleasantness_estimator import compare_pleasantness

            _hpe_cmp = compare_pleasantness(
                np.asarray(audio, dtype=np.float32), np.asarray(best_audio, dtype=np.float32), 48000
            )
            _hpe_delta = float(_hpe_cmp.get("delta_score", 0.0))

            if _hpe_delta < -0.03:
                logger.warning(
                    "§v10 HPE-GATE: Verarbeitungsschritt %s HPE %+.3f < -0.03 — Verarbeitungsschritt verworfen, Pre-Verarbeitungsschritt-Audio wiederhergestellt.",
                    phase_id,
                    _hpe_delta,
                )
                return audio, effective_scores_before, "hpe_skip", 0.0

            if -0.03 <= _hpe_delta < 0.0:
                _ultra_strength = max(0.03, best_strength * 0.30)
                logger.info(
                    "§v10 HPE-GATE: Verarbeitungsschritt %s HPE %+.3f im neutralen Bereich — "
                    "akzeptiert mit ultra-reduzierter Stärke %.2f.",
                    phase_id,
                    _hpe_delta,
                    _ultra_strength,
                )
                return best_audio, best_scores, "hpe_ultra_low", _ultra_strength
        except Exception as e:
            logger.warning("per_Verarbeitungsschritt_musical_goals_gate.py::unbekannter Ersatzpfad: %s", e)

        # §v10.16 Binary-Search State
        _bs_lo = 0.0
        _bs_hi = float(initial_strength)
        _bs_last_passed = False

        # §0l Team-Net-Delta-Tracking
        _best_team_net = sum(
            scores_after.get(g, 0.5) - effective_scores_before.get(g, 0.5) for g in effective_goals
        ) / max(len(effective_goals), 1)
        _prev_team_net = _best_team_net

        # §2.29a Fix: ML-deterministische Timing-Phasen (phase_12, phase_31)
        # können NICHT per Wet/Dry retried werden, da Timing-Phasen kein Blending
        # erlauben (Phasen-Artefakte bei Crossfade zeitversetzter Signale).
        # Alle Retries würden identisches Audio produzieren → sofort Best-Effort.
        _TIMING_PHASES = frozenset(
            {
                "phase_12_wow_flutter_fix",
                "phase_31_speed_pitch_correction",
            }
        )
        if _is_ml_deterministic and phase_id in _TIMING_PHASES:
            logger.info(
                "PMGG: %s is ML-deterministic timing Verarbeitungsschritt — Wet/Dry retries not applicable, "
                "using best-effort (regression=%.4f > Schwelle=%.3f)",
                phase_id,
                regression,
                threshold,
            )
            return best_audio, best_scores, "best_effort", initial_strength

        # Retry-Schleife
        # ML-deterministische Phasen: Wet/Dry-Reblend des gecachten audio_full
        #   (spart ~60 s pro Retry bei OMLSA + ResembleEnhance etc.)
        # DSP-Phasen: Erneuter process()-Aufruf mit geändertem strength
        #   (nichtlineare DSP-Operationen: wet/dry ≠ Neuberechnung)
        _prev_regression = regression
        _retry_t0 = time.time()
        # ── §v10.16 Binary Search Loop ────────────────────────────────
        # §F821-Fix: retry_strengths was referenced but never defined
        retry_strengths = [round(float(initial_strength) * s, 3) for s in (0.75, 0.50, 0.30, 0.15)]
        _binary_active = _USE_BINARY and len(retry_strengths) > 0
        _bs_attempt = 0
        _bs_max = _BINARY_SEARCH_MAX_ITERS if _binary_active else len(retry_strengths)

        for attempt in range(_bs_max):
            if _binary_active:
                # Binäre Suche: nächste Stärke = Intervallmitte
                if _bs_attempt == 0:
                    strength = float(initial_strength)
                else:
                    strength = (_bs_lo + _bs_hi) / 2.0
                _bs_attempt += 1
                if _bs_hi - _bs_lo < _BINARY_SEARCH_PRECISION:
                    break
            else:
                if attempt >= len(retry_strengths):
                    break
                strength = retry_strengths[attempt]
            _retry_elapsed = time.time() - _retry_t0
            if _retry_elapsed > _RETRY_BUDGET_S:
                logger.info(
                    "PMGG: %s Wiederholung time Grenze exceeded (%.0fs > %.0fs) — "
                    "using best Versuch so far (regression=%.4f, Versuch=%d)",
                    phase_id,
                    _retry_elapsed,
                    _RETRY_BUDGET_S,
                    best_regression,
                    attempt,
                )
                break

            import gc

            gc.collect()

            action_label = f"retry{attempt + 1}"

            if _is_ml_deterministic:
                # §2.29a: Wet/Dry-Reblend — keine erneute ML-Inferenz
                logger.debug(
                    "PMGG: %s Wiederholung %d mit strength=%.2f (Wet/Dry-Reblend, keine Re-Inferenz)",
                    phase_id,
                    attempt + 1,
                    strength,
                )
                audio_retry = self._wet_dry_blend(
                    audio, audio_full if audio_full is not None else audio, strength, phase
                )
            else:
                # DSP-Phase: Neu ausführen mit reduziertem strength
                logger.debug(
                    "PMGG: %s Wiederholung %d mit strength=%.2f (DSP Re-Ausfuehrung)",
                    phase_id,
                    attempt + 1,
                    strength,
                )
                audio_retry = self._run_phase(phase, audio, strength, phase_kwargs)

            _retry_sample = _extract_sample(
                audio_retry,
                sr,
                duration_s=sample_duration_s,
                defect_locations=_defect_locs,
                phase_id=phase_id,
            )
            scores_retry = _measure_quick(_retry_sample, sr, reference=_ref_sample, precise_override=False)
            regression_retry = self._max_regression(
                effective_scores_before, scores_retry, _goals_for_regression, goal_weights=goal_weights
            )
            _ci_penalty_retry, _ci_meta_retry = _content_integrity_penalty(audio, audio_retry)
            if _ci_penalty_retry > 0.0:
                regression_retry = max(regression_retry, threshold + 0.001 + 0.05 * _ci_penalty_retry)
                logger.debug(
                    "PMGG Content-Guard Wiederholung: %s r%d (rms_drop=%.2f dB corr=%.3f penalty=%.3f)",
                    phase_id,
                    attempt + 1,
                    _ci_meta_retry.get("rms_drop_db", 0.0),
                    _ci_meta_retry.get("corr", 1.0),
                    _ci_penalty_retry,
                )
            if regression_retry <= threshold:
                # Apply precise overrides once for accurate score propagation to next phase
                scores_retry = _apply_precise_metric_overrides(scores_retry, _retry_sample, sr, reference=_ref_sample)
                return audio_retry, scores_retry, action_label, strength
            # §0l Track best attempt: prefer lowest regression; bei ähnlicher Regression
            # denjenigen Versuch bevorzugen der das gesamte 15-Ziel-Team besser stellt.
            _net_retry = sum(
                scores_retry.get(g, 0.5) - effective_scores_before.get(g, 0.5) for g in effective_goals
            ) / max(len(effective_goals), 1)
            _is_better_attempt = regression_retry < best_regression or (
                abs(regression_retry - best_regression) < threshold * 0.15 and _net_retry > _best_team_net
            )
            if _is_better_attempt:
                best_audio = audio_retry
                best_scores = scores_retry
                best_regression = regression_retry
                best_strength = strength
                best_action = f"best_effort_r{attempt + 1}"
                _best_team_net = _net_retry

            # Stagnation guard: if regression barely changes across consecutive
            # retries despite strength variation, further retries are wasted.
            # §2.31a: Stagnation-Delta proportional zum Threshold — armes Material
            # (höherer Threshold) bricht früher ab; gutes Material (niedriger Threshold)
            # ist geduldiger (wartet auf kleinere Verbesserungen).
            # §0l: Stagnation nur wenn BEIDE — max-Regression UND Team-Net — stagnieren.
            # Wenn das Team sich noch verbessert, weitere Retries zulassen.
            _stagnation_delta = max(0.002, threshold * 0.15)
            _team_still_improving = (_net_retry - _prev_team_net) > threshold * 0.05
            if (
                abs(regression_retry - _prev_regression) < _stagnation_delta
                and attempt >= 1
                and not _team_still_improving
            ):
                logger.info(
                    "PMGG: %s stagnation erkannt at Wiederholung %d "
                    "(Δregression=%.6f, Δteam=%.6f) — skipping remaining retries",
                    phase_id,
                    attempt + 1,
                    abs(regression_retry - _prev_regression),
                    _net_retry - _prev_team_net,
                )
                break
            _prev_regression = regression_retry
            _prev_team_net = _net_retry

        # §2.31b Dynamic catastrophic threshold (v10.0.0):
        # Proportional to adaptive threshold so Good material (0.020) triggers
        # emergency retries at 0.08 — earlier quality protection. Poor material
        # (0.055) produces 0.22, matching the old hard-coded value.
        # Floor 0.08 prevents über-aggressive cascades on very clean material.
        _CATASTROPHIC_THRESHOLD = max(0.08, 4.0 * threshold)
        _team_thr_mult = _team_policy.get("threshold_multiplier", 1.0)
        if isinstance(_team_thr_mult, (int, float)) and float(_team_thr_mult) > 1.0:
            _CATASTROPHIC_THRESHOLD = min(0.25, _CATASTROPHIC_THRESHOLD * float(_team_thr_mult))

        # §v10.210 Closed-Loop: Emergency-Retries INCLUDING increased strengths.
        # Wenn die Defect-to-Audibility-Engine sagt, der Defekt sei noch hörbar,
        # wird auch mit ERHÖHTER Stärke wiederholt (nicht nur reduziert).
        _EMERGENCY_STRENGTHS = [0.15 * initial_strength, 0.10 * initial_strength]
        _audibility_boost = float((phase_kwargs or {}).get("audibility_strength", 0.0) or 0.0)
        if _audibility_boost > initial_strength * 1.1:
            # Defekt ist noch hörbar — booste Richtung benötigter Stärke
            _EMERGENCY_STRENGTHS = [
                _audibility_boost * 0.80,  # 80% der benötigten Stärke
                _audibility_boost,  # Volle benötigte Stärke
            ] + _EMERGENCY_STRENGTHS
        # §0l: Emergency-Retries nur wenn Team netto negativ (oder nahe null) ist.
        # Wenn best_scores bereits Team-Net-Positiv sind und Regression unter
        # 1.5×_CATASTROPHIC_THRESHOLD liegt, würden Emergency-Retries Over-Processing
        # bei einer Phase riskieren, die im Team schon funktioniert hat.
        _pre_em_team_net = sum(
            best_scores.get(g, 0.5) - effective_scores_before.get(g, 0.5) for g in effective_goals
        ) / max(len(effective_goals), 1)
        _skip_emergency_team_gate = (
            _pre_em_team_net > 0.02 and best_regression < _CATASTROPHIC_THRESHOLD * 1.5 and _worst_prio >= 3
        )
        if _skip_emergency_team_gate:
            logger.info(
                "PMGG §0l: %s Emergency-Retries übersprungen — Team-Net=+%.4f positiv "
                "(best_regression=%.4f < 1.5×catastrophic=%.4f, worst_prio=P%d)",
                phase_id,
                _pre_em_team_net,
                best_regression,
                _CATASTROPHIC_THRESHOLD * 1.5,
                _worst_prio,
            )
        if (
            (not _skip_emergency_team_gate)
            and (not self._last_retry_budget_policy)
            and _allow_emergency_retries(
                phase_id,
                _worst_prio,
                best_regression,
                _CATASTROPHIC_THRESHOLD,
                _team_policy,
            )
        ):
            logger.warning(
                "PMGG: %s catastrophic regression %.4f > %.2f"
                " (worst goal: %s P%d) — attempting emergency low-strength retries",
                phase_id,
                best_regression,
                _CATASTROPHIC_THRESHOLD,
                _worst_goal,
                _worst_prio,
            )
            for _em_strength in _EMERGENCY_STRENGTHS:
                _retry_elapsed = time.time() - _retry_t0
                if _retry_elapsed > _RETRY_BUDGET_S:
                    break
                if _is_ml_deterministic:
                    audio_em = self._wet_dry_blend(
                        audio, audio_full if audio_full is not None else best_audio, _em_strength, phase
                    )
                else:
                    audio_em = self._run_phase(phase, audio, _em_strength, phase_kwargs)
                _em_sample = _extract_sample(
                    audio_em,
                    sr,
                    duration_s=sample_duration_s,
                    defect_locations=_defect_locs,
                    phase_id=phase_id,
                )
                scores_em = _measure_quick(_em_sample, sr, reference=_ref_sample, precise_override=False)
                regression_em = self._max_regression(
                    effective_scores_before, scores_em, _goals_for_regression, goal_weights=goal_weights
                )
                _ci_penalty_em, _ci_meta_em = _content_integrity_penalty(audio, audio_em)
                if _ci_penalty_em > 0.0:
                    regression_em = max(regression_em, threshold + 0.001 + 0.05 * _ci_penalty_em)
                    logger.debug(
                        "PMGG Content-Guard emergency: %s (rms_drop=%.2f dB corr=%.3f penalty=%.3f)",
                        phase_id,
                        _ci_meta_em.get("rms_drop_db", 0.0),
                        _ci_meta_em.get("corr", 1.0),
                        _ci_penalty_em,
                    )
                if regression_em <= threshold:
                    if audio_full is not None:
                        del audio_full
                    scores_em = _apply_precise_metric_overrides(scores_em, _em_sample, sr, reference=_ref_sample)
                    return audio_em, scores_em, f"emergency_s{_em_strength:.2f}", _em_strength
                _net_em = sum(
                    scores_em.get(g, 0.5) - effective_scores_before.get(g, 0.5) for g in effective_goals
                ) / max(len(effective_goals), 1)
                _is_better_em = regression_em < best_regression or (
                    abs(regression_em - best_regression) < threshold * 0.15 and _net_em > _best_team_net
                )
                if _is_better_em:
                    best_audio = audio_em
                    best_scores = scores_em
                    best_regression = regression_em
                    best_strength = _em_strength
                    best_action = "best_effort_emergency"
                    _best_team_net = _net_em
        elif best_regression > _CATASTROPHIC_THRESHOLD and _worst_prio <= 2:
            logger.info(
                "PMGG: %s catastrophic path uebersprungen by team policy (reason=%s, regression=%.4f, Schwelle=%.3f)",
                phase_id,
                _team_policy.get("reason", "none") if isinstance(_team_policy, dict) else "none",
                best_regression,
                _CATASTROPHIC_THRESHOLD,
            )

        # §2.29 KEIN Rollback — Phase wird mit geringster Regression angewendet.
        # VERBOTEN: Phase überspringen (Original-Audio zurückgeben).
        # CausalDefectReasoner hat diese Phase als notwendig bestimmt.
        # Sofortige Freigabe: audio_full (+86 MB bei 225s) nicht bis GC halten.
        if audio_full is not None:
            del audio_full
        # Apply precise overrides once for accurate score propagation to next phase
        _best_sample = _extract_sample(
            best_audio,
            sr,
            duration_s=sample_duration_s,
            defect_locations=_defect_locs,
            phase_id=phase_id,
        )
        best_scores = _apply_precise_metric_overrides(best_scores, _best_sample, sr, reference=_ref_sample)
        _pmgg_msg = (
            "⚠️ PMGG: %s best-effort (strength=%.2f, Regression=%.4f > threshold=%.3f) — "
            "HPE-Gate prüft ob Phase für menschliche Ohren akzeptabel ist"
        )
        if _worst_prio <= 2 or best_regression >= _CATASTROPHIC_THRESHOLD:
            logger.warning(
                _pmgg_msg,
                phase_id,
                best_strength,
                best_regression,
                threshold,
            )
        else:
            logger.info(
                _pmgg_msg,
                phase_id,
                best_strength,
                best_regression,
                threshold,
            )
        # §v10.17 LAG-GATE: Keine Phase darf Stereo-Lag einführen
        try:
            if best_audio.ndim == 2 and audio.ndim == 2:
                from backend.file_import import _estimate_interchannel_lag_samples as _lag_measure

                _lag_before = _lag_measure(audio, 48000)
                _lag_after = _lag_measure(best_audio, 48000)
                _lag_delta = abs(_lag_after - _lag_before)
                if _lag_delta > 2:
                    logger.error(
                        "§v10.17 LAG-GATE [%s]: Verarbeitungsschritt introduced +%d samples lag — REVERTING to pre-Verarbeitungsschritt audio",
                        phase_id,
                        _lag_delta,
                    )
                    try:
                        from backend.core.phase_error_registry import get_error_registry

                        get_error_registry().record(
                            phase_id, "lag_introduced", f"lag delta={_lag_delta} samples", retries=0, severity="error"
                        )
                    except Exception as _e:
                        logger.debug("per_Verarbeitungsschritt_musical_goals_gate: unkritisch exception: %s", _e)
                    return audio, effective_scores_before, "lag_rejected", 0.0
        except Exception as _e:
            logger.debug("per_Verarbeitungsschritt_musical_goals_gate: unkritisch exception: %s", _e)

        # §v10.17 PSS-Gate: Perceptual Similarity gegen Original
        try:
            from backend.core.perceptual_reference_validator import get_perceptual_validator

            _prv = get_perceptual_validator()
            _anchor = _prv.get_anchor()
            if _anchor is not None:
                _pss_r = _prv.validate(best_audio, 48000, _anchor)
                if not _pss_r.accepted:
                    logger.warning("§v10.17 PSS-Gate [%s]: PSS=%.4f rejected", phase_id, _pss_r.perceptual_similarity)
                    return audio, effective_scores_before, "pss_rejected", 0.0
        except Exception as _e:
            logger.debug("per_Verarbeitungsschritt_musical_goals_gate: unkritisch exception: %s", _e)

        return best_audio, best_scores, best_action, best_strength

    def _run_phase(
        self,
        phase: Any,
        audio: np.ndarray,
        strength: float,
        phase_kwargs: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Führt Phase aus mit Wet/Dry-Modulation; nutzt bei Fehlern sicheren Audio-Fallback.

        CRITICAL FIX (v10.0.0): Ruft phase.process() statt phase() auf.
        PhaseInterface definiert kein __call__; der vorherige Code erzeugte
        TypeError, das still gefangen wurde — ALLE Phasen waren No-Ops.

        Wet/Dry-Modulation (§MusikalischeHarmonisierung):
        strength < 1.0 → audio_out = audio + strength × (processed - audio)
        Psychoakustisch korrekt: Sanftere Verarbeitung bei niedriger Stärke,
        statt binär „alles oder nichts".
        Timing-modifizierende Phasen (wow/flutter, speed) sind von Wet/Dry
        ausgenommen (Phasen-Artefakte bei Crossfade zeitversetzter Signale).
        """
        if phase_kwargs is None:
            phase_kwargs = {}

        def _safe_audio_fallback(x: np.ndarray) -> np.ndarray:
            """NaN-safe, clipped fallback that preserves input shape/layout."""
            _x = np.nan_to_num(np.asarray(x), nan=0.0, posinf=0.0, neginf=0.0)
            _x = np.clip(_x, -1.0, 1.0).astype(np.float32, copy=False)
            return np.asarray(_x)  # type: ignore[no-any-return]

        # Timing-modifizierende Phasen: kein Wet/Dry (Phasen-Artefakte)
        _TIMING_PHASES = frozenset(
            {
                "phase_12_wow_flutter_fix",
                "phase_31_speed_pitch_correction",
            }
        )
        try:
            # Strength als Kwarg übergeben, damit Phasen ihn OPTIONAL nutzen können
            kw = dict(phase_kwargs)
            kw["strength"] = strength
            # CRITICAL: phase.process() statt phase() — PhaseInterface hat kein __call__
            result = phase.process(audio, **kw)
            # §v10.18: resolved_defects aus PhaseResult extrahieren und
            # über Instanzvariable an _evaluate_and_decide weiterreichen
            _resolved = getattr(result, "resolved_defects", None) or {}
            self._last_resolved_defects = _resolved
            if hasattr(result, "audio"):
                out = result.audio
            elif hasattr(result, "processed_audio"):
                out = result.processed_audio
            elif isinstance(result, np.ndarray):
                out = result
            else:
                logger.debug(
                    "PMGG: Verarbeitungsschritt-Ausgabe kein ndarray/Ergebnis-Objekt; Ersatzpfad auf safe audio"
                )
                return _safe_audio_fallback(audio)

            if out is None or not isinstance(out, np.ndarray):
                logger.debug("PMGG: Verarbeitungsschritt-Ausgabe ungueltig (None/Typfehler); Ersatzpfad auf safe audio")
                return _safe_audio_fallback(audio)

            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            out = np.clip(out, -1.0, 1.0).astype(np.float32)

            # §2.61 Länge sicherstellen — shape-aware für channels-first Stereo.
            # BUG-GUARD: Bei channels-first (2, N) ist audio.shape[0]=2 (Kanäle), nicht
            # Samples. Ein Mono-Ausgabe (N,) würde auf 2 Samples gekürzt werden, was
            # UV3-Gain-Guards und _active_quality_intervention kaputt bricht.
            # Korrekt: Samples-Achse bestimmen und darauf abgleichen.
            _is_ch_first = audio.ndim == 2 and audio.shape[0] <= 2 and audio.shape[1] > audio.shape[0]
            _target_samples = int(audio.shape[1] if _is_ch_first else audio.shape[0])

            # Keep output layout consistent with input layout for both stereo orientations.
            # Some phases emit (N, 2) while others emit (2, N); normalize before
            # length guards so axis-0 is always interpreted correctly.
            if audio.ndim == 2 and out.ndim == 2:
                _audio_ch_first = audio.shape[0] <= 2 and audio.shape[1] > audio.shape[0]
                _out_ch_first = out.shape[0] <= 2 and out.shape[1] > out.shape[0]
                if _audio_ch_first != _out_ch_first and out.shape[0] != out.shape[1]:
                    out = out.T

            _out_samples = int(out.shape[-1] if (_is_ch_first and out.ndim == 2) else out.shape[0])
            if _out_samples != _target_samples:
                if _is_ch_first and out.ndim == 2:
                    # channels-first output: trim/pad along axis=1 (samples axis)
                    if out.shape[1] > _target_samples:
                        out = out[:, :_target_samples]
                    else:
                        out = np.pad(out, ((0, 0), (0, _target_samples - out.shape[1])))
                elif _is_ch_first and out.ndim == 1:
                    # mono output from channels-first input — only trim/pad sample dim
                    if out.shape[0] > _target_samples:
                        out = out[:_target_samples]
                    else:
                        out = np.pad(out, (0, _target_samples - out.shape[0]))
                else:
                    # standard: trim/pad along axis=0
                    if out.shape[0] > _target_samples:
                        out = out[:_target_samples, ...]
                    else:
                        pad_rows = _target_samples - int(out.shape[0])
                        pad_spec = [(0, pad_rows)] + [(0, 0)] * (max(out.ndim, 1) - 1)
                        out = np.pad(out, pad_spec)

            # Normalize channel shape before arithmetic blend.
            # If a phase emits mono from stereo input, upmix by duplication to keep
            # downstream shape contracts and avoid broadcast exceptions.
            if audio.ndim == 2:
                if out.ndim == 1:
                    if _is_ch_first:
                        out = np.tile(out[None, :], (audio.shape[0], 1))
                    else:
                        out = np.tile(out[:, None], (1, audio.shape[1]))
                elif out.ndim == 2 and out.shape != audio.shape:
                    # Last-resort channel alignment: keep available channels, preserve layout.
                    if _is_ch_first:
                        _n_ch = min(audio.shape[0], out.shape[0])
                        _aligned = np.zeros_like(audio, dtype=np.float32)
                        _aligned[:_n_ch, :] = out[:_n_ch, :]
                    else:
                        _n_ch = min(audio.shape[1], out.shape[1])
                        _aligned = np.zeros_like(audio, dtype=np.float32)
                        _aligned[:, :_n_ch] = out[:, :_n_ch]
                    out = _aligned

            # Wet/Dry-Modulation: strength < 1.0 → blende zwischen Original und Verarbeitet
            # §strength=0.0-Guard: Phasen ohne eigenes Zero-Strength-Skip (z.B.
            # CompressionPhase/LimitingPhase/FinalEQ) ignorieren den strength-Kwarg
            # sonst komplett bei strength=0.0 — 0.0 < strength schloss diesen Fall aus.
            if strength < 1.0:
                phase_id = ""
                try:
                    meta = phase.get_metadata()
                    phase_id = getattr(meta, "phase_id", "")
                except Exception as _meta_exc:
                    logger.debug("PMGG: Verarbeitungsschritt-Metadata-Zugriff fehlgeschlagen: %s", _meta_exc)
                if phase_id not in _TIMING_PHASES:
                    out = (audio + strength * (out - audio)).astype(np.float32)
                    out = np.clip(out, -1.0, 1.0)

            return np.asarray(out)  # type: ignore[no-any-return]
        except Exception as exc:
            logger.debug("PMGG: Verarbeitungsschritt-Ausführung fehlgeschlagen: %s", exc)
            return _safe_audio_fallback(audio)

    @staticmethod
    def _wet_dry_blend(
        dry: np.ndarray,
        wet: np.ndarray,
        strength: float,
        phase: Any = None,
    ) -> np.ndarray:
        """Phase-aware Wet/Dry-Blending (§9.10.118 — Kammfilter-Schutz).

        Bei niedrigen Strengths (< 0.30) erzeugt lineare Zeitdomänen-
        Interpolation Kammfilter-Artefakte (Original + phasenverschobenes
        Signal). Stattdessen: STFT-Magnitude-Interpolation mit bewahrter
        Originalphase.

        Wissensch. Basis: Wiener-Filtertheorie — Magnitude-Blending bei
        erhaltener Dry-Phase minimiert perceptual distortion (Ephraim & Malah
        1984).  Lineare Interpolation bleibt bei strength >= 0.30 (Kammfilter
        dort vernachlässigbar da Wet-Anteil dominiert).

        Timing-modifizierende Phasen (wow/flutter, speed) sind ausgenommen,
        da Crossfade zeitversetzter Signale Phasen-Artefakte erzeugt.
        """
        _TIMING_PHASES = frozenset(
            {
                "phase_12_wow_flutter_fix",
                "phase_31_speed_pitch_correction",
            }
        )
        dry = np.asarray(dry, dtype=np.float32)
        wet = np.asarray(wet, dtype=np.float32)

        _dry_ch_first = dry.ndim == 2 and dry.shape[0] <= 2 and dry.shape[1] > dry.shape[0]
        _wet_ch_first = wet.ndim == 2 and wet.shape[0] <= 2 and wet.shape[1] > wet.shape[0]
        if _dry_ch_first:
            dry = dry.T
        if _wet_ch_first:
            wet = wet.T

        def _match_time_axis(x: np.ndarray, target_len: int) -> np.ndarray:
            if x.shape[0] == target_len:
                return x
            if x.shape[0] > target_len:
                return x[:target_len, ...]
            pad_rows = target_len - int(x.shape[0])
            pad_spec = [(0, pad_rows)] + [(0, 0)] * (max(x.ndim, 1) - 1)
            return np.pad(x, pad_spec)  # type: ignore[no-any-return]

        # Time axis must always match before blending.
        wet = _match_time_axis(wet, int(dry.shape[0]))
        if strength >= 1.0:
            out = np.clip(wet, -1.0, 1.0).astype(np.float32)
            return out.T if _dry_ch_first and out.ndim == 2 else out  # type: ignore[no-any-return]
        if strength <= 0.0:
            out = dry.copy()
            return out.T if _dry_ch_first and out.ndim == 2 else out
        # Timing-Phasen: kein Blend
        phase_id = ""
        if phase is not None:
            try:
                meta = phase.get_metadata()
                phase_id = getattr(meta, "phase_id", "")
            except Exception as _meta_exc:
                logger.debug("PMGG: Wet/Dry-Blend Verarbeitungsschritt-Metadata-Zugriff fehlgeschlagen: %s", _meta_exc)
        if phase_id in _TIMING_PHASES:
            out = np.clip(wet, -1.0, 1.0).astype(np.float32)
            return out.T if _dry_ch_first and out.ndim == 2 else out  # type: ignore[no-any-return]

        # Stereo-safe handling: never run STFT blend on channel axis.
        if dry.ndim == 2 or wet.ndim == 2:
            if dry.ndim != 2 or wet.ndim != 2:
                logger.debug(
                    "PMGG Wet/Dry-Blend ndim mismatch dry=%s wet=%s; using linear Ersatzpfad",
                    dry.shape,
                    wet.shape,
                )
                if dry.ndim == 2 and wet.ndim == 1:
                    wet = np.tile(wet[:, None], (1, dry.shape[1]))
                elif dry.ndim == 1 and wet.ndim == 2:
                    wet = wet.mean(axis=1)
                out_lin = (dry + strength * (wet - dry)).astype(np.float32)
                out_lin = np.clip(out_lin, -1.0, 1.0)
                return np.asarray(out_lin.T if _dry_ch_first and out_lin.ndim == 2 else out_lin)  # type: ignore[no-any-return]

            if dry.shape[1] != wet.shape[1]:
                logger.debug(
                    "PMGG Wet/Dry-Blend channel mismatch dry=%s wet=%s; using linear Ersatzpfad",
                    dry.shape,
                    wet.shape,
                )
                n_ch = min(dry.shape[1], wet.shape[1])
                out = dry.copy()
                out[:, :n_ch] = dry[:, :n_ch] + strength * (wet[:, :n_ch] - dry[:, :n_ch])
                out = np.clip(out.astype(np.float32), -1.0, 1.0)
                return out.T if _dry_ch_first and out.ndim == 2 else out  # type: ignore[no-any-return]

            if strength < 0.30 and dry.shape[0] >= 2048:
                ch_out = []
                for ch in range(dry.shape[1]):
                    ch_out.append(
                        PerPhaseMusicalGoalsGate._wet_dry_blend(
                            dry[:, ch],
                            wet[:, ch],
                            strength,
                            phase=None,
                        )
                    )
                out = np.clip(np.stack(ch_out, axis=1).astype(np.float32), -1.0, 1.0)
                return out.T if _dry_ch_first and out.ndim == 2 else out  # type: ignore[no-any-return]

            out = (dry + strength * (wet - dry)).astype(np.float32)
            out = np.clip(out, -1.0, 1.0)
            return out.T if _dry_ch_first and out.ndim == 2 else out  # type: ignore[no-any-return]

        # §9.10.118: phase-aware STFT blending for low strengths to prevent
        # comb-filter artifacts from time-domain mixing of phase-shifted signals.
        if strength < 0.30 and len(dry) >= 2048:
            try:
                win_size = 2048
                hop = win_size // 4
                from scipy.signal import istft as _istft
                from scipy.signal import stft as _stft

                _, _, Zxx_dry = _stft(dry, fs=48000, nperseg=win_size, noverlap=win_size - hop)
                _, _, Zxx_wet = _stft(wet, fs=48000, nperseg=win_size, noverlap=win_size - hop)

                # §2.43 Phase-Preserved Wet/Dry-Blend:
                # M_blend = (1−α)·M_dry + α·M_wet, Phase vom Dry-Signal
                mag_dry = np.abs(Zxx_dry)
                mag_wet = np.abs(Zxx_wet)
                phase_dry = np.angle(Zxx_dry)

                mag_blend = mag_dry + strength * (mag_wet - mag_dry)
                Zxx_blend = mag_blend * np.exp(1j * phase_dry)

                _, out = _istft(Zxx_blend, fs=48000, nperseg=win_size, noverlap=win_size - hop)
                # Length matching
                if len(out) > len(dry):
                    out = out[: len(dry)]
                elif len(out) < len(dry):
                    out = np.pad(out, (0, len(dry) - len(out)))
                out = np.clip(out.astype(np.float32), -1.0, 1.0)
                return out.T if _dry_ch_first and out.ndim == 2 else out  # type: ignore[no-any-return]
            except Exception as _stft_exc:
                logger.debug("PMGG STFT-Blend Ersatzpfad to linear: %s", _stft_exc)

        out = (dry + strength * (wet - dry)).astype(np.float32)
        out = np.clip(out, -1.0, 1.0)
        return out.T if _dry_ch_first and out.ndim == 2 else out  # type: ignore[no-any-return]

    @staticmethod
    def _max_regression(
        before: dict[str, float],
        after: dict[str, float],
        goals: list | None = None,
        goal_weights: dict[str, float] | None = None,
    ) -> float:
        """Maximale negative Differenz in Musical Goals (positiv = Regression).

        §2.56 Song-specific goal weighting: if goal_weights is provided,
        each goal's regression is multiplied by its weight before taking the max.
        weight > 1.0 → regression is amplified (stricter for important goals).
        weight < 1.0 → regression is dampened (lenient for less relevant goals).
        """
        check_goals = goals if goals is not None else FAST_GOALS_SUBSET
        max_reg = 0.0
        for g in check_goals:
            delta = after.get(g, 0.5) - before.get(g, 0.5)
            if delta < 0:
                raw_reg = -delta
                # §2.56: Apply song-specific weight
                w = goal_weights.get(g, 1.0) if goal_weights else 1.0
                weighted_reg = raw_reg * w
                max_reg = max(max_reg, weighted_reg)
        return max_reg

    @staticmethod
    def _max_regression_priority_aware(
        before: dict[str, float],
        after: dict[str, float],
        goals: list | None = None,
        threshold: float = 0.020,
        goal_weights: dict[str, float] | None = None,
    ) -> tuple[float, int]:
        """Priority-aware regression: returns (max_regression, worst_priority).

        Only considers goals whose priority-adjusted threshold is exceeded.
        Returns the highest priority level (lowest number) among regressed goals.

        §2.56: goal_weights modulate the effective threshold per goal.
        weight > 1.0 → effective threshold is lower (stricter for important goals).
        weight < 1.0 → effective threshold is higher (lenient).

        Args:
            before: Scores before phase.
            after: Scores after phase.
            goals: Subset of goals to check.
            threshold: Base regression threshold.
            goal_weights: Per-goal importance weights ∈ [0.3, 2.0].

        Returns:
            (max_regression_value, worst_priority) where worst_priority is 1–5
            (1 = most critical). Returns (0.0, 99) if no regression detected.
        """
        from backend.core.goal_priority_protocol import get_goal_priority_protocol

        gpp = get_goal_priority_protocol()
        check_goals = goals if goals is not None else FAST_GOALS_SUBSET
        max_reg = 0.0
        worst_prio = 99
        for g in check_goals:
            delta = after.get(g, 0.5) - before.get(g, 0.5)
            if delta < 0:
                raw_reg = -delta
                # §2.56: weight amplifies regression for important goals
                w = goal_weights.get(g, 1.0) if goal_weights else 1.0
                weighted_reg = raw_reg * w
                prio = gpp.priority_of(g)
                prio_threshold = threshold * _PRIORITY_THRESHOLD_FACTOR.get(prio, 1.0)
                if weighted_reg > prio_threshold:
                    worst_prio = min(worst_prio, prio)
                    max_reg = max(max_reg, weighted_reg)
        return max_reg, worst_prio

    @staticmethod
    def _compute_team_net_delta(
        before: dict[str, float],
        after: dict[str, float],
        goals: list[str] | None = None,
        goal_weights: dict[str, float] | None = None,
        canonical_thresholds: dict[str, float] | None = None,
    ) -> tuple[float, bool, int]:
        """Gewichteter Netto-Team-Delta über alle effektiven Ziele (§0l §1.2c Teamwork-Invariante).

        Misst, ob das 15-Ziel-Team als Ganzes vom Phaseneinsatz profitiert hat —
        unabhängig davon, ob ein einzelnes Ziel leicht regressiert.
        Positiv = Team hat sich netto verbessert; negativ = Team hat sich verschlechtert.

        Args:
            before: Scores vor der Phase.
            after: Scores nach der Phase.
            goals: Subset zu prüfender Ziele. None = alle FAST_GOALS_SUBSET.
            goal_weights: §2.56 song-spezifische Gewichtungen ∈ [0.3, 2.0].
            canonical_thresholds: Normative Pflichtschwellen (Restoration oder Studio 2026).
                Wenn übergeben: P1/P2-Ziele werden gegen diese Schwellen geprüft
                um sicherzustellen, dass kein P1/P2-Ziel unter seinen absoluten Boden fällt.

        Returns:
            (net_delta, all_p1p2_above_floor, min_regressed_priority)
            - net_delta: Gewichteter Durchschnitt aller Ziel-Deltas.
              Positiv = Team verbessert, negativ = Team verschlechtert.
            - all_p1p2_above_floor: True wenn kein P1/P2-Ziel unter seinem kanonischen
              Schwellwert liegt (canonical_thresholds erforderlich; True wenn nicht übergeben).
            - min_regressed_priority: Niedrigste Priorität (1=kritisch, 5=peripher)
              eines Ziels das regressiert hat. 99 = kein Ziel regressiert.
        """
        from backend.core.goal_priority_protocol import get_goal_priority_protocol

        gpp = get_goal_priority_protocol()
        check_goals = goals if goals is not None else FAST_GOALS_SUBSET

        total_weight = 0.0
        weighted_delta_sum = 0.0
        all_p1p2_above_floor = True
        min_regressed_priority = 99

        for g in check_goals:
            b_val = before.get(g, 0.5)
            a_val = after.get(g, 0.5)
            delta = a_val - b_val
            w = (goal_weights or {}).get(g, 1.0)
            # §2.55c Anti-Korrelationsdämpfung: Wenn Goal G regressiert und ein physikalisch
            # anti-korreliertes Ziel sich gleichzeitig verbessert, ist die Regression erwartet
            # und wird im Team-Net-Beitrag gedämpft (nicht im Regressions-Gate, nur hier).
            # Quelle: GOAL_ANTI_CORRELATIONS (§2.55c). Max. 60 % Dämpfung pro Paar.
            team_delta = delta
            if delta < 0:
                for _anti_pair, _anti_factor in GOAL_ANTI_CORRELATIONS.items():
                    if g in _anti_pair:
                        _partner = next((x for x in _anti_pair if x != g), None)
                        if _partner and _partner in check_goals:
                            _partner_delta = after.get(_partner, 0.5) - before.get(_partner, 0.5)
                            if _partner_delta > 0.0:
                                # Partner verbessert sich → physikalisch erwartete Regression dämpfen.
                                _dampening = min(abs(_anti_factor) * 0.6, 0.60)
                                team_delta = delta * (1.0 - _dampening)
                                break
            weighted_delta_sum += team_delta * w
            total_weight += w

            if delta < 0:
                prio = gpp.priority_of(g)
                min_regressed_priority = min(min_regressed_priority, prio)
                # P1/P2-Bodenkontrolle: Wenn kanonische Schwellen übergeben wurden,
                # prüfen ob der Score NACH der Phase noch über dem Pflichtboden liegt.
                if canonical_thresholds is not None and prio <= 2:
                    floor = float(canonical_thresholds.get(g, 0.0))
                    if a_val < floor:
                        all_p1p2_above_floor = False

        net_delta = weighted_delta_sum / max(total_weight, 1e-9)
        return net_delta, all_p1p2_above_floor, min_regressed_priority

    @staticmethod
    def _get_phase_id(phase: Any) -> str:
        """Extrahiert Phase-ID aus MetaDaten oder Klassennamen."""
        try:
            meta = phase.get_metadata()
            return getattr(meta, "phase_id", type(phase).__name__)
        except Exception as e:
            logger.warning(
                "per_Verarbeitungsschritt_musical_goals_gate.py::_get_Verarbeitungsschritt_id Ersatzpfad: %s", e
            )
            return type(phase).__name__

    @staticmethod
    def _classify_action_decision(action: str) -> tuple[str, str]:
        """Mappt PMGG-Action auf stabile Klasse + Grundcode für Telemetrie (SOTA).

        Deterministische Klassifizierung: Jede Action wird eindeutig einer
        Kategorie zugeordnet. Bei unbekannten Actions wird "other" zurückgegeben
        mit logging-Warnung statt stillschweigendem Fallback.
        """
        a = str(action or "")
        if a == "passed":
            return "pass", "regression_within_threshold"
        if a == "passthrough":
            return "pass", "phase_passthrough_no_change"
        if a == "sub_threshold":
            return "pass", "jnd_sub_threshold_accept"
        if a == "passed_p4p5_tolerated":
            return "pass", "priority_tolerance_band_accept"
        if a == "passed_team_balanced":
            return "pass", "team_net_positive_balance_accept"
        if a == "passed_reconstruction_localized":
            return "pass", "reconstruction_localized_collateral_accept"
        if a.startswith("retry"):
            return "retry", "regression_over_threshold_retry_success"
        if a.startswith("emergency_s"):
            return "emergency", "catastrophic_regression_emergency_success"
        if a == "best_effort_emergency":
            # §v10.703 SOTA: Deterministischer Grundcode statt "unresolved"
            # Katastrophale Regression → best_effort mit konkretem Diagnose-Code
            return "best_effort", "catastrophic_regression_best_effort_applied"
        if a == "best_effort_accepted":
            return "best_effort", "legacy_best_effort_accepted"
        if a.startswith("best_effort"):
            return "best_effort", "retry_exhausted_best_effort"
        if a == "hpe_skip":
            return "skip", "hpe_pleasantness_decline_skip"  # §v10

        # Deterministischer Fallback: Unbekannte Action wird geloggt und klassifiziert
        logger.warning("PMGG unbekannte Action: '%s' — klassifiziert als 'other'", a)
        return "other", "unclassified_action_fallback"


# ---------------------------------------------------------------------------
# Convenience-Funktion
# ---------------------------------------------------------------------------


def wrap_phase(  # pylint: disable=too-many-positional-arguments
    phase: Any,
    audio: np.ndarray,
    sr: int,
    phase_id: str | None = None,
    scores_before: dict[str, float] | None = None,
    restorability_score: float = 70.0,
    applicable_goals: set[str] | None = None,
    is_studio_2026: bool = False,
    goal_weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float], PhaseGateLogEntry]:
    """
    Convenience-Wrapper: Führt eine Phase aus mit Musical-Goals-Schutz.

    Args:
        phase: PhaseInterface-Instanz
        audio: Input-Audio (float32, 48 kHz)
        sr: 48000 Hz
        phase_id: Optional explicit phase id for backward-compatible callers.
        scores_before: Vorherige Goal-Scores (optional)
        restorability_score: RestorabilityEstimator-Score ∈ [0, 100], bestimmt
                             adaptiven REGRESSION_THRESHOLD (§2.29).
        applicable_goals: Aus GoalApplicabilityFilter — nur diese Ziele geprüft.
        is_studio_2026: True for Studio 2026 mode (higher P3–P5 thresholds).
        goal_weights: §2.56 Song-specific goal importance weights.

    Returns:
        (audio_out, scores_after, log_entry)
    """
    return get_phase_gate().wrap_phase(
        phase,
        audio,
        sr,
        phase_id=phase_id,
        scores_before=scores_before,
        restorability_score=restorability_score,
        applicable_goals=applicable_goals,
        is_studio_2026=is_studio_2026,
        goal_weights=goal_weights,
    )
