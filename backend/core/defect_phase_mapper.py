"""
core/defect_phase_mapper.py
Defect-to-Phase Mapper (DPM)
==============================

Ordnet jeden der 20 DefectType-Werte den jeweils relevanten Phasen zu
und konfiguriert den ProcessingConfig-Parameter-Satz präzise für den
Primary-Defekt.

Aufgaben:
  A) Semantisch: Welche Phase(n) sind für diesen Defekt zuständig?
     → Gibt geordnete Liste von phase_id-Strings zurück
     → Priorität: primary > secondary (wichtig für zukünftige Phasen-Selektion)

  B) Aktionär: Wie soll ProcessingConfig für diesen Defekt gesetzt werden?
     → Gibt einen Dict mit Config-Delta zurück (nur geänderte Felder)
     → Wird von ARE._build_specialist_variant() verwendet

Bekannte Phase-IDs (aus core/phases/):
  phase_01_click_removal        — Multi-Scale Click-Detektion
  phase_02_hum_removal          — Netzbrumm-Entfernung (50/60 Hz + Harmonische)
  phase_03_denoise              — Breitband-Rauschunterdrückung
  phase_04_eq_correction        — Tonal-Korrektur / EQ
  phase_05_rumble_filter        — Tieffrequenz-Rumpel / Trittschall
  phase_06_frequency_restoration— Hochfrequenz-Restauration
  phase_07_harmonic_restoration — Harmonische Wiederherstellung
  phase_08_transient_preservation— Attack-Schutz / Transientenerhalt
  phase_09_crackle_removal      — Knistergeräusch-Entfernung
  phase_12_wow_flutter_fix      — Wow/Flutter-Korrektur (Bandschwankung)
  phase_13_stereo_enhancement   — Stereo-Verbesserung
  phase_14_phase_correction     — Phasenfehler-Korrektur
  phase_15_stereo_balance       — L/R-Balance-Ausgleich
  phase_18_noise_gate           — Rauschtor (Stille-Reinigung)
  phase_23_spectral_repair      — Spektrale Reparatur (Gaps, Dropouts)
  phase_24_dropout_repair       — Aussetzer-Reparatur via Interpolation
  phase_25_azimuth_correction   — Azimuth-/Phasen-Fehler (Bandmaschine)
  phase_26_dynamic_range_expansion— DRE (Compander, um Kompression rückgängig)
  phase_27_click_pop_removal    — Knacker/Pop-Entfernung (komplementär zu 01)
  phase_28_surface_noise_profiling— Vinyl/Shellac Oberflächen-Profiling
  phase_29_tape_hiss_reduction  — Bandrauschen-Reduktion
  phase_30_dc_offset_removal    — DC-Versatz-Entfernung
  phase_31_speed_pitch_correction— Tonhöhen/-geschwindigkeits-Korrektur
  phase_33_stereo_width_limiter — Stereobreiten-Begrenzung
  phase_34_mid_side_processing  — Mid/Side-Verarbeitung für Phasenfehler
  phase_50_spectral_repair      — Erweiterte spektrale Reparatur (Phase 2)

Author: Aurik Development Team
Version: 1.0.0
Date: 2026-02-17
"""

from __future__ import annotations

import copy
import logging
import threading
from dataclasses import dataclass

from backend.core.defect_scanner import (
    DefectType,
    MaterialType,  # §v10.113
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# §0a: Studio-2026-only phases — NEVER suggested in Restoration mode
# (BUG-FIX v10.0.0 §0a — UV3-Guard blocks them anyway, but CausalDefectReasoner
# and DefectPhaseMapper must not even propose them for restoration runs)
# ---------------------------------------------------------------------------
_RESTORATION_FORBIDDEN_PHASES: frozenset[str] = frozenset(
    {
        "phase_21_exciter",
        "phase_35_multiband_compression",
        "phase_42_vocal_enhancement",
    }
)

# ---------------------------------------------------------------------------
# Digitale Codec-Materialien: Kratzentferner physikalisch sinnlos (V29, §4.11)
# phase_09_crackle_removal wird für diese Materialien stets ausgeblendet.
# ---------------------------------------------------------------------------
_DIGITAL_CODEC_MATERIALS: frozenset[str] = frozenset(
    {"mp3_low", "mp3_high", "aac", "cd_digital", "dat", "streaming", "minidisc"}
)

# ---------------------------------------------------------------------------
# §2.67 Koalitions-Priorisierung im DefectMapper
# Zusammengehörige Phasen werden bereits in der Defektprofil-Selektion näher
# zusammengezogen, damit sie nicht erst spät durch globale Sortierung getrennt
# werden. Nur additive Priorisierung, keine harte Erzwingung.
# ---------------------------------------------------------------------------
_DEFECT_MAPPER_PHASE_COALITIONS: dict[str, tuple[str, ...]] = {
    "digital_repair_chain": ("phase_23_spectral_repair", "phase_50_spectral_repair"),
    "hiss_harmonic_rebuild": ("phase_29_tape_hiss_reduction", "phase_07_harmonic_restoration"),
    "stereo_alignment": ("phase_14_phase_correction", "phase_25_azimuth_correction"),
    "generation_loss_rebuild": ("phase_23_spectral_repair", "phase_07_harmonic_restoration"),
}


# ---------------------------------------------------------------------------
# Datenstruktur: PhaseAssignment
# ---------------------------------------------------------------------------


@dataclass
class PhaseAssignment:
    """Zuordnung eines DefectType zu Phase-IDs und Config-Parametern."""

    defect_type: DefectType
    """Defekt-Typ den diese Zuordnung abdeckt."""

    primary_phases: list[str]
    """Phase-IDs, die diesen Defekt direkt behandeln (höchste Priorität)."""

    secondary_phases: list[str]
    """Ergänzende Phasen (Nutzen wenn primary nicht ausreicht)."""

    description: str
    """Menschenlesbare Beschreibung der Strategie."""

    config_delta: dict[str, object]
    """
    ProcessingConfig-Felder die für diesen Defekt optimal gesetzt werden.
    Enthält NUR die Felder die vom Default abweichen.
    Stärke ist auf severity=1.0 kalibriert; wird beim Anwenden skaliert.
    """

    def apply_to_config(
        self,
        config: object,
        severity: float = 1.0,
        mode_factor: float = 1.0,
        material_factor: float = 1.0,
    ) -> None:
        """
        Wendet config_delta auf ein ProcessingConfig-Objekt an.

        Args:
            config: ProcessingConfig-Instanz (wird in-place modifiziert)
            severity: Defekt-Severity [0–1]; skaliert Stärke-Parameter
            mode_factor: Zusätzlicher Modus-Faktor (z.B. 0.8 für RESTORATION)
            material_factor: Material-Anpassungsfaktor aus _MATERIAL_PHASE_FACTORS
                (Werte < 1.0 = Charakter-Schutz, Werte > 1.0 werden auf 1.0 geklammert).
        """
        effective = max(0.0, min(1.0, severity * mode_factor * material_factor))
        for k, v in self.config_delta.items():
            if not hasattr(config, k):
                logger.warning("ProcessingConfig hat kein Feld %r — übersprungen.", k)
                continue
            if isinstance(v, float):
                # Float-Parameter: linear skalieren
                scaled = float(v) * effective
                # Klemmen auf [0.0, 1.0] für Stärke-Felder
                if k.endswith(("_strength", "_sensitivity", "_factor")):
                    scaled = max(0.0, min(1.0, scaled))
                setattr(config, k, scaled)
            elif isinstance(v, bool):
                # Bool-Flags: ab severity ≥ 0.3 aktivieren
                setattr(config, k, True if effective >= 0.3 else v)
            elif isinstance(v, int):
                setattr(config, k, v)
            else:
                setattr(config, k, v)


def _validate_phase_map_completeness() -> None:
    """Stellt sicher, dass jede Defektart eine explizite Phase-Zuordnung hat.

    Die DefectPhaseMapper ist die kanonische Entscheidungsquelle für alle
    Defekte. Lücken würden bedeuten, dass einzelne Defekte nur teilweise oder
    implizit behandelt werden.
    """
    missing = [dt for dt in DefectType if dt not in _PHASE_MAP]
    if missing:
        missing_names = ", ".join(dt.name for dt in missing)
        raise RuntimeError(f"DefectPhaseMapper unvollständig: fehlende Defekte: {missing_names}")

    for defect_type, assignment in _PHASE_MAP.items():
        if not assignment.primary_phases:
            raise RuntimeError(f"DefectPhaseMapper ungültig: {defect_type.name} hat keine Primary-Phase")


# ---------------------------------------------------------------------------
# Vollständige Mapping-Tabelle
# ---------------------------------------------------------------------------

_PHASE_MAP: dict[DefectType, PhaseAssignment] = {
    # ------------------------------------------------------------------
    # CLICKS — Vinyl/Shellac-Knackser, Klicks, Nadel-Impulse
    # ------------------------------------------------------------------
    DefectType.CLICKS: PhaseAssignment(
        defect_type=DefectType.CLICKS,
        primary_phases=[
            "phase_01_click_removal",  # Multi-Scale Adaptive Click Detection
            "phase_27_click_pop_removal",  # Click/Pop mit erweitertem Kontext
        ],
        secondary_phases=[
            "phase_08_transient_preservation",  # Schützt Musik-Transienten
            "phase_30_dc_offset_removal",  # Beseitigt DC von Klick-Resten
        ],
        description=(
            "Klickentfernung via statistischer Detektion + Interpolation. "
            "Bewahrt Musik-Transienten durch parallele Transientenanalyse."
        ),
        config_delta={
            "click_removal_sensitivity": 0.90,  # Hoch: viele Klicks treffen
            "denoise_strength": 0.0,  # Kein Rauschen entfernen (nur Klicks)
            "declip_strength": 0.30,  # Leicht: Klick-Spitzen kappen
            "preserve_analog_character": True,  # Oberflächen-Charakter bewahren
            "enable_spectral_repair": True,  # Lücken nach Click-Removal schließen
            "spectral_repair_strength": 0.60,
        },
    ),
    # ------------------------------------------------------------------
    # CRACKLE — kontinuierliches Knistern (Vinyl, Shellac)
    # ------------------------------------------------------------------
    DefectType.CRACKLE: PhaseAssignment(
        defect_type=DefectType.CRACKLE,
        primary_phases=[
            "phase_09_crackle_removal",  # Crackle-Klassifikation + Unterdrückung
            "phase_28_surface_noise_profiling",  # Oberflächen-Lärm-Profil
        ],
        secondary_phases=[
            "phase_18_noise_gate",  # Stille zwischen Knistern bereinigen
            "phase_03_denoise",  # Rest-Knisterpegel absenken
        ],
        description=(
            "Vinyl/Shellac-Knistern-Entfernung über Crackle-Klassifikator. "
            "Unterscheidet Crackle von Perkussions-Transienten."
        ),
        config_delta={
            "click_removal_sensitivity": 0.60,  # Mittel: Crackle-Events
            "denoise_strength": 0.25,  # Leichtes Rauschen nach Crackle-Profiling
            "declip_strength": 0.20,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # HUM — Netzbrumm (50/60 Hz) + Harmonische
    # ------------------------------------------------------------------
    DefectType.HUM: PhaseAssignment(
        defect_type=DefectType.HUM,
        primary_phases=[
            "phase_02_hum_removal",  # Adaptive Notch-Filter 50/60 Hz + OK
        ],
        secondary_phases=[
            "phase_04_eq_correction",  # Breitband-EQ nach Brumm-Entfernung
            "phase_05_rumble_filter",  # Tieffrequenz-Begleitgeräusch
        ],
        description=(
            "Netzbrumm-Entfernung (50/60 Hz + Harmonische bis 1000 Hz) "
            "via adaptivem Notchfilter ohne Musiksignal zu beschädigen."
        ),
        config_delta={
            "denoise_strength": 0.40,
            "click_removal_sensitivity": 0.0,  # Klicks sind kein HUM-Problem
            "low_freq_rolloff_hz": 20,  # DC-Versatz / Sub-Hz-Rumpel mitentfernen
            "declip_strength": 0.0,
        },
    ),
    # ------------------------------------------------------------------
    # WOW/FLUTTER — Bandschwankung, Leierkassetten-Effekt
    # ------------------------------------------------------------------
    DefectType.WOW: PhaseAssignment(
        defect_type=DefectType.WOW,
        primary_phases=[
            "phase_12_wow_flutter_fix",  # Pitch-Detect + Time-Warp
            "phase_31_speed_pitch_correction",  # Grobe Geschwindigkeitsfehler
        ],
        secondary_phases=[
            "phase_25_azimuth_correction",  # Azimuth-Abweichung von Bandmaschinen
        ],
        description=(
            "Wow/Flutter-Korrektur via PSOLA- oder Phase-Vocoder-Time-Warping. "
            "Detektiert Tonhöhenschwankungen < 6 Hz (Wow) und 6–100 Hz (Flutter)."
        ),
        config_delta={
            "denoise_strength": 0.15,  # Minimal: Rauschen ist Sekundärproblem
            "click_removal_sensitivity": 0.20,
            "declip_strength": 0.0,
            "preserve_analog_character": True,
        },
    ),
    DefectType.FLUTTER: PhaseAssignment(
        defect_type=DefectType.FLUTTER,
        primary_phases=[
            "phase_12_wow_flutter_fix",  # Pitch-Detect + Time-Warp
            "phase_31_speed_pitch_correction",  # Grobe Geschwindigkeitsfehler
        ],
        secondary_phases=[
            "phase_25_azimuth_correction",  # Bandmaschinen-Nebeneffekte
        ],
        description=(
            "Flutter-Korrektur via Wow/Flutter-Spezialpfad mit Fokus auf schnellere "
            "Tonhöhenschwankungen und mechanische Laufwerksinstabilität."
        ),
        config_delta={
            "denoise_strength": 0.10,
            "click_removal_sensitivity": 0.15,
            "declip_strength": 0.0,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # STEREO_IMBALANCE — L/R-Pegel, Kanaldefekte, Mono-Probleme
    # ------------------------------------------------------------------
    DefectType.STEREO_IMBALANCE: PhaseAssignment(
        defect_type=DefectType.STEREO_IMBALANCE,
        primary_phases=[
            "phase_15_stereo_balance",  # L/R-Balance-Korrektur
            "phase_33_stereo_width_limiter",  # Kanal-Extrema begrenzen
        ],
        secondary_phases=[
            "phase_14_phase_correction",  # Phasenfehler können Balance imitieren
            "phase_34_mid_side_processing",  # M/S für präzise Balance-Kontrolle
        ],
        description=(
            "L/R-Kanalbalance-Ausgleich und Stereobreiten-Normierung. "
            "Behandelt auch Mono-Summen-Probleme und Phasen-Differenzen."
        ),
        config_delta={
            "stereo_width_factor": 1.0,  # Korrigiert auf neutrale Breite
            "denoise_strength": 0.10,
            "click_removal_sensitivity": 0.20,
            "declip_strength": 0.0,
        },
    ),
    # ------------------------------------------------------------------
    # DIGITAL_ARTIFACTS — MP3/AAC-Blockartefakte, Codec-Rauschen
    # ------------------------------------------------------------------
    DefectType.DIGITAL_ARTIFACTS: PhaseAssignment(
        defect_type=DefectType.DIGITAL_ARTIFACTS,
        primary_phases=[
            "phase_23_spectral_repair",  # Spektrale Lücken füllen
            "phase_50_spectral_repair",  # Erweiterte Reparatur (Phase 2)
        ],
        secondary_phases=[
            "phase_07_harmonic_restoration",  # Verlorene Obertöne wiederherstellen
            "phase_06_frequency_restoration",  # Hochfrequenz-Ausdünnung korrigieren
        ],
        description=(
            "MP3/AAC/Streaming-Codec-Artefakt-Entfernung via spektraler Interpolation "
            "und harmonischer Rekonstruktion. Preibischner-Algorithmus-basiert."
        ),
        config_delta={
            "enable_spectral_repair": True,
            "spectral_repair_strength": 0.85,
            "spectral_repair_hole_threshold_db": -45.0,
            "denoise_strength": 0.20,
            "click_removal_sensitivity": 0.10,
            "declip_strength": 0.0,
            "high_freq_boost_db": 1.5,  # Leicht: Codec schneidet HF ab
        },
    ),
    # ------------------------------------------------------------------
    # LOW_FREQ_RUMBLE — Trittschall, HVAC, Motorschleifen < 80 Hz
    # ------------------------------------------------------------------
    DefectType.LOW_FREQ_RUMBLE: PhaseAssignment(
        defect_type=DefectType.LOW_FREQ_RUMBLE,
        primary_phases=[
            "phase_05_rumble_filter",  # High-Pass + Resonanz-Dämpfung
        ],
        secondary_phases=[
            "phase_02_hum_removal",  # Begleitet oft Rumpel
            "phase_30_dc_offset_removal",  # DC-Komponente mitentfernen
        ],
        description=(
            "Sub-Bass-Rumpel-Entfernung via Butterworth High-Pass + adaptive "
            "Resonanz-Dämpfung ohne Bassinteresse zu beschädigen."
        ),
        config_delta={
            "low_freq_rolloff_hz": 40,  # Aggressiver HP-Filter
            "denoise_strength": 0.20,
            "click_removal_sensitivity": 0.10,
            "declip_strength": 0.0,
        },
    ),
    # ------------------------------------------------------------------
    # HIGH_FREQ_NOISE — Bandrauschen, weißes Rauschen, Vinyl-Hiss
    # ------------------------------------------------------------------
    DefectType.HIGH_FREQ_NOISE: PhaseAssignment(
        defect_type=DefectType.HIGH_FREQ_NOISE,
        primary_phases=[
            "phase_03_denoise",  # Breitband-NR (Spectral Subtraction / MMSE)
            "phase_29_tape_hiss_reduction",  # Tape-spezifisches HF-Rauschen
        ],
        secondary_phases=[
            "phase_18_noise_gate",  # Tiefe Rauschpegel in Pausen beseitigen
            "phase_16_final_eq",  # HF-Shaping nach Denoising
            "phase_58_lyrics_guided_enhancement",  # §2.36 PFLICHT: Phonem-Boundary-Schutz bei HF-NR auf Vokal
            "phase_65_vocal_naturalness_restoration",  # §7.10 VQI-Korrektiv nach NR
            # (Restoration only, panns_singing-Gate intern)
        ],
        description=(
            "Hochfrequenz-Rauschreduktion via spektraler Subtraktion + Wiener-Filter. "
            "Tape-Hiss-Profiling für bandmaterialspezifische Filtercharakteristik."
        ),
        config_delta={
            "denoise_strength": 0.80,
            "click_removal_sensitivity": 0.10,
            "declip_strength": 0.0,
            "preserve_analog_character": True,  # Charakter trotz NR bewahren
        },
    ),
    # ------------------------------------------------------------------
    # COMPRESSION_ARTIFACTS — Überkomprimierung, Pumpen, Ducking
    # ------------------------------------------------------------------
    DefectType.COMPRESSION_ARTIFACTS: PhaseAssignment(
        defect_type=DefectType.COMPRESSION_ARTIFACTS,
        primary_phases=[
            "phase_26_dynamic_range_expansion",  # Compander (inverse Kompression)
            "phase_23_spectral_repair",  # Spektrale Codec-Artefakte
        ],
        secondary_phases=[
            "phase_54_transparent_dynamics",  # Transparente Dynamik-Restauration
            # §0a: phase_35_multiband_compression ist Studio-2026-only und darf hier
            # nie stehen — wird in Restoration durch Runtime-Filter abgeblockt, aber
            # §0a verlangt, dass §0a-verbotene Phasen gar nicht erst vorgeschlagen werden.
        ],
        description=(
            "Umkehrung von Dynamik-Überkompression via Upward Expansion. "
            "Behandelt Pumpen, Ducking und Ozone-Klang ohne Wasserfall-Effekt."
        ),
        config_delta={
            "enable_spectral_repair": True,
            "spectral_repair_strength": 0.65,
            "compression_ratio": 1.0,  # Keine weitere Kompression!
            "enable_multiband_compression": False,
            "denoise_strength": 0.10,
            "declip_strength": 0.0,
        },
    ),
    # ------------------------------------------------------------------
    # PHASE_ISSUES — Kanalümkehrung, Azimuth-Fehler, Phasendrift
    # ------------------------------------------------------------------
    DefectType.PHASE_ISSUES: PhaseAssignment(
        defect_type=DefectType.PHASE_ISSUES,
        primary_phases=[
            "phase_14_phase_correction",  # Phase-Inversion-Detektion
            "phase_25_azimuth_correction",  # Azimuth-Fehler bei Bandmaschinen
        ],
        secondary_phases=[
            "phase_34_mid_side_processing",  # M/S für Phasenfehler-Analyse
            "phase_15_stereo_balance",  # Begleitende Balance-Korrektur
        ],
        description=(
            "Phasenfehler-Korrektur: Kanal-Inversion-Detektion, Azimuth-Korrektur "
            "und Mid/Side-basierte Phasen-Ausrichtung."
        ),
        config_delta={
            "denoise_strength": 0.10,
            "click_removal_sensitivity": 0.15,
            "declip_strength": 0.0,
            "stereo_width_factor": 1.0,  # Neutrale Breite sicherstellen
        },
    ),
    # ------------------------------------------------------------------
    # DROPOUTS — Tonaussetzer, magnetische Löschstellen auf Band
    # ------------------------------------------------------------------
    DefectType.DROPOUTS: PhaseAssignment(
        defect_type=DefectType.DROPOUTS,
        primary_phases=[
            "phase_24_dropout_repair",  # Interpolation über Aussetzer
            "phase_50_spectral_repair",  # Spektrale Lücken in Dropout-Zonen
        ],
        secondary_phases=[
            "phase_23_spectral_repair",  # Ergänzende spektrale Reparatur
            "phase_07_harmonic_restoration",  # Teilweise verlorene Harmonische
        ],
        description=(
            "Dropout-Reparatur via adaptiver Interpolation (AR-Modell) + "
            "spektraler Auffüllung verlorener Energie-Regionen."
        ),
        config_delta={
            "enable_spectral_repair": True,
            "spectral_repair_strength": 0.90,  # Hoch: Dropouts = große Löcher
            "spectral_repair_hole_threshold_db": -30.0,  # Aggressivere Detektion
            "denoise_strength": 0.20,
            "click_removal_sensitivity": 0.30,  # Dropout-Grenzen sind klickartig
            "declip_strength": 0.10,
        },
    ),
    # ------------------------------------------------------------------
    # CLIPPING — Amplituden-Übersteuerung (Hard Clipping / Tape Saturation)
    # ------------------------------------------------------------------
    DefectType.CLIPPING: PhaseAssignment(
        defect_type=DefectType.CLIPPING,
        primary_phases=[
            "phase_07_declipper",  # §v10.18: Selbstkalibrierender Declipper (PCHIP + Spline-Adaptiv)
            "phase_23_spectral_repair",  # Spektrale Rekonstruktion clipper Obertöne
        ],
        secondary_phases=[
            "phase_26_dynamic_range_expansion",  # Dynamikbereich nach Clipping wiederherstellen
            "phase_08_transient_preservation",  # Transienten-Schutz während Declipping
        ],
        description=(
            "Amplituden-Übersteuerungs-Korrektur via selbstkalibrierendem Declipper "
            "(PCHIP + Spline-Adaptiv mit adaptiver Crossfade-Glättung). "
            "Kalibriert Clip-Schwelle automatisch per Song-Statistik. "
            "Rekonstruiert abgeschnittene Wellenformspitzen und stellt "
            "harmonische Obertöne wieder her."
        ),
        config_delta={
            "declip_strength": 0.85,  # Primäraktion: Clipping entfernen
            "enable_spectral_repair": True,
            "spectral_repair_strength": 0.70,  # Spektrale Lücken nach Declipping
            "denoise_strength": 0.10,  # Minimal: Rauschen ist Sekundärproblem
            "click_removal_sensitivity": 0.15,  # Clip-Grenzen können klickartig sein
            "compression_ratio": 1.0,  # Keine weitere Kompression!
        },
    ),
    # ------------------------------------------------------------------
    # DC_OFFSET — Gleichspannungsversatz (Nulllinien-Verschiebung)
    # ------------------------------------------------------------------
    DefectType.DC_OFFSET: PhaseAssignment(
        defect_type=DefectType.DC_OFFSET,
        primary_phases=[
            "phase_30_dc_offset_removal",  # Direkte DC-Entfernung via Hochpass-Filter
        ],
        secondary_phases=[
            "phase_05_rumble_filter",  # Sub-Bass kann DC-bedingte Rumpelkomponenten enthalten
            "phase_02_hum_removal",  # DC-begleitende Netzbrumm-Anteile
        ],
        description=(
            "DC-Gleichspannungsversatz-Entfernung via Hochpass-Filter (< 5 Hz). "
            "Beseitigt Nulllinien-Verschiebung die Headroom reduziert, "
            "Verzerrungen verursacht und Lautsprecher beschädigen kann."
        ),
        config_delta={
            "low_freq_rolloff_hz": 5,  # Sehr tiefer HP-Filter: nur DC entfernen
            "denoise_strength": 0.05,  # Minimal: DC-Removal ist kein Rauschproblem
            "click_removal_sensitivity": 0.10,
            "declip_strength": 0.0,
        },
    ),
    # ------------------------------------------------------------------
    # BANDWIDTH_LOSS — HF-Rolloff / Bandbreitenbegrenzung
    # ------------------------------------------------------------------
    DefectType.BANDWIDTH_LOSS: PhaseAssignment(
        defect_type=DefectType.BANDWIDTH_LOSS,
        primary_phases=[
            "phase_06_frequency_restoration",  # HF-Rekonstruktion (Spectral Extension)
            "phase_07_harmonic_restoration",  # Harmonische über Bandgrenze hinaus rekonstruieren
        ],
        secondary_phases=[
            "phase_04_eq_correction",  # Tonal-Shaping nach HF-Restore
        ],
        description=(
            "Hochfrequenz-Restauration für bandbreitenbegrenzte Quellen: "
            "Shellac (< 7 kHz), Kassette (< 14 kHz), MP3-Low (HF-Abschnitt). "
            "Rekonstruiert fehlende Obertöne via spektraler Extrapolation."
        ),
        config_delta={
            "high_freq_boost_db": 3.0,  # HF-Anhebung nach Restauration
            "enable_spectral_repair": True,
            "spectral_repair_strength": 0.60,  # Moderate: Vorsicht bei spektraler Erfindung
            "denoise_strength": 0.15,  # Leicht: HF-Rauschen nicht mit Restauration mischen
            "declip_strength": 0.0,
            "preserve_analog_character": True,  # Charakter bewahren, nicht übertreiben
        },
    ),
    # ------------------------------------------------------------------
    # PITCH_DRIFT — Konstanter Geschwindigkeitsfehler (Tape-Stretch, Motorabweichung)
    # ------------------------------------------------------------------
    DefectType.PITCH_DRIFT: PhaseAssignment(
        defect_type=DefectType.PITCH_DRIFT,
        primary_phases=[
            "phase_31_speed_pitch_correction",  # Konstante Pitch-Korrektur via Time-Stretch
        ],
        secondary_phases=[
            "phase_12_wow_flutter_fix",  # Begleitende Pitch-Instabilitäten mitkorrigieren
        ],
        description=(
            "Korrektur konstanter Tonhöhen-/Geschwindigkeitsfehler: "
            "Tape läuft zu schnell/langsam (z.B. 78rpm→77rpm, Tape-Stretch). "
            "Unterschied zu WOW/FLUTTER: kein periodisches Modulationsmuster, "
            "sondern monotone Frequenzabweichung über die gesamte Aufnahme."
        ),
        config_delta={
            "denoise_strength": 0.10,  # Minimal: Pitch-Fix ist keine Rauschaufgabe
            "click_removal_sensitivity": 0.10,
            "declip_strength": 0.0,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # REVERB_EXCESS — Übermäßiger / unerwünschter Raumhall
    # ------------------------------------------------------------------
    DefectType.REVERB_EXCESS: PhaseAssignment(
        defect_type=DefectType.REVERB_EXCESS,
        primary_phases=[
            "phase_49_advanced_dereverb",  # WPE-basierte Dereverberation (vollst. DSP)
            "phase_20_reverb_reduction",  # Spektral-Gating + Transientenerhalt
        ],
        secondary_phases=[
            "phase_08_transient_preservation",  # Direkt-Sound-Transienten bewahren
            "phase_18_noise_gate",  # Reverb-Schwanz in Pausen unterdrücken
        ],
        description=(
            "Entfernung unerwünschten Raumhalls aus schlecht bedämpften Aufnahmeräumen. "
            "Phase 49 (WPE DSP): Weighted Prediction Error — schätzt Nachhall-Anteil "
            "und subtrahiert ihn spektral. Phase 20: Spektrales Gating für längere Nachhall-Schwänze. "
            "Bewahrt Direkt-Sound und Transienten durch parallele Transientenanalyse."
        ),
        config_delta={
            "denoise_strength": 0.20,
            "click_removal_sensitivity": 0.10,
            "declip_strength": 0.0,
            "enable_spectral_repair": False,  # Kein Lochfüllen — Reverb kein Loch
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # PRINT_THROUGH — Magnetisches Übersprechen auf Tonband (Pre-Echo)
    # ------------------------------------------------------------------
    DefectType.PRINT_THROUGH: PhaseAssignment(
        defect_type=DefectType.PRINT_THROUGH,
        primary_phases=[
            "phase_57_print_through_reduction",  # Bidirektionale LMS-Adaptive Subtraction (Pre+Post-Echo getrennt)
            "phase_23_spectral_repair",  # Spektrale Unterdrückung verbleibender Ghost-Signalanteile
        ],
        secondary_phases=[
            "phase_18_noise_gate",  # Abklingphase des Pre-Echos per Gate unterdrücken
            "phase_08_transient_preservation",  # Echter Einsatz darf nicht angetastet werden
        ],
        description=(
            "Unterdrückung magnetischen Übersprechens (Print-Through) bei Reel-to-Reel- und "
            "Kassetten-Aufnahmen: Leises Vor-Echo (100–400 ms vor dem Einsatz, 20–45 dB unter "
            "dem Hauptsignal). Spektrale Subtraktion + adaptive Gating des Ghost-Signals. "
            "Nur relevant bei TAPE und REEL_TAPE (Threshold=1.0 für alle anderen Materialien)."
        ),
        config_delta={
            "click_removal_sensitivity": 0.50,  # Pre-Echo ist klickartig kurz
            "denoise_strength": 0.15,
            "declip_strength": 0.0,
            "enable_spectral_repair": True,
            "spectral_repair_strength": 0.55,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # QUANTIZATION_NOISE — Quantisierungsrauschen (8-Bit, ATRAC, aggressive Kodierung)
    # ------------------------------------------------------------------
    DefectType.QUANTIZATION_NOISE: PhaseAssignment(
        defect_type=DefectType.QUANTIZATION_NOISE,
        primary_phases=[
            "phase_03_denoise",  # Breitband-Rauschreduktion reduziert Granulationsrauschen
            "phase_23_spectral_repair",  # Spektrale Glättung der Quantisierungsstufen
        ],
        secondary_phases=[
            "phase_18_noise_gate",  # Gate verhindert Rauschen in Stille-Phasen
            "phase_04_eq",  # Equalizer zur Kompensation der Spektralform
        ],
        description=(
            "Reduktion von Quantisierungsrauschen durch spektrale Glättung und "
            "Rauschreduktion. Typisch bei 8-Bit-Konvertierungen, ATRAC (MiniDisc), "
            "aggressivem MP3 Low-Bitrate. Granulationsrauschen in leisen Passagen "
            "wird durch adaptives Denoising + sanftes Noise-Gate eliminiert."
        ),
        config_delta={
            "denoise_strength": 0.40,
            "declip_strength": 0.0,
            "click_removal_sensitivity": 0.0,
            "enable_spectral_repair": True,
            "spectral_repair_strength": 0.35,
            "preserve_analog_character": False,  # Digitales Artefakt — kein Analog-Charakter nötig
        },
    ),
    # ------------------------------------------------------------------
    # JITTER_ARTIFACTS — D/A-Wandler-Jitter (CD-Laufwerk, DAT, Netzwerk-Jitter)
    # ------------------------------------------------------------------
    DefectType.JITTER_ARTIFACTS: PhaseAssignment(
        defect_type=DefectType.JITTER_ARTIFACTS,
        primary_phases=[
            "phase_23_spectral_repair",  # FM-Seitenbänder im HF-Band spektral unterdrücken
            "phase_03_denoise",  # HF-Rauschreduktion eliminiert Jitter-Noise-Floor-Erhöhung
        ],
        secondary_phases=[
            "phase_04_eq",  # HF-Frequenzgangkorrektur nach Jitter-Unterdrückung
            "phase_08_transient_preservation",  # Transienten bleiben unangetastet
        ],
        description=(
            "Reduktion von D/A-Wandler-Jitter-Artefakten: FM-Seitenbänder um Sinustöne, "
            "zeitlich inkonsistente HF-Energie > 8 kHz. Typisch bei schlechten CD-Laufwerken, "
            "DAT-Recordern und Streaming-Puffer-Underruns. Spektrale Glättung der "
            "HF-Seitenband-Energie + adaptives HF-Denoising."
        ),
        config_delta={
            "denoise_strength": 0.35,
            "declip_strength": 0.0,
            "click_removal_sensitivity": 0.0,
            "enable_spectral_repair": True,
            "spectral_repair_strength": 0.45,
            "preserve_analog_character": False,
        },
    ),
    # ------------------------------------------------------------------
    # DYNAMIC_COMPRESSION_EXCESS — Loudness War / übermäßige Dynamikkompression
    # ------------------------------------------------------------------
    DefectType.DYNAMIC_COMPRESSION_EXCESS: PhaseAssignment(
        defect_type=DefectType.DYNAMIC_COMPRESSION_EXCESS,
        primary_phases=[
            "phase_08_transient_preservation",  # Transientenwiederherstellung (Anschläge, Attacks)
            "phase_07_declip",  # Declipping der übersteuerten Peaks
        ],
        secondary_phases=[
            "phase_04_eq",  # Spektrale Balance nach Kompressionsartefakten
            "phase_06_stereo_enhancement",  # Stereofeld-Restauration (Kompression verengt Stereo)
        ],
        description=(
            "Teilrestauration übermäßig komprimierter Aufnahmen (Loudness War, DR < 6 dB). "
            "Vollständige Rückgängigmachung ist ohne Quelldaten nicht möglich — aber: "
            "Transientenwiederherstellung (phase_08) rekonstruiert Attack-Energie, "
            "Declipping (phase_07) hebt begrenzte Peaks an, EQ (phase_04) restauriert "
            "Spektralbalance. Ziel: DR +2 bis +4 dB, subjektiv hörbar weniger 'komprimiert'."
        ),
        config_delta={
            "denoise_strength": 0.10,
            "declip_strength": 0.55,  # Hauptwerkzeug: begrenzte Peaks anheben
            "click_removal_sensitivity": 0.0,
            "enable_spectral_repair": False,
            "preserve_analog_character": True,  # Dynamik ist Teil des analogen Charakters
        },
    ),
    # ------------------------------------------------------------------
    # SOFT_SATURATION — Röhren-/Tape-Sättigung soll erhalten, nicht entfernt werden
    # ------------------------------------------------------------------
    DefectType.SOFT_SATURATION: PhaseAssignment(
        defect_type=DefectType.SOFT_SATURATION,
        primary_phases=[
            "phase_22_tape_saturation",  # Emulation/Erhalt statt Entfernung
        ],
        secondary_phases=[
            "phase_08_transient_preservation",
        ],
        description=(
            "Bewahrt weiche Röhren-/Tape-Sättigung als musikalischen Charakter. "
            "Keine destruktive Reparatur, sondern konservative Charaktererhaltung."
        ),
        config_delta={
            "denoise_strength": 0.0,
            "declip_strength": 0.0,
            "click_removal_sensitivity": 0.0,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # HEAD_WEAR — Frequenzband-Auslöschung durch Kopfverschleiß
    # ------------------------------------------------------------------
    DefectType.HEAD_WEAR: PhaseAssignment(
        defect_type=DefectType.HEAD_WEAR,
        primary_phases=[
            "phase_56_spectral_band_gap_repair",
            "phase_06_frequency_restoration",
        ],
        secondary_phases=[
            "phase_14_phase_correction",
            "phase_25_azimuth_correction",
        ],
        description=(
            "Repariert spektrale Bandlücken und Höhenausfälle durch Kopfverschleiß oder Kontaktprobleme im Bandpfad."
        ),
        config_delta={
            "enable_spectral_repair": True,
            "spectral_repair_strength": 0.85,
            "high_freq_boost_db": 2.0,
            "denoise_strength": 0.15,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # AZIMUTH_ERROR — HF-Phasen-Slope / Kopf-Schrägstellung
    # ------------------------------------------------------------------
    DefectType.AZIMUTH_ERROR: PhaseAssignment(
        defect_type=DefectType.AZIMUTH_ERROR,
        primary_phases=[
            "phase_25_azimuth_correction",
            "phase_14_phase_correction",
        ],
        secondary_phases=[
            "phase_34_mid_side_processing",
            "phase_15_stereo_balance",
        ],
        description=(
            "Korrigiert Azimuth-Fehler und daraus resultierende Hochton- und Phasenprobleme "
            "zwischen linkem und rechtem Kanal."
        ),
        config_delta={
            "denoise_strength": 0.05,
            "declip_strength": 0.0,
            "click_removal_sensitivity": 0.05,
            "stereo_width_factor": 1.0,
        },
    ),
    # ------------------------------------------------------------------
    # TRANSIENT_SMEARING — verschmierte Anschläge/Attacks
    # ------------------------------------------------------------------
    DefectType.TRANSIENT_SMEARING: PhaseAssignment(
        defect_type=DefectType.TRANSIENT_SMEARING,
        primary_phases=[
            "phase_08_transient_preservation",
            "phase_36_transient_shaper",
        ],
        secondary_phases=[
            "phase_23_spectral_repair",
        ],
        description=(
            "Restauriert verschmierte Transienten und Attack-Definition nach Kompression, "
            "Limiter-Artefakten oder Codec-Verschleifung."
        ),
        config_delta={
            "denoise_strength": 0.05,
            "declip_strength": 0.10,
            "click_removal_sensitivity": 0.0,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # PRE_ECHO — Codec-Pre-Echo vor Transienten
    # ------------------------------------------------------------------
    DefectType.PRE_ECHO: PhaseAssignment(
        defect_type=DefectType.PRE_ECHO,
        primary_phases=[
            "phase_23_spectral_repair",
            "phase_50_spectral_repair",
        ],
        secondary_phases=[
            "phase_08_transient_preservation",
        ],
        description=(
            "Unterdrückt Codec-Pre-Echo vor Transienten durch spektrale Reparatur, "
            "ohne den eigentlichen Einschwingvorgang zu verwischen."
        ),
        config_delta={
            "enable_spectral_repair": True,
            "spectral_repair_strength": 0.70,
            "denoise_strength": 0.10,
            "click_removal_sensitivity": 0.0,
        },
    ),
    # ------------------------------------------------------------------
    # RIAA_CURVE_ERROR — falsche Disc-Entzerrung
    # ------------------------------------------------------------------
    DefectType.RIAA_CURVE_ERROR: PhaseAssignment(
        defect_type=DefectType.RIAA_CURVE_ERROR,
        primary_phases=[
            "phase_04_eq_correction",
            "phase_06_frequency_restoration",
        ],
        secondary_phases=[
            "phase_07_harmonic_restoration",
        ],
        description=("Korrigiert falsche Entzerrungskurven bei Disc-Transfers (RIAA/AES/NAB/FFRR/Columbia)."),
        config_delta={
            "high_freq_boost_db": 2.5,
            "denoise_strength": 0.10,
            "declip_strength": 0.0,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # ALIASING — Spiegelartefakte durch unzureichenden AA-Filter
    # ------------------------------------------------------------------
    DefectType.ALIASING: PhaseAssignment(
        defect_type=DefectType.ALIASING,
        primary_phases=[
            "phase_23_spectral_repair",
        ],
        secondary_phases=[
            "phase_50_spectral_repair",
        ],
        description=(
            "Reduziert Aliasing-Spiegelfrequenzen aus fehlerhafter Digitalisierung "
            "durch spektrale Chirurgie. NR ist kontraindiziert — Spiegelfrequenzen sind "
            "kohärente Signalspiegelungen, kein Rauschen (§4.11, V30)."
        ),
        config_delta={
            "enable_spectral_repair": True,
            "spectral_repair_strength": 0.55,
            "declip_strength": 0.0,
        },
    ),
    # ------------------------------------------------------------------
    # BIAS_ERROR — falscher Vormagnetisierungsstrom bei Bandaufnahme
    # ------------------------------------------------------------------
    DefectType.BIAS_ERROR: PhaseAssignment(
        defect_type=DefectType.BIAS_ERROR,
        primary_phases=[
            "phase_04_eq_correction",
            "phase_29_tape_hiss_reduction",
        ],
        secondary_phases=[
            "phase_06_frequency_restoration",
            "phase_03_denoise",
        ],
        description=(
            "Kompensiert Bias-Fehler bei Bandaufnahmen, die zu unausgewogenem Frequenzgang, "
            "Hiss und Höhenverlust führen."
        ),
        config_delta={
            "denoise_strength": 0.30,
            "high_freq_boost_db": 1.5,
            "declip_strength": 0.0,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # SIBILANCE — überbetonte Zischlaute
    # ------------------------------------------------------------------
    DefectType.SIBILANCE: PhaseAssignment(
        defect_type=DefectType.SIBILANCE,
        primary_phases=[
            "phase_19_de_esser",
            "phase_43_ml_deesser",
        ],
        secondary_phases=[
            "phase_38_presence_boost",
        ],
        description=(
            "Kontrolliert überbetonte Sibilanten stimmtyp-adaptiv, ohne Sprachverständlichkeit "
            "oder Präsenz pauschal zu beschneiden."
        ),
        config_delta={
            "denoise_strength": 0.05,
            "declip_strength": 0.0,
            "click_removal_sensitivity": 0.0,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # TRANSPORT_BUMP — impulsive Mikro-Geschwindigkeitssprünge (50–300 ms)
    # durch mechanische Transporterschütterungen (Kassette/Bandaufnahmen)
    # ------------------------------------------------------------------
    DefectType.TRANSPORT_BUMP: PhaseAssignment(
        defect_type=DefectType.TRANSPORT_BUMP,
        primary_phases=[
            "phase_12_wow_flutter_fix",  # lokale PSOLA-Pitch-Glättung + Envelope-Morphing
            "phase_24_dropout_repair",  # Aussetzer-Interpolation an Bump-Stellen
            "phase_31_speed_pitch_correction",  # Savitzky-Golay Pitch-Korrektur
        ],
        secondary_phases=[
            "phase_08_transient_preservation",  # Schützt Musik-Transienten bei Bump-Reparatur
            "phase_03_denoise",  # Restgeräusche an Bump-Kanten
        ],
        description=(
            "Repariert impulsive Mikro-Geschwindigkeitssprünge (50–300 ms) durch "
            "mechanische Transporterschütterungen bei Kassetten- und Bandaufnahmen. "
            "Unterscheidet sich von kontinuierlichem Wow/Flutter (< 4 Hz) und Dropouts."
        ),
        config_delta={
            "bump_correction_strength": 0.80,
            "crossfade_ms": 15.0,
            "envelope_smoothing": 0.70,
            "denoise_strength": 0.15,
            "click_removal_sensitivity": 0.2,
            "declip_strength": 0.0,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # VOCAL_HARSHNESS — Vokale Härte/Übersteuerung/Kratzigkeit im 2–6 kHz Band
    # Quellen: Bandsättigung, Loudness-War-Mastering, Röhren-/Transistorverzerrung,
    #          Trichter-/Tonabnehmer-Resonanz, Codec-Artefakte
    # ------------------------------------------------------------------
    DefectType.VOCAL_HARSHNESS: PhaseAssignment(
        defect_type=DefectType.VOCAL_HARSHNESS,
        primary_phases=[
            # §0a: phase_42_vocal_enhancement ist Studio-2026-only — in Restoration
            # ist phase_65_vocal_naturalness_restoration der §0a-konforme Ersatz
            # (HNR-Blend + Spektral-Tilt + Formant-Tilt). §0a: phase_42 darf hier
            # nie stehen (V04-BUG-FIX v10.0.0).
            "phase_65_vocal_naturalness_restoration",  # §0a-konformer Restoration-Ersatz (HNR-Blend, Formant-Tilt)
            "phase_19_de_esser",  # De-Essing / De-Harshness im 2–6 kHz Band
        ],
        secondary_phases=[
            "phase_43_ml_deesser",  # ML-De-Esser: Phonem-selektive Harshness-Reduktion
            "phase_58_lyrics_guided_enhancement",  # §2.36 PFLICHT: Phonem-Alignment → De-Essing konsonant-selektiv
            "phase_03_denoise",  # Restgeräusche nach Vokal-Verarbeitung
        ],
        description=(
            "Reduziert vokale Härte, Übersteuerung und Kratzigkeit im kritischen 2–6 kHz Band "
            "durch stem-separierte Vokalverarbeitung (BSRoFormer) mit anschließendem "
            "De-Essing/De-Harshness, ohne Vokal-Präsenz und Konsonantenklarheit zu opfern."
        ),
        config_delta={
            "denoise_strength": 0.10,
            "declip_strength": 0.15,
            "click_removal_sensitivity": 0.0,
            "preserve_analog_character": True,
            "de_essing_strength": 0.65,
            "harshness_reduction_band_hz": [2000, 6000],
        },
    ),
    # ------------------------------------------------------------------
    # DOLBY_NR_MISMATCH — falsche Dolby-De/Enkodierungs-Kette
    # ------------------------------------------------------------------
    DefectType.DOLBY_NR_MISMATCH: PhaseAssignment(
        defect_type=DefectType.DOLBY_NR_MISMATCH,
        primary_phases=[
            "phase_04_eq_correction",
            "phase_14_phase_correction",
        ],
        secondary_phases=[
            "phase_06_frequency_restoration",
        ],
        description=(
            "Kompensiert Dolby-NR-Fehlanwendung (Encode/Decode-Mismatch) mit spektraler und phasenbezogener "
            "Korrektur, um HF-Überbetonung und Fehlfärbungen zu stabilisieren."
        ),
        config_delta={
            "denoise_strength": 0.10,
            "high_freq_boost_db": 1.2,
            "declip_strength": 0.0,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # TAPE_HEAD_LEVEL_DIP — graduelle Bandkopf-Pegeleinbrüche
    # ------------------------------------------------------------------
    DefectType.TAPE_HEAD_LEVEL_DIP: PhaseAssignment(
        defect_type=DefectType.TAPE_HEAD_LEVEL_DIP,
        primary_phases=[
            "phase_54_transparent_dynamics",
            "phase_24_dropout_repair",
        ],
        secondary_phases=[
            "phase_03_denoise",
        ],
        description=(
            "Stabilisiert langsam driftende Pegeleinbrüche aus Kopfkontakt-Variationen per timing-adaptiver "
            "Korrektur und segmentweiser Rekonstruktion."
        ),
        config_delta={
            "denoise_strength": 0.20,
            "click_removal_sensitivity": 0.10,
            "declip_strength": 0.0,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # SCRAPE_FLUTTER — hochfrequente Bandführungs-/Reibungsmodulation
    # ------------------------------------------------------------------
    DefectType.SCRAPE_FLUTTER: PhaseAssignment(
        defect_type=DefectType.SCRAPE_FLUTTER,
        primary_phases=[
            "phase_12_wow_flutter_fix",
            "phase_31_speed_pitch_correction",
        ],
        secondary_phases=[
            "phase_08_transient_preservation",
        ],
        description=(
            "Korrigiert hochfrequente Transportmodulationen aus Bandführung, Reibung oder Kopfkontakt über "
            "flutter-spezifische Zeitachsenstabilisierung mit nachgelagerter Feinkalibrierung."
        ),
        config_delta={
            "denoise_strength": 0.0,
            "declip_strength": 0.0,
            "preserve_analog_character": True,
            "timing_correction_strength": 0.40,
        },
    ),
    # ------------------------------------------------------------------
    # TAPE_HEAD_CLOG — temporäre HF-Auslöschung durch verschmutzten Kopf
    # ------------------------------------------------------------------
    DefectType.TAPE_HEAD_CLOG: PhaseAssignment(
        defect_type=DefectType.TAPE_HEAD_CLOG,
        primary_phases=[
            "phase_56_spectral_band_gap_repair",
            "phase_25_azimuth_correction",
        ],
        secondary_phases=[
            "phase_24_dropout_repair",
        ],
        description=(
            "Repariert lokale Hochton-Auslöschungen durch zugesetzte oder verschmutzte Magnetköpfe mit "
            "bandgap-orientierter Spektralrekonstruktion und Kopfgeometrie-Nachkorrektur."
        ),
        config_delta={
            "denoise_strength": 0.0,
            "declip_strength": 0.0,
            "preserve_analog_character": True,
            "spectral_repair_strength": 0.55,
        },
    ),
    # ------------------------------------------------------------------
    # MODULATION_NOISE — signalabhängiges Bandrauschen
    # ------------------------------------------------------------------
    DefectType.MODULATION_NOISE: PhaseAssignment(
        defect_type=DefectType.MODULATION_NOISE,
        primary_phases=[
            "phase_59_modulation_noise_reduction",
            "phase_03_denoise",
        ],
        secondary_phases=[
            "phase_29_tape_hiss_reduction",
            "phase_65_vocal_naturalness_restoration",  # §7.10 VQI-Korrektiv bei signaladaptivem Rauschen
        ],
        description=(
            "Reduziert signalmoduliertes Rauschen in Bandmaterial über adaptives Spektral-Gating plus "
            "breitbandige Rest-Rauschreduktion."
        ),
        config_delta={
            "denoise_strength": 0.45,
            "declip_strength": 0.0,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # INNER_GROOVE_DISTORTION — innenrillenbedingte Verzerrung
    # ------------------------------------------------------------------
    DefectType.INNER_GROOVE_DISTORTION: PhaseAssignment(
        defect_type=DefectType.INNER_GROOVE_DISTORTION,
        primary_phases=[
            "phase_60_inner_groove_distortion_repair",
            "phase_23_spectral_repair",
        ],
        secondary_phases=[
            "phase_04_eq_correction",
        ],
        description=(
            "Korrigiert positionsabhängige Innenrillen-Verzerrungen bei Rillenmedien mit THD-orientierter "
            "Reparatur und spektraler Nachglättung."
        ),
        config_delta={
            "denoise_strength": 0.20,
            "enable_spectral_repair": True,
            "spectral_repair_strength": 0.60,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # GROOVE_ECHO — Rillen-Pre-Echo
    # ------------------------------------------------------------------
    DefectType.GROOVE_ECHO: PhaseAssignment(
        defect_type=DefectType.GROOVE_ECHO,
        primary_phases=[
            "phase_61_groove_echo_cancellation",
            "phase_03_denoise",
        ],
        secondary_phases=[
            "phase_23_spectral_repair",
        ],
        description=(
            "Unterdrückt rillenbedingtes Pre-Echo per RPM-adaptiver Cancelation und reduziert verbleibende "
            "Ghost-Energie in kritischen Passagen."
        ),
        config_delta={
            "denoise_strength": 0.35,
            "enable_spectral_repair": True,
            "spectral_repair_strength": 0.50,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # CROSSTALK — Kanalübersprechen
    # ------------------------------------------------------------------
    DefectType.CROSSTALK: PhaseAssignment(
        defect_type=DefectType.CROSSTALK,
        primary_phases=[
            "phase_62_crosstalk_cancellation",
            "phase_15_stereo_balance",
        ],
        secondary_phases=[
            "phase_34_mid_side_processing",
        ],
        description=(
            "Reduziert kanalübergreifendes Übersprechen über BSS-/Dekorrelationstechniken und stabilisiert "
            "die resultierende Stereo-Balance."
        ),
        config_delta={
            "stereo_width_factor": 0.75,
            "denoise_strength": 0.05,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # INTERMODULATION_DISTORTION — nichtlineare Mischprodukte
    # ------------------------------------------------------------------
    DefectType.INTERMODULATION_DISTORTION: PhaseAssignment(
        defect_type=DefectType.INTERMODULATION_DISTORTION,
        primary_phases=[
            "phase_63_intermodulation_reduction",
            "phase_23_spectral_repair",
        ],
        secondary_phases=[
            "phase_04_eq_correction",
        ],
        description=(
            "Dämpft IMD-Mischprodukte aus nichtlinearen Ketten mit modellierter Reduktion und spektraler "
            "Rekonsolidierung."
        ),
        config_delta={
            "denoise_strength": 0.20,
            "enable_spectral_repair": True,
            "spectral_repair_strength": 0.55,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # TAPE_SPLICE_ARTIFACT — Bandschnitt-Artefakte
    # ------------------------------------------------------------------
    DefectType.TAPE_SPLICE_ARTIFACT: PhaseAssignment(
        defect_type=DefectType.TAPE_SPLICE_ARTIFACT,
        primary_phases=[
            "phase_64_tape_splice_repair",
            "phase_24_dropout_repair",
        ],
        secondary_phases=[
            "phase_01_click_removal",
        ],
        description=(
            "Behandelt Klebestellen-Artefakte (Klick, Pegelsprung, Phasendiskontinuität) mit splice-spezifischer "
            "Reparatur und lokaler Rekonstruktion."
        ),
        config_delta={
            "click_removal_sensitivity": 0.55,
            "denoise_strength": 0.10,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # HF_REMANENCE_LOSS — magnetische HF-Alterung
    # ------------------------------------------------------------------
    DefectType.HF_REMANENCE_LOSS: PhaseAssignment(
        defect_type=DefectType.HF_REMANENCE_LOSS,
        primary_phases=[
            "phase_06_frequency_restoration",
            "phase_07_harmonic_restoration",
        ],
        secondary_phases=[
            "phase_39_air_band_enhancement",
        ],
        description=(
            "Rekonstruiert alterungsbedingt verlorene Höhen und Obertöne bei Bandremanenz-Verlust mit "
            "vorsichtiger Air-Band-Nachführung."
        ),
        config_delta={
            "denoise_strength": 0.10,
            "high_freq_boost_db": 2.0,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # STYLUS_DAMAGE — nadelinduzierte Verzerrung
    # ------------------------------------------------------------------
    DefectType.STYLUS_DAMAGE: PhaseAssignment(
        defect_type=DefectType.STYLUS_DAMAGE,
        primary_phases=[
            "phase_09_crackle_removal",
            "phase_23_spectral_repair",
        ],
        secondary_phases=[
            "phase_50_spectral_repair",
        ],
        description=(
            "Mildert asymmetrische Verzerrungen und mikroskopische Rillenartefakte aus Nadelschäden mit "
            "zweistufiger Spektralreparatur."
        ),
        config_delta={
            "denoise_strength": 0.20,
            "enable_spectral_repair": True,
            "spectral_repair_strength": 0.60,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # STICKY_SHED_RESIDUE — binderbedingte Tape-Residuen
    # ------------------------------------------------------------------
    DefectType.STICKY_SHED_RESIDUE: PhaseAssignment(
        defect_type=DefectType.STICKY_SHED_RESIDUE,
        primary_phases=[
            "phase_24_dropout_repair",
            "phase_29_tape_hiss_reduction",
        ],
        secondary_phases=[
            "phase_03_denoise",
        ],
        description=(
            "Kompensiert kurze Pegel- und Spektralstörungen aus Sticky-Shed-Rückständen mit Dropout-Repair "
            "und tape-spezifischer Hiss-Reduktion."
        ),
        config_delta={
            "denoise_strength": 0.40,
            "click_removal_sensitivity": 0.20,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # MULTIBAND_WOW_FLUTTER — frequenzabhängige Gleichlaufschwankung
    # ------------------------------------------------------------------
    DefectType.MULTIBAND_WOW_FLUTTER: PhaseAssignment(
        defect_type=DefectType.MULTIBAND_WOW_FLUTTER,
        primary_phases=[
            "phase_12_wow_flutter_fix",
        ],
        secondary_phases=[
            "phase_08_transient_preservation",
            "phase_31_speed_pitch_correction",
        ],
        description=(
            "Korrigiert frequenzabhängige Wow/Flutter-Komponenten und schützt Transienten nach Pitch-Zeit-Korrektur."
        ),
        config_delta={
            "denoise_strength": 0.10,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # GENERATION_LOSS — kumulativer Kopierverlust
    # ------------------------------------------------------------------
    DefectType.GENERATION_LOSS: PhaseAssignment(
        defect_type=DefectType.GENERATION_LOSS,
        primary_phases=[
            "phase_03_denoise",
            "phase_06_frequency_restoration",
            "phase_23_spectral_repair",
        ],
        secondary_phases=[
            "phase_07_harmonic_restoration",
            "phase_58_lyrics_guided_enhancement",  # §2.36 PFLICHT: Phonem-Alignment
            # nach Mehrfach-NR (Konsonanten-Schutz)
            "phase_65_vocal_naturalness_restoration",  # §7.10 VQI-Korrektiv nach Mehrfach-NR (Formant-Tilt)
        ],
        description=(
            "Adressiert Mehrgenerationen-Verluste (Rauschen, Bandbreite, Spektral-Glättung) durch kombinierte "
            "NR-, Restore- und Spektralstrategie."
        ),
        config_delta={
            "denoise_strength": 0.45,
            "enable_spectral_repair": True,
            "spectral_repair_strength": 0.60,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # MOTOR_INTERFERENCE — mechanische Motorstörungen
    # ------------------------------------------------------------------
    DefectType.MOTOR_INTERFERENCE: PhaseAssignment(
        defect_type=DefectType.MOTOR_INTERFERENCE,
        primary_phases=[
            "phase_02_hum_removal",
            "phase_05_rumble_filter",
        ],
        secondary_phases=[
            "phase_04_eq_correction",
        ],
        description=(
            "Entfernt motorinduzierte Interferenzanteile im Tiefton-/Brummbereich und glättet den verbleibenden "
            "Frequenzgang."
        ),
        config_delta={
            "denoise_strength": 0.15,
            "low_freq_rolloff_hz": 30,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # AMPLITUDE_DRIFT — langsame Pegeländerungen durch Bandoxid/AGC-Drift
    # ------------------------------------------------------------------
    DefectType.AMPLITUDE_DRIFT: PhaseAssignment(
        defect_type=DefectType.AMPLITUDE_DRIFT,
        primary_phases=[
            "phase_40_loudness_normalization",
        ],
        secondary_phases=[
            "phase_12_wow_flutter_fix",
            "phase_29_tape_hiss_reduction",
        ],
        description=(
            "Korrigiert langsame Amplitudendriften durch adaptive Pegelanpassung. "
            "Tritt bei Bandoxid-Degradation, AGC-Schaltungsalterung und Temperatureinflüssen auf."
        ),
        config_delta={
            "denoise_strength": 0.10,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # PROXIMITY_EFFECT_EXCESS — Nahbesprechungseffekt (Richtmikrofon, LF-Überhöhung)
    # ------------------------------------------------------------------
    DefectType.PROXIMITY_EFFECT_EXCESS: PhaseAssignment(
        defect_type=DefectType.PROXIMITY_EFFECT_EXCESS,
        primary_phases=[
            "phase_04_eq_correction",
            "phase_05_rumble_filter",
        ],
        secondary_phases=[
            "phase_03_denoise",
        ],
        description=(
            "Korrigiert tieffrequente Überhöhung durch Nahbesprechungseffekt bei Richtmikrofonen. "
            "Tieftonabsenkung < 200 Hz, Frequenzgang-Restaurierung."
        ),
        config_delta={
            "low_freq_rolloff_hz": 80,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # ROOM_MODE_RESONANCE — Stehwellen-Resonanzen 40–200 Hz
    # §V31: phase_04_eq_correction (Notch-EQ) MUSS Primary sein; phase_05 nur Tertiary (Sub-Bass)
    # ------------------------------------------------------------------
    DefectType.ROOM_MODE_RESONANCE: PhaseAssignment(
        defect_type=DefectType.ROOM_MODE_RESONANCE,
        primary_phases=[
            "phase_04_eq_correction",  # §V31: Notch-EQ first — schmalbandige Kerbfilter für Raumresonanzen
        ],
        secondary_phases=[
            "phase_16_final_eq",  # §V31: Secondary EQ-Korrektur
            "phase_05_rumble_filter",  # §V31: nur Tertiary (Sub-Bass-Rolloff)
        ],
        description=(
            "Reduziert Raummodenresonanzen durch selektive Tieftonkorrektur und Frequenzgang-Entzerrung. "
            "Schmalbandige Kerbfilter im 40–200 Hz Bereich."
        ),
        config_delta={
            "low_freq_rolloff_hz": 50,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # NR_BREATHING_ARTIFACT — Dolby/dbx NR Pumpen-/Atemgeräusche
    # ------------------------------------------------------------------
    DefectType.NR_BREATHING_ARTIFACT: PhaseAssignment(
        defect_type=DefectType.NR_BREATHING_ARTIFACT,
        primary_phases=[
            "phase_54_transparent_dynamics",
            "phase_08_transient_preservation",
        ],
        secondary_phases=[
            "phase_58_lyrics_guided_enhancement",  # §2.36 PFLICHT: Phonem-Boundary-Schutz nach Dynamics
            "phase_65_vocal_naturalness_restoration",  # §7.10 VQI-Korrektiv: Naturalness nach Envelope-Fix
        ],
        description=(
            "Korrigiert NR-Pumpen/Atmen (Dolby B/C/S, dbx) durch Envelope-Re-Smoothing (gain_smooth_ms). "
            "KEIN phase_03/phase_29 \u2014 weiteres NR auf NR-Artefakt verst\u00e4rkt das Pumpen (\u00a74.11, V28)."
        ),
        config_delta={
            "gain_smooth_ms": 200.0,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # FLUTTER_SPECTRAL_SIDEBANDS — Flutter-Seitenbänder um tonale Peaks
    # ------------------------------------------------------------------
    DefectType.FLUTTER_SPECTRAL_SIDEBANDS: PhaseAssignment(
        defect_type=DefectType.FLUTTER_SPECTRAL_SIDEBANDS,
        primary_phases=[
            "phase_12_wow_flutter_fix",
            "phase_23_spectral_repair",
        ],
        secondary_phases=[
            "phase_08_transient_preservation",
        ],
        description=(
            "Entfernt Flutter-induzierte Seitenbänder um tonale Frequenzanteile. "
            "Kombiniert Wow/Flutter-Korrektur mit spektraler Reparatur."
        ),
        config_delta={
            "wow_flutter_strength": 0.55,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # SPEED_CALIBRATION_ERROR — konstanter Geschwindigkeitsfehler
    # ------------------------------------------------------------------
    DefectType.SPEED_CALIBRATION_ERROR: PhaseAssignment(
        defect_type=DefectType.SPEED_CALIBRATION_ERROR,
        primary_phases=[
            "phase_12_wow_flutter_fix",
            "phase_31_speed_pitch_correction",
        ],
        secondary_phases=[
            "phase_25_pitch_correction",
        ],
        description=(
            "Korrigiert konstanten Geschwindigkeitsfehler durch globale Pitch-Verschiebung. "
            "Erkennbar durch global verschobene Grundtonlage bei stabilem Pitch."
        ),
        config_delta={
            "pitch_shift_semitones": 0.0,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # OVERLOAD_DISTORTION — analoger Preamp/Console-Klirr H3/H5
    # ------------------------------------------------------------------
    DefectType.OVERLOAD_DISTORTION: PhaseAssignment(
        defect_type=DefectType.OVERLOAD_DISTORTION,
        primary_phases=[
            "phase_09_crackle_removal",
            "phase_23_spectral_repair",
        ],
        secondary_phases=[
            "phase_14_phase_correction",
        ],
        description=(
            "Reduziert analoge Überlastungsverzerrung (H2/H3/H5 Harmonische). "
            "phase_09 für asymmetrische Wellenformreparatur, phase_23 für harmonische Klirr-Produkte. "
            "KEIN phase_63 — Harmonische ≠ Intermodulationsprodukte (§4.11, V29)."
        ),
        config_delta={
            "denoise_strength": 0.10,
            "declip_strength": 0.20,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # LACQUER_DISC_DEGRADATION — Acetat-Zersetzung, Substrate-Cracking
    # ------------------------------------------------------------------
    DefectType.LACQUER_DISC_DEGRADATION: PhaseAssignment(
        defect_type=DefectType.LACQUER_DISC_DEGRADATION,
        primary_phases=[
            "phase_01_click_removal",
            "phase_09_crackle_removal",
            "phase_03_denoise",
        ],
        secondary_phases=[
            "phase_06_frequency_restoration",
        ],
        description=(
            "Restauriert Lacquer-Disc-Degradation: Acetat-Zersetzung, Substrat-Risse, Oxidation. "
            "Klickentfernung + Kratzbereinigung + Hochtonwiederherstellung."
        ),
        config_delta={
            "click_removal_sensitivity": 0.75,
            "denoise_strength": 0.55,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # Tier 2: MPEG_FRAME_LOSS — MP3/AAC-Frame-Verluste
    # ------------------------------------------------------------------
    DefectType.MPEG_FRAME_LOSS: PhaseAssignment(
        defect_type=DefectType.MPEG_FRAME_LOSS,
        primary_phases=[
            "phase_23_spectral_repair",
            "phase_06_frequency_restoration",
        ],
        secondary_phases=[
            "phase_39_air_band_enhancement",
        ],
        description=(
            "Repariert MPEG-Frame-Verluste durch spektrale Inpainting der Brickwall-Cutoff-Zonen "
            "und Wiederherstellung verlorener Hochfrequenz-Anteile."
        ),
        config_delta={
            "enable_spectral_repair": True,
            "spectral_repair_strength": 0.65,
            "preserve_analog_character": False,  # Digital → keine analoge Patina
        },
    ),
    # ------------------------------------------------------------------
    # Tier 2: STEREO_FIELD_COLLAPSE — progressiver Stereofeld-Kollaps
    # ------------------------------------------------------------------
    DefectType.STEREO_FIELD_COLLAPSE: PhaseAssignment(
        defect_type=DefectType.STEREO_FIELD_COLLAPSE,
        primary_phases=[
            "phase_13_stereo_enhancement",
            "phase_34_mid_side_processing",
        ],
        secondary_phases=[
            "phase_15_stereo_balance",
        ],
        description=(
            "Stellt kollabiertes Stereofeld wieder her durch MS-Dekorrelation und selektive "
            "Stereobreiten-Erweiterung in kollabierten Passagen."
        ),
        config_delta={
            "stereo_width": 1.35,
            "mid_side_balance": 0.60,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # Tier 2: PHASE_ROTATION — unnatürliche Allpass-Filter-Phasenrotation
    # ------------------------------------------------------------------
    DefectType.PHASE_ROTATION: PhaseAssignment(
        defect_type=DefectType.PHASE_ROTATION,
        primary_phases=[
            "phase_14_phase_correction",
            "phase_23_spectral_repair",
        ],
        secondary_phases=[
            "phase_08_transient_preservation",
        ],
        description=(
            "Korrigiert unnatürliche Phasenrotation durch adaptive Allpass-Filter-Inversion. "
            "Stellt kohärente Gruppenlaufzeit über Frequenzbänder wieder her."
        ),
        config_delta={
            "phase_correction_strength": 0.55,
            "enable_spectral_repair": True,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # Tier 2: DROPOUT_OXIDE — kurzer Oxid-Dropout (2–20 ms)
    # ------------------------------------------------------------------
    DefectType.DROPOUT_OXIDE: PhaseAssignment(
        defect_type=DefectType.DROPOUT_OXIDE,
        primary_phases=[
            "phase_24_dropout_repair",
        ],
        secondary_phases=[
            "phase_55_diffusion_inpainting",
        ],
        description=(
            "Repariert kurze Oxid-Dropouts (2–20 ms, 30–70% Pegelverlust) per Waveform-Interpolation. "
            "Bewahrt transiente Musik-Pegel durch zeitlich begrenzte Rekonstruktion."
        ),
        config_delta={
            "dropout_repair_mode": "interpolation",
            "interpolation_window_ms": 10.0,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # Tier 2: DROPOUT_HEAD_CONTACT — längerer Kopf-Kontakt-Dropout (50–200 ms)
    # ------------------------------------------------------------------
    DefectType.DROPOUT_HEAD_CONTACT: PhaseAssignment(
        defect_type=DefectType.DROPOUT_HEAD_CONTACT,
        primary_phases=[
            "phase_24_dropout_repair",
            "phase_56_head_wear_compensation",
        ],
        secondary_phases=[
            "phase_55_diffusion_inpainting",
        ],
        description=(
            "Kompensiert längere Kopf-Kontakt-Dropouts (50–200 ms, modulierter Pegelverlauf) "
            "durch Gain-Kompensation mit adaptivem Envelope-Tracking."
        ),
        config_delta={
            "dropout_repair_mode": "gain_compensation",
            "gain_smooth_ms": 50.0,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # Tier 2: DROPOUT_SPLICE — abrupter Band-Spleiß-Dropout (>95% Pegelverlust)
    # ------------------------------------------------------------------
    DefectType.DROPOUT_SPLICE: PhaseAssignment(
        defect_type=DefectType.DROPOUT_SPLICE,
        primary_phases=[
            "phase_64_tape_splice_repair",
            "phase_23_spectral_repair",
        ],
        secondary_phases=[
            "phase_24_dropout_repair",
        ],
        description=(
            "Repariert abrupter Band-Spleiß-Dropouts (>95% Pegelverlust) mit spektralem Inpainting. "
            "Rekonstruiert fehlendes Signal aus umliegenden Spektralregionen."
        ),
        config_delta={
            "dropout_repair_mode": "spectral_inpainting",
            "spectral_repair_strength": 0.85,
            "preserve_analog_character": True,
        },
    ),
    # ------------------------------------------------------------------
    # §DefectType-Konsolidierung: Legacy-Einträge für ai_framework / data_models Kompatibilität
    # ------------------------------------------------------------------
    DefectType.HISS: PhaseAssignment(
        defect_type=DefectType.HISS,
        primary_phases=[
            "phase_03_denoise",  # Breitbandrauschen entfernen
            "phase_29_tape_hiss_reduction",  # Tape-Hiss-spezifisch
        ],
        secondary_phases=[
            "phase_18_noise_gate",  # Rauschgating nach Denoise
        ],
        description=(
            "Reduziert Tape-Hiss und Breitbandrauschen über adaptives Spektral-Gating plus "
            "Tape-spezifische Hiss-Modellierung."
        ),
        config_delta={
            "denoise_strength": 0.50,
            "declip_strength": 0.0,
            "preserve_analog_character": True,
        },
    ),
    DefectType.DISTORTION: PhaseAssignment(
        defect_type=DefectType.DISTORTION,
        primary_phases=[
            "phase_17_declip",  # Clipping/Verzerrung reparieren
            "phase_23_spectral_repair",  # Spektrale Reparatur nach Declip
        ],
        secondary_phases=[
            "phase_04_eq_correction",  # EQ-Korrektur nach Verzerrungsreparatur
        ],
        description=(
            "Repariert generische Audio-Verzerrung (Clipping, Overdrive) über Declip-Algorithmen "
            "und spektrale Inpainting."
        ),
        config_delta={
            "declip_strength": 0.70,
            "denoise_strength": 0.0,
            "preserve_analog_character": True,
            "spectral_repair_strength": 0.50,
        },
    ),
    DefectType.DROPOUT: PhaseAssignment(
        defect_type=DefectType.DROPOUT,
        primary_phases=[
            "phase_24_dropout_repair",  # Dropout-Reparatur (Spektral-Inpainting)
        ],
        secondary_phases=[
            "phase_03_denoise",  # Rauschen nach Reparatur entfernen
            "phase_18_noise_gate",  # Rauschgating
        ],
        description=(
            "Repariert Tape-/Digital-Dropouts über spektrales Inpainting und Interpolation. "
            "Bewahrt musikalische Kontinuität."
        ),
        config_delta={
            "dropout_repair_mode": "spectral_inpainting",
            "denoise_strength": 0.30,
            "declip_strength": 0.0,
            "preserve_analog_character": True,
        },
    ),
}

# ---------------------------------------------------------------------------
# Material-adaptive Phase-Initialstärken (§2.29 / §2.31)
# ---------------------------------------------------------------------------
# Maps MaterialType.value → { phase_id → initial_strength ∈ (0, 1.0] }
#
# initial_strength < 1.0: Charakter-Schutz / physikalische Grenze
#   (z.B. 0.25 für phase_22_tape_saturation bei shellac → Röhrencharakter bewahren)
# initial_strength > 1.0: NICHT erlaubt — strength ∈ [0,1] in allen Phasen.
#   Stattdessen signalisiert ein Eintrag nahe 1.0 "maximale Stärke gewünscht".
# Kein Eintrag → 1.0 (volle Initialstärke, PMGG-gesteuerter Abbau bei Regression)

_MATERIAL_PHASE_FACTORS: dict[str, dict[str, float]] = {
    # ===================================================================
    # ANALOG VINTAGE MATERIALS  (high degradation, character preservation)
    # ===================================================================
    # WAX_CYLINDER — extreme noise, very limited BW (≤5 kHz), highest degradation
    "wax_cylinder": {
        "phase_03_denoise": 0.90,  # aggressive NR ok — extreme noise
        "phase_09_crackle_removal": 0.85,  # heavy crackle on wax
        "phase_22_tape_saturation": 0.20,  # protect tube/wax character
        "phase_20_reverb_reduction": 0.20,  # protect period-correct room
        "phase_49_advanced_dereverb": 0.20,  # vintage reverb is authentic
        "phase_06_frequency_restoration": 0.30,  # BW << 5kHz, don't over-extend
        "phase_07_harmonic_restoration": 0.35,  # careful harmonic reconstruction
        "phase_04_eq_correction": 0.30,  # extreme rolloff is era-authentic
        "phase_08_transient_preservation": 0.35,  # wax/needle transients fragile
        "phase_36_transient_shaper": 0.25,  # no transient shaping on wax
        "phase_35_multiband_compression": 0.30,  # preserve limited dynamics
        "phase_10_compression": 0.35,  # don't squash wax dynamics
        "phase_37_bass_enhancement": 0.25,  # BW too low for bass boost
        "phase_38_presence_boost": 0.30,  # don't artificially brighten
        "phase_39_air_band_enhancement": 0.20,  # no air band on wax
        "phase_16_final_eq": 0.35,  # gentle final EQ
        "phase_26_dynamic_range_expansion": 0.30,  # limited DR on wax
        "phase_13_stereo_enhancement": 0.20,  # mono source
        "phase_02_hum_removal": 0.50,  # wax has motor rumble + AC hum character
        "phase_18_noise_gate": 0.20,  # gating destroys continuous wax noise character
        "phase_24_dropout_repair": 0.45,  # careful with wax dropout fill
    },
    # SHELLAC — broad noise, BW ≤ 8 kHz, clicks/crackle primary
    "shellac": {
        "phase_03_denoise": 0.85,  # strong NR for shellac noise floor
        "phase_09_crackle_removal": 0.85,  # heavy crackle
        "phase_01_click_removal": 0.85,  # many deep clicks
        "phase_27_click_pop_removal": 0.80,
        "phase_22_tape_saturation": 0.25,  # protect tube character
        "phase_20_reverb_reduction": 0.20,  # vintage room — don't over-dereverberate
        "phase_49_advanced_dereverb": 0.20,
        "phase_06_frequency_restoration": 0.40,  # BW limited; careful extension
        "phase_07_harmonic_restoration": 0.45,
        "phase_04_eq_correction": 0.35,  # shellac rolloff is character
        "phase_08_transient_preservation": 0.40,  # needle transients fragile
        "phase_36_transient_shaper": 0.30,  # don't reshape shellac transients
        "phase_35_multiband_compression": 0.35,  # preserve vintage dynamics
        "phase_10_compression": 0.40,  # careful compression
        "phase_37_bass_enhancement": 0.30,  # limited bass on shellac
        "phase_38_presence_boost": 0.35,  # gentle brightness
        "phase_39_air_band_enhancement": 0.25,  # no air band on shellac
        "phase_16_final_eq": 0.40,
        "phase_13_stereo_enhancement": 0.20,  # mono era
        "phase_02_hum_removal": 0.45,  # shellac motor rumble is character
        "phase_18_noise_gate": 0.20,  # gating destroys groove-noise character
        "phase_24_dropout_repair": 0.50,  # scratch-based dropouts need care
        # v10.0.0: Shellac-spezifische Defekte vollständig abdecken
        "phase_05_rumble_filter": 0.80,  # Schellack-Dreher Subsonic-Rumble
        "phase_23_spectral_repair": 0.45,  # Spektrale Reparatur für IGD + Oberflächenlücken
        "phase_29_tape_hiss_reduction": 0.60,  # Shellac-Oberflächenrauschen ähnelt Tape-Hiss
        "phase_60_inner_groove_distortion_repair": 0.55,  # IGD tritt auch bei Schellack-Rillen auf
    },
    # LACQUER_DISC — similar to shellac, more substrate clicks
    "lacquer_disc": {
        "phase_03_denoise": 0.80,
        "phase_01_click_removal": 0.85,
        "phase_27_click_pop_removal": 0.80,
        "phase_22_tape_saturation": 0.25,
        "phase_20_reverb_reduction": 0.20,
        "phase_49_advanced_dereverb": 0.20,
        "phase_06_frequency_restoration": 0.50,
        "phase_04_eq_correction": 0.40,  # lacquer has some character rolloff
        "phase_08_transient_preservation": 0.45,
        "phase_36_transient_shaper": 0.35,
        "phase_35_multiband_compression": 0.40,
        "phase_37_bass_enhancement": 0.35,
        "phase_38_presence_boost": 0.40,
        "phase_39_air_band_enhancement": 0.30,
        "phase_02_hum_removal": 0.45,
        "phase_18_noise_gate": 0.20,
        "phase_24_dropout_repair": 0.50,
    },
    # WIRE_RECORDING — high noise, jitter, good dynamic range
    "wire_recording": {
        "phase_03_denoise": 0.85,
        "phase_12_wow_flutter_fix": 0.90,  # jitter is primary problem
        "phase_31_speed_pitch_correction": 0.85,
        "phase_22_tape_saturation": 0.35,
        "phase_10_compression": 0.40,
        "phase_04_eq_correction": 0.40,  # wire has characteristic response
        "phase_06_frequency_restoration": 0.35,  # limited BW on wire
        "phase_07_harmonic_restoration": 0.40,
        "phase_08_transient_preservation": 0.45,
        "phase_36_transient_shaper": 0.35,  # wire transients are metallic
        "phase_35_multiband_compression": 0.40,
        "phase_20_reverb_reduction": 0.25,  # vintage room
        "phase_49_advanced_dereverb": 0.25,
        "phase_37_bass_enhancement": 0.30,
        "phase_38_presence_boost": 0.35,
        "phase_39_air_band_enhancement": 0.25,
        "phase_13_stereo_enhancement": 0.20,  # mono source
        "phase_02_hum_removal": 0.50,  # wire motor hum is character
        "phase_18_noise_gate": 0.25,  # wire has continuous noise
        "phase_24_dropout_repair": 0.55,
    },
    # ===================================================================
    # GROOVE-BASED MATERIALS  (crackle/click priority, moderate NR)
    # ===================================================================
    # VINYL — crackle priority, moderate NR, warm character
    "vinyl": {
        "phase_09_crackle_removal": 0.85,
        "phase_01_click_removal": 0.80,
        "phase_27_click_pop_removal": 0.75,
        "phase_03_denoise": 0.70,  # vinyl NR gentler than shellac
        "phase_04_eq_correction": 0.55,  # RIAA curve is character
        "phase_06_frequency_restoration": 0.50,  # vinyl has natural rolloff
        "phase_07_harmonic_restoration": 0.55,
        "phase_08_transient_preservation": 0.55,  # vinyl transients have warmth
        "phase_22_tape_saturation": 0.30,  # protect analog warmth
        "phase_20_reverb_reduction": 0.25,  # vinyl room is character
        "phase_49_advanced_dereverb": 0.25,
        "phase_36_transient_shaper": 0.50,  # vinyl transients are character
        "phase_35_multiband_compression": 0.50,  # preserve vinyl dynamics
        "phase_10_compression": 0.50,
        "phase_37_bass_enhancement": 0.55,  # vinyl bass is warm
        "phase_38_presence_boost": 0.50,  # careful brightness
        "phase_39_air_band_enhancement": 0.40,  # vinyl air is limited
        "phase_16_final_eq": 0.55,
        "phase_26_dynamic_range_expansion": 0.50,
        "phase_02_hum_removal": 0.40,  # turntable motor rumble + AC hum
        "phase_18_noise_gate": 0.25,  # vinyl surface noise is continuous — gating = artifacts
        "phase_24_dropout_repair": 0.60,  # vinyl scratches need careful fill
        # v10.0.0: Vinyl-spezifische Defekte vollständig abdecken
        "phase_05_rumble_filter": 0.80,  # Plattenteller-Subsonic-Rumble (<25 Hz)
        "phase_12_wow_flutter_fix": 0.45,  # Vinyl-Warp → langsame Pitch-Schwankung
        "phase_23_spectral_repair": 0.60,  # IGD-Reste, Groove-Echo-Reparatur
        "phase_28_surface_noise_profiling": 0.80,  # Vinyl-Surface-Noise Profiler (primär)
        "phase_31_speed_pitch_correction": 0.30,  # Plattenspieler-Geschwindigkeitsfehler
        "phase_60_inner_groove_distortion_repair": 0.70,  # IGD: THD steigt mit kleinem Rillenradius
        "phase_61_groove_echo_cancellation": 0.65,  # Groove-Echo (~1.8 s @33⅓) beseitigen
    },
    # ===================================================================
    # TAPE-BASED MATERIALS  (hiss priority, tape character preservation)
    # ===================================================================
    # TAPE — hiss priority, preserve tape character (§2.22 Spec)
    "tape": {
        "phase_29_tape_hiss_reduction": 0.85,
        "phase_03_denoise": 0.75,
        "phase_22_tape_saturation": 0.35,  # tape imprint must be preserved
        "phase_04_eq_correction": 0.45,  # tape EQ rolloff is character
        "phase_06_frequency_restoration": 0.40,  # don't over-extend tape BW
        "phase_07_harmonic_restoration": 0.50,  # tape harmonics are warm
        "phase_08_transient_preservation": 0.50,  # tape transients are soft
        "phase_10_compression": 0.45,  # preserve tape dynamics
        "phase_20_reverb_reduction": 0.30,  # tape room is authentic
        "phase_49_advanced_dereverb": 0.30,
        "phase_36_transient_shaper": 0.40,  # fragile tape transients
        "phase_35_multiband_compression": 0.45,  # gentle multiband
        "phase_37_bass_enhancement": 0.50,  # tape bass is warm
        "phase_38_presence_boost": 0.45,  # don't over-brighten tape
        "phase_39_air_band_enhancement": 0.35,  # tape HF is limited
        "phase_16_final_eq": 0.50,
        "phase_26_dynamic_range_expansion": 0.50,
        "phase_02_hum_removal": 0.30,  # tape motor hum is character; 49% regression at 0.25 → very gentle
        "phase_18_noise_gate": 0.20,  # tape hiss is continuous — gating = pumping artifacts
        "phase_24_dropout_repair": 0.55,  # oxide flaking dropouts need careful fill
        "phase_12_wow_flutter_fix": 0.75,  # capstan flutter is real but ML phase has no retry
        # v10.0.0: Tape-spezifische Defekte vollständig abdecken
        "phase_01_click_removal": 0.45,  # Tape-Dropouts erzeugen click-artige Impulse
        "phase_23_spectral_repair": 0.55,  # Print-Through-Reste, Generation-Loss-Reparatur
        "phase_31_speed_pitch_correction": 0.40,  # Reel-Tape-Motorgeschwindigkeitsfehler
        "phase_40_loudness_normalization": 0.50,  # Pegelausgleich nach Hiss-Reduktion
        "phase_55_diffusion_inpainting": 0.55,  # Tape-Dropout-Diffusions-Inpainting
        "phase_57_print_through_reduction": 0.70,  # Print-Through primäres Reel-Tape-Problem
        "phase_64_tape_splice_repair": 0.60,  # Splice-Stellen: Klick + Pegelsprung + Phase
    },
    # CASSETTE — compact cassette defects must be corrected, not treated as generic tape.
    "cassette": {
        "phase_29_tape_hiss_reduction": 0.90,  # cassette hiss/Dolby residue is prominent
        "phase_59_modulation_noise_reduction": 0.80,  # Dolby/dbx breathing and modulation noise
        "phase_12_wow_flutter_fix": 0.85,  # capstan instability is a primary cassette defect
        "phase_24_dropout_repair": 0.70,  # oxide/head-contact dropouts are audible defects
        "phase_25_azimuth_correction": 0.80,  # cassette head azimuth drift is common
        "phase_04_eq_correction": 0.60,  # Dolby/EQ mismatch correction
        "phase_03_denoise": 0.70,
        "phase_06_frequency_restoration": 0.35,  # do not invent air above cassette ceiling
        "phase_07_harmonic_restoration": 0.35,
        "phase_20_reverb_reduction": 0.25,  # do not chase authentic room tone
        "phase_49_advanced_dereverb": 0.25,
        "phase_18_noise_gate": 0.15,  # gating cassette hiss creates pumping/echo illusion
        "phase_39_air_band_enhancement": 0.20,
        # v10.0.0: Kassetten-spezifische Defekte vollständig abdecken
        "phase_01_click_removal": 0.50,  # Oxide-Fehler + Head-Clog → click-artige Impulse
        "phase_02_hum_removal": 0.45,  # Kassettenrekorder-Gleichstrommotor-Brummen
        "phase_08_transient_preservation": 0.65,  # Transienten nach NR schützen (Dolby-Atmung)
        "phase_14_phase_correction": 0.75,  # Azimuth-Fehler → Phasendrehung zwischen Kanälen
        "phase_23_spectral_repair": 0.45,  # Print-Through-Reste, Generation-Loss
        "phase_26_dynamic_range_expansion": 0.30,  # Vorsichtig — Dolby-Kompression bewahren
        "phase_31_speed_pitch_correction": 0.45,  # Kassettenrekorder-Motorgeschwindigkeitsfehler
        "phase_36_transient_shaper": 0.35,  # Moderate Transientenwiederherstellung
        "phase_40_loudness_normalization": 0.55,  # Pegelausgleich nach Hiss-Reduktion
        "phase_54_transparent_dynamics": 0.60,  # Dolby-NR-Pumpen beseitigen (nr_breathing)
        "phase_55_diffusion_inpainting": 0.55,  # Oxide-/Head-Contact-Dropout-Inpainting
        "phase_56_spectral_band_gap_repair": 0.60,  # Head-Wear → frequenzspezifische Band-Lücken
        "phase_57_print_through_reduction": 0.45,  # Vorecho (weniger ausgeprägt als Reel-Tape)
        "phase_64_tape_splice_repair": 0.45,  # Klick + Pegelsprung + Phasendiskontinuität
    },
    # REEL_TAPE — higher quality, print-through focus
    "reel_tape": {
        "phase_29_tape_hiss_reduction": 0.80,
        "phase_03_denoise": 0.70,
        "phase_22_tape_saturation": 0.35,  # tape character preserved
        "phase_04_eq_correction": 0.50,  # reel has better EQ response
        "phase_06_frequency_restoration": 0.45,  # better BW than cassette
        "phase_07_harmonic_restoration": 0.55,
        "phase_08_transient_preservation": 0.55,
        "phase_10_compression": 0.50,
        "phase_20_reverb_reduction": 0.35,
        "phase_49_advanced_dereverb": 0.35,
        "phase_36_transient_shaper": 0.45,  # reel transients more robust
        "phase_35_multiband_compression": 0.50,
        "phase_37_bass_enhancement": 0.55,
        "phase_38_presence_boost": 0.50,
        "phase_39_air_band_enhancement": 0.40,
        "phase_16_final_eq": 0.55,
        "phase_26_dynamic_range_expansion": 0.55,
        "phase_02_hum_removal": 0.35,  # reel motor hum + print-through character
        "phase_18_noise_gate": 0.25,  # continuous hiss → gating artifacts
        "phase_24_dropout_repair": 0.60,  # reel dropouts less frequent
        "phase_12_wow_flutter_fix": 0.70,  # reel has more stable transport
    },
    # ===================================================================
    # DIGITAL CODEC MATERIALS  (codec artifacts, careful spectral repair)
    # ===================================================================
    # MP3_LOW — heavy codec artifacts; careful HF extension
    "mp3_low": {
        "phase_06_frequency_restoration": 0.75,  # codec cuts HF; restore carefully
        "phase_07_harmonic_restoration": 0.75,
        "phase_23_spectral_repair": 0.80,
        "phase_50_spectral_repair": 0.80,
        "phase_03_denoise": 0.40,  # codec noise ≠ analog noise
        "phase_04_eq_correction": 0.65,  # codec EQ is damage not character
        "phase_09_crackle_removal": 0.25,  # no crackle in MP3
        "phase_01_click_removal": 0.25,  # no clicks in MP3
        "phase_22_tape_saturation": 0.15,  # no tape character to protect
        "phase_35_multiband_compression": 0.60,  # gentle on compressed audio
        "phase_36_transient_shaper": 0.55,  # codec smears transients
        "phase_39_air_band_enhancement": 0.55,  # some air restoration ok
        "phase_02_hum_removal": 0.55,  # digital hum is fixable
        "phase_18_noise_gate": 0.40,  # codec noise floor is low enough for gating
        "phase_24_dropout_repair": 0.65,  # codec dropouts are real defects
    },
    # MP3_HIGH — moderate codec artifacts; mostly transparent
    "mp3_high": {
        "phase_06_frequency_restoration": 0.85,  # mild HF extension
        "phase_07_harmonic_restoration": 0.85,
        "phase_23_spectral_repair": 0.70,  # minor spectral holes
        "phase_50_spectral_repair": 0.70,
        "phase_03_denoise": 0.30,  # minimal NR needed
        "phase_09_crackle_removal": 0.20,  # no crackle
        "phase_01_click_removal": 0.20,  # no clicks
        "phase_22_tape_saturation": 0.15,  # no tape character
        "phase_04_eq_correction": 0.70,  # gentle EQ
        "phase_35_multiband_compression": 0.65,
        "phase_36_transient_shaper": 0.60,
        "phase_02_hum_removal": 0.60,
        "phase_18_noise_gate": 0.45,
        "phase_24_dropout_repair": 0.65,
    },
    # AAC — similar to MP3_HIGH; slightly better HF
    "aac": {
        "phase_06_frequency_restoration": 0.80,
        "phase_07_harmonic_restoration": 0.80,
        "phase_23_spectral_repair": 0.70,
        "phase_50_spectral_repair": 0.70,
        "phase_03_denoise": 0.30,
        "phase_09_crackle_removal": 0.20,
        "phase_01_click_removal": 0.20,
        "phase_22_tape_saturation": 0.15,
        "phase_04_eq_correction": 0.70,
        "phase_35_multiband_compression": 0.65,
        "phase_36_transient_shaper": 0.60,
        "phase_02_hum_removal": 0.60,
        "phase_18_noise_gate": 0.45,
        "phase_24_dropout_repair": 0.65,
    },
    # MINIDISC — ATRAC codec; similar to MP3_HIGH but different artifacts
    "minidisc": {
        "phase_06_frequency_restoration": 0.80,  # ATRAC cuts some HF
        "phase_07_harmonic_restoration": 0.80,
        "phase_23_spectral_repair": 0.75,  # ATRAC spectral holes
        "phase_50_spectral_repair": 0.75,
        "phase_03_denoise": 0.30,
        "phase_09_crackle_removal": 0.20,
        "phase_01_click_removal": 0.20,
        "phase_22_tape_saturation": 0.15,
        "phase_04_eq_correction": 0.65,
        "phase_35_multiband_compression": 0.60,
        "phase_36_transient_shaper": 0.55,
        "phase_02_hum_removal": 0.55,
        "phase_18_noise_gate": 0.40,
        "phase_24_dropout_repair": 0.60,
    },
    # STREAMING — variable quality; usually well-encoded
    "streaming": {
        "phase_03_denoise": 0.30,  # minor background noise possible
        "phase_06_frequency_restoration": 0.80,
        "phase_07_harmonic_restoration": 0.80,
        "phase_23_spectral_repair": 0.65,
        "phase_50_spectral_repair": 0.65,
        "phase_09_crackle_removal": 0.20,
        "phase_01_click_removal": 0.20,
        "phase_22_tape_saturation": 0.15,
        "phase_04_eq_correction": 0.70,
        "phase_35_multiband_compression": 0.65,
        "phase_36_transient_shaper": 0.65,
        "phase_02_hum_removal": 0.60,
        "phase_18_noise_gate": 0.45,
        "phase_24_dropout_repair": 0.65,
    },
    # ===================================================================
    # HIGH-QUALITY DIGITAL MATERIALS  (minimal processing needed)
    # ===================================================================
    # CD_DIGITAL — high quality input; minimal aggressive processing
    "cd_digital": {
        "phase_03_denoise": 0.25,  # minimal NR for clean digital
        "phase_09_crackle_removal": 0.20,
        "phase_01_click_removal": 0.30,
        "phase_22_tape_saturation": 0.15,  # no analog character
        "phase_04_eq_correction": 0.75,  # mild EQ ok on CD
        "phase_06_frequency_restoration": 0.85,  # CD has full BW to 22kHz
        "phase_20_reverb_reduction": 0.70,  # digital reverb ok to reduce
        "phase_49_advanced_dereverb": 0.70,
        "phase_35_multiband_compression": 0.70,
        "phase_36_transient_shaper": 0.70,
        "phase_02_hum_removal": 0.60,  # digital hum is real problem, ok to fix
        "phase_18_noise_gate": 0.50,  # digital noise floor very low — gate ok
        "phase_24_dropout_repair": 0.70,  # digital dropouts are true defects
    },
    # DAT — near-lossless digital; near-zero NR
    "dat": {
        "phase_03_denoise": 0.20,
        "phase_09_crackle_removal": 0.15,
        "phase_01_click_removal": 0.20,
        "phase_22_tape_saturation": 0.15,
        "phase_04_eq_correction": 0.75,
        "phase_06_frequency_restoration": 0.85,
        "phase_20_reverb_reduction": 0.70,
        "phase_49_advanced_dereverb": 0.70,
        "phase_35_multiband_compression": 0.70,
        "phase_36_transient_shaper": 0.70,
        "phase_02_hum_removal": 0.60,
        "phase_18_noise_gate": 0.50,
        "phase_24_dropout_repair": 0.65,
    },
}


_validate_phase_map_completeness()


def get_material_initial_strength(material: str, phase_id: str) -> float:
    """Gibt the material-adaptive initial strength for a given phase zurück.

    Used by PMGG to set the correct starting strength instead of always 1.0.
    Returns 1.0 (default / no override) when no entry is found.

    Args:
        material:  MaterialType.value string (e.g. 'shellac', 'vinyl')
        phase_id:  Full phase_id string (e.g. 'phase_03_denoise')

    Returns:
        initial_strength ∈ (0, 1.0]; default = 1.0
    """
    _mk = material.value if isinstance(material, MaterialType) else material  # §v10.113
    factors = _MATERIAL_PHASE_FACTORS.get(_mk, {})
    return float(factors.get(phase_id, 1.0))


# ---------------------------------------------------------------------------
# Haupt-Klasse: DefectPhaseMapper
# ---------------------------------------------------------------------------


class DefectPhaseMapper:
    """
    Ordnet DetectedDefects den richtigen Phase-IDs zu und
    konfiguriert ProcessingConfig präzise für jeden Defekt.

    Verwendung in AutonomousRestorationEngine._build_specialist_variant().
    """

    def get_assignment(self, defect_type: DefectType) -> PhaseAssignment | None:
        """Gibt PhaseAssignment für defect_type zurück, oder None wenn unbekannt."""
        return _PHASE_MAP.get(defect_type)

    def get_primary_phases(self, defect_type: DefectType, mode: str = "restoration") -> list[str]:
        """Gibt Primary-Phase-IDs für defect_type zurück.

        Args:
            defect_type: Der zu behandelnde Defekttyp.
            mode: Verarbeitungsmodus — "restoration" filtert §0a-verbotene Phasen
                  (phase_21_exciter, phase_35_multiband_compression, phase_42_vocal_enhancement).
        """
        a = _PHASE_MAP.get(defect_type)
        if a is None:
            return []
        phases = a.primary_phases
        if mode == "restoration":
            phases = [p for p in phases if p not in _RESTORATION_FORBIDDEN_PHASES]
        return phases

    def get_all_phases(self, defect_type: DefectType, mode: str = "restoration") -> list[str]:
        """Gibt Primary + Secondary Phase-IDs zurück (geordnet nach Priorität).

        Args:
            defect_type: Der zu behandelnde Defekttyp.
            mode: Verarbeitungsmodus — "restoration" filtert §0a-verbotene Phasen.
        """
        a = _PHASE_MAP.get(defect_type)
        if a is None:
            return []
        phases = a.primary_phases + a.secondary_phases
        if mode == "restoration":
            phases = [p for p in phases if p not in _RESTORATION_FORBIDDEN_PHASES]
        return phases

    def build_specialist_config(
        self,
        base_config: object,
        defect_type: DefectType,
        severity: float,
        is_restoration_mode: bool = True,
        material: str | None = None,
    ) -> tuple[object, str]:
        """
        Konfiguriert base_config für den angegebenen Defekt.

        Args:
            base_config:          ProcessingConfig (wird in-place geändert)
            defect_type:          Primärer Defekttyp
            severity:             Defektstärke [0–1]
            is_restoration_mode:  True = RESTORATION (sanfterer mode_factor)
            material:             MaterialType.value-String für material-adaptive Skalierung
                                  (z.B. 'shellac', 'vinyl'). None = kein Material-Override.

        Returns:
            (konfiguriertes_config, variant_name_string)
        """
        config = copy.deepcopy(base_config)
        assignment = _PHASE_MAP.get(defect_type)

        if assignment is None:
            logger.debug("Kein Mapping für %s — Basis-Config unverändert.", defect_type)
            return config, f"specialist_{defect_type.value}"

        # RESTORATION-Modus: etwas sanfter (0.8×)
        mode_factor = 0.80 if is_restoration_mode else 1.0

        # Material-Faktor: Mittelwert der primary Phase-Faktoren als config-Skalierung
        mat_factor = 1.0
        if material is not None and assignment.primary_phases:
            phase_factors = [get_material_initial_strength(material, pid) for pid in assignment.primary_phases]
            mat_factor = sum(phase_factors) / len(phase_factors)

        assignment.apply_to_config(config, severity=severity, mode_factor=mode_factor, material_factor=mat_factor)

        # Sicherheitscheck: denoise_strength nie > 0.9 (Authentizität)
        if hasattr(config, "denoise_strength") and is_restoration_mode:
            config.denoise_strength = min(config.denoise_strength, 0.90)  # type: ignore[attr-defined]

        variant_name = f"specialist_{defect_type.value.replace('_', '')}"
        logger.info(
            "Specialist-Config für %s (severity=%.2f, Betriebsart_factor=%.1f, mat_factor=%.2f): phases=%s",
            defect_type.value,
            severity,
            mode_factor,
            mat_factor,
            assignment.primary_phases[:2],
        )
        return config, variant_name

    def phases_for_defect_profile(
        self,
        defects: list,
        max_phases: int = 10,
        mode: str = "restoration",
        material: str | None = None,
        phase_coalitions: dict[str, tuple[str, ...]] | None = None,
    ) -> list[str]:
        """
        Gibt eine de-duplizierte, priorisierte Phase-Liste für mehrere Defekte zurück.

        Args:
            defects:    Liste von DefectScore-Objekten (mit .defect_type, .severity)
            max_phases: Maximale Anzahl zurückgegebener Phasen
            mode:       Verarbeitungsmodus — "restoration" filtert §0a-verbotene Phasen
                        (phase_21_exciter, phase_35_multiband_compression, phase_42_vocal_enhancement).
            material:   Optionaler MaterialType.value-String. Für digitale Codec-Materialien
                        wird phase_09_crackle_removal ausgeblendet (V29, §4.11).
            phase_coalitions:
                        Optionaler Koalitions-Mapping-Override (coalition_name → tuple[phase_id]).
                        Falls None, werden die internen DPM-Koalitionen verwendet.

        Returns:
            Geordnete Phase-ID-Liste (primary first, dann secondary, dann de-dup)
        """

        def _sanitize_01(value: object | None, default: float) -> float:
            try:
                if value is None:
                    return float(default)
                v = float(value)  # type: ignore[arg-type]
                if v != v:  # NaN
                    return float(default)
                return max(0.0, min(1.0, v))
            except Exception as e:
                logger.warning("defect_Verarbeitungsschritt_mapper.py::_sanitize_01 Ersatzpfad: %s", e)
                return float(default)

        seen: dict[str, float] = {}  # phase_id → max_severity
        _confidences: list[float] = []
        for defect in defects:
            severity = _sanitize_01(getattr(defect, "severity", 0.5), 0.5)
            confidence = _sanitize_01(getattr(defect, "confidence", None), 0.65)
            _confidences.append(confidence)
            # Unsicherheitsbewusste Priorisierung: bei niedriger Defekt-Confidence
            # werden Zusatzphasen zurückhaltender priorisiert (No-Harm).
            confidence_gain = 0.55 + 0.45 * confidence
            primary_weight = 1.00 if confidence >= 0.80 else 0.95
            secondary_weight = 0.60
            if confidence < 0.35:
                secondary_weight = 0.35 if mode == "restoration" else 0.45
            elif confidence < 0.60:
                secondary_weight = 0.50
            if confidence < 0.20 and mode == "restoration":
                secondary_weight = 0.00

            dt = getattr(defect, "defect_type", None)
            if dt is None:
                continue
            assignment = _PHASE_MAP.get(dt)
            if assignment is None:
                continue
            for phase_id in assignment.primary_phases:
                seen[phase_id] = max(seen.get(phase_id, 0.0), severity * confidence_gain * primary_weight)
            for phase_id in assignment.secondary_phases:
                seen[phase_id] = max(seen.get(phase_id, 0.0), severity * confidence_gain * secondary_weight)

        # §0a: Verbotene Phasen im Restoration-Modus herausfiltern
        if mode == "restoration":
            seen = {p: s for p, s in seen.items() if p not in _RESTORATION_FORBIDDEN_PHASES}

        # V29/§4.11: Kratzentferner ist physikalisch falsch für digitale Codec-Materialien
        # (Codec-Artefakte sind keine Vinyl/Shellac-Kratzer — KEIN phase_09 für MP3/AAC/CD)
        if material and material in _DIGITAL_CODEC_MATERIALS:
            seen.pop("phase_09_crackle_removal", None)

        # §2.67 Koalitions-Priorisierung: Wenn mindestens zwei Mitglieder einer
        # Koalition aktiv sind, werden die Scores innerhalb der Koalition enger
        # zusammengezogen. Dadurch laufen zusammengehörige Reparaturschritte
        # konsistenter als Gruppe statt durch globale Defekt-Scores getrennt.
        _coalitions = phase_coalitions if isinstance(phase_coalitions, dict) else _DEFECT_MAPPER_PHASE_COALITIONS
        for members in _coalitions.values():
            present = [pid for pid in members if pid in seen]
            if len(present) < 2:
                continue
            dominant = max(float(seen[pid]) for pid in present)
            coalition_floor = max(0.0, dominant * 0.92)
            for pid in present:
                seen[pid] = max(float(seen[pid]), coalition_floor)

        # Sortieren: primäre (severity×1.0) zuerst, sekundäre danach
        seen = {phase_id: score for phase_id, score in seen.items() if float(score) > 0.0}
        sorted_phases = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)
        _effective_max_phases = int(max_phases)
        if _confidences:
            _avg_conf = sum(_confidences) / len(_confidences)
            if _avg_conf < 0.35:
                _effective_max_phases = max(3, int(max_phases * 0.5))
            elif _avg_conf < 0.50:
                _effective_max_phases = max(4, int(max_phases * 0.7))

        return [phase_id for phase_id, _ in sorted_phases[:_effective_max_phases]]

    def describe(self, defect_type: DefectType) -> str:
        """Menschenlesbare Beschreibung der Strategie für einen Defekttyp."""
        a = _PHASE_MAP.get(defect_type)
        if a is None:
            return f"Kein Mapping für {defect_type.value} bekannt."
        return (
            f"[{defect_type.value}] {a.description}\n"
            f"  Primary : {', '.join(a.primary_phases)}\n"
            f"  Secondary: {', '.join(a.secondary_phases)}"
        )


# ---------------------------------------------------------------------------
# Reverse Phase Map — phase_id → [DefectType] (§MusikalischeHarmonisierung)
# ---------------------------------------------------------------------------
_REVERSE_PHASE_MAP: dict[str, list[DefectType]] | None = None
_REVERSE_LOCK = threading.Lock()


def _build_reverse_phase_map() -> dict[str, list[DefectType]]:
    """Erstellt reverse mapping: phase_id → [DefectType] from _PHASE_MAP.

    Only primary_phases are mapped (these are the phases designed to FIX
    the defect). Secondary phases are support roles and should not be
    severity-scaled.
    """
    reverse: dict[str, list[DefectType]] = {}
    _excluded_primary_phases = {
        # Loudness-Normalisierung ist kein defektgebundener Repair-Step für
        # die severity-proportionale Wet/Dry-Skalierung.
        "phase_40_loudness_normalization",
        # §0a/§Test: Enhancement-Phasen gehören NICHT in die Reverse-Map —
        # sie sind klangverbessernd, nicht defektgebunden (keine Severity-Skalierung).
        "phase_13_stereo_enhancement",
        "phase_21_exciter",
        "phase_17_mastering_polish",
    }
    for defect_type, assignment in _PHASE_MAP.items():
        for phase_id in assignment.primary_phases:
            if phase_id in _excluded_primary_phases and defect_type is not DefectType.AMPLITUDE_DRIFT:
                continue
            reverse.setdefault(phase_id, []).append(defect_type)
    return reverse


def get_reverse_phase_map() -> dict[str, list[DefectType]]:
    """Thread-safe accessor for the reverse phase map (cached).

    Returns:
        Dict mapping phase_id → list of DefectTypes this phase primarily targets.
        Enhancement phases (not in _PHASE_MAP as primary) are absent.
    """
    global _REVERSE_PHASE_MAP  # pylint: disable=global-statement
    if _REVERSE_PHASE_MAP is None:
        with _REVERSE_LOCK:
            if _REVERSE_PHASE_MAP is None:
                _REVERSE_PHASE_MAP = _build_reverse_phase_map()
    return _REVERSE_PHASE_MAP


def get_phase_defect_severity(
    phase_id: str,
    defect_scores: dict,
    defect_severity_map: dict[str, float] | None = None,
) -> float:
    """Gibt a severity factor ∈ [_MIN_SEVERITY_FLOOR, 1.0] for a given phase zurück.

    For defect-repair phases (present in reverse map): the factor scales
    proportionally to the maximum measured severity among all DefectTypes
    this phase primarily targets. This ensures:
      - Phases process proportionally to actual defect intensity
      - Low severity → gentle processing (psychoacoustic preservation)
      - High severity → full processing

    For enhancement phases (NOT in reverse map): returns 1.0 (no modulation).

    §v10.21: Wenn defect_severity_map (merged, saliency-weighted, resolved-defects-
    capped) übergeben wird, wird DIESE für die Severity verwendet — nicht die
    rohen defect_scores. Dies stellt sicher, dass bereits behobene Defekte
    (via resolved_defects-Accumulator) den Wet/Dry-Faktor korrekt reduzieren.

    Args:
        phase_id:            Full phase ID (e.g. 'phase_03_denoise')
        defect_scores:       dict[DefectType, DefectScore] from DefectScanner
        defect_severity_map: §v10.21: merged severity map (str → float),
                             bereits saliency-gewichtet und resolved-defects-gecapped

    Returns:
        Severity factor ∈ [0.15, 1.0].
        0.15 = minimal (defect barely present, gentle processing).
        1.0  = full severity or enhancement phase (no reduction).
    """
    rmap = get_reverse_phase_map()
    target_defects = rmap.get(phase_id)
    if not target_defects:
        return 1.0  # Enhancement phase — no modulation

    max_severity = 0.0
    found_any = False
    # §v10.21: Prefer merged defect_severity_map (saliency-weighted + resolved-defects-capped)
    # over raw defect_scores. This ensures the wet/dry factor reflects the CURRENT
    # defect state, not the pre-analysis snapshot.
    _use_merged = defect_severity_map is not None and len(defect_severity_map) > 0
    for dt in target_defects:
        if _use_merged:
            # Try string key first (defect_severity_map uses string keys like "CLICKS")
            sev = defect_severity_map.get(dt.value if hasattr(dt, "value") else str(dt))  # type: ignore[union-attr]
            if sev is not None:
                found_any = True
                max_severity = max(max_severity, float(sev))
                continue
        # Fallback to raw defect_scores
        score = defect_scores.get(dt)
        if score is not None:
            found_any = True
            sev = getattr(score, "severity", 0.0)
            max_severity = max(max_severity, float(sev))

    if not found_any:
        return 1.0  # None of the targeted DefectTypes were scanned — don't penalize

    # Floor at 0.15: even low-severity defects need minimum processing.
    # The phase was selected for a reason — never fully bypass.
    return max(0.15, min(1.0, max_severity))


def get_phase_locality_factor(
    phase_id: str,
    defect_scores: dict,  # pylint: disable=unused-argument
    defect_location_coverage_map: dict[str, float] | None,
) -> float:
    """Gibt locality factor ∈ [0.35, 1.0] for a given phase zurück.

    Locality reduces global Wet/Dry intensity for sparse, event-like defects,
    preserving timbre outside affected regions.

    - Enhancement phases (not in reverse map) always return 1.0.
    - Defect phases without location data default to 1.0.
    - For event-like defects, per-defect curves are used:
        factor = floor + (1 - floor) * coverage**gamma

    where coverage is temporal defect coverage in [0, 1].
    """
    if not defect_location_coverage_map:
        return 1.0

    rmap = get_reverse_phase_map()
    target_defects = rmap.get(phase_id)
    if not target_defects:
        return 1.0

    # Per-defect locality curve parameters (floor, gamma).
    # Lower floor + higher gamma => stronger damping for sparse events.
    _LOCALITY_CURVES: dict[DefectType, tuple[float, float]] = {
        DefectType.CLICKS: (0.30, 0.75),
        DefectType.CRACKLE: (0.35, 0.85),
        DefectType.DROPOUTS: (0.45, 1.10),
        DefectType.CLIPPING: (0.35, 0.90),
        DefectType.SIBILANCE: (0.30, 0.80),
        DefectType.PRE_ECHO: (0.35, 0.90),
        DefectType.TRANSIENT_SMEARING: (0.40, 1.00),
        DefectType.TRANSPORT_BUMP: (0.40, 1.05),
        DefectType.PRINT_THROUGH: (0.45, 1.00),
    }

    best_factor = 1.0
    found_event_target = False

    for dt in target_defects:
        curve = _LOCALITY_CURVES.get(dt)
        if curve is None:
            continue
        found_event_target = True
        key = dt.value
        coverage = float(defect_location_coverage_map.get(key, 1.0))
        coverage = max(0.0, min(1.0, coverage))
        floor, gamma = curve
        fac = float(floor + (1.0 - floor) * (coverage**gamma))
        best_factor = min(best_factor, fac)

    if not found_event_target:
        return 1.0

    # Safety clamp.
    return max(0.35, min(1.0, best_factor))
