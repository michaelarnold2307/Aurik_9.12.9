"""
backend/core/intentional_artifact_classifier.py
Aurik 10.0.0 — Spec §6.5 (v10.0.0): Authentischer Klangcharakter vs. Defekt — Taxonomy.

Klassifiziert Signal-Merkmale als PRESERVE (authentischer Epochen-Charakter) oder
REPAIR (echter Defekt). Wird VOR DefectScanner-Auswahl in CAUSE_TO_PHASES ausgeführt,
damit PRESERVE-Merkmale nicht versehentlich volle Phase-Strength auslösen.

§6.5b Implementierungs-Invarianten:
- PRESERVE-Klassifikation schlägt DefectScanner-Aktivierung.
- VERBOTEN: PRESERVE-Klasse durch Phase bearbeiten, die Strength > 0.10 hat.
- Ausnahme: artifact_freedom < 0.90 durch PRESERVE-Merkmal → Strength bis 0.20.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# §6.5a Kanonische Charakter-Bewahrungsliste pro Material (normative Quelle)
# ---------------------------------------------------------------------------

#: Klassifikation: "PRESERVE" | "REPAIR" | "AMBIGUOUS"
_Classification = Literal["PRESERVE", "REPAIR", "AMBIGUOUS"]

AUTHENTIC_CHARACTER: dict[str, dict[str, _Classification]] = {
    "shellac": {
        "surface_noise_texture": "PRESERVE",  # 78-rpm Hintergrundrauschen; NR nur bis -45 dBFS
        "h2_h4_harmonic_saturation": "PRESERVE",  # Röhren-/Kristallmikrofon-Sättigung; WärmeMetric
        "bandwidth_ceiling_8khz": "PRESERVE",  # physikalische Grenze — kein BW-Extension >8 kHz
        "mono_center_image": "PRESERVE",  # alle Shellac-Quellen sind Mono
        "soft_transients": "PRESERVE",  # AGC-bedingte weiche Transienten (kein Transient-Shaper!)
    },
    "vinyl": {
        "groove_distortion_low": "PRESERVE",  # leichte Rillenverzerrung < 1 % THD ist authentisch
        "interlabel_noise_texture": "PRESERVE",  # Zwischenspur-Rauschen zwischen Grooves
        "riaa_warmth_curve": "PRESERVE",  # RIAA-Entzerrung erzeugt charakteristischen Bassanstieg
        "inner_groove_compression": "PRESERVE",  # Innenspur-Kompression ist physikalisch bedingt
        "low_level_wow_sub1hz": "PRESERVE",  # Plattenteller-Gleichlaufschwankung < 1 Hz ist Charakter
    },
    "tape": {
        "tape_saturation_knee": "PRESERVE",  # charakteristisches Kompressionsknie bei Übersteuerung
        "high_frequency_rolloff": "PRESERVE",  # materialbedingt: Typ-I-Kassette rolliert ab ~12 kHz
        "bias_noise_texture": "PRESERVE",  # Bias-Rauschen (HF > 18 kHz ist kein Defekt)
        "dolby_breathing_slight": "PRESERVE",  # leichtes Dolby-Atmen < -3 dB ist Epochencharakter
        "tape_compression_even_harmonics": "PRESERVE",  # Bandsättigung fügt H2 hinzu: Wärme-Merkmal
    },
    "reel_tape": {
        "studio_ambience_bleed": "PRESERVE",  # Raumrauschboden des Aufnahmeraums ist Teil der Aufnahme
        "tape_hiss_floor_texture": "PRESERVE",  # NR nur bis zum reel_tape-Boden (-60 dBFS)
        "print_through_ghost": "REPAIR",  # Print-Through (Vor-/Nachhall Spur auf Spur) = echter Defekt
        "tape_head_clog": "REPAIR",  # Kopfverstopfung = echter Defekt
    },
    "wax_cylinder": {
        "trichter_bandlimit_3khz": "PRESERVE",  # akustische Aufnahme: BW ≤ 3 kHz ist Realität
        "surface_crackle_fine": "PRESERVE",  # feines Oberflächengeräusch ist Teil der Epoche
        "mechanical_resonance": "PRESERVE",  # Trichtereigenresonanz ≈ 300–600 Hz ist Charakteristik
        "coarse_crackle_clicks": "REPAIR",  # grobe Klicker > 10 ms sind Defekte
    },
    "cd_digital": {
        "dithering_noise_floor": "PRESERVE",  # 16-bit-Dithering-Rauschen ist Teil des Formats
        "pre_emphasis_curve": "PRESERVE",  # wenn Pre-Emphasis aktiv war: originalgetreu
        "linear_phase_character": "PRESERVE",  # CD hat nah-perfekte Linearphase — kein Phasenediting
        "quantization_artifacts_mild": "PRESERVE",  # leichte Quantisierungsartefakte < -90 dBFS: normal
    },
    "mp3_low": {
        "pre_echo_character_mild": "PRESERVE",  # leichtes Pre-Echo bei stabiler Musik ist Format-Charakter
        "psychoacoustic_residue": "PRESERVE",  # wahrnehmungspsychologisches Kodierresiduum: Epoche
        "severe_pre_echo": "REPAIR",  # starkes Pre-Echo > 40 ms ist echter Defekt (phase_50)
        "metallic_ringing": "REPAIR",  # metallisches Klingen ist echter Defekt
    },
    "mp3_high": {
        # §v10.95: mp3 320kbps — nah-transparent, nur minimale Artefakte
        "psychoacoustic_coding_mild": "PRESERVE",  # MP3 320kbps ist nah-transparent
        "pre_echo_minor": "PRESERVE",  # minimales Pre-Echo bei hoher Bitrate: Format-Charakter
        "severe_pre_echo": "REPAIR",  # starkes Pre-Echo: Defekt
        "metallic_ringing": "REPAIR",  # metallisches Klingen: Defekt
    },
    "lacquer_disc": {
        "substrate_texture": "PRESERVE",  # Acetat-Substrat-Textur ist Materialcharakter
        "light_surface_clicks": "PRESERVE",  # leichte Oberflächenklicker ≤ 3 ms: Epoche
        "deep_crack_clicks": "REPAIR",  # tiefe Rissklicker > 5 ms: echter Defekt
    },
    # §v10.92: Bisher fehlende Material-Einträge — jedes Material braucht
    # authentische Charakter-Profile, sonst return 1.0 (keine Preservation).
    "cassette": {
        "tape_hiss_floor": "PRESERVE",  # Kassetten-Hiss ≤ -45 dBFS ist Format-Charakter
        "hf_rolloff_12khz": "PRESERVE",  # Typ-I-Band rolliert ab ~12 kHz physikalisch
        "dolby_breathing": "PRESERVE",  # leichtes Dolby-Atmen ≤ -3 dB ist Epochenmerkmal
        "wow_flutter_mild": "PRESERVE",  # Gleichlaufschwankung < 0.2 % ist Capstan-Physik
        "azimuth_misalign_minor": "PRESERVE",  # leichte Azimuth-Fehler ≤ 5° sind normal
        "severe_wow": "REPAIR",  # starke Gleichlaufschwankung > 0.5 % ist Defekt
        "deep_azimuth_error": "REPAIR",  # Azimuth-Fehler > 10° zerstört Stereobild
    },
    "kassette": {
        # Alias für cassette (deutsche Schreibweise) — identische Physik
        "tape_hiss_floor": "PRESERVE",
        "hf_rolloff_12khz": "PRESERVE",
        "dolby_breathing": "PRESERVE",
        "wow_flutter_mild": "PRESERVE",
        "azimuth_misalign_minor": "PRESERVE",
        "severe_wow": "REPAIR",
        "deep_azimuth_error": "REPAIR",
    },
    "wire_recording": {
        "bandlimit_4khz": "PRESERVE",  # Magnetdraht: BW ≤ 4−6 kHz physikalisch
        "mechanical_flutter": "PRESERVE",  # Drahtzug-Mechanik: inhärente Flutter
        "high_noise_floor": "PRESERVE",  # Draht-Rauschboden ~-35 dBFS: Epoche
        "mono_source": "PRESERVE",  # alle Drahtaufnahmen sind Mono
    },
    "minidisc": {
        "atrac_spectral_banding": "PRESERVE",  # ATRAC-Codec-Spektralstruktur ist Format-Charakter
        "bw_ceiling_16khz": "PRESERVE",  # ATRAC-Bandbreite ≤ 16 kHz
        "mild_pre_echo": "PRESERVE",  # leichtes Pre-Echo bei Transienten: Codec-Artefakt
        "severe_pre_echo": "REPAIR",  # starkes Pre-Echo > 30 ms: Defekt
    },
    "dat": {
        "16bit_dither": "PRESERVE",  # 16-bit-Dithering ist Format-Charakter
        "tape_transport_jitter": "PRESERVE",  # minimale Transport-Jitter sind normal
        "dropout_short": "REPAIR",  # kurze Dropouts sind Defekte
    },
    "aac": {
        "psychoacoustic_coding_mild": "PRESERVE",  # AAC 256kbps+ ist nah-transparent
        "sbr_signature": "PRESERVE",  # SBR-Signatur bei HE-AAC: Format-Charakter
        "pre_echo_minor": "PRESERVE",  # minimales Pre-Echo ist Codec-normal
        "severe_pre_echo": "REPAIR",  # starkes Pre-Echo: Defekt
        "metallic_ringing": "REPAIR",  # metallisches Klingen bei niedriger Bitrate: Defekt
    },
    "streaming": {
        # Streaming (Spotify AAC 256 / Apple 256) ≈ AAC High-Quality
        "psychoacoustic_coding_mild": "PRESERVE",
        "loudness_normalization": "PRESERVE",  # LUFS-Normalisierung ist Plattform-Charakter
        "pre_echo_minor": "PRESERVE",
        "severe_pre_echo": "REPAIR",
        "metallic_ringing": "REPAIR",
    },
    "lp": {
        # LP = Vinyl-Langspielplatte — identische Physik wie vinyl
        "groove_distortion_low": "PRESERVE",
        "interlabel_noise_texture": "PRESERVE",
        "riaa_warmth_curve": "PRESERVE",
        "inner_groove_compression": "PRESERVE",
        "low_level_wow_sub1hz": "PRESERVE",
    },
}

#: §6.5b: Maximale Strength für PRESERVE-Merkmale (normal)
PRESERVE_MAX_STRENGTH: float = 0.10

#: §6.5b Ausnahme: artifact_freedom < 0.90 → Strength-Spielraum
PRESERVE_EXCEPTION_MAX_STRENGTH: float = 0.20

#: §6.5c Era-spezifische Charakter-Profile (Authentizität vs. Defekt)
ERA_CHARACTER_PROFILES: dict[tuple[int, int], dict[str, _Classification]] = {
    (1900, 1925): {
        "trichter_resonanz_300_600hz": "PRESERVE",
        "bw_ceiling_3khz": "PRESERVE",
        "mono_image": "PRESERVE",
        # Maßnahme: Kein EQ unter 300 Hz; keine BW-Extension
    },
    (1925, 1945): {
        "tube_h2_h4": "PRESERVE",  # Röhren-H2/H4
        "agc_drift": "PRESERVE",  # AGC-Drift
        "electric_hum_carpet": "AMBIGUOUS",  # Brumm nur bei f ≤ 120 Hz reparieren
    },
    (1945, 1965): {
        "riaa_warmth": "PRESERVE",  # RIAA-Kurve nie als Defekt klassifizieren
        "room_diffusion": "PRESERVE",  # Raumdiffusion Aufnahmestudio
        "needle_noise": "PRESERVE",  # Nadelgeräusch
    },
    (1965, 1980): {
        "tape_compression": "PRESERVE",  # Bandkompression
        "bandwidth_12_14khz": "PRESERVE",  # Bandbreite 12–14 kHz
        "analog_hiss": "PRESERVE",  # Hiss-Boden PRESERVE bis reel_tape-Schwellwert
    },
    (1980, 1995): {
        "dolby_bc_noise": "PRESERVE",  # Dolby-B/C Rauschen
        "cassette_sound": "PRESERVE",  # Kassettenklang
        "dolby_breathing": "PRESERVE",  # Dolby-Atmen wenn < -3 dB relativ zu Signal
        "early_digital_edges": "AMBIGUOUS",  # frühe Digital-Kanten
    },
    (1995, 2010): {
        "mp3_psychoacoustic_residue": "PRESERVE",  # MP3-Psychoakustik-Residuum
        "mild_pre_echo": "PRESERVE",  # leichte Pre-Echos als Format-Signatur
    },
    (2010, 9999): {
        "clean_noise_floor": "PRESERVE",  # Rauschen < -90 dBFS PRESERVE
        "loudness_war_clip": "REPAIR",  # Clipping REPAIR
    },
}


# ---------------------------------------------------------------------------
# Datenklasse
# ---------------------------------------------------------------------------


@dataclass
class ArtifactClassification:
    """Ergebnis der Klassifikation eines Signal-Merkmals."""

    feature_name: str
    material_type: str
    classification: _Classification
    max_strength: float
    confidence: float = 1.0
    reason: str = ""


@dataclass
class IntentionalArtifactResult:
    """Vollständiges Klassifikationsergebnis für ein Audio-Segment."""

    material_type: str
    preserve_features: list[str] = field(default_factory=list)
    repair_features: list[str] = field(default_factory=list)
    ambiguous_features: list[str] = field(default_factory=list)
    strength_caps: dict[str, float] = field(default_factory=dict)
    artifact_freedom_override: bool = False

    def should_cap_strength(self, feature_name: str) -> float:
        """
        Gibt maximale Phase-Strength für ein Feature zurück.
        PRESERVE → PRESERVE_MAX_STRENGTH (0.10).
        REPAIR → 1.0 (kein Cap).
        AMBIGUOUS → 0.40 (Mittelwert).
        """
        return self.strength_caps.get(feature_name, 1.0)


# ---------------------------------------------------------------------------
# Hauptklasse (Singleton)
# ---------------------------------------------------------------------------


class IntentionalArtifactClassifier:
    """§6.5 Authentischer Klangcharakter-Classifier.

    Klassifiziert Signal-Merkmale als PRESERVE oder REPAIR vor DefectScanner-
    CAUSE_TO_PHASES-Auswahl. PRESERVE schlägt DefectScanner-Aktivierung.
    """

    def classify(
        self,
        material_type: str,
        era_decade: int | None = None,
        artifact_freedom: float = 1.0,
    ) -> IntentionalArtifactResult:
        """Gibt Klassifikation für alle bekannten Merkmale des Materials zurück.

        Args:
            material_type:    SUPPORTED_MATERIALS-Schlüssel (z. B. "shellac", "vinyl").
            era_decade:       Aufnahme-Jahrzehnt (4-stellig, z. B. 1935).
                              Ergänzt material-spezifische Klassifikation mit Era-Profil.
            artifact_freedom: Aktueller artifact_freedom-Score [0..1].
                              < 0.90 → PRESERVE-Ausnahme (Strength bis 0.20, §6.5b).

        Returns:
            IntentionalArtifactResult mit PRESERVE/REPAIR/AMBIGUOUS-Listen
            und Strength-Caps pro Feature.
        """
        result = IntentionalArtifactResult(material_type=material_type)
        _exception_active = artifact_freedom < 0.90
        _max_preserve = PRESERVE_EXCEPTION_MAX_STRENGTH if _exception_active else PRESERVE_MAX_STRENGTH
        if _exception_active:
            result.artifact_freedom_override = True
            logger.debug(
                "IAC: artifact_freedom=%.3f < 0.90 → PRESERVE exception active (max_strength=%.2f)",
                artifact_freedom,
                _max_preserve,
            )

        # Material-spezifische Merkmale
        material_features = AUTHENTIC_CHARACTER.get(material_type, {})
        for feature, classification in material_features.items():
            _register_feature(result, feature, classification, _max_preserve)

        # Era-ergänzende Klassifikation
        if era_decade is not None:
            for (yr_start, yr_end), era_features in ERA_CHARACTER_PROFILES.items():
                if yr_start <= era_decade <= yr_end:
                    for feature, classification in era_features.items():
                        # Era-Profil überschreibt nur, wenn noch nicht aus Material bekannt
                        if feature not in material_features:
                            _register_feature(result, feature, classification, _max_preserve)
                    break

        logger.debug(
            "IAC material=%s era=%s: preserve=%d repair=%d ambiguous=%d override=%s",
            material_type,
            era_decade,
            len(result.preserve_features),
            len(result.repair_features),
            len(result.ambiguous_features),
            result.artifact_freedom_override,
        )
        return result

    def get_strength_cap(
        self,
        material_type: str,
        defect_type_key: str,
        artifact_freedom: float = 1.0,
    ) -> float:
        """Schneller Einzelabruf: maximale Phase-Strength für einen Defekttyp.

        Wird von UV3 _select_phases() und PhaseConductor aufgerufen, um
        PRESERVE-Stärke-Caps vor Phase-Ausführung zu erzwingen.

        Returns:
            float ∈ [0.10, 1.0]. 1.0 = kein Cap (REPAIR-Klasse).
        """
        _DEFECT_TO_FEATURE_MAP: dict[str, str] = {
            # DefectScanner-Schlüssel → AUTHENTIC_CHARACTER-Feature
            "HIGH_FREQ_NOISE": "surface_noise_texture",
            "SOFT_SATURATION": "h2_h4_harmonic_saturation",
            "BANDWIDTH_LOSS": "bandwidth_ceiling_8khz",
            "LOW_FREQ_RUMBLE": "low_level_wow_sub1hz",
            "WOW": "low_level_wow_sub1hz",
            "HUM": "tape_saturation_knee",
            "TAPE_HISS": "tape_hiss_floor_texture",
            "BIAS_NOISE": "bias_noise_texture",
            "QUANTIZATION_NOISE": "dithering_noise_floor",
            "PRE_ECHO": "pre_echo_character_mild",
            "COMPRESSION_ARTIFACTS": "psychoacoustic_residue",
        }
        feature = _DEFECT_TO_FEATURE_MAP.get(defect_type_key)
        if feature is None:
            return 1.0  # kein Mapping → kein Cap

        material_features = AUTHENTIC_CHARACTER.get(material_type, {})
        classification = material_features.get(feature)
        if classification == "PRESERVE":
            _exception = artifact_freedom < 0.90
            return PRESERVE_EXCEPTION_MAX_STRENGTH if _exception else PRESERVE_MAX_STRENGTH
        if classification == "AMBIGUOUS":
            return 0.40
        return 1.0  # REPAIR → kein Cap

    def get_preserve_mask(
        self,
        audio: np.ndarray,  # pylint: disable=unused-argument
        sr: int,
        iac_result: IntentionalArtifactResult | None = None,
        material_type: str | None = None,
    ) -> np.ndarray:
        """§2.44 §4.8a-ii Spektrale Schutzmaske für PRESERVE-Merkmale.

        Erzeugt eine spektrale Maske (1-D, shape: n_fft//2+1 = 1025 Bins
        für n_fft=2048 @ 48 kHz), die NR-Algorithmen in phase_03 und
        phase_29 mitgegeben wird.

        Formel-Integration (§4.8a-ii)::

            G_effective = mask * G_PRESERVE_FLOOR + (1 - mask) * G_computed

        G_PRESERVE_FLOOR = 0.90 → in PRESERVE-Bins nahezu kein NR.

        Args:
            audio:        Eingabe-Audio (unused for spectral shape; kept for API).
            sr:           Sample-Rate (Standard 48000).
            iac_result:   Vorberechnetes IntentionalArtifactResult (optional).
            material_type: Materialschlüssel — wird genutzt wenn iac_result None.

        Returns:
            np.ndarray shape (1025,) float32, Werte 0.0 (REPAIR) – 1.0 (PRESERVE).
            Bei Fehler: Zero-Array (non-blocking).
        """
        # §4.8a-ii: Feature → (freq_min_hz, freq_max_hz, mask_value)
        # Mask = 1.0 → Gain bleibt bei G_PRESERVE_FLOOR (0.90) = fast kein NR
        # Mask = 0.0 → normales NR (G_computed)
        _FEATURE_BANDS: dict[str, tuple[float, float, float]] = {
            # ── Shellac ──
            "surface_noise_texture": (0.0, 24000.0, 0.35),
            "h2_h4_harmonic_saturation": (2000.0, 8000.0, 0.80),
            "bandwidth_ceiling_8khz": (8000.0, 24000.0, 1.00),
            "mono_center_image": (0.0, 0.0, 0.00),  # time-domain only
            "soft_transients": (0.0, 0.0, 0.00),  # time-domain only
            # ── Vinyl ──
            "groove_distortion_low": (50.0, 500.0, 0.45),
            "interlabel_noise_texture": (0.0, 24000.0, 0.30),
            "riaa_warmth_curve": (50.0, 500.0, 0.50),
            "inner_groove_compression": (20.0, 300.0, 0.40),
            "low_level_wow_sub1hz": (20.0, 60.0, 0.60),
            # ── Tape / Reel Tape ──
            "tape_saturation_knee": (2000.0, 8000.0, 0.60),
            "high_frequency_rolloff": (12000.0, 24000.0, 0.80),
            "bias_noise_texture": (15000.0, 24000.0, 0.80),
            "dolby_breathing_slight": (6000.0, 14000.0, 0.30),
            "tape_compression_even_harmonics": (1000.0, 6000.0, 0.50),
            "studio_ambience_bleed": (0.0, 24000.0, 0.40),
            "tape_hiss_floor_texture": (0.0, 24000.0, 0.30),
            # ── Wax Cylinder ──
            "trichter_bandlimit_3khz": (3000.0, 24000.0, 1.00),
            "surface_crackle_fine": (0.0, 24000.0, 0.25),
            "mechanical_resonance": (300.0, 600.0, 0.50),
            # ── CD / Digital ──
            "dithering_noise_floor": (0.0, 24000.0, 0.10),
            "pre_emphasis_curve": (5000.0, 20000.0, 0.30),
            "linear_phase_character": (0.0, 24000.0, 0.10),
            "quantization_artifacts_mild": (0.0, 24000.0, 0.10),
            # ── MP3 ──
            "pre_echo_character_mild": (0.0, 24000.0, 0.20),
            "psychoacoustic_residue": (0.0, 24000.0, 0.30),
            # ── Lacquer Disc ──
            "substrate_texture": (0.0, 24000.0, 0.25),
            "light_surface_clicks": (0.0, 24000.0, 0.20),
            # ── Era Profile Features ──
            "trichter_resonanz_300_600hz": (300.0, 600.0, 0.60),
            "bw_ceiling_3khz": (3000.0, 24000.0, 1.00),
            "tube_h2_h4": (2000.0, 8000.0, 0.70),
            "agc_drift": (50.0, 200.0, 0.30),
            "electric_hum_carpet": (0.0, 0.0, 0.00),
            "riaa_warmth": (50.0, 500.0, 0.50),
            "room_diffusion": (0.0, 24000.0, 0.30),
            "needle_noise": (0.0, 24000.0, 0.25),
            "tape_compression": (2000.0, 8000.0, 0.50),
            "bandwidth_12_14khz": (12000.0, 24000.0, 0.70),
            "analog_hiss": (0.0, 24000.0, 0.30),
            "dolby_bc_noise": (0.0, 24000.0, 0.25),
            "cassette_sound": (0.0, 24000.0, 0.20),
            "dolby_breathing": (6000.0, 14000.0, 0.25),
            "early_digital_edges": (0.0, 0.0, 0.00),
            "mp3_psychoacoustic_residue": (0.0, 24000.0, 0.25),
            "mild_pre_echo": (0.0, 24000.0, 0.20),
            "clean_noise_floor": (0.0, 24000.0, 0.10),
        }
        _N_FFT = 2048
        _N_BINS = _N_FFT // 2 + 1  # 1025
        try:
            if iac_result is not None:
                preserve_features = list(iac_result.preserve_features)
            elif material_type is not None:
                _res = self.classify(material_type)
                preserve_features = list(_res.preserve_features)
            else:
                logger.debug("get_preserve_mask: keine iac_Ergebnis und kein material_type → Zero-Maske")
                return np.zeros(_N_BINS, dtype=np.float32)  # type: ignore[no-any-return]

            _sr_eff = max(int(sr), 44100)
            freqs = np.linspace(0.0, _sr_eff / 2.0, _N_BINS)
            mask = np.zeros(_N_BINS, dtype=np.float32)

            for _feat in preserve_features:
                _band = _FEATURE_BANDS.get(_feat)
                if _band is None:
                    continue
                _f_min, _f_max, _val = _band
                if _val <= 0.0 or _f_min >= _f_max:
                    continue
                _b_lo = int(np.searchsorted(freqs, _f_min))
                _b_hi = int(np.searchsorted(freqs, _f_max, side="right"))
                _b_lo = int(np.clip(_b_lo, 0, _N_BINS - 1))
                _b_hi = int(np.clip(_b_hi, _b_lo + 1, _N_BINS))
                # Max-Projektion: jeder Bin nimmt höchsten Preserve-Wert aller aktiven Features
                mask[_b_lo:_b_hi] = np.maximum(mask[_b_lo:_b_hi], _val)

            logger.debug(
                "§4.8a-ii preserve_mask: features=%d n_bins=%d active_bins=%d max=%.2f",
                len(preserve_features),
                _N_BINS,
                int((mask > 0.0).sum()),
                float(mask.max()),
            )
            return mask  # type: ignore[no-any-return]

        except Exception as _exc:
            logger.debug("get_preserve_mask nicht blockierend: %s", _exc)
            return np.zeros(_N_BINS, dtype=np.float32)  # type: ignore[no-any-return]


def _register_feature(
    result: IntentionalArtifactResult,
    feature: str,
    classification: _Classification,
    max_preserve_strength: float,
) -> None:
    """Hilfsfunktion: Feature in Ergebnis eintragen und Strength-Cap setzen."""
    if classification == "PRESERVE":
        result.preserve_features.append(feature)
        result.strength_caps[feature] = max_preserve_strength
    elif classification == "REPAIR":
        result.repair_features.append(feature)
        result.strength_caps[feature] = 1.0
    elif classification == "AMBIGUOUS":
        result.ambiguous_features.append(feature)
        result.strength_caps[feature] = 0.40


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: IntentionalArtifactClassifier | None = None
_lock = threading.Lock()


def get_intentional_artifact_classifier() -> IntentionalArtifactClassifier:
    """Thread-sicherer Singleton-Accessor."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = IntentionalArtifactClassifier()
    return _instance


# ── Convenience-Funktionen (für Dead-Import-Reparatur) ───────────────

def classify_intentional_artifacts(
    audio: np.ndarray,
    sr: int = 48000,
    material: str = "unknown",
    era_decade: int | None = None,
) -> IntentionalArtifactResult:
    """Convenience-Funktion für intentional artifact classification.

    Args:
        audio: Audio-Signal (float32)
        sr: Sample-Rate in Hz
        material: Material-Typ (shellac, vinyl, tape, etc.)
        era_decade: Ära-Dekade (z.B. 1950 für 1950er Jahre)

    Returns:
        IntentionalArtifactResult mit Klassifizierung-Ergebnissen
    """
    classifier = get_intentional_artifact_classifier()
    return classifier.classify(audio, sr, material=material, era_decade=era_decade)
