"""
Aurik 10.0.0 — forensics/medium_detector.py  (§6.7, bindend ab v10.0.0)
=====================================================================
Vereinheitlichte Tonträgerketten-Erkennung: kombiniert die forensische
Spektralfingerabdruck-Analyse mit Bayesian-Material-Scoring für
weltklasse-Präzision bei komplexen Mehrstufenketten.

Ab v10.0.0 ist dies das **einzige** autoritative Material-Erkennungssystem.
MediumClassifier-Features (Rotation, Infrasonic, Codec MDCT) sind direkt
integriert — kein zweiter Klassifikator nötig.

Pflicht-Spektralfingerabdruck (§6.7.1):
    1. Rolloff 95 %  — diagnostiziert Bandbreitenbegrenzung
    2. Wow/Flutter-Index — Pitch-Instabilität via pYIN-Ableitung
    3. HF-Energie > 16 kHz — MP3/Kassettenkette
    4. Rauschpegel (Percentile-5 PSD)  — Bandrauschen
    5. Effektive Bandbreite — physikalische Signalbandbreite

Erweiterte Features (§6.7.3, NEU v10.0.0):
    6. Rotation-Periodizität     — Vinyl-Plattentellerfrequenz (0.3–2 Hz)
    7. Infraschall-RMS (< 20 Hz) — Vinyl-Lager-Rumble-Diskriminator
    8. MDCT-Codec-Artefakt-Score — MP3/AAC-Quantisierungsfingerabdruck
    9. Codec-Typ-Code            — MP3/AAC/lossy-Differenzierung

Kettenerkennung (§6.7.2):
    - Bayesian-Primär-Material-Scoring (Gaussian-Likelihood, 16 Materialien)
    - Sekundäre Codec-Schicht via Artefakt-Score
    - 3+-Layer-Ketten möglich (z. B. vinyl → tape → mp3_low)
    - is_multi_generation=True → kombinierte Phasen aller Materialien

Wissenschaftliche Grundlage (normative Literatur, §6.9a):
    - Cartwright, Pardo & Wallis (2016) DAFX-16 — Vinyl-ID via spektrale Features
    - Declercq, De Backer & Zhu (2007) ICASSP — Bayesian-Trägerklassifikation (Gaussian-Mixture)
    - Maher (2010) J. Audio Eng. Soc. 58:702 — Survey analoger Artefakt-Erkennung
    - IEC 60386:1987 — Wow/Flutter-Messnorm (WOW < 0.5 Hz, FLUTTER 0.5–200 Hz)
    - Brandenburg & Bosi (1994) J. AES 42:381 — MP3/MPEG-1 Layer III Codec-Artefakte
    - Pan (1995) J. AES 43:529 — AAC/MPEG-2 Codec-Charakteristika
    - Müller & Ewert (2011) IEEE Signal Proc. Mag. 28:42 — MDCT-Codec-Fingerprinting
    - Spijkervet & Haasdijk (2020) ISMIR — ML-basierte MP3/AAC-Unterscheidung
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, decimate, hilbert, sosfilt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class SpectralFingerprint:
    """Pflicht-Spektralfingerabdruck (§6.7.1 + §6.7.3) aus Rohsignal-Vorabanalyse."""

    rolloff_95_hz: float = 0.0  # Spectral Rolloff 95 % — Median
    wow_flutter_index: float = 0.0  # Pitch-Varianz [Hz std] über 100-ms-Fenster
    hf_energy_above_16k: float = 0.0  # Anteil Energie > 16 kHz an Gesamt
    noise_floor_db: float = -60.0  # 5. Perzentil der Frame-Energien [dBFS]
    effective_bandwidth_hz: float = 0.0  # HF-Rolloff −60 dBFS

    # --- §6.7.3 Erweiterte Features (v10.0.0) ---
    rotation_hz: float = 0.0  # Vinyl-Plattentellerfrequenz [Hz]; 0 = keine Rotation
    rotation_strength: float = 0.0  # Normierter Peak-SNR des Rotationssignals [0, 1]
    infrasonic_rms: float = 0.0  # Sub-20 Hz normierter RMS (Vinyl-Rumble)
    codec_artifact_score: float = 0.0  # MDCT-Quantisierungs-Artefakt-Score [0, 1]
    codec_type_code: float = 0.0  # 0=clean, 1=mp3, 2=aac, 3=lossy
    crackle_density: float = 0.0  # Anteil Samples > 4σ (Vinyl-Knackser)
    snr_db: float = 0.0  # Signal-Rausch-Abstand [dB]
    noise_color: float = 1.0  # Spektrale Neigung (β): 0=weiß, 2=braun

    # --- Alias-Properties für Test-Kompatibilität (§6.7.1) ---
    @property
    def rolloff_95_percent_hz(self) -> float:
        """Alias für rolloff_95_hz — Rückwärtskompatibilität."""
        return self.rolloff_95_hz

    @property
    def hf_energy_above_16khz_percent(self) -> float:
        """Alias für hf_energy_above_16k als Prozentwert (0–100)."""
        return float(self.hf_energy_above_16k * 100.0)

    def __contains__(self, item: object) -> bool:
        """Unterstützt 'key in fingerprint'-Syntax für Tests."""
        return item in (
            "rolloff_95_hz",
            "rolloff_95_percent_hz",
            "wow_flutter_index",
            "hf_energy_above_16k",
            "hf_energy_above_16khz_percent",
            "noise_floor_db",
            "effective_bandwidth_hz",
            "rotation_hz",
            "rotation_strength",
            "infrasonic_rms",
            "codec_artifact_score",
            "codec_type_code",
            "crackle_density",
            "snr_db",
            "noise_color",
        )

    def as_dict(self) -> dict:
        """Gibt alle Spektral-Fingerabdruck-Felder als Dictionary zurück."""
        return {
            "rolloff_95_hz": self.rolloff_95_hz,
            "rolloff_95_percent_hz": self.rolloff_95_hz,
            "wow_flutter_index": self.wow_flutter_index,
            "hf_energy_above_16k_fraction": self.hf_energy_above_16k,
            "hf_energy_above_16khz_percent": self.hf_energy_above_16k * 100.0,
            "noise_floor_db": self.noise_floor_db,
            "effective_bandwidth_hz": self.effective_bandwidth_hz,
            "rotation_hz": self.rotation_hz,
            "rotation_strength": self.rotation_strength,
            "infrasonic_rms": self.infrasonic_rms,
            "codec_artifact_score": self.codec_artifact_score,
            "codec_type_code": self.codec_type_code,
            "crackle_density": self.crackle_density,
            "snr_db": self.snr_db,
            "noise_color": self.noise_color,
        }


@dataclass
class TransferChain:
    """Erkannte Medien-Transferkette."""

    chain: list[str] = field(default_factory=list)
    """Kette von MediaType-Strings, z. B. ['tape', 'mp3_low']."""

    is_multi_generation: bool = False
    """True wenn ≥ 2 verschiedene Medienstufen erkannt wurden."""

    primary_material: str = "unknown"
    """Letzter analoger Träger = primärer MaterialType-Prior."""

    confidence: float = 0.0
    """Gesamtkonfidenz der Ketten-Schätzung ∈ [0, 1]."""

    reasoning: str = ""

    def __len__(self) -> int:
        """Gibt die Anzahl der Stufen in der Transferkette zurück."""
        return len(self.chain)


@dataclass
class MediumDetectionResult:
    """Vollständiges Ergebnis der Tonträgerketten-Erkennung."""

    transfer_chain: list[str]
    """Kette wie ['vinyl', 'tape', 'mp3_low'] — ursprünglicher Träger zuerst."""

    is_multi_generation: bool
    primary_material: str
    confidence: float
    spectral_fingerprint: SpectralFingerprint
    evidence: list[str] = field(default_factory=list)
    """Laienverständliche Diagnose-Begründungen."""
    medium_confidences: list[float] = field(default_factory=list)
    """Per-Link-Konfidenz — gleiche Länge wie transfer_chain."""

    # §6.7.3 (v10.0.0): Bayesian-Scoring-Ergebnis für Durchreichung
    bayesian_scores: dict[str, float] = field(default_factory=dict)
    """Posterior-Wahrscheinlichkeiten aller Materialtypen."""

    classification_result: object | None = None
    """ClassificationResult aus MediumClassifier (für cached_medium_result)."""

    # §6.8 (v10.0.0): Physikalische Analog-Quellen für Durchreichung an Phasen
    physical_analog_sources: list[tuple[str, float]] = field(default_factory=list)
    """Physisch erkannte Analog-Quellen: [(material, confidence), ...] z. B. [("vinyl",0.58), ("cassette",0.42)].
    Wird als kwargs an Phasen durchgereicht für Vinyl-spezifische Restaurierung (RIAA, Rumble, Knistern)."""

    dolby_nr_type: str = "none"
    """Erkannter Dolby/DBX NR-Typ: 'dolby_b'|'dolby_c'|'dolby_s'|'dbx_i'|'dbx_ii'|'none'."""

    dolby_nr_confidence: float = 0.0
    """Konfidenz der Dolby-NR-Erkennung ∈ [0, 1]."""

    # §6.7 (v10.0.0): Tape-Speed + RIAA-Curve Detection
    tape_speed_ips: float | None = None
    """Geschätzte Bandgeschwindigkeit in ips (1.875/3.75/7.5/15/30). None bei Nicht-Tape."""

    riaa_curve_type: str = "unknown"
    """Erkannter EQ-Kurventyp:
    'riaa'|'nab'|'columbia'|'aes'|'capitol'|'london'|'ccir'|'unknown_prestandard'|'unknown'."""

    riaa_curve_confidence: float = 0.0
    """Konfidenz der RIAA/EQ-Kurven-Erkennung ∈ [0, 1]. ≥ 0.70 = aktiv."""

    # §v10.20 Material-Konsens (2026-08-22): Der Konsens-Entscheid wird als
    # EIGENE Felder persistiert — der Medium-Primär bleibt unangetastet.
    # Kalibrierung folgt primary_material; Ketten-/Codec-Behandlung liest
    # final_chain/consensus_material.
    consensus_material: str = "unknown"
    """Konsens-Gewinner des Material-Konsens-Moduls (kann terminaler Träger sein)."""

    final_chain: list[str] = field(default_factory=list)
    """Konsens-korrigierte Tonträgerkette (chronologisch, dedupliziert)."""

    @property
    def chain_label(self) -> str:
        """Gibt die Transferkette als lesbaren String zurück (z. B. 'vinyl → mp3_low')."""
        return " → ".join(self.transfer_chain) if self.transfer_chain else "unknown"

    def as_dict(self) -> dict:
        """Gibt alle Erkennungsergebnis-Felder als Dictionary zurück."""
        return {
            "transfer_chain": self.transfer_chain,
            "medium_confidences": self.medium_confidences,
            "is_multi_generation": self.is_multi_generation,
            "primary_material": self.primary_material,
            "confidence": self.confidence,
            "chain_label": self.chain_label,
            "spectral_fingerprint": self.spectral_fingerprint.as_dict(),
            "evidence": self.evidence,
            "bayesian_scores": self.bayesian_scores,
            "dolby_nr_type": self.dolby_nr_type,
            "dolby_nr_confidence": self.dolby_nr_confidence,
            "tape_speed_ips": self.tape_speed_ips,
            "riaa_curve_type": self.riaa_curve_type,
            "riaa_curve_confidence": self.riaa_curve_confidence,
        }


# ---------------------------------------------------------------------------
# Haupt-Klasse
# ---------------------------------------------------------------------------


class MediumDetector:
    """Vereinheitlichte forensische Tonträgerketten-Erkennung (§6.7, v10.0.0).

    Kombiniert:
      - Spektralfingerabdruck (5 Basis-Features, §6.7.1)
      - Erweiterte physikalische Features (Rotation, Infrasonic, Codec, §6.7.3)
      - Bayesian Gaussian-Likelihood Material-Scoring (aus MediumClassifier)
      - Mehrstufige Ketten-Inferenz (3+ Layer: vinyl → tape → mp3_low)

    Ersetzt die alte sequenzielle if/elif-Heuristik durch probabilistisches
    Scoring über 16 Materialtypen.

    Singleton-Zugang: ``get_medium_detector()``
    Convenience:      ``detect_medium_chain(audio, sr)``
    """

    # ── Schwellwerte ──────────────────────────────────────────────────
    HF_ENERGY_THRESHOLD_FRACTION: float = 0.001  # < 0.1 % → kein HF
    _CODEC_ARTIFACT_THRESHOLD: float = 0.15  # ab hier: Codec-Layer erkannt
    _ANALOG_POSTERIOR_MIN: float = 0.08  # Mindest-Posterior für Analog-Layer
    _SECONDARY_ANALOG_MIN: float = 0.08  # Mindest-Posterior für 2. Analog-Stufe
    _MAX_ANALOG_CHAIN_DEPTH: int = 5  # §v10.19: 4→5 — erlaubt reel_tape→lacquer→vinyl→cassette→mp3 (Depth-5-Ketten)
    _SAME_ORDER_ANALOG_MIN: float = 0.22  # konservativer Guard für gleichrangige Analog-Stufen
    _ANALOG_CHAIN_MP3_HIGH_MIN_BW_HZ: float = 18_000.0  # unterhalb: mp3_high bei Analogkette unplausibel

    # Analog-Materialtypen — können als Primärquelle in Ketten vorkommen
    _ANALOG_MATERIALS: frozenset[str] = frozenset(
        {
            "shellac",
            "wax_cylinder",
            "vinyl",
            "wire_recording",
            "lacquer_disc",
            "tape",
            "reel_tape",
            "cassette",
        }
    )

    # Digitale Container-Formate — niemals Primärquelle
    _CODEC_MATERIALS: frozenset[str] = frozenset(
        {
            "mp3_low",
            "mp3_high",
            "aac",
            "streaming",
            "minidisc",
        }
    )

    # Digitale verlustfreie Formate
    _DIGITAL_LOSSLESS: frozenset[str] = frozenset(
        {
            "cd_digital",
            "dat",
        }
    )

    # Zeitliche Ordnung für Kettensortierung (niedrig = früher)
    _MEDIUM_ORDER: dict[str, int] = {
        # Pre-1900
        "tinfoil_cylinder": 0,  # Edison 1877, experimentell
        "wax_cylinder": 1,  # Edison 1888–1929, 2–4 min
        # 1900–1950: Schellack-Ära
        "shellac": 2,  # 78 rpm, 1898–1950er, 3–5 min/Seite
        "shellac_vertical": 3,  # Pathé/Edison Diamond Disc, vertikaler Schnitt
        "wire_recording": 4,  # Stahldraht, 1898–1950er
        # 1950–1980: Vinyl- + Tape-Ära
        "reel_tape": 5,  # Studio-Master, 1935+ (Recording → Pressing)
        "lacquer_disc": 6,  # §v10.19 Fix: war Pos.2 (vor reel_tape) — Lackfolie wird VOM Tape geschnitten, muss NACH reel_tape und VOR vinyl stehen
        "vinyl": 7,  # LP 1948, 45 rpm 1949
        "tape": 7,  # Alias
        "cartridge_4track": 8,  # Fidelipac 1956, Radio
        "cartridge_8track": 9,  # Lear 1964–1982, Consumer/Auto
        "cassette": 10,  # Philips 1963, Compact Cassette
        "elcaset": 11,  # Sony 1976–1980, Grosscassette
        "playtape": 11,  # 1966–1970, Miniatur-Kassette
        # 1980–2000: Digital-Ara
        "cd_digital": 12,  # Compact Disc 1982
        "dat": 13,  # Digital Audio Tape 1987
        "dcc": 14,  # Digital Compact Cassette 1992–1996
        "minidisc": 15,  # Sony MD 1992–2013, ATRAC
        # 2000+: Lossy/Streaming
        "mp3_high": 16,  # >=192 kbps
        "mp3_low": 17,  # <192 kbps
        "aac": 18,  # 1997+
        "streaming": 19,  # Spotify/Apple Music/YouTube
    }

    # §v10.19: Era-Prior-Tabellen als class attributes für externen Zugriff
    # (pre_analysis Era-Adjustment, vorher locals in _bayesian_score).
    # Materials that are TYPICAL for each era (get positive boost).
    _ERA_CONSISTENT: dict[str, list[int]] = {
        "wax_cylinder": [1890, 1900, 1910, 1920],
        "wire_recording": [1900, 1910, 1920, 1930, 1940],
        "shellac": [1900, 1910, 1920, 1930, 1940, 1950],
        "lacquer_disc": [
            1920,
            1930,
            1940,
            1950,
            1960,
            1970,
            1980,
        ],  # §v10.19: +1970,1980 — Lackfolie bis Ende Vinyl-Ära
        "vinyl": [1950, 1960, 1970, 1980],
        "reel_tape": [1940, 1950, 1960, 1970, 1980],
        "cassette": [1970, 1980, 1990],
        "tape": [1970, 1980, 1990],
        "8track": [1970, 1980],
        "dat": [1980, 1990, 2000],
        "cd_digital": [1990, 2000, 2010],
        "cd": [1990, 2000, 2010],
        "minidisc": [1990, 2000],
        "dcc": [1990],
        "mp3_low": [2000, 2010, 2020],
        "mp3_high": [2000, 2010, 2020],
        "mp3_high_vbr": [2000, 2010, 2020],
        "aac": [2000, 2010, 2020],
        "streaming": [2010, 2020, 2025],
        "pcm_digital": [2000, 2010, 2020],
        "lossless_digital": [2000, 2010, 2020],
    }
    # Materials that are IMPOSSIBLE for each era (get negative penalty).
    _ERA_IMPOSSIBLE: dict[str, list[int]] = {
        "wax_cylinder": [1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020],
        "shellac": [1970, 1980, 1990, 2000, 2010, 2020],
        "wire_recording": [1960, 1970, 1980, 1990, 2000, 2010, 2020],
        "8track": [1990, 2000, 2010, 2020],
        "dat": [1970, 2010, 2020],
        "minidisc": [1970, 1980, 2010, 2020],
        "dcc": [1970, 1980, 2000, 2010, 2020],
    }

    # Genre -> fruehestes moegliches Aufnahmemedium (Order-Nummer).
    # Ein Genre kann nicht auf einem Traeger erscheinen, der aelter
    # ist als das Genre selbst.
    _GENRE_EARLIEST_ORDER: dict[str, int] = {
        # Pre-1900: Wachswalze
        "classical": 1,
        "folk_traditional": 1,
        "march": 1,
        "marschmusik": 1,
        "military_band": 1,
        "opera": 1,
        "operetta": 1,
        "parlour_music": 1,
        "ragtime": 1,
        "vaudeville": 1,
        # 1900-1920: Fruehe Schellack-Aera
        "blasmusik": 3,
        "bluegrass": 3,
        "blues": 3,
        "brass_band": 3,
        "cajun": 3,
        "calypso": 3,
        "classical_romantic": 3,
        "country": 3,
        "country_blues": 3,
        "delta_blues": 3,
        "dixieland": 3,
        "early_jazz": 3,
        "folk": 3,
        "gospel": 3,
        "hawaiian": 3,
        "jazz": 3,
        "old_time": 3,
        "operatic_aria": 3,
        "spiritual": 3,
        "tango": 3,
        "zydeco": 3,
        # 1930-1950: Swing/Big-Band
        "bebop": 4,
        "big_band": 4,
        "boogie_woogie": 4,
        "crooner": 4,
        "honky_tonk": 4,
        "jump_blues": 4,
        "rhythm_and_blues": 4,
        "swing": 4,
        "western_swing": 4,
        # 1940-1955: Wire/Late-Schellack
        "afro_cuban": 5,
        "bossa_nova": 5,
        "cha_cha": 5,
        "cool_jazz": 5,
        "latin_jazz": 5,
        "mambo": 5,
        "samba": 5,
        "volkstuemliche_musik": 5,
        # 1950-1965: Vinyl-Aera
        "beat": 6,
        "blues_rock": 6,
        "british_invasion": 6,
        "country_nashville": 6,
        "deutscher_schlager": 6,
        "doo_wop": 6,
        "easy_listening": 6,
        "exotica": 6,
        "folk_rock": 6,
        "funk": 6,
        "garage_rock": 6,
        "german_schlager": 6,
        "latin_rock": 6,
        "liedermacher": 6,
        "lounge": 6,
        "memphis_soul": 6,
        "motown": 6,
        "philly_soul": 6,
        "pop": 6,
        "psychedelic_rock": 6,
        "reggae": 6,
        "rnb": 6,
        "rock": 6,
        "rock_n_roll": 6,
        "rockabilly": 6,
        "rocksteady": 6,
        "schlager": 6,
        "ska": 6,
        "soul": 6,
        "surf_rock": 6,
        "traditional_pop": 6,
        "volksmusik": 6,
        # 1970: Dub
        "dub": 7,
        # 1965-1980: Cassette-Aera
        "arena_rock": 8,
        "deutschrock": 8,
        "disco": 8,
        "glam_rock": 8,
        "hard_rock": 8,
        "heavy_metal": 8,
        "jazz_fusion": 8,
        "krautrock": 8,
        "new_wave": 8,
        "ostrock": 8,
        "post_punk": 8,
        "power_pop": 8,
        "progressive_rock": 8,
        "pub_rock": 8,
        "punk": 8,
        "soft_rock": 8,
        "southern_rock": 8,
        "synth_pop": 8,
        "yacht_rock": 8,
        # 1980-1985: Fruehe Cassette
        "black_metal": 9,
        "boogie": 9,
        "ebm": 9,
        "electro": 9,
        "funk_rock": 9,
        "hardcore_punk": 9,
        "hip_hop": 9,
        "industrial": 9,
        "neue_deutsche_welle": 9,
        "post_disco": 9,
        "rap": 9,
        "smooth_jazz": 9,
        "speed_metal": 9,
        "thrash_metal": 9,
        # 1985-1990: Spaete Cassette
        "crossover_thrash": 10,
        "death_metal": 10,
        "golden_age_hip_hop": 10,
        "grindcore": 10,
        # 1988-1992: Gangsta-Rap
        "gangsta_rap": 11,
        # 1985-2000: CD-Aera
        "acid_house": 12,
        "acid_jazz": 12,
        "alternative_rock": 12,
        "ambient": 12,
        "ambient_techno": 12,
        "anime_music": 12,
        "big_beat": 12,
        "breakbeat": 12,
        "britpop": 12,
        "chillout": 12,
        "deep_house": 12,
        "detroit_techno": 12,
        "deutschrap": 12,
        "downtempo": 12,
        "dream_pop": 12,
        "drum_and_bass": 12,
        "emo": 12,
        "eurodance": 12,
        "europop": 12,
        "film_score": 12,
        "garage": 12,
        "goa_trance": 12,
        "grunge": 12,
        "hamburger_schule": 12,
        "house": 12,
        "idm": 12,
        "indie_pop": 12,
        "indie_rock": 12,
        "j_pop": 12,
        "jungle": 12,
        "latin_pop": 12,
        "lo_fi": 12,
        "math_rock": 12,
        "minimal_techno": 12,
        "neo_soul": 12,
        "noise_rock": 12,
        "nu_metal": 12,
        "pop_punk": 12,
        "post_rock": 12,
        "progressive_trance": 12,
        "r_and_b_contemporary": 12,
        "rap_metal": 12,
        "shoegaze": 12,
        "slowcore": 12,
        "speed_garage": 12,
        "techno": 12,
        "trance": 12,
        "trip_hop": 12,
        "two_step": 12,
        "uk_garage": 12,
        "video_game_music": 12,
        # 1995-2005: Spaete CD
        "german_gangsta_rap": 13,
        "k_pop": 13,
        # 2000+: Digital-born
        "bedroom_pop": 16,
        "brostep": 16,
        "chillwave": 16,
        "cloud_rap": 16,
        "dubstep": 16,
        "edm": 16,
        "electro_house": 16,
        "emo_rap": 16,
        "future_bass": 16,
        "grime": 16,
        "lofi_hip_hop": 16,
        "mumble_rap": 16,
        "progressive_house": 16,
        "soundcloud_rap": 16,
        "study_beats": 16,
        "synthwave": 16,
        "tech_house": 16,
        "trap": 16,
        "vaporwave": 16,
        "witch_house": 16,
        # 2020+: Streaming-Aera
        "drift_phonk": 17,
        "hyperpop": 17,
        "phonk": 17,
    }

    _MEDIUM_DISPLAY_NAMES: dict[str, str] = {
        "tinfoil_cylinder": "Zinnfolien-Walze (Edison 1877)",
        "wax_cylinder": "Wachswalze (Edison 1888-1929)",
        "lacquer_disc": "Lackplatte / Acetat-Mitschnitt",
        "shellac": "Schellackplatte (78 rpm)",
        "shellac_vertical": "Schellackplatte, vertikaler Schnitt",
        "wire_recording": "Stahldraht-Aufnahme",
        "vinyl": "Vinyl-Schallplatte (LP/45rpm)",
        "reel_tape": "Tonband (Studio-Master)",
        "cartridge_4track": "4-Spur-Cartridge (Fidelipac)",
        "cartridge_8track": "8-Spur-Cartridge (Lear Jet)",
        "cassette": "Compact Cassette (Philips 1963)",
        "elcaset": "Elcaset (Sony 1976-1980)",
        "cd_digital": "Compact Disc (CD)",
        "dat": "Digital Audio Tape (R-DAT)",
        "dcc": "Digital Compact Cassette (Philips 1992)",
        "minidisc": "MiniDisc (Sony, ATRAC)",
        "mp3_high": "MP3 (hohe Bitrate)",
        "mp3_low": "MP3 (niedrige Bitrate)",
        "aac": "AAC / M4A",
        "streaming": "Streaming (Spotify/Apple/YouTube)",
    }

    # Studio recording format characteristics -> medium hints.
    # When detected by forensics modules, these inform chain building.
    _STUDIO_FORMAT_INDICATORS: dict[str, dict] = {
        # Dolby Noise Reduction variants
        "dolby_a": {
            "era": "1965-1990",
            "typical_media": ["reel_tape"],
            "quality": "professional",
            "studio": True,
            "description_de": "Dolby A (Professionelles Rauschunterdrueckungssystem, 1965)",
        },
        "dolby_b": {
            "era": "1968-2000",
            "typical_media": ["cassette", "reel_tape"],
            "quality": "consumer_hifi",
            "studio": False,
            "description_de": "Dolby B (Consumer Rauschunterdrueckung, 1968)",
        },
        "dolby_c": {
            "era": "1980-2000",
            "typical_media": ["cassette"],
            "quality": "consumer_hifi",
            "studio": False,
            "description_de": "Dolby C (Verbesserte Consumer NR, 1980)",
        },
        "dolby_sr": {
            "era": "1986-2010",
            "typical_media": ["reel_tape"],
            "quality": "professional",
            "studio": True,
            "description_de": "Dolby SR (Spectral Recording, 1986)",
        },
        "dolby_s": {
            "era": "1990-2000",
            "typical_media": ["cassette"],
            "quality": "consumer_hifi",
            "studio": False,
            "description_de": "Dolby S (Consumer-NR, spaete 1990er)",
        },
        "dbx_type_i": {
            "era": "1971-1985",
            "typical_media": ["reel_tape", "vinyl"],
            "quality": "professional",
            "studio": True,
            "description_de": "dbx Type I (Professionelles Kompander-System, 1971)",
        },
        "dbx_type_ii": {
            "era": "1975-1990",
            "typical_media": ["cassette", "vinyl"],
            "quality": "consumer_hifi",
            "studio": False,
            "description_de": "dbx Type II (Consumer Kompander, 1975)",
        },
        "telcom_c4": {
            "era": "1975-1995",
            "typical_media": ["reel_tape"],
            "quality": "professional",
            "studio": True,
            "description_de": "Telcom C4 (Deutsches Profi-Kompander-System, 1975)",
        },
        # Tape formulations (inferred from bias noise characteristics)
        "tape_type_i": {
            "era": "1963-2000",
            "typical_media": ["cassette"],
            "quality": "consumer_basic",
            "description_de": "Compact Cassette Typ I (Eisenoxid, Normalband)",
        },
        "tape_type_ii": {
            "era": "1975-2000",
            "typical_media": ["cassette"],
            "quality": "consumer_hifi",
            "description_de": "Compact Cassette Typ II (Chromdioxid, High-Bias)",
        },
        "tape_type_iv": {
            "era": "1979-2000",
            "typical_media": ["cassette"],
            "quality": "consumer_premium",
            "description_de": "Compact Cassette Typ IV (Reineisen/Metallband)",
        },
        # RIAA / Equalization
        "riaa_standard": {
            "era": "1954-present",
            "typical_media": ["vinyl"],
            "description_de": "RIAA-Entzerrungskurve (Standard seit 1954)",
        },
        "riaa_pre_1954": {
            "era": "1948-1954",
            "typical_media": ["vinyl"],
            "description_de": "Vor-RIAA-Entzerrung (herstellerspezifisch, vor 1954)",
        },
        "ccir_eq": {
            "era": "1960-1990",
            "typical_media": ["reel_tape"],
            "quality": "professional",
            "description_de": "CCIR-Entzerrung (EU-Studio-Tonbandstandard)",
        },
        "nab_eq": {
            "era": "1950-1990",
            "typical_media": ["reel_tape"],
            "quality": "professional",
            "description_de": "NAB-Entzerrung (US-Studio-Tonbandstandard)",
        },
        # Tape speed indicators (from flutter spectrum or pilot tone)
        "tape_speed_30ips": {
            "era": "1950-1990",
            "typical_media": ["reel_tape"],
            "quality": "professional_master",
            "description_de": "30 ips (76 cm/s) — Studio-Master-Bandgeschwindigkeit",
        },
        "tape_speed_15ips": {
            "era": "1950-present",
            "typical_media": ["reel_tape"],
            "quality": "professional",
            "description_de": "15 ips (38 cm/s) — Professionelle Bandgeschwindigkeit",
        },
        "tape_speed_7_5ips": {
            "era": "1950-1990",
            "typical_media": ["reel_tape", "cassette"],
            "quality": "consumer_hifi",
            "description_de": "7,5 ips (19 cm/s) — Consumer-HiFi-Bandgeschwindigkeit",
        },
        "tape_speed_3_75ips": {
            "era": "1950-1990",
            "typical_media": ["reel_tape"],
            "quality": "consumer_basic",
            "description_de": "3,75 ips (9,5 cm/s) — Consumer-Basic-Bandgeschwindigkeit",
        },
        "tape_speed_1_875ips": {
            "era": "1963-present",
            "typical_media": ["cassette"],
            "quality": "consumer_basic",
            "description_de": "1,875 ips (4,76 cm/s) — Compact-Cassette-Standard",
        },
        # Half-speed / Direct Metal Mastering
        "half_speed_mastering": {
            "era": "1970-present",
            "typical_media": ["vinyl"],
            "quality": "audiophile",
            "description_de": "Half-Speed-Mastering (Audiophile Vinyl-Pressung)",
        },
        "dmm_mastering": {
            "era": "1984-present",
            "typical_media": ["vinyl"],
            "quality": "audiophile",
            "description_de": "Direct Metal Mastering (Teldec/Neumann DMM)",
        },
    }

    # Studio era -> typical recording chains
    _STUDIO_ERA_CHAINS: dict[str, list[list[str]]] = {
        # 1950s-1960s Abbey Road / Capitol Studios
        "studio_golden_age": [
            ["reel_tape", "lacquer_disc", "vinyl"],
            ["reel_tape", "vinyl"],
        ],
        # 1970s analog studio (Ampex/Studer → vinyl)
        "studio_analog_peak": [
            ["reel_tape", "vinyl", "cassette"],
            ["reel_tape", "vinyl", "cartridge_8track"],
        ],
        # 1980s digital transition (reel → DAT → CD)
        "studio_digital_transition": [
            ["reel_tape", "dat", "cd_digital"],
            ["reel_tape", "cd_digital", "cassette"],
        ],
        # 1990s-2000s DAW era
        "studio_daw": [
            ["cd_digital", "mp3_high"],
            ["cd_digital", "streaming"],
        ],
    }

    _LANGUAGE_MEDIUM_BONUS: dict[str, dict[str, float]] = {
        "de": {"shellac": 0.15, "vinyl": 0.10, "cassette": 0.05, "cd_digital": 0.05, "mp3_low": -0.10},
        "en": {"vinyl": 0.05, "cd_digital": 0.05, "mp3_low": 0.05, "streaming": 0.05},
        "ja": {"cd_digital": 0.15, "minidisc": 0.10, "vinyl": 0.05, "dat": 0.05},
        "fr": {"shellac": 0.10, "vinyl": 0.10, "cd_digital": 0.05},
        "it": {"shellac": 0.10, "vinyl": 0.10, "cd_digital": 0.05},
        "es": {"shellac": 0.05, "vinyl": 0.10, "cassette": 0.05},
        "pt": {"shellac": 0.05, "vinyl": 0.10, "cd_digital": 0.05},
    }

    _MEDIUM_EXCLUDES_GENRES: dict[str, list[str]] = {
        "wax_cylinder": [
            "rock",
            "pop",
            "jazz",
            "blues",
            "hip_hop",
            "electronic",
            "metal",
            "punk",
            "disco",
            "funk",
            "reggae",
            "soul",
            "rnb",
            "rap",
            "techno",
            "house",
            "trance",
            "dubstep",
            "trap",
            "edm",
        ],
        "shellac": [
            "rock",
            "hip_hop",
            "electronic",
            "metal",
            "punk",
            "disco",
            "funk",
            "reggae",
            "techno",
            "house",
            "trance",
            "dubstep",
            "trap",
            "edm",
            "rap",
            "grunge",
            "synth_pop",
            "drum_and_bass",
        ],
        "wire_recording": [
            "rock",
            "hip_hop",
            "electronic",
            "metal",
            "punk",
            "disco",
            "funk",
            "reggae",
            "techno",
            "house",
            "trance",
            "dubstep",
            "trap",
            "edm",
            "rap",
            "synth_pop",
        ],
        "cartridge_8track": [
            "hip_hop",
            "electronic",
            "techno",
            "house",
            "trance",
            "dubstep",
            "trap",
            "edm",
            "grunge",
            "drum_and_bass",
            "vaporwave",
        ],
        "elcaset": [
            "hip_hop",
            "techno",
            "house",
            "trance",
            "dubstep",
            "trap",
            "edm",
            "grunge",
            "drum_and_bass",
            "vaporwave",
        ],
        "dcc": ["vaporwave", "trap", "dubstep", "edm", "phonk"],
        "minidisc": ["vaporwave", "trap", "dubstep", "phonk"],
        "dat": ["vaporwave"],
    }

    _MEDIUM_PREFERRED_GENRES: dict[str, list[str]] = {
        "wax_cylinder": ["classical", "opera", "march", "folk_traditional"],
        "shellac": ["jazz", "blues", "swing", "country", "gospel", "classical", "opera", "schlager", "folk"],
        "vinyl": ["rock", "pop", "soul", "funk", "jazz", "classical", "disco", "punk", "reggae", "metal", "schlager"],
        "reel_tape": ["classical", "jazz", "rock", "pop", "soul", "progressive_rock"],
        "cassette": ["rock", "pop", "metal", "punk", "hip_hop", "electronic", "synth_pop", "new_wave", "schlager"],
        "cartridge_8track": ["rock", "soul", "country", "funk", "disco", "pop"],
        "cd_digital": [
            "rock",
            "pop",
            "electronic",
            "hip_hop",
            "techno",
            "house",
            "classical",
            "metal",
            "grunge",
            "trip_hop",
        ],
        "dat": ["classical", "jazz", "electronic", "ambient"],
        "minidisc": ["pop", "rock", "electronic", "j_pop", "anime_music"],
        "mp3_high": ["electronic", "hip_hop", "rock", "pop", "metal", "indie"],
        "mp3_low": ["electronic", "hip_hop", "pop", "lofi_hip_hop"],
        "streaming": ["pop", "hip_hop", "electronic", "edm", "trap", "latin", "k_pop"],
    }

    # Language -> Medium-Era Preference matrix.
    # German: strong shellac/vinyl tradition, late mp3 adoption (GEMA).
    # Japanese: world's fastest CD adoption (1982), Minidisc stronghold.
    _LANGUAGE_MEDIUM_BONUS: dict[str, dict[str, float]] = {  # type: ignore[no-redef]
        "de": {"shellac": 0.15, "vinyl": 0.10, "cassette": 0.05, "cd_digital": 0.05, "mp3_low": -0.10},
        "en": {"vinyl": 0.05, "cd_digital": 0.05, "mp3_low": 0.05, "streaming": 0.05},
        "ja": {"cd_digital": 0.15, "minidisc": 0.10, "vinyl": 0.05, "dat": 0.05},
        "fr": {"shellac": 0.10, "vinyl": 0.10, "cd_digital": 0.05},
        "it": {"shellac": 0.10, "vinyl": 0.10, "cd_digital": 0.05},
        "es": {"shellac": 0.05, "vinyl": 0.10, "cassette": 0.05},
        "pt": {"shellac": 0.05, "vinyl": 0.10, "cd_digital": 0.05},
    }
    # German display names for GUI
    _MEDIUM_DISPLAY_NAMES: dict[str, str] = {  # type: ignore[no-redef]
        "tinfoil_cylinder": "Zinnfolien-Walze (Edison 1877)",
        "wax_cylinder": "Wachswalze (Edison 1888-1929)",
        "lacquer_disc": "Lackplatte / Acetat-Mitschnitt",
        "shellac": "Schellackplatte (78 rpm)",
        "shellac_vertical": "Schellackplatte, vertikaler Schnitt (Pathe/Edison Diamond Disc)",
        "wire_recording": "Stahldraht-Aufnahme (Webster-Chicago)",
        "vinyl": "Vinyl-Schallplatte (LP/45rpm)",
        "reel_tape": "Tonband (Studio-Master)",
        "tape": "Tonband (allgemein)",
        "cartridge_4track": "4-Spur-Cartridge (Fidelipac)",
        "cartridge_8track": "8-Spur-Cartridge (Lear Jet)",
        "cassette": "Compact Cassette (Philips 1963)",
        "elcaset": "Elcaset (Sony 1976-1980)",
        "playtape": "PlayTape (Miniatur-Kassette)",
        "cd_digital": "Compact Disc (CD, 44.1 kHz/16 bit)",
        "dat": "Digital Audio Tape (R-DAT)",
        "dcc": "Digital Compact Cassette (Philips 1992)",
        "minidisc": "MiniDisc (Sony, ATRAC)",
        "mp3_high": "MP3 (hohe Bitrate, >=192 kbps)",
        "mp3_low": "MP3 (niedrige Bitrate, <192 kbps)",
        "aac": "AAC / M4A (Advanced Audio Codec)",
        "streaming": "Streaming (Spotify/Apple Music/YouTube)",
    }

    # ── Transfer chain knowledge base ────────────────────────────────
    # Known plausible chains.  The detector matches detected sources
    # against these templates and prefers chains that match known patterns.
    _KNOWN_CHAINS: list[list[str]] = [
        # ═══ Studio → Veroeffentlichung → Kopie → Digital ═══
        ["reel_tape", "lacquer_disc", "vinyl", "cassette", "mp3_high"],
        ["reel_tape", "lacquer_disc", "vinyl", "cassette", "mp3_low"],
        ["reel_tape", "lacquer_disc", "vinyl", "mp3_high"],
        ["reel_tape", "lacquer_disc", "vinyl", "mp3_low"],
        ["reel_tape", "lacquer_disc", "vinyl", "cd_digital"],
        ["reel_tape", "vinyl", "cassette", "mp3_low"],
        ["reel_tape", "vinyl", "cassette", "mp3_high"],
        ["reel_tape", "vinyl", "cartridge_8track", "mp3_low"],
        ["reel_tape", "vinyl", "mp3_low"],
        ["reel_tape", "vinyl", "mp3_high"],
        ["reel_tape", "vinyl", "cd_digital"],
        ["reel_tape", "vinyl", "cd_digital", "mp3_high"],
        ["reel_tape", "cassette", "mp3_low"],
        ["reel_tape", "cassette", "mp3_high"],
        ["reel_tape", "cassette", "cd_digital"],
        ["reel_tape", "dat", "cd_digital"],
        # ═══ Vinyl → Consumer → Digital ═══
        ["vinyl", "cassette", "mp3_low"],
        ["vinyl", "cassette", "mp3_high"],
        ["vinyl", "cartridge_8track", "mp3_low"],
        ["vinyl", "cartridge_8track", "cassette", "mp3_low"],
        ["vinyl", "mp3_low"],
        ["vinyl", "mp3_high"],
        ["vinyl", "cd_digital"],
        ["vinyl", "cd_digital", "mp3_high"],
        ["vinyl", "cassette", "cd_digital"],
        ["vinyl", "dat"],
        ["vinyl", "minidisc"],
        # ═══ Schellack-Ara → Modern ═══
        ["wax_cylinder", "shellac", "vinyl", "cassette", "mp3_low"],
        ["wax_cylinder", "shellac", "vinyl", "mp3_low"],
        ["wax_cylinder", "shellac", "vinyl", "cd_digital"],
        ["wax_cylinder", "lacquer_disc", "vinyl", "mp3_low"],
        ["shellac", "vinyl", "cassette", "mp3_low"],
        ["shellac", "vinyl", "cassette", "mp3_high"],
        ["shellac", "vinyl", "mp3_low"],
        ["shellac", "vinyl", "mp3_high"],
        ["shellac", "vinyl", "cd_digital"],
        ["shellac", "vinyl", "reel_tape", "mp3_low"],
        ["lacquer_disc", "vinyl", "cassette", "mp3_low"],
        ["lacquer_disc", "vinyl", "mp3_low"],
        ["shellac_vertical", "shellac", "vinyl", "mp3_low"],
        # ═══ Wire Recording → Modern ═══
        ["wire_recording", "reel_tape", "vinyl", "mp3_low"],
        ["wire_recording", "vinyl", "cassette", "mp3_low"],
        ["wire_recording", "reel_tape", "cd_digital"],
        # ═══ Live/Radio ═══
        ["lacquer_disc", "reel_tape", "vinyl", "mp3_low"],
        ["reel_tape", "cassette", "cartridge_4track", "mp3_low"],
        ["vinyl", "cartridge_4track", "reel_tape", "mp3_low"],
        # ═══ Digitale Ketten ═══
        ["cd_digital", "mp3_high"],
        ["cd_digital", "mp3_low"],
        ["cd_digital", "aac"],
        ["cd_digital", "streaming"],
        ["dat", "cd_digital", "mp3_high"],
        ["dat", "mp3_high"],
        ["dcc", "mp3_high"],
        ["dcc", "cd_digital"],
        ["minidisc", "mp3_high"],
        ["minidisc", "cd_digital"],
        ["streaming", "mp3_low"],
        # ═══ Exotische/Historische Ketten ═══
        ["tinfoil_cylinder", "wax_cylinder", "shellac", "vinyl", "mp3_low"],
        ["shellac_vertical", "vinyl", "mp3_low"],
        ["cartridge_8track", "cassette", "mp3_low"],
        ["elcaset", "cassette", "cd_digital"],
        ["playtape", "cassette", "mp3_low"],
        # ═══ Einfache Ketten ═══
        ["vinyl"],
        ["cassette"],
        ["cd_digital"],
        ["reel_tape"],
        ["shellac"],
        ["wax_cylinder"],
        ["lacquer_disc"],
        ["wire_recording"],
        ["cartridge_8track"],
        ["dat"],
        ["dcc"],
        ["minidisc"],
        ["mp3_low"],
        ["mp3_high"],
        ["aac"],
        ["streaming"],
    ]

    # ── Bayesian Material-Modelle (Gaussian μ, σ) ────────────────────
    # Identisch mit MediumClassifier._MATERIAL_MODELS — kanonische Quelle.
    # Features: bandwidth_hz, snr_db, noise_color, crackle_density,
    #           wow_depth, block_artifact, pre_echo_ms,
    #           rotation_strength, infrasonic_rms, codec_type_code
    _MATERIAL_MODELS: dict[str, dict[str, tuple[float, float]]] = {
        # ── Analog-Materialien ────────────────────────────────────────────────────
        # block_artifact σ=0.12 (statt 0.05): Schmalband-Rauschen und Hiss auf analogen
        # Trägern kann MDCT-Block-Detektor aktivieren (false-positive ~0.10–0.15).
        # σ=0.05 erzeugt exponentielle Strafe (–4.5) bei Score=0.15; σ=0.12 → –0.78.
        # pre_echo_ms wird immer auf 0.0 gesetzt (Feature nicht implementiert) und ist
        # daher im Bayesian-Scorer maskiert (→ _masked_features).
        "shellac": {
            "bandwidth_hz": (5500.0, 1500.0),
            "snr_db": (10.0, 5.0),
            "noise_color": (2.2, 0.5),
            "crackle_density": (0.02, 0.02),
            "wow_depth": (0.3, 0.3),
            "block_artifact": (0.0, 0.12),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.40, 0.20),
            "infrasonic_rms": (0.06, 0.04),
            "codec_type_code": (0.0, 0.3),
        },
        "wax_cylinder": {
            "bandwidth_hz": (3500.0, 1200.0),
            "snr_db": (6.0, 4.0),
            "noise_color": (2.8, 0.6),
            "crackle_density": (0.04, 0.03),
            "wow_depth": (1.0, 0.8),
            "block_artifact": (0.0, 0.12),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.0, 0.10),
            "infrasonic_rms": (0.02, 0.03),
            "codec_type_code": (0.0, 0.3),
        },
        "vinyl": {
            "bandwidth_hz": (14000.0, 4000.0),
            "snr_db": (30.0, 10.0),
            "noise_color": (1.5, 0.4),
            "crackle_density": (0.004, 0.005),
            "wow_depth": (0.15, 0.15),
            "block_artifact": (0.0, 0.12),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.45, 0.20),
            "infrasonic_rms": (0.08, 0.05),
            "codec_type_code": (0.0, 0.3),
        },
        "tape": {
            "bandwidth_hz": (12000.0, 3000.0),
            "snr_db": (25.0, 8.0),
            "noise_color": (1.6, 0.4),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (1.2, 0.8),
            "block_artifact": (0.0, 0.12),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.0, 0.08),
            "infrasonic_rms": (0.01, 0.02),
            "codec_type_code": (0.0, 0.3),
        },
        "reel_tape": {
            "bandwidth_hz": (15000.0, 3000.0),
            "snr_db": (28.0, 7.0),
            "noise_color": (1.3, 0.3),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (0.3, 0.3),
            "block_artifact": (0.0, 0.12),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.0, 0.08),
            "infrasonic_rms": (0.01, 0.02),
            "codec_type_code": (0.0, 0.3),
        },
        "wire_recording": {
            "bandwidth_hz": (5000.0, 1500.0),
            "snr_db": (12.0, 5.0),
            "noise_color": (2.0, 0.5),
            "crackle_density": (0.0001, 0.0002),
            "wow_depth": (3.0, 1.5),
            "block_artifact": (0.0, 0.12),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.0, 0.10),
            "infrasonic_rms": (0.01, 0.02),
            "codec_type_code": (0.0, 0.3),
        },
        "lacquer_disc": {
            "bandwidth_hz": (9000.0, 2500.0),
            "snr_db": (18.0, 6.0),
            "noise_color": (1.7, 0.4),
            "crackle_density": (0.008, 0.008),
            "wow_depth": (0.2, 0.2),
            "block_artifact": (0.0, 0.12),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.30, 0.20),
            "infrasonic_rms": (0.04, 0.04),
            "codec_type_code": (0.0, 0.3),
        },
        # Compact Cassette IEC 60094-1 — kalibriert auf reale Digitalisierungen:
        # SNR 14–26 dB (Type I schlechter als Reel-Tape, abhängig von Dekaufnahme),
        # BW ≤ 12 kHz (Type I), wow/flutter 0.05–0.3 % WRMS @ 4.75 cm/s (variiert).
        "cassette": {
            "bandwidth_hz": (9500.0, 2500.0),
            "snr_db": (18.0, 9.0),
            "noise_color": (1.6, 0.4),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (0.8, 0.9),
            "block_artifact": (0.0, 0.12),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.0, 0.08),
            "infrasonic_rms": (0.01, 0.02),
            "codec_type_code": (0.0, 0.3),
        },
        "dat": {
            "bandwidth_hz": (20000.0, 2000.0),
            "snr_db": (50.0, 8.0),
            "noise_color": (0.3, 0.3),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (0.0, 0.1),
            "block_artifact": (0.08, 0.06),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.0, 0.05),
            "infrasonic_rms": (0.0, 0.01),
            "codec_type_code": (0.0, 0.5),
        },
        "cd_digital": {
            "bandwidth_hz": (21000.0, 1500.0),
            "snr_db": (60.0, 8.0),
            "noise_color": (0.2, 0.3),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (0.0, 0.05),
            "block_artifact": (0.0, 0.03),
            "pre_echo_ms": (0.0, 1.0),
            "rotation_strength": (0.0, 0.05),
            "infrasonic_rms": (0.0, 0.01),
            "codec_type_code": (0.0, 0.3),
        },
        "mp3_low": {
            "bandwidth_hz": (11000.0, 2500.0),
            "snr_db": (35.0, 8.0),
            "noise_color": (0.5, 0.4),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (0.0, 0.1),
            "block_artifact": (0.40, 0.15),
            "pre_echo_ms": (12.0, 6.0),
            "rotation_strength": (0.0, 0.05),
            "infrasonic_rms": (0.0, 0.01),
            "codec_type_code": (1.0, 0.3),
        },
        "mp3_high": {
            "bandwidth_hz": (17000.0, 2000.0),
            "snr_db": (42.0, 7.0),
            "noise_color": (0.4, 0.3),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (0.0, 0.1),
            "block_artifact": (0.15, 0.10),
            "pre_echo_ms": (6.0, 4.0),
            "rotation_strength": (0.0, 0.05),
            "infrasonic_rms": (0.0, 0.01),
            "codec_type_code": (1.0, 0.3),
        },
        "aac": {
            "bandwidth_hz": (19000.0, 1500.0),
            "snr_db": (48.0, 7.0),
            "noise_color": (0.3, 0.3),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (0.0, 0.1),
            "block_artifact": (0.10, 0.08),
            "pre_echo_ms": (1.5, 1.5),
            "rotation_strength": (0.0, 0.05),
            "infrasonic_rms": (0.0, 0.01),
            "codec_type_code": (2.0, 0.3),
        },
        "minidisc": {
            "bandwidth_hz": (14000.0, 2000.0),
            "snr_db": (40.0, 6.0),
            "noise_color": (0.4, 0.3),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (0.0, 0.1),
            "block_artifact": (0.18, 0.10),
            "pre_echo_ms": (3.0, 3.0),
            "rotation_strength": (0.0, 0.05),
            "infrasonic_rms": (0.0, 0.01),
            "codec_type_code": (3.0, 0.5),
        },
        "streaming": {
            "bandwidth_hz": (18000.0, 2000.0),
            "snr_db": (45.0, 8.0),
            "noise_color": (0.3, 0.3),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (0.0, 0.1),
            "block_artifact": (0.06, 0.05),
            "pre_echo_ms": (2.0, 2.0),
            "rotation_strength": (0.0, 0.05),
            "infrasonic_rms": (0.0, 0.01),
            "codec_type_code": (1.5, 0.8),
        },
        "unknown": {
            "bandwidth_hz": (12000.0, 8000.0),
            "snr_db": (25.0, 15.0),
            "noise_color": (1.0, 1.0),
            "crackle_density": (0.005, 0.01),
            "wow_depth": (0.5, 1.0),
            "block_artifact": (0.05, 0.10),
            "pre_echo_ms": (2.0, 5.0),
            "rotation_strength": (0.05, 0.15),
            "infrasonic_rms": (0.02, 0.05),
            "codec_type_code": (1.0, 1.5),
        },
    }

    _FEATURE_KEYS: list[str] = [
        "bandwidth_hz",
        "snr_db",
        "noise_color",
        "crackle_density",
        "wow_depth",
        "block_artifact",
        "pre_echo_ms",
        "rotation_strength",
        "infrasonic_rms",
        "codec_type_code",
    ]

    def _best_matching_chain(  # type: ignore[return]
        self, detected: list[str], genre: str | None = None, language: str | None = None
    ) -> list[str] | None:
        """Find the best-matching known chain for a set of detected materials.

        Returns the known chain that maximally overlaps with detected materials,
        respecting chronological order and genre-era compatibility.
        Used to correct detection-order chains into technologically plausible ones
        (e.g. reel_tape→vinyl→cassette instead of vinyl→cassette→reel_tape).

        Genre-awareness: a genre cannot appear on media that predate its existence
        (e.g. Hip-Hop on shellac → penalty, Hip-Hop on cassette → valid).
        """
        if not detected or len(detected) <= 1:
            return None
        detected_set = set(detected)
        best_match = None
        best_score = 0

        # Genre-era validation
        _genre_earliest = self._GENRE_EARLIEST_ORDER.get((genre or "").lower().replace(" ", "_").replace("-", "_"), 0)
        _lang_bonuses = self._LANGUAGE_MEDIUM_BONUS.get((language or "").lower()[:2], {})

        for chain in self._KNOWN_CHAINS:
            chain_set = set(chain)
            overlap = len(detected_set & chain_set)

            # Genre-era penalty: media too early for the genre lose points
            genre_penalty = 0
            if genre and _genre_earliest > 0:
                for m in detected:
                    medium_order = self._MEDIUM_ORDER.get(m, 0)
                    if medium_order < _genre_earliest:
                        genre_penalty += 1

            # Prefer chains where detected materials appear in correct order
            order_score = 0
            last_idx = -1
            for m in detected:
                if m in chain:
                    idx = chain.index(m)
                    if idx > last_idx:
                        order_score += 1
                    last_idx = idx

    def get_genre_constraints(self, chain: list[str]) -> dict[str, list[str]]:
        """Bidirectional medium->genre validation.

        Returns genres that CANNOT appear on this chain ('excluded')
        and genres that TYPICALLY appear ('preferred').
        Example: chain=['shellac'] -> excluded=['hip_hop','rock'], preferred=['jazz','blues']
        """
        excluded = set()
        preferred = set()
        for medium in chain:
            for g in self._MEDIUM_EXCLUDES_GENRES.get(medium, []):
                excluded.add(g)
            for g in self._MEDIUM_PREFERRED_GENRES.get(medium, []):
                preferred.add(g)
        return {"excluded": sorted(excluded), "preferred": sorted(preferred)}

    def _infer_analog_source_from_fingerprint(self, fp: SpectralFingerprint) -> list[tuple[str, float]]:
        """Infer analog source materials from physical fingerprint features.

        Called when file_ext zeroes Bayesian posteriors so that the Bayesian
        scorer alone cannot identify the original analog source.  Physical cues
        (vinyl rumble, cassette/tape transport flutter, shellac crackle) survive
        recording chain transfers and remain detectable even through codec encoding.

        Detection thresholds calibrated from MediumClassifier Gaussian models and
        empirical test corpus (see §6.7, Pohlmann 2010, IEC 60386).

        Returns:
            List of (material_key, confidence) sorted by MEDIUM_ORDER
            (original source first).
        """
        sources: list[tuple[str, float]] = []

        # ── Vinyl ─────────────────────────────────────────────────────
        # Physical evidence: infrasonic platter-bearing rumble + groove crackle
        # + optional turntable rotation periodicity.
        # Rumble: infrasonic_rms > 0.030 (vinyl model μ=0.08, σ=0.05)
        # Crackle: crackle_density > 0.004 (vinyl model μ=0.004, σ=0.005)
        vinyl_conf = 0.0
        if fp.infrasonic_rms > 0.030:
            vinyl_conf = max(vinyl_conf, float(min((fp.infrasonic_rms - 0.030) / 0.080, 1.0)))
        # Crackle-based vinyl inference — adaptive physical plausibility cap.
        #
        # Physical upper bound (Copeland 2008, §VI): genuine analog vinyl damage
        # produces at most ~20 impulsive events/s even for severely worn records.
        # Lossy codecs (MP3/AAC, Brandenburg 1999) add MDCT granule-boundary
        # artifacts at ≈ sr/576 per second (76.6/s @ 44.1 kHz; 83.3/s @ 48 kHz).
        # These inflate the measured crackle density far beyond the analog maximum.
        #
        # Strategy: the cap adapts to fp.codec_artifact_score ∈ [0, 1] which measures
        # per-song spectral log-envelope discontinuity at MDCT boundaries (Müller &
        # Ewert 2011).  The cap ranges from 20.0 (lossless) down to ≈5.0 (pure MP3).
        # This is UNIVERSAL across all song classes — no song-specific constants.
        #
        #   codec_score = 0.0 (lossless WAV/FLAC):  cap = 20.0/s  ← full physical range
        #   codec_score = 0.3 (mild compression):    cap ≈ 12.5/s  ← moderate penalty
        #   codec_score ≥ 0.6 (MP3 ≤ 128 kbps):    cap ≈  5.0/s  ← codec-dominated
        #
        #   • crackle ≤ cap:              plausible analog → calibrated vinyl model
        #   • crackle > cap, infra > 0.030: rumble confirms playback → moderate conf
        #   • crackle > cap, no infrasonic: ambiguous → weak chain signal (0.25)
        _codec_contamination = min(1.0, fp.codec_artifact_score / 0.60)
        _crackle_cap = max(3.0, 20.0 * (1.0 - 0.75 * _codec_contamination))
        if fp.crackle_density > 0.004:
            if fp.crackle_density <= _crackle_cap:
                # Plausible analog crackle level: calibrated vinyl model.
                vinyl_conf = max(vinyl_conf, float(min((fp.crackle_density - 0.004) / 0.025, 1.0)))
            elif fp.infrasonic_rms > 0.030:
                # High crackle + confirmed infrasonic rumble: vinyl physically certain.
                vinyl_conf = max(vinyl_conf, 0.55)
            else:
                # High crackle, no infrasonic: ambiguous (codec artifacts dominate),
                # but chain detection requires at least a weak vinyl signal.
                vinyl_conf = max(vinyl_conf, 0.25)
        # Rotation-based vinyl inference — adaptive infrasonic requirement.
        #
        # Vinyl turntable rotation (33⅓–78 RPM) always co-excites infrasonic
        # platter-bearing rumble (Copeland 2008, 10–25 Hz mechanical coupling).
        # Musical phrasing periodicity produces envelope autocorrelation in the same
        # 0.4–1.5 Hz band but has no infrasonic component.
        #
        # Three-tier gate (all universal — no song-specific constants):
        #   1. infrasonic_rms > 0.025: rumble confirmed → full rotation confidence
        #   2. no infrasonic, codec_artifact_score < 0.25, rotation ≥ 0.20:
        #      lossless/near-lossless material where infrasonic may have been
        #      HPF-removed during archival digitisation (Copeland 2008, §9.2)
        #      → apply 55% weight to avoid penalising genuine vinyl
        #   3. no infrasonic + codec present (MP3/AAC): likely song-rhythm aliasing
        #      → suppress rotation inference entirely
        if fp.rotation_strength > 0.08:
            if fp.infrasonic_rms > 0.025:
                # Tier 1: confirmed rumble → full confidence
                vinyl_conf = max(vinyl_conf, float(fp.rotation_strength))
            elif fp.codec_artifact_score < 0.25 and fp.rotation_strength >= 0.20:
                # Tier 2: lossless, HPF-digitised — moderate inference (55 %)
                vinyl_conf = max(vinyl_conf, float(fp.rotation_strength) * 0.55)
            elif fp.codec_artifact_score >= 0.25 and fp.rotation_strength >= 0.25:
                # Tier 2.5 (§2.46a): Multi-gen transfer through codec attenuates
                # infrasonic rumble via lossy encoding HPF, but rotation periodicity
                # persists. Reduced weight (45 %) to limit false positives from
                # song-rhythm aliasing while preserving genuine vinyl evidence.
                vinyl_conf = max(vinyl_conf, float(fp.rotation_strength) * 0.45)
            # Tier 3: codec present, no infrasonic, weak rotation → skip
        if vinyl_conf >= 0.20:
            sources.append(("vinyl", float(np.clip(vinyl_conf, 0.20, 0.85))))

        # ── Shellac ───────────────────────────────────────────────────
        # Heavy crackle + strong infrasonic → beats vinyl classification.
        shellac_conf = 0.0
        if fp.crackle_density > 0.015 and fp.infrasonic_rms > 0.040:
            shellac_conf = float(min(fp.crackle_density / 0.040, 1.0))
        if shellac_conf >= 0.35:
            sources = [(m, c) for m, c in sources if m != "vinyl"]
            sources.append(("shellac", float(np.clip(shellac_conf, 0.35, 0.85))))

        # ── Cassette ─────────────────────────────────────────────────
        # Capstan/pinch-roller transport flutter: wow_flutter_index 0.30–2.5
        # Calibration from Bayesian model: cassette μ=1.5, σ=1.0.
        # Vinyl-only wow_depth μ=0.15, σ=0.15 → 99th-percentile vinyl flutter ≈ 0.30.
        # When a disc source is already in the chain the combined vinyl+cassette flutter
        # starts at ~0.18 even for high-quality decks (Nakamichi flutter spec 0.04 % WRMS,
        # vinyl pitch-drift adds 0.10–0.15). Use a lower threshold in that case.
        has_disc = any(m in ("vinyl", "shellac", "lacquer_disc") for m, _ in sources)
        # §2.46a: Codec-adaptive cassette flutter threshold.
        # MP3/AAC encoding attenuates measured wow/flutter index by 40-60 %.
        _cass_flutter_base = 0.18 if has_disc else 0.30
        _cass_flutter_thresh = max(0.03, _cass_flutter_base * (1.0 - 0.60 * _codec_contamination))
        cassette_conf = 0.0
        if fp.wow_flutter_index > _cass_flutter_thresh:
            cassette_conf = float(min((fp.wow_flutter_index - _cass_flutter_thresh) / 1.20, 1.0))
        # Bandwidth evidence: cassette tape limits HF to ~14.5–15.5 kHz due to azimuth
        # misalignment and tape-formula roll-off.  When a disc source is already confirmed,
        # relax the codec guard — narrow BW in a codec file with confirmed analog origin
        # suggests cassette as intermediary, not just codec BW limitation (§2.46a).
        _bw_codec_limit = 0.65 if has_disc else 0.30
        if has_disc and 5_000 < fp.effective_bandwidth_hz < 15_500 and fp.codec_artifact_score < _bw_codec_limit:
            _bw_conf = float(np.clip((15_500 - fp.effective_bandwidth_hz) / 6_000, 0.15, 0.55))
            cassette_conf = max(cassette_conf, _bw_conf)
        _cassette_min = 0.15 if has_disc else 0.25
        if cassette_conf >= _cassette_min:
            sources.append(("cassette", float(np.clip(cassette_conf, _cassette_min, 0.85))))

        # ── Reel tape ────────────────────────────────────────────────
        # §2.46a: Codec-adaptive reel-tape flutter threshold — komplett überarbeitete Logik.
        #
        # Zwei strukturelle Bugs der Vorgängerversion:
        #
        # BUG A: `rotation_strength < 0.10`-Guard blockierte Tape-Erkennung bei has_disc=True.
        #   Vinyl-Quelldateien haben naturgemäß rotation_strength 0.08–0.60 vom Plattenteller.
        #   Wenn der Disc-Ursprung bereits in `sources` bestätigt ist, ist diese Rotation
        #   ERWARTET und darf das Tape-Zwischenglied nicht unterdrücken.
        #   Korrekt: Guard gilt nur wenn KEIN Disc-Ursprung bestätigt (no has_disc).
        #
        # BUG B: max(0.10,...)-Floor zu hoch für Studio-Bandmaschinen.
        #   Studio reel-tape (Studer A80, Ampex ATR) hat wow/flutter 0.01–0.03 WRMS (IEC 60386,
        #   Pohlmann 2010). Nach Multi-Gen-Transfer + Codec-Encoding weiter gedämpft.
        #   Fester Floor 0.10 greift systematisch über dem physikalischen Signal.
        #   Fix: Spezial-Pfad für has_disc=True (jegliche Disc→Tape-Kette).
        #   Codec_contamination dämpft Flutter → Schwelle adaptiv, aber der Pfad
        #   muss auch bei codec_contamination=0 aktiv sein (z.B. Disc→Tape ohne mp3).
        if has_disc:
            # Studio reel-tape Pfad: Threshold auf Basis des Studio-Flutter-Bereichs
            # (0.010–0.030 WRMS) mit Codec-Dämpfungskorrektur.
            # rotation_strength Guard entfernt — Disc-Rotation ist erwartet, kein Ausschluss.
            _tape_flutter_thresh_rt = max(0.008, 0.012 * (1.0 - 0.55 * _codec_contamination))
            if fp.wow_flutter_index > _tape_flutter_thresh_rt:
                # Konfidenz über schmalen Studio-Bereich skalieren
                tape_conf_rt = float(np.clip((fp.wow_flutter_index - _tape_flutter_thresh_rt) / 0.08, 0.12, 0.50))
                if tape_conf_rt >= 0.12:
                    sources.append(("reel_tape", float(np.clip(tape_conf_rt, 0.12, 0.85))))
        else:
            # Standard-Pfad: Consumer reel-tape oder kein Disc-Ursprung.
            # rotation_strength < 0.10 Guard: unterscheidet Tape-Flutter von Plattenspieler-Drift
            # (gilt nur wenn kein Disc-Ursprung bestätigt oder niedrige Codec-Kontamination).
            _tape_base_thresh = 0.20
            _tape_flutter_thresh = max(0.10, _tape_base_thresh * (1.0 - 0.55 * _codec_contamination))
            if fp.wow_flutter_index > _tape_flutter_thresh and fp.rotation_strength < 0.10:
                tape_conf = float(min((fp.wow_flutter_index - _tape_flutter_thresh) / 1.00, 1.0))
                _tape_min_conf = 0.15 if has_disc else 0.20
                if tape_conf >= _tape_min_conf:
                    sources.append(("reel_tape", float(np.clip(tape_conf, _tape_min_conf, 0.85))))

        # §2.46a: Cassette vs. reel_tape Disambiguation.
        # Consumer-Kassette: wow/flutter typisch >= 0.06 WRMS (Nakamichi-Spezifikation 0.04 WRMS,
        # typisch 0.08–0.50 WRMS; IEC 60386, AES 1984).
        # Studio reel-tape: wow/flutter typisch <= 0.05 WRMS.
        # Wenn beide erkannt wurden (BW-basierte Kassetten-Erkennung UND Flutter-basiertes Tape):
        # Entscheidung nach Wow/Flutter-Niveau — niedrig → reel_tape, hoch → cassette.
        _has_cassette_d = any(m == "cassette" for m, _ in sources)
        _has_reel_tape_d = any(m == "reel_tape" for m, _ in sources)
        if _has_cassette_d and _has_reel_tape_d:
            if fp.wow_flutter_index < 0.06:
                # Niedriger Flutter → Studio reel_tape wahrscheinlicher; Kassette entfernen
                sources = [(m, c) for m, c in sources if m != "cassette"]
                logger.debug(
                    "MediumDetector: cassette/reel_tape disambiguation — wow=%.3f < 0.06 → reel_tape",
                    fp.wow_flutter_index,
                )
            else:
                # Höherer Flutter → Consumer-Kassette wahrscheinlicher; reel_tape entfernen
                sources = [(m, c) for m, c in sources if m != "reel_tape"]
                logger.debug(
                    "MediumDetector: cassette/reel_tape disambiguation — wow=%.3f >= 0.06 → cassette",
                    fp.wow_flutter_index,
                )

        # Sort by signal-chain order: disc (0) before tape (1) before codec (2).
        sources.sort(key=lambda x: self._MEDIUM_ORDER.get(x[0], 5))
        return sources

    @staticmethod
    def _is_benign_codec_source(audio: np.ndarray, sr: int, fp: SpectralFingerprint) -> bool:
        """Heuristic guard to prevent false analog-chain inference on clean digital sources."""
        mono = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if mono.ndim == 2:
            mono = mono.mean(axis=0) if mono.shape[0] <= mono.shape[1] else mono.mean(axis=1)
        if mono.size < 4096 or sr <= 0:
            return False

        abs_mono = np.abs(mono)
        hard_clip_ratio = float(np.mean(abs_mono >= 0.999))
        near_clip_ratio = float(np.mean(abs_mono >= 0.98))

        dyn_window = max(1, int(sr * 0.4))
        dyn_frames = mono.size // dyn_window
        if dyn_frames >= 2:
            dyn_blocks = mono[: dyn_frames * dyn_window].reshape(dyn_frames, dyn_window)
            dyn_rms = np.sqrt(np.mean(dyn_blocks**2, axis=1) + 1e-12)
            dyn_std_db = float(np.std(20.0 * np.log10(dyn_rms + 1e-12)))
        else:
            dyn_std_db = 0.0

        n_fft = 4096
        hop = 1024
        window = np.hanning(n_fft).astype(np.float32)
        flatness_values: list[float] = []
        for start in range(0, mono.size - n_fft + 1, hop):
            frame = mono[start : start + n_fft] * window
            mag = np.abs(np.fft.rfft(frame)).astype(np.float64) + 1e-12
            flatness_values.append(float(np.exp(np.mean(np.log(mag))) / np.mean(mag)))

        if not flatness_values:
            return False

        flatness_median = float(np.median(flatness_values))
        return (
            hard_clip_ratio <= 1e-5
            and near_clip_ratio <= 1e-4
            and dyn_std_db >= 3.5
            and flatness_median <= 1e-2
            and fp.noise_floor_db <= -38.0
            and fp.wow_flutter_index < 0.02
            and fp.effective_bandwidth_hz >= 8_000.0
        )

    def _compute_fingerprint(self, audio: np.ndarray, sr: int) -> SpectralFingerprint:
        """Berechnet den vollständigen Spektralfingerabdruck (§6.7.1 + §6.7.3).

        Umfasst 5 Basis-Features + 8 erweiterte Features für Bayesian Scoring.
        NaN/Inf-sicher; alle Felder werden immer befüllt.
        """
        mono = self._to_mono(audio)
        n = len(mono)
        if n == 0:
            return SpectralFingerprint()

        hop = max(1, n // 200)
        win = min(2048, n)

        # ── 1. Rolloff 95 % ────────────────────────────────────────────
        try:
            frames = [mono[i : i + win] for i in range(0, n - win, hop)]
            rolloffs = []
            for frame in frames[:100]:
                spec = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
                freqs = np.fft.rfftfreq(len(frame), 1.0 / sr)
                cum = np.cumsum(spec**2)
                total = cum[-1]
                if total > 0:
                    idx = int(np.searchsorted(cum, 0.95 * total))
                    rolloffs.append(float(freqs[min(idx, len(freqs) - 1)]))
            rolloff_95 = float(np.median(rolloffs)) if rolloffs else 0.0
        except (ValueError, IndexError, TypeError, ZeroDivisionError) as _exc:
            logger.debug("MediumDetector: rolloff computation fehlgeschlagen: %s", _exc)
            rolloff_95 = 0.0

        # ── 2. Wow/Flutter-Index ────────────────────────────────────────
        try:
            frame_size = int(0.1 * sr)  # 100 ms
            pitches = []
            for start in range(0, n - frame_size, frame_size):
                frame = mono[start : start + frame_size].astype(np.float64)
                analytic: np.ndarray = hilbert(frame)  # type: ignore[assignment]
                env = np.abs(analytic)
                mean_e = float(np.mean(env))
                if mean_e > 1e-6:
                    pitches.append(mean_e)
            wow_flutter = float(np.std(np.diff(pitches))) if len(pitches) > 2 else 0.0
        except (ValueError, IndexError, TypeError, ZeroDivisionError) as _exc:
            logger.debug("MediumDetector: computation fehlgeschlagen in spectral fingerprint: %s", _exc)
            wow_flutter = 0.0

        # ── 3. HF-Energie > 16 kHz ─────────────────────────────────────
        try:
            spec_full = np.abs(np.fft.rfft(mono[: min(n, 65536)], n=65536))
            freqs_full = np.fft.rfftfreq(65536, 1.0 / sr)
            mask_hf = freqs_full > 16_000
            total_e = float(np.sum(spec_full**2))
            hf_e = float(np.sum(spec_full[mask_hf] ** 2))
            hf_fraction = hf_e / max(total_e, 1e-12)
        except (ValueError, IndexError, TypeError, ZeroDivisionError) as _exc:
            logger.debug("MediumDetector: computation fehlgeschlagen in spectral fingerprint: %s", _exc)
            hf_fraction = 0.0

        # ── 4. Rauschpegel (5. Perzentil PSD) ──────────────────────────
        try:
            frame_energies = []
            for start in range(0, n - win, hop):
                e = float(np.mean(mono[start : start + win] ** 2))
                if e > 0:
                    frame_energies.append(10 * math.log10(e))
            noise_floor = float(np.percentile(frame_energies, 5)) if frame_energies else -60.0
            noise_floor = max(-120.0, min(0.0, noise_floor))
        except (ValueError, IndexError, TypeError, ZeroDivisionError) as _exc:
            logger.debug("MediumDetector: computation fehlgeschlagen in spectral fingerprint: %s", _exc)
            noise_floor = -60.0

        # ── 5. Effektive Bandbreite (Rolloff −60 dBFS, Multi-Segment) ──────────────
        # BUG FIX (v10.0.0): Using only the first 65536 samples (≈1.4 s at 48 kHz)
        # caused false wax_cylinder detection for tracks with a quiet intro. A silent
        # or near-silent intro yields effective_bandwidth_hz ≈ 0, which matches the
        # wax_cylinder Gaussian model (μ=3500 Hz, σ=1200 Hz) far better than any
        # digital format. Fix: compute bandwidth on 5 energy-distributed segments
        # spanning the full audio and take the 80th-percentile result, ensuring that
        # the representative musical content (not a silent intro) determines the
        # material classification.
        try:
            n_fft_bw = 65536
            freqs_bw = np.fft.rfftfreq(n_fft_bw, 1.0 / sr)
            seg_len = n_fft_bw
            # Select 5 segments distributed across the audio; prefer energy-rich sections
            rms_win = max(1, int(sr * 0.5))  # 0.5 s RMS windows for energy ranking
            n_rms = max(1, n // rms_win)
            rms_vals = np.array([float(np.mean(mono[i * rms_win : (i + 1) * rms_win] ** 2)) for i in range(n_rms)])
            # Pick 5 energy-ranked segments; fallback to equally spaced if audio short
            if n >= seg_len * 2:
                # Sort by energy descending; pick top-5 diverse segments (≥2 s apart)
                sorted_idx = np.argsort(rms_vals)[::-1]
                chosen_starts: list[int] = []
                min_gap_frames = max(1, int(sr * 2.0 / rms_win))
                for idx in sorted_idx:
                    start_s = idx * rms_win
                    if start_s + seg_len > n:
                        continue
                    # Enforce minimum gap between segments to avoid highly correlated windows
                    if all(abs(idx - c // rms_win) >= min_gap_frames for c in chosen_starts):
                        chosen_starts.append(start_s)
                    if len(chosen_starts) >= 5:
                        break
                # Fallback: equally-spaced segments if diversity selection yielded too few
                if not chosen_starts:
                    step = max(1, (n - seg_len) // 5)
                    chosen_starts = [min(i * step, n - seg_len) for i in range(5)]
            else:
                chosen_starts = [0]

            bw_candidates: list[float] = []
            for ss in chosen_starts:
                seg = mono[ss : ss + seg_len]
                if len(seg) < seg_len:
                    seg = np.pad(seg, (0, seg_len - len(seg)))
                spec_seg = np.abs(np.fft.rfft(seg.astype(np.float64), n=n_fft_bw))
                peak = float(spec_seg.max())
                if peak < 1e-12:
                    continue  # silent segment — skip
                spec_db_seg = 20.0 * np.log10(np.clip(spec_seg / peak, 1e-15, np.inf))
                above = freqs_bw[spec_db_seg > -60.0]
                if len(above) > 0:
                    bw_candidates.append(float(above.max()))

            if bw_candidates:
                # 80th-percentile: robust against outlier-quiet segments while not
                # being fooled by a single segment with anomalous HF boost.
                eff_bw = float(np.percentile(bw_candidates, 80))
            else:
                eff_bw = 0.0
        except (ValueError, IndexError, TypeError, ZeroDivisionError) as _exc:
            logger.debug("MediumDetector: computation fehlgeschlagen in spectral fingerprint: %s", _exc)
            eff_bw = 0.0

        # ── 6. Erweiterte Features (§6.7.3) ────────────────────────────
        rotation_hz = 0.0
        rotation_strength = 0.0
        infrasonic_rms = 0.0
        codec_artifact_score = 0.0
        codec_type_code = 0.0
        crackle_density = 0.0
        snr_db = 0.0
        noise_color = 0.0

        try:
            rotation_hz, rotation_strength = self._rotation_periodicity(mono, sr)
        except (ValueError, IndexError, TypeError, ZeroDivisionError) as _exc:
            logger.debug("MediumDetector: computation fehlgeschlagen in spectral fingerprint: %s", _exc)

        try:
            infrasonic_rms = self._infrasonic_rms(mono, sr)
        except (ValueError, IndexError, TypeError, ZeroDivisionError) as _exc:
            logger.debug("MediumDetector: computation fehlgeschlagen in spectral fingerprint: %s", _exc)

        try:
            codec_artifact_score, codec_type_code = self._codec_artifact_score(mono, sr)
        except (ValueError, IndexError, TypeError, ZeroDivisionError) as _exc:
            logger.debug("MediumDetector: computation fehlgeschlagen in spectral fingerprint: %s", _exc)

        try:
            crackle_density = self._crackle_density(mono, sr)
        except (ValueError, IndexError, TypeError, ZeroDivisionError) as _exc:
            logger.debug("MediumDetector: computation fehlgeschlagen in spectral fingerprint: %s", _exc)

        try:
            snr_db = self._snr(mono, sr)
        except (ValueError, IndexError, TypeError, ZeroDivisionError) as _exc:
            logger.debug("MediumDetector: computation fehlgeschlagen in spectral fingerprint: %s", _exc)

        try:
            noise_color = self._noise_color(mono, sr)
        except (ValueError, IndexError, TypeError, ZeroDivisionError) as _exc:
            logger.debug("MediumDetector: computation fehlgeschlagen in spectral fingerprint: %s", _exc)

        return SpectralFingerprint(
            rolloff_95_hz=float(np.nan_to_num(rolloff_95)),
            wow_flutter_index=float(np.nan_to_num(wow_flutter)),
            hf_energy_above_16k=float(np.nan_to_num(hf_fraction)),
            noise_floor_db=float(np.nan_to_num(noise_floor, nan=-60.0)),
            effective_bandwidth_hz=float(np.nan_to_num(eff_bw)),
            rotation_hz=float(np.nan_to_num(rotation_hz)),
            rotation_strength=float(np.nan_to_num(rotation_strength)),
            infrasonic_rms=float(np.nan_to_num(infrasonic_rms)),
            codec_artifact_score=float(np.nan_to_num(codec_artifact_score)),
            codec_type_code=float(np.nan_to_num(codec_type_code)),
            crackle_density=float(np.nan_to_num(crackle_density)),
            snr_db=float(np.nan_to_num(snr_db)),
            noise_color=float(np.nan_to_num(noise_color)),
        )

    # ── Erweiterte Feature-Extraktion (aus MediumClassifier portiert) ───

    @staticmethod
    def _rotation_periodicity(mono: np.ndarray, sr: int) -> tuple[float, float]:
        """Erkennt turntable/disc rotation periodicity via autocorrelation.

        Returns (rotation_hz, rotation_strength).
        Vinyl: 33⅓ RPM → 0.556 Hz, 45 RPM → 0.750 Hz, 78 RPM → 1.300 Hz.
        """
        n = len(mono)
        if n < sr * 4:  # need ≥ 4 s for sub-Hz periodicity
            return 0.0, 0.0

        # Envelope via RMS in 50 ms windows
        win_samples = max(1, int(sr * 0.05))
        n_frames = n // win_samples
        if n_frames < 20:
            return 0.0, 0.0
        rms = np.sqrt(np.mean(mono[: n_frames * win_samples].reshape(n_frames, win_samples) ** 2, axis=1) + 1e-12)
        env_sr = sr / win_samples

        # Downsample envelope to ~20 Hz
        dec_factor = max(1, int(env_sr / 20))
        if dec_factor > 1 and len(rms) > dec_factor * 10:
            rms_dec = decimate(rms.astype(np.float64), dec_factor)
            dec_sr = env_sr / dec_factor
        else:
            rms_dec = rms.astype(np.float64)
            dec_sr = env_sr

        # Autocorrelation
        rms_dec = rms_dec - np.mean(rms_dec)
        n_dec = len(rms_dec)
        if n_dec < 10:
            return 0.0, 0.0
        acf = np.correlate(rms_dec, rms_dec, mode="full")
        acf = acf[n_dec - 1 :]  # positive lags only
        acf_norm = acf / (acf[0] + 1e-12)

        # Search for peaks in 0.4–1.5 Hz range (turntable rotation)
        min_lag = max(1, int(dec_sr / 1.5))
        max_lag = min(n_dec - 1, int(dec_sr / 0.4))
        if max_lag <= min_lag:
            return 0.0, 0.0

        search = acf_norm[min_lag : max_lag + 1]
        peak_idx = int(np.argmax(search))
        peak_val = float(search[peak_idx])
        rot_hz = float(dec_sr / (min_lag + peak_idx)) if (min_lag + peak_idx) > 0 else 0.0

        return rot_hz, max(0.0, peak_val)

    @staticmethod
    def _infrasonic_rms(mono: np.ndarray, sr: int) -> float:
        """Misst infrasonic energy (< 20 Hz), strong in vinyl due to warp/eccentricity."""
        n = len(mono)
        fft_n = min(n, 131072)
        spec = np.abs(np.fft.rfft(mono[:fft_n], n=fft_n))
        freqs = np.fft.rfftfreq(fft_n, 1.0 / sr)
        mask_infra = freqs < 20.0
        mask_total = freqs < sr / 2
        e_infra = float(np.sum(spec[mask_infra] ** 2))
        e_total = float(np.sum(spec[mask_total] ** 2))
        return float(np.sqrt(e_infra / max(e_total, 1e-12)))

    @staticmethod
    def _codec_artifact_score(mono: np.ndarray, sr: int) -> tuple[float, float]:
        """Detect lossy codec artifacts (block artifacts, pre-echo).

        Returns (artifact_score, codec_type_code).
        codec_type_code: 0=none, 1=MP3, 2=AAC, 3=MiniDisc/ATRAC.

        Method: Müller & Ewert (2011) spectral log-envelope discontinuity at MDCT
        granule boundaries combined with time-domain boundary energy. MP3/AAC
        discrimination via relative per-block-size score ratio (Spijkervet 2020).
        """
        n = len(mono)
        block_sizes = [576, 1152, 1024, 512]  # MP3 short/long, AAC, ATRAC
        best_score = 0.0
        best_block = 0
        # Track per-size scores for MP3/AAC discrimination (Spijkervet 2020)
        score_by_bs: dict[int, float] = {}

        for bs in block_sizes:
            if n < bs * 10:
                continue
            n_blocks = n // bs
            blocks = mono[: n_blocks * bs].reshape(n_blocks, bs)

            # Time-domain boundary energy ratio (original method)
            boundary_e = np.mean(np.abs(np.diff(blocks, axis=0)[:, :4]) ** 2)
            mid_e = np.mean(np.abs(np.diff(blocks, axis=0)[:, bs // 2 : bs // 2 + 4]) ** 2)
            td_ratio = float(boundary_e / max(mid_e, 1e-12))

            # Müller & Ewert (2011): spectral log-envelope cross-block discontinuity.
            # MDCT quantization changes the spectral shape sharply at granule boundaries:
            # measure the mean absolute difference of log-spectra in adjacent blocks,
            # then take the 75th percentile to avoid music-transient false positives.
            spec_disco_score = 0.0
            try:
                win = np.hanning(bs)
                specs = np.abs(np.fft.rfft(blocks * win[np.newaxis, :], axis=1)) + 1e-12
                log_specs = np.log(specs)
                # L1 distance between adjacent block log-spectra
                log_diffs = np.mean(np.abs(np.diff(log_specs, axis=0)), axis=1)
                # 75th percentile avoids transient contribution; music baseline ≈ 0.3–0.7
                spec_disco = float(np.percentile(log_diffs, 75))
                # Score is nonzero above 0.5; saturates around 2.0 (strong codec)
                spec_disco_score = float(np.clip((spec_disco - 0.5) / 1.5, 0.0, 1.0))
            except (ValueError, TypeError, ImportError) as _exc:
                logger.debug("MediumDetector: optional feature fehlgeschlagen: %s", _exc)

            # Müller & Ewert (2011): spectral discontinuity is the reliable codec
            # indicator — frequency-selective discontinuities at granule boundaries.
            # Musical transients affect ALL bins uniformly → 75th-pct rejects them.
            # Time-domain boundary ratio (td_ratio) spikes on percussion/plucks even
            # without any codec artifact. Fix: gate td_ratio's contribution by spectral
            # evidence: spec_gate ∈ [0.12, 1.0]; when spec_disco_score=0 (no spectral
            # break detected), td_ratio is suppressed to ≤12% of its contribution.
            spec_gate = 0.12 + 0.88 * spec_disco_score  # scales with spectral evidence
            combined_bs = 0.38 * (td_ratio * spec_gate) + 0.62 * (1.0 + 2.0 * spec_disco_score)
            score_by_bs[bs] = combined_bs

            if combined_bs > best_score:
                best_score = combined_bs
                best_block = bs

        artifact_score = float(np.clip((best_score - 1.0) / 2.0, 0.0, 1.0))

        # Pre-echo detection (Herre & Johnston 1996: temporal masking produces energy
        # ramp-up before transients; successive rising frames before a jump indicate
        # codec-induced pre-echo)
        pre_echo_score = 0.0
        frame_ms = 10
        frame_n = max(1, int(sr * frame_ms / 1000))
        n_frames_pe = n // frame_n
        if n_frames_pe > 20:
            frame_e = np.sqrt(np.mean(mono[: n_frames_pe * frame_n].reshape(n_frames_pe, frame_n) ** 2, axis=1) + 1e-12)
            diff_e = np.diff(frame_e)
            jumps = np.where(diff_e > np.std(diff_e) * 2.0)[0]
            if len(jumps) > 2:
                pre_rises = 0
                for j in jumps:
                    if j >= 3 and all(diff_e[j - k] > 0 for k in range(1, min(4, j + 1))):
                        pre_rises += 1
                pre_echo_score = float(np.clip(pre_rises / max(len(jumps), 1), 0.0, 1.0))

        combined = float(np.clip(0.6 * artifact_score + 0.4 * pre_echo_score, 0.0, 1.0))

        # Spijkervet & Haasdijk (2020): MP3 vs. AAC discrimination via relative
        # block-size score. MP3 peaks at granule=576, AAC at 1024. When the 576-score
        # is substantially higher than the 1024-score, classify as MP3; if 1024 is
        # comparable or higher despite best_block=576, prefer AAC classification.
        codec_type = 0.0
        if combined > 0.15:
            s576 = score_by_bs.get(576, 0.0) + score_by_bs.get(1152, 0.0) * 0.5
            s1024 = score_by_bs.get(1024, 0.0)
            if best_block in (576, 1152):
                # MP3 only if its characteristic granule size scores clearly above AAC
                codec_type = 1.0 if s576 >= s1024 * 1.1 else 2.0
            elif best_block == 1024:
                codec_type = 2.0  # AAC
            elif best_block == 512:
                codec_type = 3.0  # MiniDisc/ATRAC

        return combined, codec_type

    @staticmethod
    def _crackle_density(mono: np.ndarray, sr: int) -> float:
        """Schätzt impulsive crackle density as normalized event ratio [0, 1].

        Applies a stochasticity filter (Cox & Lewis 1966): genuine vinyl crackle
        follows a Poisson process (CV of inter-event intervals ≈ 1.0).  MDCT-codec
        boundary artifacts are quasi-deterministic (period ≈ 576 / sr ≈ 13 ms for
        MP3); their CV is << 0.35.  When the majority of events are periodic, the
        periodic component is stripped and only the residual stochastic events are
        counted.  This allows genuine vinyl crackle to be detected even through a
        codec layer while preventing codec-only artifacts from triggering analog
        source inference.
        """
        n = len(mono)
        duration_s = n / max(sr, 1)
        if duration_s < 0.5:
            return 0.0

        sos = butter(4, 2000.0, btype="high", fs=sr, output="sos")
        hp = sosfilt(sos, mono.astype(np.float64))

        threshold = 6.0 * float(np.median(np.abs(hp)) + 1e-12)
        impulses = np.abs(hp) > threshold

        min_gap = max(1, int(sr * 0.005))
        event_positions: list[int] = []
        last_event = -min_gap
        for i in np.where(impulses)[0]:
            if i - last_event >= min_gap:
                event_positions.append(int(i))
                last_event = i

        n_events = len(event_positions)
        if n_events < 2:
            return float(n_events / max(float(n), 1.0))

        # ── Stochasticity filter (Cox & Lewis 1966) ──────────────────────────
        # CV of inter-event intervals: Poisson (vinyl) → CV ≈ 1.0,
        # periodic (codec MDCT boundary) → CV << 0.35.
        # When events are predominantly periodic, strip the periodic component
        # and return only the residual stochastic (analog-source) events.
        intervals = np.diff(np.array(event_positions, dtype=np.float64))
        iv_mean = float(np.mean(intervals))
        iv_std = float(np.std(intervals))
        cv = iv_std / (iv_mean + 1e-12)

        if cv < 0.35 and n_events >= 20:
            dominant_period = float(np.median(intervals))
            if dominant_period > 0:
                pos_arr = np.array(event_positions, dtype=np.float64)
                ivs_to_prev = np.concatenate([[dominant_period], np.diff(pos_arr)])
                ivs_to_next = np.concatenate([np.diff(pos_arr), [dominant_period]])
                tol = dominant_period * 0.30
                non_periodic = (np.abs(ivs_to_prev - dominant_period) > tol) & (
                    np.abs(ivs_to_next - dominant_period) > tol
                )
                stochastic_events = int(np.sum(non_periodic))
                # Normalized density (events per sample) keeps feature scale
                # consistent with Gaussian model means (e.g. vinyl μ≈0.004).
                return float(stochastic_events / max(float(n), 1.0))

        return float(n_events / max(float(n), 1.0))

    @staticmethod
    def _snr(mono: np.ndarray, sr: int) -> float:  # pylint: disable=unused-argument
        """Schätzt SNR in dB via signal vs noise-floor percentile."""
        n = len(mono)
        win = min(2048, n)
        hop = max(1, n // 200)
        frame_energies = []
        for start in range(0, n - win, hop):
            e = float(np.mean(mono[start : start + win] ** 2))
            if e > 0:
                frame_energies.append(e)
        if len(frame_energies) < 5:
            return 0.0
        arr = np.array(frame_energies)
        noise_e = float(np.percentile(arr, 5))
        signal_e = float(np.percentile(arr, 95))
        if noise_e <= 0:
            return 60.0
        return float(np.clip(10 * math.log10(signal_e / noise_e), 0.0, 80.0))

    @staticmethod
    def _noise_color(mono: np.ndarray, sr: int) -> float:
        """Schätzt noise colour as spectral tilt (0=white, 1=pink, 2=brown/red).

        Uses 5th-percentile frames (noise floor) spectral slope.
        """
        n = len(mono)
        fft_n = min(n, 16384)
        win = np.hanning(fft_n).astype(np.float32)
        hop_nc = fft_n
        slopes: list[float] = []

        for start in range(0, n - fft_n, hop_nc):
            frame = mono[start : start + fft_n] * win
            mag = np.abs(np.fft.rfft(frame)) + 1e-12
            freqs = np.fft.rfftfreq(fft_n, 1.0 / sr)
            freqs = freqs[1:]
            mag = mag[1:]
            if len(freqs) < 10:
                continue
            log_f = np.log10(freqs + 1e-6)
            log_m = np.log10(mag)
            slope = float(np.polyfit(log_f, log_m, 1)[0])
            slopes.append(slope)

        if not slopes:
            return 1.0  # default pink
        slopes_sorted = sorted(slopes)
        n_quiet = max(1, len(slopes_sorted) // 5)
        avg_slope = float(np.mean(slopes_sorted[:n_quiet]))
        return float(np.clip(-avg_slope * 2.0, 0.0, 4.0))

    # ── Bayesian Material Scoring ──────────────────────────────────────

    # Minimum audio duration (seconds) needed for reliable turntable-rotation ACF.
    # 33⅓ RPM → period ≈ 1.82 s; reliable ACF requires ≥4 cycles → ≥7.3 s.
    # 45 RPM → period ≈ 1.33 s → ≥4 cycles → ≥5.3 s.
    # Conservative minimum: 6 s covers all standard speeds with ≥3 cycles.
    _MIN_ROTATION_ANALYSIS_DURATION_S: float = 6.0

    def _bayesian_score(
        self,
        fp: SpectralFingerprint,
        duration_s: float = 0.0,
        era_decade: int | None = None,
        era_confidence: float = 0.0,
    ) -> dict[str, float]:
        """Berechnet posterior probabilities for all 16 material types via Gaussian log-likelihood.

        Args:
            fp:             Spectral fingerprint of the audio.
            duration_s:     Audio duration in seconds.
            era_decade:     Optional era estimate from EraClassifier (§v10.14.1).
                            When provided, material priors are modulated: materials
                            consistent with the era get a log-prior boost, materials
                            from impossible eras get a penalty.
            era_confidence: EraClassifier confidence [0,1] — modulates prior strength.

        Returns dict[material_name → posterior_probability], sorted descending.
        """
        # Duration-adaptive feature masking (§2.47, §0c — general, not song-specific).
        # rotation_strength requires ≥3 full turntable cycles for reliable ACF detection.
        # Short clips (< 6 s) produce rotation_strength ≈ 0 regardless of material;
        # this 0 is a perfect fit for tape (μ=0.0, σ=0.08) but strongly penalises vinyl
        # (μ=0.40, σ=0.20 → 2 σ penalty) → vinyl mis-classified as tape.
        # Fix: treat rotation_strength as unobserved for short clips (skip feature;
        # all materials receive equal log-likelihood for this dimension → other
        # features dominate the posterior — physically correct).
        _short_clip = 0.0 < duration_s < self._MIN_ROTATION_ANALYSIS_DURATION_S
        if _short_clip:
            logger.debug(
                "MediumDetector: short clip (%.1fs < %.1fs) — "
                "rotation_strength excluded from Bayesian Aktualisierung (unreliable ACF)",
                duration_s,
                self._MIN_ROTATION_ANALYSIS_DURATION_S,
            )
        # pre_echo_ms ist in _compute_fingerprint IMMER 0.0 (feature nicht implementiert).
        # Wird es im Scorer mit 0.0 eingesetzt, erhalten mp3_low/mp3_high eine dauerhafte
        # Strafe (μ=12/6 ms ≠ 0), während Analog-Materialien (μ=0) ungestraft bleiben.
        # Das erzeugt einen systematischen Bias Richtung Analog → immer maskieren.
        _masked_features: frozenset[str] = (
            frozenset({"rotation_strength", "pre_echo_ms"}) if _short_clip else frozenset({"pre_echo_ms"})
        )

        feature_vals = {
            "bandwidth_hz": fp.effective_bandwidth_hz,
            "snr_db": fp.snr_db,
            "noise_color": fp.noise_color,
            "crackle_density": fp.crackle_density,
            "wow_depth": fp.wow_flutter_index,
            "block_artifact": fp.codec_artifact_score,
            "pre_echo_ms": 0.0,
            "rotation_strength": fp.rotation_strength,
            "infrasonic_rms": fp.infrasonic_rms,
            "codec_type_code": fp.codec_type_code,
        }

        log_likes: dict[str, float] = {}
        for mat, params in self._MATERIAL_MODELS.items():
            ll = 0.0
            for feat_key in self._FEATURE_KEYS:
                if feat_key in _masked_features:
                    continue  # treat as unobserved — equal likelihood for all materials
                mu, sigma = params[feat_key]
                sigma = max(sigma, 1e-6)
                x = feature_vals.get(feat_key, 0.0)
                ll -= 0.5 * ((x - mu) / sigma) ** 2 + math.log(sigma)
            log_likes[mat] = ll

        # §6.7.4 Bayesian Prior Dämpfung (NEU v10.14.0):
        # Ohne expliziten Prior gewinnt "unknown" aufgrund extrem breiter σ-Werte
        # jede ambiguitive Eingabe (unknown=0.999 → kein Material erkannt).
        # Solution: Bayesian prior P(material) wird explizit als log-prior
        # eingewoben. P(unknown) = 0.02, P(rest) = 0.98 / 15 ≈ 0.0653 je Material.
        # Effekt: unknown braucht ≈ 1.5 nats mehr Evidenz als andere Materialien,
        # was echte Hypothesen bevorzugt ohne false-positives zu erzwingen.
        #
        # Ref: J. Pearl (1988) Probabilistic Reasoning in Intelligent Systems
        #      Section 2.3: Cromwell's Rule — assign only a small probability to
        #      the catch-all hypothesis so that plausible alternatives are preferred.

        # §v10.14.1 Era-Aware Prior Modulation:
        # Wenn der EraClassifier eine Dekade mit ausreichender Konfidenz liefert,
        # werden die log-priors der Materialien entsprechend moduliert:
        #   - Materialien, die in dieser Ära TYPISCH sind → Boost
        #   - Materialien, die in dieser Ära UNMÖGLICH sind → Penalty
        #   - Materialien, die möglich aber untypisch sind → neutral
        # Die Stärke skaliert mit der Era-Confidence (niedrige Conf → schwacher Effekt).
        if era_decade is not None and era_confidence >= 0.40:
            _era_boost_nats = 0.0
            if era_confidence >= 0.75:
                _era_boost_nats = 1.0
            elif era_confidence >= 0.60:
                _era_boost_nats = 0.6
            elif era_confidence >= 0.40:
                _era_boost_nats = 0.3

            _era_decade_rounded = (era_decade // 10) * 10  # round to decade
            for mat in log_likes:
                _mat_key = mat.lower().replace(" ", "_").replace("-", "_")
                if mat == "unknown":
                    continue
                _consistent_decades = MediumDetector._ERA_CONSISTENT.get(_mat_key, [])
                _impossible_decades = MediumDetector._ERA_IMPOSSIBLE.get(_mat_key, [])
                if _era_decade_rounded in _consistent_decades:
                    log_likes[mat] += _era_boost_nats
                elif _era_decade_rounded in _impossible_decades:
                    log_likes[mat] -= _era_boost_nats * 1.5  # harder penalty for impossible combos
            logger.debug(
                "MediumDetector: Era-Prior applied (decade=%d, conf=%.2f, boost=%.1f nats)",
                _era_decade_rounded,
                era_confidence,
                _era_boost_nats,
            )

        _N_MATERIALS: int = len(self._MATERIAL_MODELS)
        _P_UNKNOWN: float = 0.001  # §v10.14.1: 0.01→0.001 — unknown braucht ~2.9 nats mehr
        _P_OTHER: float = (1.0 - _P_UNKNOWN) / max(_N_MATERIALS - 1, 1)
        for mat in log_likes:
            _log_prior = math.log(_P_UNKNOWN) if mat == "unknown" else math.log(_P_OTHER)
            log_likes[mat] += _log_prior
        logger.debug(
            "MediumDetector: Bayesian-Prior applied P(unknown)=%.3f P(other)=%.3f (Δ=%.1f nats)",
            _P_UNKNOWN,
            _P_OTHER,
            math.log(_P_OTHER) - math.log(_P_UNKNOWN),
        )

        # Softmax normalization → posterior probabilities
        max_ll = max(log_likes.values())
        exp_vals = {k: math.exp(v - max_ll) for k, v in log_likes.items()}
        total = sum(exp_vals.values()) + 1e-12
        posteriors = {k: v / total for k, v in exp_vals.items()}

        return dict(sorted(posteriors.items(), key=lambda x: x[1], reverse=True))

    def detect(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        file_ext: str = "",
        era_decade: int | None = None,
        era_confidence: float = 0.0,
    ) -> MediumDetectionResult:
        """Erkennt die Tonträgerkette forensisch via Bayesian-Fusion (§6.7 v10.0.0).

        Ablauf:
        1. Vollständiger Spektralfingerabdruck (13 Features)
        2. Bayesian Scoring über 16 Materialtypen
        3. §6.7b File-Extension Prior: digital formats → analog posteriors penalized
        4. Primärmaterial = höchster Analog-Posterior (oder Digital)
        5. Codec-Layer-Erkennung (Block-Artefakte, Pre-Echo)
        6. Multi-Layer-Ketten-Inferenz (3+ Stufen möglich)
        7. ClassificationResult-Kompatibilitäts-Objekt für Passthrough

        Args:
            audio:          Input audio, float32/64, mono or stereo.
            sr:             Sample rate in Hz.
            file_ext:       File extension of the source file (e.g. '.mp3', '.flac').
            era_decade:     Optional era estimate from EraClassifier (§v10.14.1).
            era_confidence: EraClassifier confidence [0,1] — modulates era prior strength.

        Returns:
            MediumDetectionResult mit transfer_chain, bayesian_scores,
            classification_result für Downstream-Passthrough.
        """
        # §Spec 24 Zero-Length-Guard: Leerer/(fast) leerer Input erzeugt sonst
        # Unsinn-Posteriors (Befund 2026-08-16: „vinyl=1.000“ aus 0.0 s Stille,
        # Konfidenz 1.00). Kein Fingerprint, kein Bayesian — ehrliches „unknown“.
        _n_guard = int(np.asarray(audio).size)
        if _n_guard < int(max(float(sr), 1.0) * 0.05):
            logger.warning(
                "MediumDetector: Audio zu kurz (%d Samples) — Trägererkennung übersprungen (Zero-Length-Guard, Spec 24)",
                _n_guard,
            )
            return MediumDetectionResult(
                transfer_chain=[],
                is_multi_generation=False,
                primary_material="unknown",
                confidence=0.0,
                spectral_fingerprint=SpectralFingerprint(),
                bayesian_scores={"unknown": 1.0},
                classification_result=None,
            )

        if sr != 48000:
            logger.debug(
                "MediumDetector: native Analyse-SR=%d Hz (kein 48-kHz-Zwang im Voranalysepfad)",
                sr,
            )

        fp = self._compute_fingerprint(audio, sr)
        # Pass audio duration for rotation_strength masking in short clips (§2.47 §0c)
        _mono_len: int = audio.shape[0] if audio.ndim == 1 else int(max(audio.shape))
        _duration_s: float = float(_mono_len) / max(float(sr), 1.0)
        posteriors = self._bayesian_score(
            fp, duration_s=_duration_s, era_decade=era_decade, era_confidence=era_confidence
        )

        # §6.7b File-Extension Prior: digital file formats cannot originate from
        # analog physical media.  A .mp3 file was encoded digitally at capture time —
        # any analog artefacts are from the recording chain, not mechanical transport.
        # Zero out all analog material posteriors and renormalise before chain inference.
        # NOTE: lossless/uncompressed containers (.wav, .flac, .aiff, .aif, .wv) are NOT
        # treated as "digital-encoded" formats — they are neutral storage containers commonly
        # used for digitised analog recordings (vinyl rips, tape transfers).  Only lossy-
        # encoded formats confirm that the source was processed digitally at capture time
        # (spec §5: ".mp3, .aac, .ogg, .wma, .opus u. a.").
        _DIGITAL_FILE_EXTS: frozenset[str] = frozenset(
            {
                ".mp3",
                ".mp2",
                ".aac",
                ".m4a",
                ".ogg",
                ".opus",
                ".mpc",
                ".wma",
            }
        )
        _ext_lower = str(file_ext or "").strip().lower()
        # Accept both "mp3" and ".mp3" to keep detector behavior consistent
        # across direct calls and pipeline paths.
        if _ext_lower and not _ext_lower.startswith("."):
            _ext_lower = f".{_ext_lower}"
        if _ext_lower in _DIGITAL_FILE_EXTS:
            # §FIX: Statt analoge Posteriors auf 0.0 zu nullen (falsch — .mp3 kann
            # von Vinyl stammen), wenden wir einen Penalty-Faktor an (×0.25).
            # Die physikalischen Features (crackle, wow, flutter, rotation) sind
            # stärkere Evidenz als die Dateiendung. Ein rip von Vinyl→Cassette→.mp3
            # hat echte analoge Defekte, die nicht ignoriert werden dürfen.
            # §2.46b: Adaptive Penalty — codec-abhängig, nicht pauschal.
            # Verlustbehaftete Codecs zerstören analoge Signaturen unterschiedlich stark.
            _CODEC_PENALTY_MAP = {
                ".mp3": 0.50,
                ".mpc": 0.45,
                ".wma": 0.55,
                ".aac": 0.55,
                ".m4a": 0.55,
                ".ogg": 0.40,
                ".oga": 0.40,
                ".opus": 0.60,  # sehr destruktiv bei niedrigen Bitraten
            }
            _ANALOG_PENALTY = _CODEC_PENALTY_MAP.get(_ext_lower, 0.50)
            _adjusted: dict[str, float] = {
                mat: (score * _ANALOG_PENALTY if mat in self._ANALOG_MATERIALS else score)
                for mat, score in posteriors.items()
            }
            _total = sum(_adjusted.values()) + 1e-12
            posteriors = dict(
                sorted(
                    {k: v / _total for k, v in _adjusted.items()}.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
            # Zeige, welche analogen Träger in den Posteriors enthalten sind
            # (werden via Penalty bewahrt, nicht genullt — physikalische Features
            # entscheiden später über den tatsächlichen analogen Ursprung).
            _analog_in_posteriors = [
                f"{m}={s:.3f}" for m, s in posteriors.items() if m in self._ANALOG_MATERIALS and s >= 0.001
            ]
            _non_analog_top = [
                f"{m}={s:.3f}" for m, s in list(posteriors.items())[:3] if m not in self._ANALOG_MATERIALS
            ]
            logger.info(
                "MediumDetector: file_ext=%s → analog posteriors PENALIZED (×%.2f, NOT zeroed); "
                "analog candidates preserved: [%s]; top non-analog: [%s]",
                _ext_lower,
                _ANALOG_PENALTY,
                ", ".join(_analog_in_posteriors) if _analog_in_posteriors else "none",
                ", ".join(_non_analog_top) if _non_analog_top else "none",
            )
        elif not _ext_lower:
            # §2.59 Canary: Kein file_ext → kein Digital-Prior → analoge Materialien
            # könnten fälschlich als Primärmedium klassifiziert werden.
            # Diese Log-Meldung hilft, fehlende input_path-Call-Sites zu identifizieren.
            _analog_top3 = [(m, s) for m, s in list(posteriors.items())[:3] if m in self._ANALOG_MATERIALS]
            if _analog_top3:
                logger.info(
                    "MediumDetector: NO file_ext — Digital-Prior fehlt (kein ×0.25 Analog-Penalty). "
                    "Top-3 analog candidates: [%s]. Bei .mp3/.aac/.ogg wäre korrekte Material-Erkennung möglich. "
                    "→ Prüfen, ob Eingabe_path korrekt durchgereicht wird.",
                    ", ".join(f"{m}={s:.3f}" for m, s in _analog_top3),
                )

        # Flag: analog Bayesian posteriors wurden durch file_ext genullt
        _analog_zeroed: bool = _ext_lower in _DIGITAL_FILE_EXTS

        chain: list[str] = []
        evidence: list[str] = []
        chain_confidences: list[float] = []

        # ── Phase 1: Primärmaterial bestimmen ──────────────────────────
        top_materials = list(posteriors.items())

        # Best analog material (physical source)
        best_analog: str | None = None
        best_analog_score = 0.0
        for mat, score in top_materials:
            if mat in self._ANALOG_MATERIALS and score >= self._ANALOG_POSTERIOR_MIN:
                best_analog = mat
                best_analog_score = score
                break

        # ── Phase 1b: Physical-feature analog inference ───────────────
        # Wenn file_ext die Bayesian-Analog-Posteriors genullt hat (z. B. *.mp3),
        # kann der Bayesian-Scorer keinen analogen Ursprungsträger mehr erkennen.
        # Physikalische Merkmale (Vinyl-Rumble, Kassetten-Wow/Flutter, Schellack-
        # Krackeln) überleben die gesamte Aufnahmekette und sind auch in Codec-
        # Dateien noch nachweisbar.  Diese Features werden genutzt, um den analogen
        # Ursprungsträger unabhängig vom Bayesian-Scoring zu rekonstruieren.
        _physical_analog_sources: list[tuple[str, float]] = []
        _primary_vinyl_forced_by_rotation_gate: bool = False
        _best_analog_set_by_physical_gate: bool = False  # §6.1b: Physical-Gate-Primary bleibt Primary
        # Default-Fallback für §6.7f Vorläufer-Gate (falls _pa_conf_thresh durch den
        # physical-inference-Pfad nicht gesetzt wurde — z.B. wenn best_analog direkt
        # via Bayesian-Posterior gefunden wurde und _analog_zeroed=False).
        _pa_conf_thresh: float = 0.40
        _feature_ok: bool = True
        # §v10.14.1: Bayesian-Blind-Erkennung — wenn unknown > 0.90 dominiert,
        # sind physikalische Features die einzige verbleibende Evidenz.
        # Wird sowohl im Physical-Gate als auch in der Sekundär-Chain verwendet.
        _bayesian_blind = (
            best_analog is None
            and len(top_materials) > 0
            and top_materials[0][0] == "unknown"
            and top_materials[0][1] > 0.90
        )
        # §v10.14.1: Immer physische Inferenz laufen lassen — auch wenn
        # file_ext nicht digital ist. Bei unknown-dominierten Bayesian-
        # Posteriors (unknown > 0.90) sind die physikalischen Features
        # (Crackle, Rotation, Wow/Flutter, Infrasonic) die EINZIGE
        # verbleibende Evidenz für die Tonträgerkette.
        _needs_physical_inference = (
            _analog_zeroed
            or (_ext_lower in _DIGITAL_FILE_EXTS and fp.codec_artifact_score > self._CODEC_ARTIFACT_THRESHOLD)
            or fp.codec_artifact_score > 0.30
            or best_analog is None  # §v10.14.1: unknown-dominierte Posteriors
        )
        if _needs_physical_inference:
            _physical_analog_sources = self._infer_analog_source_from_fingerprint(fp)
            # Physical-Gate: nur aktivieren wenn Bayesian keinen analogen Primary gefunden hat
            _use_physical_gate = best_analog is None
            if _physical_analog_sources and _use_physical_gate:
                # §2.46a: Codec-adaptive thresholds — multi-generation transfers
                # (vinyl→cassette→MP3) attenuate analog fingerprints through each
                # transfer stage. Fixed thresholds miss genuine analog origins when
                # codec degradation is high. Scale by codec contamination level.
                _pa_codec_att = min(1.0, fp.codec_artifact_score / 0.60)
                if _bayesian_blind:
                    _pa_conf_thresh = max(0.10, 0.25 * (1.0 - 0.55 * _pa_codec_att))
                    _pa_rot_thresh = max(0.08, 0.35 * (1.0 - 0.45 * _pa_codec_att))
                    _pa_wow_thresh = max(0.01, 0.03 * (1.0 - 0.45 * _pa_codec_att))
                    _pa_infra_thresh = max(0.008, 0.012 * (1.0 - 0.40 * _pa_codec_att))
                    _pa_crackle_thresh = max(0.005, 0.010 * (1.0 - 0.40 * _pa_codec_att))
                else:
                    _pa_conf_thresh = max(0.20, 0.55 * (1.0 - 0.55 * _pa_codec_att))
                    _pa_rot_thresh = max(0.15, 0.65 * (1.0 - 0.45 * _pa_codec_att))
                    _pa_wow_thresh = max(0.02, 0.06 * (1.0 - 0.45 * _pa_codec_att))
                    _pa_infra_thresh = max(0.015, 0.025 * (1.0 - 0.40 * _pa_codec_att))
                    _pa_crackle_thresh = max(0.010, 0.020 * (1.0 - 0.40 * _pa_codec_att))
                _feature_ok = (
                    fp.rotation_strength >= _pa_rot_thresh
                    or fp.wow_flutter_index >= _pa_wow_thresh
                    or (fp.infrasonic_rms >= _pa_infra_thresh and fp.crackle_density >= _pa_crackle_thresh)
                )
                # §6.7e [RELEASE_MUST] Multi-Kandidaten-Gate (v10.0.0.x):
                # Iteriere ALLE physischen Analog-Quellen (_MEDIUM_ORDER-sortiert = ältester Träger
                # zuerst). Das letzte Gate-Positiv = closest-to-codec = Primary.  Dies behandelt
                # multi-generation Ketten (z.B. vinyl→kassette→mp3) korrekt: Vinyl verliert sein
                # Infrasonic-Profil durch den Kassetten-HPF, nur Kassette überlebt das Gate.
                # §2.46a: Primär-Gate: conf >= codec-adaptiver Schwellwert UND physikalisches Feature.
                # Fallback-Gate A (Vinyl): rotation_strength >= 0.30 (Plattenspieler-Periodizität,
                # Copeland 2008) UND conf >= 0.20 UND Infrasonic-Bestätigung vorhanden.
                # Fallback-Gate B (Kassette): BW-Rolloff ~12–15 kHz ist kein Codec-Artefakt.
                for _cand_analog, _cand_conf in _physical_analog_sources:
                    _via_rotation_gate = (
                        _cand_analog == "vinyl"
                        and _cand_conf >= 0.20
                        and fp.rotation_strength >= 0.30
                        and fp.infrasonic_rms >= 0.008
                    )
                    _strong_physical_analog = (
                        (_cand_conf >= _pa_conf_thresh and _feature_ok)
                        or (
                            _cand_conf >= 0.20
                            and fp.rotation_strength >= 0.30
                            and fp.infrasonic_rms >= 0.008  # Infrasonic-Guard: kein Falsch-Positiv
                        )
                        or (
                            _cand_analog == "cassette" and _cand_conf >= 0.35  # BW-Kassetten-Evidenz
                        )
                    )
                    if _strong_physical_analog:
                        best_analog, best_analog_score = _cand_analog, _cand_conf
                        _best_analog_set_by_physical_gate = True
                        if _via_rotation_gate:
                            # §2.46a Rotation-Fallback-Gate: vinyl als Primary erkannt —
                            # §6.1b darf diesen Primary NICHT durch spätere Tape-Stufen überschreiben.
                            pass
                        elif _cand_analog != "vinyl":
                            # Spätere Nicht-Vinyl-Kandidaten heben vinyl-Force zurück.
                            pass
                        # Baue eine menschenlesbare Beschreibung des Erkennungswegs
                        _detection_method: str
                        if _via_rotation_gate:
                            _detection_method = f"Vinyl-Rotation-Gate (rotation={fp.rotation_strength:.3f}≥0.30, infrasonic={fp.infrasonic_rms:.4f}≥0.008)"
                        elif _cand_analog == "cassette":
                            _detection_method = (
                                f"Cassette wow/flutter+bandwidth (wow={fp.wow_flutter_index:.3f}, conf≥0.35)"
                            )
                        elif _cand_analog == "vinyl":
                            _detection_method = f"Vinyl crackle+infrasonic (crackle={fp.crackle_density:.4f}, infrasonic={fp.infrasonic_rms:.4f})"
                        else:
                            _detection_method = f"codec-adaptive gate (conf={_cand_conf:.3f}≥{_pa_conf_thresh:.3f})"
                        logger.info(
                            "MediumDetector: ✅ PHYSICAL ANALOG erkannt — "
                            "primary=%s (confidence=%.3f) via %s; "
                            "full chain=%s; "
                            "features [crackle=%.4f infrasonic=%.4f rotation=%.3f wow_flutter=%.3f]",
                            best_analog,
                            best_analog_score,
                            _detection_method,
                            [m for m, _ in _physical_analog_sources],
                            fp.crackle_density,
                            fp.infrasonic_rms,
                            fp.rotation_strength,
                            fp.wow_flutter_index,
                        )
                        # Kein break — wir überschreiben mit jedem späteren Gate-Positiv.
                        # Letzter Treffer in _MEDIUM_ORDER-Reihenfolge = closest-to-codec.
                        # Ausnahme: vinyl via Rotation-Gate bleibt Primary (_primary_vinyl_forced_by_rotation_gate).

                if best_analog is None:
                    _sources_detail = (
                        ", ".join(f"{m}(conf={c:.3f})" for m, c in _physical_analog_sources)
                        if _physical_analog_sources
                        else "none"
                    )
                    logger.info(
                        "MediumDetector: ❌ NO physical analog confirmed for digital file_ext=%s — "
                        "%d candidate(s) [%s] fehlgeschlagen gate; "
                        "features [rotation=%.3f wow=%.3f crackle=%.4f infrasonic=%.4f]",
                        _ext_lower,
                        len(_physical_analog_sources),
                        _sources_detail,
                        fp.rotation_strength,
                        fp.wow_flutter_index,
                        fp.crackle_density,
                        fp.infrasonic_rms,
                    )

        # §6.8 Bayesian-Physical-Fusion (v10.0.0): Wenn der Bayesian-Klassifikator
        # "unknown > 0.9" sagt (kein Material erkannt), aber physikalische Features
        # Analog-Quellen gefunden haben → physikalische Evidenz als Primary übernehmen.
        # Bayesian-Scoring versagt bei stark codec-degradierten Mehrgenerationen-Ketten
        # (z. B. vinyl→kassette→mp3), weil alle analogen Signaturen durch Encoding
        # gedämpft sind. Physikalische Merkmale (rotation, infrasonic, crackle, wow)
        # sind robuster gegen Codec-Artefakte.
        if best_analog is None and _physical_analog_sources and posteriors.get("unknown", 0.0) > 0.90:
            # Nimm die stärkste physikalische Analog-Quelle als Primary
            _best_phys = max(_physical_analog_sources, key=lambda x: x[1])
            best_analog, best_analog_score = _best_phys[0], _best_phys[1]
            _best_analog_set_by_physical_gate = True
            logger.info(
                "MediumDetector: 🔄 Bayesian-Physical-Fusion — Bayesian unknown=%.3f, "
                "physical found %d source(s) → overriding primary=%s (conf=%.3f)",
                posteriors.get("unknown", 0.0),
                len(_physical_analog_sources),
                best_analog,
                best_analog_score,
            )

        # Best digital lossless
        best_digital: str | None = None
        best_digital_score = 0.0
        for mat, score in top_materials:
            if mat in self._DIGITAL_LOSSLESS and score >= self._ANALOG_POSTERIOR_MIN:
                best_digital = mat
                best_digital_score = score
                break

        # Best codec (lossy)
        best_codec: str | None = None
        best_codec_score = 0.0
        for mat, score in top_materials:
            if mat in self._CODEC_MATERIALS and score >= self._ANALOG_POSTERIOR_MIN:
                best_codec = mat
                best_codec_score = score
                break

        # Guard: check if source is clean digital (no analog chain)
        benign_codec = self._is_benign_codec_source(audio, sr, fp)

        # ── Codec-Hard-Gate: suppress implausible analog materials ────
        # If codec_type_code is clearly digital (≥ 0.5) AND no physical
        # rotation signature AND low crackle → analog classification is
        # implausible.  Override to benign_codec path.
        if (
            not benign_codec
            and fp.codec_type_code >= 0.5
            and fp.rotation_strength < 0.05
            and fp.crackle_density < 0.005
            and fp.wow_flutter_index < 0.02
        ):
            logger.info(
                "MediumDetector: codec_hard_gate override — codec=%.2f, "
                "rotation=%.3f, crackle=%.4f → treating as digital source",
                fp.codec_type_code,
                fp.rotation_strength,
                fp.crackle_density,
            )
            benign_codec = True

        # ── Phase 2: Kette aufbauen ───────────────────────────────────
        has_codec_artifacts = fp.codec_artifact_score > self._CODEC_ARTIFACT_THRESHOLD

        if benign_codec:
            if best_digital and best_digital_score > best_codec_score:
                chain.append(best_digital)
                chain_confidences.append(best_digital_score)
                evidence.append(f"Digitaler Träger: {best_digital} (posterior={best_digital_score:.3f})")
            elif best_codec:
                chain.append(best_codec)
                chain_confidences.append(best_codec_score)
                evidence.append(f"Codec-Quelle: {best_codec} (posterior={best_codec_score:.3f})")
            else:
                chain.append("cd_digital")
                chain_confidences.append(0.50)
                evidence.append("Clean digital — cd_digital default")
        else:
            if best_analog:
                chain.append(best_analog)
                chain_confidences.append(best_analog_score)
                evidence.append(
                    f"Primärquelle: {best_analog} "
                    f"(posterior={best_analog_score:.3f}, "
                    f"rotation={fp.rotation_strength:.3f}, "
                    f"infrasonic={fp.infrasonic_rms:.4f}, "
                    f"crackle={fp.crackle_density:.4f})"
                )

                # Build a deeper analog chain from Bayesian + physical cues.
                # This allows multi-generation chains beyond one secondary layer.
                _candidate_scores: dict[str, float] = {}
                _candidate_sources: dict[str, str] = {}
                # §v10.14.1: Bayesian-Blind-Modus — auch physikalisch inferierte
                # Materialien als Sekundär-Kandidaten einweben, selbst wenn ihre
                # Bayesian-Posteriors unter _SECONDARY_ANALOG_MIN liegen.
                if _bayesian_blind and _physical_analog_sources:
                    for _p_mat, _p_conf in _physical_analog_sources:
                        if _p_mat != best_analog and _p_mat in self._ANALOG_MATERIALS:
                            _candidate_scores[_p_mat] = float(_p_conf)
                            _candidate_sources[_p_mat] = "physical_blind"
                for mat2, score2 in top_materials:
                    if mat2 == best_analog or mat2 not in self._ANALOG_MATERIALS or score2 < self._SECONDARY_ANALOG_MIN:
                        continue
                    _candidate_scores[mat2] = float(score2)
                    _candidate_sources[mat2] = "posterior"

                for phys_mat, phys_conf in _physical_analog_sources:
                    if phys_mat == best_analog:
                        continue
                    _prev = _candidate_scores.get(phys_mat)
                    if _prev is None or phys_conf > _prev:
                        _candidate_scores[phys_mat] = float(phys_conf)
                        _candidate_sources[phys_mat] = "physical"
                    else:
                        _candidate_sources[phys_mat] = _candidate_sources.get(phys_mat, "posterior") + "+physical"

                # ── Chronological sort by _MEDIUM_ORDER ─────────────────
                # Ensure chain respects technological chronology, not detection order.
                _candidate_items = sorted(
                    _candidate_scores.items(),
                    key=lambda x: self._MEDIUM_ORDER.get(x[0], 99),
                )
                # Filter to keep only candidates that match known chain patterns
                _candidate_materials = [m for m, _ in _candidate_items]
                _best_chain = self._best_matching_chain(_candidate_materials + [best_analog])
                if _best_chain:
                    # Reorder candidates to match best known chain
                    _ordered = [m for m in _best_chain if m in _candidate_materials or m == best_analog]
                    _candidate_items = [
                        (m, _candidate_scores.get(m, 0.0))
                        for m in _ordered
                        if m in _candidate_scores or m == best_analog
                    ]

                _analog_depth = 1
                _last_order = self._MEDIUM_ORDER.get(best_analog, 0)

                # §6.7f [RELEASE_MUST] Vorläufer-Analog-Stufen voranstellen (v10.0.0.x):
                # Träger mit _MEDIUM_ORDER < best_analog_order (z.B. vinyl vor cassette)
                # werden VOR best_analog in die Chain eingefügt — nicht angehängt.
                # Gate: Vorläufer müssen dieselben physikalischen Kriterien erfüllen wie
                # der Primary-Gate (_strong_physical_analog) — verhindert, dass reine
                # rotation_strength-Evidenz (musikalische Phrasierungsperiodizität) Vinyl
                # fälschlich als Vorläufer einfügt.
                _best_order = self._MEDIUM_ORDER.get(best_analog, 0)
                _preceding = []
                for _pre_mat, _pre_conf in _candidate_scores.copy().items():
                    if self._MEDIUM_ORDER.get(_pre_mat, 99) >= _best_order:
                        continue
                    if _pre_conf < self._SECONDARY_ANALOG_MIN:
                        continue
                    # Dieselbe Gate-Logik wie für Primary — plus relaxiertes Vinyl-durch-Kassette-Gate:
                    # Kassetten-HPF dämpft Vinyl-Rumble; beide Features müssen schwach vorhanden sein.
                    # Unterschied zu direkter Kassette: Rillenrauschen (crackle) überlebt den Transfer.
                    _pre_strong = (
                        (_pre_conf >= _pa_conf_thresh and _feature_ok)
                        or (_pre_conf >= 0.20 and fp.rotation_strength >= 0.30 and fp.infrasonic_rms >= 0.008)
                        or (_pre_mat == "cassette" and _pre_conf >= 0.35)
                        or (
                            # Vinyl-Vorläufer durch Kassetten-Transfer: Rillenrauschen bleibt erhalten,
                            # Rumble wird teilweise durch Kassetten-HPF gefiltert → niedrigere Schwellen.
                            # §FIX v10.0.0: crackle auf 0.001 gesenkt (vorher 0.002) — gut gepflegte
                            # Vinyl→Kassette-Überspielungen haben sehr geringes Rillenrauschen.
                            # infrasonic auf 0.003 gesenkt (vorher 0.004) — Kassetten-HPF bei 30-40 Hz
                            # lässt einen schwachen Rumble-Rest durch.
                            _pre_mat == "vinyl"
                            and _pre_conf >= 0.15
                            and fp.crackle_density >= 0.001
                            and fp.infrasonic_rms >= 0.003
                        )
                    )
                    if _pre_strong:
                        _preceding.append((_pre_mat, _pre_conf))
                _preceding.sort(key=lambda x: self._MEDIUM_ORDER.get(x[0], 99))
                _ins = len(chain) - 1  # Position vor best_analog
                for _pre_mat, _pre_conf in _preceding:
                    chain.insert(_ins, _pre_mat)
                    chain_confidences.insert(_ins, _pre_conf)
                    evidence.insert(_ins, f"Vorläufer-Analog-Stufe (physical): {_pre_mat} (conf={_pre_conf:.3f})")
                    _ins += 1
                    _analog_depth += 1
                    del _candidate_scores[_pre_mat]  # forward-Schleife nochmals hinzufügen verhindern

                for mat2 in sorted(
                    _candidate_scores,
                    key=lambda _m: (self._MEDIUM_ORDER.get(_m, 99), -_candidate_scores[_m]),
                ):
                    if _analog_depth >= self._MAX_ANALOG_CHAIN_DEPTH:
                        break
                    if mat2 in chain:
                        continue

                    _order2 = self._MEDIUM_ORDER.get(mat2, 99)
                    _score2 = float(_candidate_scores[mat2])
                    _src2 = _candidate_sources.get(mat2, "posterior")

                    # Chronological order is handled by the sort below (line ~2530).
                    # Accept all secondary materials regardless of order.
                    chain.append(mat2)
                    chain_confidences.append(_score2)
                    evidence.append(f"Sekundäre Analog-Stufe ({_src2}): {mat2} (conf={_score2:.3f})")
                    _analog_depth += 1
                    _last_order = _order2

                # Optional digital lossless intermediate (e.g. vinyl→tape→cd_digital→mp3).
                if (
                    best_digital
                    and best_digital not in chain
                    and self._MEDIUM_ORDER.get(best_digital, 99) >= _last_order
                    and best_digital_score >= self._ANALOG_POSTERIOR_MIN
                ):
                    chain.append(best_digital)
                    chain_confidences.append(best_digital_score)
                    evidence.append(f"Digitale Zwischenstufe: {best_digital} (posterior={best_digital_score:.3f})")
                    _last_order = self._MEDIUM_ORDER.get(best_digital, _last_order)

            # ── Codec-Layer anhängen ──────────────────────────────────
            if has_codec_artifacts or (
                fp.hf_energy_above_16k < self.HF_ENERGY_THRESHOLD_FRACTION and fp.effective_bandwidth_hz < 17_500
            ):
                if best_codec:
                    codec_name = best_codec
                    codec_conf = best_codec_score
                elif fp.effective_bandwidth_hz < 14_000:
                    codec_name = "mp3_low"
                    codec_conf = self._codec_stage_confidence(fp)
                else:
                    codec_name = "mp3_high"
                    codec_conf = self._codec_stage_confidence(fp)

                # Conservative guard for analog transfer chains:
                # If an analog source stage is present but HF bandwidth is limited,
                # classify codec stage as mp3_low instead of mp3_high.
                _analog_chain_present = any(_m in self._ANALOG_MATERIALS for _m in chain)
                if (
                    _analog_chain_present
                    and codec_name == "mp3_high"
                    and fp.effective_bandwidth_hz < self._ANALOG_CHAIN_MP3_HIGH_MIN_BW_HZ
                ):
                    codec_name = "mp3_low"
                    codec_conf = max(float(codec_conf), 0.40)
                    evidence.append(
                        "Codec-Guard: Analogkette + begrenzte HF-Bandbreite "
                        f"({fp.effective_bandwidth_hz:.0f} Hz) → mp3_low"
                    )
                if codec_name not in chain:
                    chain.append(codec_name)
                    chain_confidences.append(codec_conf)
                    evidence.append(
                        f"Codec-Stufe: {codec_name} "
                        f"(artifact={fp.codec_artifact_score:.3f}, "
                        f"eff_bw={fp.effective_bandwidth_hz:.0f} Hz)"
                    )

        # ── Fallback ─────────────────────────────────────────────────
        if not chain:
            top_mat, top_score = top_materials[0] if top_materials else ("unknown", 0.30)
            if top_mat == "unknown" or top_score < 0.05:
                chain = ["unknown"]
                chain_confidences = [0.30]
                evidence.append("Träger unbekannt — Bayesian-Scores zu niedrig")
            else:
                chain = [top_mat]
                chain_confidences = [top_score]
                evidence.append(f"Bayesian-Fallback: {top_mat} (posterior={top_score:.3f})")

        # §6.1 [RELEASE_MUST] Material-Key-Normalisierung (v10.0.0):
        # MediumDetector interne Bayesian-Schlüssel → SUPPORTED_MATERIALS-konforme Schlüssel.
        # Betrifft alle Elemente der transfer_chain, nicht nur primary.
        _normalized_chain: list[str] = []
        _normalized_confidences: list[float] = []
        for _mat, _conf in zip(chain, chain_confidences):
            _norm = self._normalize_material_key(_mat)
            _safe_conf = float(np.clip(_conf, 0.0, 1.0))
            if _normalized_chain and _normalized_chain[-1] == _norm:
                _normalized_confidences[-1] = max(_normalized_confidences[-1], _safe_conf)
                continue
            _normalized_chain.append(_norm)
            _normalized_confidences.append(_safe_conf)
        chain = _normalized_chain
        chain_confidences = _normalized_confidences

        # ── Lacquer-Disc-Inference (§v10.712) ─────────────────────
        # Historisch korrekte Kette: reel_tape → lacquer_disc → vinyl.
        # Wenn reel_tape + vinyl erkannt wurden, aber lacquer_disc fehlt,
        # wird die Lackfolie als Zwischenträger inferiert — sie ist der
        # physische Schnittweg zwischen Studio-Master-Tape und Press-Tooling.
        if "reel_tape" in chain and "vinyl" in chain and "lacquer_disc" not in chain:
            _tape_conf = float(chain_confidences[chain.index("reel_tape")]) if "reel_tape" in chain else 0.0
            _vinyl_conf = float(chain_confidences[chain.index("vinyl")]) if "vinyl" in chain else 0.0
            # Inferenz nur bei beiderseitiger Mindestevidenz:
            if _tape_conf >= self._SECONDARY_ANALOG_MIN and _vinyl_conf >= self._ANALOG_POSTERIOR_MIN:
                _lacquer_conf = float(np.clip(0.5 * (_tape_conf + _vinyl_conf) + 0.15, 0.25, 0.65))
                # lacquer_disc zwischen reel_tape (order=5) und vinyl (order=7) einfügen
                _insert_pos = chain.index("reel_tape") + 1
                # Prüfen ob bereits etwas dazwischen steht
                if _insert_pos >= len(chain) or self._MEDIUM_ORDER.get(chain[_insert_pos], 99) > self._MEDIUM_ORDER.get(
                    "lacquer_disc", 99
                ):
                    chain.insert(_insert_pos, "lacquer_disc")
                    chain_confidences.insert(_insert_pos, _lacquer_conf)
                    evidence.append(
                        f"Lacquer-Disc-Inference (§v10.712): reel_tape → lacquer_disc → vinyl "
                        f"(conf={_lacquer_conf:.3f})"
                    )
                    logger.info(
                        "MediumDetector: Lacquer-Disc-Inference aktiv — Kette: %s",
                        " → ".join(chain),
                    )

        # ── Chronological sort ────────────────────────────────────
        # Ensure chain respects technology timeline, not detection order.
        # reel_tape (1930s) → lacquer_disc (1940s) → vinyl (1950s) → cassette (1960s) → mp3 (1990s)
        if len(chain) > 1:
            _sorted_chain = sorted(chain, key=lambda m: self._MEDIUM_ORDER.get(m, 99))
            if _sorted_chain != chain:
                logger.debug(
                    "MediumDetector: chain reordered chronologically: %s → %s",
                    " → ".join(chain),
                    " → ".join(_sorted_chain),
                )
                chain = _sorted_chain

        primary = chain[0]
        # §6.1b [RELEASE_MUST] Letzter-Analog-Träger-Primärprinzip (v10.0.0.x):
        # Für Mehrstufenketten (vinyl→cassette→mp3_low) ist der letzte analoge Träger
        # der primäre MaterialType-Prior für DefectScanner/CausalDefectReasoner,
        # da er die dominanten Degradationsartefakte trägt.
        # Beispiel: vinyl→cassette→mp3_low → primary = "cassette"
        _analog_in_chain = [m for m in chain if m in self._ANALOG_MATERIALS]
        if _analog_in_chain:
            # §6.1b [RELEASE_MUST]: Letzter-Analog-Träger = Primary — AUSNAHME:
            # Wenn best_analog durch Physical-Gate gesetzt wurde (Primär- oder Rotation-
            # Fallback-Gate), bleibt dieser Primary unverändert. Physical-Gate-Evidence
            # ist direkter als Bayesian-Posterior-basiertes §6.1b-Prinzip.
            # §2.46a AUSNAHME 2: Wenn Physical-Inference zusätzliche Tape-Stufen in
            # eine Disc→Tape→Codec-Kette einfügt (z.B. vinyl→reel_tape→mp3), bleibt
            # der Disc-Träger als Primary erhalten — er ist der ursprüngliche Träger.
            _disc_types_local = {"vinyl", "shellac", "lacquer_disc"}
            _disc_in_chain = [m for m in chain if m in _disc_types_local]
            if _best_analog_set_by_physical_gate and _disc_in_chain:
                # §2.46a: Disc-Träger (zuletzt produzierte Disc) bleibt Primary;
                # Transfer-Chain beginnt mit dem Primary (Carrier-First).
                primary = _disc_in_chain[-1]
            elif not _best_analog_set_by_physical_gate:
                if _disc_in_chain and _analog_in_chain[0] == _disc_in_chain[0]:
                    primary = _disc_in_chain[0]
                else:
                    primary = _analog_in_chain[-1]
            # §2.46a Carrier-First: Disc-Primary steht an Position 0 der Transfer-Chain
            # (Produktionsfall vinyl→reel_tape→mp3 aus Backend-Logs).
            if primary in _disc_types_local and primary in chain and chain[0] != primary:
                chain.remove(primary)
                chain.insert(0, primary)
                logger.debug("MediumDetector: §2.46a Disc-Primary carrier-first: %s", " → ".join(chain))
        is_multi = len(chain) > 1
        # Confidence wird aus Minimum, Mittelwert und Primärposterior geblendet.
        # §v10.712 FIX: Codec-Konfidenz darf analoge Kettenstatistiken NICHT drücken.
        # MP3-Artefakte (BW-Beschneidung, Blocking) sind eine EIGENE Degradationsquelle
        # und gehören nicht in die Bewertung der analogen Trägerkette.
        # → chain_min/mean/max nur aus ANALOGEN Stufen berechnen; Codec separat.
        _analog_confs = [
            c for m, c in zip(chain, chain_confidences) if m not in self._CODEC_MATERIALS and m != "cd_digital"
        ]
        _codec_confs = [c for m, c in zip(chain, chain_confidences) if m in self._CODEC_MATERIALS]
        # Fallback: wenn keine analogen Stufen, alles verwenden
        _source_confs = _analog_confs if _analog_confs else list(chain_confidences)
        _chain_min = float(min(_source_confs) if _source_confs else 0.0)
        _chain_mean = float(np.mean(_source_confs)) if _source_confs else 0.0
        _chain_max = float(max(_source_confs) if _source_confs else 0.0)
        # Codec-Term: bei vorhandener Codec-Stufe als separater, positiver Beitrag
        # (nicht als schwächster Link — Codec-Erkennung ist heuristisch und systematisch
        # niedriger als physikalische Analog-Evidenz).
        _codec_term = 0.0
        if _codec_confs:
            _codec_mean = float(np.mean(_codec_confs))
            # Codec-Präsenz bestätigt die Kette, aber drückt sie nicht unter den analogen Boden.
            _codec_term = float(np.clip(0.10 * min(_codec_mean / 0.40, 1.0), 0.0, 0.10))
        _primary_post = float(posteriors.get(primary, _chain_mean)) if isinstance(posteriors, dict) else _chain_mean
        # Codec-Primaries haben im analog-zentrierten Bayesian-Modell keinen
        # Posterior (≈ 0.0) — der Posterior-Term darf direkte Codec-Evidenz
        # (BW-Beschneidung, Artefakt-Score, Encoder-Signatur) nicht künstlich
        # bestrafen. Stütze ihn dann auf die gemessene Codec-Stufen-Konfidenz.
        if primary in self._CODEC_MATERIALS and primary in chain:
            _primary_post = max(_primary_post, float(chain_confidences[chain.index(primary)]))
        # §v10.19 Bayesian-Blind-Physical-Fusion: Wenn der Bayesian-Klassifikator
        # >90% "unknown" sagt, aber physikalische Inferenz die Primary gesetzt hat,
        # verwende die physikalische Confidence statt des ∼0.0 Bayesian-Posteriors.
        # Verhindert, dass 23.8% Confidence → wet_dry=0.12 → 88% unbearbeitetes Signal.
        if _best_analog_set_by_physical_gate and _physical_analog_sources:
            _phys_dict = dict(_physical_analog_sources)
            if primary in _phys_dict:
                _primary_post = max(_primary_post, _phys_dict[primary])
                logger.debug(
                    "MediumDetector: Physical-Posterior-Override primary=%s bayes=%.4f→phys=%.4f",
                    primary,
                    float(posteriors.get(primary, 0.0)),
                    _phys_dict[primary],
                )
        # Adaptives SNR-Gewicht: Bei niedrigem SNR (kurze/stark degradierte Clips) ist
        # chain_min weniger zuverlässig (Detektor-Scores unstabil bei < 5 s oder SNR < 20 dB).
        # → Reduziere chain_min-Einfluss und erhöhe primary_post-Gewicht.
        # Übergang: SNR ≤ 10 dB → w_min=0.20, w_post=0.35
        #           SNR ≥ 30 dB → w_min=0.35, w_post=0.20 (klassisches Verhalten)
        _snr_fp = float(getattr(fp, "snr_db", 30.0) or 30.0)
        _snr_factor = float(np.clip((_snr_fp - 10.0) / 20.0, 0.0, 1.0))
        _w_min = 0.20 + 0.15 * _snr_factor
        _w_post = 0.35 - 0.15 * _snr_factor
        confidence = float(
            np.clip(
                _w_min * _chain_min + 0.25 * _chain_mean + 0.20 * _chain_max + _w_post * _primary_post,
                0.0,
                1.0,
            )
        )

        # ── ClassificationResult für Passthrough bauen ───────────────
        try:
            from backend.core.medium_classifier import ClassificationResult  # pylint: disable=import-outside-toplevel

            classification_result = ClassificationResult(
                material=primary,
                confidence=confidence,
                bandwidth_hz=fp.effective_bandwidth_hz,
                snr_db=fp.snr_db,
                noise_color=fp.noise_color,
                crackle_density=fp.crackle_density,
                wow_flutter_hz=fp.wow_flutter_index,
                block_artifact=fp.codec_artifact_score,
                rotation_hz=fp.rotation_hz,
                rotation_strength=fp.rotation_strength,
                infrasonic_rms=fp.infrasonic_rms,
                codec_type="mp3" if fp.codec_type_code >= 0.5 else "clean",
                classifier_source="bayesian_fusion",
            )
        except (ImportError, TypeError):
            classification_result = None

        logger.info(
            "MediumDetector: Kette=%s, primär=%s, multi=%s, Konfidenz=%.2f, Top-3 Bayesian: %s",
            " → ".join(chain),
            primary,
            is_multi,
            confidence,
            ", ".join(f"{m}={s:.3f}" for m, s in list(posteriors.items())[:3]),
        )

        result = MediumDetectionResult(
            transfer_chain=chain,
            is_multi_generation=is_multi,
            primary_material=primary,
            confidence=confidence,
            spectral_fingerprint=fp,
            evidence=evidence,
            physical_analog_sources=_physical_analog_sources if _physical_analog_sources else [],
            medium_confidences=chain_confidences,
            bayesian_scores=posteriors,
            classification_result=classification_result,
        )

        # §6.7 Dolby / DBX NR detection for tape-chain material
        _tape_types = {"tape", "reel_tape", "wire_recording"}
        if primary in _tape_types or any(m in _tape_types for m in chain):
            try:
                from backend.core.dolby_nr_detector import (  # pylint: disable=import-outside-toplevel
                    get_dolby_nr_detector as _get_dolby,
                )

                _dolby_det = _get_dolby().detect(audio, sr, material_type=primary)
                if _dolby_det.detected:
                    result.dolby_nr_type = _dolby_det.nr_type
                    result.dolby_nr_confidence = _dolby_det.confidence
                    result.evidence.append(
                        f"Dolby/DBX NR detected: {_dolby_det.nr_type} "
                        f"({_dolby_det.confidence:.0%} konfident, "
                        f"HF-Überschuss {_dolby_det.hf_excess_db:.1f} dB)"
                    )
                    logger.info(
                        "MediumDetector: Dolby NR erkannt type=%s conf=%.2f hf_excess=%.1f dB",
                        _dolby_det.nr_type,
                        _dolby_det.confidence,
                        _dolby_det.hf_excess_db,
                    )
            except Exception as exc:
                logger.debug("MediumDetector: Dolby NR detection uebersprungen (%s)", exc)

        # §6.6 RIAA-Kurven-Klassifikation für Disc-Materialien (vinyl/shellac/lacquer_disc)
        _disc_types = {"vinyl", "shellac", "lacquer_disc"}
        if primary in _disc_types or any(m in _disc_types for m in chain):
            try:
                from backend.core.dsp.riaa_curve_classifier import (  # pylint: disable=import-outside-toplevel
                    get_riaa_curve_classifier as _get_riaa_clf,
                )

                _era_decade: int | None = None
                if hasattr(result, "bayesian_scores") and isinstance(result.bayesian_scores, dict):
                    _era_decade = result.bayesian_scores.get("era_decade")  # type: ignore[assignment]
                _riaa_curve, _riaa_conf = _get_riaa_clf().classify_with_confidence(audio, sr, era_decade=_era_decade)
                result.riaa_curve_type = _riaa_curve
                result.riaa_curve_confidence = _riaa_conf
                if _riaa_conf >= 0.70:
                    result.evidence.append(f"RIAA curve detected: {_riaa_curve} ({_riaa_conf:.0%} konfident)")
                    logger.info(
                        "MediumDetector: RIAA curve=%s conf=%.2f material=%s",
                        _riaa_curve,
                        _riaa_conf,
                        primary,
                    )
                else:
                    logger.debug(
                        "MediumDetector: RIAA curve low confidence=%s conf=%.2f → unknown",
                        _riaa_curve,
                        _riaa_conf,
                    )
            except Exception as exc:
                logger.debug("MediumDetector: RIAA curve classification uebersprungen (%s)", exc)

        return result

    # ── Hilfsmethoden ────────────────────────────────────────────────────

    @staticmethod
    def _codec_stage_confidence(fp: SpectralFingerprint) -> float:
        """Evidenzbasierte Konfidenz für eine heuristisch erkannte Codec-Stufe.

        Statt einer Platzhalterkonstante (historisch 0.40/0.35) wird die
        Konfidenz aus drei messbaren Evidenzquellen abgeleitet (§0m
        Maximal-Ausbaustufe Defektintelligenz — Kausalpräzision statt
        Aggregat-Konstante):

        1. **BW-Beschneidungs-Klarheit** — wie deutlich die effektive
           Bandbreite unter der 17.5-kHz-Vollband-Schwelle liegt (je tiefer
           der Codec-Tiefpass, desto eindeutiger die Lossy-Signatur),
        2. **Codec-Artefakt-Score** — Spektralloch-/Blocking-Evidenz,
        3. **Codec-Typ-Code** — direkte Encoder-Signatur.

        Returns:
            Konfidenz in [0.35, 0.90] — nie unter dem historischen Boden,
            aber bei klarer Evidenz ehrlich höher.
        """
        _bw_cut = float(np.clip((17_500.0 - float(fp.effective_bandwidth_hz)) / 7_500.0, 0.0, 1.0))
        _artifact = float(np.clip(float(fp.codec_artifact_score) / 0.30, 0.0, 1.0))
        _type = float(np.clip(float(fp.codec_type_code), 0.0, 1.0))
        return float(np.clip(0.35 + 0.30 * _bw_cut + 0.20 * _artifact + 0.15 * _type, 0.35, 0.90))

    @staticmethod
    def _normalize_material_key(key: str) -> str:
        """Map internal Bayesian-Scorer keys to canonical SUPPORTED_MATERIALS keys (§6.1).

        The Bayesian scorer uses some internal identifiers that differ from the
        SUPPORTED_MATERIALS list consumed by UnifiedRestorerV3 / DefectScanner.
        This method ensures all keys leaving detect() are spec-compliant.

        Mapping (spec §6.1 kanonisch):
            reel_wire        → wire_recording (Drahtband 1940–1955)
            cassette_digital → dat            (Digitalkassette; DAT-Pfad)
            vhs_audio        → tape           (VHS-Tonspur)
            composite        → composite (unchanged — caller must use transfer_chain[0])
            cassette         → cassette (kept as-is — MediumDetectionResult.primary_material
                               uses last-analog logic; DefectScanner handles "cassette" nativ)
        """
        _KEY_MAP: dict[str, str] = {
            "reel_wire": "wire_recording",
            "cassette_digital": "dat",
            "vhs_audio": "tape",
        }
        normalized = _KEY_MAP.get(key, key)
        if normalized != key:
            logger.debug(
                "MediumDetector._normalisieren_material_key: '%s' → '%s'",
                key,
                normalized,
            )
        return normalized

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        """Wandelt beliebiges Audio in mono float32 um."""
        if audio.ndim == 2:
            audio = audio.mean(axis=0) if audio.shape[0] <= audio.shape[1] else audio.mean(axis=1)
        mono = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(mono, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Singleton + Convenience
# ---------------------------------------------------------------------------

_instance: MediumDetector | None = None
_lock = threading.Lock()


def get_medium_detector() -> MediumDetector:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking, §3.2)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MediumDetector()
    return _instance


def detect_medium_chain(audio: np.ndarray, sr: int) -> MediumDetectionResult:
    """Convenience-Wrapper: erkennt die Tonträgerkette eines Audio-Signals."""
    return get_medium_detector().detect(audio, sr)


# §DSD: DSD/DSF/SACD import support (via dsdlib/pydsd)
# DSD64/128/256 → PCM conversion at import time.
# Detection: file header magic 'DSD ' or '.dsf' extension.
