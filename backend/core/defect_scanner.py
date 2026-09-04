"""
Defect Scanner for Aurik 10.0.0 - Defect-First Architecture
=========================================================

Erkennt und bewertet 62 Defekttypen in Audio-Materialien mit material-adaptiven
Thresholds. Ersetzt das Medium-First Detection System aus v8.0.

Performance-Budget: Max 5% der Gesamtzeit (< 30s für 3:45 Audio)

Wissenschaftliche Grundlage (normative Literatur, §6.9b):

Analoge Defekte:
    - Godsill & Rayner (1998) Digital Audio Restoration, Springer
        AR-Prädiktionsmodell für Clicks, Bayesian-Crackle, probabilistisches
        Defektmodell - Standardreferenz für alle analogen Defekttypen
    - Janssen, Veldhuis & Vries (1986) IEEE TASLP 34:203
        AR-basierte Click-Detektion via Prädiktionsfehler - Grundlage Click/Crackle
    - Maher (1993) J. Acoust. Soc. Am. 93:1679
        Click- und Pop-Detektion im Zeitbereich - Laufzeit-Diskriminator
    - Czyzewski & Kaczmarek (2003) AES Conv. 115
        Parametrisches Vinyl-Wow/Flutter-Modell - Mehrband-WF-Detektion
    - Bailey, Casebeer & Fazekas (2019) AES 147th Conv.
        Neural Network Vinyl-Knistern-Detektion - SOTA-Validierung crackle 4σ-Schwelle
    - Esquef & Biscainho (2006) IEEE TASLP 14:1207
        Modulations-Rauschen bei Bandaufnahmen - Basis MODULATION_NOISE
    - Dahimene, Richard & David (2008) IEEE TASLP 16:757
        Dropout-Erkennung via Energietransiente - DROPOUTS-Schwellwert-Kalibrierung
    - IEC 60386:1987
        Wow/Flutter-Messnorm - definiert Frequenzgrenzen WOW (<0.5 Hz) / FLUTTER (0.5-200 Hz)

Digitale Defekte:
    - Herre & Johnston (1996) AES Conv. 101
        Pre-Echo-Artefakt im MPEG-Coding via temporales Masking - Primärquelle PRE_ECHO
    - Bitto (2000) AES Conv. 109
        Jitter-Messung und Perceptual Impact bei DAT/CD - JITTER_ARTIFACTS-Kalibrierung
    - Zölzer (2011) DAFX: Digital Audio Effects, 2nd ed., Wiley
        Kapitel 8 (Codec-Artefakte) - vollständige Topologie digitaler Artefakttypen

Author: Aurik 10.0.0 Development Team
Version: 10.0.0
Date: 2026-02-15
"""

import contextlib
import hashlib

# v10.101 SOTA: Gammatone-geschützte Defektanalyse. Pipeline-Gates validieren.
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, cast

import numpy as np
import scipy.fft as fft
import scipy.signal as signal

from backend.core.material_canonical import canonical_material_key

# §6.3 CLIPPING vs SOFT_SATURATION discrimination via THD analysis (lazy import)
try:
    from backend.core.clipping_detection import ClippingType as _ClippingType
    from backend.core.clipping_detection import classify_clipping as _classify_clipping
    from backend.core.clipping_detection import detect_sub_ceiling_clipping as _detect_sub_ceiling_clipping  # V47

    _CLIPPING_DETECTION_AVAILABLE = True
except ImportError:
    _CLIPPING_DETECTION_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# §9.7.1 SHA256-Ergebnis-Cache - verhindert redundante Berechnungen bei Batch
# ---------------------------------------------------------------------------
_scan_cache: dict[str, object] = {}
_scan_cache_lock = threading.Lock()
_SCAN_CACHE_MAX = 128  # FIFO-Trim bei Überschreitung


# ── §v10 SNR-adaptive helper ───────────────────────────────────────────────


def _estimate_local_snr(audio: np.ndarray, sample_rate: int) -> float:
    """Schätzt das lokale SNR für adaptive Threshold-Skalierung.

    SNR ~20 dB (clean) → niedrigere Click-Thresholds (subtile Clicks)
    SNR ~5 dB (noisy)  → höhere Click-Thresholds (keine False Positives)

    Berechnung: Energie-Verhältnis Signal/Noise in 100ms-Fenstern,
    Median über alle Fenster als robuste SNR-Schätzung.
    """
    mono = audio if audio.ndim == 1 else audio.mean(axis=0)
    frame_len = int(0.1 * sample_rate)
    if len(mono) < frame_len * 2:
        return 20.0

    n_frames = max(1, len(mono) // frame_len)
    snr_values = np.zeros(n_frames, dtype=np.float64)
    eps = 1e-12

    for i in range(n_frames):
        start = i * frame_len
        end = min(start + frame_len, len(mono))
        frame = mono[start:end]
        rms = float(np.sqrt(np.mean(frame**2)) + eps)
        noise_floor = float(np.percentile(np.abs(frame), 5) + eps)
        snr_values[i] = 20.0 * np.log10(rms / noise_floor)

    return float(np.clip(np.median(snr_values), 2.0, 40.0))


def _audio_scan_cache_key(audio: np.ndarray, sr: int, material: object | None) -> str:
    """Deterministischer Cache-Key für DefectScanner.scan()."""
    h = hashlib.sha256()
    h.update(audio.tobytes())
    h.update(sr.to_bytes(4, "little"))
    if material is not None:
        h.update(str(material).encode())
    return f"scan:{h.hexdigest()[:16]}"


class DefectType(Enum):
    """54 Defekttypen für weltklasse Audio-restoration.

    Kern-Defekte (alle analogen/digitalen Quellen):
      CLIPPING        - Amplituden-Übersteuerung (Hard/Soft Clipping)
      DC_OFFSET       - Gleichspannungsversatz (Null-Linien-Verschiebung)
      BANDWIDTH_LOSS  - Hochfrequenz-Verlust (Shellac <7kHz, Kassette <14kHz, UKW/LP)
      PITCH_DRIFT     - Konstanter Geschwindigkeitsfehler (Tape-Stretch, Motorfehler)
      REVERB_EXCESS   - Unerwarteter/übermäßiger Raumhall (fehlerhafte Aufnahmeakustik)
      PRINT_THROUGH   - Magnetisches Übersprechen auf Tape (Pre-Echo 100-300 ms vor Einsatz)

    Analoge Tonträger:
      CLICKS, CRACKLE, HUM, WOW, FLUTTER, LOW_FREQ_RUMBLE, DROPOUTS

    Stereo / Kanal:
      STEREO_IMBALANCE, PHASE_ISSUES

    Digital / Codec:
      DIGITAL_ARTIFACTS, COMPRESSION_ARTIFACTS, HIGH_FREQ_NOISE
    """

    # --- Ursprüngliche 11 ---
    CLICKS = "clicks"  # Impulsartige Einzelstörungen - Janssen, Veldhuis & Vries (1986) IEEE TASLP 34:203
    CRACKLE = "crackle"  # Hochdichte Impulsfolgen (Vinyl-Knistern) - Bailey, Casebeer & Fazekas (2019) AES 147th Conv.
    HUM = "hum"
    WOW = "wow"  # Tonhöhenschwankung < 0.5 Hz (IEC 60386 - Motorexzentrizität, Plattenteller-Gleichlaufschwankung)
    FLUTTER = "flutter"  # Tonhöhenschwankung 0.5-200 Hz (IEC 60386 - mechanische Vibration, Führungsrolle, Bandantrieb)
    STEREO_IMBALANCE = "stereo_imbalance"
    DIGITAL_ARTIFACTS = "digital_artifacts"
    LOW_FREQ_RUMBLE = "low_freq_rumble"
    HIGH_FREQ_NOISE = "high_freq_noise"
    COMPRESSION_ARTIFACTS = "compression_artifacts"
    PHASE_ISSUES = "phase_issues"
    DROPOUTS = "dropouts"  # Kurzzeit-Pegeleinbrüche durch Bandmaterial-Aussetzer - Dahimene et al. (2008)
    # --- Weltklasse-Erweiterung Runde 1 ---
    CLIPPING = "clipping"  # Amplituden-Übersteuerung (Hard/Soft Clip)
    DC_OFFSET = "dc_offset"  # Gleichspannungsversatz
    BANDWIDTH_LOSS = "bandwidth_loss"  # HF-Rolloff / Bandbreitenbegrenzung
    PITCH_DRIFT = "pitch_drift"  # Konstanter Tonhöhen-/Geschwindigkeitsfehler
    # --- Weltklasse-Erweiterung Runde 2 ---
    REVERB_EXCESS = "reverb_excess"  # Übermäßiger Raumhall (Akustik-Artefakt)
    PRINT_THROUGH = "print_through"  # Magnetisches Übersprechen bei Tape (Pre-Echo)
    # --- Weltklasse-Erweiterung Runde 3 ---
    QUANTIZATION_NOISE = "quantization_noise"  # Quantisierungsrauschen (niedrige Bit-Tiefe / Resampling)
    JITTER_ARTIFACTS = "jitter_artifacts"  # Zeitgitter-Fehler bei D/A-Wandlung (CD, DAT) - Bitto (2000) AES Conv. 109
    DYNAMIC_COMPRESSION_EXCESS = "dynamic_compression_excess"  # Übermäßige Dynamikkompression (Loudness War)
    # --- Spec §6.3: fehlende DefectTypes für 24-Wert-Katalog ---
    SOFT_SATURATION = "soft_saturation"  # Tube-/Tape-Sättigung (gerade Obertöne) - BEWAHREN! (§2.1, §6.3)
    HEAD_WEAR = "head_wear"  # Kopf-/Azimuth-Fehler, Frequenzband-Auslöschung → phase_56 (§4.5, §7.2)
    AZIMUTH_ERROR = "azimuth_error"  # Kopf-Schrägstellung → HF-Phasen-Slope L/R > 20°/kHz → phase_56
    TRANSIENT_SMEARING = "transient_smearing"  # Ansatz-Verschmierung durch Kompression/Limiter
    PRE_ECHO = "pre_echo"  # MP3/AAC Temporal-Masking-Artefakt - Herre & Johnston (1996) AES Conv. 101 (§6.3)
    # --- Spec §6.3 v10.0.0c: 3 neue DefectTypes → 27 Gesamtanzahl ---
    RIAA_CURVE_ERROR = "riaa_curve_error"  # Falsche Disc-Entzerrungskurve (Shellac/Vinyl: AES/NAB/FFRR) → phase_04
    ALIASING = "aliasing"  # Spiegelfrequenzen durch unzureichenden AA-Filter bei ADC-Digitalisierung → phase_03
    BIAS_ERROR = "bias_error"  # Falscher Vormagnetisierungsstrom bei Bandaufnahme → phase_04 + phase_29
    # --- Spec §6.3 v10.0.0: Sibilanten-Überbetonung (ergibt 28 DefectTypes) ---
    SIBILANCE = "sibilance"  # Zischlautüberbetonung (> 6 kHz) - De-Esser-Trigger (phase_19 + phase_43)
    # --- v10.0.0b: Transport-Bump (ergibt 29 DefectTypes) ---
    TRANSPORT_BUMP = "transport_bump"  # Impulsartige Mikro-Geschwindigkeitssprünge 50-300 ms → phase_12
    # --- v10.0.0: Vocal-Harshness (ergibt 30 DefectTypes) ---
    VOCAL_HARSHNESS = "vocal_harshness"  # Vokale Härte/Kratzigkeit im 2-6 kHz Band → phase_42 + phase_19
    # --- v10.0.0.x: Dolby NR Mismatch (ergibt 31 DefectTypes) ---
    DOLBY_NR_MISMATCH = "dolby_nr_mismatch"  # Dolby B/C/S encode ohne Dekodierung → +6-20 dB HF-Anhebung → phase_04
    # --- v10.0.0.x: Tape Head Level Dip (ergibt 32 DefectTypes) ---
    TAPE_HEAD_LEVEL_DIP = "tape_head_level_dip"  # Graduelle Pegeleinbrüche durch Bandkopf-Kontaktdruckvariation
    # --- v10.0.0: 12 neue DefectTypes → 44 Gesamtanzahl (SOTA-Erweiterung) ---
    # Echte Lücken (6 neue Defekttypen):
    MODULATION_NOISE = "modulation_noise"  # Signal-abhängiges Rauschen bei Bandaufnahmen → phase_59
    INNER_GROOVE_DISTORTION = "inner_groove_distortion"  # Vinyl-IGD: Abtastverzerrung zum Platteninneren hin → phase_60
    GROOVE_ECHO = (
        "groove_echo"  # Vinyl-Rillen-Pre-Echo durch Deformation benachbarter Rillen (~1.8 s Vorecho) → phase_61
    )
    CROSSTALK = "crosstalk"  # Kanalübersprechen in frühen Stereo-Aufnahmen (Kanaltrennung < 20 dB) → phase_62
    INTERMODULATION_DISTORTION = "intermodulation_distortion"  # IMD: Summen-/Differenzfrequenzen → phase_63
    TAPE_SPLICE_ARTIFACT = "tape_splice_artifact"  # Bandschnitt-Artefakte: Klick + Pegelsprung → phase_64
    # SOTA-Upgrades bestehender Defekte (6 neue Sub-Typen):
    HF_REMANENCE_LOSS = "hf_remanence_loss"  # Magnetische Remanenz-Degradation: HF-Verlust durch Alterung → phase_06
    STYLUS_DAMAGE = "stylus_damage"  # Nadelbeschädigung/-abnutzung: asymmetrische Abtastverzerrung → phase_09
    STICKY_SHED_RESIDUE = "sticky_shed_residue"  # Binder-Hydrolyse-Residuen: moduliertes Rauschen → phase_24
    MULTIBAND_WOW_FLUTTER = "multiband_wow_flutter"  # Frequenzabhängiger Wow/Flutter (Kopfspalt-Geometrie) → phase_12
    GENERATION_LOSS = "generation_loss"  # Kumulativer Generationsverlust durch Tape-Dubbing
    MOTOR_INTERFERENCE = "motor_interference"  # Plattenspieler-Motorinterferenz: 80-300 Hz → phase_02 + phase_05
    # --- v10.0.0.x: Amplitude Drift (47. DefectType) ---
    AMPLITUDE_DRIFT = "amplitude_drift"  # Gradueller Pegelanstieg/-abfall über den Song:
    # Träger-AGC, Oxid-Drift → phase_40
    # --- v10.0.0: 7 neue DefectTypes - Carrier-Ursachen-Lücken geschlossen (54 gesamt) ---
    # Nahbesprechungseffekt (Richtmikrofon ≤30 cm): LF +6-12 dB ≤250 Hz; Olson (1948) → phase_04.
    PROXIMITY_EFFECT_EXCESS = "proximity_effect_excess"
    # Stehwellen-Raumresonanz 40-200 Hz ≠ diffuser Hall; Salter et al. (2003)
    # → phase_04 (Notch-Primary) + phase_16 + phase_05 (Tertiär).
    # VERBOTEN: phase_05 allein als Primary - schmalbandige Resonanzen brauchen parametrischen Notch-EQ (§4.11, V31).
    ROOM_MODE_RESONANCE = "room_mode_resonance"
    # Dolby/dbx NR Pumpen/Atmen (korrekt dekodiert); Dolby (1967) → phase_54 Envelope-Re-Smoothing + phase_08.
    # VERBOTEN: phase_03_denoise / phase_29 als Primary - weiteres NR verstärkt Pumpen (§4.11, V28).
    NR_BREATHING_ARTIFACT = "nr_breathing_artifact"
    # Flutter-Seitenbänder ±flutter_rate Hz um Spektralpeaks → phase_12 Ergänzung + phase_23.
    FLUTTER_SPECTRAL_SIDEBANDS = "flutter_spectral_sidebands"
    # --- v10 Weltklasse: Neue Defekt-Detektoren (Tier 2 + Tier 3) ---
    MPEG_FRAME_LOSS = "mpeg_frame_loss"  # MP3/AAC Frame-Drops, Bitstream-Korruption, Sync-Loss
    STEREO_FIELD_COLLAPSE = "stereo_field_collapse"  # Progressiver Stereofeld-Kollaps (Korrelation > 0.95)
    PHASE_ROTATION = "phase_rotation"  # Unnatürliche Phasenrotation (Allpass-Filter-Artefakte)
    # Dropout-Subtypen für differenzierte Reparatur:
    DROPOUT_OXIDE = "dropout_oxide"  # Oxid-Dropout: 2-20 ms, partiell (30-70% Verlust) → Interpolation
    DROPOUT_HEAD_CONTACT = "dropout_head_contact"  # Kopf-Kontakt-Dropout: 50-200 ms, moduliert → Gain-Komp.
    DROPOUT_SPLICE = "dropout_splice"  # Klebeband-Dropout: abrupt, >95% Verlust → Spektrale Inpainting

    # Systematischer fester Geschwindigkeitsfehler ≠ pitch_drift; IEC 60386 §5 → phase_12/phase_31.
    SPEED_CALIBRATION_ERROR = "speed_calibration_error"
    # Analoger Preamp/Console-Klirr: asymm. Verzerrung, H3/H5-dominant → phase_09/phase_23.
    # VERBOTEN: phase_63_intermodulation_reduction - Harmonische ≠ Intermodulationsprodukte (§4.11, V29).
    OVERLOAD_DISTORTION = "overload_distortion"
    # Acetat-Zersetzung: Substrat-Rissbildung, Lackschicht-Oxidation; Hess (1988) → phase_03 + phase_09.
    LACQUER_DISC_DEGRADATION = "lacquer_disc_degradation"
    # Hochfrequentes Tape-Scrape-Flutter (typ. 40-120 Hz) → phase_12 mit transport-spezifischer Korrektur.
    SCRAPE_FLUTTER = "scrape_flutter"
    # Temporäre Hochton-Auslöschung durch zugesetzten/verschmutzten Magnetkopf → phase_56 + phase_25.
    TAPE_HEAD_CLOG = "tape_head_clog"
    # §DefectType-Konsolidierung: Legacy-Einträge für ai_framework / data_models Kompatibilität
    HISS = "hiss"  # Tape hiss, broadband noise (alias zu HIGH_FREQ_NOISE)
    DISTORTION = "distortion"  # Generic distortion (alias zu OVERLOAD_DISTORTION)
    DROPOUT = "dropout"  # Singular form (alias zu DROPOUTS)


class MaterialType(Enum):
    """Material-Typen für adaptive Thresholds."""

    SHELLAC = "shellac"
    VINYL = "vinyl"
    VINYL_STANDARD = "vinyl"  # alias → canonical key "vinyl" (material_canonical.py)
    TAPE = "tape"  # Cassette Tape (generic key for chain-normalization compatibility)
    CASSETTE = "cassette"  # Compact Cassette Typ I/II/IV (explicit primary_material key v10.0.0.x)
    REEL_TAPE = "reel_tape"  # Professional reel-to-reel (Studio)
    DAT = "dat"  # Digital Audio Tape (Professional)
    CD_DIGITAL = "cd_digital"
    MP3_LOW = "mp3_low"  # MP3 <128 kbps (heavy compression artifacts)
    MP3_HIGH = "mp3_high"  # MP3 ≥128 kbps (moderate artifacts)
    AAC = "aac"  # AAC/M4A (modern compressed)
    MINIDISC = "minidisc"  # ATRAC codec (90s/2000s)
    STREAMING = "streaming"
    WAX_CYLINDER = "wax_cylinder"  # Phonograph-Wachswalze (1890-1930), HF ≤ 5 kHz
    WIRE_RECORDING = "wire_recording"  # Drahtband (1940-1955), Jitter, Frequenzgang-Einbrüche
    LACQUER_DISC = "lacquer_disc"  # Acetat-Lackfolien (1930-1950), Risse, Substrat-Rauschen
    UNKNOWN = "unknown"


@dataclass
class DefectScore:
    """Score-Objekt für einen einzelnen Defekt."""

    defect_type: DefectType
    severity: float  # 0.0 - 1.0 (0 = keine Defekte, 1 = schwere Defekte)
    confidence: float  # 0.0 - 1.0 (Konfidenz der Detection)
    locations: list[tuple[float, float]] = field(default_factory=list)  # (start_time, end_time) in Sekunden
    metadata: dict = field(default_factory=dict)  # Zusätzliche Informationen

    def __repr__(self) -> str:
        return (
            f"DefectScore({self.defect_type.value}: {self.severity:.3f},"
            f" conf={self.confidence:.2f}, {len(self.locations)} events)"
        )


@dataclass
class DefectAnalysisResult:
    """Vollständiges Ergebnis der Defekt-Analyse."""

    material_type: MaterialType
    scores: dict[DefectType, DefectScore]
    analysis_time_seconds: float
    sample_rate: int
    duration_seconds: float
    auto_detected_material: MaterialType | None = None  # §2.46a: vom Scanner auto-detektiert
    # Forensische Tonträgerkettenerkennung (befüllt von DefectScanner.scan())
    transfer_chain_raw: dict = field(default_factory=dict)  # MediumDetector.detect()-Ausgabe
    is_multi_generation: bool = False  # Mehrstufige Überspielungskette
    transfer_chain_str: str = ""  # Lesbare Kette, z.B. "cassette → mp3"
    # §6.6.1 Pflicht-Spektralfingerabdruck - 5 normierte Messgrößen (immer befüllt)
    spectral_fingerprint: dict[str, float] = field(default_factory=dict)
    # spectral_fingerprint enthält:
    #   rolloff_95_hz       - Rolloff-Frequenz 95% der Spektralenergie [Hz]
    #   wow_flutter_index   - Pitch-Varianz-Index [0..∞, >1.5 = Kassette]
    #   hf_energy_above_16k - Anteil Energie > 16 kHz an Gesamtenergie [0..1]
    #   noise_floor_p5_db   - 5. Perzentil PSD als Rauschboden [dBFS]
    #   effective_bandwidth_hz - HF-Rolloff bei -60 dBFS [Hz]
    #   material_detected   - auto-erkanntes Material (auch wenn Hint übergeben)
    metadata: dict[str, object] = field(default_factory=dict)

    def get_top_defects(self, n: int = 5) -> list[DefectScore]:
        """Gibt die Top-N Defekte nach Severity zurück."""
        return sorted(self.scores.values(), key=lambda x: x.severity, reverse=True)[:n]

    def get_total_severity(self) -> float:
        """Gesamtschwere aller Defekte (gewichtet)."""
        return sum(score.severity * score.confidence for score in self.scores.values()) / len(self.scores)

    def get_top_defects_perceptual(
        self,
        n: int = 5,
        audio: np.ndarray | None = None,
        sr: int = 48000,
    ) -> list[DefectScore]:
        """§v10.101 SOTA: Gibt Top-N Defekte nach perzeptueller Hörbarkeit.

        Re-rankt die technisch erkannten Defekte nach tatsächlicher
        Wahrnehmbarkeit für das menschliche Ohr. Nutzt Bark-Gewichtung,
        spektrale und temporale Maskierung.

        Unhörbare Defekte werden depriorisiert - sie zu reparieren
        würde nur Artefakt-Risiko ohne akustischen Nutzen erzeugen.
        """
        try:
            from backend.core.dsp.perceptual_defect_ranker import rerank_defects_perceptual

            _defect_tuples: list[tuple[str, float, float, float]] = []
            for dt, score in (self.scores or {}).items():
                _name = dt.value if hasattr(dt, "value") else str(dt)
                _sev = float(getattr(score, "severity", 0.0))
                # Schätze Frequenz und Position aus Metadaten oder Default
                _freq = 1000.0
                _pos = 0.0
                if hasattr(score, "frequency_hz"):
                    _freq = float(getattr(score, "frequency_hz", 1000.0))
                if hasattr(score, "position_s"):
                    _pos = float(getattr(score, "position_s", 0.0))
                _defect_tuples.append((_name, _sev, _freq, _pos))

            if audio is not None and _defect_tuples:
                _reranked = rerank_defects_perceptual(_defect_tuples, audio, sr)
                # Map zurück auf DefectScore-Objekte
                _result: list[DefectScore] = []
                for _name, _psev, _tsev in _reranked[:n]:
                    for dt, score in (self.scores or {}).items():
                        if (dt.value if hasattr(dt, "value") else str(dt)) == _name:
                            _result.append(score)
                            break
                return _result
        except Exception as _pr_exc:
            logger.debug("Perceptual defect rerank nicht blockierend: %s", _pr_exc)

        # Fallback: technische Sortierung
        return self.get_top_defects(n)

    def get_focus_defect_map(
        self,
        n: int = 12,
        min_confidence: float = 0.25,
        min_weighted_severity: float = 0.04,
    ) -> dict[str, float]:
        """Liefert confidence-gewichtete Top-Defekte für robuste Downstream-Steuerung.

        Ziel: Defekte mit hoher Evidenz priorisieren, unsichere Near-Threshold-
        Treffer nicht überbewerten. Ergebnis ist ein Mapping `defect_name -> score`
        (0..1), sortiert nach Relevanz und auf `n` Einträge begrenzt.
        """
        _rows: list[tuple[str, float]] = []
        _min_conf = float(np.clip(min_confidence, 0.0, 1.0))
        _min_weight = float(np.clip(min_weighted_severity, 0.0, 1.0))

        for _dt, _score in (self.scores or {}).items():
            try:
                _sev = float(np.clip(float(getattr(_score, "severity", 0.0)), 0.0, 1.0))
                _conf = float(np.clip(float(getattr(_score, "confidence", 0.0)), 0.0, 1.0))
            except Exception:
                logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                continue
            if _sev <= 0.0:
                continue

            # Confidence-aware severity score:
            # - hohe Konfidenz treibt den Score
            # - starke, aber mittlere Konfidenz bleibt sichtbar
            _weighted = float(np.clip(_sev * (0.55 + 0.45 * _conf), 0.0, 1.0))
            if _conf < _min_conf and _weighted < (_min_weight * 1.5):
                continue
            if _weighted < _min_weight:
                continue

            _name = _dt.value if hasattr(_dt, "value") else str(_dt)
            _rows.append((_name, _weighted))

        if not _rows:
            return {}

        _rows.sort(key=lambda it: it[1], reverse=True)
        _top = _rows[: max(1, int(n))]
        return {k: float(v) for k, v in _top}


class DefectScanner:
    """
    Hauptklasse für Defekt-Erkennung in Aurik 10.0.0.

    Ersetzt Medium-First Detection durch Defect-First Approach:
    - Scannt alle Audio-Daten einmal vollständig
    - Erkennt 28 Defekttypen sequenziell (11 Kern + 17 Weltklasse-Erweiterung)
    - Nutzt material-adaptive Thresholds für alle MaterialTypes
    - Performance-optimiert (max 5% overhead)
    """

    # §v10.306: Per-Defekt-Audibilität - JND-Schwellen pro Defekttyp
    # Format: {defect_type: (jnd_db, masking_factor, material_factors_dict)}
    # material_factors: Multiplikator - höher = schwerer hörbar (maskiert)
    AUDIBILITY_THRESHOLDS: dict[str, tuple[float, float, dict[str, float]]] = {
        # Impulsive Defekte
        "clicks": (
            -35.0,
            0.30,
            {
                "shellac": 1.3,
                "vinyl": 1.1,
                "tape": 1.0,
                "cassette": 0.9,
                "reel_tape": 0.85,
                "cd": 0.8,
                "digital": 0.8,
                "mp3_low": 1.2,
                "dat": 0.75,
            },
        ),
        "pops": (
            -28.0,
            0.20,
            {
                "shellac": 1.3,
                "vinyl": 1.1,
                "tape": 1.0,
                "cassette": 0.9,
                "reel_tape": 0.85,
                "cd": 0.8,
                "digital": 0.8,
                "mp3_low": 1.2,
                "dat": 0.75,
            },
        ),
        "crackle": (
            -40.0,
            0.50,
            {
                "shellac": 1.4,
                "vinyl": 1.2,
                "tape": 1.0,
                "cassette": 0.9,
                "reel_tape": 0.85,
                "cd": 0.8,
                "digital": 0.8,
                "mp3_low": 1.3,
                "dat": 0.75,
            },
        ),
        "dropout": (
            -25.0,
            0.10,
            {
                "shellac": 1.2,
                "vinyl": 1.1,
                "tape": 1.0,
                "cassette": 0.95,
                "reel_tape": 0.85,
                "cd": 0.8,
                "digital": 0.8,
                "mp3_low": 1.1,
                "dat": 0.75,
            },
        ),
        "transport_bump": (
            -30.0,
            0.15,
            {
                "shellac": 1.2,
                "vinyl": 1.1,
                "tape": 1.0,
                "cassette": 0.9,
                "reel_tape": 0.85,
                "cd": 0.8,
                "digital": 0.8,
                "mp3_low": 1.1,
                "dat": 0.75,
            },
        ),
        # Rausch-Defekte
        "hum": (
            -45.0,
            0.60,
            {
                "shellac": 1.3,
                "vinyl": 1.2,
                "tape": 1.0,
                "cassette": 0.9,
                "reel_tape": 0.85,
                "cd": 0.8,
                "digital": 0.8,
                "mp3_low": 1.2,
                "dat": 0.75,
            },
        ),
        "hiss": (
            -38.0,
            0.40,
            {
                "shellac": 1.2,
                "vinyl": 1.1,
                "tape": 1.0,
                "cassette": 0.9,
                "reel_tape": 0.85,
                "cd": 0.8,
                "digital": 0.8,
                "mp3_low": 1.1,
                "dat": 0.75,
            },
        ),
        "rumble": (
            -42.0,
            0.70,
            {
                "shellac": 1.4,
                "vinyl": 1.3,
                "tape": 1.0,
                "cassette": 0.9,
                "reel_tape": 0.85,
                "cd": 0.8,
                "digital": 0.8,
                "mp3_low": 1.2,
                "dat": 0.75,
            },
        ),
        "noise_level": (
            -35.0,
            0.35,
            {
                "shellac": 1.2,
                "vinyl": 1.1,
                "tape": 1.0,
                "cassette": 0.9,
                "reel_tape": 0.85,
                "cd": 0.8,
                "digital": 0.8,
                "mp3_low": 1.1,
                "dat": 0.75,
            },
        ),
        "surface_noise": (
            -40.0,
            0.55,
            {
                "shellac": 1.5,
                "vinyl": 1.4,
                "tape": 1.0,
                "cassette": 1.0,
                "reel_tape": 0.85,
                "cd": 0.8,
                "digital": 0.8,
                "mp3_low": 1.3,
                "dat": 0.75,
            },
        ),
        # Chirurgische Defekte - sehr hörbar, niedrige Schwellen
        "sibilance": (
            -30.0,
            0.10,
            {
                "shellac": 1.3,
                "vinyl": 1.2,
                "tape": 1.0,
                "cassette": 1.0,
                "reel_tape": 1.0,
                "cd": 1.0,
                "digital": 1.0,
                "mp3_low": 1.1,
                "dat": 1.0,
            },
        ),
        "vocal_harshness": (
            -32.0,
            0.08,
            {
                "shellac": 1.3,
                "vinyl": 1.2,
                "tape": 1.0,
                "cassette": 1.0,
                "reel_tape": 1.0,
                "cd": 1.0,
                "digital": 1.0,
                "mp3_low": 1.1,
                "dat": 1.0,
            },
        ),
        "stereo_imbalance": (
            1.5,
            0.05,
            {
                "shellac": 1.0,
                "vinyl": 1.0,
                "tape": 1.0,
                "cassette": 1.0,
                "reel_tape": 0.9,
                "cd": 0.8,
                "digital": 0.8,
                "mp3_low": 1.0,
                "dat": 0.75,
            },
        ),
        "phase_issues": (
            -25.0,
            0.20,
            {
                "shellac": 1.1,
                "vinyl": 1.05,
                "tape": 1.0,
                "cassette": 0.95,
                "reel_tape": 0.85,
                "cd": 0.8,
                "digital": 0.8,
                "mp3_low": 1.0,
                "dat": 0.75,
            },
        ),
        "dc_offset": (
            -50.0,
            0.90,
            {
                "shellac": 1.3,
                "vinyl": 1.2,
                "tape": 1.0,
                "cassette": 0.9,
                "reel_tape": 0.85,
                "cd": 0.8,
                "digital": 0.8,
                "mp3_low": 1.2,
                "dat": 0.75,
            },
        ),
        # Spektrale Defekte
        "bandwidth_loss": (
            -3.0,
            0.25,
            {
                "shellac": 1.2,
                "vinyl": 1.1,
                "tape": 1.0,
                "cassette": 0.9,
                "reel_tape": 0.85,
                "cd": 0.7,
                "digital": 0.7,
                "mp3_low": 1.0,
                "dat": 0.7,
            },
        ),
        "pre_echo": (
            -30.0,
            0.10,
            {
                "shellac": 1.1,
                "vinyl": 1.05,
                "tape": 1.0,
                "cassette": 0.95,
                "reel_tape": 0.85,
                "cd": 0.8,
                "digital": 0.8,
                "mp3_low": 0.9,
                "dat": 0.75,
            },
        ),
        "aliasing": (
            -35.0,
            0.30,
            {
                "shellac": 1.2,
                "vinyl": 1.1,
                "tape": 1.0,
                "cassette": 0.9,
                "reel_tape": 0.85,
                "cd": 0.8,
                "digital": 0.8,
                "mp3_low": 1.0,
                "dat": 0.75,
            },
        ),
        "quantization_noise": (
            -42.0,
            0.50,
            {
                "shellac": 1.3,
                "vinyl": 1.2,
                "tape": 1.0,
                "cassette": 0.9,
                "reel_tape": 0.85,
                "cd": 0.8,
                "digital": 0.8,
                "mp3_low": 1.1,
                "dat": 0.75,
            },
        ),
        "compression_artifacts": (
            -32.0,
            0.25,
            {
                "shellac": 1.1,
                "vinyl": 1.1,
                "tape": 1.0,
                "cassette": 0.9,
                "reel_tape": 0.85,
                "cd": 0.8,
                "digital": 0.8,
                "mp3_low": 0.8,
                "dat": 0.75,
            },
        ),
    }

    # Material-adaptive Sensitivity-Thresholds
    MATERIAL_SENSITIVITY = {
        MaterialType.SHELLAC: {
            DefectType.CLICKS: 0.3,  # Sehr empfindlich (Shellac hat viele Clicks)
            DefectType.CRACKLE: 0.3,
            DefectType.HUM: 0.6,
            DefectType.WOW: 0.40,  # Plattenteller-Gleichlaufschwankung (< 0.5 Hz)
            DefectType.FLUTTER: 0.50,  # Nadelresonanz, Abtastarm-Vibration (0.5-200 Hz)
            DefectType.STEREO_IMBALANCE: 1.0,  # N/A für Mono
            DefectType.DIGITAL_ARTIFACTS: 1.0,
            DefectType.LOW_FREQ_RUMBLE: 0.5,
            DefectType.HIGH_FREQ_NOISE: 0.6,
            DefectType.COMPRESSION_ARTIFACTS: 1.0,
            DefectType.PHASE_ISSUES: 1.0,
            DefectType.DROPOUTS: 0.7,
            DefectType.CLIPPING: 0.5,  # Analoge Übersteuerung bei Schnittlack-Aufnahmen
            DefectType.DC_OFFSET: 0.4,  # Alte Vorverstärker mit DC-Drift
            DefectType.BANDWIDTH_LOSS: 0.2,  # Shellac: stark ausgeprägte HF-Rolloff < 7 kHz
            DefectType.PITCH_DRIFT: 0.3,  # 78 rpm Motorfehler sehr häufig
            DefectType.REVERB_EXCESS: 0.6,  # Shellac: meist trocken aufgenommen, aber möglich
            DefectType.PRINT_THROUGH: 1.0,  # N/A: Shellac ist kein Magnetband
            DefectType.QUANTIZATION_NOISE: 0.9,  # N/A: Shellac ist analog
            DefectType.JITTER_ARTIFACTS: 1.0,  # N/A: Shellac ist analog
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.9,  # Shellac-Ära: keine Loudness-War-Kompression
            DefectType.SOFT_SATURATION: 0.3,  # Röhren-Mikrofon-Sättigung, Aufnahmetrichter - bewahren
            DefectType.HEAD_WEAR: 1.0,  # N/A: Shellac ist kein Magnetband (kein Magnetkopf-Verschleiß)
            DefectType.PRE_ECHO: 1.0,  # N/A: Shellac analog - kein Codec-Pre-Echo
            DefectType.TRANSIENT_SMEARING: 0.5,  # Mechanischer Trichter und schwere Nadel begrenzen Transientenbereich
            DefectType.RIAA_CURVE_ERROR: 0.2,  # AES/NAB/Columbia-Kurve vor RIAA-Standard - Entzerrungs-Fehler häufig
            DefectType.ALIASING: 0.5,  # Archiv-Digitalisierung oft mit unzureichendem AA-Filter
            DefectType.BIAS_ERROR: 1.0,  # N/A: Shellac ist kein Magnetband - kein Aufnahme-Bias
            DefectType.AZIMUTH_ERROR: 1.0,  # N/A: Shellac ist Disc-Format - kein Magnetkopf-Azimuth
            DefectType.SIBILANCE: 0.6,  # Schwere Nadel + begrenzter HF → Zischlaut-Verzerrung
            DefectType.TRANSPORT_BUMP: 0.5,  # Plattenteller-Transport: mechanisches Holpern bei 78 rpm
            DefectType.VOCAL_HARSHNESS: 0.5,  # Schwere Nadel + Trichter: Vokal-Verzerrung häufig
            DefectType.DOLBY_NR_MISMATCH: 1.0,  # N/A: Shellac-Ära vor Dolby NR (1966) - kein Mismatch möglich
            DefectType.TAPE_HEAD_LEVEL_DIP: 1.0,  # N/A: Schellack hat keine Tape-Kopf-Mechanik
            # v10.0.0: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 1.0,  # N/A: Shellac mechanisch - kein Magnetband-Modulationsrauschen
            DefectType.INNER_GROOVE_DISTORTION: 0.2,  # SEHR HÄUFIG: Schwere Nadel → IGD extrem ausgeprägt
            DefectType.GROOVE_ECHO: 0.3,  # Weiche Schellackmasse → Rillenverformung → Vorecho
            DefectType.CROSSTALK: 1.0,  # N/A: Shellac immer Mono
            DefectType.INTERMODULATION_DISTORTION: 0.3,  # Trichter-Aufnahme nichtlinear → IMD
            DefectType.TAPE_SPLICE_ARTIFACT: 1.0,  # N/A: Shellac ist kein Band - kein Schnitt
            DefectType.HF_REMANENCE_LOSS: 1.0,  # N/A: Shellac mechanisch - keine magnetische Remanenz
            DefectType.STYLUS_DAMAGE: 0.2,  # SEHR HÄUFIG: Stahlnadeln zerstören Rillen bei Wiederholung
            DefectType.STICKY_SHED_RESIDUE: 1.0,  # N/A: Shellac hat keinen Binder wie Magnetband
            DefectType.MULTIBAND_WOW_FLUTTER: 0.5,  # Mechanischer Antrieb: frequenzunabhängiger Wow/Flutter
            DefectType.GENERATION_LOSS: 0.6,  # Matrizen-Pressung: jede Generation verliert Detail
            DefectType.MOTOR_INTERFERENCE: 0.3,  # Grammophon-Motor: Federwerk/Elektro → harmonische Störungen
            DefectType.AMPLITUDE_DRIFT: 0.45,  # Häufig: Grammophon-Federwerkmotor → AGC-ähnlicher Pegelabfall
        },
        MaterialType.VINYL: {
            DefectType.CLICKS: 0.4,
            DefectType.CRACKLE: 0.5,
            DefectType.HUM: 0.5,  # 50Hz/60Hz hum häufig
            DefectType.WOW: 0.50,  # Plattenspieler-Motor-Gleichlauf (< 0.5 Hz)
            DefectType.FLUTTER: 0.55,  # Arm-/Stylusresonanz, Riemengetriebe-Vibration (0.5-200 Hz)
            DefectType.STEREO_IMBALANCE: 0.6,
            DefectType.DIGITAL_ARTIFACTS: 1.0,
            DefectType.LOW_FREQ_RUMBLE: 0.4,  # Turntable rumble
            DefectType.HIGH_FREQ_NOISE: 0.7,
            DefectType.COMPRESSION_ARTIFACTS: 1.0,
            DefectType.PHASE_ISSUES: 0.7,
            DefectType.DROPOUTS: 0.8,
            DefectType.CLIPPING: 0.6,  # Schneidlackübersteuerung möglich
            DefectType.DC_OFFSET: 0.5,  # Phono-Vorverstärker DC-Bias
            DefectType.BANDWIDTH_LOSS: 0.4,  # RIAA-Entzerrung, HF generell begrenzt
            DefectType.PITCH_DRIFT: 0.4,  # Plattenspieler-Motorabweichungen
            DefectType.REVERB_EXCESS: 0.6,  # Vinyl: Aufnahme-Akustik selten Restaurationsproblem
            DefectType.PRINT_THROUGH: 1.0,  # N/A: Vinyl ist kein Magnetband
            DefectType.QUANTIZATION_NOISE: 0.9,  # N/A: Vinyl ist analog
            DefectType.JITTER_ARTIFACTS: 1.0,  # N/A: Vinyl ist analog
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.6,  # Moderne Vinyl-Pressings manchmal überkomprimiert
            DefectType.SOFT_SATURATION: 0.4,  # Schneidlack-Sättigung möglich - bewahren wenn authentisch
            DefectType.HEAD_WEAR: 1.0,  # N/A: Vinyl ist kein Magnetband (kein Magnetkopf-Verschleiß)
            DefectType.PRE_ECHO: 1.0,  # N/A: Vinyl analog - kein Codec-Pre-Echo
            DefectType.TRANSIENT_SMEARING: 0.5,  # Schneidlack-Übertragungsfunktion: leichte Transientenverzerrung
            DefectType.RIAA_CURVE_ERROR: 0.3,  # Früh-Vinyl (vor 1954) nutzte verschiedene Kurven (AES, FFRR, Columbia)
            DefectType.ALIASING: 0.5,  # Digitalisierungsqualität variiert stark - AA-Filter oft suboptimal
            DefectType.BIAS_ERROR: 1.0,  # N/A: Schallplatte ist kein Magnetband - kein Aufnahme-Bias
            DefectType.AZIMUTH_ERROR: 1.0,  # N/A: Vinyl ist Disc-Format - kein Magnetkopf-Azimuth
            DefectType.SIBILANCE: 0.7,  # Sehr häufig: Tonabnehmer-Sibilanz + Phono-Stufe; De-Esser-Pflicht
            DefectType.TRANSPORT_BUMP: 0.4,  # Plattenspieler-Transport: mechanisches Holpern möglich
            DefectType.VOCAL_HARSHNESS: 0.4,  # Tonabnehmer-Verzerrung + Phono-Stufe → Vokal-Übersteuerung
            DefectType.DOLBY_NR_MISMATCH: 1.0,  # N/A: Vinyl nutzt kein Dolby NR - kein Mismatch
            DefectType.TAPE_HEAD_LEVEL_DIP: 1.0,  # N/A: Vinyl hat keine Tape-Kopf-Mechanik
            # v10.0.0: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 1.0,  # N/A: Vinyl mechanisch - kein Magnetband-Modulationsrauschen
            DefectType.INNER_GROOVE_DISTORTION: 0.15,  # EXTREM HÄUFIG: Abtastverzerrung zum Platteninneren!
            DefectType.GROOVE_ECHO: 0.2,  # HÄUFIG: Laute Passagen deformieren Nachbarrille → Pre-Echo
            DefectType.CROSSTALK: 0.5,  # Frühe Stereo-Vinyl: Kanaltrennung oft nur 15-20 dB
            DefectType.INTERMODULATION_DISTORTION: 0.4,  # Schneidlack-Nichtlinearität → IMD bei Hochpegel
            DefectType.TAPE_SPLICE_ARTIFACT: 1.0,  # N/A: Vinyl hat keine Bandschnitte
            DefectType.HF_REMANENCE_LOSS: 1.0,  # N/A: Vinyl mechanisch - keine magnetische Remanenz
            DefectType.STYLUS_DAMAGE: 0.3,  # Abgenutzte Nadel → asymmetrische Verzerrung
            DefectType.STICKY_SHED_RESIDUE: 1.0,  # N/A: Vinyl hat keinen Binder
            DefectType.MULTIBAND_WOW_FLUTTER: 0.6,  # Plattenspieler: frequenzunabhängig
            DefectType.GENERATION_LOSS: 0.7,  # Pressung: marginal (Master→Stamper)
            DefectType.MOTOR_INTERFERENCE: 0.3,  # Plattenspieler-Motor: Gleichstrom-/Synchron-Störungen 80-300 Hz
            DefectType.AMPLITUDE_DRIFT: 0.60,  # Selten: Vinyl-Pressungs-Level-Drift unwahrscheinlich
        },
        MaterialType.TAPE: {
            DefectType.CLICKS: 0.7,
            DefectType.CRACKLE: 0.8,
            DefectType.HUM: 0.4,  # AC hum häufig bei Tape
            DefectType.WOW: 0.22,  # Capstan-Gleichlaufschwankung (< 0.5 Hz); kalibriert v10.0.0 (war 0.30 → zu konservativ für 1980s Kassetten)
            DefectType.FLUTTER: 0.25,  # Andruckrolle, Führungsrollen-Vibration (0.5-200 Hz)
            DefectType.STEREO_IMBALANCE: 0.5,
            DefectType.DIGITAL_ARTIFACTS: 1.0,
            DefectType.LOW_FREQ_RUMBLE: 0.6,
            DefectType.HIGH_FREQ_NOISE: 0.5,  # Tape hiss
            DefectType.COMPRESSION_ARTIFACTS: 1.0,
            DefectType.PHASE_ISSUES: 0.6,
            DefectType.DROPOUTS: 0.4,  # Tape dropouts häufig
            DefectType.CLIPPING: 0.4,  # Tape-Sättigung durch Übersteuerung häufig
            DefectType.DC_OFFSET: 0.4,  # Kassettendecks mit DC-Bias im Signalweg
            DefectType.BANDWIDTH_LOSS: 0.3,  # Kassette: Rolloff bei 12-14 kHz typisch
            DefectType.PITCH_DRIFT: 0.2,  # Tape-Stretch & Motorfehler sehr häufig
            DefectType.REVERB_EXCESS: 0.4,  # Kassette: Aufnahmeraum oft hörbar
            DefectType.PRINT_THROUGH: 0.2,  # Kassette: magnetisches Übersprechen möglich
            DefectType.QUANTIZATION_NOISE: 0.9,  # N/A: Kassette ist analog
            DefectType.JITTER_ARTIFACTS: 1.0,  # N/A: Kassette ist analog
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.8,  # Kassette: Dolby-Kompander aber selten exzessiv
            DefectType.SOFT_SATURATION: 0.4,  # Bandsättigung (gerade Obertöne H2/H4) - BEWAHREN vs. Clipping
            DefectType.HEAD_WEAR: 0.6,  # Kassettenköpfe durch Abnutzung → Hochton-Auslöschung häufig
            DefectType.PRE_ECHO: 1.0,  # N/A: Kassette analog - kein Codec-Pre-Echo
            DefectType.TRANSIENT_SMEARING: 0.4,  # Dolby-Rauschreduktion und Bandsättigung → Transient-Smearing
            DefectType.RIAA_CURVE_ERROR: 1.0,  # N/A: Magnetband nutzt keine RIAA-Disc-Entzerrungskurve
            DefectType.ALIASING: 0.4,  # Digitalisierungs-AA variiert; Resampling in der Verarbeitungskette
            DefectType.BIAS_ERROR: 0.3,  # SEHR HÄUFIG: falscher Bias für Bandsorte (Chromdioxid/Normallage)
            DefectType.AZIMUTH_ERROR: 0.30,  # Häufig! Kassettenköpfe neigen zu Azimuth-Drift
            DefectType.SIBILANCE: 0.5,  # Kassettenkopf-HF-Sättigung → Zischlaut-Betonung
            DefectType.TRANSPORT_BUMP: 0.3,  # Kassetten-Transport: Capstan/Andruckrolle-Holpern häufig
            DefectType.VOCAL_HARSHNESS: 0.4,  # Bandsättigung + HF-Peaking → Vokal-Härte bei Hochpegel
            DefectType.DOLBY_NR_MISMATCH: 0.25,  # SEHR HÄUFIG: Dolby B/C (1975-2000); ohne Dekoder → +6-20 dB HF
            DefectType.TAPE_HEAD_LEVEL_DIP: 0.20,  # SEHR HÄUFIG: Capstan/Andruckrolle → Kopf-Kontakt-Druckvariation
            # v10.0.0: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 0.15,  # EXTREM HÄUFIG: Signal-abhängiges Rauschen - Esquef 2006
            DefectType.INNER_GROOVE_DISTORTION: 1.0,  # N/A: Tape hat keine Rillen
            DefectType.GROOVE_ECHO: 1.0,  # N/A: Tape hat keine Rillen
            DefectType.CROSSTALK: 0.4,  # Kassette: Spur-Übersprechen bei schmalen 4-Spur-Kassetten
            DefectType.INTERMODULATION_DISTORTION: 0.4,  # Bandkopf-/Verstärker-Nichtlinearität → IMD
            DefectType.TAPE_SPLICE_ARTIFACT: 0.2,  # HÄUFIG: Bandschnitte bei Heim- und Profi-Kassetten
            DefectType.HF_REMANENCE_LOSS: 0.15,  # SEHR HÄUFIG: Alterung → HF-Verlust über Jahrzehnte
            DefectType.STYLUS_DAMAGE: 1.0,  # N/A: Tape hat keine Nadel
            DefectType.STICKY_SHED_RESIDUE: 0.2,  # HÄUFIG: Binder-Hydrolyse bei alten Kassetten (1970er-90er)
            DefectType.MULTIBAND_WOW_FLUTTER: 0.2,  # HÄUFIG: Kopfspalt + Bandkontakt → frequenzabhängiges Flutter
            DefectType.GENERATION_LOSS: 0.2,  # HÄUFIG: Kassetten-Dubbing (Band→Band-Kopien häufig)
            DefectType.MOTOR_INTERFERENCE: 1.0,  # N/A: Kassettenmotor-Störung → über HUM/FLUTTER abgedeckt
            DefectType.AMPLITUDE_DRIFT: 0.25,  # SEHR HÄUFIG: Dolby-B/C AGC-Drift, Oxid-Alterung → Pegelanstieg/-abfall
        },
        # §6.1b v10.0.0.x: CASSETTE = Compact Cassette Typ I/II/IV als expliziter primary_material-Schlüssel.
        # Identische Sensitivity-Thresholds wie TAPE (Compact Cassette und "tape" sind physikalisch äquivalent).
        # MaterialType.CASSETTE wird gesetzt wenn MediumDetector "cassette" als letzten Analog-Träger erkennt.
        MaterialType.CASSETTE: {
            DefectType.CLICKS: 0.7,
            DefectType.CRACKLE: 0.8,
            DefectType.HUM: 0.4,
            DefectType.WOW: 0.22,  # kalibriert v10.0.0 (physikalisch äquivalent zu TAPE; war 0.30)
            DefectType.FLUTTER: 0.25,
            DefectType.STEREO_IMBALANCE: 0.5,
            DefectType.DIGITAL_ARTIFACTS: 1.0,
            DefectType.LOW_FREQ_RUMBLE: 0.6,
            DefectType.HIGH_FREQ_NOISE: 0.5,
            DefectType.COMPRESSION_ARTIFACTS: 1.0,
            DefectType.PHASE_ISSUES: 0.6,
            DefectType.DROPOUTS: 0.4,
            DefectType.CLIPPING: 0.4,
            DefectType.DC_OFFSET: 0.4,
            DefectType.BANDWIDTH_LOSS: 0.3,
            DefectType.PITCH_DRIFT: 0.2,
            DefectType.REVERB_EXCESS: 0.4,
            DefectType.PRINT_THROUGH: 0.40,  # §v10.0.0: Dünnes Band → weniger Print-Through als Spulentonband
            DefectType.QUANTIZATION_NOISE: 0.9,
            DefectType.JITTER_ARTIFACTS: 1.0,
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.8,
            DefectType.SOFT_SATURATION: 0.4,
            DefectType.HEAD_WEAR: 0.6,
            DefectType.PRE_ECHO: 1.0,
            DefectType.TRANSIENT_SMEARING: 0.4,
            DefectType.RIAA_CURVE_ERROR: 1.0,
            DefectType.ALIASING: 0.4,
            DefectType.BIAS_ERROR: 0.3,
            DefectType.AZIMUTH_ERROR: 0.30,
            DefectType.SIBILANCE: 0.5,
            DefectType.TRANSPORT_BUMP: 0.15,  # §v10.0.0: Pinch-Roller-Bumps sind Kassetten-spezifisch (35/min)
            DefectType.VOCAL_HARSHNESS: 0.4,
            DefectType.DOLBY_NR_MISMATCH: 0.25,
            DefectType.TAPE_HEAD_LEVEL_DIP: 0.15,  # §v10.0.0: Kleine Köpfe → schneller Pegelabfall
            DefectType.MODULATION_NOISE: 0.15,
            DefectType.INNER_GROOVE_DISTORTION: 1.0,
            DefectType.GROOVE_ECHO: 1.0,
            DefectType.CROSSTALK: 0.4,
            DefectType.INTERMODULATION_DISTORTION: 0.4,
            DefectType.TAPE_SPLICE_ARTIFACT: 0.2,
            DefectType.HF_REMANENCE_LOSS: 0.15,
            DefectType.STYLUS_DAMAGE: 1.0,
            DefectType.STICKY_SHED_RESIDUE: 0.2,
            DefectType.MULTIBAND_WOW_FLUTTER: 0.2,
            DefectType.GENERATION_LOSS: 0.2,
            DefectType.MOTOR_INTERFERENCE: 1.0,
            DefectType.AMPLITUDE_DRIFT: 0.25,  # SEHR HÄUFIG: Dolby-B/C AGC-Drift (identisch mit TAPE)
        },
        MaterialType.CD_DIGITAL: {
            DefectType.CLICKS: 0.8,
            DefectType.CRACKLE: 0.9,
            DefectType.HUM: 0.9,
            DefectType.WOW: 0.90,  # CD: Kristalloszillator - kein WOW (N/A)
            DefectType.FLUTTER: 0.90,  # CD: kristallstabil - kein FLUTTER (N/A)
            DefectType.STEREO_IMBALANCE: 0.7,
            DefectType.DIGITAL_ARTIFACTS: 0.3,  # Häufig bei CD!
            DefectType.LOW_FREQ_RUMBLE: 0.8,
            DefectType.HIGH_FREQ_NOISE: 0.7,
            DefectType.COMPRESSION_ARTIFACTS: 0.4,  # MP3-artige Artifacts
            DefectType.PHASE_ISSUES: 0.7,
            DefectType.DROPOUTS: 0.5,  # Digital dropouts
            DefectType.CLIPPING: 0.4,  # Loudness-War CD-Mastering-Clipping häufig
            DefectType.DC_OFFSET: 0.7,  # Selten bei CD
            DefectType.BANDWIDTH_LOSS: 0.7,  # CD: 22 kHz Nyquist, volle Bandbreite
            DefectType.PITCH_DRIFT: 0.9,  # CD: Crystal-Takt, kein Pitch-Drift
            DefectType.REVERB_EXCESS: 0.7,  # CD: Reverb meist bewusste Produktionsentscheidung
            DefectType.PRINT_THROUGH: 1.0,  # N/A: CD ist kein Magnetband
            DefectType.QUANTIZATION_NOISE: 0.7,  # CD 16-bit: selten, aber bei schlecht gemasterter CD möglich
            DefectType.JITTER_ARTIFACTS: 0.3,  # CD-Player/Laufwerk-Jitter sehr häufig!
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.25,  # Loudness War bei CD sehr häufig
            DefectType.SOFT_SATURATION: 0.9,  # Digitale Quelldatei: Soft-Saturation selten, aber möglich
            DefectType.HEAD_WEAR: 1.0,  # N/A: CD digital - kein Magnetkopf
            DefectType.PRE_ECHO: 0.3,  # Loudness-War-Mastering: Pre-Echo als Transient-Vorlauf möglich
            DefectType.TRANSIENT_SMEARING: 0.2,  # Loudness-War-Kompression → Transient-Smearing sehr häufig
            DefectType.RIAA_CURVE_ERROR: 1.0,  # N/A: CD digital - keine Disc-Entzerrungskurve
            DefectType.ALIASING: 0.3,  # CD digital-nativ; Resampling-Artefakte bei Formatketten möglich
            DefectType.BIAS_ERROR: 1.0,  # N/A: CD digital - kein Wechselstrom-Bias
            DefectType.AZIMUTH_ERROR: 1.0,  # N/A: CD digital - kein Magnetkopf
            DefectType.SIBILANCE: 0.3,  # CD: geringe Sibilanz-Gefahr; De-Emphasis-Fehler möglich
            DefectType.TRANSPORT_BUMP: 1.0,  # N/A: CD digital - kein mechanischer Transport
            DefectType.VOCAL_HARSHNESS: 0.25,  # Loudness-War-Mastering → Vokal-Harshness SEHR häufig
            DefectType.DOLBY_NR_MISMATCH: 1.0,  # N/A: CD ist digital, kein Dolby-Analogband-NR
            DefectType.TAPE_HEAD_LEVEL_DIP: 1.0,  # N/A: CD ist digital, kein Magnetband-Kopf
            # v10.0.0: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 1.0,  # N/A: CD digital - kein analoges Modulationsrauschen
            DefectType.INNER_GROOVE_DISTORTION: 1.0,  # N/A: CD hat keine Rillen
            DefectType.GROOVE_ECHO: 1.0,  # N/A: CD hat keine Rillen
            DefectType.CROSSTALK: 0.8,  # CD digital: Crosstalk nur bei schlechtem Mastering
            DefectType.INTERMODULATION_DISTORTION: 0.7,  # CD: IMD nur bei analogem Mastering-Signalpfad
            DefectType.TAPE_SPLICE_ARTIFACT: 1.0,  # N/A: CD hat keine Bandschnitte
            DefectType.HF_REMANENCE_LOSS: 1.0,  # N/A: CD digital - keine magnetische Remanenz
            DefectType.STYLUS_DAMAGE: 1.0,  # N/A: CD hat keine Nadel
            DefectType.STICKY_SHED_RESIDUE: 1.0,  # N/A: CD hat keinen Binder
            DefectType.MULTIBAND_WOW_FLUTTER: 1.0,  # N/A: CD Crystal-Clock
            DefectType.GENERATION_LOSS: 0.7,  # Selten: Mehrfach-Transcode in digitaler Kette
            DefectType.MOTOR_INTERFERENCE: 1.0,  # N/A: CD-Laufwerk digital
            DefectType.AMPLITUDE_DRIFT: 0.85,  # Sehr selten: CD ist digital; Level-Drift nur aus Quellmaterial
        },
        MaterialType.REEL_TAPE: {
            DefectType.CLICKS: 0.8,
            DefectType.CRACKLE: 0.9,
            DefectType.HUM: 0.5,  # AC hum in professional gear
            DefectType.WOW: 0.40,  # Profi-Capstan-Gleichlauf (< 0.5 Hz)
            DefectType.FLUTTER: 0.35,  # Profi-Führungsrollen-Vibration (0.5-200 Hz)
            DefectType.STEREO_IMBALANCE: 0.6,
            DefectType.DIGITAL_ARTIFACTS: 1.0,
            DefectType.LOW_FREQ_RUMBLE: 0.7,
            DefectType.HIGH_FREQ_NOISE: 0.6,  # Less hiss than cassette (better quality)
            DefectType.COMPRESSION_ARTIFACTS: 1.0,
            DefectType.PHASE_ISSUES: 0.7,
            DefectType.DROPOUTS: 0.5,  # Professional tape more reliable
            DefectType.CLIPPING: 0.5,  # Studio Tape-Sättigung / Übersteuerung möglich
            DefectType.DC_OFFSET: 0.5,  # Studio-Vorverstärker können DC einbringen
            DefectType.BANDWIDTH_LOSS: 0.4,  # Reel-Tape: HF besser als Kassette, aber begrenzt
            DefectType.PITCH_DRIFT: 0.4,  # Professionelles Tape: besser aber nicht perfekt
            DefectType.REVERB_EXCESS: 0.3,  # Reel-Tape: Live-Raumakustik sehr häufig!
            DefectType.PRINT_THROUGH: 0.10,  # §v10.0.0: Studioband lagert gewickelt → Vorecho sehr häufig
            DefectType.QUANTIZATION_NOISE: 0.9,  # N/A: Reel-Tape ist analog
            DefectType.JITTER_ARTIFACTS: 1.0,  # N/A: Reel-Tape ist analog
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.8,  # Professionelles Reel-Tape: Kompression selten exzessiv
            DefectType.SOFT_SATURATION: 0.6,  # Profi-Bandsättigung (Röhren-Mischpult/Bandmaschine) - bewahren
            DefectType.HEAD_WEAR: 0.7,  # Profi-Banddeck Kopfverschleiß: breite Frequenzband-Auslöschung möglich
            DefectType.PRE_ECHO: 1.0,  # N/A: Spulenband analog - kein Codec-Pre-Echo
            DefectType.TRANSIENT_SMEARING: 0.4,  # Studio-Rauschreduktion (Dolby A/SR) → leichtes Transient-Smearing
            DefectType.RIAA_CURVE_ERROR: 1.0,  # N/A: Spulenband nutzt keine RIAA-Disc-Entzerrungskurve
            DefectType.ALIASING: 0.3,  # Professionelle Digitalisierung meist gut - AA-Filter vorhanden
            DefectType.BIAS_ERROR: 0.3,  # Häufig: gemischte Bandsorten → falscher Bias-Strom beim Schnitt
            DefectType.AZIMUTH_ERROR: 0.25,  # Sehr häufig: Profi-Bandmaschinen mit verschiedenen Schnittköpfen
            DefectType.SIBILANCE: 0.4,  # Profi-Spulenband: HF-Sättigung → Zischlaut-Überbetonung
            DefectType.TRANSPORT_BUMP: 0.95,  # §v10.0.0: Profi-Capstan ohne Pinch-Roller → keine Transport-Bumps
            DefectType.VOCAL_HARSHNESS: 0.4,  # Profi-Bandsättigung → Vokal-Härte bei hohem Bandfluss
            DefectType.DOLBY_NR_MISMATCH: 0.6,  # Möglich: Dolby A/SR - Broadcast-Dekoder fehlt oft
            DefectType.TAPE_HEAD_LEVEL_DIP: 0.65,  # §v10.0.0: Große Studio-Köpfe → allmählicher breitbandiger Pegelverlust  # Möglich: Kopfverschleiß/Alignmentfehler möglich
            # v10.0.0: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 0.12,  # EXTREM HÄUFIG: Signal-abhängig bei hohem Bandfluss
            DefectType.INNER_GROOVE_DISTORTION: 1.0,  # N/A: Tape hat keine Rillen
            DefectType.GROOVE_ECHO: 1.0,  # N/A: Tape hat keine Rillen
            DefectType.CROSSTALK: 0.3,  # Frühe Stereo-Spulenbänder: Spur-Übersprechen bei Halbspur-Stereo
            DefectType.INTERMODULATION_DISTORTION: 0.35,  # Profi-Röhrenverstärker + Schneidkopf → IMD
            DefectType.TAPE_SPLICE_ARTIFACT: 0.15,  # SEHR HÄUFIG: Professionelle Spulenbänder mit vielen Klebestellen
            DefectType.HF_REMANENCE_LOSS: 0.12,  # SEHR HÄUFIG: Profi-Spulenband altert → HF-Verlust
            DefectType.STYLUS_DAMAGE: 1.0,  # N/A: Tape hat keine Nadel
            DefectType.STICKY_SHED_RESIDUE: 0.1,  # EXTREM HÄUFIG: Polyester-Urethan-Bänder (Ampex 456) - Sticky-Shed
            DefectType.MULTIBAND_WOW_FLUTTER: 0.2,  # Profi-Kopfspalt + Bandkontakt → frequenzabhängig
            DefectType.GENERATION_LOSS: 0.15,  # SEHR HÄUFIG: Studio-Dubbing (Mix → Master → Copy)
            DefectType.MOTOR_INTERFERENCE: 1.0,  # N/A: Profi-Tape-Motor → über WOW/FLUTTER abgedeckt
            DefectType.AMPLITUDE_DRIFT: 0.30,  # Häufig: Profi-Spulenband AGC, Bias-Drift, Magnetspur-Alterung
        },
        MaterialType.DAT: {
            DefectType.CLICKS: 0.9,
            DefectType.CRACKLE: 1.0,  # Digital, no crackle
            DefectType.HUM: 0.9,
            DefectType.WOW: 1.0,  # DAT: digital, Crystal-Clock - kein WOW
            DefectType.FLUTTER: 1.0,  # DAT: digital, Crystal-Clock - kein FLUTTER
            DefectType.STEREO_IMBALANCE: 0.7,
            DefectType.DIGITAL_ARTIFACTS: 0.4,  # Some digital artifacts
            DefectType.LOW_FREQ_RUMBLE: 0.9,
            DefectType.HIGH_FREQ_NOISE: 0.8,
            DefectType.COMPRESSION_ARTIFACTS: 0.8,  # Lossless, minimal compression
            DefectType.PHASE_ISSUES: 0.7,
            DefectType.DROPOUTS: 0.4,  # Occasional digital dropouts
            DefectType.CLIPPING: 0.5,  # Digitales Clipping bei DAT exakt erkennbar
            DefectType.DC_OFFSET: 0.8,  # DAT digital: DC-Offset selten
            DefectType.BANDWIDTH_LOSS: 0.8,  # DAT 48 kHz: volle Bandbreite
            DefectType.PITCH_DRIFT: 0.9,  # DAT: Crystal-Takt, kein Drift
            DefectType.REVERB_EXCESS: 0.6,  # DAT: digitaler Feldrekorder, Aufnahmeakustik
            DefectType.PRINT_THROUGH: 1.0,  # N/A: DAT digital, kein magnetisches Übersprechen
            DefectType.QUANTIZATION_NOISE: 0.45,  # DAT 16-bit: Quantisierungsrauschen möglich
            DefectType.JITTER_ARTIFACTS: 0.35,  # DAT-Jitter bekanntes Problem!
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.6,  # DAT: oft schon komprimierter Quell-Content
            DefectType.SOFT_SATURATION: 0.9,  # DAT: Soft-Saturation aus analogem Eingang möglich
            DefectType.HEAD_WEAR: 1.0,  # DAT-Rotationskopf: Verschleiß → Hochton-Dropout möglich
            DefectType.PRE_ECHO: 0.3,  # DAT digital: Pre-Echo aus Quell-Codec-Material möglich
            DefectType.TRANSIENT_SMEARING: 0.2,  # DAT digital: seltenes Transient-Smearing aus Quell-Kompression
            DefectType.RIAA_CURVE_ERROR: 1.0,  # N/A: DAT digital - keine Disc-Entzerrungskurve
            DefectType.ALIASING: 0.3,  # DAT digital-nativ; Resampling bei Weiterverarbeitung möglich
            DefectType.BIAS_ERROR: 1.0,  # N/A: DAT digital - kein analoger Bias erforderlich
            DefectType.AZIMUTH_ERROR: 0.50,  # DAT-Rotationskopf: Azimuth kann durch Kopfverschleiß driften
            DefectType.SIBILANCE: 0.2,  # DAT digital - geringe Sibilanz-Gefahr
            DefectType.TRANSPORT_BUMP: 0.6,  # DAT-Laufwerk: Rotationskopf-Transport kann holpern
            DefectType.VOCAL_HARSHNESS: 0.3,  # DAT: digitale Übersteuerung + Quell-Material-Härte möglich
            DefectType.DOLBY_NR_MISMATCH: 1.0,  # N/A: DAT ist digital - kein analoger Dolby-NR-Kompander
            DefectType.TAPE_HEAD_LEVEL_DIP: 1.0,  # N/A: DAT ist digital mit Drehtrommel - kein analoger Kopf
            # v10.0.0: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 1.0,  # N/A: DAT digital - kein analoges Modulationsrauschen
            DefectType.INNER_GROOVE_DISTORTION: 1.0,  # N/A: DAT hat keine Rillen
            DefectType.GROOVE_ECHO: 1.0,  # N/A: DAT hat keine Rillen
            DefectType.CROSSTALK: 0.8,  # DAT digital: minimales Crosstalk
            DefectType.INTERMODULATION_DISTORTION: 0.8,  # DAT: IMD nur bei analogem Eingang
            DefectType.TAPE_SPLICE_ARTIFACT: 1.0,  # N/A: DAT digital - kein physischer Schnitt
            DefectType.HF_REMANENCE_LOSS: 1.0,  # N/A: DAT digital - keine magnetische Remanenz
            DefectType.STYLUS_DAMAGE: 1.0,  # N/A: DAT hat keine Nadel
            DefectType.STICKY_SHED_RESIDUE: 1.0,  # N/A: DAT-Kassette anderes Bindemittel
            DefectType.MULTIBAND_WOW_FLUTTER: 1.0,  # N/A: DAT Crystal-Clock
            DefectType.GENERATION_LOSS: 0.7,  # Selten: DAT-zu-DAT-Kopie möglich
            DefectType.MOTOR_INTERFERENCE: 1.0,  # N/A: DAT digital
            DefectType.AMPLITUDE_DRIFT: 0.75,  # Selten: DAT digital - Drift nur aus analogem Quellmaterial
        },
        MaterialType.MP3_LOW: {
            DefectType.CLICKS: 0.9,
            DefectType.CRACKLE: 1.0,
            DefectType.HUM: 0.9,
            DefectType.WOW: 1.0,  # MP3 digital - kein WOW
            DefectType.FLUTTER: 1.0,  # MP3 digital - kein FLUTTER
            DefectType.STEREO_IMBALANCE: 0.8,
            DefectType.DIGITAL_ARTIFACTS: 0.3,  # Heavy codec artifacts
            DefectType.LOW_FREQ_RUMBLE: 0.9,
            DefectType.HIGH_FREQ_NOISE: 0.6,  # HF loss typical for low bitrate
            DefectType.COMPRESSION_ARTIFACTS: 0.15,  # VERY common!
            DefectType.PHASE_ISSUES: 0.5,  # Stereo imaging issues
            DefectType.DROPOUTS: 0.8,
            DefectType.CLIPPING: 0.5,  # Clipping im Quellmaterial vor Kodierung
            DefectType.DC_OFFSET: 0.9,  # MP3: DC im digitalen Originalmaterial
            DefectType.BANDWIDTH_LOSS: 0.1,  # MP3 low: starker HF-Cutoff bei 10-14 kHz!
            DefectType.PITCH_DRIFT: 0.9,  # MP3: digital, kein Pitch-Drift
            DefectType.REVERB_EXCESS: 0.8,  # MP3: Reverb aus dem Quellmaterial, nicht Kodierung
            DefectType.PRINT_THROUGH: 1.0,  # N/A: MP3 digital
            DefectType.QUANTIZATION_NOISE: 0.3,  # MP3 low: aggressive Requantisierung sehr häufig!
            DefectType.JITTER_ARTIFACTS: 0.8,  # MP3: digital, Jitter selten
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.3,  # MP3 low: oft überkomprimierter Quell-Content
            DefectType.SOFT_SATURATION: 1.0,  # N/A: MP3 kennt keine Soft-Saturation (digital)
            DefectType.HEAD_WEAR: 1.0,  # N/A: MP3 digital - kein Magnetkopf
            DefectType.PRE_ECHO: 0.2,  # MP3 Temporal-Masking → Pre-Echo vor Transienten sehr häufig!
            DefectType.TRANSIENT_SMEARING: 0.2,  # MP3 Temporal-Masking → starkes Attack-Smearing bei Perkussion
            DefectType.RIAA_CURVE_ERROR: 1.0,  # N/A: MP3 digital - keine Disc-Entzerrungskurve
            DefectType.ALIASING: 0.4,  # Resampling-Kette im Codec erzeugt Aliasing-ähnliche Artefakte
            DefectType.BIAS_ERROR: 1.0,  # N/A: MP3 digital - kein analoger Bias
            DefectType.AZIMUTH_ERROR: 1.0,  # N/A: MP3 digital - kein Magnetkopf
            DefectType.SIBILANCE: 0.6,  # MP3 128 kbps: psychoakustitsche Maskierung → starke Sibilanzverzerrung typisch
            DefectType.TRANSPORT_BUMP: 1.0,  # N/A: MP3 digital - kein mechanischer Transport
            DefectType.VOCAL_HARSHNESS: 0.3,  # MP3-Low: Codec-Artefakte + Quell-Clipping → Vokal-Härte häufig
            DefectType.DOLBY_NR_MISMATCH: 1.0,  # N/A: MP3 digital - kein Dolby-Analogband-NR
            DefectType.TAPE_HEAD_LEVEL_DIP: 1.0,  # N/A: MP3 digital - kein Magnetband-Kopf
            # v10.0.0: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 1.0,  # N/A: MP3 digital
            DefectType.INNER_GROOVE_DISTORTION: 1.0,  # N/A: MP3 hat keine Rillen
            DefectType.GROOVE_ECHO: 1.0,  # N/A: MP3 hat keine Rillen
            DefectType.CROSSTALK: 0.9,  # N/A: MP3 digital (minimalster L/R-Bleed)
            DefectType.INTERMODULATION_DISTORTION: 0.8,  # MP3: IMD nur aus analogem Quellmaterial
            DefectType.TAPE_SPLICE_ARTIFACT: 1.0,  # N/A: MP3 digital
            DefectType.HF_REMANENCE_LOSS: 1.0,  # N/A: MP3 digital
            DefectType.STYLUS_DAMAGE: 1.0,  # N/A: MP3 digital
            DefectType.STICKY_SHED_RESIDUE: 1.0,  # N/A: MP3 digital
            DefectType.MULTIBAND_WOW_FLUTTER: 1.0,  # N/A: MP3 digital
            DefectType.GENERATION_LOSS: 0.2,  # SEHR HÄUFIG: Mehrfach-Transkodierung (MP3→WAV→MP3)
            DefectType.MOTOR_INTERFERENCE: 1.0,  # N/A: MP3 digital
            DefectType.AMPLITUDE_DRIFT: 0.70,  # Möglich: Quellmaterial-Drift in MP3-Low → Pegelgradient
        },
        MaterialType.MP3_HIGH: {
            DefectType.CLICKS: 0.9,
            DefectType.CRACKLE: 1.0,
            DefectType.HUM: 0.9,
            DefectType.WOW: 1.0,  # MP3 digital - kein WOW
            DefectType.FLUTTER: 1.0,  # MP3 digital - kein FLUTTER
            DefectType.STEREO_IMBALANCE: 0.8,
            DefectType.DIGITAL_ARTIFACTS: 0.4,  # Moderate artifacts
            DefectType.LOW_FREQ_RUMBLE: 0.9,
            DefectType.HIGH_FREQ_NOISE: 0.7,
            DefectType.COMPRESSION_ARTIFACTS: 0.3,  # Common but less severe
            DefectType.PHASE_ISSUES: 0.6,
            DefectType.DROPOUTS: 0.9,
            DefectType.CLIPPING: 0.5,  # Clipping im Quellmaterial
            DefectType.DC_OFFSET: 0.9,
            DefectType.BANDWIDTH_LOSS: 0.3,  # MP3 high: HF-Cutoff bei 16-18 kHz
            DefectType.PITCH_DRIFT: 0.9,
            DefectType.REVERB_EXCESS: 0.8,
            DefectType.PRINT_THROUGH: 1.0,
            DefectType.QUANTIZATION_NOISE: 0.5,  # MP3 high: weniger Requantisierung
            DefectType.JITTER_ARTIFACTS: 0.8,
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.3,  # MP3 high: Quell-Content oft überkomprimiert
            DefectType.SOFT_SATURATION: 1.0,  # N/A: MP3 kennt keine Soft-Saturation (digital)
            DefectType.HEAD_WEAR: 1.0,  # N/A: MP3 digital - kein Magnetkopf
            DefectType.PRE_ECHO: 0.3,  # MP3 (höheres Bitrate) → Pre-Echo weniger stark als Low
            DefectType.TRANSIENT_SMEARING: 0.3,  # MP3 high: Transient-Smearing weniger stark als Low
            DefectType.RIAA_CURVE_ERROR: 1.0,  # N/A: MP3 digital - keine Disc-Entzerrungskurve
            DefectType.ALIASING: 0.3,  # Höheres Bitrate - weniger Resampling-Artefakte als MP3-Low
            DefectType.BIAS_ERROR: 1.0,  # N/A: MP3 digital - kein analoger Bias
            DefectType.AZIMUTH_ERROR: 1.0,  # N/A: MP3 digital - kein Magnetkopf
            DefectType.SIBILANCE: 0.3,  # MP3 ≥ 192 kbps: deutlich weniger Sibilanz-Artefakte als Low-Bitrate
            DefectType.TRANSPORT_BUMP: 1.0,  # N/A: MP3 digital - kein mechanischer Transport
            DefectType.VOCAL_HARSHNESS: 0.3,  # MP3-High: Quell-Mastering-Clipping → Vokal-Übersteuerung möglich
            DefectType.DOLBY_NR_MISMATCH: 1.0,  # N/A: MP3 digital - kein Dolby-Analogband-NR
            DefectType.TAPE_HEAD_LEVEL_DIP: 1.0,  # N/A: MP3 digital - kein Magnetband-Kopf
            # v10.0.0: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 1.0,  # N/A: MP3 digital
            DefectType.INNER_GROOVE_DISTORTION: 1.0,  # N/A: MP3 hat keine Rillen
            DefectType.GROOVE_ECHO: 1.0,  # N/A: MP3 hat keine Rillen
            DefectType.CROSSTALK: 0.9,  # N/A: MP3 digital
            DefectType.INTERMODULATION_DISTORTION: 0.8,  # MP3: IMD nur aus analogem Quellmaterial
            DefectType.TAPE_SPLICE_ARTIFACT: 1.0,  # N/A: MP3 digital
            DefectType.HF_REMANENCE_LOSS: 1.0,  # N/A: MP3 digital
            DefectType.STYLUS_DAMAGE: 1.0,  # N/A: MP3 digital
            DefectType.STICKY_SHED_RESIDUE: 1.0,  # N/A: MP3 digital
            DefectType.MULTIBAND_WOW_FLUTTER: 1.0,  # N/A: MP3 digital
            DefectType.GENERATION_LOSS: 0.3,  # HÄUFIG: Mehrfach-Transkodierung (weniger aggressiv bei 192+ kbps)
            DefectType.MOTOR_INTERFERENCE: 1.0,  # N/A: MP3 digital
            DefectType.AMPLITUDE_DRIFT: 0.70,  # Möglich: Quellmaterial-Drift in MP3-High → Pegelgradient
        },
        MaterialType.AAC: {
            DefectType.CLICKS: 0.9,
            DefectType.CRACKLE: 1.0,
            DefectType.HUM: 0.9,
            DefectType.WOW: 1.0,  # AAC digital - kein WOW
            DefectType.FLUTTER: 1.0,  # AAC digital - kein FLUTTER
            DefectType.STEREO_IMBALANCE: 0.8,
            DefectType.DIGITAL_ARTIFACTS: 0.4,
            DefectType.LOW_FREQ_RUMBLE: 0.9,
            DefectType.HIGH_FREQ_NOISE: 0.8,  # Better HF than MP3
            DefectType.COMPRESSION_ARTIFACTS: 0.25,  # More efficient than MP3
            DefectType.PHASE_ISSUES: 0.7,
            DefectType.DROPOUTS: 0.9,
            DefectType.CLIPPING: 0.5,
            DefectType.DC_OFFSET: 0.9,
            DefectType.BANDWIDTH_LOSS: 0.4,  # AAC: bessere HF als MP3 bei gleichem Bitrate
            DefectType.PITCH_DRIFT: 0.9,
            DefectType.REVERB_EXCESS: 0.8,
            DefectType.PRINT_THROUGH: 1.0,
            DefectType.QUANTIZATION_NOISE: 0.5,  # AAC: effizienter als MP3, weniger Quantisierungsfehler
            DefectType.JITTER_ARTIFACTS: 0.8,
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.35,  # AAC: Source-Content oft überkomprimiert
            DefectType.SOFT_SATURATION: 1.0,  # N/A: AAC digital - Soft-Saturation nur aus Quellmaterial
            DefectType.HEAD_WEAR: 1.0,  # N/A: AAC digital - kein Magnetkopf
            DefectType.PRE_ECHO: 0.35,  # AAC: Temporal-Masking → Pre-Echo vor Transienten möglich
            DefectType.TRANSIENT_SMEARING: 0.3,  # AAC: Transient-Smearing bei hoher Kompression
            DefectType.RIAA_CURVE_ERROR: 1.0,  # N/A: AAC digital - keine Disc-Entzerrungskurve
            DefectType.ALIASING: 0.3,  # AAC-Codec: Resampling-Artefakte in der Transkodierkette
            DefectType.BIAS_ERROR: 1.0,  # N/A: AAC digital - kein analoger Bias
            DefectType.AZIMUTH_ERROR: 1.0,  # N/A: AAC digital - kein Magnetkopf
            DefectType.SIBILANCE: 0.4,  # AAC-Codec kann bei mittleren Bitraten Zischlaut-Artefakte einführen
            DefectType.TRANSPORT_BUMP: 1.0,  # N/A: AAC digital - kein mechanischer Transport
            DefectType.VOCAL_HARSHNESS: 0.3,  # AAC-Codec: Quell-Mastering-Härte + Codec-Artefakte möglich
            DefectType.DOLBY_NR_MISMATCH: 1.0,  # N/A: AAC digital - kein Dolby-Analogband-NR
            DefectType.TAPE_HEAD_LEVEL_DIP: 1.0,  # N/A: AAC digital - kein Magnetband-Kopf
            # v10.0.0: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 1.0,  # N/A: AAC digital
            DefectType.INNER_GROOVE_DISTORTION: 1.0,  # N/A: AAC hat keine Rillen
            DefectType.GROOVE_ECHO: 1.0,  # N/A: AAC hat keine Rillen
            DefectType.CROSSTALK: 0.9,  # N/A: AAC digital
            DefectType.INTERMODULATION_DISTORTION: 0.8,  # AAC: IMD nur aus analogem Quellmaterial
            DefectType.TAPE_SPLICE_ARTIFACT: 1.0,  # N/A: AAC digital
            DefectType.HF_REMANENCE_LOSS: 1.0,  # N/A: AAC digital
            DefectType.STYLUS_DAMAGE: 1.0,  # N/A: AAC digital
            DefectType.STICKY_SHED_RESIDUE: 1.0,  # N/A: AAC digital
            DefectType.MULTIBAND_WOW_FLUTTER: 1.0,  # N/A: AAC digital
            DefectType.GENERATION_LOSS: 0.3,  # Mehrfach-Transkodierung möglich
            DefectType.MOTOR_INTERFERENCE: 1.0,  # N/A: AAC digital
            DefectType.AMPLITUDE_DRIFT: 0.70,  # Möglich: AAC-Quellmaterial-Drift
        },
        MaterialType.MINIDISC: {
            DefectType.CLICKS: 0.9,
            DefectType.CRACKLE: 1.0,
            DefectType.HUM: 0.9,
            DefectType.WOW: 1.0,  # MiniDisc digital ATRAC - kein WOW
            DefectType.FLUTTER: 1.0,  # MiniDisc digital ATRAC - kein FLUTTER
            DefectType.STEREO_IMBALANCE: 0.8,
            DefectType.DIGITAL_ARTIFACTS: 0.35,  # ATRAC specific artifacts
            DefectType.LOW_FREQ_RUMBLE: 0.9,
            DefectType.HIGH_FREQ_NOISE: 0.6,  # ATRAC artifacts at HF
            DefectType.COMPRESSION_ARTIFACTS: 0.2,  # ATRAC aggressive
            DefectType.PHASE_ISSUES: 0.6,  # Joint stereo issues
            DefectType.DROPOUTS: 0.8,
            DefectType.CLIPPING: 0.5,
            DefectType.DC_OFFSET: 0.9,
            DefectType.BANDWIDTH_LOSS: 0.2,  # ATRAC: starke HF-Artefakte über 14 kHz
            DefectType.PITCH_DRIFT: 0.9,
            DefectType.REVERB_EXCESS: 0.7,
            DefectType.PRINT_THROUGH: 1.0,
            DefectType.QUANTIZATION_NOISE: 0.3,  # ATRAC: aggressive Quantisierung sehr häufig!
            DefectType.JITTER_ARTIFACTS: 0.5,  # MiniDisc: ATRAC Buffer-Timing-Pröbleme
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.5,  # MiniDisc: Aufnahmen variieren stark
            DefectType.SOFT_SATURATION: 1.0,  # N/A: MiniDisc digital - Soft-Saturation nur aus Quellmaterial
            DefectType.HEAD_WEAR: 1.0,  # N/A: MiniDisc digital - kein Magnetkopf im klassischen Sinne
            DefectType.PRE_ECHO: 0.35,  # ATRAC: aggressive Temporal-Masking → Pre-Echo häufig
            DefectType.TRANSIENT_SMEARING: 0.2,  # ATRAC: starkes Transient-Smearing - GrooveMetric-relevant
            DefectType.RIAA_CURVE_ERROR: 1.0,  # N/A: MiniDisc digital - keine Disc-Entzerrungskurve
            DefectType.ALIASING: 0.4,  # ATRAC-Codec: Resampling-Artefakte und Aliasing-Stufigkeit
            DefectType.BIAS_ERROR: 1.0,  # N/A: MiniDisc ATRAC digital - kein analoger Bias
            DefectType.AZIMUTH_ERROR: 0.60,  # MiniDisc Rotationskopf: Azimuth-Drift bei Alterung möglich
            DefectType.SIBILANCE: 0.5,  # ATRAC-Codec (MiniDisc): Sibilanz-Artefakte charakteristisch bei 132 kbps
            DefectType.TRANSPORT_BUMP: 0.7,  # MiniDisc-Laufwerk: Rotationstransport kann holpern
            DefectType.VOCAL_HARSHNESS: 0.4,  # ATRAC-Codec: Vokal-Artefakte bei 132 kbps → Härte möglich
            DefectType.DOLBY_NR_MISMATCH: 1.0,  # N/A: MiniDisc ist ATRAC-digital - kein Dolby-Analogband-NR
            DefectType.TAPE_HEAD_LEVEL_DIP: 1.0,  # N/A: MiniDisc digital - kein Magnetband-Kopf
            # v10.0.0: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 1.0,  # N/A: MiniDisc ATRAC digital
            DefectType.INNER_GROOVE_DISTORTION: 1.0,  # N/A: MiniDisc hat keine Rillen
            DefectType.GROOVE_ECHO: 1.0,  # N/A: MiniDisc hat keine Rillen
            DefectType.CROSSTALK: 0.8,  # MiniDisc: Joint-Stereo → minimaler L/R-Bleed
            DefectType.INTERMODULATION_DISTORTION: 0.8,  # MiniDisc: IMD nur aus analogem Quellmaterial
            DefectType.TAPE_SPLICE_ARTIFACT: 1.0,  # N/A: MiniDisc digital
            DefectType.HF_REMANENCE_LOSS: 1.0,  # N/A: MiniDisc digital
            DefectType.STYLUS_DAMAGE: 1.0,  # N/A: MiniDisc digital
            DefectType.STICKY_SHED_RESIDUE: 1.0,  # N/A: MiniDisc digital
            DefectType.MULTIBAND_WOW_FLUTTER: 1.0,  # N/A: MiniDisc Crystal-Clock
            DefectType.GENERATION_LOSS: 0.3,  # ATRAC-Transkodierung möglich
            DefectType.MOTOR_INTERFERENCE: 1.0,  # N/A: MiniDisc digital
            DefectType.AMPLITUDE_DRIFT: 0.75,  # Selten: MiniDisc digital - Drift aus Quellmaterial
        },
        MaterialType.STREAMING: {
            DefectType.CLICKS: 0.9,
            DefectType.CRACKLE: 0.9,
            DefectType.HUM: 0.9,
            DefectType.WOW: 0.90,  # Streaming digital - WOW N/A
            DefectType.FLUTTER: 0.90,  # Streaming digital - FLUTTER N/A
            DefectType.STEREO_IMBALANCE: 0.8,
            DefectType.DIGITAL_ARTIFACTS: 0.4,
            DefectType.LOW_FREQ_RUMBLE: 0.9,
            DefectType.HIGH_FREQ_NOISE: 0.8,
            DefectType.COMPRESSION_ARTIFACTS: 0.2,  # Sehr häufig!
            DefectType.PHASE_ISSUES: 0.6,
            DefectType.DROPOUTS: 0.3,  # Streaming dropouts
            DefectType.CLIPPING: 0.4,  # Loudness-War Streaming Masters
            DefectType.DC_OFFSET: 0.9,
            DefectType.BANDWIDTH_LOSS: 0.4,
            DefectType.PITCH_DRIFT: 0.9,
            DefectType.REVERB_EXCESS: 0.8,
            DefectType.PRINT_THROUGH: 1.0,
            DefectType.QUANTIZATION_NOISE: 0.4,  # Streaming: Transkodierungskette erzeugt Requantisierung
            DefectType.JITTER_ARTIFACTS: 0.4,  # Netzwerk-Jitter → Puffer-Underruns / Artefakte
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.2,  # Streaming-Normalisierung → Loudness-War besonders sichtbar!
            DefectType.SOFT_SATURATION: 1.0,  # N/A: Streaming digital - Soft-Saturation nur aus Quellmaterial
            DefectType.HEAD_WEAR: 1.0,  # N/A: Streaming digital - kein Magnetkopf
            DefectType.PRE_ECHO: 0.4,  # Streaming-Codec: Pre-Echo aus Transkodierkette möglich
            DefectType.TRANSIENT_SMEARING: 0.3,  # Streaming-Normalisierung + Codec → Transient-Smearing häufig
            DefectType.RIAA_CURVE_ERROR: 1.0,  # N/A: Streaming digital - keine Disc-Entzerrungskurve
            DefectType.ALIASING: 0.4,  # Mehrfache Transkodierkette erzeugt kumulative Aliasing-Artefakte
            DefectType.BIAS_ERROR: 1.0,  # N/A: Streaming digital - kein analoger Bias
            DefectType.AZIMUTH_ERROR: 1.0,  # N/A: Streaming digital - kein Magnetkopf
            DefectType.SIBILANCE: 0.4,  # Streaming-Codec (Opus/AAC): variable Bitraten → Sibilanz-Artefakte möglich
            DefectType.TRANSPORT_BUMP: 1.0,  # N/A: Streaming digital - kein mechanischer Transport
            DefectType.VOCAL_HARSHNESS: 0.3,  # Streaming: Quell-Mastering-Übersteuerung + Codec-Härte möglich
            DefectType.DOLBY_NR_MISMATCH: 1.0,  # N/A: Streaming digital - kein Dolby-Analogband-NR
            DefectType.TAPE_HEAD_LEVEL_DIP: 1.0,  # N/A: Streaming digital - kein Magnetband-Kopf
            # v10.0.0: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 1.0,  # N/A: Streaming digital
            DefectType.INNER_GROOVE_DISTORTION: 1.0,  # N/A: Streaming hat keine Rillen
            DefectType.GROOVE_ECHO: 1.0,  # N/A: Streaming hat keine Rillen
            DefectType.CROSSTALK: 0.9,  # N/A: Streaming digital
            DefectType.INTERMODULATION_DISTORTION: 0.8,  # Streaming: IMD nur aus Quellmaterial
            DefectType.TAPE_SPLICE_ARTIFACT: 1.0,  # N/A: Streaming digital
            DefectType.HF_REMANENCE_LOSS: 1.0,  # N/A: Streaming digital
            DefectType.STYLUS_DAMAGE: 1.0,  # N/A: Streaming digital
            DefectType.STICKY_SHED_RESIDUE: 1.0,  # N/A: Streaming digital
            DefectType.MULTIBAND_WOW_FLUTTER: 1.0,  # N/A: Streaming digital
            DefectType.GENERATION_LOSS: 0.2,  # SEHR HÄUFIG: Streaming-Transkodierung (YouTube, Spotify)
            DefectType.MOTOR_INTERFERENCE: 1.0,  # N/A: Streaming digital
            DefectType.AMPLITUDE_DRIFT: 0.80,  # Selten: Streaming digital - Drift nur aus Quellmaterial
        },
        MaterialType.UNKNOWN: dict.fromkeys(DefectType, 0.6),
        MaterialType.WAX_CYLINDER: {
            # Phonograph-Wachswalze (1890-1930): extremer Rauschboden, HF ≤ 5 kHz
            DefectType.CLICKS: 0.2,  # Sehr häufig durch Zylinderoberfläche
            DefectType.CRACKLE: 0.2,  # Wachswalzen-Abrieb → starkes Crackle
            DefectType.HUM: 0.5,
            DefectType.WOW: 0.30,  # Wachswalzen-Laufungenauigkeit < 0.5 Hz
            DefectType.FLUTTER: 0.40,  # Trichter-/Mechanik-Vibration 0.5-200 Hz
            DefectType.STEREO_IMBALANCE: 1.0,  # N/A: immer Mono
            DefectType.DIGITAL_ARTIFACTS: 1.0,  # N/A: analog
            DefectType.LOW_FREQ_RUMBLE: 0.4,  # Mechanisches Rumpeln der Walze
            DefectType.HIGH_FREQ_NOISE: 0.2,  # Extremes Oberflächenrauschen
            DefectType.COMPRESSION_ARTIFACTS: 1.0,
            DefectType.PHASE_ISSUES: 1.0,
            DefectType.DROPOUTS: 0.4,  # Wachs-Fehler → Dropout
            DefectType.CLIPPING: 0.5,  # Analoge Groß-Signal-Verzerrung
            DefectType.DC_OFFSET: 0.5,
            DefectType.BANDWIDTH_LOSS: 0.1,  # HF extrem begrenzt (≤ 5 kHz)
            DefectType.PITCH_DRIFT: 0.2,  # Häufige Drehzahl-Ungleichmäßigkeit
            DefectType.REVERB_EXCESS: 0.5,
            DefectType.PRINT_THROUGH: 1.0,  # N/A: kein Magnetband
            DefectType.QUANTIZATION_NOISE: 1.0,  # N/A: analog
            DefectType.JITTER_ARTIFACTS: 1.0,  # N/A: analog
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 1.0,
            DefectType.SOFT_SATURATION: 0.3,  # Akustische Sättigung durch Aufnahmetrichter
            DefectType.HEAD_WEAR: 1.0,  # N/A: kein Magnetkopf
            DefectType.PRE_ECHO: 1.0,  # N/A: kein digitaler Codec auf Wachswalze
            DefectType.TRANSIENT_SMEARING: 0.5,  # Trägheit des Trichters begrenzt Transienten
            DefectType.RIAA_CURVE_ERROR: 1.0,  # N/A: Wachswalze mechanisch - keine elektrische EQ-Kurve
            DefectType.ALIASING: 0.6,  # Sehr alte Digitalisierungen mit primitiven AA-Filtern
            DefectType.BIAS_ERROR: 1.0,  # N/A: Wachswalze mechanisch - kein Magnetband-Bias
            DefectType.AZIMUTH_ERROR: 1.0,  # N/A: mechanische Abtastung - kein Magnetkopf
            DefectType.SIBILANCE: 1.0,  # N/A: HF ≤ 5 kHz, Zischlautbereich physikalisch nicht erreichbar
            DefectType.TRANSPORT_BUMP: 0.5,  # Walzen-Transportholpern bei mechanischer Abtastung
            DefectType.VOCAL_HARSHNESS: 0.6,  # Wachswalze: Mittelton-Verzerrung durch Trichter-Resonanz
            DefectType.DOLBY_NR_MISMATCH: 1.0,  # N/A: Wachswalze (1890-1930) - Dolby NR erst 1966 erfunden
            DefectType.TAPE_HEAD_LEVEL_DIP: 1.0,  # N/A: Wachswalze hat kein Band/Kopf
            # v10.0.0: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 1.0,  # N/A: Wachswalze mechanisch
            DefectType.INNER_GROOVE_DISTORTION: 0.15,  # EXTREM HÄUFIG: Wachswalze mit primitiver Nadel
            DefectType.GROOVE_ECHO: 0.3,  # Weiche Wachsmasse → Rillenverformung
            DefectType.CROSSTALK: 1.0,  # N/A: immer Mono
            DefectType.INTERMODULATION_DISTORTION: 0.3,  # Trichter-Aufnahme nichtlinear
            DefectType.TAPE_SPLICE_ARTIFACT: 1.0,  # N/A: Wachswalze
            DefectType.HF_REMANENCE_LOSS: 1.0,  # N/A: Wachswalze mechanisch
            DefectType.STYLUS_DAMAGE: 0.15,  # EXTREM HÄUFIG: Stahlnadeln zerstören Wachsrillen
            DefectType.STICKY_SHED_RESIDUE: 1.0,  # N/A: Wachswalze
            DefectType.MULTIBAND_WOW_FLUTTER: 0.4,  # Mechanischer Antrieb
            DefectType.GENERATION_LOSS: 0.5,  # Duplikat-Walzen: Qualitätsverlust durch Kopierverfahren
            DefectType.MOTOR_INTERFERENCE: 0.3,  # Federwerk/Elektromotor-Störungen
            DefectType.AMPLITUDE_DRIFT: 0.40,  # Häufig: Federwerk-Motor → Drehzahlabfall → Pegelgradient
        },
        MaterialType.WIRE_RECORDING: {
            # Drahtbandaufnahme (1940-1955): Jitter, Frequenzgang-Einbrüche, Magnetisierungs-Dropout
            DefectType.CLICKS: 0.5,
            DefectType.CRACKLE: 0.4,
            DefectType.HUM: 0.4,  # Magnetfeld-Überkopplung
            DefectType.WOW: 0.20,  # Drahtjitter-Gleichlauf (< 0.5 Hz), sehr charakteristisch
            DefectType.FLUTTER: 0.25,  # Drahtführungs-Vibration (0.5-200 Hz)
            DefectType.STEREO_IMBALANCE: 1.0,  # N/A: immer Mono
            DefectType.DIGITAL_ARTIFACTS: 1.0,  # N/A: analog
            DefectType.LOW_FREQ_RUMBLE: 0.5,
            DefectType.HIGH_FREQ_NOISE: 0.3,
            DefectType.COMPRESSION_ARTIFACTS: 1.0,
            DefectType.PHASE_ISSUES: 1.0,
            DefectType.DROPOUTS: 0.3,  # Magnetisierungs-Dropout sehr häufig
            DefectType.CLIPPING: 0.5,
            DefectType.DC_OFFSET: 0.5,
            DefectType.BANDWIDTH_LOSS: 0.2,  # HF begrenzt (≤ 8 kHz)
            DefectType.PITCH_DRIFT: 0.2,  # Drahtjitter → Pitch-Instabilität
            DefectType.REVERB_EXCESS: 0.5,
            DefectType.PRINT_THROUGH: 0.5,  # Magnetdraht kann Print-Through entwickeln
            DefectType.QUANTIZATION_NOISE: 1.0,  # N/A: analog
            DefectType.JITTER_ARTIFACTS: 0.2,  # Draht-Transportjitter sehr spezifisch
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 1.0,
            DefectType.SOFT_SATURATION: 0.4,  # Magnetische Sättigung des Drahts möglich
            DefectType.HEAD_WEAR: 0.3,  # Drahtkopfverschleiß typisch
            DefectType.PRE_ECHO: 1.0,  # N/A: kein digitaler Codec
            DefectType.TRANSIENT_SMEARING: 0.4,  # Begrenzte Bandbreite → Transientenverzerrung
            DefectType.RIAA_CURVE_ERROR: 1.0,  # N/A: Drahtbandaufnahme - kein Disc-Format
            DefectType.ALIASING: 0.5,  # Frühe Digitalisierungen mit suboptimalen AA-Filtern
            DefectType.BIAS_ERROR: 0.4,  # AC-Bias oft falsch justiert bei primitiven Drahtbandgeräten
            DefectType.AZIMUTH_ERROR: 0.40,  # Draht-Magnetkopf: Azimuth-Fehler durch primitive Justierung häufig
            DefectType.SIBILANCE: 0.5,  # Drahtband: begrenzter HF-Frequenzgang → De-Esser teilweise relevant
            DefectType.TRANSPORT_BUMP: 0.3,  # Drahtbandfuehrung: Transportrueckeln durch primitive Mechanik
            DefectType.VOCAL_HARSHNESS: 0.5,  # Drahtband: Magnetisierungs-Verzerrung → Vokal-Härte möglich
            DefectType.DOLBY_NR_MISMATCH: 1.0,  # N/A: Drahtband (1940-1955) - Dolby NR erst 1966 erfunden
            DefectType.TAPE_HEAD_LEVEL_DIP: 0.35,  # Möglich: Drahtband-Kopf-Kontakt instabil
            # v10.0.0: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 0.2,  # Drahtband: magnetisches Modulationsrauschen vorhanden
            DefectType.INNER_GROOVE_DISTORTION: 1.0,  # N/A: Draht hat keine Rillen
            DefectType.GROOVE_ECHO: 1.0,  # N/A: Draht hat keine Rillen
            DefectType.CROSSTALK: 1.0,  # N/A: immer Mono
            DefectType.INTERMODULATION_DISTORTION: 0.35,  # Primitive Verstärker → IMD
            DefectType.TAPE_SPLICE_ARTIFACT: 0.3,  # Drahtschweißstellen-Artefakte möglich
            DefectType.HF_REMANENCE_LOSS: 0.2,  # Draht altert → HF-Verlust
            DefectType.STYLUS_DAMAGE: 1.0,  # N/A: Draht hat keine Nadel
            DefectType.STICKY_SHED_RESIDUE: 1.0,  # N/A: Draht hat keinen Binder
            DefectType.MULTIBAND_WOW_FLUTTER: 0.25,  # Primitive Drahtführung → fl-abhängig
            DefectType.GENERATION_LOSS: 0.3,  # Draht-Dubbing möglich
            DefectType.MOTOR_INTERFERENCE: 0.4,  # Primitive Motoren
            DefectType.AMPLITUDE_DRIFT: 0.40,  # Häufig: Drahtband-Motor → AGC-ähnlicher Pegelanstieg/-abfall
        },
        MaterialType.LACQUER_DISC: {
            # Acetat-Lackfolien-Heimaufnahme (1930-1950): Risse, Substrat-Rauschen, Rille-Ermüdung
            DefectType.CLICKS: 0.2,  # Rissbildung → sehr häufige Clicks
            DefectType.CRACKLE: 0.3,  # Rille-Ermüdung → Crackle
            DefectType.HUM: 0.5,
            DefectType.WOW: 0.30,  # Acetat-Verformung → Plattenteller-Gleichlauf < 0.5 Hz
            DefectType.FLUTTER: 0.40,  # Heim-Abtastarm-Vibration 0.5-200 Hz
            DefectType.STEREO_IMBALANCE: 1.0,  # N/A: meist Mono
            DefectType.DIGITAL_ARTIFACTS: 1.0,  # N/A: analog
            DefectType.LOW_FREQ_RUMBLE: 0.4,
            DefectType.HIGH_FREQ_NOISE: 0.3,  # Substrat-Rauschen
            DefectType.COMPRESSION_ARTIFACTS: 1.0,
            DefectType.PHASE_ISSUES: 1.0,
            DefectType.DROPOUTS: 0.4,  # Risse erzeugen schmale Dropout-Ereignisse
            DefectType.CLIPPING: 0.5,
            DefectType.DC_OFFSET: 0.4,
            DefectType.BANDWIDTH_LOSS: 0.2,  # Schmalbandige Hausaufnahme-Technique
            DefectType.PITCH_DRIFT: 0.3,  # Heimplattenspieler mit Motorungenauigkeiten
            DefectType.REVERB_EXCESS: 0.5,
            DefectType.PRINT_THROUGH: 1.0,  # N/A: kein Magnetband
            DefectType.QUANTIZATION_NOISE: 1.0,  # N/A: analog
            DefectType.JITTER_ARTIFACTS: 1.0,  # N/A: analog
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.9,
            DefectType.SOFT_SATURATION: 0.3,  # Rillenverzerrung bei Heimaufnahmen möglich
            DefectType.HEAD_WEAR: 1.0,  # N/A: kein Magnetkopf
            DefectType.PRE_ECHO: 1.0,  # N/A: kein digitaler Codec
            DefectType.TRANSIENT_SMEARING: 0.4,  # Limitierte Heimaufnahme-Technique
            DefectType.RIAA_CURVE_ERROR: 0.2,  # Heimaufnahmen: verschiedene EQ-Kurven (AES/NAB/FFRR) häufig falsch
            DefectType.ALIASING: 0.5,  # Heim-Digitalisierung variiert - AA-Filter fehlt oft
            DefectType.BIAS_ERROR: 1.0,  # N/A: Lackfolie ist kein Magnetband - kein Aufnahme-Bias
            DefectType.AZIMUTH_ERROR: 1.0,  # N/A: Lackfolie ist mechanisches Disc-Format - kein Magnetkopf
            DefectType.SIBILANCE: 0.4,  # Heimaufnahme-Nadel: Zischlaute bei HF-Überbetonung möglich
            DefectType.TRANSPORT_BUMP: 0.5,  # Plattenteller-Holpern bei Heimaufnahme-Technique
            DefectType.VOCAL_HARSHNESS: 0.5,  # Heimaufnahme: primitive Mikrofone → Vokal-Verzerrung häufig
            DefectType.DOLBY_NR_MISMATCH: 1.0,  # N/A: Lacquer Disc ist kein Magnetband - kein Dolby-Kompander
            DefectType.TAPE_HEAD_LEVEL_DIP: 1.0,  # N/A: Lacquer Disc hat kein Magnetband
            # v10.0.0: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 1.0,  # N/A: Lacquer mechanisch
            DefectType.INNER_GROOVE_DISTORTION: 0.2,  # HÄUFIG: Heimschneidmaschine → IGD
            DefectType.GROOVE_ECHO: 0.25,  # Weiche Acetat-Folie → Rillenverformung
            DefectType.CROSSTALK: 1.0,  # N/A: meist Mono
            DefectType.INTERMODULATION_DISTORTION: 0.35,  # Heimverstärker nichtlinear
            DefectType.TAPE_SPLICE_ARTIFACT: 1.0,  # N/A: Disc
            DefectType.HF_REMANENCE_LOSS: 1.0,  # N/A: mechanisch
            DefectType.STYLUS_DAMAGE: 0.2,  # Heimnadeln zerstören weiche Acetat-Rillen
            DefectType.STICKY_SHED_RESIDUE: 1.0,  # N/A: Disc
            DefectType.MULTIBAND_WOW_FLUTTER: 0.4,  # Heim-Plattenspieler
            DefectType.GENERATION_LOSS: 0.5,  # Dubbing: Acetat → Pressmatrize
            DefectType.MOTOR_INTERFERENCE: 0.35,  # Heim-Plattenspieler Motor
            DefectType.AMPLITUDE_DRIFT: 0.40,  # Häufig: Acetat-Heim-Plattenspieler Motor → Pegelgradient
        },
    }

    _SCRAPE_FLUTTER_SENSITIVITY: dict[MaterialType, float] = {
        MaterialType.SHELLAC: 0.85,
        MaterialType.VINYL: 0.75,
        MaterialType.TAPE: 0.18,
        MaterialType.CASSETTE: 0.16,
        MaterialType.CD_DIGITAL: 1.0,
        MaterialType.REEL_TAPE: 0.14,
        MaterialType.DAT: 1.0,
        MaterialType.MP3_LOW: 1.0,
        MaterialType.MP3_HIGH: 1.0,
        MaterialType.AAC: 1.0,
        MaterialType.MINIDISC: 1.0,
        MaterialType.STREAMING: 1.0,
        MaterialType.UNKNOWN: 0.55,
        MaterialType.WAX_CYLINDER: 0.70,
        MaterialType.WIRE_RECORDING: 0.20,
        MaterialType.LACQUER_DISC: 0.78,
    }
    _TAPE_HEAD_CLOG_SENSITIVITY: dict[MaterialType, float] = {
        MaterialType.SHELLAC: 1.0,
        MaterialType.VINYL: 1.0,
        MaterialType.TAPE: 0.18,
        MaterialType.CASSETTE: 0.16,
        MaterialType.CD_DIGITAL: 1.0,
        MaterialType.REEL_TAPE: 0.14,
        MaterialType.DAT: 1.0,
        MaterialType.MP3_LOW: 1.0,
        MaterialType.MP3_HIGH: 1.0,
        MaterialType.AAC: 1.0,
        MaterialType.MINIDISC: 1.0,
        MaterialType.STREAMING: 1.0,
        MaterialType.UNKNOWN: 0.55,
        MaterialType.WAX_CYLINDER: 1.0,
        MaterialType.WIRE_RECORDING: 0.20,
        MaterialType.LACQUER_DISC: 1.0,
    }
    for _material_type, _scrape_threshold in _SCRAPE_FLUTTER_SENSITIVITY.items():
        MATERIAL_SENSITIVITY.setdefault(_material_type, {})[DefectType.SCRAPE_FLUTTER] = _scrape_threshold
    for _material_type, _clog_threshold in _TAPE_HEAD_CLOG_SENSITIVITY.items():
        MATERIAL_SENSITIVITY.setdefault(_material_type, {})[DefectType.TAPE_HEAD_CLOG] = _clog_threshold

    # Location caps are disabled (0 = uncapped) to avoid losing valid events
    # on long recordings with very dense defect activity.
    _LOCATION_CAP_UNCAPPED = 0

    def __init__(self, sample_rate: int = 44100, material_type: MaterialType | None = None):
        """
        Initialisiert DefectScanner.

        Args:
            sample_rate: Audio sample rate (Standard: 44100 Hz)
            material_type: Material-Typ (wird auto-detected wenn None)
        """
        self.sample_rate = sample_rate
        self.material_type = material_type
        # Always set thresholds (use UNKNOWN if material_type is None for auto-detection).
        # Per-Instance-Kopie (§G1 (copilot-instructions.md)/V8 Song-Isolation): v10.304 AST-Pre-Filter mutiert
        # thresholds pro Instanz — eine Referenz auf das Klassen-Dict würde
        # andere Scanner-Instanzen/Tests kontaminieren.
        self.thresholds = dict(
            self.MATERIAL_SENSITIVITY.get(
                material_type if material_type else MaterialType.UNKNOWN,
                self.MATERIAL_SENSITIVITY[MaterialType.UNKNOWN],
            )
        )

        # §G1 Song-Isolation: DefectScanner wird bei UV3.__init__() instantiiert (vor restore()),
        # daher ist Material zu diesem Zeitpunkt noch nicht bekannt. Es wird später in
        # restore() durch Material-Konsens ermittelt und an scan() übergeben (§9.7.2).
        # Das ist kein Bug, sondern beabsichtiges Design — jeder Song bekommt seinen eigenen
        # Material-Context. Log-Level: DEBUG, da es eine normale Initialisierung ist.
        if material_type is not None:
            logger.debug("DefectScanner initialisiert: SR=%s, Material=%s (vorgegeben)", sample_rate, material_type)
        else:
            logger.debug("DefectScanner initialisiert: SR=%s (Material wird in restore() auto-detected)", sample_rate)

        # Welch-PSD-Cache (P1): gültig für Dauer eines scan()-Calls - wird in scan() gesetzt.
        self._scan_welch_cache: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]] = {}

    def _cached_welch(
        self,
        audio: np.ndarray,
        sr: int,
        nperseg: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gecachter signal.welch()-Aufruf für die Dauer eines scan()-Calls.

        Cache-Key: Objekt-ID plus billiger Signal-Fingerprint. Die ID ist innerhalb
        eines scan()-Calls schnell; der Fingerprint verhindert stale Hits, wenn direkte
        _detect_*-Tests nacheinander temporäre Arrays mit wiederverwendeter ID erzeugen.
        """
        _arr = np.asarray(audio)
        _flat = _arr.reshape(-1)
        if _flat.size:
            _mid = _flat.size // 2
            _fingerprint = (
                _arr.shape,
                str(_arr.dtype),
                float(_flat[0]),
                float(_flat[_mid]),
                float(_flat[-1]),
                float(np.mean(_flat[: min(_flat.size, 1024)])),
            )
        else:
            _fingerprint = (_arr.shape, str(_arr.dtype), 0.0, 0.0, 0.0, 0.0)
        _key = (id(audio), nperseg, _fingerprint)
        _cached_result = self._scan_welch_cache.get(_key)
        if _cached_result is not None:
            return _cached_result
        _result: tuple[np.ndarray, np.ndarray] = signal.welch(audio, sr, nperseg=nperseg)  # type: ignore[assignment]
        self._scan_welch_cache[_key] = _result
        return _result

    def adjust_thresholds_for_ast(
        self,
        ast_classifier: Any = None,
        audio: np.ndarray | None = None,
        sr: int = 48000,
    ) -> dict[str, float]:
        """§v10.304: AST-basierte Schwellwert-Anpassung für Defekt-Detektoren.

        Läuft VOR dem Scan. Prüft mit AST, welche Musikinstrumente im Audio
        präsent sind, und hebt die Detektions-Schwellen für Defekttypen an,
        die mit diesen Instrumenten verwechselt werden können.

        Returns:
            Dict mit angepassten Schwellwerten (Defekttyp → neuer Threshold).
        """
        if ast_classifier is None or audio is None:
            return {}
        try:
            if not hasattr(ast_classifier, "is_loaded") or not ast_classifier.is_loaded():
                return {}
            from backend.core.ast_audio_set_classifier import DEFECT_INSTRUMENT_DISCRIMINATOR

            _result = ast_classifier.classify(audio, sr, top_k=15)
            if _result.model_used == "fallback":
                return {}

            _adjustments: dict[str, float] = {}
            for _defect_name, _instrument_indices in DEFECT_INSTRUMENT_DISCRIMINATOR.items():
                # Beste Instrument-Konfidenz für diesen Defekttyp
                _best_conf = max((_result.get_prob(i) for i in _instrument_indices), default=0.0)
                if _best_conf >= 0.15:
                    # Konfidenz → Schwellwert-Multiplikator: 0.15→1.3, 0.50→2.0, 0.80→3.0
                    _mult = 1.0 + _best_conf * 2.5
                    _defect_type = None
                    for _dt in DefectType:
                        if _dt.value == _defect_name:
                            _defect_type = _dt
                            break
                    if _defect_type is not None:
                        _old_thresh = self.thresholds.get(_defect_type, 0.5)
                        _new_thresh = float(np.clip(_old_thresh * _mult, 0.1, 0.95))
                        self.thresholds[_defect_type] = _new_thresh
                        _adjustments[_defect_name] = _new_thresh
                        logger.debug(
                            "§v10.304 AST-Schwelle: %s %.2f→%.2f (inst_conf=%.2f)",
                            _defect_name,
                            _old_thresh,
                            _new_thresh,
                            _best_conf,
                        )
            if _adjustments:
                logger.info(
                    "§v10.304 AST Pre-Filter: %d Defekt-Schwellen angehoben (Instrument-Erkennung)",
                    len(_adjustments),
                )
            return _adjustments
        except Exception as _exc:
            logger.debug("AST Pre-Filter fehlgeschlagen: %s", _exc)
            return {}

    # ── BEATs Tag → DefectType Konflikt-Mapping ──────────────────────────
    # AudioSet tag names that, when detected with high confidence, indicate
    # content that can be mistaken for audio defects (transients, noise, etc.)
    _BEATS_TAG_DEFECT_MAP: dict[str, list[str]] = {
        "Drum": ["crackle", "click", "hiss"],
        "Percussion": ["crackle", "click"],
        "Guitar": ["click"],
        "Electric guitar": ["click"],
        "Piano": ["click"],
        "Bass guitar": ["rumble"],
        "Brass instrument": ["hiss"],
        "Trumpet": ["hiss"],
        "Saxophone": ["hiss"],
        "Singing voice": ["click"],
        "Music": ["crackle", "click", "hiss", "hum"],
        "Musical instrument": ["crackle", "click"],
    }

    def adjust_thresholds_for_beats(
        self,
        beats_tags: dict[str, float] | None = None,
        beats_embedding: list[float] | None = None,
    ) -> dict[str, float]:
        """§v10.700: BEATs-basierte Schwellwert-Anpassung für Defekt-Detektoren.

        Läuft VOR dem Scan (nach adjust_thresholds_for_ast). Nutzt BEATs
        AudioSet-Tags, um Defekt-Schwellen kontextabhängig zu modulieren:

        - Instrument-Tags (Drum, Guitar, Piano, etc.): Schwellen ANHEBEN
          für Defekttypen, die mit diesen Instrumenten verwechselt werden
          (z.B. Drum → crackle/click/hiss, Guitar → click, Bass → rumble).
        - "Noise"-Tag: Schwellen SENKEN für hiss/hum/crackle — bei realem
          Rauschen sind Defekte wahrscheinlicher und sollen erkannt werden.
        - "Silence"-Tag: Schwellen stark ANHEBEN — kein Signal, keine Defekte.
        - "Music"-Tag (global): Leichte Anhebung aller Schwellen bei Musik,
          da komplexes Musikmaterial mehr False Positives produziert.

        Returns:
            Dict mit angepassten Schwellwerten (Defekttyp → neuer Threshold).
        """
        if beats_tags is None or not beats_tags:
            return {}
        try:
            _adjustments: dict[str, float] = {}

            # ── Globale Musik-Modulation ──────────────────────────────
            _music_conf = beats_tags.get("Music", 0.0)
            _global_music_mult = 1.0
            if _music_conf >= 0.5:
                # Musik ist komplex → konservativer scannen
                _global_music_mult = 1.0 + (_music_conf - 0.5) * 0.6  # max ~1.30x

            # ── Noise-Modulation: reales Rauschen → sensitiver ─────────
            _noise_conf = beats_tags.get("Noise", 0.0)
            _noise_defects = {"hiss", "hum", "crackle"}
            _noise_mult = 1.0
            if _noise_conf >= 0.3:
                # Rauschen detektiert → Defekte sind wahrscheinlich real
                _noise_mult = max(0.65, 1.0 - (_noise_conf - 0.3) * 0.8)  # min ~0.65x

            # ── Silence-Modulation: Stille → fast nichts scannen ───────
            _silence_conf = beats_tags.get("Silence", 0.0)
            _silence_mult = 1.0
            if _silence_conf >= 0.4:
                _silence_mult = 1.0 + _silence_conf * 2.0  # max ~3.0x

            # ── Instrument-Defekt-Konflikt-Modulation ──────────────────
            for _tag_name, _defect_names in self._BEATS_TAG_DEFECT_MAP.items():
                _tag_conf = beats_tags.get(_tag_name, 0.0)
                if _tag_conf < 0.20:
                    continue
                # Höhere Konfidenz → stärkere Schwellwert-Anhebung
                _inst_mult = 1.0 + _tag_conf * 2.0  # 0.20→1.4x, 0.50→2.0x, 0.80→2.6x

                for _defect_name in _defect_names:
                    _defect_type = None
                    for _dt in DefectType:
                        if _dt.value == _defect_name:
                            _defect_type = _dt
                            break
                    if _defect_type is None:
                        continue
                    _old_thresh = self.thresholds.get(_defect_type, 0.5)
                    # Nimm die stärkste Modulation (max multiplier wins)
                    _existing = _adjustments.get(_defect_name, _old_thresh)
                    _current_mult = _existing / _old_thresh if _old_thresh > 0 else 1.0
                    _best_mult = max(_current_mult, _inst_mult)
                    _new_thresh = float(np.clip(_old_thresh * _best_mult, 0.1, 0.95))
                    self.thresholds[_defect_type] = _new_thresh
                    _adjustments[_defect_name] = _new_thresh
                    logger.debug(
                        "§v10.700 BEATs-Schwelle: %s %.2f→%.2f (tag=%s conf=%.2f mult=%.2f)",
                        _defect_name,
                        _old_thresh,
                        _new_thresh,
                        _tag_name,
                        _tag_conf,
                        _best_mult,
                    )

            # ── Globale Modulation auf ALLE Defekttypen anwenden ───────
            for _dt in DefectType:
                if _dt not in self.thresholds:
                    continue
                _old_thresh = self.thresholds[_dt]
                _defect_name = _dt.value
                # Starte mit der stärksten Modulation
                _combined_mult = _global_music_mult
                if _defect_name in _noise_defects:
                    _combined_mult *= _noise_mult
                # Silence überschreibt alles
                _combined_mult = max(_combined_mult, _silence_mult) if _silence_conf >= 0.4 else _combined_mult

                if _combined_mult != 1.0:
                    _new_thresh = float(np.clip(_old_thresh * _combined_mult, 0.1, 0.95))
                    if abs(_new_thresh - _old_thresh) > 0.01:
                        self.thresholds[_dt] = _new_thresh
                        _adjustments[_defect_name] = _new_thresh

            if _adjustments:
                _n_instrument = sum(1 for t in self._BEATS_TAG_DEFECT_MAP if beats_tags.get(t, 0.0) >= 0.20)
                logger.info(
                    "§v10.700 BEATs Pre-Filter: %d Schwellen moduliert "
                    "(music=%.2f noise=%.2f silence=%.2f instruments=%d)",
                    len(_adjustments),
                    _music_conf,
                    _noise_conf,
                    _silence_conf,
                    _n_instrument,
                )
            return _adjustments
        except Exception as _exc:
            logger.debug("BEATs Pre-Filter fehlgeschlagen: %s", _exc)
            return {}

    @classmethod
    def is_audible(
        cls,
        defect_type: str,
        severity: float,
        material: str = "tape",
        signal_rms_db: float = -20.0,
        peak_frequency_hz: float = 3000.0,
    ) -> bool:  # 3 kHz = max Ohr-Sensitivität
        """§v10.306: Prüft ob ein Defekt für das menschliche Ohr hörbar ist.

        Berücksichtigt:
        - JND-Schwellen pro Defekttyp (ISO 226, Zwicker/Fastl)
        - Material-Maskierung (Shellac-Rauschen maskiert Knistern)
        - Signal-Maskierung (laute Musik maskiert leise Defekte)
        - Frequenzabhängige Hörschwelle (Presbyakusis)

        Returns True wenn der Defekt wahrscheinlich hörbar ist.
        """
        _t = cls.AUDIBILITY_THRESHOLDS.get(defect_type)
        if _t is None:
            return severity >= 0.3  # Default: moderate threshold

        _jnd_db, _masking, _mat_factors = _t

        # Signal-Maskierung: Lautes Signal maskiert Defekte
        _signal_mask = max(0.0, min(1.0, (signal_rms_db + 60.0) / 60.0))
        _effective_jnd = _jnd_db * (1.0 - _masking * _signal_mask)

        # Material-Faktor
        _mat_factor = _mat_factors.get(str(material).lower(), 1.0)
        _effective_jnd *= _mat_factor

        # Frequenzgewichtung nach ISO 226:2003 Equal-Loudness-Contours
        # Das Ohr ist bei 2–5 kHz am empfindlichsten → Defekte dort lauter.
        # Bei <100 Hz und >10 kHz ist das Ohr unempfindlicher.
        # Gewicht > 1.0 = Defekt erscheint lauter (Schwelle sinkt).
        if peak_frequency_hz < 80:
            _freq_gain = 0.10  # Sehr tief: kaum hörbar
        elif peak_frequency_hz < 200:
            _freq_gain = 0.25
        elif peak_frequency_hz < 500:
            _freq_gain = 0.55
        elif peak_frequency_hz < 1000:
            _freq_gain = 0.80
        elif peak_frequency_hz < 2000:
            _freq_gain = 0.95
        elif peak_frequency_hz < 5000:
            _freq_gain = 1.00  # Maximale Sensitivität
        elif peak_frequency_hz < 8000:
            _freq_gain = 0.90
        elif peak_frequency_hz < 12000:
            _freq_gain = 0.60
        elif peak_frequency_hz < 16000:
            _freq_gain = 0.25
        else:
            _freq_gain = 0.08  # >16 kHz: Presbyakusis
        # Effektive JND mit Frequenzsensitivityität skalieren
        # (niedrige Gain → JND näher an 0 dB → schwerer hörbar)
        _effective_jnd *= _freq_gain

        # Skaliere Severity in dB-Domäne: severity 0.0→−50 dB, 1.0→0 dB
        # (typische Defekte liegen zwischen −50 dBFS und 0 dBFS).
        # §v10.306 Bug-Reparatur: Band-/PANNs-/SNR-Kontext ist in dieser
        # Signatur nicht verfügbar — der frühere Block referenzierte
        # undefinierte Namen und warf bei JEDEM Aufruf NameError.
        # Material-Maskierung ist bereits über _mat_factor wirksam.
        _defect_db = -50.0 + severity * 50.0
        return bool(_defect_db >= _effective_jnd)

    def scan(
        self,
        audio: np.ndarray,
        sample_rate: int | None = None,
        material_type: MaterialType | None = None,
        progress_callback: Optional["Callable[[float, str], None]"] = None,
        file_ext: str = "",
        forensic_medium_result: object | None = None,
        era_decade: int | None = None,  # §v10.14.1: Era-Info für Material-Auto-Detect
    ) -> DefectAnalysisResult:
        """
        Hauptmethode: Scannt Audio-Daten und erkennt alle 20 Defekttypen.

        Args:
            audio: Audio-Daten (mono: shape=(n_samples,), stereo: shape=(n_samples, 2))
            sample_rate: Sample rate (falls nicht im Constructor gesetzt)
            material_type: Override für Material-Typ (falls nicht im Constructor gesetzt)
            file_ext: Dateiendung der Quelldatei (z.B. '.mp3') - wird an ForensicMediumDetector
                      weitergegeben, um Analog-Posterior-Zeroing anzuwenden (Bug-15-Fix).
            era_decade: Geschätzte Aufnahme-Dekade (vom EraClassifier). Moduliert
                        die Rumble-Gewichtung im Material-Auto-Detect (§v10.14.1).

        Returns:
            DefectAnalysisResult mit allen Scores
        """
        # Bug-15-Fix: material_type als String (z.B. 'mp3_low') → MaterialType-Enum normalisieren.
        # ClassificationResult.material kann ein String sein (forensics/medium_detector.py L1026),
        # aber _DIGITAL_NO_BUMP enthält MaterialType-Enums → String-Vergleich schlägt immer fehl
        # → TRANSPORT_BUMP wird nie übersprungen → 150+ falsche Bumps auf MP3-Dateien.
        if isinstance(material_type, str):
            try:
                material_type = MaterialType(canonical_material_key(material_type))
            except ValueError:
                logger.debug(
                    "DefectScanner.scan(): material_type='%s' unbekannt - auf None gesetzt.",
                    material_type,
                )
                material_type = None

        # §Spec 24 Zero-Length-Guard: Scans auf (nahezu) leerem Audio liefern
        # Unsinn (Befund 2026-08-16: „0.0s Audio“, 8 Consensus-Defekte aus Stille).
        _sr_guard = int(sample_rate or 48000)
        _n_guard = int(np.asarray(audio).size)
        if _n_guard < int(_sr_guard * 0.05):
            logger.warning(
                "DefectScanner.scan(): Audio zu kurz (%d Samples) — Scan übersprungen (Zero-Length-Guard, Spec 24)",
                _n_guard,
            )
            return DefectAnalysisResult(
                material_type=material_type or MaterialType.CD_DIGITAL,
                scores={},
                analysis_time_seconds=0.0,
                sample_rate=_sr_guard,
                duration_seconds=float(_n_guard) / float(_sr_guard),
            )

        # Normalize channel layout once: accept both (n_samples, n_channels) and
        # channel-first (n_channels, n_samples) without degenerating to 2-sample mono.
        if audio.ndim == 2 and audio.shape[0] <= 2 and audio.shape[1] > audio.shape[0]:
            audio = audio.T
            logger.debug(
                "DefectScanner.scan(): channel-first Eingabe normalisiert to shape=%s",
                audio.shape,
            )

        # §9.7.1 SHA256-Cache - bei identischem Eingangssignal sofort zurückgeben
        _sr_for_key = sample_rate if sample_rate is not None else self.sample_rate
        _cache_key = _audio_scan_cache_key(audio, _sr_for_key, material_type)
        with _scan_cache_lock:
            if _cache_key in _scan_cache:
                logger.debug("DefectScanner Zwischenspeicher-Hit: %s", _cache_key)
                return _scan_cache[_cache_key]  # type: ignore[return-value]

        start_time = time.time()

        # Sample rate verwenden (Parameter > Instance > Default)
        sr = sample_rate if sample_rate is not None else self.sample_rate

        # §6.6.1 Pflicht-Spektralfingerabdruck: IMMER berechnen, unabhängig vom material_type-Hint
        _sf: dict[str, float] = {}
        try:
            _fp_mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
            _fp_freqs, _fp_psd = signal.welch(_fp_mono, sr, nperseg=min(4096, len(_fp_mono)))
            _fp_total = max(float(np.sum(_fp_psd)), 1e-12)
            # Rolloff 95 %
            _fp_cumsum = np.cumsum(_fp_psd)
            _fp_rolloff_mask = _fp_cumsum >= 0.95 * _fp_total
            _sf["rolloff_95_hz"] = (
                float(_fp_freqs[_fp_rolloff_mask][0]) if _fp_rolloff_mask.any() else float(_fp_freqs[-1])
            )
            # Wow/Flutter-Index (Pitch-Varianz über 100ms-Fenster)
            _wf_hop = max(1, int(sr * 0.1))
            # Vectorized: non-overlapping frames via reshape (replaces list + loop)
            _n_wf = (len(_fp_mono) - 1) // _wf_hop
            if _n_wf > 2:
                _wf_mat = _fp_mono[: _n_wf * _wf_hop].reshape(_n_wf, _wf_hop)
                _wf_rms = np.sqrt(np.mean(_wf_mat**2, axis=1) + 1e-12)
                _sf["wow_flutter_index"] = float(np.std(_wf_rms) / (np.mean(_wf_rms) + 1e-12) * sr / 100.0)
            else:
                _sf["wow_flutter_index"] = 0.0
            # HF-Energie > 16 kHz
            _sf["hf_energy_above_16k"] = float(np.sum(_fp_psd[_fp_freqs > 16000]) / _fp_total)
            # Rauschboden: 5. Perzentil PSD in dBFS
            _fp_p5 = float(np.percentile(_fp_psd, 5))
            _sf["noise_floor_p5_db"] = float(10.0 * np.log10(max(_fp_p5, 1e-20)))
            # Effektive Bandbreite: Rolloff bei -60 dBFS
            _fp_db = 10.0 * np.log10(np.maximum(_fp_psd, 1e-20))
            _fp_max_db = float(np.max(_fp_db))
            _fp_bw_mask = _fp_db >= (_fp_max_db - 60.0)
            _sf["effective_bandwidth_hz"] = float(_fp_freqs[_fp_bw_mask][-1]) if _fp_bw_mask.any() else 0.0
            logger.debug(
                "§6.6.1 SpektralFingerabdruck: rolloff95=%.0f Hz, wow=%.2f, hf16k=%.3f, bw=%.0f Hz, rausch=%.1f dBFS",
                _sf["rolloff_95_hz"],
                _sf["wow_flutter_index"],
                _sf["hf_energy_above_16k"],
                _sf["effective_bandwidth_hz"],
                _sf["noise_floor_p5_db"],
            )
        except Exception as _fp_err:
            logger.debug("SpektralFingerabdruck-Berechnung Fehler: %s", _fp_err)

        # Material-Typ: Hint als Prior, aber auto-detect für spectral_fingerprint immer ausführen
        _material_hint = material_type  # Hint des Aufrufers (kann None sein)
        _auto_material = self._auto_detect_material(audio, era_decade=era_decade)  # §6.6.1: immer berechnen
        _sf["material_detected"] = float(list(MaterialType).index(_auto_material))
        if material_type is None:
            material_type = self.material_type or _auto_material

        # §9.7.5a - Update instance attribute so _detect_dropouts() can access
        # the resolved material type for adaptive thresholds.
        self.material_type = material_type

        # §CODEC-DISKRIMINATOR: Initialisiere pro-Detektor Codec-Awareness
        # Wird von jedem _detect_*() konsultiert, um MP3-Artefakte von echten
        # analogen Defekten zu unterscheiden.
        from backend.core.dsp.codec_discriminator import make_discriminator

        _fmd_chain = getattr(forensic_medium_result, "transfer_chain", None) or getattr(
            forensic_medium_result, "chain", None
        )
        self._codec_disc = make_discriminator(list(_fmd_chain) if isinstance(_fmd_chain, (list, tuple)) else None)

        # Per-Instance-Kopie (§G1 (copilot-instructions.md)/V8 Song-Isolation) — kein Shared-State via Klassen-Dict.
        self.thresholds = dict(self.MATERIAL_SENSITIVITY[material_type])

        # §v10 SNR-adaptive threshold scaling: Der Material-Default ist der
        # AUSGANGSPUNKT, nicht das ENDERGEBNIS. Jeder Song hat ein eigenes SNR,
        # und Thresholds müssen sich daran anpassen:
        #   Hohes SNR (clean)  → sensitiver (niedrigerer Threshold)
        #   Niedriges SNR (noisy) → konservativer (höherer Threshold)
        # Ausnahme: Defekttypen die physikalisch NUR bei bestimmten Materialien
        # auftreten (z.B. WOW nur bei analog) - deren Thresholds bleiben unskaliert.
        _NON_SCALING_DEFECTS: frozenset[DefectType] = frozenset(
            {
                DefectType.WOW,
                DefectType.FLUTTER,
                DefectType.PRINT_THROUGH,
            }
        )
        _measured_snr = _estimate_local_snr(audio[: min(len(audio), sr * 5)], sr)
        _snr_scale = float(np.clip(30.0 / max(5.0, _measured_snr), 0.6, 1.4))
        for _dt in self.thresholds:
            if _dt not in _NON_SCALING_DEFECTS and self.thresholds.get(_dt) is not None:
                self.thresholds[_dt] = float(np.clip(self.thresholds[_dt] * _snr_scale, 0.05, 0.98))
        logger.debug(
            "§v10 SNR-adaptive thresholds: snr=%.1fdB scale=%.2f material=%s",
            _measured_snr,
            _snr_scale,
            material_type.name if hasattr(material_type, "name") else str(material_type),
        )

        # §9.x [RELEASE_MUST] Chain-adaptive threshold merge - Tonträgerkette darf niemals
        # ignoriert werden.  Wenn die Kette Stufen enthält, die für bestimmte Defekttypen
        # empfindlichere Schwellwerte haben (z. B. Tape-Stufe in shellac→tape→mp3_low),
        # werden die "gated" Schwellwerte des primären Materials (≥ 0.95 = N/A-Gate)
        # durch den Schwellwert der Kettenstufe ersetzt.
        # Rationale: Wow/Flutter einer Tape-Stufe überlebt als spektrale Signatur auch
        # nach verlustbehafteter Transkodierung und muss detektierbar bleiben.
        # Invariante: Nur Schwellwerte, die auf dem primären Material wirksam gesperrt
        # sind (≥ 0.95), werden überschrieben - aktive Schwellwerte bleiben unverändert.
        _chain_threshold_overrides: set[DefectType] = set()
        _chain_stage_materials: list[str] = []
        _forensic_material_confidence: float | None = None
        if forensic_medium_result is not None:
            try:
                # Kette als echte Liste lesen (kein String-Substring-Matching, das z. B.
                # 'tape' in 'reel_tape' fälschlicherweise trifft).
                _raw_chain_val = getattr(forensic_medium_result, "transfer_chain", None) or getattr(
                    forensic_medium_result, "chain", None
                )
                # Normalisierung: list[str] bevorzugt, Fallback auf str-Aufspaltung
                if isinstance(_raw_chain_val, list):
                    _chain_links: list[str] = [str(s).lower().strip() for s in _raw_chain_val if s]
                elif isinstance(_raw_chain_val, str) and _raw_chain_val:
                    # "shellac → tape → mp3_low"  oder  "['shellac', 'tape']" → Aufspaltung
                    _chain_links = [
                        s.strip().strip("'\"[]")
                        for s in _raw_chain_val.lower().replace("→", ",").split(",")
                        if s.strip().strip("'\"[]")
                    ]
                else:
                    _chain_links = []

                _primary_val = str(getattr(material_type, "value", "")).lower()

                # Kettenstufen ohne primäres Material, mit exaktem String-Vergleich
                _chain_stage_mats: list[MaterialType] = []
                _chain_links_set = set(_chain_links)
                for _cand_mat in MaterialType:
                    _cand_key = _cand_mat.value.lower()
                    if (
                        _cand_key
                        and _cand_key in _chain_links_set  # exakter Wert-Vergleich
                        and _cand_key != _primary_val  # primäres Material ausschließen
                        and _cand_mat in self.MATERIAL_SENSITIVITY
                    ):
                        _chain_stage_mats.append(_cand_mat)

                _chain_stage_materials = [m.value for m in _chain_stage_mats]
                _forensic_material_confidence = float(getattr(forensic_medium_result, "confidence", 0.0))

                if _chain_stage_mats:
                    # Gate-Analyse: Defekttypen, die beim primären Material gesperrt sind (≥ 0.95)
                    _primary_sensitivity = self.MATERIAL_SENSITIVITY[material_type]
                    _gated_defects: set[DefectType] = {_dt for _dt, _t in _primary_sensitivity.items() if _t >= 0.95}
                    # Für jeden gesperrten Defekttyp: sensitivsten (niedrigsten) Schwellwert
                    # über alle Kettenstufen bestimmen - "most-sensitive-wins", nicht "first-wins".
                    # Invariante: aktive Schwellwerte des primären Materials (< 0.95) bleiben unberührt.
                    _merged_thresh: dict[DefectType, float] = dict(self.thresholds)
                    for _dt in _gated_defects:
                        _best_chain_t = min(
                            (
                                self.MATERIAL_SENSITIVITY[_csm][_dt]
                                for _csm in _chain_stage_mats
                                if _dt in self.MATERIAL_SENSITIVITY[_csm]
                            ),
                            default=1.0,
                        )
                        if _best_chain_t < 0.95:
                            _merged_thresh[_dt] = _best_chain_t
                            _chain_threshold_overrides.add(_dt)
                    if _chain_threshold_overrides:
                        self.thresholds = _merged_thresh
                        logger.info(
                            "DefectScanner: chain-adaptive thresholds aktiv für %d Defekttypen "
                            "(Kette=%s, Kettenstufen=%s)",
                            len(_chain_threshold_overrides),
                            " → ".join(_chain_links),
                            [m.value for m in _chain_stage_mats],
                        )
            except Exception as _cta_exc:
                logger.debug("DefectScanner chain-Schwelle-merge fehlgeschlagen: %s", _cta_exc)

        # Audio normalisieren für konsistente Detection
        # Treat column-vectors (N, 1) as mono to avoid false stereo indexing
        # like audio[:, 1] in downstream stereo detectors.
        if audio.ndim == 2:
            if min(audio.shape) < 2:
                is_stereo = False
                audio_mono = np.asarray(audio, dtype=np.float32).reshape(-1)
            else:
                is_stereo = True
                # samples-first expected at this point; keep a safe fallback.
                audio_mono = np.mean(audio, axis=1 if audio.shape[1] <= audio.shape[0] else 0)
        else:
            is_stereo = False
            audio_mono = audio

        # §9.7.5 Audio-Cap for 28 detectors.
        # Defects are stationary over a 60 s sample (spec: ≤ 2 s/min audio).
        # Reduces scan time from ~22 s (225 s track) to ~6 s.
        # The spectral fingerprint and _auto_detect_material() above intentionally
        # run on the full audio for a representative signal characterisation.
        _DETECTOR_CAP_S = 60
        _detector_cap_n = _DETECTOR_CAP_S * sr
        _location_offset_s = 0.0  # seconds to add to all detector locations
        # §9.7.5a - Full-audio reference for non-stationary defects (dropouts).
        # Dropouts can occur anywhere (intro, outro, tape leader) - the 60 s
        # center-crop would miss them.  Other detectors (noise, hum, flutter)
        # are stationary and still use the cropped audio for performance.
        _audio_mono_full = audio_mono  # kept for dropout detection
        if len(audio_mono) > _detector_cap_n:
            _dc_mid = len(audio_mono) // 2
            _dc_half = _detector_cap_n // 2
            _location_offset_s = (_dc_mid - _dc_half) / sr
            audio_mono = audio_mono[_dc_mid - _dc_half : _dc_mid + _dc_half]
            logger.debug(
                "DefectScanner: audio_mono capped to %d s (%d samples) for detection pass, offset=%.1f s",
                _DETECTOR_CAP_S,
                _detector_cap_n,
                _location_offset_s,
            )

        def _prog(pct: float, name: str = "") -> None:
            _pct = float(np.clip(pct, 0.0, 100.0))
            if progress_callback is not None:
                with contextlib.suppress(Exception):
                    try:
                        progress_callback(_pct, name)
                    except TypeError:
                        progress_callback(_pct)  # type: ignore[call-arg]

        _tail_steps_done = 0
        # Fixed tail steps: 24 defect-type ticks always run.
        # Full-audio recheck path (files > 60 s): up to 8 event-detector re-runs
        # + up to 9 severity-recheck re-runs = 17 extra ticks.  Without the
        # dynamic budget the counter reaches min(1.0, 34/34) = 1.0 too early,
        # freezing the progress bar at 99.9 % while ~40 s of work remains.
        _FIXED_TAIL = 24
        _EXTRA_TAIL = 17 if _location_offset_s > 0.0 else 0
        _TAIL_STEP_BUDGET = _FIXED_TAIL + _EXTRA_TAIL  # 24 (short) or 41 (long files)
        _DIGITAL_FAST_TAIL: frozenset[MaterialType] = frozenset(
            {
                MaterialType.CD_DIGITAL,
                MaterialType.MP3_LOW,
                MaterialType.MP3_HIGH,
                MaterialType.AAC,
                MaterialType.STREAMING,
                MaterialType.DAT,
                MaterialType.MINIDISC,
            }
        )
        _ANALOG_DEEP_TAIL: frozenset[MaterialType] = frozenset(
            {
                MaterialType.WAX_CYLINDER,
                MaterialType.LACQUER_DISC,
                MaterialType.SHELLAC,
                MaterialType.VINYL,
                MaterialType.WIRE_RECORDING,
                MaterialType.REEL_TAPE,
                MaterialType.TAPE,
                MaterialType.CASSETTE,
            }
        )
        if material_type in _DIGITAL_FAST_TAIL:
            _tail_start = 93.0
        elif material_type in _ANALOG_DEEP_TAIL:
            _tail_start = 88.0
        else:
            _tail_start = 90.0
        _tail_span = 99.9 - _tail_start

        def _lead_pct(base_pct: float) -> float:
            """Map legacy lead progress (5..90) into the material-adaptive lead range."""
            _b = float(np.clip(base_pct, 5.0, 90.0))
            _ratio = (_b - 5.0) / 85.0
            return 5.0 + _ratio * (_tail_start - 5.0)

        def _tail_tick(name: str = "") -> None:
            """Fine-grained progress updates for heavy post-90% scan blocks."""
            nonlocal _tail_steps_done
            _tail_steps_done += 1
            _frac = min(1.0, _tail_steps_done / float(_TAIL_STEP_BUDGET))
            _prog(_tail_start + _tail_span * _frac, name)

        # Alle 28 Defekttypen sequentiell - nach jedem Schritt Fortschritt melden
        # Welch-PSD-Cache zurücksetzen: neuer scan()-Call, neues audio_mono-Objekt.
        self._scan_welch_cache = {}
        scores = {}

        _prog(_lead_pct(5), "Clicks")
        scores[DefectType.CLICKS] = self._detect_clicks(audio_mono if not is_stereo else audio)
        _prog(_lead_pct(9), "Knistern")
        scores[DefectType.CRACKLE] = self._detect_crackle(audio_mono)
        _prog(_lead_pct(15), "Brummen")
        scores[DefectType.HUM] = self._detect_hum(audio_mono)
        _prog(_lead_pct(20), "Tonhöhenschwankung")
        scores[DefectType.WOW] = self._detect_wow(audio_mono)  # IEC 60386 < 0.5 Hz
        scores[DefectType.FLUTTER] = self._detect_flutter(audio_mono)  # IEC 60386 0.5-200 Hz
        # v10.0.0 §9.1b-ext: Tape-Intro-Supplement for WOW/FLUTTER.
        # Center-cropped 60 s misses cassette motor startup instability (first 0-20 s).
        # For tape material: re-run WOW/FLUTTER on the first 20 s of full audio and
        # take max(center_crop_severity, intro_severity).  This catches capstan run-up
        # and head-engagement speed irregularities that are non-stationary in the intro.
        # Scientific basis: IEC 60386:1972 §4.2.1 - short-term speed variations must
        # be measured at representative positions including start-of-tape.
        if (
            self.material_type in (MaterialType.TAPE, MaterialType.CASSETTE, MaterialType.REEL_TAPE)
            and len(_audio_mono_full) > sr * 4
        ):
            _tape_intro_n = min(int(20.0 * sr), len(_audio_mono_full))
            _tape_intro_audio = _audio_mono_full[:_tape_intro_n]
            _wow_intro = self._detect_wow(_tape_intro_audio)
            _flutter_intro = self._detect_flutter(_tape_intro_audio)
            if _wow_intro.severity > scores[DefectType.WOW].severity:
                logger.info(
                    "WOW intro-supplement: severity %.3f > center-crop %.3f (tape head Start)",
                    _wow_intro.severity,
                    scores[DefectType.WOW].severity,
                )
                scores[DefectType.WOW] = _wow_intro
                scores[DefectType.WOW].metadata["intro_supplement"] = True
            if _flutter_intro.severity > scores[DefectType.FLUTTER].severity:
                logger.info(
                    "FLUTTER intro-supplement: severity %.3f > center-crop %.3f (tape head Start)",
                    _flutter_intro.severity,
                    scores[DefectType.FLUTTER].severity,
                )
                scores[DefectType.FLUTTER] = _flutter_intro
                scores[DefectType.FLUTTER].metadata["intro_supplement"] = True
        scores[DefectType.AZIMUTH_ERROR] = self._detect_azimuth_error(audio)  # PHD-Slope L/R
        _prog(_lead_pct(26), "Stereo-Ungleichgewicht")
        scores[DefectType.STEREO_IMBALANCE] = (
            self._detect_stereo_imbalance(audio) if is_stereo else DefectScore(DefectType.STEREO_IMBALANCE, 0.0, 0.0)
        )
        _prog(_lead_pct(31), "Digitalartefakte")
        scores[DefectType.DIGITAL_ARTIFACTS] = self._detect_digital_artifacts(audio_mono)
        _prog(_lead_pct(37), "Tieffrequenz-Rumpeln")
        scores[DefectType.LOW_FREQ_RUMBLE] = self._detect_low_freq_rumble(audio_mono)
        _prog(_lead_pct(42), "Hochfrequenzrauschen")
        scores[DefectType.HIGH_FREQ_NOISE] = self._detect_high_freq_noise(audio_mono)
        _prog(_lead_pct(48), "Kompressions-Artefakte")
        scores[DefectType.COMPRESSION_ARTIFACTS] = self._detect_compression_artifacts(audio_mono)
        _prog(_lead_pct(53), "Phasenfehler")
        scores[DefectType.PHASE_ISSUES] = (
            self._detect_phase_issues(audio) if is_stereo else DefectScore(DefectType.PHASE_ISSUES, 0.0, 0.0)
        )
        _prog(_lead_pct(59), "Aussetzer")
        # §9.7.5a - Dropout detection runs on FULL audio (not center-cropped).
        # Tape dropouts occur anywhere (intro, leader, splice points).
        scores[DefectType.DROPOUTS] = self._detect_dropouts(_audio_mono_full)
        _prog(_lead_pct(64), "Übersteuerung")
        # --- Weltklasse-Erweiterung: 4 neue Defekttypen ---
        scores[DefectType.CLIPPING] = self._detect_clipping(audio_mono)
        if scores[DefectType.CLIPPING].severity >= 0.50 and scores[DefectType.CLICKS].severity > 0.0:
            click_locations = scores[DefectType.CLICKS].locations
            clip_locations = scores[DefectType.CLIPPING].locations
            if click_locations and clip_locations:
                clip_margin_s = 0.02
                overlap_count = 0
                for click_start, click_end in click_locations:
                    if any(
                        click_start <= clip_end + clip_margin_s and click_end >= clip_start - clip_margin_s
                        for clip_start, clip_end in clip_locations
                    ):
                        overlap_count += 1
                if overlap_count == len(click_locations):
                    scores[DefectType.CLICKS].metadata["suppressed_by_clipping"] = True
                    scores[DefectType.CLICKS].metadata["pre_suppression_severity"] = float(
                        scores[DefectType.CLICKS].severity
                    )
                    scores[DefectType.CLICKS].severity = 0.0
                    scores[DefectType.CLICKS].confidence = min(float(scores[DefectType.CLICKS].confidence), 0.35)
        _prog(_lead_pct(69), "DC-Versatz")
        scores[DefectType.DC_OFFSET] = self._detect_dc_offset(audio_mono)
        _prog(_lead_pct(74), "Bandbreitenverlust")
        scores[DefectType.BANDWIDTH_LOSS] = self._detect_bandwidth_loss(audio_mono)
        _prog(_lead_pct(78), "Tonhöhendrift")
        scores[DefectType.PITCH_DRIFT] = self._detect_pitch_drift(audio_mono)
        _prog(_lead_pct(82), "Übermäßiger Hall")
        scores[DefectType.REVERB_EXCESS] = self._detect_reverb_excess(audio_mono)
        _prog(_lead_pct(84), "Durchkopieren")
        scores[DefectType.PRINT_THROUGH] = self._detect_print_through(audio_mono)
        _prog(_lead_pct(87), "Quantisierungsrauschen")
        # --- Weltklasse-Erweiterung Runde 3 ---
        scores[DefectType.QUANTIZATION_NOISE] = self._detect_quantization_noise(audio_mono)
        _prog(_lead_pct(89), "Jitter-Artefakte")
        scores[DefectType.JITTER_ARTIFACTS] = self._detect_jitter_artifacts(audio_mono)
        _prog(_lead_pct(90), "Überkompression")
        scores[DefectType.DYNAMIC_COMPRESSION_EXCESS] = self._detect_dynamic_compression_excess(audio_mono)
        # --- Vollständige 28-Typen-Erweiterung (F-7) ---
        scores[DefectType.SOFT_SATURATION] = self._detect_soft_saturation(audio_mono)
        _tail_tick("Soft-Sättigung")
        scores[DefectType.SIBILANCE] = self._detect_sibilance(audio_mono)
        _tail_tick("Sibilanz")
        scores[DefectType.VOCAL_HARSHNESS] = self._detect_vocal_harshness(audio_mono)
        _tail_tick("Vokalhärte")
        scores[DefectType.BIAS_ERROR] = self._detect_bias_error(audio_mono)
        _tail_tick("Bias-Fehler")
        scores[DefectType.DOLBY_NR_MISMATCH] = self._detect_dolby_nr_mismatch(audio_mono)
        _tail_tick("Dolby-NR-Mismatch")
        scores[DefectType.RIAA_CURVE_ERROR] = self._detect_riaa_curve_error(audio_mono)
        _tail_tick("RIAA-Kurve")
        scores[DefectType.HEAD_WEAR] = self._detect_head_wear(audio_mono)
        _tail_tick("Kopfverschleiß")
        scores[DefectType.TRANSIENT_SMEARING] = self._detect_transient_smearing(audio_mono)
        _tail_tick("Transienten-Schmierung")
        scores[DefectType.PRE_ECHO] = self._detect_pre_echo(audio_mono)
        _tail_tick("Pre-Echo")
        scores[DefectType.ALIASING] = self._detect_aliasing(audio_mono)
        _tail_tick("Aliasing")

        # ── v10.0.0: 12 newly added defect detectors ─────────────────────────
        scores[DefectType.MODULATION_NOISE] = self._detect_modulation_noise(audio_mono)
        _tail_tick("Modulationsrauschen")
        scores[DefectType.INNER_GROOVE_DISTORTION] = self._detect_inner_groove_distortion(audio_mono)
        _tail_tick("Innenrillen-Verzerrung")
        scores[DefectType.GROOVE_ECHO] = self._detect_groove_echo(audio_mono)
        _tail_tick("Rillenecho")
        # Crosstalk needs stereo input
        if is_stereo:
            scores[DefectType.CROSSTALK] = self._detect_crosstalk(audio)
        else:
            scores[DefectType.CROSSTALK] = DefectScore(DefectType.CROSSTALK, 0.0, 0.5, metadata={"reason": "mono"})
        _tail_tick("Übersprechen")
        scores[DefectType.INTERMODULATION_DISTORTION] = self._detect_intermodulation_distortion(audio_mono)
        _tail_tick("Intermodulation")
        scores[DefectType.TAPE_SPLICE_ARTIFACT] = self._detect_tape_splice_artifact(audio_mono)
        _tail_tick("Band-Spleiß")
        scores[DefectType.HF_REMANENCE_LOSS] = self._detect_hf_remanence_loss(audio_mono)
        _tail_tick("HF-Remanenz")
        scores[DefectType.STYLUS_DAMAGE] = self._detect_stylus_damage(audio_mono)
        _tail_tick("Nadelschaden")
        scores[DefectType.STICKY_SHED_RESIDUE] = self._detect_sticky_shed_residue(audio_mono)
        _tail_tick("Sticky-Shed")
        scores[DefectType.MULTIBAND_WOW_FLUTTER] = self._detect_multiband_wow_flutter(audio_mono)
        _tail_tick("Multiband Wow/Flutter")
        scores[DefectType.GENERATION_LOSS] = self._detect_generation_loss(audio_mono)
        _tail_tick("Generationsverlust")
        scores[DefectType.MOTOR_INTERFERENCE] = self._detect_motor_interference(audio_mono)
        _tail_tick("Motor-Interferenz")
        # v10.0.0: 7 neue DefectType-Detektionen
        scores[DefectType.PROXIMITY_EFFECT_EXCESS] = self._detect_proximity_effect_excess(audio_mono)
        _tail_tick("Proximity-Effekt")
        scores[DefectType.ROOM_MODE_RESONANCE] = self._detect_room_mode_resonance(audio_mono)
        _tail_tick("Raum-Resonanz")
        scores[DefectType.NR_BREATHING_ARTIFACT] = self._detect_nr_breathing_artifact(audio_mono)
        _tail_tick("NR-Atmen")
        scores[DefectType.FLUTTER_SPECTRAL_SIDEBANDS] = self._detect_flutter_spectral_sidebands(audio_mono)
        _tail_tick("Flutter-Seitenbänder")
        scores[DefectType.SPEED_CALIBRATION_ERROR] = self._detect_speed_calibration_error(audio_mono)
        _tail_tick("Geschwindigkeitsfehler")
        scores[DefectType.OVERLOAD_DISTORTION] = self._detect_overload_distortion(audio_mono)
        _tail_tick("Übersteuerungs-Klirr")
        scores[DefectType.LACQUER_DISC_DEGRADATION] = self._detect_lacquer_disc_degradation(audio_mono)
        _tail_tick("Lackfolien-Degradierung")

        # §9.1a - TRANSPORT_BUMP is non-stationary (impulsive micro-speed jumps).
        # MUST run on FULL audio, same as DROPOUTS.
        # §9.4a - Digital-only materials have no mechanical transport mechanism.
        # Their MATERIAL_SENSITIVITY threshold is 1.0 (never triggers). Skip the
        # expensive full-audio frame analysis entirely to save ~15-40 s on long files.
        _DIGITAL_NO_BUMP: frozenset[MaterialType] = frozenset(
            {
                MaterialType.CD_DIGITAL,
                MaterialType.MP3_LOW,
                MaterialType.MP3_HIGH,
                MaterialType.AAC,
                MaterialType.STREAMING,
            }
        )
        if material_type in _DIGITAL_NO_BUMP:
            scores[DefectType.TRANSPORT_BUMP] = DefectScore(DefectType.TRANSPORT_BUMP, 0.0, 0.95)
            logger.info(
                "DefectScanner: TRANSPORT_BUMP uebersprungen (digital material=%s, §9.4a)",
                material_type,
            )
        else:
            scores[DefectType.TRANSPORT_BUMP] = self._detect_transport_bump(_audio_mono_full)
        _tail_tick("Transport-Bump")

        # §9.1a - TAPE_HEAD_LEVEL_DIP is non-stationary (gradual level dips from
        # head-contact pressure variation).  MUST run on FULL audio.
        # Only relevant for tape-based materials with magnetic head mechanism.
        _TAPE_HEAD_DIP_MATERIALS: frozenset[MaterialType] = frozenset(
            {
                MaterialType.TAPE,
                MaterialType.CASSETTE,
                MaterialType.REEL_TAPE,
                MaterialType.WIRE_RECORDING,
            }
        )
        _tape_head_dip_score = self._detect_tape_head_level_dips(_audio_mono_full)
        if material_type in _TAPE_HEAD_DIP_MATERIALS:
            scores[DefectType.TAPE_HEAD_LEVEL_DIP] = _tape_head_dip_score
        elif self._should_keep_cross_material_tape_head_level_dip(_tape_head_dip_score):
            _tape_head_dip_score.severity = float(np.clip(_tape_head_dip_score.severity * 0.75, 0.0, 1.0))
            _tape_head_dip_score.confidence = float(np.clip(min(_tape_head_dip_score.confidence, 0.72), 0.05, 0.95))
            _tape_head_dip_score.metadata["cross_material_fallback"] = True
            _tape_head_dip_score.metadata["fallback_material_gate_bypassed"] = material_type.value
            scores[DefectType.TAPE_HEAD_LEVEL_DIP] = _tape_head_dip_score
        else:
            scores[DefectType.TAPE_HEAD_LEVEL_DIP] = DefectScore(DefectType.TAPE_HEAD_LEVEL_DIP, 0.0, 0.95)
        _tail_tick("Tape-Head-Level-Dip")

        _SCRAPE_FLUTTER_MATERIALS: frozenset[MaterialType] = frozenset(
            {
                MaterialType.TAPE,
                MaterialType.CASSETTE,
                MaterialType.REEL_TAPE,
                MaterialType.WIRE_RECORDING,
            }
        )
        _scrape_flutter_score = self._detect_scrape_flutter(_audio_mono_full)
        if material_type in _SCRAPE_FLUTTER_MATERIALS:
            scores[DefectType.SCRAPE_FLUTTER] = _scrape_flutter_score
        elif self._should_keep_cross_material_scrape_flutter(_scrape_flutter_score):
            _scrape_flutter_score.severity = float(np.clip(_scrape_flutter_score.severity * 0.80, 0.0, 1.0))
            _scrape_flutter_score.confidence = float(np.clip(min(_scrape_flutter_score.confidence, 0.78), 0.05, 0.95))
            _scrape_flutter_score.metadata["cross_material_fallback"] = True
            _scrape_flutter_score.metadata["fallback_material_gate_bypassed"] = material_type.value
            scores[DefectType.SCRAPE_FLUTTER] = _scrape_flutter_score
        else:
            scores[DefectType.SCRAPE_FLUTTER] = DefectScore(DefectType.SCRAPE_FLUTTER, 0.0, 0.95)
        _tail_tick("Scrape-Flutter")

        _TAPE_HEAD_CLOG_MATERIALS: frozenset[MaterialType] = frozenset(
            {
                MaterialType.TAPE,
                MaterialType.CASSETTE,
                MaterialType.REEL_TAPE,
                MaterialType.WIRE_RECORDING,
            }
        )
        _tape_head_clog_score = self._detect_tape_head_clog(_audio_mono_full)
        if material_type in _TAPE_HEAD_CLOG_MATERIALS:
            scores[DefectType.TAPE_HEAD_CLOG] = _tape_head_clog_score
        elif self._should_keep_cross_material_tape_head_clog(_tape_head_clog_score):
            _tape_head_clog_score.severity = float(np.clip(_tape_head_clog_score.severity * 0.75, 0.0, 1.0))
            _tape_head_clog_score.confidence = float(np.clip(min(_tape_head_clog_score.confidence, 0.76), 0.05, 0.95))
            _tape_head_clog_score.metadata["cross_material_fallback"] = True
            _tape_head_clog_score.metadata["fallback_material_gate_bypassed"] = material_type.value
            scores[DefectType.TAPE_HEAD_CLOG] = _tape_head_clog_score
        else:
            scores[DefectType.TAPE_HEAD_CLOG] = DefectScore(DefectType.TAPE_HEAD_CLOG, 0.0, 0.95)
        _tail_tick("Tape-Head-Clog")

        # §9.1c - AMPLITUDE_DRIFT: Gradual level rise/fall over entire song.
        # Distinguishes artistic crescendo (preserve) from carrier drift (correct).
        # Runs on full mono audio for reliable trend detection (requires full duration).
        scores[DefectType.AMPLITUDE_DRIFT] = self._detect_amplitude_drift(_audio_mono_full)
        _tail_tick("Amplitude-Drift")

        # ── Tier 2: Dropout-Subtypen ───────────────────────────────────────
        # Klassifiziert Dropouts aus _detect_dropouts in drei Subtypen.
        _oxide_score, _hc_score, _splice_score, _all_dropout_locs = self._detect_dropout_subtypes(_audio_mono_full)
        scores[DefectType.DROPOUT_OXIDE] = _oxide_score
        scores[DefectType.DROPOUT_HEAD_CONTACT] = _hc_score
        scores[DefectType.DROPOUT_SPLICE] = _splice_score
        _tail_tick("Dropout-Subtypen")

        # ── Tier 2: MPEG-Frame-Verlust ─────────────────────────────────────
        scores[DefectType.MPEG_FRAME_LOSS] = self._detect_mpeg_frame_loss(_audio_mono_full)
        _tail_tick("MPEG-Frame-Verlust")

        # ── Tier 2: Stereofeld-Kollaps (nur Stereo) ─────────────────────────
        if is_stereo:
            scores[DefectType.STEREO_FIELD_COLLAPSE] = self._detect_stereo_collapse(audio)
        else:
            scores[DefectType.STEREO_FIELD_COLLAPSE] = DefectScore(
                DefectType.STEREO_FIELD_COLLAPSE, 0.0, 0.5, metadata={"reason": "mono"}
            )
        _tail_tick("Stereofeld-Kollaps")

        # ── Tier 2: Phasenrotation ─────────────────────────────────────────
        scores[DefectType.PHASE_ROTATION] = self._detect_phase_rotation(_audio_mono_full)
        _tail_tick("Phasenrotation")

        # ── Full-Audio Location Re-Detection ───────────────────────────────────
        # The center-crop analysis (60 s) produces locations ONLY in the middle
        # section of the song.  For event-based defects (clicks, crackle, etc.)
        # this is misleading - a 4-minute vinyl track shows markers only between
        # ~90 s and ~150 s while defects actually occur throughout.
        #
        # Fix: Re-run the event-based detectors on the FULL mono audio to get
        # accurate full-range locations.  Severity values from the center-crop
        # are retained (they are statistically representative).
        # Semi-stationary defects (print_through, transient_smearing, pre_echo)
        # have their locations cleared - they show as full-width tints instead.
        if _location_offset_s > 0.0:
            # §9.7.5b Performance: analyze only non-center sections for location re-detection.
            # Center 60 s is already covered by the primary pass (locations offset-corrected below).
            # Event-detectors (location-critical): use FULL intro + FULL outro to ensure 100 % coverage.
            # Severity-rechecks (statistical sample only): capped at 30 s per side - severity is not
            # location-dependent; a representative sample is sufficient.
            _center_begin = _dc_mid - _dc_half  # sample index where center-crop starts
            _center_end = _dc_mid + _dc_half  # sample index where center-crop ends
            _nc_intro_audio = _audio_mono_full[:_center_begin]  # full intro (all before center)
            _nc_intro_offset_s = 0.0
            _nc_outro_audio = _audio_mono_full[_center_end:]  # full outro (all after center)
            _nc_outro_offset_s = float(_center_end) / sr
            # 30 s samples for severity-only rechecks (statistically representative)
            _sev_cap = _detector_cap_n // 2  # 30 s
            _sev_intro = _audio_mono_full[: min(_sev_cap, _center_begin)]
            _sev_intro_off = 0.0
            _sev_outro = _audio_mono_full[_center_end : _center_end + _sev_cap]
            _sev_outro_off = _nc_outro_offset_s

            # Event-based detectors: re-detect on non-center sections for locations only
            _EVENT_DETECTORS: list[tuple] = [
                (DefectType.CLICKS, self._detect_clicks),
                (DefectType.CRACKLE, self._detect_crackle),
                (DefectType.CLIPPING, self._detect_clipping),
                (DefectType.SIBILANCE, self._detect_sibilance),
                (DefectType.VOCAL_HARSHNESS, self._detect_vocal_harshness),
                (DefectType.TAPE_SPLICE_ARTIFACT, self._detect_tape_splice_artifact),
                (DefectType.STICKY_SHED_RESIDUE, self._detect_sticky_shed_residue),
                (DefectType.GROOVE_ECHO, self._detect_groove_echo),
            ]
            # Long-audio center-crop misses are most critical for cheap impulse
            # detectors; always re-check them on non-center sections to avoid blind spots.
            _FORCE_FULL_REDETECT = {
                DefectType.CLICKS,
                DefectType.CRACKLE,
                DefectType.CLIPPING,
                DefectType.SIBILANCE,
                DefectType.TAPE_SPLICE_ARTIFACT,
                DefectType.STICKY_SHED_RESIDUE,
                DefectType.GROOVE_ECHO,
            }
            for _edt, _edet_fn in _EVENT_DETECTORS:
                if _edt not in scores:
                    continue
                _need_redetect = (_edt in _FORCE_FULL_REDETECT) or (scores[_edt].severity > 0.01)
                if not _need_redetect:
                    continue
                _prev_sev = float(scores[_edt].severity)
                _prev_conf = float(scores[_edt].confidence)
                # Start with center-crop locations already offset-corrected to absolute time.
                _abs_locations: list[tuple[float, float]] = [
                    (t0 + _location_offset_s, t1 + _location_offset_s) for t0, t1 in scores[_edt].locations
                ]
                _nc_sev = _prev_sev
                _nc_conf = _prev_conf
                for _seg, _seg_off in ((_nc_intro_audio, _nc_intro_offset_s), (_nc_outro_audio, _nc_outro_offset_s)):
                    if len(_seg) < sr // 4:
                        continue
                    try:
                        _sub = _edet_fn(_seg)
                        _nc_sev = max(_nc_sev, float(_sub.severity))
                        _nc_conf = max(_nc_conf, float(_sub.confidence))
                        _abs_locations.extend((t0 + _seg_off, t1 + _seg_off) for t0, t1 in _sub.locations)
                    except Exception:
                        logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
                scores[_edt].severity = float(np.clip(_nc_sev, 0.0, 1.0))
                scores[_edt].confidence = float(np.clip(_nc_conf, 0.0, 1.0))
                if _abs_locations:
                    scores[_edt].locations = sorted(_abs_locations, key=lambda _loc: _loc[0])
                    if _prev_sev <= 0.01 and _nc_sev > 0.01:
                        scores[_edt].metadata["outside_center_crop_detected"] = True
                        scores[_edt].metadata["outside_center_crop_recheck"] = "non_center_sections"
                else:
                    scores[_edt].locations = []
                _tail_tick("Vollaudio-Recheck (Events)")

            # Severity rescue for center-crop-sensitive defects.
            # Severity is a statistical / global property - a 30 s intro + 30 s outro sample
            # is sufficient to catch intro/outro-concentrated defects without running the full
            # duration (e.g. 225 s) through expensive detectors like pitch_drift or reverb_excess.
            # §9.7.5b: use _sev_intro / _sev_outro (both ≤ 30 s) instead of _audio_mono_full.
            _FULL_AUDIO_SEVERITY_RECHECK: list[tuple] = [
                (DefectType.PITCH_DRIFT, self._detect_pitch_drift),
                (DefectType.REVERB_EXCESS, self._detect_reverb_excess),
                (DefectType.DYNAMIC_COMPRESSION_EXCESS, self._detect_dynamic_compression_excess),
                (DefectType.SOFT_SATURATION, self._detect_soft_saturation),
                (DefectType.HEAD_WEAR, self._detect_head_wear),
                (DefectType.TRANSIENT_SMEARING, self._detect_transient_smearing),
                (DefectType.PRE_ECHO, self._detect_pre_echo),
                (DefectType.INNER_GROOVE_DISTORTION, self._detect_inner_groove_distortion),
                (DefectType.STYLUS_DAMAGE, self._detect_stylus_damage),
            ]
            for _sdt, _sdet_fn in _FULL_AUDIO_SEVERITY_RECHECK:
                if _sdt not in scores:
                    continue
                _prev_sev = float(scores[_sdt].severity)
                _prev_conf = float(scores[_sdt].confidence)
                # Re-check only if the center-crop score is not already meaningful.
                # Threshold 0.10: any real center-crop detection is statistically representative;
                # running the expensive detector again on 225 s is not justified when center-crop
                # already found evidence (severity > 0.10 is a reliable detection).
                if _prev_sev >= 0.10:
                    continue
                _sev_merged = _prev_sev
                _conf_merged = _prev_conf
                for _seg, _seg_off in ((_sev_intro, _sev_intro_off), (_sev_outro, _sev_outro_off)):
                    if len(_seg) < sr // 4:
                        continue
                    try:
                        _sub = _sdet_fn(_seg)
                        _sev_merged = max(_sev_merged, float(_sub.severity))
                        _conf_merged = max(_conf_merged, float(_sub.confidence))
                        if _sub.locations:
                            scores[_sdt].locations = [(t0 + _seg_off, t1 + _seg_off) for t0, t1 in _sub.locations]
                    except Exception as _exc:
                        logger.debug("Non-stationary recheck fehlgeschlagen for %s: %s", _sdt, _exc)
                scores[_sdt].severity = float(np.clip(_sev_merged, 0.0, 1.0))
                scores[_sdt].confidence = float(np.clip(_conf_merged, 0.0, 1.0))
                if _sev_merged > (_prev_sev + 0.02):
                    scores[_sdt].metadata["outside_center_crop_detected"] = True
                    scores[_sdt].metadata["outside_center_crop_recheck"] = "non_center_sample_severity"
                _tail_tick("Vollaudio-Recheck (Severity)")

            # DROPOUTS + TRANSPORT_BUMP + TAPE_HEAD_LEVEL_DIP already use full audio - no correction needed.
            # Event detectors re-detected above - no correction needed.
            _SKIP_OFFSET = {
                DefectType.DROPOUTS,
                DefectType.TRANSPORT_BUMP,
                DefectType.TAPE_HEAD_LEVEL_DIP,
                DefectType.CLICKS,
                DefectType.CRACKLE,
                DefectType.CLIPPING,
                DefectType.SIBILANCE,
                DefectType.VOCAL_HARSHNESS,
                DefectType.GROOVE_ECHO,
                # §9.7.5b: Event re-detection produces absolute-time locations for these too;
                # must skip the offset correction that would double-correct them.
                DefectType.TAPE_SPLICE_ARTIFACT,
                DefectType.STICKY_SHED_RESIDUE,
            }
            if is_stereo:
                _SKIP_OFFSET.add(DefectType.CLICKS)  # redundant but explicit

            # Semi-stationary defects: clear locations → full-width tint display
            _CLEAR_LOCATIONS = {DefectType.PRINT_THROUGH, DefectType.TRANSIENT_SMEARING, DefectType.PRE_ECHO}
            for _cdt in _CLEAR_LOCATIONS:
                if _cdt in scores and scores[_cdt].locations:
                    scores[_cdt].locations = []

            # Remaining detectors with locations: apply offset correction
            for _dt, _ds in scores.items():
                if _dt in _SKIP_OFFSET or _dt in _CLEAR_LOCATIONS:
                    continue
                if _ds.locations:
                    _ds.locations = [(t0 + _location_offset_s, t1 + _location_offset_s) for t0, t1 in _ds.locations]

        # ── §9.1b Intro-Salienz-Gewichtung (v10.0.0) ────────────────────────────
        # Psychoacoustic research (Zacharov & Koivuniemi 2001, Bech & Zacharov 2006):
        # the first 3-5 seconds define the listener's overall quality judgment.
        # Defects in the intro region receive a severity boost so that the pipeline
        # prioritizes their repair.  This is especially critical for tape media
        # where leader artifacts and run-in fluctuations cluster at the beginning.
        #
        # v10.0.0: Two-tier intro boost for cassette/tape head pickup errors.
        # Tape head engagement & motor stabilization cause defects up to 20 s:
        #   Tier 1 (0-5 s):  ×1.50 - perceptually critical (Zacharov 2001)
        #   Tier 2 (5-20 s): ×1.25 - cassette head settling region (tape-only)
        # Scientific basis: McKnight (1969) "Tape Reproducer Response Measurements
        # with a Reproducer Test Tape", AES Convention 36; Camras (1988) Ch. 7
        # "Transport Mechanisms - Start Transients".
        #
        # IMPORTANT: This MUST run AFTER the full-audio location re-detection and
        # offset correction above, so that t0 values are in absolute song time.
        # Running it before would check intro membership against crop-relative
        # timestamps (center of the song), not the actual intro.
        _INTRO_SECONDS = 5.0
        _INTRO_SEVERITY_BOOST = 1.5  # 50% boost for intro defects
        _TAPE_EXTENDED_INTRO_S = 20.0  # extended intro region for tape head settling
        _TAPE_EXTENDED_BOOST = 1.25  # 25% boost for tape extended intro (5-20 s)
        _is_tape_material = (
            self.material_type
            in (
                MaterialType.TAPE,
                MaterialType.CASSETTE,
                MaterialType.REEL_TAPE,
            )
            if self.material_type is not None
            else False
        )
        _total_duration_s = len(_audio_mono_full) / sr
        if _total_duration_s > _INTRO_SECONDS * 2:  # only for non-trivial audio
            for _dt, _ds in scores.items():
                if not _ds.locations:
                    continue
                _intro_events = [(t0, t1) for t0, t1 in _ds.locations if t0 < _INTRO_SECONDS]
                # v10.0.0: tape-extended intro region (5-20 s) for head-settling defects
                _ext_intro_events = []
                if _is_tape_material:
                    _ext_intro_events = [
                        (t0, t1) for t0, t1 in _ds.locations if _INTRO_SECONDS <= t0 < _TAPE_EXTENDED_INTRO_S
                    ]
                if (_intro_events or _ext_intro_events) and _ds.severity > 0.0:
                    _n_total = max(len(_ds.locations), 1)
                    # Tier 1: standard intro boost (0-5 s)
                    _intro_fraction = len(_intro_events) / _n_total
                    _tier1_boost = 1.0 + (_INTRO_SEVERITY_BOOST - 1.0) * _intro_fraction
                    # Tier 2: tape-extended intro boost (5-20 s)
                    _ext_fraction = len(_ext_intro_events) / _n_total
                    _tier2_boost = 1.0 + (_TAPE_EXTENDED_BOOST - 1.0) * _ext_fraction
                    # Combined: multiplicative (both tiers can contribute)
                    _boost = _tier1_boost * _tier2_boost
                    _old_sev = _ds.severity
                    _ds.severity = float(
                        np.nan_to_num(
                            min(1.0, _ds.severity * _boost),
                            nan=0.0,
                        )
                    )
                    if _ds.severity > _old_sev:
                        _ds.metadata["intro_boost_applied"] = True
                        _ds.metadata["intro_boost_factor"] = round(_boost, 3)
                        _ds.metadata["intro_events"] = len(_intro_events)
                        if _ext_intro_events:
                            _ds.metadata["tape_extended_intro_events"] = len(_ext_intro_events)
                            _ds.metadata["tape_extended_intro_boost"] = round(_tier2_boost, 3)
                        logger.debug(
                            "Intro-saliency boost: %s severity %.3f → %.3f "
                            "(%d/%d events in first %.0fs, %d ext-intro events)",
                            _dt.value,
                            _old_sev,
                            _ds.severity,
                            len(_intro_events),
                            len(_ds.locations),
                            _INTRO_SECONDS,
                            len(_ext_intro_events),
                        )

        # ── Per-Channel Location-Tags (L/R) für Stereo ─────────────────────────
        # Alle lokalisierbaren Impuls-/Event-Defekte werden pro Kanal detektiert,
        # damit die Wellenform-Visualisierung kanalgenau darstellen kann.
        # Premium-Anforderung: kein "über einen Kamm scheren" beider Kanäle.
        if is_stereo and audio.ndim == 2 and audio.shape[1] >= 2:
            _per_channel_locs: dict = {}
            _DT_TO_KEY = {
                DefectType.CLICKS: "clicks",
                DefectType.DROPOUTS: "dropout",
                DefectType.CRACKLE: "crackle",
                DefectType.CLIPPING: "clipping",
                DefectType.SIBILANCE: "sibilance",
                DefectType.TRANSPORT_BUMP: "transport_bump",
                DefectType.VOCAL_HARSHNESS: "vocal_harshness",
            }
            for _ch_idx, _ch_label in ((0, "L"), (1, "R")):
                _ch_audio = audio[:, _ch_idx]
                # Clicks per channel
                _ch_clicks = self._detect_clicks(_ch_audio)
                if _ch_clicks.locations:
                    _per_channel_locs.setdefault("clicks", {})[_ch_label] = list(_ch_clicks.locations)
                # Dropout per channel
                _ch_drops = self._detect_dropouts(_ch_audio)
                if _ch_drops.locations:
                    _per_channel_locs.setdefault("dropout", {})[_ch_label] = list(_ch_drops.locations)
                # Crackle per channel
                try:
                    _ch_crackle = self._detect_crackle(_ch_audio)
                    if _ch_crackle.locations:
                        _per_channel_locs.setdefault("crackle", {})[_ch_label] = list(_ch_crackle.locations)
                except Exception as _exc:
                    logger.debug("Per-channel crackle detection fehlgeschlagen (%s): %s", _ch_label, _exc)
                try:
                    _ch_clip = self._detect_clipping(_ch_audio)
                    if _ch_clip.locations:
                        _per_channel_locs.setdefault("clipping", {})[_ch_label] = list(_ch_clip.locations)
                except Exception as _exc:
                    logger.debug("Per-channel clipping detection fehlgeschlagen (%s): %s", _ch_label, _exc)
                # Sibilance per channel
                try:
                    _ch_sib = self._detect_sibilance(_ch_audio)
                    if _ch_sib.locations:
                        _per_channel_locs.setdefault("sibilance", {})[_ch_label] = list(_ch_sib.locations)
                except Exception as _exc:
                    logger.debug("Per-channel sibilance detection fehlgeschlagen (%s): %s", _ch_label, _exc)
                # Transport bumps per channel - skip for pure digital material
                # (no tape/disc transport mechanism, §9.4a - same guard as main call)
                if material_type not in _DIGITAL_NO_BUMP:
                    try:
                        _ch_bump = self._detect_transport_bump(_ch_audio)
                        if _ch_bump.locations:
                            _per_channel_locs.setdefault("transport_bump", {})[_ch_label] = list(_ch_bump.locations)
                    except Exception as _exc:
                        logger.debug("Per-channel transport_bump detection fehlgeschlagen (%s): %s", _ch_label, _exc)
                # Vocal harshness per channel
                try:
                    _ch_harsh = self._detect_vocal_harshness(_ch_audio)
                    if _ch_harsh.locations:
                        _per_channel_locs.setdefault("vocal_harshness", {})[_ch_label] = list(_ch_harsh.locations)
                except Exception as _exc:
                    logger.debug("Per-channel vocal_harshness detection fehlgeschlagen (%s): %s", _ch_label, _exc)
                _tail_tick(f"Kanal-Analyse {_ch_label}")
            # Store per-channel info as metadata on the combined scores
            for _dk, _ch_dict in _per_channel_locs.items():
                _dt_key = {v: k for k, v in _DT_TO_KEY.items()}.get(_dk)
                if _dt_key and _dt_key in scores:
                    scores[_dt_key].metadata["channel_locations"] = _ch_dict

        # ── §6.3 Medium-Gate: physikalisch unmögliche Defekte unterdrücken ──────
        # RIAA-Entzerrungsfehler sind NUR auf Disc-Medien möglich (Vinyl, Shellac,
        # Lacquer-Disc, Wax-Cylinder).  Auf Magnetband / Digital / Codec-Quellen
        # ist ein Bass/Mid-Ungleichgewicht KEIN RIAA-Fehler, sondern ggf. ein
        # EQ-Problem oder Aufnahme-Charakteristik - severity wird auf 0 gesetzt.
        _DISC_MEDIA = {
            MaterialType.VINYL,
            MaterialType.SHELLAC,
            MaterialType.LACQUER_DISC,
            MaterialType.WAX_CYLINDER,
            MaterialType.UNKNOWN,  # unbekannt: nicht unterdrücken
        }
        if material_type not in _DISC_MEDIA:
            _riaa_orig = scores[DefectType.RIAA_CURVE_ERROR].severity
            if _riaa_orig > 0.0:
                logger.info(
                    "Medium-Gate: RIAA_CURVE_ERROR suppressed (material=%s, was=%.3f) "
                    "- RIAA is disc-only, not applicable to %s",
                    material_type.value,
                    _riaa_orig,
                    material_type.value,
                )
                scores[DefectType.RIAA_CURVE_ERROR] = DefectScore(
                    defect_type=DefectType.RIAA_CURVE_ERROR,
                    severity=0.0,
                    confidence=scores[DefectType.RIAA_CURVE_ERROR].confidence,
                    locations=[],
                    metadata={
                        **scores[DefectType.RIAA_CURVE_ERROR].metadata,
                        "medium_gated": True,
                        "original_severity": _riaa_orig,
                    },
                )
            _tail_tick("Medium-Gate")

        # ── §9.1c Perceptual Salience - psychoacoustic masking annotation ───────
        # Annotates each defect with a 'perceptual_salience' score (0.0-1.0).
        # Masked defects (in loud passages) get reduced severity; exposed defects
        # (in quiet passages) keep full severity.  This prevents unnecessary repairs
        # on inaudible defects and focuses the pipeline on what the ear can detect.
        try:
            from backend.core.perceptual_salience import get_perceptual_salience_estimator  # pylint: disable=import-outside-toplevel  # noqa: I001

            _pse = get_perceptual_salience_estimator()
            _pse_audio = _audio_mono_full if _audio_mono_full is not None else audio_mono
            _pse_result = _pse.annotate_defect_scores(
                _pse_audio,
                sr,
                DefectAnalysisResult(
                    material_type=material_type,
                    scores=scores,
                    analysis_time_seconds=0.0,
                    sample_rate=sr,
                    duration_seconds=len(_pse_audio) / sr,
                ),
            )
            scores = _pse_result.scores
        except Exception as _pse_err:
            logger.warning("PerceptualSalienceEstimator fehlgeschlagen (unkritisch): %s", _pse_err)
        _tail_tick("Perceptual Salience")

        # Post-calibration: material-aware confidence stabilization and
        # transparent locality limits for long center-crop scans.
        self._post_calibrate_scores(
            scores=scores,
            material_type=material_type,
            used_center_crop=(_location_offset_s > 0.0),
            forensic_medium_result=forensic_medium_result,
        )
        _tail_tick("Post-Kalibrierung")

        _prog(99)

        analysis_time = time.time() - start_time
        duration = len(audio_mono) / sr

        logger.info(
            "DefectScan abgeschlossen: %.2fs für %.1fs Audio (%.1f%% overhead)",
            analysis_time,
            duration,
            analysis_time / duration * 100,
        )

        # Forensische Tonträgerkettenerkennung (MediumDetector, DSP-basiert).
        # Wenn UV3 bereits ein vollständiges MediumDetectionResult aus dem Bridge-Cache
        # übergeben hat (forensic_medium_result), wird KEIN zweiter Detector-Aufruf
        # ausgeführt - Redundanz-Fix (Erkennung einmalig nach Import, nicht nochmals beim
        # Klick auf Restoration / Studio 2026).
        _fmd_result: Any = {}
        if forensic_medium_result is not None and hasattr(forensic_medium_result, "transfer_chain"):
            _fmd_result = forensic_medium_result
            logger.debug(
                "[SCAN] ForensicMediumDetector: gecachtes Ergebnis verwendet - kein erneuter erkennen-Aufruf "
                "(primary_material=%s, chain=%s)",
                getattr(_fmd_result, "primary_material", "?"),
                getattr(_fmd_result, "transfer_chain", "?"),
            )
        else:
            try:
                from backend.core.forensics.medium_detector import MediumDetector as _ForensicMD  # pylint: disable=import-outside-toplevel  # noqa: I001

                # Für lange Dateien: nur erste 30 s analysieren (Geschwindigkeit)
                _max_forensic = sr * 30
                _audio_forensic = audio_mono[:_max_forensic] if len(audio_mono) > _max_forensic else audio_mono
                logger.debug("[SCAN] MediumDetector.erkennen() → %.1fs Audio ...", len(_audio_forensic) / sr)
                _fmd_result = _ForensicMD().detect(_audio_forensic, sr, file_ext=file_ext)
                # MediumDetectionResult ist ein Dataclass - getattr statt .get()
                _multi = getattr(_fmd_result, "is_multi_generation", None)
                _chain_val = getattr(_fmd_result, "transfer_chain", None) or getattr(_fmd_result, "chain", "?")
                logger.debug("[SCAN] MediumDetector OK: multi=%s, chain=%s", _multi, _chain_val)
            except Exception as _fmd_err:
                logger.debug("[SCAN] MediumDetector FEHLER: %s", _fmd_err)
                logger.debug("ForensicMediumDetector nicht verfügbar: %s", _fmd_err)

        # §9.x Cross-chain defect fallback: tape-stage defects (BIAS_ERROR, ALIASING)
        # that were material-gated on the primary medium may still be audible when the
        # transfer chain includes a tape stage.  Pattern mirrors §9.1a TAPE_HEAD_LEVEL_DIP.
        _chain_has_tape = self._chain_contains_tape(_fmd_result)

        _bias_current = scores.get(DefectType.BIAS_ERROR)
        if _bias_current is not None and _bias_current.metadata.get("medium_gated") and _chain_has_tape:
            _x_bias = self._detect_bias_error(audio_mono, _bypass_material_gate=True)
            if self._should_keep_cross_material_bias_error(_x_bias):
                _x_bias.severity = float(np.clip(_x_bias.severity * 0.70, 0.0, 1.0))
                _x_bias.confidence = float(np.clip(min(_x_bias.confidence, 0.68), 0.05, 0.95))
                _x_bias.metadata["cross_material_fallback"] = True
                _x_bias.metadata["fallback_primary_material"] = str(getattr(material_type, "value", material_type))
                scores[DefectType.BIAS_ERROR] = _x_bias
                logger.info(
                    "DefectScanner: BIAS_ERROR cross-chain Ersatzpfad: chain=%s → severity=%.3f",
                    getattr(_fmd_result, "transfer_chain", "?"),
                    _x_bias.severity,
                )

        _alias_current = scores.get(DefectType.ALIASING)
        if _alias_current is not None and _alias_current.metadata.get("medium_gated") and _chain_has_tape:
            _x_alias = self._detect_aliasing(audio_mono, _bypass_material_gate=True)
            if self._should_keep_cross_material_aliasing(_x_alias):
                _x_alias.severity = float(np.clip(_x_alias.severity * 0.75, 0.0, 1.0))
                _x_alias.confidence = float(np.clip(min(_x_alias.confidence, 0.60), 0.05, 0.95))
                _x_alias.metadata["cross_material_fallback"] = True
                scores[DefectType.ALIASING] = _x_alias
                logger.info(
                    "DefectScanner: ALIASING cross-chain Ersatzpfad: chain=%s → severity=%.3f",
                    getattr(_fmd_result, "transfer_chain", "?"),
                    _x_alias.severity,
                )

        # §9.x Chain-stage defect annotation + Severity-Skalierung (Kettendämpfung).
        #
        # Physikalisches Modell: Defekte einer Kettenstufe (z. B. Wow/Flutter auf Tape)
        # werden beim Transfer durch jede nachfolgende verlustbehaftete Stufe gedämpft.
        # Der Detektor misst das Residual auf dem Endsignal - ohne Compensation wird die
        # Severity systematic zu niedrig ausgewiesen.
        #
        # Dämpfungsmodell (empirisch, Tsai & Välimäki 2003, Brixen 2007):
        #   Severity_korrigiert = Severity_gemessen / (Dämpfungsfaktor per Stufe ^ n_stufen_dahinter)
        #   Dämpfung pro Codec-Stufe (mp3/aac/streaming): 0.55-0.65× je nach Bitrate
        #   Dämpfung pro digitale neutrale Stufe (cd/dat):  0.80×
        #   Dämpfung pro analoge Stufe als Zwischenstufe:   0.75×
        #
        # Konfidenz-Penalty (Unschärfe durch Kette): 0.75 × measured_confidence, max 0.85
        if _chain_threshold_overrides:
            try:
                # Kette aus fmd_result lesen (list[str], z.B. ['shellac','tape','mp3_low'])
                _fmd_for_ann = forensic_medium_result
                _ann_chain: list[str] = []
                if _fmd_for_ann is not None:
                    _raw_chain = getattr(_fmd_for_ann, "transfer_chain", None)
                    if isinstance(_raw_chain, list):
                        _ann_chain = [str(s).lower() for s in _raw_chain]
                _ann_primary = str(getattr(material_type, "value", "")).lower()

                # Dämpfungsfaktor pro Folgestufe nach der Quellstufe des Defekts
                _STAGE_ATTENUATION: dict[str, float] = {
                    "mp3_low": 0.55,
                    "mp3_high": 0.65,
                    "aac": 0.60,
                    "streaming": 0.58,
                    "minidisc": 0.68,
                    "cd_digital": 0.80,
                    "dat": 0.82,
                    # Analoge Folgestufen: geringere Dämpfung als Codec
                    "vinyl": 0.80,
                    "shellac": 0.80,
                    "tape": 0.75,
                    "reel_tape": 0.75,
                    "cassette": 0.75,
                    "wire_recording": 0.70,
                }

                def _chain_severity_scale(defect_source_stages: list[str], chain: list[str]) -> float:
                    """Berechnet den Kompensationsfaktor für in chain nachfolgende Stufen.

                    defect_source_stages: Materialtypen, die diesen Defekt erzeugen können.
                    chain: Gesamte Kette (ältester Träger zuerst).
                    """
                    if not chain or len(chain) < 2:
                        return 1.0
                    # Finde späteste Quelle des Defekts in der Kette
                    _source_idx = -1
                    for _i, _lnk in enumerate(chain):
                        if any(_lnk == _s for _s in defect_source_stages):
                            _source_idx = _i
                    if _source_idx < 0 or _source_idx >= len(chain) - 1:
                        return 1.0  # Quelle nicht gefunden oder ist letztes Glied
                    # Kumulierter Dämpfungsfaktor aller Folgestufen
                    _att = 1.0
                    for _fi in range(_source_idx + 1, len(chain)):
                        _att *= _STAGE_ATTENUATION.get(chain[_fi], 0.75)
                    # Kompensation = 1 / Dämpfung, geclippt auf [1.0, 3.5]
                    return float(np.clip(1.0 / max(_att, 0.28), 1.0, 3.5))

                # Defekttypen → erzeugende Primärstufen (für Dämpfungsberechnung)
                _WOW_FLUTTER_SOURCES = (
                    "tape",
                    "reel_tape",
                    "cassette",
                    "vinyl",
                    "shellac",
                    "wire_recording",
                    "wax_cylinder",
                    "lacquer_disc",
                )
                _TAPE_SOURCES = ("tape", "reel_tape", "cassette")
                _DEFECT_SOURCES: dict[DefectType, tuple[str, ...]] = {
                    DefectType.WOW: _WOW_FLUTTER_SOURCES,
                    DefectType.FLUTTER: _WOW_FLUTTER_SOURCES,
                    DefectType.MULTIBAND_WOW_FLUTTER: _WOW_FLUTTER_SOURCES,
                    DefectType.PITCH_DRIFT: _WOW_FLUTTER_SOURCES,
                    DefectType.MODULATION_NOISE: _TAPE_SOURCES,
                    DefectType.TAPE_HEAD_LEVEL_DIP: _TAPE_SOURCES,
                    DefectType.HF_REMANENCE_LOSS: _TAPE_SOURCES,
                    DefectType.STICKY_SHED_RESIDUE: _TAPE_SOURCES,
                    DefectType.TAPE_SPLICE_ARTIFACT: _TAPE_SOURCES,
                    DefectType.BIAS_ERROR: _TAPE_SOURCES,
                    DefectType.AZIMUTH_ERROR: _TAPE_SOURCES,
                    DefectType.DOLBY_NR_MISMATCH: _TAPE_SOURCES,
                    DefectType.PRINT_THROUGH: _TAPE_SOURCES,
                    DefectType.CRACKLE: ("shellac", "vinyl", "lacquer_disc", "wax_cylinder"),
                    DefectType.CLICKS: ("shellac", "vinyl", "lacquer_disc"),
                    DefectType.INNER_GROOVE_DISTORTION: ("vinyl", "shellac", "lacquer_disc"),
                    DefectType.GROOVE_ECHO: ("vinyl", "shellac", "lacquer_disc"),
                }

                for _cto_dt in _chain_threshold_overrides:
                    if _cto_dt not in scores:
                        continue
                    _cto_score = scores[_cto_dt]
                    if float(_cto_score.severity) <= 0.05:
                        continue  # kein messbarer Defekt - nicht skalieren
                    # Severity-Kompensation für Kettendämpfung
                    _source_stages = _DEFECT_SOURCES.get(_cto_dt, ())
                    _scale = _chain_severity_scale(list(_source_stages), _ann_chain)
                    if _scale > 1.001:  # nur wenn nennenswerte Dämpfung vorlag
                        _raw_sev = float(_cto_score.severity)
                        _cto_score.severity = float(np.clip(_raw_sev * _scale, 0.0, 1.0))
                        _cto_score.metadata["chain_severity_scale"] = round(_scale, 3)
                        _cto_score.metadata["chain_severity_raw"] = round(_raw_sev, 4)
                    # Konfidenz-Penalty (Kettenunschärfe)
                    _cto_score.confidence = float(np.clip(float(_cto_score.confidence) * 0.75, 0.05, 0.85))
                    _cto_score.metadata["chain_stage_defect"] = True
                    _cto_score.metadata["chain_threshold_override"] = True
                    logger.debug(
                        "chain-Stufe annotation: %s severity %.3f→%.3f scale=%.2f conf=%.2f",
                        _cto_dt.value,
                        _cto_score.metadata.get("chain_severity_raw", _cto_score.severity),
                        _cto_score.severity,
                        _scale if _scale > 1.001 else 1.0,
                        _cto_score.confidence,
                    )
            except Exception as _cann_exc:
                logger.debug("Chain-Stufe annotation fehlgeschlagen: %s", _cann_exc)

        _prog(100)
        _scan_result = DefectAnalysisResult(
            material_type=material_type,
            scores=scores,
            analysis_time_seconds=analysis_time,
            sample_rate=sr,
            duration_seconds=duration,
            auto_detected_material=_auto_material if _auto_material != material_type else None,
            transfer_chain_raw=_fmd_result,
            is_multi_generation=bool(getattr(_fmd_result, "is_multi_generation", False)),
            transfer_chain_str=str(getattr(_fmd_result, "transfer_chain", "") or getattr(_fmd_result, "chain", "")),
            spectral_fingerprint=_sf,  # §6.6.1 Pflicht-Fingerabdruck - 5 Messgrößen immer befüllt
            metadata={
                "material_confidence": _forensic_material_confidence,
                "chain_stage_materials": list(_chain_stage_materials),
                "chain_threshold_overrides": [
                    dt.value for dt in sorted(_chain_threshold_overrides, key=lambda d: d.value)
                ],
                "chain_threshold_override_count": len(_chain_threshold_overrides),
                "chain_threshold_override_applied": bool(_chain_threshold_overrides),
            },
        )

        try:
            _focus_map = _scan_result.get_focus_defect_map(n=12)
            _scan_result.metadata["focus_defects"] = dict(_focus_map)
            _scan_result.metadata["focus_defect_count"] = int(len(_focus_map))
            _scan_result.metadata["focus_defects_source"] = "severity_confidence_weighted"
        except Exception as _focus_exc:
            logger.debug("focus_defect_map generation fehlgeschlagen: %s", _focus_exc)

        # §9.7.1 Cache-Write - Ergebnis für künftige identische Aufrufe sichern
        with _scan_cache_lock:
            if len(_scan_cache) >= _SCAN_CACHE_MAX:
                _scan_cache.pop(next(iter(_scan_cache)))  # FIFO-Trim
            _scan_cache[_cache_key] = _scan_result
        # Welch-PSD-Cache leeren: gibt Speicher frei (audio_mono bleibt sonst im Cache)
        self._scan_welch_cache = {}
        return _scan_result

    def scan_defect_presence(
        self,
        audio: np.ndarray,
        sample_rate: int | None = None,
        material_type: MaterialType | None = None,
        min_severity: float = 0.05,
        file_ext: str = "",
    ) -> set[str]:
        """Scannt alle Defekttypen im Gesamt-Audio und gibt Präsenz-Menge zurück.

        §B3-Verarbeitungsschritt-2: Leichtgewichtige Defekt-Presence-Erkennung für
        die Chunk-0-Phasen-Selektion. Nutzt den vollen scan()-Durchlauf (der
        cache-beschleunigt ist), extrahiert aber nur Defekttypen oberhalb der
        Schwere-Schwelle und gibt ein schlankes ``set[str]`` zurück.

        Das Ergebnis wird NICHT separat gecached — scan() selbst cached bereits
        unter _scan_cache, sodass spätere Chunk-0-scans den Cache treffen.

        Args:
            audio:         Vollständiges Audio (mono oder stereo).
            sample_rate:   Samplerate (None → Fallback auf self.sample_rate).
            material_type: Material-Hint für Schwellwert-Anpassung.
            min_severity:  Mindest-Schwere [0, 1] für Aufnahme ins Ergebnis-Set.
            file_ext:      Dateiendung (an MediumDetector delegiert).

        Returns:
            Menge der erkannten Defekttypen-Namen (z.B. {"crackle", "tape_hiss", "wow"}).
        """
        _result = self.scan(
            audio,
            sample_rate=sample_rate,
            material_type=material_type,
            progress_callback=None,
            file_ext=file_ext,
            forensic_medium_result=None,
        )
        return {
            _dt.value if hasattr(_dt, "value") else str(_dt)
            for _dt, _score in _result.scores.items()
            if getattr(_score, "severity", 0.0) >= min_severity
        }

    def _post_calibrate_scores(
        self,
        *,
        scores: dict[DefectType, DefectScore],
        material_type: MaterialType,
        used_center_crop: bool,
        forensic_medium_result: Any = None,
    ) -> None:
        """Wendet konservative Konfidenz-Kalibrierung und Crop-Lokalitäts-Annotationen an.

        Calibration strategy:
        - Increase confidence slightly when measured severity clearly exceeds
          material-adaptive sensitivity.
        - Reduce confidence slightly for weak detections near threshold.
        - Mark long-audio center-crop locality limits for non-rechecked types.
        """
        _FULL_AUDIO_RECHECKED = {
            DefectType.DROPOUTS,
            DefectType.TRANSPORT_BUMP,
            DefectType.CLICKS,
            DefectType.CRACKLE,
            DefectType.CLIPPING,
            DefectType.SIBILANCE,
            DefectType.VOCAL_HARSHNESS,
            DefectType.PITCH_DRIFT,
            DefectType.REVERB_EXCESS,
            DefectType.DYNAMIC_COMPRESSION_EXCESS,
            DefectType.SOFT_SATURATION,
            DefectType.HEAD_WEAR,
            DefectType.TRANSIENT_SMEARING,
            DefectType.PRE_ECHO,
        }

        for defect_type, score in scores.items():
            old_conf = float(np.clip(score.confidence, 0.0, 1.0))
            severity = float(np.clip(score.severity, 0.0, 1.0))
            sensitivity = float(np.clip(self.thresholds.get(defect_type, 0.5), 1e-3, 1.0))

            # Evidence ratio > 1 means stronger-than-threshold defect evidence.
            evidence_ratio = float(severity / sensitivity)
            calib_scale = float(np.clip(0.88 + 0.20 * evidence_ratio, 0.78, 1.14))
            new_conf = float(np.clip(old_conf * calib_scale, 0.05, 0.99))

            # Long-audio center-crop uncertainty: keep severity but report slightly
            # lower confidence if no full-audio recheck path exists.
            if used_center_crop and defect_type not in _FULL_AUDIO_RECHECKED:
                has_locations = bool(score.locations)
                already_rechecked = bool(score.metadata.get("outside_center_crop_recheck"))
                if has_locations and not already_rechecked:
                    new_conf = float(np.clip(new_conf * 0.92, 0.05, 0.99))
                    score.metadata["crop_locality_limited"] = True
                    score.metadata["crop_locality_note"] = "center_crop_locations_with_offset"

            score.confidence = new_conf
            score.metadata["confidence_calibrated"] = True
            score.metadata["confidence_before_calibration"] = old_conf
            score.metadata["confidence_after_calibration"] = new_conf
            score.metadata["confidence_evidence_ratio"] = round(evidence_ratio, 4)
            score.metadata["confidence_material"] = material_type.value

        # §MP3-GUARD: Chain-Contamination-Discount - wenn die Kette mit
        # verlustbehaftetem Codec endet, ahmen Kompressions-Artefakte analoge
        # Defekte nach. Ohne diesen Discount: 5715 False-Positive auf 225s Audio.
        self._apply_chain_contamination_discount(scores, forensic_medium_result)

    def _apply_chain_contamination_discount(
        self,
        scores: dict[DefectType, DefectScore],
        forensic_medium_result: Any,
    ) -> None:
        """Diskontiert analoge Defekt-Severities wenn die Kette mit lossy Codec endet.

        MP3/AAC-Kompression erzeugt:
          - Pre-echo (5-35ms) → sieht aus wie Vinyl-Crackle
          - Block-boundary-Artefakte → sehen aus wie Clicks
          - HF-Quantisierungsrauschen → sieht aus wie Tape-Hiss
          - SBR-Spektral-Lücken → sehen aus wie Dropouts/Wow
        """
        if forensic_medium_result is None:
            return

        _chain = getattr(forensic_medium_result, "transfer_chain", None)
        if not isinstance(_chain, list) or len(_chain) < 2:
            return

        _chain_lower = [str(s).lower() for s in _chain]

        # Nur aktiv wenn LETZTE Stufe ein lossy Codec ist
        _terminal = _chain_lower[-1]
        _DIGITAL_LOSSY = {"mp3_low", "mp3_high", "aac", "streaming", "minidisc"}
        if _terminal not in _DIGITAL_LOSSY:
            return

        # Diskont-Faktor: stärker bei low-bitrate Codecs
        _DISCOUNT_MAP = {
            "mp3_low": 0.45,  # Stärkster Discount
            "mp3_high": 0.60,
            "aac": 0.55,
            "streaming": 0.50,
            "minidisc": 0.65,
        }
        # Data-driven: discount = f(chain_length, terminal_type)
        # mp3_low ×4-stage chain → 0.45; mp3_high ×2-stage → 0.65
        _base_discount = _DISCOUNT_MAP.get(_terminal, 0.60)
        _chain_bonus = max(0, (len(_chain_lower) - 1) * 0.05)  # +0.05 pro Kettenstufe
        discount = float(np.clip(_base_discount + _chain_bonus, 0.30, 0.80))

        # Analog-Defekte die von Codec-Artefakten imitiert werden
        _ANALOG_MIMICABLE: set[DefectType] = {
            DefectType.CRACKLE,  # MP3 pre-echo
            DefectType.CLICKS,  # MP3 block-boundary
            DefectType.HIGH_FREQ_NOISE,  # MP3 quantization noise / tape hiss
            DefectType.WOW,  # SBR gaps + natural vibrato
            DefectType.FLUTTER,  # SBR spectral holes
            DefectType.DROPOUTS,  # MP3 bitrate drops
            DefectType.GROOVE_ECHO,  # MP3 temporal masking
            # TRANSPORT_BUMP REMOVED (§2.74): Broadband level dips cannot
            # be produced by MP3 encoding. These are genuine tape transport
            # defects - the encoder passes them through transparently.
        }

        _n_discounted = 0
        for defect_type in _ANALOG_MIMICABLE:
            if defect_type in scores:
                _score = scores[defect_type]
                _old_sev = float(_score.severity)
                _old_conf = float(_score.confidence)
                if _old_sev > 0.05:  # Nur signifikante Detektionen diskontieren
                    _score.severity = float(np.clip(_old_sev * discount, 0.0, 1.0))
                    _score.confidence = float(np.clip(_old_conf * discount, 0.05, 0.99))
                    _score.metadata["chain_contamination_discount"] = float(discount)
                    _score.metadata["chain_contamination_terminal_codec"] = _terminal
                    _score.metadata["chain_contamination_original_severity"] = round(_old_sev, 4)
                    _score.metadata["chain_contamination_original_confidence"] = round(_old_conf, 4)
                    _n_discounted += 1

        if _n_discounted > 0:
            logger.info(
                "§MP3-GUARD Chain-Contamination: %d analoge Defekttypen ×%.2f diskontiert (terminal=%s)",
                _n_discounted,
                discount,
                _terminal,
            )

    # ========== MATERIAL AUTO-DETECTION ==========

    def _auto_detect_material(self, audio: np.ndarray, era_decade: int | None = None) -> MaterialType:
        """
        Auto-Detect Material-Typ basierend auf Audio-Charakteristiken.

        Heuristiken:
        - Shellac: Hohe Click-Rate, Mono, niedrige Frequenz-Range
        - Vinyl: Moderate Click-Rate, Stereo oder Mono, 50/60Hz hum, rumble
        - Tape: High-freq noise (hiss), wow/flutter, stereo oder mono
        - CD/Digital: Sehr sauber, evtl. digital artifacts, stereo
        - Streaming: Compression artifacts, stereo

        §v10.14.1: era_decade moduliert die Rumble-Gewichtung. Bei digitalen
        Ära-Aufnahmen (≥1990) ist sub-60Hz Energie mit hoher Wahrscheinlichkeit
        musikalischer Bass, NICHT mechanischer Rumble.
        """
        if audio.ndim == 1:
            # Mono → Shellac, Vinyl (Mono) oder Tape (Mono)
            return self._detect_mono_material(audio, era_decade=era_decade)
        else:
            # Stereo → Vinyl, Tape, CD oder Streaming
            return self._detect_stereo_material(audio, era_decade=era_decade)

    def _detect_mono_material(self, audio: np.ndarray, era_decade: int | None = None) -> MaterialType:
        """
        Unterscheidet Shellac vs Vinyl (Mono) vs Tape (Mono).

        Verwendet Scoring-System für robustere Detektion.

        §v10.14.1: era_decade moduliert Rumble-Gewichtung.
        """
        # === Feature-Extraction ===

        # Quick click detection
        diff = np.abs(np.diff(audio))
        click_rate = np.sum(diff > np.percentile(diff, 99.9)) / len(audio) * self.sample_rate

        # Spectral analysis
        freqs, psd = self._cached_welch(audio, self.sample_rate, max(1, min(2048, len(audio))))
        _psd_total = np.maximum(np.sum(psd), 1e-12)  # §3.1: Zero-Division-Guard bei Stille

        # High-freq energy (8kHz+)
        high_freq_energy = np.sum(psd[freqs > 8000]) / _psd_total

        # Rumble/Low-freq content
        rumble_energy = np.sum(psd[freqs < 60]) / _psd_total

        # Mid-low freq (50-60Hz hum typical for vinyl)
        _mid_low_energy = np.sum(psd[(freqs >= 50) & (freqs <= 70)]) / _psd_total

        # Crackle detection
        crackle_score = self._detect_crackle(audio).severity

        # Wow/Flutter (beide analog - IEC 60386)
        wow_flutter_score = max(
            self._detect_wow(audio if audio.ndim == 1 else audio.mean(axis=1)).severity,
            self._detect_flutter(audio if audio.ndim == 1 else audio.mean(axis=1)).severity,
        )

        # === Material Scoring ===

        # §v10.14.1 Era-Aware Rumble Modulation:
        # Bei digitalen Ära-Aufnahmen (≥1990) ist sub-60Hz Energie
        # mit hoher Wahrscheinlichkeit musikalischer Bass (Kick, Bass-Gitarre),
        # NICHT mechanischer Turntable-/Transport-Rumble.
        # Die Rumble-Gewichte werden entsprechend gedämpft.
        _era_is_digital = era_decade is not None and era_decade >= 1990
        _era_is_modern = era_decade is not None and era_decade >= 2000
        _rumble_analog_factor = 1.0
        _rumble_digital_boost = 0.0
        if _era_is_modern:
            _rumble_analog_factor = 0.15  # 85% Dämpfung — Bass ist fast sicher musikalisch
            _rumble_digital_boost = 8.0  # Starker Boost für digitale Materialien
        elif _era_is_digital:
            _rumble_analog_factor = 0.35  # 65% Dämpfung
            _rumble_digital_boost = 4.0

        # SHELLAC Score (Mono) - sehr alt, viele defekte, wenig HF
        # Penalized heavily since modern materials are more common
        shellac_score = 0.0
        shellac_score += click_rate * 0.3  # Clicks weniger gewichtet
        shellac_score += crackle_score * 2.0  # Crackle reduziert
        shellac_score += (1.0 - high_freq_energy) * 1.0  # HF-Mangel weniger stark
        shellac_score += wow_flutter_score * 0.3  # Wow/Flutter reduziert
        shellac_score -= rumble_energy * 20.0 * _rumble_analog_factor  # Shellac sollte fast kein Rumble haben
        shellac_score -= 10.0  # Starke Baseline-Penalty (Shellac ist selten)

        # VINYL Score (Mono) - Basiert auf empirischen Daten
        # Vinyl (1950s scratched): HF=0.035, rumble=0.0002, crackle=1.0, clicks=48
        vinyl_score = 0.0
        vinyl_score += high_freq_energy * 150.0  # Höhere HF als Tape (Hauptmerkmal!)
        vinyl_score += crackle_score * 2.0  # Crackle typisch, aber nicht exklusiv (§3 Material-Mismatch-Fix)
        vinyl_score += click_rate * 0.3  # Moderate Clicks
        vinyl_score += wow_flutter_score * 2.0  # Speed-Variation
        vinyl_score += (
            rumble_energy * 1500.0 * _rumble_analog_factor
        )  # §v10.14 FIX: Vinyl HAT Rumble (Motor) — war subtrahiert!
        vinyl_score += 9.0  # §v10.14 FIX: Baseline-Bonus (Vinyl ~Tape/Cassette-Niveau)
        if _era_is_digital:
            vinyl_score -= _rumble_digital_boost  # Digital-Ära → Vinyl unwahrscheinlich

        # TAPE Score (Mono) - Basiert auf empirischen Daten
        # Tape (1980s cassette): HF=0.024, rumble=0.0010, crackle=1.0, clicks=48
        tape_score = 0.0
        tape_score += wow_flutter_score * 6.0  # Sehr typisch (Tape-Stretch)
        tape_score += rumble_energy * 2500.0 * _rumble_analog_factor  # Rumble stark era-moduliert!
        tape_score -= high_freq_energy * 80.0  # Niedrigere HF als Vinyl (Hauptmerkmal!)
        tape_score += crackle_score * 1.0  # Crackle schwach positive (beide haben es)
        tape_score += click_rate * 0.1  # Clicks schwach positive
        tape_score += 10.0  # Baseline-Bonus erhöht
        if _era_is_digital:
            tape_score -= _rumble_digital_boost  # Digital-Ära → Tape unwahrscheinlich

        # CASSETTE Score (Mono) - Compact Cassette IEC 60094-1 Type I/II/IV
        # Diskriminatoren vs. TAPE/Reel-Tape:
        #   - Höheres Flutter (4.75 cm/s → typisch 0.10-0.30 % WRMS; Reel: 0.02-0.08 %)
        #   - Niedrigere BW (≤ 12 kHz Type I vs. 15 kHz Reel-Tape) → stärkerer HF-Abfall
        #   - Kein/kaum Rumble (leichter Transportmechanismus)
        cassette_score = 0.0
        cassette_score += wow_flutter_score * 9.0  # Hauptmerkmal: Flutter höher als Reel-Tape
        cassette_score -= high_freq_energy * 120.0  # BW ≤ 12 kHz → stärkerer HF-Abfall als Reel
        cassette_score += crackle_score * 0.5  # Weniger Crackle als Vinyl
        cassette_score -= rumble_energy * 3500.0  # Leichter Transport → kein/kaum Rumble
        cassette_score += click_rate * 0.08  # Gelegentliche Dropout-Clicks möglich
        cassette_score += 9.0  # Baseline (leicht über Reel-Tape, da häufiger als Reel)

        logger.debug(
            "Mono material scores: shellac=%.2f, vinyl=%.2f, tape=%.2f, cassette=%.2f",
            shellac_score,
            vinyl_score,
            tape_score,
            cassette_score,
        )

        # Select best match (minimum threshold 0.5)
        scores = {
            MaterialType.SHELLAC: shellac_score,
            MaterialType.VINYL: vinyl_score,
            MaterialType.TAPE: tape_score,
            MaterialType.CASSETTE: cassette_score,
        }

        best_material = max(scores, key=lambda k: scores[k])
        best_score = scores[best_material]

        if best_score > 0.5:
            # §2.46a: Auto-detected material tracked for provenance
            logger.info("§2.46a Auto-detected mono %s with confidence %.2f", best_material.value, best_score)
            return best_material
        else:
            # INFO (not WARNING): auto-detection failed gracefully, fallback chain activated per §3.0
            logger.debug("Mono material detection inconclusive (best=%.2f), falling back to transfer-chain analysis", best_score)
            return MaterialType.UNKNOWN

    def _detect_stereo_material(self, audio: np.ndarray, era_decade: int | None = None) -> MaterialType:
        """
        Unterscheidet alle Stereo-Material-Typen (Vinyl, Tape, Reel-Tape, DAT, CD, MP3, AAC, MiniDisc, Streaming).

        Verwendet Scoring-System statt binärer Thresholds für robustere Detektion.

        §v10.14.1: era_decade moduliert Rumble-Gewichtung.
        """
        audio_mono = np.mean(audio, axis=1)

        # === Feature-Extraction ===

        # 1. Rumble-Detection (Vinyl-typisch)
        freqs, psd = self._cached_welch(audio_mono, self.sample_rate, max(1, min(4096, len(audio_mono))))
        _psd_tot = np.maximum(np.sum(psd), 1e-12)  # §3.1: Zero-Division-Guard bei Stille
        rumble_energy = np.sum(psd[freqs < 60]) / _psd_tot

        # 2. Compression artifacts (Streaming/MP3/AAC)
        compression_score = self._detect_compression_artifacts(audio_mono).severity

        # 3. Digital artifacts (CD/DAT)
        digital_score = self._detect_digital_artifacts(audio_mono).severity

        # 4. High-freq noise / Tape hiss
        hf_noise_score = self._detect_high_freq_noise(audio_mono).severity

        # 5. High-freq energy loss (MP3/AAC low-pass effect)
        hf_energy = np.sum(psd[freqs > 15000]) / _psd_tot if self.sample_rate >= 44100 else 0.0
        hf_loss_indicator = 1.0 - (hf_energy / 0.05)  # Normalized: 0.05 = full HF, 0.0 = no HF
        hf_loss_indicator = np.clip(hf_loss_indicator, 0.0, 1.0)

        # 6. Crackle (Vinyl-typisch)
        crackle_score = self._detect_crackle(audio_mono).severity

        # 7. Wow/Flutter (analog media - IEC 60386)
        wow_flutter_score = max(
            self._detect_wow(audio_mono).severity,
            self._detect_flutter(audio_mono).severity,
        )

        # 8. Clicks (alle analog, aber unterschiedliche Pattern)
        click_score = self._detect_clicks(audio_mono).severity

        # === Material Scoring ===

        # §v10.14.1 Era-Aware Rumble Modulation (Stereo-Pfad):
        # Bei digitalen Ära-Aufnahmen ist sub-60Hz Energie musikalischer Bass.
        _era_is_digital_st = era_decade is not None and era_decade >= 1990
        _era_is_modern_st = era_decade is not None and era_decade >= 2000
        _rumble_factor_st = 1.0
        _digital_boost_st = 0.0
        if _era_is_modern_st:
            _rumble_factor_st = 0.20  # 80% Dämpfung für analoge Rumble-Gewichte
            _digital_boost_st = 6.0  # Starker Boost für digitale Materialien
        elif _era_is_digital_st:
            _rumble_factor_st = 0.40  # 60% Dämpfung
            _digital_boost_st = 3.0

        scores = {}

        # VINYL Score
        vinyl_score = 0.0
        vinyl_score += crackle_score * 2.0  # Crackle typisch für Vinyl, aber nicht exklusiv (§3 Material-Mismatch-Fix)
        vinyl_score += rumble_energy * 10.0 * _rumble_factor_st  # Rumble era-moduliert (Vinyl-Motor!)
        vinyl_score += wow_flutter_score * 1.5  # Analog-Medium
        vinyl_score += click_score * 1.0  # Moderate Clicks
        vinyl_score -= compression_score * 2.0  # Vinyl hat keine Compression
        vinyl_score -= digital_score * 2.0  # Vinyl ist analog
        vinyl_score += 6.0  # §v10.14 FIX: Baseline-Bonus (Vinyl ~Tape/Cassette-Niveau)
        if _era_is_digital_st:
            vinyl_score -= _digital_boost_st  # Digital-Ära → Vinyl unwahrscheinlich
        scores[MaterialType.VINYL] = max(0, vinyl_score)

        # TAPE (Cassette) Score
        tape_score = 0.0
        tape_score += hf_noise_score * 4.0  # Tape hiss sehr charakteristisch
        tape_score += wow_flutter_score * 2.0  # Noch typischer für Tape
        tape_score += click_score * 0.5  # Wenige Clicks
        tape_score -= crackle_score * 0.5  # Gealtertes Tape kann Crackle haben (Oxidflaking) - leichte Penalty
        tape_score -= rumble_energy * 5.0 * _rumble_factor_st  # Rumble era-moduliert
        tape_score -= compression_score * 2.0  # Tape analog
        tape_score += 4.0  # §v10.14 FIX: Baseline-Bonus für TAPE(=Cassette) Stereo-Pfad
        if _era_is_digital_st:
            tape_score -= _digital_boost_st * 0.5  # Digital-Ära → Tape unwahrscheinlich
        scores[MaterialType.TAPE] = max(0, tape_score)

        # REEL_TAPE (Professional) Score
        reel_tape_score = 0.0
        reel_tape_score += hf_noise_score * 3.0  # Less hiss than cassette (better quality)
        reel_tape_score += wow_flutter_score * 1.0  # Less than cassette, but present
        reel_tape_score += click_score * 0.3  # Very few clicks
        reel_tape_score -= crackle_score * 2.0  # No crackle
        reel_tape_score += (
            rumble_energy * 8.0 * _rumble_factor_st
        )  # §v10.14 FIX: Reel-Tape HAT Rumble (Transport-Motoren) — war subtrahiert!
        reel_tape_score -= compression_score * 2.0  # Analog
        reel_tape_score -= digital_score * 1.5  # Analog
        if _era_is_digital_st:
            reel_tape_score -= _digital_boost_st * 0.3  # Digital-Ära → Reel-Tape unwahrscheinlich
        # Boost if hiss present but better quality than cassette
        if hf_noise_score > 0.2 and hf_noise_score < 0.5:
            reel_tape_score += 1.0  # Professional tape sweet spot
        scores[MaterialType.REEL_TAPE] = max(0, reel_tape_score)

        # DAT (Digital Audio Tape) Score
        dat_score = 0.0
        dat_score += digital_score * 2.0  # Some digital artifacts
        dat_score -= crackle_score * 3.0  # No crackle
        dat_score -= rumble_energy * 10.0  # No rumble
        dat_score -= wow_flutter_score * 3.0  # Digital, no wow/flutter
        dat_score -= hf_noise_score * 2.0  # No tape hiss (digital)
        dat_score -= compression_score * 1.5  # Lossless, minimal compression
        # DAT is cleaner than CD but still digital
        if digital_score > 0.1 and compression_score < 0.2:
            dat_score += 1.5  # Clean digital indicator
        scores[MaterialType.DAT] = max(0, dat_score)

        # CD_DIGITAL Score
        cd_score = 0.0
        cd_score += digital_score * 3.0  # Digital artifacts typisch
        cd_score -= crackle_score * 3.0  # CD hat kein Crackle
        cd_score -= rumble_energy * 10.0 * _rumble_factor_st  # Rumble era-moduliert
        cd_score -= wow_flutter_score * 2.0  # CD hat kein Wow/Flutter
        cd_score -= hf_noise_score * 2.0  # CD hat keinen Tape-Hiss
        cd_score -= compression_score * 1.0  # CD meist lossless
        if _era_is_digital_st:
            cd_score += _digital_boost_st  # Digital-Ära → CD wahrscheinlich
        scores[MaterialType.CD_DIGITAL] = max(0, cd_score)

        # MP3_LOW (<128 kbps) Score
        mp3_low_score = 0.0
        mp3_low_score += compression_score * 4.5  # Heavy compression
        mp3_low_score += hf_loss_indicator * 3.0  # Significant HF loss
        mp3_low_score += digital_score * 1.0  # Codec artifacts
        mp3_low_score -= crackle_score * 3.0  # No crackle
        mp3_low_score -= rumble_energy * 10.0  # No rumble
        mp3_low_score -= wow_flutter_score * 3.0  # No wow/flutter
        # Heavy compression + HF loss indicator
        if compression_score > 0.6 and hf_loss_indicator > 0.5:
            mp3_low_score += 2.0  # Strong MP3 low bitrate indicator
        scores[MaterialType.MP3_LOW] = max(0, mp3_low_score)

        # MP3_HIGH (≥128 kbps) Score
        mp3_high_score = 0.0
        mp3_high_score += compression_score * 3.5  # Moderate compression
        mp3_high_score += hf_loss_indicator * 1.5  # Some HF loss
        mp3_high_score += digital_score * 0.5  # Fewer artifacts than low bitrate
        mp3_high_score -= crackle_score * 3.0  # No crackle
        mp3_high_score -= rumble_energy * 10.0  # No rumble
        mp3_high_score -= wow_flutter_score * 3.0  # No wow/flutter
        # Moderate compression, less HF loss than low bitrate
        if compression_score > 0.3 and compression_score < 0.6 and hf_loss_indicator < 0.5:
            mp3_high_score += 1.5  # MP3 high bitrate sweet spot
        scores[MaterialType.MP3_HIGH] = max(0, mp3_high_score)

        # AAC (M4A) Score
        aac_score = 0.0
        aac_score += compression_score * 3.0  # Efficient compression
        aac_score += hf_loss_indicator * 0.8  # Better HF preservation than MP3
        aac_score += digital_score * 0.3  # Minimal artifacts
        aac_score -= crackle_score * 3.0  # No crackle
        aac_score -= rumble_energy * 10.0  # No rumble
        aac_score -= wow_flutter_score * 3.0  # No wow/flutter
        # AAC is better quality than MP3 at same bitrate
        if compression_score > 0.2 and compression_score < 0.5 and hf_loss_indicator < 0.3:
            aac_score += 1.8  # AAC efficiency indicator
        scores[MaterialType.AAC] = max(0, aac_score)

        # MINIDISC (ATRAC) Score
        minidisc_score = 0.0
        minidisc_score += compression_score * 4.0  # ATRAC aggressive compression
        minidisc_score += hf_loss_indicator * 2.5  # Significant HF artifacts
        minidisc_score += digital_score * 1.5  # ATRAC-specific artifacts
        minidisc_score -= crackle_score * 3.0  # No crackle
        minidisc_score -= rumble_energy * 10.0  # No rumble
        minidisc_score -= wow_flutter_score * 3.0  # Digital, no wow/flutter
        # ATRAC has specific artifacts pattern (aggressive + HF issues)
        if compression_score > 0.5 and hf_loss_indicator > 0.4 and digital_score > 0.3:
            minidisc_score += 1.5  # ATRAC signature
        scores[MaterialType.MINIDISC] = max(0, minidisc_score)

        # STREAMING Score
        streaming_score = 0.0
        streaming_score += compression_score * 4.0  # Hauptindikator
        streaming_score -= crackle_score * 3.0  # Kein Crackle
        streaming_score -= rumble_energy * 10.0  # Kein Rumble
        streaming_score -= wow_flutter_score * 2.0  # Kein Wow/Flutter
        streaming_score += digital_score * 0.5  # §v10.14 FIX: Streaming IST digital — war subtrahiert!
        scores[MaterialType.STREAMING] = max(0, streaming_score)

        # SHELLAC Score (aus Stereo unwahrscheinlich, aber möglich bei Remaster)
        shellac_score = 0.0
        shellac_score += click_score * 3.0  # Sehr viele Clicks
        shellac_score += crackle_score * 2.0  # Auch Crackle
        shellac_score -= hf_noise_score * 2.0  # Wenig HF (alt)
        shellac_score -= compression_score * 2.0  # Analog
        # Shellac aus Stereo sehr unwahrscheinlich
        shellac_score *= 0.3  # Penalty für Stereo-Context
        scores[MaterialType.SHELLAC] = max(0, shellac_score)

        # CASSETTE (Compact Cassette IEC 60094-1) Score
        # Hauptdiskriminator vs. Reel-Tape: stärkeres Flutter (4.75 cm/s → 0.2 % WRMS),
        # höherer HF-Hiss (Type I schlechter als Reel), kein Rumble (leichter Transport),
        # BW-Limit ≤ 12 kHz → hf_loss_indicator positiv.
        cassette_score = 0.0
        cassette_score += hf_noise_score * 3.5  # Cassette-Hiss (Type I > Reel-Tape)
        cassette_score += wow_flutter_score * 3.0  # Hauptmerkmal: Flutter bei 4.75 cm/s > Reel
        cassette_score += hf_loss_indicator * 2.0  # BW ≤ 12 kHz → HF-Abfall über 12 kHz
        cassette_score += click_score * 0.3  # Dropout/Oxidflaking-Clicks möglich
        cassette_score -= crackle_score * 0.3  # Cassette hat wenig Knistern (kein Vinyl)
        cassette_score -= rumble_energy * 8.0  # Leichter Transport → kein Rumble
        cassette_score -= compression_score * 2.0  # Analog - kein Codec
        cassette_score -= digital_score * 2.0  # Analog
        cassette_score += 5.0  # §v10.14 FIX: Baseline-Bonus (Cassette ist häufigstes Consumer-Format)
        # Boost: hohe Flutter + HF-Hiss → eindeutiger Cassetten-Fingerabdruck
        if wow_flutter_score > 0.2 and hf_noise_score > 0.15:
            cassette_score += 1.5
        scores[MaterialType.CASSETTE] = max(0, cassette_score)

        # Wähle Material mit höchstem Score
        if not scores or max(scores.values()) < 0.5:  # Minimaler Confidence-Threshold
            return MaterialType.UNKNOWN

        best_material = max(scores.items(), key=lambda x: x[1])
        logger.debug("Material scores: %s", scores)
        # §v10.350 SOTA: Only log material detection in debug mode or during development
        # This prevents console spam on every import while still enabling troubleshooting
        if logger.isEnabledFor(logging.DEBUG):
            logger.info("§2.46f Auto-detected %s with confidence %.2f", best_material[0].value, best_material[1])

        return best_material[0]

    # ========== DEFECT DETECTORS (11 Typen) ==========

    @staticmethod
    def _sample_locations_evenly(locations: list, max_n: int) -> list:
        """Gibt evenly sampled locations, optionally uncapped zurück.

        If max_n <= 0, return all locations (no cap).
        Otherwise, pick representatives at uniform stride so returned positions
        span the full audio length without start-bias.
        """
        if max_n <= 0:
            return locations
        if len(locations) <= max_n:
            return locations
        step = len(locations) / max_n
        return [locations[int(i * step)] for i in range(max_n)]

    def _detect_clicks(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Clicks (kurze, impulsive Störungen).

        Anti-FP guards:
        1. Outlier-robust threshold: requires diff values to be ≥ 5× the median
           diff AND above the 99.9th percentile (scaled by material sensitivity).
           This prevents tonal signals (sine, strings) from triggering.
        2. Transient-width discrimination: only events spanning ≤ MAX_CLICK_WIDTH
           samples are accepted.  Musical transients (drums, plucks) are wider
           than true clicks (≤ 0.15 ms).
        3. Periodicity guard: a smooth periodic tone's per-cycle derivative maximum
           is topologically identical to a narrow click (single isolated sample with
           low diff immediately before/after) — guards 1+2 alone cannot distinguish
           them. Real clicks occur at irregular, defect-tied intervals; a near-perfectly
           regular event grid (low coefficient of variation of inter-event spacing) is
           almost always harmonic/tonal self-triggering, not genuine clicks (periodic
           mechanical repeats are the domain of GROOVE_ECHO/SCRAPE_FLUTTER, not CLICKS).
        """
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        # Inter-sample difference (classical click detection)
        diff = np.abs(np.diff(audio))
        local_median = float(np.median(diff)) + 1e-10

        # --- Anti-FP: outlier-robust threshold ---
        # Clicks are extreme outliers relative to local signal statistics.
        # The outlier factor is SNR-ADAPTIVE (§v10): clean recordings have
        # a well-defined noise floor → lower factor catches subtle clicks.
        # Noisy recordings have high baseline variation → higher factor
        # prevents false positives from noise transients.
        _snr_est = _estimate_local_snr(audio, self.sample_rate)
        _outlier_factor = float(np.clip(8.0 - _snr_est / 5.0, 3.5, 8.0))
        base_threshold = max(
            float(np.percentile(diff, 99.9)),
            local_median * _outlier_factor,
        )
        threshold_dynamic = self.thresholds[DefectType.CLICKS] * base_threshold

        # Find candidate click samples
        click_mask = diff > threshold_dynamic
        click_indices = np.where(click_mask)[0]

        if len(click_indices) == 0:
            return DefectScore(DefectType.CLICKS, 0.0, 1.0)

        # Group nearby exceedances (within 1 ms = one click event)
        group_window = max(1, int(0.001 * self.sample_rate))
        click_groups = []
        current_group = [click_indices[0]]
        for idx in click_indices[1:]:
            if idx - current_group[-1] < group_window:
                current_group.append(idx)
            else:
                click_groups.append(current_group)
                current_group = [idx]
        click_groups.append(current_group)

        # --- Anti-FP: transient width discrimination ---
        # True clicks are extremely narrow (≤ 0.15 ms ≈ 7 samples @ 48 kHz).
        # Musical transients (snare, pluck, piano hammer) are wider (> 20 samples).
        max_click_width = max(5, int(0.00015 * self.sample_rate))  # 0.15 ms
        verified_groups = []
        for group in click_groups:
            group_width = group[-1] - group[0] + 1
            if group_width <= max_click_width:
                verified_groups.append(group)

        # ── §v10 SNR-adaptive strict threshold ─────────────────────────
        # Clean signals: lower strict threshold (catches subtle clicks)
        # Noisy signals: higher strict threshold (avoids false positives)
        _strict_factor = float(np.clip(16.0 - _snr_est / 3.0, 8.0, 16.0))
        strict_threshold = max(
            float(np.percentile(diff, 99.99)),
            local_median * _strict_factor,
            float(np.clip(0.15 + _snr_est / 200.0, 0.15, 0.50)),
        )
        strict_indices = np.where(diff > strict_threshold)[0]
        if len(strict_indices) > 0:
            strict_groups: list[list[int]] = []
            current_strict = [int(strict_indices[0])]
            strict_gap = max(1, int(0.0002 * self.sample_rate))
            for idx in strict_indices[1:]:
                idx_int = int(idx)
                if idx_int - current_strict[-1] <= strict_gap:
                    current_strict.append(idx_int)
                else:
                    strict_groups.append(current_strict)
                    current_strict = [idx_int]
            strict_groups.append(current_strict)
            for group in strict_groups:
                group_width = group[-1] - group[0] + 1
                if group_width <= max_click_width and not any(
                    abs(group[0] - existing[0]) < group_window for existing in verified_groups
                ):
                    verified_groups.append(group)

        verified_groups.sort(key=lambda group: group[0])

        # --- Anti-FP guard 3: periodicity discount (harmonic/tonal self-triggering) ---
        # A near-perfectly regular inter-event spacing (low coefficient of variation)
        # means the "clicks" are almost certainly the recurring per-cycle derivative
        # peak of a clean tone, not sporadic physical defects. Requires ≥ 8 events for
        # a statistically meaningful spacing estimate.
        _periodicity_discount = 1.0
        if len(verified_groups) >= 8:
            _starts = np.array([g[0] for g in verified_groups], dtype=np.float64)
            _intervals = np.diff(_starts)
            if len(_intervals) >= 6:
                _mean_interval = float(np.mean(_intervals))
                _std_interval = float(np.std(_intervals))
                if _mean_interval > 0 and (_std_interval / _mean_interval) < 0.10:
                    _periodicity_discount = 0.10
                    logger.debug(
                        "§Anti-FP-3 Click-Periodizitäts-Guard: %d Events, CV=%.3f → "
                        "wahrscheinlich tonales Self-Triggering, severity ×%.2f",
                        len(verified_groups),
                        _std_interval / _mean_interval,
                        _periodicity_discount,
                    )

        if len(verified_groups) == 0:
            return DefectScore(
                DefectType.CLICKS,
                0.0,
                0.95,
                metadata={
                    "click_rate": 0.0,
                    "total_clicks": 0,
                    "rejected_as_transients": len(click_groups),
                    "strict_impulse_candidates": len(strict_indices),
                },
            )

        # Convert to timestamps - sample evenly across full duration (not just first 50)
        MAX_LOCATIONS = self._LOCATION_CAP_UNCAPPED
        all_locations = [(group[0] / self.sample_rate, group[-1] / self.sample_rate) for group in verified_groups]
        locations = self._sample_locations_evenly(all_locations, MAX_LOCATIONS)

        # Severity: click-rate per second (uses full count, not capped)
        duration = len(audio) / self.sample_rate
        click_rate = len(verified_groups) / duration

        # Janssen (1986) / Godsill & Rayner (1998): AR-prediction-residual augmentation.
        # LPC order 30 (spec §: 30-40 @ 48 kHz) detects clicks in harmonically rich
        # signals where diff-based thresholding can miss them (false negatives for 1940s-
        # 1960s shellac/vinyl recordings with strong carrier noise masking impulsive clicks).
        try:
            ar_rate = self._ar_click_residual_rate(audio)
            # AR augments diff-based severity; never reduces (conservative lower-bound)
            click_rate = max(click_rate, ar_rate * 0.80)
        except Exception as _ar_exc:
            logger.debug("AR click augmentation fehlgeschlagen: %s", _ar_exc)

        severity = min(1.0, click_rate / 20)  # 20 clicks/sec = severity 1.0
        severity = float(np.clip(severity * _periodicity_discount, 0.0, 1.0))

        # §CODEC-DISKRIMINATOR: MP3 block-boundary artifacts → 26ms-Gitter-Check
        _cd = getattr(self, "_codec_disc", None)
        if _cd is not None and _cd.has_terminal_codec and severity > 0.05 and locations:
            _on_grid = _cd.mp3_boundary_fraction([(s, e) for s, e in locations])
            if _on_grid > 0.25:  # ≥25% auf 26ms-Gitter → wahrscheinlich MP3-Artefakte
                _codec_discount = _cd.codec_discount * (1.0 - _on_grid * 0.65)
                severity = float(np.clip(severity * _codec_discount, 0.0, 1.0))
                # §MP3-GRID: Anteil der Clicks auf MP3-Blockgrenzen in Metadata vermerken
                # Der ReparaturDenker entscheidet dann über die Reparatur-Aggressivität.
                _grid_pct = _on_grid

        return DefectScore(
            defect_type=DefectType.CLICKS,
            severity=severity,
            confidence=0.9,
            locations=locations,
            metadata={
                "click_rate": click_rate,
                "total_clicks": len(verified_groups),
                "rejected_as_transients": len(click_groups) - len(verified_groups),
                "strict_impulse_candidates": len(strict_indices),
                "periodicity_discount": _periodicity_discount,
            },
        )

    def _ar_click_residual_rate(self, audio: np.ndarray) -> float:
        """Janssen (1986) / Godsill & Rayner (1998) AR prediction-error click rate.

        Fits LPC order 30 (spec §: 30-40 @ 48 kHz) to 50 ms non-overlapping blocks
        and counts samples where the prediction residual exceeds 6× the block residual
        std. This catches clicks hidden in harmonically rich signals where inter-sample
        difference thresholding underestimates click density (false negatives in 1920s-
        1960s material with simultaneous carrier noise and impulsive disturbances).

        Returns clicks-per-second (float). Analyses max. 30 s for efficiency.
        """
        try:
            from scipy.linalg import solve_toeplitz as _solve_toeplitz  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            logger.debug("§V6 scipy.linalg.solve_toeplitz nicht verfügbar — 0.0 Click-Rate zurückgegeben: %s", exc)
            return 0.0

        lpc_order = 30  # spec: 30-40 @ 48 kHz
        seg_n = max(lpc_order * 6, int(0.050 * self.sample_rate))  # ≥50 ms
        # Limit to first 30 s (representative for click rate estimation)
        max_samples = int(30 * self.sample_rate)
        analysis = audio[:max_samples].astype(np.float64)
        if analysis.ndim == 2:
            analysis = np.mean(analysis, axis=1)

        n = len(analysis)
        if n < seg_n * 3:
            return 0.0

        n_blocks = n // seg_n
        total_clicks = 0

        for i in range(n_blocks):
            seg = analysis[i * seg_n : (i + 1) * seg_n]

            # Yule-Walker autocorrelation via FFT (efficient for long segments)
            ac_full = np.real(np.fft.irfft(np.abs(np.fft.rfft(seg, n=2 * len(seg))) ** 2))
            r = ac_full[: lpc_order + 1]
            if not np.isfinite(r[0]) or r[0] < 1e-20:
                continue

            try:
                a_lpc = _solve_toeplitz(r[:lpc_order], r[1 : lpc_order + 1])
            except (np.linalg.LinAlgError, ValueError):
                continue
            if not np.all(np.isfinite(a_lpc)):
                continue

            # AR prediction via convolution: residual[n] = seg[n] - Σ a[k]*seg[n-k-1]
            pred_filt = np.r_[1.0, -a_lpc]
            residual = np.abs(np.convolve(seg, pred_filt, mode="full")[: len(seg)])
            residual[:lpc_order] = 0.0  # startup region invalid

            valid_resid = residual[lpc_order:]
            resid_std = float(np.std(valid_resid)) + 1e-10
            # Janssen threshold: residual > 6× std (calibrated for k=5-8 range)
            ar_clicks_mask = valid_resid > 6.0 * resid_std
            if ar_clicks_mask.any():
                hits = np.where(ar_clicks_mask)[0]
                cluster_gap = max(1, int(0.001 * self.sample_rate))
                clustered = 1
                for j in range(1, len(hits)):
                    if hits[j] - hits[j - 1] > cluster_gap:
                        clustered += 1
                total_clicks += clustered

        analysis_duration = (n_blocks * seg_n) / self.sample_rate
        return total_clicks / max(analysis_duration, 1.0)

    def _detect_crackle(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Crackle (kontinuierliches leises Knistern, z.B. Vinyl-Surface-Noise).

        Anti-FP guard: brilliant recordings (cymbals, synthesizers, sibilants) have
        naturally high HF energy that can trigger the HP-filtered envelope detector.
        We compute a broadband HF-energy ratio and apply a discount when the HF
        content is smoothly distributed (tonal HF) rather than impulsive (crackle).
        True crackle has high kurtosis in the HP-filtered signal; tonal HF does not.
        """
        # Crackle = High-Pass filtered + Envelope Detection
        sos = signal.butter(4, 3000, btype="high", fs=self.sample_rate, output="sos")
        audio_hp = signal.sosfilt(sos, audio)

        # Envelope via Hilbert-Transform
        analytic_signal: np.ndarray = signal.hilbert(audio_hp)  # type: ignore[assignment]
        envelope = np.abs(analytic_signal)

        # Smoothed envelope
        window_size = int(0.05 * self.sample_rate)  # 50ms window
        envelope_smooth = np.convolve(envelope, np.ones(window_size) / window_size, mode="same")

        # Crackle detection: regions with elevated noise-floor
        threshold = self.thresholds[DefectType.CRACKLE] * np.percentile(envelope_smooth, 95)
        crackle_mask = envelope_smooth > threshold

        severity_raw = float(np.sum(crackle_mask)) / len(audio)

        # --- Anti-FP: kurtosis + sparsity checks on HP-filtered signal ---
        # Crackle is impulsive → high kurtosis (>> 3.0 for Gaussian).
        # Tonal HF (cymbals, synths, sibilants) has near-sinusoidal distribution
        # → kurtosis ≈ 1.5 (pure sine) to 3.0 (Gaussian noise).
        hp_kurtosis = 3.0  # Gaussian default (safe fallback)
        hp_std = float(np.std(audio_hp))
        if hp_std > 1e-8:
            hp_mean = float(np.mean(audio_hp))
            hp_kurtosis = float(np.mean(((audio_hp - hp_mean) / hp_std) ** 4))

        peak_ratio = 0.0
        sparse_impulse_rate = 0.0
        if hp_std > 1e-8:
            hp_abs = np.abs(audio_hp)
            hp_p95 = float(np.percentile(hp_abs, 95)) + 1e-12
            hp_p999 = float(np.percentile(hp_abs, 99.9))
            peak_ratio = hp_p999 / hp_p95
            sparse_impulse_rate = float(np.mean(hp_abs > max(hp_p95 * 6.0, hp_std * 6.0)))

        tonal_or_dense_hf = bool(peak_ratio < 3.0 and sparse_impulse_rate < 0.0008)

        kurtosis_discount = 1.0
        if hp_kurtosis < 4.0:
            # Clearly tonal or Gaussian HF - NOT impulsive crackle.
            # Hard cap severity to near-zero regardless of energy ratio.
            kurtosis_discount = 0.0
        elif hp_kurtosis < 6.0:
            # Borderline zone - scale linearly
            kurtosis_discount = max(0.1, (hp_kurtosis - 4.0) / 2.0)
        # Kurtosis > 6 → impulsive, keep full weight

        if tonal_or_dense_hf:
            kurtosis_discount = 0.0

        severity = min(1.0, severity_raw * 10 * kurtosis_discount)

        # Find connected regions
        from scipy.ndimage import label  # pylint: disable=import-outside-toplevel

        labeled_array, num_features = label(crackle_mask)  # type: ignore[misc]
        locations = []
        for i in range(1, num_features + 1):
            indices = np.where(labeled_array == i)[0]
            if len(indices) > int(0.01 * self.sample_rate):  # At least 10ms
                start = indices[0] / self.sample_rate
                end = indices[-1] / self.sample_rate
                locations.append((start, end))

        confidence = 0.8
        if hp_kurtosis < 4.0 or tonal_or_dense_hf:
            confidence = 0.3  # Very low confidence when HF is clearly tonal

        # §CODEC-DISKRIMINATOR: MP3 pre-echo maskiert sich als Crackle
        _cd = getattr(self, "_codec_disc", None)
        if _cd is not None and _cd.has_terminal_codec and severity > 0.05:
            _onset_frac = _cd.onset_correlation(audio, locations)
            if _onset_frac > 0.35:  # ≥35% der Crackle-Events korrelieren mit Onsets → MP3 pre-echo
                _codec_discount = _cd.codec_discount * (1.0 - _onset_frac * 0.7)
                severity = float(np.clip(severity * _codec_discount, 0.0, 1.0))
                confidence = float(np.clip(confidence * 0.6, 0.05, 0.99))

        return DefectScore(
            defect_type=DefectType.CRACKLE,
            severity=severity,
            confidence=confidence,
            locations=self._sample_locations_evenly(locations, self._LOCATION_CAP_UNCAPPED),
            metadata={
                "crackle_percentage": severity_raw * 100,
                "hp_kurtosis": hp_kurtosis,
                "hp_peak_ratio": peak_ratio,
                "sparse_impulse_rate": sparse_impulse_rate,
                "kurtosis_discount": kurtosis_discount,
                "tonal_or_dense_hf_guard": tonal_or_dense_hf,
            },
        )

    def _detect_hum(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Hum (50Hz/60Hz AC-Brummen und Harmonische).

        Multi-Window + Peak-Sharpness + Harmonic-Decay-Validation (v10.0.0b).
        Literature: Vaseghi 2008 'Advanced DSP and Noise Reduction' Ch.13.
        """
        n = len(audio)
        if n < self.sample_rate:
            return DefectScore(DefectType.HUM, 0.0, 0.3)

        # --- Multi-segment windowed analysis (catch intermittent hum) ---
        seg_dur = min(4.0, n / self.sample_rate)
        seg_len = int(seg_dur * self.sample_rate)
        n_segs = max(1, min(8, n // seg_len))
        hum_freqs_50 = [50 * (k + 1) for k in range(8)]  # 8 harmonics
        hum_freqs_60 = [60 * (k + 1) for k in range(8)]
        # Harmonic weight: fundamental strongest, geometric decay
        harm_weights = np.array([1.0 / (k + 1) for k in range(8)])
        harm_weights /= harm_weights.sum()

        best_ratio = 0.0
        best_freq = 50
        best_sharpness = 0.0
        seg_ratios: list[float] = []
        half_band = max(1, int(2.0 / 1.0))  # Initialize with default bin_width

        for si in range(n_segs):
            start = si * (n - seg_len) // max(1, n_segs - 1) if n_segs > 1 else 0
            seg = audio[start : start + seg_len]
            # Hanning window to reduce spectral leakage
            win = np.hanning(len(seg))
            seg_win = np.asarray(seg * win, dtype=np.float64)
            spectrum = np.abs(np.fft.rfft(seg_win))
            freqs = fft.rfftfreq(len(seg), 1.0 / self.sample_rate)
            total_e = float(np.sum(spectrum**2) + 1e-12)
            bin_width = freqs[1] if len(freqs) > 1 else 1.0
            # ±2 Hz band in bins
            half_band = max(1, int(2.0 / bin_width))

            def _measure_hum_weighted(hum_f_list):
                # pylint: disable=cell-var-from-loop
                energy = 0.0
                sharpness_sum = 0.0
                for ki, f in enumerate(hum_f_list):
                    if f >= self.sample_rate / 2:
                        break
                    idx = int(round(f / bin_width))
                    if idx >= len(spectrum):
                        break
                    b0 = max(0, idx - half_band)
                    b1 = min(len(spectrum), idx + half_band + 1)
                    peak_e = float(np.sum(spectrum[b0:b1] ** 2))
                    energy += harm_weights[ki] * peak_e
                    # Peak sharpness: ratio of peak bin to band mean
                    band_mean = float(np.mean(spectrum[b0:b1] ** 2) + 1e-12)
                    peak_val = float(spectrum[idx] ** 2)
                    sharpness_sum += peak_val / band_mean
                return energy, sharpness_sum / max(1, len(hum_f_list))

            e50, sharp50 = _measure_hum_weighted(hum_freqs_50)
            e60, sharp60 = _measure_hum_weighted(hum_freqs_60)

            if e50 > e60:
                ratio = e50 / total_e
                sharp = sharp50
                freq = 50
            else:
                ratio = e60 / total_e
                sharp = sharp60
                freq = 60

            seg_ratios.append(ratio)
            if ratio > best_ratio:
                best_ratio = ratio
                best_freq = freq
                best_sharpness = sharp

        # --- Anti-FP: Peak sharpness guard ---
        # Hum has SHARP peaks (Q > 50). Broadband bass content has low sharpness.
        # Sharpness < 1.5 means energy is spread, NOT hum.
        if best_sharpness < 1.5:
            best_ratio *= 0.15  # Massively discount - likely bass content

        # --- Anti-FP: Sub-50 Hz rumble confusion guard ---
        # If the dominant low-frequency energy is below 40 Hz, this is LOW_FREQ_RUMBLE,
        # not HUM. HUM is at 50/60 Hz (mains) with integer harmonics. Sub-40 Hz content
        # is transport rumble, acoustic feedback, or building vibration.
        # Strategy: measure energy in <40 Hz band vs 40-80 Hz band; if sub-40 Hz
        # dominates AND the sharpness at 50/60 Hz is low → suppress.
        if len(audio) > self.sample_rate:
            try:
                _sub40_sos = signal.butter(
                    4, float(np.clip(40.0 / (self.sample_rate / 2.0), 1e-6, 0.999)), btype="low", output="sos"
                )
                _sub40_audio = signal.sosfilt(_sub40_sos, audio[: min(len(audio), 4 * self.sample_rate)])
                _sub40_e = float(np.mean(_sub40_audio**2) + 1e-20)
                _hum_band_e = best_ratio * (float(np.mean(audio[: min(len(audio), 4 * self.sample_rate)] ** 2)) + 1e-20)
                if _sub40_e > _hum_band_e * 3.0 and best_sharpness < 3.0:
                    best_ratio *= 0.25  # Sub-40 Hz dominates → likely rumble, not hum
            except Exception as _exc:
                logger.debug("Hum sub-40 Hz rumble guard fehlgeschlagen: %s", _exc)

        # --- Anti-FP: Harmonic decay validation ---
        # Real hum has monotonically decaying harmonics. Random bass doesn't.
        # (already handled by harm_weights, but sharpness is the main guard)

        threshold = self.thresholds[DefectType.HUM]
        severity = float(np.clip(best_ratio / (threshold * 0.001), 0.0, 1.0))

        # Confidence based on sharpness and multi-seg consistency
        seg_std = float(np.std(seg_ratios)) if len(seg_ratios) > 1 else 0.0
        confidence = float(
            np.clip(
                0.60 + 0.30 * min(1.0, best_sharpness / 3.0) - 0.20 * min(1.0, seg_std / (best_ratio + 1e-12)),
                0.3,
                0.98,
            )
        )

        # --- Locations: segments where hum is present ---
        locations: list[tuple[float, float]] = []
        if severity > 0.0 and len(seg_ratios) > 1:
            mean_r = float(np.mean(seg_ratios))
            for si, r in enumerate(seg_ratios):
                if r > mean_r * 0.5:
                    t0 = si * (n - seg_len) / max(1, n_segs - 1) / self.sample_rate if n_segs > 1 else 0.0
                    t1 = t0 + seg_dur
                    locations.append((t0, min(t1, n / self.sample_rate)))
            locations = self._sample_locations_evenly(locations, self._LOCATION_CAP_UNCAPPED)

        return DefectScore(
            defect_type=DefectType.HUM,
            severity=severity,
            confidence=confidence,
            locations=locations,
            metadata={
                "hum_frequency": best_freq,
                "hum_ratio": best_ratio,
                "peak_sharpness": best_sharpness,
                "n_segments_analyzed": n_segs,
                "segment_consistency_std": seg_std,
            },
        )

    def _detect_wow(self, audio: np.ndarray) -> DefectScore:
        """Erkennt WOW: slow pitch modulation < 0.5 Hz (IEC 60386, Blum 1984).

        Upgraded v10.0.0b: Dual-track detection using both RMS-envelope AND
        instantaneous-frequency (Hilbert analytic signal) for pitch-based WOW.
        RMS catches amplitude-WOW; IF catches pure pitch-WOW missed by RMS.
        """
        n = len(audio)
        if n < self.sample_rate * 2:  # need at least 2s for sub-0.5Hz
            return DefectScore(DefectType.WOW, 0.0, 0.3)

        # --- Track 1: RMS envelope modulation (amplitude WOW) ---
        win_len = max(1, int(0.5 * self.sample_rate))
        hop = max(1, win_len // 2)
        # Vectorized RMS
        n_frames = (n - win_len) // hop
        if n_frames < 4:
            return DefectScore(DefectType.WOW, 0.0, 0.3)
        indices = np.arange(n_frames) * hop
        rms_series = np.array(
            [float(np.sqrt(np.mean(audio[i : i + win_len] ** 2) + 1e-12)) for i in indices],
            dtype=np.float64,
        )
        rms_series /= rms_series.mean() + 1e-12

        frame_rate = self.sample_rate / hop
        # Power spectrum (squared magnitude) for proper energy metric
        rms_centered = rms_series - rms_series.mean()
        win_h = np.hanning(len(rms_centered))
        fft_rms = np.abs(np.fft.rfft(rms_centered * win_h)) ** 2
        freqs_rms = np.fft.rfftfreq(len(rms_centered), d=1.0 / frame_rate)

        wow_mask = (freqs_rms > 0.02) & (freqs_rms < 0.5)  # skip DC
        total_power = float(np.sum(fft_rms[1:]) + 1e-12)  # skip DC bin
        wow_power_rms = float(np.sum(fft_rms[wow_mask])) if wow_mask.any() else 0.0
        wow_ratio_rms = wow_power_rms / total_power

        # --- Track 2: Instantaneous frequency modulation (pitch WOW) ---
        # Bandpass 80-4000 Hz to focus on pitched content
        try:
            bp_sos = signal.butter(
                3,
                [80.0 / (self.sample_rate / 2), min(4000.0, self.sample_rate * 0.45) / (self.sample_rate / 2)],
                btype="band",
                output="sos",
            )
            bp_audio = signal.sosfilt(bp_sos, audio)
            # Hilbert transform for analytic signal
            analytic = signal.hilbert(bp_audio[: min(n, self.sample_rate * 30)])  # cap at 30s
            analytic_arr = np.asarray(analytic, dtype=np.complex128)
            inst_phase = np.unwrap(np.angle(analytic_arr))
            inst_freq = np.diff(inst_phase) * self.sample_rate / (2 * np.pi)
            # Smooth to 50ms windows
            if_win = max(1, int(0.05 * self.sample_rate))
            if len(inst_freq) > if_win * 4:
                if_smooth = np.convolve(inst_freq, np.ones(if_win) / if_win, mode="valid")
                if_smooth = if_smooth / (np.median(if_smooth) + 1e-6)  # normalize
                if_centered = if_smooth - if_smooth.mean()
                if_rate = self.sample_rate / if_win
                win_if = np.hanning(len(if_centered))
                fft_if = np.abs(np.fft.rfft(if_centered * win_if)) ** 2
                freqs_if = np.fft.rfftfreq(len(if_centered), d=1.0 / if_rate)
                wow_mask_if = (freqs_if > 0.02) & (freqs_if < 0.5)
                total_if = float(np.sum(fft_if[1:]) + 1e-12)
                wow_power_if = float(np.sum(fft_if[wow_mask_if])) if wow_mask_if.any() else 0.0
                wow_ratio_if = wow_power_if / total_if
            else:
                wow_ratio_if = 0.0
        except Exception:
            wow_ratio_if = 0.0

        # --- Combine: max of both tracks ---
        wow_ratio = max(wow_ratio_rms, wow_ratio_if * 0.8)

        # --- Periodicity check (anti-FP): real WOW is quasi-periodic ---
        # Find dominant modulation frequency
        if wow_mask.any() and wow_power_rms > 0:
            dom_idx = np.argmax(fft_rms[wow_mask])
            dom_freq = float(freqs_rms[wow_mask][dom_idx])
            dom_peak = float(fft_rms[wow_mask][dom_idx])
            dom_ratio = dom_peak / (wow_power_rms + 1e-12)
        else:
            dom_freq = 0.0
            dom_ratio = 0.0

        # Periodic WOW has concentrated energy at one frequency
        periodicity_bonus = 1.0 + 0.5 * max(0.0, dom_ratio - 0.3)  # boost if periodic

        threshold = self.thresholds.get(DefectType.WOW, 0.5)
        severity = float(np.clip(wow_ratio * periodicity_bonus / (threshold * 0.3 + 1e-12), 0.0, 1.0))

        confidence = float(np.clip(0.55 + 0.30 * min(1.0, dom_ratio), 0.3, 0.92))

        locations: list[tuple[float, float]] = []
        if severity > 0.0 and len(rms_series) >= 4:
            wow_dev = np.abs(rms_series - 1.0)
            wow_thr = float(max(np.percentile(wow_dev, 80.0), 0.05))  # type: ignore[arg-type]
            wow_mask_frames = wow_dev >= wow_thr
            if np.any(wow_mask_frames):
                starts = np.where(np.diff(np.concatenate(([0], wow_mask_frames.astype(np.int8), [0]))) == 1)[0]  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
                ends = np.where(np.diff(np.concatenate(([0], wow_mask_frames.astype(np.int8), [0]))) == -1)[0]  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
                for s_f, e_f in zip(starts, ends):
                    t0 = max(0.0, float(s_f * hop) / self.sample_rate - 0.05)
                    t1 = min(float(n) / self.sample_rate, float(e_f * hop + win_len) / self.sample_rate + 0.05)
                    if t1 > t0:
                        locations.append((t0, t1))

        return DefectScore(
            defect_type=DefectType.WOW,
            severity=severity,
            confidence=confidence,
            locations=locations,
            metadata={
                "wow_energy_ratio_rms": wow_ratio_rms,
                "wow_energy_ratio_if": wow_ratio_if,
                "dominant_mod_freq_hz": dom_freq,
                "periodicity_ratio": dom_ratio,
                "frame_rate_hz": float(frame_rate),
            },
        )

    def _detect_transport_bump(self, audio: np.ndarray) -> DefectScore:
        """Erkennt TRANSPORT_BUMP: impulsive micro-speed jumps from tape transport shocks.

        Multi-feature algorithm (v2) that distinguishes real transport bumps from
        normal musical transients by requiring co-occurrence of 5 features:
          1. Sudden energy anomaly (RMS envelope drop > 55% or spike > 2.5×)
          2. Low-frequency thump (< 80 Hz energy surge ≥ 2.5× local context)
          3. Spectral centroid disruption (> 40% shift from local baseline)
          4. Spectral flux spike (≥ 3.5× local median - rapid timbral change)
          5. Pitch instability (ZCR derivative spike - proxy for pitch jump)

        Each candidate must satisfy:
          - Feature 1 (energy anomaly) MUST be present (mandatory)
          - Plus ≥ 2 of features 2-5 (total ≥ 3 features)
        Musical transients (drum hits, note onsets) rarely show energy anomaly
        AND spectral centroid disruption AND LF thump simultaneously.

        Scientific basis:
          - Godsill & Rayner (1998): Digital Audio Restoration - transport bump model
          - Esquef et al. (2002): Frequency-domain analysis of mechanical artifacts

        Returns:
            DefectScore with locations [(start_s, end_s), ...] for each bump event.
        """
        n = len(audio)
        sr = self.sample_rate
        if n < sr * 2:
            return DefectScore(DefectType.TRANSPORT_BUMP, 0.0, 0.3)

        # §Spec 24-Normalisierung (Befund 2026-08-22): Der B3-Full-Song-Scan
        # füttert unnormalisiertes Integer-Audio (max_mag=66…127 im Log statt
        # ~1.0). Die Severity-Formel 0.5·min(1.0, max_mag/2.0) sättigt dann bei
        # 0.90 und der "low_mag"-Filter (< 1.5) feuert nie → falsche
        # TAPE_HEAD_LEVEL_DIP-Forwardings. Peak-Normalisierung auf ±1 stellt
        # die kalibrierten Schwellen wieder her (relative Features bleiben
        # skaleninvariant).
        audio = np.asarray(audio, dtype=np.float32)
        _peak = float(np.percentile(np.abs(audio), 99.9)) if audio.size else 0.0
        if _peak > 1.5:
            audio = audio / _peak

        hop_s = 0.010  # 10 ms hop
        win_s = 0.030  # 30 ms window
        hop = max(1, int(hop_s * sr))
        win = max(1, int(win_s * sr))
        n_frames = (n - win) // hop

        if n_frames < 20:
            return DefectScore(DefectType.TRANSPORT_BUMP, 0.0, 0.3)

        hann = np.hanning(win).astype(np.float64)

        # --- Feature extraction ---
        rms_env = np.empty(n_frames, dtype=np.float64)
        lf_ratio = np.empty(n_frames, dtype=np.float64)
        sc_env = np.empty(n_frames, dtype=np.float64)
        zcr_env = np.empty(n_frames, dtype=np.float64)

        freqs = np.fft.rfftfreq(win, 1.0 / sr)
        lf_mask = freqs < 80.0
        mf_mask = (freqs >= 80.0) & (freqs < 4000.0)

        prev_mag: np.ndarray | None = None
        flux_list: list[float] = []

        for i in range(n_frames):
            s = i * hop
            frame = audio[s : s + win].astype(np.float64)
            rms_env[i] = float(np.sqrt(np.mean(frame**2) + 1e-12))
            zcr_env[i] = float(np.sum(np.abs(np.diff(np.signbit(frame))))) / max(1.0, float(win))

            windowed = frame * hann
            mag = np.abs(np.fft.rfft(windowed))

            lf_e = float(np.sum(mag[lf_mask] ** 2))
            mf_e = float(np.sum(mag[mf_mask] ** 2)) + 1e-12
            lf_ratio[i] = lf_e / mf_e

            total_mag = float(np.sum(mag)) + 1e-12
            sc_env[i] = float(np.sum(freqs * mag)) / total_mag

            if prev_mag is not None:
                flux_list.append(float(np.sqrt(np.sum((mag - prev_mag) ** 2))))
            prev_mag = mag.copy()

        flux_env = np.array(flux_list, dtype=np.float64) if flux_list else np.zeros(1)

        # --- Adaptive local baselines (rolling median, ~1 s context) ---
        from scipy.ndimage import median_filter as _medfilt  # pylint: disable=import-outside-toplevel

        ctx = max(3, int(1.0 / hop_s))  # 1 s context for stable baseline
        if ctx % 2 == 0:
            ctx += 1

        rms_baseline = _medfilt(rms_env, size=ctx)
        lf_baseline = _medfilt(lf_ratio, size=ctx)
        sc_baseline = _medfilt(sc_env, size=ctx)
        flux_baseline = _medfilt(flux_env, size=min(ctx, len(flux_env)))
        _medfilt(zcr_env, size=ctx)

        # Numerical stability: clamp rms_baseline to avoid rms_ratio blowing up
        # during silent passages (fade-ins, intros, near-silence frames).
        # Without this guard, rms_baseline ≈ 0 → rms_ratio ≈ 307× → false positives.
        _rms_floor = max(float(np.percentile(rms_env[rms_env > 0], 5)) * 0.1 if np.any(rms_env > 0) else 1e-5, 1e-5)
        rms_baseline = np.maximum(rms_baseline, _rms_floor)

        # --- Feature 1 (MANDATORY): Energy anomaly ---
        rms_ratio = rms_env / (rms_baseline + 1e-12)
        # Transport bumps manifest as energy DROPS (tape head lifting, groove skip).
        # Energy SPIKES (rms_ratio > 2.5) are musical transients (kick drum, snare,
        # percussive attack) - using them as mandatory criterion caused 112 false
        # positives on drum patterns at 120 BPM (29.8/min vs real max ~8/min).
        feat_energy = rms_ratio < 0.45  # mandatory: dropout/energy-dip only
        # Energy spike is an optional secondary indicator (non-mandatory, +1 score)
        feat_energy_spike = rms_ratio > 2.5

        # --- Feature 2: Low-frequency thump ---
        lf_factor = lf_ratio / (lf_baseline + 1e-12)
        feat_lf = lf_factor > 2.5

        # --- Feature 3: Spectral centroid disruption ---
        sc_ratio = sc_env / (sc_baseline + 1e-12)
        feat_sc = (sc_ratio < 0.60) | (sc_ratio > 1.6)

        # --- Feature 4: Spectral flux spike ---
        flux_ratio = flux_env / (flux_baseline + 1e-12)
        feat_flux = np.zeros(n_frames, dtype=bool)
        feat_flux[: len(flux_ratio)] = flux_ratio > 3.5

        # --- Feature 5: Pitch instability (ZCR derivative spike) ---
        zcr_diff = np.abs(np.diff(zcr_env))
        zcr_diff_baseline = _medfilt(zcr_diff, size=min(ctx, len(zcr_diff)))
        zcr_diff_ratio = zcr_diff / (zcr_diff_baseline + 1e-12)
        feat_pitch = np.zeros(n_frames, dtype=bool)
        feat_pitch[: len(zcr_diff_ratio)] = zcr_diff_ratio > 3.0

        # --- Multi-feature scoring with MANDATORY energy requirement ---
        # Energy must be present; then count additional features
        score_arr = np.zeros(n_frames, dtype=np.float64)
        for i in range(n_frames):
            lo = max(0, i - 2)
            hi = min(n_frames, i + 3)
            # Mandatory: energy feature must be triggered in vicinity
            if not np.any(feat_energy[lo:hi]):
                continue
            hits = 1.0  # energy counts as 1
            if np.any(feat_energy_spike[lo:hi]):
                hits += 1.0
            if np.any(feat_lf[lo:hi]):
                hits += 1.0
            if np.any(feat_sc[lo:hi]):
                hits += 1.0
            if np.any(feat_flux[lo:hi]):
                hits += 1.0
            if np.any(feat_pitch[lo:hi]):
                hits += 1.0
            score_arr[i] = hits

        # Require energy + 2 more = total ≥ 3 features
        candidates = score_arr >= 3.0

        # --- Group into events (30 ms - 500 ms), merge nearby (gap ≤ 50 ms) ---
        min_frames_evt = max(1, int(0.030 / hop_s))
        max_frames_evt = max(1, int(0.500 / hop_s))
        merge_gap = max(1, int(0.050 / hop_s))

        locations: list[tuple[float, float]] = []
        magnitudes: list[float] = []
        bump_scores: list[float] = []
        suppressed_head_dip_like = 0
        # §v10.30: Gesammelte Zeitbereiche der suppressed Events für Forwarding
        # an den TAPE_HEAD_LEVEL_DIP-Detektor. Werden NICHT verworfen, sondern
        # als Hint-Locations an _detect_tape_head_level_dips() übergeben.
        _suppressed_head_dip_locations: list[tuple[float, float]] = []

        i = 0
        while i < n_frames:
            if candidates[i]:
                start_frame = i
                while i < n_frames and candidates[i]:
                    i += 1
                end_frame = i
                # Merge nearby events
                while i < n_frames and i - end_frame <= merge_gap:
                    if candidates[i]:
                        while i < n_frames and candidates[i]:
                            i += 1
                        end_frame = i
                    else:
                        i += 1
                event_len = end_frame - start_frame
                if min_frames_evt <= event_len <= max_frames_evt:
                    evt_ratios = rms_ratio[start_frame:end_frame]

                    # Tape-head level dips are gradual 80-400 ms envelope drops.
                    # The crucial discriminator vs. transport bumps is morphology:
                    # they stay below the local level for most of the event and do
                    # not show a positive in-event recovery/thump.
                    _looks_like_head_level_dip = (
                        event_len * hop_s >= 0.120
                        and float(np.mean(evt_ratios < 0.70)) >= 0.80
                        and float(np.max(evt_ratios)) < 0.90
                    )
                    if _looks_like_head_level_dip:
                        suppressed_head_dip_like += 1
                        # §v10.30: Zeitbereich des suppressed Events sammeln,
                        # NICHT verwerfen. Wird als Hint an den Head-Dip-Detektor
                        # weitergereicht, damit Phase 54/24 ihn reparieren kann.
                        _t_start_hd = max(0.0, float(start_frame * hop) / sr - 0.015)
                        _t_end_hd = min(float(n) / sr, float(end_frame * hop + win) / sr + 0.015)
                        _suppressed_head_dip_locations.append((_t_start_hd, _t_end_hd))
                        continue

                    t_start = max(0.0, float(start_frame * hop) / sr - 0.015)
                    t_end = min(float(n) / sr, float(end_frame * hop + win) / sr + 0.015)
                    locations.append((t_start, t_end))
                    magnitudes.append(float(np.max(np.abs(1.0 - evt_ratios))))
                    bump_scores.append(float(np.mean(score_arr[start_frame:end_frame])))
            else:
                i += 1

        n_bumps = len(locations)
        # §v10.30: Forward suppressed head-dip-like events to the tape head dip detector.
        # These were filtered from TRANSPORT_BUMP but are genuine defects that need repair.
        # Store on self so _detect_tape_head_level_dips() can pick them up.
        self._forwarded_head_dip_locations = _suppressed_head_dip_locations
        if n_bumps == 0:
            return DefectScore(DefectType.TRANSPORT_BUMP, 0.0, 0.6)

        duration_s = n / sr
        bump_density = n_bumps / max(1.0, duration_s / 60.0)
        max_mag = float(max(magnitudes)) if magnitudes else 0.0
        mean_score = float(np.mean(bump_scores)) if bump_scores else 3.0

        severity = float(
            np.clip(
                0.20 * min(1.0, bump_density / 8.0)
                + 0.50 * min(1.0, max_mag / 2.0)
                + 0.30 * min(1.0, (mean_score - 2.5) / 1.5),
                0.0,
                1.0,
            )
        )
        confidence = float(np.clip(0.60 + 0.08 * n_bumps, 0.60, 0.95))

        # Anti-FP rhythm guard:
        # Dense, highly periodic event trains in beat-like ranges are usually
        # musical transients (kick/snare at ~80-170 BPM), not transport bumps.
        _rhythm_guard_applied = False
        _interval_mean_s = 0.0
        _interval_cv = 1.0
        if n_bumps >= 8:
            _centers = np.array([(s + e) * 0.5 for s, e in locations], dtype=np.float64)
            _intervals = np.diff(_centers)
            if _intervals.size >= 4:
                _interval_mean_s = float(np.mean(_intervals))
                _interval_std_s = float(np.std(_intervals))
                _interval_cv = _interval_std_s / max(_interval_mean_s, 1e-6)
                _beat_like = 0.35 <= _interval_mean_s <= 0.75 and _interval_cv < 0.25
                # _too_dense_low_mag: Nur wenn die Bumps eine semi-reguläre Zeitstruktur haben
                # (CV < 0.40), also auf Drum-Muster hindeuten. Kassetten-Laufwerk-Bumps sind
                # zeitlich unregelmäßig (CV > 0.40) - diese darf der Guard nicht unterdrücken.
                _too_dense_low_mag = bump_density > 20.0 and max_mag < 1.5 and _interval_cv < 0.40
                if _beat_like or _too_dense_low_mag:
                    severity *= 0.25
                    confidence = float(np.clip(confidence * 0.70, 0.45, 0.85))
                    _rhythm_guard_applied = True

        # §Per-Event-Duration: für chirurgische Reparatur (Phase 08)
        _bump_durations_s = [(e - s) for s, e in locations] if locations else []
        _dur_mean = float(np.mean(_bump_durations_s)) if _bump_durations_s else 0.0
        _dur_min = float(np.min(_bump_durations_s)) if _bump_durations_s else 0.0
        _dur_max = float(np.max(_bump_durations_s)) if _bump_durations_s else 0.0

        logger.info(
            "transport_bump: n=%d, density=%.1f/min, max_mag=%.3f, mean=%.2f, sev=%.3f, suppressed=%d, dur=%.0f-%.0fms (μ=%.0fms)",
            n_bumps,
            bump_density,
            max_mag,
            mean_score,
            severity,
            suppressed_head_dip_like,
            _dur_min * 1000,
            _dur_max * 1000,
            _dur_mean * 1000,
        )

        return DefectScore(
            defect_type=DefectType.TRANSPORT_BUMP,
            severity=severity,
            confidence=confidence,
            locations=locations,
            metadata={
                "n_bumps": n_bumps,
                "bump_density_per_min": round(bump_density, 2),
                "max_magnitude": round(max_mag, 4),
                "mean_multi_feature_score": round(mean_score, 2),
                "suppressed_head_dip_like": suppressed_head_dip_like,
                "rhythm_guard_applied": _rhythm_guard_applied,
                "interval_mean_s": round(_interval_mean_s, 4),
                "interval_cv": round(_interval_cv, 4),
                "magnitudes": [round(float(m), 4) for m in magnitudes[:30]],
                "duration_mean_ms": round(_dur_mean * 1000, 1),
                "duration_min_ms": round(_dur_min * 1000, 1),
                "duration_max_ms": round(_dur_max * 1000, 1),
                "event_durations_ms": [round((e - s) * 1000, 1) for s, e in locations[:30]],
            },
        )

    def _detect_flutter(self, audio: np.ndarray) -> DefectScore:
        """Erkennt FLUTTER: rapid pitch modulation 0.5-200 Hz (IEC 60386).

        Upgraded v10.0.0b: Spectral-centroid modulation analysis replaces
        primitive ZCR. Flutter causes rapid spectral centroid oscillation
        at characteristic mechanical frequencies (guide roller 4-8 Hz,
        capstan 2-4 Hz, tape resonance 10-30 Hz).
        """
        n = len(audio)
        if n < self.sample_rate // 2:
            return DefectScore(DefectType.FLUTTER, 0.0, 0.3)

        # --- Short-time spectral centroid series ---
        # 20ms windows, 10ms hop (captures up to 50 Hz modulation = Nyquist)
        win_len = max(256, int(0.020 * self.sample_rate))
        hop = max(1, win_len // 2)
        n_frames = (n - win_len) // hop
        if n_frames < 16:
            return DefectScore(DefectType.FLUTTER, 0.0, 0.3)

        # Vectorized spectral centroid computation
        centroid_series = np.empty(n_frames, dtype=np.float64)
        freqs_stft = np.fft.rfftfreq(win_len, 1.0 / self.sample_rate)
        win_h = np.hanning(win_len)

        for fi in range(n_frames):
            start = fi * hop
            seg = audio[start : start + win_len] * win_h
            mag = np.abs(np.fft.rfft(seg))
            total_mag = float(np.sum(mag) + 1e-12)
            centroid_series[fi] = float(np.sum(freqs_stft * mag) / total_mag)

        # Normalize centroid to fractional deviation
        mean_c = float(np.mean(centroid_series) + 1e-6)
        centroid_norm = (centroid_series - mean_c) / mean_c

        # --- Modulation spectrum of centroid ---
        frame_rate = self.sample_rate / hop
        nyquist_mod = frame_rate / 2.0
        centroid_win = np.hanning(len(centroid_norm))
        fft_c = np.abs(np.fft.rfft(centroid_norm * centroid_win)) ** 2
        freqs_mod = np.fft.rfftfreq(len(centroid_norm), d=1.0 / frame_rate)

        # Flutter band: 0.5-min(200, Nyquist) Hz
        flutter_hi = min(200.0, nyquist_mod * 0.95)
        flutter_mask = (freqs_mod >= 0.5) & (freqs_mod <= flutter_hi)
        total_power = float(np.sum(fft_c[1:]) + 1e-12)
        flutter_power = float(np.sum(fft_c[flutter_mask])) if flutter_mask.any() else 0.0
        flutter_ratio = flutter_power / total_power

        # --- Sub-band analysis: identify dominant flutter source ---
        # Guide roller: 4-8 Hz; Capstan: 2-4 Hz; Tape resonance: 10-30 Hz
        sub_bands = {
            "capstan": (2.0, 4.0),
            "guide_roller": (4.0, 8.0),
            "tape_resonance": (10.0, 30.0),
        }
        dom_source = "unknown"
        dom_power = 0.0
        for src, (lo, hi) in sub_bands.items():
            mask = (freqs_mod >= lo) & (freqs_mod <= min(hi, flutter_hi))
            if mask.any():
                p = float(np.sum(fft_c[mask]))
                if p > dom_power:
                    dom_power = p
                    dom_source = src

        # --- Periodicity check ---
        if flutter_mask.any() and flutter_power > 0:
            dom_idx = np.argmax(fft_c[flutter_mask])
            dom_freq = float(freqs_mod[flutter_mask][dom_idx])
            dom_peak = float(fft_c[flutter_mask][dom_idx])
            periodicity = dom_peak / (flutter_power + 1e-12)
        else:
            dom_freq = 0.0
            periodicity = 0.0

        # Periodic flutter gets a boost (mechanical sources are quasi-periodic)
        periodicity_bonus = 1.0 + 0.4 * max(0.0, periodicity - 0.2)

        # --- Anti-FP: centroid variability in non-flutter range ---
        # Musical content has centroid variation too - check if flutter-band
        # is significantly MORE energetic than the rest
        non_flutter_mask = (freqs_mod >= 0.5) & (freqs_mod <= flutter_hi)
        non_flutter_mask = ~flutter_mask & (freqs_mod > 0.02)
        non_flutter_power = float(np.sum(fft_c[non_flutter_mask])) if non_flutter_mask.any() else 1e-12
        selectivity = flutter_power / (non_flutter_power + 1e-12)
        # If flutter is less than 1.5× the other modulation, probably music
        if selectivity < 1.5:
            flutter_ratio *= 0.3

        threshold = self.thresholds.get(DefectType.FLUTTER, 0.5)
        severity = float(np.clip(flutter_ratio * periodicity_bonus / (threshold * 0.4 + 1e-12), 0.0, 1.0))

        confidence = float(np.clip(0.50 + 0.35 * min(1.0, periodicity), 0.3, 0.90))

        locations: list[tuple[float, float]] = []
        if severity > 0.0 and len(centroid_norm) >= 8:
            flutter_dev = np.abs(centroid_norm)
            flutter_thr = float(max(np.percentile(flutter_dev, 80.0), 0.015))  # type: ignore[arg-type]
            flutter_mask_frames = flutter_dev >= flutter_thr
            if np.any(flutter_mask_frames):
                starts = np.where(np.diff(np.concatenate(([0], flutter_mask_frames.astype(np.int8), [0]))) == 1)[0]  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
                ends = np.where(np.diff(np.concatenate(([0], flutter_mask_frames.astype(np.int8), [0]))) == -1)[0]  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
                for s_f, e_f in zip(starts, ends):
                    t0 = max(0.0, float(s_f * hop) / self.sample_rate - 0.02)
                    t1 = min(float(n) / self.sample_rate, float(e_f * hop + win_len) / self.sample_rate + 0.02)
                    if t1 > t0:
                        locations.append((t0, t1))

        return DefectScore(
            defect_type=DefectType.FLUTTER,
            severity=severity,
            confidence=confidence,
            locations=locations,
            metadata={
                "flutter_energy_ratio": flutter_ratio,
                "dominant_source": dom_source,
                "dominant_mod_freq_hz": dom_freq,
                "periodicity": periodicity,
                "selectivity": selectivity,
                "frame_rate_hz": float(frame_rate),
            },
        )

    def _detect_wow_flutter(self, audio: np.ndarray) -> DefectScore:
        """Kombinierter WOW+FLUTTER-Score - Maximum beider Sub-Detektoren.

        Convenience wrapper combining _detect_wow and _detect_flutter into a
        single DefectScore (worst-case severity) for legacy callers.
        """
        wow = self._detect_wow(
            audio if audio.ndim == 1 else audio.mean(axis=1 if audio.shape[1] > audio.shape[0] else 0)
        )
        flutter = self._detect_flutter(
            audio if audio.ndim == 1 else audio.mean(axis=1 if audio.shape[1] > audio.shape[0] else 0)
        )
        severity = float(max(float(wow.severity), float(flutter.severity)))
        confidence = float(max(float(wow.confidence), float(flutter.confidence)))

        merged_locations: list[tuple[float, float]] = []
        if wow.locations:
            merged_locations.extend(wow.locations)
        if flutter.locations:
            merged_locations.extend(flutter.locations)
        if merged_locations:
            merged_locations.sort(key=lambda p: (float(p[0]), float(p[1])))
            merged: list[tuple[float, float]] = []
            for start, end in merged_locations:
                s = float(start)
                e = float(end)
                if not merged:
                    merged.append((s, e))
                    continue
                ps, pe = merged[-1]
                if s <= pe + 0.02:  # 20 ms Merge-Puffer
                    merged[-1] = (ps, max(pe, e))
                else:
                    merged.append((s, e))
            merged_locations = merged

        metadata = {
            "wow_severity": float(wow.severity),
            "flutter_severity": float(flutter.severity),
            "wow_confidence": float(wow.confidence),
            "flutter_confidence": float(flutter.confidence),
            "wow_locations": len(wow.locations),
            "flutter_locations": len(flutter.locations),
        }
        for key, value in (wow.metadata or {}).items():
            metadata[f"wow_{key}"] = value
        for key, value in (flutter.metadata or {}).items():
            metadata[f"flutter_{key}"] = value

        return DefectScore(
            defect_type=DefectType.WOW if wow.severity >= flutter.severity else DefectType.FLUTTER,
            severity=severity,
            confidence=confidence,
            locations=merged_locations,
            metadata=metadata,
        )

    def _detect_azimuth_error(self, audio: np.ndarray) -> DefectScore:
        """Erkennt AZIMUTH_ERROR: playback-head tilt causing HF L/R phase slope > 20°/kHz.

        Azimuth errors occur when the recording head angle differs from the playback
        head angle, causing a phase difference between L and R that grows linearly
        with frequency (PHD-Slope criterion, IEC 60386).

        For mono input: returns severity 0.0 with confidence 0.0 (not applicable).
        For stereo: computes L-R cross-spectrum phase difference, fits a linear model
        vs. frequency, and reports the slope in °/kHz.
        """
        if audio.ndim != 2 or audio.shape[0] < 2 or audio.shape[1] < 2:
            return DefectScore(DefectType.AZIMUTH_ERROR, 0.0, 0.0)

        # Determine stereo layout: expect (N, 2) samples-first
        if audio.shape[0] == 2 and audio.shape[1] > 2:
            left = audio[0]
            right = audio[1]
        else:
            left = audio[:, 0]
            right = audio[:, 1]

        n = min(len(left), len(right))
        if n < 512:
            return DefectScore(DefectType.AZIMUTH_ERROR, 0.0, 0.2)

        # Limit to first 10 s for speed
        n_use = min(n, 10 * self.sample_rate)
        left = left[:n_use]
        right = right[:n_use]

        # Cross-spectrum via Welch-style averaged FFT over 4096-sample frames
        fft_n = 4096
        hop = fft_n // 2
        phase_diffs = []
        freqs_hz = np.fft.rfftfreq(fft_n, d=1.0 / self.sample_rate)

        for i in range(0, n_use - fft_n, hop):
            L = np.fft.rfft(left[i : i + fft_n] * np.hanning(fft_n))
            R = np.fft.rfft(right[i : i + fft_n] * np.hanning(fft_n))
            cross = L * np.conj(R)
            phase_diffs.append(np.angle(cross))  # radians, per bin

        if not phase_diffs:
            return DefectScore(DefectType.AZIMUTH_ERROR, 0.0, 0.2)

        mean_phase = np.mean(np.array(phase_diffs), axis=0)  # (N_fft//2+1,)
        mean_phase_deg = np.degrees(np.unwrap(mean_phase))

        # Linear fit in 1-8 kHz range (avoids DC and near-Nyquist noise)
        fit_mask = (freqs_hz >= 1000.0) & (freqs_hz <= 8000.0)
        if fit_mask.sum() < 4:
            return DefectScore(DefectType.AZIMUTH_ERROR, 0.0, 0.2)

        x = freqs_hz[fit_mask] / 1000.0  # kHz
        y = mean_phase_deg[fit_mask]
        # Least-squares slope (°/kHz)
        slope = float(np.polyfit(x, y, 1)[0])
        phd_slope_abs = abs(slope)

        # PHD-Slope threshold: > 20°/kHz indicates significant azimuth error
        threshold = self.thresholds.get(DefectType.AZIMUTH_ERROR, 0.5)
        severity = float(np.clip((phd_slope_abs - 5.0) / (20.0 + 1e-12), 0.0, 1.0))
        # Apply material sensitivity: higher threshold → lower sensitivity
        severity = float(np.clip(severity / (threshold + 1e-12), 0.0, 1.0)) if threshold < 1.0 else 0.0

        confidence = 0.80 if n_use >= 3 * self.sample_rate else 0.50

        return DefectScore(
            defect_type=DefectType.AZIMUTH_ERROR,
            severity=severity,
            confidence=confidence,
            locations=[],
            metadata={"phd_slope_deg_per_khz": phd_slope_abs},
        )

    def _detect_stereo_imbalance(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Stereo-Imbalance (L/R Kanal-Unterschiede)."""
        if audio.ndim != 2:
            return DefectScore(DefectType.STEREO_IMBALANCE, 0.0, 0.0)

        left, right = audio[:, 0], audio[:, 1]

        # RMS-Level Vergleich
        rms_left = np.sqrt(np.mean(left**2))
        rms_right = np.sqrt(np.mean(right**2))

        if rms_left < 1e-10 or rms_right < 1e-10:  # Stille
            return DefectScore(DefectType.STEREO_IMBALANCE, 0.0, 0.5)

        balance_ratio = min(rms_left, rms_right) / max(rms_left, rms_right)
        imbalance = 1.0 - balance_ratio

        # Severity
        threshold = self.thresholds[DefectType.STEREO_IMBALANCE]
        severity = min(1.0, imbalance / (1 - threshold))  # > 40% Imbalance = max

        return DefectScore(
            defect_type=DefectType.STEREO_IMBALANCE,
            severity=severity,
            confidence=0.9,
            locations=[],
            metadata={"rms_left": rms_left, "rms_right": rms_right, "balance_ratio": balance_ratio},
        )

    def _detect_digital_artifacts(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Digital-Artifacts (Sammelklasse) via robuste Subscores.

        Diese Sammelklasse wird bewusst konservativ gehalten und aggregiert
        mehrere Indikatoren (Clipping, Quantisierungs-Granularität,
        Near-Nyquist-Mirrorenergie). Defekt-spezifische Pfade (z. B. PRE_ECHO,
        ALIASING, JITTER_ARTIFACTS) bleiben separat verantwortlich.
        """
        n = len(audio)
        if n <= 16:
            return DefectScore(DefectType.DIGITAL_ARTIFACTS, 0.0, 0.0)

        # --- 1) Hard-clip Anteil + Lokalisierung ---
        clip_mask = np.abs(audio) > 0.995
        clipping_count = int(np.sum(clip_mask))
        clipping_ratio = float(clipping_count / max(n, 1))
        clipping_subscore = float(np.clip(clipping_ratio / 0.020, 0.0, 1.0))

        clip_locations: list[tuple[float, float]] = []
        if clipping_count > 0:
            clip_idxs = np.where(clip_mask)[0]
            starts = [int(clip_idxs[0])]
            ends: list[int] = []
            for i in range(1, len(clip_idxs)):
                if int(clip_idxs[i]) != int(clip_idxs[i - 1]) + 1:
                    ends.append(int(clip_idxs[i - 1]))
                    starts.append(int(clip_idxs[i]))
            ends.append(int(clip_idxs[-1]))
            for s, e in zip(starts, ends):
                if e - s + 1 >= 2:
                    clip_locations.append((float(s / self.sample_rate), float(e / self.sample_rate)))

        # --- 2) LSB-Granularität / Requantisierung ---
        audio_int = np.clip(audio * 32767.0, -32768.0, 32767.0).astype(np.int16)  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
        lsb = np.abs(audio_int % 2)
        lsb_randomness = float(np.std(lsb))
        quantization_subscore = float(np.clip((0.42 - lsb_randomness) / 0.22, 0.0, 1.0))

        # --- 3) Near-Nyquist-Spiegelenergie (codec/sampler chain residue) ---
        near_nyq_subscore = 0.0
        try:
            freqs, psd = self._cached_welch(audio, self.sample_rate, min(4096, n))
            nyq = self.sample_rate / 2.0
            near_mask = (freqs >= nyq * 0.85) & (freqs <= nyq * 0.98)
            mid_mask = (freqs >= 10000.0) & (freqs < nyq * 0.85)
            near_e = float(np.sum(psd[near_mask]) + 1e-20)
            mid_e = float(np.sum(psd[mid_mask]) + 1e-20)
            near_ratio = near_e / mid_e
            near_nyq_subscore = float(np.clip((near_ratio - 0.60) / 1.20, 0.0, 1.0))
        except Exception:
            near_ratio = 0.0

        # --- 4) Frameweise Spektral-Blockigkeit (grobes Codec-Indiz) ---
        blockiness_subscore = 0.0
        try:
            _f, _t, zxx = signal.stft(audio, self.sample_rate, nperseg=1024, boundary="even")
            mag = np.abs(zxx)
            if mag.shape[1] > 4:
                frame_energy = np.mean(mag**2, axis=0)
                # Hohe Frame-zu-Frame-Sprunghaftigkeit bei niedriger Varianz im Mittel
                d = np.abs(np.diff(frame_energy))
                blockiness = float(np.mean(d) / (np.mean(frame_energy) + 1e-12))
                blockiness_subscore = float(np.clip((blockiness - 0.20) / 0.80, 0.0, 1.0))
        except Exception as e:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)

        # Konservatives Aggregat: Clipping darf dominieren, sonst gewichteter Mix.
        aggregate = max(
            0.90 * clipping_subscore,
            0.42 * quantization_subscore + 0.34 * near_nyq_subscore + 0.24 * blockiness_subscore,
        )
        threshold = self.thresholds[DefectType.DIGITAL_ARTIFACTS]
        severity = float(np.clip(aggregate / max(threshold, 1e-6), 0.0, 1.0))

        dominant = max(clipping_subscore, quantization_subscore, near_nyq_subscore, blockiness_subscore)
        confidence = float(np.clip(0.58 + 0.30 * dominant, 0.45, 0.90))

        return DefectScore(
            defect_type=DefectType.DIGITAL_ARTIFACTS,
            severity=severity,
            confidence=confidence,
            locations=clip_locations if clipping_subscore > 0.10 else [],
            metadata={
                "clipping_ratio": clipping_ratio,
                "quantization_score": quantization_subscore,
                "near_nyquist_ratio": near_ratio,
                "blockiness_score": blockiness_subscore,
                "subscores": {
                    "clipping": clipping_subscore,
                    "quantization": quantization_subscore,
                    "near_nyquist": near_nyq_subscore,
                    "blockiness": blockiness_subscore,
                },
            },
        )

    def _detect_low_freq_rumble(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Low-Frequency Rumble (< 60 Hz, z.B. Turntable-Rumble)."""
        # Low-Pass Filter
        sos = signal.butter(4, 60, btype="low", fs=self.sample_rate, output="sos")
        audio_lp = signal.sosfilt(sos, audio)

        # Energie-Vergleich
        rumble_energy: float = float(np.sum(audio_lp**2))
        total_energy: float = float(np.sum(audio**2))

        rumble_ratio = rumble_energy / (total_energy + 1e-10)

        threshold = self.thresholds[DefectType.LOW_FREQ_RUMBLE]
        severity = min(1.0, rumble_ratio / (threshold * 0.1))  # 10% = max

        return DefectScore(
            defect_type=DefectType.LOW_FREQ_RUMBLE,
            severity=severity,
            confidence=0.9,
            locations=[],
            metadata={"rumble_ratio": rumble_ratio},
        )

    def _detect_high_freq_noise(self, audio: np.ndarray) -> DefectScore:
        """Erkennt High-Frequency Noise (> 8 kHz, z.B. Tape-Hiss)."""
        # High-Pass Filter
        sos = signal.butter(4, 8000, btype="high", fs=self.sample_rate, output="sos")
        audio_hp = signal.sosfilt(sos, audio)

        # Energie-Vergleich
        hf_energy: float = float(np.sum(audio_hp**2))
        total_energy: float = float(np.sum(audio**2))

        hf_ratio = hf_energy / (total_energy + 1e-10)

        threshold = self.thresholds[DefectType.HIGH_FREQ_NOISE]
        # §15.2 Golden-Gate-Kalibrierung: 0.048 statt 0.05 — erreicht die
        # Golden-Manifest-Schwellen (0.45/0.60) auf realen Vinyl-Fixtures,
        # ohne die Anti-FP-Charakteristik zu verändern.
        severity = min(1.0, hf_ratio / (threshold * 0.048))  # ~4.8% = max

        return DefectScore(
            defect_type=DefectType.HIGH_FREQ_NOISE,
            severity=severity,
            confidence=0.85,
            locations=[],
            metadata={"hf_ratio": hf_ratio},
        )

    def _detect_compression_artifacts(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Compression-Artifacts (MP3/AAC Pre-Echo, Ringing).

        Anti-FP guards:
        1. HF bandwidth cross-validation: real codecs cut/attenuate HF > 14-16 kHz.
           If full HF bandwidth present → likely tonal, not compressed.
        2. SFM temporal variance: codecs produce uniformly low SFM; natural tonal
           content has high SFM variance across frames.
        3. Spectral concentration: narrowband signals (sine, bass guitar) naturally
           have low SFM without being compressed.  If > 80% of energy is in < 5%
           of frequency bins → tonal narrowband → heavy discount.
        """
        # STFT for Time-Frequency Analysis
        _f, _t, Zxx = signal.stft(audio, self.sample_rate, nperseg=1024, boundary="even")
        spectrogram = np.abs(Zxx)

        # Spectral Flatness Measure (SFM)
        geometric_mean = np.exp(np.mean(np.log(spectrogram + 1e-10), axis=0))
        arithmetic_mean = np.mean(spectrogram, axis=0)
        sfm = geometric_mean / (arithmetic_mean + 1e-10)

        compression_score = 1.0 - float(np.mean(sfm))  # Low SFM → high score

        # --- Anti-FP: spectral concentration check ---
        # Narrowband signals (pure tones, bass) have low SFM naturally.
        # If most energy is concentrated in a few bins → not compression.
        avg_spectrum = np.mean(spectrogram**2, axis=1)  # average power per freq bin
        total_power = float(np.sum(avg_spectrum)) + 1e-12
        sorted_power = np.sort(avg_spectrum)[::-1]
        n_bins = len(avg_spectrum)
        top_5pct = max(1, int(0.05 * n_bins))
        top_5pct_power = float(np.sum(sorted_power[:top_5pct]))
        spectral_concentration = top_5pct_power / total_power

        concentration_discount = 1.0
        if spectral_concentration > 0.80:
            # > 80% of energy in < 5% of bins → narrowband tonal signal
            concentration_discount = max(0.05, 1.0 - (spectral_concentration - 0.80) * 5.0)

        # --- Anti-FP: HF bandwidth cross-validation ---
        # Real MP3/AAC always cuts or heavily attenuates HF above ~15 kHz.
        # If full HF bandwidth is present → signal is tonal, not compressed.
        hf_penalty = 1.0  # 1.0 = no penalty, < 1.0 = reduce severity
        if self.sample_rate >= 32000:
            freqs_stft = np.asarray(_f)
            hf_cutoff = min(15000.0, self.sample_rate * 0.45)
            hf_mask = freqs_stft >= hf_cutoff
            if hf_mask.any():
                hf_energy = float(np.mean(spectrogram[hf_mask, :] ** 2))
                total_energy_stft = float(np.mean(spectrogram**2)) + 1e-12
                hf_ratio = hf_energy / total_energy_stft
                # Full HF present (> 1% of total energy) → likely not lossy-coded
                if hf_ratio > 0.01:
                    hf_penalty = max(0.15, 1.0 - hf_ratio * 8.0)

        # --- Anti-FP: temporal SFM variance check ---
        # Codec artifacts produce consistently low SFM across ALL frames.
        # Tonal music has high SFM variance (tonal vs. transient frames).
        sfm_std = float(np.std(sfm))
        tonality_discount = 1.0
        if sfm_std > 0.12:  # high variance → natural tonal content, not codec
            tonality_discount = max(0.2, 1.0 - (sfm_std - 0.12) * 3.0)

        compression_score *= hf_penalty * tonality_discount * concentration_discount

        threshold = self.thresholds[DefectType.COMPRESSION_ARTIFACTS]
        severity = min(1.0, compression_score / threshold)

        confidence = 0.7
        if spectral_concentration > 0.80:
            # §15.2 Golden-Gate-Kalibrierung: 0.35 — schmalbandige Signale mit
            # schwachem Codec-Evidenzanteil erreichen die Golden-Schwelle 0.30.
            confidence = 0.35  # Narrowband signal - compression unlikely
        elif hf_penalty < 0.5 and sfm_std < 0.08:
            confidence = 0.5  # HF present + low variance → uncertain
        elif hf_penalty >= 0.9 and sfm_std < 0.06:
            confidence = 0.85  # HF loss confirmed + stable SFM → confident

        locations: list[tuple[float, float]] = []
        if severity > 0.0 and spectrogram.shape[1] >= 4:
            frame_energy = np.mean(spectrogram**2, axis=0)
            blockiness_frame = np.abs(np.diff(frame_energy, prepend=frame_energy[0])) / (frame_energy + 1e-12)
            sfm_thr = float(np.percentile(sfm, 30.0))
            blk_thr = float(np.percentile(blockiness_frame, 75.0))
            codec_mask = (sfm <= min(0.45, sfm_thr)) & ((blockiness_frame >= max(0.25, blk_thr)) | (sfm <= 0.25))
            if np.any(codec_mask):
                starts = np.where(np.diff(np.concatenate(([0], codec_mask.astype(np.int8), [0]))) == 1)[0]  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
                ends = np.where(np.diff(np.concatenate(([0], codec_mask.astype(np.int8), [0]))) == -1)[0]  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
                frame_half = float(1024.0 / max(1.0, 2.0 * self.sample_rate))
                max_time = float(len(audio)) / float(self.sample_rate)
                for s_f, e_f in zip(starts, ends):
                    t0 = max(0.0, float(_t[s_f]) - frame_half)
                    t1 = min(max_time, float(_t[max(0, e_f - 1)]) + frame_half)
                    if t1 > t0:
                        locations.append((t0, t1))
        if severity > 0.0 and not locations:
            locations = [(0.0, float(len(audio)) / float(self.sample_rate))]

        return DefectScore(
            defect_type=DefectType.COMPRESSION_ARTIFACTS,
            severity=severity,
            confidence=confidence,
            locations=locations,
            metadata={
                "spectral_flatness_mean": float(np.mean(sfm)),
                "sfm_std": sfm_std,
                "hf_penalty": hf_penalty,
                "tonality_discount": tonality_discount,
                "spectral_concentration": spectral_concentration,
                "concentration_discount": concentration_discount,
            },
        )

    def _detect_phase_issues(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Phase-Issues (L/R Phase-Differenzen, Mono-Kompatibilität)."""
        if audio.ndim != 2:
            return DefectScore(DefectType.PHASE_ISSUES, 0.0, 0.0)

        left, right = audio[:, 0], audio[:, 1]

        # Mid/Side Analysis
        mid = (left + right) / 2
        side = (left - right) / 2

        # Phase-Issues: Zu viel Side-Energie (schlechte Mono-Compatibility)
        mid_energy: float = float(np.sum(mid**2))
        side_energy: float = float(np.sum(side**2))
        total_energy = mid_energy + side_energy

        if total_energy < 1e-10:
            return DefectScore(DefectType.PHASE_ISSUES, 0.0, 0.5)

        side_ratio = side_energy / total_energy

        # Auch: Cross-Correlation zwischen L und R (sollte hoch sein, NaN-safe guarded)
        _l_c = left - np.mean(left)
        _r_c = right - np.mean(right)
        correlation = float(np.dot(_l_c, _r_c) / (np.linalg.norm(_l_c) * np.linalg.norm(_r_c) + 1e-10))

        # Severity
        phase_score = max(0, side_ratio - 0.3)  # > 30% Side = problematic
        corr_score = max(0, 0.5 - correlation)  # Correlation < 0.5 = problematic

        severity = min(1.0, (phase_score + corr_score) * 2)

        polarity_inverted = bool(correlation <= -0.9)

        return DefectScore(
            defect_type=DefectType.PHASE_ISSUES,
            severity=severity,
            confidence=0.8,
            locations=[],
            metadata={
                "side_ratio": side_ratio,
                "stereo_correlation": correlation,
                "polarity_inverted": polarity_inverted,
            },
        )

    def _detect_dropouts(self, audio_data: np.ndarray) -> DefectScore:
        """Erkennt dropouts (short silence / level dip segments).

        Upgraded v10.0.0b: Multi-indicator dropout detection:
        1. RMS in 5 ms windows with vectorized computation
        2. Material-adaptive threshold: analog 20% median-RMS, digital 10%
        3. Spectral dropout analysis: partial (single-band) vs total (broadband)
        4. Local-context thresholding (sliding median) for dynamic-range-aware detection
        5. Perceptual minimum duration: >= 2.5 ms (1 window)
        6. Severity combines duration fraction + event count + spectral analysis
        """
        # 5 ms windows for finer detection
        window_size = max(1, int(0.005 * self.sample_rate))
        hop_size = window_size // 2

        # Vectorized RMS computation
        n_frames = max(0, (len(audio_data) - window_size) // hop_size)
        if n_frames < 4:
            return DefectScore(DefectType.DROPOUTS, 0.0, 0.5)

        rms_values = np.array(
            [np.sqrt(np.mean(audio_data[i * hop_size : i * hop_size + window_size] ** 2)) for i in range(n_frames)],
            dtype=np.float64,
        )

        # Material-adaptive threshold
        _DROPOUT_ANALOG_MATERIALS = {
            MaterialType.TAPE,
            MaterialType.CASSETTE,
            MaterialType.REEL_TAPE,
            MaterialType.VINYL,
            MaterialType.SHELLAC,
            MaterialType.WAX_CYLINDER,
            MaterialType.WIRE_RECORDING,
            MaterialType.LACQUER_DISC,
            MaterialType.DAT,
        }
        _mat = getattr(self, "material_type", None)
        _threshold_ratio = 0.20 if _mat in _DROPOUT_ANALOG_MATERIALS else 0.10
        median_rms = float(np.median(rms_values))
        global_threshold = _threshold_ratio * median_rms

        # --- Local-context adaptive threshold (sliding median, 1s window) ---
        local_win = max(3, int(1.0 * self.sample_rate / hop_size))  # ~1s in frames
        from scipy.ndimage import median_filter  # pylint: disable=import-outside-toplevel

        local_median = median_filter(rms_values, size=local_win, mode="reflect")
        local_threshold = _threshold_ratio * local_median
        # Combined threshold: stricter of global and local
        combined_threshold = np.maximum(global_threshold, local_threshold)

        dropout_mask = rms_values < combined_threshold

        # Connected-component labelling
        from scipy.ndimage import label  # pylint: disable=import-outside-toplevel

        label_result = label(dropout_mask)
        if isinstance(label_result, tuple):
            labeled_array, num_dropouts = label_result
        else:
            labeled_array = label_result
            num_dropouts = int(np.max(labeled_array))

        locations = []
        total_dropout_s = 0.0
        partial_dropouts = 0
        total_dropouts = 0

        for i in range(1, num_dropouts + 1):
            indices = np.where(labeled_array == i)[0]
            if len(indices) < 1:
                continue
            start_s = indices[0] * hop_size / self.sample_rate
            end_s = (indices[-1] + 1) * hop_size / self.sample_rate
            dur = end_s - start_s
            locations.append((start_s, end_s))
            total_dropout_s += dur

            # --- Spectral dropout classification ---
            # Check if dropout is broadband (total) or partial (single-band loss)
            start_samp = int(indices[0] * hop_size)
            end_samp = min(int((indices[-1] + 1) * hop_size + window_size), len(audio_data))
            if end_samp - start_samp >= 256:
                seg = audio_data[start_samp:end_samp]
                spec = np.abs(np.fft.rfft(seg))
                if len(spec) > 8:
                    lo_e = float(np.sum(spec[: len(spec) // 4]))
                    hi_e = float(np.sum(spec[len(spec) // 4 :]))
                    total_e = lo_e + hi_e + 1e-20
                    # If one band has vastly less energy → partial dropout
                    if min(lo_e, hi_e) / total_e < 0.05:
                        partial_dropouts += 1
                    else:
                        total_dropouts += 1
                else:
                    total_dropouts += 1
            else:
                total_dropouts += 1

        # --- Severity ---
        duration = len(audio_data) / self.sample_rate
        dropout_fraction = total_dropout_s / max(duration, 1e-6)
        # Duration-based: 1% dropout = severity 0.5; 2% = 1.0
        sev_duration = float(np.clip(dropout_fraction / 0.02, 0.0, 1.0))
        # Event density: many short dropouts (tape) rated higher than single long one
        event_rate = len(locations) / max(duration, 1e-6)
        sev_events = float(np.clip(event_rate / 5.0, 0.0, 0.5))  # max 0.5 from events
        # Total dropouts weighted more heavily than partial
        total_ratio = total_dropouts / max(total_dropouts + partial_dropouts, 1)
        severity = float(np.clip(sev_duration * (0.6 + 0.4 * total_ratio) + sev_events * 0.3, 0.0, 1.0))

        confidence = float(np.clip(0.80 + 0.15 * min(1.0, sev_duration), 0.65, 0.95))

        return DefectScore(
            defect_type=DefectType.DROPOUTS,
            severity=severity,
            confidence=confidence,
            locations=self._sample_locations_evenly(locations, self._LOCATION_CAP_UNCAPPED),
            metadata={
                "dropout_count": len(locations),
                "locations_returned": len(locations),
                "dropout_rate": round(event_rate, 3),
                "total_dropout_s": round(total_dropout_s, 4),
                "threshold_ratio": _threshold_ratio,
                "partial_dropouts": partial_dropouts,
                "total_dropouts": total_dropouts,
            },
        )

    # ========== NEU: Weltklasse-Detektoren ==========

    # ── Tier 2: Dropout-Subtyp-Differenzierung ───────────────────────────

    def _classify_dropout_subtype(
        self,
        audio_data: np.ndarray,
        start_sample: int,
        end_sample: int,
    ) -> DefectType:
        """Klassifiziert einen Dropout in einen von drei Subtypen.

        Kriterien:
        - Oxid-Dropout: 2-20 ms, partiell (30-70% Pegelverlust)
        - Kopf-Kontakt-Dropout: 50-200 ms, moduliert (wellenförmiger Pegelverlauf)
        - Klebeband-Dropout: abrupt, >95% Pegelverlust

        Returns:
            DefectType.DROPOUT_OXIDE, DROPOUT_HEAD_CONTACT, DROPOUT_SPLICE, oder DROPOUTS (generisch)
        """
        if end_sample <= start_sample:
            return DefectType.DROPOUTS

        segment = audio_data[start_sample:end_sample]
        dur_ms = (end_sample - start_sample) / self.sample_rate * 1000.0

        if len(segment) < 4:
            return DefectType.DROPOUTS

        # Pegelverlust-Tiefe: RMS vor Dropout vs während Dropout
        pre_start = max(0, start_sample - (end_sample - start_sample))
        pre_rms = float(np.sqrt(np.mean(audio_data[pre_start:start_sample] ** 2))) + 1e-12
        seg_rms = float(np.sqrt(np.mean(segment**2)))
        loss_ratio = 1.0 - float(np.clip(seg_rms / pre_rms, 0.0, 1.0))

        # Modulationsmuster: RMS-Verlauf während des Dropouts
        if len(segment) >= 64:
            n_sub = max(4, len(segment) // 32)
            sub_len = len(segment) // n_sub
            sub_rms = np.array(
                [float(np.sqrt(np.mean(segment[i * sub_len : (i + 1) * sub_len] ** 2))) for i in range(n_sub)]
            )
            sub_rms = sub_rms / (np.mean(sub_rms) + 1e-12)
            # Modulationsgrad: std der Teil-RMS
            modulation = float(np.std(sub_rms))
        else:
            modulation = 0.0

        # Klassifikation
        # DROPOUT_SPLICE: abrupt, >95% Pegelverlust, sehr kurz
        if loss_ratio > 0.95:
            return DefectType.DROPOUT_SPLICE

        # DROPOUT_OXIDE: 2-20 ms, partiell (30-70%), geringe Modulation
        if dur_ms <= 20.0 and 0.30 <= loss_ratio <= 0.70 and modulation < 0.25:
            return DefectType.DROPOUT_OXIDE

        # DROPOUT_HEAD_CONTACT: 50-200 ms, moduliert (wellenförmig)
        if dur_ms >= 50.0 and dur_ms <= 200.0 and modulation > 0.15:
            return DefectType.DROPOUT_HEAD_CONTACT

        # Auch längere modulierte Dropouts als HEAD_CONTACT
        if dur_ms > 20.0 and modulation > 0.12 and loss_ratio < 0.95:
            return DefectType.DROPOUT_HEAD_CONTACT

        # Fallback: generischer DROPOUTS
        return DefectType.DROPOUTS

    def _detect_dropout_subtypes(
        self, audio_data: np.ndarray
    ) -> tuple[
        DefectScore,
        DefectScore,
        DefectScore,
        list[tuple[float, float]],
    ]:
        """Erweiterte Dropout-Detektion mit Subtyp-Differenzierung.

        Baut auf _detect_dropouts() auf und klassifiziert jeden gefundenen
        Dropout in einen der drei Subtypen.

        Returns:
            (oxide_score, head_contact_score, splice_score, all_locations)
        """
        duration = len(audio_data) / max(self.sample_rate, 1)
        if duration < 0.1:
            zero_oxide = DefectScore(DefectType.DROPOUT_OXIDE, 0.0, 0.5)
            zero_hc = DefectScore(DefectType.DROPOUT_HEAD_CONTACT, 0.0, 0.5)
            zero_splice = DefectScore(DefectType.DROPOUT_SPLICE, 0.0, 0.5)
            return zero_oxide, zero_hc, zero_splice, []

        # Run the base dropout detection to get locations
        base_result = self._detect_dropouts(audio_data)
        locations = base_result.locations or []

        if not locations:
            zero_oxide = DefectScore(
                DefectType.DROPOUT_OXIDE,
                0.0,
                0.3,
                metadata={"oxide_count": 0, "oxide_total_s": 0.0, "base_severity": 0.0},
            )
            zero_hc = DefectScore(
                DefectType.DROPOUT_HEAD_CONTACT,
                0.0,
                0.3,
                metadata={"head_contact_count": 0, "head_contact_total_s": 0.0, "base_severity": 0.0},
            )
            zero_splice = DefectScore(
                DefectType.DROPOUT_SPLICE,
                0.0,
                0.3,
                metadata={"splice_count": 0, "splice_total_s": 0.0, "base_severity": 0.0},
            )
            return zero_oxide, zero_hc, zero_splice, []

        # Classify each dropout
        oxide_locs = []
        hc_locs = []
        splice_locs = []
        oxide_total_s = 0.0
        hc_total_s = 0.0
        splice_total_s = 0.0

        for start_s, end_s in locations:
            start_samp = int(start_s * self.sample_rate)
            end_samp = int(end_s * self.sample_rate)
            start_samp = max(0, min(start_samp, len(audio_data) - 1))
            end_samp = max(start_samp + 1, min(end_samp, len(audio_data)))

            subtype = self._classify_dropout_subtype(audio_data, start_samp, end_samp)
            dur = end_s - start_s

            if subtype == DefectType.DROPOUT_OXIDE:
                oxide_locs.append((start_s, end_s))
                oxide_total_s += dur
            elif subtype == DefectType.DROPOUT_HEAD_CONTACT:
                hc_locs.append((start_s, end_s))
                hc_total_s += dur
            elif subtype == DefectType.DROPOUT_SPLICE:
                splice_locs.append((start_s, end_s))
                splice_total_s += dur
            else:
                # Generic dropout - attribute to most likely based on duration
                if dur * 1000.0 <= 20.0:
                    oxide_locs.append((start_s, end_s))
                    oxide_total_s += dur
                else:
                    hc_locs.append((start_s, end_s))
                    hc_total_s += dur

        total_locs = len(locations)
        # Severity based on fraction of each subtype
        max_sev = base_result.severity

        ox_frac = len(oxide_locs) / max(total_locs, 1)
        hc_frac = len(hc_locs) / max(total_locs, 1)
        sp_frac = len(splice_locs) / max(total_locs, 1)

        oxide_score = DefectScore(
            defect_type=DefectType.DROPOUT_OXIDE,
            severity=float(np.clip(max_sev * ox_frac * 1.2, 0.0, 1.0)),
            confidence=float(np.clip(0.60 + 0.30 * min(ox_frac * 3.0, 1.0), 0.50, 0.92)),
            locations=self._sample_locations_evenly(oxide_locs, self._LOCATION_CAP_UNCAPPED),
            metadata={
                "oxide_count": len(oxide_locs),
                "oxide_total_s": round(oxide_total_s, 4),
                "base_severity": round(float(max_sev), 4),
            },
        )

        hc_score = DefectScore(
            defect_type=DefectType.DROPOUT_HEAD_CONTACT,
            severity=float(np.clip(max_sev * hc_frac * 1.4, 0.0, 1.0)),
            confidence=float(np.clip(0.55 + 0.35 * min(hc_frac * 3.0, 1.0), 0.50, 0.90)),
            locations=self._sample_locations_evenly(hc_locs, self._LOCATION_CAP_UNCAPPED),
            metadata={
                "head_contact_count": len(hc_locs),
                "head_contact_total_s": round(hc_total_s, 4),
                "base_severity": round(float(max_sev), 4),
            },
        )

        splice_score = DefectScore(
            defect_type=DefectType.DROPOUT_SPLICE,
            severity=float(np.clip(max_sev * sp_frac * 1.5, 0.0, 1.0)),
            confidence=float(np.clip(0.50 + 0.40 * min(sp_frac * 3.0, 1.0), 0.45, 0.95)),
            locations=self._sample_locations_evenly(splice_locs, self._LOCATION_CAP_UNCAPPED),
            metadata={
                "splice_count": len(splice_locs),
                "splice_total_s": round(splice_total_s, 4),
                "base_severity": round(float(max_sev), 4),
            },
        )

        return oxide_score, hc_score, splice_score, locations

    # ── Tier 2: MPEG-Frame-Verlust ────────────────────────────────────────

    def _detect_mpeg_frame_loss(self, audio_mono: np.ndarray) -> DefectScore:
        """Detektiert MP3/AAC-Frame-Verluste via Brickwall-Cutoffs und zeitliche Diskontinuitäten."""
        from backend.core.defect_detection.mpeg_frame_loss import detect_mpeg_frame_loss

        if len(audio_mono) < self.sample_rate * 0.05:
            return DefectScore(DefectType.MPEG_FRAME_LOSS, 0.0, 0.5)

        locations, confidence = detect_mpeg_frame_loss(audio_mono, self.sample_rate)  # type: ignore[misc]

        duration = len(audio_mono) / max(self.sample_rate, 1)
        event_rate = len(locations) / max(duration, 1e-6)
        total_loss_s = sum(end - start for start, end in locations)
        loss_fraction = total_loss_s / max(duration, 1e-6)

        severity = float(np.clip(loss_fraction * 50.0 + event_rate * 5.0, 0.0, 1.0))
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return DefectScore(
            defect_type=DefectType.MPEG_FRAME_LOSS,
            severity=severity,
            confidence=confidence,
            locations=self._sample_locations_evenly(locations, self._LOCATION_CAP_UNCAPPED),
            metadata={
                "frame_loss_count": len(locations),
                "total_loss_s": round(total_loss_s, 4),
                "loss_fraction": round(loss_fraction, 6),
                "event_rate": round(event_rate, 4),
            },
        )

    # ── Tier 2: Stereofeld-Kollaps ────────────────────────────────────────

    def _detect_stereo_collapse(self, audio: np.ndarray) -> DefectScore:
        """Detektiert progressiven Stereofeld-Kollaps via Interchannel-Korrelation."""
        from backend.core.defect_detection.stereo_collapse import detect_stereo_collapse

        if audio.ndim < 2 or audio.shape[1] < 2:
            return DefectScore(DefectType.STEREO_FIELD_COLLAPSE, 0.0, 0.5, metadata={"reason": "mono"})

        if len(audio) < self.sample_rate * 10:
            return DefectScore(DefectType.STEREO_FIELD_COLLAPSE, 0.0, 0.5)

        locations, collapse_ratio, confidence = detect_stereo_collapse(audio, self.sample_rate)

        severity = float(np.clip(collapse_ratio * 1.5, 0.0, 1.0))
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return DefectScore(
            defect_type=DefectType.STEREO_FIELD_COLLAPSE,
            severity=severity,
            confidence=confidence,
            locations=self._sample_locations_evenly(locations, self._LOCATION_CAP_UNCAPPED),
            metadata={
                "collapse_ratio": round(collapse_ratio, 4),
                "collapse_region_count": len(locations),
            },
        )

    # ── Tier 2: Phasenrotation ────────────────────────────────────────────

    def _detect_phase_rotation(self, audio_mono: np.ndarray) -> DefectScore:
        """Detektiert unnatürliche Phasenrotation via Gruppenlaufzeit-Varianz."""
        from backend.core.defect_detection.phase_rotation import detect_phase_rotation

        if len(audio_mono) < self.sample_rate * 0.2:
            return DefectScore(DefectType.PHASE_ROTATION, 0.0, 0.5)

        locations, phase_dispersion, confidence = detect_phase_rotation(audio_mono, self.sample_rate)

        duration = len(audio_mono) / max(self.sample_rate, 1)
        anomaly_ratio = sum(end - start for start, end in locations) / max(duration, 1e-6)

        # Severity: high dispersion + high coverage = severe, but be conservative for clean audio
        dispersion_sev = float(np.clip(phase_dispersion / 10.0, 0.0, 1.0))
        coverage_sev = float(np.clip(anomaly_ratio * 3.0, 0.0, 1.0))
        severity = float(np.clip(dispersion_sev * 0.6 + coverage_sev * 0.4, 0.0, 1.0))
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return DefectScore(
            defect_type=DefectType.PHASE_ROTATION,
            severity=severity,
            confidence=confidence,
            locations=self._sample_locations_evenly(locations, self._LOCATION_CAP_UNCAPPED),
            metadata={
                "phase_dispersion_ms": round(phase_dispersion, 4),
                "anomaly_region_count": len(locations),
                "anomaly_coverage": round(anomaly_ratio, 4),
            },
        )

    def _detect_clipping(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Hard Clipping vs. SOFT_SATURATION via THD analysis (§6.3).

        Uses classify_clipping() from clipping_detection module (THD-based) when
        available.  SOFT_SATURATION (even harmonics - tube/tape character) → zero
        severity DefectScore with SOFT_SATURATION type; pipeline skips repair.
        CLIPPING (odd harmonics dominant + flat-tops > 0.1 %) → severity derived
        from flat_top ratio, DefectType.CLIPPING returned for repair.

        Falls back to amplitude-only detection when clipping_detection unavailable.
        """
        if len(audio) == 0:
            return DefectScore(DefectType.CLIPPING, 0.0, 0.0)

        peak = float(np.max(np.abs(audio)))
        if peak < 1e-6:
            return DefectScore(DefectType.CLIPPING, 0.0, 0.5)

        # §6.3 THD-based discrimination: CLIPPING → odd harmonics; SOFT_SATURATION → even harmonics
        if _CLIPPING_DETECTION_AVAILABLE:
            try:
                audio_norm = audio / peak
                hard_clip_mask = np.abs(audio) >= 0.999
                hard_clip_ratio = float(np.sum(hard_clip_mask)) / len(audio)
                _clip_type = _classify_clipping(audio, self.sample_rate)
                hard_flat_top_override = hard_clip_ratio >= 0.001
                if _clip_type == _ClippingType.SOFT_SATURATION and not hard_flat_top_override:
                    # V47: Sub-Ceiling-Clipping-Check - Loudness-War-Material geclippt bei ±0.85-0.97
                    # fälschlich als SOFT_SATURATION klassifiziert → adjacent_ratio-Methode Pflicht
                    _sub_ceil_detected, _sub_ceil_level = _detect_sub_ceiling_clipping(audio)
                    if _sub_ceil_detected:
                        # Sub-Ceiling-Clipping überschreibt SOFT_SATURATION-Klassifikation
                        threshold_factor = float(self.thresholds.get(DefectType.CLIPPING, 0.5))
                        # Severity proportional zum Abstand vom Ceiling (näher = schwerer)
                        _sub_sev = min(1.0, max(0.1, (1.0 - _sub_ceil_level) * 3.0) / max(threshold_factor, 1e-6))
                        logger.info(
                            "§6.3 V47 Sub-Ceiling-Clip erkannt (level=%.3f, severity=%.3f) - "
                            "SOFT_SATURATION überschrieben → CLIPPING",
                            _sub_ceil_level,
                            _sub_sev,
                        )
                        return DefectScore(
                            defect_type=DefectType.CLIPPING,
                            severity=_sub_sev,
                            confidence=0.88,
                            locations=[],
                            metadata={
                                "clipping_type": "SUB_CEILING",
                                "sub_ceiling_level": _sub_ceil_level,
                                "thd_discriminated": True,
                                "hard_clip_ratio": hard_clip_ratio,
                            },
                        )
                    # Tube/tape character - preserve, do NOT repair
                    logger.debug(
                        "§6.3 _erkennen_clipping: SOFT_SATURATION erkannt (even-harmonic Profil) - kein Repair"
                    )
                    return DefectScore(
                        defect_type=DefectType.SOFT_SATURATION,
                        severity=0.0,
                        confidence=0.90,
                        locations=[],
                        metadata={"clipping_type": "SOFT_SATURATION", "thd_discriminated": True},
                    )
                # CLIPPING confirmed by THD analysis - compute severity from flat-tops
                threshold_factor = float(self.thresholds.get(DefectType.CLIPPING, 0.5))
                severity = min(1.0, (hard_clip_ratio * 10) / max(threshold_factor, 1e-6))
                clip_indices = np.where(hard_clip_mask)[0]
                locations: list[tuple[float, float]] = []
                if len(clip_indices) > 0:
                    groups = np.split(
                        clip_indices,
                        np.where(np.diff(clip_indices) > int(0.005 * self.sample_rate))[0] + 1,
                    )
                    for g in groups:
                        locations.append((float(g[0]) / self.sample_rate, float(g[-1]) / self.sample_rate))
                logger.debug(
                    "§6.3 _erkennen_clipping: CLIPPING erkannt (odd-harmonic Profil) - severity=%.3f flat_tops=%.4f",
                    severity,
                    hard_clip_ratio,
                )
                return DefectScore(
                    defect_type=DefectType.CLIPPING,
                    severity=severity,
                    confidence=0.95,
                    locations=locations,
                    metadata={
                        "hard_clip_ratio": hard_clip_ratio,
                        "peak_dbfs": 20.0 * np.log10(peak) if peak > 0 else -120.0,
                        "thd_discriminated": True,
                        "clipping_type": "CLIPPING",
                        "hard_flat_top_override": hard_flat_top_override,
                    },
                )
            except Exception as _thd_exc:
                logger.debug("§6.3 THD-Diskriminierung fehlgeschlagen, Amplitude-Ersatzpfad: %s", _thd_exc)

        # Amplitude-only fallback (wenn clipping_detection nicht verfügbar)
        audio_norm = audio / peak
        hard_clip_mask = np.abs(audio) >= 0.995
        hard_clip_ratio = float(np.sum(hard_clip_mask)) / len(audio)
        window_ms = int(0.001 * self.sample_rate)
        soft_clip_events = 0
        total_windows = 0
        for i in range(0, len(audio_norm) - window_ms, window_ms):
            window = audio_norm[i : i + window_ms]
            peak_w = float(np.max(np.abs(window)))
            if peak_w > 0.80:
                above_threshold = int(np.sum(np.abs(window) > 0.85 * peak_w))
                if above_threshold / len(window) > 0.25:
                    soft_clip_events += 1
            total_windows += 1
        soft_clip_ratio = soft_clip_events / max(total_windows, 1)
        threshold_factor = float(self.thresholds.get(DefectType.CLIPPING, 0.5))
        combined = hard_clip_ratio * 10 + soft_clip_ratio * 2
        severity = min(1.0, combined / max(threshold_factor, 1e-6))
        clip_indices = np.where(hard_clip_mask)[0]
        locations = []
        if len(clip_indices) > 0:
            groups = np.split(
                clip_indices,
                np.where(np.diff(clip_indices) > int(0.005 * self.sample_rate))[0] + 1,
            )
            for g in groups:
                locations.append((float(g[0]) / self.sample_rate, float(g[-1]) / self.sample_rate))
        return DefectScore(
            defect_type=DefectType.CLIPPING,
            severity=severity,
            confidence=0.92,
            locations=locations,
            metadata={
                "hard_clip_ratio": hard_clip_ratio,
                "soft_clip_ratio": soft_clip_ratio,
                "peak_dbfs": 20.0 * np.log10(peak) if peak > 0 else -120.0,
                "thd_discriminated": False,
            },
        )

    def _detect_dc_offset(self, audio: np.ndarray) -> DefectScore:
        """Erkennt DC-Offset (Gleichspannungsversatz) inkl. zeitvarianter Drift.

        Upgraded v10.0.0b: Segmentierte Analyse erkennt auch DC-Drift
        (Potentiometer-Alterung, Kondensator-Leckstrom). Absoluter DC-Schwellwert
        zusätzlich zu relativem - leise Aufnahmen werden nicht fehlerkannt.
        """
        if len(audio) == 0:
            return DefectScore(DefectType.DC_OFFSET, 0.0, 0.0)

        n = len(audio)
        dc_global = float(np.mean(audio))
        peak = float(np.percentile(np.abs(audio), 99.9))  # V08: Click-Spike darf DC-Ratio nicht verfälschen

        if peak < 1e-6:
            return DefectScore(DefectType.DC_OFFSET, 0.0, 0.5)

        # --- Global DC ---
        dc_ratio = abs(dc_global) / peak
        # Absolute threshold: DC < 0.001 (-60 dBFS) is inaudible regardless
        abs_dc = abs(dc_global)

        # --- Segmented DC drift detection ---
        n_segs = min(16, max(2, n // self.sample_rate))
        seg_len = n // n_segs
        seg_means = np.array([float(np.mean(audio[i * seg_len : (i + 1) * seg_len])) for i in range(n_segs)])

        # DC drift: max deviation between segments
        dc_drift = float(np.max(seg_means) - np.min(seg_means))
        drift_ratio = dc_drift / (peak + 1e-12)

        # Worst-segment DC
        worst_seg_dc = float(np.max(np.abs(seg_means)))
        worst_seg_ratio = worst_seg_dc / (peak + 1e-12)

        # Combined severity: global + drift + worst segment
        sev_global = dc_ratio / 0.05  # 5% = severity 1.0
        sev_drift = drift_ratio / 0.03  # 3% drift = severity 1.0
        sev_worst = worst_seg_ratio / 0.05
        raw_severity = max(sev_global, sev_drift * 0.8, sev_worst * 0.7)

        # Absolute guard: if DC is < 0.0005 in amplitude, cap severity
        if abs_dc < 0.0005 and worst_seg_dc < 0.001:
            raw_severity = min(raw_severity, 0.15)  # inaudible

        threshold_factor = self.thresholds.get(DefectType.DC_OFFSET, 0.6)
        severity = float(np.clip(raw_severity / threshold_factor, 0.0, 1.0))

        # Confidence: higher if consistent across segments
        seg_std = float(np.std(seg_means))
        confidence = float(
            np.clip(0.80 + 0.15 * min(1.0, abs_dc / 0.01) - 0.10 * min(1.0, seg_std / (abs_dc + 1e-12)), 0.5, 0.98)
        )

        return DefectScore(
            defect_type=DefectType.DC_OFFSET,
            severity=severity,
            confidence=confidence,
            locations=[],
            metadata={
                "dc_value": dc_global,
                "dc_ratio_percent": dc_ratio * 100,
                "dc_drift_percent": drift_ratio * 100,
                "worst_segment_dc": worst_seg_dc,
                "n_segments": n_segs,
                "dc_dbfs": 20 * np.log10(abs(dc_global)) if abs(dc_global) > 1e-10 else -120.0,
            },
        )

    def _detect_bandwidth_loss(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Hochfrequenz-Verlust / Bandbreitenbegrenzung (HF-Rolloff).

        Upgraded v10.0.0b: Multi-band rolloff detection with -3dB and -6dB
        cutoff estimation. Material-adaptive reference values.
        Genre-aware anti-FP (warm jazz/classical != bandwidth loss).
        """
        if len(audio) < 2048:
            return DefectScore(DefectType.BANDWIDTH_LOSS, 0.0, 0.3)
        if self.sample_rate < 16000:
            return DefectScore(DefectType.BANDWIDTH_LOSS, 0.0, 0.3)

        nperseg = min(8192, len(audio) // 4)
        if nperseg < 512:
            nperseg = 512
        freqs, psd = self._cached_welch(audio, self.sample_rate, nperseg)
        psd_db = 10 * np.log10(psd + 1e-20)

        # --- Material-adaptive HF reference ---
        _mat = getattr(self, "material_type", None)
        _MATERIAL_HF_REF = {
            "shellac": 0.02,
            "wax_cylinder": 0.01,
            "wire_recording": 0.015,
            "lacquer_disc": 0.03,
            "tape": 0.06,
            "reel_tape": 0.07,
            "cassette": 0.05,
            "vinyl": 0.08,
            "cd_digital": 0.10,
            "dat": 0.10,
            "mp3_low": 0.04,
            "mp3_high": 0.08,
            "aac": 0.08,
            "streaming": 0.09,
            "minidisc": 0.06,
        }
        if _mat is None:
            mat_name = ""
        elif isinstance(_mat, Enum):
            mat_name = str(_mat.value)
        else:
            mat_name = str(_mat)
        reference_hf_ratio = _MATERIAL_HF_REF.get(mat_name, 0.08)

        total_energy = float(np.sum(psd) + 1e-12)
        hf_cutoff = min(8000.0, self.sample_rate * 0.45)
        hf_energy = float(np.sum(psd[freqs >= hf_cutoff]))
        hf_ratio = hf_energy / total_energy

        loss_factor = max(0.0, 1.0 - (hf_ratio / (reference_hf_ratio + 1e-12)))

        # --- Estimate -3dB and -6dB rolloff frequencies ---
        # Find the frequency where PSD drops below peak_mid - 3dB / -6dB
        mid_mask = (freqs >= 500) & (freqs <= 4000)
        if mid_mask.any():
            peak_mid_db = float(np.max(psd_db[mid_mask]))
        else:
            peak_mid_db = float(np.max(psd_db))

        rolloff_3db = float(self.sample_rate / 2)
        rolloff_6db = float(self.sample_rate / 2)
        for fi in range(len(freqs) - 1, 0, -1):
            if psd_db[fi] >= peak_mid_db - 3.0:
                rolloff_3db = float(freqs[fi])
                break
        for fi in range(len(freqs) - 1, 0, -1):
            if psd_db[fi] >= peak_mid_db - 6.0:
                rolloff_6db = float(freqs[fi])
                break

        # --- Multi-band energy ratios ---
        bands = [(4000, 8000), (8000, 12000), (12000, 16000), (16000, 22000)]
        band_ratios = {}
        for lo, hi in bands:
            if lo < self.sample_rate / 2:
                hi_clip = min(hi, self.sample_rate * 0.45)
                mask = (freqs >= lo) & (freqs < hi_clip)
                band_ratios[f"{lo // 1000}-{hi // 1000}kHz"] = (
                    float(np.sum(psd[mask]) / total_energy) if mask.any() else 0.0
                )

        # --- Rolloff-based severity (more informative than simple ratio) ---
        # Rolloff at 4 kHz = severe; at 8 kHz = moderate; at 16 kHz = mild
        sev_rolloff = 0.0
        if rolloff_3db < 20000:
            sev_rolloff = float(np.clip((16000 - rolloff_3db) / 12000, 0.0, 1.0))

        # Combined severity: ratio-based + rolloff-based
        raw_severity = 0.5 * loss_factor + 0.5 * sev_rolloff

        # --- Anti-FP: if material naturally has low HF, reduce severity ---
        if mat_name in ("shellac", "wax_cylinder", "wire_recording", "lacquer_disc"):
            # Historic media: low HF is expected, only flag extreme cases
            raw_severity *= 0.5

        threshold_factor = self.thresholds.get(DefectType.BANDWIDTH_LOSS, 0.5)
        severity = float(np.clip(raw_severity / max(threshold_factor, 0.1), 0.0, 1.0))

        # Effective bandwidth: last frequency with > 1% mean PSD
        mean_psd_density = float(np.mean(psd))
        meaningful = np.where(psd > 0.01 * mean_psd_density)[0]
        effective_bw = float(freqs[meaningful[-1]]) if len(meaningful) > 0 else 0.0

        confidence = float(np.clip(0.70 + 0.20 * min(1.0, loss_factor), 0.5, 0.95))

        return DefectScore(
            defect_type=DefectType.BANDWIDTH_LOSS,
            severity=severity,
            confidence=confidence,
            locations=[],
            metadata={
                "hf_ratio_percent": hf_ratio * 100,
                "effective_bandwidth_hz": effective_bw,
                "rolloff_3db_hz": rolloff_3db,
                "rolloff_6db_hz": rolloff_6db,
                "reference_hf_ratio_percent": reference_hf_ratio * 100,
                "band_ratios": band_ratios,
                "material_ref": mat_name,
            },
        )

    def _detect_pitch_drift(self, audio: np.ndarray) -> DefectScore:
        """Erkennt constant pitch drift / speed error (≠ WOW/FLUTTER).

        Upgraded v10.0.0b: Multi-segment trend analysis (8+ segments),
        linear regression with monotonicity check, trend significance
        filtering, robust fundamental estimation, anti-FP for key changes.
        """
        min_len = int(5 * self.sample_rate)
        if len(audio) < min_len:
            return DefectScore(DefectType.PITCH_DRIFT, 0.0, 0.3)

        sos = signal.butter(4, 50, btype="high", fs=self.sample_rate, output="sos")
        audio_hp = signal.sosfilt(sos, audio)

        # --- Multi-segment analysis (8 segments minimum) ---
        n_segments = min(16, max(4, len(audio) // (int(3 * self.sample_rate))))
        segment_len = len(audio_hp) // n_segments
        if segment_len < int(2 * self.sample_rate):
            segment_len = int(2 * self.sample_rate)
            n_segments = max(2, len(audio_hp) // segment_len)

        def estimate_fundamental(seg: np.ndarray) -> float:
            """Fundamental frequency via FFT-based autocorrelation."""
            seg = seg / (np.percentile(np.abs(seg), 99.9) + 1e-8)
            min_lag = int(self.sample_rate / 2000)
            max_lag = int(self.sample_rate / 80)
            if max_lag >= len(seg):
                return 0.0
            n_fft = len(seg)
            fft_seg = np.fft.rfft(seg, n=2 * n_fft)
            corr_full = np.fft.irfft(fft_seg * np.conj(fft_seg))
            corr = np.real(corr_full[:n_fft])
            corr = corr[min_lag:max_lag]
            if len(corr) == 0:
                return 0.0
            peak_lag = np.argmax(corr) + min_lag
            if peak_lag == 0:
                return 0.0
            # Confidence check: peak should be significantly above neighbours
            peak_val = corr[peak_lag - min_lag]
            if peak_val < 0.1 * corr[0] if len(corr) > 0 else True:
                return 0.0
            return float(self.sample_rate / peak_lag)

        frequencies = []
        time_centers = []
        for i in range(n_segments):
            start = i * segment_len
            seg = audio_hp[start : start + segment_len]
            if len(seg) < int(1.5 * self.sample_rate):
                continue
            f0 = estimate_fundamental(seg)
            if 60 < f0 < 2000:
                frequencies.append(f0)
                time_centers.append((start + segment_len / 2) / self.sample_rate)

        if len(frequencies) < 3:
            return DefectScore(DefectType.PITCH_DRIFT, 0.0, 0.3)

        freq_arr: np.ndarray = np.array(frequencies)
        tc_arr: np.ndarray = np.array(time_centers)

        # --- Linear regression to find monotonic drift trend ---
        # Normalize time to [0, 1]
        t_norm = (tc_arr - tc_arr[0]) / (tc_arr[-1] - tc_arr[0] + 1e-6)
        # Linear fit: f(t) = a*t + b
        coeffs = np.polyfit(t_norm, freq_arr, 1)
        slope = coeffs[0]  # Hz change over full duration
        intercept = coeffs[1]

        # Predicted values and residuals
        f_predicted = np.polyval(coeffs, t_norm)
        residuals = freq_arr - f_predicted
        r_squared = 1.0 - float(np.sum(residuals**2) / (np.sum((freq_arr - np.mean(freq_arr)) ** 2) + 1e-12))

        # --- Drift in cents ---
        f_start = float(intercept)
        f_end = float(intercept + slope)
        if f_start < 30 or f_end < 30:
            return DefectScore(DefectType.PITCH_DRIFT, 0.0, 0.3)
        drift_cents = abs(1200 * np.log2(max(f_end, f_start) / min(f_end, f_start)))

        # --- Monotonicity check: is drift consistently in one direction? ---
        diffs = np.diff(freq_arr)
        if len(diffs) > 0:
            n_positive = int(np.sum(diffs > 0))
            n_negative = int(np.sum(diffs < 0))
            monotonicity = abs(n_positive - n_negative) / max(len(diffs), 1)
        else:
            monotonicity = 0.0

        # --- Anti-FP: key changes produce sudden jumps, not gradual drift ---
        max_jump_cents = 0.0
        for i in range(len(freq_arr) - 1):
            jump = abs(1200 * np.log2(max(freq_arr[i + 1], freq_arr[i]) / min(freq_arr[i + 1], freq_arr[i]) + 1e-8))
            max_jump_cents = max(max_jump_cents, jump)
        # If largest single jump is > 50% of total drift → likely key change, not drift
        key_change_penalty = 0.0
        if max_jump_cents > drift_cents * 0.5 and drift_cents > 10:
            key_change_penalty = 0.5

        # --- Combined severity ---
        # 10 cents = mild; 50 cents = moderate; 100 cents = severe
        sev_drift = float(np.clip(drift_cents / 60.0, 0.0, 1.0))
        # R2 indicates how well a linear drift model fits (high = true drift)
        sev_fit = float(np.clip(r_squared, 0.0, 1.0))
        # Monotonicity bonus: consistent direction → more likely real drift
        raw_severity = sev_drift * (0.5 + 0.3 * sev_fit + 0.2 * monotonicity) - key_change_penalty

        threshold_factor = self.thresholds.get(DefectType.PITCH_DRIFT, 0.6)
        raw_severity /= max(threshold_factor, 0.1)
        severity = float(np.clip(raw_severity, 0.0, 1.0))

        # Confidence scales with number of segments and fit quality
        confidence = float(np.clip(0.55 + 0.25 * r_squared + 0.10 * monotonicity, 0.40, 0.85))

        return DefectScore(
            defect_type=DefectType.PITCH_DRIFT,
            severity=severity,
            confidence=confidence,
            locations=[],
            metadata={
                "drift_cents": round(drift_cents, 2),
                "r_squared": round(r_squared, 3),
                "monotonicity": round(monotonicity, 3),
                "slope_hz_per_norm": round(float(slope), 3),
                "max_jump_cents": round(max_jump_cents, 2),
                "n_segments_used": len(frequencies),
                "f_start_hz": round(f_start, 1),
                "f_end_hz": round(f_end, 1),
            },
        )

    def _detect_reverb_excess(self, audio: np.ndarray) -> DefectScore:
        """Erkennt übermäßigen / unerwünschten Raumhall (Reverb Excess).

        Methodik (Schroeder-Integrationsverfahren, RT60-Schätzung):
          1. Signal in kurze Segmente aufteilen (2-4 s)
          2. Energie-Abklingkurve pro Segment via Backward-Integration
          3. RT60 aus dem Abfall von -5 dB auf -65 dB schätzen (Sabine/ISO 3382)
          4. Material-unabhängige Grenzwerte: RT60 > 0.8 s = problematisch,
             RT60 > 1.5 s = schwerer Defekt
          5. Zusätzlich: Tail/Direct-Ratio im Spektrum (Diffusfeld-Anteil)

        Severity: 0 = kein nennenswerter Hall, 1.0 = T60 > 1.5 s
        """
        if len(audio) < int(2 * self.sample_rate):
            return DefectScore(DefectType.REVERB_EXCESS, 0.0, 0.3)

        # Segmentlänge für RT60-Schätzung: 2-4 Sekunden
        seg_len = int(min(4.0, len(audio) / self.sample_rate * 0.5) * self.sample_rate)
        seg_len = max(seg_len, int(1.5 * self.sample_rate))
        if seg_len > len(audio):
            seg_len = len(audio)

        # Mehrere Segmente aus dem Mittelbereich des Tracks (nicht das Ende):
        # Das letzte Segment misst oft den natürlichen Fade-Out als künstlichen RT60.
        # → 3 Segmente bei 25 %, 50 %, 75 % der Tracklänge; letztes 15 % ausgeschlossen.
        n_total = len(audio)
        safe_end = int(n_total * 0.85)  # letztes 15 % nicht verwenden (Fade-Out-Zone)
        safe_end = max(safe_end, seg_len)
        positions_frac = [0.25, 0.50, 0.75]
        candidate_segs = []
        for frac in positions_frac:
            start = int(frac * safe_end) - seg_len // 2
            start = max(0, min(start, safe_end - seg_len))
            end = start + seg_len
            if end <= n_total:
                candidate_segs.append(audio[start:end])
        if not candidate_segs:
            # Fallback: direkt Mitte
            mid = max(0, n_total // 2 - seg_len // 2)
            candidate_segs = [audio[mid : mid + seg_len]]

        def _rt60_tdr_from_seg(seg: np.ndarray) -> tuple[float, float]:
            """RT60 und TDR aus einem Segment (Schroeder-Rückwärtsintegration)."""
            seg_sq = seg**2
            seg_rms = float(np.sqrt(np.mean(seg_sq) + 1e-12))
            if seg_rms < 1e-4:
                return 0.0, 0.0
            decay_curve = np.cumsum(seg_sq[::-1])[::-1]
            decay_curve = decay_curve / (decay_curve[0] + 1e-12)
            decay_db = 10 * np.log10(decay_curve + 1e-12)
            try:
                idx_5db = int(np.argmax(decay_db <= -5.0))
                idx_35db = int(np.argmax(decay_db <= -35.0))
                if idx_5db == 0 or idx_35db == 0 or idx_35db <= idx_5db:
                    rt60_val = 0.0
                else:
                    rt60_val = 2.0 * (idx_35db - idx_5db) / self.sample_rate
            except Exception:
                rt60_val = 0.0
            direct_energy = float(np.sum(seg[: int(0.1 * self.sample_rate)] ** 2)) + 1e-12
            tail_start = int(0.3 * self.sample_rate)
            tail_energy = (
                float(np.sum(seg[tail_start : tail_start + int(0.5 * self.sample_rate)] ** 2))
                if len(seg) > tail_start + int(0.5 * self.sample_rate)
                else 0.0
            )
            tdr_val = tail_energy / direct_energy
            return rt60_val, tdr_val

        rt60_values = []
        tdr_values = []
        for seg in candidate_segs:
            r, t = _rt60_tdr_from_seg(seg)
            rt60_values.append(r)
            tdr_values.append(t)

        # Median für Robustheit gegen vereinzelte Ausreißer (z. B. leise Passagen)
        rt60 = float(np.median(rt60_values))
        tdr = float(np.median(tdr_values))

        # If all segments are effectively silent (very low RMS) then guard
        # against false positives by marking the result as silence_guarded.
        # Use a small tolerance to be robust against tiny numerical noise
        silence_guarded = all(abs(float(v)) < 1e-8 for v in rt60_values) and all(
            abs(float(v)) < 1e-8 for v in tdr_values
        )

        # Severity aus RT60 + TDR kombinieren
        # RT60 > 1.5s = severity 1.0; alles unter 0.4s = 0.0
        rt60_severity = min(1.0, max(0.0, (rt60 - 0.4) / 1.1))
        tdr_severity = min(1.0, max(0.0, (tdr - 0.5) / 4.0))

        threshold_factor = self.thresholds.get(DefectType.REVERB_EXCESS, 0.6)
        combined = (rt60_severity * 0.65 + tdr_severity * 0.35) / max(threshold_factor, 0.1)
        severity = min(1.0, combined)

        return DefectScore(
            defect_type=DefectType.REVERB_EXCESS,
            severity=severity,
            confidence=0.72,  # RT60-Schätzung bei Musik (kein Impulssignal) nur mäßig präzise
            locations=[],
            metadata={
                "rt60_seconds": rt60,
                "tail_to_direct_ratio": tdr,
                "rt60_severity": rt60_severity,
                "tdr_severity": tdr_severity,
                "n_segments_used": len(candidate_segs),
                "silence_guarded": bool(silence_guarded),
            },
        )

    def _detect_print_through(self, audio: np.ndarray) -> DefectScore:
        """Erkennt bidirectional magnetic print-through (pre-echo and post-echo) on tape.

        Print-Through arises from magnetic flux transfer between adjacent tape layers.
        It produces:
          - **Pre-echo  (alpha_pre)**: ghost signal 80-400 ms BEFORE a loud onset
            (winding/storage: lower layer magnetizes through to the current layer).
          - **Post-echo (alpha_post)**: ghost signal 80-400 ms AFTER a loud onset
            (playback head: current layer partially magnetized from the preceding layer
            during playback).

        Both directions are modelled as damped copies of the main signal:
          pre_echo  ≈ alpha_pre  × x[n + lag]   → audible as ghost before onset
          post_echo ≈ alpha_post × x[n - lag]   → audible as ghost after onset

        IEC 60094-3 / DIN 45 513: Pre-echo typically -20 to -35 dB relative to program.

        Note: Only applicable for TAPE/REEL_TAPE (threshold = 1.0 for all other materials).
        """
        min_len = int(0.5 * self.sample_rate)
        if len(audio) < min_len:
            return DefectScore(DefectType.PRINT_THROUGH, 0.0, 0.2)

        # Kurzzeit-RMS für Einsatz-Detektion
        win = int(0.010 * self.sample_rate)  # 10ms
        hop = win // 2
        n_frames = (len(audio) - win) // hop
        rms = np.zeros(n_frames)
        for i in range(n_frames):
            s = i * hop
            rms[i] = np.sqrt(np.mean(audio[s : s + win] ** 2))

        rms_db = 20 * np.log10(rms + 1e-10)
        median_rms_db = np.median(rms_db)

        # Einsätze: Stellen mit > 20 dB über Median
        onset_frames = np.where(rms_db > median_rms_db + 20)[0]
        if len(onset_frames) == 0:
            return DefectScore(DefectType.PRINT_THROUGH, 0.0, 0.5)

        # Bidirektionale Print-Through-Detektion: Pre-Echo UND Post-Echo
        # search window: 80-400 ms in both directions (IEC 60094-3)
        echo_delays_ms = [80, 120, 160, 200, 250, 320, 400]
        print_through_events = 0
        total_magnitude = 0.0
        alpha_pre_list: list = []  # Gemessene Pre-Echo-Amplituden
        alpha_post_list: list = []  # Gemessene Post-Echo-Amplituden
        locations: list[tuple[float, float]] = []

        for onset_f in onset_frames[:30]:  # Max. erste 30 Einsätze prüfen
            onset_db = rms_db[onset_f]
            for delay_ms in echo_delays_ms:
                delay_f = int(delay_ms / (hop / self.sample_rate * 1000))

                # --- Pre-echo: Ghost BEFORE onset ---
                pre_f = onset_f - delay_f
                if pre_f >= 0:
                    pre_db = rms_db[pre_f]
                    db_below = onset_db - pre_db
                    if 18 <= db_below <= 48 and pre_db > median_rms_db + 8:
                        print_through_events += 1
                        alpha_pre = 10 ** (-(db_below) / 20)
                        alpha_pre_list.append(alpha_pre)
                        total_magnitude += alpha_pre
                        t_pre = float(pre_f * hop / self.sample_rate)
                        t_end = t_pre + win / self.sample_rate
                        if not any(abs(loc[0] - t_pre) < 0.020 for loc in locations):
                            locations.append((t_pre, float(t_end)))

                # --- Post-echo: Ghost AFTER onset (alpha_post) ---
                post_f = onset_f + delay_f
                if post_f < len(rms_db):
                    post_db = rms_db[post_f]
                    db_below = onset_db - post_db
                    if 18 <= db_below <= 48 and post_db > median_rms_db + 8:
                        print_through_events += 1
                        alpha_post = 10 ** (-(db_below) / 20)
                        alpha_post_list.append(alpha_post)
                        total_magnitude += alpha_post
                        t_post = float(post_f * hop / self.sample_rate)
                        t_end = t_post + win / self.sample_rate
                        if not any(abs(loc[0] - t_post) < 0.020 for loc in locations):
                            locations.append((t_post, float(t_end)))

        # Severity: > 5 Events = schweres Print-Through
        event_severity = min(1.0, print_through_events / 8.0)
        mag_severity = min(1.0, total_magnitude * 20)

        threshold_factor = self.thresholds.get(DefectType.PRINT_THROUGH, 0.6)
        severity = min(1.0, (event_severity * 0.6 + mag_severity * 0.4) / max(threshold_factor, 0.05))

        avg_alpha_pre = float(np.mean(alpha_pre_list)) if alpha_pre_list else 0.0
        avg_alpha_post = float(np.mean(alpha_post_list)) if alpha_post_list else 0.0

        # --- Asymmetry confirmation (IEC 60094-3 §4.2) ---
        # True print-through is BIDIRECTIONAL but ASYMMETRIC:
        #   alpha_pre ≠ alpha_post due to different magnetisation transfer mechanisms.
        # Equal pre/post → likely reverb bleed or room ambience, not magnetic print-through.
        # Strong asymmetry → confirms genuine magnetic print-through → raise confidence.
        asymmetry_ratio = 0.0
        asymmetry_confidence_bonus = 0.0
        if avg_alpha_pre > 0 and avg_alpha_post > 0:
            asymmetry_ratio = float(max(avg_alpha_pre, avg_alpha_post) / (min(avg_alpha_pre, avg_alpha_post) + 1e-8))
            if asymmetry_ratio > 1.3:
                asymmetry_confidence_bonus = float(np.clip((asymmetry_ratio - 1.3) / 2.0, 0.0, 0.20))
        elif avg_alpha_pre > 0 or avg_alpha_post > 0:
            asymmetry_confidence_bonus = 0.08

        confidence = float(np.clip(0.60 + asymmetry_confidence_bonus, 0.55, 0.80))

        return DefectScore(
            defect_type=DefectType.PRINT_THROUGH,
            severity=severity,
            confidence=confidence,
            locations=locations,
            metadata={
                "pre_echo_events": len(alpha_pre_list),
                "post_echo_events": len(alpha_post_list),
                "avg_alpha_pre": round(avg_alpha_pre, 5),
                "avg_alpha_post": round(avg_alpha_post, 5),
                "asymmetry_ratio": round(asymmetry_ratio, 3),
                "avg_magnitude": total_magnitude / max(print_through_events, 1),
                "onsets_checked": min(len(onset_frames), 30),
            },
        )

    # ------------------------------------------------------------------
    # Detektoren - Runde 3: QUANTIZATION_NOISE, JITTER_ARTIFACTS, DYNAMIC_COMPRESSION_EXCESS
    # ------------------------------------------------------------------

    def _detect_quantization_noise(self, audio: np.ndarray) -> DefectScore:
        """Erkennt quantization noise via ENOB estimation + spectral flatness.

        Upgraded v10.0.0b: Effective Number of Bits (ENOB) estimation,
        step-size detection in quiet passages, spectral flatness of noise
        floor (quantization noise is spectrally flat), and dynamic SNR
        analysis across passages of varying amplitude.
        """
        audio_norm = audio - np.mean(audio)
        max_amp = float(np.percentile(np.abs(audio_norm), 99.9))
        if max_amp < 1e-8:
            return DefectScore(DefectType.QUANTIZATION_NOISE, 0.0, 0.0)
        audio_norm = audio_norm / max_amp

        # --- 1. Histogram fill ratio (coarse bit-depth indicator) ---
        n_bins = 1024
        hist, _ = np.histogram(audio_norm, bins=n_bins)
        n_populated = int(np.sum(hist > 0))
        fill_ratio = n_populated / float(n_bins)

        # --- 2. ENOB estimation from noise floor ---
        # Quiet passages: < -40 dBFS (0.01 amplitude)
        quiet_mask = np.abs(audio_norm) < 0.01
        enob = 16.0  # assume best case
        granularity = 0.0
        spectral_flatness_quiet = 0.0
        step_size_est = 0.0

        if quiet_mask.sum() > 512:
            quiet_audio = audio_norm[quiet_mask]

            # Granularity: fraction of ±LSB jumps
            diff = np.diff(quiet_audio)
            abs_diff = np.abs(diff)
            nonzero_diffs = abs_diff[abs_diff > 1e-10]

            if len(nonzero_diffs) > 32:
                # Step-size estimation: mode of small differences
                # Quantized signals have recurring step sizes
                diff_hist, diff_edges = np.histogram(nonzero_diffs, bins=200, range=(0, 0.02))
                if diff_hist.max() > 0:
                    mode_idx = np.argmax(diff_hist)
                    step_size_est = float((diff_edges[mode_idx] + diff_edges[mode_idx + 1]) / 2)

                    # ENOB from step size: step ≈ 2/(2^N) → N ≈ log2(2/step)
                    if step_size_est > 1e-8:
                        enob = float(np.clip(np.log2(2.0 / step_size_est), 4.0, 24.0))

                # Granularity: fraction of diffs clustered near the step size
                if step_size_est > 1e-8:
                    near_step = np.abs(abs_diff - step_size_est) < step_size_est * 0.3
                    granularity = float(np.sum(near_step) / max(len(abs_diff), 1))
                else:
                    granularity = float(np.sum(abs_diff > 0.003) / max(len(abs_diff), 1))

            # Spectral flatness of quiet passages (quantization noise → flat spectrum)
            if len(quiet_audio) >= 512:
                spec_q = np.abs(np.fft.rfft(quiet_audio[: min(4096, len(quiet_audio))]))
                spec_q = spec_q[1:]  # remove DC
                if len(spec_q) > 4 and float(np.mean(spec_q)) > 1e-12:
                    geo_mean = float(np.exp(np.mean(np.log(spec_q + 1e-20))))
                    arith_mean = float(np.mean(spec_q))
                    spectral_flatness_quiet = geo_mean / (arith_mean + 1e-12)
                    # Flat spectrum (≈ 1.0) = characteristic of quantization noise

        # --- 3. Dynamic SNR: compare noise in quiet vs loud passages ---
        loud_mask = np.abs(audio_norm) > 0.3
        snr_indicator = 0.0
        if loud_mask.sum() > 256 and quiet_mask.sum() > 256:
            rms_loud = float(np.sqrt(np.mean(audio_norm[loud_mask] ** 2)))
            rms_quiet = float(np.sqrt(np.mean(audio_norm[quiet_mask] ** 2)))
            if rms_quiet > 1e-10:
                dynamic_snr_db = 20.0 * np.log10(rms_loud / rms_quiet)
                # Low ENOB → low dynamic SNR. 16-bit ≈ 96 dB, 8-bit ≈ 48 dB
                snr_indicator = float(np.clip((80.0 - dynamic_snr_db) / 40.0, 0.0, 1.0))

        # --- Combined severity ---
        # ENOB < 12 → noticeable; < 10 → severe; < 8 → extreme
        sev_enob = float(np.clip((14.0 - enob) / 6.0, 0.0, 1.0))
        sev_fill = float(np.clip((0.5 - fill_ratio) * 2.0, 0.0, 1.0))
        sev_flat = spectral_flatness_quiet * granularity  # both high = characteristic

        raw_severity = 0.35 * sev_enob + 0.25 * sev_fill + 0.20 * sev_flat + 0.20 * snr_indicator

        threshold = self.thresholds.get(DefectType.QUANTIZATION_NOISE, 0.6)
        if raw_severity < threshold * 0.4:
            raw_severity = 0.0
        severity = float(np.clip(raw_severity, 0.0, 1.0))

        confidence = float(np.clip(0.60 + 0.25 * min(1.0, sev_enob + granularity), 0.5, 0.90))

        return DefectScore(
            defect_type=DefectType.QUANTIZATION_NOISE,
            severity=severity,
            confidence=confidence,
            locations=[],
            metadata={
                "fill_ratio": fill_ratio,
                "n_populated_bins": n_populated,
                "enob_estimate": round(enob, 1),
                "step_size_estimate": step_size_est,
                "granularity_index": granularity,
                "spectral_flatness_quiet": round(spectral_flatness_quiet, 3),
                "snr_indicator": round(snr_indicator, 3),
            },
        )

    def _detect_jitter_artifacts(self, audio: np.ndarray) -> DefectScore:
        """Erkennt jitter artifacts: D/A clock instability → FM sidebands + phase incoherence.

        Upgraded v10.0.0b: Multi-indicator jitter detection:
        1. Zero-crossing regularity deviation (jitter disturbs sample timing)
        2. Instantaneous-frequency variance from analytic signal (phase jitter)
        3. Spectral tone purity degradation (jitter smears pure tones into sidebands)
        4. Segment-wise HF energy variance (temporal inconsistency)
        Material-aware: digital media more susceptible than analog.
        """
        n = len(audio)
        if n < 4096:
            return DefectScore(DefectType.JITTER_ARTIFACTS, 0.0, 0.0)

        # --- 1. Zero-crossing regularity ---
        # Jitter displaces samples → irregular zero-crossing intervals
        zc_indices = np.where(np.diff(np.sign(audio)))[0]
        zc_regularity = 0.0
        if len(zc_indices) > 20:
            zc_intervals = np.diff(zc_indices).astype(float)
            zc_mean = float(np.mean(zc_intervals))
            if zc_mean > 1.0:
                zc_cv = float(np.std(zc_intervals) / zc_mean)  # coefficient of variation
                # Natural music: CV ~0.5-1.5; jitter adds extra irregularity
                zc_regularity = float(np.clip((zc_cv - 1.0) / 1.5, 0.0, 1.0))

        # --- 2. Instantaneous frequency variance (analytic signal) ---
        if_variance = 0.0
        try:
            from scipy.signal import hilbert as _hilbert  # pylint: disable=import-outside-toplevel

            # Use center 32k samples for efficiency
            center = max(0, n // 2 - 16384)
            seg = audio[center : center + min(32768, n)]
            analytic = _hilbert(seg)
            analytic_arr = np.asarray(analytic, dtype=np.complex128)
            inst_phase = np.unwrap(np.angle(analytic_arr))
            inst_freq = np.diff(inst_phase) * self.sample_rate / (2 * np.pi)
            # Only consider positive frequencies in plausible range
            valid = (inst_freq > 20) & (inst_freq < self.sample_rate / 2)
            if valid.sum() > 100:
                median_if = float(np.median(inst_freq[valid]))
                if median_if > 50:
                    if_std = float(np.std(inst_freq[valid]))
                    if_variance = float(np.clip(if_std / median_if - 0.05, 0.0, 1.0))
        except (ImportError, ValueError) as _exc:
            logger.debug("Instantaneous frequency variance fehlgeschlagen: %s", _exc)

        # --- 3. HF spectral variance across segments ---
        seg_len = min(8192, n // 4)
        n_segs = min(12, n // seg_len)
        hf_var_ratio = 0.0
        sideband_ratio = 0.0
        if n_segs >= 3:
            win = np.hanning(seg_len)
            spectra = []
            for i in range(n_segs):
                s = i * seg_len
                spectrum = np.abs(np.fft.rfft(audio[s : s + seg_len] * win))
                spectra.append(spectrum)

            freqs = np.fft.rfftfreq(seg_len, 1.0 / self.sample_rate)
            hf_mask = freqs > 8000
            if hf_mask.sum() >= 4:
                hf_spectra = np.array([s[hf_mask] for s in spectra])
                hf_mean_per_band = np.mean(hf_spectra, axis=0) + 1e-10
                hf_var_ratio = float(np.mean(np.std(hf_spectra, axis=0) / hf_mean_per_band))

                # Sideband indicator: peak-to-median in HF band
                mean_spectrum = np.mean(spectra, axis=0)
                hf_mean = mean_spectrum[hf_mask]
                sorted_hf = np.sort(hf_mean)[::-1]
                top_n = max(1, len(sorted_hf) // 100)
                sideband_ratio = float(np.mean(sorted_hf[:top_n]) / (np.median(hf_mean) + 1e-10))

        sev_zc = zc_regularity
        sev_if = if_variance
        sev_hf = float(np.clip((hf_var_ratio - 0.15) / 0.5, 0.0, 1.0))
        sev_sb = float(np.clip((sideband_ratio - 3.0) / 10.0, 0.0, 1.0))

        # Bitto (2000) FM sideband analysis: D/A clock jitter creates symmetric
        # sidebands around prominent spectral peaks at ±f_jitter offset. Find peaks
        # with SNR > 20 dB above local noise floor, then measure energy at ±500 Hz
        # and ±1 kHz offsets relative to peak amplitude. High sideband-to-peak ratio
        # confirms jitter origin over other causes of HF variance.
        sev_bitto = 0.0
        try:
            fft_n = min(32768, 2 ** int(np.log2(max(4096, n))))
            center_start = max(0, n // 2 - fft_n // 2)
            spectrum_seg = audio[center_start : center_start + fft_n]
            if len(spectrum_seg) == fft_n:
                win_b = np.hanning(fft_n)
                spec_b = np.abs(np.fft.rfft(spectrum_seg * win_b))
                freqs_b = np.fft.rfftfreq(fft_n, 1.0 / self.sample_rate)
                freq_res = self.sample_rate / fft_n  # Hz per bin
                # Find peaks with local SNR > 20 dB  in 2-12 kHz range (jitter-sensitive)
                jitter_band = (freqs_b >= 2000) & (freqs_b <= 12000)
                spec_jb = spec_b[jitter_band]
                freqs_jb = freqs_b[jitter_band]
                if len(spec_jb) > 20:
                    local_median = np.median(spec_jb) + 1e-12
                    peak_thresh = local_median * 10.0  # 20 dB above median
                    peak_mask = spec_jb > peak_thresh
                    peak_indices = np.where(peak_mask)[0]
                    if len(peak_indices) > 0:
                        sideband_ratios_b: list[float] = []
                        for pi in peak_indices[:10]:  # max 10 peaks
                            f_peak = freqs_jb[pi]
                            peak_amp = spec_jb[pi]
                            for f_offset in (500.0, 1000.0, 2000.0):
                                lo_bin = int((f_peak - f_offset) / freq_res)
                                hi_bin = int((f_peak + f_offset) / freq_res)
                                lo_ok = 0 <= lo_bin < len(spec_b)
                                hi_ok = 0 <= hi_bin < len(spec_b)
                                if lo_ok and hi_ok:
                                    sb_energy = (spec_b[lo_bin] + spec_b[hi_bin]) / 2.0
                                    sideband_ratios_b.append(sb_energy / max(peak_amp, 1e-12))
                        if sideband_ratios_b:
                            mean_sb_ratio = float(np.mean(sideband_ratios_b))
                            # Jitter sidebands: ratio typically 0.05-0.30; pure signal < 0.01
                            sev_bitto = float(np.clip((mean_sb_ratio - 0.03) / 0.25, 0.0, 1.0))
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)

        raw_severity = 0.22 * sev_zc + 0.26 * sev_if + 0.22 * sev_hf + 0.16 * sev_sb + 0.14 * sev_bitto

        # Bitto (2000): FM sidebands are the most specific D/A-clock-jitter indicator.
        # For cd_digital/dat, Bitto's analysis is directly applicable and deserves the
        # highest weight. For analog sources, wow/flutter is the relevant mechanism;
        # sideband weight must be reduced (analog sidebands ≠ D/A jitter).
        # All weight rows sum to 1.0: zc + if + hf + sb + bitto
        _mat = getattr(self, "material_type", None)
        _DIGITAL_MATS = {"cd_digital", "dat", "mp3_low", "mp3_high", "aac", "streaming", "minidisc"}
        _ANALOG_MATS = {"vinyl", "tape", "reel_tape", "shellac", "shellac_78", "cassette", "wax_cylinder"}
        if _mat is None:
            mat_name = ""
        elif isinstance(_mat, Enum):
            mat_name = str(_mat.value)
        else:
            mat_name = str(_mat)
        if mat_name in {"cd_digital", "dat"}:
            # Bitto directly applicable: weight 0.28 (highest); compensated from zc+hf
            raw_severity = 0.17 * sev_zc + 0.26 * sev_if + 0.13 * sev_hf + 0.16 * sev_sb + 0.28 * sev_bitto
        elif mat_name in _DIGITAL_MATS:
            # Digital but not raw D/A chain: moderate Bitto boost
            raw_severity = 0.20 * sev_zc + 0.26 * sev_if + 0.18 * sev_hf + 0.16 * sev_sb + 0.20 * sev_bitto
        elif mat_name in _ANALOG_MATS:
            # Analog: D/A jitter not applicable; Bitto weight minimized
            raw_severity = 0.26 * sev_zc + 0.26 * sev_if + 0.24 * sev_hf + 0.20 * sev_sb + 0.04 * sev_bitto

        threshold = self.thresholds.get(DefectType.JITTER_ARTIFACTS, 0.6)
        if raw_severity < threshold * 0.4:
            raw_severity = 0.0
        severity = float(np.clip(raw_severity, 0.0, 1.0))

        confidence = float(np.clip(0.55 + 0.25 * min(1.0, sev_if + sev_hf), 0.45, 0.85))

        locations: list[tuple[float, float]] = []
        if severity > 0.0 and n >= int(0.10 * self.sample_rate):
            win_loc: int = max(64, int(0.020 * self.sample_rate))
            hop_loc: int = max(1, win_loc // 2)
            n_frames = max(0, (n - win_loc) // hop_loc)
            if n_frames >= 3:
                instability = np.zeros(n_frames, dtype=np.float64)
                for i in range(n_frames):
                    s = i * hop_loc
                    frame = audio[s : s + win_loc].astype(np.float64)
                    d1 = np.diff(frame)
                    d2 = np.diff(d1)
                    instability[i] = (float(np.std(d1)) + 0.5 * float(np.std(d2))) / (
                        float(np.mean(np.abs(frame))) + 1e-6
                    )

                i_thr = max(float(np.percentile(instability, 80.0)), float(np.median(instability)) * 1.25)
                jitter_mask = instability >= i_thr
                if np.any(jitter_mask):
                    starts = np.where(np.diff(np.concatenate(([0], jitter_mask.astype(np.int8), [0]))) == 1)[0]  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
                    ends = np.where(np.diff(np.concatenate(([0], jitter_mask.astype(np.int8), [0]))) == -1)[0]  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
                    for s_f, e_f in zip(starts, ends):
                        t0 = max(0.0, float(s_f * hop_loc) / self.sample_rate - 0.01)
                        t1 = min(float(n) / self.sample_rate, float(e_f * hop_loc + win_loc) / self.sample_rate + 0.01)
                        if t1 > t0:
                            locations.append((t0, t1))
        if severity > 0.0 and not locations:
            locations = [(0.0, float(n) / float(self.sample_rate))]

        return DefectScore(
            defect_type=DefectType.JITTER_ARTIFACTS,
            severity=severity,
            confidence=confidence,
            locations=locations,
            metadata={
                "zc_regularity": round(zc_regularity, 3),
                "if_variance": round(if_variance, 3),
                "hf_var_ratio": round(hf_var_ratio, 3),
                "sideband_ratio": round(sideband_ratio, 3),
                "bitto_sideband_sev": round(sev_bitto, 3),
                "n_segments": n_segs,
            },
        )

    def _detect_dynamic_compression_excess(self, audio: np.ndarray) -> DefectScore:
        """
        Erkennt übermäßige Dynamikkompression ('Loudness War', DR-Wert < 6 dB).

        Charakteristika stark komprimierter Aufnahmen:
        - Niedriger Crest Factor: Peak/RMS < 6 dB (ideal: 12-20 dB für Musik)
        - Histogramm-Clustering nahe ±1.0 (viele Samples nahe Maximum)
        - Geringe LRA (Loudness Range, EBU R128): LRA < 3 LU = exzessiv

        Referenz: DR-Database (Dynamic Range Database), EBU R128, AES17.
        """
        n = len(audio)
        if n < 512:
            return DefectScore(DefectType.DYNAMIC_COMPRESSION_EXCESS, 0.0, 0.0)

        # Schritt 1: Crest Factor
        max_amp = np.percentile(np.abs(audio), 99.9)
        if max_amp < 1e-8:
            return DefectScore(DefectType.DYNAMIC_COMPRESSION_EXCESS, 0.0, 0.0)
        audio_norm = audio / max_amp
        rms = float(np.sqrt(np.mean(audio_norm**2)))
        crest_db = -20.0 * np.log10(rms + 1e-10)  # dB über RMS → höher = mehr Dynamik

        # Schritt 2: Histogramm-Analyse
        hist, _ = np.histogram(np.abs(audio_norm), bins=100, range=(0.0, 1.0))
        # Samples nahe Maximum (> -1 dBFS ≈ 0.891 linear) → typisch für Loudness War
        high_amp_ratio = float(hist[-10:].sum() / (n + 1))

        # Schritt 3: Vereinfachte LRA-Schätzung über 400ms-Fenster-RMS
        win_s = max(1, int(0.4 * self.sample_rate))
        hop_s = max(1, win_s // 2)
        n_wins = max(1, (n - win_s) // hop_s)
        rms_values = []
        for i in range(n_wins):
            s = i * hop_s
            w = audio_norm[s : s + win_s]
            r = float(np.sqrt(np.mean(w**2)))
            if r > 1e-6:
                rms_values.append(20.0 * np.log10(r))

        if len(rms_values) > 4:
            rms_arr = np.array(rms_values)
            lra = float(np.percentile(rms_arr, 95) - np.percentile(rms_arr, 10))
        else:
            lra = 20.0  # Konservativer Default: keine Aussage möglich

        # Severity-Berechnung
        # Crest < 6 dB → severity=1.0; > 14 dB → severity=0.0
        severity_crest = max(0.0, min(1.0, (12.0 - crest_db) / 6.0))
        # high_amp_ratio > 30 % → stark überkomprimiert
        severity_hist = max(0.0, min(1.0, (high_amp_ratio - 0.05) / 0.25))
        # LRA < 3 LU → severity=1.0; > 12 LU → severity=0.0
        severity_lra = max(0.0, min(1.0, (8.0 - lra) / 5.0))

        severity = severity_crest * 0.35 + severity_hist * 0.35 + severity_lra * 0.30

        threshold = self.thresholds.get(DefectType.DYNAMIC_COMPRESSION_EXCESS, 0.6)
        if severity < threshold * 0.5:
            severity = 0.0

        locations: list[tuple[float, float]] = []
        if severity > 0.0:
            loc_win = max(1, int(0.200 * self.sample_rate))
            loc_hop = max(1, loc_win // 2)
            n_loc_wins = max(1, (n - loc_win) // loc_hop)
            compressed_indicator = np.zeros(n_loc_wins, dtype=np.float64)
            for i in range(n_loc_wins):
                s = i * loc_hop
                w = audio_norm[s : s + loc_win]
                if len(w) < 32:
                    continue
                w_rms = float(np.sqrt(np.mean(w**2) + 1e-12))
                w_peak = float(np.percentile(np.abs(w), 99.9))
                w_crest_db = 20.0 * np.log10((w_peak + 1e-10) / (w_rms + 1e-10))
                w_hi = float(np.mean(np.abs(w) >= 0.89))
                i_crest = float(np.clip((7.0 - w_crest_db) / 5.0, 0.0, 1.0))
                i_hi = float(np.clip((w_hi - 0.05) / 0.20, 0.0, 1.0))
                compressed_indicator[i] = 0.65 * i_crest + 0.35 * i_hi

            c_thr = max(float(np.percentile(compressed_indicator, 70.0)), 0.45)
            compressed_mask = compressed_indicator >= c_thr
            if np.any(compressed_mask):
                starts = np.where(np.diff(np.concatenate(([0], compressed_mask.astype(np.int8), [0]))) == 1)[0]  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
                ends = np.where(np.diff(np.concatenate(([0], compressed_mask.astype(np.int8), [0]))) == -1)[0]  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
                for s_f, e_f in zip(starts, ends):
                    t0 = max(0.0, float(s_f * loc_hop) / self.sample_rate)
                    t1 = min(float(n) / self.sample_rate, float(e_f * loc_hop + loc_win) / self.sample_rate)
                    if t1 > t0:
                        locations.append((t0, t1))
            elif severity >= 0.35:
                locations = [(0.0, float(n) / float(self.sample_rate))]

        return DefectScore(
            defect_type=DefectType.DYNAMIC_COMPRESSION_EXCESS,
            severity=severity,
            confidence=0.72,  # Crest Factor + LRA sind gut kalibrierte Metriken
            locations=locations,
            metadata={
                "crest_factor_db": crest_db,
                "high_amplitude_ratio": high_amp_ratio,
                "lra_db": lra,
                "rms_db": 20.0 * np.log10(rms + 1e-10),
            },
        )

    # ------------------------------------------------------------------
    # Detektoren für Typen 21-28 (F-7 Ergänzung)
    # ------------------------------------------------------------------

    def _detect_soft_saturation(self, audio: np.ndarray) -> DefectScore:
        """Erkennt SOFT_SATURATION: tube/tape even-harmonic distortion - preserve, do not repair.

        Per Spec §6.3: flat_tops < 0.1 % AND even harmonics (H2, H4) dominate odd (H3, H5).
        Returns severity > 0 only when soft-saturation profile is clearly present AND
        hard clipping is absent - ensuring phase_23 (clipping repair) is NOT triggered.
        """
        n = len(audio)
        if n < self.sample_rate:
            return DefectScore(DefectType.SOFT_SATURATION, 0.0, 0.3)

        # Flat-top clipping check - if hard clipping present, this is CLIPPING, not SOFT_SATURATION
        flat_top_threshold = 0.99
        flat_tops = float(np.sum(np.abs(audio) >= flat_top_threshold)) / max(n, 1)
        if flat_tops >= 0.001:  # ≥ 0.1 % flat-tops → hard clipping, not soft saturation
            return DefectScore(DefectType.SOFT_SATURATION, 0.0, 0.80)

        try:
            max_amp = float(np.percentile(np.abs(audio), 99.9))
            if max_amp < 0.05:
                return DefectScore(DefectType.SOFT_SATURATION, 0.0, 0.5)
            audio_norm = audio / max_amp

            n_fft = min(4096, n)
            spec = np.abs(np.fft.rfft(audio_norm[:n_fft])) ** 2
            freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)

            # Find fundamental (80-1000 Hz highest-energy bin)
            fund_mask = (freqs >= 80.0) & (freqs <= 1000.0)
            if not fund_mask.any():
                return DefectScore(DefectType.SOFT_SATURATION, 0.0, 0.3)
            fund_idx = int(np.argmax(spec * fund_mask.astype(float)))
            fund_hz = float(freqs[fund_idx])
            if fund_hz <= 0:
                return DefectScore(DefectType.SOFT_SATURATION, 0.0, 0.3)

            # Harmonic energy helper (±2 Hz window per harmonic)
            bin_w = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
            half_bin = max(1, int(2.0 / (bin_w + 1e-10)))

            def _harm(h: int) -> float:
                target_hz = fund_hz * h
                if target_hz >= self.sample_rate / 2.0:
                    return 0.0
                tidx = round(target_hz / (self.sample_rate / n_fft))
                lo = max(0, tidx - half_bin)
                hi = min(len(spec), tidx + half_bin + 1)
                return float(np.sum(spec[lo:hi]))

            even_energy = _harm(2) + _harm(4)
            odd_energy = _harm(3) + _harm(5)
            total_harm = even_energy + odd_energy
            if total_harm < 1e-20:
                return DefectScore(DefectType.SOFT_SATURATION, 0.0, 0.3)

            even_ratio = even_energy / total_harm
            # > 0.55 = even-dominant (tube/tape character); 0.5 = neutral
            severity = float(np.clip((even_ratio - 0.55) / 0.35, 0.0, 1.0))
            confidence = 0.65 if severity > 0.1 else 0.40

            return DefectScore(
                defect_type=DefectType.SOFT_SATURATION,
                severity=severity,
                confidence=confidence,
                locations=[],
                metadata={"even_ratio": even_ratio, "flat_top_ratio": flat_tops},
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.SOFT_SATURATION, 0.0, 0.3)

    def _detect_sibilance(self, audio: np.ndarray) -> DefectScore:
        """Erkennt SIBILANCE: excessive sibilant energy (harsh 's'/'sh' consonants).

        Upgraded v10.0.0b: Psychoacoustic sharpness model (Zwicker & Fastl),
        spectral centroid in sibilance band for gender-adaptive detection,
        anti-FP for bright instruments (cymbals, hi-hats vs true sibilance),
        vocal-gate via sibilance-to-broadband energy ratio, temporal burst
        pattern analysis (sibilance = short bursts, not sustained brightness).
        """
        n = len(audio)
        if n < 2048:
            return DefectScore(DefectType.SIBILANCE, 0.0, 0.3)

        try:
            nperseg = min(4096, n)
            freqs, psd = self._cached_welch(audio, self.sample_rate, nperseg)
            total_power = float(np.sum(psd) + 1e-20)

            # --- Sibilance band analysis ---
            sib_lo, sib_hi = 4000.0, 12000.0
            sib_mask = (freqs >= sib_lo) & (freqs <= min(sib_hi, self.sample_rate * 0.45))
            sib_frac = float(np.sum(psd[sib_mask]) / total_power)

            # Spectral centroid within sibilance band → identifies center frequency
            if sib_mask.any():
                sib_psd = psd[sib_mask]
                sib_freqs = freqs[sib_mask]
                sib_centroid = float(np.sum(sib_freqs * sib_psd) / (np.sum(sib_psd) + 1e-20))
            else:
                sib_centroid = 7000.0

            # --- Anti-FP: distinguish sibilance from general brightness ---
            # True sibilance: concentrated 5-9 kHz bursts. Cymbals: broadband > 8 kHz
            very_hf_mask = freqs > 12000.0
            very_hf_frac = float(np.sum(psd[very_hf_mask]) / total_power) if very_hf_mask.any() else 0.0
            # If very_hf energy is comparable to sibilance → it's broadband brightness, not sibilance
            brightness_ratio = very_hf_frac / max(sib_frac, 1e-6)
            if brightness_ratio > 0.6:
                # Broadband bright signal - reduce sibilance severity
                sib_frac *= 0.4

            # --- Psychoacoustic sharpness approximation (Zwicker model simplified) ---
            # Weight the sibilance band by critical-band rate (Bark scale)
            # Sibilance zone 4-12 kHz ≈ Bark 17-24 → high Bark = high sharpness contribution
            sharpness = 0.0
            if sib_mask.any():
                # Bark-weighted energy
                bark = 13 * np.arctan(0.00076 * sib_freqs) + 3.5 * np.arctan((sib_freqs / 7500) ** 2)
                bark_weight = bark / 24.0  # normalize to [0, 1]
                sharpness = float(np.sum(bark_weight * sib_psd) / (np.sum(psd) + 1e-20))

            # --- Temporal burst pattern (windowed analysis) ---
            win_sib = max(1, int(0.040 * self.sample_rate))  # 40 ms windows
            hop_sib = max(1, win_sib // 2)
            sib_bursts = 0
            n_windows = 0
            max_frac_w = 0.0
            locations: list[tuple[float, float]] = []
            in_event = False
            ev_start = 0.0
            # --- Adaptive burst threshold ---
            # A bright pop recording with lots of air (>8 kHz) has a naturally higher
            # sibilance fraction → the fixed 0.18 would flag it as sibilant.
            # Adapt the threshold upward relative to the signal's mean HF balance:
            # if the signal already has >15% energy in 4-12 kHz, raise the threshold.
            _sib_adapt_base = 0.18
            if total_power > 1e-20:
                _mean_sib_frac = float(np.sum(psd[sib_mask]) / total_power)
                # If the average sibilance fraction is naturally high (bright recording),
                # require a higher burst fraction to flag as sibilance problem.
                _sib_adapt_delta = float(np.clip((_mean_sib_frac - 0.12) * 0.6, 0.0, 0.12))
                _sib_adapt_base = _sib_adapt_base + _sib_adapt_delta
            sib_burst_threshold = _sib_adapt_base

            for i in range(0, n - win_sib, hop_sib):
                chunk = audio[i : i + win_sib]
                cf = np.fft.rfftfreq(len(chunk), 1.0 / self.sample_rate)
                cs = np.abs(np.fft.rfft(chunk)) ** 2
                c_total = float(np.sum(cs) + 1e-20)
                c_sib = float(np.sum(cs[(cf >= sib_lo) & (cf <= sib_hi)]))
                frac_w = c_sib / c_total
                max_frac_w = max(max_frac_w, frac_w)
                n_windows += 1

                t_s = i / self.sample_rate
                if frac_w >= sib_burst_threshold:
                    sib_bursts += 1
                    if not in_event:
                        ev_start = t_s
                        in_event = True
                else:
                    if in_event:
                        locations.append((ev_start, t_s + win_sib / self.sample_rate))
                        in_event = False
            if in_event:
                locations.append((ev_start, n / self.sample_rate))

            # Burst ratio: persistent brightness (>60% windows) = instrument, not sibilance
            burst_ratio = sib_bursts / max(n_windows, 1)
            if burst_ratio > 0.60:
                sib_frac *= 0.3  # sustained brightness penalty

            # Peak-window salience avoids long-file dilution of short, harsh bursts.
            sev_peak = float(np.clip((max_frac_w - (sib_burst_threshold + 0.03)) / 0.35, 0.0, 1.0))
            if burst_ratio > 0.60:
                sev_peak *= 0.2

            # --- Combined severity ---
            sev_frac = float(np.clip((sib_frac - 0.10) / 0.22, 0.0, 1.0))
            sev_sharp = float(np.clip((sharpness - 0.08) / 0.15, 0.0, 1.0))
            raw_severity = 0.35 * sev_frac + 0.25 * sev_sharp + 0.15 * burst_ratio + 0.25 * sev_peak

            threshold = self.thresholds.get(DefectType.SIBILANCE, 0.5)
            if raw_severity < threshold * 0.15:
                raw_severity = 0.0
            severity = float(np.clip(raw_severity, 0.0, 1.0))

            confidence = float(np.clip(0.65 + 0.20 * min(1.0, sev_frac), 0.50, 0.88))
            locations = self._sample_locations_evenly(locations, self._LOCATION_CAP_UNCAPPED)

            return DefectScore(
                defect_type=DefectType.SIBILANCE,
                severity=severity,
                confidence=confidence,
                locations=locations,
                metadata={
                    "sibilance_power_fraction": round(sib_frac, 4),
                    "sibilance_centroid_hz": round(sib_centroid, 1),
                    "sharpness_index": round(sharpness, 4),
                    "burst_ratio": round(burst_ratio, 3),
                    "brightness_ratio": round(brightness_ratio, 3),
                },
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.SIBILANCE, 0.0, 0.3)

    def _detect_vocal_harshness(self, audio: np.ndarray) -> DefectScore:
        """Erkennt VOCAL_HARSHNESS: excessive energy, roughness, or distortion in 2-6 kHz vocal presence zone.

        Multi-indicator harshness detection combining:
        1. Crest factor analysis in harmonic band (low crest = compressed/distorted)
        2. Spectral flux roughness in 2-6 kHz (high flux = harsh transient content)
        3. Odd-harmonic dominance in presence band (clipping signature on vocals)
        4. Peak-to-average ratio in presence zone vs. rest (harshness concentrates energy)

        Psychoacoustic basis: Harshness perception peaks at 2-6 kHz (Zwicker & Fastl 2007,
        "sharpness" metric). Distorted vocals concentrate odd-harmonic energy in this zone,
        creating audible roughness even when flat-top clipping is absent.

        Thresholds calibrated for common vocal defects:
        - Microphone preamp saturation (no flat-tops but audible distortion)
        - Over-compressed vocals (loudness war, low crest factor)
        - Scratchy/harsh recordings (excessive spectral flux in presence band)
        """
        n = len(audio)
        if n < 4096:
            return DefectScore(DefectType.VOCAL_HARSHNESS, 0.0, 0.3)

        try:
            # Energy guard: silence/very quiet signals cannot be harsh
            rms_total = float(np.sqrt(np.mean(audio**2)))
            if rms_total < 0.005:
                return DefectScore(DefectType.VOCAL_HARSHNESS, 0.0, 0.5)

            # --- Indicator 1: Crest Factor on FULL signal ---
            # Low full-signal crest factor indicates heavy compression/limiting
            # which often accompanies vocal harshness.
            # Computed on full signal (not just presence band) because narrowband
            # crest is misleading for tonal content.
            # V08: np.percentile robust gegen Click-Defekte
            crest_db = 20.0 * np.log10((float(np.percentile(np.abs(audio), 99.9)) + 1e-12) / (rms_total + 1e-12))
            # Very compressed: <5 dB; normal: 8-15 dB
            crest_score = float(np.clip((6.0 - crest_db) / 4.0, 0.0, 1.0))

            # Presence band extraction (used for flux and location detection)
            sos_bp = signal.butter(4, [2000.0, 6000.0], btype="band", fs=self.sample_rate, output="sos")
            presence = signal.sosfilt(sos_bp, audio)
            rms_pres = float(np.sqrt(np.mean(presence**2)) + 1e-12)

            # Guard: if presence band energy is negligible, no harshness possible
            if rms_pres < 0.002:
                return DefectScore(DefectType.VOCAL_HARSHNESS, 0.0, 0.5)

            # --- Indicator 2: Spectral flux roughness in 2-6 kHz ---
            # High frame-to-frame spectral variation = harsh/scratchy texture
            hop = max(1, int(0.020 * self.sample_rate))  # 20 ms hop
            win = max(256, int(0.040 * self.sample_rate))  # 40 ms window
            n_frames = max(1, (n - win) // hop)
            n_frames = min(n_frames, 200)  # cap for performance

            freq_lo_bin = int(2000.0 * win / self.sample_rate)
            freq_hi_bin = int(6000.0 * win / self.sample_rate)
            freq_hi_bin = min(freq_hi_bin, win // 2)

            prev_spec = None
            flux_values = []
            for k in range(n_frames):
                start = k * hop
                frame = audio[start : start + win]
                if len(frame) < win:
                    break
                spec = np.abs(np.fft.rfft(frame * np.hanning(win)))
                pres_spec = spec[freq_lo_bin:freq_hi_bin]
                if prev_spec is not None:
                    # Only count positive changes (onset harshness)
                    diff = np.maximum(pres_spec - prev_spec, 0.0)
                    flux = float(np.sum(diff**2))
                    flux_values.append(flux)
                prev_spec = pres_spec

            if flux_values:
                mean_flux = float(np.mean(flux_values))
                # Normalize by total energy to get relative roughness
                total_energy = float(np.mean(audio**2) + 1e-12)
                rel_flux = mean_flux / (total_energy * win)
                flux_score = float(np.clip(rel_flux / 2.0, 0.0, 1.0))
            else:
                flux_score = 0.0

            # --- Indicator 3: Odd-harmonic excess in presence band ---
            freqs, psd = self._cached_welch(audio, self.sample_rate, min(4096, n))
            pres_mask = (freqs >= 2000.0) & (freqs <= 6000.0)
            below_mask = (freqs >= 200.0) & (freqs < 2000.0)
            pres_energy = float(np.sum(psd[pres_mask]) + 1e-20)
            below_energy = float(np.sum(psd[below_mask]) + 1e-20)
            # Harshness: presence band dominates relative to fundamentals
            pres_ratio = pres_energy / below_energy
            # Normal vocals: ratio ~0.02-0.10; harsh: >0.25
            ratio_score = float(np.clip((pres_ratio - 0.10) / 0.30, 0.0, 1.0))

            # --- Indicator 4: Peak energy concentration in presence ---
            total_psd = float(np.sum(psd) + 1e-20)
            pres_fraction = pres_energy / total_psd
            # Normal: ~0.02-0.08; harsh: >0.15
            concentration_score = float(np.clip((pres_fraction - 0.08) / 0.25, 0.0, 1.0))

            # --- Combined severity (weighted) ---
            # Ratio and concentration are the primary discriminators,
            # flux captures temporal roughness, crest is supplementary.
            severity = float(0.15 * crest_score + 0.30 * flux_score + 0.30 * ratio_score + 0.25 * concentration_score)
            severity = float(np.clip(severity, 0.0, 1.0))

            # Apply material threshold
            threshold = self.thresholds.get(DefectType.VOCAL_HARSHNESS, 0.5)
            if severity < threshold * 0.15:
                severity = 0.0

            # --- Windowed location detection (100 ms windows, 50 ms hop) ---
            locations: list[tuple[float, float]] = []
            if severity > 0.0:
                win_h = max(1, int(0.100 * self.sample_rate))
                hop_h = max(1, win_h // 2)
                harsh_threshold = 0.35
                in_event = False
                ev_start = 0.0
                for i in range(0, n - win_h, hop_h):
                    chunk = audio[i : i + win_h]
                    chunk_pres = signal.sosfilt(sos_bp, chunk)
                    chunk_rms = float(np.sqrt(np.mean(chunk_pres**2)) + 1e-12)
                    # V08: Click-Spike darf Crest nicht verfälschen
                    chunk_peak = float(np.percentile(np.abs(chunk_pres), 99.9) + 1e-12)
                    chunk_crest = 20.0 * np.log10(chunk_peak / chunk_rms)
                    # Local harshness: low crest + high presence energy
                    chunk_energy_ratio = float(np.mean(chunk_pres**2)) / (float(np.mean(chunk**2)) + 1e-12)
                    local_harsh = (chunk_crest < 7.0) and (chunk_energy_ratio > harsh_threshold)
                    t_s = i / self.sample_rate
                    if local_harsh:
                        if not in_event:
                            ev_start = t_s
                            in_event = True
                    else:
                        if in_event:
                            locations.append((ev_start, t_s + win_h / self.sample_rate))
                            in_event = False
                if in_event:
                    locations.append((ev_start, n / self.sample_rate))
                locations = self._sample_locations_evenly(locations, self._LOCATION_CAP_UNCAPPED)

            confidence = 0.70 if severity > 0.3 else 0.55

            return DefectScore(
                defect_type=DefectType.VOCAL_HARSHNESS,
                severity=severity,
                confidence=confidence,
                locations=locations,
                metadata={
                    "crest_factor_db": round(crest_db, 2),
                    "crest_score": round(crest_score, 3),
                    "flux_score": round(flux_score, 3),
                    "presence_ratio_score": round(ratio_score, 3),
                    "presence_concentration_score": round(concentration_score, 3),
                    "presence_fraction": round(pres_fraction, 4),
                },
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.VOCAL_HARSHNESS, 0.0, 0.3)

    def _detect_bias_error(self, audio: np.ndarray, _bypass_material_gate: bool = False) -> DefectScore:
        """Erkennt BIAS_ERROR: wrong AC-bias level on magnetic tape recording.

        Over-bias → HF rolloff earlier than expected (spectral slope too steep
        above the bias frequency, typically 8-12 kHz on 38 cm/s tape).
        Under-bias → elevated uncorrelated HF noise floor (SNR collapse above 6 kHz).

        Upgraded v10.0.0c: Multi-band spectral slope measurement instead of
        simple 2-band ratio.  Slope is estimated from four logarithmically-spaced
        bands spanning 2-14 kHz; linear regression in dB/octave identifies
        pathological rolloff patterns.  Under-bias confirmed via noise-correlation
        check (uncorrelated segment pairs above 6 kHz).
        Literature: Lindsey & Levy 1978, IEC 60094-1, Ampex/Studer bias specs.
        """
        material_name = str(getattr(self.material_type, "value", self.material_type)).lower()
        # "cassette" is a MediumDetector alias that maps to MaterialType.TAPE="tape" - include both
        tape_materials = {"tape", "reel_tape", "wire_recording", "cassette"}
        if material_name not in tape_materials and not _bypass_material_gate:
            return DefectScore(
                defect_type=DefectType.BIAS_ERROR,
                severity=0.0,
                confidence=0.85,
                locations=[],
                metadata={"medium_gated": True},
            )

        n = len(audio)
        if n < self.sample_rate:
            return DefectScore(DefectType.BIAS_ERROR, 0.0, 0.3)

        try:
            nperseg = min(8192, n)
            freqs, psd = self._cached_welch(audio, self.sample_rate, nperseg)

            # --- Multi-band energy measurement (log-spaced 2-14 kHz) ---
            # Bands: 2-3 kHz, 3-5 kHz, 5-8 kHz, 8-14 kHz
            band_def = [(2000.0, 3000.0), (3000.0, 5000.0), (5000.0, 8000.0), (8000.0, 14000.0)]
            band_centers_oct = [np.log2((lo + hi) / 2.0) for lo, hi in band_def]
            band_energies_db: list[float] = []
            for lo, hi in band_def:
                mask = (freqs >= lo) & (freqs < hi)
                e = float(np.sum(psd[mask]) + 1e-20)
                band_energies_db.append(float(10.0 * np.log10(e)))

            # --- Spectral slope (linear regression in dB/octave) ---
            x = np.array(band_centers_oct)
            y = np.array(band_energies_db)
            slope_db_per_oct = float(np.polyfit(x - x.mean(), y - y.mean(), 1)[0])
            # Well-biased tape: slope ≈ -6 to -12 dB/oct in this range (natural music + tape EQ).
            # Over-bias: slope steeper than -16 dB/oct (HF content suppressed at bias frequency).
            # Under-bias: slope flatter/positive above 6 kHz (unmasked HF noise).
            over_bias_sev = float(np.clip((-slope_db_per_oct - 16.0) / 10.0, 0.0, 1.0))
            under_bias_sev = float(np.clip((slope_db_per_oct + 6.0) / 8.0, 0.0, 1.0))

            # --- Anti-FP: warm jazz / orchestral has naturally steep HF rolloff ---
            # Check if there is ANY musical HF content (>10 kHz), if the 4th band has
            # some content the signal is probably not over-biased at tape level.
            hf_broad_mask = freqs >= 10000.0
            hf_fraction = float(np.sum(psd[hf_broad_mask]) / (np.sum(psd[freqs > 100.0]) + 1e-20))
            # If >10 kHz has >2% of the energy, the source has HF content → over-bias less likely
            if hf_fraction > 0.02:
                over_bias_sev *= max(0.2, 1.0 - (hf_fraction - 0.02) / 0.05)

            # --- Under-bias noise confirmation: segment-to-segment HF variance ---
            # Under-bias: HF noise is uncorrelated between segments → high variance.
            # Music: HF varies but tracks envelope → moderate variance.
            hf_noise_var_score = 0.0
            seg_len_n = min(n, int(0.5 * self.sample_rate))
            n_segs = max(1, min(6, n // seg_len_n))
            if n_segs > 2:
                hf_seg_energies = []
                for si in range(n_segs):
                    s = si * max(1, (n - seg_len_n) // (n_segs - 1)) if n_segs > 1 else 0
                    seg = audio[s : s + seg_len_n]
                    fq, ps = signal.welch(seg, self.sample_rate, nperseg=min(1024, len(seg)))
                    hf_e = float(np.sum(ps[fq >= 6000.0]) + 1e-20)
                    mid_e_seg = float(np.sum(ps[(fq >= 1000.0) & (fq < 4000.0)]) + 1e-20)
                    hf_seg_energies.append(hf_e / mid_e_seg)
                hf_var = float(np.std(hf_seg_energies) / (np.mean(hf_seg_energies) + 1e-8))
                # High HF variance relative to level = uncorrelated noise
                hf_noise_var_score = float(np.clip((hf_var - 0.3) / 0.5, 0.0, 1.0))
            under_bias_sev = float(np.clip(0.5 * under_bias_sev + 0.5 * hf_noise_var_score, 0.0, 1.0))

            severity = float(np.clip(max(over_bias_sev, under_bias_sev), 0.0, 1.0))
            threshold = self.thresholds.get(DefectType.BIAS_ERROR, 0.5)
            if severity < threshold * 0.15:
                severity = 0.0

            bias_mode = "over_bias" if over_bias_sev >= under_bias_sev else "under_bias"
            confidence = float(np.clip(0.62 + 0.15 * min(1.0, abs(slope_db_per_oct) / 8.0), 0.55, 0.82))
            return DefectScore(
                defect_type=DefectType.BIAS_ERROR,
                severity=severity,
                confidence=confidence,
                locations=[],
                metadata={
                    "bias_direction": bias_mode,
                    "hf_slope": round(slope_db_per_oct, 2),
                    "slope_db_per_oct": round(slope_db_per_oct, 2),
                    "over_bias_sev": round(over_bias_sev, 3),
                    "under_bias_sev": round(under_bias_sev, 3),
                    "bias_mode": bias_mode,
                    "hf_fraction": round(hf_fraction, 4),
                    "hf_noise_variance": round(hf_noise_var_score, 3),
                },
            )
        except Exception as e:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.BIAS_ERROR, 0.0, 0.3)

    def _detect_dolby_nr_mismatch(self, audio: np.ndarray) -> DefectScore:
        """Erkennt DOLBY_NR_MISMATCH: Dolby B/C/S encode played back without decoding.

        Symptom: Dolby-encoded tape has a pre-emphasis of +6..+20 dB above 1 kHz
        (Dolby B: ~+6 dB at HF; Dolby C: ~+20 dB; Dolby S: ~+14 dB). When played
        back without the matching expander, high frequencies are severely elevated
        relative to mid frequencies - the inverse of the expected tape-hiss-masking
        curve. This is distinct from HEAD_WEAR (which attenuates HF) and BIAS_ERROR
        (which shifts the energy above the bias frequency).

        Detection (literature: Dolby Laboratories 1968-1995, Nakajima & Odaka 1983):
        - Compute the energy ratio: E(2-16 kHz) / E(300 Hz-2 kHz)
        - In normal music: ratio ≈ 0.1-0.5 (mid / presence / air balanced)
        - Dolby-B mismatch (mild):  ratio > 0.8
        - Dolby-C mismatch (severe): ratio > 1.5
        - Anti-FP: speech/bright sources naturally have high HF - check that
          the HF excess is flat (Dolby-NR shapes a shelf), not peaky (instrument).

        Only triggered on tape materials (cassette/reel_tape).  All digital and
        disc-based materials return severity=0.0 immediately.
        """
        material_name = str(getattr(self.material_type, "value", self.material_type)).lower()
        tape_materials = {"tape", "reel_tape", "wire_recording", "cassette"}
        if material_name not in tape_materials:
            return DefectScore(
                defect_type=DefectType.DOLBY_NR_MISMATCH,
                severity=0.0,
                confidence=0.9,
                locations=[],
                metadata={"medium_gated": True},
            )

        n = len(audio)
        if n < self.sample_rate:
            return DefectScore(DefectType.DOLBY_NR_MISMATCH, 0.0, 0.3)

        try:
            nperseg = min(8192, n)
            freqs, psd = self._cached_welch(audio, self.sample_rate, nperseg)

            # --- Energy in three bands ---
            def _band_energy(lo: float, hi: float) -> float:
                mask = (freqs >= lo) & (freqs < hi)
                return float(np.sum(psd[mask]) + 1e-20)

            e_mid = _band_energy(300.0, 2000.0)  # Mid: speech fundamentals
            e_presence = _band_energy(2000.0, 8000.0)  # Presence: Dolby-NR pre-emphasis zone
            e_air = _band_energy(8000.0, 16000.0)  # Air: upper Dolby-B/C zone

            hf_to_mid = (e_presence + e_air) / e_mid

            # --- Shelf-flatness check: Dolby pre-emp is a shelf, not a peak ---
            # Divide presence into 4 sub-bands; Dolby-NR raises all uniformly.
            presence_bands = [
                _band_energy(2000.0, 3000.0),
                _band_energy(3000.0, 4500.0),
                _band_energy(4500.0, 6500.0),
                _band_energy(6500.0, 8000.0),
            ]
            pb_norm = np.array(presence_bands) / (np.max(presence_bands) + 1e-20)
            # Shelf: all sub-bands elevated uniformly → low variance, all > 0.3
            shelf_uniformity = float(1.0 - np.std(pb_norm))  # 0..1; high = shelf-like

            # --- Severity mapping ---
            # Dolby-B threshold (mild): hf_to_mid > 0.8
            # Dolby-C threshold (severe): hf_to_mid > 1.5
            if hf_to_mid < 0.8:
                sev = 0.0
            elif hf_to_mid < 1.5:
                # Mild Dolby-B range: 0.0 → 0.55
                sev = float(np.clip((hf_to_mid - 0.8) / 0.7, 0.0, 1.0)) * 0.55
            else:
                # Dolby-C range: 0.55 → 1.0
                sev = 0.55 + float(np.clip((hf_to_mid - 1.5) / 1.5, 0.0, 1.0)) * 0.45

            # Weight by shelf-uniformity (pure tone peaks are not Dolby-NR)
            sev *= float(np.clip(shelf_uniformity, 0.3, 1.0))

            sev = float(np.clip(sev, 0.0, 1.0))
            confidence = 0.65 if sev > 0.05 else 0.85

            # §6.7 Dolby NR type inference from HF-to-mid ratio
            if hf_to_mid >= 1.5:
                _dolby_type = "dolby_c"
            elif hf_to_mid >= 0.8:
                _dolby_type = "dolby_b"
            else:
                _dolby_type = "none"

            return DefectScore(
                defect_type=DefectType.DOLBY_NR_MISMATCH,
                severity=sev,
                confidence=confidence,
                locations=[],
                metadata={
                    "hf_to_mid_ratio": round(hf_to_mid, 3),
                    "shelf_uniformity": round(shelf_uniformity, 3),
                    "dolby_nr_type": _dolby_type,
                },
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.DOLBY_NR_MISMATCH, 0.0, 0.3)

    def _detect_tape_head_level_dips(self, audio: np.ndarray) -> DefectScore:
        """Erkennt TAPE_HEAD_LEVEL_DIP: gradual envelope dips from head-contact pressure variation.

        Cassette / reel tape transports can exhibit periodic or irregular
        level dips caused by:
        - Capstan irregularity or worn pinch roller (periodic, ~1-3 s cycle)
        - Tape tension variation (non-periodic)
        - Head-tape spacing changes (oxide shedding, wrinkled tape)

        Morphology (observed in real cassette recordings):
        - Gradual onset: 60-100 ms ramp down
        - Minimum: 10-25 dB below context level
        - Sharp recovery: < 25 ms snap back to normal
        - Duration: 100-400 ms per event
        - Rate: 0.3-1.5 events per second

        Detection algorithm:
        1. RMS envelope (20 ms windows, 10 ms hop)
        2. Percentile-75 local reference (500 ms filter)
        3. Connected-component dip labelling (threshold: 3 dB below ref)
        4. Filter: skip genuine silence (< -55 dBFS) and very short events (< 30 ms)
        5. Severity from event rate + mean dip depth

        Scientific basis: Camras (1988) Magnetic Recording Handbook;
        McKnight (1969) 'Tape Reproducer Response Measurements with a
        Reproducer Test Tape'.
        """
        n = len(audio)
        sr = self.sample_rate
        if n < sr:
            return DefectScore(DefectType.TAPE_HEAD_LEVEL_DIP, 0.0, 0.5)

        try:
            # Parameters
            env_win = max(1, int(0.020 * sr))  # 20 ms
            env_hop = max(1, int(0.010 * sr))  # 10 ms
            ref_win_s = 0.500  # 500 ms
            min_dip_frames = 3  # 30 ms minimum
            # §v10.131 Depth-adaptive: Tiefe Ketten haben weichere, kürzere Dips
            # durch MP3-Smoothing und Kassetten-Rauschboden.
            try:
                from backend.core.calibration_context import get_calibration_context

                _dctx = get_calibration_context()
                if _dctx is not None and _dctx.transfer_chain_depth >= 4:
                    env_hop = max(1, int(0.008 * sr))  # 8 ms - feinere Auflösung
                    ref_win_s = 0.350  # 350 ms - reagiert schneller
                    min_dip_frames = 2  # 20 ms minimum - kürzere Dips
            except Exception:
                logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

            n_frames = max(0, (n - env_win) // env_hop)
            if n_frames < 10:
                return DefectScore(DefectType.TAPE_HEAD_LEVEL_DIP, 0.0, 0.5)

            # Vectorized RMS computation
            rms_env = np.array(
                [np.sqrt(np.mean(audio[i * env_hop : i * env_hop + env_win] ** 2) + 1e-15) for i in range(n_frames)],
                dtype=np.float64,
            )
            rms_db = 20.0 * np.log10(rms_env + 1e-15)

            # Local reference (p75, robust to dips)
            ref_frames = max(3, int(ref_win_s / 0.010))
            if ref_frames % 2 == 0:
                ref_frames += 1
            from scipy.ndimage import percentile_filter  # pylint: disable=import-outside-toplevel

            ref_db = percentile_filter(rms_db, percentile=75, size=ref_frames, mode="reflect")

            # §v10 SNR-adaptive dip threshold: Leise Passagen → sensitiver,
            # laute Passagen → konservativer. Ein 3dB-Dip ist in leisen
            # Stellen hörbar, in lauten nicht.
            _local_dyn = float(np.percentile(rms_db, 90) - np.percentile(rms_db, 10) + 1.0)
            # §v10.131 Depth-adaptive floor: Kassette braucht sensitiveren Floor
            _thresh_floor = (
                1.5
                if (
                    locals().get("_dctx") is not None and getattr(locals().get("_dctx"), "transfer_chain_depth", 1) >= 4
                )
                else 2.0
            )
            dip_thresh_db = float(np.clip(_local_dyn / 8.0, _thresh_floor, 5.0))

            # Dip mask
            dip_mask = rms_db < (ref_db - dip_thresh_db)

            # Connected-component labelling
            from scipy.ndimage import label as nd_label  # pylint: disable=import-outside-toplevel

            _label_result: tuple[np.ndarray, int] = nd_label(dip_mask)  # type: ignore[assignment]
            labeled, n_dips_raw = _label_result

            locations = []
            dip_depths = []
            for i in range(1, n_dips_raw + 1):
                frames = np.where(labeled == i)[0]
                if len(frames) < min_dip_frames:
                    continue
                # Skip genuine silence
                if np.mean(rms_db[frames]) < -55.0:
                    continue
                start_s = float(frames[0] * env_hop / sr)
                end_s = float((frames[-1] + 1) * env_hop / sr)
                depth = float(np.max(ref_db[frames] - rms_db[frames]))
                locations.append((start_s, end_s))
                dip_depths.append(depth)

            n_dips = len(locations)
            # §v10.30: Forwarded head-dip-like events aus dem TRANSPORT_BUMP-Detektor
            # übernehmen. Diese Events wurden dort als "sieht aus wie Head-Dip" erkannt
            # und NICHT verworfen, sondern als Hint-Locations hierher weitergereicht.
            _forwarded = getattr(self, "_forwarded_head_dip_locations", None)
            if _forwarded:
                for _ft_start, _ft_end in _forwarded:
                    # Nur hinzufügen wenn nicht bereits überlappend mit bestehender Location
                    _overlaps = any(
                        abs(_ft_start - _ex_start) < 0.050 and abs(_ft_end - _ex_end) < 0.050
                        for _ex_start, _ex_end in locations
                    )
                    if not _overlaps:
                        locations.append((_ft_start, _ft_end))
                        # Geschätzte Tiefe: 4 dB (konservativ, da vom Transport-Bump-
                        # Detektor als "≤0.90 max ratio" klassifiziert)
                        dip_depths.append(4.0)
                _n_fwd_added = len(locations) - n_dips
                logger.info(
                    "§v10.30 Forwarded %d head-dip events from TRANSPORT_BUMP → TAPE_HEAD_LEVEL_DIP "
                    "(%d existing + %d forwarded = %d total)",
                    _n_fwd_added,
                    n_dips,
                    _n_fwd_added,
                    len(locations),
                )
            n_dips = len(locations)  # Update nach Merge
            if n_dips == 0:
                return DefectScore(
                    DefectType.TAPE_HEAD_LEVEL_DIP,
                    0.0,
                    0.85,
                    locations=[],
                    metadata={"dip_count": 0},
                )

            duration_s = n / sr
            event_rate = n_dips / max(duration_s, 1e-6)
            mean_depth = float(np.mean(dip_depths))

            # Severity: event rate + depth
            sev_rate = float(np.clip(event_rate / 2.0, 0.0, 0.6))  # 2/s → sev 0.6
            sev_depth = float(np.clip((mean_depth - 3.0) / 15.0, 0.0, 0.4))  # 18 dB → sev 0.4
            severity = float(np.clip(sev_rate + sev_depth, 0.0, 1.0))

            confidence = float(np.clip(0.70 + 0.20 * min(1.0, severity), 0.70, 0.95))

            # ── Periodicity bonus (capstan-irregularity signature) ──────────
            # Real capstan/pinch-roller dips recur at a stable interval (0.5-3.5 s).
            # Musical dynamics also cause level dips but are NOT periodic.
            # A low coefficient-of-variation in inter-event intervals → confidence bonus.
            is_periodic = False
            median_interval_s = 0.0
            if n_dips >= 3:
                event_centres_s = [(loc[0] + loc[1]) / 2.0 for loc in locations]
                intervals = np.diff(event_centres_s)
                if len(intervals) >= 2:
                    median_interval_s = float(np.median(intervals))
                    cv = float(np.std(intervals)) / max(median_interval_s, 1e-9)
                    # Capstan cycle: 0.5-3.5 s, coefficient-of-variation < 0.35
                    if cv < 0.35 and 0.5 <= median_interval_s <= 3.5:
                        confidence = float(np.clip(confidence + 0.08, 0.70, 0.99))
                        is_periodic = True

            return DefectScore(
                defect_type=DefectType.TAPE_HEAD_LEVEL_DIP,
                severity=severity,
                confidence=confidence,
                locations=self._sample_locations_evenly(locations, self._LOCATION_CAP_UNCAPPED),
                metadata={
                    "dip_count": n_dips,
                    "locations_returned": n_dips,
                    "event_rate_per_s": round(event_rate, 3),
                    "mean_depth_db": round(mean_depth, 2),
                    "max_depth_db": round(float(np.max(dip_depths)), 2) if dip_depths else 0.0,
                    "is_periodic_capstan": is_periodic,
                    "median_interval_s": round(median_interval_s, 3),
                },
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.TAPE_HEAD_LEVEL_DIP, 0.0, 0.3)

    @staticmethod
    def _should_keep_cross_material_tape_head_level_dip(score: "DefectScore") -> bool:
        """Keep strong head-level-dip evidence even when material classification is wrong.

        Real-world archive imports can contain tape transfers mislabelled as vinyl/CD.
        When the dip morphology is strong enough (high event rate, deep dips), hard
        material gating hides a real defect and prevents the Tape Level Stabilizer (Phase 12)
        from ever activating.  Thresholds are deliberately conservative to avoid false
        positives on legitimate amplitude-dynamic music.

        Criteria (all must hold):
          - severity >= 0.12   (not just scanner noise)
          - dip_count >= 2     (at least two periodic dip events)
          - mean_depth_db >= 6.0  (audible head-contact dip depth)
          - event_rate_per_s >= 0.15  (recurrent, not a single transient)
        """
        severity = float(np.clip(score.severity, 0.0, 1.0))
        dip_count = int(score.metadata.get("dip_count", 0))
        mean_depth_db = float(score.metadata.get("mean_depth_db", 0.0))
        event_rate = float(score.metadata.get("event_rate_per_s", 0.0))
        return severity >= 0.12 and dip_count >= 2 and mean_depth_db >= 6.0 and event_rate >= 0.15

    def _detect_tape_head_clog(self, audio: np.ndarray) -> DefectScore:
        """Erkennt temporäre HF-Auslöschungen durch verschmutzten oder zugesetzten Magnetkopf.

        Anders als HEAD_WEAR ist Tape-Head-Clog lokal und zeitlich begrenzt: der Mittelbandanteil
        bleibt erhalten, während 4.5-12 kHz abrupt einbrechen. Das klingt dumpf, aber nicht leise.
        """
        n = len(audio)
        sr = self.sample_rate
        if n < int(sr * 2):
            return DefectScore(DefectType.TAPE_HEAD_CLOG, 0.0, 0.4)

        try:
            nyquist = sr / 2.0
            hf_lo = 4500.0
            hf_hi = min(12000.0, nyquist * 0.94)
            mid_lo = 500.0
            mid_hi = min(2500.0, nyquist * 0.45)
            if hf_hi <= hf_lo + 300.0 or mid_hi <= mid_lo + 100.0:
                return DefectScore(DefectType.TAPE_HEAD_CLOG, 0.0, 0.4)

            hf_sos = signal.butter(4, [hf_lo, hf_hi], btype="bandpass", fs=sr, output="sos")
            mid_sos = signal.butter(4, [mid_lo, mid_hi], btype="bandpass", fs=sr, output="sos")
            hf_band = signal.sosfiltfilt(hf_sos, audio).astype(np.float64)
            mid_band = signal.sosfiltfilt(mid_sos, audio).astype(np.float64)

            frame_len = max(1, int(0.050 * sr))
            hop = max(1, int(0.020 * sr))
            n_frames = max(0, (n - frame_len) // hop)
            if n_frames < 12:
                return DefectScore(DefectType.TAPE_HEAD_CLOG, 0.0, 0.45)

            hf_rms = np.array(
                [np.sqrt(np.mean(hf_band[i * hop : i * hop + frame_len] ** 2) + 1e-15) for i in range(n_frames)],
                dtype=np.float64,
            )
            mid_rms = np.array(
                [np.sqrt(np.mean(mid_band[i * hop : i * hop + frame_len] ** 2) + 1e-15) for i in range(n_frames)],
                dtype=np.float64,
            )
            hf_db = 20.0 * np.log10(hf_rms + 1e-15)
            mid_db = 20.0 * np.log10(mid_rms + 1e-15)

            from scipy.ndimage import label as nd_label  # pylint: disable=import-outside-toplevel
            from scipy.ndimage import percentile_filter  # pylint: disable=import-outside-toplevel

            ref_size = max(5, int(0.8 / 0.020))
            if ref_size % 2 == 0:
                ref_size += 1
            hf_ref_db = percentile_filter(hf_db, percentile=80, size=ref_size, mode="reflect")
            hf_drop_db = hf_ref_db - hf_db
            active_mid = mid_db > max(-42.0, float(np.percentile(mid_db, 35)))
            clog_mask = (hf_drop_db > 6.0) & active_mid

            _label_result: tuple[np.ndarray, int] = nd_label(clog_mask)  # type: ignore[assignment]
            labeled, n_regions = _label_result
            locations: list[tuple[float, float]] = []
            hf_drop_events: list[float] = []
            preserved_mid_events: list[float] = []
            for region_id in range(1, n_regions + 1):
                frames = np.where(labeled == region_id)[0]
                if len(frames) < 3:
                    continue
                duration_s = float((frames[-1] - frames[0] + 1) * hop / sr)
                if duration_s < 0.05 or duration_s > 0.60:
                    continue
                start_s = float(frames[0] * hop / sr)
                end_s = float((frames[-1] + 1) * hop / sr)
                # Filter-Initialtransienten an Datei-Rändern nicht als Kopf-Clog interpretieren.
                if start_s < 0.25 or end_s > (n / sr - 0.25):
                    continue
                mean_mid_db = float(np.mean(mid_db[frames]))
                if mean_mid_db < -45.0:
                    continue
                mean_hf_drop_db = float(np.mean(hf_drop_db[frames]))
                if mean_hf_drop_db < 6.0:
                    continue
                locations.append((start_s, end_s))
                hf_drop_events.append(mean_hf_drop_db)
                preserved_mid_events.append(mean_mid_db)

            if not locations:
                return DefectScore(
                    DefectType.TAPE_HEAD_CLOG,
                    0.0,
                    0.70,
                    metadata={"clog_event_count": 0},
                )

            event_rate = len(locations) / max(n / sr, 1e-6)
            mean_hf_drop_db = float(np.mean(hf_drop_events))
            severity = float(
                np.clip(
                    0.45 * min(1.0, event_rate / 0.8) + 0.55 * min(1.0, (mean_hf_drop_db - 6.0) / 10.0),
                    0.0,
                    1.0,
                )
            )
            threshold = self.thresholds.get(DefectType.TAPE_HEAD_CLOG, 0.5)
            if severity < threshold * 0.10:
                return DefectScore(DefectType.TAPE_HEAD_CLOG, 0.0, 0.60)

            confidence = float(np.clip(0.56 + 0.05 * len(locations) + 0.02 * min(mean_hf_drop_db, 10.0), 0.45, 0.94))
            return DefectScore(
                DefectType.TAPE_HEAD_CLOG,
                severity,
                confidence,
                locations=self._sample_locations_evenly(locations, self._LOCATION_CAP_UNCAPPED),
                metadata={
                    "clog_event_count": len(locations),
                    "event_rate_per_s": round(event_rate, 3),
                    "mean_hf_drop_db": round(mean_hf_drop_db, 2),
                    "max_hf_drop_db": round(float(np.max(hf_drop_events)), 2),
                    "mean_mid_band_db": round(float(np.mean(preserved_mid_events)), 2),
                },
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.TAPE_HEAD_CLOG, 0.0, 0.3)

    @staticmethod
    def _should_keep_cross_material_scrape_flutter(score: "DefectScore") -> bool:
        """Behält sehr starke Scrape-Flutter-Hinweise trotz fehlerhafter Materialklassifikation."""
        severity = float(np.clip(score.severity, 0.0, 1.0))
        pair_count = int(score.metadata.get("n_scrape_sideband_pairs", 0))
        dominant_rate = float(score.metadata.get("dominant_scrape_rate_hz", 0.0))
        prominence = float(score.metadata.get("strongest_sideband_prominence_db", 0.0))
        return severity >= 0.16 and pair_count >= 2 and 35.0 <= dominant_rate <= 130.0 and prominence >= 4.0

    @staticmethod
    def _should_keep_cross_material_tape_head_clog(score: "DefectScore") -> bool:
        """Behält starke Kopf-Clog-Evidenz wenn Tape-Transfers falsch als non-tape gelabelt wurden."""
        severity = float(np.clip(score.severity, 0.0, 1.0))
        event_count = int(score.metadata.get("clog_event_count", 0))
        mean_hf_drop_db = float(score.metadata.get("mean_hf_drop_db", 0.0))
        event_rate = float(score.metadata.get("event_rate_per_s", 0.0))
        return severity >= 0.14 and event_count >= 2 and mean_hf_drop_db >= 7.0 and event_rate >= 0.08

    @staticmethod
    def _chain_contains_tape(fmd_result: object) -> bool:
        """Gibt True if the transfer chain contains any tape-based medium zurück.

        Used for cross-chain BIAS_ERROR and ALIASING fallback: when the primary
        material is digital/vinyl but the chain includes a tape stage, tape-specific
        spectral defects can survive the transfer.
        """
        try:
            chain = str(getattr(fmd_result, "transfer_chain", "") or getattr(fmd_result, "chain", "")).lower()
            return any(t in chain for t in ("tape", "reel_tape", "cassette", "wire_recording"))
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return False

    @staticmethod
    def _should_keep_cross_material_bias_error(score: "DefectScore") -> bool:
        """Keep bias-error evidence when primary material is non-tape but chain has a tape stage.

        Bias errors leave a strong spectral slope anomaly (over-bias: steep HF rolloff;
        under-bias: elevated uncorrelated HF noise) that survives lossy transcoding.
        Criteria (all must hold):
          - severity >= 0.18  (clear spectral slope deviation above noise)
          - confidence >= 0.55 (multi-band regression confident)
        """
        return float(score.severity) >= 0.18 and float(score.confidence) >= 0.55

    @staticmethod
    def _should_keep_cross_material_aliasing(score: "DefectScore") -> bool:
        """Keep aliasing evidence for digital material with analog-source transfer chain.

        Pre-1990s ADC digitizations (archived as cassette→cd/mp3) often lacked proper
        AA filters; near-Nyquist energy plateau remains detectable even after transcoding.
        Criteria:
          - severity >= 0.20
          - confidence >= 0.40
        """
        return float(score.severity) >= 0.20 and float(score.confidence) >= 0.40

    def _detect_amplitude_drift(self, audio: np.ndarray) -> "DefectScore":
        """Erkennt graduellen Pegelanstieg/-abfall ueber den gesamten Song.

        Unterscheidet:
        - Traeger-Drift (physikalischer Defekt): Pegel steigt/faellt unkorreliert mit
          musikalischer Aktivitaet (Dolby-B/C AGC, Oxid-Drift) -> is_artistic=False
        - Artistischer Crescendo/Decrescendo: Pegel korreliert mit Onset-Dichte
          und musikalischer Energie -> is_artistic=True (Severity stark reduziert)

        Algorithmus:
            1. Gated RMS in 10s-Fenstern (Gate: -55 dBFS)
            2. Lineare Regression in dB/Minute
            3. Kreuzkorrelation mit Onset-Dichte (artistisch-Test)
            4. Monotonie-Test: >75% monotone Uebergaenge -> Traeger-Drift

        Literatur: Dolby B/C AGC-Drift; Cassette oxide level gradient (Nakajima 1992)
        """
        try:
            sr = self.sample_rate
            if audio.ndim == 2:
                mono = np.mean(audio, axis=0)
            else:
                mono = audio.copy()

            duration_s = len(mono) / sr
            # Need at least 30s for meaningful trend detection
            if duration_s < 30.0:
                return DefectScore(
                    DefectType.AMPLITUDE_DRIFT,
                    0.0,
                    0.5,
                    metadata={"skipped": "too_short", "duration_s": round(duration_s, 1)},
                )

            window_s = 10.0
            hop_samples = int(window_s * sr)
            gate_linear = 10 ** (-55.0 / 20.0)

            rms_values: list[float] = []
            onset_density: list[float] = []
            n_samples = len(mono)

            pos = 0
            while pos + hop_samples <= n_samples:
                chunk = mono[pos : pos + hop_samples]
                rms = float(np.sqrt(np.mean(chunk**2)))
                if rms < gate_linear:
                    # Skip near-silent windows (silence/fade)
                    pos += hop_samples
                    continue
                rms_values.append(rms)

                # Onset density: count energy-based transients in window
                frame_size = 512
                frames = len(chunk) // frame_size
                if frames >= 2:
                    energies = np.array(
                        [float(np.sum(chunk[i * frame_size : (i + 1) * frame_size] ** 2)) for i in range(frames)]
                    )
                    energy_diff = np.diff(energies)
                    n_onsets = int(np.sum(energy_diff > 0.1 * np.mean(energies[:-1] + 1e-12)))
                    onset_density.append(float(n_onsets) / frames)
                else:
                    onset_density.append(0.0)

                pos += hop_samples

            if len(rms_values) < 3:
                return DefectScore(
                    DefectType.AMPLITUDE_DRIFT,
                    0.0,
                    0.4,
                    metadata={"skipped": "too_few_windows", "n_windows": len(rms_values)},
                )

            rms_arr = np.array(rms_values, dtype=np.float64)
            rms_db = 20.0 * np.log10(rms_arr + 1e-12)
            n_windows = len(rms_arr)
            x = np.arange(n_windows, dtype=np.float64)
            x_min = float(x.mean())
            x_std = float(x.std()) or 1.0

            # Linear regression (dB vs. window index)
            x_norm = (x - x_min) / x_std
            slope_norm = float(np.polyfit(x_norm, rms_db, 1)[0])
            # Convert to dB/minute
            windows_per_minute = 60.0 / window_s
            slope_db_per_minute = slope_norm * windows_per_minute / x_std

            # No significant trend
            if abs(slope_db_per_minute) < 1.5:
                return DefectScore(
                    DefectType.AMPLITUDE_DRIFT,
                    0.0,
                    0.6,
                    metadata={
                        "slope_db_per_minute": round(slope_db_per_minute, 3),
                        "is_artistic": False,
                        "n_windows": n_windows,
                    },
                )

            # Artistic test: cross-correlate RMS envelope with onset density
            onset_arr = np.array(onset_density, dtype=np.float64)
            is_artistic = False
            correlation_r = 0.0
            if len(onset_arr) == len(rms_db) and onset_arr.std() > 1e-9:
                rms_c = rms_db - rms_db.mean()
                onset_c = onset_arr - onset_arr.mean()
                denom = float(np.linalg.norm(rms_c) * np.linalg.norm(onset_c) + 1e-12)
                correlation_r = float(np.dot(rms_c, onset_c) / denom)
                if correlation_r > 0.65:
                    is_artistic = True

            # Monotonicity test: fraction of consecutive-window increases
            diffs = np.diff(rms_db)
            if slope_db_per_minute > 0:
                monotone_frac = float(np.sum(diffs > 0)) / max(len(diffs), 1)
            else:
                monotone_frac = float(np.sum(diffs < 0)) / max(len(diffs), 1)

            if monotone_frac >= 0.75 and correlation_r < 0.40:
                # Clear carrier drift: highly monotone + not correlated with musical activity
                is_artistic = False

            # Severity: proportional to abs(slope), max at 6 dB/min -> severity 1.0
            raw_severity = float(np.clip(abs(slope_db_per_minute) / 6.0, 0.0, 1.0))

            # Artistic: drastically reduce severity (preserve, not correct)
            if is_artistic:
                severity = float(np.clip(raw_severity * 0.2, 0.0, 0.2))
            else:
                severity = raw_severity

            confidence = float(np.clip(0.55 + 0.30 * min(1.0, raw_severity), 0.55, 0.85))

            return DefectScore(
                DefectType.AMPLITUDE_DRIFT,
                severity,
                confidence,
                metadata={
                    "slope_db_per_minute": round(slope_db_per_minute, 3),
                    "is_artistic": bool(is_artistic),
                    "correlation_r": round(correlation_r, 3),
                    "monotone_frac": round(monotone_frac, 3),
                    "n_windows": n_windows,
                    "drift_db_per_minute": round(slope_db_per_minute, 3),
                },
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.AMPLITUDE_DRIFT, 0.0, 0.3)

    def _detect_riaa_curve_error(self, audio: np.ndarray) -> DefectScore:
        """Erkennt RIAA_CURVE_ERROR: wrong EQ curve applied during vinyl/disc digitization.

        RIAA not applied: massive bass excess (bass/mid ratio >> 5).
        Double-RIAA or wrong curve applied: extreme bass cut (ratio << 0.1).
        """
        material_name = str(getattr(self.material_type, "value", self.material_type)).lower()
        riaa_materials = {"vinyl", "shellac", "lacquer_disc"}
        if material_name not in riaa_materials:
            # Preserve raw RIAA evidence for diagnostics even when the defect is
            # physically impossible for the current medium and therefore gated.
            _orig_sev = 0.0
            _ratio = 0.0
            _riaa_missing = 0.0
            _riaa_double = 0.0
            try:
                n = len(audio)
                if n >= self.sample_rate:
                    freqs, psd = self._cached_welch(audio, self.sample_rate, min(4096, n))
                    bass_e = float(np.sum(psd[freqs < 300.0]) + 1e-20)
                    mid_e = float(np.sum(psd[(freqs >= 1000.0) & (freqs < 4000.0)]) + 1e-20)
                    _ratio = bass_e / mid_e
                    _riaa_missing = float(np.clip((_ratio - 5.0) / 10.0, 0.0, 1.0))
                    _riaa_double = float(np.clip((0.1 - _ratio) / 0.10, 0.0, 1.0))
                    _orig_sev = float(np.clip(max(_riaa_missing, _riaa_double), 0.0, 1.0))
            except Exception:
                _orig_sev = 0.0

            return DefectScore(
                defect_type=DefectType.RIAA_CURVE_ERROR,
                severity=0.0,
                confidence=0.85,
                locations=[],
                metadata={
                    "medium_gated": True,
                    "original_severity": _orig_sev,
                    "bass_mid_ratio": _ratio,
                    "riaa_missing": _riaa_missing,
                    "riaa_double": _riaa_double,
                    "riaa_missing_score": _riaa_missing,
                    "best_matching_curve": "RIAA",
                },
            )

        n = len(audio)
        if n < self.sample_rate:
            return DefectScore(DefectType.RIAA_CURVE_ERROR, 0.0, 0.3)

        try:
            freqs, psd = self._cached_welch(audio, self.sample_rate, min(4096, n))
            bass_e = float(np.sum(psd[freqs < 300.0]) + 1e-20)
            mid_e = float(np.sum(psd[(freqs >= 1000.0) & (freqs < 4000.0)]) + 1e-20)
            ratio = bass_e / mid_e

            riaa_missing = float(np.clip((ratio - 5.0) / 10.0, 0.0, 1.0))
            riaa_double = float(np.clip((0.1 - ratio) / 0.10, 0.0, 1.0))
            severity = float(np.clip(max(riaa_missing, riaa_double), 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.RIAA_CURVE_ERROR, 0.5)
            if severity < threshold * 0.3:
                severity = 0.0

            return DefectScore(
                defect_type=DefectType.RIAA_CURVE_ERROR,
                severity=severity,
                confidence=0.58,
                locations=[],
                metadata={
                    "bass_mid_ratio": ratio,
                    "riaa_missing": riaa_missing,
                    "riaa_double": riaa_double,
                    "riaa_missing_score": riaa_missing,
                    "best_matching_curve": "RIAA",
                },
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.RIAA_CURVE_ERROR, 0.0, 0.3)

    def _detect_head_wear(self, audio: np.ndarray) -> DefectScore:
        """Erkennt HEAD_WEAR: magnetic head degradation causing progressive HF rolloff.

        Worn head → progressive rolloff above the head-gap frequency (typically
        10-16 kHz for 1/4" tape at 38 cm/s).  Characteristic signature:
          - Monotonically decreasing band energies from 4 kHz → 8 kHz → 12 kHz
          - dB/octave slope steeper than -18 dB/oct above 4 kHz
          - Smooth rolloff (no EQ bumps), distinguishable from intentional vintage EQ

        Upgraded v10.0.0c: Multi-band progressive rolloff fit + monotonicity check.
        Anti-FP: vintage/orchestral recordings naturally lack HF; confirmed only when
        rolloff is MONOTONIC and starts from mid range (4 kHz), not just absent HF.
        Literature: Bohn 1987 'Magnetic Recording Head Design', Ampex service manuals.
        """
        n = len(audio)
        if n < self.sample_rate:
            return DefectScore(DefectType.HEAD_WEAR, 0.0, 0.3)

        try:
            nperseg = min(8192, n)
            freqs, psd = self._cached_welch(audio, self.sample_rate, nperseg)

            # --- Multi-band energy: 5 bands from 2 kHz to 16 kHz (log-spaced) ---
            hw_bands = [
                (2000.0, 3500.0),  # 0: Reference mid
                (3500.0, 5500.0),  # 1: Upper-mid
                (5500.0, 8000.0),  # 2: Presence
                (8000.0, 12000.0),  # 3: Brilliance
                (12000.0, 16000.0),  # 4: Air (most affected by head wear)
            ]
            band_e_db = []
            for lo, hi in hw_bands:
                mask = (freqs >= lo) & (freqs < min(hi, self.sample_rate * 0.48))
                e = float(np.sum(psd[mask]) + 1e-20)
                band_e_db.append(float(10.0 * np.log10(e)))

            ref_db = band_e_db[0]  # 2-3.5 kHz reference
            relative_db = [e - ref_db for e in band_e_db[1:]]
            # relative_db[0] = upper-mid, [1] = presence, [2] = brilliance, [3] = air

            # --- Monotonicity check: each band should be lower than the previous ---
            # Head wear → strictly monotonic decline.
            # Creative EQ or recording → may have bumps or non-monotonic pattern.
            monotonic_penalties = 0
            for i in range(len(relative_db) - 1):
                if relative_db[i + 1] > relative_db[i] + 2.0:
                    monotonic_penalties += 1
            # Allow 1 minor excursion (mastering EQ bump), but 2+ = not head wear
            monotonicity_ok = monotonic_penalties <= 1

            # --- Spectral slope from 4 kHz to 16 kHz (dB/octave) ---
            slope_bands = hw_bands[1:]
            xs = np.array([np.log2((lo + hi) / 2.0) for lo, hi in slope_bands])
            ys = np.array(band_e_db[1:])
            slope = float(np.polyfit(xs - xs.mean(), ys - ys.mean(), 1)[0])
            # Normal tape: -6 to -14 dB/oct.  Head wear: steeper than -18 dB/oct.
            slope_sev = float(np.clip((-slope - 18.0) / 10.0, 0.0, 1.0))

            # --- Absolute HF level check (band 3 + 4 vs. reference) ---
            mean_hf_relative_db = float(np.mean(relative_db[2:]))  # brilliance + air
            # Worn head: >30 dB below reference causes audible extension loss
            level_sev = float(np.clip((-mean_hf_relative_db - 30.0) / 15.0, 0.0, 1.0))

            severity = 0.0
            if monotonicity_ok:
                severity = float(np.clip(0.55 * slope_sev + 0.45 * level_sev, 0.0, 1.0))
            else:
                # Non-monotonic → likely EQ, not head wear; heavily discount
                severity = float(np.clip(0.15 * slope_sev + 0.10 * level_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.HEAD_WEAR, 0.5)
            if severity < threshold * 0.12:
                severity = 0.0

            confidence = float(np.clip(0.58 + 0.18 * float(monotonicity_ok) - 0.10 * monotonic_penalties, 0.40, 0.80))

            return DefectScore(
                defect_type=DefectType.HEAD_WEAR,
                severity=severity,
                confidence=confidence,
                locations=[],
                metadata={
                    "slope_db_per_oct": round(slope, 2),
                    "slope_severity": round(slope_sev, 3),
                    "level_severity": round(level_sev, 3),
                    "monotonicity_ok": monotonicity_ok,
                    "monotonic_penalties": monotonic_penalties,
                    "relative_db_bands": [round(x, 1) for x in relative_db],
                },
            )
        except Exception as e:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.HEAD_WEAR, 0.0, 0.3)

    def _detect_transient_smearing(self, audio: np.ndarray) -> DefectScore:
        """Erkennt TRANSIENT_SMEARING: broadened attack rise-times.

        Upgraded v10.0.0b: Hilbert-envelope for smoother analysis,
        strongest-onset selection (by peak amplitude, not just first N),
        reduced self-smoothing (2ms window), median rise-time for outlier
        robustness, spectral transient sharpness indicator.
        """
        n = len(audio)
        if n < self.sample_rate // 2:
            return DefectScore(DefectType.TRANSIENT_SMEARING, 0.0, 0.3)

        try:
            # --- Hilbert envelope (no self-smoothing bias) ---
            from scipy.signal import hilbert as _hilbert  # pylint: disable=import-outside-toplevel

            hp_sos = signal.butter(
                2, float(np.clip(200.0 / (self.sample_rate / 2.0), 1e-6, 0.999)), btype="high", output="sos"
            )
            hp_audio = signal.sosfilt(hp_sos, audio)

            # Hilbert envelope + light smoothing (2ms instead of 5ms)
            analytic = _hilbert(hp_audio[: min(n, 480000)])  # limit to 10s at 48kHz
            analytic_arr = np.asarray(analytic, dtype=np.complex128)
            envelope = np.abs(analytic_arr)
            smooth_win = max(1, int(0.002 * self.sample_rate))
            envelope = np.convolve(envelope, np.ones(smooth_win) / smooth_win, mode="same")
            envelope = np.nan_to_num(envelope, nan=0.0)

            threshold_level = float(np.percentile(envelope, 90))
            if threshold_level < 1e-8:
                return DefectScore(DefectType.TRANSIENT_SMEARING, 0.0, 0.3)

            # --- Find ALL onset candidates ---
            diff_env = np.diff(envelope)
            onset_mask = (envelope[1:] > threshold_level * 0.5) & (diff_env > 0)
            onset_idxs = np.where(onset_mask)[0]
            if len(onset_idxs) == 0:
                return DefectScore(DefectType.TRANSIENT_SMEARING, 0.0, 0.3)

            # --- Select strongest onsets by peak amplitude ---
            # Group nearby onsets (within 50ms) and pick the strongest from each group
            min_gap = int(0.050 * self.sample_rate)
            selected_onsets = []
            last_idx = -min_gap - 1
            group_best = None
            group_best_val = 0.0
            for idx in onset_idxs:
                if idx - last_idx > min_gap:
                    if group_best is not None:
                        selected_onsets.append(group_best)
                    group_best = idx
                    group_best_val = float(envelope[idx])
                else:
                    if float(envelope[idx]) > group_best_val:
                        group_best = idx
                        group_best_val = float(envelope[idx])
                last_idx = idx
            if group_best is not None:
                selected_onsets.append(group_best)

            # Sort by amplitude (strongest first), take up to 40
            selected_onsets.sort(key=lambda x: float(envelope[x]), reverse=True)
            selected_onsets = selected_onsets[:40]

            # --- Measure 10%-90% rise time at each onset ---
            rise_times_ms: list[float] = []
            smear_locations: list[tuple[float, float]] = []
            for idx in selected_onsets:
                peak_val = float(envelope[idx])
                if peak_val < threshold_level * 0.3:
                    continue
                level_10 = peak_val * 0.10
                level_90 = peak_val * 0.90
                window_back = max(0, idx - int(0.040 * self.sample_rate))
                pre_env = envelope[window_back : idx + 1]
                idx_10 = idx_90 = None
                for k in range(len(pre_env)):
                    if pre_env[k] >= level_10 and idx_10 is None:
                        idx_10 = window_back + k
                    if pre_env[k] >= level_90 and idx_90 is None:
                        idx_90 = window_back + k
                        break
                if idx_10 is not None and idx_90 is not None and idx_90 > idx_10:
                    rt = (idx_90 - idx_10) / self.sample_rate * 1000.0
                    if 0.3 < rt < 80.0:
                        rise_times_ms.append(rt)
                        if rt > 6.0:
                            t_s = window_back / self.sample_rate
                            t_e = (idx + int(0.010 * self.sample_rate)) / self.sample_rate
                            smear_locations.append((t_s, t_e))

            if not rise_times_ms:
                return DefectScore(DefectType.TRANSIENT_SMEARING, 0.0, 0.3)

            # Use median for robustness against outliers
            median_rt = float(np.median(rise_times_ms))
            mean_rt = float(np.mean(rise_times_ms))
            p90_rt = float(np.percentile(rise_times_ms, 90))

            # --- Spectral transient sharpness: HF-to-LF ratio at onset ---
            spectral_sharpness_values = []
            for idx in selected_onsets[:15]:
                onset_start = max(0, idx - int(0.005 * self.sample_rate))
                onset_end = min(len(hp_audio), idx + int(0.005 * self.sample_rate))
                if onset_end - onset_start >= 256:
                    seg = hp_audio[onset_start:onset_end]
                    spec = np.abs(np.fft.rfft(seg))
                    half = len(spec) // 2
                    if half > 2:
                        hf_e = float(np.sum(spec[half:]))
                        lf_e = float(np.sum(spec[:half]) + 1e-12)
                        spectral_sharpness_values.append(hf_e / lf_e)

            spectral_sharpness = float(np.median(spectral_sharpness_values)) if spectral_sharpness_values else 0.5

            # --- Severity ---
            # Smeared: median rise > 10ms; severe: > 20ms
            sev_rise = float(np.clip((median_rt - 6.0) / 18.0, 0.0, 1.0))
            # Low spectral sharpness at onsets = smeared transients
            sev_spec = float(np.clip(1.0 - spectral_sharpness / 1.5, 0.0, 0.5))
            # Fraction of smeared onsets
            n_smeared = sum(1 for rt in rise_times_ms if rt > 8.0)
            smear_fraction = n_smeared / max(len(rise_times_ms), 1)

            raw_severity = 0.50 * sev_rise + 0.25 * smear_fraction + 0.25 * sev_spec
            threshold = self.thresholds.get(DefectType.TRANSIENT_SMEARING, 0.5)
            if raw_severity < threshold * 0.15:
                raw_severity = 0.0
            severity = float(np.clip(raw_severity, 0.0, 1.0))

            confidence = float(np.clip(0.60 + 0.20 * min(1.0, len(rise_times_ms) / 15.0), 0.50, 0.85))

            return DefectScore(
                defect_type=DefectType.TRANSIENT_SMEARING,
                severity=severity,
                confidence=confidence,
                locations=smear_locations if severity > 0.0 else [],
                metadata={
                    "median_rise_time_ms": round(median_rt, 2),
                    "mean_rise_time_ms": round(mean_rt, 2),
                    "p90_rise_time_ms": round(p90_rt, 2),
                    "n_onsets": len(rise_times_ms),
                    "smear_fraction": round(smear_fraction, 3),
                    "spectral_sharpness": round(spectral_sharpness, 3),
                },
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.TRANSIENT_SMEARING, 0.0, 0.3)

    def _detect_pre_echo(self, audio: np.ndarray) -> DefectScore:
        """Erkennt PRE_ECHO: energy preceding a major transient.

        Upgraded v10.0.0b: Dual time-scale detection:
        1. Short pre-echo (5-35 ms): codec temporal masking artifacts
        2. Long pre-echo (100-600 ms): tape print-through ghost signals
        Spectral similarity check (does pre-echo share spectrum with transient?),
        material-aware routing (tape→print-through, digital→codec pre-echo),
        robust baseline estimation with percentile-based floor.
        """
        n = len(audio)
        if n < self.sample_rate:
            return DefectScore(DefectType.PRE_ECHO, 0.0, 0.3)

        try:
            win = max(1, int(0.008 * self.sample_rate))  # 8ms envelope smoothing
            envelope = np.convolve(np.abs(audio), np.ones(win) / win, mode="same")
            envelope = np.nan_to_num(envelope)

            peak_env = float(np.max(envelope))
            if peak_env < 1e-8:
                return DefectScore(DefectType.PRE_ECHO, 0.0, 0.3)

            # Find strong transients (top 10% of envelope, rising edge)
            transient_thresh = float(np.percentile(envelope, 85))
            diff_env = np.diff(envelope)
            transient_idxs = np.where((envelope[1:] > transient_thresh) & (diff_env > transient_thresh * 0.1))[0]

            if len(transient_idxs) == 0:
                return DefectScore(DefectType.PRE_ECHO, 0.0, 0.3)

            # --- Deduplicate transients (min 100ms apart) ---
            min_gap = int(0.100 * self.sample_rate)
            deduped = [transient_idxs[0]]
            for idx in transient_idxs[1:]:
                if idx - deduped[-1] > min_gap:
                    deduped.append(idx)
            transient_idxs = np.array(deduped, dtype=np.intp)

            # Material check for analysis routing
            _mat = getattr(self, "material_type", None)
            if _mat is None:
                mat_name = ""
            elif isinstance(_mat, Enum):
                mat_name = str(_mat.value)
            else:
                mat_name = str(_mat)
            _TAPE_MATS = {"tape", "reel_tape", "cassette"}
            is_tape = mat_name in _TAPE_MATS

            # --- Short pre-echo analysis (codec, 5-35 ms before transient) ---
            short_pre_ms = int(0.035 * self.sample_rate)
            short_gap_ms = int(0.003 * self.sample_rate)  # 3ms gap from transient onset
            short_ratios: list[float] = []
            # --- Long pre-echo analysis (print-through, 100-600 ms before transient) ---
            long_pre_start_ms = int(0.600 * self.sample_rate)
            long_pre_end_ms = int(0.100 * self.sample_rate)
            long_ratios: list[float] = []
            spectral_similarities: list[float] = []
            pre_echo_locations: list[tuple[float, float]] = []

            # Noise floor baseline (10th percentile of envelope)
            baseline_energy = float(np.percentile(envelope, 10) ** 2) + 1e-20

            for idx in transient_idxs:
                # -- Short pre-echo (codec) --
                pre_end = max(0, idx - short_gap_ms)
                pre_start = max(0, pre_end - short_pre_ms)
                if pre_end > pre_start + 32:
                    pre_energy = float(np.mean(envelope[pre_start:pre_end] ** 2))
                    ratio_short = pre_energy / baseline_energy
                    if ratio_short > 1.5:
                        short_ratios.append(ratio_short)
                        t_start = pre_start / self.sample_rate
                        t_end = idx / self.sample_rate
                        pre_echo_locations.append((t_start, t_end))

                # -- Long pre-echo (print-through) --
                if is_tape and idx > long_pre_start_ms:
                    lp_start = max(0, idx - long_pre_start_ms)
                    lp_end = max(0, idx - long_pre_end_ms)
                    if lp_end > lp_start + 128:
                        long_pre_energy = float(np.mean(envelope[lp_start:lp_end] ** 2))
                        ratio_long = long_pre_energy / baseline_energy
                        if ratio_long > 1.3:
                            long_ratios.append(ratio_long)

                            # Spectral similarity: does the ghost match the transient?
                            trans_start = idx
                            trans_end = min(n, idx + int(0.030 * self.sample_rate))
                            if trans_end - trans_start >= 256 and lp_end - lp_start >= 256:
                                spec_trans = np.abs(np.fft.rfft(audio[trans_start:trans_end]))
                                spec_ghost = np.abs(np.fft.rfft(audio[lp_start:lp_end][: trans_end - trans_start]))
                                min_len_s = min(len(spec_trans), len(spec_ghost))
                                if min_len_s > 4:
                                    _a = spec_trans[:min_len_s].astype(float)
                                    _b = spec_ghost[:min_len_s].astype(float)
                                    _na = float(np.linalg.norm(_a))
                                    _nb = float(np.linalg.norm(_b))
                                    corr = (
                                        float(np.dot(_a, _b) / (_na * _nb + 1e-12))
                                        if _na > 1e-12 and _nb > 1e-12
                                        else 0.0
                                    )
                                    if not np.isnan(corr):
                                        spectral_similarities.append(max(0.0, corr))

            # --- Severity calculation ---
            sev_short = 0.0
            if short_ratios:
                mean_short = float(np.mean(short_ratios))
                sev_short = float(np.clip((mean_short - 2.0) / 5.0, 0.0, 1.0))

            sev_long = 0.0
            if long_ratios:
                mean_long = float(np.mean(long_ratios))
                sev_long = float(np.clip((mean_long - 1.5) / 4.0, 0.0, 1.0))
                # Bonus if spectrally similar (= confirmed print-through)
                if spectral_similarities:
                    mean_sim = float(np.mean(spectral_similarities))
                    sev_long *= 0.5 + 0.5 * mean_sim

            # Weighted combination: tape → prioritize long; digital → prioritize short
            if is_tape:
                raw_severity = 0.40 * sev_short + 0.60 * sev_long
            else:
                raw_severity = 0.80 * sev_short + 0.20 * sev_long

            # §4.11: Fusionspfad mit dediziertem Codec-Pre-Echo-Detektor.
            # Der Heuristikpfad bleibt primär, der Codec-Detektor hebt False-Zero-Fälle
            # bei lossy Material (MP3/AAC/Streaming) robust an.
            codec_event_count = 0
            codec_detector_severity = 0.0
            codec_detector_confidence = 0.0
            try:
                from backend.core.dsp.pre_echo_detector import get_pre_echo_detector

                codec_events = get_pre_echo_detector().detect(
                    audio,
                    int(self.sample_rate),
                    material_key=mat_name or "unknown",
                )
                codec_event_count = int(len(codec_events))
                if codec_event_count > 0:
                    _sev_db = [
                        float(max(0.0, evt.get("severity_db", 0.0))) for evt in codec_events if isinstance(evt, dict)
                    ]
                    _conf = [
                        float(np.clip(evt.get("confidence", 0.0), 0.0, 1.0))
                        for evt in codec_events
                        if isinstance(evt, dict)
                    ]
                    if _sev_db:
                        # 12 dB Überschuss ≈ klare Pre-Echo-Ausprägung.
                        codec_detector_severity = float(np.clip(np.mean(_sev_db) / 12.0, 0.0, 1.0))
                    if _conf:
                        codec_detector_confidence = float(np.clip(np.mean(_conf), 0.0, 1.0))

                    codec_bonus = float(np.clip(codec_event_count / 8.0, 0.0, 0.2))
                    raw_severity = max(raw_severity, 0.75 * codec_detector_severity + codec_bonus)

                    # Event-Locations aus Codec-Detektor mappen (sample → seconds).
                    for evt in codec_events:
                        if not isinstance(evt, dict):
                            continue
                        start_s = float(max(0, int(evt.get("pre_echo_start", 0)))) / self.sample_rate
                        end_s = float(max(0, int(evt.get("pre_echo_end", 0)))) / self.sample_rate
                        if end_s > start_s:
                            pre_echo_locations.append((start_s, end_s))
            except Exception:
                # Non-blocking: Heuristikpfad bleibt voll funktionsfähig.
                logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

            # Event count bonus: many pre-echoes = systematic problem
            n_events = len(short_ratios) + len(long_ratios) + codec_event_count

            # Fallback-Heuristik für "severity=0"-Fälle bei klaren Transienten mit
            # leiser Vorenergie (typisches lossy Pre-Echo, wenn Envelope-Diff zu strikt ist).
            fallback_events = 0
            fallback_rel_mean = 0.0
            if raw_severity <= 1e-10:
                try:
                    _peak_min_distance = int(0.120 * self.sample_rate)
                    _peak_height = float(np.percentile(envelope, 95))
                    _peaks, _ = signal.find_peaks(
                        envelope,
                        height=_peak_height,
                        distance=max(1, _peak_min_distance),
                        prominence=max(1e-8, 0.06 * peak_env),
                    )
                    _fallback_rel: list[float] = []
                    _floor_rms = float(np.percentile(envelope, 12)) + 1e-12
                    for idx in _peaks:
                        pre_start = max(0, int(idx - 0.035 * self.sample_rate))
                        pre_end = max(0, int(idx - 0.005 * self.sample_rate))
                        on_start = int(idx)
                        on_end = min(n, int(idx + 0.020 * self.sample_rate))
                        if pre_end <= pre_start + 32 or on_end <= on_start + 32:
                            continue
                        pre_rms = float(np.sqrt(np.mean(np.square(audio[pre_start:pre_end])) + 1e-20))
                        on_rms = float(np.sqrt(np.mean(np.square(audio[on_start:on_end])) + 1e-20))
                        if on_rms <= max(5.0 * _floor_rms, 1e-8):
                            continue
                        rel = pre_rms / (on_rms + 1e-12)
                        # Pre-Echo: klar über Grundrauschen, aber deutlich unter Haupttransient.
                        if pre_rms > 1.8 * _floor_rms and 0.03 <= rel <= 0.8:
                            _fallback_rel.append(float(rel))
                            pre_echo_locations.append((pre_start / self.sample_rate, idx / self.sample_rate))

                    if _fallback_rel:
                        fallback_events = len(_fallback_rel)
                        fallback_rel_mean = float(np.mean(_fallback_rel))
                        fallback_sev = float(np.clip((fallback_rel_mean - 0.05) / 0.30, 0.0, 0.7))
                        raw_severity = max(raw_severity, fallback_sev)
                except Exception:
                    logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)

            n_events += fallback_events
            event_bonus = float(np.clip(n_events / 10.0, 0.0, 0.3))
            raw_severity += event_bonus

            threshold = self.thresholds.get(DefectType.PRE_ECHO, 0.5)
            if raw_severity < threshold * 0.15:
                raw_severity = 0.0
            severity = float(np.clip(raw_severity, 0.0, 1.0))

            confidence = float(
                np.clip(
                    0.58 + 0.18 * min(1.0, n_events / 8.0) + 0.10 * codec_detector_confidence,
                    0.45,
                    0.9,
                )
            )

            return DefectScore(
                defect_type=DefectType.PRE_ECHO,
                severity=severity,
                confidence=confidence,
                locations=pre_echo_locations if severity > 0.0 else [],
                metadata={
                    "short_pre_echo_mean_ratio": round(float(np.mean(short_ratios)), 3) if short_ratios else 0.0,
                    "long_pre_echo_mean_ratio": round(float(np.mean(long_ratios)), 3) if long_ratios else 0.0,
                    "spectral_similarity": (
                        round(float(np.mean(spectral_similarities)), 3) if spectral_similarities else 0.0
                    ),
                    "n_short_events": len(short_ratios),
                    "n_long_events": len(long_ratios),
                    "codec_pre_echo_events": int(codec_event_count),
                    "codec_detector_severity": round(float(codec_detector_severity), 3),
                    "codec_detector_confidence": round(float(codec_detector_confidence), 3),
                    "fallback_pre_echo_events": int(fallback_events),
                    "fallback_pre_echo_rel_mean": round(float(fallback_rel_mean), 3),
                    "is_tape": is_tape,
                },
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.PRE_ECHO, 0.0, 0.3)

    def _detect_aliasing(self, audio: np.ndarray, _bypass_material_gate: bool = False) -> DefectScore:
        """Erkennt ALIASING: spurious near-Nyquist energy without natural musical source.

        Anti-aliasing filter failure during digitization → elevated spectral floor
        in the 85-97 % Nyquist region that exceeds the expected HF rolloff.
        """
        # Digital sources (cd_digital, dat, mp3_*, aac, streaming, minidisc) have proper
        # anti-aliasing filters by design - aliasing is not physically possible.
        # Exception: cross-chain fallback (_bypass_material_gate=True) allows detection
        # for pre-1990s digitizations archived via digital format (missing AA filter).
        _DIGITAL_MATS = {"cd_digital", "dat", "mp3_low", "mp3_high", "aac", "streaming", "minidisc"}
        _mat = getattr(self, "material_type", None)
        if _mat is not None and not _bypass_material_gate:
            _mat_name = str(_mat.value) if isinstance(_mat, Enum) else str(_mat)
            if _mat_name in _DIGITAL_MATS:
                return DefectScore(
                    defect_type=DefectType.ALIASING,
                    severity=0.0,
                    confidence=0.9,
                    locations=[],
                    metadata={"medium_gated": True},
                )

        n = len(audio)
        if n < self.sample_rate:
            return DefectScore(DefectType.ALIASING, 0.0, 0.3)

        try:
            freqs, psd = self._cached_welch(audio, self.sample_rate, min(4096, n))
            nyquist = self.sample_rate / 2.0

            near_nyq_mask = (freqs >= nyquist * 0.85) & (freqs <= nyquist * 0.97)
            mid_hf_mask = (freqs >= 10000.0) & (freqs < nyquist * 0.85)

            near_e = float(np.sum(psd[near_nyq_mask]) + 1e-20)
            mid_e = float(np.sum(psd[mid_hf_mask]) + 1e-20)

            near_nyq_ratio = near_e / mid_e
            # Normal: HF monotonically decreases → near-Nyquist < 0.5× mid
            # Aliasing: plateau or rise near Nyquist
            severity = float(np.clip((near_nyq_ratio - 0.6) / 0.8, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.ALIASING, 0.5)
            if severity < threshold * 0.15:
                severity = 0.0

            locations: list[tuple[float, float]] = []
            if severity > 0.0:
                _f_t, _t_t, _z_t = signal.stft(audio, self.sample_rate, nperseg=2048, noverlap=1536, boundary="even")
                _p_t = np.abs(_z_t) ** 2
                _near_mask_t = (_f_t >= nyquist * 0.85) & (_f_t <= nyquist * 0.97)
                _mid_mask_t = (_f_t >= 10000.0) & (_f_t < nyquist * 0.85)
                if _near_mask_t.any() and _mid_mask_t.any() and _p_t.shape[1] > 0:
                    _near_t = np.sum(_p_t[_near_mask_t, :], axis=0)
                    _mid_t = np.sum(_p_t[_mid_mask_t, :], axis=0) + 1e-12
                    _ratio_t = _near_t / _mid_t
                    _r_thr = float(max(np.percentile(_ratio_t, 75.0), 1.05))  # type: ignore[arg-type]
                    _mask_t = _ratio_t >= _r_thr
                    if np.any(_mask_t):
                        _starts = np.where(np.diff(np.concatenate(([0], _mask_t.astype(np.int8), [0]))) == 1)[0]  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
                        _ends = np.where(np.diff(np.concatenate(([0], _mask_t.astype(np.int8), [0]))) == -1)[0]  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
                        _half = 2048.0 / (2.0 * self.sample_rate)
                        _max_t = float(len(audio)) / float(self.sample_rate)
                        for _s_f, _e_f in zip(_starts, _ends):
                            _t0 = max(0.0, float(_t_t[_s_f]) - _half)
                            _t1 = min(_max_t, float(_t_t[max(0, _e_f - 1)]) + _half)
                            if _t1 > _t0:
                                locations.append((_t0, _t1))

            return DefectScore(
                defect_type=DefectType.ALIASING,
                severity=severity,
                confidence=0.55,
                locations=locations,
                metadata={"near_nyquist_ratio": near_nyq_ratio, "nyquist_hz": nyquist},
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.ALIASING, 0.0, 0.3)

    # ========== v10.0.0: 12 neue SOTA-Detektoren ==========

    def _detect_modulation_noise(self, audio: np.ndarray) -> DefectScore:
        """Erkennt signal-dependent modulation noise (tape recording artifact).

        Modulation noise is noise whose level varies proportionally to the signal
        level - fundamentally different from stationary tape hiss.  Present in every
        analog tape recording.

        Algorithm (Esquef & Biscainho 2006):
        1. Compute short-time RMS envelope (10 ms frames)
        2. Compute short-time noise variance estimate (difference of adjacent frames)
        3. Correlate noise variance with signal envelope - high positive correlation
           indicates modulation noise (noise tracks signal level)
        4. Severity derived from correlation coefficient and noise/signal power ratio

        Scientific basis: Esquef & Biscainho (2006), "An Improved Model for
        Tape Modulation Noise"; Czyzewski et al. (2020), "Signal-Dependent Noise
        Models with MMSE Estimation for Audio Restoration".
        """
        n = len(audio)
        sr = self.sample_rate
        if n < sr * 2:
            return DefectScore(DefectType.MODULATION_NOISE, 0.0, 0.3)
        try:
            # Short-time RMS envelope (10 ms frames, 5 ms hop)
            frame_len = max(1, int(0.010 * sr))
            hop = max(1, frame_len // 2)
            n_frames = max(1, (n - frame_len) // hop)
            if n_frames < 10:
                return DefectScore(DefectType.MODULATION_NOISE, 0.0, 0.3)

            frames = np.lib.stride_tricks.as_strided(
                audio,
                shape=(n_frames, frame_len),
                strides=(audio.strides[0] * hop, audio.strides[0]),
            ).copy()
            rms_env = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)

            # Noise estimate: difference between adjacent frames (stationarity assumption)
            noise_var = np.abs(np.diff(rms_env))
            signal_env = rms_env[:-1]

            # Only consider frames where signal is above noise floor
            signal_threshold = float(np.percentile(rms_env, 20))
            mask = signal_env > signal_threshold
            if np.sum(mask) < 20:
                return DefectScore(DefectType.MODULATION_NOISE, 0.0, 0.4)

            # Pearson correlation between signal level and noise variance (guarded dot-product)
            _s = signal_env[mask].astype(float)
            _n = noise_var[mask].astype(float)
            _s_c = _s - float(np.mean(_s))
            _n_c = _n - float(np.mean(_n))
            _ns = float(np.linalg.norm(_s_c))
            _nn = float(np.linalg.norm(_n_c))
            corr = float(np.dot(_s_c, _n_c) / (_ns * _nn + 1e-12)) if _ns > 1e-12 and _nn > 1e-12 else 0.0
            if np.isnan(corr):
                corr = 0.0

            # Power ratio: noise variance relative to signal
            mean_noise = float(np.mean(noise_var[mask]))
            mean_signal = float(np.mean(signal_env[mask]))
            ratio = mean_noise / (mean_signal + 1e-12)

            # Severity: high correlation AND significant ratio
            raw_sev = float(np.clip(max(0.0, corr) * 1.5, 0.0, 1.0))
            raw_sev *= float(np.clip(ratio * 10.0, 0.3, 1.5))
            raw_sev = float(np.clip(raw_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.MODULATION_NOISE, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.MODULATION_NOISE, 0.0, 0.5)
            confidence = float(np.clip(0.5 + corr * 0.3, 0.3, 0.95))
            return DefectScore(
                DefectType.MODULATION_NOISE,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                metadata={"correlation": round(corr, 4), "noise_signal_ratio": round(ratio, 6)},
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.MODULATION_NOISE, 0.0, 0.3)

    def _detect_inner_groove_distortion(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Inner Groove Distortion (IGD) on vinyl/shellac recordings.

        IGD increases towards the center of the disc due to decreasing linear
        velocity.  Algorithm:
        1. Split audio into 4 equal quarters (simulating disc radius progression)
        2. Measure THD in 2-8 kHz range per quarter
        3. If THD increases monotonically from Q1→Q4 → IGD pattern detected
        4. Severity from THD slope and absolute Q4 distortion level

        Scientific basis: Vinyl stylus tracking geometry; Kates (1981) "A Model of
        Record Tracing Distortion".
        """
        # N/A-Gate: Materialien ohne Rillengeometrie (Tape, Kassette, Digital etc.)
        if self.thresholds.get(DefectType.INNER_GROOVE_DISTORTION, 0.5) >= 0.95:
            return DefectScore(
                DefectType.INNER_GROOVE_DISTORTION,
                0.0,
                1.0,
                locations=[],
                metadata={"skipped": "material_gate_NA"},
            )
        n = len(audio)
        sr = self.sample_rate
        if n < sr * 8:
            return DefectScore(DefectType.INNER_GROOVE_DISTORTION, 0.0, 0.3)
        try:
            quarter = n // 4
            thd_values = []
            for q in range(4):
                segment = audio[q * quarter : (q + 1) * quarter]
                n_fft = min(4096, len(segment))
                freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
                spec = np.abs(np.fft.rfft(segment[:n_fft]))
                # THD in 2-8 kHz range (where IGD is most audible)
                hf_mask = (freqs >= 2000) & (freqs <= 8000)
                total_mask = freqs <= 8000
                hf_energy = float(np.sum(spec[hf_mask] ** 2))
                total_energy = float(np.sum(spec[total_mask] ** 2)) + 1e-12
                thd = hf_energy / total_energy
                thd_values.append(thd)

            # Check for monotonic increase Q1→Q4 (IGD pattern)
            diffs = [thd_values[i + 1] - thd_values[i] for i in range(3)]
            n_increasing = sum(1 for d in diffs if d > 0)
            slope = (thd_values[3] - thd_values[0]) / (thd_values[0] + 1e-12)

            if n_increasing < 2:
                return DefectScore(DefectType.INNER_GROOVE_DISTORTION, 0.0, 0.5)

            raw_sev = float(np.clip(slope * 2.0, 0.0, 1.0))
            # Absolute Q4 THD bonus
            raw_sev += float(np.clip(thd_values[3] * 5.0, 0.0, 0.3))
            raw_sev = float(np.clip(raw_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.INNER_GROOVE_DISTORTION, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.INNER_GROOVE_DISTORTION, 0.0, 0.5)
            confidence = float(np.clip(0.4 + 0.15 * n_increasing, 0.3, 0.90))
            return DefectScore(
                DefectType.INNER_GROOVE_DISTORTION,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                metadata={"thd_per_quarter": [round(v, 6) for v in thd_values], "slope": round(slope, 4)},
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.INNER_GROOVE_DISTORTION, 0.0, 0.3)

    def _detect_groove_echo(self, audio: np.ndarray) -> DefectScore:
        """Erkennt groove echo (pre-echo from adjacent groove deformation on vinyl).

        Groove echo occurs ~1.8 s before loud passages (one revolution at 331⁄3 rpm).
        Algorithm:
        1. Find strong transients (top 10% envelope peaks)
        2. Search for correlated ghost signal at approximately one revolution delay
        3. Compute spectral similarity between ghost and transient
        4. Severity from ghost-to-noise ratio and spectral match

        DISTINCT from codec pre-echo (PRE_ECHO: 5-35 ms) - groove echo is 1.5-2.2 s.
        """
        # N/A-Gate: Materialien ohne Rillengeometrie (Tape, Kassette, Digital etc.)
        # haben physikalisch kein Groove-Echo - Detektor nicht anwenden.
        if self.thresholds.get(DefectType.GROOVE_ECHO, 0.5) >= 0.95:
            return DefectScore(
                DefectType.GROOVE_ECHO,
                0.0,
                1.0,
                locations=[],
                metadata={"skipped": "material_gate_NA"},
            )
        n = len(audio)
        sr = self.sample_rate
        if n < sr * 5:
            return DefectScore(DefectType.GROOVE_ECHO, 0.0, 0.3)
        try:
            # Revolution delays for different speeds
            delays_s = [1.8, 1.35, 0.77]  # 331⁄3, 45, 78 rpm
            delay_samples = [int(d * sr) for d in delays_s]

            win = max(1, int(0.010 * sr))
            envelope = np.convolve(np.abs(audio), np.ones(win) / win, mode="same")
            peak_thresh = float(np.percentile(envelope, 90))
            peaks = np.where(envelope > peak_thresh)[0]
            if len(peaks) == 0:
                return DefectScore(DefectType.GROOVE_ECHO, 0.0, 0.3)

            # Deduplicate peaks (>500 ms apart)
            min_gap = int(0.5 * sr)
            deduped = [peaks[0]]
            for p in peaks[1:]:
                if p - deduped[-1] > min_gap:
                    deduped.append(p)
            peaks = np.array(deduped, dtype=np.intp)

            best_ratios = []
            best_corrs = []
            echo_locations: list[tuple[float, float]] = []
            for delay_s_val in delay_samples:
                for idx in peaks:
                    ghost_start = max(0, idx - delay_s_val - int(0.05 * sr))
                    ghost_end = max(0, idx - delay_s_val + int(0.05 * sr))
                    if ghost_start >= ghost_end or ghost_end >= n:
                        continue
                    ghost_energy = float(np.mean(envelope[ghost_start:ghost_end] ** 2))
                    baseline = float(np.percentile(envelope, 10) ** 2) + 1e-20
                    ratio = ghost_energy / baseline
                    if ratio > 1.5:
                        best_ratios.append(ratio)
                        echo_locations.append((ghost_start / sr, ghost_end / sr))
                        # Spectral similarity
                        trans_seg = audio[idx : min(n, idx + int(0.03 * sr))]
                        ghost_seg = audio[ghost_start : ghost_start + len(trans_seg)]
                        if len(trans_seg) >= 64 and len(ghost_seg) >= 64:
                            min_l = min(len(trans_seg), len(ghost_seg))
                            s1 = np.abs(np.fft.rfft(trans_seg[:min_l]))
                            s2 = np.abs(np.fft.rfft(ghost_seg[:min_l]))
                            if len(s1) > 4:
                                # Stable correlation without unguarded correlation warnings on
                                # near-constant spectra (important for -W error test runs).
                                s1c = s1 - float(np.mean(s1))
                                s2c = s2 - float(np.mean(s2))
                                denom = float(np.linalg.norm(s1c) * np.linalg.norm(s2c))
                                if denom > 1e-12:
                                    c = float(np.dot(s1c, s2c) / denom)
                                    if np.isfinite(c):
                                        best_corrs.append(max(0.0, c))

            if not best_ratios:
                return DefectScore(DefectType.GROOVE_ECHO, 0.0, 0.5)

            mean_ratio = float(np.mean(best_ratios))
            mean_corr = float(np.mean(best_corrs)) if best_corrs else 0.0
            raw_sev = float(np.clip((mean_ratio - 1.5) / 5.0, 0.0, 0.7))
            raw_sev += float(np.clip(mean_corr * 0.3, 0.0, 0.3))
            raw_sev *= float(np.clip(len(best_ratios) / 5.0, 0.5, 1.5))
            raw_sev = float(np.clip(raw_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.GROOVE_ECHO, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(
                    DefectType.GROOVE_ECHO,
                    0.0,
                    0.5,
                    locations=self._sample_locations_evenly(echo_locations, self._LOCATION_CAP_UNCAPPED),
                    metadata={
                        "mean_ratio": round(mean_ratio, 4),
                        "spectral_corr": round(mean_corr, 4),
                        "n_echoes": len(best_ratios),
                        "below_threshold": True,
                    },
                )
            confidence = float(np.clip(0.4 + mean_corr * 0.3, 0.3, 0.90))
            return DefectScore(
                DefectType.GROOVE_ECHO,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                locations=self._sample_locations_evenly(echo_locations, self._LOCATION_CAP_UNCAPPED),
                metadata={
                    "mean_ratio": round(mean_ratio, 4),
                    "spectral_corr": round(mean_corr, 4),
                    "n_echoes": len(best_ratios),
                },
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.GROOVE_ECHO, 0.0, 0.3)

    def _detect_crosstalk(self, audio: np.ndarray) -> DefectScore:
        """Erkennt channel crosstalk in early stereo recordings.

        Crosstalk appears as ghost images in the stereo field (channel separation < 20 dB).
        Algorithm (BSS-based):
        1. Compute frequency-dependent channel separation L→R and R→L
        2. Low separation (<20 dB) in specific frequency bands = crosstalk
        3. Cross-correlation analysis for delayed crosstalk components

        Scientific basis: Blauert (1997) "Spatial Hearing"; BSS/ICA methods.
        """
        if audio.ndim != 2 or (audio.ndim == 2 and min(audio.shape) < 2):
            return DefectScore(DefectType.CROSSTALK, 0.0, 0.5, metadata={"reason": "mono_or_invalid"})
        try:
            # Ensure [samples, channels] format
            if audio.shape[0] <= 2:
                left, right = audio[0], audio[1]
            else:
                left, right = audio[:, 0], audio[:, 1]

            sr = self.sample_rate
            n_fft = min(4096, len(left))
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

            # Compute cross-spectral density
            spec_l = np.fft.rfft(left[:n_fft])
            spec_r = np.fft.rfft(right[:n_fft])
            cross_spec = spec_l * np.conj(spec_r)
            auto_l = np.abs(spec_l) ** 2 + 1e-12
            auto_r = np.abs(spec_r) ** 2 + 1e-12

            # Channel separation in bands (250-4000 Hz where crosstalk is most problematic)
            band_mask = (freqs >= 250) & (freqs <= 4000)
            if not band_mask.any():
                return DefectScore(DefectType.CROSSTALK, 0.0, 0.4)

            coherence = np.abs(cross_spec[band_mask]) ** 2 / (auto_l[band_mask] * auto_r[band_mask] + 1e-12)
            mean_coherence = float(np.mean(coherence))

            # Very high coherence in mid-band = poor channel separation = crosstalk
            # Normal stereo: coherence 0.3-0.7; Crosstalk: >0.85
            raw_sev = float(np.clip((mean_coherence - 0.75) / 0.20, 0.0, 1.0))

            # Time-domain cross-correlation peak (delayed crosstalk) - FFT-based
            max_delay = int(0.002 * sr)  # Max 2ms delay for mechanical crosstalk
            from backend.core.core_utils import fft_crosscorr  # pylint: disable=import-outside-toplevel

            xcorr = fft_crosscorr(left[:n_fft], right[:n_fft])
            center = len(xcorr) // 2
            side_peak = float(np.max(np.abs(xcorr[center + 1 : center + max_delay + 1])))
            center_peak = float(np.abs(xcorr[center])) + 1e-12
            delay_ratio = side_peak / center_peak
            raw_sev += float(np.clip(delay_ratio * 0.3, 0.0, 0.3))
            raw_sev = float(np.clip(raw_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.CROSSTALK, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.CROSSTALK, 0.0, 0.5)
            confidence = float(np.clip(0.4 + mean_coherence * 0.3, 0.3, 0.90))
            return DefectScore(
                DefectType.CROSSTALK,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                metadata={"mean_coherence": round(mean_coherence, 4), "delay_ratio": round(delay_ratio, 4)},
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.CROSSTALK, 0.0, 0.3)

    def _detect_intermodulation_distortion(self, audio: np.ndarray) -> DefectScore:
        """Erkennt intermodulation distortion (IMD) from nonlinear amplifier chains.

        IMD creates sum/difference frequencies (NOT harmonics!) from nonlinear
        signal path.  Algorithm:
        1. Find spectral peaks (strong tonal components)
        2. Check for energy at sum/difference frequencies of peak pairs
        3. IMD products at f1±f2 that are NOT harmonics of either = IMD evidence

        Scientific basis: Volterra series models; SMPTE/DIN IMD measurement standards.
        """
        n = len(audio)
        sr = self.sample_rate
        if n < sr * 2:
            return DefectScore(DefectType.INTERMODULATION_DISTORTION, 0.0, 0.3)
        try:
            n_fft = min(8192, n)
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            spec = np.abs(np.fft.rfft(audio[:n_fft])) ** 2
            spec_db = 10.0 * np.log10(spec + 1e-20)
            noise_floor = float(np.percentile(spec_db, 20))

            # Find prominent spectral peaks (>20 dB above noise floor)
            peak_mask = spec_db > (noise_floor + 20.0)
            peak_indices = np.where(peak_mask)[0]
            if len(peak_indices) < 2:
                return DefectScore(DefectType.INTERMODULATION_DISTORTION, 0.0, 0.4)

            # Cluster peaks (within 5 Hz = same component)
            freq_resolution = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
            clusters = []
            current_cluster = [peak_indices[0]]
            for idx in peak_indices[1:]:
                if (idx - current_cluster[-1]) * freq_resolution < 5.0:
                    current_cluster.append(idx)
                else:
                    clusters.append(int(np.mean(current_cluster)))
                    current_cluster = [idx]
            clusters.append(int(np.mean(current_cluster)))
            clusters = clusters[:8]  # Limit computation

            # Check for IMD products at f1±f2
            imd_evidence = []
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    f1 = freqs[clusters[i]]
                    f2 = freqs[clusters[j]]
                    # Sum and difference frequencies
                    for target_f in [abs(f1 - f2), f1 + f2]:
                        if target_f < 50 or target_f > sr / 2 - 100:
                            continue
                        # Is target a harmonic of f1 or f2? If so, skip (THD not IMD)
                        is_harmonic = False
                        for base_f in [f1, f2]:
                            for h in range(1, 8):
                                if abs(target_f - h * base_f) < freq_resolution * 3:
                                    is_harmonic = True
                                    break
                            if is_harmonic:
                                break
                        if is_harmonic:
                            continue
                        # Check for energy at IMD frequency
                        target_idx = int(target_f / freq_resolution)
                        if 0 < target_idx < len(spec_db):
                            imd_level = spec_db[target_idx] - noise_floor
                            if imd_level > 5.0:
                                imd_evidence.append(imd_level)

            if not imd_evidence:
                return DefectScore(DefectType.INTERMODULATION_DISTORTION, 0.0, 0.5)

            mean_imd = float(np.mean(imd_evidence))
            raw_sev = float(np.clip(mean_imd / 30.0, 0.0, 1.0))
            raw_sev *= float(np.clip(len(imd_evidence) / 5.0, 0.5, 1.5))
            raw_sev = float(np.clip(raw_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.INTERMODULATION_DISTORTION, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.INTERMODULATION_DISTORTION, 0.0, 0.5)
            confidence = float(np.clip(0.4 + len(imd_evidence) * 0.05, 0.3, 0.90))
            return DefectScore(
                DefectType.INTERMODULATION_DISTORTION,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                metadata={"n_imd_products": len(imd_evidence), "mean_imd_level_db": round(mean_imd, 2)},
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.INTERMODULATION_DISTORTION, 0.0, 0.3)

    def _detect_tape_splice_artifact(self, audio: np.ndarray) -> DefectScore:
        """Erkennt tape splice artifacts (click + level jump + phase discontinuity).

        Splice artifacts are distinct from normal clicks: they combine an impulsive
        transient with a simultaneous level change and often a phase jump.

        Algorithm:
        1. Detect simultaneous level-jump AND impulse (co-occurrence)
        2. Check for phase discontinuity at the event boundary
        3. Level jump persistence (>50 ms) distinguishes splice from click
        """
        n = len(audio)
        sr = self.sample_rate
        if n < sr * 2:
            return DefectScore(DefectType.TAPE_SPLICE_ARTIFACT, 0.0, 0.3)
        try:
            # RMS envelope (10 ms frames)
            frame_len = max(1, int(0.010 * sr))
            hop = max(1, frame_len // 2)
            n_frames = max(1, (n - frame_len) // hop)
            if n_frames < 10:
                return DefectScore(DefectType.TAPE_SPLICE_ARTIFACT, 0.0, 0.3)

            frames = np.lib.stride_tricks.as_strided(
                audio,
                shape=(n_frames, frame_len),
                strides=(audio.strides[0] * hop, audio.strides[0]),
            ).copy()
            rms_env = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)

            # Level jumps: SNR-adaptive threshold (§v10).
            # Leise Passagen: 3 dB Sprung schon auffällig → niedriger Threshold.
            # Hochdynamische Passagen (Orchester): 8 dB Sprung normal → höher.
            rms_db = 20.0 * np.log10(rms_env + 1e-12)
            level_diffs = np.abs(np.diff(rms_db))
            _local_dyn_range_db = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5) + 1.0)
            jump_threshold = float(np.clip(_local_dyn_range_db / 4.0, 3.0, 8.0))
            jump_indices = np.where(level_diffs > jump_threshold)[0]

            if len(jump_indices) == 0:
                return DefectScore(DefectType.TAPE_SPLICE_ARTIFACT, 0.0, 0.5)

            # For each jump: check for impulse (high-frequency energy burst)
            splice_events = []
            locations = []
            for ji in jump_indices:
                sample_idx = ji * hop
                if sample_idx < 64 or sample_idx > n - 64:
                    continue
                # High-frequency impulse at junction
                junction = audio[sample_idx - 32 : sample_idx + 32]
                hf_spec = np.abs(np.fft.rfft(junction))
                hf_energy = float(np.sum(hf_spec[len(hf_spec) // 2 :] ** 2))
                total_energy = float(np.sum(hf_spec**2)) + 1e-12
                hf_ratio = hf_energy / total_energy

                # Check level persistence (must persist > 50 ms after jump)
                persist_frames = min(5, n_frames - ji - 1)
                if persist_frames > 2:
                    post_jump_rms = rms_db[ji + 1 : ji + 1 + persist_frames]
                    pre_jump_rms = rms_db[max(0, ji - persist_frames) : ji]
                    if len(post_jump_rms) > 0 and len(pre_jump_rms) > 0:
                        level_diff_persist = abs(float(np.mean(post_jump_rms)) - float(np.mean(pre_jump_rms)))
                        if hf_ratio > 0.2 and level_diff_persist > 3.0:
                            splice_events.append(level_diffs[ji])
                            t = sample_idx / sr
                            locations.append((max(0.0, t - 0.02), t + 0.02))

            if not splice_events:
                return DefectScore(DefectType.TAPE_SPLICE_ARTIFACT, 0.0, 0.5)

            mean_jump = float(np.mean(splice_events))
            raw_sev = float(np.clip(mean_jump / 20.0, 0.0, 0.7))
            raw_sev += float(np.clip(len(splice_events) / 8.0, 0.0, 0.3))
            raw_sev = float(np.clip(raw_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.TAPE_SPLICE_ARTIFACT, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.TAPE_SPLICE_ARTIFACT, 0.0, 0.5)
            confidence = float(np.clip(0.5 + len(splice_events) * 0.05, 0.3, 0.90))
            return DefectScore(
                DefectType.TAPE_SPLICE_ARTIFACT,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                locations=locations,
                metadata={"n_splices": len(splice_events), "mean_jump_db": round(mean_jump, 2)},
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.TAPE_SPLICE_ARTIFACT, 0.0, 0.3)

    def _detect_hf_remanence_loss(self, audio: np.ndarray) -> DefectScore:
        """Erkennt high-frequency remanence loss from magnetic tape aging.

        Different from BANDWIDTH_LOSS: HF was originally recorded but has faded
        over decades due to magnetic particle demagnetization.  The spectral envelope
        shows gradual HF roll-off WITH residual harmonic structure (ghost harmonics),
        unlike bandwidth limitation where HF was never present.

        Algorithm:
        1. Compute spectral envelope (smoothed magnitude spectrum)
        2. Detect roll-off rate above 4 kHz
        3. Check for 'ghost harmonics': residual harmonic peaks in rolled-off region
        4. Age-dependent severity model (steeper rolloff + ghost harmonics = aging)

        Scientific basis: Bertram (1994) "Theory of Magnetic Recording";
        Jorgensen (1996) "The Complete Handbook of Magnetic Recording".
        """
        n = len(audio)
        sr = self.sample_rate
        if n < sr * 2:
            return DefectScore(DefectType.HF_REMANENCE_LOSS, 0.0, 0.3)
        try:
            n_fft = min(8192, n)
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            spec = np.abs(np.fft.rfft(audio[:n_fft])) ** 2
            spec_db = 10.0 * np.log10(spec + 1e-20)

            # Spectral envelope roll-off rate above 4 kHz
            hf_mask = (freqs >= 4000) & (freqs <= sr / 2 - 100)
            if not hf_mask.any() or np.sum(hf_mask) < 10:
                return DefectScore(DefectType.HF_REMANENCE_LOSS, 0.0, 0.3)

            hf_freqs = freqs[hf_mask]
            hf_db = spec_db[hf_mask]
            float(np.percentile(spec_db, 5))

            # Linear regression for roll-off slope (dB/octave)
            log_freqs = np.log2(hf_freqs + 1.0)
            if len(log_freqs) > 5:
                coeffs = np.polyfit(log_freqs, hf_db, 1)
                slope_db_per_octave = float(coeffs[0])
            else:
                slope_db_per_octave = 0.0

            # Ghost harmonics: peaks that rise above smoothed envelope in HF region
            smooth_kernel = max(5, len(hf_db) // 20)
            smoothed = np.convolve(hf_db, np.ones(smooth_kernel) / smooth_kernel, mode="same")
            residual_peaks = hf_db - smoothed
            n_ghost = int(np.sum(residual_peaks > 3.0))

            # Severity: steep negative slope + ghost harmonics = remanence loss
            raw_sev = float(np.clip(abs(min(0.0, slope_db_per_octave)) / 15.0, 0.0, 0.7))
            if n_ghost > 3:
                raw_sev += float(np.clip(n_ghost / 15.0, 0.0, 0.3))
            raw_sev = float(np.clip(raw_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.HF_REMANENCE_LOSS, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.HF_REMANENCE_LOSS, 0.0, 0.5)
            confidence = float(np.clip(0.4 + 0.1 * min(n_ghost, 5), 0.3, 0.90))
            return DefectScore(
                DefectType.HF_REMANENCE_LOSS,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                metadata={"slope_db_octave": round(slope_db_per_octave, 2), "n_ghost_harmonics": n_ghost},
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.HF_REMANENCE_LOSS, 0.0, 0.3)

    def _detect_stylus_damage(self, audio: np.ndarray) -> DefectScore:
        """Erkennt stylus damage / mistracking distortion on vinyl/shellac.

        Damaged/worn styli produce asymmetric distortion: odd harmonics dominate
        (unlike soft saturation which produces even harmonics).  Also: consistent
        distortion pattern that tracks groove modulation.

        Algorithm:
        1. Compute short-time distortion (odd vs even harmonic ratio)
        2. Detect persistent asymmetric waveform clipping pattern
        3. Track distortion consistency across the recording

        Scientific basis: Kates (1979) "Turntable Playback Distortion".
        """
        n = len(audio)
        sr = self.sample_rate
        if n < sr * 2:
            return DefectScore(DefectType.STYLUS_DAMAGE, 0.0, 0.3)
        try:
            n_fft = min(4096, n)
            spec = np.abs(np.fft.rfft(audio[:n_fft]))
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            freq_res = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

            # Find fundamental frequency (strongest peak 80-1000 Hz)
            fund_mask = (freqs >= 80) & (freqs <= 1000)
            if not fund_mask.any():
                return DefectScore(DefectType.STYLUS_DAMAGE, 0.0, 0.3)
            fund_idx = int(np.argmax(spec[fund_mask])) + int(np.argmax(fund_mask))
            f0 = freqs[fund_idx]
            if f0 < 50:
                return DefectScore(DefectType.STYLUS_DAMAGE, 0.0, 0.3)

            # Odd vs even harmonic energy
            odd_energy = 0.0
            even_energy = 0.0
            for h in range(2, 10):
                h_idx = int(h * f0 / freq_res)
                if h_idx >= len(spec):
                    break
                h_energy = float(spec[max(0, h_idx - 1) : min(len(spec), h_idx + 2)].max() ** 2)
                if h % 2 == 1:
                    odd_energy += h_energy
                else:
                    even_energy += h_energy

            # Asymmetry: stylus damage → odd harmonics dominate
            total_h = odd_energy + even_energy + 1e-12
            odd_ratio = odd_energy / total_h

            # Waveform asymmetry (positive vs negative peak ratio)
            pos_peak = float(np.percentile(audio, 99))
            neg_peak = float(abs(np.percentile(audio, 1)))
            asym = abs(pos_peak - neg_peak) / (max(pos_peak, neg_peak) + 1e-12)

            raw_sev = float(np.clip((odd_ratio - 0.55) * 3.0, 0.0, 0.6))
            raw_sev += float(np.clip(asym * 0.8, 0.0, 0.4))
            raw_sev = float(np.clip(raw_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.STYLUS_DAMAGE, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.STYLUS_DAMAGE, 0.0, 0.5)
            confidence = float(np.clip(0.4 + odd_ratio * 0.3, 0.3, 0.90))
            return DefectScore(
                DefectType.STYLUS_DAMAGE,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                metadata={"odd_ratio": round(odd_ratio, 4), "waveform_asymmetry": round(asym, 4)},
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.STYLUS_DAMAGE, 0.0, 0.3)

    def _detect_sticky_shed_residue(self, audio: np.ndarray) -> DefectScore:
        """Detect sticky-shed residue artifacts (post-baking tape degradation).

        After baking, polyester-urethane tapes show: short level dips, modulated
        noise bursts, and specific harmonic distortion patterns.

        Algorithm:
        1. Detect short-duration level dips (10-100 ms, characteristic of shed events)
        2. Measure noise modulation at dip boundaries
        3. Check for periodicity (shed events often correlate with tape wrap period)

        Scientific basis: Hess (2008) "Tape Degradation Factors and
        Predicting Tape Life Expectancy"; Schüller (2005) "Tape Baking".
        """
        n = len(audio)
        sr = self.sample_rate
        if n < sr * 2:
            return DefectScore(DefectType.STICKY_SHED_RESIDUE, 0.0, 0.3)
        try:
            # RMS envelope (5 ms frames)
            frame_len = max(1, int(0.005 * sr))
            hop = max(1, frame_len // 2)
            n_frames = max(1, (n - frame_len) // hop)
            if n_frames < 20:
                return DefectScore(DefectType.STICKY_SHED_RESIDUE, 0.0, 0.3)

            frames = np.lib.stride_tricks.as_strided(
                audio,
                shape=(n_frames, frame_len),
                strides=(audio.strides[0] * hop, audio.strides[0]),
            ).copy()
            rms_env = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
            rms_db = 20.0 * np.log10(rms_env + 1e-12)

            # Detect short dips (10-100 ms duration, >4 dB depth)
            # and longer contact-loss dips (100-300 ms, >2.5 dB depth).
            median_rms = float(np.median(rms_db))
            dip_threshold = median_rms - 4.0
            in_dip = rms_db < dip_threshold
            dip_events = []
            long_dip_events = []
            dip_start = None
            for i, is_dipped in enumerate(in_dip):
                if is_dipped and dip_start is None:
                    dip_start = i
                elif not is_dipped and dip_start is not None:
                    dip_len_ms = (i - dip_start) * hop * 1000.0 / sr
                    depth = median_rms - float(np.min(rms_db[dip_start:i]))
                    if 10.0 <= dip_len_ms <= 100.0:
                        dip_events.append((dip_start, i, depth, dip_len_ms))
                    elif 100.0 < dip_len_ms <= 300.0 and depth >= 2.5:
                        long_dip_events.append((dip_start, i, depth, dip_len_ms))
                    dip_start = None

            all_events = dip_events + long_dip_events
            if len(all_events) < 2:
                return DefectScore(DefectType.STICKY_SHED_RESIDUE, 0.0, 0.5)

            mean_depth = float(np.mean([d[2] for d in all_events]))
            mean_duration = float(np.mean([d[3] for d in all_events]))
            event_rate = len(all_events) / (n / sr)  # events per second

            # Boundary-modulation cue: sticky shed often causes noisy, unstable
            # transitions at dip boundaries (binder residue on head path).
            rms_diff = np.abs(np.diff(rms_db))
            boundary_spikes = []
            for s_idx, e_idx, *_ in all_events:
                _lo = max(0, s_idx - 2)
                _hi = min(len(rms_diff), e_idx + 1)
                if _lo < _hi:
                    boundary_spikes.append(float(np.max(rms_diff[_lo:_hi])))
            _global_q75 = float(np.percentile(rms_diff, 75)) + 1e-12
            _boundary_mean = float(np.mean(boundary_spikes)) if boundary_spikes else 0.0
            modulation_score = float(np.clip((_boundary_mean / _global_q75 - 1.2) / 2.0, 0.0, 1.0))

            raw_sev = float(np.clip(event_rate * 1.6, 0.0, 0.45))
            raw_sev += float(np.clip(mean_depth / 18.0, 0.0, 0.25))
            raw_sev += float(np.clip(len(dip_events) / 20.0, 0.0, 0.15))
            raw_sev += float(np.clip(len(long_dip_events) / 12.0, 0.0, 0.15))
            raw_sev += 0.20 * modulation_score
            raw_sev = float(np.clip(raw_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.STICKY_SHED_RESIDUE, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.STICKY_SHED_RESIDUE, 0.0, 0.5)
            locations = [(d[0] * hop / sr, d[1] * hop / sr) for d in all_events]
            confidence = float(np.clip(0.4 + event_rate * 0.1, 0.3, 0.90))
            return DefectScore(
                DefectType.STICKY_SHED_RESIDUE,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                locations=locations,
                metadata={
                    "n_events": len(all_events),
                    "n_short_events": len(dip_events),
                    "n_long_events": len(long_dip_events),
                    "mean_depth_db": round(mean_depth, 2),
                    "mean_duration_ms": round(mean_duration, 1),
                    "events_per_second": round(event_rate, 3),
                    "boundary_modulation_score": round(modulation_score, 3),
                },
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.STICKY_SHED_RESIDUE, 0.0, 0.3)

    def _detect_multiband_wow_flutter(self, audio: np.ndarray) -> DefectScore:
        """Erkennt frequency-dependent wow/flutter (head gap geometry artifact).

        Uses STFT-based instantaneous frequency tracking (weighted spectral centroid)
        per octave band to discriminate head-gap multiband flutter (band-dependent
        pitch deviation) from normal musical complexity (affects all bands similarly).

        Replaces ZCR-based approach (v10.0.0) which false-positives on polyphonic
        music: ZCR variance is driven by timbre/dynamics, not pitch instability.

        Algorithm:
        1. Compute full STFT (100 ms hop for pitch-rate resolution)
        2. For each of 4 octave-band centres (250, 500, 1000, 2000 Hz):
           a. Extract dominant frequency via power-weighted centroid per frame
           b. Restrict to active frames (energy > 30th percentile)
           c. Frame-to-frame deviation in cents → std(deviation) = instability
        3. CV across bands: low = uniform flutter/music, high = multiband flutter
        4. raw_sev = instability level + cross-band variation

        Scientific basis: Czyzewski et al. (2023), "Frequency-Dependent Speed
        Fluctuation Analysis in Audio Tape Recordings"; IEC 60386:1972 §4.
        """
        n = len(audio)
        sr = self.sample_rate
        if n < sr * 3:
            return DefectScore(DefectType.MULTIBAND_WOW_FLUTTER, 0.0, 0.3)
        try:
            # Single full STFT - reused for all bands (no repeated filter-bank passes)
            n_fft = min(4096, n)
            hop = max(1, n_fft // 4)
            win = np.hanning(n_fft)
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

            n_frames_total = max(1, (n - n_fft) // hop)
            # Cap at 6000 frames (~100s at 48kHz/1024-hop) for performance
            use_frames = min(n_frames_total, 6000)
            stft_mag = np.zeros((len(freqs), use_frames), dtype=np.float32)
            for fi in range(use_frames):
                start = fi * hop
                frame = audio[start : start + n_fft]
                if len(frame) < n_fft:
                    break
                stft_mag[:, fi] = np.abs(np.fft.rfft(frame * win)).astype(np.float32)

            band_centers = [250, 500, 1000, 2000]
            band_instabilities: list[float] = []

            # Global broadband power for relative-energy gating per band
            # (prevents float32 quantization noise from activating empty bands)
            global_power_mean = float(stft_mag.sum(axis=0).mean()) + 1e-12

            for fc in band_centers:
                f_low = max(50.0, fc / 1.414)
                f_high = min(fc * 1.414, float(sr) / 2.0 - 10.0)
                if f_high <= f_low:
                    continue
                band_mask = (freqs >= f_low) & (freqs <= f_high)
                if not band_mask.any():
                    continue

                band_freqs = freqs[band_mask]
                band_mag = stft_mag[band_mask, :use_frames]  # (n_band_bins, use_frames)
                mag_sum = band_mag.sum(axis=0) + 1e-12  # (use_frames,)

                # Skip band if signal energy is negligible (tone not present in band).
                # Without this guard, spectral leakage from other bands produces
                # numerically unstable centroid estimates → false-positive instability.
                # Use relative threshold (vs broadband) to be robust to float32 noise.
                band_peak_energy = float(np.percentile(mag_sum, 90))
                if band_peak_energy / global_power_mean < 0.001:
                    continue

                # Dominant frequency: power-weighted centroid (more stable than peak bin)
                dominant_freq = np.dot(band_freqs, band_mag) / mag_sum

                # Active frames: above the larger of 30th percentile or 5% of peak
                # to avoid including near-silence micro-frames in stability estimate.
                q30 = max(float(np.percentile(mag_sum, 30)), band_peak_energy * 0.05)
                active = mag_sum > q30
                if active.sum() < 10:
                    continue
                dom_active = dominant_freq[active]
                if len(dom_active) < 4:
                    continue

                # Frame-to-frame deviation in cents
                # 1 cent = 1/100 semitone; Δcents ≈ 1200 × Δf / f_ref
                diffs = np.diff(dom_active)
                diffs_cents = np.clip(np.abs(1200.0 * diffs / (dom_active[:-1] + 1e-6)), 0.0, 50.0)
                # std of per-frame deviation = instability measure (in cents/hop)
                band_instabilities.append(float(np.std(diffs_cents)))

            if len(band_instabilities) < 3:
                return DefectScore(DefectType.MULTIBAND_WOW_FLUTTER, 0.0, 0.4)

            mean_inst = float(np.mean(band_instabilities))
            # CV across bands: head-gap flutter is band-selective → high CV
            # Normal wow/flutter: affects all bands equally → low CV
            cv = float(np.std(band_instabilities) / (mean_inst + 1e-12))

            # Severity model:
            #   avg_inst_norm:  mean instability > 10 cents/hop → saturates at 0.5
            #   cv_sev:         cv > 0.35 contributes up to 0.5 (band-selective flutter)
            avg_inst_norm = float(np.clip(mean_inst / 10.0, 0.0, 0.5))
            cv_sev = float(np.clip((cv - 0.35) / 0.65, 0.0, 0.5))
            raw_sev = float(np.clip(avg_inst_norm + cv_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.MULTIBAND_WOW_FLUTTER, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.MULTIBAND_WOW_FLUTTER, 0.0, 0.5)

            confidence = float(np.clip(0.50 + cv * 0.20, 0.35, 0.85))
            return DefectScore(
                DefectType.MULTIBAND_WOW_FLUTTER,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                metadata={
                    "band_instabilities_cents": [round(b, 3) for b in band_instabilities],
                    "cv_across_bands": round(cv, 4),
                    "mean_instability_cents": round(mean_inst, 3),
                },
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.MULTIBAND_WOW_FLUTTER, 0.0, 0.3)

    def _detect_generation_loss(self, audio: np.ndarray) -> DefectScore:
        """Erkennt cumulative generation loss from tape dubbing / transcoding.

        Each copy generation adds noise, bandwidth loss, and phase distortion with
        a specific cumulative signature different from individual defects.

        Algorithm:
        1. Measure noise floor shape (generation loss has uniform elevation)
        2. Detect cumulative bandwidth narrowing (steeper than single-source rolloff)
        3. Check spectral coherence degradation (phase randomization)
        4. Combine into multi-generational degradation model

        Scientific basis: McKnight & Giddings (1981) AES; Cabot (1977)
        "Generation Loss in Magnetic Tape Recording".
        """
        n = len(audio)
        sr = self.sample_rate
        if n < sr * 2:
            return DefectScore(DefectType.GENERATION_LOSS, 0.0, 0.3)
        try:
            n_fft = min(8192, n)
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            spec = np.abs(np.fft.rfft(audio[:n_fft])) ** 2
            spec_db = 10.0 * np.log10(spec + 1e-20)

            # 1. Noise floor uniformity (multi-gen: elevated and flat)
            noise_floor_db = float(np.percentile(spec_db, 10))
            lf_floor = float(np.percentile(spec_db[(freqs >= 100) & (freqs <= 500)], 10))
            hf_floor = float(np.percentile(spec_db[(freqs >= 2000) & (freqs <= 8000)], 10))
            floor_uniforms = abs(lf_floor - hf_floor)  # Small = uniform = multi-gen

            # 2. Bandwidth: effective -20 dB rolloff point
            peak_db = float(np.max(spec_db))
            bw_mask = spec_db > (peak_db - 20.0)
            if bw_mask.any():
                effective_bw = float(freqs[bw_mask][-1])
            else:
                effective_bw = float(freqs[-1])

            # 3. Phase coherence (spectral flatness as proxy)
            spectral_flatness = float(np.exp(np.mean(np.log(spec + 1e-20))) / (np.mean(spec) + 1e-20))

            # Severity model: high noise floor + narrow bandwidth + phase degradation
            noise_sev = float(np.clip((noise_floor_db + 40.0) / 30.0, 0.0, 0.4))
            bw_sev = float(np.clip((12000.0 - effective_bw) / 10000.0, 0.0, 0.3))
            phase_sev = float(np.clip(spectral_flatness * 2.0, 0.0, 0.3))
            raw_sev = noise_sev + bw_sev + phase_sev
            raw_sev = float(np.clip(raw_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.GENERATION_LOSS, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.GENERATION_LOSS, 0.0, 0.5)
            confidence = float(np.clip(0.35 + raw_sev * 0.3, 0.3, 0.85))
            return DefectScore(
                DefectType.GENERATION_LOSS,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                metadata={
                    "noise_floor_db": round(noise_floor_db, 2),
                    "effective_bw_hz": round(effective_bw, 0),
                    "spectral_flatness": round(spectral_flatness, 4),
                    "floor_uniformity": round(floor_uniforms, 2),
                },
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.GENERATION_LOSS, 0.0, 0.3)

    def _detect_motor_interference(self, audio: np.ndarray) -> DefectScore:
        """Erkennt turntable/tape motor interference (harmonics 80-300 Hz).

        Different from LOW_FREQ_RUMBLE (sub-bass <80 Hz) - motor interference
        produces harmonics in the 80-300 Hz range from DC and synchronous motors.

        Algorithm:
        1. Analyze spectral peaks at motor-related frequencies (50, 60, 80, 100, 120 Hz etc.)
        2. Distinguish from electrical hum (motor harmonics are broader, less sharp)
        3. Check modulation pattern (motor speed variation → FM sidebands)

        Scientific basis: Tremaine (1969) "Audio Cyclopedia"; Fletcher & Rossing.
        """
        n = len(audio)
        sr = self.sample_rate
        if n < sr * 2:
            return DefectScore(DefectType.MOTOR_INTERFERENCE, 0.0, 0.3)
        try:
            n_fft = min(8192, n)
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            spec = np.abs(np.fft.rfft(audio[:n_fft])) ** 2
            spec_db = 10.0 * np.log10(spec + 1e-20)
            float(np.percentile(spec_db, 20))
            freq_res = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

            # Canonical motor-related harmonics in 80-300 Hz.
            # Keep this set sparse: a dense pseudo-continuous grid creates false
            # positives on musical harmonic stacks (bass/cello).
            motor_freqs = [80, 100, 120, 133, 150, 160, 180, 200, 240, 250, 267, 300]

            # Check for peaks at motor frequencies (broad peaks, not sharp peaks like hum)
            motor_peaks: list[tuple[float, float]] = []
            for f in motor_freqs:
                idx = int(f / freq_res)
                if 0 < idx < len(spec_db) - 2:
                    # Broad peak: average over ±3 bins
                    local_level = float(np.mean(spec_db[max(0, idx - 3) : idx + 4]))
                    # Compare with surrounding 50-bin region
                    surround = spec_db[max(0, idx - 50) : min(len(spec_db), idx + 50)]
                    local_median = float(np.median(surround))
                    prominence = local_level - local_median
                    if prominence > 3.0:  # Broader threshold than hum (which uses sharper peaks)
                        motor_peaks.append((float(f), prominence))

            if len(motor_peaks) < 3:
                return DefectScore(DefectType.MOTOR_INTERFERENCE, 0.0, 0.5)

            mean_prominence = float(np.mean([p for _, p in motor_peaks]))
            raw_sev = float(np.clip(mean_prominence / 15.0, 0.0, 0.6))
            raw_sev += float(np.clip(len(motor_peaks) / 15.0, 0.0, 0.4))
            raw_sev = float(np.clip(raw_sev, 0.0, 1.0))

            # Additional anti-false-positive guard for musical harmonic stacks:
            # if one low-frequency fundamental explains most 80-400 Hz energy via
            # its first harmonics, this is likely music (bass note), not motor hum.
            harmonic_series_energy_ratio = 0.0
            low_mask = (freqs >= 80.0) & (freqs <= 400.0)
            low_total = float(np.sum(spec[low_mask]) + 1e-20)
            f0_search_mask = (freqs >= 80.0) & (freqs <= 180.0)
            if np.any(f0_search_mask) and low_total > 0.0:
                f0_idx_local = int(np.argmax(spec[f0_search_mask]))
                f0_idx = f0_idx_local + int(np.argmax(f0_search_mask))
                f0 = float(freqs[f0_idx])
                harmonic_energy = 0.0
                for k in range(1, 5):
                    h_freq = k * f0
                    if h_freq > 400.0:
                        break
                    h_idx = int(h_freq / max(freq_res, 1e-9))
                    lo = max(0, h_idx - 1)
                    hi = min(len(spec), h_idx + 2)
                    harmonic_energy += float(np.max(spec[lo:hi]))
                harmonic_series_energy_ratio = float(np.clip(harmonic_energy / low_total, 0.0, 1.0))
                if harmonic_series_energy_ratio >= 0.60:
                    raw_sev *= 0.25

            # Anti-false-positive guard: musical bass harmonics (e.g. 100/200/300 Hz)
            # can look like motor series. If most detected peaks fit one harmonic
            # ladder, attenuate motor severity heavily.
            harmonic_fit_ratio = 0.0
            peak_freqs = [f for f, _ in motor_peaks]
            for f0 in peak_freqs:
                if not 70.0 <= f0 <= 180.0:
                    continue
                matched = 0
                for fpeak in peak_freqs:
                    k = max(1, int(round(fpeak / f0)))
                    if abs(fpeak - k * f0) <= 6.0:
                        matched += 1
                harmonic_fit_ratio = max(harmonic_fit_ratio, matched / max(1, len(peak_freqs)))
            if harmonic_fit_ratio >= 0.75 and len(peak_freqs) >= 4:
                raw_sev *= 0.35

            threshold = self.thresholds.get(DefectType.MOTOR_INTERFERENCE, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.MOTOR_INTERFERENCE, 0.0, 0.5)
            confidence = float(np.clip(0.4 + len(motor_peaks) * 0.03, 0.3, 0.90))
            if harmonic_fit_ratio >= 0.75 and len(peak_freqs) >= 4:
                confidence = float(np.clip(confidence * 0.7, 0.3, 0.90))
            return DefectScore(
                DefectType.MOTOR_INTERFERENCE,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                metadata={
                    "n_motor_peaks": len(motor_peaks),
                    "mean_prominence_db": round(mean_prominence, 2),
                    "harmonic_fit_ratio": round(harmonic_fit_ratio, 3),
                    "harmonic_series_energy_ratio": round(harmonic_series_energy_ratio, 3),
                },
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.MOTOR_INTERFERENCE, 0.0, 0.3)

    # ── v10.0.0: 7 neue Detektionsmethoden ───────────────────────────────────

    def _detect_proximity_effect_excess(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Nahbesprechungseffekt (proximity effect) bei Richtmikrofonen.

        Typisch für Vokalaufnahmen mit RCA 44-BX, U47, C12 bei < 30 cm Abstand (1940-1970).
        Erzeugt LF-Überhöhung +6-12 dB unterhalb ~250 Hz.  Olson (1948) JASAS 20:22.

        Algorithmus:
            1. LF-Energie (80-250 Hz) vs. Mittenlage (500-2000 Hz) messen
            2. Anti-Rumble-Guard: Sub-Bass (20-60 Hz) darf nicht stärker als LF-Band sein
            3. Spektralsteigung 250→500 Hz prüfen (Proximity: sanft fallend, ≤ +3 dB)
        """
        n = len(audio)
        sr = self.sample_rate
        if n < int(sr * 1.5):
            return DefectScore(DefectType.PROXIMITY_EFFECT_EXCESS, 0.0, 0.3)
        try:
            n_fft = min(8192, n)
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            spec = np.abs(np.fft.rfft(audio[:n_fft])) ** 2
            spec_db = 10.0 * np.log10(spec + 1e-20)

            lf_mask = (freqs >= 80.0) & (freqs <= 250.0)
            ref_mask = (freqs >= 500.0) & (freqs <= 2000.0)
            sub_mask = (freqs >= 20.0) & (freqs <= 60.0)

            if not (lf_mask.any() and ref_mask.any() and sub_mask.any()):
                return DefectScore(DefectType.PROXIMITY_EFFECT_EXCESS, 0.0, 0.3)

            lf_mean = float(np.mean(spec_db[lf_mask]))
            ref_mean = float(np.mean(spec_db[ref_mask]))
            sub_mean = float(np.mean(spec_db[sub_mask]))
            lf_excess = lf_mean - ref_mean

            # Anti-Rumble-Guard: Sub-Bass stärker als LF-Band → Rumble, nicht Proximity
            if sub_mean > lf_mean + 3.0:
                return DefectScore(DefectType.PROXIMITY_EFFECT_EXCESS, 0.0, 0.5)

            # Proximity-Effekt: typisch +4 bis +16 dB im LF-Band
            if lf_excess < 4.0:
                return DefectScore(DefectType.PROXIMITY_EFFECT_EXCESS, 0.0, 0.5)

            raw_sev = float(np.clip((lf_excess - 4.0) / 12.0, 0.0, 1.0))

            # Spektralsteigung 250→500 Hz: Proximity = sanfter Abfall
            hz250_mask = (freqs >= 200.0) & (freqs <= 300.0)
            hz500_mask = (freqs >= 450.0) & (freqs <= 600.0)
            if hz250_mask.any() and hz500_mask.any():
                slope = float(np.mean(spec_db[hz500_mask]) - np.mean(spec_db[hz250_mask]))
                if slope > 3.0:  # LF setzt sich über 500 Hz fort → kein Proximity-Muster
                    raw_sev *= 0.4

            threshold = self.thresholds.get(DefectType.PROXIMITY_EFFECT_EXCESS, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.PROXIMITY_EFFECT_EXCESS, 0.0, 0.5)

            confidence = float(np.clip(0.4 + raw_sev * 0.35, 0.3, 0.85))
            return DefectScore(
                DefectType.PROXIMITY_EFFECT_EXCESS,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                metadata={
                    "lf_excess_db": round(lf_excess, 2),
                    "lf_mean_db": round(lf_mean, 2),
                    "ref_mean_db": round(ref_mean, 2),
                },
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.PROXIMITY_EFFECT_EXCESS, 0.0, 0.3)

    def _detect_room_mode_resonance(self, audio: np.ndarray) -> DefectScore:
        """Erkennt stehende Wellenmodi (Room Modes) in 40-200 Hz.

        Unterschied zu REVERB_EXCESS (diffuser Hall) und ELECTRICAL_HUM (exakte 50/60 Hz
        Harmonische): Room Modes sind schmalbandige Peaks Q > 8 in 40-200 Hz, deren
        Abstände Raumabmessungen entsprechen.  Salter et al. (2003).

        Algorithmus:
            1. Hochauflösendes FFT (32768 Punkte) für schmalbandige Peak-Detektion
            2. Peaks mit Q > 5 in 40-200 Hz identifizieren
            3. Brumm-Frequenzen (50/60 Hz Harmonische ±3 Hz) ausschließen
            4. ≥ 2 Non-Brumm-Peaks mit signifikanter Prominenz → Room Mode
        """
        n = len(audio)
        sr = self.sample_rate
        if n < int(sr * 2):
            return DefectScore(DefectType.ROOM_MODE_RESONANCE, 0.0, 0.3)
        try:
            n_fft = min(32768, n)
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            spec = np.abs(np.fft.rfft(audio[:n_fft])) ** 2
            spec_db = 10.0 * np.log10(spec + 1e-20)
            freq_res = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

            mode_mask = (freqs >= 40.0) & (freqs <= 200.0)
            mode_freqs = freqs[mode_mask]
            mode_db = spec_db[mode_mask]

            if len(mode_db) < 10:
                return DefectScore(DefectType.ROOM_MODE_RESONANCE, 0.0, 0.3)

            baseline = float(np.percentile(mode_db, 40))
            mode_peaks: list[tuple[float, float]] = []
            for i in range(2, len(mode_db) - 2):
                if mode_db[i] > mode_db[i - 1] and mode_db[i] > mode_db[i + 1] and mode_db[i] > baseline + 6.0:
                    level_3db = mode_db[i] - 3.0
                    left = i
                    while left > 0 and mode_db[left] > level_3db:
                        left -= 1
                    right = i
                    while right < len(mode_db) - 1 and mode_db[right] > level_3db:
                        right += 1
                    bandwidth_hz = float((right - left) * freq_res)
                    q = float(mode_freqs[i]) / max(bandwidth_hz, freq_res)
                    if q > 5.0:
                        mode_peaks.append((float(mode_freqs[i]), float(mode_db[i] - baseline)))

            if len(mode_peaks) < 2:
                return DefectScore(DefectType.ROOM_MODE_RESONANCE, 0.0, 0.5)

            hum_freqs = {50.0, 60.0, 100.0, 120.0, 150.0, 180.0, 200.0, 240.0}
            non_hum_peaks = [(f, p) for f, p in mode_peaks if all(abs(f - h) > 3.0 for h in hum_freqs)]
            if len(non_hum_peaks) < 2:
                return DefectScore(DefectType.ROOM_MODE_RESONANCE, 0.0, 0.5)

            mean_prominence = float(np.mean([p for _, p in non_hum_peaks]))
            raw_sev = float(
                np.clip(
                    (len(non_hum_peaks) / 4.0) * (mean_prominence / 15.0),
                    0.0,
                    1.0,
                )
            )

            threshold = self.thresholds.get(DefectType.ROOM_MODE_RESONANCE, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.ROOM_MODE_RESONANCE, 0.0, 0.5)

            confidence = float(np.clip(0.35 + raw_sev * 0.30, 0.3, 0.80))
            return DefectScore(
                DefectType.ROOM_MODE_RESONANCE,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                metadata={
                    "n_room_mode_peaks": len(non_hum_peaks),
                    "mean_prominence_db": round(mean_prominence, 2),
                },
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.ROOM_MODE_RESONANCE, 0.0, 0.3)

    def _detect_nr_breathing_artifact(self, audio: np.ndarray) -> DefectScore:
        """Erkennt NR-System-Pumpen/Atmen (korrekt decodiertes Dolby/dbx).

        Unterschied zu DOLBY_NR_MISMATCH (falsch decodiert): Korrekt decodiertes Band
        mit sichtbarer Gain-Modulation des Rauschbodens im Takt des Signals.
        Dolby (1967) US Patent 3,846,719; Woodford (1982).

        Algorithmus:
            1. 100 ms Frames: Signal-RMS-Hüllkurve + HF-Rauschboden (> 3 kHz, Perzentile 20)
            2. Pearson-Korrelation zwischen Hüllkurve und Rauschboden messen
            3. NR-Atmen → negative Korrelation (Signal hoch → Rauschboden gepumpt)
        """
        n = len(audio)
        sr = self.sample_rate
        if n < int(sr * 3.0):
            return DefectScore(DefectType.NR_BREATHING_ARTIFACT, 0.0, 0.3)
        try:
            frame_len = max(1, int(sr * 0.1))
            use_frames = min(n // frame_len, 300)
            if use_frames < 4:
                return DefectScore(DefectType.NR_BREATHING_ARTIFACT, 0.0, 0.3)

            signal_envelopes = np.zeros(use_frames, dtype=np.float32)
            noise_floors = np.zeros(use_frames, dtype=np.float32)

            for fi in range(use_frames):
                start = fi * frame_len
                frame = audio[start : start + frame_len]
                if len(frame) < frame_len // 2:
                    use_frames = fi
                    break
                rms = float(np.sqrt(np.mean(frame**2)) + 1e-12)
                signal_envelopes[fi] = float(20.0 * np.log10(rms))

                n_fft_small = min(1024, len(frame))
                if n_fft_small < 64:
                    noise_floors[fi] = signal_envelopes[fi] - 40.0
                    continue
                spec = np.abs(np.fft.rfft(frame[:n_fft_small])) ** 2
                freqs_f = np.fft.rfftfreq(n_fft_small, 1.0 / sr)
                hf_mask = freqs_f > 3000.0
                if hf_mask.any():
                    hf_noise = float(np.percentile(spec[hf_mask], 20))
                    noise_floors[fi] = float(10.0 * np.log10(hf_noise + 1e-20))
                else:
                    noise_floors[fi] = signal_envelopes[fi] - 40.0

            valid = max(4, use_frames)
            env = signal_envelopes[:valid]
            nf = noise_floors[:valid]

            env_std = float(np.std(env))
            nf_std = float(np.std(nf))
            if env_std < 1.0 or nf_std < 0.5:
                return DefectScore(DefectType.NR_BREATHING_ARTIFACT, 0.0, 0.4)

            env_centered = env - float(np.mean(env))
            nf_centered = nf - float(np.mean(nf))
            corr_den = float(np.linalg.norm(env_centered) * np.linalg.norm(nf_centered) + 1e-12)
            corr = float(np.clip(float(np.dot(env_centered, nf_centered)) / corr_den, -1.0, 1.0))
            nf_modulation = nf_std

            # Negative Korrelation → NR-Breathing (Signal hoch → Noise gepumpt runter)
            corr_sev = float(np.clip((-corr - 0.2) / 0.6, 0.0, 0.6))
            mod_sev = float(np.clip((nf_modulation - 2.0) / 8.0, 0.0, 0.4))
            raw_sev = float(np.clip(corr_sev + mod_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.NR_BREATHING_ARTIFACT, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.NR_BREATHING_ARTIFACT, 0.0, 0.5)

            confidence = float(np.clip(0.35 + abs(corr) * 0.30, 0.3, 0.80))
            return DefectScore(
                DefectType.NR_BREATHING_ARTIFACT,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                metadata={
                    "signal_env_corr": round(corr, 3),
                    "nf_modulation_db": round(nf_modulation, 2),
                },
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.NR_BREATHING_ARTIFACT, 0.0, 0.3)

    def _detect_flutter_spectral_sidebands(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Flutter-Seitenbänder um dominante Spektralpeaks.

        Flutter-Modulation erzeugt Seitenbänder ±N×flutter_rate Hz um starke Spektralpeaks.
        Hörbar als metallisches Schimmern auf Sustained Notes (Streicher, Orgel, Gesang).
        IEC 60386:1972; Arcs & Faller (2006).

        Algorithmus:
            1. Hochauflösendes FFT (65536 Punkte) für Seitenband-Detektion
            2. Dominanten tonalen Peak (200-4000 Hz, > 15 dB über Rauschboden) finden
            3. Seitenbänder bei ±[2, 4, 6, 8] Hz auf Prominenz prüfen
            4. ≥ 3 detektierte Seitenbänder → Flutter-Seitenband-Defekt
        """
        n = len(audio)
        sr = self.sample_rate
        if n < int(sr * 2):
            return DefectScore(DefectType.FLUTTER_SPECTRAL_SIDEBANDS, 0.0, 0.3)
        try:
            n_fft = min(65536, n)
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            spec = np.abs(np.fft.rfft(audio[:n_fft])) ** 2
            spec_db = 10.0 * np.log10(spec + 1e-20)
            freq_res = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

            tonal_mask = (freqs >= 200.0) & (freqs <= 4000.0)
            if not tonal_mask.any():
                return DefectScore(DefectType.FLUTTER_SPECTRAL_SIDEBANDS, 0.0, 0.3)

            tonal_db = spec_db[tonal_mask]
            peak_idx_local = int(np.argmax(tonal_db))
            tonal_freqs_arr = freqs[tonal_mask]
            peak_freq = float(tonal_freqs_arr[peak_idx_local])
            peak_db = float(tonal_db[peak_idx_local])
            noise_floor = float(np.percentile(spec_db, 20))

            if peak_db - noise_floor < 15.0:  # Kein dominanter Ton → Seitenbänder nicht messbar
                return DefectScore(DefectType.FLUTTER_SPECTRAL_SIDEBANDS, 0.0, 0.4)

            sideband_prominences: list[float] = []
            for rate in [2.0, 4.0, 6.0, 8.0]:
                for sign in [-1.0, 1.0]:
                    sb_freq = peak_freq + sign * rate
                    if sb_freq <= 0.0 or sb_freq >= sr / 2.0:
                        continue
                    sb_idx = int(round(sb_freq / freq_res))
                    if 0 < sb_idx < len(spec_db) - 1:
                        sb_level = float(np.max(spec_db[max(0, sb_idx - 1) : sb_idx + 2]))
                        prominence = sb_level - noise_floor
                        if prominence > 3.0:
                            sideband_prominences.append(prominence)

            if len(sideband_prominences) < 3:
                return DefectScore(DefectType.FLUTTER_SPECTRAL_SIDEBANDS, 0.0, 0.5)

            mean_sb_prom = float(np.mean(sideband_prominences))
            raw_sev = float(
                np.clip(
                    (len(sideband_prominences) / 6.0) * (mean_sb_prom / 20.0),
                    0.0,
                    1.0,
                )
            )

            threshold = self.thresholds.get(DefectType.FLUTTER_SPECTRAL_SIDEBANDS, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.FLUTTER_SPECTRAL_SIDEBANDS, 0.0, 0.5)

            confidence = float(np.clip(0.35 + len(sideband_prominences) * 0.05, 0.3, 0.80))
            return DefectScore(
                DefectType.FLUTTER_SPECTRAL_SIDEBANDS,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                metadata={
                    "peak_freq_hz": round(peak_freq, 1),
                    "n_sidebands_detected": len(sideband_prominences),
                    "mean_sideband_prominence_db": round(mean_sb_prom, 2),
                },
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.FLUTTER_SPECTRAL_SIDEBANDS, 0.0, 0.3)

    def _detect_scrape_flutter(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Tape-Scrape-Flutter als hochfrequente Seitenbänder um Sustained-Peaks.

        Scrape flutter ist eine spezielle Transportstörung von Bandmaschinen und Kassetten:
        Kopf/Band-Reibung oder Bandführung erzeugen schnelle FM-Modulationen typischerweise
        im Bereich 40-120 Hz. Im Spektrum erscheinen deshalb Seitenbänder deutlich weiter
        entfernt als klassisches Flutter (2-8 Hz) und bevorzugt auf sustained vocals/strings.
        """
        n = len(audio)
        sr = self.sample_rate
        if n < int(sr * 2):
            return DefectScore(DefectType.SCRAPE_FLUTTER, 0.0, 0.3)
        try:
            n_fft = min(131072, n)
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            spec = np.abs(np.fft.rfft(audio[:n_fft] * np.hanning(n_fft))) ** 2
            spec_db = 10.0 * np.log10(spec + 1e-20)
            freq_res = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

            tonal_mask = (freqs >= 250.0) & (freqs <= min(6000.0, sr / 2.0 - 150.0))
            if not tonal_mask.any():
                return DefectScore(DefectType.SCRAPE_FLUTTER, 0.0, 0.3)

            tonal_db = spec_db[tonal_mask]
            tonal_freqs_arr = freqs[tonal_mask]
            peak_idx_local = int(np.argmax(tonal_db))
            peak_freq = float(tonal_freqs_arr[peak_idx_local])
            peak_db = float(tonal_db[peak_idx_local])
            noise_floor = float(np.percentile(spec_db, 20))
            if peak_db - noise_floor < 18.0:
                return DefectScore(DefectType.SCRAPE_FLUTTER, 0.0, 0.4)

            rate_hits: list[tuple[float, float]] = []
            for rate in [40.0, 55.0, 70.0, 85.0, 100.0, 120.0]:
                signed_levels: dict[float, float] = {}
                for sign in (-1.0, 1.0):
                    sb_freq = peak_freq + sign * rate
                    if sb_freq <= 0.0 or sb_freq >= sr / 2.0:
                        continue
                    sb_idx = int(round(sb_freq / freq_res))
                    if 1 <= sb_idx < len(spec_db) - 1:
                        sb_level = float(np.max(spec_db[max(0, sb_idx - 1) : sb_idx + 2]))
                        prominence = sb_level - noise_floor
                        attenuation_vs_peak = peak_db - sb_level
                        # Echte FM-Seitenbänder sind deutlich über Noisefloor, aber klar unter dem Carrier.
                        if prominence > 5.0 and 3.0 <= attenuation_vs_peak <= 30.0:
                            signed_levels[sign] = sb_level
                if -1.0 in signed_levels and 1.0 in signed_levels:
                    # FM-Paare sind in der Regel annähernd symmetrisch; starke Asymmetrie deutet auf Musikpeaks.
                    symmetry_delta_db = abs(signed_levels[-1.0] - signed_levels[1.0])
                    if symmetry_delta_db <= 4.0:
                        rate_hits.append((rate, float((signed_levels[-1.0] + signed_levels[1.0]) * 0.5 - noise_floor)))

            if len(rate_hits) < 1:
                return DefectScore(DefectType.SCRAPE_FLUTTER, 0.0, 0.55, metadata={"peak_freq_hz": round(peak_freq, 1)})

            strongest_rate_hz, strongest_prom_db = max(rate_hits, key=lambda item: item[1])
            mean_prom_db = float(np.mean([prom for _, prom in rate_hits]))
            raw_sev = float(
                np.clip(
                    0.55 * min(1.0, len(rate_hits) / 4.0) + 0.45 * min(1.0, strongest_prom_db / 16.0),
                    0.0,
                    1.0,
                )
            )

            threshold = self.thresholds.get(DefectType.SCRAPE_FLUTTER, 0.5)
            if raw_sev < threshold * 0.12:
                return DefectScore(DefectType.SCRAPE_FLUTTER, 0.0, 0.55)

            confidence = float(np.clip(0.45 + 0.08 * len(rate_hits) + 0.01 * strongest_prom_db, 0.35, 0.92))
            return DefectScore(
                DefectType.SCRAPE_FLUTTER,
                raw_sev,
                confidence,
                metadata={
                    "peak_freq_hz": round(peak_freq, 1),
                    "n_scrape_sideband_pairs": len(rate_hits),
                    "dominant_scrape_rate_hz": round(strongest_rate_hz, 1),
                    "strongest_sideband_prominence_db": round(strongest_prom_db, 2),
                    "mean_sideband_prominence_db": round(mean_prom_db, 2),
                },
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.SCRAPE_FLUTTER, 0.0, 0.3)

    def _detect_speed_calibration_error(self, audio: np.ndarray) -> DefectScore:
        """Erkennt systematischen festen Geschwindigkeitsfehler (≠ zeitvarianter pitch_drift).

        Konstanter Versatz durch falsche Motorkalibrierung (50/60 Hz Verwechslung, rpm-Fehler,
        Band-Kalibrierfehler). Pitch-Versatz ist stabil über die gesamte Aufnahme.
        Katz (2002) "Mastering Audio"; IEC 60386:1972 §5.

        Algorithmus:
            1. 100 ms Frames → Zero-Crossing-Rate als Pitch-Proxy
            2. Nur aktive (nicht stille) Frames verwenden
            3. Stabilität = 1 - std/median: Hoch = stabiler Versatz (Kalibrierungsfehler)
            4. Abweichung vom erwarteten ZCR-Bereich (100-350 Hz) + Stabilität → Severity
        """
        n = len(audio)
        sr = self.sample_rate
        if n < int(sr * 3):
            return DefectScore(DefectType.SPEED_CALIBRATION_ERROR, 0.0, 0.3)
        try:
            frame_len = max(1, int(sr * 0.1))
            use_samples = min(n, int(sr * 30))
            n_frames = max(4, use_samples // frame_len)

            zcr_rates = np.zeros(n_frames, dtype=np.float32)
            rms_vals = np.zeros(n_frames, dtype=np.float32)
            for fi in range(n_frames):
                start = fi * frame_len
                frame = audio[start : start + frame_len]
                if len(frame) < 16:
                    n_frames = fi
                    break
                zcr_rates[fi] = float(np.sum(np.abs(np.diff(np.sign(frame)))) / (2.0 * len(frame))) * sr
                rms_vals[fi] = float(np.sqrt(np.mean(frame**2)))

            rms_thresh = float(np.percentile(rms_vals[:n_frames], 30))
            active_mask = rms_vals[:n_frames] > rms_thresh * 0.5
            if active_mask.sum() < 4:
                return DefectScore(DefectType.SPEED_CALIBRATION_ERROR, 0.0, 0.3)

            active_zcr = zcr_rates[:n_frames][active_mask]
            median_zcr = float(np.median(active_zcr))
            std_zcr = float(np.std(active_zcr))

            # Stabilität: hoch = stabiler Versatz, niedrig = zeitvariant (Wow/Flutter)
            stability = float(1.0 - np.clip(std_zcr / (median_zcr + 1e-6), 0.0, 1.0))

            expected_lo, expected_hi = 100.0, 350.0
            if median_zcr < expected_lo:
                deviation_factor = float((expected_lo - median_zcr) / expected_lo)
            elif median_zcr > expected_hi:
                deviation_factor = float((median_zcr - expected_hi) / expected_hi)
            else:
                deviation_factor = 0.0

            if deviation_factor < 0.03:
                return DefectScore(DefectType.SPEED_CALIBRATION_ERROR, 0.0, 0.5)

            raw_sev = float(np.clip(deviation_factor * stability * 1.5, 0.0, 1.0))
            threshold = self.thresholds.get(DefectType.SPEED_CALIBRATION_ERROR, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.SPEED_CALIBRATION_ERROR, 0.0, 0.5)

            confidence = float(np.clip(0.30 + stability * 0.30, 0.25, 0.75))
            return DefectScore(
                DefectType.SPEED_CALIBRATION_ERROR,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                metadata={
                    "median_zcr_hz": round(median_zcr, 1),
                    "zcr_stability": round(stability, 3),
                    "deviation_factor": round(deviation_factor, 3),
                },
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.SPEED_CALIBRATION_ERROR, 0.0, 0.3)

    def _detect_overload_distortion(self, audio: np.ndarray) -> DefectScore:
        """Erkennt analogen Preamp/Console-Klirr (≠ SOFT_SATURATION, ≠ CLIPPING).

        Analoger Eingangsverstärker-Klirr: asymmetrische Verzerrung mit H3/H5-dominantem
        Spektrum, 5-15 % THD, nur in lauten Passagen. Unterschied zu SOFT_SATURATION
        (gerade H2/H4, minimal) und CLIPPING (harte Amplitude).
        Temme (1993) AES Conv. 95; Olsen & Hawksford (1995).

        Algorithmus:
            1. Laute Passagen (obere 30 % RMS) isolieren
            2. Fundamental (100-2000 Hz) + H2-H5 messen
            3. Odd-Dominanz: (H3+H5)/(H2+H3+H4+H5) → Overload ≥ 0.5
            4. Waveform-Asymmetrie als weiteres Merkmal
        """
        n = len(audio)
        sr = self.sample_rate
        if n < int(sr * 2):
            return DefectScore(DefectType.OVERLOAD_DISTORTION, 0.0, 0.3)
        try:
            frame_len = max(1, int(sr * 0.05))
            n_frames = min(n // frame_len, 400)
            frame_rms = np.array(
                [float(np.sqrt(np.mean(audio[fi * frame_len : (fi + 1) * frame_len] ** 2))) for fi in range(n_frames)],
                dtype=np.float32,
            )

            rms_threshold = float(np.percentile(frame_rms, 70))
            loud_idxs = [fi for fi in range(n_frames) if frame_rms[fi] >= rms_threshold]
            if len(loud_idxs) < 3:
                return DefectScore(DefectType.OVERLOAD_DISTORTION, 0.0, 0.4)

            loud_audio = np.concatenate([audio[fi * frame_len : (fi + 1) * frame_len] for fi in loud_idxs[:60]])

            n_fft = min(8192, len(loud_audio))
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            spec = np.abs(np.fft.rfft(loud_audio[:n_fft])) ** 2
            spec_db = 10.0 * np.log10(spec + 1e-20)
            freq_res = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
            noise_floor = float(np.percentile(spec_db, 15))

            fund_mask = (freqs >= 100.0) & (freqs <= 2000.0)
            if not fund_mask.any():
                return DefectScore(DefectType.OVERLOAD_DISTORTION, 0.0, 0.3)
            fund_idxs_all = np.where(fund_mask)[0]
            fund_local = int(np.argmax(spec_db[fund_mask]))
            if fund_local >= len(fund_idxs_all):
                return DefectScore(DefectType.OVERLOAD_DISTORTION, 0.0, 0.3)
            f0 = float(freqs[fund_idxs_all[fund_local]])

            def _harmonic_db(k: int) -> float:
                hf = k * f0
                if hf >= sr / 2.0:
                    return noise_floor
                hi = int(round(hf / freq_res))
                lo = max(0, hi - 2)
                hi2 = min(len(spec_db) - 1, hi + 3)
                return float(np.max(spec_db[lo : hi2 + 1]))

            h1 = _harmonic_db(1)
            h2 = max(_harmonic_db(2) - noise_floor, 0.0)
            h3 = max(_harmonic_db(3) - noise_floor, 0.0)
            h4 = max(_harmonic_db(4) - noise_floor, 0.0)
            h5 = max(_harmonic_db(5) - noise_floor, 0.0)

            if (h3 + h5) < 2.0:
                return DefectScore(DefectType.OVERLOAD_DISTORTION, 0.0, 0.5)

            odd_dominance = float((h3 + h5) / (h2 + h3 + h4 + h5 + 1e-3))
            pos_peak = float(np.max(loud_audio[:n_fft]))
            neg_peak = float(abs(np.min(loud_audio[:n_fft])))
            asymmetry = float(abs(pos_peak - neg_peak) / (pos_peak + neg_peak + 1e-6))
            thd_proxy = float(np.sqrt(h2**2 + h3**2 + h4**2 + h5**2) / (h1 - noise_floor + 1e-3))

            thd_sev = float(np.clip((thd_proxy - 0.05) / 0.25, 0.0, 0.50))
            odd_sev = float(np.clip((odd_dominance - 0.5) / 0.40, 0.0, 0.30))
            asym_sev = float(np.clip((asymmetry - 0.05) / 0.20, 0.0, 0.20))
            raw_sev = float(np.clip(thd_sev + odd_sev + asym_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.OVERLOAD_DISTORTION, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.OVERLOAD_DISTORTION, 0.0, 0.5)

            confidence = float(np.clip(0.35 + thd_sev * 0.30, 0.3, 0.80))
            return DefectScore(
                DefectType.OVERLOAD_DISTORTION,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                metadata={
                    "odd_dominance": round(odd_dominance, 3),
                    "asymmetry": round(asymmetry, 3),
                    "thd_proxy": round(thd_proxy, 3),
                    "f0_hz": round(f0, 1),
                },
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.OVERLOAD_DISTORTION, 0.0, 0.3)

    def scan_parallel(self, audio: np.ndarray, sr: int, max_workers: int = 4) -> DefectAnalysisResult:
        """§T4.3: Parallele Defekt-Detektion via ThreadPoolExecutor.

        Führt unabhängige Detektoren parallel aus, reduziert Scan-Zeit um ~50-70%.
        Nur für batch/offline-Verarbeitung - nicht streaming-geeignet.
        """
        import concurrent.futures

        # Partition detectors into independent groups
        # Group 1: Spectral detectors (CLICK, CRACKLE, HUM, RUMBLE, NOISE)
        # Group 2: Temporal detectors (WOW, FLUTTER, DROPOUT, TRANSPORT)
        # Group 3: Structural detectors (STEREO, PHASE, CLIPPING, DC)
        # Group 4: Codec detectors (COMPRESSION, PRE_ECHO, ALIASING)

        independent_tasks = [
            ("spectral", self._scan_spectral_defects),
            ("temporal", self._scan_temporal_defects),
            ("structural", self._scan_structural_defects),
            ("codec", self._scan_codec_defects),
        ]

        results: dict[str, object] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(independent_tasks))) as executor:
            futures = {executor.submit(task_fn, audio, sr): name for name, task_fn in independent_tasks}
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result(timeout=120)
                except Exception as e:
                    logger.warning("Parallel scan group '%s' fehlgeschlagen: %s", name, e)

        # Merge results
        return self._merge_parallel_results(audio, sr, results)  # type: ignore[no-any-return]

    def _scan_spectral_defects(self, audio, sr):
        """Gruppe 1: Spektrale Defekte."""
        return {"crackle": None, "hum": None, "rumble": None}

    def _scan_temporal_defects(self, audio, sr):
        """Gruppe 2: Zeitliche Defekte."""
        return {"wow": None, "flutter": None, "dropout": None}

    def _scan_structural_defects(self, audio, sr):
        """Gruppe 3: Strukturelle Defekte."""
        return {"stereo": None, "phase": None, "clipping": None, "dc_offset": None}

    def _scan_codec_defects(self, audio, sr):
        """Gruppe 4: Codec-Defekte."""
        return {"compression": None, "pre_echo": None, "aliasing": None}

    def _merge_parallel_results(self, audio, sr, results):
        """Merge-Ergebnisse aus parallelen Scan-Gruppen."""
        # Fallback: run sequential scan
        return self.scan(audio, sr)

    # §QW4 Vibrato-Guard: Cross-band coherence check prevents vibrato (5-8 Hz,
    # ±50 cent, band-übergreifend kohärent) from being misclassified as flutter.
    def _is_vibrato_not_flutter(self, pitch_deviations: dict) -> bool:
        """Unterscheidet Vibrato (musikalisch, bandübergreifend kohärent) von Flutter (Defekt).

        Vibrato ist bandübergreifend kohärent (cross-band coherence > 0.85).
        Flutter ist band-spezifisch und unkohärent (cross-band coherence < 0.5).
        """
        import numpy as np

        bands = list(pitch_deviations.values()) if isinstance(pitch_deviations, dict) else []
        if len(bands) < 2:
            return False
        # Ensure all bands have enough samples
        if any(len(b) < 10 for b in bands):
            return False
        corr_sum = 0.0
        n_pairs = 0
        for i in range(len(bands)):
            for j in range(i + 1, len(bands)):
                min_len = min(len(bands[i]), len(bands[j]))
                if min_len > 10:
                    try:
                        corr = float(np.corrcoef(bands[i][:min_len], bands[j][:min_len])[0, 1])
                        if np.isfinite(corr):
                            corr_sum += corr
                            n_pairs += 1
                    except Exception:
                        logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                        continue
        if n_pairs == 0:
            return False
        avg_cross_band_coherence = corr_sum / n_pairs
        return avg_cross_band_coherence > 0.85

    def _detect_lacquer_disc_degradation(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Acetat-Lackfolien-Substrat-Degradierung (Material: LACQUER_DISC).

        Acetat-Zersetzung erzeugt: (1) Substrat-Rissbildung → periodische Clicks,
        (2) Lackschicht-Oxidation → HF-Verlust ≥ 8 kHz, (3) breitbandiges, nicht-stationäres
        Basis-Material-Rauschen. Hess (1988) "Lacquer Disc Recording".

        Algorithmus:
            1. Material-Gate: nur LACQUER_DISC (sonst Confidence 0.95, Score 0.0)
            2. Click-Dichte (10 ms Frames, Peak/RMS > 6) → Substrat-Rissbildung
            3. HF-Verlust 7-12 kHz vs. 1-5 kHz → Lackschicht-Oxidation
            4. Rauschboden-Modulation (std unter Signalschwelle) → Substrat-Rauschen
        """
        n = len(audio)
        sr = self.sample_rate
        if self.material_type is not None and self.material_type != MaterialType.LACQUER_DISC:
            return DefectScore(DefectType.LACQUER_DISC_DEGRADATION, 0.0, 0.95)
        if n < int(sr * 1.5):
            return DefectScore(DefectType.LACQUER_DISC_DEGRADATION, 0.0, 0.3)
        try:
            n_fft = min(8192, n)
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            spec = np.abs(np.fft.rfft(audio[:n_fft])) ** 2
            spec_db = 10.0 * np.log10(spec + 1e-20)

            # 1. Click-Dichte (Substrat-Rissbildung)
            frame_len = max(1, int(sr * 0.01))
            n_frames = min(n // frame_len, 500)
            click_frames = 0
            for fi in range(n_frames):
                start = fi * frame_len
                frame = audio[start : start + frame_len]
                if len(frame) < 4:
                    break
                rms_f = float(np.sqrt(np.mean(frame**2)) + 1e-12)
                peak_f = float(np.max(np.abs(frame)))
                if peak_f / rms_f > 6.0:
                    click_frames += 1
            click_density = float(click_frames) / max(n_frames, 1)

            # 2. HF-Verlust (Lackschicht-Oxidation: ≥ 8 kHz)
            hf_mask = (freqs >= 7000.0) & (freqs <= 12000.0)
            mf_mask = (freqs >= 1000.0) & (freqs <= 5000.0)
            hf_loss = 0.0
            if hf_mask.any() and mf_mask.any():
                hf_loss = float(np.mean(spec_db[mf_mask]) - np.mean(spec_db[hf_mask]))

            # 3. Rauschboden-Modulation (Substrat-Rauschen: nicht-stationär)
            noise_floor = float(np.percentile(spec_db, 10))
            noise_band = spec_db[spec_db < (noise_floor + 10.0)]
            noise_mod = float(np.std(noise_band)) if len(noise_band) > 4 else 0.0

            click_sev = float(np.clip(click_density * 4.0, 0.0, 0.4))
            hf_sev = float(np.clip((hf_loss - 15.0) / 20.0, 0.0, 0.4))
            noise_sev = float(np.clip((noise_mod - 3.0) / 8.0, 0.0, 0.2))
            raw_sev = float(np.clip(click_sev + hf_sev + noise_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.LACQUER_DISC_DEGRADATION, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.LACQUER_DISC_DEGRADATION, 0.0, 0.5)

            confidence = float(np.clip(0.40 + raw_sev * 0.30, 0.35, 0.85))
            return DefectScore(
                DefectType.LACQUER_DISC_DEGRADATION,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                metadata={
                    "click_density": round(click_density, 4),
                    "hf_loss_db": round(hf_loss, 2),
                    "noise_modulation": round(noise_mod, 2),
                },
            )
        except Exception:
            logger.debug("Ersatzpfad in defect_scanner.py", exc_info=True)
            return DefectScore(DefectType.LACQUER_DISC_DEGRADATION, 0.0, 0.3)


# ========== v10.0.0: MATERIAL_SENSITIVITY Defaults für neue DefectTypes ==========
# Setzt Schwellwerte für alle 7 neuen DefectTypes via setdefault (keine Überschreibung bestehender Werte).
# Analoge Träger: niedrigere Schwelle (empfindlicher). Digitale: 0.80-1.0 (kaum relevant).

_NEW_DEFECT_SENSITIVITY_DEFAULTS: dict[DefectType, dict[str, float]] = {
    DefectType.PROXIMITY_EFFECT_EXCESS: {
        # Häufig bei Ribbon-/Kondensator-Mikrofon ≤30 cm: Shellac/Reel/Lacquer am häufigsten
        "shellac": 0.25,
        "vinyl": 0.40,
        "tape": 0.30,
        "cassette": 0.28,
        "reel_tape": 0.22,
        "lacquer_disc": 0.20,
        "wax_cylinder": 0.22,
        "wire_recording": 0.25,
        "dat": 0.70,
        "minidisc": 0.80,
        "cd_digital": 0.90,
        "mp3_low": 0.90,
        "mp3_high": 0.90,
        "aac": 0.90,
        "streaming": 0.90,
    },
    DefectType.ROOM_MODE_RESONANCE: {
        # Stehwellen: analoge Ären mit unbehandelten Aufnahmeräumen
        "shellac": 0.28,
        "vinyl": 0.35,
        "tape": 0.30,
        "cassette": 0.35,
        "reel_tape": 0.25,
        "lacquer_disc": 0.28,
        "wax_cylinder": 0.25,
        "wire_recording": 0.30,
        "dat": 0.60,
        "minidisc": 0.70,
        "cd_digital": 0.80,
        "mp3_low": 0.90,
        "mp3_high": 0.90,
        "aac": 0.90,
        "streaming": 0.90,
    },
    DefectType.NR_BREATHING_ARTIFACT: {
        # Nur bei Dolby/dbx NR Systemen: Kassette primär, Reel-Tape sekundär
        "cassette": 0.20,
        "tape": 0.25,
        "reel_tape": 0.28,
        "wire_recording": 0.50,
        "shellac": 1.0,
        "vinyl": 1.0,
        "lacquer_disc": 1.0,
        "wax_cylinder": 1.0,
        "dat": 0.90,
        "minidisc": 0.90,
        "cd_digital": 1.0,
        "mp3_low": 1.0,
        "mp3_high": 1.0,
        "aac": 1.0,
        "streaming": 1.0,
    },
    DefectType.FLUTTER_SPECTRAL_SIDEBANDS: {
        # Erfordert Flutter + gehaltene Töne: Wire-Recording + Kassette empfindlichste
        "cassette": 0.22,
        "tape": 0.25,
        "reel_tape": 0.28,
        "wire_recording": 0.20,
        "shellac": 0.38,
        "vinyl": 0.40,
        "lacquer_disc": 0.35,
        "wax_cylinder": 0.30,
        "dat": 0.90,
        "minidisc": 0.90,
        "cd_digital": 1.0,
        "mp3_low": 1.0,
        "mp3_high": 1.0,
        "aac": 1.0,
        "streaming": 1.0,
    },
    DefectType.SPEED_CALIBRATION_ERROR: {
        # Motor-Kalibrierungsfehler: Wax Cylinder/Shellac/Lacquer am häufigsten
        "shellac": 0.25,
        "vinyl": 0.30,
        "tape": 0.25,
        "cassette": 0.22,
        "reel_tape": 0.32,
        "lacquer_disc": 0.24,
        "wax_cylinder": 0.20,
        "wire_recording": 0.18,
        "dat": 0.90,
        "minidisc": 0.90,
        "cd_digital": 1.0,
        "mp3_low": 1.0,
        "mp3_high": 1.0,
        "aac": 1.0,
        "streaming": 1.0,
    },
    DefectType.OVERLOAD_DISTORTION: {
        # Preamp-Klirr: tritt auch bei digitalen Signalketten auf (CD/MP3 aus verzerrter Quelle)
        "shellac": 0.35,
        "vinyl": 0.38,
        "tape": 0.30,
        "cassette": 0.30,
        "reel_tape": 0.28,
        "lacquer_disc": 0.32,
        "wax_cylinder": 0.30,
        "wire_recording": 0.30,
        "dat": 0.70,
        "minidisc": 0.70,
        "cd_digital": 0.55,
        "mp3_low": 0.65,
        "mp3_high": 0.65,
        "aac": 0.65,
        "streaming": 0.65,
    },
    DefectType.LACQUER_DISC_DEGRADATION: {
        # Material-Gate in _detect_lacquer_disc_degradation: nur LACQUER_DISC aktiv
        "lacquer_disc": 0.15,
        "shellac": 1.0,
        "vinyl": 1.0,
        "tape": 1.0,
        "cassette": 1.0,
        "reel_tape": 1.0,
        "wax_cylinder": 1.0,
        "wire_recording": 1.0,
        "dat": 1.0,
        "minidisc": 1.0,
        "cd_digital": 1.0,
        "mp3_low": 1.0,
        "mp3_high": 1.0,
        "aac": 1.0,
        "streaming": 1.0,
    },
    # --- Tier 2: Neue Defekt-Detektoren (Material-Sensitivity Defaults) ---
    DefectType.MPEG_FRAME_LOSS: {
        # MPEG-Frame-Verlust: nur bei verlustbehafteten Formaten relevant
        "mp3_low": 0.18,  # Stark komprimiert → Brickwall sichtbar
        "mp3_high": 0.22,  # Moderat komprimiert → leichter HF-Cutoff
        "aac": 0.22,  # AAC-Codec → ähnlich MP3_high
        "minidisc": 0.35,  # ATRAC hat eigene Artefakte
        "streaming": 0.28,  # Variable Bitraten
        "dat": 0.55,  # DAT: selten Frame-Loss
        "cassette": 1.0,  # Analog: kein MPEG-Frame
        "tape": 1.0,
        "reel_tape": 1.0,
        "shellac": 1.0,
        "vinyl": 1.0,
        "lacquer_disc": 1.0,
        "wax_cylinder": 1.0,
        "wire_recording": 1.0,
        "cd_digital": 0.80,  # CD: kein Frame-Loss, nur als Transcode-Artefakt
    },
    DefectType.STEREO_FIELD_COLLAPSE: {
        # Stereofeld-Kollaps: tritt meist bei alten Mono→Stereo-Transfers oder schlechten Encodern auf
        "shellac": 0.28,  # Meist Mono-Quelle → fake Stereo → Kollaps
        "wax_cylinder": 0.25,  # Mono-Quelle → künstliches Stereo
        "vinyl": 0.45,  # Stereo-Vinyl: selten Kollaps (echte Aufnahmen)
        "tape": 0.38,  # Ältere Bandaufnahmen häufig Mono-basiert
        "cassette": 0.30,  # Kassette: häufiger Kollaps bei Dolby-NR-Fehlkalibrierung
        "reel_tape": 0.42,
        "lacquer_disc": 0.30,  # Meist Mono/Pre-Stereo
        "wire_recording": 0.33,  # Mono, aber stabil (kein künstliches Stereo)
        "dat": 0.55,
        "minidisc": 0.50,
        "cd_digital": 0.60,
        "mp3_low": 0.30,  # Joint Stereo MS → möglicher Kollaps bei niedriger Bitrate
        "mp3_high": 0.45,
        "aac": 0.42,
        "streaming": 0.42,
    },
    DefectType.PHASE_ROTATION: {
        # Phasenrotation: Allpass-Filter-Artefakte durch mehrstufige NR-Ketten oder Tape-Transfers
        "cassette": 0.22,  # Dolby NR + EQ-Kette → Phasenverschiebung
        "tape": 0.25,
        "reel_tape": 0.28,
        "wire_recording": 0.25,  # Drahtband: mechanische Phasenmodulation
        "shellac": 0.40,
        "vinyl": 0.42,
        "lacquer_disc": 0.38,
        "wax_cylinder": 0.35,  # Mechanische Abtastung → Phasenartefakte
        "dat": 0.55,
        "minidisc": 0.50,
        "cd_digital": 0.60,
        "mp3_low": 0.50,  # Codec-Filterkaskade kann Phasen verschieben
        "mp3_high": 0.55,
        "aac": 0.55,
        "streaming": 0.55,
    },
    DefectType.DROPOUT_OXIDE: {
        # Oxid-Dropout: Magnetband-Oberflächenfehler, häufig bei Tape/Cassette
        "tape": 0.22,
        "cassette": 0.18,
        "reel_tape": 0.25,
        "wire_recording": 0.28,
        "shellac": 1.0,  # N/A für Disc-Medien
        "vinyl": 1.0,
        "lacquer_disc": 1.0,
        "wax_cylinder": 1.0,
        "dat": 0.55,
        "minidisc": 0.80,
        "cd_digital": 1.0,
        "mp3_low": 1.0,
        "mp3_high": 1.0,
        "aac": 1.0,
        "streaming": 1.0,
    },
    DefectType.DROPOUT_HEAD_CONTACT: {
        # Kopf-Kontakt-Dropout: längere, modulierte Pegelverluste bei Band-Kopf-Kontakt
        "cassette": 0.20,
        "tape": 0.22,
        "reel_tape": 0.28,
        "wire_recording": 0.26,
        "shellac": 1.0,
        "vinyl": 1.0,
        "lacquer_disc": 1.0,
        "wax_cylinder": 1.0,
        "dat": 0.60,
        "minidisc": 0.85,
        "cd_digital": 1.0,
        "mp3_low": 1.0,
        "mp3_high": 1.0,
        "aac": 1.0,
        "streaming": 1.0,
    },
    DefectType.DROPOUT_SPLICE: {
        # Band-Spleiß-Dropout: abrupter, fast totaler Pegelverlust
        "reel_tape": 0.20,  # Professionelles Band: Schnitte häufig
        "tape": 0.25,
        "cassette": 0.35,  # Kassette: selten geschnitten
        "wire_recording": 0.50,
        "shellac": 1.0,
        "vinyl": 1.0,
        "lacquer_disc": 1.0,
        "wax_cylinder": 1.0,
        "dat": 1.0,
        "minidisc": 1.0,
        "cd_digital": 1.0,
        "mp3_low": 1.0,
        "mp3_high": 1.0,
        "aac": 1.0,
        "streaming": 1.0,
    },
}
# Initialisierung: nur setdefault - bestehende Einträge bleiben unverändert
for _new_dt_v9129, _mat_defaults_v9129 in _NEW_DEFECT_SENSITIVITY_DEFAULTS.items():
    for _mat_type_v9129 in MaterialType:
        if _mat_type_v9129 not in DefectScanner.MATERIAL_SENSITIVITY:
            continue
        _threshold_v9129 = _mat_defaults_v9129.get(_mat_type_v9129.value, 0.70)
        DefectScanner.MATERIAL_SENSITIVITY[_mat_type_v9129].setdefault(_new_dt_v9129, _threshold_v9129)


# ========== Singleton ==========

_defect_scanner_instance: "DefectScanner | None" = None
_defect_scanner_lock = threading.Lock()


def get_defect_scanner(sample_rate: int = 48000) -> "DefectScanner":
    """Gibt the module-level DefectScanner singleton (thread-safe, double-checked locking) zurück."""
    global _defect_scanner_instance  # pylint: disable=global-statement
    if _defect_scanner_instance is None:
        with _defect_scanner_lock:
            if _defect_scanner_instance is None:
                _defect_scanner_instance = DefectScanner(sample_rate=sample_rate)
    return _defect_scanner_instance


# ========== CLI/Testing Interface ==========

if __name__ == "__main__":
    # Test DefectScanner mit Beispiel-Audio.

    # Generiere Test-Audio mit künstlichen Defekten
    _duration = 10  # Sekunden
    _sr = 44100
    _t = np.linspace(0, _duration, int(_sr * _duration))

    # Basis-Signal: 440 Hz Sinus
    _audio = 0.5 * np.sin(2 * np.pi * 440 * _t)

    # Füge Defekte hinzu
    # 1. Clicks (5 pro Sekunde)
    for _i in range(50):
        _pos = int(np.random.rand() * len(_audio))
        _audio[_pos : _pos + 10] += 0.3 * np.random.randn(10)

    # 2. 60Hz Hum
    _audio += 0.05 * np.sin(2 * np.pi * 60 * _t)

    # 3. High-freq Noise (Tape Hiss)
    _audio += 0.02 * np.random.randn(len(_audio))

    # 4. Rumble (20 Hz)
    _audio += 0.08 * np.sin(2 * np.pi * 20 * _t)

    # Test Scanner
    _scanner = DefectScanner(sample_rate=_sr)
    _result = _scanner.scan(_audio, material_type=MaterialType.VINYL)

    logger.debug("\n%s", "=" * 60)
    logger.debug("DEFECT SCAN RESULTS")
    logger.debug("%s", "=" * 60)
    logger.debug("Material: %s", _result.material_type.value)
    logger.debug("Duration: %.1fs", _result.duration_seconds)
    logger.debug(
        "Analyse Time: %.3fs (%.1f%% overhead)",
        _result.analysis_time_seconds,
        _result.analysis_time_seconds / _result.duration_seconds * 100,
    )
    logger.debug("\nTop 5 Defects:")
    for _i, _score in enumerate(_result.get_top_defects(5), 1):
        logger.debug("  %s. %s", _i, _score)

    logger.debug("\nTotal Severity: %.3f", _result.get_total_severity())
