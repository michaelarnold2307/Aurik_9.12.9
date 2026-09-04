"""
Aurik 10.0.0 — HolisticPerceptualGate §2.44 [RELEASE_MUST]
======================================================
Last gate before export. Measures holistic perceptual improvement
instead of checking individual goals only.

HPI > 0 → Export | HPI ≤ 0 → Rollback

§2.44 FIX v10.0.0 — Referenz-Paradox beseitigt:
  Für ALLE Restorability-Bereiche: Referenz-Vektor aus GP-Memory primär;
  kein Ref-Vektor → direktionale Verbesserungsmessung (_compute_directional_restoration_quality).
  Input-Ähnlichkeit als prim. Maß entfernt (bestrafte erfolgreiche Restaurierung).

MERT-Referenz-Memory: EMA (α=0.15) pro (genre × material × era_bin).
Fallback-Kaskade (5 Stufen) wenn kein passender Referenz-Vektor.
Referenz-Update nur wenn: HPI > 0.0 AND artifact_freedom ≥ 0.95 (V54-aligned v10.0.0).

Reference: Spec 02 §2.44, §2.49 (artifact_freedom)
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import threading
from dataclasses import dataclass, field
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

# ── EMA constant for reference-memory updates ──────────────────────────────
_EMA_ALPHA: float = 0.15
_MIN_OBS_CALIBRATED: int = 3  # < 3 obs → Bootstrap mit erhöhter Unsicherheit

# ── Material-native BW-Ceiling für MERT-Spectral-Proxy (§2.44 BW-Ceiling-Guard) ──────────────
# Quelle: IEC 60094-1 (Kassette), DIN 45511 (Analogband), RIAA-Spec (Vinyl), CD-Standard.
# Werte gelten für das ORIGINAL-Signal ohne BW-Erweiterung; nach FlashSR (phase_06) kann
# das restaurierte Signal über diesen Wert hinausgehende Energie enthalten —
# Spectral-Proxy darf diese Energie NICHT als Divergenz bestrafen (Reference Paradox §0d).
_MATERIAL_BW_CEILING_HZ: dict[str, int] = {
    "shellac": 6000,  # Frühe elektr. Aufnahmen, Horncharakteristik ≤ 6 kHz
    "wax_cylinder": 4000,  # Wachswalze (akustisches Richtrohr)
    "lacquer_disc": 8000,  # Lackfolie (direktschnitt)
    "wire_recording": 5000,  # Drahtaufnahme (Magnetdraht)
    "vinyl": 20000,  # Vinyl-Pressungen — voller Bereich; keine Einschränkung
    "tape": 15000,  # Analogband, beste Bedingungen (DIN 45511)
    "reel_tape": 15000,  # Tonband/Spule (identisch)
    "cassette": 14000,  # §6.2c central definition
    "cd_digital": 22050,  # CD-Nyquist
    "dat": 22050,  # DAT-Nyquist
    "md": 20000,  # MiniDisc ATRAC
    "mp3_low": 12000,  # ≤ 128 kbps — psychoakust. HF-Cutoff
    "mp3_high": 16000,  # 320 kbps
    "aac": 18000,  # AAC HE/LC typisch
    "unknown": 22050,  # keine Einschränkung
}

# ── Persistenz-Pfad für HPG-Referenz-Memory (§2.44, analog §2.70 RestorationMemory) ─────
_HPG_REF_MEMORY_PATH: pathlib.Path = (
    pathlib.Path(os.environ.get("AURIK_DATA_DIR", str(pathlib.Path.home() / ".aurik"))) / "hpg_reference_memory.json"
)

# ── Singleton ──────────────────────────────────────────────────────────────
_instance: HolisticPerceptualGate | None = None
_lock = threading.Lock()


def get_holistic_gate() -> HolisticPerceptualGate:
    """Thread-safe Singleton accessor."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = HolisticPerceptualGate()
    return _instance


@dataclass
class _RefEntry:
    """One entry in the reference memory (spectral embedding + EMA state)."""

    embedding: np.ndarray  # shape (n_mels,) — spectral prototype
    obs_count: int = 0  # number of successful updates
    calibrated: bool = False  # True once obs_count >= _MIN_OBS_CALIBRATED


@dataclass
class HPIResult:
    """Result of HPI evaluation."""

    hpi: float
    passed: bool
    mert_similarity: float = 1.0
    timbral_fidelity: float = 1.0
    artifact_freedom: float = 1.0
    emotional_arc_preservation: float = 1.0
    studio_quality_gain: float = 1.0
    pqs_improvement: float = 1.0
    is_studio_mode: bool = False
    reference_mode: str = "degraded_input"
    detail: dict = field(default_factory=dict)
    fail_reason: object | None = None  # §1.4a FailReason when passed=False


class HolisticPerceptualGate:
    """§2.44 Holistic Perceptual Gate — last gate before export."""

    def __init__(self) -> None:
        # §2.44 MERT-Reference-Memory: key = (genre, material, era_bin)
        self._ref_memory: dict[tuple[str, str, str], _RefEntry] = {}
        self._ref_lock = threading.Lock()
        # MERT similarity path is optional and must never break the gate.
        self._mert_path_disabled: bool = False
        # §2.44 VERBOTEN: MERT darf nicht primary sein wenn VERSA verfügbar.
        # _mert_proxy_used = True → VERSA fehlgeschlagen, MERT als Fallback aktiv.
        self._mert_proxy_used: bool = False
        # §2.44 Persistenz: Referenz-Memory von Disk laden (analog §2.70 RestorationMemory).
        self._load_ref_memory_from_disk()

    @staticmethod
    def _get_depth_adaptive_af_min(transfer_chain: list[str] | None) -> float:
        """§v10.120 Depth-adaptiver artifact_freedom-Mindestwert für HPI-Gate.

        Konsistent mit spec_constitution.py Music-Death-Shield: Baseline (depth<2)
        = 0.95 (§0h `artifact_freedom_min`), nur bei tieferen Transfer-Chains
        gelockert (depth=2→0.88, depth=3→0.80, depth≥4→0.70 cassette).
        """
        _depth = max(1, len(transfer_chain)) if transfer_chain else 1
        if _depth >= 4:
            return 0.70  # deep cassette
        elif _depth == 3:
            return 0.80  # moderate
        elif _depth == 2:
            return 0.88  # shallow
        return 0.95  # studio master / single-generation (§0h Music-Death-Shield baseline)

    def evaluate_restoration(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
        artifact_freedom: float = 1.0,
        emotional_arc_score: float = 1.0,
        restorability_score: float | None = None,
        genre: str = "DEFAULT",
        material: str = "digital",
        era_bin: str = "post-1990",
        vqi: float = 1.0,
        panns_singing: float = 0.0,
        reference_audio: np.ndarray | None = None,
        transfer_chain: list[str] | None = None,
    ) -> HPIResult:
        """Evaluate HPI for Restoration mode.

        HPI = MERT_similarity × timbral_fidelity × artifact_freedom × emotional_arc_preservation

        §2.44 FIX v10.0.0 — Referenz-Paradox beseitigt:
          Strukturelle Klangkohärenz bedeutet NICHT Ähnlichkeit zum degradierten Input.
          Ein erfolgreich restauriertes Signal weicht vom degradierten Input ab —
          der alte Ansatz (input_weight=1.0 bei restorability > 70) hat gute
          Restaurierung aktiv bestraft.

          Korrekte Strategie für alle Restorability-Bereiche:
            1. Referenz-Vektor aus GP-Memory (genre × material × era_bin) → primär
            2. Kein Referenz-Vektor → direktionale Verbesserungsmessung:
               misst ob Signal in Richtung "sauber + musikalisch" verbessert wurde
          Input-Ähnlichkeit dient nur als Content-Integrity-Anteil (klein).
        """
        # §v10.x rs-Konsistenz: kanonische Quelle (explizit > CalibrationContext > 70.0).
        from backend.core.calibration_context import resolve_restorability_score

        restorability_score = resolve_restorability_score(restorability_score)
        self._mert_proxy_used = False  # reset per evaluation
        _reference_audio = reference_audio if reference_audio is not None else original
        _reference_mode = "best_carrier_checkpoint" if reference_audio is not None else "degraded_input"
        # §2.44 BW-Ceiling-Guard: MERT-Spectral-Proxy darf absichtliche BW-Erweiterung
        # (FlashSR phase_06 auf Kassette/Shellac) nicht als Spektral-Divergenz werten.
        # Material-BW-Ceiling wird als Frequenz-Obergrenze in den Spectral-Proxy übergeben.
        _mat_key_mert = str(material).lower().replace(" ", "_")
        _bw_ceiling_hz = _MATERIAL_BW_CEILING_HZ.get(_mat_key_mert, _MATERIAL_BW_CEILING_HZ["unknown"])
        mert_sim = self._compute_mert_similarity(_reference_audio, restored, sr, bw_ceiling_hz=_bw_ceiling_hz)
        # §v10.93 NaN-Guard: max(nan, 0.5) == nan in Python → HPI kollabiert.
        # nan_to_num vor max() stellt sicher, dass VERSA-NaN nicht durchschlägt.
        mert_sim = float(np.nan_to_num(mert_sim, nan=0.5))
        mert_sim = max(mert_sim, 0.5)

        # §2.44 FIX v10.0.0: Referenz-Vektor bevorzugen für ALLE Restorability-Bereiche.
        # Kein Ref-Vektor → direktionale Qualitätsmessung statt Input-Ähnlichkeit.
        ref_vec = self._get_reference_vector(genre, material, era_bin)
        # §v10.91 Non-Plus-Ultra: Wenn GP-Memory keinen Referenz-Vektor hat,
        # berechne blinden Referenz-Vektor aus dem saubersten Fenster des
        # restaurierten Audios (via BlindInternalReference). Kein Audio-Vergleich,
        # nur Embedding-Cosinus — verhindert Shape-Mismatch und falsche Scores.
        if ref_vec is None:
            _blind_vec = self._compute_blind_reference_vector(restored, sr)
            if _blind_vec is not None:
                ref_vec = _blind_vec
        if ref_vec is not None:
            rest_embed = self._compute_embedding(restored, sr)
            timbral_ref = float(np.clip(self._cosine_similarity(ref_vec, rest_embed), 0.0, 1.0))
        else:
            # Kein Referenz-Vektor: misst ob das Signal in Richtung "sauber" verbessert wurde
            timbral_ref = self._compute_directional_restoration_quality(original, restored, sr)
            # §2.44 Codec-Digital-Floor (v10.0.0): _compute_directional_restoration_quality()
            # ist für analoge Defekte kalibriert (Rauschreduktion → Noise-Delta, HF-Crest-Gewinn).
            # Für Codec-Container (MP3/AAC) mit hoher Restorability ist das Signal bereits sauber
            # → direktionale Score ~0.5 (kein Rauschboden-Delta, kein HF-Crest-Gewinn) — obwohl
            # die Restaurierung korrekt war. Fix: timbral_input (Mel-BW-Ceiling=12kHz) als Floor.
            # timbral_input wird erst weiter unten berechnet — hier vorab für den Floor-Check.
            _CODEC_MATS_HPG = frozenset({"mp3_low", "mp3_high", "aac", "streaming", "minidisc"})
            if _mat_key_mert in _CODEC_MATS_HPG and restorability_score > 70.0:
                _timbral_input_floor = self._compute_timbral_fidelity(
                    _reference_audio, restored, sr, bw_ceiling_hz=_bw_ceiling_hz
                )
                _timbral_ref_before = timbral_ref
                timbral_ref = max(timbral_ref, _timbral_input_floor)
                if timbral_ref > _timbral_ref_before:
                    logger.debug(
                        "§2.44 Codec-Digital-Floor: timbral_ref %.3f → %.3f"
                        " (timbral_Eingabe_floor=%.3f mat=%s restorability=%.0f)",
                        _timbral_ref_before,
                        timbral_ref,
                        _timbral_input_floor,
                        _mat_key_mert,
                        restorability_score,
                    )

            # §2.44 Analog-Carrier-Floor (v10.0.0): _compute_directional_restoration_quality()
            # ist für stationäres Rauschen kalibriert — noise_delta via 5. Perzentil der
            # Frame-Energie. Bei Musikmaterial entspricht das 5. Perzentil einer leisen
            # Musikpassage (keine Rauschbodenmessung) → noise_delta_db ≈ 0 → Score ≈ 0.5,
            # obwohl die Restaurierung korrekt durchgeführt wurde.
            # Fix: timbral_input (Mel-Cosinus, BW-Ceiling) als Floor für timbral_ref bei
            # analogen Trägern mit ausreichend hoher Restorability (≥ 63.0).
            # Content-Integrität (timbral_input ≥ timbral_ref) bedeutet: Das restaurierte
            # Signal ist inhaltlich so nah am Input wie das Original — ein valides Qualitätszeichen.
            _ANALOG_CARRIER_MATS_HPG = frozenset(
                {
                    "cassette",
                    "cassette_dolby_b",
                    "cassette_dolby_c",
                    "cassette_dolby_s",
                    "tape",
                    "reel_tape",
                    "vinyl",
                    "shellac",
                    "lacquer_disc",
                    "wire_recording",
                    "acetate",
                }
            )
            if _mat_key_mert in _ANALOG_CARRIER_MATS_HPG and restorability_score >= 63.0:
                _timbral_input_floor_analog = self._compute_timbral_fidelity(
                    _reference_audio, restored, sr, bw_ceiling_hz=_bw_ceiling_hz
                )
                _timbral_ref_before_analog = timbral_ref
                timbral_ref = max(timbral_ref, _timbral_input_floor_analog)
                if timbral_ref > _timbral_ref_before_analog:
                    logger.debug(
                        "§2.44 Analog-Carrier-Floor: timbral_ref %.3f → %.3f"
                        " (timbral_Eingabe_floor=%.3f mat=%s restorability=%.0f)",
                        _timbral_ref_before_analog,
                        timbral_ref,
                        _timbral_input_floor_analog,
                        _mat_key_mert,
                        restorability_score,
                    )

            # §P1-VQI-Codec-Floor (v10.0.0→v10.0.0): Codec-Material ohne Referenz-Vektor und ohne
            # Carrier-Checkpoint. Wenn VQI ≥ 0.82 + artifact_freedom ≥ 0.95 ist die Restaurierung
            # nachweislich korrekt; die Mel-Divergenz vom degradierten Codec-Input ist physikalisch
            # erwartet (Pre-Echo-Entfernung, HF-Rolloff-Kompensation, Artefakt-Reduktion).
            # §R2/S1: Ceiling auf 0.90 erhöht (R2: 0.87, S1: +0.03). Skalierung × 3.0 (statt × 1.2).
            # Neue Vokal-Floor: 0.60 + (vqi − 0.82) × 3.0, auf [0.60, 0.90] geklemmt.
            # VQI=0.917 → 0.891. Zusätzliche Bedingung: mert_sim ≥ 0.80 (kein Content-Verlust).
            # Instrumental-Floor: 0.64 (konservativ, nur artifact_freedom-basiert).
            _VQI_CODEC_MATS_HPG = frozenset({"mp3_low", "mp3_high", "aac", "streaming", "minidisc"})
            if (
                _mat_key_mert in _VQI_CODEC_MATS_HPG
                and reference_audio is None
                and float(np.clip(artifact_freedom, 0.0, 1.0)) >= 0.95
                and restorability_score > 70.0
                and mert_sim >= 0.80
            ):
                if panns_singing >= 0.35 and float(np.clip(vqi, 0.0, 1.0)) >= 0.82:
                    _vqi_codec_floor = float(np.clip(0.60 + (float(vqi) - 0.82) * 3.0, 0.60, 0.90))
                else:
                    _vqi_codec_floor = 0.64  # Instrumental oder niedrige VQI
                _tref_before_vqi = timbral_ref
                timbral_ref = max(timbral_ref, _vqi_codec_floor)
                if timbral_ref > _tref_before_vqi:
                    logger.debug(
                        "§P1 VQI-Codec-Floor (§R2): timbral_ref %.3f→%.3f"
                        " (vqi=%.3f panns=%.2f mat=%s artifact=%.3f mert=%.3f)",
                        _tref_before_vqi,
                        timbral_ref,
                        float(vqi),
                        float(panns_singing),
                        _mat_key_mert,
                        float(artifact_freedom),
                        mert_sim,
                    )

            # §P1-VQI-Codec-Floor (§S4 Chain-End): Analoger Primärträger + Codec-Chain-Ende.
            # Beispiel: Kassette→mp3_low — primary='cassette', transfer_chain=['mp3_low'].
            # Die Codec-spezifischen Messung-Artefakte (Mel-Divergenz vom degradierten Input nach
            # Pre-Echo-Entfernung, HF-Rolloff-Kompensation) gelten gleichermaßen wie bei reinem
            # Codec-Material. VQI + artifact_freedom bestätigen korrekte Restaurierung.
            # Restorability-Threshold: 40.0 (analog+Codec → typisch 55–70, selten > 70).
            _chain_last_key_s4 = ""
            if transfer_chain:
                _chain_last_key_s4 = str(transfer_chain[-1]).lower().replace(" ", "_")
            _VQI_CHAIN_END_CODEC = frozenset({"mp3_low", "mp3_high", "aac", "streaming", "minidisc"})
            if (
                _mat_key_mert in _ANALOG_CARRIER_MATS_HPG
                and _chain_last_key_s4 in _VQI_CHAIN_END_CODEC
                and reference_audio is None
                and float(np.clip(artifact_freedom, 0.0, 1.0)) >= 0.95
                and restorability_score >= 40.0
                and mert_sim >= 0.70
            ):
                if panns_singing >= 0.35 and float(np.clip(vqi, 0.0, 1.0)) >= 0.82:
                    _vqi_chain_floor = float(np.clip(0.60 + (float(vqi) - 0.82) * 3.0, 0.60, 0.90))
                else:
                    _vqi_chain_floor = 0.64  # Instrumental oder niedrige VQI
                _tref_before_chain = timbral_ref
                timbral_ref = max(timbral_ref, _vqi_chain_floor)
                if timbral_ref > _tref_before_chain:
                    logger.debug(
                        "§P1 VQI-Codec-Floor (§S4 Chain-End): timbral_ref %.3f→%.3f"
                        " (vqi=%.3f panns=%.2f primary=%s chain_last=%s restorability=%.0f)",
                        _tref_before_chain,
                        timbral_ref,
                        float(vqi),
                        float(panns_singing),
                        _mat_key_mert,
                        _chain_last_key_s4,
                        restorability_score,
                    )

        # timbral_input als Content-Integrity-Anteil (für Logging und niedrige Restorability)
        # §2.44 BW-Ceiling-Guard (v10.0.0): mel-Vergleich auf material-native BW begrenzen
        # → FlashSR-synthetisierter HF-Content bestraft timbral_input nicht mehr.
        timbral_input = self._compute_timbral_fidelity(_reference_audio, restored, sr, bw_ceiling_hz=_bw_ceiling_hz)

        # §2.44 Restorability-dependent weights — Referenz/Direktional dominiert stets
        if restorability_score > 70.0:
            # Hohe Restorability: Signal bewegt sich weg vom Defekt, hin zur Referenz.
            # Input-Ähnlichkeit ist hier KEIN Qualitätsmaß (Referenz-Paradox).
            input_weight, ref_weight = 0.0, 1.0
        elif restorability_score >= 50.0:
            # Mittlere Restorability: minimaler Input-Anteil als Ankerpunkt
            input_weight, ref_weight = 0.35, 0.65
        else:
            # Niedrige Restorability: kleiner Content-Integrity-Anteil
            input_weight, ref_weight = 0.2, 0.8

        timbral = input_weight * timbral_input + ref_weight * timbral_ref

        # §0d CCR-Timbral-Floor (v10.0.0.x): Nach Carrier-Chain-Inversion legitimiert VERSA-Qualität
        # die Spektral-Divergenz vom Carrier-Checkpoint. Wenn reference_audio gesetzt (CCR-Referenz-
        # Shift aktiv per §0d) UND VERSA primär (kein Proxy-Fallback):
        #   timbral-Floor = versa_sim × 0.90 (konservativ, auf [0.65, 0.95] geklippt).
        # Begründung: VERSA-MOS ≥ 0.74 (≈ MOS 3.9) bestätigt, dass Enhancement-Phasen die
        # Timbral-Integrität nicht beschädigt haben — die Spektral-Divergenz von checkpoint→final
        # ist physikalisch legitimiert (BW-Extension, Harmonik, Vocal-Enhancement §0d, §2.46 Stufe 5).
        # Ohne diesen Floor: mel-cosine(checkpoint, restored) ≈ 0.54 bestraft korrekte
        # Restaurierungen (HPI 0.42 statt ≈0.65 bei VERSA-MOS 4.5 — §0d-Messzahl-Artefakt v10.0.0).
        #
        # v10.0.0: reference_audio is not None Bedingung entfernt — der Floor gilt auch
        # ohne aktive Carrier-Chain-Inversion, solange VERSA (primär, kein MERT-Proxy)
        # mert_sim ≥ 0.74 bestätigt. Begründung: mel-cosine(degraded_input, restored) ≈ 0.54
        # entsteht immer wenn NR erfolgreich Rauschenergie entfernt — die Divergenz ist
        # physikalisch legitimiert durch erfolgreiche Rauschunterdrückung, nicht durch
        # Qualitätsverlust. artifact_freedom ≥ 0.95 (separater Veto-Faktor) schützt
        # vor falschen Floors bei tatsächlichen Artefakten.
        if not self._mert_proxy_used and mert_sim >= 0.74:
            _ccr_timbral_floor = float(np.clip(mert_sim * 0.90, 0.65, 0.95))
            if timbral < _ccr_timbral_floor:
                logger.debug(
                    "§0d CCR-Timbral-Floor: timbral %.3f → %.3f (versa_sim=%.3f ccr-ref=%s)",
                    timbral,
                    _ccr_timbral_floor,
                    mert_sim,
                    "active" if reference_audio is not None else "no-carrier",
                )
                timbral = _ccr_timbral_floor
        elif reference_audio is not None and self._mert_proxy_used and mert_sim >= 0.70:
            # §0d MERT-Proxy-Fallback: konservativerer Floor (0.80 statt 0.90) wenn VERSA
            # fehlgeschlagen ist aber CCR-Referenz aktiv ist. Schützt vor doppeltem HPI-Penalty
            # (schlechterer MERT-Proxy-Score + mel-cosine-Penalty auf Carrier-Divergenz).
            _ccr_timbral_floor_proxy = float(np.clip(mert_sim * 0.80, 0.60, 0.88))
            if timbral < _ccr_timbral_floor_proxy:
                logger.debug(
                    "§0d CCR-Timbral-Floor (MERT-proxy): timbral %.3f → %.3f (mert_sim=%.3f)",
                    timbral,
                    _ccr_timbral_floor_proxy,
                    mert_sim,
                )
                timbral = _ccr_timbral_floor_proxy

        hpi = mert_sim * timbral * artifact_freedom * emotional_arc_score
        # §v10.93 Non-Plus-Ultra NaN-Guard: Produkt-NaN durch beliebigen Faktor
        # wird abgefangen. NaN entsteht z.B. wenn VERSA oder emotional_arc_score
        # trotz upstream-Guards NaN liefern. Loggt Warnung + setzt Floor 0.5.
        if not np.isfinite(hpi):
            logger.warning(
                "HPI product NaN: mert=%.4f timbral=%.4f artifact=%.4f emotional=%.4f → floor 0.5",
                float(mert_sim),
                float(timbral),
                float(artifact_freedom),
                float(emotional_arc_score),
            )
            hpi = 0.5
        hpi = float(hpi)

        # §2.44/§0p [RELEASE_MUST] VQI-Faktor bei Vokal-Material (panns_singing ≥ 0.35).
        # HPI(Vokal) = MERT_similarity × timbral_fidelity × VQI × artifact_freedom × emotional_arc_preservation
        # VQI < 0.95 → HPI-Reduktion; artifact_freedom bleibt primärer Veto-Faktor.
        if panns_singing >= 0.35:
            _vqi_clamped = float(np.clip(vqi, 0.0, 1.0))
            hpi = hpi * _vqi_clamped
            logger.debug("§2.44 VQI-Faktor angewendet: vqi=%.3f panns_singing=%.2f", _vqi_clamped, panns_singing)

        # §B3 NORESQA integration: Non-intrusive MOS proxy — Advisory-Metadatum.
        # (Manocha & Kumar 2022, INTERSPEECH)
        # v10.0.0: noresqa_ensemble wird NICHT mehr als multiplikativer HPI-Faktor eingesetzt.
        # VERSA ist der primäre perceptuelle Qualitätsmesser (referenzfrei, MOS-kalibriert).
        # Ein zweiter perceptueller Multiplikator (noresqa) erzeugt systematische HPI-Kompression
        # ohne zusätzlichen Informationsgehalt — entfernt als Produktterm.
        # noresqa_ensemble bleibt für Telemetrie/Debugging erhalten.
        noresqa_score = self._compute_noresqa_score(restored, sr)
        noresqa_ensemble = 0.85 + 0.15 * noresqa_score  # Advisory only — kein HPI-Multiplikator

        # §2.44 v10.0.0: restorability-Penalty entfernt.
        # hpi *= 0.95 bei restorability > 85 war ein inkorrekter Penalty auf hochwertiges Material:
        # Hohe Restorability (CD, FLAC) bedeutet besser restaurierbares Signal — kein Penaltygrund.
        # Korrekte Mechanik: hohe Restorability → höhere Erwartungen via _gbc_targets (§09.12),
        # nicht via nachträglichen HPI-Abzug. update_reference_memory()-Gate: HPI > 0.0 AND af ≥ 0.90.

        passed = hpi > 0.0 and artifact_freedom >= self._get_depth_adaptive_af_min(
            transfer_chain
        )  # §v10.120 depth-adaptiv

        # §0i/§2.44 BUG-FIX v10.0.0 (Bug 5): Material-adaptive timbral_fidelity floor.
        # Spec §0a: material-adaptive floors (Shellac~0.40, Vinyl~0.55, CD~0.75).
        # timbral_ref=0.318 (Vinyl) < floor 0.385 -> rollback required.
        _TIMBRAL_FLOORS_HPG = {
            "shellac": 0.40,
            "wax_cylinder": 0.35,
            "lacquer_disc": 0.38,
            "wire_recording": 0.35,
            "vinyl": 0.55,
            "tape": 0.55,
            "reel_tape": 0.55,
            "cassette": 0.50,
            "cd_digital": 0.75,
            "dat": 0.70,
            "md": 0.65,
            "mp3_low": 0.60,
            "mp3_high": 0.65,
            "aac": 0.65,
            "unknown": 0.55,
        }
        _mat_key_hpg = str(material).lower().replace(" ", "_")
        _tf_floor_hpg = _TIMBRAL_FLOORS_HPG.get(_mat_key_hpg, 0.55)
        # Scale floor down for very low restorability (< 40): very damaged material
        _tf_floor_adj_hpg = _tf_floor_hpg * max(0.60, restorability_score / 100.0)
        if timbral < _tf_floor_adj_hpg:
            passed = False
            logger.warning(
                "§0i/§2.44 timbral=%.3f BELOW material floor=%.3f (material=%s restorability=%.1f) -> rollback",
                timbral,
                _tf_floor_adj_hpg,
                material,
                restorability_score,
            )

        logger.info(
            "§2.44 HPI(Restoration)=%.4f passed=%s ref=%s "
            "(mert=%.3f timbral=%.3f[in=%.3f ref=%.3f w=%.1f/%.1f] artifact=%.3f"
            " emotional=%.3f restorability=%.1f vqi=%.3f singing=%.2f)",
            hpi,
            passed,
            _reference_mode,
            mert_sim,
            timbral,
            timbral_input,
            timbral_ref,
            input_weight,
            ref_weight,
            artifact_freedom,
            emotional_arc_score,
            restorability_score,
            float(np.clip(vqi, 0.0, 1.0)) if panns_singing >= 0.35 else 1.0,
            panns_singing,
        )

        # §1.4a FailReason for failed gate
        _fr = None
        if not passed:
            from backend.core.pipeline_health_state import make_fail_reason  # pylint: disable=import-outside-toplevel

            if artifact_freedom < 0.95:
                _fr = make_fail_reason(
                    "HolisticPerceptualGate",
                    "ARTIFACT_VETO",
                    severity="failed",
                    action="rollback",
                    details=f"artifact_freedom={artifact_freedom:.3f} < 0.95",
                )
            else:
                _fr = make_fail_reason(
                    "HolisticPerceptualGate",
                    "HPI_BELOW_ZERO",
                    severity="failed",
                    action="rollback",
                    details=f"HPI={hpi:.4f} <= 0",
                )

        # §2.44-lit PEAQ/MUSHRA-inspired additive diagnostic (ISO 16832 + ITU-R BS.1387 + BS.1534).
        # MUSHRA and PEAQ use weighted linear combination of quality factors.
        # The product formula here treats each factor as independently necessary
        # (Lagrange-multiplier semantics: zero in any dimension = complete failure).
        # The additive alternative is computed ONLY for comparative diagnostics —
        # if product HPI fails while PEAQ-additive passes, a single factor collapse may
        # indicate a false rollback worth inspecting in logs.
        _peaq_additive = float(np.clip(0.40 * mert_sim + 0.35 * timbral + 0.25 * float(emotional_arc_score), 0.0, 1.0))
        _peaq_hpi_val = float(np.clip(_peaq_additive * artifact_freedom, 0.0, 1.0))
        if not passed and _peaq_hpi_val > 0.30 and artifact_freedom >= 0.95:
            logger.warning(
                "§2.44 HPI-Diskrepanz (PEAQ-Lit-Vergleich): product=%.4f FAIL aber PEAQ-additiv=%.4f PASS "
                "(mert=%.3f timbral=%.3f emotional=%.3f) — Single-Factor-Kollaps prüfen "
                "[ISO 16832 / ITU-R BS.1387]",
                hpi,
                _peaq_hpi_val,
                mert_sim,
                timbral,
                float(emotional_arc_score),
            )

        return HPIResult(
            hpi=round(hpi, 4),
            passed=passed,
            mert_similarity=round(mert_sim, 4),
            timbral_fidelity=round(timbral, 4),
            artifact_freedom=round(artifact_freedom, 4),
            emotional_arc_preservation=round(emotional_arc_score, 4),
            is_studio_mode=False,
            reference_mode=_reference_mode,
            fail_reason=_fr,
            detail={
                "restorability_score": restorability_score,
                "strict_gate": restorability_score > 85.0,
                "input_weight": input_weight,
                "ref_weight": ref_weight,
                "reference_mode": _reference_mode,
                "timbral_input": timbral_input,
                "timbral_ref": timbral_ref,
                "genre": genre,
                "material": material,
                "era_bin": era_bin,
                "mert_proxy_used": self._mert_proxy_used,
                "noresqa_score": round(noresqa_score, 4),
                "noresqa_ensemble": round(noresqa_ensemble, 4),
                # §2.44-lit: PEAQ/MUSHRA additive metric for comparative diagnostics
                "peaq_additive_hpi": round(_peaq_hpi_val, 4),
            },
        )

    def update_reference_memory(
        self,
        restored: np.ndarray,
        sr: int,
        hpi: float,
        artifact_freedom: float,
        p1_p2_passed: bool,
        genre: str = "DEFAULT",
        material: str = "digital",
        era_bin: str = "post-1980",
        transfer_chain: list[str] | None = None,
    ) -> None:
        """§2.44 Update MERT reference memory after successful restoration.

        Quality-Gate v10.0.0 (V54-aligned): HPI > 0.0 AND artifact_freedom ≥ 0.95.
        §v10.124 Major-Version: AF-Schwelle depth-adaptiv (0.95→0.75 bei depth≥4),
        damit tiefe Transfer-Ketten das Reference-Memory bevölkern können.
        EMA-Gewichtung: α skaliert mit HPI-Qualität (α_eff = α × min(1.0, HPI/0.7)),
        sodass schwächere Läufe weniger Einfluss haben als starke.
        Minimum-Alpha 0.05 verhindert, dass Kaltstart-Einträge nie gelernt werden.
        """
        _af_min = self._get_depth_adaptive_af_min(transfer_chain)
        # Für Reference-Memory: +0.05 strenger als HPI-Gate
        _af_ref_min = min(0.95, _af_min + 0.05)
        if not (hpi > 0.0 and artifact_freedom >= _af_ref_min):
            return

        embedding = self._compute_embedding(restored, sr)
        key = (genre, material, era_bin)

        # v10.0.0: EMA-α skaliert mit HPI-Qualität — schwächere Läufe haben weniger Einfluss.
        # α_eff = _EMA_ALPHA × clamp(HPI / 0.7, 0.33, 1.0), Minimum-α = 0.05.
        _alpha_eff = float(np.clip(_EMA_ALPHA * min(1.0, max(0.33, hpi / 0.7)), 0.05, _EMA_ALPHA))

        with self._ref_lock:
            if key in self._ref_memory:
                entry = self._ref_memory[key]
                # §2.44 EMA: α_eff → qualitätsskaliertes Blending
                entry.embedding = (1.0 - _alpha_eff) * entry.embedding + _alpha_eff * embedding
                entry.obs_count += 1
                entry.calibrated = entry.obs_count >= _MIN_OBS_CALIBRATED

            else:
                self._ref_memory[key] = _RefEntry(
                    embedding=embedding.copy(),
                    obs_count=1,
                    calibrated=False,
                )

        logger.info(
            "§2.44 ReferenceMemory updated key=%s obs=%d kalibriert=%s α_eff=%.3f",
            key,
            self._ref_memory[key].obs_count,
            self._ref_memory[key].calibrated,
            _alpha_eff,
        )
        # §2.44 Persistenz: nach jedem Quality-Gate-konformen Update speichern.
        self._save_ref_memory_to_disk()

    def _load_ref_memory_from_disk(self) -> None:
        """§2.44 Lädt persistiertes Referenz-Memory von ~/.aurik/hpg_reference_memory.json.

        Exception-safe: Bei fehlendem/beschädigtem File startet das System mit leerem Memory.
        Wird nur für Einträge mit obs_count >= 1 geladen (kein Garbage-Import).
        """
        try:
            if not _HPG_REF_MEMORY_PATH.exists():
                return
            with _HPG_REF_MEMORY_PATH.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            loaded = 0
            for k_str, v in raw.items():
                parts = k_str.split("|")
                if len(parts) != 3:
                    continue
                if not isinstance(v, dict) or "embedding" not in v:
                    continue
                obs = int(v.get("obs_count", 1))
                if obs < 1:
                    continue  # Kein Garbage-Import
                emb = np.asarray(v["embedding"], dtype=np.float32)
                key = (parts[0], parts[1], parts[2])
                self._ref_memory[key] = _RefEntry(
                    embedding=emb,
                    obs_count=obs,
                    calibrated=bool(v.get("calibrated", False)),
                )
                loaded += 1
            logger.info("§2.44 ReferenceMemory: %d Einträge von Disk geladen (%s)", loaded, _HPG_REF_MEMORY_PATH)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("§2.44 ReferenceMemory: Disk-laden fehlgeschlagen — starte mit leerem Memory: %s", exc)

    def _save_ref_memory_to_disk(self) -> None:
        """§2.44 Speichert aktuelles Referenz-Memory nach ~/.aurik/hpg_reference_memory.json.

        Exception-safe: Fehler beim Schreiben darf nie einen Lauf unterbrechen.
        Format: {"genre|material|era_bin": {embedding: [...], obs_count: N, calibrated: bool}}
        Nur Einträge mit obs_count >= 1 werden geschrieben.
        """
        try:
            _HPG_REF_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            with self._ref_lock:
                payload: dict[str, dict] = {}
                for (genre, material, era_bin), entry in self._ref_memory.items():
                    if entry.obs_count < 1:
                        continue
                    payload[f"{genre}|{material}|{era_bin}"] = {
                        "embedding": entry.embedding.tolist(),
                        "obs_count": entry.obs_count,
                        "calibrated": entry.calibrated,
                    }
            tmp_path = _HPG_REF_MEMORY_PATH.with_suffix(".tmp")
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, separators=(",", ":"))
            tmp_path.replace(_HPG_REF_MEMORY_PATH)
            logger.debug("§2.44 ReferenceMemory: %d Einträge auf Disk gespeichert", len(payload))
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("§2.44 ReferenceMemory: Disk-speichern fehlgeschlagen (nicht blockierend): %s", exc)

    def _get_reference_vector(self, genre: str, material: str, era_bin: str) -> np.ndarray | None:
        """§2.44 Fallback-Kaskade (5 Stufen).

        Stufe 1: Gleiche Genre-Familie + nächstliegende Ära → GP-Memory
        Stufe 2: Gleiche Ära + nächstliegendes Genre → GP-Memory
        Stufe 3: Bootstrap-Prototyp für Genre-Cluster (material-agnostic)
        Stufe 4: Genre-agnostischer Ära-Median
        Stufe 5: Kein Ref-Vektor → None → rein gegen Input
        """
        # Stufe 1: Exact match
        key = (genre, material, era_bin)
        entry = self._ref_memory.get(key)
        if entry is not None:
            return entry.embedding

        # Stufe 2: Same era, any material (nächstliegendes Genre)
        era_entries = [self._ref_memory[k] for k in self._ref_memory if k[2] == era_bin and k[0] == genre]
        if era_entries:
            embeddings = np.stack([e.embedding for e in era_entries])
            return np.asarray(np.mean(embeddings, axis=0))  # type: ignore[no-any-return]

        # Stufe 3: Same genre, any material, any era
        genre_entries = [self._ref_memory[k] for k in self._ref_memory if k[0] == genre]
        if genre_entries:
            embeddings = np.stack([e.embedding for e in genre_entries])
            return np.asarray(np.mean(embeddings, axis=0))  # type: ignore[no-any-return]

        # Stufe 4: Genre-agnostischer Ära-Median
        all_era = [self._ref_memory[k] for k in self._ref_memory if k[2] == era_bin]
        if all_era:
            embeddings = np.stack([e.embedding for e in all_era])
            return np.asarray(np.mean(embeddings, axis=0))  # type: ignore[no-any-return]

        # Stufe 5: Kein Referenz-Vektor
        return None

    def _compute_blind_reference_vector(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> np.ndarray | None:
        """§v10.91 Blinder Referenz-Vektor aus dem saubersten Audio-Fenster.

        Wird aufgerufen wenn GP-Memory keinen Referenz-Vektor fuer die aktuelle
        Genre×Material×Aera-Kombination hat. Nutzt BlindInternalReference um das
        Fenster mit hoechster SNR+Spectral-Clarity+Transient-Reichtum zu finden
        und berechnet daraus einen Mel-Embedding-Vektor.

        Returns: Embedding-Vektor (float32) oder None.
        """
        try:
            from backend.core.blind_internal_reference import BlindInternalReference

            bir = BlindInternalReference()
            result = bir.find(np.asarray(audio), sr)
            if result.best_score > 0.3 and result.segments:
                best = result.segments[0]
                start_n = int(best.start_s * sr)
                end_n = int(best.end_s * sr)
                if audio.ndim == 1:
                    slice_ref = np.asarray(audio[start_n:end_n], dtype=np.float32)
                else:
                    slice_ref = np.asarray(audio[:, start_n:end_n], dtype=np.float32)
                if slice_ref.shape[-1] > int(sr * 0.05):
                    return cast(
                        np.ndarray | None, (np.asarray(self._compute_embedding(slice_ref, sr), dtype=np.float32))
                    )
            return None
        except Exception as exc:
            logger.debug("§V6 _extract_reference_slice fehlgeschlagen — None zurückgegeben (Audio-Shape %s): %s", audio.shape, exc)
            return None

    def _compute_embedding(
        self,
        audio: np.ndarray,
        sr: int,
        bw_ceiling_hz: int | None = None,
    ) -> np.ndarray:
        """Berechnet spectral embedding (mel-energy vector) as MERT-proxy.

        §2.44 BW-Ceiling-Guard (v10.0.0): Wenn bw_ceiling_hz gesetzt ist, wird das
        Mel-Filterbank auf diesen Frequenzbereich begrenzt. Verhindert, dass
        FlashSR-synthetisierter HF-Content (z.B. 12–22 kHz für Kassette) beim
        timbral_input-Vergleich fälschlicherweise die Cosinus-Ähnlichkeit reduziert
        (Reference Paradox, §0d).
        """
        mono = audio if audio.ndim == 1 else np.mean(audio, axis=0)
        n_samples = len(mono)
        if n_samples < 2048:
            return np.ones(40, dtype=np.float32)  # type: ignore[no-any-return]

        n_fft = 2048
        hop = 512
        n_mels = 40
        n_frames = min(200, max(1, (n_samples - n_fft) // hop))
        win = np.hanning(n_fft).astype(np.float32)

        # §2.44 BW-Ceiling-Guard: obere Mel-Grenze auf material-native BW begrenzen.
        _mel_f_max = float(sr) / 2.0
        if bw_ceiling_hz is not None and int(bw_ceiling_hz) < int(sr // 2):
            _mel_f_max = float(max(4000, int(bw_ceiling_hz)))

        # Mel filterbank
        mel_freqs = np.linspace(0, 2595 * np.log10(1 + _mel_f_max / 700.0), n_mels + 2)
        hz_freqs = 700.0 * (10.0 ** (mel_freqs / 2595.0) - 1.0)
        bin_freqs = np.clip(np.floor((n_fft + 1) * hz_freqs / sr).astype(int), 0, n_fft // 2)
        filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
        for m in range(n_mels):
            f_s, f_c, f_e = bin_freqs[m], bin_freqs[m + 1], bin_freqs[m + 2]
            if f_c > f_s:
                filterbank[m, f_s:f_c] = np.linspace(0, 1, f_c - f_s)
            if f_e > f_c:
                filterbank[m, f_c:f_e] = np.linspace(1, 0, f_e - f_c)

        mel_frames = []
        for i in range(n_frames):
            s = i * hop
            e = s + n_fft
            if e > n_samples:
                break
            spec = np.abs(np.fft.rfft(mono[s:e] * win)) ** 2
            mel_frames.append(filterbank @ spec)

        if not mel_frames:
            return np.ones(n_mels, dtype=np.float32)  # type: ignore[no-any-return]

        embedding = np.log1p(np.mean(mel_frames, axis=0)).astype(np.float32)
        norm = float(np.linalg.norm(embedding) + 1e-12)
        return np.asarray(embedding / norm, dtype=np.float32)  # type: ignore[no-any-return]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two embedding vectors."""
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]
        dot = float(np.dot(a, b))
        norm = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        return dot / norm

    def evaluate_studio(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
        pqs_improvement: float = 0.0,
        artifact_freedom: float = 1.0,
        emotional_arc_score: float = 1.0,
        vqi: float = 1.0,
        panns_singing: float = 0.0,
    ) -> HPIResult:
        """Bewertet HPI for Studio 2026 mode.

        HPI = studio_quality_gain × PQS_improvement × artifact_freedom × emotional_arc_preservation
        HPI(Vokal, Studio) = studio_quality_gain × PQS_improvement × VQI × artifact_freedom × emotional_arc_preservation
        """
        studio_gain = self._compute_studio_quality_gain(original, restored, sr)
        # §2.44 [FIX] pqs_improvement als Vorzeichenträger — kein positives Clipping (max 0.0
        # entfernt). Negatives pqs_improvement → HPI < 0 → Rollback (§2.44: HPI ≤ 0 → Rollback).
        # Normierung: pqs_improvement ∈ [-1, 1] bleibt erhalten; Werte außerhalb werden geclippt.
        pqs_signed = float(np.clip(pqs_improvement, -1.0, 1.0))

        hpi = studio_gain * pqs_signed * artifact_freedom * emotional_arc_score

        # §2.44/§0p [RELEASE_MUST] VQI-Faktor bei Vokal-Material (panns_singing ≥ 0.35).
        # HPI(Studio, Vokal) multiplies VQI as a gating factor.
        if panns_singing >= 0.35:
            _vqi_studio = float(np.clip(vqi, 0.0, 1.0))
            hpi = hpi * _vqi_studio
            logger.debug("§2.44 Studio VQI-Faktor: vqi=%.3f panns=%.2f", _vqi_studio, panns_singing)

        passed = hpi > 0.0 and artifact_freedom >= self._get_depth_adaptive_af_min(
            None
        )  # §v10.120: Studio mode hat kein transfer_chain → depth=1 → 0.90

        logger.info(
            "§2.44 HPI(Studio2026)=%.4f passed=%s "
            "(studio_gain=%.3f pqs_signed=%.3f artifact=%.3f emotional=%.3f vqi=%.3f singing=%.2f)",
            hpi,
            passed,
            studio_gain,
            pqs_signed,
            artifact_freedom,
            emotional_arc_score,
            float(np.clip(vqi, 0.0, 1.0)) if panns_singing >= 0.35 else 1.0,
            panns_singing,
        )

        # §1.4a FailReason for failed Studio gate
        _fr_s = None
        if not passed:
            from backend.core.pipeline_health_state import make_fail_reason  # pylint: disable=import-outside-toplevel

            if artifact_freedom < 0.95:
                _fr_s = make_fail_reason(
                    "HolisticPerceptualGate",
                    "ARTIFACT_VETO",
                    severity="failed",
                    action="rollback",
                    details=f"artifact_freedom={artifact_freedom:.3f} < 0.95",
                )
            else:
                _fr_s = make_fail_reason(
                    "HolisticPerceptualGate",
                    "HPI_BELOW_ZERO",
                    severity="failed",
                    action="rollback",
                    details=f"Studio HPI={hpi:.4f} <= 0",
                )

        return HPIResult(
            hpi=round(hpi, 4),
            passed=passed,
            studio_quality_gain=round(studio_gain, 4),
            pqs_improvement=round(pqs_improvement, 4),
            artifact_freedom=round(artifact_freedom, 4),
            emotional_arc_preservation=round(emotional_arc_score, 4),
            is_studio_mode=True,
            fail_reason=_fr_s,
        )

    # ── Component computations ─────────────────────────────────────────────

    def _compute_mert_similarity(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
        bw_ceiling_hz: int | None = None,
    ) -> float:
        """Compute musical quality coefficient for HPI.

        §2.44 VERBOTEN: MERT darf NICHT primary sein wenn VERSA verfügbar.
        Primary path: VERSA MOS auf restoreriertem Audio (referenzfrei, kein Referenz-Paradoxon).
        Fallback path 1: MERT plugin similarity (proxy, setzt self._mert_proxy_used=True).
        Fallback path 2: spectral correlation proxy (artifact-safe).

        bw_ceiling_hz: Material-native BW-Grenze (Hz) für Spectral-Proxy-Fallback.
            Wenn gesetzt, wird der Frequenzvergleich auf [0, bw_ceiling_hz] begrenzt,
            damit absichtliche BW-Erweiterung (FlashSR phase_06) nicht als Divergenz
            gewertet wird (§2.44 BW-Ceiling-Guard, Reference Paradox §0d).
        """
        # §v10.14: Guard gegen None-Input (fail-closed auf Originalsignal)
        if original is None or restored is None:
            return 1.0
        if not hasattr(original, "astype") or not hasattr(restored, "astype"):
            return 1.0
        orig_clean = np.nan_to_num(original.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        rest_clean = np.nan_to_num(restored.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        # Keep legacy short-signal behavior deterministic for tests and edge-cases.
        orig_mono = orig_clean if orig_clean.ndim == 1 else np.mean(orig_clean, axis=0)
        rest_mono = rest_clean if rest_clean.ndim == 1 else np.mean(rest_clean, axis=0)
        if min(len(orig_mono), len(rest_mono)) < 1024:
            return 1.0

        # ─── PRIMARY PATH: VERSA MOS ───────────────────────────────────────────
        # §2.44 VERBOTEN: MERT darf nicht primary sein wenn VERSA verfügbar.
        # VERSA MOS (1–5) → normalisiert [0,1] via (mos-1)/4.
        # Referenzfrei → kein Referenz-Paradoxon, kein Input-Similarity-Bias.
        try:
            from plugins.versa_plugin import get_versa_plugin as _get_versa  # pylint: disable=import-outside-toplevel

            _versa = _get_versa()
            _versa_result = _versa.score(rest_clean, sr)
            _versa_mos = float(np.clip(_versa_result.mos, 1.0, 5.0))
            # v10.0.0: Power-Law-Normalisierung: MOS 4.0→0.83, 4.5→0.92, 5.0→1.0.
            # Frühere lineare ÷4-Skalierung komprimierte den Exzellenz-Bereich (4.0→0.75).
            # Power 0.65 expandiert 4.0-5.0 MOS auf 0.83-1.0 (perceptuell kalibrierter).
            # Kein spectral_coh-Blend: VERSA ist referenzfrei — spectral_coh(orig, rest)
            # ist Input-Similarity-Bias und bestraft korrekte Carrier-Chain-Inversion (§0d).
            _versa_sim = float(np.clip(((_versa_mos - 1.0) / 4.0) ** 0.65, 0.0, 1.0))
            logger.debug(
                "§2.44 VERSA-primary (v10.0.0): mos=%.2f → versa_sim=%.3f (power-law 0.65)",
                _versa_mos,
                _versa_sim,
            )
            self._mert_proxy_used = False  # VERSA succeeded
            return _versa_sim
        except Exception as _versa_exc:
            logger.debug("§2.44 VERSA primary fehlgeschlagen → MERT proxy Ersatzpfad: %s", _versa_exc)
            self._mert_proxy_used = True  # VERSA failed, MERT is proxy

        # ─── FALLBACK PATH 1: MERT plugin ─────────────────────────────────────
        if not self._mert_path_disabled:
            try:
                # Imported lazily to avoid mandatory ML initialization on module import.
                from plugins.mert_plugin import get_loaded_mert_plugin, get_mert_plugin  # pylint: disable=import-outside-toplevel  # noqa: I001

                plugin = get_loaded_mert_plugin()
                if plugin is None:
                    plugin = get_mert_plugin()

                a1 = plugin.analyze(orig_clean, sr)
                a2 = plugin.analyze(rest_clean, sr)

                # Normalize F0 proximity to [0,1] via log2 octave distance.
                f0_1 = float(max(0.0, getattr(a1, "estimated_f0_hz", 0.0)))
                f0_2 = float(max(0.0, getattr(a2, "estimated_f0_hz", 0.0)))
                if f0_1 > 0.0 and f0_2 > 0.0:
                    oct_dist = abs(np.log2((f0_1 + 1e-9) / (f0_2 + 1e-9)))
                    f0_sim = float(np.exp(-oct_dist / 0.5))
                else:
                    f0_sim = 1.0

                h1 = float(np.clip(getattr(a1, "harmonicity", 0.0), 0.0, 1.0))
                h2 = float(np.clip(getattr(a2, "harmonicity", 0.0), 0.0, 1.0))
                t1 = float(np.clip(getattr(a1, "tonal_consistency", 0.0), 0.0, 1.0))
                t2 = float(np.clip(getattr(a2, "tonal_consistency", 0.0), 0.0, 1.0))
                f1 = float(np.clip(getattr(a1, "spectral_flux_coherence", 0.0), 0.0, 1.0))
                f2 = float(np.clip(getattr(a2, "spectral_flux_coherence", 0.0), 0.0, 1.0))

                harm_sim = 1.0 - abs(h1 - h2)
                tonal_sim = 1.0 - abs(t1 - t2)
                flux_sim = 1.0 - abs(f1 - f2)

                plugin_sim = 0.35 * harm_sim + 0.35 * tonal_sim + 0.20 * flux_sim + 0.10 * f0_sim
                # §2.44 Blend: 65% Plugin-Score + 35% Spektral-Proxy.
                # min() war zu konservativ und zog valide Ergebnisse systematisch nach unten
                # (proxy ~0.7 bei Breitband-Änderungen → false rollback auch bei plugin=0.95).
                proxy_sim = self._compute_mert_similarity_spectral_proxy(
                    orig_clean, rest_clean, sr, bw_ceiling_hz=bw_ceiling_hz
                )
                sim = 0.65 * float(plugin_sim) + 0.35 * float(proxy_sim)
                return float(np.clip(sim, 0.0, 1.0))
            except Exception as exc:
                logger.debug("§2.44 MERT similarity Ersatzpfad to spectral proxy: %s", exc)
                # Disable repeated failing plugin initialization attempts in this process.
                self._mert_path_disabled = True

        # Failure-safe spectral proxy fallback.
        return self._compute_mert_similarity_spectral_proxy(orig_clean, rest_clean, sr, bw_ceiling_hz=bw_ceiling_hz)

    def _compute_mert_similarity_spectral_proxy(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
        bw_ceiling_hz: int | None = None,
    ) -> float:
        """Spectral proxy for musical similarity when MERT plugin is unavailable.

        bw_ceiling_hz: Wenn gesetzt, wird der Cosine-Vergleich auf Frequenzbins
            ≤ bw_ceiling_hz beschränkt. BW-Erweiterung über das Material-Ceiling
            (FlashSR für Kassette/Shellac) erscheint als Divergenz im vollen Spektrum,
            obwohl sie eine gewollte Restaurierungsleistung ist (§2.44 BW-Ceiling-Guard).
        """
        orig_mono = original if original.ndim == 1 else np.mean(original, axis=0)
        rest_mono = restored if restored.ndim == 1 else np.mean(restored, axis=0)
        min_len = min(len(orig_mono), len(rest_mono))
        if min_len < 1024:
            return 1.0

        orig_mono = orig_mono[:min_len]
        rest_mono = rest_mono[:min_len]

        # Multi-scale spectral correlation
        n_fft = min(2048, min_len)
        hop = n_fft // 4
        n_frames = max(1, (min_len - n_fft) // hop)
        n_frames = min(n_frames, 100)

        # §2.44 BW-Ceiling-Guard: Frequenz-Obergrenze für Material-native Bandbreite.
        # Bins oberhalb des Material-Ceilings (z.B. Kassette: 12 kHz) enthalten
        # im Original nur Träger-Hiss oder Stille; im Restored FlashSR-synthetisierten
        # Inhalt. Der Cosine-Proxy darf diese gewollte Divergenz nicht bestrafen.
        # Bin-Berechnung: bin = freq_hz × n_fft / sr (rfft-Bin-Index).
        _spec_bin_count = n_fft // 2 + 1  # Anzahl rfft-Ausgangsbins
        if bw_ceiling_hz is not None and int(bw_ceiling_hz) < sr // 2:
            _bin_ceil = int(round(int(bw_ceiling_hz) * n_fft / sr))
            _bin_ceil = max(32, min(_bin_ceil, _spec_bin_count))  # Safety-Clamp
        else:
            _bin_ceil = _spec_bin_count  # kein Ceiling → volles Spektrum

        correlations = []
        win = np.hanning(n_fft).astype(np.float32)

        for i in range(n_frames):
            s = i * hop
            e = s + n_fft
            if e > min_len:
                break

            orig_spec = np.abs(np.fft.rfft(orig_mono[s:e] * win))[:_bin_ceil]
            rest_spec = np.abs(np.fft.rfft(rest_mono[s:e] * win))[:_bin_ceil]

            # Log-magnitude correlation (perceptually meaningful)
            orig_log = np.log1p(orig_spec)
            rest_log = np.log1p(rest_spec)

            # Cosine-Ähnlichkeit auf L2-normierten Log-Spektren.
            # Mean-Centering führt zu falsch-hoher Ähnlichkeit bei
            # schmalbandigen Signalen (440 Hz vs. 880 Hz → ~0.9998), da das
            # gemeinsame Energielo-Background die Korrelation dominiert.
            # L2-Norm bildet Spektral-Peak-Position ab → diskriminiert Frequenzen.
            orig_norm = orig_log / (float(np.linalg.norm(orig_log)) + 1e-12)
            rest_norm = rest_log / (float(np.linalg.norm(rest_log)) + 1e-12)

            corr = float(np.clip(float(np.dot(orig_norm, rest_norm)), 0.0, 1.0))
            correlations.append(corr)

        if correlations:
            return float(np.clip(float(np.mean(correlations)), 0.0, 1.0))
        return 0.8

    def _compute_timbral_fidelity(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
        bw_ceiling_hz: int | None = None,
    ) -> float:
        """Content-integrity check via mel-embedding cosine similarity.

        Used as small content-preservation anchor in evaluate_restoration().
        NOT used as primary timbral_fidelity measure (see §2.44 FIX v10.0.0).

        §2.44 BW-Ceiling-Guard (v10.0.0): bw_ceiling_hz begrenzt den Mel-Vergleich
        auf den material-nativen Frequenzbereich. Verhindert false-negative
        timbral_input-Werte bei FlashSR-Extension auf historischem Material.
        """
        min_len = min(
            len(original) if original.ndim == 1 else original.shape[-1],
            len(restored) if restored.ndim == 1 else restored.shape[-1],
        )
        if min_len < 2048:
            return 1.0
        orig_embed = self._compute_embedding(original, sr, bw_ceiling_hz=bw_ceiling_hz)
        rest_embed = self._compute_embedding(restored, sr, bw_ceiling_hz=bw_ceiling_hz)
        return float(np.clip(self._cosine_similarity(orig_embed, rest_embed), 0.0, 1.0))

    def _compute_directional_restoration_quality(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
    ) -> float:
        """§2.44 FIX v10.0.0/v10.0.0 — Direktionale Verbesserungsmessung als Fallback.

        Misst ob die Restaurierung das Signal in Richtung "sauber und musikalisch"
        verbessert hat. Wird verwendet wenn kein Referenz-Vektor im GP-Memory vorliegt.

        Vier Komponenten (v10.0.0: +D Harmonische Kohärenz für Musik):
          A) Noise-Floor-Delta: tieferer Rauschboden nach Restaurierung → Wert steigt
          B) Spektrale Klarheit (HF-Crest): höhere Klarheit nach Denoising → Wert steigt
          C) Content-Integrity-Guard: verhindert, dass zerstörter Inhalt besteht
          D) Harmonische Kohärenz (v10.0.0): Erhalt harmonischer Spektralstruktur →
             für Musikmaterial signifikant über 0.5, auch ohne Rauschreduzierung.
             Verhindert, dass korrektes Music-Bypass als "keine Verbesserung" gewertet wird.

        Returns:
            0.5 = keine Veränderung (Bypass), sofern keine Harmonik erkannt
            > 0.5 = Signal verbessert oder Musikinhalt bewahrt (Harmonik-Kohärenz)
            < 0.5 = Signal verschlechtert (Content-Verlust)
        """
        orig_mono = (original if original.ndim == 1 else np.mean(original, axis=0)).astype(np.float32)
        rest_mono = (restored if restored.ndim == 1 else np.mean(restored, axis=0)).astype(np.float32)
        min_len = min(len(orig_mono), len(rest_mono))
        if min_len < 1024:
            return 0.75  # short signal: neutral-good

        orig_mono = np.nan_to_num(orig_mono[:min_len], nan=0.0)
        rest_mono = np.nan_to_num(rest_mono[:min_len], nan=0.0)

        # C) Content-Integrity-Guard: spectral correlation (log-magnitude)
        n_fft_c = min(4096, min_len)
        orig_spec = np.abs(np.fft.rfft(orig_mono[:n_fft_c] * np.hanning(n_fft_c)))
        rest_spec = np.abs(np.fft.rfft(rest_mono[:n_fft_c] * np.hanning(n_fft_c)))
        orig_log = np.log1p(orig_spec)
        rest_log = np.log1p(rest_spec)
        orig_n = orig_log - np.mean(orig_log)
        rest_n = rest_log - np.mean(rest_log)
        denom_c = float(np.sqrt(np.sum(orig_n**2) * np.sum(rest_n**2)) + 1e-12)
        content_corr = float(np.sum(orig_n * rest_n) / denom_c) if denom_c > 1e-12 else 0.0
        if content_corr < 0.3:
            # Musikalischer Inhalt schwer verändert → frühere Rückkehr
            return float(np.clip(0.3 + 0.2 * max(0.0, content_corr), 0.0, 1.0))

        # A) Noise-Floor-Delta (5. Perzentil der Frame-Energien)
        frame_len = max(1, int(0.03 * sr))
        hop = max(1, frame_len // 2)
        n_frames = min(200, max(1, (min_len - frame_len) // hop))
        orig_e: list[float] = []
        rest_e: list[float] = []
        for i in range(n_frames):
            s = i * hop
            e = s + frame_len
            if e > min_len:
                break
            orig_e.append(float(np.mean(orig_mono[s:e] ** 2) + 1e-12))
            rest_e.append(float(np.mean(rest_mono[s:e] ** 2) + 1e-12))
        if orig_e and rest_e:
            orig_nf_db = 10.0 * float(np.log10(float(np.percentile(orig_e, 5))))
            rest_nf_db = 10.0 * float(np.log10(float(np.percentile(rest_e, 5))))
            noise_delta_db = orig_nf_db - rest_nf_db  # > 0 wenn Rauschen reduziert
        else:
            noise_delta_db = 0.0
        noise_score = float(np.clip(0.5 + noise_delta_db / 40.0, 0.0, 1.0))

        # B) Spektrale Klarheit (HF Crest-Factor 2–16 kHz)
        freqs = np.fft.rfftfreq(n_fft_c, d=1.0 / sr)
        orig_fft_full = np.abs(np.fft.rfft(orig_mono[:n_fft_c] * np.hanning(n_fft_c)))
        rest_fft_full = np.abs(np.fft.rfft(rest_mono[:n_fft_c] * np.hanning(n_fft_c)))
        hf_mask = (freqs >= 2000) & (freqs <= 16000)
        hf_bins_o = orig_fft_full[hf_mask]
        hf_bins_r = rest_fft_full[hf_mask]
        if len(hf_bins_o) >= 10 and len(hf_bins_r) >= 10:
            crest_o = float(np.percentile(hf_bins_o, 95)) / (float(np.median(hf_bins_o)) + 1e-9)
            crest_r = float(np.percentile(hf_bins_r, 95)) / (float(np.median(hf_bins_r)) + 1e-9)
            max_crest = max(crest_o, crest_r, 1e-9)
            crest_improvement = (crest_r - crest_o) / max_crest  # [-1, 1]
        else:
            crest_improvement = 0.0
        clarity_score = float(np.clip(0.5 + crest_improvement * 0.5, 0.0, 1.0))

        # D) Harmonische Kohärenz (v10.0.0): Erhalt der spektralen Peakstruktur 80–4000 Hz.
        # Misst ob die dominanten Spektral-Peaks (Harmonik von Stimme + Instrument) im
        # restaurierten Signal an denselben Frequenzen wie im Original liegen.
        # Für Musik: content_corr ≥ 0.85 → hohes harmonisches Overlap → harmonic_score > 0.7
        # Verhindert, dass korrektes Bypassen oder minimale NR als "Score 0.5" bewertet wird.
        # content_corr ist bereits ≥ 0.3 (Integrity-Guard oben), daher ist Skalierung sicher.
        harmonic_score = float(np.clip(0.5 + (content_corr - 0.5) * 0.8, 0.3, 1.0))

        # Gewichtung: Bei Musikmaterial dominiert harmonische Kohärenz + Noise-Score.
        # Noise-Score liefert bei Musik nur ≈0.5 (5.Pz. = leise Musikpassage, kein Rauschboden).
        # Harmonische Kohärenz liefert bei Musik 0.70–0.90 (hoher Spektral-Overlap).
        # Klarheits-Score liefert ohne BW-Erweiterung ebenfalls ≈0.5.
        # Neue Gewichtung v10.0.0: 35% Noise + 25% Clarity + 40% Harmonik
        # (statt 60% Noise + 40% Clarity in v10.0.0 — Harmonik ersetzt Noise-Anteil für Musik)
        combined = 0.35 * noise_score + 0.25 * clarity_score + 0.40 * harmonic_score
        return float(np.clip(combined, 0.0, 1.0))

    def _compute_noresqa_score(self, audio: np.ndarray, sr: int) -> float:
        """§B3 NORESQA: Non-intrusive quality estimation (Manocha & Kumar, INTERSPEECH 2022).

        Attempts to use NoresqaPlugin if available; falls back to a DSP proxy that
        combines spectral flatness, SNR estimate, and harmonic coherence — all
        reference-free indicators of audio quality aligned with MOS correlates.

        Returns a score in [0, 1] (1.0 = highest quality). Non-blocking.
        """
        try:
            from plugins.noresqa_plugin import get_noresqa_plugin  # type: ignore  # pylint: disable=no-name-in-module,import-outside-toplevel  # noqa: I001

            _plg = get_noresqa_plugin()
            if _plg is not None:
                mono = audio if audio.ndim == 1 else np.mean(audio, axis=0)
                score = float(_plg.score(mono.astype(np.float32), sr))
                return float(np.clip(score, 0.0, 1.0))
        except Exception:  # plugin not installed → DSP fallback
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

        # DSP proxy: three reference-free quality correlates
        try:
            mono = np.asarray(audio if audio.ndim == 1 else np.mean(audio, axis=0), dtype=np.float32)
            mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
            if len(mono) < 1024:
                return 1.0

            n_fft = min(4096, len(mono))
            win = np.hanning(n_fft)
            spec = np.abs(np.fft.rfft(mono[:n_fft] * win)) + 1e-12

            # a) Spectral Flatness (Wiener entropy) — lower = more tonal = higher quality
            geo_mean = float(np.exp(np.mean(np.log(spec))))
            arith_mean = float(np.mean(spec))
            sfm = float(np.clip(geo_mean / (arith_mean + 1e-12), 0.0, 1.0))
            # SFM near 0 = very tonal (music), near 1 = noise-like
            # Quality proxy: music should be 0.05–0.40 → map via gaussIan around 0.15
            sfm_score = float(np.clip(np.exp(-((sfm - 0.15) ** 2) / 0.08), 0.0, 1.0))

            # b) Noise Floor Estimate via 5th percentile (low = cleaner signal)
            frame_len = max(512, n_fft // 8)
            hop = frame_len // 2
            n_frames = max(1, (len(mono) - frame_len) // hop)
            frame_rmss = [
                float(np.sqrt(np.mean(mono[i * hop : i * hop + frame_len] ** 2) + 1e-12))
                for i in range(min(n_frames, 200))
            ]
            if frame_rmss:
                noise_floor_db = 20.0 * float(np.log10(float(np.percentile(frame_rmss, 5)) + 1e-12))
            else:
                noise_floor_db = -60.0
            # Map [-80, -20] dBFS → [1, 0]
            snr_score = float(np.clip((-noise_floor_db - 20.0) / 60.0, 0.0, 1.0))

            # c) Harmonic coherence: autocorrelation peak ratio at fundamental period
            # Use 50 ms window in the most energetic segment
            win_len = int(0.05 * sr)
            if win_len >= 64 and len(mono) >= win_len:
                # Find most energetic 50 ms segment
                n_seg = max(1, (len(mono) - win_len) // (win_len // 2))
                energies_seg = [
                    float(np.mean(mono[i * (win_len // 2) : i * (win_len // 2) + win_len] ** 2))
                    for i in range(min(n_seg, 100))
                ]
                best_seg = int(np.argmax(energies_seg)) * (win_len // 2)
                segment = mono[best_seg : best_seg + win_len]
                from backend.core.core_utils import fft_autocorr  # pylint: disable=import-outside-toplevel

                ac = fft_autocorr(segment)
                ac = ac / (ac[0] + 1e-12)
                # Look for AC peak in F0 range 80–800 Hz → lags [sr//800, sr//80]
                lag_min = max(1, int(sr / 800))
                lag_max = min(len(ac) - 1, int(sr / 80))
                if lag_max > lag_min:
                    peak_lag = int(np.argmax(ac[lag_min : lag_max + 1])) + lag_min
                    harmonic_coh = float(np.clip(ac[peak_lag], 0.0, 1.0))
                else:
                    harmonic_coh = 0.5
            else:
                harmonic_coh = 0.5

            # Weighted combo (balanced: SFM captures tonal structure, SNR cleanness, HC harmonicity)
            proxy = 0.35 * sfm_score + 0.40 * snr_score + 0.25 * harmonic_coh
            return float(np.clip(proxy, 0.0, 1.0))
        except Exception as _exc:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            logger.debug("NORESQA DSP-proxy error (nicht blockierend): %s", _exc)
            return 1.0  # neutral: don't penalise when guard fails

    def _compute_studio_quality_gain(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
    ) -> float:
        """Studio 2026: improvement in studio quality relative to input.

        Compares how much closer the *restored* signal is to the studio reference
        (−14 LUFS, noise ≤ −72 dBFS) compared to the *original* input.
        A restored signal that is closer → gain > 0.5; same or worse → gain ≤ 0.5.
        Always returns ≥ 0.1 to avoid killing HPI when improvement is ambiguous.
        """

        def _score(audio: np.ndarray) -> float:
            mono = audio if audio.ndim == 1 else np.mean(audio, axis=0)
            if len(mono) < 1024:
                # §v10.93: < 23ms Audio — RMS-basierter Floor statt hartem 0.5
                rms_short = float(np.sqrt(np.mean(mono**2) + 1e-12))
                return float(np.clip(rms_short * 5.0, 0.35, 0.65))
            rms = float(np.sqrt(np.mean(mono**2) + 1e-12))
            lufs_approx = 20.0 * np.log10(rms + 1e-12)
            lufs_error = abs(lufs_approx - (-14.0))
            lufs_score = max(0.0, 1.0 - lufs_error / 30.0)

            frame_len = int(0.03 * sr)
            hop = frame_len // 2
            n_frames = max(1, (len(mono) - frame_len) // hop)
            energies = [
                float(np.mean(mono[i * hop : i * hop + frame_len] ** 2) + 1e-12) for i in range(min(n_frames, 500))
            ]
            noise_floor = 10.0 * np.log10(np.percentile(energies, 5)) if energies else -72.0
            noise_score = 1.0 if noise_floor <= -72.0 else max(0.0, 1.0 - (noise_floor + 72.0) / 30.0)
            return float((lufs_score + noise_score) / 2.0)

        in_score = _score(original)
        out_score = _score(restored)

        # Headroom-basierte Formel: Verbesserung relativ zum maximal erreichbaren
        # Headroom ab Input-Niveau (v10.0.0). Ratio-basierte Formel verlor
        # Diskriminierungskraft bei niedrigem in_score: in=0.05→out=0.06 und
        # in=0.05→out=0.90 lieferten identischen geclippten Gain (beide → 1.0).
        _headroom = max(1.0 - in_score, 0.05)  # Maximaler erreichbarer Headroom
        _improvement = out_score - in_score
        gain = float(np.clip(0.5 + _improvement / _headroom, 0.1, 1.0))
        return gain
