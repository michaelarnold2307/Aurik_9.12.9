"""
backend/core/musikalischer_globalplan.py — Musikalischer Globalplan-Dienst
===========================================================================

Der fehlende "Dach"-Layer zwischen Denker-Schicht und Pipeline-Schicht.

KONZEPT
-------
Vor Ausführung der 64-Phasen-Pipeline erzeugt dieser Dienst einen
*stilbewussten Restaurierungsplan*. Der Plan kodiert musikalisches Wissen
über Ära, Genre, Materialcharakter und emotionale Intention in konkreten
Per-Phasen-Parametern — kein nachträgliches Messen, sondern *planmäßiges
Handeln* von Anfang an.

ARCHITEKTUR (Cross-Phase-Reasoning)
-------------------------------------
                ┌─────────────────────────────────┐
                │      MusikalischerGlobalplan     │
                │                                  │
  EraClassifier ─► decade + era_style              │
  GermanSchlager ─► genre + subgenre + bpm         │
  CLAP-Embedding ─► semantisches Audio-Portrait    │
  DefectAnalysis ─► causal_cause + severity        │
                │         ▼                         │
                │  StilbewussterRestaurierungsplan  │
                │  • phase_adjustments[phase_id]    │
                │  • authenticity_target            │
                │  • tolerance_profile              │
                │  • emotional_intention            │
                └───────────┬─────────────────────-┘
                            │  wird in RestorationConfig.globalplan gesetzt
                            ▼
                   UnifiedRestorerV3 (64 Phasen)
                   — jede Phase kann plan.get(phase_id) lesen

SINGLETON-PATTERN (§3.x)
------------------------
Thread-sichere Instanziierung via Double-Checked Locking.

Author: Aurik Development Team
Version: 10.0.0
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.core.era_classifier import (
    EraClassifier,
)
from backend.core.era_classifier import (
    _dsp_fingerprint_decade as dsp_fingerprint_decade,
)
from backend.core.era_classifier import (
    _dsp_hf_rolloff as dsp_hf_rolloff,
)
from backend.core.era_classifier import (
    _estimate_snr as dsp_estimate_snr,
)
from backend.core.genre_classifier import GermanSchlagerClassifier
from backend.core.musical_phrase_context import get_phrase_extractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ära-Style-Profile: musikalisches Charakterwissen pro Dekade
# ---------------------------------------------------------------------------

# Für jede Ära definieren wir, wie die Pipeline sich verhalten soll.
# Werte sind relative Verstärkungs-/Dämpfungsfaktoren (1.0 = neutral).
_ERA_PROFILES: dict[int, dict[str, Any]] = {
    1890: {
        "label": "Früheste Aufnahmen (Zylinder/Wachswalze)",
        "nr_aggressiveness": 0.45,  # Sehr sanfte NR — Originalcharakter erhalten
        "harmonic_restore": 1.6,  # Starke Harmonik-Wiederherstellung
        "hf_ceiling_khz": 4.5,  # Authentische Bandbreitenbegrenzung
        "presence_boost": 0.0,  # Kein HF-Boost (wäre unhistorisch)
        "stereo_width": 0.0,  # Immer Mono
        "warmth_target": 0.92,  # Wärme ist Ära-Merkmal
        "authenticity_weight": 0.95,  # Authentizität hat höchste Priorität
        "nr_preserves_grain": True,  # Kornrauschen ist Teil des Charakters
    },
    1910: {
        "label": "Frühes Edison-Zeitalter",
        "nr_aggressiveness": 0.50,
        "harmonic_restore": 1.5,
        "hf_ceiling_khz": 5.5,
        "presence_boost": 0.0,
        "stereo_width": 0.0,
        "warmth_target": 0.90,
        "authenticity_weight": 0.93,
        "nr_preserves_grain": True,
    },
    1920: {
        "label": "Elektrische Aufnahmen (Kondensatormikrofon)",
        "nr_aggressiveness": 0.55,
        "harmonic_restore": 1.4,
        "hf_ceiling_khz": 7.5,
        "presence_boost": 0.05,
        "stereo_width": 0.0,
        "warmth_target": 0.88,
        "authenticity_weight": 0.90,
        "nr_preserves_grain": True,
    },
    1930: {
        "label": "Goldene Ära der Schellackplatte",
        "nr_aggressiveness": 0.60,
        "harmonic_restore": 1.3,
        "hf_ceiling_khz": 8.5,
        "presence_boost": 0.08,
        "stereo_width": 0.0,
        "warmth_target": 0.87,
        "authenticity_weight": 0.88,
        "nr_preserves_grain": False,
    },
    1940: {
        "label": "Krieg/Nachkriegszeit (frühe Magnetbandaufnahmen)",
        "nr_aggressiveness": 0.65,
        "harmonic_restore": 1.2,
        "hf_ceiling_khz": 11.0,
        "presence_boost": 0.10,
        "stereo_width": 0.08,
        "warmth_target": 0.85,
        "authenticity_weight": 0.85,
        "nr_preserves_grain": False,
    },
    1950: {
        "label": "High-Fidelity Ära (frühe Stereophonie)",
        "nr_aggressiveness": 0.70,
        "harmonic_restore": 1.1,
        "hf_ceiling_khz": 13.0,
        "presence_boost": 0.12,
        "stereo_width": 0.25,
        "warmth_target": 0.83,
        "authenticity_weight": 0.82,
        "nr_preserves_grain": False,
    },
    1960: {
        "label": "Studio-Stereo-Ära (Mehrspur-Bandmaschinen)",
        "nr_aggressiveness": 0.78,
        "harmonic_restore": 1.05,
        "hf_ceiling_khz": 16.0,
        "presence_boost": 0.14,
        "stereo_width": 0.45,
        "warmth_target": 0.80,
        "authenticity_weight": 0.78,
        "nr_preserves_grain": False,
    },
    1970: {
        "label": "Analoges Studio-Goldzeitalter",
        "nr_aggressiveness": 0.82,
        "harmonic_restore": 1.0,
        "hf_ceiling_khz": 18.0,
        "presence_boost": 0.15,
        "stereo_width": 0.55,
        "warmth_target": 0.78,
        "authenticity_weight": 0.74,
        "nr_preserves_grain": False,
    },
    1980: {
        "label": "Digitale Übergangsära (PCM + Analog-Mix)",
        "nr_aggressiveness": 0.88,
        "harmonic_restore": 0.95,
        "hf_ceiling_khz": 20.0,
        "presence_boost": 0.18,
        "stereo_width": 0.60,
        "warmth_target": 0.75,
        "authenticity_weight": 0.70,
        "nr_preserves_grain": False,
    },
    1990: {
        "label": "Digital-CD-Ära",
        "nr_aggressiveness": 0.92,
        "harmonic_restore": 0.92,
        "hf_ceiling_khz": 22.0,
        "presence_boost": 0.20,
        "stereo_width": 0.65,
        "warmth_target": 0.72,
        "authenticity_weight": 0.65,
        "nr_preserves_grain": False,
    },
    2000: {
        "label": "Loudness-War-Ära",
        "nr_aggressiveness": 0.95,
        "harmonic_restore": 0.90,
        "hf_ceiling_khz": 22.05,
        "presence_boost": 0.18,
        "stereo_width": 0.65,
        "warmth_target": 0.70,
        "authenticity_weight": 0.60,
        "nr_preserves_grain": False,
    },
    2010: {
        "label": "Streaming-Ära",
        "nr_aggressiveness": 0.95,
        "harmonic_restore": 0.90,
        "hf_ceiling_khz": 22.05,
        "presence_boost": 0.17,
        "stereo_width": 0.65,
        "warmth_target": 0.70,
        "authenticity_weight": 0.58,
        "nr_preserves_grain": False,
    },
    2020: {
        "label": "Moderne Produktion",
        "nr_aggressiveness": 0.95,
        "harmonic_restore": 0.90,
        "hf_ceiling_khz": 22.05,
        "presence_boost": 0.16,
        "stereo_width": 0.65,
        "warmth_target": 0.70,
        "authenticity_weight": 0.55,
        "nr_preserves_grain": False,
    },
}

# Genre-Modifikatoren — überlagern Ära-Profil additiv/multiplikativ
_GENRE_MODIFIERS: dict[str, dict[str, float]] = {
    "schlager": {
        "warmth_boost": +0.04,
        "presence_boost_add": +0.03,
        "nr_aggressiveness_mult": 0.95,  # Etwas sanftere NR für Vokal-Wärme
        "stereo_width_add": +0.05,
        "harmonic_restore_mult": 1.10,
    },
    "jazz": {
        "warmth_boost": +0.02,
        "presence_boost_add": +0.05,
        "nr_aggressiveness_mult": 0.90,  # Jazz lebt vom Raumklang — minimal NR
        "harmonic_restore_mult": 1.15,
    },
    "klassik": {
        "warmth_boost": +0.03,
        "nr_aggressiveness_mult": 0.85,  # Kleinste NR — Raumakustik erhalten
        "harmonic_restore_mult": 1.20,
        "stereo_width_add": +0.10,
    },
    "oper": {
        "warmth_boost": +0.02,
        "presence_boost_add": +0.04,
        "nr_aggressiveness_mult": 0.88,
        "harmonic_restore_mult": 1.18,
    },
    "volksmusik": {
        "warmth_boost": +0.05,
        "nr_aggressiveness_mult": 0.93,
        "harmonic_restore_mult": 1.08,
    },
    "rock": {
        "presence_boost_add": +0.06,
        "nr_aggressiveness_mult": 1.05,
        "stereo_width_add": +0.08,
    },
    "pop": {
        "presence_boost_add": +0.04,
        "nr_aggressiveness_mult": 1.00,
        "stereo_width_add": +0.05,
    },
    "unknown": {},  # Keine Modifikation
}

# Physically plausible lower decade bounds per source medium.
_MATERIAL_DECADE_FLOOR: dict[str, int] = {
    "vinyl": 1950,
    "cassette": 1965,
    "kassette": 1965,
    "reel_tape": 1940,
    "minidisc": 1990,
    "cd_digital": 1980,
    "dat": 1980,
    "mp3_low": 1990,
    "mp3_high": 1990,
    "aac": 1990,
}


# ---------------------------------------------------------------------------
# Ergebnis-Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class MusikalischesPortrait:
    """Semantisches Verständnis eines Audiostücks — das 'Gehör' des Denkers.

    Dieser Datencontainer kodiert alles, was Aurik über ein Stück 'weiß',
    bevor die erste Phase der Pipeline startet.
    """

    # Ära-Wissen
    decade: int  # z.B. 1940
    era_label: str  # z.B. "Goldene Ära der Schellackplatte"
    era_confidence: float  # Konfidenz [0,1]

    # Genre-Wissen
    genre: str  # z.B. "schlager", "jazz", "unbekannt"
    subgenre: str  # z.B. "wiener_schlager"
    genre_confidence: float  # Konfidenz [0,1]
    bpm: float  # Geschätztes Tempo

    # Semantisches CLAP-Portrait
    clap_available: bool  # Ob CLAP-Embeddings genutzt wurden
    semantic_similarity: float  # Ähnlichkeit zu Ära-Ankern [0,1]
    semantic_description: str  # Textuelles Portrait des Stücks

    # Emotionale Charakteristik
    estimated_mood: str  # z.B. "nostalgisch", "tänzerisch"
    warmth_score: float  # Wärme-Schätzung aus Spektralenergie [0,1]
    brightness_score: float  # Brillanz [0,1]
    dynamic_range_estimate: float  # Dynamikumfang [0,1]

    # Material
    material: str  # z.B. "shellac", "tape"

    def as_dict(self) -> dict[str, Any]:
        """Serialisiert das Portrait in ein kompakt gerundetes Dict."""
        return {
            "decade": self.decade,
            "era_label": self.era_label,
            "era_confidence": round(self.era_confidence, 3),
            "genre": self.genre,
            "subgenre": self.subgenre,
            "genre_confidence": round(self.genre_confidence, 3),
            "bpm": round(self.bpm, 1),
            "clap_available": self.clap_available,
            "semantic_similarity": round(self.semantic_similarity, 3),
            "semantic_description": self.semantic_description,
            "estimated_mood": self.estimated_mood,
            "warmth_score": round(self.warmth_score, 3),
            "brightness_score": round(self.brightness_score, 3),
            "dynamic_range_estimate": round(self.dynamic_range_estimate, 3),
            "material": self.material,
        }


@dataclass
class StilbewussterRestaurierungsplan:
    """Cross-Phase-aware Restaurierungsplan — das eigentliche 'Dach'.

    Jede Phase kann diesen Plan lesen und ihre Parameter entsprechend
    anpassen. So entsteht zum ersten Mal eine kohärente, stilbewusste
    Klangvorstellung ÜBER die gesamte Pipeline hinweg.
    """

    portrait: MusikalischesPortrait

    # Globale Ziele (aus Ära + Genre synthetisiert)
    authenticity_target: float  # Gewünschte Authentizitätsschwelle [0,1]
    warmth_target: float  # Ziel-Wärmewert [0,1]
    presence_target: float  # Ziel-Präsenz (Brillanz) [0,1]
    stereo_width_target: float  # Ziel-Stereobreite [0,1]
    hf_ceiling_khz: float  # Authentische HF-Grenzfrequenz

    # Emotionale Intention
    emotional_intention: str  # "wärme_erhalten", "brillanz_stärken", etc.
    preserve_grain: bool  # Ob Rauschen/Korn als Charaktermerkmal gilt

    # Per-Phase-Anpassungen: phase_id → {param: delta}
    # Jede Phase kann plan.get_phase_params(phase_id) aufrufen
    phase_adjustments: dict[str, dict[str, float]] = field(default_factory=dict)

    # Toleranzprofil: phase_id → erlaubte Qualitätsverschlechterung
    tolerance_profile: dict[str, float] = field(default_factory=dict)

    # Planungs-Metadaten
    plan_version: str = "9.10.47"
    reasoning_trace: list[str] = field(default_factory=list)

    def get_phase_params(self, phase_id: str) -> dict[str, float]:
        """Gibt stilbewusste Parameter für eine spezifische Phase zurück.

        Wenn keine phasenspezifischen Parameter existieren, werden
        die globalen Planwerte als Defaults zurückgegeben.
        """
        base = {
            "authenticity_weight": self.authenticity_target,
            "warmth_weight": self.warmth_target,
            "presence_weight": self.presence_target,
            "stereo_width": self.stereo_width_target,
            "hf_ceiling_khz": self.hf_ceiling_khz,
            "preserve_grain": 1.0 if self.preserve_grain else 0.0,
        }
        phase_specific = self.phase_adjustments.get(phase_id, {})
        base.update(phase_specific)
        return base

    def get_nr_aggressiveness(self) -> float:
        """Gibt die global geplante NR-Aggressivität zurück [0,1]."""
        return self.phase_adjustments.get("phase_03_denoise", {}).get("aggressiveness", 0.75)

    def as_dict(self) -> dict[str, Any]:
        """Serialisiert den Plan inklusive Portrait und Phase-Parametern."""
        return {
            "portrait": self.portrait.as_dict(),
            "authenticity_target": round(self.authenticity_target, 3),
            "warmth_target": round(self.warmth_target, 3),
            "presence_target": round(self.presence_target, 3),
            "stereo_width_target": round(self.stereo_width_target, 3),
            "hf_ceiling_khz": round(self.hf_ceiling_khz, 1),
            "emotional_intention": self.emotional_intention,
            "preserve_grain": self.preserve_grain,
            "phase_adjustments": {
                k: {pk: round(pv, 4) for pk, pv in v.items()} for k, v in self.phase_adjustments.items()
            },
            "reasoning_trace": self.reasoning_trace,
            "plan_version": self.plan_version,
        }


# ---------------------------------------------------------------------------
# DSP-Hilfsfunktionen (kein ML, rein numerisch)
# ---------------------------------------------------------------------------


def _safe_mono(audio: np.ndarray) -> np.ndarray:
    """Konvertiert to mono without NaN propagation."""
    arr = np.nan_to_num(np.asarray(audio, dtype=np.float32))
    if arr.ndim == 2:
        return arr.mean(axis=0) if arr.shape[0] <= arr.shape[1] else arr.mean(axis=1)  # type: ignore[no-any-return]
    return arr  # type: ignore[no-any-return]


def _estimate_warmth(mono: np.ndarray, sr: int) -> float:
    """Schätzt Wärme aus Verhältnis Low/Mid-Energie (< 1 kHz vs. 1–4 kHz).

    Algorithm:
        DFT-Energie in zwei Bändern via rfft; Verhältnis nach Newton-Cotes:

        .. math::

            w = \\frac{\\sum_{f < 1\\,\\text{kHz}} |X_f|^2}
                      {\\sum_{f < 1\\,\\text{kHz}} |X_f|^2
                       + \\sum_{1\\,\\text{kHz} \\le f < 4\\,\\text{kHz}} |X_f|^2}

        Clip auf [0, 1]. Fallback 0.5 bei zu kurzem Signal oder Fehler.
    """
    if not np.isfinite(mono).all():
        # §3.1 (NaN/Inf-Schutz): NaN-Input nicht durchschlagen lassen.
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
    if len(mono) < 512:
        # §v10.92: Statt hartem 0.5 — Zeitdomain-Wärme aus RMS-Ratio
        # Kurze Signale: Low/Mid-Approximation via subsample-Energie
        rms_full = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + 1e-12))
        if len(mono) >= 64:
            # Approximiere via Lowpass-Residuum
            lp_kernel = np.ones(min(64, len(mono))) / min(64, len(mono))
            lp_signal = np.convolve(np.abs(mono), lp_kernel, mode="same")
            low_power = float(np.mean(lp_signal[: min(256, len(mono))] ** 2) + 1e-12)
            total_power = float(np.mean(mono.astype(np.float64) ** 2) + 1e-12)
            return float(np.clip(low_power / (low_power + total_power), 0.0, 1.0))
        return float(np.clip(rms_full * 5.0, 0.0, 1.0))  # RMS-basierter Fallback
    try:
        fft = np.abs(np.fft.rfft(mono[: min(len(mono), 65536)]))
        freqs = np.fft.rfftfreq(min(len(mono), 65536), d=1.0 / sr)
        low_energy = float(np.sum(fft[freqs < 1000.0] ** 2) + 1e-12)
        mid_energy = float(np.sum(fft[(freqs >= 1000.0) & (freqs < 4000.0)] ** 2) + 1e-12)
        warmth = float(np.clip(low_energy / (low_energy + mid_energy), 0.0, 1.0))
        return warmth
    except Exception as e:
        logger.warning("musikalischer_globalplan.py::_estimate_warmth Ersatzpfad: %s", e, exc_info=True)
        # §v10.92: Statt hartem 0.5 — Zeitdomain-Proxy aus Signal-Energie
        try:
            rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + 1e-12))
            return float(np.clip(rms * 5.0, 0.3, 0.7))
        except Exception:
            logger.warning("§V6 ML→DSP-Fallback: _estimate_warmth Ersatzpfad fehlgeschlagen → neutraler Return (0.5)")
            return 0.5


def _estimate_brightness(mono: np.ndarray, sr: int) -> float:
    """Schätzt Brillanz aus HF-Energie (> 8 kHz) relativ zur Gesamtenergie.

    Algorithm:
        Spektraler Hochfrequenzanteil, skaliert auf [0, 1]:

        .. math::

            b = \\min\\!\\left(1,\\; 10 \\cdot
                \\frac{\\sum_{f > 8\\,\\text{kHz}} |X_f|^2}
                     {\\sum_f |X_f|^2}\\right)

        Faktor 10 normiert typische HF-Anteile auf intuitives [0, 1]-Intervall.
        Fallback 0.5 bei zu kurzem Signal oder Fehler.
    """
    if len(mono) < 512:
        # §v10.92: Statt hartem 0.5 — Zeitdomain-Brillanz via Zero-Crossing-Rate
        if len(mono) >= 32:
            zcr = float(np.mean(np.abs(np.diff(np.sign(mono[: min(512, len(mono))])))) / 2.0)
            return float(np.clip(zcr * 3.0, 0.0, 1.0))  # ZCR ~0.15 → Brillanz ~0.45
        return 0.5
    try:
        fft = np.abs(np.fft.rfft(mono[: min(len(mono), 65536)]))
        freqs = np.fft.rfftfreq(min(len(mono), 65536), d=1.0 / sr)
        hf_energy = float(np.sum(fft[freqs > 8000.0] ** 2) + 1e-12)
        total_energy = float(np.sum(fft**2) + 1e-12)
        brightness = float(np.clip(hf_energy / total_energy * 10.0, 0.0, 1.0))
        return brightness
    except Exception as e:
        logger.warning("musikalischer_globalplan.py::_estimate_brightness Ersatzpfad: %s", e, exc_info=True)
        # §v10.92: Zeitdomain-Brillanz aus Zero-Crossing-Rate
        try:
            zcr = float(np.mean(np.abs(np.diff(np.sign(mono))) / 2.0))
            return float(np.clip(zcr * 3.0, 0.3, 0.7))
        except Exception:
            logger.warning("§V6 ML→DSP-Fallback: _estimate_brightness Ersatzpfad fehlgeschlagen → neutraler Return (0.5)")
            return 0.5


def _estimate_dynamic_range(mono: np.ndarray) -> float:
    """Schätzt Dynamikumfang via Top-Percentile-Verhältnis.

    Algorithm:
        RMS-Energie in 4096-Sample-Blöcken; Interpercentile-Ratio:

        .. math::

            \\text{DR} = \\frac{P_{95}(\\text{RMS}) - P_{10}(\\text{RMS})}
                              {P_{95}(\\text{RMS})}

        Clip auf [0, 1]. Fallback 0.5 bei zu kurzem Signal oder Fehler.
    """
    if len(mono) < 128:
        # §v10.92: Statt hartem 0.5 — Peak/RMS-Ratio als DR-Proxy
        peak = float(np.max(np.abs(mono)) + 1e-12)
        rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + 1e-12))
        crest = peak / rms if rms > 0 else 1.0
        return float(np.clip((crest - 1.0) / 10.0, 0.0, 1.0))  # Crest ~10 → DR ~0.9
    try:
        rms_blocks = []
        block = 4096
        for i in range(0, len(mono) - block, block):
            b = mono[i : i + block]
            rms_blocks.append(float(np.sqrt(np.mean(b.astype(np.float64) ** 2) + 1e-12)))
        if not rms_blocks:
            return 0.5
        rms_arr = np.array(rms_blocks)
        p95 = float(np.percentile(rms_arr, 95))
        p10 = float(np.percentile(rms_arr, 10))
        dr = float(np.clip((p95 - p10) / (p95 + 1e-12), 0.0, 1.0))
        return dr
    except Exception as e:
        logger.warning("musikalischer_globalplan.py::_estimate_dynamic_range Ersatzpfad: %s", e, exc_info=True)
        # §v10.92: Peak/RMS-Crest-Faktor als DR-Proxy
        try:
            peak = float(np.max(np.abs(mono)) + 1e-12)
            rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + 1e-12))
            crest = peak / rms if rms > 0 else 1.0
            return float(np.clip((crest - 1.0) / 10.0, 0.3, 0.7))
        except Exception:
            logger.warning("§V6 ML→DSP-Fallback: _estimate_dynamic_range Ersatzpfad fehlgeschlagen → neutraler Return (0.5)")
            return 0.5


def _estimate_mood(warmth: float, brightness: float, bpm: float, genre: str) -> str:
    """Leitet eine textuell-semantische Stimmungsschätzung aus DSP-Merkmalen ab."""
    if genre == "schlager":
        if bpm < 70:
            return "melancholisch-nostalgisch"
        elif bpm < 100:
            return "schwungvoll-festlich"
        else:
            return "tänzerisch-fröhlich"
    elif genre == "jazz":
        if warmth > 0.70:
            return "warm-improvisierend"
        else:
            return "klar-analytisch"
    elif genre == "klassik":
        if brightness > 0.60:
            return "strahlend-festlich"
        elif warmth > 0.65:
            return "warm-romantisch"
        else:
            return "ruhig-kontemplativ"
    else:
        # Generische Stimmungsschätzung
        if warmth > 0.70 and brightness < 0.40:
            return "warm-nostalgisch"
        elif brightness > 0.65 and bpm > 100:
            return "lebendig-energetisch"
        elif warmth < 0.40 and brightness < 0.35:
            return "dunkel-expressiv"
        else:
            return "ausgewogen-natürlich"


def _nearest_era_profile(decade: int) -> dict[str, Any]:
    """Findet das nächste definierte Ära-Profil für eine Dekade."""
    available = sorted(_ERA_PROFILES.keys())
    # Nächstgelegene Dekade (Snap to defined)
    nearest = min(available, key=lambda d: abs(d - decade))
    return _ERA_PROFILES[nearest]


def _build_semantic_description(
    decade: int,
    genre: str,
    subgenre: str,
    material: str,
    mood: str,
    bpm: float,
) -> str:
    """Erzeugt ein textuelles Portrait des Stücks — die 'Sprache des Denkers'."""
    decade_str = f"{decade}er Jahre" if decade >= 1900 else f"um {decade}"
    genre_str = genre.capitalize() if genre != "unknown" else "unbekanntes Genre"
    subgenre_str = f" ({subgenre})" if subgenre and subgenre != "unknown" else ""
    material_str = {
        "shellac": "Schellackplatte",
        "vinyl": "Vinylschallplatte",
        "tape": "Magnetband",
        "broadcast": "Rundfunkaufnahme",
        "cd": "CD-Aufnahme",
        "digital": "Digitalaufnahme",
        "wax_cylinder": "Wachszylinder",
    }.get(material, material)
    bpm_str = f", ca. {bpm:.0f} BPM" if bpm > 0.0 else ""
    return f"{genre_str}{subgenre_str} aus den {decade_str}, aufgenommen auf {material_str}{bpm_str}. Stimmung: {mood}."


def _estimate_era_via_dsp_classifier(mono: np.ndarray, sr: int) -> tuple[int, float, float, float]:
    """Schätzt Dekade via EraClassifier-DSP-Helferfunktionen.

    Returns:
        tuple(decade, confidence, rolloff_hz, snr_db)
    """
    rolloff_hz = float(dsp_hf_rolloff(mono, sr))
    snr_db = float(dsp_estimate_snr(mono, sr))
    decade, era_conf = dsp_fingerprint_decade(rolloff_hz, snr_db)
    return int(decade), float(era_conf), rolloff_hz, snr_db


def _estimate_bpm_with_phrase_extractor(mono: np.ndarray, sr: int) -> float | None:
    """Liest BPM aus dem Phrase-Extractor, falls eine öffentliche API verfügbar ist."""
    extractor = get_phrase_extractor()
    try:
        tempo_fn = extractor.estimate_tempo
    except AttributeError as exc:
        logger.debug("§V6 Phrase-Extractor.estimate_tempo nicht verfügbar — None zurückgegeben (BPM-Schätzung): %s", exc)
        return None

    bpm = float(tempo_fn(mono, sr))
    return bpm if np.isfinite(bpm) and bpm > 0.0 else None


def _estimate_bpm_heuristic(mono: np.ndarray, sr: int) -> float:
    """Einfache BPM-Heuristik als robuster DSP-Fallback."""
    diff = np.diff(np.abs(mono))
    onsets: int = int(np.sum(diff > diff.std() * 2.5))
    duration_s = max(len(mono) / sr, 1.0)
    return float(np.clip(onsets / duration_s * 0.5 * 60.0, 60.0, 200.0))


# ---------------------------------------------------------------------------
# Kern-Klasse: MusikalischerGlobalplanDienst
# ---------------------------------------------------------------------------


class MusikalischerGlobalplanDienst:
    """Erzeugt stilbewusste Restaurierungspläne — das Dach über der Pipeline.

    Verbindet EraClassifier + GermanSchlagerClassifier + CLAP-Embedding
    mit konkreten Per-Phasen-Parametern, die die gesamte 64-Phasen-Pipeline
    stilkohärent steuern.

    Algorithmus (§2.2 Globalplan-Ausführungsreihenfolge):
    1. Ära-Klassifikation (CLAP Tier-1 → DSP Tier-2 → Heuristik Tier-3)
    2. Genre-Klassifikation (CLAP Zero-Shot → DSP-Rhythmus → Fallback)
    3. DSP-Portrait (Wärme, Brillanz, Dynamik, Tempo)
    4. Semantisches Portrait → emotional_intention + mood
    5. Ära-Profil × Genre-Modifikatoren → stilbewusste Zielwerte
    6. Cross-Phase-Anpassung: phasenspezifische Parameter ableiten
    7. Rückgabe: StilbewussterRestaurierungsplan
    """

    def __init__(self) -> None:
        self._era_classifier: Any = None
        self._genre_classifier: Any = None

    def _get_era_classifier(self) -> Any:
        """Gibt den EraClassifier als lazy initialisierte Instanz zurück."""
        if self._era_classifier is None:
            try:
                self._era_classifier = EraClassifier()
            except Exception as exc:
                logger.debug("EraClassifier nicht verfügbar: %s", exc)
        return self._era_classifier

    def _get_genre_classifier(self) -> Any:
        """Gibt den GermanSchlagerClassifier als lazy Instanz zurück."""
        if self._genre_classifier is None:
            try:
                self._genre_classifier = GermanSchlagerClassifier()
            except Exception as exc:
                logger.debug("GermanSchlagerClassifier nicht verfügbar: %s", exc)
        return self._genre_classifier

    def erstelle_plan(
        self,
        audio: np.ndarray,
        sr: int,
        material: str = "unknown",
        hint_genre: str | None = None,
        hint_decade: int | None = None,
        use_ml_classifiers: bool = True,
        chain_info: dict | None = None,
    ) -> StilbewussterRestaurierungsplan:
        """Erstellt den musikalischen Globalplan für ein Audiostück.

        Args:
            audio:               Eingabe-Audio (float32, 48 kHz empfohlen).
            sr:                  Sample-Rate.
            material:            Vorerkanntes Trägermedium (oder 'unknown').
            hint_genre:          Optionaler Genre-Hinweis (überwältigt Klassifikation).
            hint_decade:         Optionaler Dekaden-Hinweis (überwältigt Klassifikation).
            use_ml_classifiers:  Falls False, werden EraClassifier und
                                 GermanSchlagerClassifier übersprungen und nur
                                 DSP-Heuristiken verwendet. Setzt man auf False,
                                 wenn die ML-Klassifikatoren bereits an anderer
                                 Stelle in der Pipeline laufen (Anti-Parallelwelten,
                                 §Pflicht-Workflow), um Doppelausführung zu vermeiden.
            chain_info:          Optionaler Ketten-Dict aus TontraegerketteDenker
                                 (z. B. {"chain": ["tape", "mp3_low"], "primary": "tape"}).
                                 Wird für kettenadaptive NR-Caps verwendet.

        Returns:
            StilbewussterRestaurierungsplan bereit für Pipeline-Integration.
        """
        assert sr == 48000, f"Globalplan erwartet SR=48000, erhalten: {sr}"
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mono = _safe_mono(audio)
        reasoning: list[str] = []

        chain_materials: set[str] = set()
        primary_chain_material = ""
        if isinstance(chain_info, dict):
            for _key in ("chain", "transfer_chain"):
                _chain_raw = chain_info.get(_key)
                if isinstance(_chain_raw, list):
                    chain_materials.update(str(x).strip().lower() for x in _chain_raw if x is not None)
            for _key in ("primary", "primary_material"):
                _value = chain_info.get(_key)
                if isinstance(_value, str) and _value.strip():
                    primary_chain_material = _value.strip().lower()
                    break
            for _key in ("primary", "secondary", "tertiary", "primary_material"):
                _value = chain_info.get(_key)
                if isinstance(_value, str) and _value.strip():
                    chain_materials.add(_value.strip().lower())

        if not primary_chain_material and chain_materials:
            # Keep deterministic fallback when explicit primary is absent.
            primary_chain_material = sorted(chain_materials)[0]

        is_tape_chain = any(m in {"tape", "cassette", "kassette", "reel_tape"} for m in chain_materials)
        is_lossy_chain = any(m in {"mp3_low", "mp3_high", "aac"} for m in chain_materials)

        # ── 1. Ära-Klassifikation ───────────────────────────────────────────
        decade = hint_decade
        era_label = "Unbekannte Ära"
        era_conf = 0.0
        clap_available = False

        if decade is None and use_ml_classifiers:
            logger.debug("MusikalischerGlobalplan: CLAP-Ära-Pfad (use_ml_classifiers=True)")
            era_clf = self._get_era_classifier()
            if era_clf is not None:
                try:
                    era_result = era_clf.classify(audio, sr)
                    decade = era_result.decade
                    era_conf = float(era_result.confidence)
                    era_label = getattr(era_result, "era_label", f"{decade}er")
                    clap_available = getattr(era_result, "tier_used", 2) == 1
                    reasoning.append(
                        f"EraClassifier: {decade}er (Conf={era_conf:.2f}, Tier={'CLAP' if clap_available else 'DSP'})"
                    )
                except Exception as exc:
                    logger.debug("EraClassifier fehlgeschlagen: %s", exc)
                    reasoning.append(f"EraClassifier Fallback: {exc}")
        elif decade is None and not use_ml_classifiers:
            reasoning.append("ML-Klassifikatoren deaktiviert (use_ml_classifiers=False) — DSP-only")

        if decade is None:
            logger.debug(
                "MusikalischerGlobalplan: DSP-only Ära-Pfad (use_ml_classifiers=%s, hint_decade=%s)",
                use_ml_classifiers,
                hint_decade,
            )
            # Reuse the calibrated DSP era path instead of the old 95 %-energy bandwidth,
            # which overestimates modern bandwidth on bass-heavy music and pushes tape→MP3
            # transfers into the 1990s.
            try:
                decade, era_conf, rolloff_hz, snr_db = _estimate_era_via_dsp_classifier(mono, sr)
                bw_khz = rolloff_hz / 1000.0
                reasoning.append(f"Era-DSP-Heuristik: BW={bw_khz:.1f} kHz, SNR={snr_db:.1f} dB → {decade}er")
            except Exception:
                fft = np.abs(np.fft.rfft(mono[: min(len(mono), 32768)]))
                freqs = np.fft.rfftfreq(min(len(mono), 32768), d=1.0 / sr)
                energy = np.cumsum(fft**2)
                total = energy[-1] + 1e-12
                idx_95 = int(np.searchsorted(energy, 0.95 * total))
                bw_khz = float(freqs[min(idx_95, len(freqs) - 1)]) / 1000.0
                if bw_khz < 5.0:
                    decade = 1910
                elif bw_khz < 8.0:
                    decade = 1930
                elif bw_khz < 12.0:
                    decade = 1950
                elif bw_khz < 16.0:
                    decade = 1970
                else:
                    decade = 1990
                era_conf = 0.40
                reasoning.append(f"Bandbreiten-Heuristik: BW={bw_khz:.1f} kHz → {decade}er")

            if is_tape_chain and is_lossy_chain and 10.5 <= bw_khz <= 14.5 and decade >= 1950:
                decade = 1970
                era_conf = max(era_conf, 0.55)
                reasoning.append(
                    "Ketten-Prior: tape→lossy bei 10.5–14.5 kHz → 1970er Aufnahmeära plausibler als 1990er"
                )

            # §Fix9: Reine Lossy-Kette (MP3/AAC ohne Bandträger): Bandbreite ist
            # codec-bedingt, NICHT Aufnahme-ära-bedingt. MP3 existiert erst ab 1993.
            # Minimaler Decade-Floor = 1975 verhindert 1930er-Fehlzuweisung bei
            # niedrig-bitraten- oder bandbreite-limitierten Codec-Quellen.
            if is_lossy_chain and not is_tape_chain and decade < 1975:
                _decade_before_fix9 = decade
                decade = 1975
                era_conf = max(era_conf, 0.55)
                reasoning.append(
                    f"Lossy-Codec-Korrektur: Ära {_decade_before_fix9}er → 1975 "
                    "(MP3/AAC-Bandbreite ist codec-bedingt, kein Ära-Indikator; §Fix9)"
                )

        if decade is not None:
            medium_floor = _MATERIAL_DECADE_FLOOR.get(primary_chain_material, 0)
            if medium_floor and decade < medium_floor:
                decade_before_floor = int(decade)
                decade = medium_floor
                era_conf = max(era_conf, 0.60)
                reasoning.append(
                    f"Material-Floor: {primary_chain_material} erfordert mindestens {medium_floor}er "
                    f"(war {decade_before_floor}er)"
                )

        # Profil aus dezennium-nächstem Profil
        era_profile = _nearest_era_profile(decade)
        era_label = era_profile.get("label", f"{decade}er")

        # ── 2. Genre-Klassifikation ─────────────────────────────────────────
        genre = hint_genre or "unknown"
        subgenre = "unknown"
        genre_conf = 0.0
        bpm = 0.0

        if hint_genre is None and use_ml_classifiers:
            genre_clf = self._get_genre_classifier()
            if genre_clf is not None:
                try:
                    g_result = genre_clf.classify(audio, sr)
                    if g_result.is_schlager and g_result.confidence >= 0.30:
                        genre = "schlager"
                        subgenre = getattr(g_result, "subgenre", "unknown")
                        genre_conf = float(g_result.confidence)
                        bpm = float(getattr(g_result, "bpm", 0.0))
                        reasoning.append(
                            f"GermanSchlagerClassifier: Schlager (Conf={genre_conf:.2f}, "
                            f"Subgenre={subgenre}, BPM={bpm:.0f})"
                        )
                    else:
                        reasoning.append(f"GermanSchlagerClassifier: kein Schlager (Conf={g_result.confidence:.2f})")
                except Exception as exc:
                    logger.debug("GermanSchlagerClassifier fehlgeschlagen: %s", exc)
                    reasoning.append(f"GenreClassifier Fallback: {exc}")

        # BPM-Schätzung via DSP wenn nicht bekannt
        if bpm <= 0.0:
            try:
                bpm_from_phrase = _estimate_bpm_with_phrase_extractor(mono, sr)
                if bpm_from_phrase is not None:
                    bpm = bpm_from_phrase
                    reasoning.append(f"BPM via PhraseExtractor: {bpm:.1f}")
                else:
                    bpm = _estimate_bpm_heuristic(mono, sr)
                    reasoning.append(f"BPM-Heuristik: {bpm:.1f}")
            except Exception:
                # Einfache Onset-Heuristik als Fallback
                try:
                    bpm = _estimate_bpm_heuristic(mono, sr)
                    reasoning.append(f"BPM-Heuristik: {bpm:.1f}")
                except Exception:
                    bpm = 100.0

        # ── 3. DSP-Portrait ─────────────────────────────────────────────────
        warmth_raw = _estimate_warmth(mono, sr)
        brightness_raw = _estimate_brightness(mono, sr)
        dynamic_range = _estimate_dynamic_range(mono)

        if genre == "unknown" and is_tape_chain and is_lossy_chain and 1960 <= decade <= 1980:
            if 90.0 <= bpm <= 150.0 and warmth_raw >= 0.45 and brightness_raw <= 0.60:
                genre = "schlager"
                subgenre = "deutscher_schlager"
                genre_conf = 0.36
                reasoning.append("DSP-Fallback: tape→lossy + 1970er + moderates Tempo/Wärmeprofil → deutscher Schlager")

        # Semantische CLAP-Ähnlichkeit (wenn CLAP aktiv)
        semantic_sim = era_conf if clap_available else era_conf * 0.7

        # ── 4. Emotionale Intention ─────────────────────────────────────────
        mood = _estimate_mood(warmth_raw, brightness_raw, bpm, genre)
        semantic_desc = _build_semantic_description(decade, genre, subgenre, material, mood, bpm)
        reasoning.append(f"Semantisches Portrait: '{semantic_desc}'")

        # ── 5. Stilbewusste Zielwerte (Ära × Genre) ─────────────────────────
        genre_mod = _GENRE_MODIFIERS.get(genre, {})

        nr_aggressiveness = float(era_profile["nr_aggressiveness"]) * float(
            genre_mod.get("nr_aggressiveness_mult", 1.0)
        )
        warmth_target = float(era_profile["warmth_target"]) + float(genre_mod.get("warmth_boost", 0.0))
        presence_target = float(era_profile["presence_boost"]) + float(genre_mod.get("presence_boost_add", 0.0))
        stereo_target = float(era_profile["stereo_width"]) + float(genre_mod.get("stereo_width_add", 0.0))
        harmonic_restore = float(era_profile["harmonic_restore"]) * float(genre_mod.get("harmonic_restore_mult", 1.0))
        hf_ceiling_khz = float(era_profile["hf_ceiling_khz"])
        authenticity_target = float(era_profile["authenticity_weight"])
        preserve_grain = bool(era_profile["nr_preserves_grain"])

        # Clip alle Werte auf valide Bereiche
        nr_aggressiveness = float(np.clip(nr_aggressiveness, 0.1, 1.0))

        # Material-aware NR cap (§6.2): digitale/codec-komprimierte Quellen haben kein
        # Breitrauschen — Codec-Artefakte werden von Apollo/Resemble-Enhance behandelt,
        # NICHT von DeepFilterNet. Hohe NR-Stärke auf MP3/AAC zerhackt Musikinhalte
        # (Musical Noise / "Kratzen"). Deckeln auf ein minimales Schutzniveau.
        _DIGITAL_NR_CAP: dict[str, float] = {
            "mp3_low": 0.20,  # schwere Codec-Kompression → Apollo primär, NR minimal
            "mp3_high": 0.25,  # mittlere Kompression
            "aac": 0.25,
            "cd_digital": 0.30,  # nur echter Clipping-Schutz, kein Breitrauschen
            "dat": 0.30,
        }

        is_digital_chain = bool((material in _DIGITAL_NR_CAP) or any(m in _DIGITAL_NR_CAP for m in chain_materials))

        _nr_cap = _DIGITAL_NR_CAP.get(material)
        if _nr_cap is None:
            for _m in ("mp3_low", "mp3_high", "aac", "cd_digital", "dat"):
                if _m in chain_materials:
                    _nr_cap = _DIGITAL_NR_CAP[_m]
                    reasoning.append(f"NR-Cap aus Tonträgerkette übernommen: '{material}' + '{_m}'")
                    break

        if _nr_cap is None and material in {"tape", "kassette", "cassette", "reel_tape"} and decade >= 1970:
            _nr_cap = 0.25
            reasoning.append(f"NR-Cap für digitalisierte Bandquelle aktiviert (Material='{material}', Ära={decade}er)")

        if _nr_cap is not None and nr_aggressiveness > _nr_cap:
            reasoning.append(
                f"NR von {nr_aggressiveness:.2f} → {_nr_cap:.2f} begrenzt "
                f"(digitale Quelle '{material}': Codec-Artefakte via Apollo, kein Breitrauschen)"
            )
            nr_aggressiveness = _nr_cap

        warmth_target = float(np.clip(warmth_target, 0.0, 1.0))
        presence_target = float(np.clip(presence_target, 0.0, 1.0))
        stereo_target = float(np.clip(stereo_target, 0.0, 1.0))
        harmonic_restore = float(np.clip(harmonic_restore, 0.5, 2.0))
        hf_ceiling_khz = float(np.clip(hf_ceiling_khz, 3.0, 25.0))

        reasoning.append(
            f"Zielwerte: NR={nr_aggressiveness:.2f}, Wärme={warmth_target:.2f}, "
            f"Präsenz={presence_target:.2f}, Authentizität={authenticity_target:.2f}"
        )

        # ── 6. Cross-Phase-Anpassungen ──────────────────────────────────────
        phase_adjustments: dict[str, dict[str, float]] = {}

        # Phase 01: Click Removal — stärker bei Pre-1940-Material
        click_strength = float(np.clip(1.5 - (decade - 1890) / 200.0, 0.5, 1.5))
        phase_adjustments["phase_01_click_removal"] = {"strength": click_strength}

        # Phase 02: Hum Removal — 50Hz-Hum: stärker bei 1920–1950
        hum_strength = 1.2 if 1920 <= decade <= 1950 else 0.9
        phase_adjustments["phase_02_hum_removal"] = {"aggressiveness": hum_strength}

        # Phase 03: Denoise — Kernentscheidung: NR-Aggressivität
        # target_snr_db: für digitale Quellen ohne echtes Rauschen kleiner ansetzen,
        # damit DeepFilterNet nicht ins Musiksignal übergreift.
        _target_snr = 20.0 if is_digital_chain else (30.0 if decade < 1950 else 50.0)
        phase_adjustments["phase_03_denoise"] = {
            "aggressiveness": nr_aggressiveness,
            "preserve_grain": 1.0 if preserve_grain else 0.0,
            "target_snr_db": _target_snr,
        }

        # Phase 23: Spectral Repair (Clipping) — für Codec-Quellen stark begrenzen.
        # MP3/AAC-Quellen haben keine echten Clipping-Flat-Tops — phase_23 würde
        # Codec-Kerbmuster als Clipping misinterpretieren und zusätzliche Artefakte erzeugen.
        _spectral_repair_strength = 0.30 if is_digital_chain else 1.0
        phase_adjustments["phase_23_spectral_repair"] = {
            "strength": _spectral_repair_strength,
        }

        # Phase 04: EQ Correction — HF-Deckel aus Ära-Profil
        phase_adjustments["phase_04_eq_correction"] = {
            "hf_ceiling_khz": hf_ceiling_khz,
            "warmth_boost_db": float(np.clip((warmth_target - 0.75) * 4.0, -2.0, 4.0)),
        }

        # Phase 06: Frequency Restoration — harmonische Vollständigkeit
        phase_adjustments["phase_06_frequency_restoration"] = {
            "restoration_strength": harmonic_restore,
            "max_freq_khz": hf_ceiling_khz,
        }

        # Phase 07: Harmonic Restoration — Kernentscheidung für Klangcharakter
        phase_adjustments["phase_07_harmonic_restoration"] = {
            "harmonic_strength": harmonic_restore,
            "preserve_overtones": authenticity_target,
            "warmth_target": warmth_target,
        }

        # Phase 13: Stereo Enhancement — äraauthentische Breite
        phase_adjustments["phase_13_stereo_enhancement"] = {
            "target_width": stereo_target,
            "force_mono": 1.0 if stereo_target < 0.05 else 0.0,
        }

        # Phase 14: Phase Correction — wichtiger bei Pre-1960 (schlecht synchronisierte Bandmaschinen)
        pc_strength = float(np.clip(1.5 - (decade - 1890) / 150.0, 0.6, 1.4))
        phase_adjustments["phase_14_phase_correction"] = {"correction_strength": pc_strength}

        # Phase 17: Mastering Polish — Ära-aware Sättigungscharakter
        saturation = float(np.clip((1950 - decade) / 200.0, 0.0, 0.3)) if decade < 1950 else 0.0
        phase_adjustments["phase_17_mastering_polish"] = {
            "saturation_amount": saturation,
            "warmth": warmth_target,
            "presence": presence_target,
        }

        # Phase 21: Exciter — sparsam einsetzen (Authentizität)
        exciter_strength = float(np.clip((1.0 - authenticity_target) * 0.5, 0.0, 0.3))
        phase_adjustments["phase_21_exciter"] = {"strength": exciter_strength}

        # Phase 22: Tape Saturation — nur für tape-Material & Vor-1970
        tape_sat = (
            float(np.clip((1.0 - (decade - 1940) / 100.0), 0.0, 1.0))
            if material in ("tape", "unknown") and decade < 1970
            else 0.0
        )
        phase_adjustments["phase_22_tape_saturation"] = {"amount": tape_sat}

        # Phase 35: Multiband Compression — weniger aggressiv für historische Materialien
        mb_ratio = float(np.clip(1.0 + (decade - 1950) / 100.0, 1.0, 3.0))
        phase_adjustments["phase_35_multiband_compression"] = {"ratio": mb_ratio}

        # Phase 37: Bass Enhancement — Wärme-gesteuert
        bass_boost = float(np.clip((warmth_target - 0.75) * 3.0, 0.0, 2.0))
        phase_adjustments["phase_37_bass_enhancement"] = {"boost_db": bass_boost}

        # Phase 38: Presence Boost — Brillanz-gesteuert
        phase_adjustments["phase_38_presence_boost"] = {"boost_db": presence_target * 4.0}

        # Phase 39: Air Band — stark einschränken für historische Materialien
        air_max = float(np.clip((hf_ceiling_khz - 10.0) / 15.0, 0.0, 1.0))
        phase_adjustments["phase_39_air_band_enhancement"] = {"max_gain": air_max}

        # Phase 46: Spatial Enhancement — äraauthentisch
        phase_adjustments["phase_46_spatial_enhancement"] = {
            "width_target": stereo_target,
            "depth": float(np.clip(authenticity_target * 0.8, 0.0, 0.8)),
        }

        # Phase 48: Stereo Width Enhancer — streng limitiert durch Ära
        phase_adjustments["phase_48_stereo_width_enhancer"] = {
            "max_width": stereo_target + 0.05,
        }

        # Emotionale Intention ableiten
        if warmth_target > 0.85 and authenticity_target > 0.85:
            emotional_intention = "wärme_und_authentizität_maximieren"
        elif warmth_target > 0.82:
            emotional_intention = "wärme_erhalten_brillanz_schonen"
        elif presence_target > 0.15:
            emotional_intention = "brillanz_stärken_dynamik_erhalten"
        elif preserve_grain:
            emotional_intention = "charakter_erhalten_artefakte_minimieren"
        else:
            emotional_intention = "ausgewogen_modernisieren"

        reasoning.append(f"Emotionale Intention: '{emotional_intention}'")

        # ── Portrait zusammenstellen ─────────────────────────────────────────
        portrait = MusikalischesPortrait(
            decade=decade,
            era_label=era_label,
            era_confidence=era_conf,
            genre=genre,
            subgenre=subgenre,
            genre_confidence=genre_conf,
            bpm=bpm,
            clap_available=clap_available,
            semantic_similarity=float(np.clip(semantic_sim, 0.0, 1.0)),
            semantic_description=semantic_desc,
            estimated_mood=mood,
            warmth_score=warmth_raw,
            brightness_score=brightness_raw,
            dynamic_range_estimate=dynamic_range,
            material=material,
        )

        plan = StilbewussterRestaurierungsplan(
            portrait=portrait,
            authenticity_target=authenticity_target,
            warmth_target=warmth_target,
            presence_target=presence_target,
            stereo_width_target=stereo_target,
            hf_ceiling_khz=hf_ceiling_khz,
            emotional_intention=emotional_intention,
            preserve_grain=preserve_grain,
            phase_adjustments=phase_adjustments,
            reasoning_trace=reasoning,
        )

        logger.info(
            "🎼 MusikalischerGlobalplan: %s | Ära=%s (%.0f%%) | Genre=%s | NR=%.2f | Wärme=%.2f | Intention='%s'",
            portrait.semantic_description[:60],
            decade,
            era_conf * 100,
            genre,
            nr_aggressiveness,
            warmth_target,
            emotional_intention,
        )
        return plan


# ---------------------------------------------------------------------------
# Singleton (§3.x — Thread-sicher, Double-Checked Locking)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_INSTANCE_HOLDER: dict[str, MusikalischerGlobalplanDienst | None] = {"instance": None}


def get_musikalischer_globalplan_dienst() -> MusikalischerGlobalplanDienst:
    """Thread-safe Singleton-Zugriff auf den Globalplan-Dienst."""
    if _INSTANCE_HOLDER["instance"] is None:
        with _lock:
            if _INSTANCE_HOLDER["instance"] is None:
                _INSTANCE_HOLDER["instance"] = MusikalischerGlobalplanDienst()
    dienst = _INSTANCE_HOLDER["instance"]
    assert dienst is not None
    return dienst


def erstelle_globalplan(
    audio: np.ndarray,
    sr: int,
    material: str = "unknown",
    hint_genre: str | None = None,
    hint_decade: int | None = None,
    use_ml_classifiers: bool = True,
    chain_info: dict | None = None,
) -> StilbewussterRestaurierungsplan:
    """Convenience-Funktion: erstellt den Globalplan via Singleton-Dienst."""
    return get_musikalischer_globalplan_dienst().erstelle_plan(
        audio,
        sr,
        material=material,
        hint_genre=hint_genre,
        hint_decade=hint_decade,
        use_ml_classifiers=use_ml_classifiers,
        chain_info=chain_info,
    )
