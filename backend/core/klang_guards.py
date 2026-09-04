"""§L–Q: Klangwirksame Perceptual Guards für das menschliche Ohr.

L: BassPunchCoupling   – Bass/Punch-Verhältnis nach Bass- und Transient-Phasen
M: VocalFormantGuard   – Formant-Stabilität nach Stimm-Phasen (F2-Drift)
N: StereoCoherenceGuard – ICCC nach Stereo-Phasen
O: DynamicsArcGuard    – LUFS-Bogen-Erhalt über die Pipeline
P: DefectEQProfile     – Defekt→Frequenzband-Mapping für phase_16_final_eq
Q: ListeningMode       – Hörer-Perspektive modifiziert Goal-Gewichte
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# §L Bass-Punch-Koordination
# ═══════════════════════════════════════════════════════════════════════════════

_BASS_PHASES = frozenset(
    {
        "phase_37_bass_enhancement",
        "phase_06_frequency_restoration",
    }
)
_PUNCH_PHASES = frozenset(
    {
        "phase_08_transient_preservation",
        "phase_54_transparent_dynamics",
    }
)


class BassPunchCoupling:
    """Misst Sub-Bass (20-60Hz) / Kick-Punch (60-200Hz) Verhältnis."""

    def __init__(self) -> None:
        self._baseline_ratio: float | None = None
        self._lock = threading.Lock()

    def measure(self, audio: np.ndarray, sr: int) -> float:
        """Gibt das Bass/Punch-Energieverhältnis zurück (0.5=ausgewogen,>1=basslastig)."""
        try:
            mono = np.mean(audio, axis=0) if audio.ndim == 2 else audio
            from backend.core.audio_utils import safe_to_mono

            mono = safe_to_mono(np.asarray(mono, dtype=np.float32))
            fft = np.abs(np.fft.rfft(mono, n=min(65536, len(mono))))
            freqs = np.fft.rfftfreq(min(65536, len(mono)), d=1.0 / sr)
            sub = float(np.mean(fft[(freqs >= 20) & (freqs <= 60)]))
            punch = float(np.mean(fft[(freqs >= 60) & (freqs <= 200)]))
            if punch < 1e-10:
                return 1.0
            return sub / punch
        except Exception as e:
            logger.warning("klang_guards.py::measure Ersatzpfad: %s", e)
            return 1.0

    def set_baseline(self, audio: np.ndarray, sr: int) -> None:
        with self._lock:
            self._baseline_ratio = self.measure(audio, sr)

    def check_and_adjust(self, audio: np.ndarray, sr: int, phase_id: str, strength: float) -> float:
        """Prüft ratio vs baseline. Gibt gecappte Stärke zurück."""
        if self._baseline_ratio is None or strength <= 0:
            return strength
        current = self.measure(audio, sr)
        ratio_drift = current / max(self._baseline_ratio, 1e-6)
        if phase_id in _BASS_PHASES and ratio_drift > 1.8:
            return strength * 0.7
        if phase_id in _PUNCH_PHASES and ratio_drift < 0.5:
            return strength * 0.7
        return strength


# ═══════════════════════════════════════════════════════════════════════════════
# §M Vocal-Naturalness-Monitor (Formant-Tracking light)
# ═══════════════════════════════════════════════════════════════════════════════

_VOCAL_PHASES = frozenset(
    {
        "phase_03_denoise",
        "phase_49_advanced_dereverb",
        "phase_65_vocal_naturalness_restoration",
        "phase_66_vocal_deesser",
    }
)


class VocalFormantGuard:
    """Misst spektrale Zentroid-Drift als Proxy für Formant-Stabilität."""

    def __init__(self) -> None:
        self._baseline_centroid: float | None = None
        self._baseline_harmonicity: float | None = None

    def _measure(self, audio: np.ndarray, sr: int) -> tuple[float, float]:
        """Gibt (spectral_centroid_hz, harmonicity_ratio) zurück."""
        try:
            mono = np.mean(audio, axis=0) if audio.ndim == 2 else np.asarray(audio, dtype=np.float32)
            fft = np.abs(np.fft.rfft(mono))
            freqs = np.fft.rfftfreq(len(mono), d=1.0 / sr)
            mask = (freqs >= 200) & (freqs <= 4000)
            if not np.any(mask):
                return 800.0, 0.5
            centroid = float(np.average(freqs[mask], weights=fft[mask] + 1e-10))
            # Harmonicity proxy: ratio of harmonic peaks to total energy
            total = float(np.sum(fft[mask]))
            if total < 1e-10:
                return centroid, 0.0
            # Simple peak detection: every 100Hz
            harmonic_energy = 0.0
            for f0 in range(100, 1000, 100):
                idx = int(f0 * len(mono) / sr)
                if 0 < idx < len(fft) - 2:
                    harmonic_energy += float(np.max(fft[max(0, idx - 2) : idx + 3]))
            return centroid, min(1.0, harmonic_energy / total)
        except Exception as e:
            logger.warning("klang_guards.py::_measure Ersatzpfad: %s", e)
            return 800.0, 0.5

    def set_baseline(self, audio: np.ndarray, sr: int) -> None:
        self._baseline_centroid, self._baseline_harmonicity = self._measure(audio, sr)

    def check_formant_drift(self, audio: np.ndarray, sr: int) -> tuple[bool, float]:
        """Returns (is_unnatural, drift_pct)."""
        if self._baseline_centroid is None:
            return False, 0.0
        c, h = self._measure(audio, sr)
        drift = abs(c - self._baseline_centroid) / max(self._baseline_centroid, 1.0)
        h_loss = max(0.0, self._baseline_harmonicity - h) if self._baseline_harmonicity else 0.0
        return drift > 0.08 or h_loss > 0.15, drift


# ═══════════════════════════════════════════════════════════════════════════════
# §N Stereo-Feld-Integritätswächter (ICCC)
# ═══════════════════════════════════════════════════════════════════════════════

_STEREO_PHASES = frozenset(
    {
        "phase_13_stereo_enhancement",
        "phase_14_phase_correction",
        "phase_15_stereo_balance",
    }
)


class StereoCoherenceGuard:
    """ICCC-basierter Wächter für Stereo-Integrität."""

    def __init__(self) -> None:
        self._baseline_iccc: float | None = None

    def _iccc(self, audio: np.ndarray) -> float:
        """Interchannel Cross-Correlation Coefficient."""
        try:
            if audio.ndim < 2 or audio.shape[0] < 2:
                return 1.0
            L = np.asarray(audio[0], dtype=np.float64)
            R = np.asarray(audio[1], dtype=np.float64)
            eps = 1e-12
            num = np.mean(L * R)
            den = np.sqrt(np.mean(L * L) * np.mean(R * R)) + eps
            return float(np.clip(num / den, -1.0, 1.0))
        except Exception as e:
            logger.warning("klang_guards.py::_iccc Ersatzpfad: %s", e)
            return 1.0

    def set_baseline(self, audio: np.ndarray) -> None:
        if audio.ndim >= 2:
            self._baseline_iccc = self._iccc(audio)

    def check_coherence(self, audio: np.ndarray) -> tuple[bool, float]:
        """Returns (field_degraded, iccc_drop_pct)."""
        if self._baseline_iccc is None:
            return False, 0.0
        current = self._iccc(audio)
        drop = self._baseline_iccc - current
        return drop > 0.15, drop


# ═══════════════════════════════════════════════════════════════════════════════
# §O Dynamik-Bogen-Erhalt (LUFS Arc)
# ═══════════════════════════════════════════════════════════════════════════════

_DYNAMICS_PHASES = frozenset(
    {
        "phase_03_denoise",
        "phase_54_transparent_dynamics",
        "phase_16_final_eq",
        "phase_38_presence_boost",
    }
)


class DynamicsArcGuard:
    """Misst LUFS-Verlauf über den Song und prüft Bogen-Erhalt."""

    def __init__(self) -> None:
        self._baseline_arc: np.ndarray | None = None
        self._segment_lufs: list[float] = []

    def _measure_lufs_arc(self, audio: np.ndarray, sr: int, segments: int = 8) -> np.ndarray:
        """Misst Short-Term LUFS in N Segmenten."""
        try:
            mono = np.mean(audio, axis=0) if audio.ndim == 2 else np.asarray(audio, dtype=np.float32)
            n = len(mono)
            seg_len = max(sr // 2, n // segments)
            lufs_vals = []
            for i in range(0, n - seg_len + 1, seg_len):
                seg = mono[i : i + seg_len]
                power = np.mean(seg * seg) + 1e-12
                lufs_vals.append(-0.691 + 10.0 * math.log10(power))
            return cast(np.ndarray, (np.array(lufs_vals, dtype=np.float32)))
        except Exception as e:
            logger.warning("klang_guards.py::_measure_lufs_arc Ersatzpfad: %s", e)
            return cast(np.ndarray, (np.zeros(segments, dtype=np.float32)))

    def set_baseline(self, audio: np.ndarray, sr: int) -> None:
        self._baseline_arc = self._measure_lufs_arc(audio, sr)

    def check_arc_preserved(self, audio: np.ndarray, sr: int, max_lufs_drift: float = 2.0) -> tuple[bool, float]:
        """Returns (arc_compromised, max_drift_lufs)."""
        if self._baseline_arc is None:
            return False, 0.0
        current = self._measure_lufs_arc(audio, sr)
        min_len = min(len(self._baseline_arc), len(current))
        drift = float(np.max(np.abs(self._baseline_arc[:min_len] - current[:min_len])))
        return drift > max_lufs_drift, drift


# ═══════════════════════════════════════════════════════════════════════════════
# §P Defekt-spezifische EQ-Profile
# ═══════════════════════════════════════════════════════════════════════════════

_DEFECT_EQ_BANDS: dict[str, dict[str, Any]] = {
    "hum": {"freqs": [(48, 52, -3.0), (148, 152, -2.5)], "comment": "50Hz-Brummen + Oberwellen"},
    "buzz": {"freqs": [(48, 52, -3.0), (98, 102, -3.0), (148, 152, -2.5)], "comment": "Brummen+Surren"},
    "click": {"freqs": [(2000, 6000, -1.0)], "comment": "Click-Reparatur-Residuen glätten"},
    "crackle": {"freqs": [(3000, 10000, -1.5)], "comment": "Knistern-Residuen dämpfen"},
    "hiss": {"freqs": [(6000, 16000, -2.0)], "comment": "Band-Höhenrauschen"},
    "clipping": {"freqs": [(2000, 8000, +1.5)], "comment": "Clipping: Höhen rekonstruieren"},
    "wow_flutter": {"freqs": [(200, 800, +0.8)], "comment": "Pitch-Korrektur-Mitten glätten"},
    "rumble": {"freqs": [(20, 60, -3.0)], "comment": "Sub-Bass-Rumpeln"},
    "surface_noise": {"freqs": [(4000, 12000, -2.0)], "comment": "Oberflächengeräusch"},
}


def get_defect_eq_profile(defect_types: list[str]) -> list[dict[str, Any]]:
    """Erstellt ein EQ-Profil aus detektierten Defekten für phase_16_final_eq."""
    profile: list[dict[str, Any]] = []
    seen_bands: set[tuple[int, int]] = set()
    for dt in defect_types:
        dt_lower = dt.lower().replace(" ", "_").replace("-", "_")
        for key, spec in _DEFECT_EQ_BANDS.items():
            if key in dt_lower or dt_lower in key:
                for f_low, f_high, gain_db in spec["freqs"]:
                    band_key = (f_low, f_high)
                    if band_key not in seen_bands:
                        seen_bands.add(band_key)
                        profile.append({"f_low": f_low, "f_high": f_high, "gain_db": gain_db, "reason": key})
    return profile


# ═══════════════════════════════════════════════════════════════════════════════
# §Q Hörer-Perspektiven-Kalibrierung (Listening Mode)
# ═══════════════════════════════════════════════════════════════════════════════

_LISTENING_MODES: dict[str, dict[str, Any]] = {
    "headphones": {
        "label": "Kopfhörer",
        "goal_adjust": {
            "raeumlichkeit": 1.4,
            "transparenz": 1.3,
            "mikrodynamik": 1.2,
            "bass_praesenz": 0.7,
            "punch": 0.8,
        },
    },
    "nearfield": {
        "label": "Nahfeld-Monitore",
        "goal_adjust": {},  # neutral = keine Anpassung
    },
    "farfield": {
        "label": "Wohnzimmer/HiFi",
        "goal_adjust": {
            "bass_praesenz": 1.3,
            "punch": 1.2,
            "waerme": 1.2,
            "brillanz": 1.1,
            "hoehen_luft": 1.15,
        },
    },
    "car": {
        "label": "Auto",
        "goal_adjust": {
            "bass_praesenz": 1.5,
            "punch": 1.4,
            "hoehen_luft": 1.3,
            "brillanz": 1.2,
            "textverstaendlichkeit": 1.3,
            "raeumlichkeit": 0.6,
            "mikrodynamik": 0.7,
        },
    },
}


def get_listening_mode(mode_key: str = "nearfield") -> dict[str, Any]:
    """Gibt Listening-Mode-Config zurück (nearfield = Default)."""
    return _LISTENING_MODES.get(mode_key, _LISTENING_MODES["nearfield"])


def apply_listening_mode_to_weights(
    base_weights: dict[str, float],
    mode_key: str = "nearfield",
) -> dict[str, float]:
    """Multipliziert Goal-Gewichte mit Listening-Mode-Faktoren."""
    mode = get_listening_mode(mode_key)
    adjusted = dict(base_weights)
    for goal, factor in mode.get("goal_adjust", {}).items():
        if goal in adjusted:
            adjusted[goal] = round(adjusted[goal] * factor, 3)
    return adjusted


# ═══════════════════════════════════════════════════════════════════════════════
# §R Cross-Phase-Guard-Koordination
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# Guard-Wisdom: Akkumuliertes Wissen aus allen Guards, phasenübergreifend
# ═══════════════════════════════════════════════════════════════════════════════


class GuardWisdom:
    """Sammelt Guard-Ergebnisse über alle Phasen und leitet Korrekturen ab."""

    def __init__(self, material: str = "unknown", genre: str = ""):
        self._material = material
        self._genre = genre
        self._history: list[dict] = []
        self._rollback_count: int = 0
        self._strength_mod: float = 1.0

    def record(self, phase_id: str, guard_name: str, metrics: dict, verdict: str = "ok"):
        self._history.append({"phase": phase_id, "guard": guard_name, "metrics": metrics, "verdict": verdict})
        if verdict == "violation":
            self._rollback_count += 1
            self._strength_mod = max(0.3, 1.0 - 0.15 * self._rollback_count)

    def get_strength_mod(self) -> float:
        """Multiplikator für nachfolgende Phasen-Stärken basierend auf Guard-Historie."""
        return self._strength_mod

    def should_retry(self, phase_id: str) -> tuple[bool, float]:
        """(should_retry, recommended_strength) für die aktuelle Phase."""
        recent = [h for h in self._history[-3:] if h["verdict"] == "violation"]
        if len(recent) >= 2:
            return True, max(0.3, self._strength_mod - 0.1)
        if recent:
            return True, self._strength_mod
        return False, 1.0

    def adaptive_threshold(self, guard_name: str, base_threshold: float) -> float:
        """Passt Schwellwerte an Material und Genre an."""
        # Material-spezifische Lockerung
        material_factor = {
            "wax_cylinder": 1.5,
            "shellac": 1.3,
            "vinyl": 1.1,
            "tape": 1.0,
            "cassette": 1.15,
            "cd_digital": 0.8,
        }.get(self._material, 1.0)
        # Genre-spezifische Lockerung
        genre_factor = {
            "schlager": 0.9,
            "classical": 0.8,
            "jazz": 0.85,
            "rock": 1.1,
            "metal": 1.2,
            "electronic": 1.15,
        }.get(self._genre, 1.0)
        return base_threshold * material_factor * genre_factor

    def snapshot(self) -> dict:
        return {
            "history_len": len(self._history),
            "rollbacks": self._rollback_count,
            "strength_mod": self._strength_mod,
        }


class CrossGuardCoordinator:
    """Integriert L–O Guard-Ergebnisse und findet Kompromisse.

    Problem: L sagt „Bass ist tight", N sagt „Stereo-Feld leidet".
    Einzeln würden sie sich widersprechen. Der Coordinator entscheidet
    gemeinsam: wenn 2 von 3 Guards ok sind, wird nicht eingegriffen.
    """

    def __init__(self) -> None:
        self._snapshots: dict[str, dict[str, Any]] = {}

    def record(self, guard_name: str, phase_id: str, metrics: dict[str, Any]) -> None:
        key = f"{guard_name}:{phase_id}"
        self._snapshots[key] = dict(metrics)

    def evaluate(self) -> dict[str, Any]:
        """Gibt eine Kompromiss-Entscheidung zurück mit Einzelwertungen."""
        results: dict[str, Any] = {"verdict": "ok", "conflicts": [], "actions": []}
        # Zähle Warnungen pro Kategorie
        bass_warnings = [v for k, v in self._snapshots.items() if k.startswith("bass_punch")]
        formant_warnings = [v for k, v in self._snapshots.items() if k.startswith("formant")]
        stereo_warnings = [v for k, v in self._snapshots.items() if k.startswith("iccc")]
        dynamics_warnings = [v for k, v in self._snapshots.items() if k.startswith("dynamics_arc")]

        unhealthy = 0
        if any(w.get("ratio_drift", 0) > 1.8 for w in bass_warnings):
            unhealthy += 1
            results["conflicts"].append("bass_overboost")
        if any(w.get("formant_drift", 0) > 0.08 for w in formant_warnings):
            unhealthy += 1
            results["conflicts"].append("vocal_unnatural")
        if any(w.get("iccc_drop", 0) > 0.15 for w in stereo_warnings):
            unhealthy += 1
            results["conflicts"].append("stereo_degraded")
        if any(w.get("lufs_drift", 0) > 2.0 for w in dynamics_warnings):
            unhealthy += 1
            results["conflicts"].append("dynamics_compromised")

        if unhealthy >= 2:
            results["verdict"] = "degraded"
            results["actions"].append("§R Kompromiss: 2+ Bereiche betroffen → reduziere Stärke um 15 %")
        elif unhealthy == 1:
            results["verdict"] = "warning"
            results["actions"].append("§R Ein Bereich marginal → keine automatische Korrektur")
        return results

    def reset(self) -> None:
        self._snapshots.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# §S Emotional-Arc-Preservation (Arousal/Valence)
# ═══════════════════════════════════════════════════════════════════════════════


class EmotionalArcPreserver:
    """Misst Arousal/Valence-Kurve über den Song und prüft Erhalt.

    Arousal-Proxy: RMS-Energie in 2 kHz–8 kHz (Präsenz-Band), geglättet.
    Valence-Proxy:  Spektrales Zentroid in 200 Hz–2 kHz (Mitten-Wärme).

    Die emotionale Kurve muss erhalten bleiben — Musik soll „atmen".
    """

    def __init__(self) -> None:
        self._baseline_arousal: np.ndarray | None = None
        self._baseline_valence: np.ndarray | None = None

    def _measure(self, audio: np.ndarray, sr: int, segments: int = 16) -> tuple[np.ndarray, np.ndarray]:
        """Returns (arousal_curve, valence_curve) as segment arrays."""
        try:
            mono = np.mean(audio, axis=0) if audio.ndim == 2 else np.asarray(audio, dtype=np.float32)
            n = len(mono)
            seg_len = max(sr // 2, n // segments)
            arousal, valence = [], []
            for i in range(0, n - seg_len + 1, seg_len):
                seg = mono[i : i + seg_len]
                fft = np.abs(np.fft.rfft(seg))
                freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
                # Arousal: Energie 2k–8k / Gesamt
                a_mask = (freqs >= 2000) & (freqs <= 8000)
                arousal.append(float(np.sum(fft[a_mask]) / (np.sum(fft) + 1e-10)))
                # Valence: Spektrales Zentroid 200–2k
                v_mask = (freqs >= 200) & (freqs <= 2000)
                if np.sum(fft[v_mask]) > 1e-10:
                    valence.append(float(np.average(freqs[v_mask], weights=fft[v_mask] + 1e-10)))
                else:
                    valence.append(800.0)
            return np.array(arousal, dtype=np.float32), np.array(valence, dtype=np.float32)
        except Exception as e:
            logger.warning("klang_guards.py::_measure Ersatzpfad: %s", e)
            return np.zeros(segments, dtype=np.float32), np.zeros(segments, dtype=np.float32)

    def set_baseline(self, audio: np.ndarray, sr: int) -> None:
        self._baseline_arousal, self._baseline_valence = self._measure(audio, sr)

    def check_preserved(self, audio: np.ndarray, sr: int) -> tuple[bool, dict[str, Any]]:
        """Returns (arc_intact, metrics_dict)."""
        if self._baseline_arousal is None:
            return True, {}
        cur_a, cur_v = self._measure(audio, sr)
        min_len = min(len(self._baseline_arousal), len(cur_a))
        # Korrelation der Verläufe (nicht absolute Werte)
        a_corr = float(np.corrcoef(self._baseline_arousal[:min_len], cur_a[:min_len])[0, 1]) if min_len > 2 else 1.0
        v_corr = float(np.corrcoef(self._baseline_valence[:min_len], cur_v[:min_len])[0, 1]) if min_len > 2 else 1.0  # type: ignore[index]
        ok = a_corr > 0.85 and v_corr > 0.85
        return ok, {
            "arousal_corr": a_corr,
            "valence_corr": v_corr,
            "warning": "Emotionaler Bogen verändert" if not ok else "",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# §T Hörermüdigkeits-Prävention (Humanization-Pass)
# ═══════════════════════════════════════════════════════════════════════════════


class HumanizationPass:
    """Fügt minimale, nichthörbare Variation hinzu um Listening Fatigue zu verhindern.

    37 DSP-Phasen können das Audio „zu perfekt" machen — steriler Klang,
    Ermüdung nach 15 min Hören. Der Humanization-Pass fügt Mikro-Variation
    (< 0.3 dB, < 0.5 ms) hinzu, die das Gehirn als „lebendig" registriert,
    ohne messbare Klangveränderung.

    §v10.15: Adaptive Stärke statt hartcodiertem 0.15.
    """

    # §v10.15: Kalibrierungs-Konstanten
    _STRENGTH_MIN: float = 0.05  # Sehr lebendiges Material
    _STRENGTH_MAX: float = 0.25  # Sehr steriles Material
    _STRENGTH_DEFAULT: float = 0.15  # Fallback wenn Kalibrierung fehlschlägt

    @staticmethod
    def calibrate_strength(audio: np.ndarray, sr: int) -> float:
        """Kalibriert die Humanization-Stärke basierend auf den akustischen
        Eigenschaften des Materials.

        Analysiert:
          - Mikrodynamik (Perzentil-Spread der RMS-Hüllkurve)
          - Spektrale Varianz (wie „flach" oder „belebt" das Spektrum ist)

        Returns:
            Stärke ∈ [_STRENGTH_MIN, _STRENGTH_MAX].
            Niedrig = Material bereits lebendig.
            Hoch = Material klingt steril, braucht mehr Humanization.
        """
        try:
            import numpy as np

            arr = np.asarray(audio, dtype=np.float64)
            if arr.ndim > 1:
                arr = arr.mean(axis=0) if arr.shape[0] <= arr.shape[1] else arr.mean(axis=1)

            # ── 1. Mikrodynamik: RMS-Hüllkurve ───────────────────────
            # Block-RMS über 50 ms Fenster → Perzentil-Spread (p95−p5)
            block_s = max(1, int(sr * 0.05))  # 50 ms
            n_blocks = max(1, len(arr) // block_s)
            rms_env = np.array(
                [
                    float(np.sqrt(np.mean(arr[i * block_s : (i + 1) * block_s] ** 2) + 1e-12))
                    for i in range(min(n_blocks, 200))  # max 10 s
                ]
            )
            if len(rms_env) < 4:
                return HumanizationPass._STRENGTH_DEFAULT
            p5 = float(np.percentile(rms_env, 5))
            p95 = float(np.percentile(rms_env, 95))
            p50 = float(np.percentile(rms_env, 50))
            # Normalisierter Spread: hoher Spread = lebendig
            micro_dyn = (p95 - p5) / (p50 + 1e-12) if p50 > 1e-10 else 0.5
            micro_dyn = float(np.clip(micro_dyn, 0.0, 5.0))

            # ── 2. Spektrale Varianz ──────────────────────────────────
            # Wie stark variiert das Spektrum über die Zeit?
            # Berechne Kurzzeit-Spektren und deren Frame-zu-Frame-Varianz
            fft_n = 2048
            hop = fft_n // 4
            n_frames = min(40, max(4, (len(arr) - fft_n) // hop))
            if n_frames >= 4:
                frames = np.array(
                    [np.abs(np.fft.rfft(arr[i * hop : i * hop + fft_n] * np.hamming(fft_n))) for i in range(n_frames)]
                )
                # Frame-zu-Frame Kosinus-Distanz der Spektral-Hüllkurve
                spectral_var = 0.0
                for i in range(1, n_frames):
                    a, b = frames[i - 1], frames[i]
                    denom = np.sqrt(np.sum(a**2) * np.sum(b**2)) + 1e-12
                    cos_sim = float(np.dot(a, b) / denom)
                    spectral_var += 1.0 - cos_sim
                spectral_var /= max(1, n_frames - 1)
                spectral_var = float(np.clip(spectral_var, 0.0, 1.0))
            else:
                spectral_var = 0.5

            # ── 3. Stärke berechnen ───────────────────────────────────
            # Hohe Mikrodynamik + hohe spektrale Varianz → Material ist
            # bereits lebendig → NIEDRIGE Stärke.
            # Niedrige Werte → Material ist „steril" → HÖHERE Stärke.

            # Normalisiere Mikrodynamik: 0.0 (flach) … 1.0 (sehr lebendig)
            dyn_norm = float(np.clip(1.0 - micro_dyn / 3.0, 0.0, 1.0))
            # Spektrale Varianz: 0.0 (statisch) … 1.0 (variabel)
            spec_norm = float(np.clip(1.0 - spectral_var / 0.3, 0.0, 1.0))

            # Sterilitäts-Index: 0 = lebendig, 1 = steril
            sterility = 0.55 * dyn_norm + 0.45 * spec_norm
            sterility = float(np.clip(sterility, 0.0, 1.0))

            # Mapping auf Stärke-Bereich
            strength = HumanizationPass._STRENGTH_MIN + sterility * (
                HumanizationPass._STRENGTH_MAX - HumanizationPass._STRENGTH_MIN
            )
            strength = float(np.clip(strength, HumanizationPass._STRENGTH_MIN, HumanizationPass._STRENGTH_MAX))

            logger.debug(
                "HumanizationPass.calibrate_strength: dyn=%.3f spec=%.3f sterility=%.3f → strength=%.3f",
                micro_dyn,
                spectral_var,
                sterility,
                strength,
            )
            return strength

        except Exception as exc:
            logger.debug("§V6 HumanizationPass.calibrate_strength fehlgeschlagen — Default-Stärke zurückgegeben (%.2f): %s", HumanizationPass._STRENGTH_DEFAULT, exc)
            return HumanizationPass._STRENGTH_DEFAULT

    @staticmethod
    def apply(audio: np.ndarray, sr: int, strength: float | None = None, *, is_studio_2026: bool = False) -> np.ndarray:
        """Wendet Humanization auf das restaurierte Audio an.

        Args:
            audio: float32, shape=(channels, samples) oder (samples,)
            sr: sample rate
            strength: 0.0–1.0, Intensität.
                None → automatische Kalibrierung via calibrate_strength().
                0.15 → bisheriger Default (manuell).
            is_studio_2026: Wenn True → mutigere Stärke. False = Restoration → konservativ.

        Returns:
            Humanisiertes Audio, selbe Shape.
        """
        if strength is None:
            strength = HumanizationPass.calibrate_strength(audio, sr)
            # §v10.15: Restoration-Mode → konservativer (×0.7)
            # Studio-Mode → mutiger (×1.15)
            if not is_studio_2026:
                strength *= 0.70
            else:
                strength *= 1.15
            strength = float(np.clip(strength, 0.03, 0.30))
        try:
            audio_f = np.asarray(audio, dtype=np.float32)
            if audio_f.ndim == 2:
                result = audio_f.copy()
                for ch in range(audio_f.shape[0]):
                    result[ch] = HumanizationPass._process_channel(audio_f[ch], sr, strength)
                return cast(np.ndarray, result)
            return HumanizationPass._process_channel(audio_f, sr, strength)
        except Exception as e:
            logger.warning("klang_guards.py::anwenden Ersatzpfad: %s", e)
            return audio

    @staticmethod
    def _process_channel(channel: np.ndarray, sr: int, strength: float) -> np.ndarray:
        """Pro-Kanal Humanization."""
        n = len(channel)
        # §v10.17: Deterministischer Seed aus Audio-Hash — gleicher Input = gleiches Output
        _seed = int(sum(channel.view(np.uint8).reshape(-1, 4).mean(axis=1)) * 1e6) % (2**31)
        rng = np.random.default_rng(_seed)
        # 1. Mikro-Amplituden-Modulation (0–0.3 dB, langsam, ~0.5 Hz)
        t = np.linspace(0, n / sr, n, endpoint=False)
        amp_mod = 1.0 + strength * 0.02 * np.sin(2.0 * np.pi * 0.47 * t + rng.random() * np.pi)
        # 2. Phasen-Jitter: allpass mit zufälligem, sehr kurzem Delay
        delay_samples = max(1, int(sr * 0.0003))
        rng.uniform(0.3, 0.7)
        # Einfacher 1. Ordnung allpass
        g = strength * 0.003
        out = np.zeros_like(channel)
        buf = 0.0
        for i in range(n):
            _delayed = channel[i - delay_samples] if i >= delay_samples else 0.0
            # Allpass: y[n] = -g * x[n] + x[n-1] + g * y[n-1]
            x_del = channel[i - 1] if i > 0 else 0.0
            out[i] = -g * channel[i] + x_del + g * buf
            buf = out[i]
        # Blend
        result = channel * (1.0 - strength * 0.05) + out * strength * 0.05
        result *= amp_mod
        return cast(np.ndarray, (np.clip(result, -1.0, 1.0).astype(np.float32)))
