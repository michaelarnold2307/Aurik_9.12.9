"""
§v10 Aurik Completion Engine — Alle 8 Optimierungen vereint.

1.  SWEET_SPOT_FIXES: Echte Metrik-Korrekturen (nicht *0.98)
2.  PMGG_HPE_WRAP: HPE-Check pro Phase via Phase-Monkey-Patch
3.  SONG_LEARNING: Genre-Medium-Profil-Persistenz
4.  RETRY_DIFFERENT: Alternativ-Plugin-Verdrahtung
5.  AB_COMPARE: A/B-Vergleich an kritischen Defektstellen
6.  BOUNDARY_ACTIVE: Grenzwert-Optimizer im DSP-Pfad
7.  DYNAMIC_EQ: Frequenz-Korrekturen via Dynamic EQ
8.  EARLY_STOP: Phase-Budget nach 60% mit SweetSpot-Check

SOLID: Jede Optimierung ist optional und fällt bei Fehler auf No-Op zurück.
Keine negativen Seiteneffekte auf bestehenden Code.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

MATERIAL_EXPECTED_BW = 20000.0

# ═══════════════════════════════════════════════════════════════════════════
# 1. SWEET_SPOT_FIXES — Echte Korrekturen pro Metrik
# ═══════════════════════════════════════════════════════════════════════════


def apply_sweet_spot_fix(audio: np.ndarray, sr: int, metric: str, current_value: float, target: float) -> np.ndarray:
    """Wendet eine METRIK-SPEZIFISCHE Korrektur an.

    Args:
        audio:         Aktuelles Audio (float64)
        sr:            Sample-Rate
        metric:        Welche Metrik soll verbessert werden?
        current_value: Aktueller Wert der Metrik
        target:        Zielwert

    Returns:
        Korrigiertes Audio (oder unverändert bei Fehler)
    """
    arr = audio.copy()
    try:
        gap = max(target - current_value, 0.0)

        if metric == "hpe":
            # HPE verbessern: Sanfte Loudness-Normalisierung + leichte EQ-Glättung
            from scipy.signal import butter, sosfiltfilt

            rms = float(np.sqrt(np.mean(arr**2)) + 1e-12)
            target_rms = 0.15  # Angenehmer Pegel
            if rms < 0.05:
                arr *= min(target_rms / rms, 2.0)
            elif rms > 0.4:
                arr *= 0.85
            # Sanfte Höhenanhebung wenn dumpf
            sos = butter(2, 6000 / (sr / 2), btype="high", output="sos")
            shelf = sosfiltfilt(sos, arr) * 0.02 * gap
            arr += shelf

        elif metric == "inviting":
            # Inviting verbessern: Peaks dämpfen, Bass balancieren
            from scipy.signal import butter, sosfiltfilt

            if gap > 0.2:
                # Frequenz-Peaks dämpfen via sanftem Lowpass
                sos = butter(3, 12000 / (sr / 2), btype="low", output="sos")
                arr = sosfiltfilt(sos, arr)

        elif metric == "transparency":
            # Transparenz: Spektrale Glättung gegen wässrige Artefakte
            from scipy.ndimage import uniform_filter1d

            # Sehr sanfte Glättung (Fenster 3 Samples)
            arr_float = arr.astype(np.float64)
            arr = uniform_filter1d(arr_float, size=3).astype(np.float64)

        elif metric == "goosebumps":
            # Gänsehaut: Dynamik etwas erweitern für mehr Kontrast
            arr_centered = arr - np.mean(arr)
            arr = np.mean(arr) + arr_centered * (1.0 + gap * 0.3)

        elif metric in ("comb_filter", "comb"):
            # Kammfilter: Phase leicht rotieren um Notches zu verschieben
            phase_shift = int(sr * 0.001 * gap * 10)  # 1-10ms
            if phase_shift > 0:
                arr = np.roll(arr, phase_shift)

        elif metric in ("musical_compression", "compress"):
            # Böse Kompression: Leichte Expansion (Krestfaktor erhöhen)
            rms = float(np.sqrt(np.mean(arr**2)) + 1e-12)
            peak = float(np.max(np.abs(arr)) + 1e-12)
            crest = peak / rms
            if crest < 3.0:  # Sehr flach
                arr_centered = arr - np.mean(arr)
                arr = np.mean(arr) + arr_centered * (1.0 + gap * 0.5)

        elif metric in ("masking_health", "masking"):
            # Masking reduzieren: Leichte Multiband-Trennung
            from scipy.signal import butter, sosfiltfilt

            sos_low = butter(2, 300 / (sr / 2), btype="low", output="sos")
            sos_mid = butter(2, [300 / (sr / 2), 3000 / (sr / 2)], btype="bandpass", output="sos")
            low = sosfiltfilt(sos_low, arr)
            mid = sosfiltfilt(sos_mid, arr)
            high = arr - low - mid
            # Reduziere Überlappung
            arr = low + mid * 0.95 + high * 0.90

    except Exception as e:
        logger.debug("SweetSpot fix %s fehlgeschlagen: %s", metric, e)

    return cast(np.ndarray, (np.clip(arr, -1.0, 1.0)))


# ═══════════════════════════════════════════════════════════════════════════
# 2. PMGG_HPE_WRAP — HPE-Monkey-Patch für UV3-Phasen
# ═══════════════════════════════════════════════════════════════════════════


class PMGGHPEMonkeyPatch:
    """Wrappt PMGG-Aufrufe mit HPE-Check.

    Wird VOR UV3.restore() aktiviert und danach deaktiviert.
    Jeder PMGG.pre_check/post_check-Aufruf wird auf HPE-Delta geprüft.
    """

    _active = False
    _hpe_baseline = 0.5
    _hpe_current = 0.5
    _phase_count = 0
    _hpe_drops = 0

    @classmethod
    def activate(cls, audio: np.ndarray, sr: int) -> None:
        """Aktiviert HPE-Monitoring für UV3-Phasen."""
        try:
            from backend.core.human_pleasantness_estimator import compute_pleasantness

            cls._hpe_baseline = compute_pleasantness(audio, sr).score
            cls._hpe_current = cls._hpe_baseline
            cls._phase_count = 0
            cls._hpe_drops = 0
            cls._active = True
            logger.info("PMGG-HPE-Wrap AKTIV (baseline=%.3f)", cls._hpe_baseline)
        except Exception:
            cls._active = False

    @classmethod
    def deactivate(cls) -> dict:
        """Deaktiviert und gibt Statistik zurück."""
        cls._active = False
        stats = {
            "phases": cls._phase_count,
            "hpe_drops": cls._hpe_drops,
            "hpe_baseline": cls._hpe_baseline,
            "hpe_final": cls._hpe_current,
        }
        return stats

    @classmethod
    def check_phase(cls, audio: np.ndarray, sr: int, phase_name: str) -> tuple[bool, str]:
        """Prüft HPE-Delta nach einer Phase. Gibt (continue, reason) zurück."""
        if not cls._active:
            return True, ""

        try:
            from backend.core.human_pleasantness_estimator import compute_pleasantness

            new_hpe = compute_pleasantness(audio, sr).score
            delta = new_hpe - cls._hpe_current
            cls._hpe_current = new_hpe
            cls._phase_count += 1

            if delta < -0.03:
                cls._hpe_drops += 1
                return False, f"HPE {delta:+.3f} in {phase_name}"
            return True, f"HPE {delta:+.3f}"
        except Exception as e:
            logger.warning("aurik_completion_engine.py::Pruefung_Verarbeitungsschritt Ersatzpfad: %s", e)
            return True, ""


# ═══════════════════════════════════════════════════════════════════════════
# 3. SONG_LEARNING — Genre-Medium-Profil-Persistenz
# ═══════════════════════════════════════════════════════════════════════════

SONG_PROFILE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "song_profiles")


@dataclass
class SongProfile:
    genre: str = "unknown"
    medium: str = "unknown"
    optimal_nr_strength: float = 0.5
    optimal_eq_presence: float = 0.0
    optimal_compression_ratio: float = 1.5
    success_count: int = 0
    avg_hpe_delta: float = 0.0
    last_used: float = 0.0


def load_song_profile(genre: str, medium: str) -> SongProfile:
    """Lädt gespeichertes Profil für Genre+Medium-Kombination."""
    key = f"{genre}_{medium}".lower().replace(" ", "_")
    path = os.path.join(SONG_PROFILE_DIR, f"{key}.json")
    try:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            return SongProfile(**data)
    except Exception as e:
        logger.warning("aurik_completion_engine.py::laden_song_Profil Ersatzpfad: %s", e)
    return SongProfile(genre=genre, medium=medium)


def save_song_profile(profile: SongProfile) -> None:
    """Speichert optimiertes Profil für Genre+Medium."""
    os.makedirs(SONG_PROFILE_DIR, exist_ok=True)
    key = f"{profile.genre}_{profile.medium}".lower().replace(" ", "_")
    path = os.path.join(SONG_PROFILE_DIR, f"{key}.json")
    try:
        with open(path, "w") as f:
            json.dump(
                {
                    "genre": profile.genre,
                    "medium": profile.medium,
                    "optimal_nr_strength": profile.optimal_nr_strength,
                    "optimal_eq_presence": profile.optimal_eq_presence,
                    "optimal_compression_ratio": profile.optimal_compression_ratio,
                    "success_count": profile.success_count,
                    "avg_hpe_delta": profile.avg_hpe_delta,
                    "last_used": time.time(),
                },
                f,
                indent=2,
            )
    except Exception as e:
        logger.debug("Song Profil speichern fehlgeschlagen: %s", e)


def update_song_profile(
    genre: str,
    medium: str,
    hpe_delta: float,
    nr_strength: float = 0.5,
    eq_presence: float = 0.0,
    comp_ratio: float = 1.5,
) -> None:
    """Aktualisiert Profil nach erfolgreicher Restaurierung."""
    profile = load_song_profile(genre, medium)
    n = profile.success_count + 1
    profile.success_count = n
    profile.avg_hpe_delta = (profile.avg_hpe_delta * (n - 1) + hpe_delta) / n
    # Exponentiell gleitender Mittelwert für Parameter
    alpha = 0.3
    profile.optimal_nr_strength = profile.optimal_nr_strength * (1 - alpha) + nr_strength * alpha
    profile.optimal_eq_presence = profile.optimal_eq_presence * (1 - alpha) + eq_presence * alpha
    profile.optimal_compression_ratio = profile.optimal_compression_ratio * (1 - alpha) + comp_ratio * alpha
    save_song_profile(profile)
    logger.info(
        "Song-Learning: %s/%s n=%d avgHPE=%.3f NR=%.2f EQ=%.2f Comp=%.2f",
        genre,
        medium,
        profile.success_count,
        profile.avg_hpe_delta,
        profile.optimal_nr_strength,
        profile.optimal_eq_presence,
        profile.optimal_compression_ratio,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. RETRY_DIFFERENT — Alternativ-Plugins
# ═══════════════════════════════════════════════════════════════════════════

ALTERNATIVE_PLUGINS_ACTIVE = {
    "denoise+declick": [
        ("deepfilternet", "DeepFilterNet"),
        ("resemble_enhance", "ResembleEnhance"),
        ("mp_senet", "MpSenet"),
        ("wpe", "WPE"),
    ],
    "declip": [("mdx23c", "MDX23C"), ("bs_roformer", "BSRoFormer")],
    "source_separation": [("demucs", "HTDemucs"), ("uvr_mdxnet", "UVRMDXNet"), ("gacela", "GACELA")],
    "mastering_chain": [("matchering", "Matchering"), ("panns", "PANNs")],
}


def try_alternative_plugin(operation: str, retry_count: int, pipeline: Any = None) -> tuple[str, Any] | None:
    """RETRY_DIFFERENT: Findet und initialisiert Alternativ-Plugin.

    Returns (plugin_name, plugin_instance) oder None.
    """
    alts = ALTERNATIVE_PLUGINS_ACTIVE.get(operation, [])
    if retry_count >= len(alts):
        return None

    alt_name, alt_class_name = alts[retry_count]
    logger.info("Wiederholung_DIFFERENT: %s -> %s (versuch %d)", operation, alt_name, retry_count + 1)

    # Versuche Plugin zu laden
    try:
        module_name = f"plugins.{alt_name}_plugin"
        class_name = f"{alt_class_name}Plugin"
        mod = __import__(module_name, fromlist=[class_name])
        plugin_cls = getattr(mod, class_name)
        instance = plugin_cls()
        return alt_name, instance
    except Exception as e:
        logger.debug("Wiederholung_DIFFERENT %s not verfuegbar: %s", alt_name, e)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 5. AB_COMPARE — A/B-Vergleich an Defektstellen
# ═══════════════════════════════════════════════════════════════════════════


def ab_compare_defect_sites(
    original: np.ndarray,
    restored: np.ndarray,
    sr: int,
    defect_positions: list[int] | None = None,
) -> dict:
    """Vergleicht Original und restauriert GEZIELT an Defektpositionen.

    Returns dict mit 'improved_sites', 'degraded_sites', 'neutral_sites'.
    """
    if defect_positions is None:
        # Auto-Detektion: Finde Maximal-Differenz-Stellen
        diff = np.abs(original[: len(restored)] - restored[: len(original)])
        win = int(0.05 * sr)  # 50ms Fenster
        positions = []
        for i in range(win, len(diff) - win, win):
            local_max = np.argmax(diff[i : i + win]) + i
            if diff[local_max] > 0.01:
                positions.append(local_max)
        # Top-10 Positionen
        if positions:
            importance = [diff[p] for p in positions]
            sorted_positions = sorted(zip(positions, importance), key=lambda x: -x[1])
            defect_positions = [p for p, _ in sorted_positions[:10]]  # type: ignore[misc]
        else:
            defect_positions = [len(original) // 2]  # Mitte als Fallback

    window = int(0.05 * sr)
    results = {"improved": 0, "degraded": 0, "neutral": 0, "sites": []}

    try:
        from backend.core.human_pleasantness_estimator import compare_pleasantness

        for pos in defect_positions:
            start = max(0, pos - window)
            end = min(len(original), pos + window)
            orig_chunk = original[start:end].astype(np.float32)
            rest_chunk = restored[start:end].astype(np.float32)

            cmp = compare_pleasantness(orig_chunk, rest_chunk, sr)
            delta = cmp.get("delta_score", 0.0)
            site_result = "improved" if delta > 0.02 else "degraded" if delta < -0.02 else "neutral"
            results[site_result] += 1  # type: ignore[operator]
            results["sites"].append({"pos": pos, "delta": float(delta), "result": site_result})  # type: ignore[attr-defined]

    except Exception as e:
        logger.debug("AB compare fehlgeschlagen: %s", e)
        results["neutral"] = len(defect_positions)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 6. BOUNDARY_ACTIVE — Grenzwert-Optimizer im DSP-Pfad
# ═══════════════════════════════════════════════════════════════════════════


def find_optimal_intensity(
    audio: np.ndarray,
    sr: int,
    material: str,
    process_fn,  # Callable: fn(audio, intensity) -> audio
    intensity_range: tuple[float, float] = (0.3, 1.0),
    steps: int = 5,
) -> tuple[float, np.ndarray]:
    """Tastet Intensitätsbereich ab und findet Pleasantness-Maximum.

    Args:
        process_fn: Funktion die Audio mit gegebener Intensität verarbeitet
        intensity_range: (min, max) Suchbereich
        steps: Anzahl Abtast-Schritte

    Returns:
        (optimale_intensität, bestes_audio)
    """
    try:
        from backend.core.human_pleasantness_estimator import compute_pleasantness

        best_intensity = intensity_range[0]
        best_audio = audio.copy()
        best_hpe = compute_pleasantness(audio, sr).score

        step_size = (intensity_range[1] - intensity_range[0]) / max(steps - 1, 1)

        for i in range(steps):
            intensity = intensity_range[0] + i * step_size
            try:
                processed = process_fn(audio, intensity)
                hpe = compute_pleasantness(processed.astype(np.float32), sr).score
                if hpe > best_hpe + 0.005:
                    best_hpe = hpe
                    best_intensity = intensity
                    best_audio = processed.copy()
                    logger.debug("BoundaryActive: intensity=%.2f HPE=%.3f (best)", intensity, hpe)
            except Exception:
                continue

        logger.info(
            "BoundaryActive: optimal intensity=%.2f HPE=%.3f (baseline=%.3f)",
            best_intensity,
            best_hpe,
            compute_pleasantness(audio, sr).score,
        )
        return best_intensity, best_audio

    except Exception as e:
        logger.debug("BoundaryActive fehlgeschlagen: %s", e)
        return 0.5, audio


# ═══════════════════════════════════════════════════════════════════════════
# 7. DYNAMIC_EQ — Frequenz-Korrekturen via echten EQ
# ═══════════════════════════════════════════════════════════════════════════


def apply_dynamic_eq(
    audio: np.ndarray,
    sr: int,
    corrections: dict[str, float],
) -> np.ndarray:
    """Wendet Dynamic-EQ-Korrekturen pro Band an.

    Args:
        corrections: dict mit Band-Name -> dB-Korrektur
          {'bass': -2.0, 'presence': -1.5}

    Returns:
        EQ-korrigiertes Audio
    """
    if not corrections:
        return audio

    from scipy.signal import butter, sosfiltfilt

    BAND_FREQS = {
        "sub_bass": (40, "lowshelf"),
        "bass": (150, "lowshelf"),
        "low_mid": (375, "peaking"),
        "mid": (1000, "peaking"),
        "high_mid": (3000, "peaking"),
        "presence": (6000, "highshelf"),
        "brilliance": (12000, "highshelf"),
    }

    result = audio.copy().astype(np.float64)
    for band, gain_db in corrections.items():
        if band not in BAND_FREQS:
            continue
        freq, eq_type = BAND_FREQS[band]

        # Gain in linear: +/-3dB Maximum
        gain_linear = 10.0 ** (np.clip(gain_db, -3.0, 3.0) / 20.0)

        if eq_type == "lowshelf":
            sos = butter(2, freq / (sr / 2), btype="low", output="sos")
            low = sosfiltfilt(sos, result)
            result = low * gain_linear + (result - low)
        elif eq_type == "highshelf":
            sos = butter(2, freq / (sr / 2), btype="high", output="sos")
            high = sosfiltfilt(sos, result)
            result = high * gain_linear + (result - high)
        elif eq_type == "peaking":
            Q = 1.0
            bw = freq / Q
            sos = butter(
                2,
                [max(20, freq - bw / 2) / (sr / 2), min(MATERIAL_EXPECTED_BW, freq + bw / 2) / (sr / 2)],
                btype="bandpass",
                output="sos",
            )
            band_signal = sosfiltfilt(sos, result)
            result = result + band_signal * (gain_linear - 1.0)

    return cast(np.ndarray, (np.clip(result, -1.0, 1.0)))


# ═══════════════════════════════════════════════════════════════════════════
# 8. EARLY_STOP — Adaptiv, kein fixes Prozent-Limit
# ═══════════════════════════════════════════════════════════════════════════

EARLY_STOP_THRESHOLD = 0.008  # HPE < 0.008 = nahezu unhörbar
EARLY_STOP_LOOKBACK = 4  # So viele Phasen rückwärts prüfen
EARLY_STOP_MIN_PHASES = 8  # Mindestens 8 Phasen bevor Stop erlaubt


def should_early_stop(
    hpe_history: list[float],
    total_phases: int,
    completed_phases: int,
) -> tuple[bool, str]:
    """Adaptiver Early-Stop: Nur wenn Verbesserung nahezu unhörbar.

    KEIN fixes Prozent-Limit. Phase 35 mit +0.05 HPE läuft weiter.
    Nur wenn die letzten LOOKBACK Phasen ALLE < THRESHOLD brachten,
    ist das Plateau erreicht und weitere Phasen sind unhörbar.
    """
    if len(hpe_history) < EARLY_STOP_LOOKBACK + 1:
        return False, ""
    if completed_phases < EARLY_STOP_MIN_PHASES:
        return False, ""

    recent = hpe_history[-EARLY_STOP_LOOKBACK:]
    improvements = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
    max_imp = max(abs(imp) for imp in improvements)

    # Wenn eine der letzten Phasen noch > 0.05 brachte → definitiv weitermachen
    if any(imp > 0.05 for imp in improvements):
        return False, ""

    # Alle Verbesserungen < 0.008 → Plateau, unhörbar
    if max_imp < EARLY_STOP_THRESHOLD:
        return True, (
            f"Genug-ist-genug: Letzte {EARLY_STOP_LOOKBACK} Phasen brachten "
            f"max {max_imp:.4f} HPE — unterhalb der Hörschwelle "
            f"({EARLY_STOP_THRESHOLD:.3f}). {completed_phases}/{total_phases} Phasen"
        )

    return False, ""
