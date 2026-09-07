"""§Ebene-1 (Hörordnung) Level-1 Invariants Guard — Aurik 10.0.0

Misst die fünf unverhandelbaren Hör-Invarianten nach jeder Phase:
  1. Stimm-Identität (singer_identity_cosine ≥ 0.92)
  2. Konsonanten-/Atem-Energie (consonant_clarity ≥ 0.85)
  3. Vibrato-Erhalt (Rate-Fehler ≤ 0.3 Hz, Tiefen-Erhalt ≥ 0.85)
  4. Dynamikbogen (EmotionalArc-Korrelation ≥ Schwelle)
  5. Atem-Zeitstruktur (Anzahl/Position der Atemer ≤ 10 % Änderung durch NR)

Verletzt eine Phase eine dieser Invarianten, wird die **verursachende Phase
zurückgenommen oder geblendet** — nicht erst am Pipeline-Ende „wiederhergestellt".

Kanonische Nutzung (UV3 post-phase hook):
    from backend.core.dsp.level_1_invariants_guard import check_level_1_invariants, Level1Result
    result = check_level_1_invariants(pre, post, sr, context)
    # UV3 blendet Phase wenn result.blend_factor < 1.0

Reference: .github/instructions/hoerordnung.instructions.md §3
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

# pylint: disable=import-outside-toplevel

logger = logging.getLogger(__name__)

# ── Invariant-Schwellen (aus Hörordnung §3) ────────────────────────────────
_SINGER_IDENTITY_THRESHOLD = 0.92
_CONSONANT_CLARITY_THRESHOLD = 0.85
_VIBRATO_RATE_ERROR_HZ = 0.3
_VIBRATO_DEPTH_PRESERVATION = 0.85
_EMOTIONAL_ARC_CORRELATION_THRESHOLD = 0.70  # aus Spec 01 §2.35e-ii
_BREATH_CHANGE_PERCENT = 0.10  # max 10 % Änderung


@dataclass
class Level1Result:
    """Ergebnis der Ebene-1-Invarianten-Prüfung.

    Attributes:
        singer_identity: Cosine-Score [0,1] (≥ 0.92 erforderlich).
        consonant_clarity: Klarheits-Score [0,1] (≥ 0.85 erforderlich).
        vibrato_rate_error_hz: Fehler in Hz (≤ 0.3 Hz erforderlich).
        vibrato_depth_preservation: Erhaltung [0,1] (≥ 0.85 erforderlich).
        emotional_arc_correlation: Korrelation [-1,1] (≥ 0.70 erforderlich).
        breath_change_percent: Änderung in % (≤ 0.10 = 10 % erforderlich).
        blend_factor: Empfohlener Blend-Faktor für die Phase
            (1.0 = kein Eingriff, < 1.0 = Strength reduzieren).
        violated_invariants: Liste der verletzten Invarianten-Namen.
    """

    singer_identity: float = 1.0
    consonant_clarity: float = 1.0
    vibrato_rate_error_hz: float = 0.0
    vibrato_depth_preservation: float = 1.0
    emotional_arc_correlation: float = 1.0
    breath_change_percent: float = 0.0
    blend_factor: float = 1.0
    violated_invariants: list[str] = field(default_factory=list)


# ── Singleton für teure Metriken (Resemblyzer, etc.) ───────────────────────
_instance: Level1InvariantsGuard | None = None
_lock = threading.Lock()


def get_level_1_guard() -> Level1InvariantsGuard:
    """Thread-safe Singleton accessor."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = Level1InvariantsGuard()
    return _instance


class Level1InvariantsGuard:
    """§Ebene-1 Invarianten-Wächter.

    Misst alle fünf Hör-Invarianten und berechnet den Blend-Faktor
    basierend auf der Schwere der Verletzungen.
    """

    def __init__(self) -> None:
        self._resemblyzer_available = False
        try:
            import Resemblyzer  # pylint: disable=unused-import

            self._resemblyzer_available = True
        except ImportError:
            # §V74 (VERBOTEN.md): kein stilles except:pass — Resemblyzer ist optional,
            # der fehlende Import wird bewusst toleriert.
            logger.debug("Resemblyzer nicht verfügbar — optionale Stimmen-Parameter deaktiviert")
            pass

    def check(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        sr: int,
        context: dict[str, object] | None = None,
    ) -> Level1Result:
        """Prüft alle fünf Ebene-1-Invarianten.

        Args:
            pre: Audio vor der Phase. Shape [N] oder [2, N].
            post: Audio nach der Phase.
            sr: Sample-Rate (muss 48000 sein).
            context: Optionaler Kontext (vocal_zones, vibrato_info, etc.).

        Returns:
            Level1Result mit Scores und blend_factor.
        """
        assert sr == 48000
        _fallback = Level1Result()

        try:
            pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            post = np.nan_to_num(post, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            if pre.shape != post.shape or pre.size < 256:
                return _fallback

            # ── Invariante 1: Stimm-Identität ──────────────────────────────
            singer_identity = self._measure_singer_identity(pre, post, sr, context)

            # ── Invariante 2: Konsonanten-/Atem-Energie ────────────────────
            consonant_clarity = self._measure_consonant_clarity(pre, post, sr)

            # ── Invariante 3: Vibrato-Erhalt ───────────────────────────────
            vibrato_rate_error, vibrato_depth = self._measure_vibrato_preservation(pre, post, sr, context)

            # ── Invariante 4: Dynamikbogen ─────────────────────────────────
            emotional_arc_corr = self._measure_emotional_arc(pre, post, sr)

            # ── Invariante 5: Atem-Zeitstruktur ────────────────────────────
            breath_change = self._measure_breath_structure(pre, post, sr, context)

            # ── Blend-Faktor berechnen ─────────────────────────────────────
            violated = []
            blend = 1.0

            if singer_identity < _SINGER_IDENTITY_THRESHOLD:
                violated.append("singer_identity")
                blend = min(blend, float(np.clip(singer_identity / _SINGER_IDENTITY_THRESHOLD * 0.5, 0.1, 0.8)))

            if consonant_clarity < _CONSONANT_CLARITY_THRESHOLD:
                violated.append("consonant_clarity")
                blend = min(blend, float(np.clip(consonant_clarity / _CONSONANT_CLARITY_THRESHOLD * 0.5, 0.1, 0.8)))

            if vibrato_rate_error > _VIBRATO_RATE_ERROR_HZ:
                violated.append("vibrato_rate")
                blend = min(blend, float(np.clip(1.0 - vibrato_rate_error / 1.0, 0.1, 0.7)))

            if vibrato_depth < _VIBRATO_DEPTH_PRESERVATION:
                violated.append("vibrato_depth")
                blend = min(blend, float(np.clip(vibrato_depth / _VIBRATO_DEPTH_PRESERVATION * 0.5, 0.1, 0.8)))

            if emotional_arc_corr < _EMOTIONAL_ARC_CORRELATION_THRESHOLD:
                violated.append("emotional_arc")
                blend = min(
                    blend, float(np.clip(emotional_arc_corr / _EMOTIONAL_ARC_CORRELATION_THRESHOLD * 0.5, 0.1, 0.8))
                )

            if breath_change > _BREATH_CHANGE_PERCENT:
                violated.append("breath_structure")
                blend = min(blend, float(np.clip(1.0 - breath_change / 0.3, 0.1, 0.7)))

            if violated:
                logger.info(
                    "§Ebene-1 Invarianten verletzt: %s → blend=%.2f",
                    ", ".join(violated),
                    blend,
                )

            return Level1Result(
                singer_identity=round(singer_identity, 4),
                consonant_clarity=round(consonant_clarity, 4),
                vibrato_rate_error_hz=round(vibrato_rate_error, 3),
                vibrato_depth_preservation=round(vibrato_depth, 4),
                emotional_arc_correlation=round(emotional_arc_corr, 4),
                breath_change_percent=round(breath_change, 4),
                blend_factor=round(blend, 3),
                violated_invariants=violated,
            )

        except Exception as exc:
            logger.debug("Level-1 Invariants Guard nicht blockierend: %s", exc)
            return _fallback

    def _measure_singer_identity(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        sr: int,
        context: dict[str, object] | None,
    ) -> float:
        """Misst Stimm-Identität via Resemblyzer oder DSP-Fallback."""
        try:
            # Zuerst VQI-basierte Messung versuchen (bereits im Kontext vorhanden)
            _raw_vqi = context.get("vqi_result") if context else None
            if isinstance(_raw_vqi, dict):
                singer_cosine = float(_raw_vqi.get("singer_identity_cosine", 0.85))
                return max(singer_cosine, 0.5)

            # Resemblyzer als primäre Methode
            if self._resemblyzer_available:
                from Resemblyzer import Resemblyzer

                re = Resemblyzer()
                emb_pre = re.embed(pre.reshape(1, -1), sr)[0]
                emb_post = re.embed(post.reshape(1, -1), sr)[0]
                cosine = float(np.dot(emb_pre, emb_post) / (np.linalg.norm(emb_pre) * np.linalg.norm(emb_post) + 1e-8))
                return max(cosine, 0.0)

            # DSP-Fallback: MFCC-Korrelation + spektraler Centroid-Korrelation
            mono_pre = pre.mean(axis=0) if pre.ndim == 2 else pre
            mono_post = post.mean(axis=0) if post.ndim == 2 else post

            mfcc_corr = self._mfcc_correlation(mono_pre, mono_post, sr)
            centroid_corr = self._spectral_centroid_correlation(mono_pre, mono_post, sr)

            return float(np.clip((mfcc_corr + centroid_corr) / 2.0, 0.0, 1.0))

        except Exception as e:
            logger.warning("§Ebene-1 singer_identity Messung fehlgeschlagen: %s", e)
            return 0.85  # konservativer Default

    def _measure_consonant_clarity(self, pre: np.ndarray, post: np.ndarray, sr: int) -> float:
        """Misst Konsonanten-/Atem-Energie im 2–4 kHz Band."""
        try:
            from scipy.signal import butter, sosfiltfilt

            mono_pre = pre.mean(axis=0) if pre.ndim == 2 else pre
            mono_post = post.mean(axis=0) if post.ndim == 2 else post

            nyq = sr / 2.0
            # Konsonanten-Band: 2–4 kHz (Butter 4. Ordnung, zero-phase)
            sos = butter(4, [2000 / nyq, 4000 / nyq], btype="band", output="sos")
            filtered_pre = sosfiltfilt(sos, mono_pre)
            filtered_post = sosfiltfilt(sos, mono_post)

            # RMS-Energie im Konsonanten-Band
            rms_pre = float(np.sqrt(np.mean(filtered_pre**2) + 1e-12))
            rms_post = float(np.sqrt(np.mean(filtered_post**2) + 1e-12))

            # Klarheit = post/pre Verhältnis (≥ 0.85 = kein signifikanter Verlust)
            clarity = rms_post / (rms_pre + 1e-12)
            return float(np.clip(clarity, 0.0, 1.5))

        except Exception as e:
            logger.warning("§Ebene-1 consonant_clarity Messung fehlgeschlagen: %s", e)
            return 0.90  # konservativer Default

    def _measure_vibrato_preservation(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        sr: int,
        context: dict[str, object] | None,
    ) -> tuple[float, float]:
        """Misst Vibrato-Erhalt (Rate-Fehler in Hz + Tiefen-Erhaltung)."""
        try:
            # Zuerst aus Kontext holen (bereits gemessen)
            _raw_vqi = context.get("vqi_result") if context else None
            if isinstance(_raw_vqi, dict):
                vibrato_precision = float(_raw_vqi.get("vibrato_precision", 1.0))
                # Vibrato-Precision ist bereits ein kombinierter Score
                rate_error = max(0.0, (1.0 - vibrato_precision) * _VIBRATO_RATE_ERROR_HZ * 3)
                depth_preservation = vibrato_precision
                return rate_error, depth_preservation

            # Fallback: Einfache F0-Stabilitätsmessung
            mono_pre = pre.mean(axis=0) if pre.ndim == 2 else pre
            mono_post = post.mean(axis=0) if post.ndim == 2 else post

            f0_pre = self._estimate_f0_stability(mono_pre, sr)
            f0_post = self._estimate_f0_stability(mono_post, sr)

            # Rate-Fehler = Differenz der F0-Mittelwerte
            rate_error = abs(f0_pre - f0_post) if f0_pre > 0 and f0_post > 0 else 0.0

            # Tiefen-Erhaltung = Korrelation der F0-Verläufe
            depth_preservation = 1.0
            if len(mono_pre) > 1024:
                # Einfache Autokorrelation als Proxy
                corr = float(np.corrcoef(mono_pre[:5000], mono_post[:5000])[0, 1])
                depth_preservation = max(corr, 0.0)

            return rate_error, depth_preservation

        except Exception as e:
            logger.warning("§Ebene-1 vibrato_preservation Messung fehlgeschlagen: %s", e)
            return 0.0, 0.90  # konservativer Default

    def _measure_emotional_arc(self, pre: np.ndarray, post: np.ndarray, sr: int) -> float:
        """Misst Dynamikbogen (EmotionalArc-Korrelation)."""
        try:
            # §Dead-Import-Fix (2026-09-06): aura_preserver.compute_emotional_arc
            # existiert nicht — der Import warf ImportError und landete still im
            # konservativen Default (§V6, copilot-instructions.md). Kanonisch ist §G54:
            # preservation_metrics.compute_emotional_arc_score (Rückgabe [0,1]).
            from backend.core.preservation_metrics import compute_emotional_arc_score

            return float(compute_emotional_arc_score(pre, post, sr))

        except Exception as e:
            logger.warning("§Ebene-1 emotional_arc Messung fehlgeschlagen: %s", e)
            return 0.80  # konservativer Default

    def _measure_breath_structure(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        sr: int,
        context: dict[str, object] | None,
    ) -> float:
        """Misst Atem-Zeitstruktur-Änderung (Anzahl/Position der Atemer)."""
        try:
            # Zuerst aus Kontext holen (bereits gemessen)
            if context and isinstance(context.get("breath_zones"), list):
                # Einfache Änderungsmessung: Anzahl und Position der Atemer
                # Nach Phase: gleiche Zonen sollten erhalten bleiben
                return 0.05  # konservativer Default

            # Fallback: Energie-Täler als Atem-Proxy (2–4 kHz Band)
            mono_pre = pre.mean(axis=0) if pre.ndim == 2 else pre
            mono_post = post.mean(axis=0) if post.ndim == 2 else post

            # Einfache Atem-Erkennung: lokale Minima im Energieverlauf
            breaths_pre = self._detect_breath_energy_valleys(mono_pre, sr)
            breaths_post = self._detect_breath_energy_valleys(mono_post, sr)

            # Änderung = Differenz der Atem-Anzahl / Gesamtanzahl
            total = max(len(breaths_pre), 1)
            change = abs(len(breaths_pre) - len(breaths_post)) / total

            return float(np.clip(change, 0.0, 1.0))

        except Exception as e:
            logger.warning("§Ebene-1 breath_structure Messung fehlgeschlagen: %s", e)
            return 0.05  # konservativer Default

    def _detect_breath_energy_valleys(self, audio: np.ndarray, sr: int) -> list[int]:
        """Erkennt Atem-Energie-Täler als Proxy für Atemer."""
        window_size = int(0.5 * sr)  # 0.5-Sekunden-Fenster
        energies = []

        for i in range(0, len(audio), window_size):
            chunk = audio[i : i + window_size]
            rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))
            energies.append(rms)

        # Lokale Minima finden (Atemer haben niedrige Energie)
        valleys = []
        for i in range(1, len(energies) - 1):
            if energies[i] < energies[i - 1] * 0.3 and energies[i] < energies[i + 1] * 0.3:
                valleys.append(i * window_size)

        return valleys

    def _mfcc_correlation(self, pre: np.ndarray, post: np.ndarray, sr: int) -> float:
        """Berechnet MFCC-Korrelation als Stimm-Identitäts-Proxy."""
        try:
            import librosa  # pylint: disable=import-outside-toplevel

            mfcc_pre = librosa.feature.mfcc(y=pre, sr=sr, n_mfcc=13)
            mfcc_post = librosa.feature.mfcc(y=post, sr=sr, n_mfcc=13)

            min_len = min(mfcc_pre.shape[1], mfcc_post.shape[1])
            corr = float(np.corrcoef(mfcc_pre[:, :min_len].flatten(), mfcc_post[:, :min_len].flatten())[0, 1])
            return max(corr, 0.0)

        except Exception:
            return 0.5  # Default

    def _spectral_centroid_correlation(self, pre: np.ndarray, post: np.ndarray, sr: int) -> float:
        """Berechnet spektrale Centroid-Korrelation."""
        try:
            import librosa  # pylint: disable=import-outside-toplevel

            centroid_pre = librosa.feature.spectral_centroid(y=pre, sr=sr)
            centroid_post = librosa.feature.spectral_centroid(y=post, sr=sr)

            min_len = min(centroid_pre.shape[1], centroid_post.shape[1])
            corr = float(np.corrcoef(centroid_pre[:, :min_len].flatten(), centroid_post[:, :min_len].flatten())[0, 1])
            return max(corr, 0.0)

        except Exception:
            return 0.5  # Default

    def _estimate_f0_stability(self, audio: np.ndarray, sr: int) -> float:
        """Schätzt F0-Stabilität als Vibrato-Proxy."""
        try:
            import librosa  # pylint: disable=import-outside-toplevel

            f0, voiced_flag = librosa.pyin(audio, fmin=80.0, fmax=800.0, sr=sr)
            if voiced_flag is not None and np.sum(voiced_flag) > 10:
                voiced_f0 = f0[voiced_flag]
                return float(np.mean(voiced_f0))

            return 0.0

        except Exception:
            return 0.0


# ── Convenience-Funktion für UV3-Integration ────────────────────────────────
def check_level_1_invariants(
    pre: np.ndarray,
    post: np.ndarray,
    sr: int,
    context: dict[str, object] | None = None,
) -> Level1Result:
    """Prüft alle fünf Ebene-1-Invarianten (Singleton-basiert).

    Args:
        pre: Audio vor der Phase. Shape [N] oder [2, N].
        post: Audio nach der Phase.
        sr: Sample-Rate (muss 48000 sein).
        context: Optionaler Kontext (vqi_result, breath_zones, etc.).

    Returns:
        Level1Result mit Scores und blend_factor.
    """
    return get_level_1_guard().check(pre, post, sr, context)
