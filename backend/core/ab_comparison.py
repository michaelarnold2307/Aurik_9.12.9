"""A/B-Vergleich — Vorher/Nachher-Audio-Vergleich mit Blindtest und Delta.

Spec 14 §14.9, Spec 08 §8.1 AB-Sync-Loop, Spec v10.207 Audio-Player-SOTA.

Bietet:
- ABComparison: synchronisiertes A/B-Umschalten während der Wiedergabe
- ABLoop: wiederholtes Abspielen eines Segments im A/B-Wechsel
- ABBlindTest: randomisierte X-Zuweisung für verblindete Hörtests
- ABDelta: berechnet Differenzsignal und Metriken zwischen A und B
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)


# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class ABSegment:
    """Ein A/B-Vergleichssegment (z.B. 5 Sekunden)."""

    start_sample: int
    end_sample: int
    label: str = ""  # "A" oder "B" (im Blindtest: "X1", "X2")


@dataclass
class ABDeltaResult:
    """Ergebnis der A/B-Delta-Berechnung."""

    rms_delta_db: float = 0.0  # RMS-Differenz in dB
    peak_delta_db: float = 0.0  # Peak-Differenz in dB
    lufs_delta: float = 0.0  # LUFS-Differenz
    spectral_centroid_delta_hz: float = 0.0  # Spektrale-Schwerpunkt-Verschiebung
    correlation: float = 1.0  # Pearson-Korrelation A vs B
    diff_audio: np.ndarray | None = None  # Differenzsignal A - B
    segments_compared: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class ABBlindResult:
    """Ergebnis eines AB-Blindtests."""

    total_trials: int = 0
    correct: int = 0
    p_value: float = 1.0  # Binomialtest-p-Wert
    preference_a: int = 0  # "A klang besser"
    preference_b: int = 0
    no_preference: int = 0
    trials: list[dict[str, Any]] = field(default_factory=list)
    is_significant: bool = False  # p < 0.05


# ── A/B-Vergleich-Klasse ─────────────────────────────────────────────────────


class ABComparison:
    """A/B-Vergleich zwischen Original (A) und restauriertem Audio (B).

    Spec 14 §14.9: Synchronisiertes A/B-Umschalten für präzisen
    Vorher/Nachher-Vergleich. Unterstützt segmentiertes Looping
    und Delta-Berechnung.

    Verwendung:
        abc = ABComparison(original, restored, sr=48000)
        abc.set_segment(start_s=10.0, duration_s=5.0)
        abc.toggle()  # Wechselt zwischen A und B
        delta = abc.compute_delta()
    """

    def __init__(
        self,
        audio_a: np.ndarray,  # Original
        audio_b: np.ndarray,  # Restauriert
        sr: int = 48000,
        *,
        label_a: str = "Original",
        label_b: str = "Restauriert",
    ) -> None:
        self._a = np.asarray(audio_a, dtype=np.float32)
        self._b = np.asarray(audio_b, dtype=np.float32)
        self.sr = sr
        self.label_a = label_a
        self.label_b = label_b

        # Zustand
        self._current_is_a: bool = True
        self._segment_start: int = 0
        self._segment_end: int = min(len(self._a), len(self._b))
        self._loop_enabled: bool = False
        self._loop_count: int = 0
        self._sync_group: int = 0  # Für Sync mit anderen ABComparison-Instanzen

        # Metadaten
        self._toggle_count: int = 0
        self._total_listen_a_s: float = 0.0
        self._total_listen_b_s: float = 0.0
        self._last_toggle_sample: int = 0

        logger.debug(
            "ABComparison erstellt: A=%d samples, B=%d samples, sr=%d",
            len(self._a),
            len(self._b),
            sr,
        )

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def current_is_a(self) -> bool:
        """True wenn aktuell A (Original) aktiv ist."""
        return self._current_is_a

    @property
    def current_audio(self) -> np.ndarray:
        """Gibt das aktuell aktive Audio (A oder B) zurück."""
        return cast(np.ndarray, self._a if self._current_is_a else self._b)

    @property
    def current_label(self) -> str:
        """Label des aktuell aktiven Audios."""
        return self.label_a if self._current_is_a else self.label_b

    @property
    def segment_start_s(self) -> float:
        """Segment-Start in Sekunden."""
        return self._segment_start / self.sr

    @property
    def segment_end_s(self) -> float:
        """Segment-Ende in Sekunden."""
        return self._segment_end / self.sr

    @property
    def segment_duration_s(self) -> float:
        """Segment-Dauer in Sekunden."""
        return (self._segment_end - self._segment_start) / self.sr

    @property
    def is_looping(self) -> bool:
        """True wenn A/B-Loop aktiv ist."""
        return self._loop_enabled

    # ── Steuerung ─────────────────────────────────────────────────────────

    def toggle(self) -> str:
        """Wechselt zwischen A und B. Gibt das neue Label zurück."""
        self._current_is_a = not self._current_is_a
        self._toggle_count += 1
        logger.debug("AB-Toggle → %s (Toggle #%d)", self.current_label, self._toggle_count)
        return self.current_label

    def set_a(self) -> None:
        """Erzwingt A (Original)."""
        self._current_is_a = True
        logger.debug("AB → A (%s)", self.label_a)

    def set_b(self) -> None:
        """Erzwingt B (Restauriert)."""
        self._current_is_a = False
        logger.debug("AB → B (%s)", self.label_b)

    def set_segment(self, start_s: float, duration_s: float | None = None) -> None:
        """Setzt das A/B-Vergleichssegment.

        Args:
            start_s: Startzeit in Sekunden
            duration_s: Dauer in Sekunden (None = bis Ende)
        """
        self._segment_start = max(0, int(start_s * self.sr))
        max_len = min(len(self._a), len(self._b))
        if duration_s is not None:
            self._segment_end = min(self._segment_start + int(duration_s * self.sr), max_len)
        else:
            self._segment_end = max_len
        logger.debug(
            "AB-Segment: %.1fs – %.1fs (%.1fs)",
            self.segment_start_s,
            self.segment_end_s,
            self.segment_duration_s,
        )

    def enable_loop(self, enabled: bool = True) -> None:
        """Aktiviert/deaktiviert den A/B-Loop."""
        self._loop_enabled = enabled
        logger.debug("AB-Loop: %s", "AN" if enabled else "AUS")

    # ── Audio-Zugriff ─────────────────────────────────────────────────────

    def get_segment(self, which: str = "current") -> np.ndarray:
        """Gibt das Audiosegment für A, B oder current zurück.

        Args:
            which: "A", "B", oder "current"
        """
        src = {"A": self._a, "B": self._b, "current": self.current_audio}[which]
        s, e = self._segment_start, min(self._segment_end, len(src))
        return cast(np.ndarray, src[s:e])

    def get_current_slice(self, position_sample: int, length_samples: int) -> np.ndarray:
        """Gibt einen Slice des aktuell aktiven Audios zurück."""
        src = self.current_audio
        start = max(0, position_sample)
        end = min(start + length_samples, len(src))
        result = src[start:end]
        # Wenn am Loop-Ende: wrap-around
        if self._loop_enabled and end >= self._segment_end:
            remaining = length_samples - len(result)
            if remaining > 0:
                wrap = src[self._segment_start : self._segment_start + remaining]
                result = np.concatenate([result, wrap])
        return result

    # ── Delta-Berechnung ──────────────────────────────────────────────────

    def compute_delta(self, segment_only: bool = True) -> ABDeltaResult:
        """Berechnet das Delta zwischen A (Original) und B (Restauriert).

        Args:
            segment_only: Wenn True, nur innerhalb des gesetzten Segments.

        Returns:
            ABDeltaResult mit Differenzmetriken.
        """
        if segment_only:
            a_seg = self.get_segment("A")
            b_seg = self.get_segment("B")
        else:
            a_seg, b_seg = self._a, self._b

        min_len = min(len(a_seg), len(b_seg))
        a_seg = a_seg[:min_len]
        b_seg = b_seg[:min_len]

        result = ABDeltaResult(segments_compared=1)

        diff = a_seg.astype(np.float64) - b_seg.astype(np.float64)
        result.diff_audio = diff.astype(np.float32)

        # RMS-Delta
        rms_a = float(np.sqrt(np.mean(a_seg**2)) + 1e-12)
        rms_b = float(np.sqrt(np.mean(b_seg**2)) + 1e-12)
        result.rms_delta_db = float(20.0 * np.log10(rms_a / rms_b))

        # Peak-Delta
        peak_a = float(np.max(np.abs(a_seg)))
        peak_b = float(np.max(np.abs(b_seg)))
        result.peak_delta_db = float(20.0 * np.log10((peak_a + 1e-12) / (peak_b + 1e-12)))

        # Korrelation
        result.correlation = float(np.corrcoef(a_seg.flat, b_seg.flat)[0, 1])

        # Spektrale-Schwerpunkt-Verschiebung
        try:
            from scipy.signal import periodogram

            f_a, p_a = periodogram(a_seg if a_seg.ndim == 1 else a_seg.mean(axis=1), fs=self.sr, scaling="density")
            _, p_b = periodogram(b_seg if b_seg.ndim == 1 else b_seg.mean(axis=1), fs=self.sr, scaling="density")
            centroid_a = float(np.sum(f_a * p_a) / (np.sum(p_a) + 1e-12))
            centroid_b = float(np.sum(f_a * p_b) / (np.sum(p_b) + 1e-12))
            result.spectral_centroid_delta_hz = centroid_b - centroid_a
        except Exception:
            result.notes.append("spectral_centroid: scipy nicht verfügbar")

        logger.info(
            "AB-Delta: RMS=%.1fdB, Peak=%.1fdB, Corr=%.4f, Centroid=%.0fHz",
            result.rms_delta_db,
            result.peak_delta_db,
            result.correlation,
            result.spectral_centroid_delta_hz,
        )
        return result

    # ── Statistik ─────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Gibt A/B-Vergleichsstatistiken zurück."""
        return {
            "toggle_count": self._toggle_count,
            "total_listen_a_s": self._total_listen_a_s,
            "total_listen_b_s": self._total_listen_b_s,
            "current_is_a": self._current_is_a,
            "segment_start_s": self.segment_start_s,
            "segment_end_s": self.segment_end_s,
            "loop_enabled": self._loop_enabled,
            "label_a": self.label_a,
            "label_b": self.label_b,
        }

    def record_listen_time(self, sample_count: int) -> None:
        """Zeichnet Hörzeit für die aktuell aktive Quelle auf."""
        seconds = sample_count / self.sr
        if self._current_is_a:
            self._total_listen_a_s += seconds
        else:
            self._total_listen_b_s += seconds


# ── A/B-Blindtest ────────────────────────────────────────────────────────────


class ABBlindTest:
    """A/B-Blindtest: Randomisiert A/B-Zuweisung für verblindete Vergleiche.

    Spec 14 §14.9, Spec 15 §15.3.2 ABX-Interface.

    Verwendung:
        bt = ABBlindTest(comparison)
        bt.start_session(n_trials=10)
        for trial in bt.trials:
            choice = user_hears_and_chooses(trial.x_audio)
            bt.record_answer(trial, choice_is_a=True)
        result = bt.get_result()
    """

    def __init__(self, comparison: ABComparison, *, seed: int | None = None) -> None:
        self._cmp = comparison
        self._rng = random.Random(seed)
        self._trials: list[dict[str, Any]] = []
        self._current_trial: int = 0

    @property
    def trials(self) -> list[dict[str, Any]]:
        return self._trials

    def start_session(self, n_trials: int = 10) -> None:
        """Startet eine neue Blindtest-Session mit n Trials.

        Jeder Trial: X wird zufällig A oder B zugewiesen.
        """
        self._trials = []
        for i in range(n_trials):
            x_is_a = self._rng.choice([True, False])
            self._trials.append(
                {
                    "trial_id": i,
                    "x_is_a": x_is_a,
                    "x_label": f"X{i + 1}",
                    "chosen_a": None,  # Nutzer-Antwort: True=A, False=B
                    "preference": None,  # "a", "b", "none"
                    "confidence": None,  # 1-5
                    "response_time_s": None,
                }
            )
        self._current_trial = 0
        logger.info("Blindtest-Session gestartet: %d Trials", n_trials)

    def get_current_trial(self) -> dict[str, Any] | None:
        """Gibt den aktuellen Trial zurück oder None wenn fertig."""
        if self._current_trial >= len(self._trials):
            return None
        return self._trials[self._current_trial]

    def get_x_audio(self, trial: dict[str, Any]) -> np.ndarray:
        """Gibt das X-Audio für einen Trial zurück (randomisiert A oder B)."""
        return cast(np.ndarray, self._cmp._a if trial["x_is_a"] else self._cmp._b)

    def record_answer(
        self,
        trial: dict[str, Any],
        *,
        chosen_a: bool | None = None,
        preference: str | None = None,  # "a", "b", "none"
        confidence: int | None = None,  # 1-5
        response_time_s: float | None = None,
    ) -> None:
        """Zeichnet die Antwort für einen Trial auf."""
        trial["chosen_a"] = chosen_a
        trial["preference"] = preference
        trial["confidence"] = confidence
        trial["response_time_s"] = response_time_s
        self._current_trial += 1
        logger.debug("Trial %d: chosen_a=%s, pref=%s", trial["trial_id"], chosen_a, preference)

    def get_result(self) -> ABBlindResult:
        """Wertet die Blindtest-Session aus (Binomialtest)."""
        completed = [t for t in self._trials if t["chosen_a"] is not None]
        n = len(completed)

        if n == 0:
            return ABBlindResult()

        correct = sum(1 for t in completed if t["chosen_a"] == t["x_is_a"])

        # Binomialtest: H0 = p=0.5 (Raten)
        from math import comb

        p_value = 0.0
        for k in range(correct, n + 1):
            p_value += comb(n, k) * (0.5**n)
        # Zweiseitig
        p_value = min(p_value * 2.0, 1.0)

        pref_a = sum(1 for t in completed if t.get("preference") == "a")
        pref_b = sum(1 for t in completed if t.get("preference") == "b")
        no_pref = n - pref_a - pref_b

        return ABBlindResult(
            total_trials=n,
            correct=correct,
            p_value=float(p_value),
            preference_a=pref_a,
            preference_b=pref_b,
            no_preference=no_pref,
            trials=completed,
            is_significant=p_value < 0.05,
        )


# ── Synchronisierte A/B-Gruppe ───────────────────────────────────────────────


class ABComparisonGroup:
    """Gruppe synchronisierter A/B-Vergleiche (z.B. für Multi-Track-A/B).

    Alle Instanzen schalten gleichzeitig um.
    """

    def __init__(self) -> None:
        self._comparisons: list[ABComparison] = []
        self._group_size: int = 0

    def add(self, comparison: ABComparison) -> None:
        """Fügt einen A/B-Vergleich zur synchronisierten Gruppe hinzu."""
        comparison._sync_group = id(self)
        self._comparisons.append(comparison)
        self._group_size += 1
        logger.debug("AB-Sync-Gruppe: +1 (Größe=%d)", self._group_size)

    def toggle_all(self) -> None:
        """Schaltet alle Vergleiche gleichzeitig um."""
        for c in self._comparisons:
            c.toggle()
        logger.debug("AB-Sync-Gruppe: alle getoggled (%d)", self._group_size)

    def set_all_a(self) -> None:
        """Setzt alle Vergleiche auf A."""
        for c in self._comparisons:
            c.set_a()

    def set_all_b(self) -> None:
        """Setzt alle Vergleiche auf B."""
        for c in self._comparisons:
            c.set_b()

    @property
    def size(self) -> int:
        return self._group_size
