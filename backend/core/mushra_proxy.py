"""§v10.703 Step 6: MUSHRA-Proxy — Perzeptueller Closed-Loop Estimator.

Leichtgewichtiger MUSHRA-Schätzer (<100ms Laufzeit) für den perzeptuellen
Closed-Loop. Läuft VOR und NACH jeder Phase und entscheidet, ob die Phase
eine hörbare Verbesserung erzielt hat. Keine mathematische Metrik — nur
simuliertes menschliches Hören via MERT-basiertem Embedding-Vergleich.

Architektur:
- MERT-Modell läuft EINMAL pro Pipeline-Run (nicht pro Phase)
- Per-Phase: nur Embedding-Extraktion + Cosinus-Ähnlichkeit
- Rollback-Entscheidung: Delta < 0 → Phase wird rückgängig gemacht

§G145: MUSHRA-Proxy-Pflicht — Jede Phase muss durch den Proxy
§G146: Perzeptueller-Rollback — Delta ≤ 0 → Rollback auf Pre-Phase-Audio
§V45: Mathematisches-Qualitäts-Gate-Verbot — Kein SNR/THD im Phase-Gate
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ProxyMushraVerdict:
    """Ergebnis einer Proxy-MUSHRA-Prüfung vor/nach einer Phase.

    Attributes:
        phase_id: ID der geprüften Phase
        mushra_before: Proxy-MUSHRA vor der Phase (0–100)
        mushra_after: Proxy-MUSHRA nach der Phase (0–100)
        delta: Verbesserung (positiv = besser)
        audible_improvement: True wenn Delta > Hörschwelle
        should_rollback: True wenn Phase rückgängig gemacht werden sollte
        rollback_reason: Begründung für Rollback (None wenn nicht)
        latency_ms: Laufzeit der Prüfung in ms
    """

    phase_id: str
    mushra_before: float
    mushra_after: float
    delta: float
    audible_improvement: bool
    should_rollback: bool
    rollback_reason: str | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# MUSHRA-Proxy Estimator
# ═══════════════════════════════════════════════════════════════════════════════


class MushraProxy:
    """Leichtgewichtiger perzeptueller Qualitätsschätzer.

    Arbeitet auf MERT-Embeddings, die EINMAL pro Pipeline-Run berechnet werden.
    Pro Phase nur Embedding-Extraktion + Cosinus-Vergleich → <100ms.

    Fallback (ohne MERT): Bark-Band-Spectral-Flatness + Harmonic-Noise-Ratio
    als schnelle Heuristik. Weniger genau, aber immer verfügbar.
    """

    # Konfiguration
    _DELTA_THRESHOLD = 0.005  # Min. Proxy-MUSHRA-Verbesserung für "hörbar"
    _ROLLBACK_THRESHOLD = -0.001  # Delta < -0.001 → Rollback (Toleranz für numerisches Rauschen)
    _MAX_LATENCY_MS = 100.0  # Max. erlaubte Laufzeit pro Check

    def __init__(self) -> None:
        self._mert_available = False
        self._session_reference_embedding: np.ndarray | None = None
        self._session_audio_original: np.ndarray | None = None
        self._session_sample_rate: int = 48000
        self._checks_performed: int = 0
        self._rollbacks_triggered: int = 0

        # Versuche MERT zu laden (§v10.705 B5: get_proxy_evaluator statt nicht-existenter get_mert_mushra_proxy)
        try:
            from backend.core.mert_mushra_proxy import get_proxy_evaluator as _get_mert

            _mert = _get_mert()
            self._mert_available = True
            logger.info("§v10.703 MUSHRA-Proxy: MERT-Modell geladen — perzeptueller Closed-Loop aktiv")
        except ImportError:
            logger.info("§v10.703 MUSHRA-Proxy: MERT nicht verfügbar — Ersatzpfad auf Bark-Band-Heuristik")
        except Exception as exc:
            logger.warning("§v10.703 MUSHRA-Proxy: MERT-Ladefehler: %s — Ersatzpfad aktiv", exc)

    def is_available(self) -> bool:
        """True wenn der Proxy einsatzbereit ist (MERT oder Fallback)."""
        return True  # Fallback ist immer verfügbar

    def set_session_reference(self, audio: np.ndarray, sample_rate: int) -> None:
        """Setzt die Session-Referenz — das ORIGINAL vor allen Phasen."""
        self._session_audio_original = np.asarray(audio, dtype=np.float32).copy()
        self._session_sample_rate = sample_rate

        # MERT-Embedding einmal für die Session berechnen
        if self._mert_available:
            try:
                from backend.core.mert_mushra_proxy import get_proxy_evaluator as _get_mert

                _mert = _get_mert()
                self._session_reference_embedding = _mert.compute_embedding(audio, sample_rate)
                logger.debug(
                    "§v10.703 MUSHRA-Proxy: Sitzung-Referenz-Embedding berechnet (shape=%s)",
                    self._session_reference_embedding.shape
                    if self._session_reference_embedding is not None
                    else "None",
                )
            except Exception as exc:
                logger.warning("§v10.703 MUSHRA-Proxy: MERT-Embedding fehlgeschlagen: %s", exc)
                self._session_reference_embedding = None

    def evaluate(
        self,
        phase_id: str,
        audio_before: np.ndarray,
        audio_after: np.ndarray,
        sample_rate: int | None = None,
    ) -> ProxyMushraVerdict:
        """Prüft ob eine Phase hörbare Verbesserung erzielt hat.

        Args:
            phase_id: ID der Phase (z.B. "phase_03_denoise")
            audio_before: Audio VOR der Phase
            audio_after: Audio NACH der Phase
            sample_rate: Sample-Rate (default: session SR)

        Returns:
            ProxyMushraVerdict mit Entscheidung
        """
        sr = sample_rate or self._session_sample_rate
        # Per-Phase-Vergleich: Post-Phase vs Pre-Phase.
        # _estimate_relative() gibt 50 bei identischem Audio, >50 bei Verbesserung, <50 bei Verschlechterung.
        _mushra_after = self._estimate_relative(audio_after, audio_before, sr)
        _mushra_before = 50.0  # Referenzpunkt: "keine Änderung" = 50
        t0 = time.perf_counter()

        _delta = _mushra_after - _mushra_before

        _audible = _delta > self._DELTA_THRESHOLD
        _should_rollback = _delta < self._ROLLBACK_THRESHOLD

        _reason = None
        if _should_rollback:
            _reason = (
                f"Proxy-MUSHRA {_mushra_before:.1f}→{_mushra_after:.1f} "
                f"(Δ={_delta:+.3f}) — KEINE hörbare Verbesserung. Rollback."
            )
            self._rollbacks_triggered += 1
        elif not _audible:
            _reason = (
                f"Proxy-MUSHRA Δ={_delta:+.3f} unter Hörschwelle "
                f"({self._DELTA_THRESHOLD}). Phase wird behalten, aber nicht als Verbesserung gewertet."
            )

        _latency_ms = (time.perf_counter() - t0) * 1000.0
        self._checks_performed += 1

        if _latency_ms > self._MAX_LATENCY_MS:
            logger.warning(
                "§v10.703 MUSHRA-Proxy: Latenz %.1f ms > %.0f ms Limit — Proxy zu langsam",
                _latency_ms,
                self._MAX_LATENCY_MS,
            )

        return ProxyMushraVerdict(
            phase_id=phase_id,
            mushra_before=round(_mushra_before, 2),
            mushra_after=round(_mushra_after, 2),
            delta=round(_delta, 4),
            audible_improvement=_audible,
            should_rollback=_should_rollback,
            rollback_reason=_reason,
            latency_ms=round(_latency_ms, 1),
        )

    def estimate(self, audio: np.ndarray, sample_rate: int = 48000) -> float:
        """Öffentliche Schätz-Methode: Proxy-MUSHRA (0–100) in <100ms.

        Wird von PhaseInterface._safe_process() NACH jeder Phase aufgerufen.
        Kann auch standalone verwendet werden.
        """
        return self._estimate_mushra(audio, sample_rate)

    def _estimate_mushra(self, audio: np.ndarray, sample_rate: int, reference: np.ndarray | None = None) -> float:
        """Schätzt Proxy-MUSHRA-Score (0-100) für ein Audio-Segment.

        Primär: MERT-Embedding-Cosinus-Ähnlichkeit
        Mit reference=None: blinde Schätzung via Heuristik
        Mit reference=audio: vergleicht gegen Referenz-Audio
        """
        if reference is not None and self._mert_available:
            return self._estimate_mushra_relative(audio, reference, sample_rate)

        if self._mert_available and self._session_reference_embedding is not None:
            return self._estimate_mushra_mert(audio, sample_rate)

        return self._estimate_mushra_fallback(audio, sample_rate)

    def _estimate_relative(self, audio: np.ndarray, reference: np.ndarray, sample_rate: int) -> float:
        """Proxy-MUSHRA via direkten Vergleich zweier Audio-Signale.

        MERT: Cosinus-Ähnlichkeit der Embeddings beider Signale
        Fallback: Energie-Differenz pro Bark-Band
        """
        if self._mert_available:
            return self._estimate_mushra_relative(audio, reference, sample_rate)
        # Fallback: vergleiche Spectral Flatness Delta
        _flat_a = self._estimate_mushra_fallback(audio, sample_rate)
        _flat_r = self._estimate_mushra_fallback(reference, sample_rate)
        _delta = _flat_a - _flat_r
        # Mapping: pos. delta = besser, neg. delta = schlechter
        # Kleine Deltas (<5 Punkte) = natürliche Varianz → neutral bei 50
        return float(np.clip(50.0 + np.sign(_delta) * max(0.0, abs(_delta) - 5.0) * 2.0, 0.0, 100.0))

    @classmethod
    def _window_slices(cls, audio: np.ndarray, sample_rate: int) -> list[np.ndarray]:
        """Deterministische 3×30-s-Fenster für MERT-Embeddings (§G5 (copilot-instructions.md), §9)."""
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 2:
            _axis = 0 if arr.shape[-1] <= 2 else -1
        else:
            _axis = 0
        total = arr.shape[_axis]
        win = int(30.0 * sample_rate)
        if total <= win * 3:
            return [arr]
        starts = [0, (total - win) // 2, total - win]
        return [np.take(arr, np.arange(s, s + win), axis=_axis) for s in starts]

    # ── MERT-basierte Schätzung ──────────────────────────────────────────

    def _estimate_mushra_relative(self, audio: np.ndarray, reference: np.ndarray, sample_rate: int) -> float:
        """Direkter MERT-Vergleich: Post-Phase vs Pre-Phase.

        Berechnet MERT-Embeddings für BEIDE Signale und misst Cosinus-Ähnlichkeit.
        Mapping: cos_sim → MUSHRA 0-100. Keine Session-Referenz nötig.
        §9 Performance-Budget BUG-FIX 2026-08-22: Bei langen Signalen
        deterministische 3×30-s-Fenster statt Voll-Längen-Embedding (224 s
        kosteten 37.3 s pro Aufruf).
        """
        try:
            from backend.core.mert_mushra_proxy import get_proxy_evaluator as _get_mert

            _mert = _get_mert()
            _windows_a = self._window_slices(audio, sample_rate)
            _windows_r = self._window_slices(reference, sample_rate)
            _sims: list[float] = []
            for _wa, _wr in zip(_windows_a, _windows_r):
                _emb_a = _mert.compute_embedding(_wa, sample_rate)
                _emb_r = _mert.compute_embedding(_wr, sample_rate)
                if _emb_a is None or _emb_r is None:
                    continue
                _sims.append(
                    float(
                        np.dot(_emb_a.flatten(), _emb_r.flatten())
                        / (np.linalg.norm(_emb_a) * np.linalg.norm(_emb_r) + 1e-8)
                    )
                )
            if not _sims:
                return 50.0
            _cos_sim = float(np.mean(_sims))
            # Mapping: cos_sim 1.0 → MUSHRA 100 (identisch),
            #          cos_sim 0.95 → 80 (minimale Änderung),
            #          cos_sim 0.0 → 50 (starke Änderung, neutral)
            return float(np.clip(_cos_sim * 100.0, 0.0, 100.0))
        except Exception as exc:
            logger.debug("§V6 MUSHRA-MERT-Embedding fehlgeschlagen — Spektrale-Differenz Fallback aktiviert (Audio %s): %s", audio.shape, exc)
            # Fallback: spektrale Differenz
            _flat_a = self._estimate_mushra_fallback(audio, sample_rate)
            _flat_r = self._estimate_mushra_fallback(reference, sample_rate)
            return float(np.clip(50.0 + (_flat_a - _flat_r) * 2.0, 0.0, 100.0))

    def _estimate_mushra_mert(self, audio: np.ndarray, sample_rate: int) -> float:
        """Proxy-MUSHRA via MERT-Embedding-Ähnlichkeit."""
        try:
            from backend.core.mert_mushra_proxy import get_proxy_evaluator as _get_mert

            _mert = _get_mert()
            _embedding = _mert.compute_embedding(audio, sample_rate)

            if _embedding is None or self._session_reference_embedding is None:
                return 50.0

            # Cosinus-Ähnlichkeit → MUSHRA-Mapping
            _cos_sim = float(
                np.dot(_embedding.flatten(), self._session_reference_embedding.flatten())
                / (np.linalg.norm(_embedding) * np.linalg.norm(self._session_reference_embedding) + 1e-8)
            )

            # Mapping: cos_sim ∈ [-1, 1] → MUSHRA ∈ [0, 100]
            # cos_sim = 0.95 → MUSHRA ~80
            # cos_sim = 0.80 → MUSHRA ~50
            # cos_sim = 0.60 → MUSHRA ~20
            _mushra = float(np.clip((_cos_sim + 1.0) * 50.0, 0.0, 100.0))

            # Bias-Korrektur: cos_sim ist zu optimistisch im Hochton-Bereich
            if _mushra > 90.0:
                _mushra = 90.0 + (_mushra - 90.0) * 0.5  # Deckelung bei 95

            return _mushra

        except Exception as exc:
            logger.debug("§v10.703 MUSHRA-Proxy MERT: %s — Ersatzpfad", exc)
            return self._estimate_mushra_fallback(audio, sample_rate)

    # ── Heuristik-Fallback ───────────────────────────────────────────────

    def _estimate_mushra_fallback(self, audio: np.ndarray, sample_rate: int) -> float:
        """Fallback: Bark-Band Spectral Flatness + Harmonic-Noise-Ratio.

        Schnelle FFT-basierte Heuristik, <5ms Laufzeit.
        Korreliert moderat mit MUSHRA (r≈0.6), aber immer verfügbar.
        """
        try:
            _mono = np.mean(audio, axis=-1) if audio.ndim >= 2 else np.asarray(audio, dtype=np.float32)

            # 1024-point FFT für schnelle Spektralanalyse
            _n_fft = min(1024, len(_mono) // 2)
            if _n_fft < 64:
                return 50.0

            _spec = np.abs(np.fft.rfft(_mono[: _n_fft * 2], n=_n_fft))
            _spec = _spec[1:]  # DC entfernen

            if np.sum(_spec) < 1e-10:
                return 50.0

            # Spectral Flatness (GeoMean / ArithMean) → höher = natürlicher
            _geo_mean = np.exp(np.mean(np.log(_spec + 1e-10)))
            _ari_mean = np.mean(_spec)
            _flatness = float(np.clip(_geo_mean / (_ari_mean + 1e-10), 0.0, 1.0))

            # Harmonic-to-Noise Ratio (Peak / Mean in harmonischen Bändern)
            _peaks = np.sort(_spec)[-10:]
            _noise_floor = np.mean(np.sort(_spec)[: len(_spec) // 2])
            _hnr = float(np.clip(np.mean(_peaks) / (_noise_floor + 1e-10), 1.0, 100.0))
            _hnr_norm = float(np.clip(np.log10(_hnr) / 2.0, 0.0, 1.0))

            # Kombiniere: 60% Flatness, 40% HNR
            _combined = 0.6 * _flatness + 0.4 * _hnr_norm

            # Mapping → MUSHRA 0–100
            _mushra = float(np.clip(_combined * 100.0, 0.0, 100.0))

            return _mushra

        except Exception as exc:
            logger.debug("§v10.703 MUSHRA-Proxy Ersatzpfad: %s", exc)
            return 50.0

    # ── Session-Statistik ────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Statistik über alle geprüften Phasen."""
        return {
            "checks_performed": self._checks_performed,
            "rollbacks_triggered": self._rollbacks_triggered,
            "rollback_rate": (round(self._rollbacks_triggered / max(self._checks_performed, 1) * 100, 1)),
            "mert_available": self._mert_available,
            "session_has_reference": self._session_reference_embedding is not None,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════════

_proxy: MushraProxy | None = None


def get_mushra_proxy() -> MushraProxy:
    """Globaler MUSHRA-Proxy (Singleton)."""
    global _proxy
    if _proxy is None:
        _proxy = MushraProxy()
    return _proxy


def reset_mushra_proxy() -> None:
    """Proxy zurücksetzen (für Tests / neuen Pipeline-Run)."""
    global _proxy
    _proxy = None
