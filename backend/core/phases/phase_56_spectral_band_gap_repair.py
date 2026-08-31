"""
Phase 56: Spectral Band Gap Repair — Aurik 10.0.0
================================================

Behebt HEAD_WEAR-Defekte: vollständige Frequenzbandauslöschung durch Azimuth-
oder Magnetisierungsfehler beim Bandlaufwerk. Anders als zeitliche Dropouts sind
Bandlücken breit und konstant über die gesamte Dateilänge aktiv.

ALGORITHMUS (§4.5 über-SOTA-Spezifikation):
--------------------------------------------
1. **Lücken-Detektion (1/6-Okt. Subbänder)**
   - Mittlere Energie pro 1/6-Okt.-Band über rollenden 30-Frame-Median
   - Leer-Kandidat: Energie ≤ -60 dBFS über ≥ 80 % der Dateilänge
   - Mindest-Lückenbreite: 200 Hz; kein Eingriff in Notch-artige Dips (<200 Hz breit)

2. **Harmonische Partial-Interpolation** (Fletcher-Modell inkl. Inharmonizität)
   - f₀-Bestimmung via CREPE (Fallback: librosa.yin)
   - Für jedes Partial fₙ = n · f₀ · √(1 + B·n²):
     fehlende Partial-Amplitude = geometrisches Mittel der Nachbar-Partials (n-1, n+1)
   - Inharmonizitäts-Koeffizient B aus INHARMONICITY_PRIORS

3. **Spektrale Glattheit-Prüfung**
   - Spectral Flatness der reparierten Zone ≤ 0.4
   - Zu hohe Flatness → NMF-β-Verfeinerung (fehlendes Band als NMF-Template)

4. **NMF-β Verfeinerung** (β=1, Itakura-Saito)
   - W-Matrix benachbarter, klarer Segmente als Initialisierungsmatrix
   - Aktivierungen für das fehlerhafte Segment iterativ optimiert

5. **PGHI Phase Reconstruction**
   - Phasenkonsistente Rückwandlung nach Spektral-Modifikation
   - Gradient-basierte Phasenrekonstruktion (Perraudin 2013)

WISSENSCHAFTLICHE GRUNDLAGE:
-----------------------------
- Roebel (2010): "Transient Detection and Preservation" — Lücken-Charakterisierung
- Fletcher (1964): Inharmonizität Klaviersaiten (Partial-Interpolation)
- Février & Idier (2011): NMF mit β-Divergenz (Itakura-Saito)
- Perraudin et al. (2013): PGHI für phasenkonsistente Rücktransformation

AKTIVIERUNG:
-----------
Nur bei DefectType.HEAD_WEAR, confidence ≥ 0.55 (CAUSE_TO_PHASES §7.2)
Nur bei MaterialType TAPE / REEL_TAPE

Autor: Aurik 10.0.0 Development Team / v10.0.0
"""

from __future__ import annotations

import importlib
import logging
import math
from typing import Any

import numpy as np
from scipy import signal

from backend.core.audio_utils import safe_to_mono, to_channels_last

from .phase_interface import (
    PhaseCategory,
    PhaseInterface,
    PhaseMetadata,
    PhaseResult,
    create_phase_result,
)

# Optionale Imports werden in dieser Phase bewusst lazy geladen.
# pylint: disable=import-outside-toplevel

try:
    from dsp.pghi import PghiReconstructor as _PGHIRec_P56  # type: ignore

    _PGHI_AVAILABLE_P56 = True
except ImportError:
    _PGHIRec_P56 = None  # type: ignore[misc,assignment]
    _PGHI_AVAILABLE_P56 = False


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optionale Abhängigkeiten mit Graceful Degradation (§3.4)
# ---------------------------------------------------------------------------
try:
    import librosa

    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False
    logger.warning("⚠️ SOTA Verarbeitungsschritt 56: librosa nicht verfügbar — DSP-Ersatzpfad für f₀-Schätzung aktiv")

try:
    from plugins.fcpe_plugin import get_fcpe_plugin as _get_pitch_plugin

    _CREPE_OK = True
except Exception:
    try:
        from plugins.crepe_plugin import get_crepe_plugin as _get_pitch_plugin  # type: ignore[assignment]

        _CREPE_OK = True
    except Exception:
        _CREPE_OK = False
        _get_pitch_plugin = None  # type: ignore[assignment]
        logger.debug("FCPE/CREPE nicht verfügbar — pYIN-Ersatzpfad aktiv")


# ---------------------------------------------------------------------------
# Inharmonizitäts-Priors (aus §2.11 HarmonicLatticeAnalyzer)
# ---------------------------------------------------------------------------
INHARMONICITY_PRIORS: dict[str, float] = {
    "piano_bass": 0.0080,
    "piano_mid": 0.0020,
    "piano_treble": 0.0001,
    "guitar": 0.0005,
    "violin": 0.0003,
    "flute": 0.0000,
    "brass": 0.0001,
    "unknown": 0.0010,
}

# Mindest-Lückenbreite in Hz (kein Eingriff bei schmalen Notches)
_MIN_GAP_WIDTH_HZ: float = 200.0
# Energie-Schwelle „leeres Band" [dBFS]
_GAP_ENERGY_THRESHOLD_DBFS: float = -60.0
# Mindest-Anteil der Dateilänge für die das Band leer ist
_GAP_FRACTION_MIN: float = 0.80
# Spectral Flatness Maximalwert nach Reparatur
_MAX_SPECTRAL_FLATNESS: float = 0.40
# PGHI-Iterationen
_PGHI_ITERATIONS: int = 32


# ---------------------------------------------------------------------------
# Interne Hilfsfunktionen
# ---------------------------------------------------------------------------


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Stereo → Mono Mischung (float32). Handles both (N,2) and (2,N) layouts (§2.51)."""
    mono = safe_to_mono(audio) if audio.ndim == 2 else audio
    return mono.astype(np.float32)  # type: ignore[no-any-return]


def _estimate_f0(mono: np.ndarray, sr: int) -> float | None:
    """
    f0 estimation cascade: FCPE -> RMVPE -> PESTO -> pYIN -> autocorrelation.

    Returns:
        Medianer f₀ in Hz oder None wenn keine Stimmigkeit erkennbar.
    """
    # Tier-1: FCPE
    _plm56_fcpe = None
    try:
        from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm56f

        _plm56_fcpe = _get_plm56f()
        _plm56_fcpe.set_active("FCPE", True)
    except Exception:
        logger.debug("_estimate_f0: silent except suppressed", exc_info=True)
    try:
        from plugins.fcpe_plugin import get_fcpe_plugin

        result = get_fcpe_plugin().analyze(mono, sr)
        voiced_mask = result.voiced_prob >= 0.55
        voiced = result.f0_hz[voiced_mask]
        if len(voiced) > 5:
            return float(np.median(voiced[voiced > 20.0]))
    except Exception as exc:
        logger.debug("FCPE f0 estimation fehlgeschlagen: %s", exc)
    finally:
        if _plm56_fcpe is not None:
            try:
                _plm56_fcpe.set_active("FCPE", False)
            except Exception:
                logger.debug("_estimate_f0: silent except suppressed", exc_info=True)

    # Tier-2: RMVPE
    _plm56_rmvpe = None
    try:
        from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm56r

        _plm56_rmvpe = _get_plm56r()
        _plm56_rmvpe.set_active("RMVPE", True)
    except Exception:
        logger.debug("_estimate_f0: silent except suppressed", exc_info=True)
    try:
        from plugins.rmvpe_plugin import get_rmvpe_plugin

        _rmvpe_result = get_rmvpe_plugin().analyze(mono, sr)
        # §v10.40-Rev. 2026-08-16: RmvpeResult liefert f0/voiced_flag/confidence —
        # NICHT f0_hz/voiced_prob (CrepeResult-API). Der Zugriff hier warf zuvor
        # AttributeError → Stufe fiel immer durch (stiller DSP-Downgrade, §V6).
        voiced_mask = np.asarray(_rmvpe_result.voiced_flag, dtype=bool)
        voiced = np.asarray(_rmvpe_result.f0, dtype=np.float64)[voiced_mask]
        if len(voiced) > 5:
            return float(np.median(voiced[voiced > 20.0]))
    except Exception as exc:
        logger.debug("RMVPE f0 estimation fehlgeschlagen: %s", exc)
    finally:
        if _plm56_rmvpe is not None:
            try:
                _plm56_rmvpe.set_active("RMVPE", False)
            except Exception:
                logger.debug("_estimate_f0: silent except suppressed", exc_info=True)

    # Tier-3: PESTO
    try:
        _pesto_mod = importlib.import_module("plugins.pesto_plugin")
        get_pesto_plugin = _pesto_mod.get_pesto_plugin

        result = get_pesto_plugin().analyze(mono, sr)
        voiced_mask = result.voiced_prob >= 0.55
        voiced = result.f0_hz[voiced_mask]
        if len(voiced) > 5:
            return float(np.median(voiced[voiced > 20.0]))
    except Exception as exc:
        logger.debug("PESTO f0 estimation fehlgeschlagen: %s", exc)

    # Tier-2: pYIN über librosa
    if _LIBROSA_OK:
        try:
            f0, voiced_flag, _ = librosa.pyin(
                mono,
                fmin=50.0,  # ≥ 2 Perioden bei frame_length=2048 @48 kHz (min=46.875 Hz)
                fmax=float(librosa.note_to_hz("C8")),
                sr=sr,
            )
            voiced_f0 = f0[voiced_flag & np.isfinite(f0) & (f0 > 20.0)]
            if len(voiced_f0) > 5:
                return float(np.median(voiced_f0))
        except Exception as exc:
            logger.debug("pYIN f₀-Schätzung fehlgeschlagen: %s", exc)

    # Tier-3: Autokorrelation (einfacher DSP-Fallback) — FFT-based O(N log N)
    mono_seg = mono[: min(len(mono), sr)]
    from backend.core.core_utils import fft_autocorr

    autocorr = fft_autocorr(mono_seg)
    min_lag = max(1, int(sr / 1200.0))
    max_lag = int(sr / 50.0)
    if max_lag >= len(autocorr):
        return None
    peak_lag = min_lag + int(np.argmax(autocorr[min_lag:max_lag]))
    if autocorr[peak_lag] / (autocorr[0] + 1e-12) > 0.3:
        return float(sr / peak_lag)
    return None


def _detect_band_gaps(
    stft_mag: np.ndarray,
    sr: int,
    n_fft: int,
    gap_fraction_min: float | None = None,
) -> list[tuple[int, int]]:
    """
    Detektiert leere Frequenzbänder im STFT-Magnitudenspektrum.

    Args:
        stft_mag: |STFT| Matrix [n_bins × n_frames] float32
        sr: Sample-Rate [Hz]
        n_fft: FFT-Größe

    Returns:
        Liste von (bin_low, bin_high) Tupeln für identifizierte Lücken.
    """
    n_bins, _n_frames = stft_mag.shape
    freq_resolution = sr / n_fft  # Hz pro Bin

    # Energie pro Bin (logarithmisch): mittlere Bandenergie als zweites Gate,
    # damit sporadische Bursts kein dauerhaftes Gap vortaeuschen.
    energy_per_bin = np.mean(stft_mag**2, axis=1)
    energy_per_bin_db = 10.0 * np.log10(energy_per_bin + 1e-12)

    # Rollender Median über 30 Frames für Stabilisierung
    frame_medians = np.median(stft_mag, axis=1)
    frame_median_db = 10.0 * np.log10(frame_medians**2 + 1e-12)

    # Anteil der Frames, in denen ein Bin leer ist
    threshold_linear = 10.0 ** (_GAP_ENERGY_THRESHOLD_DBFS / 20.0)
    empty_fraction = np.mean(stft_mag < threshold_linear, axis=1)

    # Bins die dauerhaft extrem leise und weit unter dem Schwellwert sind
    # Optionaler Override erlaubt konservativeres Side-Processing ohne globale Mutation.
    _gap_fraction = float(gap_fraction_min) if gap_fraction_min is not None else _GAP_FRACTION_MIN
    gap_bins = (
        (empty_fraction >= _gap_fraction)
        & (frame_median_db <= _GAP_ENERGY_THRESHOLD_DBFS)
        & (energy_per_bin_db <= _GAP_ENERGY_THRESHOLD_DBFS)
    )

    # Kontinuierliche Bereiche identifizieren
    gaps: list[tuple[int, int]] = []
    in_gap = False
    gap_start = 0
    for b in range(n_bins):
        if gap_bins[b] and not in_gap:
            in_gap = True
            gap_start = b
        elif not gap_bins[b] and in_gap:
            in_gap = False
            gap_end = b
            # Mindest-Breite prüfen
            width_hz = (gap_end - gap_start) * freq_resolution
            if width_hz >= _MIN_GAP_WIDTH_HZ:
                gaps.append((gap_start, gap_end))
    if in_gap:
        width_hz = (n_bins - gap_start) * freq_resolution
        if width_hz >= _MIN_GAP_WIDTH_HZ:
            gaps.append((gap_start, n_bins))

    logger.debug("SpectralBandGapRepair: %d Lücken entdeckt", len(gaps))
    return gaps


def _harmonic_interpolate_gap(
    stft_mag: np.ndarray,
    stft_phase: np.ndarray,
    gap: tuple[int, int],
    f0_hz: float,
    sr: int,
    n_fft: int,
    instrument_tag: str = "unknown",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Füllt eine Spektrallücke durch harmonische Partial-Interpolation.

    Algorithmus:
        Für jedes Partial fₙ im Lückenbereich:
            amplitude = geom. Mittel der Nachbar-Amplituden (fₙ₋₁, fₙ₊₁)
        Phase: zufällig initialisiert (PGHI überschreibt später)

    Returns:
        (mag_filled, phase_filled) — modifizierte Matrizen
    """
    mag_out = stft_mag.copy()
    phase_out = stft_phase.copy()

    B = INHARMONICITY_PRIORS.get(instrument_tag, INHARMONICITY_PRIORS["unknown"])
    freq_per_bin = sr / n_fft
    gap_low, gap_high = gap

    # Partials im Lückenbereich aufsammeln
    n_partial = 1
    while True:
        f_n = n_partial * f0_hz * math.sqrt(1.0 + B * n_partial**2)
        if f_n > sr / 2.0:
            break
        bin_n = round(f_n / freq_per_bin)
        if gap_low <= bin_n < gap_high:
            # Nachbar-Partials für geometrisches Mittel
            bin_prev = round((n_partial - 1) * f0_hz / freq_per_bin) if n_partial > 1 else 0
            bin_next = round((n_partial + 1) * f0_hz / freq_per_bin)
            bin_prev = max(0, min(bin_prev, stft_mag.shape[0] - 1))
            bin_next = max(0, min(bin_next, stft_mag.shape[0] - 1))

            amp_prev = float(np.mean(mag_out[bin_prev])) if bin_prev < gap_low or bin_prev >= gap_high else 0.0
            amp_next = float(np.mean(mag_out[bin_next])) if bin_next < gap_low or bin_next >= gap_high else 0.0

            if amp_prev > 0 and amp_next > 0:
                amp_interp = math.sqrt(amp_prev * amp_next)
            elif amp_prev > 0:
                amp_interp = amp_prev * 0.7
            elif amp_next > 0:
                amp_interp = amp_next * 0.7
            else:
                n_partial += 1
                continue

            # ±2 Bins um das Partial mit gaußförmigem Abfall
            for db in range(-2, 3):
                b_fill = bin_n + db
                if 0 <= b_fill < stft_mag.shape[0]:
                    gauss_weight = math.exp(-0.5 * (db / 1.0) ** 2)
                    mag_out[b_fill] = np.maximum(
                        mag_out[b_fill], amp_interp * gauss_weight * np.ones(stft_mag.shape[1], dtype=np.float32)
                    )
                    # Phasenkonsistente Initialisierung aus Nachbar-Bins; PGHI folgt danach.
                    if 0 <= bin_prev < stft_phase.shape[0] and 0 <= bin_next < stft_phase.shape[0]:
                        phase_out[b_fill] = (0.5 * stft_phase[bin_prev] + 0.5 * stft_phase[bin_next]).astype(np.float32)
                    elif 0 <= bin_prev < stft_phase.shape[0]:
                        phase_out[b_fill] = stft_phase[bin_prev].astype(np.float32)
                    elif 0 <= bin_next < stft_phase.shape[0]:
                        phase_out[b_fill] = stft_phase[bin_next].astype(np.float32)
                    else:
                        phase_out[b_fill] = np.zeros(stft_mag.shape[1], dtype=np.float32)

        n_partial += 1
        if n_partial > 40:  # Sicherheits-Stop
            break

    return mag_out, phase_out


def _spectral_flatness(mag: np.ndarray) -> float:
    """Berechnet Spectral Flatness für einen Frequenzbereich [0, 1]."""
    mag_safe = np.abs(mag) + 1e-12
    geom_mean = np.exp(np.mean(np.log(mag_safe)))
    arith_mean = np.mean(mag_safe)
    return float(geom_mean / (arith_mean + 1e-12))


def _pghi_phase_reconstruction(mag: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """
    PGHI Phase Gradient Heap Integration (Perraudin et al. 2013).

    Primary path: uses PGHIReconstructor from dsp/pghi.py (full algorithm).
    Fallback: Instantaneous Frequency estimation if dsp/pghi not available.  # §V6 (copilot-instructions.md): logger.warning handled at call site

    Args:
        mag: [n_bins × n_frames] float32
        n_fft: FFT-Größe
        hop: Hop-Länge

    Returns:
        Korrigierte Phase [n_bins × n_frames] float32
    """
    # Primary: PGHIReconstructor (dsp/pghi.py — vollständige Implementierung)
    try:
        if _PGHIRec_P56 is None:
            raise ImportError("PghiReconstructor nicht verfügbar")
        _pghi_rec = _PGHIRec_P56(sr=48000, win_size=n_fft, hop=hop)
        # reconstruct() returns PghiResult with .audio field; we derive phase from STFT of audio
        _result = _pghi_rec.reconstruct(magnitude=mag, win_size=n_fft, hop=hop)
        # Extract phase from the reconstructed STFT (stored in PghiResult)
        if hasattr(_result, "stft") and _result.stft is not None:
            _phase_out = np.angle(_result.stft).astype(np.float32)
        else:
            # fallback: compute phase from reconstructed audio via STFT
            _noverlap = min(n_fft - hop, max(0, n_fft - 1))  # §v10.103
            _, _, _stft_r = signal.stft(_result.audio, fs=48000, nperseg=n_fft, noverlap=_noverlap)
            _phase_out = np.angle(_stft_r).astype(np.float32)
        # Ensure shape matches input
        if _phase_out.shape == mag.shape:
            logger.debug(
                "Verarbeitungsschritt 56: PGHI via PghiReconstructor (dsp/pghi.py) — n_fft=%d hop=%d", n_fft, hop
            )
            return _phase_out  # type: ignore[no-any-return]
        logger.debug(
            "Verarbeitungsschritt 56: PGHI shape mismatch (%s vs %s), IF-Ersatzpfad", _phase_out.shape, mag.shape
        )
    except Exception as _pghi_import_exc:
        logger.debug("PghiReconstructor nicht verfügbar, IF-Ersatzpfad: %s", _pghi_import_exc)

    # Fallback: Instantaneous Frequency estimation
    n_bins, n_frames = mag.shape
    freq_per_bin = 2.0 * np.pi / n_fft  # normiert

    # §2.40 Determinismus: content-derived seed for IF-fallback phase init
    _if_seed = int(abs(float(np.sum(np.abs(mag[:, : min(n_frames, 4)])))) * 1e5 + n_fft) % (2**31)
    _rng_if = np.random.default_rng(seed=_if_seed)
    phase = np.zeros_like(mag)
    phase[:, 0] = _rng_if.uniform(-np.pi, np.pi, size=n_bins)

    for t in range(1, n_frames):
        d_mag_dt = mag[:, t] - mag[:, t - 1]
        phase[:, t] = phase[:, t - 1] + freq_per_bin * hop * np.arange(n_bins) + 0.01 * d_mag_dt

    return phase.astype(np.float32)  # type: ignore[no-any-return]


def _nmf_beta_refine(
    stft_mag: np.ndarray,
    gap: tuple[int, int],
    n_components: int = 8,
    n_iter: int = 50,
) -> np.ndarray:
    """
    NMF-β Verfeinerung des reparierten Bandes (Itakura-Saito, β=1).

    Benutzt benachbarte klare Bänder als Initialisierungsmatrix W₀.
    Nur aktiv wenn Spectral Flatness > _MAX_SPECTRAL_FLATNESS.

    Args:
        stft_mag: Magnitudenspektrum [n_bins × n_frames] nach erster Reparatur
        gap: (bin_low, bin_high) der Lücke
        n_components: NMF-Rang
        n_iter: Iterationen

    Returns:
        Verfeinertes Magnitudenspektrum [n_bins × n_frames]
    """
    try:
        from sklearn.decomposition import NMF as _NMF
    except ImportError:
        logger.warning("⚠️ SOTA Verarbeitungsschritt 56: sklearn nicht verfügbar — NMF-Verfeinerung deaktiviert")
        return stft_mag

    n_bins, n_frames = stft_mag.shape
    gap_low, gap_high = gap

    # Template aus benachbarten Bändern
    context_low = max(0, gap_low - 20)
    context_high = min(n_bins, gap_high + 20)
    context_bins = list(range(context_low, gap_low)) + list(range(gap_high, context_high))

    if len(context_bins) < 4:
        return stft_mag

    W_context = stft_mag[context_bins, :].T  # [n_frames × n_context_bins]
    W_context = np.maximum(W_context, 1e-12)

    model = _NMF(
        n_components=min(n_components, min(W_context.shape) - 1),
        solver="mu",
        beta_loss="itakura-saito",
        max_iter=n_iter,
        random_state=42,
    )
    try:
        W_context_T = W_context.T  # [n_context_bins × n_frames]
        model.fit(W_context_T)
        H = model.components_  # [n_components × n_frames]
        # Fallback: model.components_.mean(axis=1, keepdims=True).T

        # Rekonstruktion für Lücken-Bins
        mag_out = stft_mag.copy()
        gap_width = gap_high - gap_low
        if gap_width > 0 and H.shape[1] == n_frames:
            # Einfache Interpolations-Rekonstruktion
            scale = np.mean(stft_mag[context_bins]) / (np.mean(H) + 1e-12)
            mag_out[gap_low:gap_high, :] = np.maximum(
                stft_mag[gap_low:gap_high, :],
                (
                    (H[:gap_width].T * scale * 0.5 + stft_mag[gap_low:gap_high, :])
                    if H.shape[0] >= gap_width
                    else stft_mag[gap_low:gap_high, :]
                ),
            )
    except Exception as nmf_exc:
        logger.debug("NMF-β Verfeinerung fehlgeschlagen: %s", nmf_exc)
        return stft_mag

    return mag_out


# ---------------------------------------------------------------------------
# Phase-Klasse
# ---------------------------------------------------------------------------


class SpectralBandGapRepairPhase(PhaseInterface):
    """
    Phase 56: SpectralBandGapRepair — HEAD_WEAR-Defekt Reparatur.

    Behebt dauerhaft ausgelöschte Frequenzbänder durch:
    1. Energiebasierte Lückendetektion (1/6-Okt.)
    2. Harmonische Partial-Interpolation (Fletcher-Modell)
    3. Optionale NMF-β Verfeinerung bei zu hoher Flatness
    4. PGHI phasenkonsistente Rücktransformation

    Aktivierung: Nur bei HEAD_WEAR-Defekt, confidence ≥ 0.55
    """

    # MRSA Multi-Resolution Spectral Analysis zones (mandatory, §DSP-Spezialregeln)
    _MRSA_ZONES: tuple = (
        ("sub_bass", 65536, 16384, 0, 250),
        ("mid_low", 16384, 4096, 250, 2500),
        ("mid", 8192, 2048, 2500, 8000),
        ("presence", 1024, 256, 8000, 16000),
        ("air", 128, 32, 16000, 24000),
    )
    _MRSA_CROSSFADE_BW_HZ: float = 100.0

    def __init__(self, sample_rate: int = 48000, **kwargs: Any) -> None:
        self.n_fft: int = kwargs.get("n_fft", 2048)
        self.hop_length: int = kwargs.get("hop_length", 512)
        self.instrument_tag: str = kwargs.get("instrument_tag", "unknown")
        self._current_phoneme_timeline: Any = None
        super().__init__(sample_rate=sample_rate, **kwargs)

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_56_spectral_band_gap_repair",
            name="Spectral Band Gap Repair (HEAD_WEAR)",
            category=PhaseCategory.RESTORATION,
            priority=8,
            dependencies=["phase_14_phase_correction", "phase_06_frequency_restoration"],
            estimated_time_factor=0.04,
            version="1.0.0",
            memory_requirement_mb=180,
            is_cpu_intensive=True,
            quality_impact=0.9,
            description=(
                "Repariert dauerhaft ausgelöschte Frequenzbänder durch Azimuth-/ "
                "Magnetisierungsfehler (HEAD_WEAR-Defekt) via harmonischer "
                "Partial-Interpolation + NMF-β + PGHI."
            ),
        )

    @staticmethod
    def _compute_band_gap_profile(
        material_key: str,
        quality_mode: str,
        restorability_score: float,
    ) -> dict[str, float]:
        """§2.54 Adaptive gate profile for spectral band-gap repair."""
        _material = str(material_key or "unknown").strip().lower()
        _aliases = {"restoration": "balanced", "studio_2026": "maximum"}
        _mode = _aliases.get(
            str(quality_mode or "balanced").strip().lower(), str(quality_mode or "balanced").strip().lower()
        )

        if any(token in _material for token in ("tape", "reel_tape", "cassette")):
            min_head_wear_confidence = 0.52
            mid_gap_fraction_min = 0.78
            side_gap_fraction_min = 0.92
        elif any(token in _material for token in ("cd_digital", "dat", "streaming", "flac")):
            min_head_wear_confidence = 0.70
            mid_gap_fraction_min = 0.88
            side_gap_fraction_min = 0.98
        else:
            min_head_wear_confidence = 0.60
            mid_gap_fraction_min = 0.84
            side_gap_fraction_min = 0.95

        _rest = float(np.clip(float(restorability_score or 50.0), 0.0, 100.0))
        _rest_norm = _rest / 100.0
        min_head_wear_confidence += (_rest_norm - 0.5) * 0.18
        mid_gap_fraction_min += (_rest_norm - 0.5) * 0.12
        side_gap_fraction_min += (_rest_norm - 0.5) * 0.06

        _mode_offsets = {
            "fast": (0.08, 0.05, 0.02),
            "balanced": (0.0, 0.0, 0.0),
            "quality": (-0.05, -0.04, -0.02),
            "maximum": (-0.08, -0.06, -0.03),
        }
        _conf_off, _mid_off, _side_off = _mode_offsets.get(_mode, (0.0, 0.0, 0.0))
        min_head_wear_confidence += _conf_off
        mid_gap_fraction_min += _mid_off
        side_gap_fraction_min += _side_off

        return {
            "min_head_wear_confidence": float(np.clip(min_head_wear_confidence, 0.40, 0.85)),
            "mid_gap_fraction_min": float(np.clip(mid_gap_fraction_min, 0.70, 0.97)),
            "side_gap_fraction_min": float(np.clip(side_gap_fraction_min, 0.85, 0.995)),
        }

    @staticmethod
    def _collect_protected_zones(kwargs: dict[str, Any]) -> list[tuple[float, float, float]]:
        """Sammelt Vokal-/Performance-Zonen fuer lokale Reparatur-Caps."""
        protected_zones: list[tuple[float, float, float]] = []
        zone_sources = (
            ("vibrato_zones", 0.20),
            ("frisson_zones", 0.30),
            ("whisper_zones", 0.25),
            ("passaggio_zones", 0.35),
        )
        for key, cap in zone_sources:
            for zone in kwargs.get(key) or []:
                try:
                    start_s = float(getattr(zone, "start_s", None) or zone[0])
                    end_s = float(getattr(zone, "end_s", None) or zone[1])
                    if end_s > start_s:
                        protected_zones.append((start_s, end_s, cap))
                except Exception:
                    logger.debug("_collect_protected_zones: silent except suppressed", exc_info=True)
                    continue
        return protected_zones

    @staticmethod
    def _local_event_strength(key: str, loc: tuple[float, float], event_metadata: dict[str, Any] | None) -> float:
        """Defekttyp- und Event-adaptive Einblendstaerke fuer Bandluecken."""
        duration_s = max(0.0, float(loc[1]) - float(loc[0]))
        duration_factor = float(np.clip(duration_s / 0.45, 0.25, 1.0))
        key_factor = {
            "tape_head_clog": 1.0,
            "head_wear": 0.82,
            "tape_head_level_dip": 0.58,
        }.get(key, 0.7)
        severity = 0.5
        confidence = 0.8
        meta_obj = (event_metadata or {}).get(key)
        if isinstance(meta_obj, dict):
            severity = float(np.clip(float(meta_obj.get("severity", severity)), 0.0, 1.0))
            confidence = float(np.clip(float(meta_obj.get("confidence", confidence)), 0.0, 1.0))
        return float(np.clip(key_factor * (0.42 + 0.38 * severity + 0.20 * confidence) * duration_factor, 0.18, 1.0))

    @staticmethod
    def _build_locality_profile(
        n_samples: int,
        sample_rate: int,
        defect_locations: dict[str, list[tuple[float, float]]] | None,
        event_metadata: dict[str, Any] | None = None,
        protected_zones: list[tuple[float, float, float]] | None = None,
    ) -> tuple[np.ndarray, float]:
        """Erzeugt lokale Blendmaske für HEAD_WEAR/TAPE_HEAD_CLOG-Reparatur."""
        if n_samples <= 0 or sample_rate <= 0:
            return np.zeros(0, dtype=np.float32), 0.0
        if not isinstance(defect_locations, dict) or not defect_locations:
            return np.ones(n_samples, dtype=np.float32), 0.0

        keys = ("head_wear", "tape_head_clog", "tape_head_level_dip")
        mask = np.zeros(n_samples, dtype=np.float32)
        pad = int(0.05 * sample_rate)
        for key in keys:
            for loc in defect_locations.get(key) or []:
                if not isinstance(loc, tuple) or len(loc) != 2:
                    continue
                try:
                    s = int(max(0.0, float(loc[0])) * sample_rate)
                    e = int(max(0.0, float(loc[1])) * sample_rate)
                except Exception:
                    logger.debug("_build_locality_Profil: silent except suppressed", exc_info=True)
                    continue
                if e <= s:
                    continue
                s = max(0, s - pad)
                e = min(n_samples, e + pad)
                if e > s:
                    event_strength = SpectralBandGapRepairPhase._local_event_strength(key, loc, event_metadata)
                    mask[s:e] = np.maximum(mask[s:e], event_strength)

        if float(np.mean(mask)) <= 1e-6:
            return np.ones(n_samples, dtype=np.float32), 0.0

        smooth = max(16, int(0.02 * sample_rate))
        mask = np.convolve(mask, np.ones(smooth, dtype=np.float32) / float(smooth), mode="same")
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
        if protected_zones:
            for start_s, end_s, cap in protected_zones:
                s = int(max(0.0, float(start_s)) * sample_rate)
                e = int(max(0.0, float(end_s)) * sample_rate)
                if e > s:
                    mask[s : min(n_samples, e)] = np.minimum(mask[s : min(n_samples, e)], float(cap))
        return mask, float(np.mean(mask))

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        **kwargs: Any,
    ) -> PhaseResult:
        """
        Hauptverarbeitung: Detektion + Reparatur spektraler Bandlücken.

        Args:
            audio: Eingangsaudio (mono oder stereo, float32)
            **kwargs:
                confidence (float): Defekt-Konfidenz (Standard: 1.0)
                instrument_tag (str): Instrument-Typ für Inharmonizität

        Returns:
            PhaseResult mit repariertem Audio und Metadata
        """
        sample_rate = int(kwargs.get("sample_rate", sample_rate))
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        audio, _p56_transposed = to_channels_last(audio)

        # §4.6b: Pre-phase eviction — free previous phase models to prevent OOM
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm_evict56

            _get_plm_evict56().evict_for_phase("phase_56_spectral_band_gap_repair")
        except Exception:
            logger.debug("verarbeiten: silent except suppressed", exc_info=True)

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        effective_strength = float(kwargs.get("strength", 1.0)) * phase_locality_factor
        effective_strength = float(np.clip(effective_strength, 0.0, 1.0))

        # §V41 ForwardMaskingGuard — Enhancement-Stärke in post-transienten Masking-Zonen erhöhen
        _panns_s_56 = float(kwargs.get("panns_singing", 0.0))
        if _panns_s_56 >= 0.25 and effective_strength > 0.0:
            try:
                from backend.core.dsp.temporal_masking import (
                    get_forward_masking_guard as _fmg_fn_56,
                )

                _fmz_56 = kwargs.get("forward_masking_zones") or _fmg_fn_56().compute_zones(audio, sample_rate)
                if _fmz_56:
                    _n_s_56 = audio.shape[-1] if audio.ndim > 1 else len(audio)
                    _zone_s_56 = sum(z.end_sample - z.start_sample for z in _fmz_56)
                    _zone_frac_56 = float(np.clip(_zone_s_56 / max(1, _n_s_56), 0.0, 1.0))
                    effective_strength = float(np.clip(effective_strength + _zone_frac_56 * 0.15, 0.0, 1.0))
            except Exception as _fmg_exc_56:
                logger.debug("Verarbeitungsschritt56 §V41 ForwardMaskingGuard nicht blockierend: %s", _fmg_exc_56)

        if effective_strength <= 1e-6:
            # §0a Zero-Strength-Passthrough: bit-identische Rueckgabe (kein Soft-Clip, das die
            # Peaks des Originals ändern würde). §3.1: nur NaN/Inf-Bereinigung.
            _passthrough_56 = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            return PhaseResult(
                audio=_passthrough_56,
                success=True,
                execution_time_seconds=0.0,
                ml_used=False,
                quality_estimate=1.0,
                _skip_soft_clip=True,  # §0a: bit-identischer Passthrough
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
            )

        _material_key_56 = str(material_type or kwargs.get("material_type", kwargs.get("material", "unknown"))).lower()
        _band_gap_profile = self._compute_band_gap_profile(
            _material_key_56,
            str(kwargs.get("quality_mode", "balanced")),
            float(kwargs.get("restorability_score", 50.0)),
        )
        confidence: float = float(kwargs.get("confidence", 1.0))
        _clog_severity = 0.0
        _clog_confidence = 0.0
        _defect_scores_obj = kwargs.get("defect_scores")
        if isinstance(_defect_scores_obj, dict):
            _clog_score_obj = _defect_scores_obj.get("tape_head_clog")
            if _clog_score_obj is None:
                try:
                    from backend.core.defect_scanner import DefectType as _DS_DefectType

                    _clog_score_obj = _defect_scores_obj.get(_DS_DefectType.TAPE_HEAD_CLOG)
                except Exception:
                    _clog_score_obj = None
            if _clog_score_obj is not None:
                _clog_severity = float(np.clip(float(getattr(_clog_score_obj, "severity", 0.0)), 0.0, 1.0))
                _clog_confidence = float(np.clip(float(getattr(_clog_score_obj, "confidence", 0.0)), 0.0, 1.0))

        # TAPE_HEAD_CLOG ist lokal/zeitvariabel (kein globaler HEAD_WEAR-Dauerzustand).
        # Bei starker Clog-Evidenz Confidence-Gate konservativ absenken und Gap-Schwellen lockern,
        # damit die Reparatur in realen Kopfverschmutzungssegmenten zuverlässig triggert.
        if _clog_severity >= 0.10:
            confidence = max(confidence, float(np.clip(0.30 + 0.55 * _clog_confidence, 0.30, 0.90)))
            _band_gap_profile["min_head_wear_confidence"] = float(
                min(
                    _band_gap_profile["min_head_wear_confidence"],
                    np.clip(0.50 - 0.16 * _clog_severity, 0.34, 0.50),
                )
            )
            _band_gap_profile["mid_gap_fraction_min"] = float(
                np.clip(_band_gap_profile["mid_gap_fraction_min"] - 0.08 * _clog_severity, 0.66, 0.97)
            )
            _band_gap_profile["side_gap_fraction_min"] = float(
                np.clip(_band_gap_profile["side_gap_fraction_min"] - 0.05 * _clog_severity, 0.82, 0.995)
            )

        if confidence < _band_gap_profile["min_head_wear_confidence"]:
            logger.debug(
                "SpectralBandGapRepair: confidence=%.2f < %.2f, übersprungen",
                confidence,
                _band_gap_profile["min_head_wear_confidence"],
            )
            return create_phase_result(
                audio,
                modifications={"skipped": True},
                metadata={
                    "band_gap_profile": dict(_band_gap_profile),
                    "min_head_wear_confidence": float(_band_gap_profile["min_head_wear_confidence"]),
                    "mid_gap_fraction_min": float(_band_gap_profile["mid_gap_fraction_min"]),
                    "side_gap_fraction_min": float(_band_gap_profile["side_gap_fraction_min"]),
                    "tape_head_clog_severity": float(_clog_severity),
                    "tape_head_clog_confidence": float(_clog_confidence),
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": effective_strength,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
            )

        instrument_tag: str = kwargs.get("instrument_tag", self.instrument_tag)
        sr = self._sample_rate
        # §2.36a: PhonemeTimeline formant-target for vowel-dominant segments
        self._current_phoneme_timeline = kwargs.get("phoneme_timeline")
        t_start = __import__("time").time()

        # §2.51 M/S: Reparatur auf Mid-Kanal voll, Side konservativ (Stereo-Kohärenz-Invariante).
        # Verboten: unabhängiges L/R mit gain- oder zeitvarianter Operation.
        if audio.ndim == 2:
            _sqrt2 = float(np.sqrt(2.0))
            mid = (audio[:, 0] + audio[:, 1]) / _sqrt2
            side = (audio[:, 0] - audio[:, 1]) / _sqrt2

            # Full repair on Mid channel
            mid_repaired = self._process_channel(mid, sr, instrument_tag)
            mid_repaired = self._mrsa_gain_refinement(
                mid.astype(np.float64), mid_repaired.astype(np.float64), sr
            ).astype(np.float32)

            # Conservative repair on Side channel: gap_fraction_min=0.95 (vs. 0.80 default)
            # → nur Lücken reparieren, die in ≥95 % der Frames leer sind (robusterer Befund nötig)
            side_repaired = self._process_channel(
                side,
                sr,
                instrument_tag,
                gap_fraction_min=_band_gap_profile["side_gap_fraction_min"],
            )
            side_repaired = self._mrsa_gain_refinement(
                side.astype(np.float64), side_repaired.astype(np.float64), sr
            ).astype(np.float32)

            # M/S decode back to L/R
            left = (mid_repaired + side_repaired) / _sqrt2
            right = (mid_repaired - side_repaired) / _sqrt2
            out = np.stack([left.astype(np.float32), right.astype(np.float32)], axis=1)
        else:
            out = self._process_channel(
                audio,
                sr,
                instrument_tag,
                gap_fraction_min=_band_gap_profile["mid_gap_fraction_min"],
            )
            # MRSA post-processing: zone-specific gain refinement + PGHI
            out = self._mrsa_gain_refinement(audio.astype(np.float64), out.astype(np.float64), sr).astype(np.float32)

        if 0.0 < effective_strength < 1.0:
            out = audio + effective_strength * (out - audio)

        _locality_profile, _locality_coverage = self._build_locality_profile(
            n_samples=int(out.shape[0]),
            sample_rate=sample_rate,
            defect_locations=kwargs.get("defect_locations"),
            event_metadata=kwargs.get("defect_event_metadata"),
            protected_zones=self._collect_protected_zones(kwargs),
        )
        if _locality_profile.size > 0:
            if out.ndim == 2:
                out = audio + _locality_profile[:, np.newaxis] * (out - audio)
            else:
                out = audio + _locality_profile * (out - audio)

        # NaN/Inf-Guard (§3.1)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, -1.0, 1.0).astype(np.float32)

        # §2.46f NPA-Guard: Atemgeräusche/Vibrato in reparierten Bändern nicht überschreiben.
        # §2.46e Hallucination-Guard: Additive Spektral-Synthese darf kein neues Material einbringen.
        try:
            from backend.core.dsp.hallucination_guard import check_hallucination as _check_hg56
            from backend.core.natural_performance_detector import get_natural_performance_detector

            _mono56 = audio.mean(axis=-1) if audio.ndim == 2 else audio
            n_samples56 = _mono56.shape[0]
            # §2.46f NPA-Guard
            try:
                _npa_mask56 = (
                    get_natural_performance_detector()
                    .detect(_mono56, sample_rate)
                    .get_protected_mask(n_samples56, sample_rate)
                )
                if _npa_mask56 is not None and _npa_mask56.any():
                    if out.ndim == 2:
                        out[_npa_mask56, :] = audio[_npa_mask56, :]
                    else:
                        out[_npa_mask56] = audio[_npa_mask56]
            except Exception as _npa56_exc:
                logger.debug("§2.46f Verarbeitungsschritt56 NPA-Guard (nicht blockierend): %s", _npa56_exc)
            # §2.46e Hallucination-Guard (Restoration-Modus only)
            try:
                _mode56 = str(kwargs.get("mode", "restoration")).lower()
                if "studio" not in _mode56:
                    _bw_cap56 = float(_band_gap_profile.get("bw_cap_hz", 22050.0))
                    _mono_out56 = out.mean(axis=-1) if out.ndim == 2 else out
                    _hg_result56 = _check_hg56(
                        _mono56,
                        _mono_out56,
                        sr=sample_rate,
                        mode=_mode56,
                        material_bw_ceiling_hz=_bw_cap56,
                        bw_extension_context=True,
                    )
                    if _hg_result56.requires_rollback:
                        logger.debug(
                            "§2.46e Verarbeitungsschritt56 Hallucination rollback: spectral_novelty=%.3f",
                            _hg_result56.spectral_novelty,
                        )
                        out = audio.copy()
                    if _hg_result56.score_penalty > 0:
                        logger.info(
                            "§2.46e Verarbeitungsschritt56 Wert_penalty=%.1f (spectral_novelty=%.3f)",
                            _hg_result56.score_penalty,
                            _hg_result56.spectral_novelty,
                        )
            except Exception as _hg56_exc:
                logger.debug("§2.46e Verarbeitungsschritt56 Hallucination-Guard (nicht blockierend): %s", _hg56_exc)
        except Exception as _guard56_exc:
            logger.debug("§2.46f/§2.46e Verarbeitungsschritt56 guards (nicht blockierend): %s", _guard56_exc)

        elapsed = __import__("time").time() - t_start
        logger.info(
            "⚙️ SpectralBandGapRepair: %.2fs | instrument=%s | confidence=%.2f",
            elapsed,
            instrument_tag,
            confidence,
        )
        return create_phase_result(
            out,
            modifications={"spectral_band_gaps_repaired": True, "instrument_tag": instrument_tag},
            metadata={
                "band_gap_profile": dict(_band_gap_profile),
                "min_head_wear_confidence": float(_band_gap_profile["min_head_wear_confidence"]),
                "mid_gap_fraction_min": float(_band_gap_profile["mid_gap_fraction_min"]),
                "side_gap_fraction_min": float(_band_gap_profile["side_gap_fraction_min"]),
                "tape_head_clog_severity": float(_clog_severity),
                "tape_head_clog_confidence": float(_clog_confidence),
                "repair_locality_coverage": float(_locality_coverage),
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": effective_strength,
                "execution_time_seconds": elapsed,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
        )

    def _mrsa_gain_refinement(self, audio_in: np.ndarray, audio_out: np.ndarray, sr: int) -> np.ndarray:
        """MRSA zone-specific gain refinement with PGHI reconstruction.

        For each MRSA zone, computes the input→output gain ratio at zone-specific
        resolution, blends zones via Hanning crossfades, and reconstructs via PGHI
        (fallback: iSTFT).  Post-processing step applied after _process_channel().

        Algorithm mirrors Phase 06 _mrsa_gain_refinement (post-processing pattern):
            1. Reference STFT of both audio_in and audio_out (win=2048)
            2. Baseline gain G_ref = |Zxx_out| / (|Zxx_in| + eps)
            3. For each zone: zone STFT → G_zone = |Zxx_out_z| / (|Zxx_in_z| + eps)
            4. Blend G_zone into G_ref via Hanning crossfade mask in freq domain
            5. Apply blended gain to audio_in STFT → PGHI reconstruct

        Args:
            audio_in:  Original channel audio (mono, float64).
            audio_out: Channel processed by _process_channel() (mono, float64).
            sr:        Sample rate (48000 Hz).

        Returns:
            Refined output audio (mono, float64, same length as audio_in).
        """
        REF_WIN = 2048
        REF_HOP = 512
        nyq = sr / 2.0
        n = len(audio_in)

        try:
            _, _, Zxx_in = signal.stft(audio_in, sr, nperseg=REF_WIN, noverlap=REF_WIN - REF_HOP)
            _, _, Zxx_out = signal.stft(audio_out, sr, nperseg=REF_WIN, noverlap=REF_WIN - REF_HOP)
        except Exception as e:
            logger.warning(
                "Verarbeitungsschritt_56_spectral_band_gap_repair.py::_mrsa_gain_refinement Ersatzpfad: %s", e
            )
            return audio_out

        n_freq = Zxx_in.shape[0]
        freqs_ref = np.linspace(0.0, nyq, n_freq)

        mag_in_ref = np.abs(Zxx_in) + 1e-8
        mag_out_ref = np.abs(Zxx_out)
        G_blend = mag_out_ref / mag_in_ref  # start with reference gain

        for _zone_name, win_z, hop_z, f_lo, f_hi in self._MRSA_ZONES:
            f_lo_z = min(float(f_lo), nyq)
            f_hi_z = min(float(f_hi), nyq)
            if f_lo_z >= nyq:
                continue

            try:
                _, _, Zxx_in_z = signal.stft(audio_in, sr, nperseg=win_z, noverlap=win_z - hop_z)
                _, _, Zxx_out_z = signal.stft(audio_out, sr, nperseg=win_z, noverlap=win_z - hop_z)
            except Exception:
                logger.debug("_mrsa_gain_refinement: silent except suppressed", exc_info=True)
                continue

            n_freq_z = Zxx_in_z.shape[0]
            n_t_z = Zxx_in_z.shape[1]
            G_zone_z = np.abs(Zxx_out_z) / (np.abs(Zxx_in_z) + 1e-8)  # [n_freq_z, n_t_z]

            # Resample G_zone to reference STFT grid
            freqs_z = np.linspace(0.0, nyq, n_freq_z)
            n_t_ref = G_blend.shape[1]
            G_zone_ref = np.zeros_like(G_blend)
            for tf in range(n_t_ref):
                # Map time frame
                t_z = min(int(round(tf * n_t_z / max(n_t_ref, 1))), n_t_z - 1)
                G_zone_ref[:, tf] = np.interp(freqs_ref, freqs_z, G_zone_z[:, t_z])

            # Hanning crossfade frequency mask for this zone
            bw = self._MRSA_CROSSFADE_BW_HZ
            f_mask = np.zeros(n_freq)
            for k, fk in enumerate(freqs_ref):
                if fk <= f_lo_z - bw or fk >= f_hi_z + bw:
                    f_mask[k] = 0.0
                elif f_lo_z - bw < fk < f_lo_z + bw:
                    f_mask[k] = 0.5 * (1.0 + np.cos(np.pi * (f_lo_z - fk) / bw))
                elif f_hi_z - bw < fk < f_hi_z + bw:
                    f_mask[k] = 0.5 * (1.0 + np.cos(np.pi * (fk - f_hi_z) / bw))
                else:
                    f_mask[k] = 1.0

            f_mask_2d = f_mask[:, np.newaxis]
            G_blend = f_mask_2d * G_zone_ref + (1.0 - f_mask_2d) * G_blend

        # Apply blended gain to input STFT → reconstruct
        G_blend = np.clip(G_blend, 0.0, 50.0)
        Zxx_refined = Zxx_in * G_blend

        # Direct ISTFT — Zxx_refined retains phase info from input STFT.
        # ISTFT is semantically correct and 50-100× faster than PGHI.
        try:
            _, result = signal.istft(
                np.asarray(Zxx_refined, dtype=np.complex64), sr, nperseg=REF_WIN, noverlap=REF_WIN - REF_HOP
            )
        except Exception as e:
            logger.warning("Verarbeitungsschritt_56_spectral_band_gap_repair.py::unbekannter Ersatzpfad: %s", e)
            return audio_out

        if len(result) < n:
            result = np.pad(result, (0, n - len(result)))
        result = result[:n]

        return np.clip(np.nan_to_num(result), -1.0, 1.0)  # type: ignore[no-any-return]

    def _process_channel(
        self,
        mono: np.ndarray,
        sr: int,
        instrument_tag: str,
        gap_fraction_min: float | None = None,
    ) -> np.ndarray:
        """Verarbeitet einen Kanal (Mono-Array).

        Args:
            gap_fraction_min: Override für _GAP_FRACTION_MIN (None = Standardwert 0.80).
                              Höherer Wert = konservativer (z.B. 0.95 für Side-Kanal, §2.51).
        """
        n_fft = self.n_fft
        hop = self.hop_length

        # STFT
        np.fft.rfft(
            np.pad(mono.astype(np.float32), (n_fft // 2, n_fft // 2), mode="reflect")[: len(mono) + n_fft - 1],
        )

        # Korrekte STFT als Frames
        win = np.hanning(n_fft).astype(np.float32)
        frames = []
        for i in range(0, max(1, len(mono) - n_fft + 1), hop):
            frame = mono[i : i + n_fft].astype(np.float32)
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            frames.append(np.fft.rfft(frame * win))
        if not frames:
            return mono

        stft_frames = np.array(frames).T  # [n_bins × n_frames]
        stft_mag = np.abs(stft_frames).astype(np.float32)
        stft_phase = np.angle(stft_frames).astype(np.float32)

        # Lücken detektieren — optional konservativer für Side-Kanal (§2.51)
        gaps = _detect_band_gaps(stft_mag, sr, n_fft, gap_fraction_min=gap_fraction_min)
        if not gaps:
            return mono

        # f₀ schätzen (für harmonische Interpolation)
        f0 = _estimate_f0(mono, sr)
        if f0 is None or f0 < 20.0:
            f0 = 220.0  # Fallback: A3

        # Jede Lücke reparieren
        for gap in gaps:
            gap_low, gap_high = gap
            logger.info(
                "SpectralBandGapRepair: Lücke %d–%d Bins (%.0f–%.0f Hz) wird repariert",
                gap_low,
                gap_high,
                gap_low * sr / n_fft,
                gap_high * sr / n_fft,
            )

            # Harmonische Interpolation
            stft_mag, stft_phase = _harmonic_interpolate_gap(stft_mag, stft_phase, gap, f0, sr, n_fft, instrument_tag)

            # §2.36a Formant-Boost: wenn vowel_stressed-Segment dominant, hebe F1/F2-Partials ×1.15
            _ptl_56 = getattr(self, "_current_phoneme_timeline", None)
            if _ptl_56 is not None:
                try:
                    _dur_s = mono.shape[-1] / sr
                    _f_target = _ptl_56.formant_target_for_range(0.0, _dur_s)
                    if _f_target is not None:
                        _freq_per_bin = sr / n_fft
                        for _f_hz in _f_target:
                            _b_center = round(float(_f_hz) / _freq_per_bin)
                            for _db in range(-2, 3):
                                _b = _b_center + _db
                                if gap_low <= _b < gap_high and 0 <= _b < stft_mag.shape[0]:
                                    _gauss = math.exp(-0.5 * (_db / 1.5) ** 2)
                                    stft_mag[_b] = stft_mag[_b] * (1.0 + 0.15 * _gauss)
                except Exception as _formant_exc:
                    logger.debug("Formant-guided band gap repair fehlgeschlagen, using standard fill: %s", _formant_exc)

            # Spectral Flatness prüfen
            gap_region = stft_mag[gap_low:gap_high, :]
            flatness = _spectral_flatness(gap_region.flatten())
            if flatness > _MAX_SPECTRAL_FLATNESS:
                logger.debug("Flatness=%.3f > %.2f → NMF-β Verfeinerung", flatness, _MAX_SPECTRAL_FLATNESS)
                stft_mag = _nmf_beta_refine(stft_mag, gap)

        # PGHI Phase Reconstruction
        stft_phase = _pghi_phase_reconstruction(stft_mag, n_fft, hop)

        # ISTFT via OLA
        stft_complex = stft_mag * np.exp(1j * stft_phase)
        audio_out = np.zeros((stft_complex.shape[1] - 1) * hop + n_fft, dtype=np.float32)
        win_sum = np.zeros_like(audio_out)

        for i, t in enumerate(range(stft_complex.shape[1])):
            ifft_frame = np.real(np.fft.irfft(stft_complex[:, t], n=n_fft)).astype(np.float32)
            start = i * hop
            end = start + n_fft
            if end > len(audio_out):
                trim = end - len(audio_out)
                audio_out[start:] += ifft_frame[: n_fft - trim] * win[: n_fft - trim]
                win_sum[start:] += win[: n_fft - trim] ** 2
            else:
                audio_out[start:end] += ifft_frame * win
                win_sum[start:end] += win**2

        win_sum = np.maximum(win_sum, 1e-8)
        audio_out /= win_sum

        # Länge an Original anpassen
        if len(audio_out) > len(mono):
            audio_out = audio_out[: len(mono)]
        elif len(audio_out) < len(mono):
            audio_out = np.pad(audio_out, (0, len(mono) - len(audio_out)))

        # Adaptiver Blend: breite/verlässliche Lücken stärker, schmale Lücken konservativer.
        if gaps:
            _widths = [max(0, gh - gl) for gl, gh in gaps]
            _avg_width = float(np.mean(_widths)) if _widths else 0.0
            _conf_norm = float(np.clip((_avg_width / max(stft_mag.shape[0], 1)) * 6.0, 0.0, 1.0))
        else:
            _conf_norm = 0.0
        blend = float(np.clip(0.65 + 0.30 * _conf_norm, 0.65, 0.95))
        audio_out = blend * audio_out + (1.0 - blend) * mono.astype(np.float32)

        return audio_out.astype(np.float32)  # type: ignore[no-any-return]
