"""§v10.701 D3 / §18.1 PresenceEmbedding — Wahrnehmungsmetrik für menschliche Präsenz.

Misst die Distanz zwischen „Aufnahme" und „Live-Präsenz". Löst das
„43→43"-Paradox: technische Metriken sagen keine Verbesserung, perzeptuelle
Metriken sagen exzellent — PresenceEmbedding misst, was tatsächlich verbessert
wurde: **die menschliche Anwesenheit in der Aufnahme.**

PresenceScore = f(
    vocal_formant_coherence,   # Wie "echt" klingt die Stimme?
    transient_immediacy,       # Wie direkt sind die Transienten?
    room_tone_continuity,      # Atmet der Raum natürlich?
    microdynamic_liveliness,   # Lebt die Dynamik?
    spectral_air_authenticity  # Ist die Luft echt oder synthetisch?
)

Integration: Läuft NACH allen Restaurierungsphasen, VOR dem Export.
Ersetzt NICHT die technischen Metriken — ergänzt sie.
Schwellwert: PresenceScore ≥ 0.70 für „hörbare Verbesserung".

Status: ✅ Neu implementiert §v10.701 D3 / §G90
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ── Helper-Funktionen ────────────────────────────────────────────────


def _safe_rms(x: np.ndarray) -> float:
    """NaN-sichere RMS-Berechnung."""
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.nan_to_num(x, nan=0.0) ** 2)))


def _crest_factor(x: np.ndarray, window: int = 1920) -> float:
    """Crest-Faktor (Peak/RMS) in gleitenden Fenstern — Microdynamic-Liveliness."""
    if len(x) < window:
        return 0.5
    peaks = []
    rms_vals = []
    for i in range(0, len(x) - window, window // 2):
        seg = x[i : i + window]
        p = float(np.max(np.abs(seg)))
        r = _safe_rms(seg)
        if r > 1e-15:
            peaks.append(p)
            rms_vals.append(r)
    if not peaks or not rms_vals:
        return 0.5
    crest_ratios = [p / r for p, r in zip(peaks, rms_vals)]
    # Normalisiert auf [0, 1]: hoher Crest-Faktor → lebendige Dynamik
    mean_cf = float(np.mean(crest_ratios))
    return float(np.clip(mean_cf / 20.0, 0.0, 1.0))


def _onset_strength(x: np.ndarray, sr: int) -> float:
    """Onset-Stärke-Verteilung — Transient Immediacy."""
    if len(x) < 480:
        return 0.5
    # Simple onset detection via energy difference
    frame_len = 256
    hop = 128
    energies = []
    for i in range(0, len(x) - frame_len, hop):
        seg = x[i : i + frame_len]
        energies.append(float(np.sum(seg**2)))
    if len(energies) < 2:
        return 0.5
    # Onsets are frames where energy jumps significantly
    diffs = np.diff(energies)
    onsets = np.where(diffs > np.mean(diffs) + np.std(diffs))[0]
    if len(onsets) == 0:
        return 0.3
    # Measure how "direct" the onsets are (sharp attack vs gradual swell)
    onset_strengths = []
    for idx in onsets[:50]:  # Limit to first 50 onsets
        if idx + 1 < len(energies):
            strength = float(diffs[idx] / (energies[idx] + 1e-15))
            onset_strengths.append(strength)
    if not onset_strengths:
        return 0.5
    # High values → sharp, direct attacks → high immediacy
    mean_strength = float(np.mean(onset_strengths))
    return float(np.clip(mean_strength / 5.0, 0.0, 1.0))


def _noise_floor_variance(x: np.ndarray, sr: int) -> float:
    """Varianz des Rauschbodens über die Zeit — Room Tone Continuity."""
    if len(x) < 4800:
        return 0.5
    # Extract noise floor from quiet segments (bottom 10% energy)
    frame_len = 1024
    hop = 512
    energies = []
    for i in range(0, len(x) - frame_len, hop):
        seg = x[i : i + frame_len]
        energies.append(float(np.sum(seg**2)))
    if not energies:
        return 0.5
    # Bottom 10% energy frames → noise floor candidates
    threshold = float(np.percentile(energies, 10))
    noise_frames = [e for e in energies if e <= threshold]
    if len(noise_frames) < 3:
        return 0.5
    # Low variance → continuous room tone → natural
    var = float(np.var(noise_frames))
    mean_energy = float(np.mean(noise_frames))
    cv = (var**0.5) / (mean_energy + 1e-15)  # Coefficient of variation
    # Invert: low CV → high continuity score
    return float(np.clip(1.0 - cv, 0.0, 1.0))


def _hf_envelope_correlation(x: np.ndarray, sr: int) -> float:
    """Korrelation der HF-Hüllkurve (>10 kHz) — Spectral Air Authenticity."""
    if len(x) < 4800:
        return 0.5
    # Simple high-frequency energy envelope via block-wise FFT
    n_fft = 2048
    hop = 512
    window = np.hanning(n_fft)

    blocks = []
    for i in range(0, min(len(x), 96000) - n_fft, hop):
        seg = x[i : i + n_fft] * window
        fft_mag = np.abs(np.fft.rfft(seg))
        blocks.append(fft_mag)

    if not blocks:
        return 0.5

    Zxx = np.array(blocks).T  # shape: (freq_bins, num_blocks)

    if Zxx.size == 0:
        return 0.5

    # Find >10 kHz band (approximate for 48kHz SR)
    bin_10k = int(10000 * n_fft / sr)
    hf_band = Zxx[bin_10k:, :] if Zxx.shape[0] > bin_10k else Zxx[-20:, :]
    # Compute envelope (RMS per frame)
    envelope = np.sqrt(np.mean(hf_band**2, axis=0))
    if len(envelope) < 2:
        return 0.5
    # Correlation of adjacent frames → natural air has smooth envelope
    corr = float(np.corrcoef(envelope[:-1], envelope[1:])[0, 1])
    if not np.isfinite(corr):
        return 0.5
    return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))


def _vocal_formant_coherence(x: np.ndarray, sr: int) -> float:
    """Formant-Plausibilität via spektrale Peaks — Vocal Formant Coherence."""
    if len(x) < 4800:
        return 0.5
    # Simple formant detection via spectral peaks in vowel-like segments
    n_fft = 2048
    hop = 512
    try:
        from scipy.signal import stft as _stft

        _, _, Zxx = _stft(x[: min(len(x), 96000)], nperseg=n_fft, noverlap=hop // 2)
    except ImportError:
        block_size = n_fft
        overlap = hop
        blocks = []
        for i in range(0, len(x) - block_size, overlap):
            seg = x[i : i + block_size] * np.hanning(block_size)
            fft_mag = np.abs(np.fft.rfft(seg))
            blocks.append(fft_mag)
        if not blocks:
            return 0.5
        Zxx = np.array(blocks).T

    if Zxx is None or Zxx.size == 0:
        return 0.5

    # Look for formant-like peaks (F1 ~500Hz, F2 ~1500Hz, F3 ~2500Hz)
    bin_f1 = int(500 * n_fft / sr)
    bin_f2 = int(1500 * n_fft / sr)
    bin_f3 = int(2500 * n_fft / sr)

    # Check for consistent peaks across frames
    coherence_scores = []
    for frame_idx in range(min(Zxx.shape[1], 20)):
        spectrum = Zxx[:, frame_idx]
        if len(spectrum) > max(bin_f1, bin_f2, bin_f3):
            # Look for local maxima near formant frequencies
            peaks_found = 0
            for target_bin in [bin_f1, bin_f2, bin_f3]:
                window = slice(max(0, target_bin - 5), min(len(spectrum), target_bin + 6))
                if np.max(spectrum[window]) > np.mean(spectrum) * 1.2:
                    peaks_found += 1
            coherence_scores.append(peaks_found / 3.0)

    if not coherence_scores:
        return 0.5
    return float(np.clip(np.mean(coherence_scores), 0.0, 1.0))


# ── Result Dataclass ────────────────────────────────────────────────


@dataclass
class PresenceScoreResult:
    """Ergebnis der PresenceEmbedding-Bewertung (§G90).

    Attributes:
        overall:                 Gesamtscore [0, 1] — ≥ 0.70 = hörbare Verbesserung
        is_hearable_improvement: True wenn overall ≥ 0.70
        vocal_formant_coherence: Stimme-Echtheit [0, 1]
        transient_immediacy:     Transienten-Direktheit [0, 1]
        room_tone_continuity:    Raumton-Kontinuität [0, 1]
        microdynamic_liveliness: Dynamik-Lebendigkeit [0, 1]
        spectral_air_authenticity: Höhen-Luft-Echtheit [0, 1]
        component_scores:        Alle Sub-Scores für Debugging/Reporting
    """

    overall: float = 0.5
    is_hearable_improvement: bool = False
    vocal_formant_coherence: float = 0.5
    transient_immediacy: float = 0.5
    room_tone_continuity: float = 0.5
    microdynamic_liveliness: float = 0.5
    spectral_air_authenticity: float = 0.5
    component_scores: dict[str, float] = field(default_factory=dict)

    @property
    def presence_score(self) -> float:
        """Alias für overall (rückwärtskompatibel)."""
        return self.overall

    def passes_threshold(self, min_score: float = 0.70) -> bool:
        """Prüft ob der PresenceScore den Schwellwert für hörbare Verbesserung erreicht."""
        return self.overall >= min_score


# ── Hauptklasse ─────────────────────────────────────────────────────


class PresenceEmbedding:
    """Berechnet den PresenceScore — perzeptuelle Metrik für menschliche Präsenz.

    §18.1 / §G90: Jeder Export MUSS einen PresenceScore berechnen und im Quality
    Report ausweisen. PresenceScore ≥ 0.70 definiert „hörbare Verbesserung".

    Nutzung::
        pe = PresenceEmbedding()
        result = pe.score(audio, sr=48000)
        if not result.passes_threshold():
            logger.warning("PresenceScore %.2f < 0.70 — keine hörbare Verbesserung", result.presence_score)
    """

    # Gewichtung der Sub-Komponenten (summiert zu 1.0)
    _WEIGHTS: dict[str, float] = {
        "vocal_formant_coherence": 0.25,
        "transient_immediacy": 0.20,
        "room_tone_continuity": 0.15,
        "microdynamic_liveliness": 0.20,
        "spectral_air_authenticity": 0.20,
    }

    def __init__(self) -> None:
        self._last_result: PresenceScoreResult | None = None

    def compute(self, audio: np.ndarray, sample_rate: int = 48000) -> PresenceScoreResult:
        """Berechnet den PresenceScore für ein Audio-Signal.

        Args:
            audio: Mono oder Stereo Audio (float32, normiert auf [-1, 1])
            sample_rate: Sample-Rate in Hz

        Returns:
            PresenceScoreResult mit Gesamtscore und Sub-Komponenten
        """
        # Normalisieren auf Mono für Analyse
        if audio.ndim == 2 and audio.shape[0] <= 8:
            mono = audio.mean(axis=0)
        else:
            mono = np.asarray(audio, dtype=np.float32).flatten()

        mono = np.nan_to_num(mono, nan=0.0, posinf=1.0, neginf=-1.0)

        # Sub-Scores berechnen
        vfc = _vocal_formant_coherence(mono, sample_rate)
        ti = _onset_strength(mono, sample_rate)
        rtc = _noise_floor_variance(mono, sample_rate)
        mdl = _crest_factor(mono)
        saa = _hf_envelope_correlation(mono, sample_rate)

        # Gewichtete Summe
        weights = self._WEIGHTS
        overall = (
            vfc * weights["vocal_formant_coherence"]
            + ti * weights["transient_immediacy"]
            + rtc * weights["room_tone_continuity"]
            + mdl * weights["microdynamic_liveliness"]
            + saa * weights["spectral_air_authenticity"]
        )

        overall = float(np.clip(overall, 0.0, 1.0))
        is_hearable = overall >= 0.70

        result = PresenceScoreResult(
            overall=overall,
            is_hearable_improvement=is_hearable,
            vocal_formant_coherence=vfc,
            transient_immediacy=ti,
            room_tone_continuity=rtc,
            microdynamic_liveliness=mdl,
            spectral_air_authenticity=saa,
            component_scores={
                "vocal_formant_coherence": round(vfc, 4),
                "transient_immediacy": round(ti, 4),
                "room_tone_continuity": round(rtc, 4),
                "microdynamic_liveliness": round(mdl, 4),
                "spectral_air_authenticity": round(saa, 4),
            },
        )

        self._last_result = result
        logger.info(
            "§G90 PresenceEmbedding: overall=%.2f (hearable=%s) VFC=%.2f TI=%.2f RTC=%.2f MDL=%.2f SAA=%.2f",
            overall,
            is_hearable,
            vfc,
            ti,
            rtc,
            mdl,
            saa,
        )
        return result

    def score(self, audio: np.ndarray, sr: int = 48000) -> PresenceScoreResult:
        """Alias für compute (rückwärtskompatibel)."""
        return self.compute(audio, sample_rate=sr)

    def delta(self, before: np.ndarray, after: np.ndarray, sr: int = 48000) -> float:
        """Berechnet die Präsenz-Verbesserung (nach - vor).

        Positive Werte → Präsenz gestiegen. Negative Werte → Präsenz verloren.
        """
        score_before = self.score(before, sr).presence_score
        score_after = self.score(after, sr).presence_score
        delta_val = float(score_after - score_before)
        logger.info(
            "§G90 PresenceDelta: %.2f → %.2f (Δ=%.3f)",
            score_before,
            score_after,
            delta_val,
        )
        return delta_val


# ── Singleton-Factory (§3.x Double-Checked Locking) ───────────────

_presence_instance: PresenceEmbedding | None = None
_presence_lock = threading.Lock()


def get_presence_embedding() -> PresenceEmbedding:
    """Thread-safe Singleton für PresenceEmbedding.

    Returns:
        PresenceEmbedding-Instanz (lazy-initialized)
    """
    global _presence_instance  # pylint: disable=global-statement

    if _presence_instance is not None:
        return _presence_instance

    with _presence_lock:
        if _presence_instance is None:
            _presence_instance = PresenceEmbedding()
            logger.debug("§G90 PresenceEmbedding Singleton initialisiert")
        return _presence_instance
