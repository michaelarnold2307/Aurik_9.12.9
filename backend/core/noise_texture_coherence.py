"""Noise-Texture-Coherence-Guard (§4.7, v10.0.0).

Misst, ob die spektrale Form des Restrauschens nach Denoising zum
erkannten Trägerprofil passt (z.B. rosa für Vinyl, Brown+HF-Hiss für Tape).

Singleton: ``get_noise_texture_coherence_guard()``
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ── §4.7 Trägerprofil-Referenz (10 Material-Profile) ─────────────────────
# Format: (label, spectral_slope_db_oct, optional_hf_hiss_peak_hz)
# spectral_slope_db_oct: Steigung der log-PSD über log-Frequenz
#   0.0 = weiß, -1.0 = rosa (-3 dB/oct ≈ -1.0 Slope in log-log),
#   Negative = Energy fällt mit Frequenz (typisch für analoge Medien).

_CARRIER_NOISE_PROFILES: dict[str, tuple[float, float | None]] = {
    "vinyl": (-1.0, None),  # Rosa / Pink ≈ −3 dB/oct
    "shellac": (-0.83, None),  # Flacher als Vinyl ≈ −2.5 dB/oct
    "wax_cylinder": (-0.67, None),  # Noch flacher ≈ −2 dB/oct
    "wire_recording": (-0.83, None),  # Ähnlich Shellac
    "lacquer_disc": (-0.83, None),  # Ähnlich Shellac
    "tape": (-1.5, 8000),  # Brown + HF-Hiss-Buckel ≈ −4.5 + Peak@8kHz
    "reel_tape": (-1.33, 10000),  # Weniger HF-Hiss ≈ −4 dB/oct + Peak@10kHz
    "cassette": (-1.5, 6000),  # Stark ≈ −4.5 + Peak@6kHz
    "cd_digital": (0.0, None),  # Weiß / Flat
    "unknown": (-0.5, None),  # Konservativ: leicht rosa
}

# Für alle nicht gelisteten digitalen Materialien → Weiß
_DIGITAL_MATERIALS = frozenset({"dat", "minidisc", "mp3_low", "mp3_high", "aac", "streaming"})


@dataclass
class NoiseTextureResult:
    """Ergebnis der Rauschtextur-Kohärenz-Messung."""

    coherence: float
    """Korrelation zur Referenz-Textur [0, 1]."""

    material_type: str
    """Erkanntes Material."""

    reference_slope: float
    """Referenz-Slope des Trägerprofils."""

    measured_slope: float
    """Gemessener Slope des Restrauschens."""

    is_compliant: bool
    """True wenn coherence ≥ 0.80 (Restoration-Pflicht)."""

    min_required_coherence: float = 0.80
    """Aktive End-of-Pipeline-Mindestschwelle (mode/material/duration-adaptiv)."""


def _generate_reference_profile(
    freqs: np.ndarray,
    slope: float,
    hf_peak_hz: float | None,
) -> np.ndarray:
    """Erzeuge ein normiertes Referenz-Rauschprofil im log-Frequenzraum."""
    # Avoid log(0) — start from first positive frequency
    safe_freqs = np.maximum(freqs, 1.0)

    # Base slope: PSD ∝ f^slope (in log-log: linear with given slope)
    ref = slope * np.log10(safe_freqs / safe_freqs[0])

    # Optional HF-Hiss peak (Gaussian bump in log-frequency space)
    if hf_peak_hz is not None and hf_peak_hz > 0:
        log_center = np.log10(hf_peak_hz)
        log_freqs = np.log10(safe_freqs)
        sigma = 0.15  # ~1/3 Oktave Breite
        bump = 3.0 * np.exp(-0.5 * ((log_freqs - log_center) / sigma) ** 2)
        ref = ref + bump

    # Normalize to zero mean, unit variance
    mean = np.mean(ref)
    std = np.std(ref)
    if std > 1e-10:
        ref = (ref - mean) / std
    else:
        ref = ref - mean

    return np.asarray(ref, dtype=np.float64)  # type: ignore[no-any-return]


def compute_noise_texture_coherence(
    residual_noise: np.ndarray,
    sr: int,
    material_type: str,
) -> NoiseTextureResult:
    """§4.7 — Berechne die Kohärenz zwischen Restrauschen und Trägerprofil.

    Args:
        residual_noise: Restrauschen (restored - clean_estimate oder noise_floor_segment).
        sr: Sample-Rate.
        material_type: Erkannter Materialtyp.

    Returns:
        NoiseTextureResult mit coherence [0, 1].
    """
    # Lookup reference profile
    if material_type in _DIGITAL_MATERIALS:
        slope, hf_peak = 0.0, None
    else:
        slope, hf_peak = _CARRIER_NOISE_PROFILES.get(material_type, (-0.5, None))

    # Mono-Downmix if stereo
    if residual_noise.ndim == 2:
        residual_noise = np.mean(residual_noise, axis=-1)

    n_samples = len(residual_noise)
    if n_samples < 1024:
        return NoiseTextureResult(
            coherence=1.0,  # Too short → pass
            material_type=material_type,
            reference_slope=slope,
            measured_slope=0.0,
            is_compliant=True,
        )

    # PSD via Welch
    nperseg = min(4096, n_samples // 2)
    try:
        from scipy.signal import welch

        freqs, psd = welch(residual_noise, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
    except Exception as exc:
        logger.debug("Welch PSD fehlgeschlagen (nicht blockierend): %s", exc)
        return NoiseTextureResult(
            coherence=1.0,
            material_type=material_type,
            reference_slope=slope,
            measured_slope=0.0,
            is_compliant=True,
        )

    # Ignore DC bin and frequencies above Nyquist/2 (unreliable)
    valid = (freqs > 20) & (freqs < sr / 2)
    if np.sum(valid) < 10:
        return NoiseTextureResult(
            coherence=1.0,
            material_type=material_type,
            reference_slope=slope,
            measured_slope=0.0,
            is_compliant=True,
        )

    freqs_v = freqs[valid]
    psd_v = psd[valid]

    # Log-normalize PSD
    psd_log = 10.0 * np.log10(np.maximum(psd_v, 1e-20))
    psd_mean = np.mean(psd_log)
    psd_std = np.std(psd_log)
    if psd_std > 1e-10:
        psd_norm = (psd_log - psd_mean) / psd_std
    else:
        psd_norm = psd_log - psd_mean

    # Generate reference profile
    ref = _generate_reference_profile(freqs_v, slope, hf_peak)

    # Pearson correlation — guarded (§ VERBOTEN: np.corrcoef auf near-constant)
    dot = np.dot(psd_norm, ref)
    norm_a = np.linalg.norm(psd_norm)
    norm_b = np.linalg.norm(ref)
    eps = 1e-10
    coherence = float(dot / (norm_a * norm_b + eps))
    coherence = float(np.clip(coherence, 0.0, 1.0))

    # Measure actual slope (linear regression in log-log space)
    try:
        log_f = np.log10(freqs_v)
        coeffs = np.polyfit(log_f, psd_log, 1)
        measured_slope = float(coeffs[0]) / 3.0  # Convert dB/decade → dB/octave approx
    except Exception:
        measured_slope = 0.0

    return NoiseTextureResult(
        coherence=coherence,
        material_type=material_type,
        reference_slope=slope,
        measured_slope=measured_slope,
        min_required_coherence=0.80,
        is_compliant=coherence >= 0.80,
    )


def _resolve_end_of_pipeline_min_coherence(material_type: str, quality_mode: str, duration_s: float) -> float:
    """Liefert die adaptive End-of-Pipeline-Mindestschwelle für Noise-Texture-Coherence.

    Prinzip:
    - Studio 2026: keine harte Carrier-Texturpflicht (0.0).
        - Digitale Carrier: kein Mindestwert (0.0), da keine analoge Trägertextur
            erhalten werden muss.
    - Analoge Carrier: 0.80 als Langform-Ziel; für sehr kurze Snippets wird die
      Schwelle auf 0.35 abgesenkt und zwischen 3s..8s linear angehoben, um
      statistische Instabilität der Residual-PSD nicht als Qualitätsfehler zu werten.
    """
    _qm = str(quality_mode or "restoration").strip().lower()
    if _qm in {"studio_2026", "studio2026", "studio"}:
        return 0.0

    _mat = str(material_type or "unknown").strip().lower()
    _is_digital = _mat in _DIGITAL_MATERIALS or "digital" in _mat or "stream" in _mat or "mp3" in _mat or "aac" in _mat
    if _is_digital:
        return 0.0

    _dur = float(max(duration_s, 0.0))
    if _dur <= 3.0:
        return 0.35
    if _dur >= 8.0:
        return 0.80

    # Linear ramp 3s..8s: 0.35 → 0.80
    _t = (_dur - 3.0) / 5.0
    return float(0.35 + _t * (0.80 - 0.35))


class NoiseTextureCoherenceGuard:
    """§4.7 — Guard für Rauschtextur-Kohärenz nach subtraktiven Phasen.

    Integration:
        - Per-Phase (nach Subtraktiven): coherence < 0.60 → wet ×0.70
        - End-of-Pipeline: metadata["noise_texture_coherence"] setzen
    """

    def check(
        self,
        residual_audio: np.ndarray,
        material_type: str,
        sr: int,
    ) -> NoiseTextureResult:
        """Direkte Kohärenz-Prüfung auf Restrauschen (Alias für Legacy-Aufrufe).

        Args:
            residual_audio: Geschätztes Restrauschen (Residual)
            material_type: Erkanntes Material
            sr: Sample-Rate

        Returns:
            NoiseTextureResult mit coherence und weiteren Metriken
        """
        return compute_noise_texture_coherence(residual_audio, sr, material_type)

    def check_per_phase(
        self,
        audio_before: np.ndarray,
        audio_after: np.ndarray,
        sr: int,
        material_type: str,
    ) -> tuple[float, float]:
        """Per-Phase-Check nach subtraktiver Phase.

        Returns:
            (coherence, wet_multiplier): wet_multiplier < 1.0 bei Texturschwund.
        """
        # Restrauschen ≈ (before - after) — was entfernt wurde
        residual = audio_before - audio_after

        result = compute_noise_texture_coherence(residual, sr, material_type)

        if result.coherence < 0.60:
            wet_mult = 0.70
            logger.info(
                "§4.7 NoiseTexture: coherence=%.2f < 0.60 → wet ×0.70 (material=%s)",
                result.coherence,
                material_type,
            )
        elif result.coherence < 0.80:
            # §4.7-v10.0.0: Schwelle 0.60–0.80 → wet ×0.85 (bisher nur Warning ohne Wirkung).
            # Rauschtextur im 0.60–0.80-Band ist für sensible Hörer bereits hörbar inkohärent
            # (Vinyl klingt 'digital-flach'). Konservative Wet-Dämpfung erzwingt mehr Retention
            # des Carrier-Profils ohne die subtraktive Wirkung komplett zu neutralisieren.
            wet_mult = 0.85
            logger.info(
                "§4.7 NoiseTexture: coherence=%.2f in [0.60,0.80) → wet ×0.85 (material=%s)",
                result.coherence,
                material_type,
            )
        else:
            wet_mult = 1.0

        return result.coherence, wet_mult

    def check_end_of_pipeline(
        self,
        original_audio: np.ndarray,
        restored_audio: np.ndarray,
        sr: int,
        material_type: str,
        quality_mode: str = "restoration",
    ) -> NoiseTextureResult:
        """End-of-Pipeline-Check — Ergebnis für metadata.

        Args:
            original_audio: Degradiertes Original.
            restored_audio: Restauriertes Audio.
            sr: Sample-Rate.
            material_type: Erkanntes Material.
            quality_mode: 'restoration' oder 'studio_2026'.

        Returns:
            NoiseTextureResult für metadata + Export-Gate.
        """
        residual = original_audio - restored_audio
        result = compute_noise_texture_coherence(residual, sr, material_type)

        # Dauer robust bestimmen (unterstützt Mono, [N,2] und [2,N]).
        _arr = np.asarray(original_audio)
        if _arr.ndim == 2:
            _n_samples = int(max(_arr.shape))
        else:
            _n_samples = int(_arr.shape[0]) if _arr.size > 0 else 0
        _duration_s = float(_n_samples / max(int(sr), 1)) if _n_samples > 0 else 0.0

        _min_required = _resolve_end_of_pipeline_min_coherence(material_type, quality_mode, _duration_s)
        result = NoiseTextureResult(
            coherence=float(result.coherence),
            material_type=str(result.material_type),
            reference_slope=float(result.reference_slope),
            measured_slope=float(result.measured_slope),
            min_required_coherence=float(_min_required),
            is_compliant=bool(float(result.coherence) >= float(_min_required)),
        )

        if quality_mode == "restoration" and not result.is_compliant:
            logger.warning(
                "§4.7 NoiseTexture End-Gate: coherence=%.2f < %.2f (material=%s, duration=%.2fs) "
                "→ recommendation: reduce denoising aggressiveness",
                result.coherence,
                result.min_required_coherence,
                material_type,
                _duration_s,
            )

        return result


# ── Singleton ─────────────────────────────────────────────────────────────
_instance: NoiseTextureCoherenceGuard | None = None
_lock = threading.Lock()


def get_noise_texture_coherence_guard() -> NoiseTextureCoherenceGuard:
    """Thread-sicherer Singleton-Zugriff (§ Pflicht-Pattern)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = NoiseTextureCoherenceGuard()
    return _instance
